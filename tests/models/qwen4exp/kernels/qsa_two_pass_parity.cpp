// tests/models/qwen4exp/kernels/qsa_two_pass_parity.cpp - the two-pass QSA block scores (qsa_two_pass.hpp) against
// the 3xTF32 prompt scorer they replace and against a CPU reference in double (GPU, synthetic, no model).
//
// For prompt chunks of up to 256 queries at depths from 2K to 256K and several kinds of keys (normal, indexer-like,
// wide magnitudes, small integers with exact ties, near-duplicate and duplicate keys), qsa_block_topk over the
// two-pass row must return exactly the ids it returns over qsa_block_scores_tc's row, every scored-again block must
// hold its 3xTF32 score bit for bit, and every kept single-pass score must lie inside its bound (qsa_two_pass_check's
// counters).  A query's ids must not depend on the call it is in (the same queries in one call and split in two).
// Against the CPU reference (relu per head of q . k in double, summed) the test prints the largest absolute and
// relative error of both rows.
//
//   qsa_two_pass_parity --selftest   parity (exit 0 = identical everywhere); skipped (exit 0) where the 3xTF32 scorer
//                                    does not run
//   qsa_two_pass_parity --bench      per-call times and TFLOP/s of both scorers at 32K / 128K / 256K
#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_select.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_two_pass.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;
namespace tp = basalt::kernels::two_pass;

namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        std::printf("FAIL CUDA: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

enum Kind { kNormal, kIndexer, kWide, kSmall, kNear, kDup, kKinds };
const char* kKindName[kKinds] = {"normal", "indexer-like", "wide magnitudes", "small integers", "near-duplicate keys",
                                 "duplicate keys"};

struct Case { int64_t ctx, queries; Kind kind; };

// One synthetic chunk: pooled keys, the dead key, the queries and their step records (the chunk ends at ctx).
struct Chunk {
    int64_t max_blocks = 0, nq = 0;
    std::vector<float> pooled, dead, q;
    std::vector<int32_t> steps;
};

// stride: the score row's stride in blocks; 0 = the case's own (ctx / 4 + 2), else the engine's at a 262,144-token
// context (65,538), so qsa_block_topk takes the dispatch it takes in the engine
Chunk make_chunk(const Case& c, int64_t stride) {
    const auto s = k::qsa_real_shapes();
    const int64_t D = s.idx_dim, H = s.idx_n_head, own = c.ctx / s.idx_block + 2;
    Chunk ch;
    ch.max_blocks = stride > own ? stride : own;
    ch.nq = c.queries;
    std::mt19937 rng(77001 + (unsigned) c.ctx + 13 * (unsigned) c.queries + 1009 * (unsigned) c.kind);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> small(-3, 3), expo(-16, 16), coin(0, 9);
    ch.pooled.resize((size_t) (ch.max_blocks * D));
    ch.dead.resize((size_t) D);
    ch.q.resize((size_t) (ch.nq * H * D));
    auto unit = [&](float* v, int n, float norm) {
        double ss = 0;
        for (int i = 0; i < n; ++i) ss += (double) v[i] * v[i];
        const float f = ss > 0 ? (float) (norm / std::sqrt(ss)) : 0.f;
        for (int i = 0; i < n; ++i) v[i] *= f;
    };
    for (int64_t b = 0; b < ch.max_blocks; ++b) {
        float* kb = ch.pooled.data() + b * D;
        if (c.kind == kIndexer) {   // the mean of 4 RMS-normed cell keys, as the indexer pools them
            for (int i = 0; i < D; ++i) kb[i] = 0.f;
            for (int cell = 0; cell < 4; ++cell) {
                float cell_key[128];
                for (int i = 0; i < D; ++i) cell_key[i] = nd(rng);
                unit(cell_key, (int) D, std::sqrt((float) D));
                for (int i = 0; i < D; ++i) kb[i] += 0.25f * cell_key[i];
            }
        } else if (c.kind == kNear && b > 0 && coin(rng) < 5) {   // half the keys a hair from the one before
            for (int i = 0; i < D; ++i) kb[i] = kb[i - D] * (1.0f + 1e-6f * nd(rng));
        } else if (c.kind == kDup && b >= 8 && coin(rng) < 5) {   // half the keys copies of an earlier one: exact ties
            const int64_t src = (int64_t) (rng() % (uint32_t) b);
            std::memcpy(kb, ch.pooled.data() + src * D, (size_t) D * 4);
        } else {
            for (int i = 0; i < D; ++i) {
                if (c.kind == kWide) kb[i] = coin(rng) == 0 ? 0.f : std::ldexp(nd(rng), expo(rng));
                else if (c.kind == kSmall) kb[i] = (float) small(rng);
                else kb[i] = nd(rng);
            }
        }
    }
    for (auto& v : ch.dead) v = nd(rng);
    for (int64_t i = 0; i < ch.nq; ++i)
        for (int64_t h = 0; h < H; ++h) {
            float* qh = ch.q.data() + (i * H + h) * D;
            if (c.kind == kIndexer) {   // RMS-normed, leaning towards a few keys (a peaked score row)
                for (int j = 0; j < D; ++j) qh[j] = nd(rng);
                for (int r = 0; r < 3; ++r) {
                    const float* kb = ch.pooled.data() + (int64_t) (rng() % (uint32_t) (ch.max_blocks - 2)) * D;
                    for (int j = 0; j < D; ++j) qh[j] += 0.5f * kb[j];
                }
                unit(qh, (int) D, std::sqrt((float) D));
            } else {
                for (int j = 0; j < D; ++j) {
                    if (c.kind == kWide) qh[j] = coin(rng) == 0 ? 0.f : std::ldexp(nd(rng), expo(rng));
                    else if (c.kind == kSmall) qh[j] = (float) small(rng);
                    else qh[j] = nd(rng);
                }
            }
        }
    ch.steps.resize((size_t) (ch.nq * k::kStepCount));
    for (int64_t i = 0; i < ch.nq; ++i) {
        int32_t* st = ch.steps.data() + i * k::kStepCount;
        const int64_t pos = c.ctx - ch.nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    return ch;
}

struct Dev {
    float *pooled = nullptr, *dead = nullptr, *q = nullptr, *ref = nullptr, *two = nullptr;
    int32_t *steps = nullptr, *ref_ids = nullptr, *two_ids = nullptr;
    void* scratch = nullptr;
    uint64_t sbytes = 0;
    size_t n_sc = 0, n_id = 0;
};

Dev upload(const Chunk& ch, int64_t nq_cap, int64_t cap) {
    Dev d;
    d.sbytes = k::qsa_two_pass_scratch_bytes(nq_cap, ch.max_blocks);
    d.n_sc = (size_t) (ch.nq * ch.max_blocks);
    d.n_id = (size_t) (ch.nq * cap);
    ck(cudaMalloc(&d.pooled, ch.pooled.size() * 4)); ck(cudaMalloc(&d.dead, ch.dead.size() * 4));
    ck(cudaMalloc(&d.q, ch.q.size() * 4)); ck(cudaMalloc(&d.steps, ch.steps.size() * 4));
    ck(cudaMalloc(&d.ref, d.n_sc * 4)); ck(cudaMalloc(&d.two, d.n_sc * 4));
    ck(cudaMalloc(&d.ref_ids, d.n_id * 4)); ck(cudaMalloc(&d.two_ids, d.n_id * 4)); ck(cudaMalloc(&d.scratch, d.sbytes));
    ck(cudaMemcpy(d.pooled, ch.pooled.data(), ch.pooled.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d.dead, ch.dead.data(), ch.dead.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d.q, ch.q.data(), ch.q.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d.steps, ch.steps.data(), ch.steps.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemset(d.ref, 0x7f, d.n_sc * 4)); ck(cudaMemset(d.two, 0x3e, d.n_sc * 4));
    ck(cudaMemset(d.ref_ids, 0xff, d.n_id * 4)); ck(cudaMemset(d.two_ids, 0xee, d.n_id * 4));
    return d;
}

void release(Dev& d) {
    ck(cudaFree(d.pooled)); ck(cudaFree(d.dead)); ck(cudaFree(d.q)); ck(cudaFree(d.steps)); ck(cudaFree(d.ref));
    ck(cudaFree(d.two)); ck(cudaFree(d.ref_ids)); ck(cudaFree(d.two_ids)); ck(cudaFree(d.scratch));
}

int64_t nbid_of(const Chunk& ch, int64_t i) { return ch.steps[(size_t) (i * k::kStepCount + k::kStepNBid)]; }

// Queries [q0, q0 + nq) in one two-pass call and its qsa_block_topk, ids into `ids` (rows q0 .. of the cap-wide buffer).
bool two_pass_call(const Chunk& ch, const Dev& d, int64_t q0, int64_t nq, int64_t nq_cap, int64_t cap, float* scores,
                   int32_t* ids) {
    const auto s = k::qsa_real_shapes();
    const int64_t D = s.idx_dim, H = s.idx_n_head;
    const int64_t active = nbid_of(ch, q0 + nq - 1) + 1;
    const int32_t* steps = d.steps + q0 * k::kStepCount;
    if (!k::qsa_block_scores_two_pass(d.pooled, d.dead, d.q + q0 * H * D, steps, nq, nq_cap, ch.max_blocks, s,
                                      scores + q0 * ch.max_blocks, d.scratch, d.sbytes, nullptr, active))
        return false;
    k::qsa_block_topk(scores + q0 * ch.max_blocks, steps, nq, ch.max_blocks, cap, s, ids + q0 * cap, nullptr, active);
    return true;
}

// The CPU reference of block b's score for query i: relu per head of q . k in double, summed.
double cpu_score(const Chunk& ch, int64_t i, int64_t b) {
    const auto s = k::qsa_real_shapes();
    const int64_t D = s.idx_dim, H = s.idx_n_head;
    double sum = 0.0;
    for (int64_t h = 0; h < H; ++h) {
        const float* qh = ch.q.data() + (i * H + h) * D;
        const float* kb = ch.pooled.data() + b * D;
        double dot = 0.0;
        for (int64_t j = 0; j < D; ++j) dot += (double) qh[j] * (double) kb[j];
        sum += dot > 0.0 ? dot : 0.0;
    }
    return sum;
}

struct Err { double abs = 0.0, rel = 0.0; };

// The largest error of a scores row against the CPU reference over the completed blocks of three queries: absolute,
// and relative to the query's largest reference score (a block score near 0 has no meaningful relative error).
Err row_error(const Chunk& ch, const std::vector<float>& row) {
    Err e;
    for (const int64_t i : {(int64_t) 0, ch.nq / 2, ch.nq - 1}) {
        const int64_t nb = nbid_of(ch, i);
        double worst = 0.0, top = 0.0;
        for (int64_t b = 0; b < nb; ++b) {
            const double ref = cpu_score(ch, i, b), got = row[(size_t) (i * ch.max_blocks + b)];
            worst = std::max(worst, std::fabs(got - ref));
            top = std::max(top, std::fabs(ref));
        }
        e.abs = std::max(e.abs, worst);
        if (top > 0.0) e.rel = std::max(e.rel, worst / top);
    }
    return e;
}

bool run(const Case& c, int64_t stride, Err& e3, Err& e1) {
    const auto s = k::qsa_real_shapes();
    const int64_t nq_cap = 256, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const Chunk ch = make_chunk(c, stride);
    Dev d = upload(ch, nq_cap, cap);
    const int64_t nq = ch.nq, last_nbid = nbid_of(ch, nq - 1), active = last_nbid + 1;
    k::qsa_block_scores_tc(d.pooled, d.dead, d.q, d.steps, nq, ch.max_blocks, s, d.ref, nullptr, active);
    k::qsa_block_topk(d.ref, d.steps, nq, ch.max_blocks, cap, s, d.ref_ids, nullptr, active);
    unsigned long long st[tp::kStatCount];
    k::qsa_two_pass_stats(st, true, nullptr);   // zero the counters
    if (!k::qsa_two_pass_keys(d.pooled, last_nbid, nq_cap, ch.max_blocks, d.scratch, d.sbytes, nullptr) ||
        !two_pass_call(ch, d, 0, nq, nq_cap, cap, d.two, d.two_ids)) {
        std::printf("FAIL ctx %lld: the two-pass scorer did not run\n", (long long) c.ctx);
        release(d);
        return false;
    }
    if (!k::qsa_two_pass_check(d.two, d.two_ids, d.ref, d.ref_ids, d.steps, nq, nq_cap, ch.max_blocks, cap, d.scratch,
                               nullptr)) {
        std::printf("FAIL ctx %lld: the check did not run\n", (long long) c.ctx);
        release(d);
        return false;
    }
    ck(cudaDeviceSynchronize());
    if (!k::qsa_two_pass_stats(st, true, nullptr)) {
        std::printf("FAIL: counters\n");
        release(d);
        return false;
    }
    std::vector<int32_t> a(d.n_id), b(d.n_id), split(d.n_id);
    ck(cudaMemcpy(a.data(), d.ref_ids, d.n_id * 4, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(b.data(), d.two_ids, d.n_id * 4, cudaMemcpyDeviceToHost));
    std::vector<float> r3(d.n_sc), r1(d.n_sc);
    ck(cudaMemcpy(r3.data(), d.ref, d.n_sc * 4, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(r1.data(), d.two, d.n_sc * 4, cudaMemcpyDeviceToHost));
    // the same queries split into two calls: every query's ids as in the one call
    bool same_split = true;
    if (nq > 1) {
        const int64_t cut = nq > 37 ? 37 : nq / 2;
        ck(cudaMemset(d.two_ids, 0xee, d.n_id * 4));
        if (!two_pass_call(ch, d, 0, cut, nq_cap, cap, d.two, d.two_ids) ||
            !two_pass_call(ch, d, cut, nq - cut, nq_cap, cap, d.two, d.two_ids)) {
            std::printf("FAIL ctx %lld: a split call did not run\n", (long long) c.ctx);
            release(d);
            return false;
        }
        ck(cudaMemcpy(split.data(), d.two_ids, d.n_id * 4, cudaMemcpyDeviceToHost));
        unsigned long long zero[tp::kStatCount];
        k::qsa_two_pass_stats(zero, true, nullptr);   // the split calls' counts are not reported
    }
    int64_t bad_q = 0, bad_split = 0, compared = 0;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t n_kv = ch.steps[(size_t) (i * k::kStepCount + k::kStepNKv)];
        const int64_t w = std::min<int64_t>(n_kv, ch.steps[(size_t) (i * k::kStepCount + k::kStepWidth)]);
        compared += w;
        if (!std::equal(a.begin() + i * cap, a.begin() + i * cap + w, b.begin() + i * cap)) ++bad_q;
        if (nq > 1 && !std::equal(b.begin() + i * cap, b.begin() + i * cap + w, split.begin() + i * cap)) ++bad_split;
    }
    same_split = bad_split == 0;
    e3 = row_error(ch, r3);
    e1 = row_error(ch, r1);
    const bool ok = bad_q == 0 && same_split && st[tp::kChkIdQueries] == 0 && st[tp::kChkRescoredDiff] == 0 &&
                    st[tp::kChkKeptOut] == 0 && st[tp::kChkQueries] == (unsigned long long) nq;
    float ratio;
    const uint32_t rb = (uint32_t) st[tp::kChkRatioMax];
    std::memcpy(&ratio, &rb, 4);
    const unsigned long long kept = st[tp::kStQueries] - st[tp::kStFull];
    std::printf("%s ctx %7lld stride %5lld queries %3lld %-20s: ids of %lld queries differ (%lld compared), %lld after "
                "a split; scored again %llu (%.1f a query), %llu of %llu not bitwise; flagged %llu (%llu tiles in "
                "full); kept %llu, outside the bound %llu, max |S1-S3|/eps %.4f\n",
                ok ? "ok  " : "FAIL", (long long) c.ctx, (long long) ch.max_blocks, (long long) nq, kKindName[c.kind],
                (long long) bad_q, (long long) compared, (long long) bad_split, st[tp::kStBand],
                kept ? (double) st[tp::kStBand] / (double) kept : 0.0, st[tp::kChkRescoredDiff], st[tp::kChkRescored],
                st[tp::kStFull], st[tp::kStFullTiles], st[tp::kChkKept], st[tp::kChkKeptOut], (double) ratio);
    release(d);
    return ok;
}

// Per-call times of both scorers on one 256-query call ending at ctx (indexer-like keys), the two-pass time with the
// key norms once per 8 calls (one 2,048-token chunk); FLOPs: 2 * 128 per (query, block, head), three times for 3xTF32.
void bench_one(int64_t ctx) {
    const auto s = k::qsa_real_shapes();
    const int64_t nq_cap = 256, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const Chunk ch = make_chunk({ctx, 256, kIndexer}, 65538);
    Dev d = upload(ch, nq_cap, cap);
    const int64_t last_nbid = nbid_of(ch, ch.nq - 1), active = last_nbid + 1;
    double blocks = 0.0;
    for (int64_t i = 0; i < ch.nq; ++i) blocks += (double) nbid_of(ch, i);
    const double flop1 = blocks * s.idx_n_head * s.idx_dim * 2.0, flop3 = 3.0 * flop1;
    cudaEvent_t e0, e1, e2;
    ck(cudaEventCreate(&e0)); ck(cudaEventCreate(&e1)); ck(cudaEventCreate(&e2));
    constexpr int N = 24;
    for (int r = 0; r < 2; ++r) {   // warm-up
        k::qsa_block_scores_tc(d.pooled, d.dead, d.q, d.steps, ch.nq, ch.max_blocks, s, d.ref, nullptr, active);
        k::qsa_two_pass_keys(d.pooled, last_nbid, nq_cap, ch.max_blocks, d.scratch, d.sbytes, nullptr);
        k::qsa_block_scores_two_pass(d.pooled, d.dead, d.q, d.steps, ch.nq, nq_cap, ch.max_blocks, s, d.two, d.scratch,
                                     d.sbytes, nullptr, active);
    }
    ck(cudaEventRecord(e0));
    for (int r = 0; r < N; ++r)
        k::qsa_block_scores_tc(d.pooled, d.dead, d.q, d.steps, ch.nq, ch.max_blocks, s, d.ref, nullptr, active);
    ck(cudaEventRecord(e1));
    for (int r = 0; r < N; ++r) {   // the key norms once per 8 calls, as for a 2,048-token chunk
        if (r % 8 == 0) k::qsa_two_pass_keys(d.pooled, last_nbid, nq_cap, ch.max_blocks, d.scratch, d.sbytes, nullptr);
        k::qsa_block_scores_two_pass(d.pooled, d.dead, d.q, d.steps, ch.nq, nq_cap, ch.max_blocks, s, d.two, d.scratch,
                                     d.sbytes, nullptr, active);
    }
    ck(cudaEventRecord(e2));
    ck(cudaEventSynchronize(e2));
    float t3 = 0, t2 = 0;
    ck(cudaEventElapsedTime(&t3, e0, e1)); ck(cudaEventElapsedTime(&t2, e1, e2));
    const double us3 = 1000.0 * t3 / N, us2 = 1000.0 * t2 / N;
    std::printf("ctx %6lld, 256 queries: 3xTF32 scorer %8.1f us a call (%.1f TFLOP/s), two-pass %8.1f us a call with "
                "the key norms once per 8 calls (%.1f TFLOP/s single-pass equivalent), %.2fx\n",
                (long long) ctx, us3, flop3 / (us3 * 1e6), us2, flop1 / (us2 * 1e6), us3 / us2);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    unsigned long long zero[tp::kStatCount];
    k::qsa_two_pass_stats(zero, true, nullptr);
    release(d);
}
}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 0;
    }
    {   // the 3xTF32 scorer is the reference: without it there is nothing to compare
        const auto s = k::qsa_real_shapes();
        const Chunk ch = make_chunk({2100, 4, kNormal}, 0);
        Dev d = upload(ch, 256, k::qsa_selection_width(k::kTopkMaxCells, s));
        const bool tc = k::qsa_block_scores_tc(d.pooled, d.dead, d.q, d.steps, ch.nq, ch.max_blocks, s, d.ref, nullptr,
                                               nbid_of(ch, ch.nq - 1) + 1);
        release(d);
        if (!tc) {
            std::printf("SKIP: the 3xTF32 scorer does not run on this card\n");
            return 0;
        }
    }
    if (do_bench) {
        for (const int64_t ctx : {(int64_t) 32768, (int64_t) 131072, (int64_t) 262144}) bench_one(ctx);
        return 0;
    }
    const Case cases[] = {
        {2100, 64, kNormal},     {4096, 256, kNormal},   {9000, 37, kNormal},     {20000, 256, kNormal},
        {20481, 256, kSmall},    {32768, 256, kIndexer}, {40000, 256, kWide},     {50000, 200, kNear},
        {70000, 256, kDup},      {131072, 256, kIndexer}, {131075, 129, kNormal}, {262144, 256, kIndexer},
        {262144, 256, kNormal},  {200003, 256, kSmall},  {65536, 256, kIndexer},
    };
    int fails = 0, n = 0;
    Err w3, w1;
    for (const int64_t stride : {(int64_t) 0, (int64_t) 65538})
        for (const Case& c : cases) {
            if (stride > 0 && c.ctx / 4 + 2 > stride) continue;
            Err e3, e1;
            fails += run(c, stride, e3, e1) ? 0 : 1;
            ++n;
            if (c.kind == kNormal || c.kind == kIndexer) {   // the error report: the kinds with real-valued keys
                w3.abs = std::max(w3.abs, e3.abs); w3.rel = std::max(w3.rel, e3.rel);
                w1.abs = std::max(w1.abs, e1.abs); w1.rel = std::max(w1.rel, e1.rel);
            }
        }
    std::printf("against the CPU reference (double), normal and indexer-like keys: 3xTF32 row max abs %.3g (%.3g of the "
                "query's largest score); two-pass row max abs %.3g (%.3g)\n", w3.abs, w3.rel, w1.abs, w1.rel);
    std::printf("%d cases, %d failed\n", n, fails);
    std::printf("%s\n", fails == 0 ? "PASS: two-pass ids identical to the 3xTF32 scorer's, scored-again blocks bitwise, "
                                     "kept blocks inside the bound, the same ids in a split call"
                                   : "FAIL");
    return fails == 0 ? 0 : 1;
}
