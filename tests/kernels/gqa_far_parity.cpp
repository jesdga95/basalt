// tests/kernels/gqa_far_parity.cpp - a cache split in two places (KvFar, basalt/kernels/gqa_attn.hpp): the cells from `at` on
// in other buffers, page-locked device-mapped host memory for decode (K/V streaming) or a device copy for prompt chunks.
// At the Qwen3.5-27B attention geometry (24 query heads, 4 KV heads, 256 wide), fp16 and int8 caches, a split that falls
// inside a decode chunk and a prompt key tile:
//
//   - attn_qk_prepare and attn_qk_prepare_bf16 write each cell where it lives: the two buffers hold the one-buffer bytes;
//   - gqa_decode_attn (windows of 1 to 16 rows, host and device first cell) and gqa_decode_attn_segs, gqa_prompt_attn and
//     gqa_prompt_attn_bf16 (fp16 and int8, narrow, split and 8-warp launches) are bitwise their one-buffer launches.
//
// `--bench`: a decode row and an 8-row window over 128K cells with none, half and all of them in host memory (the bus
// reads they cost).  Exit 77 without a GPU.
#include "basalt/kernels/bf16_bits.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/gqa_attn.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int NH = 24, NKV = 4, HD = 256, ROW = NKV * HD;   // ROW: values of one cell

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void* dev_bytes(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n), "cudaMalloc");
    return p;
}
void* host_bytes(size_t n) {   // page-locked, device-mapped: the same address on the card
    void* p = nullptr;
    ck(cudaHostAlloc(&p, n, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
    return p;
}

// One layer's cache of `cells` cells: one buffer (`one`), and the same cells split at `at` (`near` on the card, `far` in
// host memory or on the card).  fp16 K/V or int8 codes with their fp16 scales.
struct Cache {
    bool int8 = false;
    int64_t cells = 0, at = 0;
    uint16_t *k = nullptr, *v = nullptr;            // one buffer, fp16
    basalt::kernels::KvInt8 q8;                     // one buffer, int8
    uint16_t *nk = nullptr, *nv = nullptr;          // split: near_p
    basalt::kernels::KvInt8 n8;
    basalt::kernels::KvFar far_kv;                     // split: far
    size_t bytes(int64_t n, bool scales) const { return (size_t) n * ROW * (int8 ? 1 : 2) / (scales ? 32 : 1); }

    bool on_host = false;
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;
    ~Cache() {
        for (void* p : {(void*) k, (void*) v, (void*) nk, (void*) nv, (void*) q8.k, (void*) q8.v, (void*) q8.ks, (void*) q8.vs,
                        (void*) n8.k, (void*) n8.v, (void*) n8.ks, (void*) n8.vs})
            if (p) cudaFree(p);
        for (void* p : {(void*) far_kv.k, (void*) far_kv.v, (void*) far_kv.q8.k, (void*) far_kv.q8.v, (void*) far_kv.q8.ks, (void*) far_kv.q8.vs})
            if (p) (void) (on_host ? cudaFreeHost(p) : cudaFree(p));
    }
    Cache(bool i8, int64_t n, int64_t split, bool far_on_host, std::mt19937& rng) : int8(i8), cells(n), at(split), on_host(far_on_host) {
        far_kv.at = split;
        const auto alloc_far = [&](size_t b) { return far_on_host ? host_bytes(b) : dev_bytes(b); };
        if (!int8) {
            k = (uint16_t*) dev_bytes(bytes(n, false));
            v = (uint16_t*) dev_bytes(bytes(n, false));
            nk = (uint16_t*) dev_bytes(bytes(split, false));
            nv = (uint16_t*) dev_bytes(bytes(split, false));
            far_kv.k = (uint16_t*) alloc_far(bytes(n - split, false));
            far_kv.v = (uint16_t*) alloc_far(bytes(n - split, false));
        } else {
            q8 = {(int8_t*) dev_bytes(bytes(n, false)), (int8_t*) dev_bytes(bytes(n, false)), (uint16_t*) dev_bytes(bytes(n, true)),
                  (uint16_t*) dev_bytes(bytes(n, true))};
            n8 = {(int8_t*) dev_bytes(bytes(split, false)), (int8_t*) dev_bytes(bytes(split, false)),
                  (uint16_t*) dev_bytes(bytes(split, true)), (uint16_t*) dev_bytes(bytes(split, true))};
            far_kv.q8 = {(int8_t*) alloc_far(bytes(n - split, false)), (int8_t*) alloc_far(bytes(n - split, false)),
                      (uint16_t*) alloc_far(bytes(n - split, true)), (uint16_t*) alloc_far(bytes(n - split, true))};
        }
        fill(rng);
    }
    // random cells into both forms
    void fill(std::mt19937& rng) {
        std::normal_distribution<float> nd(0.f, 1.f);
        std::uniform_int_distribution<int> code(-127, 127);
        const auto put = [&](void* one, void* near_p, void* farp, size_t elem, bool scales, auto gen) {
            const size_t n = bytes(cells, scales) / elem, a = bytes(at, scales) / elem;
            std::vector<uint8_t> h(n * elem);
            for (size_t i = 0; i < n; ++i) gen(h.data() + i * elem);
            ck(cudaMemcpy(one, h.data(), n * elem, cudaMemcpyDefault), "copy");
            ck(cudaMemcpy(near_p, h.data(), a * elem, cudaMemcpyDefault), "copy");
            ck(cudaMemcpy(farp, h.data() + a * elem, (n - a) * elem, cudaMemcpyDefault), "copy");
        };
        const auto f16 = [&](uint8_t* p) { const uint16_t x = basalt::kernels::f16_from_f32(nd(rng)); std::memcpy(p, &x, 2); };
        const auto i8 = [&](uint8_t* p) { *(int8_t*) p = (int8_t) code(rng); };
        const auto sc = [&](uint8_t* p) {
            const uint16_t x = basalt::kernels::f16_from_f32(0.004f + 0.02f * std::fabs(nd(rng)));
            std::memcpy(p, &x, 2);
        };
        if (!int8) {
            put(k, nk, far_kv.k, 2, false, f16);
            put(v, nv, far_kv.v, 2, false, f16);
        } else {
            put(q8.k, n8.k, far_kv.q8.k, 1, false, i8);
            put(q8.v, n8.v, far_kv.q8.v, 1, false, i8);
            put(q8.ks, n8.ks, far_kv.q8.ks, 2, true, sc);
            put(q8.vs, n8.vs, far_kv.q8.vs, 2, true, sc);
        }
    }
    // the split form's bytes, near then far, equal the one buffer's
    bool same() const {
        const auto eq = [&](const void* one, const void* near_p, const void* farp, bool scales) {
            const size_t n = bytes(cells, scales), a = bytes(at, scales);
            std::vector<uint8_t> x(n), y(n);
            ck(cudaMemcpy(x.data(), one, n, cudaMemcpyDefault), "copy");
            ck(cudaMemcpy(y.data(), near_p, a, cudaMemcpyDefault), "copy");
            ck(cudaMemcpy(y.data() + a, farp, n - a, cudaMemcpyDefault), "copy");
            return x == y;
        };
        if (!int8) return eq(k, nk, far_kv.k, false) && eq(v, nv, far_kv.v, false);
        return eq(q8.k, n8.k, far_kv.q8.k, false) && eq(q8.v, n8.v, far_kv.q8.v, false) && eq(q8.ks, n8.ks, far_kv.q8.ks, true) &&
               eq(q8.vs, n8.vs, far_kv.q8.vs, true);
    }
};

template <class T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "copy");
    return h;
}

int fails = 0;
void report(const char* what, bool ok) {
    std::printf("%s: %s\n", what, ok ? "bitwise" : "DIFFERENT");
    if (!ok) ++fails;
}

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no GPU\n");
        return 77;
    }
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int64_t L = 6000, at = 3001;   // the split inside a 32-cell decode chunk and a 64-cell prompt tile
    const int TP = 300, TB = 2048;
    std::vector<float> q((size_t) TB * NH * HD), qf((size_t) 16 * NH * 2 * HD), kr((size_t) 16 * ROW), vr(kr.size());
    for (auto& x : q) x = 2.0f * nd(rng);
    for (auto& x : qf) x = nd(rng);
    for (auto& x : kr) x = nd(rng);
    for (auto& x : vr) x = nd(rng);
    std::vector<uint16_t> qb(q.size());
    for (size_t i = 0; i < q.size(); ++i) qb[i] = basalt::kernels::bf16_from_f32(q[i]);
    float* d_q = (float*) dev_bytes(q.size() * 4);
    uint16_t* d_qb = (uint16_t*) dev_bytes(qb.size() * 2);
    float *d_qf = (float*) dev_bytes(qf.size() * 4), *d_kr = (float*) dev_bytes(kr.size() * 4), *d_vr = (float*) dev_bytes(vr.size() * 4);
    ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_qb, qb.data(), qb.size() * 2, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_qf, qf.data(), qf.size() * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_kr, kr.data(), kr.size() * 4, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_vr, vr.data(), vr.size() * 4, cudaMemcpyHostToDevice), "copy");
    std::vector<float> norm(HD, 1.0f);
    float* d_norm = (float*) dev_bytes(HD * 4);
    ck(cudaMemcpy(d_norm, norm.data(), HD * 4, cudaMemcpyHostToDevice), "copy");
    const size_t out_floats = (size_t) TB * NH * HD;
    float *d_o1 = (float*) dev_bytes(out_floats * 4), *d_o2 = (float*) dev_bytes(out_floats * 4);
    float* d_qo = (float*) dev_bytes((size_t) 16 * NH * HD * 4);
    float* d_scr = (float*) dev_bytes(basalt::kernels::gqa_decode_scratch_floats(L, basalt::kernels::kGqaMaxWindow, NH) * 4);
    const size_t split_floats = basalt::kernels::gqa_prompt_split_floats(TP, NH, 8);
    float* d_split = (float*) dev_bytes(split_floats * 4);
    int32_t* d_pos = (int32_t*) dev_bytes(16 * 3 * 4);
    int32_t* d_at = (int32_t*) dev_bytes(2 * 4);
    basalt::kernels::RopeSpec rope;
    rope.rot = 64;
    rope.theta = 1e7f;
    const auto same = [&](size_t floats) { return down(d_o1, floats) == down(d_o2, floats); };

    for (const bool int8 : {false, true}) {
        const char* kv = int8 ? "int8" : "fp16";
        char what[160];
        {   // the cache writes: rows at cells at - 2 .. at + 2, across the split, far cells in host memory
            Cache c(int8, L, at, true, rng);
            const int T = 5;
            const int64_t cell0 = at - 2;
            std::vector<int32_t> pos((size_t) T * 3);
            for (int t = 0; t < T; ++t) pos[(size_t) t * 3] = pos[(size_t) t * 3 + 1] = pos[(size_t) t * 3 + 2] = (int32_t) (cell0 + t);
            ck(cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice), "copy");
            basalt::kernels::attn_qk_prepare(d_qf, d_kr, d_vr, d_norm, d_norm, 1e-6f, rope, d_pos, T, NH, NKV, d_qo, c.k, c.v, cell0,
                                             nullptr, 256, true, 0, nullptr, c.q8);
            basalt::kernels::attn_qk_prepare(d_qf, d_kr, d_vr, d_norm, d_norm, 1e-6f, rope, d_pos, T, NH, NKV, d_qo, c.nk, c.nv,
                                             cell0, nullptr, 256, true, 0, nullptr, c.n8, c.far_kv);
            ck(cudaDeviceSynchronize(), "prepare");
            std::snprintf(what, sizeof what, "%s attn_qk_prepare across the split: the two buffers hold the one buffer's bytes", kv);
            report(what, c.same());
        }
        // decode: far cells in host memory
        Cache c(int8, L, at, true, rng);
        bool ok = true;
        for (int T = 1; T <= basalt::kernels::kGqaMaxWindow; ++T) {
            const int64_t p0 = L - T;
            basalt::kernels::gqa_decode_attn(d_q, c.k, c.v, NH, NKV, p0, T, d_scr, L, d_o1, nullptr, nullptr, c.q8);
            basalt::kernels::gqa_decode_attn(d_q, c.nk, c.nv, NH, NKV, p0, T, d_scr, L, d_o2, nullptr, nullptr, c.n8, nullptr,
                                             nullptr, c.far_kv);
            ok = ok && same((size_t) T * NH * HD);
        }
        std::snprintf(what, sizeof what, "%s gqa_decode_attn, windows of 1-16 rows, far cells in host memory", kv);
        report(what, ok);
        {   // a device first cell (a captured window), and a window that ends before the split
            const int32_t a2[2] = {(int32_t) (L - 8), 0};
            ck(cudaMemcpy(d_at, a2, sizeof a2, cudaMemcpyHostToDevice), "copy");
            basalt::kernels::gqa_decode_attn(d_q, c.k, c.v, NH, NKV, L - 8, 8, d_scr, L, d_o1, nullptr, d_at, c.q8);
            basalt::kernels::gqa_decode_attn(d_q, c.nk, c.nv, NH, NKV, L - 8, 8, d_scr, L, d_o2, nullptr, d_at, c.n8, nullptr, nullptr,
                                             c.far_kv);
            bool ok2 = same((size_t) 8 * NH * HD);
            basalt::kernels::gqa_decode_attn(d_q, c.k, c.v, NH, NKV, at - 40, 3, d_scr, L, d_o1, nullptr, nullptr, c.q8);
            basalt::kernels::gqa_decode_attn(d_q, c.nk, c.nv, NH, NKV, at - 40, 3, d_scr, L, d_o2, nullptr, nullptr, c.n8, nullptr,
                                             nullptr, c.far_kv);
            ok2 = ok2 && same((size_t) 3 * NH * HD);
            std::snprintf(what, sizeof what, "%s gqa_decode_attn, a device first cell and a window before the split", kv);
            report(what, ok2);
        }
        {   // two segments in one launch: a split cache and a one-buffer one
            basalt::kernels::GqaSeg s1[2], s2[2];
            s1[0].T = s2[0].T = 5;
            s1[0].pos0 = s2[0].pos0 = L - 5;
            s1[1].T = s2[1].T = 7;
            s1[1].pos0 = s2[1].pos0 = 2500;
            s1[0].k_cache = s1[1].k_cache = s2[1].k_cache = c.k;
            s1[0].v_cache = s1[1].v_cache = s2[1].v_cache = c.v;
            s1[0].q8 = s1[1].q8 = s2[1].q8 = c.q8;
            s2[0].k_cache = c.nk;
            s2[0].v_cache = c.nv;
            s2[0].q8 = c.n8;
            s2[0].far_kv = c.far_kv;
            basalt::kernels::gqa_decode_attn_segs(d_q, NH, NKV, s1, 2, d_scr, L, d_o1, nullptr);
            basalt::kernels::gqa_decode_attn_segs(d_q, NH, NKV, s2, 2, d_scr, L, d_o2, nullptr);
            std::snprintf(what, sizeof what, "%s gqa_decode_attn_segs, a split segment beside a one-buffer one", kv);
            report(what, same((size_t) 12 * NH * HD));
        }
        // prompt chunks: far cells in device memory (the stage copy)
        Cache p(int8, L, at, false, rng);
        for (int cfg = 0; cfg < 3; ++cfg) {
            const int T = cfg < 2 ? TP : TB;
            const int64_t p0 = L - T;
            float* sp = cfg == 1 ? d_split : nullptr;
            const size_t sf = cfg == 1 ? split_floats : 0;
            basalt::kernels::gqa_prompt_attn(d_q, p.k, p.v, NH, NKV, p0, T, d_o1, nullptr, p.q8, sp, sf);
            basalt::kernels::gqa_prompt_attn(d_q, p.nk, p.nv, NH, NKV, p0, T, d_o2, nullptr, p.n8, sp, sf, p.far_kv);
            bool okp = same((size_t) T * NH * HD);
            if (int8) {   // the BF16 form
                basalt::kernels::gqa_prompt_attn_bf16(d_qb, NH, NKV, p0, T, (uint16_t*) d_o1, nullptr, p.q8, sp, sf);
                basalt::kernels::gqa_prompt_attn_bf16(d_qb, NH, NKV, p0, T, (uint16_t*) d_o2, nullptr, p.n8, sp, sf, p.far_kv);
                okp = okp && same((size_t) T * NH * HD / 2);
            }
            static const char* shape[3] = {"300 rows", "300 rows split", "2048 rows"};
            std::snprintf(what, sizeof what, "%s gqa_prompt_attn%s, %s, far cells in device memory", kv, int8 ? " (and _bf16)" : "",
                          shape[cfg]);
            report(what, okp);
        }
    }

    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        // a decode row and an 8-row window over 128K int8 and fp16 cells, none, half or all of them in host memory
        const int64_t N = 131072;
        float* d_s = (float*) dev_bytes(basalt::kernels::gqa_decode_scratch_floats(N, 8, NH) * 4);
        for (const bool int8 : {true, false})
            for (const int64_t split : {N - 1, N / 2, (int64_t) 1}) {   // (about) none, half and all of the cells far
                Cache b(int8, N, split, true, rng);
                for (const int T : {1, 8}) {
                    cudaEvent_t e0, e1;
                    cudaEventCreate(&e0);
                    cudaEventCreate(&e1);
                    for (int i = 0; i < 3; ++i)
                        basalt::kernels::gqa_decode_attn(d_q, b.nk, b.nv, NH, NKV, N - T, T, d_s, N, d_o2, nullptr, nullptr, b.n8,
                                                         nullptr, nullptr, b.far_kv);
                    cudaEventRecord(e0);
                    const int reps = 20;
                    for (int i = 0; i < reps; ++i)
                        basalt::kernels::gqa_decode_attn(d_q, b.nk, b.nv, NH, NKV, N - T, T, d_s, N, d_o2, nullptr, nullptr, b.n8,
                                                         nullptr, nullptr, b.far_kv);
                    cudaEventRecord(e1);
                    cudaEventSynchronize(e1);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, e0, e1);
                    const double us = 1000.0 * ms / reps, far_b = (double) b.bytes(N - b.at, false) * 2 * (int8 ? 1 + 1.0 / 32 : 1);
                    std::printf("bench: %s decode %d row(s) x %lld cells, %lld in host memory: %.1f us (%.1f GB/s of far K/V)\n",
                                int8 ? "int8" : "fp16", T, (long long) N, (long long) (N - b.at), us, far_b / us / 1e3);
                }
            }
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
