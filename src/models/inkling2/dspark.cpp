// src/models/inkling2/dspark.cpp - Inkling 2's DSpark drafter (see dspark.hpp). The block reuses the shared kernels (RMS
// norms, Q8_0 MMVQ over q8_1 rows, attn_qk_prepare for the rotary heads and the cache writes, gqa_window_attn with the
// block's shared window, the fused SwiGLU quantize, BF16 MMVF); the mixes, the block's inputs, the recurrence, the pick and
// the folds are in kernels/dspark.cu.
//
// The forward follows vLLM's Inkling 2 DSpark draft (vllm/models/inkling2/nvidia/dspark.py, Apache-2.0, Copyright
// contributors to the vLLM project).
#include "basalt/core/device_copy.hpp"
#include "models/inkling2/dspark.hpp"

#include "basalt/core/diag.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/nvfp4_planes.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/models/inkling2/dspark_kernels.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

constexpr int kTypeF32 = 0;
constexpr int kTypeBF16 = 30;
constexpr int kE = kWidth;
constexpr int kQ = kHeads * kHeadDim;      // 3072
constexpr int kKV = kKvHeads * kHeadDim;   // 1024
constexpr int kFrRing = 32;                // positions whose Markov context the ring keeps
constexpr int kSlice = 8;                  // rows of one commit step (MMVQ's widest)
constexpr int kRingExtra = 64;             // a sliding-window layer's ring: its window, a block and a window's rows past it
constexpr int kDumpBlocks = 24;            // blocks --diag-dump writes
constexpr int kFp4Mats = 7;                // a layer's block products in NVFP4 planes: q, k, v, o, gate, up, down

uint64_t q8_bytes(uint64_t rows, uint64_t n_in) { return rows * (n_in / 32) * 34; }

int64_t ring_cells(const DsparkGeometry& d) { return (int64_t) d.window + kRingExtra; }
int64_t full_cells(int64_t context) { return context + kDsBlock; }

bk::RopeSpec rope_of(const DsparkGeometry& d) {
    bk::RopeSpec r;
    r.rot = kHeadDim;
    r.theta = d.rope_base;
    r.scaling.freq_base = d.rope_base;
    return r;
}

// The window gemm's split-K scratch for the block's products: q + k + v, o, gate + up, down
void ws_need(const DsparkGeometry& d, size_t& partial, size_t& tickets) {
    namespace wg = bk::wgemm;
    partial = tickets = 0;
    const auto need = [&](int64_t K, std::initializer_list<int64_t> rows, wg::Epilogue epi) {
        wg::Call c;
        c.fmt = wg::Format::Nvfp4;
        c.K = K;
        c.epi = epi;
        c.n = 0;
        for (int64_t r : rows) c.m[c.n++].rows = r;
        size_t p = 0, t = 0;
        wg::workspace_need(c, p, t);
        partial = std::max(partial, p);
        tickets = std::max(tickets, t);
    };
    need(kE, {kQ, kKV, kKV}, wg::Epilogue::Store);
    need(kQ, {kE}, wg::Epilogue::Add);
    need(kE, {d.n_ff, d.n_ff}, wg::Epilogue::SwiGLU);
    need(d.n_ff, {kE}, wg::Epilogue::Add);
}

uint64_t dspark_ws_bytes(const DsparkGeometry& d) {
    size_t p = 0, t = 0;
    ws_need(d, p, t);
    return (uint64_t) (std::max<size_t>(p, 1) + std::max<size_t>(t, 1)) * 4;
}

}  // namespace

// ---- weights --------------------------------------------------------------------------------------------------------------

DsparkWeights::~DsparkWeights() {
    for (void* p : allocs_) cudaFree(p);
    for (void* p : pinned_) cudaFreeHost(p);
}

void* DsparkWeights::mapped(size_t bytes, bool& ok) {
    void* h = nullptr;
    void* d = nullptr;
    if (!ok || cudaHostAlloc(&h, std::max<size_t>(bytes, 16), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        ok = false;
        return nullptr;
    }
    pinned_.push_back(h);
    if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
        ok = false;
        return nullptr;
    }
    return d;
}

void* DsparkWeights::dev(size_t bytes, bool& ok) {
    void* p = nullptr;
    if (!ok || cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
        ok = false;
        return nullptr;
    }
    allocs_.push_back(p);
    bytes_ += bytes;
    return p;
}

model::Result dspark_plan(const Pack& pack, core::LoadPlan& reads, int file, int device, std::unique_ptr<DsparkWeights>& out) {
    out.reset();
    const DsparkGeometry& d = pack.dspark;
    if (!d.present) return {};
    const Geometry& g = pack.geo;
    auto fail = [](const std::string& e) { return model::Result{model::Status::Error, "inkling2: the drafter: " + e}; };
    // the layout the kernels are built for
    if (d.n_layers != kDsMixes - 1 || d.block != kDsBlock || d.n_head != kHeads || d.n_head_kv != kKvHeads ||
        d.head_dim != kHeadDim || d.markov_rank != kDsRank || d.n_ff <= 0 || d.n_ff % 256 || d.window <= 0 ||
        (int) d.swa.size() != d.n_layers || (int) d.targets.size() != kDsTargets || g.n_embd != kE)
        return fail("a DSpark layout this engine does not run (3 layers of 24 query and 8 key/value heads of 128, a block of 8, "
                    "a rank-256 Markov head, 4 target features)");
    for (int k = 0; k + 1 < kDsTargets; ++k)
        if (d.targets[(size_t) k] < 0 || d.targets[(size_t) k] >= (int) g.visits.size() ||
            (k > 0 && d.targets[(size_t) k] <= d.targets[(size_t) k - 1]))
            return fail("its target visits are not in order within the target's visits");
    if (d.targets.back() != (int) g.visits.size()) return fail("its last target is not the target's final normed state");

    auto w = std::make_unique<DsparkWeights>();
    w->geo = d;
    w->n_vocab = g.n_vocab;
    w->logit_softcap = g.logit_softcap;
    bool ok = true;
    std::string err;
    // a tensor of the pack to the card: its type and size checked, read as it is or rounded as it loads
    auto load = [&](const std::string& name, int type, uint64_t values, core::Conv conv, uint64_t dst_bytes, bool host = false) -> void* {
        if (!ok) return nullptr;
        const TensorRef* t = pack.find(name);
        const uint64_t want = values * (type == kTypeF32 ? 4 : 2);
        if (!t || t->type != type || t->bytes != want) {
            err = name + (t ? " has another type or size" : " is missing");
            ok = false;
            return nullptr;
        }
        void* p = host ? w->mapped(dst_bytes, ok) : w->dev(dst_bytes, ok);
        if (!ok) {
            err = std::string("out of ") + (host ? "pinned host" : "device") + " memory for " + name;
            return nullptr;
        }
        core::LoadItem it;
        it.file = file;
        it.off = t->offset;
        it.bytes = t->bytes;
        if (host) it.host = (uint8_t*) w->pinned_.back();
        else it.dev = p;
        it.device = device;
        it.conv = conv;
        reads.add(it);
        return p;
    };
    auto f32 = [&](const std::string& name, uint64_t n) { return (float*) load(name, kTypeF32, n, core::Conv::Copy, n * 4); };
    auto bf16 = [&](const std::string& name, uint64_t n) { return (uint16_t*) load(name, kTypeBF16, n, core::Conv::Copy, n * 2); };
    auto q8 = [&](const std::string& name, uint64_t rows, uint64_t n_in) {
        return load(name, kTypeBF16, rows * n_in, core::Conv::Bf16ToQ8_0, q8_bytes(rows, n_in));
    };
    w->layer.assign((size_t) d.n_layers, DsparkLayer{});
    for (int l = 0; l < d.n_layers; ++l) {
        DsparkLayer& L = w->layer[(size_t) l];
        const std::string b = "dspark.blk." + std::to_string(l) + ".";
        L.local = d.swa[(size_t) l];
        L.wq = q8(b + "attn_q.weight", kQ, kE);
        L.wk = q8(b + "attn_k.weight", kKV, kE);
        L.wv = q8(b + "attn_v.weight", kKV, kE);
        L.wo = q8(b + "attn_output.weight", kE, kQ);
        L.k_ctx = q8(b + "attn_k_ctx.weight", kKV, kE);
        L.v_ctx = q8(b + "attn_v_ctx.weight", kKV, kE);
        L.k_ctx_p = w->dev(q8_bytes(kKV, kE), ok);
        L.v_ctx_p = w->dev(q8_bytes(kKV, kE), ok);
        L.gate = q8(b + "ffn_gate.weight", (uint64_t) d.n_ff, kE);
        L.up = q8(b + "ffn_up.weight", (uint64_t) d.n_ff, kE);
        L.down = q8(b + "ffn_down.weight", kE, (uint64_t) d.n_ff);
        L.attn_norm = f32(b + "attn_norm.weight", kE);
        L.post_norm = f32(b + "post_attention_norm.weight", kE);
        L.dflare_norm = f32(b + "dflare_norm.weight", kE);
        L.q_norm = f32(b + "attn_q_norm.weight", kHeadDim);
        L.k_norm = f32(b + "attn_k_norm.weight", kHeadDim);
        L.k_ctx_norm = f32(b + "attn_k_ctx_norm.weight", kHeadDim);
        L.fp4_scale = (float*) w->dev(kFp4Mats * 4, ok);
    }
    w->alpha = f32("dspark.fuse_weight", kDsMixes * kDsTargets);   // the raw weights; finish() softmaxes them in place
    w->mask = f32("dspark.mask_embed", kE);
    w->norm = f32("dspark.norm.weight", kE);
    w->markov_norm = f32("dspark.markov.dflare_norm.weight", kE);
    {   // the Markov head's token rows in mapped host memory: a step reads the previous draft's row
        const uint64_t n = (uint64_t) g.n_vocab * kDsRank;
        w->w1 = (uint16_t*) load("dspark.markov.w1.weight", kTypeBF16, n, core::Conv::Copy, n * 2, true);
    }
    w->rnn_w = bf16("dspark.markov.rnn.weight", (uint64_t) kDsRnn * (2 * kDsRank + 2 * kE));
    w->rnn_b = f32("dspark.markov.rnn.bias", kDsRnn);
    w->out_w = bf16("dspark.markov.out.weight", (uint64_t) kE * kDsRank);
    w->out_b = f32("dspark.markov.out.bias", kE);
    if (!ok) return fail(err);
    w->w_sp = (uint16_t*) w->dev((size_t) kDsRnn * 2 * kDsRank * 2, ok);
    w->w_h = (uint16_t*) w->dev((size_t) kDsRnn * kE * 2, ok);
    w->w_f = (uint16_t*) w->dev((size_t) kDsRnn * kE * 2, ok);
    w->fold = w->dev(q8_bytes((uint64_t) g.n_vocab, kDsRank), ok);
    w->fold_b = (float*) w->dev((size_t) g.n_vocab * 4, ok);
    w->fold_bp = (float*) w->dev((size_t) dspark_screen_rows(g.n_vocab) * 4, ok);
    w->head_sc = w->dev(dspark_screen_bytes(kE, g.n_vocab), ok);
    w->fold_sc = w->dev(dspark_screen_bytes(kDsRank, g.n_vocab), ok);
    if (!ok) return fail("out of device memory");
    basalt::log::info("basalt inkling2", "drafter: DSpark, %d layers reading target visits %d, %d, %d, %.2f GiB on the GPU (NVFP4 block products, Q8_0 context projections, the screen's copies), "
                      "%.0f MB of token rows in host memory",
                      d.n_layers, d.targets[0], d.targets[1], d.targets[2], (double) w->bytes() / (double) (1ull << 30),
                      (double) g.n_vocab * kDsRank * 2 / 1e6);
    out = std::move(w);
    return {};
}

model::Result DsparkWeights::finish(const void* head_q8, const uint16_t* emb, const float* en, void* stream) {
    head = head_q8;
    embedding = emb;
    embed_norm = en;
    if (finished_) return {};
    const cudaStream_t s = (cudaStream_t) stream;
    float raw[kDsMixes * kDsTargets], al[kDsMixes * kDsTargets];
    if (basalt::core::device_copy(raw, alpha, sizeof(raw), cudaMemcpyDeviceToHost) != cudaSuccess)
        return {model::Status::Error, "inkling2: the drafter's fuse weights cannot be read"};
    for (int j = 0; j < kDsMixes; ++j) {   // softmax over the targets, in double
        double m = raw[j * kDsTargets], z = 0;
        for (int k = 1; k < kDsTargets; ++k) m = std::max(m, (double) raw[j * kDsTargets + k]);
        for (int k = 0; k < kDsTargets; ++k) z += std::exp((double) raw[j * kDsTargets + k] - m);
        for (int k = 0; k < kDsTargets; ++k) al[j * kDsTargets + k] = (float) (std::exp((double) raw[j * kDsTargets + k] - m) / z);
    }
    cudaMemcpyAsync(alpha, al, sizeof(al), cudaMemcpyHostToDevice, s);
    try {
        dspark_split_rnn(rnn_w, w_sp, w_h, w_f, s);
        dspark_fold(head, n_vocab, out_w, out_b, fold, fold_b, s);
        dspark_screen_pack(head, kE, n_vocab, head_sc, s);
        dspark_screen_pack(fold, kDsRank, n_vocab, fold_sc, s);
        dspark_screen_order(fold_b, n_vocab, fold_bp, s);
        // the block's products rounded from their Q8_0 to NVFP4 planes in place (the planes are shorter), through a copy of
        // each: the drafts' precision only, every draft is verified exactly; about half the bytes a block reads
        const uint64_t ff = (uint64_t) geo.n_ff;
        void* tmp = nullptr;
        if (cudaMalloc(&tmp, q8_bytes(ff, kE)) != cudaSuccess) throw std::runtime_error("out of device memory rounding its products");
        for (DsparkLayer& L : layer) {
            bk::wgemm::q8_planes(L.k_ctx, kKV, kE, L.k_ctx_p, s);
            bk::wgemm::q8_planes(L.v_ctx, kKV, kE, L.v_ctx_p, s);
            const struct { void* w; uint64_t rows, K; } m[kFp4Mats] = {
                {L.wq, kQ, kE}, {L.wk, kKV, kE}, {L.wv, kKV, kE}, {L.wo, kE, kQ}, {L.gate, ff, kE}, {L.up, ff, kE}, {L.down, kE, ff}};
            for (int i = 0; i < kFp4Mats; ++i) {
                cudaMemcpyAsync(tmp, m[i].w, q8_bytes(m[i].rows, m[i].K), cudaMemcpyDeviceToDevice, s);
                bk::nvfp4p::from_q8(tmp, (int64_t) m[i].rows, (int64_t) m[i].K, m[i].w, L.fp4_scale + i, s);
            }
        }
        const cudaError_t e = cudaStreamSynchronize(s);
        cudaFree(tmp);
        if (e != cudaSuccess) throw std::runtime_error("rounding its products failed");
    } catch (const std::exception& e) {
        return {model::Status::Error, std::string("inkling2: the drafter: ") + e.what()};
    }
    if (cudaStreamSynchronize(s) != cudaSuccess) return {model::Status::Error, "inkling2: the drafter's folds failed"};
    // the projection as loaded is no longer read: its split columns are
    allocs_.erase(std::find(allocs_.begin(), allocs_.end(), (void*) rnn_w));
    cudaFree(rnn_w);
    bytes_ -= (uint64_t) kDsRnn * (2 * kDsRank + 2 * kE) * 2;
    rnn_w = nullptr;
    finished_ = true;
    return {};
}

DsHead DsparkWeights::head_args() const {
    DsHead h;
    h.head = head;
    h.fold = fold;
    h.fold_b = fold_b;
    h.fold_bp = fold_bp;
    h.head_sc = head_sc;
    h.head_planes = true;
    h.fold_sc = fold_sc;
    h.n_vocab = n_vocab;
    h.cap = logit_softcap;
    return h;
}

uint64_t dspark_tap_bytes(int tap_rows) { return 4ull * (uint64_t) std::max(tap_rows, kSlice) * kE * 4; }

uint64_t dspark_session_bytes(const Pack& pack, int64_t context) {
    const DsparkGeometry& d = pack.dspark;
    if (!d.present) return 0;
    const uint64_t E = kE, V = (uint64_t) pack.geo.n_vocab, ff = (uint64_t) d.n_ff, R = kDsBlock;
    uint64_t n = 3 * kSlice * E * 4 + kFrRing * E * 4 + 2 * kSlice * kKV * 4 + 64;        // a commit step's rows
    n += bk::native_q8_1_bytes(kE, kSlice);
    for (int l = 0; l < d.n_layers; ++l)                                           // the caches, fp16 K and V
        n += 2ull * (uint64_t) (d.swa[(size_t) l] ? ring_cells(d) : full_cells(context)) * kKV * 2;
    n += (2 * R * E + 3 * R * kQ) * 4;                                             // the block's rows
    n += 2 * bk::wgemm::act_bytes((int64_t) std::max<uint64_t>(ff, kQ)) + dspark_ws_bytes(d) + 1024;   // the window gemm's
    n += bk::gqa_decode_scratch_floats(std::max(ring_cells(d), full_cells(context)), kDsBlock, kHeads, kHeadDim) * 4;
    n += ((uint64_t) kDsDrafts * dspark_screen_rows((int64_t) V) + (uint64_t) kDsDrafts * kDsRnn + kDsRnn + 3 * kDsRank) * 4;   // the head's
    n += bk::native_q8_1_bytes(kE, kDsDrafts) + bk::native_q8_1_bytes(kDsRank, kDsDrafts) + dspark_head_scratch_bytes((int64_t) V) + 1024;
    return n;
}

// ---- the session's hook ---------------------------------------------------------------------------------------------------

DsparkHook::~DsparkHook() {
    for (cudaGraphExec_t g : g_stage_)
        if (g) cudaGraphExecDestroy(g);
    for (cudaGraphExec_t g : g_head_)
        if (g) cudaGraphExecDestroy(g);
    if (dump_) {
        basalt::log::info("basalt inkling2", "--diag-dump: %lld drafts, %lld screened picks differ from the exact head's",
                          (long long) dump_drafts_, (long long) dump_differ_);
        std::fclose(dump_);
    }
    for (void* p : allocs_) cudaFree(p);
    for (void* p : pinned_) cudaFreeHost(p);
}

void* DsparkHook::dev(size_t bytes, bool& ok) {
    void* p = nullptr;
    if (!ok || cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
        ok = false;
        return nullptr;
    }
    cudaMemset(p, 0, std::max<size_t>(bytes, 16));
    allocs_.push_back(p);
    return p;
}

bool DsparkHook::init(const DsparkWeights* w, const Geometry& g, int64_t context, int tap_rows, float* tap, const std::string& dump,
                      int64_t backed, std::string& err) {
    if (!w) return true;
    const DsparkGeometry& d = w->geo;
    const size_t E = kE, V = (size_t) w->n_vocab, ff = (size_t) d.n_ff, R = kDsBlock;
    bool ok = true;
    auto f = [&](size_t n) { return (float*) dev(n * 4, ok); };
    tap_rows_ = std::max(tap_rows, kSlice);
    context_ = context;
    slot_.assign(g.visits.size(), -1);
    for (int k = 0; k + 1 < kDsTargets; ++k) slot_[(size_t) d.targets[(size_t) k]] = k;
    tap_ = tap;
    mix_ = f(3 * kSlice * E);
    fr_ = f(kFrRing * E);
    pos3_ = (int32_t*) dev(kSlice * 3 * 4, ok);
    ring_bounds_ = (int32_t*) dev(2 * 4, ok);
    if (ok) {
        const int32_t rb[2] = {0, (int32_t) ring_cells(d)};
        if (basalt::core::device_copy(ring_bounds_, rb, sizeof(rb), cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
    }
    kb_ = f(kSlice * kKV);
    vb_ = f(kSlice * kKV);
    act_ = dev(bk::native_q8_1_bytes(kE, kSlice), ok);
    cache_.assign((size_t) d.n_layers, Cache{});
    for (int l = 0; l < d.n_layers; ++l) {
        Cache& c = cache_[(size_t) l];
        c.ring = d.swa[(size_t) l];
        c.cap = c.ring ? ring_cells(d) : full_cells(context);
        if (!c.ring && backed > 0) {   // the range reserved, the first positions' cells backed
            for (int j = 0; j < 2; ++j) {
                auto a = std::make_unique<core::ElasticArray>();
                if (ok && (!a->open((uint64_t) c.cap * kKV * 2, err) ||
                           !a->ensure((uint64_t) std::min(c.cap, full_cells(backed)) * kKV * 2, 0, err)))
                    ok = false;
                (j ? c.v : c.k) = (uint16_t*) a->data();
                full_.push_back(std::move(a));
            }
            continue;
        }
        c.k = (uint16_t*) dev((size_t) c.cap * kKV * 2, ok);
        c.v = (uint16_t*) dev((size_t) c.cap * kKV * 2, ok);
    }
    count_backed();
    x_ = f(R * E);
    h_ = f(R * E);
    q_ = f(R * kQ);
    qr_ = f(R * kQ);
    attn_ = f(R * kQ);
    act_a_ = dev(bk::wgemm::act_bytes(kQ), ok);
    act_h_ = dev(bk::wgemm::act_bytes((int64_t) ff), ok);
    {
        size_t pf = 0, tk = 0;
        ws_need(d, pf, tk);
        ws_.partial = f(std::max<size_t>(pf, 1));
        ws_.tickets = (int32_t*) dev(std::max<size_t>(tk, 1) * 4, ok);   // zeroed by dev()
        ws_.partial_floats = pf;
        ws_.tickets_n = tk;
    }
    pos3b_ = (int32_t*) dev(R * 3 * 4, ok);
    att_cells_ = std::max(ring_cells(d), full_cells(context));
    att_scr_ = f(bk::gqa_decode_scratch_floats(att_cells_, kDsBlock, kHeads, kHeadDim));
    approx_ = f((size_t) kDsDrafts * dspark_screen_rows((int64_t) V));
    hq_ = dev(bk::native_q8_1_bytes(kE, kDsDrafts), ok);
    ctx_h_ = f((size_t) kDsDrafts * kDsRnn);
    ctx_f_ = f(kDsRnn);
    state_ = f(2 * kDsRank);
    latent_ = f(kDsRank);
    lat_q_ = dev(bk::native_q8_1_bytes(kDsRank, kDsDrafts), ok);
    head_scr_ = dev(dspark_head_scratch_bytes((int64_t) V), ok);
    prev_ = (int32_t*) dev(4, ok);
    w1_row_ = (uint16_t*) dev(kDsRank * 2, ok);
    ids_d_ = (int32_t*) dev(kDsBlock * 4, ok);
    conf_d_ = f(kDsBlock);
    frc_ = f(E);
    meta_d_ = (int32_t*) dev(2 * 4, ok);
    if (ok && cudaMallocHost((void**) &meta_h_, 2 * 4) != cudaSuccess) ok = false;
    if (meta_h_) pinned_.push_back(meta_h_);
    if (ok && (cudaMallocHost((void**) &ids_h_, kDsBlock * 4) != cudaSuccess ||
               cudaMallocHost((void**) &conf_h_, kDsBlock * 4) != cudaSuccess))
        ok = false;
    if (ids_h_) pinned_.push_back(ids_h_);
    if (conf_h_) pinned_.push_back(conf_h_);
    if (!dump.empty()) {
        dump_logits_ = f((size_t) kDsDrafts * V);
        const std::string path = dump + "/dspark.bin";
        dump_ = std::fopen(path.c_str(), "wb");
        if (!dump_) basalt::log::warn("basalt inkling2", "--diag-dump: cannot write %s", path.c_str());
    }
    if (!ok) {
        err = "out of device memory for the drafter";
        return false;
    }
    w_ = w;
    return true;
}

int DsparkHook::slot_of_visit(int visit) const {
    return w_ && visit >= 0 && visit < (int) slot_.size() ? slot_[(size_t) visit] : -1;
}

void DsparkHook::commit(const int32_t* pos_d, int rows, void* stream, int row0) {
    if (!w_ || rows <= 0) return;
    const DsparkGeometry& d = w_->geo;
    DsMixArgs ma;
    ma.alpha = w_->alpha;
    for (int l = 0; l < d.n_layers; ++l) ma.norm[l] = w_->layer[(size_t) l].dflare_norm;
    ma.norm[kDsMixes - 1] = w_->markov_norm;
    ma.eps = d.rms_eps;
    for (int r0 = 0; r0 < rows; r0 += kSlice) {
        const int n = std::min(kSlice, rows - r0);
        dspark_mix(tap_ + (size_t) (row0 + r0) * kE, (int64_t) tap_rows_ * kE, n, ma, mix_, (int64_t) kSlice * kE, pos_d + r0, fr_, kFrRing,
                   pos3_, stream);
        for (int l = 0; l < d.n_layers; ++l) {
            const DsparkLayer& L = w_->layer[(size_t) l];
            const Cache& c = cache_[(size_t) l];
            DsCommitSegs segs;
            segs.k[0] = c.k;
            segs.v[0] = c.v;
            bk::native_quantize_q8_1(mix_ + (size_t) l * kSlice * kE, act_, kE, n, stream);
            bk::native_q8_planes_mmvq(L.k_ctx_p, act_, kb_, kE, kKV, n, stream);
            bk::native_q8_planes_mmvq(L.v_ctx_p, act_, vb_, kE, kKV, n, stream);
            dspark_kv_commit_batch(kb_, vb_, L.k_ctx_norm, d.rms_eps, d.rope_base, pos_d + r0, n, segs, c.cap, stream);
        }
    }
}

void DsparkHook::commit_prompt(const int32_t* pos_d, int rows, basalt::prefill::mmq::Context& mmq, const PromptScratch& s,
                               void* stream) {
    if (!w_ || rows <= 0) return;
    if (rows <= kSlice) {
        commit(pos_d, rows, stream);
        return;
    }
    namespace pm = basalt::prefill::mmq;
    const DsparkGeometry& d = w_->geo;
    DsMixArgs ma;
    ma.alpha = w_->alpha;
    for (int l = 0; l < d.n_layers; ++l) ma.norm[l] = w_->layer[(size_t) l].dflare_norm;
    ma.norm[kDsMixes - 1] = w_->markov_norm;
    ma.eps = d.rms_eps;
    const bk::RopeSpec rope = rope_of(d);
    dspark_mix(tap_, (int64_t) tap_rows_ * kE, rows, ma, s.mix, (int64_t) rows * kE, pos_d, fr_, kFrRing, s.pos3, stream);
    for (int l = 0; l < d.n_layers; ++l) {
        const DsparkLayer& L = w_->layer[(size_t) l];
        const Cache& c = cache_[(size_t) l];
        // A sliding-window layer only keeps its last c.cap cells; older rows of the chunk would be overwritten.
        const int keep = c.ring ? (int) std::min<int64_t>(rows, c.cap) : rows;
        const int r0 = rows - keep;
        const float* mix_l = s.mix + ((size_t) l * rows + r0) * kE;
        pm::quantize(mix_l, nullptr, s.xq, 8, kE, kE, keep, stream);
        pm::Product p;
        p.type = 8;
        p.w_rows = kKV;
        p.w_cols = kE;
        p.expert_bytes = pm::matrix_bytes(8, kKV, kE);
        p.n = 1;
        p.xq = s.xq;
        p.bounds = keep == rows ? s.bounds : ring_bounds_;
        p.ids = s.ident;
        p.total_rows = keep;
        p.max_rows = keep;
        p.ld_dst = kKV;
        p.w = static_cast<const uint8_t*>(L.k_ctx);
        p.dst = s.kb;
        mmq.run(p, stream);
        p.w = static_cast<const uint8_t*>(L.v_ctx);
        p.dst = s.vb;
        mmq.run(p, stream);
        bk::attn_qk_prepare(nullptr, s.kb, s.vb, nullptr, L.k_ctx_norm, d.rms_eps, rope, s.pos3 + (size_t) r0 * 3, keep, 0,
                            kKvHeads, nullptr, c.k, c.v, 0, stream, kHeadDim, false, c.cap, pos_d + r0);
    }
}

void DsparkHook::note(int64_t pos0, int rows) {
    if (!w_ || rows <= 0) return;
    if (pos0 > hi_ || pos0 + rows < lo_) lo_ = pos0;   // a gap or a jump back: the record starts again at this run
    else lo_ = std::min(lo_, pos0);
    hi_ = pos0 + rows;
    if (dump_) {   // 'C', pos0, rows, then each row's four features
        std::vector<float> h((size_t) kDsTargets * rows * kE), row((size_t) kDsTargets * kE);
        cudaDeviceSynchronize();
        for (int k = 0; k < kDsTargets; ++k)
            basalt::core::device_copy(h.data() + (size_t) k * rows * kE, tap(k), (size_t) rows * kE * 4, cudaMemcpyDeviceToHost);
        const int32_t n = rows;
        std::fputc('C', dump_);
        std::fwrite(&pos0, 8, 1, dump_);
        std::fwrite(&n, 4, 1, dump_);
        for (int r = 0; r < rows; ++r) {
            for (int k = 0; k < kDsTargets; ++k) std::memcpy(&row[(size_t) k * kE], &h[((size_t) k * rows + r) * kE], kE * 4);
            std::fwrite(row.data(), 4, row.size(), dump_);
        }
    }
}

// The block's stage k (0 .. layers): the post-attention half of layer k - 1 (k > 0), then the pre-attention half of layer k
// (k < layers) or the final norm and the head's `n` drafts (k = layers). Everything a stage reads of the block's anchor and
// position is on the device (meta_d_), so a stage is captured once and replayed.
void DsparkHook::stage(int k, int n, void* stream) {
    bk::PdlScope pdl(true);   // each kernel may start while the one before it finishes: a product stages its first weights
    const DsparkWeights& w = *w_;
    const DsparkGeometry& d = w.geo;
    if (k == 0) dspark_block_inputs(w.embedding, meta_d_, w.embed_norm, d.rms_eps, w.mask, kDsBlock, x_, pos3b_, prev_, state_, fr_,
                                    kFrRing, frc_, w.w1, w1_row_, stream);
    namespace wg = bk::wgemm;
    const auto mat = [](const void* p, int64_t rows, const float* scale, float* y, int64_t ld_y) {
        return wg::Matrix{(const uint8_t*) p, rows, scale, y, ld_y};
    };
    if (k > 0) {   // x += o attn; x += down(silu(gate z) * up z), z = post_norm(x)
        const DsparkLayer& L = w.layer[(size_t) k - 1];
        const float* sc = L.fp4_scale;
        wg::pack(attn_, kQ, kQ, kDsBlock, act_a_, stream);
        wg::Call o;
        o.fmt = wg::Format::Nvfp4;
        o.K = kQ;
        o.epi = wg::Epilogue::Add;
        o.m[0] = mat(L.wo, kE, sc + 3, x_, kE);
        wg::gemm(o, act_a_, kDsBlock, ws_, stream);
        wg::norm_pack(x_, L.post_norm, d.rms_eps, kE, kDsBlock, act_a_, nullptr, stream);
        wg::Call gu;
        gu.fmt = wg::Format::Nvfp4;
        gu.K = kE;
        gu.n = 2;
        gu.epi = wg::Epilogue::SwiGLU;
        gu.m[0] = mat(L.gate, d.n_ff, sc + 4, nullptr, d.n_ff);
        gu.m[1] = mat(L.up, d.n_ff, sc + 5, nullptr, d.n_ff);
        gu.act_out = act_h_;
        wg::gemm(gu, act_a_, kDsBlock, ws_, stream);
        wg::Call dn;
        dn.fmt = wg::Format::Nvfp4;
        dn.K = d.n_ff;
        dn.epi = wg::Epilogue::Add;
        dn.m[0] = mat(L.down, kE, sc + 6, x_, kE);
        wg::gemm(dn, act_h_, kDsBlock, ws_, stream);
    }
    if (k < d.n_layers) {
        const DsparkLayer& L = w.layer[(size_t) k];
        const Cache& c = cache_[(size_t) k];
        const float* sc = L.fp4_scale;
        wg::norm_pack(x_, L.attn_norm, d.rms_eps, kE, kDsBlock, act_a_, nullptr, stream);
        wg::Call qkv;
        qkv.fmt = wg::Format::Nvfp4;
        qkv.K = kE;
        qkv.n = 3;
        qkv.m[0] = mat(L.wq, kQ, sc + 0, q_, kQ);
        qkv.m[1] = mat(L.wk, kKV, sc + 1, kb_, kKV);
        qkv.m[2] = mat(L.wv, kKV, sc + 2, vb_, kKV);
        wg::gemm(qkv, act_a_, kDsBlock, ws_, stream);
        bk::attn_qk_prepare(q_, kb_, vb_, L.q_norm, L.k_norm, d.rms_eps, rope_of(d), pos3b_, kDsBlock, kHeads, kKvHeads, qr_, c.k, c.v,
                            0, stream, kHeadDim, false, c.cap, meta_d_ + 1);
        return;
    }
    bk::rms_norm_rows(x_, w.norm, kDsBlock, kE, d.rms_eps, h_, stream);
    // the head: the screen's head term of the draft rows (block rows 1..n), the recurrence's hidden and context terms
    const float* hd = h_ + kE;
    const DsHead head = w.head_args();
    bk::native_quantize_q8_1(hd, hq_, kE, n, stream);
    dspark_head_base(head, hq_, n, approx_, stream);
    bk::bf16_gemv_fp32_mmvf_multi(hd, kE, w.w_h, ctx_h_, kDsRnn, kE, kDsRnn, n, stream);
    bk::bf16_gemv_fp32_mmvf_multi(frc_, kE, w.w_f, ctx_f_, kDsRnn, kE, kDsRnn, 1, stream);
    for (int i = 0; i < n; ++i) {
        dspark_markov_step(ctx_h_ + (size_t) i * kDsRnn, ctx_f_, w.rnn_b, w.w_sp, w1_row_, state_ + (i & 1) * kDsRank,
                           state_ + ((i + 1) & 1) * kDsRank, latent_, stream);
        void* lq = static_cast<char*>(lat_q_) + (size_t) i * bk::native_q8_1_bytes(kDsRank, 1);
        bk::native_quantize_q8_1(latent_, lq, kDsRank, 1, stream);
        if (dump_ && dumped_ < kDumpBlocks) dspark_head_exact(head, hq_, lq, i, dump_logits_ + (size_t) i * w.n_vocab, stream);
        dspark_head_pick(head, hq_, lq, approx_, i, head_scr_, prev_, ids_d_, conf_d_, w.w1, w1_row_, stream);
    }
    cudaMemcpyAsync(ids_h_, ids_d_, (size_t) n * 4, cudaMemcpyDeviceToHost, (cudaStream_t) stream);
    cudaMemcpyAsync(conf_h_, conf_d_, (size_t) n * 4, cudaMemcpyDeviceToHost, (cudaStream_t) stream);
}

// Stage k as a graph: captured the first time, then launched. False when the stream cannot capture (the stage then ran
// directly, and every later block runs its stages directly).
bool DsparkHook::stage_graph(int k, int n, void* stream) {
    cudaGraphExec_t& g = k < w_->geo.n_layers ? g_stage_[k] : g_head_[n];
    const cudaStream_t st = (cudaStream_t) stream;
    if (!g) {
        cudaGraph_t graph = nullptr;
        if (cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        stage(k, n, stream);
        const cudaError_t ce = cudaStreamEndCapture(st, &graph);
        const cudaError_t ci = ce == cudaSuccess ? cudaGraphInstantiateWithFlags(&g, graph, 0) : ce;
        if (graph) cudaGraphDestroy(graph);
        if (ci != cudaSuccess) {
            (void) cudaGetLastError();
            g = nullptr;
            basalt::log::warn("basalt inkling2", "drafter: a graph of its block failed (%s); it runs without graphs", cudaGetErrorString(ci));
            return false;
        }
    }
    return cudaGraphLaunch(g, st) == cudaSuccess;
}

model::Result DsparkHook::draft_begin(int32_t anchor, int64_t pos, int n, void* stream) {
    if (!w_) return {model::Status::NotImplemented, "inkling2: the pack has no drafter"};
    pending_ = 0;
    batch_owner_ = nullptr;
    batch_slot_ = -1;
    n = std::min(n, kDsDrafts);
    // the Markov head's context is the row before the anchor's: kept for the last kFrRing positions run
    if (n < 1 || pos < 1 || pos - 1 >= hi_ || pos - 1 < std::max(lo_, hi_ - kFrRing) || pos + kDsBlock > backed_cells_)
        return {};
    const DsparkGeometry& d = w_->geo;
    meta_h_[0] = anchor;
    meta_h_[1] = (int32_t) pos;
    cudaMemcpyAsync(meta_d_, meta_h_, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) stream);
    // the stages captured as graphs (not with --diag-dump, whose head writes more); the attention between them runs
    // directly: its window follows the position
    if (dump_ || !stream) graphs_ = false;
    try {
        for (int k = 0; k <= d.n_layers; ++k) {
            if (k > 0) {
                const Cache& c = cache_[(size_t) k - 1];
                // a sliding-window layer sees its window before the block, the full one every position, and both the block
                const int64_t window = c.ring ? (int64_t) d.window + 1 : pos + kDsBlock;
                // the splits from the cells the block reads, not from the scratch's capacity (this call is not captured): a
                // short context runs few splits; the drafts only shape the speed
                const int64_t cells = pos + kDsBlock - std::max<int64_t>(0, pos - window + 1) + 64;
                bk::gqa_window_attn(qr_, c.k, c.v, kHeads, kKvHeads, kHeadDim, pos, kDsBlock, pos + kDsBlock, window, c.cap, att_scr_,
                                    std::min(att_cells_, cells), attn_, stream, true);
            }
            if (!graphs_ || !stage_graph(k, n, stream)) {
                graphs_ = false;
                stage(k, n, stream);
            }
        }
    } catch (const std::exception& e) {
        return {model::Status::Error, std::string("inkling2: the drafter: ") + e.what()};
    }
    pending_ = n;
    block_pos_ = pos;
    block_anchor_ = anchor;
    return {};
}

model::Result DsparkHook::draft_end(int32_t* ids, float* conf, int max_n, int& n, const model::ModelSession::DraftNeed&,
                                    void* stream) {
    n = 0;
    if (!w_) return {model::Status::NotImplemented, "inkling2: the pack has no drafter"};
    if (pending_ == 0) return {};
    if (batch_owner_ != nullptr) {
        DsparkBatch* owner = batch_owner_;
        const int slot = batch_slot_, pend = pending_;
        batch_owner_ = nullptr;
        batch_slot_ = -1;
        pending_ = 0;
        return owner->take(slot, pend, ids, conf, max_n, n);
    }
    if (cudaStreamSynchronize((cudaStream_t) stream) != cudaSuccess)
        return {model::Status::Error, std::string("inkling2: the drafter: ") + cudaGetErrorString(cudaGetLastError())};
    n = std::min(pending_, max_n);
    for (int i = 0; i < n; ++i) {
        ids[i] = ids_h_[i];
        conf[i] = conf_h_[i];
    }
    if (dump_ && dumped_ < kDumpBlocks) {   // 'B', pos, anchor, drafts, the block's final rows, each draft's logits, the drafts and chances
        const size_t V = (size_t) w_->n_vocab;
        const int32_t m = pending_;
        std::vector<float> hf((size_t) kDsBlock * kE), base((size_t) m * V);
        basalt::core::device_copy(hf.data(), h_, hf.size() * 4, cudaMemcpyDeviceToHost);
        basalt::core::device_copy(base.data(), dump_logits_, base.size() * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < m; ++i) {   // the screened pick against the exact head's first argmax
            const float* l = base.data() + (size_t) i * V;
            const size_t best = (size_t) (std::max_element(l, l + V) - l);
            ++dump_drafts_;
            if ((int32_t) best == ids_h_[i]) continue;
            ++dump_differ_;
            double z = 0;
            for (size_t v = 0; v < V; ++v) z += std::exp((double) l[v] - l[best]);
            basalt::log::info("basalt inkling2", "--diag-dump: block at %lld draft %d: screened %d (logit %.4f), exact %zu (logit %.4f, p %.3f)",
                              (long long) block_pos_, i, ids_h_[i], l[ids_h_[i]], best, l[best], 1.0 / z);
        }
        std::fputc('B', dump_);
        std::fwrite(&block_pos_, 8, 1, dump_);
        std::fwrite(&block_anchor_, 4, 1, dump_);
        std::fwrite(&m, 4, 1, dump_);
        std::fwrite(hf.data(), 4, hf.size(), dump_);
        std::fwrite(base.data(), 4, base.size(), dump_);
        std::fwrite(ids_h_, 4, (size_t) m, dump_);
        std::fwrite(conf_h_, 4, (size_t) m, dump_);
        std::fflush(dump_);
        ++dumped_;
    }
    pending_ = 0;
    return {};
}

size_t DsparkHook::running_bytes() const {
    size_t n = 0;
    for (const Cache& c : cache_)
        if (c.ring) n += 2 * (size_t) c.cap * kKV * 2;
    return n;
}

void DsparkHook::running_regions(core::RunningState& st) const {
    for (const Cache& c : cache_) {
        if (!c.ring) continue;
        st.regions.push_back({c.k, (size_t) c.cap * kKV * 2});
        st.regions.push_back({c.v, (size_t) c.cap * kKV * 2});
    }
}

size_t DsparkHook::cells_bytes(int64_t positions) const {
    size_t n = 0;
    for (const Cache& c : cache_)
        if (!c.ring) n += 2 * (size_t) positions * kKV * 2;
    return n;
}

void DsparkHook::count_backed() {
    backed_cells_ = full_cells(context_);
    for (const auto& a : full_) backed_cells_ = std::min<int64_t>(backed_cells_, (int64_t) (a->backed() / (kKV * 2)));
}

uint64_t DsparkHook::grow_bytes(int64_t positions) const {
    uint64_t n = 0;
    for (const auto& a : full_) n += a->growth((uint64_t) std::min(full_cells(context_), full_cells(positions)) * kKV * 2);
    return n;
}

bool DsparkHook::ensure(int64_t positions, std::string& err) {
    if (full_.empty() || full_cells(positions) <= backed_cells_) return true;
    const uint64_t bytes = (uint64_t) std::min(full_cells(context_), full_cells(positions)) * kKV * 2;
    for (const auto& a : full_)
        if (!a->ensure(bytes, 0, err)) {
            err = "the drafter's caches cannot grow to " + std::to_string(positions) + " positions: " + err;
            return false;
        }
    count_backed();
    return true;
}

void DsparkHook::trim(int64_t positions) {
    for (const auto& a : full_) a->trim((uint64_t) full_cells(positions) * kKV * 2);
    count_backed();
}

bool DsparkHook::save_cells(std::vector<core::ConversationKv>& kv, int64_t positions, void* stream) const {
    if (positions < 0 || positions > backed_cells_) return false;
    cudaStreamSynchronize((cudaStream_t) stream);
    for (const Cache& c : cache_) {
        if (c.ring) continue;
        core::ConversationKv k;
        k.format = 3;   // fp16 rows, cell-major: [cell][kv head][head width]
        k.page_size = 1;
        k.cells = positions;
        k.heads = kKvHeads;
        k.head_dim = kHeadDim;
        k.k.resize_for_overwrite((size_t) positions * kKV * 2);
        k.v.resize_for_overwrite((size_t) positions * kKV * 2);
        const auto down = [&](const uint16_t* dev, core::ConversationBuffer& to) {
            return to.visit(0, to.size(), [&](uint8_t* p, size_t n, size_t at) {
                return basalt::core::device_copy(p, (const uint8_t*) dev + at, n, cudaMemcpyDeviceToHost) == cudaSuccess;
            });
        };
        if (!down(c.k, k.k) || !down(c.v, k.v)) return false;
        kv.push_back(std::move(k));
    }
    return true;
}

bool DsparkHook::restore_cells(const std::vector<core::ConversationKv>& kv, size_t& at, int64_t positions, void* stream) {
    if (positions < 0 || positions > backed_cells_) return false;
    cudaStreamSynchronize((cudaStream_t) stream);
    const size_t run = (size_t) positions * kKV * sizeof(uint16_t);
    for (const Cache& c : cache_) {
        if (c.ring) continue;
        if (at >= kv.size()) return false;
        const core::ConversationKv& k = kv[at];
        if (k.format != 3 || k.cells != positions || k.heads != kKvHeads || k.head_dim != kHeadDim ||
            k.k.size() != run || k.v.size() != run || positions > c.cap)
            return false;
        const auto up = [&](uint16_t* dev, const core::ConversationBuffer& from) {
            return from.visit(0, run, [&](const uint8_t* p, size_t n, size_t off) {
                return basalt::core::device_copy((uint8_t*) dev + off, p, n, cudaMemcpyHostToDevice) == cudaSuccess;
            });
        };
        if (!up(c.k, k.k) || !up(c.v, k.v)) return false;
        ++at;
    }
    return true;
}

// ---- multi-session batched DSpark commit & draft --------------------------------------------------------------------------

DsparkBatch::~DsparkBatch() {
    for (void* p : allocs_) cudaFree(p);
    for (void* p : pinned_) cudaFreeHost(p);
}

void* DsparkBatch::dev(size_t bytes, bool& ok) {
    void* p = nullptr;
    if (!ok || cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
        ok = false;
        return nullptr;
    }
    cudaMemset(p, 0, std::max<size_t>(bytes, 16));
    allocs_.push_back(p);
    return p;
}

bool DsparkBatch::init(const DsparkWeights* w, int max_sessions, int max_rows, int tap_rows, float* tap, int64_t context,
                       std::string& err) {
    if (!w || max_sessions <= 1) return true;
    const DsparkGeometry& d = w->geo;
    const size_t E = kE, V = (size_t) w->n_vocab, ff = (size_t) d.n_ff;
    const int S = std::min(max_sessions, bk::kWindowSegsMax);
    const int CR = std::min(S, kChunkSlots) * kDsBlock;   // <= 32 == wgemm::kRows
    bool ok = true;
    auto f = [&](size_t n) { return (float*) dev(n * 4, ok); };
    max_sessions_ = S;
    max_rows_ = max_rows;
    tap_rows_ = std::max(tap_rows, kSlice);
    tap_ = tap;
    // commit_batch scratch
    c_mix_ = f((size_t) 3 * max_rows * E);
    c_kb_ = f((size_t) max_rows * kKV);
    c_vb_ = f((size_t) max_rows * kKV);
    c_act_ = dev(bk::native_q8_1_bytes(kE, max_rows), ok);
    // block chunk scratch
    x_ = f((size_t) CR * E);
    h_ = f((size_t) CR * E);
    q_ = f((size_t) CR * kQ);
    kb_ = f((size_t) CR * kKV);
    vb_ = f((size_t) CR * kKV);
    qr_ = f((size_t) CR * kQ);
    attn_ = f((size_t) CR * kQ);
    act_a_ = dev(bk::wgemm::act_bytes(kQ), ok);
    act_h_ = dev(bk::wgemm::act_bytes((int64_t) ff), ok);
    {
        size_t pf = 0, tk = 0;
        ws_need(d, pf, tk);
        ws_.partial = f(std::max<size_t>(pf, 1));
        ws_.tickets = (int32_t*) dev(std::max<size_t>(tk, 1) * 4, ok);
        ws_.partial_floats = pf;
        ws_.tickets_n = tk;
    }
    att_cells_ = std::max(ring_cells(d), full_cells(context));
    att_scr_ = f(bk::gqa_decode_scratch_floats(att_cells_, CR, kHeads, kHeadDim));
    // head batch scratch
    hd_ = f((size_t) S * kDsDrafts * E);
    hq_ = dev(bk::native_q8_1_bytes(kE, S * kDsDrafts), ok);
    approx_ = f((size_t) S * kDsDrafts * dspark_screen_rows((int64_t) V));
    ctx_h_ = f((size_t) S * kDsDrafts * kDsRnn);
    ctx_f_ = f((size_t) S * kDsRnn);
    state_ = f((size_t) S * 2 * kDsRank);
    latent_ = f((size_t) S * kDsRank);
    lat_q_ = dev(bk::native_q8_1_bytes(kDsRank, S * kDsDrafts), ok);
    head_scr_stride_ = (dspark_head_scratch_bytes((int64_t) V) + 255) & ~(size_t) 255;
    head_scr_ = dev((size_t) S * head_scr_stride_, ok);
    prev_ = (int32_t*) dev((size_t) S * 4, ok);
    w1_row_ = (uint16_t*) dev((size_t) S * kDsRank * 2, ok);
    frc_ = f((size_t) S * E);
    ids_d_ = (int32_t*) dev((size_t) S * kDsBlock * 4, ok);
    conf_d_ = f((size_t) S * kDsBlock);
    meta_d_ = (int32_t*) dev((size_t) S * 2 * 4, ok);
    if (ok && cudaMallocHost((void**) &meta_h_, (size_t) S * 2 * 4) != cudaSuccess) ok = false;
    if (meta_h_) pinned_.push_back(meta_h_);
    if (ok && cudaMallocHost((void**) &ids_h_, (size_t) S * kDsBlock * 4) != cudaSuccess) ok = false;
    if (ids_h_) pinned_.push_back(ids_h_);
    if (ok && cudaMallocHost((void**) &conf_h_, (size_t) S * kDsBlock * 4) != cudaSuccess) ok = false;
    if (conf_h_) pinned_.push_back(conf_h_);
    if (!ok) {
        cudaGetLastError();
        err = "out of device memory for the batched drafter";
        return false;
    }
    w_ = w;
    return true;
}

void DsparkBatch::commit(DsparkHook* const* hooks, int n_seg, const bk::WindowSegs* at_d, const int32_t* pos_d, int rows,
                         void* stream) {
    if (!w_ || rows <= 0 || n_seg <= 0) return;
    const DsparkGeometry& d = w_->geo;
    DsMixArgs ma;
    ma.alpha = w_->alpha;
    for (int l = 0; l < d.n_layers; ++l) ma.norm[l] = w_->layer[(size_t) l].dflare_norm;
    ma.norm[kDsMixes - 1] = w_->markov_norm;
    ma.eps = d.rms_eps;
    DsCommitSegs segs;
    segs.at_d = at_d;
    for (int i = 0; i < n_seg; ++i) segs.fr[i] = hooks[i]->fr_;
    dspark_mix_batch(tap_, (int64_t) tap_rows_ * kE, rows, ma, c_mix_, (int64_t) rows * kE, pos_d, segs, kFrRing, stream);
    for (int l = 0; l < d.n_layers; ++l) {
        const DsparkLayer& L = w_->layer[(size_t) l];
        const int64_t cap = hooks[0]->cache_[(size_t) l].cap;
        for (int i = 0; i < n_seg; ++i) {
            segs.k[i] = hooks[i]->cache_[(size_t) l].k;
            segs.v[i] = hooks[i]->cache_[(size_t) l].v;
        }
        bk::native_quantize_q8_1(c_mix_ + (size_t) l * rows * kE, c_act_, kE, rows, stream);
        bk::native_q8_planes_mmvq(L.k_ctx_p, c_act_, c_kb_, kE, kKV, rows, stream);
        bk::native_q8_planes_mmvq(L.v_ctx_p, c_act_, c_vb_, kE, kKV, rows, stream);
        dspark_kv_commit_batch(c_kb_, c_vb_, L.k_ctx_norm, d.rms_eps, d.rope_base, pos_d, rows, segs, cap, stream);
    }
}

model::Result DsparkBatch::draft(DsparkHook* const* hooks, const int32_t* anchors, const int64_t* pos, const int* n,
                                 int count, void* stream) {
    if (!w_) return {model::Status::NotImplemented, "inkling2: the pack has no drafter"};
    if (count <= 0) return {};
    if (count > max_sessions_) {
        for (int i = 0; i < count; ++i)
            if (model::Result r = hooks[i]->draft_begin(anchors[i], pos[i], n[i], stream); !r.ok()) return r;
        return {};
    }
    DsparkHook* act[bk::kWindowSegsMax] = {};
    int act_n[bk::kWindowSegsMax] = {};
    int64_t act_pos[bk::kWindowSegsMax] = {};
    int B = 0, n_max = 0;
    for (int i = 0; i < count; ++i) {
        DsparkHook& hk = *hooks[i];
        hk.pending_ = 0;
        hk.batch_owner_ = nullptr;
        hk.batch_slot_ = -1;
        const int ni = std::min(n[i], kDsDrafts);
        const int64_t pi = pos[i];
        if (ni < 1 || pi < 1 || pi - 1 >= hk.hi_ || pi - 1 < std::max(hk.lo_, hk.hi_ - kFrRing) ||
            pi + kDsBlock > hk.backed_cells_)
            continue;
        if (hk.dump_) {
            for (int j = 0; j < count; ++j)
                if (model::Result r = hooks[j]->draft_begin(anchors[j], pos[j], n[j], stream); !r.ok()) return r;
            return {};
        }
        meta_h_[2 * B] = anchors[i];
        meta_h_[2 * B + 1] = (int32_t) pi;
        act[B] = &hk;
        act_n[B] = ni;
        act_pos[B] = pi;
        n_max = std::max(n_max, ni);
        ++B;
    }
    if (B == 0) return {};
    if (B == 1) return act[0]->draft_begin(meta_h_[0], act_pos[0], act_n[0], stream);
    const DsparkWeights& w = *w_;
    const DsparkGeometry& d = w.geo;
    const cudaStream_t st = (cudaStream_t) stream;
    cudaMemcpyAsync(meta_d_, meta_h_, (size_t) B * 2 * sizeof(int32_t), cudaMemcpyHostToDevice, st);
    namespace wg = bk::wgemm;
    const auto mat = [](const void* p, int64_t rows, const float* scale, float* y, int64_t ld_y) {
        return wg::Matrix{(const uint8_t*) p, rows, scale, y, ld_y};
    };
    try {
        for (int s0 = 0; s0 < B; s0 += kChunkSlots) {
            const int cn = std::min(kChunkSlots, B - s0);
            const int R = cn * kDsBlock;
            DsBatchFr frs;
            for (int c = 0; c < cn; ++c) frs.fr[c] = act[s0 + c]->fr_;
            dspark_block_inputs_batch(w.embedding, meta_d_ + 2 * s0, w.embed_norm, d.rms_eps, w.mask, kDsBlock, cn, x_,
                                      prev_ + s0, state_ + (size_t) s0 * 2 * kDsRank, frs, kFrRing, frc_ + (size_t) s0 * kE,
                                      w.w1, w1_row_ + (size_t) s0 * kDsRank, st);
            for (int l = 0; l < d.n_layers; ++l) {
                const DsparkLayer& L = w.layer[(size_t) l];
                const float* sc = L.fp4_scale;
                wg::norm_pack(x_, L.attn_norm, d.rms_eps, kE, R, act_a_, nullptr, st);
                wg::Call qkv;
                qkv.fmt = wg::Format::Nvfp4;
                qkv.K = kE;
                qkv.n = 3;
                qkv.m[0] = mat(L.wq, kQ, sc + 0, q_, kQ);
                qkv.m[1] = mat(L.wk, kKV, sc + 1, kb_, kKV);
                qkv.m[2] = mat(L.wv, kKV, sc + 2, vb_, kKV);
                wg::gemm(qkv, act_a_, R, ws_, st);
                DsBatchKv kv;
                bk::GqaWindowSeg gsegs[kChunkSlots] = {};
                int64_t max_cells = 0;
                const int64_t cap = act[s0]->cache_[(size_t) l].cap;
                const bool ring = act[s0]->cache_[(size_t) l].ring;
                for (int c = 0; c < cn; ++c) {
                    const DsparkHook::Cache& cc = act[s0 + c]->cache_[(size_t) l];
                    kv.k[c] = cc.k;
                    kv.v[c] = cc.v;
                    const int64_t p0 = act_pos[s0 + c];
                    const int64_t win = ring ? (int64_t) d.window + 1 : p0 + kDsBlock;
                    const int64_t cells = p0 + kDsBlock - std::max<int64_t>(0, p0 - win + 1) + 64;
                    max_cells = std::max(max_cells, std::min(att_cells_, cells));
                    gsegs[c] = {cc.k, cc.v, p0, kDsBlock, p0 + kDsBlock, win, cc.cap, true};
                }
                dspark_block_qk_prep_batch(q_, kb_, vb_, L.q_norm, L.k_norm, d.rms_eps, d.rope_base, meta_d_ + 2 * s0,
                                           kDsBlock, cn, qr_, kv, cap, st);
                bk::gqa_window_attn_segs(qr_, kHeads, kKvHeads, kHeadDim, gsegs, cn, att_scr_, max_cells, attn_, st);
                wg::pack(attn_, kQ, kQ, R, act_a_, st);
                wg::Call o;
                o.fmt = wg::Format::Nvfp4;
                o.K = kQ;
                o.epi = wg::Epilogue::Add;
                o.m[0] = mat(L.wo, kE, sc + 3, x_, kE);
                wg::gemm(o, act_a_, R, ws_, st);
                wg::norm_pack(x_, L.post_norm, d.rms_eps, kE, R, act_a_, nullptr, st);
                wg::Call gu;
                gu.fmt = wg::Format::Nvfp4;
                gu.K = kE;
                gu.n = 2;
                gu.epi = wg::Epilogue::SwiGLU;
                gu.m[0] = mat(L.gate, d.n_ff, sc + 4, nullptr, d.n_ff);
                gu.m[1] = mat(L.up, d.n_ff, sc + 5, nullptr, d.n_ff);
                gu.act_out = act_h_;
                wg::gemm(gu, act_a_, R, ws_, st);
                wg::Call dn;
                dn.fmt = wg::Format::Nvfp4;
                dn.K = d.n_ff;
                dn.epi = wg::Epilogue::Add;
                dn.m[0] = mat(L.down, kE, sc + 6, x_, kE);
                wg::gemm(dn, act_h_, R, ws_, st);
            }
            bk::rms_norm_rows(x_, w.norm, R, kE, d.rms_eps, h_, st);
            dspark_pack_draft_rows(h_, kDsBlock, cn, s0, n_max, hd_, st);
        }
        const DsHead head = w.head_args();
        bk::native_quantize_q8_1(hd_, hq_, kE, B * n_max, st);
        dspark_head_base_batch(head, hq_, B, n_max, approx_, st);
        bk::bf16_gemv_fp32_mmvf_cols(hd_, w.w_h, ctx_h_, kE, kDsRnn, B * n_max, st);
        bk::bf16_gemv_fp32_mmvf_cols(frc_, w.w_f, ctx_f_, kE, kDsRnn, B, st);
        const size_t lq_step_bytes = (size_t) B * bk::native_q8_1_bytes(kDsRank, 1);
        for (int i = 0; i < n_max; ++i) {
            dspark_markov_step_batch(ctx_h_, ctx_f_, w.rnn_b, w.w_sp, w1_row_, state_, i, n_max, B, latent_, st);
            void* lq_step = static_cast<char*>(lat_q_) + (size_t) i * lq_step_bytes;
            bk::native_quantize_q8_1(latent_, lq_step, kDsRank, B, st);
            dspark_head_pick_batch(head, hq_, lq_step, approx_, i, n_max, B, head_scr_, head_scr_stride_, prev_, ids_d_,
                                   conf_d_, w.w1, w1_row_, st);
        }
        cudaMemcpyAsync(ids_h_, ids_d_, (size_t) B * kDsBlock * 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(conf_h_, conf_d_, (size_t) B * kDsBlock * 4, cudaMemcpyDeviceToHost, st);
    } catch (const std::exception& e) {
        return {model::Status::Error, std::string("inkling2: the drafter: ") + e.what()};
    }
    sync_stream_ = stream;
    synced_ = false;
    for (int b = 0; b < B; ++b) {
        act[b]->pending_ = act_n[b];
        act[b]->batch_owner_ = this;
        act[b]->batch_slot_ = b;
    }
    return {};
}

model::Result DsparkBatch::take(int slot, int pending, int32_t* ids, float* conf, int max_n, int& n) {
    n = 0;
    if (!synced_) {
        if (cudaStreamSynchronize((cudaStream_t) sync_stream_) != cudaSuccess)
            return {model::Status::Error, std::string("inkling2: the drafter: ") + cudaGetErrorString(cudaGetLastError())};
        synced_ = true;
    }
    n = std::min(pending, max_n);
    const int32_t* si = ids_h_ + (size_t) slot * kDsBlock;
    const float* sc = conf_h_ + (size_t) slot * kDsBlock;
    for (int i = 0; i < n; ++i) {
        ids[i] = si[i];
        conf[i] = sc[i];
    }
    return {};
}

// ---- flags ----------------------------------------------------------------------------------------------------------------

bool dspark_parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o) {
    if (a == "--spec") {
        o.spec = std::atoi(args.next("--spec"));
        o.spec_given = true;
    } else if (a == "--suffix-draft") {
        o.suffix_draft = std::atoi(args.next("--suffix-draft"));
    } else if (a == "--draft-log") {
        core::diagnostics().draft_log = args.next("--draft-log");
    } else {
        return false;
    }
    return true;
}

void dspark_finish_flags(program::CommonOptions& o) {
    core::ModelFile file;
    std::string err;
    const core::Json* kv = !o.model.empty() && file.open(o.model, err) && file.load_meta(err) ? file.meta().get("gguf_kv") : nullptr;
    const bool present = kv != nullptr && kv->get("inkling2.dspark.block_count") != nullptr;
    // the pack's drafter offers a whole block (the window policy takes the rows that pay) unless --spec caps it
    if (!o.spec_given) o.spec = present ? (o.suffix_draft > 0 ? kDecodeRows : kDsBlock) : 0;
    if (o.spec >= 2 && !present) {
        basalt::log::warn("basalt inkling2", "--spec %d off: the pack has no drafter", o.spec);
        o.spec = 0;
    }
}

}  // namespace basalt::models::inkling2
