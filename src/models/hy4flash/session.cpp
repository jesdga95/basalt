// src/models/hy4flash/session.cpp - one conversation of Hy4-flash: its latent caches, indexer rows and pooled keys, and the
// forward of a run of rows that every decode window (up to 8 rows) and every prompt chunk (up to kMaxChunk) goes through.
//
// Per layer: the Engram step (on its layers), the attention's hyper-connection pre-mix and norm, the absorbed MLA over the
// shared latent (a ring on the window layers, the indexer's selection on the others), the gate, the output projection and its
// post-norm, the stream expansion, and the same for the FFN: the dense MLP, or the latent MoE (router, routed_down, the experts
// the tier places in groups of up to 8 rows beside the shared expert, routed_up). A row's result does not depend on the other
// rows of its call: the dense products take 8 rows at most per launch with the single-row kernels' reduction order, the
// attention kernels take a block per row, and the experts are run per row.
#include "basalt/core/conversation_cache.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/core/elastic_array.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/elastic_kv.hpp"
#include "basalt/core/expert_tier.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/mhc.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/prefill/cpu_split.hpp"
#include "basalt/program/expert_spill.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/log.hpp"
#include "basalt/model/batch_driver.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/models/hy4flash/kernels.hpp"
#include "models/hy4flash/context_policy.hpp"
#include "models/hy4flash/dfly.hpp"   // DFLY HOOK
#include "models/hy4flash/engram.hpp"
#include "models/hy4flash/model.hpp"
#include "models/hy4flash/vision_encoder.hpp"   // [vision]

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <cstring>
#include <memory>
#include <set>
#include <utility>
#include <string>
#include <atomic>
#include <thread>
#include <vector>
#include <condition_variable>
#include <mutex>

namespace basalt::models::hy4flash {
namespace {

namespace bk = basalt::kernels;

constexpr int kRowWidth = 576;                 // a cache row: 512 latent values and 64 rotated
constexpr int kMaxChunk = 16384;               // rows a prompt chunk may carry (the residual streams and the experts' pairs)
constexpr int kAttRows = 2048;                 // rows of one attention step inside a chunk (the rings are sized for it)
constexpr int kStageExperts = 16;              // experts of one prompt launch (a blob each in the staging)
constexpr int kLaunchRows = 16384;              // most (row, expert) pairs of one prompt launch
constexpr int kMaxWindow = 8;                  // rows of a decode window, and of one expert group
constexpr int kRawRing = kAttRows + 16;        // the indexer's rows of the last positions (a multiple of the pool size)
constexpr int64_t kGrowPositions = 16384;      // positions the caches are backed in
constexpr uint64_t kModuleRoom = 7ull << 27;   // kernel modules still to load when the tier opens
constexpr uint64_t kPromptRoom = 5ull << 29;   // the prompt path's workspaces beside the homes
constexpr int kLookupK = core::ExpertTier::kMaxK;
constexpr int kTypeNvfp4 = 40;                 // ggml's NVFP4: blobs end in the FP32 scales {s_gate, s_up, s_down, 0}

// One group's GPU plan for the grouped expert kernel, uploaded with a single copy.
struct WindowPlan {
    unsigned long long ptr[kLookupK];
    int32_t start[kLookupK + 2];
    int32_t ng[4];
    int32_t dst[kLookupK];
    int32_t tok[kLookupK];
};
// The hits' plan and every routed row's pointer (a slot's output or the CPU's mapped row), put in one launch.
struct WindowPut {
    WindowPlan plan;
    unsigned long long rows[kLookupK];
};
static_assert(sizeof(WindowPut) <= 4 * kPutWords, "a window's plan fits put_words");

// The buffers that scale with the rows of a call: a decode window's, or a prompt chunk's (opened while a prompt runs, with the tier's
// slots lent to them).
struct Bufs {
    int rows = 0, arows = 0, steps = 0;   // the chunk's rows, the rows of an attention step, and the steps of a chunk
    std::vector<void*> dev, host;
    float *X = nullptr, *Xn = nullptr, *coef = nullptr, *hcws = nullptr, *e = nullptr, *cur = nullptr, *xn = nullptr, *attn = nullptr,
          *attn_raw = nullptr, *mlp = nullptr, *mlp2 = nullptr, *qa = nullptr, *qan = nullptr, *q = nullptr, *kvraw = nullptr,
          *qabs = nullptr, *qcur = nullptr, *olat = nullptr, *vout = nullptr, *gate = nullptr, *ik = nullptr, *ikn = nullptr,
          *ivt = nullptr, *igt = nullptr, *iq = nullptr, *iw = nullptr, *gu = nullptr, *h = nullptr, *rlog = nullptr, *rw = nullptr,
          *xlat = nullptr, *lat_out = nullptr, *moe_out = nullptr, *sh = nullptr, *erows = nullptr, *ekv = nullptr, *evalid = nullptr,
          *pair_rows = nullptr, *h_erows = nullptr, *h_e = nullptr, *h_evalid = nullptr;
    int32_t *icells = nullptr, *meta_d = nullptr, *meta_h = nullptr, *ids = nullptr, *pair_map = nullptr, *pair_ident = nullptr,
            *mmq_ident = nullptr;
    void* mmq_xq = nullptr;
    void release() {
        for (void* p : dev) cudaFree(p);
        for (void* p : host) cudaFreeHost(p);
        *this = Bufs();
    }
};

// The forward scratch every session of a model shares (docs/design/batching.md, "Per-slot state and shared scratch"): the row
// buffers of a decode window, the experts' staging and the products' workspaces. The engine runs one call at a time and each call
// syncs its stream before it returns, so one set serves every session; only a conversation's own state is per session.
struct Scratch {
    Bufs dec;
    std::vector<void*> dev, pinned;
    float *iscore = nullptr, *ex_rows = nullptr;
    int32_t *iidx_meta = nullptr, *pos_d = nullptr, *pos_h = nullptr, *heads_ids = nullptr, *heads_bounds = nullptr, *bounds_tab = nullptr;
    void *attw = nullptr, *q8 = nullptr, *xq_n = nullptr, *ex_xq = nullptr, *ex_scr = nullptr, *ex_plan = nullptr, *ex_plan2 = nullptr;
    void *xq1 = nullptr, *xq2 = nullptr;
    struct Staged {
        uint8_t* blobs = nullptr;
        int32_t *tok = nullptr, *bounds = nullptr;
    } stage[2];
    const float** rowptr_d = nullptr;
    const float** h_rowptr = nullptr;
    void* h_plan = nullptr;
    ~Scratch() {
        dec.release();
        for (void* p : dev) cudaFree(p);
        for (void* p : pinned) cudaFreeHost(p);
    }
};

class Session final : public model::BatchSession {
public:
    explicit Session(Hy4Model& m) : m_(m), g_(m.geo()) {}
    ~Session() override {
        settle_early();
        big_.release();
        platform::free_aligned(cpu_act_);
        std::string e;
        join_moe(e);
        for (cudaGraphExec_t g : gx_)
            if (g) cudaGraphExecDestroy(g);
        for (cudaGraphExec_t g : gbx_)
            if (g) cudaGraphExecDestroy(g);
        if (joiner_.joinable()) {
            {
                std::lock_guard<std::mutex> g(jm_);
                jstop_ = true;
            }
            jcv_.notify_one();
            joiner_.join();
        }
        if (ra_st_) cudaStreamSynchronize(ra_st_);
        for (void* p : bufs_) cudaFree(p);
        if (d_hist_) cudaFree(d_hist_);
        for (void* p : pinned_) cudaFreeHost(p);
        if (ev_) cudaEventDestroy(ev_);
        if (ra_ev_) cudaEventDestroy(ra_ev_);
        if (ra_st_) cudaStreamDestroy(ra_st_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool init(std::string& err) {
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_, cudaEventDisableTiming) != cudaSuccess ||
            cudaStreamCreateWithFlags(&ra_st_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&ra_ev_, cudaEventDisableTiming) != cudaSuccess) {
            err = "cannot create a stream";
            return false;
        }
        const int64_t E = g_.n_embd, ctx = m_.context();
        (void) E;
        int max_heads = 0;
        for (int h : g_.heads) max_heads = std::max(max_heads, h);
        bool ok = true;
        auto f = [&](int64_t n) { return (float*) alloc((size_t) n * 4, ok); };
        auto pin = [&](size_t n) {
            void* p = nullptr;
            if (cudaMallocHost(&p, n * 4) != cudaSuccess) ok = false;
            else pinned_.push_back(p);
            return (float*) p;
        };
        fin_ = f(kMaxWindow * E);
        const int64_t pools = ctx / g_.idx_pool + 1;
        ra_mark_ = (uint32_t*) alloc(8, ok);
        if (ok) cudaMemset(ra_mark_, 0, 8);
        cells_stride_ = g_.idx_top_k + 2 * g_.idx_pool;
        max_heads_ = max_heads;
        logits_ = f((int64_t) kMaxWindow * g_.n_vocab); pick_ = (int32_t*) alloc(4 * 32, ok);
        pick_h_ = (int32_t*) pin(kMaxWindow);
        lp_ = f((int64_t) bk::row_logprobs_scratch_floats(kMaxWindow, (int) g_.n_vocab) + (int64_t) kMaxWindow * (bk::kLogprobStride + 1));
        // the router's ids reach the host through mapped memory behind a sequence word; the weighted sum waits for a word the
        // host writes once the CPU's and the second card's rows landed
        {
            constexpr unsigned kMapped = cudaHostAllocPortable | cudaHostAllocMapped;
            if (cudaHostAlloc((void**) &h_ids_, (size_t) kLookupK * 4, kMapped) != cudaSuccess ||
                cudaHostAlloc((void**) &h_words_, 64, kMapped) != cudaSuccess)
                ok = false;
            else {
                pinned_.push_back(h_ids_);
                pinned_.push_back(h_words_);
                h_words_[0] = h_words_[1] = 0;
                if (cudaHostGetDevicePointer((void**) &d_ids_, h_ids_, 0) != cudaSuccess ||
                    cudaHostGetDevicePointer((void**) &d_words_, h_words_, 0) != cudaSuccess)
                    ok = false;
            }
            if (cudaHostAlloc((void**) &stage_h_, kStageInts * 4, kMapped) != cudaSuccess) ok = false;
            else {
                pinned_.push_back(stage_h_);
                if (cudaHostGetDevicePointer((void**) &stage_d_, stage_h_, 0) != cudaSuccess) ok = false;
            }
        }
        rope_dev_ = (int32_t*) alloc((size_t) ctx * 12, ok);   // [vision] the 3-D rotary position of every cell
        stride_ = m_.expert_stride();
        for (Stage& b : stage_)
            if (cudaEventCreateWithFlags(&b.copied, cudaEventDisableTiming) != cudaSuccess || cudaEventCreateWithFlags(&b.done, cudaEventDisableTiming) != cudaSuccess) ok = false;
        if (cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) != cudaSuccess) ok = false;
        // the forward scratch: the first session allocates it, the rest share it
        if (ok) {
            std::shared_ptr<void>& slot = m_.forward_scratch();
            if (!slot) {
                auto sc = std::make_shared<Scratch>();
                if (!build_scratch(*sc, err)) return false;
                slot = sc;
            }
            sc_ = std::static_pointer_cast<Scratch>(slot);
            adopt(*sc_);
        }
        cache_.assign((size_t) g_.n_layers, nullptr);
        ival_.assign((size_t) g_.n_layers, nullptr);
        igate_.assign((size_t) g_.n_layers, nullptr);
        ipool_.assign((size_t) g_.n_layers, nullptr);
        ea_.resize((size_t) g_.n_layers);
        pea_.resize((size_t) g_.n_layers);
        for (int l = 0; ok && l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) {
                cache_[(size_t) l] = (uint8_t*) alloc((size_t) swa_ring() * kRowWidth * elem(), ok);
            } else {
                ea_[(size_t) l] = std::make_unique<core::ElasticArray>();
                pea_[(size_t) l] = std::make_unique<core::ElasticArray>();
                if (!ea_[(size_t) l]->open((uint64_t) ctx * sparse_row_bytes(), err) || !pea_[(size_t) l]->open((uint64_t) pools * g_.idx_dim * 2, err)) return false;
                cache_[(size_t) l] = ea_[(size_t) l]->data();
                ipool_[(size_t) l] = (uint16_t*) pea_[(size_t) l]->data();
                ival_[(size_t) l] = f((int64_t) kRawRing * g_.idx_dim);
                igate_[(size_t) l] = f((int64_t) kRawRing * g_.idx_dim);
            }
        }
        if (!ok) {
            err = "out of device memory for the session";
            return false;
        }
        if (!ensure_positions(kv_base(), err)) return false;   // the elastic KV's base; the rest comes from the expert tier
        use(sc_->dec);
        // DFLY HOOK: the drafter's tap and scratch (nothing without a drafter in the pack)
        if (!dfly_.init(m_.dfly(), (int) (g_.idx_top_k + g_.idx_pool), ctx, [this](int32_t t, float* o) { return m_.embedding_row(t, o); }, m_.diag_dump(), fp8(), err))
            return false;
        dfly_.on_row([this](int j, int32_t tok) { early_draft_row(j, tok); });
        return true;
    }

    // The shared forward scratch, allocated once (alloc and pin put it in `sc`, which frees it with the last session).
    bool build_scratch(Scratch& sc, std::string& err) {
        const int64_t ctx = m_.context();
        namespace mmq = basalt::prefill::mmq;
        bool ok = true;
        alloc_to_ = &sc.dev;
        pin_to_ = &sc.pinned;
        auto f = [&](int64_t n) { return (float*) alloc((size_t) n * 4, ok); };
        const int64_t pools = ctx / g_.idx_pool + 1;
        sc.iscore = f(pools * kIdxRows);
        sc.iidx_meta = (int32_t*) alloc((size_t) kIdxRows * 3 * 4, ok);
        sc.pos_d = (int32_t*) alloc(16, ok);
        if (ok && cudaMallocHost((void**) &sc.pos_h, 16) == cudaSuccess) sc.pinned.push_back(sc.pos_h);
        else ok = false;
        size_t attw = 0;   // the chunked attention's workspace: the largest a layer needs; zero from the start
        for (int l = 0; l < g_.n_layers; ++l) attw = std::max(attw, mla_attend_work_bytes(g_.heads[(size_t) l], max_cells(l), kMaxWindow));
        sc.attw = alloc(attw, ok);
        if (ok) cudaMemset(sc.attw, 0, attw);
        sc.q8 = alloc(std::max<size_t>(bk::native_q8_1_bytes(16384, kMaxWindow), 1 << 16), ok);
        sc.xq_n = alloc(bk::native_q8_1_bytes(g_.n_embd, kMaxWindow), ok);
        sc.ex_xq = alloc(bk::native_q8_1_bytes(g_.n_latent, kMaxWindow), ok);
        sc.ex_scr = alloc(bk::native_expert_scratch_bytes(kLookupK, g_.n_ff_exp), ok);
        sc.ex_rows = f((int64_t) kLookupK * g_.n_latent);
        sc.ex_plan = alloc(sizeof(WindowPut), ok);
        sc.ex_plan2 = alloc(sizeof(WindowPlan), ok);
        // the prompt path's experts: two staged launches (blobs, rows and bounds), the pairs' outputs, and the quantised rows
        const size_t stride = m_.expert_stride();
        for (Scratch::Staged& b : sc.stage) {
            b.blobs = (uint8_t*) alloc(stride * kStageExperts, ok);
            b.tok = (int32_t*) alloc((size_t) kLaunchRows * 4, ok);
            b.bounds = (int32_t*) alloc((kStageExperts + 1) * 4, ok);
        }
        sc.xq1 = alloc(mmq::q8_bytes(kLaunchRows, g_.n_latent), ok);
        sc.xq2 = alloc(mmq::q8_bytes(kLaunchRows, g_.n_ff_exp), ok);
        sc.heads_ids = (int32_t*) alloc((size_t) max_heads_ * kAttRows * 4, ok);
        sc.heads_bounds = (int32_t*) alloc((size_t) (max_heads_ + 1) * 4, ok);
        sc.rowptr_d = (const float**) alloc(kLookupK * sizeof(void*), ok);
        {
            void* q = nullptr;
            if (ok && cudaMallocHost(&q, kLookupK * 2 * 4) == cudaSuccess) { sc.pinned.push_back(q); sc.h_rowptr = (const float**) q; }
            else ok = false;
        }
        // MMQ's row bounds {0, n} of a product of n rows, for every n of an attention step
        sc.bounds_tab = (int32_t*) alloc((size_t) 2 * (kAttRows + 1) * 4, ok);
        if (ok) {
            std::vector<int32_t> tab((size_t) 2 * (kAttRows + 1));
            for (int n = 0; n <= kAttRows; ++n) tab[(size_t) 2 * n + 1] = n;
            cudaMemcpy(sc.bounds_tab, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice);
        }
        if (cudaHostAlloc(&sc.h_plan, sizeof(WindowPlan), cudaHostAllocPortable) != cudaSuccess) ok = false;
        else sc.pinned.push_back(sc.h_plan);
        alloc_to_ = &bufs_;
        pin_to_ = &pinned_;
        if (!ok) {
            err = "out of device memory for the forward scratch";
            return false;
        }
        return make_bufs(sc.dec, kMaxWindow, err);
    }

    // The session's pointers into the shared scratch.
    void adopt(Scratch& sc) {
        iscore_ = sc.iscore; iidx_meta_ = sc.iidx_meta; pos_d_ = sc.pos_d; pos_h_ = sc.pos_h; attw_ = sc.attw; q8_ = sc.q8;
        xq_n_ = sc.xq_n; ex_xq_ = sc.ex_xq; ex_scr_ = sc.ex_scr; ex_rows_ = sc.ex_rows; ex_plan_ = sc.ex_plan; ex_plan2_ = sc.ex_plan2;
        for (int i = 0; i < 2; ++i) {
            stage_[i].blobs = sc.stage[i].blobs;
            stage_[i].tok = sc.stage[i].tok;
            stage_[i].bounds = sc.stage[i].bounds;
        }
        xq1_ = sc.xq1; xq2_ = sc.xq2; heads_ids_ = sc.heads_ids; heads_bounds_ = sc.heads_bounds; rowptr_d_ = sc.rowptr_d;
        h_rowptr_ = sc.h_rowptr; bounds_tab_ = sc.bounds_tab; h_plan_ = (WindowPlan*) sc.h_plan;
    }

    // The buffers for `R` rows; false (with nothing left allocated) when the card has no room. With `count` it allocates nothing and
    // adds up the device bytes they would take (each rounded up to the driver's 2 MiB pages, an upper bound).
    bool make_bufs(Bufs& b, int R, std::string& err, uint64_t* count = nullptr) {
        const int64_t E = g_.n_embd;
        bool ok = true;
        static float counted[4];   // a stand-in address the count mode hands out (nothing reads it)
        auto f = [&](int64_t n) {
            void* p = nullptr;
            if (count) {
                *count += (std::max<uint64_t>((uint64_t) n * 4, 16) + (2ull << 20) - 1) / (2ull << 20) * (2ull << 20);
                return counted;
            }
            if (!ok || cudaMalloc(&p, std::max<size_t>((size_t) n * 4, 16)) != cudaSuccess) {
                ok = false;
                return (float*) nullptr;
            }
            b.dev.push_back(p);
            return (float*) p;
        };
        auto pin = [&](int64_t n) {
            void* p = nullptr;
            if (count) return counted;
            if (!ok || cudaMallocHost(&p, (size_t) n * 4) != cudaSuccess) {
                ok = false;
                return (float*) nullptr;
            }
            b.host.push_back(p);
            return (float*) p;
        };
        const int A = std::min(R, kAttRows);
        b.rows = R;
        b.arows = A;
        b.X = f((int64_t) R * 4 * E); b.Xn = f((int64_t) R * 4 * E); b.coef = f((int64_t) R * bk::kHcCoef);
        b.hcws = f((int64_t) bk::hc_plain_workspace_floats(R));
        b.e = f((int64_t) R * E); b.cur = f((int64_t) R * E); b.xn = f((int64_t) R * E); b.attn = f((int64_t) R * E);
        b.attn_raw = f((int64_t) A * E); b.mlp = f((int64_t) R * E); b.mlp2 = f((int64_t) R * E);
        b.qa = f((int64_t) A * g_.q_lora); b.qan = f((int64_t) A * g_.q_lora); b.q = f((int64_t) A * max_heads_ * g_.qk_dim);
        b.kvraw = f((int64_t) A * kRowWidth); b.qabs = f((int64_t) A * max_heads_ * g_.kv_lora);
        b.qcur = f((int64_t) A * max_heads_ * kRowWidth); b.olat = f((int64_t) A * max_heads_ * g_.kv_lora);
        b.vout = f((int64_t) A * max_heads_ * g_.v_dim); b.gate = f((int64_t) A * max_heads_ * g_.v_dim);
        b.ik = f((int64_t) A * g_.idx_dim); b.ikn = f((int64_t) A * g_.idx_dim); b.ivt = f((int64_t) A * g_.idx_dim);
        b.igt = f((int64_t) A * g_.idx_dim); b.iq = f((int64_t) A * g_.idx_heads * g_.idx_dim); b.iw = f((int64_t) A * g_.idx_heads);
        b.icells = (int32_t*) f((int64_t) A * cells_stride_);
        b.meta_d = (int32_t*) f(4 * (int64_t) A);
        // a layer's steps each have their own: the copy of step 0's spans is still queued when the host writes step 1's
        b.steps = (R + A - 1) / A;
        b.meta_h = (int32_t*) pin((int64_t) g_.n_layers * b.steps * 4 * A);
        // gate/up products of the dense layer (A rows of n_ff), of the shared expert (R rows of the expert width) and of a prompt
        // launch (kLaunchRows rows of twice the expert width); the SwiGLU outputs likewise
        const int64_t gu_n = std::max<int64_t>({2 * (int64_t) A * g_.n_ff, 2 * (int64_t) R * g_.n_ff_exp, 2 * (int64_t) kLaunchRows * g_.n_ff_exp});
        const int64_t h_n = std::max<int64_t>({(int64_t) A * g_.n_ff, (int64_t) R * g_.n_ff_exp, (int64_t) kLaunchRows * g_.n_ff_exp});
        b.gu = f(gu_n); b.h = f(h_n);
        b.rlog = f((int64_t) R * g_.n_expert); b.rw = f((int64_t) R * 16); b.ids = (int32_t*) f((int64_t) R * 16);
        b.xlat = f((int64_t) R * g_.n_latent); b.lat_out = f((int64_t) R * g_.n_latent); b.moe_out = f((int64_t) R * E); b.sh = f((int64_t) R * E);
        b.erows = f((int64_t) A * g_.n_embd); b.ekv = f((int64_t) A * 5 * E); b.evalid = f(A);
        b.pair_rows = f((int64_t) R * g_.n_used * g_.n_latent);
        b.pair_map = (int32_t*) f((int64_t) R * g_.n_used); b.pair_ident = (int32_t*) f((int64_t) R * g_.n_used);
        b.mmq_ident = (int32_t*) f(A);
        b.h_erows = pin((int64_t) A * g_.n_embd); b.h_e = pin((int64_t) R * E); b.h_evalid = pin(A);
        if (count) {
            f((int64_t) (std::max(basalt::prefill::mmq::q8_bytes(A, 16384), basalt::prefill::mmq::q8_bytes((int64_t) A * max_heads_, std::max(g_.kv_lora, g_.qk_dim))) + 3) / 4);
            b = Bufs();
            return true;
        }
        if (ok) {
            void* q = nullptr;
            if (cudaMalloc(&q, std::max(basalt::prefill::mmq::q8_bytes(A, 16384), basalt::prefill::mmq::q8_bytes((int64_t) A * max_heads_, std::max(g_.kv_lora, g_.qk_dim)))) != cudaSuccess) ok = false;
            else {
                b.dev.push_back(q);
                b.mmq_xq = q;
            }
        }
        if (!ok) {
            b.release();
            cudaGetLastError();   // a smaller try must not find this failed allocation as the last error of its first launch
            err = "out of device memory for " + std::to_string(R) + " rows";
            return false;
        }
        if (basalt::prefill::mmq::built()) {
            basalt::prefill::mmq::iota(b.mmq_ident, A, nullptr);
            basalt::prefill::mmq::iota(b.pair_ident, (int64_t) R * g_.n_used, nullptr);
        }
        cudaDeviceSynchronize();
        return true;
    }

    // The session's row buffers are `b`'s from here on.
    void use(const Bufs& b) {
        rows_ = b.rows;
        arows_ = b.arows;
        steps_ = b.steps;
        X_ = b.X; Xn_ = b.Xn; coef_ = b.coef; hcws_ = b.hcws; e_ = b.e; cur_ = b.cur; xn_ = b.xn; attn_ = b.attn; attn_raw_ = b.attn_raw;
        mlp_ = b.mlp; mlp2_ = b.mlp2; qa_ = b.qa; qan_ = b.qan; q_ = b.q; kvraw_ = b.kvraw; qabs_ = b.qabs; qcur_ = b.qcur;
        olat_ = b.olat; vout_ = b.vout; gate_ = b.gate; ik_ = b.ik; ikn_ = b.ikn; ivt_ = b.ivt; igt_ = b.igt; iq_ = b.iq; iw_ = b.iw;
        icells_ = b.icells; meta_d_ = b.meta_d; meta_h_ = b.meta_h; gu_ = b.gu; h_ = b.h; rlog_ = b.rlog; rw_ = b.rw; ids_ = b.ids;
        xlat_ = b.xlat; lat_out_ = b.lat_out; moe_out_ = b.moe_out; sh_ = b.sh; erows_ = b.erows; ekv_ = b.ekv; evalid_ = b.evalid;
        pair_rows_ = b.pair_rows; pair_map_ = b.pair_map; pair_ident_ = b.pair_ident; mmq_ident_ = b.mmq_ident; mmq_xq_ = b.mmq_xq;
        h_erows_ = b.h_erows; h_e_ = b.h_e; h_evalid_ = b.h_evalid;
    }

    // A prompt of `total` rows and the row buffers of a chunk (up to kMaxChunk rows). A short prompt (below kRingRows: an agent
    // turn's new messages) keeps the card's expert slots, which its experts are staged from card to card, and makes its buffers
    // beside them: first in what the tier left free, then in the slots beside the base arena (shrink: the base keeps its
    // experts). A long prompt routes to every expert anyway: the tier lends all its slots for the chunk buffers and the ring.
    bool open_prompt(int total, std::string& err) {
        const double to = now_ms();
        struct L2 { double t; ~L2() { basalt::log::info("basalt hy4flash", "open_prompt %.0f ms", now_ms() - t); } } l2{to};
        if (big_.rows) {
            use(big_);
            return true;
        }
        cudaStreamSynchronize(stream_);
        n_peer_rows_ = n_cpu_rows_ = n_stage_dev_ = n_stage_bus_ = n_stage_peer_ = n_cpu_exp_ = 0;
        t_cpu_run_ms_ = 0;
        t_cpu_wait_ms_ = t_peer_wait_ms_ = t_gpu_wait_ms_ = 0;
        t_pre_ms_ = t_moe_ms_ = t_build_ms_ = t_issue_ms_ = t_att_ms_ = t_ffn_ms_ = 0;
        t_eng_ms_ = 0;
        t_eng_read_ms_ = t_eng_norm_ms_ = 0;
        t_ring_ms_ = 0;
        if (tier_ && total < kRingRows) {
            int want = kMaxWindow;
            while (want < total && want < kShortChunk) want *= 2;
            std::string e;
            for (int pass = 0; pass < 2; ++pass) {
                if (pass == 1) tier_->shrink();
                for (int R = want; R >= std::min(want, 2 * kMaxWindow); R /= 2)
                    if (make_bufs(big_, R, e)) {
                        use(big_);
                        lent_ = false;
                        return true;
                    }
            }
        }
        lent_ = true;
        int want = kMaxWindow;
        while (want < total && want < kMaxChunk) want *= 2;
        want = std::min(want, kMaxChunk);
        const bool ringed = total >= kRingRows && tier_ && !ring_;
        // a bounded lend: the tier gives up only the slots the chunk's buffers and the ring need, and the rest keep their experts
        // (the ring stages those card to card, and the first turn after the prompt finds them); all of them when that is not enough
        if (tier_ && tier_->partial_lend()) {
            Bufs probe;
            uint64_t need = 0;
            std::string e;
            make_bufs(probe, want, e, &need);
            if (ringed) need += expert_ring_bytes();
            if (tier_->lend(need) && make_bufs(big_, want, e)) {
                if (!ringed || ring_open() ||
                    (tier_->lend(expert_ring_bytes() + kExpertLendSegment) && ring_open())) {
                    use(big_);
                    return true;
                }
                big_.release();
            }
        }
        if (tier_) tier_->lend_all();
        for (int R = want; R >= kMaxWindow; R /= 2)
            if (make_bufs(big_, R, err)) {
                use(big_);
                if (ringed && !ring_open())
                    basalt::log::info("basalt hy4flash", "cannot open the expert ring (%.1f GB): the prompt stages its experts per layer",
                                      (double) expert_ring_bytes() / 1e9);
                return true;
            }
        if (tier_) {
            std::string e;
            tier_->reclaim(e);
            tier_->grow();
        }
        return false;
    }

    // The prompt's buffers go back; the tier takes its slots again, with the experts they held (most recently used first), so
    // the turn's first tokens find the card as the last turn left it rather than refilling it a promotion at a time.
    void release_prompt() {
        if (!big_.rows) return;
        ring_join();
        cudaStreamSynchronize(stream_);
        basalt::log::info("basalt hy4flash", "prompt time: %.0f ms queued ahead of the MoE (attention, dense), "
                          "%.0f ms in the MoE (%.0f ms sorting the pairs, %.0f ms issuing the launches), "
                          "%.0f ms in the Engram's host step (%.0f ms waiting for rows, %.0f ms normalizing); "
                          "host time in attention() %.0f ms, in ffn() %.0f ms, issuing the ring copies %.0f ms; "
                          "expert rows on the second card %lld, on the CPU %lld",
                          t_pre_ms_, t_moe_ms_, t_build_ms_, t_issue_ms_, t_eng_ms_, t_eng_read_ms_, t_eng_norm_ms_,
                          t_att_ms_, t_ffn_ms_, t_ring_ms_.load(), (long long) n_peer_rows_, (long long) n_cpu_rows_);
        basalt::log::info("basalt hy4flash", "prompt experts: staged from the card %lld, from RAM over the bus %lld, from the second card %lld, on the CPU %lld (its thread %.0f ms, waited for %.0f ms), second card's share waited for %.0f ms, the stream after the launches %.0f ms",
                          (long long) n_stage_dev_, (long long) n_stage_bus_, (long long) n_stage_peer_, (long long) n_cpu_exp_, t_cpu_run_ms_.load(), t_cpu_wait_ms_, t_peer_wait_ms_, t_gpu_wait_ms_);
        use(sc_->dec);
        ring_close();
        big_.release();
        if (tier_) {
            std::string e;
            if (lent_ && !tier_->reclaim(e, /*restore=*/true)) basalt::log::warn("basalt hy4flash", "the expert tier stays small: %s", e.c_str());
            tier_->grow();   // the slots beside the base arena, if a short prompt took them
        }
        lent_ = false;
    }

    void end_prompt() override { release_prompt(); }

    // The decode tier hits of the turn so far: the first kEarlyTokens tokens after a prompt and the rest, logged when the next
    // prompt starts (a turn that starts cold shows a low first share).
    static constexpr int kEarlyTokens = 32;
    int64_t turn_tokens_ = 0, turn_h0_ = 0, turn_l0_ = 0, early_h_ = 0, early_l_ = 0, late_h_ = 0, late_l_ = 0;
    void turn_count(int n) {
        if (!tier_) return;
        const int64_t h = tier_->hits(), lk = tier_->lookups();
        ((turn_tokens_ < kEarlyTokens) ? early_h_ : late_h_) += h - turn_h0_;
        ((turn_tokens_ < kEarlyTokens) ? early_l_ : late_l_) += lk - turn_l0_;
        turn_h0_ = h;
        turn_l0_ = lk;
        turn_tokens_ += n;
    }
    void turn_report() {
        if (early_l_ + late_l_ > 0)
            basalt::log::info("basalt hy4flash", "turn decode hits: first %d tokens %.1f%% (%lld lookups), later %.1f%% (%lld lookups)", kEarlyTokens,
                              early_l_ ? 100.0 * early_h_ / early_l_ : 0.0, (long long) early_l_, late_l_ ? 100.0 * late_h_ / late_l_ : 0.0,
                              (long long) late_l_);
        turn_tokens_ = early_h_ = early_l_ = late_h_ = late_l_ = 0;
        if (tier_) { turn_h0_ = tier_->hits(); turn_l0_ = tier_->lookups(); }
    }

    // DFLY HOOK: --spec sets the window (DeepSeek's rule: 0 or 1 is plain decode), and the drafter's calls (dfly.hpp)
    int max_window() const override { return std::clamp(m_.spec(), 1, kMaxWindow); }
    model::Result draft_begin(int32_t anchor, int64_t pos, int n) override {
        const model::Result r = dfly_.draft_begin(anchor, pos, n, cache_, stream_, rope_on_ ? text_rope(pos) : pos);   // [vision]
        early_window(anchor, pos);   // while the card runs the block
        return r;
    }
    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n, const DraftNeed& need) override {
        return dfly_.draft_end(ids, conf, max_n, n, stream_, need);
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens < 1) return {model::Status::Error, "hy4flash: an empty window"};
        std::string err;
        release_prompt();
        // the lossy routing flags apply to decode windows only: prompts route as the model does
        struct Lossy {
            core::ExpertTier*& t;
            ~Lossy() { t = nullptr; }
        } lossy{lossy_};
        lossy_ = tier_ && tier_->route_lossy() ? tier_.get() : nullptr;
        dfly_.begin_window(r.pos);   // DFLY HOOK
        for (int t0 = 0; t0 < r.n_tokens; t0 += kMaxWindow) {
            const int n = std::min(kMaxWindow, r.n_tokens - t0);
            dfly_.run(t0);   // DFLY HOOK
            const bool last = t0 + n == r.n_tokens;
            const bool want = r.row_tokens || (last && (r.next_token || r.logits));
            const int head = !want ? 0 : r.row_tokens ? n : 1;
            graphs_ = use_graphs_ && m_.diag_dump().empty() && !rope_on_ && n <= kMaxWindow && n <= arows_;
            const bool ran = run_rows(r.tokens + t0, n, r.pos + t0, head, err);
            graphs_ = false;
            if (!ran) return {model::Status::Error, "hy4flash: " + err};
            dfly_.run(-1);   // DFLY HOOK
            if (!want) continue;
            if (!pick(r, t0, head, err)) return {model::Status::Error, "hy4flash: " + err};
            if (r.row_tokens)
                for (int i = 0; i < n; ++i) r.row_tokens[t0 + i] = picks_[(size_t) i];
            if (last && r.next_token) *r.next_token = picks_[(size_t) head - 1];
            if (last && r.logits) cudaMemcpyAsync(r.logits, logits_ + (size_t) (head - 1) * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost, stream_);
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "hy4flash: the window failed"};
        turn_count(r.n_tokens);
        return {};
    }

    // ---- batch windows (docs/design/batching.md): several sessions' windows in one pass through the weights, on this
    // session's scratch and stream. Each conversation keeps its latent caches, rings, pooled keys, Engram history, drafter
    // context, logits and draws; the attention, the indexer and the Engram's lookups run per segment over its own state, the
    // rest over every row. The host places the experts between the layers' graphs, so the forward runs directly every window
    // (batch_captured false) and replays its own graphs by row count. The cap is the experts' group (8 rows).
    int batch_rows() const override { return kMaxWindow; }

protected:
    const char* batch_name() const override { return "hy4flash"; }
    int64_t batch_context() const override { return m_.context(); }
    model::BatchShared& batch_shared() override { return m_.batch_shared(); }
    cudaStream_t batch_stream() const override { return stream_; }
    bool batch_captured() const override { return false; }
    bool batch_ready() const override { return !rope_on_ && m_.diag_dump().empty(); }   // [vision] a picture's rotary table: in turn
    bool batch_reserve(const model::DecodeRequest& r, std::string& err) override {
        release_prompt();
        if (r.pos > (int64_t) hist_.size()) {
            err = "position " + std::to_string(r.pos) + " does not follow the state's " + std::to_string(hist_.size());
            return false;
        }
        return ensure_positions(r.pos + r.n_tokens, err);
    }
    bool batch_stage(const model::BatchView& v, model::GraphLru<core::CapturedGraph>::Key&, std::string& err) override {
        for (const int32_t t : v.w.tokens)
            if (t < 0 || t >= g_.n_vocab) {
                err = "token " + std::to_string(t) + " is out of the vocabulary";
                return false;
            }
        return true;
    }
    bool batch_forward(const model::BatchView& v, std::string& err) override {
        bsegs_.clear();
        for (int i = 0; i < v.n; ++i) {
            Session* s = static_cast<Session*>(v.ss[i]);
            const model::DecodeRequest& r = v.reqs[i];
            const model::BatchWindow::Seg& sg = v.w.segs[(size_t) i];
            bsegs_.push_back({s, sg.row0, sg.rows, r.pos, v.head[i]});
            s->hist_.resize((size_t) r.pos);   // a position before the end drops the rows after it
            s->hist_.insert(s->hist_.end(), r.tokens, r.tokens + r.n_tokens);
            s->place_rope(r.pos, r.n_tokens);   // [vision]
            s->dfly_.begin_window(r.pos);   // DFLY HOOK
            s->dfly_.run(0);
        }
        struct Lossy {
            core::ExpertTier*& t;
            ~Lossy() { t = nullptr; }
        } lossy{lossy_};
        lossy_ = tier_ && tier_->route_lossy() ? tier_.get() : nullptr;
        graphs_ = use_graphs_;
        const bool ok = run_rows(v.w.tokens.data(), v.w.rows, v.reqs[0].pos, v.w.rows, err);
        graphs_ = false;
        for (const Seg& sg : bsegs_) sg.s->dfly_.run(-1);   // DFLY HOOK
        bsegs_.clear();
        return ok;
    }
    void batch_take(const model::BatchView& v, int i) override {
        const model::DecodeRequest& r = v.reqs[i];
        const model::BatchWindow::Seg& sg = v.w.segs[(size_t) i];
        const Session* lead = static_cast<const Session*>(v.ss[0]);
        const int h = v.head[i];
        const size_t V = (size_t) g_.n_vocab;
        turn_count(r.n_tokens);
        last_rows_ = h;
        if (h == 0) return;
        const float* src = lead->logits_ + (size_t) (sg.row0 + sg.rows - h) * V;
        // the leader's own rows first: its copy lands below every later segment's rows
        if (src != logits_) cudaMemcpyAsync(logits_, src, (size_t) h * V * 4, cudaMemcpyDeviceToDevice, v.stream);
        if (r.logits) cudaMemcpyAsync(r.logits, logits_ + (size_t) (h - 1) * V, V * 4, cudaMemcpyDeviceToHost, v.stream);
    }
    const int32_t* batch_draw(const model::BatchView& v, int i, std::string& err) override {
        if (!draw(v.reqs[i], 0, v.head[i], v.stream, err)) return nullptr;
        cudaMemcpyAsync(pick_h_, pick_, (size_t) v.head[i] * 4, cudaMemcpyDeviceToHost, v.stream);
        return pick_h_;
    }

public:

    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        std::string err;
        dfly_.invalidate();   // DFLY HOOK
        if (!big_.rows) {   // a prompt starts
            turn_report();
            if (tier_) tier_->forget_use();
        }
        if (!big_.rows && elastic()) {   // a prompt starts: its positions against what the caches back and can back
            const int64_t need = std::min<int64_t>(m_.context(), r.pos + std::max(r.total_tokens, r.n_tokens));
            const int64_t keep = std::max(kv_base(), std::min<int64_t>(m_.context(), (need + kGrowPositions - 1) / kGrowPositions * kGrowPositions));
            if (backed_ > keep) trim_positions(keep);
            if (need > backed_) {
                const int64_t ceiling = kv_ceiling();
                if (need > ceiling)
                    return {model::Status::Refused, "hy4flash: elastic KV: " + std::to_string(need) + " positions would take the expert tier below the " +
                                                      std::to_string(prompt_floor_bytes() / stride_) + " slots a prompt chunk needs (about " +
                                                      std::to_string(ceiling) + " fit; --kv fp8 halves a position's bytes)"};
            }
        }
        if (!open_prompt(std::max(r.total_tokens, r.n_tokens), err)) return {model::Status::Error, "hy4flash: " + err};
        if (tier_) { turn_h0_ = tier_->hits(); turn_l0_ = tier_->lookups(); }
        const int chunk = m_.diag_rows() > 0 ? std::min(rows_, m_.diag_rows()) : rows_;
        for (int t0 = 0; t0 < r.n_tokens; t0 += chunk) {
            if (r.stop && r.stop()) break;
            const int n = std::min(chunk, r.n_tokens - t0);
            if (!run_rows(r.tokens + t0, n, r.pos + t0, 0, err)) return {grow_failed_ ? model::Status::Refused : model::Status::Error, "hy4flash: " + err};
            if (r.progress) {
                cudaStreamSynchronize(stream_);
                r.progress(t0 + n);
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "hy4flash: the prompt failed"};
        if (!r.more) release_prompt();
        return {};
    }

    // ---- [vision] images (vision.hpp, vision_encoder.hpp) ------------------------------------------------------------------
    // The Engram of an image row: llama.cpp's hy-v4-flash treats an embedding position as a token that cannot be hashed, so its own
    // contribution is masked (valid 0) and, to the rows after it, it is a padded history entry. The history therefore holds -1 for
    // an image row, which engram_ids reads as "no n-gram" on both counts.
    model::Result encode_image(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, model::ImageGrid& grid) override {
        ImageEncoder& enc = m_.encoder();
        if (!enc.staged()) return {model::Status::NotImplemented, "hy4flash: started without --vision"};
        vision::Image img;
        ImagePlan plan;
        std::string err;
        if (!vision::decode_image(data, bytes, img, err) || !plan_image(img.w, img.h, plan, err)) return {model::Status::Error, "hy4flash: " + err};
        // the encoder's region goes where a short prompt's buffers go (open_prompt): in what the tier left free, then in the
        // slots beside the base arena, then in the fewest base slots that make room for it (a bounded lend), and only then in
        // the room of every slot the tier lends; images_done gives them back with their experts
        for (int step = 0; enc.needs_region(plan); ++step) {
            if (enc.reserve(plan, err)) break;
            if (!tier_ || step == 2) return {model::Status::Error, "hy4flash: " + err};
            cudaStreamSynchronize(stream_);
            if (step == 0) tier_->shrink();
            else {
                if (!tier_->partial_lend() || !tier_->lend(enc.region_bytes(plan))) tier_->lend_all();
                vis_lent_ = true;
            }
            vis_shrunk_ = true;
        }
        if (!enc.encode(img, plan, rows, n_rows, grid, err)) return {model::Status::Error, "hy4flash: " + err};
        if (n_rows <= 0 || rows.size() != (size_t) n_rows * (size_t) g_.n_embd)
            return {model::Status::Error, "hy4flash: the projector's rows are not the model's width"};
        return {};
    }
    void images_done() override {
        m_.encoder().release();
        if (!vis_shrunk_) return;
        std::string e;
        if (vis_lent_ && !tier_->reclaim(e, /*restore=*/true)) basalt::log::warn("basalt hy4flash", "the expert tier stays small: %s", e.c_str());
        tier_->grow();
        vis_lent_ = vis_shrunk_ = false;
    }
    // The rows of a picture block (newline rows included) at positions pos..: chunks of the row pass with the rows as the embeddings.
    model::Result prefill_embeddings(const float* rows, int n_rows, int64_t pos, model::ImageGrid grid) override {
        if (grid.nx <= 0 || (int64_t) grid.nx * grid.ny != n_rows) return {model::Status::Error, "hy4flash: an image block needs its grid"};
        sync_rope(pos);
        img_start_ = rope_.next();
        img_nx_ = grid.nx;
        std::string err;
        if (!open_prompt(n_rows, err)) return {model::Status::Error, "hy4flash: " + err};
        const std::vector<int32_t> dead((size_t) rows_, -1);
        for (int at = 0; at < n_rows; at += rows_) {
            const int n = std::min(rows_, n_rows - at);
            emb_rows_ = rows + (size_t) at * (size_t) g_.n_embd;
            img_j_ = at;
            const bool ok = run_rows(dead.data(), n, pos + at, 0, err);
            if (!ok && grow_failed_) return {model::Status::Refused, "hy4flash: " + err};
            emb_rows_ = nullptr;
            if (!ok) return {model::Status::Error, "hy4flash: " + err};
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "hy4flash: the image rows failed"};
        return {};
    }

    model::Result prepare() override {
        std::string e;
        if (!ensure_tier(e)) return {model::Status::Error, "hy4flash: " + e};
        if (elastic())
            core::log_elastic_kv_ceiling(kv_ceiling(), m_.sessions(), (int64_t) m_.sessions() * m_.context(), (int64_t) (prompt_floor_bytes() / stride_));
        return {};
    }

    // ---- the elastic KV (core/elastic_kv.hpp): with a context above kElasticKvBase a session backs its first kElasticKvBase
    // positions at start; more come out of the expert tier when a prompt or a generation needs them (ensure_positions), and a
    // later prompt that needs fewer gives them back (trim_positions), so a short conversation decodes with the larger tier.
    bool elastic() const { return m_.context() > core::kElasticKvBase; }
    int64_t kv_base() const { return std::min<int64_t>(m_.context(), core::kElasticKvBase); }
    // device bytes a position takes in the sparse layers' caches (its row and a quarter of a pooled key)
    double cell_bytes() const {
        int sparse = 0;
        for (int l = 0; l < g_.n_layers; ++l) sparse += !g_.swa[(size_t) l];
        return (double) sparse * ((double) sparse_row_bytes() + (double) g_.idx_dim * 2.0 / (double) g_.idx_pool);
    }
    // the free memory a prompt chunk needs: the expert ring and the buffers of a chunk of attention-step rows
    static constexpr uint64_t kPromptFloorBuffers = 1ull << 30;
    uint64_t prompt_floor_bytes() const { return 2ull * (uint64_t) g_.n_expert * stride_ + kPromptFloorBuffers; }
    // the positions the caches can back before the expert tier's slots and the free memory above its floor would no longer hold
    // a prompt chunk's needs
    int64_t kv_ceiling() override {
        if (!elastic() || !tier_) return -1;
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const double spare = (double) free_b - (double) tier_->free_floor() + (double) tier_->slots() * (double) stride_ - (double) prompt_floor_bytes();
        return std::min<int64_t>(m_.context(), backed_ + (int64_t) (std::max(0.0, spare) / cell_bytes()));
    }
    // the backing past `keep` positions goes back to the driver (the tier takes it at the prompt's end)
    void trim_positions(int64_t keep) {
        cudaStreamSynchronize(stream_);
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) continue;
            ea_[(size_t) l]->trim((uint64_t) keep * sparse_row_bytes());
            pea_[(size_t) l]->trim((uint64_t) (keep / g_.idx_pool + 1) * g_.idx_dim * 2);
        }
        basalt::log::info("basalt hy4flash", "elastic KV: %lld -> %lld positions backed", (long long) backed_, (long long) keep);
        backed_ = keep;
    }

    ExpertCounts expert_counts() const override {
        const core::ExpertTier* t = tier_.get();
        return t == nullptr ? ExpertCounts{} : ExpertCounts{t->hits(), t->lookups(), t->offloaded(), t->file_lookups()};
    }

    std::vector<CardExperts> experts_by_card() const override {
        std::vector<CardExperts> out;
        if (tier_)
            for (const core::ExpertTier::CardSlots& c : tier_->slots_by_card()) out.push_back({c.device, c.slots, c.bytes});
        return out;
    }
    // the sparse layers' caches as backed now (the elastic KV)
    int64_t kv_bytes() const override { return backed_ * (int64_t) cell_bytes(); }

    bool expert_map(int64_t& total, std::vector<uint8_t>& homes, std::vector<uint32_t>& routed) override {
        if (!tier_) return false;
        tier_->expert_map(total, homes, routed);
        return true;
    }

    std::string info_words() const override {
        const std::string kv = m_.cache_fp8() ? "kv=fp8" : "kv=fp16";   // what a community row records
        return tier_ ? "expert_slots=" + std::to_string(tier_->gpu_slots()) + " expert_slots_primary=" + std::to_string(tier_->slots()) + " ram_experts_mib=" + std::to_string(tier_->ram_bytes() >> 20) + " " + kv + " " + program::spill_words(tier_->spill())
                     : kv + " " + program::spill_words({});   // no tier: every expert is on the card, none read from the file
    }

    // SLEEP / WAKE: the tier's card slots go back to the card (every expert keeps its RAM copy and its slot is remembered), and
    // return at the wake. The latent caches, the rings and the drafter's rows are conversation state and stay.
    model::Result set_low_memory(bool low, int64_t& expert_slots) override {
        std::string e;
        if (!tier_) return {model::Status::Error, "hy4flash: the expert tier is not open"};
        cudaStreamSynchronize(stream_);
        if (low) {
            tier_->lend_all();
        } else {
            if (!tier_->reclaim(e, /*restore=*/true)) return {model::Status::Error, "hy4flash: " + e};
            tier_->grow();
        }
        expert_slots = tier_->gpu_slots();
        return {};
    }

    model::Result set_vram_reserve(int64_t mib, int64_t& expert_slots) override {
        std::string e;
        if (!tier_) return {model::Status::Error, "hy4flash: the expert tier is not open"};
        cudaStreamSynchronize(stream_);
        if (!tier_->replan(mib, e)) return {model::Status::Error, "hy4flash: " + e};
        expert_slots = tier_->gpu_slots();
        return {};
    }

    // ---- conversations: the latent rows and pooled keys by position (save_cells / restore_cells), the rings by checkpoint
    // (running_state). A ring holds the last swa_ring() (window layers) or kRawRing (indexer rows) positions, so a checkpoint
    // at `pos` is intact while the session has not run more than kMaxRows rows past it.

    int cell_format() const { return fp8() ? 1 : f32() ? 2 : 3; }   // 3: fp16 rows
    size_t ring_bytes() const { return (size_t) swa_ring() * kRowWidth * elem(); }
    size_t raw_bytes() const { return (size_t) kRawRing * g_.idx_dim * 4; }

    size_t running_bytes() const override {
        size_t n = 0;
        for (int l = 0; l < g_.n_layers; ++l) n += g_.swa[(size_t) l] ? ring_bytes() : 2 * raw_bytes();
        return n;
    }

    model::Result running_state(int64_t pos, bool save, core::RunningState& st) override {
        if (save && (int64_t) hist_.size() - pos > kAttRows)
            return {model::Status::Error, "hy4flash: the rings have moved on more than " + std::to_string(kAttRows) + " positions past the checkpoint"};
        if (save && pos > (int64_t) hist_.size()) return {model::Status::Error, "hy4flash: a checkpoint past the session's position"};
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) {
                st.regions.push_back({cache_[(size_t) l], ring_bytes()});
            } else {
                st.regions.push_back({ival_[(size_t) l], raw_bytes()});
                st.regions.push_back({igate_[(size_t) l], raw_bytes()});
            }
        }
        st.stream = stream_;
        return {};
    }

    model::Result running_restored(int64_t pos, const uint8_t*) override {
        if ((int64_t) hist_.size() > pos) hist_.resize((size_t) pos);
        dfly_.invalidate();   // DFLY HOOK
        return {};
    }

    size_t cells_bytes(int64_t positions) const override {
        int sparse = 0;
        for (int l = 0; l < g_.n_layers; ++l) sparse += !g_.swa[(size_t) l];
        return ctxp::cache_bytes(positions, sparse, fp8()) * (f32() ? 2 : 1);
    }

    model::Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (positions > backed_) return {model::Status::Error, "hy4flash: positions that were never run"};
        cudaStreamSynchronize(stream_);
        kv.clear();
        const size_t rb = (size_t) sparse_row_bytes(), pools = (size_t) (positions / g_.idx_pool + 1), pb = (size_t) g_.idx_dim * 2;
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) continue;
            core::ConversationKv k;
            k.format = cell_format();
            k.page_size = g_.idx_pool;
            k.cells = positions;
            k.pooled_rows = (int64_t) pools;
            k.idx_dim = g_.idx_dim;
            k.k.resize_for_overwrite((size_t) positions * rb);
            k.pooled.resize_for_overwrite(pools * pb);
            const auto down = [&](const uint8_t* dev, core::ConversationBuffer& to) {
                return to.visit(0, to.size(), [&](uint8_t* p, size_t n, size_t at) { return cudaMemcpy(p, dev + at, n, cudaMemcpyDeviceToHost) == cudaSuccess; });
            };
            if (!down(cache_[(size_t) l], k.k) || !down((const uint8_t*) ipool_[(size_t) l], k.pooled))
                return {model::Status::Error, "hy4flash: reading a cache back failed"};
            kv.push_back(std::move(k));
        }
        return {};
    }

    model::Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) override {
        std::string e;
        if (!ensure_positions(positions, e)) return {model::Status::Error, "hy4flash: " + e};
        cudaStreamSynchronize(stream_);
        size_t i = 0;
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) continue;
            if (i >= kv.size() || kv[i].format != cell_format() || kv[i].page_size != g_.idx_pool || kv[i].cells != positions)
                return {model::Status::Error, "hy4flash: a parked conversation has other caches"};
            const auto up = [&](uint8_t* dev, const core::ConversationBuffer& from) {
                return from.visit(0, from.size(), [&](const uint8_t* p, size_t n, size_t at) { return cudaMemcpy(dev + at, p, n, cudaMemcpyHostToDevice) == cudaSuccess; });
            };
            if (!up(cache_[(size_t) l], kv[i].k) || !up((uint8_t*) ipool_[(size_t) l], kv[i].pooled))
                return {model::Status::Error, "hy4flash: writing a cache back failed"};
            ++i;
        }
        dfly_.invalidate();   // DFLY HOOK
        return {};
    }

    // the tokens below `n` come back with a restored conversation: the Engram's history
    model::Result set_history(const int32_t* tokens, int64_t n) override {
        hist_.assign(tokens, tokens + n);
        return {};
    }

    model::Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (!bias_on_) return {};
        std::string e;
        if (!device_bias(d_bias_, bias, e)) return {model::Status::Error, "hy4flash: " + e};
        return {};
    }

    // a draw of one row again, with the tokens the mask bans
    model::Result resample_row(int row, const float* bias, int32_t& token) override {
        std::string e;
        if (row < 0 || row >= last_rows_) return {model::Status::Error, "hy4flash: the row is not in the last window"};
        if (!device_bias(d_mask_, bias, e)) return {model::Status::Error, "hy4flash: " + e};
        bk::SamplerParams sp = sampling_;
        sp.logit_bias = d_mask_;
        sp.counter += (uint64_t) row;
        bk::sample_tokens(logits_ + (size_t) row * g_.n_vocab, 1, (int) g_.n_vocab, history_len_ > 0 ? d_hist_ + (size_t) row * history_len_ : nullptr,
                          history_len_, sp, pick_ + 24, stream_);
        int32_t t = 0;
        if (cudaMemcpyAsync(&t, pick_ + 24, 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess || cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "hy4flash: the masked draw failed"};
        token = t;
        return {};
    }

    model::Result copy_logits(int row, float* out) override {
        if (row < 0 || row >= last_rows_) return {model::Status::Error, "hy4flash: the session keeps the last window's logits only"};
        cudaMemcpy(out, logits_ + (size_t) row * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost);
        return {};
    }

    model::Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (row0 < 0 || n < 1 || row0 + n > last_rows_) return {model::Status::Error, "hy4flash: the session keeps the last window's logits only"};
        float* d_out = lp_ + bk::row_logprobs_scratch_floats(n, (int) g_.n_vocab);
        int32_t* d_pick = pick_ + 16;
        cudaMemcpyAsync(d_pick, picks, (size_t) n * 4, cudaMemcpyHostToDevice, stream_);
        bk::row_logprobs(logits_ + (size_t) row0 * g_.n_vocab, n, (int) g_.n_vocab, d_pick, top, d_out, lp_, stream_);
        cudaMemcpyAsync(out, d_out, (size_t) n * bk::kLogprobStride * 4, cudaMemcpyDeviceToHost, stream_);
        return cudaStreamSynchronize(stream_) == cudaSuccess ? model::Result{} : model::Result{model::Status::Error, "hy4flash: logprobs failed"};
    }

private:
    void* alloc(size_t bytes, bool& ok) {
        void* p = nullptr;
        if (!ok || cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
            ok = false;
            return nullptr;
        }
        alloc_to_->push_back(p);
        return p;
    }

    int swa_ring() const { return g_.window + kAttRows; }
    int max_cells(int l) const { return g_.swa[(size_t) l] ? g_.window : cells_stride_; }   // a row's cells in layer l
    bool fp8() const { return m_.cache_fp8(); }
    bool f32() const { return m_.cache_f32(); }
    int elem() const { return f32() ? 4 : 2; }
    int sparse_row_bytes() const { return fp8() ? kFp8RowBytes : kRowWidth * elem(); }

    // A prompt chunk's small inputs from the host: staged in mapped memory and copied by a kernel, so they do not wait on a copy
    // engine behind the expert ring's copies (one piece queued there holds every later copy). A window copies as before.
    void put_i32(int32_t* dst, const int32_t* src, size_t n) {
        if (rows_ <= kMaxWindow || n == 0) {
            cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyHostToDevice, stream_);
            return;
        }
        if (stage_at_ + n > kStageInts) {   // full: wait until the kernels queued so far have read their rows
            cudaStreamSynchronize(stream_);
            stage_at_ = 0;
        }
        std::memcpy(stage_h_ + stage_at_, src, n * 4);
        bk::copy_i32x3_from_mapped(dst, stage_d_ + stage_at_, (int64_t) n, nullptr, nullptr, 0, nullptr, nullptr, 0, stream_);
        stage_at_ += (n + 3) & ~(size_t) 3;
    }
    // n floats from pinned host memory the caller keeps until the stream has read it
    void put_f32(float* dst, const float* src, int64_t n) {
        if (rows_ <= kMaxWindow) {
            cudaMemcpyAsync(dst, src, (size_t) n * 4, cudaMemcpyHostToDevice, stream_);
            return;
        }
        if (n % 4 || (uintptr_t) src % 16 || (uintptr_t) dst % 16)
            bk::copy_i32x3_from_mapped((int32_t*) dst, (const int32_t*) src, n, nullptr, nullptr, 0, nullptr, nullptr, 0, stream_);
        else
            bk::copy_from_mapped(dst, src, n, stream_);
    }

    // y[r] = W x[r] for R rows, 8 at a time: a row's product does not depend on the rows beside it
    void gemm(const DevTensor* w, const float* x, float* y, int n_in, int n_out, int R) {
        namespace mmq = basalt::prefill::mmq;
        if (R > kMaxWindow && mmq::built() && mmq::fits(w->type, n_out)) {   // a chunk: the tensor-core MMQ product, arows_ rows a time
            if (loaded_.insert({w->type, n_out}).second) mmq::load(w->type, n_out);
            for (int r0 = 0; r0 < R; r0 += arows_) {
                const int nb = std::min(arows_, R - r0);
                mmq::quantize(x + (size_t) r0 * n_in, nullptr, mmq_xq_, w->type, n_in, n_in, nb, stream_);
                mmq::Product p;
                p.w = w->ptr;
                p.type = w->type;
                p.w_rows = n_out;
                p.w_cols = n_in;
                p.expert_bytes = mmq::matrix_bytes(w->type, n_out, n_in);
                p.n = 1;
                p.xq = mmq_xq_;
                p.bounds = bounds_tab_ + 2 * nb;
                p.ids = mmq_ident_;
                p.total_rows = nb;
                p.max_rows = nb;
                p.dst = y + (size_t) r0 * n_out;
                p.ld_dst = n_out;
                mmq_.run(p, stream_);
            }
            return;
        }
        for (int c0 = 0; c0 < R; c0 += kMaxWindow) {
            const int nc = std::min(kMaxWindow, R - c0);
            bk::native_quantize_q8_1(x + (size_t) c0 * n_in, q8_, n_in, nc, stream_);
            bk::native_mmvq(w->type, w->ptr, q8_, y + (size_t) c0 * n_out, n_in, n_out, nc, stream_);
        }
    }
    // The per-head products of the absorbed attention (k_b: 192 -> 512, v_b: 512 -> 256 per head): a chunk takes MMQ, a head at a time
    // over the rows of the step; a window keeps the kernel that rounds like the other single-row products.
    void heads_product(const DevTensor* w, const float* x, int x_head_stride, int x_tok_stride, int H, int rows, int k, int R, float* y) {
        namespace mmq = basalt::prefill::mmq;
        if (R > kMaxWindow && R <= arows_ && mmq::built() && mmq::fits(w->type, rows)) {
            if (loaded_.insert({w->type, rows}).second) mmq::load(w->type, rows);
            // every head is an "expert" of the product with the R rows of the step: one quantisation and one launch for all heads
            head_major_ids(heads_ids_, heads_bounds_, H, R, stream_);
            const size_t mat = mmq::matrix_bytes(w->type, rows, k);
            mmq::quantize(x, heads_ids_, mmq_xq_, w->type, k, x_head_stride, (int64_t) H * R, stream_);
            mmq::Product p;
            p.w = (const uint8_t*) w->ptr;
            p.type = w->type;
            p.w_rows = rows;
            p.w_cols = k;
            p.expert_bytes = mat;
            p.n = H;
            p.xq = mmq_xq_;
            p.bounds = heads_bounds_;
            p.ids = heads_ids_;
            p.total_rows = (int64_t) H * R;
            p.max_rows = R;
            p.dst = y;
            p.ld_dst = rows;
            mmq_.run(p, stream_);
            return;
        }
        q8_0_heads_gemv(w->ptr, x, x_head_stride, x_tok_stride, H, rows, k, R, y, stream_);
    }
    // A window's normed input xn_ quantized once for the products that read it (gemm quantizes per call): the same kernels and
    // bits, fewer launches. A chunk keeps gemm.
    void quant_n(int R) {
        if (R <= kMaxWindow) bk::native_quantize_q8_1(xn_, xq_n_, g_.n_embd, R, stream_);
    }
    void gemm_n(const DevTensor* w, float* y, int n_out, int R) {
        if (R > kMaxWindow) return gemm(w, xn_, y, g_.n_embd, n_out, R);
        bk::native_mmvq(w->type, w->ptr, xq_n_, y, g_.n_embd, n_out, R, stream_);
    }
    const DevTensor* T(int l, const char* s) const { return m_.tensor(l, s); }
    const float* F(int l, const char* s) const { return (const float*) m_.tensor(l, s)->ptr; }

    MlaArgs mla_args(int heads, float base) const {
        MlaArgs a;
        a.heads = heads;
        a.nope = g_.qk_dim - g_.rope_dim;
        a.rope = g_.rope_dim;
        a.rank = g_.kv_lora;
        a.rope_base = base;
        a.scale = 1.0f / std::sqrt((float) g_.qk_dim);
        a.rms_eps = g_.rms_eps;
        a.cell_rope = rope_on_ ? rope_dev_ : nullptr;   // [vision]
        a.sec_h = kMropeH;
        a.sec_w = kMropeW;
        return a;
    }

    // A batch window's segments while batch_forward runs (empty: the rows are this session's, at pos0..). The attention, the
    // indexer and the Engram run each segment over its own session's state; a row's position and history are its segment's.
    struct Seg {
        Session* s;
        int row0, rows;
        int64_t pos0;
        int head;
    };
    std::vector<Seg> bsegs_;
    template <class Fn>
    void each_seg(int64_t pos0, int R, Fn&& fn) {
        if (bsegs_.empty()) fn(Seg{this, 0, R, pos0, 0});
        else
            for (const Seg& sg : bsegs_) fn(sg);
    }
    // Row t's session and position.
    const Seg& seg_of(int t) const {
        size_t i = 0;
        while (i + 1 < bsegs_.size() && bsegs_[i + 1].row0 <= t) ++i;
        return bsegs_[i];
    }
    int64_t row_pos(int64_t pos0, int t) const {
        if (bsegs_.empty()) return pos0 + t;
        const Seg& sg = seg_of(t);
        return sg.pos0 + (t - sg.row0);
    }
    const std::vector<int32_t>& row_hist(int t) const { return bsegs_.empty() ? hist_ : seg_of(t).s->hist_; }

    // Attention of layer `l` over R rows at pos0.. of the normed input xn_; the post-normed result goes to attn_. Steps of arows_ rows run
    // one after the other (each writes its rows' cache cells before it attends, so a row sees the rows before it), over the streams
    // offset to the step.
    bool attention(int l, int64_t pos0, int R, std::string& err) {
        float *xn0 = xn_, *at0 = attn_;
        const int E = g_.n_embd;
        bool ok = true;
        for (int off = 0; ok && off < R; off += arows_) {
            xn_ = xn0 + (size_t) off * E;
            attn_ = at0 + (size_t) off * E;
            ok = attention_step(l, pos0 + off, std::min(arows_, R - off), off / arows_, err);
        }
        xn_ = xn0;
        attn_ = at0;
        return ok;
    }

    // The attention of R rows in three phases: the query and key products (kQuery: no position in them), the cache rows, the
    // cells each row attends and the indexer (kCells: the positions), the attention over them and the output (kOutput: no
    // position). A decode window replays the first and last from graphs.
    static constexpr int kQuery = 1, kCells = 2, kOutput = 4, kAll = 7;
    static constexpr int kRaw = 8;   // with kOutput: leave the output projection un-normed in attn_raw_ (add_norm_hc_post norms it)
    static constexpr int kDevMeta = 16;   // with kCells on a sliding-window layer: the row spans from pos_d_ on the card (capturable)
    static constexpr int kAttend = 32;    // the attention over the cells alone (a batch runs it per segment, between its graphs)
    static constexpr int kTail = 64;      // with kOutput: the output alone, the attention over the cells already run
    bool attention_step(int l, int64_t pos0, int R, int step, std::string& err, int phases = kAll) {
        const int H = g_.heads[(size_t) l];
        const bool swa = g_.swa[(size_t) l];
        const int E = g_.n_embd, rank = g_.kv_lora;
        const MlaArgs a = mla_args(H, swa ? g_.rope_base_swa : g_.rope_base);
        if (phases & kQuery) {
            quant_n(R);
            gemm_n(T(l, "attn_q_a.weight"), qa_, g_.q_lora, R);
            bk::rms_norm_rows(qa_, F(l, "attn_q_a_norm.weight"), qan_, R, g_.q_lora, g_.rms_eps, stream_);
            gemm(T(l, "attn_q_b.weight"), qan_, q_, g_.q_lora, H * g_.qk_dim, R);
            gemm_n(T(l, "attn_kv_a_mqa.weight"), kvraw_, rank + g_.rope_dim, R);
        }
        if (!(phases & (kCells | kOutput | kAttend))) return true;
        // the cells each row attends
        int32_t* meta = meta_h_ + ((size_t) l * steps_ + step) * 4 * arows_;   // firsts, counts, fill, positions
        int32_t *firsts = meta, *counts = meta + arows_, *fill = meta + 2 * arows_, *posv = meta + 3 * arows_;
        bool selecting = false;
        std::vector<int> pools_of(R, 0);
        const int keep = g_.idx_top_k / g_.idx_pool - (g_.idx_select_tail ? 1 : 0);
        if ((phases & kCells) && (phases & kDevMeta) && swa) {
            swa_meta(pos_d_, R, g_.window, arows_, meta_d_, stream_);
        } else if (phases & kCells) {   // the cells and positions: a window replays only what comes before and after
            for (int t = 0; t < R; ++t) posv[t] = (int32_t) row_pos(pos0, t);
            if (swa) {
                for (int t = 0; t < R; ++t) {
                    const int64_t pos = posv[t], first = std::max<int64_t>(0, pos + 1 - g_.window);
                    firsts[t] = (int32_t) first;
                    counts[t] = (int32_t) (pos + 1 - first);
                    fill[t] = 0;
                }
            } else {
                for (int t = 0; t < R; ++t) {
                    const int64_t pos = posv[t];
                    pools_of[(size_t) t] = (int) (pos / g_.idx_pool);   // complete groups before the row's own
                    firsts[t] = 0;
                    if (pools_of[(size_t) t] > keep) {
                        selecting = true;
                        counts[t] = keep * g_.idx_pool + (int32_t) (pos - (int64_t) pools_of[(size_t) t] * g_.idx_pool + 1);
                        fill[t] = 0;
                    } else {
                        counts[t] = (int32_t) (pos + 1);
                        fill[t] = counts[t];
                    }
                }
            }
            put_i32(meta_d_, meta, (size_t) 4 * arows_);
        }
        const int ring = swa ? swa_ring() : 0;
        const bool f8 = !swa && fp8();
        const float* kvn = F(l, "attn_kv_a_norm.weight");
        if (phases & kCells) {
        if (R <= kMaxWindow) {   // a window: the k_b product, the query rows and the cache rows in one launch (a segment's)
            const void* kb = T(l, "attn_k_b.weight")->ptr;
            each_seg(pos0, R, [&](const Seg& sg) {
                uint8_t* cache = sg.s->cache_[(size_t) l];
                const int32_t* pos_d = meta_d_ + 3 * arows_ + sg.row0;
                const float *q = q_ + (size_t) sg.row0 * H * g_.qk_dim, *kv = kvraw_ + (size_t) sg.row0 * kRowWidth;
                float* qc = qcur_ + (size_t) sg.row0 * H * kRowWidth;
                if (f8) mla_query(kb, q, kv, kvn, a, pos_d, sg.rows, ring, qc, cache, stream_);
                else if (f32()) mla_query(kb, q, kv, kvn, a, pos_d, sg.rows, ring, qc, (float*) cache, stream_);
                else mla_query(kb, q, kv, kvn, a, pos_d, sg.rows, ring, qc, (uint16_t*) cache, stream_);
            });
        } else {
            uint8_t* cache = cache_[(size_t) l];
            heads_product(T(l, "attn_k_b.weight"), q_, g_.qk_dim, H * g_.qk_dim, H, rank, a.nope, R, qabs_);
            if (f8) mla_prepare(q_, qabs_, kvraw_, kvn, a, (int) pos0, R, ring, qcur_, cache, stream_);
            else if (f32()) mla_prepare(q_, qabs_, kvraw_, kvn, a, (int) pos0, R, ring, qcur_, (float*) cache, stream_);
            else mla_prepare(q_, qabs_, kvraw_, kvn, a, (int) pos0, R, ring, qcur_, (uint16_t*) cache, stream_);
        }
        if (diag(l) && f32() && !swa) {   // the inputs of the attention, for the reference's tensors (fp32 caches only)
            const uint8_t* cache = cache_[(size_t) l];
            diag_sync();
            dump("q_resid-" + std::to_string(l), qan_, (int64_t) R * g_.q_lora);
            std::vector<float> rows((size_t) R * kRowWidth), lat((size_t) R * rank), pe((size_t) R * g_.rope_dim), qp((size_t) R * H * g_.rope_dim),
                               qc((size_t) R * H * kRowWidth);
            cudaMemcpy(rows.data(), (const float*) cache + (size_t) pos0 * kRowWidth, rows.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(qc.data(), qcur_, qc.size() * 4, cudaMemcpyDeviceToHost);
            for (int t = 0; t < R; ++t) {
                for (int c = 0; c < rank; ++c) lat[(size_t) t * rank + c] = rows[(size_t) t * kRowWidth + c];
                for (int c = 0; c < g_.rope_dim; ++c) pe[(size_t) t * g_.rope_dim + c] = rows[(size_t) t * kRowWidth + rank + c];
                for (int h = 0; h < H; ++h)
                    for (int c = 0; c < g_.rope_dim; ++c) qp[((size_t) t * H + h) * g_.rope_dim + c] = qc[((size_t) t * H + h) * kRowWidth + rank + c];
            }
            auto put = [&](const std::string& n, const std::vector<float>& v) {
                std::FILE* f = std::fopen((m_.diag_dump() + "/" + n + ".f32").c_str(), "ab");
                if (f) { std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); }
            };
            put("kv_cmpr-" + std::to_string(l), lat);
            put("k_pe-" + std::to_string(l), pe);
            put("q_pe-" + std::to_string(l), qp);
        }
        if (!swa) {
            indexer(l, pos0, R, selecting, pools_of, err);
            each_seg(pos0, R, [&](const Seg& sg) {   // DFLY HOOK: each row's selection, for its conversation's drafter
                for (int t = sg.row0; t < sg.row0 + sg.rows; ++t) {
                    const bool sel = pools_of[(size_t) t] > keep;
                    sg.s->dfly_.selection(l, t - sg.row0, sel ? icells_ + (size_t) t * cells_stride_ : nullptr, sel ? counts[t] : -1, stream_);
                }
            });
        }
        }
        if (!(phases & (kOutput | kAttend))) return true;
        if (!(phases & kTail)) {   // each segment over its own cache
            const float* sinks = F(l, "attn_sinks.weight");
            each_seg(pos0, R, [&](const Seg& sg) {
                const uint8_t* cache = sg.s->cache_[(size_t) l];
                RowSpans rs;
                rs.firsts = meta_d_ + sg.row0;
                rs.counts = meta_d_ + arows_ + sg.row0;
                if (!swa) {
                    rs.cells = icells_ + (size_t) sg.row0 * cells_stride_;
                    rs.cells_stride = cells_stride_;
                    rs.fill = meta_d_ + 2 * arows_ + sg.row0;
                }
                const float* qc = qcur_ + (size_t) sg.row0 * H * kRowWidth;
                float* ol = olat_ + (size_t) sg.row0 * H * rank;
                if (f8) mla_attend(qc, cache, rs, sg.rows, sinks, a, ring, max_cells(l), attw_, ol, stream_);
                else if (f32()) mla_attend(qc, (const float*) cache, rs, sg.rows, sinks, a, ring, max_cells(l), attw_, ol, stream_);
                else mla_attend(qc, (const uint16_t*) cache, rs, sg.rows, sinks, a, ring, max_cells(l), attw_, ol, stream_);
            });
        }
        if (!(phases & kOutput)) return true;
        heads_product(T(l, "attn_v_b.weight"), olat_, rank, H * rank, H, g_.v_dim, rank, R, vout_);
        if (diag(l)) dump("attn_kqv-" + std::to_string(l), vout_, (int64_t) R * H * g_.v_dim);
        gemm_n(T(l, "attn_gate.weight"), gate_, H * g_.v_dim, R);   // xn_ as the query phase quantized it
        mul_sigmoid(gate_, vout_, R * H * g_.v_dim, stream_);
        if (diag(l)) dump("attn_gated-" + std::to_string(l), vout_, (int64_t) R * H * g_.v_dim);
        gemm(T(l, "attn_output.weight"), vout_, attn_raw_, H * g_.v_dim, E, R);
        if (phases & kRaw) return true;
        bk::rms_norm_rows(attn_raw_, F(l, "post_attention_norm.weight"), attn_, R, E, g_.rms_eps, stream_);
        if (diag(l)) dump("attn_out-" + std::to_string(l), attn_, (int64_t) R * E);
        return true;
    }

    // The indexer of sparse layer `l` for R rows at pos0..: the rows of the keys and the pooled keys of the groups they
    // complete; for the rows with more pools than it keeps, its selection into icells_.
    static constexpr int kIdxRows = 64;
    void indexer(int l, int64_t pos0, int R, bool selecting, const std::vector<int>& pools_of, std::string&) {
        const int D = g_.idx_dim, P = g_.idx_pool, E = g_.n_embd;
        IndexerArgs a;
        a.heads = g_.idx_heads;
        a.dim = D;
        a.pool = P;
        a.rope_base = g_.rope_base;
        a.eps = g_.rms_eps;
        a.cell_rope = rope_on_ ? rope_dev_ : nullptr;   // [vision]
        a.sec_h = kMropeH;
        a.sec_w = kMropeW;
        gemm(T(l, "indexer.attn_k.weight"), xn_, ik_, E, D, R);
        bk::rms_norm_rows(ik_, F(l, "indexer.k_norm.weight"), ikn_, R, D, g_.rms_eps, stream_);
        gemm(T(l, "indexer_compressor_value.weight"), ikn_, ivt_, D, D, R);
        gemm(T(l, "indexer_compressor_gate.weight"), ikn_, igt_, D, D, R);
        // each segment's rows into its ring of the last positions, at cell pos % kRawRing (one wrap at most), and the pooled keys
        // of the groups whose last position is among them
        each_seg(pos0, R, [&](const Seg& sg) {
            Session& c = *sg.s;
            const int64_t s0 = sg.pos0 % kRawRing;
            const int n1 = (int) std::min<int64_t>(sg.rows, kRawRing - s0);
            const size_t at = (size_t) sg.row0 * D;
            for (auto [src, dst] : {std::pair<const float*, float*>{ivt_ + at, c.ival_[(size_t) l]}, {igt_ + at, c.igate_[(size_t) l]}}) {
                cudaMemcpyAsync(dst + (size_t) s0 * D, src, (size_t) n1 * D * 4, cudaMemcpyDeviceToDevice, stream_);
                if (n1 < sg.rows) cudaMemcpyAsync(dst, src + (size_t) n1 * D, (size_t) (sg.rows - n1) * D * 4, cudaMemcpyDeviceToDevice, stream_);
            }
            const int64_t p0 = sg.pos0, last = p0 + sg.rows - 1;
            const int64_t g0 = p0 >= 3 ? (p0 - 3 + P - 1) / P : 0, g1 = last >= 3 ? (last - 3) / P : -1;
            if (g1 >= g0) idx_pool(c.ival_[(size_t) l], c.igate_[(size_t) l], F(l, "indexer_compressor_ape.weight"), F(l, "indexer_compressor_norm.weight"),
                                   a, kRawRing, (int) (g0 * P), (int) (g1 - g0 + 1), c.ipool_[(size_t) l], stream_);
        });
        if (!selecting) return;
        const int keep = g_.idx_top_k / P - (g_.idx_select_tail ? 1 : 0);
        gemm(T(l, "indexer.attn_q_b.weight"), qan_, iq_, g_.q_lora, a.heads * D, R);
        each_seg(pos0, R, [&](const Seg& sg) { idx_rope_heads(iq_ + (size_t) sg.row0 * a.heads * D, a, (int) sg.pos0, sg.rows, stream_); });
        f32_gemm_rows(xn_, F(l, "indexer.proj.weight"), a.heads, E, 1.0f / std::sqrt((float) (D * a.heads)), R, iw_, stream_);
        // each segment's rows that select over its own pooled keys, kIdxRows at a time
        const int stride = (int) (m_.context() / P + 1);
        each_seg(pos0, R, [&](const Seg& sg) {
            for (int t0 = sg.row0; t0 < sg.row0 + sg.rows; t0 += kIdxRows) {
                const int n = std::min(kIdxRows, sg.row0 + sg.rows - t0);
                int32_t meta[kIdxRows * 3];
                int max_pool = 0;
                for (int i = 0; i < n; ++i) {
                    const int pools = pools_of[(size_t) (t0 + i)], tail0 = pools * P;
                    meta[i * 3] = pools;
                    meta[i * 3 + 1] = tail0;
                    meta[i * 3 + 2] = (int) (sg.pos0 + (t0 - sg.row0) + i - tail0 + 1);
                    max_pool = std::max(max_pool, pools);
                }
                if (max_pool <= keep) continue;
                put_i32(iidx_meta_, meta, (size_t) n * 3);
                idx_scores_rows(iq_ + (size_t) t0 * a.heads * D, iw_ + (size_t) t0 * a.heads, sg.s->ipool_[(size_t) l], iidx_meta_, max_pool, n, a, iscore_, stride, stream_);
                idx_select_rows(iscore_, stride, iidx_meta_, n, keep, P, icells_ + (size_t) t0 * cells_stride_, cells_stride_, stream_);
            }
        });
    }

    // The Engram step of layer `l` on the streams of R rows, arows_ rows a step.
    bool engram(int l, int64_t pos0, int R, const int32_t* tokens, std::string& err) {
        float* x0 = X_;
        bool ok = true;
        for (int off = 0; ok && off < R; off += arows_) {
            X_ = x0 + (size_t) off * 4 * g_.n_embd;
            ok = engram_step(l, pos0 + off, std::min(arows_, R - off), tokens + off, err, off);
        }
        X_ = x0;
        return ok;
    }

    // What the Engram reads of a prompt need is known from the token ids alone: every Engram layer's rows for the whole chunk are
    // asked for before the first layer runs, so the SSD works while the GPU does the layers in between.
    struct Prefetch {
        std::vector<float> valid;
        std::vector<uint8_t> raw;    // the rows read here, in order
        std::vector<int> at;         // [row] its place in raw, or -1 for a row asked for early
        ngram::PleReader::Ticket tk;
        bool issued = false, collected = false;
    };
    std::vector<Prefetch> pf_;
    // A decode window's rows are known before it runs: the anchor's when the drafter starts, each draft's as the heads make it.
    // Their Engram rows are asked for then (every Engram layer's rows of a window row in one ticket), so they land during the
    // draft instead of during layer 0; the window takes a row whose position, token and two tokens before match.
    struct EarlyRow {
        int64_t pos = 0;
        int32_t tok = 0, p1 = 0, p2 = 0;
        float valid = 1.f;
        std::vector<uint8_t> raw;    // [Engram layer][column][row bytes]
        ngram::PleReader::Ticket tk;
        bool collected = false;
    };
    std::vector<EarlyRow> early_;
    std::vector<int> early_of_;      // [window row] its early row, or -1
    int64_t early_pos_ = -1;         // the window the drafter is making
    int32_t early_tok_[kMaxWindow] = {};

    // A prompt chunk of kRingRows rows or more routes to every expert of every layer, so the next layer's experts do not wait for
    // its router: all of a layer's experts are copied into one of two slots of the ring while the layer before it computes.
    static constexpr int kRingRows = 1024;
    static constexpr int kShortChunk = 512;   // the largest chunk of a short prompt, which keeps the card's expert slots
    bool lent_ = false;                       // the open prompt lent the tier's slots
    uint8_t* ring_ = nullptr;
    cudaEvent_t ring_copied_[2] = {nullptr, nullptr}, ring_done_[2] = {nullptr, nullptr};
    int ring_layer_[2] = {-1, -1};
    std::thread ring_thr_[2];
    bool ring_ok_[2] = {false, false};

    size_t expert_ring_bytes() const { return 2 * (size_t) g_.n_expert * stride_; }
    bool ring_open() {
        if (cudaMalloc((void**) &ring_, expert_ring_bytes()) != cudaSuccess) {
            cudaGetLastError();
            ring_ = nullptr;
            return false;
        }
        for (int i = 0; i < 2; ++i) {
            if (cudaEventCreateWithFlags(&ring_copied_[i], cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&ring_done_[i], cudaEventDisableTiming) != cudaSuccess ||
                cudaEventRecord(ring_done_[i], stream_) != cudaSuccess) {
                cudaGetLastError();
                ring_close();
                return false;
            }
        }
        ring_layer_[0] = ring_layer_[1] = -1;
        return true;
    }
    void ring_join() {
        for (std::thread& t : ring_thr_)
            if (t.joinable()) t.join();
    }
    void ring_close() {
        if (!ring_) return;
        ring_join();
        cudaStreamSynchronize(copy_);
        cudaStreamSynchronize(stream_);
        cudaFree(ring_);
        ring_ = nullptr;
        for (int i = 0; i < 2; ++i) {
            if (ring_copied_[i]) cudaEventDestroy(ring_copied_[i]);
            if (ring_done_[i]) cudaEventDestroy(ring_done_[i]);
            ring_copied_[i] = ring_done_[i] = nullptr;
        }
    }
    // All the experts of tier layer `tl` into its slot, on the copy stream once the layer that used the slot is done.
    bool ring_copy(int tl) {
        const int slot = tl & 1;
        const ExpertLayer& el = m_.pack().experts[(size_t) (tl + g_.n_dense)];
        uint8_t* base = ring_ + (size_t) slot * g_.n_expert * stride_;
        cudaStreamWaitEvent(copy_, ring_done_[slot], 0);
        // runs of experts in RAM at an even spacing go in one strided copy
        const std::vector<uint8_t>* away = (size_t) tl < away_.size() && !away_[(size_t) tl].empty() ? &away_[(size_t) tl] : nullptr;
        auto gone = [&](int x) { return away && (*away)[(size_t) x]; };   // the second card computes it
        int e = 0;
        while (e < g_.n_expert) {
            if (gone(e)) {
                ++e;
                continue;
            }
            const core::ExpertTier::Source src = tier_->locate(tl, e);
            uint8_t* dst = base + stride_ * (size_t) e;
            if (src.pinned) {
                int n = 1;
                ptrdiff_t gap = 0;
                while (e + n < g_.n_expert) {
                    if (gone(e + n)) break;
                    const core::ExpertTier::Source nx = tier_->locate(tl, e + n);
                    if (!nx.pinned) break;
                    const ptrdiff_t d = (const uint8_t*) nx.pinned - (const uint8_t*) src.pinned - (ptrdiff_t) (n - 1) * gap;
                    if (n == 1) gap = (const uint8_t*) nx.pinned - (const uint8_t*) src.pinned;
                    else if (d != gap) break;
                    if (gap < (ptrdiff_t) el.blob_bytes) break;
                    ++n;
                }
                if (n > 1) {
                    cudaMemcpy2DAsync(dst, stride_, src.pinned, (size_t) gap, el.blob_bytes, (size_t) n, cudaMemcpyHostToDevice, copy_);
                } else {
                    cudaMemcpyAsync(dst, src.pinned, el.blob_bytes, cudaMemcpyHostToDevice, copy_);
                }
                e += n;
                continue;
            }
            if (src.dev) cudaMemcpyAsync(dst, src.dev, el.blob_bytes, cudaMemcpyDeviceToDevice, copy_);
            else if (src.peer) cudaMemcpyPeerAsync(dst, m_.device(), src.peer, tier_->peer_for(tl)->device(), el.blob_bytes, copy_);
            else return false;
            ++e;
        }
        cudaEventRecord(ring_copied_[slot], copy_);
        return true;
    }
    // The copy is issued from a thread of its own: hundreds of copy calls can hold the caller until the bus has taken most of them.
    void ring_start(int tl, int R) {
        const int slot = tl & 1;
        if (ring_thr_[slot].joinable()) ring_thr_[slot].join();
        ring_layer_[slot] = tl;
        prepare_away(tl, R);
        ring_ok_[slot] = false;
        const int dev = m_.device();
        ring_thr_[slot] = std::thread([this, tl, slot, dev] {
            cudaSetDevice(dev);
            const double t0 = now_ms();
            ring_ok_[slot] = ring_copy(tl);
            t_ring_ms_ += now_ms() - t0;
        });
    }
    // Layer `tl`'s experts are in their slot (waits for the thread that issued the copy); false when they could not be.
    bool ring_ready(int tl, int R) {
        const int slot = tl & 1;
        if (ring_layer_[slot] != tl) ring_start(tl, R);
        if (ring_thr_[slot].joinable()) ring_thr_[slot].join();
        return ring_ok_[slot];
    }
    // At the start of layer `l`: the next layer with routed experts starts its copy.
    void ring_ahead(int l, int R) {
        if (!ring_ || !tier_ || R < kRingRows) return;
        const int target = l < g_.n_dense ? 0 : l - g_.n_dense + 1;
        if (target >= g_.n_layers - g_.n_dense) return;
        if (ring_layer_[target & 1] != target) ring_start(target, R);
    }

    // A short chunk's experts the CPU computes: their jobs, (token, slot) pairs in row order, and the thread that runs them.
    static constexpr int kCpuRows = 8;          // a RAM expert with this few rows in a short chunk may go to the CPU
    static constexpr int kCpuMaxRows = 1024;    // the most rows the CPU takes of a layer
    // The split's prices (prefill::cpu_take), measured on an RTX 5090 (x8 PCIe 5) with 15 pool workers over short turns: the CPU reads
    // about 2.67 GB/s of expert weights a worker; the main card stages and runs RAM experts at 0.85 of its link's rate.
    static constexpr double kCpuWorkerBytesPerMs = 2.67e6, kCpuMsPerRow = 0.006, kCardLinkShare = 0.85;
    double link_bytes_ = -1;   // the main card's link (NVML), read once
    prefill::CpuBusRates cpu_rates() {
        if (link_bytes_ < 0) link_bytes_ = core::device_rates(m_.device()).link_bytes;
        prefill::CpuBusRates r;
        r.cpu_bytes_per_ms = kCpuWorkerBytesPerMs * tier_->pool()->workers();
        r.cpu_ms_per_row = kCpuMsPerRow;
        r.card_bytes_per_ms = link_bytes_ / 1000.0 * kCardLinkShare;
        return r;
    }
    struct CpuShare {
        std::vector<kernels::cpu::ExpertJobMulti> jobs;
        std::vector<int32_t> pair, act_tokens;
        int64_t at = 0;
        double ms = 0;   // the thread's time
        std::thread th;
        void join() {
            if (th.joinable()) th.join();
        }
        ~CpuShare() { join(); }
    };
    float *cpu_rows_ = nullptr, *cpu_x_ = nullptr;   // pinned: the CPU's output rows, a short chunk's latent rows
    uint8_t* cpu_act_ = nullptr;                     // the latent rows quantised for the CPU's products
    // Takes out of `used` the RAM experts of tier layer tl the CPU computes, fewest rows first, while the bus would still carry as many
    // experts as the CPU has rows, and starts them.
    bool plan_cpu(int tl, int R, const std::vector<std::vector<int32_t>>& by, std::vector<int>& used, CpuShare& s, std::string& err) {
        const int K = g_.n_used, L = g_.n_latent;
        if (!tier_ || !tier_->pool() || R > kRingRows) return true;
        int bus = 0;
        for (int e : used) {
            const core::ExpertTier::Source src = tier_->locate(tl, e);
            bus += src.pinned && !src.dev;
        }
        if (bus == 0) return true;
        if (!cpu_rows_) {
            if (cudaMallocHost((void**) &cpu_rows_, (size_t) kCpuMaxRows * L * 4) != cudaSuccess ||
                cudaMallocHost((void**) &cpu_x_, (size_t) kRingRows * L * 4) != cudaSuccess ||
                !(cpu_act_ = (uint8_t*) platform::alloc_aligned((size_t) kRingRows * kernels::cpu::kNativeActBytes, 64))) {
                err = "cannot allocate the CPU share's rows";
                return false;
            }
            pinned_.push_back(cpu_rows_);
            pinned_.push_back(cpu_x_);
        }
        // the candidates, fewest rows first, and how many of them the priced times give the CPU
        std::vector<int> cand, cand_rows;
        int crow = 0;
        for (int c = 1; c <= kCpuRows; ++c)
            for (int e : used) {
                const core::ExpertTier::Source src = tier_->locate(tl, e);
                if (!src.pinned || src.dev || (int) by[(size_t) e].size() != c || crow + c > kCpuMaxRows) continue;
                cand.push_back(e);
                cand_rows.push_back(c);
                crow += c;
            }
        // one card: the priced split; with a second card reading its share from the same memory, DeepSeek's rule (the CPU's rows stay
        // below the experts the bus still carries), which the priced split did not beat there
        int take = 0;
        if (tier_->peer_share()) {
            int r = 0;
            while (take < (int) cand.size() && r + cand_rows[(size_t) take] <= bus - take - 1) r += cand_rows[(size_t) take++];
        } else {
            take = prefill::cpu_take(cpu_rates(), (double) m_.pack().experts[(size_t) (tl + g_.n_dense)].blob_bytes, bus, cand_rows);
        }
        std::vector<uint8_t> takes((size_t) g_.n_expert, 0);
        std::vector<int32_t> tok_row((size_t) R, -1);
        int rows = 0;
        for (int ci = 0; ci < take; ++ci) {
            const int e = cand[(size_t) ci], c = cand_rows[(size_t) ci];
            const core::ExpertTier::Source src = tier_->locate(tl, e);
            kernels::cpu::ExpertJobMulti j;
            j.blob = src.pinned;
            j.nt = c;
            for (int i = 0; i < c; ++i) {
                const int32_t p = by[(size_t) e][(size_t) i], t = p / K;
                if (tok_row[(size_t) t] < 0) {
                    tok_row[(size_t) t] = (int32_t) s.act_tokens.size();
                    s.act_tokens.push_back(t);
                }
                j.nact[i] = cpu_act_ + (size_t) tok_row[(size_t) t] * kernels::cpu::kNativeActBytes;
                j.out[i] = cpu_rows_ + (size_t) rows * L;
                s.pair.push_back(p);
                ++rows;
            }
            s.jobs.push_back(j);
            takes[(size_t) e] = 1;
        }
        if (s.jobs.empty()) return true;
        used.erase(std::remove_if(used.begin(), used.end(), [&](int e) { return takes[(size_t) e] != 0; }), used.end());
        if (cudaMemcpy(cpu_x_, xlat_, (size_t) R * L * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "reading the chunk's latent rows for the CPU failed";
            return false;
        }
        n_cpu_exp_ += (int64_t) s.jobs.size();
        s.th = std::thread([this, tl, &s, L, tc = now_ms()] {
            const kernels::cpu::NativeFmt& f = tier_->fmt(tl);
            for (size_t i = 0; i < s.act_tokens.size(); ++i)
                kernels::cpu::native_quant_act(f, cpu_x_ + (size_t) s.act_tokens[i] * L, cpu_act_ + i * kernels::cpu::kNativeActBytes);
            tier_->pool()->run_split_multi_native(f, s.jobs.data(), (int) s.jobs.size());
            s.ms = now_ms() - tc;
            t_cpu_run_ms_ += s.ms;
        });
        return true;
    }

    // The second card's share of tier layer `tl` (prefill::PeerShare): its formats and the clamped SwiGLU with an NVFP4 blob's scales.
    prefill::PeerShare::Layer share_layer(int tl) const {
        const int l = tl + g_.n_dense;
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        const kernels::NativeExpertLayout lay = kernels::native_expert_layout(el.gu_type, el.d_type, g_.n_latent, g_.n_ff_exp);
        prefill::PeerShare::Layer sl;
        sl.gu_type = el.gu_type;
        sl.d_type = el.d_type;
        sl.blob_bytes = el.blob_bytes;
        sl.stride = sl.gu_stride = sl.d_stride = stride_;
        sl.down_off = lay.down_off;
        const int F = g_.n_ff_exp;
        const float limit = g_.clamp_exp[(size_t) l];
        const bool fp4 = el.gu_type == kTypeNvfp4;
        const size_t stride = stride_, tail = (size_t) el.blob_bytes - 16;
        sl.act = [=](const float* gu, float* h, const prefill::PeerShare::Launch& x, void* stream) {
            const ExpertTails t{x.stage + stride * (size_t) x.q0, stride, tail, x.bounds, x.n};
            swiglu_split_clamped(gu, h, x.rows, F, limit, stream, fp4 ? &t : nullptr);
        };
        return sl;
    }
    // Of tier layer `tl`'s experts with rows[e] rows each, those the second card computes: its own, and the RAM ones (highest
    // ids first) its link and products price lower than the main card's (prefill::peer_take).
    // In a ringed chunk of `tokens` rows the main card's link reads ahead while the layer's attention runs (kPreRouteCycles a token).
    static constexpr double kPreRouteCycles = 1.7e6;   // main card SM-cycles a row takes in a layer before its routing (measured)
    std::vector<uint8_t> away_of(int tl, const std::vector<int64_t>& rows, int tokens = 0) const {
        const int NE = g_.n_expert;
        std::vector<uint8_t> away((size_t) NE, 0);
        const ExpertLayer& el = m_.pack().experts[(size_t) (tl + g_.n_dense)];
        std::vector<int> ram;
        std::vector<int64_t> ram_rows;
        int64_t main_rows = 0, peer_rows = 0;
        double main_bytes = 0;
        for (int e = NE - 1; e >= 0; --e) {
            if (rows[(size_t) e] == 0) continue;
            const core::ExpertTier::Source src = tier_->locate(tl, e);
            if (src.peer) {
                away[(size_t) e] = 1;
                peer_rows += rows[(size_t) e];
            } else if (src.pinned) {
                ram.push_back(e);
                ram_rows.push_back(rows[(size_t) e]);
            } else {
                main_rows += rows[(size_t) e];
                if (!src.dev) main_bytes += (double) el.blob_bytes;
            }
        }
        const int L = g_.n_latent;
        const prefill::ShareRates& r = tier_->share_rates(tl);
        const double lead = r.sm_hz0 > 0 ? (double) tokens * kPreRouteCycles / r.sm_hz0 : 0;
        const int take = prefill::peer_take(r, ram_rows, main_rows, main_bytes, peer_rows, (double) el.blob_bytes,
                                            (double) (basalt::prefill::mmq::q8_bytes(2, L) - basalt::prefill::mmq::q8_bytes(1, L)), 4.0 * L, lead);
        for (int i = 0; i < take; ++i) away[(size_t) ram[(size_t) i]] = 1;
        return away;
    }
    // A short chunk: from its routing.
    std::vector<uint8_t> plan_away(int tl, const std::vector<std::vector<int32_t>>& by) const {
        std::vector<int64_t> rows((size_t) g_.n_expert);
        for (int e = 0; e < g_.n_expert; ++e) rows[(size_t) e] = (int64_t) by[(size_t) e].size();
        return away_of(tl, rows);
    }
    // A ringed chunk of R rows routes to every expert: the plan takes an even share of the rows each, before the routing, so the ring
    // copy leaves the second card's experts out.
    void prepare_away(int tl, int R) {
        prefill::PeerShare* share = tier_ ? tier_->share_for(tl) : nullptr;
        if (away_.size() != (size_t) (g_.n_layers - g_.n_dense)) away_.assign((size_t) (g_.n_layers - g_.n_dense), {});
        away_[(size_t) tl].clear();
        if (!share) return;
        away_[(size_t) tl] = away_of(tl, std::vector<int64_t>((size_t) g_.n_expert, std::max<int64_t>(1, (int64_t) R * g_.n_used / g_.n_expert)), R);
    }
    // The second card starts staging tier layer tl's planned experts (after the layer before it has run its share).
    void stage_away(int tl) {
        prefill::PeerShare* share = tier_ ? tier_->share_for(tl) : nullptr;
        if (!share || (size_t) tl >= away_.size() || away_[(size_t) tl].empty()) return;
        std::vector<prefill::PeerShare::Expert> ex;
        for (int e = 0; e < g_.n_expert; ++e) {
            if (!away_[(size_t) tl][(size_t) e]) continue;
            const core::ExpertTier::Source src = tier_->locate(tl, e);
            prefill::PeerShare::Expert x1;
            x1.blob = src.peer ? src.peer : src.pinned;
            x1.on_peer = src.peer != nullptr;
            ex.push_back(x1);
        }
        if (!ex.empty()) share->prepare(share_layer(tl), ex);
    }
    std::vector<std::vector<uint8_t>> away_;
    int64_t n_stage_dev_ = 0, n_stage_bus_ = 0, n_stage_peer_ = 0, n_cpu_exp_ = 0;   // a prompt's staged experts by source, its CPU experts
    std::atomic<double> t_cpu_run_ms_{0};
    double t_cpu_wait_ms_ = 0, t_peer_wait_ms_ = 0, t_gpu_wait_ms_ = 0;
    int64_t n_peer_rows_ = 0, n_cpu_rows_ = 0;   // a prompt's expert rows the second card and the CPU computed   // [tier layer] the experts the second card computes in this chunk (ringed)

    void prefetch_engram(int64_t pos0, int R) {
        const EngramInfo& en = m_.pack().engram;
        const int n_cols = en.n_cols(), W = en.row_values;
        early_of_.assign((size_t) R, -1);
        std::vector<int> own;   // the rows not asked for early
        for (int t = 0; t < R; ++t) {
            const int64_t pos = pos0 + t;
            const int32_t tok = hist_[(size_t) pos], p1 = pos >= 1 ? hist_[(size_t) pos - 1] : -1, p2 = pos >= 2 ? hist_[(size_t) pos - 2] : -1;
            for (size_t k = 0; k < early_.size() && early_of_[(size_t) t] < 0; ++k) {
                const EarlyRow& er = early_[k];
                if (er.pos == pos && er.tok == tok && er.p1 == p1 && er.p2 == p2) early_of_[(size_t) t] = (int) k;
            }
            if (early_of_[(size_t) t] < 0) own.push_back(t);
        }
        pf_.assign(en.layers.size(), Prefetch());
        for (size_t e = 0; e < en.layers.size(); ++e) {
            Prefetch& p = pf_[e];
            std::vector<uint32_t> ids(own.size() * n_cols);
            p.valid.assign((size_t) R, 1.f);
            p.at.assign((size_t) R, -1);
            for (int t = 0; t < R; ++t)
                if (early_of_[(size_t) t] >= 0) p.valid[(size_t) t] = early_[(size_t) early_of_[(size_t) t]].valid;
            for (size_t i = 0; i < own.size(); ++i) {
                const int t = own[i];
                const int64_t pos = pos0 + t;
                const int32_t p1 = pos >= 1 ? hist_[(size_t) pos - 1] : -1, p2 = pos >= 2 ? hist_[(size_t) pos - 2] : -1;
                engram_ids(en, (int) e, hist_[(size_t) pos], p1, p2, pos, false, ids.data() + i * n_cols, p.valid[(size_t) t]);
                p.at[(size_t) t] = (int) i;
            }
            if (own.empty()) continue;
            p.raw.resize(own.size() * n_cols * W);
            p.tk = m_.engram_reader().issue(ids.data(), ids.size(), p.raw.data());
            p.issued = true;
        }
    }
    // Waits for the reads still in flight (their buffers are ours).
    void drain_prefetch() {
        std::string ignore;
        for (Prefetch& p : pf_)
            if (p.issued && !p.collected) m_.engram_reader().collect(p.tk, ignore);
        pf_.clear();
        settle_early();
    }
    // Every Engram layer's rows of window row (pos, tok) after p1 and p2, in one ticket.
    void issue_early(int64_t pos, int32_t tok, int32_t p1, int32_t p2) {
        const EngramInfo& en = m_.pack().engram;
        if (en.layers.empty() || early_.size() >= (size_t) kMaxWindow) return;
        const int n_cols = en.n_cols(), W = en.row_values, ne = (int) en.layers.size();
        EarlyRow r;
        r.pos = pos;
        r.tok = tok;
        r.p1 = p1;
        r.p2 = p2;
        std::vector<uint32_t> ids((size_t) ne * n_cols);
        for (int e = 0; e < ne; ++e) engram_ids(en, e, tok, p1, p2, pos, false, ids.data() + (size_t) e * n_cols, r.valid);
        r.raw.resize((size_t) ne * n_cols * W);
        r.tk = m_.engram_reader().issue(ids.data(), ids.size(), r.raw.data());
        early_.push_back(std::move(r));
    }
    // The drafter starts the window at `pos`: its first row, the anchor, has all it needs.
    void early_window(int32_t anchor, int64_t pos) {
        settle_early();
        if (pos > (int64_t) hist_.size()) return;
        early_pos_ = pos;
        early_tok_[0] = anchor;
        issue_early(pos, anchor, pos >= 1 ? hist_[(size_t) pos - 1] : -1, pos >= 2 ? hist_[(size_t) pos - 2] : -1);
    }
    // The heads made draft j - 1, window row j's token (DflyRunner::finish's on_row: the card runs the next head meanwhile).
    void early_draft_row(int j, int32_t tok) {
        if (early_pos_ < 0 || j < 1 || j >= kMaxWindow) return;
        early_tok_[j] = tok;
        const int64_t pos = early_pos_ + j;
        const int32_t p2 = j >= 2 ? early_tok_[j - 2] : early_pos_ >= 1 ? hist_[(size_t) early_pos_ - 1] : -1;
        issue_early(pos, tok, early_tok_[j - 1], p2);
    }
    // The early rows' reads land (their buffers are ours) and are dropped.
    void settle_early() {
        std::string ignore;
        for (EarlyRow& r : early_)
            if (!r.collected) m_.engram_reader().collect(r.tk, ignore);
        early_.clear();
        early_pos_ = -1;
    }

    bool engram_step(int l, int64_t pos0, int R, const int32_t* tokens, std::string& err, int off = 0) {
        const EngramInfo& en = m_.pack().engram;
        int e = -1;
        for (size_t k = 0; k < en.layers.size(); ++k)
            if (en.layers[k] == l) e = (int) k;
        if (e < 0) return true;
        // The prompt line reports a prompt, so a decode that follows one adds nothing to it.
        const bool time_prompt = big_.rows != 0;
        struct Lap {
            double* acc;
            double start;
            ~Lap() { if (acc) *acc += now_ms() - start; }
        } lap{time_prompt ? &t_eng_ms_ : nullptr, time_prompt ? now_ms() : 0};
        const auto collect = [&](ngram::PleReader::Ticket ticket) {
            Lap wait{time_prompt ? &t_eng_read_ms_ : nullptr, time_prompt ? now_ms() : 0};
            return m_.engram_reader().collect(ticket, err);
        };
        const int n_cols = en.n_cols(), W = en.row_values;
        static const std::vector<float> lut = [] {
            std::vector<float> t(256);
            for (int i = 0; i < 256; ++i) t[(size_t) i] = e4m3_to_f32((uint8_t) i);
            return t;
        }();
        if (eng_ev_) cudaEventSynchronize(eng_ev_);   // the last Engram layer's copies out of these host rows have landed
        Prefetch* pf = (size_t) e < pf_.size() ? &pf_[(size_t) e] : nullptr;
        std::vector<uint8_t> own_raw;
        const uint8_t* raw = nullptr;
        if (pf) {
            if (pf->issued && !pf->collected) {
                if (!collect(pf->tk)) return false;
                pf->collected = true;
            }
            for (int t = 0; t < R; ++t) {
                const int k = early_of_[(size_t) (off + t)];
                if (k >= 0 && !early_[(size_t) k].collected) {
                    if (!collect(early_[(size_t) k].tk)) return false;
                    early_[(size_t) k].collected = true;
                }
                h_evalid_[t] = pf->valid[(size_t) (off + t)];
            }
        } else {
            std::vector<uint32_t> ids((size_t) R * n_cols);
            for (int t = 0; t < R; ++t) {
                const int64_t pos = row_pos(pos0 - off, off + t);
                const std::vector<int32_t>& hist = row_hist(off + t);
                const int32_t p1 = pos >= 1 ? hist[(size_t) pos - 1] : -1, p2 = pos >= 2 ? hist[(size_t) pos - 2] : -1;
                float valid = 1.f;
                engram_ids(en, e, tokens[t], p1, p2, pos, false, ids.data() + (size_t) t * n_cols, valid);
                h_evalid_[t] = valid;
            }
            own_raw.resize((size_t) R * n_cols * W);
            ngram::PleReader& rd = m_.engram_reader();
            const ngram::PleReader::Ticket tk = rd.issue(ids.data(), ids.size(), own_raw.data());
            if (!collect(tk)) return false;
            raw = own_raw.data();
        }
        if (lookup_host_.empty()) lookup_host_.resize((size_t) n_cols * W * en.layers.size());
        float* nrm = lookup_host_.data() + (size_t) e * n_cols * W;
        if (lookup_loaded_.empty()) lookup_loaded_.assign(en.layers.size(), false);
        if (!lookup_loaded_[(size_t) e]) {
            cudaMemcpy(nrm, T(l, "engram_lookup_norm.weight")->ptr, (size_t) n_cols * W * 4, cudaMemcpyDeviceToHost);
            lookup_loaded_[(size_t) e] = true;
        }
        // row t's n_cols rows: from the window's own reads or from its early ones
        const auto rows_of = [&](int t) -> const uint8_t* {
            if (!pf) return raw + (size_t) t * n_cols * W;
            const int k = early_of_[(size_t) (off + t)];
            return k >= 0 ? early_[(size_t) k].raw.data() + (size_t) e * n_cols * W : pf->raw.data() + (size_t) pf->at[(size_t) (off + t)] * n_cols * W;
        };
        const double norm_start = time_prompt ? now_ms() : 0;
        for (int t = 0; t < R; ++t) {
            const float valid = h_evalid_[t];
            const uint8_t* trow = rows_of(t);
            for (int c = 0; c < n_cols; ++c) {
                float* row = h_erows_ + ((size_t) t * n_cols + c) * W;
                const uint8_t* src = trow + (size_t) c * W;
                float ss = 0.f;
                for (int d = 0; d < W; ++d) {
                    row[d] = lut[src[d]];
                    ss += row[d] * row[d];
                }
                const float r = 1.0f / std::sqrt(ss / (float) W + 1.1920929e-07f);
                for (int d = 0; d < W; ++d) row[d] = row[d] * r * nrm[(size_t) c * W + d] * valid;
            }
        }
        if (time_prompt) t_eng_norm_ms_ += now_ms() - norm_start;
        put_f32(erows_, h_erows_, (int64_t) R * n_cols * W);
        put_f32(evalid_, h_evalid_, R);
        if (!eng_ev_) cudaEventCreateWithFlags(&eng_ev_, cudaEventDisableTiming);
        cudaEventRecord(eng_ev_, stream_);
        gemm(T(l, "engram_kv.weight"), erows_, ekv_, n_cols * W, 5 * g_.n_embd, R);
        EngramArgs a;
        a.n_embd = g_.n_embd;
        engram_apply(ekv_, F(l, "engram_kv_b.weight"), F(l, "engram_gate.weight"), F(l, "engram_pre_gamma.weight"),
                     F(l, "engram_hc_scale.weight"), evalid_, R, a, X_, stream_);
        return true;
    }

    bool ffn(int l, int R, std::string& err) {
        const int E = g_.n_embd;
        if (l < g_.n_dense) {
            float *xn0 = xn_, *ml0 = mlp_;
            for (int off = 0; off < R; off += arows_) {
                const int n = std::min(arows_, R - off);
                xn_ = xn0 + (size_t) off * E;
                mlp_ = ml0 + (size_t) off * E;
                float *gate = gu_, *up = gu_ + (size_t) n * g_.n_ff;
                gemm(T(l, "ffn_gate.weight"), xn_, gate, E, g_.n_ff, n);
                gemm(T(l, "ffn_up.weight"), xn_, up, E, g_.n_ff, n);
                bk::swiglu_scaled(gate, up, nullptr, nullptr, h_, n, g_.n_ff, g_.n_ff, stream_);
                gemm(T(l, "ffn_down.weight"), h_, mlp_, g_.n_ff, E, n);
            }
            xn_ = xn0;
            mlp_ = ml0;
            return true;
        }
        return moe(l, R, err);
    }

    // The expert tier, opened by the first session while the model starts and shared by the rest: the sessions run one at a
    // time over the same pinned rows.
    bool ensure_tier(std::string& err) {
        if (tier_) return true;
        Hy4Model::SharedTier& shared = m_.shared_tier();
        constexpr unsigned kMapped = cudaHostAllocPortable | cudaHostAllocMapped;
        if (!shared.tier) {
            if (cudaHostAlloc((void**) &shared.h_x, (size_t) kMaxWindow * g_.n_latent * 4, kMapped) != cudaSuccess ||
                cudaHostAlloc((void**) &shared.h_rows, (size_t) kLookupK * g_.n_latent * 4, kMapped) != cudaSuccess) {
                err = "cannot allocate the pinned expert staging";
                return false;
            }
            auto t = std::make_shared<core::ExpertTier>(m_.tier_spec());
            core::ExpertTier::Reserve r;
            r.copies = kModuleRoom;
            r.homes = kPromptRoom + kModuleRoom;
            r.beside_homes = 1ull << 30;
            r.modules = kModuleRoom;
            core::LoadPlan* reads = m_.planning();
            if (reads == nullptr) {
                err = "the expert tier opens only while the model starts";
                return false;
            }
            if (!t->open(r, shared.h_rows, *reads, err)) return false;
            shared.tier = std::move(t);
        }
        tier_ = shared.tier;
        h_x_ = shared.h_x;
        h_rows_ = shared.h_rows;
        if (cudaHostGetDevicePointer((void**) &d_rows_, h_rows_, 0) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &d_x_, h_x_, 0) != cudaSuccess) {
            err = "cannot map the expert rows";
            return false;
        }
        return true;
    }

    // The latent MoE of a layer over its R normed rows: the router and routed_down and the shared expert for all rows, the
    // routed experts in groups of up to 8 rows (the tier places them: the card's slots and the second cards' on the GPUs, the rest
    // on the CPU), then routed_up.
    bool moe(int l, int R, std::string& err) {
        const int E = g_.n_embd, K = g_.n_used, L = g_.n_latent;
        const int tl = l - g_.n_dense;   // the tier's layer
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        kernels::NativeExpertLayout lay = kernels::native_expert_layout(el.gu_type, el.d_type, L, g_.n_ff_exp);
        if (lay.bytes != el.blob_bytes) {
            err = "layer " + std::to_string(l) + "'s expert blob size does not match the layout";
            return false;
        }
        bk::RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = E;
        ra.k = K;
        ra.weight_scale = g_.expert_scale;
        ra.gate = bk::RouterGate::Sigmoid;
        const bool prompt = R > kMaxWindow && basalt::prefill::mmq::built();
        moe_head(l, R, prompt);
        if (prompt) {
            if (!moe_prompt(l, tl, R, lay, err)) return false;
        } else {
            for (int g0 = 0; g0 < R; g0 += kMaxWindow)
                if (!moe_group(l, tl, lay, g0, std::min(kMaxWindow, R - g0), err)) return false;
        }
        if (diag(l)) dump("moe_lat-" + std::to_string(l), lat_out_, (int64_t) R * L);
        if (diag(l)) dump("xlat-" + std::to_string(l), xlat_, (int64_t) R * L);
        moe_tail(l, R);
        return true;
    }

    // The router, routed_down and, for a window, its first group to the host; the shared expert behind them, which runs while
    // the host places the experts. No position in it: a decode window replays it from a graph.
    void moe_head(int l, int R, bool prompt) {
        const int E = g_.n_embd, K = g_.n_used, L = g_.n_latent;
        bk::RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = E;
        ra.k = K;
        ra.weight_scale = g_.expert_scale;
        ra.gate = bk::RouterGate::Sigmoid;
        const bool lossy = lossy_ && !prompt;
        if (lossy) {   // a decode window with the lossy routing flags: the experts the GPUs hold
            ra.held = lossy_->held_tabs(l - g_.n_dense);
            ra.prior = lossy_->route_prior();
        }
        bk::router(xn_, F(l, "ffn_gate_inp.weight"), F(l, "exp_probs_b.bias"), R, ra, rlog_, ids_, rw_, stream_);
        if (lossy && lossy_->route_tail_skip() > 0)   // each group of rows the tier places together
            for (int g0 = 0; g0 < R; g0 += kMaxWindow)
                bk::route_tail_skip(ids_ + (size_t) g0 * K, rw_ + (size_t) g0 * K, ra.held, std::min(kMaxWindow, R - g0) * K, K,
                                    lossy_->route_tail_skip(), g_.n_expert, lossy_->route_counts(), stream_);
        quant_n(R);
        gemm_n(T(l, "ffn_routed_down.weight"), xlat_, L, R);
        if (!prompt) publish(0, std::min(kMaxWindow, R));
        float *sg = gu_, *su = gu_ + (size_t) R * g_.n_ff_exp;
        gemm_n(T(l, "ffn_gate_shexp.weight"), sg, g_.n_ff_exp, R);
        gemm_n(T(l, "ffn_up_shexp.weight"), su, g_.n_ff_exp, R);
        bk::swiglu_scaled(sg, su, nullptr, nullptr, h_, R, g_.n_ff_exp, g_.n_ff_exp, stream_);
        gemm(T(l, "ffn_down_shexp.weight"), h_, sh_, g_.n_ff_exp, E, R);
    }
    void moe_tail(int l, int R) {
        gemm(T(l, "ffn_routed_up.weight"), lat_out_, moe_out_, g_.n_latent, g_.n_embd, R);
        add_vectors(moe_out_, sh_, mlp_, R * g_.n_embd, stream_);
    }

    // ---- CUDA graphs of a decode window's layers: the stretches with no position in them, captured once per layer and row
    // count and replayed (the launches of a token were ~2,200 one by one, and the host's launch rate set the pace).
    std::vector<cudaGraphExec_t> gx_;   // [(R - 1) * n_layers * 3 + l * 3 + part]
    std::vector<cudaGraphExec_t> gbx_;  // a batch window's, the same way (layer_batched)
    bool graphs_ = false;               // this run_rows replays them (a decode window, no dumps, no picture)
    template <class Fn>
    bool replay(int l, int R, int part, Fn&& body, std::string& err) { return replay_in(gx_, l, R, part, body, err); }
    template <class Fn>
    bool replay_in(std::vector<cudaGraphExec_t>& set, int l, int R, int part, Fn&& body, std::string& err) {
        if (set.empty()) set.assign((size_t) kMaxWindow * g_.n_layers * 3, nullptr);
        cudaGraphExec_t& ex = set[((size_t) (R - 1) * g_.n_layers + l) * 3 + part];
        if (!ex) {
            if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
                err = "cannot begin a graph capture";
                return false;
            }
            body();
            cudaGraph_t g = nullptr;
            const cudaError_t ce = cudaStreamEndCapture(stream_, &g);
            const cudaError_t ci = ce == cudaSuccess ? cudaGraphInstantiateWithFlags(&ex, g, 0) : ce;
            if (g) cudaGraphDestroy(g);
            if (ci != cudaSuccess) {
                ex = nullptr;
                err = std::string("graph capture: ") + cudaGetErrorString(ci);
                return false;
            }
        }
        return cudaGraphLaunch(ex, stream_) == cudaSuccess || (err = "a graph launch failed", false);
    }

    // A decode window's MoE layer: the attention's query part, the hyper-connection collapse and norm before it (graph 0);
    // the cells and the cache rows (by position); the attention's output, its expansion, the FFN's collapse, the router,
    // routed_down, the doorbell and the shared expert (graph 1); the experts (placed by the host); routed_up and the FFN's
    // expansion (graph 2). The same kernels in the same order as the direct path.
    bool layer_replayed(int l, int64_t pos0, int R, const bk::HcPlainArgs& hc, std::string& err) {
        const int E = g_.n_embd;
        const int tl = l - g_.n_dense;
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        const kernels::NativeExpertLayout lay = kernels::native_expert_layout(el.gu_type, el.d_type, g_.n_latent, g_.n_ff_exp);
        if (lay.bytes != el.blob_bytes) {
            err = "layer " + std::to_string(l) + "'s expert blob size does not match the layout";
            return false;
        }
        float *xa = X_, *xb = Xn_;
        if (g_.swa[(size_t) l]) {   // a sliding-window layer's attention is one graph: its row spans come from pos_d_ on the card
            const uint32_t seq0 = seq_want_;
            if (!replay(l, R, 0, [&] {
                    bk::hc_mixes_plain(xa, F(l, "hc_attn_fn.weight"), F(l, "hc_attn_scale.weight"), F(l, "hc_attn_base.weight"), R, 8, hc, hcws_, coef_, stream_);
                    bk::hc_pre_rms_norm(xa, coef_, bk::kHcCoef, F(l, "attn_norm.weight"), R, E, g_.rms_eps, cur_, xn_, stream_);
                    std::string e;
                    attention_step(l, pos0, R, 0, e, kQuery | kCells | kDevMeta | kOutput | kRaw);
                    add_norm_hc_post(attn_raw_, nullptr, F(l, "post_attention_norm.weight"), g_.rms_eps, xa, coef_, bk::kHcCoef, R, E, xb, stream_);
                    bk::hc_mixes_plain(xb, F(l, "hc_ffn_fn.weight"), F(l, "hc_ffn_scale.weight"), F(l, "hc_ffn_base.weight"), R, 8, hc, hcws_, coef_, stream_);
                    bk::hc_pre_rms_norm(xb, coef_, bk::kHcCoef, F(l, "ffn_norm.weight"), R, E, g_.rms_eps, cur_, xn_, stream_);
                    moe_head(l, R, false);
                }, err))
                return false;
            seq_want_ = seq0 + 1;
        } else {
        if (!replay(l, R, 0, [&] {
                bk::hc_mixes_plain(xa, F(l, "hc_attn_fn.weight"), F(l, "hc_attn_scale.weight"), F(l, "hc_attn_base.weight"), R, 8, hc, hcws_, coef_, stream_);
                bk::hc_pre_rms_norm(xa, coef_, bk::kHcCoef, F(l, "attn_norm.weight"), R, E, g_.rms_eps, cur_, xn_, stream_);
                std::string e;
                attention_step(l, pos0, R, 0, e, kQuery);
            }, err))
            return false;
        if (!attention_step(l, pos0, R, 0, err, kCells)) return false;
        const uint32_t seq0 = seq_want_;
        if (!replay(l, R, 1, [&] {
                std::string e;
                attention_step(l, pos0, R, 0, e, kOutput | kRaw);
                add_norm_hc_post(attn_raw_, nullptr, F(l, "post_attention_norm.weight"), g_.rms_eps, xa, coef_, bk::kHcCoef, R, E, xb, stream_);
                bk::hc_mixes_plain(xb, F(l, "hc_ffn_fn.weight"), F(l, "hc_ffn_scale.weight"), F(l, "hc_ffn_base.weight"), R, 8, hc, hcws_, coef_, stream_);
                bk::hc_pre_rms_norm(xb, coef_, bk::kHcCoef, F(l, "ffn_norm.weight"), R, E, g_.rms_eps, cur_, xn_, stream_);
                moe_head(l, R, false);
            }, err))
            return false;
        seq_want_ = seq0 + 1;   // the doorbell in the graph rings once a replay
        }
        std::swap(X_, Xn_);
        if (!moe_group(l, tl, lay, 0, R, err)) return false;
        xa = X_;
        xb = Xn_;
        if (!replay(l, R, 2, [&] {
                gemm(T(l, "ffn_routed_up.weight"), lat_out_, moe_out_, g_.n_latent, g_.n_embd, R);
                add_norm_hc_post(moe_out_, sh_, F(l, "post_ffw_norm.weight"), g_.rms_eps, xa, coef_, bk::kHcCoef, R, E, xb, stream_);
            }, err))
            return false;
        std::swap(X_, Xn_);
        return true;
    }

    // A batch window's MoE layer: graph 0 (the attention's collapse, norm and query products: no position, no session), the
    // cells, the cache rows and the attention per segment over each session's caches, graph 1 (the attention's output, its
    // expansion, the FFN's collapse, the router, routed_down, the doorbell and the shared expert), the experts, graph 2. The
    // graphs hold no session's state, so they are keyed by the row count alone. The same kernels in the same order as a solo
    // window's layer, per row.
    bool layer_batched(int l, int64_t pos0, int R, const bk::HcPlainArgs& hc, std::string& err) {
        const int E = g_.n_embd;
        const int tl = l - g_.n_dense;
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        const kernels::NativeExpertLayout lay = kernels::native_expert_layout(el.gu_type, el.d_type, g_.n_latent, g_.n_ff_exp);
        if (lay.bytes != el.blob_bytes) {
            err = "layer " + std::to_string(l) + "'s expert blob size does not match the layout";
            return false;
        }
        float *xa = X_, *xb = Xn_;
        if (!replay_in(gbx_, l, R, 0, [&] {
                bk::hc_mixes_plain(xa, F(l, "hc_attn_fn.weight"), F(l, "hc_attn_scale.weight"), F(l, "hc_attn_base.weight"), R, 8, hc, hcws_, coef_, stream_);
                bk::hc_pre_rms_norm(xa, coef_, bk::kHcCoef, F(l, "attn_norm.weight"), R, E, g_.rms_eps, cur_, xn_, stream_);
                std::string e;
                attention_step(l, pos0, R, 0, e, kQuery);
            }, err))
            return false;
        if (!attention_step(l, pos0, R, 0, err, kCells | kAttend)) return false;
        const uint32_t seq0 = seq_want_;
        if (!replay_in(gbx_, l, R, 1, [&] {
                std::string e;
                attention_step(l, pos0, R, 0, e, kOutput | kTail | kRaw);
                add_norm_hc_post(attn_raw_, nullptr, F(l, "post_attention_norm.weight"), g_.rms_eps, xa, coef_, bk::kHcCoef, R, E, xb, stream_);
                bk::hc_mixes_plain(xb, F(l, "hc_ffn_fn.weight"), F(l, "hc_ffn_scale.weight"), F(l, "hc_ffn_base.weight"), R, 8, hc, hcws_, coef_, stream_);
                bk::hc_pre_rms_norm(xb, coef_, bk::kHcCoef, F(l, "ffn_norm.weight"), R, E, g_.rms_eps, cur_, xn_, stream_);
                moe_head(l, R, false);
            }, err))
            return false;
        seq_want_ = seq0 + 1;   // the doorbell in the graph rings once a replay
        std::swap(X_, Xn_);
        if (!moe_group(l, tl, lay, 0, R, err)) return false;
        xa = X_;
        xb = Xn_;
        if (!replay_in(gbx_, l, R, 2, [&] {
                gemm(T(l, "ffn_routed_up.weight"), lat_out_, moe_out_, g_.n_latent, g_.n_embd, R);
                add_norm_hc_post(moe_out_, sh_, F(l, "post_ffw_norm.weight"), g_.rms_eps, xa, coef_, bk::kHcCoef, R, E, xb, stream_);
            }, err))
            return false;
        std::swap(X_, Xn_);
        return true;
    }

    // The routed experts of a prompt chunk, by expert: each used expert's blob is staged on the card (a copy from its slot, its
    // pinned RAM or the second card), its rows (the pairs routed to it, with their activations quantised for MMQ) go through the
    // gate/up product, the clamped SwiGLU and the down product, and a pair's output row is kept for the weighted sum.
    bool moe_prompt(int l, int tl, int R, const kernels::NativeExpertLayout& lay, std::string& err) {
        namespace mmq = basalt::prefill::mmq;
        const int K = g_.n_used, L = g_.n_latent, F = g_.n_ff_exp;
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        if (!mmq::fits(el.gu_type, 2 * F) || !mmq::fits(el.d_type, L)) {
            err = "the prompt MMQ path does not cover this layer's expert formats";
            return false;
        }
        if (loaded_.insert({el.gu_type, 2 * F}).second) mmq::load(el.gu_type, 2 * F);
        if (loaded_.insert({el.d_type, L}).second) mmq::load(el.d_type, L);
        cudaStream_t st = stream_;
        std::vector<int32_t> hid((size_t) R * K);
        const double t0 = now_ms();
        cudaStreamSynchronize(st);
        stage_at_ = 0;   // the stream has drained: what was staged is read
        cudaMemcpy(hid.data(), ids_, hid.size() * 4, cudaMemcpyDeviceToHost);
        const double t1 = now_ms();
        t_pre_ms_ += t1 - t0;
        std::vector<std::vector<int32_t>> by((size_t) g_.n_expert);   // pair = t * K + slot
        for (int p = 0; p < R * K; ++p) {
            if (hid[(size_t) p] < 0 || hid[(size_t) p] >= g_.n_expert) {
                err = "the router returned expert " + std::to_string(hid[(size_t) p]);
                return false;
            }
            by[(size_t) hid[(size_t) p]].push_back(p);
        }
        bool ringed = ring_ && R >= kRingRows && tier_ && ring_ready(tl, R);
        // the second card's experts: in a ringed chunk the ones planned with the ring copy (their staging may have started), else
        // the ones its link prices lower for this routing
        prefill::PeerShare* share = tier_ ? tier_->share_for(tl) : nullptr;
        std::vector<uint8_t> away;
        if (share && ringed && (size_t) tl < away_.size()) away = away_[(size_t) tl];
        else if (share) away = plan_away(tl, by);
        std::vector<int> used, peer;
        for (int e = 0; e < g_.n_expert; ++e) {
            if (!away.empty() && away[(size_t) e]) {
                if (ringed || !by[(size_t) e].empty()) peer.push_back(e);
            } else if (!by[(size_t) e].empty()) {
                used.push_back(e);
            }
        }
        const double t2 = now_ms();
        t_build_ms_ += t2 - t1;
        std::vector<int32_t> map_h((size_t) R * K);
        int64_t pos = 0;
        // a short chunk: the CPU takes RAM experts with few rows (DeepSeek's prefill_ffn rule), their rows after the second card's
        CpuShare cpu;
        if (!ringed && !plan_cpu(tl, R, by, used, cpu, err)) return false;
        if (!cpu.jobs.empty()) {
            int64_t at = 0;
            for (int e : used) at += (int64_t) by[(size_t) e].size();
            for (int e : peer) at += (int64_t) by[(size_t) e].size();
            cpu.at = at;
            for (size_t i = 0; i < cpu.pair.size(); ++i) map_h[(size_t) cpu.pair[i]] = (int32_t) (at + (int64_t) i);
            n_cpu_rows_ += (int64_t) cpu.pair.size();
        }
        if (!peer.empty()) {   // its rows after the main card's
            int64_t at = 0;
            for (int e : used) at += (int64_t) by[(size_t) e].size();
            std::vector<std::vector<int32_t>> tok(peer.size());
            std::vector<prefill::PeerShare::Expert> ex;
            int64_t end = at;
            for (size_t i = 0; i < peer.size(); ++i) {
                const int e = peer[i];
                const core::ExpertTier::Source src = tier_->locate(tl, e);
                for (int32_t p : by[(size_t) e]) {
                    tok[i].push_back(p / K);
                    map_h[(size_t) p] = (int32_t) end++;
                }
                prefill::PeerShare::Expert x1;
                x1.blob = src.peer ? src.peer : src.pinned;
                x1.on_peer = src.peer != nullptr;
                x1.tok = tok[i].data();
                x1.n = (int) tok[i].size();
                ex.push_back(x1);
            }
            if (!share->run(share_layer(tl), xlat_, R, ex, pair_rows_ + (size_t) at * L, st, err)) return false;
            n_peer_rows_ += end - at;
        }
        if (ringed && tl + 1 < g_.n_layers - g_.n_dense && ring_layer_[(tl + 1) & 1] == tl + 1) stage_away(tl + 1);
        auto launch = [&](Stage& B, const uint8_t* blobs, int nb, int64_t rows, int64_t max_rows, const std::vector<int32_t>& tok,
                          const std::vector<int32_t>& bounds) {
            put_i32(B.tok, tok.data(), tok.size());
            put_i32(B.bounds, bounds.data(), bounds.size());
            mmq::quantize(xlat_, B.tok, xq1_, el.gu_type, L, L, rows, st);
            mmq::Product p;
            p.w = blobs;
            p.type = el.gu_type;
            p.w_rows = 2 * (int64_t) F;
            p.w_cols = L;
            p.expert_bytes = stride_;
            p.n = nb;
            p.xq = xq1_;
            p.bounds = B.bounds;
            p.ids = pair_ident_;
            p.total_rows = rows;
            p.max_rows = max_rows;
            p.dst = gu_;
            p.ld_dst = p.w_rows;
            mmq_.run(p, st);
            // an NVFP4 blob's per-expert scales (its last 16 bytes), which the MMQ products leave out
            ExpertTails tails;
            if (el.gu_type == kTypeNvfp4) tails = ExpertTails{blobs, stride_, (size_t) el.blob_bytes - 16, B.bounds, nb};
            swiglu_split_clamped(gu_, h_, rows, F, g_.clamp_exp[(size_t) l], st, el.gu_type == kTypeNvfp4 ? &tails : nullptr);
            mmq::quantize(h_, nullptr, xq2_, el.d_type, F, F, rows, st);
            mmq::Product d;
            d.w = blobs + lay.down_off;
            d.type = el.d_type;
            d.w_rows = L;
            d.w_cols = F;
            d.expert_bytes = stride_;
            d.n = nb;
            d.xq = xq2_;
            d.bounds = B.bounds;
            d.ids = pair_ident_ + pos;
            d.total_rows = rows;
            d.max_rows = max_rows;
            d.dst = pair_rows_;
            d.ld_dst = L;
            mmq_.run(d, st);
            cudaEventRecord(B.done, st);
            pos += rows;
        };
        if (ringed) {
            const int slot = tl & 1;
            cudaStreamWaitEvent(st, ring_copied_[slot], 0);
            const uint8_t* base = ring_ + (size_t) slot * g_.n_expert * stride_;
            for (size_t b0 = 0; b0 < used.size();) {
                const int e0 = used[b0];
                size_t b1 = b0;
                int64_t rows = 0, max_rows = 0;
                while (b1 < used.size() && used[b1] < e0 + kStageExperts) {
                    const int64_t c = (int64_t) by[(size_t) used[b1]].size();
                    if (b1 > b0 && rows + c > kLaunchRows) break;
                    rows += c;
                    max_rows = std::max(max_rows, c);
                    ++b1;
                }
                const int nb = used[b1 - 1] - e0 + 1;
                std::vector<int32_t> tok, bounds{0};
                for (int e = e0; e < e0 + nb; ++e) {
                    if (!away.empty() && away[(size_t) e]) {   // the second card's: no rows here
                        bounds.push_back((int32_t) tok.size());
                        continue;
                    }
                    for (int32_t p : by[(size_t) e]) {
                        map_h[(size_t) p] = (int32_t) (pos + (int64_t) tok.size());
                        tok.push_back(p / K);
                    }
                    bounds.push_back((int32_t) tok.size());
                }
                launch(stage_[launch_++ & 1], base + (size_t) e0 * stride_, nb, rows, max_rows, tok, bounds);
                b0 = b1;
            }
            cudaEventRecord(ring_done_[slot], st);
        } else
        for (size_t b0 = 0; b0 < used.size();) {
            int nb = 0;
            int64_t rows = 0, max_rows = 0;
            while (b0 + (size_t) nb < used.size() && nb < kStageExperts) {
                const int64_t c = (int64_t) by[(size_t) used[b0 + (size_t) nb]].size();
                if (nb > 0 && rows + c > kLaunchRows) break;
                rows += c;
                max_rows = std::max(max_rows, c);
                ++nb;
            }
            Stage& B = stage_[launch_++ & 1];
            cudaStreamWaitEvent(copy_, B.done, 0);   // the device buffers of the launch before last are free
            for (int q = 0; q < nb; ++q) {
                const int e = used[b0 + (size_t) q];
                uint8_t* dst = B.blobs + stride_ * (size_t) q;
                const core::ExpertTier::Source src = tier_->locate(tl, e);
                if (src.dev) {
                    cudaMemcpyAsync(dst, src.dev, el.blob_bytes, cudaMemcpyDeviceToDevice, copy_);
                    ++n_stage_dev_;
                } else if (src.pinned) {
                    cudaMemcpyAsync(dst, src.pinned, el.blob_bytes, cudaMemcpyHostToDevice, copy_);
                    ++n_stage_bus_;
                } else if (src.peer) {
                    ++n_stage_peer_;
                    core::PeerExperts* pe = tier_->peer_for(tl);
                    cudaMemcpyPeerAsync(dst, m_.device(), src.peer, pe->device(), el.blob_bytes, copy_);
                } else {
                    err = "expert " + std::to_string(e) + " of layer " + std::to_string(l) + " is not in RAM (a pack larger than RAM is not run yet)";
                    return false;
                }
            }
            cudaEventRecord(B.copied, copy_);
            std::vector<int32_t> tok, bounds{0};
            for (int q = 0; q < nb; ++q) {
                for (int32_t p : by[(size_t) used[b0 + (size_t) q]]) {
                    map_h[(size_t) p] = (int32_t) (pos + (int64_t) tok.size());
                    tok.push_back(p / K);
                }
                bounds.push_back((int32_t) tok.size());
            }
            cudaStreamWaitEvent(st, B.copied, 0);
            launch(B, B.blobs, nb, rows, max_rows, tok, bounds);
            b0 += (size_t) nb;
        }
        const double t3 = now_ms();
        t_issue_ms_ += t3 - t2;
        const double tw0 = now_ms();
        if (!peer.empty() && !share->join(st, err)) return false;
        const double tw1 = now_ms();
        t_peer_wait_ms_ += tw1 - tw0;
        if (!cpu.jobs.empty()) {
            cpu.join();
            t_cpu_wait_ms_ += now_ms() - tw1;
            put_f32(pair_rows_ + (size_t) cpu.at * L, cpu_rows_, (int64_t) cpu.pair.size() * L);
        }
        put_i32(pair_map_, map_h.data(), (size_t) R * K);
        weighted_sum_tokens(pair_rows_, pair_map_, rw_, R, K, L, lat_out_, st);
        const double tg = now_ms();
        cudaStreamSynchronize(st);   // the host's map is read until the copy lands
        t_gpu_wait_ms_ += now_ms() - tg;
        t_moe_ms_ += now_ms() - t1;
        return true;
    }

    // Rows g0..g0+G of the latent, their ids, to the mapped staging, behind the sequence word the host waits on; and the rows
    // quantised for the card's experts.
    void publish(int g0, int G) {
        const int K = g_.n_used, L = g_.n_latent;
        bk::doorbell_publish(xlat_ + (size_t) g0 * L, ids_ + (size_t) g0 * K, nullptr, (int64_t) G * L, (int64_t) G * K, d_x_, d_ids_,
                             nullptr, d_words_, stream_);
        ++seq_want_;
        bk::native_quantize_q8_1(xlat_ + (size_t) g0 * L, ex_xq_, L, G, stream_);
    }

    // The published ids have landed (the stream goes on meanwhile); a launch that failed never writes them.
    bool await_ids(std::string& err) {
        const volatile uint32_t* seq = h_words_;
        for (int s = 0; s < (1 << 22); ++s) {
            if (*seq == seq_want_) {
                std::atomic_thread_fence(std::memory_order_acquire);
                return true;
            }
            platform::cpu_relax();
        }
        cudaStreamSynchronize(stream_);
        if (*seq == seq_want_) return true;
        err = "the router's ids did not arrive";
        seq_want_ = *seq;
        return false;
    }

    // While a layer's weighted sum waits for the CPU's and the second card's rows, the card's memory idles once its own experts
    // are done: the next layer's two largest products, its attention gate and output (the head after the last layer), are read
    // into L2 meanwhile, up to kReadAheadBytes, so those products read L2 instead of memory. The requests start behind the
    // card's experts and stop once the wait is over; they change no result.
    static constexpr uint64_t kReadAheadBytes = 56ull << 20;
    void read_ahead(int l, uint32_t word, cudaStream_t st) {
        if (ra_.empty()) {   // every layer's ranges, once
            ra_.resize((size_t) g_.n_layers);
            for (int k = 0; k < g_.n_layers; ++k) {
                L2Ranges& r = ra_[(size_t) k];
                uint64_t left = kReadAheadBytes;
                int n = 0;
                const auto add = [&](const DevTensor* t) {
                    if (t == nullptr || n == kL2Ranges || ((uintptr_t) t->ptr & 15) != 0) return;
                    r.p[n] = t->ptr;
                    r.n[n] = std::min<uint64_t>(t->bytes, left) & ~15ull;
                    left -= r.n[n];
                    n += r.n[n] > 0;
                };
                if (k + 1 < g_.n_layers) {
                    add(T(k + 1, "attn_gate.weight"));
                    add(T(k + 1, "attn_output.weight"));
                } else {
                    add(m_.tensor("output.weight"));
                }
            }
        }
        cudaEventRecord(ra_ev_, st);
        cudaStreamWaitEvent(ra_st_, ra_ev_, 0);
        l2_read_ahead(ra_[(size_t) l], ra_mark_, word, ra_st_);
    }

    // The routed experts of rows g0..g0+G of a layer, as DeepSeek's decode places them: the tier is asked once per distinct
    // expert of the group; the second card's share is raised first, then the CPU pool starts on the RAM experts, the card
    // runs its slot hits and then the bus fetches, and the weighted sum is queued behind a word the host writes once the
    // CPU's and the second card's rows landed. The host does not wait here: join_moe, called once the next layer's work is
    // queued, finishes the CPU share and writes the word.
    bool moe_group(int l, int tl, const kernels::NativeExpertLayout& lay, int g0, int G, std::string& err) {
        const int K = g_.n_used, L = g_.n_latent;
        cudaStream_t st = stream_;
        if (g0 > 0) publish(g0, G);
        if (!join_moe(err)) return false;
        Pending& pd = pend_;
        pd = Pending{};
        pd.on = true;
        pd.tl = tl;
        // the second card's share goes out while the head runs: its kernels wait on the card for the plan
        core::PeerExperts* peer = tier_->peer_for(tl);
        if (peer) {
            kernels::NativeExpertLayout limited = lay;
            limited.swiglu_limit = g_.clamp_exp[(size_t) l];
            if (!peer->arm(limited, h_x_, G, K, core::policy::peer_groups(G), err)) return false;
            pd.peer = peer;
        }
        tier_->pump(tl, 2 + G);   // what this layer's head leaves the bus of the promotions queued before it
        if (!await_ids(err)) return false;
        int32_t uniq[kLookupK];
        int uidx[kLookupK], nu = 0;
        core::ExpertTier::Use* use = pd.use;
        for (int t = 0; t < G; ++t)
            for (int k = 0; k < K; ++k) {
                const int32_t e = h_ids_[t * K + k];
                if ((e < 0 && !(e == -1 && lossy_)) || e >= g_.n_expert) {   // -1: --route-tail-skip dropped it
                    err = "the router picked expert " + std::to_string(e);
                    return false;
                }
                int u = 0;
                while (u < nu && uniq[u] != e) ++u;
                if (u == nu) {
                    uniq[nu++] = e;
                    use[u] = core::ExpertTier::Use{};
                }
                uidx[t * K + k] = u;
                if (e == -1) continue;   // --route-tail-skip dropped it: no rows (a token may drop several)
                core::ExpertTier::Use& us = use[u];
                us.tok[us.nt] = t;
                us.row[us.nt] = t * K + k;
                ++us.nt;
            }
        core::ExpertTier::Lookup& lk = pd.lk;
        if (!tier_->lookup(tl, uniq, nu, lk, err)) return false;
        const int n_fetched = tier_->fetch(tl, lk);
        // the second card's plan first: its share waits for it, and its rows are the last of the two cards' to land
        if (peer) {
            int32_t kind[kLookupK] = {};
            for (int r = 0; r < G * K; ++r) kind[r] = !lk.peer[uidx[r]] ? 0 : lk.fetched[uidx[r]] ? 3 : 2;
            if (!peer->raise(tl, h_ids_, kind, err)) return false;
            pd.peer_raised = true;
        }
        bool on_peer = false;
        for (int i = 0; i < nu; ++i) {
            pd.cpu = pd.cpu || (!lk.dev[i] && !lk.peer[i] && !lk.reading[i] && !lk.dropped[i]);
            pd.reading = pd.reading || lk.reading[i];
            on_peer = on_peer || lk.peer[i];
        }
        if (pd.cpu || pd.reading || on_peer) pd.word = ++join_seq_;   // the weighted sum waits for the host's rows
        // a promotion only queues its copies (they go out at a later pump), so it is decided now and the copies start behind
        // this layer's sum; experts still being read from the file are decided once they landed
        if (pd.cpu && !pd.reading) tier_->promote(tl, lk);
        // then the CPU's share: it takes longest, and the pool's workers run it while the card's share launches
        if (pd.cpu) tier_->start_cpu_window(tl, h_x_, lk, use, h_rows_);
        // the card's share: the experts its slots hold, then the fetched ones once their copies landed; the plans travel in
        // kernel parameters, in stream order
        const auto make_plan = [&](WindowPlan& p, bool fetched, int& ne) {
            int ng = 0;
            ne = 0;
            for (int i = 0; i < nu; ++i) {
                if (!lk.dev[i] || lk.fetched[i] != fetched) continue;
                p.ptr[ng] = (unsigned long long) lk.dev[i];
                p.start[ng] = ne;
                for (int j = 0; j < use[i].nt; ++j) {
                    p.dst[ne] = use[i].row[j];
                    p.tok[ne] = use[i].tok[j];
                    ++ne;
                }
                ++ng;
            }
            p.start[ng] = ne;
            p.ng[0] = ng;
            return ng;
        };
        const auto launch = [&](void* dev, int ng, int ne) {
            WindowPlan* dp = (WindowPlan*) dev;
            kernels::native_expert_grouped_clamped(lay, g_.clamp_exp[(size_t) l], dp->ptr, dp->start, dp->ng, dp->dst, dp->tok, ng, ne,
                                                   ex_xq_, ex_scr_, ex_rows_, st);
        };
        WindowPut put;
        int ne = 0;
        const int nhits = make_plan(put.plan, false, ne);
        for (int r = 0; r < G * K; ++r)
            put.rows[r] = (unsigned long long) (lk.dev[uidx[r]]       ? ex_rows_ + (size_t) r * L
                                                : lk.dropped[uidx[r]] ? tier_->zero_row()
                                                                      : d_rows_ + (size_t) r * L);
        put_words(&put, ex_plan_, (int) ((offsetof(WindowPut, rows) + (size_t) G * K * 8) / 4), st);
        if (nhits) launch(ex_plan_, nhits, ne);
        if (pd.word) read_ahead(l, pd.word, st);
        if (n_fetched) {
            WindowPlan p2;
            const int ng = make_plan(p2, true, ne);
            tier_->await_fetch(st);
            put_words(&p2, ex_plan2_, (int) (sizeof(WindowPlan) / 4), st);
            launch(ex_plan2_, ng, ne);
        }
        // the second card's fetched experts run once their copies land, long after these launches
        if (peer && !peer->run_fetched(err)) return false;
        if (pd.word) wait_word_mark(d_words_ + 1, pd.word, ra_mark_, st);
        const float* const* rows = (const float* const*) ((uint8_t*) ex_plan_ + offsetof(WindowPut, rows));
        weighted_sum_window(rows, rw_ + (size_t) g0 * K, G, K, L, lat_out_ + (size_t) g0 * L, st);
        if (pd.cpu && !pd.reading) tier_->mark_gap(st);   // the promotions' copies start behind the sum
        // the rows land and the word is written on the joiner thread, while this thread queues the next layer's work: the
        // card goes on as soon as the CPU's and the second card's rows are in, not once the next launches are out
        if (use_joiner_ && pd.word && !pd.reading) {
            if (!joiner_.joinable()) joiner_ = std::thread([this] { joiner_loop(); });
            {
                std::lock_guard<std::mutex> g(jm_);
                jwork_ = true;
                jdone_.store(false, std::memory_order_relaxed);
            }
            jcv_.notify_one();
        }
        return true;
    }

    void joiner_loop() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(jm_);
                jcv_.wait(lk, [this] { return jwork_ || jstop_; });
                if (jstop_ && !jwork_) return;
                jwork_ = false;
            }
            std::string e;
            jok_ = join_body(e);
            jerr_ = e;
            jdone_.store(true, std::memory_order_release);
        }
    }

    // The CPU's and the second card's share of the last group placed: the pool's rows land, the experts read from the file
    // run, the second card's rows land, and the word its weighted sum waits for is written. On every way out, so the stream
    // never waits for good.
    bool join_moe(std::string& err) {
        if (!pend_.on) return true;
        if (use_joiner_ && pend_.word && !pend_.reading && joiner_.joinable()) {   // handed to the joiner: its result
            while (!jdone_.load(std::memory_order_acquire)) platform::cpu_relax();
            pend_.on = false;
            if (!jok_ && err.empty()) err = jerr_;
            return jok_;
        }
        const bool ok = join_body(err);
        pend_.on = false;
        return ok;
    }
    bool join_body(std::string& err) {
        Pending& pd = pend_;
        bool ok = true;
        if (pd.cpu) tier_->finish_cpu();
        if (pd.reading) {
            if (tier_->await_reads(err)) {
                tier_->start_cpu_window(pd.tl, h_x_, pd.lk, pd.use, h_rows_, /*reading=*/true);
                tier_->finish_cpu();
            } else {
                ok = false;
            }
            tier_->promote(pd.tl, pd.lk);
        }
        std::string pe;
        if (pd.peer && pd.peer_raised && !pd.peer->finish(nullptr, pe)) ok = false;
        if (pd.peer && !pd.peer_raised) pd.peer->raise(0, nullptr, nullptr, pe);
        if (!pe.empty() && err.empty()) err = pe;
        if (pd.word) {
            std::atomic_thread_fence(std::memory_order_release);   // the rows before the word
            *(volatile uint32_t*) (h_words_ + 1) = pd.word;
        }
        if (pd.reading) tier_->mark_gap(stream_);
        return ok;
    }

    // R rows at pos0..: the whole forward; the logits of the last `head_rows` rows go to logits_.
    bool run_rows(const int32_t* tokens, int R, int64_t pos0, int head_rows, std::string& err) {
        bool ok = run_rows_impl(tokens, R, pos0, head_rows, err);
        if (!join_moe(err)) ok = false;
        if (!ok) cudaStreamSynchronize(stream_);
        return ok;
    }
    bool run_rows_impl(const int32_t* tokens, int R, int64_t pos0, int head_rows, std::string& err) {
        if (!bsegs_.empty()) return run_body(tokens, R, pos0, head_rows, err);   // a batch: each session checked and placed its rows
        if (pos0 + R > m_.context()) {
            err = "position " + std::to_string(pos0 + R - 1) + " is past the context of " + std::to_string(m_.context());
            return false;
        }
        for (int t = 0; t < R; ++t)
            if ((tokens[t] < 0 && !emb_rows_) || tokens[t] >= g_.n_vocab) {   // [vision] -1: an embedding row
                err = "token " + std::to_string(tokens[t]) + " is out of the vocabulary";
                return false;
            }
        if (pos0 > (int64_t) hist_.size()) {
            err = "position " + std::to_string(pos0) + " does not follow the state's " + std::to_string(hist_.size());
            return false;
        }
        if (!ensure_positions(pos0 + R, err)) return false;   // before anything changes: a refusal leaves the state as it was
        hist_.resize((size_t) pos0);   // a position before the end drops the rows after it
        hist_.insert(hist_.end(), tokens, tokens + R);
        place_rope(pos0, R);   // [vision]
        return run_body(tokens, R, pos0, head_rows, err);
    }
    bool run_body(const int32_t* tokens, int R, int64_t pos0, int head_rows, std::string& err) {
        if (tier_) tier_->begin_token(std::min(R, kMaxWindow));
        if (lossy_) lossy_->publish_held(stream_);   // ahead of the rows' first router
        struct Drain {
            Session* s;
            ~Drain() { s->drain_prefetch(); }
        } drain{this};
        if (bsegs_.empty()) prefetch_engram(pos0, R);   // every Engram layer's rows are asked for before the first layer runs
        const int E = g_.n_embd;
        bk::HcPlainArgs hc;
        hc.n_embd = E;
        hc.rms_eps = g_.rms_eps;
        hc.hc_eps = g_.hc_eps;
        hc.magnitude = g_.hc_magnitude;
        for (int t = 0; t < R; ++t)
            if (emb_rows_) {   // [vision] an image row stands in for the token's embedding
                std::memcpy(h_e_ + (size_t) t * E, emb_rows_ + (size_t) t * E, (size_t) E * 4);
            } else if (!m_.embedding_row(tokens[t], h_e_ + (size_t) t * E)) {
                err = "no embedding row";
                return false;
            }
        put_f32(e_, h_e_, (int64_t) R * E);
        if (graphs_) {   // the window's first position, for the graphs' row spans (sliding-window layers)
            *pos_h_ = (int32_t) pos0;
            cudaMemcpyAsync(pos_d_, pos_h_, 4, cudaMemcpyHostToDevice, stream_);
        }
        fill_streams(e_, E, R, X_, stream_);
        if (!m_.diag_dump().empty()) dump("hc_init", X_, (int64_t) R * 4 * E);
        ring_join();
        ring_layer_[0] = ring_layer_[1] = -1;
        for (int l = 0; l < g_.n_layers; ++l) {
            ring_ahead(l, R);
            if (!engram(l, pos0, R, tokens, err)) return false;
            if (diag(l) && !g_.swa.empty() && l >= 1) dump("engram_out-" + std::to_string(l), X_, (int64_t) R * 4 * E);
            if (graphs_ && l >= g_.n_dense) {
                if (!(bsegs_.empty() ? layer_replayed(l, pos0, R, hc, err) : layer_batched(l, pos0, R, hc, err))) return false;
                continue;
            }
            bk::hc_mixes_plain(X_, F(l, "hc_attn_fn.weight"), F(l, "hc_attn_scale.weight"), F(l, "hc_attn_base.weight"), R, 8, hc, hcws_, coef_, stream_);
            bk::hc_pre_rms_norm(X_, coef_, bk::kHcCoef, F(l, "attn_norm.weight"), R, E, g_.rms_eps, cur_, xn_, stream_);
            if (diag(l)) dump("attn_norm-" + std::to_string(l), xn_, (int64_t) R * E);
            const double ta = now_ms();
            if (!attention(l, pos0, R, err)) return false;
            t_att_ms_ += now_ms() - ta;
            bk::hc_post(attn_, X_, coef_, bk::kHcCoef, R, E, Xn_, stream_);
            std::swap(X_, Xn_);
            if (diag(l)) dump("hc_attn_out-" + std::to_string(l), X_, (int64_t) R * 4 * E);
            bk::hc_mixes_plain(X_, F(l, "hc_ffn_fn.weight"), F(l, "hc_ffn_scale.weight"), F(l, "hc_ffn_base.weight"), R, 8, hc, hcws_, coef_, stream_);
            bk::hc_pre_rms_norm(X_, coef_, bk::kHcCoef, F(l, "ffn_norm.weight"), R, E, g_.rms_eps, cur_, xn_, stream_);
            if (diag(l)) dump("ffn_norm-" + std::to_string(l), xn_, (int64_t) R * E);
            const double tf = now_ms();
            if (!ffn(l, R, err)) return false;
            t_ffn_ms_ += now_ms() - tf;
            bk::rms_norm_rows(mlp_, F(l, "post_ffw_norm.weight"), mlp2_, R, E, g_.rms_eps, stream_);
            if (diag(l)) dump("ffn_out-" + std::to_string(l), mlp2_, (int64_t) R * E);
            bk::hc_post(mlp2_, X_, coef_, bk::kHcCoef, R, E, Xn_, stream_);
            std::swap(X_, Xn_);
            if (!m_.diag_dump().empty()) dump("l_out-" + std::to_string(l), X_, (int64_t) R * 4 * E);
        }
        if (!join_moe(err)) return false;
        if (tier_) tier_->pump(-1, core::ExpertTier::kAllPieces);
        if (head_rows > 0) {
            const float* x = X_ + (size_t) (R - head_rows) * 4 * E;
            bk::hc_mixes_plain(x, (const float*) m_.tensor("output_hc_fn.weight")->ptr, (const float*) m_.tensor("output_hc_scale.weight")->ptr,
                               (const float*) m_.tensor("output_hc_base.weight")->ptr, head_rows, 4, hc, hcws_, coef_, stream_);
            bk::hc_pre(x, coef_, bk::hc_plain_coef_stride(4), head_rows, E, cur_, stream_);
            bk::rms_norm_rows(cur_, (const float*) m_.tensor("output_norm.weight")->ptr, fin_, head_rows, E, g_.rms_eps, stream_);
            if (bsegs_.empty()) dfly_.hidden(R - head_rows, head_rows, fin_, stream_);   // DFLY HOOK
            else   // a batch's head runs every row: each conversation's drafter its segment's head rows
                for (const Seg& sg : bsegs_)
                    sg.s->dfly_.hidden(sg.rows - sg.head, sg.head, fin_ + (size_t) (sg.row0 + sg.rows - sg.head) * E, stream_);
            const DevTensor* head = m_.tensor("output.weight");
            gemm(head, fin_, logits_, E, g_.n_vocab, head_rows);
            if (!m_.diag_dump().empty()) {
                dump("result_norm", fin_, (int64_t) head_rows * E);
                dump("result_output", logits_, (int64_t) head_rows * g_.n_vocab);
            }
            last_rows_ = head_rows;
        }
        const cudaError_t ce = cudaGetLastError();
        if (ce != cudaSuccess) {
            err = std::string("CUDA: ") + cudaGetErrorString(ce);
            return false;
        }
        return true;
    }

    // Backs the sparse layers' caches for `positions` positions, in steps of kGrowPositions. When the card is full (a long
    // generation past the prompt, the expert tier holding the rest), the tier gives back its spare slots and the mapping is
    // tried again, then lends all its slots and takes back what is left once the caches have grown (as DeepSeek's caches do).
    bool grow_failed_ = false;   // the last ensure_positions found no room (a refusal: nothing was run past `backed_`)
    // the exact remaining device bytes the sparse layers' caches will map to reach `target` positions
    // (ElasticArray::ensure grows each array in steps of 4 granules = 8 MiB on 2 MiB VMM pages)
    uint64_t grow_bytes(int64_t target) const {
        constexpr uint64_t kStep = 8ull << 20;
        auto need = [&](const std::unique_ptr<core::ElasticArray>& a, uint64_t bytes) -> uint64_t {
            if (!a || bytes <= a->backed()) return 0;
            const uint64_t want = std::min(a->capacity(), (bytes + kStep - 1) / kStep * kStep);
            return want > a->backed() ? want - a->backed() : 0;
        };
        uint64_t total = 0;
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) continue;
            total += need(ea_[(size_t) l], (uint64_t) target * sparse_row_bytes());
            total += need(pea_[(size_t) l], (uint64_t) (target / g_.idx_pool + 1) * g_.idx_dim * 2);
        }
        return total;
    }
    bool ensure_positions(int64_t positions, std::string& err) {
        grow_failed_ = false;
        if (positions <= backed_) return true;
        const int64_t target = std::min<int64_t>(m_.context(), (positions + kGrowPositions - 1) / kGrowPositions * kGrowPositions);
        const uint64_t keep = tier_ ? tier_->free_floor() : 256ull << 20;
        bool lent = false, lent_all = false;
        for (int attempt = 0; attempt < 4; ++attempt) {
            std::string e;
            bool ok = true;
            for (int l = 0; ok && l < g_.n_layers; ++l) {
                if (g_.swa[(size_t) l]) continue;
                ok = ea_[(size_t) l]->ensure((uint64_t) target * sparse_row_bytes(), keep, e) &&
                     pea_[(size_t) l]->ensure((uint64_t) (target / g_.idx_pool + 1) * g_.idx_dim * 2, keep, e);
            }
            if (ok) {
                backed_ = target;
                if (lent && !big_.rows) {   // lent here, outside a prompt (which takes its slots back at its end)
                    std::string r;
                    if (!tier_->reclaim(r)) basalt::log::warn("basalt hy4flash", "the latent caches took expert slots at %lld positions: %s", (long long) target, r.c_str());
                }
                return true;
            }
            if (!tier_ || attempt == 3 || lent_all) {
                if (lent && !big_.rows) {
                    std::string r;
                    tier_->reclaim(r);
                }
                err = "the latent caches cannot grow to " + std::to_string(target) + " positions: " + e + "; a smaller --max-context leaves more for the expert tier";
                grow_failed_ = true;
                return false;
            }
            cudaStreamSynchronize(stream_);
            if (attempt == 0) tier_->shrink();
            else if (attempt == 1 && tier_->partial_lend() && tier_->lend(grow_bytes(target))) lent = true;
            else {
                tier_->lend_all();
                lent = lent_all = true;
            }
        }
        return false;
    }

    // [vision] The rotary positions of the R cells about to be filled (text, or the rows of the picture block being fed), kept per
    // cell on the host and, once a picture is in the session, on the device where the kernels read them (MlaArgs::cell_rope).
    void sync_rope(int64_t pos) {
        rope_.rewind(pos);
        while (rope_.size() < pos) rope_.push_text();   // cells the session never placed (a restored history) count as text
    }
    // The rotary position a text token at cell `pos` takes, the track left as it is (the drafter's block, before its window).
    int64_t text_rope(int64_t pos) const {
        const int64_t have = std::min<int64_t>(pos, rope_.size());
        return (have > 0 ? rope_.at(have - 1).last() + 1 : 0) + (pos - have);
    }
    void place_rope(int64_t pos0, int R) {
        sync_rope(pos0);
        const bool was_on = rope_on_;
        for (int t = 0; t < R; ++t) {
            if (emb_rows_) { rope_.push_row(img_start_, img_nx_, img_j_ + t); rope_on_ = true; }
            else rope_.push_text();
        }
        if (!rope_on_) return;
        std::vector<int32_t> tab;
        const int64_t from = was_on ? pos0 : 0;
        for (int64_t c = from; c < pos0 + R; ++c) tab.insert(tab.end(), {rope_.at(c).t, rope_.at(c).h, rope_.at(c).w});
        cudaMemcpyAsync(rope_dev_ + from * 3, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice, stream_);
        cudaStreamSynchronize(stream_);   // the host table is a temporary
    }

    // diagnostic: a layer whose inner tensors are dumped (the first three)
    bool diag(int l) const { return !m_.diag_dump().empty() && l < 3; }

    // diagnostic: the stream drained. A window's weighted sum waits on the device for the word join_moe writes, so the MoE is
    // joined first: a wait for the stream before it would never end.
    void diag_sync() {
        std::string e;
        if (!join_moe(e)) basalt::log::warn("basalt hy4flash", "--diag-dump: %s", e.c_str());
        cudaStreamSynchronize(stream_);
    }

    // diagnostic: a tensor of this call appended to <dir>/<name>.f32
    void dump(const std::string& name, const float* d, int64_t n) {
        std::vector<float> h((size_t) n);
        diag_sync();
        cudaMemcpyAsync(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        const std::string path = m_.diag_dump() + "/" + name + ".f32";
        std::FILE* f = std::fopen(path.c_str(), "ab");
        if (!f) {
            basalt::log::warn("basalt hy4flash", "--diag-dump: cannot write %s", path.c_str());
            return;
        }
        std::fwrite(h.data(), 4, h.size(), f);
        std::fclose(f);
    }

    bool device_bias(float*& dev, const float* host, std::string& err) {
        const size_t bytes = (size_t) g_.n_vocab * sizeof(float);
        if (dev == nullptr) {
            void* v = nullptr;
            if (cudaMalloc(&v, bytes) != cudaSuccess) { cudaGetLastError(); err = "out of device memory for a logit bias"; return false; }
            bufs_.push_back(v);
            dev = (float*) v;
        }
        if (cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, stream_) != cudaSuccess) { err = "the logit bias upload failed"; return false; }
        return true;
    }

    // The picks of the last `rows` rows of logits_ (row t of the window is draw counter + t0 + t).
    bool pick(const model::DecodeRequest& r, int t0, int rows, std::string& err) {
        if (!draw(r, t0, rows, stream_, err)) return false;
        picks_.assign((size_t) rows, 0);
        if (cudaMemcpyAsync(picks_.data(), pick_, (size_t) rows * 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess || cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "the pick failed";
            return false;
        }
        return true;
    }
    // The draws of those rows into pick_, queued on `st`.
    bool draw(const model::DecodeRequest& r, int t0, int rows, cudaStream_t st, std::string& err) {
        bk::SamplerParams sp;
        if (r.sampling) sp = *r.sampling;
        else sp.greedy = true;
        if (!r.sampling) sp.temperature = 0.f;
        sp.counter += (uint64_t) t0;
        sp.logit_bias = bias_on_ ? d_bias_ : nullptr;
        history_len_ = r.history != nullptr ? r.history_len : 0;
        if (history_len_ > 0) {   // the penalties' history of each row of the window
            const size_t n = (size_t) kMaxWindow * history_len_;
            if (d_hist_len_ < n) {
                if (d_hist_) cudaFree(d_hist_);
                if (cudaMalloc((void**) &d_hist_, n * 4) != cudaSuccess) { cudaGetLastError(); err = "out of device memory for the sampling history"; return false; }
                d_hist_len_ = n;
            }
            cudaMemcpyAsync(d_hist_, r.history, (size_t) rows * history_len_ * 4, cudaMemcpyHostToDevice, st);
        }
        sampling_ = sp;
        bk::sample_tokens(logits_, rows, (int) g_.n_vocab, history_len_ > 0 ? d_hist_ : nullptr, history_len_, sp, pick_, st);
        return cudaGetLastError() == cudaSuccess || (err = "the draw failed", false);
    }

    Hy4Model& m_;
    const Geometry& g_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t ev_ = nullptr, eng_ev_ = nullptr;
    std::vector<void*> bufs_, pinned_;
    std::vector<void*>* alloc_to_ = &bufs_;   // where alloc keeps what it gives: the session's, or the shared scratch's while that is built
    std::vector<void*>* pin_to_ = &pinned_;
    std::shared_ptr<Scratch> sc_;
    float *X_ = nullptr, *Xn_ = nullptr, *coef_ = nullptr, *hcws_ = nullptr;
    float *e_ = nullptr, *cur_ = nullptr, *xn_ = nullptr, *attn_ = nullptr, *attn_raw_ = nullptr, *mlp_ = nullptr, *mlp2_ = nullptr, *fin_ = nullptr;
    float *qa_ = nullptr, *qan_ = nullptr, *q_ = nullptr, *kvraw_ = nullptr, *qabs_ = nullptr, *qcur_ = nullptr, *olat_ = nullptr, *vout_ = nullptr, *gate_ = nullptr;
    float *ik_ = nullptr, *ikn_ = nullptr, *ivt_ = nullptr, *igt_ = nullptr, *iq_ = nullptr, *iw_ = nullptr, *iscore_ = nullptr;
    int32_t *iidx_meta_ = nullptr, *icells_ = nullptr, *meta_d_ = nullptr, *meta_h_ = nullptr;
    int32_t* pos_d_ = nullptr;   // a decode window's first position on the card (swa_meta reads it)
    int32_t* pos_h_ = nullptr;   // its pinned host copy (a window ends synchronized, so the last copy has read it)
    void* attw_ = nullptr;   // the chunked attention's workspace
    int cells_stride_ = 0, rows_ = 0, arows_ = 0, steps_ = 0, max_heads_ = 0;
    static constexpr size_t kStageInts = (size_t) 4 << 20;   // the mapped staging of a prompt's small inputs
    int32_t *stage_h_ = nullptr, *stage_d_ = nullptr, *bounds_tab_ = nullptr;
    size_t stage_at_ = 0;
    Bufs big_;   // a window's row buffers, and a prompt chunk's while one runs
    float *gu_ = nullptr, *h_ = nullptr, *rlog_ = nullptr, *rw_ = nullptr;
    int32_t* ids_ = nullptr;
    float *xlat_ = nullptr, *lat_out_ = nullptr, *moe_out_ = nullptr, *sh_ = nullptr, *logits_ = nullptr, *lp_ = nullptr, *erows_ = nullptr, *ekv_ = nullptr, *evalid_ = nullptr;
    int32_t* pick_ = nullptr;
    int32_t* pick_h_ = nullptr;   // pinned: a batch draw's picks, read after the batch's sync
    struct Stage {
        uint8_t* blobs = nullptr;
        int32_t *tok = nullptr, *bounds = nullptr;
        cudaEvent_t copied = nullptr, done = nullptr;
    } stage_[2];
    cudaStream_t copy_ = nullptr;
    size_t stride_ = 0;
    uint64_t launch_ = 0;
    float* pair_rows_ = nullptr;
    int32_t *pair_map_ = nullptr, *pair_ident_ = nullptr;
    void *xq1_ = nullptr, *xq2_ = nullptr;
    basalt::prefill::mmq::Context mmq_;
    std::set<std::pair<int, int>> loaded_;   // (type, rows) the MMQ tiles are loaded for
    void *mmq_xq_ = nullptr;
    int32_t *mmq_ident_ = nullptr, *heads_ids_ = nullptr, *heads_bounds_ = nullptr;
    void* xq_n_ = nullptr;   // quant_n
    void *q8_ = nullptr, *ex_xq_ = nullptr, *ex_scr_ = nullptr, *ex_plan_ = nullptr, *ex_plan2_ = nullptr;
    float* ex_rows_ = nullptr;
    float *h_x_ = nullptr, *d_x_ = nullptr, *h_rows_ = nullptr, *d_rows_ = nullptr, *h_e_ = nullptr, *h_erows_ = nullptr, *h_evalid_ = nullptr;
    const float** rowptr_d_ = nullptr;
    const float** h_rowptr_ = nullptr;
    WindowPlan* h_plan_ = nullptr;
    float* h_rw_ = nullptr;
    int32_t *h_ids_ = nullptr, *d_ids_ = nullptr;
    uint32_t *h_words_ = nullptr, *d_words_ = nullptr;   // mapped: [0] the ids' sequence word, [1] the word a weighted sum waits for
    uint32_t seq_want_ = 0, join_seq_ = 0;
    // the L2 read-ahead during a MoE layer's wait (read_ahead): its stream, the event it starts behind, the mark it stops at
    // (device: the last word the weighted sum's wait saw) and each layer's ranges
    cudaStream_t ra_st_ = nullptr;
    cudaEvent_t ra_ev_ = nullptr;
    uint32_t* ra_mark_ = nullptr;
    std::vector<L2Ranges> ra_;
    // the joiner thread (moe_group hands it a group's CPU and second-card share; join_moe waits for its result)
    bool use_joiner_ = true;
    bool use_graphs_ = true;
    core::ExpertTier* lossy_ = nullptr;   ///< in a decode window with --route-tail-skip / --route-prior on: the tier
    std::thread joiner_;
    std::mutex jm_;
    std::condition_variable jcv_;
    bool jwork_ = false, jstop_ = false, jok_ = true;
    std::atomic<bool> jdone_{true};
    std::string jerr_;
    // the group whose CPU and second-card share is still running (join_moe ends it)
    struct Pending {
        bool on = false, cpu = false, reading = false, peer_raised = false;
        int tl = 0;
        uint32_t word = 0;
        core::PeerExperts* peer = nullptr;
        core::ExpertTier::Lookup lk;
        core::ExpertTier::Use use[kLookupK];
    } pend_;
    std::shared_ptr<core::ExpertTier> tier_;
    std::vector<float> lookup_host_;
    std::vector<bool> lookup_loaded_;
    std::vector<uint8_t*> cache_;   // a sparse layer's rows in the --kv format, a window layer's fp16
    std::vector<float*> ival_, igate_;
    std::vector<uint16_t*> ipool_;   // fp16 pooled keys
    DflyHook dfly_;   // DFLY HOOK
    std::vector<std::unique_ptr<core::ElasticArray>> ea_, pea_;
    int64_t backed_ = 0;   // positions the sparse layers' caches are backed for
    std::vector<int32_t> hist_, picks_;
    bool bias_on_ = false;
    float *d_bias_ = nullptr, *d_mask_ = nullptr;
    int* d_hist_ = nullptr;
    size_t d_hist_len_ = 0;
    int history_len_ = 0;
    bk::SamplerParams sampling_;
    // [vision]
    bool vis_shrunk_ = false;            // the image encoder took the slots beside the base arena
    bool vis_lent_ = false;              // ... and every slot the tier lends
    RopeTrack rope_;
    int32_t* rope_dev_ = nullptr;
    bool rope_on_ = false;               // a picture has been fed: the kernels read the rotary table
    const float* emb_rows_ = nullptr;    // the embeddings of the rows being run, in place of the tokens'
    int32_t img_start_ = 0;              // the first position of the block being fed
    int img_nx_ = 1, img_j_ = 0;         // its columns (newline included) and the row of its first run row
    int last_rows_ = 0;
    // prompt timing (host wall time at the sync points): the work queued before a layer's MoE, the MoE itself, the Engram's reads
    double t_pre_ms_ = 0, t_moe_ms_ = 0, t_eng_ms_ = 0, t_build_ms_ = 0, t_issue_ms_ = 0, t_att_ms_ = 0, t_ffn_ms_ = 0;
    double t_eng_read_ms_ = 0, t_eng_norm_ms_ = 0;
    std::atomic<double> t_ring_ms_{0};
    static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(Hy4Model& model, std::string& err) {
    auto s = std::make_unique<Session>(model);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::hy4flash
