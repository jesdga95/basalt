// src/models/ernie55/session.cpp - ERNIE-5.5-Flash inference session:
//   - 43-layer hybrid HCA (37x) + MLA-DSA (6x) + mHC (4 streams) + Latent MXFP4 MoE (42x 512 top-10)
//   - Decode windows (1..8 rows) via core::ExpertTier (GPU 0 LRU + GPU 1 Peer + CPU pool) + DSpark drafter
//   - Prompt prefill via Basalt's Tensor-Core prefill::mmq (MXFP4 + SiTU-GLU 4/25) + copy-stream double-buffered
//     staging + dual-GPU prefill::PeerShare + temporary tier->lend()/reclaim() (keeps Decode VRAM cache warm)
#include "models/ernie55/session.hpp"

#include "basalt/core/checkpoint_regions.hpp"
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/device_copy.hpp"
#include "basalt/core/expert_tier.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/log.hpp"
#include "basalt/models/ernie55/cache_layout.hpp"
#include "basalt/models/ernie55/kernels.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/prefill/peer_share.hpp"
#include "basalt/program/expert_spill.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <set>
#include <thread>
#include <utility>
#include <vector>

namespace basalt::models::ernie55 {
namespace {

constexpr int kMaxDecodeWindow = 8;
constexpr int kAttRows = 128;          ///< attention & dense-layer slice width (bounds attention scratch VRAM)
constexpr int kMaxPromptChunk = 16384; ///< max prompt MoE chunk width (amortizes 512-expert PCIe streaming)
constexpr int kHomePromptChunk = 8192; ///< prompt MMQ chunk guaranteed beside permanent VRAM homes (`homes_.main > 0`)
constexpr int kRingRows = 8192;        ///< a prompt chunk this wide routes to ~all 512 experts, justifying ring preload
constexpr int kRingMinExperts = 64;    ///< the smallest prompt expert ring worth opening (experts per layer slot)
constexpr int kHomeRingExperts = 128;  ///< ring experts per layer slot reserved beside permanent VRAM homes
constexpr int kStageExperts = 16;      ///< MXFP4 experts per GPU-0 double-buffered staging slot
constexpr int kIdxKeyTile = 4096;      ///< BF16 Tensor-Core Lightning Indexer key tile width
constexpr uint64_t kL2Prefetch = 56ull << 20;   ///< next-layer weight budget prefetched into L2 while waiting on host
constexpr int kLaunchRows = 2048;      ///< max token-expert rows per MMQ launch
constexpr int kMmqTile = 128;          ///< MMQ tile alignment padding for identity row tables
constexpr int kCpuShortLimit = 256;    ///< prompts <= this may offload 1..4-row RAM experts to CPU pool
constexpr int kPeerPrepareRows = 1024; ///< long chunks pre-stage next layer's GPU-1 resident experts during attention
constexpr int kCpuMaxRowPerExp = 4;
constexpr int kCpuMaxRows = 1024;
constexpr int kHcaWinCap = 256;
constexpr int kHcaFp8RowBytes = layout::kHcaFp8RowBytes;
constexpr int kDsaFp8RowBytes = layout::kDsaFp8RowBytes;
constexpr int kIdxFp4RowBytes = layout::kIdxFp4RowBytes;
constexpr int kTypeQ8_0 = 8;
constexpr size_t kCublasWorkspace = 4ull << 20;
constexpr size_t kMmqHeadroom = 160ull << 20;   ///< free VRAM a prompt chunk leaves for MMQ's own buffers
constexpr int kTypeBF16 = 30;
constexpr int kTypeMXFP4 = 39;

double now_ms() {
    using C = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(C::now().time_since_epoch()).count();
}

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("ernie55: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

template <typename T>
bool dalloc(T*& p, size_t count, const char* what, std::string& err) {
    p = nullptr;
    if (count == 0) return true;
    return ck(cudaMalloc(&p, count * sizeof(T)), what, err);
}

template <typename T>
void dfree(T*& p) {
    if (p) { cudaFree(p); p = nullptr; }
}

struct HcaLayerState {
    uint8_t* win_cache = nullptr;
    uint8_t* comp_cache = nullptr;
    float* state_kv = nullptr;
    float* state_score = nullptr;
    size_t win_bytes = 0, comp_bytes = 0, state_bytes = 0;
};

struct DsaLayerState {
    uint8_t* kv_cache = nullptr;
    uint8_t* idx_k_cache = nullptr;
    size_t kv_bytes = 0, idx_bytes = 0;
};

struct MmqStage {
    uint8_t* blobs = nullptr;
    int32_t* tok = nullptr;
    int32_t* bounds = nullptr;
    cudaEvent_t copied = nullptr;
    cudaEvent_t done = nullptr;
};

// Temporary prompt buffers lent from the VRAM expert tier for the duration of a prompt and returned in end_prompt().
struct PromptBufs {
    bool ready = false;
    int rows = 0;
    bool own_row_bufs = false;   ///< true when rows > kAttRows

    // Row buffers (point to permanent kAttRows buffers when rows <= kAttRows, else owned here)
    int32_t *pos = nullptr, *exp_ids = nullptr;
    float *res_a = nullptr, *res_b = nullptr, *x_in = nullptr, *x_norm = nullptr, *sub_out = nullptr;
    float *mhc_ws = nullptr, *mhc_coef = nullptr;
    float *ff_g = nullptr, *ff_u = nullptr, *ff_h = nullptr;
    float *rlogits = nullptr, *exp_w = nullptr;
    float *x_lat = nullptr, *lat_sum = nullptr, *lat_norm = nullptr, *routed_out = nullptr;
    float* router_w = nullptr;   ///< a layer's router widened to F32 for a chunk's SGEMM (rows > kMaxDecodeWindow)

    // Prompt MMQ MoE buffers (always owned here)
    float* pair_rows = nullptr;
    int32_t *pair_map = nullptr, *pair_ident = nullptr;
    int32_t *h_pair_map = nullptr, *h_ids = nullptr;
    int32_t *h_launch_tok = nullptr, *h_launch_bounds = nullptr;
    size_t h_launch_bounds_cap = 0;   ///< entries h_launch_bounds holds (both launch paths write into it)
    float *gu = nullptr, *h = nullptr;
    void *xq1 = nullptr, *xq2 = nullptr, *xq_lat = nullptr;
    // Prompt Attention acceleration scratch (MLA-DSA FP16 W_UK/W_UV + BF16 TC Indexer + HCA 8-group batched oa MMQ)
    uint16_t *kvb_f16 = nullptr, *q_nope_f16 = nullptr, *o_lat_f16 = nullptr;
    uint16_t *idx_q_bf16 = nullptr, *idx_k_bf16 = nullptr;
    float* idx_dots = nullptr;
    int32_t *oa_bounds = nullptr, *oa_ids = nullptr;
    int oa_cached_A = -1;
    MmqStage stage[2]{};
    cudaEvent_t ev_route = nullptr, ev_ids = nullptr, ev_xlat = nullptr, ev_map = nullptr;

    std::vector<void*> dev_owned;
    std::vector<void*> host_owned;

    void release() {
        for (int i = 0; i < 2; ++i) {
            if (stage[i].copied) { cudaEventDestroy(stage[i].copied); stage[i].copied = nullptr; }
            if (stage[i].done) { cudaEventDestroy(stage[i].done); stage[i].done = nullptr; }
        }
        auto destroy_ev = [](cudaEvent_t& e) { if (e) { cudaEventDestroy(e); e = nullptr; } };
        destroy_ev(ev_route);
        destroy_ev(ev_ids);
        destroy_ev(ev_xlat);
        destroy_ev(ev_map);
        for (void* p : dev_owned) cudaFree(p);
        for (void* p : host_owned) cudaFreeHost(p);
        dev_owned.clear();
        host_owned.clear();
        ready = false;
        rows = 0;
        own_row_bufs = false;
        h_launch_bounds_cap = 0;
        router_w = nullptr;
        idx_q_bf16 = nullptr;
        idx_k_bf16 = nullptr;
        idx_dots = nullptr;
        oa_cached_A = -1;
    }
};

class ErnieSession final : public model::ModelSession {
public:
    explicit ErnieSession(ErnieModel& m) : m_(m), g_(m.geo()) {}

    ~ErnieSession() override { release(); }

    bool init(std::string& err) {
        cudaSetDevice(m_.device());
        if (!ck(cudaStreamCreate(&stream_), "stream", err) ||
            !ck(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), "copy_stream", err) ||
            !ck(cudaStreamCreateWithFlags(&ring_stream_, cudaStreamNonBlocking), "ring stream", err) ||
            !ck(cudaStreamCreateWithFlags(&pf_stream_, cudaStreamNonBlocking), "prefetch stream", err) ||
            !ck(cudaEventCreateWithFlags(&pf_fork_, cudaEventDisableTiming), "prefetch event", err) ||
            !ck(cudaEventCreateWithFlags(&pf_join_, cudaEventDisableTiming), "prefetch event", err) ||
            !ck(cudaStreamCreateWithFlags(&hint_stream_, cudaStreamNonBlocking), "hint stream", err) ||
            !ck(cudaEventCreateWithFlags(&hint_fork_, cudaEventDisableTiming), "hint event", err) ||
            !ck(cudaEventCreateWithFlags(&hint_join_, cudaEventDisableTiming), "hint event", err))
            return false;
        if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
            err = "ernie55: cublasCreate failed";
            return false;
        }
        cublasSetStream(cublas_, stream_);
        // a workspace of its own, so the router's GEMM can be captured into the decode windows' graphs
        if (!ck(cudaMalloc(&cublas_ws_, kCublasWorkspace), "cublas workspace", err)) return false;
        cublasSetWorkspace(cublas_, cublas_ws_, kCublasWorkspace);

        stride_ = m_.expert_stride();
        if (kernels::native_q8_1_bytes(g_.hca_heads / g_.hca_o_groups * g_.hca_head_dim, 8) !=
            8 * kernels::native_q8_1_bytes(g_.hca_heads / g_.hca_o_groups * g_.hca_head_dim, 1) || !kernels::native_mmvq_multi_exact()) {
            err = "ernie55: the q8_1 layout of several columns is not the single-column one repeated";
            return false;
        }
        dec_layout_ = kernels::native_expert_layout(kTypeMXFP4, kTypeMXFP4, g_.n_latent, g_.n_ff_exp, 0.0f);
        dec_layout_.gate_softcap = g_.gate_softcap;
        dec_layout_.up_softcap = g_.up_softcap;
        hca_rope_ = rope_table_yarn(g_.comp_rope_theta, g_.yarn_factor, g_.yarn_beta_fast, g_.yarn_beta_slow, g_.yarn_orig_ctx);
        dsa_rope_ = rope_table_plain(g_.rope_theta);

        // Allocate KV & compressor state caches for this conversation
        const int64_t ctx = m_.context();
        const size_t hca_row = hca_row_bytes();
        const size_t dsa_row = dsa_row_bytes();
        const size_t idx_row = idx_row_bytes();

        hca_.resize((size_t) g_.n_layers);
        dsa_.resize((size_t) g_.n_layers);
        win_cap_ = std::max(g_.window, kHcaWinCap);
        kv_bytes_ = 0;
        for (int l = 0; l < g_.n_layers; ++l) {
            if (!g_.is_dsa(l)) {
                const int ratio = std::max(1, g_.compress_ratios[(size_t) l]);
                const size_t n_comp = (size_t) (ctx / ratio + 2);
                auto& s = hca_[(size_t) l];
                s.win_bytes = (size_t) win_cap_ * hca_row;
                s.comp_bytes = n_comp * hca_row;
                s.state_bytes = (size_t) ratio * (size_t) g_.hca_head_dim * sizeof(float);
                if (!ck(cudaMalloc(&s.win_cache, s.win_bytes), "hca win_cache", err) ||
                    !ck(cudaMalloc(&s.comp_cache, s.comp_bytes), "hca comp_cache", err) ||
                    !ck(cudaMalloc(&s.state_kv, s.state_bytes), "hca state_kv", err) ||
                    !ck(cudaMalloc(&s.state_score, s.state_bytes), "hca state_score", err))
                    return false;
                cudaMemsetAsync(s.win_cache, 0, s.win_bytes, stream_);
                cudaMemsetAsync(s.comp_cache, 0, s.comp_bytes, stream_);
                cudaMemsetAsync(s.state_kv, 0, s.state_bytes, stream_);
                cudaMemsetAsync(s.state_score, 0, s.state_bytes, stream_);
                kv_bytes_ += (int64_t) (s.win_bytes + s.comp_bytes + 2 * s.state_bytes);
            } else {
                auto& s = dsa_[(size_t) l];
                s.kv_bytes = (size_t) ctx * dsa_row;
                s.idx_bytes = (size_t) ctx * idx_row;
                if (!ck(cudaMalloc(&s.kv_cache, s.kv_bytes), "dsa kv_cache", err) ||
                    !ck(cudaMalloc(&s.idx_k_cache, s.idx_bytes), "dsa idx_k_cache", err))
                    return false;
                cudaMemsetAsync(s.kv_cache, 0, s.kv_bytes, stream_);
                cudaMemsetAsync(s.idx_k_cache, 0, s.idx_bytes, stream_);
                kv_bytes_ += (int64_t) (s.kv_bytes + s.idx_bytes);
            }
        }

        // Permanent forward scratch sized for kAttRows (128) rows — attention & dense layer 0 always slice to <= 128
        const int maxT = kAttRows;
        const int H = g_.n_embd;
        const int q_dim_max = std::max(g_.hca_heads * g_.hca_head_dim, g_.mla_heads * (g_.mla_qk_nope + g_.mla_qk_rope));
        const int o_dim_max = std::max({g_.hca_heads * g_.hca_head_dim, g_.mla_heads * g_.mla_v_dim,
                                        g_.mla_heads * g_.mla_kv_lora});
        const int gate_dim_max = std::max(g_.hca_o_groups * g_.hca_o_lora, g_.mla_heads * g_.mla_v_dim);
        const int ff_max = std::max(g_.n_ff, g_.n_ff_exp);
        const int q8_cols_max = std::max({H, gate_dim_max, ff_max, g_.hca_q_lora, g_.n_latent});
        const int max_entries = maxT * g_.n_used;

        if (!dalloc(d_pos_, maxT, "d_pos", err) ||
            !dalloc(d_res_a_, (size_t) maxT * kHcMult * H, "d_res_a", err) ||
            !dalloc(d_res_b_, (size_t) maxT * kHcMult * H, "d_res_b", err) ||
            !dalloc(d_x_in_, (size_t) maxT * std::max(H, (g_.hca_heads / g_.hca_o_groups) * g_.hca_head_dim), "d_x_in", err) ||
            !dalloc(d_x_norm_, (size_t) maxT * H, "d_x_norm", err) ||
            !dalloc(d_sub_out_, (size_t) maxT * H, "d_sub_out", err) ||
            !dalloc(d_mhc_ws_, mhc_workspace_bytes(maxT) / sizeof(float), "d_mhc_ws", err) ||
            !dalloc(d_mhc_coef_, (size_t) maxT * kHcCoef, "d_mhc_coef", err) ||
            !dalloc(d_mhc_cnt_, kMaxDecodeWindow, "d_mhc_cnt", err) ||
            !dalloc(d_qr_, (size_t) maxT * std::max(g_.hca_q_lora, g_.mla_q_lora), "d_qr", err) ||
            !dalloc(d_q_, (size_t) maxT * q_dim_max, "d_q", err) ||
            !dalloc(d_kv_, (size_t) maxT * std::max(g_.hca_head_dim, g_.mla_kv_lora + g_.mla_qk_rope), "d_kv", err) ||
            !dalloc(d_ckv_, (size_t) maxT * g_.hca_head_dim, "d_ckv", err) ||
            !dalloc(d_cscore_, (size_t) maxT * g_.hca_head_dim, "d_cscore", err) ||
            !dalloc(d_attn_o_, (size_t) maxT * o_dim_max, "d_attn_o", err) ||
            !dalloc(d_attn_o2_, (size_t) maxT * o_dim_max, "d_attn_o2", err) ||
            !dalloc(d_gate_, (size_t) maxT * gate_dim_max, "d_gate", err) ||
            !dalloc(d_z_, (size_t) maxT * gate_dim_max, "d_z", err) ||
            !dalloc(d_iq_, (size_t) maxT * g_.idx_heads * g_.idx_dim, "d_iq", err) ||
            !dalloc(d_ik_, (size_t) maxT * g_.idx_dim, "d_ik", err) ||
            !dalloc(d_iw_, (size_t) maxT * g_.idx_heads, "d_iw", err) ||
            !dalloc(d_sel_idx_, (size_t) maxT * (g_.idx_top_k + g_.window), "d_sel_idx", err) ||
            !dalloc(d_idx_scores_, (size_t) kMaxDecodeWindow * (size_t) ctx, "d_idx_scores", err) ||
            !dalloc(d_ff_g_, (size_t) maxT * ff_max, "d_ff_g", err) ||
            !dalloc(d_ff_u_, (size_t) maxT * ff_max, "d_ff_u", err) ||
            !dalloc(d_ff_h_, (size_t) maxT * ff_max, "d_ff_h", err) ||
            !dalloc(d_rlogits_, (size_t) maxT * 2 * g_.n_expert, "d_rlogits", err) ||
            !dalloc(d_exp_ids_, (size_t) max_entries, "d_exp_ids", err) ||
            !dalloc(d_exp_w_, (size_t) max_entries, "d_exp_w", err) ||
            !dalloc(d_hint_logits_, (size_t) kMaxDecodeWindow * 2 * g_.n_expert, "d_hint_logits", err) ||
            !dalloc(d_hint_ids_, (size_t) kMaxDecodeWindow * kMoeHintK, "d_hint_ids", err) ||
            !dalloc(d_hint_w_, (size_t) kMaxDecodeWindow * kMoeHintK, "d_hint_w", err) ||
            !dalloc(d_x_lat_, (size_t) maxT * g_.n_latent, "d_x_lat", err) ||
            !dalloc(d_lat_sum_, (size_t) maxT * g_.n_latent, "d_lat_sum", err) ||
            !dalloc(d_lat_norm_, (size_t) maxT * g_.n_latent, "d_lat_norm", err) ||
            !dalloc(d_routed_out_, (size_t) maxT * H, "d_routed_out", err) ||
            !dalloc(d_exp_out_, (size_t) max_entries * g_.n_latent, "d_exp_out", err) ||
            !dalloc(d_logits_, (size_t) kMaxDecodeWindow * g_.n_vocab, "d_logits", err) ||
            !dalloc(d_picks_, kMaxDecodeWindow, "d_picks", err) ||
            !dalloc(d_lp_scratch_, kernels::row_logprobs_scratch_floats(kMaxDecodeWindow, g_.n_vocab), "d_lp_scratch", err) ||
            !dalloc(d_lp_out_, (size_t) kMaxDecodeWindow * kernels::kLogprobStride, "d_lp_out", err) ||
            !dalloc(d_mmq_bounds_, 2, "d_mmq_bounds", err) ||
            !dalloc(d_mmq_iota_, maxT + kMmqTile, "d_mmq_iota", err) ||
            !dalloc(d_attn_part_, (size_t) kAttnSplitSlots * std::max(g_.hca_heads, g_.mla_heads) * kAttnPartFloats, "d_attn_part", err) ||
            !dalloc(d_attn_qlat_, (size_t) kMaxDecodeWindow * g_.mla_heads * g_.mla_kv_lora, "d_attn_qlat", err) ||
            !dalloc(d_bf16_x_, (size_t) maxT * H, "d_bf16_x", err))
            return false;

        if (!ck(cudaMemset(d_mhc_cnt_, 0, kMaxDecodeWindow * sizeof(unsigned)), "d_mhc_cnt", err)) return false;

        // Quantized activation scratch for both MMVQ (up to 8 cols) and MMQ (up to kAttRows cols)
        const size_t q8_bytes = std::max(kernels::native_q8_1_bytes(q8_cols_max, kMaxDecodeWindow),
                                         prefill::mmq::q8_bytes(maxT, q8_cols_max));
        if (!ck(cudaMalloc(&d_xq_, q8_bytes), "d_xq", err) ||
            !ck(cudaMalloc(&d_xq_lat_, kernels::native_q8_1_bytes(g_.n_latent, kMaxDecodeWindow)), "d_xq_lat", err))
            return false;

        // Grouped GPU expert dispatch buffers for Decode (sized for kMaxDecodeWindow * n_used)
        const int dec_entries = kMaxDecodeWindow * g_.n_used;
        const size_t exp_scratch = kernels::native_expert_scratch_bytes(dec_entries, g_.n_ff_exp);
        if (!ck(cudaMalloc(&d_exp_scratch_, exp_scratch), "d_exp_scratch", err) ||
            !dalloc(d_grp_ptr_, dec_entries, "d_grp_ptr", err) ||
            !dalloc(d_grp_start_, dec_entries + 1, "d_grp_start", err) ||
            !dalloc(d_n_groups_, 1, "d_n_groups", err) ||
            !dalloc(d_ent_dst_, dec_entries, "d_ent_dst", err) ||
            !dalloc(d_ent_tok_, dec_entries, "d_ent_tok", err))
            return false;

        h_emb_.resize((size_t) maxT * H);
        h_pos_.resize(maxT);
        h_ids_.resize((size_t) kMaxPromptChunk * g_.n_used);
        h_grp_ptr_.resize(dec_entries);
        h_grp_start_.resize(dec_entries + 1);
        h_ent_dst_.resize(dec_entries);
        h_ent_tok_.resize(dec_entries);
        prefill::mmq::iota(d_mmq_iota_, maxT + kMmqTile, stream_);

        if (m_.dspark()) {
            drafter_ = m_.dspark();
            dwin_ = drafter_->open_window();
            if (dwin_ < 0) {
                err = "ernie55: more sessions than dspark windows";
                return false;
            }
            if (!dalloc(d_feat_, (size_t) kDsparkWindow * (size_t) drafter_->feature_floats(), "d_feat", err))
                return false;
        }

        return ck(cudaStreamSynchronize(stream_), "init sync", err);
    }

    model::Result prepare() override {
        auto& st = m_.shared_tier();
        if (st.tier) return {};
        cudaSetDevice(m_.device());
        std::string err;
        const size_t x_bytes = (size_t) kAttRows * (size_t) g_.n_latent * sizeof(float);
        const size_t row_bytes = (size_t) kAttRows * (size_t) g_.n_used * (size_t) g_.n_latent * sizeof(float);
        if (!ck(cudaHostAlloc(&st.h_x, x_bytes, cudaHostAllocMapped | cudaHostAllocPortable), "h_x", err) ||
            !ck(cudaHostAlloc(&st.h_rows, row_bytes, cudaHostAllocMapped | cudaHostAllocPortable), "h_rows", err))
            return {model::Status::Error, err};

        auto tier = std::make_shared<core::ExpertTier>(m_.tier_spec());
        PromptBufs probe;
        uint64_t home_prompt = 0;
        std::string pe;
        make_prompt_bufs(probe, kHomePromptChunk, pe, &home_prompt);
        // the ring beside the homes is permanent prompt scratch: cards that keep expert homes size it in too
        const uint64_t home_ring = expert_ring_bytes(std::min(kHomeRingExperts, g_.n_expert));
        const ctxp::TierReserve tr = ctxp::plan_tier_reserve(home_prompt, kMmqHeadroom, home_ring);
        core::ExpertTier::Reserve res;
        res.copies = tr.copies;
        res.homes = tr.homes;
        res.beside_homes = tr.beside_homes;
        res.modules = tr.modules;
        core::LoadPlan* lp = m_.planning();
        core::LoadPlan local_lp;
        if (!tier->open(res, st.h_rows, lp ? *lp : local_lp, err)) return {model::Status::Error, err};
        st.tier = std::move(tier);
        return {};
    }

    void end_prompt() override { release_prompt(); }

    int max_window() const override {
        if (!drafter_ || m_.flags().spec < 2) return 1;
        return std::clamp(m_.flags().spec, 1, std::min(kMaxDecodeWindow, drafter_->block_size() + 1));
    }

    int window_limit(int64_t pos) const override {
        if (max_window() <= 1) return 0;
        const int ratio = hca_ratio();
        const int rem = ratio - (int) (pos % ratio);
        return rem < kMaxDecodeWindow ? rem : 0;
    }

    model::Result draft_begin(int32_t anchor, int64_t pos, int n) override {
        if (!drafter_) return {model::Status::NotImplemented, "ernie55: no dspark drafter loaded"};
        std::string err;
        if (!flush_features(pos, err)) return {model::Status::Error, err};
        drafter_->forget_from(dwin_, pos);
        const int lim = std::min(max_window() - 1, drafter_->block_size());
        if (lim < 1) {
            last_draft_ = {};
            return {};
        }
        const int want = std::clamp(n, 1, lim);
        if (!drafter_->draft(dwin_, anchor, pos, want, stream_, last_draft_, err))
            return {model::Status::Error, err};
        return {};
    }

    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n, const DraftNeed&) override {
        if (!drafter_) return {model::Status::NotImplemented, "ernie55: no dspark drafter loaded"};
        n = std::min(max_n, last_draft_.n);
        std::copy(last_draft_.token, last_draft_.token + n, ids);
        std::copy(last_draft_.conf, last_draft_.conf + n, conf);
        return {};
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (big_.ready) release_prompt();
        if (r.n_tokens <= 0 || r.n_tokens > kMaxDecodeWindow)
            return {model::Status::Error, "ernie55: decode_window n_tokens out of range"};
        if (r.pos < 0 || r.pos + r.n_tokens > m_.context())
            return {model::Status::Refused, "ernie55: context limit reached"};
        std::string err;
        if (drafter_) {
            if (r.pos == 0) {
                drafter_->reset(dwin_);
                feat_n_ = 0;
            } else {
                if (!flush_features(r.pos, err)) return {model::Status::Error, err};
                drafter_->forget_from(dwin_, r.pos);
            }
        }
        auto& st = m_.shared_tier();
        if (st.tier && !st.dec) {
            st.tier->settle();
            if (!(st.dec = DecodeMoe::make(m_, *st.tier, st.h_rows, err))) return {model::Status::Error, err};
        }
        DecodeMoe* dec = st.dec.get();
        if (dec && !dec->begin_window(stream_, placement_dirty_, err)) return {model::Status::Error, err};
        placement_dirty_ = false;
        dec_ = dec;
        bool fwd = stage_rows(r.tokens, r.pos, r.n_tokens, err);
        if (fwd && st.tier && st.tier->route_lossy()) st.tier->publish_held(stream_);
        if (fwd) {
            const bool graphs = dec != nullptr;
            staged_ = true;
            fwd = graphs ? replay_window(r.tokens, r.pos, r.n_tokens, err)
                         : forward_rows(r.tokens, r.pos, r.n_tokens, /*want_logits=*/true, /*use_prompt_moe=*/false, err);
            staged_ = false;
            if (fwd && dec) dec->issue_copies();
        }
        dec_ = nullptr;
        if (!fwd) return {model::Status::Error, err};
        if (dec) dec->end_window(r.n_tokens, stream_);
        last_window_n_ = r.n_tokens;
        if (drafter_) {
            feat_pos0_ = r.pos;
            feat_n_ = r.n_tokens;
        }

        // Sample next token(s)
        sampling_ = r.sampling ? *r.sampling : kernels::SamplerParams{};
        if (!r.sampling) sampling_.greedy = true;
        sampling_.logit_bias = bias_on_ ? d_bias_ : nullptr;

        history_len_ = (r.history && r.history_len > 0) ? r.history_len : 0;
        if (history_len_ > 0) {
            const size_t elems = (size_t) r.n_tokens * (size_t) history_len_;
            if (!ensure_hist(elems, err)) return {model::Status::Error, err};
            if (!ck(cudaMemcpyAsync(d_hist_, r.history, elems * sizeof(int32_t), cudaMemcpyHostToDevice, stream_),
                    "history copy", err))
                return {model::Status::Error, err};
        }
        kernels::sample_tokens(d_logits_, r.n_tokens, g_.n_vocab, history_len_ > 0 ? d_hist_ : nullptr, history_len_,
                               sampling_, d_picks_, stream_);
        std::vector<int32_t> picks((size_t) r.n_tokens);
        if (!ck(cudaMemcpyAsync(picks.data(), d_picks_, (size_t) r.n_tokens * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                stream_),
                "picks copy", err))
            return {model::Status::Error, err};
        if (r.logits) {
            if (!ck(cudaMemcpyAsync(r.logits, d_logits_ + (size_t) (r.n_tokens - 1) * g_.n_vocab,
                                    (size_t) g_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, stream_),
                    "logits copy", err))
                return {model::Status::Error, err};
        }
        if (!ck(cudaStreamSynchronize(stream_), "decode sync", err)) return {model::Status::Error, err};
        if (dec && (!dec->after_window(stream_, err) || dec->failed(err))) return {model::Status::Error, err};
        if (r.next_token) *r.next_token = picks.back();
        if (r.row_tokens) std::memcpy(r.row_tokens, picks.data(), (size_t) r.n_tokens * sizeof(int32_t));
        return {};
    }

    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        if (r.n_tokens <= 0) return {};
        if (r.pos < 0 || r.pos + r.n_tokens > m_.context())
            return {model::Status::Refused, "ernie55: prefill exceeds context"};
        if (auto& st = m_.shared_tier(); st.tier) st.tier->forget_use();
        placement_dirty_ = true;   // a prompt lends and takes back slots and looks experts up itself
        std::string err;
        if (drafter_) {
            if (r.pos == 0) {
                drafter_->reset(dwin_);
                feat_n_ = 0;
            } else {
                if (!flush_features(r.pos, err)) return {model::Status::Error, err};
                drafter_->forget_from(dwin_, r.pos);
            }
        }

        const int total = r.total_tokens > 0 ? r.total_tokens : r.n_tokens;
        const bool want_mmq = total > kMaxDecodeWindow && prefill::mmq::built() &&
                              prefill::mmq::fits(kTypeMXFP4, 2 * g_.n_ff_exp) &&
                              prefill::mmq::fits(kTypeMXFP4, g_.n_latent);
        if (want_mmq && !big_.ready) {
            if (!open_prompt(total, err)) {
                basalt::log::warn("basalt ernie55", "prompt MMQ staging unavailable (%s); falling back to decode path",
                                  err.c_str());
                err.clear();
            }
        }

        const int chunk = big_.ready ? big_.rows : kAttRows;
        int done = 0;
        while (done < r.n_tokens) {
            if (r.stop && r.stop()) break;
            const int take = std::min(chunk, r.n_tokens - done);
            const bool use_prompt_moe = big_.ready && take > kMaxDecodeWindow;
            if (!forward_rows(r.tokens + done, r.pos + done, take, /*want_logits=*/false, use_prompt_moe, err)) {
                if (!r.more) release_prompt();
                return {model::Status::Error, err};
            }
            if (drafter_) {
                const int tap = std::min(take, kDsparkWindow);
                feat_pos0_ = r.pos + done + take - tap;
                feat_n_ = tap;
                if (!flush_features(r.pos + done + take, err)) {
                    if (!r.more) release_prompt();
                    return {model::Status::Error, err};
                }
            }
            done += take;
            if (r.progress) r.progress(done);
        }
        if (!r.more) release_prompt();
        return {};
    }

    model::Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (!bias_on_) return {};
        cudaSetDevice(m_.device());
        std::string err;
        if (!device_bias(d_bias_, bias, err)) return {model::Status::Error, err};
        return {};
    }

    model::Result resample_row(int row, const float* bias, int32_t& token) override {
        if (row < 0 || row >= last_window_n_) return {model::Status::Error, "ernie55: resample_row out of range"};
        cudaSetDevice(m_.device());
        std::string err;
        if (bias && !device_bias(d_mask_, bias, err)) return {model::Status::Error, err};
        kernels::SamplerParams sp = sampling_;
        sp.logit_bias = bias ? d_mask_ : (bias_on_ ? d_bias_ : nullptr);
        sp.counter += (uint64_t) row;
        const int32_t* hrow = history_len_ > 0 ? d_hist_ + (size_t) row * (size_t) history_len_ : nullptr;
        kernels::sample_tokens(d_logits_ + (size_t) row * g_.n_vocab, 1, g_.n_vocab, hrow, history_len_, sp, d_picks_,
                               stream_);
        if (!ck(cudaMemcpyAsync(&token, d_picks_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_), "resample copy",
                err) ||
            !ck(cudaStreamSynchronize(stream_), "resample sync", err))
            return {model::Status::Error, err};
        return {};
    }

    model::Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (row0 < 0 || n <= 0 || row0 + n > last_window_n_)
            return {model::Status::Error, "ernie55: row_logprobs out of range"};
        std::string err;
        cudaMemcpyAsync(d_picks_, picks, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
        kernels::row_logprobs(d_logits_ + (size_t) row0 * g_.n_vocab, n, g_.n_vocab, d_picks_, top, d_lp_out_,
                              d_lp_scratch_, stream_);
        cudaMemcpyAsync(out, d_lp_out_, (size_t) n * kernels::kLogprobStride * sizeof(float), cudaMemcpyDeviceToHost,
                        stream_);
        if (!ck(cudaStreamSynchronize(stream_), "logprobs sync", err)) return {model::Status::Error, err};
        return {};
    }

    model::Result copy_logits(int row, float* out) override {
        if (row < 0 || row >= last_window_n_) return {model::Status::Error, "ernie55: copy_logits out of range"};
        std::string err;
        cudaMemcpyAsync(out, d_logits_ + (size_t) row * g_.n_vocab, (size_t) g_.n_vocab * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        if (!ck(cudaStreamSynchronize(stream_), "copy_logits sync", err)) return {model::Status::Error, err};
        return {};
    }

    size_t running_bytes() const override {
        size_t tot = 0;
        for (int l = 0; l < g_.n_layers; ++l) {
            if (!g_.is_dsa(l)) {
                const auto& s = hca_[(size_t) l];
                tot += s.win_bytes + 2 * s.state_bytes;
            }
        }
        return tot;
    }

    model::Result running_state(int64_t, bool, core::RunningState& rs) override {
        rs.stream = stream_;
        rs.regions.clear();
        rs.head.clear();
        for (int l = 0; l < g_.n_layers; ++l) {
            if (!g_.is_dsa(l)) {
                const auto& s = hca_[(size_t) l];
                rs.regions.push_back({s.win_cache, s.win_bytes});
                rs.regions.push_back({s.state_kv, s.state_bytes});
                rs.regions.push_back({s.state_score, s.state_bytes});
            }
        }
        return {};
    }

    model::Result running_restored(int64_t pos, const uint8_t*) override {
        if (drafter_) drafter_->forget_from(dwin_, pos);
        return {};
    }

    // ---- conversations: DSA latent/indexer rows (k/v) and HCA compressed rows (pooled) by position
    int cell_format() const { return m_.cache_fp8() ? 1 : 3; }   // 1: FP8/FP4 packed, 3: FP16
    size_t hca_row_bytes() const { return m_.cache_fp8() ? (size_t) kHcaFp8RowBytes : (size_t) g_.hca_head_dim * 2; }
    size_t dsa_row_bytes() const {
        return m_.cache_fp8() ? (size_t) kDsaFp8RowBytes : (size_t) (g_.mla_kv_lora + g_.mla_qk_rope) * 2;
    }
    size_t idx_row_bytes() const { return m_.cache_fp8() ? (size_t) kIdxFp4RowBytes : (size_t) g_.idx_dim * 2; }
    int hca_ratio() const {
        for (int l = 0; l < g_.n_layers; ++l)
            if (!g_.is_dsa(l)) return std::max(1, g_.compress_ratios[(size_t) l]);
        return 1;
    }

    int64_t disk_row_cells() const override { return hca_ratio(); }

    size_t cells_bytes(int64_t positions) const override {
        if (positions <= 0) return 0;
        size_t tot = 0;
        const size_t hca_r = hca_row_bytes(), dsa_r = dsa_row_bytes(), idx_r = idx_row_bytes();
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.is_dsa(l)) {
                tot += (size_t) positions * (dsa_r + idx_r);
            } else {
                const int ratio = std::max(1, g_.compress_ratios[(size_t) l]);
                tot += (size_t) (positions / ratio + 1) * hca_r;
            }
        }
        return tot;
    }

    model::Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (positions < 0 || positions > m_.context())
            return {model::Status::Error, "ernie55: positions past context"};
        cudaSetDevice(m_.device());
        std::string err;
        if (!ck(cudaStreamSynchronize(stream_), "save_cells sync", err)) return {model::Status::Error, err};
        kv.clear();
        if (positions == 0) return {};
        kv.reserve((size_t) g_.n_layers);
        const size_t hca_r = hca_row_bytes(), dsa_r = dsa_row_bytes(), idx_r = idx_row_bytes();
        const auto down = [](const uint8_t* dev, core::ConversationBuffer& to) {
            return to.visit(0, to.size(), [&](uint8_t* p, size_t n, size_t at) {
                return basalt::core::device_copy(p, dev + at, n, cudaMemcpyDeviceToHost) == cudaSuccess;
            });
        };
        for (int l = 0; l < g_.n_layers; ++l) {
            core::ConversationKv k;
            k.format = cell_format();
            k.page_size = 1;
            k.cells = positions;
            k.heads = 1;
            if (g_.is_dsa(l)) {
                const auto& s = dsa_[(size_t) l];
                k.head_dim = g_.mla_kv_lora + g_.mla_qk_rope;
                k.idx_dim = g_.idx_dim;
                k.k.resize_for_overwrite((size_t) positions * dsa_r);
                k.v.resize_for_overwrite((size_t) positions * idx_r);
                if (!down(s.kv_cache, k.k) || !down(s.idx_k_cache, k.v))
                    return {model::Status::Error, "ernie55: reading a DSA cache back failed"};
            } else {
                const auto& s = hca_[(size_t) l];
                const int ratio = std::max(1, g_.compress_ratios[(size_t) l]);
                const int64_t rows = positions / ratio + 1;
                k.head_dim = g_.hca_head_dim;
                k.pooled_rows = rows;
                k.idx_dim = g_.hca_head_dim;
                k.pooled.resize_for_overwrite((size_t) rows * hca_r);
                if (!down(s.comp_cache, k.pooled))
                    return {model::Status::Error, "ernie55: reading an HCA compressed cache back failed"};
            }
            kv.push_back(std::move(k));
        }
        return {};
    }

    model::Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (positions < 0 || positions > m_.context())
            return {model::Status::Error, "ernie55: positions past context"};
        if (positions == 0) {
            if (drafter_) drafter_->reset(dwin_);
            return {};
        }
        if (kv.size() != (size_t) g_.n_layers)
            return {model::Status::Error, "ernie55: a parked conversation has a different layer count"};
        cudaSetDevice(m_.device());
        std::string err;
        if (!ck(cudaStreamSynchronize(stream_), "restore_cells sync", err)) return {model::Status::Error, err};
        const size_t hca_r = hca_row_bytes(), dsa_r = dsa_row_bytes(), idx_r = idx_row_bytes();
        const auto up = [](uint8_t* dev, const core::ConversationBuffer& from) {
            return from.visit(0, from.size(), [&](const uint8_t* p, size_t n, size_t at) {
                return basalt::core::device_copy(dev + at, p, n, cudaMemcpyHostToDevice) == cudaSuccess;
            });
        };
        for (int l = 0; l < g_.n_layers; ++l) {
            const core::ConversationKv& k = kv[(size_t) l];
            if (k.format != cell_format() || k.page_size != 1 || k.cells != positions)
                return {model::Status::Error, "ernie55: a parked conversation has other caches"};
            if (g_.is_dsa(l)) {
                const auto& s = dsa_[(size_t) l];
                if (k.pooled_rows != 0 || k.k.size() != (size_t) positions * dsa_r ||
                    k.v.size() != (size_t) positions * idx_r)
                    return {model::Status::Error, "ernie55: a parked conversation has mismatched DSA caches"};
                if (!up(s.kv_cache, k.k) || !up(s.idx_k_cache, k.v))
                    return {model::Status::Error, "ernie55: writing a DSA cache back failed"};
            } else {
                const auto& s = hca_[(size_t) l];
                const int ratio = std::max(1, g_.compress_ratios[(size_t) l]);
                const int64_t rows = positions / ratio + 1;
                if (k.pooled_rows != rows || k.pooled.size() != (size_t) rows * hca_r)
                    return {model::Status::Error, "ernie55: a parked conversation has mismatched HCA caches"};
                if (!up(s.comp_cache, k.pooled))
                    return {model::Status::Error, "ernie55: writing an HCA compressed cache back failed"};
            }
        }
        if (drafter_) drafter_->reset(dwin_);
        return {};
    }

    ExpertCounts expert_counts() const override {
        const auto& t = m_.shared_tier().tier;
        if (!t) return {};
        return {t->hits(), t->lookups(), t->offloaded(), t->file_lookups()};
    }

    std::vector<CardExperts> experts_by_card() const override {
        const auto& t = m_.shared_tier().tier;
        if (!t) return {};
        std::vector<CardExperts> out;
        for (const auto& c : t->slots_by_card()) out.push_back({c.device, c.slots, c.bytes});
        return out;
    }

    int64_t kv_bytes() const override { return kv_bytes_; }

    bool expert_map(int64_t& total, std::vector<uint8_t>& homes, std::vector<uint32_t>& routed) override {
        const auto& t = m_.shared_tier().tier;
        if (!t) return false;
        t->expert_map(total, homes, routed);
        return true;
    }

    std::string info_words() const override {
        const std::string kv = m_.cache_fp8() ? "kv=fp8" : "kv=fp16";
        const auto& t = m_.shared_tier().tier;
        return t ? "expert_slots=" + std::to_string(t->gpu_slots()) + " expert_slots_primary=" +
                       std::to_string(t->slots()) + " ram_experts_mib=" + std::to_string(t->ram_bytes() >> 20) + " " +
                       kv + " " + program::spill_words(t->spill())
                 : kv + " " + program::spill_words({});
    }

    model::Result set_low_memory(bool low, int64_t& expert_slots) override {
        std::string e;
        const auto& t = m_.shared_tier().tier;
        if (!t) return {model::Status::Error, "ernie55: the expert tier is not open"};
        if (const cudaError_t se = cudaStreamSynchronize(stream_); se != cudaSuccess)
            return {model::Status::Error, std::string("ernie55: cudaStreamSynchronize: ") + cudaGetErrorString(se)};
        if (low) {
            t->lend_all();
        } else {
            if (!t->reclaim(e, /*restore_lent=*/true)) return {model::Status::Error, "ernie55: " + e};
            t->grow();
        }
        placement_dirty_ = true;
        expert_slots = t->gpu_slots();
        return {};
    }

    model::Result set_vram_reserve(int64_t mib, int64_t& expert_slots) override {
        std::string e;
        const auto& t = m_.shared_tier().tier;
        if (!t) return {model::Status::Error, "ernie55: the expert tier is not open"};
        if (const cudaError_t se = cudaStreamSynchronize(stream_); se != cudaSuccess)
            return {model::Status::Error, std::string("ernie55: cudaStreamSynchronize: ") + cudaGetErrorString(se)};
        if (!t->replan(mib, e)) return {model::Status::Error, "ernie55: " + e};
        placement_dirty_ = true;
        expert_slots = t->gpu_slots();
        return {};
    }

private:
    const DevTensor* req_t(const std::string& name, std::string& err) const {
        const DevTensor* t = m_.tensor(name);
        if (!t) err = "ernie55: missing tensor " + name;
        return t;
    }
    const DevTensor* req_t(int l, const char* suf, std::string& err) const {
        return req_t("blk." + std::to_string(l) + "." + suf, err);
    }

    // ---- Temporary Prompt Buffer Lifecycle (lends VRAM from ExpertTier and returns it cleanly) ----

    bool make_prompt_bufs(PromptBufs& b, int R, std::string& err, uint64_t* count_only = nullptr) {
        namespace mmq = prefill::mmq;
        b.release();
        bool ok = true;
        auto alloc_bytes = [&](size_t bytes) -> void* {
            const size_t n = std::max<size_t>(bytes, 16);
            if (count_only) {
                *count_only += (n + (2ull << 20) - 1) / (2ull << 20) * (2ull << 20);
                return (void*) 1;
            }
            if (!ok) return nullptr;
            void* p = nullptr;
            if (cudaMalloc(&p, n) != cudaSuccess) { ok = false; return nullptr; }
            b.dev_owned.push_back(p);
            return p;
        };
        auto alloc_f = [&](size_t n) { return (float*) alloc_bytes(n * sizeof(float)); };
        auto alloc_i = [&](size_t n) { return (int32_t*) alloc_bytes(n * sizeof(int32_t)); };
        auto pin_i = [&](size_t n) -> int32_t* {
            if (count_only) return (int32_t*) 1;
            if (!ok) return nullptr;
            void* p = nullptr;
            if (cudaMallocHost(&p, std::max<size_t>(n * sizeof(int32_t), 16)) != cudaSuccess) { ok = false; return nullptr; }
            b.host_owned.push_back(p);
            return (int32_t*) p;
        };

        const int H = g_.n_embd, K = g_.n_used, L = g_.n_latent, F = g_.n_ff_exp;
        b.rows = R;
        if (R > kAttRows) {
            b.own_row_bufs = true;
            b.pos = alloc_i(R);
            b.res_a = alloc_f((size_t) R * kHcMult * H);
            b.res_b = b.res_a;   // mhc_post_kernel is in-place safe (loads all 4 streams into registers before write)
            b.x_in = alloc_f((size_t) R * H);
            b.x_norm = alloc_f((size_t) R * H);
            b.sub_out = alloc_f((size_t) R * H);
            b.mhc_ws = alloc_f(mhc_workspace_bytes(R) / sizeof(float));
            b.mhc_coef = alloc_f((size_t) R * kHcCoef);
            b.ff_g = alloc_f((size_t) R * std::max(F, 2 * g_.n_expert));
            b.ff_u = alloc_f((size_t) R * std::max(F, L));
            // Disjoint-lifetime aliases (strictly serialized on stream_, cuts VRAM ~40% so R=4096/8192 fit in partial lend)
            b.ff_h = b.ff_g;        // situ_glu_f32 is in-place safe
            b.rlogits = b.ff_g;     // router completes before sg writes ff_g
            b.x_lat = b.x_in;       // x_in dead after rms_norm_rows (H >= L); x_lat dead after moe_prompt
            b.lat_sum = b.ff_g;     // ff_g/ff_h dead after sd; lat_sum written at end of moe_prompt
            b.lat_norm = b.ff_u;    // ff_u dead after situ_glu; lat_norm written after moe_prompt
            b.routed_out = b.x_in;  // x_lat dead after moe_prompt; routed_out written by lu
            b.exp_ids = alloc_i((size_t) R * K);
            b.exp_w = alloc_f((size_t) R * K);
        } else {
            b.own_row_bufs = false;
            b.pos = d_pos_; b.res_a = d_res_a_; b.res_b = d_res_b_; b.x_in = d_x_in_;
            b.x_norm = d_x_norm_; b.sub_out = d_sub_out_; b.mhc_ws = d_mhc_ws_; b.mhc_coef = d_mhc_coef_;
            b.ff_g = d_ff_g_; b.ff_u = d_ff_u_; b.ff_h = d_ff_h_;
            b.rlogits = d_rlogits_; b.exp_ids = d_exp_ids_; b.exp_w = d_exp_w_;
            b.x_lat = d_x_lat_; b.lat_sum = d_lat_sum_; b.lat_norm = d_lat_norm_; b.routed_out = d_routed_out_;
        }

        if (R > kMaxDecodeWindow) b.router_w = alloc_f((size_t) 2 * g_.n_expert * H);
        b.pair_rows = alloc_f((size_t) R * K * L);
        b.pair_map = alloc_i((size_t) R * K);
        b.pair_ident = alloc_i((size_t) R * K + kMmqTile);
        b.h_pair_map = pin_i((size_t) R * K);
        b.h_ids = pin_i((size_t) R * K);
        b.h_launch_tok = pin_i((size_t) R * K);
        // The provable worst case for the pre-packed bounds tables, which the ringed and the staged launch path share:
        // a path's launches walk <= n_expert expert slices plus one slice per kLaunchRows of its rows (<= R * K /
        // kLaunchRows extra), and each launch writes 1 + (the slices it covers) entries, so <= 2 * (n_expert +
        // R * K / kLaunchRows + 1) per path.  Twice that, plus slack (a few KiB of pinned memory), holds both at once.
        b.h_launch_bounds_cap = 4 * (size_t) g_.n_expert + 4 * ((size_t) R * K / kLaunchRows + 1) + 16;
        b.h_launch_bounds = pin_i(b.h_launch_bounds_cap);
        b.gu = alloc_f((size_t) kLaunchRows * 2 * F);
        b.h = alloc_f((size_t) kLaunchRows * F);
        b.xq1 = alloc_bytes(mmq::q8_bytes(std::max(kLaunchRows, R), std::max(L, H)));
        b.xq2 = alloc_bytes(mmq::q8_bytes(std::max(kLaunchRows, R), std::max(F, L)));
        b.xq_lat = alloc_bytes(mmq::q8_bytes(R, L));

        b.kvb_f16 = (uint16_t*) alloc_bytes((size_t) g_.mla_heads * (size_t) (g_.mla_qk_nope + g_.mla_v_dim) *
                                            (size_t) g_.mla_kv_lora * sizeof(uint16_t));
        b.q_nope_f16 = (uint16_t*) alloc_bytes((size_t) g_.mla_heads * (size_t) kAttRows *
                                               (size_t) g_.mla_qk_nope * sizeof(uint16_t));
        b.o_lat_f16 = (uint16_t*) alloc_bytes((size_t) g_.mla_heads * (size_t) kAttRows *
                                              (size_t) g_.mla_kv_lora * sizeof(uint16_t));
        b.idx_q_bf16 = (uint16_t*) alloc_bytes((size_t) kAttRows * (size_t) g_.idx_heads * (size_t) g_.idx_dim *
                                               sizeof(uint16_t));
        b.idx_k_bf16 = (uint16_t*) alloc_bytes((size_t) kIdxKeyTile * (size_t) g_.idx_dim * sizeof(uint16_t));
        b.idx_dots = alloc_f((size_t) kMaxDecodeWindow * (size_t) g_.idx_heads * (size_t) kIdxKeyTile);
        b.oa_bounds = alloc_i((size_t) g_.hca_o_groups + 1);
        b.oa_ids = alloc_i((size_t) g_.hca_o_groups * (size_t) kAttRows);
        b.oa_cached_A = -1;

        for (int i = 0; i < 2; ++i) {
            b.stage[i].blobs = (uint8_t*) alloc_bytes((size_t) kStageExperts * stride_);
            b.stage[i].tok = alloc_i(kLaunchRows);
            b.stage[i].bounds = alloc_i(kStageExperts + 1);
        }

        if (count_only) {
            b.dev_owned.clear();
            b.host_owned.clear();
            b.ready = false;
            return true;
        }
        if (ok) {
            for (int i = 0; i < 2; ++i) {
                if (cudaEventCreateWithFlags(&b.stage[i].copied, cudaEventDisableTiming) != cudaSuccess ||
                    cudaEventCreateWithFlags(&b.stage[i].done, cudaEventDisableTiming) != cudaSuccess) {
                    ok = false;
                    break;
                }
                cudaEventRecord(b.stage[i].done, stream_);
            }
            if (ok && (cudaEventCreateWithFlags(&b.ev_route, cudaEventDisableTiming) != cudaSuccess ||
                       cudaEventCreateWithFlags(&b.ev_ids, cudaEventDisableTiming) != cudaSuccess ||
                       cudaEventCreateWithFlags(&b.ev_xlat, cudaEventDisableTiming) != cudaSuccess ||
                       cudaEventCreateWithFlags(&b.ev_map, cudaEventDisableTiming) != cudaSuccess)) {
                ok = false;
            }
            if (ok) cudaEventRecord(b.ev_map, stream_);
        }
        if (!ok) {
            b.release();
            cudaGetLastError();
            err = "out of device memory for prompt chunk R=" + std::to_string(R);
            return false;
        }
        mmq::iota(b.pair_ident, (int64_t) R * K + kMmqTile, stream_);
        b.ready = true;
        return true;
    }

    // The provable capacity of the pinned launch-bounds table (see make_prompt_bufs): one launch writes 1 + nb
    // entries at bnd_cur, and exceeding that bound would overrun the table, so the chunk fails instead of it.
    static bool bounds_fit(const PromptBufs& b, int32_t bnd_cur, int nb, std::string& err) {
        if ((size_t) bnd_cur + (size_t) (nb + 1) <= b.h_launch_bounds_cap) return true;
        assert(false);
        err = "ernie55: prompt launch bounds overflow (" + std::to_string((long long) bnd_cur) + " + " +
              std::to_string(nb + 1) + " of " + std::to_string((long long) b.h_launch_bounds_cap) + ")";
        return false;
    }

    bool open_prompt(int total, std::string& err) {
        if (big_.ready) return true;
        cudaSetDevice(m_.device());
        cudaStreamSynchronize(stream_);
        core::ExpertTier* tier = m_.shared_tier().tier.get();

        // Pre-load MMQ MXFP4 & Q8_0 modules before any async copy stream starts
        if (loaded_mmq_.insert({kTypeMXFP4, 2 * g_.n_ff_exp}).second) prefill::mmq::load(kTypeMXFP4, 2 * g_.n_ff_exp);
        if (loaded_mmq_.insert({kTypeMXFP4, g_.n_latent}).second) prefill::mmq::load(kTypeMXFP4, g_.n_latent);
        if (loaded_mmq_.insert({kTypeQ8_0, g_.n_ff_exp}).second) prefill::mmq::load(kTypeQ8_0, g_.n_ff_exp);
        if (loaded_mmq_.insert({kTypeQ8_0, g_.n_latent}).second) prefill::mmq::load(kTypeQ8_0, g_.n_latent);
        if (loaded_mmq_.insert({kTypeQ8_0, g_.n_embd}).second) prefill::mmq::load(kTypeQ8_0, g_.n_embd);
        if (loaded_mmq_.insert({kTypeQ8_0, g_.hca_o_lora}).second) prefill::mmq::load(kTypeQ8_0, g_.hca_o_lora);
        if (loaded_mmq_.insert({kTypeQ8_0, g_.hca_head_dim}).second) prefill::mmq::load(kTypeQ8_0, g_.hca_head_dim);
        if (loaded_mmq_.insert({kTypeQ8_0, g_.hca_o_groups * g_.hca_o_lora}).second)
            prefill::mmq::load(kTypeQ8_0, g_.hca_o_groups * g_.hca_o_lora);

        // the promotions queued by the decode windows land first: until then an evicted home has neither its slot nor
        // its RAM place, and the prompt reads the places
        if (tier) tier->settle();

        int want = kMaxDecodeWindow * 2;
        if (m_.flags().diag_rows > 0) {
            want = std::clamp(m_.flags().diag_rows, kMaxDecodeWindow * 2, kMaxPromptChunk);
        } else {
            while (want < total && want < kMaxPromptChunk) want *= 2;
            want = std::min(want, kMaxPromptChunk);
        }

        t_prompt_moe_ms_ = 0;
        t_prompt_peer_wait_ms_ = 0;
        t_ring_ms_ = 0.0;
        n_prompt_dev_ = n_prompt_bus_ = n_prompt_peer_ = n_prompt_cpu_ = n_prompt_file_ = n_prompt_ring_ = 0;
        n_prompt_peer_rows_ = n_prompt_cpu_rows_ = 0;

        // First try bounded partial lend, which leaves GPU 0's hottest decode experts on the card. A chunk wide enough
        // to route to (almost) every expert asks for its ring first and for `want` without the ring second: the ring
        // only pays off at that width (its bytes are 2 * cap expert blobs, and a narrower chunk reloads each layer's
        // experts either way), so a long prompt that does not fit goes to lend_all() whole rather than to a chunk
        // whose MoE PCIe reloads are no longer amortized. A shorter chunk keeps its halving shrink instead.
        if (tier && tier->partial_lend()) {
            struct Try {
                int r;
                uint64_t ring;   ///< ring bytes to leave room for beside the chunk
            };
            std::vector<Try> tries;
            const uint64_t ring_need =
                (total >= kRingRows && want >= kRingRows) ? expert_ring_bytes(ring_target_cap()) : 0;
            tries.push_back({want, ring_need});
            if (ring_need > 0) tries.push_back({want, 0});   // the ring is what a long chunk pays for, R is not
            else
                for (int R = want / 2; R >= kMaxDecodeWindow * 2; R /= 2) tries.push_back({R, 0});
            for (const Try& t : tries) {
                PromptBufs probe;
                uint64_t need = 0;
                std::string pe;
                make_prompt_bufs(probe, t.r, pe, &need);
                if (!tier->lend(need + t.ring + kMmqHeadroom)) {
                    if (m_.flags().diag_rows > 0 && t.ring == 0) break;   // --diag-rows tests exactly the width it asked for
                    continue;
                }
                // From the byte lend() gives up, `lent_` says the tier owns VRAM again: every way out of this function
                // (and release_prompt) then runs tier->reclaim(), so nothing a lend took can stay unmapped.
                lent_ = true;
                if (make_prompt_bufs(big_, t.r, pe)) {
                    // MMQ allocates its stream-k buffers on its first launches: a chunk that left the card no room
                    // for them ended the engine
                    size_t free_b = 0, total_b = 0;
                    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b >= kMmqHeadroom) {
                        open_prompt_ring(total, t.r);
                        return true;
                    }
                    big_.release();
                }
                break;   // a partial lend has given all it may give: only lend_all() below can top that up
            }
        }
        if (tier) {
            tier->lend_all();
            lent_ = true;
        }
        for (int R = want; R >= kMaxDecodeWindow * 2; R /= 2) {
            if (make_prompt_bufs(big_, R, err)) {
                // MMQ allocates its stream-k buffers on its first launches: a chunk that left the card no room for
                // them ended the engine
                size_t free_b = 0, total_b = 0;
                if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b >= kMmqHeadroom) {
                    open_prompt_ring(total, R);
                    return true;
                }
                big_.release();
                err = "no room for MMQ beside the prompt chunk R=" + std::to_string(R);
            }
            if (m_.flags().diag_rows > 0) break;
        }
        if (tier && lent_) {
            std::string e;
            tier->reclaim(e, true);
            tier->grow();
            lent_ = false;
        }
        return false;
    }

    // The packed prefill expert ring beside the chunk's own buffers (long chunks only). Non-fatal: with no room for
    // it every layer's experts keep staging through stage_[2] as before.
    void open_prompt_ring(int total, int R) {
        if (ring_ || total < kRingRows || R < kRingRows || !m_.shared_tier().tier) return;
        if (!ring_open(ring_target_cap(), /*allow_shrink=*/true)) cudaGetLastError();
    }

    void release_prompt() {
        if (!big_.ready) {
            ring_close();   // belt and braces: a ring only ever opens beside a ready prompt chunk
            return;
        }
        cudaSetDevice(m_.device());
        cudaStreamSynchronize(stream_);
        cudaStreamSynchronize(copy_stream_);
        if (t_prompt_moe_ms_ > 0.0) {
            const std::string ring_w =
                n_prompt_ring_ > 0 || t_ring_ms_.load() > 0.0
                    ? " ring=" + std::to_string((long long) n_prompt_ring_) + "(" +
                          std::to_string((long long) t_ring_ms_.load()) + "ms)"
                    : "";
            basalt::log::info("basalt ernie55",
                              "prompt MMQ (R=%d): moe=%.0fms (peer_wait=%.0fms), experts dev=%lld bus=%lld file=%lld "
                              "peer=%lld(%lld rows) cpu=%lld(%lld rows)%s",
                              big_.rows, t_prompt_moe_ms_, t_prompt_peer_wait_ms_,
                              (long long) n_prompt_dev_, (long long) n_prompt_bus_, (long long) n_prompt_file_,
                              (long long) n_prompt_peer_, (long long) n_prompt_peer_rows_,
                              (long long) n_prompt_cpu_, (long long) n_prompt_cpu_rows_, ring_w.c_str());
        }
        ring_close();
        big_.release();
        if (core::ExpertTier* tier = m_.shared_tier().tier.get(); tier && lent_) {
            std::string e;
            if (!tier->reclaim(e, /*restore_lent=*/true))
                basalt::log::warn("basalt ernie55", "expert tier reclaim: %s", e.c_str());
            tier->grow();
        }
        lent_ = false;
    }

    // ---- Packed adaptive prefill expert ring (ring_stream_, long chunks only) ----

    // A prompt chunk of kRingRows rows or more routes to (almost) every expert of every layer, so the next layer's
    // experts do not wait for its router: a layer's local experts are packed into one of two ring slots on the ring
    // stream while the layer before it computes. ring_stream_ is its own non-blocking stream: hundreds of H2D copies
    // queue there without ever holding the copy_stream_ (the ev_ids D2H and the stage_[2] overflow staging share it)
    // or letting the routing readback wait behind them.

    // The bytes the prompt MMQ products read past the last expert of a blob buffer: the last row of a matrix whose
    // rows are not a multiple of 256 values is read on to the next multiple (zeros there, a stray scale could be NaN).
    size_t expert_past() const {
        namespace mmq = prefill::mmq;
        size_t past = 16;
        for (const ExpertLayer& el : m_.pack().experts) {
            if (el.blob_bytes == 0) continue;
            if (mmq::supported(el.gu_type)) past = std::max(past, mmq::read_past_bytes(el.gu_type, g_.n_latent));
            if (mmq::supported(el.d_type)) past = std::max(past, mmq::read_past_bytes(el.d_type, g_.n_ff_exp));
        }
        return past;
    }
    size_t expert_ring_bytes(int cap) const { return 2 * (size_t) cap * stride_ + expert_past(); }

    // How many of one layer's experts GPU 0 needs: all non-peer experts on single-GPU, or at most half of the
    // non-peer RAM experts when PeerShare splits work across two cards (so GPU 0 never starves GPU 1).
    int ring_target_cap() const {
        core::ExpertTier* tier = m_.shared_tier().tier.get();
        if (!tier || !tier->share_for(0)) return g_.n_expert;
        int peer_res = 0;
        for (int e = 0; e < g_.n_expert; ++e)
            if (tier->locate(0, e).peer) ++peer_res;
        const int half_ram = std::max(0, (g_.n_expert - peer_res) / 2);
        return std::clamp(half_ram / kStageExperts * kStageExperts, kRingMinExperts, g_.n_expert);
    }

    // Shrinks the ring to whatever fits above the MMQ headroom, in whole kStageExperts steps, never below
    // kRingMinExperts; false (and nothing allocated) when even that does not fit.
    bool ring_open(int want_cap, bool allow_shrink) {
        const size_t past = expert_past();
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return false;
        const uint64_t floor = kMmqHeadroom + (64ull << 20);
        if (stride_ == 0 || free_b <= floor + past) return false;
        int cap = std::clamp(want_cap > 0 ? want_cap : g_.n_expert, kRingMinExperts, g_.n_expert);
        int fit = (int) ((free_b - floor - past) / (2 * stride_));
        fit = std::min(fit, cap) / kStageExperts * kStageExperts;   // whole staging groups, rounded down
        if (cap > fit) {
            if (!allow_shrink) return false;
            cap = fit;
        }
        while (cap >= kRingMinExperts) {
            if (cudaMalloc((void**) &ring_, expert_ring_bytes(cap)) == cudaSuccess) {
                size_t after_free = 0;
                if (cudaMemGetInfo(&after_free, &total_b) == cudaSuccess && after_free >= kMmqHeadroom) break;
                cudaFree(ring_);   // MMQ still needs its own room beside the ring
                ring_ = nullptr;
            } else {
                cudaGetLastError();
            }
            cap -= kStageExperts;
        }
        if (!ring_) return false;
        // the whole ring starts zeroed, the read-past bytes above the last slot of slot 1 included
        if (cudaMemset(ring_, 0, expert_ring_bytes(cap)) != cudaSuccess) {
            cudaGetLastError();
            cudaFree(ring_);
            ring_ = nullptr;
            return false;
        }
        ring_cap_ = cap;
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
        basalt::log::info("basalt ernie55", "prompt expert ring: %d of %d experts per layer (%.2f GiB)", ring_cap_,
                          g_.n_expert, (double) expert_ring_bytes(ring_cap_) / 1073741824.0);
        return true;
    }

    void ring_join() {
        for (std::thread& t : ring_thr_)
            if (t.joinable()) t.join();
    }

    void ring_close() {
        ring_join();
        if (!ring_) return;
        cudaStreamSynchronize(ring_stream_);
        cudaStreamSynchronize(stream_);
        cudaFree(ring_);
        ring_ = nullptr;
        ring_cap_ = 0;
        ring_pos_[0].clear();
        ring_pos_[1].clear();
        for (int i = 0; i < 2; ++i) {
            if (ring_copied_[i]) cudaEventDestroy(ring_copied_[i]);
            if (ring_done_[i]) cudaEventDestroy(ring_done_[i]);
            ring_copied_[i] = ring_done_[i] = nullptr;
        }
    }

    // Tier layer `moe_l`'s experts that GPU 0 computes itself, packed into its slot on ring_stream_ once the layer
    // that last used the slot is done. Peer experts are PeerShare's business, file-only experts stage_[2]'s. When the
    // ring shrunk below the layer's local experts the RAM ones go FIRST: their slow PCIe copies then overlap the
    // attention, and the card-resident ones (a fast D2D) may fall back to staging.
    bool ring_copy(int moe_l) {
        const int slot = moe_l & 1;
        const int l = moe_l + g_.n_dense;
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        core::ExpertTier* tier = m_.shared_tier().tier.get();
        uint8_t* base = ring_ + (size_t) slot * (size_t) ring_cap_ * stride_;
        std::vector<int32_t>& pos = ring_pos_[slot];
        pos.assign((size_t) g_.n_expert, -1);
        cudaStreamWaitEvent(ring_stream_, ring_done_[slot], 0);

        int local = 0;
        for (int e = 0; e < g_.n_expert; ++e) {
            const core::ExpertTier::Source src = tier->locate(moe_l, e);
            local += (!src.peer && (src.dev || src.pinned)) ? 1 : 0;
        }
        const bool ram_first = local > ring_cap_;
        int p = 0;
        for (int pass = 0; pass < (ram_first ? 2 : 1) && p < ring_cap_; ++pass) {
            for (int e = 0; e < g_.n_expert && p < ring_cap_; ++e) {
                if (pos[(size_t) e] >= 0) continue;   // a RAM expert already packed in the first pass
                const core::ExpertTier::Source src = tier->locate(moe_l, e);
                if (src.peer || (!src.dev && !src.pinned)) continue;
                if (ram_first && pass == 0 && !(src.pinned && !src.dev)) continue;
                uint8_t* dst = base + stride_ * (size_t) p;
                if (src.pinned && !src.dev && stride_ == el.blob_bytes) {
                    // a run of experts at consecutive addresses in one RAM arena goes in a single copy
                    int n = 1;
                    while (e + n < g_.n_expert && p + n < ring_cap_) {
                        const core::ExpertTier::Source nx = tier->locate(moe_l, e + n);
                        if (pos[(size_t) (e + n)] >= 0) break;
                        if (nx.peer || !nx.pinned || nx.dev) break;
                        if ((const uint8_t*) nx.pinned - (const uint8_t*) src.pinned !=
                            (ptrdiff_t) n * (ptrdiff_t) el.blob_bytes)
                            break;
                        ++n;
                    }
                    cudaMemcpyAsync(dst, src.pinned, (size_t) n * el.blob_bytes, cudaMemcpyHostToDevice, ring_stream_);
                    for (int j = 0; j < n; ++j) pos[(size_t) (e + j)] = p + j;
                    e += n - 1;
                    p += n;
                    continue;
                }
                if (src.dev)
                    cudaMemcpyAsync(dst, src.dev, el.blob_bytes, cudaMemcpyDeviceToDevice, ring_stream_);
                else
                    cudaMemcpyAsync(dst, src.pinned, el.blob_bytes, cudaMemcpyHostToDevice, ring_stream_);
                pos[(size_t) e] = p;
                ++p;
            }
        }
        // The MMQ reads a few bytes past the last expert it launches (a matrix whose last row is not a multiple of 256
        // values is read on to the next multiple, and a stray scale there would be NaN): zero what follows this layer's
        // packed experts, so a layer that packed fewer of them than the one before never reads that one's bytes.
        if (p < ring_cap_) {
            const size_t zero_n = std::min(expert_past(), (size_t) (ring_cap_ - p) * stride_);
            cudaMemsetAsync(base + (size_t) p * stride_, 0, zero_n, ring_stream_);
        }
        cudaEventRecord(ring_copied_[slot], ring_stream_);
        return true;
    }

    // The copies issue from a thread of their own: hundreds of H2D calls can hold the host thread until the bus has
    // taken most of them, and the attention this chunk runs in parallel needs that thread to keep launching.
    // Joining both slots before launching ensures only one host thread queues onto ring_stream_ at a time, preserving
    // strict FIFO order between layer moe_l-1's ring_copied_ event and layer moe_l's copies.
    void ring_start(int moe_l) {
        if (moe_l < 0 || moe_l >= g_.n_layers - g_.n_dense) return;
        const int slot = moe_l & 1;
        ring_join();
        ring_layer_[slot] = moe_l;
        ring_ok_[slot] = false;
        const int dev = m_.device();
        ring_thr_[slot] = std::thread([this, moe_l, slot, dev] {
            cudaSetDevice(dev);
            const double t0 = now_ms();
            ring_ok_[slot] = ring_copy(moe_l);
            t_ring_ms_ += now_ms() - t0;
        });
    }

    // Layer `moe_l`'s experts are in their slot (waits for the thread that issued the copy); false when they are not.
    bool ring_ready(int moe_l) {
        const int slot = moe_l & 1;
        if (ring_layer_[slot] != moe_l) ring_start(moe_l);
        if (ring_thr_[slot].joinable()) ring_thr_[slot].join();
        return ring_ok_[slot];
    }

    // At the start of dense layer 0, kick MoE layer 0 onto ring_stream_ so it loads through layer 0's dense FFN and
    // layer 1's attention; subsequent MoE layers are kicked from moe_prompt() right after ring_ready(moe_l) joins.
    void ring_ahead(int l, int T) {
        if (!ring_ || T < kRingRows || l != 0) return;
        if (ring_layer_[0] != 0) ring_start(0);
    }

    prefill::PeerShare::Layer make_share_layer(int moe_l) const {
        const ExpertLayer& el = m_.pack().experts[(size_t) (moe_l + g_.n_dense)];
        const kernels::NativeExpertLayout lay = kernels::native_expert_layout(el.gu_type, el.d_type, g_.n_latent, g_.n_ff_exp);
        prefill::PeerShare::Layer sl;
        sl.gu_type = el.gu_type;
        sl.d_type = el.d_type;
        sl.blob_bytes = el.blob_bytes;
        sl.stride = sl.gu_stride = sl.d_stride = stride_;
        sl.down_off = lay.down_off;
        const int F = g_.n_ff_exp;
        const float cg = g_.gate_softcap, cu = g_.up_softcap;
        sl.act = [F, cg, cu](const float* gu, float* h, const prefill::PeerShare::Launch& x, void* stream) {
            prefill::mmq::swiglu(gu, h, x.rows, F, /*interleaved=*/false, stream, nullptr, nullptr, 0,
                                 prefill::mmq::SwigluForm{cg, cu});
        };
        return sl;
    }

    void ensure_oa_plan(int A) {
        if (big_.oa_cached_A == A) return;
        const int G = g_.hca_o_groups;
        std::vector<int32_t> hb((size_t) G + 1), hi((size_t) G * A);
        for (int grp = 0; grp <= G; ++grp) hb[(size_t) grp] = grp * A;
        for (int grp = 0; grp < G; ++grp) {
            for (int t = 0; t < A; ++t) hi[(size_t) grp * A + t] = t * G + grp;
        }
        cudaMemcpyAsync(big_.oa_bounds, hb.data(), hb.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(big_.oa_ids, hi.data(), hi.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
        big_.oa_cached_A = A;
    }

    void set_mmq_bounds(int tw) {
        if (last_mmq_bounds_t_ != tw) {
            const int32_t bounds[2] = {0, tw};
            cudaMemcpyAsync(d_mmq_bounds_, bounds, sizeof(bounds), cudaMemcpyHostToDevice, stream_);
            last_mmq_bounds_t_ = tw;
        }
    }

    // Quantize FP32 activations [T, n_in] into Q8_1 scratch (MMVQ layout for T <= 8, MMQ layout for T > 8).
    void quantize_q8_input(const float* x, void* xq, int n_in, int T) {
        if (T <= kMaxDecodeWindow) {
            kernels::native_quantize_q8_1(x, xq, n_in, T, stream_);
        } else {
            prefill::mmq::quantize(x, nullptr, xq, kTypeQ8_0, n_in, n_in, T, stream_);
        }
    }

    // Q8_0 matmul y[T, n_out] = xq[T, n_in] @ W[n_out, n_in]^T using pre-quantized xq.
    void matmul_q8_prequant(const DevTensor* w, const void* xq, float* y, int n_in, int n_out, int T) {
        if (T <= kMaxDecodeWindow) {
            const Q8xMat m{w->ptr, xq, y, n_out};
            q8_run(&m, 1, n_in, T);
            return;
        }
        if (loaded_mmq_.insert({kTypeQ8_0, n_out}).second) prefill::mmq::load(kTypeQ8_0, n_out);
        set_mmq_bounds(T);
        prefill::mmq::Product p;
        p.w = w->ptr;
        p.type = kTypeQ8_0;
        p.w_rows = n_out;
        p.w_cols = n_in;
        p.expert_bytes = w->bytes;
        p.n = 1;
        p.xq = xq;
        p.bounds = d_mmq_bounds_;
        p.ids = (T <= kAttRows) ? d_mmq_iota_ : big_.pair_ident;
        p.total_rows = T;
        p.max_rows = T;
        p.dst = y;
        p.ld_dst = n_out;
        mmq_ctx_.run(p, stream_);
    }

    // Q8_0 matmul y[T, n_out] = x[T, n_in] @ W[n_out, n_in]^T (MMVQ for T <= 8, full-T or sliced MMQ for T > 8).
    void matmul_q8(const DevTensor* w, const float* x, float* y, int n_in, int n_out, int T) {
        if (T <= kAttRows) {
            quantize_q8_input(x, d_xq_, n_in, T);
            matmul_q8_prequant(w, d_xq_, y, n_in, n_out, T);
            return;
        }
        if (big_.ready && T <= big_.rows && n_in <= std::max(g_.n_latent, g_.n_embd)) {
            prefill::mmq::quantize(x, nullptr, big_.xq1, kTypeQ8_0, n_in, n_in, T, stream_);
            matmul_q8_prequant(w, big_.xq1, y, n_in, n_out, T);
            return;
        }
        for (int t0 = 0; t0 < T; t0 += kAttRows) {
            const int tw = std::min(kAttRows, T - t0);
            quantize_q8_input(x + (size_t) t0 * n_in, d_xq_, n_in, tw);
            matmul_q8_prequant(w, d_xq_, y + (size_t) t0 * n_out, n_in, n_out, tw);
        }
    }

    // A decode window's (T <= 8) inputs quantized once for the matmuls that share them (mmvq8 then reads d_xq_), the
    // same bits as matmul_q8 for each.
    void quant8(const float* x, int n_in, int T) { kernels::native_quantize_q8_1(x, d_xq_, n_in, T, stream_); }
    void mmvq8(const DevTensor* w, float* y, int n_in, int n_out, int T) {
        kernels::native_q8_0_mmvq(w->ptr, d_xq_, y, n_in, n_out, T, stream_);
    }
    // A window's products of one width from the quantized inputs `xq` (the matrices' own, `xq_step` bytes apart, or
    // all the same): q8_mmvq (kernels/q8x.cu, the same bits). Called right after a kernel.
    void q8_products(std::initializer_list<std::pair<const DevTensor*, float*>> mats, const std::vector<int>& rows,
                     int n_in, int T, const void* xq = nullptr, size_t xq_step = 0) {
        Q8xMat m[kQ8xMats];
        int n = 0;
        for (const auto& [w, y] : mats) {
            m[n].w = w->ptr;
            m[n].x = (const uint8_t*) (xq ? xq : d_xq_) + (size_t) n * xq_step;
            m[n].y = y;
            m[n].n_out = rows[(size_t) n];
            ++n;
        }
        q8_run(m, n, n_in, T);
    }
    void q8_run(const Q8xMat* m, int n, int n_in, int T) { q8_mmvq(m, n, n_in, T, stream_); }

    // BF16 matmul using pre-converted BF16 activations x_bf16[T, n_in] via cuBLAS Tensor Cores (FP32 accumulate).
    void matmul_bf16_preconv(const DevTensor* w, const uint16_t* x_bf16, float* y, int n_in, int n_out, int T) {
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &alpha, w->ptr, CUDA_R_16BF, n_in,
                     x_bf16, CUDA_R_16BF, n_in, &beta, y, CUDA_R_32F, n_out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // BF16 matmul y[T, n_out] = x[T, n_in] @ W[n_out, n_in]^T (MMVF GEMV for T <= 8, cuBLAS BF16 GEMM for T > 8).
    void matmul_bf16(const DevTensor* w, const float* x, float* y, int n_in, int n_out, int T) {
        if (T <= kMaxDecodeWindow || n_in > g_.n_embd) {
            for (int t0 = 0; t0 < T; t0 += 8) {
                const int take = std::min(8, T - t0);
                kernels::bf16_gemv_fp32_mmvf_cols(x + (size_t) t0 * n_in, (const uint16_t*) w->ptr,
                                                  y + (size_t) t0 * n_out, n_in, n_out, take, stream_);
            }
            return;
        }
        for (int t0 = 0; t0 < T; t0 += kAttRows) {
            const int tw = std::min(kAttRows, T - t0);
            kernels::f32_to_bf16_bulk(x + (size_t) t0 * n_in, d_bf16_x_, (int64_t) tw * n_in, stream_);
            matmul_bf16_preconv(w, d_bf16_x_, y + (size_t) t0 * n_out, n_in, n_out, tw);
        }
    }

    // While a decode window's layer waits for its host experts the card's memory is idle: the next layer's first
    // weights (its attention's input products, in the order it reads them, up to kL2Prefetch bytes) are pulled into
    // L2 then, on a stream of its own that joins the window at its end (prefetching only: nothing waits for it).
    void prefetch_next(int l) {
        if (l + 1 >= g_.n_layers) return;
        kernels::PdlScope plain(false);   // a side stream's chain
        static constexpr const char* kHca[] = {"attn_q_a.weight", "attn_kv.weight", "attn_q_b.weight", "attn_gate.weight",
                                               "attn_output_a.weight"};
        static constexpr const char* kDsa[] = {"attn_q_a.weight", "attn_kv_a_mqa.weight", "attn_gate.weight",
                                               "attn_q_b.weight", "indexer.attn_q_b.weight"};
        const bool dsa = g_.is_dsa(l + 1);
        L2Spans sp;
        uint64_t left = kL2Prefetch;
        for (const char* name : dsa ? kDsa : kHca) {
            const DevTensor* t = m_.tensor(l + 1, name);
            if (!t || left == 0 || sp.n == kL2Spans) break;
            const uint64_t take = std::min<uint64_t>(t->bytes, left);
            sp.ptr[sp.n] = t->ptr;
            sp.bytes[sp.n++] = (uint32_t) take;
            left -= take;
        }
        cudaEventRecord(pf_fork_, stream_);
        cudaStreamWaitEvent(pf_stream_, pf_fork_, 0);
        l2_prefetch(sp, pf_stream_);
        pf_used_ = true;
    }

    // Once layer l's own work is queued, on a stream of its own (while the merge waits for the host, the card's memory
    // is idle): layer l + 1's router over layer l's router input, its top kMoeHintK experts a row to the host as a hint
    // (DecodeMoe::hint), which warms the predicted RAM experts while the cards finish layer l. The window waits for the
    // stream before it writes that input again (hint_join).
    bool hint_next(int l, const float* x_norm, int T, std::string& err) {
        if (l + 1 >= g_.n_layers || T > kMoeHintRows) return true;
        const DevTensor *rg = req_t(l + 1, "ffn_gate_inp.weight", err), *rb = req_t(l + 1, "exp_probs_b.bias", err),
                        *rs = req_t(l + 1, "exp_probs_scale.weight", err);
        if (!rg || !rb || !rs) return false;
        if (rg->type != kTypeBF16) return true;
        const int H = g_.n_embd, n = 2 * g_.n_expert;
        kernels::PdlScope plain(false);   // a side stream's chain
        cudaEventRecord(hint_fork_, stream_);
        cudaStreamWaitEvent(hint_stream_, hint_fork_, 0);
        kernels::bf16_gemv_fp32_mmvf_cols(x_norm, (const uint16_t*) rg->ptr, d_hint_logits_, H, n, T, hint_stream_);
        split_sigmoid_route(d_hint_logits_, (const float*) rb->ptr, (const float*) rs->ptr, T, g_.n_expert, kMoeHintK,
                            g_.norm_topk_prob, d_hint_ids_, d_hint_w_, hint_stream_);
        dec_->hint(l + 1 - g_.n_dense, T, d_hint_ids_, hint_stream_);
        cudaEventRecord(hint_join_, hint_stream_);
        hint_used_ = true;
        return true;
    }
    void hint_join() {
        if (!hint_used_) return;
        cudaStreamWaitEvent(stream_, hint_join_, 0);
        after_join();
        hint_used_ = false;
    }
    // The kernel after a cross-stream wait or an event node starts plainly (a programmatic edge joins two kernels).
    static void after_join() { kernels::pdl_state().plain = 1; }

    // The router's logits (BF16 weights, model.cpp): a decode window's rows by the BF16 GEMV, a prompt chunk's by SGEMM
    // over the weights widened (exactly) into the prompt's scratch.
    void router_logits(const DevTensor* rg, const float* x, float* y, int T) {
        const int H = g_.n_embd, n = 2 * g_.n_expert;
        if (rg->type != kTypeBF16) {
            matmul_f32((const float*) rg->ptr, x, y, H, n, T);
        } else if (T > kMaxDecodeWindow && big_.ready && big_.router_w) {
            bf16_widen_f32((const uint16_t*) rg->ptr, big_.router_w, (int64_t) H * n, stream_);
            matmul_f32(big_.router_w, x, y, H, n, T);
        } else {
            matmul_bf16(rg, x, y, H, n, T);
        }
    }

    // F32 matmul y[T, n_out] = x[T, n_in] @ W[n_out, n_in]^T via cuBLAS SGEMM.
    void matmul_f32(const float* w, const float* x, float* y, int n_in, int n_out, int T) {
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &alpha, w, n_in, x, n_in, &beta, y, n_out);
    }

    // ---- Fast Prompt MoE (Basalt Tensor-Core MMQ + Copy-Stream Double-Buffering + Dual-GPU PeerShare) ----

    // Pre-stage layer `moe_l`'s GPU-1 resident experts on the peer card while the main card runs attention.
    void prepare_peer(int moe_l, core::ExpertTier* tier) {
        peer_prepared_ = -1;
        if (!tier || moe_l < 0 || moe_l >= g_.n_layers - g_.n_dense) return;
        prefill::PeerShare* share = tier->share_for(moe_l);
        if (!share) return;
        std::vector<prefill::PeerShare::Expert> ex;
        for (int e = 0; e < g_.n_expert; ++e) {
            const core::ExpertTier::Source src = tier->locate(moe_l, e);
            if (!src.peer) continue;
            prefill::PeerShare::Expert x1;
            x1.blob = src.peer;
            x1.on_peer = true;
            ex.push_back(x1);
        }
        if (!ex.empty()) {
            share->prepare(make_share_layer(moe_l), ex);
            peer_prepared_ = moe_l;
        }
    }

    bool moe_prompt(int l, int moe_l, int T, const float* x_lat, const float* exp_w, float* lat_sum,
                    std::string& err) {
        namespace mmq = prefill::mmq;
        core::ExpertTier* tier = m_.shared_tier().tier.get();
        if (!tier) { err = "ernie55: expert tier not prepared"; return false; }

        const double t0 = now_ms();
        const int K = g_.n_used, L = g_.n_latent, F = g_.n_ff_exp, NE = g_.n_expert;
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        const kernels::NativeExpertLayout lay = kernels::native_expert_layout(el.gu_type, el.d_type, L, F);

        const size_t n_pairs = (size_t) T * K;
        if (!ck(cudaEventSynchronize(big_.ev_ids), "moe_prompt ids sync", err)) return false;

        // Flat CSR counting-sort bucketing (zero heap allocations per layer, identical ascending pair order)
        h_cnt_.assign((size_t) NE, 0);
        for (size_t p = 0; p < n_pairs; ++p) {
            const int32_t e = big_.h_ids[p];
            if (e < 0 || e >= NE) {
                err = "ernie55: router returned out-of-range expert " + std::to_string(e);
                return false;
            }
            ++h_cnt_[(size_t) e];
        }
        h_off_.resize((size_t) NE + 1);
        h_off_[0] = 0;
        for (int e = 0; e < NE; ++e) h_off_[(size_t) e + 1] = h_off_[(size_t) e] + h_cnt_[(size_t) e];
        h_cur_.assign(h_off_.begin(), h_off_.begin() + NE);
        if (h_pairs_.size() < n_pairs) h_pairs_.resize(n_pairs);
        for (size_t p = 0; p < n_pairs; ++p) {
            const int32_t e = big_.h_ids[p];
            h_pairs_[(size_t) h_cur_[(size_t) e]++] = (int32_t) p;
        }

        // Join this layer's prefill expert ring BEFORE the PeerShare split so experts already copied into GPU 0's
        // ring stay on GPU 0 (counted with 0 remaining bus bytes) instead of being reassigned and copied again to GPU 1.
        const bool ringed = ring_ && T >= kRingRows && ring_ready(moe_l);
        const std::vector<int32_t>* rpos_ptr = ringed ? &ring_pos_[moe_l & 1] : nullptr;

        // Plan GPU 1 (PeerShare) split: GPU 1 takes its own resident experts + balanced RAM share via peer_take.
        // Interleave heavy and light RAM experts so every prefix [0..k) has the same average row density as the
        // remaining suffix, letting peer_take balance BOTH compute rows and PCIe blob bytes 50/50 across both GPUs.
        prefill::PeerShare* share = tier->share_for(moe_l);
        const bool had_prepared = (share != nullptr && peer_prepared_ == moe_l);
        peer_prepared_ = -1;
        std::vector<uint8_t> away((size_t) NE, 0);
        if (share) {
            std::vector<int> ram_all;
            int64_t main_rows = 0, peer_rows = 0;
            double main_bytes = 0;
            for (int e = NE - 1; e >= 0; --e) {
                const int64_t nr = h_cnt_[(size_t) e];
                if (nr == 0) continue;
                const core::ExpertTier::Source src = tier->locate(moe_l, e);
                const bool in_ring = rpos_ptr && (size_t) e < rpos_ptr->size() && (*rpos_ptr)[(size_t) e] >= 0;
                if (src.peer) {
                    away[(size_t) e] = 1;
                    peer_rows += nr;
                } else if (src.pinned && !src.dev && !in_ring) {
                    ram_all.push_back(e);
                } else {
                    main_rows += nr;
                    if (!src.dev && !in_ring) main_bytes += (double) el.blob_bytes;
                }
            }
            std::sort(ram_all.begin(), ram_all.end(),
                      [&](int a, int b) { return h_cnt_[(size_t) a] > h_cnt_[(size_t) b]; });
            std::vector<int> ram;
            std::vector<int64_t> ram_rows;
            ram.reserve(ram_all.size());
            ram_rows.reserve(ram_all.size());
            size_t lo = 0, hi = ram_all.size();
            while (lo < hi) {
                const int e_hi = ram_all[lo++];
                ram.push_back(e_hi);
                ram_rows.push_back((int64_t) h_cnt_[(size_t) e_hi]);
                if (lo < hi) {
                    const int e_lo = ram_all[--hi];
                    ram.push_back(e_lo);
                    ram_rows.push_back((int64_t) h_cnt_[(size_t) e_lo]);
                }
            }
            const int take = tier->peer_take(moe_l, ram_rows, main_rows, main_bytes, peer_rows, /*tokens=*/0, 0.0);
            for (int i = 0; i < take; ++i) away[(size_t) ram[(size_t) i]] = 1;
        }

        // Order peer experts with GPU-1 resident experts FIRST (ascending id) to match PeerShare::prepare() prefix,
        // followed by peer_take RAM experts.
        std::vector<int> used, peer;
        used.reserve((size_t) NE);
        peer.reserve((size_t) NE);
        if (share) {
            for (int e = 0; e < NE; ++e) {
                const core::ExpertTier::Source src = tier->locate(moe_l, e);
                if (src.peer && (h_cnt_[(size_t) e] > 0 || had_prepared)) peer.push_back(e);
            }
            for (int e = 0; e < NE; ++e) {
                if (h_cnt_[(size_t) e] == 0) continue;
                if (away[(size_t) e]) {
                    if (!tier->locate(moe_l, e).peer) peer.push_back(e);
                } else {
                    used.push_back(e);
                }
            }
        } else {
            for (int e = 0; e < NE; ++e)
                if (h_cnt_[(size_t) e] > 0) used.push_back(e);
        }

        // Optional CPU helper ONLY for short prompts (T <= 256): offload 1..4-row RAM experts in parallel
        struct CpuWork {
            std::vector<kernels::cpu::ExpertJobMulti> jobs;
            std::vector<int32_t> pair, act_tok;
            int64_t at = 0;
            std::thread th;
            void join() { if (th.joinable()) th.join(); }
            ~CpuWork() { join(); }
        } cpu;

        if (T <= kCpuShortLimit && tier->pool()) {
            int bus = 0;
            for (int e : used) {
                const core::ExpertTier::Source src = tier->locate(moe_l, e);
                bus += (src.pinned && !src.dev);
            }
            if (bus > 2) {
                if (!cpu_rows_) {
                    if (cudaMallocHost((void**) &cpu_rows_, (size_t) kCpuMaxRows * L * sizeof(float)) != cudaSuccess ||
                        cudaMallocHost((void**) &cpu_x_, (size_t) kCpuShortLimit * L * sizeof(float)) != cudaSuccess ||
                        !(cpu_act_ = (uint8_t*) platform::alloc_aligned((size_t) kCpuShortLimit * kernels::cpu::kNativeActBytes, 64))) {
                        err = "ernie55: out of host memory for short-prompt CPU share";
                        return false;
                    }
                }
                std::vector<int> cand, cand_rows;
                int crow = 0;
                for (int c = 1; c <= kCpuMaxRowPerExp; ++c) {
                    for (int e : used) {
                        const core::ExpertTier::Source src = tier->locate(moe_l, e);
                        if (!src.pinned || src.dev || h_cnt_[(size_t) e] != c || crow + c > kCpuMaxRows) continue;
                        cand.push_back(e);
                        cand_rows.push_back(c);
                        crow += c;
                    }
                }
                int take = 0, rsum = 0;
                while (take < (int) cand.size() && rsum + cand_rows[(size_t) take] <= bus - take - 1)
                    rsum += cand_rows[(size_t) take++];
                if (take > 0) {
                    std::vector<uint8_t> taken((size_t) NE, 0);
                    std::vector<int32_t> tok_slot((size_t) T, -1);
                    int out_row = 0;
                    for (int ci = 0; ci < take; ++ci) {
                        const int e = cand[(size_t) ci], c = cand_rows[(size_t) ci];
                        const core::ExpertTier::Source src = tier->locate(moe_l, e);
                        kernels::cpu::ExpertJobMulti j;
                        j.blob = src.pinned;
                        j.nt = c;
                        const int32_t* ep = h_pairs_.data() + h_off_[(size_t) e];
                        for (int i = 0; i < c; ++i) {
                            const int32_t p = ep[i], t = p / K;
                            if (tok_slot[(size_t) t] < 0) {
                                tok_slot[(size_t) t] = (int32_t) cpu.act_tok.size();
                                cpu.act_tok.push_back(t);
                            }
                            j.nact[i] = cpu_act_ + (size_t) tok_slot[(size_t) t] * kernels::cpu::kNativeActBytes;
                            j.out[i] = cpu_rows_ + (size_t) out_row * L;
                            cpu.pair.push_back(p);
                            ++out_row;
                        }
                        cpu.jobs.push_back(j);
                        taken[(size_t) e] = 1;
                    }
                    used.erase(std::remove_if(used.begin(), used.end(), [&](int e) { return taken[(size_t) e] != 0; }),
                               used.end());
                    cudaStreamWaitEvent(copy_stream_, big_.ev_xlat, 0);
                    cudaMemcpyAsync(cpu_x_, x_lat, (size_t) T * L * sizeof(float), cudaMemcpyDeviceToHost, copy_stream_);
                    if (!ck(cudaStreamSynchronize(copy_stream_), "moe_prompt cpu_x sync", err)) return false;
                    n_prompt_cpu_ += (int64_t) cpu.jobs.size();
                    n_prompt_cpu_rows_ += (int64_t) cpu.pair.size();
                    cpu.th = std::thread([this, tier, moe_l, &cpu, L] {
                        const kernels::cpu::NativeFmt& f = tier->fmt(moe_l);
                        for (size_t i = 0; i < cpu.act_tok.size(); ++i)
                            kernels::cpu::native_quant_act(f, cpu_x_ + (size_t) cpu.act_tok[i] * L,
                                                           cpu_act_ + i * kernels::cpu::kNativeActBytes);
                        tier->pool()->run_split_multi_native(f, cpu.jobs.data(), (int) cpu.jobs.size());
                    });
                }
            }
        }

        // Ensure previous layer's async H2D of h_pair_map / h_launch_tok / h_launch_bounds finished before overwriting
        if (!ck(cudaEventSynchronize(big_.ev_map), "moe_prompt map sync", err)) return false;

        // Compute row layout in big_.pair_rows: [GPU 0 used rows | GPU 1 peer rows | CPU rows]
        int64_t main_total = 0, peer_total = 0;
        for (int e : used) main_total += h_cnt_[(size_t) e];
        for (int e : peer) peer_total += h_cnt_[(size_t) e];
        int32_t* map_h = big_.h_pair_map;

        if (!cpu.jobs.empty()) {
            cpu.at = main_total + peer_total;
            for (size_t i = 0; i < cpu.pair.size(); ++i) map_h[(size_t) cpu.pair[i]] = (int32_t) (cpu.at + (int64_t) i);
        }

        // Launch GPU 1 PeerShare asynchronously while GPU 0 runs its double-buffered MMQ groups
        if (!peer.empty()) {
            peer_tok_flat_.clear();
            peer_tok_flat_.reserve((size_t) peer_total);
            std::vector<prefill::PeerShare::Expert> ex;
            ex.reserve(peer.size());
            int64_t end = main_total;
            for (size_t i = 0; i < peer.size(); ++i) {
                const int e = peer[i];
                const core::ExpertTier::Source src = tier->locate(moe_l, e);
                const size_t tok_off = peer_tok_flat_.size();
                const int32_t cnt = h_cnt_[(size_t) e];
                const int32_t* ep = h_pairs_.data() + h_off_[(size_t) e];
                for (int32_t j = 0; j < cnt; ++j) {
                    const int32_t p = ep[j];
                    peer_tok_flat_.push_back(p / K);
                    map_h[(size_t) p] = (int32_t) end++;
                }
                prefill::PeerShare::Expert x1;
                x1.blob = src.peer ? src.peer : src.pinned;
                x1.on_peer = src.peer != nullptr;
                x1.tok = cnt > 0 ? peer_tok_flat_.data() + tok_off : nullptr;
                x1.n = cnt;
                ex.push_back(x1);
            }
            if (!share->run(make_share_layer(moe_l), x_lat, T, ex, big_.pair_rows + (size_t) main_total * L, stream_, err))
                return false;
            n_prompt_peer_ += (int64_t) peer.size();
            n_prompt_peer_rows_ += peer_total;
        }

        // Quantize x_lat ONCE for all T rows on stream_, then gather per GPU-0 launch via mmq::gather_q8
        if (!used.empty()) mmq::quantize(x_lat, nullptr, big_.xq_lat, el.gu_type, L, L, T, stream_);

        // The pre-packed launch tables and the double-buffered stage alternate from counters the ring's direct
        // launches and the staged overflow experts share, so the two paths continue seamlessly into each other.
        int32_t tok_cur = 0, bnd_cur = 0;
        int64_t pos_cur = 0;
        int launch_idx = 0;

        // A prompt chunk of >= kRingRows rows routes to (almost) every expert: those whose blobs the ring already
        // packed on ring_stream_ (while earlier layers computed) launch their MMQ straight from the ring slot; the
        // overflow experts keep the copy_stream_ staging below. Bit-identical either way: the same blob bytes at the
        // same stride, the same tok/bounds row layout (a gap between packed slots runs as empty bounds entries), the
        // same row mapping and weighted sum.
        if (ringed && moe_l + 1 < g_.n_layers - g_.n_dense && ring_layer_[(moe_l + 1) & 1] != moe_l + 1)
            ring_start(moe_l + 1);   // the next layer's copies queue on ring_stream_ while this layer's MMQ runs
        if (ringed) {
            const int slot = moe_l & 1;
            cudaStreamWaitEvent(stream_, ring_copied_[slot], 0);
            const uint8_t* rbase = ring_ + (size_t) slot * (size_t) ring_cap_ * stride_;
            const std::vector<int32_t>& rpos = ring_pos_[slot];
            std::vector<int> in_ring, rest;
            in_ring.reserve(used.size());
            rest.reserve(used.size());
            for (int e : used) {
                if ((size_t) e < rpos.size() && rpos[(size_t) e] >= 0) in_ring.push_back(e);
                else rest.push_back(e);
            }
            used = std::move(rest);
            n_prompt_ring_ += (int64_t) in_ring.size();
            // ascending packed slot: every launch walks the ring from a low position to strictly higher ones
            std::sort(in_ring.begin(), in_ring.end(),
                      [&](int a, int b) { return rpos[(size_t) a] < rpos[(size_t) b]; });

            // Flatten the ringed experts into slices of <= kLaunchRows just like the staged path below
            struct RingSlice {
                int e;
                int32_t off;
                int32_t n;
            };
            std::vector<RingSlice> rslices;
            rslices.reserve(in_ring.size());
            for (int e : in_ring) {
                const int32_t total_c = h_cnt_[(size_t) e];
                for (int32_t off = 0; off < total_c; off += kLaunchRows)
                    rslices.push_back({e, off, std::min<int32_t>(kLaunchRows, total_c - off)});
            }
            for (size_t rb0 = 0; rb0 < rslices.size();) {
                const int p0 = rpos[(size_t) rslices[rb0].e];
                size_t rb1 = rb0;
                int64_t rows = 0, max_rows = 0;
                while (rb1 < rslices.size() && rpos[(size_t) rslices[rb1].e] < p0 + kStageExperts) {
                    if (rb1 > rb0 && rslices[rb1].e == rslices[rb1 - 1].e) break;
                    const int64_t c = rslices[rb1].n;
                    if (rb1 > rb0 && rows + c > kLaunchRows) break;
                    rows += c;
                    max_rows = std::max(max_rows, c);
                    ++rb1;
                }
                const int nb = rpos[(size_t) rslices[rb1 - 1].e] - p0 + 1;   // packed slots, gaps included
                const int32_t tok_off = tok_cur, bnd_off = bnd_cur;
                int32_t ntok = 0;
                if (!bounds_fit(big_, bnd_cur, nb, err)) return false;   // the launch writes 1 + nb entries here
                big_.h_launch_bounds[bnd_cur++] = 0;
                size_t cur = rb0;
                for (int q = 0; q < nb; ++q) {
                    if (cur < rb1 && rpos[(size_t) rslices[cur].e] == p0 + q) {   // a gap expert keeps empty bounds
                        const RingSlice& sl = rslices[cur];
                        const int32_t* ep = h_pairs_.data() + h_off_[(size_t) sl.e] + sl.off;
                        for (int32_t j = 0; j < sl.n; ++j) {
                            const int32_t p = ep[j];
                            map_h[(size_t) p] = (int32_t) (pos_cur + ntok);
                            big_.h_launch_tok[tok_cur++] = p / K;
                            ++ntok;
                        }
                        ++cur;
                    }
                    big_.h_launch_bounds[bnd_cur++] = ntok;
                }

                // The tok/bounds tables ride stream_ (which owns them); the blobs are already on the card, so this
                // launch reads them straight from its ring slot instead of a stage_[2] blob buffer.
                MmqStage& B = big_.stage[launch_idx++ & 1];
                cudaMemcpyAsync(B.tok, big_.h_launch_tok + tok_off, (size_t) rows * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream_);
                cudaMemcpyAsync(B.bounds, big_.h_launch_bounds + bnd_off, (size_t) (nb + 1) * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream_);
                mmq::gather_q8(big_.xq_lat, T, B.tok, big_.xq1, rows, L, stream_);
                const uint8_t* w = rbase + (size_t) p0 * stride_;
                mmq::Product p;
                p.w = w;
                p.type = el.gu_type;
                p.w_rows = 2 * (int64_t) F;
                p.w_cols = L;
                p.expert_bytes = stride_;
                p.n = nb;
                p.xq = big_.xq1;
                p.bounds = B.bounds;
                p.ids = big_.pair_ident;
                p.total_rows = rows;
                p.max_rows = max_rows;
                p.dst = big_.gu;
                p.ld_dst = p.w_rows;
                mmq_ctx_.run(p, stream_);

                mmq::swiglu(big_.gu, big_.h, rows, F, /*interleaved=*/false, stream_, nullptr, nullptr, 0,
                            mmq::SwigluForm{g_.gate_softcap, g_.up_softcap});

                mmq::quantize(big_.h, nullptr, big_.xq2, el.d_type, F, F, rows, stream_);
                mmq::Product d;
                d.w = w + lay.down_off;
                d.type = el.d_type;
                d.w_rows = L;
                d.w_cols = F;
                d.expert_bytes = stride_;
                d.n = nb;
                d.xq = big_.xq2;
                d.bounds = B.bounds;
                d.ids = big_.pair_ident + pos_cur;
                d.total_rows = rows;
                d.max_rows = max_rows;
                d.dst = big_.pair_rows;
                d.ld_dst = L;
                mmq_ctx_.run(d, stream_);
                cudaEventRecord(B.done, stream_);   // the overflow staging may recycle B only once its MMQ is done
                pos_cur += rows;
                rb0 = rb1;
            }
            cudaEventRecord(ring_done_[slot], stream_);   // the slot's ring bytes are read: it may be refilled
        }

        // Flatten GPU-0 experts into slices of <= kLaunchRows so even a single super-hot expert at R=16384 never overruns
        struct UsedSlice {
            int e;
            int32_t off;
            int32_t n;
        };
        std::vector<UsedSlice> slices;
        slices.reserve(used.size());
        for (int e : used) {
            const int32_t total_c = h_cnt_[(size_t) e];
            for (int32_t off = 0; off < total_c; off += kLaunchRows) {
                slices.push_back({e, off, std::min<int32_t>(kLaunchRows, total_c - off)});
            }
        }

        // Pre-pack all GPU-0 launch tok/bounds into pinned host tables once so the staging loop needs ZERO host syncs
        struct LaunchPlan {
            size_t b0;
            int nb;
            int64_t rows, max_rows, pos;
            int32_t tok_off, bnd_off;
        };
        std::vector<LaunchPlan> plans;
        plans.reserve((slices.size() + kStageExperts - 1) / kStageExperts + 4);
        for (size_t b0 = 0; b0 < slices.size();) {
            int nb = 0;
            int64_t rows = 0, max_rows = 0;
            while (b0 + (size_t) nb < slices.size() && nb < kStageExperts) {
                const int64_t c = slices[b0 + (size_t) nb].n;
                if (nb > 0 && rows + c > kLaunchRows) break;
                rows += c;
                max_rows = std::max(max_rows, c);
                ++nb;
            }
            const int32_t tok_off = tok_cur, bnd_off = bnd_cur;
            int32_t ntok = 0;
            if (!bounds_fit(big_, bnd_cur, nb, err)) return false;   // the launch writes 1 + nb entries here
            big_.h_launch_bounds[bnd_cur++] = 0;
            for (int q = 0; q < nb; ++q) {
                const UsedSlice& sl = slices[b0 + (size_t) q];
                const int32_t* ep = h_pairs_.data() + h_off_[(size_t) sl.e] + sl.off;
                for (int32_t j = 0; j < sl.n; ++j) {
                    const int32_t p = ep[j];
                    map_h[(size_t) p] = (int32_t) (pos_cur + ntok);
                    big_.h_launch_tok[tok_cur++] = p / K;
                    ++ntok;
                }
                big_.h_launch_bounds[bnd_cur++] = ntok;
            }
            plans.push_back({b0, nb, rows, max_rows, pos_cur, tok_off, bnd_off});
            pos_cur += rows;
            b0 += (size_t) nb;
        }

        // GPU 0 double-buffered staging on copy_stream_ overlapped with Blackwell Tensor-Core MMQ on stream_ (async, 0 host syncs)
        for (const LaunchPlan& lp : plans) {
            MmqStage& B = big_.stage[launch_idx++ & 1];
            cudaStreamWaitEvent(copy_stream_, B.done, 0);
            for (int q = 0; q < lp.nb; ++q) {
                const int e = slices[lp.b0 + (size_t) q].e;
                uint8_t* dst = B.blobs + (size_t) q * stride_;
                const core::ExpertTier::Source src = tier->locate(moe_l, e);
                if (src.dev) {
                    cudaMemcpyAsync(dst, src.dev, el.blob_bytes, cudaMemcpyDeviceToDevice, copy_stream_);
                    ++n_prompt_dev_;
                } else if (src.pinned) {
                    cudaMemcpyAsync(dst, src.pinned, el.blob_bytes, cudaMemcpyHostToDevice, copy_stream_);
                    ++n_prompt_bus_;
                } else if (src.peer) {
                    core::PeerExperts* pe = tier->peer_for(moe_l);
                    cudaMemcpyPeerAsync(dst, m_.device(), src.peer, pe->device(), el.blob_bytes, copy_stream_);
                } else if (!stage_from_file(el, e, q, dst, err)) {   // only the file holds it (a spilled pack)
                    return false;
                }
            }
            cudaEventRecord(B.copied, copy_stream_);

            cudaStreamWaitEvent(stream_, B.copied, 0);
            cudaMemcpyAsync(B.tok, big_.h_launch_tok + lp.tok_off, (size_t) lp.rows * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream_);
            cudaMemcpyAsync(B.bounds, big_.h_launch_bounds + lp.bnd_off, (size_t) (lp.nb + 1) * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream_);

            mmq::gather_q8(big_.xq_lat, T, B.tok, big_.xq1, lp.rows, L, stream_);
            mmq::Product p;
            p.w = B.blobs;
            p.type = el.gu_type;
            p.w_rows = 2 * (int64_t) F;
            p.w_cols = L;
            p.expert_bytes = stride_;
            p.n = lp.nb;
            p.xq = big_.xq1;
            p.bounds = B.bounds;
            p.ids = big_.pair_ident;
            p.total_rows = lp.rows;
            p.max_rows = lp.max_rows;
            p.dst = big_.gu;
            p.ld_dst = p.w_rows;
            mmq_ctx_.run(p, stream_);

            mmq::swiglu(big_.gu, big_.h, lp.rows, F, /*interleaved=*/false, stream_, nullptr, nullptr, 0,
                        mmq::SwigluForm{g_.gate_softcap, g_.up_softcap});

            mmq::quantize(big_.h, nullptr, big_.xq2, el.d_type, F, F, lp.rows, stream_);
            mmq::Product d;
            d.w = B.blobs + lay.down_off;
            d.type = el.d_type;
            d.w_rows = L;
            d.w_cols = F;
            d.expert_bytes = stride_;
            d.n = lp.nb;
            d.xq = big_.xq2;
            d.bounds = B.bounds;
            d.ids = big_.pair_ident + lp.pos;
            d.total_rows = lp.rows;
            d.max_rows = lp.max_rows;
            d.dst = big_.pair_rows;
            d.ld_dst = L;
            mmq_ctx_.run(d, stream_);
            cudaEventRecord(B.done, stream_);
        }

        const double tw0 = now_ms();
        if (!peer.empty() && !share->join(stream_, err)) return false;
        t_prompt_peer_wait_ms_ += now_ms() - tw0;
        if (T >= kPeerPrepareRows && l + 1 < g_.n_layers) prepare_peer(moe_l + 1, tier);

        if (!cpu.jobs.empty()) {
            cpu.join();
            cudaMemcpyAsync(big_.pair_rows + (size_t) cpu.at * L, cpu_rows_,
                            cpu.pair.size() * (size_t) L * sizeof(float), cudaMemcpyHostToDevice, stream_);
        }

        cudaMemcpyAsync(big_.pair_map, map_h, n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
        cudaEventRecord(big_.ev_map, stream_);
        weighted_sum_mapped_f32(big_.pair_rows, big_.pair_map, exp_w, T, K, L, lat_sum, stream_);
        t_prompt_moe_ms_ += now_ms() - t0;
        return true;
    }

    // Reads expert e of layer `el` from the pack into bounce slot q and queues its copy into `dst` on the copy stream; the
    // slot's previous copy has left it first.
    bool stage_from_file(const ExpertLayer& el, int e, int q, uint8_t* dst, std::string& err) {
        if (!file_bounce_ &&
            !ck(cudaMallocHost((void**) &file_bounce_, (size_t) kStageExperts * stride_), "file bounce", err))
            return false;
        uint8_t* b = file_bounce_ + (size_t) q * stride_;
        if (!ck(cudaStreamSynchronize(copy_stream_), "file bounce sync", err)) return false;
        const uint64_t off = m_.pack().experts_range.offset + el.offset + (uint64_t) e * el.blob_bytes;
        if (!platform::pread_all(m_.fd(), b, el.blob_bytes, off)) {
            err = "ernie55: cannot read expert " + std::to_string(e) + " from the pack";
            return false;
        }
        cudaMemcpyAsync(dst, b, el.blob_bytes, cudaMemcpyHostToDevice, copy_stream_);
        ++n_prompt_file_;
        return true;
    }

    // ---- Decode MoE (1..8 rows via core::ExpertTier LRU + PeerExperts + CPU pool) ----

    bool run_moe_experts_decode(int moe_l, int T, const float* x_lat, const int32_t* exp_ids, std::string& err) {
        auto& st = m_.shared_tier();
        core::ExpertTier* tier = st.tier.get();
        if (!tier) { err = "ernie55: expert tier not prepared"; return false; }

        const int K = g_.n_used;
        const int Ldim = g_.n_latent;
        const auto L = [&]() {
            auto lay = kernels::native_expert_layout(kTypeMXFP4, kTypeMXFP4, Ldim, g_.n_ff_exp, 0.0f);
            lay.gate_softcap = g_.gate_softcap;
            lay.up_softcap = g_.up_softcap;
            return lay;
        }();

        for (int t0 = 0; t0 < T; t0 += kMaxDecodeWindow) {
            const int tw = std::min(kMaxDecodeWindow, T - t0);
            const int entries = tw * K;
            const float* x_lat_w = x_lat + (size_t) t0 * (size_t) Ldim;
            const int32_t* ids_w = exp_ids + (size_t) t0 * (size_t) K;
            float* exp_out_w = d_exp_out_ + (size_t) t0 * (size_t) K * (size_t) Ldim;

            cudaMemcpyAsync(h_ids_.data(), ids_w, (size_t) entries * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
            cudaMemcpyAsync(st.h_x, x_lat_w, (size_t) tw * (size_t) Ldim * sizeof(float), cudaMemcpyDeviceToHost,
                            stream_);
            if (!ck(cudaStreamSynchronize(stream_), "moe sync", err)) return false;

            tier->begin_token(tw);
            std::vector<int32_t> uniq;
            std::vector<core::ExpertTier::Use> uses;
            uniq.reserve((size_t) entries);
            uses.reserve((size_t) entries);
            for (int idx = 0; idx < entries; ++idx) {
                const int32_t eid = h_ids_[(size_t) idx];
                if (eid < 0) continue;   // dropped by --route-tail-skip (weight is 0.0)
                const int tok = idx / K;
                int found = -1;
                for (int u = 0; u < (int) uniq.size(); ++u) {
                    if (uniq[(size_t) u] == eid) { found = u; break; }
                }
                if (found < 0) {
                    found = (int) uniq.size();
                    uniq.push_back(eid);
                    uses.push_back(core::ExpertTier::Use{});
                }
                auto& u = uses[(size_t) found];
                u.tok[u.nt] = tok;
                u.row[u.nt] = idx;
                ++u.nt;
            }

            kernels::native_quantize_q8_1(x_lat_w, d_xq_, Ldim, tw, stream_);

            std::vector<int> cpu_rows;
            cpu_rows.reserve((size_t) entries);

            int offset = 0;
            core::PeerExperts* peer = tier->peer_for(moe_l);
            while (offset < (int) uniq.size()) {
                const int batch = std::min((int) uniq.size() - offset, core::ExpertTier::kMaxK);
                if (peer && !peer->arm(L, st.h_x, tw, K, 8, err)) return false;
                core::ExpertTier::Lookup lk;
                if (!tier->lookup(moe_l, uniq.data() + offset, batch, lk, err)) return false;
                tier->fetch(moe_l, lk);

                if (peer) {
                    std::vector<int32_t> p_ids((size_t) entries, 0), p_kind((size_t) entries, 0);
                    for (int i = 0; i < batch; ++i) {
                        if (!lk.peer[i]) continue;
                        const int kind_v = lk.fetched[i] ? 3 : 2;
                        const auto& u = uses[(size_t) (offset + i)];
                        for (int j = 0; j < u.nt; ++j) {
                            p_ids[(size_t) u.row[j]] = uniq[(size_t) (offset + i)];
                            p_kind[(size_t) u.row[j]] = kind_v;
                            cpu_rows.push_back(u.row[j]);
                        }
                    }
                    if (!peer->raise(moe_l, p_ids.data(), p_kind.data(), err)) return false;
                }

                bool any_cpu = false;
                for (int i = 0; i < batch; ++i) {
                    if (!lk.dev[i] && !lk.peer[i] && !lk.dropped[i] && !lk.reading[i]) {
                        any_cpu = true;
                        const auto& u = uses[(size_t) (offset + i)];
                        for (int j = 0; j < u.nt; ++j) cpu_rows.push_back(u.row[j]);
                    }
                }
                if (any_cpu) tier->start_cpu_window(moe_l, st.h_x, lk, uses.data() + offset, st.h_rows, false);

                int n_gpu_grp = 0, n_gpu_ent = 0;
                for (int i = 0; i < batch; ++i) {
                    if (!lk.dev[i]) continue;
                    h_grp_ptr_[(size_t) n_gpu_grp] = (unsigned long long) lk.dev[i];
                    h_grp_start_[(size_t) n_gpu_grp] = n_gpu_ent;
                    const auto& u = uses[(size_t) (offset + i)];
                    for (int j = 0; j < u.nt; ++j) {
                        h_ent_dst_[(size_t) n_gpu_ent] = u.row[j];
                        h_ent_tok_[(size_t) n_gpu_ent] = u.tok[j];
                        ++n_gpu_ent;
                    }
                    ++n_gpu_grp;
                }
                h_grp_start_[(size_t) n_gpu_grp] = n_gpu_ent;
                if (n_gpu_grp > 0) {
                    tier->await_fetch(stream_);
                    cudaMemcpyAsync(d_grp_ptr_, h_grp_ptr_.data(), (size_t) n_gpu_grp * sizeof(unsigned long long),
                                    cudaMemcpyHostToDevice, stream_);
                    cudaMemcpyAsync(d_grp_start_, h_grp_start_.data(), (size_t) (n_gpu_grp + 1) * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream_);
                    cudaMemcpyAsync(d_n_groups_, &n_gpu_grp, sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
                    cudaMemcpyAsync(d_ent_dst_, h_ent_dst_.data(), (size_t) n_gpu_ent * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream_);
                    cudaMemcpyAsync(d_ent_tok_, h_ent_tok_.data(), (size_t) n_gpu_ent * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream_);
                    kernels::native_expert_grouped(L, d_grp_ptr_, d_grp_start_, d_n_groups_, d_ent_dst_, d_ent_tok_,
                                                   n_gpu_grp, n_gpu_ent, d_xq_, d_exp_scratch_, exp_out_w, stream_);
                }
                if (peer && !peer->run_fetched(err)) return false;

                if (any_cpu) tier->finish_cpu();
                if (peer) {
                    if (!peer->finish(st.h_rows, err)) return false;
                    cudaSetDevice(m_.device());
                }

                bool any_read = false;
                for (int i = 0; i < batch; ++i) {
                    if (lk.reading[i]) {
                        any_read = true;
                        const auto& u = uses[(size_t) (offset + i)];
                        for (int j = 0; j < u.nt; ++j) cpu_rows.push_back(u.row[j]);
                    }
                }
                if (any_read) {
                    if (!tier->await_reads(err)) return false;
                    tier->start_cpu_window(moe_l, st.h_x, lk, uses.data() + offset, st.h_rows, true);
                    tier->finish_cpu();
                }

                tier->promote(moe_l, lk);

                for (int row : cpu_rows) {
                    cudaMemcpyAsync(exp_out_w + (size_t) row * Ldim, st.h_rows + (size_t) row * Ldim,
                                    (size_t) Ldim * sizeof(float), cudaMemcpyHostToDevice, stream_);
                }
                cpu_rows.clear();
                offset += batch;
            }
        }
        return true;
    }

    // A decode window's positions and embedding rows into the row buffers (pinned staging), outside its graph.
    bool stage_rows(const int32_t* tokens, int64_t pos0, int T, std::string& err) {
        const int H = g_.n_embd;
        if (!h_stage_pos_ && (!ck(cudaMallocHost(&h_stage_pos_, kMaxDecodeWindow * sizeof(int32_t)), "stage pos", err) ||
                              !ck(cudaMallocHost(&h_stage_emb_, (size_t) kMaxDecodeWindow * H * sizeof(float)), "stage emb", err)))
            return false;
        for (int t = 0; t < T; ++t) {
            h_stage_pos_[t] = (int32_t) (pos0 + t);
            if (!m_.embedding_row(tokens[t], h_stage_emb_ + (size_t) t * H)) {
                err = "ernie55: token id out of vocabulary range";
                return false;
            }
        }
        cudaMemcpyAsync(d_pos_, h_stage_pos_, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_x_in_, h_stage_emb_, (size_t) T * H * sizeof(float), cudaMemcpyHostToDevice, stream_);
        return true;
    }

    // A decode window of T rows as one CUDA graph: every launch of forward_rows reads its positions on the card and
    // the expert tables by address, so the graph captured the first time a window of T rows runs serves every later
    // one (a window was thousands of launches, and the host's launch rate set the pace).
    bool replay_window(const int32_t* tokens, int64_t pos0, int T, std::string& err) {
        cudaGraphExec_t& ex = graph_[(size_t) T - 1];
        if (!ex) {
            if (!ck(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed), "graph capture", err)) return false;
            bool ok = false;
            {   // the window's kernels start as programmatic dependents of the kernel before them (pdl.hpp)
                kernels::PdlScope pdl(true, /*first_plain=*/true);
                capturing_ = true;
                ok = forward_rows(tokens, pos0, T, /*want_logits=*/true, /*use_prompt_moe=*/false, err);
                capturing_ = false;
            }
            cudaGraph_t g = nullptr;
            const cudaError_t ce = cudaStreamEndCapture(stream_, &g);
            if (!ok) {
                if (g) cudaGraphDestroy(g);
                return false;
            }
            const cudaError_t ci = ce == cudaSuccess ? cudaGraphInstantiateWithFlags(&ex, g, 0) : ce;
            if (g) cudaGraphDestroy(g);
            if (ci != cudaSuccess) {
                ex = nullptr;
                err = std::string("ernie55: decode window graph: ") + cudaGetErrorString(ci);
                return false;
            }
        }
        return ck(cudaGraphLaunch(ex, stream_), "decode window graph launch", err);
    }

    bool forward_rows(const int32_t* tokens, int64_t pos0, int T, bool want_logits, bool use_prompt_moe,
                      std::string& err) {
        cudaSetDevice(m_.device());
        const int H = g_.n_embd;
        const bool fp8 = m_.cache_fp8();

        // Select row buffers (temporary PromptBufs when T > kAttRows, else permanent kAttRows buffers)
        const bool big = big_.ready && big_.own_row_bufs && T > kAttRows;
        int32_t* p_pos = big ? big_.pos : d_pos_;
        float* p_res_a = big ? big_.res_a : d_res_a_;
        float* p_res_b = big ? big_.res_b : d_res_b_;
        float* p_x_in = big ? big_.x_in : d_x_in_;
        float* p_x_norm = big ? big_.x_norm : d_x_norm_;
        float* p_sub_out = big ? big_.sub_out : d_sub_out_;
        float* p_mhc_ws = big ? big_.mhc_ws : d_mhc_ws_;
        float* p_mhc_coef = big ? big_.mhc_coef : d_mhc_coef_;
        float* p_ff_g = big ? big_.ff_g : d_ff_g_;
        float* p_ff_u = big ? big_.ff_u : d_ff_u_;
        float* p_ff_h = big ? big_.ff_h : d_ff_h_;
        float* p_rlogits = big ? big_.rlogits : d_rlogits_;
        int32_t* p_exp_ids = big ? big_.exp_ids : d_exp_ids_;
        float* p_exp_w = big ? big_.exp_w : d_exp_w_;
        float* p_x_lat = big ? big_.x_lat : d_x_lat_;
        float* p_lat_sum = big ? big_.lat_sum : d_lat_sum_;
        float* p_lat_norm = big ? big_.lat_norm : d_lat_norm_;
        float* p_routed_out = big ? big_.routed_out : d_routed_out_;

        if (!staged_) {   // a decode window stages its rows before its graph (stage_rows)
            if ((int) h_pos_.size() < T) h_pos_.resize(T);
            if ((int) h_emb_.size() < T * H) h_emb_.resize((size_t) T * H);
            for (int t = 0; t < T; ++t) {
                h_pos_[(size_t) t] = (int32_t) (pos0 + t);
                if (!m_.embedding_row(tokens[t], h_emb_.data() + (size_t) t * H)) {
                    err = "ernie55: token id out of vocabulary range";
                    return false;
                }
            }
            cudaMemcpyAsync(p_pos, h_pos_.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
            cudaMemcpyAsync(p_x_in, h_emb_.data(), (size_t) T * (size_t) H * sizeof(float), cudaMemcpyHostToDevice,
                            stream_);
        }
        broadcast_streams(p_x_in, T, H, p_res_a, stream_);

        core::ExpertTier* tier_ptr = m_.shared_tier().tier.get();
        const bool lossy_route = want_logits && !use_prompt_moe && tier_ptr && tier_ptr->route_lossy();
        if (lossy_route && !staged_) {
            tier_ptr->publish_held(stream_);
            after_join();
        }

        float* res = p_res_a;
        float* res_next = p_res_b;

        peer_prepared_ = -1;
        if (use_prompt_moe && ring_ && T >= kRingRows) {   // a ring outlives one chunk: start the layer slots fresh
            ring_join();
            ring_layer_[0] = ring_layer_[1] = -1;
        }
        for (int l = 0; l < g_.n_layers; ++l) {
            if (use_prompt_moe) {
                // Keep Basalt's 60s serve stall watchdog informed during large prompt chunks (e.g. 8K tokens @ 130K+ ctx)
                core::progress_beat();
                core::progress_tick();
                core::progress_at("ernie55 prompt chunk: layer", l, pos0);
                if (l == g_.n_dense && T >= kPeerPrepareRows) prepare_peer(0, tier_ptr);
                // MoE layer 0's experts ride the ring stream through layer 0's attention and dense FFN; later layers
                // through the attention before them
                ring_ahead(l, T);
            }
            // ---- Attention sublayer (mHC on full T; attention itself sliced to <= kAttRows = 128) ----
            const DevTensor *hc_fn = req_t(l, "hc_attn_fn.weight", err),
                            *hc_sc = req_t(l, "hc_attn_scale.weight", err),
                            *hc_bs = req_t(l, "hc_attn_base.weight", err),
                            *anorm = req_t(l, "attn_norm.weight", err),
                            *vha_m = req_t(l, "attn_vha_mix.weight", err),
                            *agate = req_t(l, "attn_gate.weight", err),
                            *asink = req_t(l, "attn_sinks.weight", err);
            if (!hc_fn || !hc_sc || !hc_bs || !anorm || !vha_m || !agate || !asink) return false;

            // a window (T <= 8) takes the fused launches (kernels.hpp: the same bits as the sequences they replace)
            const bool fuse = T <= kMaxDecodeWindow;
            if (fuse) {   // x_norm and its Q8_1 in d_xq_
                hint_join();
                mhc_pre_norm(res, (const float*) hc_fn->ptr, (const float*) hc_sc->ptr, (const float*) hc_bs->ptr, T, H,
                             g_.rms_eps, g_.hc_eps, g_.sinkhorn_iters, (const float*) anorm->ptr, g_.rms_eps, p_mhc_ws,
                             d_mhc_cnt_, p_mhc_coef, p_x_norm, d_xq_, stream_);
            } else {
                mhc_mixes(res, (const float*) hc_fn->ptr, (const float*) hc_sc->ptr, (const float*) hc_bs->ptr, T, H,
                          g_.rms_eps, g_.hc_eps, g_.sinkhorn_iters, p_mhc_ws, p_mhc_coef, stream_);
                mhc_pre(res, p_mhc_coef, kHcCoef, T, H, p_x_in, stream_);
                hint_join();
                kernels::rms_norm_rows(p_x_in, (const float*) anorm->ptr, T, H, g_.rms_eps, p_x_norm, stream_);
            }

            if (!g_.is_dsa(l)) {
                const DevTensor *qa = req_t(l, "attn_q_a.weight", err), *qan = req_t(l, "attn_q_a_norm.weight", err),
                                *qb = req_t(l, "attn_q_b.weight", err), *wkv = req_t(l, "attn_kv.weight", err),
                                *kvn = req_t(l, "attn_kv_a_norm.weight", err), *oa = req_t(l, "attn_output_a.weight", err),
                                *ob = req_t(l, "attn_output_b.weight", err), *cape = req_t(l, "attn_compressor_ape.weight", err),
                                *ckv = req_t(l, "attn_compressor_kv.weight", err), *cg = req_t(l, "attn_compressor_gate.weight", err),
                                *cn = req_t(l, "attn_compressor_norm.weight", err);
                if (!qa || !qan || !qb || !wkv || !kvn || !oa || !ob || !cape || !ckv || !cg || !cn) return false;

                auto& hs = hca_[(size_t) l];
                const int ratio = std::max(1, g_.compress_ratios[(size_t) l]);
                const int hpg = g_.hca_heads / g_.hca_o_groups;
                const int gin = hpg * g_.hca_head_dim;
                const int gout = g_.hca_o_lora;
                const int gate_w = g_.hca_o_groups * g_.hca_o_lora;
                const size_t oa_grp_bytes = (size_t) gout * (size_t) (gin / 32) * 34;

                const bool hca_full_t = use_prompt_moe && big &&
                                        2 * gate_w <= g_.n_used * g_.n_latent &&
                                        g_.hca_q_lora <= std::max(g_.n_ff_exp, 2 * g_.n_expert) &&
                                        g_.hca_head_dim <= std::max(g_.n_ff_exp, g_.n_latent) &&
                                        2 * g_.hca_head_dim <= H;
                if (hca_full_t) {
                    // Full-T HCA prefill: run qa, wkv, ckv, cg, parallel hca_compress, agate, and ob over full chunk
                    // (reusing idle MoE PromptBufs: ff_g, ff_u, x_in, pair_rows) instead of 64x 128-row weight re-reads!
                    float* p_qr_full = big_.ff_g;
                    float* p_kv_full = big_.ff_u;
                    float* p_ckv_full = big_.x_in;
                    float* p_cg_full = big_.x_in + (size_t) T * g_.hca_head_dim;
                    uint16_t* p_bf16_full = reinterpret_cast<uint16_t*>(big_.pair_rows);

                    prefill::mmq::quantize(p_x_norm, nullptr, big_.xq1, kTypeQ8_0, H, H, T, stream_);
                    matmul_q8_prequant(qa, big_.xq1, p_qr_full, H, g_.hca_q_lora, T);
                    matmul_q8_prequant(wkv, big_.xq1, p_kv_full, H, g_.hca_head_dim, T);
                    kernels::rms_norm_rows(p_qr_full, (const float*) qan->ptr, T, g_.hca_q_lora, g_.rms_eps, p_qr_full,
                                           stream_);
                    hca_kv_norm_rope(p_kv_full, (const float*) kvn->ptr, p_pos, T, g_.hca_head_dim, g_.rms_eps,
                                     hca_rope_, p_kv_full, stream_);

                    kernels::f32_to_bf16_bulk(p_x_norm, p_bf16_full, (int64_t) T * H, stream_);
                    matmul_bf16_preconv(ckv, p_bf16_full, p_ckv_full, H, g_.hca_head_dim, T);
                    matmul_bf16_preconv(cg, p_bf16_full, p_cg_full, H, g_.hca_head_dim, T);
                    hca_compress(p_ckv_full, p_cg_full, (const float*) cape->ptr, (const float*) cn->ptr, p_pos, T,
                                 ratio, g_.hca_head_dim, g_.rms_eps, hca_rope_, hs.state_kv, hs.state_score,
                                 hs.comp_cache, fp8, stream_, pos0);

                    float* p_gate_full = big_.pair_rows;
                    float* p_z_full = big_.pair_rows + (size_t) T * gate_w;
                    prefill::mmq::quantize(p_qr_full, nullptr, big_.xq2, kTypeQ8_0, g_.hca_q_lora, g_.hca_q_lora, T,
                                           stream_);
                    matmul_q8_prequant(agate, big_.xq2, p_gate_full, g_.hca_q_lora, gate_w, T);

                    for (int a0 = 0; a0 < T; a0 += kAttRows) {
                        const int A = std::min(kAttRows, T - a0);
                        const int32_t* pos_s = p_pos + a0;
                        const float* qr_s = p_qr_full + (size_t) a0 * g_.hca_q_lora;
                        const float* kv_s = p_kv_full + (size_t) a0 * g_.hca_head_dim;
                        float* z_s = p_z_full + (size_t) a0 * gate_w;

                        quantize_q8_input(qr_s, d_xq_, g_.hca_q_lora, A);
                        matmul_q8_prequant(qb, d_xq_, d_q_, g_.hca_q_lora, g_.hca_heads * g_.hca_head_dim, A);
                        hca_q_rope(d_q_, pos_s, A, g_.hca_heads, g_.hca_head_dim, g_.rms_eps, hca_rope_, stream_);
                        hca_attention(d_q_, hs.comp_cache, hs.win_cache, kv_s, (const float*) asink->ptr, pos_s, A,
                                      g_.hca_heads, g_.hca_head_dim, g_.window, ratio, fp8, d_attn_o_, stream_,
                                      win_cap_);
                        hca_win_commit(kv_s, pos_s, A, g_.hca_head_dim, g_.window, hs.win_cache, fp8, stream_,
                                       win_cap_);
                        vha_postmix_f32(d_attn_o_, (const float*) vha_m->ptr, A, g_.hca_heads, g_.hca_head_dim,
                                        d_attn_o2_, stream_);
                        hca_inv_rope(d_attn_o2_, pos_s, A, g_.hca_heads, g_.hca_head_dim, hca_rope_, stream_);

                        if (A > kMaxDecodeWindow) {
                            ensure_oa_plan(A);
                            if (loaded_mmq_.insert({kTypeQ8_0, gout}).second) prefill::mmq::load(kTypeQ8_0, gout);
                            prefill::mmq::quantize(d_attn_o2_, big_.oa_ids, big_.xq1, kTypeQ8_0, gin, gin,
                                                   (int64_t) g_.hca_o_groups * A, stream_);
                            prefill::mmq::Product op;
                            op.w = oa->ptr;
                            op.type = kTypeQ8_0;
                            op.w_rows = gout;
                            op.w_cols = gin;
                            op.expert_bytes = oa_grp_bytes;
                            op.n = g_.hca_o_groups;
                            op.xq = big_.xq1;
                            op.bounds = big_.oa_bounds;
                            op.ids = big_.oa_ids;
                            op.total_rows = (int64_t) g_.hca_o_groups * A;
                            op.max_rows = A;
                            op.dst = z_s;
                            op.ld_dst = gout;
                            mmq_ctx_.run(op, stream_);
                        } else {
                            for (int grp = 0; grp < g_.hca_o_groups; ++grp) {
                                DevTensor sub_w{(uint8_t*) oa->ptr + (size_t) grp * oa_grp_bytes, kTypeQ8_0,
                                                oa_grp_bytes, {}};
                                hca_group_split_f32(d_attn_o2_, A, g_.hca_o_groups, grp, gin, d_x_in_, stream_);
                                matmul_q8(&sub_w, d_x_in_, d_ff_h_, gin, gout, A);
                                hca_group_merge_f32(d_ff_h_, A, g_.hca_o_groups, grp, gout, z_s, stream_);
                            }
                        }
                    }

                    sigmoid_gate_mul_f32(p_z_full, p_gate_full, (int64_t) T * gate_w, stream_);
                    const int ob_chunk = std::max(kAttRows, big_.rows / 2);
                    for (int a0 = 0; a0 < T; a0 += ob_chunk) {
                        const int tw = std::min(ob_chunk, T - a0);
                        if (tw <= kMaxDecodeWindow) {
                            matmul_q8(ob, p_z_full + (size_t) a0 * gate_w, p_sub_out + (size_t) a0 * H, gate_w, H, tw);
                        } else {
                            prefill::mmq::quantize(p_z_full + (size_t) a0 * gate_w, nullptr, big_.xq1, kTypeQ8_0,
                                                   gate_w, gate_w, tw, stream_);
                            matmul_q8_prequant(ob, big_.xq1, p_sub_out + (size_t) a0 * H, gate_w, H, tw);
                        }
                    }
                } else {
                for (int a0 = 0; a0 < T; a0 += kAttRows) {
                    const int A = std::min(kAttRows, T - a0);
                    const float* x_s = p_x_norm + (size_t) a0 * H;
                    const int32_t* pos_s = p_pos + a0;
                    float* out_s = p_sub_out + (size_t) a0 * H;

                    const bool win = A <= kMaxDecodeWindow;
                    if (win) {
                        if (!fuse) quant8(x_s, H, A);
                        q8_products({{qa, d_qr_}, {wkv, d_kv_}}, {g_.hca_q_lora, g_.hca_head_dim}, H, A);
                    } else {
                        quantize_q8_input(x_s, d_xq_, H, A);
                        matmul_q8_prequant(qa, d_xq_, d_qr_, H, g_.hca_q_lora, A);
                        matmul_q8_prequant(wkv, d_xq_, d_kv_, H, g_.hca_head_dim, A);
                    }
                    if (win) {
                        rms_norm_q8(d_qr_, (const float*) qan->ptr, A, g_.hca_q_lora, g_.rms_eps, nullptr, d_xq_, stream_);
                        q8_products({{qb, d_q_}, {agate, d_gate_}},
                                    {g_.hca_heads * g_.hca_head_dim, g_.hca_o_groups * g_.hca_o_lora}, g_.hca_q_lora, A);
                    } else {
                        kernels::rms_norm_rows(d_qr_, (const float*) qan->ptr, A, g_.hca_q_lora, g_.rms_eps, d_qr_, stream_);
                        quantize_q8_input(d_qr_, d_xq_, g_.hca_q_lora, A);
                        matmul_q8_prequant(qb, d_xq_, d_q_, g_.hca_q_lora, g_.hca_heads * g_.hca_head_dim, A);
                        matmul_q8_prequant(agate, d_xq_, d_gate_, g_.hca_q_lora, g_.hca_o_groups * g_.hca_o_lora, A);
                    }
                    // a window commits its rows before the attention when the ring keeps them apart from the ones it reads
                    const bool commit_early = win && win_cap_ >= g_.window + kMaxDecodeWindow;
                    if (commit_early) {
                        hca_qkv_rope_commit(d_q_, d_kv_, (const float*) kvn->ptr, pos_s, A, g_.hca_heads, g_.hca_head_dim,
                                            g_.rms_eps, hca_rope_, d_kv_, win_cap_, hs.win_cache, fp8, stream_);
                    } else {
                        hca_q_rope(d_q_, pos_s, A, g_.hca_heads, g_.hca_head_dim, g_.rms_eps, hca_rope_, stream_);
                        hca_kv_norm_rope(d_kv_, (const float*) kvn->ptr, pos_s, A, g_.hca_head_dim, g_.rms_eps, hca_rope_,
                                         d_kv_, stream_);
                    }

                    if (win) {
                        matmul_bf16(ckv, x_s, d_ckv_, H, g_.hca_head_dim, A);
                        matmul_bf16(cg, x_s, d_cscore_, H, g_.hca_head_dim, A);
                    } else {
                        kernels::f32_to_bf16_bulk(x_s, d_bf16_x_, (int64_t) A * H, stream_);
                        matmul_bf16_preconv(ckv, d_bf16_x_, d_ckv_, H, g_.hca_head_dim, A);
                        matmul_bf16_preconv(cg, d_bf16_x_, d_cscore_, H, g_.hca_head_dim, A);
                    }
                    hca_compress(d_ckv_, d_cscore_, (const float*) cape->ptr, (const float*) cn->ptr, pos_s, A, ratio,
                                 g_.hca_head_dim, g_.rms_eps, hca_rope_, hs.state_kv, hs.state_score, hs.comp_cache,
                                 fp8, stream_, win ? -1 : (pos0 + a0));

                    if (A <= kMaxDecodeWindow && g_.hca_head_dim == 512)
                        hca_attention_split(d_q_, hs.comp_cache, hs.win_cache, d_kv_, (const float*) asink->ptr, pos_s,
                                            A, g_.hca_heads, g_.window, ratio, fp8, d_attn_o_, d_attn_part_, stream_,
                                            win_cap_);
                    else
                        hca_attention(d_q_, hs.comp_cache, hs.win_cache, d_kv_, (const float*) asink->ptr, pos_s, A,
                                      g_.hca_heads, g_.hca_head_dim, g_.window, ratio, fp8, d_attn_o_, stream_, win_cap_);
                    if (!commit_early)
                        hca_win_commit(d_kv_, pos_s, A, g_.hca_head_dim, g_.window, hs.win_cache, fp8, stream_, win_cap_);

                    if (win) {   // the groups side by side: the mix, rotation, split and quantization in one launch, a
                                 // matmul a group, then the merge, gate and quantization in one launch
                        const int G = g_.hca_o_groups;
                        const size_t col = kernels::native_q8_1_bytes(gin, 1);
                        hca_post_rope_q8(d_attn_o_, (const float*) vha_m->ptr, pos_s, A, g_.hca_heads, g_.hca_head_dim, hpg,
                                         hca_rope_, d_xq_, stream_);
                        Q8xMat og[kQ8xMats];
                        for (int grp = 0; grp < G; ++grp)
                            og[grp] = {(uint8_t*) oa->ptr + (size_t) grp * oa_grp_bytes, (uint8_t*) d_xq_ + (size_t) grp * A * col,
                                       d_ff_h_ + (size_t) grp * A * gout, gout};
                        q8_run(og, G, gin, A);
                        merge_gate_q8(d_ff_h_, d_gate_, A, G, gout, d_xq_, stream_);
                        const Q8xMat o{ob->ptr, d_xq_, out_s, H};
                        q8_run(&o, 1, G * gout, A);
                    } else {
                        vha_postmix_f32(d_attn_o_, (const float*) vha_m->ptr, A, g_.hca_heads, g_.hca_head_dim, d_attn_o2_,
                                        stream_);
                        hca_inv_rope(d_attn_o2_, pos_s, A, g_.hca_heads, g_.hca_head_dim, hca_rope_, stream_);
                        if (big_.ready) {
                            ensure_oa_plan(A);
                            if (loaded_mmq_.insert({kTypeQ8_0, gout}).second) prefill::mmq::load(kTypeQ8_0, gout);
                            prefill::mmq::quantize(d_attn_o2_, big_.oa_ids, big_.xq1, kTypeQ8_0, gin, gin,
                                                   (int64_t) g_.hca_o_groups * A, stream_);
                            prefill::mmq::Product op;
                            op.w = oa->ptr;
                            op.type = kTypeQ8_0;
                            op.w_rows = gout;
                            op.w_cols = gin;
                            op.expert_bytes = oa_grp_bytes;
                            op.n = g_.hca_o_groups;
                            op.xq = big_.xq1;
                            op.bounds = big_.oa_bounds;
                            op.ids = big_.oa_ids;
                            op.total_rows = (int64_t) g_.hca_o_groups * A;
                            op.max_rows = A;
                            op.dst = d_z_;
                            op.ld_dst = gout;
                            mmq_ctx_.run(op, stream_);
                        } else {
                            for (int grp = 0; grp < g_.hca_o_groups; ++grp) {
                                DevTensor sub_w{(uint8_t*) oa->ptr + (size_t) grp * oa_grp_bytes, kTypeQ8_0, oa_grp_bytes, {}};
                                hca_group_split_f32(d_attn_o2_, A, g_.hca_o_groups, grp, gin, d_x_in_, stream_);
                                matmul_q8(&sub_w, d_x_in_, d_ff_h_, gin, gout, A);
                                hca_group_merge_f32(d_ff_h_, A, g_.hca_o_groups, grp, gout, d_z_, stream_);
                            }
                        }
                        sigmoid_gate_mul_f32(d_z_, d_gate_, (int64_t) A * g_.hca_o_groups * g_.hca_o_lora, stream_);
                        matmul_q8(ob, d_z_, out_s, g_.hca_o_groups * g_.hca_o_lora, H, A);
                    }
                }
                }
            } else {
                const DevTensor *qa = req_t(l, "attn_q_a.weight", err), *qan = req_t(l, "attn_q_a_norm.weight", err),
                                *qb = req_t(l, "attn_q_b.weight", err), *kva = req_t(l, "attn_kv_a_mqa.weight", err),
                                *kvn = req_t(l, "attn_kv_a_norm.weight", err), *kvb = req_t(l, "attn_kv_b.weight", err),
                                *wo = req_t(l, "attn_output.weight", err), *iqb = req_t(l, "indexer.attn_q_b.weight", err),
                                *ikw = req_t(l, "indexer.attn_k.weight", err), *iknw = req_t(l, "indexer.k_norm.weight", err),
                                *iknb = req_t(l, "indexer.k_norm.bias", err), *ipr = req_t(l, "indexer.proj.weight", err);
                if (!qa || !qan || !qb || !kva || !kvn || !kvb || !wo || !iqb || !ikw || !iknw || !iknb || !ipr)
                    return false;

                auto& ds = dsa_[(size_t) l];
                const int sel_w = g_.idx_top_k + g_.window;
                const bool use_f16_kvb = big_.ready && T > kMaxDecodeWindow &&
                                         g_.mla_kv_lora == 512 && g_.mla_qk_rope == 64;
                if (use_f16_kvb) {
                    const int64_t kvb_blocks = (int64_t) g_.mla_heads * (g_.mla_qk_nope + g_.mla_v_dim) *
                                               (g_.mla_kv_lora / 32);
                    dequant_q8_0_to_f16(kvb->ptr, big_.kvb_f16, kvb_blocks, stream_);
                }
                const bool dsa_full_t = use_prompt_moe && big &&
                                        g_.mla_q_lora <= std::max(g_.n_ff_exp, 2 * g_.n_expert) &&
                                        (g_.mla_kv_lora + g_.mla_qk_rope) <= std::max(g_.n_ff_exp, g_.n_latent) &&
                                        (g_.idx_dim + g_.idx_heads) <= H;
                float* p_qr_full = big_.ff_g;
                float* p_kv_full = big_.ff_u;
                float* p_ik_full = big_.x_in;
                float* p_iw_full = big_.x_in + (size_t) T * g_.idx_dim;
                if (dsa_full_t) {
                    prefill::mmq::quantize(p_x_norm, nullptr, big_.xq1, kTypeQ8_0, H, H, T, stream_);
                    matmul_q8_prequant(qa, big_.xq1, p_qr_full, H, g_.mla_q_lora, T);
                    matmul_q8_prequant(kva, big_.xq1, p_kv_full, H, g_.mla_kv_lora + g_.mla_qk_rope, T);
                    kernels::rms_norm_rows(p_qr_full, (const float*) qan->ptr, T, g_.mla_q_lora, g_.rms_eps, p_qr_full,
                                           stream_);
                    uint16_t* p_bf16_full = reinterpret_cast<uint16_t*>(big_.pair_rows);
                    kernels::f32_to_bf16_bulk(p_x_norm, p_bf16_full, (int64_t) T * H, stream_);
                    matmul_bf16_preconv(ikw, p_bf16_full, p_ik_full, H, g_.idx_dim, T);
                    matmul_bf16_preconv(ipr, p_bf16_full, p_iw_full, H, g_.idx_heads, T);
                }
                for (int a0 = 0; a0 < T; a0 += kAttRows) {
                    const int A = std::min(kAttRows, T - a0);
                    const float* x_s = p_x_norm + (size_t) a0 * H;
                    const int32_t* pos_s = p_pos + a0;
                    float* out_s = p_sub_out + (size_t) a0 * H;

                    const bool win = A <= kMaxDecodeWindow;
                    const float* qr_s = dsa_full_t ? (p_qr_full + (size_t) a0 * g_.mla_q_lora) : d_qr_;
                    const float* kv_s = dsa_full_t ? (p_kv_full + (size_t) a0 * (g_.mla_kv_lora + g_.mla_qk_rope)) : d_kv_;
                    float* ik_s = dsa_full_t ? (p_ik_full + (size_t) a0 * g_.idx_dim) : d_ik_;
                    const float* iw_s = dsa_full_t ? (p_iw_full + (size_t) a0 * g_.idx_heads) : d_iw_;

                    if (!dsa_full_t) {
                        if (win) {
                            if (!fuse) quant8(x_s, H, A);
                            q8_products({{qa, d_qr_}, {kva, d_kv_}}, {g_.mla_q_lora, g_.mla_kv_lora + g_.mla_qk_rope}, H, A);
                        } else {
                            quantize_q8_input(x_s, d_xq_, H, A);
                            matmul_q8_prequant(qa, d_xq_, d_qr_, H, g_.mla_q_lora, A);
                            matmul_q8_prequant(kva, d_xq_, d_kv_, H, g_.mla_kv_lora + g_.mla_qk_rope, A);
                        }
                    }
                    if (win && !dsa_full_t) {
                        rms_norm_q8(d_qr_, (const float*) qan->ptr, A, g_.mla_q_lora, g_.rms_eps, nullptr, d_xq_, stream_);
                        q8_products({{agate, d_gate_}, {qb, d_q_}, {iqb, d_iq_}},
                                    {g_.mla_heads * g_.mla_v_dim, g_.mla_heads * (g_.mla_qk_nope + g_.mla_qk_rope),
                                     g_.idx_heads * g_.idx_dim}, g_.mla_q_lora, A);
                    } else {
                        if (!dsa_full_t)
                            kernels::rms_norm_rows(d_qr_, (const float*) qan->ptr, A, g_.mla_q_lora, g_.rms_eps, d_qr_, stream_);
                        quantize_q8_input(qr_s, d_xq_, g_.mla_q_lora, A);
                        matmul_q8_prequant(agate, d_xq_, d_gate_, g_.mla_q_lora, g_.mla_heads * g_.mla_v_dim, A);
                        matmul_q8_prequant(qb, d_xq_, d_q_, g_.mla_q_lora, g_.mla_heads * (g_.mla_qk_nope + g_.mla_qk_rope), A);
                        matmul_q8_prequant(iqb, d_xq_, d_iq_, g_.mla_q_lora, g_.idx_heads * g_.idx_dim, A);
                    }

                    dsa_qkv_rope_and_store(d_q_, kv_s, (const float*) kvn->ptr, pos_s, A, g_.mla_heads, g_.mla_qk_nope,
                                           g_.mla_qk_rope, g_.mla_kv_lora, g_.rms_eps, dsa_rope_, ds.kv_cache, fp8,
                                           stream_);

                    if (!dsa_full_t) {
                        if (win) {
                            matmul_bf16(ikw, x_s, d_ik_, H, g_.idx_dim, A);
                            matmul_bf16(ipr, x_s, d_iw_, H, g_.idx_heads, A);
                        } else {
                            kernels::f32_to_bf16_bulk(x_s, d_bf16_x_, (int64_t) A * H, stream_);
                            matmul_bf16_preconv(ikw, d_bf16_x_, d_ik_, H, g_.idx_dim, A);
                            matmul_bf16_preconv(ipr, d_bf16_x_, d_iw_, H, g_.idx_heads, A);
                        }
                    }
                    dsa_indexer_prepare(d_iq_, ik_s, (const float*) iknw->ptr, (const float*) iknb->ptr, pos_s, A,
                                        g_.idx_heads, g_.idx_dim, g_.mla_qk_rope, g_.rms_eps, dsa_rope_, ds.idx_k_cache,
                                        fp8, stream_);
                    const bool use_tc_indexer = big_.ready && A > kMaxDecodeWindow && g_.idx_dim == 128 &&
                                                g_.idx_heads <= 64 && big_.idx_q_bf16 && big_.idx_k_bf16 &&
                                                big_.idx_dots;
                    if (use_tc_indexer) {
                        const int64_t slice_older = std::max<int64_t>(0, (pos0 + a0 + A - 1) + 1 - g_.window);
                        if (slice_older > g_.idx_top_k) {
                            kernels::f32_to_bf16_bulk(d_iq_, big_.idx_q_bf16,
                                                      (int64_t) A * g_.idx_heads * g_.idx_dim, stream_);
                        }
                        const float alpha = 1.0f, beta = 0.0f;
                        // Reuse idle big_.pair_rows (unused during the DSA attention loop) to batch up to all A = 128
                        // query tokens per k0 pass, cutting idx_k_cache FP4->BF16 dequantization by up to 16x and
                        // widening the Tensor-Core GEMM N dimension from 512 to 8192.
                        const size_t pair_floats = big_.pair_rows
                                                       ? (size_t) big_.rows * (size_t) g_.n_used * (size_t) g_.n_latent
                                                       : 0;
                        const int score_stride =
                            (int) std::max<int64_t>(1, std::min<int64_t>(m_.context(), pos0 + a0 + A));
                        const size_t per_tok_floats =
                            (size_t) g_.idx_heads * (size_t) kIdxKeyTile + (size_t) score_stride;
                        const int idx_step = (per_tok_floats > 0 &&
                                              pair_floats / per_tok_floats > (size_t) kMaxDecodeWindow)
                                                 ? std::min(A, (int) (pair_floats / per_tok_floats))
                                                 : kMaxDecodeWindow;
                        const bool use_pair_idx = idx_step > kMaxDecodeWindow;
                        float* idx_dots_buf = use_pair_idx ? big_.pair_rows : big_.idx_dots;
                        float* idx_scores_buf =
                            use_pair_idx
                                ? (big_.pair_rows + (size_t) idx_step * (size_t) g_.idx_heads * (size_t) kIdxKeyTile)
                                : d_idx_scores_;
                        const int idx_scores_stride = use_pair_idx ? score_stride : (int) m_.context();
                        for (int t0 = 0; t0 < A; t0 += idx_step) {
                            const int tw = std::min(idx_step, A - t0);
                            const int tw_older =
                                (int) std::max<int64_t>(0, (pos0 + a0 + t0 + tw - 1) + 1 - g_.window);
                            if (tw_older > g_.idx_top_k) {
                                const uint16_t* q_bf16 = big_.idx_q_bf16 + (size_t) t0 * g_.idx_heads * g_.idx_dim;
                                for (int k0 = 0; k0 < tw_older; k0 += kIdxKeyTile) {
                                    const int kn = std::min(kIdxKeyTile, tw_older - k0);
                                    dsa_dequant_idx_k_bf16(ds.idx_k_cache, k0, kn, g_.idx_dim, fp8, big_.idx_k_bf16,
                                                           stream_);
                                    cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, kn, tw * g_.idx_heads, g_.idx_dim,
                                                 &alpha, big_.idx_k_bf16, CUDA_R_16BF, g_.idx_dim, q_bf16, CUDA_R_16BF,
                                                 g_.idx_dim, &beta, idx_dots_buf, CUDA_R_32F, kn, CUBLAS_COMPUTE_32F,
                                                 CUBLAS_GEMM_DEFAULT);
                                    dsa_indexer_reduce_tile(idx_dots_buf, iw_s + (size_t) t0 * g_.idx_heads, pos_s + t0,
                                                            tw, g_.idx_heads, g_.idx_dim, g_.idx_top_k, g_.window, k0,
                                                            kn, idx_scores_buf, idx_scores_stride, stream_);
                                }
                            }
                            dsa_indexer_select_cached(idx_scores_buf, idx_scores_stride, pos_s + t0, tw, g_.idx_top_k,
                                                      g_.window, d_sel_idx_ + (size_t) t0 * sel_w, stream_);
                        }
                    } else {
                        for (int t0 = 0; t0 < A; t0 += kMaxDecodeWindow) {
                            const int tw = std::min(kMaxDecodeWindow, A - t0);
                            dsa_indexer_topk_merge(d_iq_ + (size_t) t0 * g_.idx_heads * g_.idx_dim,
                                                   iw_s + (size_t) t0 * g_.idx_heads, ds.idx_k_cache, pos_s + t0, tw,
                                                   g_.idx_heads, g_.idx_dim, g_.idx_top_k, g_.window, fp8,
                                                   d_sel_idx_ + (size_t) t0 * sel_w, stream_, d_idx_scores_,
                                                   (int) m_.context());
                        }
                    }

                    if (use_f16_kvb && A > kMaxDecodeWindow) {
                        // Tensor-Core batched FP16 W_UK^T -> pure pre-absorbed sparse attention -> batched FP16 W_UV
                        dsa_pack_q_nope_f16(d_q_, big_.q_nope_f16, A, g_.mla_heads, g_.mla_qk_nope, g_.mla_qk_rope,
                                            stream_);
                        const float alpha = 1.0f, beta = 0.0f;
                        const long long stride_kvb = (long long) (g_.mla_qk_nope + g_.mla_v_dim) * g_.mla_kv_lora;
                        cublasGemmStridedBatchedEx(
                            cublas_, CUBLAS_OP_N, CUBLAS_OP_N, g_.mla_kv_lora, A, g_.mla_qk_nope, &alpha, big_.kvb_f16,
                            CUDA_R_16F, g_.mla_kv_lora, stride_kvb, big_.q_nope_f16, CUDA_R_16F, g_.mla_qk_nope,
                            (long long) A * g_.mla_qk_nope, &beta, d_attn_o_, CUDA_R_32F, g_.mla_kv_lora,
                            (long long) A * g_.mla_kv_lora, g_.mla_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                        dsa_mla_attention_preabsorbed(d_attn_o_, d_q_, ds.kv_cache, (const float*) asink->ptr,
                                                      d_sel_idx_, A, g_.mla_heads, g_.mla_qk_nope, g_.mla_qk_rope,
                                                      g_.mla_kv_lora, sel_w, fp8, big_.o_lat_f16, stream_);
                        const uint16_t* w_uv_f16 = big_.kvb_f16 + (size_t) g_.mla_qk_nope * g_.mla_kv_lora;
                        cublasGemmStridedBatchedEx(
                            cublas_, CUBLAS_OP_T, CUBLAS_OP_N, g_.mla_v_dim, A, g_.mla_kv_lora, &alpha, w_uv_f16,
                            CUDA_R_16F, g_.mla_kv_lora, stride_kvb, big_.o_lat_f16, CUDA_R_16F, g_.mla_kv_lora,
                            (long long) A * g_.mla_kv_lora, &beta, d_attn_o_, CUDA_R_32F, g_.mla_v_dim,
                            (long long) A * g_.mla_v_dim, g_.mla_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                        vha_postmix_f32(d_attn_o_, (const float*) vha_m->ptr, A, g_.mla_heads, g_.mla_v_dim, d_attn_o2_,
                                        stream_, /*head_major=*/true);
                    } else {
                        if (A > kMaxDecodeWindow ||
                            !dsa_mla_attention_split(d_q_, ds.kv_cache, kvb->ptr, (const float*) asink->ptr, d_sel_idx_, pos_s, A,
                                                     g_.mla_heads, g_.mla_qk_nope, g_.mla_qk_rope, g_.mla_kv_lora,
                                                     g_.mla_v_dim, sel_w, fp8, d_attn_o_, d_attn_part_, d_attn_qlat_, stream_))
                            dsa_mla_attention(d_q_, ds.kv_cache, kvb->ptr, (const float*) asink->ptr, d_sel_idx_, A,
                                              g_.mla_heads, g_.mla_qk_nope, g_.mla_qk_rope, g_.mla_kv_lora, g_.mla_v_dim,
                                              sel_w, fp8, d_attn_o_, stream_);
                        vha_postmix_f32(d_attn_o_, (const float*) vha_m->ptr, A, g_.mla_heads, g_.mla_v_dim, d_attn_o2_,
                                        stream_);
                    }
                    if (win) {   // the gate and the quantization in one launch
                        merge_gate_q8(d_attn_o2_, d_gate_, A, 1, g_.mla_heads * g_.mla_v_dim, d_xq_, stream_);
                        const Q8xMat o{wo->ptr, d_xq_, out_s, H};
                        q8_run(&o, 1, g_.mla_heads * g_.mla_v_dim, A);
                    } else {
                        sigmoid_gate_mul_f32(d_attn_o2_, d_gate_, (int64_t) A * g_.mla_heads * g_.mla_v_dim, stream_);
                        matmul_q8(wo, d_attn_o2_, out_s, g_.mla_heads * g_.mla_v_dim, H, A);
                    }
                }
            }

            mhc_post(p_sub_out, res, p_mhc_coef, kHcCoef, T, H, res_next, stream_);
            std::swap(res, res_next);

            // ---- FFN / Latent MoE sublayer ----
            const DevTensor *fh_fn = req_t(l, "hc_ffn_fn.weight", err),
                            *fh_sc = req_t(l, "hc_ffn_scale.weight", err),
                            *fh_bs = req_t(l, "hc_ffn_base.weight", err),
                            *fnorm = req_t(l, "ffn_norm.weight", err);
            if (!fh_fn || !fh_sc || !fh_bs || !fnorm) return false;

            if (fuse) {   // x_norm and its Q8_1 in d_xq_
                mhc_pre_norm(res, (const float*) fh_fn->ptr, (const float*) fh_sc->ptr, (const float*) fh_bs->ptr, T, H,
                             g_.rms_eps, g_.hc_eps, g_.sinkhorn_iters, (const float*) fnorm->ptr, g_.rms_eps, p_mhc_ws,
                             d_mhc_cnt_, p_mhc_coef, p_x_norm, d_xq_, stream_);
            } else {
                mhc_mixes(res, (const float*) fh_fn->ptr, (const float*) fh_sc->ptr, (const float*) fh_bs->ptr, T, H,
                          g_.rms_eps, g_.hc_eps, g_.sinkhorn_iters, p_mhc_ws, p_mhc_coef, stream_);
                mhc_pre(res, p_mhc_coef, kHcCoef, T, H, p_x_in, stream_);
                kernels::rms_norm_rows(p_x_in, (const float*) fnorm->ptr, T, H, g_.rms_eps, p_x_norm, stream_);
            }

            if (l < g_.n_dense) {
                const DevTensor *fg = req_t(l, "ffn_gate.weight", err),
                                *fu = req_t(l, "ffn_up.weight", err),
                                *fd = req_t(l, "ffn_down.weight", err);
                if (!fg || !fu || !fd) return false;
                for (int a0 = 0; a0 < T; a0 += kAttRows) {
                    const int A = std::min(kAttRows, T - a0);
                    matmul_q8(fg, p_x_norm + (size_t) a0 * H, d_ff_g_, H, g_.n_ff, A);
                    matmul_q8(fu, p_x_norm + (size_t) a0 * H, d_ff_u_, H, g_.n_ff, A);
                    situ_glu_f32(d_ff_g_, d_ff_u_, A, g_.n_ff, g_.gate_softcap, g_.up_softcap, d_ff_h_, stream_);
                    matmul_q8(fd, d_ff_h_, p_sub_out + (size_t) a0 * H, g_.n_ff, H, A);
                }
            } else {
                const DevTensor *rg = req_t(l, "ffn_gate_inp.weight", err),
                                *rb = req_t(l, "exp_probs_b.bias", err),
                                *rs = req_t(l, "exp_probs_scale.weight", err),
                                *ld = req_t(l, "ffn_routed_down.weight", err),
                                *ln = req_t(l, "ffn_latent_norm.weight", err),
                                *lu = req_t(l, "ffn_routed_up.weight", err),
                                *sg = req_t(l, "ffn_gate_shexp.weight", err),
                                *su = req_t(l, "ffn_up_shexp.weight", err),
                                *sd = req_t(l, "ffn_down_shexp.weight", err);
                if (!rg || !rb || !rs || !ld || !ln || !lu || !sg || !su || !sd) return false;

                // Split-feature routing (with optional decode-only --route-prior / --route-tail-skip)
                const int moe_l = l - g_.n_dense;
                router_logits(rg, p_x_norm, p_rlogits, T);
                kernels::ResidentTabs held{};
                float prior = 0.0f;
                if (lossy_route) {
                    held = tier_ptr->held_tabs(moe_l);
                    prior = tier_ptr->route_prior();
                }
                split_sigmoid_route(p_rlogits, (const float*) rb->ptr, (const float*) rs->ptr, T, g_.n_expert,
                                    g_.n_used, g_.norm_topk_prob, p_exp_ids, p_exp_w, stream_, held, prior);
                if (lossy_route && tier_ptr->route_tail_skip() > 0) {
                    for (int g0 = 0; g0 < T; g0 += kMaxDecodeWindow) {
                        const int tw = std::min(kMaxDecodeWindow, T - g0);
                        kernels::route_tail_skip(p_exp_ids + (size_t) g0 * g_.n_used,
                                                 p_exp_w + (size_t) g0 * g_.n_used, held, tw * g_.n_used, g_.n_used,
                                                 tier_ptr->route_tail_skip(), g_.n_expert, tier_ptr->route_counts(),
                                                 stream_);
                    }
                }

                // Routed experts in latent space: with the decode experts the pairs off the card start on the host
                // first, and the shared expert and the card's pairs run while it computes them
                const bool dec = dec_ && !use_prompt_moe;
                if (use_prompt_moe) {
                    // Kick async D2H of routed expert IDs on copy_stream_ immediately after split_sigmoid_route
                    const size_t n_pairs = (size_t) T * (size_t) g_.n_used;
                    cudaEventRecord(big_.ev_route, stream_);
                    cudaStreamWaitEvent(copy_stream_, big_.ev_route, 0);
                    cudaMemcpyAsync(big_.h_ids, p_exp_ids, n_pairs * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                    copy_stream_);
                    cudaEventRecord(big_.ev_ids, copy_stream_);

                    // Quantize p_x_norm ONCE for full T, then run ld + shared expert (sg, su, sd) on stream_
                    // while IDs transfer and moe_prompt buckets + starts GPU-0/GPU-1 PCIe staging in parallel.
                    prefill::mmq::quantize(p_x_norm, nullptr, big_.xq1, kTypeQ8_0, H, H, T, stream_);
                    matmul_q8_prequant(ld, big_.xq1, p_x_lat, H, g_.n_latent, T);
                    cudaEventRecord(big_.ev_xlat, stream_);
                    matmul_q8_prequant(sg, big_.xq1, p_ff_g, H, g_.n_ff_exp, T);
                    matmul_q8_prequant(su, big_.xq1, p_ff_u, H, g_.n_ff_exp, T);
                    situ_glu_f32(p_ff_g, p_ff_u, T, g_.n_ff_exp, g_.gate_softcap, g_.up_softcap, p_ff_h, stream_);
                    prefill::mmq::quantize(p_ff_h, nullptr, big_.xq2, kTypeQ8_0, g_.n_ff_exp, g_.n_ff_exp, T, stream_);
                    matmul_q8_prequant(sd, big_.xq2, p_sub_out, g_.n_ff_exp, H, T);

                    if (!moe_prompt(l, moe_l, T, p_x_lat, p_exp_w, p_lat_sum, err)) return false;
                    kernels::rms_norm_rows(p_lat_sum, (const float*) ln->ptr, T, g_.n_latent, g_.rms_eps, p_lat_norm, stream_);
                    matmul_q8(lu, p_lat_norm, p_routed_out, g_.n_latent, H, T);
                    add_f32(p_sub_out, p_routed_out, (int64_t) T * H, stream_);
                } else {
                    if (dec) {   // routed_down and the shared expert's gate and up read one quantization
                        if (!fuse) quant8(p_x_norm, H, T);
                        q8_products({{ld, p_x_lat}}, {g_.n_latent}, H, T);
                    } else {
                        matmul_q8(ld, p_x_norm, p_x_lat, H, g_.n_latent, T);
                    }
                    if (dec) {
                        moe_dec_plan(p_exp_ids, T, g_.n_used, g_.n_expert, dec_->tab(moe_l), d_grp_ptr_, d_grp_start_,
                                     d_n_groups_, d_ent_dst_, d_ent_tok_, dec_->log(moe_l), stream_);
                        dec_->ring(moe_l, T, p_x_lat, p_exp_ids, stream_);
                    }

                    // Shared expert
                    if (dec) {
                        q8_products({{sg, p_ff_g}, {su, p_ff_u}}, {g_.n_ff_exp, g_.n_ff_exp}, H, T);
                    } else {
                        matmul_q8(sg, p_x_norm, p_ff_g, H, g_.n_ff_exp, T);
                        matmul_q8(su, p_x_norm, p_ff_u, H, g_.n_ff_exp, T);
                    }
                    if (dec) {   // the activation and its quantization in one launch
                        situ_glu_q8(p_ff_g, p_ff_u, T, g_.n_ff_exp, g_.gate_softcap, g_.up_softcap, d_xq_, stream_);
                        const Q8xMat m{sd->ptr, d_xq_, p_sub_out, H};
                        q8_run(&m, 1, g_.n_ff_exp, T);
                    } else {
                        situ_glu_f32(p_ff_g, p_ff_u, T, g_.n_ff_exp, g_.gate_softcap, g_.up_softcap, p_ff_h, stream_);
                        matmul_q8(sd, p_ff_h, p_sub_out, g_.n_ff_exp, H, T);
                    }

                    if (dec) {
                        const int pairs = T * g_.n_used;
                        kernels::native_quantize_q8_1(p_x_lat, d_xq_lat_, g_.n_latent, T, stream_);
                        kernels::native_expert_grouped(dec_layout_, d_grp_ptr_, d_grp_start_, d_n_groups_, d_ent_dst_,
                                                       d_ent_tok_, std::min(pairs, g_.n_expert), pairs, d_xq_lat_,
                                                       d_exp_scratch_, d_exp_out_, stream_);
                        prefetch_next(l);
                        if (!hint_next(l, p_x_norm, T, err)) return false;
                        dec_->merge(moe_l, T, p_exp_ids, d_exp_out_, stream_);
                        cudaEventRecordWithFlags(dec_->layer_done(moe_l), stream_,
                                                 capturing_ ? cudaEventRecordExternal : cudaEventRecordDefault);
                        after_join();
                        // the weighted sum, the latent norm and its quantization in one launch, then routed_up
                        rms_norm_q8(d_exp_out_, (const float*) ln->ptr, T, g_.n_latent, g_.rms_eps, nullptr, d_xq_, stream_,
                                    p_exp_w, g_.n_used);
                        const Q8xMat m{lu->ptr, d_xq_, p_routed_out, H};
                        q8_run(&m, 1, g_.n_latent, T);
                    } else {
                        if (!run_moe_experts_decode(l - g_.n_dense, T, p_x_lat, p_exp_ids, err)) return false;
                        weighted_sum_f32(d_exp_out_, p_exp_w, T, g_.n_used, g_.n_latent, p_lat_sum, stream_);
                        kernels::rms_norm_rows(p_lat_sum, (const float*) ln->ptr, T, g_.n_latent, g_.rms_eps, p_lat_norm, stream_);
                        matmul_q8(lu, p_lat_norm, p_routed_out, g_.n_latent, H, T);
                        add_f32(p_sub_out, p_routed_out, (int64_t) T * H, stream_);
                    }
                }
            }

            // the decode experts' routed output joins the shared expert's in the update (add_f32's sum)
            const bool routed_apart = l >= g_.n_dense && dec_ && !use_prompt_moe;
            mhc_post(p_sub_out, res, p_mhc_coef, kHcCoef, T, H, res_next, stream_, routed_apart ? p_routed_out : nullptr);
            std::swap(res, res_next);

            // DSpark feature tap: stream-averaged mHC OUTPUT of each target layer [40, 41, 42]
            if (drafter_) {
                const int slot = drafter_->feature_slot(l);
                if (slot >= 0) {
                    const int tap = std::min(T, kDsparkWindow);
                    const int t0 = T - tap;
                    mhc_stream_mean(res + (size_t) t0 * kHcMult * H, tap, H, d_feat_ + (size_t) slot * H,
                                    drafter_->feature_floats(), stream_);
                }
            }
        }

        hint_join();
        if (pf_used_) {   // the prefetch stream joins the window
            cudaEventRecord(pf_join_, pf_stream_);
            cudaStreamWaitEvent(stream_, pf_join_, 0);
            after_join();
            pf_used_ = false;
        }

        if (want_logits) {
            const DevTensor *hfn = req_t("hc_head_fn.weight", err),
                            *hsc = req_t("hc_head_scale.weight", err),
                            *hbs = req_t("hc_head_base.weight", err),
                            *onorm = req_t("output_norm.weight", err),
                            *head = req_t("output.weight", err),
                            *mmr = req_t("output_multimax_ranges", err),
                            *mmt = req_t("output_multimax_ts", err);
            if (!hfn || !hsc || !hbs || !onorm || !head || !mmr || !mmt) return false;

            mhc_contract(res, (const float*) hfn->ptr, (const float*) hsc->ptr, (const float*) hbs->ptr, T, H,
                         g_.rms_eps, g_.hc_eps, p_mhc_ws, p_mhc_coef, p_x_in, stream_);
            kernels::rms_norm_rows(p_x_in, (const float*) onorm->ptr, T, H, g_.rms_eps, p_x_norm, stream_);
            if (head->type == kTypeQ8_0) matmul_q8(head, p_x_norm, d_logits_, H, g_.n_vocab, T);
            else matmul_bf16(head, p_x_norm, d_logits_, H, g_.n_vocab, T);
            multimax_f32(d_logits_, (const float*) mmr->ptr, (const float*) mmt->ptr, T, g_.n_vocab, stream_);
        }
        return true;
    }

    void release() {
        release_prompt();
        ring_close();   // belt and braces: the ring and its threads die before the streams they use
        for (auto& s : hca_) {
            dfree(s.win_cache); dfree(s.comp_cache); dfree(s.state_kv); dfree(s.state_score);
        }
        for (auto& s : dsa_) {
            dfree(s.kv_cache); dfree(s.idx_k_cache);
        }
        dfree(d_pos_); dfree(d_res_a_); dfree(d_res_b_); dfree(d_x_in_); dfree(d_x_norm_); dfree(d_sub_out_);
        dfree(d_mhc_ws_); dfree(d_mhc_coef_); dfree(d_mhc_cnt_); dfree(d_qr_); dfree(d_q_); dfree(d_kv_); dfree(d_ckv_); dfree(d_cscore_);
        dfree(d_attn_o_); dfree(d_attn_o2_); dfree(d_gate_); dfree(d_z_); dfree(d_iq_); dfree(d_ik_); dfree(d_iw_);
        dfree(d_sel_idx_); dfree(d_idx_scores_); dfree(d_ff_g_); dfree(d_ff_u_); dfree(d_ff_h_); dfree(d_rlogits_); dfree(d_exp_ids_);
        dfree(d_exp_w_); dfree(d_x_lat_); dfree(d_lat_sum_); dfree(d_lat_norm_); dfree(d_routed_out_); dfree(d_exp_out_);
        dfree(d_logits_); dfree(d_picks_); dfree(d_lp_scratch_); dfree(d_lp_out_); dfree(d_mmq_bounds_); dfree(d_mmq_iota_);
        dfree(d_bias_); dfree(d_mask_); dfree(d_hist_); d_hist_cap_ = 0;
        dfree(d_attn_part_); dfree(d_attn_qlat_); dfree(d_bf16_x_);
        dfree(d_xq_); dfree(d_xq_lat_); dfree(d_exp_scratch_); dfree(d_grp_ptr_); dfree(d_grp_start_); dfree(d_n_groups_);
        dfree(d_ent_dst_); dfree(d_ent_tok_); dfree(d_feat_);
        if (cpu_rows_) { cudaFreeHost(cpu_rows_); cpu_rows_ = nullptr; }
        if (cpu_x_) { cudaFreeHost(cpu_x_); cpu_x_ = nullptr; }
        if (cpu_act_) { platform::free_aligned(cpu_act_); cpu_act_ = nullptr; }
        if (file_bounce_) { cudaFreeHost(file_bounce_); file_bounce_ = nullptr; }
        for (cudaGraphExec_t& ex : graph_)
            if (ex) { cudaGraphExecDestroy(ex); ex = nullptr; }
        if (h_stage_pos_) { cudaFreeHost(h_stage_pos_); h_stage_pos_ = nullptr; }
        if (h_stage_emb_) { cudaFreeHost(h_stage_emb_); h_stage_emb_ = nullptr; }
        if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
        if (cublas_ws_) { cudaFree(cublas_ws_); cublas_ws_ = nullptr; }
        if (copy_stream_) { cudaStreamDestroy(copy_stream_); copy_stream_ = nullptr; }
        if (ring_stream_) { cudaStreamDestroy(ring_stream_); ring_stream_ = nullptr; }
        if (pf_stream_) { cudaStreamDestroy(pf_stream_); pf_stream_ = nullptr; }
        if (hint_stream_) { cudaStreamDestroy(hint_stream_); hint_stream_ = nullptr; }
        if (hint_fork_) { cudaEventDestroy(hint_fork_); hint_fork_ = nullptr; }
        if (hint_join_) { cudaEventDestroy(hint_join_); hint_join_ = nullptr; }
        dfree(d_hint_logits_); dfree(d_hint_ids_); dfree(d_hint_w_);
        if (pf_fork_) { cudaEventDestroy(pf_fork_); pf_fork_ = nullptr; }
        if (pf_join_) { cudaEventDestroy(pf_join_); pf_join_ = nullptr; }
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    }

    bool device_bias(float*& dev, const float* host, std::string& err) {
        const size_t bytes = (size_t) g_.n_vocab * sizeof(float);
        if (!dev && !ck(cudaMalloc(&dev, bytes), "logit bias", err)) return false;
        return ck(cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, stream_), "logit bias copy", err);
    }

    bool ensure_hist(size_t elems, std::string& err) {
        if (elems <= d_hist_cap_) return true;
        const size_t cap = std::max(elems, d_hist_cap_ * 2);
        int32_t* next = nullptr;
        if (!ck(cudaMalloc(&next, cap * sizeof(int32_t)), "history", err)) return false;
        dfree(d_hist_);
        d_hist_ = next;
        d_hist_cap_ = cap;
        return true;
    }

    bool flush_features(int64_t upto, std::string& err) {
        if (!drafter_ || feat_n_ <= 0) return true;
        const int q = (int) std::clamp<int64_t>(upto - feat_pos0_, 0, feat_n_);
        feat_n_ = 0;
        if (q <= 0) return true;
        return drafter_->inject_device(dwin_, feat_pos0_, q, d_feat_, stream_, err);
    }

    ErnieModel& m_;
    const Geometry& g_;
    cudaStream_t stream_ = nullptr;
    cudaStream_t copy_stream_ = nullptr;
    cudaStream_t hint_stream_ = nullptr;          ///< a decode window's routing hints (hint_next)
    cudaEvent_t hint_fork_ = nullptr, hint_join_ = nullptr;
    bool hint_used_ = false;
    float *d_hint_logits_ = nullptr, *d_hint_w_ = nullptr;
    int32_t* d_hint_ids_ = nullptr;
    cudaStream_t pf_stream_ = nullptr;            ///< a decode window's L2 prefetches (prefetch_next)
    cudaEvent_t pf_fork_ = nullptr, pf_join_ = nullptr;
    bool pf_used_ = false;
    cublasHandle_t cublas_ = nullptr;
    prefill::mmq::Context mmq_ctx_;
    std::set<std::pair<int, int64_t>> loaded_mmq_;
    RopeTable hca_rope_{}, dsa_rope_{};

    std::vector<HcaLayerState> hca_;
    std::vector<DsaLayerState> dsa_;
    size_t stride_ = 0;
    int win_cap_ = kHcaWinCap;
    int64_t kv_bytes_ = 0;
    int last_window_n_ = 0;
    bool bias_on_ = false;
    float *d_bias_ = nullptr, *d_mask_ = nullptr;
    int32_t* d_hist_ = nullptr;
    size_t d_hist_cap_ = 0;
    int history_len_ = 0;
    kernels::SamplerParams sampling_{};

    PromptBufs big_{};
    bool lent_ = false;

    // The packed adaptive prefill expert ring: two slots of ring_cap_ expert blobs each. A long prompt chunk's next
    // MoE layer's local experts are packed into the slot it does not use, on a dedicated non-blocking stream, so the
    // PCIe RAM transfers overlap the HCA/MLA-DSA attention and dense/shared-expert compute instead of queueing
    // behind (and holding) the copy_stream_ the routing readback and the stage_[2] overflow staging share.
    cudaStream_t ring_stream_ = nullptr;
    uint8_t* ring_ = nullptr;
    int ring_cap_ = 0;                        ///< experts per ring slot (<= g_.n_expert)
    std::vector<int32_t> ring_pos_[2];        ///< [slot][expert] -> packed slot index in ring_, or -1
    cudaEvent_t ring_copied_[2] = {nullptr, nullptr}, ring_done_[2] = {nullptr, nullptr};
    int ring_layer_[2] = {-1, -1};            ///< the tier layer whose copy fills (slot) / whose MMQ read (done) it
    std::thread ring_thr_[2];                 ///< the issuer of the slot's copies (a ring stream's queue can be long)
    bool ring_ok_[2] = {false, false};
    std::atomic<double> t_ring_ms_{0};        ///< host ms the ring-copy threads spent queueing copies
    float *cpu_rows_ = nullptr, *cpu_x_ = nullptr;
    uint8_t* cpu_act_ = nullptr;
    double t_prompt_moe_ms_ = 0, t_prompt_peer_wait_ms_ = 0;
    int64_t n_prompt_dev_ = 0, n_prompt_bus_ = 0, n_prompt_peer_ = 0, n_prompt_cpu_ = 0, n_prompt_file_ = 0,
            n_prompt_ring_ = 0;
    uint8_t* file_bounce_ = nullptr;   ///< pinned staging of the prompt experts only the file holds (kStageExperts)
    int64_t n_prompt_peer_rows_ = 0, n_prompt_cpu_rows_ = 0;

    ErnieDsparkDrafter* drafter_ = nullptr;
    int dwin_ = -1;
    float* d_feat_ = nullptr;
    int64_t feat_pos0_ = 0;
    int feat_n_ = 0;
    DsparkDraft last_draft_{};

    int32_t *d_pos_ = nullptr, *d_sel_idx_ = nullptr, *d_exp_ids_ = nullptr, *d_picks_ = nullptr;
    int32_t *d_mmq_bounds_ = nullptr, *d_mmq_iota_ = nullptr;
    int32_t *d_grp_start_ = nullptr, *d_n_groups_ = nullptr, *d_ent_dst_ = nullptr, *d_ent_tok_ = nullptr;
    unsigned long long* d_grp_ptr_ = nullptr;
    float *d_res_a_ = nullptr, *d_res_b_ = nullptr, *d_x_in_ = nullptr, *d_x_norm_ = nullptr, *d_sub_out_ = nullptr;
    float *d_mhc_ws_ = nullptr, *d_mhc_coef_ = nullptr;
    unsigned* d_mhc_cnt_ = nullptr;   ///< mhc_pre_norm's counters, a row each
    float *d_qr_ = nullptr, *d_q_ = nullptr, *d_kv_ = nullptr, *d_ckv_ = nullptr, *d_cscore_ = nullptr;
    float *d_attn_o_ = nullptr, *d_attn_o2_ = nullptr, *d_gate_ = nullptr, *d_z_ = nullptr;
    float *d_iq_ = nullptr, *d_ik_ = nullptr, *d_iw_ = nullptr, *d_idx_scores_ = nullptr;
    float *d_ff_g_ = nullptr, *d_ff_u_ = nullptr, *d_ff_h_ = nullptr;
    float *d_rlogits_ = nullptr, *d_exp_w_ = nullptr, *d_x_lat_ = nullptr, *d_lat_sum_ = nullptr;
    float *d_lat_norm_ = nullptr, *d_routed_out_ = nullptr, *d_exp_out_ = nullptr;
    float *d_logits_ = nullptr, *d_lp_scratch_ = nullptr, *d_lp_out_ = nullptr;
    float *d_attn_part_ = nullptr, *d_attn_qlat_ = nullptr;
    uint16_t* d_bf16_x_ = nullptr;
    int last_mmq_bounds_t_ = -1;
    void *d_xq_ = nullptr, *d_xq_lat_ = nullptr, *d_exp_scratch_ = nullptr;
    DecodeMoe* dec_ = nullptr;           ///< the decode experts while a decode window runs
    bool staged_ = false;                ///< the window's rows are in the row buffers (stage_rows)
    bool capturing_ = false;             ///< true while stream_ is capturing a decode window CUDA graph
    cudaGraphExec_t graph_[kMaxDecodeWindow] = {};   ///< [rows - 1] the decode windows' graphs
    int32_t* h_stage_pos_ = nullptr;
    float* h_stage_emb_ = nullptr;
    void* cublas_ws_ = nullptr;
    bool placement_dirty_ = true;        ///< the tier's placement may have changed outside the decode windows
    kernels::NativeExpertLayout dec_layout_{};

    std::vector<float> h_emb_;
    std::vector<int32_t> h_pos_, h_ids_, h_grp_start_, h_ent_dst_, h_ent_tok_;
    std::vector<int32_t> h_cnt_, h_off_, h_cur_, h_pairs_, peer_tok_flat_;
    std::vector<unsigned long long> h_grp_ptr_;
    int peer_prepared_ = -1;
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(ErnieModel& model, std::string& err) {
    auto s = std::make_unique<ErnieSession>(model);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::ernie55
