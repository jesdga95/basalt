// src/models/deepseek4/dspark.cpp - the DSpark drafter: three stages of window-only attention and six routed MXFP4
// experts, a block-parallel pass over [anchor, noise x4], and the Markov and confidence heads.
//
// The stages' kept experts (the ranking's most routed; --draft-experts) live on the card as packed gate, up and down
// matrices, and the block's stages and head run as two CUDA graphs captured once per row count. The main model hands
// the drafter, for every verified position, the mean over the four hyper-connection streams of its target layers;
// `stage` turns a verify window's features into the stages' latents beside the window, `commit` moves the accepted
// ones into the rings, and `draft_begin`/`draft_end` split the block's launch from its result.
#include "models/deepseek4/dspark.hpp"

#include "basalt/core/expert_tier.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/kernels/sparse_mqa.hpp"
#include "basalt/log.hpp"
#include "basalt/models/deepseek4/attention.hpp"
#include "basalt/models/deepseek4/head.hpp"
#include "basalt/models/deepseek4/layer.hpp"
#include "basalt/models/deepseek4/rope.hpp"
#include "basalt/models/deepseek4/router.hpp"
#include "models/deepseek4/model.hpp"
#include "models/deepseek4/model_weights.hpp"
#include "models/deepseek4/routed_expert.hpp"
#include "models/deepseek4/shared_expert.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace basalt::models::deepseek4 {
namespace {

constexpr int kArgmaxBlocks = 256;
constexpr int kMxfp4Type = 39;

inline float bf16_to_f32(uint16_t u) {
    uint32_t x = (uint32_t) u << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("deepseek4 dspark ") + what + ": " + cudaGetErrorString(e);
    return false;
}

struct DevTensor {
    void* ptr = nullptr;
    int type = 0;
    uint64_t bytes = 0;
};

struct WindowState {
    float* rings = nullptr;  ///< [n_stages, kDsparkWinCap, 512] F32 SWA KV rings
    int64_t valid_end = 0;
};

// One stage's weights as the block's launches read them.
struct StageWeights {
    AttentionLayerWeights attn;
    FfnRouteWeights ffn;
    SharedExpertWeights shared;
};

}  // namespace

std::vector<std::vector<int>> dspark_kept_experts(const std::vector<std::pair<int32_t, int32_t>>& ranked,
                                                  int stages, int experts, int keep) {
    std::vector<std::vector<int>> out((size_t) stages);
    std::vector<uint8_t> taken((size_t) stages * (size_t) experts, 0);
    for (const auto& [s, e] : ranked) {
        if (s < 0 || s >= stages || e < 0 || e >= experts || (int) out[(size_t) s].size() >= keep) continue;
        if (taken[(size_t) s * (size_t) experts + (size_t) e]) continue;
        taken[(size_t) s * (size_t) experts + (size_t) e] = 1;
        out[(size_t) s].push_back(e);
    }
    for (int s = 0; s < stages; ++s)
        for (int e = 0; e < experts && (int) out[(size_t) s].size() < keep; ++e)
            if (!taken[(size_t) s * (size_t) experts + (size_t) e]) out[(size_t) s].push_back(e);
    for (auto& v : out) std::sort(v.begin(), v.end());
    return out;
}

struct DsparkDrafter::Impl {
    const DsModel* main = nullptr;
    Geometry g;
    DsparkPack dp;
    int device = 0;
    cudaStream_t stream = nullptr;     ///< the block's and the commits' stream
    cudaStream_t stage_st = nullptr;   ///< the verify window's latents, beside it
    cudaEvent_t ready_ev = nullptr, ev_done = nullptr, ev_staged = nullptr, ev_free = nullptr;

    std::map<std::string, DevTensor> dev;
    uint64_t dev_bytes = 0;
    uint16_t* d_markov_w1 = nullptr;

    std::vector<std::vector<int>> kept;   ///< ascending expert indices each stage keeps
    int kept_n = 0;
    std::string kept_note;
    std::vector<void*> gate, up, down;          ///< each [kept_n] matrices
    std::vector<float*> kept_gate, kept_bias;   ///< [kept_n][n_embd] F32 and [kept_n]
    std::vector<StageWeights> stages_w;
    HeadWeights head;
    const void* markov_w2 = nullptr;   // Q8_0 in this pack
    int markov_w2_type = -1;
    const float* conf_w = nullptr;
    bool built = false;

    std::vector<WindowState> windows;
    int next_win = 0;

    // device scratch
    std::vector<void*> dev_allocs;
    float *d_streams = nullptr, *d_next_streams = nullptr;
    float *d_hc_ws = nullptr, *d_coef = nullptr, *d_collapsed = nullptr, *d_normed = nullptr;
    void *d_q8 = nullptr, *d_ql_q8 = nullptr, *d_out_q8 = nullptr;
    float *d_ql = nullptr, *d_qh = nullptr, *d_kv_proj = nullptr, *d_kv_row = nullptr;
    float *d_mqa_out = nullptr, *d_mqa_ws = nullptr, *d_out_lat = nullptr, *d_attn_out = nullptr;
    float *d_logits_route = nullptr, *d_route_w = nullptr, *d_shared = nullptr;
    float *d_shared_gate = nullptr, *d_shared_up = nullptr, *d_shared_hidden = nullptr;
    void *d_shared_hq8 = nullptr;
    int32_t *d_route_ids = nullptr, *d_pos = nullptr, *d_wstart = nullptr, *d_wn = nullptr;
    float *d_feat_proj = nullptr, *d_feat_norm = nullptr;
    void *d_feat_q8 = nullptr;
    float *d_head_hc_ws = nullptr, *d_head_coef = nullptr, *d_head_col = nullptr, *d_head_norm = nullptr;
    float *d_base_logits = nullptr, *d_me = nullptr, *d_mbias = nullptr, *d_amax_v = nullptr, *d_conf = nullptr;
    void* d_me_q8 = nullptr;
    int32_t *d_amax_i = nullptr, *d_tok = nullptr, *d_stage_pos = nullptr;
    float *d_moe_h = nullptr, *d_moe_d = nullptr;
    void *d_moe_hq = nullptr;
    float* d_staged = nullptr;   // [stages][8][512], the verify window's latents before commit

    // host staging
    float* h_init = nullptr;
    float* h_conf = nullptr;
    int32_t* h_tok = nullptr;

    // graphs: the body by window and row count (its rings belong to the window), the head by row count
    std::vector<std::vector<cudaGraphExec_t>> body_exec;
    cudaGraphExec_t head_exec[kDsparkBlock + 1] = {};

    int staged_w = -1;
    int64_t staged_pos0 = -1;
    int staged_n = 0;
    bool staged_pending = false;
    int pending_n = 0;

    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        for (auto& v : body_exec) for (cudaGraphExec_t e : v) if (e) cudaGraphExecDestroy(e);
        for (cudaGraphExec_t& e : head_exec) if (e) cudaGraphExecDestroy(e);
        for (WindowState& w : windows)
            if (w.rings) cudaFree(w.rings);
        for (auto& [_, t] : dev)
            if (t.ptr) cudaFree(t.ptr);
        for (void* p : gate) if (p) cudaFree(p);
        for (void* p : up) if (p) cudaFree(p);
        for (void* p : down) if (p) cudaFree(p);
        for (float* p : kept_gate) if (p) cudaFree(p);
        for (float* p : kept_bias) if (p) cudaFree(p);
        if (d_markov_w1) cudaFree(d_markov_w1);
        for (void* p : dev_allocs) cudaFree(p);
        if (h_init) cudaFreeHost(h_init);
        if (h_conf) cudaFreeHost(h_conf);
        if (h_tok) cudaFreeHost(h_tok);
        if (ready_ev) cudaEventDestroy(ready_ev);
        if (ev_done) cudaEventDestroy(ev_done);
        if (ev_staged) cudaEventDestroy(ev_staged);
        if (ev_free) cudaEventDestroy(ev_free);
        if (stream) cudaStreamDestroy(stream);
        if (stage_st) cudaStreamDestroy(stage_st);
    }

    void* alloc_dev(size_t bytes) {
        void* p = nullptr;
        if (!bytes || cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        dev_allocs.push_back(p);
        dev_bytes += bytes;
        return p;
    }

    const DevTensor* req(const std::string& name, std::string& err) const {
        auto it = dev.find(name);
        if (it == dev.end() || !it->second.ptr) {
            err = "deepseek4 dspark: missing tensor " + name;
            return nullptr;
        }
        return &it->second;
    }

    // The stage and head weights as the captured graphs read them; built once, after the start's reads landed.
    bool build(std::string& err) {
        if (built) return true;
        const int H = g.n_embd;
        stages_w.assign((size_t) dp.num_stages, {});
        for (int s = 0; s < dp.num_stages; ++s) {
            const std::string pre = "stage." + std::to_string(s) + ".";
            const DevTensor *hc_af = req(pre + "hc_attn_fn.weight", err), *hc_as = req(pre + "hc_attn_scale.weight", err),
                            *hc_ab = req(pre + "hc_attn_base.weight", err), *anorm = req(pre + "attn_norm.weight", err),
                            *wqa = req(pre + "attn_q_a.weight", err), *qn = req(pre + "attn_q_a_norm.weight", err),
                            *wqb = req(pre + "attn_q_b.weight", err), *wkv = req(pre + "attn_kv.weight", err),
                            *kvn = req(pre + "attn_kv_a_norm.weight", err), *sinks = req(pre + "attn_sinks.weight", err),
                            *woa = req(pre + "attn_output_a.weight", err), *wob = req(pre + "attn_output_b.weight", err),
                            *hc_ff = req(pre + "hc_ffn_fn.weight", err), *hc_fs = req(pre + "hc_ffn_scale.weight", err),
                            *hc_fb = req(pre + "hc_ffn_base.weight", err), *fnorm = req(pre + "ffn_norm.weight", err),
                            *sg = req(pre + "ffn_gate_shexp.weight", err), *su = req(pre + "ffn_up_shexp.weight", err),
                            *sd = req(pre + "ffn_down_shexp.weight", err);
            if (!hc_af || !hc_as || !hc_ab || !anorm || !wqa || !qn || !wqb || !wkv || !kvn || !sinks || !woa ||
                !wob || !hc_ff || !hc_fs || !hc_fb || !fnorm || !sg || !su || !sd)
                return false;
            StageWeights& w = stages_w[(size_t) s];
            w.attn.hc = {H, g.rms_eps, g.hc_eps, g.sinkhorn_iters};
            w.attn.hc_fn = static_cast<const float*>(hc_af->ptr);
            w.attn.hc_scale = static_cast<const float*>(hc_as->ptr);
            w.attn.hc_base = static_cast<const float*>(hc_ab->ptr);
            w.attn.norm = static_cast<const float*>(anorm->ptr);
            w.attn.attention.q = {wqa->ptr, wqa->type, static_cast<const float*>(qn->ptr), wqb->ptr, wqb->type,
                                  g.rms_eps, H, g.q_lora, 1e-6f};
            w.attn.attention.kv = wkv->ptr;
            w.attn.attention.kv_type = wkv->type;
            w.attn.attention.kv_norm = static_cast<const float*>(kvn->ptr);
            w.attn.attention.sinks = static_cast<const float*>(sinks->ptr);
            w.attn.attention.output_a = woa->ptr;
            w.attn.attention.output_a_type = woa->type;
            w.attn.attention.output_b = wob->ptr;
            w.attn.attention.output_b_type = wob->type;

            w.ffn.hc = w.attn.hc;
            w.ffn.hc_fn = static_cast<const float*>(hc_ff->ptr);
            w.ffn.hc_scale = static_cast<const float*>(hc_fs->ptr);
            w.ffn.hc_base = static_cast<const float*>(hc_fb->ptr);
            w.ffn.norm = static_cast<const float*>(fnorm->ptr);
            w.ffn.gate = kept_gate[(size_t) s];
            w.ffn.routing = router_mixed_args(kept_n, g.n_vocab, nullptr, nullptr, nullptr, nullptr,
                                              kept_bias[(size_t) s], nullptr);

            w.shared = {sg->type, su->type, sd->type, sg->ptr, su->ptr, sd->ptr, H, g.n_ff_exp, g.swiglu_limit};
        }
        const DevTensor *hfn = req("hc_head_fn.weight", err), *hsc = req("hc_head_scale.weight", err),
                        *hbs = req("hc_head_base.weight", err), *onorm = req("output_norm.weight", err);
        if (!hfn || !hsc || !hbs || !onorm) return false;
        const DevTensor* mw2 = req("markov_w2.weight", err);
        const DevTensor* cw = req("conf_proj.weight", err);
        if (!mw2 || !cw || !main->weights().head_weights(head, err)) return false;
        markov_w2 = mw2->ptr;
        markov_w2_type = mw2->type;
        conf_w = static_cast<const float*>(cw->ptr);
        head.hc_fn = static_cast<const float*>(hfn->ptr);
        head.hc_scale = static_cast<const float*>(hsc->ptr);
        head.hc_base = static_cast<const float*>(hbs->ptr);
        head.norm = static_cast<const float*>(onorm->ptr);
        built = true;
        return true;
    }

    // The block's inputs: the anchor's row and the noise rows in all four streams, the positions and window bounds.
    bool upload_inputs(int32_t anchor, int64_t pos, int B, std::string& err) {
        const size_t H = (size_t) g.n_embd;
        std::vector<float> emb;
        for (int r = 0; r < B; ++r) {
            const int32_t tok = (r == 0) ? anchor : dp.noise_token_id;
            if (!main->weights().embedding_row(tok, emb, err)) return false;
            for (int s = 0; s < 4; ++s)
                std::copy_n(emb.data(), H, h_init + ((size_t) r * 4 + s) * H);
        }
        if (!ck(cudaMemcpyAsync(d_streams, h_init, (size_t) B * 4 * H * sizeof(float), cudaMemcpyHostToDevice, stream),
                "embed H2D", err))
            return false;
        int32_t hp[kDsparkBlock]{}, hws[kDsparkBlock]{}, hwn[kDsparkBlock]{};
        for (int r = 0; r < B; ++r) {
            hp[r] = (int32_t) (pos + r);
            hws[r] = (int32_t) std::max<int64_t>(0, pos + r - kDsparkWindow + 1);
            hwn[r] = hp[r] - hws[r] + 1;
        }
        if (!ck(cudaMemcpyAsync(d_pos, hp, (size_t) B * sizeof(int32_t), cudaMemcpyHostToDevice, stream), "pos", err) ||
            !ck(cudaMemcpyAsync(d_wstart, hws, (size_t) B * sizeof(int32_t), cudaMemcpyHostToDevice, stream), "ws", err) ||
            !ck(cudaMemcpyAsync(d_wn, hwn, (size_t) B * sizeof(int32_t), cudaMemcpyHostToDevice, stream), "wn", err))
            return false;
        return true;
    }

    // The block's stages over the B rows already uploaded.
    void run_body(int w, int B) {
        WindowState& ws = windows[(size_t) w];
        float* streams = d_streams;
        float* next = d_next_streams;
        const int H = g.n_embd;
        const int K = g.n_used;
        for (int s = 0; s < dp.num_stages; ++s) {
            StageWeights& st = stages_w[(size_t) s];
            AttentionLayerScratch as;
            as.hc_workspace = d_hc_ws;
            as.coefficients = d_coef;
            as.collapsed = d_collapsed;
            as.normalized = d_normed;
            as.attention = {d_q8, d_ql, d_ql_q8, d_qh, d_kv_proj, d_kv_row, d_mqa_out, d_mqa_ws, d_out_q8, d_out_lat,
                            d_attn_out};
            kernels::SparseMqaArgs sel;
            sel.n_tokens = B;
            sel.win = ws.rings + (size_t) s * kDsparkWinCap * 512;
            sel.win_type = kernels::MqaRowType::F32;
            sel.win_cap = kDsparkWindow;
            sel.win_start = d_wstart;
            sel.win_n = d_wn;
            sel.pos = d_pos;
            sel.rope = rope_for(s);
            attention_layer_step(streams, st.attn, sel.rope, ws.rings + (size_t) s * kDsparkWinCap * 512, sel, as, next,
                                 stream);
            std::swap(streams, next);

            FfnRouteScratch rs{d_hc_ws, d_coef, d_collapsed, d_normed, d_logits_route, d_route_ids, d_route_w};
            prepare_ffn_routes(streams, B, st.ffn, rs, stream);
            SharedExpertScratch ss{d_q8, d_shared_hq8, d_shared_gate, d_shared_up, d_shared_hidden};
            shared_expert_step(d_normed, B, st.shared, ss, d_shared, stream);
            void* gate_s = gate[(size_t) s];
            void* up_s = up[(size_t) s];
            void* down_s = down[(size_t) s];
            dspark_moe_gate_up(gate_s, up_s, H, g.n_ff_exp, d_q8, d_route_ids, B, K, g.swiglu_limit, nullptr, d_moe_h,
                               stream);
            kernels::native_quantize_q8_1(d_moe_h, d_moe_hq, g.n_ff_exp, B * K, stream);
            dspark_moe_down(down_s, g.n_ff_exp, H, d_moe_hq, d_route_ids, B, K, nullptr, d_moe_d, stream);
            weighted_route_sum(d_moe_d, d_route_w, B, H, d_routed_rows_, stream);
            finish_ffn_layer(streams, d_coef, d_shared, d_routed_rows_, B, H, next, stream);
            std::swap(streams, next);
        }
        if (streams != d_streams)   // an odd stage count ends in the ping-pong buffer
            cudaMemcpyAsync(d_streams, streams, (size_t) B * 4 * (size_t) g.n_embd * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
    }

    // The main model's output matrix over the rows a window can use, then the Markov chain and each row's confidence.
    void run_head(int n) {
        const size_t V = (size_t) g.n_vocab;
        HeadScratch hs{d_head_hc_ws, d_head_coef, d_head_col, d_head_norm};
        model_head_step(d_streams, n, head, hs, d_base_logits, stream);
        for (int i = 0; i < n; ++i) {
            dspark_markov_gather(d_markov_w1, d_tok + i, d_me, stream);
            kernels::native_quantize_q8_1(d_me, d_me_q8, dp.markov_rank, 1, stream);
            kernels::native_mmvq(markov_w2_type, markov_w2, d_me_q8, d_mbias, dp.markov_rank, (int) V, 1, stream);
            dspark_markov_argmax(d_base_logits + (size_t) i * V, d_mbias, (int) V, d_amax_v, d_amax_i, d_tok + i + 1,
                                 stream);
            dspark_confidence(conf_w, d_head_col + (size_t) i * g.n_embd, d_me, g.n_embd, dp.markov_rank, d_conf + i,
                              stream);
        }
    }

    // The captured graph of `body` for `rows`, launched after the caller's inputs are queued.
    template <class F>
    bool launch(cudaGraphExec_t& exec, F&& body, std::string& err) {
        if (!exec) {
            if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
                err = "deepseek4 dspark: cannot begin a graph capture";
                return false;
            }
            body();
            cudaGraph_t graph = nullptr;
            const cudaError_t ce = cudaStreamEndCapture(stream, &graph);
            const cudaError_t ci = ce == cudaSuccess ? cudaGraphInstantiateWithFlags(&exec, graph, 0) : ce;
            if (graph) cudaGraphDestroy(graph);
            if (ci != cudaSuccess) {
                err = std::string("deepseek4 dspark: graph capture: ") + cudaGetErrorString(ci);
                return false;
            }
        }
        if (cudaGraphLaunch(exec, stream) != cudaSuccess) {
            err = "deepseek4 dspark: graph launch failed";
            return false;
        }
        return true;
    }

    float* d_routed_rows_ = nullptr;   // [B][n_embd] the stage's routed sum
    RopeTables ropes[kDsparkBlock];    // per-stage rope tables (the stage ids are the target layers)
    const kernels::MqaRopeTable& rope_for(int stage) const { return ropes[stage].attention; }
};

DsparkDrafter::DsparkDrafter(std::unique_ptr<Impl> impl) : p_(std::move(impl)) {}
DsparkDrafter::~DsparkDrafter() = default;

std::unique_ptr<DsparkDrafter> DsparkDrafter::plan_load(const DsModel& main, int file_handle, core::LoadPlan& reads,
                                                        int max_windows, const std::string& ranking, int keep_asked,
                                                        std::string& err) {
    const Pack& pack = main.weights().pack();
    if (!pack.dspark.present) {
        err = "deepseek4 dspark: pack has no dspark section";
        return nullptr;
    }
    auto p = std::make_unique<Impl>();
    p->main = &main;
    p->g = pack.geo;
    p->dp = pack.dspark;
    if (!ck(cudaGetDevice(&p->device), "get device", err) ||
        !ck(cudaStreamCreateWithFlags(&p->stream, cudaStreamNonBlocking), "create stream", err) ||
        !ck(cudaEventCreateWithFlags(&p->ready_ev, cudaEventDisableTiming), "create event", err) ||
        !ck(cudaEventCreateWithFlags(&p->ev_done, cudaEventDisableTiming), "create event", err) ||
        !ck(cudaEventCreateWithFlags(&p->ev_staged, cudaEventDisableTiming), "create event", err) ||
        !ck(cudaEventCreateWithFlags(&p->ev_free, cudaEventDisableTiming), "create event", err))
        return nullptr;
    {
        int lo = 0, hi = 0;
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        if (!ck(cudaStreamCreateWithPriority(&p->stage_st, cudaStreamNonBlocking, lo), "create stage stream", err))
            return nullptr;
    }
    // the stage rope tables: the drafter's stages are the model's last target layers
    for (int s = 0; s < p->dp.num_stages; ++s) {
        RopeParams rp;
        rp.rope_dim = p->g.rope_dim;
        rp.rope_base = p->g.rope_base;
        rp.compressed_base = p->g.comp_rope_base;
        rp.factor = p->g.yarn_factor;
        rp.original_context = p->g.yarn_orig_ctx;
        rp.beta_fast = p->g.yarn_beta_fast;
        rp.beta_slow = p->g.yarn_beta_slow;
        // the stages are window-only attention: plain rope, as the checkpoint's drafter encodes its own positions
        if (!make_rope_tables(rp, 0, p->ropes[s], err)) return nullptr;
    }

    // the kept experts: the ranking's most routed, ascending, and their router rows and biases
    const int stages = p->dp.num_stages, ne = p->dp.experts[0].n_expert;
    std::vector<std::pair<int32_t, int32_t>> ranked;
    int64_t slots = 0;
    std::string why;
    bool have_ranking = false;
    uint64_t rbytes = 0;
    if (keep_asked > 0 && keep_asked < ne) {
        if (ranking.empty()) {
            why = "no ranking";
        } else if (!basalt::platform::file_size(ranking, rbytes) || rbytes == 0) {
            why = "no ranking at " + ranking;
        } else {
            have_ranking = core::read_expert_profile(core::FileRange{ranking, 0, rbytes}, stages, ne, ranked, slots, why);
        }
    }
    const int keep = have_ranking ? keep_asked : ne;
    p->kept = dspark_kept_experts(ranked, stages, ne, keep);
    p->kept_n = keep;
    p->kept_note = std::to_string(keep) + " of " + std::to_string(ne) + " experts per stage" +
                   (why.empty() ? "" : " (" + why + ")");

    const kernels::NativeExpertLayout layout =
        kernels::native_expert_layout(kMxfp4Type, kMxfp4Type, p->g.n_embd, p->g.n_ff_exp, p->g.swiglu_limit);
    const uint64_t gu_bytes = (uint64_t) p->g.n_ff_exp * layout.gu_row;
    const uint64_t up_bytes = (uint64_t) p->g.n_ff_exp * layout.up_row;
    const uint64_t d_bytes = (uint64_t) p->g.n_embd * layout.d_row;
    for (int s = 0; s < stages; ++s) {
        const DsparkExpertStage& ex = p->dp.experts[(size_t) s];
        const int n = (int) p->kept[(size_t) s].size();
        // one packed buffer per matrix: the block's kernels index expert e at e * matrix bytes (+16: the MXFP4 dots
        // read a block with one 16-byte load, three bytes past the last one)
        void* g_base = p->alloc_dev((size_t) n * gu_bytes + 16);
        void* u_base = p->alloc_dev((size_t) n * up_bytes + 16);
        void* d_base = p->alloc_dev((size_t) n * d_bytes + 16);
        if (!g_base || !u_base || !d_base) {
            err = "deepseek4 dspark: out of device memory for the kept experts";
            return nullptr;
        }
        p->gate.push_back(g_base);
        p->up.push_back(u_base);
        p->down.push_back(d_base);
        for (int k = 0; k < n; ++k) {
            const int e = p->kept[(size_t) s][(size_t) k];
            const auto queue = [&](uint8_t* dst, uint64_t at, uint64_t bytes) {
                core::LoadItem it;
                it.file = file_handle;
                it.off = at;
                it.bytes = bytes;
                it.device = p->device;
                it.dev = dst;
                reads.add(it);
            };
            const uint64_t base = ex.offset + (uint64_t) e * ex.blob_bytes;
            queue(static_cast<uint8_t*>(g_base) + (size_t) k * gu_bytes, base, gu_bytes);
            queue(static_cast<uint8_t*>(u_base) + (size_t) k * up_bytes, base + layout.up_off, up_bytes);
            queue(static_cast<uint8_t*>(d_base) + (size_t) k * d_bytes, base + layout.down_off, d_bytes);
        }
        // the kept router rows and biases, from the stage's F32 gate and bias tensors
        const auto gate_t = p->dp.tensors.find("stage." + std::to_string(s) + ".ffn_gate_inp.weight");
        const auto bias_t = p->dp.tensors.find("stage." + std::to_string(s) + ".exp_probs_b.bias");
        if (gate_t == p->dp.tensors.end() || bias_t == p->dp.tensors.end()) {
            err = "deepseek4 dspark: the stage's router tensors are missing";
            return nullptr;
        }
        float* kg = static_cast<float*>(p->alloc_dev((size_t) n * p->g.n_embd * sizeof(float)));
        float* kb = static_cast<float*>(p->alloc_dev((size_t) n * sizeof(float)));
        if (!kg || !kb) {
            err = "deepseek4 dspark: out of device memory for the kept routers";
            return nullptr;
        }
        p->kept_gate.push_back(kg);
        p->kept_bias.push_back(kb);
        for (int k = 0; k < n; ++k) {
            const int e = p->kept[(size_t) s][(size_t) k];
            core::LoadItem g_it;
            g_it.file = file_handle;
            g_it.off = gate_t->second.offset + (uint64_t) e * p->g.n_embd * sizeof(float);
            g_it.bytes = (uint64_t) p->g.n_embd * sizeof(float);
            g_it.device = p->device;
            g_it.dev = kg + (size_t) k * p->g.n_embd;
            reads.add(g_it);
            core::LoadItem b_it;
            b_it.file = file_handle;
            b_it.off = bias_t->second.offset + (uint64_t) e * sizeof(float);
            b_it.bytes = sizeof(float);
            b_it.device = p->device;
            b_it.dev = kb + k;
            reads.add(b_it);
        }
    }

    // everything else the block reads, and the Markov head's first matrix, all on the card
    for (const auto& [name, t] : p->dp.tensors) {
        if (name.find(".ffn_gate_inp.weight") != std::string::npos ||
            name.find(".exp_probs_b.bias") != std::string::npos)
            continue;   // the kept subsets replace them
        if (name == "markov_w1.weight") {
            if (cudaMalloc((void**) &p->d_markov_w1, t.bytes) != cudaSuccess) {
                err = "deepseek4 dspark: out of device memory for the Markov head";
                return nullptr;
            }
            p->dev_bytes += t.bytes;
            core::LoadItem it;
            it.file = file_handle;
            it.off = t.offset;
            it.bytes = t.bytes;
            it.device = p->device;
            it.dev = p->d_markov_w1;
            reads.add(it);
            continue;
        }
        void* dptr = nullptr;
        if (!ck(cudaMalloc(&dptr, t.bytes), ("alloc " + name).c_str(), err)) return nullptr;
        p->dev[name] = {dptr, t.type, t.bytes};
        p->dev_bytes += t.bytes;
        core::LoadItem it;
        it.file = file_handle;
        it.off = t.offset;
        it.bytes = t.bytes;
        it.device = p->device;
        it.dev = dptr;
        reads.add(it);
    }

    // windows and scratch
    const int B = p->dp.num_stages > 0 ? std::min(kDsparkBlock, p->dp.block_size) : kDsparkBlock;
    const size_t H = (size_t) p->g.n_embd;
    p->windows.resize((size_t) std::max(1, max_windows));
    for (WindowState& w : p->windows) {
        const size_t bytes = (size_t) p->dp.num_stages * kDsparkWinCap * 512 * sizeof(float);
        if (!ck(cudaMalloc(&w.rings, bytes), "alloc rings", err) ||
            !ck(cudaMemset(w.rings, 0, bytes), "zero rings", err))
            return nullptr;
        p->dev_bytes += bytes;
    }
    const size_t mqa_ws_floats = kernels::sparse_mqa_workspace_floats(B, 0);
    p->d_streams = static_cast<float*>(p->alloc_dev((size_t) B * 4 * H * sizeof(float)));
    p->d_next_streams = static_cast<float*>(p->alloc_dev((size_t) B * 4 * H * sizeof(float)));
    p->d_hc_ws = static_cast<float*>(p->alloc_dev(kernels::hc_mixes_workspace_floats(B) * sizeof(float)));
    p->d_coef = static_cast<float*>(p->alloc_dev((size_t) B * kernels::kHcCoef * sizeof(float)));
    p->d_collapsed = static_cast<float*>(p->alloc_dev((size_t) B * H * sizeof(float)));
    p->d_normed = static_cast<float*>(p->alloc_dev((size_t) B * H * sizeof(float)));
    p->d_q8 = p->alloc_dev(kernels::native_q8_1_bytes((int) H, 8));
    p->d_ql = static_cast<float*>(p->alloc_dev((size_t) B * 1024 * sizeof(float)));
    p->d_ql_q8 = p->alloc_dev(kernels::native_q8_1_bytes(1024, 8));
    p->d_qh = static_cast<float*>(p->alloc_dev((size_t) B * 64 * 512 * sizeof(float)));
    p->d_kv_proj = static_cast<float*>(p->alloc_dev((size_t) 8 * 512 * sizeof(float)));
    p->d_kv_row = static_cast<float*>(p->alloc_dev((size_t) 8 * 512 * sizeof(float)));
    p->d_mqa_out = static_cast<float*>(p->alloc_dev((size_t) B * 64 * 512 * sizeof(float)));
    p->d_mqa_ws = static_cast<float*>(p->alloc_dev(mqa_ws_floats * sizeof(float)));
    p->d_out_q8 = p->alloc_dev(kernels::sparse_mqa_output_scratch_bytes(B));
    p->d_out_lat = static_cast<float*>(p->alloc_dev(kernels::sparse_mqa_output_workspace_floats(B) * sizeof(float)));
    p->d_attn_out = static_cast<float*>(p->alloc_dev((size_t) B * H * sizeof(float)));
    p->d_logits_route = static_cast<float*>(p->alloc_dev((size_t) B * p->kept_n * sizeof(float)));
    p->d_route_ids = static_cast<int32_t*>(p->alloc_dev((size_t) B * p->g.n_used * sizeof(int32_t)));
    p->d_route_w = static_cast<float*>(p->alloc_dev((size_t) B * p->g.n_used * sizeof(float)));
    p->d_shared = static_cast<float*>(p->alloc_dev((size_t) B * H * sizeof(float)));
    p->d_shared_gate = static_cast<float*>(p->alloc_dev((size_t) B * p->g.n_ff_exp * sizeof(float)));
    p->d_shared_up = static_cast<float*>(p->alloc_dev((size_t) B * p->g.n_ff_exp * sizeof(float)));
    p->d_shared_hidden = static_cast<float*>(p->alloc_dev((size_t) B * p->g.n_ff_exp * sizeof(float)));
    p->d_shared_hq8 = p->alloc_dev(kernels::native_q8_1_bytes(p->g.n_ff_exp, B));
    p->d_routed_rows_ = static_cast<float*>(p->alloc_dev((size_t) B * H * sizeof(float)));
    p->d_pos = static_cast<int32_t*>(p->alloc_dev(8 * sizeof(int32_t)));
    p->d_wstart = static_cast<int32_t*>(p->alloc_dev(8 * sizeof(int32_t)));
    p->d_wn = static_cast<int32_t*>(p->alloc_dev(8 * sizeof(int32_t)));
    p->d_feat_q8 = p->alloc_dev(kernels::native_q8_1_bytes(3 * (int) H, 8));
    p->d_feat_proj = static_cast<float*>(p->alloc_dev(8 * H * sizeof(float)));
    p->d_feat_norm = static_cast<float*>(p->alloc_dev(8 * H * sizeof(float)));
    p->d_head_hc_ws = static_cast<float*>(p->alloc_dev(kernels::hc_plain_workspace_floats(B) * sizeof(float)));
    p->d_head_coef = static_cast<float*>(p->alloc_dev((size_t) B * 4 * sizeof(float)));
    p->d_head_col = static_cast<float*>(p->alloc_dev((size_t) B * H * sizeof(float)));
    p->d_head_norm = static_cast<float*>(p->alloc_dev((size_t) B * H * sizeof(float)));
    p->d_base_logits = static_cast<float*>(p->alloc_dev((size_t) B * p->g.n_vocab * sizeof(float)));
    p->d_me = static_cast<float*>(p->alloc_dev((size_t) p->dp.markov_rank * sizeof(float)));
    p->d_me_q8 = p->alloc_dev(kernels::native_q8_1_bytes(p->dp.markov_rank, 1));
    p->d_mbias = static_cast<float*>(p->alloc_dev((size_t) p->g.n_vocab * sizeof(float)));
    p->d_stage_pos = static_cast<int32_t*>(p->alloc_dev(8 * sizeof(int32_t)));
    p->d_amax_v = static_cast<float*>(p->alloc_dev(kArgmaxBlocks * sizeof(float)));
    p->d_amax_i = static_cast<int32_t*>(p->alloc_dev(kArgmaxBlocks * sizeof(int32_t)));
    p->d_tok = static_cast<int32_t*>(p->alloc_dev((B + 1) * sizeof(int32_t)));
    p->d_conf = static_cast<float*>(p->alloc_dev(B * sizeof(float)));
    p->d_moe_h = static_cast<float*>(p->alloc_dev((size_t) B * p->g.n_used * p->g.n_ff_exp * sizeof(float)));
    p->d_moe_d = static_cast<float*>(p->alloc_dev((size_t) B * p->g.n_used * H * sizeof(float)));
    p->d_moe_hq = p->alloc_dev(kernels::native_q8_1_bytes(p->g.n_ff_exp, B * p->g.n_used));
    p->d_staged = static_cast<float*>(p->alloc_dev((size_t) p->dp.num_stages * 8 * 512 * sizeof(float)));
    if (!p->d_streams || !p->d_base_logits || !p->d_conf || !p->d_moe_h || !p->d_moe_d || !p->d_staged ||
        !ck(cudaHostAlloc(&p->h_init, (size_t) B * 4 * H * sizeof(float), cudaHostAllocDefault), "h_init", err) ||
        !ck(cudaHostAlloc(&p->h_tok, (B + 1) * sizeof(int32_t), cudaHostAllocDefault), "h_tok", err) ||
        !ck(cudaHostAlloc(&p->h_conf, B * sizeof(float), cudaHostAllocDefault), "h_conf", err)) {
        if (err.empty()) err = "deepseek4 dspark: out of device memory for drafter scratch";
        return nullptr;
    }
    basalt::log::info("basalt deepseek4",
                      "dspark drafter on GPU %d: %.2f GiB, %s, block %d, windows of at most %d tokens (--spec 0: off)",
                      p->device, (double) p->dev_bytes / (1ull << 30), p->kept_note.c_str(), B, kDsparkBlock + 1);
    return std::unique_ptr<DsparkDrafter>(new DsparkDrafter(std::move(p)));
}

int DsparkDrafter::open_window() {
    if (p_->next_win >= (int) p_->windows.size()) return -1;
    return p_->next_win++;
}

int DsparkDrafter::feature_slot(int layer) const {
    for (size_t i = 0; i < p_->dp.target_layer_ids.size(); ++i)
        if (p_->dp.target_layer_ids[i] == layer) return (int) i;
    return -1;
}

int DsparkDrafter::feature_floats() const { return (int) p_->dp.target_layer_ids.size() * p_->g.n_embd; }
int DsparkDrafter::block_size() const { return std::min(kDsparkBlock, p_->dp.block_size); }
int DsparkDrafter::n_stages() const { return p_->dp.num_stages; }
int DsparkDrafter::kept_per_stage() const { return p_->kept_n; }
uint64_t DsparkDrafter::device_bytes() const { return p_->dev_bytes; }
uint64_t DsparkDrafter::host_bytes() const { return 0; }
const std::string& DsparkDrafter::kept() const { return p_->kept_note; }

void DsparkDrafter::forget_from(int w, int64_t pos) {
    if (w >= 0 && w < (int) p_->windows.size() && pos < p_->windows[(size_t) w].valid_end)
        p_->windows[(size_t) w].valid_end = std::max<int64_t>(0, pos);
}

void DsparkDrafter::reset(int w) {
    if (w < 0 || w >= (int) p_->windows.size()) return;
    p_->windows[(size_t) w].valid_end = 0;
    if (p_->staged_w == w) {
        p_->staged_w = -1;
        p_->staged_n = 0;
        p_->staged_pending = false;
    }
}

bool DsparkDrafter::inject_device(int w, int64_t pos0, int n_rows, const float* d_features, void* stream, std::string& err) {
    if (w < 0 || w >= (int) p_->windows.size() || n_rows < 1 || n_rows > 8 || !d_features) return true;
    const DevTensor *mproj = p_->req("main_proj.weight", err), *mnorm = p_->req("main_norm.weight", err);
    if (!mproj || !mnorm) return false;
    const cudaStream_t s = (cudaStream_t) stream;
    if (p_->staged_pending) {   // the staging shares this scratch
        if (!ck(cudaStreamWaitEvent(s, p_->ev_staged, 0), "inject waits the staging", err)) return false;
    }
    const int H = p_->g.n_embd;
    const int in_cols = feature_floats();
    int32_t hp[8]{};
    for (int r = 0; r < n_rows; ++r) hp[r] = (int32_t) (pos0 + r);
    if (!ck(cudaMemcpyAsync(p_->d_pos, hp, (size_t) n_rows * sizeof(int32_t), cudaMemcpyHostToDevice, s), "inject pos", err))
        return false;
    kernels::native_quantize_q8_1(d_features, p_->d_feat_q8, in_cols, n_rows, s);
    kernels::native_mmvq(mproj->type, mproj->ptr, p_->d_feat_q8, p_->d_feat_proj, in_cols, H, n_rows, s);
    kernels::rms_norm_rows(p_->d_feat_proj, static_cast<const float*>(mnorm->ptr), n_rows, H, p_->g.rms_eps,
                           p_->d_feat_norm, s);
    WindowState& ws = p_->windows[(size_t) w];
    for (int st = 0; st < p_->dp.num_stages; ++st) {
        const std::string pre = "stage." + std::to_string(st) + ".";
        const DevTensor *wkv = p_->req(pre + "attn_kv.weight", err), *kvn = p_->req(pre + "attn_kv_a_norm.weight", err);
        if (!wkv || !kvn) return false;
        project_attention_kv(p_->d_feat_norm, wkv->type, wkv->ptr, static_cast<const float*>(kvn->ptr), p_->g.rms_eps,
                             p_->d_pos, p_->ropes[st].attention, n_rows, p_->d_q8, p_->d_kv_proj, p_->d_kv_row, s);
        cache_attention_window_f32(ws.rings + (size_t) st * kDsparkWinCap * 512, p_->d_kv_row, p_->d_pos, n_rows, s);
    }
    ws.valid_end = pos0 + n_rows;
    if (!ck(cudaEventRecord(p_->ev_free, s), "inject done", err)) return false;
    return true;
}

bool DsparkDrafter::stage(int w, int64_t pos0, int n_rows, const float* d_features, void* ready, std::string& err) {
    if (w < 0 || w >= (int) p_->windows.size() || n_rows < 1 || n_rows > 8 || !d_features) return false;
    const DevTensor *mproj = p_->req("main_proj.weight", err), *mnorm = p_->req("main_norm.weight", err);
    if (!mproj || !mnorm) return false;
    const cudaStream_t s = p_->stage_st;
    if (ready && !ck(cudaStreamWaitEvent(s, (cudaEvent_t) ready, 0), "stage waits features", err)) return false;
    if (!ck(cudaStreamWaitEvent(s, p_->ev_free, 0), "stage waits scratch", err)) return false;
    if (p_->staged_pending && !ck(cudaStreamWaitEvent(s, p_->ev_staged, 0), "stage waits prior", err)) return false;
    const int H = p_->g.n_embd;
    const int in_cols = feature_floats();
    int32_t hp[8]{};
    for (int r = 0; r < n_rows; ++r) hp[r] = (int32_t) (pos0 + r);
    if (!ck(cudaMemcpyAsync(p_->d_stage_pos, hp, (size_t) n_rows * sizeof(int32_t), cudaMemcpyHostToDevice, s),
            "stage pos", err))
        return false;
    kernels::native_quantize_q8_1(d_features, p_->d_feat_q8, in_cols, n_rows, s);
    kernels::native_mmvq(mproj->type, mproj->ptr, p_->d_feat_q8, p_->d_feat_proj, in_cols, H, n_rows, s);
    kernels::rms_norm_rows(p_->d_feat_proj, static_cast<const float*>(mnorm->ptr), n_rows, H, p_->g.rms_eps,
                           p_->d_feat_norm, s);
    for (int st = 0; st < p_->dp.num_stages; ++st) {
        const std::string pre = "stage." + std::to_string(st) + ".";
        const DevTensor *wkv = p_->req(pre + "attn_kv.weight", err), *kvn = p_->req(pre + "attn_kv_a_norm.weight", err);
        if (!wkv || !kvn) return false;
        project_attention_kv(p_->d_feat_norm, wkv->type, wkv->ptr, static_cast<const float*>(kvn->ptr), p_->g.rms_eps,
                             p_->d_stage_pos, p_->ropes[st].attention, n_rows, p_->d_q8, p_->d_kv_proj, p_->d_kv_row, s);
        // the row's latent, kept in the staging set at slot pos % 8
        for (int r = 0; r < n_rows; ++r)
            if (!ck(cudaMemcpyAsync(p_->d_staged + ((size_t) st * 8 + (size_t) ((pos0 + r) % 8)) * 512,
                                    p_->d_kv_row + (size_t) r * 512, 512 * sizeof(float), cudaMemcpyDeviceToDevice, s),
                    "stage rows", err))
                return false;
    }
    if (!ck(cudaEventRecord(p_->ev_staged, s), "staged", err)) return false;
    p_->staged_w = w;
    p_->staged_pos0 = pos0;
    p_->staged_n = n_rows;
    p_->staged_pending = true;
    return true;
}

bool DsparkDrafter::commit(int w, int64_t pos0, int n_rows, std::string& err) {
    if (!p_->staged_pending || p_->staged_w != w || p_->staged_pos0 != pos0 || n_rows > p_->staged_n || n_rows < 0)
        return false;
    WindowState& ws = p_->windows[(size_t) w];
    if (!ck(cudaStreamWaitEvent(p_->stream, p_->ev_staged, 0), "commit waits staging", err)) return false;
    if (n_rows > 0)
        dspark_commit_rows(p_->d_staged, 8, ws.rings, ws.rings + (size_t) kDsparkWinCap * 512,
                           ws.rings + (size_t) 2 * kDsparkWinCap * 512, kDsparkWindow, pos0, n_rows, p_->stream);
    ws.valid_end = std::max<int64_t>(ws.valid_end, pos0 + n_rows);
    p_->staged_pending = false;
    p_->staged_w = -1;
    if (!ck(cudaEventRecord(p_->ev_free, p_->stream), "commit done", err)) return false;
    return true;
}

bool DsparkDrafter::draft_begin(int w, int32_t anchor, int64_t pos, int n_out, void* caller_stream, std::string& err) {
    if (w < 0 || w >= (int) p_->windows.size()) {
        err = "deepseek4 dspark: invalid window";
        return false;
    }
    const int B = std::clamp(n_out, 1, block_size());
    const int n = std::clamp(n_out, 1, B);
    if (!p_->build(err)) {
        if (err.empty()) err = "deepseek4 dspark: cannot build the stage weights";
        return false;
    }
    if (!p_->upload_inputs(anchor, pos, B, err)) return false;
    if (n_out >= 0) {
        const int32_t hp[1] = {anchor};
        if (!ck(cudaMemcpyAsync(p_->d_tok, hp, sizeof(int32_t), cudaMemcpyHostToDevice, p_->stream), "anchor", err))
            return false;
    }
    if (caller_stream) {
        if (!ck(cudaEventRecord(p_->ready_ev, (cudaStream_t) caller_stream), "record caller", err) ||
            !ck(cudaStreamWaitEvent(p_->stream, p_->ready_ev, 0), "wait caller", err))
            return false;
    }
    if (p_->body_exec.size() <= (size_t) w) p_->body_exec.resize((size_t) w + 1);
    if (p_->body_exec[(size_t) w].size() <= (size_t) B) p_->body_exec[(size_t) w].resize((size_t) B + 1);
    if (!p_->launch(p_->body_exec[(size_t) w][(size_t) B], [&] { p_->run_body(w, B); }, err)) return false;
    if (!p_->launch(p_->head_exec[n], [&] { p_->run_head(n); }, err)) return false;
    if (!ck(cudaMemcpyAsync(p_->h_tok + 1, p_->d_tok + 1, (size_t) n * sizeof(int32_t), cudaMemcpyDeviceToHost,
                            p_->stream), "tokens D2H", err) ||
        !ck(cudaMemcpyAsync(p_->h_conf, p_->d_conf, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost, p_->stream),
            "conf D2H", err) ||
        !ck(cudaEventRecord(p_->ev_done, p_->stream), "block done", err))
        return false;
    p_->pending_n = n;
    return true;
}

bool DsparkDrafter::draft_end(DsparkDraft& out, std::string& err) {
    out = {};
    if (!ck(cudaEventSynchronize(p_->ev_done), "block wait", err)) return false;
    out.n = p_->pending_n;
    for (int i = 0; i < out.n; ++i) {
        out.token[i] = p_->h_tok[i + 1];
        out.conf[i] = p_->h_conf[i];
    }
    return true;
}

}  // namespace basalt::models::deepseek4
