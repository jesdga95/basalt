#include "models/deepseek4/dspark.hpp"

#include "basalt/core/expert_tier.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/dense_ops.hpp"
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
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <unordered_map>

namespace basalt::models::deepseek4 {
namespace {

constexpr int kTypeQ8_0 = 8;
constexpr int kTypeBF16 = 30;
constexpr int kTypeMXFP4 = 39;
constexpr int kArgmaxBlocks = 256;

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
    float* rings = nullptr;  ///< [n_stages, kDsparkWinCap, 512] FP32 SWA KV rings
    int64_t valid_end = 0;
};

}  // namespace

struct DsparkDrafter::Impl {
    const DsModel* main = nullptr;
    Geometry g;
    DsparkPack dp;
    int device = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready_ev = nullptr;

    std::map<std::string, DevTensor> dev;
    uint64_t dev_bytes = 0;
    std::vector<uint16_t> markov_w1;
    std::vector<uint8_t> experts_ram;
    kernels::cpu::NativeFmt nat_fmt{};
    RopeTables rope{};

    std::vector<WindowState> windows;
    int next_win = 0;

    // Device scratch for block pass (B <= kDsparkBlock)
    std::vector<void*> dev_allocs;
    float *d_streams = nullptr, *d_next_streams = nullptr;
    float *d_hc_ws = nullptr, *d_coef = nullptr, *d_collapsed = nullptr, *d_normed = nullptr;
    void *d_q8 = nullptr, *d_ql_q8 = nullptr, *d_out_q8 = nullptr;
    float *d_ql = nullptr, *d_qh = nullptr, *d_kv_proj = nullptr, *d_kv_row = nullptr;
    float *d_mqa_out = nullptr, *d_mqa_ws = nullptr, *d_out_lat = nullptr, *d_attn_out = nullptr;
    float *d_logits_route = nullptr, *d_route_w = nullptr, *d_shared = nullptr, *d_routed_rows = nullptr, *d_routed = nullptr;
    float *d_shared_gate = nullptr, *d_shared_up = nullptr, *d_shared_hidden = nullptr;
    void *d_shared_hq8 = nullptr;
    int32_t *d_route_ids = nullptr, *d_pos = nullptr, *d_wstart = nullptr, *d_wn = nullptr;
    float *d_feat_proj = nullptr, *d_feat_norm = nullptr;
    void *d_feat_q8 = nullptr;
    float *d_head_hc_ws = nullptr, *d_head_coef = nullptr, *d_head_col = nullptr, *d_head_norm = nullptr;
    float *d_base_logits = nullptr, *d_me = nullptr, *d_mbias = nullptr, *d_amax_v = nullptr, *d_conf = nullptr;
    void *d_me_q8 = nullptr;
    int32_t *d_amax_i = nullptr, *d_tok = nullptr;

    // Host staging
    float *h_init = nullptr, *h_x_norm = nullptr, *h_routed_rows = nullptr, *h_me = nullptr, *h_conf = nullptr;
    int32_t *h_route_ids = nullptr, *h_tok = nullptr;
    std::vector<uint8_t> h_nact;

    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        for (WindowState& w : windows)
            if (w.rings) cudaFree(w.rings);
        for (auto& [_, t] : dev)
            if (t.ptr) cudaFree(t.ptr);
        for (void* p : dev_allocs) cudaFree(p);
        if (h_init) cudaFreeHost(h_init);
        if (h_x_norm) cudaFreeHost(h_x_norm);
        if (h_routed_rows) cudaFreeHost(h_routed_rows);
        if (h_me) cudaFreeHost(h_me);
        if (h_conf) cudaFreeHost(h_conf);
        if (h_route_ids) cudaFreeHost(h_route_ids);
        if (h_tok) cudaFreeHost(h_tok);
        if (ready_ev) cudaEventDestroy(ready_ev);
        if (stream) cudaStreamDestroy(stream);
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

    bool run_cpu_experts(int stage, int B, const float* d_x, const int32_t* d_ids, const float* d_w,
                         float* d_out, std::string& err) {
        core::ExpertTier* tier = main->tier();
        kernels::cpu::ExpertPool* pool = tier ? tier->pool() : nullptr;
        if (!pool) {
            err = "deepseek4 dspark: CPU expert pool unavailable";
            return false;
        }
        const int H = g.n_embd;
        const int K = g.n_used;
        const int n_ids = B * K;
        if (!ck(cudaMemcpyAsync(h_x_norm, d_x, (size_t) B * H * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "dspark expert x D2H", err) ||
            !ck(cudaMemcpyAsync(h_route_ids, d_ids, (size_t) n_ids * sizeof(int32_t), cudaMemcpyDeviceToHost, stream),
                "dspark expert ids D2H", err) ||
            !ck(cudaStreamSynchronize(stream), "dspark expert D2H sync", err))
            return false;

        for (int t = 0; t < B; ++t)
            kernels::cpu::native_quant_act(nat_fmt, h_x_norm + (size_t) t * H,
                                           h_nact.data() + (size_t) t * nat_fmt.act_bytes);

        struct Slot {
            int n = 0;
            int tok[kernels::cpu::MAXT] = {};
            int row[kernels::cpu::MAXT] = {};
        };
        std::unordered_map<int, Slot> by_exp;
        for (int r = 0; r < n_ids; ++r) {
            const int e = h_route_ids[r];
            if (e < 0 || e >= g.n_expert) continue;
            Slot& sl = by_exp[e];
            if (sl.n < kernels::cpu::MAXT) {
                sl.tok[sl.n] = r / K;
                sl.row[sl.n] = r;
                sl.n++;
            }
        }
        const uint64_t blob_bytes = dp.experts[(size_t) stage].blob_bytes;
        const uint8_t* stage_base = experts_ram.data() + (size_t) stage * (size_t) g.n_expert * blob_bytes;
        std::array<kernels::cpu::ExpertJobMulti, 64> jobs{};
        int n_jobs = 0;
        for (const auto& [e, sl] : by_exp) {
            if (n_jobs >= (int) jobs.size()) break;
            auto& jb = jobs[(size_t) n_jobs++];
            jb.blob = stage_base + (size_t) e * blob_bytes;
            jb.nt = sl.n;
            for (int j = 0; j < sl.n; ++j) {
                jb.nact[j] = h_nact.data() + (size_t) sl.tok[j] * nat_fmt.act_bytes;
                jb.out[j] = h_routed_rows + (size_t) sl.row[j] * H;
            }
        }
        if (n_jobs > 0) {
            pool->run_split_multi_native(nat_fmt, jobs.data(), n_jobs);
        }
        if (!ck(cudaMemcpyAsync(d_routed_rows, h_routed_rows, (size_t) n_ids * H * sizeof(float),
                                cudaMemcpyHostToDevice, stream), "dspark routed rows H2D", err))
            return false;
        weighted_route_sum(d_routed_rows, d_w, B, H, d_out, stream);
        return true;
    }
};

DsparkDrafter::DsparkDrafter(std::unique_ptr<Impl> impl) : p_(std::move(impl)) {}
DsparkDrafter::~DsparkDrafter() = default;

std::unique_ptr<DsparkDrafter> DsparkDrafter::plan_load(const DsModel& main, int file_handle, core::LoadPlan& reads,
                                                        int max_windows, std::string& err) {
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
        !ck(cudaEventCreateWithFlags(&p->ready_ev, cudaEventDisableTiming), "create event", err))
        return nullptr;

    if (!kernels::cpu::native_fmt(kTypeMXFP4, kTypeMXFP4, p->g.n_embd, p->g.n_ff_exp, p->nat_fmt, err)) {
        err = "deepseek4 dspark: " + err;
        return nullptr;
    }
    p->nat_fmt.swiglu_limit = p->g.swiglu_limit;
    RopeParams rp;
    rp.rope_dim = p->g.rope_dim;
    rp.rope_base = p->g.rope_base;
    rp.compressed_base = p->g.comp_rope_base;
    rp.factor = p->g.yarn_factor;
    rp.original_context = p->g.yarn_orig_ctx;
    rp.beta_fast = p->g.yarn_beta_fast;
    rp.beta_slow = p->g.yarn_beta_slow;
    if (!make_rope_tables(rp, 0, p->rope, err)) return nullptr;

    // 1. Plan dense tensors: markov_w1 to host RAM, everything else to GPU 0
    for (const auto& [name, t] : p->dp.tensors) {
        if (name == "markov_w1.weight") {
            p->markov_w1.resize(t.bytes / 2);
            core::LoadItem it;
            it.file = file_handle;
            it.off = t.offset;
            it.bytes = t.bytes;
            it.host = reinterpret_cast<uint8_t*>(p->markov_w1.data());
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

    // 2. Plan 3-stage MXFP4 experts to host RAM
    const uint64_t stage_bytes = (uint64_t) p->g.n_expert * p->dp.experts[0].blob_bytes;
    p->experts_ram.resize((size_t) p->dp.num_stages * stage_bytes);
    for (int s = 0; s < p->dp.num_stages; ++s) {
        core::LoadItem it;
        it.file = file_handle;
        it.off = p->dp.experts[(size_t) s].offset;
        it.bytes = stage_bytes;
        it.host = p->experts_ram.data() + (size_t) s * stage_bytes;
        reads.add(it);
    }

    // 3. Allocate windows & scratch
    const int B = std::min(kDsparkBlock, p->dp.block_size);
    const size_t H = (size_t) p->g.n_embd;
    p->windows.resize((size_t) std::max(1, max_windows));
    for (WindowState& w : p->windows) {
        const size_t rbytes = (size_t) p->dp.num_stages * kDsparkWinCap * 512 * sizeof(float);
        if (!ck(cudaMalloc(&w.rings, rbytes), "alloc rings", err) ||
            !ck(cudaMemset(w.rings, 0, rbytes), "zero rings", err))
            return nullptr;
        p->dev_bytes += rbytes;
    }

    const size_t mqa_ws_floats = kernels::sparse_mqa_workspace_floats(B, 0);
    p->d_streams = static_cast<float*>(p->alloc_dev((size_t) B * 4 * H * sizeof(float)));
    p->d_next_streams = static_cast<float*>(p->alloc_dev((size_t) B * 4 * H * sizeof(float)));
    p->d_hc_ws = static_cast<float*>(p->alloc_dev(kernels::hc_mixes_workspace_floats(B) * sizeof(float)));
    p->d_coef = static_cast<float*>(p->alloc_dev((size_t) B * kernels::kHcCoef * sizeof(float)));
    p->d_collapsed = static_cast<float*>(p->alloc_dev((size_t) B * H * sizeof(float)));
    p->d_normed = static_cast<float*>(p->alloc_dev((size_t) B * H * sizeof(float)));
    p->d_q8 = p->alloc_dev(kernels::native_q8_1_bytes(H, B));
    p->d_ql = static_cast<float*>(p->alloc_dev((size_t) B * 1024 * sizeof(float)));
    p->d_ql_q8 = p->alloc_dev(kernels::native_q8_1_bytes(1024, B));
    p->d_qh = static_cast<float*>(p->alloc_dev((size_t) B * 64 * 512 * sizeof(float)));
    p->d_kv_proj = static_cast<float*>(p->alloc_dev((size_t) B * 512 * sizeof(float)));
    p->d_kv_row = static_cast<float*>(p->alloc_dev((size_t) B * 512 * sizeof(float)));
    p->d_mqa_out = static_cast<float*>(p->alloc_dev((size_t) B * 64 * 512 * sizeof(float)));
    p->d_mqa_ws = static_cast<float*>(p->alloc_dev(mqa_ws_floats * sizeof(float)));
    p->d_out_q8 = p->alloc_dev(kernels::sparse_mqa_output_scratch_bytes(B));
    p->d_out_lat = static_cast<float*>(p->alloc_dev(kernels::sparse_mqa_output_workspace_floats(B) * sizeof(float)));
    p->d_attn_out = static_cast<float*>(p->alloc_dev((size_t) B * H * sizeof(float)));
    p->d_logits_route = static_cast<float*>(p->alloc_dev((size_t) B * p->g.n_expert * sizeof(float)));
    p->d_route_ids = static_cast<int32_t*>(p->alloc_dev((size_t) B * p->g.n_used * sizeof(int32_t)));
    p->d_route_w = static_cast<float*>(p->alloc_dev((size_t) B * p->g.n_used * sizeof(float)));
    p->d_shared = static_cast<float*>(p->alloc_dev((size_t) B * H * sizeof(float)));
    p->d_shared_gate = static_cast<float*>(p->alloc_dev((size_t) B * p->g.n_ff_exp * sizeof(float)));
    p->d_shared_up = static_cast<float*>(p->alloc_dev((size_t) B * p->g.n_ff_exp * sizeof(float)));
    p->d_shared_hidden = static_cast<float*>(p->alloc_dev((size_t) B * p->g.n_ff_exp * sizeof(float)));
    p->d_shared_hq8 = p->alloc_dev(kernels::native_q8_1_bytes(p->g.n_ff_exp, B));
    p->d_routed_rows = static_cast<float*>(p->alloc_dev((size_t) B * p->g.n_used * H * sizeof(float)));
    p->d_routed = static_cast<float*>(p->alloc_dev((size_t) B * H * sizeof(float)));
    p->d_pos = static_cast<int32_t*>(p->alloc_dev(8 * sizeof(int32_t)));
    p->d_wstart = static_cast<int32_t*>(p->alloc_dev(8 * sizeof(int32_t)));
    p->d_wn = static_cast<int32_t*>(p->alloc_dev(8 * sizeof(int32_t)));
    p->d_q8 = p->alloc_dev(kernels::native_q8_1_bytes(H, 8));
    p->d_kv_proj = static_cast<float*>(p->alloc_dev(8 * 512 * sizeof(float)));
    p->d_kv_row = static_cast<float*>(p->alloc_dev(8 * 512 * sizeof(float)));
    p->d_feat_q8 = p->alloc_dev(kernels::native_q8_1_bytes(3 * H, 8));
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
    p->d_amax_v = static_cast<float*>(p->alloc_dev(kArgmaxBlocks * sizeof(float)));
    p->d_amax_i = static_cast<int32_t*>(p->alloc_dev(kArgmaxBlocks * sizeof(int32_t)));
    p->d_tok = static_cast<int32_t*>(p->alloc_dev(sizeof(int32_t)));
    p->d_conf = static_cast<float*>(p->alloc_dev(sizeof(float)));

    if (!p->d_streams || !p->d_base_logits || !p->d_conf ||
        !ck(cudaHostAlloc(&p->h_init, (size_t) B * 4 * H * sizeof(float), cudaHostAllocDefault), "h_init", err) ||
        !ck(cudaHostAlloc(&p->h_x_norm, (size_t) B * H * sizeof(float), cudaHostAllocDefault), "h_x_norm", err) ||
        !ck(cudaHostAlloc(&p->h_routed_rows, (size_t) B * p->g.n_used * H * sizeof(float), cudaHostAllocDefault), "h_routed", err) ||
        !ck(cudaHostAlloc(&p->h_route_ids, (size_t) B * p->g.n_used * sizeof(int32_t), cudaHostAllocDefault), "h_rids", err) ||
        !ck(cudaHostAlloc(&p->h_me, (size_t) p->dp.markov_rank * sizeof(float), cudaHostAllocDefault), "h_me", err) ||
        !ck(cudaHostAlloc(&p->h_tok, sizeof(int32_t), cudaHostAllocDefault), "h_tok", err) ||
        !ck(cudaHostAlloc(&p->h_conf, sizeof(float), cudaHostAllocDefault), "h_conf", err)) {
        if (err.empty()) err = "deepseek4 dspark: out of device memory for drafter scratch";
        return nullptr;
    }
    p->h_nact.resize((size_t) B * p->nat_fmt.act_bytes);
    basalt::log::info("basalt deepseek4", "dspark drafter: %d stages, block %d, %.2f GiB RAM, %.1f MiB VRAM",
                      p->dp.num_stages, B, (double) (p->experts_ram.size() + p->markov_w1.size() * 2) / (1ull << 30),
                      (double) p->dev_bytes / (1ull << 20));
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
uint64_t DsparkDrafter::device_bytes() const { return p_->dev_bytes; }
uint64_t DsparkDrafter::host_bytes() const { return p_->experts_ram.size() + p_->markov_w1.size() * 2; }

void DsparkDrafter::forget_from(int w, int64_t pos) {
    if (w >= 0 && w < (int) p_->windows.size() && pos < p_->windows[(size_t) w].valid_end)
        p_->windows[(size_t) w].valid_end = std::max<int64_t>(0, pos);
}

bool DsparkDrafter::inject_device(int w, int64_t pos0, int n_rows, const float* d_features, void* stream, std::string& err) {
    if (w < 0 || w >= (int) p_->windows.size() || n_rows < 1 || n_rows > 8 || !d_features) return true;
    const DevTensor *mproj = p_->req("main_proj.weight", err), *mnorm = p_->req("main_norm.weight", err);
    if (!mproj || !mnorm) return false;
    const int H = p_->g.n_embd;
    const int in_cols = feature_floats();
    int32_t hp[8]{};
    for (int r = 0; r < n_rows; ++r) hp[r] = (int32_t) (pos0 + r);
    if (!ck(cudaMemcpyAsync(p_->d_pos, hp, (size_t) n_rows * sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) stream),
            "inject pos", err))
        return false;
    kernels::native_quantize_q8_1(d_features, p_->d_feat_q8, in_cols, n_rows, stream);
    kernels::native_mmvq(mproj->type, mproj->ptr, p_->d_feat_q8, p_->d_feat_proj, in_cols, H, n_rows, stream);
    kernels::rms_norm_rows(p_->d_feat_proj, static_cast<const float*>(mnorm->ptr), n_rows, H, p_->g.rms_eps,
                           p_->d_feat_norm, stream);
    WindowState& ws = p_->windows[(size_t) w];
    for (int s = 0; s < p_->dp.num_stages; ++s) {
        const std::string pre = "stage." + std::to_string(s) + ".";
        const DevTensor *wkv = p_->req(pre + "attn_kv.weight", err),
                        *kvn = p_->req(pre + "attn_kv_a_norm.weight", err);
        if (!wkv || !kvn) return false;
        project_attention_kv(p_->d_feat_norm, wkv->type, wkv->ptr, static_cast<const float*>(kvn->ptr),
                             p_->g.rms_eps, p_->d_pos, p_->rope.attention, n_rows,
                             p_->d_q8, p_->d_kv_proj, p_->d_kv_row, stream);
        cache_attention_window(ws.rings + (size_t) s * kDsparkWinCap * 512, p_->d_kv_row, p_->d_pos, n_rows, stream);
    }
    ws.valid_end = pos0 + n_rows;
    return true;
}

bool DsparkDrafter::draft(int w, int32_t anchor, int64_t pos, int n_out, void* caller_stream,
                          DsparkDraft& out, std::string& err) {
    out = {};
    if (w < 0 || w >= (int) p_->windows.size()) {
        err = "deepseek4 dspark: invalid window";
        return false;
    }
    const int B = std::clamp(n_out, 1, block_size());
    if (caller_stream) {
        if (!ck(cudaEventRecord(p_->ready_ev, (cudaStream_t) caller_stream), "record caller", err) ||
            !ck(cudaStreamWaitEvent(p_->stream, p_->ready_ev, 0), "wait caller", err))
            return false;
    }
    const size_t H = (size_t) p_->g.n_embd;
    std::vector<float> emb;
    for (int r = 0; r < B; ++r) {
        const int32_t tok = (r == 0) ? anchor : p_->dp.noise_token_id;
        if (!p_->main->weights().embedding_row(tok, emb, err)) return false;
        for (int s = 0; s < 4; ++s)
            std::copy_n(emb.data(), H, p_->h_init + ((size_t) r * 4 + s) * H);
    }
    if (!ck(cudaMemcpyAsync(p_->d_streams, p_->h_init, (size_t) B * 4 * H * sizeof(float),
                            cudaMemcpyHostToDevice, p_->stream), "embed H2D", err))
        return false;

    int32_t hp[kDsparkBlock]{}, hws[kDsparkBlock]{}, hwn[kDsparkBlock]{};
    for (int r = 0; r < B; ++r) {
        hp[r] = (int32_t) (pos + r);
        hws[r] = (int32_t) std::max<int64_t>(0, pos + r - kDsparkWindow + 1);
        hwn[r] = hp[r] - hws[r] + 1;
    }
    if (!ck(cudaMemcpyAsync(p_->d_pos, hp, (size_t) B * sizeof(int32_t), cudaMemcpyHostToDevice, p_->stream), "pos", err) ||
        !ck(cudaMemcpyAsync(p_->d_wstart, hws, (size_t) B * sizeof(int32_t), cudaMemcpyHostToDevice, p_->stream), "ws", err) ||
        !ck(cudaMemcpyAsync(p_->d_wn, hwn, (size_t) B * sizeof(int32_t), cudaMemcpyHostToDevice, p_->stream), "wn", err))
        return false;

    WindowState& ws = p_->windows[(size_t) w];
    for (int s = 0; s < p_->dp.num_stages; ++s) {
        const std::string pre = "stage." + std::to_string(s) + ".";
        const DevTensor *hc_af = p_->req(pre + "hc_attn_fn.weight", err),
                        *hc_as = p_->req(pre + "hc_attn_scale.weight", err),
                        *hc_ab = p_->req(pre + "hc_attn_base.weight", err),
                        *anorm = p_->req(pre + "attn_norm.weight", err),
                        *wqa = p_->req(pre + "attn_q_a.weight", err),
                        *qn = p_->req(pre + "attn_q_a_norm.weight", err),
                        *wqb = p_->req(pre + "attn_q_b.weight", err),
                        *wkv = p_->req(pre + "attn_kv.weight", err),
                        *kvn = p_->req(pre + "attn_kv_a_norm.weight", err),
                        *sinks = p_->req(pre + "attn_sinks.weight", err),
                        *woa = p_->req(pre + "attn_output_a.weight", err),
                        *wob = p_->req(pre + "attn_output_b.weight", err);
        if (!hc_af || !hc_as || !hc_ab || !anorm || !wqa || !qn || !wqb || !wkv || !kvn || !sinks || !woa || !wob)
            return false;

        AttentionLayerWeights aw;
        aw.hc = {p_->g.n_embd, p_->g.rms_eps, p_->g.hc_eps, p_->g.sinkhorn_iters};
        aw.hc_fn = static_cast<const float*>(hc_af->ptr);
        aw.hc_scale = static_cast<const float*>(hc_as->ptr);
        aw.hc_base = static_cast<const float*>(hc_ab->ptr);
        aw.norm = static_cast<const float*>(anorm->ptr);
        aw.attention.q = {wqa->ptr, wqa->type, static_cast<const float*>(qn->ptr), wqb->ptr, wqb->type,
                          p_->g.rms_eps, p_->g.n_embd, p_->g.q_lora, 1e-6f};
        aw.attention.kv = wkv->ptr;
        aw.attention.kv_type = wkv->type;
        aw.attention.kv_norm = static_cast<const float*>(kvn->ptr);
        aw.attention.sinks = static_cast<const float*>(sinks->ptr);
        aw.attention.output_a = woa->ptr;
        aw.attention.output_a_type = woa->type;
        aw.attention.output_b = wob->ptr;
        aw.attention.output_b_type = wob->type;

        AttentionLayerScratch as;
        as.hc_workspace = p_->d_hc_ws;
        as.coefficients = p_->d_coef;
        as.collapsed = p_->d_collapsed;
        as.normalized = p_->d_normed;
        as.attention = {p_->d_q8, p_->d_ql, p_->d_ql_q8, p_->d_qh, p_->d_kv_proj, p_->d_kv_row,
                        p_->d_mqa_out, p_->d_mqa_ws, p_->d_out_q8, p_->d_out_lat, p_->d_attn_out};

        kernels::SparseMqaArgs sel;
        sel.n_tokens = B;
        sel.win = ws.rings + (size_t) s * kDsparkWinCap * 512;
        sel.win_type = kernels::MqaRowType::F32;
        sel.win_cap = kDsparkWindow;
        sel.win_start = p_->d_wstart;
        sel.win_n = p_->d_wn;
        sel.pos = p_->d_pos;
        sel.rope = p_->rope.attention;

        try {
            attention_layer_step(p_->d_streams, aw, p_->rope.attention,
                                 ws.rings + (size_t) s * kDsparkWinCap * 512, sel, as, p_->d_next_streams, p_->stream);
        } catch (const std::exception& e) {
            err = std::string("deepseek4 dspark attn: ") + e.what();
            return false;
        }
        std::swap(p_->d_streams, p_->d_next_streams);

        const DevTensor *hc_ff = p_->req(pre + "hc_ffn_fn.weight", err),
                        *hc_fs = p_->req(pre + "hc_ffn_scale.weight", err),
                        *hc_fb = p_->req(pre + "hc_ffn_base.weight", err),
                        *fnorm = p_->req(pre + "ffn_norm.weight", err),
                        *gate = p_->req(pre + "ffn_gate_inp.weight", err),
                        *bias = p_->req(pre + "exp_probs_b.bias", err),
                        *sg = p_->req(pre + "ffn_gate_shexp.weight", err),
                        *su = p_->req(pre + "ffn_up_shexp.weight", err),
                        *sd = p_->req(pre + "ffn_down_shexp.weight", err);
        if (!hc_ff || !hc_fs || !hc_fb || !fnorm || !gate || !bias || !sg || !su || !sd) return false;

        FfnRouteWeights rw;
        rw.hc = {p_->g.n_embd, p_->g.rms_eps, p_->g.hc_eps, p_->g.sinkhorn_iters};
        rw.hc_fn = static_cast<const float*>(hc_ff->ptr);
        rw.hc_scale = static_cast<const float*>(hc_fs->ptr);
        rw.hc_base = static_cast<const float*>(hc_fb->ptr);
        rw.norm = static_cast<const float*>(fnorm->ptr);
        rw.gate = static_cast<const float*>(gate->ptr);
        rw.routing = router_mixed_args(p_->g.n_expert, p_->g.n_vocab, nullptr, nullptr, nullptr, nullptr,
                                       static_cast<const float*>(bias->ptr), nullptr);

        FfnRouteScratch rs{p_->d_hc_ws, p_->d_coef, p_->d_collapsed, p_->d_normed,
                           p_->d_logits_route, p_->d_route_ids, p_->d_route_w};
        SharedExpertWeights sw{sg->type, su->type, sd->type, sg->ptr, su->ptr, sd->ptr,
                               p_->g.n_embd, p_->g.n_ff_exp, p_->g.swiglu_limit};
        SharedExpertScratch ss{p_->d_q8, p_->d_shared_hq8, p_->d_shared_gate, p_->d_shared_up, p_->d_shared_hidden};
        try {
            prepare_ffn_routes(p_->d_streams, B, rw, rs, p_->stream);
            shared_expert_step(p_->d_normed, B, sw, ss, p_->d_shared, p_->stream);
        } catch (const std::exception& e) {
            err = std::string("deepseek4 dspark ffn: ") + e.what();
            return false;
        }
        if (!p_->run_cpu_experts(s, B, p_->d_normed, p_->d_route_ids, p_->d_route_w, p_->d_routed, err))
            return false;
        try {
            finish_ffn_layer(p_->d_streams, p_->d_coef, p_->d_shared, p_->d_routed, B, p_->g.n_embd,
                             p_->d_next_streams, p_->stream);
        } catch (const std::exception& e) {
            err = std::string("deepseek4 dspark ffn finish: ") + e.what();
            return false;
        }
        std::swap(p_->d_streams, p_->d_next_streams);
    }

    // Final head + autoregressive Markov chain + confidence
    const DevTensor *hfn = p_->req("hc_head_fn.weight", err),
                    *hsc = p_->req("hc_head_scale.weight", err),
                    *hbs = p_->req("hc_head_base.weight", err),
                    *onorm = p_->req("output_norm.weight", err),
                    *mw2 = p_->req("markov_w2.weight", err),
                    *conf_w = p_->req("conf_proj.weight", err);
    if (!hfn || !hsc || !hbs || !onorm || !mw2 || !conf_w) return false;
    HeadWeights main_head;
    if (!p_->main->weights().head_weights(main_head, err)) return false;

    HeadWeights hw = main_head;
    hw.hc_fn = static_cast<const float*>(hfn->ptr);
    hw.hc_scale = static_cast<const float*>(hsc->ptr);
    hw.hc_base = static_cast<const float*>(hbs->ptr);
    hw.norm = static_cast<const float*>(onorm->ptr);
    HeadScratch hs{p_->d_head_hc_ws, p_->d_head_coef, p_->d_head_col, p_->d_head_norm};
    try {
        model_head_step(p_->d_streams, B, hw, hs, p_->d_base_logits, p_->stream);
    } catch (const std::exception& e) {
        err = std::string("deepseek4 dspark head: ") + e.what();
        return false;
    }

    const int rank = p_->dp.markov_rank;
    int32_t prev = anchor;
    for (int i = 0; i < B; ++i) {
        if (prev < 0 || prev >= p_->g.n_vocab) {
            err = "deepseek4 dspark: markov token out of range";
            return false;
        }
        const uint16_t* w1_row = p_->markov_w1.data() + (size_t) prev * (size_t) rank;
        for (int j = 0; j < rank; ++j) p_->h_me[j] = bf16_to_f32(w1_row[j]);
        if (!ck(cudaMemcpyAsync(p_->d_me, p_->h_me, (size_t) rank * sizeof(float),
                                cudaMemcpyHostToDevice, p_->stream), "me H2D", err))
            return false;
        kernels::native_quantize_q8_1(p_->d_me, p_->d_me_q8, rank, 1, p_->stream);
        kernels::native_mmvq(mw2->type, mw2->ptr, p_->d_me_q8, p_->d_mbias, rank, p_->g.n_vocab, 1, p_->stream);
        dspark_markov_argmax(p_->d_base_logits + (size_t) i * p_->g.n_vocab, p_->d_mbias, p_->g.n_vocab,
                             p_->d_amax_v, p_->d_amax_i, p_->d_tok, p_->stream);
        dspark_confidence(static_cast<const float*>(conf_w->ptr), p_->d_head_col + (size_t) i * H, p_->d_me,
                          p_->g.n_embd, rank, p_->d_conf, p_->stream);
        if (!ck(cudaMemcpyAsync(p_->h_tok, p_->d_tok, sizeof(int32_t), cudaMemcpyDeviceToHost, p_->stream), "tok D2H", err) ||
            !ck(cudaMemcpyAsync(p_->h_conf, p_->d_conf, sizeof(float), cudaMemcpyDeviceToHost, p_->stream), "conf D2H", err) ||
            !ck(cudaStreamSynchronize(p_->stream), "markov sync", err))
            return false;
        out.token[i] = *p_->h_tok;
        out.conf[i] = *p_->h_conf;
        prev = out.token[i];
    }
    out.n = B;
    return true;
}

}  // namespace basalt::models::deepseek4
