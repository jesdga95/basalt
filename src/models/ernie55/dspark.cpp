// src/models/ernie55/dspark.cpp - ERNIE-5.5-Flash DSpark speculative drafter runtime.
#include "models/ernie55/dspark.hpp"

#include "basalt/core/expert_tier.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/log.hpp"
#include "basalt/models/ernie55/kernels.hpp"
#include "models/ernie55/model.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <map>
#include <string_view>
#include <vector>

namespace basalt::models::ernie55 {

// The drafter's routed experts: the top 4 of the main model's 10 by the same router (renormalized). A draft only has to
// be accepted, and skipping the lower-ranked RAM experts cuts draft host reads while keeping acceptance steady.
constexpr int kDraftTopK = 4;
// A stage's hint for the next one (its router over this stage's router input): the experts a row it names, the
// likeliest first, and the ones the pool warms at most (each L3 group its rows) while the card runs the next stage up
// to its router.
constexpr int kDraftHintK = 8;
constexpr int kDraftWarm = 6;

namespace {

constexpr int kTypeQ8_0 = 8;
constexpr int kTypeBF16 = 30;
constexpr int kTypeF32 = 0;
constexpr int kTypeMXFP4 = 39;
constexpr int kMaxInjectBatch = 8;

inline float bf16_to_f32(uint16_t u) {
    uint32_t bits = (uint32_t) u << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, 4);
    return f;
}

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("ernie55 dspark: ") + what + ": " + cudaGetErrorString(e);
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

struct Window {
    std::vector<uint16_t*> ctx_ring;   ///< [n_stages], each [kDsparkWinCap, head_dim] FP16
    int64_t ctx_start = 0;
    int64_t ctx_end = 0;
};

// Per-stage lossless LRU cache of hot MXFP4 routed experts on GPU 0 (100% of experts remain routable via CPU RAM).
struct StageGpuCache {
    uint8_t* dev_pool = nullptr;
    int capacity = 0;
    size_t blob_bytes = 0;
    std::vector<int32_t> slot_of;    ///< [n_expert] -> slot index or -1
    std::vector<int32_t> owner;      ///< [capacity] -> expert id or -1
    std::vector<uint64_t> last_use;  ///< [capacity] -> monotonic tick
    std::vector<uint32_t> miss_cnt;  ///< [n_expert] -> misses seen (admit on 2nd use)
    uint64_t tick = 0;

    void init(int cap, int n_exp, size_t blob) {
        capacity = cap;
        blob_bytes = blob;
        slot_of.assign((size_t) n_exp, -1);
        owner.assign((size_t) cap, -1);
        last_use.assign((size_t) cap, 0);
        miss_cnt.assign((size_t) n_exp, 0);
        tick = 0;
    }

    uint8_t* lookup(int e) {
        if (capacity <= 0) return nullptr;
        const int sl = slot_of[(size_t) e];
        if (sl < 0) return nullptr;
        last_use[(size_t) sl] = ++tick;
        return dev_pool + (size_t) sl * blob_bytes;
    }

    int alloc_victim(int e) {
        if (capacity <= 0) return -1;
        if (slot_of[(size_t) e] >= 0) return slot_of[(size_t) e];
        int victim = 0;
        uint64_t best_t = UINT64_MAX;
        for (int i = 0; i < capacity; ++i) {
            if (owner[(size_t) i] < 0) { victim = i; break; }
            if (last_use[(size_t) i] < best_t) { best_t = last_use[(size_t) i]; victim = i; }
        }
        if (owner[(size_t) victim] >= 0) slot_of[(size_t) owner[(size_t) victim]] = -1;
        owner[(size_t) victim] = e;
        slot_of[(size_t) e] = victim;
        last_use[(size_t) victim] = ++tick;
        return victim;
    }
};

}  // namespace

struct ErnieDsparkDrafter::Impl {
    const ErnieModel* main = nullptr;
    Geometry g;
    DsparkPack dp;
    int device = 0;
    RopeTable rope{};

    std::map<std::string, DevTensor> dev;
    uint64_t dev_bytes = 0;
    std::vector<uint16_t> markov_w1;    ///< BF16 [n_vocab, markov_rank] in host RAM
    std::vector<uint8_t> experts_ram;   ///< MXFP4 [n_stages, n_expert, blob_bytes] in host RAM
    kernels::cpu::NativeFmt nat_fmt{};
    kernels::NativeExpertLayout gpu_lay{};
    std::vector<StageGpuCache> gpu_cache;

    std::vector<Window> windows;
    int next_win = 0;

    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    cudaEvent_t sync_ev = nullptr;
    cudaEvent_t ids_ev = nullptr, hint_ev = nullptr;   // a stage's choices and inputs on the host; its hint there

    // Device scratch buffers sized for max(kDsparkBlock, kMaxInjectBatch) = 8 rows
    int32_t* d_pos = nullptr;
    float *d_res_a = nullptr, *d_res_b = nullptr, *d_x_in = nullptr, *d_x_norm = nullptr, *d_sub_out = nullptr;
    float *d_mhc_ws = nullptr, *d_mhc_coef = nullptr;
    float *d_qr = nullptr, *d_q = nullptr, *d_kv = nullptr;
    float *d_attn_o = nullptr, *d_attn_o2 = nullptr, *d_gate = nullptr, *d_z = nullptr;
    float *d_ff_g = nullptr, *d_ff_u = nullptr, *d_ff_h = nullptr;
    float *d_rlogits = nullptr, *d_exp_w = nullptr;
    int32_t* d_exp_ids = nullptr;
    int32_t* d_hint_ids = nullptr;
    float* d_hint_w = nullptr;
    float *d_x_lat = nullptr, *d_exp_out = nullptr, *d_lat_sum = nullptr, *d_lat_norm = nullptr, *d_routed_out = nullptr;
    float *d_hcout = nullptr, *d_hnorm = nullptr;
    float *d_base_logits = nullptr, *d_mbias = nullptr, *d_me = nullptr;
    float *d_amax_v = nullptr, *d_conf = nullptr;
    int32_t *d_amax_i = nullptr, *d_tok = nullptr;
    void *d_xq = nullptr, *d_exp_scratch = nullptr;
    unsigned long long* d_grp_ptr = nullptr;
    int32_t *d_grp_start = nullptr, *d_n_groups = nullptr, *d_ent_dst = nullptr, *d_ent_tok = nullptr;

    // Pinned host staging buffers for fast GPU<->CPU transfers
    float* h_emb = nullptr;
    float* h_x_lat = nullptr;
    float* h_exp_out = nullptr;
    int32_t* h_exp_ids = nullptr;
    int32_t* h_hint_ids = nullptr;
    float* h_me = nullptr;
    int32_t* h_tok = nullptr;
    float* h_conf = nullptr;
    std::vector<uint8_t> h_nact;
    std::vector<unsigned long long> h_grp_ptr;
    std::vector<int32_t> h_grp_start, h_ent_dst, h_ent_tok;

    // Diagnostics
    int64_t n_blocks = 0, cache_hits = 0, cache_lookups = 0;
    double total_draft_ms = 0.0;

    ~Impl() {
        if (n_blocks > 0) {
            const double pct = cache_lookups > 0 ? 100.0 * (double) cache_hits / (double) cache_lookups : 0.0;
            basalt::log::info("basalt ernie55", "dspark stats: %lld blocks, %.2f ms/block, gpu-cache hit=%.1f%% (%lld/%lld)",
                              (long long) n_blocks, total_draft_ms / (double) n_blocks, pct,
                              (long long) cache_hits, (long long) cache_lookups);
        }
        for (auto& [k, t] : dev)
            if (t.ptr) cudaFree(t.ptr);
        for (auto& c : gpu_cache) dfree(c.dev_pool);
        for (auto& w : windows)
            for (uint16_t* r : w.ctx_ring) dfree(r);
        dfree(d_pos); dfree(d_res_a); dfree(d_res_b); dfree(d_x_in); dfree(d_x_norm); dfree(d_sub_out);
        dfree(d_mhc_ws); dfree(d_mhc_coef); dfree(d_qr); dfree(d_q); dfree(d_kv);
        dfree(d_attn_o); dfree(d_attn_o2); dfree(d_gate); dfree(d_z);
        dfree(d_ff_g); dfree(d_ff_u); dfree(d_ff_h); dfree(d_rlogits); dfree(d_exp_w); dfree(d_exp_ids);
        dfree(d_x_lat); dfree(d_exp_out); dfree(d_lat_sum); dfree(d_lat_norm); dfree(d_routed_out);
        dfree(d_hcout); dfree(d_hnorm); dfree(d_base_logits); dfree(d_mbias); dfree(d_me);
        dfree(d_amax_v); dfree(d_conf); dfree(d_amax_i); dfree(d_tok); dfree(d_xq); dfree(d_exp_scratch);
        dfree(d_grp_ptr); dfree(d_grp_start); dfree(d_n_groups); dfree(d_ent_dst); dfree(d_ent_tok);
        dfree(d_hint_ids); dfree(d_hint_w);
        if (h_hint_ids) cudaFreeHost(h_hint_ids);
        if (ids_ev) cudaEventDestroy(ids_ev);
        if (hint_ev) cudaEventDestroy(hint_ev);
        if (h_emb) cudaFreeHost(h_emb);
        if (h_x_lat) cudaFreeHost(h_x_lat);
        if (h_exp_out) cudaFreeHost(h_exp_out);
        if (h_exp_ids) cudaFreeHost(h_exp_ids);
        if (h_me) cudaFreeHost(h_me);
        if (h_tok) cudaFreeHost(h_tok);
        if (h_conf) cudaFreeHost(h_conf);
        if (sync_ev) cudaEventDestroy(sync_ev);
        if (cublas) cublasDestroy(cublas);
        if (stream) cudaStreamDestroy(stream);
    }

    const DevTensor* req(const std::string& name, std::string& err) const {
        auto it = dev.find(name);
        if (it == dev.end()) {
            err = "ernie55 dspark: missing tensor " + name;
            return nullptr;
        }
        return &it->second;
    }

    const DevTensor* req(int s, const char* suf, std::string& err) const {
        return req("blk." + std::to_string(s) + "." + suf, err);
    }

    void matmul_q8_on(const DevTensor* w, const float* x, float* y, int n_in, int n_out, int T, cudaStream_t st) const {
        kernels::native_quantize_q8_1(x, d_xq, n_in, T, st);
        const Q8xMat m{w->ptr, d_xq, y, n_out};
        q8_mmvq(&m, 1, n_in, T, st);
    }

    void matmul_f32_on(const float* w, const float* x, float* y, int n_in, int n_out, int T, cublasHandle_t cb) const {
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &alpha, w, n_in, x, n_in, &beta, y, n_out);
    }

    // A stage's choices and latent inputs to the host (behind its router and down projection), its own event behind them.
    void begin_stage(int B) {
        const int entries = B * kDraftTopK;
        cudaMemcpyAsync(h_exp_ids, d_exp_ids, (size_t) entries * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_x_lat, d_x_lat, (size_t) B * (size_t) g.n_latent * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(ids_ev, stream);
    }

    // Stage s + 1's routing predicted from stage s's router input (`x_norm`, B rows) to the host behind its own event.
    bool hint_stage(int s, const float* x_norm, int B, std::string& err) {
        const DevTensor *rg = req(s + 1, "ffn_gate_inp.weight", err), *rb = req(s + 1, "exp_probs_b.bias", err),
                        *rs = req(s + 1, "exp_probs_scale.weight", err);
        if (!rg || !rb || !rs) return false;
        if (rg->type != kTypeBF16) return true;
        kernels::bf16_gemv_fp32_mmvf_cols(x_norm, (const uint16_t*) rg->ptr, d_rlogits, g.n_embd, 2 * g.n_expert, B, stream);
        split_sigmoid_route(d_rlogits, (const float*) rb->ptr, (const float*) rs->ptr, B, g.n_expert, kDraftHintK,
                            g.norm_topk_prob, d_hint_ids, d_hint_w, stream);
        cudaMemcpyAsync(h_hint_ids, d_hint_ids, (size_t) B * kDraftHintK * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(hint_ev, stream);
        hint_rows = B;
        return true;
    }
    int hint_rows = 0;   // the rows of the hint queued for the next stage (0: none)

    // Once stage s's experts are done: the next stage's hinted experts, the likeliest first over the rows, read into
    // the pool's caches while the card runs that stage up to its router.
    bool warm_next(int s, kernels::cpu::ExpertPool* pool, std::string& err) {
        if (hint_rows <= 0 || pool->groups() < 2) return true;
        if (!ck(cudaEventSynchronize(hint_ev), "stage hint sync", err)) return false;
        const uint8_t* next_base = experts_ram.data() + (size_t) (s + 1) * dp.stage_stride;
        int picked[kDraftWarm];
        int n = 0;
        pool->warm_clear();
        for (int j = 0; j < kDraftHintK && n < kDraftWarm; ++j)
            for (int r = 0; r < hint_rows && n < kDraftWarm; ++r) {
                const int e = h_hint_ids[r * kDraftHintK + j];
                if (e < 0 || e >= dp.n_expert || std::find(picked, picked + n, e) != picked + n) continue;
                picked[n++] = e;
                pool->warm_expert(nat_fmt, next_base + (size_t) e * dp.blob_bytes);
            }
        hint_rows = 0;
        pool->warm_start();
        return true;
    }

    bool run_stage_experts(int s, int B, std::string& err) {
        auto& st = main->shared_tier();
        if (!st.tier || !st.tier->pool()) {
            err = "ernie55 dspark: CPU expert pool not ready";
            return false;
        }
        kernels::cpu::ExpertPool* pool = st.tier->pool();
        const int K = kDraftTopK;
        const int L = g.n_latent;
        const int entries = B * K;

        if (!ck(cudaEventSynchronize(ids_ev), "stage expert D2H sync", err)) return false;

        // Deduplicate experts across the B block rows
        struct Slot {
            int expert = -1;
            int nt = 0;
            int tok[kernels::cpu::MAXT] = {};
            int row[kernels::cpu::MAXT] = {};
        };
        std::array<Slot, 128> slots{};
        int n_slots = 0;
        for (int t = 0; t < B; ++t) {
            for (int k = 0; k < K; ++k) {
                const int e = h_exp_ids[t * K + k];
                if (e < 0 || e >= dp.n_expert) {
                    err = "ernie55 dspark: routed expert id out of range";
                    return false;
                }
                int idx = -1;
                for (int i = 0; i < n_slots; ++i) {
                    if (slots[(size_t) i].expert == e) {
                        idx = i;
                        break;
                    }
                }
                if (idx < 0) {
                    idx = n_slots++;
                    slots[(size_t) idx].expert = e;
                    slots[(size_t) idx].nt = 0;
                }
                auto& sl = slots[(size_t) idx];
                sl.tok[sl.nt] = t;
                sl.row[sl.nt] = t * K + k;
                sl.nt++;
            }
        }

        // Partition distinct experts into GPU-0 LRU hits vs CPU-RAM misses
        StageGpuCache& cache = gpu_cache[(size_t) s];
        const uint8_t* stage_base = experts_ram.data() + (size_t) s * dp.stage_stride;
        int n_gpu_grp = 0, n_gpu_ent = 0;
        std::array<kernels::cpu::ExpertJobMulti, 128> jobs{};
        int n_cpu_jobs = 0;
        std::array<int, 128> cpu_rows{};
        int n_cpu_rows = 0;
        std::array<int, 8> promote_ids{};
        int n_promote = 0;

        for (int i = 0; i < n_slots; ++i) {
            const auto& sl = slots[(size_t) i];
            ++cache_lookups;
            uint8_t* dev_blob = cache.lookup(sl.expert);
            if (dev_blob) {
                ++cache_hits;
                h_grp_ptr[(size_t) n_gpu_grp] = (unsigned long long) dev_blob;
                h_grp_start[(size_t) n_gpu_grp] = n_gpu_ent;
                for (int j = 0; j < sl.nt; ++j) {
                    h_ent_dst[(size_t) n_gpu_ent] = sl.row[j];
                    h_ent_tok[(size_t) n_gpu_ent] = sl.tok[j];
                    ++n_gpu_ent;
                }
                ++n_gpu_grp;
            } else {
                auto& jb = jobs[(size_t) n_cpu_jobs++];
                jb.blob = stage_base + (size_t) sl.expert * dp.blob_bytes;
                jb.nt = sl.nt;
                for (int j = 0; j < sl.nt; ++j) {
                    jb.nact[j] = h_nact.data() + (size_t) sl.tok[j] * nat_fmt.act_bytes;
                    jb.out[j] = h_exp_out + (size_t) sl.row[j] * L;
                    cpu_rows[(size_t) n_cpu_rows++] = sl.row[j];
                }
                if (cache.capacity > 0) {
                    const uint32_t m = ++cache.miss_cnt[(size_t) sl.expert];
                    if (m >= 2 && n_promote < 2) promote_ids[(size_t) n_promote++] = sl.expert;
                }
            }
        }
        h_grp_start[(size_t) n_gpu_grp] = n_gpu_ent;

        // 1. Launch GPU-0 grouped MXFP4 hits asynchronously FIRST so GPU & CPU overlap!
        if (n_gpu_grp > 0) {
            kernels::native_quantize_q8_1(d_x_lat, d_xq, L, B, stream);
            cudaMemcpyAsync(d_grp_ptr, h_grp_ptr.data(), (size_t) n_gpu_grp * sizeof(unsigned long long),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_grp_start, h_grp_start.data(), (size_t) (n_gpu_grp + 1) * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_n_groups, &n_gpu_grp, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_ent_dst, h_ent_dst.data(), (size_t) n_gpu_ent * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_ent_tok, h_ent_tok.data(), (size_t) n_gpu_ent * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            kernels::native_expert_grouped(gpu_lay, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, d_ent_tok,
                                           n_gpu_grp, n_gpu_ent, d_xq, d_exp_scratch, d_exp_out, stream);
        }

        // 2. Run CPU-RAM misses in parallel on the 15-worker AVX-VNNI multi-row MXFP4 pool!
        if (n_cpu_jobs > 0) {
            for (int t = 0; t < B; ++t) {
                kernels::cpu::native_quant_act(nat_fmt, h_x_lat + (size_t) t * L,
                                               h_nact.data() + (size_t) t * nat_fmt.act_bytes);
            }
            pool->start_split_multi_native(nat_fmt, jobs.data(), n_cpu_jobs);   // the L3 groups' batch (warm_next)
            pool->finish_split_multi_native();

            if (n_gpu_grp == 0) {
                cudaMemcpyAsync(d_exp_out, h_exp_out, (size_t) entries * (size_t) L * sizeof(float),
                                cudaMemcpyHostToDevice, stream);
            } else {
                for (int i = 0; i < n_cpu_rows; ++i) {
                    const int row = cpu_rows[(size_t) i];
                    cudaMemcpyAsync(d_exp_out + (size_t) row * L, h_exp_out + (size_t) row * L,
                                    (size_t) L * sizeof(float), cudaMemcpyHostToDevice, stream);
                }
            }
        }

        if (s + 1 < dp.n_stages && !warm_next(s, pool, err)) return false;

        // 3. Promote up to 2 hot experts into the stage GPU-0 LRU cache after GPU kernel reads finished
        for (int i = 0; i < n_promote; ++i) {
            const int e = promote_ids[(size_t) i];
            const int sl = cache.alloc_victim(e);
            if (sl >= 0) {
                cudaMemcpyAsync(cache.dev_pool + (size_t) sl * dp.blob_bytes,
                                stage_base + (size_t) e * dp.blob_bytes, dp.blob_bytes, cudaMemcpyHostToDevice, stream);
            }
        }
        return true;
    }
};

ErnieDsparkDrafter::ErnieDsparkDrafter(std::unique_ptr<Impl> p) : p_(std::move(p)) {}
ErnieDsparkDrafter::~ErnieDsparkDrafter() = default;

std::unique_ptr<ErnieDsparkDrafter> ErnieDsparkDrafter::plan_load(const ErnieModel& main, int file_handle,
                                                                  core::LoadPlan& reads, int max_windows,
                                                                  int gpu_cache_per_stage, std::string& err) {
    const Pack& pack = main.pack();
    if (!pack.dspark.present) {
        err = "ernie55 dspark: pack has no dspark section";
        return nullptr;
    }
    auto p = std::make_unique<Impl>();
    p->main = &main;
    p->g = pack.geo;
    p->dp = pack.dspark;
    p->device = main.device();
    cudaSetDevice(p->device);

    if (p->dp.block_size <= 0 || p->dp.block_size > kDsparkBlock) {
        err = "ernie55 dspark: unsupported block_size " + std::to_string(p->dp.block_size);
        return nullptr;
    }

    p->rope = rope_table_plain(p->dp.rope_theta);

    if (!kernels::cpu::native_fmt(kTypeMXFP4, kTypeMXFP4, p->g.n_latent, p->g.n_ff_exp, p->nat_fmt, err)) {
        err = "ernie55 dspark: " + err;
        return nullptr;
    }
    p->nat_fmt.gate_softcap = p->g.gate_softcap;
    p->nat_fmt.up_softcap = p->g.up_softcap;
    p->gpu_lay = kernels::native_expert_layout(kTypeMXFP4, kTypeMXFP4, p->g.n_latent, p->g.n_ff_exp, 0.0f);
    p->gpu_lay.gate_softcap = p->g.gate_softcap;
    p->gpu_lay.up_softcap = p->g.up_softcap;

    // Allocate optional per-stage lossless GPU-0 hot-expert LRU cache
    const int cache_cap = std::clamp(gpu_cache_per_stage, 0, p->dp.n_expert);
    p->gpu_cache.resize((size_t) p->dp.n_stages);
    for (int s = 0; s < p->dp.n_stages; ++s) {
        p->gpu_cache[(size_t) s].init(cache_cap, p->dp.n_expert, p->dp.blob_bytes);
        if (cache_cap > 0) {
            const size_t pool_bytes = (size_t) cache_cap * p->dp.blob_bytes;
            if (!ck(cudaMalloc(&p->gpu_cache[(size_t) s].dev_pool, pool_bytes), "dspark gpu cache", err))
                return nullptr;
            p->dev_bytes += pool_bytes;
        }
    }

    if (!ck(cudaStreamCreate(&p->stream), "stream", err) ||
        !ck(cudaEventCreateWithFlags(&p->sync_ev, cudaEventDisableTiming), "event", err) ||
        !ck(cudaEventCreateWithFlags(&p->ids_ev, cudaEventDisableTiming), "event", err) ||
        !ck(cudaEventCreateWithFlags(&p->hint_ev, cudaEventDisableTiming), "event", err))
        return nullptr;
    if (cublasCreate(&p->cublas) != CUBLAS_STATUS_SUCCESS) {
        err = "ernie55 dspark: cublasCreate failed";
        return nullptr;
    }
    cublasSetStream(p->cublas, p->stream);

    // Load dense tensors: markov_w1 to host RAM (BF16), everything else to GPU 0
    for (const auto& [name, t] : p->dp.tensors) {
        if (name == "markov_w1.weight") {
            if (t.type != kTypeBF16 || t.shape.size() != 2 || t.shape[0] != (uint64_t) p->dp.markov_rank ||
                t.shape[1] != (uint64_t) p->g.n_vocab) {
                err = "ernie55 dspark: markov_w1.weight shape/type mismatch";
                return nullptr;
            }
            p->markov_w1.resize(t.bytes / 2);
            core::LoadItem it;
            it.file = file_handle;
            it.off = t.offset;
            it.bytes = t.bytes;
            it.host = reinterpret_cast<uint8_t*>(p->markov_w1.data());
            reads.add(it);
            continue;
        }
        DevTensor d;
        d.type = t.type;
        d.bytes = t.bytes;
        d.shape = t.shape;
        // Pack stage routers as BF16 via Bf16High only when every F32 element is an exact BF16 value.
        const std::string_view suffix = "ffn_gate_inp.weight";
        bool router_bf16 = false;
        if (t.type == kTypeF32 && name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            if (!f32_range_is_bf16(main.fd(), t.offset, t.bytes, router_bf16, err)) return nullptr;
        }
        if (router_bf16) {
            d.type = kTypeBF16;
            d.bytes = t.bytes / 2;
        }
        const uint64_t pad = (d.type == kTypeQ8_0) ? 256 : 0;
        if (!ck(cudaMalloc(&d.ptr, d.bytes + pad), name.c_str(), err)) return nullptr;
        if (pad) cudaMemset((uint8_t*) d.ptr + d.bytes, 0, pad);
        core::LoadItem it;
        it.file = file_handle;
        it.off = t.offset;
        it.bytes = t.bytes;
        it.dev = d.ptr;
        it.device = p->device;
        if (router_bf16) it.conv = core::Conv::Bf16High;
        reads.add(it);
        p->dev_bytes += d.bytes + pad;
        p->dev.emplace(name, std::move(d));
    }

    // Load all stage MXFP4 routed experts into host RAM
    p->experts_ram.resize(p->dp.experts_bytes);
    {
        // Split into 64 MiB LoadItems for parallel pipelined reading
        constexpr uint64_t kChunk = 64ull << 20;
        uint64_t done = 0;
        while (done < p->dp.experts_bytes) {
            const uint64_t take = std::min(kChunk, p->dp.experts_bytes - done);
            core::LoadItem it;
            it.file = file_handle;
            it.off = p->dp.experts_offset + done;
            it.bytes = take;
            it.host = p->experts_ram.data() + done;
            reads.add(it);
            done += take;
        }
    }

    // Allocate per-session FP16 context rings
    const int nwin = std::max(1, max_windows);
    p->windows.resize((size_t) nwin);
    const size_t ring_elems = (size_t) kDsparkWinCap * (size_t) p->g.hca_head_dim;
    for (int w = 0; w < nwin; ++w) {
        p->windows[(size_t) w].ctx_ring.resize((size_t) p->dp.n_stages, nullptr);
        for (int s = 0; s < p->dp.n_stages; ++s) {
            if (!dalloc(p->windows[(size_t) w].ctx_ring[(size_t) s], ring_elems, "ctx_ring", err)) return nullptr;
            cudaMemsetAsync(p->windows[(size_t) w].ctx_ring[(size_t) s], 0, ring_elems * sizeof(uint16_t), p->stream);
            p->dev_bytes += ring_elems * sizeof(uint16_t);
        }
    }

    // Allocate device scratch buffers for max(block_size, kMaxInjectBatch) = 8 rows
    const int maxR = std::max(kDsparkBlock, kMaxInjectBatch);
    const int H = p->g.n_embd;
    const int q_dim = p->g.hca_heads * p->g.hca_head_dim;
    const int gate_dim = p->g.hca_o_groups * p->g.hca_o_lora;
    const int feat_dim = (int) p->dp.target_layers.size() * H;
    const int q8_cols_max = std::max({feat_dim, H, gate_dim, p->g.n_ff_exp, p->g.hca_q_lora, p->g.n_latent,
                                      p->dp.markov_rank});
    const int max_entries = maxR * p->g.n_used;

    if (!dalloc(p->d_pos, maxR, "d_pos", err) ||
        !dalloc(p->d_res_a, (size_t) maxR * kHcMult * H, "d_res_a", err) ||
        !dalloc(p->d_res_b, (size_t) maxR * kHcMult * H, "d_res_b", err) ||
        !dalloc(p->d_x_in, (size_t) maxR * std::max(H, (p->g.hca_heads / p->g.hca_o_groups) * p->g.hca_head_dim), "d_x_in", err) ||
        !dalloc(p->d_x_norm, (size_t) maxR * H, "d_x_norm", err) ||
        !dalloc(p->d_sub_out, (size_t) maxR * H, "d_sub_out", err) ||
        !dalloc(p->d_mhc_ws, mhc_workspace_bytes(maxR) / sizeof(float), "d_mhc_ws", err) ||
        !dalloc(p->d_mhc_coef, (size_t) maxR * kHcCoef, "d_mhc_coef", err) ||
        !dalloc(p->d_qr, (size_t) maxR * p->g.hca_q_lora, "d_qr", err) ||
        !dalloc(p->d_q, (size_t) maxR * q_dim, "d_q", err) ||
        !dalloc(p->d_kv, (size_t) maxR * p->g.hca_head_dim, "d_kv", err) ||
        !dalloc(p->d_attn_o, (size_t) maxR * q_dim, "d_attn_o", err) ||
        !dalloc(p->d_attn_o2, (size_t) maxR * q_dim, "d_attn_o2", err) ||
        !dalloc(p->d_gate, (size_t) maxR * gate_dim, "d_gate", err) ||
        !dalloc(p->d_z, (size_t) maxR * gate_dim, "d_z", err) ||
        !dalloc(p->d_ff_g, (size_t) maxR * p->g.n_ff_exp, "d_ff_g", err) ||
        !dalloc(p->d_ff_u, (size_t) maxR * p->g.n_ff_exp, "d_ff_u", err) ||
        !dalloc(p->d_ff_h, (size_t) maxR * std::max(p->g.n_ff_exp, p->g.hca_o_lora), "d_ff_h", err) ||
        !dalloc(p->d_rlogits, (size_t) maxR * 2 * p->g.n_expert, "d_rlogits", err) ||
        !dalloc(p->d_exp_w, (size_t) max_entries, "d_exp_w", err) ||
        !dalloc(p->d_exp_ids, (size_t) max_entries, "d_exp_ids", err) ||
        !dalloc(p->d_hint_ids, (size_t) maxR * kDraftHintK, "d_hint_ids", err) ||
        !dalloc(p->d_hint_w, (size_t) maxR * kDraftHintK, "d_hint_w", err) ||
        !dalloc(p->d_x_lat, (size_t) maxR * p->g.n_latent, "d_x_lat", err) ||
        !dalloc(p->d_exp_out, (size_t) max_entries * p->g.n_latent, "d_exp_out", err) ||
        !dalloc(p->d_lat_sum, (size_t) maxR * p->g.n_latent, "d_lat_sum", err) ||
        !dalloc(p->d_lat_norm, (size_t) maxR * p->g.n_latent, "d_lat_norm", err) ||
        !dalloc(p->d_routed_out, (size_t) maxR * H, "d_routed_out", err) ||
        !dalloc(p->d_hcout, (size_t) maxR * H, "d_hcout", err) ||
        !dalloc(p->d_hnorm, (size_t) maxR * H, "d_hnorm", err) ||
        !dalloc(p->d_base_logits, (size_t) kDsparkBlock * (size_t) p->g.n_vocab, "d_base_logits", err) ||
        !dalloc(p->d_mbias, (size_t) p->g.n_vocab, "d_mbias", err) ||
        !dalloc(p->d_me, (size_t) p->dp.markov_rank, "d_me", err) ||
        !dalloc(p->d_amax_v, 256, "d_amax_v", err) ||
        !dalloc(p->d_amax_i, 256, "d_amax_i", err) ||
        !dalloc(p->d_tok, 1, "d_tok", err) ||
        !dalloc(p->d_conf, 1, "d_conf", err))
        return nullptr;

    // the widest input at maxR columns, or the attention output's eight groups at maxR columns each
    const int o_gin = p->g.hca_heads / p->g.hca_o_groups * p->g.hca_head_dim;
    const size_t q8_bytes = std::max(kernels::native_q8_1_bytes(q8_cols_max, maxR),
                                     kernels::native_q8_1_bytes(o_gin, 1) * (size_t) (p->g.hca_o_groups * maxR));
    const size_t exp_scratch = kernels::native_expert_scratch_bytes(max_entries, p->g.n_ff_exp);
    if (!ck(cudaMalloc(&p->d_xq, q8_bytes), "d_xq", err) ||
        !ck(cudaMalloc(&p->d_exp_scratch, exp_scratch), "d_exp_scratch", err) ||
        !dalloc(p->d_grp_ptr, max_entries, "d_grp_ptr", err) ||
        !dalloc(p->d_grp_start, max_entries + 1, "d_grp_start", err) ||
        !dalloc(p->d_n_groups, 1, "d_n_groups", err) ||
        !dalloc(p->d_ent_dst, max_entries, "d_ent_dst", err) ||
        !dalloc(p->d_ent_tok, max_entries, "d_ent_tok", err))
        return nullptr;
    p->h_grp_ptr.resize(max_entries);
    p->h_grp_start.resize(max_entries + 1);
    p->h_ent_dst.resize(max_entries);
    p->h_ent_tok.resize(max_entries);

    // Pinned host staging buffers
    if (!ck(cudaHostAlloc(&p->h_emb, (size_t) maxR * H * sizeof(float), cudaHostAllocPortable), "h_emb", err) ||
        !ck(cudaHostAlloc(&p->h_x_lat, (size_t) maxR * p->g.n_latent * sizeof(float), cudaHostAllocPortable), "h_x_lat", err) ||
        !ck(cudaHostAlloc(&p->h_exp_out, (size_t) max_entries * p->g.n_latent * sizeof(float), cudaHostAllocPortable), "h_exp_out", err) ||
        !ck(cudaHostAlloc(&p->h_exp_ids, (size_t) max_entries * sizeof(int32_t), cudaHostAllocPortable), "h_exp_ids", err) ||
        !ck(cudaHostAlloc(&p->h_hint_ids, (size_t) maxR * kDraftHintK * sizeof(int32_t), cudaHostAllocPortable), "h_hint_ids", err) ||
        !ck(cudaHostAlloc(&p->h_me, (size_t) p->dp.markov_rank * sizeof(float), cudaHostAllocPortable), "h_me", err) ||
        !ck(cudaHostAlloc(&p->h_tok, sizeof(int32_t), cudaHostAllocPortable), "h_tok", err) ||
        !ck(cudaHostAlloc(&p->h_conf, sizeof(float), cudaHostAllocPortable), "h_conf", err))
        return nullptr;
    p->h_nact.resize((size_t) maxR * p->nat_fmt.act_bytes);

    return std::unique_ptr<ErnieDsparkDrafter>(new ErnieDsparkDrafter(std::move(p)));
}

int ErnieDsparkDrafter::open_window() {
    if (p_->next_win >= (int) p_->windows.size()) return -1;
    return p_->next_win++;
}

int ErnieDsparkDrafter::feature_slot(int layer) const {
    for (size_t i = 0; i < p_->dp.target_layers.size(); ++i)
        if (p_->dp.target_layers[i] == layer) return (int) i;
    return -1;
}

int ErnieDsparkDrafter::feature_floats() const { return (int) p_->dp.target_layers.size() * p_->g.n_embd; }
int ErnieDsparkDrafter::block_size() const { return p_->dp.block_size; }
int ErnieDsparkDrafter::n_stages() const { return p_->dp.n_stages; }
uint64_t ErnieDsparkDrafter::device_bytes() const { return p_->dev_bytes; }
uint64_t ErnieDsparkDrafter::host_bytes() const {
    return (uint64_t) p_->markov_w1.size() * 2 + (uint64_t) p_->experts_ram.size();
}

void ErnieDsparkDrafter::forget_from(int w, int64_t pos) {
    if (w < 0 || w >= (int) p_->windows.size()) return;
    auto& win = p_->windows[(size_t) w];
    if (pos <= win.ctx_start) {
        win.ctx_start = pos;
        win.ctx_end = pos;
    } else if (pos < win.ctx_end) {
        win.ctx_end = pos;
    }
}

void ErnieDsparkDrafter::reset(int w) {
    if (w < 0 || w >= (int) p_->windows.size()) return;
    auto& win = p_->windows[(size_t) w];
    win.ctx_start = 0;
    win.ctx_end = 0;
}

bool ErnieDsparkDrafter::inject_device(int w, int64_t pos0, int n_rows, const float* d_features, void* stream,
                                       std::string& err) {
    if (n_rows <= 0) return true;
    if (w < 0 || w >= (int) p_->windows.size()) {
        err = "ernie55 dspark: invalid window";
        return false;
    }
    cudaSetDevice(p_->device);
    cudaStream_t st = (cudaStream_t) stream;
    const DevTensor *fproj = p_->req("feature_proj.weight", err),
                    *fnorm = p_->req("feature_norm.weight", err);
    if (!fproj || !fnorm) return false;

    const int H = p_->g.n_embd;
    const int F = feature_floats();
    auto& win = p_->windows[(size_t) w];

    for (int r0 = 0; r0 < n_rows; r0 += kMaxInjectBatch) {
        const int tw = std::min(kMaxInjectBatch, n_rows - r0);
        const float* fslice = d_features + (size_t) r0 * (size_t) F;
        p_->matmul_q8_on(fproj, fslice, p_->d_x_norm, F, H, tw, st);
        kernels::rms_norm_rows(p_->d_x_norm, (const float*) fnorm->ptr, tw, H, p_->g.rms_eps, p_->d_x_norm, st);
        for (int s = 0; s < p_->dp.n_stages; ++s) {
            const DevTensor *wkv = p_->req(s, "attn_kv.weight", err),
                            *kvn = p_->req(s, "attn_kv_a_norm.weight", err);
            if (!wkv || !kvn) return false;
            p_->matmul_q8_on(wkv, p_->d_x_norm, p_->d_kv, H, p_->g.hca_head_dim, tw, st);
            dspark_kv_ctx_store(p_->d_kv, (const float*) kvn->ptr, pos0 + r0, tw, p_->g.hca_head_dim, kDsparkWinCap,
                                p_->g.rms_eps, p_->rope, win.ctx_ring[(size_t) s], st);
        }
    }

    if (win.ctx_end == 0 && pos0 > 0) win.ctx_start = pos0;
    else if (pos0 > win.ctx_end) win.ctx_start = pos0;
    win.ctx_end = pos0 + n_rows;
    if (win.ctx_end - win.ctx_start > kDsparkWinCap) win.ctx_start = win.ctx_end - kDsparkWinCap;
    return true;
}

bool ErnieDsparkDrafter::draft(int w, int32_t anchor, int64_t pos, int n_out, void* caller_stream, DsparkDraft& out,
                               std::string& err) {
    out.n = 0;
    if (w < 0 || w >= (int) p_->windows.size()) {
        err = "ernie55 dspark: invalid window";
        return false;
    }
    cudaSetDevice(p_->device);

    // Order DSpark stream after caller_stream (e.g. inject_device)
    if (caller_stream && caller_stream != (void*) p_->stream) {
        if (!ck(cudaEventRecord(p_->sync_ev, (cudaStream_t) caller_stream), "record sync_ev", err) ||
            !ck(cudaStreamWaitEvent(p_->stream, p_->sync_ev, 0), "wait sync_ev", err))
            return false;
    }

    const auto t_start = std::chrono::steady_clock::now();
    // the block holds the drafts asked for, not the whole trained block: the rows past them change little in the
    // drafts and every row's experts are read from RAM
    const int n_gen = std::clamp(n_out, 1, p_->dp.block_size);
    const int B = n_gen;
    const int H = p_->g.n_embd;
    const auto& g = p_->g;
    const auto& win = p_->windows[(size_t) w];

    // Prepare block input tokens [anchor, noise, ..., noise] and positions [pos, pos+1, ..., pos+B-1]
    int32_t h_pos[kDsparkBlock];
    for (int r = 0; r < B; ++r) {
        const int32_t tok = (r == 0) ? anchor : p_->dp.noise_token_id;
        h_pos[r] = (int32_t) (pos + r);
        if (!p_->main->embedding_row(tok, p_->h_emb + (size_t) r * H)) {
            err = "ernie55 dspark: token id out of range";
            return false;
        }
    }
    cudaMemcpyAsync(p_->d_pos, h_pos, (size_t) B * sizeof(int32_t), cudaMemcpyHostToDevice, p_->stream);
    cudaMemcpyAsync(p_->d_x_in, p_->h_emb, (size_t) B * (size_t) H * sizeof(float), cudaMemcpyHostToDevice, p_->stream);
    broadcast_streams(p_->d_x_in, B, H, p_->d_res_a, p_->stream);

    float* res = p_->d_res_a;
    float* res_next = p_->d_res_b;
    const int64_t eff_pos0 = std::min(pos, win.ctx_end);

    for (int s = 0; s < p_->dp.n_stages; ++s) {
        // ---- Attention sublayer (SWA + bidirectional block) ----
        const DevTensor *hc_fn = p_->req(s, "hc_attn_fn.weight", err),
                        *hc_sc = p_->req(s, "hc_attn_scale.weight", err),
                        *hc_bs = p_->req(s, "hc_attn_base.weight", err),
                        *anorm = p_->req(s, "attn_norm.weight", err),
                        *qa = p_->req(s, "attn_q_a.weight", err),
                        *qan = p_->req(s, "attn_q_a_norm.weight", err),
                        *qb = p_->req(s, "attn_q_b.weight", err),
                        *wkv = p_->req(s, "attn_kv.weight", err),
                        *kvn = p_->req(s, "attn_kv_a_norm.weight", err),
                        *asink = p_->req(s, "attn_sinks.weight", err),
                        *vha_m = p_->req(s, "attn_vha_mix.weight", err),
                        *agate = p_->req(s, "attn_gate.weight", err),
                        *oa = p_->req(s, "attn_output_a.weight", err),
                        *ob = p_->req(s, "attn_output_b.weight", err);
        if (!hc_fn || !hc_sc || !hc_bs || !anorm || !qa || !qan || !qb || !wkv || !kvn || !asink || !vha_m || !agate ||
            !oa || !ob)
            return false;

        mhc_mixes(res, (const float*) hc_fn->ptr, (const float*) hc_sc->ptr, (const float*) hc_bs->ptr, B, H,
                  g.rms_eps, g.hc_eps, g.sinkhorn_iters, p_->d_mhc_ws, p_->d_mhc_coef, p_->stream);
        mhc_pre(res, p_->d_mhc_coef, kHcCoef, B, H, p_->d_x_in, p_->stream);
        kernels::rms_norm_rows(p_->d_x_in, (const float*) anorm->ptr, B, H, g.rms_eps, p_->d_x_norm, p_->stream);

        p_->matmul_q8_on(qa, p_->d_x_norm, p_->d_qr, H, g.hca_q_lora, B, p_->stream);
        kernels::rms_norm_rows(p_->d_qr, (const float*) qan->ptr, B, g.hca_q_lora, g.rms_eps, p_->d_qr, p_->stream);
        p_->matmul_q8_on(qb, p_->d_qr, p_->d_q, g.hca_q_lora, g.hca_heads * g.hca_head_dim, B, p_->stream);
        hca_q_rope(p_->d_q, p_->d_pos, B, g.hca_heads, g.hca_head_dim, g.rms_eps, p_->rope, p_->stream);
        p_->matmul_q8_on(agate, p_->d_qr, p_->d_gate, g.hca_q_lora, g.hca_o_groups * g.hca_o_lora, B, p_->stream);

        p_->matmul_q8_on(wkv, p_->d_x_norm, p_->d_kv, H, g.hca_head_dim, B, p_->stream);
        hca_kv_norm_rope(p_->d_kv, (const float*) kvn->ptr, p_->d_pos, B, g.hca_head_dim, g.rms_eps, p_->rope,
                         p_->d_kv, p_->stream);

        dspark_swa_block_attn(p_->d_q, win.ctx_ring[(size_t) s], p_->d_kv, (const float*) asink->ptr, eff_pos0,
                              win.ctx_start, B, g.hca_heads, g.hca_head_dim, p_->dp.window, kDsparkWinCap,
                              p_->d_attn_o, p_->stream);
        vha_postmix_f32(p_->d_attn_o, (const float*) vha_m->ptr, B, g.hca_heads, g.hca_head_dim, p_->d_attn_o2,
                        p_->stream);
        hca_inv_rope(p_->d_attn_o2, p_->d_pos, B, g.hca_heads, g.hca_head_dim, p_->rope, p_->stream);

        // Grouped wo_a (8 groups of [1024, 4096]) -> sigmoid gate -> wo_b
        const int hpg = g.hca_heads / g.hca_o_groups;
        const int gin = hpg * g.hca_head_dim;
        const int gout = g.hca_o_lora;
        const size_t oa_grp_bytes = (size_t) gout * (size_t) (gin / 32) * 34;
        {   // the groups side by side, as the main model's windows do (session.cpp): one split, the quantizations, one
            // launch of the eight products, one merge
            const int G = g.hca_o_groups;
            const size_t col = kernels::native_q8_1_bytes(gin, 1);
            hca_groups_split_f32(p_->d_attn_o2, B, G, gin, p_->d_attn_o, p_->stream);   // d_attn_o is free here
            for (int c0 = 0; c0 < G * B; c0 += 8)   // a quantization takes 8 columns at most
                kernels::native_quantize_q8_1(p_->d_attn_o + (size_t) c0 * gin, (uint8_t*) p_->d_xq + (size_t) c0 * col, gin,
                                              std::min(8, G * B - c0), p_->stream);
            Q8xMat og[kQ8xMats];
            for (int grp = 0; grp < G; ++grp)   // the products into d_q, free here too
                og[grp] = {(uint8_t*) oa->ptr + (size_t) grp * oa_grp_bytes, (uint8_t*) p_->d_xq + (size_t) grp * B * col,
                           p_->d_q + (size_t) grp * B * gout, gout};
            q8_mmvq(og, G, gin, B, p_->stream);
            hca_groups_merge_f32(p_->d_q, B, G, gout, p_->d_z, p_->stream);
        }
        sigmoid_gate_mul_f32(p_->d_z, p_->d_gate, (int64_t) B * g.hca_o_groups * g.hca_o_lora, p_->stream);
        p_->matmul_q8_on(ob, p_->d_z, p_->d_sub_out, g.hca_o_groups * g.hca_o_lora, H, B, p_->stream);

        mhc_post(p_->d_sub_out, res, p_->d_mhc_coef, kHcCoef, B, H, res_next, p_->stream);
        std::swap(res, res_next);

        // ---- Latent MoE sublayer (final stage only needs the n_gen rows that reach the LM head!) ----
        const int Bm = (s == p_->dp.n_stages - 1) ? n_gen : B;
        const DevTensor *fh_fn = p_->req(s, "hc_ffn_fn.weight", err),
                        *fh_sc = p_->req(s, "hc_ffn_scale.weight", err),
                        *fh_bs = p_->req(s, "hc_ffn_base.weight", err),
                        *fnorm = p_->req(s, "ffn_norm.weight", err),
                        *rg = p_->req(s, "ffn_gate_inp.weight", err),
                        *rb = p_->req(s, "exp_probs_b.bias", err),
                        *rs = p_->req(s, "exp_probs_scale.weight", err),
                        *ld = p_->req(s, "ffn_routed_down.weight", err),
                        *ln = p_->req(s, "ffn_latent_norm.weight", err),
                        *lu = p_->req(s, "ffn_routed_up.weight", err),
                        *sg = p_->req(s, "ffn_gate_shexp.weight", err),
                        *su = p_->req(s, "ffn_up_shexp.weight", err),
                        *sd = p_->req(s, "ffn_down_shexp.weight", err);
        if (!fh_fn || !fh_sc || !fh_bs || !fnorm || !rg || !rb || !rs || !ld || !ln || !lu || !sg || !su || !sd)
            return false;

        mhc_mixes(res, (const float*) fh_fn->ptr, (const float*) fh_sc->ptr, (const float*) fh_bs->ptr, Bm, H,
                  g.rms_eps, g.hc_eps, g.sinkhorn_iters, p_->d_mhc_ws, p_->d_mhc_coef, p_->stream);
        mhc_pre(res, p_->d_mhc_coef, kHcCoef, Bm, H, p_->d_x_in, p_->stream);
        kernels::rms_norm_rows(p_->d_x_in, (const float*) fnorm->ptr, Bm, H, g.rms_eps, p_->d_x_norm, p_->stream);

        if (rg->type == kTypeBF16)
            kernels::bf16_gemv_fp32_mmvf_cols(p_->d_x_norm, (const uint16_t*) rg->ptr, p_->d_rlogits, H, 2 * g.n_expert, Bm,
                                              p_->stream);
        else
            p_->matmul_f32_on((const float*) rg->ptr, p_->d_x_norm, p_->d_rlogits, H, 2 * g.n_expert, Bm, p_->cublas);
        split_sigmoid_route(p_->d_rlogits, (const float*) rb->ptr, (const float*) rs->ptr, Bm, g.n_expert, kDraftTopK,
                            g.norm_topk_prob, p_->d_exp_ids, p_->d_exp_w, p_->stream);

        // The routed experts' inputs to the host first, then the next stage's hint, then the shared expert, which the
        // card runs while the host computes the routed experts
        p_->matmul_q8_on(ld, p_->d_x_norm, p_->d_x_lat, H, g.n_latent, Bm, p_->stream);
        p_->begin_stage(Bm);
        if (s + 1 < p_->dp.n_stages && !p_->hint_stage(s, p_->d_x_norm, Bm, err)) return false;
        p_->matmul_q8_on(sg, p_->d_x_norm, p_->d_ff_g, H, g.n_ff_exp, Bm, p_->stream);
        p_->matmul_q8_on(su, p_->d_x_norm, p_->d_ff_u, H, g.n_ff_exp, Bm, p_->stream);
        situ_glu_f32(p_->d_ff_g, p_->d_ff_u, Bm, g.n_ff_exp, g.gate_softcap, g.up_softcap, p_->d_ff_h, p_->stream);
        p_->matmul_q8_on(sd, p_->d_ff_h, p_->d_sub_out, g.n_ff_exp, H, Bm, p_->stream);
        if (!p_->run_stage_experts(s, Bm, err)) return false;

        weighted_sum_f32(p_->d_exp_out, p_->d_exp_w, Bm, kDraftTopK, g.n_latent, p_->d_lat_sum, p_->stream);
        kernels::rms_norm_rows(p_->d_lat_sum, (const float*) ln->ptr, Bm, g.n_latent, g.rms_eps, p_->d_lat_norm,
                               p_->stream);
        p_->matmul_q8_on(lu, p_->d_lat_norm, p_->d_routed_out, g.n_latent, H, Bm, p_->stream);
        add_f32(p_->d_sub_out, p_->d_routed_out, (int64_t) Bm * H, p_->stream);

        mhc_post(p_->d_sub_out, res, p_->d_mhc_coef, kHcCoef, Bm, H, res_next, p_->stream);
        std::swap(res, res_next);
    }

    // ---- Final contraction + shared target LM head (BEFORE multimax) ----
    const DevTensor *hfn = p_->req("hc_head_fn.weight", err),
                    *hsc = p_->req("hc_head_scale.weight", err),
                    *hbs = p_->req("hc_head_base.weight", err),
                    *onorm = p_->req("output_norm.weight", err),
                    *mw2 = p_->req("markov_w2.weight", err),
                    *conf_w = p_->req("conf_proj.weight", err);
    const DevTensor *head = p_->main->tensor("output.weight"),
                    *mmr = p_->main->tensor("output_multimax_ranges"),
                    *mmt = p_->main->tensor("output_multimax_ts");
    if (!hfn || !hsc || !hbs || !onorm || !mw2 || !conf_w) return false;
    if (!head || !mmr || !mmt) {
        err = "ernie55 dspark: main model missing output.weight / multimax tensors";
        return false;
    }

    mhc_contract(res, (const float*) hfn->ptr, (const float*) hsc->ptr, (const float*) hbs->ptr, n_gen, H, g.rms_eps,
                 g.hc_eps, p_->d_mhc_ws, p_->d_mhc_coef, p_->d_hcout, p_->stream);
    kernels::rms_norm_rows(p_->d_hcout, (const float*) onorm->ptr, n_gen, H, g.rms_eps, p_->d_hnorm, p_->stream);

    if (head->type == kTypeQ8_0) {
        p_->matmul_q8_on(head, p_->d_hnorm, p_->d_base_logits, H, g.n_vocab, n_gen, p_->stream);
    } else {
        kernels::bf16_gemv_fp32_mmvf_cols(p_->d_hnorm, (const uint16_t*) head->ptr, p_->d_base_logits, H, g.n_vocab,
                                          n_gen, p_->stream);
    }

    // ---- Autoregressive Markov chain + multimax + confidence ----
    const int rank = p_->dp.markov_rank;
    int32_t prev = anchor;
    for (int i = 0; i < n_gen; ++i) {
        if (prev < 0 || prev >= g.n_vocab) {
            err = "ernie55 dspark: markov prev token out of range";
            return false;
        }
        const uint16_t* w1_row = p_->markov_w1.data() + (size_t) prev * (size_t) rank;
        for (int j = 0; j < rank; ++j) p_->h_me[j] = bf16_to_f32(w1_row[j]);
        cudaMemcpyAsync(p_->d_me, p_->h_me, (size_t) rank * sizeof(float), cudaMemcpyHostToDevice, p_->stream);

        p_->matmul_q8_on(mw2, p_->d_me, p_->d_mbias, rank, g.n_vocab, 1, p_->stream);
        dspark_markov_multimax_argmax(p_->d_base_logits + (size_t) i * g.n_vocab, p_->d_mbias,
                                      (const float*) mmr->ptr, (const float*) mmt->ptr, g.n_vocab, p_->d_amax_v,
                                      p_->d_amax_i, p_->d_tok, p_->stream);
        dspark_confidence((const float*) conf_w->ptr, p_->d_hcout + (size_t) i * H, p_->d_me, H, rank, p_->d_conf,
                          p_->stream);
        cudaMemcpyAsync(p_->h_tok, p_->d_tok, sizeof(int32_t), cudaMemcpyDeviceToHost, p_->stream);
        cudaMemcpyAsync(p_->h_conf, p_->d_conf, sizeof(float), cudaMemcpyDeviceToHost, p_->stream);
        if (!ck(cudaStreamSynchronize(p_->stream), "markov step sync", err)) return false;

        out.token[i] = *p_->h_tok;
        out.conf[i] = *p_->h_conf;
        prev = out.token[i];
    }
    out.n = n_gen;
    if (auto& st = p_->main->shared_tier(); st.tier && st.tier->pool()) st.tier->pool()->warm_stop();   // the windows' pool
    ++p_->n_blocks;
    p_->total_draft_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    return true;
}

}  // namespace basalt::models::ernie55
