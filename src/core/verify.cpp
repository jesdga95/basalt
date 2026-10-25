// src/core/verify.cpp - see include/basalt/core/verify.hpp.
#include "basalt/core/load_progress.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/remote_expert_opt.hpp"

#include "basalt/core/native_head.hpp"
#include "basalt/core/gdn_head_split.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/native_router.hpp"
#include "basalt/kernels/native_moe.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/fused_gr.hpp"
#include "basalt/kernels/mid_chain.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/cvec.hpp"
#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/gr.hpp"
#include "basalt/kernels/native_ple_postops.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/native_qsa.hpp"
#include "basalt/kernels/native_qsa_indexer.hpp"
#include "basalt/kernels/native_rope.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/ple.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"
#include "basalt/kernels/qsa_dec_attn.hpp"
#include "basalt/kernels/qsa_dec_fused.hpp"
#include "basalt/kernels/q5k_head.hpp"
#include "basalt/kernels/q6k_head.hpp"
#include "basalt/kernels/qsa_select.hpp"
#include "basalt/kernels/quantize_act.hpp"
#include "basalt/kernels/rope.hpp"
#include "basalt/kernels/s2_expert_grouped.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/kernels/shared_expert.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/dec_moe.hpp"

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>
#include "basalt/log.hpp"

namespace basalt::core {

// The CPU tier's cache warm-up.  After layer l's doorbell the parked pool workers score layer l + 1's router on layer
// l's FFN input (the ring's q8_1 rows; the router in int8 with one scale per 32 values, as the peer's warm-up has it:
// a prediction only), take every token's top k, rank the predicted experts only the CPU holds by their summed weight,
// and read the likeliest one into their caches.  The next batch stops the work; the experts are computed as before.
struct CpuWarm {
    std::vector<int8_t> rq;   ///< [n_layers][n_expert][n_embd]
    std::vector<float> rs;    ///< [n_layers][n_expert][n_embd / 32]
    int n_expert = 0, n_embd = 0, k = 0;
    size_t bytes = 0, down_off = 0;   ///< a blob's bytes, where its down rows start
    ExpertSource* src = nullptr;
    // the job in flight
    int layer = 0, n_tok = 0;
    const uint8_t* xq = nullptr;                      ///< the ring's q8_1 rows, tokens back to back
    const int32_t* res0 = nullptr, *res1 = nullptr;   ///< the layer's CUDA0 / peer residency (res1 may be null)
    std::vector<float> logits;                        ///< [kVerifyMaxT][n_expert]
    alignas(64) std::atomic<int> arrive{0};
    alignas(64) std::atomic<int> ready{0};
    const uint8_t* blob = nullptr;                    ///< the expert to read, or null
    std::atomic<uint64_t> sink{0};
};

// the MoE tail (dec_moe.hpp): planned on CUDA0 from device residency tables, the peer card served from mapped rings
// by one persistent kernel per window, the CPU pool by the host loop (only its own entries).
struct DecMoe {
    int64_t rings = 0;                       ///< rings per window (one per layer)
    int32_t* res0 = nullptr;                 ///< CUDA0: [n_layers * n_expert] CUDA0 slot or -1 (host_res mirrored)
    int32_t* res1 = nullptr;                 ///< CUDA0: the same for the peer's slots
    std::vector<int32_t> shadow0, shadow1;   ///< what res0 / res1 hold
    CpuWarm* warm = nullptr;                 ///< the CPU tier's warm-up (null: off)
    basalt::kernels::DmPlan* plan = nullptr;
    unsigned int* ctr = nullptr;
    unsigned long long* slot_off = nullptr;  ///< CUDA0's per-slot offsets (null: slot * blob)
    int vram = -1;   // the VRAM share's kernels (dec_moe_vram's kind; -1 grouped)
    void* vscr = nullptr;                    ///< its gate/up rows
    void* xq_pub = nullptr;   // the side stream's q8_1 rows for the ring
    unsigned int* vsync = nullptr;           ///< its counters
    unsigned long long* wstat = nullptr;   // the per-layer wait counters (device), with --wait-stats
    unsigned long long* ptab0 = nullptr;   // [n_layers * n_expert] CUDA0 blob address or 0
    unsigned int* pub = nullptr;           // [rings][4] the FFN read's counts for the ring's block (HcFusedArgs::pub_ctr)
    unsigned int* cseen = nullptr;         // the combine's last ring whose CPU word it saw (dec_moe_combine's cpu_seen)
    // [rings + 1]: per layer the last adaptive round whose copies into its slots are in, then the round the window's
    // gate/up launches wait for (DmGlueArgs::landed)
    unsigned int* landed = nullptr;
    uint32_t* landed_pin = nullptr;        // pinned: the need copies' sources, a ring of kPinSlots
    int landed_i = 0;
    int64_t wstat_windows = 0, wstat_t = 0;
    // mapped: per ring header, FFN input and q8_1 rows; [0] the peer's ring, [16] its done ring, [32] its abort
    // record (kDmDiagWords); the peer's rows
    basalt::kernels::DmRing* h_ring = nullptr; basalt::kernels::DmRing* m_ring = nullptr;
    float* h_x = nullptr; float* m_x = nullptr;
    uint8_t* h_xq = nullptr; uint8_t* m_xq = nullptr;
    uint32_t* h_w = nullptr; uint32_t* m_w = nullptr;
    float* h_rows = nullptr; float* m_rows = nullptr;
    bool tag = false;       ///< the combine polls the peer's rows (each dword not yet written holds the tag word)
    bool tags = false;      ///< the combine waits for the CPU pool's word itself: no wait launch (without --wait-stats)
    size_t rows_words = 0;
    bool tag_dirty = false; ///< a released window may have left written rows behind
    size_t x_stride = 0, xq_stride = 0;      ///< per ring, floats / bytes
    // the peer card
    int pdev = -1;
    cudaStream_t ps = nullptr;
    unsigned long long* ptab = nullptr;      ///< [n_layers * n_expert] blob address or 0
    basalt::kernels::NativeExpertLayout* lay = nullptr;
    void* pscr = nullptr;
    unsigned int* parr = nullptr;
    int pblocks = 0;
    int8_t* rq = nullptr;   ///< the warm-up's predictor: every layer's router in int8, one scale per 32 values (rs)
    float* rs = nullptr;
    bool pq2 = false;   // a layer's gate/up is Q2_0: the peer kernel's Q2_0 build
    bool peer_live = false;                  ///< a peer window was launched and not yet waited for
    /// a further peer card: its own residency, ring, doorbell and done word ([0], [16], [32] of h_w as the first's)
    /// and its own persistent window; the FFN input's q8_1 rows and the rows it writes are the first peer's.  With
    /// further peers the combine ends its row poll on the first peer's word 60, raised only by a release.
    struct XPeer {
        PeerExperts* peer = nullptr;
        int dev = -1;
        cudaStream_t ps = nullptr;
        int32_t* res = nullptr;                  ///< CUDA0: [n_layers * n_expert] its slot or -1
        std::vector<int32_t> shadow;             ///< what res holds
        unsigned long long* ptab = nullptr;      ///< its card: blob address or 0
        basalt::kernels::NativeExpertLayout* lay = nullptr;
        void* pscr = nullptr;
        unsigned int* parr = nullptr;
        int pblocks = 0;
        int8_t* rq = nullptr;
        float* rs = nullptr;
        basalt::kernels::DmRing* h_ring = nullptr; basalt::kernels::DmRing* m_ring = nullptr;
        uint32_t* h_w = nullptr; uint32_t* m_w = nullptr;
        bool live = false;
    };
    std::vector<XPeer> xp;
    // pinned staging of the residency uploads: the copies take their source from here, so the call does
    // not wait for the stream; a ring of slots, each reused only after the window's stream sync
    uint8_t* pin = nullptr;
    size_t pin_stride = 0;
    int pin_i = 0;
    static constexpr int kPinSlots = 8;
    std::vector<uint32_t> put_i;             ///< dm_sync: the peer entries that changed
};

namespace {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// layer `c.layer`'s router rows [e0, e1) on the job's tokens
BASALT_TARGET_AVX2_FMA_F16C void warm_logits(CpuWarm& c, int e0, int e1, const std::atomic<bool>& stop) {
    const int NB = c.n_embd / 32, pt = c.n_tok;
    const int8_t* R = c.rq.data() + (size_t) c.layer * c.n_expert * c.n_embd;
    const float* S = c.rs.data() + (size_t) c.layer * c.n_expert * NB;
    for (int e = e0; e < e1; ++e) {
        if (stop.load(std::memory_order_relaxed)) return;
        __m256 acc[basalt::kernels::kVerifyMaxT];
        for (int t = 0; t < pt; ++t) acc[t] = _mm256_setzero_ps();
        const int8_t* row = R + (size_t) e * c.n_embd;
        for (int b = 0; b < NB; ++b) {
            const __m256i w = _mm256_loadu_si256((const __m256i*) (row + b * 32));
            const __m256i wl = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
            const __m256i wh = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
            const float sw = S[(size_t) e * NB + b];
            for (int t = 0; t < pt; ++t) {
                const uint8_t* blk = c.xq + ((size_t) t * NB + b) * 36;   // block_q8_1: half d, half s, 32 x int8
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                const __m256i x = _mm256_loadu_si256((const __m256i*) (blk + 4));
                const __m256i p = _mm256_add_epi32(_mm256_madd_epi16(wl, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x))),
                                                   _mm256_madd_epi16(wh, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 1))));
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(sw * _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(dh)))), acc[t]);
            }
        }
        for (int t = 0; t < pt; ++t) {
            __m128 v = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
            v = _mm_add_ps(v, _mm_movehl_ps(v, v));
            v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
            c.logits[(size_t) t * c.n_expert + e] = _mm_cvtss_f32(v);
        }
    }
}

// every token's top k (softmax weights over them), summed per predicted expert only the CPU holds; the likeliest
void warm_select(CpuWarm& c) {
    const int NE = c.n_expert, K = std::min(c.k, 16);
    float score[512] = {};
    for (int t = 0; t < c.n_tok; ++t) {
        const float* lg = c.logits.data() + (size_t) t * NE;
        int top[16];
        float val[16];
        int m = 0;
        for (int e = 0; e < NE; ++e) {
            const float v = lg[e];
            if (m == K && v <= val[K - 1]) continue;
            int j = m < K ? m++ : K - 1;
            while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; top[j] = top[j - 1]; --j; }
            val[j] = v;
            top[j] = e;
        }
        float w[16], sum = 0.0f;
        for (int j = 0; j < m; ++j) sum += (w[j] = std::exp(val[j] - val[0]));
        for (int j = 0; j < m; ++j) {
            const int e = top[j];
            if (c.res0[e] >= 0 || (c.res1 != nullptr && c.res1[e] >= 0)) continue;   // a GPU holds it
            score[e] += w[j] / sum;
        }
    }
    int best = -1;
    for (int e = 0; e < NE; ++e)
        if (score[e] > 0.0f && (best < 0 || score[e] > score[best])) best = e;
    c.blob = best >= 0 ? c.src->resident_blob(c.layer, best) : nullptr;
}

void warm_job(void* ctx, int w, int W, const std::atomic<bool>& stop) {
    CpuWarm& c = *(CpuWarm*) ctx;
    warm_logits(c, c.n_expert * w / W, c.n_expert * (w + 1) / W, stop);
    if (c.arrive.fetch_add(1, std::memory_order_acq_rel) + 1 == W) {
        if (!stop.load(std::memory_order_relaxed)) warm_select(c);
        c.ready.store(1, std::memory_order_release);
    } else {
        while (c.ready.load(std::memory_order_acquire) == 0) {
            if (stop.load(std::memory_order_relaxed)) return;
            _mm_pause();
        }
    }
    // this worker's share of the blob, one load per line: the gate and up rows first (computed first), then the down
    // rows, so a read the doorbell cuts short has warmed the start of the computation
    if (c.blob == nullptr) return;
    uint64_t x = 0;
    for (int part = 0; part < 2; ++part) {
        const size_t r0 = part == 0 ? 0 : c.down_off, r1 = part == 0 ? c.down_off : c.bytes;
        const size_t per = ((r1 - r0 + (size_t) W - 1) / (size_t) W + 63) & ~(size_t) 63;
        const size_t b0 = std::min(r1, r0 + per * (size_t) w), b1 = std::min(r1, b0 + per);
        for (size_t o = b0; o < b1; o += 4096) {
            if (stop.load(std::memory_order_relaxed)) { part = 2; break; }
            const uint8_t* p = c.blob + o;
            const size_t n = std::min<size_t>(4096, b1 - o);
            for (size_t q = 0; q + 8 <= n; q += 64) x += *(const volatile uint64_t*) (p + q);
        }
    }
    if (x == 0x9e3779b97f4a7c15ull) c.sink.fetch_add(x, std::memory_order_relaxed);   // keeps the loads
}

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

// The decode window's overlap structure (all always on): in windows of 2+ tokens the shared expert's side kernels run
// at the greatest stream priority, so their blocks are scheduled ahead of the routed experts' queued blocks; in windows
// of 1-2 tokens a GDN layer's z projection is forked at that priority beside the qkv projection; the MoE route is
// split (CUDA0's plan and q8_1 rows on the main stream, the ring for the peer card and the CPU pool on a stream of its
// own) with the hc read's router items writing the FFN input's q8_1 rows and the gate/up launch building CUDA0's plan
// itself; the combine runs as the wait kernel's programmatic dependent.  A QSA layer's K, V and the indexer key/query
// run on branches of their own beside the query projection; its block scores skip a window whose selections are all
// the identity; the head's hyper-connection read does every column in one launch; layer 1's PLE takes two launches;
// the attention's merge also writes the output projection's q8_1 rows.  The attention's hc read writes the
// projections' q8_1 rows, and a GDN layer's qkv and z projections are one dense_tc_pair launch.
// The commit graph is two launches, its GDN part from 32 persistent blocks so it trickles beside the draft instead of
// filling the card
constexpr int kCommitBlocks = 32;
bool g_commit_async = false;   // see Verifier::set_commit_async

// Every layer's BF16 router [n_expert x n_embd] as int8 with one scale per 32 values (the warm-ups' predictor, a
// prediction only).  False when a layer's router is not such a table.
bool router_int8(const WeightTable& wt, const ModelGeometry& g, std::vector<int8_t>& q, std::vector<float>& sc) {
    const int64_t N = g.n_embd, NE = g.n_expert, L = g.n_layers, NB = N / 32;
    if (NE > basalt::kernels::kDmPredExperts || N % 32 != 0) return false;
    std::vector<uint16_t> src((size_t) (NE * N));
    q.assign((size_t) (L * NE * N), 0);
    sc.assign((size_t) (L * NE * NB), 0.0f);
    for (int64_t l = 0; l < L; ++l) {
        const WeightRef* w = LayerView(wt, l).get("ffn_gate_inp.weight");
        if (w == nullptr || w->bytes != (uint64_t) (NE * N * 2)) return false;
        if (cudaMemcpy(src.data(), w->data, (size_t) w->bytes, cudaMemcpyDefault) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        for (int64_t r = 0; r < NE; ++r)
            for (int64_t b = 0; b < NB; ++b) {
                float v[32], amax = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    const uint32_t u = (uint32_t) src[(size_t) (r * N + b * 32 + i)] << 16;
                    std::memcpy(&v[i], &u, 4);
                    amax = std::max(amax, std::fabs(v[i]));
                }
                const float d = amax / 127.0f;
                sc[(size_t) ((l * NE + r) * NB + b)] = d;
                for (int i = 0; i < 32; ++i)
                    q[(size_t) ((l * NE + r) * N + b * 32 + i)] = (int8_t) (d > 0.0f ? std::lrintf(v[i] / d) : 0);
            }
    }
    return true;
}

// The peer warm-up's predictor rows (router_int8's) on the current device.
bool warm_router(const std::vector<int8_t>& q, const std::vector<float>& sc, int8_t** rq, float** rs) {
    if (cudaMalloc((void**) rq, q.size()) != cudaSuccess || cudaMalloc((void**) rs, sc.size() * 4) != cudaSuccess ||
        cudaMemcpy(*rq, q.data(), q.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(*rs, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();
        if (*rq) cudaFree(*rq);
        if (*rs) cudaFree(*rs);
        *rq = nullptr;
        *rs = nullptr;
        return false;
    }
    return true;
}

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped | (peer_portable() ? cudaHostAllocPortable : 0)) != cudaSuccess) return false;   // multi-GPU: portable only with a peer - the peer card writes its rows into them
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

basalt::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively";
        return false;
    }
    return true;
}

}  // namespace

namespace {
std::atomic<const Verifier*> g_diag_verifier{nullptr};
void diag_active_verifier(std::FILE* f) {
    if (const Verifier* v = g_diag_verifier.load()) v->diag(f);
}
// every live verifier (a layer split has one per stage), for the release before the engine ends
constexpr int kLiveMax = 16;
std::atomic<Verifier*> g_live[kLiveMax];
void release_live_verifiers(std::FILE* f) {
    for (auto& slot : g_live)
        if (Verifier* v = slot.load()) {
            const Clock::time_point t0 = Clock::now();
            const bool done = v->release_gpu_waits(5000);
            if (f != nullptr)
                std::fprintf(f, "basalt: released the verify window's GPU waits: the GPU %s\n",
                             done ? ("finished in " + std::to_string((long long) ms_since(t0)) + " ms").c_str()
                                  : "did not finish within 5 s");
        }
    if (f != nullptr) std::fflush(f);
}
std::string released_note(bool drained) {
    return drained ? "; its GPU waits were released and the GPU finished"
                   : "; its GPU waits were released but the GPU did not finish within 5 s";
}
// one stderr line per distinct fast-path fallback (what, a, b)
void note_fallback(const char* what, int a, int b) {
    static std::mutex mu;
    static std::set<std::tuple<std::string, int, int>> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert({what, a, b}).second)
        basalt::log::warn("basalt verify", "fallback: %s (types %d / %d)", what, a, b);
}
// a dense projection of the window: dense_tc where dense_tc_mask has the projection's bit and the type, else
// native_mmvq (either one for every window width, so a column's value never depends on T)
void dense_proj(int bit, int type, const void* w, const void* xq, float* y, int n_in, int n_out, int n, void* s) {
    if ((kernels::dense_tc_mask() & bit) && kernels::dense_tc_supported(type, n_in, n_out, w) &&
        kernels::dense_tc_type_on(type))
        kernels::dense_tc(type, w, xq, y, n_in, n_out, n, s);
    else {
        if (kernels::dense_tc_mask() & bit) note_fallback("dense_tc projection on native_mmvq", bit, type);
        kernels::native_mmvq(type, w, xq, y, n_in, n_out, n, s);
    }
}
}  // namespace

bool Verifier::release_gpu_waits(int timeout_ms) {
    released_.store(true);
    // the words the spin kernels read (wait_flag_ge, wait_flag_ge_or) are mapped host memory, so a store here reaches
    // them with no API call; UINT32_MAX is past every ring.  (The skip words are device memory, but wait_flag_ge_or
    // also returns on its flag.)  A host function raising flag B later only raises.
    for (uint32_t* p : {h_flag_, h_flagA_, h_flagB_})
        if (p != nullptr) *(volatile uint32_t*) p = UINT32_MAX;
    if (h_flag_ != nullptr) *(volatile uint32_t*) (h_flag_ + 8) = UINT32_MAX;
    if (dm_ != nullptr) {   // the peer window stops at its next ring; CUDA0's waits on it pass
        *(volatile uint32_t*) (dm_->h_w + 0) = UINT32_MAX;
        *(volatile uint32_t*) (dm_->h_w + 16) = UINT32_MAX;
        *(volatile uint32_t*) (dm_->h_w + 48) = UINT32_MAX;   // the mixer's doorbell and done word
        *(volatile uint32_t*) (dm_->h_w + 52) = UINT32_MAX;
        *(volatile uint32_t*) (dm_->h_w + 60) = UINT32_MAX;   // the release word the combine polls with further peers
        for (DecMoe::XPeer& x : dm_->xp) {
            *(volatile uint32_t*) (x.h_w + 0) = UINT32_MAX;
            *(volatile uint32_t*) (x.h_w + 16) = UINT32_MAX;
        }
        dm_->tag_dirty = true;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    for (cudaStream_t s : {cs_}) {
        if (s == nullptr) continue;
        while (cudaStreamQuery(s) == cudaErrorNotReady) {
            if (ms_since(t0) > timeout_ms) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return true;
}

std::string Verifier::dm_note() const {
    const DecMoe* d = dm_;
    if (d == nullptr) return "";
    auto rd = [](const uint32_t* p) { return *(const volatile uint32_t*) p; };
    const uint32_t seq = rd(h_seq_);
    std::string s = "; device plan: ring " + std::to_string(seq) + " published";
    if (seq >= 1 && (int64_t) seq <= d->rings) {
        const basalt::kernels::DmRing& r = d->h_ring[seq - 1];
        s += " (entries: VRAM " + std::to_string(r.n_vram) + ", peer " + std::to_string(r.n_peer) + ", CPU " +
             std::to_string(r.n_cpu) + ")";
    }
    s += ", host flag " + std::to_string(rd(h_flag_));
    if (d->pdev >= 0) {
        const uint32_t w = rd(d->h_w + 0), done = rd(d->h_w + 16);
        std::string st = "not launched";
        if (d->peer_live) {
            const OnDevice on(d->pdev);
            const cudaError_t q = cudaStreamQuery(d->ps);
            st = q == cudaErrorNotReady ? "running" : q == cudaSuccess ? "finished" : cudaGetErrorString(q);
        }
        s += "; peer card: rung for ring " + std::to_string(w & 0xffffu) + ((w >> 28) & 1u ? " (with entries)" : "") +
             ", finished ring " + std::to_string(done) + ", its kernel " + st;
        const std::string ab = basalt::kernels::dec_moe_peer_diag(d->h_w + 32);
        if (!ab.empty()) s += "; the peer window " + ab;
    }
    for (size_t q = 0; q < d->xp.size(); ++q) {
        const DecMoe::XPeer& x = d->xp[q];
        const uint32_t w = rd(x.h_w + 0), done = rd(x.h_w + 16);
        s += "; peer " + std::to_string(q + 2) + ": rung for ring " + std::to_string(w & 0xffffu) +
             ((w >> 28) & 1u ? " (with entries)" : "") + ", finished ring " + std::to_string(done);
        const std::string ab = basalt::kernels::dec_moe_peer_diag(x.h_w + 32);
        if (!ab.empty()) s += ", its window " + ab;
    }
    return s;
}

// After release_gpu_waits: whether the peer window ended once its doorbell was raised (it waited for CUDA0) or
// still runs (it is stuck inside a ring).
std::string Verifier::peer_released() {
    DecMoe* d = dm_;
    if (d == nullptr || d->pdev < 0 || !d->peer_live) return "";
    const OnDevice on(d->pdev);
    const Clock::time_point t0 = Clock::now();
    cudaError_t q;
    while ((q = cudaStreamQuery(d->ps)) == cudaErrorNotReady && ms_since(t0) < 2000)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (q == cudaErrorNotReady) return "; the peer kernel still runs after the release";
    d->peer_live = false;
    for (DecMoe::XPeer& x : d->xp) {
        if (!x.live) continue;
        const OnDevice onx(x.dev);
        const Clock::time_point tx = Clock::now();
        cudaError_t qx;
        while ((qx = cudaStreamQuery(x.ps)) == cudaErrorNotReady && ms_since(tx) < 2000)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (qx == cudaErrorNotReady) return "; a further peer's kernel still runs after the release";
        x.live = false;
    }
    return q == cudaSuccess ? "; the peer kernel finished once released" : std::string("; the peer kernel: ") + cudaGetErrorString(q);
}

void Verifier::diag(std::FILE* f) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    // outside a verify stage these are the LAST window's numbers (it finished), not the stalled work's
    const char* where = progress().where.load();
    const bool current = where != nullptr && std::strncmp(where, "verify window", 13) == 0;
    std::fprintf(f, "  verify window%s: %d tokens at position %lld, host at layer step %u; the GPU rang %u; flags: "
                    "served %u, plan (A) %u, copies (B) %u\n", current ? "" : " (last window, not the current stage)",
                 last_t_, (long long) last_pos0_, cur_layer_ + 1, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_));
    if (dm_ != nullptr) std::fprintf(f, "  %s\n", dm_note().c_str() + 2);
}

bool Verifier::dm_init(std::string& err) {
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    const auto& lay = basalt::kernels::cpu::expert_layout();
    const char* why = nullptr;
    if (!lay.native) why = "the experts are not a native pack";
    else if (remote_opt_ != nullptr) why = "--remote-expert-opt";
    else if (all_resident_) why = "every expert is resident";
    else if (hits_.h_res == nullptr) why = "no host residency table";
    else if ((int64_t) max_t_ * ss_->k > kDmCap) why = "the window is too large";
    for (int64_t l = 0; why == nullptr && l < g.n_layers; ++l) {
        const auto& f = lay.fmt[(size_t) l];
        if (!dec_moe_formats_ok(f.gu_type, f.d_type, f.n_embd, f.n_ff)) why = "an expert format the kernels do not cover";
    }
    if (why != nullptr) {
        basalt::log::warn("basalt verify", "the device-planned MoE tail is off: %s", why);
        return true;
    }
    DecMoe* d = new DecMoe;
    dm_ = d;
    const int64_t LE = g.n_layers * g.n_expert, N = g.n_embd, T = max_t_, K = ss_->k;
    d->rings = le_ - lb_;
    d->x_stride = (size_t) (T * N);
    d->xq_stride = (size_t) (T * (N / 32) * 36);
    bool ok = cudaMalloc((void**) &d->res0, (size_t) LE * 4) == cudaSuccess &&
              cudaMalloc((void**) &d->res1, (size_t) LE * 4) == cudaSuccess &&
              cudaMalloc((void**) &d->plan, sizeof(DmPlan)) == cudaSuccess &&
              cudaMalloc((void**) &d->ctr, 64) == cudaSuccess && cudaMemset(d->ctr, 0, 64) == cudaSuccess &&
              cudaMemset(d->res1, 0xff, (size_t) LE * 4) == cudaSuccess &&
              mapped((size_t) d->rings * sizeof(DmRing), (void**) &d->h_ring, (void**) &d->m_ring) &&
              mapped((size_t) d->rings * d->x_stride * 4, (void**) &d->h_x, (void**) &d->m_x) &&
              mapped((size_t) d->rings * d->xq_stride, (void**) &d->h_xq, (void**) &d->m_xq) &&
              mapped(256, (void**) &d->h_w, (void**) &d->m_w) &&
              mapped((size_t) (T * K * N) * 4, (void**) &d->h_rows, (void**) &d->m_rows) &&
              cudaMalloc(&d->vscr, dec_moe_vram_scratch_bytes((int) lay.fmt[0].n_ff)) == cudaSuccess &&
              cudaMalloc((void**) &d->vsync, (kDmCap + 1) * 4) == cudaSuccess &&
              cudaMemset(d->vsync, 0, (kDmCap + 1) * 4) == cudaSuccess &&
              cudaMalloc(&d->xq_pub, d->xq_stride) == cudaSuccess &&
              cudaMalloc((void**) &d->pub, (size_t) d->rings * 16) == cudaSuccess &&
              cudaMemset(d->pub, 0, (size_t) d->rings * 16) == cudaSuccess &&
              cudaMalloc((void**) &d->cseen, 4) == cudaSuccess && cudaMemset(d->cseen, 0, 4) == cudaSuccess &&
              cudaMalloc((void**) &d->landed, (size_t) (d->rings + 1) * 4) == cudaSuccess &&
              cudaMemset(d->landed, 0, (size_t) (d->rings + 1) * 4) == cudaSuccess;
    if (ok && cudaHostAlloc((void**) &d->landed_pin, DecMoe::kPinSlots * 4, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        d->landed_pin = nullptr;
    }
    d->vram = -1;
    if (ok && hits_.slot_off != nullptr && hits_.n_slots > 0)
        ok = cudaMalloc((void**) &d->slot_off, (size_t) hits_.n_slots * 8) == cudaSuccess &&
             cudaMemcpy(d->slot_off, hits_.slot_off, (size_t) hits_.n_slots * 8, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok)
        ok = cudaMalloc((void**) &d->ptab0, (size_t) LE * 8) == cudaSuccess &&
             cudaMemset(d->ptab0, 0, (size_t) LE * 8) == cudaSuccess;
    if (ok && diagnostics().wait_stats)
        ok = cudaMalloc((void**) &d->wstat, kDmStatWords * 8) == cudaSuccess &&
             cudaMemset(d->wstat, 0, kDmStatWords * 8) == cudaSuccess;
    if (!ok) { err = "verify: device-planned MoE buffers"; return false; }
    d->rows_words = (size_t) (T * K * N);
    d->tags = d->wstat == nullptr;
    d->shadow0.assign((size_t) LE, INT32_MIN);
    d->shadow1.assign((size_t) LE, -1);
    d->pin_stride = (size_t) LE * 24;   // [res0 | ptab0 | res1 | ptab1] per slot
    if (cudaHostAlloc((void**) &d->pin, d->pin_stride * DecMoe::kPinSlots, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        d->pin = nullptr;
    }
    // the warm-ups' predictor (the peer's and the CPU pool's): every layer's router in int8, quantized once
    const bool cpu_warm = warm_pool_ != nullptr && warm_src_ != nullptr && K <= 16 &&
                          std::all_of(lay.fmt.begin(), lay.fmt.end(), [&](const basalt::kernels::cpu::NativeFmt& f) {
                              return f.bytes == lay.fmt[0].bytes && f.down_off == lay.fmt[0].down_off;
                          });
    std::vector<int8_t> rq_h;
    std::vector<float> rs_h;
    const bool router_ok = (cpu_warm || (peer_ != nullptr && peer_->valid())) && router_int8(*wt_, g, rq_h, rs_h);
    if (peer_ != nullptr && peer_->valid()) {
        const OnDevice on(peer_->device());
        d->pdev = peer_->device();
        std::vector<NativeExpertLayout> ls((size_t) g.n_layers);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const auto& f = lay.fmt[(size_t) l];
            ls[(size_t) l] = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            d->pq2 |= f.gu_type == 42;
        }
        d->pblocks = dec_moe_peer_blocks(d->pq2);
        ok = d->pblocks > 0 && cudaStreamCreateWithFlags(&d->ps, cudaStreamNonBlocking) == cudaSuccess &&
             basalt::kernels::dec_moe_table_load(d->ps) &&
             cudaMalloc((void**) &d->ptab, (size_t) LE * 8) == cudaSuccess &&
             cudaMemset(d->ptab, 0, (size_t) LE * 8) == cudaSuccess &&
             cudaMalloc((void**) &d->lay, ls.size() * sizeof(NativeExpertLayout)) == cudaSuccess &&
             cudaMemcpy(d->lay, ls.data(), ls.size() * sizeof(NativeExpertLayout), cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMalloc(&d->pscr, dec_moe_peer_scratch_bytes((int) N, (int) lay.fmt[0].n_ff)) == cudaSuccess &&
             cudaMemset(d->pscr, 0, dec_moe_peer_scratch_bytes((int) N, (int) lay.fmt[0].n_ff)) == cudaSuccess &&
             cudaMalloc((void**) &d->parr, 64) == cudaSuccess;
        d->tag = true;
        for (size_t i = 0; i < d->rows_words; ++i) ((volatile uint32_t*) d->h_rows)[i] = basalt::kernels::dec_moe_tag_word();
        if (!ok) { err = "verify: device-planned MoE peer buffers"; return false; }
        d->shadow1.assign((size_t) LE, INT32_MIN);
        if (!router_ok || !warm_router(rq_h, rs_h, &d->rq, &d->rs))
            basalt::log::warn("basalt verify", "the peer's L2 warm-up is off (a router the predictor does not cover)");
        // further peers: a window, ring and doorbell each; cards that hold several peers split their blocks
        std::vector<int> devs = {d->pdev};
        for (PeerExperts* xpe : xpeers_)
            if (xpe != nullptr && xpe->valid()) devs.push_back(xpe->device());
        auto sharing = [&](int dev) { return (int) std::count(devs.begin(), devs.end(), dev); };
        d->pblocks = std::max(1, d->pblocks / sharing(d->pdev));
        for (PeerExperts* xpe : xpeers_) {
            if (xpe == nullptr || !xpe->valid()) continue;
            if ((int) d->xp.size() == kDmMaxX) {
                basalt::log::warn("basalt verify", "the device plan serves %d further peers; the others' experts run on the CPU",
                                  kDmMaxX);
                break;
            }
            d->xp.emplace_back();
            DecMoe::XPeer& x = d->xp.back();
            x.peer = xpe;
            x.dev = xpe->device();
            {   // CUDA0 reads its residency
                const OnDevice on0(device_);
                ok = cudaMalloc((void**) &x.res, (size_t) LE * 4) == cudaSuccess &&
                     cudaMemset(x.res, 0xff, (size_t) LE * 4) == cudaSuccess &&
                     mapped((size_t) d->rings * sizeof(DmRing), (void**) &x.h_ring, (void**) &x.m_ring) &&
                     mapped(256, (void**) &x.h_w, (void**) &x.m_w);
            }
            {
                const OnDevice onx(x.dev);
                x.pblocks = std::max(1, dec_moe_peer_blocks(d->pq2) / sharing(x.dev));
                ok = ok && cudaStreamCreateWithFlags(&x.ps, cudaStreamNonBlocking) == cudaSuccess &&
                     basalt::kernels::dec_moe_table_load(x.ps) &&
                     cudaMalloc((void**) &x.ptab, (size_t) LE * 8) == cudaSuccess &&
                     cudaMemset(x.ptab, 0, (size_t) LE * 8) == cudaSuccess &&
                     cudaMalloc((void**) &x.lay, ls.size() * sizeof(NativeExpertLayout)) == cudaSuccess &&
                     cudaMemcpy(x.lay, ls.data(), ls.size() * sizeof(NativeExpertLayout), cudaMemcpyHostToDevice) == cudaSuccess &&
                     cudaMalloc(&x.pscr, dec_moe_peer_scratch_bytes((int) N, (int) lay.fmt[0].n_ff)) == cudaSuccess &&
                     cudaMemset(x.pscr, 0, dec_moe_peer_scratch_bytes((int) N, (int) lay.fmt[0].n_ff)) == cudaSuccess &&
                     cudaMalloc((void**) &x.parr, 64) == cudaSuccess;
                if (ok && router_ok && d->rq != nullptr) (void) warm_router(rq_h, rs_h, &x.rq, &x.rs);
            }
            x.shadow.assign((size_t) LE, INT32_MIN);
            if (!ok) { err = "verify: device-planned MoE buffers of a further peer"; return false; }
        }
    }
    if (d->pdev >= 0) {
        hs_init();
        gs_init();
    }
    if (cpu_warm && router_ok) {
        CpuWarm* w = new CpuWarm;
        w->rq = std::move(rq_h);
        w->rs = std::move(rs_h);
        w->n_expert = (int) g.n_expert; w->n_embd = (int) N; w->k = (int) K; w->src = warm_src_;
        w->bytes = lay.fmt[0].bytes;
        w->down_off = lay.fmt[0].down_off;
        w->logits.assign((size_t) kVerifyMaxT * g.n_expert, 0.0f);
        d->warm = w;
    }
    std::string served;
    if (d->pdev >= 0) {
        served = ", the peer card served by a persistent kernel (" + std::to_string(d->pblocks) + " blocks";
        for (const DecMoe::XPeer& x : d->xp) served += ", then " + std::to_string(x.pblocks) + " on CUDA" + std::to_string(x.dev);
        served += ")";
    }
    basalt::log::info("basalt verify", "the MoE tail planned on the device%s", served.c_str());
    if (d->warm != nullptr)
        basalt::log::info("basalt verify", "the CPU pool warms its likeliest next-layer expert between doorbells");
    return true;
}

// A host-to-device upload in pieces of at most kSmallCopy bytes: a copy that small passes the expert copies queued on
// the card's copy engine (an adaptive round's), a larger one waits behind all of them.
constexpr size_t kSmallCopy = 16384;
static cudaError_t copy_small(void* dst, const void* src, size_t bytes, cudaStream_t s) {
    for (size_t o = 0; o < bytes; o += kSmallCopy) {
        const cudaError_t e = cudaMemcpyAsync((uint8_t*) dst + o, (const uint8_t*) src + o, std::min(kSmallCopy, bytes - o),
                                              cudaMemcpyHostToDevice, s);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

// The residency tables the window classifies with: host_res and the peer's, as they are now (between windows).
// Each upload is queued on the stream that reads it (CUDA0's window, the peer's): a plain cudaMemcpy from pageable
// memory may return before its DMA lands, and the window's route kernel would then read a table half old, half new
// (an expert with two tiers in one ring).
bool Verifier::dm_sync(std::string& err) {
    DecMoe* d = dm_;
    const size_t LE = d->shadow0.size();
    auto span = [&](const int32_t* now, std::vector<int32_t>& sh, size_t& a, size_t& b) {
        a = LE; b = 0;
        if (std::memcmp(now, sh.data(), LE * 4) == 0) return false;
        for (size_t i = 0; i < LE; ++i)
            if (now[i] != sh[i]) { if (a == LE) a = i; b = i + 1; }
        if (a < b) std::memcpy(sh.data() + a, now + a, (b - a) * 4);
        return a < b;
    };
    size_t a = 0, b = 0;
    uint8_t* pin = nullptr;
    if (d->pin != nullptr) {
        pin = d->pin + (size_t) d->pin_i * d->pin_stride;
        d->pin_i = (d->pin_i + 1) % DecMoe::kPinSlots;
    }
    int32_t* const pin_res0 = (int32_t*) pin;
    unsigned long long* const pin_ptab0 = (unsigned long long*) (pin + (pin ? (size_t) LE * 4 : 0));
    int32_t* const pin_res1 = (int32_t*) (pin + (pin ? (size_t) LE * 12 : 0));
    unsigned long long* const pin_ptab1 = (unsigned long long*) (pin + (pin ? (size_t) LE * 16 : 0));
    const bool up0 = span(hits_.h_res, d->shadow0, a, b);
    if (up0) {
        decode_events().tag(1u << 4);
        const int32_t* src = d->shadow0.data() + a;
        if (pin) { std::memcpy(pin_res0 + a, src, (b - a) * 4); src = pin_res0 + a; }
        if (copy_small(d->res0 + a, src, (b - a) * 4, cs_) != cudaSuccess) {
            err = "verify: device-planned MoE residency upload";
            return false;
        }
    }
    if (d->ptab0 != nullptr && a < b) {   // the same span as blob addresses (a < b only after an upload above)
        std::vector<unsigned long long> pv;
        unsigned long long* p = pin_ptab0 + a;
        if (!pin) { pv.resize(b - a); p = pv.data(); }
        for (size_t i = a; i < b; ++i) {
            const int32_t slot = d->shadow0[i];
            p[i - a] = slot >= 0 ? (unsigned long long) (hits_.cache_base + (hits_.slot_off ? (size_t) hits_.slot_off[slot]
                                                                                             : (size_t) slot * hits_.blob))
                                 : 0ull;
        }
        if (copy_small(d->ptab0 + a, p, (b - a) * 8, cs_) != cudaSuccess) {
            err = "verify: glue table upload";
            return false;
        }
    }
    std::vector<uint32_t>& ch = d->put_i;
    ch.clear();
    if (d->pdev >= 0) {
        const int32_t* now = peer_->residency().data();
        if (std::memcmp(now, d->shadow1.data(), LE * 4) != 0)
            for (size_t i = 0; i < LE; ++i)
                if (now[i] != d->shadow1[i]) ch.push_back((uint32_t) i);
        if (!ch.empty()) {
            a = ch.front();
            b = (size_t) ch.back() + 1;
            std::memcpy(d->shadow1.data() + a, now + a, (b - a) * 4);
        }
    }
    if (!ch.empty()) {
        decode_events().tag(1u << 5);
        const int32_t* src = d->shadow1.data() + a;
        if (pin) { std::memcpy(pin_res1 + a, src, (b - a) * 4); src = pin_res1 + a; }
        if (copy_small(d->res1 + a, src, (b - a) * 4, cs_) != cudaSuccess) {
            err = "verify: device-planned MoE peer residency upload";
            return false;
        }
        auto addr = [&](size_t i) {
            return d->shadow1[i] >= 0 ? (unsigned long long) peer_->slot_address(d->shadow1[i]) : 0ull;
        };
        const OnDevice on(d->pdev);
        if (ch.size() > (size_t) basalt::kernels::kDmTablePutMax * 4) {   // a wholesale change (the first window)
            std::vector<unsigned long long> pv;
            unsigned long long* p = pin_ptab1 + a;
            if (!pin) { pv.resize(b - a); p = pv.data(); }
            for (size_t i = a; i < b; ++i) p[i - a] = addr(i);
            if (cudaMemcpyAsync(d->ptab + a, p, (b - a) * 8, cudaMemcpyHostToDevice, d->ps) != cudaSuccess) {
                err = "verify: device-planned MoE peer table upload";
                return false;
            }
        } else {
            // the changed entries ride in kernel arguments: the peer window behind them on the stream never waits
            // for the refill copies queued ahead of it on the card's copy engine
            basalt::kernels::DmTablePut tp;
            tp.tab = d->ptab;
            for (size_t k = 0; k < ch.size(); k += (size_t) basalt::kernels::kDmTablePutMax) {
                tp.n = (int) std::min(ch.size() - k, (size_t) basalt::kernels::kDmTablePutMax);
                for (int j = 0; j < tp.n; ++j) {
                    tp.idx[j] = ch[k + (size_t) j];
                    tp.val[j] = addr(ch[k + (size_t) j]);
                }
                if (!basalt::kernels::dec_moe_table_put(tp, d->ps)) {
                    err = "verify: device-planned MoE peer table update";
                    return false;
                }
            }
        }
    }
    // further peers: pageable sources, so each copy has taken its bytes when the call returns
    for (DecMoe::XPeer& x : d->xp) {
        ch.clear();
        const int32_t* now = x.peer->residency().data();
        if (std::memcmp(now, x.shadow.data(), LE * 4) != 0)
            for (size_t i = 0; i < LE; ++i)
                if (now[i] != x.shadow[i]) ch.push_back((uint32_t) i);
        if (ch.empty()) continue;
        a = ch.front();
        b = (size_t) ch.back() + 1;
        std::memcpy(x.shadow.data() + a, now + a, (b - a) * 4);
        if (copy_small(x.res + a, x.shadow.data() + a, (b - a) * 4, cs_) != cudaSuccess) {
            err = "verify: device-planned MoE residency upload of a further peer";
            return false;
        }
        auto addr = [&](size_t i) {
            return x.shadow[i] >= 0 ? (unsigned long long) x.peer->slot_address(x.shadow[i]) : 0ull;
        };
        const OnDevice on(x.dev);
        if (ch.size() > (size_t) basalt::kernels::kDmTablePutMax * 4) {   // a wholesale change (the first window)
            std::vector<unsigned long long> pv(b - a);
            for (size_t i = a; i < b; ++i) pv[i - a] = addr(i);
            if (cudaMemcpyAsync(x.ptab + a, pv.data(), (b - a) * 8, cudaMemcpyHostToDevice, x.ps) != cudaSuccess) {
                err = "verify: device-planned MoE table upload of a further peer";
                return false;
            }
            continue;
        }
        basalt::kernels::DmTablePut tp;
        tp.tab = x.ptab;
        for (size_t k = 0; k < ch.size(); k += (size_t) basalt::kernels::kDmTablePutMax) {
            tp.n = (int) std::min(ch.size() - k, (size_t) basalt::kernels::kDmTablePutMax);
            for (int j = 0; j < tp.n; ++j) {
                tp.idx[j] = ch[k + (size_t) j];
                tp.val[j] = addr(ch[k + (size_t) j]);
            }
            if (!basalt::kernels::dec_moe_table_put(tp, x.ps)) {
                err = "verify: device-planned MoE table update of a further peer";
                return false;
            }
        }
    }
    return true;
}

unsigned int* Verifier::landed_words() const { return dm_ != nullptr && dm_->landed_pin != nullptr ? dm_->landed : nullptr; }

bool Verifier::landed_ok() const {
    if (landed_words() == nullptr || !landed_glue_) return false;
    for (int t = 1; t <= max_t_; ++t)
        if ((ar_off_ ? exec_nr_[t] : exec_[t]) == nullptr) return false;
    return true;
}

bool Verifier::landed_need(uint32_t round, std::string& err) {
    if (landed_words() == nullptr) return true;
    const OnDevice on_device(device_);
    uint32_t* src = dm_->landed_pin + dm_->landed_i;
    dm_->landed_i = (dm_->landed_i + 1) % DecMoe::kPinSlots;
    *src = round;
    if (cudaMemcpyAsync(dm_->landed + dm_->rings, src, 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
        err = "verify: the landed words' upload";
        return false;
    }
    return true;
}

bool Verifier::dm_fence(cudaStream_t s) {
    if (dm_ == nullptr) return true;
    const OnDevice on_device(device_);
    if (fence_ev_ == nullptr && cudaEventCreateWithFlags(&fence_ev_, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        fence_ev_ = nullptr;
        return false;
    }
    return cudaEventRecord(fence_ev_, cs_) == cudaSuccess && cudaStreamWaitEvent(s, fence_ev_, 0) == cudaSuccess;
}

// Memory bandwidth of a card from its attributes, bytes per second.
static double card_bandwidth(int dev) {
    int khz = 0, bits = 0;
    if (cudaDeviceGetAttribute(&khz, cudaDevAttrMemoryClockRate, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&bits, cudaDevAttrGlobalMemoryBusWidth, dev) != cudaSuccess) {
        cudaGetLastError();
        return 0.0;
    }
    return 2.0 * khz * 1e3 * bits / 8.0;
}

// The head's rows are split by the time each card takes for its share: CUDA0 streams its rows, the peer streams
// its rows and its logits cross PCIe twice (to the host, to CUDA0).  A split that leaves CUDA0 most of the rows
// does not pay for its two hand-offs, so it is taken only when the peer's share is large.
// CUDA0's rows of the head split by vocabulary (0: no split), from the cards' bandwidths.
static int hs_plan(size_t rb, int type, int V, int N, int T, int dev0, int dev1) {
    if (type == 13 ? !basalt::kernels::q5k_head_supported(N, T) : type == 14) return 0;
    const double bw0 = card_bandwidth(dev0), bw1 = card_bandwidth(dev1);
    if (bw0 <= 0.0 || bw1 <= 0.0 || rb == 0) return 0;
    const double pcie = 25e9, cols = 3.0;   // the hand-off rate and a typical window
    const double c0 = (double) rb / bw0, c1 = (double) rb / bw1 + 2.0 * cols * 4.0 / pcie;
    const double share0 = c1 / (c0 + c1);
    if (share0 > 0.75) return 0;
    const int rows = (int) (share0 * V) / 256 * 256;
    return rows <= 0 || rows >= V ? 0 : rows;
}

int64_t Verifier::hs_peer_bytes(size_t row_bytes, int type, int64_t n_vocab, int64_t n_embd, int max_t, int dev0, int dev1) {
    if (dev1 < 0 || dev1 == dev0) return 0;
    const int rows = hs_plan(row_bytes, type, (int) n_vocab, (int) n_embd, max_t, dev0, dev1);
    if (rows == 0) return 0;
    const int64_t r1 = n_vocab - rows;
    return r1 * (int64_t) row_bytes + (int64_t) basalt::kernels::native_q8_1_bytes((int) n_embd, max_t) +
           (int64_t) max_t * r1 * 4;
}

void Verifier::hs_init() {
    if (head_ == nullptr || !head_->loaded() || head_->vproxy() != nullptr || dm_ == nullptr || dm_->pdev < 0) return;
    const int N = (int) g_->n_embd, V = (int) n_vocab_, T = max_t_;
    const size_t rb = head_->row_bytes();
    const int rows = hs_plan(rb, head_->type(), V, N, T, device_, dm_->pdev);
    if (rows == 0) return;
    const int r1 = V - rows;
    const size_t xb = basalt::kernels::native_q8_1_bytes(N, T);
    bool ok;
    {
        const OnDevice on(dm_->pdev);
        ok = cudaMalloc(&hs_w_, (size_t) r1 * rb) == cudaSuccess && cudaMalloc(&hs_xq_, xb) == cudaSuccess &&
             cudaMalloc((void**) &hs_y_, (size_t) T * r1 * 4) == cudaSuccess &&
             cudaStreamCreateWithFlags(&hs_s_, cudaStreamNonBlocking) == cudaSuccess &&
             cudaEventCreateWithFlags(&hs_out_, cudaEventDisableTiming) == cudaSuccess;
        ok = ok && cudaMemcpyPeer(hs_w_, dm_->pdev, (const uint8_t*) head_->weights() + (size_t) rows * rb, device_,
                                  (size_t) r1 * rb) == cudaSuccess;
    }
    {
        const OnDevice on(device_);
        ok = ok && cudaMalloc((void**) &hs_y0_, (size_t) T * rows * 4) == cudaSuccess &&
             cudaEventCreateWithFlags(&hs_in_, cudaEventDisableTiming) == cudaSuccess &&
             cudaHostAlloc(&hs_hxq_, xb, cudaHostAllocPortable) == cudaSuccess &&
             cudaHostAlloc((void**) &hs_hy_, (size_t) T * r1 * 4, cudaHostAllocPortable) == cudaSuccess;
    }
    if (!ok) {
        cudaGetLastError();
        hs_free();
        basalt::log::warn("basalt verify", "the head stays whole on CUDA0: no room for the peer's share");
        return;
    }
    hs_rows_ = rows;
    hs_dev_ = dm_->pdev;
    basalt::log::info("basalt verify", "the head is split by vocabulary: CUDA0 %d rows, the peer %d rows", rows, r1);
}

void Verifier::hs_free() {
    if (hs_dev_ >= 0 || hs_w_ != nullptr || hs_s_ != nullptr) {
        const OnDevice on(dm_ != nullptr && dm_->pdev >= 0 ? dm_->pdev : hs_dev_);
        if (hs_s_) { cudaStreamSynchronize(hs_s_); cudaStreamDestroy(hs_s_); }
        if (hs_out_) cudaEventDestroy(hs_out_);
        for (void* p : {hs_w_, hs_xq_, (void*) hs_y_}) if (p) cudaFree(p);
    }
    {
        const OnDevice on(device_);
        if (hs_y0_) cudaFree(hs_y0_);
        if (hs_in_) cudaEventDestroy(hs_in_);
        if (hs_hxq_) cudaFreeHost(hs_hxq_);
        if (hs_hy_) cudaFreeHost(hs_hy_);
    }
    hs_rows_ = 0; hs_dev_ = -1;
    hs_w_ = hs_xq_ = hs_hxq_ = nullptr;
    hs_y_ = hs_y0_ = hs_hy_ = nullptr;
    hs_s_ = nullptr;
    hs_in_ = hs_out_ = nullptr;
}

// The GDN mixer splits by key head when the peer card streams the mixer's bytes faster than CUDA0.  The unit is the
// fewest key heads whose value-head runs fill whole out-projection blocks; the peer's share of the units follows both
// cards' memory bandwidth, rounded down, and is taken only from a quarter up, CUDA0 keeping at least one unit.  The
// peer holds its heads' weights and state in gdn_head_split.hpp's layout.
// The key heads CUDA0 keeps when the GDN mixer splits by key head between CUDA0 (dev0) and the peer (dev1); 0 when the
// split is not taken.  Units of key heads whose out-projection columns fill whole 256-value blocks; the peer's units
// follow the two cards' memory bandwidth, taken only from a quarter of them.
static bool g_gs_off = false;   // set by Verifier::gs_decide when the split does not pay

static int gs_plan_bw(const ModelGeometry& g, int dev0, int dev1) {
    const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, S = (int) g.ssm_state_size, HIST = (int) g.ssm_d_conv - 1;
    const int64_t N = g.n_embd, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    constexpr int kBlockCols = 256;
    int unit = 1;
    while (unit <= HK && (unit * S) % kBlockCols != 0) ++unit;
    if (HK <= 0 || HV % HK != 0 || unit > HK || HK % unit != 0 || N % kBlockCols != 0 || HIST < 1 ||
        C != (int64_t) (2 * HK + HV) * S || ZV != (int64_t) HV * S)
        return 0;
    const double bw0 = card_bandwidth(dev0), bw1 = card_bandwidth(dev1);
    if (bw0 <= 0.0 || bw1 <= 0.0) return 0;
    const int units = HK / unit;
    const int u1 = std::min(units - 1, (int) std::floor(units * bw1 / (bw0 + bw1)));
    if (u1 * 4 < units) return 0;
    return HK - u1 * unit;
}

static int gs_plan(const ModelGeometry& g, int dev0, int dev1) { return g_gs_off ? 0 : gs_plan_bw(g, dev0, dev1); }

// The split's start-time price, in ms per verify window.  It saves the main card's stream of the peer's heads' mixer
// weights at its achieved bandwidth, less the peer item's growth per window token beyond the first.  Its reserve
// displaces the same bytes of experts from the peer to RAM, which decode pays as CPU-tier misses.  The prompt path is
// not priced: it streams the displaced experts with the rest of a chunk's, at no measurable cost.
bool Verifier::gs_decide(const WeightTable& wt, const ModelGeometry& g, int dev0, int dev1, int max_t,
                         int64_t displaced) {
    g_gs_off = false;
    if (dev1 < 0 || dev1 == dev0) return false;
    const int nk0 = gs_plan_bw(g, dev0, dev1);
    if (nk0 == 0) return false;
    constexpr double kMainEff = 0.71;        // the main card's achieved share of its bandwidth on the mixer
    constexpr double kTokGrowth = 0.71;      // the peer item's growth per token, in units of its weight stream
    constexpr double kMissMsPerGB = 0.263;   // decode CPU-tier misses per window per GB of displaced experts
    const int64_t HK = g.ssm_k_heads, N = g.n_embd, nk1 = HK - nk0;
    const int64_t C1 = g.ssm_conv_channels / HK * nk1, Z1 = g.ssm_value_dim / HK * nk1;
    double bm = 0.0;
    int layers = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (is_qsa_layer(g, l)) continue;
        const LayerView v(wt, l);
        const WeightRef *q = v.get("attn_qkv.weight"), *z = v.get("attn_gate.weight"), *o = v.get("ssm_out.weight");
        if (!q || !z || !o) return false;
        bm += (double) basalt::kernels::native_mmvq_weight_bytes(q->native_type, (int) N, (int) C1) +
              (double) basalt::kernels::native_mmvq_weight_bytes(z->native_type, (int) N, (int) Z1) +
              (double) basalt::kernels::native_mmvq_weight_bytes(o->native_type, (int) Z1, (int) N);
        ++layers;
    }
    if (layers == 0) return false;
    bm /= layers;
    const double bw0 = card_bandwidth(dev0), bw1 = card_bandwidth(dev1);
    const double t_exp = std::max(1.0, (max_t + 2) / 3.0);   // the expected window: a third of the drafted run
    const double save = layers * (bm / (kMainEff * bw0) - kTokGrowth * (t_exp - 1.0) * bm / bw1) * 1e3;
    const double dgb = (double) std::max<int64_t>(displaced, 0) / 1e9;
    const double miss = kMissMsPerGB * dgb;
    const bool take = save - miss > 0.0;
    g_gs_off = !take;
    basalt::log::info("basalt verify", "GDN mixer split by key head: saves %.2f ms per %.1f-token window, costs %.2f "
                         "(%.2f GB of experts off the peer, read as decode misses) -> %s",
                 save, t_exp, miss, dgb, take ? "split" : "whole on CUDA0");
    return take;
}

int64_t Verifier::gs_cuda0_bytes(const WeightTable& wt, const ModelGeometry& g, int dev0, int dev1) {
    if (dev1 < 0 || dev1 == dev0) return 0;
    const int nk0 = gs_plan(g, dev0, dev1);
    if (nk0 == 0) return 0;
    const int64_t HK = g.ssm_k_heads, N = g.n_embd, C0 = g.ssm_conv_channels / HK * nk0, Z0 = g.ssm_value_dim / HK * nk0;
    int64_t b = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (is_qsa_layer(g, l)) continue;
        const LayerView v(wt, l);
        const WeightRef *q = v.get("attn_qkv.weight"), *z = v.get("attn_gate.weight"), *o = v.get("ssm_out.weight");
        if (!q || !z || !o) return 0;
        b += (int64_t) basalt::kernels::native_mmvq_weight_bytes(q->native_type, (int) N, (int) C0) +
             (int64_t) basalt::kernels::native_mmvq_weight_bytes(z->native_type, (int) N, (int) Z0) +
             (int64_t) basalt::kernels::native_mmvq_weight_bytes(o->native_type, (int) Z0, (int) N);
    }
    return b;
}

int64_t Verifier::gs_peer_bytes(const WeightTable& wt, const ModelGeometry& g, int dev0, int dev1) {
    if (dev1 < 0 || dev1 == dev0) return 0;
    const int nk0 = gs_plan(g, dev0, dev1);
    if (nk0 == 0) return 0;
    const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, S = (int) g.ssm_state_size, HIST = (int) g.ssm_d_conv - 1;
    const basalt::core::GdnHeadSlice sl{HK, HV, S, nk0, HK - nk0};
    const int64_t N = g.n_embd, C1 = sl.c(), Z1 = sl.zv(), V1 = sl.nv();
    int64_t b = 0, layers = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (is_qsa_layer(g, l)) continue;
        const LayerView v(wt, l);
        const WeightRef *q = v.get("attn_qkv.weight"), *z = v.get("attn_gate.weight"), *o = v.get("ssm_out.weight");
        if (!q || !z || !o) return 0;
        b += (int64_t) basalt::kernels::native_mmvq_weight_bytes(q->native_type, (int) N, (int) C1) +
             (int64_t) basalt::kernels::native_mmvq_weight_bytes(z->native_type, (int) N, (int) Z1) +
             (int64_t) basalt::kernels::native_mmvq_weight_bytes(o->native_type, (int) Z1, (int) N) +
             C1 * (HIST + 1) * 4 + V1 * N * 2 * 2 + V1 * 4 * 2 + (int64_t) S * 4 +
             (int64_t) basalt::core::gdn_slice_state_floats(sl, HIST) * 4;
        ++layers;
    }
    // the mixer's rows and per-allocation rounding
    return b + layers * (2 << 20);
}

void Verifier::gs_init() {
    using basalt::core::GdnHeadSlice;
    if (dm_ == nullptr || dm_->pdev < 0 || wt_ == nullptr) return;
    const ModelGeometry& g = *g_;
    const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, S = (int) g.ssm_state_size, HIST = (int) g.ssm_d_conv - 1;
    const int64_t N = g.n_embd, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    constexpr int kBlockCols = 256;
    const int nk0 = gs_plan(g, device_, dm_->pdev);
    if (nk0 == 0) return;
    const GdnHeadSlice sl{HK, HV, S, nk0, HK - nk0};
    auto blocked = [](int t) { return t == 11 || t == 12 || t == 13 || t == 14 || t == 23; };   // 256-value blocks
    // every GDN layer's tensors, as the verify window reads them
    struct Src {
        const WeightRef *qkv, *z, *out, *a, *b, *conv, *norm, *dt, *sa;
    };
    std::vector<Src> src;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (is_qsa_layer(g, l)) continue;
        const LayerView v(*wt_, l);
        Src s{v.get("attn_qkv.weight"), v.get("attn_gate.weight"), v.get("ssm_out.weight"), v.get("ssm_alpha.weight"),
              v.get("ssm_beta.weight"), v.get("ssm_conv1d.weight"), v.get("ssm_norm.weight"), v.get("ssm_dt.bias"),
              v.get("ssm_a")};
        if (!s.qkv || !s.z || !s.out || !s.a || !s.b || !s.conv || !s.norm || !s.dt || !s.sa || !s.qkv->native_data ||
            !s.z->native_data || !s.out->native_data || !blocked(s.qkv->native_type) || !blocked(s.z->native_type) ||
            !blocked(s.out->native_type) || s.conv->bytes != (uint64_t) C * (HIST + 1) * 4 ||
            s.a->bytes != (uint64_t) HV * N * 2 || s.b->bytes != (uint64_t) HV * N * 2 ||
            s.dt->bytes != (uint64_t) HV * 4 || s.sa->bytes != (uint64_t) HV * 4 || s.norm->bytes != (uint64_t) S * 4) {
            basalt::log::warn("basalt verify", "the GDN mixer stays whole on CUDA0: a layer the split does not cover");
            return;
        }
        src.push_back(s);
    }
    if (src.empty()) return;
    // each tensor comes down from CUDA0, is sliced on the host and goes up to the peer
    std::vector<uint8_t> full, part;
    size_t peer_bytes = 0;
    bool ok = true;
    auto down = [&](const void* d, size_t n) {
        full.resize(n);
        const OnDevice on(device_);
        ok = ok && cudaMemcpy(full.data(), d, n, cudaMemcpyDeviceToHost) == cudaSuccess;
    };
    size_t own_bytes = 0;
    auto up = [&](size_t n, void** dst, int dev) {
        const OnDevice on(dev);
        ok = ok && cudaMalloc(dst, n) == cudaSuccess && cudaMemcpy(*dst, part.data(), n, cudaMemcpyHostToDevice) == cudaSuccess;
        (dev == device_ ? own_bytes : peer_bytes) += n;
    };
    auto rows = [&](const void* d, size_t row_bytes, int64_t n_rows, const std::vector<basalt::core::GdnRun>& runs,
                    void** dst, int dev) {
        if (!ok) return;
        down(d, row_bytes * (size_t) n_rows);
        part.resize(row_bytes * (size_t) basalt::core::gdn_runs_len(runs));
        basalt::core::gdn_gather_rows(full.data(), row_bytes, runs, part.data());
        up(part.size(), dst, dev);
    };
    // the out projection's columns of a slice's value heads
    auto cols = [&](const void* d, int type, const GdnHeadSlice& h, void** dst, int dev) {
        if (!ok) return;
        const size_t rb = basalt::kernels::native_mmvq_weight_bytes(type, (int) ZV, 1);
        down(d, rb * (size_t) N);
        part.resize(basalt::kernels::native_mmvq_weight_bytes(type, (int) h.zv(), (int) N));
        basalt::core::gdn_gather_cols(full.data(), N, ZV, kBlockCols, rb / (size_t) (ZV / kBlockCols),
                                      basalt::core::gdn_vhead_runs(h, S), part.data());
        up(part.size(), dst, dev);
    };
    const int P = dm_->pdev, D0 = device_;
    // CUDA0 runs its own heads in place in the full state (dec_gdn_rec's head map) from packed qkv/z rows and out columns
    const GdnHeadSlice sl0{HK, HV, S, 0, nk0};
    const std::vector<basalt::core::GdnRun> qr = basalt::core::gdn_qkv_runs(sl), v1 = basalt::core::gdn_vhead_runs(sl, 1),
                                           vs = basalt::core::gdn_vhead_runs(sl, S),
                                           qr0 = basalt::core::gdn_qkv_runs(sl0), vs0 = basalt::core::gdn_vhead_runs(sl0, S);
    gs_layers_.assign(src.size(), GsLayer{});
    gs_dev_ = dm_->pdev;
    for (size_t i = 0; i < src.size() && ok; ++i) {
        const Src& s = src[i];
        GsLayer& L = gs_layers_[i];
        L.tq = s.qkv->native_type; L.tz = s.z->native_type; L.to = s.out->native_type;
        const size_t rq = basalt::kernels::native_mmvq_weight_bytes(L.tq, (int) N, 1),
                     rz = basalt::kernels::native_mmvq_weight_bytes(L.tz, (int) N, 1);
        rows(s.qkv->native_data, rq, C, qr, &L.wqkv, P);
        rows(s.z->native_data, rz, ZV, vs, &L.wz, P);
        rows(s.conv->data, (size_t) (HIST + 1) * 4, C, qr, (void**) &L.conv_w, P);
        rows(s.a->data, (size_t) N * 2, HV, v1, (void**) &L.wa, P);
        rows(s.b->data, (size_t) N * 2, HV, v1, (void**) &L.wb, P);
        rows(s.dt->data, 4, HV, v1, (void**) &L.dt, P);
        rows(s.sa->data, 4, HV, v1, (void**) &L.ssm_a, P);
        rows(s.norm->data, (size_t) S * 4, 1, {{0, 1}}, (void**) &L.gamma, P);
        cols(s.out->native_data, L.to, sl, &L.wo, P);
        rows(s.qkv->native_data, rq, C, qr0, &L.wqkv0, D0);
        rows(s.z->native_data, rz, ZV, vs0, &L.wz0, D0);
        cols(s.out->native_data, L.to, sl0, &L.wo0, D0);
    }
    const size_t state_bytes = basalt::core::gdn_slice_state_floats(sl, HIST) * 4 * src.size();
    if (ok) {
        const OnDevice on(dm_->pdev);
        ok = cudaMalloc((void**) &gs_state_, state_bytes) == cudaSuccess && cudaMemset(gs_state_, 0, state_bytes) == cudaSuccess;
        peer_bytes += state_bytes;
        // the mixer doorbell: one per GDN layer's ring
        std::vector<uint8_t> mix((size_t) dm_->rings, 0);
        for (int64_t l = lb_; l < le_; ++l) mix[(size_t) (l - lb_)] = is_qsa_layer(g, l) ? 0 : 1;
        ok = ok && cudaMalloc((void**) &gs_mix_, mix.size()) == cudaSuccess &&
             cudaMemcpy(gs_mix_, mix.data(), mix.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMalloc(&gs_pxq_, dm_->xq_stride) == cudaSuccess;
        // the peer's scratch: per layer qkv, h, gate, beta (kept for its commit); shared x, z, o, yq
        const int64_t MT = max_t_, C1 = sl.c(), Z1 = sl.zv(), V1 = sl.nv();
        const size_t per = (size_t) (MT * (2 * C1 + 2 * V1)), shared = (size_t) (MT * (N + 2 * Z1) + MT * Z1 / 32 * 9);
        const size_t nl = src.size();
        ok = ok && cudaMalloc((void**) &gs_pscr_, (per * nl + shared) * 4) == cudaSuccess;
        peer_bytes += (per * nl + shared) * 4;
        std::vector<basalt::kernels::DmMixLayer> tab((size_t) dm_->rings);
        if (ok) {
            float* sh = gs_pscr_ + per * nl;
            float *px = sh, *pz = px + MT * N, *po = pz + MT * Z1;
            void* pyq = po + MT * Z1;
            size_t i = 0;
            for (int64_t l = lb_; l < le_; ++l) {
                if (is_qsa_layer(g, l)) continue;
                const GsLayer& L = gs_layers_[i];
                basalt::kernels::DmMixLayer& m = tab[(size_t) (l - lb_)];
                m.wqkv = L.wqkv; m.wz = L.wz; m.wo = L.wo; m.tq = L.tq; m.tz = L.tz; m.to = L.to;
                basalt::kernels::DecRecArgs& r = m.rec;
                float* pl = gs_pscr_ + per * i;
                r.state = gs_state_ + basalt::core::gdn_slice_state_floats(sl, HIST) * i;
                r.qkv = pl; r.h = pl + MT * C1; r.gate = pl + 2 * MT * C1; r.beta = r.gate + MT * V1;
                r.conv_w = L.conv_w; r.x = px; r.w_alpha = L.wa; r.w_beta = L.wb; r.dt = L.dt; r.ssm_a = L.ssm_a;
                r.z = pz; r.gamma = L.gamma; r.o = po; r.yq = pyq;
                r.C = (int) C1; r.n_embd = (int) N; r.h_k = HK - nk0; r.h_v = (int) V1; r.eps = EPS;
                r.early_trigger = false;
                ++i;
            }
        }
        gs_ca_ = basalt::kernels::GdnCommitLayers{};
        gs_ca_.state = gs_state_; gs_ca_.qkv = gs_pscr_; gs_ca_.h = gs_pscr_ + MT * C1;
        gs_ca_.gate = gs_pscr_ + 2 * MT * C1; gs_ca_.beta = gs_ca_.gate + MT * V1;
        gs_ca_.state_stride = basalt::core::gdn_slice_state_floats(sl, HIST); gs_ca_.h_stride = per; gs_ca_.gb_stride = per;
        gs_ca_.C = (int) C1; gs_ca_.h_k = HK - nk0; gs_ca_.h_v = (int) V1; gs_ca_.n_layers = (int) nl;
        ok = ok && cudaMalloc((void**) &gs_mixl_, tab.size() * sizeof(tab[0])) == cudaSuccess &&
             cudaMemcpy(gs_mixl_, tab.data(), tab.size() * sizeof(tab[0]), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    ok = ok && mapped((size_t) dm_->rings * dm_->xq_stride, (void**) &gs_h_xq_, (void**) &gs_m_xq_) &&
         mapped((size_t) dm_->rings * dm_->x_stride * 4, (void**) &gs_h_x_, (void**) &gs_m_x_) &&
         mapped((size_t) max_t_ * N * 4, (void**) &gs_h_part_, (void**) &gs_m_part_);
    if (ok) {   // gs_move's staging, pinned so the moves run at the bus rate
        const size_t full_f = (size_t) S * HV * S + (size_t) C * HIST, nl = (size_t) gs_ca_.n_layers;
        ok = cudaHostAlloc((void**) &gs_h_full_, full_f * nl * 4, cudaHostAllocPortable) == cudaSuccess &&
             cudaHostAlloc((void**) &gs_h_slice_, basalt::core::gdn_slice_state_floats(sl, HIST) * nl * 4,
                           cudaHostAllocPortable) == cudaSuccess;
    }
    for (const Src& s : src)
        ok = ok && basalt::kernels::dec_moe_mix_type_ok(s.qkv->native_type) &&
             basalt::kernels::dec_moe_mix_type_ok(s.z->native_type) && basalt::kernels::dec_moe_mix_type_ok(s.out->native_type);
    if (ok) {
        const OnDevice on(device_);
        const GdnHeadSlice s0{HK, HV, S, 0, nk0};
        ok = cudaMalloc((void**) &gs_qkv0_, (size_t) max_t_ * s0.c() * 4) == cudaSuccess &&
             cudaMalloc((void**) &gs_z0_, (size_t) max_t_ * s0.zv() * 4) == cudaSuccess;
        own_bytes += (size_t) max_t_ * (s0.c() + s0.zv()) * 4;
    }
    if (!ok) {
        cudaGetLastError();
        gs_free();
        basalt::log::warn("basalt verify", "the GDN mixer stays whole on CUDA0: no room for the peer's share");
        return;
    }
    gs_nk0_ = nk0;
    basalt::log::info("basalt verify",
                      "the GDN mixer is split by key head: CUDA0 %d of %d key heads (%.0f MiB), the peer %d (%.0f MiB)",
                      nk0, HK, own_bytes / 1048576.0, HK - nk0, peer_bytes / 1048576.0);
}

void Verifier::gs_free() {
    if (gs_dev_ >= 0) {
        const OnDevice on(gs_dev_);
        for (GsLayer& L : gs_layers_)
            for (void* p : {L.wqkv, L.wz, L.wo, (void*) L.conv_w, (void*) L.dt, (void*) L.ssm_a, (void*) L.gamma,
                            (void*) L.wa, (void*) L.wb})
                if (p) cudaFree(p);
        if (gs_state_) cudaFree(gs_state_);
        if (gs_mix_) cudaFree(gs_mix_);
        if (gs_pxq_) cudaFree(gs_pxq_);
        if (gs_mixl_) cudaFree(gs_mixl_);
        if (gs_pscr_) cudaFree(gs_pscr_);
        const OnDevice on0(device_);
        for (GsLayer& L : gs_layers_)
            for (void* p : {L.wqkv0, L.wz0, L.wo0})
                if (p) cudaFree(p);
        if (gs_qkv0_) cudaFree(gs_qkv0_);
        if (gs_z0_) cudaFree(gs_z0_);
    }
    for (void* h : {(void*) gs_h_xq_, (void*) gs_h_x_, (void*) gs_h_part_, (void*) gs_h_full_, (void*) gs_h_slice_})
        if (h) cudaFreeHost(h);
    gs_h_full_ = gs_h_slice_ = nullptr;
    gs_h_x_ = gs_m_x_ = gs_h_part_ = gs_m_part_ = nullptr;
    gs_mixl_ = nullptr; gs_pscr_ = nullptr; gs_qkv0_ = gs_z0_ = nullptr;
    gs_live_ = gs_split_ = false;
    gs_ss_ = nullptr;
    gs_layers_.clear();
    gs_state_ = nullptr;
    gs_mix_ = nullptr; gs_pxq_ = nullptr; gs_h_xq_ = nullptr; gs_m_xq_ = nullptr;
    gs_nk0_ = 0;
    gs_dev_ = -1;
}

bool Verifier::gs_move(bool to_peer, std::string& err) {
    using basalt::core::GdnHeadSlice;
    const ModelGeometry& g = *g_;
    const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, S = (int) g.ssm_state_size, HIST = (int) g.ssm_d_conv - 1;
    const GdnHeadSlice sl{HK, HV, S, gs_nk0_, HK - gs_nk0_};
    const size_t full_f = (size_t) S * HV * S + (size_t) g.ssm_conv_channels * HIST;
    const size_t part_f = basalt::core::gdn_slice_state_floats(sl, HIST), nl = (size_t) gs_ca_.n_layers;
    SessionState& ss = to_peer ? *ss_ : *gs_ss_;
    const auto t0 = std::chrono::steady_clock::now();
    int64_t gdn_index = 0;
    for (int64_t l = 0; l < lb_; ++l) gdn_index += is_qsa_layer(g, l) ? 0 : 1;
    float* const full_d = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * full_f;
    float* const full = gs_h_full_;
    float* const part = gs_h_slice_;
    const size_t full_n = full_f * nl, part_n = part_f * nl;
    bool ok;
    {
        const OnDevice on(device_);
        ok = cudaStreamSynchronize(cs_) == cudaSuccess && cudaStreamSynchronize(cm_) == cudaSuccess &&
             cudaMemcpy(full, full_d, full_n * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    if (to_peer) {
        for (size_t i = 0; i < nl; ++i) basalt::core::gdn_slice_state(sl, HIST, full + full_f * i, part + part_f * i);
        const OnDevice on(gs_dev_);
        ok = ok && cudaStreamSynchronize(dm_->ps) == cudaSuccess &&
             cudaMemcpy(gs_state_, part, part_n * 4, cudaMemcpyHostToDevice) == cudaSuccess;
        gs_ss_ = ss_;
    } else {
        {
            const OnDevice on(gs_dev_);
            ok = ok && cudaStreamSynchronize(dm_->ps) == cudaSuccess &&
                 cudaMemcpy(part, gs_state_, part_n * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        for (size_t i = 0; i < nl; ++i) basalt::core::gdn_unslice_state(sl, HIST, part + part_f * i, full + full_f * i);
        const OnDevice on(device_);
        ok = ok && cudaMemcpy(full_d, full, full_n * 4, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!ok) {
        err = std::string("verify: the GDN state slice did not move: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    gs_live_ = to_peer;
    if (diagnostics().decode_timing)
        basalt::log::info("basalt verify", "the GDN state slice moved %s in %.1f ms (%.0f MiB)", to_peer ? "to the peer" : "back",
                          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                          (double) (full_n + part_n) * 4 / (1 << 20));
    return true;
}

void Verifier::dm_free() {
    DecMoe* d = dm_;
    if (d == nullptr) return;
    hs_free();
    gs_free();
    if (d->warm != nullptr) {
        warm_pool_->idle_stop();
        delete d->warm;
    }
    if (d->pdev >= 0) {
        const OnDevice on(d->pdev);
        if (d->ps) { cudaStreamSynchronize(d->ps); cudaStreamDestroy(d->ps); }
        for (void* p : {(void*) d->ptab, (void*) d->lay, d->pscr, (void*) d->parr, (void*) d->rq, (void*) d->rs})
            if (p) cudaFree(p);
    }
    for (DecMoe::XPeer& x : d->xp) {
        {
            const OnDevice on(x.dev);
            if (x.ps) { cudaStreamSynchronize(x.ps); cudaStreamDestroy(x.ps); }
            for (void* p : {(void*) x.ptab, (void*) x.lay, x.pscr, (void*) x.parr, (void*) x.rq, (void*) x.rs})
                if (p) cudaFree(p);
        }
        if (x.res) cudaFree(x.res);
        for (void* h : {(void*) x.h_ring, (void*) x.h_w})
            if (h) cudaFreeHost(h);
    }
    for (void* p : {(void*) d->res0, (void*) d->res1, (void*) d->plan, (void*) d->ctr, (void*) d->slot_off, d->vscr,
                    (void*) d->vsync, d->xq_pub, (void*) d->wstat,
                    (void*) d->ptab0, (void*) d->pub, (void*) d->cseen, (void*) d->landed})
        if (p) cudaFree(p);
    if (d->pin) cudaFreeHost(d->pin);
    if (d->landed_pin) cudaFreeHost(d->landed_pin);
    for (void* h : {(void*) d->h_ring, (void*) d->h_x, (void*) d->h_xq, (void*) d->h_w, (void*) d->h_rows})
        if (h) cudaFreeHost(h);
    delete d;
    dm_ = nullptr;
}

Verifier::~Verifier() {
    const Verifier* self = this;
    g_diag_verifier.compare_exchange_strong(self, nullptr);
    for (auto& slot : g_live) {
        Verifier* me = this;
        slot.compare_exchange_strong(me, nullptr);
    }
    if (cs_) cudaStreamSynchronize(cs_);
    if (cm_) cudaStreamSynchronize(cm_);
    if (sh_cs_) cudaStreamSynchronize(sh_cs_);
    if (sh_hi_) cudaStreamSynchronize(sh_hi_);
    if (sh_hi2_) cudaStreamSynchronize(sh_hi2_);
    if (ring_hi_) cudaStreamSynchronize(ring_hi_);
    for (cudaStream_t q : qs_)
        if (q) cudaStreamSynchronize(q);
    dm_free();
    for (auto& e : exec_)
        if (e) cudaGraphExecDestroy(e);
    for (auto& e : exec_nr_)
        if (e) cudaGraphExecDestroy(e);
    if (commit_exec_) cudaGraphExecDestroy(commit_exec_);
    for (auto& kv : solo_sets_) {
        for (auto& e : kv.second.exec)
            if (e) cudaGraphExecDestroy(e);
        for (auto& e : kv.second.exec_nr)
            if (e) cudaGraphExecDestroy(e);
        if (kv.second.commit) cudaGraphExecDestroy(kv.second.commit);
    }
    for (auto& kv : exec_bm_)
        if (kv.second) cudaGraphExecDestroy(kv.second);
    for (auto& kv : commit_bm_)
        if (kv.second) cudaGraphExecDestroy(kv.second);
    if (arena_b_) cudaFree(arena_b_);
    if (logit_bias_device_) cudaFree(logit_bias_device_);
    if (row_bias_device_) cudaFree(row_bias_device_);
    if (h_commitb_) cudaFreeHost(h_commitb_);
    if (h_segb_) cudaFreeHost(h_segb_);
    if (h_qseg_) cudaFreeHost(h_qseg_);
    if (cs_) cudaStreamDestroy(cs_);
    if (cm_) cudaStreamDestroy(cm_);
    if (ev_commit_go_) cudaEventDestroy(ev_commit_go_);
    if (commit_ctl_) cudaFree(commit_ctl_);
    if (lz_s_) cudaStreamDestroy(lz_s_);
    if (lz_fork_) cudaEventDestroy(lz_fork_);
    if (lz_join_) cudaEventDestroy(lz_join_);
    if (sh_cs_) cudaStreamDestroy(sh_cs_);
    if (sh_hi_) cudaStreamDestroy(sh_hi_);
    if (sh_hi2_) cudaStreamDestroy(sh_hi2_);
    if (ev_sh2_) cudaEventDestroy(ev_sh2_);
    if (ring_hi_) cudaStreamDestroy(ring_hi_);
    if (ple_s_) cudaStreamDestroy(ple_s_);
    if (ev_ple_fork_) cudaEventDestroy(ev_ple_fork_);
    if (ev_ple_join_) cudaEventDestroy(ev_ple_join_);
    for (cudaStream_t q : qs_)
        if (q) cudaStreamDestroy(q);
    for (cudaEvent_t e : qev_)
        if (e) cudaEventDestroy(e);
    if (ev_ring_) cudaEventDestroy(ev_ring_);
    if (ev_ring_join_) cudaEventDestroy(ev_ring_join_);
    if (commit_done_) cudaEventDestroy(commit_done_);
    if (ev_fork_) cudaEventDestroy(ev_fork_);
    if (ev_join_) cudaEventDestroy(ev_join_);
    if (arena_) cudaFree(arena_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_, h_pend_,
                     h_flagA_, h_plan_, h_flagB_, h_q_, h_qp_};
    for (void* h : hosts)
        if (h) cudaFreeHost(h);
}

bool hc8_table_has(const std::string& name, int rows, int cols) { return basalt::core::hc8_find(name, rows, cols) != nullptr; }

const void* hc8_table_adopt(const std::string& name, const uint16_t* w, int rows, int cols) {
    const Hc8Copy* c = hc8_find(name, rows, cols);
    if (c == nullptr) return nullptr;
    return basalt::kernels::hc8_upload(w, c->q(), c->s(), rows, cols);
}

namespace {
/// The hyper-connection matrices, by name: (name, rows, cols).
struct Hc8Tensor { std::string name; int rows, cols; };
std::vector<Hc8Tensor> hc8_tensors(const ModelGeometry& g) {
    const int LR = (int) g.hc_lr, D = (int) (g.hc * g.n_embd), HCn = (int) g.hc;
    std::vector<Hc8Tensor> v;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const std::string blk = "blk." + std::to_string(l) + ".";
        for (const char* p : {"hc_attn_", "hc_ffn_"}) {
            const std::string s(p);
            v.push_back({blk + s + "down.weight", LR, D});
            v.push_back({blk + s + "up.weight", D, LR});
            v.push_back({blk + s + "inject.weight", HCn, D});
        }
    }
    v.push_back({"output_hc_down.weight", LR, D});
    v.push_back({"output_hc_up.weight", D, LR});
    return v;
}
// the copies' size: the int8 codes and one fp16 scale per 32 values
uint64_t hc8_copy_bytes(const Hc8Tensor& t) { return (uint64_t) t.rows * (uint64_t) t.cols * 9 / 8; }
}  // namespace

uint64_t Verifier::hc8_plan_bytes(const std::function<bool(const std::string&)>& has, const ModelGeometry& g) {
    if (!basalt::kernels::hc8_enabled()) return 0;
    uint64_t total = 0;
    for (const Hc8Tensor& t : hc8_tensors(g))
        if (has(t.name)) total += hc8_copy_bytes(t);
    return total;
}

void Verifier::prepare_hc8(const WeightTable& wt, const ModelGeometry& g) {
    if (!basalt::kernels::hc8_enabled()) return;
    size_t from_table = 0;
    const std::vector<Hc8Tensor> tensors = hc8_tensors(g);
    const uint64_t total = hc8_plan_bytes([&](const std::string& n) { return wt.find(n) != nullptr; }, g);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    LoadProgress progress(LoadProgress::cuda(dev), total);
    for (const Hc8Tensor& t : tensors) {
        const WeightRef* r = wt.find(t.name);
        if (r == nullptr) continue;
        const uint16_t* w = (const uint16_t*) r->data;
        if (w != nullptr && hc8_table_adopt(t.name, w, t.rows, t.cols) != nullptr) {
            ++from_table;
        } else {
            basalt::kernels::hc8_prepare(w, t.rows, t.cols);
        }
        progress.add(hc8_copy_bytes(t));
    }
    progress.finish();
    (void) from_table;
    basalt::log::info("basalt hc8", "8-bit hyper-connection copies, %.0f MiB of VRAM",
                 (double) basalt::kernels::hc8_bytes() / 1048576.0);
}

bool hc8_dump(const WeightTable& wt, const ModelGeometry& g, const std::vector<std::pair<std::string, const void*>>& extra,
              const std::string& path, std::string& err) {
    std::vector<Hc8Copy> copies;
    auto add = [&](const std::string& name, const void* key) {
        basalt::kernels::Hc8View v;
        if (key == nullptr || !basalt::kernels::hc8_view(key, v)) return;
        Hc8Copy c;
        c.name = name;
        c.rows = (uint32_t) v.rows;
        c.cols = (uint32_t) v.cols;
        c.srow = (uint32_t) v.srow;
        c.data.resize(v.bytes);
        if (cudaMemcpy(c.data.data(), v.q, v.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaGetLastError();
            c.data.clear();
            return;
        }
        copies.push_back(std::move(c));
    };
    for (const Hc8Tensor& t : hc8_tensors(g))
        if (const WeightRef* r = wt.find(t.name)) add(t.name, r->data);
    for (const auto& [name, key] : extra) add(name, key);
    if (copies.empty()) { err = "hc8: there are no 8-bit copies to write"; return false; }
    if (!hc8_write(path, copies, err)) return false;
    basalt::log::info("basalt hc8", "wrote %zu copies to %s", copies.size(), path.c_str());
    return true;
}

bool Verifier::set_logit_bias(const std::vector<float>& bias, std::string& err) {
    if (next_ && !next_->set_logit_bias(bias, err)) return false;
    if (next_) return true;   // an earlier stage: the head, and so the sampling, is on the last one
    const OnDevice on_device(device_);
    if (!bias.empty() && bias.size() != (size_t) n_vocab_) {
        err = "logit_bias: the vocabulary size does not match";
        return false;
    }
    if (bias != logit_bias_host_ && !bias.empty()) {
        const size_t bytes = bias.size() * sizeof(float);
        if ((!logit_bias_device_ && cudaMalloc((void**) &logit_bias_device_, bytes) != cudaSuccess) ||
            cudaMemcpy(logit_bias_device_, bias.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "logit_bias: the upload failed";
            return false;
        }
    }
    logit_bias_host_ = bias;
    sampling_.logit_bias = bias.empty() ? nullptr : logit_bias_device_;
    return true;
}

bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
                    const NativeHead* head, int max_t, std::string& err) {
    g_diag_verifier.store(this);
    diag_verify_fn().store(&diag_active_verifier);
    for (auto& slot : g_live) {
        Verifier* none = nullptr;
        if (slot.load() == this || slot.compare_exchange_strong(none, this)) break;
    }
    release_gpu_fn().store(&release_live_verifiers);
    cudaGetDevice(&device_);   // a layer split's stage on another GPU: its streams, graphs and buffers live there
    basalt::kernels::fused_gr_check();   // once per card: which bitwise-equal hyper-connection read runs there
    basalt::kernels::hc_fused_init();
    basalt::kernels::mid_chain_init();
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    sampling_.greedy = true;      // a fresh verifier samples greedily until set_sampling says otherwise
    sampling_.temperature = 0.0f;
    if (max_t < 2 || max_t > basalt::kernels::kVerifyMaxT || max_t > basalt::kernels::cpu::MAXT) {
        err = "verify: the window must hold 2.." + std::to_string(basalt::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    if (hits.d_res == nullptr || hits.cache_base == nullptr || hits.blob <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (the model file's profile and --expert-cache); with "
              "--expert-cache auto, no VRAM was left for it - the 'no VRAM is left for the expert cache' line above "
              "says how much is short and what makes room (a smaller --max-context, --kv q4_0, a smaller draft vocabulary, "
              "images on the CPU)";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!basalt::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    if (le_ < 0) le_ = g.n_layers;
    if (lb_ < 0 || lb_ >= le_ || le_ > g.n_layers || (lb_ > 0 && hand_in_ == nullptr) ||
        (le_ < g.n_layers && hand_out_ == nullptr)) {
        err = "verify: the stage's layer range or its hand-off buffers are wrong";
        return false;
    }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;

    const basalt::kernels::QsaShapes s = shapes_of(g);
    cap_ = basalt::kernels::qsa_selection_width(basalt::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) basalt::kernels::qsa_decode_attn_scratch_floats(cap_, s);

    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) g.n_gdn_layers(), nQ = (uint64_t) g.n_qsa_layers();
    const uint64_t HS = (uint64_t) basalt::kernels::NG_HIST * basalt::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

    // ---- mapped staging
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * basalt::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * (NH + NKV + IQ) * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(64, (void**) &h_pend_, (void**) &m_pend_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 16, (void**) &h_q_, (void**) &m_q_) &&
              mapped(T * 4 + 16, (void**) &h_qp_, (void**) &m_qp_) &&
              mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              mapped(64, (void**) &h_flag_, (void**) &m_flag_) &&
              mapped(64, (void**) &h_flagA_, (void**) &m_flagA_) &&
              mapped(64, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(64, (void**) &h_plan_err_, (void**) &m_plan_err_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    // the GPU plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
    {
        const int64_t cap = (int64_t) (T * K);
        const int64_t i32 = 4 + (cap + 1) + cap + cap;
        const int64_t ptr_off = (i32 + 1) & ~1ll;
        plan_i32_ = ptr_off + 4 * cap + (cap + 1) + 1;
        if (!mapped((size_t) plan_i32_ * 4 + 64, (void**) &h_plan_, (void**) &m_plan_)) {
            err = "verify: mapped plan allocation failed";
            return false;
        }
        sink_.counts = h_plan_;
        sink_.start = h_plan_ + 4;
        sink_.dst = sink_.start + cap + 1;
        sink_.tok = sink_.dst + cap;
        sink_.ptr = (unsigned long long*) (h_plan_ + ptr_off);
        sink_.ptr2 = sink_.ptr + cap;
        sink_.start2 = h_plan_ + ptr_off + 4 * cap;
        sink_.cap = cap;
        sink_.publish = &Verifier::publish_plan;
        sink_.staged = &Verifier::raise_flag_b;
        sink_.ctx = this;
    }

    // ---- the device arena: the same sequence counted, then carved
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(T * basalt::kernels::kStepCount);
        pos_ = b.take<int32_t>(T * (NH + NKV + IQ)); commit_ = b.take<int32_t>(2 + T);
        ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); bo_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(basalt::kernels::native_q8_1_bytes(max_in, (int) T));
        sh_xq_ = b.take<uint8_t>(basalt::kernels::native_q8_1_bytes(max_in, (int) T));
        qkv_L_ = b.take<float>(nG * T * C); h_L_ = b.take<float>(nG * T * C);
        gate_L_ = b.take<float>(nG * T * HV); beta_L_ = b.take<float>(nG * T * HV);
        z_ = b.take<float>(T * ZV); y_ = b.take<float>(T * ZV); y_dummy_ = b.take<float>(T * ZV);
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        idx_raw_L_ = b.take<float>(nQ * T * ID); qidx_ = b.take<float>(T * IQ * ID);
        scores_ = b.take<float>(T * (uint64_t) max_blocks_); sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        tail_snap_ = b.take<float>(nQ * TS);
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); hit_out_ = b.take<float>(T * K * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        plan_ = b.take<int32_t>((uint64_t) plan_i32_ + 16);
        staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * basalt::kernels::cpu::expert_layout().max_blob);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            basalt::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            basalt::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        head_mixed_ = b.take<float>(T * N); head_inj_ = b.take<float>(HC);
        sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); sh_g_ = b.take<float>(T + 4);
        head_logits_ = b.take<float>(T * (uint64_t) n_vocab_);
        if (head_ != nullptr && head_->vproxy() != nullptr)
            vhead_scratch_ = b.take<uint8_t>(basalt::kernels::q6k_vhead_scratch_bytes());
        hist_snap_ = b.take<float>(T * HS);
        ple_win_ = b.take<float>(2 * T * HC * N);
        ple_key_ = b.take<float>(T * HC * N); ple_val_ = b.take<float>(T * N);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) {
        err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    cudaMemset(arena_, 0, count.used);
    prof_on_ = diagnostics().profile;
    if (prof_on_) decode_gaps().host_on = true;
    if (next_ == nullptr && exp_flag("host_ev", 0) != 0) decode_events().enable(4000);
    if (prof_on_) {
        const size_t np = (size_t) g.n_layers * kProfPer + 8;
        if (cudaMalloc((void**) &prof_, np * 8) != cudaSuccess) { prof_on_ = false; prof_ = nullptr; cudaGetLastError(); }
        else { cudaMemset(prof_, 0, np * 8); prof_h_.assign(np, 0); }
    }
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    sink_.staging_cap = kStagingBlobs;
    (void) TS;
    int cs_least = 0, cs_greatest = 0;
    cudaDeviceGetStreamPriorityRange(&cs_least, &cs_greatest);
    const int cs_prio = main_high_ ? cs_greatest : 0;
    if (cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, cs_prio) != cudaSuccess ||
        cudaStreamCreateWithFlags(&sh_cs_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "verify: stream create failed";
        return false;
    }
    {
        int least = 0, greatest = 0;
        if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess ||
            cudaStreamCreateWithPriority(&sh_hi_, cudaStreamNonBlocking, greatest) != cudaSuccess ||
            cudaStreamCreateWithPriority(&sh_hi2_, cudaStreamNonBlocking, greatest) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_sh2_, cudaEventDisableTiming) != cudaSuccess ||
            cudaStreamCreateWithPriority(&ring_hi_, cudaStreamNonBlocking, greatest) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_ring_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_ring_join_, cudaEventDisableTiming) != cudaSuccess ||
            cudaStreamCreateWithFlags(&ple_s_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_ple_fork_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_ple_join_, cudaEventDisableTiming) != cudaSuccess) {
            err = "verify: side stream create failed";
            return false;
        }
        basalt::log::info("basalt verify", "side branches at stream priority %d", greatest);
    }
    if (cudaStreamCreateWithPriority(&cm_, cudaStreamNonBlocking, cs_prio) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_commit_go_, cudaEventDisableTiming) != cudaSuccess ||
        cudaMalloc((void**) &commit_ctl_, sizeof(basalt::kernels::CommitCtl)) != cudaSuccess ||
        cudaMemset(commit_ctl_, 0, sizeof(basalt::kernels::CommitCtl)) != cudaSuccess) {
        err = "verify: commit stream create failed";
        return false;
    }
    {
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_ >= 0 ? device_ : 0);
        fin_blocks_ = sms;
    }
    if (!basalt::kernels::gdn_commit_lazy_none(commit_ctl_) ||
        cudaStreamCreateWithPriority(&lz_s_, cudaStreamNonBlocking, cs_prio) != cudaSuccess ||
        cudaEventCreateWithFlags(&lz_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&lz_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "verify: lazy commit stream create failed";
        return false;
    }
    *(volatile uint32_t*) h_pend_ = 0;
    if (cudaEventCreateWithFlags(&commit_done_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "verify: event create failed";
        return false;
    }
    {
        int least = 0, greatest = 0;
        bool qok = cudaDeviceGetStreamPriorityRange(&least, &greatest) == cudaSuccess;
        for (auto& q : qs_) qok = qok && cudaStreamCreateWithPriority(&q, cudaStreamNonBlocking, greatest) == cudaSuccess;
        for (auto& e : qev_) qok = qok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
        if (!qok) { err = "verify: QSA branch stream create failed"; return false; }
    }
    // Check whether 100% of experts across [lb_, le_) are resident in this stage's VRAM cache.
    // When true, every layer plans on device and writes directly into parts_ without any CPU doorbells,
    // wait_flag_ge spins, PCIe empty launches, or moe_hit_add copies.
    all_resident_ = false;
    ar_off_ = false;
    if (h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
    if (hits.h_res != nullptr && hits.d_res != nullptr && hits.cache_base != nullptr) {
        {
            bool all_ok = true;
            for (int64_t l = lb_; l < le_ && all_ok; ++l) {
                for (int64_t e = 0; e < g.n_expert; ++e) {
                    if (hits.h_res[l * g.n_expert + e] < 0) {
                        all_ok = false;
                        break;
                    }
                }
            }
            all_resident_ = all_ok;
        }
    }
    device_plan_ = false;
    if (all_resident_ || device_plan_) {
        bool ok2 = true;
        if (device_plan_)
            ok2 = cudaMalloc((void**) &skip_, 64) == cudaSuccess && cudaMemset(skip_, 0, 64) == cudaSuccess;
        if (ok2 && hits.slot_off != nullptr && hits.n_slots > 0) {
            ok2 = cudaMalloc((void**) &slot_off_d_, (size_t) hits.n_slots * sizeof(unsigned long long)) == cudaSuccess &&
                  cudaMemcpy(slot_off_d_, hits.slot_off, (size_t) hits.n_slots * sizeof(unsigned long long),
                             cudaMemcpyHostToDevice) == cudaSuccess;
        }
        if (!ok2) { cudaGetLastError(); all_resident_ = false; device_plan_ = false; }
    }
    if (!dm_init(err)) return false;
    basalt::log::info("basalt verify", "window up to %d tokens, %.1f MiB of device buffers%s", max_t,
                 (double) count.used / 1048576.0,
                 all_resident_ ? " (100% VRAM resident: zero-doorbell graph)" : "");
    return true;
}

const float* Verifier::final_R(int t) const { return R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd); }

// ================================ THE WINDOW, AS CAPTURED ================================
//
// The stream is ordered pre(0) | post(0) pre(1) | post(1) pre(2) | ...: each layer's mixer and router, then its
// experts and combine, with the next layer's mixer launched behind them.
bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready() && ple_stage();
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const bool dm = dm_ != nullptr && (!batch_rec_ || batch_dm_);   // the device-planned tail (one group)
    // each hyper-connection read (and the FFN read's router) as one kernel; the next read's w_norm is prefetched into
    // L2 at its end
    const bool dec_hc = hc_fused_supported();
    // the dense chain's kernels (pdl.hpp) may start during the previous kernel's tail.  Only where that previous kernel
    // is one of them or another plain kernel; never with the profile's stamps.
    const bool pdl = pdl_rec_ && !prof_on_;
    // the chain's kernels let their dependents in early (pdl.hpp), which stage their weights
    const bool pdl2 = pdl && pdl2_requested();
    auto stamp = [&](int64_t l, int i) {
        if (prof_on_) gpu_stamp(prof_, (int) (l * kProfPer + i), cs);
    };
    const int hrow0 = batch_rec_ ? row_base_ : 0;   // a slot group's own hand-off rows

    if (prof_on_ && !batch_rec_) gpu_stamp(prof_, (int) (g.n_layers * kProfPer + 2), cs);
    // layer 1's PLE projections read only the host-gathered rows, so they run beside layer 0 from the window's start;
    // layer 1 joins them before the PLE window
    const bool ple_side = ple_on && !batch_rec_ && dm && lb_ <= 1 && le_ > 1 && !prof_on_ &&
                          ple_s_ != nullptr && ev_ple_fork_ != nullptr && ev_ple_join_ != nullptr &&
                          ss.ple.w.key_bf16 != nullptr && ss.ple.w.value_bf16 != nullptr;
    if (!batch_rec_) ple_side_ = ple_side;
    if (ple_side) {
        try {
            cudaEventRecord(ev_ple_fork_, cs);
            cudaStreamWaitEvent(ple_s_, ev_ple_fork_, 0);
            wait_flag_ge(m_flag_ + 8, 1, ple_s_);
            copy_from_mapped(ple_, m_ple_, (int64_t) T * N, ple_s_);
            bf16_gemv_fp32_mmvf_multi(ple_, N, ss.ple.w.key_bf16, ple_key_, HC * N, N, HC * N, T, ple_s_);
            bf16_gemv_fp32_mmvf_multi(ple_, N, ss.ple.w.value_bf16, ple_val_, N, N, N, T, ple_s_);
            cudaEventRecord(ev_ple_join_, ple_s_);
        } catch (const std::exception& e) {
            err = std::string("verify PLE side: ") + e.what();
            return false;
        }
    }
    // ---- the async commit's remainder (a no-op when none is pending): nothing below reads or writes what it works on until
    // this node returns - the GDN state, conv history, indexer buffers and PLE history, nor the rows it reads (h_L_,
    // qkv_L_, gate_L_, beta_L_, idx_raw_L_, tail_snap_, hist_snap_), which the first kernels' launches order after it
    if (finisher_on() && !batch_rec_ && cs == cs_) {
        const GdnCommitLayers fa = commit_gdn_args();
        if (fa.n_layers > 0) gdn_commit_finish(fa, commit_ctl_, cs, fin_blocks_);
    }
    // ---- the window's inputs, from mapped staging
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    if (batch_rec_) {   // the segment tables: which slot each row is
        copy_i32_from_mapped((int32_t*) segb_, (const int32_t*) m_segb_, (int64_t) T * 8, cs);
        copy_i32_from_mapped((int32_t*) d_qrows_, (const int32_t*) m_qseg_, (int64_t) (kQsaSegMaxRows + 8) * 4, cs);
        if (ple_on) ple_hist_gather(ple_work_, d_ple_slots_, segb_, HS, T, cs);
    }
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), cs);
    // per-ROW positions of the K rows [t][NKV] and the indexer query rows [t][IQ] (for batched RoPE)
    const int32_t* pos_k = pos_ + MT * NH;
    const int32_t* pos_i = pos_ + MT * (NH + NKV);

    // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
    // residual, pending write and inject (see set_stage)
    const int64_t HB = Verifier::handoff_floats(g);
    if (lb_ > 0) {
        const float* hin = hand_in_ + (size_t) hrow0 * HB;   // the group's rows: [R][bo][inj] contiguous
        copy_from_mapped(R_, hin, (int64_t) T * HC * N, cs);
        copy_from_mapped(bo_, hin + (size_t) T * HC * N, (int64_t) T * N, cs);
        copy_from_mapped(inj2_, hin + (size_t) T * (HC + 1) * N, (int64_t) T * HC, cs);
    } else if (const NativeEmbed* ne = native_embed()) {   // the GGUF-form table
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    }

    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }
    // a lazy GDN layer: one after the stage's first, in a solo window (lazy_on)
    int64_t gdn_first = -1;
    for (int64_t l = lb_; l < le_ && gdn_first < 0; ++l) gdn_first = gdn_idx[(size_t) l];
    const bool lazy = lazy_on() && !batch_rec_ && cs == cs_ && dm;
    auto lazy_layer = [&](int64_t l) { return lazy && l < le_ && gdn_idx[(size_t) l] > gdn_first; };
    const GdnCommitLayers lazy_args = lazy ? commit_gdn_lazy_args() : GdnCommitLayers{};
    bool lz_open = false;   // a lazy launch not joined yet

    // ---------------------------------------------------------------- pre(l, group): up to the ring
    bool ring_fork = false;   // the group's ring stream is joined in post
    int mid_ord = 0;   // the mid chain's launches so far in this graph
    bool glue_on = false;   // post's gate/up launch plans the group's VRAM share
    bool sh_read = false;   // the FFN read gave the shared expert its gate and q8_1 rows; post's combine gates it
    DmRouteArgs glue_ring;   // the ring's work for post's gate/up launch
    auto pre = [&](int64_t l) -> bool {
        const int n = T;
        stamp(l, 0);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        // the previous layer's FFN write, folded into this layer's first read (a control vector after it has
        // already applied it)
        bool pending = l > 0 && !cvec().covers(l - 1);
        if (l == 1 && ple_on) {
            if (!ple_side) {
                if (ar_on() || dm) wait_flag_ge(m_flag_, 1, cs);
                copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);
            }
            float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
            const bool ple_batch_kv = n > 1 && ss.ple.w.key_bf16 != nullptr &&
                                      ss.ple.w.value_bf16 != nullptr;
            // the rows' write, post-ops, history advance and snapshots in two launches, the value
            // projection on the side stream beside the key's
            // (one row too: there the multi-row GEMVs are ple_block's own)
            const bool ple_win = ple_batch_kv || (n == 1 && ss.ple.w.key_bf16 != nullptr && ss.ple.w.value_bf16 != nullptr);
            if (batch_rec_) batch_seg_ple_ = ple_win;
            cudaStream_t pside = sh_hi_ != nullptr ? sh_hi_ : sh_cs_;
            const bool ple_fork = ple_win && pside != nullptr && ev_fork_ != nullptr && ev_join_ != nullptr;
            if (ple_side) {
                cudaStreamWaitEvent(cs, ev_ple_join_, 0);
            } else if (ple_batch_kv || ple_win) {
                try {
                    if (ple_fork) {
                        cudaEventRecord(ev_fork_, cs);
                        cudaStreamWaitEvent(pside, ev_fork_, 0);
                        bf16_gemv_fp32_mmvf_multi(ple_, N, ss.ple.w.value_bf16,
                                                  z_, N, N, N, n, pside);
                        cudaEventRecord(ev_join_, pside);
                    }
                    bf16_gemv_fp32_mmvf_multi(ple_, N, ss.ple.w.key_bf16,
                                              xn_, HC * N, N, HC * N, n, cs);
                    if (ple_fork) cudaStreamWaitEvent(cs, ev_join_, 0);
                    else bf16_gemv_fp32_mmvf_multi(ple_, N, ss.ple.w.value_bf16,
                                                   z_, N, N, N, n, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE multi: ") + e.what();
                    return false;
                }
            }
            if (ple_win) {
                PleWindowArgs pa;
                pa.R = Rt(0); pa.block_out = bo_; pa.inject = inj2_;
                pa.key = ple_side ? ple_key_ : xn_;
                pa.value = ple_side ? ple_val_ : z_;
                pa.history = batch_rec_ ? ple_work_ : ss.ple.hist; pa.segs = batch_rec_ ? segb_ : nullptr;
                pa.hist_snap = hist_snap_; pa.scratch = ple_win_; pa.T = n;
                try {
                    native_ple_window(pa, ss.ple.w, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE window: ") + e.what();
                    return false;
                }
            }
            for (int t = 0; t < T && !ple_win; ++t) {
                gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(t);
                float* hist = batch_rec_ ? ple_work_ + (size_t) t * HS : ss.ple.hist;   // the row's own history
                try {
                    if (ple_batch_kv) {
                        ple_block_projected(xn_ + (size_t) t * HC * N, z_ + (size_t) t * N, Rt(t), hist,
                                            ss.ple.w, po, ss.ple.scratch, cs);
                    } else {
                        ple_block(ple_ + t * N, Rt(t), hist, ss.ple.w, po, ss.ple.scratch, cs);
                    }
                    ple_history_advance(hist, normalized, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE: ") + e.what();
                    return false;
                }
                copy_from_mapped(hist_snap_ + (size_t) t * HS, hist, HS, cs);
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                FusedGrArgs& a = fa[t];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, n, xn_, cs, prof_on_ ? prof_ : nullptr,
                                (int) (l * kProfPer + (half == 0 ? 27 : 30)));
        };
        auto hc_args = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            HcFusedArgs h;
            h.T = n; h.R = Rt(0); h.R_out = Rt(0); h.apply = apply;
            h.bo_prev = bo_; h.inj_prev = inj_prev;
            h.w_norm = (const float*) wn[half]->data; h.w_down = (const uint16_t*) wd[half]->data;
            h.w_up = (const uint16_t*) wu[half]->data; h.w_inject = (const uint16_t*) wi[half]->data;
            h.eps = EPS; h.lo = lo_; h.rs = rs_; h.inject_out = inj_out;
            h.mixed = mixed_; h.xn = xn_;
            return h;
        };
        const bool read_q8 = dec_hc && hc_fused_q8_rows_ok();
        const WeightRef* w_router = v.get("ffn_gate_inp.weight");
        const bool router_multi = w_router != nullptr && NE == 512 && K == 10;
        // a GDN layer's middle as one persistent kernel (mid_chain.hpp) between the layer's two hc reads: the qkv + z pair,
        // the recurrence and the output projection, bitwise the kernels it replaces
        // the GDN mixer split by key head: the layer's q8_1 rows and a doorbell for the peer's key heads; in a batch
        // window CUDA0 runs the whole layer and the doorbell only passes
        const bool gs_ring = dm && gs_nk0_ > 0 && !is_qsa_layer(g, l);
        const bool gs_split = gs_ring && !batch_rec_;
        bool mid_ok = false;
        MidChainArgs ma;
        if (dec_hc && read_q8 && router_multi && !is_qsa_layer(g, l) && !prof_on_ && !batch_rec_ && !gs_split &&
            n <= kMidChainMaxT && N == 2560 && dec_gdnf_mode() != 0 && (dense_tc_mask() & 4)) {
            const WeightRef *wqkv = v.get("attn_qkv.weight"), *wg = v.get("attn_gate.weight"),
                            *wout = v.get("ssm_out.weight"), *wa = v.get("ssm_alpha.weight"),
                            *wb = v.get("ssm_beta.weight"), *wc = v.get("ssm_conv1d.weight"),
                            *wnm = v.get("ssm_norm.weight"), *wdt = v.get("ssm_dt.bias"), *wsa = v.get("ssm_a");
            if (wqkv && wg && wout && wa && wb && wc && wnm && wdt && wsa && wqkv->native_data && wg->native_data &&
                wout->native_data && dense_tc_pair_supported(wqkv->native_type, wg->native_type, (int) N) &&
                dense_tc_supported(wout->native_type, (int) ZV) &&
                dense_tc_type_on(wout->native_type)) {
                const int64_t gi = gdn_idx[(size_t) l];
                ma.T = n;
                ma.xq = xq_;
                ma.wq = wqkv->native_data; ma.wz = wg->native_data; ma.wo = wout->native_data;
                ma.type_q = wqkv->native_type; ma.type_z = wg->native_type; ma.type_o = wout->native_type;
                DecRecArgs& ra = ma.rec;
                ra.state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
                ra.qkv = qkv_L_ + (size_t) gi * MT * C; ra.conv_w = (const float*) wc->data; ra.x = mixed_;
                ra.w_alpha = (const uint16_t*) wa->data; ra.w_beta = (const uint16_t*) wb->data;
                ra.dt = (const float*) wdt->data; ra.ssm_a = (const float*) wsa->data; ra.z = z_;
                ra.gamma = (const float*) wnm->data; ra.h = h_L_ + (size_t) gi * MT * C;
                ra.gate = gate_L_ + (size_t) gi * MT * HV; ra.beta = beta_L_ + (size_t) gi * MT * HV; ra.yq = xq_;
                ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
                ma.bo = bo_;
                mid_ok = mid_chain_supported(ma);
            }
        }
        {
            // the read's first kernel follows a combine, copy or wait: a plain launch, except after the device-planned
            // tail's combine kernel (PDL2)
            const bool after_kernel = pdl && dm && dec_hc && l > lb_ && !(l == 1 && ple_on);
            PdlScope ps(pdl, !after_kernel);
            if (dec_hc) {
                HcFusedArgs h = hc_args(0, pending, inj2_, inj_);
                if (read_q8) h.xq = xq_;
                // the GDN qkv+z pair kernel follows (dense_tc_pair_mask & 16): its blocks start on the SMs the read leaves
                h.next_early = mid_ok || (read_q8 && (kernels::dense_tc_pair_mask() & 16) &&
                               !is_qsa_layer(g, l) && N == 2560 && n <= kDecMaxT &&
                               dec_gdnf_mode() != 0);
                // a QSA layer's query projection follows (qsaf_wide path): its blocks start with the read
                if (read_q8 && is_qsa_layer(g, l) && N == 2560 && n <= kDecMaxT) {
                    const QsaState& st0 = ss.qsa_states[qsa_idx[(size_t) l]];
                    h.next_early = st0.kv_int8 && !st0.kv_rot && !st0.kv_hybrid && !st0.kv_q4 &&
                                   qsa_dec_fused_supported(s) &&
                                   qs_[0] != nullptr;
                }
                h.pf[0] = wn[1]->data; h.pf_bytes[0] = (size_t) HC * N * sizeof(float);
                hc_fused_read(h, cs);
            } else {
                gr_read_group(0, pending, inj2_, inj_);
            }
        }
        stamp(l, 1);
        if (gs_ring) {
            if (!read_q8) native_quantize_q8_1(mixed_, xq_, (int) N, n, cs);
            const int64_t r = l - lb_;
            dec_moe_mix_publish(xq_, mixed_, n, (int) N, gs_m_xq_ + (size_t) r * dm_->xq_stride,
                                gs_m_x_ + (size_t) r * dm_->x_stride, dm_->m_w + 48, (uint32_t) (r + 1), gs_split, cs);
        }
        if (lazy_layer(l)) {   // the layer's commit, forked in the previous post
            cudaStreamWaitEvent(cs, lz_join_, 0);
            lz_open = false;
        }
        if (mid_ok) {
            // the first read's dependent: its first pair tiles are already issued by the time the read ends (PDL)
            PdlScope ps(pdl, mid_ord == 0);
            ma.ord = mid_ord++;
            mid_chain(ma, cs);
        }
        float* xm = mixed_;
        try {
            if (mid_ok) {
                // the middle ran in the mid chain
            } else if (!is_qsa_layer(g, l)) {
                // ======================= GDN =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                float* qkv = qkv_L_ + (size_t) gi * MT * C;
                float* hb = h_L_ + (size_t) gi * MT * C;
                float* gate = gate_L_ + (size_t) gi * MT * HV;
                float* beta = beta_L_ + (size_t) gi * MT * HV;
                PdlScope ps(pdl);   // the layer's GDN kernels, after the hc read
                if (gs_split) {   // CUDA0's key heads in place (dec_gdn_rec's head map); the peer's arrive in the combine
                    const GsLayer& GL = gs_layers_[(size_t) gi];
                    const int C0 = (int) (C / HK * gs_nk0_), Z0 = (int) (ZV / HK * gs_nk0_);
                    kernels::dense_tc_pair(GL.tq, GL.wqkv0, gs_qkv0_, C0, GL.tz, GL.wz0, gs_z0_, Z0, xq_, (int) N, n, cs);
                    DecRecArgs ra;
                    ra.state = state; ra.qkv = gs_qkv0_; ra.conv_w = (const float*) wc->data; ra.x = xm;
                    ra.w_alpha = (const uint16_t*) wa->data; ra.w_beta = (const uint16_t*) wb->data;
                    ra.dt = (const float*) wdt->data; ra.ssm_a = (const float*) wsa->data; ra.z = gs_z0_;
                    ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = xq_;
                    ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
                    ra.k0 = 0; ra.nk = gs_nk0_; ra.qkv_out = qkv;
                    dec_gdn_rec(ra, n, cs);
                    dense_proj(4, GL.to, GL.wo0, xq_, bo_, Z0, (int) N, n, cs);
                    goto gdn_done;
                }
                // conv, gates, recurrence, gated norm and the output's q8_1 in dec_gdn_rec's one or two launches; at
                // 2/3 the z projection runs on the side stream beside the recurrence (dec_gdn.hpp)
                {
                // a batch window may take the fused recurrence where the qkv and z projections are one pair launch
                const bool pair_ok = kernels::dense_tc_pair_supported(wqkv->native_type, wg->native_type, (int) N) &&
                                     ((uintptr_t) wqkv->native_data & 15) == 0 && ((uintptr_t) wg->native_data & 15) == 0;
                const int gdnf = ((!batch_rec_ || pair_ok) && N == 2560 && n <= kDecMaxT) ? dec_gdnf_mode() : 0;
                auto rec = [&](const DecRecArgs& ra) {
                    if (batch_rec_) dec_gdn_rec_seg(ra, segb_, (size_t) gi * gdn_floats, n, n, cs);
                    else dec_gdn_rec(ra, n, cs);
                };
                const bool gdnf_fork = gdnf >= 2 && sh_cs_ != nullptr && ev_fork_ != nullptr && ev_join_ != nullptr;
                auto gdnf_z_side = [&] {
                    cudaEventRecord(ev_fork_, cs);
                    cudaStreamWaitEvent(sh_cs_, ev_fork_, 0);
                    PdlScope pz(pdl, true);
                    dense_proj(2, wg->native_type, wg->native_data, xq_, z_, (int) N, (int) ZV, n, sh_cs_);
                    cudaEventRecord(ev_join_, sh_cs_);
                };
                if (!read_q8) native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                // qkv and z in one dense_tc_pair launch for every window width, then the
                // one-launch recurrence with the gated norm
                if (gdnf && !pair_ok)
                    note_fallback("GDN qkv + z without dense_tc_pair", wqkv->native_type,
                                  wg->native_type);
                if (gdnf && pair_ok) {
                    {
                        PdlScope pp(pdl);   // after the hc read or the q8_1 kernel
                        kernels::dense_tc_pair(wqkv->native_type, wqkv->native_data, qkv, (int) C,
                                               wg->native_type, wg->native_data, z_, (int) ZV, xq_, (int) N, n, cs);
                    }
                    for (int i = 2; i <= 5; ++i) stamp(l, i);
                    DecRecArgs ra;
                    ra.state = state; ra.qkv = qkv; ra.conv_w = (const float*) wc->data; ra.x = xm;
                    ra.w_alpha = (const uint16_t*) wa->data; ra.w_beta = (const uint16_t*) wb->data;
                    ra.dt = (const float*) wdt->data; ra.ssm_a = (const float*) wsa->data; ra.z = z_;
                    ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = xq_;
                    ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
                    ra.early_trigger = pdl;   // PDL2: the out projection stages its weights
                    {
                        PdlScope pr(pdl);
                        rec(ra);
                    }
                    stamp(l, 6);
                    dense_proj(4, wout->native_type, wout->native_data, xq_, bo_, (int) ZV, (int) N, n, cs);
                    goto gdn_done;
                }
                const bool pk_z = gdnf_fork && sh_hi_ != nullptr && n <= 2;
                if (pk_z) {   // z at the greatest priority beside qkv, then the one-launch recurrence
                    cudaEventRecord(ev_fork_, cs);
                    cudaStreamWaitEvent(sh_hi_, ev_fork_, 0);
                    {
                        PdlScope pz(pdl, true);
                        dense_proj(2, wg->native_type, wg->native_data, xq_, z_, (int) N, (int) ZV, n, sh_hi_);
                    }
                    cudaEventRecord(ev_join_, sh_hi_);
                    dense_proj(1, wqkv->native_type, wqkv->native_data, xq_, qkv, (int) N, (int) C, n, cs);
                    stamp(l, 2);
                    DecRecArgs ra;
                    ra.state = state; ra.qkv = qkv; ra.conv_w = (const float*) wc->data; ra.x = xm;
                    ra.w_alpha = (const uint16_t*) wa->data; ra.w_beta = (const uint16_t*) wb->data;
                    ra.dt = (const float*) wdt->data; ra.ssm_a = (const float*) wsa->data; ra.z = z_;
                    ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = xq_;
                    ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
                    cudaStreamWaitEvent(cs, ev_join_, 0);
                    for (int i = 3; i <= 5; ++i) stamp(l, i);
                    {
                        PdlScope pr(pdl, true);   // after the join
                        dec_gdn_rec(ra, n, cs);
                    }
                    stamp(l, 6);
                    dense_proj(4, wout->native_type, wout->native_data, xq_, bo_, (int) ZV, (int) N, n, cs);
                    goto gdn_done;
                }
                if (gdnf_fork && gdnf == 3) gdnf_z_side();
                dense_proj(1, wqkv->native_type, wqkv->native_data, xq_, qkv, (int) N, (int) C, n, cs);
                stamp(l, 2);
                if (gdnf) {
                    DecRecArgs ra;
                    ra.state = state; ra.qkv = qkv; ra.conv_w = (const float*) wc->data; ra.x = xm;
                    ra.w_alpha = (const uint16_t*) wa->data; ra.w_beta = (const uint16_t*) wb->data;
                    ra.dt = (const float*) wdt->data; ra.ssm_a = (const float*) wsa->data; ra.z = z_;
                    ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = xq_;
                    ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
                    ra.early_trigger = pdl;   // PDL2: the out projection stages its weights
                    if (gdnf_fork) {
                        if (gdnf == 2) gdnf_z_side();
                        ra.o = y_;
                        dec_gdn_rec_split(ra, n, cs);
                        for (int i = 3; i <= 5; ++i) stamp(l, i);
                        cudaStreamWaitEvent(cs, ev_join_, 0);
                        PdlScope pn(pdl, true);   // the norm follows the join
                        dec_gdn_rec_norm(ra, n, cs);
                        stamp(l, 6);
                        dense_proj(4, wout->native_type, wout->native_data, xq_, bo_, (int) ZV, (int) N, n, cs);
                    } else {
                        dense_proj(2, wg->native_type, wg->native_data, xq_, z_, (int) N, (int) ZV, n, cs);
                        for (int i = 3; i <= 5; ++i) stamp(l, i);
                        dec_gdn_rec(ra, n, cs);
                        stamp(l, 6);
                        dense_proj(4, wout->native_type, wout->native_data, xq_, bo_, (int) ZV, (int) N, n, cs);
                    }
                    goto gdn_done;
                }
                }
                if (batch_rec_) {   // each row from its segment's conv history, the slots read from the table
                    gdn_conv_l2_seg(segb_, (size_t) gi * gdn_floats + (size_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size,
                                    qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs);
                } else
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, 0);
                stamp(l, 3);
                gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                             (const float*) wsa->data, gate, beta, (int) N, (int) HV,
                             n, cs);
                stamp(l, 4);
                dense_proj(2, wg->native_type, wg->native_data, xq_, z_, (int) N, (int) ZV, n, cs);
                stamp(l, 5);
                // the recurrence from the untouched state over tokens [0, T); outputs only for this group's
                if (batch_rec_) {   // each segment's recurrence from its own slot's state
                    gdn_step_norm_seg(segb_, (size_t) gi * gdn_floats, hb, (int) C, gate, beta, z_, (const float*) wnm->data,
                                      EPS, y_, (int) HK, (int) HV, n, cs);
                } else
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                    (int) HV, T, nullptr, cs, 0);
                stamp(l, 6);
                native_quantize_q8_1(y_, xq_, (int) ZV, n, cs);
                dense_proj(4, wout->native_type, wout->native_data, xq_, bo_, (int) ZV, (int) N, n, cs);
            gdn_done:;
            } else {
                // ======================= QSA =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, cs);
                    native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), pos, cs);
                };
                float* idx_raw = idx_raw_L_ + (size_t) qi * MT * ID;
                // the per-token GEMVs / norms / RoPEs / copies of this layer as one launch over the window's rows each
                // - row-wise identical arithmetic
                const bool qb = n > 1 && !st.kv_q4;
                // the K norm + RoPE, the INT8 appends, the tail snapshot and the indexer appends of the group's rows in
                // one launch, the same bits
                const bool fused_pre = st.kv_int8 && !st.kv_rot && !st.kv_hybrid && !st.kv_q4 &&
                                       qsa_dec_fused_supported(s);
                auto kv_post_args = [&] {
                    QsaDecKvPost kp;
                    kp.kcur = kcur_; kp.vcur = vcur_; kp.pos_k = pos_k;
                    kp.k_norm = (const float*) wkn->data; kp.steps = step_;
                    kp.k_q = st.k_q; kp.v_q = st.v_q; kp.k_scale = st.k_scale; kp.v_scale = st.v_scale;
                    kp.page_table = st.page_table; kp.host = st.host;
                    kp.idx_raw = idx_raw; kp.idx_k_norm = (const float*) wikn->data;
                    kp.tail = st.idx_tail; kp.dead = st.idx_dead; kp.pooled = st.idx_pooled; kp.block_pos = st.idx_block_pos;
                    kp.tail_snap = batch_rec_ ? tail_snap_b_ : tail_snap_ + (size_t) qi * TS;
                    kp.max_cells = (int) st.max_cells; kp.eps = EPS;
                    return kp;
                };
                // a batch window's rows are different sequences: the per-slot pointers come from the layer's table
                const QsaSegTable seg_tab{batch_rec_ ? d_qslots_[(size_t) qi] : nullptr, d_qrows_, d_qsegs_, (int) n, (int) n};
                auto kv_post = [&](const QsaDecKvPost& a, cudaStream_t q) {
                    if (batch_rec_) qsa_dec_kv_post_seg(a, seg_tab, (int) n, s, rope_scaling(), q);
                    else qsa_dec_kv_post(a, n, s, rope_scaling(), q);
                };
                auto q_post_args = [&] {
                    QsaDecQPost qp;
                    qp.q_full = qfull_; qp.qcur = qcur_; qp.pos_q = pos_;
                    qp.q_norm = (const float*) wqn->data; qp.q_idx = qidx_; qp.pos_i = pos_i;
                    qp.idx_q_norm = (const float*) wiqn->data; qp.eps = EPS;
                    return qp;
                };
                // K, V and the indexer key/query each on a branch of their own, beside the query (one row: the
                // multi-row GEMV is the one-row one)
                const bool qsaf_wide = fused_pre && (qb || n == 1) && qs_[0] != nullptr;
                // the indexer appends gate the scores, the K/V appends the attention chunks: each on its own branch,
                // joined after the query post, which runs beside them
                if (qsaf_wide) {
                    {
                        PdlScope ps(pdl);   // after the hc read
                        if (!read_q8) native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                    }
                    cudaEventRecord(qev_[0], cs);
                    for (cudaStream_t q : qs_) cudaStreamWaitEvent(q, qev_[0], 0);
                    {
                        PdlScope pz(pdl, true);
                        dense_proj(32, wv->native_type, wv->native_data, xq_, vcur_, (int) N, (int) (NKV * HD),
                                   n, qs_[1]);
                    }
                    cudaEventRecord(qev_[1], qs_[1]);
                    {
                        PdlScope pz(pdl, true);
                        bf16_gemv_fp32_mmvf_multi(mixed_, N, (const uint16_t*) wik->data, idx_raw, ID, N, ID,
                                                  n, qs_[2]);
                    }
                    {
                        QsaDecKvPost ia = kv_post_args();
                        ia.part = 2;
                        kv_post(ia, qs_[2]);
                    }
                    cudaEventRecord(qev_[2], qs_[2]);
                    {
                        PdlScope pz(pdl, true);
                        bf16_gemv_fp32_mmvf_multi(mixed_, N, (const uint16_t*) wiq->data, qidx_,
                                                  IQ * ID, N, IQ * ID, n, qs_[1]);
                    }
                    {
                        PdlScope pz(pdl, true);
                        dense_proj(32, wk->native_type, wk->native_data, xq_, kcur_, (int) N, (int) (NKV * HD),
                                   n, qs_[0]);
                    }
                    cudaStreamWaitEvent(qs_[0], qev_[1], 0);
                    {
                        QsaDecKvPost ka = kv_post_args();
                        ka.part = 1;
                        kv_post(ka, qs_[0]);
                    }
                    cudaEventRecord(qev_[3], qs_[0]);
                    cudaEventRecord(qev_[4], qs_[1]);
                    {
                        PdlScope pq(pdl);   // after the q8_1 kernel
                        dense_proj(8, wq->native_type, wq->native_data, xq_, qfull_, (int) N,
                                   (int) (NH * 2 * HD), n, cs);
                    }
                    cudaStreamWaitEvent(cs, qev_[4], 0);
                    for (int i = 7; i <= 9; ++i) stamp(l, i);
                    qsa_dec_q_post(q_post_args(), n, s, rope_scaling(), cs);
                    cudaStreamWaitEvent(cs, qev_[3], 0);
                    cudaStreamWaitEvent(cs, qev_[2], 0);
                } else {
                {
                    PdlScope ps(pdl);   // after the hc read, up to the K/V norms
                    if (!read_q8) native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                    if (qb) bf16_gemv_fp32_mmvf_multi(mixed_, N, (const uint16_t*) wik->data, idx_raw, ID, N, ID, n, cs);
                    else for (int t = 0; t < T; ++t)
                        bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, cs);
                    stamp(l, 7);
                    dense_proj(32, wk->native_type, wk->native_data, xq_, kcur_, (int) N, (int) (NKV * HD), n, cs);
                    dense_proj(32, wv->native_type, wv->native_data, xq_, vcur_, (int) N, (int) (NKV * HD), n, cs);
                }
                if (fused_pre) {
                    kv_post(kv_post_args(), cs);
                    stamp(l, 8);
                } else {
                    if (qb) norm_rope(kcur_, wkn, (int) (n * NKV), (int) HD, pos_k);
                    else for (int t = 0; t < T; ++t) norm_rope(kcur_ + t * NKV * HD, wkn, (int) NKV, (int) HD, pos_ + t * NH);
                    if (st.kv_rot) {   // K and V rotated before they are stored (kv_q4.hpp)
                        fwht256_inplace_cuda(kcur_, (int64_t) n * NKV, cs);
                        fwht256_inplace_cuda(vcur_, (int64_t) n * NKV, cs);
                    } else if (st.kv_hybrid) {   // K8V4: only V is rotated
                        fwht256_inplace_cuda(vcur_, (int64_t) n * NKV, cs);
                    }
                    stamp(l, 8);
                    copy_from_mapped(tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, cs);
                    for (int t = 0; t < T; ++t) {
                                                const int32_t* step_t = step_ + t * kStepCount;
                        if (st.kv_hybrid) {   // K8V4: the unused half's lanes folded onto the used pool (layer.cpp)
                            kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_t,
                                              kcur_ + t * NKV * HD, kcur_ + t * NKV * HD, s, cs, nullptr);
                            kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, step_t, vcur_ + t * NKV * HD,
                                              vcur_ + t * NKV * HD, s, cs, nullptr);
                        } else if (st.kv_q4)
                            kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, kcur_ + t * NKV * HD,
                                              vcur_ + t * NKV * HD, s, cs, &st.host);
                        else if (st.kv_int8)
                            kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t,
                                              kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                        else
                            kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kcur_ + t * NKV * HD,
                                           vcur_ + t * NKV * HD, s, cs, &st.host);
                    }
                    for (int t = 0; t < T; ++t) {
                        const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                        native_qsa_indexer_append(idx_raw + t * ID, step_ + t * kStepCount + kStepPos, 0,
                                                  (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                  rope_scaling(), cs);
                    }
                }
                stamp(l, 9);
                {
                    PdlScope pq(pdl2 && fused_pre && qb);   // PDL2: after the K/V post kernel
                    dense_proj(8, wq->native_type, wq->native_data, xq_, qfull_, (int) N,
                               (int) (NH * 2 * HD), n, cs);
                    if (fused_pre && qb)
                        bf16_gemv_fp32_mmvf_multi(mixed_, N, (const uint16_t*) wiq->data, qidx_,
                                                  IQ * ID, N, IQ * ID, n, cs);
                }
                if (fused_pre) {   // the query split, norms and RoPEs in one launch
                    if (qb) {}   // the indexer query projection ran above
                    else for (int t = 0; t < T; ++t)
                        bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wiq->data, qidx_ + t * IQ * ID, (int) N,
                                            (int) (IQ * ID), cs);
                    qsa_dec_q_post(q_post_args(), n, s, rope_scaling(), cs);
                } else if (qb) {
                    if (cudaMemcpy2DAsync(qcur_, (size_t) HD * 4, qfull_, (size_t) HD * 2 * 4,
                                          (size_t) HD * 4, (size_t) (n * NH), cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                        err = "verify: the q/gate split failed";
                        return false;
                    }
                    norm_rope(qcur_, wqn, (int) (n * NH), (int) HD, pos_);
                    if (st.kv_rot) fwht256_inplace_cuda(qcur_, (int64_t) n * NH, cs);   // <Hq, Hk> = <q, k>
                    bf16_gemv_fp32_mmvf_multi(mixed_, N, (const uint16_t*) wiq->data, qidx_, IQ * ID,
                                              N, IQ * ID, n, cs);
                    norm_rope(qidx_, wiqn, (int) (n * IQ), (int) ID, pos_i);
                } else {
                for (int t = 0; t < T; ++t) {
                    float* qc = qcur_ + t * NH * HD;
                    if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4,
                                          (size_t) HD * 4, (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                        err = "verify: the q/gate split failed";
                        return false;
                    }
                    norm_rope(qc, wqn, (int) NH, (int) HD, pos_ + t * NH);
                    if (st.kv_rot) fwht256_inplace_cuda(qc, NH, cs);   // <Hq, Hk> = <q, k>
                }
                for (int t = 0; t < T; ++t) {
                    float* qx = qidx_ + t * IQ * ID;
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wiq->data, qx, (int) N, (int) (IQ * ID), cs);
                    norm_rope(qx, wiqn, (int) IQ, (int) ID, pos_ + t * NH);
                }
                }
                }
                stamp(l, 10);
                bool fused_gate = false;
                bool attn_q8 = false;   // the merge wrote the q8_1 rows
                if (batch_rec_) {   // each row selects and attends over its own slot's K/V, from the layer's table
                    qsa_block_scores_seg(seg_tab, qidx_, step_, n, max_blocks_, s, scores_, cs, true);
                    qsa_block_topk(scores_, step_, n, max_blocks_, cap_, s, sel_, cs);
                    stamp(l, 11);
                    stamp(l, 12);
                    fused_gate = true;
                    attn_q8 = true;
                    qsa_dec_attn_seg(qcur_, seg_tab, true, sel_, step_, cap_, s, attn_scratch_, attn_, n, cs, qfull_,
                                     attn32_, xq_);
                } else {
                {
                    {
                        PdlScope pb(pdl);   // after the query post kernel
                        qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_, step_, n,
                                         max_blocks_, s, scores_, cs, -1, true);
                    }
                    PdlScope pt(pdl);   // after the scores kernel
                    qsa_block_topk(scores_, step_, n, max_blocks_, cap_, s,
                                   sel_, cs);
                }
                stamp(l, 11);
                // KV streaming: the n selections' blocks resident (device-side, inside the graph)
                qsa_kv_resolve(st, *g_, sel_, step_, n, cap_, cs);
                stamp(l, 12);
                const QsaAttnPools pools = qsa_attn_pools(st);
                // staged chunks and the gate in the merge, the same bits
                fused_gate = !st.kv_rot && !st.kv_hybrid &&
                             qsa_dec_attn_supported(pools, s);
                attn_q8 = fused_gate && !st.kv_rot && !st.kv_hybrid;
                PdlScope pa(pdl);   // the chunks after the top-k, the merge after them
                if (fused_gate)
                    qsa_dec_attn(qcur_, pools, sel_, step_, cap_, s,
                                 attn_scratch_, attn_, n, cs,
                                 qfull_, attn32_, attn_q8 ? xq_ : nullptr);
                else
                qsa_decode_attn_batch(qcur_, pools, sel_, step_, cap_,
                                      s, attn_scratch_, attn_, n, cs);
                }
                stamp(l, 13);
                if (st.kv_rot || st.kv_hybrid) fwht256_inplace_cuda(attn_, (int64_t) n * NH, cs);   // back: H^-1 = H
                if (fused_gate) {}   // the merge wrote attn32_
                else if (qb) native_qsa_gate_apply(attn_, qfull_, attn32_,
                                              (int) (n * NH), (int) HD, cs);
                else
                for (int t = 0; t < T; ++t) {
                    native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD,
                                          (int) NH, (int) HD, cs);
                }
                stamp(l, 14);
                PdlScope ps(pdl);   // after the gate kernel
                if (!attn_q8) native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), n, cs);
                dense_proj(16, wo->native_type, wo->native_data, xq_, bo_, (int) (NH * HD), (int) N, n, cs);
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        if (gs_split)   // the peer's key heads added after CUDA0's
            dec_moe_mix_combine(dm_->m_w + 52, (uint32_t) (l - lb_ + 1), gs_m_part_, bo_, n * (int) N, cs);
        else if (gs_ring)
            dec_moe_mix_wait(dm_->m_w + 52, (uint32_t) (l - lb_ + 1), cs);
        stamp(l, 16);
        bool routed = false;   // the fused read ran the router too
        const bool sh_fork = !prof_on_ && sh_cs_ != nullptr && ev_fork_ != nullptr &&
                             ev_join_ != nullptr;
        cudaStream_t sh_stream = sh_fork ? (sh_hi_ != nullptr ? sh_hi_ : sh_cs_) : cs;
        // the ring on its own stream, forked after the router, joined where the side branch is
        const bool pk_ring = dm && sh_fork && sh_stream == sh_hi_ && ring_hi_ != nullptr;
        ring_fork = pk_ring;
        // the read writes the q8_1 rows, post's gate/up launch plans CUDA0's share
        glue_on = false;
        if (pk_ring && dm_->ptab0 != nullptr && dec_hc && router_multi && hc_fused_q8_rows_ok()) {
            const auto& f = basalt::kernels::cpu::expert_layout().fmt[(size_t) l];
            glue_on = dm_->vram < 0 && dec_moe_glue_ok(native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff));
            if (!glue_on) note_fallback("glue off for the layer's experts", f.gu_type, f.d_type);
        }
        // the read's router items also dot the shared expert's gate input and write its q8_1 rows: its side branch is
        // its three projections (gate and up side by side), and post's combine applies the gate
        const WeightRef* w_gi = v.get("ffn_gate_inp_shexp.weight");
        sh_read = glue_on && w_gi != nullptr && w_gi->bytes == (uint64_t) N * 2 && ((uintptr_t) w_gi->data & 15u) == 0;
        {
            PdlScope ps(pdl);   // after the out projection's kernel
            if (dec_hc) {
                HcFusedArgs h = hc_args(1, true, inj_, inj2_);
                if (router_multi) {
                    h.w_router = (const uint16_t*) w_router->data;
                    h.logits = logits_; h.ids = ids_; h.weights = w_;
                    if (glue_on) {   // the ring's block starts with the read and follows its counts
                        h.xq = nat_xq_;
                        h.pub_ctr = dm_->pub + 4 * (l - lb_);
                        h.next_early = true;
                    }
                    if (sh_read) {
                        h.w_gate_sh = (const uint16_t*) w_gi->data;
                        h.gate_sh = sh_g_;
                        h.xq_sh = sh_xq_;
                    }
                    routed = true;
                }
                const WeightRef* next_norm = nullptr;
                if (l + 1 < g.n_layers) next_norm = LayerView(wt, l + 1).get("hc_attn_norm.weight");
                else next_norm = wt.find("output_hc_norm.weight");
                if (next_norm != nullptr) { h.pf[0] = next_norm->data; h.pf_bytes[0] = (size_t) HC * N * sizeof(float); }
                hc_fused_read(h, cs);
            } else {
                gr_read_group(1, true, inj_, inj2_);
            }
        }
        if (sh_fork) {
            cudaEventRecord(ev_fork_, cs);
            cudaStreamWaitEvent(sh_stream, ev_fork_, 0);
            if (sh_read) cudaStreamWaitEvent(sh_hi2_, ev_fork_, 0);
        }
        // the window's rows routed in 2 launches (one router GEMV reading the weight once, one top-10) instead of 2
        // per token; every row's arithmetic is the single-token call's
        if (routed) {
        } else if (router_multi) {
            try {
                PdlScope ps(pdl);   // after the hc read's last kernel (the fork adds no node on this stream)
                bf16_gemv_fp32_mmvf_multi(mixed_, N, (const uint16_t*) w_router->data, logits_, NE, N,
                                          NE, n, cs);
                native_router_top10_multi(logits_, ids_, w_, n, cs);
            } catch (const std::exception& e) { err = "verify router: " + std::string(e.what()); return false; }
        } else
        for (int t = 0; t < T; ++t) {
            MoEBuffers mb = ss.moe;
            mb.logits = logits_ + t * NE; mb.ids = ids_ + t * K; mb.weights = w_ + t * K;
            if (!moe_route(wt, g, l, K, mb, mixed_ + t * N, cs, err, nullptr)) return false;
        }
        if (dm) {   // the tiers, CUDA0's plan, the q8_1 rows and the rings in one launch
            DmRouteArgs ra;
            ra.ids = ids_; ra.x = xm; ra.n_tok = n; ra.k = (int) K; ra.n_embd = (int) N;
            ra.n_expert = (int) g.n_expert; ra.layer = (int) l;
            ra.res0 = dm_->res0 + l * g.n_expert;
            ra.res1 = dm_->pdev >= 0 ? dm_->res1 + l * g.n_expert : nullptr;
            ra.cache_base = hits_.cache_base; ra.slot_off = dm_->slot_off; ra.blob = (long long) hits_.blob;
            ra.plan = dm_->plan; ra.xq = nat_xq_;
            const int64_t r = l - lb_;
            ra.ring_m = dm_->m_ring + r; ra.x_m = dm_->m_x + (size_t) r * dm_->x_stride;
            ra.xq_m = dm_->m_xq + (size_t) r * dm_->xq_stride;
            ra.host_seq_m = m_seq_; ra.peer_seq_m = dm_->m_w;
            ra.n_x = (int) dm_->xp.size();
            for (int q = 0; q < ra.n_x; ++q) {
                const DecMoe::XPeer& x = dm_->xp[(size_t) q];
                ra.xres[q] = x.res + l * g.n_expert;
                ra.xring_m[q] = x.m_ring + r;
                ra.xseq_m[q] = x.m_w;
            }
            ra.ring = (uint32_t) (r + 1); ra.ctr = dm_->ctr;
            ra.stats = dm_->wstat;
            ra.peer_cap = 0;
            ra.rows = dm_->rq != nullptr || dm_->warm != nullptr;
            PdlScope pr(false);   // after the router's kernel: a plain launch
            if (glue_on) {   // the ring's work inside post's gate/up launch, released by the read's counts
                glue_ring = ra;
                glue_ring.mode = 2; glue_ring.kinds = true;
                glue_ring.pub_ctr = dm_->pub + 4 * r; glue_ring.pub_target = hc_fused_pub_arrivals();
                ring_fork = false;
            } else if (pk_ring) {   // the ring beside, CUDA0's plan and rows here
                cudaEventRecord(ev_ring_, cs);
                cudaStreamWaitEvent(ring_hi_, ev_ring_, 0);
                DmRouteArgs rp = ra;
                rp.mode = 2; rp.xq_pub = dm_->xq_pub; rp.kinds = glue_on;
                {
                    PdlScope pp(false);
                    dec_moe_route(rp, ring_hi_);
                }
                cudaEventRecord(ev_ring_join_, ring_hi_);
                ra.mode = 1;
            }
            if (!glue_on) dec_moe_route(ra, cs);
        } else if (ar_on()) {
            resident_plan(ids_, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                          hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                          plan_, (long long) max_t_ * K, nullptr, 0, cs,
                          m_plan_err_);
        } else {
            if (device_plan_)   // every routed expert resident: this group's plan without the host
                resident_plan(ids_, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                              hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                              plan_, (long long) max_t_ * K, skip_,
                              (uint32_t) (l - lb_ + 1), cs);
            if (remote_opt_)   // the helper GPUs reduce with the routing weights - publish them
                doorbell_publish(xm, ids_, w_, (int64_t) n * N, (int64_t) n * K, m_x_,
                                 m_ids_, m_w_, m_seq_, cs);
            else {
                const int32_t* layer_res = hits_.d_res != nullptr ? (hits_.d_res + l * g.n_expert) : nullptr;
                doorbell_publish_res(xm, ids_, layer_res, (int) g.n_expert, (int64_t) n * N, (int64_t) n * K,
                                     m_x_, m_ids_, m_seq_, cs);
            }
        }
        stamp(l, 17);
        {
            const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                            *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
            if (!wgi || !wsg || !wsu || !wsd) return false;
            if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
                !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
                return false;
            NativeSharedWeights nsw;
            nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
            nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
            nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
            nsw.q8_1 = sh_fork ? sh_xq_ : xq_;
            {
            try {
                PdlScope psh(false);   // the side chain's kernels follow each other
                if (sh_read)
                    shared_expert_multi_rows(n, nsw, sh_gate_, sh_up_, shared_, N, g.n_ff, sh_stream, sh_hi2_,
                                             ev_sh2_);
                else
                    shared_expert_multi(n, xm, nsw, (const uint16_t*) wgi->data, sh_gate_,
                                        sh_up_, sh_g_, shared_, N, g.n_ff, sh_stream);
            } catch (const std::exception& e) {
                err = std::string("verify shared expert: ") + e.what();
                return false;
            }
            if (sh_fork) cudaEventRecord(ev_join_, sh_stream);
            }
        }
        if (dm) {}   // the route kernel wrote nat_xq_
        else if (basalt::kernels::cpu::expert_layout().native)
            quantize_q8_1_rows(xm, n, N, nat_xq_, cs);
        else
            quantize_q8_0_scaled(xm, hit_xq_, hit_xs_, (int64_t) n * N, cs);
        stamp(l, 18);
        return true;
    };

    // ---------------------------------------------------------------- post(l, group): experts, combine
    const bool fuse_head_gr = (le_ == g.n_layers) && (head_ != nullptr && head_->loaded()) &&
                              !cvec().covers(g.n_layers - 1);
    auto post = [&](int64_t l) -> bool {
        const int n = T;
        const uint32_t ring = (uint32_t) (l - lb_ + 1);
        const int64_t cap = (int64_t) n * K, capx = (int64_t) max_t_ * K;
        int32_t* pl = plan_;
        const int32_t* p_counts = pl;
        const int32_t* p_start = pl + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
        const unsigned long long* p_ptr2 = p_ptr + capx;
        const int32_t* p_start2 = pl + ptr_off + 4 * capx;
        float* hit_out = hit_out_;
        float* parts_out = parts_;
        const auto& lay = basalt::kernels::cpu::expert_layout();
        // the VRAM groups now; the PCIe groups once the copy engine has landed them in staging.
        // `gy`: the native launch's groups side by side (0: cap, one block row per possible group).
        auto grouped = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn, int64_t gy, float* dst_buf) {
            if (lay.native) {
                // the layer's GGUF formats (i-quant gate/up, Q2_0 / IQ4_NL down)
                const auto& f = lay.fmt[(size_t) l];
                const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
                native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap,
                                      nat_xq_, hit_scratch_, dst_buf, cs, gy);
            } else {
                moe_grouped_s2(gp, gs, gn, p_dst, p_tok, cap, cap, hit_xq_,
                               hit_xs_, hit_scratch_, dst_buf, cs);
            }
        };
        if (dm) {   // CUDA0's entries straight into their parts rows; the other tiers are read by the combine
            stamp(l, 19);
            const auto& f = lay.fmt[(size_t) l];
            const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            DmPlan* P = dm_->plan;
            // the next GDN layer's commit (and later ones while this layer waits for the CPU), and the next layer's
            // projection weights into L2 while this layer's combine waits
            const bool lz_next = lazy && l + 1 < le_ && (lazy_layer(l + 1) || is_qsa_layer(g, l + 1));
            if (lz_next) {
                GdnLazyPf pf;
                {
                    const LayerView nv(wt, l + 1);
                    const char* names_gdn[] = {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"};
                    const char* names_qsa[] = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "indexer.q_proj.weight",
                                               "indexer.k_proj.weight", "attn_output.weight"};
                    const bool qsa = is_qsa_layer(g, l + 1);
                    const int nr = qsa ? 6 : 3;
                    int r = 0;
                    auto add = [&](const void* p, uint64_t b) {
                        if (p != nullptr && (b & ~15ull) != 0 && r < kLazyPfRanges) { pf.p[r] = p; pf.n[r++] = b & ~15ull; }
                    };
                    for (int i = 0; i < nr; ++i) {
                        const WeightRef* w = nv.get(qsa ? names_qsa[i] : names_gdn[i]);
                        if (w == nullptr) continue;
                        if (w->native_data != nullptr)
                            add(w->native_data, (uint64_t) iq_row_bytes(w->native_type, w->ne0) * (uint64_t) w->ne1);
                        else
                            add(w->data, w->bytes);
                    }
                    pf.ids = ids_; pf.n_ids = n * (int) K; pf.n_expert = (int) g.n_expert; pf.ptab = dm_->ptab0 + l * g.n_expert;
                }
                cudaEventRecord(lz_fork_, cs);
                cudaStreamWaitEvent(lz_s_, lz_fork_, 0);
                gdn_commit_lazy(lazy_args, commit_ctl_,
                                lazy_layer(l + 1) ? (int) (gdn_idx[(size_t) (l + 1)] - gdn_first) : 0,
                                dm_->tags ? dm_->cseen : nullptr, ring, lz_s_, fin_blocks_, &pf);
                cudaEventRecord(lz_join_, lz_s_);
                lz_open = true;
            }
            if (glue_on) {
                DmGlueArgs ga;
                ga.ids = ids_; ga.ptab = dm_->ptab0 + l * g.n_expert;
                ga.n_tok = n; ga.k = (int) K; ga.n_expert = (int) g.n_expert;
                ga.plan = P; ga.xq = nat_xq_;
                { ga.ring_on = true; ga.ring = glue_ring; }
                if (dm_->landed_pin != nullptr) {
                    ga.landed = dm_->landed + (l - lb_);
                    ga.landed_need = dm_->landed + dm_->rings;
                    ga.landed_stop = dm_->m_w + 60;
                }
                dec_moe_glue(L, ga, hit_scratch_, parts_out, (int) cap, cs);
            } else if (dm_->vram >= 0) {
                landed_glue_ = false;
                dec_moe_vram(L, P, nat_xq_, dm_->vscr, parts_out, dm_->vsync, cs, dm_->vram,
                             (int) cap);
            } else {
                landed_glue_ = false;
                native_expert_grouped(L, P->ptr, P->start, P->counts, P->dst, P->tok, cap, cap,
                                      nat_xq_, hit_scratch_, parts_out, cs);
            }
            stamp(l, 20);
            stamp(l, 21);
            stamp(l, 22);
        } else if (ar_on()) {
            stamp(l, 19);
            grouped(p_ptr, p_start, p_counts, 0, parts_out);
            stamp(l, 20);
        } else {
            if (device_plan_) {   // skipped when the device planned this group (all its experts resident)
                wait_flag_ge_or(m_flagA_, ring, skip_, cs);
                copy_i32_from_mapped_unless(pl, m_plan_, plan_i32_, skip_, ring, cs);
            } else {
                wait_flag_ge(m_flagA_, ring, cs);                  // the pool published this group's GPU plan
                copy_i32_from_mapped(pl, m_plan_, plan_i32_, cs);
            }
            stamp(l, 19);
            grouped(p_ptr, p_start, p_counts, 0, hit_out);
            stamp(l, 20);
            if (device_plan_) wait_flag_ge_or(m_flagB_, ring, skip_, cs);
            else wait_flag_ge(m_flagB_, ring, cs);                 // the PCIe groups are planned
            {   // stage them with a copy kernel, then point at staging
                const int64_t per = kStagingBlobs;
                uint8_t* stage = staging_;
                fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs);
                rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), cs);
            }
            stamp(l, 21);
            // the PCIe share is pcie_frac of the misses: a few groups when the cache is cold, usually none (none when the
            // share is 0), so its launch is kPcieGroupRows block rows striding over the groups, not cap of them
            grouped(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows, hit_out);
            stamp(l, 22);
            if (device_plan_) {   // no CPU share when the device planned the group: its rows are zeros
                wait_flag_ge_or(m_flag_, ring, skip_, cs);
                copy_or_zero_from_mapped(parts_out, m_ymiss_, (long long) n * K * N,
                                         skip_, ring, cs);
            } else {
                wait_flag_ge(m_flag_, ring, cs);               // the CPU's share is in the mapped rows
                stamp(l, 23);
                if (remote_opt_)   // the helper's rows come back reduced; skip them as well
                    remote_opt_->copy_rows(parts_out, m_ymiss_, 0, n, p_dst, p_counts + 1, cs);
                else   // only the CPU rows cross PCIe (p_dst[0, counts[1]) = the GPU's own rows)
                    copy_rows_from_mapped(parts_out, m_ymiss_, (int64_t) n * K, N,
                                          p_dst, p_counts + 1, cs);
            }
            moe_hit_add(parts_out, hit_out, p_dst, p_counts + 1, cap, N, cs);
        }
        if (!prof_on_ && sh_cs_ != nullptr && ev_fork_ != nullptr && ev_join_ != nullptr) {
            cudaStreamWaitEvent(cs, ev_join_, 0);
            if (ring_fork) cudaStreamWaitEvent(cs, ev_ring_join_, 0);   // the ring's launch
        }
        if (dm) {
            // the combine polls the peers' rows themselves (with further peers always: the wait counts one done word);
            // one peer's done word also ends the poll, but with further peers only a release does (word 60).  Without
            // --wait-stats it also waits for the CPU pool's word itself, with no wait launch before it.
            const bool dm_tag = dm_->tags || (dm_->tag && (dm_->wstat == nullptr || !dm_->xp.empty()));
            const volatile uint32_t* tag_done = !dm_tag ? nullptr : dm_->xp.empty() ? dm_->m_w + 16 : dm_->m_w + 60;
            if (!dm_->tags) {
                PdlScope pw(false);
                dec_moe_wait(dm_->plan, dm_->m_w + 16, m_flag_, ring, cs, dm_->wstat, (int) l, true, dm_tag);
            }   // true: the combine is the wait's programmatic dependent
            stamp(l, 23);
            try {
                dec_moe_combine(parts_, dm_->m_rows, m_ymiss_, dm_->plan, w_, shared_, bo_, N, K, n, nullptr, nullptr,
                                ring, cs, !dm_->tags, tag_done, dm_->tags ? m_flag_ : nullptr, dm_->cseen,
                                sh_read ? sh_g_ : nullptr);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else if (n > 1) {   // one launch for the window's rows
            try {
                native_moe_combine_multi(parts_, w_, shared_, bo_, N, K, n, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        for (int t = 0; t < T; ++t) {
            MoEBuffers mb = ss.moe;
            mb.weights = w_ + t * K; mb.shared = shared_ + t * N;
            if (!moe_combine_parts(g, l, K, mb, parts_ + (size_t) t * K * N, bo_ + t * N, cs, err)) return false;
        }
        if (remote_opt_) remote_opt_->combine(bo_, y_dummy_, 0, n,
                              device_plan_ ? skip_ : nullptr, ring, cs);
        stamp(l, 24);
        if (l == g.n_layers - 1) {
            if (!fuse_head_gr) {
                for (int t = 0; t < T; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                if (cvec().covers(l)) cvec_apply(Rt(0), l, n, HC * N, nullptr, 0, nullptr, 0, false, cs);
            }
        } else if (cvec().covers(l)) {
            cvec_apply(Rt(0), l, n, HC * N, bo_, N, inj2_, HC, true, cs);
        }
        return true;
    };

    if (!pre(lb_)) return false;
    for (int64_t l = lb_; l < le_; ++l) {
        if (!post(l)) return false;
        if (l + 1 < le_ && !pre(l + 1)) return false;
    }
    if (lz_open) cudaStreamWaitEvent(cs, lz_join_, 0);
    if (le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head
        float* hout = hand_out_ + (size_t) hrow0 * HB;
        copy_from_mapped(hout, R_, (int64_t) T * HC * N, cs);
        copy_from_mapped(hout + (size_t) T * HC * N, bo_, (int64_t) T * N, cs);
        copy_from_mapped(hout + (size_t) T * (HC + 1) * N, inj2_, (int64_t) T * HC, cs);
        return true;
    }

    // ---- the head, T columns, and the argmax of each
    stamp(g.n_layers, 0);
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        if (fuse_head_gr) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                FusedGrArgs& a = fa[t];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = true;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj2_ + t * HC;
                a.w_norm = (const float*) hn->data;
                a.w_down = (const uint16_t*) hd->data;
                a.w_up = (const uint16_t*) hu->data;
                a.w_inject = nullptr;
                a.eps = EPS;
                a.lo = lo_ + t * g.hc_lr;
                a.rs = rs_ + t * HC;
                a.inject_out = head_inj_;
                a.mixed = head_mixed_ + t * N;
            }
            if (dec_hc) {
                HcFusedArgs h;
                h.T = T; h.R = Rt(0); h.R_out = Rt(0); h.apply = true; h.bo_prev = bo_; h.inj_prev = inj2_;
                h.w_norm = (const float*) hn->data; h.w_down = (const uint16_t*) hd->data; h.w_up = (const uint16_t*) hu->data;
                h.eps = EPS; h.lo = lo_; h.rs = rs_; h.mixed = head_mixed_; h.xn = xn_;
                hc_fused_read(h, cs);
            } else {
                fused_gr_read_multi(fa, T, xn_, cs);
            }
        } else {
            for (int t = 0; t < T; ++t) {
                BlockBuffers bb = ss.block;
                bb.R = Rt(t);
                bb.mixed = head_mixed_ + t * N;
                if (head_ != nullptr && head_->loaded()) {
                    if (!lm_head_mix(wt, g, bb, cs, err)) return false;
                } else if (!lm_head(wt, g, bb, head_logits_ + (size_t) t * n_vocab_, cs, err)) {
                    return false;
                }
            }
        }
        if (head_ != nullptr && head_->loaded()) {
            try {
                native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
                // rows [0, n) of `w` into y (T rows of n); the same kernel and bits as the whole head
                auto head_rows = [&](const void* w, float* y, int n, cudaStream_t s) {
                    const void* x = s == cs ? (const void*) xq_ : hs_xq_;
                    if (head_->type() == 13 && q5k_head_supported((int) N, T))
                        q5k_head_mmvq(w, x, y, (int) N, n, T, s);
                    else if (head_->type() == 14 && q6k_head_supported((int) N, T))
                        q6k_head_mmvq(w, x, y, (int) N, n, T, s);
                    else
                        native_mmvq(head_->type(), w, x, y, (int) N, n, T, s);
                };
                if (hs_rows_ > 0) {   // the peer's rows while CUDA0 does its own; its logits land beside CUDA0's
                    const int r0 = hs_rows_, r1 = (int) n_vocab_ - r0;
                    const size_t xb = native_q8_1_bytes((int) N, T);
                    bool ok = cudaMemcpyAsync(hs_hxq_, xq_, xb, cudaMemcpyDeviceToHost, cs) == cudaSuccess &&
                              cudaEventRecord(hs_in_, cs) == cudaSuccess;
                    {
                        const OnDevice on(hs_dev_);
                        ok = ok && cudaStreamWaitEvent(hs_s_, hs_in_, 0) == cudaSuccess &&
                             cudaMemcpyAsync(hs_xq_, hs_hxq_, xb, cudaMemcpyHostToDevice, hs_s_) == cudaSuccess;
                        if (ok) head_rows(hs_w_, hs_y_, r1, hs_s_);
                        ok = ok && cudaMemcpyAsync(hs_hy_, hs_y_, (size_t) T * r1 * 4, cudaMemcpyDeviceToHost, hs_s_) == cudaSuccess &&
                             cudaEventRecord(hs_out_, hs_s_) == cudaSuccess;
                    }
                    head_rows(head_->weights(), hs_y0_, r0, cs);
                    ok = ok && cudaMemcpy2DAsync(head_logits_, (size_t) n_vocab_ * 4, hs_y0_, (size_t) r0 * 4, (size_t) r0 * 4, T,
                                                 cudaMemcpyDeviceToDevice, cs) == cudaSuccess &&
                         cudaStreamWaitEvent(cs, hs_out_, 0) == cudaSuccess &&
                         cudaMemcpy2DAsync(head_logits_ + r0, (size_t) n_vocab_ * 4, hs_hy_, (size_t) r1 * 4, (size_t) r1 * 4, T,
                                           cudaMemcpyHostToDevice, cs) == cudaSuccess;
                    if (!ok) { err = "verify: the split head's hand-off failed"; return false; }
                } else if (head_->type() == 13 && q5k_head_supported((int) N, T))   // the same bits, one wave
                    q5k_head_mmvq(head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
                else if (head_->type() == 14 && vhead_scratch_ != nullptr && q6k_vhead_supported((int) N, (int) n_vocab_, T))
                    q6k_vhead(head_->vproxy(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T,
                              vhead_scratch_, cs);
                else if (head_->type() == 14 && q6k_head_supported((int) N, T))
                    q6k_head_mmvq(head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
                else
                    native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        // Greedy, the default, is recorded here as before (no extra launch or sync per window). A request that
        // samples or penalizes is sampled again host-side after the replay (run()) with its own parameters and a
        // fresh draw counter: a captured sampler would bake them in and replay the same draws forever.
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
    }
    stamp(g.n_layers, 1);
    return true;
}

std::string Verifier::profile_report() {
    if (!prof_on_ || prof_windows_ == 0) return std::string();
    static const char* names[kProfPer] = {"-", "hc-read0", "q8+qkv/q-idx gemv", "conv", "ab", "z", "rec", "q8+kv-idx",
                                          "k/v+norm-rope", "kv+idx append", "q+q-idx", "scores+topk", "kv-resolve",
                                          "attention", "gate", "", "out-proj", "hc-read1+router", "shared+quant",
                                          "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU", "copy+combine",
                                          "(gap)", "head", "  hc0 norm", "  hc0 down", "  hc0 up", "", "", ""};
    std::string out;
    char b[80];
    double total = 0;
    for (int k = 0; k < 2; ++k) {
        out += k == 0 ? " GDN layers:" : " | QSA layers:";
        for (int i = 0; i < kProfPer; ++i) {
            if (prof_sum_[k][i] <= 0) continue;
            total += prof_sum_[k][i];
            std::snprintf(b, sizeof b, " %s %.2f", names[i], prof_sum_[k][i] / 1e6 / (double) prof_windows_);
            out += b;
        }
    }
    std::snprintf(b, sizeof b, " | total %.2f ms/window over %lld windows", total / 1e6 / (double) prof_windows_, (long long) prof_windows_);
    out += b;
    std::snprintf(b, sizeof b, " | graph launch %.3f ms/window", ms_launch / (double) prof_windows_);
    out += b;
    ms_launch = 0;
    DecodeGaps& dg = decode_gaps();
    if (dg.n_v2d > 0 && dg.n_d2v > 0) {
        std::snprintf(b, sizeof b, " | GPU idle verify->draft %.3f, draft GPU %.3f, idle draft->verify %.3f ms",
                      dg.v2d / 1e6 / (double) dg.n_v2d, dg.draft_gpu / 1e6 / (double) dg.n_v2d,
                      dg.d2v / 1e6 / (double) dg.n_d2v);
        out += b;
    }
    if (dg.host_on) {
        static const char* hn[32] = {"peer/copy sync", "profile read", "host sample", "run tail", "commit", "emit",
                                     "draft stage", "draft launch", "draft wait", "draft sync", "draft prof read",
                                     "run entry", "stage+ple", "dm_sync", "peer launch", "graph launch",
                                     "adapt join", "observe", "apply_pending", "window/penalty", "adapt thread", "", "", "", "", "", "", "", "commit GPU",
                                     "commit>draft", "verify>commit", ""};
        out += " | host ms/window:";
        for (int i = 0; i < 31; ++i) {
            if (!hn[i][0]) continue;
            std::snprintf(b, sizeof b, " %s %.3f", hn[i], dg.hms[i] / (double) prof_windows_);
            out += b;
            dg.hms[i] = 0;
        }
        const double gs = prof_graph0_ / 1e6 / (double) prof_windows_;
        std::snprintf(b, sizeof b, " | graph start->layer0 %.3f ms", gs);
        out += b;
        prof_graph0_ = 0;
    }
    dg.v2d = dg.d2v = dg.draft_gpu = 0;
    dg.n_v2d = dg.n_d2v = 0;
    for (auto& r : prof_sum_) for (double& d : r) d = 0;
    prof_windows_ = 0;
    return out;
}

// Whether the zero-doorbell graph is right for the next window: only while every expert of [lb_, le_) is in VRAM
// now (the host table is the one the device copy was uploaded from).
void Verifier::refresh_ar() {
    if (!all_resident_ || hits_.h_res == nullptr || g_ == nullptr) return;
    const int64_t ne = g_->n_expert;
    const int32_t* r = hits_.h_res + lb_ * ne;
    const int64_t n = (le_ < 0 ? g_->n_layers : le_) * ne - lb_ * ne;
    int32_t lo = 0;
    for (int64_t i = 0; i < n; ++i) lo = std::min(lo, r[i]);
    const bool off = lo < 0;
    if (off == ar_off_) return;
    ar_off_ = off;
    basalt::log::info("basalt verify", "layers [%lld, %lld) %s", (long long) lb_,
                 (long long) (le_ < 0 ? g_->n_layers : le_),
                 off ? "have experts out of VRAM: the doorbell graph runs the windows"
                     : "are 100% VRAM resident again: the zero-doorbell graph");
}

bool Verifier::capture(int T, std::string& err) {
    cudaGraphExec_t& exec_t = ar_off_ ? exec_nr_[T] : exec_[T];
    if (exec_t != nullptr) return true;
    const bool pdl_try = basalt::kernels::pdl_requested() && !pdl_off_;
    // a window that does not capture or instantiate with programmatic dependent launch is captured again without it
    auto retry_plain = [&](const std::string& why) {
        basalt::log::warn("basalt verify", "the %d-token window failed with programmatic dependent launch (%s): capturing it without", T,
                     why.c_str());
        (void) cudaGetLastError();
        pdl_off_ = true;
        exec_t = nullptr;
        return capture(T, err);
    };
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    pdl_rec_ = pdl_try;
    basalt::kernels::pdl_count() = 0;
    const bool ok = record_window(T, cs_, rerr);
    pdl_rec_ = false;
    const long pdl_n = basalt::kernels::pdl_count();
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        if (pdl_try) return retry_plain(rerr);
        err = rerr;
        return false;
    }
    if (ce != cudaSuccess) {
        if (pdl_try) return retry_plain(cudaGetErrorString(ce));
        err = std::string("verify: end capture: ") + cudaGetErrorString(ce);
        return false;
    }
    const cudaError_t ie = cudaGraphInstantiateWithFlags(&exec_t, graph, cudaGraphInstantiateFlagUseNodePriority);
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess) {
        if (pdl_try) return retry_plain(cudaGetErrorString(ie));
        err = std::string("verify: instantiate: ") + cudaGetErrorString(ie);
        return false;
    }
    const cudaError_t ue = cudaGraphUpload(exec_t, cs_);
    const cudaError_t us = cudaStreamSynchronize(cs_);
    if (pdl_try)
        basalt::log::info("basalt verify", "captured the %d-token window (upload %s, sync %s), %ld programmatic "
                             "dependent launches", T, cudaGetErrorString(ue), cudaGetErrorString(us), pdl_n);
    else
        basalt::log::info("basalt verify", "captured the %d-token window (upload %s, sync %s)", T,
                     cudaGetErrorString(ue), cudaGetErrorString(us));
    return true;
}

basalt::kernels::GdnCommitLayers Verifier::commit_gdn_layer(int64_t gdn_index) const {
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    const SessionState& ss = *ss_;
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    GdnCommitLayers ga;
    ga.state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
    ga.h = h_L_ + (size_t) gdn_index * MT * C; ga.qkv = qkv_L_ + (size_t) gdn_index * MT * C;
    ga.gate = gate_L_ + (size_t) gdn_index * MT * HV; ga.beta = beta_L_ + (size_t) gdn_index * MT * HV;
    ga.state_stride = gdn_floats; ga.h_stride = (size_t) MT * C; ga.gb_stride = (size_t) MT * HV;
    ga.C = (int) C; ga.h_k = (int) g.ssm_k_heads; ga.h_v = (int) HV; ga.n_keep = commit_;
    ga.n_layers = 1;
    return ga;
}

// the stage's GDN layers from its first (only that one when the later ones are lazy)
basalt::kernels::GdnCommitLayers Verifier::commit_gdn_args() const {
    const ModelGeometry& g = *g_;
    int64_t gdn_index = 0;
    for (int64_t l = 0; l < lb_; ++l) gdn_index += is_qsa_layer(g, l) ? 0 : 1;
    basalt::kernels::GdnCommitLayers ga = commit_gdn_layer(gdn_index);
    ga.n_layers = 0;
    for (int64_t l = lb_; l < le_; ++l) ga.n_layers += is_qsa_layer(g, l) ? 0 : 1;
    if (lazy_on()) ga.n_layers = std::min(ga.n_layers, 1);
    return ga;
}

bool Verifier::finisher_on() const {
    return g_commit_async && next_ == nullptr && lb_ == 0 && le_ >= g_->n_layers && commit_ctl_ != nullptr &&
           fin_blocks_ > 0;
}

bool Verifier::lazy_on() const { return finisher_on() && dm_ != nullptr && gs_nk0_ == 0 && lz_s_ != nullptr; }

// the stage's GDN layers after its first, the lazy ones
basalt::kernels::GdnCommitLayers Verifier::commit_gdn_lazy_args() const {
    const ModelGeometry& g = *g_;
    int64_t gdn_index = 0, n = 0;
    for (int64_t l = 0; l < lb_; ++l) gdn_index += is_qsa_layer(g, l) ? 0 : 1;
    for (int64_t l = lb_; l < le_; ++l) n += is_qsa_layer(g, l) ? 0 : 1;
    basalt::kernels::GdnCommitLayers ga = commit_gdn_layer(gdn_index + 1);
    ga.n_layers = (int) std::max<int64_t>(n - 1, 0);
    return ga;
}

// every lazy layer the last commit still owes, on the commit's stream (what a window already took is done)
void Verifier::apply_lazy() {
    if (!lz_pending_) return;
    lz_pending_ = false;
    const basalt::kernels::GdnCommitLayers la = commit_gdn_lazy_args();
    basalt::kernels::gdn_commit_lazy(la, commit_ctl_, la.n_layers, nullptr, 0, cm_, fin_blocks_);
    cudaEventRecord(commit_done_, cm_);
    commit_pending_ = true;
}

bool Verifier::commit_begun() const {
    const Clock::time_point t0 = Clock::now();
    while (*(volatile uint32_t*) h_pend_ != commit_epoch_) {
        if (ms_since(t0) > 0.5) return false;
    }
    return true;
}

void Verifier::order_after_commit(bool solo) {
    if (!solo) apply_lazy();
    if (commit_pending_) cudaStreamWaitEvent(cs_, commit_done_, 0);
}

// The commit on its own stream: the epoch's counters, the indexer and PLE commits, then the GDN items from `kCommitBlocks`
// blocks that claim them one at a time (the next window's finisher claims what is left).
bool Verifier::capture_commit(std::string& err) {
    if (commit_exec_ != nullptr) return true;
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t ID = g.idx_key_dim, MT = max_t_;
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cm_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        if (prof_on_) gpu_stamp(prof_, (int) (g.n_layers * kProfPer + 3), cm_);
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cm_);
        gdn_commit_begin(commit_ctl_, m_commit_ + 2 + MT, m_pend_, cm_);
        int64_t qsa_index = 0;
        for (int64_t l = 0; l < lb_; ++l) qsa_index += is_qsa_layer(g, l) ? 1 : 0;
        {   // every QSA layer's indexer appends in one launch, every GDN layer's replay in another
            const GdnCommitLayers ga = commit_gdn_args();
            QsaCommitLayers qa;
            qa.pos = commit_ + 2; qa.epsilon = EPS; qa.n_tok = (int) MT;
            for (int64_t l = lb_; l < le_ && ok; ++l) {
                if (!is_qsa_layer(g, l)) continue;
                const LayerView v(*wt_, l);
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn || qa.n_layers >= kQsaCommitMaxLayers) { ok = false; break; }
                const QsaState& st = ss.qsa_states[qsa_index + qa.n_layers];
                QsaCommitLayer& q = qa.layer[qa.n_layers++];
                q.raw = idx_raw_L_ + (size_t) ((qsa_index + qa.n_layers - 1) * MT) * ID;
                q.tail_snap = tail_snap_ + (size_t) (qsa_index + qa.n_layers - 1) * TS;
                q.gamma = (const float*) wikn->data;
                q.tail = st.idx_tail; q.dead = st.idx_dead; q.pooled = st.idx_pooled; q.block_pos = st.idx_block_pos;
                q.max_cells = (int) st.max_cells;
            }
            if (ok) native_qsa_indexer_commit_layers(qa, s, rope_scaling(), cm_);
            if (ok && ss.ple.ready() && ple_stage()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cm_);
            if (ok) gdn_commit_layers_dyn(ga, commit_ctl_, cm_, kCommitBlocks);
        }
        if (prof_on_) gpu_stamp(prof_, (int) (g.n_layers * kProfPer + 4), cm_);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cm_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&commit_exec_, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    return true;
}

void Verifier::stage_inputs(int T, const int32_t* tokens, int64_t pos0) {
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    int32_t* const pk = h_pos_ + (size_t) max_t_ * g.n_head;
    int32_t* const pi = pk + (size_t) max_t_ * g.n_head_kv;
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s);
        const int32_t pos_t = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = pos_t;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = pos_t;
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = pos_t;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) (h_flag_ + 8) = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    staged_ = true;
}

DecodeGaps& decode_gaps() {
    static DecodeGaps g;
    return g;
}

DecodeEvents& decode_events() {
    static DecodeEvents e;
    return e;
}

void DecodeEvents::enable(size_t max_windows) {
    if (on) return;
    ev.assign((max_windows + 2) * kKinds, nullptr);
    set.assign(ev.size(), 0);
    tags.assign(max_windows + 2, 0);
    hs.assign(max_windows + 2, std::array<double, 10>{});
    for (auto& e : ev)
        if (cudaEventCreate(&e) != cudaSuccess) { cudaGetLastError(); e = nullptr; }
    on = true;
    // align the host clock to the event clock: poll an event's completion tightly, keep the smallest offset
    auto host_ms = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    cudaEventCreate(&cal);
    cudaEventRecord(cal, nullptr);
    cudaEventSynchronize(cal);
    cal_host = 1e30;
    cudaEvent_t e;
    cudaEventCreate(&e);
    for (int i = 0; i < 200; ++i) {
        cudaEventRecord(e, nullptr);
        while (cudaEventQuery(e) == cudaErrorNotReady) {}
        const double h = host_ms();
        float t = 0;
        cudaEventElapsedTime(&t, cal, e);
        cal_host = std::min(cal_host, h - (double) t);
    }
    cudaEventDestroy(e);
}

void DecodeEvents::record(Kind k, cudaStream_t s, size_t ahead) {
    if (!on) return;
    const size_t i = (window + ahead) * kKinds + k;
    if (i >= ev.size() || ev[i] == nullptr) return;
    cudaEventRecord(ev[i], s);
    set[i] = 1;
}

std::string DecodeEvents::report() {
    if (!on) return {};
    cudaDeviceSynchronize();
    std::vector<double> v[8];
    std::vector<std::pair<double, unsigned>> idle;
    std::vector<size_t> idle_w;
    auto at = [&](size_t w, int k) { return ev[w * kKinds + k]; };
    auto ok = [&](size_t w, int k) { return w * kKinds + k < set.size() && set[w * kKinds + k]; };
    auto el = [&](cudaEvent_t a, cudaEvent_t b) { float ms = 0; cudaEventElapsedTime(&ms, a, b); return (double) ms; };
    for (size_t w = 1; w + 1 <= window; ++w) {
        bool all = ok(w + 1, kVerifyPre) && ok(w + 1, kRunEntry);
        for (int k = 0; k < kRunEntry; ++k) all = all && ok(w, k);
        if (!all) continue;
        v[0].push_back(el(at(w, kVerifyPre), at(w, kVerifyPost)));
        v[1].push_back(el(at(w, kVerifyPost), at(w, kCommitPre)));
        v[2].push_back(el(at(w, kCommitPre), at(w, kCommitPost)));
        v[3].push_back(el(at(w, kVerifyPost), at(w, kDraftPre)));
        v[4].push_back(el(at(w, kDraftPre), at(w, kDraftPost)));
        const double c = el(at(w, kVerifyPre), at(w, kCommitPost)), d = el(at(w, kVerifyPre), at(w, kDraftPost));
        v[5].push_back(el(at(w, kVerifyPre), at(w + 1, kVerifyPre)) - std::max(c, d));
        idle.emplace_back(v[5].back(), tags[w]);
        idle_w.push_back(w);
        v[6].push_back(el(at(w, kVerifyPre), at(w + 1, kVerifyPre)));
        v[7].push_back(el(at(w + 1, kRunEntry), at(w + 1, kVerifyPre)));
    }
    static const char* nm[8] = {"verify", "verify>commit", "commit", "verify>draft", "draft", "idle before next verify",
                                "window", "run entry>launch"};
    std::string out;
    char b[400];
    std::snprintf(b, sizeof b, "basalt decode events (GPU ms, %zu windows; median / mean):", v[0].size());
    out = b;
    for (int i = 0; i < 8; ++i) {
        if (v[i].empty()) continue;
        std::vector<double> x = v[i];
        std::sort(x.begin(), x.end());
        double sum = 0;
        for (double y : x) sum += y;
        std::snprintf(b, sizeof b, " %s %.3f/%.3f;", nm[i], x[x.size() / 2], sum / (double) x.size());
        out += b;
    }
    // the idle gap's tail: the mean over the windows by tag, and how much of the total the gaps over 0.1 ms hold
    double tot = 0, big = 0;
    for (auto& g : idle) { tot += g.first; if (g.first > 0.1) big += g.first; }
    std::snprintf(b, sizeof b, " | idle total %.3f ms/window, in gaps over 0.1 ms %.3f", idle.empty() ? 0.0 : tot / (double) idle.size(),
                  idle.empty() ? 0.0 : big / (double) idle.size());
    out += b;
    std::vector<double> sorted;
    for (auto& g : idle) sorted.push_back(g.first);
    std::sort(sorted.begin(), sorted.end());
    if (!sorted.empty()) {
        std::snprintf(b, sizeof b, " p90 %.3f p99 %.3f max %.3f", sorted[sorted.size() * 9 / 10], sorted[sorted.size() * 99 / 100],
                      sorted.back());
        out += b;
    }
    for (unsigned m = 0; m < 12; ++m) {
        double sum = 0;
        long n = 0;
        for (auto& g : idle) if (m == 0 ? g.second == 0 : (g.second & (1u << (m - 1)))) { sum += g.first; ++n; }
        if (n == 0) continue;
        std::snprintf(b, sizeof b, " | tag%u: n %ld mean idle %.3f", m, n, sum / (double) n);
        out += b;
    }
    // the host's stations between a draft's return and the next launch: mean interval for gaps over 0.1 ms and the rest
    for (int big_set = 1; big_set >= 0; --big_set) {
        double sum[8] = {}, th[4] = {};
        long n = 0, thn = 0;
        for (size_t i = 0; i < idle.size(); ++i) {
            if ((idle[i].first > 0.1) != (big_set == 1)) continue;
            const auto& h = hs[idle_w[i]];
            bool ok2 = true;
            for (int k = 0; k < 6; ++k) ok2 = ok2 && h[(size_t) k] > 0;
            if (!ok2) continue;
            for (int k = 0; k < 5; ++k) sum[k] += h[(size_t) k + 1] - h[(size_t) k];
            sum[5] += idle[i].first;
            sum[6] += v[7][i];
            if (h[8] > 0 && h[7] > 0 && h[6] > 0) { th[3] += h[9] > 0 ? h[8] - h[9] : 0; th[0] += h[8] - h[0]; th[1] += h[6] - h[8]; th[2] += h[7] - h[6]; ++thn; }
            ++n;
        }
        if (n == 0) continue;
        if (thn > 0) {
            std::snprintf(b, sizeof b, " | adapt thread (windows with one, %s gaps): start after stamp0 %.3f, work %.3f, presync %.3f, start after spawn %.3f",
                          big_set ? "big" : "small", th[0] / (double) thn, th[1] / (double) thn, th[2] / (double) thn, th[3] / (double) thn);
            out += b;
        }
        std::snprintf(b, sizeof b, " | host %s gaps (n %ld) ms: draft>adapt-join %.3f join>loop-end %.3f loop-end>top %.3f top>run-dm_sync %.3f dm_sync>launch %.3f (GPU idle %.3f, of it run-entry>launch %.3f)",
                      big_set ? "big" : "small", n, sum[0] / (double) n, sum[1] / (double) n, sum[2] / (double) n,
                      sum[3] / (double) n, sum[4] / (double) n, sum[5] / (double) n, sum[6] / (double) n);
        out += b;
    }
    // the stamps on the GPU timeline: ms after the later of the commit's and the draft's end
    for (int big_set = 1; big_set >= 0; --big_set) {
        double sum[10] = {};
        long n = 0;
        for (size_t i = 0; i < idle.size(); ++i) {
            if ((idle[i].first > 0.1) != (big_set == 1)) continue;
            const size_t w = idle_w[i];
            const auto& h = hs[w];
            bool ok2 = true;
            for (int k = 0; k < 6; ++k) ok2 = ok2 && h[(size_t) k] > 0;
            if (!ok2) continue;
            const double c = el(cal, at(w, kCommitPost)), d = el(cal, at(w, kDraftPost));
            const double done = std::max(c, d) + cal_host;
            for (int k = 0; k < 6; ++k) sum[k] += h[(size_t) k] - done;
            sum[6] += el(cal, at(w + 1, kRunEntry)) + cal_host - done;
            sum[7] += el(cal, at(w + 1, kVerifyPre)) + cal_host - done;
            sum[8] += c - d;
            ++n;
        }
        if (n == 0) continue;
        std::snprintf(b, sizeof b, " | %s gaps, host stamps after the GPU's draft+commit end (ms):", big_set ? "big" : "small");
        out += b;
        static const char* sn[6] = {"draft returned", "adapt joined", "loop end", "loop top", "after dm_sync", "pre-launch"};
        for (int k = 0; k < 6; ++k) {
            const int kk = k == 5 ? 5 : k;
            std::snprintf(b, sizeof b, " %s %.3f;", sn[k], sum[kk] / (double) n);
            out += b;
        }
        std::snprintf(b, sizeof b, " run-entry event %.3f; verify-pre event %.3f; commit end - draft end %.3f", sum[6] / (double) n,
                      sum[7] / (double) n, sum[8] / (double) n);
        out += b;
    }
    std::fill(set.begin(), set.end(), 0);
    std::fill(tags.begin(), tags.end(), 0);
    for (auto& h : hs) h.fill(0.0);
    window = 0;
    return out;
}

void Verifier::collect_profile() {
    const ModelGeometry& g = *g_;
    cudaMemcpy(prof_h_.data(), prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost);
    const int64_t L = g.n_layers;
    auto at = [&](int64_t l, int i) { return prof_h_[(size_t) (l * kProfPer + i)]; };
    // D8: the derived columns below read stamps the hc-read kernels write themselves or the next
    // layer's first.  A slot no kernel stamped is 0 and its unsigned difference wrapped to ~1e19 ns -
    // which is why (gap)/head/hc0 printed ~1e14 ms per window.  A missing or out-of-order stamp now
    // contributes nothing.
    const auto gap = [](unsigned long long to, unsigned long long from) {
        return (from != 0 && to != 0 && to >= from) ? (double) (to - from) : 0.0;
    };
    for (int64_t l = 0; l < L; ++l) {
        const int kind = is_qsa_layer(g, l) ? 1 : 0;
        unsigned long long prev = at(l, 0);
        for (int i = 1; i <= 24; ++i) {
            const unsigned long long x = at(l, i);
            if (x == 0 || x < prev) continue;
            prof_sum_[kind][i] += (double) (x - prev);
            prev = x;
        }
        if (l + 1 < L) prof_sum_[kind][25] += gap(at(l + 1, 0), at(l, 24));
        const double dn = gap(at(l, 27), at(l, 0)), dd = gap(at(l, 28), at(l, 27)), du = gap(at(l, 1), at(l, 28));
        if (dn > 0 && dd > 0 && du > 0) {   // the split exists: show it split, not twice
            prof_sum_[kind][27] += dn;      // hc-read0: norm
            prof_sum_[kind][28] += dd;      //           down
            prof_sum_[kind][29] += du;      //           up (through both halves, as before)
            prof_sum_[kind][1] -= gap(at(l, 1), at(l, 0));   // (hc-read0 shown split)
        }
    }
    prof_sum_[0][26] += gap(at(L, 1), at(L, 0));
    prof_graph0_ += gap(at(0, 0), at(L, 2));
    DecodeGaps& dg = decode_gaps();
    if (dg.draft_end != 0 && at(0, 0) > dg.draft_end) {
        dg.d2v += (double) (at(0, 0) - dg.draft_end);
        ++dg.n_d2v;
        dg.hms[28] += gap(at(L, 4), at(L, 3)) / 1e6;          // the commit's GPU span
        dg.hms[29] += gap(at(L, 4), dg.draft_end) / 1e6;      // the commit still running after the draft
        dg.hms[30] += gap(at(L, 3), dg.verify_prev_end) / 1e6;   // the window's end to the commit's start
    }
    dg.verify_prev_end = at(L, 1);
    dg.draft_end = 0;
    dg.verify_end = at(L, 1);
    ++prof_windows_;
}

// The device-planned tail's window start: residency tables, the flag words, and the peer card's persistent window.
bool Verifier::dm_begin(std::string& err, DecodeGaps* gm) {
    if (!dm_sync(err)) return false;
    decode_events().stamp(4);
    if (gm) gm->mark(13);
    *(volatile uint32_t*) (dm_->h_w + 0) = 0;
    *(volatile uint32_t*) (dm_->h_w + 16) = 0;
    *(volatile uint32_t*) (dm_->h_w + 32) = 0;
    *(volatile uint32_t*) (dm_->h_w + 48) = 0;
    *(volatile uint32_t*) (dm_->h_w + 52) = 0;
    *(volatile uint32_t*) (dm_->h_w + 60) = 0;
    if (dm_->tag && dm_->tag_dirty) {   // a released window left rows written: back to unwritten
        for (size_t i = 0; i < dm_->rows_words; ++i)
            ((volatile uint32_t*) dm_->h_rows)[i] = basalt::kernels::dec_moe_tag_word();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        dm_->tag_dirty = false;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (dm_->pdev >= 0) {
        const OnDevice on(dm_->pdev);
        basalt::kernels::DmPeerArgs pa;
        pa.rings_m = dm_->h_ring; pa.xq_m = dm_->h_xq; pa.xq_stride = dm_->xq_stride;
        pa.seq_m = dm_->h_w + 0; pa.done_m = dm_->h_w + 16; pa.rows_m = dm_->h_rows;
        pa.ptr = dm_->ptab; pa.lay = dm_->lay; pa.n_expert = (int) g_->n_expert; pa.k = (int) ss_->k;
        pa.n_embd = (int) g_->n_embd; pa.n_ff = (int) basalt::kernels::cpu::expert_layout().fmt[0].n_ff;
        pa.steps = (int) dm_->rings; pa.scratch = dm_->pscr; pa.arrive = dm_->parr; pa.diag_m = dm_->h_w + 32;
        pa.rq = dm_->rq; pa.rs = dm_->rs;
        if (gs_nk0_ > 0) {
            pa.mix = gs_mix_; pa.mseq_m = dm_->h_w + 48; pa.mdone_m = dm_->h_w + 52;
            pa.mxq_m = gs_h_xq_; pa.mxq = gs_pxq_;
            pa.mx_m = gs_h_x_; pa.mixl = gs_mixl_; pa.part_m = gs_h_part_;
        }
        if (!basalt::kernels::dec_moe_zero_words(dm_->parr, 16, dm_->ps) ||
            !basalt::kernels::dec_moe_peer_launch(pa, dm_->pblocks, dm_->ps, dm_->pq2)) {
            err = "verify: the peer window did not launch";
            return false;
        }
        dm_->peer_live = true;
    }
    for (DecMoe::XPeer& x : dm_->xp) {   // further peers: the same window on their own ring
        for (int w : {0, 16, 32}) *(volatile uint32_t*) (x.h_w + w) = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const OnDevice on(x.dev);
        basalt::kernels::DmPeerArgs pa;
        pa.rings_m = x.h_ring; pa.xq_m = dm_->h_xq; pa.xq_stride = dm_->xq_stride;
        pa.seq_m = x.h_w + 0; pa.done_m = x.h_w + 16; pa.rows_m = dm_->h_rows;
        pa.ptr = x.ptab; pa.lay = x.lay; pa.n_expert = (int) g_->n_expert; pa.k = (int) ss_->k;
        pa.n_embd = (int) g_->n_embd; pa.n_ff = (int) basalt::kernels::cpu::expert_layout().fmt[0].n_ff;
        pa.steps = (int) dm_->rings; pa.scratch = x.pscr; pa.arrive = x.parr; pa.diag_m = x.h_w + 32;
        pa.rq = x.rq; pa.rs = x.rs;
        if (!basalt::kernels::dec_moe_zero_words(x.parr, 16, x.ps) ||
            !basalt::kernels::dec_moe_peer_launch(pa, x.pblocks, x.ps, dm_->pq2)) {
            err = "verify: a further peer's window did not launch";
            return false;
        }
        x.live = true;
    }
    if (gm) gm->mark(14);
    return true;
}

// The peer window ends with CUDA0's last ring.
bool Verifier::dm_end(std::string& err) {
    if (dm_ == nullptr) return true;
    for (DecMoe::XPeer& x : dm_->xp) {
        if (!x.live) continue;
        const OnDevice on(x.dev);
        x.live = false;
        const cudaError_t pe = cudaStreamSynchronize(x.ps);
        if (pe != cudaSuccess) { err = std::string("verify: a further peer's window: ") + cudaGetErrorString(pe); return false; }
        const std::string ab = basalt::kernels::dec_moe_peer_diag(x.h_w + 32);
        if (!ab.empty()) { err = "verify: a further peer's window " + ab; return false; }
    }
    if (!dm_->peer_live) return true;
    const OnDevice on(dm_->pdev);
    dm_->peer_live = false;
    const cudaError_t pe = cudaStreamSynchronize(dm_->ps);
    if (pe != cudaSuccess) { err = std::string("verify: the peer window: ") + cudaGetErrorString(pe); return false; }
    const std::string ab = basalt::kernels::dec_moe_peer_diag(dm_->h_w + 32);
    if (!ab.empty()) { err = "verify: the peer window " + ab; return false; }
    return true;
}

bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err) {
    using namespace basalt::kernels;
    const OnDevice on_device(device_);
    last_batch_ = false;
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU; restart the engine"; return false; }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    refresh_ar();
    if (!capture(T, err) || !capture_commit(err)) return false;
    DecodeGaps& gm = decode_gaps();
    gm.mark(11);
    const Clock::time_point t0 = Clock::now();
    decode_events().record(DecodeEvents::kRunEntry, cs_, 1);
    if (!staged_) stage_inputs(T, tokens, pos0);
    staged_ = false;
    const bool do_ple = ss.ple.ready() && ple_stage();
    uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];
    if (do_ple) {
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, ple_rows + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
            ss.ple.table->prefetch_rows(ple_rows + t * PLE_N_HEADS);
        }
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    gm.mark(12);
    gs_split_ = dm_ != nullptr && gs_mixl_ != nullptr;
    if (gs_split_ && !gs_live_ && !gs_move(true, err)) return false;
    if (dm_ != nullptr) {   // this window's residency, then the peer window ahead of CUDA0's
        if (!dm_begin(err, &gm)) return false;
    }
    ms_host += ms_since(t0);
    if (prof_on_) decode_gaps().verify_end = 0;   // set again only by a collected window
    const Clock::time_point tl = Clock::now();
    DecodeEvents& dev = decode_events();
    dev.stamp(5);
    if (next_ == nullptr && dev.on) { dev.next_window(); dev.record(DecodeEvents::kVerifyPre, cs_); }
    if (ar_on() && h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
    // an async commit still running: the window's first node waits for it once the commit's begin kernel has run (the
    // device then knows the epoch); if that does not happen within moments, the window follows the commit on the device
    if (!finisher_on() || (commit_pending_ && !commit_begun()))
        order_after_commit(true);
    const cudaError_t le = cudaGraphLaunch(ar_off_ ? exec_nr_[T] : exec_[T], cs_);
    if (next_ == nullptr) dev.record(DecodeEvents::kVerifyPost, cs_);
    ms_launch += ms_since(tl);
    if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
    lz_pending_ = false;   // the window applies the lazy layers (lazy_on) before their GDN kernels
    // a sampled or penalized request's head sampling queued behind the window now, not after a sync
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    const bool biased = sampling_.logit_bias != nullptr;
    const bool early_sample = chain_on() && le_ >= g.n_layers && head_sampling_ && (sampled || hist_d_ != nullptr || biased);
    // the draft probe: row t asks for the probability of the window's token t + 1
    const bool probe = draft_probe_ && head_sampling_ && sampled && T > 1 && le_ >= g.n_layers;
    if (probe)
        for (int t = 0; t + 1 < T; ++t) h_q_[t] = tokens[t + 1];
    if (early_sample) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        sample_tokens_probe(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, probe ? m_q_ : nullptr,
                            probe ? T - 1 : 0, probe ? m_qp_ : nullptr, cs_);
    }
    (void) cudaStreamQuery(cs_);
    if (post_launch_) { auto f = std::move(post_launch_); post_launch_ = nullptr; f(); }
    gm.mark(15);
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int64_t steps = le_ - lb_;
    if (ar_on()) {
        if (do_ple) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            *flag = 1;
            *(volatile uint32_t*) (h_flag_ + 8) = 1;
            ms_host += ms_since(tp);
        } else {
            *flag = 1;
            *(volatile uint32_t*) (h_flag_ + 8) = 1;
        }
    } else {
    // the side branch of layer 1's PLE projections waits for the rows alone
    const bool ple_early = ple_side_ && do_ple;
    if (ple_early) {
        const Clock::time_point tp = Clock::now();
        if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        *(volatile uint32_t*) (h_flag_ + 8) = 1;
        ms_host += ms_since(tp);
    }
    for (int64_t k = 0; k < steps; ++k) {
        const int64_t l = lb_ + k;
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify window: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) {
                // the caller ends the engine; no spin kernel may outlive it
                const std::string words = dm_note();   // read before the release overwrites them
                const bool drained = release_gpu_waits(5000);
                err = "verify: timed out at layer " + std::to_string(l) + words + released_note(drained) + peer_released();
                return false;
            }
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot();
        const int n = T;
        progress_at("verify window: the CPU experts of layer", l);
        if (remote_opt_) remote_opt_->begin(h_w_, 0, n);
        if (dm_ != nullptr) {   // the ring's own rows and tiers: the pool computes the CPU's entries only
            const basalt::kernels::DmRing& R = dm_->h_ring[k];
            sink_.ext_kind = R.kind;
            if (pool != nullptr)
                pool(user, dm_->h_x + (size_t) k * dm_->x_stride, R.ids, n, ss.k, h_ymiss_, l);
            sink_.ext_kind = nullptr;
        } else if (pool != nullptr)
            pool(user, h_x_, h_ids_, n, ss.k,
                 h_ymiss_, l);
        if (remote_opt_) remote_opt_->end();
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        // Layer 1's pre(1, 0) copies h_ple_ -> ple_ after Layer 0's wait_flag_ge(m_flag_, 1).
        // By collecting PLE here at k == 0 (after publishing flagA/flagB for Layer 0 so the GPU can run
        // Layer 0's VRAM experts, and before raising *flag = 1), the NVMe PLE reads overlap with both
        // MtpDrafter::draft and Layer 0's attention + router + expert execution!
        if (k == 0 && do_ple && !ple_early) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            ms_host += ms_since(tp);
        }
        *flag = want;
        if (dm_ != nullptr && dm_->warm != nullptr && l + 1 < le_) {   // the next layer's warm-up
            CpuWarm& w = *dm_->warm;
            warm_pool_->idle_stop();
            w.layer = (int) (l + 1);
            w.n_tok = std::min(n, (int) kVerifyMaxT);
            w.xq = dm_->h_xq + (size_t) k * dm_->xq_stride;
            w.res0 = dm_->shadow0.data() + (size_t) (l + 1) * g.n_expert;
            w.res1 = dm_->pdev >= 0 ? dm_->shadow1.data() + (size_t) (l + 1) * g.n_expert : nullptr;
            w.arrive.store(0, std::memory_order_relaxed);
            w.ready.store(0, std::memory_order_relaxed);
            w.blob = nullptr;
            warm_pool_->idle_start(&warm_job, &w);
        }
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    if (dm_ != nullptr && dm_->warm != nullptr) warm_pool_->idle_stop();
    }
    // each stage stages its own inputs (the layer split's order)
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    // a window the GPU never finishes (a spin kernel that never sees its flag) holds the host here; the stall watchdog
    // then releases every verifier's GPU waits (release_live_verifiers) before it ends the engine, so no spin kernel
    // outlives the process - which could leave the GPU in a lost state.  The wait itself stays a blocking sync: polling
    // with cudaStreamQuery here would cost decode time (a core calling the driver beside the expert workers).
    const cudaError_t se = cudaStreamSynchronize(cs_);
    gm.mark(-1);
    if (!dm_end(err)) return false;
    if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
    if (ar_on() && h_plan_err_ != nullptr && *(volatile uint32_t*) h_plan_err_ != 0) {   // the table changed under the window
        *(volatile uint32_t*) h_plan_err_ = 0;
        err = "verify: the zero-doorbell plan met an expert that is not in VRAM (the residency table changed during the window)";
        return false;
    }
    constexpr int64_t wstat_every = 200;   // windows per report
    if (dm_ != nullptr && dm_->wstat != nullptr && (dm_->wstat_t += T, ++dm_->wstat_windows == wstat_every)) {
        // --wait-stats: the last wstat_every windows' counters
        std::vector<unsigned long long> w((size_t) kDmStatWords);
        cudaMemcpy(w.data(), dm_->wstat, w.size() * 8, cudaMemcpyDeviceToHost);
        cudaMemset(dm_->wstat, 0, w.size() * 8);
        const double nw = (double) dm_->wstat_windows;
        basalt::log::raw("basalt verify: dm_wait over %lld windows (avg T %.2f):\n%s", (long long) dm_->wstat_windows,
                     dm_->wstat_t / nw, basalt::kernels::dec_moe_stats_text(w.data(), nw).c_str());
        dm_->wstat_windows = 0;
        dm_->wstat_t = 0;
    }
    // the window does not retire the commit: whoever reads the session from the host or another stream calls wait_commit(),
    // and a finished event is noticed here at no cost
    if (commit_pending_ && cudaEventQuery(commit_done_) == cudaSuccess) commit_pending_ = false;
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    gm.mark(0);
    if (prof_on_) collect_profile();   // the window's GPU stage stamps
    gm.mark(1);
    // ---- a sampled or penalized request: the head's sampling again, host-side so its parameters are this call's
    // own (a captured kernel would replay the same draws forever).  Row t's draw is Philox(seed, pos0 + t): tied to
    // the POSITION it samples, not to how the text was cut into windows, so a seed replays the same text whatever
    // the drafts were. Exact: a rejected row's draw is discarded, and no kept decision depends on a reused draw.
    if (le_ < g.n_layers) {   // a layer split's earlier stage: the hand-off is written (synced above)
        ++windows;
        return next_ == nullptr || next_->run(T, tokens, pos0, pool, next_user_, out, err);
    }
    if (head_sampling_ && (sampled || hist_d_ != nullptr || biased) && !early_sample) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        sample_tokens_probe(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, probe ? m_q_ : nullptr,
                            probe ? T - 1 : 0, probe ? m_qp_ : nullptr, cs_);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {   // m_out_ is the mapped h_out_: synced, it is readable
            err = "verify: the head sampling failed";
            return false;
        }
    }
    gm.mark(2);
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    gm.mark(3);
    ++windows;
    progress_at("decode");
    progress_beat();
    return true;
}

void Verifier::set_plan_slot() {
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_;
    const int64_t i32 = 4 + (cap + 1) + cap + cap;
    const int64_t ptr_off = (i32 + 1) & ~1ll;
    sink_.counts = base;
    sink_.start = base + 4;
    sink_.dst = sink_.start + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr = (unsigned long long*) (base + ptr_off);
    sink_.ptr2 = sink_.ptr + cap;
    sink_.start2 = base + ptr_off + 4 * cap;
    sink_.staging_cap = kStagingBlobs;
}

// Flag B only rises.
void Verifier::raise_flag(uint32_t* flag, uint32_t value) {
    std::atomic_ref<uint32_t> f(*flag);
    uint32_t cur = f.load();
    while (cur < value && !f.compare_exchange_strong(cur, value)) {}
}

// The PCIe groups are planned: raise flag B for the layer (the graph waits for it before staging them).
void Verifier::raise_flag_b(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    raise_flag(v->h_flagB_, v->cur_layer_ + 1);
}

void Verifier::publish_plan(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    _mm_sfence();
    *(volatile uint32_t*) v->h_flagA_ = v->cur_layer_ + 1;
}

void Verifier::set_commit_async(bool on) { g_commit_async = on; }

bool Verifier::chain_on() { return true; }

bool Verifier::commit(int n_keep, std::string& err, int parts) {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    if (parts & 1) {
        h_commit_[0] = n_keep;
        h_commit_[1] = n_keep - 1;
        for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
        if (++commit_epoch_ == 0) ++commit_epoch_;
        h_commit_[2 + max_t_] = (int32_t) commit_epoch_;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        // after the window that produced the rows, on the commit's own stream
        cudaEventRecord(ev_commit_go_, cs_);
        cudaStreamWaitEvent(cm_, ev_commit_go_, 0);
        decode_events().record(DecodeEvents::kCommitPre, cm_);
        const cudaError_t le = cudaGraphLaunch(commit_exec_, cm_);
        decode_events().record(DecodeEvents::kCommitPost, cm_);
        if (gs_split_ && gs_live_) {   // the peer's heads, on its stream behind its window
            const OnDevice on(gs_dev_);
            gs_ca_.n_keep = h_commit_;
            basalt::kernels::gdn_commit_layers(gs_ca_, dm_->ps, kCommitBlocks);
        }
        if (le != cudaSuccess) { err = std::string("verify: commit launch: ") + cudaGetErrorString(le); return false; }
        // set_commit_async: no wait here - the next window's first node waits for the epoch (the host names it in
        // h_pend_), and the drafter (its own stream) reads only this window's final rows and its own K/V.  h_commit_ is
        // next written after the next window's results are read, i.e. after this graph has run.  Everything else waits
        // on commit_done_ (wait_commit).
        if (!g_commit_async || next_ != nullptr) {
            const cudaError_t se = cudaStreamSynchronize(cm_);
            if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
        } else {
            const cudaError_t re = cudaEventRecord(commit_done_, cm_);
            if (re != cudaSuccess) { err = std::string("verify: commit event: ") + cudaGetErrorString(re); return false; }
            (void) cudaStreamQuery(cm_);
            commit_pending_ = true;
            lz_pending_ = lazy_on();   // its later GDN layers: the next window, or apply_lazy
        }
        decode_gaps().mark(4);
    }
    if ((parts & 2) && ple_stage())   // stages that share one session must advance it once
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit(n_keep, err, parts);
}

bool Verifier::wait_commit(std::string& err) {
    if (commit_pending_ || lz_pending_) {
        const OnDevice on_device(device_);
        apply_lazy();
        commit_pending_ = false;
        const cudaError_t se = cudaEventSynchronize(commit_done_);
        if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    }
    if (gs_live_ && !gs_move(false, err)) return false;
    return next_ == nullptr || next_->wait_commit(err);
}


// ================================ BATCH WINDOWS (see init_slots) ================================

bool Verifier::init_slots(const std::vector<SessionState*>& slots, std::string& err) {
    const OnDevice on_device(device_);
    if (g_ == nullptr || ss_ == nullptr) { err = "verify: init_slots before init"; return false; }
    if (slots.empty() || (int) slots.size() > max_t_) {
        err = "verify: init_slots needs 1.." + std::to_string(max_t_) + " sessions";
        return false;
    }
    for (SessionState* x : slots) {
        if (x == nullptr || x->layer_lo != ss_->layer_lo || x->layer_hi != ss_->layer_hi ||
            x->max_cells > ss_->max_cells || x->gdn_ord0 != ss_->gdn_ord0 || x->qsa_ord0 != ss_->qsa_ord0) {
            err = "verify: a slot's session is not carved like the stage's own (layer range, a context at most the stage's)";
            return false;
        }
    }
    const basalt::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t S = (int64_t) slots.size(), CB = 2 + max_t_;
    const int64_t TS = (s.idx_block - 1) * g_->idx_key_dim, nQ = g_->n_qsa_layers();
    void* d = nullptr;
    if (!mapped((size_t) (S * CB * 4 + 16), (void**) &h_commitb_, (void**) &m_commitb_)) {
        err = "verify: the batch commit staging failed";
        return false;
    }
    if (!mapped((size_t) max_t_ * sizeof(basalt::kernels::SegRow), (void**) &h_segb_, (void**) &m_segb_)) {
        err = "verify: the batch segment staging failed";
        return false;
    }
    using namespace basalt::kernels;
    if (!mapped((size_t) (basalt::kernels::kQsaSegMaxRows + 8) * sizeof(QsaSeg), (void**) &h_qseg_, (void**) &m_qseg_)) {
        err = "verify: the batch QSA table staging failed";
        return false;
    }
    std::memset(h_qseg_, 0, (size_t) (basalt::kernels::kQsaSegMaxRows + 8) * sizeof(QsaSeg));
    // the QSA chain's segment kernels need resident, unrotated INT8 K/V in every slot
    if (!(qsa_dec_fused_supported(s))) {
        err = "verify: batch windows need the native QSA kernels";
        return false;
    }
    for (int64_t l = lb_; l < le_; ++l) {
        if (!is_qsa_layer(*g_, l)) continue;
        int64_t qi = 0;
        for (int64_t m = 0; m < l; ++m) qi += is_qsa_layer(*g_, m) ? 1 : 0;
        for (SessionState* x : slots) {
            const QsaState& st = x->qsa_states[qi];
            if (st.kv_mode != 0 || !st.kv_int8 || st.kv_rot || st.kv_hybrid || st.kv_q4) {
                err = "verify: batch windows need resident INT8 K/V without rotation (--kv int8)";
                return false;
            }
        }
    }
    const uint64_t a = ((uint64_t) S * CB * 4 + 255) & ~255ull;
    const uint64_t b = (a + (uint64_t) S * std::max<int64_t>(nQ, 1) * TS * 4 + 255) & ~255ull;
    const uint64_t b2 = (b + (uint64_t) max_t_ * sizeof(basalt::kernels::SegRow) + 255) & ~255ull;
    const uint64_t b3 = (b2 + (uint64_t) (basalt::kernels::kQsaSegMaxRows + 8) * sizeof(QsaSeg) + 255) & ~255ull;
    const uint64_t HSb = (uint64_t) NG_HIST * NG_HC_DIM * sizeof(float);
    const uint64_t b4 = (b3 + (uint64_t) S * sizeof(float*) + 255) & ~255ull;
    const uint64_t b5 = (b4 + (uint64_t) max_t_ * HSb + 255) & ~255ull;
    if (cudaMalloc(&d, b5 + (uint64_t) std::max<int64_t>(nQ, 1) * S * sizeof(QsaSegSlot)) != cudaSuccess) {
        err = "verify: the batch buffers do not fit";
        return false;
    }
    arena_b_ = d;
    commitb_ = (int32_t*) d;
    tail_snap_b_ = (float*) ((uint8_t*) d + a);
    segb_ = (basalt::kernels::SegRow*) ((uint8_t*) d + b);
    d_qrows_ = (QsaSeg*) ((uint8_t*) d + b2);
    d_qsegs_ = d_qrows_ + basalt::kernels::kQsaSegMaxRows;
    d_ple_slots_ = (float**) ((uint8_t*) d + b3);
    ple_work_ = (float*) ((uint8_t*) d + b4);
    {   // static per-slot pointers: the PLE histories and, per QSA layer, the slots' caches and indexer state
        std::vector<float*> ph((size_t) S);
        for (int64_t i = 0; i < S; ++i) ph[(size_t) i] = slots[(size_t) i]->ple_hist;
        cudaMemcpy(d_ple_slots_, ph.data(), (size_t) S * sizeof(float*), cudaMemcpyHostToDevice);
        d_qslots_.assign((size_t) nQ, nullptr);
        QsaSegSlot* dev = (QsaSegSlot*) ((uint8_t*) d + b5);
        for (int64_t l = lb_; l < le_; ++l) {
            if (!is_qsa_layer(*g_, l)) continue;
            int64_t qi = 0;
            for (int64_t m = 0; m < l; ++m) qi += is_qsa_layer(*g_, m) ? 1 : 0;
            std::vector<QsaSegSlot> hs((size_t) S);
            for (int64_t i = 0; i < S; ++i) {
                const QsaState& st = slots[(size_t) i]->qsa_states[qi];
                QsaSegSlot& e = hs[(size_t) i];
                e.k_pool = st.k_pool; e.v_pool = st.v_pool; e.k_q = st.k_q; e.v_q = st.v_q;
                e.k_scale = st.k_scale; e.v_scale = st.v_scale; e.page_table = st.page_table; e.host = st.host;
                e.tail = st.idx_tail; e.dead = st.idx_dead; e.pooled = st.idx_pooled;
                e.tail_snap = tail_snap_b_ + ((size_t) i * nQ + qi) * TS;
                e.block_pos = st.idx_block_pos; e.max_cells = (int32_t) st.max_cells;
            }
            cudaMemcpy(dev, hs.data(), (size_t) S * sizeof(QsaSegSlot), cudaMemcpyHostToDevice);
            d_qslots_[(size_t) qi] = dev;
            dev += S;
        }
    }
    slots_ = slots;
    batch_dm_ = dm_ != nullptr;
    slot_sp_.assign(slots.size(), sampling_);   // greedy until set_slot_sampling
    basalt::log::info("basalt verify", "batch windows of up to %lld sequences (layers [%lld, %lld))", (long long) S,
                 (long long) lb_, (long long) le_);
    return true;
}

bool Verifier::capture_batch(int R, int hbase, std::string& err) {
    const uint64_t key = bkey(R, hbase);
    cudaGraphExec_t& ex = exec_bm_[key];
    if (ex != nullptr) return true;
    const bool pdl_try = basalt::kernels::pdl_requested() && !pdl_off_;
    // a window that does not capture or instantiate with programmatic dependent launch is captured again without it
    auto retry_plain = [&](const std::string& why) {
        basalt::log::warn("basalt verify", "the %d-row batch window failed with programmatic dependent launch (%s): "
                             "capturing it without", R, why.c_str());
        (void) cudaGetLastError();
        pdl_off_ = true;
        ex = nullptr;
        return capture_batch(R, hbase, err);
    };
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin batch capture failed";
        return false;
    }
    batch_rec_ = true;
    pdl_rec_ = pdl_try;
    row_base_ = hbase;
    std::string rerr;
    const bool ok = record_window(R, cs_, rerr);
    batch_rec_ = false;
    pdl_rec_ = false;
    row_base_ = 0;
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok || ce != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        if (pdl_try) return retry_plain(!ok ? rerr : cudaGetErrorString(ce));
        err = !ok ? rerr : std::string("verify: end batch capture: ") + cudaGetErrorString(ce);
        return false;
    }
    if (!instantiate_batch(ex, graph, err, true)) {
        if (pdl_try) return retry_plain(err);
        return false;
    }
    basalt::log::info("basalt verify", "captured the batch window for %d rows", R);
    return true;
}

bool Verifier::bind_session(SessionState* s, std::string& err) {
    if (s == nullptr || ss_ == nullptr) { err = "verify: bind_session without a session"; return false; }
    if (s == ss_) return true;
    if (!wait_commit(err)) return false;   // the last commit still works on the old session's state
    SoloGraphs cur;
    std::copy(std::begin(exec_), std::end(exec_), std::begin(cur.exec));
    std::copy(std::begin(exec_nr_), std::end(exec_nr_), std::begin(cur.exec_nr));
    cur.commit = commit_exec_;
    solo_sets_[ss_] = cur;
    SoloGraphs nx;
    const auto it = solo_sets_.find(s);
    if (it != solo_sets_.end()) {
        nx = it->second;
        solo_sets_.erase(it);
    }
    std::copy(std::begin(nx.exec), std::end(nx.exec), std::begin(exec_));
    std::copy(std::begin(nx.exec_nr), std::end(nx.exec_nr), std::begin(exec_nr_));
    commit_exec_ = nx.commit;
    ss_ = s;
    return true;
}

bool Verifier::warm_session(SessionState* s, std::string& err) {
    SessionState* home = ss_;
    if (!bind_session(s, err)) return false;
    refresh_ar();
    bool ok = capture_commit(err);
    for (int T = 1; ok && T <= max_t_; ++T) ok = capture(T, err);
    std::string e2;
    if (!bind_session(home, e2) && ok) { err = e2; return false; }
    return ok;
}

bool Verifier::warm_graphs(std::string& err) {
    refresh_ar();
    for (int T = 1; T <= max_t_; ++T)
        if (!capture(T, err)) return false;
    if (slots_.empty()) return true;
    for (int R = 1; R <= max_t_; ++R)
        if (!capture_batch(R, 0, err) || !capture_commit_batch(R, err)) return false;
    return true;
}

bool Verifier::instantiate_batch(cudaGraphExec_t& ex, cudaGraph_t graph, std::string& err, bool priority) {
    const cudaError_t ie = priority ? cudaGraphInstantiateWithFlags(&ex, graph, cudaGraphInstantiateFlagUseNodePriority)
                                    : cudaGraphInstantiate(&ex, graph, 0);
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess) { ex = nullptr; err = std::string("verify: batch instantiate: ") + cudaGetErrorString(ie); return false; }
    cudaGraphUpload(ex, cs_);
    cudaStreamSynchronize(cs_);
    return true;
}

bool Verifier::capture_commit_batch(int R, std::string& err) {
    cudaGraphExec_t& cex = commit_bm_[R];
    if (cex != nullptr) return true;
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const int64_t CB = 2 + MT;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin batch commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commitb_, m_commitb_, (int64_t) slots_.size() * CB, cs_);
        copy_i32_from_mapped((int32_t*) segb_, (const int32_t*) m_segb_, (int64_t) R * 8, cs_);
        copy_i32_from_mapped((int32_t*) d_qrows_, (const int32_t*) m_qseg_, (int64_t) (kQsaSegMaxRows + 8) * 4, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        {   // every segment's GDN layers in one launch (the keep counts come from the table)
            GdnCommitSeg ga;
            ga.rows = segb_; ga.state_off = (size_t) gdn_index * gdn_floats; ga.state_stride = gdn_floats;
            ga.h = h_L_ + (size_t) gdn_index * MT * C; ga.qkv = qkv_L_ + (size_t) gdn_index * MT * C;
            ga.gate = gate_L_ + (size_t) gdn_index * MT * HV; ga.beta = beta_L_ + (size_t) gdn_index * MT * HV;
            ga.h_stride = (size_t) MT * C; ga.gb_stride = (size_t) MT * HV;
            ga.C = (int) C; ga.h_k = (int) g.ssm_k_heads; ga.h_v = (int) HV; ga.n_rows = R;
            for (int64_t l = lb_; l < le_; ++l) ga.n_layers += is_qsa_layer(g, l) ? 0 : 1;
            gdn_commit_layers_seg(ga, cs_, kCommitBlocks);
        }
        {   // every QSA layer's indexer tail restore and appends, per segment, in one launch
            QsaSegCommit qc;
            qc.segs = d_qsegs_; qc.pos = commitb_ + 2; qc.pos_stride = (int) CB; qc.epsilon = EPS; qc.n_seg = R;
            for (int64_t l = lb_; l < le_ && ok; ++l) {
                if (!is_qsa_layer(g, l)) continue;
                const LayerView v(*wt_, l);
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn || qc.n_layers >= kQsaCommitMaxLayers) { ok = false; break; }
                QsaSegCommitLayer& q = qc.layer[qc.n_layers++];
                q.slots = d_qslots_[(size_t) qsa_index];
                q.raw = idx_raw_L_ + (size_t) (qsa_index * MT) * ID;
                q.gamma = (const float*) wikn->data;
                ++qsa_index;
            }
            if (ok) native_qsa_indexer_commit_seg(qc, s, rope_scaling(), cs_);
        }
        if (ok && ss_->ple.ready() && ple_stage()) ple_hist_scatter(d_ple_slots_, hist_snap_, segb_, HS, R, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify batch commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: batch commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    return instantiate_batch(cex, graph, err);
}

bool Verifier::stage_batch(const int* slots, const int* lens, int S, int hbase, const int32_t* tokens,
                           const int64_t* pos0, std::string& err) {
    using namespace basalt::kernels;
    int R = 0;
    for (int s = 0; s < S && s < 9; ++s) R += lens != nullptr ? lens[s] : 1;
    if (S < 1 || S > max_t_ || R < S || R > max_t_ || hbase < 0 || (hbase > 0 && hbase + R > (int) slots_.size())) {
        err = "verify: batch rows out of range (init_slots)";
        return false;
    }
    for (int s = 0; s < S; ++s) {
        bool dup = false;
        for (int u = 0; u < s; ++u) dup = dup || slots[u] == slots[s];
        if (slots[s] < 0 || slots[s] >= (int) slots_.size() || dup || (lens != nullptr && lens[s] < 1)) {
            err = "verify: a batch segment's slot is out of range or twice, or its length is empty";
            return false;
        }
    }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU; restart the engine"; return false; }
    const ModelGeometry& g = *g_;
    for (int s = 0; s < S; ++s)
        if (pos0[s] < 0 || pos0[s] + (lens != nullptr ? lens[s] : 1) > slots_[(size_t) slots[s]]->max_cells) {
            err = "verify: slot " + std::to_string(slots[s]) + " runs past its context";
            return false;
        }
    refresh_ar();
    if (!capture_batch(R, hbase, err) || !capture_commit_batch(R, err)) return false;
    if (R > S && !batch_seg_ple_) {
        err = "verify: batch segments of several rows need the PLE window kernels";
        return false;
    }
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    const int64_t CB = 2 + max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const bool do_ple = ss_->ple.ready() && ple_stage();
    uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];   // (not `rows`: that is the slots of the window's rows)
    int32_t* const pk = h_pos_ + (size_t) max_t_ * g.n_head;
    int32_t* const pi = pk + (size_t) max_t_ * g.n_head_kv;
    for (int t = 0; t < kQsaSegMaxRows + 8; ++t) h_qseg_[t] = QsaSeg{-1, 0, 0, 0};   // unused until set
    for (int sg = 0, r0 = 0; sg < S; ++sg) {
        const int n = lens != nullptr ? lens[sg] : 1;
        const SessionState& sx = *slots_[(size_t) slots[sg]];
        int32_t prev[2] = {sx.ple_prev[0], sx.ple_prev[1]};
        h_qseg_[kQsaSegMaxRows + sg] = QsaSeg{slots[sg], r0, n, 0};
        int32_t* c = h_commitb_ + (size_t) slots[sg] * CB;   // the commit's rows, all kept until commit_slots says less
        c[0] = n;
        c[1] = n - 1;
        for (int64_t j = 0; j < CB - 2; ++j) c[2 + j] = j < n ? (int32_t) (pos0[sg] + j) : -1;
        for (int j = 0; j < n; ++j) {
            const int t = r0 + j;
            const int64_t pos = pos0[sg] + j;
            h_tok_[t] = tokens[t];
            qsa_step_fill(h_step_ + t * kStepCount, pos, s);
            for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) pos;
            for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) pos;
            for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) pos;
            if (do_ple) {
                ngram_rows(&tokens[t], prev, 1, ss_->ple.consts, ple_rows + t * PLE_N_HEADS);
                prev[0] = prev[1];
                prev[1] = tokens[t];
            }
            h_qseg_[t] = QsaSeg{slots[sg], r0, n, 0};
            SegRow& e = h_segb_[t];   // the state of layer 0 of the slot's session
            e.gdn = (float*) ((uintptr_t) sx.gdn_state - (uintptr_t) sx.gdn_ord0 * gdn_floats * sizeof(float));
            e.row0 = r0; e.j = j; e.n = n; e.n_keep = n; e.slot = slots[sg]; e.pos = (int32_t) pos;
            last_rows_[t] = slots[sg];
            last_tokens_[t] = tokens[t];
            last_pos_b_[t] = pos;
        }
        last_seg_slot_[sg] = slots[sg];
        last_seg_len_[sg] = n;
        r0 += n;
    }
    if (do_ple && !ss_->ple.table->gather_batch(ple_rows, (size_t) R, h_ple_, err)) return false;
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = R;
    last_nseg_ = S;
    row_base_ = hbase;   // the graphs' key (and nothing else) reads it until the next stage_batch
    last_batch_ = true;
    ms_host += ms_since(t0);
    return true;
}

bool Verifier::run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                         std::string& err) {
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = t;
    return run_slot_rows(rows, S, tokens, pos, pool, user, out, err);
}

bool Verifier::run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool,
                             void* user, int32_t* out, std::string& err) {
    return run_slot_segs(rows, nullptr, S, tokens, pos, pool, user, out, err);
}

bool Verifier::run_slot_segs(const int* rows, const int* lens, int S, const int32_t* tokens, const int64_t* pos,
                             PoolMultiFn pool, void* user, int32_t* out, std::string& err) {
    using namespace basalt::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    if (gs_live_ && !gs_move(false, err)) return false;
    gs_split_ = false;
    if (!stage_batch(rows, lens, S, 0, tokens, pos, err)) return false;
    const int R = last_t_;   // the window's rows
    const bool dm = dm_ != nullptr && batch_dm_;
    if (dm && !dm_begin(err, nullptr)) return false;
    order_after_commit();
    const cudaError_t le = cudaGraphLaunch(exec_bm_[bkey(R, 0)], cs_);
    if (le != cudaSuccess) { err = std::string("verify: batch launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int64_t steps = le_ - lb_;
    for (int64_t k = 0; k < steps; ++k) {
        const int64_t l = lb_ + k;
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify batch: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify batch: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) {
                err = "verify batch: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return false;
            }
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot();
        progress_at("verify batch: the CPU experts of layer", l);
        if (dm) {   // the ring's own rows and tiers: the pool computes the CPU's entries only
            const basalt::kernels::DmRing& Rg = dm_->h_ring[k];
            sink_.ext_kind = Rg.kind;
            if (pool != nullptr) pool(user, dm_->h_x + (size_t) k * dm_->x_stride, Rg.ids, R, ss_->k, h_ymiss_, l);
            sink_.ext_kind = nullptr;
        } else if (pool != nullptr) pool(user, h_x_, h_ids_, R, ss_->k, h_ymiss_, l);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *flag = want;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (dm && !dm_end(err)) return false;
    if (se != cudaSuccess) { err = std::string("verify batch: ") + cudaGetErrorString(se); return false; }
    if (prof_on_) collect_profile();
    ++windows;
    if (le_ < g.n_layers) return next_ == nullptr || next_->run_slot_segs(rows, lens, S, tokens, pos, pool, next_user_, out, err);
    if (!sample_rows(R, err)) return false;
    for (int t = 0; t < R; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    progress_at("decode");
    progress_beat();
    return true;
}

bool Verifier::commit_slots(std::string& err) { return commit_slots(nullptr, err); }

bool Verifier::commit_slots(const int* n_keep, std::string& err) {
    return commit_slots_begin(n_keep, err) && commit_slots_end(err);
}

bool Verifier::commit_slots_begin(const int* n_keep, std::string& err) {
    const OnDevice on_device(device_);
    if (!last_batch_ || last_t_ < 1) { err = "verify: commit_slots without a batch window"; return false; }
    commit_t0_ = Clock::now();
    const int64_t CB = 2 + max_t_;
    if (n_keep != nullptr) {
        for (int sg = 0; sg < last_nseg_; ++sg)
            if (n_keep[sg] < 1 || n_keep[sg] > last_seg_len_[sg]) { err = "verify: commit count out of range"; return false; }
        for (int sg = 0, r0 = 0; sg < last_nseg_; r0 += last_seg_len_[sg], ++sg) {
            const int k = n_keep[sg], n = last_seg_len_[sg];
            for (int j = 0; j < n; ++j) h_segb_[r0 + j].n_keep = k;
            int32_t* c = h_commitb_ + (size_t) last_seg_slot_[sg] * CB;
            c[0] = k;
            c[1] = k - 1;
            for (int j = k; j < n; ++j) c[2 + j] = -1;
        }
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    order_after_commit();
    const cudaError_t le = cudaGraphLaunch(commit_bm_[last_t_], cs_);
    if (le != cudaSuccess) { err = std::string("verify: batch commit launch: ") + cudaGetErrorString(le); return false; }
    // (the PLE history the commit leaves is known from the tokens: it advances now, the graph runs beside the caller)
    if (ple_stage())
        for (int sg = 0, r0 = 0; sg < last_nseg_; r0 += last_seg_len_[sg], ++sg) {
            SessionState& sx = *slots_[(size_t) last_seg_slot_[sg]];
            const int k = n_keep != nullptr ? n_keep[sg] : last_seg_len_[sg];
            for (int j = 0; j < k; ++j) {
                sx.ple_prev[0] = sx.ple_prev[1];
                sx.ple_prev[1] = last_tokens_[r0 + j];
            }
        }
    commit_next_keep_.assign(n_keep != nullptr ? n_keep : nullptr, n_keep != nullptr ? n_keep + last_nseg_ : nullptr);
    return true;
}

bool Verifier::commit_slots_end(std::string& err) {
    const OnDevice on_device(device_);
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify: batch commit: ") + cudaGetErrorString(se); return false; }
    ms_commit += ms_since(commit_t0_);
    return next_ == nullptr || next_->commit_slots(commit_next_keep_.empty() ? nullptr : commit_next_keep_.data(), err);
}

bool Verifier::sample_rows(int S, std::string& err) {
    bool any = false;
    for (int t = 0; t < S; ++t) {
        basalt::kernels::SamplerParams sp = slot_sp_[(size_t) last_rows_[t]];
        if (sp.greedy || sp.temperature <= 0.0f) continue;
        sp.counter = (uint64_t) last_pos_b_[t];   // Philox(seed, position): the solo window's draw for this position
        sp.penalty_last_n = 0;
        basalt::kernels::sample_tokens(head_logits_ + (size_t) t * (size_t) n_vocab_, 1, (int) n_vocab_, nullptr, 0, sp,
                                       m_out_ + t, cs_);
        any = true;
    }
    if (any && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "verify batch: the row sampling failed"; return false; }
    return true;
}

bool Verifier::batch_launch(int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err) {
    const OnDevice on_device(device_);
    if (b_running_) { err = "verify: batch_launch while this stage is busy"; return false; }
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = base + t;
    if (gs_live_ && !gs_move(false, err)) return false;
    gs_split_ = false;
    if (!stage_batch(rows, nullptr, S, base, tokens, pos, err)) return false;
    order_after_commit();
    cudaError_t le = cudaGraphLaunch(exec_bm_[bkey(S, base)], cs_);
    if (le == cudaSuccess) le = cudaGraphLaunch(commit_bm_[S], cs_);   // right behind it: every row is kept
    if (le != cudaSuccess) { err = std::string("verify: batch launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    if (ple_stage())   // the host's side of the commit (the hash's last two tokens)
        for (int t = 0; t < S; ++t) {
            SessionState& sx = *slots_[(size_t) rows[t]];
            sx.ple_prev[0] = sx.ple_prev[1];
            sx.ple_prev[1] = tokens[t];
        }
    b_running_ = true;
    b_k_ = 0;
    b_steps_ = le_ - lb_;
    b_last_ = Clock::now();
    return true;
}

int Verifier::batch_poll(PoolMultiFn pool, void* user, std::string& err) {
    if (!b_running_) return 1;
    const OnDevice on_device(device_);
    volatile uint32_t* const seq = h_seq_;
    const int S = last_t_;
    while (b_k_ < b_steps_) {
        const uint32_t want = (uint32_t) (b_k_ + 1);
        if (*seq < want) {
            const auto now = Clock::now();
            if (now - b_last_ > std::chrono::milliseconds(2)) {
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify batch: layer " + std::to_string(lb_ + b_k_) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    b_running_ = false;
                    return -1;
                }
                if (now - b_last_ > std::chrono::seconds(20)) {
                    err = "verify batch: timed out at layer " + std::to_string(lb_ + b_k_) + released_note(release_gpu_waits(5000));
                    b_running_ = false;
                    return -1;
                }
            }
            return 0;
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot();
        if (pool != nullptr) pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, lb_ + b_k_);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *(volatile uint32_t*) h_flag_ = want;
        ms_pool += ms_since(b);
        b_last_ = Clock::now();
        ++b_k_;
    }
    const cudaError_t q = cudaStreamQuery(cs_);
    if (q == cudaErrorNotReady) return 0;
    if (q != cudaSuccess) { err = std::string("verify batch: ") + cudaGetErrorString(q); b_running_ = false; return -1; }
    if (prof_on_) collect_profile();
    if (last_stage()) {
        if (!sample_rows(S, err)) { b_running_ = false; return -1; }
        for (int t = 0; t < S; ++t) b_out_[t] = ((volatile int32_t*) h_out_)[t];
    }
    ++windows;
    b_running_ = false;
    progress_beat();
    return 1;
}

bool Verifier::resample_row(int t, int64_t pos0, const std::vector<float>& bias, int32_t& token, std::string& err) {
    if (next_ != nullptr) return next_->resample_row(t, pos0, bias, token, err);   // a layer split: the head is on the last stage
    const OnDevice on_device(device_);
    if (head_logits_ == nullptr || t < 0 || bias.size() != (size_t) n_vocab_) { err = "resample_row: no such row or a bad bias"; return false; }
    const size_t bytes = bias.size() * sizeof(float);
    // on cs_ itself: the stream does not order against a plain cudaMemcpy, so the draw could read the previous bias
    if ((!row_bias_device_ && cudaMalloc((void**) &row_bias_device_, bytes) != cudaSuccess) ||
        cudaMemcpyAsync(row_bias_device_, bias.data(), bytes, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
        err = "resample_row: the bias upload failed";
        return false;
    }
    // a batch window's row draws with its own slot's sampling and position, as the window drew it
    basalt::kernels::SamplerParams sp;
    const int32_t* hist = nullptr;
    int hist_n = 0;
    if (last_batch_ && t < last_t_) {
        sp = slot_sp_[(size_t) last_rows_[t]];
        sp.counter = (uint64_t) last_pos_b_[t];
        sp.penalty_last_n = 0;
    } else {
        sp = sampling_;
        sp.counter = (uint64_t) pos0 + (uint64_t) t;
        if (hist_d_ != nullptr) { hist = hist_d_ + (size_t) t * (size_t) hist_len_; hist_n = (int) hist_len_; }
    }
    sp.logit_bias = row_bias_device_;
    basalt::kernels::sample_tokens(head_logits_ + (size_t) t * (size_t) n_vocab_, 1, (int) n_vocab_, hist, hist_n, sp,
                  m_out_, cs_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "resample_row: the sampling failed"; return false; }
    token = ((volatile int32_t*) h_out_)[0];
    return true;
}

bool Verifier::copy_logits(int t, float* host) const {
    if (next_ != nullptr) return next_->copy_logits(t, host);   // a layer split: the head is on the last stage
    if (head_logits_ == nullptr || host == nullptr || t < 0 || n_vocab_ <= 0) return false;
    return cudaMemcpy(host, head_logits_ + (size_t) t * (size_t) n_vocab_, (size_t) n_vocab_ * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace basalt::core
