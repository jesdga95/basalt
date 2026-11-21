// src/models/inkling2/kernels/ops.cu - Inkling 2's block-residual mixes, loop-controller glue, attention preparation
// (short convolutions, clips, cache writes), output gate, post-sublayer convolutions, expert plan and sum, and softcap.
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

constexpr int kThreads = 256;
constexpr int kPerThread = kWidth / kThreads;   // 9
static_assert(kWidth % kThreads == 0, "the width splits evenly over a block");

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Sums of N values per thread over a block of kThreads, in a fixed order; every thread gets the N results.
template <int N>
__device__ __forceinline__ void block_sums(float (&v)[N], float* red) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = warp_sum(v[i]);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0)
#pragma unroll
        for (int i = 0; i < N; ++i) red[i * (kThreads / 32) + w] = v[i];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; ++i) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < kThreads / 32; ++j) s += red[i * (kThreads / 32) + j];
        v[i] = s;
    }
    __syncthreads();
}

__device__ __forceinline__ float sigmoidf_(float v) { return 1.f / (1.f + expf(-v)); }

// ---- block residuals ---------------------------------------------------------------------------------------------------

template <int NQ>
__global__ void __launch_bounds__(kThreads) attnres_mix_kernel(MixStates st, const float* __restrict__ queries, float eps,
                                                               float* __restrict__ out, int64_t out_stride) {
    __shared__ float red[kMaxMix * (NQ + 1) * (kThreads / 32)];
    const int r = blockIdx.x, n = st.n;
    float v[kMaxMix][kPerThread];
    float part[kMaxMix * (NQ + 1)];
#pragma unroll
    for (int i = 0; i < kMaxMix * (NQ + 1); ++i) part[i] = 0.f;
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i) {
        if (i >= n) break;
        const float* s = st.s[i] + (size_t) r * kWidth;
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            const int c = threadIdx.x + j * kThreads;
            const float x = s[c];
            v[i][j] = x;
            part[i * (NQ + 1)] = fmaf(x, x, part[i * (NQ + 1)]);
#pragma unroll
            for (int q = 0; q < NQ; ++q) part[i * (NQ + 1) + 1 + q] = fmaf(queries[q * kWidth + c], x, part[i * (NQ + 1) + 1 + q]);
        }
    }
    block_sums<kMaxMix * (NQ + 1)>(part, red);
    float inv[kMaxMix];
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i) inv[i] = i < n ? rsqrtf(part[i * (NQ + 1)] / (float) kWidth + eps) : 0.f;
    const float scale = rsqrtf((float) kWidth);
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float sc[kMaxMix], m = -INFINITY;
#pragma unroll
        for (int i = 0; i < kMaxMix; ++i) {
            sc[i] = i < n ? part[i * (NQ + 1) + 1 + q] * inv[i] * scale : -INFINITY;
            m = fmaxf(m, sc[i]);
        }
        float den = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxMix; ++i) {
            sc[i] = i < n ? expf(sc[i] - m) : 0.f;
            den += sc[i];
        }
        float* o = out + (size_t) q * out_stride + (size_t) r * kWidth;
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            float acc = 0.f;
#pragma unroll
            for (int i = 0; i < kMaxMix; ++i)
                if (i < n) acc = fmaf(sc[i] / den, v[i][j] * inv[i], acc);
            o[threadIdx.x + j * kThreads] = acc;
        }
    }
}

// ---- loop controllers --------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads) ctrl_pair_norm_kernel(const float* __restrict__ dest, const float* __restrict__ msg,
                                                                  float* __restrict__ pair) {
    __shared__ float red[2 * (kThreads / 32)];
    const int r = blockIdx.x;
    float a[kPerThread], b[kPerThread], ss[2] = {0.f, 0.f};
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int c = threadIdx.x + j * kThreads;
        a[j] = dest[(size_t) r * kWidth + c];
        b[j] = msg[(size_t) r * kWidth + c];
        ss[0] = fmaf(a[j], a[j], ss[0]);
        ss[1] = fmaf(b[j], b[j], ss[1]);
    }
    block_sums<2>(ss, red);
    const float ia = rsqrtf(ss[0] / (float) kWidth + 1e-6f), ib = rsqrtf(ss[1] / (float) kWidth + 1e-6f);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int c = threadIdx.x + j * kThreads;
        pair[(size_t) r * 2 * kWidth + c] = a[j] * ia;
        pair[(size_t) r * 2 * kWidth + kWidth + c] = b[j] * ib;
    }
}

__global__ void ctrl_blend_kernel(const float* __restrict__ dest, const float* __restrict__ msg, const float* __restrict__ gf,
                                  int n, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int r = i / kWidth, c = i - r * kWidth;
    const float g = tanhf(gf[(size_t) r * 2 * kWidth + c]), f = tanhf(gf[(size_t) r * 2 * kWidth + kWidth + c]);
    out[i] = (1.f - f) * dest[i] + g * msg[i];
}

// ---- attention preparation ---------------------------------------------------------------------------------------------

__device__ __forceinline__ float tau_of(int p, float floor_, float alpha) {
    return 1.f + alpha * logf(fmaxf((float) (p + 1) / floor_, 1.f));
}

// The convolution input of channel c at position pos[r] - j: a row of the window, the ring, or zero before position 0.
__device__ __forceinline__ float conv_in(const float* __restrict__ y, int ld, int off, const float* __restrict__ ring,
                                         int ring_w, const int32_t* __restrict__ pos, int r, int j, int c) {
    if (r - j >= 0) return y[(size_t) (r - j) * ld + off + c];
    const int p = pos[r] - j;
    if (p < 0) return 0.f;
    return ring[(size_t) (p % kConvRing) * ring_w + c];
}

__device__ __forceinline__ void store_kv(void* cache, bool fp8, size_t idx, float v) {
    if (fp8) {
        __nv_fp8_e4m3 q(v);
        static_cast<uint8_t*>(cache)[idx] = *reinterpret_cast<uint8_t*>(&q);
    } else {
        static_cast<__half*>(cache)[idx] = __float2half_rn(v);
    }
}

// A block per row: warp w holds kv head w's 128 key and value channels (4 a lane), then three query heads and their
// relative states.
__global__ void __launch_bounds__(kThreads) attn_prepare_kernel(AttnPrep a) {
    const int r = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* yq = a.yq + (size_t) r * a.ldq;
    const int p = a.pos[r];
    const float inv_k = a.full ? (1.f / 3.f) : rsqrtf((float) kConv + 1.f);
    const float tau = a.full ? tau_of(p, a.log_floor, a.log_alpha) : 1.f;
    // keys and values
#pragma unroll
    for (int which = 0; which < 2; ++which) {
        const float* yall = which ? a.yv : a.yk;
        const int64_t ld = which ? a.ldv : a.ldk;
        const float* y = yall + (size_t) r * ld;
        const float* taps = which ? a.v_conv : a.k_conv;
        const float* ring = which ? a.v_ring : a.k_ring;
        float val[4], ss = 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int c = w * kHeadDim + lane * 4 + u;
            float acc = 0.f, x0 = 0.f;
#pragma unroll
            for (int j = 0; j < kConv; ++j) {
                const float x = conv_in(yall, (int) ld, 0, ring, kKvHeads * kHeadDim, a.pos, r, j, c);
                if (j == 0) x0 = x;
                float t = taps[c * kConv + (kConv - 1 - j)];
                if (a.full) t += y[kOffTaps + w * kConv + j];
                acc = fmaf(t, x, acc);
            }
            val[u] = (acc + x0) * inv_k;
            ss = fmaf(val[u], val[u], ss);
        }
        ss = warp_sum(ss);
        const float f = which == 0 ? fminf(a.qk_clip / (sqrtf(ss) + 1e-6f), 1.f) : rsqrtf(ss / (float) kHeadDim + a.eps);
        const int64_t cell = p % a.cap;
        void* cache = which ? a.v_cache : a.k_cache;
#pragma unroll
        for (int u = 0; u < 4; ++u)
            store_kv(cache, a.fp8, ((size_t) cell * kKvHeads + w) * kHeadDim + lane * 4 + u, val[u] * f);
    }
    // queries: warp w takes heads 3w .. 3w + 2
    for (int h = w * 3; h < w * 3 + 3; ++h) {
        float q[4], ss = 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int i = lane * 4 + u;
            q[u] = yq[kOffQ + h * kHeadDim + i] * a.q_scale[h * kHeadDim + i];
            ss = fmaf(q[u], q[u], ss);
        }
        ss = warp_sum(ss);
        const float f = fminf(a.qk_clip / (sqrtf(ss) + 1e-6f), 1.f) * tau;
#pragma unroll
        for (int u = 0; u < 4; ++u) a.q[((size_t) r * kHeads + h) * kHeadDim + lane * 4 + u] = q[u] * f;
        // the relative state: 16 values, lanes 0..15
        const float rv = lane < kRel ? yq[kOffR + h * kRel + lane] * a.r_scale[h * kRel + lane] : 0.f;
        const float rs = warp_sum(rv * rv);
        const float fr = fminf(a.rel_clip / (sqrtf(rs) + 1e-6f), 1.f) * tau;
        if (lane < kRel) a.r[((size_t) r * kHeads + h) * kRel + lane] = rv * fr;
        if (lane == 0) a.sink[(size_t) r * kHeads + h] = yq[kOffSink + h];
    }
}

// The rings take the inputs of the window's last positions (after every row read the positions before the window).
__global__ void ring_store_kernel(const float* __restrict__ x, int ld, int off, int width, const int32_t* __restrict__ pos,
                                  int rows, float* __restrict__ ring) {
    const int first = rows > kConvRing ? rows - kConvRing : 0;
    const int r = first + blockIdx.y;
    if (r >= rows) return;
    float* dst = ring + (size_t) (pos[r] % kConvRing) * width;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < width; c += gridDim.x * blockDim.x)
        dst[c] = x[(size_t) r * ld + off + c];
}

void ring_store(const float* x, int ld, int off, int width, const int32_t* pos, int rows, float* ring, cudaStream_t s) {
    const int n = rows < kConvRing ? rows : kConvRing;
    ring_store_kernel<<<dim3((unsigned) ((width + 255) / 256), (unsigned) n), 256, 0, s>>>(x, ld, off, width, pos, rows, ring);
}

__global__ void attn_gate_kernel(const float* __restrict__ attn, const float* __restrict__ yq, int n, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int r = i / (kHeads * kHeadDim), c = i - r * kHeads * kHeadDim;
    out[i] = attn[i] * sigmoidf_(yq[(size_t) r * kProjQ + kOffGate + c]);
}

__global__ void post_conv_kernel(const float* __restrict__ x, const int32_t* __restrict__ pos, const float* __restrict__ taps,
                                 const float* __restrict__ ring, const float* __restrict__ add, int rows, float* __restrict__ out) {
    const int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kWidth) return;
    float acc = 0.f, x0 = 0.f;
#pragma unroll
    for (int j = 0; j < kConv; ++j) {
        const float v = conv_in(x, kWidth, 0, ring, kWidth, pos, r, j, c);
        if (j == 0) x0 = v;
        acc = fmaf(taps[c * kConv + (kConv - 1 - j)], v, acc);
    }
    const float conv = (acc + x0) * rsqrtf((float) kConv + 1.f);
    const size_t i = (size_t) r * kWidth + c;
    out[i] = add ? conv + add[i] : conv;
}

// ---- experts -----------------------------------------------------------------------------------------------------------

// One block: the (row, slot) pairs counted by expert (the experts on the card); a warp scans the counts into each used
// expert's group and first entry; each pair then takes its place in its group from the pairs before it that chose the same
// expert (the rows' order). The blob table (constant) is staged in shared memory before pdl_wait.
__global__ void moe_plan_kernel(const int32_t* __restrict__ ids, int n, int n_expert, const unsigned long long* __restrict__ blobs,
                                unsigned long long* __restrict__ ptr, int32_t* __restrict__ start, int32_t* __restrict__ n_groups,
                                int32_t* __restrict__ dst, int32_t* __restrict__ tok, int k, uint32_t* __restrict__ tally,
                                unsigned long long* __restrict__ rec_blobs, int32_t* __restrict__ rec_n) {
    extern __shared__ unsigned long long smb[];
    unsigned long long* blob = smb;                                // [n_expert]
    int32_t* count = reinterpret_cast<int32_t*>(smb + n_expert);   // [n_expert]
    int32_t* first = count + n_expert;                             // [n_expert]: the expert's first entry
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        blob[e] = blobs[e];
        count[e] = 0;
    }
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (blob[ids[i]]) atomicAdd(&count[ids[i]], 1);
    __syncthreads();
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        int at = 0, g = 0;
        for (int e0 = 0; e0 < n_expert; e0 += 32) {
            const int e = e0 + lane;
            const int c = e < n_expert ? count[e] : 0;
            int s = c, used = c > 0;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int t = __shfl_up_sync(0xffffffffu, s, o), u = __shfl_up_sync(0xffffffffu, used, o);
                if (lane >= o) {
                    s += t;
                    used += u;
                }
            }
            if (e < n_expert) first[e] = at + s - c;
            if (c > 0) {
                ptr[g + used - 1] = blob[e];
                start[g + used - 1] = at + s - c;
            }
            at += __shfl_sync(0xffffffffu, s, 31);
            g += __shfl_sync(0xffffffffu, used, 31);
        }
        if (lane == 0) {
            start[g] = at;
            start[g + 1] = at;
            *n_groups = g;
        }
    }
    __syncthreads();
    if (rec_blobs && threadIdx.x < kAheadExperts) {   // the window's first groups, for the next window's read-ahead
        const int ng = *n_groups;
        rec_blobs[threadIdx.x] = (int) threadIdx.x < ng ? ptr[threadIdx.x] : 0ull;
        if (threadIdx.x == 0) *rec_n = ng < kAheadExperts ? ng : kAheadExperts;
    }
    if (tally)   // a reduction (its result unused), so no thread waits on the load a plain add would make
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x)
            if (count[e]) atomicAdd(&tally[e], (uint32_t) count[e]);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int e = ids[i];
        if (!blob[e]) continue;
        int before = 0;
        for (int j = 0; j < i; ++j) before += ids[j] == e;
        const int at = first[e] + before;
        dst[at] = i;
        tok[at] = i / k;
    }
}

constexpr int kAheadBlocks = 32;
constexpr uint64_t kAheadStep = 128 << 10;
// Block b requests steps b, b + blocks, ... of the blobs' first `bytes` laid end to end, 32 KiB at a time, then reads the
// step's last 16 bytes, so each block keeps one step in flight.
__global__ void __launch_bounds__(32) expert_read_ahead_kernel(const unsigned long long* __restrict__ blobs, const int32_t* __restrict__ n,
                                                               uint64_t bytes) {
    if (threadIdx.x != 0) return;
    const uint64_t total = (uint64_t) *n * bytes;
    uint32_t acc = 0;
    for (uint64_t o0 = (uint64_t) blockIdx.x * kAheadStep; o0 < total; o0 += (uint64_t) gridDim.x * kAheadStep) {
        const uint64_t o1 = min(o0 + kAheadStep, total);
        const char* last = nullptr;
        for (uint64_t o = o0; o < o1;) {
            const uint64_t i = o / bytes, ro = o - i * bytes;
            const uint64_t take = min(min(o1 - o, bytes - ro), (uint64_t) 32768);
            const char* src = (const char*) blobs[i] + ro;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"((unsigned) take) : "memory");
            last = src + take - 16;
            o += take;
        }
        const uint4 w = __ldcg((const uint4*) last);
        acc ^= w.x ^ w.y ^ w.z ^ w.w;
    }
    if (acc == 0x9e3779b9u && total == 1) asm volatile("" ::"r"(acc));   // keeps the reads
}

// Every block loads its share of the experts' first `bytes`, 16 bytes a thread at a time, four loads in flight.
constexpr int kPrefetchBlocks = 170, kPrefetchThreads = 256, kPrefetchMax = 16;
__global__ void __launch_bounds__(kPrefetchThreads) expert_prefetch_kernel(const int32_t* __restrict__ ids, int n,
                                                                            const unsigned long long* __restrict__ blobs, uint64_t bytes,
                                                                            uint32_t* __restrict__ sink) {
    __shared__ unsigned long long addr[kPrefetchMax];
    __shared__ int nu;
    if (threadIdx.x == 0) {   // the distinct experts
        int m = 0;
        for (int i = 0; i < n; ++i) {
            bool seen = false;
            for (int j = 0; j < i; ++j) seen |= ids[j] == ids[i];
            if (!seen) addr[m++] = blobs[ids[i]];
        }
        nu = m;
    }
    __syncthreads();
    const uint64_t n16 = bytes / 16, total = (uint64_t) nu * n16, T = (uint64_t) gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (uint64_t c = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; c < total; c += 4 * T) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint64_t i = c + u * T;
            v[u] = i < total ? __ldcg(reinterpret_cast<const uint4*>((const char*) addr[i / n16] + (i % n16) * 16)) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x9e3779b9u) *sink = acc;   // so the loads are not dropped as unused
}

__global__ void moe_combine_kernel(const float* __restrict__ out, const float* __restrict__ w, int k, float* __restrict__ y) {
    const int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kWidth) return;
    float acc = 0.f;
    for (int s = 0; s < k; ++s) acc = fmaf(w[r * k + s], out[((size_t) r * k + s) * kWidth + c], acc);
    y[(size_t) r * kWidth + c] = acc;
}

__global__ void softcap_kernel(float* __restrict__ x, int n, int64_t ld, float cap) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.y;
    const float inv = 1.f / cap;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float& v = x[(size_t) r * ld + i];
        v = cap * tanhf(v * inv);
    }
}

}  // namespace

void attnres_mix(const MixStates& st, const float* queries, int n_q, int rows, float eps, float* out, int64_t out_stride,
                 void* stream) {
    if (rows <= 0) return;
    if (st.n < 1 || st.n > kMaxMix) throw std::runtime_error("inkling2 attnres_mix: state count");
    cudaStream_t s = (cudaStream_t) stream;
    switch (n_q) {
        case 1: attnres_mix_kernel<1><<<rows, kThreads, 0, s>>>(st, queries, eps, out, out_stride); break;
        case 3: attnres_mix_kernel<3><<<rows, kThreads, 0, s>>>(st, queries, eps, out, out_stride); break;
        case 4: attnres_mix_kernel<4><<<rows, kThreads, 0, s>>>(st, queries, eps, out, out_stride); break;
        default: throw std::runtime_error("inkling2 attnres_mix: query count");
    }
    check("attnres_mix");
}

void ctrl_pair_norm(const float* dest, const float* msg, int rows, float* pair, void* stream) {
    if (rows <= 0) return;
    ctrl_pair_norm_kernel<<<rows, kThreads, 0, (cudaStream_t) stream>>>(dest, msg, pair);
    check("ctrl_pair_norm");
}

void ctrl_blend(const float* dest, const float* msg, const float* gf, int rows, float* out, void* stream) {
    const int n = rows * kWidth;
    if (n <= 0) return;
    ctrl_blend_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(dest, msg, gf, n, out);
    check("ctrl_blend");
}

void attn_prepare(const AttnPrep& a, int rows, void* stream) {
    if (rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    attn_prepare_kernel<<<rows, kThreads, 0, s>>>(a);
    check("attn_prepare");
    ring_store(a.yk, (int) a.ldk, 0, kKvHeads * kHeadDim, a.pos, rows, a.k_ring, s);
    ring_store(a.yv, (int) a.ldv, 0, kKvHeads * kHeadDim, a.pos, rows, a.v_ring, s);
    check("attn_prepare/rings");
}

void attn_gate(const float* attn, const float* yq, int rows, float* out, void* stream) {
    const int n = rows * kHeads * kHeadDim;
    if (n <= 0) return;
    attn_gate_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(attn, yq, n, out);
    check("attn_gate");
}

void post_conv(const float* x, const int32_t* pos, const float* taps, float* ring, const float* add, int rows, float* out,
               void* stream) {
    if (rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    post_conv_kernel<<<dim3((kWidth + 255) / 256, (unsigned) rows), 256, 0, s>>>(x, pos, taps, ring, add, rows, out);
    check("post_conv");
    ring_store(x, kWidth, 0, kWidth, pos, rows, ring, s);
    check("post_conv/ring");
}

void expert_read_ahead(const unsigned long long* blobs, const int32_t* n, uint64_t bytes, void* stream) {
    if (bytes == 0 || bytes % 16) throw std::runtime_error("inkling2 expert_read_ahead: bytes");
    expert_read_ahead_kernel<<<kAheadBlocks, 32, 0, (cudaStream_t) stream>>>(blobs, n, bytes);
    check("expert_read_ahead");
}

void expert_prefetch(const int32_t* ids, int n, const unsigned long long* blobs, uint64_t bytes, uint32_t* sink, void* stream) {
    if (n <= 0) return;
    if (n > kPrefetchMax || bytes == 0 || bytes % 16) throw std::runtime_error("inkling2 expert_prefetch: experts or bytes");
    expert_prefetch_kernel<<<kPrefetchBlocks, kPrefetchThreads, 0, (cudaStream_t) stream>>>(ids, n, blobs, bytes, sink);
    check("expert_prefetch");
}

void moe_plan(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* blobs, unsigned long long* ptr,
              int32_t* start, int32_t* n_groups, int32_t* dst, int32_t* tok, void* stream, uint32_t* tally,
              unsigned long long* rec_blobs, int32_t* rec_n) {
    const int n = rows * k;
    if (n <= 0) return;
    if (n > 1024) throw std::runtime_error("inkling2 moe_plan: more than 1024 pairs");
    bk::launch_pdl(moe_plan_kernel, dim3(1), dim3(256), (size_t) n_expert * (sizeof(unsigned long long) + 2 * sizeof(int32_t)),
                   (cudaStream_t) stream, ids, n,
                   n_expert, blobs, ptr, start, n_groups, dst, tok, k, tally, rec_blobs, rec_n);
    check("moe_plan");
}

void moe_combine(const float* out, const float* w, int rows, int k, float* y, void* stream) {
    if (rows <= 0) return;
    moe_combine_kernel<<<dim3((kWidth + 255) / 256, (unsigned) rows), 256, 0, (cudaStream_t) stream>>>(out, w, k, y);
    check("moe_combine");
}

void softcap(float* x, int rows, int n, int64_t ld, float cap, void* stream) {
    if (rows <= 0 || n <= 0) return;
    bk::launch_pdl(softcap_kernel, dim3(256, (unsigned) rows), dim3(256), 0, (cudaStream_t) stream, x, n, ld, cap);
    check("softcap");
}

}  // namespace basalt::models::inkling2
