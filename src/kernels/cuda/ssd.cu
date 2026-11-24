// src/kernels/cuda/ssd.cu - see include/basalt/kernels/ssd.hpp: the Mamba-2 mixer's convolution, recurrence, gated
// norm and commit.
//
// The recurrence runs kParts blocks per head, each a quarter of its state rows: warp w holds kRows of them, lane l
// their columns l, l + 32, l + 64 and l + 96, in registers for the whole run, so the state is read once and written once (state_out) whatever
// the run's length, every load and store a warp's 32 consecutive floats.  The run goes kTile rows at a time: their x,
// B, C, dt and dA are staged in shared memory once per tile.  A row's step is ssd_step for every held value; its
// output for state row p is the lane's four columns' fmaf chain, then the warp's butterfly.  The commit replays the
// same step on the values the scan stored, so its state is the scan's bit for bit.
#include "basalt/kernels/ssd.hpp"
#include "basalt/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int kHeadDim = 64, kState = 128, kConv = 4;
constexpr int kThreads = 256;                   // a block: 8 warps
constexpr int kParts = 4;                       // blocks per head
constexpr int kBlockRows = kHeadDim / kParts;   // state rows a block holds
constexpr int kRows = kBlockRows / (kThreads / 32);   // state rows a warp holds
constexpr int kCols = kState / 32;                  // columns a lane holds in each
constexpr int kTile = 16;                       // rows of a run staged at once
constexpr int kNormThreads = 256;

__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// One held value's step: S dA + (dt x) B, the product and the sum each rounded once.
__device__ __forceinline__ float ssd_step(float s, float dA, float dtx, float b) {
    return __fmaf_rn(s, dA, __fmul_rn(dtx, b));
}

// ---------------------------------------------------------------- convolution
__device__ __forceinline__ void conv_stage(int idx, int C, const float* __restrict__ raw, int64_t ld_raw,
                                           const float* __restrict__ hist, const float* __restrict__ w,
                                           const float* __restrict__ bias, int T, float* __restrict__ xbc) {
    if (idx >= T * C) return;
    const int t = idx / C, c = idx % C;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < kConv; ++k) {
        const int j = t - (kConv - 1) + k;   // the source row: the history before row 0
        const float v = j < 0 ? hist[(size_t) c * (kConv - 1) + (kConv - 1) + j] : raw[(size_t) j * ld_raw + c];
        acc = __fmaf_rn(w[(size_t) c * kConv + k], v, acc);
    }
    xbc[(size_t) t * C + c] = silu(__fadd_rn(acc, bias[c]));
}

__global__ void conv_kernel(int C, const float* raw, int64_t ld_raw, const float* hist, const float* w, const float* bias,
                            int T, float* xbc) {
    conv_stage((int) (blockIdx.x * blockDim.x + threadIdx.x), C, raw, ld_raw, hist, w, bias, T, xbc);
}

__device__ __forceinline__ void conv_commit_stage(int c, int C, float* __restrict__ hist, const float* __restrict__ raw,
                                                  int64_t ld_raw, int n) {
    if (c >= C || n <= 0) return;
    float h[kConv - 1];
#pragma unroll
    for (int j = 0; j < kConv - 1; ++j) {
        const int src = n - (kConv - 1) + j;   // in [hist | raw_0 .. raw_{n-1}]: the history while negative
        h[j] = src < 0 ? hist[(size_t) c * (kConv - 1) + (kConv - 1) + src] : raw[(size_t) src * ld_raw + c];
    }
#pragma unroll
    for (int j = 0; j < kConv - 1; ++j) hist[(size_t) c * (kConv - 1) + j] = h[j];
}

__global__ void conv_commit_kernel(int C, float* hist, const float* raw, int64_t ld_raw, const int32_t* n_keep, int n) {
    conv_commit_stage((int) (blockIdx.x * blockDim.x + threadIdx.x), C, hist, raw, ld_raw, n_keep ? *n_keep : n);
}

// ---------------------------------------------------------------- recurrence
struct ScanArgs {
    const float* state;
    const float* xbc;
    const float* dt_raw;
    int64_t ld_dt;
    const float *dt_bias, *A, *D;
    int T, n_heads, n_groups, C, d_inner;
    float *y, *dt_out, *dA_out, *state_out;
};

// A tile's staged values: rows [t0, t0 + n) of head h's x, its group's B and C, and their dt and dA.
struct Tile {
    float x[kTile][kBlockRows], b[kTile][kState], c[kTile][kState], dt[kTile], dA[kTile];
};

__device__ __forceinline__ void load_state(const float* st, int warp, int lane, float (&s)[kRows][kCols]) {
#pragma unroll
    for (int r = 0; r < kRows; ++r)
#pragma unroll
        for (int j = 0; j < kCols; ++j) s[r][j] = st[(size_t) (warp * kRows + r) * kState + j * 32 + lane];
}

__device__ __forceinline__ void store_state(float* st, int warp, int lane, const float (&s)[kRows][kCols]) {
#pragma unroll
    for (int r = 0; r < kRows; ++r)
#pragma unroll
        for (int j = 0; j < kCols; ++j) st[(size_t) (warp * kRows + r) * kState + j * 32 + lane] = s[r][j];
}

// Part `part` of head h's run.  Every thread computes a row's dt and dA alike; part 0 stores them.
__device__ __forceinline__ void scan_stage(int h, int part, int tid, const ScanArgs& a) {
    __shared__ Tile tl;
    const int warp = tid >> 5, lane = tid & 31;
    const int g = h / (a.n_heads / a.n_groups);
    const size_t s0 = ((size_t) h * kHeadDim + part * kBlockRows) * kState;
    float s[kRows][kCols];
    load_state(a.state + s0, warp, lane, s);
    const float A = a.A[h], D = a.D[h], bias = a.dt_bias[h];
    const int xb = h * kHeadDim + part * kBlockRows, bb = a.d_inner + g * kState, cb = a.d_inner + a.n_groups * kState + g * kState;
    for (int t0 = 0; t0 < a.T; t0 += kTile) {
        const int n = min(kTile, a.T - t0);
        __syncthreads();   // the previous tile is read
        for (int i = tid; i < n * kBlockRows; i += kThreads)
            tl.x[i / kBlockRows][i % kBlockRows] = a.xbc[(size_t) (t0 + i / kBlockRows) * a.C + xb + i % kBlockRows];
        for (int i = tid; i < n * kState; i += kThreads) {
            const float* row = a.xbc + (size_t) (t0 + i / kState) * a.C;
            tl.b[i / kState][i % kState] = row[bb + i % kState];
            tl.c[i / kState][i % kState] = row[cb + i % kState];
        }
        if (tid < n) {
            const float dt = softplus(__fadd_rn(a.dt_raw[(size_t) (t0 + tid) * a.ld_dt + h], bias));
            const float dA = expf(__fmul_rn(dt, A));
            tl.dt[tid] = dt;
            tl.dA[tid] = dA;
            if (part == 0 && a.dt_out) a.dt_out[(size_t) (t0 + tid) * a.n_heads + h] = dt;
            if (part == 0 && a.dA_out) a.dA_out[(size_t) (t0 + tid) * a.n_heads + h] = dA;
        }
        __syncthreads();
        for (int t = 0; t < n; ++t) {
            const float dt = tl.dt[t], dA = tl.dA[t];
            float bj[kCols], cj[kCols];
#pragma unroll
            for (int j = 0; j < kCols; ++j) {
                bj[j] = tl.b[t][j * 32 + lane];
                cj[j] = tl.c[t][j * 32 + lane];
            }
            float mine = 0.0f;   // lane r keeps state row warp * kRows + r's output
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const float x = tl.x[t][warp * kRows + r];
                const float dtx = __fmul_rn(dt, x);
                float acc = 0.0f;
#pragma unroll
                for (int j = 0; j < kCols; ++j) {
                    s[r][j] = ssd_step(s[r][j], dA, dtx, bj[j]);
                    acc = __fmaf_rn(s[r][j], cj[j], acc);
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
                if (lane == r) mine = __fmaf_rn(D, x, acc);
            }
            if (lane < kRows) a.y[(size_t) (t0 + t) * a.d_inner + xb + warp * kRows + lane] = mine;
        }
    }
    if (a.state_out) store_state(a.state_out + s0, warp, lane, s);
}

__global__ void __launch_bounds__(kThreads) scan_kernel(const ScanArgs a) {
    scan_stage((int) blockIdx.x / kParts, (int) blockIdx.x % kParts, (int) threadIdx.x, a);
}

struct CommitArgs {
    float* state;
    const float *xbc, *dt, *dA;
    const int32_t* n_keep;
    int n_heads, n_groups, C, d_inner;
};

__device__ __forceinline__ void commit_stage(int h, int part, int tid, const CommitArgs& a) {
    __shared__ Tile tl;
    const int nk = *a.n_keep;
    if (nk <= 0) return;
    const int warp = tid >> 5, lane = tid & 31;
    const int g = h / (a.n_heads / a.n_groups);
    float s[kRows][kCols];
    float* st = a.state + ((size_t) h * kHeadDim + part * kBlockRows) * kState;
    load_state(st, warp, lane, s);
    const int xb = h * kHeadDim + part * kBlockRows, bb = a.d_inner + g * kState;
    for (int t0 = 0; t0 < nk; t0 += kTile) {
        const int n = min(kTile, nk - t0);
        __syncthreads();
        for (int i = tid; i < n * kBlockRows; i += kThreads)
            tl.x[i / kBlockRows][i % kBlockRows] = a.xbc[(size_t) (t0 + i / kBlockRows) * a.C + xb + i % kBlockRows];
        for (int i = tid; i < n * kState; i += kThreads)
            tl.b[i / kState][i % kState] = a.xbc[(size_t) (t0 + i / kState) * a.C + bb + i % kState];
        if (tid < n) {
            tl.dt[tid] = a.dt[(size_t) (t0 + tid) * a.n_heads + h];
            tl.dA[tid] = a.dA[(size_t) (t0 + tid) * a.n_heads + h];
        }
        __syncthreads();
        for (int t = 0; t < n; ++t) {
            const float dt = tl.dt[t], dA = tl.dA[t];
            float bj[kCols];
#pragma unroll
            for (int j = 0; j < kCols; ++j) bj[j] = tl.b[t][j * 32 + lane];
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const float dtx = __fmul_rn(dt, tl.x[t][warp * kRows + r]);
#pragma unroll
                for (int j = 0; j < kCols; ++j) s[r][j] = ssd_step(s[r][j], dA, dtx, bj[j]);
            }
        }
    }
    store_state(st, warp, lane, s);
}

__global__ void __launch_bounds__(kThreads) commit_kernel(const CommitArgs a) {
    commit_stage((int) blockIdx.x / kParts, (int) blockIdx.x % kParts, (int) threadIdx.x, a);
}

// ---------------------------------------------------------------- gated norm
// Row t, group g: v = y silu(z); the block's sum of v^2 in a fixed order (each thread's values in order, then the
// warps' butterflies, then the warps in order).
__device__ __forceinline__ void norm_stage(int t, int grp, int tid, int G, const float* __restrict__ y, const float* __restrict__ z,
                                           int64_t ld_z, const float* __restrict__ w, float eps, int d_inner,
                                           float* __restrict__ out, uint16_t* __restrict__ out16) {
    __shared__ float red[kNormThreads / 32];
    constexpr int kMaxPer = 4;   // values per thread: groups of up to 1024
    const int base = grp * G;
    float v[kMaxPer];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxPer; ++i) {
        const int c = tid + i * kNormThreads;
        v[i] = 0.0f;
        if (c < G) {
            v[i] = __fmul_rn(y[(size_t) t * d_inner + base + c], silu(z[(size_t) t * ld_z + base + c]));
            ss = __fmaf_rn(v[i], v[i], ss);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int i = 0; i < kNormThreads / 32; ++i) tot = __fadd_rn(tot, red[i]);
    const float r = 1.0f / sqrtf(__fadd_rn(__fdiv_rn(tot, (float) G), eps));
#pragma unroll
    for (int i = 0; i < kMaxPer; ++i) {
        const int c = tid + i * kNormThreads;
        if (c >= G) continue;
        const float o = __fmul_rn(__fmul_rn(v[i], r), w[base + c]);
        if (out) out[(size_t) t * d_inner + base + c] = o;
        if (out16) out16[(size_t) t * d_inner + base + c] = bf16_from_f32(o);
    }
}

__global__ void __launch_bounds__(kNormThreads) norm_kernel(int G, const float* y, const float* z, int64_t ld_z, const float* w,
                                                            float eps, int d_inner, float* out, uint16_t* out16) {
    norm_stage((int) blockIdx.x, (int) blockIdx.y, (int) threadIdx.x, G, y, z, ld_z, w, eps, d_inner, out, out16);
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

std::string ssd_check(const SsdGeometry& g) {
    if (g.head_dim != kHeadDim || g.d_state != kState) return "the SSD kernels take head_dim 64 and d_state 128";
    if (g.d_conv != kConv) return "the SSD kernels take a convolution of 4 taps";
    if (g.n_heads <= 0 || g.n_groups <= 0 || g.n_heads % g.n_groups) return "SSD heads that do not divide into their groups";
    const int G = g.d_inner() / g.n_groups;
    if (G > 4 * kNormThreads || g.d_inner() % g.n_groups) return "an SSD norm group wider than 1024 channels";
    return {};
}

void ssd_conv(const SsdGeometry& g, const float* raw, int64_t ld_raw, const float* hist, const float* w, const float* bias,
              int T, float* xbc, void* stream) {
    if (T <= 0) return;
    const int C = g.conv_dim();
    const long long n = (long long) T * C;
    conv_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(C, raw, ld_raw, hist, w, bias, T, xbc);
    check("ssd_conv");
}

void ssd_scan(const SsdGeometry& g, const float* state, const float* xbc, const float* dt_raw, int64_t ld_dt,
              const float* dt_bias, const float* A, const float* D, int T, float* y, float* dt_out, float* dA_out,
              float* state_out, void* stream) {
    if (T <= 0) return;
    const ScanArgs a{state, xbc, dt_raw, ld_dt, dt_bias, A, D, T, g.n_heads, g.n_groups, g.conv_dim(), g.d_inner(),
                     y, dt_out, dA_out, state_out};
    scan_kernel<<<(unsigned) (g.n_heads * kParts), kThreads, 0, (cudaStream_t) stream>>>(a);
    check("ssd_scan");
}

void ssd_commit(const SsdGeometry& g, float* state, const float* xbc, const float* dt, const float* dA, const int32_t* n_keep,
                void* stream) {
    const CommitArgs a{state, xbc, dt, dA, n_keep, g.n_heads, g.n_groups, g.conv_dim(), g.d_inner()};
    commit_kernel<<<(unsigned) (g.n_heads * kParts), kThreads, 0, (cudaStream_t) stream>>>(a);
    check("ssd_commit");
}

void ssd_conv_commit(const SsdGeometry& g, float* hist, const float* raw, int64_t ld_raw, const int32_t* n_keep, void* stream) {
    const int C = g.conv_dim();
    conv_commit_kernel<<<(unsigned) ((C + 255) / 256), 256, 0, (cudaStream_t) stream>>>(C, hist, raw, ld_raw, n_keep, 0);
    check("ssd_conv_commit");
}

void ssd_conv_advance(const SsdGeometry& g, float* hist, const float* raw, int64_t ld_raw, int n, void* stream) {
    if (n <= 0) return;
    const int C = g.conv_dim();
    conv_commit_kernel<<<(unsigned) ((C + 255) / 256), 256, 0, (cudaStream_t) stream>>>(C, hist, raw, ld_raw, nullptr, n);
    check("ssd_conv_advance");
}

void ssd_gated_norm(const SsdGeometry& g, const float* y, const float* z, int64_t ld_z, const float* w, float eps, int T,
                    float* out, uint16_t* out16, void* stream) {
    if (T <= 0) return;
    const dim3 grid((unsigned) T, (unsigned) g.n_groups);
    norm_kernel<<<grid, kNormThreads, 0, (cudaStream_t) stream>>>(g.d_inner() / g.n_groups, y, z, ld_z, w, eps, g.d_inner(),
                                                                   out, out16);
    check("ssd_gated_norm");
}

}  // namespace basalt::kernels
