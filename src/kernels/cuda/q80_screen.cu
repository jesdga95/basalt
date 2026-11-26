// src/kernels/cuda/q80_screen.cu - see include/basalt/kernels/q80_screen.hpp.
//
// Built with --use_fast_math like native_mmvq.cu, whose one-column Q8_0 arithmetic the exact rows repeat.
#include "basalt/kernels/q80_screen.hpp"
#include "basalt/kernels/dp4a.hpp"
#include "basalt/kernels/head_proxy.cuh"
#include "basalt/kernels/pdl.hpp"
#include "basalt/log.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace basalt::kernels {
namespace {

constexpr int WARP = 32;

struct Q80B {
    __half d;
    int8_t qs[32];
};
struct Q81B {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q80B) == 34 && sizeof(Q81B) == 36);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    return x;
}

// A Q8_0 word of a block's codes (the blocks are 2-byte aligned).
__device__ __forceinline__ int q80_word(const int8_t* qs, int i32) {
    const auto* x = reinterpret_cast<const uint16_t*>(qs);
    return (int) x[2 * i32] | ((int) x[2 * i32 + 1] << 16);
}

// One thread per 32-weight group: the group's Q8_0 values fitted into the copy.
template <int N_IN>
__global__ void q80_screen_pack_kernel(const Q80B* __restrict__ head, int64_t n_rows, uint4* __restrict__ wq,
                                       uint16_t* __restrict__ sc) {
    constexpr int NB = N_IN / 32;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * NB) return;
    const int64_t row = i / NB;
    const int g = (int) (i % NB);
    const Q80B* b = head + (size_t) row * NB + g;
    const float d = __half2float(b->d);
    float w[32];
    for (int l = 0; l < 32; ++l) w[l] = d * (float) b->qs[l];
    px::pack_group<N_IN, true>(w, row, g, wq, sc);
}

template <int N_IN>
__global__ void __launch_bounds__(128) q80_screen_score_kernel(const uint4* __restrict__ wq,
                                                               const uint32_t* __restrict__ sc,
                                                               const px::X81* __restrict__ x, int n_rows,
                                                               float2* __restrict__ cand, float2* __restrict__ part) {
    __shared__ px::ScoreShared<N_IN> sh;
    px::score_tile<N_IN, false>(sh, (int) blockIdx.x, wq, sc, x, n_rows, nullptr, cand, part);
}

// ---- one row by one warp, in native_mmvq's order
//
// native_mmvq's one-column Q8_0 kernel for rows of 32 blocks or more runs a row on four warps: thread t takes blocks
// t / 4 + 32 p (p = 0, 1, ...) at words 2 (t % 4), +1, summing in p order; warp 0 adds the other warps' sums in warp
// order before the xor tree.  Lane L stands for its threads L + 32 v (v = 0..3).
template <int NB>
struct Q80Row {
    static constexpr int NP = (NB + 31) / 32;
    int v0[4][NP], v1[4][NP];
    float d[4][NP];
};

template <int NB>
__device__ __forceinline__ void q80_row_load(const Q80B* __restrict__ w, int lane, Q80Row<NB>& r) {
    const int kqs = 2 * (lane % 4);
#pragma unroll
    for (int v = 0; v < 4; ++v)
#pragma unroll
        for (int p = 0; p < Q80Row<NB>::NP; ++p) {
            const int kbx = 8 * v + lane / 4 + 32 * p;
            if (kbx < NB) {
                r.v0[v][p] = q80_word(w[kbx].qs, kqs);
                r.v1[v][p] = q80_word(w[kbx].qs, kqs + 1);
                r.d[v][p] = __half2float(w[kbx].d);
            }
        }
}

// xe: per q8_1 block b and word pair p (0..3), {q[2p], q[2p + 1], the block's scale as float}
template <int NB>
__device__ __forceinline__ float q80_row_dot(const Q80Row<NB>& r, const int4* __restrict__ xe, int lane) {
    float s[4];
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        s[v] = 0.0f;
#pragma unroll
        for (int p = 0; p < Q80Row<NB>::NP; ++p) {
            const int kbx = 8 * v + lane / 4 + 32 * p;
            if (kbx < NB) {
                const int4 e = xe[kbx * 4 + lane % 4];
                int sumi = 0;
                sumi = DP4A(r.v0[v][p], e.x, sumi);
                sumi = DP4A(r.v1[v][p], e.y, sumi);
                s[v] += r.d[v][p] * __int_as_float(e.z) * float(sumi);
            }
        }
    }
    float t = s[0];
#pragma unroll
    for (int v = 1; v < 4; ++v) t += s[v];
    return warp_sum(t);
}

// One block: the rows near the best approximate score are scored exactly (a warp per row, two rows in flight), the
// exact best is the pick.
template <int N_IN>
__global__ void __launch_bounds__(512) q80_screen_final_kernel(const Q80B* __restrict__ head,
                                                               const Q81B* __restrict__ x, int n_tiles,
                                                               const float2* __restrict__ cand,
                                                               int32_t* __restrict__ out_id, float* __restrict__ out_logit) {
    constexpr int NB = N_IN / 32;
    static_assert(NB >= 32, "the one-column layout of rows of 32 blocks or more");
    __shared__ __align__(16) int4 xe[NB * 4];
    __shared__ px::Final f;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    pdl_wait();
    if (tid == 0) f.cnt = 0;
    for (int i = tid; i < NB * 4; i += blockDim.x) {
        const Q81B* blk = x + i / 4;
        const int* q8 = reinterpret_cast<const int*>(blk->qs) + 2 * (i % 4);
        const unsigned dsb = __ldg(reinterpret_cast<const unsigned*>(&blk->ds));
        xe[i] = make_int4(__ldg(q8), __ldg(q8 + 1), __float_as_int(__low2float(*reinterpret_cast<const half2*>(&dsb))), 0);
    }
    const int n = px::collect<false, false>(f, n_tiles, nullptr, cand, out_id, nullptr);
    if (n == 0) {
        if (out_logit && tid == 0) *out_logit = __int_as_float(0xff800000);
        return;
    }
    Q80Row<NB> cur, nxt;
    if (warp < n) q80_row_load<NB>(head + (size_t) f.list[warp] * NB, lane, cur);
    for (int c = warp; c < n; c += 16) {
        if (c + 16 < n) q80_row_load<NB>(head + (size_t) f.list[c + 16] * NB, lane, nxt);
        const float y = q80_row_dot<NB>(cur, xe, lane);
        if (lane == 0) f.ex[c] = y;
        cur = nxt;
    }
    __syncthreads();
    if (warp != 0) return;
    px::pick<false, false>(f, n, n_tiles, nullptr, nullptr, out_id, nullptr);
    if (out_logit && lane == 0) {
        float best = __int_as_float(0xff800000);
        const int32_t picked = *out_id;
        for (int c = 0; c < n; ++c)
            if (f.list[c] == picked) { best = f.ex[c]; break; }
        *out_logit = best;
    }
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void check_width(int n_in, const char* what) {
    if (!q80_screen_supported(n_in)) {
        basalt::log::error("basalt", "%s: no screen for rows of %d values", what, n_in);
        std::exit(1);
    }
}

}  // namespace

bool q80_screen_supported(int n_in) noexcept { return n_in == 3072; }

size_t q80_screen_bytes(int n_in, int64_t n_rows) { return px::bytes(n_rows, n_in); }

size_t q80_screen_scratch_bytes(int64_t n_rows) { return px::scratch_bytes(n_rows); }

void q80_screen_pack(const void* head_q8_0, int n_in, int64_t n_rows, void* screen, void* stream) {
    check_width(n_in, "q80_screen_pack");
    const cudaStream_t s = (cudaStream_t) stream;
    cudaMemsetAsync(screen, 0, px::bytes(n_rows, n_in), s);   // the last tile's rows past n_rows score as zeros
    const int64_t total = n_rows * (n_in / 32);
    q80_screen_pack_kernel<3072><<<(unsigned) ((total + 127) / 128), 128, 0, s>>>(
        (const Q80B*) head_q8_0, n_rows, (uint4*) screen, px::scales(screen, n_rows, n_in));
    check("q80_screen_pack");
}

void q80_screen_top1(const void* screen, const void* head_q8_0, const void* x_q8_1, int n_in, int64_t n_rows,
                     void* scratch, int32_t* out_id, void* stream, float* out_logit) {
    check_width(n_in, "q80_screen_top1");
    const cudaStream_t s = (cudaStream_t) stream;
    const int tiles = (int) px::tiles(n_rows);
    float2* cand = (float2*) scratch;
    float2* part = cand + (size_t) tiles * px::kPer;
    launch_pdl(q80_screen_score_kernel<3072>, dim3((unsigned) tiles), dim3(128), 0, s, (const uint4*) screen,
               (const uint32_t*) px::scales(const_cast<void*>(screen), n_rows, n_in), (const px::X81*) x_q8_1,
               (int) n_rows, cand, part);
    launch_pdl(q80_screen_final_kernel<3072>, dim3(1), dim3(512), 0, s, (const Q80B*) head_q8_0, (const Q81B*) x_q8_1,
               tiles, (const float2*) cand, out_id, out_logit);
    check("q80_screen_top1");
}

}  // namespace basalt::kernels
