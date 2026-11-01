// src/kernels/cuda/q80_proxy.cu - see include/basalt/kernels/q80_proxy.hpp.
//
// Built with --use_fast_math like native_mmvq.cu, whose one-column Q8_0 x q8_1 dot the exact rows repeat.
#include "basalt/kernels/q80_proxy.hpp"
#include "basalt/kernels/dp4a.hpp"
#include "basalt/kernels/pdl.hpp"
#include "head_proxy.cuh"
#include "q80p.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int WARP = 32, BPR = 80;   // a row of n_in 2560: 80 blocks of 32
constexpr size_t RB = q80p::kRowBytes;

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    return x;
}

// ---- the 2-bit copy

// One thread per 32-weight group: the group's Q8_0 values fitted into the copy (head_proxy.cuh).
__global__ void q80_proxy_pack_kernel(const uint8_t* __restrict__ q8, int64_t row0, int64_t n,
                                      uint4* __restrict__ wq, uint16_t* __restrict__ sc,
                                      const int32_t* __restrict__ rows) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * BPR) return;
    const int64_t row = row0 + i / BPR;
    const int g = (int) (i % BPR);
    const int64_t wr = rows != nullptr ? (int64_t) rows[row] : row;
    if (wr < 0) return;
    int8_t q[32];
    float d;
    q80p::block_codes(q8 + (size_t) wr * RB, g, q, d);
    float w[32];
    for (int l = 0; l < 32; ++l) w[l] = d * (float) q[l];
    px::pack_group<true>(w, row, g, wq, sc);
}

// ---- one row by one warp, in native_mmvq's order

// native_mmvq's one-column Q8_0 kernel runs a row on four warps: thread tid takes blocks kbx = tid / 4, += 32, words
// kqs = 2 (tid % 4), kqs + 1.  Lane L stands for its threads L, L + 32, L + 64, L + 96 (virtual threads 0..3: three
// blocks for 0 and 1, two for 2 and 3); their sums are added in the cross-warp order before the same xor tree.  The
// rows are the head's bit planes (q80p.cuh), read back to the Q8_0 codes.
constexpr int kNch = 10;
__device__ __forceinline__ int q80_kb(int lane, int c) {
    constexpr int off[kNch] = {0, 32, 64, 8, 40, 72, 16, 48, 24, 56};
    return lane / 4 + off[c];
}
__device__ __forceinline__ void q80_row_load(const uint8_t* __restrict__ w, int lane, q80p::Quarter (&r)[kNch]) {
#pragma unroll
    for (int c = 0; c < kNch; ++c) r[c] = q80p::load_quarter<true>(w, q80_kb(lane, c), lane % 4);
}
// xe: per q8_1 block b and word pair p (0..3), {q[2p], q[2p + 1], the block's scale as float}
__device__ __forceinline__ float q80_row_dot(const q80p::Quarter (&r)[kNch], const int4* __restrict__ xe, int lane) {
    constexpr int vt[kNch] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3};
    const int p = lane % 4;
    float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int c = 0; c < kNch; ++c) {
        const int4 e = xe[q80_kb(lane, c) * 4 + p];
        int v0, v1;
        q80p::codes(r[c].nib, r[c].top, r[c].low, p, v0, v1);
        int sumi = 0;
        sumi = DP4A(v0, e.x, sumi);
        sumi = DP4A(v1, e.y, sumi);
        s[vt[c]] += r[c].d * __int_as_float(e.z) * float(sumi);
    }
    float t = s[0];
#pragma unroll
    for (int v = 1; v < 4; ++v) t += s[v];
    return warp_sum(t);
}

// The columns' activations into shared memory: per q8_1 block b and word pair p (0..3) {q[2p], q[2p + 1], scale}.
__device__ __forceinline__ void load_xe(const Q81Block* __restrict__ x, int4* __restrict__ xe) {
    for (int i = threadIdx.x; i < BPR * 4; i += blockDim.x) {
        const Q81Block* blk = x + i / 4;
        const int* q8 = reinterpret_cast<const int*>(blk->qs) + 2 * (i % 4);
        const unsigned dsb = __ldg(reinterpret_cast<const unsigned*>(&blk->ds));
        xe[i] = make_int4(__ldg(q8), __ldg(q8 + 1), __float_as_int(__low2float(*reinterpret_cast<const half2*>(&dsb))), 0);
    }
}

// One block: the rows near the best approximate score are scored exactly (a warp per row, two rows in flight), the
// exact best is the pick.  `by_map`: w is the whole head and row r is w's row map[r].
__global__ void __launch_bounds__(512) q80_proxy_final_kernel(const uint8_t* __restrict__ w,
                                                              const Q81Block* __restrict__ x, int n_tiles,
                                                              const int32_t* __restrict__ map,
                                                              const float2* __restrict__ cand,
                                                              const float2* __restrict__ part, int* __restrict__ out_id,
                                                              float* __restrict__ out_prob, int by_map) {
    __shared__ __align__(16) int4 xe[BPR * 4];
    __shared__ px::Final f;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    pdl_wait();
    if (tid == 0) f.cnt = 0;
    load_xe(x, xe);
    const int n = px::collect(f, n_tiles, map, cand, out_id, out_prob);
    if (n == 0) return;
    auto row_of = [&](int c) { const int r = f.list[c]; return w + (size_t) (by_map ? __ldg(map + r) : r) * RB; };
    q80p::Quarter cur[kNch], nxt[kNch];
    if (warp < n) q80_row_load(row_of(warp), lane, cur);
    for (int c = warp; c < n; c += 16) {
        if (c + 16 < n) q80_row_load(row_of(c + 16), lane, nxt);
        const float y = q80_row_dot(cur, xe, lane);
        if (lane == 0) f.ex[c] = y;
#pragma unroll
        for (int k = 0; k < kNch; ++k) cur[k] = nxt[k];
    }
    __syncthreads();
    if (warp != 0) return;
    px::pick(f, n, n_tiles, map, part, out_id, out_prob);
}

// ---- every row of a subset through a row map

// A warp per row (the rescoring's arithmetic), row r read from head row rows[r]; a negative entry reads row 0.
constexpr int kRowWarps = 8;
__global__ void __launch_bounds__(kRowWarps * WARP) q80_rows_kernel(const uint8_t* __restrict__ w,
                                                                     const int32_t* __restrict__ rows,
                                                                     const Q81Block* __restrict__ x,
                                                                     float* __restrict__ y, int n_out) {
    __shared__ __align__(16) int4 xe[BPR * 4];
    const int lane = threadIdx.x & 31, r = (int) blockIdx.x * kRowWarps + (int) (threadIdx.x >> 5);
    pdl_wait();
    load_xe(x, xe);
    __syncthreads();
    if (r >= n_out) return;
    const int32_t id = rows[r];
    q80p::Quarter cur[kNch];
    q80_row_load(w + (size_t) (id < 0 ? 0 : id) * RB, lane, cur);
    const float v = q80_row_dot(cur, xe, lane);
    if (lane == 0) y[r] = v;
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void q80_proxy_pack(const void* q80_rows, int64_t row0, int64_t n, void* proxy, int64_t n_cap, void* stream,
                    const int32_t* rows) {
    if (n <= 0) return;
    const int64_t total = n * BPR;
    q80_proxy_pack_kernel<<<(unsigned) ((total + 127) / 128), 128, 0, (cudaStream_t) stream>>>(
        (const uint8_t*) q80_rows, row0, n, (uint4*) proxy, px::scales(proxy, n_cap), rows);
    check("q80_proxy_pack");
}

bool q80_proxy_top1(const void* proxy, const void* q80_rows, const void* x_q8_1, int n_rows, int64_t n_cap,
                    const int32_t* map, int* out_id, float* out_prob, void* scratch, void* stream, bool rows_by_map) {
    if (n_rows <= 0 || n_rows > n_cap || !proxy || !q80_rows || !x_q8_1 || !map || !out_id || !out_prob || !scratch)
        return false;
    const int tiles = (int) px::tiles(n_rows);
    const uint4* wq = (const uint4*) proxy;
    const uint32_t* sc = (const uint32_t*) px::scales(const_cast<void*>(proxy), n_cap);
    float2* cand = (float2*) scratch;
    float2* part = cand + (size_t) px::tiles(n_cap) * px::kPer;
    launch_pdl(px::score_kernel, dim3((unsigned) tiles), dim3(128), 0, (cudaStream_t) stream, wq, sc,
               (const px::X81*) x_q8_1, n_rows, map, cand, part);
    launch_pdl(q80_proxy_final_kernel, dim3(1), dim3(512), 0, (cudaStream_t) stream, (const uint8_t*) q80_rows,
               (const Q81Block*) x_q8_1, tiles, map, (const float2*) cand, (const float2*) part, out_id, out_prob,
               rows_by_map ? 1 : 0);
    check("q80_proxy_top1");
    return true;
}

void q80_mmvq_rows(const void* weights, const int32_t* rows, const void* x_q8_1, float* y, int n_in, int n_out,
                   void* stream) {
    if (n_in != BPR * 32 || n_out <= 0 || !weights || !rows || !x_q8_1 || !y) {
        basalt::log::error("basalt", "q80_mmvq_rows: unsupported shape (n_in %d, n_out %d)", n_in, n_out);
        std::exit(1);
    }
    launch_pdl(q80_rows_kernel, dim3((unsigned) ((n_out + kRowWarps - 1) / kRowWarps)), dim3(kRowWarps * WARP), 0,
               (cudaStream_t) stream, (const uint8_t*) weights, rows, (const Q81Block*) x_q8_1, y, n_out);
    check("q80_mmvq_rows");
}

}  // namespace basalt::kernels
