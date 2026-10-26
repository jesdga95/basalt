// src/kernels/cuda/q80_head.cu - see include/basalt/kernels/q80_head.hpp.
//
// Built with --use_fast_math like native_mmvq.cu, whose Q8_0 x q8_1 dot this repeats.
#include "basalt/kernels/q80_head.hpp"
#include "basalt/kernels/dp4a.hpp"
#include "basalt/kernels/pdl.hpp"
#include "smem_opt_in.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int QK = 32, WARP = 32;
constexpr int VT = 4;      // virtual threads per lane: native_mmvq's four warps of one row
constexpr int WPB = 8;     // warps (rows in flight) per block

struct Q80Block {
    half d;
    int8_t qs[32];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q80Block) == 34 && sizeof(Q81Block) == 36);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    return x;
}

// native_mmvq.cu's load_int_b2: a 2-byte aligned word as two 16-bit loads
__device__ __forceinline__ int word_b2(const int8_t* qs, int i32) {
    const auto* x = reinterpret_cast<const unsigned short*>(qs);
    int v = __ldg(x + 2 * i32);
    v |= __ldg(x + 2 * i32 + 1) << 16;
    return v;
}
// One (row, block) of one virtual thread: the two weight words at kqs and the block scale
struct W8 {
    int v0, v1;
    float d;
};
__device__ __forceinline__ W8 q80_load(const Q80Block* __restrict__ b, int kqs) {
    W8 r;
    r.v0 = word_b2(b->qs, kqs);
    r.v1 = word_b2(b->qs, kqs + 1);
    r.d = __half2float(__ldg(&b->d));
    return r;
}
// a column's activations in shared memory: per q8_1 block b and word pair p (0..3), {q[2p], q[2p + 1], the block's
// scale as float}
__device__ __forceinline__ float q80_apply(const W8& r, const int4* __restrict__ xe, int b, int p) {
    const int4 e = xe[b * 4 + p];
    int sumi = 0;
    sumi = DP4A(r.v0, e.x, sumi);
    sumi = DP4A(r.v1, e.y, sumi);
    return r.d * __int_as_float(e.z) * float(sumi);
}

// Rows of 80 blocks (n_in 2560) by one warp each, the next row's weights loaded while this one is computed.  Lane L
// stands for native_mmvq's threads L, L + 32, L + 64, L + 96 (kbx = tid / 4, += 32; kqs = 2 (tid % 4)): virtual thread
// v takes blocks L / 4 + 8 v + 32 i below 80, three for v = 0, 1 and two for v = 2, 3.  Its sums are added in the
// cross-warp order before the same xor tree.
template <int NC>
__device__ __forceinline__ void q80_rows_80(const Q80Block* __restrict__ w, const int4* __restrict__ xe,
                                            float* __restrict__ y, int n_out, int row0, int stride, int lane,
                                            W8 (&cur)[10]) {
    constexpr int NCH = 10;
    const int a = lane / 4, p = lane % 4, kqs = 2 * p;
    const int kb[NCH] = {a, a + 32, a + 64, a + 8, a + 40, a + 72, a + 16, a + 48, a + 24, a + 56};
    constexpr int vt[NCH] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3};
    for (int row = row0; row < n_out; row += stride) {
        const int nrow = row + stride;
        W8 nxt[NCH];
        if (nrow < n_out) {
#pragma unroll
            for (int c = 0; c < NCH; ++c) nxt[c] = q80_load(w + (size_t) nrow * 80 + kb[c], kqs);
        }
        float s[VT][NC];
#pragma unroll
        for (int v = 0; v < VT; ++v)
#pragma unroll
            for (int j = 0; j < NC; ++j) s[v][j] = 0.0f;
#pragma unroll
        for (int c = 0; c < NCH; ++c)
#pragma unroll
            for (int j = 0; j < NC; ++j) s[vt[c]][j] += q80_apply(cur[c], xe + (size_t) j * 80 * 4, kb[c], p);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float t = s[0][j];
#pragma unroll
            for (int v = 1; v < VT; ++v) t += s[v][j];
            t = warp_sum(t);
            if (lane == 0) y[(size_t) j * n_out + row] = t;
        }
#pragma unroll
        for (int c = 0; c < NCH; ++c) cur[c] = nxt[c];
    }
}

template <int NC>
__global__ void __launch_bounds__(WPB * WARP) q80_head_kernel(const Q80Block* __restrict__ w, const Q81Block* __restrict__ x,
                                                              float* __restrict__ y, int n_out) {
    extern __shared__ __align__(16) int4 xe[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row0 = blockIdx.x * WPB + warp, stride = gridDim.x * WPB;
    W8 cur[10];
    if (row0 < n_out) {   // the weights are constant: the warp's first row is loaded before the wait
        const int a = lane / 4, kqs = 2 * (lane % 4);
        const int kb[10] = {a, a + 32, a + 64, a + 8, a + 40, a + 72, a + 16, a + 48, a + 24, a + 56};
#pragma unroll
        for (int c = 0; c < 10; ++c) cur[c] = q80_load(w + (size_t) row0 * 80 + kb[c], kqs);
    }
    pdl_wait();   // no early dependents: a kernel let in now would hold SM room the resident wave needs
    for (int i = threadIdx.x; i < NC * 80 * 4; i += blockDim.x) {
        const int b = i / 4, p = i % 4;
        const Q81Block* blk = x + b;
        const int* q8 = reinterpret_cast<const int*>(blk->qs) + 2 * p;
        const unsigned dsb = __ldg(reinterpret_cast<const unsigned*>(&blk->ds));
        xe[i] = make_int4(__ldg(q8), __ldg(q8 + 1), __float_as_int(__low2float(*reinterpret_cast<const half2*>(&dsb))), 0);
    }
    __syncthreads();
    q80_rows_80<NC>(w, xe, y, n_out, row0, stride, lane, cur);
}

template <int NC>
void launch_nc(const void* w, const void* x, float* y, int n_out, cudaStream_t s) {
    // one resident wave: every warp walks rows with the grid's stride
    static std::atomic<int> grid_max[64];
    static std::atomic<bool> smem_set[64];
    const int dev = opt_in_device("q80_head");
    const size_t smem = (size_t) NC * 80 * 4 * sizeof(int4);
    if (!smem_set[dev].load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(opt_in_mutex());
        if (!smem_set[dev].load(std::memory_order_relaxed)) {
            int sms = 0, per_sm = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            opt_in_check(cudaFuncSetAttribute(q80_head_kernel<NC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem),
                         "q80_head");
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, q80_head_kernel<NC>, WPB * WARP, smem);
            grid_max[dev].store((sms > 0 ? sms : 1) * (per_sm > 0 ? per_sm : 1), std::memory_order_relaxed);
            smem_set[dev].store(true, std::memory_order_release);
        }
    }
    const long long want = ((long long) n_out + WPB - 1) / WPB;
    const long long gm = grid_max[dev].load(std::memory_order_relaxed);
    const unsigned grid = (unsigned) (want < gm ? want : gm);
    launch_pdl(q80_head_kernel<NC>, dim3(grid), dim3(WPB * WARP), smem, s, (const Q80Block*) w, (const Q81Block*) x, y,
               n_out);
}

}  // namespace

bool q80_head_supported(int n_in, int ncols) { return n_in == 80 * QK && ncols >= 1 && ncols <= 8; }

void q80_head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!q80_head_supported(n_in, ncols) || n_out <= 0 || !weights || !x_q8_1 || !y) {
        basalt::log::error("basalt", "q80_head_mmvq: unsupported shape (n_in %d, ncols %d)", n_in, ncols);
        std::exit(1);
    }
    const auto s = (cudaStream_t) stream;
    switch (ncols) {
        case 1: launch_nc<1>(weights, x_q8_1, y, n_out, s); break;
        case 2: launch_nc<2>(weights, x_q8_1, y, n_out, s); break;
        case 3: launch_nc<3>(weights, x_q8_1, y, n_out, s); break;
        case 4: launch_nc<4>(weights, x_q8_1, y, n_out, s); break;
        case 5: launch_nc<5>(weights, x_q8_1, y, n_out, s); break;
        case 6: launch_nc<6>(weights, x_q8_1, y, n_out, s); break;
        case 7: launch_nc<7>(weights, x_q8_1, y, n_out, s); break;
        default: launch_nc<8>(weights, x_q8_1, y, n_out, s); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "q80_head_mmvq: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace basalt::kernels
