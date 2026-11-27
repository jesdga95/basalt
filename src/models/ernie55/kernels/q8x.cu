// src/models/ernie55/kernels/q8x.cu - a decode window's Q8_0 products, several matrices of one width per launch, with
// the bits of kernels::native_q8_0_mmvq: each (row, column) sum is the one-row exact layout's (thread tid of 128 adds
// d_w * d_x * float(sumi) of the blocks kbx = tid / 4 (mod 32) at ints 2 (tid % 4) and 2 (tid % 4) + 1, ascending
// kbx; warp 0 adds warps 1, 2, 3 in order; then the warp butterfly). A block takes R rows of a matrix with every load
// of its weights issued before pdl_wait (they are constant), so a short row (1024 values: one load a thread) keeps
// enough bytes in flight, and a launch carries up to kQ8xMats matrices that read the same width.
#include "basalt/kernels/dp4a.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/ernie55/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace basalt::models::ernie55 {
namespace {

namespace bk = basalt::kernels;

struct Q80B {
    half d;
    int8_t qs[32];
};
struct Q81B {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q80B) == 34 && sizeof(Q81B) == 36);

struct Q8xArgs {
    const Q80B* w[kQ8xMats];
    const Q81B* x[kQ8xMats];
    float* y[kQ8xMats];
    int n_out[kQ8xMats];
    int end[kQ8xMats];   // a matrix's last block + 1, cumulative
    int n;
};

__device__ __forceinline__ int word_b2(const int8_t* qs, int i) {
    const auto* p = reinterpret_cast<const uint16_t*>(qs);
    int v = p[2 * i];
    v |= p[2 * i + 1] << 16;
    return v;
}

__device__ __forceinline__ float bfly(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o, 32);
    return x;
}

// IT: blocks of 32 Q8_0 blocks a row holds (n_in / 1024); R rows a block; NC columns.
template <int NC, int R, int IT>
__global__ void __launch_bounds__(128) q8x_kernel(Q8xArgs a) {
    int m = 0;
    while (m < a.n - 1 && (int) blockIdx.x >= a.end[m]) ++m;
    const int bx = (int) blockIdx.x - (m ? a.end[m - 1] : 0);
    const Q80B* __restrict__ w = a.w[m];
    const int n_out = a.n_out[m], row0 = bx * R;
    const int tid = 32 * (int) threadIdx.y + (int) threadIdx.x;
    const int kbx0 = tid >> 2, kqs = 2 * (tid & 3);
    constexpr int bpr = IT * 32;
    int v0[R][IT], v1[R][IT];
    float d0[R][IT];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, n_out - 1);
#pragma unroll
        for (int it = 0; it < IT; ++it) {
            const Q80B* b = w + (size_t) row * bpr + kbx0 + 32 * it;
            v0[r][it] = word_b2(b->qs, kqs);
            v1[r][it] = word_b2(b->qs, kqs + 1);
            d0[r][it] = __half2float(b->d);
        }
    }
    bk::pdl_launch_dependents();
    bk::pdl_wait();
    const Q81B* __restrict__ x = a.x[m];
    float tmp[NC][R];
#pragma unroll
    for (int j = 0; j < NC; ++j)
#pragma unroll
        for (int r = 0; r < R; ++r) tmp[j][r] = 0.0f;
#pragma unroll
    for (int it = 0; it < IT; ++it) {
        const int kbx = kbx0 + 32 * it;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const Q81B* xj = x + (size_t) j * bpr + kbx;
            const int* u = reinterpret_cast<const int*>(xj->qs);
            const int u0 = u[kqs], u1 = u[kqs + 1];
            const float d1 = __low2float(xj->ds);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int sumi = 0;
                sumi = DP4A(v0[r][it], u0, sumi);
                sumi = DP4A(v1[r][it], u1, sumi);
                tmp[j][r] += d0[r][it] * d1 * float(sumi);
            }
        }
    }
    __shared__ float part[3][NC * R][32];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j)
#pragma unroll
            for (int r = 0; r < R; ++r) part[threadIdx.y - 1][j * R + r][threadIdx.x] = tmp[j][r];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
    float* __restrict__ y = a.y[m];
#pragma unroll
    for (int j = 0; j < NC; ++j)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = tmp[j][r];
#pragma unroll
            for (int l = 0; l < 3; ++l) v += part[l][j * R + r][threadIdx.x];
            v = bfly(v);
            if (threadIdx.x == 0 && row0 + r < n_out) y[(size_t) j * n_out + row0 + r] = v;
        }
}

// rows a block takes: its weight loads (rows x IT) at most 8 a thread, rows x columns at most 16
constexpr int q8x_rows(int nc, int it) {
    const int r = it >= 8 ? 1 : 8 / it;
    return nc * r <= 16 ? r : (16 / nc > 0 ? 16 / nc : 1);
}

template <int NC, int IT>
void launch_it(const Q8xArgs& a, int blocks, cudaStream_t s) {
    bk::launch_pdl(q8x_kernel<NC, q8x_rows(NC, IT), IT>, dim3((unsigned) blocks), dim3(32, 4), 0, s, a);
}

template <int IT>
void launch_nc(Q8xArgs a, const Q8xMat* mats, int n, int ncols, cudaStream_t s) {
    const int R = q8x_rows(ncols, IT);
    int end = 0;
    for (int i = 0; i < n; ++i) {
        end += (mats[i].n_out + R - 1) / R;
        a.end[i] = end;
    }
    switch (ncols) {
    case 1: launch_it<1, IT>(a, end, s); break;
    case 2: launch_it<2, IT>(a, end, s); break;
    case 3: launch_it<3, IT>(a, end, s); break;
    case 4: launch_it<4, IT>(a, end, s); break;
    case 5: launch_it<5, IT>(a, end, s); break;
    case 6: launch_it<6, IT>(a, end, s); break;
    case 7: launch_it<7, IT>(a, end, s); break;
    default: launch_it<8, IT>(a, end, s); break;
    }
}

}  // namespace

void q8x_mmvq(const Q8xMat* mats, int n, int n_in, int ncols, void* stream) {
    if (n < 1 || n > kQ8xMats || ncols < 1 || ncols > 8 || n_in % 1024 != 0)
        throw std::runtime_error("ernie55 q8x_mmvq: matrices, columns or width");
    Q8xArgs a{};
    a.n = n;
    for (int i = 0; i < n; ++i) {
        if (mats[i].n_out <= 0) throw std::runtime_error("ernie55 q8x_mmvq: rows");
        a.w[i] = static_cast<const Q80B*>(mats[i].w);
        a.x[i] = static_cast<const Q81B*>(mats[i].x);
        a.y[i] = mats[i].y;
        a.n_out[i] = mats[i].n_out;
    }
    cudaStream_t s = (cudaStream_t) stream;
    switch (n_in / 1024) {
    case 1: launch_nc<1>(a, mats, n, ncols, s); break;
    case 2: launch_nc<2>(a, mats, n, ncols, s); break;
    case 4: launch_nc<4>(a, mats, n, ncols, s); break;
    case 8: launch_nc<8>(a, mats, n, ncols, s); break;
    case 16: launch_nc<16>(a, mats, n, ncols, s); break;
    default: throw std::runtime_error("ernie55 q8x_mmvq: width " + std::to_string(n_in));
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("ernie55 q8x_mmvq: ") + cudaGetErrorString(e));
}

void q8_mmvq(const Q8xMat* mats, int n, int n_in, int ncols, void* stream) {
    // q8x is as fast or faster for these widths at three columns or fewer and wherever matrices share a launch; one
    // matrix of a short width at four columns or more takes native_q8_0_mmvq's tensor core path
    if (n_in % 1024 != 0 || n_in > 4096 || (n == 1 && ncols >= 4 && n_in <= 2048)) {
        for (int i = 0; i < n; ++i)
            kernels::native_q8_0_mmvq(mats[i].w, mats[i].x, mats[i].y, n_in, mats[i].n_out, ncols, stream);
        return;
    }
    bk::PdlScope pdl(true);
    q8x_mmvq(mats, n, n_in, ncols, stream);
}

}  // namespace basalt::models::ernie55
