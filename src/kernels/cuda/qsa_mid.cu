// src/kernels/cuda/qsa_mid.cu - see include/basalt/kernels/qsa_mid.hpp.
//
// Its own translation unit: dense_tc.cu's device code without its host side.  Built WITHOUT fast math, like it.
#define FZ_DEV_ONLY
#include "dense_tc.cu"
#include "basalt/kernels/qsa_mid.hpp"
#include "smem_opt_in.cuh"

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace basalt::kernels {
namespace {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// dense_tc_tile_block over the tiles of one half: the block's tile i is the half's tile j = blk + i * nblk, which is
// the projection's tile 32 (j / 16) + 16 half + j % 16; every step of a tile is dense_tc_tile_block's.  Two tiles in
// flight per block.  early: the dependents are let in at the start, else once the last tile is read.
template<int TYPE>
__device__ __forceinline__ void qproj_half_block(int blk, int nblk, int half, bool early, uint8_t* smem,
                                                 const uint8_t* __restrict__ w, const uint8_t* __restrict__ x,
                                                 float* __restrict__ y, int n_in, int n_out, int ncols) {
    constexpr int NS = 2;
    constexpr int SB = Fmt<TYPE>::SB, CB = 2 * SB;
    uint64_t* bar = reinterpret_cast<uint64_t*>(smem);
    float* red = reinterpret_cast<float*>(smem + 128);   // [warp][row][column]
    uint8_t* slots = smem + 128 + 8 * 16 * kMaxCols * 4;
    const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int hb = wp & 3, sbi = hb >> 1, n = hb & 1;
    const int nchunk = n_in / 512, nb32 = n_in / 32;
    const int row_bytes = (n_in / 256) * SB, tb = 16 * row_bytes, ts = 16 * row_bytes;
    const int ntiles = n_out / 32;   // the half's tiles
    const int mine = blk < ntiles ? (ntiles - 1 - blk) / nblk + 1 : 0;
    if (early) pdl_launch_dependents();
    auto tile_of = [&](int i) {
        const int j = blk + i * nblk;
        return 32 * (j / 16) + 16 * half + j % 16;
    };
    // tile i into slot i % NS by warp 0 (whole tiles: n_out is a multiple of 16)
    auto issue = [&](int i) {
        const int tile = tile_of(i);
        const uint32_t sb = (uint32_t) __cvta_generic_to_shared(slots + (size_t) (i % NS) * ts);
        const uint32_t bb = (uint32_t) __cvta_generic_to_shared(bar + (i % NS));
        const uint32_t bytes = (uint32_t) ((16 * row_bytes + 15) & ~15);
        if (lane == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bb), "r"(bytes) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         ::"r"(sb), "l"(w + (size_t) tile * tb), "r"(bytes), "r"(bb) : "memory");
        }
    };
    if (wp == 0) {
        if (lane == 0) {
            for (int q = 0; q < NS; ++q)
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t) __cvta_generic_to_shared(bar + q)));
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        __syncwarp();
        for (int i = 0; i < NS && i < mine; ++i) issue(i);
    }
    __syncthreads();
    pdl_wait();
    const uint32_t* tbl = reinterpret_cast<const uint32_t*>(dtc_iq4nl);
    uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    if constexpr (TYPE == 23) { t0 = tbl[0]; t1 = tbl[1]; t2 = tbl[2]; t3 = tbl[3]; }
    for (int i = 0; i < mine; ++i) {
        const int row0 = tile_of(i) * 16;
        {
            uint32_t done = 0;
            const uint32_t bb = (uint32_t) __cvta_generic_to_shared(bar + (i % NS)), par = (uint32_t) (i / NS) & 1u;
            do {
                asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                             " selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(bb), "r"(par) : "memory");
            } while (!done);
        }
        const uint8_t* slot = slots + (size_t) (i % NS) * ts;
        const uint8_t* rowA = slot + g * row_bytes + sbi * SB;
        const uint8_t* rowB = slot + (g + 8) * row_bytes + sbi * SB;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int c = wp >> 2; c < nchunk; c += 2) {
            XFrag xf;
#pragma unroll
            for (int k4 = 0; k4 < 4; ++k4) {
                const int k = 16 * c + 4 * hb + k4;
                xf.b0[k4] = xf.b1[k4] = 0;
                if (g < ncols) {
                    const uint8_t* xb = x + ((size_t) g * nb32 + k) * kQ81;
                    xf.b0[k4] = __ldg(reinterpret_cast<const unsigned int*>(xb + 4) + t);
                    xf.b1[k4] = __ldg(reinterpret_cast<const unsigned int*>(xb + 20) + t);
                }
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int col = 2 * t + e;
                    xf.d8[k4][e] = col < ncols ? __low2float(__ldg(reinterpret_cast<const half2*>(
                                                     x + ((size_t) col * nb32 + k) * kQ81)))
                                               : 0.0f;
                }
            }
            const uint8_t* const p[2] = {rowA + c * CB, rowB + c * CB};
            half_sb_terms<TYPE>(acc, p, n, t, xf, t0, t1, t2, t3);
        }
        float* rw = red + wp * 16 * kMaxCols;
        rw[g * kMaxCols + 2 * t] = acc[0];
        rw[g * kMaxCols + 2 * t + 1] = acc[1];
        rw[(g + 8) * kMaxCols + 2 * t] = acc[2];
        rw[(g + 8) * kMaxCols + 2 * t + 1] = acc[3];
        if (!early && i == mine - 1) pdl_launch_dependents();
        __syncthreads();   // the slot is read and the warp sums are in
        if (wp == 0 && i + NS < mine) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            issue(i + NS);
        }
        if (tid < 16 * kMaxCols) {
            const int row = tid / kMaxCols, col = tid % kMaxCols;
            if (col < ncols) {
                const float* rr = red + row * kMaxCols + col;
                constexpr int WS = 16 * kMaxCols;   // one warp's sums
                const float v = ((rr[0] + rr[4 * WS]) + (rr[WS] + rr[5 * WS])) +
                                ((rr[2 * WS] + rr[6 * WS]) + (rr[3 * WS] + rr[7 * WS]));
                y[(size_t) col * n_out + row0 + row] = v;
            }
        }
        __syncthreads();   // the sums are read
    }
    if (!early && mine == 0) pdl_launch_dependents();
}
#endif

template<int TYPE>
__global__ void __launch_bounds__(256) qproj_half_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ x,
                                                         float* __restrict__ y, int n_in, int n_out, int ncols, int half) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) uint8_t smem_qh[];
    qproj_half_block<TYPE>((int) blockIdx.x, (int) gridDim.x, half, half == 1, smem_qh, w, x, y, n_in, n_out, ncols);
#elif defined(__CUDA_ARCH__)
    __trap();
#endif
}

int sb_of(int type) {
    switch (type) {
        case 8: return Fmt<8>::SB;
        case 12: return Fmt<12>::SB;
        case 13: return Fmt<13>::SB;
        case 14: return Fmt<14>::SB;
        case 23: return Fmt<23>::SB;
        case 11: return Fmt<11>::SB;
        case 42: return Fmt<42>::SB;
        default: return 0;
    }
}

int sms_now() {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return sms > 0 ? sms : 1;
}

template<int TYPE>
void launch_half(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, int half, int blocks,
                 cudaStream_t s) {
    const int smem = 128 + 8 * 16 * kMaxCols * 4 + 2 * 16 * ((n_in / 256) * Fmt<TYPE>::SB);
    static std::atomic<bool> attr[64];
    const int dev = opt_in_device("qsa_qproj_half");
    if (!attr[dev].load(std::memory_order_acquire)) {
        opt_in_check(cudaFuncSetAttribute(qproj_half_kernel<TYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024),
                     "qsa_qproj_half");
        attr[dev].store(true, std::memory_order_release);
    }
    const int ntiles = n_out / 32;
    // the gate half's share of the SMs leaves the attention's selection and chunk blocks the rest
    int grid = blocks > 0 ? blocks : half == 0 ? sms_now() : (3 * sms_now() + 4) / 8;
    if (grid > ntiles) grid = ntiles;
    launch_pdl(qproj_half_kernel<TYPE>, dim3((unsigned) grid), dim3(256), (size_t) smem, s, (const uint8_t*) w,
               (const uint8_t*) x, y, n_in, n_out, ncols, half);
}

}  // namespace

bool qsa_qproj_half_supported(int type, const void* w, int n_in, int n_out) {
    if (!(dense_tc_mask() & 8) || !dense_tc_supported(type, n_in, n_out, w) || !dense_tc_type_on(type)) return false;
    const int sb = sb_of(type);
    if (sb == 0 || n_in % 512 != 0 || n_out % 512 != 0 || ((uintptr_t) w & 15) != 0) return false;
    // dense_tc's whole-tile layout for this shape: more tiles than SMs (not the one-tile path), two tiles in 90 KB; and
    // SMs enough for the gate half's share beside the attention
    const int sms = sms_now();
    return sms >= 128 && (n_out + 15) / 16 > sms && 2 * 16 * (n_in / 256) * sb <= 90 * 1024;
}

void qsa_qproj_half(int type, const void* w, const void* x, float* y, int n_in, int n_out, int ncols, int half,
                    int blocks, void* stream) {
    if (!qsa_qproj_half_supported(type, w, n_in, n_out) || ncols < 1 || ncols > kMaxCols || (half != 0 && half != 1) ||
        ((uintptr_t) x & 3) != 0) {
        basalt::log::error("basalt", "qsa_qproj_half: type %d n_in %d n_out %d ncols %d not supported", type, n_in, n_out,
                           ncols);
        std::exit(1);
    }
    const auto s = (cudaStream_t) stream;
    switch (type) {
        case 8: launch_half<8>(w, x, y, n_in, n_out, ncols, half, blocks, s); break;
        case 12: launch_half<12>(w, x, y, n_in, n_out, ncols, half, blocks, s); break;
        case 13: launch_half<13>(w, x, y, n_in, n_out, ncols, half, blocks, s); break;
        case 14: launch_half<14>(w, x, y, n_in, n_out, ncols, half, blocks, s); break;
        case 23: launch_half<23>(w, x, y, n_in, n_out, ncols, half, blocks, s); break;
        case 11: launch_half<11>(w, x, y, n_in, n_out, ncols, half, blocks, s); break;
        default: launch_half<42>(w, x, y, n_in, n_out, ncols, half, blocks, s); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "qsa_qproj_half: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace basalt::kernels
