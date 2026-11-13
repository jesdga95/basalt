// src/kernels/cuda/window_gemm.cu - see include/basalt/kernels/window_gemm.hpp: the activation packs, the plan and the
// dispatch to the kernel instantiations (window_gemm_f8.cu, window_gemm_nvfp4.cu).
#include "window_gemm_kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>

#include "basalt/log.hpp"

namespace basalt::kernels::wgemm {
namespace {

using detail::Args;
using detail::Kernel;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
[[noreturn]] void refuse(const char* what, const Call& c) {
    basalt::log::error("basalt", "window gemm: %s (%lld x %lld)", what, (long long) c.m[0].rows, (long long) c.K);
    std::exit(1);
}

// ---- activations

constexpr int kPackThreads = 512;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// as dense_ops' block_sum (the same order, so norm_pack matches rms_norm_rows bit for bit)
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += red[i];
    return t;
}
__device__ __forceinline__ uint16_t bf16_bits(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

// The pack kernels start their dependents at once (the products stage weights meanwhile) and wait for their inputs.
__global__ void __launch_bounds__(256) pack_kernel(const float* x, int64_t ld_x, int64_t K, int64_t n, uint16_t* out) {
    pdl_launch_dependents();
    pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t t = i / K, c = i - t * K;
    out[i] = bf16_bits(x[t * ld_x + c]);
}

__device__ __forceinline__ float widen(float v) { return v; }
__device__ __forceinline__ float widen(uint16_t v) { return __bfloat162float(__ushort_as_bfloat16(v)); }

template <class X>
__global__ void __launch_bounds__(kPackThreads) norm_pack_kernel(const X* x, const float* __restrict__ w, float eps,
                                                                 int64_t K, uint16_t* out, float* xn) {
    __shared__ float red[32];
    pdl_launch_dependents();
    pdl_wait();
    const X* xr = x + (int64_t) blockIdx.x * K;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < K; c += blockDim.x) ss = fmaf(widen(xr[c]), widen(xr[c]), ss);
    const float inv = rsqrtf(block_sum(ss, red) / (float) K + eps);
    float* yr = xn ? xn + (int64_t) blockIdx.x * K : nullptr;
    uint16_t* o = out + (int64_t) blockIdx.x * K;
    for (int64_t c = threadIdx.x; c < K; c += blockDim.x) {
        const float v = widen(xr[c]) * inv * w[c];
        if (yr) yr[c] = v;
        o[c] = bf16_bits(v);
    }
}

// ---- the plan

struct Plan {
    const Kernel* k = nullptr;
    int tiles = 0, ks = 1, cps = 0, nchunks = 0, grid = 1;
};

int sm_count() {
    static int n = [] {
        int dev = 0, v = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
        return v > 0 ? v : 1;
    }();
    return n;
}

// blocks of k that fit on the device at once (its smem allowance raised first)
int resident(const Kernel& k) {
    static std::mutex mu;
    static std::map<const void*, int> known;
    std::lock_guard<std::mutex> lock(mu);
    const auto it = known.find((const void*) k.fn);
    if (it != known.end()) return it->second;
    cudaFuncSetAttribute(k.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, k.smem);
    int per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k.fn, k.threads, k.smem);
    check("window gemm: occupancy");
    const int r = std::max(1, per_sm) * sm_count();
    known[(const void*) k.fn] = r;
    return r;
}

const Kernel& kernel_for(const Call& c, int nt) {
    const bool swiglu = c.epi == Epilogue::SwiGLU;
    if (c.fmt != Format::Nvfp4 && swiglu) refuse("an FP8 or BF16 SwiGLU pair", c);
    if (c.fmt == Format::F8) return detail::f8(nt);
    if (c.fmt == Format::Bf16) return detail::bf16(nt);
    if (c.fmt == Format::Q8) return detail::q8(nt);
    return swiglu ? detail::nvfp4_swiglu(nt) : detail::nvfp4(nt);
}

// The slice count from the call's shapes and the device alone (the T = 1..8 kernel's residency for every T): the fewest
// rounds of items over the resident blocks times the chunks an item streams, plus its share of the slices' sums.
Plan plan_for(const Call& c) {
    Plan p;
    p.k = &kernel_for(c, 1);
    const int nm = p.k->nm, rs = p.k->rs;
    p.tiles = 0;
    if (nm == 2) p.tiles = (int) ((c.m[0].rows + rs - 1) / rs);
    else
        for (int i = 0; i < c.n; ++i) p.tiles += (int) ((c.m[i].rows + rs - 1) / rs);
    p.nchunks = (int) (c.K / p.k->ch);
    const int64_t R = resident(*p.k);
    const int cb = c.fmt == Format::F8 || c.fmt == Format::Q8 ? p.k->ch : c.fmt == Format::Bf16 ? 2 * p.k->ch : p.k->ch / 2;   // weight bytes per row per chunk
    const double sums = 1.0 + 64.0 / cb;   // an item's fixed cost in chunks: its pipeline turn and its slice's sums
    double best = 1e30;
    for (int ks = 1; ks <= std::min(p.nchunks, 64); ++ks) {
        const int cps = (p.nchunks + ks - 1) / ks, kse = (p.nchunks + cps - 1) / cps;
        if (kse != ks) continue;
        const int64_t items = (int64_t) p.tiles * kse;
        // plus the tile's last block's walk over the slices' sums, about a quarter chunk a slice at 16 rows (40 slices
        // of a 256-row matrix cost 15.7 us at T = 16 against 7.7 at 10)
        const double tail = kse > 1 ? 0.25 * kse : 0.0;
        const double cost = (double) ((items + R - 1) / R) * (cps + (kse > 1 ? sums : 0.0)) + tail;
        if (cost < best - 1e-9) {
            best = cost;
            p.ks = kse;
            p.cps = cps;
        }
    }
    p.grid = (int) std::min<int64_t>((int64_t) p.tiles * p.ks, R);
    return p;
}

int64_t part_rows(const Call& c) {
    if (c.epi == Epilogue::SwiGLU) return 2 * c.m[0].rows;
    int64_t r = 0;
    for (int i = 0; i < c.n; ++i) r += c.m[i].rows;
    return r;
}

// one thread per (row, 32-value block): the fp16 scale to the scale plane, the 32 codes to the code plane
__global__ void q8_planes_kernel(const uint8_t* __restrict__ src, int64_t rows, int64_t K, uint8_t* __restrict__ dst) {
    const int64_t nb = K / 32, i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nb) return;
    const int64_t r = i / nb, b = i - r * nb;
    const uint8_t* blk = src + i * 34;
    uint8_t* codes = dst + r * K + b * 32;
    for (int k = 0; k < 32; ++k) codes[k] = blk[2 + k];
    uint8_t* sc = dst + rows * K + (r * nb + b) * 2;
    sc[0] = blk[0];
    sc[1] = blk[1];
}

}  // namespace

void q8_planes(const void* blocks, int64_t rows, int64_t K, void* planes, void* stream) {
    if (K % 32 != 0) refuse("Q8 planes of K not a multiple of 32", Call{});
    const int64_t n = rows * (K / 32);
    q8_planes_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>((const uint8_t*) blocks, rows, K, (uint8_t*) planes);
    check("window gemm: q8 planes");
}

std::size_t act_bytes(int64_t K) { return (size_t) kRows * K * sizeof(uint16_t); }

void pack(const float* x, int64_t ld_x, int64_t K, int T, void* act, void* stream) {
    if (T <= 0) return;
    const int64_t n = (int64_t) T * K;
    launch_pdl(pack_kernel, dim3((unsigned) ((n + 255) / 256)), dim3(256), 0, (cudaStream_t) stream, x, ld_x, K, n, (uint16_t*) act);
    check("window gemm: pack");
}

void norm_pack(const float* x, const float* w, float eps, int64_t K, int T, void* act, float* xn, void* stream) {
    if (T <= 0) return;
    launch_pdl(norm_pack_kernel<float>, dim3(T), dim3(kPackThreads), 0, (cudaStream_t) stream, x, w, eps, K, (uint16_t*) act, xn);
    check("window gemm: norm_pack");
}

void norm_pack(const uint16_t* x, const float* w, float eps, int64_t K, int T, void* act, float* xn, void* stream) {
    if (T <= 0) return;
    launch_pdl(norm_pack_kernel<uint16_t>, dim3(T), dim3(kPackThreads), 0, (cudaStream_t) stream, x, w, eps, K, (uint16_t*) act, xn);
    check("window gemm: norm_pack");
}

void workspace_need(const Call& c, std::size_t& partial_floats, std::size_t& tickets) {
    const Plan p = plan_for(c);
    partial_floats = p.ks > 1 ? (size_t) p.ks * kRows * part_rows(c) : 0;
    tickets = (size_t) p.tiles;
}

void gemm(const Call& c, const void* act, int T, const Workspace& ws, void* stream) {
    if (T <= 0) return;
    if (T > kRows) refuse("more rows than a window holds", c);
    if (c.n < 1 || c.n > kMaxMatrices || (c.epi == Epilogue::SwiGLU && (c.n != 2 || c.m[1].rows != c.m[0].rows)))
        refuse("matrices", c);
    if (c.act_out && c.epi != Epilogue::SwiGLU) refuse("bf16 rows out of a product other than SwiGLU", c);
    const Plan p = plan_for(c);
    if (c.K % p.k->ch != 0) refuse("K is not a multiple of the chunk", c);
    for (int i = 0; i < c.n; ++i) {
        if (((uintptr_t) c.m[i].w & 15) != 0) refuse("unaligned matrix", c);
        if (c.fmt == Format::Nvfp4 && c.m[i].rows % 128 != 0) refuse("NVFP4 rows not a multiple of 128", c);
    }
    const int64_t pr = part_rows(c);
    if (p.ks > 1 && ((size_t) p.ks * kRows * pr > ws.partial_floats || (size_t) p.tiles > ws.tickets_n))
        refuse("workspace too small", c);
    const Kernel& k = kernel_for(c, T > 8 ? 2 : 1);
    resident(k);   // its smem allowance
    Args a{};
    int t0 = 0;
    int64_t off = 0;
    for (int i = 0; i < c.n; ++i) {
        a.w[i] = c.m[i].w;
        a.sp[i] = c.m[i].scales ? c.m[i].scales : c.m[i].w + c.m[i].rows * (c.fmt == Format::Q8 ? c.K : c.K / 2);
        a.rows[i] = c.m[i].rows;
        a.scale[i] = c.m[i].scale;
        a.y[i] = c.m[i].y;
        a.y16[i] = c.m[i].y16;
        a.ld_y[i] = c.m[i].ld_y;
        a.row_off[i] = off;
        a.tile0[i] = t0;
        off += c.m[i].rows;
        if (c.epi != Epilogue::SwiGLU) t0 += (int) ((c.m[i].rows + k.rs - 1) / k.rs);
    }
    a.tile0[c.n] = c.epi == Epilogue::SwiGLU ? p.tiles : t0;
    a.n = c.epi == Epilogue::SwiGLU ? 1 : c.n;
    a.K = c.K;
    a.part_rows = pr;
    a.T = T;
    a.tiles = p.tiles;
    a.ks = p.ks;
    a.cps = p.cps;
    a.nchunks = p.nchunks;
    a.epi = (int) c.epi;
    a.act = (const uint16_t*) act;
    a.act_out = (uint16_t*) c.act_out;
    a.partial = ws.partial;
    a.tickets = ws.tickets;
    launch_pdl(k.fn, dim3((unsigned) p.grid), dim3((unsigned) k.threads), (size_t) k.smem,
               (cudaStream_t) stream, a);
    check("window gemm");
}

}  // namespace basalt::kernels::wgemm
