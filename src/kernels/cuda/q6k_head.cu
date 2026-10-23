// src/kernels/cuda/q6k_head.cu - see include/basalt/kernels/q6k_head.hpp.
//
// Built with --use_fast_math like native_mmvq.cu, whose Q6_K x q8_1 dot this repeats.
#include "basalt/kernels/q6k_head.hpp"
#include "basalt/kernels/dp4a.hpp"
#include "basalt/kernels/pdl.hpp"
#include "head_proxy.cuh"
#include "smem_opt_in.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int QK = 256, Q8K = 32, WARP = 32;
constexpr int VT = 4;      // virtual threads per lane: native_mmvq's four warps of one row
constexpr int WPB = 8;     // warps (rows in flight) per block
constexpr int BPR = 10;    // Q6_K blocks per row (n_in 2560)

struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q6KBlock) == 210 && sizeof(Q81Block) == 36);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    return x;
}

// the 16-bit loads of native_mmvq's load_int_b2: blocks are only two-byte aligned
__device__ __forceinline__ int load_b2(const void* ptr, int i32) {
    const uint16_t* x = static_cast<const uint16_t*>(ptr);
    return (int) x[2 * i32] | ((int) x[2 * i32 + 1] << 16);
}

// native_mmvq's Q6KTraits::load for lane `iqs`: the weight half of one virtual thread's dot
struct W6 {
    int vl, vh;
    int sc;      // the two signed scales, bytes 0 and 1
    float d;
};
__device__ __forceinline__ W6 q6_load(const Q6KBlock* __restrict__ b, int iqs) {
    W6 r;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    r.vl = load_b2(b->ql, iqs);
    r.vh = load_b2(b->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    r.sc = (int) (uint8_t) b->scales[scale_offset] | ((int) (uint8_t) b->scales[scale_offset + 4] << 8);
    r.d = __half2float(b->d);
    return r;
}
// a column in shared memory: per q8_1 block b and position p (0..7), {q[p], the block's scale as float}
__device__ __forceinline__ float q6_apply(const W6& r, const int2* __restrict__ x, int kby, int iqs) {
    const int bo = 4 * (iqs / 16) + (iqs % 16) / 8;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int2 e = x[(kby + bo + 2 * i) * 8 + iqs % 8];
        const int sc = (int) (int8_t) (r.sc >> (8 * i));
        const int vil = (r.vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((r.vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += __int_as_float(e.y) * (DP4A(vi, e.x, 0) * sc);
    }
    return r.d * sumf;
}

// the running (max, its first row, sum of exp(l - max)) of the logits a warp has produced
struct Top {
    float m;
    int idx;   // 0x7fffffff: no rows yet
    float s;
    __device__ __forceinline__ void init() {
        m = __int_as_float(0xff800000);
        idx = 0x7fffffff;
        s = 0.0f;
    }
    __device__ __forceinline__ void take(float l, int row) {
        if (l > m) {
            s = s * __expf(m - l) + 1.0f;
            m = l;
            idx = row;
        } else {
            s += __expf(l - m);
            if (l == m && row < idx) idx = row;
        }
    }
    __device__ __forceinline__ void merge(const Top& o) {
        if (o.idx == 0x7fffffff) return;   // no rows
        if (idx == 0x7fffffff || o.m > m) {
            s = o.s + s * __expf(m - o.m);
            m = o.m;
            idx = o.idx;
        } else {
            s += o.s * __expf(o.m - m);
            if (o.m == m && o.idx < idx) idx = o.idx;
        }
    }
};

// Lane L stands for native_mmvq's threads L, L+32, L+64, L+96, whose blocks are kbx = v, v + 4, v + 8 (v = tid / 32);
// their sums are added in the cross-warp order before the same xor tree.  The next row's weights are loaded while
// this one is computed.
template <int NC, bool TOP = false>
__device__ __forceinline__ void q6k_rows(const Q6KBlock* __restrict__ w, const int2* const* xc, float* __restrict__ y,
                                         int n_out, int row0, int stride, int lane, Top* top = nullptr,
                                         const int32_t* __restrict__ map = nullptr) {
    constexpr int kb[BPR] = {0, 4, 8, 1, 5, 9, 2, 6, 3, 7};
    constexpr int vt[BPR] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3};
    W6 cur[BPR];
    if (row0 < n_out) {
#pragma unroll
        for (int c = 0; c < BPR; ++c) cur[c] = q6_load(w + (size_t) row0 * BPR + kb[c], lane);
    }
    for (int row = row0; row < n_out; row += stride) {
        const int nrow = row + stride;
        W6 nxt[BPR];
        if (nrow < n_out) {
#pragma unroll
            for (int c = 0; c < BPR; ++c) nxt[c] = q6_load(w + (size_t) nrow * BPR + kb[c], lane);
        }
        float p[VT][NC];
#pragma unroll
        for (int v = 0; v < VT; ++v)
#pragma unroll
            for (int j = 0; j < NC; ++j) p[v][j] = 0.0f;
#pragma unroll
        for (int c = 0; c < BPR; ++c)
#pragma unroll
            for (int j = 0; j < NC; ++j) p[vt[c]][j] += q6_apply(cur[c], xc[j], kb[c] * (QK / Q8K), lane);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float t = p[0][j];
#pragma unroll
            for (int v = 1; v < VT; ++v) t += p[v][j];
            t = warp_sum(t);
            if constexpr (TOP) {
                // a row whose id is negative holds no token: it never wins and adds nothing to the sum
                if (map != nullptr && map[row] < 0) t = __int_as_float(0xff800000);
                else top->take(t, row);
            }
            if (lane == 0) y[(size_t) j * n_out + row] = t;
        }
#pragma unroll
        for (int c = 0; c < BPR; ++c) cur[c] = nxt[c];
    }
}

constexpr int XBLK = BPR * QK / Q8K;   // q8_1 blocks per column

template <int NC>
__global__ void __launch_bounds__(WPB * WARP) q6k_head_kernel(const Q6KBlock* __restrict__ w,
                                                              const Q81Block* __restrict__ x, float* __restrict__ y,
                                                              int n_out) {
    extern __shared__ __align__(16) int2 xe[];
    for (int i = threadIdx.x; i < NC * XBLK * 8; i += blockDim.x) {
        const int b = i / 8, pos = i % 8;
        const Q81Block* blk = x + b;
        const unsigned dsb = __ldg(reinterpret_cast<const unsigned*>(&blk->ds));
        xe[i] = make_int2(__ldg(reinterpret_cast<const int*>(blk->qs) + pos),
                          __float_as_int(__low2float(*reinterpret_cast<const half2*>(&dsb))));
    }
    __syncthreads();
    const int2* xc[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) xc[j] = xe + (size_t) j * XBLK * 8;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    q6k_rows<NC>(w, xc, y, n_out, blockIdx.x * WPB + warp, gridDim.x * WPB, lane);
}

// One column, n_in 2560: the logits, and the argmax (first on ties) with its softmax probability over all rows,
// by block partials and the last block's merge; `map` turns the row into a token id.
__global__ void __launch_bounds__(WPB * WARP) q6k_head_top1_kernel(const Q6KBlock* __restrict__ w,
                                                                   const Q81Block* __restrict__ x,
                                                                   float* __restrict__ y, int n_out,
                                                                   const int32_t* __restrict__ map, int* __restrict__ out_id,
                                                                   float* __restrict__ out_prob, float* __restrict__ part,
                                                                   unsigned* __restrict__ count) {
    __shared__ __align__(16) int2 xe[XBLK * 8];
    __shared__ Top wt[WPB];
    __shared__ bool last;
    for (int i = threadIdx.x; i < XBLK * 8; i += blockDim.x) {
        const int b = i / 8, pos = i % 8;
        const Q81Block* blk = x + b;
        const unsigned dsb = __ldg(reinterpret_cast<const unsigned*>(&blk->ds));
        xe[i] = make_int2(__ldg(reinterpret_cast<const int*>(blk->qs) + pos),
                          __float_as_int(__low2float(*reinterpret_cast<const half2*>(&dsb))));
    }
    __syncthreads();
    const int2* xc[1] = {xe};
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    Top top;
    top.init();
    q6k_rows<1, true>(w, xc, y, n_out, blockIdx.x * WPB + warp, gridDim.x * WPB, lane, &top, map);
    if (lane == 0) wt[warp] = top;
    __syncthreads();
    if (threadIdx.x == 0) {
        Top b = wt[0];
        for (int i = 1; i < WPB; ++i) b.merge(wt[i]);
        part[3 * blockIdx.x] = b.m;
        part[3 * blockIdx.x + 1] = __int_as_float(b.idx);
        part[3 * blockIdx.x + 2] = b.s;
        __threadfence();
        last = atomicAdd(count, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    Top g;   // each thread a strided share of the block partials, then the block
    g.init();
    for (int i = threadIdx.x; i < (int) gridDim.x; i += blockDim.x) {
        Top o;
        o.m = __ldcg(part + 3 * i);
        o.idx = __float_as_int(__ldcg(part + 3 * i + 1));
        o.s = __ldcg(part + 3 * i + 2);
        g.merge(o);
    }
    for (int off = 16; off > 0; off >>= 1) {
        Top o;
        o.m = __shfl_xor_sync(0xffffffffu, g.m, off);
        o.idx = __shfl_xor_sync(0xffffffffu, g.idx, off);
        o.s = __shfl_xor_sync(0xffffffffu, g.s, off);
        g.merge(o);
    }
    if (lane == 0) wt[warp] = g;
    __syncthreads();
    if (threadIdx.x == 0) {
        Top b = wt[0];
        for (int i = 1; i < WPB; ++i) b.merge(wt[i]);
        const int id = b.idx < n_out ? b.idx : 0;
        *out_id = map != nullptr ? map[id] : id;
        *out_prob = 1.0f / b.s;
        *count = 0u;
    }
}

template <int NC>
void launch_nc(const void* w, const void* x, float* y, int n_out, cudaStream_t s) {
    // one resident wave: every warp walks rows with the grid's stride
    static int grid_max[64] = {};
    static std::atomic<bool> ready[64];
    const int dev = opt_in_device("q6k_head");
    const size_t smem = (size_t) NC * XBLK * 8 * sizeof(int2);
    if (!ready[dev].load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(opt_in_mutex());
        if (!ready[dev].load(std::memory_order_relaxed)) {
            opt_in_check(cudaFuncSetAttribute(q6k_head_kernel<NC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem),
                         "q6k_head");
            int sms = 0, per_sm = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, q6k_head_kernel<NC>, WPB * WARP, smem);
            grid_max[dev] = (sms > 0 ? sms : 1) * (per_sm > 0 ? per_sm : 1);
            ready[dev].store(true, std::memory_order_release);
        }
    }
    const long long want = ((long long) n_out + WPB - 1) / WPB;
    const unsigned grid = (unsigned) (want < grid_max[dev] ? want : grid_max[dev]);
    q6k_head_kernel<NC><<<grid, WPB * WARP, smem, s>>>((const Q6KBlock*) w, (const Q81Block*) x, y, n_out);
}

// ---- the 2-bit copy and the two-stage argmax (head_proxy.cuh)

// One row's logit by one warp, in q6k_rows' order (the same bits as its one-column result).
__device__ __forceinline__ float q6k_row_one(const Q6KBlock* __restrict__ w, const int2* __restrict__ x, int lane) {
    constexpr int kb[BPR] = {0, 4, 8, 1, 5, 9, 2, 6, 3, 7};
    constexpr int vt[BPR] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3};
    float p[VT] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int c = 0; c < BPR; ++c) p[vt[c]] += q6_apply(q6_load(w + kb[c], lane), x, kb[c] * (QK / Q8K), lane);
    float t = p[0];
#pragma unroll
    for (int v = 1; v < VT; ++v) t += p[v];
    return warp_sum(t);
}

// One thread per 32-weight group: the group's dequantized Q6_K values to the 2-bit copy.
__global__ void q6k_proxy_pack_kernel(const Q6KBlock* __restrict__ q6, int64_t row0, int64_t n,
                                      uint4* __restrict__ wq, uint16_t* __restrict__ sc) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 80) return;
    const int64_t row = row0 + i / 80;
    const int g = (int) (i % 80);
    const Q6KBlock* b = q6 + (size_t) row * BPR + g / 8;
    const float d = __half2float(b->d);
    float w[32];
    for (int l = 0; l < 32; ++l) {
        const int k = (g % 8) * 32 + l;   // the weight within the block
        const int h = k / 128, quarter = (k % 128) / 32, ll = k % 32;
        const int lo = (b->ql[64 * h + ll + 32 * (quarter & 1)] >> (4 * (quarter >> 1))) & 0xF;
        const int hi = (b->qh[32 * h + ll] >> (2 * quarter)) & 3;
        w[l] = d * (float) b->scales[8 * h + ll / 16 + 2 * quarter] * (float) ((lo | (hi << 4)) - 32);
    }
    px::pack_group<true>(w, row, g, wq, sc);
}

// One block: the rows near the best approximate score are scored exactly (a warp per row), the exact best is the pick.
__global__ void __launch_bounds__(512) q6k_proxy_final_kernel(const Q6KBlock* __restrict__ w,
                                                              const Q81Block* __restrict__ x, int n_tiles,
                                                              const int32_t* __restrict__ map,
                                                              const float2* __restrict__ cand,
                                                              const float2* __restrict__ part, int* __restrict__ out_id,
                                                              float* __restrict__ out_prob) {
    __shared__ __align__(16) int2 xe[XBLK * 8];
    __shared__ px::Final f;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    pdl_wait();
    if (tid == 0) f.cnt = 0;
    for (int i = tid; i < XBLK * 8; i += blockDim.x) {
        const int b = i / 8, pos = i % 8;
        const Q81Block* blk = x + b;
        const unsigned dsb = __ldg(reinterpret_cast<const unsigned*>(&blk->ds));
        xe[i] = make_int2(__ldg(reinterpret_cast<const int*>(blk->qs) + pos),
                          __float_as_int(__low2float(*reinterpret_cast<const half2*>(&dsb))));
    }
    const int n = px::collect(f, n_tiles, map, cand, out_id, out_prob);
    if (n == 0) return;
    for (int c = warp; c < n; c += 16) {
        const float y = q6k_row_one(w + (size_t) f.list[c] * BPR, xe, lane);
        if (lane == 0) f.ex[c] = y;
    }
    __syncthreads();
    if (warp != 0) return;
    px::pick(f, n, n_tiles, map, part, out_id, out_prob);
}

// ---- the verify head: a 2-bit copy of every row scored on the tensor cores, then exact rescoring near the top

// Copy layout, per tile of 16 rows and quad q of 32-weight groups (groups 4q..4q+3): one uint4 of levels per lane of
// an m16n8k32 A fragment (word k: group 4q+k), then per row pair (g, g + 8) a uint4 of four half2 scales.  Word k of
// lane (g, t) holds the levels of rows g and g + 8 at k-offsets 4t..4t+3 and 16+4t..16+4t+3: register r of the
// fragment is (word >> 2r) & 0x03030303.
constexpr int VQ = 20;               // group quads per row
constexpr int VHIST = 1024;          // bins per column of the block-maximum histogram
constexpr float VBIN = 8.0f;         // bins per logit unit, centred on zero
constexpr int VCAP = 65536;          // rows rescored per call at most
constexpr float VDELTA = 4.0f;       // rescored: within this of the column's 64th best (approximate) logit
constexpr int VKEEP = 64;            // the sampler's widest list

__host__ __device__ inline size_t vtiles(int64_t n) { return (size_t) (n / 16); }
__host__ __device__ inline size_t vlevels_bytes(int64_t n) { return vtiles(n) * VQ * 32 * 16; }

struct VScratch {   // each kernel's last block leaves its part zeroed for the next call
    unsigned hist[8][VHIST];
    float thr[8];
    unsigned count, done_score, done_rescore;
};

__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}

// One thread per (row, group): the group's dequantized Q6_K values fitted as head_proxy.cuh fits them, its levels
// or-ed into the zeroed copy.
__global__ void q6k_vproxy_pack_kernel(const Q6KBlock* __restrict__ q6, int64_t n, uint32_t* __restrict__ wq,
                                       __half* __restrict__ sc) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 80) return;
    const int64_t row = i / 80;
    const int g = (int) (i % 80);
    const Q6KBlock* b = q6 + (size_t) row * BPR + g / 8;
    const float d = __half2float(b->d);
    float w[32];
    for (int l = 0; l < 32; ++l) {
        const int k = (g % 8) * 32 + l;
        const int h = k / 128, quarter = (k % 128) / 32, ll = k % 32;
        const int lo = (b->ql[64 * h + ll + 32 * (quarter & 1)] >> (4 * (quarter >> 1))) & 0xF;
        const int hi = (b->qh[32 * h + ll] >> (2 * quarter)) & 3;
        w[l] = d * (float) b->scales[8 * h + ll / 16 + 2 * quarter] * (float) ((lo | (hi << 4)) - 32);
    }
    float sh;
    const __half hs = px::fit_group<true>(w, sh);
    const size_t tile = (size_t) (row >> 4);
    const int rr = (int) (row & 15), gg = rr & 7, hi8 = rr >> 3, q = g / 4, k = g % 4;
    for (int t = 0; t < 4; ++t) {
        uint32_t word = 0u;
        for (int half = 0; half < 2; ++half)   // registers hi8 (k-offsets 4t..) and 2 + hi8 (16 + 4t..)
            for (int bb = 0; bb < 4; ++bb)
                word |= (uint32_t) px::level(w[16 * half + 4 * t + bb], sh) << (8 * bb + 2 * (2 * half + hi8));
        atomicOr(wq + ((tile * VQ + q) * 32 + gg * 4 + t) * 4 + k, word);
    }
    sc[(((tile * VQ + q) * 8 + gg) * 4 + k) * 2 + hi8] = hs;
}

// Approximate logits of every row into y, persistent blocks of four warps; warp w holds groups 20w..20w+19 of every
// tile (its column fragments stay in registers) and the block adds the four partial sums in warp order.  Each block
// adds its per-column maximum to the histogram.
template <int NC>
__global__ void __launch_bounds__(128) q6k_vscore_kernel(const uint4* __restrict__ wq, const uint4* __restrict__ sc,
                                                         const Q81Block* __restrict__ x, float* __restrict__ y,
                                                         int n_out, VScratch* __restrict__ vs) {
    __shared__ float4 xf[80][4];   // per group and column pair: the two scales, then -1.5 x scale x sum of q
    __shared__ float red[4][16][9];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int n_tiles = n_out / 16;
    int tile = blockIdx.x;
    uint4 cw[5], cs[5];
    if (tile < n_tiles) {
#pragma unroll
        for (int j = 0; j < 5; ++j) {   // constant data: read before the wait
            cw[j] = __ldcs(wq + ((size_t) tile * VQ + warp * 5 + j) * 32 + lane);
            cs[j] = __ldcs(sc + ((size_t) tile * VQ + warp * 5 + j) * 8 + g);
        }
    }
    pdl_wait();
    for (int i = tid; i < 80 * 8; i += 128) {
        const int grp = i >> 3, col = i & 7;
        float dd = 0.0f, xs = 0.0f;
        if (col < NC) {
            const Q81Block* b = x + (size_t) col * 80 + grp;
            const int* q = reinterpret_cast<const int*>(b->qs);
            int sum = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) sum = DP4A(0x01010101, q[k], sum);
            dd = __low2float(b->ds);
            xs = -1.5f * dd * (float) sum;
        }
        float* f = reinterpret_cast<float*>(&xf[grp][col >> 1]);
        f[col & 1] = dd;
        f[2 + (col & 1)] = xs;
    }
    uint32_t bf[20][2];
#pragma unroll
    for (int j = 0; j < 20; ++j) {
        const int grp = warp * 20 + j;
        if (g < NC) {
            const int* q = reinterpret_cast<const int*>(x[(size_t) g * 80 + grp].qs);
            bf[j][0] = (uint32_t) __ldg(q + t);
            bf[j][1] = (uint32_t) __ldg(q + 4 + t);
        } else {
            bf[j][0] = bf[j][1] = 0u;
        }
    }
    __syncthreads();
    const int orow = tid & 15, ocol = tid >> 4;
    float vmax = __int_as_float(0xff800000);
    for (; tile < n_tiles; tile += gridDim.x) {
        const int nt = tile + gridDim.x;
        uint4 nw[5], ns[5];
        if (nt < n_tiles) {
#pragma unroll
            for (int j = 0; j < 5; ++j) {
                nw[j] = __ldcs(wq + ((size_t) nt * VQ + warp * 5 + j) * 32 + lane);
                ns[j] = __ldcs(sc + ((size_t) nt * VQ + warp * 5 + j) * 8 + g);
            }
        }
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < 5; ++j) {
            const uint32_t ww[4] = {cw[j].x, cw[j].y, cw[j].z, cw[j].w};
            const uint32_t ss[4] = {cs[j].x, cs[j].y, cs[j].z, cs[j].w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int jj = j * 4 + k;
                const uint32_t a[4] = {ww[k] & 0x03030303u, (ww[k] >> 2) & 0x03030303u, (ww[k] >> 4) & 0x03030303u,
                                       (ww[k] >> 6) & 0x03030303u};
                int c[4];
                mma_s8(c, a, bf[jj][0], bf[jj][1]);
                const float4 f = xf[warp * 20 + jj][t];
                const float2 s2 = __half22float2(*reinterpret_cast<const half2*>(&ss[k]));
                acc[0] += s2.x * fmaf(f.x, (float) c[0], f.z);
                acc[1] += s2.x * fmaf(f.y, (float) c[1], f.w);
                acc[2] += s2.y * fmaf(f.x, (float) c[2], f.z);
                acc[3] += s2.y * fmaf(f.y, (float) c[3], f.w);
            }
        }
        red[warp][g][2 * t] = acc[0];
        red[warp][g][2 * t + 1] = acc[1];
        red[warp][g + 8][2 * t] = acc[2];
        red[warp][g + 8][2 * t + 1] = acc[3];
        __syncthreads();
        if (ocol < NC) {
            const float v = ((red[0][orow][ocol] + red[1][orow][ocol]) + red[2][orow][ocol]) + red[3][orow][ocol];
            y[(size_t) ocol * n_out + (size_t) tile * 16 + orow] = v;
            vmax = fmaxf(vmax, v);
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 5; ++j) { cw[j] = nw[j]; cs[j] = ns[j]; }
    }
    // the block's column maxima (16 lanes per column) into the histogram
#pragma unroll
    for (int o = 8; o; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
    if ((lane & 15) == 0 && ocol < NC && vmax > __int_as_float(0xff800000)) {
        const int bin = min(VHIST - 1, max(0, (int) floorf(vmax * VBIN) + VHIST / 2));
        atomicAdd(&vs->hist[ocol][bin], 1u);
    }
    // the last block: each column's threshold, the lower edge of the bin holding the 64th largest block maximum
    // (so at or below the 64th largest logit) less VDELTA; then the histogram is cleared
    __shared__ bool last;
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        last = atomicAdd(&vs->done_score, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int c = warp; c < NC; c += 4) {
        unsigned v[VHIST / 32], tot[VHIST / 32];   // lane l of row i: bin 32 i + l
#pragma unroll
        for (int i = 0; i < VHIST / 32; ++i) v[i] = __ldcg(&vs->hist[c][i * 32 + lane]);
#pragma unroll
        for (int i = 0; i < VHIST / 32; ++i) tot[i] = __reduce_add_sync(0xffffffffu, v[i]);
        unsigned run = 0, row_v = 0;
        int row = -1;
#pragma unroll
        for (int i = VHIST / 32 - 1; i >= 0; --i)
            if (row < 0) {
                if (run + tot[i] >= (unsigned) VKEEP) { row = i; row_v = v[i]; }
                else run += tot[i];
            }
        int bin = -1;
        if (row >= 0) {
            unsigned suf = row_v;   // this bin's count and those of the bins above it in the row
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned u = __shfl_down_sync(0xffffffffu, suf, o);
                if (lane + o < 32) suf += u;
            }
            const unsigned m = __ballot_sync(0xffffffffu, run + suf >= (unsigned) VKEEP);
            bin = row * 32 + 31 - __clz((int) m);
        }
        if (lane == 0)   // fewer than VKEEP maxima, or the lowest bin: every row
            vs->thr[c] = bin > 0 ? (float) (bin - VHIST / 2) / VBIN - VDELTA : __int_as_float(0xff800000);
        for (int i = lane; i < VHIST; i += 32) vs->hist[c][i] = 0u;
    }
    if (tid == 0) vs->done_score = 0u;
}

// The (row, column) pairs at or above the column's threshold, listed.
__global__ void __launch_bounds__(256) q6k_vselect_kernel(const float* __restrict__ y, int n_out, int ncols,
                                                          VScratch* __restrict__ vs, uint32_t* __restrict__ list) {
    __shared__ float thr[8];
    const int tid = threadIdx.x, lane = tid & 31;
    pdl_wait();
    if (tid < ncols) thr[tid] = __ldcg(&vs->thr[tid]);
    __syncthreads();
    const int n4 = n_out / 4;
    const int total = ncols * n4;
    for (int i0 = blockIdx.x * blockDim.x; i0 < total; i0 += gridDim.x * blockDim.x) {
        const int i = i0 + tid;
        unsigned m = 0u;
        int col = 0, r4 = 0;
        if (i < total) {
            col = i / n4;
            r4 = i - col * n4;
            const float4 v = __ldcg(reinterpret_cast<const float4*>(y + (size_t) col * n_out) + r4);
            const float th = thr[col];
            m = (v.x >= th ? 1u : 0u) | (v.y >= th ? 2u : 0u) | (v.z >= th ? 4u : 0u) | (v.w >= th ? 8u : 0u);
        }
        const int cnt = __popc(m);
        int pre = cnt;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, pre, o);
            if (lane >= o) pre += v;
        }
        const int wtot = __shfl_sync(0xffffffffu, pre, 31);
        if (wtot == 0) continue;
        unsigned base = 0;
        if (lane == 31) base = atomicAdd(&vs->count, (unsigned) wtot);
        base = __shfl_sync(0xffffffffu, base, 31) + (unsigned) (pre - cnt);
        for (int b = 0; b < 4; ++b)
            if (m & (1u << b)) {
                if (base < (unsigned) VCAP) list[base] = ((uint32_t) col << 18) | (uint32_t) (4 * r4 + b);
                ++base;
            }
    }
}

// The listed rows' exact logits (q6k_rows' bits), a warp per (row, column); the last block clears the count.
__global__ void __launch_bounds__(512) q6k_vrescore_kernel(const Q6KBlock* __restrict__ w,
                                                           const Q81Block* __restrict__ x, float* __restrict__ y,
                                                           int n_out, int ncols, VScratch* __restrict__ vs,
                                                           const uint32_t* __restrict__ list) {
    extern __shared__ __align__(16) int2 xv[];
    __shared__ bool last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    pdl_wait();
    for (int i = tid; i < ncols * XBLK * 8; i += blockDim.x) {
        const int b = i / 8, pos = i % 8;
        const Q81Block* blk = x + b;
        const unsigned dsb = __ldg(reinterpret_cast<const unsigned*>(&blk->ds));
        xv[i] = make_int2(__ldg(reinterpret_cast<const int*>(blk->qs) + pos),
                          __float_as_int(__low2float(*reinterpret_cast<const half2*>(&dsb))));
    }
    __syncthreads();
    const unsigned n = min(__ldcg(&vs->count), (unsigned) VCAP);
    for (unsigned c = blockIdx.x * 16 + warp; c < n; c += gridDim.x * 16) {
        const uint32_t e = __ldcg(list + c);
        const int row = (int) (e & 0x3ffffu), col = (int) (e >> 18);
        const float v = q6k_row_one(w + (size_t) row * BPR, xv + (size_t) col * XBLK * 8, lane);
        if (lane == 0) y[(size_t) col * n_out + row] = v;
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        last = atomicAdd(&vs->done_rescore, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (last && tid == 0) { vs->count = 0u; vs->done_rescore = 0u; }
}

// One grid size for every column count: the blocks' rows, so their maxima and the thresholds, and with them which rows
// are exact, do not depend on the window's width.
int vscore_grid(int n_out) {
    static int grid_max[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    if (grid_max[dev] == 0) {
        int sms = 0, per_sm = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, q6k_vscore_kernel<8>, 128, 0);
        grid_max[dev] = (sms > 0 ? sms : 1) * (per_sm > 0 ? per_sm : 1);
    }
    const int tiles = n_out / 16;
    return tiles < grid_max[dev] ? tiles : grid_max[dev];
}

template <int NC>
void vscore_launch(const void* proxy, const void* x, float* y, int n_out, VScratch* vs, cudaStream_t s) {
    const unsigned grid = (unsigned) vscore_grid(n_out);
    const uint4* wq = (const uint4*) proxy;
    const uint4* sc = (const uint4*) ((const char*) proxy + vlevels_bytes(n_out));
    launch_pdl(q6k_vscore_kernel<NC>, dim3(grid), dim3(128), 0, s, wq, sc, (const Q81Block*) x, y, n_out, vs);
}

}  // namespace

bool q6k_head_supported(int n_in, int ncols) { return n_in == BPR * QK && ncols >= 1 && ncols <= 8; }

size_t q6k_head_top1_scratch_bytes() { return 4096 * 3 * sizeof(float) + 256; }

bool q6k_head_top1(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, const int32_t* map,
                   int* out_id, float* out_prob, void* scratch, void* stream) {
    if (n_in != BPR * QK || n_out <= 0 || !weights || !x_q8_1 || !y || !out_id || !out_prob || !scratch) return false;
    static int grid_max[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    if (grid_max[dev] == 0) {
        int sms = 0, per_sm = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, q6k_head_top1_kernel, WPB * WARP, 0);
        grid_max[dev] = (sms > 0 ? sms : 1) * (per_sm > 0 ? per_sm : 1);
        if (grid_max[dev] > 4096) grid_max[dev] = 4096;
    }
    const long long want = ((long long) n_out + WPB - 1) / WPB;
    const unsigned grid = (unsigned) (want < grid_max[dev] ? want : grid_max[dev]);
    float* part = (float*) scratch;
    unsigned* count = (unsigned*) ((char*) scratch + 4096 * 3 * sizeof(float));
    q6k_head_top1_kernel<<<grid, WPB * WARP, 0, (cudaStream_t) stream>>>((const Q6KBlock*) weights,
                                                                         (const Q81Block*) x_q8_1, y, n_out, map, out_id,
                                                                         out_prob, part, count);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "q6k_head_top1: %s", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

void q6k_head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!q6k_head_supported(n_in, ncols) || n_out <= 0 || !weights || !x_q8_1 || !y) {
        basalt::log::error("basalt", "q6k_head_mmvq: unsupported shape (n_in %d, ncols %d)", n_in, ncols);
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
        basalt::log::error("basalt", "q6k_head_mmvq: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

void q6k_proxy_pack(const void* q6k_rows, int64_t row0, int64_t n, void* proxy, int64_t n_cap, void* stream) {
    if (n <= 0) return;
    const int64_t total = n * 80;
    q6k_proxy_pack_kernel<<<(unsigned) ((total + 127) / 128), 128, 0, (cudaStream_t) stream>>>(
        (const Q6KBlock*) q6k_rows, row0, n, (uint4*) proxy, px::scales(proxy, n_cap));
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "q6k_proxy_pack: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

bool q6k_proxy_top1(const void* proxy, const void* q6k_rows, const void* x_q8_1, int n_rows, int64_t n_cap,
                    const int32_t* map, int* out_id, float* out_prob, void* scratch, void* stream) {
    if (n_rows <= 0 || n_rows > n_cap || !proxy || !q6k_rows || !x_q8_1 || !map || !out_id || !out_prob || !scratch)
        return false;
    const int tiles = (int) px::tiles(n_rows);
    const uint4* wq = (const uint4*) proxy;
    const uint32_t* sc = (const uint32_t*) px::scales(const_cast<void*>(proxy), n_cap);
    float2* cand = (float2*) scratch;
    float2* part = cand + (size_t) px::tiles(n_cap) * px::kPer;
    launch_pdl(px::score_kernel, dim3((unsigned) tiles), dim3(128), 0, (cudaStream_t) stream, wq, sc,
               (const px::X81*) x_q8_1, n_rows, map, cand, part);
    launch_pdl(q6k_proxy_final_kernel, dim3(1), dim3(512), 0, (cudaStream_t) stream, (const Q6KBlock*) q6k_rows,
               (const Q81Block*) x_q8_1, tiles, map, (const float2*) cand, (const float2*) part, out_id, out_prob);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "q6k_proxy_top1: %s", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

size_t q6k_vproxy_bytes(int64_t n) { return n > 0 && n % 16 == 0 ? vlevels_bytes(n) + vtiles(n) * VQ * 8 * 16 : 0; }

void q6k_vproxy_pack(const void* q6k_rows, int64_t n, void* proxy, void* stream) {
    if (n <= 0 || n % 16 != 0) {
        basalt::log::error("basalt", "q6k_vproxy_pack: %lld rows (a positive multiple of 16)", (long long) n);
        std::exit(1);
    }
    const auto s = (cudaStream_t) stream;
    cudaMemsetAsync(proxy, 0, vlevels_bytes(n), s);
    const int64_t total = n * 80;
    q6k_vproxy_pack_kernel<<<(unsigned) ((total + 127) / 128), 128, 0, s>>>(
        (const Q6KBlock*) q6k_rows, n, (uint32_t*) proxy, (__half*) ((char*) proxy + vlevels_bytes(n)));
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "q6k_vproxy_pack: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

size_t q6k_vhead_scratch_bytes() { return sizeof(VScratch) + (size_t) VCAP * sizeof(uint32_t); }

bool q6k_vhead_supported(int n_in, int n_out, int ncols) {
    return n_in == BPR * QK && n_out > 0 && n_out % 16 == 0 && n_out <= (1 << 18) && ncols >= 1 && ncols <= 8;
}

void q6k_vhead(const void* proxy, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
               void* scratch, void* stream) {
    if (!q6k_vhead_supported(n_in, n_out, ncols) || !proxy || !weights || !x_q8_1 || !y || !scratch) {
        basalt::log::error("basalt", "q6k_vhead: unsupported shape (n_in %d, n_out %d, ncols %d)", n_in, n_out, ncols);
        std::exit(1);
    }
    const auto s = (cudaStream_t) stream;
    VScratch* vs = (VScratch*) scratch;
    uint32_t* list = (uint32_t*) ((char*) scratch + sizeof(VScratch));
    switch (ncols) {
        case 1: vscore_launch<1>(proxy, x_q8_1, y, n_out, vs, s); break;
        case 2: vscore_launch<2>(proxy, x_q8_1, y, n_out, vs, s); break;
        case 3: vscore_launch<3>(proxy, x_q8_1, y, n_out, vs, s); break;
        case 4: vscore_launch<4>(proxy, x_q8_1, y, n_out, vs, s); break;
        case 5: vscore_launch<5>(proxy, x_q8_1, y, n_out, vs, s); break;
        case 6: vscore_launch<6>(proxy, x_q8_1, y, n_out, vs, s); break;
        case 7: vscore_launch<7>(proxy, x_q8_1, y, n_out, vs, s); break;
        default: vscore_launch<8>(proxy, x_q8_1, y, n_out, vs, s); break;
    }
    static int sms[64] = {};
    static std::atomic<bool> ready[64];
    const int dev = opt_in_device("q6k_vhead");
    if (!ready[dev].load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(opt_in_mutex());
        if (!ready[dev].load(std::memory_order_relaxed)) {
            cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev);
            opt_in_check(cudaFuncSetAttribute(q6k_vrescore_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              8 * XBLK * 8 * (int) sizeof(int2)),
                         "q6k_vhead");
            if (sms[dev] <= 0) sms[dev] = 1;
            ready[dev].store(true, std::memory_order_release);
        }
    }
    launch_pdl(q6k_vselect_kernel, dim3((unsigned) sms[dev] * 2), dim3(256), 0, s, (const float*) y, n_out, ncols, vs,
               list);
    launch_pdl(q6k_vrescore_kernel, dim3((unsigned) sms[dev]), dim3(512), (size_t) ncols * XBLK * 8 * sizeof(int2), s,
               (const Q6KBlock*) weights, (const Q81Block*) x_q8_1, y, n_out, ncols, vs, (const uint32_t*) list);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "q6k_vhead: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace basalt::kernels
