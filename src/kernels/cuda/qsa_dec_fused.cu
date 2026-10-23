// src/kernels/cuda/qsa_dec_fused.cu - see include/basalt/kernels/qsa_dec_fused.hpp.
//
// Built with --use_fast_math like native_qsa.cu, native_rope.cu and native_qsa_indexer.cu, whose expressions this
// repeats; the INT8 append (kv_q8.cu, built without it) uses explicit IEEE instructions.
#include "basalt/kernels/qsa_dec_fused.hpp"
#include "basalt/kernels/qsa_seg.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/mrope.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int HD = 256, ID = 128, R = 4, ROT = 64, THREADS = 256;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}
// IEEE single-precision operations without flush-to-zero, as a file built without fast math emits them
__device__ __forceinline__ float abs_ieee(float x) { float y; asm("abs.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float max_ieee(float a, float b) { float y; asm("max.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float div_ieee(float a, float b) { float y; asm("div.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }

// native_qsa.cu's norm<256> for one row of n_cols <= 256 (thread d holds column d, x = 0 past n_cols): the scale
__device__ __forceinline__ float norm_scale256(float x, int d, int n_cols, float epsilon, float* sums) {
    float partial = 0.0f;
    if (d < n_cols) partial += x * x;
    partial = warp_sum(partial);
    const int lane = d % 32;
    if (lane == 0) sums[d / 32] = partial;
    __syncthreads();
    partial = lane < THREADS / 32 ? sums[lane] : 0.0f;
    partial = warp_sum(partial);
    const float mean = partial / n_cols;
    return rsqrtf(mean + epsilon);
}

struct RopeArgs {
    float theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale;
    const int32_t* mtab;
    RopeTab rt;
};

// native_rope.cu's apply<TAB> on a row held in shared memory (pairs below n_rot/2 rotate in place)
template <bool TAB>
__device__ __forceinline__ void rope_row(float* row, int pos, int d, const RopeArgs& ra) {
    float r0 = 0.0f, r1 = 0.0f;
    const int pair = d;
    if (pair < ROT / 2) {
        float c, s;
        if (!(TAB && rope_tab_cs(ra.rt, mrope_pos(ra.mtab, pos, pair), pair, c, s))) {
            const float theta_extrap = mrope_pos(ra.mtab, pos, pair) * powf(ra.theta_scale, float(pair));
            rope_scaled_angle(theta_extrap, ra.freq_scale, ra.corr_low, ra.corr_high, ra.ext_factor, ra.mscale, pair, c, s);
        }
        const float a = row[pair], b = row[pair + ROT / 2];
        r0 = a * c - b * s;
        r1 = a * s + b * c;
    }
    __syncthreads();
    if (pair < ROT / 2) {
        row[pair] = r0;
        row[pair + ROT / 2] = r1;
    }
    __syncthreads();
}

// kv_q8.cu's kv_append_q8_kernel for one row of one KV head: thread d holds value d; groups of 64 are warp pairs
__device__ __forceinline__ void append_q8_row(float x, int d, long long pos, int h, int n_kv_heads, int page_size,
                                              const int32_t* table, int8_t* codes, uint16_t* scales, int8_t* host_codes,
                                              uint16_t* host_scales, float* wmax) {
    const int groups = HD / KV_Q8_GROUP, g = d / KV_Q8_GROUP, t = d % KV_Q8_GROUP;
    float a = abs_ieee(x);
    for (int o = 16; o > 0; o >>= 1) a = max_ieee(a, __shfl_xor_sync(0xffffffffu, a, o));
    if ((d & 31) == 0) wmax[d >> 5] = a;
    __syncthreads();
    const float amax = max_ieee(wmax[2 * g], wmax[2 * g + 1]);
    const uint16_t sbits = f16_from_f32(div_ieee(amax, 127.0f));
    const float sf = f32_from_f16(sbits);
    int q = 0;
    if (sf > 0.0f) {
        q = __float2int_rn(div_ieee(x, sf));
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) {
        const long long row = (page * n_kv_heads + h) * page_size + (pos % page_size);
        codes[row * HD + g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) scales[row * groups + g] = sbits;
    }
    if (host_codes != nullptr) {
        const long long row = ((pos / page_size) * n_kv_heads + h) * page_size + (pos % page_size);
        host_codes[row * HD + g * KV_Q8_GROUP + t] = (int8_t) q;
        if (t == 0) host_scales[row * groups + g] = sbits;
    }
    __syncthreads();
}

// native_qsa_indexer.cu's append<TAB> for one row (pos_base 0)
template <bool TAB>
__device__ __forceinline__ void indexer_append_row(const float* raw, int pos, const float* gamma, float epsilon,
                                                   float* tail, float* dead, float* pooled, int32_t* block_pos,
                                                   int max_cells, const RopeArgs& ra, float* values, float* partials) {
    const int d = threadIdx.x;
    if (pos < 0 || pos >= max_cells) return;
    const int slot = pos % R;
    float incoming = 0.0f;
    if (d < ID) {
        incoming = __half2float(__float2half_rn(raw[d]));
        if (slot < R - 1) tail[slot * ID + d] = incoming;
    }
    if (pos != 0 && slot != R - 1) return;
    float mean = 0.0f;
    if (d < ID) {
        float sum = pos == 0 ? incoming : tail[d];
#pragma unroll
        for (int j = 1; j < R; ++j) sum = __fadd_rn(sum, pos == 0 || j == R - 1 ? incoming : tail[j * ID + d]);
        mean = __fmaf_rn(0.25f, sum, 0.0f);
    }
    float square_sum = 0.0f;
    if (d < ID) square_sum += mean * mean;
    square_sum = warp_sum(square_sum);
    const int lane = d % 32;
    if (lane == 0) partials[d / 32] = square_sum;
    __syncthreads();
    square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
    square_sum = warp_sum(square_sum);
    const float scale = rsqrtf(square_sum / ID + epsilon);
    if (d < ID) values[d] = scale * mean * gamma[d];
    __syncthreads();
    if (d < ID) {
        const int b = pos / R;
        const int rope_pos = pos == 0 ? 0 : R * b;
        float y = values[d];
        if (d < ROT) {
            const int pair = d % (ROT / 2);
            float c, s;
            if (!(TAB && rope_tab_cs(ra.rt, pos == 0 ? 0 : mrope_pos(ra.mtab, rope_pos, pair), pair, c, s))) {
                const float theta_extrap = (pos == 0 ? 0 : mrope_pos(ra.mtab, rope_pos, pair)) * powf(ra.theta_scale, float(pair));
                rope_scaled_angle(theta_extrap, ra.freq_scale, ra.corr_low, ra.corr_high, ra.ext_factor, ra.mscale, pair, c, s);
            }
            const float a = values[pair], z = values[pair + ROT / 2];
            y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
        }
        pooled[std::size_t(b) * ID + d] = y;
        if (pos == 0) dead[d] = y;
        else pooled[std::size_t(b + 1) * ID + d] = dead[d];
        if (d == 0 && pos != 0) *block_pos = rope_pos;
    }
    __syncthreads();
}

// blocks [0, n * n_kv): row t = b / n_kv, KV head b % n_kv - K norm + RoPE, K and V appends; block n * n_kv: the
// indexer tail snapshot and the n indexer appends in row order
template <bool TAB>
__device__ __forceinline__ void kv_post_dev(const QsaDecKvPost& a, int n, int n_kv, int page_size, const RopeArgs& ra,
                                            int b) {
    __shared__ float row[HD];
    __shared__ float sums[32];
    const int d = threadIdx.x;
    if (b < n * n_kv) {
        const int t = b / n_kv, h = b % n_kv;
        const long long pos = (long long) __ldg(a.steps + (size_t) t * kStepCount + kStepPos);
        float* kp = a.kcur + ((size_t) t * n_kv + h) * HD;
        const float x = kp[d];
        const float scale = norm_scale256(x, d, HD, a.eps, sums);
        row[d] = scale * x * a.k_norm[d];
        __syncthreads();
        rope_row<TAB>(row, a.pos_k[t * (a.pos_stride > 0 ? a.pos_stride : n_kv) + h], d, ra);
        const float y = row[d];
        kp[d] = y;
        append_q8_row(y, d, pos, h, n_kv, page_size, a.page_table, a.k_q, a.k_scale, a.host.k_q, a.host.k_scale, sums);
        const float v = a.vcur[((size_t) t * n_kv + h) * HD + d];
        append_q8_row(v, d, pos, h, n_kv, page_size, a.page_table, a.v_q, a.v_scale, a.host.v_q, a.host.v_scale, sums);
        return;
    }
    if (a.tail_snap != nullptr)
        for (int i = d; i < (R - 1) * ID; i += THREADS) a.tail_snap[i] = a.tail[i];
    __syncthreads();
    for (int t = 0; t < n; ++t) {
        const int pos = __ldg(a.steps + (size_t) t * kStepCount + kStepPos);
        indexer_append_row<TAB>(a.idx_raw + (size_t) t * ID, pos, a.idx_k_norm, a.eps, a.tail, a.dead, a.pooled,
                                a.block_pos, a.max_cells, ra, row, sums);
        __syncthreads();
    }
}

// blocks [0, n * n_head): the query heads; then n * idx_n_head blocks: the indexer query heads
template <bool TAB>
__device__ __forceinline__ void q_post_dev(const QsaDecQPost& a, int n, int n_head, int idx_heads, const RopeArgs& ra,
                                           int b) {
    __shared__ float row[HD];
    __shared__ float sums[32];
    const int d = threadIdx.x;
    pdl_launch_dependents();
    if (b < n * n_head) {
        const float x = a.q_full[(size_t) b * 2 * HD + d];
        const float scale = norm_scale256(x, d, HD, a.eps, sums);
        row[d] = scale * x * a.q_norm[d];
        __syncthreads();
        rope_row<TAB>(row, a.pos_q[b], d, ra);
        a.qcur[(size_t) b * HD + d] = row[d];
        return;
    }
    b -= n * n_head;
    float* qp = a.q_idx + (size_t) b * ID;
    const float x = d < ID ? qp[d] : 0.0f;
    const float scale = norm_scale256(x, d, ID, a.eps, sums);
    if (d < ID) row[d] = scale * x * a.idx_q_norm[d];
    __syncthreads();
    rope_row<TAB>(row, a.pos_i[b], d, ra);
    if (d < ID) qp[d] = row[d];
}

template <bool TAB>
__global__ void __launch_bounds__(THREADS) kv_post_kernel(QsaDecKvPost a, int n, int n_kv, int page_size, RopeArgs ra) {
    kv_post_dev<TAB>(a, n, n_kv, page_size, ra, blockIdx.x + (a.part == 2 ? n * n_kv : 0));
}
// the per-sequence fields of a post call, from a slot
__device__ __forceinline__ void take_slot(QsaDecKvPost& a, const QsaSegSlot& sl) {
    a.k_q = sl.k_q; a.v_q = sl.v_q; a.k_scale = sl.k_scale; a.v_scale = sl.v_scale; a.page_table = sl.page_table;
    a.host = sl.host; a.tail = sl.tail; a.dead = sl.dead; a.pooled = sl.pooled; a.block_pos = sl.block_pos;
    a.max_cells = sl.max_cells;
}

// kv_post_kernel with each row's slot from the table: blocks [0, n * n_kv) the rows' K/V, then one block per segment
// for its indexer appends (the segment's rows, in order, on its slot's tail)
template <bool TAB>
__global__ void __launch_bounds__(THREADS) kv_post_seg_kernel(QsaDecKvPost a, QsaSegTable t, int n, int n_kv,
                                                              int page_size, RopeArgs ra) {
    const int b = blockIdx.x + (a.part == 2 ? n * n_kv : 0);
    if (b < n * n_kv) {
        take_slot(a, t.slots[t.rows[b / n_kv].slot]);
        a.tail_snap = nullptr;
        kv_post_dev<TAB>(a, n, n_kv, page_size, ra, b);
        return;
    }
    const QsaSeg sg = t.segs[b - n * n_kv];
    if (sg.slot < 0) return;   // an unused entry of the table
    const QsaSegSlot& sl = t.slots[sg.slot];
    take_slot(a, sl);
    a.tail_snap = a.tail_snap != nullptr ? sl.tail_snap : nullptr;
    a.idx_raw += (size_t) sg.first * ID;
    a.steps += (size_t) sg.first * kStepCount;
    kv_post_dev<TAB>(a, sg.len, n_kv, page_size, ra, sg.len * n_kv);
}

template <bool TAB>
__global__ void __launch_bounds__(THREADS) q_post_kernel(QsaDecQPost a, int n, int n_head, int idx_heads, RopeArgs ra) {
    q_post_dev<TAB>(a, n, n_head, idx_heads, ra, blockIdx.x);
}

RopeArgs rope_args(const RopeScaling& scaling) {
    RopeArgs ra;
    ra.theta_scale = powf((float) scaling.freq_base, -2.0f / ROT);
    const RopeKernelArgs k = scaling.kernel_args(ROT);
    ra.freq_scale = k.freq_scale; ra.corr_low = k.corr_low; ra.corr_high = k.corr_high;
    ra.ext_factor = k.ext_factor; ra.mscale = k.attn_factor;
    ra.mtab = mrope_table();
    ra.rt = rope_table_for(scaling);
    return ra;
}
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

bool qsa_dec_fused_supported(const QsaShapes& s) {
    return s.head_dim == HD && s.idx_dim == ID && s.idx_block == R && s.n_rot == ROT && s.n_head_kv > 0 &&
           s.page_size > 0 && KV_Q8_GROUP == 64;
}

void qsa_dec_kv_post(const QsaDecKvPost& a, int n, const QsaShapes& s, const RopeScaling& scaling, void* stream) {
    if (n <= 0) return;
    if (!qsa_dec_fused_supported(s) || rope_scaling_invalid(scaling) != nullptr) {
        basalt::log::error("basalt", "qsa_dec_kv_post: unsupported geometry or scaling");
        std::exit(1);
    }
    const RopeArgs ra = rope_args(scaling);
    const unsigned kv_blocks = a.part == 2 ? 0u : (unsigned) (n * s.n_head_kv);
    const unsigned blocks = kv_blocks + (a.idx_raw != nullptr && a.part != 1 ? 1 : 0);
    if (blocks == 0) return;
    if (ra.rt.cos != nullptr)
        kv_post_kernel<true><<<blocks, THREADS, 0, (cudaStream_t) stream>>>(a, n, (int) s.n_head_kv, (int) s.page_size, ra);
    else
        kv_post_kernel<false><<<blocks, THREADS, 0, (cudaStream_t) stream>>>(a, n, (int) s.n_head_kv, (int) s.page_size, ra);
    check("qsa_dec_kv_post");
}

void qsa_dec_kv_post_seg(const QsaDecKvPost& a, const QsaSegTable& t, int n, const QsaShapes& s,
                         const RopeScaling& scaling, void* stream) {
    if (n <= 0) return;
    if (!qsa_dec_fused_supported(s) || rope_scaling_invalid(scaling) != nullptr || t.n_seg <= 0 || t.n_seg > n) {
        basalt::log::error("basalt", "qsa_dec_kv_post_seg: unsupported geometry, scaling or table");
        std::exit(1);
    }
    const RopeArgs ra = rope_args(scaling);
    const unsigned kv_blocks = a.part == 2 ? 0u : (unsigned) (n * s.n_head_kv);
    const unsigned blocks = kv_blocks + (a.idx_raw != nullptr && a.part != 1 ? (unsigned) t.n_seg : 0u);
    if (blocks == 0) return;
    if (ra.rt.cos != nullptr)
        kv_post_seg_kernel<true><<<blocks, THREADS, 0, (cudaStream_t) stream>>>(a, t, n, (int) s.n_head_kv, (int) s.page_size, ra);
    else
        kv_post_seg_kernel<false><<<blocks, THREADS, 0, (cudaStream_t) stream>>>(a, t, n, (int) s.n_head_kv, (int) s.page_size, ra);
    check("qsa_dec_kv_post_seg");
}

void qsa_dec_q_post(const QsaDecQPost& a, int n, const QsaShapes& s, const RopeScaling& scaling, void* stream) {
    if (n <= 0) return;
    if (!qsa_dec_fused_supported(s) || rope_scaling_invalid(scaling) != nullptr) {
        basalt::log::error("basalt", "qsa_dec_q_post: unsupported geometry or scaling");
        std::exit(1);
    }
    const RopeArgs ra = rope_args(scaling);
    const unsigned blocks = (unsigned) (n * (s.n_head + (a.q_idx != nullptr ? s.idx_n_head : 0)));
    if (ra.rt.cos != nullptr)
        q_post_kernel<true><<<blocks, THREADS, 0, (cudaStream_t) stream>>>(a, n, (int) s.n_head, (int) s.idx_n_head, ra);
    else
        q_post_kernel<false><<<blocks, THREADS, 0, (cudaStream_t) stream>>>(a, n, (int) s.n_head, (int) s.idx_n_head, ra);
    check("qsa_dec_q_post");
}

}  // namespace basalt::kernels
