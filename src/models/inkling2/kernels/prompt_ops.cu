// src/models/inkling2/kernels/prompt_ops.cu - the prompt path's small kernels: the experts' plan as a stable counting sort
// of a chunk's (row, slot) pairs by expert, and the staged sliding-window keys moved into the window rings.
#include "basalt/models/inkling2/prompt_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

constexpr int kSortPairs = 1024;   // pairs a counting block takes
constexpr int kMaxExpert = 512;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 ") + what + ": " + cudaGetErrorString(e));
}

int sort_blocks(int pairs) { return (pairs + kSortPairs - 1) / kSortPairs; }

// Block b counts the experts of pairs [b kSortPairs, (b + 1) kSortPairs).
__global__ void __launch_bounds__(256) sort_count_kernel(const int32_t* __restrict__ ids, int pairs, int n_expert,
                                                         int32_t* __restrict__ counts) {
    __shared__ int32_t hist[kMaxExpert];
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) hist[e] = 0;
    __syncthreads();
    const int lo = blockIdx.x * kSortPairs, hi = min(pairs, lo + kSortPairs);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) atomicAdd(&hist[ids[i]], 1);
    __syncthreads();
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) counts[(size_t) blockIdx.x * n_expert + e] = hist[e];
}

// One block: each expert's first entry (bounds) and each counting block's first entry of each expert (counts, in place).
__global__ void __launch_bounds__(kMaxExpert) sort_scan_kernel(int32_t* __restrict__ counts, int n_blocks, int n_expert,
                                                               int pairs, int32_t* __restrict__ bounds) {
    __shared__ int32_t total[kMaxExpert];
    const int e = threadIdx.x;
    int run = 0;
    if (e < n_expert)
        for (int b = 0; b < n_blocks; ++b) {
            const int c = counts[(size_t) b * n_expert + e];
            counts[(size_t) b * n_expert + e] = run;
            run += c;
        }
    total[e] = e < n_expert ? run : 0;
    __syncthreads();
    if (e == 0) {
        int at = 0;
        for (int x = 0; x < n_expert; ++x) {
            const int t = total[x];
            total[x] = at;
            at += t;
        }
    }
    __syncthreads();
    if (e < n_expert) {
        bounds[e] = total[e];
        for (int b = 0; b < n_blocks; ++b) counts[(size_t) b * n_expert + e] += total[e];
    }
    if (e == 0) bounds[n_expert] = pairs;
}

// One warp a counting block: its pairs in order, 32 at a time; a pair's place is its expert's next entry plus the lanes
// before it with the same expert, so the entries of an expert keep the pairs' order.
__global__ void __launch_bounds__(32) sort_scatter_kernel(const int32_t* __restrict__ ids, int pairs, int k, int n_expert,
                                                          const int32_t* __restrict__ counts, int32_t* __restrict__ tok,
                                                          int32_t* __restrict__ dst, int32_t* __restrict__ exp) {
    __shared__ int32_t next[kMaxExpert];
    const int lane = threadIdx.x;
    for (int e = lane; e < n_expert; e += 32) next[e] = counts[(size_t) blockIdx.x * n_expert + e];
    __syncwarp();
    const int lo = blockIdx.x * kSortPairs, hi = min(pairs, lo + kSortPairs);
    for (int i0 = lo; i0 < hi; i0 += 32) {
        const int i = i0 + lane;
        const int e = i < hi ? ids[i] : -1;
        const unsigned same = __match_any_sync(0xffffffffu, e);
        const int rank = __popc(same & ((1u << lane) - 1u));
        if (e >= 0) {
            const int at = next[e] + rank;
            tok[at] = i / k;
            dst[at] = i;
            exp[at] = e;
        }
        __syncwarp();
        if (e >= 0 && rank == 0) next[e] += __popc(same);
        __syncwarp();
    }
}

// One 16-byte piece of one position's cell (its key or value rows of every head): x = position, y = (value).
__global__ void stage_to_ring_kernel(const uint4* __restrict__ k_stage, const uint4* __restrict__ v_stage, int64_t stage_cap,
                                     uint4* __restrict__ k_ring, uint4* __restrict__ v_ring, int64_t ring_cap, int64_t first,
                                     int pieces) {
    const int64_t p = first + blockIdx.x;
    const uint4* src = (blockIdx.y ? v_stage : k_stage) + (size_t) (p % stage_cap) * pieces;
    uint4* dst = (blockIdx.y ? v_ring : k_ring) + (size_t) (p % ring_cap) * pieces;
    for (int i = threadIdx.x; i < pieces; i += blockDim.x) dst[i] = src[i];
}

__global__ void prompt_rows_kernel(int32_t* __restrict__ pos, int64_t pos0, int rows, int32_t* __restrict__ bounds) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) pos[i] = (int32_t) (pos0 + i);
    if (i == 0) {
        bounds[0] = 0;
        bounds[1] = rows;
    }
}

}  // namespace

void prompt_rows(int32_t* pos, int64_t pos0, int rows, int32_t* bounds, void* stream) {
    if (rows <= 0) return;
    prompt_rows_kernel<<<(rows + 255) / 256, 256, 0, (cudaStream_t) stream>>>(pos, pos0, rows, bounds);
    check("prompt_rows");
}

size_t moe_sort_scratch(int pairs, int n_expert) { return (size_t) sort_blocks(pairs) * (size_t) n_expert * sizeof(int32_t); }

void moe_sort(const int32_t* ids, int rows, int k, int n_expert, void* scratch, int32_t* bounds, int32_t* tok, int32_t* dst,
              int32_t* exp, void* stream) {
    const int pairs = rows * k;
    if (pairs <= 0) return;
    if (n_expert > kMaxExpert) throw std::runtime_error("inkling2 moe_sort: more than 512 experts");
    cudaStream_t s = (cudaStream_t) stream;
    int32_t* counts = static_cast<int32_t*>(scratch);
    const int nb = sort_blocks(pairs);
    sort_count_kernel<<<nb, 256, 0, s>>>(ids, pairs, n_expert, counts);
    check("moe_sort/count");
    sort_scan_kernel<<<1, kMaxExpert, 0, s>>>(counts, nb, n_expert, pairs, bounds);
    check("moe_sort/scan");
    sort_scatter_kernel<<<nb, 32, 0, s>>>(ids, pairs, k, n_expert, counts, tok, dst, exp);
    check("moe_sort/scatter");
}

namespace {
__global__ void remap_ids_kernel(int32_t* __restrict__ ids, int n, const int32_t* __restrict__ map) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ids[i] = map[ids[i]];
}
}  // namespace

void remap_ids(int32_t* ids, int n, const int32_t* map, void* stream) {
    if (n <= 0) return;
    remap_ids_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(ids, n, map);
    check("remap_ids");
}

void stage_to_ring(const void* k_stage, const void* v_stage, int64_t stage_cap, void* k_ring, void* v_ring, int64_t ring_cap,
                   int64_t pos0, int rows, int keep, int row_bytes, void* stream) {
    const int64_t end = pos0 + rows, first = end - keep > pos0 ? end - keep : pos0;
    const int n = (int) (end - first);
    if (n <= 0) return;
    const int pieces = kKvHeads * row_bytes / 16;
    stage_to_ring_kernel<<<dim3((unsigned) n, 2), 128, 0, (cudaStream_t) stream>>>(
        static_cast<const uint4*>(k_stage), static_cast<const uint4*>(v_stage), stage_cap, static_cast<uint4*>(k_ring),
        static_cast<uint4*>(v_ring), ring_cap, first, pieces);
    check("stage_to_ring");
}


namespace {

__device__ __forceinline__ unsigned tf32_of(float f) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(f));
    return r;
}

__device__ __forceinline__ void cp16_cg(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src));
}

__device__ __forceinline__ void mma_tf32_2(float (&acc)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                           unsigned l0, unsigned l1, unsigned l2, unsigned l3, unsigned wv) {
    const unsigned b0 = wv << 16, b1 = wv & 0xFFFF0000u;
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                 : "r"(l0), "r"(l1), "r"(l2), "r"(l3), "r"(b0), "r"(b1));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float quad_sum_f(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// 16 rows x 64 outputs per block (4 warps, each 16 rows x 16 outputs), K=kWidth in 64-wide double-buffered stages.
constexpr int kCdT = 16, kCdN = 64, kCdK = 64, kCdW = 4;
constexpr int kCdXS = kCdK + 4, kCdWS = kCdK + 8;
static_assert(kWidth % kCdK == 0, "kWidth must be a multiple of kCdK");

__global__ void __launch_bounds__(32 * kCdW) ctrl_down_prompt_kernel(const float* __restrict__ dest,
                                                                     const float* __restrict__ msg,
                                                                     const uint16_t* __restrict__ w_down,
                                                                     float* __restrict__ hid, int rows, int rank) {
    __shared__ __align__(16) float xs[2][kCdT][kCdXS];
    __shared__ __align__(16) uint16_t ws[2][kCdN][kCdWS];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int t0 = (int) blockIdx.x * kCdT, nb = (int) blockIdx.y * kCdN, n0 = nb + wid * 16;
    const int nkc = kWidth / kCdK;
    const int64_t w_ld = (int64_t) 2 * kWidth;

    const auto load_stage = [&](int buf, const float* xsrc, int w_col0, int kc) {
        const int k0 = kc * kCdK;
        for (int q = tid; q < kCdT * kCdK / 4; q += 32 * kCdW) {
            const int r = q / (kCdK / 4), k4 = (q % (kCdK / 4)) * 4;
            cp16_cg(&xs[buf][r][k4], xsrc + (size_t) min(t0 + r, rows - 1) * kWidth + k0 + k4);
        }
        for (int q = tid; q < kCdN * kCdK / 8; q += 32 * kCdW) {
            const int r = q / (kCdK / 8), k8 = (q % (kCdK / 8)) * 8;
            cp16_cg(&ws[buf][r][k8], w_down + (size_t) min(nb + r, rank - 1) * w_ld + w_col0 + k0 + k8);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float acc_d[2][4] = {}, acc_m[2][4] = {};
    float ss_da = 0.f, ss_db = 0.f, ss_ma = 0.f, ss_mb = 0.f;

    const auto run_half = [&](const float* xsrc, int w_col0, float (&acc)[2][4], float& ss_a, float& ss_b) {
        load_stage(0, xsrc, w_col0, 0);
        for (int kc = 0; kc < nkc; ++kc) {
            if (kc + 1 < nkc) {
                load_stage((kc + 1) & 1, xsrc, w_col0, kc + 1);
                asm volatile("cp.async.wait_group 1;\n" ::);
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::);
            }
            __syncthreads();
            const int buf = kc & 1;
            const float* xa = &xs[buf][g][2 * c];
            const float* xb = &xs[buf][g + 8][2 * c];
            const uint16_t* w0 = &ws[buf][wid * 16 + g][2 * c];
            const uint16_t* w1 = &ws[buf][wid * 16 + 8 + g][2 * c];
#pragma unroll
            for (int k = 0; k < kCdK; k += 8) {
                const float2 fa = *reinterpret_cast<const float2*>(xa + k);
                const float2 fb = *reinterpret_cast<const float2*>(xb + k);
                ss_a = fmaf(fa.y, fa.y, fmaf(fa.x, fa.x, ss_a));
                ss_b = fmaf(fb.y, fb.y, fmaf(fb.x, fb.x, ss_b));
                const unsigned a0 = tf32_of(fa.x), a1 = tf32_of(fb.x), a2 = tf32_of(fa.y), a3 = tf32_of(fb.y);
                const unsigned l0 = tf32_of(fa.x - __uint_as_float(a0)), l1 = tf32_of(fb.x - __uint_as_float(a1));
                const unsigned l2 = tf32_of(fa.y - __uint_as_float(a2)), l3 = tf32_of(fb.y - __uint_as_float(a3));
                const unsigned wv0 = *reinterpret_cast<const unsigned*>(w0 + k);
                const unsigned wv1 = *reinterpret_cast<const unsigned*>(w1 + k);
                mma_tf32_2(acc[0], a0, a1, a2, a3, l0, l1, l2, l3, wv0);
                mma_tf32_2(acc[1], a0, a1, a2, a3, l0, l1, l2, l3, wv1);
            }
            __syncthreads();
        }
    };

    run_half(dest, 0, acc_d, ss_da, ss_db);
    run_half(msg, kWidth, acc_m, ss_ma, ss_mb);

    const float inv_da = rsqrtf(quad_sum_f(ss_da) / (float) kWidth + 1e-6f);
    const float inv_db = rsqrtf(quad_sum_f(ss_db) / (float) kWidth + 1e-6f);
    const float inv_ma = rsqrtf(quad_sum_f(ss_ma) / (float) kWidth + 1e-6f);
    const float inv_mb = rsqrtf(quad_sum_f(ss_mb) / (float) kWidth + 1e-6f);

    const int ta = t0 + g, tb = t0 + g + 8;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int col = n0 + j * 8 + 2 * c;
        if (col + 1 < rank) {
            if (ta < rows) {
                *reinterpret_cast<float2*>(hid + (size_t) ta * rank + col) =
                    make_float2(acc_d[j][0] * inv_da + acc_m[j][0] * inv_ma,
                                acc_d[j][1] * inv_da + acc_m[j][1] * inv_ma);
            }
            if (tb < rows) {
                *reinterpret_cast<float2*>(hid + (size_t) tb * rank + col) =
                    make_float2(acc_d[j][2] * inv_db + acc_m[j][2] * inv_mb,
                                acc_d[j][3] * inv_db + acc_m[j][3] * inv_mb);
            }
        }
    }
}

// 64 rows x 32 channels per block (4 warps, each 16 rows x 32 channels gain+forget).
constexpr int kCuT = 64, kCuC = 32, kCuK = 64, kCuW = 4;
constexpr int kCuHS = kCuK + 4, kCuWS = kCuK + 8;
static_assert(kWidth % kCuC == 0, "kWidth must be a multiple of kCuC");

__global__ void __launch_bounds__(32 * kCuW) ctrl_up_blend_prompt_kernel(const float* __restrict__ dest,
                                                                         const float* __restrict__ msg,
                                                                         const float* __restrict__ hid,
                                                                         const uint16_t* __restrict__ w_gf,
                                                                         float* __restrict__ out, int rows, int rank) {
    __shared__ __align__(16) float hs[kCuT][kCuHS];
    __shared__ __align__(16) uint16_t wg[kCuC][kCuWS];
    __shared__ __align__(16) uint16_t wf[kCuC][kCuWS];
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int t0 = (int) blockIdx.x * kCuT, c0 = (int) blockIdx.y * kCuC;

    float acc_g[4][4] = {}, acc_f[4][4] = {};
    for (int k0 = 0; k0 < rank; k0 += kCuK) {
        const int kend = min(kCuK, rank - k0);
        for (int q = tid; q < kCuT * (kCuK / 4); q += 32 * kCuW) {
            const int r = q / (kCuK / 4), k4 = (q % (kCuK / 4)) * 4;
            *reinterpret_cast<float4*>(&hs[r][k4]) =
                k4 < kend ? *reinterpret_cast<const float4*>(hid + (size_t) min(t0 + r, rows - 1) * rank + k0 + k4)
                          : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        for (int q = tid; q < kCuC * (kCuK / 8); q += 32 * kCuW) {
            const int r = q / (kCuK / 8), k8 = (q % (kCuK / 8)) * 8;
            *reinterpret_cast<uint4*>(&wg[r][k8]) =
                k8 < kend ? *reinterpret_cast<const uint4*>(w_gf + (size_t) (c0 + r) * rank + k0 + k8)
                          : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(&wf[r][k8]) =
                k8 < kend ? *reinterpret_cast<const uint4*>(w_gf + (size_t) (kWidth + c0 + r) * rank + k0 + k8)
                          : make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();
        const float* ha = &hs[wid * 16 + g][2 * c];
        const float* hb = &hs[wid * 16 + 8 + g][2 * c];
#pragma unroll
        for (int k = 0; k < kCuK; k += 8) {
            if (k >= kend) break;
            const float2 fa = *reinterpret_cast<const float2*>(ha + k);
            const float2 fb = *reinterpret_cast<const float2*>(hb + k);
            const unsigned a0 = tf32_of(fa.x), a1 = tf32_of(fb.x), a2 = tf32_of(fa.y), a3 = tf32_of(fb.y);
            const unsigned l0 = tf32_of(fa.x - __uint_as_float(a0)), l1 = tf32_of(fb.x - __uint_as_float(a1));
            const unsigned l2 = tf32_of(fa.y - __uint_as_float(a2)), l3 = tf32_of(fb.y - __uint_as_float(a3));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned vg = *reinterpret_cast<const unsigned*>(&wg[j * 8 + g][k + 2 * c]);
                const unsigned vf = *reinterpret_cast<const unsigned*>(&wf[j * 8 + g][k + 2 * c]);
                mma_tf32_2(acc_g[j], a0, a1, a2, a3, l0, l1, l2, l3, vg);
                mma_tf32_2(acc_f[j], a0, a1, a2, a3, l0, l1, l2, l3, vf);
            }
        }
        __syncthreads();
    }

    const int ta = t0 + wid * 16 + g, tb = ta + 8;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = c0 + j * 8 + 2 * c;
        if (ta < rows) {
            const size_t off = (size_t) ta * kWidth + col;
            const float2 d = *reinterpret_cast<const float2*>(dest + off);
            const float2 m = *reinterpret_cast<const float2*>(msg + off);
            *reinterpret_cast<float2*>(out + off) =
                make_float2((1.f - tanhf(acc_f[j][0])) * d.x + tanhf(acc_g[j][0]) * m.x,
                            (1.f - tanhf(acc_f[j][1])) * d.y + tanhf(acc_g[j][1]) * m.y);
        }
        if (tb < rows) {
            const size_t off = (size_t) tb * kWidth + col;
            const float2 d = *reinterpret_cast<const float2*>(dest + off);
            const float2 m = *reinterpret_cast<const float2*>(msg + off);
            *reinterpret_cast<float2*>(out + off) =
                make_float2((1.f - tanhf(acc_f[j][2])) * d.x + tanhf(acc_g[j][2]) * m.x,
                            (1.f - tanhf(acc_f[j][3])) * d.y + tanhf(acc_g[j][3]) * m.y);
        }
    }
}

}  // namespace

void ctrl_down_prompt(const float* dest, const float* msg, const uint16_t* w_down, int rows, int rank, float* hid, void* stream) {
    if (rows <= 0) return;
    if (rank <= 0 || rank % 16 != 0) throw std::runtime_error("inkling2 ctrl_down_prompt: rank");
    const dim3 grid((unsigned) ((rows + kCdT - 1) / kCdT), (unsigned) ((rank + kCdN - 1) / kCdN));
    ctrl_down_prompt_kernel<<<grid, 32 * kCdW, 0, (cudaStream_t) stream>>>(dest, msg, w_down, hid, rows, rank);
    check("ctrl_down_prompt");
}

void ctrl_up_blend_prompt(const float* dest, const float* msg, const float* hid, const uint16_t* w_gf, int rows, int rank,
                          float* out, void* stream) {
    if (rows <= 0) return;
    if (rank <= 0 || rank % 8 != 0) throw std::runtime_error("inkling2 ctrl_up_blend_prompt: rank");
    const dim3 grid((unsigned) ((rows + kCuT - 1) / kCuT), (unsigned) (kWidth / kCuC));
    ctrl_up_blend_prompt_kernel<<<grid, 32 * kCuW, 0, (cudaStream_t) stream>>>(dest, msg, hid, w_gf, out, rows, rank);
    check("ctrl_up_blend_prompt");
}

namespace {

// Matches block_q8_1_mmq (MMQ_Q8_1_DS_LAYOUT_D4): 4 float scales + 128 int8 codes = 144 bytes per 128 values.
// Writes 4 floats of a 32-value block (8 consecutive lanes -> 32 values) into prefill::q8 planes:
// int8 codes [rows][cols] and coalesced fp32 scales [cols / 32][sc_ld] (sc_ld = (rows + 63) & ~63).
__device__ __forceinline__ void emit_q8_planes_sub32(float4 v, int col, int r, int cols, int64_t sc_ld,
                                                     int8_t* __restrict__ codes, float* __restrict__ scales) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax * (1.0f / 127.0f);
    const float id = amax > 0.0f ? (127.0f / amax) : 0.0f;
    const uint32_t q0 = (uint32_t) (uint8_t) (int8_t) __float2int_rn(v.x * id);
    const uint32_t q1 = (uint32_t) (uint8_t) (int8_t) __float2int_rn(v.y * id);
    const uint32_t q2 = (uint32_t) (uint8_t) (int8_t) __float2int_rn(v.z * id);
    const uint32_t q3 = (uint32_t) (uint8_t) (int8_t) __float2int_rn(v.w * id);
    *reinterpret_cast<uint32_t*>(codes + (size_t) r * cols + col) = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
    if ((threadIdx.x & 7) == 0) scales[(int64_t) (col >> 5) * sc_ld + r] = d;
}

constexpr int kNormQ8Threads = 288;   // 288 * 8 floats = kWidth (2304); 2 float4s per thread, 8 threads per 32-block
constexpr int kNormQ8Vecs = (kWidth / 4) / kNormQ8Threads;
static_assert(kNormQ8Threads * 4 * kNormQ8Vecs == kWidth, "kWidth splits across kNormQ8Threads float4s");

__global__ void __launch_bounds__(kNormQ8Threads)
    rms_norm_q8_planes_prompt_kernel(const float* __restrict__ x, const float* __restrict__ w, int rows, float eps,
                                     int64_t sc_ld, int8_t* __restrict__ codes, float* __restrict__ scales) {
    __shared__ float red[kNormQ8Threads / 32];
    const int r = (int) blockIdx.x, tid = (int) threadIdx.x;
    const float4* x4 = reinterpret_cast<const float4*>(x + (size_t) r * kWidth);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4 xv[kNormQ8Vecs];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < kNormQ8Vecs; ++j) {
        const float4 v = x4[j * kNormQ8Threads + tid];
        xv[j] = v;
        ss = fmaf(v.x, v.x, ss);
        ss = fmaf(v.y, v.y, ss);
        ss = fmaf(v.z, v.z, ss);
        ss = fmaf(v.w, v.w, ss);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < kNormQ8Threads / 32; ++i) tot += red[i];
    const float inv = rsqrtf(tot / (float) kWidth + eps);
#pragma unroll
    for (int j = 0; j < kNormQ8Vecs; ++j) {
        const int c4 = j * kNormQ8Threads + tid;
        const float4 wv = w4[c4];
        const float4 nv = make_float4(xv[j].x * inv * wv.x, xv[j].y * inv * wv.y,
                                      xv[j].z * inv * wv.z, xv[j].w * inv * wv.w);
        emit_q8_planes_sub32(nv, c4 * 4, r, kWidth, sc_ld, codes, scales);
    }
}

constexpr int kGateQ8Threads = 256;
static_assert((kHeads * kHeadDim) % (4 * kGateQ8Threads) == 0, "kHeads * kHeadDim splits into float4 blocks");

__global__ void __launch_bounds__(kGateQ8Threads)
    attn_gate_q8_planes_prompt_kernel(const float* __restrict__ attn, const float* __restrict__ yq, int64_t yq_ld,
                                      int rows, int64_t sc_ld, int8_t* __restrict__ codes, float* __restrict__ scales) {
    const int r = (int) blockIdx.x;
    const int c4 = (int) blockIdx.y * kGateQ8Threads + (int) threadIdx.x;
    const int col = c4 * 4;
    const float4 a = *reinterpret_cast<const float4*>(attn + (size_t) r * (kHeads * kHeadDim) + col);
    const float4 g = *reinterpret_cast<const float4*>(yq + (size_t) r * yq_ld + kOffGate + col);
    const float4 v = make_float4(a.x / (1.f + expf(-g.x)), a.y / (1.f + expf(-g.y)),
                                 a.z / (1.f + expf(-g.z)), a.w / (1.f + expf(-g.w)));
    emit_q8_planes_sub32(v, col, r, kHeads * kHeadDim, sc_ld, codes, scales);
}

}  // namespace

void rms_norm_q8_planes_prompt(const float* x, const float* w, int rows, float eps, void* act, void* stream) {
    if (rows <= 0) return;
    const int64_t sc_ld = ((int64_t) rows + 63) & ~int64_t(63);
    const size_t sc_off = ((size_t) rows * kWidth + 15) & ~size_t(15);
    rms_norm_q8_planes_prompt_kernel<<<(unsigned) rows, kNormQ8Threads, 0, (cudaStream_t) stream>>>(
        x, w, rows, eps, sc_ld, static_cast<int8_t*>(act),
        reinterpret_cast<float*>(static_cast<uint8_t*>(act) + sc_off));
    check("rms_norm_q8_planes_prompt");
}

void attn_gate_q8_planes_prompt(const float* attn, const float* yq, int64_t yq_ld, int rows, void* act, void* stream) {
    if (rows <= 0) return;
    const int64_t sc_ld = ((int64_t) rows + 63) & ~int64_t(63);
    const size_t sc_off = ((size_t) rows * (kHeads * kHeadDim) + 15) & ~size_t(15);
    const dim3 grid((unsigned) rows, (unsigned) ((kHeads * kHeadDim) / (4 * kGateQ8Threads)));
    attn_gate_q8_planes_prompt_kernel<<<grid, kGateQ8Threads, 0, (cudaStream_t) stream>>>(
        attn, yq, yq_ld, rows, sc_ld, static_cast<int8_t*>(act),
        reinterpret_cast<float*>(static_cast<uint8_t*>(act) + sc_off));
    check("attn_gate_q8_planes_prompt");
}

namespace {

constexpr int kCombF16Threads = kWidth / 8;   // 288 threads: each owns 8 channels (1 uint4 fp16 -> 2 float4 fp32)
static_assert(kWidth % 8 == 0 && kCombF16Threads <= 1024, "kWidth splits into 8-half uint4s");

__global__ void __launch_bounds__(kCombF16Threads)
    moe_combine_f16_kernel(const uint16_t* __restrict__ out, const float* __restrict__ w, int k, float* __restrict__ y) {
    __shared__ float sw[32];
    const int r = (int) blockIdx.x, tid = (int) threadIdx.x;
    if (tid < k) sw[tid] = w[(size_t) r * k + tid];
    __syncthreads();
    const uint4* out8 = reinterpret_cast<const uint4*>(out);
    float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f), a1 = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int s = 0; s < k; ++s) {
        const float ws = sw[s];
        const uint4 u = out8[((size_t) r * k + s) * kCombF16Threads + tid];
        const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
        const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
        const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&u.z));
        const float2 f3 = __half22float2(*reinterpret_cast<const __half2*>(&u.w));
        a0.x = fmaf(ws, f0.x, a0.x); a0.y = fmaf(ws, f0.y, a0.y);
        a0.z = fmaf(ws, f1.x, a0.z); a0.w = fmaf(ws, f1.y, a0.w);
        a1.x = fmaf(ws, f2.x, a1.x); a1.y = fmaf(ws, f2.y, a1.y);
        a1.z = fmaf(ws, f3.x, a1.z); a1.w = fmaf(ws, f3.y, a1.w);
    }
    float4* y4 = reinterpret_cast<float4*>(y + (size_t) r * kWidth);
    y4[2 * tid] = a0;
    y4[2 * tid + 1] = a1;
}

}  // namespace

void moe_combine_f16(const uint16_t* out, const float* w, int rows, int k, float* y, void* stream) {
    if (rows <= 0) return;
    if (k <= 0 || k > 32) throw std::runtime_error("inkling2 moe_combine_f16: k");
    moe_combine_f16_kernel<<<(unsigned) rows, kCombF16Threads, 0, (cudaStream_t) stream>>>(out, w, k, y);
    check("moe_combine_f16");
}

}  // namespace basalt::models::inkling2
