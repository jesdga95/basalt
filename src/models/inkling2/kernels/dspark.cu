// src/models/inkling2/kernels/dspark.cu - the DSpark drafter's own kernels (basalt/models/inkling2/dspark_kernels.hpp).
#include "basalt/models/inkling2/dspark_kernels.hpp"

#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

constexpr int kThreads = 256;
constexpr int kPer = kWidth / kThreads;   // 9
constexpr int kQ8Block = 34;              // ggml's block_q8_0: an fp16 scale, then 32 codes
constexpr int kQ8Row = kWidth / 32 * kQ8Block;
static_assert(kWidth % kThreads == 0, "the width splits evenly over a block");

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 dspark ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float bf16f(uint16_t u) { return __uint_as_float((uint32_t) u << 16); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

// The sum of one value per thread over a block of kThreads, in a fixed order; every thread gets it.
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[w] = v;
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < kThreads / 32; ++j) s += red[j];
    __syncthreads();
    return s;
}

// ---- the mixes of the target's features -------------------------------------------------------------------------------

// Row r's mix j.
__device__ __forceinline__ void mix_stage(int r, int j, const float* __restrict__ tap, int64_t tap_stride, const DsMixArgs& a,
                                          float* __restrict__ out, int64_t out_stride, const int32_t* __restrict__ pos,
                                          float* __restrict__ fr, int fr_cap, int32_t* __restrict__ pos3, float* red) {
    float al[kDsTargets];
#pragma unroll
    for (int k = 0; k < kDsTargets; ++k) al[k] = a.alpha[j * kDsTargets + k];
    float v[kPer];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * kThreads;
        float m = 0.f;
#pragma unroll
        for (int k = 0; k < kDsTargets; ++k) m += al[k] * tap[(int64_t) k * tap_stride + (int64_t) r * kWidth + c];
        v[i] = m;
        ss += m * m;
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float) kWidth + a.eps);
    const int32_t p = pos[r];
    float* dst = j < kDsMixes - 1 ? out + j * out_stride + (int64_t) r * kWidth : fr + (int64_t) (p % fr_cap) * kWidth;
    const float* w = a.norm[j];
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * kThreads;
        dst[c] = v[i] * inv * w[c];
    }
    if (j == 0 && threadIdx.x < 3) pos3[r * 3 + threadIdx.x] = p;
}

__global__ void __launch_bounds__(kThreads) mix_kernel(const float* __restrict__ tap, int64_t tap_stride, DsMixArgs a,
                                                       float* __restrict__ out, int64_t out_stride,
                                                       const int32_t* __restrict__ pos, float* __restrict__ fr, int fr_cap,
                                                       int32_t* __restrict__ pos3) {
    __shared__ float red[kThreads / 32];
    mix_stage((int) blockIdx.x, (int) blockIdx.y, tap, tap_stride, a, out, out_stride, pos, fr, fr_cap, pos3, red);
}

// ---- the block's inputs -------------------------------------------------------------------------------------------------

// Block row t: the anchor's normed embedding (row 0) or the mask row.
__device__ __forceinline__ void block_inputs_stage(int t, const uint16_t* __restrict__ emb, int32_t anchor,
                                                   const float* __restrict__ norm, float eps, const float* __restrict__ mask,
                                                   int64_t pos0, float* __restrict__ x, int32_t* __restrict__ pos3,
                                                   int32_t* __restrict__ prev, float* __restrict__ state, float* red) {
    float* row = x + (int64_t) t * kWidth;
    if (threadIdx.x < 3) pos3[t * 3 + threadIdx.x] = (int32_t) (pos0 + t);
    if (t == 1) {   // the Markov head's start
        if (threadIdx.x == 0) *prev = anchor;
        for (int i = threadIdx.x; i < kDsRank; i += kThreads) state[i] = 0.f;
    }
    if (t > 0) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) row[threadIdx.x + i * kThreads] = mask[threadIdx.x + i * kThreads];
        return;
    }
    const uint16_t* e = emb + (int64_t) anchor * kWidth;
    float v[kPer];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        v[i] = bf16f(e[threadIdx.x + i * kThreads]);
        ss += v[i] * v[i];
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float) kWidth + eps);
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * kThreads;
        row[c] = v[i] * inv * norm[c];
    }
}

__global__ void __launch_bounds__(kThreads) block_inputs_kernel(const uint16_t* __restrict__ emb, int32_t anchor,
                                                                const float* __restrict__ norm, float eps,
                                                                const float* __restrict__ mask, int64_t pos0,
                                                                float* __restrict__ x, int32_t* __restrict__ pos3,
                                                                int32_t* __restrict__ prev, float* __restrict__ state) {
    __shared__ float red[kThreads / 32];
    block_inputs_stage((int) blockIdx.x, emb, anchor, norm, eps, mask, pos0, x, pos3, prev, state, red);
}

// ---- the Markov head's recurrence ---------------------------------------------------------------------------------------

constexpr int kStepWarps = 32;                     // a warp per recurrence index
constexpr int kStepBlocks = kDsRank / kStepWarps;  // 8

// Block b: recurrence indices 32 b .. 32 b + 31, each with its gate, candidate and latent rows.
__device__ __forceinline__ void markov_step_stage(int b, const float* __restrict__ ctx_h, const float* __restrict__ ctx_f,
                                                  const float* __restrict__ bias, const uint16_t* __restrict__ w_sp,
                                                  const uint16_t* __restrict__ w1, const int32_t* __restrict__ prev,
                                                  const float* __restrict__ state_in, float* __restrict__ state_out,
                                                  float* __restrict__ latent, float* xs) {
    const int tok = *prev;
    for (int i = threadIdx.x; i < 2 * kDsRank; i += kStepWarps * 32)
        xs[i] = i < kDsRank ? state_in[i] : bf16f(w1[(int64_t) tok * kDsRank + i - kDsRank]);
    __syncthreads();
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = b * kStepWarps + w;
    float raw[3];
#pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int o = g * kDsRank + j;
        const uint32_t* row = reinterpret_cast<const uint32_t*>(w_sp + (int64_t) o * 2 * kDsRank);   // pairs of BF16
        float acc = 0.f;
#pragma unroll
        for (int k = lane; k < kDsRank; k += 32) {
            const uint32_t u = row[k];
            acc += __uint_as_float(u << 16) * xs[2 * k] + __uint_as_float(u & 0xffff0000u) * xs[2 * k + 1];
        }
        raw[g] = warp_sum(acc) + ctx_h[o] + ctx_f[o] + bias[o];
    }
    if (lane == 0) {
        const float gate = 1.f / (1.f + expf(-raw[0]));
        state_out[j] = gate * xs[j] + (1.f - gate) * tanhf(raw[1]);
        latent[j] = tanhf(raw[2]);
    }
}

__global__ void __launch_bounds__(kStepWarps * 32) markov_step_kernel(const float* __restrict__ ctx_h, const float* __restrict__ ctx_f,
                                                                      const float* __restrict__ bias, const uint16_t* __restrict__ w_sp,
                                                                      const uint16_t* __restrict__ w1, const int32_t* __restrict__ prev,
                                                                      const float* __restrict__ state_in, float* __restrict__ state_out,
                                                                      float* __restrict__ latent) {
    __shared__ float xs[2 * kDsRank];
    markov_step_stage((int) blockIdx.x, ctx_h, ctx_f, bias, w_sp, w1, prev, state_in, state_out, latent, xs);
}

// ---- the pick -----------------------------------------------------------------------------------------------------------

constexpr int kPickThreads = 256;
constexpr int kPickPer = 4;                         // logits a thread reads: a block takes 1,024
constexpr int kPickSpan = kPickThreads * kPickPer;
constexpr float kNegInf = -3.0e38f;                 // no logit yet: softcapped logits lie within +-cap

int pick_blocks(int n) { return (n + kPickSpan - 1) / kPickSpan; }

// (m, idx, s): the best logit, its lowest index, and the sum of exp(l - m) over the logits seen. Folds (m2, i2, s2) in.
__device__ __forceinline__ void pick_fold(float& m, int& idx, float& s, float m2, int i2, float s2) {
    if (m2 == kNegInf) return;
    if (m == kNegInf) {
        m = m2;
        idx = i2;
        s = s2;
        return;
    }
    const float mm = fmaxf(m, m2);
    s = s * __expf(m - mm) + s2 * __expf(m2 - mm);
    if (m2 > m || (m2 == m && i2 < idx)) idx = i2;
    m = mm;
}

// The block's fold of one (m, idx, s) per thread; thread 0 gets the result.
__device__ __forceinline__ void pick_block(float& m, int& idx, float& s, float* sm, int* si, float* ss) {
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        const float m2 = __shfl_xor_sync(0xffffffffu, m, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, idx, o);
        const float s2 = __shfl_xor_sync(0xffffffffu, s, o);
        pick_fold(m, idx, s, m2, i2, s2);
    }
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) {
        sm[w] = m;
        si[w] = idx;
        ss[w] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0)
        for (int k = 1; k < kPickThreads / 32; ++k) pick_fold(m, idx, s, sm[k], si[k], ss[k]);
}

__global__ void __launch_bounds__(kPickThreads) pick_kernel(const float* __restrict__ base, const float* __restrict__ bias,
                                                            const float* __restrict__ fold_b, int n, float cap,
                                                            float* part_m, int* part_i, float* part_s, unsigned* counter,
                                                            int32_t* __restrict__ prev, int32_t* __restrict__ ids,
                                                            float* __restrict__ conf, int at) {
    __shared__ float sm[kPickThreads / 32], ss[kPickThreads / 32];
    __shared__ int si[kPickThreads / 32];
    __shared__ bool last;
    const float inv = 1.f / cap;
    float m = kNegInf, s = 0.f;
    int idx = INT_MAX;
    const int v0 = (int) blockIdx.x * kPickSpan + (int) threadIdx.x;
#pragma unroll
    for (int k = 0; k < kPickPer; ++k) {
        const int v = v0 + k * kPickThreads;
        if (v >= n) break;
        const float l = cap * tanhf((base[v] + bias[v] + fold_b[v]) * inv);
        if (l > m) {
            s = (m == kNegInf ? 0.f : s * __expf(m - l)) + 1.f;
            m = l;
            idx = v;
        } else {
            s += __expf(l - m);
        }
    }
    pick_block(m, idx, s, sm, si, ss);
    if (threadIdx.x == 0) {
        part_m[blockIdx.x] = m;
        part_i[blockIdx.x] = idx;
        part_s[blockIdx.x] = s;
        __threadfence();
        last = atomicAdd(counter, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;
    m = kNegInf;
    s = 0.f;
    idx = INT_MAX;
    for (int b = (int) threadIdx.x; b < (int) gridDim.x; b += kPickThreads)
        pick_fold(m, idx, s, __ldcg(part_m + b), __ldcg(part_i + b), __ldcg(part_s + b));
    __syncthreads();
    pick_block(m, idx, s, sm, si, ss);
    if (threadIdx.x == 0) {
        *prev = idx;
        ids[at] = idx;
        conf[at] = 1.f / s;
        *counter = 0u;
    }
}

// ---- the folds, once at load ---------------------------------------------------------------------------------------------

constexpr int kFoldRows = 32;   // head rows of a block
constexpr int kFoldK = 64;      // the width a step stages

// Block b: head rows 32 b .. 32 b + 31 times w; thread t is column t of the rank. Each warp's 32 columns of a row are one
// Q8_0 block of the output.
__device__ __forceinline__ void fold_stage(int b, const uint8_t* __restrict__ head, int64_t n_rows, const uint16_t* __restrict__ w,
                                           uint8_t* __restrict__ fold, float (*hs)[kFoldK]) {
    const int64_t r0 = (int64_t) b * kFoldRows;
    const int t = threadIdx.x;
    float acc[kFoldRows];
#pragma unroll
    for (int r = 0; r < kFoldRows; ++r) acc[r] = 0.f;
    for (int e0 = 0; e0 < kWidth; e0 += kFoldK) {
        for (int idx = t; idx < kFoldRows * kFoldK; idx += kDsRank) {
            const int rr = idx / kFoldK, e = e0 + idx % kFoldK;
            const int64_t row = r0 + rr;
            float val = 0.f;
            if (row < n_rows) {
                const uint8_t* blk = head + row * kQ8Row + (int64_t) (e / 32) * kQ8Block;
                val = __half2float(*reinterpret_cast<const __half*>(blk)) * (float) (int8_t) blk[2 + e % 32];
            }
            hs[rr][idx % kFoldK] = val;
        }
        __syncthreads();
        float wv[kFoldK];
#pragma unroll
        for (int e = 0; e < kFoldK; ++e) wv[e] = bf16f(w[(int64_t) (e0 + e) * kDsRank + t]);
#pragma unroll
        for (int r = 0; r < kFoldRows; ++r) {
            float a = acc[r];
#pragma unroll
            for (int e = 0; e < kFoldK; ++e) a += hs[r][e] * wv[e];
            acc[r] = a;
        }
        __syncthreads();
    }
    const int lane = t & 31, wb = t >> 5;
#pragma unroll
    for (int r = 0; r < kFoldRows; ++r) {
        const int64_t row = r0 + r;
        if (row >= n_rows) break;
        const float x = acc[r];
        const float d = warp_max(fabsf(x)) / 127.f;   // ggml's quantize_row_q8_0_ref
        const float id = d != 0.f ? 1.f / d : 0.f;
        uint8_t* blk = fold + row * (kDsRank / 32 * kQ8Block) + (int64_t) wb * kQ8Block;
        blk[2 + lane] = (uint8_t) (int8_t) roundf(x * id);
        if (lane == 0) *reinterpret_cast<__half*>(blk) = __float2half(d);
    }
}

__global__ void __launch_bounds__(kDsRank) fold_kernel(const uint8_t* __restrict__ head, int64_t n_rows,
                                                       const uint16_t* __restrict__ w, uint8_t* __restrict__ fold) {
    __shared__ float hs[kFoldRows][kFoldK];
    fold_stage((int) blockIdx.x, head, n_rows, w, fold, hs);
}

// A warp per head row: fold_b[row] = head[row] . b.
__global__ void __launch_bounds__(256) fold_b_kernel(const uint8_t* __restrict__ head, int64_t n_rows, const float* __restrict__ b,
                                                     float* __restrict__ out) {
    const int64_t row = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n_rows) return;
    const uint8_t* r = head + row * kQ8Row;
    float acc = 0.f;
    for (int k = 0; k < kWidth / 32; ++k) {
        const uint8_t* blk = r + (int64_t) k * kQ8Block;
        acc += __half2float(*reinterpret_cast<const __half*>(blk)) * (float) (int8_t) blk[2 + lane] * b[k * 32 + lane];
    }
    acc = warp_sum(acc);
    if (lane == 0) out[row] = acc;
}

__global__ void __launch_bounds__(256) split_rnn_kernel(const uint16_t* __restrict__ w, uint16_t* __restrict__ w_sp,
                                                        uint16_t* __restrict__ w_h, uint16_t* __restrict__ w_f) {
    constexpr int kIn = 2 * kDsRank + 2 * kWidth;
    const int64_t o = blockIdx.x;
    const uint16_t* src = w + o * kIn;
    for (int c = threadIdx.x; c < kIn; c += 256) {
        const uint16_t v = src[c];
        if (c < 2 * kDsRank) w_sp[o * 2 * kDsRank + c] = v;
        else if (c < 2 * kDsRank + kWidth) w_h[o * kWidth + c - 2 * kDsRank] = v;
        else w_f[o * kWidth + c - 2 * kDsRank - kWidth] = v;
    }
}

}  // namespace

void dspark_mix(const float* tap, int64_t tap_stride, int rows, const DsMixArgs& a, float* out, int64_t out_stride,
                const int32_t* pos, float* fr, int fr_cap, int32_t* pos3, void* stream) {
    if (rows <= 0) return;
    mix_kernel<<<dim3((unsigned) rows, kDsMixes), kThreads, 0, (cudaStream_t) stream>>>(tap, tap_stride, a, out, out_stride, pos, fr,
                                                                                         fr_cap, pos3);
    check("mix");
}

void dspark_block_inputs(const uint16_t* embedding, int32_t anchor, const float* embed_norm, float eps, const float* mask,
                         int block, int64_t pos0, float* x, int32_t* pos3, int32_t* prev, float* state, void* stream) {
    if (block < 2) throw std::runtime_error("inkling2 dspark block inputs: a block of at least two rows");
    block_inputs_kernel<<<(unsigned) block, kThreads, 0, (cudaStream_t) stream>>>(embedding, anchor, embed_norm, eps, mask, pos0,
                                                                                    x, pos3, prev, state);
    check("block inputs");
}

void dspark_markov_step(const float* ctx_h, const float* ctx_f, const float* bias, const uint16_t* w_sp, const uint16_t* w1,
                        const int32_t* prev, const float* state_in, float* state_out, float* latent, void* stream) {
    markov_step_kernel<<<kStepBlocks, kStepWarps * 32, 0, (cudaStream_t) stream>>>(ctx_h, ctx_f, bias, w_sp, w1, prev, state_in,
                                                                                    state_out, latent);
    check("Markov step");
}

size_t dspark_pick_scratch_bytes(int n_vocab) {
    return (size_t) pick_blocks(n_vocab) * (2 * sizeof(float) + sizeof(int)) + 16;
}

void dspark_pick(const float* base, const float* bias, const float* fold_b, int n_vocab, float cap, void* scratch,
                 int32_t* prev, int32_t* ids, float* conf, int i, void* stream) {
    const int nb = pick_blocks(n_vocab);
    unsigned* counter = static_cast<unsigned*>(scratch);   // zero when allocated; the last block leaves it zero
    float* pm = reinterpret_cast<float*>(static_cast<char*>(scratch) + 16);
    float* ps = pm + nb;
    int* pi = reinterpret_cast<int*>(ps + nb);
    pick_kernel<<<(unsigned) nb, kPickThreads, 0, (cudaStream_t) stream>>>(base, bias, fold_b, n_vocab, cap, pm, pi, ps, counter,
                                                                            prev, ids, conf, i);
    check("pick");
}

void dspark_fold(const void* head_q8_0, int64_t n_rows, const uint16_t* w, const float* b, void* fold_q8_0, float* fold_b,
                 void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    fold_kernel<<<(unsigned) ((n_rows + kFoldRows - 1) / kFoldRows), kDsRank, 0, s>>>(
        static_cast<const uint8_t*>(head_q8_0), n_rows, w, static_cast<uint8_t*>(fold_q8_0));
    check("fold");
    fold_b_kernel<<<(unsigned) ((n_rows + 7) / 8), 256, 0, s>>>(static_cast<const uint8_t*>(head_q8_0), n_rows, b, fold_b);
    check("fold bias");
}

void dspark_split_rnn(const uint16_t* w, uint16_t* w_sp, uint16_t* w_h, uint16_t* w_f, void* stream) {
    split_rnn_kernel<<<kDsRnn, 256, 0, (cudaStream_t) stream>>>(w, w_sp, w_h, w_f);
    check("split");
}

}  // namespace basalt::models::inkling2
