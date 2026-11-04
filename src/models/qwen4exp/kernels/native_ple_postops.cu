// Arithmetic adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// the qwen4exp graph and ggml-cuda/{reduce_rows.cuh,sumrows.cu,unary.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "basalt/models/qwen4exp/kernels/native_ple_postops.hpp"
#include "basalt/models/qwen4exp/kernels/native_gr_norm.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"
#include "basalt/models/qwen4exp/kernels/ngram.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace basalt::kernels {
namespace {
constexpr int N = 2560, H = 4, D = N * H, HISTORY = 9;
__device__ float warp_sum(float x) {
    for (int offset = 16; offset; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}
__global__ void gate_kernel(const float* key, const float* query, float* gate, float scale) {
    // SUM_ROWS selects 512 threads for four rows on the target GPU. Preserve
    // its eight partial lanes and materialized MUL rounding (never a dot FMA).
    float sums[8] = {};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = threadIdx.x + j * 512;
        const float p = d < N ? __fmul_rn(key[blockIdx.x * N + d], query[blockIdx.x * N + d]) : 0.0f;
        sums[j] += p;
    }
    float sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += sums[j];
    __shared__ float partials[32];
    sum = warp_sum(sum);
    const int lane = threadIdx.x % 32;
    if (!lane) partials[threadIdx.x / 32] = sum;
    __syncthreads();
    sum = lane < 16 ? partials[lane] : 0.0f;
    sum = warp_sum(sum);
    if (threadIdx.x == 0) {
        const float s = __fmaf_rn(scale, sum, 0.0f); // ggml SCALE's zero bias
        const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        const float sign = float((s > 0.0f) - (s < 0.0f));
        gate[blockIdx.x] = 1.0f / (1.0f + expf(-__fmul_rn(sign, mag)));
    }
}
__global__ void broadcast_kernel(const float* value, const float* gate, float* gated) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) gated[i] = __fmul_rn(value[i % N], gate[i / N]);
}
__global__ void conv_residual_kernel(const float* history, const float* normalized,
                                    const uint16_t* weights, const float* hidden,
                                    const float* gated, float* conv, float* result) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float x = k == 3 ? normalized[c] : history[c * HISTORY + 3 * k];
        const float w = __half2float(__ushort_as_half(weights[c * 4 + k]));
        const float term = __fmul_rn(x, w);
        sum = k == 0 ? term : __fadd_rn(sum, term);
    }
    const float activation = sum / (1.0f + expf(-sum));
    conv[c] = activation;
    // Exact hidden/result alias is safe: each thread owns one element.
    result[c] = __fadd_rn(hidden[c], __fadd_rn(gated[c], activation));
}
// ---- the batch: T tokens, the same arithmetic per element as the kernels above
// weighted_rms_norm (native_gr_norm.cu) with the gamma row repeating every H rows (one token's H groups)
__device__ float norm_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}
__global__ void rms_rep_kernel(const float* __restrict__ input, const float* __restrict__ gamma,
                               float* __restrict__ output) {
    constexpr int BlockSize = 1024;             // native_gr_rms_norm_weighted's choice for 2560 columns
    const int tid = threadIdx.x;
    const size_t row_offset = size_t(blockIdx.x) * N;
    input += row_offset;
    output += row_offset;
    gamma += size_t(blockIdx.x % H) * N;
    float partial = 0.0f;
    for (int col = tid; col < N; col += BlockSize) {
        const float value = input[col];
        partial += value * value;
    }
    __shared__ float sums[32];
    partial = norm_warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    __syncthreads();
    partial = 0.0f;
    if (lane < BlockSize / 32) partial = sums[lane];
    partial = norm_warp_sum(partial);
    const float mean = partial / N;
    const float scale = rsqrtf(mean + NG_RMS_EPS);
    for (int col = tid; col < N; col += BlockSize) output[col] = scale * input[col] * gamma[col];
}
__global__ void broadcast_batch_kernel(const float* value, const float* gate, float* gated, int T) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(T) * D) return;
    const size_t t = i / D, d = i % D;
    gated[i] = __fmul_rn(value[t * N + d % N], gate[t * H + d / N]);
}
// the dilated conv (taps 9, 6, 3 tokens back and this one) and the residual; a tap before the chunk reads the history
__global__ void conv_residual_batch_kernel(const float* history, const float* normalized, const uint16_t* weights,
                                           float* hidden, const float* gated, int T) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(T) * D) return;
    const int t = int(i / D), c = int(i % D);
    float sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int p = t - 9 + 3 * k;             // the token this tap reads (k == 3: this one)
        const float x = p >= 0 ? normalized[size_t(p) * D + c] : history[size_t(c) * HISTORY + (9 + p)];
        const float wk = __half2float(__ushort_as_half(weights[c * 4 + k]));
        const float term = __fmul_rn(x, wk);
        sum = k == 0 ? term : __fadd_rn(sum, term);
    }
    const float activation = sum / (1.0f + expf(-sum));
    hidden[i] = __fadd_rn(hidden[i], __fadd_rn(gated[i], activation));
}
// the history after the chunk: the last nine normalized rows (older ones from the history when T < 9)
__global__ void history_batch_kernel(float* history, const float* normalized, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float h[HISTORY];
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) {
        const int p = T - HISTORY + r;
        h[r] = p >= 0 ? normalized[size_t(p) * D + c] : history[size_t(c) * HISTORY + (T + r)];
    }
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) history[size_t(c) * HISTORY + r] = h[r];
}

// ---- the verify window: native_gr_post's write, the rows' norms and gate (block per row), then the conv, residual
// and history (thread per channel); each expression as the per-token kernels write it
__device__ __forceinline__ float post_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
// weighted_rms_norm<1024>'s scale for a row of n_cols held as x[j] = column tid + 1024 j
__device__ __forceinline__ float win_rms_scale(const float (&x)[3], int n_cols, float epsilon, float* sums) {
    const int tid = threadIdx.x;
    float partial = 0.0f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int col = tid + j * 1024;
        if (col < n_cols) {
            const float value = x[j];
            partial += value * value;
        }
    }
    partial = norm_warp_sum(partial);
    const int lane = tid % 32;
    __syncthreads();   // sums of a previous call are read
    if (lane == 0) sums[tid / 32] = partial;
    __syncthreads();
    partial = 0.0f;
    if (lane < 1024 / 32) partial = sums[lane];
    partial = norm_warp_sum(partial);
    const float mean = partial / n_cols;
    return rsqrtf(mean + epsilon);
}
struct PleWinRows {
    float* R; const float* bo; const float* inject; const float* key; const float* value;
    const float *norm_key, *norm_query, *norm_conv;
    float *gated, *normalized;
    int n_cols; float epsilon, inv_hc, gate_scale;
};
// row (t, h): block (h, t) of 1024 threads
__device__ __forceinline__ void ple_win_rows_dev(const PleWinRows& a, int h, int t) {
    __shared__ float sums[32];
    __shared__ float kq[2][N];
    __shared__ float s_gate;
    const int tid = threadIdx.x, n = a.n_cols;
    const size_t row = size_t(t) * D + size_t(h) * N;
    const float weight = __fmaf_rn(2.0f, post_sigmoid(__fmaf_rn(a.inject[t * H + h], a.inv_hc, 0.0f)), 0.0f);
    float hid[3], kx[3], x[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int col = tid + j * 1024;
        hid[j] = 0.0f; kx[j] = 0.0f;
        if (col < n) {
            hid[j] = __fmaf_rn(a.bo[size_t(t) * N + col], weight, a.R[row + col]);
            a.R[row + col] = hid[j];
            kx[j] = a.key[row + col];
        }
    }
    float scale = win_rms_scale(kx, n, a.epsilon, sums);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int col = tid + j * 1024;
        if (col < n) kq[0][col] = scale * kx[j] * a.norm_key[size_t(h) * N + col];
    }
    scale = win_rms_scale(hid, n, a.epsilon, sums);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int col = tid + j * 1024;
        if (col < n) kq[1][col] = scale * hid[j] * a.norm_query[size_t(h) * N + col];
    }
    __syncthreads();
    if (tid < 512) {   // gate_kernel's 512 threads
        float s8[8] = {};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = tid + j * 512;
            const float p = d < N ? __fmul_rn(kq[0][d], kq[1][d]) : 0.0f;
            s8[j] += p;
        }
        float sum = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sum += s8[j];
        sum = warp_sum(sum);
        if (tid % 32 == 0) sums[tid / 32] = sum;
    }
    __syncthreads();
    if (tid < 32) {
        float sum = tid < 16 ? sums[tid] : 0.0f;
        sum = warp_sum(sum);
        if (tid == 0) {
            const float s = __fmaf_rn(a.gate_scale, sum, 0.0f);
            const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
            const float sign = float((s > 0.0f) - (s < 0.0f));
            s_gate = 1.0f / (1.0f + expf(-__fmul_rn(sign, mag)));
        }
    }
    __syncthreads();
    const float gate = s_gate;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int col = tid + j * 1024;
        x[j] = 0.0f;
        if (col < n) {
            x[j] = __fmul_rn(a.value[size_t(t) * N + col], gate);
            a.gated[row + col] = x[j];
        }
    }
    scale = win_rms_scale(x, n, a.epsilon, sums);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int col = tid + j * 1024;
        if (col < n) a.normalized[row + col] = scale * x[j] * a.norm_conv[size_t(h) * N + col];
    }
}
__global__ void __launch_bounds__(1024) ple_win_rows_kernel(PleWinRows a) { ple_win_rows_dev(a, blockIdx.x, blockIdx.y); }
// channel c: the T rows' conv and residual in order, each row's advanced history copied out, the last kept
__device__ __forceinline__ void ple_win_conv_dev(float* R, const float* gated, const float* normalized,
                                                 const uint16_t* weights, float* history, float* snap, int T, int c) {
    if (c >= D) return;
    float hist[HISTORY], wk[4];
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) hist[r] = history[size_t(c) * HISTORY + r];
#pragma unroll
    for (int k = 0; k < 4; ++k) wk[k] = __half2float(__ushort_as_half(weights[c * 4 + k]));
    float last[HISTORY];
    for (int t = 0; t < T; ++t) {
        float sum = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int p = t - 9 + 3 * k;
            const float x = p >= 0 ? normalized[size_t(p) * D + c] : hist[9 + p];
            const float term = __fmul_rn(x, wk[k]);
            sum = k == 0 ? term : __fadd_rn(sum, term);
        }
        const float activation = sum / (1.0f + expf(-sum));
        const size_t i = size_t(t) * D + c;
        R[i] = __fadd_rn(R[i], __fadd_rn(gated[i], activation));
#pragma unroll
        for (int r = 0; r < HISTORY; ++r) {
            const int p = t - (HISTORY - 1) + r;
            last[r] = p >= 0 ? normalized[size_t(p) * D + c] : hist[t + 1 + r];
            snap[size_t(t) * HISTORY * D + size_t(c) * HISTORY + r] = last[r];
        }
    }
#pragma unroll
    for (int r = 0; r < HISTORY; ++r) history[size_t(c) * HISTORY + r] = last[r];
}
__global__ void __launch_bounds__(128) ple_win_conv_kernel(float* R, const float* gated, const float* normalized,
                                                          const uint16_t* weights, float* history, float* snap, int T) {
    ple_win_conv_dev(R, gated, normalized, weights, history, snap, T, blockIdx.x * blockDim.x + threadIdx.x);
}

// the rows are segments of different sequences: a segment's first row runs its rows' conv in order over the history
// held at that row, the others are idle
__global__ void __launch_bounds__(128) ple_win_conv_seg_kernel(float* R, const float* gated, const float* normalized,
                                                              const uint16_t* weights, float* history, float* snap,
                                                              const SegRow* __restrict__ rows) {
    const SegRow r = rows[blockIdx.y];
    if (r.j != 0) return;
    const size_t t = r.row0;
    ple_win_conv_dev(R + t * D, gated + t * D, normalized + t * D, weights, history + t * HISTORY * D,
                     snap + t * HISTORY * D, r.n, blockIdx.x * blockDim.x + threadIdx.x);
}

struct Span { const void* p; size_t bytes; size_t alignment; };
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.bytes && y < x + a.bytes;
}
void validate(Span span) {
    const auto p = reinterpret_cast<uintptr_t>(span.p);
    if (!p || p % span.alignment || p > std::numeric_limits<uintptr_t>::max() - span.bytes)
        throw std::invalid_argument("native PLE postops require nonnull aligned bounded spans");
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string("native PLE postops launch: ") + cudaGetErrorString(error));
}
} // namespace

void native_ple_postops(const float* projected_key, const float* hidden,
                        const float* value, const float* history,
                        const PleWeights& w, const NativePlePostopsBuffers& b, void* stream) {
    if (!stream) throw std::invalid_argument("native PLE postops require an explicit stream");
    const Span inputs[] = {{projected_key,D*4,4}, {hidden,D*4,4}, {value,N*4,4},
        {history,HISTORY*D*4,4}, {w.norm_key,D*4,4}, {w.norm_query,D*4,4},
        {w.norm_conv,D*4,4}, {w.conv1d_f16,4*D*2,2}};
    const Span outputs[] = {{b.key,D*4,4}, {b.query,D*4,4}, {b.gate,H*4,4},
        {b.gated,D*4,4}, {b.normalized,D*4,4}, {b.conv,D*4,4}, {b.result,D*4,4}};
    for (const auto& span : inputs) validate(span);
    for (const auto& span : outputs) validate(span);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 8; ++j)
            if (!(i == 6 && j == 1 && b.result == hidden) && overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("native PLE postops output overlaps an input or weight");
        for (size_t j = i + 1; j < 7; ++j)
            if (!(i == 1 && j == 4 && b.query == b.normalized) && overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("native PLE postops writable spans overlap");
    }
    native_gr_rms_norm_weighted(projected_key,w.norm_key,b.key,N,H,NG_RMS_EPS,stream);
    native_gr_rms_norm_weighted(hidden,w.norm_query,b.query,N,H,NG_RMS_EPS,stream);
    auto st = static_cast<cudaStream_t>(stream);
    gate_kernel<<<H,512,0,st>>>(b.key,b.query,b.gate,1.0f / std::sqrt(float(N)));
    broadcast_kernel<<<D/256,256,0,st>>>(value,b.gate,b.gated);
    launch_check();
    native_gr_rms_norm_weighted(b.gated,w.norm_conv,b.normalized,N,H,NG_RMS_EPS,stream);
    conv_residual_kernel<<<D/256,256,0,st>>>(history,b.normalized,w.conv1d_f16,hidden,b.gated,b.conv,b.result);
    launch_check();
}

void native_ple_window(const PleWindowArgs& a, const PleWeights& w, void* stream) {
    if (!stream || a.T <= 0 || !a.R || !a.block_out || !a.inject || !a.key || !a.value || !a.history || !a.hist_snap ||
        !a.scratch || !w.norm_key || !w.norm_query || !w.norm_conv || !w.conv1d_f16)
        throw std::invalid_argument("native PLE window: null input or empty window");
    auto st = static_cast<cudaStream_t>(stream);
    PleWinRows r;
    r.R = a.R; r.bo = a.block_out; r.inject = a.inject; r.key = a.key; r.value = a.value;
    r.norm_key = w.norm_key; r.norm_query = w.norm_query; r.norm_conv = w.norm_conv;
    r.gated = a.scratch; r.normalized = a.scratch + size_t(a.T) * D;
    r.n_cols = N; r.epsilon = NG_RMS_EPS; r.inv_hc = 1.0f / float(H); r.gate_scale = 1.0f / std::sqrt(float(N));
    ple_win_rows_kernel<<<dim3(H, unsigned(a.T)), 1024, 0, st>>>(r);
    if (a.segs)
        ple_win_conv_seg_kernel<<<dim3(D / 128, unsigned(a.T)), 128, 0, st>>>(a.R, r.gated, r.normalized, w.conv1d_f16,
                                                                              a.history, a.hist_snap, a.segs);
    else
        ple_win_conv_kernel<<<D / 128, 128, 0, st>>>(a.R, r.gated, r.normalized, w.conv1d_f16, a.history, a.hist_snap, a.T);
    launch_check();
}

void native_ple_postops_batch(float* key, float* hidden, const float* value, float* history, const PleWeights& w,
                              float* query_norm, float* gated, float* gate, int T, void* stream) {
    if (!stream || T <= 0 || !key || !hidden || !value || !history || !query_norm || !gated || !gate)
        throw std::invalid_argument("native PLE postops batch: null input or empty batch");
    auto st = static_cast<cudaStream_t>(stream);
    const unsigned rows = unsigned(T) * H;
    const unsigned blocks = unsigned((size_t(T) * D + 255) / 256);
    rms_rep_kernel<<<rows, 1024, 0, st>>>(key, w.norm_key, key);
    rms_rep_kernel<<<rows, 1024, 0, st>>>(hidden, w.norm_query, query_norm);
    gate_kernel<<<rows, 512, 0, st>>>(key, query_norm, gate, 1.0f / std::sqrt(float(N)));
    broadcast_batch_kernel<<<blocks, 256, 0, st>>>(value, gate, gated, T);
    rms_rep_kernel<<<rows, 1024, 0, st>>>(gated, w.norm_conv, query_norm);
    conv_residual_batch_kernel<<<blocks, 256, 0, st>>>(history, query_norm, w.conv1d_f16, hidden, gated, T);
    history_batch_kernel<<<D / 256, 256, 0, st>>>(history, query_norm, T);
    launch_check();
}
} // namespace basalt::kernels
