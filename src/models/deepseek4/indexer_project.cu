#include "basalt/models/deepseek4/indexer.hpp"

#include "basalt/kernels/native_mmvq.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::deepseek4 {
namespace {

constexpr int D = kIndexerDim;
constexpr int H = kIndexerHeads;
constexpr int R = 64;
constexpr int QRank = kIndexerQueryRank;
constexpr int Embd = kIndexerEmbedding;
constexpr int TMax = 8;
constexpr int GemvWarps = 4;

__device__ __forceinline__ float bf16_to_float(uint16_t value) { return __uint_as_float((uint32_t) value << 16); }
__device__ __forceinline__ float bf16_rn(float value) { return __bfloat162float(__float2bfloat16_rn(value)); }

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

__device__ float block_max(float value, float* partial) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x / 32;
    __syncthreads();
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < warps ? partial[threadIdx.x] : 0.f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    if (threadIdx.x == 0) partial[0] = value;
    __syncthreads();
    value = partial[0];
    __syncthreads();
    return value;
}

__device__ __forceinline__ float pow2_ceil(float positive) {
    const uint32_t bits = __float_as_uint(positive);
    const int exponent = (int) ((bits >> 23) & 0xffu) - 127 + ((bits & 0x7fffffu) != 0u);
    return __uint_as_float((uint32_t) (exponent + 127) << 23);
}

__device__ __forceinline__ float fp4_e2m1_rn(float value) {
    constexpr float levels[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float magnitude = fabsf(value);
    int selected = 7;
#pragma unroll
    for (int i = 0; i < 7; ++i) {
        const float midpoint = (levels[i] + levels[i + 1]) * .5f;
        if (magnitude < midpoint || (magnitude == midpoint && (i & 1) == 0)) {
            selected = i;
            break;
        }
    }
    return copysignf(levels[selected], value);
}

__global__ void gemv_bf16_kernel(const uint16_t* __restrict__ weights, const float* __restrict__ x, int n_tokens,
                                 int in, int out, float* __restrict__ y) {
    const int token = (int) blockIdx.y;
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const int output = (int) blockIdx.x * GemvWarps + warp;
    if (token >= n_tokens || output >= out) return;
    float sum = 0.f;
    for (int k = lane; k < in; k += 32) {
        const float a = bf16_to_float(weights[(size_t) output * in + k]);
        const float b = bf16_rn(x[(size_t) token * in + k]);
        sum = fmaf(a, b, sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) y[(size_t) token * out + output] = bf16_rn(sum);
}

__global__ void round_bf16_kernel(const float* __restrict__ in, int n, float* __restrict__ out) {
    const int i = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = bf16_rn(in[i]);
}

__device__ __forceinline__ void rotate_pair(float& x0, float& x1, float frequency) {
    float sine, cosine;
    sincosf(frequency, &sine, &cosine);
    const float y0 = x0 * cosine - x1 * sine;
    const float y1 = x0 * sine + x1 * cosine;
    x0 = y0;
    x1 = y1;
}

__global__ void indexer_query_post_kernel(const float* __restrict__ raw, const float* __restrict__ rope_angle,
                                          int n_tokens, float* __restrict__ q_out) {
    __shared__ float values[D];
    __shared__ float hadamard_a[D];
    __shared__ float hadamard_b[D];
    __shared__ float partial[32];
    __shared__ float fp4_scales[D / 32];
    const int token = (int) blockIdx.x / H;
    const int head = (int) blockIdx.x % H;
    const int c = (int) threadIdx.x;
    if (token >= n_tokens) return;
    const float* q = raw + ((size_t) token * H + head) * D;
    const float* angle = rope_angle + (size_t) token * (R / 2);
    float value = bf16_rn(q[c]);
    if (c >= D - R) {
        const int pair = (c - (D - R)) >> 1;
        float x0 = bf16_rn(q[D - R + pair * 2]);
        float x1 = bf16_rn(q[D - R + pair * 2 + 1]);
        rotate_pair(x0, x1, angle[pair]);
        value = bf16_rn((c & 1) ? x1 : x0);
    }
    values[c] = value;
    __syncthreads();

    float* input = hadamard_a;
    float* output = hadamard_b;
    input[c] = values[c];
    __syncthreads();
    for (int stride = 1; stride < D; stride <<= 1) {
        const float own = input[c];
        const float other = input[c ^ stride];
        output[c] = bf16_rn((c & stride) ? other - own : own + other);
        __syncthreads();
        float* swap = input;
        input = output;
        output = swap;
    }
    values[c] = bf16_rn(input[c] * rsqrtf((float) D));
    __syncthreads();

    constexpr float kMinAmax = 6.f * 0x1p-126f;
    for (int group = 0; group < D / 32; ++group) {
        const float local = c >= group * 32 && c < (group + 1) * 32 ? fabsf(values[c]) : 0.f;
        const float amax = fmaxf(block_max(local, partial), kMinAmax);
        if (c == 0) fp4_scales[group] = pow2_ceil(amax * (1.f / 6.f));
        __syncthreads();
    }
    const float scale = fp4_scales[c / 32];
    const float quantized = fp4_e2m1_rn(fminf(6.f, fmaxf(-6.f, values[c] / scale))) * scale;
    q_out[((size_t) token * H + head) * D + c] = bf16_rn(quantized);
}

__global__ void scale_head_weights_kernel(const float* __restrict__ raw, int n_tokens, float* __restrict__ out) {
    const int i = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tokens * H) return;
    const float scale = (1.f / sqrtf((float) D)) * (1.f / sqrtf((float) H));
    out[i] = bf16_rn(raw[i] * scale);
}

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("deepseek4 indexer projection ") + operation + ": " +
                                                       cudaGetErrorString(error));
}

size_t align256(size_t bytes) { return (bytes + 255u) & ~size_t(255u); }

}  // namespace

size_t indexer_project_workspace_bytes(int max_tokens) {
    if (max_tokens < 1 || max_tokens > TMax) throw std::invalid_argument("deepseek4 indexer: invalid token capacity");
    const size_t q_bytes = (size_t) max_tokens * H * D * sizeof(float);
    const size_t w_bytes = (size_t) max_tokens * H * sizeof(float);
    const size_t latent_bytes = (size_t) max_tokens * QRank * sizeof(float);
    const size_t activation_bytes = kernels::native_q8_1_bytes(QRank, max_tokens);
    return align256(q_bytes) + align256(w_bytes) + align256(latent_bytes) + align256(activation_bytes) + 256;
}

void indexer_project_q8_0(const void* w_q, const uint16_t* w_proj, const float* qr, const float* x,
                          const float* rope_angle, int n_tokens, float* q_out, float* head_weights,
                          void* workspace, size_t workspace_bytes, void* stream) {
    if (!w_q || !w_proj || !qr || !x || !rope_angle || !q_out || !head_weights || !workspace || !stream ||
        n_tokens < 1 || n_tokens > TMax || workspace_bytes < indexer_project_workspace_bytes(n_tokens))
        throw std::invalid_argument("deepseek4 indexer Q8_0 projection: invalid arguments");
    auto* base = static_cast<unsigned char*>(workspace);
    float* q_raw = reinterpret_cast<float*>(base);
    float* w_raw = reinterpret_cast<float*>(base + align256((size_t) n_tokens * H * D * sizeof(float)));
    float* qr_bf16 = reinterpret_cast<float*>(base + align256((size_t) n_tokens * H * D * sizeof(float)) +
                                             align256((size_t) n_tokens * H * sizeof(float)));
    void* qr_q8_1 = base + align256((size_t) n_tokens * H * D * sizeof(float)) +
                    align256((size_t) n_tokens * H * sizeof(float)) +
                    align256((size_t) n_tokens * QRank * sizeof(float));
    const cudaStream_t s = (cudaStream_t) stream;
    round_bf16_kernel<<<((size_t) n_tokens * QRank + 255) / 256, 256, 0, s>>>(qr, n_tokens * QRank, qr_bf16);
    check(cudaGetLastError(), "Q8_0 query activation boundary launch");
    kernels::native_quantize_q8_1(qr_bf16, qr_q8_1, QRank, n_tokens, stream);
    kernels::native_q8_0_mmvq(w_q, qr_q8_1, q_raw, QRank, H * D, n_tokens, stream);
    gemv_bf16_kernel<<<dim3((H + GemvWarps - 1) / GemvWarps, n_tokens), 32 * GemvWarps, 0, s>>>(
        w_proj, x, n_tokens, Embd, H, w_raw);
    check(cudaGetLastError(), "head-weight projection launch");
    indexer_query_post_kernel<<<n_tokens * H, D, 0, s>>>(q_raw, rope_angle, n_tokens, q_out);
    check(cudaGetLastError(), "query transform launch");
    scale_head_weights_kernel<<<(n_tokens * H + 127) / 128, 128, 0, s>>>(w_raw, n_tokens, head_weights);
    check(cudaGetLastError(), "head-weight scaling launch");
}

void indexer_round_bf16(const float* qr, int n_tokens, float* qr_bf16, void* stream) {
    if (!qr || !qr_bf16 || !stream || n_tokens < 1)
        throw std::invalid_argument("deepseek4 indexer_round_bf16: invalid arguments");
    round_bf16_kernel<<<((size_t) n_tokens * QRank + 255) / 256, 256, 0, (cudaStream_t) stream>>>(
        qr, n_tokens * QRank, qr_bf16);
    check(cudaGetLastError(), "indexer_round_bf16 launch");
}

void indexer_project_post(const uint16_t* w_proj, const float* q_raw, const float* x,
                          const float* rope_angle, int n_tokens, float* w_raw, float* q_out,
                          float* head_weights, void* stream) {
    if (!w_proj || !q_raw || !x || !rope_angle || !w_raw || !q_out || !head_weights || !stream || n_tokens < 1)
        throw std::invalid_argument("deepseek4 indexer_project_post: invalid arguments");
    const cudaStream_t s = (cudaStream_t) stream;
    gemv_bf16_kernel<<<dim3((H + GemvWarps - 1) / GemvWarps, n_tokens), 32 * GemvWarps, 0, s>>>(
        w_proj, x, n_tokens, Embd, H, w_raw);
    check(cudaGetLastError(), "head-weight projection launch");
    indexer_query_post_kernel<<<n_tokens * H, D, 0, s>>>(q_raw, rope_angle, n_tokens, q_out);
    check(cudaGetLastError(), "query transform launch");
    scale_head_weights_kernel<<<(n_tokens * H + 127) / 128, 128, 0, s>>>(w_raw, n_tokens, head_weights);
    check(cudaGetLastError(), "head-weight scaling launch");
}

}  // namespace basalt::models::deepseek4
