// src/models/glm5next/kernels/mla_sparse.cu - see mla_sparse.hpp: nope sparse-MQA attention.
//
// The chunked scoring, softmax and weighted sum come from the shared basalt/kernels/mqa_chunk.cuh stage (the same one
// DeepSeek-V4.1 and Hy3 use) - latent width 512, one block per (chunk of 64 cells, group of 8 heads, token), with the
// Rope template set to 0 so the nope model skips every rotary term and the row type is fp32. The window half of that
// stage is unused here: GLM has no sliding-window ring, all attended cells come from one latent cache gathered by ids.
// The merge of the per-chunk (max, sum-of-weights, weighted-latent) partials is written here directly (it is short) with
// no attention sink and no inverse RoPE, matching the chunk stage's partials so the double-precision reference agrees.
#include "models/glm5next/kernels/mla_sparse.hpp"

#include "basalt/kernels/mqa_chunk.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace basalt::models::glm5next {
namespace {

namespace mqa = basalt::kernels::mqa;

constexpr int kHeadsPerBlock = 8;  // mqa::chunk requires 8 or 16; 64 heads -> 8 head groups

// The fp32 latent row: 512 contiguous floats, key == value, no rope half, no scales.
struct RowF32 {
    using Raw4 = float4;
    using Raw2 = float2;
    using Scales = mqa::NoScales;
    static constexpr int kRowBytes = mqa::kLatent * 4;
    static __device__ __forceinline__ Scales scales(const uint8_t*) { return {}; }
    static __device__ __forceinline__ Raw4 load4(const uint8_t* r, int i) { return reinterpret_cast<const float4*>(r)[i]; }
    static __device__ __forceinline__ Raw2 load2(const uint8_t* r, int i) { return reinterpret_cast<const float2*>(r)[i]; }
    static __device__ __forceinline__ void cvt4(const float4& v, float k[4]) { k[0] = v.x; k[1] = v.y; k[2] = v.z; k[3] = v.w; }
    static __device__ __forceinline__ void cvt4(const Raw4& v, const Scales&, int, float k[4]) { cvt4(v, k); }
    static __device__ __forceinline__ void cvt2(const Raw2& v, float k[2]) { k[0] = v.x; k[1] = v.y; }
};

// One block per (chunk of cells, group of HB heads, token). All chunks gather from the one latent cache by ids.
__global__ void __launch_bounds__(mqa::kThreads) mla_partial_kernel(const float* __restrict__ q_abs,
                                                                    const float* __restrict__ latent,
                                                                    const int32_t* __restrict__ ids,
                                                                    const int32_t* __restrict__ cache_len, float scale,
                                                                    int n_head, int n_sel, int n_chunks, float* o_part,
                                                                    float* ml) {
    const int slot = blockIdx.x;
    const int c = slot;
    const int head0 = blockIdx.y * kHeadsPerBlock;
    const int t = blockIdx.z;
    const int tid = threadIdx.x;

    __shared__ float4 qs[kHeadsPerBlock * mqa::kLatent / 4];
    __shared__ float sc[kHeadsPerBlock][mqa::kChunk];
    __shared__ int rowid[mqa::kChunk];

    int r = -1;
    if (tid < mqa::kChunk) {
        const int j = c * mqa::kChunk + tid;
        if (j < n_sel) {
            const int id = ids[(size_t) t * n_sel + j];
            if (id >= 0 && id < cache_len[t]) r = id;
        }
        rowid[tid] = r;
    }
    const bool any = __syncthreads_or(tid < mqa::kChunk && r >= 0);
    float* mlp0 = ml + (((size_t) t * n_head + head0) * n_chunks + slot) * 2;
    if (!any) {
        if (tid < kHeadsPerBlock) {
            float* m = mlp0 + (size_t) tid * n_chunks * 2;
            m[0] = -INFINITY;
            m[1] = 0.f;
        }
        return;
    }

    const float4* qg = reinterpret_cast<const float4*>(q_abs + ((size_t) t * n_head + head0) * mqa::kLatent);
    for (int i = tid; i < kHeadsPerBlock * mqa::kLatent / 4; i += mqa::kThreads) qs[i] = qg[i];
    __syncthreads();

    float* o0 = o_part + (((size_t) t * n_head + head0) * n_chunks + slot) * mqa::kLatent;
    mqa::chunk<RowF32, RowF32, kHeadsPerBlock, 0>(false, reinterpret_cast<const uint8_t*>(latent), scale, qs, sc, rowid, mlp0,
                                                  (size_t) n_chunks * 2, o0, (size_t) n_chunks * mqa::kLatent);
}

// Merge one head's chunks into its weighted latent (no sink, no rope). Thread tid takes latent values 2*tid, 2*tid+1.
__global__ void __launch_bounds__(mqa::kLatent / 2) mla_merge_kernel(const float* __restrict__ o_part,
                                                                     const float* __restrict__ ml, int n_chunks,
                                                                     int n_head, float* __restrict__ out) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int tid = threadIdx.x;
    const size_t head = (size_t) t * n_head + h;
    const float* mlp = ml + head * n_chunks * 2;
    const float* op = o_part + head * n_chunks * mqa::kLatent;

    float M = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, mlp[2 * c]);
    float denom = 0.f, acc0 = 0.f, acc1 = 0.f;
    for (int c = 0; c < n_chunks; ++c) {
        const float m = mlp[2 * c];
        if (m == -INFINITY) continue;
        const float e = expf(m - M);
        denom = fmaf(mlp[2 * c + 1], e, denom);
        const float2 v = *reinterpret_cast<const float2*>(op + (size_t) c * mqa::kLatent + 2 * tid);
        acc0 = fmaf(v.x, e, acc0);
        acc1 = fmaf(v.y, e, acc1);
    }
    const float inv = denom > 0.f ? 1.0f / denom : 0.f;
    reinterpret_cast<float2*>(out + head * mqa::kLatent)[tid] = make_float2(acc0 * inv, acc1 * inv);
}

}  // namespace

size_t mla_sparse_workspace_floats(int n_tokens, int n_head, int n_sel) {
    const size_t n_chunks = (size_t) (n_sel + mqa::kChunk - 1) / mqa::kChunk;
    return (size_t) n_tokens * n_head * n_chunks * (mqa::kLatent + 2);
}

void mla_sparse(const float* q_abs, const float* latent, const int32_t* ids, const int32_t* cache_len, float* out,
                float scale, int n_tokens, int n_head, int n_sel, float* workspace, cudaStream_t stream) {
    if (n_tokens <= 0 || n_head <= 0 || n_sel <= 0 || n_head % kHeadsPerBlock != 0) return;
    const int n_chunks = (n_sel + mqa::kChunk - 1) / mqa::kChunk;
    float* o_part = workspace;
    float* ml = workspace + (size_t) n_tokens * n_head * n_chunks * mqa::kLatent;
    mla_partial_kernel<<<dim3(n_chunks, n_head / kHeadsPerBlock, n_tokens), mqa::kThreads, 0, stream>>>(
        q_abs, latent, ids, cache_len, scale, n_head, n_sel, n_chunks, o_part, ml);
    mla_merge_kernel<<<dim3(n_head, n_tokens), mqa::kLatent / 2, 0, stream>>>(o_part, ml, n_chunks, n_head, out);
}

}  // namespace basalt::models::glm5next
