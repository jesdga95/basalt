// Shared 64-head sparse MQA with an attention sink and inverse RoPE.
#include "basalt/kernels/sparse_mqa.hpp"
#include "basalt/kernels/mqa_chunk.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::kernels {
namespace {

constexpr int kThreads = mqa::kThreads;
constexpr int kChunk = mqa::kChunk;

struct RowF32 {
    using Raw4 = float4;
    using Raw2 = float2;
    using Scales = mqa::NoScales;
    static constexpr int kRowBytes = kSparseMqaDim * 4;
    static __device__ __forceinline__ Scales scales(const uint8_t*) { return {}; }
    static __device__ __forceinline__ Raw4 load4(const uint8_t* r, int i) { return reinterpret_cast<const float4*>(r)[i]; }
    static __device__ __forceinline__ Raw2 load2(const uint8_t* r, int i) { return reinterpret_cast<const float2*>(r)[i]; }
    static __device__ __forceinline__ void cvt4(const Raw4& v, const Scales&, int, float k[4]) { cvt4(v, k); }
    static __device__ __forceinline__ void cvt4(const Raw4& v, float k[4]) { k[0] = v.x; k[1] = v.y; k[2] = v.z; k[3] = v.w; }
    static __device__ __forceinline__ void cvt2(const Raw2& v, float k[2]) { k[0] = v.x; k[1] = v.y; }
};

struct RowF16 {
    using Raw4 = uint2;
    using Raw2 = uint32_t;
    using Scales = mqa::NoScales;
    static constexpr int kRowBytes = kSparseMqaDim * 2;
    static __device__ __forceinline__ Scales scales(const uint8_t*) { return {}; }
    static __device__ __forceinline__ Raw4 load4(const uint8_t* r, int i) { return reinterpret_cast<const uint2*>(r)[i]; }
    static __device__ __forceinline__ Raw2 load2(const uint8_t* r, int i) { return reinterpret_cast<const uint32_t*>(r)[i]; }
    static __device__ __forceinline__ void cvt4(const Raw4& v, const Scales&, int, float k[4]) { cvt4(v, k); }
    static __device__ __forceinline__ void cvt4(const Raw4& v, float k[4]) {
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
        k[0] = a.x; k[1] = a.y; k[2] = b.x; k[3] = b.y;
    }
    static __device__ __forceinline__ void cvt2(const Raw2& v, float k[2]) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&v));
        k[0] = f.x; k[1] = f.y;
    }
};

struct RowBF16 {
    using Raw4 = uint2;
    using Raw2 = uint32_t;
    using Scales = mqa::NoScales;
    static constexpr int kRowBytes = kSparseMqaDim * 2;
    static __device__ __forceinline__ Scales scales(const uint8_t*) { return {}; }
    static __device__ __forceinline__ Raw4 load4(const uint8_t* r, int i) { return reinterpret_cast<const uint2*>(r)[i]; }
    static __device__ __forceinline__ Raw2 load2(const uint8_t* r, int i) { return reinterpret_cast<const uint32_t*>(r)[i]; }
    static __device__ __forceinline__ void cvt4(const Raw4& v, const Scales&, int, float k[4]) { cvt4(v, k); }
    static __device__ __forceinline__ void cvt4(const Raw4& v, float k[4]) {
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
        k[0] = a.x; k[1] = a.y; k[2] = b.x; k[3] = b.y;
    }
    static __device__ __forceinline__ void cvt2(const Raw2& v, float k[2]) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
        k[0] = f.x; k[1] = f.y;
    }
};

__device__ __forceinline__ void rotate_pair(float& x0, float& x1, float pos, float step, float mscale) {
    float s, c;
    sincosf(pos * step, &s, &c);
    c *= mscale;
    s *= -mscale;
    const float a = x0 * c - x1 * s;
    const float b = x0 * s + x1 * c;
    x0 = a;
    x1 = b;
}

template <class W, class C, int HB>
__global__ void __launch_bounds__(kThreads) mqa_partial_kernel(SparseMqaArgs a, int n_win, int n_chunks_total,
                                                                float* o_part, float* ml) {
    const int slot = blockIdx.x;
    const bool kWindow = slot < n_win;
    const int c = kWindow ? slot : slot - n_win;
    const int g = blockIdx.y;
    const int t = blockIdx.z;
    const int tid = threadIdx.x;

    __shared__ float4 qs[HB * kSparseMqaDim / 4];
    __shared__ float sc[HB][kChunk];
    __shared__ int rowid[kChunk];

    int r = -1;
    if (tid < kChunk) {
        const int j = c * kChunk + tid;
        if (kWindow) {
            if (j < a.win_n[t]) {
                const int cap = a.win_cap;
                r = (int) (((int64_t) a.win_start[t] + j) % cap);
                if (r < 0) r += cap;
            }
        } else if (j < a.n_sel) {
            const int id = a.sel[(size_t) t * a.n_sel + j];
            if (id >= 0 && id < a.comp_limit[t]) r = id;
        }
        rowid[tid] = r;
    }
    const bool any = __syncthreads_or(tid < kChunk && r >= 0);
    const int head0 = g * HB;
    float* mlp0 = ml + (((size_t) t * kSparseMqaHeads + head0) * n_chunks_total + slot) * 2;
    if (!any) {
        if (tid < HB) {
            float* m = mlp0 + (size_t) tid * n_chunks_total * 2;
            m[0] = -INFINITY;
            m[1] = 0.f;
        }
        return;
    }

    const float4* qg = reinterpret_cast<const float4*>(a.q + ((size_t) t * kSparseMqaHeads + head0) * kSparseMqaDim);
    for (int i = tid; i < HB * kSparseMqaDim / 4; i += kThreads) qs[i] = qg[i];
    __syncthreads();

    float* o0 = o_part + (((size_t) t * kSparseMqaHeads + head0) * n_chunks_total + slot) * kSparseMqaDim;
    mqa::chunk<W, C, HB, 0>(kWindow, static_cast<const uint8_t*>(kWindow ? a.win : a.comp), a.scale, qs, sc, rowid, mlp0,
                            (size_t) n_chunks_total * 2, o0, (size_t) n_chunks_total * kSparseMqaDim);
}

__global__ void __launch_bounds__(256) mqa_merge_kernel(const float* o_part, const float* ml, int n_chunks,
                                                         const float* sinks, const int32_t* pos, MqaRopeTable rope,
                                                         bool round_bf16, float* out) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int tid = threadIdx.x;
    const size_t head = (size_t) t * kSparseMqaHeads + h;
    float2 o[1];
    mqa::merge<1, false>(o_part + head * n_chunks * kSparseMqaDim, 0, ml + head * n_chunks * 2, 0, n_chunks, sinks + h,
                        nullptr, o);
    float acc0 = o[0].x, acc1 = o[0].y;
    const int d = 2 * tid;
    if (d >= kSparseMqaDim - kSparseMqaRopeDim)
        rotate_pair(acc0, acc1, (float) pos[t], rope.step[(d - (kSparseMqaDim - kSparseMqaRopeDim)) / 2], rope.mscale);
    if (round_bf16) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(acc0, acc1);
        acc0 = __low2float(v);
        acc1 = __high2float(v);
    }
    reinterpret_cast<float2*>(out + head * kSparseMqaDim)[tid] = make_float2(acc0, acc1);
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class W, class C>
void launch_partial(const SparseMqaArgs& a, int n_win, int n_total, float* o_part, float* ml, cudaStream_t s) {
    int dev = 0, sms = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaGetDevice: ") + cudaGetErrorString(e));
    e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaDeviceGetAttribute: ") + cudaGetErrorString(e));
    if (n_total * (kSparseMqaHeads / 8) * a.n_tokens <= sms)
        mqa_partial_kernel<W, C, 8><<<dim3(n_total, kSparseMqaHeads / 8, a.n_tokens), kThreads, 0, s>>>(a, n_win, n_total,
                                                                                                      o_part, ml);
    else
        mqa_partial_kernel<W, C, 16><<<dim3(n_total, kSparseMqaHeads / 16, a.n_tokens), kThreads, 0, s>>>(a, n_win, n_total,
                                                                                                          o_part, ml);
    check_launch("sparse_mqa partial");
}

template <class W, class C>
void dispatch(const SparseMqaArgs& a, int n_win, int n_chunks, float* o_part, float* ml, cudaStream_t s) {
    launch_partial<W, C>(a, n_win, n_chunks, o_part, ml, s);
}

void launch(const SparseMqaArgs& a, int n_win, int n_chunks, float* o_part, float* ml, cudaStream_t s) {
    if (!a.sel || a.n_sel == 0) {
        switch (a.win_type) {
            case MqaRowType::F32: return dispatch<RowF32, RowF32>(a, n_win, n_chunks, o_part, ml, s);
            case MqaRowType::F16: return dispatch<RowF16, RowF16>(a, n_win, n_chunks, o_part, ml, s);
            case MqaRowType::BF16: return dispatch<RowBF16, RowBF16>(a, n_win, n_chunks, o_part, ml, s);
        }
    } else if (a.win_type == MqaRowType::F32 && a.comp_type == MqaRowType::F32) {
        return dispatch<RowF32, RowF32>(a, n_win, n_chunks, o_part, ml, s);
    } else if (a.win_type == MqaRowType::F16 && a.comp_type == MqaRowType::F16) {
        return dispatch<RowF16, RowF16>(a, n_win, n_chunks, o_part, ml, s);
    } else if (a.win_type == MqaRowType::F16 && a.comp_type == MqaRowType::F32) {
        return dispatch<RowF16, RowF32>(a, n_win, n_chunks, o_part, ml, s);
    } else if (a.win_type == MqaRowType::BF16 && a.comp_type == MqaRowType::BF16) {
        return dispatch<RowBF16, RowBF16>(a, n_win, n_chunks, o_part, ml, s);
    }
    throw std::invalid_argument("sparse_mqa: unsupported row type pair");
}

}  // namespace

size_t sparse_mqa_workspace_floats(int n_tokens, int n_sel) {
    if (n_tokens < 1 || n_tokens > kSparseMqaMaxTokens || n_sel < 0 || n_sel > kSparseMqaTopK)
        throw std::invalid_argument("sparse_mqa_workspace_floats: bounds");
    const size_t n_chunks = (size_t) ((kSparseMqaWindow + kChunk - 1) / kChunk) + (size_t) ((n_sel + kChunk - 1) / kChunk);
    return (size_t) n_tokens * kSparseMqaHeads * n_chunks * (kSparseMqaDim + 2);
}

void sparse_mqa(const SparseMqaArgs& a, float* workspace, void* stream) {
    if (a.n_tokens < 1 || a.n_tokens > kSparseMqaMaxTokens) throw std::invalid_argument("sparse_mqa: n_tokens");
    if (a.n_sel < 0 || a.n_sel > kSparseMqaTopK) throw std::invalid_argument("sparse_mqa: n_sel");
    if (a.win_cap < 1) throw std::invalid_argument("sparse_mqa: win_cap");
    if (!a.q || !a.win || !a.win_start || !a.win_n || !a.sinks || !a.pos || !a.out || !workspace)
        throw std::invalid_argument("sparse_mqa: null required pointer");
    if (a.n_sel > 0 && (!a.sel || !a.comp || !a.comp_limit)) throw std::invalid_argument("sparse_mqa: null selected-row input");
    if (!(a.scale > 0.f) || !std::isfinite(a.scale) || !std::isfinite(a.rope.mscale))
        throw std::invalid_argument("sparse_mqa: invalid scale");
    const int n_win = (kSparseMqaWindow + kChunk - 1) / kChunk;
    const bool sparse = a.sel != nullptr && a.n_sel > 0;
    const int n_chunks = n_win + (sparse ? (a.n_sel + kChunk - 1) / kChunk : 0);
    float* o_part = workspace;
    float* ml = workspace + (size_t) a.n_tokens * kSparseMqaHeads * n_chunks * kSparseMqaDim;
    launch(a, n_win, n_chunks, o_part, ml, (cudaStream_t) stream);
    mqa_merge_kernel<<<dim3(kSparseMqaHeads, a.n_tokens), 256, 0, (cudaStream_t) stream>>>(o_part, ml, n_chunks, a.sinks,
                                                                                          a.pos, a.rope,
                                                                                          a.round_output_bf16, a.out);
    check_launch("sparse_mqa merge");
}

}  // namespace basalt::kernels
