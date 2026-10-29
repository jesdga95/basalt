// src/prefill/gemm_fp8.cu - see include/basalt/prefill/gemm_fp8.hpp.
#include "basalt/prefill/gemm_fp8.hpp"

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <tuple>

namespace basalt::prefill::fp8 {
namespace {

constexpr int kThreads = 256;

__host__ __device__ __forceinline__ size_t round_up(size_t v, size_t m) { return (v + m - 1) / m * m; }

// The exponent e of the smallest power of two with amax / 2^e <= 448 (the largest finite E4M3 value), so the largest
// magnitude lands in (224, 448]; 0 for an all-zero or non-finite amax.
__device__ __forceinline__ int pow2_exp(float amax) {
    if (!(amax > 0.0f) || !isfinite(amax)) return 0;
    int e = 0;
    const float m = frexpf(amax * (1.0f / 448.0f), &e);   // amax / 448 = m * 2^e, m in [0.5, 1)
    return m == 0.5f ? e - 1 : e;
}

__device__ __forceinline__ float absmax8(const uint4& v) {
    const __half2* h = reinterpret_cast<const __half2*>(&v);
    float a = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        a = fmaxf(a, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
    return a;
}

// 8 FP16 values times `inv` (a power of two: exact) to 8 E4M3 bytes, in order.
__device__ __forceinline__ uint2 cvt8(const uint4& v, float inv) {
    const __half2* h = reinterpret_cast<const __half2*>(&v);
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __half22float2(h[i]);
        f.x *= inv;
        f.y *= inv;
        w[i] = (uint32_t) __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
    }
    return make_uint2(w[0] | (w[1] << 16), w[2] | (w[3] << 16));
}

// cuBLASLt's block-scale layout: 128-row x 4-block tiles of 512 bytes, the tiles of a 128-row band consecutive along
// K; inside a tile, row r and block b at (r % 32) * 16 + (r % 128) / 32 * 4 + b % 4.
__device__ __forceinline__ int64_t sf_offset(int64_t r, int64_t b, int64_t blocks_pad) {
    return ((r / 128) * (blocks_pad / 4) + b / 4) * 512 + (r % 32) * 16 + (r % 128) / 32 * 4 + (b % 4);
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float part[kThreads / 32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < kThreads / 32; ++i) m = fmaxf(m, part[i]);
    return m;
}

// One block per row; a thread takes 8 values at a time.
template <Scale S>
__global__ void __launch_bounds__(kThreads) e4m3_rows_kernel(const uint16_t* __restrict__ in, int64_t cols, int64_t ld,
                                                             uint8_t* __restrict__ q, void* __restrict__ scales,
                                                             int64_t row0, int64_t total_rows,
                                                             const float* __restrict__ tensor_scale) {
    const int64_t r = blockIdx.x;
    const uint4* src = reinterpret_cast<const uint4*>(in + r * ld);
    uint2* dst = reinterpret_cast<uint2*>(q + r * cols);
    const int nc = (int) (cols / 8);
    if constexpr (S == Scale::Row) {
        float a = 0.0f;
        for (int c = threadIdx.x; c < nc; c += kThreads) a = fmaxf(a, absmax8(src[c]));
        const int e = pow2_exp(block_max(a));
        if (threadIdx.x == 0) static_cast<float*>(scales)[row0 + r] = ldexpf(1.0f, e);
        const float inv = ldexpf(1.0f, -e);
        for (int c = threadIdx.x; c < nc; c += kThreads) dst[c] = cvt8(src[c], inv);
    } else if constexpr (S == Scale::Tensor) {
        const float inv = 1.0f / *tensor_scale;
        for (int c = threadIdx.x; c < nc; c += kThreads) dst[c] = cvt8(src[c], inv);
    } else {
        // four consecutive lanes hold one 32-value block (cols is a multiple of 32, so nc of 4)
        const int64_t blocks_pad = (int64_t) round_up((size_t) (cols / 32), 4);
        uint8_t* sf = static_cast<uint8_t*>(scales);
        for (int base = 0; base < nc; base += kThreads) {
            const int c = base + (int) threadIdx.x;
            const bool on = c < nc;
            const uint4 v = on ? src[c] : make_uint4(0, 0, 0, 0);
            float a = absmax8(v);
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
            const int e = pow2_exp(a);
            if (on) {
                dst[c] = cvt8(v, ldexpf(1.0f, -e));
                if ((c & 3) == 0) sf[sf_offset(row0 + r, c / 4, blocks_pad)] = (uint8_t) (e + 127);
            }
        }
        (void) total_rows;
    }
}

__global__ void amax_kernel(const uint16_t* __restrict__ in, int64_t n, unsigned* __restrict__ out) {
    float a = 0.0f;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    const int64_t n8 = n / 8;
    const uint4* v = reinterpret_cast<const uint4*>(in);
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += stride) a = fmaxf(a, absmax8(v[i]));
    for (int64_t i = n8 * 8 + (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        a = fmaxf(a, fabsf(__half2float(__ushort_as_half(in[i]))));
    a = block_max(a);
    if (threadIdx.x == 0) atomicMax(out, __float_as_uint(a));   // non-negative floats order as their bits
}

__global__ void amax_to_scale_kernel(float* s) {
    *s = ldexpf(1.0f, pow2_exp(__uint_as_float(*reinterpret_cast<const unsigned*>(s))));
}

}  // namespace

const char* scale_name(Scale s) {
    switch (s) {
    case Scale::Row: return "per-row";
    case Scale::Block32: return "per-32 block";
    case Scale::Tensor: return "per-tensor";
    }
    return "?";
}

size_t scale_bytes(Scale s, int64_t rows, int64_t cols) {
    switch (s) {
    case Scale::Row: return (size_t) rows * 4;
    case Scale::Block32: return round_up((size_t) rows, 128) * round_up((size_t) (cols / 32), 4);
    case Scale::Tensor: return 4;
    }
    return 0;
}

void to_e4m3(const uint16_t* in, int64_t rows, int64_t cols, int64_t ld, uint8_t* q, void* scales, int64_t row0,
             int64_t total_rows, Scale s, const float* tensor_scale, void* stream) {
    if (rows <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    const dim3 grid((unsigned) rows);
    switch (s) {
    case Scale::Row:
        e4m3_rows_kernel<Scale::Row><<<grid, kThreads, 0, st>>>(in, cols, ld, q, scales, row0, total_rows, nullptr);
        break;
    case Scale::Block32:
        e4m3_rows_kernel<Scale::Block32><<<grid, kThreads, 0, st>>>(in, cols, ld, q, scales, row0, total_rows, nullptr);
        break;
    case Scale::Tensor:
        e4m3_rows_kernel<Scale::Tensor><<<grid, kThreads, 0, st>>>(in, cols, ld, q, scales, row0, total_rows,
                                                                   tensor_scale);
        break;
    }
}

void tensor_scale(const uint16_t* in, int64_t n, float* out, void* stream) {
    cudaStream_t st = (cudaStream_t) stream;
    cudaMemsetAsync(out, 0, 4, st);
    const int64_t blocks = std::min<int64_t>(1024, (n / 8 + kThreads - 1) / kThreads + 1);
    amax_kernel<<<(unsigned) blocks, kThreads, 0, st>>>(in, n, reinterpret_cast<unsigned*>(out));
    amax_to_scale_kernel<<<1, 1, 0, st>>>(out);
}

// ---- cuBLASLt
//
// One algorithm per weight shape (scale mode, N, K, ldy, workspace), chosen once from the heuristic at a long chunk
// (kRefRows rows) among those that do not split K, and then run for every chunk size: each output element is the same
// K loop of the same kernel whatever the chunk's size or the token's place in it.  The plans are bounded by the
// model's weight shapes.

namespace {
constexpr int64_t kRefRows = 32768;

struct Layouts {
    cublasLtMatrixLayout_t b = nullptr, c = nullptr;
    bool ok = false;
    Layouts(int64_t T, int64_t N, int64_t K, int64_t ldy) {
        ok = cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, (uint64_t) K, (uint64_t) T, K) == CUBLAS_STATUS_SUCCESS &&
             cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, (uint64_t) N, (uint64_t) T, ldy) == CUBLAS_STATUS_SUCCESS;
    }
    ~Layouts() {
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (c) cublasLtMatrixLayoutDestroy(c);
    }
    Layouts(const Layouts&) = delete;
    Layouts& operator=(const Layouts&) = delete;
};
}  // namespace

struct Lt::State {
    cublasLtHandle_t h = nullptr;
    struct Plan {
        cublasLtMatmulDesc_t op = nullptr;
        cublasLtMatrixLayout_t a = nullptr;
        cublasLtMatmulAlgo_t algo{};
        bool ok = false;
    };
    using Key = std::tuple<int, int64_t, int64_t, int64_t, size_t, bool>;
    std::map<Key, Plan> plans;

    ~State() {
        for (auto& kv : plans) {
            if (kv.second.op) cublasLtMatmulDescDestroy(kv.second.op);
            if (kv.second.a) cublasLtMatrixLayoutDestroy(kv.second.a);
        }
        if (h) cublasLtDestroy(h);
    }

    // The scale pointers go in before the heuristic is asked (block-scaled modes want them); a run sets its own.
    Plan& plan(Scale s, int64_t N, int64_t K, int64_t ldy, size_t ws, bool fast, const void* ws_ptr, const void* xs_ptr) {
        const Key key{(int) s, N, K, ldy, ws, fast};
        auto it = plans.find(key);
        if (it != plans.end()) return it->second;
        Plan& p = plans[key];
        bool ok = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS;
        // column-major view: Y^T [N, T] = op(W) [N, K] . X^T [K, T], with W stored as K x N (rows of K values)
        const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        const int32_t mode = s == Scale::Row       ? CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
                             : s == Scale::Block32 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
                                                   : CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        const int8_t fa = fast ? 1 : 0;
        auto set = [&](cublasLtMatmulDescAttributes_t a, const void* v, size_t n) {
            ok = ok && cublasLtMatmulDescSetAttribute(p.op, a, v, n) == CUBLAS_STATUS_SUCCESS;
        };
        set(CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
        set(CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
        set(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof mode);
        set(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof mode);
        set(CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fa, sizeof fa);
        set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ws_ptr, sizeof ws_ptr);
        set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &xs_ptr, sizeof xs_ptr);
        ok = ok && cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8F_E4M3, (uint64_t) K, (uint64_t) N, K) == CUBLAS_STATUS_SUCCESS;
        const Layouts ref(kRefRows, N, K, ldy);
        cublasLtMatmulPreference_t pref = nullptr;
        if (!ok || !ref.ok || cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return p;
        const uint64_t wsb = ws;
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb);
        cublasLtMatmulHeuristicResult_t res[16] = {};
        int n = 0;
        if (cublasLtMatmulAlgoGetHeuristic(h, p.op, p.a, ref.b, ref.c, ref.c, pref, 16, res, &n) != CUBLAS_STATUS_SUCCESS)
            n = 0;
        cublasLtMatmulPreferenceDestroy(pref);
        // the first offered algorithm that keeps K in one piece; else the first, told not to split
        int pick = -1;
        for (int i = 0; i < n && pick < 0; ++i) {
            int32_t splits = 1;
            size_t got = 0;
            if (res[i].state == CUBLAS_STATUS_SUCCESS &&
                cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splits,
                                                     sizeof splits, &got) == CUBLAS_STATUS_SUCCESS &&
                splits <= 1)
                pick = i;
        }
        if (pick < 0 && n > 0 && res[0].state == CUBLAS_STATUS_SUCCESS) {
            const int32_t one = 1;
            const uint32_t none = CUBLASLT_REDUCTION_SCHEME_NONE;
            cublasLtMatmulAlgoConfigSetAttribute(&res[0].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, sizeof one);
            cublasLtMatmulAlgoConfigSetAttribute(&res[0].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &none, sizeof none);
            cublasLtMatmulHeuristicResult_t chk{};
            if (cublasLtMatmulAlgoCheck(h, p.op, p.a, ref.b, ref.c, ref.c, &res[0].algo, &chk) == CUBLAS_STATUS_SUCCESS)
                pick = 0;
        }
        if (pick < 0) return p;
        p.algo = res[pick].algo;
        p.ok = true;
        return p;
    }

    // The plan's algorithm runs this shape within the workspace.
    bool fits(Plan& p, int64_t T, int64_t N, int64_t K, int64_t ldy, size_t ws) {
        const Layouts l(T, N, K, ldy);
        cublasLtMatmulHeuristicResult_t chk{};
        return p.ok && l.ok && cublasLtMatmulAlgoCheck(h, p.op, p.a, l.b, l.c, l.c, &p.algo, &chk) == CUBLAS_STATUS_SUCCESS &&
               chk.workspaceSize <= ws;
    }
};

Lt::~Lt() { delete state_; }

bool Lt::init(std::string& err) {
    if (state_) return true;
    auto* s = new State;
    if (const cublasStatus_t e = cublasLtCreate(&s->h); e != CUBLAS_STATUS_SUCCESS) {
        delete s;
        err = "cublasLtCreate: cuBLAS status " + std::to_string((int) e);
        return false;
    }
    state_ = s;
    return true;
}

bool Lt::fits(Scale s, int64_t T, int64_t N, int64_t K, int64_t ldy, size_t ws_bytes, const void* w_scales,
              const void* x_scales, bool fast_accum) {
    if (!state_ || T <= 0) return false;
    State::Plan& p = state_->plan(s, N, K, ldy, ws_bytes, fast_accum, w_scales, x_scales);
    return state_->fits(p, T, N, K, ldy, ws_bytes);
}

std::string Lt::describe(Scale s, int64_t N, int64_t K, int64_t ldy, size_t ws_bytes, const void* w_scales,
                         const void* x_scales) {
    if (!state_) return {};
    const State::Plan& p = state_->plan(s, N, K, ldy, ws_bytes, false, w_scales, x_scales);
    if (!p.ok) return {};
    int32_t id = 0, splits = 0;
    uint32_t tile = 0, red = 0, swz = 0, custom = 0, stages = 0;
    size_t got = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &got);
    cublasLtMatmulAlgoConfigGetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &got);
    cublasLtMatmulAlgoConfigGetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splits, sizeof splits, &got);
    cublasLtMatmulAlgoConfigGetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof red, &got);
    cublasLtMatmulAlgoConfigGetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swz, sizeof swz, &got);
    cublasLtMatmulAlgoConfigGetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof custom, &got);
    cublasLtMatmulAlgoConfigGetAttribute(&p.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages, &got);
    return "id " + std::to_string(id) + " tile " + std::to_string(tile) + " splits " + std::to_string(splits) +
           " reduction " + std::to_string(red) + " swizzle " + std::to_string(swz) + " custom " +
           std::to_string(custom) + " stages " + std::to_string(stages);
}

bool Lt::run(Scale s, const uint8_t* W, const void* w_scales, const uint8_t* X, const void* x_scales, float* Y,
             int64_t T, int64_t N, int64_t K, int64_t ldy, void* workspace, size_t ws_bytes, void* stream,
             bool fast_accum) {
    if (!state_) return false;
    State::Plan& p = state_->plan(s, N, K, ldy, ws_bytes, fast_accum, w_scales, x_scales);
    const Layouts l(T, N, K, ldy);
    if (!p.ok || !l.ok ||
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_scales, sizeof w_scales) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &x_scales, sizeof x_scales) !=
            CUBLAS_STATUS_SUCCESS)
        return false;
    const float alpha = 1.0f, beta = 0.0f;
    return cublasLtMatmul(state_->h, p.op, &alpha, W, p.a, X, l.b, &beta, Y, l.c, Y, l.c, &p.algo, workspace, ws_bytes,
                          (cudaStream_t) stream) == CUBLAS_STATUS_SUCCESS;
}

}  // namespace basalt::prefill::fp8
