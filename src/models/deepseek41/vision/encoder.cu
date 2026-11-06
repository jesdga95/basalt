// DeepSeek-V4.1 vision encoder on one CUDA device. Weights stay bf16 as stored; the residual stream, attention scores
// and aligner output are fp32; the inputs of the products are bf16. Attention runs one head at a time as two products
// around a row softmax. The graph follows llama.cpp's mtmd deepseek4v encoder (MIT).
#include "basalt/models/deepseek41/vision.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace basalt::models::deepseek41 {

namespace {

constexpr int E = 1024, F = 2816, H = 16, D = 64, QKV = 3 * E, MERGED = 9 * E;

__global__ void k_rmsnorm(const float* x, const float* w, __nv_bfloat16* ob, float* of, float eps) {
    __shared__ float red[8];
    const float* r = x + (size_t) blockIdx.x * E;
    float s = 0.f;
    for (int i = threadIdx.x; i < E; i += 256) s += r[i] * r[i];
    for (int m = 16; m; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < 8; ++i) t += red[i];
    const float inv = rsqrtf(t / E + eps);
    for (int i = threadIdx.x; i < E; i += 256) {
        const float v = r[i] * inv * w[i];
        if (ob) ob[(size_t) blockIdx.x * E + i] = __float2bfloat16(v);
        if (of) of[(size_t) blockIdx.x * E + i] = v;
    }
}

// g [n][3072] + bias -> q, k (rotated), v; all [n][1024] fp32. Pair p of a head turns (x[p], x[p + 32]); pairs 0..15
// use the patch row, 16..31 the patch column, each with frequencies 10000^(-e/16).
__global__ void k_qkv(const float* g, const float* bias, float* q, float* k, float* v, int n, int gx) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t) n * E) return;
    const int t = (int) (idx / E), c = (int) (idx % E);
    const int h = c / D, p = c % D;
    const float* row = g + (size_t) t * QKV;
    v[idx] = row[2 * E + c] + bias[2 * E + c];
    if (p >= 32) return;
    const int e = p & 15;
    const float pos = (float) (p < 16 ? t / gx : t % gx);
    const float theta = pos * powf(10000.f, -(float) e / 16.f);
    float sn, cs;
    sincosf(theta, &sn, &cs);
    for (int which = 0; which < 2; ++which) {
        const int base = which * E + h * D;
        const float x0 = row[base + p] + bias[base + p], x1 = row[base + p + 32] + bias[base + p + 32];
        float* o = (which ? k : q) + (size_t) t * E + h * D;
        o[p] = x0 * cs - x1 * sn;
        o[p + 32] = x0 * sn + x1 * cs;
    }
}

__global__ void k_softmax(float* s, int n, float scale) {
    __shared__ float red[8];
    float* r = s + (size_t) blockIdx.x * n;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += 256) m = fmaxf(m, r[i] * scale);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = red[0];
    for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) {
        const float e = __expf(r[i] * scale - m);
        r[i] = e;
        sum += e;
    }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    float tot = 0.f;
    for (int i = 0; i < 8; ++i) tot += red[i];
    const float inv = 1.f / tot;
    for (int i = threadIdx.x; i < n; i += 256) r[i] *= inv;
}

__global__ void k_to_bf16(const float* a, __nv_bfloat16* b, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __float2bfloat16(a[i]);
}

// x[t][c] += g[t][c] (+ bias[c])
__global__ void k_residual(float* x, const float* g, const float* bias, size_t n, int cols) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += g[i] + (bias ? bias[i % cols] : 0.f);
}

__global__ void k_bias_add(float* x, const float* bias, size_t n, int cols) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += bias[i % cols];
}

__global__ void k_swiglu(const float* gate, const float* up, __nv_bfloat16* y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float a = gate[i];
        y[i] = __float2bfloat16(a / (1.f + __expf(-a)) * up[i]);
    }
}

// merged row (oy, ox), column c * 9 + ky * 3 + kx = the ViT output of patch (oy*3 + ky, ox*3 + kx), zero past the grid
__global__ void k_merge(const float* y, __nv_bfloat16* out, int gx, int gy, int lw, int lh) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) lw * lh * MERGED) return;
    const int col = (int) (i % MERGED);
    const size_t row = i / MERGED;
    const int oy = (int) (row / lw), ox = (int) (row % lw);
    const int c = col / 9, ky = (col % 9) / 3, kx = col % 3;
    const int py = oy * 3 + ky, px = ox * 3 + kx;
    out[i] = __float2bfloat16(py < gy && px < gx ? y[((size_t) py * gx + px) * E + c] : 0.f);
}

// h = bf16(gelu_erf(g + bias))
__global__ void k_gelu(const float* g, const float* bias, __nv_bfloat16* y, size_t n, int cols) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float a = g[i] + bias[i % cols];
        y[i] = __float2bfloat16(0.5f * a * (1.f + erff(a * 0.70710678118654752f)));
    }
}

// block rows: START, then lh rows of (lw merged rows, NEWLINE), END
__global__ void k_assemble(const float* aligned, const float* start, const float* end, const float* nl, float* out,
                           int lw, int lh, int width) {
    const int r = blockIdx.x;
    const float* src;
    if (r == 0) src = start;
    else if (r == lh * (lw + 1) + 1) src = end;
    else {
        const int j = r - 1, row = j / (lw + 1), col = j % (lw + 1);
        src = col == lw ? nl : aligned + ((size_t) row * lw + col) * width;
    }
    for (int i = threadIdx.x; i < width; i += blockDim.x) out[(size_t) r * width + i] = src[i];
}

}  // namespace

struct VisionEncoder::Impl {
    int layers = 0, proj = 0, cap = 0, ng_cap = 0, tok_cap = 0;
    float eps = 1e-6f;
    cublasHandle_t blas = nullptr;
    struct Layer {
        float *ln1, *ln2, *bqkv, *bo;
        __nv_bfloat16 *wqkv, *wo, *wgate, *wup, *wdown;
    };
    std::vector<Layer> L;
    std::vector<void*> owned;
    float *wpatch = nullptr, *bpatch = nullptr, *postw = nullptr, *b1 = nullptr, *b2 = nullptr, *start = nullptr, *endv = nullptr, *nl = nullptr;
    __nv_bfloat16 *w1 = nullptr, *w2 = nullptr;
    std::vector<void*> work;
    float *d_rows = nullptr, *x = nullptr, *g = nullptr, *g2 = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *ao = nullptr, *S = nullptr,
          *y = nullptr, *aout = nullptr, *dout = nullptr;
    __nv_bfloat16 *act = nullptr, *mrg = nullptr, *h1 = nullptr;
    std::string err;

    ~Impl() {
        for (void* p : owned) cudaFree(p);
        for (void* p : work) cudaFree(p);
        if (blas) cublasDestroy(blas);
    }

    template <class T>
    T* upload(const basalt::vision::TensorRef* t) {
        void* p = nullptr;
        if (cudaMalloc(&p, t->bytes) != cudaSuccess) return nullptr;
        owned.push_back(p);
        if (cudaMemcpy(p, t->data, t->bytes, cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
        return (T*) p;
    }
    template <class T>
    T* alloc(T*& slot, size_t count) {
        void* p = nullptr;
        if (cudaMalloc(&p, count * sizeof(T)) != cudaSuccess) return nullptr;
        work.push_back(p);
        return slot = (T*) p;
    }

    // Y[M][N] = X[M][K] . W[N][K]^T with arbitrary row strides
    bool gemm(cudaDataType ta, const void* X, int ldx, const void* W, int ldw, float* Y, int ldy, int M, int N, int K) {
        const float one = 1.f, zero = 0.f;
        return cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, ta, ldw, X, ta, ldx, &zero, Y, CUDA_R_32F, ldy,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
    }

    bool load(const basalt::vision::Mmproj& m, std::string& e) {
        using basalt::vision::TensorRef;
        const auto& hp = m.hp();
        if (hp.embd != E || hp.heads != H || hp.ffn != F || hp.patch != kVisionPatch || hp.merge != 3 || hp.layers < 1) {
            e = "unexpected projector hyper-parameters";
            return false;
        }
        layers = hp.layers;
        proj = hp.proj;
        eps = hp.eps;
        auto need = [&](const std::string& n, uint32_t type, int64_t elems) -> const TensorRef* {
            return m.need(n, type, elems, e);
        };
        auto f32 = [&](const std::string& n, int64_t elems) -> float* {
            const TensorRef* t = need(n, 0, elems);
            return t ? upload<float>(t) : nullptr;
        };
        auto bf = [&](const std::string& n, int64_t elems) -> __nv_bfloat16* {
            const TensorRef* t = need(n, 30, elems);
            return t ? upload<__nv_bfloat16>(t) : nullptr;
        };
        wpatch = f32("v.patch_embd.weight", (int64_t) E * kVisionPatchValues);
        bpatch = f32("v.patch_embd.bias", E);
        postw = f32("v.post_ln.weight", E);
        w1 = bf("mm.1.weight", (int64_t) MERGED * proj);
        b1 = f32("mm.1.bias", proj);
        w2 = bf("mm.2.weight", (int64_t) proj * proj);
        b2 = f32("mm.2.bias", proj);
        start = f32("v.token_embd.img_start", proj);
        endv = f32("v.token_embd.img_end", proj);
        nl = f32("v.image_newline", proj);
        if (!wpatch || !bpatch || !postw || !w1 || !b1 || !w2 || !b2 || !start || !endv || !nl) return false;
        L.assign(layers, {});
        for (int i = 0; i < layers; ++i) {
            const std::string p = "v.blk." + std::to_string(i) + ".";
            Layer& y = L[i];
            y.ln1 = f32(p + "ln1.weight", E);
            y.ln2 = f32(p + "ln2.weight", E);
            y.bqkv = f32(p + "attn_qkv.bias", QKV);
            y.bo = f32(p + "attn_out.bias", E);
            y.wqkv = bf(p + "attn_qkv.weight", (int64_t) QKV * E);
            y.wo = bf(p + "attn_out.weight", (int64_t) E * E);
            y.wgate = bf(p + "ffn_gate.weight", (int64_t) F * E);
            y.wup = bf(p + "ffn_up.weight", (int64_t) F * E);
            y.wdown = bf(p + "ffn_down.weight", (int64_t) E * F);
            if (!y.ln1 || !y.ln2 || !y.bqkv || !y.bo || !y.wqkv || !y.wo || !y.wgate || !y.wup || !y.wdown) return false;
        }
        return true;
    }

    bool ensure(int n, int ng, int ntok) {
        if (n <= cap && ng <= ng_cap && ntok <= tok_cap) return true;
        for (void* p : work) cudaFree(p);
        work.clear();
        cap = std::max(n, cap);
        ng_cap = std::max(ng, ng_cap);
        tok_cap = std::max(ntok, tok_cap);
        const size_t N = cap;
        return alloc(d_rows, N * kVisionPatchValues) && alloc(x, N * E) && alloc(g, N * QKV) && alloc(g2, N * F) && alloc(q, N * E) &&
               alloc(k, N * E) && alloc(v, N * E) && alloc(ao, N * E) && alloc(S, N * N) && alloc(y, N * E) &&
               alloc(act, N * F) && alloc(aout, (size_t) ng_cap * proj) && alloc(mrg, (size_t) ng_cap * MERGED) &&
               alloc(h1, (size_t) ng_cap * proj) && alloc(dout, (size_t) tok_cap * proj);
    }
};

VisionEncoder::VisionEncoder() : impl_(new Impl) {}
VisionEncoder::~VisionEncoder() = default;
int VisionEncoder::width() const { return impl_->proj; }

bool VisionEncoder::init(const basalt::vision::Mmproj& mp, int max_patches, std::string& err) {
    (void) max_patches;
    if (cublasCreate(&impl_->blas) != CUBLAS_STATUS_SUCCESS) {
        err = "cannot create the product handle";
        return false;
    }
    if (!impl_->load(mp, err)) {
        if (err.empty()) err = "cannot upload the projector weights";
        return false;
    }
    return true;
}

static int blocks(size_t n) { return (int) ((n + 255) / 256); }

bool VisionEncoder::encode(const std::vector<float>& rows, const VisionPlan& pl, std::vector<float>& out, std::string& err) {
    Impl& s = *impl_;
    const int n = pl.n_patches(), ng = pl.lw * pl.lh, ntok = pl.n_tokens();
    if ((size_t) n * kVisionPatchValues != rows.size()) {
        err = "patch rows do not match the plan";
        return false;
    }
    if (!s.ensure(n, ng, ntok)) {
        err = "cannot allocate the encoder's work buffers";
        return false;
    }
    cudaMemcpy(s.d_rows, rows.data(), rows.size() * sizeof(float), cudaMemcpyHostToDevice);
    // patch embedding (fp32 product) + bias
    s.gemm(CUDA_R_32F, s.d_rows, kVisionPatchValues, s.wpatch, kVisionPatchValues, s.x, E, n, E, kVisionPatchValues);
    k_bias_add<<<blocks((size_t) n * E), 256>>>(s.x, s.bpatch, (size_t) n * E, E);
    bool ok = true;
    for (int l = 0; l < s.layers && ok; ++l) {
        const Impl::Layer& Ly = s.L[l];
        k_rmsnorm<<<n, 256>>>(s.x, Ly.ln1, s.act, nullptr, s.eps);
        ok &= s.gemm(CUDA_R_16BF, s.act, E, Ly.wqkv, E, s.g, QKV, n, QKV, E);
        k_qkv<<<blocks((size_t) n * E), 256>>>(s.g, Ly.bqkv, s.q, s.k, s.v, n, pl.gx);
        const float one = 1.f, zero = 0.f;
        for (int h = 0; h < H && ok; ++h) {
            // scores S[i][j] = q_i . k_j, row softmax, then out_i = sum_j S[i][j] v_j (column-major views of row-major data)
            ok &= cublasGemmEx(s.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, D, &one, s.k + h * D, CUDA_R_32F, E, s.q + h * D, CUDA_R_32F, E,
                               &zero, s.S, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
            k_softmax<<<n, 256>>>(s.S, n, 1.0f / 8.0f);
            ok &= cublasGemmEx(s.blas, CUBLAS_OP_N, CUBLAS_OP_N, D, n, n, &one, s.v + h * D, CUDA_R_32F, E, s.S, CUDA_R_32F, n, &zero,
                               s.ao + h * D, CUDA_R_32F, E, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
        }
        k_to_bf16<<<blocks((size_t) n * E), 256>>>(s.ao, s.act, (size_t) n * E);
        ok &= s.gemm(CUDA_R_16BF, s.act, E, Ly.wo, E, s.g, E, n, E, E);
        k_residual<<<blocks((size_t) n * E), 256>>>(s.x, s.g, Ly.bo, (size_t) n * E, E);
        k_rmsnorm<<<n, 256>>>(s.x, Ly.ln2, s.act, nullptr, s.eps);
        ok &= s.gemm(CUDA_R_16BF, s.act, E, Ly.wgate, E, s.g, F, n, F, E);
        ok &= s.gemm(CUDA_R_16BF, s.act, E, Ly.wup, E, s.g2, F, n, F, E);
        k_swiglu<<<blocks((size_t) n * F), 256>>>(s.g, s.g2, s.act, (size_t) n * F);
        ok &= s.gemm(CUDA_R_16BF, s.act, F, Ly.wdown, F, s.g, E, n, E, F);
        k_residual<<<blocks((size_t) n * E), 256>>>(s.x, s.g, nullptr, (size_t) n * E, E);
    }
    if (!ok) {
        err = "a product failed";
        return false;
    }
    k_rmsnorm<<<n, 256>>>(s.x, s.postw, nullptr, s.y, s.eps);
    k_merge<<<blocks((size_t) ng * MERGED), 256>>>(s.y, s.mrg, pl.gx, pl.gy, pl.lw, pl.lh);
    ok &= s.gemm(CUDA_R_16BF, s.mrg, MERGED, s.w1, MERGED, s.aout, s.proj, ng, s.proj, MERGED);
    k_gelu<<<blocks((size_t) ng * s.proj), 256>>>(s.aout, s.b1, s.h1, (size_t) ng * s.proj, s.proj);
    ok &= s.gemm(CUDA_R_16BF, s.h1, s.proj, s.w2, s.proj, s.aout, s.proj, ng, s.proj, s.proj);
    k_bias_add<<<blocks((size_t) ng * s.proj), 256>>>(s.aout, s.b2, (size_t) ng * s.proj, s.proj);
    k_assemble<<<ntok, 256>>>(s.aout, s.start, s.endv, s.nl, s.dout, pl.lw, pl.lh, s.proj);
    out.resize((size_t) ntok * s.proj);
    const cudaError_t ce = cudaMemcpy(out.data(), s.dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (!ok || ce != cudaSuccess) {
        err = ok ? std::string("encode failed: ") + cudaGetErrorString(ce) : "a product failed";
        return false;
    }
    return true;
}

}  // namespace basalt::models::deepseek41
