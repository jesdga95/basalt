// src/vision/encoder_cuda.cu - see basalt/vision/encoder_cuda.hpp.
#include "basalt/vision/encoder_cuda.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "vision_kernels.cuh"

namespace basalt::vision {

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) { fail(std::string(#x) + ": " + cudaGetErrorString(e_)); return false; } \
    } while (0)
#define CB(x)                                                                                    \
    do {                                                                                         \
        cublasStatus_t s_ = (x);                                                                 \
        if (s_ != CUBLAS_STATUS_SUCCESS) { fail(std::string(#x) + " failed: " + std::to_string((int) s_)); return false; } \
    } while (0)

namespace {
__global__ void probe_kernel(int*) {}
}  // namespace

bool probe_device(std::string& err) {
    probe_kernel<<<1, 1>>>(nullptr);
    const cudaError_t launched = cudaGetLastError();     // a cubin this card lacks is reported by the launch
    const cudaError_t e = launched == cudaSuccess ? cudaDeviceSynchronize() : launched;
    if (e != cudaSuccess) {
        err = cudaGetErrorString(e);
        (void) cudaGetLastError();                       // leave the error state clean for whatever tries next
        return false;
    }
    return true;
}

namespace {
constexpr int E = 1152, F = 4304, H = 16, D = 72, DP = 80;

constexpr size_t kAlign = 256;
// the product library's scratch: a fixed size, so the algorithm choice never depends on where the arena sits
constexpr size_t kBlasWorkspace = 32u << 20;

size_t up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

}  // namespace

struct CudaEncoder::Impl {
    // byte offsets of the tensors inside the weight image (host copy and device region share the layout)
    struct LayerOff {
        size_t ln1w, ln1b, ln2w, ln2b, bqkv, bo, bup, bdown, wqkv, wo, wup, wdown;
    };
    struct Layer {
        float *ln1w, *ln1b, *ln2w, *ln2b, *bqkv, *bo, *bup, *bdown;
        __nv_bfloat16 *wqkv, *wo, *wup, *wdown;
    };
    int layers = 0, proj = 0, nmax = 0;
    float eps = 1e-6f;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    std::string err;
    // weight image
    uint8_t* image = nullptr;
    bool image_pinned = false;
    size_t wbytes = 0;
    size_t o_wpatch = 0, o_bpatch = 0, o_pos = 0, o_postw = 0, o_postb = 0, o_bm0 = 0, o_bm2 = 0, o_wm0 = 0, o_wm2 = 0;
    std::vector<LayerOff> LO;
    // placement
    uint8_t* arena = nullptr;
    size_t arena_size = 0;
    std::vector<Layer> L;
    float *wpatch = nullptr, *bpatch = nullptr, *pos = nullptr, *postw = nullptr, *postb = nullptr, *bm0 = nullptr, *bm2 = nullptr;
    __nv_bfloat16 *wm0 = nullptr, *wm2 = nullptr;
    // work buffers (nmax patches)
    float *d_patches = nullptr, *x = nullptr, *g = nullptr, *rope = nullptr, *d_out = nullptr;
    __nv_bfloat16* act = nullptr;
    __half *qh = nullptr, *kh = nullptr, *vh = nullptr;

    void fail(const std::string& s) { err = s; }
    ~Impl() {
        release_image();
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }
    void release_image() {
        if (!image) return;
        if (image_pinned) cudaFreeHost(image);
        else std::free(image);
        image = nullptr;
    }

    // the work buffers' sizes for `patches` patches, in placement order
    struct Work {
        size_t patches, x, g, act, rope, q, k, v, out, blas;
        size_t total() const { return patches + x + g + act + rope + q + k + v + out + blas; }
    };
    Work work(size_t N) const {
        return {up(N * 768 * 4), up(N * E * 4), up(N * F * 4), up(N * F * 2), up(N * 36 * 2 * 4),
                up((size_t) H * N * DP * 2), up((size_t) H * N * DP * 2), up((size_t) H * N * DP * 2),
                up(N / 4 * (size_t) proj * 4), kBlasWorkspace};
    }

    // Y[M][N] = X[M][K] . W[N][K]^T
    bool gemm_bf16(const __nv_bfloat16* X, const __nv_bfloat16* W, float* Y, int M, int N, int K) {
        const float one = 1.f, zero = 0.f;
        CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &zero, Y,
                        CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        return true;
    }
    bool gemm_f32(const float* X, const float* W, float* Y, int M, int N, int K) {
        const float one = 1.f, zero = 0.f;
        CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, CUDA_R_32F, K, X, CUDA_R_32F, K, &zero, Y,
                        CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        return true;
    }

    bool create(std::string& e) {
        if (stream) return true;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess || cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS ||
            cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) {
            e = "cannot create the encoder's stream and product handle";
            return false;
        }
        return true;
    }

    bool stage(const Mmproj& m, bool pinned, std::string& e) {
        const Hparams& h = m.hp();
        if (h.embd != E || h.heads != H || h.ffn != F || h.patch != 16 || h.merge != 2 || h.layers < 1) {
            e = "unexpected projector hyper-parameters";
            return false;
        }
        layers = h.layers;
        proj = h.proj;
        eps = h.eps;
        size_t cur = 0;
        auto take = [&](size_t bytes) { const size_t o = cur; cur += up(bytes); return o; };
        o_wpatch = take((size_t) E * 768 * 4);
        o_bpatch = take((size_t) E * 4);
        o_pos = take((size_t) E * 2304 * 4);
        o_postw = take((size_t) E * 4);
        o_postb = take((size_t) E * 4);
        o_bm0 = take((size_t) 4 * E * 4);
        o_bm2 = take((size_t) proj * 4);
        o_wm0 = take((size_t) 4 * E * 4 * E * 2);
        o_wm2 = take((size_t) 4 * E * proj * 2);
        LO.assign(layers, {});
        for (LayerOff& y : LO) {
            y.ln1w = take(E * 4); y.ln1b = take(E * 4); y.ln2w = take(E * 4); y.ln2b = take(E * 4);
            y.bqkv = take(3 * E * 4); y.bo = take(E * 4); y.bup = take((size_t) F * 4); y.bdown = take(E * 4);
            y.wqkv = take((size_t) 3 * E * E * 2); y.wo = take((size_t) E * E * 2);
            y.wup = take((size_t) F * E * 2); y.wdown = take((size_t) E * F * 2);
        }
        wbytes = cur;
        release_image();
        if (pinned) {
            if (cudaHostAlloc((void**) &image, wbytes, cudaHostAllocDefault) != cudaSuccess) { e = "cannot pin the encoder weights"; return false; }
            image_pinned = true;
        } else {
            image = (uint8_t*) std::malloc(wbytes);
            image_pinned = false;
            if (!image) { e = "out of memory"; return false; }
        }
        std::memset(image, 0, wbytes);
        std::string er;
        bool ok = true;
        auto put = [&](size_t off, const std::string& n, uint32_t type, int64_t count, size_t esize) {
            const auto* t = m.need(n, type, count, er);
            if (!t) { ok = false; return; }
            std::memcpy(image + off, t->data, (size_t) count * esize);
        };
        auto f32 = [&](size_t off, const std::string& n, int64_t c) { put(off, n, 0, c, 4); };
        auto b16 = [&](size_t off, const std::string& n, int64_t c) { put(off, n, 30, c, 2); };
        {
            const auto* w0 = m.need("v.patch_embd.weight", 0, (int64_t) E * 768, er);
            const auto* w1 = m.need("v.patch_embd.weight.1", 0, (int64_t) E * 768, er);
            if (w0 && w1) {
                float* w = (float*) (image + o_wpatch);
                for (size_t i = 0; i < (size_t) E * 768; ++i) w[i] = ((const float*) w0->data)[i] + ((const float*) w1->data)[i];
            } else ok = false;
        }
        f32(o_bpatch, "v.patch_embd.bias", E);
        f32(o_pos, "v.position_embd.weight", (int64_t) E * 2304);
        f32(o_postw, "v.post_ln.weight", E);
        f32(o_postb, "v.post_ln.bias", E);
        f32(o_bm0, "mm.0.bias", 4 * E);
        f32(o_bm2, "mm.2.bias", proj);
        b16(o_wm0, "mm.0.weight", (int64_t) 4 * E * 4 * E);
        b16(o_wm2, "mm.2.weight", (int64_t) 4 * E * proj);
        for (int l = 0; ok && l < layers; ++l) {
            const std::string p = "v.blk." + std::to_string(l) + ".";
            const LayerOff& y = LO[l];
            f32(y.ln1w, p + "ln1.weight", E); f32(y.ln1b, p + "ln1.bias", E);
            f32(y.ln2w, p + "ln2.weight", E); f32(y.ln2b, p + "ln2.bias", E);
            f32(y.bqkv, p + "attn_qkv.bias", 3 * E); f32(y.bo, p + "attn_out.bias", E);
            f32(y.bup, p + "ffn_up.bias", F); f32(y.bdown, p + "ffn_down.bias", E);
            b16(y.wqkv, p + "attn_qkv.weight", (int64_t) 3 * E * E); b16(y.wo, p + "attn_out.weight", (int64_t) E * E);
            b16(y.wup, p + "ffn_up.weight", (int64_t) F * E); b16(y.wdown, p + "ffn_down.weight", (int64_t) E * F);
        }
        if (!ok) { e = er; release_image(); return false; }
        return true;
    }

    size_t arena_bytes(int max_tokens) const { return wbytes + work((size_t) max_tokens * 4).total(); }

    bool attach(void* base, size_t bytes, int max_tokens, std::string& e) {
        if (!image) { e = "the encoder weights are not staged"; return false; }
        if (!create(e)) return false;
        nmax = max_tokens * 4;
        const Work w = work((size_t) nmax);
        if (bytes < wbytes + w.total()) { e = "the encoder's device region is too small"; return false; }
        arena = (uint8_t*) base;
        arena_size = bytes;
        auto at = [&](size_t off) { return arena + off; };
        wpatch = (float*) at(o_wpatch); bpatch = (float*) at(o_bpatch); pos = (float*) at(o_pos);
        postw = (float*) at(o_postw); postb = (float*) at(o_postb); bm0 = (float*) at(o_bm0); bm2 = (float*) at(o_bm2);
        wm0 = (__nv_bfloat16*) at(o_wm0); wm2 = (__nv_bfloat16*) at(o_wm2);
        L.resize(layers);
        for (int l = 0; l < layers; ++l) {
            const LayerOff& o = LO[l];
            Layer& y = L[l];
            y.ln1w = (float*) at(o.ln1w); y.ln1b = (float*) at(o.ln1b); y.ln2w = (float*) at(o.ln2w); y.ln2b = (float*) at(o.ln2b);
            y.bqkv = (float*) at(o.bqkv); y.bo = (float*) at(o.bo); y.bup = (float*) at(o.bup); y.bdown = (float*) at(o.bdown);
            y.wqkv = (__nv_bfloat16*) at(o.wqkv); y.wo = (__nv_bfloat16*) at(o.wo);
            y.wup = (__nv_bfloat16*) at(o.wup); y.wdown = (__nv_bfloat16*) at(o.wdown);
        }
        size_t cur = wbytes;
        auto next = [&](size_t n) { uint8_t* p = at(cur); cur += n; return p; };
        d_patches = (float*) next(w.patches); x = (float*) next(w.x); g = (float*) next(w.g);
        act = (__nv_bfloat16*) next(w.act); rope = (float*) next(w.rope);
        qh = (__half*) next(w.q); kh = (__half*) next(w.k); vh = (__half*) next(w.v);
        d_out = (float*) next(w.out);
        uint8_t* ws = next(w.blas);
        CB(cublasSetWorkspace(blas, ws, w.blas));
        CK(cudaMemcpyAsync(arena, image, wbytes, cudaMemcpyHostToDevice, stream));
        // the padding columns of q, k, v stay zero for good: the kernels write columns 0..71 only
        CK(cudaMemsetAsync(qh, 0, (size_t) H * nmax * DP * sizeof(__half), stream));
        CK(cudaMemsetAsync(kh, 0, (size_t) H * nmax * DP * sizeof(__half), stream));
        CK(cudaMemsetAsync(vh, 0, (size_t) H * nmax * DP * sizeof(__half), stream));
        CK(cudaStreamSynchronize(stream));
        return true;
    }

    void detach() {
        if (stream) cudaStreamSynchronize(stream);
        arena = nullptr;
        arena_size = 0;
        nmax = 0;
    }

    bool init(const Mmproj& m, int max_tokens, std::string& e) {
        if (!stage(m, false, e)) return false;
        void* base = nullptr;
        const size_t bytes = arena_bytes(max_tokens);
        if (cudaMalloc(&base, bytes) != cudaSuccess) { e = "out of device memory"; return false; }
        own = base;
        if (!attach(base, bytes, max_tokens, e)) { if (e.empty()) e = err; return false; }
        release_image();   // the weights are on the device
        return true;
    }
    void* own = nullptr;   // the arena `init` allocated

    bool run(const std::vector<float>& patches, Size canvas, std::vector<float>& out, const TapFn& tap) {
        const int gx = canvas.w / 16, gy = canvas.h / 16, n = gx * gy, m = n / 4;
        if (n % 4 != 0 || n > nmax || (int) patches.size() != n * 768) { fail("image outside the sized range"); return false; }
        auto host_tap = [&](const std::string& name, const void* dev, size_t count, bool bf) -> bool {
            if (!tap) return true;
            CK(cudaStreamSynchronize(stream));
            std::vector<float> h(count);
            if (!bf) CK(cudaMemcpy(h.data(), dev, count * 4, cudaMemcpyDeviceToHost));
            else {
                std::vector<__nv_bfloat16> b(count);
                CK(cudaMemcpy(b.data(), dev, count * 2, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < count; ++i) h[i] = __bfloat162float(b[i]);
            }
            tap(name, h.data(), count);
            return true;
        };
        CK(cudaMemcpyAsync(d_patches, patches.data(), patches.size() * 4, cudaMemcpyHostToDevice, stream));
        if (!gemm_f32(d_patches, wpatch, x, n, E, 768)) return false;
        launch_embed_finish(x, bpatch, pos, n, gx, gy, stream);
        launch_rope_table(rope, n, gx, stream);
        if (!host_tap("embed", x, (size_t) n * E, false)) return false;
        const float scale_log2 = (1.0f / sqrtf((float) D)) * 1.4426950408889634f;
        for (int l = 0; l < layers; ++l) {
            const Layer& y = L[l];
            launch_layernorm(x, y.ln1w, y.ln1b, act, n, eps, stream);
            if (!gemm_bf16(act, y.wqkv, g, n, 3 * E, E)) return false;
            launch_qkv_epilogue(g, y.bqkv, rope, qh, kh, vh, n, stream);
            launch_attention(qh, kh, vh, act, n, scale_log2, stream);
            if (!gemm_bf16(act, y.wo, g, n, E, E)) return false;
            launch_bias_residual(x, g, y.bo, n, E, stream);
            if (!host_tap("ffn_inp." + std::to_string(l), x, (size_t) n * E, false)) return false;
            launch_layernorm(x, y.ln2w, y.ln2b, act, n, eps, stream);
            if (!gemm_bf16(act, y.wup, g, n, F, E)) return false;
            launch_bias_gelu(g, y.bup, act, n, F, stream);
            if (!gemm_bf16(act, y.wdown, g, n, E, F)) return false;
            launch_bias_residual(x, g, y.bdown, n, E, stream);
            if (!host_tap("layer_out." + std::to_string(l), x, (size_t) n * E, false)) return false;
        }
        launch_layernorm(x, postw, postb, act, n, eps, stream);
        if (!host_tap("post_ln", act, (size_t) n * E, true)) return false;
        if (!gemm_bf16(act, wm0, g, m, 4 * E, 4 * E)) return false;
        launch_bias_gelu(g, bm0, act, m, 4 * E, stream);
        if (!gemm_bf16(act, wm2, g, m, proj, 4 * E)) return false;
        launch_bias_copy(g, bm2, d_out, m, proj, stream);
        out.resize((size_t) m * proj);
        CK(cudaMemcpyAsync(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        CK(cudaGetLastError());
        if (tap) tap("out", out.data(), out.size());
        return true;
    }
};

CudaEncoder::CudaEncoder() : impl_(new Impl) {}
CudaEncoder::~CudaEncoder() {
    if (impl_->own) { impl_->detach(); cudaFree(impl_->own); }
}

bool CudaEncoder::init(const Mmproj& mp, int max_tokens, std::string& err) {
    if (impl_->init(mp, max_tokens, err)) return true;
    if (err.empty()) err = impl_->err;
    return false;
}

bool CudaEncoder::encode(const std::vector<float>& patches, Size canvas, std::vector<float>& out, const TapFn& tap) {
    return impl_->run(patches, canvas, out, tap);
}

bool CudaEncoder::stage(const Mmproj& mp, bool pinned, std::string& err) { return impl_->stage(mp, pinned, err); }
bool CudaEncoder::create(std::string& err) { return impl_->create(err); }
size_t CudaEncoder::weights_bytes() const { return impl_->wbytes; }
size_t CudaEncoder::arena_bytes(int max_tokens) const { return impl_->arena_bytes(max_tokens); }
bool CudaEncoder::attach(void* arena, size_t bytes, int max_tokens, std::string& err) {
    if (impl_->attach(arena, bytes, max_tokens, err)) return true;
    if (err.empty()) err = impl_->err;
    return false;
}
void CudaEncoder::detach() { impl_->detach(); }
size_t CudaEncoder::device_bytes() const { return impl_->arena_size; }
std::string CudaEncoder::last_error() const { return impl_->err; }

}  // namespace basalt::vision
