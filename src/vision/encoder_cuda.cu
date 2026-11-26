// src/vision/encoder_cuda.cu - see basalt/vision/encoder_cuda.hpp.
#include "basalt/core/device_copy.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/vision/encoder_cuda.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "basalt/vision/preprocess.hpp"
#include "hy4_weights.hpp"
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

void* current_context() {
    // the runtime makes its context current on first use; the driver's own call through the runtime's entry
    // points, so nothing links the driver library
    if (cudaFree(nullptr) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q{};
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion("cuCtxGetCurrent", &fn, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint("cuCtxGetCurrent", &fn, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || fn == nullptr || q != cudaDriverEntryPointSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    void* ctx = nullptr;
    return reinterpret_cast<int (*)(void**)>(fn)(&ctx) == 0 ? ctx : nullptr;
}

namespace {
constexpr int H = 16;   // heads, every tower

constexpr size_t kAlign = 256;
// the product library's scratch: a fixed size, so the algorithm choice never depends on where the arena sits
constexpr size_t kBlasWorkspace = 32u << 20;
// merged rows of room beyond a picture's tokens for the newline rows of its lines (a picture 200 times as wide as it is tall
// at most, 4096 merged tokens: 905 lines)
constexpr size_t kNewlineRows = 1024;
// rows of one product of the class-token tower: its work buffers hold a chunk of rows, not the whole sequence
constexpr int kChunkRows = 8192;

size_t up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

}  // namespace

struct CudaEncoder::Impl {
    // byte offsets of the tensors inside the weight image (host copy and device region share the layout)
    struct LayerOff {
        size_t ln1w, ln1b, ln2w, ln2b, bqkv, bo, bup, bdown, wqkv, wo, wup, wdown, ls1, ls2;
    };
    struct Layer {
        float *ln1w, *ln1b, *ln2w, *ln2b, *bqkv, *bo, *bup, *bdown, *ls1, *ls2;
        __nv_bfloat16 *wqkv, *wo, *wup, *wdown;
    };
    int layers = 0, proj = 0, nmax = 0;
    float eps = 1e-6f;
    // the tower's width, MLP width, head width and its padded row in q, k and v, and the values of a patch row
    int E = 1152, F = 4304, D = 72, DP = 80, PV = 768;
    // A SigLIP2-derived tower (projector hy_v4_flash): one patch tap, an edge x edge position table resampled per grid, no rotation,
    // an RMS norm then a 2x2 merge product with an erf GELU, and a newline row after each merged row
    bool hy4 = false;
    int edge = 48;
    // The class-token tower (a file with v.class_embd): 20-pixel patches of a picture on whole cells of patch x merge pixels (the
    // encoder pads and normalises it), a class token in front, a pre-norm, 2D rotary positions on interleaved pairs, layer
    // scale, no post-norm, and the merged rows through two products without bias
    bool cls = false;
    int patch = 16;
    PixelNorm norm{};
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    int device = -1;
    int image_device = -1;
    std::string err;
    // weight image
    uint8_t* image = nullptr;
    bool image_pinned = false;
    size_t wbytes = 0;
    size_t o_wpatch = 0, o_wpatch2 = 0, o_bpatch = 0, o_pos = 0, o_postw = 0, o_postb = 0, o_bm0 = 0, o_bm2 = 0, o_wm0 = 0, o_wm2 = 0, o_newline = 0;
    size_t o_wpatch16 = 0, o_cls = 0;
    std::vector<LayerOff> LO;
    // placement
    uint8_t* arena = nullptr;
    size_t arena_size = 0;
    std::vector<Layer> L;
    float *wpatch = nullptr, *wpatch2 = nullptr, *bpatch = nullptr, *pos = nullptr, *postw = nullptr, *postb = nullptr, *bm0 = nullptr, *bm2 = nullptr, *newline = nullptr;
    float* clsrow = nullptr;
    __nv_bfloat16 *wm0 = nullptr, *wm2 = nullptr, *wpatch16 = nullptr;
    // work buffers (nmax patches; the class-token tower's products a chunk of rows at a time); d_patches holds the picture's
    // bytes for the class-token tower
    float *d_patches = nullptr, *x = nullptr, *g = nullptr, *rope = nullptr, *d_out = nullptr;
    __nv_bfloat16* act = nullptr;
    __half *qh = nullptr, *kh = nullptr, *vh = nullptr;

    struct StreamSyncOnFail {
        cudaStream_t s;
        bool ok = false;
        ~StreamSyncOnFail() {
            if (!ok && s) {
                (void) cudaStreamSynchronize(s);
                (void) cudaGetLastError();
            }
        }
    };

    void fail(const std::string& s) { err = s; }
    ~Impl() {
        const basalt::core::OnDevice od(device);
        if (stream) (void) cudaStreamSynchronize(stream);
        if (own) {
            cudaFree(own);
            own = nullptr;
        }
        release_image();
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }
    void release_image() {
        if (!image) return;
        if (image_pinned) {
            const basalt::core::OnDevice od(image_device >= 0 ? image_device : device);
            cudaFreeHost(image);
        } else {
            std::free(image);
        }
        image = nullptr;
        image_pinned = false;
        image_device = -1;
    }

    // the work buffers' sizes for `patches` patches, in placement order
    struct Work {
        size_t patches, x, g, act, rope, q, k, v, out, blas;
        size_t total() const { return patches + x + g + act + rope + q + k + v + out + blas; }
    };
    Work work(size_t N) const {
        if (cls) {   // the picture's bytes, the sequence (class token first), a chunk's products
            const size_t T = N + 1, C = std::min<size_t>(T, kChunkRows);
            const size_t gcols = std::max<size_t>({(size_t) 3 * E, (size_t) F, (size_t) 4 * E}), acols = std::max<size_t>(gcols, (size_t) PV);
            return {up(N * (size_t) patch * patch * 3), up(T * E * 4), up(C * gcols * 4), up(C * acols * 2), up(T * 32 * 2 * 4),
                    up((size_t) H * T * DP * 2), up((size_t) H * T * DP * 2), up((size_t) H * T * DP * 2),
                    up(N / 4 * (size_t) proj * 4), kBlasWorkspace};
        }
        return {up(N * 768 * 4), up(N * E * 4), up(N * F * 4), up(N * F * 2), up(N * 36 * 2 * 4),
                up((size_t) H * N * DP * 2), up((size_t) H * N * DP * 2), up((size_t) H * N * DP * 2),
                up((N / 4 + (hy4 ? kNewlineRows : 0)) * (size_t) proj * 4), kBlasWorkspace};
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
        if (stream && blas) return true;
        int dev = -1;
        if (cudaGetDevice(&dev) != cudaSuccess) {
            (void) cudaGetLastError();
            e = "cannot query the encoder's CUDA device";
            return false;
        }
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess || cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS ||
            cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) {
            if (blas) { cublasDestroy(blas); blas = nullptr; }
            if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
            (void) cudaGetLastError();
            e = "cannot create the encoder's stream and product handle";
            return false;
        }
        device = dev;
        return true;
    }

    bool stage(const Mmproj& m, bool pinned, std::string& e) {
        const Hparams& h = m.hp();
        cls = m.find("v.class_embd") != nullptr;
        if (cls) {
            E = 1024; F = 4096; D = 64; DP = 64; patch = 20; PV = 3 * 20 * 20;
            for (int c = 0; c < 3; ++c) {
                norm.clip_mean[c] = vision::kClipMean[c];
                norm.clip_std[c] = vision::kClipStd[c];
                norm.mean[c] = h.mean[c];
                norm.std[c] = h.std[c];
            }
        }
        if (h.embd != E || h.heads != H || h.ffn != F || h.patch != patch || h.merge != 2 || h.layers < 1) {
            e = "unexpected projector hyper-parameters";
            return false;
        }
        if (cls) return stage_cls(m, pinned, e);
        hy4 = m.projector() == "hy_v4_flash";
        if (hy4) {
            const TensorRef* pt = m.find("v.position_embd.weight");
            const int64_t rows = pt && pt->ne.size() == 2 && pt->ne[0] == E ? pt->ne[1] : 0;
            edge = (int) std::lround(std::sqrt((double) rows));
            if (edge < 2 || (int64_t) edge * edge != rows) {
                e = "the position table of the projector is not a square grid";
                return false;
            }
        }
        layers = h.layers;
        proj = h.proj;
        eps = h.eps;
        size_t cur = 0;
        auto take = [&](size_t bytes) { const size_t o = cur; cur += up(bytes); return o; };
        o_wpatch = take((size_t) E * 768 * 4);
        o_wpatch2 = take(hy4 ? 0 : (size_t) E * 1536 * 4);
        o_bpatch = take((size_t) E * 4);
        o_pos = take((size_t) E * (hy4 ? (size_t) edge * edge : 2304) * 4);
        o_postw = take((size_t) E * 4);
        o_postb = take(hy4 ? 0 : (size_t) E * 4);
        o_newline = take(hy4 ? (size_t) proj * 4 : 0);
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
            int img_dev = -1;
            if (cudaGetDevice(&img_dev) == cudaSuccess) image_device = img_dev;
            else { (void) cudaGetLastError(); image_device = -1; }
            if (cudaHostAlloc((void**) &image, wbytes, cudaHostAllocDefault) != cudaSuccess) {
                (void) cudaGetLastError();
                image = nullptr;
                image_device = -1;
                e = "cannot pin the encoder weights";
                return false;
            }
            image_pinned = true;
        } else {
            image = (uint8_t*) std::malloc(wbytes);
            image_pinned = false;
            image_device = -1;
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
        if (hy4) {
            f32(o_wpatch, "v.patch_embd.weight", (int64_t) E * 768);
            f32(o_newline, "v.image_newline", proj);
        } else {
            const auto* w0 = m.need("v.patch_embd.weight", 0, (int64_t) E * 768, er);
            const auto* w1 = m.need("v.patch_embd.weight.1", 0, (int64_t) E * 768, er);
            if (w0 && w1) {
                float* w = (float*) (image + o_wpatch);
                for (size_t i = 0; i < (size_t) E * 768; ++i) w[i] = ((const float*) w0->data)[i] + ((const float*) w1->data)[i];
                // a video group's taps apart: per channel, tap 0's 256 weights then tap 1's
                float* w2 = (float*) (image + o_wpatch2);
                for (size_t e = 0; e < (size_t) E; ++e)
                    for (size_t c = 0; c < 3; ++c) {
                        std::memcpy(w2 + e * 1536 + c * 512, (const float*) w0->data + e * 768 + c * 256, 256 * sizeof(float));
                        std::memcpy(w2 + e * 1536 + c * 512 + 256, (const float*) w1->data + e * 768 + c * 256, 256 * sizeof(float));
                    }
            } else ok = false;
        }
        f32(o_bpatch, "v.patch_embd.bias", E);
        f32(o_pos, "v.position_embd.weight", hy4 ? (int64_t) E * edge * edge : (int64_t) E * 2304);
        if (hy4) {
            f32(o_postw, "mm.pre_norm.weight", E);
        } else {
            f32(o_postw, "v.post_ln.weight", E);
            f32(o_postb, "v.post_ln.bias", E);
        }
        f32(o_bm0, "mm.0.bias", 4 * E);
        f32(o_bm2, "mm.2.bias", proj);
        if (hy4) {   // F32 convolutions in the file: the merge product's columns in merge order, bf16
            if (!hy4::merge_matrix(m, E, (uint16_t*) (image + o_wm0), er) || !hy4::out_matrix(m, E, proj, (uint16_t*) (image + o_wm2), er)) ok = false;
        } else {
            b16(o_wm0, "mm.0.weight", (int64_t) 4 * E * 4 * E);
            b16(o_wm2, "mm.2.weight", (int64_t) 4 * E * proj);
        }
        for (int l = 0; ok && l < layers; ++l) {
            const std::string p = "v.blk." + std::to_string(l) + ".";
            const LayerOff& y = LO[l];
            f32(y.ln1w, p + "ln1.weight", E); f32(y.ln1b, p + "ln1.bias", E);
            f32(y.ln2w, p + "ln2.weight", E); f32(y.ln2b, p + "ln2.bias", E);
            f32(y.bo, p + "attn_out.bias", E);
            f32(y.bup, p + "ffn_up.bias", F); f32(y.bdown, p + "ffn_down.bias", E);
            if (hy4) {   // q, k and v are three tensors in the file, one product here
                static const char* const qkv[3] = {"attn_q", "attn_k", "attn_v"};
                for (int i = 0; i < 3; ++i) {
                    f32(y.bqkv + (size_t) i * E * 4, p + qkv[i] + ".bias", E);
                    b16(y.wqkv + (size_t) i * E * E * 2, p + qkv[i] + ".weight", (int64_t) E * E);
                }
            } else {
                f32(y.bqkv, p + "attn_qkv.bias", 3 * E);
                b16(y.wqkv, p + "attn_qkv.weight", (int64_t) 3 * E * E);
            }
            b16(y.wo, p + "attn_out.weight", (int64_t) E * E);
            b16(y.wup, p + "ffn_up.weight", (int64_t) F * E); b16(y.wdown, p + "ffn_down.weight", (int64_t) E * F);
        }
        if (!ok) { e = er; release_image(); return false; }
        return true;
    }

    // The class-token tower's weight image: the patch product and every matrix bf16 (the file's F32 patch convolution holds
    // bf16 values), the class token and its position summed, the merge products' missing biases as zeros.
    bool stage_cls(const Mmproj& m, bool pinned, std::string& e) {
        const Hparams& h = m.hp();
        const TensorRef* pt = m.find("v.position_embd.weight");
        const int64_t rows = pt && pt->ne.size() == 2 && pt->ne[0] == E ? pt->ne[1] : 0;
        edge = (int) std::lround(std::sqrt((double) rows));
        if (edge < 2 || (int64_t) edge * edge != rows) {
            e = "the position table of the projector is not a square grid";
            return false;
        }
        layers = h.layers;
        proj = h.proj;
        eps = h.eps;
        size_t cur = 0;
        auto take = [&](size_t bytes) { const size_t o = cur; cur += up(bytes); return o; };
        o_wpatch16 = take((size_t) E * PV * 2);
        o_pos = take((size_t) E * edge * edge * 4);
        o_cls = take((size_t) E * 4);
        o_postw = take((size_t) E * 4);   // the pre-norm
        o_postb = take((size_t) E * 4);
        o_bm0 = take((size_t) 4 * E * 4);
        o_bm2 = take((size_t) proj * 4);
        o_wm0 = take((size_t) 4 * E * 4 * E * 2);
        o_wm2 = take((size_t) 4 * E * proj * 2);
        LO.assign(layers, {});
        for (LayerOff& y : LO) {
            y.ln1w = take(E * 4); y.ln1b = take(E * 4); y.ln2w = take(E * 4); y.ln2b = take(E * 4);
            y.ls1 = take(E * 4); y.ls2 = take(E * 4);
            y.bqkv = take(3 * E * 4); y.bo = take(E * 4); y.bup = take((size_t) F * 4); y.bdown = take(E * 4);
            y.wqkv = take((size_t) 3 * E * E * 2); y.wo = take((size_t) E * E * 2);
            y.wup = take((size_t) F * E * 2); y.wdown = take((size_t) E * F * 2);
        }
        wbytes = cur;
        release_image();
        if (pinned) {
            int img_dev = -1;
            if (cudaGetDevice(&img_dev) == cudaSuccess) image_device = img_dev;
            else { (void) cudaGetLastError(); image_device = -1; }
            if (cudaHostAlloc((void**) &image, wbytes, cudaHostAllocDefault) != cudaSuccess) {
                (void) cudaGetLastError();
                image = nullptr;
                image_device = -1;
                e = "cannot pin the encoder weights";
                return false;
            }
            image_pinned = true;
        } else {
            image = (uint8_t*) std::malloc(wbytes);
            image_pinned = false;
            image_device = -1;
            if (!image) { e = "out of memory"; return false; }
        }
        std::memset(image, 0, wbytes);   // the merge products' biases stay zero
        std::string er;
        bool ok = true;
        auto put = [&](size_t off, const std::string& n, uint32_t type, int64_t count, size_t esize) {
            const auto* t = m.need(n, type, count, er);
            if (!t) { ok = false; return; }
            std::memcpy(image + off, t->data, (size_t) count * esize);
        };
        auto f32 = [&](size_t off, const std::string& n, int64_t c) { put(off, n, 0, c, 4); };
        auto b16 = [&](size_t off, const std::string& n, int64_t c) { put(off, n, 30, c, 2); };
        if (const auto* w = m.need("v.patch_embd.weight", 0, (int64_t) E * PV, er)) {   // [E][3][patch][patch]: channel, row, column
            uint16_t* d = (uint16_t*) (image + o_wpatch16);
            for (int64_t i = 0; i < (int64_t) E * PV; ++i) d[i] = hy4::bf16_of(((const float*) w->data)[i]);
        } else ok = false;
        f32(o_pos, "v.position_embd.weight", (int64_t) E * edge * edge);
        {
            const auto* a = m.need("v.class_embd", 0, E, er);
            const auto* b = m.need("v.class_position_embd.weight", 0, E, er);
            if (a && b) {
                float* d = (float*) (image + o_cls);
                for (int i = 0; i < E; ++i) d[i] = ((const float*) a->data)[i] + ((const float*) b->data)[i];
            } else ok = false;
        }
        f32(o_postw, "v.pre_ln.weight", E);
        f32(o_postb, "v.pre_ln.bias", E);
        b16(o_wm0, "mm.1.weight", (int64_t) 4 * E * 4 * E);
        b16(o_wm2, "mm.2.weight", (int64_t) 4 * E * proj);
        for (int l = 0; ok && l < layers; ++l) {
            const std::string p = "v.blk." + std::to_string(l) + ".";
            const LayerOff& y = LO[l];
            f32(y.ln1w, p + "ln1.weight", E); f32(y.ln1b, p + "ln1.bias", E);
            f32(y.ln2w, p + "ln2.weight", E); f32(y.ln2b, p + "ln2.bias", E);
            f32(y.ls1, p + "ls1.weight", E); f32(y.ls2, p + "ls2.weight", E);
            static const char* const qkv[3] = {"attn_q", "attn_k", "attn_v"};
            for (int i = 0; i < 3; ++i) {
                f32(y.bqkv + (size_t) i * E * 4, p + qkv[i] + ".bias", E);
                b16(y.wqkv + (size_t) i * E * E * 2, p + qkv[i] + ".weight", (int64_t) E * E);
            }
            f32(y.bo, p + "attn_out.bias", E);
            b16(y.wo, p + "attn_out.weight", (int64_t) E * E);
            f32(y.bup, p + "ffn_up.bias", F); f32(y.bdown, p + "ffn_down.bias", E);
            b16(y.wup, p + "ffn_up.weight", (int64_t) F * E); b16(y.wdown, p + "ffn_down.weight", (int64_t) E * F);
        }
        if (!ok) { e = er; release_image(); return false; }
        return true;
    }

    size_t arena_bytes(int max_tokens) const {
        return max_tokens > 0 && max_tokens <= INT_MAX / 4 ? wbytes + work((size_t) max_tokens * 4).total() : 0;
    }

    bool attach(void* base, size_t bytes, int max_tokens, std::string& e) {
        if (!image) { e = "the encoder weights are not staged"; return false; }
        if (max_tokens <= 0 || max_tokens > INT_MAX / 4) { e = "invalid max_tokens"; return false; }
        if (!create(e)) return false;
        const int new_nmax = max_tokens * 4;
        const Work w = work((size_t) new_nmax);
        if (bytes < wbytes + w.total()) { e = "the encoder's device region is too small"; return false; }
        const basalt::core::OnDevice od(device);
        struct AttachGuard {
            Impl* self;
            bool ok = false;
            ~AttachGuard() {
                if (!ok) {
                    if (self->stream) {
                        (void) cudaStreamSynchronize(self->stream);
                        (void) cudaGetLastError();
                    }
                    self->arena = nullptr;
                    self->arena_size = 0;
                    self->nmax = 0;
                }
            }
        } guard{this};
        arena = (uint8_t*) base;
        arena_size = bytes;
        nmax = new_nmax;
        auto at = [&](size_t off) { return arena + off; };
        wpatch = (float*) at(o_wpatch); wpatch2 = (float*) at(o_wpatch2); bpatch = (float*) at(o_bpatch); pos = (float*) at(o_pos);
        postw = (float*) at(o_postw); postb = (float*) at(o_postb); newline = (float*) at(o_newline); bm0 = (float*) at(o_bm0); bm2 = (float*) at(o_bm2);
        wm0 = (__nv_bfloat16*) at(o_wm0); wm2 = (__nv_bfloat16*) at(o_wm2);
        wpatch16 = (__nv_bfloat16*) at(o_wpatch16); clsrow = (float*) at(o_cls);
        L.resize(layers);
        for (int l = 0; l < layers; ++l) {
            const LayerOff& o = LO[l];
            Layer& y = L[l];
            y.ln1w = (float*) at(o.ln1w); y.ln1b = (float*) at(o.ln1b); y.ln2w = (float*) at(o.ln2w); y.ln2b = (float*) at(o.ln2b);
            y.bqkv = (float*) at(o.bqkv); y.bo = (float*) at(o.bo); y.bup = (float*) at(o.bup); y.bdown = (float*) at(o.bdown);
            y.wqkv = (__nv_bfloat16*) at(o.wqkv); y.wo = (__nv_bfloat16*) at(o.wo);
            y.wup = (__nv_bfloat16*) at(o.wup); y.wdown = (__nv_bfloat16*) at(o.wdown);
            y.ls1 = (float*) at(o.ls1); y.ls2 = (float*) at(o.ls2);
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
        if (DP != D) {
            CK(cudaMemsetAsync(qh, 0, (size_t) H * nmax * DP * sizeof(__half), stream));
            CK(cudaMemsetAsync(kh, 0, (size_t) H * nmax * DP * sizeof(__half), stream));
            CK(cudaMemsetAsync(vh, 0, (size_t) H * nmax * DP * sizeof(__half), stream));
        }
        CK(cudaStreamSynchronize(stream));
        guard.ok = true;
        return true;
    }

    void detach() {
        const basalt::core::OnDevice od(device);
        if (stream) cudaStreamSynchronize(stream);
        arena = nullptr;
        arena_size = 0;
        nmax = 0;
    }

    bool init(const Mmproj& m, int max_tokens, std::string& e) {
        if (max_tokens <= 0 || max_tokens > INT_MAX / 4) { e = "invalid max_tokens"; return false; }
        if (own) {
            const basalt::core::OnDevice od(device);
            detach();
            cudaFree(own);
            own = nullptr;
        }
        if (!stage(m, false, e)) return false;
        if (!create(e)) { release_image(); return false; }
        const basalt::core::OnDevice od(device);
        void* base = nullptr;
        const size_t bytes = arena_bytes(max_tokens);
        if (cudaMalloc(&base, bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            release_image();
            e = "out of device memory";
            return false;
        }
        if (!attach(base, bytes, max_tokens, e)) {
            cudaFree(base);
            release_image();
            if (e.empty()) e = err;
            return false;
        }
        own = base;
        release_image();   // the weights are on the device
        return true;
    }
    void* own = nullptr;   // the arena `init` allocated

    bool run(const std::vector<float>& patches, Size canvas, std::vector<float>& out, const TapFn& tap) {
        if (cls) { fail("this projector reads a picture, not patch rows"); return false; }
        if (!arena || nmax <= 0 || canvas.w <= 0 || canvas.h <= 0) {
            fail("image outside the sized range");
            return false;
        }
        const int gx = canvas.w / 16, gy = canvas.h / 16;
        const int64_t n64 = (int64_t) gx * (int64_t) gy;
        if (gx <= 0 || gy <= 0 || (gx & 1) != 0 || (gy & 1) != 0 || n64 > nmax) {
            fail("image outside the sized range");
            return false;
        }
        const int n = (int) n64, m = n / 4, ow = gx / 2;
        const bool pair = !hy4 && (int64_t) patches.size() == (int64_t) n * 1536;   // a video group: the two taps apart
        const int out_rows = hy4 ? m + m / ow : m;
        if ((!pair && (int64_t) patches.size() != (int64_t) n * 768) || (hy4 && out_rows > nmax / 4 + (int) kNewlineRows)) {
            fail("image outside the sized range");
            return false;
        }
        const basalt::core::OnDevice od(device);
        StreamSyncOnFail guard{stream};
        auto host_tap = [&](const std::string& name, const void* dev, size_t count, bool bf) -> bool {
            if (!tap) return true;
            CK(cudaStreamSynchronize(stream));
            std::vector<float> h(count);
            if (!bf) CK(basalt::core::device_copy(h.data(), dev, count * 4, cudaMemcpyDeviceToHost));
            else {
                std::vector<__nv_bfloat16> b(count);
                CK(basalt::core::device_copy(b.data(), dev, count * 2, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < count; ++i) h[i] = __bfloat162float(b[i]);
            }
            tap(name, h.data(), count);
            return true;
        };
        // a video group's rows are twice as wide: they go to the FFN buffer, which is free until the first layer
        float* in = pair ? g : d_patches;
        CK(cudaMemcpyAsync(in, patches.data(), patches.size() * 4, cudaMemcpyHostToDevice, stream));
        if (!gemm_f32(in, pair ? wpatch2 : wpatch, x, n, E, pair ? 1536 : 768)) return false;
        if (hy4) launch_embed_finish_interp(x, bpatch, pos, n, gx, gy, edge, stream);
        else launch_embed_finish(x, bpatch, pos, n, gx, gy, stream);
        if (!hy4) launch_rope_table(rope, n, gx, stream);
        if (!host_tap("embed", x, (size_t) n * E, false)) return false;
        const float scale_log2 = (1.0f / sqrtf((float) D)) * 1.4426950408889634f;
        for (int l = 0; l < layers; ++l) {
            const Layer& y = L[l];
            launch_layernorm(x, y.ln1w, y.ln1b, act, n, E, eps, stream);
            if (!gemm_bf16(act, y.wqkv, g, n, 3 * E, E)) return false;
            if (hy4) launch_qkv_plain(g, y.bqkv, qh, kh, vh, n, stream);
            else launch_qkv_epilogue(g, y.bqkv, rope, qh, kh, vh, n, stream);
            launch_attention(qh, kh, vh, act, n, 0, n, D, scale_log2, stream);
            if (!gemm_bf16(act, y.wo, g, n, E, E)) return false;
            launch_bias_residual(x, g, y.bo, n, E, stream);
            if (!host_tap("ffn_inp." + std::to_string(l), x, (size_t) n * E, false)) return false;
            launch_layernorm(x, y.ln2w, y.ln2b, act, n, E, eps, stream);
            if (!gemm_bf16(act, y.wup, g, n, F, E)) return false;
            launch_bias_gelu(g, y.bup, act, n, F, stream);
            if (!gemm_bf16(act, y.wdown, g, n, E, F)) return false;
            launch_bias_residual(x, g, y.bdown, n, E, stream);
            if (!host_tap("layer_out." + std::to_string(l), x, (size_t) n * E, false)) return false;
        }
        if (hy4) launch_rmsnorm(x, postw, act, n, eps, stream);
        else launch_layernorm(x, postw, postb, act, n, E, eps, stream);
        if (!host_tap("post_ln", act, (size_t) n * E, true)) return false;
        if (!gemm_bf16(act, wm0, g, m, 4 * E, 4 * E)) return false;
        if (hy4) launch_bias_gelu_erf(g, bm0, act, m, 4 * E, stream);
        else launch_bias_gelu(g, bm0, act, m, 4 * E, stream);
        if (!gemm_bf16(act, wm2, g, m, proj, 4 * E)) return false;
        if (hy4) launch_bias_copy_lines(g, bm2, newline, d_out, m, proj, ow, stream);
        else launch_bias_copy(g, bm2, d_out, m, proj, stream);
        out.resize((size_t) out_rows * proj);
        CK(cudaMemcpyAsync(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        CK(cudaGetLastError());
        guard.ok = true;
        if (tap) tap("out", out.data(), out.size());
        return true;
    }
    // The class-token tower on a picture at its own size: its bytes go to the device, which cuts them into normalised bf16
    // patch rows; the sequence is the class token and the patches in merge order; every product runs a chunk of rows.
    bool run_cls(const Image& img, std::vector<float>& out, const TapFn& tap) {
        if (!arena || nmax <= 0 || img.w <= 0 || img.h <= 0 ||
            img.rgb.size() != (size_t) img.w * (size_t) img.h * 3) {
            fail("image outside the sized range");
            return false;
        }
        const int cell = 2 * patch;
        const int64_t cols64 = ((int64_t) img.w + cell - 1) / cell;
        const int64_t rows64 = ((int64_t) img.h + cell - 1) / cell;
        const int64_t n64 = (2 * cols64) * (2 * rows64);
        if (n64 <= 0 || n64 > nmax) {
            fail("image outside the sized range");
            return false;
        }
        const int cols = (int) cols64, rows = (int) rows64;
        const int gx = 2 * cols, gy = 2 * rows, n = (int) n64, T = n + 1, m = cols * rows;
        const basalt::core::OnDevice od(device);
        StreamSyncOnFail guard{stream};
        auto host_tap = [&](const std::string& name, const float* dev, size_t count) -> bool {
            if (!tap) return true;
            CK(cudaStreamSynchronize(stream));
            std::vector<float> hbuf(count);
            CK(basalt::core::device_copy(hbuf.data(), dev, count * 4, cudaMemcpyDeviceToHost));
            tap(name, hbuf.data(), count);
            return true;
        };
        const uint8_t* d_img = (const uint8_t*) d_patches;
        CK(cudaMemcpyAsync(d_patches, img.rgb.data(), img.rgb.size(), cudaMemcpyHostToDevice, stream));
        for (int r0 = 0; r0 < n; r0 += kChunkRows) {
            const int c = std::min(kChunkRows, n - r0);
            launch_patch_rows(d_img, img.w, img.h, gx, patch, norm, act, r0, c, stream);
            if (!gemm_bf16(act, wpatch16, x + (size_t) (1 + r0) * E, c, E, PV)) return false;
        }
        launch_embed_cls(x, clsrow, pos, gx, gy, edge, E, stream);
        if (!host_tap("embed", x, (size_t) T * E)) return false;
        launch_layernorm_inplace(x, postw, postb, T, eps, stream);
        if (!host_tap("pre_ln", x, (size_t) T * E)) return false;
        launch_rope_axial(rope, T, gx, stream);
        const float scale_log2 = (1.0f / sqrtf((float) D)) * 1.4426950408889634f;
        for (int l = 0; l < layers; ++l) {
            const Layer& y = L[l];
            for (int r0 = 0; r0 < T; r0 += kChunkRows) {   // q, k and v of every row before any row's attention
                const int c = std::min(kChunkRows, T - r0);
                launch_layernorm(x + (size_t) r0 * E, y.ln1w, y.ln1b, act, c, E, eps, stream);
                if (!gemm_bf16(act, y.wqkv, g, c, 3 * E, E)) return false;
                launch_qkv_axial(g, y.bqkv, rope, qh, kh, vh, T, r0, c, stream);
            }
            for (int r0 = 0; r0 < T; r0 += kChunkRows) {
                const int c = std::min(kChunkRows, T - r0);
                float* xr = x + (size_t) r0 * E;
                launch_attention(qh, kh, vh, act, T, r0, r0 + c, D, scale_log2, stream);
                if (!gemm_bf16(act, y.wo, g, c, E, E)) return false;
                launch_bias_scale_residual(xr, g, y.bo, y.ls1, c, E, stream);
                launch_layernorm(xr, y.ln2w, y.ln2b, act, c, E, eps, stream);
                if (!gemm_bf16(act, y.wup, g, c, F, E)) return false;
                launch_bias_gelu_erf(g, y.bup, act, c, F, stream);
                if (!gemm_bf16(act, y.wdown, g, c, E, F)) return false;
                launch_bias_scale_residual(xr, g, y.bdown, y.ls2, c, E, stream);
            }
            if (!host_tap("layer_out." + std::to_string(l), x, (size_t) T * E)) return false;
        }
        // a cell's four patch rows follow each other: the patches' rows are the merge products' [m][4E] input
        for (int c0 = 0; c0 < m; c0 += kChunkRows) {
            const int c = std::min(kChunkRows, m - c0);
            launch_to_bf16(x + (size_t) (1 + 4 * c0) * E, act, (size_t) c * 4 * E, stream);
            if (!gemm_bf16(act, wm0, g, c, 4 * E, 4 * E)) return false;
            launch_bias_gelu_erf(g, bm0, act, c, 4 * E, stream);
            if (!gemm_bf16(act, wm2, d_out + (size_t) c0 * proj, c, proj, 4 * E)) return false;
        }
        out.resize((size_t) m * proj);
        CK(cudaMemcpyAsync(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        CK(cudaGetLastError());
        guard.ok = true;
        if (tap) tap("out", out.data(), out.size());
        return true;
    }
};

CudaEncoder::CudaEncoder() : impl_(new Impl) {}
CudaEncoder::~CudaEncoder() = default;

bool CudaEncoder::init(const Mmproj& mp, int max_tokens, std::string& err) {
    if (impl_->init(mp, max_tokens, err)) return true;
    if (err.empty()) err = impl_->err;
    return false;
}

bool CudaEncoder::encode(const std::vector<float>& patches, Size canvas, std::vector<float>& out, const TapFn& tap) {
    return impl_->run(patches, canvas, out, tap);
}

bool CudaEncoder::encode_padded(const Image& img, std::vector<float>& out, const TapFn& tap) {
    if (!impl_->cls) {
        impl_->fail("this projector reads patch rows, not a picture");
        return false;
    }
    return impl_->run_cls(img, out, tap);
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
