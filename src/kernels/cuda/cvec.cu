// src/kernels/cuda/cvec.cu - see include/basalt/kernels/cvec.hpp.
#include "basalt/kernels/cvec.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace basalt::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int64_t kMaxEmb = 32 * THREADS;   // the widest residual the kernels below hold in registers (K = 32)

// the block's two-level sum over one row (its value ranges never span more than one block)
__device__ __forceinline__ float block_sum(float v, float* part) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float p = threadIdx.x < THREADS / 32 ? part[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
        if (threadIdx.x == 0) part[0] = p;
    }
    __syncthreads();
    return part[0];
}

// the fused hyper-connection read's gate (fused_gr.cu), so a write done here is bitwise the one it would have folded
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// One block per (stream, token) of a hyper-connection model's FP32 residual: the pending FFN write, then h . v over
// the stream, then the update.  The row is held in registers between the two, so it is read once.  `K` is the number
// of value ranges of THREADS a row takes; every instantiation covers one width band.
template <int K>
__global__ void cvec_kernel(float* __restrict__ R, const float* __restrict__ dir, const float* __restrict__ s_l,
                            const int* __restrict__ on, int mode, int64_t layer, int n, int hc, int64_t r_ld,
                            const float* __restrict__ bo, int64_t bo_ld, const float* __restrict__ inj,
                            int64_t inj_ld, int write) {
    __shared__ float part[THREADS / 32];
    const int c = blockIdx.x;
    const int64_t t = blockIdx.y;
    float* r = R + t * r_ld + (int64_t) c * n;
    const float s = s_l[layer];
    const bool steer = *on != 0 && s != 0.0f;   // uniform over the block
    if (!steer && !write) return;
    const float* v = dir + layer * n;
    const float w = write ? 2.0f * sigmoidf_(inj[t * inj_ld + c] / (float) hc) : 0.0f;
    const float* b = write ? bo + t * bo_ld : nullptr;
    float x[K];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            float xv = r[d];
            if (write) xv = fmaf(b[d], w, xv);
            x[k] = xv;
            if (steer && mode == 0) dot = fmaf(xv, v[d], dot);
        }
    }
    if (steer && mode == 0) dot = block_sum(dot, part) * s;   // s (h . v)
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            float xv = x[k];
            if (steer) xv = mode == 0 ? fmaf(-dot, v[d], xv) : xv + v[d];
            r[d] = xv;
        }
    }
}

// One block per token of a dense model's BF16 residual row, its layer's output already added: the dot over the row,
// then the update, each value rounded back to BF16 once.  A row is wider than a hyper-connection stream, so it is
// read twice instead of held in registers; the second read is the block's own L2 lines.
template <int K>
__global__ void cvec_bf16_kernel(__nv_bfloat16* __restrict__ R, const float* __restrict__ dir,
                                 const float* __restrict__ s_l, const int* __restrict__ on, int mode, int64_t layer,
                                 int n, int64_t r_ld) {
    __shared__ float part[THREADS / 32];
    pdl_launch_dependents();
    pdl_wait();
    __nv_bfloat16* r = R + (int64_t) blockIdx.x * r_ld;
    const float s = s_l[layer];
    if (*on == 0 || s == 0.0f) return;
    const float* v = dir + layer * n;
    float dot = 0.0f;
    if (mode == 0) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int d = threadIdx.x + k * THREADS;
            if (d < n) dot = fmaf(__bfloat162float(r[d]), v[d], dot);
        }
        dot = block_sum(dot, part) * s;   // s (h . v)
    }
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            const float x = __bfloat162float(r[d]);
            r[d] = __float2bfloat16_rn(mode == 0 ? fmaf(-dot, v[d], x) : x + v[d]);
        }
    }
}

Cvec g_cvec;                 // the description; its device pointers are the uploading device's
bool g_on_host = false;
// the tables on every device that holds them (a layer split applies the vector on several)
constexpr int kDevices = 64;
struct DevTables { float* dir = nullptr; float* s = nullptr; int* on = nullptr; };
DevTables g_dev[kDevices];
std::vector<float> g_dir_host, g_s_host;
int cur_device() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kDevices) d = 0;
    return d;
}
bool upload_here(std::string& err) {
    DevTables& t = g_dev[cur_device()];
    if (t.dir != nullptr) return true;
    const int flag = g_on_host ? 1 : 0;
    if (cudaMalloc(&t.dir, g_dir_host.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&t.s, g_s_host.size() * sizeof(float)) != cudaSuccess || cudaMalloc(&t.on, sizeof(int)) != cudaSuccess ||
        cudaMemcpy(t.dir, g_dir_host.data(), g_dir_host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(t.s, g_s_host.data(), g_s_host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(t.on, &flag, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "control vector: device allocation failed";
        t = DevTables{};
        return false;
    }
    return true;
}

/// The `K` a row of `n` values runs with, or -1 when no instantiation covers it.
constexpr int chunks_for(int64_t n) {
    const int64_t c = (n + THREADS - 1) / THREADS;   // the value ranges of THREADS the row takes
    return c <= 4 ? 4 : c <= 8 ? 8 : c <= 16 ? 16 : c <= 24 ? 24 : c <= 32 ? 32 : -1;
}

}  // namespace

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || chunks_for(n_embd) < 0) {
        err = "control vector: a residual of " + std::to_string((long long) n_embd) + " values is wider than the " +
              std::to_string((long long) kMaxEmb) + " the kernel covers";
        return false;
    }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    int prev = 0;   // a new vector replaces the old one on every device
    cudaGetDevice(&prev);
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].dir == nullptr) continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();
        cudaFree(g_dev[d].dir);
        cudaFree(g_dev[d].s);
        cudaFree(g_dev[d].on);
        g_dev[d] = DevTables{};
    }
    cudaSetDevice(prev);
    g_dir_host = dir;
    g_s_host = s;
    g_on_host = true;
    if (!upload_here(err)) return false;
    const DevTables& t = g_dev[cur_device()];
    g_cvec.dir = t.dir;
    g_cvec.s = t.s;
    g_cvec.on = t.on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    return true;
}

bool cvec_replicate(std::string& err) { return !g_cvec.loaded() || upload_here(err); }

void cvec_set_enabled(bool on) {
    if (!g_cvec.loaded() || on == g_on_host) return;
    int prev = 0;
    cudaGetDevice(&prev);
    const int v = on ? 1 : 0;
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].on == nullptr) continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();   // nothing in flight may still read the flag
        cudaMemcpy(g_dev[d].on, &v, sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaSetDevice(prev);
    g_on_host = on;
}

bool cvec_enabled() { return g_cvec.loaded() && g_on_host; }

void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj,
                int64_t inj_ld, bool write, void* stream) {
    if (!g_cvec.loaded() || T < 1) return;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) throw std::runtime_error("cvec_apply: the control vector is not on this device (cvec_replicate)");
    const dim3 grid((unsigned) g_cvec.hc, (unsigned) T);
    const cudaStream_t s = (cudaStream_t) stream;
    const int n = (int) g_cvec.n_embd, hc = (int) g_cvec.hc, mode = g_cvec.mode;
    const int w = write ? 1 : 0;
#define CVEC_FP32(K) cvec_kernel<K><<<grid, THREADS, 0, s>>>(R, t.dir, t.s, t.on, mode, layer, n, hc, r_ld, bo, bo_ld, inj, inj_ld, w)
    switch (chunks_for(g_cvec.n_embd)) {   // cvec_upload refused anything wider, so one of these always covers the row
        case 4: CVEC_FP32(4); break;
        case 8: CVEC_FP32(8); break;
        case 16: CVEC_FP32(16); break;
        case 24: CVEC_FP32(24); break;
        case 32: CVEC_FP32(32); break;
        default: throw std::runtime_error("cvec_apply: no kernel for this row width");
    }
#undef CVEC_FP32
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("cvec_apply: launch failed");
}

void cvec_apply_bf16(uint16_t* R, int64_t layer, int64_t T, int64_t r_ld, void* stream) {
    if (!g_cvec.loaded() || T < 1) return;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr)
        throw std::runtime_error("cvec_apply_bf16: the control vector is not on this device (cvec_replicate)");
    const dim3 grid((unsigned) T);
    const cudaStream_t s = (cudaStream_t) stream;
    const int n = (int) g_cvec.n_embd, mode = g_cvec.mode;
    auto* dir = (__nv_bfloat16*) R;
#define CVEC_BF16(K) launch_pdl(cvec_bf16_kernel<K>, grid, dim3(THREADS), 0, s, dir, t.dir, t.s, t.on, mode, layer, n, r_ld)
    switch (chunks_for(g_cvec.n_embd)) {   // cvec_upload refused anything wider, so one of these always covers the row
        case 4: CVEC_BF16(4); break;
        case 8: CVEC_BF16(8); break;
        case 16: CVEC_BF16(16); break;
        case 24: CVEC_BF16(24); break;
        case 32: CVEC_BF16(32); break;
        default: throw std::runtime_error("cvec_apply_bf16: no kernel for this row width");
    }
#undef CVEC_BF16
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("cvec_apply_bf16: launch failed");
}

}  // namespace basalt::kernels
