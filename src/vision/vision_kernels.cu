// src/vision/vision_kernels.cu - see vision_kernels.cuh.
#include "vision_kernels.cuh"

#include <cstdint>

namespace basalt::vision {

namespace {

constexpr int E = 1152, H = 16, D = 72, DP = 80, DS = 88;   // DS: shared-memory row stride (halves), conflict-free
constexpr int BQ = 64, BK = 64;

__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
}

// bilinear source taps along one axis of the 48-entry table, align-corners
__device__ __forceinline__ void axis_taps(int out_n, int i, int& i0, int& i1, float& d) {
    constexpr int S = 48;
    float sf = (float) out_n / S;
    if (out_n > 1) sf = (float) (out_n - 1) / (S - 1);
    const float v = (float) i / sf;
    int a = (int) floorf(v), b = a + 1;
    a = max(0, min(a, S - 1));
    b = max(0, min(b, S - 1));
    d = fmaxf(0.0f, fminf(v - (float) a, 1.0f));
    i0 = a;
    i1 = b;
}

// merge-order row -> patch row and column
__device__ __forceinline__ void patch_pos(int t, int gx, int& py, int& px) {
    const int g = t >> 2, r = t & 3, per = gx >> 1;
    py = (g / per) * 2 + (r >> 1);
    px = (g % per) * 2 + (r & 1);
}

__global__ void embed_finish_k(float* x, const float* bias, const float* pos, int gx, int gy) {
    const int t = blockIdx.x, S = 48;
    int py, px, x0, x1, y0, y1;
    float dx, dy;
    patch_pos(t, gx, py, px);
    axis_taps(gx, px, x0, x1, dx);
    axis_taps(gy, py, y0, y1, dy);
    for (int c = threadIdx.x; c < E; c += blockDim.x) {
        const float a = pos[((size_t) y0 * S + x0) * E + c], b = pos[((size_t) y0 * S + x1) * E + c],
                    cc = pos[((size_t) y1 * S + x0) * E + c], d = pos[((size_t) y1 * S + x1) * E + c];
        const float pe = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + cc * (1 - dx) * dy + d * dx * dy;
        x[(size_t) t * E + c] += bias[c] + pe;
    }
}

__global__ void rope_table_k(float* tab, int n, int gx) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int py, px;
    patch_pos(t, gx, py, px);
    const float ts = powf(10000.0f, -2.0f / 36);
    float th = (float) py;
    for (int j = 0; j < 36; ++j) {
        if (j == 18) th = (float) px;
        float s, c;
        sincosf(th, &s, &c);
        tab[((size_t) t * 36 + j) * 2] = c;
        tab[((size_t) t * 36 + j) * 2 + 1] = s;
        th *= ts;
    }
}

// one warp a row: 36 values a lane
__global__ void layernorm_k(const float* x, const float* w, const float* b, __nv_bfloat16* y, int n, float eps) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= n) return;
    const float* xr = x + (size_t) row * E;
    float v[36];
    float s = 0;
#pragma unroll
    for (int i = 0; i < 36; ++i) { v[i] = xr[i * 32 + lane]; s += v[i]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / E;
    float q = 0;
#pragma unroll
    for (int i = 0; i < 36; ++i) { const float d = v[i] - mean; q += d * d; }
#pragma unroll
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float inv = 1.0f / sqrtf(q / E + eps);
#pragma unroll
    for (int i = 0; i < 36; ++i) {
        const int c = i * 32 + lane;
        y[(size_t) row * E + c] = __float2bfloat16_rn((v[i] - mean) * inv * w[c] + b[c]);
    }
}

// one block a token: the 16 x 2 x 36 rotary pairs of q and k, then v
__global__ void qkv_epilogue_k(const float* g, const float* bias, const float* rope, __half* q, __half* k, __half* v, int n) {
    const int t = blockIdx.x;
    const float* r = g + (size_t) t * 3 * E;
    for (int i = threadIdx.x; i < 2 * H * 36; i += blockDim.x) {
        const int part = i / (H * 36), rem = i % (H * 36), hd = rem / 36, j = rem % 36;
        const int base = part * E + hd * D;
        const float a = r[base + j] + bias[base + j], b = r[base + j + 36] + bias[base + j + 36];
        const float c = rope[((size_t) t * 36 + j) * 2], s = rope[((size_t) t * 36 + j) * 2 + 1];
        __half* dst = (part == 0 ? q : k) + ((size_t) hd * n + t) * DP;
        dst[j] = __float2half_rn(a * c - b * s);
        dst[j + 36] = __float2half_rn(a * s + b * c);
    }
    for (int i = threadIdx.x; i < H * D; i += blockDim.x) {
        const int hd = i / D, d = i % D;
        v[((size_t) hd * n + t) * DP + d] = __float2half_rn(r[2 * E + i] + bias[2 * E + i]);
    }
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Flash attention, one head and 64 queries a block (4 warps, 16 query rows each).  QK^T over the zero-padded
// 80 columns in 5 k-steps; P.V over the 72 real columns in 9 n-tiles.
__global__ void __launch_bounds__(128) attention_k(const __half* __restrict__ Q, const __half* __restrict__ K,
                                                    const __half* __restrict__ V, __nv_bfloat16* __restrict__ O, int n,
                                                    float scale_log2) {
    __shared__ __align__(16) __half sQ[BQ * DS], sK[BK * DS], sV[BK * DS];
    const int h = blockIdx.y, q0 = blockIdx.x * BQ, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gq = lane >> 2, tg = lane & 3;
    const __half* Qh = Q + (size_t) h * n * DP;
    const __half* Kh = K + (size_t) h * n * DP;
    const __half* Vh = V + (size_t) h * n * DP;
    const uint4 zero4 = make_uint4(0, 0, 0, 0);
    for (int i = tid; i < BQ * 10; i += 128) {
        const int r = i / 10, c = i % 10;
        *reinterpret_cast<uint4*>(&sQ[r * DS + c * 8]) =
            q0 + r < n ? *reinterpret_cast<const uint4*>(Qh + (size_t) (q0 + r) * DP + c * 8) : zero4;
    }
    __syncthreads();
    uint32_t qa[5][4];
    const int r0 = warp * 16 + gq;
#pragma unroll
    for (int kk = 0; kk < 5; ++kk) {
        qa[kk][0] = *reinterpret_cast<const uint32_t*>(&sQ[r0 * DS + kk * 16 + tg * 2]);
        qa[kk][1] = *reinterpret_cast<const uint32_t*>(&sQ[(r0 + 8) * DS + kk * 16 + tg * 2]);
        qa[kk][2] = *reinterpret_cast<const uint32_t*>(&sQ[r0 * DS + kk * 16 + tg * 2 + 8]);
        qa[kk][3] = *reinterpret_cast<const uint32_t*>(&sQ[(r0 + 8) * DS + kk * 16 + tg * 2 + 8]);
    }
    float o[9][4];
#pragma unroll
    for (int i = 0; i < 9; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    for (int k0 = 0; k0 < n; k0 += BK) {
        __syncthreads();
        for (int i = tid; i < BK * 10; i += 128) {
            const int r = i / 10, c = i % 10;
            const bool ok = k0 + r < n;
            *reinterpret_cast<uint4*>(&sK[r * DS + c * 8]) = ok ? *reinterpret_cast<const uint4*>(Kh + (size_t) (k0 + r) * DP + c * 8) : zero4;
            *reinterpret_cast<uint4*>(&sV[r * DS + c * 8]) = ok ? *reinterpret_cast<const uint4*>(Vh + (size_t) (k0 + r) * DP + c * 8) : zero4;
        }
        __syncthreads();
        float s[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 5; ++kk)
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sK[(nt * 8 + gq) * DS + kk * 16 + tg * 2]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sK[(nt * 8 + gq) * DS + kk * 16 + tg * 2 + 8]);
                mma16816(s[nt], qa[kk], b0, b1);
            }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int key = k0 + nt * 8 + tg * 2 + (e & 1);
                s[nt][e] = key < n ? s[nt][e] * scale_log2 : -INFINITY;
                if (e < 2) mx0 = fmaxf(mx0, s[nt][e]); else mx1 = fmaxf(mx1, s[nt][e]);
            }
#pragma unroll
        for (int o2 = 1; o2 <= 2; o2 <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o2));
        }
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float al0 = exp2f(m0 - mn0), al1 = exp2f(m1 - mn1);
        m0 = mn0;
        m1 = mn1;
        float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = exp2f(s[nt][0] - mn0);
            s[nt][1] = exp2f(s[nt][1] - mn0);
            s[nt][2] = exp2f(s[nt][2] - mn1);
            s[nt][3] = exp2f(s[nt][3] - mn1);
            sum0 += s[nt][0] + s[nt][1];
            sum1 += s[nt][2] + s[nt][3];
        }
        l0 = l0 * al0 + sum0;
        l1 = l1 * al1 + sum1;
#pragma unroll
        for (int dt = 0; dt < 9; ++dt) {
            o[dt][0] *= al0; o[dt][1] *= al0;
            o[dt][2] *= al1; o[dt][3] *= al1;
        }
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[4];
            a[0] = pack_half2(s[2 * kt][0], s[2 * kt][1]);
            a[1] = pack_half2(s[2 * kt][2], s[2 * kt][3]);
            a[2] = pack_half2(s[2 * kt + 1][0], s[2 * kt + 1][1]);
            a[3] = pack_half2(s[2 * kt + 1][2], s[2 * kt + 1][3]);
#pragma unroll
            for (int dt = 0; dt < 9; ++dt) {
                uint32_t b0, b1;
                const uint32_t addr = (uint32_t) __cvta_generic_to_shared(&sV[(kt * 16 + (lane & 15)) * DS + dt * 8]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(addr));
                mma16816(o[dt], a, b0, b1);
            }
        }
    }
#pragma unroll
    for (int o2 = 1; o2 <= 2; o2 <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, o2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, o2);
    }
    const float i0 = 1.0f / l0, i1 = 1.0f / l1;
    const int ra = q0 + warp * 16 + gq, rb = ra + 8;
#pragma unroll
    for (int dt = 0; dt < 9; ++dt) {
        const int col = h * D + dt * 8 + tg * 2;
        if (ra < n) *reinterpret_cast<__nv_bfloat162*>(&O[(size_t) ra * E + col]) = __floats2bfloat162_rn(o[dt][0] * i0, o[dt][1] * i0);
        if (rb < n) *reinterpret_cast<__nv_bfloat162*>(&O[(size_t) rb * E + col]) = __floats2bfloat162_rn(o[dt][2] * i1, o[dt][3] * i1);
    }
}

__global__ void bias_residual_k(float* x, const float* g, const float* bias, size_t total, int cols) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += g[i] + bias[i % cols];
}
__global__ void bias_gelu_k(const float* g, const float* bias, __nv_bfloat16* y, size_t total, int cols) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) y[i] = __float2bfloat16_rn(gelu_tanh(g[i] + bias[i % cols]));
}
__global__ void bias_copy_k(const float* g, const float* bias, float* y, size_t total, int cols) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) y[i] = g[i] + bias[i % cols];
}

__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f)); }

// Hy4: bilinear taps along one axis of an S-entry table, align_corners false, with the reference's scale factor
// (size + 0.1) / S: source = (i + 0.5) / scale_factor - 0.5, clamped at 0 (torch's upsample_bilinear2d)
// An axis whose output size is the table's is copied (torch's special case; inv_scale 0).
__device__ __forceinline__ void interp_taps(float inv_scale, int S, int i, int& i0, int& i1, float& d) {
    if (inv_scale == 0.0f) { i0 = i1 = i; d = 0.0f; return; }
    const float src = fmaxf(inv_scale * ((float) i + 0.5f) - 0.5f, 0.0f);
    const int a = min((int) floorf(src), S - 1);
    i0 = a;
    i1 = a + (a < S - 1 ? 1 : 0);
    d = fminf(fmaxf(src - (float) a, 0.0f), 1.0f);
}

__global__ void embed_finish_interp_k(float* x, const float* bias, const float* pos, int gx, int S, float inv_sx, float inv_sy) {
    const int t = blockIdx.x;
    int py, px, x0, x1, y0, y1;
    float dx, dy;
    patch_pos(t, gx, py, px);
    interp_taps(inv_sx, S, px, x0, x1, dx);
    interp_taps(inv_sy, S, py, y0, y1, dy);
    for (int c = threadIdx.x; c < E; c += blockDim.x) {
        const float a = pos[((size_t) y0 * S + x0) * E + c], b = pos[((size_t) y0 * S + x1) * E + c],
                    cc = pos[((size_t) y1 * S + x0) * E + c], d = pos[((size_t) y1 * S + x1) * E + c];
        const float pe = (1 - dy) * ((1 - dx) * a + dx * b) + dy * ((1 - dx) * cc + dx * d);
        x[(size_t) t * E + c] += bias[c] + pe;
    }
}

// the split of the plain qkv product (no rotation): bias, then the fp16 head-major q, k, v
__global__ void qkv_plain_k(const float* g, const float* bias, __half* q, __half* k, __half* v, int n) {
    const int t = blockIdx.x;
    const float* r = g + (size_t) t * 3 * E;
    for (int i = threadIdx.x; i < H * D; i += blockDim.x) {
        const int hd = i / D, d = i % D;
        const size_t o = ((size_t) hd * n + t) * DP + d;
        q[o] = __float2half_rn(r[i] + bias[i]);
        k[o] = __float2half_rn(r[E + i] + bias[E + i]);
        v[o] = __float2half_rn(r[2 * E + i] + bias[2 * E + i]);
    }
}

// one warp a row: y = x / sqrt(mean(x^2) + eps) * w, as bf16
__global__ void rmsnorm_k(const float* x, const float* w, __nv_bfloat16* y, int n, float eps) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= n) return;
    const float* xr = x + (size_t) row * E;
    float v[36];
    float q = 0;
#pragma unroll
    for (int i = 0; i < 36; ++i) { v[i] = xr[i * 32 + lane]; q += v[i] * v[i]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float inv = 1.0f / sqrtf(q / E + eps);
#pragma unroll
    for (int i = 0; i < 36; ++i) {
        const int c = i * 32 + lane;
        y[(size_t) row * E + c] = __float2bfloat16_rn(v[i] * inv * w[c]);
    }
}

__global__ void bias_gelu_erf_k(const float* g, const float* bias, __nv_bfloat16* y, size_t total, int cols) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) y[i] = __float2bfloat16_rn(gelu_erf(g[i] + bias[i % cols]));
}

// m merged rows g[m][cols] + bias as lines of `ow` rows, each followed by the newline row: y[m + m / ow][cols]
__global__ void bias_copy_lines_k(const float* g, const float* bias, const float* newline, float* y, size_t total, int cols, int ow) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t r = i / cols;
    const int c = (int) (i % cols);
    const size_t line = r / (ow + 1), col = r % (ow + 1);
    y[i] = col == (size_t) ow ? newline[c] : g[(line * ow + col) * cols + c] + bias[c];
}

inline unsigned blocks(size_t total, int per) { return (unsigned) ((total + per - 1) / per); }

}  // namespace

void launch_embed_finish(float* x, const float* bias, const float* pos_table, int n, int gx, int gy, cudaStream_t s) {
    embed_finish_k<<<n, 128, 0, s>>>(x, bias, pos_table, gx, gy);
}
void launch_rope_table(float* tab, int n, int gx, cudaStream_t s) { rope_table_k<<<blocks(n, 128), 128, 0, s>>>(tab, n, gx); }
void launch_layernorm(const float* x, const float* w, const float* b, __nv_bfloat16* y, int n, float eps, cudaStream_t s) {
    layernorm_k<<<blocks(n, 4), 128, 0, s>>>(x, w, b, y, n, eps);
}
void launch_qkv_epilogue(const float* g, const float* bias, const float* rope, __half* q, __half* k, __half* v, int n,
                         cudaStream_t s) {
    qkv_epilogue_k<<<n, 256, 0, s>>>(g, bias, rope, q, k, v, n);
}
void launch_attention(const __half* q, const __half* k, const __half* v, __nv_bfloat16* out, int n, float scale_log2,
                      cudaStream_t s) {
    attention_k<<<dim3(blocks(n, BQ), H), 128, 0, s>>>(q, k, v, out, n, scale_log2);
}
void launch_bias_residual(float* x, const float* g, const float* bias, int n, int cols, cudaStream_t s) {
    bias_residual_k<<<blocks((size_t) n * cols, 256), 256, 0, s>>>(x, g, bias, (size_t) n * cols, cols);
}
void launch_bias_gelu(const float* g, const float* bias, __nv_bfloat16* y, int n, int cols, cudaStream_t s) {
    bias_gelu_k<<<blocks((size_t) n * cols, 256), 256, 0, s>>>(g, bias, y, (size_t) n * cols, cols);
}
void launch_bias_copy(const float* g, const float* bias, float* y, int n, int cols, cudaStream_t s) {
    bias_copy_k<<<blocks((size_t) n * cols, 256), 256, 0, s>>>(g, bias, y, (size_t) n * cols, cols);
}

void launch_embed_finish_interp(float* x, const float* bias, const float* pos_table, int n, int gx, int gy, int table_edge,
                                cudaStream_t s) {
    const float isx = gx == table_edge ? 0.0f : (float) (1.0 / (((double) gx + 0.1) / table_edge)),
                isy = gy == table_edge ? 0.0f : (float) (1.0 / (((double) gy + 0.1) / table_edge));
    embed_finish_interp_k<<<n, 128, 0, s>>>(x, bias, pos_table, gx, table_edge, isx, isy);
}
void launch_qkv_plain(const float* g, const float* bias, __half* q, __half* k, __half* v, int n, cudaStream_t s) {
    qkv_plain_k<<<n, 256, 0, s>>>(g, bias, q, k, v, n);
}
void launch_rmsnorm(const float* x, const float* w, __nv_bfloat16* y, int n, float eps, cudaStream_t s) {
    rmsnorm_k<<<blocks(n, 4), 128, 0, s>>>(x, w, y, n, eps);
}
void launch_bias_gelu_erf(const float* g, const float* bias, __nv_bfloat16* y, int n, int cols, cudaStream_t s) {
    bias_gelu_erf_k<<<blocks((size_t) n * cols, 256), 256, 0, s>>>(g, bias, y, (size_t) n * cols, cols);
}
void launch_bias_copy_lines(const float* g, const float* bias, const float* newline, float* y, int m, int cols, int ow,
                            cudaStream_t s) {
    const size_t rows = (size_t) m + (size_t) (m / ow);
    bias_copy_lines_k<<<blocks(rows * cols, 256), 256, 0, s>>>(g, bias, newline, y, rows * cols, cols, ow);
}

}  // namespace basalt::vision
