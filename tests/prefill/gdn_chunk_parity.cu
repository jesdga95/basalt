// tests/prefill/gdn_chunk_parity.cu - the prompt path's chunked DeltaNet recurrence (gdn_recurrence,
// src/prefill/gdn_chunked.cu) against the token-serial FP32 kernel it replaced and a double-precision CPU reference.
//
// Synthetic inputs of the model's shape (16 key heads, 48 value heads, head size 128, q/k not yet normalized), a
// non-zero state as after an earlier chunk.  Checked: the output before the norm and the state after the chunk, for
// every value head against the old kernel and for three heads against the CPU reference; and the split check (the
// same tokens in two calls, the second starting off the 16-token grid, against one call).
// Gates come in two mixes: the regular one (decay per token in e^-0.82..e^-0.02) and an extreme one (a tenth of the value
// heads at g in [-30, -5], a tenth at g = -1e-6), at T up to 32768 (a whole production chunk in one call); every output
// and state must be finite.
//   gdn_chunk_parity [seed]          the checks
//   gdn_chunk_parity --bench         and the timings at T = 4096, 16384 and 32768 (one layer), old and new alternating
#include "basalt/prefill/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 10240;
constexpr float EPS = 1e-6f;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ------------------------------------------------------------------ the kernel before (verbatim)
__device__ __forceinline__ void gdn_cp4(float* smem, const float* gmem) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void gdn_cp16(float* smem, const float* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void gdn_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void gdn_cp_wait_prev() { asm volatile("cp.async.wait_group 1;\n" ::); }
constexpr int VPK = HV / HK;
constexpr int GDN_WC = 16, GDN_WNCB = S / GDN_WC, GDN_WTB = 16;
__global__ void __launch_bounds__(GDN_WC * 8) gdn_rec_wl_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                const float* __restrict__ gate,
                                                                const float* __restrict__ beta,
                                                                float* __restrict__ oc_out, int64_t T) {
    constexpr int TB = GDN_WTB, NT = GDN_WC * 8, QKP = S / 4, VP = GDN_WC / 4;   // threads, 16-byte pieces
    __shared__ __align__(16) float sq[2][TB][S];
    __shared__ __align__(16) float sk[2][TB][S];
    __shared__ __align__(16) float sv[2][TB][VPK][GDN_WC];
    __shared__ float sg[2][TB][VPK], sb[2][TB][VPK];
    const int qh = blockIdx.x / GDN_WNCB, cb = blockIdx.x % GDN_WNCB;
    const int tid = threadIdx.x, lane = tid & 31, rg = lane & 7, c = 4 * (tid >> 5) + (lane >> 3);
    const int col = cb * GDN_WC + c;
    float s[VPK][16];   // [value head][4 m + e]: row 4 rg + 32 m + e
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int j = 0; j < VPK; ++j)
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                s[j][4 * m + e] = state[(size_t) (4 * rg + 32 * m + e) * rs + (size_t) (qh + j * HK) * S + col];
    const int64_t nblk = (T + TB - 1) / TB;
    auto stage = [&](int64_t k) {   // tokens [k * TB, k * TB + TB) into buffer k & 1
        const int bb = (int) (k & 1);
        const int64_t t0 = k * TB;
        for (int p = tid; p < TB * 2 * QKP; p += NT) {
            const int i = p / (2 * QKP), w = p % (2 * QKP), isk = w / QKP, jj = (w % QKP) * 4;
            if (t0 + i < T)
                gdn_cp16(isk ? &sk[bb][i][jj] : &sq[bb][i][jj], h + (t0 + i) * C + (isk ? HK * S : 0) + qh * S + jj);
        }
        for (int p = tid; p < TB * VPK * VP; p += NT) {
            const int i = p / (VPK * VP), w = p % (VPK * VP), j = w / VP, jj = (w % VP) * 4;
            if (t0 + i < T)
                gdn_cp16(&sv[bb][i][j][jj], h + (t0 + i) * C + 2 * HK * S + (qh + j * HK) * S + cb * GDN_WC + jj);
        }
        for (int p = tid; p < 2 * TB * VPK; p += NT) {
            const int isb = p / (TB * VPK), w = p % (TB * VPK), i = w / VPK, j = w % VPK;
            if (t0 + i < T)
                gdn_cp4(isb ? &sb[bb][i][j] : &sg[bb][i][j], (isb ? beta : gate) + (t0 + i) * HV + qh + j * HK);
        }
    };
    if (nblk > 0) stage(0);
    gdn_cp_commit();
    for (int64_t k = 0; k < nblk; ++k) {
        __syncthreads();   // every warp is done with block k - 1, the buffer stage(k + 1) fills
        if (k + 1 < nblk) stage(k + 1);
        gdn_cp_commit();
        gdn_cp_wait_prev();
        __syncthreads();
        const int bb = (int) (k & 1);
        const int n = (int) ((T - k * TB) < TB ? (T - k * TB) : TB);
#pragma unroll 2
        for (int i = 0; i < n; ++i) {
            const int64_t t = k * TB + i;
            const float4* k4 = reinterpret_cast<const float4*>(&sk[bb][i][4 * rg]);
            const float4* q4 = reinterpret_cast<const float4*>(&sq[bb][i][4 * rg]);
            float kc[16], qc[16];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const float4 a = k4[8 * m], b = q4[8 * m];
                kc[4 * m] = a.x; kc[4 * m + 1] = a.y; kc[4 * m + 2] = a.z; kc[4 * m + 3] = a.w;
                qc[4 * m] = b.x; qc[4 * m + 1] = b.y; qc[4 * m + 2] = b.z; qc[4 * m + 3] = b.w;
            }
            float g[VPK], kv[VPK], delta[VPK], o[VPK];
#pragma unroll
            for (int j = 0; j < VPK; ++j) { g[j] = __expf(sg[bb][i][j]); kv[j] = 0.0f; o[j] = 0.0f; }
#pragma unroll
            for (int r = 0; r < 16; ++r)
#pragma unroll
                for (int j = 0; j < VPK; ++j) kv[j] = fmaf(s[j][r], kc[r], kv[j]);
#pragma unroll
            for (int x = 1; x < 8; x <<= 1)
#pragma unroll
                for (int j = 0; j < VPK; ++j) kv[j] += __shfl_xor_sync(0xffffffffu, kv[j], x);
#pragma unroll
            for (int j = 0; j < VPK; ++j) delta[j] = (sv[bb][i][j][c] - g[j] * kv[j]) * sb[bb][i][j];
#pragma unroll
            for (int r = 0; r < 16; ++r)
#pragma unroll
                for (int j = 0; j < VPK; ++j) {
                    s[j][r] = fmaf(g[j], s[j][r], kc[r] * delta[j]);
                    o[j] = fmaf(s[j][r], qc[r], o[j]);
                }
#pragma unroll
            for (int x = 1; x < 8; x <<= 1)
#pragma unroll
                for (int j = 0; j < VPK; ++j) o[j] += __shfl_xor_sync(0xffffffffu, o[j], x);
            if (rg < VPK)   // row group j writes value head j's output
                oc_out[t * HV * S + (qh + rg * HK) * S + col] =
                    (rg == 0 ? o[0] : rg == 1 ? o[1] : o[2]) * rsqrtf((float) S);
        }
    }
#pragma unroll
    for (int j = 0; j < VPK; ++j)
#pragma unroll
        for (int m = 0; m < 4; ++m)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                state[(size_t) (4 * rg + 32 * m + e) * rs + (size_t) (qh + j * HK) * S + col] = s[j][4 * m + e];
}
__global__ void gdn_l2_kernel(float* __restrict__ h, float eps) {
    const int64_t t = blockIdx.y;
    const int head = blockIdx.x;
    float* x = h + t * C + head * S;
    const float v = x[threadIdx.x];
    float sq = warp_sum(v * v);
    __shared__ float part[4];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    x[threadIdx.x] = v * rsqrtf(ss + eps);
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__global__ void __launch_bounds__(S) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                         float eps, const float* __restrict__ y,
                                                         uint16_t* __restrict__ y16) {
    __shared__ float wsum[4];
    const int64_t t = blockIdx.x;
    const int head = blockIdx.y, col = threadIdx.x;
    const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
    const float oc = y[at];
    float sp = warp_sum(oc * oc);
    if ((col & 31) == 0) wsum[col >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[col] * sigm(z[t * HV * S + head * S + col]);
    y16[at] = __half_as_ushort(__float2half_rn(v));
}

// ------------------------------------------------------------------ synthetic inputs
__device__ __forceinline__ uint32_t mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return (uint32_t) x;
}
__device__ __forceinline__ float uni(uint64_t key) { return ((mix(key) >> 8) + 0.5f) * (1.0f / 16777216.0f); }
__device__ __forceinline__ float gauss(uint64_t key) {
    const float u1 = uni(key * 2 + 1), u2 = uni(key * 2 + 2);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
__global__ void fill_gauss(float* x, int64_t n, float scale, uint64_t seed, float offset = 0.0f) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] = offset + scale * gauss(seed * 0x9E3779B97F4A7C15ULL + (uint64_t) i);
}
// gate = log of the decay, beta in (0, 1).  Mode 0: mostly near 1, some strong.  Mode 1 (extreme): value heads
// 0, 10, 20, ... decay hard (g in [-30, -5]: the decay of a few tokens underflows a prefix product) and heads 1, 11, 21, ...
// barely at all (g = -1e-6), the rest as mode 0.
__global__ void fill_gates(float* gate, float* beta, int64_t n, uint64_t seed, int mode) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float u = uni(seed * 0x9E3779B97F4A7C15ULL + 7 * (uint64_t) i);
        const int head = (int) (i % HV);
        gate[i] = -0.02f - 0.8f * u * u * u;
        if (mode == 1 && head % 10 == 0) gate[i] = -5.0f - 25.0f * u;
        if (mode == 1 && head % 10 == 1) gate[i] = -1e-6f;
        beta[i] = 1.0f / (1.0f + expf(-gauss(seed * 0xD1B54A32D192ED03ULL + (uint64_t) i)));
    }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 256), "cudaMalloc");
    return p;
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

struct Inputs {
    int64_t T = 0;
    float *h = nullptr, *hn = nullptr, *gate = nullptr, *beta = nullptr, *z = nullptr, *gamma = nullptr, *state0 = nullptr;
    Inputs(int64_t t, uint64_t seed, int mode = 0) : T(t) {
        h = dalloc<float>((size_t) T * C);
        hn = dalloc<float>((size_t) T * C);
        gate = dalloc<float>((size_t) T * HV);
        beta = dalloc<float>((size_t) T * HV);
        z = dalloc<float>((size_t) T * HV * S);
        gamma = dalloc<float>(S);
        state0 = dalloc<float>((size_t) S * HV * S);
        fill_gauss<<<4096, 256>>>(h, T * C, 1.0f, seed);
        ck(cudaMemcpy(hn, h, (size_t) T * C * 4, cudaMemcpyDeviceToDevice), "copy");
        gdn_l2_kernel<<<dim3(2 * HK, (unsigned) T), S>>>(hn, EPS);   // what the old kernel read
        fill_gates<<<1024, 256>>>(gate, beta, T * HV, seed + 1, mode);
        fill_gauss<<<4096, 256>>>(z, T * HV * S, 1.0f, seed + 2);
        fill_gauss<<<1, 128>>>(gamma, S, 0.1f, seed + 3, 1.0f);
        fill_gauss<<<1024, 256>>>(state0, (int64_t) S * HV * S, 0.05f, seed + 4);
        ck(cudaDeviceSynchronize(), "inputs");
    }
    ~Inputs() { cudaFree(h); cudaFree(hn); cudaFree(gate); cudaFree(beta); cudaFree(z); cudaFree(gamma); cudaFree(state0); }
};

struct Bufs {
    float *state, *y;
    uint16_t* y16;
    void* work;
    explicit Bufs(int64_t T) {
        state = dalloc<float>((size_t) S * HV * S);
        y = dalloc<float>((size_t) T * HV * S);
        y16 = dalloc<uint16_t>((size_t) T * HV * S);
        ck(cudaMalloc(&work, basalt::prefill::gdn_work_bytes(T) + 256), "work");
    }
    ~Bufs() { cudaFree(state); cudaFree(y); cudaFree(y16); cudaFree(work); }
};
void run_old(const Inputs& in, Bufs& b, int64_t T) {
    ck(cudaMemcpy(b.state, in.state0, (size_t) S * HV * S * 4, cudaMemcpyDeviceToDevice), "state");
    gdn_rec_wl_kernel<<<HK * GDN_WNCB, GDN_WC * 8>>>(b.state, in.hn, in.gate, in.beta, b.y, T);
    gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S>>>(in.z, in.gamma, EPS, b.y, b.y16);
}
void run_new(const Inputs& in, Bufs& b, int64_t T) {
    ck(cudaMemcpy(b.state, in.state0, (size_t) S * HV * S * 4, cudaMemcpyDeviceToDevice), "state");
    basalt::prefill::gdn_recurrence(b.state, in.h, in.gate, in.beta, in.z, in.gamma, EPS, b.work, b.y, b.y16, T, nullptr);
}

struct Err {
    double rms, worst;   // relative RMS, worst absolute over the largest reference magnitude
};
Err compare(const float* x, const double* r, size_t n, size_t stride = 1, size_t count = 0) {
    double d2 = 0, r2 = 0, dm = 0, rm = 0;
    if (count == 0) count = n;
    for (size_t i = 0; i < count; ++i) {
        const size_t at = i * stride;
        if (!std::isfinite(x[at])) return {NAN, NAN};
        const double d = (double) x[at] - r[i];
        d2 += d * d; r2 += r[i] * r[i];
        dm = std::max(dm, std::fabs(d)); rm = std::max(rm, std::fabs(r[i]));
    }
    return {r2 > 0 ? std::sqrt(d2 / r2) : std::sqrt(d2), rm > 0 ? dm / rm : dm};
}
Err compare_f(const std::vector<float>& x, const std::vector<float>& r) {
    std::vector<double> rd(r.begin(), r.end());
    return compare(x.data(), rd.data(), r.size());
}

// the recurrence for one value head in double: q/k normalized, S = g S + k (beta (v - g S^T k))^T, o = S^T q / sqrt(S)
void cpu_head(const std::vector<float>& h, const std::vector<float>& gate, const std::vector<float>& beta,
              const std::vector<float>& st0, int head, int64_t T, std::vector<double>& out, std::vector<double>& st) {
    const int qh = head % HK;
    st.assign((size_t) S * S, 0.0);
    for (int r = 0; r < S; ++r)
        for (int c = 0; c < S; ++c) st[(size_t) r * S + c] = st0[(size_t) r * HV * S + (size_t) head * S + c];
    out.assign((size_t) T * S, 0.0);
    std::vector<double> q(S), k(S), kv(S);
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h.data() + (size_t) t * C;
        double qs = 0, ks = 0;
        for (int i = 0; i < S; ++i) { q[i] = ht[qh * S + i]; k[i] = ht[HK * S + qh * S + i]; qs += q[i] * q[i]; ks += k[i] * k[i]; }
        const double qi = 1.0 / std::sqrt(qs + EPS), ki = 1.0 / std::sqrt(ks + EPS);
        for (int i = 0; i < S; ++i) { q[i] *= qi; k[i] *= ki; }
        const double g = std::exp((double) gate[(size_t) t * HV + head]), b = beta[(size_t) t * HV + head];
        for (int c = 0; c < S; ++c) {
            double a = 0;
            for (int r = 0; r < S; ++r) a += st[(size_t) r * S + c] * k[r];
            kv[c] = a;
        }
        for (int c = 0; c < S; ++c) {
            const double d = (ht[2 * HK * S + head * S + c] - g * kv[c]) * b;
            double o = 0;
            for (int r = 0; r < S; ++r) {
                double& s = st[(size_t) r * S + c];
                s = g * s + k[r] * d;
                o += s * q[r];
            }
            out[(size_t) t * S + c] = o / std::sqrt((double) S);
        }
    }
}

// bars: the chunked form rounds q/k and the state update's Delta to BF16 (see docs/precision/index.md)
constexpr double kRms = 1e-2, kWorst = 1.5e-2;

double g_worst_seen = 0;   // the largest worst-element error any gated comparison reported

int check(int64_t T, uint64_t seed, int mode = 0) {
    Inputs in(T, seed, mode);
    const size_t st_n = (size_t) S * HV * S, y_n = (size_t) T * HV * S;
    Bufs o(T), n(T);
    run_old(in, o, T);
    run_new(in, n, T);
    ck(cudaDeviceSynchronize(), "run");
    const auto yo = down(o.y, y_n), yn = down(n.y, y_n), so = down(o.state, st_n), sn = down(n.state, st_n);
    const Err ey = compare_f(yn, yo), es = compare_f(sn, so);
    int fails = 0;
    auto verdict = [&](const Err& e) {
        const bool ok = std::isfinite(e.rms) && e.rms < kRms && e.worst < kWorst;
        if (std::isfinite(e.worst)) g_worst_seen = std::max(g_worst_seen, e.worst);
        fails += !ok;
        return ok ? "" : " FAIL";
    };
    std::printf("T=%-6lld %s gates: new vs old kernel: output rel RMS %.2e worst %.2e%s, state rel RMS %.2e worst %.2e%s\n",
                (long long) T, mode ? "extreme" : "regular", ey.rms, ey.worst, verdict(ey), es.rms, es.worst, verdict(es));
    // CPU reference for three heads (one per value-head group)
    const auto h = down(in.h, (size_t) T * C), gate = down(in.gate, (size_t) T * HV), beta = down(in.beta, (size_t) T * HV),
               st0 = down(in.state0, st_n);
    const std::vector<int> heads = mode ? std::vector<int>{0, 1, 17, 47} : std::vector<int>{0, 17, 47};
    for (int head : heads) {
        std::vector<double> ro, rs;
        cpu_head(h, gate, beta, st0, head, T, ro, rs);
        // gather this head's output and state
        std::vector<float> ynh((size_t) T * S), yoh((size_t) T * S), snh((size_t) S * S), soh((size_t) S * S);
        for (int64_t t = 0; t < T; ++t)
            for (int c = 0; c < S; ++c) {
                ynh[(size_t) t * S + c] = yn[((size_t) t * HV + head) * S + c];
                yoh[(size_t) t * S + c] = yo[((size_t) t * HV + head) * S + c];
            }
        for (int r = 0; r < S; ++r)
            for (int c = 0; c < S; ++c) {
                snh[(size_t) r * S + c] = sn[(size_t) r * HV * S + (size_t) head * S + c];
                soh[(size_t) r * S + c] = so[(size_t) r * HV * S + (size_t) head * S + c];
            }
        const Err a = compare(ynh.data(), ro.data(), ro.size()), b = compare(snh.data(), rs.data(), rs.size());
        const Err c = compare(yoh.data(), ro.data(), ro.size()), d = compare(soh.data(), rs.data(), rs.size());
        std::printf("         head %2d vs CPU double: new output %.2e/%.2e%s state %.2e/%.2e%s | old output %.2e/%.2e "
                    "state %.2e/%.2e\n", head, a.rms, a.worst, verdict(a), b.rms, b.worst, verdict(b), c.rms, c.worst,
                    d.rms, d.worst);
    }
    // split: the same tokens in two calls, the second off the 16-token grid
    if (T > 40) {
        const int64_t T1 = 37 + (T / 3 / 16) * 16;
        ck(cudaMemcpy(n.state, in.state0, st_n * 4, cudaMemcpyDeviceToDevice), "state");
        basalt::prefill::gdn_recurrence(n.state, in.h, in.gate, in.beta, in.z, in.gamma, EPS, n.work, n.y, n.y16, T1, nullptr);
        basalt::prefill::gdn_recurrence(n.state, in.h + T1 * C, in.gate + T1 * HV, in.beta + T1 * HV, in.z + T1 * HV * S,
                                        in.gamma, EPS, n.work, n.y + T1 * HV * S, n.y16 + T1 * HV * S, T - T1, nullptr);
        ck(cudaDeviceSynchronize(), "split");
        const auto ys = down(n.y, y_n), ss = down(n.state, st_n);
        const Err e1 = compare_f(ys, yn), e2 = compare_f(ss, sn);
        std::printf("         split at %lld vs one call: output rel RMS %.2e worst %.2e%s, state rel RMS %.2e worst %.2e%s\n",
                    (long long) T1, e1.rms, e1.worst, verdict(e1), e2.rms, e2.worst, verdict(e2));
    }
    return fails;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0f : v[v.size() / 2];
}
void bench(int64_t T) {
    Inputs in(T, 0xB3);
    Bufs b(T);
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    const int warm = 3, rounds = 10;
    std::vector<float> tm[2];
    for (int i = 0; i < warm + rounds; ++i)
        for (int j = 0; j < 2; ++j) {
            const int v = (i + j) % 2;
            if (v == 0) ck(cudaMemcpy(in.hn, in.h, (size_t) T * C * 4, cudaMemcpyDeviceToDevice), "copy");
            cudaEventRecord(e0);
            if (v == 0) {   // the old path: the norm pass, the recurrence, the output norm
                gdn_l2_kernel<<<dim3(2 * HK, (unsigned) T), S>>>(in.hn, EPS);
                gdn_rec_wl_kernel<<<HK * GDN_WNCB, GDN_WC * 8>>>(b.state, in.hn, in.gate, in.beta, b.y, T);
                gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S>>>(in.z, in.gamma, EPS, b.y, b.y16);
            } else {
                basalt::prefill::gdn_recurrence(b.state, in.h, in.gate, in.beta, in.z, in.gamma, EPS, b.work, b.y, b.y16, T,
                                                nullptr);
            }
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "timed");
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (i >= warm) tm[v].push_back(ms);
        }
    const float r0 = median(tm[0]), r1 = median(tm[1]);
    // the chunked form's tensor-core work per layer: 48 heads x T x (K S, Q S, K^T Delta: 3 x 2 x 128 x 128)
    const double tf = 48.0 * (double) T * 3 * 2 * 128 * 128 * 1e-12;
    std::printf("  T = %6lld one layer: old (l2 + recurrence + norm) %7.2f ms, new (prepare + chunks + norm) %7.2f ms  "
                "%.2fx; x36 layers: %.0f -> %.0f ms (%.1f TFLOP/s on the state products)\n",
                (long long) T, r0, r1, r0 / r1, 36 * r0, 36 * r1, tf / (r1 * 1e-3));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    uint64_t seed = 0x5EED;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench")) do_bench = true;
        else seed = std::strtoull(argv[i], nullptr, 0);
    }
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, 0), "props");
    std::printf("gdn_chunk_parity on %s (sm_%d%d), seed %llu\n", p.name, p.major, p.minor, (unsigned long long) seed);
    int fails = 0;
    if (do_bench) {
        for (int64_t T : {4096, 16384, 32768}) bench(T);
    } else {
        for (int64_t T : {1, 5, 16, 17, 100, 1000, 4099}) fails += check(T, seed + (uint64_t) T);
        // the decay's extremes, and a whole production chunk in one call
        for (int64_t T : {100, 1000, 4099}) fails += check(T, seed + (uint64_t) T, 1);
        fails += check(32768, seed + 32768);
        fails += check(32768, seed + 32769, 1);
        int ndev = 0;
        ck(cudaGetDeviceCount(&ndev), "devices");
        // the launch's shared-memory opt-in is per device: every other device must run it too
        for (int d = 1; d < ndev; ++d) {
            ck(cudaSetDevice(d), "set device");
            std::printf("device %d\n", d);
            fails += check(1000, seed + 1000);
        }
        std::printf("gdn_chunk_parity: %d failures (worst element seen %.2e, bar %.1e)\n", fails, g_worst_seen, kWorst);
    }
    return fails ? 1 : 0;
}
