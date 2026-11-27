// tests/models/glm5next/kernels/kda_front_parity.cpp - the KDA front kernel against a CPU oracle that mirrors its
// operation order exactly: the conv's tap-3-first fmaf chain, the per-warp butterfly fold of the L2 sums, the
// ascending warp sum, and the bounded sigmoid decay. Any remaining difference is CUDA libm vs host libm ULP noise,
// checked to be tiny. One token, 4 heads of 128 channels (the model's head geometry).
#include "models/glm5next/kernels/kda_front.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

struct Lcg {
    unsigned long long s = 88172645463325252ull;
    float operator()() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long r = s >> 11;
        return (float) (double) r / 9007199254740992.0f * 2.0f - 1.0f;
    }
};

// the butterfly-shuffle-down fold, emulated exactly: each level adds the partner o lanes up
float butterfly_fold(const float* part, int lane0) {  // part[lane0 + o] chain for the lane-0 value
    float v[32];
    for (int i = 0; i < 32; ++i) v[i] = part[lane0 + i];
    for (int o = 16; o > 0; o >>= 1)
        for (int i = 0; i < o; ++i) v[i] = v[i] + v[i + o];
    return v[0];
}

float* d_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        std::printf("FAIL: cudaMalloc\n");
        std::exit(1);
    }
    return (float*) p;
}

void copy_h2d(float* dst, const std::vector<float>& src) { cudaMemcpy(dst, src.data(), src.size() * 4, cudaMemcpyHostToDevice); }

}  // namespace

int main() {
    int n_gpus = 0;
    if (cudaGetDeviceCount(&n_gpus) != cudaSuccess) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    bool have = false;
    for (int d = 0; d < n_gpus; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
        if (prop.major == 12 && cudaSetDevice(d) == cudaSuccess) { have = true; break; }
    }
    if (!have) {
        std::printf("SKIP: no sm_120 device\n");
        return 77;
    }

    constexpr int kHeads = 4, kHc = 128;
    const size_t n = (size_t) kHeads * kHc;  // channels per stream
    Lcg rng;
    std::vector<float> x(3 * n), hist(9 * n), conv(12 * n), a(n), dt(n), a_neg(kHeads), b_raw(kHeads);
    for (auto& v : x) v = rng();
    for (auto& v : hist) v = rng();
    for (auto& v : conv) v = 0.5f * rng();
    for (auto& v : a) v = rng();
    for (auto& v : dt) v = 0.3f * rng();
    for (auto& v : a_neg) v = -std::exp(0.5f * rng());
    for (auto& v : b_raw) v = rng();

    std::vector<float> qn_gpu(n), kn_gpu(n), v_gpu(n), g_gpu(n), beta_gpu(kHeads);
    std::vector<float> qn_cpu(n), kn_cpu(n), v_cpu(n), g_cpu(n), beta_cpu(kHeads);

    float *d_x = d_alloc(x.size() * 4), *d_hist = d_alloc(hist.size() * 4), *d_conv = d_alloc(conv.size() * 4),
          *d_a = d_alloc(n * 4), *d_dt = d_alloc(n * 4), *d_aneg = d_alloc(kHeads * 4), *d_braw = d_alloc(kHeads * 4),
          *d_qn = d_alloc(n * 4), *d_kn = d_alloc(n * 4), *d_v = d_alloc(n * 4), *d_g = d_alloc(n * 4),
          *d_beta = d_alloc(kHeads * 4);
    copy_h2d(d_x, x); copy_h2d(d_hist, hist); copy_h2d(d_conv, conv); copy_h2d(d_a, a); copy_h2d(d_dt, dt);
    copy_h2d(d_aneg, a_neg); copy_h2d(d_braw, b_raw);

    basalt::models::glm5next::kda_front(d_x, n, d_hist, d_conv, d_a, d_dt, d_aneg, d_braw, d_qn, d_kn, d_v, d_g, d_beta,
                                        kHeads, kHc, 0);
    check(cudaGetLastError() == cudaSuccess, "front launches");
    check(cudaDeviceSynchronize() == cudaSuccess, "front runs");
    cudaMemcpy(qn_gpu.data(), d_qn, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(kn_gpu.data(), d_kn, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(v_gpu.data(), d_v, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_gpu.data(), d_g, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(beta_gpu.data(), d_beta, kHeads * 4, cudaMemcpyDeviceToHost);

    // the oracle
    const float qscale = 1.0f / std::sqrt((float) kHc);
    for (int h = 0; h < kHeads; ++h) {
        float sq[32] = {}, sk[32] = {};  // the per-warp folded squares
        std::vector<float> y0(kHc), y1(kHc), y2(kHc);
        for (int ch = 0; ch < kHc; ++ch) {
            const int c = h * kHc + ch;
            const size_t off[3] = {(size_t) 0 * n + c, (size_t) 1 * n + c, (size_t) 2 * n + c};
            for (int s = 0; s < 3; ++s) {
                const float* wv = conv.data() + (size_t) s * 4 * n;
                const float* hs = hist.data() + (size_t) s * n;
                float acc = wv[3 * n + c] * x[off[s]];
                acc = fmaf(wv[2 * n + c], hs[2 * 3 * n + c], acc);
                acc = fmaf(wv[1 * n + c], hs[1 * 3 * n + c], acc);
                acc = fmaf(wv[0 * n + c], hs[0 * 3 * n + c], acc);
                const float y = acc / (1.0f + std::expf(-acc));
                (s == 0 ? y0 : s == 1 ? y1 : y2)[ch] = y;
            }
        }
        // lane-partials in butterfly order: the kernel folds each warp's 32 channel squares
        float part_q[32], part_k[32];
        for (int ch = 0; ch < 32; ++ch) { part_q[ch] = 0.0f; part_k[ch] = 0.0f; }
        for (int w = 0; w < kHc / 32; ++w) {
            for (int l = 0; l < 32; ++l) { part_q[l] = y0[w * 32 + l] * y0[w * 32 + l]; part_k[l] = y1[w * 32 + l] * y1[w * 32 + l]; }
            sq[w] = butterfly_fold(part_q, 0);
            sk[w] = butterfly_fold(part_k, 0);
        }
        float aq = 0.0f, ak = 0.0f;
        for (int w = 0; w < kHc / 32; ++w) { aq += sq[w]; ak += sk[w]; }
        const float norm_q = 1.0f / std::sqrt(aq + basalt::models::glm5next::kL2Eps),
                    norm_k = 1.0f / std::sqrt(ak + basalt::models::glm5next::kL2Eps);
        for (int ch = 0; ch < kHc; ++ch) {
            const int c = h * kHc + ch;
            qn_cpu[c] = y0[ch] * norm_q * qscale;
            kn_cpu[c] = y1[ch] * norm_k;
            v_cpu[c] = y2[ch];
            const float gr = basalt::models::glm5next::kGateLowerBound /
                             (1.0f + std::expf(a_neg[h] * (a[c] + dt[c])));  // lb * sigmoid(-(a_neg * S))
            g_cpu[c] = std::expf(gr);
        }
        beta_cpu[h] = 1.0f / (1.0f + std::exp(-b_raw[h]));
    }

    float worst = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        worst = std::fmax(worst, std::fabs(qn_gpu[i] - qn_cpu[i]));
        worst = std::fmax(worst, std::fabs(kn_gpu[i] - kn_cpu[i]));
        worst = std::fmax(worst, std::fabs(v_gpu[i] - v_cpu[i]));
        worst = std::fmax(worst, std::fabs(g_gpu[i] - g_cpu[i]));
    }
    for (int h = 0; h < kHeads; ++h) worst = std::fmax(worst, std::fabs(beta_gpu[h] - beta_cpu[h]));
    std::printf("max |gpu - oracle| = %.3e\n", worst);
    if (worst >= 1e-4f) {
        float m_q = 0, m_k = 0, m_v = 0, m_g = 0, m_b = 0;
        int iq = -1;
        for (size_t i = 0; i < n; ++i) {
            float d;
            d = std::fabs(qn_gpu[i] - qn_cpu[i]); if (d > m_q) { m_q = d; iq = (int) i; }
            m_k = std::fmax(m_k, std::fabs(kn_gpu[i] - kn_cpu[i]));
            m_v = std::fmax(m_v, std::fabs(v_gpu[i] - v_cpu[i]));
            m_g = std::fmax(m_g, std::fabs(g_gpu[i] - g_cpu[i]));
        }
        for (int h = 0; h < kHeads; ++h) m_b = std::fmax(m_b, std::fabs(beta_gpu[h] - beta_cpu[h]));
        std::printf("max q=%.3e (i=%d head %d) k=%.3e v=%.3e g=%.3e beta=%.3e\n", m_q, iq, iq / kHc, m_k, m_v, m_g, m_b);
        std::printf("qn_gpu[i]=%.3e qn_cpu=%.3e  kn_gpu head-wise norms:\n", qn_gpu[iq], qn_cpu[iq]);
        for (int h = 0; h < kHeads; ++h)
            std::printf("  head %d qn[0]=%.6f kn[0]=%.6f v[0]=%.6f g[0]=%.6f\n", h, qn_gpu[h * kHc], kn_gpu[h * kHc],
                        v_gpu[h * kHc], g_gpu[h * kHc]);
    }
    check(worst < 1e-4f, "front matches the oracle within 1e-4");
    check(std::isfinite(worst), "no NaN/Inf");

    // The forward pass hands the front one [T][D] slab per stream, so the streams are not side by side. The same
    // numbers, laid twice as far apart with poison in the gaps, must come out unchanged: the stride is honored.
    const size_t stride = 2 * n;
    std::vector<float> x2(3 * stride, 1e30f);
    for (size_t s = 0; s < 3; ++s)
        for (size_t i = 0; i < n; ++i) x2[s * stride + i] = x[s * n + i];
    float* d_x2 = d_alloc(x2.size() * 4);
    copy_h2d(d_x2, x2);
    basalt::models::glm5next::kda_front(d_x2, stride, d_hist, d_conv, d_a, d_dt, d_aneg, d_braw, d_qn, d_kn, d_v,
                                        d_g, d_beta, kHeads, kHc, 0);
    check(cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess, "strided front runs");
    std::vector<float> qn2(n), kn2(n), v2(n), g2(n);
    cudaMemcpy(qn2.data(), d_qn, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(kn2.data(), d_kn, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(v2.data(), d_v, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(g2.data(), d_g, n * 4, cudaMemcpyDeviceToHost);
    float worst2 = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        worst2 = std::fmax(worst2, std::fabs(qn2[i] - qn_gpu[i]));
        worst2 = std::fmax(worst2, std::fabs(kn2[i] - kn_gpu[i]));
        worst2 = std::fmax(worst2, std::fabs(v2[i] - v_gpu[i]));
        worst2 = std::fmax(worst2, std::fabs(g2[i] - g_gpu[i]));
    }
    std::printf("max |strided - contiguous| = %.3e\n", worst2);
    check(worst2 == 0.0f, "a strided layout gives the same answer");
    cudaFree(d_x2);

    cudaFree(d_x); cudaFree(d_hist); cudaFree(d_conv); cudaFree(d_a); cudaFree(d_dt); cudaFree(d_aneg); cudaFree(d_braw);
    cudaFree(d_qn); cudaFree(d_kn); cudaFree(d_v); cudaFree(d_g); cudaFree(d_beta);
    if (g_failures != 0) {
        std::printf("KDA FRONT PARITY: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("KDA FRONT PARITY: ok\n");
    return 0;
}
