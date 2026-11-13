// tests/kernels/dense_ops_parity.cpp - the dense stack's element-wise kernels (basalt/kernels/dense_ops.hpp) and the block
// drafter's windowed attention (gqa_window_attn) against CPU references in double precision, at the Qwen3.8-27B widths
// (5120 hidden, 17408 FFN) and its DFlash 2 drafter's (32 x 128 query heads, 8 KV heads, a 2048-cell window, groups of
// 16 channels, 16 candidates of rank 256).  Prints max abs / rel error per kernel.  Exit 77 without a GPU.
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/logprobs.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

std::mt19937 rng(11);
std::normal_distribution<float> nd(0.f, 1.f);
cudaStream_t S;
int fails = 0;

template <class T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaStreamSynchronize(S);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
std::vector<float> randn(size_t n, float sd = 1.f) {
    std::vector<float> v(n);
    for (auto& x : v) x = sd * nd(rng);
    return v;
}
void report(const char* what, const std::vector<float>& got, const std::vector<double>& want, double tol) {
    double a = 0, r = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double e = std::fabs(got[i] - want[i]);
        a = std::max(a, e);
        r = std::max(r, e / std::max(std::fabs(want[i]), 1e-3));
    }
    std::printf("%-22s max abs %.3g rel %.3g%s\n", what, a, r, a > tol ? "  FAIL" : "");
    if (a > tol) ++fails;
}
uint16_t bf(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float fb(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) return 77;
    cudaStreamCreate(&S);
    const int H = 5120, FF = 17408, T = 8;
    {   // rms_norm_rows into a strided output
        auto x = randn((size_t) T * H), w = randn(H, 0.2f);
        for (auto& v : w) v += 1.0f;
        float *dx = up(x), *dw = up(w), *dy = nullptr;
        cudaMalloc(&dy, (size_t) T * 2 * H * 4);
        basalt::kernels::rms_norm_rows(dx, dw, dy, T, H, 1e-6f, S, 2 * H);
        auto y = down(dy, (size_t) T * 2 * H);
        std::vector<float> got;
        std::vector<double> want;
        for (int t = 0; t < T; ++t) {
            double ss = 0;
            for (int c = 0; c < H; ++c) ss += (double) x[(size_t) t * H + c] * x[(size_t) t * H + c];
            const double inv = 1.0 / std::sqrt(ss / H + 1e-6);
            for (int c = 0; c < H; ++c) {
                got.push_back(y[(size_t) t * 2 * H + c]);
                want.push_back(x[(size_t) t * H + c] * inv * w[c]);
            }
        }
        report("rms_norm_rows", got, want, 1e-4);
    }
    {   // swiglu_scaled with NVFP4 tensor scales
        auto g = randn((size_t) T * FF, 3.f), u = randn((size_t) T * FF, 3.f);
        std::vector<float> sg = {0.37f}, su = {1.9f};
        float *dg = up(g), *du = up(u), *dsg = up(sg), *dsu = up(su), *dh = nullptr;
        cudaMalloc(&dh, (size_t) T * FF * 4);
        basalt::kernels::swiglu_scaled(dg, du, dsg, dsu, dh, T, FF, FF, S);
        auto h = down(dh, (size_t) T * FF);
        std::vector<double> want(h.size());
        for (size_t i = 0; i < h.size(); ++i) {
            const double a = (double) g[i] * sg[0], b = (double) u[i] * su[0];
            want[i] = a / (1 + std::exp(-a)) * b;
        }
        report("swiglu_scaled", h, want, 1e-4);
    }
    {   // bf16_rows_dot (the GDN gates' and the selector's projections)
        const int N = 96;
        auto x = randn((size_t) T * H);
        std::vector<uint16_t> w((size_t) N * H);
        for (auto& v : w) v = bf(0.02f * nd(rng));
        float* dx = up(x);
        uint16_t* dw = up(w);
        float* dy = nullptr;
        cudaMalloc(&dy, (size_t) T * N * 4);
        basalt::kernels::bf16_rows_dot(dx, dw, dy, T, N, H, S);
        auto y = down(dy, (size_t) T * N);
        std::vector<double> want(y.size());
        for (int t = 0; t < T; ++t)
            for (int n = 0; n < N; ++n) {
                double a = 0;
                for (int k = 0; k < H; ++k) a += (double) x[(size_t) t * H + k] * fb(w[(size_t) n * H + k]);
                want[(size_t) t * N + n] = a;
            }
        report("bf16_rows_dot", y, want, 1e-4);
    }
    {   // dyn_conv2: base + dynamic two-tap convolution along the block, 16-channel groups
        const int G = 16, groups = H / G, LD = 4 * groups;
        auto x = randn((size_t) T * H), dyn = randn((size_t) T * LD, 0.3f), base = randn(2 * (size_t) H, 0.5f);
        float *dx = up(x), *dd = up(dyn), *db = up(base), *dy = nullptr;
        cudaMalloc(&dy, (size_t) T * H * 4);
        basalt::kernels::dyn_conv2(dx, dd + 2 * groups, LD, db, dy, T, H, G, S);   // the second half, as `finish` reads
        auto y = down(dy, (size_t) T * H);
        std::vector<double> want(y.size());
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < H; ++c) {
                const float* dr = &dyn[(size_t) t * LD + 2 * groups];
                double a = ((double) base[c] + dr[c / G]) * x[(size_t) t * H + c];
                if (t > 0) a += ((double) base[H + c] + dr[groups + c / G]) * x[(size_t) (t - 1) * H + c];
                want[(size_t) t * H + c] = a;
            }
        report("dyn_conv2", y, want, 1e-5);
    }
    {   // candidate_path against the same walk on the CPU
        const int V = 4096, R = 256, K = 16, P = 7;
        std::vector<uint16_t> pred((size_t) V * R), succ((size_t) V * R);
        for (auto& v : pred) v = bf(0.3f * nd(rng));
        for (auto& v : succ) v = bf(0.3f * nd(rng));
        auto h = randn((size_t) P * R, 0.3f);
        std::vector<float> lp((size_t) P * basalt::kernels::kLogprobStride, 0.f);
        for (int p = 0; p < P; ++p)
            for (int k = 0; k < K; ++k) {
                int32_t id = (int32_t) (rng() % V);
                std::memcpy(&lp[(size_t) p * basalt::kernels::kLogprobStride + 2 + 2 * k], &id, 4);
                lp[(size_t) p * basalt::kernels::kLogprobStride + 3 + 2 * k] = -0.3f * k + 0.1f * nd(rng);
            }
        const int32_t anchor = 17;
        float *dlp = up(lp), *dh = up(h), *dconf = nullptr;
        uint16_t *dp = up(pred), *ds = up(succ);
        int32_t *da = up(std::vector<int32_t>{anchor}), *dids = nullptr;
        cudaMalloc(&dids, P * 4);
        cudaMalloc(&dconf, P * 4);
        basalt::kernels::candidate_path(dlp, basalt::kernels::kLogprobStride, K, dh, dp, ds, R, da, P, dids, dconf, S);
        auto ids = down(dids, P);
        auto conf = down(dconf, P);
        int32_t prev = anchor;
        int differ = 0;
        std::vector<float> gc;
        std::vector<double> wc;
        for (int p = 0; p < P; ++p) {
            double best = -1e300, z = 0;
            int bk = 0;
            std::vector<double> sc(K);
            for (int k = 0; k < K; ++k) {
                int32_t c;
                std::memcpy(&c, &lp[(size_t) p * basalt::kernels::kLogprobStride + 2 + 2 * k], 4);
                double d = 0;
                for (int r = 0; r < R; ++r) d += (double) fb(pred[(size_t) prev * R + r]) * h[(size_t) p * R + r] * fb(succ[(size_t) c * R + r]);
                sc[(size_t) k] = lp[(size_t) p * basalt::kernels::kLogprobStride + 3 + 2 * k] + d;
                if (sc[(size_t) k] > best) { best = sc[(size_t) k]; bk = k; }
            }
            for (double v : sc) z += std::exp(v - best);
            std::memcpy(&prev, &lp[(size_t) p * basalt::kernels::kLogprobStride + 2 + 2 * bk], 4);
            differ += prev != ids[(size_t) p];
            prev = ids[(size_t) p];   // follow the GPU's path so one tie does not cascade
            gc.push_back(conf[(size_t) p]);
            wc.push_back(1.0 / z);
        }
        std::printf("candidate_path          picks differing from the CPU walk: %d of %d\n", differ, P);
        if (differ) ++fails;
        report("candidate_path conf", gc, wc, 1e-4);
    }
    {   // gqa_window_attn at the drafter's shape: 8 non-causal rows over a 2048-cell window in a cyclic cache
        const int NH = 32, NKV = 8, HD = 128, W = 2048, CAP = W + 256, B = 8;
        const int64_t pos = 5000;   // rows at 5000..5007; the cache wraps
        std::vector<uint16_t> kc((size_t) CAP * NKV * HD), vc(kc.size());
        for (auto& v : kc) v = basalt::kernels::f16_from_f32(nd(rng));
        for (auto& v : vc) v = basalt::kernels::f16_from_f32(nd(rng));
        auto q = randn((size_t) B * NH * HD);
        float* dq = up(q);
        uint16_t *dk = up(kc), *dv = up(vc);
        float *dout = nullptr, *dscr = nullptr;
        cudaMalloc(&dout, (size_t) B * NH * HD * 4);
        cudaMalloc(&dscr, basalt::kernels::gqa_decode_scratch_floats(CAP, B, NH, HD) * 4);
        basalt::kernels::gqa_window_attn(dq, dk, dv, NH, NKV, HD, pos, B, pos + B, W, CAP, dscr, CAP, dout, S);
        auto out = down(dout, (size_t) B * NH * HD);
        std::vector<double> want(out.size());
        for (int t = 0; t < B; ++t)
            for (int h = 0; h < NH; ++h) {
                const int kvh = h / (NH / NKV);
                const int64_t lo = pos + t - W + 1, hi = pos + B;
                std::vector<double> s;
                double mx = -1e300;
                for (int64_t p = lo; p < hi; ++p) {
                    double d = 0;
                    for (int i = 0; i < HD; ++i)
                        d += (double) q[((size_t) t * NH + h) * HD + i] * basalt::kernels::f32_from_f16(kc[(((size_t) (p % CAP)) * NKV + kvh) * HD + i]);
                    s.push_back(d / std::sqrt((double) HD));
                    mx = std::max(mx, s.back());
                }
                double l = 0;
                for (auto& v : s) { v = std::exp(v - mx); l += v; }
                for (int i = 0; i < HD; ++i) {
                    double a = 0;
                    for (size_t j = 0; j < s.size(); ++j)
                        a += s[j] * basalt::kernels::f32_from_f16(vc[(((size_t) ((lo + (int64_t) j) % CAP)) * NKV + kvh) * HD + i]);
                    want[((size_t) t * NH + h) * HD + i] = a / l;
                }
            }
        report("gqa_window_attn", out, want, 2e-4);
    }
    {   // gdn_rows_scatter: a batched window's packed GDN rows to three sessions' stores, bitwise
        const int C = 10240, HV = 48, W = 8, NL = 3, layer = 2, rows = 7;
        auto qkv = randn((size_t) rows * C), h = randn((size_t) rows * C), gate = randn((size_t) rows * HV), beta = randn((size_t) rows * HV);
        float *dq = up(qkv), *dh = up(h), *dg = up(gate), *db = up(beta);
        const int seg_of[7] = {0, 0, 0, 1, 1, 2, 2}, j_of[7] = {0, 1, 2, 0, 1, 0, 1};
        float *sq[3], *sh[3], *sg[3], *sb[3];
        for (int k = 0; k < 3; ++k) {
            cudaMalloc(&sq[k], (size_t) NL * W * C * 4); cudaMalloc(&sh[k], (size_t) NL * W * C * 4);
            cudaMalloc(&sg[k], (size_t) NL * W * HV * 4); cudaMalloc(&sb[k], (size_t) NL * W * HV * 4);
            cudaMemset(sq[k], 0, (size_t) NL * W * C * 4); cudaMemset(sh[k], 0, (size_t) NL * W * C * 4);
            cudaMemset(sg[k], 0, (size_t) NL * W * HV * 4); cudaMemset(sb[k], 0, (size_t) NL * W * HV * 4);
        }
        std::vector<basalt::kernels::GdnRowDst> tab(rows);
        for (int r = 0; r < rows; ++r) {
            const int k = seg_of[r], j = j_of[r];
            tab[r] = {sq[k] + (size_t) j * C, sh[k] + (size_t) j * C, sg[k] + (size_t) j * HV, sb[k] + (size_t) j * HV};
        }
        auto* dt = up(tab);
        basalt::kernels::gdn_rows_scatter(dt, rows, dq, dh, dg, db, C, HV, layer, W, S);
        bool same = true;
        for (int r = 0; r < rows; ++r) {
            const int k = seg_of[r], j = j_of[r];
            auto gq = down(sq[k] + ((size_t) layer * W + j) * C, C), gh = down(sh[k] + ((size_t) layer * W + j) * C, C);
            auto gg = down(sg[k] + ((size_t) layer * W + j) * HV, HV), gb = down(sb[k] + ((size_t) layer * W + j) * HV, HV);
            same = same && !std::memcmp(gq.data(), &qkv[(size_t) r * C], C * 4) && !std::memcmp(gh.data(), &h[(size_t) r * C], C * 4) &&
                   !std::memcmp(gg.data(), &gate[(size_t) r * HV], HV * 4) && !std::memcmp(gb.data(), &beta[(size_t) r * HV], HV * 4);
        }
        // layers other than the one named are untouched
        auto other = down(sq[0], (size_t) layer * W * C);
        for (float v : other) same = same && v == 0.0f;
        std::printf("%-22s %s\n", "gdn_rows_scatter", same ? "bitwise" : "MISMATCH  FAIL");
        if (!same) ++fails;
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
