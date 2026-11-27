// src/models/ernie55/kernels/kernels_parity.cpp - GPU parity tests for all ERNIE-5.5-Flash CUDA kernels
// against FP64 CPU references:
//   1. mHC mixes (1 / (sqrt(mean) + rms_eps)), pre, post (destination-first comb[d,s]), contract
//   2. FP32 LM-head multimax
//   3. SiTU-GLU (beta=4.0, linear_beta=25.0)
//   4. Split-feature sigmoid MoE router (512 experts, top-10, bias, renorm, per-expert scale)
//   5. VHA PostMix, sigmoid output gate, LayerNorm, orthonormal Walsh-Hadamard-128
//   6. HCA attention pipeline: q_rope, kv_store_window, 128:1 streaming compressor w/ additive APE,
//      multi-query attention w/ attn_sink, inv_rope
//   7. MLA-DSA + Lightning Indexer pipeline: dsa_qkv_rope_and_store, dsa_indexer_prepare,
//      dsa_indexer_topk_merge, dsa_mla_attention w/ Q8_0 kv_b and attn_sink
#include "basalt/models/ernie55/kernels.hpp"
#include "basalt/models/ernie55/cache_layout.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/row_norm.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace basalt::models::ernie55;

#define CUDA_CHECK(expr)                                                                                 \
    do {                                                                                                 \
        cudaError_t _e = (expr);                                                                         \
        if (_e != cudaSuccess) {                                                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
            std::exit(2);                                                                                \
        }                                                                                                \
    } while (0)

template <typename T>
struct DevBuf {
    T* ptr = nullptr;
    size_t count = 0;
    explicit DevBuf(size_t n = 0) : count(n) {
        if (n) CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~DevBuf() {
        if (ptr) cudaFree(ptr);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    void upload(const T* src, size_t n) { CUDA_CHECK(cudaMemcpy(ptr, src, n * sizeof(T), cudaMemcpyHostToDevice)); }
    void upload(const std::vector<T>& v) { upload(v.data(), v.size()); }
    void download(std::vector<T>& dst) const {
        CUDA_CHECK(cudaMemcpy(dst.data(), ptr, dst.size() * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() {
        if (count) CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }
};

std::vector<float> rand_vec(size_t n, float lo, float hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> out(n);
    for (float& v : out) v = dist(rng);
    return out;
}

double max_abs_err(const std::vector<float>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs((double) a[i] - b[i]));
    return m;
}

double rel_l2_err(const std::vector<float>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = (double) a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

int g_failures = 0;

void check_tol(const char* name, double err, double tol) {
    if (err <= tol && std::isfinite(err)) {
        std::printf("[PASS] %-36s err=%.3e (tol=%.1e)\n", name, err, tol);
    } else {
        std::printf("[FAIL] %-36s err=%.3e (tol=%.1e)\n", name, err, tol);
        ++g_failures;
    }
}

void quantize_q8_0(const std::vector<float>& src, int rows, int cols, std::vector<uint8_t>& dst) {
    const int nb = cols / 32;
    dst.resize((size_t) rows * nb * 34);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const float* x = src.data() + (size_t) r * cols + b * 32;
            float am = 0.0f;
            for (int k = 0; k < 32; ++k) am = std::max(am, std::abs(x[k]));
            const float d = am / 127.0f;
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            uint8_t* out = dst.data() + ((size_t) r * nb + b) * 34;
            const __half dh = __float2half_rn(d);
            std::memcpy(out, &dh, 2);
            int8_t* qs = reinterpret_cast<int8_t*>(out + 2);
            for (int k = 0; k < 32; ++k) {
                int q = (int) std::lrint(x[k] * inv);
                qs[k] = (int8_t) std::max(-127, std::min(127, q));
            }
        }
    }
}

std::vector<double> dequantize_q8_0(const std::vector<uint8_t>& src, int rows, int cols) {
    const int nb = cols / 32;
    std::vector<double> out((size_t) rows * cols);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const uint8_t* in = src.data() + ((size_t) r * nb + b) * 34;
            __half dh;
            std::memcpy(&dh, in, 2);
            const double d = __half2float(dh);
            const int8_t* qs = reinterpret_cast<const int8_t*>(in + 2);
            for (int k = 0; k < 32; ++k) out[(size_t) r * cols + b * 32 + k] = d * (double) qs[k];
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// 1. mHC mixes, pre, post, contract
// -----------------------------------------------------------------------------
void test_mhc() {
    const int T = 3, H = 256, flat = kHcMult * H;
    // Large rms_eps intentionally distinguishes ERNIE 1/(sqrt(mean)+eps) from rsqrt(mean+eps)
    const float rms_eps = 1e-2f, hc_eps = 1e-4f;
    const int sink_iters = 20;

    auto x = rand_vec((size_t) T * flat, -1.5f, 1.5f, 101);
    auto fn = rand_vec((size_t) kHcCoef * flat, -0.1f, 0.1f, 102);
    auto sc = std::vector<float>{0.15f, 0.25f, 0.35f};
    auto bs = rand_vec(kHcCoef, -0.5f, 0.5f, 103);
    auto f_in = rand_vec((size_t) T * H, -1.0f, 1.0f, 104);

    // Host FP64 reference
    std::vector<double> ref_coef((size_t) T * kHcCoef, 0.0);
    std::vector<double> ref_pre((size_t) T * H, 0.0);
    std::vector<double> ref_post((size_t) T * flat, 0.0);
    for (int t = 0; t < T; ++t) {
        double ss = 0.0;
        for (int i = 0; i < flat; ++i) ss += (double) x[t * flat + i] * x[t * flat + i];
        const double inv_rms = 1.0 / (std::sqrt(ss / flat) + rms_eps);
        double mix[kHcCoef];
        for (int m = 0; m < kHcCoef; ++m) {
            double dot = 0.0;
            for (int i = 0; i < flat; ++i) dot += (double) x[t * flat + i] * fn[m * flat + i];
            mix[m] = dot * inv_rms;
        }
        auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
        for (int k = 0; k < 4; ++k) ref_coef[t * kHcCoef + k] = sig(mix[k] * sc[0] + bs[k]) + hc_eps;
        for (int k = 0; k < 4; ++k) ref_coef[t * kHcCoef + 4 + k] = sig(mix[4 + k] * sc[1] + bs[4 + k]) * 2.0;

        double cm[4][4];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) cm[r][c] = mix[8 + r * 4 + c] * sc[2] + bs[8 + r * 4 + c];
        for (int r = 0; r < 4; ++r) {
            double mx = std::max({cm[r][0], cm[r][1], cm[r][2], cm[r][3]});
            double sum = 0.0;
            for (int c = 0; c < 4; ++c) { cm[r][c] = std::exp(cm[r][c] - mx); sum += cm[r][c]; }
            for (int c = 0; c < 4; ++c) cm[r][c] = cm[r][c] / sum + hc_eps;
        }
        for (int c = 0; c < 4; ++c) {
            double cs = cm[0][c] + cm[1][c] + cm[2][c] + cm[3][c] + hc_eps;
            for (int r = 0; r < 4; ++r) cm[r][c] /= cs;
        }
        for (int it = 1; it < sink_iters; ++it) {
            for (int r = 0; r < 4; ++r) {
                double rs = cm[r][0] + cm[r][1] + cm[r][2] + cm[r][3] + hc_eps;
                for (int c = 0; c < 4; ++c) cm[r][c] /= rs;
            }
            for (int c = 0; c < 4; ++c) {
                double cs = cm[0][c] + cm[1][c] + cm[2][c] + cm[3][c] + hc_eps;
                for (int r = 0; r < 4; ++r) cm[r][c] /= cs;
            }
        }
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) ref_coef[t * kHcCoef + 8 + r * 4 + c] = cm[r][c];

        for (int i = 0; i < H; ++i) {
            double acc = 0.0;
            for (int s = 0; s < 4; ++s) acc += ref_coef[t * kHcCoef + s] * x[(t * 4 + s) * H + i];
            ref_pre[t * H + i] = acc;
        }
        for (int d = 0; d < 4; ++d) {
            for (int i = 0; i < H; ++i) {
                double acc = ref_coef[t * kHcCoef + 4 + d] * f_in[t * H + i];
                for (int s = 0; s < 4; ++s) acc += cm[s][d] * x[(t * 4 + s) * H + i];
                ref_post[(t * 4 + d) * H + i] = acc;
            }
        }
    }

    DevBuf<float> dx(x.size()), dfn(fn.size()), dsc(sc.size()), dbs(bs.size()), df(f_in.size());
    DevBuf<float> dws(mhc_workspace_bytes(T) / sizeof(float));
    DevBuf<float> dcoef((size_t) T * kHcCoef), dpre((size_t) T * H), dpost((size_t) T * flat);
    dx.upload(x); dfn.upload(fn); dsc.upload(sc); dbs.upload(bs); df.upload(f_in);

    mhc_mixes(dx.ptr, dfn.ptr, dsc.ptr, dbs.ptr, T, H, rms_eps, hc_eps, sink_iters, dws.ptr, dcoef.ptr, nullptr);
    mhc_pre(dx.ptr, dcoef.ptr, kHcCoef, T, H, dpre.ptr, nullptr);
    mhc_post(df.ptr, dx.ptr, dcoef.ptr, kHcCoef, T, H, dpost.ptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hcoef(ref_coef.size()), hpre(ref_pre.size()), hpost(ref_post.size());
    dcoef.download(hcoef); dpre.download(hpre); dpost.download(hpost);
    check_tol("mhc_mixes (ERNIE rms_eps outside)", max_abs_err(hcoef, ref_coef), 1e-5);
    check_tol("mhc_pre", max_abs_err(hpre, ref_pre), 1e-5);
    check_tol("mhc_post (comb[d,s])", max_abs_err(hpost, ref_post), 1e-5);

    // Final mhc_contract
    auto hfn = rand_vec((size_t) 4 * flat, -0.1f, 0.1f, 105);
    auto hsc = std::vector<float>{0.2f};
    auto hbs = rand_vec(4, -0.4f, 0.4f, 106);
    std::vector<double> ref_con((size_t) T * H, 0.0);
    for (int t = 0; t < T; ++t) {
        double ss = 0.0;
        for (int i = 0; i < flat; ++i) ss += (double) x[t * flat + i] * x[t * flat + i];
        const double inv_rms = 1.0 / (std::sqrt(ss / flat) + rms_eps);
        double w4[4];
        for (int k = 0; k < 4; ++k) {
            double dot = 0.0;
            for (int i = 0; i < flat; ++i) dot += (double) x[t * flat + i] * hfn[k * flat + i];
            w4[k] = 1.0 / (1.0 + std::exp(-((dot * inv_rms) * hsc[0] + hbs[k]))) + hc_eps;
        }
        for (int i = 0; i < H; ++i) {
            double acc = 0.0;
            for (int s = 0; s < 4; ++s) acc += w4[s] * x[(t * 4 + s) * H + i];
            ref_con[t * H + i] = acc;
        }
    }
    DevBuf<float> dhfn(hfn.size()), dhsc(1), dhbs(4), dcoef4((size_t) T * 4), dcon((size_t) T * H);
    dhfn.upload(hfn); dhsc.upload(hsc); dhbs.upload(hbs);
    mhc_contract(dx.ptr, dhfn.ptr, dhsc.ptr, dhbs.ptr, T, H, rms_eps, hc_eps, dws.ptr, dcoef4.ptr, dcon.ptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hcon(ref_con.size());
    dcon.download(hcon);
    check_tol("mhc_contract", max_abs_err(hcon, ref_con), 1e-5);
}

// -----------------------------------------------------------------------------
// 2. Multimax & SiTU-GLU & Split-Sigmoid Router & VHA / Gate / LN / Hadamard
// -----------------------------------------------------------------------------
void test_elementwise_and_router() {
    // Multimax
    const int T = 2, V = 4096;
    auto logits = rand_vec((size_t) T * V, -25.0f, 25.0f, 201);
    std::vector<float> ranges = {-10.0f, 12.0f, -15.0f, 18.0f};
    std::vector<float> ts = {0.25f, -0.3f, 0.05f, -0.02f};
    std::vector<double> ref_mm(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        double x = logits[i];
        double a = std::max(0.0, (double) ranges[0] - x);
        double b = std::max(0.0, x - (double) ranges[1]);
        double c = std::max(0.0, (double) ranges[2] - x);
        double d = std::max(0.0, x - (double) ranges[3]);
        ref_mm[i] = x + ts[0] * a + ts[1] * b + ts[2] * c * c + ts[3] * d * d;
    }
    DevBuf<float> dlog(logits.size()), dr(4), dts(4);
    dlog.upload(logits); dr.upload(ranges); dts.upload(ts);
    multimax_f32(dlog.ptr, dr.ptr, dts.ptr, T, V, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hlog(logits.size());
    dlog.download(hlog);
    check_tol("multimax_f32", max_abs_err(hlog, ref_mm), 1e-4);

    // SiTU-GLU (beta=4.0, linear_beta=25.0)
    const int FF = 2048;
    auto gate = rand_vec((size_t) T * FF, -10.0f, 10.0f, 202);
    auto up = rand_vec((size_t) T * FF, -40.0f, 40.0f, 203);
    std::vector<double> ref_situ(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        double g = gate[i], u = up[i];
        double sg = 4.0 * std::tanh(g / 4.0) * (1.0 / (1.0 + std::exp(-g)));
        double su = 25.0 * std::tanh(u / 25.0);
        ref_situ[i] = sg * su;
    }
    DevBuf<float> dg(gate.size()), du(up.size()), do_situ(gate.size());
    dg.upload(gate); du.upload(up);
    situ_glu_f32(dg.ptr, du.ptr, T, FF, 4.0f, 25.0f, do_situ.ptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hsitu(gate.size());
    do_situ.download(hsitu);
    check_tol("situ_glu_f32 (4.0 / 25.0)", max_abs_err(hsitu, ref_situ), 1e-4);

    // Split-sigmoid MoE router (512 experts, top-10)
    const int E = 512, K = 10;
    auto rlog = rand_vec((size_t) T * 2 * E, -3.0f, 3.0f, 204);
    auto rbias = rand_vec(E, -0.5f, 0.5f, 205);
    auto rscale = rand_vec(E, 0.5f, 2.5f, 206);
    std::vector<int32_t> ref_ids((size_t) T * K);
    std::vector<double> ref_w((size_t) T * K);
    for (int t = 0; t < T; ++t) {
        std::vector<double> sc(E), sel(E);
        for (int e = 0; e < E; ++e) {
            double l0 = rlog[t * 2 * E + e], l1 = rlog[t * 2 * E + E + e];
            sc[e] = 1.0 / (1.0 + std::exp(-l0)) + 1.0 / (1.0 + std::exp(-l1));
            sel[e] = sc[e] + rbias[e];
        }
        std::vector<int> idx(E);
        for (int e = 0; e < E; ++e) idx[e] = e;
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return sel[a] > sel[b]; });
        double tot = 0.0;
        for (int k = 0; k < K; ++k) { ref_ids[t * K + k] = idx[k]; tot += sc[idx[k]]; }
        for (int k = 0; k < K; ++k) ref_w[t * K + k] = (sc[idx[k]] / tot) * rscale[idx[k]];
    }
    DevBuf<float> drl(rlog.size()), drb(E), drs(E), drw((size_t) T * K);
    DevBuf<int32_t> dri((size_t) T * K);
    drl.upload(rlog); drb.upload(rbias); drs.upload(rscale);
    split_sigmoid_route(drl.ptr, drb.ptr, drs.ptr, T, E, K, true, dri.ptr, drw.ptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> hri(ref_ids.size());
    std::vector<float> hrw(ref_w.size());
    dri.download(hri); drw.download(hrw);
    double id_mismatch = 0.0;
    for (size_t i = 0; i < ref_ids.size(); ++i) id_mismatch += (hri[i] != ref_ids[i]) ? 1.0 : 0.0;
    check_tol("split_sigmoid_route (ids)", id_mismatch, 0.0);
    check_tol("split_sigmoid_route (weights)", max_abs_err(hrw, ref_w), 1e-5);

    // VHA PostMix + Sigmoid gate + LayerNorm + Hadamard-128
    const int NH = 16, HD = 64;
    auto vin = rand_vec((size_t) T * NH * HD, -1.0f, 1.0f, 207);
    auto vmix = rand_vec((size_t) NH * NH, -0.5f, 0.5f, 208);
    auto gvec = rand_vec((size_t) T * NH * HD, -2.0f, 2.0f, 209);
    std::vector<double> ref_vha(vin.size(), 0.0);
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < NH; ++h)
            for (int d = 0; d < HD; ++d) {
                double acc = 0.0;
                for (int g = 0; g < NH; ++g) acc += (double) vmix[h * NH + g] * vin[(t * NH + g) * HD + d];
                ref_vha[(t * NH + h) * HD + d] = acc * (1.0 / (1.0 + std::exp(-(double) gvec[(t * NH + h) * HD + d])));
            }
    DevBuf<float> dvin(vin.size()), dvmix(vmix.size()), dvout(vin.size()), dgvec(gvec.size());
    dvin.upload(vin); dvmix.upload(vmix); dgvec.upload(gvec);
    vha_postmix_f32(dvin.ptr, dvmix.ptr, T, NH, HD, dvout.ptr, nullptr);
    sigmoid_gate_mul_f32(dvout.ptr, dgvec.ptr, (int64_t) vin.size(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hvout(vin.size());
    dvout.download(hvout);
    check_tol("vha_postmix + sigmoid_gate_mul", max_abs_err(hvout, ref_vha), 1e-5);

    // LayerNorm + Hadamard-128 (orthonormal involution: H(H(x)) == x)
    const int D = 128;
    auto lnx = rand_vec((size_t) T * D, -2.0f, 3.0f, 210);
    auto lnw = rand_vec(D, 0.5f, 1.5f, 211);
    auto lnb = rand_vec(D, -0.5f, 0.5f, 212);
    std::vector<double> ref_ln(lnx.size());
    for (int t = 0; t < T; ++t) {
        double mean = 0.0;
        for (int i = 0; i < D; ++i) mean += lnx[t * D + i];
        mean /= D;
        double var = 0.0;
        for (int i = 0; i < D; ++i) { double d = lnx[t * D + i] - mean; var += d * d; }
        double inv = 1.0 / std::sqrt(var / D + 1e-6);
        for (int i = 0; i < D; ++i) ref_ln[t * D + i] = (lnx[t * D + i] - mean) * inv * lnw[i] + lnb[i];
    }
    DevBuf<float> dlnx(lnx.size()), dlnw(D), dlnb(D), dlno(lnx.size());
    dlnx.upload(lnx); dlnw.upload(lnw); dlnb.upload(lnb);
    layernorm_f32(dlnx.ptr, dlnw.ptr, dlnb.ptr, T, D, 1e-6f, dlno.ptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hlno(lnx.size());
    dlno.download(hlno);
    check_tol("layernorm_f32", max_abs_err(hlno, ref_ln), 1e-5);

    // Hadamard-128 exact Sylvester check + H(H(x)) == x
    std::vector<double> ref_had(lnx.size());
    const double inv_sqrt128 = 1.0 / std::sqrt(128.0);
    for (int t = 0; t < T; ++t) {
        std::vector<double> v(D);
        for (int i = 0; i < D; ++i) v[i] = lnx[t * D + i];
        for (int len = 1; len < D; len <<= 1) {
            for (int i = 0; i < D; i += 2 * len) {
                for (int j = 0; j < len; ++j) {
                    double a = v[i + j], b = v[i + j + len];
                    v[i + j] = a + b;
                    v[i + j + len] = a - b;
                }
            }
        }
        for (int i = 0; i < D; ++i) ref_had[t * D + i] = v[i] * inv_sqrt128;
    }
    hadamard_128_f32(dlnx.ptr, T, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hhad(lnx.size());
    dlnx.download(hhad);
    check_tol("hadamard_128_f32 (Sylvester)", max_abs_err(hhad, ref_had), 1e-5);
    hadamard_128_f32(dlnx.ptr, T, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    dlnx.download(hhad);
    std::vector<double> lnx_d(lnx.begin(), lnx.end());
    check_tol("hadamard_128_f32 (involution H^2=I)", max_abs_err(hhad, lnx_d), 1e-5);
}

// -----------------------------------------------------------------------------
// 3. HCA Attention Pipeline (128:1 streaming compress w/ APE + window + sink + inv_rope)
// -----------------------------------------------------------------------------
void test_hca() {
    const int N = 160;   // crosses 128-token boundary -> 1 closed compressed block + 128 window
    const int ratio = 128, win = 128, HD = 512, NH = 4;
    const float rms_eps = 1e-6f;
    const RopeTable rt = rope_table_plain(100000.0f);
    const RopeTable comp_rt = rope_table_yarn(160000.0f, 8.0f, 32.0f, 1.0f, 131072);

    auto q_all = rand_vec((size_t) N * NH * HD, -0.5f, 0.5f, 301);
    auto kv_in = rand_vec((size_t) N * HD, -0.5f, 0.5f, 302);
    auto kv_nw = rand_vec(HD, 0.8f, 1.2f, 303);
    auto ckv = rand_vec((size_t) N * HD, -0.5f, 0.5f, 304);
    auto csc = rand_vec((size_t) N * HD, -1.0f, 1.0f, 305);
    auto ape = rand_vec((size_t) ratio * HD, -0.5f, 0.5f, 306);
    auto comp_nw = rand_vec(HD, 0.8f, 1.2f, 307);
    auto sink = rand_vec(NH, -1.0f, 1.0f, 308);
    std::vector<int32_t> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = i;

    auto rot_pair = [](double& x0, double& x1, double p, float step, float mscale, bool inv) {
        double ang = p * (double) step;
        double c = std::cos(ang) * mscale, s = std::sin(ang) * mscale;
        if (inv) s = -s;
        double a = x0 * c - x1 * s, b = x0 * s + x1 * c;
        x0 = a; x1 = b;
    };

    // Host FP64 reference for the last token (t = N - 1 = 159: n_comp = 1, win = [32..159])
    std::vector<std::vector<double>> win_ref(win, std::vector<double>(HD));
    for (int t = 0; t < N; ++t) {
        double ss = 0.0;
        for (int c = 0; c < HD; ++c) ss += (double) kv_in[t * HD + c] * kv_in[t * HD + c];
        double inv = 1.0 / std::sqrt(ss / HD + rms_eps);
        std::vector<double> row(HD);
        for (int c = 0; c < HD; ++c) row[c] = kv_in[t * HD + c] * inv * kv_nw[c];
        for (int pr = 0; pr < 32; ++pr)
            rot_pair(row[HD - 64 + 2 * pr], row[HD - 64 + 2 * pr + 1], (double) t, rt.step[pr], rt.mscale, false);
        win_ref[t % win] = row;
    }
    // Compressed block 0: positions 0..127
    std::vector<double> comp0(HD);
    for (int c = 0; c < HD; ++c) {
        double mx = -1e30;
        for (int j = 0; j < ratio; ++j) mx = std::max(mx, (double) csc[j * HD + c] + ape[j * HD + c]);
        double num = 0.0, den = 0.0;
        for (int j = 0; j < ratio; ++j) {
            double e = std::exp((double) csc[j * HD + c] + ape[j * HD + c] - mx);
            den += e;
            num += (double) ckv[j * HD + c] * e;
        }
        comp0[c] = num / den;
    }
    {
        double ss = 0.0;
        for (int c = 0; c < HD; ++c) ss += comp0[c] * comp0[c];
        double inv = 1.0 / std::sqrt(ss / HD + rms_eps);
        for (int c = 0; c < HD; ++c) comp0[c] = comp0[c] * inv * comp_nw[c];
        for (int pr = 0; pr < 32; ++pr)
            rot_pair(comp0[HD - 64 + 2 * pr], comp0[HD - 64 + 2 * pr + 1], 0.0, comp_rt.step[pr], comp_rt.mscale, false);
    }

    const int t_last = N - 1;
    std::vector<double> ref_out((size_t) NH * HD);
    for (int h = 0; h < NH; ++h) {
        std::vector<double> qh(HD);
        double qss = 0.0;
        for (int c = 0; c < HD; ++c) {
            qh[c] = q_all[(t_last * NH + h) * HD + c];
            qss += qh[c] * qh[c];
        }
        const double qinv = 1.0 / std::sqrt(qss / HD + rms_eps);
        for (int c = 0; c < HD; ++c) qh[c] *= qinv;
        for (int pr = 0; pr < 32; ++pr)
            rot_pair(qh[HD - 64 + 2 * pr], qh[HD - 64 + 2 * pr + 1], (double) t_last, rt.step[pr], rt.mscale, false);
        double scale = 1.0 / std::sqrt((double) HD);
        std::vector<double> scores;
        std::vector<const std::vector<double>*> rows;
        auto add_row = [&](const std::vector<double>& r) {
            double dot = 0.0;
            for (int c = 0; c < HD; ++c) dot += qh[c] * r[c];
            scores.push_back(dot * scale);
            rows.push_back(&r);
        };
        add_row(comp0);
        for (int j = t_last - win + 1; j <= t_last; ++j) add_row(win_ref[j % win]);
        double mx = sink[h];
        for (double s : scores) mx = std::max(mx, s);
        double l = std::exp((double) sink[h] - mx);
        std::vector<double> acc(HD, 0.0);
        for (size_t i = 0; i < scores.size(); ++i) {
            double w = std::exp(scores[i] - mx);
            l += w;
            for (int c = 0; c < HD; ++c) acc[c] += w * (*rows[i])[c];
        }
        for (int c = 0; c < HD; ++c) acc[c] /= l;
        for (int pr = 0; pr < 32; ++pr)
            rot_pair(acc[HD - 64 + 2 * pr], acc[HD - 64 + 2 * pr + 1], (double) t_last, rt.step[pr], rt.mscale, true);
        for (int c = 0; c < HD; ++c) ref_out[h * HD + c] = acc[c];
    }

    // Run GPU HCA pipeline (FP16 cache for tight parity check)
    DevBuf<float> dq(q_all.size()), dkv(kv_in.size()), dkvnw(HD), dckv(ckv.size()), dcsc(csc.size());
    DevBuf<float> dape(ape.size()), dcnw(HD), dsink(NH), dst_kv((size_t) ratio * HD), dst_sc((size_t) ratio * HD);
    DevBuf<float> dout((size_t) NH * HD);
    DevBuf<int32_t> dpos(N), dpos_last(1);
    DevBuf<uint8_t> dwin((size_t) win * HD * 2), dcomp((size_t) 4 * HD * 2);
    dq.upload(q_all); dkv.upload(kv_in); dkvnw.upload(kv_nw); dckv.upload(ckv); dcsc.upload(csc);
    dape.upload(ape); dcnw.upload(comp_nw); dsink.upload(sink); dpos.upload(pos);
    int32_t plast = t_last;
    dpos_last.upload(&plast, 1);
    dst_kv.zero(); dst_sc.zero(); dwin.zero(); dcomp.zero();

    hca_q_rope(dq.ptr, dpos.ptr, N, NH, HD, rms_eps, rt, nullptr);
    hca_kv_store_window(dkv.ptr, dkvnw.ptr, dpos.ptr, N, HD, win, rms_eps, rt, dwin.ptr, false, nullptr);
    hca_compress(dckv.ptr, dcsc.ptr, dape.ptr, dcnw.ptr, dpos.ptr, N, ratio, HD, rms_eps, comp_rt,
                 dst_kv.ptr, dst_sc.ptr, dcomp.ptr, false, nullptr);
    hca_attention(dq.ptr + (size_t) t_last * NH * HD, dcomp.ptr, dwin.ptr, nullptr, dsink.ptr, dpos_last.ptr,
                  1, NH, HD, win, ratio, false, dout.ptr, nullptr);
    hca_inv_rope(dout.ptr, dpos_last.ptr, 1, NH, HD, rt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hout(ref_out.size());
    dout.download(hout);
    check_tol("HCA pipeline (compress+APE+attn+inv_rope)", rel_l2_err(hout, ref_out), 2e-3);

    // the decode windows' split attention over the same caches
    DevBuf<float> dout2((size_t) NH * HD), dpart((size_t) kAttnSplitSlots * NH * kAttnPartFloats);
    hca_attention_split(dq.ptr + (size_t) t_last * NH * HD, dcomp.ptr, dwin.ptr, nullptr, dsink.ptr, dpos_last.ptr, 1,
                        NH, win, ratio, false, dout2.ptr, dpart.ptr, nullptr, 0);
    hca_inv_rope(dout2.ptr, dpos_last.ptr, 1, NH, HD, rt, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hout2(ref_out.size());
    dout2.download(hout2);
    check_tol("hca_attention_split", rel_l2_err(hout2, ref_out), 2e-3);
}

// -----------------------------------------------------------------------------
// 4. MLA-DSA + Lightning Indexer Pipeline
// -----------------------------------------------------------------------------
void test_mla_dsa() {
    const int N = 40;
    const int win = 8, top_k = 6, width = top_k + win;
    const int idx_h = 64, idx_d = 128, rope_d = 64;
    const int NH = 4, qk_nope = 64, qk_rope = 64, kv_lora = 128, v_dim = 64;
    const float eps = 1e-6f;
    const RopeTable rt = rope_table_plain(100000.0f);

    // Indexer prepare + topk_merge (FP16 cache)
    auto iq = rand_vec((size_t) N * idx_h * idx_d, -0.5f, 0.5f, 401);
    auto ik = rand_vec((size_t) N * idx_d, -0.5f, 0.5f, 402);
    auto knw = rand_vec(idx_d, 0.8f, 1.2f, 403);
    auto knb = rand_vec(idx_d, -0.2f, 0.2f, 404);
    auto iw = rand_vec((size_t) N * idx_h, 0.1f, 1.0f, 405);
    std::vector<int32_t> pos(N);
    for (int i = 0; i < N; ++i) pos[i] = i;

    DevBuf<float> diq(iq.size()), dik(ik.size()), dknw(idx_d), dknb(idx_d), diw(iw.size());
    DevBuf<int32_t> dpos(N), dsel((size_t) N * width);
    DevBuf<uint8_t> didx_k((size_t) N * idx_d * 2);
    DevBuf<float> dscores((size_t) N * N);
    diq.upload(iq); dik.upload(ik); dknw.upload(knw); dknb.upload(knb); diw.upload(iw); dpos.upload(pos);
    dsa_indexer_prepare(diq.ptr, dik.ptr, dknw.ptr, dknb.ptr, dpos.ptr, N, idx_h, idx_d, rope_d, eps, rt,
                        didx_k.ptr, false, nullptr);
    dsa_indexer_topk_merge(diq.ptr, diw.ptr, didx_k.ptr, dpos.ptr, N, idx_h, idx_d, top_k, win, false,
                           dsel.ptr, nullptr, dscores.ptr, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download transformed iq, ik and verify top-K for last token t = N - 1 against CPU scoring
    std::vector<float> hiq(iq.size()), hik(ik.size());
    std::vector<int32_t> hsel(N * width);
    diq.download(hiq); dik.download(hik); dsel.download(hsel);

    const int t = N - 1;
    const int older_n = N - win;
    const double wscale = 1.0 / std::sqrt((double) (idx_d * idx_h));
    std::vector<std::pair<double, int>> scored;
    for (int j = 0; j < older_n; ++j) {
        double tot = 0.0;
        for (int h = 0; h < idx_h; ++h) {
            double dot = 0.0;
            for (int d = 0; d < idx_d; ++d) dot += (double) hiq[(t * idx_h + h) * idx_d + d] * hik[j * idx_d + d];
            if (dot > 0.0) tot += dot * iw[t * idx_h + h] * wscale;
        }
        scored.push_back({tot, j});
    }
    std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<int> gpu_older(hsel.begin() + t * width, hsel.begin() + t * width + top_k);
    std::vector<int> ref_older;
    for (int k = 0; k < top_k; ++k) ref_older.push_back(scored[k].second);
    std::sort(gpu_older.begin(), gpu_older.end());
    std::sort(ref_older.begin(), ref_older.end());
    double topk_err = 0.0;
    for (int k = 0; k < top_k; ++k) topk_err += (gpu_older[k] != ref_older[k]) ? 1.0 : 0.0;
    // Also verify trailing sliding window [N - win .. N - 1]
    for (int k = 0; k < win; ++k) topk_err += (hsel[t * width + top_k + k] != (N - win + k)) ? 1.0 : 0.0;
    check_tol("dsa_indexer_prepare + topk_merge", topk_err, 0.0);

    // Absorbed sparse MLA attention parity (FP16 KV cache + Q8_0 kv_b)
    auto q_mla = rand_vec((size_t) N * NH * (qk_nope + qk_rope), -0.5f, 0.5f, 406);
    auto kv_a = rand_vec((size_t) N * (kv_lora + qk_rope), -0.5f, 0.5f, 407);
    auto kv_nw = rand_vec(kv_lora, 0.8f, 1.2f, 408);
    auto kv_b_f = rand_vec((size_t) NH * (qk_nope + v_dim) * kv_lora, -0.3f, 0.3f, 409);
    auto sink = rand_vec(NH, -1.0f, 1.0f, 410);
    std::vector<uint8_t> kv_b_q8;
    quantize_q8_0(kv_b_f, NH * (qk_nope + v_dim), kv_lora, kv_b_q8);
    auto kv_b_deq = dequantize_q8_0(kv_b_q8, NH * (qk_nope + v_dim), kv_lora);

    DevBuf<float> dqm(q_mla.size()), dkva(kv_a.size()), dkvnw(kv_lora), dsink(NH), dmo((size_t) NH * v_dim);
    DevBuf<uint8_t> dkv_cache((size_t) N * (kv_lora + qk_rope) * 2), dkvb(kv_b_q8.size());
    dqm.upload(q_mla); dkva.upload(kv_a); dkvnw.upload(kv_nw); dsink.upload(sink); dkvb.upload(kv_b_q8);

    dsa_qkv_rope_and_store(dqm.ptr, dkva.ptr, dkvnw.ptr, dpos.ptr, N, NH, qk_nope, qk_rope, kv_lora, eps, rt,
                           dkv_cache.ptr, false, nullptr);
    dsa_mla_attention(dqm.ptr + (size_t) t * NH * (qk_nope + qk_rope), dkv_cache.ptr, dkvb.ptr, dsink.ptr,
                      dsel.ptr + (size_t) t * width, 1, NH, qk_nope, qk_rope, kv_lora, v_dim, width, false,
                      dmo.ptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download post-RoPE q and FP16 kv_cache to compare attention math directly
    std::vector<float> hqm(q_mla.size()), hmo((size_t) NH * v_dim);
    std::vector<uint8_t> hkvc(dkv_cache.count);
    dqm.download(hqm); dmo.download(hmo); dkv_cache.download(hkvc);
    const __half* hkv_h = reinterpret_cast<const __half*>(hkvc.data());

    std::vector<double> ref_mla((size_t) NH * v_dim, 0.0);
    const int qd = qk_nope + qk_rope;
    const double scale = 1.0 / std::sqrt((double) qd);
    for (int h = 0; h < NH; ++h) {
        const float* qr = hqm.data() + (t * NH + h) * qd;
        const double* w_uk = kv_b_deq.data() + (size_t) h * (qk_nope + v_dim) * kv_lora;
        const double* w_uv = w_uk + (size_t) qk_nope * kv_lora;
        std::vector<double> q_lat(kv_lora, 0.0);
        for (int c = 0; c < kv_lora; ++c)
            for (int r = 0; r < qk_nope; ++r) q_lat[c] += (double) qr[r] * w_uk[r * kv_lora + c];
        std::vector<double> scs;
        std::vector<int> cells;
        for (int s_i = 0; s_i < width; ++s_i) {
            int cell = hsel[t * width + s_i];
            if (cell < 0) break;
            const __half* row = hkv_h + (size_t) cell * (kv_lora + qk_rope);
            double dot = 0.0;
            for (int c = 0; c < kv_lora; ++c) dot += q_lat[c] * __half2float(row[c]);
            for (int c = 0; c < qk_rope; ++c) dot += (double) qr[qk_nope + c] * __half2float(row[kv_lora + c]);
            scs.push_back(dot * scale);
            cells.push_back(cell);
        }
        double mx = sink[h];
        for (double s : scs) mx = std::max(mx, s);
        double l = std::exp((double) sink[h] - mx);
        std::vector<double> o_lat(kv_lora, 0.0);
        for (size_t i = 0; i < scs.size(); ++i) {
            double w = std::exp(scs[i] - mx);
            l += w;
            const __half* row = hkv_h + (size_t) cells[i] * (kv_lora + qk_rope);
            for (int c = 0; c < kv_lora; ++c) o_lat[c] += w * __half2float(row[c]);
        }
        for (int c = 0; c < kv_lora; ++c) o_lat[c] /= l;
        for (int r = 0; r < v_dim; ++r) {
            double acc = 0.0;
            for (int c = 0; c < kv_lora; ++c) acc += w_uv[r * kv_lora + c] * o_lat[c];
            ref_mla[h * v_dim + r] = acc;
        }
    }
    check_tol("dsa_mla_attention (absorbed Q8_0 + sink)", rel_l2_err(hmo, ref_mla), 5e-4);
}

}  // namespace

// The decode windows' split DSA attention against the one-block-a-head-group kernel, at the model's geometry
// (64 heads, nope 192, rope 64, lora 512, v 256), FP8 cache, for 1 and 3 rows with trailing -1 selections.
void test_dsa_split() {
    const int NH = 64, nope = 192, rope = 64, lora = 512, vd = 256, N = 700, width = 2176, T = 3;
    const int row_bytes = 656;
    auto q = rand_vec((size_t) T * NH * (nope + rope), -0.5f, 0.5f, 501);
    auto kv_b_f = rand_vec((size_t) NH * (nope + vd) * lora, -0.05f, 0.05f, 502);
    auto sink = rand_vec(NH, -1.0f, 1.0f, 503);
    std::vector<uint8_t> kv_b_q8;
    quantize_q8_0(kv_b_f, NH * (nope + vd), lora, kv_b_q8);
    // an FP8 cache row: 512 E4M3 codes, 4 float scales, 64 BF16 rope values
    std::vector<uint8_t> cache((size_t) N * row_bytes);
    uint32_t st = 7;
    for (int c = 0; c < N; ++c) {
        uint8_t* r = cache.data() + (size_t) c * row_bytes;
        for (int i = 0; i < lora; ++i) {
            st = st * 1664525u + 1013904223u;
            uint8_t b = (uint8_t) (st >> 24);
            if ((b & 0x7f) >= 0x7e) b &= 0xf0;   // no NaN codes
            r[i] = b;
        }
        float sc[4] = {0.01f, 0.02f, 0.015f, 0.012f};
        std::memcpy(r + lora, sc, sizeof(sc));
        for (int i = 0; i < rope; ++i) {
            st = st * 1664525u + 1013904223u;
            const float v = ((float) (st >> 8) / 16777216.0f - 0.5f);
            uint32_t u;
            std::memcpy(&u, &v, 4);
            const uint16_t bf = (uint16_t) (u >> 16);
            std::memcpy(r + lora + 16 + 2 * i, &bf, 2);
        }
    }
    std::vector<int32_t> sel((size_t) T * width, -1);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < N - 50 * t; ++i) sel[(size_t) t * width + i] = (i * 7 + t) % N;
    DevBuf<float> dq(q.size()), dsink(NH), da((size_t) T * NH * vd), db((size_t) T * NH * vd);
    DevBuf<float> dpart((size_t) kAttnSplitSlots * NH * kAttnPartFloats), dql((size_t) 8 * NH * lora);
    DevBuf<uint8_t> dkvb(kv_b_q8.size()), dcache(cache.size());
    DevBuf<int32_t> dsel(sel.size());
    std::vector<int32_t> pos(T);   // a row's selection holds its first pos + 1 positions
    for (int t = 0; t < T; ++t) pos[(size_t) t] = N - 50 * t - 1;
    DevBuf<int32_t> dpos(pos.size());
    dq.upload(q); dsink.upload(sink); dkvb.upload(kv_b_q8); dcache.upload(cache); dsel.upload(sel); dpos.upload(pos);
    for (int rows : {1, T}) {
        dsa_mla_attention(dq.ptr, dcache.ptr, dkvb.ptr, dsink.ptr, dsel.ptr, rows, NH, nope, rope, lora, vd, width, true,
                          da.ptr, nullptr);
        const bool ok = dsa_mla_attention_split(dq.ptr, dcache.ptr, dkvb.ptr, dsink.ptr, dsel.ptr, dpos.ptr, rows, NH, nope,
                                                rope, lora, vd, width, true, db.ptr, dpart.ptr, dql.ptr, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> ha((size_t) T * NH * vd), hb((size_t) T * NH * vd);
        da.download(ha);
        db.download(hb);
        ha.resize((size_t) rows * NH * vd);
        hb.resize((size_t) rows * NH * vd);
        std::vector<double> ref(ha.begin(), ha.end());
        check_tol(rows == 1 ? "dsa_mla_attention_split (1 row)" : "dsa_mla_attention_split (3 rows)",
                  ok ? rel_l2_err(hb, ref) : 1.0, 1e-5);
    }
}

// -----------------------------------------------------------------------------
// 8. a decode window's fused launches against the sequences they replace: bit for bit
// -----------------------------------------------------------------------------
template <typename T>
bool same_bits(const DevBuf<T>& a, const DevBuf<T>& b, size_t n) {
    std::vector<T> ha(n), hb(n);
    CUDA_CHECK(cudaMemcpy(ha.data(), a.ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), b.ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return std::memcmp(ha.data(), hb.data(), n * sizeof(T)) == 0;
}

void check_bits(const char* name, bool ok) {
    std::printf("[%s] %-48s %s\n", ok ? "PASS" : "FAIL", name, ok ? "bitwise" : "differs");
    if (!ok) ++g_failures;
}

void test_fused() {
    namespace bk = basalt::kernels;
    {   // many values: the quantization's division is native_mmvq.cu's fast-math one
        cudaStream_t st = nullptr;
        CUDA_CHECK(cudaStreamCreate(&st));
        const int T = 8, cols = 8192;
        bool ok = true;
        for (uint32_t seed = 0; seed < 40 && ok; ++seed) {
            auto v = rand_vec((size_t) T * cols, -3.f, 3.f, 900 + seed), lw = rand_vec(cols, 0.5f, 1.5f, 950 + seed);
            DevBuf<float> dv(v.size()), dl(cols), nrm(v.size());
            dv.upload(v); dl.upload(lw);
            const size_t qn = bk::native_q8_1_bytes(cols, T);
            DevBuf<uint8_t> qa(qn), qb(qn);
            bk::rms_norm_rows(dv.ptr, dl.ptr, T, cols, 1e-6f, nrm.ptr, st);
            bk::native_quantize_q8_1(nrm.ptr, qa.ptr, cols, T, st);
            rms_norm_q8(dv.ptr, dl.ptr, T, cols, 1e-6f, nullptr, qb.ptr, st);
            CUDA_CHECK(cudaStreamSynchronize(st));
            ok = same_bits(qa, qb, qn);
        }
        check_bits("rms_norm_q8 (40 x 8 rows of 8192)", ok);
        {   // the router's picks with many ties (the lower id first among equals), against a serial pass
            const int E = 512, K = 10, R = 8;
            bool ok3 = true;
            for (uint32_t seed = 0; seed < 8 && ok3; ++seed) {
                std::mt19937 rng(seed);
                std::vector<float> lg((size_t) R * 2 * E), bias(E), sc(E);
                for (auto& v : lg) v = (float) ((int) (rng() % 7) - 3) * 0.5f;
                for (auto& v : bias) v = (float) ((int) (rng() % 3)) * 0.25f;
                for (auto& v : sc) v = 0.5f + (float) (rng() % 100) / 100.0f;
                DevBuf<float> dl(lg.size()), db(E), ds(E), dw((size_t) R * K);
                DevBuf<int32_t> di((size_t) R * K);
                dl.upload(lg); db.upload(bias); ds.upload(sc);
                split_sigmoid_route(dl.ptr, db.ptr, ds.ptr, R, E, K, true, di.ptr, dw.ptr, st);
                CUDA_CHECK(cudaStreamSynchronize(st));
                std::vector<int32_t> hi((size_t) R * K);
                di.download(hi);
                for (int r = 0; r < R && ok3; ++r) {
                    std::vector<float> score(E), sel(E);
                    for (int e = 0; e < E; ++e) {
                        const float a1 = 1.0f / (1.0f + std::exp(-lg[(size_t) r * 2 * E + e]));
                        const float a2 = 1.0f / (1.0f + std::exp(-lg[(size_t) r * 2 * E + E + e]));
                        score[e] = a1 + a2;
                        sel[e] = score[e] + bias[e];
                    }
                    for (int k = 0; k < K; ++k) {
                        int best = -1;
                        for (int e = 0; e < E; ++e) if (best < 0 || sel[e] > sel[best]) best = e;
                        if (hi[(size_t) r * K + k] != best) ok3 = false;
                        sel[best] = -INFINITY;
                    }
                }
            }
            check_bits("split_sigmoid_route ties (8 x 8 rows)", ok3);
        }
        {   // hca_post_rope_q8 over many rows and positions
            const int NH = 64, HD = 512, G = 8, hpg = NH / G, gin = hpg * HD;
            const RopeTable rt = rope_table_yarn(160000.0f, 16.0f, 32.0f, 1.0f, 65536);
            bool ok2 = true;
            for (uint32_t seed = 0; seed < 24 && ok2; ++seed) {
                std::vector<int32_t> pos(T);
                for (int t = 0; t < T; ++t) pos[t] = (int32_t) (seed * 4099 + 37 * t);
                DevBuf<int32_t> dpos(T);
                dpos.upload(pos);
                auto in = rand_vec((size_t) T * NH * HD, -4.f, 4.f, 700 + seed), mix = rand_vec((size_t) NH * NH, -0.5f, 0.5f, 800 + seed);
                DevBuf<float> din(in.size()), dmix(mix.size()), o2(in.size()), o3(in.size());
                din.upload(in); dmix.upload(mix);
                const size_t qn = bk::native_q8_1_bytes(gin, 1) * G * T;
                DevBuf<uint8_t> qa(qn), qb(qn);
                vha_postmix_f32(din.ptr, dmix.ptr, T, NH, HD, o2.ptr, st);
                hca_inv_rope(o2.ptr, dpos.ptr, T, NH, HD, rt, st);
                hca_groups_split_f32(o2.ptr, T, G, gin, o3.ptr, st);
                for (int c0 = 0; c0 < G * T; c0 += 8)
                    bk::native_quantize_q8_1(o3.ptr + (size_t) c0 * gin, qa.ptr + (size_t) c0 * bk::native_q8_1_bytes(gin, 1), gin,
                                             std::min(8, G * T - c0), st);
                hca_post_rope_q8(din.ptr, dmix.ptr, dpos.ptr, T, NH, HD, hpg, rt, qb.ptr, st);
                CUDA_CHECK(cudaStreamSynchronize(st));
                ok2 = same_bits(qa, qb, qn);
            }
            check_bits("hca_post_rope_q8 (24 x 8 rows)", ok2);
        }
        CUDA_CHECK(cudaStreamDestroy(st));
    }
    const int H = 4096, flat = kHcMult * H;
    const float rms_eps = 1e-6f, hc_eps = 1e-6f;
    for (int T : {1, 3, 8}) {
        char name[96];
        // mhc_pre_norm
        auto x = rand_vec((size_t) T * flat, -2.f, 2.f, 11 + T);
        auto fn = rand_vec((size_t) kHcCoef * flat, -0.02f, 0.02f, 12);
        auto sc = rand_vec(3, 0.5f, 1.5f, 13), bs = rand_vec(kHcCoef, -0.5f, 0.5f, 14);
        auto nw = rand_vec(H, 0.5f, 1.5f, 15);
        DevBuf<float> dx(x.size()), dfn(fn.size()), dsc(3), dbs(kHcCoef), dnw(H);
        dx.upload(x); dfn.upload(fn); dsc.upload(sc); dbs.upload(bs); dnw.upload(nw);
        DevBuf<float> ws(mhc_workspace_bytes(T) / sizeof(float)), coef_a((size_t) T * kHcCoef), coef_b((size_t) T * kHcCoef);
        DevBuf<float> xin((size_t) T * H), xn_a((size_t) T * H), xn_b((size_t) T * H);
        const size_t qb = bk::native_q8_1_bytes(H, T);
        DevBuf<uint8_t> q_a(qb), q_b(qb);
        DevBuf<unsigned> cnt(8);
        cnt.zero();
        cudaStream_t st = nullptr;
        CUDA_CHECK(cudaStreamCreate(&st));
        mhc_mixes(dx.ptr, dfn.ptr, dsc.ptr, dbs.ptr, T, H, rms_eps, hc_eps, 20, ws.ptr, coef_a.ptr, st);
        mhc_pre(dx.ptr, coef_a.ptr, kHcCoef, T, H, xin.ptr, st);
        bk::rms_norm_rows(xin.ptr, dnw.ptr, T, H, rms_eps, xn_a.ptr, st);
        bk::native_quantize_q8_1(xn_a.ptr, q_a.ptr, H, T, st);
        for (int rep = 0; rep < 2; ++rep)   // twice: the counters are left zeroed
            mhc_pre_norm(dx.ptr, dfn.ptr, dsc.ptr, dbs.ptr, T, H, rms_eps, hc_eps, 20, dnw.ptr, rms_eps, ws.ptr, cnt.ptr,
                         coef_b.ptr, xn_b.ptr, q_b.ptr, st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        std::snprintf(name, sizeof name, "mhc_pre_norm coef+x_norm+q8 (T=%d)", T);
        check_bits(name, same_bits(coef_a, coef_b, coef_a.count) && same_bits(xn_a, xn_b, xn_a.count) &&
                             same_bits(q_a, q_b, qb));

        // mhc_post with the routed output added
        auto f = rand_vec((size_t) T * H, -1.f, 1.f, 21), f2 = rand_vec((size_t) T * H, -1.f, 1.f, 22);
        DevBuf<float> df(f.size()), df2(f2.size()), out_a((size_t) T * flat), out_b((size_t) T * flat);
        df.upload(f); df2.upload(f2);
        mhc_post(df.ptr, dx.ptr, coef_a.ptr, kHcCoef, T, H, out_b.ptr, st, df2.ptr);
        add_f32(df.ptr, df2.ptr, (int64_t) T * H, st);
        mhc_post(df.ptr, dx.ptr, coef_a.ptr, kHcCoef, T, H, out_a.ptr, st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        std::snprintf(name, sizeof name, "mhc_post with f2 (T=%d)", T);
        check_bits(name, same_bits(out_a, out_b, out_a.count));

        // rms_norm_q8, plain and over the weighted sum
        for (int cols : {1024, 2048}) {
            const int K = 10;
            auto v = rand_vec((size_t) T * K * cols, -3.f, 3.f, 31 + cols), wv = rand_vec((size_t) T * K, -1.f, 1.f, 32);
            for (size_t i = 0; i < wv.size(); i += 4) wv[i] = 0.0f;
            auto lw = rand_vec(cols, 0.5f, 1.5f, 33);
            DevBuf<float> dv(v.size()), dw(wv.size()), dl(cols), sum((size_t) T * cols), nrm((size_t) T * cols);
            dv.upload(v); dw.upload(wv); dl.upload(lw);
            const size_t qn = bk::native_q8_1_bytes(cols, T);
            DevBuf<uint8_t> qa(qn), qb2(qn), qc(qn), qd(qn);
            bk::rms_norm_rows(dv.ptr, dl.ptr, T, cols, rms_eps, nrm.ptr, st);
            bk::native_quantize_q8_1(nrm.ptr, qa.ptr, cols, T, st);
            rms_norm_q8(dv.ptr, dl.ptr, T, cols, rms_eps, nullptr, qb2.ptr, st);
            weighted_sum_f32(dv.ptr, dw.ptr, T, K, cols, sum.ptr, st);
            bk::rms_norm_rows(sum.ptr, dl.ptr, T, cols, rms_eps, nrm.ptr, st);
            bk::native_quantize_q8_1(nrm.ptr, qc.ptr, cols, T, st);
            rms_norm_q8(dv.ptr, dl.ptr, T, cols, rms_eps, nullptr, qd.ptr, st, dw.ptr, K);
            CUDA_CHECK(cudaStreamSynchronize(st));
            std::snprintf(name, sizeof name, "rms_norm_q8 (T=%d, %d cols)", T, cols);
            check_bits(name, same_bits(qa, qb2, qn));
            std::snprintf(name, sizeof name, "rms_norm_q8 weighted sum (T=%d, %d cols)", T, cols);
            check_bits(name, same_bits(qc, qd, qn));
        }

        // situ_glu_q8
        {
            const int F = 2048;
            auto g = rand_vec((size_t) T * F, -8.f, 8.f, 41), u = rand_vec((size_t) T * F, -30.f, 30.f, 42);
            DevBuf<float> dg(g.size()), du(u.size()), hh(g.size());
            dg.upload(g); du.upload(u);
            const size_t qn = bk::native_q8_1_bytes(F, T);
            DevBuf<uint8_t> qa(qn), qb2(qn);
            situ_glu_f32(dg.ptr, du.ptr, T, F, 4.0f, 25.0f, hh.ptr, st);
            bk::native_quantize_q8_1(hh.ptr, qa.ptr, F, T, st);
            situ_glu_q8(dg.ptr, du.ptr, T, F, 4.0f, 25.0f, qb2.ptr, st);
            CUDA_CHECK(cudaStreamSynchronize(st));
            std::snprintf(name, sizeof name, "situ_glu_q8 (T=%d)", T);
            check_bits(name, same_bits(qa, qb2, qn));
        }

        // merge_gate_q8 (8 groups of 1024; one group of 16384)
        for (int G : {8, 1}) {
            const int gout = G == 8 ? 1024 : 16384;
            auto src = rand_vec((size_t) T * G * gout, -2.f, 2.f, 51 + G), gate = rand_vec((size_t) T * G * gout, -4.f, 4.f, 52);
            DevBuf<float> ds(src.size()), dg(gate.size()), z(src.size());
            ds.upload(src); dg.upload(gate);
            const size_t qn = bk::native_q8_1_bytes(G * gout, T);
            DevBuf<uint8_t> qa(qn), qb2(qn);
            if (G > 1) hca_groups_merge_f32(ds.ptr, T, G, gout, z.ptr, st);
            else CUDA_CHECK(cudaMemcpyAsync(z.ptr, ds.ptr, src.size() * sizeof(float), cudaMemcpyDeviceToDevice, st));
            sigmoid_gate_mul_f32(z.ptr, dg.ptr, (int64_t) T * G * gout, st);
            bk::native_quantize_q8_1(z.ptr, qa.ptr, G * gout, T, st);
            merge_gate_q8(ds.ptr, dg.ptr, T, G, gout, qb2.ptr, st);
            CUDA_CHECK(cudaStreamSynchronize(st));
            std::snprintf(name, sizeof name, "merge_gate_q8 (T=%d, %d groups)", T, G);
            check_bits(name, same_bits(qa, qb2, qn));
        }

        // hca_post_rope_q8 and hca_qkv_rope_commit
        {
            const int NH = 64, HD = 512, G = 8, hpg = NH / G, gin = hpg * HD;
            const RopeTable rt = rope_table_yarn(160000.0f, 16.0f, 32.0f, 1.0f, 65536);
            std::vector<int32_t> pos(T);
            for (int t = 0; t < T; ++t) pos[t] = 1000 + t;
            DevBuf<int32_t> dpos(T);
            dpos.upload(pos);
            auto in = rand_vec((size_t) T * NH * HD, -1.f, 1.f, 61), mix = rand_vec((size_t) NH * NH, -0.2f, 0.2f, 62);
            DevBuf<float> din(in.size()), dmix(mix.size()), o2(in.size()), o3(in.size());
            din.upload(in); dmix.upload(mix);
            const size_t qn = bk::native_q8_1_bytes(gin, 1) * G * T;
            DevBuf<uint8_t> qa(qn), qb2(qn);
            vha_postmix_f32(din.ptr, dmix.ptr, T, NH, HD, o2.ptr, st);
            hca_inv_rope(o2.ptr, dpos.ptr, T, NH, HD, rt, st);
            hca_groups_split_f32(o2.ptr, T, G, gin, o3.ptr, st);
            for (int c0 = 0; c0 < G * T; c0 += 8)
                bk::native_quantize_q8_1(o3.ptr + (size_t) c0 * gin, qa.ptr + (size_t) c0 * bk::native_q8_1_bytes(gin, 1), gin,
                                         std::min(8, G * T - c0), st);
            hca_post_rope_q8(din.ptr, dmix.ptr, dpos.ptr, T, NH, HD, hpg, rt, qb2.ptr, st);
            CUDA_CHECK(cudaStreamSynchronize(st));
            std::snprintf(name, sizeof name, "hca_post_rope_q8 (T=%d)", T);
            check_bits(name, same_bits(qa, qb2, qn));

            const int win = 128, cap = 256;
            auto q = rand_vec((size_t) T * NH * HD, -1.f, 1.f, 71), kv = rand_vec((size_t) T * HD, -1.f, 1.f, 72);
            auto kvw = rand_vec(HD, 0.5f, 1.5f, 73);
            for (bool fp8 : {true, false}) {
                const size_t row = fp8 ? 608 : HD * 2;
                DevBuf<float> qa1(q.size()), qb1(q.size()), kva(kv.size()), kvb(kv.size()), dkvw(HD);
                DevBuf<uint8_t> wa(cap * row), wb(cap * row);
                qa1.upload(q); qb1.upload(q); kva.upload(kv); kvb.upload(kv); dkvw.upload(kvw);
                wa.zero(); wb.zero();
                hca_q_rope(qa1.ptr, dpos.ptr, T, NH, HD, rms_eps, rt, st);
                hca_kv_norm_rope(kva.ptr, dkvw.ptr, dpos.ptr, T, HD, rms_eps, rt, kva.ptr, st);
                hca_win_commit(kva.ptr, dpos.ptr, T, HD, win, wa.ptr, fp8, st, cap);
                hca_qkv_rope_commit(qb1.ptr, kvb.ptr, dkvw.ptr, dpos.ptr, T, NH, HD, rms_eps, rt, kvb.ptr, cap, wb.ptr, fp8, st);
                CUDA_CHECK(cudaStreamSynchronize(st));
                std::snprintf(name, sizeof name, "hca_qkv_rope_commit (T=%d, %s)", T, fp8 ? "fp8" : "f16");
                const bool sq = same_bits(qa1, qb1, q.size()), sk = same_bits(kva, kvb, kv.size()), sw = same_bits(wa, wb, cap * row);
                if (!sq || !sk || !sw) std::printf("  q %d kv %d window %d\n", (int) sq, (int) sk, (int) sw);
                if (!sq) {
                    std::vector<float> ha(q.size()), hb(q.size());
                    qa1.download(ha); qb1.download(hb);
                    int nd = 0, first = -1;
                    for (size_t i = 0; i < ha.size(); ++i) if (std::memcmp(&ha[i], &hb[i], 4)) { if (first < 0) first = (int) i; ++nd; }
                    std::printf("  %d differ, first %d: %.9g %.9g orig %.9g\n", nd, first, ha[first], hb[first], q[first]);
                }
                check_bits(name, sq && sk && sw);
            }
        }
        CUDA_CHECK(cudaStreamDestroy(st));
    }
}

void test_prefill_fast_kernels() {
    // 1. dequant_q8_0_to_f16 + dsa_pack_q_nope_f16 + dsa_mla_attention_preabsorbed (512/64/192)
    const int N = 12, NH = 16, qk_nope = 192, qk_rope = 64, kv_lora = 512, v_dim = 256, sel_w = N;
    auto qf = rand_vec((size_t) N * NH * (qk_nope + qk_rope), -0.4f, 0.4f, 501);
    auto kv_f = rand_vec((size_t) N * (kv_lora + qk_rope), -0.4f, 0.4f, 502);
    auto kvn = rand_vec(kv_lora, 0.9f, 1.1f, 503);
    auto sink = rand_vec(NH, -1.0f, 1.0f, 504);
    auto kvb_f = rand_vec((size_t) NH * (qk_nope + v_dim) * kv_lora, -0.25f, 0.25f, 505);
    std::vector<uint8_t> kvb_q8;
    quantize_q8_0(kvb_f, NH * (qk_nope + v_dim), kv_lora, kvb_q8);
    std::vector<double> kvb_deq = dequantize_q8_0(kvb_q8, NH * (qk_nope + v_dim), kv_lora);

    std::vector<int32_t> pos(N), sel((size_t) N * sel_w, -1);
    for (int t = 0; t < N; ++t) {
        pos[t] = t;
        for (int j = 0; j <= t; ++j) sel[(size_t) t * sel_w + j] = j;
    }

    DevBuf<float> dq(qf.size()), dkv(kv_f.size()), dkvn(kv_lora), dsink(NH);
    DevBuf<int32_t> dpos(N), dsel(sel.size());
    DevBuf<uint8_t> dkv_c((size_t) N * (kv_lora + qk_rope) * 2), dkvb(kvb_q8.size());
    DevBuf<uint16_t> dkvb_f16(kvb_f.size()), dqn_f16((size_t) NH * N * qk_nope), dolat_f16((size_t) NH * N * kv_lora);
    DevBuf<float> dqlat((size_t) NH * N * kv_lora), dout_ref((size_t) N * NH * v_dim);
    dq.upload(qf); dkv.upload(kv_f); dkvn.upload(kvn); dsink.upload(sink); dpos.upload(pos); dsel.upload(sel); dkvb.upload(kvb_q8);

    const RopeTable rt = rope_table_plain(100000.0f);
    dsa_qkv_rope_and_store(dq.ptr, dkv.ptr, dkvn.ptr, dpos.ptr, N, NH, qk_nope, qk_rope, kv_lora, 1e-6f, rt, dkv_c.ptr,
                           false, nullptr);
    dsa_mla_attention(dq.ptr, dkv_c.ptr, dkvb.ptr, dsink.ptr, dsel.ptr, N, NH, qk_nope, qk_rope, kv_lora, v_dim, sel_w,
                      false, dout_ref.ptr, nullptr);
    dequant_q8_0_to_f16(dkvb.ptr, dkvb_f16.ptr, (int64_t) kvb_f.size() / 32, nullptr);
    dsa_pack_q_nope_f16(dq.ptr, dqn_f16.ptr, N, NH, qk_nope, qk_rope, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> hkvb16(kvb_f.size()), hqn16((size_t) NH * N * qk_nope);
    dkvb_f16.download(hkvb16);
    dqn_f16.download(hqn16);
    std::vector<float> hkvb_got(kvb_f.size());
    for (size_t i = 0; i < kvb_f.size(); ++i) hkvb_got[i] = __half2float(__ushort_as_half(hkvb16[i]));
    check_tol("dequant_q8_0_to_f16", rel_l2_err(hkvb_got, kvb_deq), 1e-3);

    std::vector<float> hqlat((size_t) NH * N * kv_lora, 0.0f);
    for (int h = 0; h < NH; ++h) {
        const float* w_uk = hkvb_got.data() + (size_t) h * (qk_nope + v_dim) * kv_lora;
        for (int t = 0; t < N; ++t) {
            float* ql = hqlat.data() + ((size_t) h * N + t) * kv_lora;
            for (int r = 0; r < qk_nope; ++r) {
                const float qv = __half2float(__ushort_as_half(hqn16[((size_t) h * N + t) * qk_nope + r]));
                for (int c = 0; c < kv_lora; ++c) ql[c] += qv * w_uk[(size_t) r * kv_lora + c];
            }
        }
    }
    dqlat.upload(hqlat);
    dsa_mla_attention_preabsorbed(dqlat.ptr, dq.ptr, dkv_c.ptr, dsink.ptr, dsel.ptr, N, NH, qk_nope, qk_rope, kv_lora,
                                  sel_w, false, dolat_f16.ptr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> holat16((size_t) NH * N * kv_lora);
    std::vector<float> href_out((size_t) N * NH * v_dim);
    dolat_f16.download(holat16);
    dout_ref.download(href_out);
    std::vector<float> hgot_out((size_t) N * NH * v_dim, 0.0f);
    std::vector<double> href_d(href_out.begin(), href_out.end());
    for (int h = 0; h < NH; ++h) {
        const float* w_uv = hkvb_got.data() + ((size_t) h * (qk_nope + v_dim) + qk_nope) * kv_lora;
        for (int t = 0; t < N; ++t) {
            for (int r = 0; r < v_dim; ++r) {
                float acc = 0.0f;
                for (int c = 0; c < kv_lora; ++c) {
                    const float ov = __half2float(__ushort_as_half(holat16[((size_t) h * N + t) * kv_lora + c]));
                    acc += w_uv[(size_t) r * kv_lora + c] * ov;
                }
                hgot_out[((size_t) t * NH + h) * v_dim + r] = acc;
            }
        }
    }
    check_tol("dsa_mla_attention_preabsorbed (512/64/192)", rel_l2_err(hgot_out, href_d), 2e-3);

    // 1b. Same prefill-vs-decode MLA drift check with FP8 KV cache (the default runtime cache format)
    {
        DevBuf<uint8_t> dkv_fp8((size_t) N * layout::kDsaFp8RowBytes);
        DevBuf<float> dq_fp8(qf.size()), dout_fp8_ref((size_t) N * NH * v_dim);
        DevBuf<uint16_t> dqn_fp8((size_t) NH * N * qk_nope), dolat_fp8((size_t) NH * N * kv_lora);
        dq_fp8.upload(qf);
        dsa_qkv_rope_and_store(dq_fp8.ptr, dkv.ptr, dkvn.ptr, dpos.ptr, N, NH, qk_nope, qk_rope, kv_lora, 1e-6f, rt,
                               dkv_fp8.ptr, true, nullptr);
        dsa_mla_attention(dq_fp8.ptr, dkv_fp8.ptr, dkvb.ptr, dsink.ptr, dsel.ptr, N, NH, qk_nope, qk_rope, kv_lora,
                          v_dim, sel_w, true, dout_fp8_ref.ptr, nullptr);
        dsa_mla_attention_preabsorbed(dqlat.ptr, dq_fp8.ptr, dkv_fp8.ptr, dsink.ptr, dsel.ptr, N, NH, qk_nope, qk_rope,
                                      kv_lora, sel_w, true, dolat_fp8.ptr, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint16_t> holat_fp8((size_t) NH * N * kv_lora);
        std::vector<float> href_fp8((size_t) N * NH * v_dim), hgot_fp8((size_t) N * NH * v_dim, 0.0f);
        dolat_fp8.download(holat_fp8);
        dout_fp8_ref.download(href_fp8);
        std::vector<double> href_fp8_d(href_fp8.begin(), href_fp8.end());
        for (int h = 0; h < NH; ++h) {
            const float* w_uv = hkvb_got.data() + ((size_t) h * (qk_nope + v_dim) + qk_nope) * kv_lora;
            for (int t = 0; t < N; ++t) {
                for (int r = 0; r < v_dim; ++r) {
                    float acc = 0.0f;
                    for (int c = 0; c < kv_lora; ++c) {
                        const float ov = __half2float(__ushort_as_half(holat_fp8[((size_t) h * N + t) * kv_lora + c]));
                        acc += w_uv[(size_t) r * kv_lora + c] * ov;
                    }
                    hgot_fp8[((size_t) t * NH + h) * v_dim + r] = acc;
                }
            }
        }
        check_tol("dsa_mla_attention_preabsorbed (FP8 KV)", rel_l2_err(hgot_fp8, href_fp8_d), 2e-3);
    }

    // 1c. Prompt SGEMM router (bf16_widen_f32 + F32 matmul) vs decode BF16 GEMV router (bf16_gemv_fp32_mmvf_cols)
    {
        const int RT = 4, RH = 512, RN = 2 * layout::kNumExperts;
        auto rx = rand_vec((size_t) RT * RH, -1.0f, 1.0f, 521);
        auto rw_f = rand_vec((size_t) RN * RH, -0.2f, 0.2f, 522);
        std::vector<uint16_t> rw_bf16(rw_f.size());
        for (size_t i = 0; i < rw_f.size(); ++i) rw_bf16[i] = __bfloat16_as_ushort(__float2bfloat16(rw_f[i]));
        DevBuf<float> drx(rx.size()), drw_wide(rw_f.size()), dry_dec((size_t) RT * RN);
        DevBuf<uint16_t> drw(rw_bf16.size());
        drx.upload(rx);
        drw.upload(rw_bf16);
        bf16_widen_f32(drw.ptr, drw_wide.ptr, (int64_t) rw_f.size(), nullptr);
        basalt::kernels::bf16_gemv_fp32_mmvf_cols(drx.ptr, drw.ptr, dry_dec.ptr, RH, RN, RT, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> hw_wide(rw_f.size()), hy_dec((size_t) RT * RN);
        drw_wide.download(hw_wide);
        dry_dec.download(hy_dec);
        std::vector<double> hy_sgemm((size_t) RT * RN, 0.0);
        for (int t = 0; t < RT; ++t) {
            for (int o = 0; o < RN; ++o) {
                double acc = 0.0;
                for (int k = 0; k < RH; ++k) acc += (double) rx[(size_t) t * RH + k] * (double) hw_wide[(size_t) o * RH + k];
                hy_sgemm[(size_t) t * RN + o] = acc;
            }
        }
        check_tol("router prompt SGEMM vs decode BF16 GEMV", rel_l2_err(hy_dec, hy_sgemm), 1e-5);
    }

    // 2. vha_postmix_f32 head_major=true vs head_major=false
    auto vha_mix = rand_vec((size_t) NH * NH, -0.3f, 0.3f, 506);
    std::vector<float> hgot_hm((size_t) NH * N * v_dim);
    for (int t = 0; t < N; ++t)
        for (int h = 0; h < NH; ++h)
            for (int d = 0; d < v_dim; ++d)
                hgot_hm[((size_t) h * N + t) * v_dim + d] = hgot_out[((size_t) t * NH + h) * v_dim + d];
    DevBuf<float> dvin(hgot_out.size()), dvhm(hgot_hm.size()), dvmix(vha_mix.size()), dv1(hgot_out.size()), dv2(hgot_out.size());
    dvin.upload(hgot_out); dvhm.upload(hgot_hm); dvmix.upload(vha_mix);
    vha_postmix_f32(dvin.ptr, dvmix.ptr, N, NH, v_dim, dv1.ptr, nullptr, false);
    vha_postmix_f32(dvhm.ptr, dvmix.ptr, N, NH, v_dim, dv2.ptr, nullptr, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hv1(hgot_out.size()), hv2(hgot_out.size());
    dv1.download(hv1); dv2.download(hv2);
    std::vector<double> hv1d(hv1.begin(), hv1.end());
    check_tol("vha_postmix_f32 (head_major=true)", rel_l2_err(hv2, hv1d), 1e-6);

    // 3. parallel hca_compress (pos0_hint >= 0) vs sequential (pos0_hint = -1), including unaligned start
    const int ratio = 128, hd = 512, T = 300, p0 = 37;
    auto ckv = rand_vec((size_t) T * hd, -0.5f, 0.5f, 507);
    auto csc = rand_vec((size_t) T * hd, -0.5f, 0.5f, 508);
    auto ape = rand_vec((size_t) ratio * hd, -0.2f, 0.2f, 509);
    auto cnw = rand_vec(hd, 0.8f, 1.2f, 510);
    std::vector<int32_t> hpos(T);
    for (int i = 0; i < T; ++i) hpos[i] = p0 + i;
    DevBuf<float> dckv(ckv.size()), dcsc(csc.size()), dape(ape.size()), dcnw(hd);
    DevBuf<float> skv1((size_t) ratio * hd), ssc1((size_t) ratio * hd), skv2((size_t) ratio * hd), ssc2((size_t) ratio * hd);
    DevBuf<int32_t> dp2(T);
    const size_t comp_bytes = 4 * (size_t) hd * 2;
    DevBuf<uint8_t> cc1(comp_bytes), cc2(comp_bytes);
    dckv.upload(ckv); dcsc.upload(csc); dape.upload(ape); dcnw.upload(cnw); dp2.upload(hpos);
    CUDA_CHECK(cudaMemset(skv1.ptr, 0, (size_t) ratio * hd * sizeof(float)));
    CUDA_CHECK(cudaMemset(ssc1.ptr, 0, (size_t) ratio * hd * sizeof(float)));
    CUDA_CHECK(cudaMemset(skv2.ptr, 0, (size_t) ratio * hd * sizeof(float)));
    CUDA_CHECK(cudaMemset(ssc2.ptr, 0, (size_t) ratio * hd * sizeof(float)));
    CUDA_CHECK(cudaMemset(cc1.ptr, 0, comp_bytes));
    CUDA_CHECK(cudaMemset(cc2.ptr, 0, comp_bytes));
    const RopeTable hrt = rope_table_yarn(160000.0f, 16.0f, 32.0f, 1.0f, 65536);
    hca_compress(dckv.ptr, dcsc.ptr, dape.ptr, dcnw.ptr, dp2.ptr, T, ratio, hd, 1e-6f, hrt, skv1.ptr, ssc1.ptr,
                 cc1.ptr, false, nullptr, -1);
    hca_compress(dckv.ptr, dcsc.ptr, dape.ptr, dcnw.ptr, dp2.ptr, T, ratio, hd, 1e-6f, hrt, skv2.ptr, ssc2.ptr,
                 cc2.ptr, false, nullptr, p0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> hcc1(comp_bytes), hcc2(comp_bytes);
    cc1.download(hcc1); cc2.download(hcc2);
    check_tol("hca_compress parallel vs sequential", std::memcmp(hcc1.data(), hcc2.data(), comp_bytes) == 0 ? 0.0 : 1.0,
              0.0);

    // 4. BF16 tiled Lightning Indexer (dsa_dequant_idx_k_bf16 + dot + dsa_indexer_reduce_tile + select_cached) vs fast128
    {
        const int TN = 48, IH = 64, ID = 128, win_i = 8, tk_i = 12, sw_i = tk_i + win_i, tile_k = 16;
        auto iq_v = rand_vec((size_t) TN * IH * ID, -0.5f, 0.5f, 601);
        auto ik_v = rand_vec((size_t) TN * ID, -0.5f, 0.5f, 602);
        auto iw_v = rand_vec((size_t) TN * IH, 0.1f, 1.0f, 603);
        auto knw = rand_vec(ID, 0.9f, 1.1f, 604), knb = rand_vec(ID, -0.1f, 0.1f, 605);
        std::vector<int32_t> pos_i(TN);
        for (int i = 0; i < TN; ++i) pos_i[i] = i;
        DevBuf<float> diq(iq_v.size()), dik(ik_v.size()), diw(iw_v.size()), dknw(ID), dknb(ID);
        DevBuf<float> dsc_ref((size_t) TN * TN), dsc_tc((size_t) TN * TN);
        DevBuf<int32_t> dpos_i(TN), dsel_ref((size_t) TN * sw_i), dsel_tc((size_t) TN * sw_i);
        DevBuf<uint8_t> dkcache((size_t) TN * 68);
        diq.upload(iq_v); dik.upload(ik_v); diw.upload(iw_v); dknw.upload(knw); dknb.upload(knb); dpos_i.upload(pos_i);
        dsa_indexer_prepare(diq.ptr, dik.ptr, dknw.ptr, dknb.ptr, dpos_i.ptr, TN, IH, ID, 64, 1e-6f, rt, dkcache.ptr,
                            true, nullptr);
        dsa_indexer_topk_merge(diq.ptr, diw.ptr, dkcache.ptr, dpos_i.ptr, TN, IH, ID, tk_i, win_i, true, dsel_ref.ptr,
                               nullptr, dsc_ref.ptr, TN);

        // Compute BF16 tiled scores (using dsa_dequant_idx_k_bf16 + host BF16 dot + dsa_indexer_reduce_tile)
        std::vector<float> hiq_prep(iq_v.size());
        diq.download(hiq_prep);
        DevBuf<uint16_t> dk_bf16((size_t) tile_k * ID);
        DevBuf<float> ddots((size_t) TN * IH * tile_k);
        CUDA_CHECK(cudaMemset(dsc_tc.ptr, 0, (size_t) TN * TN * sizeof(float)));
        const int max_old = TN - win_i;
        for (int k0 = 0; k0 < max_old; k0 += tile_k) {
            const int kn = std::min(tile_k, max_old - k0);
            dsa_dequant_idx_k_bf16(dkcache.ptr, k0, kn, ID, true, dk_bf16.ptr, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<uint16_t> hk16((size_t) tile_k * ID);
            dk_bf16.download(hk16);
            std::vector<float> hdots((size_t) TN * IH * kn, 0.0f);
            for (int t = 0; t < TN; ++t) {
                for (int h = 0; h < IH; ++h) {
                    const float* qrow = hiq_prep.data() + ((size_t) t * IH + h) * ID;
                    for (int j = 0; j < kn; ++j) {
                        float acc = 0.0f;
                        for (int d = 0; d < ID; ++d) {
                            const float qv = __bfloat162float(__float2bfloat16(qrow[d]));
                            const float kv = __bfloat162float(__ushort_as_bfloat16(hk16[(size_t) j * ID + d]));
                            acc += qv * kv;
                        }
                        hdots[((size_t) t * IH + h) * kn + j] = acc;
                    }
                }
            }
            CUDA_CHECK(cudaMemcpy(ddots.ptr, hdots.data(), hdots.size() * sizeof(float), cudaMemcpyHostToDevice));
            dsa_indexer_reduce_tile(ddots.ptr, diw.ptr, dpos_i.ptr, TN, IH, ID, tk_i, win_i, k0, kn, dsc_tc.ptr, TN,
                                    nullptr);
        }
        dsa_indexer_select_cached(dsc_tc.ptr, TN, dpos_i.ptr, TN, tk_i, win_i, dsel_tc.ptr, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> hsc_ref((size_t) TN * TN), hsc_tc((size_t) TN * TN);
        dsc_ref.download(hsc_ref); dsc_tc.download(hsc_tc);
        std::vector<double> ref_valid;
        std::vector<float> got_valid;
        for (int t = win_i + tk_i; t < TN; ++t) {
            for (int j = 0; j < t + 1 - win_i; ++j) {
                ref_valid.push_back(hsc_ref[(size_t) t * TN + j]);
                got_valid.push_back(hsc_tc[(size_t) t * TN + j]);
            }
        }
        check_tol("dsa_indexer BF16 tiled vs fast128 (64 heads, MXFP4)", rel_l2_err(got_valid, ref_valid), 5e-3);
    }

    // 7. Deterministic index-order compaction & lowest-index tie breaking in dsa_indexer_select_cached
    {
        const int older_n = 700, win_s = 16, tk_s = 32, width = tk_s + win_s;
        const int seq_len = older_n + win_s;
        std::vector<float> scores((size_t) seq_len, -10.0f);
        const std::vector<int> gt_idx = {3, 19, 100, 255, 256, 300, 511, 512, 600, 699};
        for (size_t i = 0; i < gt_idx.size(); ++i) scores[(size_t) gt_idx[i]] = 5.0f + (float) i * 0.25f;
        std::vector<int> eq_idx;
        for (int j = 5; j < older_n; j += 13) {
            if (scores[(size_t) j] < 0.0f) {
                scores[(size_t) j] = 2.5f;
                eq_idx.push_back(j);
            }
        }
        std::vector<int32_t> expected(width, -1);
        int out_p = 0;
        for (int j : gt_idx) expected[out_p++] = j;
        for (int j : eq_idx) {
            if (out_p >= tk_s) break;
            expected[out_p++] = j;
        }
        for (int i = 0; i < win_s; ++i) expected[tk_s + i] = older_n + i;

        DevBuf<float> dsc(seq_len);
        DevBuf<int32_t> dpos(1), dsel1(width), dsel2(width);
        dsc.upload(scores);
        dpos.upload(std::vector<int32_t>{seq_len - 1});
        dsa_indexer_select_cached(dsc.ptr, seq_len, dpos.ptr, 1, tk_s, win_s, dsel1.ptr, nullptr);
        dsa_indexer_select_cached(dsc.ptr, seq_len, dpos.ptr, 1, tk_s, win_s, dsel2.ptr, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int32_t> got1(width), got2(width);
        dsel1.download(got1);
        dsel2.download(got2);
        const bool ok = (got1 == expected) && (got2 == expected);
        check_tol("dsa_indexer_select_cached deterministic order & tie-break", ok ? 0.0 : 1.0, 0.0);
    }
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::fprintf(stderr, "No CUDA device available\n");
        return 77;
    }
    test_mhc();
    test_elementwise_and_router();
    test_hca();
    test_mla_dsa();
    test_dsa_split();
    test_fused();
    test_prefill_fast_kernels();
    std::printf("ernie55_kernels_parity: %d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
