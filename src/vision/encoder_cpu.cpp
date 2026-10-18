// src/vision/encoder_cpu.cpp - see basalt/vision/encoder_cpu.hpp.
#include "basalt/vision/encoder_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

#if defined(_MSC_VER)
#include <immintrin.h>
#endif

#include "gemm_cpu.hpp"

namespace basalt::vision {

namespace {

// The fp32 -> half -> fp32 round trip the reference quantizes with. Where the compiler has a half type it does the
// conversion; MSVC has none, and its F16C intrinsic pair is the same instruction with the same rounding.
#if defined(_MSC_VER)
float f16_round(float x) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph(_mm_set_ss(x), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
}
#else
float f16_round(float x) { return (float) (_Float16) x; }
#endif

void parallel_for(int threads, int n, const std::function<void(int, int)>& fn) {
    threads = std::max(1, std::min(threads, n));
    if (threads == 1) { fn(0, n); return; }
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t) {
        const int a = (int) ((int64_t) n * t / threads), b = (int) ((int64_t) n * (t + 1) / threads);
        ts.emplace_back([=, &fn] { fn(a, b); });
    }
    for (auto& t : ts) t.join();
}

float gelu_exact(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
}

// the reference's table version: fp16 in, fp16 out, flat outside +-10
float gelu_f16(float x) {
    if (x <= -10.f) return 0.f;
    if (x >= 10.f) return x;
    return f16_round(gelu_exact(f16_round(x)));
}

void layer_norm(const float* x, const float* w, const float* b, float* y, int rows, int n, float eps, int threads) {
    parallel_for(threads, rows, [&](int r0, int r1) {
        for (int r = r0; r < r1; ++r) {
            const float* xr = x + (size_t) r * n;
            float* yr = y + (size_t) r * n;
            double sum = 0;
            for (int i = 0; i < n; ++i) sum += xr[i];
            const float mean = (float) (sum / n);
            double sq = 0;
            for (int i = 0; i < n; ++i) { const float d = xr[i] - mean; sq += (double) d * d; }
            const float scale = 1.0f / sqrtf((float) (sq / n) + eps);
            for (int i = 0; i < n; ++i) yr[i] = (xr[i] - mean) * scale * w[i] + b[i];
        }
    });
}

}  // namespace

bool CpuEncoder::init(const Mmproj& mp, int threads, bool round_bf16, bool f16_quirks, std::string& err) {
    f16_ = f16_quirks;
    mp_ = &mp;
    threads_ = threads > 0 ? threads : (int) std::max(1u, std::thread::hardware_concurrency());
    round_ = round_bf16;
    const Hparams& h = mp.hp();
    if (h.embd != 1152 || h.heads != 16 || h.patch != 16 || h.ffn != 4304 || h.merge != 2 || h.layers < 1) {
        err = "unexpected projector hyper-parameters";
        return false;
    }
    const int64_t E = h.embd, F = h.ffn;
    auto f32 = [&](const std::string& n, int64_t e) { const auto* t = mp.need(n, 0, e, err); return t ? (const float*) t->data : nullptr; };
    auto b16 = [&](const std::string& n, int64_t e) { const auto* t = mp.need(n, 30, e, err); return t ? (const uint16_t*) t->data : nullptr; };
    const float *w0 = f32("v.patch_embd.weight", E * 768), *w1 = f32("v.patch_embd.weight.1", E * 768);
    bpatch_ = f32("v.patch_embd.bias", E);
    pos_ = f32("v.position_embd.weight", E * 2304);
    postw_ = f32("v.post_ln.weight", E);
    postb_ = f32("v.post_ln.bias", E);
    bm0_ = f32("mm.0.bias", 4 * E);
    bm2_ = f32("mm.2.bias", h.proj);
    wm0_ = b16("mm.0.weight", 4 * E * 4 * E);
    wm2_ = b16("mm.2.weight", 4 * E * h.proj);
    if (!w0 || !w1 || !bpatch_ || !pos_ || !postw_ || !postb_ || !bm0_ || !bm2_ || !wm0_ || !wm2_) return false;
    wpatch_.resize((size_t) E * 768);
    for (size_t i = 0; i < wpatch_.size(); ++i) wpatch_[i] = w0[i] + w1[i];
    layers_.resize(h.layers);
    for (int l = 0; l < h.layers; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        Layer& L = layers_[l];
        L.ln1w = f32(p + "ln1.weight", E);
        L.ln1b = f32(p + "ln1.bias", E);
        L.ln2w = f32(p + "ln2.weight", E);
        L.ln2b = f32(p + "ln2.bias", E);
        L.bqkv = f32(p + "attn_qkv.bias", 3 * E);
        L.bo = f32(p + "attn_out.bias", E);
        L.bup = f32(p + "ffn_up.bias", F);
        L.bdown = f32(p + "ffn_down.bias", E);
        L.wqkv = b16(p + "attn_qkv.weight", 3 * E * E);
        L.wo = b16(p + "attn_out.weight", E * E);
        L.wup = b16(p + "ffn_up.weight", F * E);
        L.wdown = b16(p + "ffn_down.weight", E * F);
        if (!L.ln1w || !L.ln1b || !L.ln2w || !L.ln2b || !L.bqkv || !L.bo || !L.bup || !L.bdown || !L.wqkv || !L.wo ||
            !L.wup || !L.wdown)
            return false;
    }
    return true;
}

bool CpuEncoder::encode(const std::vector<float>& patches, Size canvas, std::vector<float>& out, const TapFn& tap) {
    const Hparams& hp = mp_->hp();
    const int E = hp.embd, F = hp.ffn, H = hp.heads, D = E / H, gx = canvas.w / 16, gy = canvas.h / 16;
    const int n = gx * gy;
    if (n % 4 != 0 || (int) patches.size() != n * 768) return false;
    std::vector<float> x((size_t) n * E), h((size_t) n * E), qkv((size_t) n * 3 * E), att((size_t) n * E),
        tmp((size_t) n * E), ffn((size_t) n * F), attk((size_t) H * n * D), attv((size_t) H * D * n);
    const int T = threads_;

    // patch embedding + bilinear position embedding, rows in merge order
    std::vector<float> p16(patches);
    if (f16_) for (float& v : p16) v = f16_round(v);
    gemm_f32w(p16.data(), wpatch_.data(), x.data(), n, E, 768, T);
    {
        const int S = 48;
        auto src = [&](int axis_out, int axis_in, int i, int& i0, int& i1, float& d) {
            float sf = (float) axis_out / axis_in;
            if (axis_out > 1 && axis_in > 1) sf = (float) (axis_out - 1) / (axis_in - 1);
            const float v = (float) i / sf;
            int64_t a = (int64_t) floorf(v), b = a + 1;
            a = std::max<int64_t>(0, std::min<int64_t>(a, axis_in - 1));
            b = std::max<int64_t>(0, std::min<int64_t>(b, axis_in - 1));
            d = std::max(0.0f, std::min(v - (float) a, 1.0f));
            i0 = (int) a;
            i1 = (int) b;
        };
        parallel_for(T, n, [&](int t0, int t1) {
            for (int t = t0; t < t1; ++t) {
                const int g = t / 4, r = t % 4;
                const int py = (g / (gx / 2)) * 2 + r / 2, px = (g % (gx / 2)) * 2 + r % 2;
                int x0, x1, y0, y1;
                float dx, dy;
                src(gx, S, px, x0, x1, dx);
                src(gy, S, py, y0, y1, dy);
                float* row = &x[(size_t) t * E];
                for (int c = 0; c < E; ++c) {
                    const float a = pos_[((size_t) y0 * S + x0) * E + c], b = pos_[((size_t) y0 * S + x1) * E + c],
                                cc = pos_[((size_t) y1 * S + x0) * E + c], d = pos_[((size_t) y1 * S + x1) * E + c];
                    const float pe = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + cc * (1 - dx) * dy + d * dx * dy;
                    row[c] += bpatch_[c] + pe;
                }
            }
        });
    }
    if (tap) tap("embed", x.data(), x.size());

    // rotation tables per patch: 36 cos/sin pairs, rows in merge order
    std::vector<float> rc((size_t) n * 36), rs((size_t) n * 36);
    {
        const float ts = powf(10000.0f, -2.0f / 36);
        for (int t = 0; t < n; ++t) {
            const int g = t / 4, r = t % 4;
            const int py = (g / (gx / 2)) * 2 + r / 2, px = (g % (gx / 2)) * 2 + r % 2;
            float th = (float) py;
            for (int j = 0; j < 36; ++j) {
                if (j == 18) th = (float) px;
                rc[(size_t) t * 36 + j] = cosf(th);
                rs[(size_t) t * 36 + j] = sinf(th);
                th *= ts;
            }
        }
    }
    const float kq = 1.0f / sqrtf((float) D);
    for (int l = 0; l < hp.layers; ++l) {
        const Layer& L = layers_[l];
        layer_norm(x.data(), L.ln1w, L.ln1b, h.data(), n, E, hp.eps, T);
        gemm_bf16w(h.data(), L.wqkv, qkv.data(), n, 3 * E, E, T, round_);
        parallel_for(T, n, [&](int t0, int t1) {
            for (int t = t0; t < t1; ++t) {
                float* r = &qkv[(size_t) t * 3 * E];
                for (int i = 0; i < 3 * E; ++i) r[i] += L.bqkv[i];
                for (int part = 0; part < 2; ++part)
                    for (int hd = 0; hd < H; ++hd) {
                        float* v = r + part * E + hd * D;
                        for (int j = 0; j < 36; ++j) {
                            const float c = rc[(size_t) t * 36 + j], s = rs[(size_t) t * 36 + j];
                            const float a = v[j], b = v[j + 36];
                            v[j] = a * c - b * s;
                            v[j + 36] = a * s + b * c;
                        }
                    }
            }
        });
        // attention: every token over every token; per head, blocks of 64 queries against all keys as two products
        {
            const int QB = 64, nqb = (n + QB - 1) / QB;
            std::vector<float>& kh = attk;
            std::vector<float>& vt = attv;
            parallel_for(T, H, [&](int h0, int h1) {
                for (int hd = h0; hd < h1; ++hd)
                    for (int t = 0; t < n; ++t) {
                        const float* r = &qkv[(size_t) t * 3 * E + hd * D];
                        std::memcpy(&kh[((size_t) hd * n + t) * D], r + E, D * sizeof(float));
                        for (int i = 0; i < D; ++i) vt[((size_t) hd * D + i) * n + t] = r[2 * E + i];
                    }
            });
            parallel_for(T, H * nqb, [&](int a0, int a1) {
                std::vector<float> qc((size_t) QB * D), sc((size_t) QB * n), oc((size_t) QB * D);
                for (int a = a0; a < a1; ++a) {
                    const int hd = a / nqb, q0 = (a % nqb) * QB, rows = std::min(QB, n - q0);
                    for (int r = 0; r < rows; ++r) std::memcpy(&qc[(size_t) r * D], &qkv[(size_t) (q0 + r) * 3 * E + hd * D], D * sizeof(float));
                    gemm_f32w(qc.data(), &kh[(size_t) hd * n * D], sc.data(), rows, n, D, 1);
                    for (int r = 0; r < rows; ++r) {
                        float* s = &sc[(size_t) r * n];
                        float mx = -INFINITY;
                        for (int u = 0; u < n; ++u) { s[u] *= kq; mx = std::max(mx, s[u]); }
                        float sum = 0;
                        for (int u = 0; u < n; ++u) { s[u] = expf(s[u] - mx); sum += s[u]; }
                        const float inv = 1.0f / sum;
                        for (int u = 0; u < n; ++u) s[u] *= inv;
                    }
                    gemm_f32w(sc.data(), &vt[(size_t) hd * D * n], oc.data(), rows, D, n, 1);
                    for (int r = 0; r < rows; ++r) std::memcpy(&att[(size_t) (q0 + r) * E + hd * D], &oc[(size_t) r * D], D * sizeof(float));
                }
            });
        }
        gemm_bf16w(att.data(), L.wo, tmp.data(), n, E, E, T, round_);
        parallel_for(T, n, [&](int t0, int t1) {
            for (int t = t0; t < t1; ++t)
                for (int c = 0; c < E; ++c) x[(size_t) t * E + c] += tmp[(size_t) t * E + c] + L.bo[c];
        });
        if (tap) tap("ffn_inp." + std::to_string(l), x.data(), x.size());
        layer_norm(x.data(), L.ln2w, L.ln2b, h.data(), n, E, hp.eps, T);
        gemm_bf16w(h.data(), L.wup, ffn.data(), n, F, E, T, round_);
        parallel_for(T, n, [&](int t0, int t1) {
            for (int t = t0; t < t1; ++t)
                for (int c = 0; c < F; ++c) ffn[(size_t) t * F + c] += L.bup[c];
        });
        if (tap) tap("ffn_up_b." + std::to_string(l), ffn.data(), ffn.size());
        parallel_for(T, n, [&](int t0, int t1) {
            for (int t = t0; t < t1; ++t)
                for (int c = 0; c < F; ++c) ffn[(size_t) t * F + c] = f16_ ? gelu_f16(ffn[(size_t) t * F + c]) : gelu_exact(ffn[(size_t) t * F + c]);
        });
        if (tap) tap("ffn_gelu." + std::to_string(l), ffn.data(), ffn.size());
        gemm_bf16w(ffn.data(), L.wdown, tmp.data(), n, E, F, T, round_);
        if (tap) tap("ffn_down." + std::to_string(l), tmp.data(), tmp.size());
        parallel_for(T, n, [&](int t0, int t1) {
            for (int t = t0; t < t1; ++t)
                for (int c = 0; c < E; ++c) x[(size_t) t * E + c] += tmp[(size_t) t * E + c] + L.bdown[c];
        });
        if (tap) tap("layer_out." + std::to_string(l), x.data(), x.size());
    }
    layer_norm(x.data(), postw_, postb_, h.data(), n, E, hp.eps, T);
    if (tap) tap("post_ln", h.data(), h.size());
    const int m = n / 4, P = hp.proj;
    std::vector<float> mid((size_t) m * 4 * E);
    gemm_bf16w(h.data(), wm0_, mid.data(), m, 4 * E, 4 * E, T, round_);
    parallel_for(T, m, [&](int t0, int t1) {
        for (int t = t0; t < t1; ++t)
            for (int c = 0; c < 4 * E; ++c) mid[(size_t) t * 4 * E + c] = f16_ ? gelu_f16(mid[(size_t) t * 4 * E + c] + bm0_[c]) : gelu_exact(mid[(size_t) t * 4 * E + c] + bm0_[c]);
    });
    out.resize((size_t) m * P);
    gemm_bf16w(mid.data(), wm2_, out.data(), m, P, 4 * E, T, round_);
    for (int t = 0; t < m; ++t)
        for (int c = 0; c < P; ++c) out[(size_t) t * P + c] += bm2_[c];
    if (tap) tap("out", out.data(), out.size());
    return true;
}

}  // namespace basalt::vision
