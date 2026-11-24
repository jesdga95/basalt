// src/vision/encoder_cpu.cpp - see basalt/vision/encoder_cpu.hpp.
#include "basalt/vision/encoder_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

#include "basalt/platform/compiler.hpp"

#include "gemm_cpu.hpp"
#include "hy4_weights.hpp"

namespace basalt::vision {

namespace {

float f16_round(float x) { return platform::f16_round(x); }

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

float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f)); }

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

// Attention of every token over every token from the rows of q, k, v [n][3E] (H heads of E / H): per head, blocks of 64
// queries against all keys as two products. kh and vt are scratch of n * E floats each.
void attend_all(const float* qkv, int n, int E, int H, float kq, std::vector<float>& kh, std::vector<float>& vt, float* att,
                int T) {
    const int D = E / H, QB = 64, nqb = (n + QB - 1) / QB;
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

}  // namespace

bool CpuEncoder::init(const Mmproj& mp, int threads, bool round_bf16, bool f16_quirks, std::string& err) {
    f16_ = f16_quirks;
    mp_ = &mp;
    threads_ = threads > 0 ? threads : (int) std::max(1u, std::thread::hardware_concurrency());
    round_ = round_bf16;
    const Hparams& h = mp.hp();
    if (mp.find("v.class_embd")) return init_cls(mp, err);   // the class-token tower
    if (h.embd != 1152 || h.heads != 16 || h.patch != 16 || h.ffn != 4304 || h.merge != 2 || h.layers < 1) {
        err = "unexpected projector hyper-parameters";
        return false;
    }
    const int64_t E = h.embd, F = h.ffn;
    hy4_ = mp.projector() == "hy_v4_flash";
    auto f32 = [&](const std::string& n, int64_t e) { const auto* t = mp.need(n, 0, e, err); return t ? (const float*) t->data : nullptr; };
    auto b16 = [&](const std::string& n, int64_t e) { const auto* t = mp.need(n, 30, e, err); return t ? (const uint16_t*) t->data : nullptr; };
    const float *w0 = f32("v.patch_embd.weight", E * 768), *w1 = hy4_ ? w0 : f32("v.patch_embd.weight.1", E * 768);
    bpatch_ = f32("v.patch_embd.bias", E);
    if (hy4_) {
        const TensorRef* pt = mp.find("v.position_embd.weight");
        const int64_t rows = pt && pt->ne.size() == 2 && pt->ne[0] == E ? pt->ne[1] : 0;
        edge_ = (int) std::lround(std::sqrt((double) rows));
        if (edge_ < 2 || (int64_t) edge_ * edge_ != rows) { err = "the position table of the projector is not a square grid"; return false; }
        pos_ = f32("v.position_embd.weight", E * rows);
        postw_ = f32("mm.pre_norm.weight", E);
        postb_ = postw_;
        newline_ = f32("v.image_newline", h.proj);
        if (!newline_) return false;
    } else {
        pos_ = f32("v.position_embd.weight", E * 2304);
        postw_ = f32("v.post_ln.weight", E);
        postb_ = f32("v.post_ln.bias", E);
    }
    bm0_ = f32("mm.0.bias", 4 * E);
    bm2_ = f32("mm.2.bias", h.proj);
    if (hy4_) {
        own_w_.resize(2);
        own_w_[0].resize((size_t) 4 * E * 4 * E);
        own_w_[1].resize((size_t) 4 * E * h.proj);
        if (!hy4::merge_matrix(mp, (int) E, own_w_[0].data(), err) || !hy4::out_matrix(mp, (int) E, h.proj, own_w_[1].data(), err)) return false;
        wm0_ = own_w_[0].data();
        wm2_ = own_w_[1].data();
    } else {
        wm0_ = b16("mm.0.weight", 4 * E * 4 * E);
        wm2_ = b16("mm.2.weight", 4 * E * h.proj);
    }
    if (!w0 || !w1 || !bpatch_ || !pos_ || !postw_ || !postb_ || !bm0_ || !bm2_ || !wm0_ || !wm2_) return false;
    wpatch_.resize((size_t) E * 768);
    for (size_t i = 0; i < wpatch_.size(); ++i) wpatch_[i] = hy4_ ? w0[i] : w0[i] + w1[i];
    wpatch2_.resize(hy4_ ? 0 : (size_t) E * 1536);
    for (int64_t e = 0; !hy4_ && e < E; ++e)
        for (int c = 0; c < 3; ++c) {
            std::memcpy(&wpatch2_[(size_t) e * 1536 + c * 512], w0 + e * 768 + c * 256, 256 * sizeof(float));
            std::memcpy(&wpatch2_[(size_t) e * 1536 + c * 512 + 256], w1 + e * 768 + c * 256, 256 * sizeof(float));
        }
    layers_.resize(h.layers);
    for (int l = 0; l < h.layers; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        Layer& L = layers_[l];
        L.ln1w = f32(p + "ln1.weight", E);
        L.ln1b = f32(p + "ln1.bias", E);
        L.ln2w = f32(p + "ln2.weight", E);
        L.ln2b = f32(p + "ln2.bias", E);
        if (hy4_) {   // q, k and v are three tensors in the file, one product here
            own_w_.emplace_back((size_t) 3 * E * E);
            own_b_.emplace_back((size_t) 3 * E);
            static const char* const qkv[3] = {"attn_q", "attn_k", "attn_v"};
            for (int i = 0; i < 3; ++i) {
                const uint16_t* w = b16(p + qkv[i] + ".weight", E * E);
                const float* b = f32(p + qkv[i] + ".bias", E);
                if (!w || !b) return false;
                std::memcpy(&own_w_.back()[(size_t) i * E * E], w, (size_t) E * E * 2);
                std::memcpy(&own_b_.back()[(size_t) i * E], b, (size_t) E * 4);
            }
            L.wqkv = own_w_.back().data();
            L.bqkv = own_b_.back().data();
        } else {
            L.bqkv = f32(p + "attn_qkv.bias", 3 * E);
        }
        L.bo = f32(p + "attn_out.bias", E);
        L.bup = f32(p + "ffn_up.bias", F);
        L.bdown = f32(p + "ffn_down.bias", E);
        if (!hy4_) L.wqkv = b16(p + "attn_qkv.weight", 3 * E * E);
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
    const bool pair = !hy4_ && (int64_t) patches.size() == (int64_t) n * 1536;   // a video group: the two taps apart
    if (hy4_ && gx % 2 != 0) return false;
    if (n % 4 != 0 || (!pair && (int) patches.size() != n * 768)) return false;
    std::vector<float> x((size_t) n * E), h((size_t) n * E), qkv((size_t) n * 3 * E), att((size_t) n * E),
        tmp((size_t) n * E), ffn((size_t) n * F), attk((size_t) H * n * D), attv((size_t) H * D * n);
    const int T = threads_;

    // patch embedding + bilinear position embedding, rows in merge order
    std::vector<float> p16(patches);
    if (f16_) for (float& v : p16) v = f16_round(v);
    if (pair) gemm_f32w(p16.data(), wpatch2_.data(), x.data(), n, E, 1536, T);
    else gemm_f32w(p16.data(), wpatch_.data(), x.data(), n, E, 768, T);
    {
        const int S = hy4_ ? edge_ : 48;
        auto src = [&](int axis_out, int axis_in, int i, int& i0, int& i1, float& d) {
            if (hy4_) {   // the reference's scale factor (size + 0.1) / S, align_corners false (torch's upsample_bilinear2d)
                if (axis_out == axis_in) { i0 = i1 = i; d = 0.0f; return; }   // torch copies an axis whose size is unchanged
                const float inv = (float) (1.0 / (((double) axis_out + 0.1) / axis_in));
                const float v = std::max(inv * ((float) i + 0.5f) - 0.5f, 0.0f);
                const int a = std::min((int) floorf(v), axis_in - 1);
                i0 = a;
                i1 = a + (a < axis_in - 1 ? 1 : 0);
                d = std::max(0.0f, std::min(v - (float) a, 1.0f));
                return;
            }
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
                    const float pe = hy4_ ? (1 - dy) * ((1 - dx) * a + dx * b) + dy * ((1 - dx) * cc + dx * d)
                                          : a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + cc * (1 - dx) * dy + d * dx * dy;
                    row[c] += bpatch_[c] + pe;
                }
            }
        });
    }
    if (tap) tap("embed", x.data(), x.size());

    // rotation tables per patch: 36 cos/sin pairs, rows in merge order
    std::vector<float> rc(hy4_ ? 0 : (size_t) n * 36), rs(hy4_ ? 0 : (size_t) n * 36);
    if (!hy4_) {
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
                for (int part = 0; !hy4_ && part < 2; ++part)
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
        attend_all(qkv.data(), n, E, H, kq, attk, attv, att.data(), T);
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
    if (hy4_) {
        for (int t = 0; t < n; ++t) {
            const float* xr = &x[(size_t) t * E];
            double sq = 0;
            for (int i = 0; i < E; ++i) sq += (double) xr[i] * xr[i];
            const float inv = 1.0f / sqrtf((float) (sq / E) + hp.eps);
            for (int i = 0; i < E; ++i) h[(size_t) t * E + i] = xr[i] * inv * postw_[i];
        }
    } else {
        layer_norm(x.data(), postw_, postb_, h.data(), n, E, hp.eps, T);
    }
    if (tap) tap("post_ln", h.data(), h.size());
    const int m = n / 4, P = hp.proj;
    std::vector<float> mid((size_t) m * 4 * E);
    gemm_bf16w(h.data(), wm0_, mid.data(), m, 4 * E, 4 * E, T, round_);
    parallel_for(T, m, [&](int t0, int t1) {
        for (int t = t0; t < t1; ++t)
            for (int c = 0; c < 4 * E; ++c) {
                const float v = mid[(size_t) t * 4 * E + c] + bm0_[c];
                mid[(size_t) t * 4 * E + c] = hy4_ ? gelu_erf(v) : f16_ ? gelu_f16(v) : gelu_exact(v);
            }
    });
    out.resize((size_t) m * P);
    gemm_bf16w(mid.data(), wm2_, out.data(), m, P, 4 * E, T, round_);
    for (int t = 0; t < m; ++t)
        for (int c = 0; c < P; ++c) out[(size_t) t * P + c] += bm2_[c];
    if (hy4_) {   // each line of gx / 2 merged rows is followed by the newline row
        const int ow = gx / 2, lines = m / ow;
        std::vector<float> lined((size_t) (m + lines) * P);
        for (int r = 0; r < lines; ++r) {
            std::memcpy(&lined[(size_t) r * (ow + 1) * P], &out[(size_t) r * ow * P], (size_t) ow * P * sizeof(float));
            std::memcpy(&lined[((size_t) r * (ow + 1) + ow) * P], newline_, (size_t) P * sizeof(float));
        }
        out.swap(lined);
    }
    if (tap) tap("out", out.data(), out.size());
    return true;
}

bool CpuEncoder::init_cls(const Mmproj& mp, std::string& err) {
    const Hparams& h = mp.hp();
    if (h.embd != 1024 || h.heads != 16 || h.patch != 20 || h.ffn != 4096 || h.merge != 2 || h.layers < 1) {
        err = "unexpected projector hyper-parameters";
        return false;
    }
    cls_ = true;
    const int64_t E = h.embd, F = h.ffn, PV = 3 * 20 * 20;
    auto f32 = [&](const std::string& n, int64_t e) { const auto* t = mp.need(n, 0, e, err); return t ? (const float*) t->data : nullptr; };
    auto b16 = [&](const std::string& n, int64_t e) { const auto* t = mp.need(n, 30, e, err); return t ? (const uint16_t*) t->data : nullptr; };
    const TensorRef* pt = mp.find("v.position_embd.weight");
    const int64_t rows = pt && pt->ne.size() == 2 && pt->ne[0] == E ? pt->ne[1] : 0;
    edge_ = (int) std::lround(std::sqrt((double) rows));
    if (edge_ < 2 || (int64_t) edge_ * edge_ != rows) { err = "the position table of the projector is not a square grid"; return false; }
    const float* wp = f32("v.patch_embd.weight", E * PV);   // [E][3][20][20]: channel, row, column, as patchify_padded's rows
    pos_ = f32("v.position_embd.weight", E * rows);
    const float *ce = f32("v.class_embd", E), *cp = f32("v.class_position_embd.weight", E);
    postw_ = f32("v.pre_ln.weight", E);
    postb_ = f32("v.pre_ln.bias", E);
    wm0_ = b16("mm.1.weight", 4 * E * 4 * E);
    wm2_ = b16("mm.2.weight", 4 * E * h.proj);
    if (!wp || !pos_ || !ce || !cp || !postw_ || !postb_ || !wm0_ || !wm2_) return false;
    wpatch_.assign(wp, wp + E * PV);
    cls_row_.resize((size_t) E);
    for (int64_t i = 0; i < E; ++i) cls_row_[(size_t) i] = ce[i] + cp[i];
    layers_.resize(h.layers);
    for (int l = 0; l < h.layers; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        Layer& L = layers_[l];
        L.ln1w = f32(p + "ln1.weight", E); L.ln1b = f32(p + "ln1.bias", E);
        L.ln2w = f32(p + "ln2.weight", E); L.ln2b = f32(p + "ln2.bias", E);
        L.ls1 = f32(p + "ls1.weight", E); L.ls2 = f32(p + "ls2.weight", E);
        own_w_.emplace_back((size_t) 3 * E * E);
        own_b_.emplace_back((size_t) 3 * E);
        static const char* const qkv[3] = {"attn_q", "attn_k", "attn_v"};
        for (int i = 0; i < 3; ++i) {
            const uint16_t* w = b16(p + qkv[i] + ".weight", E * E);
            const float* b = f32(p + qkv[i] + ".bias", E);
            if (!w || !b) return false;
            std::memcpy(&own_w_.back()[(size_t) i * E * E], w, (size_t) E * E * 2);
            std::memcpy(&own_b_.back()[(size_t) i * E], b, (size_t) E * 4);
        }
        L.wqkv = own_w_.back().data();
        L.bqkv = own_b_.back().data();
        L.wo = b16(p + "attn_out.weight", E * E); L.bo = f32(p + "attn_out.bias", E);
        L.wup = b16(p + "ffn_up.weight", F * E); L.bup = f32(p + "ffn_up.bias", F);
        L.wdown = b16(p + "ffn_down.weight", E * F); L.bdown = f32(p + "ffn_down.bias", E);
        if (!L.ln1w || !L.ln1b || !L.ln2w || !L.ln2b || !L.ls1 || !L.ls2 || !L.wo || !L.bo || !L.wup || !L.bup || !L.wdown || !L.bdown)
            return false;
    }
    return true;
}

bool CpuEncoder::encode_padded(const Image& img, std::vector<float>& out, const TapFn& tap) {
    if (!cls_ || img.w <= 0 || img.h <= 0 || img.rgb.size() != (size_t) img.w * img.h * 3) return false;
    const Hparams& hp = mp_->hp();
    const int E = hp.embd, F = hp.ffn, H = hp.heads, P = hp.patch, PV = 3 * P * P, S = edge_;
    Geometry g;
    g.patch = P;
    g.merge = hp.merge;
    const Size canvas = padded_canvas(img.w, img.h, g);
    const int gx = canvas.w / P, gy = canvas.h / P, n = gx * gy, Tn = n + 1, m = n / 4;
    const int T = threads_;
    std::vector<float> rows;
    patchify_padded(img, g, hp.mean, hp.std, rows);
    if (round_)   // the device rounds its patch rows to bf16, as the reference casts its pixels
        for (float& v : rows) v = hy4::unbf16(hy4::bf16_of(v));
    std::vector<float> x((size_t) Tn * E), h((size_t) Tn * E), qkv((size_t) Tn * 3 * E), att((size_t) Tn * E),
        tmp((size_t) Tn * E), ffn((size_t) Tn * F), attk((size_t) Tn * E), attv((size_t) Tn * E);
    gemm_f32w(rows.data(), wpatch_.data(), x.data() + E, n, E, PV, T);
    std::memcpy(x.data(), cls_row_.data(), (size_t) E * sizeof(float));
    // the position table resampled to the grid: torch's bilinear taps, align_corners false, clamped to the table
    auto taps = [&](int size, int i, int& i0, int& i1, float& w0, float& w1) {
        const float src = ((float) i + 0.5f) * (float) S / (float) size - 0.5f;
        const float fl = std::floor(src);
        w0 = std::max(1.0f - std::fabs(src - fl), 0.0f);
        w1 = std::max(1.0f - std::fabs(src - fl - 1.0f), 0.0f);
        i0 = std::min(std::max((int) fl, 0), S - 1);
        i1 = std::min(std::max((int) fl + 1, 0), S - 1);
    };
    auto patch_at = [&](int t, int& py, int& px) {
        const int grp = t / 4, r = t % 4;
        py = (grp / (gx / 2)) * 2 + r / 2;
        px = (grp % (gx / 2)) * 2 + r % 2;
    };
    parallel_for(T, n, [&](int t0, int t1) {
        for (int t = t0; t < t1; ++t) {
            int py, px, x0, x1, y0, y1;
            float wx0, wx1, wy0, wy1;
            patch_at(t, py, px);
            taps(gx, px, x0, x1, wx0, wx1);
            taps(gy, py, y0, y1, wy0, wy1);
            const float w00 = wy0 * wx0, w01 = wy0 * wx1, w10 = wy1 * wx0, w11 = wy1 * wx1;
            float* row = &x[(size_t) (t + 1) * E];
            for (int c = 0; c < E; ++c)
                row[c] += pos_[((size_t) y0 * S + x0) * E + c] * w00 + pos_[((size_t) y0 * S + x1) * E + c] * w01 +
                          pos_[((size_t) y1 * S + x0) * E + c] * w10 + pos_[((size_t) y1 * S + x1) * E + c] * w11;
        }
    });
    if (tap) tap("embed", x.data(), x.size());
    layer_norm(x.data(), postw_, postb_, h.data(), Tn, E, hp.eps, T);
    x.swap(h);
    if (tap) tap("pre_ln", x.data(), x.size());
    // rotation of the interleaved pairs: 0..15 by the patch's column + 1, 16..31 by its row + 1; the class token at (0, 0)
    std::vector<float> rc((size_t) Tn * 32), rs((size_t) Tn * 32);
    for (int t = 0; t < Tn; ++t) {
        int py = -1, px = -1;
        if (t > 0) patch_at(t - 1, py, px);
        for (int j = 0; j < 32; ++j) {
            const float inv = 1.0f / powf(10000.0f, (float) (2 * (j % 16)) / 32.0f);
            const float th = (float) (j < 16 ? px + 1 : py + 1) * inv;
            rc[(size_t) t * 32 + j] = cosf(th);
            rs[(size_t) t * 32 + j] = sinf(th);
        }
    }
    const int D = E / H;
    const float kq = 1.0f / sqrtf((float) D);
    for (int l = 0; l < hp.layers; ++l) {
        const Layer& L = layers_[l];
        layer_norm(x.data(), L.ln1w, L.ln1b, h.data(), Tn, E, hp.eps, T);
        gemm_bf16w(h.data(), L.wqkv, qkv.data(), Tn, 3 * E, E, T, round_);
        parallel_for(T, Tn, [&](int t0, int t1) {
            for (int t = t0; t < t1; ++t) {
                float* r = &qkv[(size_t) t * 3 * E];
                for (int i = 0; i < 3 * E; ++i) r[i] += L.bqkv[i];
                for (int part = 0; part < 2; ++part)
                    for (int hd = 0; hd < H; ++hd) {
                        float* v = r + part * E + hd * D;
                        for (int j = 0; j < 32; ++j) {
                            const float c = rc[(size_t) t * 32 + j], sn = rs[(size_t) t * 32 + j];
                            const float a = v[2 * j], b = v[2 * j + 1];
                            v[2 * j] = a * c - b * sn;
                            v[2 * j + 1] = b * c + a * sn;
                        }
                    }
            }
        });
        attend_all(qkv.data(), Tn, E, H, kq, attk, attv, att.data(), T);
        gemm_bf16w(att.data(), L.wo, tmp.data(), Tn, E, E, T, round_);
        parallel_for(T, Tn, [&](int t0, int t1) {
            for (int t = t0; t < t1; ++t)
                for (int c = 0; c < E; ++c) x[(size_t) t * E + c] += L.ls1[c] * (tmp[(size_t) t * E + c] + L.bo[c]);
        });
        if (tap) tap("ffn_inp." + std::to_string(l), x.data(), x.size());
        layer_norm(x.data(), L.ln2w, L.ln2b, h.data(), Tn, E, hp.eps, T);
        gemm_bf16w(h.data(), L.wup, ffn.data(), Tn, F, E, T, round_);
        parallel_for(T, Tn, [&](int t0, int t1) {
            for (int t = t0; t < t1; ++t)
                for (int c = 0; c < F; ++c) ffn[(size_t) t * F + c] = gelu_erf(ffn[(size_t) t * F + c] + L.bup[c]);
        });
        gemm_bf16w(ffn.data(), L.wdown, tmp.data(), Tn, E, F, T, round_);
        parallel_for(T, Tn, [&](int t0, int t1) {
            for (int t = t0; t < t1; ++t)
                for (int c = 0; c < E; ++c) x[(size_t) t * E + c] += L.ls2[c] * (tmp[(size_t) t * E + c] + L.bdown[c]);
        });
        if (tap) tap("layer_out." + std::to_string(l), x.data(), x.size());
    }
    // a cell's four patch rows follow each other: rows 1.. of the sequence are the merge products' [m][4E] input
    const int Pj = hp.proj;
    std::vector<float> mid((size_t) m * 4 * E);
    gemm_bf16w(x.data() + E, wm0_, mid.data(), m, 4 * E, 4 * E, T, round_);
    parallel_for(T, m, [&](int t0, int t1) {
        for (size_t i = (size_t) t0 * 4 * E; i < (size_t) t1 * 4 * E; ++i) mid[i] = gelu_erf(mid[i]);
    });
    out.resize((size_t) m * Pj);
    gemm_bf16w(mid.data(), wm2_, out.data(), m, Pj, 4 * E, T, round_);
    if (tap) tap("out", out.data(), out.size());
    return true;
}

}  // namespace basalt::vision
