// tests/vision/pool_encoder_parity.cpp - the pooled tower on the shared encoder, on random weights (no model file).
//
//   pool_encoder_parity
//
// Writes a projector GGUF of the pooled layout (two layers at the real widths, 64-row position tables, llama.cpp's tensor
// names and types: the patch embedding as an F32 convolution, BF16 products, F32 norms) and checks each of the tower's own
// kernels against the host: the patch rows in pool order (bit for bit), the column and row position tables (bit for bit),
// the rotation table and the per-head norms and turns of q, k and v, the normed residual, the gated GELU and the pooling
// against double precision. Then the CUDA encoder against a host reference of the whole tower (float products of the same
// bf16 inputs, double norms and softmax) per stage on pictures of 1, 15 and 54 cells, and a microbench line for each of the
// tower's own kernels at the largest picture (280 cells: 2520 patches). Exit status 1 when a gate fails.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "basalt/core/model_file.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "vision_kernels.cuh"

using namespace basalt::vision;

namespace {

constexpr int E = 1152, F = 4304, LAYERS = 2, S = 64, PROJ = 256, P = 16, H = 16, D = 72, DP = 80, K = 3, PV = 3 * P * P;
constexpr float kEps = 1e-6f, kTheta = 100.f;
int g_fail = 0;
std::mt19937 rng(29);

void verdict(const std::string& what, double got, double gate, bool less = true) {
    const bool ok = less ? got <= gate : got >= gate;
    std::printf("%-58s %10.3e (%s %.1e) %s\n", what.c_str(), got, less ? "<=" : ">=", gate, ok ? "ok" : "FAIL");
    g_fail += !ok;
}

// ---- a GGUF writer ----------------------------------------------------------------------------------------------------
struct Gguf {
    std::string kv, infos;
    std::vector<uint8_t> data;
    int n_kv = 0, n_t = 0;
    template <class T> static void put(std::string& s, T v) { s.append((const char*) &v, sizeof v); }
    static void str(std::string& s, const std::string& v) { put<uint64_t>(s, v.size()); s += v; }
    void kv_str(const char* k, const char* v) { str(kv, k); put<uint32_t>(kv, 8); str(kv, v); ++n_kv; }
    void kv_u32(const char* k, uint32_t v) { str(kv, k); put<uint32_t>(kv, 4); put(kv, v); ++n_kv; }
    void kv_f32(const char* k, float v) { str(kv, k); put<uint32_t>(kv, 6); put(kv, v); ++n_kv; }
    void kv_f32s(const char* k, std::vector<float> v) {
        str(kv, k); put<uint32_t>(kv, 9); put<uint32_t>(kv, 6); put<uint64_t>(kv, v.size());
        for (float x : v) put(kv, x);
        ++n_kv;
    }
    // ne in ggml order (ne[0] contiguous)
    void tensor(const std::string& name, std::vector<uint64_t> ne, uint32_t type, const void* p, size_t bytes) {
        str(infos, name); put<uint32_t>(infos, (uint32_t) ne.size());
        for (uint64_t d : ne) put(infos, d);
        put(infos, type); put<uint64_t>(infos, data.size());
        data.insert(data.end(), (const uint8_t*) p, (const uint8_t*) p + bytes);
        data.resize((data.size() + 31) / 32 * 32);
        ++n_t;
    }
    std::string bytes() const {
        std::string h = "GGUF";
        put<uint32_t>(h, 3); put<uint64_t>(h, (uint64_t) n_t); put<uint64_t>(h, (uint64_t) n_kv);
        h += kv; h += infos;
        h.resize((h.size() + 31) / 32 * 32);
        return h + std::string((const char*) data.data(), data.size());
    }
};

uint16_t bf16(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}
float unbf16(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}
float round_bf16(float v) { return unbf16(bf16(v)); }
float round_half(float v) { return __half2float(__float2half_rn(v)); }

std::vector<float> randn(size_t n, float sd, float mean = 0.f) {
    std::normal_distribution<float> nd(mean, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}
std::vector<float> randn_bf16(size_t n, float sd, float mean = 0.f) {
    std::vector<float> v = randn(n, sd, mean);
    for (float& x : v) x = round_bf16(x);
    return v;
}

std::map<std::string, std::vector<float>> W;   // the weights as the file keeps them

void add_f32(Gguf& g, const std::string& name, std::vector<uint64_t> ne, std::vector<float> v) {
    g.tensor(name, ne, 0, v.data(), v.size() * 4);
    W[name] = std::move(v);
}
void add_bf16(Gguf& g, const std::string& name, std::vector<uint64_t> ne, std::vector<float> v) {
    std::vector<uint16_t> b(v.size());
    for (size_t i = 0; i < v.size(); ++i) { b[i] = bf16(v[i]); v[i] = unbf16(b[i]); }
    g.tensor(name, ne, 30, b.data(), b.size() * 2);
    W[name] = std::move(v);
}

void write_gguf(const char* path) {
    Gguf g;
    g.kv_str("general.architecture", "clip");
    g.kv_str("clip.vision.projector_type", "pooled_test");
    g.kv_u32("clip.vision.image_size", S * P);
    g.kv_u32("clip.vision.patch_size", P);
    g.kv_u32("clip.vision.embedding_length", E);
    g.kv_u32("clip.vision.feed_forward_length", F);
    g.kv_u32("clip.vision.block_count", LAYERS);
    g.kv_u32("clip.vision.attention.head_count", H);
    g.kv_u32("clip.vision.projection_dim", PROJ);
    g.kv_u32("clip.vision.projector.scale_factor", K);
    g.kv_f32("clip.vision.attention.layer_norm_epsilon", kEps);
    g.kv_f32("clip.vision.rope.freq_base", kTheta);
    g.kv_f32s("clip.vision.image_mean", {0.f, 0.f, 0.f});
    g.kv_f32s("clip.vision.image_std", {1.f, 1.f, 1.f});
    add_f32(g, "v.patch_embd.weight", {P, P, 3, E}, randn_bf16((size_t) E * PV, 0.05f));
    add_f32(g, "v.position_embd.weight", {E, S, 2}, randn_bf16((size_t) 2 * S * E, 0.5f));
    add_f32(g, "v.std_bias", {E}, randn_bf16(E, 1.f));
    add_f32(g, "v.std_scale", {E}, randn_bf16(E, 0.1f, 0.5f));
    for (int l = 0; l < LAYERS; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        for (const char* n : {"ln1", "attn_post_norm", "ln2", "ffn_post_norm"}) add_f32(g, p + n + ".weight", {E}, randn_bf16(E, 0.1f, 1.f));
        add_f32(g, p + "attn_q_norm.weight", {D}, randn_bf16(D, 0.1f, 1.f));
        add_f32(g, p + "attn_k_norm.weight", {D}, randn_bf16(D, 0.1f, 1.f));
        for (const char* n : {"attn_q", "attn_k", "attn_v", "attn_out"}) add_bf16(g, p + n + ".weight", {E, E}, randn((size_t) E * E, 0.03f));
        add_bf16(g, p + "ffn_gate.weight", {E, F}, randn((size_t) E * F, 0.03f));
        add_bf16(g, p + "ffn_up.weight", {E, F}, randn((size_t) E * F, 0.03f));
        add_bf16(g, p + "ffn_down.weight", {F, E}, randn((size_t) E * F, 0.03f));
    }
    add_bf16(g, "mm.input_projection.weight", {E, PROJ}, randn((size_t) PROJ * E, 0.03f));
    const std::string b = g.bytes();
    std::FILE* f = std::fopen(path, "wb");
    if (!f || std::fwrite(b.data(), 1, b.size(), f) != b.size()) { std::printf("cannot write %s\n", path); std::exit(2); }
    std::fclose(f);
}

double rel_l2(const float* a, const float* ref, size_t n) {
    double d = 0, r = 0;
    for (size_t i = 0; i < n; ++i) { d += ((double) a[i] - ref[i]) * ((double) a[i] - ref[i]); r += (double) ref[i] * ref[i]; }
    return std::sqrt(d / std::max(r, 1e-300));
}

Image random_picture(int w, int h) {
    Image img;
    img.w = w;
    img.h = h;
    img.rgb.resize((size_t) w * h * 3);
    std::uniform_int_distribution<int> u(0, 255);
    for (auto& v : img.rgb) v = (uint8_t) u(rng);
    return img;
}

// pool-order row t of a gx-wide grid -> patch row and column
void pool_at(int t, int gx, int& py, int& px) {
    const int c = t / (K * K), r = t % (K * K), per = gx / K;
    py = (c / per) * K + r / K;
    px = (c % per) * K + r % K;
}

// the patch rows as the reference makes its bf16 pixels: bf16(2 (v / 255 - 0.5))
std::vector<float> host_patch_rows(const Image& img) {
    const int gx = img.w / P, n = gx * (img.h / P);
    std::vector<float> out((size_t) n * PV);
    volatile float inv = 1.0f / 255.0f;   // no contraction: the device's roundings
    for (int t = 0; t < n; ++t) {
        int py, px;
        pool_at(t, gx, py, px);
        for (int j = 0; j < PV; ++j) {
            const int c = j / (P * P), rem = j % (P * P), X = px * P + rem % P, Y = py * P + rem / P;
            volatile float a = (float) img.rgb[((size_t) Y * img.w + X) * 3 + c] * inv;
            volatile float b = a - 0.5f;
            out[(size_t) t * PV + j] = round_bf16(2.0f * b);
        }
    }
    return out;
}

template <class T>
T* dev_copy(const std::vector<T>& v) {
    T* p = nullptr;
    cudaMalloc(&p, std::max<size_t>(v.size(), 1) * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T>
std::vector<T> host_copy(const T* p, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

// The device's patch rows and position tables against the host's: the same bits.
void check_rows_and_positions(int w, int h) {
    const Image img = random_picture(w, h);
    const std::vector<float> host = host_patch_rows(img);
    const int gx = w / P, n = gx * (h / P);
    uint8_t* d_img = dev_copy(img.rgb);
    __nv_bfloat16* d_rows = nullptr;
    cudaMalloc(&d_rows, (size_t) n * PV * 2);
    launch_pool_patch_rows(d_img, w, gx, K, d_rows, n, nullptr);
    const std::vector<uint16_t> dev = host_copy((const uint16_t*) d_rows, (size_t) n * PV);
    size_t differ = 0;
    for (size_t i = 0; i < dev.size(); ++i) differ += dev[i] != bf16(host[i]);
    const std::string tag = " " + std::to_string(w) + "x" + std::to_string(h);
    verdict("patch rows in pool order" + tag + " (values unlike the host's)", (double) differ, 0.0);
    const std::vector<float> x0 = randn((size_t) n * E, 1.f), &tab = W["v.position_embd.weight"];
    float *d_x = dev_copy(x0), *d_tab = dev_copy(tab);
    launch_pool_embed(d_x, d_tab, n, gx, K, S, nullptr);
    const std::vector<float> x = host_copy(d_x, x0.size());
    differ = 0;
    for (int t = 0; t < n; ++t) {
        int py, px;
        pool_at(t, gx, py, px);
        for (int c = 0; c < E; ++c) {
            const float want = x0[(size_t) t * E + c] + (tab[(size_t) px * E + c] + tab[((size_t) S + py) * E + c]);
            differ += x[(size_t) t * E + c] != want;
        }
    }
    verdict("column and row positions" + tag + " (values unlike the host's)", (double) differ, 0.0);
    for (void* p : {(void*) d_img, (void*) d_rows, (void*) d_x, (void*) d_tab}) cudaFree(p);
}

// The rotation table and the q, k, v epilogue: per head RMS-normed (q, k with their weights, v without), q and k turned
// within each half (dim i with i + 18; the first half by the column, the second by the row), against double.
void check_qkv(int gx, int gy) {
    const int n = gx * gy;
    const std::vector<float> g = randn((size_t) n * 3 * E, 1.f), qn = randn(D, 0.1f, 1.f), kn = randn(D, 0.1f, 1.f);
    float *d_g = dev_copy(g), *d_qn = dev_copy(qn), *d_kn = dev_copy(kn), *d_tab = nullptr;
    cudaMalloc(&d_tab, (size_t) n * 72 * 4);
    __half *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    for (__half** p : {&d_q, &d_k, &d_v}) {
        cudaMalloc(p, (size_t) H * n * DP * 2);
        cudaMemset(*p, 0, (size_t) H * n * DP * 2);
    }
    launch_rope_pool(d_tab, n, gx, K, kTheta, nullptr);
    launch_qkv_normed(d_g, d_qn, d_kn, d_tab, d_q, d_k, d_v, n, kEps, nullptr);
    const std::vector<__half> q = host_copy(d_q, (size_t) H * n * DP), k = host_copy(d_k, q.size()), v = host_copy(d_v, q.size());
    std::vector<float> got, ref;
    double pad = 0;
    for (int t = 0; t < n; ++t) {
        int py, px;
        pool_at(t, gx, py, px);
        for (int part = 0; part < 3; ++part)
            for (int h = 0; h < H; ++h) {
                const float* src = g.data() + (size_t) t * 3 * E + part * E + h * D;
                double ss = 0;
                for (int d = 0; d < D; ++d) ss += (double) src[d] * src[d];
                const double inv = 1.0 / std::sqrt(ss / D + kEps);
                double y[D];
                for (int d = 0; d < D; ++d) y[d] = src[d] * inv * (part == 0 ? qn[d] : part == 1 ? kn[d] : 1.0);
                const std::vector<__half>& dst = part == 0 ? q : part == 1 ? k : v;
                for (int d = 0; d < D; ++d) {
                    double r = y[d];
                    if (part < 2) {
                        const int half = d / 36, i = d % 36, f = i % 18;
                        const double th = (half == 0 ? px : py) * std::pow((double) kTheta, -2.0 * f / 36.0);
                        r = i < 18 ? y[d] * std::cos(th) - y[d + 18] * std::sin(th) : y[d] * std::cos(th) + y[d - 18] * std::sin(th);
                    }
                    got.push_back(__half2float(dst[((size_t) h * n + t) * DP + d]));
                    ref.push_back((float) r);
                }
                for (int d = D; d < DP; ++d) pad += std::fabs(__half2float(dst[((size_t) h * n + t) * DP + d]));
            }
    }
    const std::string tag = " " + std::to_string(gx) + "x" + std::to_string(gy) + " patches";
    verdict("q, k normed and turned, v normed" + tag, rel_l2(got.data(), ref.data(), got.size()), 1e-3);
    verdict("padding columns untouched" + tag, pad, 0.0);
    for (void* p : {(void*) d_g, (void*) d_qn, (void*) d_kn, (void*) d_tab, (void*) d_q, (void*) d_k, (void*) d_v}) cudaFree(p);
}

double gelu_tanh(double x) { return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x))); }

// The normed residual, the gated GELU and the pooling with its scale, standardisation and norm, against double.
void check_glue(int m) {
    const int n = m * K * K;
    const std::vector<float> x0 = randn((size_t) n * E, 1.f), g = randn((size_t) n * 2 * F, 2.f), w = randn(E, 0.1f, 1.f);
    const std::vector<float> &sb = W["v.std_bias"], &ss = W["v.std_scale"];
    float *d_x = dev_copy(x0), *d_g = dev_copy(g), *d_w = dev_copy(w), *d_sb = dev_copy(sb), *d_ss = dev_copy(ss);
    __nv_bfloat16* d_y = nullptr;
    cudaMalloc(&d_y, (size_t) n * F * 2);
    launch_norm_residual(d_x, d_g, d_w, n, kEps, nullptr);   // g read as [n][1152]
    std::vector<float> x = host_copy(d_x, x0.size()), ref(x0.size());
    for (int t = 0; t < n; ++t) {
        const float* gr = g.data() + (size_t) t * E;
        double s = 0;
        for (int c = 0; c < E; ++c) s += (double) gr[c] * gr[c];
        const double inv = 1.0 / std::sqrt(s / E + kEps);
        for (int c = 0; c < E; ++c) ref[(size_t) t * E + c] = (float) (x0[(size_t) t * E + c] + gr[c] * inv * w[c]);
    }
    verdict("normed residual (" + std::to_string(n) + " rows)", rel_l2(x.data(), ref.data(), ref.size()), 1e-6);
    launch_gelu_gate(d_g, d_y, n, F, nullptr);
    const std::vector<uint16_t> y = host_copy((const uint16_t*) d_y, (size_t) n * F);
    std::vector<float> gy(y.size()), gref(y.size());
    for (int t = 0; t < n; ++t)
        for (int c = 0; c < F; ++c) {
            gy[(size_t) t * F + c] = unbf16(y[(size_t) t * F + c]);
            gref[(size_t) t * F + c] = (float) (gelu_tanh(g[(size_t) t * 2 * F + c]) * g[(size_t) t * 2 * F + F + c]);
        }
    verdict("gated GELU (" + std::to_string(n) + " x " + std::to_string(F) + ")", rel_l2(gy.data(), gref.data(), gref.size()), 4e-3);
    launch_pool_finish(d_x, d_sb, d_ss, d_y, m, K, kEps, nullptr);
    const std::vector<uint16_t> pz = host_copy((const uint16_t*) d_y, (size_t) m * E);
    std::vector<float> pg(pz.size()), pref(pz.size());
    for (int i = 0; i < m; ++i) {
        std::vector<double> v(E);
        double q = 0;
        for (int c = 0; c < E; ++c) {
            double s = 0;
            for (int p = 0; p < K * K; ++p) s += x[((size_t) i * K * K + p) * E + c];
            v[c] = (s / (K * K) * std::sqrt((double) E) - sb[c]) * ss[c];
            q += v[c] * v[c];
        }
        const double inv = 1.0 / std::sqrt(q / E + kEps);
        for (int c = 0; c < E; ++c) {
            pg[(size_t) i * E + c] = unbf16(pz[(size_t) i * E + c]);
            pref[(size_t) i * E + c] = (float) (v[c] * inv);
        }
    }
    verdict("pool, scale, standardise, norm (" + std::to_string(m) + " cells)", rel_l2(pg.data(), pref.data(), pref.size()), 4e-3);
    for (void* p : {(void*) d_x, (void*) d_g, (void*) d_w, (void*) d_sb, (void*) d_ss, (void*) d_y}) cudaFree(p);
}

// ---- the host reference of the whole tower ----------------------------------------------------------------------------
void parallel_rows(int n, const std::function<void(int, int)>& fn) {
    const int T = (int) std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int i = 0; i < T; ++i) {
        const int a = (int) ((int64_t) n * i / T), b = (int) ((int64_t) n * (i + 1) / T);
        if (a < b) th.emplace_back(fn, a, b);
    }
    for (auto& t : th) t.join();
}

// y[n][out] = x[n][in] . w[out][in]^T in float
std::vector<float> matmul(const std::vector<float>& x, const std::vector<float>& w, int n, int in, int out) {
    std::vector<float> y((size_t) n * out);
    parallel_rows(n, [&](int a, int b) {
        for (int r = a; r < b; ++r)
            for (int o = 0; o < out; ++o) {
                const float *xr = x.data() + (size_t) r * in, *wr = w.data() + (size_t) o * in;
                float s = 0;
                for (int i = 0; i < in; ++i) s += xr[i] * wr[i];
                y[(size_t) r * out + o] = s;
            }
    });
    return y;
}

std::vector<float> rms_bf16(const std::vector<float>& x, const std::vector<float>& w, int n) {
    std::vector<float> y(x.size());
    for (int t = 0; t < n; ++t) {
        double s = 0;
        for (int c = 0; c < E; ++c) s += (double) x[(size_t) t * E + c] * x[(size_t) t * E + c];
        const double inv = 1.0 / std::sqrt(s / E + kEps);
        for (int c = 0; c < E; ++c) y[(size_t) t * E + c] = round_bf16((float) (x[(size_t) t * E + c] * inv * w[c]));
    }
    return y;
}

void norm_add(std::vector<float>& x, const std::vector<float>& g, const std::vector<float>& w, int n) {
    for (int t = 0; t < n; ++t) {
        double s = 0;
        for (int c = 0; c < E; ++c) s += (double) g[(size_t) t * E + c] * g[(size_t) t * E + c];
        const double inv = 1.0 / std::sqrt(s / E + kEps);
        for (int c = 0; c < E; ++c) x[(size_t) t * E + c] += (float) (g[(size_t) t * E + c] * inv * w[c]);
    }
}

// The whole tower on a picture of whole cells, its stages as the encoder's taps name them.
std::map<std::string, std::vector<float>> reference(const Image& img) {
    std::map<std::string, std::vector<float>> tap;
    const int gx = img.w / P, n = gx * (img.h / P), m = n / (K * K);
    std::vector<float> x = matmul(host_patch_rows(img), W["v.patch_embd.weight"], n, PV, E);
    const std::vector<float>& tab = W["v.position_embd.weight"];
    for (int t = 0; t < n; ++t) {
        int py, px;
        pool_at(t, gx, py, px);
        for (int c = 0; c < E; ++c) x[(size_t) t * E + c] += tab[(size_t) px * E + c] + tab[((size_t) S + py) * E + c];
    }
    tap["embed"] = x;
    for (int l = 0; l < LAYERS; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        std::vector<float> wqkv = W[p + "attn_q.weight"];
        wqkv.insert(wqkv.end(), W[p + "attn_k.weight"].begin(), W[p + "attn_k.weight"].end());
        wqkv.insert(wqkv.end(), W[p + "attn_v.weight"].begin(), W[p + "attn_v.weight"].end());
        const std::vector<float> g = matmul(rms_bf16(x, W[p + "ln1.weight"], n), wqkv, n, E, 3 * E);
        std::vector<float> q((size_t) H * n * D), k = q, v = q;   // [head][row][dim], fp16-rounded as the device keeps them
        for (int t = 0; t < n; ++t) {
            int py, px;
            pool_at(t, gx, py, px);
            for (int part = 0; part < 3; ++part)
                for (int h = 0; h < H; ++h) {
                    const float* src = g.data() + (size_t) t * 3 * E + part * E + h * D;
                    double ss = 0;
                    for (int d = 0; d < D; ++d) ss += (double) src[d] * src[d];
                    const double inv = 1.0 / std::sqrt(ss / D + kEps);
                    const std::vector<float>& nw = W[p + (part == 0 ? "attn_q_norm.weight" : "attn_k_norm.weight")];
                    double y[D];
                    for (int d = 0; d < D; ++d) y[d] = src[d] * inv * (part < 2 ? nw[d] : 1.0f);
                    std::vector<float>& dst = part == 0 ? q : part == 1 ? k : v;
                    for (int d = 0; d < D; ++d) {
                        double r = y[d];
                        if (part < 2) {
                            const int half = d / 36, i = d % 36, f = i % 18;
                            const double th = (half == 0 ? px : py) * std::pow((double) kTheta, -2.0 * f / 36.0);
                            r = i < 18 ? y[d] * std::cos(th) - y[d + 18] * std::sin(th) : y[d] * std::cos(th) + y[d - 18] * std::sin(th);
                        }
                        dst[((size_t) h * n + t) * D + d] = round_half((float) r);
                    }
                }
        }
        std::vector<float> att((size_t) n * E);
        parallel_rows(n, [&](int a, int b) {
            std::vector<double> s(n);
            for (int t = a; t < b; ++t)
                for (int h = 0; h < H; ++h) {
                    const float* qr = q.data() + ((size_t) h * n + t) * D;
                    double mx = -1e300;
                    for (int j = 0; j < n; ++j) {
                        const float* kr = k.data() + ((size_t) h * n + j) * D;
                        double d = 0;
                        for (int i = 0; i < D; ++i) d += (double) qr[i] * kr[i];
                        s[j] = d;
                        mx = std::max(mx, d);
                    }
                    double sum = 0;
                    for (int j = 0; j < n; ++j) sum += (s[j] = std::exp(s[j] - mx));
                    for (int i = 0; i < D; ++i) {
                        double o = 0;
                        for (int j = 0; j < n; ++j) o += s[j] * v[((size_t) h * n + j) * D + i];
                        att[(size_t) t * E + h * D + i] = round_bf16((float) (o / sum));
                    }
                }
        });
        norm_add(x, matmul(att, W[p + "attn_out.weight"], n, E, E), W[p + "attn_post_norm.weight"], n);
        const std::vector<float> a2 = rms_bf16(x, W[p + "ln2.weight"], n);
        const std::vector<float> gt = matmul(a2, W[p + "ffn_gate.weight"], n, E, F), u = matmul(a2, W[p + "ffn_up.weight"], n, E, F);
        std::vector<float> hm(gt.size());
        for (size_t i = 0; i < hm.size(); ++i) hm[i] = round_bf16((float) (gelu_tanh(gt[i]) * u[i]));
        norm_add(x, matmul(hm, W[p + "ffn_down.weight"], n, F, E), W[p + "ffn_post_norm.weight"], n);
        tap["layer_out." + std::to_string(l)] = x;
    }
    std::vector<float> pooled((size_t) m * E);
    const std::vector<float> &sb = W["v.std_bias"], &ss = W["v.std_scale"];
    for (int i = 0; i < m; ++i) {
        std::vector<double> v(E);
        double q = 0;
        for (int c = 0; c < E; ++c) {
            double s = 0;
            for (int p = 0; p < K * K; ++p) s += x[((size_t) i * K * K + p) * E + c];
            v[c] = (s / (K * K) * std::sqrt((double) E) - sb[c]) * ss[c];
            q += v[c] * v[c];
        }
        const double inv = 1.0 / std::sqrt(q / E + kEps);
        for (int c = 0; c < E; ++c) pooled[(size_t) i * E + c] = round_bf16((float) (v[c] * inv));
    }
    tap["pooled"] = pooled;
    tap["out"] = matmul(pooled, W["mm.input_projection.weight"], m, E, PROJ);
    return tap;
}

void check_encoder(const basalt::core::FileRange& fr, int w, int h) {
    Mmproj mp;
    std::string err;
    if (!mp.open(fr, err, "pooled_test")) { std::printf("open: %s\n", err.c_str()); g_fail++; return; }
    const int m = (w / P) * (h / P) / (K * K);
    CudaEncoder gpu;
    if (!gpu.init(mp, m, err)) { std::printf("cuda init: %s\n", err.c_str()); g_fail++; return; }
    const Image img = random_picture(w, h);
    std::map<std::string, std::vector<float>> d;
    std::vector<float> od;
    if (!gpu.encode_padded(img, od, [&](const std::string& n, const float* p, size_t k) { d[n].assign(p, p + k); })) {
        std::printf("encode failed: %s\n", gpu.last_error().c_str());
        g_fail++;
        return;
    }
    const std::map<std::string, std::vector<float>> c = reference(img);
    const std::string tag = " " + std::to_string(w) + "x" + std::to_string(h) + " (" + std::to_string(m) + " cells)";
    for (const auto& kv : c) {
        const std::vector<float>& got = kv.first == "out" ? od : d[kv.first];
        if (got.size() != kv.second.size()) { verdict(kv.first + tag + " (size)", (double) got.size(), (double) kv.second.size()); continue; }
        const double gate = kv.first == "embed" ? 1e-5 : 3e-2;
        verdict(kv.first + ", GPU against the host" + tag, rel_l2(got.data(), kv.second.data(), got.size()), gate);
    }
}

// us per call of fn over 10 calls after a warm-up
template <class Fn>
double time_us(Fn fn) {
    fn();
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < 10; ++i) fn();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms * 100.0;
}

// The tower's own kernels at the largest picture (280 cells: 60 x 42 patches), each against the bytes it moves.
void bench() {
    const int gx = 60, gy = 42, n = gx * gy, m = n / (K * K), w = gx * P, h = gy * P;
    uint8_t* d_img = nullptr;
    __nv_bfloat16 *d_rows = nullptr, *act = nullptr;
    float *x = nullptr, *g = nullptr, *tab = nullptr, *vec = nullptr, *pos = nullptr;
    __half *q = nullptr, *k = nullptr, *v = nullptr;
    cudaMalloc(&d_img, (size_t) w * h * 3);
    cudaMalloc(&d_rows, (size_t) n * PV * 2);
    cudaMalloc(&act, (size_t) n * F * 2);
    cudaMalloc(&x, (size_t) n * E * 4);
    cudaMalloc(&g, (size_t) n * 2 * F * 4);
    cudaMalloc(&tab, (size_t) n * 72 * 4);
    cudaMalloc(&vec, (size_t) E * 4);
    cudaMalloc(&pos, (size_t) 2 * S * E * 4);
    for (__half** p : {&q, &k, &v}) cudaMalloc(p, (size_t) H * n * DP * 2);
    cudaMemset(d_img, 100, (size_t) w * h * 3);
    cudaMemset(g, 0, (size_t) n * 2 * F * 4);
    cudaMemset(x, 0, (size_t) n * E * 4);
    cudaMemset(vec, 0, (size_t) E * 4);
    cudaMemset(pos, 0, (size_t) 2 * S * E * 4);
    launch_rope_pool(tab, n, gx, K, kTheta, nullptr);
    auto line = [](const char* what, double us, double bytes) {
        std::printf("  bench %-38s %9.1f us/call  %6.0f GB/s\n", what, us, bytes / (us * 1e-6) / 1e9);
    };
    line("patch rows (960 x 672 picture)", time_us([&] { launch_pool_patch_rows(d_img, w, gx, K, d_rows, n, nullptr); }),
         (double) w * h * 3 + (double) n * PV * 2);
    line("positions (2520 rows)", time_us([&] { launch_pool_embed(x, pos, n, gx, K, S, nullptr); }), (double) n * E * 16);
    line("rotation table (2520 rows)", time_us([&] { launch_rope_pool(tab, n, gx, K, kTheta, nullptr); }), (double) n * 72 * 4);
    line("q, k, v normed and turned (2520 rows)", time_us([&] { launch_qkv_normed(g, vec, vec, tab, q, k, v, n, kEps, nullptr); }),
         (double) n * 3 * E * 4 + (double) n * 3 * H * DP * 2 + (double) n * 72 * 4);
    line("normed residual (2520 x 1152)", time_us([&] { launch_norm_residual(x, g, vec, n, kEps, nullptr); }), (double) n * E * 12);
    line("gated GELU (2520 x 4304)", time_us([&] { launch_gelu_gate(g, act, n, F, nullptr); }), (double) n * F * 10);
    line("pool, standardise, norm (280 cells)", time_us([&] { launch_pool_finish(x, vec, vec, act, m, K, kEps, nullptr); }),
         (double) n * E * 4 + (double) m * E * 2);
    for (void* p : {(void*) d_img, (void*) d_rows, (void*) act, (void*) x, (void*) g, (void*) tab, (void*) vec, (void*) pos,
                    (void*) q, (void*) k, (void*) v})
        cudaFree(p);
}

}  // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) { std::printf("no CUDA device\n"); return 77; }
    const std::string path = "pool_encoder_parity.gguf";
    write_gguf(path.c_str());
    check_rows_and_positions(48, 48);
    check_rows_and_positions(240, 144);
    check_rows_and_positions(960, 672);
    check_qkv(6, 3);
    check_qkv(60, 42);
    check_glue(7);
    check_glue(280);
    basalt::core::FileRange fr;
    fr.path = path;
    check_encoder(fr, 48, 48);      // one cell
    check_encoder(fr, 240, 144);    // 5 x 3 cells
    check_encoder(fr, 432, 288);    // 9 x 6 cells
    std::remove(path.c_str());
    bench();
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
