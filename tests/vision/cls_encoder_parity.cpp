// tests/vision/cls_encoder_parity.cpp - the class-token tower on the shared encoder, on random weights (no model file).
//
//   cls_encoder_parity
//
// Writes a projector GGUF of the class-token layout (two layers at the real widths, a 32 x 32 position table, llama.cpp's
// tensor names and types: the patch embedding as an F32 convolution, BF16 products, no merge biases) and checks: the
// device's patch rows against the host's (bit for bit); the position table's resampling against a double-precision copy of
// the reference's bilinear helper on grids smaller than, equal to and larger than the table; the rotation of the
// interleaved pairs of q and k against double precision; the CUDA encoder against the CPU encoder per stage on padded
// pictures, one past a chunk of product rows; and the merge and projection against an independent double-precision
// reference. Then a microbench line for each of the tower's own kernels at the largest picture. Exit status 1 when a gate
// fails.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "basalt/core/model_file.hpp"
#include "basalt/vision/encoder_cpu.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "vision_kernels.cuh"

using namespace basalt::vision;

namespace {

constexpr int E = 1024, F = 4096, LAYERS = 2, EDGE = 32, PROJ = 2304, P = 20, H = 16, D = 64;
int g_fail = 0;
std::mt19937 rng(23);

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

std::vector<float> randn(size_t n, float sd, float mean = 0.f) {
    std::normal_distribution<float> nd(mean, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}
// values a bf16 holds exactly, as a checkpoint converted to the file's F32 has them
std::vector<float> randn_bf16(size_t n, float sd, float mean = 0.f) {
    std::vector<float> v = randn(n, sd, mean);
    for (float& x : v) x = round_bf16(x);
    return v;
}

// the weights as the file keeps them (host copies for the reference too)
std::map<std::string, std::vector<float>> W;

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
    g.kv_str("clip.vision.projector_type", "class_token_test");
    g.kv_u32("clip.vision.image_size", EDGE * P);
    g.kv_u32("clip.vision.patch_size", P);
    g.kv_u32("clip.vision.embedding_length", E);
    g.kv_u32("clip.vision.feed_forward_length", F);
    g.kv_u32("clip.vision.block_count", LAYERS);
    g.kv_u32("clip.vision.attention.head_count", H);
    g.kv_u32("clip.vision.projection_dim", PROJ);
    g.kv_u32("clip.vision.spatial_merge_size", 2);
    g.kv_f32("clip.vision.attention.layer_norm_epsilon", 1e-5f);
    g.kv_f32s("clip.vision.image_mean", {0.5f, 0.5f, 0.5f});
    g.kv_f32s("clip.vision.image_std", {0.5f, 0.5f, 0.5f});
    add_f32(g, "v.class_embd", {E}, randn_bf16(E, 0.5f));
    add_f32(g, "v.class_position_embd.weight", {E}, randn_bf16(E, 0.5f));
    add_f32(g, "v.patch_embd.weight", {P, P, 3, E}, randn_bf16((size_t) E * 3 * P * P, 0.03f));
    add_f32(g, "v.position_embd.weight", {E, EDGE * EDGE}, randn_bf16((size_t) E * EDGE * EDGE, 0.5f));
    add_f32(g, "v.pre_ln.weight", {E}, randn_bf16(E, 0.1f, 1.f));
    add_f32(g, "v.pre_ln.bias", {E}, randn_bf16(E, 0.05f));
    for (int l = 0; l < LAYERS; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        add_f32(g, p + "ln1.weight", {E}, randn(E, 0.1f, 1.f)); add_f32(g, p + "ln1.bias", {E}, randn(E, 0.05f));
        add_f32(g, p + "ln2.weight", {E}, randn(E, 0.1f, 1.f)); add_f32(g, p + "ln2.bias", {E}, randn(E, 0.05f));
        add_f32(g, p + "ls1.weight", {E}, randn(E, 0.05f, 0.1f)); add_f32(g, p + "ls2.weight", {E}, randn(E, 0.05f, 0.1f));
        for (const char* n : {"attn_q", "attn_k", "attn_v"}) {
            add_bf16(g, p + n + ".weight", {E, E}, randn((size_t) E * E, 0.03f));
            add_f32(g, p + n + ".bias", {E}, randn(E, 0.02f));
        }
        add_bf16(g, p + "attn_out.weight", {E, E}, randn((size_t) E * E, 0.03f)); add_f32(g, p + "attn_out.bias", {E}, randn(E, 0.02f));
        add_bf16(g, p + "ffn_up.weight", {E, F}, randn((size_t) E * F, 0.03f)); add_f32(g, p + "ffn_up.bias", {F}, randn(F, 0.02f));
        add_bf16(g, p + "ffn_down.weight", {F, E}, randn((size_t) E * F, 0.03f)); add_f32(g, p + "ffn_down.bias", {E}, randn(E, 0.02f));
    }
    add_bf16(g, "mm.1.weight", {4 * E, 4 * E}, randn((size_t) 4 * E * 4 * E, 0.02f));
    add_bf16(g, "mm.2.weight", {4 * E, PROJ}, randn((size_t) PROJ * 4 * E, 0.02f));
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

// merge-order row t of a gx-wide grid -> patch row and column
void patch_at(int t, int gx, int& py, int& px) {
    const int grp = t / 4, r = t % 4;
    py = (grp / (gx / 2)) * 2 + r / 2;
    px = (grp % (gx / 2)) * 2 + r % 2;
}

// The device's patch rows against the host's patchify_padded rounded to bf16: the same bits.
void check_patch_rows(int w, int h) {
    const Image img = random_picture(w, h);
    const Geometry g{P, 2, 1, 1 << 20};
    const float mean[3] = {0.5f, 0.5f, 0.5f}, sd[3] = {0.5f, 0.5f, 0.5f};
    std::vector<float> host;
    patchify_padded(img, g, mean, sd, host);
    const Size canvas = padded_canvas(w, h, g);
    const int gx = canvas.w / P, n = gx * (canvas.h / P), vals = 3 * P * P;
    uint8_t* d_img = nullptr;
    __nv_bfloat16* d_rows = nullptr;
    cudaMalloc(&d_img, img.rgb.size());
    cudaMalloc(&d_rows, (size_t) n * vals * 2);
    cudaMemcpy(d_img, img.rgb.data(), img.rgb.size(), cudaMemcpyHostToDevice);
    const PixelNorm nm{{127.5f, 127.5f, 127.5f}, {127.5f, 127.5f, 127.5f}};
    // two launches of parts of the rows, as the encoder's chunks make them
    const int half = n / 2;
    launch_patch_rows(d_img, w, h, gx, P, nm, d_rows, 0, half, nullptr);
    launch_patch_rows(d_img, w, h, gx, P, nm, d_rows + (size_t) half * vals, half, n - half, nullptr);
    std::vector<uint16_t> dev((size_t) n * vals);
    cudaMemcpy(dev.data(), d_rows, dev.size() * 2, cudaMemcpyDeviceToHost);
    size_t differ = 0;
    for (size_t i = 0; i < dev.size(); ++i) differ += dev[i] != bf16(host[i]);
    verdict("patch rows " + std::to_string(w) + "x" + std::to_string(h) + " (values unlike the host's)", (double) differ, 0.0);
    cudaFree(d_img);
    cudaFree(d_rows);
}

// torch's bilinear resampling, align_corners false, as vision_utils' helper computes its taps, in double
double ref_taps(int S, int size, int i, int& i0, int& i1, double& w0, double& w1) {
    const double src = (i + 0.5) * S / size - 0.5, fl = std::floor(src);
    w0 = std::max(1.0 - std::fabs(src - fl), 0.0);
    w1 = std::max(1.0 - std::fabs(src - fl - 1.0), 0.0);
    i0 = std::min(std::max((int) fl, 0), S - 1);
    i1 = std::min(std::max((int) fl + 1, 0), S - 1);
    return src;
}

// The position table resampled to a gx x gy grid (x = 0): against double precision, and the class token's row.
void check_positions(int gx, int gy) {
    const std::vector<float>& tab = W["v.position_embd.weight"];
    const int n = gx * gy;
    float *d_x = nullptr, *d_tab = nullptr, *d_cls = nullptr;
    cudaMalloc(&d_x, (size_t) (n + 1) * E * 4);
    cudaMalloc(&d_tab, tab.size() * 4);
    cudaMalloc(&d_cls, E * 4);
    cudaMemset(d_x, 0, (size_t) (n + 1) * E * 4);
    cudaMemcpy(d_tab, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cls, W["v.class_embd"].data(), E * 4, cudaMemcpyHostToDevice);
    launch_embed_cls(d_x, d_cls, d_tab, gx, gy, EDGE, E, nullptr);
    std::vector<float> x((size_t) (n + 1) * E);
    cudaMemcpy(x.data(), d_x, x.size() * 4, cudaMemcpyDeviceToHost);
    double worst = 0;
    for (int c = 0; c < E; ++c) worst = std::max(worst, (double) std::fabs(x[c] - W["v.class_embd"][(size_t) c]));
    std::vector<float> ref((size_t) n * E);
    for (int t = 0; t < n; ++t) {
        int py, px, x0, x1, y0, y1;
        double wx0, wx1, wy0, wy1;
        patch_at(t, gx, py, px);
        ref_taps(EDGE, gx, px, x0, x1, wx0, wx1);
        ref_taps(EDGE, gy, py, y0, y1, wy0, wy1);
        for (int c = 0; c < E; ++c)
            ref[(size_t) t * E + c] = (float) (wy0 * wx0 * tab[((size_t) y0 * EDGE + x0) * E + c] + wy0 * wx1 * tab[((size_t) y0 * EDGE + x1) * E + c] +
                                               wy1 * wx0 * tab[((size_t) y1 * EDGE + x0) * E + c] + wy1 * wx1 * tab[((size_t) y1 * EDGE + x1) * E + c]);
    }
    const std::string tag = " " + std::to_string(gy) + "x" + std::to_string(gx) + " patches";
    verdict("class token row" + tag, worst, 0.0);
    verdict("position table against double" + tag, rel_l2(x.data() + E, ref.data(), ref.size()), 2e-6);
    cudaFree(d_x);
    cudaFree(d_tab);
    cudaFree(d_cls);
}

// q and k turned on their interleaved pairs (pairs 0..15 by the column + 1, 16..31 by the row + 1, the class token not
// at all), v as it is: the fp16 head-major rows against double precision.
void check_rotation(int gx, int gy) {
    const int n = gx * gy + 1;
    const std::vector<float> g = randn((size_t) n * 3 * E, 1.f), bias = randn(3 * E, 0.1f);
    float *d_g = nullptr, *d_b = nullptr, *d_tab = nullptr;
    __half *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    cudaMalloc(&d_g, g.size() * 4);
    cudaMalloc(&d_b, bias.size() * 4);
    cudaMalloc(&d_tab, (size_t) n * 64 * 4);
    for (__half** p : {&d_q, &d_k, &d_v}) cudaMalloc(p, (size_t) H * n * D * 2);
    cudaMemcpy(d_g, g.data(), g.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice);
    launch_rope_axial(d_tab, n, gx, nullptr);
    // two chunks of rows, as the encoder runs them
    const int half = n / 2;
    launch_qkv_axial(d_g, d_b, d_tab, d_q, d_k, d_v, n, 0, half, nullptr);
    launch_qkv_axial(d_g + (size_t) half * 3 * E, d_b, d_tab, d_q, d_k, d_v, n, half, n - half, nullptr);
    std::vector<__half> q((size_t) H * n * D), k = q, v = q;
    cudaMemcpy(q.data(), d_q, q.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(k.data(), d_k, k.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(v.data(), d_v, v.size() * 2, cudaMemcpyDeviceToHost);
    std::vector<float> got, ref;
    for (int t = 0; t < n; ++t) {
        int py = -1, px = -1;
        if (t > 0) patch_at(t - 1, gx, py, px);
        for (int part = 0; part < 3; ++part)
            for (int h = 0; h < H; ++h)
                for (int j = 0; j < 32; ++j) {
                    const size_t s = (size_t) t * 3 * E + part * E + h * D + 2 * j;
                    const double a = (double) g[s] + bias[part * E + h * D + 2 * j], b = (double) g[s + 1] + bias[part * E + h * D + 2 * j + 1];
                    double ra = a, rb = b;
                    if (part < 2) {
                        const double th = (j < 16 ? px + 1 : py + 1) * std::pow(10000.0, -(double) (j % 16) / 16.0);
                        ra = a * std::cos(th) - b * std::sin(th);
                        rb = b * std::cos(th) + a * std::sin(th);
                    }
                    const std::vector<__half>& dst = part == 0 ? q : part == 1 ? k : v;
                    const size_t o = ((size_t) h * n + t) * D + 2 * j;
                    got.push_back(__half2float(dst[o]));
                    got.push_back(__half2float(dst[o + 1]));
                    ref.push_back((float) ra);
                    ref.push_back((float) rb);
                }
    }
    verdict("q, k turned, v kept " + std::to_string(gy) + "x" + std::to_string(gx) + " patches", rel_l2(got.data(), ref.data(), got.size()), 1e-3);
    for (void* p : {(void*) d_g, (void*) d_b, (void*) d_tab, (void*) d_q, (void*) d_k, (void*) d_v}) cudaFree(p);
}

// The CUDA encoder against the CPU encoder per stage on a w x h picture, and the merge + projection against double.
void check_encoder(const basalt::core::FileRange& fr, int w, int h) {
    Mmproj mp;
    std::string err;
    if (!mp.open(fr, err, "class_token_test")) { std::printf("open: %s\n", err.c_str()); g_fail++; return; }
    CpuEncoder cpu;
    if (!cpu.init(mp, 0, true, false, err)) { std::printf("cpu init: %s\n", err.c_str()); g_fail++; return; }
    const int cols = (w + 2 * P - 1) / (2 * P), rows = (h + 2 * P - 1) / (2 * P), m = cols * rows;
    CudaEncoder gpu;
    if (!gpu.init(mp, m, err)) { std::printf("cuda init: %s\n", err.c_str()); g_fail++; return; }
    const Image img = random_picture(w, h);
    std::map<std::string, std::vector<float>> c, d;
    std::vector<float> oc, od;
    auto tap_c = [&](const std::string& n, const float* p, size_t k) { c[n].assign(p, p + k); };
    auto tap_d = [&](const std::string& n, const float* p, size_t k) { d[n].assign(p, p + k); };
    if (!cpu.encode_padded(img, oc, tap_c) || !gpu.encode_padded(img, od, tap_d)) {
        std::printf("encode failed: %s\n", gpu.last_error().c_str());
        g_fail++;
        return;
    }
    const std::string tag = " " + std::to_string(w) + "x" + std::to_string(h) + " (" + std::to_string(m) + " cells)";
    verdict("embed (class token, patches + positions)" + tag, rel_l2(d["embed"].data(), c["embed"].data(), c["embed"].size()), 1e-5);
    verdict("pre-norm" + tag, rel_l2(d["pre_ln"].data(), c["pre_ln"].data(), c["pre_ln"].size()), 1e-5);
    for (int l = 0; l < LAYERS; ++l) {
        const std::string n = "layer_out." + std::to_string(l);
        verdict(n + tag, rel_l2(d[n].data(), c[n].data(), c[n].size()), 3e-2);
    }
    verdict("rows" + tag, (double) od.size(), (double) m * PROJ, false);
    verdict("rows (not more)" + tag, (double) od.size(), (double) m * PROJ);
    verdict("output, GPU against CPU" + tag, rel_l2(od.data(), oc.data(), oc.size()), 3e-2);
    // independent: the merge (a cell's four patch rows in merge order) and both products in double from the CPU encoder's
    // last layer, bf16-rounded as the products take them
    const std::vector<float>& x = c["layer_out." + std::to_string(LAYERS - 1)];
    const std::vector<float>& w1 = W["mm.1.weight"];   // [4E][4E]
    const std::vector<float>& w2 = W["mm.2.weight"];   // [PROJ][4E]
    double worst = 0;
    for (int cell : {0, m / 2, m - 1}) {
        std::vector<double> mid(4 * E);
        for (int o = 0; o < 4 * E; ++o) {
            double s = 0;
            for (int k = 0; k < 4 * E; ++k) s += (double) round_bf16(x[(size_t) (1 + 4 * cell) * E + k]) * w1[(size_t) o * 4 * E + k];
            mid[(size_t) o] = round_bf16((float) (0.5 * s * (1.0 + std::erf(s / std::sqrt(2.0)))));
        }
        double d2 = 0, r2 = 0;
        for (int o = 0; o < PROJ; o += 13) {
            double s = 0;
            for (int k = 0; k < 4 * E; ++k) s += mid[(size_t) k] * w2[(size_t) o * 4 * E + k];
            d2 += (od[(size_t) cell * PROJ + o] - s) * (od[(size_t) cell * PROJ + o] - s);
            r2 += s * s;
        }
        worst = std::max(worst, std::sqrt(d2 / std::max(r2, 1e-300)));
    }
    verdict("merge + projection against double" + tag, worst, 2e-2);
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

// The tower's own kernels at the largest picture (103 x 103 cells: 42436 patches, 42437 rows), each against the bytes it moves.
void bench() {
    const int cols = 103, gx = 2 * cols, n = gx * gx, T = n + 1, w = cols * 2 * P;
    uint8_t* d_img = nullptr;
    __nv_bfloat16* d_rows = nullptr;
    float *x = nullptr, *g = nullptr, *tab = nullptr, *vec = nullptr, *pos = nullptr;
    __half *q = nullptr, *k = nullptr, *v = nullptr;
    cudaMalloc(&d_img, (size_t) w * w * 3);
    cudaMalloc(&d_rows, (size_t) n * 3 * P * P * 2);
    cudaMalloc(&x, (size_t) T * E * 4);
    cudaMalloc(&g, (size_t) T * 3 * E * 4);
    cudaMalloc(&tab, (size_t) T * 64 * 4);
    cudaMalloc(&vec, (size_t) 3 * E * 4);
    cudaMalloc(&pos, (size_t) EDGE * EDGE * E * 4);
    for (__half** p : {&q, &k, &v}) cudaMalloc(p, (size_t) H * T * D * 2);
    cudaMemset(d_img, 100, (size_t) w * w * 3);
    cudaMemset(g, 0, (size_t) T * 3 * E * 4);
    cudaMemset(x, 0, (size_t) T * E * 4);
    cudaMemset(vec, 0, (size_t) 3 * E * 4);
    cudaMemset(pos, 0, (size_t) EDGE * EDGE * E * 4);
    launch_rope_axial(tab, T, gx, nullptr);
    const PixelNorm nm{{127.5f, 127.5f, 127.5f}, {127.5f, 127.5f, 127.5f}};
    auto line = [](const char* what, double us, double bytes) {
        std::printf("  bench %-34s %9.1f us/call  %6.0f GB/s\n", what, us, bytes / (us * 1e-6) / 1e9);
    };
    line("patch rows (4120 x 4120 picture)", time_us([&] { launch_patch_rows(d_img, w, w, gx, P, nm, d_rows, 0, n, nullptr); }),
         (double) w * w * 3 + (double) n * 3 * P * P * 2);
    line("class row + positions (42437 rows)", time_us([&] { launch_embed_cls(x, vec, pos, gx, gx, EDGE, E, nullptr); }),
         (double) T * E * 8);
    line("rotation table (42437 rows)", time_us([&] { launch_rope_axial(tab, T, gx, nullptr); }), (double) T * 64 * 4);
    line("q, k, v epilogue (42437 rows)", time_us([&] { launch_qkv_axial(g, vec, tab, q, k, v, T, 0, T, nullptr); }),
         (double) T * 3 * E * 4 + (double) T * 3 * E * 2 + (double) T * 64 * 4);
    line("scaled residual (42437 x 1024)", time_us([&] { launch_bias_scale_residual(x, g, vec, vec, T, E, nullptr); }),
         (double) T * E * 12);
    line("fp32 -> bf16 (42436 x 1024)", time_us([&] { launch_to_bf16(x + E, d_rows, (size_t) n * E, nullptr); }),
         (double) n * E * 6);
    line("layernorm in place (42437 x 1024)", time_us([&] { launch_layernorm_inplace(x, vec, vec, T, 1e-5f, nullptr); }),
         (double) T * E * 8);
    for (void* p : {(void*) d_img, (void*) d_rows, (void*) x, (void*) g, (void*) tab, (void*) vec, (void*) pos, (void*) q,
                    (void*) k, (void*) v})
        cudaFree(p);
}

}  // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) { std::printf("no CUDA device\n"); return 77; }
    const std::string path = "cls_encoder_parity.gguf";
    write_gguf(path.c_str());
    check_patch_rows(57, 23);
    check_patch_rows(40, 40);
    check_patch_rows(121, 83);
    check_positions(2, 2);     // smaller than the 32 x 32 table
    check_positions(10, 6);
    check_positions(32, 32);   // the table itself
    check_positions(64, 40);   // larger than it
    check_rotation(6, 4);
    basalt::core::FileRange fr;
    fr.path = path;
    check_encoder(fr, 57, 23);       // 2 x 1 cells, padded both ways
    check_encoder(fr, 240, 160);     // 6 x 4 cells: patches fewer than the table's
    check_encoder(fr, 1281, 640);    // 33 x 16 cells, the grid wider than the table
    check_encoder(fr, 1840, 1840);   // 46 x 46 cells: the sequence past one chunk of product rows
    std::remove(path.c_str());
    bench();
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
