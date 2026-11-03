// tests/vision/hy4_encoder_parity.cpp - Hy4-flash's tower on the shared encoder, on random weights (no model file).
//
//   hy4_encoder_parity
//
// Writes a projector GGUF of the hy_v4_flash layout (two layers, a 24 x 24 position table, llama.cpp's tensor names and types:
// F32 convolutions, BF16 products) and checks: the position-table resampling kernel against torch's
// interpolate(scale_factor=(size + 0.1) / table, bilinear, align_corners=False) on a table of 8 x 8; the CUDA encoder against the
// CPU encoder per stage on a grid smaller and a grid larger than the table; the block of rows (a newline row after each line of
// merged rows, the newline equal to the file's); and the encoder as a whole against an independent double-precision
// reference of the merge and projection. Exit status 1 when a gate fails.
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

constexpr int E = 1152, F = 4304, LAYERS = 2, EDGE = 24, PROJ = 3072;
int g_fail = 0;
std::mt19937 rng(11);

void verdict(const std::string& what, double got, double gate, bool less = true) {
    const bool ok = less ? got <= gate : got >= gate;
    std::printf("%-52s %10.3e (%s %.1e) %s\n", what.c_str(), got, less ? "<=" : ">=", gate, ok ? "ok" : "FAIL");
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

std::vector<float> randn(size_t n, float sd, float mean = 0.f) {
    std::normal_distribution<float> nd(mean, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
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

std::string write_gguf(const char* path) {
    Gguf g;
    g.kv_str("general.architecture", "clip");
    g.kv_str("clip.projector_type", "hy_v4_flash");
    g.kv_u32("clip.vision.image_size", 2048);
    g.kv_u32("clip.vision.patch_size", 16);
    g.kv_u32("clip.vision.embedding_length", E);
    g.kv_u32("clip.vision.feed_forward_length", F);
    g.kv_u32("clip.vision.block_count", LAYERS);
    g.kv_u32("clip.vision.attention.head_count", 16);
    g.kv_u32("clip.vision.projection_dim", PROJ);
    g.kv_u32("clip.vision.spatial_merge_size", 2);
    g.kv_f32("clip.vision.attention.layer_norm_epsilon", 1e-5f);
    g.kv_f32s("clip.vision.image_mean", {0.5f, 0.5f, 0.5f});
    g.kv_f32s("clip.vision.image_std", {0.5f, 0.5f, 0.5f});
    add_f32(g, "v.patch_embd.weight", {16, 16, 3, E}, randn((size_t) E * 768, 0.03f));
    add_f32(g, "v.patch_embd.bias", {E}, randn(E, 0.02f));
    add_f32(g, "v.position_embd.weight", {E, EDGE * EDGE}, randn((size_t) E * EDGE * EDGE, 0.1f));
    for (int l = 0; l < LAYERS; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        add_f32(g, p + "ln1.weight", {E}, randn(E, 0.1f, 1.f)); add_f32(g, p + "ln1.bias", {E}, randn(E, 0.05f));
        add_f32(g, p + "ln2.weight", {E}, randn(E, 0.1f, 1.f)); add_f32(g, p + "ln2.bias", {E}, randn(E, 0.05f));
        for (const char* n : {"attn_q", "attn_k", "attn_v"}) {
            add_bf16(g, p + n + ".weight", {E, E}, randn((size_t) E * E, 0.03f));
            add_f32(g, p + n + ".bias", {E}, randn(E, 0.02f));
        }
        add_bf16(g, p + "attn_out.weight", {E, E}, randn((size_t) E * E, 0.03f)); add_f32(g, p + "attn_out.bias", {E}, randn(E, 0.02f));
        add_bf16(g, p + "ffn_up.weight", {E, F}, randn((size_t) E * F, 0.03f)); add_f32(g, p + "ffn_up.bias", {F}, randn(F, 0.02f));
        add_bf16(g, p + "ffn_down.weight", {F, E}, randn((size_t) E * F, 0.03f)); add_f32(g, p + "ffn_down.bias", {E}, randn(E, 0.02f));
    }
    add_f32(g, "mm.pre_norm.weight", {E}, randn(E, 0.1f, 1.f));
    add_f32(g, "mm.0.weight", {2, 2, E, 4 * E}, randn((size_t) 4 * E * 4 * E, 0.02f));
    add_f32(g, "mm.0.bias", {4 * E}, randn(4 * E, 0.02f));
    add_f32(g, "mm.2.weight", {1, 1, 4 * E, PROJ}, randn((size_t) PROJ * 4 * E, 0.02f));
    add_f32(g, "mm.2.bias", {PROJ}, randn(PROJ, 0.02f));
    add_f32(g, "v.image_newline", {PROJ}, randn(PROJ, 0.5f));
    const std::string b = g.bytes();
    std::FILE* f = std::fopen(path, "wb");
    if (!f || std::fwrite(b.data(), 1, b.size(), f) != b.size()) { std::printf("cannot write %s\n", path); std::exit(2); }
    std::fclose(f);
    return path;
}

double rel_l2(const float* a, const float* ref, size_t n) {
    double d = 0, r = 0;
    for (size_t i = 0; i < n; ++i) { d += ((double) a[i] - ref[i]) * ((double) a[i] - ref[i]); r += (double) ref[i] * ref[i]; }
    return std::sqrt(d / std::max(r, 1e-300));
}

// merge-order patch rows of a random picture of gx x gy patches
std::vector<float> random_patches(int gx, int gy) { return randn((size_t) gx * gy * 768, 0.7f); }

// the position table against torch: x = 0, bias = 0
void check_interp() {
    constexpr int T = 8;
    std::vector<float> table((size_t) T * T * E);
    for (int y = 0; y < T; ++y)
        for (int x = 0; x < T; ++x) {
            const float v = (float) (((y * 5 + x * 3) % 7) * 1.1 + y * x * 0.07);
            for (int c = 0; c < E; ++c) table[((size_t) y * T + x) * E + c] = v + 0.5f * (float) c;
        }
    struct Ref { int gy, gx; std::vector<float> v; };
    const std::vector<Ref> refs = {
        {6, 10, {0.856557369f,2.30967259f,4.55816746f,5.08412886f,2.10315943f,4.72565699f,3.13227463f,2.94147778f,3.38057852f,0.932868779f,4.47213125f,3.99061728f,3.67881727f,3.38701391f,2.8652482f,2.71092224f,4.88965511f,4.19229221f,2.5800786f,5.19106531f,1.58688545f,3.99152207f,3.06192493f,1.8273772f,4.48563433f,5.90378046f,4.53862906f,4.44827032f,4.86782885f,2.94844294f,6.40163994f,3.57093334f,4.01009083f,4.94323015f,2.64095664f,4.93169069f,3.12933612f,3.20167851f,5.65394735f,8.40581989f,3.51639342f,2.87695122f,2.86828351f,4.17408991f,6.59190369f,3.40625119f,5.21416378f,6.0975399f,5.71393347f,6.16319609f,0.631147563f,3.22529912f,5.15060711f,5.49336004f,4.3413558f,5.57771349f,4.68566704f,5.74174213f,6.93623114f,3.92057395f}},
        {8, 4, {1.56951225f,4.72195101f,3.83658481f,2.95121884f,3.44060969f,2.69182897f,4.85402441f,3.72963452f,4.93609762f,3.94829297f,2.96048808f,1.97268355f,2.76939034f,1.91817105f,5.85597515f,5.38036537f,4.64048767f,4.30146313f,3.96243858f,6.15878153f,2.47378039f,5.55792713f,4.97987843f,4.40182972f,3.96926856f,3.52780509f,3.08634186f,7.80951118f,1.80256104f,5.91109705f,5.98182869f,6.05256033f}},
        {2, 2, {3.53853154f,5.27940416f,2.35916686f,6.34249926f}},
    };
    float* d_tab = nullptr;
    cudaMalloc(&d_tab, table.size() * 4);
    cudaMemcpy(d_tab, table.data(), table.size() * 4, cudaMemcpyHostToDevice);
    for (const Ref& r : refs) {
        const int n = r.gx * r.gy;
        float *d_x = nullptr, *d_b = nullptr;
        cudaMalloc(&d_x, (size_t) n * E * 4);
        cudaMalloc(&d_b, E * 4);
        cudaMemset(d_x, 0, (size_t) n * E * 4);
        cudaMemset(d_b, 0, E * 4);
        launch_embed_finish_interp(d_x, d_b, d_tab, n, r.gx, r.gy, T, nullptr);
        std::vector<float> x((size_t) n * E);
        cudaMemcpy(x.data(), d_x, x.size() * 4, cudaMemcpyDeviceToHost);
        // row t is merge order; channel 0 and channel 7 (an offset of 3.5 added by the table)
        double worst = 0;
        for (int t = 0; t < n; ++t) {
            const int g = t / 4, q = t % 4, per = r.gx / 2;
            const int py = (g / per) * 2 + q / 2, px = (g % per) * 2 + q % 2;
            worst = std::max({worst, (double) std::fabs(x[(size_t) t * E] - r.v[(size_t) py * r.gx + px]),
                              (double) std::fabs(x[(size_t) t * E + 7] - 3.5f - r.v[(size_t) py * r.gx + px])});
        }
        verdict("position table " + std::to_string(r.gy) + "x" + std::to_string(r.gx) + " against torch", worst, 2e-5);
        cudaFree(d_x);
        cudaFree(d_b);
    }
    cudaFree(d_tab);
}

// double precision merge, projection and newline of post-encoder rows, to check the end of the pipeline independently
void check_encoder(const basalt::core::FileRange& fr, int gx, int gy) {
    Mmproj mp;
    std::string err;
    if (!mp.open(fr, err, "hy_v4_flash")) { std::printf("open: %s\n", err.c_str()); g_fail++; return; }
    CpuEncoder cpu;
    if (!cpu.init(mp, 0, true, false, err)) { std::printf("cpu init: %s\n", err.c_str()); g_fail++; return; }
    CudaEncoder gpu;
    if (!gpu.init(mp, 1024, err)) { std::printf("cuda init: %s\n", err.c_str()); g_fail++; return; }
    const std::vector<float> patches = random_patches(gx, gy);
    const Size canvas{gx * 16, gy * 16};
    std::map<std::string, std::vector<float>> c, d;
    std::vector<float> oc, od;
    auto tap_c = [&](const std::string& n, const float* p, size_t k) { c[n].assign(p, p + k); };
    auto tap_d = [&](const std::string& n, const float* p, size_t k) { d[n].assign(p, p + k); };
    if (!cpu.encode(patches, canvas, oc, tap_c) || !gpu.encode(patches, canvas, od, tap_d)) { std::printf("encode failed: %s\n", gpu.last_error().c_str()); g_fail++; return; }
    const std::string tag = " " + std::to_string(gy) + "x" + std::to_string(gx) + " patches";
    verdict("embed (patch + position)" + tag, rel_l2(d["embed"].data(), c["embed"].data(), c["embed"].size()), 1e-5);
    for (int l = 0; l < LAYERS; ++l) {
        const std::string n = "layer_out." + std::to_string(l);
        verdict(n + tag, rel_l2(d[n].data(), c[n].data(), c[n].size()), 3e-2);
    }
    verdict("post norm" + tag, rel_l2(d["post_ln"].data(), c["post_ln"].data(), c["post_ln"].size()), 3e-2);
    const int ow = gx / 2, oh = gy / 2;
    verdict("block rows" + tag, (double) od.size(), (double) oh * (ow + 1) * PROJ, false);
    verdict("block rows (not more)" + tag, (double) od.size(), (double) oh * (ow + 1) * PROJ);
    verdict("output block, GPU against CPU" + tag, rel_l2(od.data(), oc.data(), oc.size()), 3e-2);
    // newline rows are the file's row, bit for bit, on both
    double nl = 0;
    for (int r = 0; r < oh; ++r)
        for (int k = 0; k < PROJ; ++k) {
            const float want = W["v.image_newline"][(size_t) k];
            nl = std::max({nl, (double) std::fabs(od[((size_t) r * (ow + 1) + ow) * PROJ + k] - want),
                           (double) std::fabs(oc[((size_t) r * (ow + 1) + ow) * PROJ + k] - want)});
        }
    verdict("newline rows equal the file's" + tag, nl, 0.0);
    // independent: merge and projection in double from the CPU encoder's post-norm rows (bf16-rounded as the product takes them)
    const std::vector<float>& h = c["post_ln"];
    const std::vector<float>& w0 = W["mm.0.weight"];   // [4E][E][2][2]
    const std::vector<float>& w2 = W["mm.2.weight"];   // [PROJ][4E]
    const int m = gx * gy / 4;
    double worst = 0;
    for (int tok : {0, m / 2, m - 1}) {
        std::vector<double> mid(4 * E);
        for (int o = 0; o < 4 * E; ++o) {
            double s = W["mm.0.bias"][(size_t) o];
            for (int p = 0; p < 4; ++p)
                for (int ch = 0; ch < E; ++ch) s += (double) unbf16(bf16(h[((size_t) tok * 4 + p) * E + ch])) * w0[((size_t) o * E + ch) * 4 + p];
            mid[(size_t) o] = unbf16(bf16((float) (0.5 * s * (1.0 + std::erf(s / std::sqrt(2.0))))));
        }
        const int line = tok / ow, col = tok % ow;
        const float* row = &od[((size_t) line * (ow + 1) + col) * PROJ];
        double d2 = 0, r2 = 0;
        for (int o = 0; o < PROJ; o += 17) {
            double s = W["mm.2.bias"][(size_t) o];
            for (int k = 0; k < 4 * E; ++k) s += mid[(size_t) k] * w2[(size_t) o * 4 * E + k];
            d2 += (row[o] - s) * (row[o] - s);
            r2 += s * s;
        }
        worst = std::max(worst, std::sqrt(d2 / std::max(r2, 1e-300)));
    }
    verdict("merge + projection against double" + tag, worst, 2e-2);
}

}  // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) { std::printf("no CUDA device\n"); return 77; }
    check_interp();
    const std::string path = "hy4_encoder_parity.gguf";
    write_gguf(path.c_str());
    basalt::core::FileRange fr;
    fr.path = path;
    check_encoder(fr, 8, 6);     // smaller than the 24 x 24 table
    check_encoder(fr, 40, 24);   // larger than it
    std::remove(path.c_str());
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
