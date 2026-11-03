// The DeepSeek-V4.1 image encoder when the card is short of room, on random weights (no model file).
//
//   ds41_vision_reserve_test
//
// Writes a one-layer deepseek41v projector and encodes a small and a large picture with fresh encoders as the reference.
// Then, with the card filled: the projector upload and the large picture's work buffers are refused with no CUDA error
// left behind, and the small picture still reserves. With the room given back (what the expert tier's lend does), the
// same encoder encodes both pictures bit for bit as the fresh ones did. Exit status 77 without a CUDA device, 1 when a
// check fails.
#include "basalt/core/gpu_mem_info.hpp"
#include "basalt/models/deepseek41/vision.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace basalt::models::deepseek41;

namespace {

constexpr int E = 1024, F = 2816, QKV = 3 * E, MERGED = 9 * E, PROJ = 256;
int g_fail = 0;

void check(const std::string& what, bool ok) {
    std::printf("%-72s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    g_fail += !ok;
}

// ---- a GGUF writer (tensors as flat rows: the projector checks type and element count) ----------------------------------
struct Gguf {
    std::string kv, infos;
    std::vector<uint8_t> data;
    int n_kv = 0, n_t = 0;
    template <class T> static void put(std::string& s, T v) { s.append((const char*) &v, sizeof v); }
    static void str(std::string& s, const std::string& v) { put<uint64_t>(s, v.size()); s += v; }
    void kv_str(const char* k, const char* v) { str(kv, k); put<uint32_t>(kv, 8); str(kv, v); ++n_kv; }
    void kv_u32(const char* k, uint32_t v) { str(kv, k); put<uint32_t>(kv, 4); put(kv, v); ++n_kv; }
    void kv_f32(const char* k, float v) { str(kv, k); put<uint32_t>(kv, 6); put(kv, v); ++n_kv; }
    void kv_f32s(const char* k, const std::vector<float>& v) {
        str(kv, k); put<uint32_t>(kv, 9); put<uint32_t>(kv, 6); put<uint64_t>(kv, v.size());
        for (float x : v) put(kv, x);
        ++n_kv;
    }
    void tensor(const std::string& name, uint32_t type, const void* p, size_t elems, size_t esz) {
        str(infos, name); put<uint32_t>(infos, 1); put<uint64_t>(infos, elems);
        put(infos, type); put<uint64_t>(infos, data.size());
        data.insert(data.end(), (const uint8_t*) p, (const uint8_t*) p + elems * esz);
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

void write_projector(const char* path) {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    Gguf g;
    g.kv_str("general.architecture", "clip");
    g.kv_str("clip.projector_type", "deepseek41v");
    g.kv_u32("clip.vision.image_size", 1024);
    g.kv_u32("clip.vision.patch_size", kVisionPatch);
    g.kv_u32("clip.vision.embedding_length", E);
    g.kv_u32("clip.vision.feed_forward_length", F);
    g.kv_u32("clip.vision.block_count", 1);
    g.kv_u32("clip.vision.attention.head_count", 16);
    g.kv_u32("clip.vision.projection_dim", PROJ);
    g.kv_u32("clip.vision.spatial_merge_size", 3);
    g.kv_f32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
    g.kv_f32s("clip.vision.image_mean", {0.5f, 0.5f, 0.5f});
    g.kv_f32s("clip.vision.image_std", {0.5f, 0.5f, 0.5f});
    const auto f32 = [&](const std::string& n, size_t elems, float scale, float offset = 0.f) {
        std::vector<float> v(elems);
        for (float& x : v) x = offset + scale * nd(rng);
        g.tensor(n, 0, v.data(), elems, 4);
    };
    const auto bf = [&](const std::string& n, size_t elems, size_t fan_in) {
        const float s = 1.f / std::sqrt((float) fan_in);
        std::vector<uint16_t> v(elems);
        for (uint16_t& x : v) x = bf16(s * nd(rng));
        g.tensor(n, 30, v.data(), elems, 2);
    };
    f32("v.patch_embd.weight", (size_t) E * kVisionPatchValues, 1.f / std::sqrt((float) kVisionPatchValues));
    f32("v.patch_embd.bias", E, 0.02f);
    f32("v.post_ln.weight", E, 0.05f, 1.f);
    bf("mm.1.weight", (size_t) MERGED * PROJ, MERGED);
    f32("mm.1.bias", PROJ, 0.02f);
    bf("mm.2.weight", (size_t) PROJ * PROJ, PROJ);
    f32("mm.2.bias", PROJ, 0.02f);
    f32("v.token_embd.img_start", PROJ, 1.f);
    f32("v.token_embd.img_end", PROJ, 1.f);
    f32("v.image_newline", PROJ, 1.f);
    f32("v.blk.0.ln1.weight", E, 0.05f, 1.f);
    f32("v.blk.0.ln2.weight", E, 0.05f, 1.f);
    f32("v.blk.0.attn_qkv.bias", QKV, 0.02f);
    f32("v.blk.0.attn_out.bias", E, 0.02f);
    bf("v.blk.0.attn_qkv.weight", (size_t) QKV * E, E);
    bf("v.blk.0.attn_out.weight", (size_t) E * E, E);
    bf("v.blk.0.ffn_gate.weight", (size_t) F * E, E);
    bf("v.blk.0.ffn_up.weight", (size_t) F * E, E);
    bf("v.blk.0.ffn_down.weight", (size_t) E * F, F);
    const std::string b = g.bytes();
    std::FILE* f = std::fopen(path, "wb");
    if (!f || std::fwrite(b.data(), 1, b.size(), f) != b.size()) { std::printf("cannot write %s\n", path); std::exit(2); }
    std::fclose(f);
}

struct Picture {
    VisionPlan plan;
    std::vector<float> rows;
};

Picture picture(int w, int h, uint32_t seed, const basalt::vision::Mmproj& mp) {
    basalt::vision::Image img;
    img.w = w;
    img.h = h;
    img.rgb.resize((size_t) w * h * 3);
    std::mt19937 rng(seed);
    for (uint8_t& c : img.rgb) c = (uint8_t) (rng() & 0xff);
    Picture p;
    p.plan = plan_image(w, h, mp.hp().merge, mp.hp().min_pixels);
    image_patches(img, p.plan, p.rows);
    return p;
}

// Holds device memory until no more than `left` bytes stay free.
struct Fill {
    std::vector<void*> held;
    size_t free_after = 0;
    explicit Fill(size_t left) {
        size_t chunk = (size_t) 256 << 20;
        for (;;) {
            size_t free_b = 0, total_b = 0;
            basalt::core::gpu_mem_info(&free_b, &total_b);
            free_after = free_b;
            if (free_b <= left) break;
            const size_t want = std::min(chunk, free_b - left);
            void* p = nullptr;
            if (want >= ((size_t) 1 << 20) && cudaMalloc(&p, want) == cudaSuccess) {
                held.push_back(p);
                continue;
            }
            cudaGetLastError();
            if (chunk <= ((size_t) 2 << 20) || want < ((size_t) 1 << 20)) break;
            chunk /= 2;
        }
        cudaGetLastError();
    }
    ~Fill() {
        for (void* p : held) cudaFree(p);
    }
};

bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool finite(const std::vector<float>& a) {
    return !a.empty() && std::all_of(a.begin(), a.end(), [](float x) { return std::isfinite(x); });
}

}  // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) { std::printf("no CUDA device\n"); return 77; }
    const char* path = "ds41_vision_reserve_test.gguf";
    write_projector(path);
    std::string err;
    basalt::vision::Mmproj mp;
    if (!mp.open(basalt::core::FileRange(path), err, "deepseek41v")) { std::printf("%s\n", err.c_str()); std::remove(path); return 1; }
    const Picture small = picture(100, 80, 1, mp), large = picture(1800, 1350, 2, mp);
    std::printf("small %d patches, %d rows; large %d patches, %d rows\n", small.plan.n_patches(), small.plan.n_tokens(),
                large.plan.n_patches(), large.plan.n_tokens());
    VisionBlas blas;

    // the reference: each picture on a fresh encoder
    std::vector<float> ref_small, ref_large;
    for (const Picture* p : {&small, &large}) {
        VisionEncoder fresh;
        err.clear();
        const bool ok = fresh.init(mp, blas, err) && fresh.encode(p->rows, p->plan, p == &small ? ref_small : ref_large, err);
        check("fresh encoder, " + std::string(p == &small ? "small" : "large") + " picture" + (ok ? "" : ": " + err), ok);
        if (!ok) { std::remove(path); return 1; }
    }
    check("the reference blocks are finite", finite(ref_small) && finite(ref_large));

    VisionEncoder enc;
    err.clear();
    check("encoder init", enc.init(mp, blas, err));
    std::vector<float> out;
    err.clear();
    check("small picture before the card is filled", enc.encode(small.rows, small.plan, out, err) && same(out, ref_small));
    {
        const Fill fill((size_t) 64 << 20);   // far below the large picture's buffers, above the small one's
        std::printf("card filled: %zu MiB free\n", fill.free_after >> 20);
        err.clear();
        const bool reserved = enc.reserve(large.plan, err);
        check("large picture's work buffers are refused" + (reserved ? std::string() : ": " + err),
              !reserved && err.find("work buffers") != std::string::npos);
        check("no CUDA error left behind by the refusal", cudaGetLastError() == cudaSuccess);
        err.clear();
        check("small picture still reserves in the room left", enc.reserve(small.plan, err));
        check("no CUDA error left behind", cudaGetLastError() == cudaSuccess);
    }
    {
        const Fill fill((size_t) 8 << 20);   // below the projector's weights
        std::printf("card filled: %zu MiB free\n", fill.free_after >> 20);
        VisionEncoder late;
        err.clear();
        const bool loaded = late.init(mp, blas, err);
        check("projector upload is refused" + (loaded ? std::string() : ": " + err), !loaded && !err.empty());
        check("no CUDA error left behind by the refusal", cudaGetLastError() == cudaSuccess);
    }
    // the room is back: the same encoder grows its buffers again and encodes as a fresh one does
    err.clear();
    const bool ok_large = enc.encode(large.rows, large.plan, out, err);
    check("large picture after the room is back" + (ok_large ? std::string() : ": " + err), ok_large && same(out, ref_large));
    err.clear();
    const bool ok_small = enc.encode(small.rows, small.plan, out, err);
    check("small picture after the large one" + (ok_small ? std::string() : ": " + err), ok_small && same(out, ref_small));
    check("no CUDA error at the end", cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess);
    std::remove(path);
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
