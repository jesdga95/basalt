// tests/vision/cpu_parity.cpp - the CPU encoder against tensors saved from a reference run.
//
//   vision_cpu_parity <mmproj.gguf> <image> <reference dir> [threads] [bf16]
//
// The reference directory holds raw float32 files: inp_pos_emb.f32, layer_out-N.f32, norm_b-27.f32 (the post
// layer norm) and out.f32, plus the FFN taps of layer 0.  Prints the relative L2 error and the maximum absolute
// error of each tap.  Exit status 1 when a tensor is missing or an error exceeds its gate: 1e-6 for the patch + position
// embedding, 5e-5 for the layer-0 attention half (no rounding flips yet), 1.5e-1 after that, where a bf16 rounding
// flip in one element spreads: the reference's own runs with and without flash attention differ by 1e-2 after
// 5 layers and 1e-1 at the output on a photo.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "basalt/vision/encoder_cpu.hpp"

using namespace basalt::vision;

static bool read_f32(const std::string& path, std::vector<float>& v) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    const std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    v.resize(b.size() / 4);
    std::copy(b.begin(), b.begin() + v.size() * 4, (char*) v.data());
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: vision_cpu_parity <mmproj> <image> <refdir> [threads] [bf16]\n"); return 2; }
    const std::string refdir = argv[3];
    const int threads = argc > 4 ? std::atoi(argv[4]) : 2;
    const bool bf16 = argc > 5 && std::atoi(argv[5]) != 0;
    std::string err;
    Mmproj mp;
    if (!mp.open(argv[1], err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    std::ifstream f(argv[2], std::ios::binary);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Image img;
    if (!decode_image(bytes.data(), bytes.size(), img, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    const Geometry g;
    const Size sz = fit_size(img.w, img.h, g);
    const Image canvas = to_canvas(img, sz);
    std::vector<float> planar, patches;
    normalize_planar(canvas, planar);
    patchify(planar, sz, g, patches);
    std::printf("image %dx%d -> canvas %dx%d, %d tokens\n", img.w, img.h, sz.w, sz.h, sz.w / 32 * (sz.h / 32));
    CpuEncoder enc;
    if (!enc.init(mp, threads, bf16, true, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    int bad = 0;
    auto check = [&](const std::string& name, const float* a, size_t n) {
        std::string ref = name == "embed" ? "inp_pos_emb" : name == "post_ln" ? "norm_b-27" : name;
        if (ref.rfind("layer_out.", 0) == 0) ref = "layer_out-" + ref.substr(10);
        for (const char* pre : {"ffn_inp", "ffn_up_b", "ffn_gelu", "ffn_down"})
            if (ref.rfind(std::string(pre) + ".", 0) == 0) ref = std::string(pre) + "-" + ref.substr(std::strlen(pre) + 1);
        std::vector<float> r;
        if (!read_f32(refdir + "/" + ref + ".f32", r)) { if (name == "out" || name == "embed") { std::printf("%-14s no reference\n", name.c_str()); ++bad; } return; }
        if (r.size() != n) { std::printf("%-14s size %zu vs %zu\n", name.c_str(), n, r.size()); ++bad; return; }
        double d2 = 0, r2 = 0, mx = 0;
        for (size_t i = 0; i < n; ++i) {
            const double d = (double) a[i] - r[i];
            d2 += d * d; r2 += (double) r[i] * r[i]; mx = std::max(mx, std::fabs(d));
        }
        const double rel = std::sqrt(d2 / std::max(r2, 1e-30));
        const double gate = name == "embed" ? 1e-6 : name == "ffn_inp.0" ? 5e-5 : 1.5e-1;
        std::printf("%-14s rel L2 %.3e  max abs %.3e%s\n", name.c_str(), rel, mx, rel > gate ? "  FAIL" : "");
        if (rel > gate) ++bad;
    };
    std::vector<float> out;
    if (!enc.encode(patches, sz, out, check)) { std::fprintf(stderr, "encode failed\n"); return 2; }
    return bad ? 1 : 0;
}
