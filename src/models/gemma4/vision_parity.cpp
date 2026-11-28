// src/models/gemma4/vision_parity.cpp - Gemma 4's image encoder on the real projector: per picture its plan, the encode time
// and the rows, written for an outside reference with the picture the encoder read.
//
//   gemma4_vision_parity <model.basalt | mmproj.gguf> [--dump DIR] [--reps N] [--cells N] <image>...
//
// --dump: DIR/<name>.rgb (the prepared picture, interleaved u8), DIR/<name>.dims (its width and height), DIR/<name>.f32 (the
// rows [cells][width] as float32) and DIR/<name>.pooled.f32 (the pooled rows before the projection). The time is the mean of
// N encodes after a first one (the picture's bytes to the card and the rows back included). --cells: the soft-token budget
// (default 280, the model's). Exit status 1 when an encode fails.
#include "basalt/core/model_file.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "basalt/vision/mmproj.hpp"
#include "models/gemma4/vision.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace basalt;
using namespace basalt::models::gemma4;

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

std::string stem(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    std::string s = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t dot = s.find_last_of('.');
    return dot == std::string::npos ? s : s.substr(0, dot);
}

bool write_bytes(const std::string& path, const void* p, size_t n) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    const bool ok = f && std::fwrite(p, 1, n, f) == n;
    if (f) std::fclose(f);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, dump;
    int reps = 3, cells = 280, fail = 0;
    std::vector<std::string> images;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dump" && i + 1 < argc) dump = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
        else if (a == "--cells" && i + 1 < argc) cells = std::max(1, std::atoi(argv[++i]));
        else if (model.empty()) model = a;
        else images.push_back(a);
    }
    if (model.empty() || images.empty()) {
        std::fprintf(stderr, "usage: gemma4_vision_parity <model.basalt | mmproj.gguf> [--dump DIR] [--reps N] [--cells N] <image>...\n");
        return 2;
    }
    std::string err;
    core::ModelFile file;
    core::FileRange range(model);
    if (model.size() > 7 && model.compare(model.size() - 7, 7, ".basalt") == 0) {
        if (!file.open(model, err) || file.range("vision/mmproj").empty()) {
            std::printf("no vision/mmproj in %s %s\n", model.c_str(), err.c_str());
            return 2;
        }
        range = file.range("vision/mmproj");
    }
    vision::Mmproj mp;
    if (!mp.open(range, err, kProjector)) { std::printf("%s\n", err.c_str()); return 2; }
    vision::CudaEncoder gpu;
    if (!gpu.init(mp, cells, err)) { std::printf("cuda init: %s\n", err.c_str()); return 2; }
    const int patch = mp.hp().patch, k = mp.hp().merge, width = mp.hp().proj;
    std::printf("encoder: %d layers, %d wide, cells of %d x %d patches, device region %.0f MiB (weights %.0f MiB) for %d cells\n",
                mp.hp().layers, mp.hp().embd, k, k, gpu.device_bytes() / 1048576.0, gpu.weights_bytes() / 1048576.0, cells);
    for (const std::string& path : images) {
        std::ifstream f(path, std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        vision::Image img;
        std::vector<uint8_t> alpha;
        ImagePlan plan;
        const auto t0 = Clock::now();
        if (bytes.empty() || !vision::decode_image(bytes.data(), bytes.size(), img, err, vision::kMaxImagePixels, &alpha) ||
            !plan_image(img.w, img.h, patch, k, cells, plan, err)) {
            std::printf("%s: %s\n", path.c_str(), bytes.empty() ? "cannot read" : err.c_str());
            ++fail;
            continue;
        }
        const vision::Image in = encoder_input(img, alpha, plan);
        const double t_prep = ms_since(t0);
        std::vector<float> out, pooled;
        auto tap = [&](const std::string& n, const float* p, size_t c) {
            if (n == "pooled") pooled.assign(p, p + c);
        };
        if (!gpu.encode_padded(in, out, tap)) { std::printf("  encode: %s\n", gpu.last_error().c_str()); ++fail; continue; }
        const auto t1 = Clock::now();
        for (int r = 0; r < reps; ++r) gpu.encode_padded(in, out);
        const double t_enc = ms_since(t1) / reps;
        std::printf("%s: %dx%d%s -> %dx%d, %d x %d cells; decode and prepare %.1f ms, encode %.2f ms\n", stem(path).c_str(), img.w,
                    img.h, alpha.empty() ? "" : " with alpha", plan.w, plan.h, plan.cols, plan.rows, t_prep, t_enc);
        if (out.size() != (size_t) plan.cells() * width) { std::printf("  %zu values, not %d rows\n", out.size(), plan.cells()); ++fail; }
        if (dump.empty()) continue;
        const std::string base = dump + "/" + stem(path);
        const std::string dims = std::to_string(in.w) + " " + std::to_string(in.h) + "\n";
        if (!write_bytes(base + ".rgb", in.rgb.data(), in.rgb.size()) || !write_bytes(base + ".dims", dims.data(), dims.size()) ||
            !write_bytes(base + ".f32", out.data(), out.size() * 4) || !write_bytes(base + ".pooled.f32", pooled.data(), pooled.size() * 4)) {
            std::printf("  cannot write the dump\n");
            ++fail;
        }
    }
    std::printf("%s\n", fail ? "FAILED" : "all ok");
    return fail ? 1 : 0;
}
