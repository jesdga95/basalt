// src/models/inkling2/vision_parity.cpp - Inkling 2's image encoder on the real projector: per picture its plan, the CUDA
// encoder against the CPU encoder stage by stage (--cpu), the encode time, and the rows written for an outside reference.
//
//   inkling2_vision_parity <model.basalt | mmproj.gguf> [--cpu] [--dump DIR] [--reps N] <image>...
//
// --cpu: per tap relative L2 of the CUDA encoder against the CPU encoder run with the same bf16 rounding of the product
// inputs, and the lowest cosine of an output row. --dump: DIR/<name>.basalt.f32 (and .cpu.f32 with --cpu), the rows
// [cells][2304] as float32. The time is the mean of N encodes after a first one (the picture's bytes to the card and the
// rows back included), next to the host's decode and preparation. Exit status 1 when a gate fails.
#include "basalt/core/model_file.hpp"
#include "basalt/vision/encoder_cpu.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "basalt/vision/mmproj.hpp"
#include "models/inkling2/vision.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

using namespace basalt;
using namespace basalt::models::inkling2;

namespace {

int g_fail = 0;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

double rel_l2(const float* a, const float* ref, size_t n) {
    double d = 0, r = 0;
    for (size_t i = 0; i < n; ++i) { d += ((double) a[i] - ref[i]) * ((double) a[i] - ref[i]); r += (double) ref[i] * ref[i]; }
    return std::sqrt(d / std::max(r, 1e-300));
}

double min_row_cosine(const std::vector<float>& a, const std::vector<float>& b, int width) {
    double mn = 1;
    for (size_t r = 0; r * width < a.size(); ++r) {
        double d = 0, na = 0, nb = 0;
        for (int c = 0; c < width; ++c) {
            const double x = a[r * width + c], y = b[r * width + c];
            d += x * y; na += x * x; nb += y * y;
        }
        mn = std::min(mn, d / std::sqrt(std::max(na * nb, 1e-300)));
    }
    return mn;
}

void gate(const std::string& what, double got, double limit) {
    const bool ok = got <= limit;
    std::printf("    %-24s %10.3e (<= %.0e)%s\n", what.c_str(), got, limit, ok ? "" : "  FAIL");
    g_fail += !ok;
}

std::string stem(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    std::string s = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t dot = s.find_last_of('.');
    return dot == std::string::npos ? s : s.substr(0, dot);
}

bool write_f32(const std::string& path, const std::vector<float>& v) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    const bool ok = f && std::fwrite(v.data(), sizeof(float), v.size(), f) == v.size();
    if (f) std::fclose(f);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, dump;
    bool cpu_ref = false;
    int reps = 3;
    std::vector<std::string> images;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cpu") cpu_ref = true;
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
        else if (model.empty()) model = a;
        else images.push_back(a);
    }
    if (model.empty() || images.empty()) {
        std::fprintf(stderr, "usage: inkling2_vision_parity <model.basalt | mmproj.gguf> [--cpu] [--dump DIR] [--reps N] <image>...\n");
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
    if (!gpu.init(mp, kMaxCells, err)) { std::printf("cuda init: %s\n", err.c_str()); return 2; }
    vision::CpuEncoder cpu;
    if (cpu_ref && !cpu.init(mp, 0, true, false, err)) { std::printf("cpu init: %s\n", err.c_str()); return 2; }
    const int width = mp.hp().proj;
    std::printf("encoder: %d layers, %d wide, device region %.0f MiB (weights %.0f MiB) for %d cells\n", mp.hp().layers,
                mp.hp().embd, gpu.device_bytes() / 1048576.0, gpu.weights_bytes() / 1048576.0, kMaxCells);
    for (const std::string& path : images) {
        std::ifstream f(path, std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const auto t0 = Clock::now();
        vision::Image img;
        std::vector<uint8_t> alpha;
        ImagePlan plan;
        if (bytes.empty() || !vision::decode_image(bytes.data(), bytes.size(), img, err, vision::kMaxImagePixels, &alpha) ||
            !plan_image(img.w, img.h, plan, err)) {
            std::printf("%s: %s\n", path.c_str(), bytes.empty() ? "cannot read" : err.c_str());
            ++g_fail;
            continue;
        }
        const double t_decode = ms_since(t0);
        const auto t1 = Clock::now();
        vision::Image prepared;   // as the engine's encoder: a picture with alpha or scaled is prepared, else read as decoded
        if (!alpha.empty() || plan.scaled) prepared = encoder_input(img, alpha, plan);
        const vision::Image& in = prepared.rgb.empty() ? img : prepared;
        const double t_prep = ms_since(t1);
        std::printf("%s: %dx%d%s -> %d x %d cells%s\n", stem(path).c_str(), img.w, img.h, alpha.empty() ? "" : " with alpha",
                    plan.cols, plan.rows, plan.scaled ? (" (scaled to " + std::to_string(plan.w) + "x" + std::to_string(plan.h) + ")").c_str() : "");
        std::map<std::string, std::vector<float>> taps_d;
        std::vector<float> out;
        auto tap_d = [&](const std::string& n, const float* p, size_t k) {
            if (cpu_ref) taps_d[n].assign(p, p + k);
        };
        if (!gpu.encode_padded(in, out, tap_d)) { std::printf("  encode: %s\n", gpu.last_error().c_str()); ++g_fail; continue; }
        const auto t2 = Clock::now();
        for (int r = 0; r < reps; ++r) gpu.encode_padded(in, out);
        const double t_enc = ms_since(t2) / reps;
        std::printf("  decode %.1f ms, prepare %.1f ms, encode %.2f ms (%d tokens in the tower)\n", t_decode, t_prep, t_enc,
                    4 * plan.cells() + 1);
        if (out.size() != (size_t) plan.cells() * width) { std::printf("  %zu values, not %d rows\n", out.size(), plan.cells()); ++g_fail; }
        if (!dump.empty() && !write_f32(dump + "/" + stem(path) + ".basalt.f32", out)) { std::printf("  cannot write the dump\n"); ++g_fail; }
        if (!cpu_ref) continue;
        std::map<std::string, std::vector<float>> taps_c;
        std::vector<float> oc;
        const auto t3 = Clock::now();
        if (!cpu.encode_padded(in, oc, [&](const std::string& n, const float* p, size_t k) { taps_c[n].assign(p, p + k); })) {
            std::printf("  cpu encode failed\n");
            ++g_fail;
            continue;
        }
        std::printf("  cpu encode %.0f ms\n", ms_since(t3));
        if (!dump.empty()) write_f32(dump + "/" + stem(path) + ".cpu.f32", oc);
        const int L = mp.hp().layers;
        for (const std::string& n : {std::string("embed"), std::string("pre_ln"), std::string("layer_out.0"),
                                    "layer_out." + std::to_string(L / 2), "layer_out." + std::to_string(L - 1)}) {
            const auto &a = taps_d[n], &b = taps_c[n];
            if (a.size() != b.size() || a.empty()) { std::printf("    tap %s missing\n", n.c_str()); ++g_fail; continue; }
            gate(n, rel_l2(a.data(), b.data(), a.size()), n == "embed" || n == "pre_ln" ? 1e-5 : 3e-2);
        }
        gate("out", rel_l2(out.data(), oc.data(), oc.size()), 3e-2);
        std::printf("    %-24s %10.6f\n", "min row cosine", min_row_cosine(out, oc, width));
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
