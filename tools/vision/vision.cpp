// tools/vision/vision.cpp - the image half of the engine's multimodal path.
//
// Turns an image into the embeddings the text model reads in place of its <|image_pad|> tokens, with Basalt's own
// encoder (src/vision) and the model's mmproj file (vision encoder + projector).  The engine does the rest: it
// places the rows at the image's pad tokens and gives them their 2-D M-RoPE positions (see --serve GENI in
// generate.cpp).
//
//   basalt-vision --mmproj <mmproj.gguf> [--model <text vocab .gguf>] [--gpu] [--threads N] [--max-tokens N]
//                 [--flash-attn on|off|auto]
//
// Resident: prints "READY <n_embd>", then per stdin line
//   ENC <image path> <output path>   ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//   QUIT
// The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd],
// row i at grid position (x = i % nx, y = i / nx).  --model and --flash-attn are accepted for the callers that
// pass them and have no effect.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#ifdef BASALT_VISION_CUDA
#include "basalt/vision/encoder_cuda.hpp"
#endif
#include "basalt/vision/encoder_cpu.hpp"
#include "basalt/vision/mmproj.hpp"
#include "basalt/vision/preprocess.hpp"

namespace {

using namespace basalt::vision;

// "ENC <image> <out>": paths may contain spaces, so the image path ends at the last " " before the output path;
// the server always writes paths without spaces, but a user calling the tool by hand might not
bool parse_enc(const std::string& line, std::string& img, std::string& out) {
    if (line.rfind("ENC ", 0) != 0) return false;
    const std::string rest = line.substr(4);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    img = rest.substr(0, sp);
    out = rest.substr(sp + 1);
    return !img.empty() && !out.empty();
}

const char* const kUsage =
    "usage: basalt-vision --mmproj <mmproj.gguf> [--model <model.gguf>] [--gpu] [--threads N] [--max-tokens N] "
    "[--flash-attn on|off|auto]\n"
    "  reads \"ENC <image> <output>\" lines on stdin and writes each image's embeddings to <output>;\n"
    "  --gpu runs the encoder on the GPU, else on the CPU\n";

}  // namespace

int main(int argc, char** argv) {
    std::string mmproj;
    bool gpu = false;
    int threads = 0, max_tokens = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            std::printf("%s", kUsage);
            return 0;
        }
        if (a == "--mmproj") mmproj = next();
        else if (a == "--model" || a == "--flash-attn") next();
        else if (a == "--gpu") gpu = true;
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--max-tokens") max_tokens = std::atoi(next().c_str());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (mmproj.empty()) {
        std::fprintf(stderr, "%s", kUsage);
        return 2;
    }
#ifdef BASALT_VISION_CUDA
    if (gpu) {                    // a build made for another card says so here, before it reads any model file
        std::string perr;
        if (!probe_device(perr)) {
            std::printf("ERR this build cannot run on this card: %s\n", perr.c_str());
            std::fflush(stdout);
            return 1;
        }
    }
#endif
    const auto load_t0 = std::chrono::steady_clock::now();
    std::string err;
    Mmproj mp;
    if (!mp.open(mmproj, err)) { std::printf("ERR cannot load the vision encoder %s: %s\n", mmproj.c_str(), err.c_str()); std::fflush(stdout); return 1; }
    Geometry geo;
    if (max_tokens > 0) geo.max_tokens = std::max(max_tokens, geo.min_tokens);
    if (threads <= 0) threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    CpuEncoder cpu;
#ifdef BASALT_VISION_CUDA
    CudaEncoder cuda;
#endif
    if (gpu) {
#ifdef BASALT_VISION_CUDA
        // Sized and warmed at the largest picture before READY: the engine is started after this process and sizes
        // its expert cache from the memory that is left, so nothing may be allocated later.
        if (!cuda.init(mp, geo.max_tokens, err)) { std::printf("ERR cannot load the vision encoder: %s\n", err.c_str()); std::fflush(stdout); return 1; }
        mp.release();   // the weights are on the device: give the mapped file back
#else
        std::printf("ERR this build has no GPU encoder\n");
        std::fflush(stdout);
        return 1;
#endif
    } else if (!cpu.init(mp, threads, true, false, err)) {
        std::printf("ERR cannot load the vision encoder: %s\n", err.c_str());
        std::fflush(stdout);
        return 1;
    }
    auto encode = [&](const std::vector<float>& patches, Size sz, std::vector<float>& out) {
#ifdef BASALT_VISION_CUDA
        if (gpu) return cuda.encode(patches, sz, out);
#endif
        return cpu.encode(patches, sz, out);
    };
    std::fprintf(stderr, "basalt-vision: model files loaded in %.1f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - load_t0).count());
    if (gpu) {
        const auto warm_t0 = std::chrono::steady_clock::now();
        const Image gray{2048, 2048, std::vector<uint8_t>((size_t) 2048 * 2048 * 3, 128)};
        const Size sz = fit_size(gray.w, gray.h, geo);
        std::vector<float> planar, patches, out;
        normalize_planar(to_canvas(gray, sz), planar);
        patchify(planar, sz, geo, patches);
        const bool ok = encode(patches, sz, out);
        std::fprintf(stderr, "basalt-vision: warmed up at %d image tokens%s in %.1f s\n", sz.w / 32 * (sz.h / 32), ok ? "" : " (failed)",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_t0).count());
    } else {
        std::fprintf(stderr, "basalt-vision: on the CPU, %d threads, no warm-up\n", threads);
    }
    const int n_embd = mp.hp().proj;
    std::printf("READY %d\n", n_embd);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        std::string img_path, out_path;
        if (!parse_enc(line, img_path, out_path)) { std::printf("ERR expected: ENC <image> <output>\n"); std::fflush(stdout); continue; }
        const auto t0 = std::chrono::steady_clock::now();
        std::ifstream f(img_path, std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        Image img;
        if (!f || bytes.empty() || !decode_image(bytes.data(), bytes.size(), img, err)) {
            std::printf("ERR cannot read the image %s\n", img_path.c_str());
            std::fflush(stdout);
            continue;
        }
        const Size sz = fit_size(img.w, img.h, geo);
        std::vector<float> planar, patches, out;
        normalize_planar(to_canvas(img, sz), planar);
        patchify(planar, sz, geo, patches);
        if (!encode(patches, sz, out)) { std::printf("ERR the vision encoder failed\n"); std::fflush(stdout); continue; }
        const int nx = sz.w / 32, ny = sz.h / 32, n = nx * ny;
        FILE* o = std::fopen(out_path.c_str(), "wb");
        if (!o) { std::printf("ERR cannot write %s\n", out_path.c_str()); std::fflush(stdout); continue; }
        const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
        const bool ok = std::fwrite(hdr, sizeof hdr, 1, o) == 1 &&
                        std::fwrite(out.data(), sizeof(float) * (size_t) n_embd, (size_t) n, o) == (size_t) n;
        std::fclose(o);
        if (!ok) { std::printf("ERR short write to %s\n", out_path.c_str()); std::fflush(stdout); continue; }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("OK %d %d %d %.0f\n", n, nx, ny, ms);
        std::fflush(stdout);
    }
    return 0;
}
