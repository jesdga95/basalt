// The resident expert copy's registration around a sleep: dropped (`unpin_complement`), registered again at the same
// addresses (`repin_start` / `repin_wait` / `repin_finish`), twice, with the device alias still the host address and
// the bytes unchanged; then a copy out of it into the GPU cache and back.  A synthetic expert file and a small GPU
// expert cache; needs a CUDA device (exits 77 without one).
#include "basalt/core/expert_cache.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace basalt::core;

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

struct TempDirectory {
    fs::path path;
    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("basalt-repin-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }
    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

bool registered(const void* p) {
    cudaPointerAttributes a{};
    const cudaError_t e = cudaPointerGetAttributes(&a, p);
    (void) cudaGetLastError();
    return e == cudaSuccess && a.type == cudaMemoryTypeHost;
}

int run() {
    constexpr int64_t layers = 2, experts = 128, held = 20;   // the GPU cache holds experts [0, held) of each layer
    TempDirectory dir;
    basalt::kernels::cpu::NativeFmt fmt;
    std::string err;
    require(basalt::kernels::cpu::native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, 2560, 640, fmt, err), err);
    const uint64_t blob = (uint64_t) fmt.bytes, layer_bytes = blob * experts;
    std::ostringstream text;
    for (int64_t l = 0; l < layers; ++l)
        text << l << ' ' << GGML_TYPE_IQ3_XXS << ' ' << GGML_TYPE_IQ4_NL << ' ' << (uint64_t) l * layer_bytes << ' '
             << blob << '\n';
    {
        std::ofstream out(dir.path / "experts.bin", std::ios::binary);
        std::vector<char> b((size_t) blob);
        for (int64_t i = 0; i < layers * experts; ++i) {
            for (size_t k = 0; k < b.size(); ++k) b[k] = (char) (i * 31 + (int64_t) k * 7 + 1);
            out.write(b.data(), (std::streamsize) b.size());
        }
        require((bool) out, "could not write the synthetic experts");
    }
    require(basalt::kernels::cpu::expert_layout_load_text(text.str(), layers, experts, err), err);

    FileExpertSource src;
    require(src.open(FileRange((dir.path / "experts.bin").string()), layers, experts, err), err);
    ExpertCache cache;
    require(cache.open_sized(std::vector<int64_t>((size_t) (layers * held), (int64_t) blob), layers, experts, err), err);
    for (int64_t l = 0; l < layers; ++l)
        for (int64_t e = 0; e < held; ++e) {
            const int32_t slot = cache.admit(l, e);
            require(slot >= 0 && cache.fill_slot_blocking(slot, src.blob(l, e), err, (int64_t) blob), "cache fill " + err);
        }
    require(src.pin_cache_complement(cache, err, true, {}, -1, 0), err);
    if (!src.complement_pinned()) {
        std::puts("the driver did not register the copy: skipped");
        return 77;
    }
    // the experts the copy holds, and what they read now
    std::vector<std::pair<int64_t, int64_t>> probe = {{0, held}, {0, experts - 1}, {1, held + 7}, {1, experts - 1}};
    std::vector<std::vector<uint8_t>> bytes;
    for (const auto& [l, e] : probe) {
        require(src.has_resident(l, e) && registered(src.blob(l, e)), "a resident expert is registered after the load");
        require(src.device_alias(l, e) == src.blob(l, e), "the device alias is the host address");
        bytes.emplace_back(src.blob(l, e), src.blob(l, e) + blob);
    }
    for (int cycle = 0; cycle < 2; ++cycle) {
        require(src.unpin_complement(err), "unpin: " + err);
        require(src.complement_unpinned(), "unpinned");
        for (const auto& [l, e] : probe) require(!registered(src.blob(l, e)), "the copy is no longer registered");
        src.repin_start(2);
        for (const auto& [l, e] : probe)
            require(src.repin_wait(src.blob(l, e), blob) && registered(src.blob(l, e)), "a range waited for is registered");
        require(src.repin_finish(err), "repin: " + err);
        require(!src.complement_unpinned(), "registered again");
        for (size_t k = 0; k < probe.size(); ++k) {
            const auto& [l, e] = probe[k];
            require(registered(src.blob(l, e)) && src.device_alias(l, e) == src.blob(l, e), "the same addresses again");
            require(std::memcmp(src.blob(l, e), bytes[k].data(), (size_t) blob) == 0, "the bytes stay");
        }
    }
    // a copy out of the registered copy into a slot and back
    const int32_t slot = 0;
    require(cache.fill_slot_blocking(slot, src.blob(probe[0].first, probe[0].second), err, (int64_t) blob), err);
    std::vector<uint8_t> back((size_t) blob);
    require(cudaMemcpy(back.data(), cache.device_slot(slot), (size_t) blob, cudaMemcpyDeviceToHost) == cudaSuccess,
            "copy back");
    require(back == bytes[0], "the slot holds the expert's bytes");
    // a sleep left in place at the end: closing frees an unregistered copy
    require(src.unpin_complement(err), "unpin at the end: " + err);
    src.close();
    return 0;
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    try {
        const int rc = run();
        if (rc == 0) std::puts("file_expert_repin_test: PASS");
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "file_expert_repin_test: %s\n", e.what());
        return 1;
    }
}
