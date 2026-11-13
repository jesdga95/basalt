// The resident expert copy's registration around a sleep: dropped (`unpin_complement`), registered again at the same
// addresses (`repin_start` / `repin_wait` / `repin_finish`), twice, with the device alias still mapping the
// same bytes and the bytes unchanged; then a copy out of it into the GPU cache and back.  A synthetic expert file and a
// small GPU expert cache; needs a CUDA device (exits 77 without one).
#include "basalt/core/expert_cache.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/elementwise.hpp"

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

// the alias is a device address (cudaHostGetDevicePointer), not the host pointer: on a driver that does not map registered
// host memory at its own address the two differ, so the check is that the alias reads back the host bytes
void require_alias_maps(const void* alias, const uint8_t* host, size_t bytes, const char* what) {
    require(alias != nullptr, std::string(what) + ": no device alias");
    std::vector<uint8_t> via(bytes);
    require(cudaMemcpy(via.data(), alias, bytes, cudaMemcpyDeviceToHost) == cudaSuccess, std::string(what) + ": copy through the alias");
    require(std::memcmp(via.data(), host, bytes) == 0, std::string(what) + ": the alias maps the host bytes");
}

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
    require(basalt::kernels::cpu::expert_layout_load_text(text.str(), layers, experts, err, 2560, 640), err);

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
    std::vector<const void*> alias;
    bool host_alias = true;   // the driver maps the registered copy at its host address: the sleep needs that
    for (const auto& [l, e] : probe) {
        require(src.has_resident(l, e) && src.pinned(l, e), "a resident expert is pinned after the load");
        require(src.complement_allocation_pinned() || registered(src.blob(l, e)), "ordinary expert registration after load");
        require_alias_maps(src.device_alias(l, e), src.blob(l, e), (size_t) blob, "the device alias after the load");
        bytes.emplace_back(src.blob(l, e), src.blob(l, e) + blob);
        alias.push_back(src.device_alias(l, e));
        host_alias = host_alias && src.device_alias(l, e) == src.blob(l, e);
    }
    const bool allocation_pinned = src.complement_allocation_pinned();
    if (allocation_pinned) {
        for (int cycle = 0; cycle < 2; ++cycle) {
            require(!src.unpin_complement(err) && err.find("host VMM") != std::string::npos,
                    "allocation-backed pins must not be passed to cudaHostUnregister");
            require(!src.complement_unpinned(), "allocation-backed sleep lost pin state");
            src.repin_start(2);
            require(src.repin_finish(err), "allocation-backed wake");
            int32_t* gpu = nullptr;
            require(cudaMalloc(&gpu, 128) == cudaSuccess, "mapped-read buffer");
            for (size_t k = 0; k < probe.size(); ++k) {
                const auto& [l, e] = probe[k];
                require(src.device_alias(l, e) == alias[k] && src.pinned(l, e), "allocation-backed wake changed aliases");
                require(std::memcmp(src.blob(l, e), bytes[k].data(), (size_t) blob) == 0, "wake changed expert weights");
                basalt::kernels::copy_i32_from_mapped(gpu, (const int32_t*) src.device_alias(l, e), 32, nullptr);
                int32_t back[32]{};
                require(cudaMemcpy(back, gpu, sizeof back, cudaMemcpyDeviceToHost) == cudaSuccess, "mapped kernel read");
                require(std::memcmp(back, bytes[k].data(), sizeof back) == 0, "mapped kernel changed expert bytes");
            }
            require(cudaFree(gpu) == cudaSuccess, "mapped-read cleanup");
        }
    } else if (!host_alias) {
        // the sleep is refused where the device alias is not the host address (one registered range read at its host
        // addresses); the copy stays registered and the read path below still runs
        const bool refused = src.unpin_complement(err);
        require(!refused, "the sleep is accepted although the device alias is not the host address");
        std::printf("file_expert_repin_test: this driver does not map the registered copy at its host address, the sleep is "
                    "refused: %s; the repin cycles are skipped\n",
                    err.c_str());
    }
    for (int cycle = 0; !allocation_pinned && host_alias && cycle < 2; ++cycle) {
        const bool unpinned = src.unpin_complement(err);
        require(unpinned, "unpin: " + err);
        require(src.complement_unpinned(), "unpinned");
        for (const auto& [l, e] : probe) require(!registered(src.blob(l, e)), "the copy is no longer registered");
        src.repin_start(2);
        for (const auto& [l, e] : probe)
            require(src.repin_wait(src.blob(l, e), blob) && registered(src.blob(l, e)), "a range waited for is registered");
        const bool repinned = src.repin_finish(err);
        require(repinned, "repin: " + err);
        require(!src.complement_unpinned(), "registered again");
        for (size_t k = 0; k < probe.size(); ++k) {
            const auto& [l, e] = probe[k];
            require(registered(src.blob(l, e)) && src.device_alias(l, e) == alias[k], "the same addresses again");
            require(std::memcmp(src.blob(l, e), bytes[k].data(), (size_t) blob) == 0, "the bytes stay");
            require_alias_maps(src.device_alias(l, e), bytes[k].data(), (size_t) blob, "the device alias after a repin");
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
    if (!allocation_pinned && host_alias) {
        const bool unpinned = src.unpin_complement(err);
        require(unpinned, "unpin at the end: " + err);
    }
    src.close();
    if (!allocation_pinned && !host_alias) {
        std::puts("file_expert_repin_test: skipped: this driver cannot sleep the resident copy, there is no repin to test");
        return 77;
    }
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
