// RAM homes of the resident expert copy (--vram-elastic): taking, releasing, reclaiming and compacting them after an
// adaptive swap has happened, the refusals while an exchange is in flight, and the RAM reserve.  A synthetic expert
// file and a small GPU expert cache; needs a CUDA device (exits 77 without one).
#include "basalt/core/expert_cache.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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
        path = fs::temp_directory_path() / ("basalt-homes-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }
    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

int run() {
    constexpr int64_t layers = 2, experts = 128, held = 50;   // the GPU cache holds experts [0, held) of each layer
    TempDirectory dir;
    basalt::kernels::cpu::NativeFmt fmt;
    std::string err;
    require(basalt::kernels::cpu::native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, 2560, 640, fmt, err), err);
    const uint64_t blob = (uint64_t) fmt.bytes, layer_bytes = blob * experts, total = layer_bytes * layers;
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
    require(src.pin_cache_complement(cache, err, false, {}, -1, 0), err);
    require(src.reserve_exchanges(4, err), err);

    // an adaptive swap lands: from here the exchange table stays allocated (the case a plain emptiness guard misses)
    require(src.has_resident(0, 60) && !src.has_resident(0, 0), "start state");
    std::vector<uint8_t> out_blob(src.blob(0, 0), src.blob(0, 0) + blob);
    std::copy(out_blob.begin(), out_blob.end(), src.exchange_buffer(0));
    require(src.stage_exchange(0, 60, 0, 0), "stage_exchange");
    require(src.exchanges_staged() && src.commit_exchanges() == 1, "commit_exchanges");
    require(src.has_resident(0, 0) && !src.has_resident(0, 60), "the swap moved the place");

    // homes for the cache's experts the copy does not hold (a shrink giving them back)
    std::vector<std::pair<int32_t, int32_t>> give;
    for (int32_t l = 0; l < layers; ++l)
        for (int32_t e = 0; e < held; ++e)
            if (!src.has_resident(l, e)) give.emplace_back(l, e);
    require(give.size() == (size_t) (layers * held - 1), "give list");
    std::vector<uint8_t*> dst;
    require(src.homes_take(give, err, dst), "homes_take: " + err);
    for (size_t k = 0; k < give.size(); ++k) std::copy(src.blob(give[k].first, give[k].second),
                                                      src.blob(give[k].first, give[k].second) + blob, dst[k]);
    // (blob() reads the file for these: no copy exists yet)
    src.homes_commit();
    require(src.has_resident(1, 3) && src.equals_file(1, 3, src.blob(1, 3)), "a homed expert reads its file bytes");
    uint64_t homed = src.homes_bytes();
    require(homed >= (uint64_t) give.size() * blob && homed > (192ull << 20), "several regions were taken");

    // every expert but each 20th goes back into a slot: the regions with a survivor stay, an empty one is returned
    std::vector<std::pair<int32_t, int32_t>> keep;
    for (size_t k = 0; k < give.size(); ++k) {
        if (k % 20 == 0) keep.push_back(give[k]);
        else require(src.home_release(give[k].first, give[k].second), "home_release");
    }
    const uint64_t returned = src.homes_reclaim();
    require(returned > 0 && src.homes_bytes() + returned == homed, "a region with no expert was not returned");
    homed -= returned;
    require(src.homes_bytes() > (96ull << 20), "regions with a survivor were freed");

    // an exchange in flight: no new take, no compaction
    std::copy(out_blob.begin(), out_blob.end(), src.exchange_buffer(1));
    require(src.stage_exchange(0, 61, 60, 1), "second stage_exchange");
    std::vector<uint8_t*> none;
    require(!src.homes_take({{0, 62}}, err, none) && err.find("in flight") != std::string::npos, "take with a swap staged");
    require(src.homes_compact() == 0 && src.homes_bytes() == homed, "compaction with a swap staged");
    require(src.commit_exchanges() == 1, "second commit_exchanges");

    // the exchange table is allocated and stays so: compaction still runs and frees the sparse regions
    const uint64_t freed = src.homes_compact();
    require(freed == homed && src.homes_bytes() == keep.size() * blob, "compaction after an adaptive swap kept more than the survivors");
    for (const auto& [l, e] : keep)
        require(src.has_resident(l, e) && src.equals_file(l, e, src.blob(l, e)), "a survivor lost its bytes");
    for (size_t k = 0; k < give.size(); ++k)
        if (k % 20 != 0) require(!src.has_resident(give[k].first, give[k].second), "a released expert still has a place");

    // the RAM reserve counts against the budget
    src.set_homes_reserve(~0ull >> 1);
    const uint64_t kept_bytes = src.homes_bytes();
    require(!src.homes_take({{0, 2}}, err, none) && !err.empty() && src.homes_bytes() == kept_bytes,
            "a take that leaves less than the reserve free was accepted");
    src.set_homes_reserve(0);
    const uint64_t before = src.homes_bytes();
    require(src.homes_take({{0, 2}}, err, none), "take after the reserve is lifted: " + err);
    src.homes_abort();
    require(src.homes_bytes() == before && !src.has_resident(0, 2), "an aborted take left a region or a place");
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
        if (rc == 0) std::puts("file_expert_homes_test: PASS");
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "file_expert_homes_test: %s\n", e.what());
        return 1;
    }
}
