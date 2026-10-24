#include "basalt/core/expert_source.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"

#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/platform/system.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("basalt-file-source-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void create_experts(const fs::path& dir, uint64_t bytes, const std::vector<std::pair<uint64_t, char>>& markers = {}) {
    const fs::path file = dir / "experts.bin";
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        require((bool) out, "could not create synthetic experts file");
        if (bytes > 0) {
            out.seekp((std::streamoff) (bytes - 1));
            out.put('\0');
        }
        require((bool) out, "could not size synthetic experts file");
    }
    for (const auto& [offset, marker] : markers) {
        require(offset < bytes, "synthetic marker lies beyond experts.bin");
        std::fstream file_out(file, std::ios::binary | std::ios::in | std::ios::out);
        require((bool) file_out, "could not open synthetic experts file for markers");
        file_out.seekp((std::streamoff) offset);
        file_out.put(marker);
        require((bool) file_out, "could not write synthetic expert marker");
    }
}

void check_file_size_rejection(basalt::core::FileExpertSource& source, const fs::path& dir, int64_t layers,
                               int64_t experts, uint64_t expected_bytes) {
    std::string err;
    source.close();
    fs::resize_file(dir / "experts.bin", expected_bytes - 1);
    require(!source.open(basalt::core::FileRange((dir / "experts.bin").string()), layers, experts, err),
            "a truncated expert file was accepted");
    require(!source.mapped() && !err.empty(), "truncated-file rejection left the source mapped or unreported");

    fs::resize_file(dir / "experts.bin", expected_bytes + 1);
    err.clear();
    require(!source.open(basalt::core::FileRange((dir / "experts.bin").string()), layers, experts, err),
            "an oversized expert file was accepted");
    require(!source.mapped() && !err.empty(), "oversized-file rejection left the source mapped or unreported");
}

void test_native_variable_layout() {
    using namespace basalt::core;
    using namespace basalt::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    constexpr int64_t embedding = 2560;
    constexpr int64_t feed_forward = 640;
    TempDirectory dir;

    NativeFmt first_fmt, second_fmt;
    std::string err;
    const bool first_ok = native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, embedding, feed_forward, first_fmt, err);
    require(first_ok, "IQ3_XXS/IQ4_NL synthetic format is unavailable: " + err);
    err.clear();
    const bool second_ok = native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, embedding, feed_forward, second_fmt, err);
    require(second_ok, "IQ2_XS/IQ4_NL synthetic format is unavailable: " + err);
    require(first_fmt.bytes != second_fmt.bytes, "chosen native formats do not exercise variable layer sizes");

    const uint64_t layer0_bytes = (uint64_t) first_fmt.bytes * experts;
    const uint64_t total = layer0_bytes + (uint64_t) second_fmt.bytes * experts;
    std::ostringstream layout_text;
    layout_text << "0 " << GGML_TYPE_IQ3_XXS << ' ' << GGML_TYPE_IQ4_NL << " 0 " << first_fmt.bytes << '\n';
    layout_text << "1 " << GGML_TYPE_IQ2_XS << ' ' << GGML_TYPE_IQ4_NL << ' ' << layer0_bytes << ' '
                << second_fmt.bytes << '\n';
    create_experts(dir.path, total, {{0, 'a'}, {(uint64_t) first_fmt.bytes, 'b'},
                                  {layer0_bytes, 'c'}, {layer0_bytes + (uint64_t) second_fmt.bytes, 'd'}});

    const bool layout_ok = expert_layout_load_text(layout_text.str(), layers, experts, err);
    require(layout_ok, "could not load synthetic native layout: " + err);
    FileExpertSource source;
    bool opened = source.open(FileRange((dir.path / "experts.bin").string()), layers, experts, err);
    require(opened, "could not map the native experts: " + err);
    require(source.blobs() == layers * experts, "native blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    const uint8_t* next_layer_second = source.blob(1, 1);
    require(first && second && next_layer && next_layer_second, "valid native blob lookup failed");
    require(second - first == (ptrdiff_t) first_fmt.bytes && next_layer - first == (ptrdiff_t) layer0_bytes &&
                next_layer_second - next_layer == (ptrdiff_t) second_fmt.bytes,
            "native per-layer expert stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c' && next_layer_second[0] == 'd',
            "native blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "native bounds check accepted an invalid layer or expert");
    require(source.reads() == 4, "invalid native lookups changed the read count");

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_experts(dir.path, total, {{layer0_bytes, 'c'}});
    err.clear();
    opened = source.open(FileRange((dir.path / "experts.bin").string()), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened native source kept stale state");
    source.close();
}

void test_complement_plan() {
    using namespace basalt::core::detail;
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    require(bytes == 16 && offsets == std::vector<uint64_t>{0, kNoCacheComplement, 3, 6, 11, kNoCacheComplement},
            "wrong compact offsets for variable native layer sizes");
    const uint8_t resident[16] = {}, mapped[5] = {};
    require(cache_complement_blob_or_fallback(1, offsets, resident, mapped) == mapped,
            "GPU-resident expert lost mmap fallback");
    require(cache_complement_blob_or_fallback(4, offsets, resident, mapped) == resident + 11,
            "CPU miss did not use resident complement");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {0, 1}}, {}, offsets, bytes, error)
            && offsets.empty() && bytes == 0, "duplicate pair accepted");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}}, {{0, 1}}, offsets, bytes, error),
            "overlapping tiers accepted");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{2, 0}}, {}, offsets, bytes, error),
            "out-of-range pair accepted");
    require(!make_cache_complement_plan(2, 3, {0, 5}, {}, {}, offsets, bytes, error), "zero-size layer accepted");
}

void test_resident_lend_region() {
    using namespace basalt::core::detail;
    // four slots (sizes 5, 3, 3, 5); the experts no slot holds take 10 bytes
    const std::vector<uint64_t> slots{5, 3, 3, 5};
    require(choose_resident_keep_from(slots, 10, 9, 1) == -1, "a base larger than the budget was accepted");
    require(choose_resident_keep_from(slots, 10, 10, 1) == 4, "no room: the lend region must stay on the file");
    require(choose_resident_keep_from(slots, 10, 15, 1) == 3, "the last slot fits and must be kept first");
    require(choose_resident_keep_from(slots, 10, 17, 1) == 3, "a slot that does not fit ends the lend region's walk");
    require(choose_resident_keep_from(slots, 10, 18, 1) == 2, "two slots fit");
    require(choose_resident_keep_from(slots, 10, 100, 1) == 1, "the walk must stop at the lend region's first slot");
    require(choose_resident_keep_from(slots, 10, 100, -1) == 4, "no lend region kept slots in RAM");
    require(choose_resident_keep_from(slots, 10, 100, 9) == 4, "an out-of-range lend region kept slots in RAM");
    bool fit = false;
    require(choose_resident_room(slots, 10, 19, 1, 4, fit) == 3 && fit,
            "the swaps' free places must be placed before the lent slots");
    require(choose_resident_room(slots, 10, 15, 1, 6, fit) == 3 && !fit,
            "free places that do not fit must leave the whole budget to the lent slots");
    require(choose_resident_room(slots, 10, 100, 1, 0, fit) == 1 && !fit, "no free places asked for");
    require(choose_resident_keep_from({}, 0, 0, 0) == 0, "an empty cache");
}

void test_resident_room() {
    using namespace basalt::core::detail;
    constexpr uint64_t kGiB = 1ull << 30;
    constexpr uint64_t kBlob = 17407795;   // one expert, bytes
    // The gate and the keep-what-fits budget share resident_room(); the budget packs whole experts into it, so what it
    // keeps always passes the gate, and the copy loses the shortfall plus less than one expert.
    auto kept = [&](uint64_t physical, uint64_t headroom, uint64_t complement) {
        const uint64_t room = resident_room(physical, headroom);
        const uint64_t budget = std::min(room, complement);
        return budget / kBlob * kBlob;
    };
    const uint64_t physical = 78 * kGiB + kGiB / 6, headroom = 15 * kGiB;
    const uint64_t room = resident_room(physical, headroom);
    require(room == physical - headroom - physical / 256, "the room is the available RAM less the headroom and overhead");
    require(resident_room(10 * kGiB, 15 * kGiB) == 0, "a headroom above the RAM leaves no room");
    // the whole complement fits
    const uint64_t whole = 60 * kGiB;
    require(whole <= room && kept(physical, headroom, whole) > whole - kBlob, "a complement that fits was cut");
    // a small shortfall (about 1.8 GiB): the hottest experts that fit are kept, and the gate accepts them
    const uint64_t small = 64 * kGiB + 700 * (1ull << 20);
    uint64_t k = kept(physical, headroom, small);
    require(small > room && k > 0 && k <= room, "a small shortfall must keep what fits");
    require(small - k < (small - room) + kBlob, "a small shortfall lost more than the shortfall");
    // a big shortfall keeps the room, not nothing
    const uint64_t big = 120 * kGiB;
    k = kept(physical, headroom, big);
    require(k <= room && room - k < kBlob, "a big shortfall must keep the whole room");
    // across readings: the kept bytes never exceed the gate's room
    for (uint64_t gb = 20; gb <= 120; ++gb)
        for (uint64_t h = 4; h <= 16; h += 3)
            require(kept(gb * kGiB + 12345, h * kGiB, 200 * kGiB) <= resident_room(gb * kGiB + 12345, h * kGiB),
                    "kept bytes failed the gate");
}

void test_resident_exchange() {
    using namespace basalt::core;
    using namespace basalt::core::detail;
    // 2 layers x 3 experts; the GPU holds (0,1) and (1,2): the compact copy holds the other four
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    const std::vector<uint64_t> before = offsets;
    // (1,0) moves into the GPU, (1,2) leaves it: (1,2) takes (1,0)'s bytes' place
    require(exchange_cache_complement(offsets, 3, 5), "a valid exchange was refused");
    require(offsets[5] == before[3] && offsets[3] == kNoCacheComplement, "the exchange did not move the place");
    for (size_t i : {0u, 1u, 2u, 4u}) require(offsets[i] == before[i], "an exchange touched another expert");
    // the copy's size and its set of places are unchanged
    std::vector<uint64_t> a, b;
    for (uint64_t o : before) if (o != kNoCacheComplement) a.push_back(o);
    for (uint64_t o : offsets) if (o != kNoCacheComplement) b.push_back(o);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    require(a == b, "an exchange changed the compact copy's places");
    const std::vector<uint64_t> after = offsets;
    require(!exchange_cache_complement(offsets, 3, 0), "an `in` the copy does not hold was accepted");
    require(!exchange_cache_complement(offsets, 0, 2), "an `out` the copy holds already was accepted");
    require(!exchange_cache_complement(offsets, 0, 0), "a self-exchange was accepted");
    require(!exchange_cache_complement(offsets, 0, 6), "an out-of-range `out` was accepted");
    require(offsets == after, "a refused exchange changed the offsets");
    // and back: the copy is again what the plan makes for the original placement
    require(exchange_cache_complement(offsets, 5, 3), "the reverse exchange was refused");
    require(offsets == before, "exchanging back did not restore the plan");
}

void test_home_regions() {
    using namespace basalt::core::detail;
    // two regions after a 1000-byte complement: [1000, 1300) and [1300, 1400)
    std::vector<HomeRegion> regions(2);
    regions[0].base = 1000; regions[0].bytes = 300;
    regions[1].base = 1300; regions[1].bytes = 100;
    require(home_region_find(regions, 1000, 100) == 0 && home_region_find(regions, 1200, 100) == 0, "a place in region 0");
    require(home_region_find(regions, 1300, 100) == 1, "a place in region 1");
    require(home_region_find(regions, 1250, 100) == -1, "a place straddling two regions was found");
    require(home_region_find(regions, 999, 10) == -1 && home_region_find(regions, 1400, 1) == -1, "a place outside every region");
    // a region is used while any expert's place lies in it
    std::vector<uint64_t> offsets = {0, kNoCacheComplement, 1100, kNoCacheComplement};
    require(home_region_used(regions[0], offsets) && !home_region_used(regions[1], offsets), "region use");
    offsets[2] = kNoCacheComplement;   // the expert went back into a slot: its place is released
    require(!home_region_used(regions[0], offsets), "a region with no expert is still used");
    offsets[3] = 1399;
    require(home_region_used(regions[1], offsets), "the last byte of a region");
    // a release that leaves one expert in a region keeps the region; only the last release frees it
    offsets = {1000, 1100, 1200, 1300};
    offsets[0] = offsets[1] = kNoCacheComplement;
    require(home_region_used(regions[0], offsets), "a region with one expert left was called unused");
    offsets[2] = kNoCacheComplement;
    require(!home_region_used(regions[0], offsets) && home_region_used(regions[1], offsets), "release of the last expert");
}

void test_cgroup_memory_budget() {
    using namespace basalt::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    uint64_t bytes = 0;

    CgroupMemoryStat clean_cache{40 * GiB, 32 * GiB, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, clean_cache, bytes), "valid cgroup memory.stat was rejected");
    require(bytes == 48 * GiB - 56 * GiB / 64, "clean inactive file cache was not credited against the cgroup cap");

    CgroupMemoryStat dirty_cache{40 * GiB, 32 * GiB, 4 * GiB, 2 * GiB, true};
    require(cgroup_available_bytes(56 * GiB, dirty_cache, bytes), "valid dirty-cache counters were rejected");
    require(bytes == 42 * GiB - 56 * GiB / 64, "dirty/writeback pages were incorrectly counted as reclaimable");

    CgroupMemoryStat oversized_inactive{10 * GiB, 20 * GiB, 0, 0, true};
    require(cgroup_available_bytes(12 * GiB, oversized_inactive, bytes),
            "valid oversized inactive-file counter was rejected");
    require(bytes == 12 * GiB - 12 * GiB / 64, "inactive-file accounting exceeded charged current usage");

    const uint64_t max = std::numeric_limits<uint64_t>::max();
    CgroupMemoryStat large_counters{max, max, max - 1, max, true};
    require(cgroup_available_bytes(max, large_counters, bytes),
            "large memory.stat counters were rejected");
    require(bytes == 0, "dirty/writeback subtraction overflowed or escaped the usage cap");

    CgroupMemoryStat usage_over_limit{60 * GiB, 0, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, usage_over_limit, bytes),
            "valid over-limit memory counters were rejected");
    require(bytes == 0, "over-limit cgroup usage produced a positive budget");

    CgroupMemoryStat missing_stat{};
    bytes = 123;
    require(!cgroup_available_bytes(56 * GiB, missing_stat, bytes) && bytes == 0,
            "missing memory.stat counters did not fail closed");
}

// memory.stat text -> available bytes, with no /sys read.  A container's own reads sit on the active list, so the
// file total (active + inactive) is what counts as reclaimable.
void test_cgroup_memory_stat_text() {
    using namespace basalt::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    auto gib = [](uint64_t n) { return std::to_string(n * GiB); };
    CgroupMemoryStat stat;
    uint64_t bytes = 0;
    const uint64_t limit = 64 * GiB, current = 48 * GiB;

    // file present: mostly active pages (inactive alone would credit 2 GiB)
    require(parse_cgroup_memory_stat("anon 1\nfile " + gib(40) + "\ninactive_file " + gib(2) +
                                         "\nactive_file " + gib(38) + "\nfile_dirty 0\nfile_writeback 0\n",
                                     current, stat) && stat.file == 40 * GiB, "memory.stat with file was rejected");
    require(cgroup_available_bytes(limit, stat, bytes), "file present: available refused");
    require(bytes == (limit - (current - 40 * GiB)) - limit / 64, "active file pages were not counted reclaimable");

    // file missing: the inactive list alone
    require(parse_cgroup_memory_stat("inactive_file " + gib(2) + "\nfile_dirty 0\nfile_writeback 0\n", current, stat) &&
                stat.file == 0, "memory.stat without file was rejected");
    require(cgroup_available_bytes(limit, stat, bytes), "file missing: available refused");
    require(bytes == 17 * GiB, "file missing: wrong fallback to inactive_file");

    // dirty and writeback pages are not reclaimable
    require(parse_cgroup_memory_stat("file " + gib(40) + "\ninactive_file " + gib(10) + "\nfile_dirty " + gib(6) +
                                         "\nfile_writeback " + gib(2) + "\n", current, stat), "dirty text rejected");
    require(cgroup_available_bytes(limit, stat, bytes), "dirty: available refused");
    require(bytes == (limit - (current - 32 * GiB)) - limit / 64, "dirty/writeback pages counted reclaimable");

    // the margin never exceeds half of what is reclaimable
    require(parse_cgroup_memory_stat("file 640\ninactive_file 0\nfile_dirty 0\nfile_writeback 0\n", 1000, stat),
            "small text rejected");
    require(cgroup_available_bytes(64000, stat, bytes) && bytes == 64000 - 680, "margin exceeded half the reclaimable");

    // malformed text fails closed
    require(!parse_cgroup_memory_stat("inactive_file 1\nfile_dirty 0\n", current, stat), "missing writeback accepted");
    require(!parse_cgroup_memory_stat("file 1\nfile 2\ninactive_file 1\nfile_dirty 0\nfile_writeback 0\n", current, stat),
            "duplicate file accepted");
    require(!parse_cgroup_memory_stat("inactive_file x\n", current, stat), "non-numeric value accepted");
}

// The host RAM probe with fake /proc and cgroup trees: v2 (a limit, "max", a missing or malformed limit), v1
// (a limit, unlimited), and no cgroup line at all.
void test_host_memory() {
    using namespace basalt::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    HostMemory m;
    TempDirectory t;
    auto put = [&](const fs::path& rel, const std::string& text) {
        fs::create_directories((t.path / rel).parent_path());
        std::ofstream(t.path / rel) << text;
    };
    put("meminfo", "MemTotal: 134217728 kB\nMemAvailable: 104857600 kB\n");   // 100 GiB available
    const std::string mi = (t.path / "meminfo").string(), cg = (t.path / "cgroup").string(),
                      root = (t.path / "fs").string();
    auto probe = [&](const std::string& self) {
        put("cgroup", self);
        return host_available_memory(m, mi, cg, root);
    };
    // no cgroup line at all: MemAvailable alone
    require(probe("") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "no cgroup: MemAvailable alone");
    // v2, a 48 GiB limit with 16 GiB charged, 4 GiB of it clean cache
    put("fs/cgroup.controllers", "memory\n");
    put("fs/box/memory.max", std::to_string(48 * GiB) + "\n");
    put("fs/box/memory.current", std::to_string(16 * GiB) + "\n");
    put("fs/box/memory.stat", "inactive_file " + std::to_string(4 * GiB) + "\nfile_dirty 0\nfile_writeback 0\n");
    require(probe("0::/box\n") && m.available == 36 * GiB - 48 * GiB / 64 && m.cgroup_limit == 48 * GiB, "v2 limit");
    put("fs/box/memory.max", "max\n");
    require(probe("0::/box\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v2 max");
    put("fs/box/memory.max", "48G\n");
    require(!probe("0::/box\n"), "v2 malformed limit accepted");
    fs::remove(t.path / "fs/box/memory.max");
    require(probe("0::/box\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull,
            "v2 group without memory.max (no memory controller there): MemAvailable alone");
    // v1: the memory controller's group (a hybrid line list), a 32 GiB limit with 10 GiB used
    put("fs/memory/docker/abc/memory.limit_in_bytes", std::to_string(32 * GiB) + "\n");
    put("fs/memory/docker/abc/memory.usage_in_bytes", std::to_string(10 * GiB) + "\n");
    put("fs/memory/memory.limit_in_bytes", "9223372036854771712\n");   // the root: unlimited
    put("fs/memory/memory.usage_in_bytes", std::to_string(50 * GiB) + "\n");
    require(probe("12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n1:name=systemd:/docker/abc\n") &&
            m.available == 22 * GiB && m.cgroup_limit == 32 * GiB, "v1 limit");
    put("fs/memory/docker/abc/memory.limit_in_bytes", "9223372036854771712\n");
    require(probe("4:memory:/docker/abc\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v1 unlimited");
    // v1 memory.stat: the clean inactive file cache counts as free, dirty and writeback pages do not
    put("fs/memory/docker/abc/memory.limit_in_bytes", std::to_string(32 * GiB) + "\n");
    put("fs/memory/docker/abc/memory.stat", "cache 1\ntotal_inactive_file " + std::to_string(6 * GiB) +
                                            "\ntotal_dirty " + std::to_string(1 * GiB) + "\ntotal_writeback " +
                                            std::to_string(1 * GiB) + "\n");
    require(probe("4:memory:/docker/abc\n") && m.available == 26 * GiB - 32 * GiB / 64 && m.cgroup_limit == 32 * GiB,
            "v1 page cache counted as used");
    // a v1 group not visible here, and no limit at the mount's root: MemAvailable alone
    require(probe("4:memory:/elsewhere\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull,
            "v1 group not mounted");
    // no /proc/self/cgroup at all: MemAvailable alone
    require(host_available_memory(m, mi, (t.path / "missing").string(), root) && m.available == 100 * GiB &&
                m.cgroup_limit == ~0ull, "no cgroup file");
    // no MemAvailable: the RAM cannot be determined
    put("meminfo-old", "MemTotal: 134217728 kB\nMemFree: 1024 kB\n");
    require(!host_available_memory(m, (t.path / "meminfo-old").string(), cg, root), "no MemAvailable accepted");
    // the system's own files (empty paths): some RAM, and no container limit where the OS keeps no accounting file
    require(host_available_memory(m) && m.available > 0, "this machine's RAM");
    if (basalt::platform::meminfo_path().empty()) require(m.cgroup_limit == ~0ull, "no cgroup limit without cgroup files");
}

// A container whose own cgroup namespace shows its group as the mount's root, while /proc/self/cgroup names the
// group by its host path (not visible here): the limit is read at the root.  The hybrid layout (v1 controllers plus a
// unified line) reads the v1 memory controller.
void test_host_memory_namespaced() {
    using namespace basalt::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    const std::string id = "/docker/4f3c2a";
    struct Tree {
        TempDirectory t;
        void put(const fs::path& rel, const std::string& text) {
            fs::create_directories((t.path / rel).parent_path());
            std::ofstream(t.path / rel) << text;
        }
        bool probe(HostMemory& m, const std::string& self) {
            put("meminfo", "MemTotal: 134217728 kB\nMemAvailable: 104857600 kB\n");   // 100 GiB available
            put("cgroup", self);
            return host_available_memory(m, (t.path / "meminfo").string(), (t.path / "cgroup").string(),
                                         (t.path / "fs").string());
        }
    };
    HostMemory m;
    {   // hybrid (the rented container's layout): v1 controllers, memory among them, and a unified line
        Tree h;
        h.put("fs/memory/memory.limit_in_bytes", std::to_string(64 * GiB) + "\n");
        h.put("fs/memory/memory.usage_in_bytes", std::to_string(20 * GiB) + "\n");
        h.put("fs/memory/memory.stat", "total_inactive_file " + std::to_string(8 * GiB) +
                                       "\ntotal_dirty 0\ntotal_writeback 0\n");
        h.put("fs/unified/cgroup.controllers", "\n");
        const std::string self = "12:pids:" + id + "\n11:cpu,cpuacct:" + id + "\n4:memory:" + id +
                                 "\n1:name=systemd:" + id + "\n0::" + id + "\n";
        require(h.probe(m, self) && m.available == 52 * GiB - 64 * GiB / 64 && m.cgroup_limit == 64 * GiB, "hybrid layout");
        // the same without a limit on the container: MemAvailable alone
        h.put("fs/memory/memory.limit_in_bytes", "9223372036854771712\n");
        require(h.probe(m, self) && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "hybrid, unlimited");
        // and without the v1 mount at all
        fs::remove_all(h.t.path / "fs/memory");
        require(h.probe(m, self) && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "hybrid, no v1 mount");
    }
    {   // pure v1, the group's path hidden: read at the memory mount's root
        Tree v1;
        v1.put("fs/memory/memory.limit_in_bytes", std::to_string(16 * GiB) + "\n");
        v1.put("fs/memory/memory.usage_in_bytes", std::to_string(4 * GiB) + "\n");
        require(v1.probe(m, "4:memory:" + id + "\n3:cpuset:" + id + "\n") && m.available == 12 * GiB &&
                    m.cgroup_limit == 16 * GiB, "v1, hidden path");
    }
    {   // pure v2, the group's path hidden: read at the mount's root
        Tree v2;
        v2.put("fs/cgroup.controllers", "cpu memory pids\n");
        v2.put("fs/memory.max", std::to_string(48 * GiB) + "\n");
        v2.put("fs/memory.current", std::to_string(16 * GiB) + "\n");
        v2.put("fs/memory.stat", "inactive_file " + std::to_string(4 * GiB) + "\nfile_dirty 0\nfile_writeback 0\n");
        require(v2.probe(m, "0::" + id + "\n") && m.available == 36 * GiB - 48 * GiB / 64 && m.cgroup_limit == 48 * GiB,
                "v2, hidden path");
        // the host's own root (no memory.max): MemAvailable alone
        fs::remove(v2.t.path / "fs/memory.max");
        require(v2.probe(m, "0::/\n") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "v2, root group");
    }
    {   // no cgroup line at all
        Tree none;
        require(none.probe(m, "") && m.available == 100 * GiB && m.cgroup_limit == ~0ull, "no cgroup line");
    }
}

}  // namespace

int main() {
    try {
        test_home_regions();
        test_complement_plan();
        test_resident_lend_region();
        test_resident_exchange();
        test_resident_room();
        test_cgroup_memory_budget();
        test_cgroup_memory_stat_text();
        test_host_memory();
        test_host_memory_namespaced();
        test_native_variable_layout();
        std::cout << "file_expert_source_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "file_expert_source_test: " << error.what() << '\n';
        return 1;
    }
}
