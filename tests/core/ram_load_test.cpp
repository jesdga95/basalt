// The shared RAM loader: both layouts place every blob where the plan says, on any alignment, with or without the OS cache.
#include "basalt/core/ram_load.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>
#include <utility>

using namespace basalt::core;

namespace {

int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}

struct File {
    std::string path;
    std::vector<uint8_t> bytes;
    int fd = -1;
    explicit File(uint64_t n) : path((std::filesystem::temp_directory_path() / "ram_load_test.bin").string()), bytes(n) {
        std::mt19937_64 rng(7);
        for (uint8_t& b : bytes) b = (uint8_t) rng();
        std::FILE* f = basalt::platform::open_file_utf8(path, "wb");
        std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
        fd = basalt::platform::open_fd_utf8(path);
    }
    ~File() {
        basalt::platform::close_fd(fd);
        std::filesystem::remove(path);
    }
};

// Layers of different blob sizes, aligned to the block size or not.
RamGrid make_grid(uint64_t base, uint64_t blob) {
    RamGrid g;
    g.n_layers = 6;
    g.n_expert = 16;
    g.file_base = base;
    uint64_t at = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        g.layer_offset.push_back(at);
        g.blob_bytes.push_back(blob + (uint64_t) l * 4096);
        at += g.blob_bytes.back() * (uint64_t) g.n_expert;
    }
    return g;
}

uint64_t grid_end(const RamGrid& g) {
    return g.file_pos(g.n_layers - 1, g.n_expert - 1) + g.blob_bytes.back();
}

void arena_case(uint64_t base, uint64_t blob, bool unbuffered, int threads) {
    const RamGrid g = make_grid(base, blob);
    File f(grid_end(g) + 4096);
    constexpr uint64_t kNone = ~0ull;
    std::mt19937_64 rng(3);
    std::vector<uint64_t> offsets((size_t) (g.n_layers * g.n_expert), kNone);
    std::vector<uint8_t> no_bar(offsets.size(), 0);
    uint64_t at = 0, bar = 0, placed = 0;
    for (int64_t l = 0; l < g.n_layers; ++l)
        for (int64_t e = 0; e < g.n_expert; ++e) {
            const size_t i = (size_t) (l * g.n_expert + e);
            if (rng() % 4 == 0) {
                if (rng() % 2) at += 4096;   // a hole breaks the run
                continue;
            }
            offsets[i] = at;
            at += g.blob_bytes[(size_t) l];
            placed += g.blob_bytes[(size_t) l];
            no_bar[i] = rng() % 5 == 0;
            if (!no_bar[i]) bar += g.blob_bytes[(size_t) l];
        }
    uint8_t* mem = alloc_ram(at);
    std::memset(mem, 0xee, (size_t) at);
    RamLoad L;
    L.fd = f.fd;
    L.unbuffered = unbuffered;
    L.threads = threads;
    L.groups = ram_arena_runs(g, mem, offsets, kNone, no_bar);
    uint64_t run_bar = 0;
    std::vector<uint8_t> blob_seen(offsets.size(), 0);
    for (size_t l = 0; l < L.groups.size(); ++l)
        for (const RamRun& r : L.groups[l]) {
            run_bar += r.progress;
            check(r.bytes == (uint64_t) r.count * r.blob_bytes, "a run's size is its blobs");
            for (int64_t k = 0; k < r.count; ++k) blob_seen[l * (size_t) g.n_expert + (size_t) (r.first + k)] = 1;
        }
    check(run_bar == bar, "the runs carry the bar's total");
    for (size_t i = 0; i < offsets.size(); ++i) check((offsets[i] != kNone) == (blob_seen[i] != 0), "each placed blob in one run");
    RamLoadStats st;
    std::string err;
    check(read_ram_runs(L, st, err), err.c_str());
    check(st.copied == placed, "every placed byte counted");
    for (int64_t l = 0; l < g.n_layers; ++l)
        for (int64_t e = 0; e < g.n_expert; ++e) {
            const uint64_t off = offsets[(size_t) (l * g.n_expert + e)];
            if (off == kNone) continue;
            check(std::memcmp(mem + off, f.bytes.data() + g.file_pos(l, e), (size_t) g.blob_bytes[(size_t) l]) == 0,
                  "arena blob matches the file");
        }
    basalt::platform::free_aligned(mem);
}

void span_case(uint64_t base, uint64_t blob) {
    const RamGrid g = make_grid(base, blob);
    File f(grid_end(g) + 4096);
    std::vector<uint8_t*> spans;
    std::vector<int32_t> slots((size_t) (g.n_layers * g.n_expert), -1), sunk(slots.size(), -1);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        spans.push_back(alloc_ram(g.blob_bytes[(size_t) l] * (uint64_t) g.n_expert));
        int32_t next = 0;
        for (int64_t e = 0; e < g.n_expert; ++e) {
            const size_t i = (size_t) (l * g.n_expert + e);
            if (e % 3 == 0) sunk[i] = 1;
            else if (e % 3 == 1) slots[i] = next++;
        }
    }
    RamLoad L;
    L.fd = f.fd;
    L.threads = 3;
    L.groups = ram_layer_runs(g, spans, slots, sunk);
    std::vector<uint8_t> got(slots.size(), 0);
    L.sink = [&](int64_t l, const RamRun& r, const uint8_t* bytes, std::string& e) {
        if (std::memcmp(bytes, f.bytes.data() + g.file_pos(l, r.first), (size_t) r.bytes) != 0) {
            e = "sink bytes differ";
            return false;
        }
        got[(size_t) (l * g.n_expert + r.first)] = 1;
        return true;
    };
    RamLoadStats st;
    std::string err;
    check(read_ram_runs(L, st, err), err.c_str());
    for (int64_t l = 0; l < g.n_layers; ++l)
        for (int64_t e = 0; e < g.n_expert; ++e) {
            const size_t i = (size_t) (l * g.n_expert + e);
            if (sunk[i] >= 0) check(got[i] != 0, "sink took its blob");
            if (slots[i] >= 0)
                check(std::memcmp(spans[(size_t) l] + (uint64_t) slots[i] * g.blob_bytes[(size_t) l],
                                  f.bytes.data() + g.file_pos(l, e), (size_t) g.blob_bytes[(size_t) l]) == 0,
                      "span blob matches the file");
        }
    for (uint8_t* p : spans) basalt::platform::free_aligned(p);
}

void fallback_case() {
    const RamGrid g = make_grid(4096, 4096);
    File f(grid_end(g));
    std::vector<uint8_t> mem((size_t) g.blob_bytes[0]);
    RamLoad L;
    L.fd = f.fd;
    L.unbuffered = false;
    L.groups.resize(1);
    add_ram_blob(L.groups[0], mem.data(), g.file_pos(0, 0), g.blob_bytes[0], 0, 0);
    int calls = 0;
    L.fallback = [&](int64_t, const RamRun& r, uint64_t done, std::string&) {
        ++calls;
        std::memcpy(r.dst + done, f.bytes.data() + r.file_pos + done, (size_t) (r.bytes - done));
        return true;
    };
    bool after = false;
    L.after_group = [&](int64_t, int64_t done, uint64_t copied, bool fell_back, std::string&) {
        after = done == 1 && copied == g.blob_bytes[0] && fell_back;
        return true;
    };
    RamLoadStats st;
    std::string err;
    check(read_ram_runs(L, st, err), err.c_str());
    check(calls == 1 && after, "the fallback and after_group hooks run");
    check(std::memcmp(mem.data(), f.bytes.data() + g.file_pos(0, 0), mem.size()) == 0, "fallback bytes");
    L.fallback = [&](int64_t, const RamRun&, uint64_t, std::string& e) {
        e = "refused";
        return false;
    };
    check(!read_ram_runs(L, st, err) && err == "refused", "a failing hook fails the load with its error");
}

}  // namespace

int main() {
    {
        // Reader threads may move layer owners while their destination pointers stay live, without a CUDA context.
        RamAllocation span;
        check(span.allocate(4097, false), "ordinary RAM span");
        if (span.data()) {
            std::memset(span.data(), 0x5b, 4097);
            const auto* at = span.data();
            std::vector<RamAllocation> owners;
            owners.push_back(std::move(span));
            owners.emplace_back();
            check(!span.data() && owners[0].data() == at && owners[0].data()[4096] == 0x5b,
                  "moving a layer owner changed its address or bytes");
            check(!owners[1].allocate(~uint64_t(0), false) && !owners[1].data(), "RAM extent overflow");
        }
    }
    for (bool unbuffered : {true, false})
        for (int threads : {1, 4}) {
            arena_case(0, 4096 * 8, unbuffered, threads);      // block aligned: straight into RAM
            arena_case(4096, 4096 * 2 + 100, unbuffered, threads);   // not aligned: the staging or the fallback
            arena_case(1000, 4096 * 3, unbuffered, threads);
        }
    span_case(0, 4096 * 4);
    span_case(1000, 4096 * 2 + 100);
    fallback_case();
    std::printf(fails == 0 ? "ram_load_test: ok\n" : "ram_load_test: %d failures\n", fails);
    return fails == 0 ? 0 : 1;
}
