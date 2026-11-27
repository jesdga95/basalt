// A promoted expert keeps its slot reusable when the requested replacement cannot take it.
#include "basalt/core/peer_experts.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace basalt::core;
constexpr int dim = 256, layers = 2, experts = 5;

void check(bool ok, const std::string& error) {
    if (!ok) throw std::runtime_error(error);
}

struct FixtureFile {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("basalt-peer-reassign-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~FixtureFile() { std::error_code e; std::filesystem::remove(path, e); }
};

void run(const FixtureFile& file, size_t bytes, bool exclusive) {
    std::string err;
    FileExpertSource src;
    check(src.open(FileRange(file.path.string()), layers, experts, err), err);
    ExpertCache primary;
    if (exclusive) {
        check(primary.open(1, layers, experts, (int64_t) bytes, err), err);
        src.set_spares(2);
        check(src.pin_cache_complement(primary, err, false, {{0, 0}, {0, 1}}, -1, 0), err);
    }
    PeerExperts peer;
    check(peer.prepare(1, src, layers, experts, dim, dim, err), err);
    check(peer.open_slots(std::vector<int64_t>(2, (int64_t) bytes), err), err);
    for (int e = 0; e < 2; ++e) {
        peer.admit(0, e, e);
        peer.slot_in(e, src.blob(0, e), 0, bytes);
    }
    check(peer.read_wait(err), err);
    if (exclusive) peer.set_exclusive(&src);

    auto verify = [&](int a, int b) {
        const auto& r = peer.residency();
        const auto s = peer.slot_stats();
        check(s.allocated == 2 && s.mapped == 2 && s.in_flight == 0 && s.loaned == 0, "settled slot counts");
        check(r[a] >= 0 && r[b] >= 0 && r[a] != r[b] &&
              std::count_if(r.begin(), r.end(), [](int s) { return s >= 0; }) == 2, "lost or aliased peer slot");
        std::vector<uint8_t> read(bytes);
        for (int i : {a, b}) {
            peer.slot_out(r[i], read.data(), 0, bytes);
            check(peer.read_wait(err), err);
            check(std::all_of(read.begin(), read.end(), [i](uint8_t v) { return v == i + 1; }), "wrong slot bytes");
        }
    };
    auto move = [&](int in, int out, const uint8_t* blob) {
        check(peer.reassign(in / experts, in % experts, out / experts, out % experts, blob, err), err);
    };
    std::vector<int32_t> main(layers * experts, -1);
    std::vector<float> usage(layers * experts, 0);
    main[0] = 0;
    usage[0] = 100;
    usage[2] = 3;
    move(0, 1, src.blob(0, 1));         // already on the peer
    verify(0, 1);
    move(0, 2, nullptr);                // no source for the proposed handoff
    verify(0, 1);
    move(4, 2, src.blob(0, 2));          // the promotion source has no peer mapping
    check(peer.adapt(usage.data(), main.data(), 1, err), err);
    if (exclusive) {
        check(!src.has_resident(0, 0), "victim already had a RAM home");
        move(1, 2, src.blob(0, 2));     // another slot wants the pending eviction's destination
        move(0, 3, src.blob(0, 3));     // eviction still owns this victim, its D2H must finish
    }
    peer.apply_pending(true);
    verify(2, 1);
    if (exclusive) {
        check(src.has_resident(0, 0) && !src.has_resident(0, 2), "exclusive RAM ownership");
        check(src.blob(0, 0)[0] == 1, "exclusive eviction lost the old bytes");
    }

    move(2, 3, src.blob(0, 3));        // a normal promotion queues its replacement
    const auto refilling = peer.slot_stats();
    check(refilling.allocated == 2 && refilling.mapped == 1 && refilling.in_flight == 1, "refill ownership counts");
    move(1, 3, src.blob(0, 3));        // the same destination cannot own two slots
    peer.apply_pending(true);
    verify(1, 3);
    const auto skips = peer.slot_stats().skipped_refills;
    check(skips.no_source_mapping == 1 && skips.destination_mapped == 1 && skips.missing_host_copy == 1 &&
          skips.destination_in_flight == (exclusive ? 2 : 1) && skips.source_in_flight == (exclusive ? 1 : 0),
          "promotion refill skip counts");
    if (!exclusive) {
        move(1, experts, src.blob(1, 0));   // equal-sized slots may move across layers
        peer.apply_pending(true);
        verify(experts, 3);
        std::vector<std::pair<int32_t, int32_t>> lent;
        check(peer.lend(0, lent, err), err);
        const auto borrowed = peer.slot_stats();
        check(borrowed.allocated == 2 && borrowed.mapped == 0 && borrowed.loaned == 2, "loaned slot counts");
        check(peer.give_back(lent, err), err);
        check(peer.slot_stats().loaned == 2, "loan remains until refill lands");
        check(peer.give_back_wait(lent, err), err);
        verify(experts, 3);
    }
}
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 2) return 77;
    try {
        using namespace basalt::kernels::cpu;
        NativeFmt fmt;
        std::string err;
        check(native_fmt(GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, dim, dim, fmt, err), err);
        std::string layout;
        FixtureFile file;
        std::ofstream out(file.path, std::ios::binary);
        for (int l = 0; l < layers; ++l) {
            layout += std::to_string(l) + " " + std::to_string(GGML_TYPE_Q4_K) + " " + std::to_string(GGML_TYPE_Q4_K) +
                      " " + std::to_string(l * experts * fmt.bytes) + " " + std::to_string(fmt.bytes) + "\n";
            for (int e = 0; e < experts; ++e) {
                std::vector<char> data(fmt.bytes, (char) (l * experts + e + 1));
                out.write(data.data(), (std::streamsize) data.size());
            }
        }
        out.close();
        check((bool) out, "write expert fixture");
        check(expert_layout_load_text(layout, layers, experts, err, dim, dim), err);
        run(file, fmt.bytes, false);
        run(file, fmt.bytes, true);
        std::puts("peer promotion, refill, slot bytes and exclusive RAM ownership passed");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "peer_reassign_test: %s\n", e.what());
        return 1;
    }
    return 0;
}
