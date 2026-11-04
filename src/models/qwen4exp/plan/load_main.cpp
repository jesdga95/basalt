// src/models/qwen4exp/plan/load_main.cpp - `basalt-load`: the expert arena load, measurable at any scale.
//
// It loads with several threads in large chunks and checksums each layer.  The size is a FLAG rather than hard-coded because the honest way to measure a 34 GB
// load is to measure a smaller one first and project - and because pinning 34 GB evicts every page of page
// cache on the machine, which is not something a test should do while other work is running.
#include "basalt/core/pinned.hpp"
#include "basalt/core/ram_load.hpp"
#include "basalt/platform/file.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "basalt/log.hpp"

int main(int argc, char** argv) {
    std::string path;
    uint64_t layers = 2;
    int threads = 8;
    uint64_t chunk = 16ull << 20;
    uint64_t blob_bytes = 1382400;      // one expert blob
    uint64_t blobs_per_layer = 512;
    bool pin = true;
    bool stream = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) { basalt::log::error("basalt", "%s needs a value", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--file") path = val();
        else if (a == "--layers") layers = std::strtoull(val(), nullptr, 10);
        else if (a == "--threads") threads = std::atoi(val());
        else if (a == "--chunk-mb") chunk = std::strtoull(val(), nullptr, 10) << 20;
        else if (a == "--no-pin") pin = false;
        else if (a == "--stream") stream = true;
        else if (a == "--help" || a == "-h") {
            std::printf("usage: basalt-load --file experts.bin [--layers N] [--threads N] [--chunk-mb N]\n"
                        "                   [--no-pin]\n");
            return 0;
        } else {
            basalt::log::error("basalt", "unknown argument: %s", a.c_str());
            return 2;
        }
    }
    if (path.empty()) { basalt::log::error("basalt", "--file is required"); return 2; }

    const uint64_t bytes = layers * blobs_per_layer * blob_bytes;
    std::printf("expert arena: %llu layers x %llu blobs x %llu B = %.3f GiB\n", (unsigned long long) layers,
                (unsigned long long) blobs_per_layer, (unsigned long long) blob_bytes,
                (double) bytes / (1024.0 * 1024 * 1024));

    basalt::core::PinnedArena arena;
    uint8_t* dst = nullptr;
    if (pin) {
        // constructed in place so the note from the constructor can be printed before the load
        new (&arena) basalt::core::PinnedArena(bytes);
        std::printf("backing: %s\n", arena.note.c_str());
        if (!arena.valid()) { basalt::log::error("basalt", "arena allocation failed"); return 1; }
        dst = arena.data();
    } else {
        dst = (uint8_t*) std::malloc((size_t) bytes);
        if (!dst) { basalt::log::error("basalt", "malloc failed"); return 1; }
        std::printf("backing: malloc, NOT pinned (--no-pin)\n");
    }

    const int fd = basalt::platform::open_fd_utf8(path);
    if (fd < 0) { basalt::log::error("basalt", "cannot open %s", path.c_str()); return 1; }
    const uint64_t layer_bytes = blobs_per_layer * blob_bytes;
    basalt::core::LoadProgress progress("ram", bytes);
    basalt::core::RamLoad load;
    load.fd = fd;
    load.unbuffered = false;
    load.threads = threads;
    load.chunk = chunk;
    load.progress = &progress;
    load.groups.resize((size_t) layers);
    for (uint64_t l = 0; l < layers; ++l)
        for (uint64_t pos = 0; pos < layer_bytes; pos += chunk) {
            basalt::core::RamRun r;
            r.dst = dst + (size_t) (l * layer_bytes + pos);
            r.file_pos = l * layer_bytes + pos;
            r.bytes = layer_bytes - pos < chunk ? layer_bytes - pos : chunk;
            r.progress = r.bytes;
            load.groups[(size_t) l].push_back(r);
        }
    basalt::core::RamLoadStats st;
    std::string err;
    if (!basalt::core::read_ram_runs(load, st, err)) {
        basalt::log::error("basalt", "loading the experts failed: %s", err.c_str());
        return 1;
    }
    progress.finish();
    basalt::platform::close_fd(fd);
    const double gib_per_s = st.seconds > 0 ? (double) st.copied / (1024.0 * 1024 * 1024) / st.seconds : 0.0;

    std::printf("loaded %.3f GiB in %.3f s with %d threads and %llu MiB chunks  ->  %.2f GiB/s\n",
                (double) st.copied / (1024.0 * 1024 * 1024), st.seconds, threads,
                (unsigned long long) (chunk >> 20), gib_per_s);
    std::printf("per-layer checksums (FNV-1a 64):\n");
    for (uint64_t i = 0; i < layers && i < 4; ++i) {
        std::printf("  layer %llu  %016llx\n", (unsigned long long) i,
                    (unsigned long long) basalt::core::fnv1a64(dst + (size_t) (i * layer_bytes), layer_bytes));
    }
    if (layers > 4) {
        std::printf("  ... %llu more (all layers loaded)\n", (unsigned long long) layers - 4);
    }
    if (stream) {
        // The bus is the ceiling, so measure it at the granularity the engine actually uses.  Printed as a
        // sweep rather than one number because "bandwidth" without a transfer size is not a specification.
        std::printf("\nexpert stream over the pinned arena (host -> device, one reused destination):\n");
        const uint64_t granularities[] = {blob_bytes, 16 * blob_bytes, 512 * blob_bytes, bytes};
        const char* names[] = {"1 blob", "16 blobs", "1 layer", "whole arena"};
        for (int i = 0; i < 4; ++i) {
            const uint64_t g = granularities[i];
            if (g > bytes) continue;
            const int iters = (int) (g >= bytes ? 3 : (bytes / g >= 8 ? 8 : 3));
            const basalt::core::StreamStats ss =
                basalt::core::stream_bandwidth(dst, (bytes / g) * g, g, iters);
            if (ss.seconds < 0) continue;
            std::printf("  %-18s chunk %8.2f MB  %6.2f GB/s  (%.3f GiB in %.3f s)\n", names[i],
                        (double) g / 1e6, ss.gib_per_second() * 1.073741824,
                        (double) ss.bytes / (1024.0 * 1024 * 1024), ss.seconds);
        }
        // And what it means, using the design's own constants, stated with the assumption attached.
        const double best = 60.0;    // GB/s
        (void) best;
        std::printf("\nper-token expert stream at E = 663.6 MB and hit rate h (VRAM-resident fraction):\n");
        std::printf("  %-8s %-14s %-14s %s\n", "h", "miss MB/token", "ms at 1 blob", "tok/s ceiling");
        const basalt::core::StreamStats one = basalt::core::stream_bandwidth(dst, blob_bytes, blob_bytes, 64);
        for (double h : {0.0, 0.5, 0.833, 0.9, 0.95}) {
            const double miss_mb = (1.0 - h) * 663.6;
            const double ms = miss_mb / 1000.0 / (one.gib_per_second() * 1.073741824) * 1000.0;
            std::printf("  %-8.3f %-14.1f %-14.2f %.1f\n", h, miss_mb, ms, ms > 0 ? 1000.0 / ms : 0.0);
        }
        std::printf("  E is the FULL expert set per token, so which fraction is cached is the question the\n"
                    "  number hinges on.\n");
    }

    // The projection is printed as a projection rather than as the result, because at --layers 2 this
    // program has NOT loaded the full set.
    const double full = 33973862400.0;
    std::printf("projection for the full %.2f GiB at this rate: %.1f s\n", full / (1024.0 * 1024 * 1024),
                full / (1024.0 * 1024 * 1024) / (gib_per_s > 0 ? gib_per_s : 1e-9));
    return 0;
}
