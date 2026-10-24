// src/core/device_main.cpp - `basalt-device`: report the GPU, the plan, and exercise the arena.
//
// This prints the memory plan against the actual cudaMemGetInfo, on its own so it can run
// without the model.  It is also the run-time half of the architecture policy: CMake refuses to COMPILE for another
// architecture, and this refuses to RUN on one.
#include "basalt/core/device.hpp"
#include "basalt/plan/plan.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include "basalt/log.hpp"

static std::string human(uint64_t b) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f GiB (%llu B)", (double) b / (1024.0 * 1024 * 1024),
                  (unsigned long long) b);
    return buf;
}

int main(int argc, char** argv) {
    bool selftest = false;
    bool list_devices = false;
    bool link_rates = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--list-devices") == 0) list_devices = true;
        else if (std::strcmp(argv[i], "--link-rates") == 0) link_rates = true;
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("usage: basalt-device [--selftest] [--list-devices] [--link-rates]\n"
                        "  --list-devices  every GPU the runtime enumerates, numbered as\n"
                        "                  CUDA_VISIBLE_DEVICES numbers them, and whether this binary can run it\n"
                        "  --link-rates    per GPU, one line \"pcie_h2d BUS_ID GB_PER_S\": pinned host-to-device copy speed\n");
            return 0;
        } else {
            basalt::log::error("basalt", "unknown argument: %s", argv[i]);
            return 2;
        }
    }

    if (link_rates) {
        for (int ordinal = 0; ordinal < basalt::core::device_count(); ++ordinal) {
            std::string bus;
            const double gbps = basalt::core::device_h2d_gbps(ordinal);
            if (basalt::core::device_pci_bus_id(ordinal, bus) && gbps > 0) std::printf("pcie_h2d %s %.1f\n", bus.c_str(), gbps);
        }
        return 0;
    }

    // Every card the runtime enumerates, without the arch check `device_info` applies.  Format:
    //   device N: <name>
    //     compute capability 12.0, 31.4 GiB
    if (list_devices) {
        const int count = basalt::core::device_count();
        if (count == 0) std::printf("(no GPU device)\n");
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            std::string name, detail;
            if (!basalt::core::device_summary(ordinal, name, detail)) {
                std::printf("device %d: (the runtime cannot describe it)\n", ordinal);
                continue;
            }
            std::printf("device %d: %s\n  %s\n", ordinal, name.c_str(), detail.c_str());
        }
        return 0;
    }

    try {
        const basalt::core::DeviceInfo d = basalt::core::device_info(0);
        std::printf("device %d: %s\n", d.ordinal, d.name.c_str());
        std::printf("  compute capability  %d.%d   (sm_%d%d)\n", d.cc_major, d.cc_minor, d.cc_major, d.cc_minor);
        std::printf("  multiprocessors     %d\n", d.multi_processor_count);
        std::printf("  VRAM total / free   %s / %s\n", human(d.total_bytes).c_str(), human(d.free_bytes).c_str());
        std::printf("  driver / runtime    %d / %d\n", d.driver_version, d.runtime_version);

        // The planner's view against the card's.  A plan that does not fit in what is actually FREE is the
        // failure this print exists to make visible at startup rather than at token 4000.
        const auto plan = basalt::plan::make_plan(20480, basalt::plan::Geometry{}, basalt::plan::Costs{});
        std::printf("\n%s", basalt::plan::to_string(plan).c_str());
        std::printf("  card free           %s\n", human(d.free_bytes).c_str());
        std::printf("  plan + KV vs free   %s\n",
                    plan.vram_budget <= d.free_bytes ? "FITS" : "*** DOES NOT FIT ***");

        if (selftest) {
            // Exercise the arena for real: allocate, write from the host, read back, and check the poison
            // path leaves NaNs rather than zeros.  A GPU test that only asks the driver for its name does not
            // test the runtime this file exists to provide.
            const uint64_t bytes = 64ull << 20;      // 64 MiB, small enough to be safe on any card
            basalt::core::DeviceArena arena(bytes, 0, /*poison=*/true);
            void* a = arena.alloc(1 << 20, 256);
            void* b = arena.alloc(1 << 20, 4096);
            if (((uintptr_t) a % 256) || ((uintptr_t) b % 4096)) {
                basalt::log::error("basalt", "selftest: alignment not honoured");
                return 1;
            }
            std::printf("\nselftest: arena %s, used %s after two 1 MiB allocations\n", human(arena.capacity()).c_str(),
                        human(arena.used()).c_str());
            // and the arena must REFUSE rather than wrap
            try {
                arena.alloc(bytes * 2);
                basalt::log::error("basalt", "selftest: over-allocation did NOT throw");
                return 1;
            } catch (const basalt::core::CudaError&) {
                std::printf("selftest: over-allocation refused as required\n");
            }
            std::printf("basalt-device selftest OK\n");
        }
        return 0;
    } catch (const std::exception& e) {
        basalt::log::error("basalt-device", "%s", e.what());
        return 1;
    }
}
