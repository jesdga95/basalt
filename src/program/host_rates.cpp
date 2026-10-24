// src/program/host_rates.cpp - see include/basalt/program/host_rates.hpp.
#include "basalt/program/host_rates.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <sstream>
#include <thread>

#include "basalt/platform/cpu.hpp"
#include "basalt/platform/memory.hpp"

namespace basalt::program {

namespace {
constexpr size_t kBytes = 2ull << 30;
constexpr int kPasses = 8;
constexpr size_t kLine = 64;
}  // namespace

double measure_ram_read_gbps() {
    const size_t threads = std::max<size_t>(1, platform::detect_cpu_topology(false).worker_cores.size());
    if (platform::available_memory() < kBytes + (1ull << 30)) return 0;
    std::unique_ptr<uint64_t[]> buf(new (std::nothrow) uint64_t[kBytes / sizeof(uint64_t)]);
    if (!buf) return 0;
    const size_t words = kBytes / sizeof(uint64_t), per = words / threads / (kLine / 8) * (kLine / 8);

    std::atomic<size_t> ready{0};
    std::atomic<int> go{0};   // pass number to run; -1 ends the workers
    std::atomic<size_t> done{0};
    std::atomic<uint64_t> sink{0};
    std::vector<std::thread> pool;
    for (size_t t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            uint64_t* mine = buf.get() + t * per;
            for (size_t i = 0; i < per; ++i) mine[i] = i + t;   // first touch by the reader: pages land near it
            int seen = 0;
            ready.fetch_add(1);
            for (;;) {
                int g;
                while ((g = go.load(std::memory_order_acquire)) == seen) std::this_thread::yield();
                if (g < 0) return;
                seen = g;
                uint64_t x = 0;
                for (size_t i = 0; i < per; i += kLine / 8) x += *(const volatile uint64_t*) (mine + i);
                sink.fetch_add(x, std::memory_order_relaxed);
                done.fetch_add(1);
            }
        });
    while (ready.load() < threads) std::this_thread::yield();

    double best = 0;
    for (int pass = 1; pass <= kPasses; ++pass) {   // pass 1 only warms the cores and the TLB
        done.store(0);
        const auto t0 = std::chrono::steady_clock::now();
        go.store(pass, std::memory_order_release);
        while (done.load() < threads) std::this_thread::yield();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (pass > 1 && s > 0) best = std::max(best, (double) (per * threads * 8) / s / 1e9);
    }
    go.store(-1, std::memory_order_release);
    for (std::thread& t : pool) t.join();
    return best;
}

std::string normalize_bus_id(const std::string& id) {
    std::string s;
    for (unsigned char c : id) s += (char) std::tolower(c);
    const size_t nz = s.find_first_not_of('0');
    return nz == std::string::npos ? std::string() : s.substr(nz);
}

std::vector<std::pair<std::string, double>> parse_link_rates(const std::string& text) {
    std::vector<std::pair<std::string, double>> out;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
        std::istringstream w(line);
        std::string tag, bus, gbps;
        if (!(w >> tag >> bus >> gbps) || tag != "pcie_h2d") continue;
        char* end = nullptr;
        const double v = std::strtod(gbps.c_str(), &end);
        if (end != gbps.c_str() && v > 0) out.emplace_back(bus, v);
    }
    return out;
}

}  // namespace basalt::program
