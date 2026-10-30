// src/kernels/cuda/wait_abort.cu - see include/basalt/kernels/wait_abort.hpp.
#include "basalt/kernels/wait_abort.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace basalt::kernels {

namespace {
constexpr int kAbortDevs = 16;
unsigned long long* g_abort_host = nullptr;
unsigned long long* g_abort_dev[kAbortDevs] = {};
}  // namespace

void wait_abort_init() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= kAbortDevs) {
        basalt::log::error("basalt", "wait abort word: device %d out of range", dev);
        std::exit(1);
    }
    if (g_abort_dev[dev] != nullptr) return;
    if (g_abort_host == nullptr) {
        void* p = nullptr;
        if (cudaHostAlloc(&p, 64, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            basalt::log::error("basalt", "wait abort word: %s", cudaGetErrorString(cudaGetLastError()));
            std::exit(1);
        }
        *(volatile unsigned long long*) p = 0;
        g_abort_host = (unsigned long long*) p;
    }
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, g_abort_host, 0) != cudaSuccess) {
        basalt::log::error("basalt", "wait abort word: %s", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
    g_abort_dev[dev] = (unsigned long long*) d;
}

unsigned long long* wait_abort_device_word() {
    int dev = 0;
    cudaGetDevice(&dev);
    unsigned long long* const w = dev >= 0 && dev < kAbortDevs ? g_abort_dev[dev] : nullptr;
    if (w == nullptr) {
        basalt::log::error("basalt", "wait abort word: not initialised on device %d", dev);
        std::exit(1);
    }
    return w;
}

unsigned long long wait_abort_code() { return g_abort_host != nullptr ? *(volatile unsigned long long*) g_abort_host : 0ull; }

std::string wait_abort_text(unsigned long long code) {
    const unsigned who = (unsigned) (code >> 56) & 0x7fu;
    char buf[128];
    std::snprintf(buf, sizeof buf, "%s: kind %u, line %u, block %u",
                  who == kWaitMidChain ? "mid chain" : who == kWaitCommitFinisher ? "commit finisher" : "device wait",
                  (unsigned) (code >> 48) & 0xffu, (unsigned) (code >> 32) & 0xffffu, (unsigned) code);
    return buf;
}

void wait_abort_clear() {
    if (g_abort_host != nullptr) *(volatile unsigned long long*) g_abort_host = 0;
}

}  // namespace basalt::kernels
