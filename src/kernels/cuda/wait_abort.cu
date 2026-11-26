// src/kernels/cuda/wait_abort.cu - see include/basalt/kernels/wait_abort.hpp.
#include "basalt/kernels/wait_abort.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>

namespace basalt::kernels {

namespace {
constexpr int kAbortDevs = 16;
std::mutex g_abort_mu;
unsigned long long* g_abort_shared = nullptr;
std::atomic<unsigned long long*> g_abort_host[kAbortDevs];
std::atomic<unsigned long long*> g_abort_dev[kAbortDevs];
}  // namespace

bool wait_abort_try_init() {
    int dev = 0;
    if (const cudaError_t e = cudaGetDevice(&dev); e != cudaSuccess) {
        (void) cudaGetLastError();
        basalt::log::error("basalt", "wait abort word: %s", cudaGetErrorString(e));
        return false;
    }
    if (dev < 0 || dev >= kAbortDevs) {
        basalt::log::error("basalt", "wait abort word: device %d out of range", dev);
        return false;
    }
    if (g_abort_dev[dev].load(std::memory_order_acquire) != nullptr) return true;

    std::lock_guard<std::mutex> lock(g_abort_mu);
    if (g_abort_dev[dev].load(std::memory_order_relaxed) != nullptr) return true;

    if (g_abort_shared != nullptr) {
        void* d = nullptr;
        if (cudaHostGetDevicePointer(&d, g_abort_shared, 0) == cudaSuccess) {
            g_abort_host[dev].store(g_abort_shared, std::memory_order_release);
            g_abort_dev[dev].store((unsigned long long*) d, std::memory_order_release);
            return true;
        }
        (void) cudaGetLastError();
    }

    int count = 1;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void) cudaGetLastError();
        count = 1;
    }

    if (g_abort_shared == nullptr && count > 1) {
        void* p = nullptr;
        const cudaError_t pe = cudaHostAlloc(&p, 64, cudaHostAllocMapped | cudaHostAllocPortable);
        if (pe == cudaSuccess) {
            *(volatile unsigned long long*) p = 0;
            void* d = nullptr;
            const cudaError_t de = cudaHostGetDevicePointer(&d, p, 0);
            if (de == cudaSuccess) {
                g_abort_shared = (unsigned long long*) p;
                g_abort_host[dev].store(g_abort_shared, std::memory_order_release);
                g_abort_dev[dev].store((unsigned long long*) d, std::memory_order_release);
                return true;
            }
            (void) cudaGetLastError();
            (void) cudaFreeHost(p);
            basalt::log::warn("basalt", "wait abort word: portable device alias failed on device %d (%s); falling back to per-device mapped memory",
                              dev, cudaGetErrorString(de));
        } else {
            (void) cudaGetLastError();
            basalt::log::warn("basalt", "wait abort word: portable allocation failed on device %d (%s); falling back to per-device mapped memory",
                              dev, cudaGetErrorString(pe));
        }
    }

    void* p = nullptr;
    if (const cudaError_t e = cudaHostAlloc(&p, 64, cudaHostAllocMapped); e != cudaSuccess) {
        (void) cudaGetLastError();
        basalt::log::error("basalt", "wait abort word: mapped allocation failed on device %d: %s", dev, cudaGetErrorString(e));
        return false;
    }
    *(volatile unsigned long long*) p = 0;
    void* d = nullptr;
    if (const cudaError_t e = cudaHostGetDevicePointer(&d, p, 0); e != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFreeHost(p);
        basalt::log::error("basalt", "wait abort word: device alias failed on device %d: %s", dev, cudaGetErrorString(e));
        return false;
    }
    g_abort_host[dev].store((unsigned long long*) p, std::memory_order_release);
    g_abort_dev[dev].store((unsigned long long*) d, std::memory_order_release);
    return true;
}

void wait_abort_init() { (void) wait_abort_try_init(); }

unsigned long long* wait_abort_device_word() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kAbortDevs) {
        (void) cudaGetLastError();
        return nullptr;
    }
    unsigned long long* w = g_abort_dev[dev].load(std::memory_order_acquire);
    if (w == nullptr && wait_abort_try_init()) w = g_abort_dev[dev].load(std::memory_order_acquire);
    return w;
}

unsigned long long wait_abort_code() {
    for (int d = 0; d < kAbortDevs; ++d) {
        unsigned long long* const h = g_abort_host[d].load(std::memory_order_acquire);
        if (h == nullptr) continue;
        const unsigned long long v = *(volatile unsigned long long*) h;
        if (v != 0ull) return v;
    }
    return 0ull;
}

std::string wait_abort_text(unsigned long long code) {
    const unsigned who = (unsigned) (code >> 56) & 0x7fu;
    char buf[128];
    std::snprintf(buf, sizeof buf, "%s: kind %u, line %u, block %u",
                  who == kWaitMidChain ? "mid chain" : who == kWaitCommitFinisher ? "commit finisher"
                  : who == kWaitHostRelease ? "released by the host" : "device wait",
                  (unsigned) (code >> 48) & 0xffu, (unsigned) (code >> 32) & 0xffffu, (unsigned) code);
    return buf;
}

void wait_abort_release() {
    const unsigned long long code = (1ull << 63) | ((unsigned long long) kWaitHostRelease << 56);
    for (int d = 0; d < kAbortDevs; ++d) {
        volatile unsigned long long* const w = g_abort_host[d].load(std::memory_order_acquire);
        if (w != nullptr && *w == 0) *w = code;   // a wait's own code stays
    }
}

void wait_abort_clear() {
    for (int d = 0; d < kAbortDevs; ++d) {
        volatile unsigned long long* const w = g_abort_host[d].load(std::memory_order_acquire);
        if (w != nullptr) *w = 0;
    }
}

}  // namespace basalt::kernels
