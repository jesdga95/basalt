// src/core/card_link.cpp - see include/basalt/core/card_link.hpp.
#include "basalt/core/gpu_mem_info.hpp"
#include "basalt/core/card_link.hpp"

#include "basalt/core/on_device.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

namespace basalt::core {

int enable_peer_access(int n, int& tried) {
    int pairs = 0, prev = 0;
    tried = 0;
    cudaGetDevice(&prev);
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
            int can = 0;
            if (a == b || cudaDeviceCanAccessPeer(&can, a, b) != cudaSuccess || !can) {
                (void) cudaGetLastError();
                continue;
            }
            ++tried;
            if (cudaSetDevice(a) != cudaSuccess) {
                (void) cudaGetLastError();
                continue;
            }
            const cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
            if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) ++pairs;
            (void) cudaGetLastError();
        }
    (void) cudaSetDevice(prev);
    return pairs;
}

// A single timed burst can read a link that is still in a low-power state, or sharing DMA, far too low.  Such a disturbance
// only ever slows a copy: nothing makes a correctly timed copy faster than the link carries.  So 1 GiB is copied as four
// bursts of 256 MiB, timed one by one, and the fastest is the link's figure.
double probe_h2d_gbps(std::string* samples) {
    constexpr size_t kBytes = 256ull << 20;
    constexpr int kBursts = 4;
    void* h = nullptr;
    void* d = nullptr;
    if (cudaMallocHost(&h, kBytes) != cudaSuccess) return -1.0;
    if (cudaMalloc(&d, kBytes) != cudaSuccess) {
        cudaFreeHost(h);
        return -1.0;
    }
    std::memset(h, 0, kBytes);   // fault the pages in before timing
    cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);   // warmup: context up, copy engine primed
    float ms[kBursts] = {};
    // the bursts run back to back on the stream, an event between each two
    cudaEvent_t ev[kBursts + 1];
    int n_ev = 0;
    while (n_ev <= kBursts && cudaEventCreate(&ev[n_ev]) == cudaSuccess) ++n_ev;
    bool ok = n_ev == kBursts + 1;
    if (ok) {
        cudaEventRecord(ev[0]);
        for (int b = 0; b < kBursts; ++b) {
            cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);
            cudaEventRecord(ev[b + 1]);
        }
        ok = cudaEventSynchronize(ev[kBursts]) == cudaSuccess;
        for (int b = 0; b < kBursts && ok; ++b) ok = cudaEventElapsedTime(&ms[b], ev[b], ev[b + 1]) == cudaSuccess;
    }
    for (int i = 0; i < n_ev; ++i) cudaEventDestroy(ev[i]);
    double bw = -1.0;
    if (samples != nullptr) samples->clear();
    for (int b = 0; b < kBursts && ok; ++b) {
        const double s = ms[b] > 0.01f ? (double) kBytes / (ms[b] * 1e-3) / 1e9 : -1.0;
        bw = std::max(bw, s);
        if (samples != nullptr) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%s%.1f", b == 0 ? "" : " ", s);
            *samples += buf;
        }
    }
    if (!ok) (void) cudaGetLastError();
    cudaFree(d);
    cudaFreeHost(h);
    return bw;
}

uint64_t card_free_bytes(int device) {
    const OnDevice on(device);
    size_t free_b = 0, total_b = 0;
    if (basalt::core::gpu_mem_info(&free_b, &total_b) != cudaSuccess) {
        (void) cudaGetLastError();
        return 0;
    }
    return free_b;
}

}  // namespace basalt::core
