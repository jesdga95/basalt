// Device waits with a host writer of mapped flags and payloads.
#include "basalt/kernels/doorbell.cuh"
#include "basalt/kernels/elementwise.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

namespace bk = basalt::kernels;
using Clock = std::chrono::steady_clock;
using SystemWord = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;
using namespace std::chrono_literals;

constexpr uint32_t kWant = 3, kPoison = 0xf4f4f4f4u;
struct Bell {
    uint32_t flag = 0, abort = 0, started = 0, payload = kPoison;
    uint32_t matched = kPoison, observed = kPoison;
};

__global__ void consume(Bell* bell, const uint32_t* seq) {
    bk::doorbell_store_release(&bell->started, 1);
    const bool matched = bk::doorbell_spin(&bell->flag, *seq, &bell->abort, 1);
    if (matched) bell->observed = __ldcv(&bell->payload);
    bell->matched = matched;
}

__global__ void read_payload(Bell* bell) {
    bell->observed = __ldcv(&bell->payload);
}

[[noreturn]] void fail(const char* what, const char* reason) {
    std::printf("doorbell_abort_test: %s: %s\n", what, reason);
    std::fflush(stdout);
    std::_Exit(1);
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) fail(what, cudaGetErrorString(status));
}

void drain(cudaStream_t stream, const char* what) {
    const auto deadline = Clock::now() + 5s;
    do {
        const cudaError_t status = cudaStreamQuery(stream);
        if (status == cudaSuccess) return;
        if (status != cudaErrorNotReady) fail(what, cudaGetErrorString(status));
        std::this_thread::sleep_for(100us);
    } while (Clock::now() < deadline);
    fail(what, "stream did not drain within 5 seconds");
}

void wait_for_consumer(uint32_t& started) {
    const auto deadline = Clock::now() + 5s;
    do {
        if (SystemWord(started).load(cuda::memory_order_acquire)) return;
        std::this_thread::yield();
    } while (Clock::now() < deadline);
    fail("published payload", "consumer did not start");
}

void publish(Bell& bell, uint32_t payload) {
    bell.payload = payload;
    SystemWord(bell.flag).store(kWant, cuda::memory_order_release);
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;

    Bell *host = nullptr, *device = nullptr;
    uint32_t* seq = nullptr;
    cudaStream_t stream = nullptr;
    check(cudaHostAlloc((void**) &host, sizeof(*host), cudaHostAllocMapped), "allocate mapped bell");
    check(cudaHostGetDevicePointer((void**) &device, host, 0), "get device bell");
    check(cudaMalloc(&seq, sizeof(*seq)), "allocate sequence");
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    check(cudaMemcpyAsync(seq, &kWant, sizeof(kWant), cudaMemcpyHostToDevice, stream), "initialize sequence");

    *host = Bell{};
    bk::doorbell_wait(&device->flag, seq, &device->abort, stream);
    const bool waiting = cudaStreamQuery(stream) == cudaErrorNotReady;
    SystemWord(host->abort).store(1, cuda::memory_order_release);
    drain(stream, "abort a waiting stream");
    if (!waiting || host->flag != 0 || host->abort != 1 || host->payload != kPoison ||
        host->started != 0 || host->matched != kPoison || host->observed != kPoison)
        fail("abort a waiting stream", "unexpected bell state");

    for (uint32_t i = 0; i < 1000; ++i) {
        *host = Bell{};
        consume<<<1, 1, 0, stream>>>(device, seq);
        check(cudaGetLastError(), "launch payload consumer");
        wait_for_consumer(host->started);
        std::this_thread::sleep_for(20us);
        const uint32_t pattern = 0x67420000u + i;
        publish(*host, pattern);
        drain(stream, "published payload");
        if (host->matched != 1 || host->observed != pattern)
            fail("published payload", "sequence matched without its payload");
    }

    *host = Bell{};
    host->flag = kWant - 1;
    SystemWord(host->abort).store(1, cuda::memory_order_release);
    consume<<<1, 1, 0, stream>>>(device, seq);
    check(cudaGetLastError(), "launch pre-aborted consumer");
    drain(stream, "pre-aborted stale flag");
    if (host->matched != 0 || host->observed != kPoison) fail("pre-aborted stale flag", "accepted a stale completion");

    *host = Bell{};
    bk::doorbell_wait(&device->flag, seq, stream);
    read_payload<<<1, 1, 0, stream>>>(device);
    check(cudaGetLastError(), "launch legacy payload read");
    const bool legacy_waiting = cudaStreamQuery(stream) == cudaErrorNotReady;
    publish(*host, 0x87dea123u);
    drain(stream, "legacy three-argument wait");
    if (!legacy_waiting || host->observed != 0x87dea123u)
        fail("legacy three-argument wait", "sequence did not order the payload read");

    check(cudaStreamDestroy(stream), "destroy stream");
    check(cudaFree(seq), "free sequence");
    check(cudaFreeHost(host), "free bell");
    std::printf("doorbell_abort_test: ok\n");
    return 0;
}
