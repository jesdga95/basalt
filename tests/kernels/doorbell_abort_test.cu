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

cudaError_t poll_stream(cudaStream_t stream, Clock::duration timeout) {
    const auto deadline = Clock::now() + timeout;
    do {
        const cudaError_t status = cudaStreamQuery(stream);
        if (status != cudaErrorNotReady) return status;
        std::this_thread::sleep_for(100us);
    } while (Clock::now() < deadline);
    return cudaErrorNotReady;
}

[[noreturn]] void fail(const char* what, const char* reason, Bell* bell = nullptr, cudaStream_t stream = nullptr) {
    if (bell) {
        SystemWord(bell->abort).store(1, cuda::memory_order_release);
        // The legacy wait only observes the flag.
        SystemWord(bell->flag).store(kWant, cuda::memory_order_release);
    }
    const cudaError_t cleanup = stream ? poll_stream(stream, 5s) : cudaSuccess;
    std::printf("doorbell_abort_test: %s: %s\n", what, reason);
    if (cleanup != cudaSuccess)
        std::printf("doorbell_abort_test: failure drain: %s\n", cudaGetErrorString(cleanup));
    std::fflush(stdout);
    std::_Exit(1);
}

void check(cudaError_t status, const char* what, Bell* bell = nullptr, cudaStream_t stream = nullptr) {
    if (status != cudaSuccess) fail(what, cudaGetErrorString(status), bell, stream);
}

void drain(Bell& bell, cudaStream_t stream, const char* what) {
    const cudaError_t status = poll_stream(stream, 60s);
    if (status == cudaErrorNotReady) fail(what, "stream did not drain within 60 seconds", &bell, stream);
    check(status, what, &bell, stream);
}

void wait_for_consumer(Bell& bell, cudaStream_t stream) {
    const auto deadline = Clock::now() + 60s;
    do {
        if (SystemWord(bell.started).load(cuda::memory_order_acquire)) return;
        std::this_thread::yield();
    } while (Clock::now() < deadline);
    fail("published payload", "consumer did not start within 60 seconds", &bell, stream);
}

void passed(const char* what, Clock::time_point start) {
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::printf("doorbell_abort_test: %s: ok (%.3f ms)\n", what, ms);
    std::fflush(stdout);
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

    auto start = Clock::now();
    *host = Bell{};
    bk::doorbell_wait(&device->flag, seq, &device->abort, stream);
    const bool waiting = cudaStreamQuery(stream) == cudaErrorNotReady;
    SystemWord(host->abort).store(1, cuda::memory_order_release);
    drain(*host, stream, "abort a waiting stream");
    if (host->flag != 0 || host->abort != 1 || host->payload != kPoison ||
        host->started != 0 || host->matched != kPoison || host->observed != kPoison)
        fail("abort a waiting stream", "unexpected bell state", host, stream);
    std::printf("  abortable wait: immediate not-ready observed: %s (diagnostic only)\n", waiting ? "yes" : "no");
    passed("abort a waiting stream", start);

    start = Clock::now();
    constexpr uint32_t rounds = 1000;
    for (uint32_t i = 0; i < rounds; ++i) {
        *host = Bell{};
        consume<<<1, 1, 0, stream>>>(device, seq);
        check(cudaGetLastError(), "launch payload consumer", host, stream);
        wait_for_consumer(*host, stream);
        std::this_thread::sleep_for(20us);
        const uint32_t pattern = 0x67420000u + i;
        publish(*host, pattern);
        drain(*host, stream, "published payload");
        if (host->matched != 1 || host->observed != pattern)
            fail("published payload", "sequence matched without its payload", host, stream);
    }
    std::printf("  published payload: %u rounds\n", rounds);
    passed("published payload", start);

    start = Clock::now();
    *host = Bell{};
    host->flag = kWant - 1;
    SystemWord(host->abort).store(1, cuda::memory_order_release);
    consume<<<1, 1, 0, stream>>>(device, seq);
    check(cudaGetLastError(), "launch pre-aborted consumer", host, stream);
    drain(*host, stream, "pre-aborted stale flag");
    if (host->matched != 0 || host->observed != kPoison)
        fail("pre-aborted stale flag", "accepted a stale completion", host, stream);
    passed("pre-aborted stale flag", start);

    start = Clock::now();
    *host = Bell{};
    bk::doorbell_wait(&device->flag, seq, stream);
    read_payload<<<1, 1, 0, stream>>>(device);
    check(cudaGetLastError(), "launch legacy payload read", host, stream);
    const bool legacy_waiting = cudaStreamQuery(stream) == cudaErrorNotReady;
    publish(*host, 0x87dea123u);
    drain(*host, stream, "legacy three-argument wait");
    if (host->observed != 0x87dea123u)
        fail("legacy three-argument wait", "sequence did not order the payload read", host, stream);
    std::printf("  legacy wait: immediate not-ready observed: %s (diagnostic only)\n", legacy_waiting ? "yes" : "no");
    passed("legacy three-argument wait", start);

    check(cudaStreamDestroy(stream), "destroy stream");
    check(cudaFree(seq), "free sequence");
    check(cudaFreeHost(host), "free bell");
    std::printf("doorbell_abort_test: ok\n");
    return 0;
}
