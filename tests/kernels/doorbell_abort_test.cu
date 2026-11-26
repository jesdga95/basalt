// Device waits with a host writer of mapped flags and payloads.
#include "basalt/kernels/doorbell.cuh"
#include "basalt/kernels/elementwise.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
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

void release_waiters(Bell& bell) {
    SystemWord(bell.abort).store(1, cuda::memory_order_release);
    // The legacy wait only observes the flag.
    SystemWord(bell.flag).store(kWant, cuda::memory_order_release);
}

class Watchdog {
public:
    Watchdog() : thread_([this, deadline = Clock::now() + 300s] {
        std::unique_lock<std::mutex> lock(mutex_);
        if (wake_.wait_until(lock, deadline, [this] { return done_; })) return;
        if (bell_) release_waiters(*bell_);
        std::printf("doorbell_abort_test: watchdog: %s\n", current_case_);
        std::fflush(stdout);
        std::_Exit(1);
    }) {}

    ~Watchdog() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        wake_.notify_one();
        thread_.join();
    }

    void set_bell(Bell* bell) {
        std::lock_guard<std::mutex> lock(mutex_);
        bell_ = bell;
    }

    void set_case(const char* name) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_case_ = name;
    }

    void reset_bell() {
        std::lock_guard<std::mutex> lock(mutex_);
        *bell_ = Bell{};
    }

private:
    // Protects the bell's lifetime and whole-bell resets without locking across CUDA calls.
    std::mutex mutex_;
    std::condition_variable wake_;
    Bell* bell_ = nullptr;
    const char* current_case_ = "setup";
    bool done_ = false;
    std::thread thread_;
};

[[noreturn]] void fail(const char* what, const char* reason, Bell* bell = nullptr, cudaStream_t stream = nullptr) {
    if (bell) release_waiters(*bell);
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
    Watchdog watchdog;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;

    Bell *host = nullptr, *device = nullptr;
    uint32_t* seq = nullptr;
    cudaStream_t stream = nullptr;
    check(cudaHostAlloc((void**) &host, sizeof(*host), cudaHostAllocMapped), "allocate mapped bell");
    *host = Bell{};
    watchdog.set_bell(host);
    check(cudaHostGetDevicePointer((void**) &device, host, 0), "get device bell");
    check(cudaMalloc(&seq, sizeof(*seq)), "allocate sequence");
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    check(cudaMemcpyAsync(seq, &kWant, sizeof(kWant), cudaMemcpyHostToDevice, stream), "initialize sequence");

    watchdog.set_case("abort a waiting stream");
    auto start = Clock::now();
    watchdog.reset_bell();
    bk::doorbell_wait(&device->flag, seq, &device->abort, stream);
    const bool waiting = cudaStreamQuery(stream) == cudaErrorNotReady;
    SystemWord(host->abort).store(1, cuda::memory_order_release);
    drain(*host, stream, "abort a waiting stream");
    if (SystemWord(host->flag).load(cuda::memory_order_relaxed) != 0 ||
        SystemWord(host->abort).load(cuda::memory_order_relaxed) != 1 || host->payload != kPoison ||
        host->started != 0 || host->matched != kPoison || host->observed != kPoison)
        fail("abort a waiting stream", "unexpected bell state", host, stream);
    std::printf("  abortable wait: immediate not-ready observed: %s (diagnostic only)\n", waiting ? "yes" : "no");
    passed("abort a waiting stream", start);

    watchdog.set_case("published payload");
    start = Clock::now();
    constexpr uint32_t rounds = 1000;
    for (uint32_t i = 0; i < rounds; ++i) {
        watchdog.reset_bell();
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

    watchdog.set_case("pre-aborted stale flag");
    start = Clock::now();
    watchdog.reset_bell();
    SystemWord(host->flag).store(kWant - 1, cuda::memory_order_relaxed);
    SystemWord(host->abort).store(1, cuda::memory_order_release);
    consume<<<1, 1, 0, stream>>>(device, seq);
    check(cudaGetLastError(), "launch pre-aborted consumer", host, stream);
    drain(*host, stream, "pre-aborted stale flag");
    if (host->matched != 0 || host->observed != kPoison)
        fail("pre-aborted stale flag", "accepted a stale completion", host, stream);
    passed("pre-aborted stale flag", start);

    watchdog.set_case("legacy three-argument wait");
    start = Clock::now();
    watchdog.reset_bell();
    constexpr uint32_t legacy_payload = 0x87dea123u;
    // Plain launches enqueue during the sleep (an early read sees poison); serialized launches enqueue the reader after the wait.
    std::thread publisher([host] {
        std::this_thread::sleep_for(100ms);
        publish(*host, legacy_payload);
    });
    bk::doorbell_wait(&device->flag, seq, stream);
    read_payload<<<1, 1, 0, stream>>>(device);
    check(cudaGetLastError(), "launch legacy payload read", host, stream);
    const bool legacy_waiting = cudaStreamQuery(stream) == cudaErrorNotReady;
    drain(*host, stream, "legacy three-argument wait");
    publisher.join();
    if (host->observed != legacy_payload)
        fail("legacy three-argument wait", "sequence did not order the payload read", host, stream);
    std::printf("  legacy wait: immediate not-ready observed: %s (diagnostic only)\n", legacy_waiting ? "yes" : "no");
    passed("legacy three-argument wait", start);

    watchdog.set_case("teardown");
    check(cudaStreamDestroy(stream), "destroy stream");
    check(cudaFree(seq), "free sequence");
    watchdog.set_bell(nullptr);
    check(cudaFreeHost(host), "free bell");
    std::printf("doorbell_abort_test: ok\n");
    return 0;
}
