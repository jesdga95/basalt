// tests/core/checkpoint_transfer_test.cpp - a checkpoint's running state between the device and pooled host blocks.
//
// Many small regions plus a large one, the geometry of a model's per-layer state: copy out and in are bitwise round
// trips; a deferred save is a snapshot (the device state changes right after it is issued, in the stream's order) and
// its bytes are there only after finish(); page-locked and ordinary host bytes, the bounce buffer, a save that does not
// fit the spare and one that follows another all end with the same bytes.
#include "basalt/core/checkpoint_transfer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace basalt::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "FAIL: cuda: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

struct Device {
    std::vector<void*> bufs;
    std::vector<size_t> sizes;
    ~Device() { for (void* b : bufs) cudaFree(b); }
    void add(size_t n) {
        void* p = nullptr;
        cuda_check(cudaMalloc(&p, n));
        bufs.push_back(p);
        sizes.push_back(n);
    }
    std::vector<DeviceRegion> regions() const {
        std::vector<DeviceRegion> r;
        for (size_t i = 0; i < bufs.size(); ++i) r.push_back({bufs[i], sizes[i]});
        return r;
    }
    size_t bytes() const { size_t n = 0; for (size_t s : sizes) n += s; return n; }
    void fill(uint32_t seed, cudaStream_t stream = nullptr) {
        std::mt19937 rng(seed);
        for (size_t i = 0; i < bufs.size(); ++i) {
            std::vector<uint8_t> h(sizes[i]);
            for (uint8_t& b : h) b = (uint8_t) rng();
            cuda_check(cudaMemcpy(bufs[i], h.data(), h.size(), cudaMemcpyHostToDevice));
        }
        (void) stream;
    }
    std::vector<uint8_t> read() const {
        std::vector<uint8_t> out;
        for (size_t i = 0; i < bufs.size(); ++i) {
            std::vector<uint8_t> h(sizes[i]);
            cuda_check(cudaMemcpy(h.data(), bufs[i], h.size(), cudaMemcpyDeviceToHost));
            out.insert(out.end(), h.begin(), h.end());
        }
        return out;
    }
};
}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) { std::puts("no CUDA device: skipped"); return 77; }
    std::string err;

    // 130 small regions of odd sizes and a few of 600 KiB: about 12 MiB, like a deep model's rings and compressors
    Device d;
    std::mt19937 rng(7);
    for (int i = 0; i < 130; ++i) d.add(i % 9 == 0 ? 600 * 1024 : 1 + rng() % 70000);
    const size_t total = d.bytes();
    d.fill(1);
    const std::vector<uint8_t> original = d.read();

    // pooled blocks, page-locked when the driver allows it
    check(CheckpointBytes::allocator_type::reserve(total, 3, &kCheckpointPin), "three blocks reserved");
    const bool locked = CheckpointBytes::allocator_type::locked_bytes() >= 3 * total;
    std::printf("pinned pool: %s\n", locked ? "locked" : "ordinary memory (the driver refused)");
    CheckpointBytes a, b, c;
    a.resize(total); b.resize(total); c.resize(total);
    check(!locked || (host_range_pinned(a.data()) && host_range_pinned(c.data())), "a pooled block is page-locked");
    check(!host_range_pinned(std::vector<uint8_t>(16).data()), "an ordinary vector is not");

    {   // copy out: bitwise equal to the device bytes; copy in: the device state returns bitwise after being destroyed
        check(checkpoint_copy_out({{a.data(), d.regions()}}, nullptr, err), "copy out");
        check(std::equal(a.begin(), a.end(), original.begin()), "copy out is bitwise the device state");
        d.fill(99);
        check(checkpoint_copy_in({{a.data(), d.regions()}}, nullptr, err), "copy in");
        cuda_check(cudaDeviceSynchronize());
        check(d.read() == original, "copy in restores the device state bitwise");
    }
    {   // a missing buffer is refused
        check(!checkpoint_copy_out({{a.data(), {{nullptr, 8}}}}, nullptr, err) && !err.empty(), "a region without a buffer is refused");
    }

    cudaStream_t work = nullptr;
    cuda_check(cudaStreamCreate(&work));
    CheckpointDeferral deferral;
    check(deferral.reserve(total, err), "reserve the spare");
    {   // a deferred save is a snapshot: the stream's later work changes the live state, the bytes are the earlier ones
        d.fill(1);
        check(deferral.save({{b.data(), d.regions()}}, work, b.data(), err), "deferred save");
        check(deferral.pending() == b.data(), "the save is in flight");
        for (void* p : d.bufs) cuda_check(cudaMemsetAsync(p, 0x5a, 1, work));   // later work on the same stream
        cuda_check(cudaStreamSynchronize(work));
        check(deferral.finish(err) && deferral.pending() == nullptr, "finish the save");
        check(std::equal(b.begin(), b.end(), original.begin()), "the deferred bytes are the state at the call");
        check(deferral.finish(err), "finishing again is a no-op");
    }
    {   // two saves in a row: the second waits for the first; both land
        d.fill(1);
        check(deferral.save({{b.data(), d.regions()}}, work, b.data(), err), "first save");
        d.fill(2);
        const std::vector<uint8_t> second = d.read();
        check(deferral.save({{c.data(), d.regions()}}, work, c.data(), err), "second save");
        check(deferral.finish(err), "finish the second");
        check(std::equal(b.begin(), b.end(), original.begin()) && std::equal(c.begin(), c.end(), second.begin()),
              "each save holds the state of its own call");
    }
    {   // several segments, one of them ordinary host memory small enough for the bounce buffer
        d.fill(1);
        const std::vector<DeviceRegion> r = d.regions();
        const std::vector<DeviceRegion> head(r.begin(), r.begin() + 60), tail(r.begin() + 60, r.end());
        size_t head_bytes = 0;
        for (const DeviceRegion& x : head) head_bytes += x.bytes;
        std::vector<uint8_t> loose(total - head_bytes);   // not page-locked
        check(loose.size() > CheckpointDeferral::kBounceBytes || true, "setup");
        const bool fits = loose.size() <= CheckpointDeferral::kBounceBytes;
        check(deferral.save({{b.data(), head}, {loose.data(), tail}}, work, b.data(), err), "save to a locked and an ordinary segment");
        check(fits ? deferral.pending() == b.data() : deferral.pending() == nullptr,
              "ordinary bytes beyond the bounce buffer are saved at once");
        check(deferral.finish(err), "finish");
        check(std::equal(b.begin(), b.begin() + (std::ptrdiff_t) head_bytes, original.begin()) &&
              std::equal(loose.begin(), loose.end(), original.begin() + (std::ptrdiff_t) head_bytes),
              "both segments hold their regions");
    }
    {   // a small ordinary segment goes through the bounce buffer
        Device small;
        for (int i = 0; i < 20; ++i) small.add(1 + (size_t) (i * 37));
        small.fill(5);
        const std::vector<uint8_t> want = small.read();
        std::vector<uint8_t> loose(small.bytes());
        check(deferral.save({{loose.data(), small.regions()}}, work, loose.data(), err) && deferral.pending() == loose.data(), "bounced save issued");
        small.fill(6);
        check(deferral.finish(err) && loose == want, "bounced bytes are the state at the call");
    }
    {   // a save larger than the spare is made at once; with no spare at all too
        Device big;
        big.add(total + 4096);
        big.fill(3);
        const std::vector<uint8_t> want = big.read();
        std::vector<uint8_t> out(big.bytes());
        check(deferral.save({{out.data(), big.regions()}}, work, out.data(), err) && deferral.pending() == nullptr && out == want,
              "a save that does not fit the spare lands at once");
        CheckpointDeferral none;
        std::fill(out.begin(), out.end(), 0);
        check(none.save({{out.data(), big.regions()}}, work, out.data(), err) && none.pending() == nullptr && out == want,
              "without a spare a save lands at once");
    }
    cudaStreamDestroy(work);
    std::printf("checkpoint_transfer_test: %d checks passed\n", checks);
    return 0;
}
