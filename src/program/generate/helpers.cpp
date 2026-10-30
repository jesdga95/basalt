#include "helpers.hpp"

namespace basalt::program::generate {

// A byte count in GiB with one decimal, rounded down (a safe limit must not round up past itself).
std::string gib_text(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f", std::floor((double) bytes / 1073741824.0 * 10.0) / 10.0);
    return buf;
}

// A start-ending failure: one `basalt error:` line, which the server shows as the reason.  Returns the exit code.
int fatal_start_error(const std::string& text) {
    basalt::log::error("basalt", "%s", text.c_str());
    return 1;
}
// " (N MiB of M MiB VRAM free on this GPU)" for an allocation's failure message (clears the error first)
std::string vram_free_note() {
    (void) cudaGetLastError();
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void) cudaGetLastError();
        return {};
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, " (%llu MiB of %llu MiB VRAM free on this GPU)", (unsigned long long) (free_b >> 20),
                  (unsigned long long) (total_b >> 20));
    return buf;
}

/// Calls `f` with the tensor name of every row of a pack index (the first field of each non-comment line).
bool for_each_index_name(const basalt::core::FileRange& index, const std::function<void(const std::string&)>& f) {
    std::vector<uint8_t> raw;
    std::string err;
    if (!basalt::core::read_range(index, raw, err)) return false;
    const std::string text(raw.begin(), raw.end());
    size_t at = 0;
    while (at < text.size()) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(at, nl - at);
        at = nl + 1;
        char name[256];
        if (line.empty() || line[0] == '#' || std::sscanf(line.c_str(), "%255s", name) != 1) continue;
        f(name);
    }
    return true;
}


int argmax(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[best]) best = (int) i;
    return best;
}

// The effective host->device bandwidth of the PCIe link: copies from pinned host memory, as the expert arena's
// reads are.  The native default share (0.55) suits a x16 link; a x8 card in a x8 slot carries about half of that.
// Returns < 0 when the probe cannot run (then the caller keeps the default).
//
// A single timed burst can read a link that is still in a low-power state, or sharing DMA, far too low, and the low
// reading would set the share.  Such a disturbance only ever slows a copy: nothing makes a correctly timed copy
// faster than the link carries.  So 1 GiB is copied as four bursts of 256 MiB, timed one by one, and the fastest is
// the link's figure (the median would still follow a disturbance that lasts through half the bursts).  `samples`,
// when given, gets every burst's reading for the log.
double probe_pcie_h2d_gbps(std::string* samples) {
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

// Host->device bandwidth with every card in `devs` copying at once from its own pinned buffer, in GB/s (the best of
// two rounds); -1 when a buffer or a copy fails.  Cards that share a switch, a root port or the RAM read below the sum
// of their own links.
double probe_h2d_together_gbps(const std::vector<int>& devs) {
    constexpr size_t kBytes = 256ull << 20;
    constexpr int kBursts = 4;
    struct Card { int dev; void* h = nullptr; void* d = nullptr; cudaStream_t s = nullptr; };
    std::vector<Card> cards;
    bool ok = !devs.empty();
    for (int dev : devs) {
        if (!ok) break;
        const basalt::core::OnDevice on(dev);
        Card c{dev};
        ok = cudaHostAlloc(&c.h, kBytes, cudaHostAllocPortable) == cudaSuccess && cudaMalloc(&c.d, kBytes) == cudaSuccess &&
             cudaStreamCreateWithFlags(&c.s, cudaStreamNonBlocking) == cudaSuccess;
        if (c.h != nullptr) std::memset(c.h, 0, kBytes);   // fault the pages in before timing
        ok = ok && cudaMemcpyAsync(c.d, c.h, kBytes, cudaMemcpyHostToDevice, c.s) == cudaSuccess &&   // warmup
             cudaStreamSynchronize(c.s) == cudaSuccess;
        cards.push_back(c);
    }
    double bw = -1.0;
    for (int round = 0; ok && round < 2; ++round) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int b = 0; ok && b < kBursts; ++b)
            for (const Card& c : cards) {
                const basalt::core::OnDevice on(c.dev);
                ok = ok && cudaMemcpyAsync(c.d, c.h, kBytes, cudaMemcpyHostToDevice, c.s) == cudaSuccess;
            }
        for (const Card& c : cards) {
            const basalt::core::OnDevice on(c.dev);
            ok = cudaStreamSynchronize(c.s) == cudaSuccess && ok;
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (ok && sec > 0) bw = std::max(bw, (double) kBytes * kBursts * (double) cards.size() / sec / 1e9);
    }
    for (const Card& c : cards) {
        const basalt::core::OnDevice on(c.dev);
        if (c.s != nullptr) cudaStreamDestroy(c.s);
        if (c.d != nullptr) cudaFree(c.d);
        if (c.h != nullptr) cudaFreeHost(c.h);
    }
    if (!ok) {
        (void) cudaGetLastError();
        return -1.0;
    }
    return bw;
}

// The PCIe share of the missed experts for a link measured at `gbps`: `base` (the share for a x16 link) from 20 GB/s
// up, and below that in proportion to the bandwidth, so the time the link spends on its share stays about what the
// x16 share costs.  Continuous, so a reading near either edge does not move the share by a large part of its range.
double pcie_frac_for_gbps(double gbps, double base) {
    return gbps <= 0.0 ? base : base * std::min(1.0, gbps / 20.0);
}


}  // namespace
