// tests/kernels/dm_table_put_parity.cpp - the peer table update by kernel, against the copy it replaces.
//
//     build/dm_table_put_parity [--bench]
//
// The peer card's blob-address table (one entry per layer and expert) changes between verify windows as its swaps
// land.  dec_moe_table_put applies a list of changed entries with a kernel, the list riding in the launch's
// arguments; dec_moe_table_load launches both kernels with nothing to do, so their code is loaded up front.  For
// change lists of every size the decode loop makes (one entry, a round, more than one launch's worth) at the model's
// table size, the table it leaves must equal, bit for bit, the one the span copy before it leaves and a CPU
// reference.  dec_moe_zero_words must zero exactly the words a memset zeroes and no others.
//
// --bench: each update timed alone and with a burst of expert-sized host-to-device copies queued ahead of it on
// another stream of the same card.  The span copy goes through the card's copy queue and finishes only after the
// burst; the kernels do not wait for it.
#include "basalt/kernels/dec_moe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kLayers = 48, kExperts = 512;            // the model's table: layers x routed experts
constexpr size_t kLE = (size_t) kLayers * kExperts;
constexpr size_t kBlob = 2764816;                        // one expert's bytes in the pack the peer holds

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "dm_table_put_parity: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// The dm_sync path: changed entries in ascending order, applied in launches of at most kDmTablePutMax.
bool put_list(unsigned long long* d_tab, const std::vector<uint32_t>& idx, const std::vector<unsigned long long>& host,
              cudaStream_t s) {
    basalt::kernels::DmTablePut tp;
    tp.tab = d_tab;
    for (size_t k = 0; k < idx.size(); k += (size_t) basalt::kernels::kDmTablePutMax) {
        tp.n = (int) std::min(idx.size() - k, (size_t) basalt::kernels::kDmTablePutMax);
        for (int j = 0; j < tp.n; ++j) {
            tp.idx[j] = idx[k + (size_t) j];
            tp.val[j] = host[idx[k + (size_t) j]];
        }
        if (!basalt::kernels::dec_moe_table_put(tp, s)) return false;
    }
    return true;
}

// The copy it replaces: the span from the first to the last changed entry.
bool put_span(unsigned long long* d_tab, const std::vector<uint32_t>& idx, const unsigned long long* pinned,
              cudaStream_t s) {
    if (idx.empty()) return true;
    const size_t a = idx.front(), b = (size_t) idx.back() + 1;
    return ck(cudaMemcpyAsync(d_tab + a, pinned + a, (b - a) * 8, cudaMemcpyHostToDevice, s), "span copy");
}

// Entries of the device table that differ from `ref` (reported unless `quiet`).
size_t differ(const unsigned long long* d_tab, const std::vector<unsigned long long>& ref, const char* what, size_t n,
              bool quiet = false) {
    std::vector<unsigned long long> got(kLE);
    if (!ck(cudaMemcpy(got.data(), d_tab, kLE * 8, cudaMemcpyDeviceToHost), "read back")) return kLE;
    size_t bad = 0, first = kLE;
    for (size_t i = 0; i < kLE; ++i)
        if (got[i] != ref[i]) { if (first == kLE) first = i; ++bad; }
    if (bad != 0 && !quiet)
        std::fprintf(stderr, "dm_table_put_parity: %s, %zu changes: %zu entries differ (first %zu: %016llx vs "
                     "%016llx)\n", what, n, bad, first, got[first], ref[first]);
    return bad;
}

double us_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        cudaGetLastError();
        std::printf("dm_table_put_parity: no CUDA device, skipped\n");
        return 77;
    }
    std::mt19937_64 rng(20261024);
    cudaStream_t s = nullptr, r = nullptr;
    unsigned long long *d_put = nullptr, *d_copy = nullptr, *pinned = nullptr;
    unsigned int* d_zero = nullptr;
    if (!ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream") ||
        !ck(cudaStreamCreateWithFlags(&r, cudaStreamNonBlocking), "stream") ||
        !ck(cudaMalloc((void**) &d_put, kLE * 8), "table") || !ck(cudaMalloc((void**) &d_copy, kLE * 8), "table") ||
        !ck(cudaMalloc((void**) &d_zero, 256 * 4), "words") ||
        !ck(cudaHostAlloc((void**) &pinned, kLE * 8, cudaHostAllocPortable), "pinned"))
        return 1;
    // the engine loads both kernels up front (dm_init); here that launch must leave everything as it was
    if (!basalt::kernels::dec_moe_table_load(s) || !ck(cudaStreamSynchronize(s), "load")) return 1;
    // a table as the peer holds it: slot addresses or 0 for an expert it does not hold
    std::vector<unsigned long long> ref(kLE);
    for (auto& v : ref) v = (rng() & 1) ? (rng() & ~0xffull) : 0ull;
    std::memcpy(pinned, ref.data(), kLE * 8);
    if (!ck(cudaMemcpy(d_put, ref.data(), kLE * 8, cudaMemcpyHostToDevice), "init") ||
        !ck(cudaMemcpy(d_copy, ref.data(), kLE * 8, cudaMemcpyHostToDevice), "init"))
        return 1;
    int fails = 0;
    const int sizes[] = {1, 2, 7, 96, 191, 320, 321, 640, 960, 1283};
    for (int rep = 0; rep < 3; ++rep)
        for (int n : sizes) {
            // n distinct entries anywhere in the table, ascending (dm_sync's order), each a new value
            std::vector<uint32_t> idx;
            std::vector<uint8_t> hit(kLE, 0);
            while ((int) idx.size() < n) {
                const uint32_t i = (uint32_t) (rng() % kLE);
                if (!hit[i]) { hit[i] = 1; idx.push_back(i); }
            }
            std::sort(idx.begin(), idx.end());
            for (uint32_t i : idx) ref[i] = (rng() & 3) ? (rng() & ~0xffull) : 0ull;
            std::memcpy(pinned, ref.data(), kLE * 8);
            if (!put_list(d_put, idx, ref, s) || !put_span(d_copy, idx, pinned, s) ||
                !ck(cudaStreamSynchronize(s), "sync"))
                return 1;
            fails += differ(d_put, ref, "table put", (size_t) n) != 0;
            fails += differ(d_copy, ref, "span copy", (size_t) n) != 0;
        }
    // the check has power: a table one entry off is caught
    {
        std::vector<unsigned long long> off = ref;
        off[kLE / 3] ^= 0x100ull;
        if (differ(d_put, off, "negative control", 1, true) != 1) {
            std::fprintf(stderr, "dm_table_put_parity: the comparison missed a changed entry\n");
            ++fails;
        }
    }
    // zero words: exactly the first n of them
    for (int n : {0, 1, 5, 16, 256}) {
        std::vector<unsigned int> w(256);
        if (!ck(cudaMemset(d_zero, 0xff, 256 * 4), "memset") ||
            !basalt::kernels::dec_moe_zero_words(d_zero, n, s) || !ck(cudaStreamSynchronize(s), "sync") ||
            !ck(cudaMemcpy(w.data(), d_zero, 256 * 4, cudaMemcpyDeviceToHost), "read back"))
            return 1;
        for (int i = 0; i < 256; ++i)
            if (w[(size_t) i] != (i < n ? 0u : 0xffffffffu)) {
                std::fprintf(stderr, "dm_table_put_parity: zero_words(%d): word %d is %08x\n", n, i, w[(size_t) i]);
                ++fails;
                break;
            }
    }
    if (basalt::kernels::dec_moe_zero_words(d_zero, 257, s)) {
        std::fprintf(stderr, "dm_table_put_parity: zero_words took more words than it can zero\n");
        ++fails;
    }
    std::printf("dm_table_put_parity: %d table updates (1 to %d entries) and 5 zero sizes, kernel vs span copy vs "
                "CPU: %s (a one-entry change is caught)\n",
                3 * (int) (sizeof sizes / sizeof sizes[0]), sizes[sizeof sizes / sizeof sizes[0] - 1],
                fails == 0 ? "IDENTICAL" : "DIFFERENT");
    if (bench && fails == 0) {
        // a refill burst on stream r, then each update on stream s; the time from queueing the update to its end
        const int kCopies = 40;
        uint8_t *h_src = nullptr, *d_dst = nullptr;
        if (!ck(cudaHostAlloc((void**) &h_src, kBlob * kCopies, cudaHostAllocPortable), "burst source") ||
            !ck(cudaMalloc((void**) &d_dst, kBlob * kCopies), "burst slots"))
            return 1;
        std::memset(h_src, 1, kBlob * kCopies);
        cudaEvent_t ev_r, ev_s;
        if (!ck(cudaEventCreateWithFlags(&ev_r, cudaEventDisableTiming), "event") ||
            !ck(cudaEventCreateWithFlags(&ev_s, cudaEventDisableTiming), "event"))
            return 1;
        std::vector<uint32_t> idx(96);
        for (size_t k = 0; k < idx.size(); ++k) idx[k] = (uint32_t) (k * (kLE / idx.size()) + 7);   // a round's span
        const char* names[4] = {"span copy (a round, ~190 KB)", "table put kernel (96 entries)", "memset (64 B)",
                                "zero kernel (16 words)"};
        for (int burst = 0; burst < 2; ++burst)
            for (int m = 0; m < 4; ++m) {
                double best = 1e30, burst_us = 0;
                for (int it = 0; it < 5; ++it) {
                    if (!ck(cudaDeviceSynchronize(), "sync")) return 1;
                    if (burst)
                        for (int c = 0; c < kCopies; ++c)
                            if (!ck(cudaMemcpyAsync(d_dst + (size_t) c * kBlob, h_src + (size_t) c * kBlob, kBlob,
                                                    cudaMemcpyHostToDevice, r), "burst copy"))
                                return 1;
                    cudaEventRecord(ev_r, r);
                    const auto t0 = std::chrono::steady_clock::now();
                    bool ok = true;
                    if (m == 0) ok = put_span(d_copy, idx, pinned, s);
                    else if (m == 1) ok = put_list(d_put, idx, ref, s);
                    else if (m == 2) ok = ck(cudaMemsetAsync(d_zero, 0, 64, s), "memset");
                    else ok = basalt::kernels::dec_moe_zero_words(d_zero, 16, s);
                    cudaEventRecord(ev_s, s);
                    if (!ok) return 1;
                    while (cudaEventQuery(ev_s) == cudaErrorNotReady) {
                    }
                    const double t_s = us_since(t0);
                    while (cudaEventQuery(ev_r) == cudaErrorNotReady) {
                    }
                    best = std::min(best, t_s);
                    burst_us = us_since(t0);
                }
                std::printf("bench: %-30s %s: done %8.1f us after queueing (best of 5)%s\n", names[m],
                            burst ? "behind a burst" : "alone         ", best,
                            burst ? (" | the burst (" + std::to_string(kCopies) + " expert copies) ends after " +
                                     std::to_string((int) burst_us) + " us").c_str() : "");
            }
        cudaFree(d_dst);
        cudaFreeHost(h_src);
    }
    cudaFree(d_put);
    cudaFree(d_copy);
    cudaFree(d_zero);
    cudaFreeHost(pinned);
    return fails == 0 ? 0 : 1;
}
