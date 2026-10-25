// src/kernels/pair_snapshot_parity.cpp - the kernels behind the overlapped windows' snapshot and hand-off.
//
//     build/pair_snapshot_parity [--bench]
//
// word_copy_list copies up to kWordCopyMax disjoint word ranges in one launch (the snapshot of a stage's indexer rows and
// block positions, and their restore).  For lists of every length, with ranges of one word (a block position) and of the
// indexer's pooled rows (kRows x the model's key dimension, the snapshot's unit) at random offsets, it must write exactly
// those words, bit for bit as a CPU copy does, and leave every other word alone.  signal_flag raises a mapped host flag
// behind everything queued on its stream: the host that sees the flag must see every word the stream wrote before it, and
// a stream waiting on another flag must not raise it early.  Inputs are random.
//
// --bench: the list copy of a full snapshot (32 ranges of the indexer's rows) against one copy call per range, in
// microseconds.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "basalt/kernels/verify_kernels.hpp"

namespace {

using namespace basalt::kernels;

constexpr int kKeyDim = 128;       // the model's indexer key dimension
constexpr int kRows = 4;           // pooled rows a window's commit can complete
constexpr int kRowWords = kRows * kKeyDim;

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "pair_snapshot_parity: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

double us_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count();
}

// Lists of 1..kWordCopyMax ranges, each of one word or kRowWords, placed at random increasing offsets with gaps.
bool test_list(bool bench) {
    std::mt19937 rng(7);
    constexpr size_t kWords = 1 << 20;
    int32_t *d_src = nullptr, *d_dst = nullptr;
    if (!ck(cudaMalloc((void**) &d_src, kWords * 4), "malloc src") || !ck(cudaMalloc((void**) &d_dst, kWords * 4), "malloc dst"))
        return false;
    std::vector<int32_t> src(kWords), dst(kWords), ref;
    for (int32_t& v : src) v = (int32_t) rng();
    if (!ck(cudaMemcpy(d_src, src.data(), kWords * 4, cudaMemcpyHostToDevice), "upload src")) return false;
    size_t bad = 0, trials = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const int n = 1 + (int) (rng() % (unsigned) kWordCopyMax);
        for (int32_t& v : dst) v = (int32_t) rng();
        ref = dst;
        WordCopyList list;
        size_t at = rng() % 1000;
        for (int i = 0; i < n; ++i) {
            const int words = (rng() & 1u) ? kRowWords : 1;
            WordCopy& e = list.e[list.n++];
            const size_t from = at + rng() % 50;
            e.src = d_src + from;
            e.dst = d_dst + from + (rng() & 1u ? 0 : 3);   // the destination offset need not equal the source's
            e.n = words;
            std::memcpy(ref.data() + (e.dst - d_dst), src.data() + from, (size_t) words * 4);
            at = (size_t) (e.dst - d_dst) + (size_t) words + 1 + rng() % 40;
            if (at > kWords - 2 * kRowWords) { list.n = i + 1; break; }
        }
        if (!ck(cudaMemcpy(d_dst, dst.data(), kWords * 4, cudaMemcpyHostToDevice), "upload dst")) return false;
        word_copy_list(list, nullptr);
        if (!ck(cudaDeviceSynchronize(), "word_copy_list")) return false;
        if (!ck(cudaMemcpy(dst.data(), d_dst, kWords * 4, cudaMemcpyDeviceToHost), "read back")) return false;
        ++trials;
        if (std::memcmp(dst.data(), ref.data(), kWords * 4) != 0) ++bad;
    }
    std::printf("word_copy_list: %zu lists (1..%d ranges of 1 or %d words): %s\n", trials, kWordCopyMax, kRowWords,
                bad == 0 ? "BITWISE" : "DIFFERS");
    if (bad != 0) return false;
    if (bench) {
        WordCopyList list;
        for (int i = 0; i < kWordCopyMax; ++i) {
            WordCopy& e = list.e[list.n++];
            e.src = d_src + (size_t) i * 1024;
            e.dst = d_dst + (size_t) i * 1024;
            e.n = kRowWords;
        }
        auto time_it = [&](auto&& fn) {
            for (int w = 0; w < 20; ++w) fn();
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 200; ++r) fn();
            cudaDeviceSynchronize();
            return us_since(t0) / 200.0;
        };
        const double one = time_it([&] { word_copy_list(list, nullptr); });
        const double many = time_it([&] {
            for (int i = 0; i < list.n; ++i)
                cudaMemcpyAsync(list.e[i].dst, list.e[i].src, (size_t) list.e[i].n * 4, cudaMemcpyDeviceToDevice, nullptr);
        });
        std::printf("bench: %d ranges of %d words: one launch %.1f us, one copy per range %.1f us\n", list.n, kRowWords, one,
                    many);
    }
    cudaFree(d_src);
    cudaFree(d_dst);
    return true;
}

// signal_flag: ordered behind the stream's writes to mapped memory, and not before the stream's own wait.
bool test_signal() {
    uint32_t *h_flag = nullptr, *d_flag = nullptr, *h_gate = nullptr, *d_gate = nullptr;
    int32_t *h_data = nullptr, *d_data = nullptr, *d_src = nullptr;
    if (!ck(cudaHostAlloc((void**) &h_flag, 64, cudaHostAllocMapped | cudaHostAllocPortable), "alloc flag") ||
        !ck(cudaHostAlloc((void**) &h_gate, 64, cudaHostAllocMapped | cudaHostAllocPortable), "alloc gate") ||
        !ck(cudaHostAlloc((void**) &h_data, kRowWords * 4, cudaHostAllocMapped | cudaHostAllocPortable), "alloc data") ||
        !ck(cudaHostGetDevicePointer((void**) &d_flag, h_flag, 0), "flag pointer") ||
        !ck(cudaHostGetDevicePointer((void**) &d_gate, h_gate, 0), "gate pointer") ||
        !ck(cudaHostGetDevicePointer((void**) &d_data, h_data, 0), "data pointer") ||
        !ck(cudaMalloc((void**) &d_src, kRowWords * 4), "malloc src"))
        return false;
    cudaStream_t s = nullptr;
    if (!ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream")) return false;
    std::mt19937 rng(11);
    size_t bad = 0;
    const int rounds = 100;
    for (int r = 1; r <= rounds; ++r) {
        std::vector<int32_t> pat(kRowWords);
        for (int32_t& v : pat) v = (int32_t) rng();
        if (!ck(cudaMemcpy(d_src, pat.data(), kRowWords * 4, cudaMemcpyHostToDevice), "upload pattern")) return false;
        std::memset(h_data, 0, kRowWords * 4);
        *(volatile uint32_t*) h_gate = 0;
        // the stream waits for the gate, writes the pattern into host memory, then raises the flag
        wait_flag_ge(d_gate, 1, s);
        WordCopyList l;
        l.e[0] = WordCopy{d_src, d_data, kRowWords};
        l.n = 1;
        word_copy_list(l, s);
        signal_flag(d_flag, (uint32_t) r, s);
        std::this_thread::sleep_for(std::chrono::microseconds(300));
        if (*(volatile uint32_t*) h_flag >= (uint32_t) r) { ++bad; std::fprintf(stderr, "flag raised before the gate\n"); }
        *(volatile uint32_t*) h_gate = 1;
        const auto t0 = std::chrono::steady_clock::now();
        while (*(volatile uint32_t*) h_flag < (uint32_t) r && us_since(t0) < 5e6) std::this_thread::yield();
        if (*(volatile uint32_t*) h_flag < (uint32_t) r) { ++bad; std::fprintf(stderr, "flag never raised\n"); break; }
        if (std::memcmp(h_data, pat.data(), kRowWords * 4) != 0) { ++bad; std::fprintf(stderr, "data behind the flag differs\n"); }
        if (!ck(cudaStreamSynchronize(s), "sync")) return false;
    }
    std::printf("signal_flag: %d rounds (gated stream writes %d words to host memory, then raises the flag): %s\n", rounds,
                kRowWords, bad == 0 ? "ORDERED" : "FAILED");
    cudaStreamDestroy(s);
    cudaFreeHost(h_flag);
    cudaFreeHost(h_gate);
    cudaFreeHost(h_data);
    cudaFree(d_src);
    return bad == 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i) bench = bench || std::strcmp(argv[i], "--bench") == 0;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        std::printf("pair_snapshot_parity: no GPU\n");
        return 77;
    }
    const bool ok = test_list(bench) && test_signal();
    std::printf(ok ? "pair_snapshot_parity: PASS\n" : "pair_snapshot_parity: FAIL\n");
    return ok ? 0 : 1;
}
