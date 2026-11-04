// tests/kernels/ds41_peer_parity.cpp - a second card's share of a layer's experts against the main card's grouped kernels,
// bit for bit on the real geometry, whatever the share's size and the rows it lands in.
//
//     build/ds41_peer_parity            the checks (passes with a note on one card)
//     build/ds41_peer_parity --bench    the checks, then a microbench line per format and share size
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/core/peer_experts.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace kg = basalt::kernels;

namespace {

constexpr int H = 5120, FF = 2304, NB = 8;
constexpr float kLimit = 10.0f;
int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
        ++g_fail;
    }
}

uint16_t f16_bits(float scale, std::mt19937& rng) {
    std::uniform_int_distribution<int> man(0, 1023);
    const int e = 15 + (int) std::floor(std::log2(scale));
    return (uint16_t) ((e << 10) | man(rng));
}

// rows of n values: random bytes with a finite fp16 scale in each block
std::vector<uint8_t> random_rows(int type, int64_t rows, int64_t n, float scale, std::mt19937& rng) {
    const size_t rb = ggml_row_size((ggml_type) type, n), bs = ggml_type_size((ggml_type) type);
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = f16_bits(scale, rng);
        if (type == GGML_TYPE_IQ1_M) {   // the fp16 scale sits in the top nibbles of the block's last four int16
            for (int i = 0; i < 4; ++i) {
                uint16_t sc;
                std::memcpy(&sc, &w[o + bs - 8 + 2 * (size_t) i], 2);
                sc = (uint16_t) ((sc & 0x0FFF) | (((h >> (4 * i)) & 0xF) << 12));
                std::memcpy(&w[o + bs - 8 + 2 * (size_t) i], &sc, 2);
            }
            continue;
        }
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}

template <class T>
T* dalloc(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return (T*) p;
}

const char* nm(int t) { return t == GGML_TYPE_IQ2_XXS ? "IQ2_XXS" : t == GGML_TYPE_IQ2_XS ? "IQ2_XS" : t == GGML_TYPE_IQ1_M ? "IQ1_M" : "?"; }

void one_pair(int gu, int d, bool bench, const float* h_x, float* h_rows, cudaStream_t s) {
    std::mt19937 rng(700 + gu * 7 + d);
    const kg::NativeExpertLayout L = kg::native_expert_layout(gu, d, H, FF);
    std::vector<std::vector<uint8_t>> blob;
    for (int b = 0; b < NB; ++b) {
        auto g = random_rows(gu, FF, H, 0.002f, rng), u = random_rows(gu, FF, H, 0.002f, rng), dn = random_rows(d, H, FF, 0.0005f, rng);
        g.insert(g.end(), u.begin(), u.end());
        g.insert(g.end(), dn.begin(), dn.end());
        check(g.size() == L.bytes, "layout bytes");
        blob.push_back(std::move(g));
    }

    // the main card: every expert over the activation in h_x
    const size_t slot = (L.bytes + 255) & ~(size_t) 255;
    auto* blobs = dalloc<uint8_t>(slot * NB);
    for (int b = 0; b < NB; ++b) ck(cudaMemcpy(blobs + slot * b, blob[(size_t) b].data(), L.bytes, cudaMemcpyHostToDevice), "blob");
    auto* dx = dalloc<float>(H);
    ck(cudaMemcpy(dx, h_x, H * sizeof(float), cudaMemcpyHostToDevice), "x");
    auto* xq = dalloc<uint8_t>(kg::native_q8_1_bytes(H, 1));
    kg::quantize_q8_1_rows(dx, 1, H, xq, s);
    std::vector<unsigned long long> gptr;
    std::vector<int32_t> start, dst, tk;
    for (int b = 0; b < NB; ++b) {
        gptr.push_back((unsigned long long) (blobs + slot * b));
        start.push_back(b);
        dst.push_back(b);
        tk.push_back(0);
    }
    start.push_back(NB);
    auto* dptr = dalloc<unsigned long long>(NB);
    auto* dstart = dalloc<int32_t>(NB + 1);
    auto* dng = dalloc<int32_t>(1);
    auto* ddst = dalloc<int32_t>(NB);
    auto* dtok = dalloc<int32_t>(NB);
    const int32_t ng = NB;
    ck(cudaMemcpy(dptr, gptr.data(), NB * 8, cudaMemcpyHostToDevice), "ptr");
    ck(cudaMemcpy(dstart, start.data(), (NB + 1) * 4, cudaMemcpyHostToDevice), "start");
    ck(cudaMemcpy(dng, &ng, 4, cudaMemcpyHostToDevice), "ng");
    ck(cudaMemcpy(ddst, dst.data(), NB * 4, cudaMemcpyHostToDevice), "dst");
    ck(cudaMemcpy(dtok, tk.data(), NB * 4, cudaMemcpyHostToDevice), "tok");
    auto* scr = dalloc<uint8_t>(kg::native_expert_scratch_bytes(NB, FF));
    auto* dout = dalloc<float>((size_t) NB * H);
    kg::native_expert_grouped_clamped(L, kLimit, dptr, dstart, dng, ddst, dtok, NB, NB, xq, scr, dout, s);
    ck(cudaStreamSynchronize(s), "main");
    std::vector<float> main_rows((size_t) NB * H);
    ck(cudaMemcpy(main_rows.data(), dout, main_rows.size() * sizeof(float), cudaMemcpyDeviceToHost), "rows");

    // the second card: shares of every size, landing in rows other than their order
    basalt::core::PeerExperts peer;
    std::string err;
    if (!peer.open_plain(1, 1, NB, H, FF, h_rows, err)) {
        std::printf("FAIL: open: %s\n", err.c_str());
        ++g_fail;
        return;
    }
    if (!peer.open_slots(std::vector<int64_t>(NB, (int64_t) L.bytes), err)) {
        std::printf("FAIL: slots: %s\n", err.c_str());
        ++g_fail;
        return;
    }
    uint8_t* stage = nullptr;
    ck(cudaHostAlloc((void**) &stage, L.bytes, cudaHostAllocPortable), "stage");
    for (int b = 0; b < NB; ++b) {
        peer.admit(0, b, b);
        std::memcpy(stage, blob[(size_t) b].data(), L.bytes);
        check(peer.fill_slot(b, stage, L.bytes), "fill");
    }
    kg::NativeExpertLayout Lp = L;
    Lp.swiglu_limit = kLimit;
    bool same = true;
    for (int n = 1; n <= 6; ++n) {
        int32_t ids[8] = {}, kind[8] = {};
        for (int i = 0; i < n; ++i) {
            ids[n - 1 - i] = (i * 3) % NB;   // landing in rows other than their order
            kind[n - 1 - i] = 2;
        }
        std::memset(h_rows, 0, (size_t) n * H * sizeof(float));
        if (!peer.launch_plain(0, Lp, h_x, ids, n, kind, err) || !peer.finish(nullptr, err)) {
            std::printf("FAIL: share: %s\n", err.c_str());
            ++g_fail;
            return;
        }
        for (int i = 0; i < n; ++i)
            same = same && !std::memcmp(h_rows + (size_t) (n - 1 - i) * H, main_rows.data() + (size_t) ((i * 3) % NB) * H, (size_t) H * sizeof(float));
    }
    check(same, "the second card's rows differ from the main card's");
    std::printf("%-7s/%-7s second card vs main card, shares of 1..6: %s\n", nm(gu), nm(d), same ? "bitwise same" : "DIFFER");

    if (bench) {
        for (int n : {1, 2, 3, 4, 6}) {
            int32_t ids[8] = {}, kind[8] = {};
            for (int i = 0; i < n; ++i) { ids[i] = i; kind[i] = 2; }
            std::vector<double> us;
            for (int it = 0; it < 201; ++it) {
                const auto t0 = std::chrono::steady_clock::now();
                peer.launch_plain(0, Lp, h_x, ids, n, kind, err);
                peer.finish(nullptr, err);
                if (it > 0) us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(us.begin(), us.end());
            const double med = us[us.size() / 2];
            std::printf("bench second card %s/%s share of %d, launch to rows in host memory: %.1f us/call (p90 %.1f), %.1f MB read, %.0f GB/s (weights only)\n",
                        nm(gu), nm(d), n, med, us[us.size() * 9 / 10], (double) n * (double) L.bytes / 1e6, (double) n * (double) L.bytes / (med * 1e3));
        }
    }
    cudaFreeHost(stage);
    for (void* p : {(void*) blobs, (void*) dx, (void*) xq, (void*) dptr, (void*) dstart, (void*) dng, (void*) ddst, (void*) dtok, (void*) scr, (void*) dout})
        cudaFree(p);
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && !std::strcmp(argv[1], "--bench");
    int n = 0;
    ck(cudaGetDeviceCount(&n), "devices");
    if (n < 2) {
        std::printf("ds41_peer_parity: one card, nothing to compare\n");
        return 0;
    }
    float* h_x = nullptr;
    float* h_rows = nullptr;
    ck(cudaHostAlloc((void**) &h_x, H * sizeof(float), cudaHostAllocPortable), "x");
    ck(cudaHostAlloc((void**) &h_rows, 8 * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "rows");
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < H; ++i) h_x[i] = nd(rng);
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    // the format pairs a GSQ-RCO pack mixes
    const int pairs[][2] = {{GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_M},   {GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS}, {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_M},
                            {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS}, {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS}, {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ1_M},
                            {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS},  {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS}};
    for (const auto& p : pairs) one_pair(p[0], p[1], bench, h_x, h_rows, s);
    cudaStreamDestroy(s);
    cudaFreeHost(h_x);
    cudaFreeHost(h_rows);
    std::printf(g_fail ? "ds41_peer_parity: FAILED\n" : "ds41_peer_parity: ok\n");
    return g_fail ? 1 : 0;
}
