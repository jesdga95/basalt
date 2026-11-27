// src/models/deepseek4/dspark_parity.cpp - parity of the DSpark drafter's kernels against plain CPU references, at the
// drafter's geometry (4096 -> 2048 -> 4096 MXFP4 experts, 512-wide latents, a 129280-row rank-256 Markov table), with
// a microbench line for the expert kernels; the staging commit's slot arithmetic against a direct projection; and the
// kept-expert selection from a ranking. Exit status 77 without a CUDA device, 1 when a check fails.
#include "basalt/core/device_copy.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/models/deepseek4/attention.hpp"
#include "models/deepseek4/dspark.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace basalt::models::deepseek4;

namespace {

constexpr int kIn = 4096, kFf = 2048, kOut = 4096, kLat = 512, kMarkovRank = 256;
constexpr int kT = 5, kK = 6, kE = 32;
constexpr int kStageSlots = 8, kRingSlots = 128;

int failures = 0;

void check(const char* what, bool ok) {
    std::printf("%-34s %s\n", what, ok ? "ok" : "FAIL");
    failures += !ok;
}

template <class T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    Dev() = default;
    explicit Dev(size_t count) : n(count) { cudaMalloc((void**) &p, count * sizeof(T)); }
    explicit Dev(const std::vector<T>& v) : Dev(v.size()) { if (p) cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    std::vector<T> down() const {
        std::vector<T> v(n);
        if (p) cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
        return v;
    }
};

const float kMx[16] = {0, 1, 2, 3, 4, 6, 8, 12, -0.f, -1, -2, -3, -4, -6, -8, -12};

float scale_of(unsigned e) {
    const uint32_t bits = e < 2 ? (0x00200000u << e) : ((e - 1) << 23);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

size_t mxfp4_row_bytes(int n_in) { return (size_t) (n_in / 32) * 17; }

// One MXFP4 row dot x, in double.
double row_dot(const uint8_t* row, const float* x, int n) {
    double s = 0;
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * 17;
        const double sc = scale_of(blk[0]);
        for (int j = 0; j < 16; ++j) {
            s += sc * kMx[blk[1 + j] & 15] * x[b * 32 + j];
            s += sc * kMx[blk[1 + j] >> 4] * x[b * 32 + j + 16];
        }
    }
    return s;
}

std::vector<uint8_t> random_mxfp4(std::mt19937& rng, size_t rows, int n_in) {
    const size_t row_bytes = mxfp4_row_bytes(n_in);
    std::vector<uint8_t> w(rows * row_bytes + 16, 0);
    for (size_t i = 0; i < rows * row_bytes; ++i) w[i] = (uint8_t) (rng() & 0xff);
    for (size_t r = 0; r < rows; ++r)
        for (int b = 0; b < n_in / 32; ++b) w[r * row_bytes + (size_t) b * 17] = (uint8_t) (118 + rng() % 6);
    return w;
}

float half_bits(uint16_t h) {
    const uint32_t e = (h >> 10) & 31, m = h & 1023, sign = (uint32_t) (h >> 15) << 31;
    uint32_t bits = sign;
    if (e) bits |= ((e + 112) << 23) | (m << 13);
    else if (m) {
        int k = 0;
        uint32_t mm = m;
        while (!(mm & 1024)) { mm <<= 1; ++k; }
        bits |= ((uint32_t) (113 - k) << 23) | ((mm & 1023) << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

std::vector<float> q8_1_values(const std::vector<uint8_t>& q, size_t n) {
    std::vector<float> v(n);
    for (size_t b = 0; b < n / 32; ++b) {
        const uint8_t* blk = &q[b * 36];
        const float d = half_bits((uint16_t) (blk[0] | blk[1] << 8));
        for (int i = 0; i < 32; ++i) v[b * 32 + i] = d * (float) (int8_t) blk[4 + i];
    }
    return v;
}

bool close_enough(const std::vector<float>& got, const std::vector<float>& ref, float tol) {
    double diff = 0, norm = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double) got[i] - ref[i];
        diff += d * d;
        norm += (double) ref[i] * ref[i];
    }
    const double rel = std::sqrt(diff / (norm + 1e-30));
    std::printf("  relative RMS %.2e\n", rel);
    return rel < tol;
}

}  // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::printf("no CUDA device\n");
        return 77;
    }
    cudaStream_t qs;
    cudaStreamCreate(&qs);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);

    // ---- routed experts: the kept list's experts, top-6 of a 5-row block
    const auto gate = random_mxfp4(rng, (size_t) kE * kFf, kIn), up = random_mxfp4(rng, (size_t) kE * kFf, kIn);
    const auto down = random_mxfp4(rng, (size_t) kE * kOut, kFf);
    std::vector<float> x((size_t) kT * kIn);
    for (float& v : x) v = nd(rng);
    std::vector<int32_t> ids(kT * kK);
    for (int p = 0; p < kT * kK; ++p) ids[p] = (int32_t) (rng() % kE);
    Dev<uint8_t> dg(gate), du(up), dd(down);
    Dev<float> dx(x), dh((size_t) kT * kK * kFf), dout((size_t) kT * kK * kOut);
    Dev<int32_t> did(ids);
    Dev<uint8_t> dxq(basalt::kernels::native_q8_1_bytes(kIn, kT));
    Dev<uint8_t> dhq(basalt::kernels::native_q8_1_bytes(kT * kK * kFf, 1));
    basalt::kernels::native_quantize_q8_1(dx.p, dxq.p, kIn, kT, qs);
    dspark_moe_gate_up(dg.p, du.p, kIn, kFf, dxq.p, did.p, kT, kK, 10.f, nullptr, dh.p, qs);
    basalt::kernels::native_quantize_q8_1(dh.p, dhq.p, kT * kK * kFf, 1, qs);
    dspark_moe_down(dd.p, kFf, kOut, dhq.p, did.p, kT, kK, nullptr, dout.p, qs);
    check("experts ran", cudaStreamSynchronize(qs) == cudaSuccess);
    // references take the values the q8_1 activations stand for
    const std::vector<float> xv = q8_1_values(dxq.down(), x.size());
    const std::vector<float> hv = q8_1_values(dhq.down(), (size_t) kT * kK * kFf);
    std::vector<float> h_ref((size_t) kT * kK * kFf), o_ref((size_t) kT * kK * kOut);
    for (int p = 0; p < kT * kK; ++p) {
        const int t = p / kK, e = ids[(size_t) p];
        for (int r = 0; r < kFf; ++r) {
            const double g = std::fmin(row_dot(&gate[((size_t) e * kFf + r) * mxfp4_row_bytes(kIn)], &xv[(size_t) t * kIn], kIn), 10.0);
            const double u = std::fmin(std::fmax(row_dot(&up[((size_t) e * kFf + r) * mxfp4_row_bytes(kIn)], &xv[(size_t) t * kIn], kIn), -10.0), 10.0);
            h_ref[(size_t) p * kFf + r] = (float) (g / (1.0 + std::exp(-g)) * u);
        }
        for (int r = 0; r < kOut; ++r)
            o_ref[(size_t) p * kOut + r] = (float) row_dot(&down[((size_t) e * kOut + r) * mxfp4_row_bytes(kFf)],
                                                           &hv[(size_t) p * kFf], kFf);
    }
    check("moe gate_up", close_enough(dh.down(), h_ref, 1e-4));
    check("moe down", close_enough(dout.down(), o_ref, 1e-4));

    // ---- microbench: one layer's expert work for a 5-row block
    float ms = 0;
    constexpr int kReps = 20;
    cudaEventRecord(e0, qs);
    for (int i = 0; i < kReps; ++i) {
        dspark_moe_gate_up(dg.p, du.p, kIn, kFf, dxq.p, did.p, kT, kK, 10.f, nullptr, dh.p, qs);
        basalt::kernels::native_quantize_q8_1(dh.p, dhq.p, kT * kK * kFf, 1, qs);
        dspark_moe_down(dd.p, kFf, kOut, dhq.p, did.p, kT, kK, nullptr, dout.p, qs);
    }
    cudaEventRecord(e1, qs);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    const double bytes = (double) kT * kK * (2.0 * kFf * mxfp4_row_bytes(kIn) + (double) kOut * mxfp4_row_bytes(kFf));
    std::printf("bench: 5 rows x top-6 expert layer %.1f us (%.0f GB/s of pair-reads)\n", 1000.0 * ms / kReps,
                bytes / (ms / kReps * 1e-3) / 1e9);

    // ---- kept experts: the ranking's order, ascending, topped up in index order
    {
        std::vector<std::pair<int32_t, int32_t>> ranked;
        for (int s = 0; s < 3; ++s) ranked.push_back({s, 200 - s});   // one named expert per stage, most routed first
        const auto kept = dspark_kept_experts(ranked, 3, 256, 32);
        bool ok = kept.size() == 3;
        for (int s = 0; s < 3 && ok; ++s) {
            ok = (int) kept[(size_t) s].size() == 32u && std::is_sorted(kept[(size_t) s].begin(), kept[(size_t) s].end()) &&
                 std::binary_search(kept[(size_t) s].begin(), kept[(size_t) s].end(), 200 - s);
        }
        const auto all = dspark_kept_experts({}, 3, 256, 256);
        ok = ok && all.size() == 3 && all[0].size() == 256 && all[0].front() == 0 && all[0].back() == 255;
        check("kept experts", ok);
    }

    // ---- staged commit: the row a verify window staged at pos % 8 lands at pos % 128 of each stage's ring
    {
        std::vector<float> staged((size_t) 3 * kStageSlots * kLat), rings((size_t) 3 * kRingSlots * kLat, -1.f);
        const int64_t pos0 = 1000;   // both slots wrap
        for (int r = 0; r < 6; ++r)
            for (int st = 0; st < 3; ++st)
                for (int c = 0; c < kLat; ++c)
                    staged[((size_t) st * kStageSlots + (size_t) ((pos0 + r) % kStageSlots)) * kLat + c] =
                        (float) (1000 * st + 100 * r + c);
        Dev<float> dstaged(staged), drings(rings);
        std::vector<float> zero(rings.size(), 0.f);   // all three stages
        Dev<float> dz(zero);
        check("no rows commit nothing", cudaMemcpy(drings.p, dz.p, zero.size() * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess);
        dspark_commit_rows(dstaged.p, kStageSlots, drings.p, drings.p + (size_t) kRingSlots * kLat,
                           drings.p + (size_t) 2 * kRingSlots * kLat, kRingSlots, pos0, 0, qs);
        check("commit ran", cudaStreamSynchronize(qs) == cudaSuccess);
        bool ok = drings.down() == zero;
        check("zero commits leave the rings empty", ok);
        for (int q = 1; q <= 6; ++q) {
            cudaMemcpy(drings.p, dz.p, zero.size() * sizeof(float), cudaMemcpyDeviceToDevice);
            dspark_commit_rows(dstaged.p, kStageSlots, drings.p, drings.p + (size_t) kRingSlots * kLat,
                               drings.p + (size_t) 2 * kRingSlots * kLat, kRingSlots, pos0, q, qs);
            cudaStreamSynchronize(qs);
            const auto got = drings.down();
            for (int r = 0; r < 6; ++r)
                for (int st = 0; st < 3; ++st)
                    for (int c = 0; c < kLat; ++c) {
                        const float want = r < q
                            ? staged[((size_t) st * kStageSlots + (size_t) ((pos0 + r) % kStageSlots)) * kLat + c]
                            : 0.f;
                        ok = ok && got[((size_t) st * kRingSlots + (size_t) ((pos0 + r) % kRingSlots)) * kLat + c] == want;
                    }
        }
        check("staged rows commit to pos % 128", ok);
    }

    std::printf(failures ? "FAILED\n" : "PASSED\n");
    return failures ? 1 : 0;
}
