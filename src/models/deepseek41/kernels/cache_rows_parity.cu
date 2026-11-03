// Parity test of the packed cache rows (cache_rows.cuh): the device writers against the CPU reference packing, byte for
// byte, on random rows and on the edge cases (ties, saturation, tiny and zero groups, -0); the readers against the CPU
// unpacking, bit for bit; and the decoded values against the reference roundings held as fp32 and as fp16 (what an fp16
// cache of the rounded values would hold).
#include "basalt/models/deepseek41/attn_index.hpp"
#include "cache_rows.cuh"
#include "test_util.hpp"

#include <cstring>

using namespace dstest;
using namespace basalt::models::deepseek41;

namespace {

// kind 0 window rows (Fp8), 1 compressed rows (Fp4), 2 index keys (Fp4p); a block per row, a thread per value
__global__ void pack_kernel(const float* x, int d, int row_bytes, int kind, uint8_t* out) {
    const float v = x[(size_t) blockIdx.x * d + threadIdx.x];
    uint8_t* r = out + (size_t) blockIdx.x * row_bytes;
    if (kind == 0) rows::put_fp8(v, threadIdx.x, d, r);
    else if (kind == 1) rows::put_fp4(v, threadIdx.x, d, r);
    else rows::put_fp4p(v, threadIdx.x, d, r);
}

// the decoded rows: 4 values a thread through the readers the kernels use
__global__ void unpack_kernel(const uint8_t* in, int d, int row_bytes, int kind, float* y) {
    const uint8_t* r = in + (size_t) blockIdx.x * row_bytes;
    const int i = threadIdx.x;   // values 4i .. 4i+3
    float k[4];
    if (kind == 0) rows::fp8_values<4>(*reinterpret_cast<const uint32_t*>(r + 4 * i), r[d + 4 * i / 32], k);
    else if (kind == 1) rows::fp4_values<4>(*reinterpret_cast<const uint16_t*>(r + 2 * i), r[d / 2 + 4 * i / 16], k);
    else rows::fp4p_values<4>(*reinterpret_cast<const uint16_t*>(r + 2 * i), r[d / 2 + 4 * i / 32], k);
    for (int u = 0; u < 4; ++u) y[(size_t) blockIdx.x * d + 4 * i + u] = k[u];
}

__global__ void round_q_kernel(const float* x, float* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    y[i] = rows::round_fp4p(x[i]);
}

// Rows of d values: normal values at a random magnitude per group, plus groups of edge cases.
std::vector<float> make_rows(int n_rows, int d, int g, uint32_t seed, Round k) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> mag(-30, 14);
    std::vector<float> v((size_t) n_rows * d);
    for (int r = 0; r < n_rows; ++r)
        for (int g0 = 0; g0 < d; g0 += g) {
            const int kind = (int) (rng() % 10);
            const float m = std::ldexp(1.0f, mag(rng));
            for (int i = g0; i < g0 + g; ++i) {
                float x = nd(rng) * m;
                if (kind == 1) x = 0.f;                                            // a zero group
                if (kind == 2) x = (i & 1) ? -0.f : 1e-30f;                        // tiny, -0
                if (kind == 3) {   // ties half-way between codes, at a group scale of 1 (the group's first value)
                    const float h[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f, 1.0625f, 1.1875f};
                    x = (rng() & 1 ? -1.f : 1.f) * h[rng() % 9];
                }
                if (kind == 4) x = nd(rng) * 1e4f;                                 // large
                v[(size_t) r * d + i] = x;
            }
            if (kind == 3) v[(size_t) r * d + g0] = k == Round::Fp8Pow2 ? 448.f : 6.f;
        }
    return v;
}

void test_kind(Verdict& v, int kind) {
    const Round k = kind == 0 ? Round::Fp8Pow2 : kind == 1 ? Round::Fp4E4m3 : Round::Fp4Pow2;
    const int d = kind == 2 ? kIdxDim : kLatent, rb = kind == 0 ? kWinRowBytes : kind == 1 ? kCompRowBytes : kIdxRowBytes;
    const char* name = kind == 0 ? "window rows (E4M3, 2^k per 32)" : kind == 1 ? "compressed rows (E2M1, E4M3 per 16)"
                                                                                : "index keys (E2M1, 2^k per 32)";
    const int n = 4096;
    const auto x = make_rows(n, d, group_of(k), 1000 + (uint32_t) kind, k);
    Dev<float> dx(x), dy((size_t) n * d);
    Dev<uint8_t> dp((size_t) n * rb);
    check(cudaMemset(dp.p, 0, (size_t) n * rb), "memset");
    pack_kernel<<<n, d>>>(dx.p, d, rb, kind, dp.p);
    unpack_kernel<<<n, d / 4>>>(dp.p, d, rb, kind, dy.p);
    check(cudaDeviceSynchronize(), "codec");
    const auto got = dp.down();
    const auto y = dy.down();
    size_t bad_bytes = 0, bad_dec = 0, bad_round = 0, fp16_off = 0;
    for (int r = 0; r < n; ++r) {
        const auto want = ref_pack(&x[(size_t) r * d], d, k, rb);
        bad_bytes += std::memcmp(want.data(), &got[(size_t) r * rb], (size_t) rb) != 0;
        const auto dec = ref_unpack(&got[(size_t) r * rb], d, k);
        std::vector<float> rounded(x.begin() + (size_t) r * d, x.begin() + (size_t) (r + 1) * d);
        ref_round(rounded.data(), rounded.size(), k);
        for (int i = 0; i < d; ++i) {
            const float a = y[(size_t) r * d + i];
            bad_dec += std::memcmp(&a, &dec[(size_t) i], 4) != 0;
            bad_round += std::memcmp(&a, &rounded[(size_t) i], 4) != 0;
            const float h = __half2float(__float2half(rounded[(size_t) i]));
            fp16_off += std::memcmp(&h, &rounded[(size_t) i], 4) != 0;
        }
    }
    std::printf("%-38s %d rows: rows differing from the CPU packing %zu, decoded values differing from the CPU decode %zu "
                "and from the reference rounding %zu; rounded values an fp16 cache would change %zu of %zu\n",
                name, n, bad_bytes, bad_dec, bad_round, fp16_off, (size_t) n * d);
    v.expect(bad_bytes == 0, "packed rows equal the CPU packing");
    v.expect(bad_dec == 0, "decoded rows equal the CPU decode");
    v.expect(bad_round == 0, "decoded rows equal the reference rounding");
    if (kind == 2) {
        Dev<float> dq((size_t) n * d);
        round_q_kernel<<<n, d>>>(dx.p, dq.p);
        check(cudaDeviceSynchronize(), "round_q");
        auto want = x;
        ref_round(want.data(), want.size(), k);
        const auto q = dq.down();
        v.expect(std::memcmp(q.data(), want.data(), q.size() * 4) == 0, "query rounding equals the reference rounding");
        std::printf("index queries (rounded in fp32): %s\n", std::memcmp(q.data(), want.data(), q.size() * 4) == 0 ? "bitwise" : "DIFFER");
    }
}

}  // namespace

int main() {
    Verdict v;
    for (int kind = 0; kind < 3; ++kind) test_kind(v, kind);
    std::printf(v.failures ? "FAILED (%d)\n" : "PASS\n", v.failures);
    return v.failures ? 1 : 0;
}
