// tests/models/qwen4exp/kernels/kv_fp4_parity.cpp - Parity test for the NVFP4 KV storage (`--kv nvfp4`, kv_q4.hpp).
// Validates:
// 1. The E4M3 / E2M1 round trips of fp4_bits.cuh against host tables (every code byte's value, every magnitude's
//    nearest code with ties to even).
// 2. kv_append_fp4_step: the pool's bytes (128 code bytes + 16 E4M3 scale bytes per 256-value head row) bitwise
//    against a host reference quantizer, on a permuted page table.
// 3. kv_gather_fp4_step: the FP16 scratch equals the host dequantization of exactly those bytes, bitwise.
// 4. Reconstruction quality against pre-rotation values (reported, not gated): the NVFP4 error through the
//    Hadamard rotation is what --kv nvfp4 trades q4_0's coarser scale for.
// Run without arguments; exit 0 when every exactness check holds.

#include "basalt/kernels/f16_bits.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q4.hpp"
#include "basalt/models/qwen4exp/kernels/qsa.hpp"

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

// ---- host reference for the NVFP4 rounding. The E4M3 pair goes through cuda_fp8.h's HOST conversions, the very
// functions fp4_bits.cuh calls on the device, so the references' rounding is the engine's by construction; the E2M1
// pair is re-implemented here from fp4_bits.cuh's thresholds (the device's is an exact comparison chain). ----

float e4m3_value_host(uint8_t b) {   // E4M3 -> float, exact (every code fits FP16); the NaN code never appears
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) b, __NV_E4M3)));
}

uint8_t e4m3_code_host(float v) { return (uint8_t) __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3); }

uint8_t e2m1_code_host(float v) {   // nearest of 0 .5 1 1.5 2 3 4 6, ties to even, bit 3 the sign
    const float a = std::fabs(v);
    uint8_t m;
    if (a <= 0.25f) m = 0;
    else if (a < 0.75f) m = 1;
    else if (a <= 1.25f) m = 2;
    else if (a < 1.75f) m = 3;
    else if (a <= 2.5f) m = 4;
    else if (a < 3.5f) m = 5;
    else if (a <= 5.0f) m = 6;
    else m = 7;
    return (uint8_t) (m | ((std::signbit(v) ? 1u : 0u) << 3));
}

float e2m1_value_host(uint8_t c) {
    static const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float v = mag[c & 7];
    return (c & 8) ? -v : v;
}

// One 16-value group: writes the 8 code bytes and the scale byte the kernels write for it.
void quantize_group_fp4(const float* x, uint8_t* codes8, uint8_t* scale) {
    float amax = 0.0f;
    for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(x[i]));
    const uint8_t sc = e4m3_code_host(std::max(amax, 6.0f * 0.001953125f) / 6.0f);
    const float s = e4m3_value_host(sc);
    for (int j = 0; j < 8; ++j) {
        uint8_t lo = 0, hi = 0;
        for (int t = 0; t < 2; ++t) {
            const float v = x[2 * j + t];
            float r = v / s;
            r = std::min(std::max(r, -6.0f), 6.0f);
            const uint8_t c = e2m1_code_host(r);
            if (t == 0) lo = c; else hi = c;
        }
        codes8[j] = (uint8_t) (lo | (hi << 4));   // even value in the low nibble (kv_q4.hpp)
    }
    *scale = sc;
}

// One 256-value head row: 128 code bytes then 16 scale bytes.
void quantize_row_fp4_host(const float* x, uint8_t* row) {
    for (int g = 0; g < 16; ++g) quantize_group_fp4(x + g * 16, row + g * 8, row + 128 + g);
}

// Exact CPU reference for the orthonormal Fast Walsh-Hadamard Transform (D = 256), as kv_q4_parity's.
void fwht256_host(float* x) {
    constexpr int n = 256;
    for (int s = 1; s < n; s *= 2)
        for (int i = 0; i < n; i += 2 * s)
            for (int j = 0; j < s; ++j) {
                const float u = x[i + j], v = x[i + j + s];
                x[i + j] = u + v;
                x[i + j + s] = u - v;
            }
    for (int i = 0; i < n; ++i) x[i] *= 1.0f / 16.0f;
}

}  // namespace

int main() {
    std::printf("=== Running kv_fp4_parity test ===\n");
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // [1/4] the rounding tables: every E4M3 code's value and every sampled magnitude's code, host vs nothing to
    // compare on the device here - the pool comparison in [2/4] is the bitwise end-to-end check; this one pins the
    // host reference itself (and that the NaN code is never produced by the encoder).
    std::printf("[1/4] Checking the host NVFP4 reference's own round trips...\n");
    {
        for (int b = 0; b < 256; ++b) {
            if (b == 0x7F || b == 0xFF) continue;   // the NaN codes: never decoded from pools the kernels wrote
            const float v = e4m3_value_host((uint8_t) b);
            if (e4m3_code_host(v) != (uint8_t) b) {
                std::fprintf(stderr, "FAIL: E4M3 0x%02x -> %g -> 0x%02x\n", b, v, e4m3_code_host(v));
                ++g_fail;
                break;
            }
        }
        // E2M1: every grid point re-encodes to itself, and midpoints go to the EVEN neighbour (the thresholds above)
        for (int c = 0; c < 8; ++c) {
            if (e2m1_code_host(e2m1_value_host((uint8_t) c)) != (uint8_t) c) {
                std::fprintf(stderr, "FAIL: E2M1 %d round trip\n", c);
                ++g_fail;
            }
        }
        const float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
        const uint8_t evens[7] = {0, 2, 2, 4, 4, 6, 6};   // ties to even: 0, 1, 1, 2, 2, 4, 4 (magnitudes)
        for (int i = 0; i < 7; ++i)
            if (e2m1_code_host(mids[i]) != evens[i]) {
                std::fprintf(stderr, "FAIL: E2M1 tie at %g -> code %u, expected %u\n", mids[i],
                             e2m1_code_host(mids[i]), evens[i]);
                ++g_fail;
            }
        std::printf("  -> host reference OK\n");
    }

    // [2/4] + [3/4] paged append and gather against the host reference.
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv;   // 2
    const int D = (int) s.head_dim;    // 256
    const int P = (int) s.page_size;
    const int pages = 8;
    const int cells = pages * P;

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 3 + 5) % pages;
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy table");

    const size_t row_bytes = (size_t) k::kv_q4_bytes_per_head(D);   // the pool rows are q4_0-sized (144 at 256)
    if (k::kv_fp4_bytes_per_head(D) != row_bytes) {
        std::fprintf(stderr, "FAIL: the NVFP4 row (%llu B) does not fit the shared pool row (%llu B)\n",
                     (unsigned long long) k::kv_fp4_bytes_per_head(D), (unsigned long long) row_bytes);
        return 1;
    }
    const size_t pool_bytes = (size_t) pages * H * P * row_bytes;
    uint8_t* d_k = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_v = dalloc<uint8_t>(pool_bytes);
    float* d_kcur = dalloc<float>(H * D);
    float* d_vcur = dalloc<float>(H * D);
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);

    std::printf("[2/4] Appending %d rotated cells and comparing the pool bytes to the host quantizer...\n", cells - 25);
    std::vector<std::vector<float>> host_k(cells), host_v(cells);
    std::vector<int> positions(cells);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);

    const int n_fill = cells - 25;   // leave some cells empty
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        std::vector<float> kv(H * D), vv(H * D);
        for (auto& x : kv) x = nd(rng) * 2.0f;
        for (auto& x : vv) x = nd(rng) * 2.0f;
        for (int h = 0; h < H; ++h) {   // rotated before they are stored, as layer.cpp does
            fwht256_host(kv.data() + h * D);
            fwht256_host(vv.data() + h * D);
        }
        host_k[pos] = kv;
        host_v[pos] = vv;

        int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
        ck(cudaMemcpy(d_kcur, kv.data(), kv.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy k");
        ck(cudaMemcpy(d_vcur, vv.data(), vv.size() * sizeof(float), cudaMemcpyHostToDevice), "memcpy v");
        k::kv_append_fp4_step(d_k, d_v, d_table, d_step, d_kcur, d_vcur, s, nullptr);
        ck(cudaDeviceSynchronize(), "kv_append_fp4_step");
    }

    std::vector<uint8_t> h_k(pool_bytes), h_v(pool_bytes);
    ck(cudaMemcpy(h_k.data(), d_k, pool_bytes, cudaMemcpyDeviceToHost), "d2h k");
    ck(cudaMemcpy(h_v.data(), d_v, pool_bytes, cudaMemcpyDeviceToHost), "d2h v");

    long bad_rows = 0;
    std::vector<uint8_t> ref_row(row_bytes);
    for (int n = 0; n < n_fill; ++n) {
        const int pos = positions[n];
        for (int is_v = 0; is_v < 2; ++is_v)
            for (int h = 0; h < H; ++h) {
                const long long page = (long long) table[pos / P];
                const long long row = (page * H + h) * P + (pos % P);
                quantize_row_fp4_host((is_v ? host_v[pos] : host_k[pos]).data() + h * D, ref_row.data());
                const uint8_t* gpu = (is_v ? h_v : h_k).data() + row * row_bytes;
                if (std::memcmp(gpu, ref_row.data(), k::kv_fp4_bytes_per_head(D)) != 0) {
                    if (bad_rows == 0) {
                        std::printf("MISMATCH pos=%d is_v=%d h=%d\n  host: ", pos, is_v, h);
                        for (int i = 0; i < 16; ++i) std::printf("%02x ", ref_row[i]);
                        std::printf("\n  gpu : ");
                        for (int i = 0; i < 16; ++i) std::printf("%02x ", gpu[i]);
                        std::printf("\n");
                    }
                    ++bad_rows;
                }
            }
    }
    if (bad_rows > 0) {
        std::fprintf(stderr, "FAIL: %ld NVFP4 rows differ from the host quantizer\n", bad_rows);
        ++g_fail;
    } else {
        std::printf("  -> all NVFP4 pool rows bitwise equal to the host quantizer (OK)\n");
    }

    std::printf("[3/4] Gathering cells and comparing the FP16 scratch to the host dequantization...\n");
    const int max_ids = 128;
    int32_t* d_ids = dalloc<int32_t>(max_ids);
    uint16_t* d_ks = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t* d_vs = dalloc<uint16_t>((size_t) max_ids * H * D);
    std::vector<int32_t> ids(max_ids);
    for (int i = 0; i < max_ids; ++i) ids[i] = positions[rng() % n_fill];
    ck(cudaMemcpy(d_ids, ids.data(), max_ids * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy ids");
    int32_t hstep[k::kStepCount] = {0, 0, 0, max_ids};
    ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");

    k::kv_gather_fp4_step(d_k, d_v, d_table, d_ids, d_step, max_ids, s, d_ks, d_vs, nullptr);
    ck(cudaDeviceSynchronize(), "kv_gather_fp4_step");

    std::vector<uint16_t> h_ks((size_t) max_ids * H * D), h_vs((size_t) max_ids * H * D);
    ck(cudaMemcpy(h_ks.data(), d_ks, h_ks.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "d2h ks");
    ck(cudaMemcpy(h_vs.data(), d_vs, h_vs.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "d2h vs");

    long bad_vals = 0;
    double worst = 0.0;
    for (int i = 0; i < max_ids; ++i) {
        const int cell = ids[i];
        for (int is_v = 0; is_v < 2; ++is_v)
            for (int h = 0; h < H; ++h) {
                quantize_row_fp4_host((is_v ? host_v[cell] : host_k[cell]).data() + h * D, ref_row.data());
                const uint16_t* got = (is_v ? h_vs : h_ks).data() + (i * H + h) * D;
                for (int d = 0; d < D; ++d) {
                    const float want = e2m1_value_host(ref_row[d / 2] >> (4 * (d & 1))) *
                                       e4m3_value_host(ref_row[D / 2 + d / 16]);
                    const uint16_t got_bits = got[d];
                    const float got_f = k::f32_from_f16(got_bits);
                    const uint16_t want_bits = k::f16_from_f32(want);
                    if (got_bits != want_bits && !(std::isnan(got_f) && std::isnan(want))) {
                        if (bad_vals < 5)
                            std::printf("gather mismatch i=%d is_v=%d h=%d d=%d: gpu 0x%04x host 0x%04x (%g vs %g)\n",
                                        i, is_v, h, d, got_bits, want_bits, got_f, want);
                        ++bad_vals;
                    }
                    worst = std::max(worst, (double) std::fabs(got_f -
                                                               (is_v ? host_v[cell] : host_k[cell])[h * D + d]));
                }
            }
    }
    if (bad_vals > 0) {
        std::fprintf(stderr, "FAIL: %ld gathered values differ from the host dequantization\n", bad_vals);
        ++g_fail;
    } else {
        std::printf("  -> all gathered FP16 values bitwise equal to the host dequantization (OK)\n");
    }

    // [4/4] reported quality: error through rotation + quantization, vs the same numbers q4_0's parity prints.
    std::printf("[4/4] Reconstruction error through the rotation (reported): worst |x - x'| = %.4f\n", worst);

    cudaFree(d_table); cudaFree(d_k); cudaFree(d_v); cudaFree(d_kcur); cudaFree(d_vcur);
    cudaFree(d_step); cudaFree(d_ids); cudaFree(d_ks); cudaFree(d_vs);

    if (g_fail == 0) {
        std::printf("=== kv_fp4_parity: ALL TESTS PASSED ===\n");
        return 0;
    }
    std::fprintf(stderr, "=== kv_fp4_parity: FAILED (%d failures) ===\n", g_fail);
    return 1;
}
