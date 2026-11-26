// tests/kernels/moe_mxfp8_test.cpp - prefill::mxfp8 parity against double-precision CPU reference on real Inkling 2
// Mini expert shapes (E=2304, F=1152) and --bench TOP/s. Exit 77 without a GPU.
#include "basalt/prefill/moe_mxfp8.hpp"
#define GGML_COMMON_DECL_CPP
#include "basalt/ggml/ggml_common.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mx = basalt::prefill::mxfp8;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "moe_mxfp8_test: CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
double e4m3_d(uint8_t b) {
    int e = (b >> 3) & 0xF, m = b & 0x7;
    if (e == 15 && m == 7) return 0.0;
    double v = e == 0 ? std::ldexp((double) m, -9) : std::ldexp(1.0 + (double) m * 0.125, e - 7);
    return (b & 0x80) ? -v : v;
}
double ue8m0_d(uint8_t u) {
    if (u == 255) return 0.0;
    return std::ldexp(1.0, (int) u - 127);
}
double h2d(uint16_t u) {
    __half_raw r;
    r.x = u;
    return (double) __half2float(__half(r));
}
void fill_blocks(std::vector<uint8_t>& buf, uint64_t seed) {
    uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
    const size_t nblk = buf.size() / sizeof(block_mxfp8);
    auto* blk = reinterpret_cast<block_mxfp8*>(buf.data());
    for (size_t i = 0; i < nblk; ++i) {
        for (int s = 0; s < 4; ++s) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            blk[i].d[s] = (uint8_t) (115 + (x % 6)); // 2^-12 .. 2^-7 (realistic weights)
        }
        for (int k = 0; k < 128; ++k) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            uint8_t b = (uint8_t) (x >> 24);
            if ((b & 0x7f) == 0x7f) b = 0x38;
            blk[i].qs[k] = b;
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return 77;

    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        const int E = 2304, F = 1152, n = 128, rows = 4096, k_used = 8, pairs = rows * k_used;
        const size_t gu_bytes = (size_t) (2 * F) * (E / 128) * sizeof(block_mxfp8);
        const size_t d_bytes = (size_t) E * (F / 128) * sizeof(block_mxfp8);
        const size_t blob = gu_bytes + d_bytes;
        const size_t stride = ((blob + 15) / 16) * 16;
        uint8_t* dw; cudaMalloc(&dw, (size_t) n * stride + 64); cudaMemset(dw, 0x38, (size_t) n * stride);
        float *dx, *dout; cudaMalloc(&dx, (size_t) rows * E * 4); cudaMalloc(&dout, (size_t) pairs * E * 4);
        uint16_t* dh16; cudaMalloc(&dh16, (size_t) pairs * F * 2);
        void *da_x, *da_h;
        cudaMalloc(&da_x, mx::act_bytes(rows, E));
        cudaMalloc(&da_h, mx::act_bytes(pairs, F));
        std::vector<int32_t> hb(n + 1), hids(pairs), hdst(pairs);
        for (int j = 0; j <= n; ++j) hb[j] = j * (pairs / n);
        for (int i = 0; i < pairs; ++i) { hids[i] = i % rows; hdst[i] = i; }
        int32_t *db, *dids, *ddst, *dtile, *dblk;
        cudaMalloc(&db, (n + 1) * 4); cudaMalloc(&dids, pairs * 4); cudaMalloc(&ddst, pairs * 4);
        cudaMalloc(&dtile, (n + 1) * 4); cudaMalloc(&dblk, (n + 1) * 4);
        cudaMemcpy(db, hb.data(), (n + 1) * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dids, hids.data(), pairs * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ddst, hdst.data(), pairs * 4, cudaMemcpyHostToDevice);
        mx::plan_blocks(db, n, dblk, nullptr);

        mx::Product gu{};
        gu.base = dw; gu.stride = stride; gu.w_off = 0; gu.w_rows = 2 * F; gu.w_cols = E; gu.n = n;
        gu.act = da_x; gu.act_rows = rows; gu.bounds = db; gu.blk_off = dblk; gu.blocks = mx::max_blocks(pairs, n);
        gu.ids = dids; gu.dst_f16 = dh16; gu.ld_dst = F; gu.swiglu = true; gu.gate_softcap = 15.0f; gu.up_softcap = 15.0f;

        mx::Product dn{};
        dn.base = dw; dn.stride = stride; dn.w_off = gu_bytes; dn.w_rows = E; dn.w_cols = F; dn.n = n;
        dn.act = da_h; dn.act_rows = pairs; dn.bounds = db; dn.blk_off = dblk; dn.blocks = mx::max_blocks(pairs, n);
        dn.dst_ids = ddst; dn.dst = dout; dn.ld_dst = E; dn.swiglu = false;

        mx::quantize_tokens(dx, rows, E, E, da_x, nullptr);
        mx::run(gu, nullptr);
        mx::quantize_f16(dh16, pairs, F, F, da_h, nullptr);
        mx::run(dn, nullptr);
        cudaDeviceSynchronize();

        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        auto meas = [&](const char* label, double ops, auto fn) {
            cudaEventRecord(a);
            for (int i = 0; i < 10; ++i) fn();
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b); ms /= 10.0f;
            std::printf("%-14s: %7.1f us (%5.1f TOP/s)\n", label, ms * 1e3, ops / (ms * 1e9));
        };
        meas("quant_x", 0.0, [&]{ mx::quantize_tokens(dx, rows, E, E, da_x, nullptr); });
        meas("gu+swiglu", 2.0 * pairs * (2 * F) * E, [&]{ mx::run(gu, nullptr); });
        meas("quant_h16", 0.0, [&]{ mx::quantize_f16(dh16, pairs, F, F, da_h, nullptr); });
        meas("down", 2.0 * pairs * E * F, [&]{ mx::run(dn, nullptr); });
        meas("full_layer", 2.0 * pairs * (3 * F) * E, [&]{
            mx::quantize_tokens(dx, rows, E, E, da_x, nullptr);
            mx::run(gu, nullptr);
            mx::quantize_f16(dh16, pairs, F, F, da_h, nullptr);
            mx::run(dn, nullptr);
        });
        return 0;
    }

    // Full parity test: 3 experts, uneven row counts (0, 67, 135 rows -> tests 0-row, 1-tile, 2-tile, tail masking)
    const int E = 256, F = 128, n = 3, tokens = 91;
    const int counts[3] = {0, 67, 271};
    const int pairs = counts[0] + counts[1] + counts[2];
    const size_t gu_bytes = (size_t) (2 * F) * (E / 128) * sizeof(block_mxfp8);
    const size_t d_bytes = (size_t) E * (F / 128) * sizeof(block_mxfp8);
    const size_t stride = ((gu_bytes + d_bytes + 15) / 16) * 16;

    std::vector<uint8_t> hw((size_t) n * stride + 64, 0);
    for (int j = 0; j < n; ++j) {
        std::vector<uint8_t> tmp(gu_bytes + d_bytes);
        fill_blocks(tmp, 100 + j);
        std::memcpy(hw.data() + j * stride, tmp.data(), tmp.size());
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::vector<float> hx((size_t) tokens * E);
    for (float& v : hx) v = nd(rng);
    std::vector<int32_t> hb = {0, counts[0], counts[0] + counts[1], pairs};
    std::vector<int32_t> hids(pairs), hdst(pairs);
    for (int i = 0; i < pairs; ++i) { hids[i] = (i * 17 + 3) % tokens; hdst[i] = pairs - 1 - i; }

    uint8_t* dw; float *dx, *dout; uint16_t* dh16; void *da_x, *da_h;
    int32_t *db, *dids, *ddst, *dtile, *dblk;
    ck(cudaMalloc(&dw, hw.size()), "malloc dw"); ck(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice), "copy dw");
    ck(cudaMalloc(&dx, hx.size() * 4), "malloc dx"); ck(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "copy dx");
    ck(cudaMalloc(&dh16, (size_t) pairs * F * 2), "malloc dh16");
    ck(cudaMalloc(&dout, (size_t) pairs * E * 4), "malloc dout");
    ck(cudaMalloc(&da_x, mx::act_bytes(tokens, E)), "malloc da_x");
    ck(cudaMalloc(&da_h, mx::act_bytes(pairs, F)), "malloc da_h");
    ck(cudaMalloc(&db, 4 * 4), "malloc db"); ck(cudaMemcpy(db, hb.data(), 16, cudaMemcpyHostToDevice), "copy db");
    ck(cudaMalloc(&dids, pairs * 4), "malloc dids"); ck(cudaMemcpy(dids, hids.data(), pairs * 4, cudaMemcpyHostToDevice), "copy dids");
    ck(cudaMalloc(&ddst, pairs * 4), "malloc ddst"); ck(cudaMemcpy(ddst, hdst.data(), pairs * 4, cudaMemcpyHostToDevice), "copy ddst");
    ck(cudaMalloc(&dtile, 16), "malloc dtile"); ck(cudaMalloc(&dblk, 16), "malloc dblk");

    mx::plan_blocks(db, n, dblk, nullptr);
    mx::quantize_tokens(dx, tokens, E, E, da_x, nullptr);

    mx::Product gu{};
    gu.base = dw; gu.stride = stride; gu.w_off = 0; gu.w_rows = 2 * F; gu.w_cols = E; gu.n = n;
    gu.act = da_x; gu.act_rows = tokens; gu.bounds = db; gu.blk_off = dblk; gu.blocks = mx::max_blocks(pairs, n);
    gu.ids = dids; gu.dst_f16 = dh16; gu.ld_dst = F; gu.swiglu = true; gu.gate_softcap = 15.0f; gu.up_softcap = 15.0f;
    mx::run(gu, nullptr);

    mx::quantize_f16(dh16, pairs, F, F, da_h, nullptr);
    mx::Product dn{};
    dn.base = dw; dn.stride = stride; dn.w_off = gu_bytes; dn.w_rows = E; dn.w_cols = F; dn.n = n;
    dn.act = da_h; dn.act_rows = pairs; dn.bounds = db; dn.blk_off = dblk; dn.blocks = mx::max_blocks(pairs, n);
    dn.dst_ids = ddst; dn.dst = dout; dn.ld_dst = E; dn.swiglu = false;
    mx::run(dn, nullptr);
    ck(cudaDeviceSynchronize(), "sync parity");

    // Read back quantized activations and outputs and compare against double-precision CPU reference on the exact same MXFP8 codes
    const size_t ax_bytes = mx::act_bytes(tokens, E);
    const size_t ah_bytes = mx::act_bytes(pairs, F);
    std::vector<uint8_t> hax(ax_bytes), hah(ah_bytes);
    std::vector<uint16_t> hh16((size_t) pairs * F);
    std::vector<float> hout((size_t) pairs * E);
    ck(cudaMemcpy(hax.data(), da_x, ax_bytes, cudaMemcpyDeviceToHost), "copy hax");
    ck(cudaMemcpy(hah.data(), da_h, ah_bytes, cudaMemcpyDeviceToHost), "copy hah");
    ck(cudaMemcpy(hh16.data(), dh16, hh16.size() * 2, cudaMemcpyDeviceToHost), "copy hh16");
    ck(cudaMemcpy(hout.data(), dout, hout.size() * 4, cudaMemcpyDeviceToHost), "copy hout");

    const size_t ax_soff = ((size_t) tokens * E + 15) & ~(size_t) 15;
    const size_t ah_soff = ((size_t) pairs * F + 15) & ~(size_t) 15;
    auto dot_row = [](const uint8_t* ac, const uint8_t* as, const block_mxfp8* wb, int cols) {
        double sum = 0.0;
        for (int b = 0; b < cols / 128; ++b) {
            for (int sb = 0; sb < 4; ++sb) {
                double d_a = ue8m0_d(as[b * 4 + sb]);
                double d_w = ue8m0_d(wb[b].d[sb]);
                double part = 0.0;
                for (int k = 0; k < 32; ++k) part += e4m3_d(ac[b * 128 + sb * 32 + k]) * e4m3_d(wb[b].qs[sb * 32 + k]);
                sum += part * d_a * d_w;
            }
        }
        return sum;
    };

    double max_h_err = 0.0, max_d_err = 0.0;
    int non_finite = 0;
    for (int j = 0; j < n; ++j) {
        const uint8_t* w_gu = hw.data() + j * stride;
        const uint8_t* w_dn = w_gu + gu_bytes;
        const int row_blks_gu = E / 128, row_blks_dn = F / 128;
        for (int i = hb[j]; i < hb[j + 1]; ++i) {
            const int tk = hids[i];
            for (int r = 0; r < F; ++r) {
                const double g = dot_row(hax.data() + tk * E, hax.data() + ax_soff + tk * (E / 32),
                                         reinterpret_cast<const block_mxfp8*>(w_gu) + r * row_blks_gu, E);
                const double u = dot_row(hax.data() + tk * E, hax.data() + ax_soff + tk * (E / 32),
                                         reinterpret_cast<const block_mxfp8*>(w_gu) + (F + r) * row_blks_gu, E);
                const double ref_h = (15.0 * std::tanh(g / 15.0)) * (1.0 / (1.0 + std::exp(-g))) * (15.0 * std::tanh(u / 15.0));
                const double got_h = h2d(hh16[(size_t) i * F + r]);
                const double err = std::abs(got_h - ref_h) / std::max(1.0, std::abs(ref_h));
                if (!std::isfinite(got_h) || !std::isfinite(err)) ++non_finite;
                else max_h_err = std::max(max_h_err, err);
            }
            const int out_r = hdst[i];
            for (int r = 0; r < E; ++r) {
                const double ref_d = dot_row(hah.data() + i * F, hah.data() + ah_soff + i * (F / 32),
                                             reinterpret_cast<const block_mxfp8*>(w_dn) + r * row_blks_dn, F);
                const double got_d = (double) hout[(size_t) out_r * E + r];
                const double err = std::abs(got_d - ref_d) / std::max(1.0, std::abs(ref_d));
                if (!std::isfinite(got_d) || !std::isfinite(err)) ++non_finite;
                else max_d_err = std::max(max_d_err, err);
            }
        }
    }
    ck(cudaGetLastError(), "final");
    std::printf("moe_mxfp8 parity: max_h_rel=%.3e (fp16), max_down_rel=%.3e (fp32), non_finite=%d\n", max_h_err, max_d_err, non_finite);
    if (non_finite > 0 || !std::isfinite(max_h_err) || !std::isfinite(max_d_err) || max_h_err > 2e-3 || max_d_err > 2e-5) {
        std::printf("FAIL parity!\n");
        return 1;
    }
    std::printf("moe_mxfp8_test: PASS\n");
    return 0;
}
