// tests/kernels/qsa_mid_parity.cpp - qsa_qproj_half against dense_tc and a CPU reference.
//
//     qsa_mid_parity [--bench]
//
// The QSA query projection's shape (2560 inputs, 24 heads of 256 query + 256 gate rows) for each format dense_tc takes
// there, random blocks (fp16 scales rewritten as normal values), normal activations quantized to q8_1 once.  Checks:
//   - the query half then the gate half (on the launcher's grids, 64, 33 and 1 blocks) give dense_tc's output bit for
//     bit, T = 1..8, and each half writes only its own rows;
//   - the error of both against a double reference over the dequantized weights and the q8_1 columns.
// --bench: dense_tc, the query half and the gate half on the launcher's grids, per T, timed as CUDA graphs over enough
// weight copies that the L2 holds none.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/qsa_mid.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = basalt::kernels;

namespace {

struct Case {
    const char* name;
    int type, block_bytes, scale_at[2];
};
const Case CASES[] = {
    {"Q6_K", 14, 210, {208, -1}}, {"Q8_0", 8, 34, {0, -1}},   {"Q4_K", 12, 144, {0, 2}}, {"Q5_K", 13, 176, {0, 2}},
    {"IQ4_XS", 23, 136, {0, -1}}, {"Q3_K", 11, 110, {108, -1}}, {"Q2_0", 42, 18, {0, -1}},
};
constexpr int NIN = 2560, NOUT = 24 * 512, TM = 8;

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
double h2d(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const double v = e == 0 ? std::ldexp((double) m, -24) : std::ldexp((double) (1024 + m), e - 25);
    return (h >> 15) ? -v : v;
}

int run_case(const Case& c, cudaStream_t s, bool bench) {
    int fails = 0;
    std::mt19937 rng(4242u + (unsigned) c.type * 17u);
    const std::size_t wbytes = k::native_mmvq_weight_bytes(c.type, NIN, NOUT);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    for (std::size_t q = 0; q < wbytes / (std::size_t) c.block_bytes; ++q)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[q * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    std::vector<float> x((std::size_t) TM * NIN);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    const std::size_t qcol = k::native_q8_1_bytes(NIN, 1);
    void *dw = nullptr, *xq = nullptr;
    float *dx = nullptr, *yref = nullptr, *yh = nullptr;
    const std::size_t ybytes = (std::size_t) TM * NOUT * 4;
    ck(cudaMalloc(&dw, wbytes), "malloc");
    ck(cudaMalloc(&xq, qcol * TM), "malloc");
    ck(cudaMalloc(&dx, x.size() * 4), "malloc");
    ck(cudaMalloc(&yref, ybytes), "malloc");
    ck(cudaMalloc(&yh, ybytes), "malloc");
    ck(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "up");
    k::native_quantize_q8_1(dx, xq, NIN, TM, s);
    ck(cudaStreamSynchronize(s), "quantize");
    if (!k::qsa_qproj_half_supported(c.type, dw, NIN, NOUT)) {
        std::printf("%-7s not taken by the halves (dense_tc does not tile this shape)\n", c.name);
        cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(yref); cudaFree(yh);
        return 0;
    }
    // the double reference: dequantized rows times the q8_1 columns' values
    std::vector<uint8_t> hq(qcol * TM);
    ck(cudaMemcpy(hq.data(), xq, hq.size(), cudaMemcpyDeviceToHost), "down");
    std::vector<double> xd((std::size_t) TM * NIN);
    for (int j = 0; j < TM; ++j)
        for (int b = 0; b < NIN / 32; ++b) {
            const uint8_t* blk = &hq[j * qcol + (std::size_t) b * 36];
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            for (int e = 0; e < 32; ++e) xd[(std::size_t) j * NIN + b * 32 + e] = h2d(dh) * (double) (int8_t) blk[4 + e];
        }
    const auto* tr = ggml_get_type_traits((ggml_type) c.type);
    const std::size_t rb = wbytes / NOUT;
    std::vector<double> ycpu((std::size_t) TM * NOUT);
    {
        std::vector<float> row(NIN);
        for (int r = 0; r < NOUT; ++r) {
            tr->to_float(&w[(std::size_t) r * rb], row.data(), NIN);
            for (int j = 0; j < TM; ++j) {
                double acc = 0;
                for (int e = 0; e < NIN; ++e) acc += (double) row[e] * xd[(std::size_t) j * NIN + e];
                ycpu[(std::size_t) j * NOUT + r] = acc;
            }
        }
    }
    double worst_ref = 0, worst_half = 0;
    for (int T = 1; T <= TM; ++T) {
        ck(cudaMemsetAsync(yref, 0, ybytes, s), "memset");
        k::dense_tc(c.type, dw, xq, yref, NIN, NOUT, T, s);
        std::vector<float> ref((std::size_t) T * NOUT);
        ck(cudaMemcpyAsync(ref.data(), yref, ref.size() * 4, cudaMemcpyDeviceToHost, s), "down");
        ck(cudaStreamSynchronize(s), "dense_tc");
        for (const int blocks : {0, 64, 33, 1}) {   // 0: the launcher's own grid (every SM, the gate share)
            std::vector<float> got((std::size_t) T * NOUT);
            ck(cudaMemsetAsync(yh, 0xff, ybytes, s), "memset");   // NaN everywhere
            k::qsa_qproj_half(c.type, dw, xq, yh, NIN, NOUT, T, 0, blocks, s);
            ck(cudaMemcpyAsync(got.data(), yh, got.size() * 4, cudaMemcpyDeviceToHost, s), "down");
            ck(cudaStreamSynchronize(s), "half 0");
            std::size_t wrong_rows = 0, diff = 0;
            for (int j = 0; j < T; ++j)
                for (int r = 0; r < NOUT; ++r) {
                    const bool mine = (r % 512) < 256;
                    const float v = got[(std::size_t) j * NOUT + r];
                    if (!mine && !std::isnan(v)) ++wrong_rows;
                    if (mine && std::memcmp(&v, &ref[(std::size_t) j * NOUT + r], 4) != 0) ++diff;
                }
            k::qsa_qproj_half(c.type, dw, xq, yh, NIN, NOUT, T, 1, blocks, s);
            ck(cudaMemcpyAsync(got.data(), yh, got.size() * 4, cudaMemcpyDeviceToHost, s), "down");
            ck(cudaStreamSynchronize(s), "half 1");
            for (std::size_t i = 0; i < got.size(); ++i) diff += std::memcmp(&got[i], &ref[i], 4) != 0;
            if (wrong_rows || diff) {
                ++fails;
                std::printf("%-7s T %d blocks %3d: FAIL (%zu outputs differ from dense_tc, %zu gate rows written by the query half)\n",
                            c.name, T, blocks, diff, wrong_rows);
            }
            if (blocks == 0)
                for (int j = 0; j < T; ++j)
                    for (int r = 0; r < NOUT; ++r) {
                        const double cr = ycpu[(std::size_t) j * NOUT + r], sc = std::max(1.0, std::fabs(cr));
                        worst_ref = std::max(worst_ref, std::fabs((double) ref[(std::size_t) j * NOUT + r] - cr) / sc);
                        worst_half = std::max(worst_half, std::fabs((double) got[(std::size_t) j * NOUT + r] - cr) / sc);
                    }
        }
    }
    std::printf("%-7s halves vs dense_tc: %s (T 1..8, grids own/64/33/1); vs the double reference: dense_tc %.3g, halves %.3g (max |d| / max(1, |ref|))\n",
                c.name, fails ? "FAIL" : "bitwise", worst_ref, worst_half);
    if (bench) {
        // copies of the weights past the L2, one per launch in turn
        int dev = 0, l2 = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
        const int ncopy = (int) std::max<std::size_t>(4, (std::size_t) 3 * (std::size_t) l2 / wbytes + 2);
        std::vector<void*> cw(ncopy);
        for (auto& p : cw) {
            ck(cudaMalloc(&p, wbytes), "malloc");
            ck(cudaMemcpy(p, dw, wbytes, cudaMemcpyDeviceToDevice), "copy");
        }
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const double gb = (double) wbytes / 2 / 1e9;
        for (int T = 1; T <= 4; ++T) {
            float ms[3] = {0, 0, 0};
            for (int arm = 0; arm < 3; ++arm) {
                cudaGraph_t g;
                cudaGraphExec_t ge;
                ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
                for (int i = 0; i < ncopy; ++i) {
                    k::PdlScope pdl(true, i == 0);
                    if (arm == 0) k::dense_tc(c.type, cw[i], xq, yref, NIN, NOUT, T, s);
                    else k::qsa_qproj_half(c.type, cw[i], xq, yh, NIN, NOUT, T, arm - 1, 0, s);
                }
                ck(cudaStreamEndCapture(s, &g), "capture end");
                ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
                ck(cudaGraphLaunch(ge, s), "warm");
                ck(cudaStreamSynchronize(s), "warm");
                cudaEventRecord(e0, s);
                for (int rep = 0; rep < 5; ++rep) cudaGraphLaunch(ge, s);
                cudaEventRecord(e1, s);
                ck(cudaEventSynchronize(e1), "time");
                cudaEventElapsedTime(&ms[arm], e0, e1);
                ms[arm] /= 5.0f * (float) ncopy;
                cudaGraphExecDestroy(ge);
                cudaGraphDestroy(g);
            }
            std::printf("   bench %-7s T %d: dense_tc %6.2f us (%5.0f GB/s) | query half, block per SM %6.2f us (%5.0f GB/s) | gate half, its share %6.2f us (%5.0f GB/s)\n",
                        c.name, T, 1e3 * ms[0], 2 * gb / ms[0] * 1e3, 1e3 * ms[1], gb / ms[1] * 1e3, 1e3 * ms[2], gb / ms[2] * 1e3);
        }
        for (auto p : cw) cudaFree(p);
    }
    cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(yref); cudaFree(yh);
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        std::printf("qsa_mid_parity: no CUDA device\n");
        return 77;
    }
    cudaDeviceProp pr;
    cudaGetDeviceProperties(&pr, 0);
    if (pr.major < 9) {
        std::printf("qsa_mid_parity: needs compute capability 9.0+\n");
        return 77;
    }
    cudaStream_t s;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    int fails = 0;
    for (const Case& c : CASES) fails += run_case(c, s, bench && (c.type == 14 || c.type == 8));
    std::printf("qsa_mid_parity: %s\n", fails ? "FAIL" : "OK (bitwise)");
    return fails ? 1 : 0;
}
