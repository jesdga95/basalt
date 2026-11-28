// tests/kernels/q8_wide_window_test.cpp - the window gemm on Q8 planes at a routed-expert model's shapes: a window of more than 16 rows
// (batched sessions) must give each row the same bits as the same rows in windows of 16 or fewer.  Single matrices, the groups
// a layer runs together, and the BF16 residual add.
#include "basalt/kernels/window_gemm.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace wg = basalt::kernels::wgemm;

namespace {
std::mt19937 rng(5);
wg::Workspace ws;
int fails = 0;

void need(const wg::Call& c) {
    size_t p, t;
    wg::workspace_need(c, p, t);
    if (p > ws.partial_floats || t > ws.tickets_n) {
        cudaDeviceSynchronize();
        cudaFree(ws.partial);
        cudaFree(ws.tickets);
        ws.partial_floats = std::max(ws.partial_floats, p);
        ws.tickets_n = std::max(ws.tickets_n, t);
        cudaMalloc(&ws.partial, std::max<size_t>(ws.partial_floats, 1) * 4);
        cudaMalloc(&ws.tickets, ws.tickets_n * 4);
        cudaMemset(ws.tickets, 0, ws.tickets_n * 4);
    }
}

struct Planes {
    uint8_t* w = nullptr;
    int64_t rows = 0;
};

Planes make_planes(int64_t rows, int64_t K) {
    std::vector<uint8_t> blk((size_t) rows * K / 32 * 34);
    std::normal_distribution<float> nd;
    for (size_t b = 0; b < blk.size() / 34; ++b) {
        const __half d = __float2half(0.01f * (1.0f + 0.5f * nd(rng)));
        std::memcpy(&blk[b * 34], &d, 2);
        for (int k = 0; k < 32; ++k) blk[b * 34 + 2 + k] = (uint8_t) (int8_t) (rng() % 255 - 127);
    }
    uint8_t *d_blk, *d_pl;
    cudaMalloc(&d_blk, blk.size());
    cudaMalloc(&d_pl, blk.size());
    cudaMemcpy(d_blk, blk.data(), blk.size(), cudaMemcpyHostToDevice);
    wg::q8_planes(d_blk, rows, K, d_pl, nullptr);
    cudaFree(d_blk);
    return {d_pl, rows};
}

// one case: matrices `rows` against K, T rows in one call and the same rows as 16 + the rest
void run(const char* name, std::vector<int64_t> rows, int64_t K, bool add_bf16, int T) {
    std::vector<Planes> pl;
    for (int64_t r : rows) pl.push_back(make_planes(r, K));
    std::vector<float> x((size_t) wg::kRows * K);
    std::normal_distribution<float> nd;
    for (float& v : x) v = nd(rng);
    float* d_x;
    cudaMalloc(&d_x, x.size() * 4);
    cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    void* act;
    cudaMalloc(&act, wg::act_bytes(K));
    wg::pack(d_x, K, K, wg::kRows, act, nullptr);
    const int n = (int) rows.size();
    std::vector<float*> y1(n), y2(n);
    std::vector<uint16_t*> h1(n), h2(n);
    size_t total = 0;
    for (int i = 0; i < n; ++i) {
        const size_t sz = (size_t) wg::kRows * rows[i];
        cudaMalloc(&y1[i], sz * 4);
        cudaMalloc(&y2[i], sz * 4);
        cudaMalloc(&h1[i], sz * 2);
        cudaMalloc(&h2[i], sz * 2);
        cudaMemset(y1[i], 0, sz * 4);
        cudaMemset(y2[i], 0, sz * 4);
        std::vector<uint16_t> init(sz);
        for (uint16_t& v : init) v = (uint16_t) (0x3F00 + (rng() & 0xFF));
        cudaMemcpy(h1[i], init.data(), sz * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(h2[i], init.data(), sz * 2, cudaMemcpyHostToDevice);
        total += sz;
    }
    auto call = [&](std::vector<float*>& y, std::vector<uint16_t*>& h, int row0, int t) {
        wg::Call c;
        c.fmt = wg::Format::Q8;
        c.K = K;
        c.epi = add_bf16 ? wg::Epilogue::AddBf16 : wg::Epilogue::Store;
        c.n = n;
        for (int i = 0; i < n; ++i) {
            c.m[i] = {pl[i].w, rows[i], nullptr, add_bf16 ? nullptr : y[i] + (size_t) row0 * rows[i], rows[i]};
            if (add_bf16) c.m[i].y16 = h[i] + (size_t) row0 * rows[i];
        }
        need(c);
        // the activation rows row0.. of `act`: pack them fresh at the front
        void* a2;
        cudaMalloc(&a2, wg::act_bytes(K));
        wg::pack(d_x + (size_t) row0 * K, K, K, t, a2, nullptr);
        wg::gemm(c, a2, t, ws, nullptr);
        cudaDeviceSynchronize();
        cudaFree(a2);
    };
    call(y1, h1, 0, T);                       // the whole window
    call(y2, h2, 0, 16);                      // the same rows in windows of 16 and the rest
    call(y2, h2, 16, T - 16);
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        const size_t cnt = (size_t) T * rows[i];
        if (add_bf16) {
            std::vector<uint16_t> a(cnt), b(cnt);
            cudaMemcpy(a.data(), h1[i], cnt * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), h2[i], cnt * 2, cudaMemcpyDeviceToHost);
            bad += std::memcmp(a.data(), b.data(), cnt * 2) != 0;
        } else {
            std::vector<float> a(cnt), b(cnt);
            cudaMemcpy(a.data(), y1[i], cnt * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), y2[i], cnt * 4, cudaMemcpyDeviceToHost);
            bad += std::memcmp(a.data(), b.data(), cnt * 4) != 0;
        }
    }
    std::printf("%-34s T=%2d %s%s\n", name, T, add_bf16 ? "add-bf16 " : "store    ", bad ? "MISMATCH" : "same bits");
    fails += bad != 0;
    (void) total;
}
}  // namespace

int main() {
    for (int T : {20, 24, 32}) {
        run("gate_shexp 512 x2048", {512}, 2048, false, T);
        run("down_shexp 2048 x512", {2048}, 512, false, T);
        run("down_shexp 2048 x512", {2048}, 512, true, T);
        run("attn_out 2048 x4096", {2048}, 4096, true, T);
        run("q 8192 x2048", {8192}, 2048, false, T);
        run("q+k+v", {8192, 512, 512}, 2048, false, T);
        run("qkv+z (gdn)", {8192, 4096}, 2048, false, T);
        run("gate+up shared", {512, 512}, 2048, false, T);
        run("ssm_out 2048 x4096", {2048}, 4096, true, T);
    }
    std::printf("%s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
