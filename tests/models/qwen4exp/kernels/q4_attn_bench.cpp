// tests/models/qwen4exp/kernels/q4_attn_bench.cpp - the fused decode attention on a q4_0 cache against int8.
//
// At the selection cap this reports the time per call and effective bandwidth over the K/V pool bytes.
// Each Q4_0 row reads 144 B per head and cell. Each int8 row reads 264 B, including scales.
//
//   ./q4_attn_bench [cells] [iterations]

#include "basalt/models/qwen4exp/kernels/kv_q4.hpp"
#include "basalt/models/qwen4exp/kernels/kv_q8.hpp"
#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_dec_attn.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T)), "memset");
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    const int cells = argc > 1 ? std::atoi(argv[1]) : 2048;   // the selection budget
    const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, G = D / k::KV_Q8_GROUP;
    const int q4row = (int) k::kv_q4_bytes_per_head(D);   // 144

    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> q((size_t) s.n_head * D);
    for (auto& x : q) x = nd(rng);

    int32_t* d_ids = dalloc<int32_t>(cells);
    std::vector<int32_t> ids(cells);
    for (int i = 0; i < cells; ++i) ids[i] = i;
    ck(cudaMemcpy(d_ids, ids.data(), cells * 4, cudaMemcpyHostToDevice), "ids");
    int32_t step[k::kStepCount];
    k::qsa_step_fill(step, cells - 1, s);
    step[k::kStepWidth] = cells;
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);
    ck(cudaMemcpy(d_step, step, sizeof step, cudaMemcpyHostToDevice), "step");

    const int64_t pages = (cells + s.page_size - 1) / s.page_size;
    int32_t* d_table = dalloc<int32_t>(pages);
    std::vector<int32_t> table(pages);
    for (int i = 0; i < (int) pages; ++i) table[i] = i;
    ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "table");

    float* d_q = dalloc<float>(q.size());
    ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "q");
    float* d_scr = dalloc<float>(k::qsa_decode_attn_scratch_floats(cells, s) + 16);
    float* d_attn = dalloc<float>((size_t) s.n_head * D);

    // int8 pools: codes + an fp16 scale per 64, 1,056 B per cell over both planes
    const size_t rows = (size_t) pages * H * s.page_size;
    int8_t* d_q8 = dalloc<int8_t>(rows * D * 2);
    uint16_t* d_s8 = dalloc<uint16_t>(rows * G * 2);
    // q4_0 pools: packed blocks, 576 B per cell over both planes
    uint8_t* d_q4 = dalloc<uint8_t>(rows * q4row * 2);

    k::QsaAttnPools i8;
    i8.k_q = d_q8; i8.v_q = d_q8 + rows * D;
    i8.k_scale = d_s8; i8.v_scale = d_s8 + rows * G;
    i8.page_table = d_table;
    k::QsaAttnPools q4;
    q4.k_q4 = d_q4; q4.v_q4 = d_q4 + rows * q4row;
    q4.page_table = d_table;

    cudaEvent_t t0, t1;
    ck(cudaEventCreate(&t0), "ev");
    ck(cudaEventCreate(&t1), "ev");
    auto time = [&](const k::QsaAttnPools& p) {
        for (int i = 0; i < 10; ++i)   // warm
            k::qsa_dec_attn(d_q, p, d_ids, d_step, cells, s, d_scr, d_attn, 1, nullptr, nullptr, nullptr, nullptr, nullptr);
        ck(cudaDeviceSynchronize(), "sync");
        ck(cudaEventRecord(t0), "rec");
        for (int i = 0; i < iters; ++i)
            k::qsa_dec_attn(d_q, p, d_ids, d_step, cells, s, d_scr, d_attn, 1, nullptr, nullptr, nullptr, nullptr, nullptr);
        ck(cudaEventRecord(t1), "rec");
        ck(cudaEventSynchronize(t1), "sync");
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, t0, t1), "elapsed");
        return (double) ms / iters;
    };

    const double ms8 = time(i8), ms4 = time(q4);
    // bytes a call reads from the pool: both planes of every selected cell
    const double bytes8 = (double) cells * H * D * 2 + (double) cells * H * G * 2 * 2;
    const double bytes4 = (double) cells * H * q4row * 2;
    std::printf("fused decode attention, %d cells, %d iterations, one query\n", cells, iters);
    std::printf("  int8: %8.1f us/call  %6.0f GB/s  (%.0f KiB read)\n", ms8 * 1000, bytes8 / (ms8 * 1e6),
                bytes8 / 1024);
    std::printf("  q4_0: %8.1f us/call  %6.0f GB/s  (%.0f KiB read)\n", ms4 * 1000, bytes4 / (ms4 * 1e6),
                bytes4 / 1024);
    std::printf("  q4_0 costs %.2fx the time of int8 for %.2fx the bytes\n", ms4 / ms8, bytes4 / bytes8);
    return 0;
}
