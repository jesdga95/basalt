#include "basalt/core/mtp.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/kernels/q5k_head.hpp"
#include "basalt/kernels/q6k_head.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = basalt::kernels;

namespace {
void cuda_ok(cudaError_t status) {
    if (status != cudaSuccess) {
        std::printf("FAIL: CUDA: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

std::vector<uint8_t> head_weights(int rows, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> bytes((size_t) rows * 2100);
    for (int b = 0; b < rows * 10; ++b) {
        uint8_t* block = bytes.data() + b * 210;
        for (int i = 0; i < 192; ++i) block[i] = (uint8_t) rng();
        for (int i = 192; i < 208; ++i) block[i] = (uint8_t) (1 + rng() % 7);
        const uint16_t scale = 0x2400;
        std::memcpy(block + 208, &scale, sizeof scale);
    }
    return bytes;
}
}  // namespace

namespace basalt::core {
struct MtpHeadFixture {
    static bool run() {
        constexpr int rows = 96, subset = 65;
        NativeHead head;
        head.device_ = 0;
        head.type_ = 14;
        head.n_in_ = 2560;
        head.n_out_ = rows;
        head.bytes_ = rows * 2100;
        cuda_ok(cudaMalloc(&head.weights_, head.bytes_));

        MtpDrafter drafter;
        drafter.device_ = 0;
        drafter.head_ = &head;
        drafter.rows_mapped_ = true;
        drafter.n_dvocab_ = drafter.n_tier1_ = drafter.xbase_ = subset;
        cuda_ok(cudaStreamCreateWithFlags(&drafter.cs_, cudaStreamNonBlocking));
        cuda_ok(cudaMalloc((void**) &drafter.dvocab_, subset * sizeof(int32_t)));
        int32_t map[subset];
        for (int i = 0; i < subset; ++i) map[i] = rows - 1 - i;
        cuda_ok(cudaMemcpy(drafter.dvocab_, map, sizeof map, cudaMemcpyHostToDevice));
        const size_t bytes = K::q5k_proxy_bytes(subset);
        cuda_ok(cudaMalloc((void**) &drafter.proxy_, bytes));
        cuda_ok(cudaMemset(drafter.proxy_, 0, bytes));
        cuda_ok(cudaMalloc(&drafter.proxy_scratch_, K::q5k_proxy_scratch_bytes(subset)));
        cuda_ok(cudaMemset(drafter.proxy_scratch_, 0, K::q5k_proxy_scratch_bytes(subset)));
        const auto base = head_weights(rows, 11);
        cuda_ok(cudaMemcpy(head.weights_, base.data(), base.size(), cudaMemcpyHostToDevice));
        K::q6k_proxy_pack(head.weights_, 0, subset, drafter.proxy_, subset, nullptr, drafter.dvocab_);
        cuda_ok(cudaDeviceSynchronize());

        void *expected_proxy = nullptr, *activation = nullptr;
        int* token = nullptr;
        float* probability = nullptr;
        cuda_ok(cudaMalloc(&expected_proxy, bytes));
        cuda_ok(cudaMemset(expected_proxy, 0, bytes));
        std::vector<uint8_t> x(80 * 36, 0);
        for (int b = 0; b < 80; ++b) {
            const uint16_t scale = 0x3c00;
            std::memcpy(x.data() + b * 36, &scale, sizeof scale);
            for (int i = 0; i < 32; ++i) x[b * 36 + 4 + i] = (uint8_t) (i % 7 - 3);
        }
        cuda_ok(cudaMalloc(&activation, x.size()));
        cuda_ok(cudaMemcpy(activation, x.data(), x.size(), cudaMemcpyHostToDevice));
        cuda_ok(cudaMalloc((void**) &token, sizeof(int)));
        cuda_ok(cudaMalloc((void**) &probability, sizeof(float)));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        cuda_ok(cudaStreamBeginCapture(drafter.cs_, cudaStreamCaptureModeGlobal));
        const bool supported = K::q6k_proxy_top1(drafter.proxy_, head.weights_, activation, subset, subset,
                                               drafter.dvocab_, token, probability, drafter.proxy_scratch_,
                                               drafter.cs_, true);
        cuda_ok(cudaStreamEndCapture(drafter.cs_, &graph));
        cuda_ok(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

        bool ok = supported;
        std::string err;
        // Independent compositions followed by restoration exercise the same captured addresses.
        for (unsigned seed : {11u, 29u, 47u, 11u}) {
            const auto weights = head_weights(rows, seed);
            cuda_ok(cudaMemcpy(head.weights_, weights.data(), weights.size(), cudaMemcpyHostToDevice));
            if (!drafter.refresh_shared_head(err)) {
                std::printf("FAIL: %s\n", err.c_str());
                ok = false;
                break;
            }
            std::vector<uint8_t> actual(bytes), expected(bytes);
            cuda_ok(cudaMemcpy(actual.data(), drafter.proxy_, bytes, cudaMemcpyDeviceToHost));
            K::q6k_proxy_pack(head.weights_, 0, subset, expected_proxy, subset, nullptr, drafter.dvocab_);
            cuda_ok(cudaDeviceSynchronize());
            cuda_ok(cudaMemcpy(expected.data(), expected_proxy, bytes, cudaMemcpyDeviceToHost));
            if (actual != expected) {
                std::printf("FAIL: mapped proxy differs from fresh head at seed %u\n", seed);
                ok = false;
            }
            cuda_ok(cudaGraphLaunch(exec, drafter.cs_));
            cuda_ok(cudaStreamSynchronize(drafter.cs_));
            int actual_token, expected_token;
            float actual_prob, expected_prob;
            cuda_ok(cudaMemcpy(&actual_token, token, sizeof(int), cudaMemcpyDeviceToHost));
            cuda_ok(cudaMemcpy(&actual_prob, probability, sizeof(float), cudaMemcpyDeviceToHost));
            K::q6k_proxy_top1(expected_proxy, head.weights_, activation, subset, subset, drafter.dvocab_,
                             token, probability, drafter.proxy_scratch_, drafter.cs_, true);
            cuda_ok(cudaStreamSynchronize(drafter.cs_));
            cuda_ok(cudaMemcpy(&expected_token, token, sizeof(int), cudaMemcpyDeviceToHost));
            cuda_ok(cudaMemcpy(&expected_prob, probability, sizeof(float), cudaMemcpyDeviceToHost));
            if (actual_token != expected_token || actual_prob != expected_prob) {
                std::printf("FAIL: captured draft scoring differs from fresh head at seed %u\n", seed);
                ok = false;
            }
        }
        cuda_ok(cudaGraphExecDestroy(exec));
        cuda_ok(cudaGraphDestroy(graph));
        for (void* p : {expected_proxy, activation, (void*) token, (void*) probability}) cuda_ok(cudaFree(p));
        return ok;
    }
};
}  // namespace basalt::core

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    cuda_ok(cudaSetDevice(0));
    const bool ok = basalt::core::MtpHeadFixture::run();
    std::printf("mtp_head_refresh: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
