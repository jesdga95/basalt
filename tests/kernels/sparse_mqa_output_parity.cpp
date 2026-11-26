// Shared grouped output projection against the exact sequence of native single-column MMVQ calls.
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/sparse_mqa.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { if (n) check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc"); }
    ~Device() { if (p) cudaFree(p); }
    void upload(const std::vector<T>& v) { check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
    std::vector<T> download(size_t n) const {
        std::vector<T> v(n);
        check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return v;
    }
};

struct Q80 { __half d; int8_t qs[32]; };
static_assert(sizeof(Q80) == 34);

std::vector<uint8_t> random_q8_weights(size_t rows, int cols, uint32_t seed) {
    std::mt19937 rng(seed);
    const size_t blocks = (size_t) cols / 32;
    std::vector<uint8_t> out(rows * blocks * sizeof(Q80));
    for (size_t i = 0; i < rows * blocks; ++i) {
        const __half d = __float2half(0.0007f);
        uint8_t* b = out.data() + i * sizeof(Q80);
        std::memcpy(b, &d, sizeof(d));
        for (int j = 0; j < 32; ++j) b[sizeof(d) + j] = (uint8_t) (int8_t) ((int) (rng() % 31) - 15);
    }
    return out;
}

}  // namespace

int main() {
    using namespace basalt::kernels;
    constexpr int T = 3, n_embd = 4096;
    constexpr int in = kSparseMqaHeads * kSparseMqaDim;
    constexpr int mid = kSparseMqaOutputGroups * kSparseMqaOutputRank;
    constexpr int group_in = kSparseMqaGroupInput;
    constexpr int rank = kSparseMqaOutputRank;
    constexpr int type = 8;  // Q8_0, native MMVQ fallback
    const size_t wa_bytes = native_mmvq_weight_bytes(type, group_in, kSparseMqaOutputGroups * rank);
    const size_t wb_bytes = native_mmvq_weight_bytes(type, mid, n_embd);
    auto wa = random_q8_weights(kSparseMqaOutputGroups * rank, group_in, 19);
    auto wb = random_q8_weights(n_embd, mid, 23);
    std::vector<float> x((size_t) T * in);
    std::mt19937 rng(29);
    for (float& v : x) v = ((int) (rng() % 2001) - 1000) * 0.001f;

    Device<uint8_t> dwa(wa_bytes), dwb(wb_bytes);
    dwa.upload(wa);
    dwb.upload(wb);
    Device<float> dx(x.size()), got_latent((size_t) T * mid), got((size_t) T * n_embd);
    dx.upload(x);
    Device<float> ref_latent((size_t) T * mid), ref((size_t) T * n_embd);
    Device<uint8_t> q8(sparse_mqa_output_scratch_bytes(T));
    Device<uint8_t> q8_ref(sparse_mqa_output_scratch_bytes(T));
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream), "stream");

    sparse_mqa_output(dx.p, type, dwa.p, type, dwb.p, n_embd, T, q8.p, got_latent.p, got.p, stream);
    native_quantize_q8_1(dx.p, q8_ref.p, in, T, stream);
    const size_t group_x = native_q8_1_bytes(group_in, 1);
    const size_t group_w = native_mmvq_weight_bytes(type, group_in, rank);
    for (int t = 0; t < T; ++t)
        for (int g = 0; g < kSparseMqaOutputGroups; ++g)
            native_mmvq(type, (const uint8_t*) dwa.p + (size_t) g * group_w,
                        (const uint8_t*) q8_ref.p + (size_t) (t * kSparseMqaOutputGroups + g) * group_x,
                        ref_latent.p + (size_t) t * mid + (size_t) g * rank, group_in, rank, 1, stream);
    native_quantize_q8_1(ref_latent.p, q8_ref.p, mid, T, stream);
    native_mmvq(type, dwb.p, q8_ref.p, ref.p, mid, n_embd, T, stream);
    check(cudaStreamSynchronize(stream), "sync");
    const auto y = got.download((size_t) T * n_embd);
    const auto z = ref.download((size_t) T * n_embd);
    const bool equal = std::memcmp(y.data(), z.data(), y.size() * sizeof(float)) == 0;
    std::printf("sparse_mqa_output T=%d n_embd=%d: bitwise native composition %s\n", T, n_embd, equal ? "yes" : "NO");
    cudaStreamDestroy(stream);
    return equal ? 0 : 1;
}
