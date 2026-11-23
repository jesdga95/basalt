// src/kernels/cuda/window_handoff.cu - see window_handoff.hpp.
#include "basalt/kernels/window_handoff.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>

namespace basalt::kernels {
namespace {

__global__ void wsum_kernel(const float* const* __restrict__ rows, const float* __restrict__ w, int k, int n, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    float acc = 0.f;
    for (int e = 0; e < k; ++e) acc += w[t * k + e] * rows[t * k + e][i];
    out[(size_t) t * n + i] = acc;
}

struct PutWords {
    int w[kPutWords];
};

__global__ void put_words_kernel(const PutWords s, int* dst, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = s.w[i];
}

__global__ void wait_word_mark_kernel(const volatile uint32_t* word, uint32_t value, volatile uint32_t* mark) {
    while (*word != value) __nanosleep(100);
    __threadfence_system();
    *mark = value;
}

constexpr int kReadAheadBlocks = 32;              // blocks that request the ranges
constexpr uint64_t kReadAheadStep = 128 << 10;    // bytes a block requests before it looks at the mark again

// Block b requests steps b, b + blocks, ... of the ranges laid end to end, a step's pieces as bulk L2 prefetches; it then
// reads the step's last 16 bytes (they land once the step is in L2), so each block keeps one step in flight.
__device__ void l2_read_ahead_block(const L2Ranges& r, uint32_t* mark, uint32_t value, int b, int blocks) {
    uint64_t total = 0;
    for (int i = 0; i < kL2Ranges; ++i) total += r.n[i];
    uint32_t acc = 0;
    for (uint64_t o0 = (uint64_t) b * kReadAheadStep; o0 < total && *(volatile uint32_t*) mark != value;
         o0 += (uint64_t) blocks * kReadAheadStep) {
        const uint64_t o1 = min(o0 + kReadAheadStep, total);
        const char* last = nullptr;
        for (uint64_t o = o0; o < o1;) {
            int i = 0;
            uint64_t ro = o;
            while (i < kL2Ranges - 1 && ro >= r.n[i]) {
                ro -= r.n[i];
                ++i;
            }
            const uint64_t take = min(min(o1 - o, r.n[i] - ro), (uint64_t) 32768);
            const char* src = (const char*) r.p[i] + ro;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"((unsigned) take) : "memory");
            last = src + take - 16;
            o += take;
        }
        const uint4 w = __ldcg((const uint4*) last);
        acc ^= w.x ^ w.y ^ w.z ^ w.w;
    }
    if (acc == 0x9e3779b9u && value == 0x9e3779b9u) mark[1] = acc;   // keeps the reads (never in practice)
}

__global__ void __launch_bounds__(32) l2_read_ahead_kernel(const L2Ranges r, uint32_t* mark, uint32_t value) {
    if (threadIdx.x == 0) l2_read_ahead_block(r, mark, value, (int) blockIdx.x, (int) gridDim.x);
}

}  // namespace

void weighted_sum_window(const float* const* rows, const float* w, int n_tok, int K, int n, float* out, void* stream) {
    wsum_kernel<<<dim3((n + 255) / 256, n_tok), 256, 0, (cudaStream_t) stream>>>(rows, w, K, n, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void put_words(const void* src, void* dst, int n_words, void* stream) {
    if (n_words < 1 || n_words > kPutWords) throw std::invalid_argument("put_words: n_words");
    PutWords s;
    std::memcpy(s.w, src, (size_t) n_words * 4);
    put_words_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(s, (int*) dst, n_words);
}

void wait_word_mark(const uint32_t* word, uint32_t value, uint32_t* mark, void* stream) {
    wait_word_mark_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(word, value, mark);
}

void l2_read_ahead(const L2Ranges& r, uint32_t* mark, uint32_t value, void* stream) {
    if (r.n[0] == 0) return;
    l2_read_ahead_kernel<<<kReadAheadBlocks, 32, 0, (cudaStream_t) stream>>>(r, mark, value);
}

}  // namespace basalt::kernels
