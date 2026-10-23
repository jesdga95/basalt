// src/kernels/cuda/ple.cu - the PLE block's GPU half.
//
// See include/basalt/kernels/ple.hpp for the structure and for the `normalized`-is-the-conv-input finding.
//
// The arithmetic follows the captured CPU ggml graph (captured input and output vectors):
//
//   * `rms_norm` accumulates `(double)(x*x)` with the PRODUCT rounded in f32 first - that is literally what
//     `ggml_compute_forward_rms_norm_f32` does (`sum += (ggml_float)(x[i00]*x[i00])` with `ggml_float` =
//     double), and it is not the same as accumulating in f32 or as widening before the multiply.
//   * `silu` is `x / (1 + expf(-x))` in f32, which is `ggml_silu_f32`.
//   * the gate's `sqrt` and `sigmoid` are f32.
//
// ple_value is the single-token BF16/F32 MMVF, and the post-projection arithmetic is `native_ple_postops`.
#include "basalt/kernels/ple.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/quantize_act.hpp"
#include "basalt/kernels/s2_gemv_q8.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/native_ple_postops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int THREADS = 256;

bool overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    if (a == nullptr || b == nullptr || a_bytes == 0 || b_bytes == 0) return false;
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    return aa < bb ? bb - aa < a_bytes : aa - bb < b_bytes;
}

__global__ void history_advance_kernel(float* __restrict__ history, const float* __restrict__ normalized) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= NG_HC_DIM) return;
    float* column = history + (size_t) channel * NG_HIST;
    for (int row = 0; row + 1 < NG_HIST; ++row) column[row] = column[row + 1];
    column[NG_HIST - 1] = normalized[channel];
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "ple_block: %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void ple_history_advance(float* hist, const float* normalized, void* stream) {
    if (hist == nullptr || normalized == nullptr)
        throw std::invalid_argument("ple_history_advance: null history or normalized input");
    if (overlap(hist, (size_t) NG_HIST * NG_HC_DIM * sizeof(float),
                normalized, (size_t) NG_HC_DIM * sizeof(float)))
        throw std::invalid_argument("ple_history_advance: history and normalized input overlap");
    history_advance_kernel<<<(NG_HC_DIM + THREADS - 1) / THREADS, THREADS, 0, (cudaStream_t) stream>>>(hist, normalized);
    ck(cudaGetLastError(), "history advance launch");
}

bool ple_block_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

uint64_t ple_block_scratch_bytes() {
    // five hc_dim floats + n_embd + hc, then the Q8 activation image, each 16-byte aligned because `d_scratch` is
    // cast to `float*` and `d_act` to `uint8_t*` at those offsets.
    const size_t f = (size_t) (5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * sizeof(float);
    const size_t q = (size_t) (NG_N_EMBD / 32) * 34;
    return ((f + 15) & ~(size_t) 15) + ((q + 15) & ~(size_t) 15) + 256;
}

void ple_block(const float* emb, const float* hidden, const float* hist_rows, const PleWeights& w,
               PleOut& out, void* scratch, void* stream) {
    const bool native_key = w.key_native_data != nullptr && w.key_bf16 == nullptr;
    if (native_key && (!emb || !hidden || !hist_rows || !out.result || !scratch || !stream ||
                       !w.key_native_q8_1 ||
                       (w.key_native_type != 42 && w.key_native_type != 18 && w.key_native_type != 23 &&
                        w.key_native_type != 8)))
        throw std::invalid_argument("ple_block: native key requires Q2_0, IQ3_XXS, IQ4_XS or Q8_0 weights, input/output, private scratch and explicit stream");
    if (emb == nullptr || hidden == nullptr || hist_rows == nullptr || out.result == nullptr) return;
    const int n_embd = NG_N_EMBD, hc = NG_HC, hc_dim = NG_HC_DIM;
    static_assert(NG_N_EMBD == 2560 && NG_HC_DIM == 10240, "native PLE key geometry changed");
    const size_t float_bytes = (size_t) (5 * hc_dim + n_embd + hc) * sizeof(float);
    cudaStream_t st = (cudaStream_t) stream;

    // FIVE SEPARATE hc_dim BUFFERS, deliberately.  `key` and `normalized` are both needed at the end for the
    // oracle comparison, so aliasing one onto the other would silently return the WRONG `key`.  Two buffers of the same size look like an obvious saving and the only thing it saves is
    // 40 KB.
    // **THE WORKSPACE IS CARVED FROM THE CALLER'S, NOT ALLOCATED**: allocation and synchronization are errors
    // inside a stream capture, and the token path makes no allocations.
    const size_t q8_bytes = (size_t) (n_embd / 32) * 34;
    if (scratch == nullptr) {
        basalt::log::error("basalt", "ple_block: scratch is null; the caller owns it (see ple_block_scratch_bytes)");
        std::exit(1);
    }
    const struct Export { const float* pointer; size_t count; const char* name; } exports[] = {
        {out.key, (size_t) hc_dim, "key"}, {out.value, (size_t) n_embd, "value"},
        {out.gate, (size_t) hc, "gate"}, {out.gated, (size_t) hc_dim, "gated"},
        {out.normalized, (size_t) hc_dim, "normalized"}, {out.conv, (size_t) hc_dim, "conv"},
        {out.result, (size_t) hc_dim, "result"}
    };
    for (const auto& item : exports) {
        if (overlap(item.pointer, item.count * sizeof(float), scratch, (size_t) ple_block_scratch_bytes()))
            throw std::invalid_argument(std::string("ple_block: output ") + item.name + " overlaps scratch");
    }
    if (native_key) {
        const size_t native_bytes = native_q8_1_bytes(n_embd);
        const size_t weight_bytes = native_mmvq_weight_bytes(w.key_native_type, n_embd, hc_dim);
        if ((reinterpret_cast<uintptr_t>(w.key_native_data) & 3u) ||
            (reinterpret_cast<uintptr_t>(w.key_native_q8_1) & 3u))
            throw std::invalid_argument("ple_block: native key buffers require four-byte alignment");
        const struct Region { const void* pointer; size_t bytes; } regions[] = {
            {scratch, (size_t) ple_block_scratch_bytes()}, {emb, (size_t) n_embd * 4},
            {hidden, (size_t) hc_dim * 4}, {hist_rows, (size_t) NG_HIST * hc_dim * 4},
            {w.key_native_data, weight_bytes}, {w.value_bf16, (size_t) n_embd * n_embd * 2},
            {w.norm_key, (size_t) hc_dim * 4}, {w.norm_query, (size_t) hc_dim * 4},
            {w.norm_conv, (size_t) hc_dim * 4}, {w.conv1d_f16, (size_t) PLE_CONV_KERNEL * hc_dim * 2}
        };
        for (const auto& region : regions)
            if (overlap(w.key_native_q8_1, native_bytes, region.pointer, region.bytes))
                throw std::invalid_argument("ple_block: native key scratch overlaps workspace, input or weight");
        for (const auto& item : exports)
            if (overlap(w.key_native_q8_1, native_bytes, item.pointer, item.count * sizeof(float)))
                throw std::invalid_argument(std::string("ple_block: native key scratch overlaps output ") + item.name);
    }
    uint8_t* base = (uint8_t*) scratch;
    float* d_scratch = (float*) base;
    uint8_t* d_act = base + ((float_bytes + 15) & ~(size_t) 15);
    float* d_key = d_scratch;
    float* d_query = d_key + hc_dim;
    float* d_norm = d_query + hc_dim;
    float* d_gated = d_norm + hc_dim;
    float* d_conv = d_gated + hc_dim;
    float* d_value = d_conv + hc_dim;
    float* d_gate = d_value + n_embd;

    // ---- key = grouped_norm(ple_key @ emb). A GGUF-native key follows the Q8_1 MMVQ; the S2 key the Q8_0 path.
    if (w.key_bf16 != nullptr) {
        bf16_gemv_fp32_mmvf(emb, w.key_bf16, d_key, n_embd, hc_dim, stream);
    } else if (native_key) {
        native_quantize_q8_1(emb, w.key_native_q8_1, n_embd, 1, stream);
        native_mmvq(w.key_native_type, w.key_native_data, w.key_native_q8_1,
                    d_key, n_embd, hc_dim, 1, stream);
    } else {
        quantize_q8_0(emb, d_act, n_embd, st);
        s2_gemv_q8(d_act, w.key_codes, w.key_scales, d_key, n_embd, hc_dim, 8, st);
    }
    bf16_gemv_fp32_mmvf(emb, w.value_bf16, d_value, n_embd, n_embd, stream);

    // The key norm has a distinct destination; temporary query storage is reused for the normalized gated values
    // after the gate consumes it.
    NativePlePostopsBuffers buffers{d_query,d_norm,d_gate,d_gated,d_norm,d_conv,out.result};
    native_ple_postops(d_key,hidden,d_value,hist_rows,w,buffers,stream);
    const float* normalized_key = d_query;

    // ---- the intermediates the oracle comparison needs.  `key` is the NORMALISED key, because the source's
    //      `cb(key, ...)` capture is after `gnorm`; `value` is the projection before the gate.  Each stage the
    //      oracle records is reproduced here so a mismatch can be attributed instead of guessed at.
    if (out.key) ck(cudaMemcpyAsync(out.key, normalized_key, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "key");
    if (out.value)
        ck(cudaMemcpyAsync(out.value, d_value, n_embd * sizeof(float), cudaMemcpyDeviceToDevice, st), "value");
    if (out.gate) ck(cudaMemcpyAsync(out.gate, d_gate, hc * sizeof(float), cudaMemcpyDeviceToDevice, st), "gate");
    if (out.gated)
        ck(cudaMemcpyAsync(out.gated, d_gated, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "gated");
    if (out.normalized)
        ck(cudaMemcpyAsync(out.normalized, d_norm, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "norm");
    if (out.conv)
        ck(cudaMemcpyAsync(out.conv, d_conv, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "conv");

    ck(cudaGetLastError(), "launch");
    // **NO `cudaStreamSynchronize` HERE.**  It was there to make the function self-contained for the parity
    // test, and inside a capture it is an error - a caller that wants the result immediately synchronises
    // itself, and the engine's caller does not want that at all.
}

void ple_block_projected(const float* projected_key, const float* projected_value, const float* hidden,
                         const float* hist_rows, const PleWeights& w, PleOut& out, void* scratch, void* stream) {
    if (!projected_key || !projected_value || !hidden || !hist_rows || !out.result || !scratch) return;
    const int n_embd = NG_N_EMBD, hc = NG_HC, hc_dim = NG_HC_DIM;
    float* d_scratch = (float*) scratch;
    float* d_key = d_scratch;
    float* d_query = d_key + hc_dim;
    float* d_norm = d_query + hc_dim;
    float* d_gated = d_norm + hc_dim;
    float* d_conv = d_gated + hc_dim;
    float* d_value = d_conv + hc_dim;
    float* d_gate = d_value + n_embd;
    float* norm_dst = out.normalized ? out.normalized : d_norm;
    NativePlePostopsBuffers buffers{d_query, norm_dst, d_gate, d_gated, norm_dst, d_conv, out.result};
    native_ple_postops(projected_key, hidden, projected_value, hist_rows, w, buffers, stream);
}

}  // namespace basalt::kernels
