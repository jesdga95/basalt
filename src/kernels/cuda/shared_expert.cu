// src/kernels/cuda/shared_expert.cu - the shared expert.
//
// the Python reference, transcribed:
//
//     h = silu(x @ gate_shexp.T) * (x @ up_shexp.T)      <- SILU GOES ON GATE, not on up
//     h = h @ down_shexp.T                                (nt, n_embd)
//     g = sigmoid(x @ gate_inp_shexp)                     (nt,) ONE SCALAR PER TOKEN
//     return h * g[:, None]
//
// TWO THINGS HERE WERE BELIEVED WRONG FOR MANY ROUNDS and the model definition records both.  SILU GOES ON THE
// GATE TENSOR: the opposite reading is plausible, produces the right shapes, and is wrong.  And
// `ffn_gate_inp_shexp` is `(n_embd,)` whose output is "one value per token" - a SCALAR gate obtained by dotting
// that vector with the hidden state, NOT a per-expert gate and NOT a per-dimension elementwise one.  Both were
// live readings of the same shapes until the source comment settled them.
//
// The result is ADDED to the routed output - not router-weighted, not renormalised against it.
//
// The canonical projections use their explicit Q8_0/Q8_K activation images. The scalar gate reads the original F32
// input and uses the BF16 MMVF plus an FP32 sigmoid. Native projection overrides independently select CUDA Q8_1
// MMVQ and FP32 SwiGLU.
#include "basalt/kernels/shared_expert.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/bf16_bits.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/quantize_act.hpp"
#include "basalt/kernels/s2_gemv_q8.hpp"
#include "basalt/kernels/s_gemv.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <climits>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int THREADS = 128;

// The fp16 conversion lives in one place (`f16_bits.hpp`), with the regimes spelled out: a private copy that
// tests `exp >= 31` conflates an f32 inf/NaN with an out-of-range finite exponent and returns a NaN for every
// FINITE value that overflows fp16 (1e30, 65536, 1e45) instead of saturating to inf.

// silu(x) = x / (1 + exp(-x)), written as the Python reference writes it, in DOUBLE then cast - the reference works in
// float64 and a float32 exp differs in the last bits.  The multiplication by `up` is the reference's order.
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                              float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double x = (double) gate[i];
    out[i] = (float) (x / (1.0 + exp(-x))) * up[i];
}

__global__ void native_swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Pinned CUDA unary.cuh op_silu, then unary_gated_op_kernel's multiply.
    // Explicit intrinsics reproduce its --use_fast_math operations without
    // changing compilation of the canonical arithmetic in this file.
    out[i] = __fdividef(gate[i], 1.0f + __expf(-gate[i])) * up[i];
}

/// The MoE block's final combination.  See the header for the two readings it exists to pin.
__global__ void moe_combine_kernel(const float* __restrict__ parts, const float* __restrict__ weights,
                                   const float* __restrict__ shared, float* __restrict__ y, int n_embd,
                                   int k, int has_shared) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_embd) return;
    // Accumulate in DOUBLE, in the reference's own order (`for i in range(k): out[t] += w[t,i] * g[0]`).
    // f32 would be defensible for ten terms, but the reference is float64 and this is one line.
    double acc = 0.0;
    for (int e = 0; e < k; ++e) acc += (double) weights[e] * (double) parts[(size_t) e * n_embd + j];
    // The SHARED output is added PLAIN - not router-weighted, not renormalised against the routed sum.
    if (has_shared) acc += (double) shared[j];
    y[j] = (float) acc;
}

}  // namespace

namespace {
__global__ void sigmoid_scale_rows_kernel(float* __restrict__ out, const float* __restrict__ g, int n) {
    pdl_launch_dependents();
    pdl_wait();
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float gt = __fdividef(1.0f, 1.0f + __expf(-g[t]));
        out[(size_t) t * n + i] *= gt;
    }
}

constexpr bool fused_swiglu_q81_enabled() { return true; }
}  // namespace

void shared_expert_multi(int n_tok, const float* x, const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out, int64_t n_embd,
                         int64_t n_ff, void* stream) {
    if (n_tok < 1 || n_tok > 8 || !nw.q8_1 || !nw.gate_data || !nw.up_data || !nw.down_data || !stream)
        throw std::invalid_argument("shared_expert_multi: needs 1..8 tokens, native weights, scratch and a stream");
    cudaStream_t cs = (cudaStream_t) stream;
    native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
    const void* x_q8_1 = nw.q8_1;
    native_mmvq(nw.gate_type, nw.gate_data, x_q8_1, gate, (int) n_embd, (int) n_ff, n_tok, stream);
    native_mmvq(nw.up_type, nw.up_data, x_q8_1, up, (int) n_embd, (int) n_ff, n_tok, stream);
    native_swiglu_quantize_q8_1(gate, up, nw.q8_1, (int) n_ff, n_tok, stream);
    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);
    if (n_tok > 1) {   // one gemv for all rows (outputs identical), fused sigmoid+scale
        bf16_gemv_fp32_mmvf_multi(x, n_embd, gate_inp_bf16, g, 1, n_embd, 1, n_tok, stream);
        launch_pdl(sigmoid_scale_rows_kernel, dim3((unsigned) ((n_embd + THREADS - 1) / THREADS), (unsigned) n_tok),
                   dim3(THREADS), 0, cs, out, g, (int) n_embd);
    } else {
        bf16_gemv_fp32_mmvf(x, gate_inp_bf16, g, n_embd, 1, stream);
        sigmoid_scale_rows_kernel<<<dim3((unsigned) ((n_embd + THREADS - 1) / THREADS), 1u), THREADS, 0, cs>>>(
            out, g, (int) n_embd);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("shared_expert_multi: ") + cudaGetErrorString(e));
}

uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    // gate (n_ff f32) | up (n_ff f32) | q8_0 (n_ff/32*34) | q8k (n_ff/256*292) | g (1 f32), 16-byte aligned
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    return a * 2 + q0 + qk + 32;
}

void shared_expert(const uint8_t* x_q8_0, const uint8_t* x_q8k, const SForm& gate_form,
                   const uint8_t* gate_codes, const float* gate_scales, const float* gate_off,
                   const SForm& up_form, const uint8_t* up_codes, const float* up_scales, const float* up_off,
                   const SForm& down_form, const uint8_t* down_codes, const float* down_scales,
                   const float* down_off, const uint16_t* gate_inp_bf16, float* scratch, float* out,
                   int64_t n_embd, int64_t n_ff, int tpr, void* stream, const float* x_f32,
                   const NativeSharedWeights* native) {
    if (n_embd <= 0 || n_ff <= 0) return;
    const bool native_gate = native && native->gate_data && native_mmvq_supported(native->gate_type);
    const bool native_up = native && native->up_data && native_mmvq_supported(native->up_type);
    const bool native_down = native && native->down_data && native_mmvq_supported(native->down_type);
    const bool native_projection = native_gate || native_up || native_down;
    if (!x_f32) throw std::invalid_argument("shared_expert requires the unrounded x_f32");
    if (native_projection) {
        if (!native->q8_1 || !stream || n_embd > INT_MAX || n_ff > INT_MAX)
            throw std::invalid_argument("shared_expert native projections require scratch, stream and int32 dimensions");
        // Validate every active shape before any kernel is enqueued.
        if (native_gate) native_mmvq_weight_bytes(native->gate_type, (int) n_embd, (int) n_ff);
        if (native_up) native_mmvq_weight_bytes(native->up_type, (int) n_embd, (int) n_ff);
        if (native_down) native_mmvq_weight_bytes(native->down_type, (int) n_ff, (int) n_embd);
    }
    if (scratch == nullptr) {
        basalt::log::error("basalt", "shared_expert: scratch is null; the caller owns it "
                             "(see shared_expert_scratch_bytes)");
        std::exit(1);
    }
    // CARVED FROM THE CALLER'S SCRATCH: allocation is illegal during stream capture and the token path
    // makes none.
    uint8_t* p = (uint8_t*) scratch;
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    float* gate = (float*) p;
    float* up = (float*) (p + a);
    uint8_t* h_q8_0 = (uint8_t*) (p + a * 2);
    uint8_t* h_q8k = (uint8_t*) (p + a * 2 + q0);
    float* g = (float*) (p + a * 2 + q0 + qk);

    // WHICH ACTIVATION THIS PROJECTION WANTS, READ FROM ITS OWN FORM.  See `SForm::act_kind`: the three
    // families cannot be told apart by the other fields, so the kind is carried rather than derived.
    auto gemv = [&](const SForm& f, const uint8_t* codes, const float* scales, const float* off,
                    const uint8_t* act80, const uint8_t* actq8k, float* y, int64_t nin, int64_t nout) {
        if (f.code_bits == 2) {
            s2_gemv_q8(act80, codes, scales, y, nin, nout, tpr, stream);
        } else if (f.act_kind == 1) {
            s_gemv_q8k_split(actq8k, codes, scales, off, y, nin, nout, f, stream);
        } else {
            s_gemv_q8_0_split(act80, codes, scales, off, y, nin, nout, f, stream);
        }
    };

    const unsigned g_ff = (unsigned) ((n_ff + THREADS - 1) / THREADS);
    const unsigned g_embd = (unsigned) ((n_embd + THREADS - 1) / THREADS);

    // gate and up projections, then silu(gate) * up in place in `gate`
    if (native_gate || native_up)
        native_quantize_q8_1(x_f32, native->q8_1, (int) n_embd, 1, stream);
    if (native_gate)
        native_mmvq(native->gate_type, native->gate_data, native->q8_1, gate, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(gate_form, gate_codes, gate_scales, gate_off, x_q8_0, x_q8k, gate, n_embd, n_ff);
    if (native_up)
        native_mmvq(native->up_type, native->up_data, native->q8_1, up, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(up_form, up_codes, up_scales, up_off, x_q8_0, x_q8k, up, n_embd, n_ff);
    if (native_projection && native_down && fused_swiglu_q81_enabled()) {
        native_swiglu_quantize_q8_1(gate, up, native->q8_1, (int) n_ff, 1, stream);
        native_mmvq(native->down_type, native->down_data, native->q8_1, out, (int) n_ff, (int) n_embd, 1, stream);
    } else {
        if (native_projection)
            native_swiglu_kernel<<<g_ff, THREADS, 0, (cudaStream_t) stream>>>(gate, up, gate, (int) n_ff);
        else
            swiglu_kernel<<<g_ff, THREADS, 0, (cudaStream_t) stream>>>(gate, up, gate, (int) n_ff);

        // down: (n_ff) -> (n_embd), and THE INTERMEDIATE IS QUANTIZED TO THE DOWN WEIGHT'S OWN CONTRACT - which is
        // what `ggml_mul_mat` does for every matmul in the model.
        if (native_down) {
            native_quantize_q8_1(gate, native->q8_1, (int) n_ff, 1, stream);
            native_mmvq(native->down_type, native->down_data, native->q8_1, out, (int) n_ff, (int) n_embd, 1, stream);
        } else if (down_form.act_kind == 1) {
            if (n_ff % 256 != 0) {
                basalt::log::error("basalt", "shared_expert: the down weight wants Q8_K but n_ff %lld is not a multiple "
                                     "of 256; Q8_K is structurally impossible here", (long long) n_ff);
                std::exit(1);
            }
            quantize_q8_K(gate, h_q8k, n_ff, stream);
            gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
        } else if (down_form.code_bits == 2) {
            quantize_q8_0(gate, h_q8_0, n_ff, stream);
            gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
        } else {
            quantize_q8_0(gate, h_q8_0, n_ff, stream);
            gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
        }
    }

    // the per-token scalar gate, then the multiply.  Note the gate is computed from `x`, the ORIGINAL hidden
    // state, not from anything the expert produced.  The dot is the FP32-activation BF16 MMVF.
    bf16_gemv_fp32_mmvf(x_f32, gate_inp_bf16, g, n_embd, 1, stream);
    sigmoid_scale_rows_kernel<<<dim3(g_embd, 1u), THREADS, 0, (cudaStream_t) stream>>>(out, g, (int) n_embd);

    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            basalt::log::error("basalt", "shared_expert: %s", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

void moe_combine(const float* parts, const float* weights, const float* shared, float* y, int64_t n_embd,
                 int64_t k, void* stream) {
    if (n_embd <= 0 || k <= 0) return;
    // k > 64 is refused rather than truncated: silently summing the first 64 of a longer list would be a
    // wrong answer that looks like a right one, and no geometry in this artifact comes close to it.
    if (k > 64) {
        basalt::log::error("basalt", "moe_combine: k = %lld exceeds 64", (long long) k);
        std::exit(1);
    }
    const unsigned grid = (unsigned) ((n_embd + THREADS - 1) / THREADS);
    moe_combine_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(parts, weights, shared, y, (int) n_embd,
                                                                   (int) k, shared != nullptr);
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            basalt::log::error("basalt", "moe_combine: %s", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

}  // namespace basalt::kernels
