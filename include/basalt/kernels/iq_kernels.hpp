// include/basalt/kernels/iq_kernels.hpp - the i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
// IQ4_NL) and Q2_0 on the GPU for the IQ2_XS / IQ3_XXS model files, and Q4_K / Q5_K / Q5_1 / Q8_0 for Unsloth's
// UD-Q4_K_XL (gate/up Q4_K or Q5_K, down Q5_1 or Q8_0, a Q8_0 embedding).
//
// The block layouts, codebook grids and dot products are llama.cpp's (ggml-common.h, ggml-cuda/vecdotq.cuh,
// ggml-cuda/dequantize.cuh; MIT, llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d, see THIRD_PARTY_LICENSES), so a weight means exactly what it
// means in llama.cpp.  Activations are q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

/// ggml type ids handled here.
bool iq_supported(int ggml_type) noexcept;
/// The token-embedding types iq_embed_rows and iq_dequant_f32 read: the i-quants above and BF16 (30).
bool embed_type_supported(int ggml_type) noexcept;
/// Bytes of one row of `n` values of `ggml_type` (n a multiple of the type's block).
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept;

/// q8_1 blocks for `n_rows` rows of `n_cols` floats (n_cols a multiple of 32): y is n_rows * n_cols/32 blocks.
void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream);

/// y[c][r] = W[r] . x[c] for `ncols` columns of q8_1 activations (x stride n_in/32 blocks per column).
void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// Dequantize `n` contiguous values (n a multiple of 256) to fp16 / fp32.  `scale` (device, NVFP4 only): the
/// matrix's FP32 scale, applied in fp32 before the fp16 rounding.
void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream,
                    const float* scale = nullptr);
void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream);
/// Rows `tokens[0..n_tok)` (device ids) of a GGUF embedding table (`row_bytes` per row; the table may be mapped
/// host memory) dequantized to fp32, `n_embd` per row (a multiple of 256).  `hot` (optional): a device copy of the
/// table's first `n_hot` rows, read instead of the table for those ids (the same bytes).
void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                   int64_t n_embd, float* out, void* stream, const void* hot = nullptr, int64_t n_hot = 0);
/// One expert's gate and up matrices (n_ff rows of n_embd each) into the interleaved fp16 layout the prompt path
/// uses: row 2r = gate row r, row 2r+1 = up row r.  `scale` (device, NVFP4 only): {s_gate, s_up}, as iq_dequant_f16.
void iq_dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                       void* stream, const float* scale = nullptr);

/// The layout of one native expert blob: [gate rows | up rows | down rows], raw GGUF blocks.
struct NativeExpertLayout {
    int gu_type = -1, d_type = -1;
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       // bytes per row
    size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
    size_t bytes = 0;                   // the whole blob (NVFP4: with its 16-byte FP32 scale tail)
    // APR (prism.hadamard): the down weights are Hadamard-folded, so native_expert_grouped rotates h before
    // quantizing it (hadamard_rows).  had_block 0 = none; d_signs is a DEVICE vector of n_ff signs or nullptr.
    int had_block = 0;
    const float* d_signs = nullptr;
};
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
namespace cpu { struct NativeFmt; }
/// A layer's layout with its Hadamard fold (had_block, d_signs).  The device sign vectors are made on first use
/// (hadamard_device_signs): call it for every layer at setup, before any graph capture.
NativeExpertLayout native_expert_layout(const cpu::NativeFmt& f);
/// The device sign vector of a layer's gate/up input when its gate/up are Hadamard-folded, else nullptr (the same
/// first-use rule).
const float* native_expert_gu_signs(const cpu::NativeFmt& f);
/// Whether `native_expert_grouped` has kernels for this gate/up and down type pair at these dimensions, and the
/// prompt path's dequantizer takes both (checked for every layer at startup, before anything is allocated).
bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;

/// Bytes of scratch `native_expert_grouped` needs for `cap_entries` entries.
size_t native_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff);

/// Grouped experts in the native format: group g's blob at device address grp_ptr[g]; its entries
/// [grp_start[g], grp_start[g+1]) read token ent_tok[e]'s q8_1 activation (n_embd/32 blocks per token in x_q8_1)
/// and write row ent_dst[e] of `out` (n_embd floats).  Counts are read on the device.
/// `grid_groups` (1 .. cap_groups; 0 = cap_groups) groups run side by side, a block row each striding over the rest:
/// a call that usually has few groups or none (the verify window's PCIe share) launches less for the ones it does
/// not have.  The results do not depend on it.
void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups = 0);
/// true: `native_expert_grouped`'s launches before the group stride (GROUPED_V1=1 at startup) - a block row
/// per possible group, SwiGLU and the q8_1 quantization as two kernels over all cap_entries, and gate/up on the dp4a
/// kernels even with GU_MMA=1.  Bitwise the same results (native_grouped_parity checks it); kept for A/B
/// timing.  Set before graph capture; captured graphs keep theirs.
void native_grouped_set_v1(bool v1);
/// GU_MMA=2: this thread's next grouped calls take the tensor-core gate/up (where it applies) when on
void native_grouped_set_mma_hint(bool on);
/// DEC_GU2: 1 the grid formats' gate/up rows on native_gu2_kernel, 2 also the SwiGLU pass inside the down
/// kernel (bitwise the same; not with v1).  Set before graph capture.
void native_grouped_set_gu2(int mode);

/// The bench only: the kernel layout (EXP_MODE values; -1 = the environment's) and the phase
/// (0 all, 1 gate/up + SwiGLU + quantize, 2 down).
void native_expert_set_mode(int mode, int phase);

/// APR (prism.hadamard): y = H_block (signs * x) per block of `block` values of each of `rows` rows of `width`
/// (x == y allowed).  `signs`: a device vector of `width` signs, or nullptr (identity).  The same floats as the CPU's
/// hadamard_rotate.  The f16 form rotates in place, computing in float.
void hadamard_rows(const float* x, float* y, int64_t rows, int64_t width, int block, const float* signs, void* stream);
void hadamard_rows_f16(uint16_t* x, int64_t rows, int64_t width, int block, const float* signs, void* stream);
/// hadamard_rows then quantize_q8_1_rows in one pass (the same q8_1 bytes); `block` a multiple of 32.
void hadamard_q8_1_rows(const float* x, int64_t rows, int64_t width, int block, const float* signs, void* y,
                        void* stream);
/// A device copy (on the current device) of the host sign vector `host` of `width` signs, made on first use and
/// kept for the process.  Call it at setup, not inside a CUDA graph capture.  nullptr for a nullptr `host`.
const float* hadamard_device_signs(const float* host, int width);
/// `iq_mmvq` and `native_expert_grouped` decode each weight part once and apply it to every column / entry;
/// true selects the older kernels that decode it again per column (OLD_IQ_MMVQ=1 at startup).  Both give
/// bitwise the same results.  Set before graph capture; captured graphs keep the kernels they captured.
void iq_set_old_kernels(bool old);
bool iq_old_kernels();
/// MMVQ_MMA=1 (read at startup; the setter overrides it, for tests): `iq_mmvq`'s IQ3_S calls of 2..8 columns
/// with n_in % 512 == 0 run on int8 tensor cores where the device has them (int8 MMA), never with the old kernels;
/// bitwise the same results.  Set before graph capture; captured graphs keep the kernels they captured.
void iq_set_mmvq_mma(bool on);
bool iq_mmvq_mma();
/// Whether the current device runs the tensor-core kernel.
bool iq_mmvq_mma_available();
/// Tensor-core launches so far in this process (tests: proof that the path ran).
long iq_mmvq_mma_launches();

}  // namespace basalt::kernels
