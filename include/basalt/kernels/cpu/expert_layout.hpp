// include/basalt/kernels/cpu/expert_layout.hpp - where each routed expert lives in the model's `experts` section.
//
// The experts keep their raw GGUF slices, so the blob size and the formats change from layer to layer; the
// `experts.layout` section says how.  Everything that touches an expert blob - the arena, the VRAM tier, the prompt
// path, the CPU pool, the GPU window - asks this table.
#pragma once

#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::kernels::cpu {

struct ExpertLayout {
    bool native = false;
    int64_t n_layers = 0, n_expert = NE;
    std::vector<NativeFmt> fmt;           ///< per layer
    std::vector<uint64_t> offset, bytes;  ///< per layer: where its 512 blobs start, bytes per blob
    int version = 0;                      ///< the layout header's version (0 = none given)
    uint64_t max_blob = BLOB;
    uint64_t total = 0;                   ///< the experts section's size

    uint64_t blob_bytes(int64_t layer) const { return native ? bytes[(size_t) layer] : (uint64_t) BLOB; }
    uint64_t layer_offset(int64_t layer) const {
        return native ? offset[(size_t) layer] : (uint64_t) layer * (uint64_t) n_expert * (uint64_t) BLOB;
    }
    uint64_t blob_offset(int64_t layer, int64_t expert) const {
        return layer_offset(layer) + (uint64_t) expert * blob_bytes(layer);
    }
};

/// Plan v0.3 P6: whether this CPU (and its OS) runs the AVX-512 kernels (F, BW, VL, VNNI, VBMI).  Probed in a
/// file compiled without AVX-512, so asking is safe everywhere.
bool cpu_avx512_ok();
/// Whether this CPU runs AVX-512F + AVX-512BW + AVX-512VL + AVX-512DQ (e.g. Skylake-X i9-9920X) for iq_avx512.cpp.
bool cpu_avx512bw_ok();
/// Whether this CPU (and its OS) runs the AVX2 kernels (AVX, AVX2, FMA, F16C): the floor of every expert kernel
/// (q2_avx2.cpp, iq_avx2.cpp, and the ggml dot products).
bool cpu_avx2_ok();
/// Whether this CPU (and its OS) also has the 256-bit AVX-VNNI instructions (`vpdpwssd`, `vpdpbusd`).
bool cpu_avxvnni_ok();
/// Whether this CPU (and its OS) runs AVX (Sandy Bridge and newer): the AVX1 router dot (kq_avx1.cpp).
bool cpu_avx1_ok();
/// SSE4.2 + POPCNT.
bool cpu_sse42_ok();
/// The cap on the probes above (3 = no cap, 2 = AVX2, 1 = AVX, 0 = SSE4.2); always 3.
int cpu_isa_cap();
/// Whether a layer of this ggml type runs on the engine's own Q2_0 kernels (AVX2 / AVX-512).
inline bool q2_native_kernels(int type) { return type == 42 && cpu_avx2_ok(); }
/// The CPU's brand string (CPUID 0x80000002..4), for messages; "unknown" when it has none.
std::string cpu_name();
/// Q2_0 GGUF rows / activation quantizer on the kernels this CPU has.
void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1);
void act_quant_any(const float* x, int n, ActQ& a);

/// The process-wide layout (empty until `expert_layout_load_text` succeeds).
const ExpertLayout& expert_layout();
/// Loads the layout from its text (the model's `experts.layout` section: "# experts vN: ..." then one line per layer,
/// `layer gu_type d_type offset blob_bytes`; further columns are ignored).  A newer version is refused.
bool expert_layout_load_text(const std::string& text, int64_t n_layers, int64_t n_expert, std::string& err);
/// The newest layout version this engine reads.
inline constexpr int kExpertLayoutVersion = 6;   // v6: NVFP4 blobs end in a 16-byte scale tail

}  // namespace basalt::kernels::cpu
