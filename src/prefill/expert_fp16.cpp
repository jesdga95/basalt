#include "experts.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/prefill/kernels.hpp"

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::Experts::Group::compute_fp16(size_t j, const uint8_t* blob_dev, int slot) {
    const int32_t e = order[j];
    const int q = (int) (j % DQ);
    if (lay.native) {
        // a native pack's layer, dequantized by llama.cpp's own formulas
        const auto& f = lay.fmt[(size_t) l];
        // NVFP4: the tail's FP32 scales {s_gate, s_up, s_down} go into the fp16 weights
        const float* tail = f.tail ? (const float*) (blob_dev + f.bytes - f.tail) : nullptr;
        basalt::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd,
                                           m.dq_gu[q], m.cs, tail);
        basalt::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, m.dq_d[q], m.cs,
                                        tail ? tail + 2 : nullptr);
    } else {
        blob_dequant_f16(blob_dev, m.dq_gu[q], m.dq_d[q], m.cs);
    }
    if (slot >= 0) { cudaEventRecord(m.used[slot], m.cs); m.used_of[slot] = slot; }
    const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
    m.gemm.f16(m.Xs + o0 * N, m.dq_gu[q], m.GU + o0 * 1280, ne, 1280, N);
    swiglu_interleaved(m.GU + o0 * 1280, m.Hh + o0 * 640, ne, m.cs);
    m.gemm.f16(m.Hh + o0 * 640, m.dq_d[q], m.Dm + o0 * N, ne, N, 640);
    return true;
}

}  // namespace basalt::prefill
