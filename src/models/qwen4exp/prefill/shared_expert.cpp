#include "run.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/prefill/kernels.hpp"

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::shared_expert(const core::LayerView& v, const core::WeightRef* wgi, const core::WeightRef* wsg,
                                              const core::WeightRef* wsu, const core::WeightRef* wsd) {
    // the shared expert and its scalar gate
    if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
    if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
    swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);
    if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
    if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
    m.gemm.bf16(m.mixed_bf, (const uint16_t*) wgi->data, m.sg, T, 1, N);
    return true;
}

}  // namespace basalt::prefill
