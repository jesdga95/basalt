#include "run.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/prefill/kernels.hpp"

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::gdn(const core::LayerView& v) {
    // ======================= GDN =======================
    const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                          *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                          *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                          *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                          *wsa = need(v, "ssm_a", err);
    if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
    float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
    const Gemm::Rows8* xa = act8(1 | 2, m.mixed_h, N, m.bo, cap_bo, x8);
    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err, 0, kind8(1, xa))) return false;
    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err, 0, kind8(2, xa))) return false;
    if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV)) return false;
    if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV)) return false;
    gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);
    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, m.cs);
    // the conv input is dead after the conv: the recurrence's scratch
    gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.qkv, m.y, m.y_h, T, m.cs);
    const Gemm::Rows8* ya = act8(4, m.y_h, ZV, m.y, cap_zv, y8);
    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err, 0, ya)) return false;
    ++gdn_index;
    return true;
}

}  // namespace basalt::prefill
