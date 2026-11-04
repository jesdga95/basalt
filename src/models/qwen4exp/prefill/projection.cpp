#include "state.hpp"
#include "layout.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/kernels.hpp"
#include "basalt/log.hpp"
#include <string>

namespace basalt::prefill {
using namespace detail;

namespace detail {
// The hc up product and the mix: one kernel from kUpMixMinT tokens up (the gate rows never reach memory), else the
// product into `gated` and gr_mix_r.  The same bits either way.
void up_mix(Gemm& gm, const uint16_t* w_up, const uint16_t* lo16, float* gated, const float* R, const float* rs,
            const float* w_norm, float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t T, void* stream) {
    const uint16_t* w = T >= kUpMixMinT ? gm.bf16_rows(w_up, D, LR) : nullptr;
    if (w != nullptr && gr_up_mix(lo16, w, R, rs, w_norm, mixed, mixed16, T, stream, mixed_h)) return;
    gm.bf16_w(lo16, w_up, gated, T, D, LR);
    gr_mix_r(R, rs, w_norm, gated, mixed, mixed16, T, stream, mixed_h);
}

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
}
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (the model pack must carry them)"; return false; }
    gm.native(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, ldy);
    return true;
}
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ldy) {
    if (w->kind != core::WeightKind::Bf16InF32 || !w->data) { err = "prefill: " + name + " is not a resident BF16 tensor"; return false; }
    gm.bf16_w(X, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy);
    return true;
}

}

}  // namespace basalt::prefill
