#include "run.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/kernels/cvec.hpp"
#include "basalt/prefill/kernels.hpp"
#include <algorithm>
#include <string>

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::layer(int64_t l) {
    core::progress_beat();   // the serve watchdog: a prompt chunk of 8192 tokens is still moving
    if (owner.on_layer && LB == 0 && owner.next_ == nullptr) owner.on_layer(l, g.n_layers, T, p0);
    core::progress_at("reading the prompt (batched): layer", l, p0);   // a stall names layer and chunk
    const core::LayerView v(*m.wt, l);
    if (l == std::max<int64_t>(LB, 1) && !ple_land()) return false;   // the PLE rows, read from layer 1 on
    if (l == 1 && LB == 0 && ple_on) copy_ahead(1);
    // ---- the PLE block at layer 1, token by token (its conv reads the previous tokens' rows)
    if (!ple_layer(l)) return false;
    for (int half = 0; half < 2; ++half) {
        // ---- the hyper-connection read of this half
        const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
        const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                          su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
        const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                              *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
        if (!wn || !wd || !wu || !wi) return false;
        if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.grs, m.xn16, T, m.cs);
        normed = false;
        if (!bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err)) return false;
        gr_silu(m.lo, m.lo16, T, m.cs);
        if (wu->kind != core::WeightKind::Bf16InF32 || !wu->data || wu->ne1 != D || wu->ne0 != LR) {
            err = "prefill: " + su + " is not a resident BF16 tensor of the hc shape";
            return false;
        }
        // the attention half reads only the BF16 and FP16 images of the mixed rows
        up_mix(m.gemm, (const uint16_t*) wu->data, m.lo16, m.gated, m.R, m.grs, (const float*) wn->data,
               half == 0 ? nullptr : m.mixed, m.mixed_bf, m.mixed_h, T, m.cs);
        if (!bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err)) return false;

        if (half == 0 && !core::is_qsa_layer(g, l)) {
            if (!gdn(v)) return false;
        } else if (half == 0) {
            if (!qsa(v)) return false;
        } else {
            if (!moe(l, v)) return false;
        }
        // ---- the hyper-connection write of this half; fused with the next half's norm when nothing else
        // touches R in between (not the stage's last half, not before the PLE block of layer 1, not under a
        // control vector)
        const int64_t nl = half == 0 ? l : l + 1;
        const bool fuse = nl < LE && !(half == 1 && nl == 1 && ple_on) &&
                          !(half == 1 && basalt::kernels::cvec().covers(l));
        const core::WeightRef* wnn = nullptr;
        if (fuse) {
            const core::LayerView vn(*m.wt, nl);
            wnn = need(vn, half == 0 ? "hc_ffn_norm.weight" : "hc_attn_norm.weight", err);
            if (!wnn) return false;
        }
        if (wnn) {
            gr_write_norm_rs(m.R, m.bo, m.inj, HC, (const float*) wnn->data, EPS, m.grs, m.xn16, T, m.cs);
            normed = true;
        } else {
            gr_write(m.R, m.bo, m.inj, HC, T, m.cs);
        }
        if (half == 1 && basalt::kernels::cvec().covers(l))   // --control-vector-scaled
            basalt::kernels::cvec_apply(m.R, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
    }
    return true;
}

}  // namespace basalt::prefill
