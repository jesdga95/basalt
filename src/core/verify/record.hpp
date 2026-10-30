#pragma once

#include "common.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/gr.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/dec_moe.hpp"
#include "basalt/kernels/fused_gr.hpp"
#include <string>
#include <vector>

namespace basalt::core {

struct Verifier::WindowRecorder {
    WindowRecorder(Verifier& v, int T, cudaStream_t cs, std::string& err) : v_(v), T(T), cs(cs), err(err) {}

    bool run();
    bool inputs();
    bool pre(int64_t l);
    bool gdn(int64_t l, const LayerView& v, bool read_q8, bool gs_split);
    bool qsa(int64_t l, const LayerView& v, bool read_q8);
    bool post(int64_t l);
    bool head();
    float* Rt(int t) const { return v_.R_ + (size_t) t * HC * N; }
    void stamp(int64_t l, int i) const {
        if (v_.prof_on_) kernels::gpu_stamp(v_.prof_, (int) (l * kProfPer + i), cs);
    }
    bool lazy_layer(int64_t l) const { return lazy && l < v_.le_ && gdn_idx[(size_t) l] > gdn_first; }

    Verifier& v_;
    int T;
    cudaStream_t cs;
    std::string& err;
    const ModelGeometry& g = *v_.g_;
    const WeightTable& wt = *v_.wt_;
    SessionState& ss = *v_.ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = v_.max_t_;
    const kernels::QsaShapes s = verify_detail::shapes_of(g);
    const kernels::GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) kernels::NG_HIST * kernels::NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready() && v_.ple_stage();
    const bool dm = v_.dm_ != nullptr && (!v_.batch_rec_ || v_.batch_dm_);   // the device-planned tail (one group)
    // each hyper-connection read (and the FFN read's router) as one kernel; the next read's w_norm is prefetched into
    // L2 at its end
    const bool dec_hc = kernels::hc_fused_supported();
    // the dense chain's kernels (pdl.hpp) may start during the previous kernel's tail.  Only where that previous kernel
    // is one of them or another plain kernel; never with the profile's stamps.
    const bool pdl = v_.pdl_rec_ && !v_.prof_on_;
    // the chain's kernels let their dependents in early (pdl.hpp), which stage their weights
    const bool pdl2 = pdl && kernels::pdl2_requested();
    const int hrow0 = v_.batch_rec_ ? v_.row_base_ : 0;   // a slot group's own hand-off rows

    bool ple_side = false;
    const int32_t* pos_k = nullptr;
    const int32_t* pos_i = nullptr;
    const int64_t HB = Verifier::handoff_floats(g);
    std::vector<int64_t> gdn_idx, qsa_idx;
    int64_t gdn_first = -1;
    bool lazy = false;
    kernels::GdnCommitLayers lazy_args{};
    bool lz_open = false;
    // ---------------------------------------------------------------- pre(l, group): up to the ring
    bool ring_fork = false;   // the group's ring stream is joined in post
    int mid_ord = 0;   // the mid chain's launches so far in this graph
    bool glue_on = false;   // post's gate/up launch plans the group's VRAM share
    bool sh_read = false;   // the FFN read gave the shared expert its gate and q8_1 rows; post's combine gates it
    kernels::DmRouteArgs glue_ring;   // the ring's work for post's gate/up launch

    bool fuse_head_gr = false;
};

}  // namespace basalt::core
