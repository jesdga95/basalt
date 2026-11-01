#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

// The attention projections' weights (fc_embedding, k, v, q, o) into L2 by the prefetch stream, from this point of
// `from` on, while the chain's latency-bound kernels run; joined back by end_branches.
void MtpDrafter::pf_fork(cudaStream_t from) {
    if (!par_) return;
    const ModelGeometry& g = *g_;
    const int64_t N = g.n_embd, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    auto q8_bytes = [&](int64_t rows, int64_t cols) { return (size_t) (rows * (cols / 32) * 34); };
    auto pf = [&](const void* w, size_t bytes) { basalt::kernels::dense_tc_l2_prefetch(w, bytes, cs3_); };
    cudaEventRecord(ev_pf_, from);
    cudaStreamWaitEvent(cs3_, ev_pf_, 0);
    pf(q8("fc_embedding.weight"), q8_bytes(N, N));
    pf(q8("self_attn.k_proj.weight"), q8_bytes(NKV * HD, N));
    pf(q8("self_attn.v_proj.weight"), q8_bytes(NKV * HD, N));
    pf(q8("self_attn.q_proj.weight"), q8_bytes(NH * 2 * HD, N));
    pf(q8("self_attn.o_proj.weight"), q8_bytes(N, NH * HD));
    pf_open_ = true;
}

void MtpDrafter::prefetch_end(cudaStream_t cs) {
    if (!pf_open_) return;
    cudaEventRecord(ev_pfj_, cs3_);
    cudaStreamWaitEvent(cs, ev_pfj_, 0);
    pf_open_ = false;
}

// The end of chain step j: its token and probability to the mapped outputs and the next step's token, in one launch
// that also sets the conditional handles this step ends with (the next step's input residual is the hidden input
// already computed beside the head, so the residual is not carried).
bool MtpDrafter::step_end(int j) {
    using namespace basalt::kernels;
    const CondTail* c = ctail_;
    if (!mtp_step_end(out_ids_, row_ + 1, tok_, m_out_, j, probs_, m_prob_, c ? c->h : nullptr, c ? c->n : 0, row_ + 3,
                      m_prob_, cs_))
        return false;
    ctail_done_ = c != nullptr;
    return true;
}

int MtpDrafter::diag_top(int j, int k, int32_t* ids, float* probs) const {
    if (diag_ == nullptr || j < 0 || j >= max_t_) return 0;
    const bool sub = dvocab_ != nullptr;
    const int64_t nv = sub ? (tier_ == 0 ? n_tier1_ : n_dvocab_) : n_vocab_;
    const float* l = diag_ + (size_t) j * (size_t) diag_stride_;
    std::vector<std::pair<float, int64_t>> best;
    float mx = l[0];
    for (int64_t i = 1; i < nv; ++i) mx = std::max(mx, l[i]);
    double sum = 0.0;
    for (int64_t i = 0; i < nv; ++i) {
        sum += std::exp((double) (l[i] - mx));
        if ((int) best.size() < k || l[i] > best.back().first) {
            best.emplace_back(l[i], i);
            std::sort(best.begin(), best.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
            if ((int) best.size() > k) best.pop_back();
        }
    }
    for (size_t i = 0; i < best.size(); ++i) {
        const int64_t r = best[i].second;
        ids[i] = sub && (size_t) r < diag_map_.size() ? diag_map_[(size_t) r] : (int32_t) r;
        probs[i] = (float) (std::exp((double) (best[i].first - mx)) / sum);
    }
    return (int) best.size();
}

void MtpDrafter::end_branches() {
    if (fork_open_) join(cs_);
    prefetch_end(cs_);
    fork_open_ = par_ = pre_next_ = have_pre_ = false;
}


}
