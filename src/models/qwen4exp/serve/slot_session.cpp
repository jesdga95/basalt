// src/models/qwen4exp/serve/slot_session.cpp - a batch slot as a session of the shared batch driver: its windows on the
// verifier, its kept rows committed at once, its drafts from the drafter's store of the slot.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_state.hpp"

namespace basalt::models::qwen4exp::serve {

using basalt::model::Result;
using basalt::model::Status;

namespace {
// stable insertion sort for the few slots a window holds
template <class Less>
void sort_small(int* v, int n, Less less) {
    for (int i = 1; i < n; ++i) {
        const int x = v[i];
        int j = i;
        for (; j > 0 && less(x, v[j - 1]); --j) v[j] = v[j - 1];
        v[j] = x;
    }
}
}  // namespace

void QfnSlotSession::reset() {
    cum_ = basalt::core::DraftCumAdapt{};
    acc_ = 0.6;
    since_draft_ = 0;
    n_dr_ = 0;
}

int QfnSlotSession::batch_rows() const { return std::min(basalt::kernels::kVerifyMaxT, rt_.ver.max_t()); }
int QfnSlotSession::max_window() const { return rt_.batch_spec ? rt_.S : 1; }
int64_t QfnSlotSession::batch_context() const { return rt_.slot_cap[(size_t) b_]; }
basalt::model::BatchShared& QfnSlotSession::batch_shared() { return rt_.slot_shared; }
cudaStream_t QfnSlotSession::batch_stream() const { return rt_.ver.stream(); }

// the last window's commit lands before this window's rows are staged
bool QfnSlotSession::batch_reserve(const basalt::model::DecodeRequest&, std::string& err) { return rt_.commit_settle(err); }

bool QfnSlotSession::batch_forward(const basalt::model::BatchView& v, std::string& err) {
    QfnSlotSession* ss[basalt::kernels::kVerifyMaxT] = {};
    if (v.n > basalt::kernels::kVerifyMaxT) { err = "a batch of more slots than the verifier's window"; return false; }
    for (int i = 0; i < v.n; ++i) ss[i] = static_cast<QfnSlotSession*>(v.ss[i]);   // the driver checked they are this model's
    return rt_.slot_windows(ss, v.reqs, v.n, err);
}

void QfnSlotSession::batch_take(const basalt::model::BatchView& v, int i) {
    const basalt::model::DecodeRequest& r = v.reqs[i];
    if (r.logits != nullptr) rt_.ver.copy_logits(r0_ + rows_ - 1, r.logits);
}

// the window's picks are on the host already (the verifier draws every row with the slot's sampling)
const int32_t* QfnSlotSession::batch_draw(const basalt::model::BatchView& v, int i, std::string&) {
    return rt_.win_out + r0_ + rows_ - v.head[i];
}

Result QfnSlotSession::decode_window(const basalt::model::DecodeRequest& r) {
    QfnSlotSession* self = this;
    std::string err;
    if (!rt_.slot_windows(&self, &r, 1, err)) return {Status::Error, err};
    if (r.row_tokens != nullptr) std::copy(rt_.win_out, rt_.win_out + r.n_tokens, r.row_tokens);
    if (r.next_token != nullptr) *r.next_token = rt_.win_out[r.n_tokens - 1];
    if (r.logits != nullptr && !rt_.ver.copy_logits(r.n_tokens - 1, r.logits)) return {Status::Error, "qwen4exp: the logits read failed"};
    return {};
}

// The prompt through the slot's own windows, each kept whole: exact and slow, for the batch check (a served prompt is read
// by the runtime's prompt path and copied into the slot).
Result QfnSlotSession::prefill_chunk(const basalt::model::PrefillRequest& r) {
    std::vector<int32_t> picks((size_t) batch_rows());
    for (int at = 0; at < r.n_tokens;) {
        basalt::model::DecodeRequest d;
        d.tokens = r.tokens + at;
        d.n_tokens = std::min(batch_rows(), r.n_tokens - at);
        d.pos = r.pos + at;
        d.row_tokens = picks.data();
        basalt::model::ModelSession* self = this;
        if (Result e = decode_window(d); !e.ok()) return e;
        if (Result e = decode_kept(&self, &d.n_tokens, 1); !e.ok()) return e;
        at += d.n_tokens;
    }
    std::string err;
    return rt_.commit_settle(err) ? Result{} : Result{Status::Error, err};
}

// The rows each segment of the last window keeps: the drafter's length rule and draft share learn from them, then the
// commit, beside the drafts when they follow (draft_batch waits for it), else at once.
Result QfnSlotSession::decode_kept(basalt::model::ModelSession* const* sessions, const int* kept, int n) {
    QfnRuntime& rt = rt_;
    if (n != rt.win_segs) return {Status::Error, "qwen4exp: the kept rows do not match the last window's segments"};
    int nk[basalt::kernels::kVerifyMaxT] = {};
    for (int i = 0; i < n; ++i) {
        QfnSlotSession& s = *static_cast<QfnSlotSession*>(sessions[i]);
        if (kept[i] < 1 || kept[i] > s.rows_) return {Status::Error, "qwen4exp: a kept row count out of range"};
        nk[i] = s.kept_ = kept[i];
        const int a = kept[i] - 1, Tw = s.rows_;
        if (Tw > 1 && s.own_) s.cum_.observe(s.dp_, Tw - 1, a);
        if (Tw > 1) s.acc_ = 0.8 * s.acc_ + 0.2 * ((double) a / (double) (Tw - 1));
        rt.bt_slotwin += 1;
        rt.bt_tokens += kept[i];
        rt.bt_dr_off += Tw - 1;
        rt.bt_dr_acc += a;
        if (rt.logit_hash)
            for (int e = 0; e < kept[i]; ++e)
                basalt::log::raw("LH %d %lld %016llx\n", s.b_, (long long) (s.pos0_ + e), (unsigned long long) rt.row_hash(s.r0_ + e));
    }
    std::string err;
    const Clock::time_point c0 = Clock::now();
    if (rt.batch_spec) {
        if (!rt.ver.commit_slots_begin(nk, err)) return {Status::Error, err};
        rt.commit_open = true;
    } else if (!rt.ver.commit_slots(nk, err)) {
        return {Status::Error, err};
    }
    rt.bt_commit += std::chrono::duration<double, std::milli>(Clock::now() - c0).count();
    return {};
}

Result QfnSlotSession::draft_begin(int32_t anchor, int64_t pos, int n) {
    basalt::model::ModelSession* self = this;
    return draft_batch(&self, &anchor, &pos, &n, 1);
}

// The next windows' drafts of the slots that go on, one slot at a time on its own draft K/V, beside the last window's
// commit.  When not every slot can verify drafts (the window holds as many rows as slots), the slots likeliest to use
// them draft and the others only advance their draft K/V.
Result QfnSlotSession::draft_batch(basalt::model::ModelSession* const* peers, const int32_t* anchors, const int64_t* pos, const int*, int count) {
    constexpr int MR = basalt::kernels::kVerifyMaxT;
    QfnRuntime& rt = rt_;
    if (!rt.batch_spec) return {Status::NotImplemented, "qwen4exp: the slots draft nothing"};
    if (count < 1 || count > MR) return {Status::Error, "qwen4exp: drafts for more slots than a window holds"};
    QfnSlotSession* ss[MR] = {};
    for (int i = 0; i < count; ++i) {
        ss[i] = static_cast<QfnSlotSession*>(peers[i]);
        if (pos[i] != ss[i]->pos0_ + ss[i]->kept_) return {Status::Error, "qwen4exp: drafts after a window the slot did not keep"};
    }
    const Clock::time_point d0 = Clock::now();
    std::string err;
    int ord[MR];
    for (int i = 0; i < count; ++i) ord[i] = i;
    sort_small(ord, count, [&](int x1, int x2) {
        return ss[x1]->acc_ + 0.05 * (double) ss[x1]->since_draft_ > ss[x2]->acc_ + 0.05 * (double) ss[x2]->since_draft_;
    });
    const int nfull = std::min(count, MR - count);
    const float* Rw = rt.mtp.keep_rows(rt.ver.final_R_all(), rt.win_rows, err);
    if (Rw == nullptr) return {Status::Error, err};
    const int64_t HCN = (int64_t) rt.g.hc * rt.g.n_embd;
    for (int ci = 0; ci < count; ++ci) {
        QfnSlotSession& s = *ss[ord[ci]];
        const float* rows = Rw + (size_t) s.r0_ * HCN;
        int32_t picks[MR] = {};
        for (int r = 0; r < s.rows_; ++r) picks[r] = rt.win_out[s.r0_ + r];
        if (anchors != nullptr && s.kept_ > 0 && s.kept_ <= MR) picks[s.kept_ - 1] = anchors[ord[ci]];
        rt.mtp.set_ple_session(rt.bslot_ss[0][(size_t) s.b_].get());
        s.n_dr_ = 0;
        if (ci < nfull) {
            rt.mtp.set_cum_th(s.cum_.th());
            int nd = 0;
            if (!rt.mtp.draft_seq(s.b_, rows, s.rows_, picks, s.pos0_, s.kept_ - 1, s.dr_, err, s.dp_, &nd))
                return {Status::Error, err};
            s.n_dr_ = nd;
            s.since_draft_ = 0;
        } else {
            ++s.since_draft_;
            if (!rt.mtp.defer_rows(s.b_, rows, picks, s.kept_, s.pos0_, err)) return {Status::Error, err};
        }
    }
    rt.mtp.set_ple_session(rt.rsp);
    if (!rt.mtp.idle(err)) return {Status::Error, err};   // (the kept rows are copied)
    rt.bt_draft += std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
    if (!rt.commit_settle(err)) return {Status::Error, err};
    return {};
}

// every draft the chain made (a lookup proposal is matched against the first)
Result QfnSlotSession::draft_end(int32_t* ids, float* conf, int max_n, int& n, const DraftNeed&) {
    n = std::min(n_dr_, max_n);
    std::copy(dr_, dr_ + n, ids);
    std::copy(dp_, dp_ + n, conf);
    return {};
}

// the drafts the draft-length rule keeps, as the chain applied it, within the drafter's windows
int QfnSlotSession::own_window(const float* chance, int n) const {
    int T = 1;
    while (T < rt_.S_mtp && T - 1 < n && basalt::core::draft_keep(chance, T - 1, cum_.th())) ++T;
    return T;
}

Result QfnSlotSession::row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) {
    if (row0 < 0 || row0 + n > rows_) return {Status::Error, "qwen4exp: logprobs of rows outside the slot's window"};
    if (!rt_.ver.row_logprobs(r0_ + row0, n, picks, top, out)) return {Status::Error, "logprobs: the head logits read failed"};
    return {};
}

Result QfnSlotSession::copy_logits(int row, float* out) {
    if (row < 0 || row >= rows_ || !rt_.ver.copy_logits(r0_ + row, out)) return {Status::Error, "qwen4exp: the logits read failed"};
    return {};
}

Result QfnSlotSession::resample_row(int row, const float* bias, int32_t& token) {
    if (row < 0 || row >= rows_) return {Status::Error, "qwen4exp: a draw outside the slot's window"};
    const std::vector<float> b(bias, bias + rt_.n_vocab);
    std::string err;
    if (!rt_.ver.resample_row(r0_ + row, pos0_, b, token, err)) return {Status::Error, err};
    rt_.win_out[r0_ + row] = token;
    return {};
}

}  // namespace basalt::models::qwen4exp::serve
