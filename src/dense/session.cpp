// src/dense/session.cpp - a conversation of the dense stack as a model::ModelSession.
#include "internal.hpp"

#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/log.hpp"
#include "basalt/program/expert_spill.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "basalt/vision/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace basalt::dense {
namespace {

using model::Result;
using model::Status;

Result fail(const std::string& m) { return {Status::Error, m}; }

}  // namespace

// One session's device state on the card of model `m` (its layers' cells and recurrent state; on the card that holds layer 0
// also the logits, the picks and the drafter's state), on the current device.
bool init_state(Model& m, SessionState& st, std::string& err) {
    const Geometry& g = m.g;
    if (cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&st.side, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&st.fork, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&st.join, cudaEventDisableTiming) != cudaSuccess) {
        err = "cannot create a CUDA stream";
        return false;
    }
    // the drafter's state, the logits and the picks on the card that holds layer 0
    const bool head = m.lb == 0, mtp = head && m.o.drafter == Drafter::Mtp, block = head && m.o.drafter == Drafter::Block;
    size_t blockbytes = 0;
    if (block) {
        const BlockDrafter& bd = m.w.dflash;
        blockbytes = bd.layers.size() * 2 * (size_t) (bd.window + 256) * bd.n_kv * bd.head_dim * 2 +
                      (size_t) kWindow * (g.n_embd * 4 + 16) + 64 * 256;
    }
    const uint64_t dev_bytes = session_bytes(g, m.o.max_context, mtp, m.o.kv_format(), m.o.kv_resident, m.lb, m.le);
    if (!st.arena.reserve(dev_bytes + blockbytes)) {
        err = m.o.arch + ": out of device memory for a session of " + std::to_string((long long) m.o.max_context) +
              " cells (" + std::to_string((unsigned long long) (dev_bytes >> 20)) + " MiB); lower --max-context or --parallel";
        return false;
    }
    const size_t C = (size_t) g.conv_channels(), HV = (size_t) g.gdn_v_heads;
    st.gdn_floats = (size_t) g.gdn_state * HV * g.gdn_state + C * 3;
    const size_t ng = (size_t) m.n_gdn;
    if (ng) {
        st.gdn = st.gdn_first = st.arena.take<float>(ng * st.gdn_floats);
        st.gdn_alt = st.arena.take<float>(ng * st.gdn_floats);
        st.rec_qkv = st.arena.take<float>(ng * kWindow * C);
        st.rec_h = st.arena.take<float>(ng * kWindow * C);
        st.rec_gate = st.arena.take<float>(ng * kWindow * HV);
        st.rec_beta = st.arena.take<float>(ng * kWindow * HV);
    }
    const size_t cells = (size_t) m.o.max_context * g.n_kv * g.head_dim;
    // with K/V streaming each layer keeps its first `near` cells here and the rest in page-locked host memory
    const int64_t near_cells = resident_cells(m.o.max_context, m.o.kv_resident);
    const size_t n_near = (size_t) near_cells * g.n_kv * g.head_dim, n_far = cells - n_near;
    uint8_t* host = nullptr;
    if (n_far > 0 && m.n_attn > 0) {
        const uint64_t hb = session_host_bytes(g, m.o.max_context, m.o.kv_format(), m.o.kv_resident) / (uint64_t) g.attn_layers() * m.n_attn;
        if (cudaHostAlloc(&st.far_host, hb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            st.far_host = nullptr;
            err = m.o.arch + ": out of page-locked memory for the streamed K/V (" + std::to_string((unsigned long long) (hb >> 20)) +
                  " MiB); raise --kv-resident or lower --max-context or --parallel";
            return false;
        }
        host = (uint8_t*) st.far_host;
    }
    // the far cells of one layer, carved from the host block (device-mapped: the same address on the card)
    const auto far_view = [&](kernels::KvFar& f) {
        f.at = near_cells;
        if (m.o.kv_int8) {
            f.q8.k = (int8_t*) host;
            f.q8.v = f.q8.k + n_far;
            f.q8.ks = (uint16_t*) (f.q8.v + n_far);
            f.q8.vs = f.q8.ks + n_far / 64;
            host = (uint8_t*) (f.q8.vs + n_far / 64);
        } else {
            f.k = (uint16_t*) host;
            f.v = f.k + n_far;
            host = (uint8_t*) (f.v + n_far);
        }
    };
    for (int i = 0; i < m.n_attn; ++i) {
        kernels::KvInt8 q8;
        st.far_kv.emplace_back();
        if (n_far > 0) far_view(st.far_kv.back());
        if (m.o.kv_int8) {
            const bool f4 = m.o.kv_fp4;
            q8.fp4 = f4;
            q8.k = st.arena.take<int8_t>(kernels::kv_code_bytes(n_near, f4));
            q8.v = st.arena.take<int8_t>(kernels::kv_code_bytes(n_near, f4));
            q8.ks = st.arena.take<uint16_t>(kernels::kv_scale_bytes(n_near, f4) / 2);
            q8.vs = st.arena.take<uint16_t>(kernels::kv_scale_bytes(n_near, f4) / 2);
            st.kc.push_back(nullptr);
            st.vc.push_back(nullptr);
            if (!q8.vs) {
                err = m.o.arch + ": the KV cache does not fit the session's arena";
                return false;
            }
        } else {
            st.kc.push_back(st.arena.take<uint16_t>(n_near));
            st.vc.push_back(st.arena.take<uint16_t>(n_near));
        }
        st.kq.push_back(q8);
    }
    if (mtp) {
        st.mtp_kc = st.arena.take<uint16_t>(cells);
        st.mtp_vc = st.arena.take<uint16_t>(cells);
        st.hid = st.arena.take<float>((size_t) kWindow * g.n_embd);
        st.tail = st.arena.take<float>(g.n_embd);
        st.chain = st.arena.take<float>(2 * (size_t) g.n_embd);
        st.dlogits = st.arena.take<float>(g.n_vocab);
        st.lp = st.arena.take<float>((size_t) kWindow * kernels::kLogprobStride);
        st.win_tok = st.arena.take<int32_t>(kWindow);
        if (!st.win_tok) {
            err = m.o.arch + ": the drafter's buffers do not fit the session's arena";
            return false;
        }
    }
    if (block) {
        const BlockDrafter& bd = m.w.dflash;
        st.dctx_cap = bd.window + 256;
        for (size_t l = 0; l < bd.layers.size(); ++l) {
            st.dk.push_back(st.arena.take<uint16_t>((size_t) st.dctx_cap * bd.n_kv * bd.head_dim));
            st.dv.push_back(st.arena.take<uint16_t>((size_t) st.dctx_cap * bd.n_kv * bd.head_dim));
        }
        st.dacc = st.arena.take<float>((size_t) kWindow * g.n_embd);
        st.dids = st.arena.take<int32_t>(kWindow);
        st.dconf = st.arena.take<float>(kWindow);
        st.dq_ids = st.arena.take<int32_t>((size_t) kWindow * kDraftQ);
        st.dq_p = st.arena.take<float>((size_t) kWindow * kDraftQ);
        st.win_tok = st.arena.take<int32_t>(kWindow);
        if (!st.win_tok) {
            err = m.o.arch + ": the block drafter's buffers do not fit the session's arena";
            return false;
        }
    }
    if (head) {
        st.logits = st.arena.take<float>((size_t) kWindow * g.n_vocab);
        st.picks = st.arena.take<int32_t>(kWindow);
    }
    st.n_keep = st.arena.take<int32_t>(4);
    if (!st.n_keep || (ng && !st.gdn) || (!st.kc.empty() && !m.o.kv_int8 && !st.vc.back())) {
        err = m.o.arch + ": a session's buffers do not fit its arena";
        return false;
    }
    if (ng && cudaMemsetAsync(st.gdn, 0, ng * st.gdn_floats * 4, st.stream) != cudaSuccess) {
        err = "clearing the recurrent state failed";
        return false;
    }
    return cudaStreamSynchronize(st.stream) == cudaSuccess;
}

// What init_state made, on the current device (the arena frees itself).
void free_state(SessionState& st) {
    for (auto& per_buffer : st.wg)
        for (cudaGraphExec_t& e : per_buffer)
            if (e) cudaGraphExecDestroy(e);
    if (st.stream) cudaStreamDestroy(st.stream);
    if (st.side) cudaStreamDestroy(st.side);
    if (st.fork) cudaEventDestroy(st.fork);
    if (st.join) cudaEventDestroy(st.join);
    if (st.far_host) cudaFreeHost(st.far_host);
}

namespace {

class Session final : public model::BatchSession {
public:
    explicit Session(Model& m) : m_(m), g_(m.g) {}
    ~Session() override {
        st_.peer.reset();
        free_state(st_);
        if (done_) cudaEventDestroy(done_);
        if (h_picks_) cudaFreeHost(h_picks_);
    }

    bool init(std::string& err) {
        drafter_ = m_.o.drafter == Drafter::Mtp;
        block_ = m_.o.drafter == Drafter::Block;
        if (!init_state(m_, st_, err)) return false;
        if (m_.split && !split_session(m_, st_, err)) return false;
        if (cudaHostAlloc((void**) &h_picks_, kWindow * sizeof(int32_t), cudaHostAllocPortable) != cudaSuccess) {
            err = "out of page-locked memory";
            return false;
        }
        lp_scratch_.resize(1);
        return true;
    }

    int max_window() const override {
        return block_ || drafter_ ? std::clamp(m_.o.spec, 1, kWindow) : 1;
    }

    bool copy_drafts() const override { return max_window() > 1; }

    // The block drafter's width for n drafts: its own block, or wider (up to the window) right after a window that kept
    // most of a block, so a run of accepted drafts can grow past the block; the policy reads each draft's confidence.
    int block_width(int n) const { return wide_ ? std::clamp(n + 1, m_.w.dflash.block, kWindow) : m_.w.dflash.block; }

    // The drafter's chain after `anchor` (the token at `pos`): step 0 pairs main's last normed row (cell pos - 1) with
    // the anchor, each later step its own output with the draft before it; all on the stream, read by draft_end.
    Result draft_begin(int32_t anchor, int64_t pos, int n) override {
        if (block_) {
            // the GDN commit runs beside the drafter (which reads none of its rows); the stream waits for it after
            // the drafter's work, so whatever comes next sees the committed state
            struct Join {
                Session& s;
                ~Join() { s.join_commit(); }
            } join{*this};
            if (Result s = settle(pos, true); !s.ok()) return s;
            std::string err;
            // a block's drafts cost about one pass at any count, so it offers all of them whatever was asked (a wide
            // block the whole window) and the policy picks the window from their chances
            n = wide_ ? max_window() - 1 : std::clamp(n, 1, max_window() - 1);
            const int width = block_width(n);
            if (pos + width > m_.o.max_context) return fail(m_.o.arch + ": no room for a draft block");
            if (!block_draft(m_, st_, anchor, pos, width, err)) return fail(err);
            n_draft_ = std::min(width - 1, max_window() - 1);
            return {};
        }
        if (!drafter_) return {Status::NotImplemented, m_.o.arch + ": no drafter is loaded"};
        if (Result s = settle(pos); !s.ok()) return s;
        if (st_.tail_cell != pos - 1) return fail(m_.o.arch + ": the drafter has no row before position " + std::to_string((long long) pos));
        n = std::clamp(n, 1, max_window() - 1);
        if (pos + n > m_.o.max_context) n = (int) std::max<int64_t>(1, m_.o.max_context - pos);
        std::string err;
        cudaMemcpyAsync(st_.win_tok, &anchor, sizeof(int32_t), cudaMemcpyHostToDevice, st_.stream);
        const float* hid = st_.tail;
        const int32_t* tok = st_.win_tok;
        for (int k = 0; k < n; ++k) {
            float* out = st_.chain + (size_t) (k & 1) * g_.n_embd;
            float* lp = st_.lp + (size_t) k * kernels::kLogprobStride;
            if (!mtp_window(m_, st_, hid, tok, 1, pos - 1 + k, out, lp, err)) return fail(err);
            hid = out;
            tok = reinterpret_cast<const int32_t*>(lp + 2);
        }
        st_.tail_cell = -1;   // cell pos - 1 holds the right pair now
        n_draft_ = n;
        return {};
    }

    // The block drafter for several sessions in passes of as many blocks as a batch window holds: each session is settled
    // on its own stream (one after another: they share the window scratch), the passes run on the first one's stream, and
    // the others' streams wait for them before their draft_end reads.
    Result draft_batch(model::ModelSession* const* peers, const int32_t* anchors, const int64_t* pos, const int* n,
                       int count) override {
        if (count <= 1 || !block_ || 2 * m_.w.dflash.block > kBatch) return ModelSession::draft_batch(peers, anchors, pos, n, count);
        std::vector<Session*> ss((size_t) count);
        for (int i = 0; i < count; ++i) {
            ss[(size_t) i] = dynamic_cast<Session*>(peers[i]);
            if (!ss[(size_t) i] || &ss[(size_t) i]->m_ != &m_) return fail(m_.o.arch + ": a draft batch of sessions of another model");
            if (Result r = ss[(size_t) i]->settle(pos[i], true); !r.ok()) return r;   // its commit beside the rest
            if (cudaStreamSynchronize(ss[(size_t) i]->st_.stream) != cudaSuccess) return fail(m_.o.arch + ": " + cudaGetErrorString(cudaGetLastError()));
        }
        if (!done_ && cudaEventCreateWithFlags(&done_, cudaEventDisableTiming) != cudaSuccess) return fail(m_.o.arch + ": no event");
        std::vector<SessionState*> sts((size_t) count);
        for (int i = 0; i < count; ++i) sts[(size_t) i] = &ss[(size_t) i]->st_;
        std::string err;
        // the blocks in passes of at most a batch window's rows, in order
        std::vector<int> nn((size_t) count), width((size_t) count);
        // a slot's next window shares the batch's rows: no more drafts than its share takes
        const int share = std::max(1, std::min(max_window(), kBatch / count) - 1);
        for (int i = 0; i < count; ++i) {
            nn[(size_t) i] = std::clamp(n[i], 1, share);
            width[(size_t) i] = ss[(size_t) i]->block_width(nn[(size_t) i]);
            if (pos[i] + width[(size_t) i] > m_.o.max_context) return fail(m_.o.arch + ": no room for a draft block");
        }
        const cudaStream_t s0 = sts[0]->stream;
        for (int i = 0; i < count;) {
            int k = 0, rows = 0;
            while (i + k < count && rows + width[(size_t) (i + k)] <= kBatch) rows += width[(size_t) (i + k++)];
            if (!block_draft_batch(m_, sts.data() + i, anchors + i, pos + i, width.data() + i, k, err, s0)) return fail(err);
            i += k;
        }
        cudaEventRecord(done_, s0);
        for (int i = 1; i < count; ++i) cudaStreamWaitEvent(sts[(size_t) i]->stream, done_, 0);
        for (int i = 0; i < count; ++i) ss[(size_t) i]->join_commit();   // the sessions' commits ran side by side
        for (int i = 0; i < count; ++i) ss[(size_t) i]->n_draft_ = std::min(share, width[(size_t) i] - 1);   // all the block's, as solo
        return {};
    }

    Result draft_end(int32_t* ids, float* conf, int max_n, int& n, const DraftNeed&) override {
        if (block_) {
            n = std::min(n_draft_, max_n);
            if (cudaMemcpyAsync(ids, st_.dids, (size_t) n * sizeof(int32_t), cudaMemcpyDeviceToHost, st_.stream) != cudaSuccess ||
                cudaMemcpyAsync(conf, st_.dconf, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost, st_.stream) != cudaSuccess ||
                cudaStreamSynchronize(st_.stream) != cudaSuccess)
                return fail(m_.o.arch + ": the block drafter failed");
            return {};
        }
        if (!drafter_) return {Status::NotImplemented, m_.o.arch + ": no drafter is loaded"};
        float h[kWindow * kernels::kLogprobStride];
        n = std::min(n_draft_, max_n);
        if (cudaMemcpyAsync(h, st_.lp, (size_t) n_draft_ * kernels::kLogprobStride * sizeof(float), cudaMemcpyDeviceToHost,
                            st_.stream) != cudaSuccess || cudaStreamSynchronize(st_.stream) != cudaSuccess)
            return fail(m_.o.arch + ": the drafter failed");
        for (int k = 0; k < n; ++k) {
            const float* r = h + (size_t) k * kernels::kLogprobStride;
            int32_t id;
            std::memcpy(&id, r + 2, sizeof id);
            ids[k] = id;
            conf[k] = std::exp(r[3]);
        }
        return {};
    }

    Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens < 1 || r.n_tokens > kWindow) return fail(m_.o.arch + ": a decode window of " + std::to_string(r.n_tokens) + " rows");
        if (r.pos + r.n_tokens > m_.o.max_context) return fail(m_.o.arch + ": the window passes the context");
        if (Result s = settle(r.pos); !s.ok()) return s;
        const int T = r.n_tokens;
        int32_t pos3[kWindow * 3];
        for (int t = 0; t < T; ++t) pos3[t * 3] = pos3[t * 3 + 1] = pos3[t * 3 + 2] = (int32_t) (r.pos + t + st_.rope_delta);
        std::string err;
        if (!prepare_draw(r, err)) return fail(err);
        if (drafter_) cudaMemcpyAsync(st_.win_tok, r.tokens, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, st_.stream);
        if (!forward_window(m_, st_, r.tokens, T, r.pos, pos3, err)) return fail(err);
        draw(T);
        pend_pos_ = r.pos;
        pend_T_ = T;
        if (r.logits) cudaMemcpyAsync(r.logits, st_.logits + (size_t) (T - 1) * g_.n_vocab, (size_t) g_.n_vocab * 4,
                                      cudaMemcpyDeviceToHost, st_.stream);
        cudaMemcpyAsync(h_picks_, st_.picks, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost, st_.stream);
        if (cudaStreamSynchronize(st_.stream) != cudaSuccess) return fail(m_.o.arch + ": " + cudaGetErrorString(cudaGetLastError()));
        if (r.row_tokens) std::copy(h_picks_, h_picks_ + T, r.row_tokens);
        if (r.next_token) *r.next_token = h_picks_[T - 1];
        return {};
    }

    int batch_rows() const override { return kBatch; }

protected:
    // ---- several sessions' windows as one batch, on model/batch_driver.hpp: each session is settled (its commit on its side
    // stream, all of them side by side) and its draw prepared as decode_window does, the rows run together on the first
    // session's stream, each session draws from its own logits

    const char* batch_name() const override { return m_.o.arch.c_str(); }
    int64_t batch_context() const override { return m_.o.max_context; }
    model::BatchShared& batch_shared() override { return m_.batch; }
    cudaStream_t batch_stream() const override { return st_.stream; }
    // a layer split's forward runs its two cards' graphs itself (forward_batch)
    bool batch_captured() const override { return !m_.split; }

    bool batch_reserve(const model::DecodeRequest& r, std::string& err) override {
        if (r.n_tokens < 1 || r.n_tokens > kWindow) {
            err = "a decode window of " + std::to_string(r.n_tokens) + " rows";
            return false;
        }
        if (Result s = settle(r.pos, true); !s.ok()) {
            err = s.message;
            return false;
        }
        join_commit();
        if (st_.peer && cudaStreamSynchronize(st_.peer->stream) != cudaSuccess) {
            err = cudaGetErrorString(cudaGetLastError());
            return false;
        }
        return prepare_draw(r, err);
    }

    static BatchSeg seg_of(const model::BatchView& v, int i) {
        Session& se = *static_cast<Session*>(v.ss[i]);
        return {&se.st_, v.w.segs[(size_t) i].rows, v.w.segs[(size_t) i].pos0};
    }

    // every row's token and rope position (its session's offset), the drafter's window tokens; the composition is the key
    void batch_pack(const model::BatchView& v, std::vector<BatchSeg>& segs, std::vector<int32_t>& pos3) const {
        segs.resize((size_t) v.n);
        pos3.clear();
        for (int i = 0; i < v.n; ++i) {
            segs[(size_t) i] = seg_of(v, i);
            const SessionState& st = *segs[(size_t) i].st;
            for (int t = 0; t < segs[(size_t) i].T; ++t) {
                const int32_t p = (int32_t) (segs[(size_t) i].pos + t + st.rope_delta);
                pos3.insert(pos3.end(), {p, p, p});
            }
        }
    }

    bool batch_stage(const model::BatchView& v, model::GraphLru<core::CapturedGraph>::Key&, std::string&) override {
        for (int i = 0; i < v.n; ++i) {
            Session& se = *static_cast<Session*>(v.ss[i]);
            if (se.drafter_)
                cudaMemcpyAsync(se.st_.win_tok, v.reqs[i].tokens, (size_t) v.reqs[i].n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, v.stream);
        }
        if (m_.split) return true;   // forward_batch stages both cards
        std::vector<BatchSeg> segs;
        std::vector<int32_t> pos3;
        batch_pack(v, segs, pos3);
        stage_batch(m_, segs.data(), v.n, v.w.tokens.data(), pos3.data());
        return true;
    }

    bool batch_forward(const model::BatchView& v, std::string& err) override {
        std::vector<BatchSeg> segs;
        std::vector<int32_t> pos3;
        batch_pack(v, segs, pos3);
        if (m_.split) return forward_batch(m_, segs.data(), v.n, v.w.tokens.data(), pos3.data(), err);
        batch_body(m_, segs.data(), v.n);
        return true;
    }

    void batch_take(const model::BatchView& v, int i) override {
        const model::DecodeRequest& r = v.reqs[i];
        pend_pos_ = r.pos;
        pend_T_ = r.n_tokens;
        if (r.logits) cudaMemcpyAsync(r.logits, st_.logits + (size_t) (r.n_tokens - 1) * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost, v.stream);
    }

    const int32_t* batch_draw(const model::BatchView& v, int i, std::string&) override {
        const int T = v.reqs[i].n_tokens;
        draw(T, v.stream);
        cudaMemcpyAsync(h_picks_, st_.picks, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost, v.stream);
        return h_picks_ + (T - v.head[i]);   // the driver reads the window's last head[i] rows
    }

public:
    Result prefill_chunk(const model::PrefillRequest& r) override {
        if (r.pos + r.n_tokens > m_.o.max_context) return fail(m_.o.arch + ": the prompt passes the context");
        if (Result s = settle(r.pos); !s.ok()) return s;
        if (m_.split && !drafter_) {   // the layer split: the chunks pipelined over the two cards
            std::string err;
            if (!split_read(m_, st_, r.tokens, r.n_tokens, r.pos, r.progress, r.stop, err)) return fail(err);
            return {};
        }
        if (m_.prompt_split) {   // a long prompt across the cards (the prompt split)
            std::string err;
            if (prompt_split_read(*m_.prompt_split, m_, st_, r.tokens, r.n_tokens, r.pos, r.progress, r.stop, err)) return {};
            if (!err.empty()) return fail(err);
        }
        std::vector<int32_t> pos3;
        for (int at = 0; at < r.n_tokens;) {
            const int n = std::min(kChunk, r.n_tokens - at);
            const int64_t p = r.pos + at;
            pos3.resize((size_t) n * 3);
            for (int t = 0; t < n; ++t) pos3[(size_t) t * 3] = pos3[(size_t) t * 3 + 1] = pos3[(size_t) t * 3 + 2] = (int32_t) (p + t + st_.rope_delta);
            std::string err;
            if (!forward_chunk(m_, st_, r.tokens + at, nullptr, n, p, pos3.data(), err)) return fail(err);
            if (cudaStreamSynchronize(st_.stream) != cudaSuccess) return fail(m_.o.arch + ": " + cudaGetErrorString(cudaGetLastError()));
            at += n;
            if (r.progress) r.progress(at);
            if (at < r.n_tokens && r.stop && r.stop()) break;
        }
        return {};
    }

    bool reads_patches() const override { return m_.encoder != nullptr; }

    Result encode_image(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, model::ImageGrid& grid) override {
        if (!m_.encoder) return {Status::NotImplemented, m_.o.arch + ": started without --vision"};
        vision::Image img;
        std::string err;
        if (!vision::decode_image(data, bytes, img, err)) return fail(err);
        const vision::Geometry geo;
        const vision::Size sz = vision::fit_size(img.w, img.h, geo);
        std::vector<float> planar, patches;
        vision::normalize_planar(vision::to_canvas(img, sz), planar);
        vision::patchify(planar, sz, geo, patches);
        if (Result r = encode_patches(patches, sz.w, sz.h, rows); !r.ok()) return r;
        n_rows = (int) (rows.size() / (size_t) g_.n_embd);
        grid = {sz.w / (geo.patch * geo.merge), sz.h / (geo.patch * geo.merge)};
        return {};
    }

    Result encode_patches(const std::vector<float>& patches, int w, int h, std::vector<float>& rows) override {
        if (!m_.encoder) return {Status::NotImplemented, m_.o.arch + ": started without --vision"};
        std::string err;
        if (!m_.vision_attach((w / 16) * (h / 16) / 4, err)) return fail(err);
        if (!m_.encoder->encode(patches, vision::Size{w, h}, rows)) return fail("the vision encoder failed: " + m_.encoder->last_error());
        return {};
    }

    void images_done() override { m_.vision_detach(); }

    Result prefill_embeddings(const float* rows, int n_rows, int64_t pos, model::ImageGrid image_grid) override {
        if (pos + n_rows > m_.o.max_context) return fail(m_.o.arch + ": the image passes the context");
        if (Result s = settle(pos); !s.ok()) return s;
        // one grid of merged rows: row j at (t, h, w) = (p, p + j / nx, p + j % nx), the text after it at p + max(nx, ny)
        const int64_t p = pos + st_.rope_delta;
        const int nx = image_grid.nx > 0 ? image_grid.nx : 0, ny = image_grid.ny > 0 ? image_grid.ny : 0;
        const bool grid = nx > 0 && ny > 0 && (int64_t) nx * ny == n_rows;
        std::vector<int32_t> pos3;
        for (int at = 0; at < n_rows;) {
            const int n = std::min(kChunk, n_rows - at);
            pos3.resize((size_t) n * 3);
            for (int t = 0; t < n; ++t) {
                const int j = at + t;
                pos3[(size_t) t * 3] = (int32_t) (grid ? p : p + j);
                pos3[(size_t) t * 3 + 1] = (int32_t) (grid ? p + j / nx : p + j);
                pos3[(size_t) t * 3 + 2] = (int32_t) (grid ? p + j % nx : p + j);
            }
            std::string err;
            if (!forward_chunk(m_, st_, nullptr, rows + (size_t) at * g_.n_embd, n, pos + at, pos3.data(), err)) return fail(err);
            if (cudaStreamSynchronize(st_.stream) != cudaSuccess) return fail(m_.o.arch + ": " + cudaGetErrorString(cudaGetLastError()));
            at += n;
        }
        if (grid) st_.rope_delta += std::max(nx, ny) - n_rows;
        return {};
    }

    // The running state for the shared checkpoint transfer: the head {rope_delta, pos, tail_cell, valid} on the host,
    // then the regions the engine moves itself: the GDN state, the drafter's waiting row, and the block drafter's
    // context rings per layer (K, V) whole as they lie. A ring holds cell c at slot c % capacity, so a restore at the
    // same position finds every cell where it was; `valid` says how many cells below the position are this
    // conversation's.
    static constexpr size_t kHeadBytes = 4 * sizeof(int64_t);
    size_t block_ring_bytes() const { return (size_t) st_.dctx_cap * m_.w.dflash.n_kv * m_.w.dflash.head_dim * sizeof(uint16_t); }
    size_t running_bytes() const override {
        return kHeadBytes + (size_t) (m_.n_gdn + (st_.peer ? m_.split_model->n_gdn : 0)) * st_.gdn_floats * 4 + (drafter_ ? (size_t) g_.n_embd * 4 : 0) +
               (block_ ? st_.dk.size() * 2 * block_ring_bytes() : 0);
    }

    Result running_state(int64_t pos, bool save, core::RunningState& s) override {
        if (save) {
            if (Result r = settle(pos); !r.ok()) return r;
            const int64_t W = block_ ? std::min<int64_t>(m_.w.dflash.window, st_.dctx_cap) : 0;
            const int64_t valid = std::clamp<int64_t>(pos - st_.dctx_from, 0, std::min<int64_t>(W, pos));
            const int64_t head[4] = {st_.rope_delta, pos, st_.tail_cell, valid};
            s.head.assign((const uint8_t*) head, (const uint8_t*) head + sizeof head);
        } else {
            s.head.assign(kHeadBytes, 0);
        }
        s.regions.clear();
        if (const size_t n = (size_t) m_.n_gdn * st_.gdn_floats * 4) s.regions.push_back({st_.gdn, n});
        if (drafter_) s.regions.push_back({st_.tail, (size_t) g_.n_embd * 4});
        if (block_)
            for (size_t l = 0; l < st_.dk.size(); ++l) {
                s.regions.push_back({st_.dk[l], block_ring_bytes()});
                s.regions.push_back({st_.dv[l], block_ring_bytes()});
            }
        s.stream = st_.stream;
        s.peer_regions.clear();
        if (st_.peer) {   // the second card's layers' state
            if (const size_t n = (size_t) m_.split_model->n_gdn * st_.gdn_floats * 4) s.peer_regions.push_back({st_.peer->gdn, n});
            s.peer_stream = st_.peer->stream;
            s.peer_device = m_.split_dev;
        }
        return {};
    }

    Result running_restored(int64_t pos, const uint8_t* head) override {
        pend_T_ = 0;
        int64_t h[4];
        std::memcpy(h, head, sizeof h);
        const int64_t W = block_ ? std::min<int64_t>(m_.w.dflash.window, st_.dctx_cap) : 0;
        if (h[1] != pos || h[3] < 0 || h[3] > W || h[3] > pos) {
            st_.dctx_from = pos;   // the block drafter's cells are not this conversation's
            return fail("restoring the state failed: its head does not match the position");
        }
        st_.rope_delta = h[0];
        st_.tail_cell = h[2];
        st_.dctx_from = pos - h[3];
        return {};
    }

    // No cache to lend: all it holds is state the driver moves to host memory. Both cards' work is finished first.
    Result set_low_memory(bool, int64_t&) override {
        cudaStreamSynchronize(st_.stream);
        if (st_.peer) cudaStreamSynchronize(st_.peer->stream);
        return {};
    }

    // a layer's cells: fp16 K and V (format 1), or int8 codes with fp16 scales per 64 values (format 2); the MTP
    // drafter's own layer is fp16 and comes last
    size_t row_bytes(bool int8) const { return (size_t) g_.n_kv * g_.head_dim * (int8 ? 1 : 2); }
    // a quantized cache's code bytes per position, and how many code bytes one scale byte covers (int8: 32, NVFP4: 8)
    size_t q_row() const { return kernels::kv_code_bytes((size_t) g_.n_kv * g_.head_dim, m_.o.kv_fp4); }
    size_t q_per_scale() const { return m_.o.kv_fp4 ? 8 : 32; }
    // the format a parked conversation's layer records: fp16 1, int8 2, NVFP4 3
    int q_format() const { return m_.o.kv_fp4 ? 3 : 2; }
    // the conversation's attention layers: the main card's, then the second card's (a layer split); kv_state maps i to its
    // card's state and its index there
    size_t kv_layers() const { return st_.kc.size() + (st_.peer ? st_.peer->kc.size() : 0); }
    const SessionState& kv_state(size_t& i) const {
        if (i < st_.kc.size()) return st_;
        i -= st_.kc.size();
        return *st_.peer;
    }
    size_t cells_bytes(int64_t positions) const override {
        const size_t main = m_.o.kv_int8 ? 2 * (q_row() + q_row() / q_per_scale()) : 2 * row_bytes(false);
        return (size_t) positions * (kv_layers() * main + (drafter_ ? 2 * row_bytes(false) : 0));
    }

    Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) override {
        cudaStreamSynchronize(st_.stream);
        if (st_.peer) cudaStreamSynchronize(st_.peer->stream);
        kv.clear();
        // a layer's array: its first `split` bytes on the card (`dev`), the rest streamed (`far`, host memory)
        const auto down = [&](const void* dev, const void* far_buf, size_t split, core::ConversationBuffer& to, size_t bytes) {
            to.resize_for_overwrite(bytes);
            return to.visit(0, to.size(), [&](uint8_t* p, size_t n, size_t at) {
                const size_t a = at < split ? std::min(n, split - at) : 0;
                return (a == 0 || cudaMemcpy(p, (const uint8_t*) dev + at, a, cudaMemcpyDeviceToHost) == cudaSuccess) &&
                       (a == n || cudaMemcpy(p + a, (const uint8_t*) far_buf + (at + a - split), n - a, cudaMemcpyDefault) == cudaSuccess);
            });
        };
        for (size_t i = 0; i < kv_layers() + (drafter_ ? 1 : 0); ++i) {
            const bool mtp = i == kv_layers();
            size_t j = i;
            const SessionState& ls = mtp ? st_ : kv_state(j);
            const core::OnDevice on(&ls == &st_ ? -1 : m_.split_dev);
            const bool int8 = !mtp && m_.o.kv_int8;
            core::ConversationKv k;
            k.format = int8 ? q_format() : 1;
            k.cells = positions;
            k.heads = g_.n_kv;
            k.head_dim = g_.head_dim;
            k.page_size = 1;
            const size_t rows = (size_t) positions * (int8 ? q_row() : row_bytes(false));
            const kernels::KvFar f = mtp ? kernels::KvFar{} : ls.far_kv[j];
            const size_t split = (size_t) std::min<int64_t>(f.at, positions) * (int8 ? q_row() : row_bytes(false));
            bool ok;
            if (int8) {
                const kernels::KvInt8& q = ls.kq[j];
                ok = down(q.k, f.q8.k, split, k.k, rows) && down(q.v, f.q8.v, split, k.v, rows) &&
                     down(q.ks, f.q8.ks, split / q_per_scale(), k.k_scale, rows / q_per_scale()) &&
                     down(q.vs, f.q8.vs, split / q_per_scale(), k.v_scale, rows / q_per_scale());
            } else {
                ok = down(mtp ? st_.mtp_kc : ls.kc[j], f.k, split, k.k, rows) && down(mtp ? st_.mtp_vc : ls.vc[j], f.v, split, k.v, rows);
            }
            if (!ok) return fail("reading the KV cells back failed");
            kv.push_back(std::move(k));
        }
        return {};
    }

    Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) override {
        cudaStreamSynchronize(st_.stream);
        if (st_.peer) cudaStreamSynchronize(st_.peer->stream);
        if (kv.size() != kv_layers() + (drafter_ ? 1 : 0)) return fail(m_.o.arch + ": a parked conversation has other caches");
        if (positions > m_.o.max_context) return fail(m_.o.arch + ": a parked conversation is longer than the context");
        const auto up = [&](void* dev, void* far_buf, size_t split, const core::ConversationBuffer& from, size_t bytes) {
            return from.size() >= bytes && from.visit(0, bytes, [&](const uint8_t* p, size_t n, size_t at) {
                const size_t a = at < split ? std::min(n, split - at) : 0;
                return (a == 0 || cudaMemcpy((uint8_t*) dev + at, p, a, cudaMemcpyHostToDevice) == cudaSuccess) &&
                       (a == n || cudaMemcpy((uint8_t*) far_buf + (at + a - split), p + a, n - a, cudaMemcpyDefault) == cudaSuccess);
            });
        };
        for (size_t i = 0; i < kv.size(); ++i) {
            const bool mtp = i == kv_layers();
            size_t j = i;
            const SessionState& ls = mtp ? st_ : kv_state(j);
            const core::OnDevice on(&ls == &st_ ? -1 : m_.split_dev);
            const bool int8 = !mtp && m_.o.kv_int8;
            const core::ConversationKv& k = kv[i];
            if (k.format != (int8 ? q_format() : 1) || k.heads != g_.n_kv || k.head_dim != g_.head_dim)
                return fail(m_.o.arch + ": a parked conversation has other caches");
            const size_t rows = (size_t) positions * (int8 ? q_row() : row_bytes(false));
            const kernels::KvFar f = mtp ? kernels::KvFar{} : ls.far_kv[j];
            const size_t split = (size_t) std::min<int64_t>(f.at, positions) * (int8 ? q_row() : row_bytes(false));
            bool ok;
            if (int8) {
                const kernels::KvInt8& q = ls.kq[j];
                ok = up(q.k, f.q8.k, split, k.k, rows) && up(q.v, f.q8.v, split, k.v, rows) &&
                     up(q.ks, f.q8.ks, split / q_per_scale(), k.k_scale, rows / q_per_scale()) &&
                     up(q.vs, f.q8.vs, split / q_per_scale(), k.v_scale, rows / q_per_scale());
            } else {
                ok = up(mtp ? st_.mtp_kc : ls.kc[j], f.k, split, k.k, rows) && up(mtp ? st_.mtp_vc : ls.vc[j], f.v, split, k.v, rows);
            }
            if (!ok) return fail(m_.o.arch + ": a parked conversation has other caches");
        }
        pend_T_ = 0;
        return {};
    }

    Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (!bias_on_) return {};
        std::string e;
        if (!device_bias(d_bias_, bias, e)) return fail(e);
        return {};
    }

    Result resample_row(int row, const float* bias, int32_t& token) override {
        std::string e;
        if (!device_bias(d_mask_, bias, e)) return fail(e);
        kernels::SamplerParams sp = sampling_;
        sp.logit_bias = d_mask_;
        sp.counter += (uint64_t) row;
        sp.n_drawn = 0;   // the row is drawn again under the mask, whatever its draft was
        kernels::sample_tokens(st_.logits + (size_t) row * g_.n_vocab, 1, g_.n_vocab,
                               history_len_ > 0 ? d_hist_ + (size_t) row * history_len_ : nullptr, history_len_, sp,
                               st_.picks, st_.stream);
        if (cudaMemcpyAsync(h_picks_, st_.picks, sizeof(int32_t), cudaMemcpyDeviceToHost, st_.stream) != cudaSuccess ||
            cudaStreamSynchronize(st_.stream) != cudaSuccess)
            return fail("the masked draw failed");
        token = h_picks_[0];
        return {};
    }

    Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (n <= 0 || row0 < 0 || row0 + n > kWindow) return fail("row_logprobs: rows out of range");
        std::string e;
        if (!lp_buffers(e)) return fail(e);
        float* d_out = lp_buf_ + kernels::row_logprobs_scratch_floats(kWindow, g_.n_vocab);
        int32_t* d_picks = reinterpret_cast<int32_t*>(d_out + (size_t) kWindow * kernels::kLogprobStride);
        if (cudaMemcpyAsync(d_picks, picks, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice, st_.stream) != cudaSuccess)
            return fail("row_logprobs: the picks copy failed");
        kernels::row_logprobs(st_.logits + (size_t) row0 * g_.n_vocab, n, g_.n_vocab, d_picks, top, d_out, lp_buf_, st_.stream);
        if (cudaMemcpyAsync(out, d_out, (size_t) n * kernels::kLogprobStride * sizeof(float), cudaMemcpyDeviceToHost,
                            st_.stream) != cudaSuccess || cudaStreamSynchronize(st_.stream) != cudaSuccess)
            return fail("row_logprobs: the kernel failed");
        return {};
    }

    Result copy_logits(int row, float* out) override {
        cudaStreamSynchronize(st_.stream);
        if (cudaMemcpy(out, st_.logits + (size_t) row * g_.n_vocab, (size_t) g_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail("the logits copy failed");
        return {};
    }

    std::string info_words() const override {
        const std::string kv = m_.o.kv_fp4 ? "kv=nvfp4" : m_.o.kv_int8 ? "kv=int8" : "kv=fp16";
        // a dense model has no routed experts, so none can sit on the SSD (the community check reads this)
        // host_weight_bytes: model weights held in host RAM, none for the dense stack (the community check refuses a row if
        // any). gpu_weight_bytes is what the load log calls "GiB on the GPU". The token embedding and the --vision dynamic encoder wait in host RAM by design and are told apart.
        const uint64_t host_vision = m_.encoder ? m_.encoder->weights_bytes() : 0;
        return kv + " gpu_weight_bytes=" + std::to_string(m_.o.gpu_weight_bytes) + " host_weight_bytes=0 host_embedding_bytes=" + std::to_string(m_.o.host_embedding_bytes) +
               " host_vision_bytes=" + std::to_string(host_vision) +
               " " + program::spill_words({}) + (block_ ? " drafter=block" : drafter_ ? " drafter=mtp" : "");
    }
    int64_t disk_min_tokens() const override { return 4096; }

private:
    // The last window's accepted rows go into the recurrent state once the next call says where it starts.
    void join_commit() {
        if (!commit_pending_) return;
        if (m_.n_gdn) cudaStreamWaitEvent(st_.stream, st_.join, 0);
        if (st_.peer && m_.split_model->n_gdn) {
            const core::OnDevice on(m_.split_dev);
            cudaStreamWaitEvent(st_.peer->stream, st_.peer->join, 0);
        }
        commit_pending_ = false;
    }
    // `defer`: the commit's join is left to the caller (join_commit), which queues work beside it first
    Result settle(int64_t pos, bool defer = false) {
        if (pos == 0) {   // a new conversation: the running state starts empty
            pend_T_ = 0;
            st_.rope_delta = 0;
            st_.tail_cell = -1;
            st_.dctx_from = 0;
            wide_ = false;
            st_.alt_ready = false;
            if (m_.n_gdn &&
                cudaMemsetAsync(st_.gdn, 0, (size_t) m_.n_gdn * st_.gdn_floats * 4, st_.stream) != cudaSuccess)
                return fail("clearing the recurrent state failed");
            if (SessionState* p = st_.peer.get()) {
                const core::OnDevice on(m_.split_dev);
                p->alt_ready = false;
                if (cudaMemsetAsync(p->gdn, 0, (size_t) m_.split_model->n_gdn * p->gdn_floats * 4, p->stream) != cudaSuccess)
                    return fail("clearing the recurrent state failed");
            }
            return {};
        }
        if (pend_T_ == 0) {
            st_.alt_ready = false;
            if (st_.peer) st_.peer->alt_ready = false;
            return {};
        }
        const int64_t n = pos - pend_pos_;
        pend_T_ = std::max(pend_T_, 0);
        const int T = pend_T_;
        pend_T_ = 0;
        if (n < 0 || n > T) return fail(m_.o.arch + ": position " + std::to_string((long long) pos) + " is not reachable from the last window");
        wide_ = n >= std::max(2, m_.w.dflash.block - 2);
        const bool swap = st_.alt_ready && T == 1 && n == 1;
        st_.alt_ready = false;
        // the window left its state after the row: nothing to commit (on both cards of a split, which ran it alike)
        if (swap) std::swap(st_.gdn, st_.gdn_alt);
        if (SessionState* p = st_.peer.get(); p && swap) std::swap(p->gdn, p->gdn_alt);
        if (st_.peer) st_.peer->alt_ready = false;
        if (n > 0 && !swap) {
            const auto commit = [&](Model& m, SessionState& s) {
                if (m.n_gdn == 0) return;
                cudaEventRecord(s.fork, s.stream);
                cudaStreamWaitEvent(s.side, s.fork, 0);
                commit_window(m, s, (int) n, s.side);
                cudaEventRecord(s.join, s.side);
                commit_pending_ = true;
            };
            commit(m_, st_);
            if (st_.peer) {
                const core::OnDevice on(m_.split_dev);
                commit(*m_.split_model, *st_.peer);
            }
            if (!defer) join_commit();
        }
        if (block_ && n > 0) {   // the kept rows become the block drafter's context
            std::string err;
            if (!block_ctx_window(m_, st_, (int) n, pend_pos_, err)) return fail(err);
        }
        if (drafter_ && n > 0) {
            // the drafter's history: the waiting row with the window's first token, then each kept row with the next
            std::string err;
            if (st_.tail_cell == pend_pos_ - 1 && !mtp_window(m_, st_, st_.tail, st_.win_tok, 1, pend_pos_ - 1, nullptr, nullptr, err))
                return fail(err);
            if (n > 1 && !mtp_window(m_, st_, st_.hid, st_.win_tok + 1, (int) n - 1, pend_pos_, nullptr, nullptr, err)) return fail(err);
            cudaMemcpyAsync(st_.tail, st_.hid + (size_t) (n - 1) * g_.n_embd, (size_t) g_.n_embd * sizeof(float),
                            cudaMemcpyDeviceToDevice, st_.stream);
            st_.tail_cell = pend_pos_ + n - 1;
        }
        return {};
    }

    bool device_bias(float*& dev, const float* host, std::string& err) {
        const size_t bytes = (size_t) g_.n_vocab * sizeof(float);
        if (dev == nullptr) {
            void* v = nullptr;
            if (cudaMalloc(&v, bytes) != cudaSuccess) { err = "out of device memory for a logit bias"; return false; }
            owned_.emplace_back(v);
            dev = (float*) v;
        }
        // on the sampling stream itself: a plain cudaMemcpy does not order against it, so a draw could read the previous bias
        if (cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, st_.stream) != cudaSuccess) { err = "the logit bias upload failed"; return false; }
        return true;
    }

    bool lp_buffers(std::string& err) {
        if (lp_buf_) return true;
        const size_t f = kernels::row_logprobs_scratch_floats(kWindow, g_.n_vocab) + (size_t) kWindow * (kernels::kLogprobStride + 1);
        void* v = nullptr;
        if (cudaMalloc(&v, f * sizeof(float)) != cudaSuccess) { err = "out of device memory for logprobs"; return false; }
        owned_.emplace_back(v);
        lp_buf_ = (float*) v;
        return true;
    }

    // How the window's picks are drawn: the shared sampler (greedy when the request does not sample).
    bool prepare_draw(const model::DecodeRequest& r, std::string& err) {
        history_len_ = 0;
        sampling_ = kernels::SamplerParams{};
        sampling_.greedy = true;
        if (r.sampling != nullptr) sampling_ = *r.sampling;
        sampling_.logit_bias = bias_on_ ? d_bias_ : nullptr;
        history_len_ = r.history != nullptr ? r.history_len : 0;
        const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
        const bool narrow = kernels::sampler_list_len(sampling_.top_k, g_.n_vocab) <= kernels::kSamplerNarrowMax;
        if (sampled && !narrow && !kernels::sampler_wide_reserve(g_.n_vocab, st_.stream)) {
            err = "out of device memory for the sampler";
            return false;
        }
        // Speculative sampling: on the narrow lists the block drafter draws its drafts from its own distribution and a
        // window of them keeps each by min(1, p / q), else draws the residual (sampler.hpp): the tokens are distributed as
        // the model's, and a drafter that spreads its guesses like the model keeps more of them than its best guesses would.
        // Greedy requests, the wide lists and a request with its own seed keep the drafter's best candidates and the
        // plain draws.
        st_.dtemp = block_ && sampled && narrow && sampling_.draw_drafts ? sampling_.temperature : 0.0f;
        st_.dseed = sampling_.seed;
        if (r.drawn_drafts && st_.dtemp > 0.0f && r.n_tokens > 1) {
            sampling_.drafts = st_.dids;
            sampling_.q_ids = st_.dq_ids;
            sampling_.q_p = st_.dq_p;
            sampling_.q_n = m_.w.dflash.top_k;
            sampling_.n_drawn = r.n_tokens - 1;
        }
        if (history_len_ > 0) {
            if (history_len_ > kHistoryCap) history_len_ = kHistoryCap;
            if (d_hist_ == nullptr) {
                void* v = nullptr;
                if (cudaMalloc(&v, (size_t) kHistoryCap * kWindow * sizeof(int32_t)) != cudaSuccess) { err = "out of device memory for the penalties"; return false; }
                owned_.emplace_back(v);
                d_hist_ = (int32_t*) v;
            }
            if (cudaMemcpy(d_hist_, r.history, (size_t) r.n_tokens * r.history_len * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "the penalty history upload failed";
                return false;
            }
            history_len_ = r.history_len;
        }
        return true;
    }

    void draw(int T) { draw(T, st_.stream); }
    void draw(int T, cudaStream_t stream) {
        kernels::sample_tokens(st_.logits, T, g_.n_vocab, history_len_ > 0 ? d_hist_ : nullptr, history_len_, sampling_,
                               st_.picks, stream);
    }

    struct DevFree {
        void operator()(void* p) const { cudaFree(p); }
    };

    static constexpr int kHistoryCap = 4096;
    Model& m_;
    const Geometry& g_;
    SessionState st_;
    int32_t* h_picks_ = nullptr;
    bool drafter_ = false, block_ = false;
    int n_draft_ = 0;
    cudaEvent_t done_ = nullptr;   // a batched drafter pass, for the other sessions' streams
    bool commit_pending_ = false;  // the GDN commit runs on st_.side, not yet joined to st_.stream
    bool wide_ = false;            // the last settled window kept most of a block (block - 2 rows or more)
    int64_t pend_pos_ = 0;
    int pend_T_ = 0;
    kernels::SamplerParams sampling_{};
    bool bias_on_ = false;
    int history_len_ = 0;
    float *d_bias_ = nullptr, *d_mask_ = nullptr, *lp_buf_ = nullptr;
    int32_t* d_hist_ = nullptr;
    std::vector<int> lp_scratch_;
    std::vector<std::unique_ptr<void, DevFree>> owned_;
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(Model& m, std::string& err) {
    auto s = std::make_unique<Session>(m);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::dense
