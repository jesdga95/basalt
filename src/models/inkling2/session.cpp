// src/models/inkling2/session.cpp - one conversation of Inkling 2: its caches (a window ring per sliding-window visit, a
// position per cell for the full-attention visits, the short convolutions' rings) and the forward of a run of rows that
// every decode window and every prompt chunk goes through.
//
// A token visits 48 layer applications: the prelude (layers 0..5), the loop's core (6..17) three times, the coda
// (18..23). Each visit has its own caches. A layer's inputs are block-attention mixes of the finished blocks and the
// partial sum; in the loop's later passes they also take a message read from the previous pass through gated controllers.
// A row's result does not depend on the other rows of its call: the products take 8 rows at most per launch with the
// single-row reduction order, the attention and the mixes run a block per row, the experts per (row, slot) entry.
#include "basalt/core/conversation_cache.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/log.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "models/inkling2/caches.hpp"
#include "models/inkling2/model.hpp"
#include "models/inkling2/prompt.hpp"
#include "models/inkling2/walk.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

constexpr int kMaxRows = 8;              // rows of one forward call (a decode window, a step of a prompt)
constexpr int kTypeBF16 = 30;
constexpr int kOutRows = 3072;           // the attention output's width (heads x head width)
constexpr int kStage = 64;               // windows whose tokens and positions are in flight from the pinned ring at once

class Session final : public model::ModelSession {
public:
    explicit Session(InklingModel& m) : m_(m), g_(m.geo()) {}
    ~Session() override {
        if (stream_) cudaStreamSynchronize(stream_);
        for (void* p : bufs_) cudaFree(p);
        for (void* p : pinned_) cudaFreeHost(p);
        for (cudaEvent_t e : stage_ev_)
            if (e) cudaEventDestroy(e);
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool init(std::string& err) {
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
            err = "cannot create a stream";
            return false;
        }
        if (!walk(g_, walk_, err)) return false;
        bool ok = true;
        auto f = [&](int64_t n) { return (float*) alloc((size_t) n * 4, ok); };
        const int64_t E = g_.n_embd, R = kMaxRows;
        const size_t elem = m_.cache_fp8() ? 1 : 2;
        const int64_t ctx = m_.context();
        int max_slot = 0;
        for (const Visit& v : g_.visits) max_slot = std::max(max_slot, v.slot);
        vc_.assign((size_t) max_slot + 1, VisitCache{});
        for (const Visit& v : g_.visits) {
            VisitCache& c = vc_[(size_t) v.slot];
            c.ring = g_.swa[(size_t) v.layer];
            c.cap = c.ring ? (int64_t) g_.window + kMaxRows : ctx;
            const size_t bytes = (size_t) kKvHeads * (size_t) c.cap * kHeadDim * elem;
            c.k = alloc(bytes, ok);
            c.v = alloc(bytes, ok);
            c.k_ring = f((int64_t) kConvRing * kKvHeads * kHeadDim);
            c.v_ring = f((int64_t) kConvRing * kKvHeads * kHeadDim);
            c.a_ring = f((int64_t) kConvRing * E);
            c.m_ring = f((int64_t) kConvRing * E);
        }
        // the row buffers of one call, by their roles in the walk
        for (float*& b : role_) b = f(R * E);
        emb_ = f(R * E);
        mix_ = f(3 * R * E);
        msg_ = f(kStreams * R * E);
        ctrl_pair_ = f(R * 2 * E);
        ctrl_hid_ = f(R * g_.gate_rank);
        ctrl_gf_ = f(R * 2 * E);
        ctrl_out_ = f(3 * R * E);
        nrm_ = f(3 * R * E);
        yq_ = f(R * kProjQ); yk_ = f(R * kProjKV); yv_ = f(R * kProjKV);
        q_ = f(R * kHeads * kHeadDim); r_ = f(R * kHeads * kRel); sink_ = f(R * kHeads);
        attn_ = f(R * kOutRows); gated_ = f(R * kOutRows);
        o_ = f(R * E); resid_ = f(R * E); fin_ = f(R * E);
        att_scr_ = alloc(attention_decode_scratch(kMaxRows, std::max<int64_t>(ctx, g_.window)), ok);
        xq_ = alloc(bk::native_q8_1_bytes(std::max<int>(kOutRows, 2 * (int) E), kMaxRows), ok);
        rlog_ = f(R * g_.n_expert); rw_ = f(R * g_.n_used); ids_ = (int32_t*) alloc((size_t) R * g_.n_used * 4, ok);
        const int64_t pairs = R * g_.n_used;
        ex_ptr_ = (unsigned long long*) alloc((size_t) pairs * 8, ok);
        ex_start_ = (int32_t*) alloc((size_t) (pairs + 2) * 4, ok);
        ex_ng_ = (int32_t*) alloc(16, ok);
        ex_dst_ = (int32_t*) alloc((size_t) pairs * 4, ok);
        ex_tok_ = (int32_t*) alloc((size_t) pairs * 4, ok);
        ex_scr_ = alloc(bk::native_expert_scratch_bytes(pairs, g_.n_ff_exp), ok);
        ex_rows_ = f(pairs * E);
        moe_ = f(R * E);
        logits_ = f(R * (int64_t) g_.n_vocab);
        lp_ = f((int64_t) bk::row_logprobs_scratch_floats(kMaxRows, g_.n_vocab) + R * (bk::kLogprobStride + 1));
        pick_ = (int32_t*) alloc(4 * 32, ok);
        io_d_ = (int32_t*) alloc((size_t) 2 * R * 4, ok);   // a window's tokens, then its positions
        tok_d_ = io_d_;
        pos_d_ = io_d_ + R;
        if (ok && cudaMallocHost((void**) &io_h_, (size_t) kStage * 2 * R * 4) != cudaSuccess) ok = false;
        else if (ok) pinned_.push_back(io_h_);
        for (cudaEvent_t& e : stage_ev_)
            if (ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) ok = false;
        if (!ok) {
            err = "out of device memory for the session";
            return false;
        }
        for (const VisitCache& c : vc_) {
            cudaMemsetAsync(c.k_ring, 0, (size_t) kConvRing * kKvHeads * kHeadDim * 4, stream_);
            cudaMemsetAsync(c.v_ring, 0, (size_t) kConvRing * kKvHeads * kHeadDim * 4, stream_);
            cudaMemsetAsync(c.a_ring, 0, (size_t) kConvRing * E * 4, stream_);
            cudaMemsetAsync(c.m_ring, 0, (size_t) kConvRing * E * 4, stream_);
        }
        cudaStreamSynchronize(stream_);
        prompt_ = make_prompt(m_, vc_, stream_, err);   // PROMPT HOOK
        return prompt_ != nullptr || err.empty();
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens < 1) return {model::Status::Error, "inkling2: an empty window"};
        if (r.pos + r.n_tokens > m_.context()) return {model::Status::Refused, "inkling2: the window runs past the context"};
        std::string err;
        hist_.resize((size_t) r.pos);
        for (int t0 = 0; t0 < r.n_tokens; t0 += kMaxRows) {
            const int n = std::min(kMaxRows, r.n_tokens - t0);
            const bool last = t0 + n == r.n_tokens;
            const bool want = r.row_tokens || (last && (r.next_token || r.logits));
            const int head = !want ? 0 : r.row_tokens ? n : 1;
            if (!forward(r.tokens + t0, nullptr, n, r.pos + t0, head, err)) return {model::Status::Error, "inkling2: " + err};
            for (int i = 0; i < n; ++i) hist_.push_back(r.tokens[t0 + i]);
            head_rows_ = head;
            if (!want) continue;
            if (!pick(r, t0, head, err)) return {model::Status::Error, "inkling2: " + err};
            if (r.row_tokens)
                for (int i = 0; i < n; ++i) r.row_tokens[t0 + i] = picks_[(size_t) i];
            if (last && r.next_token) *r.next_token = picks_[(size_t) head - 1];
            if (last && r.logits)
                cudaMemcpyAsync(r.logits, logits_ + (size_t) (head - 1) * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost, stream_);
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "inkling2: the window failed"};
        return {};
    }

    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        if (r.pos + r.n_tokens > m_.context()) return {model::Status::Refused, "inkling2: the prompt runs past the context"};
        hist_.resize((size_t) r.pos);
        if (prompt_ && r.n_tokens > 16) return prompt_->read(r, hist_);   // PROMPT HOOK
        std::string err;
        for (int t0 = 0; t0 < r.n_tokens; t0 += kMaxRows) {
            if (r.stop && r.stop()) break;
            const int n = std::min(kMaxRows, r.n_tokens - t0);
            if (!forward(r.tokens + t0, nullptr, n, r.pos + t0, 0, err)) return {model::Status::Error, "inkling2: " + err};
            for (int i = 0; i < n; ++i) hist_.push_back(r.tokens[t0 + i]);
            if (r.progress && ((t0 / kMaxRows) % 64 == 63 || t0 + n == r.n_tokens)) {
                cudaStreamSynchronize(stream_);
                r.progress(t0 + n);
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "inkling2: the prompt failed"};
        return {};
    }

    model::Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (!bias_on_) return {};
        std::string e;
        if (!device_bias(d_bias_, bias, e)) return {model::Status::Error, "inkling2: " + e};
        return {};
    }

    // a draw of one row again, with the tokens the mask bans
    model::Result resample_row(int row, const float* bias, int32_t& token) override {
        std::string e;
        if (row < 0 || row >= head_rows_) return {model::Status::Error, "inkling2: the row is not in the last window"};
        if (!device_bias(d_mask_, bias, e)) return {model::Status::Error, "inkling2: " + e};
        bk::SamplerParams sp = sampling_;
        sp.logit_bias = d_mask_;
        sp.counter += (uint64_t) row;
        bk::sample_tokens(logits_ + (size_t) row * g_.n_vocab, 1, g_.n_vocab,
                          history_len_ > 0 ? d_hist_ + (size_t) row * history_len_ : nullptr, history_len_, sp, pick_ + 24, stream_);
        int32_t t = 0;
        if (cudaMemcpyAsync(&t, pick_ + 24, 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess || cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "inkling2: the masked draw failed"};
        token = t;
        return {};
    }

    model::Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (row0 < 0 || n < 1 || row0 + n > head_rows_) return {model::Status::Error, "inkling2: the session keeps the last window's logits only"};
        float* d_out = lp_ + bk::row_logprobs_scratch_floats(n, g_.n_vocab);
        int32_t* d_pick = pick_ + 16;
        cudaMemcpyAsync(d_pick, picks, (size_t) n * 4, cudaMemcpyHostToDevice, stream_);
        bk::row_logprobs(logits_ + (size_t) row0 * g_.n_vocab, n, g_.n_vocab, d_pick, top, d_out, lp_, stream_);
        if (cudaMemcpyAsync(out, d_out, (size_t) n * bk::kLogprobStride * 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "inkling2: the log probabilities failed"};
        return {};
    }

    model::Result copy_logits(int row, float* out) override {
        if (row < 0 || row >= head_rows_) return {model::Status::Error, "inkling2: no logits for that row"};
        if (cudaMemcpy(out, logits_ + (size_t) row * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
            return {model::Status::Error, "inkling2: copying the logits failed"};
        return {};
    }

    model::Result set_history(const int32_t* tokens, int64_t n) override {
        hist_.assign(tokens, tokens + n);
        return {};
    }

    // ---- conversations: the full-attention visits' cells by position (save_cells / restore_cells), the window rings and
    // the convolutions' rings by checkpoint (running_state). A window ring holds window + kMaxRows positions and a
    // convolution ring kConvRing, so a checkpoint at `pos` is intact while the session has not run more than kMaxRows
    // rows past it.
    size_t elem() const { return m_.cache_fp8() ? 1 : 2; }
    size_t kv_bytes_of(const VisitCache& c) const { return (size_t) kKvHeads * (size_t) c.cap * kHeadDim * elem(); }
    size_t conv_ring_bytes() const { return (size_t) kConvRing * (2 * kKvHeads * kHeadDim + 2 * (size_t) g_.n_embd) * 4; }

    size_t running_bytes() const override {
        size_t n = 0;
        for (const VisitCache& c : vc_) n += (c.ring ? 2 * kv_bytes_of(c) : 0) + conv_ring_bytes();
        return n;
    }

    model::Result running_state(int64_t pos, bool save, core::RunningState& st) override {
        if (save && (int64_t) hist_.size() - pos > kMaxRows)
            return {model::Status::Error, "inkling2: the rings have moved on more than " + std::to_string(kMaxRows) + " positions past the checkpoint"};
        if (save && pos > (int64_t) hist_.size()) return {model::Status::Error, "inkling2: a checkpoint past the session's position"};
        const size_t kv_ring = (size_t) kConvRing * kKvHeads * kHeadDim * 4, e_ring = (size_t) kConvRing * g_.n_embd * 4;
        for (VisitCache& c : vc_) {
            if (c.ring) {
                st.regions.push_back({c.k, kv_bytes_of(c)});
                st.regions.push_back({c.v, kv_bytes_of(c)});
            }
            st.regions.push_back({c.k_ring, kv_ring});
            st.regions.push_back({c.v_ring, kv_ring});
            st.regions.push_back({c.a_ring, e_ring});
            st.regions.push_back({c.m_ring, e_ring});
        }
        st.stream = stream_;
        return {};
    }

    model::Result running_restored(int64_t pos, const uint8_t*) override {
        if ((int64_t) hist_.size() > pos) hist_.resize((size_t) pos);
        return {};
    }

    int cell_format() const { return m_.cache_fp8() ? 1 : 3; }   // 3: fp16 rows, 1: unit-scale E4M3

    size_t cells_bytes(int64_t positions) const override {
        size_t n = 0;
        for (const VisitCache& c : vc_)
            if (!c.ring) n += 2 * (size_t) positions * kKvHeads * kHeadDim * elem();
        return n;
    }

    model::Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (positions > m_.context()) return {model::Status::Error, "inkling2: positions past the context"};
        cudaStreamSynchronize(stream_);
        kv.clear();
        const size_t row = (size_t) kHeadDim * elem(), head = (size_t) positions * row;
        for (const VisitCache& c : vc_) {
            if (c.ring) continue;
            core::ConversationKv k;
            k.format = cell_format();
            k.page_size = 1;
            k.cells = positions;
            k.heads = kKvHeads;
            k.head_dim = kHeadDim;
            k.k.resize_for_overwrite(kKvHeads * head);
            k.v.resize_for_overwrite(kKvHeads * head);
            // the cache is [head][cell][dim]: each head's first `positions` cells are one run
            const auto down = [&](const void* dev, core::ConversationBuffer& to) {
                return to.visit(0, to.size(), [&](uint8_t* p, size_t n, size_t at) {
                    while (n > 0) {
                        const size_t h = at / head, off = at % head, take = std::min(n, head - off);
                        const uint8_t* src = (const uint8_t*) dev + h * (size_t) c.cap * row + off;
                        if (cudaMemcpy(p, src, take, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
                        p += take;
                        at += take;
                        n -= take;
                    }
                    return true;
                });
            };
            if (!down(c.k, k.k) || !down(c.v, k.v)) return {model::Status::Error, "inkling2: reading a cache back failed"};
            kv.push_back(std::move(k));
        }
        return {};
    }

    model::Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (positions > m_.context()) return {model::Status::Error, "inkling2: positions past the context"};
        cudaStreamSynchronize(stream_);
        const size_t row = (size_t) kHeadDim * elem(), head = (size_t) positions * row;
        size_t i = 0;
        for (const VisitCache& c : vc_) {
            if (c.ring) continue;
            if (i >= kv.size() || kv[i].format != cell_format() || kv[i].cells != positions || kv[i].heads != kKvHeads)
                return {model::Status::Error, "inkling2: a parked conversation has other caches"};
            const auto up = [&](void* dev, const core::ConversationBuffer& from) {
                return from.visit(0, from.size(), [&](const uint8_t* p, size_t n, size_t at) {
                    while (n > 0) {
                        const size_t h = at / head, off = at % head, take = std::min(n, head - off);
                        uint8_t* dst = (uint8_t*) dev + h * (size_t) c.cap * row + off;
                        if (cudaMemcpy(dst, p, take, cudaMemcpyHostToDevice) != cudaSuccess) return false;
                        p += take;
                        at += take;
                        n -= take;
                    }
                    return true;
                });
            };
            if (!up(c.k, kv[i].k) || !up(c.v, kv[i].v)) return {model::Status::Error, "inkling2: writing a cache back failed"};
            ++i;
        }
        return {};
    }

    int64_t kv_bytes() const override {
        int64_t n = 0;
        for (const VisitCache& c : vc_) n += 2 * (int64_t) kv_bytes_of(c);
        return n;
    }

private:
    void* alloc(size_t bytes, bool& ok) {
        void* p = nullptr;
        if (!ok || cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
            ok = false;
            return nullptr;
        }
        bufs_.push_back(p);
        return p;
    }

    // q8_1 of `rows` rows of `n` floats into the shared activation buffer, then W [n_out][n] (Q8_0) times them: y[r][n_out]
    void gemv_q8(const uint8_t* w, const float* x, int n, int n_out, int rows, float* y) {
        bk::native_quantize_q8_1(x, xq_, n, rows, stream_);
        bk::native_q8_0_mmvq(w, xq_, y, n, n_out, rows, stream_);
    }


    // One loop controller over `rows` rows: out = blend(dest, msg) of core layer `w`'s stream `s`.
    void controller(const LayerWeights& w, int s, const float* dest, const float* msg, int rows, float* out) {
        const int E = g_.n_embd, rank = g_.gate_rank;
        ctrl_pair_norm(dest, msg, rows, ctrl_pair_, stream_);
        bk::bf16_gemv_fp32_mmvf_multi(ctrl_pair_, 2 * E, w.ctrl_down[s], ctrl_hid_, rank, 2 * E, rank, rows, stream_);
        bk::bf16_gemv_fp32_mmvf_multi(ctrl_hid_, rank, w.ctrl_gf[s], ctrl_gf_, 2 * E, rank, 2 * E, rows, stream_);
        ctrl_blend(dest, msg, ctrl_gf_, rows, out, stream_);
    }

    // Layer `l` (visit `v`) over `rows` rows: `hid` its input, `blocks` the finished blocks before it, `msgs` (or null)
    // the previous pass's messages [kStreams][rows][E]. Writes its output to `out` (not `hid`).
    void run_layer(const Visit& v, const float* hid, const float* const* blocks, int n_blocks, const float* msgs, int rows,
                   float* out) {
        const int l = v.layer;
        const LayerWeights& w = m_.layer(l);
        const bool full = !g_.swa[(size_t) l];
        const int E = g_.n_embd;
        const size_t RE = (size_t) rows * E;
        VisitCache& c = vc_[(size_t) v.slot];
        // the attention's inputs
        const float* in[3] = {hid, hid, hid};
        if (w.attn_query) {
            MixStates st;
            for (int i = 0; i < n_blocks; ++i) st.s[st.n++] = blocks[i];
            st.s[st.n++] = hid;
            const int nq = full ? 3 : 1;
            attnres_mix(st, w.attn_query, nq, rows, 1e-6f, mix_, (int64_t) RE, stream_);
            for (int j = 0; j < 3; ++j) in[j] = mix_ + (full ? j : 0) * RE;
        }
        if (msgs) {
            for (int j = 0; j < 3; ++j) controller(w, j, in[j], msgs + j * RE, rows, ctrl_out_ + j * RE);
            for (int j = 0; j < 3; ++j) in[j] = ctrl_out_ + j * RE;
        }
        const float* nq_ = nullptr, *nk = nullptr, *nv = nullptr;
        if (!full && in[0] == in[1] && in[1] == in[2]) {
            bk::rms_norm_rows(in[0], w.attn_norm, rows, E, g_.rms_eps, nrm_, stream_);
            nq_ = nk = nv = nrm_;
        } else {
            const float* wn[3] = {full ? w.q_norm : w.attn_norm, full ? w.k_norm : w.attn_norm, full ? w.v_norm : w.attn_norm};
            for (int j = 0; j < 3; ++j) bk::rms_norm_rows(in[j], wn[j], rows, E, g_.rms_eps, nrm_ + j * RE, stream_);
            nq_ = nrm_;
            nk = nrm_ + RE;
            nv = nrm_ + 2 * RE;
        }
        // the projections: the query input's rows, then the key's, then the value's
        const size_t row_bytes = (size_t) E / 32 * 34;
        gemv_q8(w.proj, nq_, E, kProjQ, rows, yq_);
        gemv_q8(w.proj + (size_t) kRowK * row_bytes, nk, E, kProjKV, rows, yk_);
        gemv_q8(w.proj + (size_t) kRowV * row_bytes, nv, E, kProjKV, rows, yv_);
        // convolutions, clips, caches
        AttnPrep ap;
        ap.yq = yq_;
        ap.yk = yk_;
        ap.yv = yv_;
        ap.pos = pos_d_;
        ap.full = full;
        ap.k_conv = w.k_conv;
        ap.v_conv = w.v_conv;
        ap.q_scale = w.q_clip;
        ap.r_scale = w.r_clip;
        ap.qk_clip = g_.qk_clip;
        ap.rel_clip = g_.rel_clip;
        ap.eps = g_.rms_eps;
        ap.log_floor = g_.log_floor;
        ap.log_alpha = g_.log_alpha;
        ap.k_ring = c.k_ring;
        ap.v_ring = c.v_ring;
        ap.k_cache = c.k;
        ap.v_cache = c.v;
        ap.cap = c.cap;
        ap.fp8 = m_.cache_fp8();
        ap.q = q_;
        ap.r = r_;
        ap.sink = sink_;
        attn_prepare(ap, rows, stream_);
        AttnArgs aa;
        aa.q = q_;
        aa.r = r_;
        aa.sink = sink_;
        aa.rel_table = w.rel_table;
        aa.pos = pos_d_;
        aa.k_cache = c.k;
        aa.v_cache = c.v;
        aa.cap = c.cap;
        aa.fp8 = m_.cache_fp8();
        aa.window = c.ring ? g_.window : 0;
        aa.scale = 1.0f / std::sqrt((float) kHeadDim);
        const int64_t max_keys = c.ring ? g_.window : max_pos_ + 1;
        attention_decode(aa, rows, max_keys, att_scr_, attn_, stream_);
        attn_gate(attn_, yq_, rows, gated_, stream_);
        gemv_q8(w.out, gated_, kOutRows, E, rows, o_);
        // the attention's convolution and the residual: a block's first layer starts a new sum
        const bool boundary = g_.boundary(l);
        post_conv(o_, pos_d_, w.attn_post_conv, c.a_ring, boundary ? nullptr : hid, rows, resid_, stream_);
        // the MLP's input
        MixStates st;
        for (int i = 0; i < n_blocks; ++i) st.s[st.n++] = blocks[i];
        if (boundary) st.s[st.n++] = hid;
        st.s[st.n++] = resid_;
        attnres_mix(st, w.ffn_query, 1, rows, 1e-6f, mix_, (int64_t) RE, stream_);
        const float* mlp_in = mix_;
        if (msgs) {
            controller(w, 3, mix_, msgs + 3 * RE, rows, ctrl_out_);
            mlp_in = ctrl_out_;
        }
        bk::rms_norm_rows(mlp_in, w.ffn_norm, rows, E, g_.rms_eps, nrm_, stream_);
        // the experts
        bk::RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = E;
        ra.k = g_.n_used;
        ra.weight_scale = g_.expert_scale;
        ra.gate = bk::RouterGate::Sigmoid;
        bk::router(nrm_, w.router, w.router_bias, rows, ra, rlog_, ids_, rw_, stream_);
        moe_plan(ids_, rows, g_.n_used, g_.n_expert, w.experts, w.expert_stride, ex_ptr_, ex_start_, ex_ng_, ex_dst_, ex_tok_, stream_);
        bk::native_quantize_q8_1(nrm_, xq_, E, rows, stream_);
        bk::NativeExpertLayout L = bk::native_expert_layout(w.gu_type, w.d_type, E, g_.n_ff_exp);
        L.gate_softcap = g_.gate_softcap;
        L.up_softcap = g_.up_softcap;
        const int pairs = rows * g_.n_used;
        bk::native_expert_grouped(L, ex_ptr_, ex_start_, ex_ng_, ex_dst_, ex_tok_, std::min(pairs, g_.n_expert), pairs, xq_, ex_scr_,
                                  ex_rows_, stream_);
        moe_combine(ex_rows_, rw_, rows, g_.n_used, moe_, stream_);
        post_conv(moe_, pos_d_, w.ffn_post_conv, c.m_ring, resid_, rows, out, stream_);
    }

    float* buf(Buf b) const { return role_[(int) b]; }

    // The forward of `rows` rows at positions pos0.. (tokens, or host embedding rows when `emb` is set, which must stay
    // as they are until the window's sync); then the head over the last `head` rows (0: none).
    bool forward(const int32_t* tokens, const float* emb, int rows, int64_t pos0, int head, std::string& err) {
        const int E = g_.n_embd;
        const size_t RE = (size_t) rows * E;
        max_pos_ = pos0 + rows - 1;
        // the tokens and positions through a ring of pinned slots: a slot is written again once its copy has run
        const int slot = (int) (stage_n_++ % kStage);
        cudaEventSynchronize(stage_ev_[slot]);
        int32_t* io = io_h_ + (size_t) slot * 2 * kMaxRows;
        for (int i = 0; i < rows; ++i) {
            io[i] = tokens ? tokens[i] : 0;
            io[kMaxRows + i] = (int32_t) (pos0 + i);
        }
        cudaMemcpyAsync(io_d_, io, (size_t) 2 * kMaxRows * 4, cudaMemcpyHostToDevice, stream_);
        cudaEventRecord(stage_ev_[slot], stream_);
        float* x0 = buf(Buf::X0);
        if (emb) {
            cudaMemcpyAsync(x0, emb, RE * 4, cudaMemcpyHostToDevice, stream_);
        } else {
            bk::iq_embed_rows(kTypeBF16, m_.embedding(), (size_t) E * 2, tok_d_, rows, E, emb_, stream_);
            bk::rms_norm_rows(emb_, m_.embed_norm(), rows, E, g_.rms_eps, x0, stream_);
        }
        const float* blocks[kMaxMix];
        for (const Step& s : walk_.steps) {
            const Visit& v = g_.visits[(size_t) s.visit];
            const float* msgs = nullptr;
            if (s.msgs) {
                MixStates st;
                st.s[st.n++] = buf(s.bank[0]);
                st.s[st.n++] = buf(s.bank[1]);
                attnres_mix(st, m_.layer(v.layer).msg_query, kStreams, rows, 1e-6f, msg_, (int64_t) RE, stream_);
                msgs = msg_;
            }
            for (int i = 0; i < s.n_blocks; ++i) blocks[i] = buf(s.blocks[i]);
            run_layer(v, buf(s.hid), blocks, s.n_blocks, msgs, rows, buf(s.out));
        }
        if (head <= 0) return check(err);
        // the head over the last `head` rows
        const int first = rows - head;
        MixStates st;
        for (int i = 0; i < walk_.n_blocks; ++i) st.s[st.n++] = buf(walk_.blocks[i]) + (size_t) first * E;
        st.s[st.n++] = buf(walk_.final_hid) + (size_t) first * E;
        attnres_mix(st, m_.output_query(), 1, head, 1e-6f, mix_, (int64_t) head * E, stream_);
        bk::rms_norm_rows(mix_, m_.output_norm(), head, E, g_.rms_eps, fin_, stream_);
        gemv_q8(m_.head(), fin_, E, g_.n_vocab, head, logits_);
        softcap(logits_, head, g_.n_vocab, g_.n_vocab, g_.logit_softcap, stream_);
        return check(err);
    }

    bool check(std::string& err) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            err = std::string("CUDA: ") + cudaGetErrorString(e);
            return false;
        }
        return true;
    }

    bool device_bias(float*& dev, const float* host, std::string& err) {
        const size_t bytes = (size_t) g_.n_vocab * sizeof(float);
        if (dev == nullptr) {
            bool ok = true;
            dev = (float*) alloc(bytes, ok);
            if (!ok) {
                cudaGetLastError();
                err = "out of device memory for a logit bias";
                return false;
            }
        }
        if (cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            err = "the logit bias upload failed";
            return false;
        }
        return true;
    }

    bool pick(const model::DecodeRequest& r, int t0, int rows, std::string& err) {
        bk::SamplerParams sp;
        if (r.sampling) sp = *r.sampling;
        else sp.greedy = true;
        if (!r.sampling) sp.temperature = 0.f;
        sp.counter += (uint64_t) t0;
        sp.logit_bias = bias_on_ ? d_bias_ : nullptr;
        history_len_ = r.history != nullptr ? r.history_len : 0;
        if (history_len_ > 0) {   // the penalties' history of each row of the window
            const size_t n = (size_t) kMaxRows * history_len_;
            if (d_hist_len_ < n) {
                bool ok = true;
                d_hist_ = (int32_t*) alloc(n * 4, ok);
                if (!ok) {
                    cudaGetLastError();
                    err = "out of device memory for the sampling history";
                    return false;
                }
                d_hist_len_ = n;
            }
            cudaMemcpyAsync(d_hist_, r.history + (size_t) t0 * history_len_, (size_t) rows * history_len_ * 4, cudaMemcpyHostToDevice, stream_);
        }
        sampling_ = sp;
        bk::sample_tokens(logits_, rows, g_.n_vocab, history_len_ > 0 ? d_hist_ : nullptr, history_len_, sp, pick_, stream_);
        picks_.assign((size_t) rows, 0);
        if (cudaMemcpyAsync(picks_.data(), pick_, (size_t) rows * 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "the pick failed";
            return false;
        }
        return true;
    }

    InklingModel& m_;
    const Geometry& g_;
    cudaStream_t stream_ = nullptr;
    std::vector<void*> bufs_, pinned_;
    std::vector<VisitCache> vc_;
    std::unique_ptr<PromptRunner> prompt_;   // PROMPT HOOK
    Walk walk_;
    float* role_[kBufs] = {};
    float* emb_ = nullptr;
    float *mix_ = nullptr, *msg_ = nullptr, *ctrl_pair_ = nullptr, *ctrl_hid_ = nullptr, *ctrl_gf_ = nullptr, *ctrl_out_ = nullptr;
    float *nrm_ = nullptr, *yq_ = nullptr, *yk_ = nullptr, *yv_ = nullptr, *q_ = nullptr, *r_ = nullptr, *sink_ = nullptr;
    float *attn_ = nullptr, *gated_ = nullptr, *o_ = nullptr, *resid_ = nullptr, *fin_ = nullptr;
    void* att_scr_ = nullptr;
    void* xq_ = nullptr;
    float *rlog_ = nullptr, *rw_ = nullptr;
    int32_t* ids_ = nullptr;
    unsigned long long* ex_ptr_ = nullptr;
    int32_t *ex_start_ = nullptr, *ex_ng_ = nullptr, *ex_dst_ = nullptr, *ex_tok_ = nullptr;
    void* ex_scr_ = nullptr;
    float *ex_rows_ = nullptr, *moe_ = nullptr, *logits_ = nullptr;
    int32_t *pick_ = nullptr, *io_d_ = nullptr, *tok_d_ = nullptr, *pos_d_ = nullptr, *io_h_ = nullptr;
    cudaEvent_t stage_ev_[kStage] = {};
    uint64_t stage_n_ = 0;
    std::vector<int32_t> picks_;
    std::vector<int32_t> hist_;
    float *d_bias_ = nullptr, *d_mask_ = nullptr, *lp_ = nullptr;
    bool bias_on_ = false;
    int32_t* d_hist_ = nullptr;
    size_t d_hist_len_ = 0;
    int history_len_ = 0;
    bk::SamplerParams sampling_;
    int head_rows_ = 0;
    int64_t max_pos_ = 0;
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(InklingModel& model, std::string& err) {
    auto s = std::make_unique<Session>(model);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::inkling2
