// src/models/inkling2/session.cpp - one conversation of Inkling 2: its caches (a window ring per sliding-window visit, a
// position per cell for the full-attention visits, the short convolutions' rings) and the forward of a decode window,
// replayed from a CUDA graph per window shape.
//
// A token visits 48 layer applications: the prelude (layers 0..5), the loop's core (6..17) three times, the coda
// (18..23). Each visit has its own caches. A layer's inputs are block-attention mixes of the finished blocks and the
// partial sum; in the loop's later passes they also take a message read from the previous pass through gated controllers.
// A row's result does not depend on the other rows of its call: the products take 8 rows at most per launch with the
// single-row reduction order, the attention and the glue run a block per row (or row and head), the experts per (row,
// slot) entry. Nothing in a window's forward after its inputs are staged depends on the host: the positions and tokens
// are read on the card, and the attention's launch bound (a power of two of keys) only sizes its grid.
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/graph.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
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
#include <map>
#include <set>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

constexpr int kMaxRows = kDecodeRows;    // rows of one forward call (a decode window, a step of a prompt)
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
        ctrl_scr_ = f((int64_t) ctrl_scratch_floats(g_.gate_rank));
        nrm_ = f(R * E);
        y_ = f(R * kProj);
        q_ = f(R * kHeads * kHeadDim); r_ = f(R * kHeads * kRel); sink_ = f(R * kHeads);
        o_ = f(R * E); resid_ = f(R * E); fin_ = f(R * E);
        att_scr_ = alloc(attention_decode_scratch(kMaxRows, std::max<int64_t>(ctx, g_.window)), ok);
        for (void*& x : xq_) x = alloc(bk::native_q8_1_bytes((int) E, kMaxRows), ok);
        xo_ = alloc(bk::native_q8_1_bytes(kOutRows, kMaxRows), ok);
        rlog_ = f(R * g_.n_expert); rw_ = f(R * g_.n_used); ids_ = (int32_t*) alloc((size_t) R * g_.n_used * 4, ok);
        const int64_t pairs = R * g_.n_used;
        ex_ptr_ = (unsigned long long*) alloc((size_t) pairs * 8, ok);
        ex_start_ = (int32_t*) alloc((size_t) (pairs + 2) * 4, ok);
        ex_ng_ = (int32_t*) alloc(16, ok);
        ex_dst_ = (int32_t*) alloc((size_t) pairs * 4, ok);
        ex_tok_ = (int32_t*) alloc((size_t) pairs * 4, ok);
        ex_scr_ = alloc(bk::native_expert_scratch_bytes(pairs, g_.n_ff_exp), ok);
        ex_rows_ = f(pairs * E);
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

    // A full-attention visit's cache is cell-major ([cell][head][dim]): its first `positions` cells are one run.
    model::Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (positions > m_.context()) return {model::Status::Error, "inkling2: positions past the context"};
        cudaStreamSynchronize(stream_);
        kv.clear();
        const size_t run = (size_t) positions * kKvHeads * kHeadDim * elem();
        for (const VisitCache& c : vc_) {
            if (c.ring) continue;
            core::ConversationKv k;
            k.format = cell_format();
            k.page_size = 1;
            k.cells = positions;
            k.heads = kKvHeads;
            k.head_dim = kHeadDim;
            k.k.resize_for_overwrite(run);
            k.v.resize_for_overwrite(run);
            const auto down = [](const void* dev, core::ConversationBuffer& to) {
                return to.visit(0, to.size(), [&](uint8_t* p, size_t n, size_t at) {
                    return cudaMemcpy(p, (const uint8_t*) dev + at, n, cudaMemcpyDeviceToHost) == cudaSuccess;
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
        const size_t run = (size_t) positions * kKvHeads * kHeadDim * elem();
        size_t i = 0;
        for (const VisitCache& c : vc_) {
            if (c.ring) continue;
            if (i >= kv.size() || kv[i].format != cell_format() || kv[i].cells != positions || kv[i].heads != kKvHeads ||
                kv[i].k.size() < run || kv[i].v.size() < run)
                return {model::Status::Error, "inkling2: a parked conversation has other caches"};
            const auto up = [&](void* dev, const core::ConversationBuffer& from) {
                return from.visit(0, run, [&](const uint8_t* p, size_t n, size_t at) {
                    return cudaMemcpy((uint8_t*) dev + at, p, n, cudaMemcpyHostToDevice) == cudaSuccess;
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

    // Layer `l` (visit `v`) over `rows` rows: `hid` its input, `blocks` the finished blocks before it, `msgs` (or null)
    // the previous pass's messages [kStreams][rows][E]; `full_keys` bounds a full-attention visit's keys. Writes its
    // output to `out` (not `hid`).
    void run_layer(const Visit& v, const float* hid, const float* const* blocks, int n_blocks, const float* msgs, int rows,
                   int64_t full_keys, float* out) {
        const int l = v.layer;
        const LayerWeights& w = m_.layer(l);
        const bool full = !g_.swa[(size_t) l];
        const int E = g_.n_embd;
        const size_t RE = (size_t) rows * E;
        VisitCache& c = vc_[(size_t) v.slot];
        const float* norms[3] = {full ? w.q_norm : w.attn_norm, full ? w.k_norm : w.attn_norm, full ? w.v_norm : w.attn_norm};
        const int n_in = full ? 3 : 1;   // the attention's distinct inputs before the controllers
        // the attention's inputs: the mixes (layer 0's input as it is), normed into q8_1; in the loop's later passes each
        // first takes its message through its controller
        MixNorm mn;
        for (int i = 0; i < n_blocks; ++i) mn.st.s[mn.st.n++] = blocks[i];
        mn.st.s[mn.st.n++] = hid;
        mn.queries = w.attn_query;
        mn.eps = g_.rms_eps;
        if (!msgs) {
            for (int j = 0; j < n_in; ++j) {
                mn.norm[j] = norms[j];
                mn.q8[j] = xq_[j];
            }
            mix_norm(mn, n_in, rows, stream_);
        } else {
            mn.mix_out = mix_;
            mn.out_stride = (int64_t) RE;
            mix_norm(mn, n_in, rows, stream_);
            CtrlArgs ca;
            for (int s = 0; s < 3; ++s) {
                ca.dest[s] = mix_ + (full ? s : 0) * RE;
                ca.msg[s] = msgs + s * RE;
                ca.down[s] = w.ctrl_down[s];
                ca.up[s] = w.ctrl_gf[s];
                ca.norm[s] = norms[s];
                ca.q8[s] = xq_[s];
            }
            ca.scratch = ctrl_scr_;
            ca.rank = g_.gate_rank;
            ca.eps = g_.rms_eps;
            controllers(ca, 3, rows, stream_);
        }
        // the projections: one product over every row of the fused matrix when the three inputs are one
        const size_t row_bytes = (size_t) E / 32 * 34;
        AttnPrep ap;
        if (!full && !msgs) {
            bk::native_q8_0_mmvq(w.proj, xq_[0], y_, E, kProj, rows, stream_);
            ap.yq = y_;
            ap.yk = y_ + kProjQ;
            ap.yv = y_ + kProjQ + kProjKV;
            ap.ldq = ap.ldk = ap.ldv = kProj;
        } else {
            float* yk = y_ + (size_t) rows * kProjQ;
            float* yv = yk + (size_t) rows * kProjKV;
            bk::native_q8_0_mmvq(w.proj, xq_[0], y_, E, kProjQ, rows, stream_);
            bk::native_q8_0_mmvq(w.proj + (size_t) kRowK * row_bytes, xq_[1], yk, E, kProjKV, rows, stream_);
            bk::native_q8_0_mmvq(w.proj + (size_t) kRowV * row_bytes, xq_[2], yv, E, kProjKV, rows, stream_);
            ap.yq = y_;
            ap.yk = yk;
            ap.yv = yv;
            ap.ldq = kProjQ;
            ap.ldk = ap.ldv = kProjKV;
        }
        // convolutions, clips, caches
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
        decode_prepare(ap, rows, stream_);
        // the attention, gated by its row's gate projection and rounded for the output projection
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
        aa.gate = ap.yq + kOffGate;
        aa.gate_ld = ap.ldq;
        aa.q8 = xo_;
        attention_decode(aa, rows, c.ring ? (int64_t) g_.window : full_keys, att_scr_, nullptr, stream_);
        bk::native_q8_0_mmvq(w.out, xo_, o_, kOutRows, E, rows, stream_);
        // the attention's convolution and the residual (a block's first layer starts a new sum), then the MLP's input
        PostAttn pa;
        pa.o = o_;
        pa.pos = pos_d_;
        pa.taps = w.attn_post_conv;
        pa.ring = c.a_ring;
        pa.hid = hid;
        pa.boundary = g_.boundary(l);
        pa.resid = resid_;
        for (int i = 0; i < n_blocks; ++i) pa.blocks[pa.n_blocks++] = blocks[i];
        pa.query = w.ffn_query;
        pa.norm = w.ffn_norm;
        pa.eps = g_.rms_eps;
        pa.q8 = xq_[0];
        pa.norm_out = nrm_;
        if (msgs) pa.mix_out = mix_;
        post_attn(pa, rows, stream_);
        if (msgs) {
            CtrlArgs ca;
            ca.dest[0] = mix_;
            ca.msg[0] = msgs + 3 * RE;
            ca.down[0] = w.ctrl_down[3];
            ca.up[0] = w.ctrl_gf[3];
            ca.norm[0] = w.ffn_norm;
            ca.q8[0] = xq_[0];
            ca.norm_out = nrm_;
            ca.scratch = ctrl_scr_;
            ca.rank = g_.gate_rank;
            ca.eps = g_.rms_eps;
            controllers(ca, 1, rows, stream_);
        }
        // the experts
        bk::RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = E;
        ra.k = g_.n_used;
        ra.weight_scale = g_.expert_scale;
        ra.gate = bk::RouterGate::Sigmoid;
        bk::router(nrm_, w.router, w.router_bias, rows, ra, rlog_, ids_, rw_, stream_);
        moe_plan(ids_, rows, g_.n_used, g_.n_expert, w.blob_table, ex_ptr_, ex_start_, ex_ng_, ex_dst_, ex_tok_, stream_);
        bk::NativeExpertLayout L = bk::native_expert_layout(w.gu_type, w.d_type, E, g_.n_ff_exp);
        L.gate_softcap = g_.gate_softcap;
        L.up_softcap = g_.up_softcap;
        const int pairs = rows * g_.n_used;
        bk::native_expert_grouped(L, ex_ptr_, ex_start_, ex_ng_, ex_dst_, ex_tok_, std::min(pairs, g_.n_expert), pairs, xq_[0], ex_scr_,
                                  ex_rows_, stream_);
        MoeFinish mf;
        mf.rows = ex_rows_;
        mf.w = rw_;
        mf.k = g_.n_used;
        mf.pos = pos_d_;
        mf.taps = w.ffn_post_conv;
        mf.ring = c.m_ring;
        mf.resid = resid_;
        mf.out = out;
        moe_finish(mf, rows, stream_);
    }

    float* buf(Buf b) const { return role_[(int) b]; }

    // The forward of `rows` rows at positions pos0.. (tokens, or host embedding rows when `emb` is set, which must stay
    // as they are until the window's sync); then the head over the last `head` rows (0: none).
    bool forward(const int32_t* tokens, const float* emb, int rows, int64_t pos0, int head, std::string& err) {
        const int E = g_.n_embd;
        const size_t RE = (size_t) rows * E;
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
        // the full-attention visits' launch bound: the window's keys rounded up to a power of two
        int64_t full_keys = 1024;
        while (full_keys < pos0 + rows) full_keys <<= 1;
        return layers(rows, head, full_keys, err) && check(err);
    }

    // The layers and the head of a window of `rows` rows: replayed from its shape's graph, captured on the shape's second
    // window (its first runs direct, outside a capture).
    bool layers(int rows, int head, int64_t full_keys, std::string& err) {
        const uint64_t key = ((uint64_t) full_keys << 16) | ((uint64_t) head << 8) | (uint64_t) rows;
        auto it = graphs_.find(key);
        if (it != graphs_.end()) return it->second.launch(stream_, err);
        if (direct_.insert(key).second) {
            body(rows, head, full_keys);
            return true;
        }
        core::CapturedGraph g;
        if (!g.begin(stream_, err)) return false;
        body(rows, head, full_keys);
        if (!g.end(stream_, err)) return false;
        return graphs_.emplace(key, std::move(g)).first->second.launch(stream_, err);
    }

    // Launches only: the visits, then the head. Each kernel may start while the one before it finishes (it waits for it
    // before reading what it wrote), so a product streams its first weights during the glue before it.
    void body(int rows, int head, int64_t full_keys) {
        bk::PdlScope pdl(true);
        const int E = g_.n_embd;
        const size_t RE = (size_t) rows * E;
        const float* blocks[kMaxMix];
        for (const Step& s : walk_.steps) {
            const Visit& v = g_.visits[(size_t) s.visit];
            const float* msgs = nullptr;
            if (s.msgs) {   // the four streams' messages from the bank
                MixNorm mm;
                mm.st.s[mm.st.n++] = buf(s.bank[0]);
                mm.st.s[mm.st.n++] = buf(s.bank[1]);
                mm.queries = m_.layer(v.layer).msg_query;
                mm.mix_out = msg_;
                mm.out_stride = (int64_t) RE;
                mix_norm(mm, kStreams, rows, stream_);
                msgs = msg_;
            }
            for (int i = 0; i < s.n_blocks; ++i) blocks[i] = buf(s.blocks[i]);
            run_layer(v, buf(s.hid), blocks, s.n_blocks, msgs, rows, full_keys, buf(s.out));
        }
        if (head <= 0) return;
        const int first = rows - head;
        MixNorm hn;
        for (int i = 0; i < walk_.n_blocks; ++i) hn.st.s[hn.st.n++] = buf(walk_.blocks[i]) + (size_t) first * E;
        hn.st.s[hn.st.n++] = buf(walk_.final_hid) + (size_t) first * E;
        hn.queries = m_.output_query();
        hn.norm[0] = m_.output_norm();
        hn.q8[0] = xq_[0];
        hn.norm_out = fin_;
        hn.eps = g_.rms_eps;
        mix_norm(hn, 1, head, stream_);
        bk::native_q8_0_mmvq(m_.head(), xq_[0], logits_, E, g_.n_vocab, head, stream_);
        softcap(logits_, head, g_.n_vocab, g_.n_vocab, g_.logit_softcap, stream_);
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
    std::map<uint64_t, core::CapturedGraph> graphs_;   // by window shape: rows, head rows, the full visits' launch bound
    std::set<uint64_t> direct_;                        // the shapes whose first window has run
    float* role_[kBufs] = {};
    float* emb_ = nullptr;
    float *mix_ = nullptr, *msg_ = nullptr, *ctrl_scr_ = nullptr, *nrm_ = nullptr, *y_ = nullptr;
    float *q_ = nullptr, *r_ = nullptr, *sink_ = nullptr, *o_ = nullptr, *resid_ = nullptr, *fin_ = nullptr;
    void* att_scr_ = nullptr;
    void* xq_[3] = {};                       // the products' q8_1 inputs of the hidden width
    void* xo_ = nullptr;                     // the output projection's
    float *rlog_ = nullptr, *rw_ = nullptr;
    int32_t* ids_ = nullptr;
    unsigned long long* ex_ptr_ = nullptr;
    int32_t *ex_start_ = nullptr, *ex_ng_ = nullptr, *ex_dst_ = nullptr, *ex_tok_ = nullptr;
    void* ex_scr_ = nullptr;
    float *ex_rows_ = nullptr, *logits_ = nullptr;
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
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(InklingModel& model, std::string& err) {
    auto s = std::make_unique<Session>(model);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::inkling2
