// src/models/nemotronh/session.cpp - a Nemotron-H conversation on one card: the layers of a decode window (up to 16 rows)
// or a prompt chunk, the recurrent state of the Mamba-2 layers, the attention layers' fp16 caches, the routed experts
// through the expert tier.
//
// A decode window never writes the Mamba-2 state or its convolution history (ssd.hpp): the next call says where it starts,
// and the window's first rows up to there are committed then.  A one-row window writes its state into the layer's second
// buffer instead, and keeping its row swaps the two, so plain decoding pays no commit.  A prompt chunk passes the state
// through in place.  The attention layers write their cells as they run; a later call that starts earlier writes over
// the rows it drops.
//
// The routed experts of a window go to the tier in groups of up to five rows (five rows of 22 experts fit one lookup):
// the card computes those its slots hold and some RAM misses it fetches over the bus, the CPU pool the rest from the
// pinned RAM tier, and the router's weights sum both on the card.  The host never waits for the CPU's share: the sum
// waits on the card for a word the joiner thread writes once the CPU's rows landed, while the host queues the next
// layers and the card reads their first weights into L2 (Hy4-flash's and DeepSeek's decode, window_handoff.hpp).  A prompt chunk stages every expert its rows use on the card (from a slot, the RAM tier or the pack)
// and computes them there with llama.cpp's MMQ (q8_1 activations against the NVFP4 blobs).
//
// The MTP head (when the pack has one and drafting is on) drafts: its cache position p holds the pair (the model's normed
// hidden row h_p, the token t_{p+1}).  The model's rows queue their h until the tokens after them are accepted; the queue
// is flushed through the head (its fusion, attention layer and MoE layer) at a window's start, at draft_begin (whose
// anchor is the last token) and in a prompt chunk.  The first draft comes from the last flushed row; each further draft
// feeds the head's own normed output back with the draft before.
#include "models/nemotronh/model.hpp"
#include "models/nemotronh/ops.hpp"

#include "basalt/core/conversation_cache.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/ssd.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/kernels/window_handoff.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/prefill/fp4_dense.hpp"
#include "basalt/prefill/fp8_gemm.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace basalt::models::nemotronh {
namespace {

namespace bk = basalt::kernels;
namespace wg = basalt::kernels::wgemm;

constexpr int kMaxWindow = 16;    // a decode window's rows (the window gemm's)
constexpr int kMoeGroup = 5;      // rows of one tier lookup: 5 x 22 entries fit core::ExpertTier::kMaxK
constexpr int kChunk = 4096;      // rows of a prompt chunk: each restages its layers' experts, so fewer, longer chunks
constexpr int kGroupRows = 4096;  // the most gathered rows of one group of a prompt chunk's experts
constexpr int kGroupExperts = 16; // the most experts of one group (a staging half)
constexpr int kRingRows = 1024;   // a prompt from this long routes to every expert: it borrows the card's room for the ring
static_assert(kChunk <= kGroupRows, "an expert's rows of a chunk fit one group");
constexpr int kTypeNvfp4 = 40;

constexpr int kLookupK = core::ExpertTier::kMaxK;
// A window group's grouped expert launch: the experts' blobs, their entries' bounds, the count, each entry's routed row
// and token.
struct WindowPlan {
    unsigned long long ptr[kLookupK];
    int32_t start[kLookupK + 2];
    int32_t ng[4];
    int32_t dst[kLookupK];
    int32_t tok[kLookupK];
};
// The hits' plan and every routed row's pointer (a card's row or the CPU's mapped row), put in one launch.
struct WindowPut {
    WindowPlan plan;
    unsigned long long rows[kLookupK];
};
static_assert(sizeof(WindowPut) <= 4 * bk::kPutWords, "a window's plan fits put_words");

class Session final : public model::ModelSession {
public:
    explicit Session(NemotronModel& m)
        : m_(m), g_(m.geo()), sg_{g_.ssm_heads, g_.ssm_head_dim, g_.ssm_state, g_.ssm_groups, g_.ssm_conv},
          mtp_on_(m.has_mtp() && m.flags().draft) {}

    ~Session() override {
        if (joiner_.joinable()) {
            {
                std::lock_guard<std::mutex> g(jm_);
                jstop_ = true;
            }
            jcv_.notify_one();
            joiner_.join();
        }
        if (stream_) cudaStreamSynchronize(stream_);
        if (ra_st_) cudaStreamSynchronize(ra_st_);
        if (ra_ev_) cudaEventDestroy(ra_ev_);
        if (ra_st_) cudaStreamDestroy(ra_st_);
        close_ring();
        for (void* p : bufs_) cudaFree(p);
        for (void* p : pinned_) cudaFreeHost(p);
        if (copy_) cudaStreamSynchronize(copy_);
        for (cudaEvent_t e : {ev_ready_, ev_copied_[0], ev_copied_[1], ev_done_[0], ev_done_[1], ring_filled_[0], ring_filled_[1],
                              ring_freed_[0], ring_freed_[1]})
            if (e) cudaEventDestroy(e);
        if (copy_) cudaStreamDestroy(copy_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool init(std::string& err) {
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
            err = "cannot create a stream";
            return false;
        }
        bool ok = true;
        const int64_t E = g_.n_embd, L = g_.n_latent, K = g_.n_used, R = kChunk;
        const int64_t PW = g_.proj_width(), C = g_.conv_dim(), DI = g_.ssm_inner;
        const int64_t QW = (int64_t) g_.n_head * g_.head_dim, KW = (int64_t) g_.n_kv * g_.head_dim;
        auto f = [&](int64_t n) { return (float*) alloc((size_t) n * 4, ok); };
        X_ = f(R * E);
        XN_ = f(R * E);
        proj_ = f(R * PW);
        xbc_ = f(R * C);
        y_ = f(R * DI);
        gn_ = f(R * DI);
        q_ = f(R * QW);
        k_ = f(R * KW);
        v_ = f(R * KW);
        att_ = f(R * QW);
        lat_ = f(R * L);
        latout_ = f(R * L);
        moe_ = f(R * E);
        shh_ = f(R * g_.n_ff_shared);
        sho_ = f(R * E);
        rlog_ = f(R * g_.n_expert);
        rw_ = f(R * K);
        ids_ = (int32_t*) alloc((size_t) R * K * 4, ok);
        tok_ = (int32_t*) alloc((size_t) R * 4, ok);
        logits_ = f((int64_t) kMaxWindow * g_.n_vocab);
        pick_ = (int32_t*) alloc(64 * 4, ok);
        lp_ = f((int64_t) bk::row_logprobs_scratch_floats(kMaxWindow, g_.n_vocab) + (int64_t) kMaxWindow * (bk::kLogprobStride + 1));
        keep_d_ = (int32_t*) alloc(16, ok);
        // the window gemm's activations and its split-K workspace, sized for every call this model makes
        const int64_t kmax = std::max({E, DI, (int64_t) g_.n_ff_shared, L, QW});
        act_ = alloc(wg::act_bytes(kmax), ok);
        {
            size_t pf = 0, tk = 0;
            const auto need = [&](wg::Format fmt, int64_t rows, int64_t k) {
                wg::Call c;
                c.fmt = fmt;
                c.K = k;
                c.m[0].rows = rows;
                size_t a = 0, b = 0;
                wg::workspace_need(c, a, b);
                pf = std::max(pf, a);
                tk = std::max(tk, b);
            };
            need(wg::Format::F8, PW, E);
            need(wg::Format::F8, E, DI);
            need(wg::Format::F8, QW, E);
            need(wg::Format::F8, E, QW);
            need(wg::Format::F8, g_.n_vocab, E);
            need(wg::Format::Nvfp4, g_.n_ff_shared, E);
            need(wg::Format::Nvfp4, E, g_.n_ff_shared);
            need(wg::Format::Bf16, L, E);
            need(wg::Format::Bf16, E, L);
            if (mtp_on_) {   // the MTP head's fusion and its BF16 attention
                need(wg::Format::Bf16, E, 2 * E);
                need(wg::Format::Bf16, E, QW);
            }
            for (wg::Format fmt : {wg::Format::F8, wg::Format::Bf16}) {   // q, k and v in one call (their rows together)
                wg::Call c;
                c.fmt = fmt;
                c.K = E;
                c.n = 3;
                c.m[0].rows = QW;
                c.m[1].rows = c.m[2].rows = KW;
                size_t a = 0, b = 0;
                wg::workspace_need(c, a, b);
                pf = std::max(pf, a);
                tk = std::max(tk, b);
            }
            ws_.partial_floats = pf;
            ws_.tickets_n = tk;
            ws_.partial = pf ? f((int64_t) pf) : nullptr;
            ws_.tickets = tk ? (int32_t*) alloc(tk * 4, ok) : nullptr;
            if (tk && ws_.tickets) cudaMemset(ws_.tickets, 0, tk * 4);
        }
        // the prompt path's activations: FP8 rows (the widest input, d_inner) and FP4 planes (the shared expert)
        xq8_ = alloc(prefill::fp8::act_bytes(R, std::max(E, DI)), ok);
        a4_ = alloc(prefill::fp4_dense::act_bytes(R, std::max(E, (int64_t) g_.n_ff_shared)), ok);
        gs4_ = f(R);
        // the routed experts: a window group's q8_1 latent rows, the grouped kernels' scratch and rows, the plan tables
        exq_ = alloc((size_t) std::max<int64_t>(R, kMaxWindow) * (size_t) (L / 32) * 36, ok);   // q8_1 rows of the latent
        exscr_ = alloc(bk::native_expert_scratch_bytes((int64_t) kMoeGroup * K, g_.n_ff_exp), ok);
        exrows_ = f(R * K * L);
        ex_plan_ = alloc(sizeof(WindowPut), ok);
        ex_plan2_ = alloc(sizeof(WindowPlan), ok);
        ra_mark_ = (uint32_t*) alloc(8, ok);
        if (ok) cudaMemset(ra_mark_, 0, 8);
        if (cudaStreamCreateWithFlags(&ra_st_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&ra_ev_, cudaEventDisableTiming) != cudaSuccess)
            ok = false;
        rowtab_d_ = (const float**) alloc((size_t) R * K * sizeof(float*), ok);
        {   // the prompt path's experts
            namespace mmq = basalt::prefill::mmq;
            const int64_t rows = R * K, ng_max = g_.n_expert;   // at worst a group an expert
            if (!mmq::built() || !mmq::fits(kTypeNvfp4, g_.n_ff_exp) || !mmq::fits(kTypeNvfp4, L) ||
                m_.expert_stride() + 16 - (size_t) bk::native_expert_layout_relu2(kTypeNvfp4, kTypeNvfp4, L, g_.n_ff_exp).bytes < mmq::read_past_bytes(kTypeNvfp4, g_.n_ff_exp)) {
                err = "the prompt path's MMQ does not take these experts";
                return false;
            }
            mmq::load(kTypeNvfp4, g_.n_ff_exp);
            mmq::load(kTypeNvfp4, L);
            mmq_ = std::make_unique<mmq::Context>();
            stage_ = (uint8_t*) alloc((size_t) 2 * kGroupExperts * m_.expert_stride(), ok);
            d_gtok_ = (int32_t*) alloc((size_t) rows * 4, ok);
            d_gbounds_ = (int32_t*) alloc((size_t) ng_max * (kGroupExperts + 1) * 4, ok);
            tails_d_ = f((int64_t) 2 * kGroupExperts * 4);
            ring_tails_ = f((int64_t) 2 * g_.n_expert * 4);
            mxq_ = alloc(mmq::q8_bytes(kGroupRows, L), ok);
            mhq_ = alloc(mmq::q8_bytes(kGroupRows, g_.n_ff_exp), ok);
            hbuf_ = f((int64_t) kGroupRows * g_.n_ff_exp);
            ident_d_ = (int32_t*) alloc((size_t) kGroupRows * 4, ok);
            if (ok) {
                std::vector<int32_t> id(kGroupRows);
                for (int i = 0; i < kGroupRows; ++i) id[(size_t) i] = i;
                cudaMemcpy(ident_d_, id.data(), id.size() * 4, cudaMemcpyHostToDevice);
            }
            if (cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) != cudaSuccess ||
                cudaEventCreateWithFlags(&ev_ready_, cudaEventDisableTiming) != cudaSuccess)
                ok = false;
            for (int i = 0; i < 2; ++i)
                if (cudaEventCreateWithFlags(&ev_copied_[i], cudaEventDisableTiming) != cudaSuccess ||
                    cudaEventCreateWithFlags(&ev_done_[i], cudaEventDisableTiming) != cudaSuccess ||
                    cudaEventCreateWithFlags(&ring_filled_[i], cudaEventDisableTiming) != cudaSuccess ||
                    cudaEventCreateWithFlags(&ring_freed_[i], cudaEventDisableTiming) != cudaSuccess)
                    ok = false;
        }
        const auto pin = [&](size_t n) {
            void* p = nullptr;
            if (cudaHostAlloc(&p, n, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) ok = false;
            else pinned_.push_back(p);
            return p;
        };
        // the router's ids of a window group reach the host through mapped memory behind a sequence word; a weighted sum
        // waits for a word the host writes once the CPU's rows landed
        h_wids_ = (int32_t*) pin((size_t) kLookupK * 4);
        h_words_ = (uint32_t*) pin(64);
        if (ok) {
            h_words_[0] = h_words_[1] = 0;
            if (cudaHostGetDevicePointer((void**) &d_wids_, h_wids_, 0) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &d_words_, h_words_, 0) != cudaSuccess)
                ok = false;
        }
        rowtab_h_ = (const float**) pin((size_t) R * K * sizeof(float*));
        h_ids_ = (int32_t*) pin((size_t) R * K * 4);
        keep_h_ = (int32_t*) pin(16);
        bounce_ = (uint8_t*) pin(m_.expert_stride());
        h_gtok_ = (int32_t*) pin((size_t) R * K * 4);
        h_gbounds_ = (int32_t*) pin((size_t) g_.n_expert * (kGroupExperts + 1) * 4);
        // the recurrent state of each Mamba-2 layer (two buffers), its history and a window's rows a commit replays; the
        // attention layers' caches
        const size_t st = sg_.state_floats(), hs = sg_.hist_floats();
        for (int l = 0; l < g_.n_layers; ++l) {
            const Kind kd = g_.kinds[(size_t) l];
            if (kd == Kind::Mamba) {
                Mamba mb;
                mb.state = f((int64_t) st);
                mb.alt = f((int64_t) st);
                mb.hist = f((int64_t) hs);
                mb.raw = f((int64_t) kMaxWindow * C);
                mb.xbc = f((int64_t) kMaxWindow * C);
                mb.dt = f((int64_t) kMaxWindow * g_.ssm_heads);
                mb.dA = f((int64_t) kMaxWindow * g_.ssm_heads);
                mamba_.push_back(mb);
            } else if (kd == Kind::Attention) {
                const size_t cells = (size_t) m_.context() * (size_t) KW;
                kc_.push_back((uint16_t*) alloc(cells * 2, ok));
                vc_.push_back((uint16_t*) alloc(cells * 2, ok));
            }
        }
        if (mtp_on_) {   // the MTP head's cache (after the model's), its queue of hidden rows, its output row, its logits
            const size_t cells = (size_t) m_.context() * (size_t) KW;
            kc_.push_back((uint16_t*) alloc(cells * 2, ok));
            vc_.push_back((uint16_t*) alloc(cells * 2, ok));
            hq_ = f((int64_t) (kChunk + kMaxWindow + 1) * E);
            hm_ = f(E);
            dlogits_ = f(g_.n_vocab);
            dlp_ = f((int64_t) bk::row_logprobs_scratch_floats(1, g_.n_vocab) + bk::kLogprobStride);
            dpick_ = (int32_t*) alloc(16, ok);
            if (ok) cudaMemset(dpick_, 0, 16);
        }
        attw_ = f((int64_t) bk::gqa_decode_scratch_floats(m_.context(), kMaxWindow, g_.n_head, g_.head_dim));
        if (!ok) {
            err = "out of device memory for the session's buffers (--max-context sets the caches)";
            return false;
        }
        for (const Mamba& mb : mamba_) {
            cudaMemsetAsync(mb.state, 0, st * 4, stream_);
            cudaMemsetAsync(mb.hist, 0, hs * 4, stream_);
        }
        lay_ = bk::native_expert_layout_relu2(kTypeNvfp4, kTypeNvfp4, L, g_.n_ff_exp);
        cudaStreamSynchronize(stream_);
        return true;
    }

    // ---- the ModelSession calls ------------------------------------------------------------------------------------------
    int max_window() const override { return kMaxWindow; }
    bool copy_drafts() const override { return true; }   // prompt lookup: any window's rows verify alike

    // The MTP head's drafts after `anchor` at `pos`: made in draft_end, one at a time, so a draft the window policy rules
    // out is never made.
    model::Result draft_begin(int32_t anchor, int64_t pos, int n) override {
        if (!mtp_on_) return {model::Status::NotImplemented, "nemotronh: the pack has no MTP head, or --spec is off"};
        draft_anchor_ = anchor;
        draft_pos_ = pos;
        draft_n_ = std::max(0, std::min(n, kMaxWindow - 1));
        return {};
    }

    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n, const DraftNeed& need) override {
        n = 0;
        if (!mtp_on_) return {model::Status::NotImplemented, "nemotronh: the pack has no MTP head, or --spec is off"};
        const int want = std::min({draft_n_, max_n, std::max(0, m_.flags().spec - 1)});
        const int64_t pos = draft_pos_;
        if (want <= 0 || pos < 1) return {};
        std::string err;
        // the head's row at pos - 1 runs again if it ran already: its output is the first draft
        if (mtp_valid_ > pos - 1 && hq_pos_ <= pos - 1 && hq_pos_ + hq_n_ > pos - 1) mtp_valid_ = pos - 1;
        if (!mtp_flush(pos, draft_anchor_, true, err)) return {model::Status::Error, "nemotronh: " + err};
        if (!head_ready_) return {};   // the queue lacked the row (a restored state without it): no drafts this time
        for (int j = 0; j < want; ++j) {
            if (j > 0) {
                if (pos + j >= m_.context()) break;
                const int32_t prev = ids[j - 1];
                if (!mtp_rows(hm_, &prev, 1, pos + j - 1, true, err)) return {model::Status::Error, "nemotronh: " + err};
            }
            float lp = 0.f;
            if (!draft_top(ids[j], lp, err)) return {model::Status::Error, "nemotronh: " + err};
            conf[j] = std::exp(lp);
            if (need && !need(conf, j, want)) break;
            n = j + 1;
        }
        return {};
    }

    model::Result prepare() override {
        std::string e;
        if (!ensure_tier(e)) return {model::Status::Error, "nemotronh: " + e};
        return {};
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens < 1 || r.n_tokens > kMaxWindow) return {model::Status::Error, "nemotronh: a window of 1 to 16 rows"};
        std::string err;
        if (!settle(r.pos, err)) return {model::Status::Error, "nemotronh: " + err};
        const bool want = r.row_tokens || r.next_token || r.logits;
        const int head = !want ? 0 : r.row_tokens ? r.n_tokens : 1;
        if (tier_) tier_->begin_token(r.n_tokens);
        // the head's cache through the accepted rows (the window's first token follows them)
        if (mtp_on_ && !mtp_flush(r.pos, r.tokens[0], false, err)) return {model::Status::Error, "nemotronh: " + err};
        // CUDA loads a kernel lazily at its first launch, which can wait for the card, and the card may be holding at a word
        // the host writes only after that launch: the first window of each row count (which picks its kernels) joins its
        // groups itself, so every kernel it needs is loaded while nothing holds the card
        join_now_ = !(warmed_ >> r.n_tokens & 1u);
        if (!run_rows(r.tokens, r.n_tokens, r.pos, true, head, err)) return {model::Status::Error, "nemotronh: " + err};
        warmed_ |= 1u << r.n_tokens;
        pend_pos_ = r.pos;
        pend_T_ = r.n_tokens;
        alt_ready_ = r.n_tokens == 1;
        last_rows_ = head;
        if (want) {
            if (!pick(r, head, err)) return {model::Status::Error, "nemotronh: " + err};
            if (r.row_tokens)
                for (int i = 0; i < head; ++i) r.row_tokens[i] = picks_[(size_t) i];
            if (r.next_token) *r.next_token = picks_[(size_t) head - 1];
            if (r.logits)
                cudaMemcpyAsync(r.logits, logits_ + (size_t) (head - 1) * g_.n_vocab, (size_t) g_.n_vocab * 4,
                                cudaMemcpyDeviceToHost, stream_);
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "nemotronh: the window failed"};
        return {};
    }

    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        std::string err;
        if (!settle(r.pos, err)) return {model::Status::Error, "nemotronh: " + err};
        if (tier_) {
            tier_->forget_use();
            tier_->settle();
            if (std::max(r.total_tokens, r.n_tokens) >= kRingRows) open_ring();
        }
        const auto fail = [&](const std::string& e) {
            close_ring();
            return model::Result{model::Status::Error, "nemotronh: " + e};
        };
        for (int t0 = 0; t0 < r.n_tokens; t0 += kChunk) {
            if (r.stop && r.stop()) break;
            const int n = std::min(kChunk, r.n_tokens - t0);
            ring_more_ = t0 + n < r.n_tokens || r.more;
            if (!run_rows(r.tokens + t0, n, r.pos + t0, false, 0, err)) return fail(err);
            // the head's cache through the chunk's rows but its last (the token after it comes with the next call)
            if (mtp_on_ && !mtp_flush(r.pos + t0 + n - 1, -1, false, err)) return fail(err);
            if (r.progress) {
                cudaStreamSynchronize(stream_);
                r.progress(t0 + n);
            }
        }
        if (!r.more) close_ring();
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "nemotronh: the prompt failed"};
        return {};
    }
    void end_prompt() override { close_ring(); }

    size_t running_bytes() const override {
        return mamba_.size() * (sg_.state_floats() + sg_.hist_floats()) * 4 + (mtp_on_ ? (size_t) g_.n_embd * 4 : 0);
    }

    // The Mamba-2 states and histories, and with the MTP head the model's hidden row at pos - 1 (its cache through pos - 1
    // is flushed first; the cells travel with the attention caches)
    model::Result running_state(int64_t pos, bool save, core::RunningState& st) override {
        if (save) {
            std::string err;
            if (!settle(pos, err)) return {model::Status::Error, "nemotronh: " + err};
            if (mtp_on_ && pos > 0 && !mtp_flush(pos - 1, -1, false, err)) return {model::Status::Error, "nemotronh: " + err};
            if (mtp_on_ && pos > 0) queue_from(pos - 1);   // the row the state carries first
            if (mtp_on_ && pos > 0 && !(hq_pos_ == pos - 1 && hq_n_ >= 1))
                return {model::Status::Error, "nemotronh: the MTP head has no hidden row at the saved position"};
        }
        for (const Mamba& mb : mamba_) {
            st.regions.push_back({mb.state, sg_.state_floats() * 4});
            st.regions.push_back({mb.hist, sg_.hist_floats() * 4});
        }
        if (mtp_on_) st.regions.push_back({hq_, (size_t) g_.n_embd * 4});
        st.stream = stream_;
        return {};
    }

    model::Result running_restored(int64_t pos, const uint8_t*) override {
        pend_T_ = 0;
        alt_ready_ = false;
        if (mtp_on_) {   // the head's cache holds positions before pos - 1; the hidden row of pos - 1 waits for its token
            mtp_valid_ = std::max<int64_t>(pos - 1, 0);
            hq_pos_ = mtp_valid_;
            hq_n_ = pos > 0 ? 1 : 0;
            tok_hist_.assign((size_t) pos, -1);
        }
        return {};
    }

    size_t cells_bytes(int64_t positions) const override {
        return kc_.size() * 2 * (size_t) positions * (size_t) g_.n_kv * g_.head_dim * 2;
    }

    model::Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (positions > m_.context()) return {model::Status::Error, "nemotronh: positions past the context"};
        cudaStreamSynchronize(stream_);
        kv.clear();
        const size_t row = (size_t) g_.n_kv * g_.head_dim * 2;
        for (size_t a = 0; a < kc_.size(); ++a) {
            core::ConversationKv k;
            k.format = kCellFp16;
            k.cells = positions;
            k.heads = g_.n_kv;
            k.head_dim = g_.head_dim;
            k.page_size = 1;
            k.k.resize_for_overwrite((size_t) positions * row);
            k.v.resize_for_overwrite((size_t) positions * row);
            const auto down = [&](const uint16_t* dev, core::ConversationBuffer& to) {
                return to.visit(0, to.size(), [&](uint8_t* p, size_t n, size_t at) {
                    return cudaMemcpy(p, (const uint8_t*) dev + at, n, cudaMemcpyDeviceToHost) == cudaSuccess;
                });
            };
            if (!down(kc_[a], k.k) || !down(vc_[a], k.v)) return {model::Status::Error, "nemotronh: reading a cache back failed"};
            kv.push_back(std::move(k));
        }
        return {};
    }

    model::Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (kv.size() != kc_.size() || positions > m_.context()) return {model::Status::Error, "nemotronh: a parked conversation has other caches"};
        cudaStreamSynchronize(stream_);
        for (size_t a = 0; a < kc_.size(); ++a) {
            const core::ConversationKv& k = kv[a];
            if (k.format != kCellFp16 || k.cells != positions || k.heads != g_.n_kv || k.head_dim != g_.head_dim)
                return {model::Status::Error, "nemotronh: a parked conversation has other caches"};
            const auto up = [&](uint16_t* dev, const core::ConversationBuffer& from) {
                return from.visit(0, from.size(), [&](const uint8_t* p, size_t n, size_t at) {
                    return cudaMemcpy((uint8_t*) dev + at, p, n, cudaMemcpyHostToDevice) == cudaSuccess;
                });
            };
            if (!up(kc_[a], k.k) || !up(vc_[a], k.v)) return {model::Status::Error, "nemotronh: writing a cache back failed"};
        }
        return {};
    }

    model::Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (!bias_on_) return {};
        std::string e;
        if (!device_bias(d_bias_, bias, e)) return {model::Status::Error, "nemotronh: " + e};
        return {};
    }

    model::Result resample_row(int row, const float* bias, int32_t& token) override {
        std::string e;
        if (row < 0 || row >= last_rows_) return {model::Status::Error, "nemotronh: the row is not in the last window"};
        if (!device_bias(d_mask_, bias, e)) return {model::Status::Error, "nemotronh: " + e};
        bk::SamplerParams sp = sampling_;
        sp.logit_bias = d_mask_;
        sp.counter += (uint64_t) row;
        bk::sample_tokens(logits_ + (size_t) row * g_.n_vocab, 1, g_.n_vocab,
                          history_len_ > 0 ? d_hist_ + (size_t) row * history_len_ : nullptr, history_len_, sp, pick_ + 24, stream_);
        int32_t t = 0;
        if (cudaMemcpyAsync(&t, pick_ + 24, 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess || cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "nemotronh: the masked draw failed"};
        token = t;
        return {};
    }

    model::Result copy_logits(int row, float* out) override {
        if (row < 0 || row >= last_rows_) return {model::Status::Error, "nemotronh: the session keeps the last window's logits only"};
        cudaMemcpy(out, logits_ + (size_t) row * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost);
        return {};
    }

    model::Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (row0 < 0 || n < 1 || row0 + n > last_rows_) return {model::Status::Error, "nemotronh: the session keeps the last window's logits only"};
        float* d_out = lp_ + bk::row_logprobs_scratch_floats(n, g_.n_vocab);
        int32_t* d_pick = pick_ + 32;
        cudaMemcpyAsync(d_pick, picks, (size_t) n * 4, cudaMemcpyHostToDevice, stream_);
        bk::row_logprobs(logits_ + (size_t) row0 * g_.n_vocab, n, g_.n_vocab, d_pick, top, d_out, lp_, stream_);
        cudaMemcpyAsync(out, d_out, (size_t) n * bk::kLogprobStride * 4, cudaMemcpyDeviceToHost, stream_);
        return cudaStreamSynchronize(stream_) == cudaSuccess ? model::Result{} : model::Result{model::Status::Error, "nemotronh: logprobs failed"};
    }

    ExpertCounts expert_counts() const override {
        const core::ExpertTier* t = tier_.get();
        return t == nullptr ? ExpertCounts{} : ExpertCounts{t->hits(), t->lookups(), t->offloaded(), t->file_lookups()};
    }

    std::vector<CardExperts> experts_by_card() const override {
        std::vector<CardExperts> out;
        if (tier_)
            for (const core::ExpertTier::CardSlots& c : tier_->slots_by_card()) out.push_back({c.device, c.slots, c.bytes});
        return out;
    }

    int64_t kv_bytes() const override { return (int64_t) cells_bytes(m_.context()); }

    bool expert_map(int64_t& total, std::vector<uint8_t>& homes, std::vector<uint32_t>& routed) override {
        if (!tier_) return false;
        tier_->expert_map(total, homes, routed);
        return true;
    }

    std::string info_words() const override {
        return tier_ ? "expert_slots=" + std::to_string(tier_->gpu_slots()) + " ram_experts_mib=" + std::to_string(tier_->ram_bytes() >> 20) + " kv=fp16"
                     : "kv=fp16";
    }

private:
    static constexpr int kCellFp16 = 3;   // a saved cache's format: fp16 rows

    struct Mamba {
        float *state = nullptr, *alt = nullptr, *hist = nullptr;   // the state (and a one-row window's), the convolution history
        float *raw = nullptr, *xbc = nullptr, *dt = nullptr, *dA = nullptr;   // the last window's rows a commit replays
    };

    void* alloc(size_t bytes, bool& ok) {
        if (!ok) return nullptr;
        void* p = nullptr;
        if (cudaMalloc(&p, std::max<size_t>(bytes, 256)) != cudaSuccess) {
            cudaGetLastError();
            ok = false;
            return nullptr;
        }
        bufs_.push_back(p);
        return p;
    }

    // ---- the window's acceptance -----------------------------------------------------------------------------------------
    // The last window's rows before `pos` were kept: they go into each Mamba-2 layer's state and history.  pos 0 starts a
    // new conversation.
    bool settle(int64_t pos, std::string& err) {
        if (pos == 0) {
            pend_T_ = 0;
            alt_ready_ = false;
            mtp_valid_ = hq_pos_ = hq_n_ = 0;
            tok_hist_.clear();
            for (const Mamba& mb : mamba_) {
                cudaMemsetAsync(mb.state, 0, sg_.state_floats() * 4, stream_);
                cudaMemsetAsync(mb.hist, 0, sg_.hist_floats() * 4, stream_);
            }
            return true;
        }
        if (pend_T_ == 0) return true;
        const int64_t n = pos - pend_pos_;
        const int T = pend_T_;
        pend_T_ = 0;
        if (n < 0 || n > T) {
            err = "position " + std::to_string((long long) pos) + " is not reachable from the last window";
            return false;
        }
        const bool swap = alt_ready_ && T == 1 && n == 1;
        alt_ready_ = false;
        if (n == 0) return true;
        if (swap) {
            for (Mamba& mb : mamba_) {
                std::swap(mb.state, mb.alt);
                bk::ssd_conv_advance(sg_, mb.hist, mb.raw, g_.conv_dim(), 1, stream_);
            }
            return true;
        }
        *keep_h_ = (int32_t) n;
        cudaMemcpyAsync(keep_d_, keep_h_, 4, cudaMemcpyHostToDevice, stream_);
        for (const Mamba& mb : mamba_) {
            bk::ssd_commit(sg_, mb.state, mb.xbc, mb.dt, mb.dA, keep_d_, stream_);
            bk::ssd_conv_commit(sg_, mb.hist, mb.raw, g_.conv_dim(), keep_d_, stream_);
        }
        return true;
    }

    // ---- the forward pass ------------------------------------------------------------------------------------------------
    // R rows at pos0..: a decode window (`window`, R <= 16) or a prompt chunk; the logits of the last `head` rows to logits_.
    bool run_rows(const int32_t* tokens, int R, int64_t pos0, bool window, int head, std::string& err) {
        bool ok = run_rows_impl(tokens, R, pos0, window, head, err);
        if (!join_moe(err)) ok = false;
        if (!ok) cudaStreamSynchronize(stream_);
        return ok;
    }
    bool run_rows_impl(const int32_t* tokens, int R, int64_t pos0, bool window, int head, std::string& err) {
        if (pos0 + R > m_.context()) {
            err = "position " + std::to_string(pos0 + R - 1) + " is past the context of " + std::to_string(m_.context());
            return false;
        }
        for (int t = 0; t < R; ++t)
            if (tokens[t] < 0 || tokens[t] >= g_.n_vocab) {
                err = "token " + std::to_string(tokens[t]) + " is out of the vocabulary";
                return false;
            }
        const int64_t E = g_.n_embd;
        if (mtp_on_) {   // the tokens by position (a call that starts earlier drops those after it)
            tok_hist_.resize((size_t) pos0, -1);
            tok_hist_.insert(tok_hist_.end(), tokens, tokens + R);
        }
        cudaMemcpyAsync(tok_, tokens, (size_t) R * 4, cudaMemcpyHostToDevice, stream_);
        bk::iq_embed_rows(kTypeBf16, m_.embedding(), (size_t) E * 2, tok_, R, E, X_, stream_);
        int mi = 0, ai = 0;
        for (int l = 0; l < g_.n_layers; ++l) {
            const Layer& L = m_.layer(l);
            bool ok = true;
            if (L.kind == Kind::Mamba) ok = window ? mamba_window(L, mamba_[(size_t) mi++], R) : mamba_chunk(L, mamba_[(size_t) mi++], R);
            else if (L.kind == Kind::Attention) ok = window ? attn_window(L, ai++, R, pos0) : attn_chunk(L, ai++, R, pos0);
            else ok = window ? moe_window(l, L, R, err) : moe_chunk(L, R, err);
            if (!ok) {
                if (err.empty()) err = "layer " + std::to_string(l) + " failed";
                return false;
            }
        }
        if (mtp_on_) queue_rows(pos0, R);
        if (head > 0) {
            const Mat& h = m_.head();
            wg::norm_pack(X_ + (size_t) (R - head) * E, m_.out_norm(), g_.rms_eps, E, head, act_, nullptr, stream_);
            gemm(h, logits_, g_.n_vocab, head);
        }
        const cudaError_t ce = cudaGetLastError();
        if (ce != cudaSuccess) {
            err = std::string("a kernel failed: ") + cudaGetErrorString(ce);
            return false;
        }
        return true;
    }

    // y[t * ld + r] (= or +=) the window gemm product of mat over act_'s rows [0, T)
    void gemm(const Mat& w, float* y, int64_t ld, int T, bool add = false) {
        wg::Call c;
        c.fmt = w.type == kTypeF8 ? wg::Format::F8 : w.type == kTypeNvfp4Planes ? wg::Format::Nvfp4 : wg::Format::Bf16;
        c.K = w.cols;
        c.epi = add ? wg::Epilogue::Add : wg::Epilogue::Store;
        c.m[0] = wg::Matrix{w.w, w.rows, w.scale, y, ld};
        wg::gemm(c, act_, T, ws_, stream_);
    }
    // a BF16 projection of R rows (any R): window gemm calls of 16 rows
    void bf16_rows(const Mat& w, const float* x, int64_t ld_x, float* y, int R) {
        for (int t0 = 0; t0 < R; t0 += kMaxWindow) {
            const int n = std::min(kMaxWindow, R - t0);
            wg::pack(x + (size_t) t0 * ld_x, ld_x, w.cols, n, act_, stream_);
            gemm(w, y + (size_t) t0 * w.rows, w.rows, n);
        }
    }

    bool mamba_window(const Layer& L, Mamba& mb, int R) {
        const int64_t E = g_.n_embd, PW = g_.proj_width(), DI = g_.ssm_inner, C = g_.conv_dim();
        wg::norm_pack(X_, L.norm, g_.rms_eps, E, R, act_, nullptr, stream_);
        gemm(L.in, proj_, PW, R);
        cudaMemcpy2DAsync(mb.raw, (size_t) C * 4, proj_ + DI, (size_t) PW * 4, (size_t) C * 4, (size_t) R, cudaMemcpyDeviceToDevice, stream_);
        bk::ssd_conv(sg_, mb.raw, C, mb.hist, L.conv_w, L.conv_b, R, mb.xbc, stream_);
        bk::ssd_scan(sg_, mb.state, mb.xbc, proj_ + DI + C, PW, L.dt_bias, L.A, L.D, R, y_, mb.dt, mb.dA, R == 1 ? mb.alt : nullptr, stream_);
        bk::ssd_gated_norm(sg_, y_, proj_, PW, L.ssm_norm, g_.rms_eps, R, nullptr, (uint16_t*) act_, stream_);
        gemm(L.out, X_, E, R, true);
        return true;
    }

    bool mamba_chunk(const Layer& L, Mamba& mb, int R) {
        const int64_t E = g_.n_embd, PW = g_.proj_width(), DI = g_.ssm_inner, C = g_.conv_dim();
        prefill::fp8::rms_norm_quantize(X_, L.norm, g_.rms_eps, R, E, xq8_, nullptr, stream_);
        prefill::fp8::gemm(L.in.w, L.in.rows, L.in.cols, L.in.scale, xq8_, R, proj_, PW, prefill::fp8::Epilogue::Store, stream_);
        bk::ssd_conv(sg_, proj_ + DI, PW, mb.hist, L.conv_w, L.conv_b, R, xbc_, stream_);
        bk::ssd_scan(sg_, mb.state, xbc_, proj_ + DI + C, PW, L.dt_bias, L.A, L.D, R, y_, nullptr, nullptr, mb.state, stream_);
        bk::ssd_conv_advance(sg_, mb.hist, proj_ + DI, PW, R, stream_);
        bk::ssd_gated_norm(sg_, y_, proj_, PW, L.ssm_norm, g_.rms_eps, R, gn_, nullptr, stream_);
        prefill::fp8::quantize(gn_, R, DI, DI, xq8_, stream_);
        prefill::fp8::gemm(L.out.w, L.out.rows, L.out.cols, L.out.scale, xq8_, R, X_, E, prefill::fp8::Epilogue::Add, stream_);
        return true;
    }

    bool attn_window(const Layer& L, int ai, int R, int64_t pos0) {
        const int64_t E = g_.n_embd, QW = (int64_t) g_.n_head * g_.head_dim, KW = (int64_t) g_.n_kv * g_.head_dim;
        wg::norm_pack(X_, L.norm, g_.rms_eps, E, R, act_, nullptr, stream_);
        wg::Call c;
        c.fmt = L.q.type == kTypeF8 ? wg::Format::F8 : wg::Format::Bf16;   // the MTP head's attention is BF16
        c.K = E;
        c.n = 3;
        c.m[0] = wg::Matrix{L.q.w, L.q.rows, L.q.scale, q_, QW};
        c.m[1] = wg::Matrix{L.k.w, L.k.rows, L.k.scale, k_, KW};
        c.m[2] = wg::Matrix{L.v.w, L.v.rows, L.v.scale, v_, KW};
        wg::gemm(c, act_, R, ws_, stream_);
        kv_store(k_, v_, R, pos0, g_.n_kv, g_.head_dim, kc_[(size_t) ai], vc_[(size_t) ai], stream_);
        bk::gqa_decode_attn(q_, kc_[(size_t) ai], vc_[(size_t) ai], g_.n_head, g_.n_kv, pos0, R, attw_, m_.context(), att_, stream_,
                            nullptr, {}, nullptr, nullptr, {}, g_.head_dim);
        wg::pack(att_, QW, QW, R, act_, stream_);
        gemm(L.o, X_, E, R, true);
        return true;
    }

    bool attn_chunk(const Layer& L, int ai, int R, int64_t pos0) {
        const int64_t E = g_.n_embd, QW = (int64_t) g_.n_head * g_.head_dim, KW = (int64_t) g_.n_kv * g_.head_dim;
        if (L.q.type == kTypeBf16) {   // the MTP head's: BF16 projections in window gemm calls
            bk::rms_norm_rows(X_, L.norm, XN_, R, E, g_.rms_eps, stream_);
            bf16_rows(L.q, XN_, E, q_, R);
            bf16_rows(L.k, XN_, E, k_, R);
            bf16_rows(L.v, XN_, E, v_, R);
            kv_store(k_, v_, R, pos0, g_.n_kv, g_.head_dim, kc_[(size_t) ai], vc_[(size_t) ai], stream_);
            bk::gqa_prompt_attn(q_, kc_[(size_t) ai], vc_[(size_t) ai], g_.n_head, g_.n_kv, pos0, R, att_, stream_, {}, nullptr, 0, {},
                                g_.head_dim);
            for (int t0 = 0; t0 < R; t0 += kMaxWindow) {
                const int n = std::min(kMaxWindow, R - t0);
                wg::pack(att_ + (size_t) t0 * QW, QW, QW, n, act_, stream_);
                gemm(L.o, X_ + (size_t) t0 * E, E, n, true);
            }
            return true;
        }
        prefill::fp8::rms_norm_quantize(X_, L.norm, g_.rms_eps, R, E, xq8_, nullptr, stream_);
        prefill::fp8::gemm(L.q.w, L.q.rows, L.q.cols, L.q.scale, xq8_, R, q_, QW, prefill::fp8::Epilogue::Store, stream_);
        prefill::fp8::gemm(L.k.w, L.k.rows, L.k.cols, L.k.scale, xq8_, R, k_, KW, prefill::fp8::Epilogue::Store, stream_);
        prefill::fp8::gemm(L.v.w, L.v.rows, L.v.cols, L.v.scale, xq8_, R, v_, KW, prefill::fp8::Epilogue::Store, stream_);
        kv_store(k_, v_, R, pos0, g_.n_kv, g_.head_dim, kc_[(size_t) ai], vc_[(size_t) ai], stream_);
        bk::gqa_prompt_attn(q_, kc_[(size_t) ai], vc_[(size_t) ai], g_.n_head, g_.n_kv, pos0, R, att_, stream_, {}, nullptr, 0, {},
                            g_.head_dim);
        prefill::fp8::quantize(att_, R, QW, QW, xq8_, stream_);
        prefill::fp8::gemm(L.o.w, L.o.rows, L.o.cols, L.o.scale, xq8_, R, X_, E, prefill::fp8::Epilogue::Add, stream_);
        return true;
    }

    // ---- the MTP head ----------------------------------------------------------------------------------------------------
    static constexpr int kQueueRows = kChunk + kMaxWindow + 1;   // a chunk's rows and the one before it, or a window's

    // The model's normed hidden rows of R rows at pos0 (X_ after the last layer) into the queue: its rows are positions
    // hq_pos_ .. hq_pos_ + hq_n_; a call that starts earlier drops those from pos0 on, and the head's cache is good only
    // before pos0.
    void queue_rows(int64_t pos0, int R) {
        const int64_t E = g_.n_embd;
        mtp_valid_ = std::min(mtp_valid_, pos0);
        if (hq_n_ == 0 || pos0 < hq_pos_ || pos0 > hq_pos_ + hq_n_ || pos0 - hq_pos_ + R > kQueueRows) {
            hq_pos_ = pos0;   // a fresh queue: a row before pos0 it lacks leaves the head's cache as it is there
            hq_n_ = 0;
        }
        hq_n_ = pos0 - hq_pos_;
        bk::rms_norm_rows(X_, m_.out_norm(), hq_ + (size_t) hq_n_ * E, R, E, g_.rms_eps, stream_);
        hq_n_ += R;
    }

    // The head's cache through position p_end - 1: the pairs (h_p, t_{p+1}) of the queued rows from mtp_valid_ on, t_{p_end}
    // `last_tok` (or the token history's when -1).  `want_head`: the last row's draft logits to dlogits_ and its output to hm_
    // (head_ready_ says they are there).  The queue keeps its rows from p_end - 1 on.
    bool mtp_flush(int64_t p_end, int32_t last_tok, bool want_head, std::string& err) {
        const int64_t E = g_.n_embd;
        head_ready_ = false;
        if (mtp_valid_ < hq_pos_) mtp_valid_ = hq_pos_;   // positions the queue never held keep what the cache has
        const int64_t end = std::min(p_end, hq_pos_ + hq_n_);
        std::vector<int32_t> toks;
        for (int64_t p = mtp_valid_; p < end;) {
            const int n = (int) std::min<int64_t>(kChunk, end - p);
            toks.assign((size_t) n, -1);
            for (int i = 0; i < n; ++i) {
                const int64_t q = p + i + 1;
                toks[(size_t) i] = q == p_end && last_tok >= 0 ? last_tok : q < (int64_t) tok_hist_.size() ? tok_hist_[(size_t) q] : -1;
                if (toks[(size_t) i] < 0 || toks[(size_t) i] >= g_.n_vocab) {
                    err = "the MTP head has no token at position " + std::to_string((long long) q);
                    return false;
                }
            }
            const bool head = want_head && p + n == p_end;
            if (!mtp_rows(hq_ + (size_t) (p - hq_pos_) * E, toks.data(), n, p, head, err)) return false;
            head_ready_ = head;
            p += n;
        }
        mtp_valid_ = std::max(mtp_valid_, end);
        queue_from(end - 1);
        return true;
    }

    // The queue's rows from position `pos` on to its front (through proj_: they may overlap).
    void queue_from(int64_t pos) {
        if (pos <= hq_pos_) return;
        const int64_t E = g_.n_embd, drop = pos - hq_pos_, keep = hq_n_ - drop;
        if (keep > 0) {
            cudaMemcpyAsync(proj_, hq_ + (size_t) drop * E, (size_t) keep * E * 4, cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(hq_, proj_, (size_t) keep * E * 4, cudaMemcpyDeviceToDevice, stream_);
        }
        hq_pos_ = pos;
        hq_n_ = std::max<int64_t>(keep, 0);
    }

    // n rows of the head at pos0: the fusion of enorm(embedding of toks) and hnorm(h), its attention layer and its MoE
    // layer; with `head` the last row's normed output to hm_ and its logits to dlogits_.
    bool mtp_rows(const float* h, const int32_t* toks, int n, int64_t pos0, bool head, std::string& err) {
        const int64_t E = g_.n_embd;
        const MtpHead& H = m_.mtp();
        const int la = g_.n_layers, lm = g_.n_layers + 1;
        if (pos0 + n > m_.context()) {
            err = "the MTP head's position " + std::to_string((long long) (pos0 + n - 1)) + " is past the context";
            return false;
        }
        cudaMemcpyAsync(tok_, toks, (size_t) n * 4, cudaMemcpyHostToDevice, stream_);
        bk::iq_embed_rows(kTypeBf16, m_.embedding(), (size_t) E * 2, tok_, n, E, XN_, stream_);
        float* cat = proj_;   // [n][2 E]: the normed embedding, then the normed hidden row
        bk::rms_norm_rows(XN_, H.enorm, cat, n, E, g_.rms_eps, stream_, 2 * E);
        bk::rms_norm_rows(h, H.hnorm, cat + E, n, E, g_.rms_eps, stream_, 2 * E);
        const bool window = n <= kMaxWindow;
        if (window) {
            wg::pack(cat, 2 * E, 2 * E, n, act_, stream_);
            gemm(H.eh, X_, E, n);
        } else {
            bf16_rows(H.eh, cat, 2 * E, X_, n);
        }
        const int ai = (int) kc_.size() - 1;   // the head's cache, after the model's
        join_now_ = window && !(mtp_warmed_ >> n & 1u);   // as a window's first run of n rows (decode_window)
        bool ok = window ? attn_window(m_.layer(la), ai, n, pos0) : attn_chunk(m_.layer(la), ai, n, pos0);
        ok = ok && (window ? moe_window(lm, m_.layer(lm), n, err) : moe_chunk(m_.layer(lm), n, err));
        if (!join_moe(err)) ok = false;
        if (!ok) {
            cudaStreamSynchronize(stream_);
            if (err.empty()) err = "the MTP head failed";
            return false;
        }
        if (window) mtp_warmed_ |= 1u << n;
        if (head) {
            bk::rms_norm_rows(X_ + (size_t) (n - 1) * E, H.final_norm, hm_, 1, E, g_.rms_eps, stream_);
            wg::pack(hm_, E, E, 1, act_, stream_);
            gemm(m_.head(), dlogits_, g_.n_vocab, 1);
        }
        return true;
    }

    // The likeliest token of dlogits_ and its log probability.
    bool draft_top(int32_t& id, float& logprob, std::string& err) {
        float* out = dlp_ + bk::row_logprobs_scratch_floats(1, g_.n_vocab);
        bk::row_logprobs(dlogits_, 1, g_.n_vocab, dpick_, 1, out, dlp_, stream_);
        float top[2];
        if (cudaMemcpyAsync(top, out + 2, 8, cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "the draft's logits failed";
            return false;
        }
        std::memcpy(&id, &top[0], 4);
        logprob = top[1];
        return id >= 0 && id < g_.n_vocab;
    }

    // ---- the MoE ---------------------------------------------------------------------------------------------------------
    bk::RouterArgs router_args() const {
        bk::RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = g_.n_embd;
        ra.k = g_.n_used;
        ra.weight_scale = g_.expert_scale;
        ra.gate = bk::RouterGate::Sigmoid;
        return ra;
    }

    // the router and the latent of R normed rows (XN_, act_ packed for a window)
    void route(const Layer& L, int R, bool window) {
        bk::router(XN_, L.router, L.router_bias, R, router_args(), rlog_, ids_, rw_, stream_);
        if (window) gemm(L.lat_down, lat_, g_.n_latent, R);
        else bf16_rows(L.lat_down, XN_, g_.n_embd, lat_, R);
    }
    // the shared expert of R normed rows (act_ packed for a window, else XN_)
    void shared_expert(const Layer& L, int R, bool window) {
        const int64_t E = g_.n_embd, FS = g_.n_ff_shared;
        if (window) {
            gemm(L.sh_up, shh_, FS, R);
            relu2(shh_, (int64_t) R * FS, stream_);
            wg::pack(shh_, FS, FS, R, act_, stream_);
            gemm(L.sh_down, sho_, E, R);
        } else {
            prefill::fp4_dense::quantize(XN_, R, E, E, a4_, gs4_, stream_);
            prefill::fp4_dense::product(L.sh_up.w, L.sh_up.scale, FS, E, a4_, gs4_, R, shh_, FS, false, stream_);
            relu2(shh_, (int64_t) R * FS, stream_);
            prefill::fp4_dense::quantize(shh_, R, FS, FS, a4_, gs4_, stream_);
            prefill::fp4_dense::product(L.sh_down.w, L.sh_down.scale, E, FS, a4_, gs4_, R, sho_, E, false, stream_);
        }
    }
    // the latent's projection back and the residual
    void moe_tail(const Layer& L, int R) {
        bf16_rows(L.lat_up, latout_, g_.n_latent, moe_, R);
        add2(X_, moe_, sho_, (int64_t) R * g_.n_embd, stream_);
    }

    // A window's MoE layer: the routing, the first group's rows and ids published, the shared expert (it runs while the
    // host and the CPU work), each group's experts, the tail.  Nothing here waits for the card.
    bool moe_window(int l, const Layer& L, int R, std::string& err) {
        wg::norm_pack(X_, L.norm, g_.rms_eps, g_.n_embd, R, act_, XN_, stream_);
        route(L, R, true);
        publish(0, std::min(kMoeGroup, R));
        shared_expert(L, R, true);
        for (int g0 = 0; g0 < R; g0 += kMoeGroup)
            if (!moe_group(l, L, g0, std::min(kMoeGroup, R - g0), err)) return false;
        moe_tail(L, R);
        return true;
    }

    // Rows g0..g0+G of the latent and their ids to the mapped staging, behind the sequence word the host waits on; the
    // rows quantized for the card's experts.  Queued behind the group before's sum, so its CPU rows were read.
    void publish(int g0, int G) {
        const int K = g_.n_used, Ln = g_.n_latent;
        bk::doorbell_publish(lat_ + (size_t) g0 * Ln, ids_ + (size_t) g0 * K, nullptr, (int64_t) G * Ln, (int64_t) G * K, d_x_, d_wids_,
                             nullptr, d_words_, stream_);
        ++seq_want_;
        bk::quantize_q8_1_rows(lat_ + (size_t) g0 * Ln, G, Ln, exq_, stream_);
    }

    // The published ids have landed (the stream goes on meanwhile); a launch that failed never writes them.
    bool await_ids(std::string& err) {
        const volatile uint32_t* seq = h_words_;
        for (int s = 0; s < (1 << 22); ++s) {
            if (*seq == seq_want_) {
                std::atomic_thread_fence(std::memory_order_acquire);
                return true;
            }
            platform::cpu_relax();
        }
        cudaStreamSynchronize(stream_);
        if (*seq == seq_want_) return true;
        err = "the router's ids did not arrive";
        seq_want_ = *seq;
        return false;
    }

    // While a layer's weighted sum waits for the CPU's rows, the card's memory idles once its own experts are done: the
    // weights the card reads next (this layer's latent projection back, then the next layer's input products, or the head
    // after the last layer) are read into L2 meanwhile, up to 60% of the L2, and the requests stop once the wait is over.
    // They change no result.
    void read_ahead(int l, uint32_t word) {
        if (ra_.empty()) {   // every layer's ranges, once
            int l2 = 0;
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
            const uint64_t budget = (uint64_t) l2 * 3 / 5;
            ra_.resize((size_t) g_.all_layers());
            for (int k = 0; k < g_.all_layers(); ++k) {
                bk::L2Ranges& r = ra_[(size_t) k];
                uint64_t left = budget;
                int n = 0;
                const auto add = [&](const Mat& w) {
                    const uint64_t bytes = w.type == kTypeF8 ? (uint64_t) w.rows * w.cols : w.type == kTypeBf16 ? (uint64_t) w.rows * w.cols * 2 : 0;
                    if (n == bk::kL2Ranges || bytes == 0 || ((uintptr_t) w.w & 15) != 0) return;
                    r.p[n] = w.w;
                    r.n[n] = std::min<uint64_t>(bytes, left) & ~15ull;
                    left -= r.n[n];
                    n += r.n[n] > 0;
                };
                const Layer& L = m_.layer(k);
                if (L.kind == Kind::Moe) add(L.lat_up);
                if (k + 1 < g_.n_layers) {   // the model's next layer; after its last and after the MTP head's, the head
                    const Layer& N = m_.layer(k + 1);
                    if (N.kind == Kind::Mamba) add(N.in);
                    else if (N.kind == Kind::Attention) {
                        add(N.q);
                        add(N.k);
                        add(N.v);
                    }
                } else {
                    add(m_.head());
                }
            }
        }
        cudaEventRecord(ra_ev_, stream_);
        cudaStreamWaitEvent(ra_st_, ra_ev_, 0);
        bk::l2_read_ahead(ra_[(size_t) l], ra_mark_, word, ra_st_);
    }

    // A group of G rows from window row g0 through the tier: the CPU pool starts on the RAM experts, the card runs its slot
    // hits and then the bus fetches, and the weighted sum is queued behind the word the joiner writes once the CPU's rows
    // landed.  The host does not wait here: join_moe, called at the next group or the window's end, finishes the share.
    bool moe_group(int l, const Layer& L, int g0, int G, std::string& err) {
        const int K = g_.n_used, Ln = g_.n_latent, tl = L.tier_layer;
        if (g0 > 0) publish(g0, G);
        if (!join_moe(err)) return false;
        Pending& pd = pend_;
        pd = Pending{};
        pd.on = true;
        pd.tl = tl;
        // the second card's share goes out while the head runs: its kernels wait on the card for the plan
        core::PeerExperts* peer = tier_->peer_for(tl);
        if (peer) {
            if (!peer->arm(lay_, h_x_, G, K, core::policy::peer_groups(G), err)) return false;
            pd.peer = peer;
        }
        tier_->pump(tl, 2 + G);   // what this layer's head leaves the bus of the promotions queued before it
        if (!await_ids(err)) return false;
        int32_t uniq[kLookupK];
        int uidx[kLookupK], nu = 0;
        core::ExpertTier::Use* use = pd.use;
        for (int t = 0; t < G; ++t)
            for (int k = 0; k < K; ++k) {
                const int32_t e = h_wids_[t * K + k];
                if (e < 0 || e >= g_.n_expert) {
                    err = "the router picked expert " + std::to_string(e);
                    return false;
                }
                int u = 0;
                while (u < nu && uniq[u] != e) ++u;
                if (u == nu) {
                    uniq[nu++] = e;
                    use[u] = core::ExpertTier::Use{};
                }
                uidx[t * K + k] = u;
                core::ExpertTier::Use& us = use[u];
                us.tok[us.nt] = t;
                us.row[us.nt] = t * K + k;
                ++us.nt;
            }
        core::ExpertTier::Lookup& lk = pd.lk;
        if (!tier_->lookup(tl, uniq, nu, lk, err)) return false;
        const int n_fetched = tier_->fetch(tl, lk);
        // the second card's plan first: its share waits for it, and its rows are the last of the two cards' to land
        if (peer) {
            int32_t kind[kLookupK] = {};
            for (int r = 0; r < G * K; ++r) kind[r] = !lk.peer[uidx[r]] ? 0 : lk.fetched[uidx[r]] ? 3 : 2;
            if (!peer->raise(tl, h_wids_, kind, err)) return false;
            pd.peer_raised = true;
        }
        bool on_peer = false;
        for (int i = 0; i < nu; ++i) {
            pd.cpu = pd.cpu || (!lk.dev[i] && !lk.peer[i] && !lk.reading[i]);
            pd.reading = pd.reading || lk.reading[i];
            on_peer = on_peer || lk.peer[i];
        }
        if (pd.cpu || pd.reading || on_peer) pd.word = ++join_seq_;   // the weighted sum waits for the host's rows
        // a promotion only queues its copies (they go out at a later pump); experts still being read from the file are
        // decided once they landed
        if (pd.cpu && !pd.reading) tier_->promote(tl, lk);
        // the CPU's share first: it takes longest, and the pool's workers run it while the card's share launches
        if (pd.cpu) tier_->start_cpu_window(tl, h_x_, lk, use, h_rows_);
        const auto make_plan = [&](WindowPlan& p, bool fetched, int& ne) {
            int ng = 0;
            ne = 0;
            for (int i = 0; i < nu; ++i) {
                if (!lk.dev[i] || lk.fetched[i] != fetched) continue;
                p.ptr[ng] = (unsigned long long) lk.dev[i];
                p.start[ng] = ne;
                for (int j = 0; j < use[i].nt; ++j) {
                    p.dst[ne] = use[i].row[j];
                    p.tok[ne] = use[i].tok[j];
                    ++ne;
                }
                ++ng;
            }
            p.start[ng] = ne;
            p.ng[0] = ng;
            return ng;
        };
        const auto launch = [&](void* dev, int ng, int ne) {
            WindowPlan* dp = (WindowPlan*) dev;
            bk::native_expert_grouped(lay_, dp->ptr, dp->start, dp->ng, dp->dst, dp->tok, ng, ne, exq_, exscr_, exrows_, stream_);
        };
        WindowPut put;
        int ne = 0;
        const int nhits = make_plan(put.plan, false, ne);
        for (int r = 0; r < G * K; ++r)
            put.rows[r] = (unsigned long long) (lk.dev[uidx[r]] ? exrows_ + (size_t) r * Ln : d_rows_ + (size_t) r * Ln);
        bk::put_words(&put, ex_plan_, (int) ((offsetof(WindowPut, rows) + (size_t) G * K * 8) / 4), stream_);
        if (nhits) launch(ex_plan_, nhits, ne);
        if (pd.word) read_ahead(l, pd.word);
        if (n_fetched) {
            WindowPlan p2;
            const int ng = make_plan(p2, true, ne);
            tier_->await_fetch(stream_);
            bk::put_words(&p2, ex_plan2_, (int) (sizeof(WindowPlan) / 4), stream_);
            launch(ex_plan2_, ng, ne);
        }
        // the second card's fetched experts run once their copies land, long after these launches
        if (peer && !peer->run_fetched(err)) return false;
        if (pd.word) bk::wait_word_mark(d_words_ + 1, pd.word, ra_mark_, stream_);
        const float* const* rows = (const float* const*) ((uint8_t*) ex_plan_ + offsetof(WindowPut, rows));
        bk::weighted_sum_window(rows, rw_ + (size_t) g0 * K, G, K, Ln, latout_ + (size_t) g0 * Ln, stream_);
        if (pd.cpu && !pd.reading) tier_->mark_gap(stream_);   // the promotions' copies start behind the sum
        // the rows land and the word is written on the joiner thread, while this thread queues the next layers' work
        if (join_now_) return join_moe(err);
        if (pd.word && !pd.reading) {
            if (!joiner_.joinable()) joiner_ = std::thread([this] { joiner_loop(); });
            pd.handed = true;
            {
                std::lock_guard<std::mutex> g(jm_);
                jwork_ = true;
                jdone_.store(false, std::memory_order_relaxed);
            }
            jcv_.notify_one();
        }
        return true;
    }

    void joiner_loop() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(jm_);
                jcv_.wait(lk, [this] { return jwork_ || jstop_; });
                if (jstop_ && !jwork_) return;
                jwork_ = false;
            }
            std::string e;
            jok_ = join_body(e);
            jerr_ = e;
            jdone_.store(true, std::memory_order_release);
        }
    }

    // The CPU's share of the last group placed: the pool's rows land, the experts read from the file run, and the word its
    // weighted sum waits for is written.  On every way out, so the stream never waits for good.
    bool join_moe(std::string& err) {
        if (!pend_.on) return true;
        if (pend_.handed) {   // handed to the joiner: its result
            while (!jdone_.load(std::memory_order_acquire)) platform::cpu_relax();
            pend_.on = false;
            if (!jok_ && err.empty()) err = jerr_;
            return jok_;
        }
        const bool ok = join_body(err);
        pend_.on = false;
        return ok;
    }
    bool join_body(std::string& err) {
        Pending& pd = pend_;
        bool ok = true;
        if (pd.cpu) tier_->finish_cpu();
        if (pd.reading) {
            if (tier_->await_reads(err)) {
                tier_->start_cpu_window(pd.tl, h_x_, pd.lk, pd.use, h_rows_, /*reading=*/true);
                tier_->finish_cpu();
            } else {
                ok = false;
            }
            tier_->promote(pd.tl, pd.lk);
        }
        // the second card's rows have landed (an armed share not raised is raised empty, so the card is not left waiting)
        std::string pe;
        if (pd.peer && pd.peer_raised && !pd.peer->finish(nullptr, pe)) ok = false;
        if (pd.peer && !pd.peer_raised) pd.peer->raise(0, nullptr, nullptr, pe);
        if (!pe.empty() && err.empty()) err = pe;
        if (pd.word) {
            std::atomic_thread_fence(std::memory_order_release);   // the rows before the word
            *(volatile uint32_t*) (h_words_ + 1) = pd.word;
        }
        if (pd.reading) tier_->mark_gap(stream_);
        return ok;
    }

    // The expert ring of a long prompt: the tier lends the room of two layers' experts for the prompt (and takes it back,
    // with the experts it held, at its end); without the room the prompt stages its experts a group at a time.
    size_t ring_half() const { return (size_t) g_.n_expert * m_.expert_stride(); }
    void open_ring() {
        if (ring_ || !tier_) return;
        if (!tier_->partial_lend() || !tier_->lend(2 * ring_half())) {
            basalt::log::info("basalt nemotronh", "cannot borrow the expert ring (%.1f GB): the prompt stages its experts a group at a time",
                              (double) (2 * ring_half()) / 1e9);
            return;
        }
        if (cudaMalloc((void**) &ring_, 2 * ring_half()) != cudaSuccess) {
            cudaGetLastError();
            ring_ = nullptr;
            std::string e;
            if (!tier_->reclaim(e, /*restore=*/true)) basalt::log::warn("basalt nemotronh", "the expert tier stays small: %s", e.c_str());
            return;
        }
        ring_layer_[0] = ring_layer_[1] = -1;
        ring_last_ = 1;
    }
    void close_ring() {
        if (!ring_) return;
        cudaStreamSynchronize(copy_);
        cudaStreamSynchronize(stream_);
        cudaFree(ring_);
        ring_ = nullptr;
        ring_layer_[0] = ring_layer_[1] = -1;
        std::string e;
        if (tier_ && !tier_->reclaim(e, /*restore=*/true)) basalt::log::warn("basalt nemotronh", "the expert tier stays small: %s", e.c_str());
    }
    // The MoE layer a prompt chunk runs after layer li: the model's next, then the MTP head's, then (another chunk to come)
    // the first again; -1 when none.
    int next_moe(int li) const {
        const int n = g_.n_layers;
        if (li < n)
            for (int l = li + 1; l < n; ++l)
                if (g_.kind(l) == Kind::Moe) return l;
        if (li < n && mtp_on_) return n + 1;
        if (!ring_more_) return -1;
        for (int l = 0; l < n; ++l)
            if (g_.kind(l) == Kind::Moe) return l;
        return -1;
    }
    // Every expert of MoE layer li into half h on the copy stream, once the layer before in that half has been read: from
    // the card's slots, runs of the RAM tier in one strided copy each, a second card, or the pack; then the tails aside.
    bool fill_ring(int h, int li, std::string& err) {
        const int tl = m_.layer(li).tier_layer;
        const size_t stride = m_.expert_stride();
        const uint64_t blob = lay_.bytes;
        uint8_t* base = ring_ + (size_t) h * ring_half();
        cudaStreamWaitEvent(copy_, ring_freed_[h], 0);
        ring_layer_[h] = li;
        for (int e = 0; e < g_.n_expert;) {
            const core::ExpertTier::Source src = tier_->locate(tl, e);
            uint8_t* dst = base + (size_t) e * stride;
            if (src.dev) {
                cudaMemcpyAsync(dst, src.dev, blob, cudaMemcpyDeviceToDevice, copy_);
                ++e;
            } else if (src.pinned) {
                int k = 1;
                while (e + k < g_.n_expert) {
                    const core::ExpertTier::Source nx = tier_->locate(tl, e + k);
                    if (nx.dev || nx.pinned != src.pinned + (size_t) k * blob) break;
                    ++k;
                }
                cudaMemcpy2DAsync(dst, stride, src.pinned, blob, blob, (size_t) k, cudaMemcpyHostToDevice, copy_);
                e += k;
            } else if (src.peer) {
                cudaMemcpyPeerAsync(dst, m_.device(), src.peer, tier_->peer_for(tl)->device(), blob, copy_);
                ++e;
            } else {
                const core::FileRange& er = m_.pack().experts_range;
                cudaStreamSynchronize(copy_);   // the bounce buffer's last copy has left
                if (!platform::pread_all(m_.fd(), bounce_, blob, er.offset + m_.pack().experts[(size_t) li].offset + (uint64_t) e * blob)) {
                    err = "reading an expert from the pack failed";
                    return false;
                }
                cudaMemcpyAsync(dst, bounce_, blob, cudaMemcpyHostToDevice, copy_);
                ++e;
            }
        }
        tails_aside(base, stride, blob, g_.n_expert, ring_tails_ + (size_t) h * g_.n_expert * 4, copy_);
        cudaEventRecord(ring_filled_[h], copy_);
        return true;
    }
    // The half MoE layer li runs from (filled now unless the layer before filled it), and the next MoE layer's fill into
    // the other half, which runs while this one computes.
    int ring_slot(int li, std::string& err) {
        int h = ring_layer_[0] == li ? 0 : ring_layer_[1] == li ? 1 : -1;
        if (h < 0) {
            h = 1 - ring_last_;
            if (!fill_ring(h, li, err)) return -1;
        }
        ring_last_ = h;
        const int nx = next_moe(li);
        if (nx >= 0 && nx != li && ring_layer_[1 - h] != nx && !fill_ring(1 - h, nx, err)) return -1;
        return h;
    }

    // A prompt chunk's routed experts on the int8 tensor cores (llama.cpp's MMQ over the NVFP4 blobs, q8_1 activations):
    // every expert its rows route to, in groups of up to kGroupExperts.  A group's blobs are copied (from a card slot, the
    // RAM tier or the pack) into one of two staging halves on the copy stream while the group before computes, at one
    // stride; each blob's scale tail is kept aside and its place zeroed, as MMQ reads a little past a down matrix.  The
    // group's rows (the latent of each token that chose the expert, gathered by expert) go through the input product, the
    // squared ReLU with both tail scales applied, and the down product, one latent row per (token, choice) for the
    // weighted sum.
    bool moe_chunk(const Layer& L, int R, std::string& err) {
        namespace mmq = basalt::prefill::mmq;
        const int64_t E = g_.n_embd, Ln = g_.n_latent, K = g_.n_used, FF = g_.n_ff_exp;
        bk::rms_norm_rows(X_, L.norm, XN_, R, E, g_.rms_eps, stream_);
        route(L, R, false);
        shared_expert(L, R, false);
        cudaMemcpyAsync(h_ids_, ids_, (size_t) R * K * 4, cudaMemcpyDeviceToHost, stream_);
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "the router failed";
            return false;
        }
        // the entries (t K + k) of each expert, the experts in id order, and each entry's gathered row
        std::vector<std::vector<int32_t>> by((size_t) g_.n_expert);
        for (int64_t r = 0; r < R * K; ++r) {
            const int32_t e = h_ids_[r];
            if (e < 0 || e >= g_.n_expert) {
                err = "the router picked expert " + std::to_string(e);
                return false;
            }
            by[(size_t) e].push_back((int32_t) r);
        }
        // with the ring every expert is in order (a group is a run of ids, those without rows empty), else the routed ones
        const int li = (int) (&L - &m_.layer(0));
        const int slot = ring_ ? ring_slot(li, err) : -1;
        if (ring_ && slot < 0) return false;
        std::vector<int> order;
        for (int e = 0; e < g_.n_expert; ++e)
            if (slot >= 0 || !by[(size_t) e].empty()) order.push_back(e);
        // the host tables of every group, uploaded once: a gathered row's token, each group's relative row bounds
        int32_t* gtok = h_gtok_;
        int32_t* gb = h_gbounds_;
        std::vector<int> g_first, g_n, g_row0;   // per group: its first expert in `order`, its experts, its first row
        int64_t row = 0;
        for (size_t i = 0; i < order.size();) {
            int n = 0;   // up to kGroupExperts experts and kGroupRows rows (an expert has at most R <= kGroupRows)
            for (int64_t rows = 0; i + (size_t) n < order.size() && n < kGroupExperts; ++n) {
                rows += (int64_t) by[(size_t) order[i + (size_t) n]].size();
                if (rows > kGroupRows) break;
            }
            g_first.push_back((int) i);
            g_n.push_back(n);
            g_row0.push_back((int) row);
            int32_t* bb = gb + (g_first.size() - 1) * (kGroupExperts + 1);
            bb[0] = 0;
            for (int j = 0; j < n; ++j) {
                for (int32_t r : by[(size_t) order[i + (size_t) j]]) {
                    gtok[row] = r / (int32_t) K;
                    rowtab_h_[r] = exrows_ + (size_t) row * Ln;
                    ++row;
                }
                bb[j + 1] = (int32_t) (row - g_row0.back());
            }
            i += (size_t) n;
        }
        const int ng = (int) g_first.size();
        cudaMemcpyAsync(d_gtok_, gtok, (size_t) row * 4, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_gbounds_, gb, (size_t) ng * (kGroupExperts + 1) * 4, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(rowtab_d_, rowtab_h_, (size_t) R * K * sizeof(float*), cudaMemcpyHostToDevice, stream_);
        cudaEventRecord(ev_ready_, stream_);   // the tables are in before any copy
        cudaStreamWaitEvent(copy_, ev_ready_, 0);
        const int tl = L.tier_layer;
        const size_t stride = m_.expert_stride();
        const uint64_t blob = lay_.bytes;
        const core::FileRange& er = m_.pack().experts_range;
        const uint64_t layer_off = m_.pack().experts[(size_t) (&L - &m_.layer(0))].offset;
        // group gi's blobs into staging half gi & 1 on the copy stream, their tails aside and zeroed
        const auto stage = [&](int gi) {
            const int h = gi & 1;
            cudaStreamWaitEvent(copy_, ev_done_[h], 0);   // the group two before has read this half
            for (int j = 0; j < g_n[(size_t) gi]; ++j) {
                const int e = order[(size_t) g_first[(size_t) gi] + (size_t) j];
                const core::ExpertTier::Source src = tier_->locate(tl, e);
                uint8_t* slot = stage_ + ((size_t) h * kGroupExperts + (size_t) j) * stride;
                if (src.dev) {
                    cudaMemcpyAsync(slot, src.dev, blob, cudaMemcpyDeviceToDevice, copy_);
                } else if (src.pinned) {
                    cudaMemcpyAsync(slot, src.pinned, blob, cudaMemcpyHostToDevice, copy_);
                } else if (src.peer) {
                    cudaMemcpyPeerAsync(slot, m_.device(), src.peer, tier_->peer_for(tl)->device(), blob, copy_);
                } else {
                    cudaStreamSynchronize(copy_);   // the bounce buffer's last copy has left
                    if (!platform::pread_all(m_.fd(), bounce_, blob, er.offset + layer_off + (uint64_t) e * blob)) return false;
                    cudaMemcpyAsync(slot, bounce_, blob, cudaMemcpyHostToDevice, copy_);
                }
                cudaMemcpyAsync(tails_d_ + ((size_t) h * kGroupExperts + (size_t) j) * 4, slot + blob - 16, 16,
                                cudaMemcpyDeviceToDevice, copy_);
                cudaMemsetAsync(slot + blob - 16, 0, stride - blob + 16, copy_);
            }
            cudaEventRecord(ev_copied_[h], copy_);
            return true;
        };
        if (slot < 0 && ng > 0 && !stage(0)) {
            err = "reading an expert from the pack failed";
            return false;
        }
        if (slot >= 0) cudaStreamWaitEvent(stream_, ring_filled_[slot], 0);
        for (int gi = 0; gi < ng; ++gi) {
            const int h = gi & 1, n = g_n[(size_t) gi];
            if (slot < 0 && gi + 1 < ng && !stage(gi + 1)) {   // the next group's copies go out while this one runs
                err = "reading an expert from the pack failed";
                return false;
            }
            const int32_t* bounds = d_gbounds_ + (size_t) gi * (kGroupExperts + 1);
            const int32_t* bh = gb + (size_t) gi * (kGroupExperts + 1);
            const int64_t rows = bh[n];
            int64_t maxr = 0;
            for (int j = 0; j < n; ++j) maxr = std::max<int64_t>(maxr, bh[j + 1] - bh[j]);
            if (rows == 0) continue;   // a run of the ring's experts no row chose
            const int e0 = order[(size_t) g_first[(size_t) gi]];
            const uint8_t* base = slot >= 0 ? ring_ + (size_t) slot * ring_half() + (size_t) e0 * stride
                                            : stage_ + (size_t) h * kGroupExperts * stride;
            const float* tails = slot >= 0 ? ring_tails_ + ((size_t) slot * g_.n_expert + (size_t) e0) * 4
                                           : tails_d_ + (size_t) h * kGroupExperts * 4;
            mmq::quantize(lat_, d_gtok_ + g_row0[(size_t) gi], mxq_, kTypeNvfp4, Ln, Ln, rows, stream_);
            if (slot < 0) cudaStreamWaitEvent(stream_, ev_copied_[h], 0);
            mmq::Product up;
            up.w = base;
            up.type = kTypeNvfp4;
            up.w_rows = FF;
            up.w_cols = Ln;
            up.expert_bytes = stride;
            up.n = n;
            up.xq = mxq_;
            up.bounds = bounds;
            up.ids = ident_d_;
            up.total_rows = rows;
            up.max_rows = maxr;
            up.dst = hbuf_;
            up.ld_dst = FF;
            mmq_->run(up, stream_);
            relu2_scaled_rows(hbuf_, rows, (int) FF, bounds, n, tails, stream_);
            mmq::quantize(hbuf_, nullptr, mhq_, kTypeNvfp4, FF, FF, rows, stream_);
            mmq::Product dn = up;
            dn.w = base + lay_.down_off;
            dn.w_rows = Ln;
            dn.w_cols = FF;
            dn.xq = mhq_;
            dn.dst = exrows_ + (size_t) g_row0[(size_t) gi] * Ln;
            dn.ld_dst = Ln;
            mmq_->run(dn, stream_);
            if (slot < 0) cudaEventRecord(ev_done_[h], stream_);
        }
        if (slot >= 0) cudaEventRecord(ring_freed_[slot], stream_);
        bk::weighted_sum_window(rowtab_d_, rw_, R, (int) K, (int) Ln, latout_, stream_);
        moe_tail(L, R);
        // the host tables are this layer's until the stream has read them
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "a prompt's experts failed";
            return false;
        }
        return true;
    }

    // The expert tier, opened by the first session while the model starts and shared by the rest.
    bool ensure_tier(std::string& err) {
        if (tier_) return true;
        NemotronModel::SharedTier& shared = m_.shared_tier();
        constexpr unsigned kMapped = cudaHostAllocPortable | cudaHostAllocMapped;
        const size_t rows = (size_t) kMaxWindow * g_.n_used;
        if (!shared.tier) {
            if (cudaHostAlloc((void**) &shared.h_x, (size_t) kMaxWindow * g_.n_latent * 4, kMapped) != cudaSuccess ||
                cudaHostAlloc((void**) &shared.h_rows, rows * g_.n_latent * 4, kMapped) != cudaSuccess) {
                err = "cannot allocate the pinned expert staging";
                return false;
            }
            auto t = std::make_shared<core::ExpertTier>(m_.tier_spec());
            core::ExpertTier::Reserve r;
            r.copies = kModuleRoom;
            r.homes = kModuleRoom;
            r.beside_homes = 1ull << 30;
            r.modules = kModuleRoom;
            core::LoadPlan* reads = m_.planning();
            if (reads == nullptr) {
                err = "the expert tier opens only while the model starts";
                return false;
            }
            if (!t->open(r, shared.h_rows, *reads, err)) return false;
            shared.tier = std::move(t);
        }
        tier_ = shared.tier;
        h_rows_ = shared.h_rows;
        h_x_ = shared.h_x;
        if (cudaHostGetDevicePointer((void**) &d_rows_, h_rows_, 0) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &d_x_, h_x_, 0) != cudaSuccess) {
            err = "cannot map the expert rows";
            return false;
        }
        return true;
    }
    static constexpr uint64_t kModuleRoom = 512ull << 20;   // the kernel modules' first loads

    bool device_bias(float*& dev, const float* host, std::string& err) {
        const size_t bytes = (size_t) g_.n_vocab * sizeof(float);
        if (dev == nullptr) {
            bool ok = true;
            dev = (float*) alloc(bytes, ok);
            if (!ok) {
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

    // The picks of the window's last `rows` rows of logits_ (row t is draw counter + t).
    bool pick(const model::DecodeRequest& r, int rows, std::string& err) {
        bk::SamplerParams sp;
        if (r.sampling) sp = *r.sampling;
        else {
            sp.greedy = true;
            sp.temperature = 0.f;
        }
        sp.logit_bias = bias_on_ ? d_bias_ : nullptr;
        history_len_ = r.history != nullptr ? r.history_len : 0;
        if (history_len_ > 0) {
            const size_t n = (size_t) kMaxWindow * history_len_;
            if (d_hist_len_ < n) {
                bool ok = true;
                d_hist_ = (int32_t*) alloc(n * 4, ok);
                if (!ok) {
                    err = "out of device memory for the sampling history";
                    return false;
                }
                d_hist_len_ = n;
            }
            cudaMemcpyAsync(d_hist_, r.history, (size_t) rows * history_len_ * 4, cudaMemcpyHostToDevice, stream_);
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

    NemotronModel& m_;
    const Geometry& g_;
    const bk::SsdGeometry sg_;
    const bool mtp_on_;   // the pack's MTP head drafts (drafting is on)
    cudaStream_t stream_ = nullptr;
    std::vector<void*> bufs_, pinned_;
    float *X_ = nullptr, *XN_ = nullptr, *proj_ = nullptr, *xbc_ = nullptr, *y_ = nullptr, *gn_ = nullptr;
    float *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr, *attw_ = nullptr;
    float *lat_ = nullptr, *latout_ = nullptr, *moe_ = nullptr, *shh_ = nullptr, *sho_ = nullptr, *rlog_ = nullptr, *rw_ = nullptr;
    float *logits_ = nullptr, *lp_ = nullptr, *gs4_ = nullptr, *exrows_ = nullptr;
    int32_t *ids_ = nullptr, *tok_ = nullptr, *pick_ = nullptr, *keep_d_ = nullptr;
    void *act_ = nullptr, *xq8_ = nullptr, *a4_ = nullptr, *exq_ = nullptr, *exscr_ = nullptr;
    wg::Workspace ws_;
    std::vector<Mamba> mamba_;
    std::vector<uint16_t*> kc_, vc_;
    bk::NativeExpertLayout lay_;
    void *ex_plan_ = nullptr, *ex_plan2_ = nullptr;   // a window group's plan and row pointers; its fetched experts' plan
    const float **rowtab_h_ = nullptr, **rowtab_d_ = nullptr;
    int32_t *h_ids_ = nullptr, *keep_h_ = nullptr;
    // a window group's ids (mapped), the sequence word and the joiner's word (mapped: [0], [1]); the L2 read-ahead
    int32_t *h_wids_ = nullptr, *d_wids_ = nullptr;
    uint32_t *h_words_ = nullptr, *d_words_ = nullptr, *ra_mark_ = nullptr;
    uint32_t seq_want_ = 0, join_seq_ = 0;
    std::vector<bk::L2Ranges> ra_;
    cudaStream_t ra_st_ = nullptr;
    cudaEvent_t ra_ev_ = nullptr;
    // the CPU's share of the last group placed, until join_moe; the thread that finishes it
    struct Pending {
        bool on = false, cpu = false, reading = false, handed = false;   // handed: to the joiner thread
        bool peer_raised = false;
        core::PeerExperts* peer = nullptr;   // the second card that holds this layer's block, armed for the group
        int tl = 0;
        uint32_t word = 0;
        core::ExpertTier::Lookup lk;
        core::ExpertTier::Use use[kLookupK];
    } pend_;
    uint32_t warmed_ = 0;     // bit T: a window of T rows ran (its kernels are loaded)
    uint32_t mtp_warmed_ = 0; // the same for the MTP head's runs
    // the MTP head: the queue of the model's normed hidden rows (positions hq_pos_ ...), the head's output row, the draft
    // logits and their top token; its cache is good through mtp_valid_ - 1; the tokens by position; the draft asked for
    float *hq_ = nullptr, *hm_ = nullptr, *dlogits_ = nullptr, *dlp_ = nullptr;
    int32_t* dpick_ = nullptr;
    int64_t hq_pos_ = 0, hq_n_ = 0, mtp_valid_ = 0;
    std::vector<int32_t> tok_hist_;
    bool head_ready_ = false;
    int32_t draft_anchor_ = 0;
    int64_t draft_pos_ = 0;
    int draft_n_ = 0;
    bool join_now_ = false;   // this window's groups are joined as they are queued
    std::thread joiner_;
    std::mutex jm_;
    std::condition_variable jcv_;
    bool jwork_ = false, jstop_ = false, jok_ = true;
    std::atomic<bool> jdone_{true};
    std::string jerr_;
    uint8_t *stage_ = nullptr, *bounce_ = nullptr;
    // the prompt path's experts: gathered rows' tokens and the groups' bounds (pinned and on the card), an identity row
    // table, the staged blobs' tails, the q8_1 input and H, H; the MMQ context; the copy stream and its events
    int32_t *h_gtok_ = nullptr, *h_gbounds_ = nullptr, *d_gtok_ = nullptr, *d_gbounds_ = nullptr, *ident_d_ = nullptr;
    float *tails_d_ = nullptr, *hbuf_ = nullptr;
    void *mxq_ = nullptr, *mhq_ = nullptr;
    std::unique_ptr<basalt::prefill::mmq::Context> mmq_;
    cudaStream_t copy_ = nullptr;
    cudaEvent_t ev_ready_ = nullptr, ev_copied_[2] = {}, ev_done_[2] = {};
    // a long prompt's expert ring: two MoE layers' every expert at e x stride (the card's room the tier lends for the
    // prompt), the layer each half holds, their tails; a half is filled on the copy stream while the other computes
    uint8_t* ring_ = nullptr;
    float* ring_tails_ = nullptr;
    int ring_layer_[2] = {-1, -1}, ring_last_ = 1;
    bool ring_more_ = false;   // another chunk of this prompt follows: the last MoE layer fills the first's half
    cudaEvent_t ring_filled_[2] = {}, ring_freed_[2] = {};
    std::shared_ptr<core::ExpertTier> tier_;
    float *h_rows_ = nullptr, *d_rows_ = nullptr, *h_x_ = nullptr, *d_x_ = nullptr;   // the CPU's rows and its input (mapped)
    // the last window, until the next call says how many of its rows were kept
    int64_t pend_pos_ = 0;
    int pend_T_ = 0;
    bool alt_ready_ = false;
    int last_rows_ = 0;
    // sampling
    bk::SamplerParams sampling_;
    std::vector<int32_t> picks_;
    int32_t* d_hist_ = nullptr;
    size_t d_hist_len_ = 0;
    int history_len_ = 0;
    bool bias_on_ = false;
    float *d_bias_ = nullptr, *d_mask_ = nullptr;
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(NemotronModel& model, std::string& err) {
    auto s = std::make_unique<Session>(model);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::nemotronh
