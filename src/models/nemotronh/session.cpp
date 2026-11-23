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
// the card computes those its slots hold, the CPU pool the rest from the pinned RAM tier, and the router's weights sum
// both on the card.  A prompt chunk stages every expert its rows use on the card (from a slot, the RAM tier or the pack)
// and computes them there with llama.cpp's MMQ (q8_1 activations against the NVFP4 blobs).
#include "models/nemotronh/model.hpp"
#include "models/nemotronh/ops.hpp"

#include "basalt/core/conversation_cache.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/ssd.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/prefill/fp4_dense.hpp"
#include "basalt/prefill/fp8_gemm.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
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
static_assert(kChunk <= kGroupRows, "an expert's rows of a chunk fit one group");
constexpr int kTypeNvfp4 = 40;

struct GroupPlan {   // a window group's grouped expert launch tables, uploaded together
    unsigned long long ptr[core::ExpertTier::kMaxK];
    int32_t start[core::ExpertTier::kMaxK + 1];
    int32_t ng;
    int32_t dst[core::ExpertTier::kMaxK];
    int32_t tok[core::ExpertTier::kMaxK];
};

class Session final : public model::ModelSession {
public:
    explicit Session(NemotronModel& m)
        : m_(m), g_(m.geo()), sg_{g_.ssm_heads, g_.ssm_head_dim, g_.ssm_state, g_.ssm_groups, g_.ssm_conv} {}

    ~Session() override {
        if (stream_) cudaStreamSynchronize(stream_);
        for (void* p : bufs_) cudaFree(p);
        for (void* p : pinned_) cudaFreeHost(p);
        if (copy_) cudaStreamSynchronize(copy_);
        for (cudaEvent_t e : {ev_ready_, ev_copied_[0], ev_copied_[1], ev_done_[0], ev_done_[1]})
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
        plan_d_ = (GroupPlan*) alloc(sizeof(GroupPlan), ok);
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
                    cudaEventCreateWithFlags(&ev_done_[i], cudaEventDisableTiming) != cudaSuccess)
                    ok = false;
        }
        const auto pin = [&](size_t n) {
            void* p = nullptr;
            if (cudaHostAlloc(&p, n, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) ok = false;
            else pinned_.push_back(p);
            return p;
        };
        plan_h_ = (GroupPlan*) pin(sizeof(GroupPlan));
        rowtab_h_ = (const float**) pin((size_t) R * K * sizeof(float*));
        h_ids_ = (int32_t*) pin((size_t) R * K * 4);
        h_lat_ = (float*) pin((size_t) R * L * 4);
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
        if (!run_rows(r.tokens, r.n_tokens, r.pos, true, head, err)) return {model::Status::Error, "nemotronh: " + err};
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
        }
        for (int t0 = 0; t0 < r.n_tokens; t0 += kChunk) {
            if (r.stop && r.stop()) break;
            const int n = std::min(kChunk, r.n_tokens - t0);
            if (!run_rows(r.tokens + t0, n, r.pos + t0, false, 0, err)) return {model::Status::Error, "nemotronh: " + err};
            if (r.progress) {
                cudaStreamSynchronize(stream_);
                r.progress(t0 + n);
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "nemotronh: the prompt failed"};
        return {};
    }

    size_t running_bytes() const override { return mamba_.size() * (sg_.state_floats() + sg_.hist_floats()) * 4; }

    model::Result running_state(int64_t pos, bool save, core::RunningState& st) override {
        if (save) {
            std::string err;
            if (!settle(pos, err)) return {model::Status::Error, "nemotronh: " + err};
        }
        for (const Mamba& mb : mamba_) {
            st.regions.push_back({mb.state, sg_.state_floats() * 4});
            st.regions.push_back({mb.hist, sg_.hist_floats() * 4});
        }
        st.stream = stream_;
        return {};
    }

    model::Result running_restored(int64_t, const uint8_t*) override {
        pend_T_ = 0;
        alt_ready_ = false;
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
        cudaMemcpyAsync(tok_, tokens, (size_t) R * 4, cudaMemcpyHostToDevice, stream_);
        bk::iq_embed_rows(kTypeBf16, m_.embedding(), (size_t) E * 2, tok_, R, E, X_, stream_);
        int mi = 0, ai = 0;
        for (int l = 0; l < g_.n_layers; ++l) {
            const Layer& L = m_.layer(l);
            bool ok = true;
            if (L.kind == Kind::Mamba) ok = window ? mamba_window(L, mamba_[(size_t) mi++], R) : mamba_chunk(L, mamba_[(size_t) mi++], R);
            else if (L.kind == Kind::Attention) ok = window ? attn_window(L, ai++, R, pos0) : attn_chunk(L, ai++, R, pos0);
            else ok = window ? moe_window(L, R, err) : moe_chunk(L, R, err);
            if (!ok) {
                if (err.empty()) err = "layer " + std::to_string(l) + " failed";
                return false;
            }
        }
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
        c.fmt = wg::Format::F8;
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

    // the router, the latent and the shared expert of R normed rows (XN_, act_ packed for a window)
    void moe_head(const Layer& L, int R, bool window) {
        const int64_t E = g_.n_embd, FS = g_.n_ff_shared;
        bk::router(XN_, L.router, L.router_bias, R, router_args(), rlog_, ids_, rw_, stream_);
        if (window) {
            gemm(L.lat_down, lat_, g_.n_latent, R);
            gemm(L.sh_up, shh_, FS, R);
            relu2(shh_, (int64_t) R * FS, stream_);
            wg::pack(shh_, FS, FS, R, act_, stream_);
            gemm(L.sh_down, sho_, E, R);
        } else {
            bf16_rows(L.lat_down, XN_, E, lat_, R);
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

    bool moe_window(const Layer& L, int R, std::string& err) {
        const int64_t E = g_.n_embd, Ln = g_.n_latent, K = g_.n_used;
        wg::norm_pack(X_, L.norm, g_.rms_eps, E, R, act_, XN_, stream_);
        moe_head(L, R, true);
        // the routing and the latent rows to the host, once for the window
        cudaMemcpyAsync(h_ids_, ids_, (size_t) R * K * 4, cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_lat_, lat_, (size_t) R * Ln * 4, cudaMemcpyDeviceToHost, stream_);
        bk::quantize_q8_1_rows(lat_, R, Ln, exq_, stream_);
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "the router failed";
            return false;
        }
        for (int g0 = 0; g0 < R; g0 += kMoeGroup)
            if (!moe_group(L, g0, std::min(kMoeGroup, R - g0), err)) return false;
        moe_tail(L, R);
        return true;
    }

    // A group of G rows from window row g0 through the tier: the card's experts, the CPU's, then the weighted sum.
    bool moe_group(const Layer& L, int g0, int G, std::string& err) {
        const int K = g_.n_used, Ln = g_.n_latent, tl = L.tier_layer;
        int32_t uniq[core::ExpertTier::kMaxK];
        int uidx[core::ExpertTier::kMaxK], nu = 0;
        core::ExpertTier::Use use[core::ExpertTier::kMaxK];
        for (int t = 0; t < G; ++t)
            for (int k = 0; k < K; ++k) {
                const int32_t e = h_ids_[(size_t) (g0 + t) * K + k];
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
        tier_->pump(tl, 2 + G);
        core::ExpertTier::Lookup lk;
        if (!tier_->lookup(tl, uniq, nu, lk, err)) return false;
        bool cpu = false, reading = false;
        for (int i = 0; i < nu; ++i) {
            cpu = cpu || (!lk.dev[i] && !lk.reading[i]);
            reading = reading || lk.reading[i];
        }
        // the CPU's share runs on the pool while the card computes its own
        if (cpu) tier_->start_cpu_window(tl, h_lat_ + (size_t) g0 * Ln, lk, use, h_rows_);
        GroupPlan& p = *plan_h_;
        int ng = 0, ne = 0;
        for (int i = 0; i < nu; ++i) {
            if (!lk.dev[i]) continue;
            p.ptr[ng] = (unsigned long long) lk.dev[i];
            p.start[ng] = ne;
            for (int j = 0; j < use[i].nt; ++j) {
                p.dst[ne] = use[i].row[j];
                p.tok[ne] = g0 + use[i].tok[j];
                ++ne;
            }
            ++ng;
        }
        p.start[ng] = ne;
        p.ng = ng;
        for (int r = 0; r < G * K; ++r)
            rowtab_h_[r] = lk.dev[uidx[r]] ? exrows_ + (size_t) r * Ln : d_rows_ + (size_t) r * Ln;
        cudaMemcpyAsync(plan_d_, plan_h_, sizeof(GroupPlan), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(rowtab_d_, rowtab_h_, (size_t) G * K * sizeof(float*), cudaMemcpyHostToDevice, stream_);
        if (ng)
            bk::native_expert_grouped(lay_, plan_d_->ptr, plan_d_->start, &plan_d_->ng, plan_d_->dst, plan_d_->tok, ng, ne, exq_,
                                      exscr_, exrows_, stream_);
        if (cpu) tier_->finish_cpu();
        if (reading) {
            if (!tier_->await_reads(err)) return false;
            tier_->start_cpu_window(tl, h_lat_ + (size_t) g0 * Ln, lk, use, h_rows_, true);
            tier_->finish_cpu();
        }
        tier_->promote(tl, lk);
        weighted_sum(rowtab_d_, rw_ + (size_t) g0 * K, G, K, Ln, latout_ + (size_t) g0 * Ln, stream_);
        tier_->mark_gap(stream_);
        // the host rows, the tables and the plan are this group's until the sum has read them
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "a group of experts failed";
            return false;
        }
        return true;
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
        moe_head(L, R, false);
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
        std::vector<int> order;
        for (int e = 0; e < g_.n_expert; ++e)
            if (!by[(size_t) e].empty()) order.push_back(e);
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
        if (ng > 0 && !stage(0)) {
            err = "reading an expert from the pack failed";
            return false;
        }
        for (int gi = 0; gi < ng; ++gi) {
            const int h = gi & 1, n = g_n[(size_t) gi];
            if (gi + 1 < ng && !stage(gi + 1)) {   // the next group's copies go out while this one runs
                err = "reading an expert from the pack failed";
                return false;
            }
            const int32_t* bounds = d_gbounds_ + (size_t) gi * (kGroupExperts + 1);
            const int32_t* bh = gb + (size_t) gi * (kGroupExperts + 1);
            const int64_t rows = bh[n];
            int64_t maxr = 0;
            for (int j = 0; j < n; ++j) maxr = std::max<int64_t>(maxr, bh[j + 1] - bh[j]);
            const uint8_t* base = stage_ + (size_t) h * kGroupExperts * stride;
            mmq::quantize(lat_, d_gtok_ + g_row0[(size_t) gi], mxq_, kTypeNvfp4, Ln, Ln, rows, stream_);
            cudaStreamWaitEvent(stream_, ev_copied_[h], 0);
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
            relu2_scaled_rows(hbuf_, rows, (int) FF, bounds, n, tails_d_ + (size_t) h * kGroupExperts * 4, stream_);
            mmq::quantize(hbuf_, nullptr, mhq_, kTypeNvfp4, FF, FF, rows, stream_);
            mmq::Product dn = up;
            dn.w = base + lay_.down_off;
            dn.w_rows = Ln;
            dn.w_cols = FF;
            dn.xq = mhq_;
            dn.dst = exrows_ + (size_t) g_row0[(size_t) gi] * Ln;
            dn.ld_dst = Ln;
            mmq_->run(dn, stream_);
            cudaEventRecord(ev_done_[h], stream_);
        }
        weighted_sum(rowtab_d_, rw_, R, (int) K, (int) Ln, latout_, stream_);
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
        if (cudaHostGetDevicePointer((void**) &d_rows_, h_rows_, 0) != cudaSuccess) {
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
    GroupPlan *plan_h_ = nullptr, *plan_d_ = nullptr;
    const float **rowtab_h_ = nullptr, **rowtab_d_ = nullptr;
    int32_t *h_ids_ = nullptr, *keep_h_ = nullptr;
    float* h_lat_ = nullptr;
    uint8_t *stage_ = nullptr, *bounce_ = nullptr;
    // the prompt path's experts: gathered rows' tokens and the groups' bounds (pinned and on the card), an identity row
    // table, the staged blobs' tails, the q8_1 input and H, H; the MMQ context; the copy stream and its events
    int32_t *h_gtok_ = nullptr, *h_gbounds_ = nullptr, *d_gtok_ = nullptr, *d_gbounds_ = nullptr, *ident_d_ = nullptr;
    float *tails_d_ = nullptr, *hbuf_ = nullptr;
    void *mxq_ = nullptr, *mhq_ = nullptr;
    std::unique_ptr<basalt::prefill::mmq::Context> mmq_;
    cudaStream_t copy_ = nullptr;
    cudaEvent_t ev_ready_ = nullptr, ev_copied_[2] = {}, ev_done_[2] = {};
    std::shared_ptr<core::ExpertTier> tier_;
    float *h_rows_ = nullptr, *d_rows_ = nullptr;
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
