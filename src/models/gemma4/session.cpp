// src/models/gemma4/session.cpp - one conversation of Gemma 4: its caches (a window ring per sliding-window layer, a cell
// per position for each full layer) and the forward of a decode window, replayed from a CUDA graph per window shape.
//
// A row's result does not depend on the other rows of its call: the products take a fixed reduction order per row, the
// experts compute per (row, slot) entry, the attention splits a row's keys by its own key count and the glue runs a block
// per row. The positions and tokens are read on the card, and the full layers' launch bound (a power of two of keys)
// only sizes their grid. A batch window (docs/architecture/batching.md) is the same forward over several sessions' rows, each
// segment's attention reading and writing its own session's caches.
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/device_copy.hpp"
#include "basalt/core/graph.hpp"
#include "basalt/kernels/expert_plan.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/log.hpp"
#include "basalt/model/batch_driver.hpp"
#include "basalt/models/gemma4/kernels.hpp"
#include "basalt/program/expert_spill.hpp"
#include "models/gemma4/experts.hpp"
#include "models/gemma4/model.hpp"
#include "models/gemma4/prompt.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace basalt::models::gemma4 {
namespace {

namespace bk = basalt::kernels;
namespace wg = basalt::kernels::wgemm;

constexpr int kStage = 4;   // pinned staging slots for a call's tokens and positions

// The forward scratch every session of the model shares (one call at a time: each call syncs its stream before it
// returns, and a batch window syncs every session's stream before it starts), and the batch windows' graphs.
struct Shared {
    float *x = nullptr, *tmp = nullptr, *mlp = nullptr, *xr = nullptr, *xn = nullptr;   // [R][E]
    float *q = nullptr, *k = nullptr, *v = nullptr, *qp = nullptr;                      // [R][qmax] / [R][kmax]
    float *g = nullptr, *u = nullptr;                                                   // [R][n_ff_pad]
    void* act = nullptr;                                                                // wgemm activations
    wg::Workspace ws;   // the NVFP4 MLP products' split-K sums (FP8 products take none: the drafters' chains never touch it)
    float *rlog = nullptr, *rw = nullptr;
    int32_t* ids = nullptr;
    void* xq = nullptr;                                                                 // q8_1 of the experts' input
    unsigned long long* ex_ptr = nullptr;
    int32_t *ex_start = nullptr, *ex_ng = nullptr, *ex_dst = nullptr, *ex_tok = nullptr;
    void* ex_scr = nullptr;
    float* ex_rows = nullptr;                                                           // [R * k][E]
    float* attn_scr = nullptr;
    float* blog = nullptr;                                                              // a batch's logits [R][V]
    int32_t* io = nullptr;                                                              // tokens [R], positions [R]
    model::BatchShared batch;
    PromptPath prompt;
    const Gemma4Model* owner = nullptr;   // registered the prompt path as the image encoder's lender
    std::vector<void*> bufs;
    ~Shared() {
        if (owner) owner->set_prompt(nullptr);
        for (void* p : bufs) cudaFree(p);
    }
    bool init(const Gemma4Model& m, std::string& err) {
        const Geometry& g = m.geo();
        const int64_t R = kRows, E = g.n_embd, F = m.n_ff_pad();
        int64_t qmax = 0, kmax = 0;
        for (int l = 0; l < g.n_layers; ++l) {
            const LayerShape s = g.shape(l);
            qmax = std::max<int64_t>(qmax, (int64_t) s.n_head * s.hd);
            kmax = std::max<int64_t>(kmax, (int64_t) s.n_kv * s.hd);
        }
        bool ok = true;
        auto a = [&](size_t bytes) -> void* {
            void* p = nullptr;
            if (!ok || cudaMalloc(&p, std::max<size_t>(bytes, 256)) != cudaSuccess) {
                ok = false;
                return nullptr;
            }
            bufs.push_back(p);
            return p;
        };
        auto f = [&](int64_t n) { return (float*) a((size_t) n * 4); };
        x = f(R * E), tmp = f(R * E), mlp = f(R * E), xr = f(R * E), xn = f(R * E);
        q = f(R * qmax), k = f(R * kmax), v = f(R * kmax), qp = f(R * qmax);
        g_alloc(f, R, F);
        act = a(wg::act_bytes(std::max({E, qmax, F})));
        attn_scr = f((int64_t) attn_scratch_floats(kRows, g.n_head, std::max(g.hd_swa, g.hd_full)));
        blog = f(R * g.n_vocab);
        io = (int32_t*) a(2 * R * 4);
        mlp_workspace(g, F, m.mlp_nvfp4(), ws.partial_floats, ws.tickets_n);
        if (ws.tickets_n) {
            ws.partial = f((int64_t) ws.partial_floats);
            ws.tickets = (int32_t*) a(ws.tickets_n * 4);
            if (ok && cudaMemset(ws.tickets, 0, ws.tickets_n * 4) != cudaSuccess) ok = false;
        }
        if (g.moe()) {
            const int64_t P = R * g.n_used;
            rlog = f(R * g.n_expert), rw = f(P);
            ids = (int32_t*) a((size_t) P * 4);
            xq = a((size_t) R * (E / 32) * 36);   // q8_1 blocks of 32 values
            ex_ptr = (unsigned long long*) a((size_t) P * 8);
            ex_start = (int32_t*) a((size_t) (P + 2) * 4);
            ex_ng = (int32_t*) a(4);
            ex_dst = (int32_t*) a((size_t) P * 4);
            ex_tok = (int32_t*) a((size_t) P * 4);
            ex_scr = a(bk::native_expert_scratch_bytes(P, g.n_ff_exp));
            ex_rows = f(P * E);
        }
        if (!ok) {
            cudaGetLastError();
            err = "out of device memory for the forward's scratch";
            return false;
        }
        if (!prompt.init(m, err)) return false;
        owner = &m;
        m.set_prompt(&prompt);
        return true;
    }
    template <class F>
    void g_alloc(F& f, int64_t R, int64_t FF) {
        g = f(R * FF), u = f(R * FF);
    }
};

// A session's drafter state: the chain's buffers (the drafts of several sessions may be in flight at once, each on its
// session's stream), its staged inputs and its graphs by draft count.
struct Drafts {
    float *in = nullptr, *hs = nullptr, *q = nullptr, *qp = nullptr, *att = nullptr, *tmp = nullptr, *xn = nullptr;
    float *gt = nullptr, *up = nullptr, *h = nullptr, *mlp = nullptr, *logits = nullptr, *lp = nullptr, *lp_scr = nullptr;
    float* attn_scr = nullptr;
    void* act = nullptr;
    int32_t* io = nullptr;          // [0] pos, [1] pos - 1, [2..] the anchor then each draft
    int32_t* io_h = nullptr;        // pinned
    float* lp_h = nullptr;          // pinned: each draft's log probability row
    std::map<int, core::CapturedGraph> graphs;
    std::set<int> direct;
    int pending = 0;
};

class Session final : public model::BatchSession {
public:
    explicit Session(Gemma4Model& m) : m_(m), g_(m.geo()) {}
    ~Session() override {
        if (stream_) cudaStreamSynchronize(stream_);
        graphs_.clear();
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
        bool ok = true;
        caches_.assign((size_t) g_.n_layers, SegKv{});
        const int eb = kv_elem_bytes(m_.kv_type());
        for (int l = 0; l < g_.n_layers; ++l) {
            const LayerShape s = g_.shape(l);
            SegKv& c = caches_[(size_t) l];
            c.cap = s.window > 0 ? (int64_t) s.window + kChunk : m_.context();
            const size_t bytes = (size_t) c.cap * s.n_kv * s.hd * eb;
            c.k = alloc(bytes, ok);
            if (!s.one) c.v = alloc(bytes, ok);
        }
        logits_ = (float*) alloc((size_t) kRows * g_.n_vocab * 4, ok);
        if (const DrafterWeights* dw = m_.drafter()) {
            tap_ = (float*) alloc((size_t) kRows * g_.n_embd * 4, ok);
            const int64_t E = g_.n_embd, DE = dw->n_embd, F = dw->n_ff_pad;
            int64_t qmax = 0;
            for (const DrafterLayer& l : dw->layers) qmax = std::max<int64_t>(qmax, (int64_t) l.sh.n_head * l.sh.hd);
            auto f = [&](int64_t n) { return (float*) alloc((size_t) n * 4, ok); };
            ds_.in = f(2 * E), ds_.hs = f(DE), ds_.q = f(qmax), ds_.qp = f(qmax), ds_.att = f(qmax), ds_.tmp = f(DE), ds_.xn = f(DE);
            ds_.gt = f(F), ds_.up = f(F), ds_.h = f(F), ds_.mlp = f(DE), ds_.logits = f(g_.n_vocab);
            ds_.lp = f((int64_t) kRows * bk::kLogprobStride), ds_.lp_scr = f((int64_t) bk::row_logprobs_scratch_floats(1, g_.n_vocab));
            ds_.attn_scr = f((int64_t) attn_scratch_floats(1, dw->layers[0].sh.n_head, std::max(g_.hd_swa, g_.hd_full)));
            ds_.act = alloc(wg::act_bytes(std::max({2 * E, qmax, F})), ok);
            ds_.io = (int32_t*) alloc((2 + kRows + 1) * 4, ok);
            if (ok && cudaMallocHost((void**) &ds_.io_h, (2 + kRows + 1) * 4) != cudaSuccess) ok = false;
            else if (ok) pinned_.push_back(ds_.io_h);
            if (ok && cudaMallocHost((void**) &ds_.lp_h, (size_t) kRows * bk::kLogprobStride * 4) != cudaSuccess) ok = false;
            else if (ok) pinned_.push_back(ds_.lp_h);
        }
        pick_ = (int32_t*) alloc(4 * 32, ok);
        lp_ = (float*) alloc((bk::row_logprobs_scratch_floats(kRows, g_.n_vocab) + (size_t) kRows * bk::kLogprobStride) * 4, ok);
        if (ok && cudaMallocHost((void**) &pick_h_, 4 * 32) != cudaSuccess) ok = false;
        else if (ok) pinned_.push_back(pick_h_);
        if (ok && cudaMallocHost((void**) &io_h_, (size_t) kStage * 2 * kRows * 4) != cudaSuccess) ok = false;
        else if (ok) pinned_.push_back(io_h_);
        for (cudaEvent_t& e : stage_ev_)
            if (ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) ok = false;
        if (!ok) {
            cudaGetLastError();
            err = "out of device memory for the session";
            return false;
        }
        sh_ = std::static_pointer_cast<Shared>(m_.shared().lock());
        if (!sh_) {
            sh_ = std::make_shared<Shared>();
            if (!sh_->init(m_, err)) return false;
            m_.shared() = sh_;
        }
        return true;
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens < 1) return {model::Status::Error, "gemma4: an empty window"};
        if (stale_ && r.pos > 0) return stale();
        if (r.pos + r.n_tokens > m_.context()) return {model::Status::Refused, "gemma4: the window runs past the context"};
        std::string err;
        hist_.resize((size_t) std::min<int64_t>(r.pos, (int64_t) hist_.size()));
        for (int t0 = 0; t0 < r.n_tokens; t0 += kRows) {
            const int n = std::min(kRows, r.n_tokens - t0);
            const bool last = t0 + n == r.n_tokens;
            const bool want = r.row_tokens || (last && (r.next_token || r.logits));
            const int head = !want ? 0 : r.row_tokens ? n : 1;
            if (!forward(r.tokens + t0, n, r.pos + t0, head, err)) return {model::Status::Error, "gemma4: " + err};
            for (int i = 0; i < n; ++i) hist_.push_back(r.tokens[t0 + i]);
            head_rows_ = head;
            win_pos_ = r.pos + t0;
            win_rows_ = n;
            if (!want) {
                if (TierExperts* ex = m_.experts(); ex && ex->off_card() && !last && cudaStreamSynchronize(stream_) != cudaSuccess)
                    return {model::Status::Error, "gemma4: the window failed"};
                continue;
            }
            const int32_t* p = draw(r, t0, head, err, stream_);
            if (!p || cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "gemma4: the pick failed"};
            if (r.row_tokens) std::copy(p, p + n, r.row_tokens + t0);
            if (last && r.next_token) *r.next_token = p[head - 1];
            if (last && r.logits &&
                core::device_copy(r.logits, logits_ + (size_t) (head - 1) * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
                return {model::Status::Error, "gemma4: copying the logits failed"};
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "gemma4: the window failed"};
        return {};
    }

    int batch_rows() const override { return kRows; }

protected:
    // the hooks of the shared driver (model/batch_driver.hpp)
    const char* batch_name() const override { return "gemma4"; }
    int64_t batch_context() const override { return m_.context(); }
    model::BatchShared& batch_shared() override { return sh_->batch; }
    cudaStream_t batch_stream() const override { return stream_; }
    bool batch_reserve(const model::DecodeRequest&, std::string& err) override {
        if (stale_) {
            err = stale().message;
            return false;
        }
        return true;
    }
    bool batch_stage(const model::BatchView& v, model::GraphLru<core::CapturedGraph>::Key& key, std::string& err) override {
        if (TierExperts* ex = m_.experts(); ex && !ex->begin_window(v.stream, err)) return false;
        key.push_back((uint64_t) stage(v.w.tokens.data(), v.w.rows, 0, v.w.positions.data()));
        return true;
    }
    bool batch_forward(const model::BatchView& v, std::string&) override {
        view_ = &v;
        body(v.w.rows, v.w.rows, full_keys_of(v.w.max_pos), sh_->blog, nullptr);
        view_ = nullptr;
        return true;
    }
    void batch_take(const model::BatchView& v, int i) override {
        if (TierExperts* ex = m_.experts(); ex && i == 0) {   // once a window, after its launch
            ex->issue_copies();
            ex->end_window(v.w.rows, v.stream);
        }
        const model::DecodeRequest& r = v.reqs[i];
        const model::BatchWindow::Seg& sg = v.w.segs[(size_t) i];
        const int h = v.head[i];
        const int64_t V = g_.n_vocab;
        hist_.resize((size_t) std::min<int64_t>(r.pos, (int64_t) hist_.size()));
        hist_.insert(hist_.end(), r.tokens, r.tokens + r.n_tokens);
        head_rows_ = h;
        win_pos_ = r.pos;
        win_rows_ = r.n_tokens;
        if (tap_)
            cudaMemcpyAsync(tap_, sh_->xn + (size_t) sg.row0 * g_.n_embd, (size_t) sg.rows * g_.n_embd * 4, cudaMemcpyDeviceToDevice,
                            v.stream);
        if (h == 0) return;
        cudaMemcpyAsync(logits_, sh_->blog + (size_t) (sg.row0 + sg.rows - h) * V, (size_t) h * V * 4, cudaMemcpyDeviceToDevice, v.stream);
        if (r.logits) cudaMemcpyAsync(r.logits, logits_ + (size_t) (h - 1) * V, (size_t) V * 4, cudaMemcpyDeviceToHost, v.stream);
    }
    const int32_t* batch_draw(const model::BatchView& v, int i, std::string& err) override {
        return draw(v.reqs[i], 0, v.head[i], err, v.stream);
    }

public:
    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        if (stale_ && r.pos > 0) return stale();
        if (r.pos + r.n_tokens > m_.context()) return {model::Status::Refused, "gemma4: the prompt runs past the context"};
        hist_.resize((size_t) std::min<int64_t>(r.pos, (int64_t) hist_.size()));
        stale_ = true;   // until the rows are in (an error or an exception leaves it set)
        std::string err;
        if (r.n_tokens > kRows) {   // the prompt path, chunk by chunk
            if (!sh_->prompt.begin(stream_, err)) return {model::Status::Error, "gemma4: " + err};
            struct End {
                PromptPath& p;
                ~End() { p.end(); }
            } end{sh_->prompt};
            try {
                for (int t0 = 0; t0 < r.n_tokens; t0 += kChunk) {
                    if (r.stop && r.stop()) break;
                    const int n = std::min(kChunk, r.n_tokens - t0);
                    sh_->prompt.chunk(r.tokens + t0, n, r.pos + t0, caches_, stream_);
                    if (!check(err)) return {model::Status::Error, "gemma4: " + err};
                    hist_.insert(hist_.end(), r.tokens + t0, r.tokens + t0 + n);
                    if (r.progress) {
                        cudaStreamSynchronize(stream_);
                        r.progress(t0 + n);
                    }
                }
            } catch (const std::exception& e) {
                return {model::Status::Error, std::string("gemma4: the prompt: ") + e.what()};
            }
            win_rows_ = 0;     // no drafter state and no logits for these rows
            head_rows_ = 0;
            if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "gemma4: the prompt failed"};
            stale_ = false;
            return {};
        }
        for (int t0 = 0; t0 < r.n_tokens; t0 += kRows) {
            if (r.stop && r.stop()) break;
            const int n = std::min(kRows, r.n_tokens - t0);
            if (!forward(r.tokens + t0, n, r.pos + t0, 0, err)) return {model::Status::Error, "gemma4: " + err};
            for (int i = 0; i < n; ++i) hist_.push_back(r.tokens[t0 + i]);
            win_pos_ = r.pos + t0;
            win_rows_ = n;
            if (r.progress && ((t0 / kRows) % 64 == 63 || t0 + n == r.n_tokens)) {
                cudaStreamSynchronize(stream_);
                r.progress(t0 + n);
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "gemma4: the prompt failed"};
        stale_ = false;
        return {};
    }

    // ---- [vision] images (vision.hpp, vision_encoder.hpp): a picture's rows stand in for its <|image|> tokens' embeddings,
    // read as one prompt chunk whose window layers attend both ways among them.
    model::Result encode_image(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, model::ImageGrid& grid) override {
        ImageEncoder& enc = m_.encoder();
        if (!enc.staged()) return {model::Status::NotImplemented, "gemma4: started without --vision"};
        std::string err;
        if (!enc.encode(data, bytes, rows, n_rows, err)) return {model::Status::Error, "gemma4: " + err};
        grid = {};
        return {};
    }
    void images_done() override { m_.encoder().release(); }
    model::Result prefill_embeddings(const float* rows, int n_rows, int64_t pos, model::ImageGrid) override {
        if (stale_ && pos > 0) return stale();
        if (n_rows < 1 || n_rows > kChunk) return {model::Status::Error, "gemma4: a picture of more rows than a prompt chunk"};
        if (pos + n_rows > m_.context()) return {model::Status::Refused, "gemma4: the prompt runs past the context"};
        hist_.resize((size_t) std::min<int64_t>(pos, (int64_t) hist_.size()));
        stale_ = true;   // until the rows are in
        std::string berr;
        if (!sh_->prompt.begin(stream_, berr)) return {model::Status::Error, "gemma4: " + berr};
        struct End {
            PromptPath& p;
            ~End() { p.end(); }
        } end{sh_->prompt};
        try {
            sh_->prompt.picture(rows, n_rows, pos, caches_, stream_);
        } catch (const std::exception& e) {
            return {model::Status::Error, std::string("gemma4: the picture: ") + e.what()};
        }
        std::string err;
        if (!check(err)) return {model::Status::Error, "gemma4: " + err};
        hist_.insert(hist_.end(), (size_t) n_rows, g_.image_token);
        win_rows_ = 0;     // no drafter state and no logits for these rows
        head_rows_ = 0;
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "gemma4: the picture failed"};
        stale_ = false;
        return {};
    }

    model::Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (!bias_on_) return {};
        std::string e;
        if (!device_bias(d_bias_, bias, e)) return {model::Status::Error, "gemma4: " + e};
        return {};
    }

    model::Result resample_row(int row, const float* bias, int32_t& token) override {
        std::string e;
        if (row < 0 || row >= head_rows_) return {model::Status::Error, "gemma4: the row is not in the last window"};
        if (!device_bias(d_mask_, bias, e)) return {model::Status::Error, "gemma4: " + e};
        bk::SamplerParams sp = sampling_;
        sp.logit_bias = d_mask_;
        sp.counter += (uint64_t) row;
        bk::sample_tokens(logits_ + (size_t) row * g_.n_vocab, 1, g_.n_vocab,
                          history_len_ > 0 ? d_hist_ + (size_t) row * history_len_ : nullptr, history_len_, sp, pick_ + 24, stream_);
        int32_t t = 0;
        if (cudaMemcpyAsync(&t, pick_ + 24, 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess || cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "gemma4: the masked draw failed"};
        token = t;
        return {};
    }

    model::Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (row0 < 0 || n < 1 || row0 + n > head_rows_) return {model::Status::Error, "gemma4: the session keeps the last window's logits only"};
        float* d_out = lp_ + bk::row_logprobs_scratch_floats(n, g_.n_vocab);
        int32_t* d_pick = pick_ + 16;
        cudaMemcpyAsync(d_pick, picks, (size_t) n * 4, cudaMemcpyHostToDevice, stream_);
        bk::row_logprobs(logits_ + (size_t) row0 * g_.n_vocab, n, g_.n_vocab, d_pick, top, d_out, lp_, stream_);
        if (cudaMemcpyAsync(out, d_out, (size_t) n * bk::kLogprobStride * 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "gemma4: the log probabilities failed"};
        return {};
    }

    // the KV format and the experts' placement (a community row reads both); a dense model has no expert to spill
    std::string info_words() const override {
        const std::string kv = m_.kv_type() == KvType::F8 ? "kv=fp8" : "kv=fp16";
        core::ExpertTier* t = m_.tier();
        return t ? "expert_slots=" + std::to_string(t->gpu_slots()) + " expert_slots_primary=" + std::to_string(t->slots()) +
                       " ram_experts_mib=" + std::to_string(t->ram_bytes() >> 20) + " " + kv + " " + program::spill_words(t->spill())
                 : kv + " " + program::spill_words({});
    }

    // SLEEP / WAKE: every weight stays on the card, so the session has nothing to lend: ok, no slots, SLEEP goes on to the
    // driver.
    model::Result set_low_memory(bool, int64_t&) override {
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "gemma4: the session stream failed"};
        return {};
    }

    model::Result copy_logits(int row, float* out) override {
        if (row < 0 || row >= head_rows_) return {model::Status::Error, "gemma4: no logits for that row"};
        if (core::device_copy(out, logits_ + (size_t) row * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
            return {model::Status::Error, "gemma4: copying the logits failed"};
        return {};
    }

    model::Result set_history(const int32_t* tokens, int64_t n) override {
        hist_.assign(tokens, tokens + n);
        return {};
    }

    // ---- the assistant drafter: its drafts after an anchor read the model's state at the row before it (the tap of the
    // last window) and the caches of the model's last layer of each kind, all at the anchor's position.
    int max_window() const override { return m_.drafter() ? std::clamp(m_.flags().spec, 1, kRows) : 1; }
    int max_verify_window() const override { return kRows; }
    bool copy_drafts() const override { return max_window() > 1; }   // prompt lookup: any window's rows verify alike
    model::Result draft_begin(int32_t anchor, int64_t pos, int n) override {
        ds_.pending = 0;
        if (!m_.drafter()) return {model::Status::NotImplemented, "gemma4: the pack has no drafter"};
        n = std::min(n, kRows - 1);
        if (n < 1 || pos < 1 || pos - 1 < win_pos_ || pos - 1 >= win_pos_ + win_rows_) return {};   // no state for the row before
        ds_.io_h[0] = (int32_t) pos;
        ds_.io_h[1] = (int32_t) pos - 1;
        ds_.io_h[2] = anchor;
        cudaMemcpyAsync(ds_.io, ds_.io_h, 3 * 4, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(ds_.in + g_.n_embd, tap_ + (size_t) (pos - 1 - win_pos_) * g_.n_embd, (size_t) g_.n_embd * 4,
                        cudaMemcpyDeviceToDevice, stream_);
        std::string err;
        try {
            auto it = ds_.graphs.find(n);
            if (it != ds_.graphs.end()) {
                if (!it->second.launch(stream_, err)) return {model::Status::Error, "gemma4: the drafter: " + err};
            } else if (ds_.direct.insert(n).second) {
                draft_chain(n);
            } else {
                core::CapturedGraph gr;
                if (!gr.begin(stream_, err)) return {model::Status::Error, "gemma4: the drafter: " + err};
                draft_chain(n);
                if (!gr.end(stream_, err) || !gr.launch(stream_, err)) return {model::Status::Error, "gemma4: the drafter: " + err};
                ds_.graphs.emplace(n, std::move(gr));
            }
        } catch (const std::exception& e) {
            return {model::Status::Error, std::string("gemma4: the drafter: ") + e.what()};
        }
        cudaMemcpyAsync(ds_.io_h + 3, ds_.io + 3, (size_t) n * 4, cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(ds_.lp_h, ds_.lp, (size_t) n * bk::kLogprobStride * 4, cudaMemcpyDeviceToHost, stream_);
        ds_.pending = n;
        return {};
    }
    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n, const DraftNeed&) override {
        n = 0;
        if (!m_.drafter()) return {model::Status::NotImplemented, "gemma4: the pack has no drafter"};
        if (ds_.pending == 0) return {};
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "gemma4: the drafter failed"};
        n = std::min(ds_.pending, max_n);
        for (int i = 0; i < n; ++i) {
            ids[i] = ds_.io_h[3 + i];
            conf[i] = std::exp(ds_.lp_h[(size_t) i * bk::kLogprobStride]);
        }
        ds_.pending = 0;
        return {};
    }

    // ---- conversations: the full layers' cells by position (save_cells / restore_cells), the window rings by checkpoint
    // (running_state). A checkpoint at `pos` holds a ring's cells of the window's positions before `pos`, the only ones a
    // row at `pos` or later reads; the ring holds them while the session has not run more than kChunk rows past `pos`.
    size_t ring_bytes(int l) const { return (size_t) caches_[(size_t) l].cap * cell_bytes(l); }
    size_t running_bytes() const override {
        size_t n = 0;
        for (int l = 0; l < g_.n_layers; ++l)
            if (g_.swa[(size_t) l]) n += 2 * (size_t) g_.shape(l).window * cell_bytes(l);
        return n;
    }
    model::Result running_state(int64_t pos, bool save, core::RunningState& st) override {
        if (save && stale_) return stale();
        if (save && (int64_t) hist_.size() - pos > kChunk)
            return {model::Status::Error, "gemma4: the rings have moved on more than " + std::to_string(kChunk) + " positions past the checkpoint"};
        if (save && pos > (int64_t) hist_.size()) return {model::Status::Error, "gemma4: a checkpoint past the session's position"};
        for (int l = 0; l < g_.n_layers; ++l)
            if (g_.swa[(size_t) l]) {
                // cells pos - window .. pos - 1 in ring order: a run to the ring's end, then its start (two pieces at every
                // position; the second is empty when the run does not wrap). Before the window fills, the cells of the
                // negative positions are ones a later row writes before it reads them.
                const SegKv& c = caches_[(size_t) l];
                const int64_t w = g_.shape(l).window, first = ((pos - w) % c.cap + c.cap) % c.cap;
                const int64_t run = std::min(w, c.cap - first);
                const size_t cb = cell_bytes(l);
                for (void* base : {c.k, c.v}) {
                    st.regions.push_back({(uint8_t*) base + (size_t) first * cb, (size_t) run * cb});
                    st.regions.push_back({base, (size_t) (w - run) * cb});
                }
            }
        st.stream = stream_;
        return {};
    }
    model::Result running_restored(int64_t pos, const uint8_t*) override {
        if ((int64_t) hist_.size() > pos) hist_.resize((size_t) pos);
        stale_ = false;
        return {};
    }

    int cell_format() const { return m_.kv_type() == KvType::F8 ? 1 : 3; }   // 3: fp16, 1: E4M3
    size_t cell_bytes(int l) const {
        const LayerShape s = g_.shape(l);
        return (size_t) s.n_kv * s.hd * kv_elem_bytes(m_.kv_type());
    }
    size_t cells_bytes(int64_t positions) const override {
        size_t n = 0;
        for (int l = 0; l < g_.n_layers; ++l)
            if (!g_.swa[(size_t) l]) n += (g_.shape(l).one ? 1 : 2) * (size_t) positions * cell_bytes(l);
        return n;
    }
    // A full layer's cache is cell-major ([cell][head][dim]): its first `positions` cells are one run (V alone when its keys
    // are its values).
    model::Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (stale_) return stale();
        if (positions < 0 || positions > m_.context()) return {model::Status::Error, "gemma4: positions past the context"};
        cudaStreamSynchronize(stream_);
        kv.clear();
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) continue;
            const LayerShape s = g_.shape(l);
            const size_t run = (size_t) positions * cell_bytes(l);
            core::ConversationKv k;
            k.format = cell_format();
            k.page_size = 1;
            k.cells = positions;
            k.heads = s.n_kv;
            k.head_dim = s.hd;
            const auto down = [](const void* dev, core::ConversationBuffer& to) {
                return to.visit(0, to.size(), [&](uint8_t* p, size_t n, size_t at) {
                    return core::device_copy(p, (const uint8_t*) dev + at, n, cudaMemcpyDeviceToHost) == cudaSuccess;
                });
            };
            k.k.resize_for_overwrite(run);
            if (!s.one) k.v.resize_for_overwrite(run);
            if (!down(caches_[(size_t) l].k, k.k) || (!s.one && !down(caches_[(size_t) l].v, k.v)))
                return {model::Status::Error, "gemma4: reading a cache back failed"};
            kv.push_back(std::move(k));
        }
        return {};
    }
    model::Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (positions < 0 || positions > m_.context()) return {model::Status::Error, "gemma4: positions past the context"};
        cudaStreamSynchronize(stream_);
        size_t i = 0;
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) continue;
            const LayerShape s = g_.shape(l);
            const size_t run = (size_t) positions * cell_bytes(l);
            if (i >= kv.size() || kv[i].format != cell_format() || kv[i].cells != positions || kv[i].heads != s.n_kv ||
                kv[i].head_dim != s.hd || kv[i].k.size() < run || (!s.one && kv[i].v.size() < run))
                return {model::Status::Error, "gemma4: a parked conversation has other caches"};
            const auto up = [&](void* dev, const core::ConversationBuffer& from) {
                return from.visit(0, run, [&](const uint8_t* p, size_t n, size_t at) {
                    return core::device_copy((uint8_t*) dev + at, p, n, cudaMemcpyHostToDevice) == cudaSuccess;
                });
            };
            if (!up(caches_[(size_t) l].k, kv[i].k) || (!s.one && !up(caches_[(size_t) l].v, kv[i].v)))
                return {model::Status::Error, "gemma4: writing a cache back failed"};
            ++i;
        }
        return {};
    }

    int64_t kv_bytes() const override {
        int64_t n = 0;
        for (int l = 0; l < g_.n_layers; ++l) n += (int64_t) (g_.swa[(size_t) l] ? 2 * ring_bytes(l) : cells_bytes_layer(l));
        return n;
    }
    size_t cells_bytes_layer(int l) const { return (g_.shape(l).one ? 1 : 2) * (size_t) m_.context() * cell_bytes(l); }

    // the tier's lookups when experts are off the card; else every lookup is a hit
    ExpertCounts expert_counts() const override {
        const core::ExpertTier* t = m_.tier();
        if (t && m_.experts()->off_card()) return {t->hits(), t->lookups(), t->offloaded(), t->file_lookups()};
        return {lookups_, lookups_, 0, 0};
    }
    std::vector<CardExperts> experts_by_card() const override {
        const core::ExpertTier* t = m_.tier();
        if (!t) return {};
        std::vector<CardExperts> out;
        for (const auto& c : t->slots_by_card()) out.push_back({c.device, c.slots, c.bytes});
        return out;
    }

private:
    static model::Result stale() {
        return {model::Status::Error, "gemma4: a failed read left the conversation's state unknown: start it over or restore a checkpoint"};
    }

    void* alloc(size_t bytes, bool& ok) {
        void* p = nullptr;
        if (!ok || cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
            ok = false;
            return nullptr;
        }
        bufs_.push_back(p);
        return p;
    }

    // The full layers' launch bound: a power of two at least the keys of the call's last position.
    static int64_t full_keys_of(int64_t last) {
        int64_t k = 1024;
        while (k < last + 1) k <<= 1;
        return k;
    }

    // A call's tokens and positions to the card (pos0.., or every row's own in `positions`); returns the full layers'
    // launch bound.
    int64_t stage(const int32_t* tokens, int rows, int64_t pos0, const int32_t* positions) {
        const int slot = (int) (stage_n_++ % kStage);
        cudaEventSynchronize(stage_ev_[slot]);
        int32_t* io = io_h_ + (size_t) slot * 2 * kRows;
        int64_t last = 0;
        for (int i = 0; i < rows; ++i) {
            io[i] = tokens[i];
            io[kRows + i] = positions ? positions[i] : (int32_t) (pos0 + i);
            last = std::max<int64_t>(last, io[kRows + i]);
        }
        cudaMemcpyAsync(sh_->io, io, (size_t) 2 * kRows * 4, cudaMemcpyHostToDevice, stream_);
        cudaEventRecord(stage_ev_[slot], stream_);
        return full_keys_of(last);
    }

    bool forward(const int32_t* tokens, int rows, int64_t pos0, int head, std::string& err) {
        TierExperts* ex = m_.experts();
        if (ex && !ex->begin_window(stream_, err)) return false;
        const bool ok = launch_window(tokens, rows, pos0, head, err);
        if (ex) {
            ex->issue_copies();
            ex->end_window(rows, stream_);
        }
        return ok && (!ex || !ex->failed(err));
    }

    bool launch_window(const int32_t* tokens, int rows, int64_t pos0, int head, std::string& err) {
        const int64_t keys = stage(tokens, rows, pos0, nullptr);
        lookups_ += (int64_t) rows * g_.n_used * g_.n_layers;
        const uint64_t key = ((uint64_t) keys << 16) | ((uint64_t) head << 8) | (uint64_t) rows;
        auto it = graphs_.find(key);
        if (it != graphs_.end()) return it->second.launch(stream_, err) && check(err);
        if (direct_.insert(key).second) {   // a shape's first window runs direct
            body(rows, head, keys, logits_, tap_);
            return check(err);
        }
        core::CapturedGraph gr;
        if (!gr.begin(stream_, err)) return false;
        body(rows, head, keys, logits_, tap_);
        if (!gr.end(stream_, err)) return false;
        if (!gr.launch(stream_, err)) return false;
        graphs_.emplace(key, std::move(gr));
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

    // The caches of layer l for the rows of the call: the batch window's segments, or this session's own.
    KvSegs kv_segs(int l) const {
        KvSegs s;
        if (!view_) {
            s.c[0] = caches_[(size_t) l];
            return s;
        }
        s.at = view_->w.device_segs();
        for (int i = 0; i < view_->n; ++i) s.c[i] = static_cast<const Session*>(view_->ss[i])->caches_[(size_t) l];
        return s;
    }

    void product(const Mat& m, float* y, int64_t ld_y, int rows) {
        wg::Call c;
        c.fmt = m.fmt;
        c.K = m.K;
        c.m[0] = {m.w, m.rows, m.scale, y, ld_y};
        c.n = 1;
        wg::gemm(c, sh_->act, rows, sh_->ws, stream_);
    }

    // The layers and the head over `rows` rows (tokens and positions staged): every row's final normed state into x.xn
    // (and `tap`, when set: the drafter's input), the head over the last `head` rows into `logits` ([head][V], softcapped).
    void body(int rows, int head, int64_t full_keys, float* logits, float* tap) {
        Shared& x = *sh_;
        const int E = g_.n_embd;
        const int64_t F = m_.n_ff_pad();
        const int32_t* tok = x.io;
        const int32_t* pos = x.io + kRows;
        bk::PdlScope pdl(true, true);
        embed_rows(m_.embedding(), tok, rows, E, g_.embed_scale, x.x, stream_);
        wg::norm_pack(x.x, m_.layer(0).attn_norm, g_.rms_eps, E, rows, x.act, nullptr, stream_);
        for (int l = 0; l < g_.n_layers; ++l) {   // a layer starts with its attention's input in x.act
            const LayerWeights& w = m_.layer(l);
            const LayerShape& s = w.sh;
            const int64_t qd = (int64_t) s.n_head * s.hd, kd = (int64_t) s.n_kv * s.hd;
            {
                wg::Call c;
                c.fmt = wg::Format::F8;
                c.K = E;
                c.m[0] = {w.q.w, w.q.rows, w.q.scale, x.q, qd};
                c.m[1] = {w.k.w, w.k.rows, w.k.scale, x.k, kd};
                c.n = 2;
                if (!s.one) {
                    c.m[2] = {w.v.w, w.v.rows, w.v.scale, x.v, kd};
                    c.n = 3;
                }
                wg::gemm(c, x.act, rows, sh_->ws, stream_);
            }
            Prep p;
            p.q = x.q, p.k = x.k, p.v = x.v, p.ldq = qd, p.ldk = kd, p.ldv = kd;
            p.pos = pos;
            p.q_norm = w.q_norm, p.k_norm = w.k_norm, p.rope = w.rope, p.eps = g_.rms_eps;
            p.sh = s;
            p.kv_type = m_.kv_type();
            p.q_out = x.qp;
            p.kv = kv_segs(l);
            attn_prep(p, rows, stream_);
            Attn a;
            a.q = x.qp, a.pos = pos, a.k_norm = w.k_norm, a.rope = w.rope, a.sh = s, a.kv_type = m_.kv_type();
            a.kv = p.kv;
            a.max_keys = s.window > 0 ? s.window : full_keys;
            a.scratch = x.attn_scr;
            a.act = x.act;
            attn_decode(a, rows, stream_);
            product(w.o, x.tmp, E, rows);
            // the MLP
            add_norm(x.x, x.tmp, w.post_attn, g_.rms_eps, rows, E, stream_);
            wg::norm_pack(x.x, w.ffn_norm, g_.rms_eps, E, rows, x.act, nullptr, stream_);
            {
                wg::Call c;
                c.fmt = w.gate.fmt;
                c.K = E;
                c.m[0] = {w.gate.w, w.gate.rows, w.gate.scale, x.g, F};
                c.m[1] = {w.up.w, w.up.rows, w.up.scale, x.u, F};
                c.n = 2;
                wg::gemm(c, x.act, rows, sh_->ws, stream_);
            }
            gelu_pack(x.g, x.u, rows, g_.n_ff, F, F, x.act, stream_);
            product(w.down, x.mlp, E, rows);
            FfnEnd fe;
            fe.x = x.x, fe.mlp = x.mlp, fe.n_post = w.post_ffw, fe.scalar = w.scalar, fe.eps = g_.rms_eps;
            if (g_.moe()) {   // the router reads the residual unweighted-normed; the experts its pre_ffw2 norm
                moe_inputs(x.x, w.pre_ffw2, g_.rms_eps, rows, E, x.xr, x.xq, stream_);
                bk::RouterArgs ra;
                ra.n_expert = g_.n_expert, ra.n_embd = E, ra.k = g_.n_used, ra.weight_scale = 1.f, ra.gate = bk::RouterGate::Softmax;
                bk::router(x.xr, w.router, nullptr, rows, ra, x.rlog, x.ids, x.rw, stream_);
                TierExperts& ex = *m_.experts();
                const bool off = ex.off_card();
                if (off) ex.ring(l, rows, x.xr, w.pre_ffw2, x.ids, stream_);
                bk::moe_plan(x.ids, rows, g_.n_used, g_.n_expert, ex.tab(l), x.ex_ptr, x.ex_start, x.ex_ng, x.ex_dst, x.ex_tok, stream_);
                bk::NativeExpertLayout L = bk::native_expert_layout(w.gu_type, w.d_type, E, g_.n_ff_exp);
                L.gelu = true;
                const int pairs = rows * g_.n_used;
                bk::native_expert_grouped(L, x.ex_ptr, x.ex_start, x.ex_ng, x.ex_dst, x.ex_tok, std::min(pairs, g_.n_expert), pairs,
                                          x.xq, x.ex_scr, x.ex_rows, stream_);
                if (off) ex.merge(l, rows, x.ids, x.ex_rows, stream_);
                fe.experts = x.ex_rows, fe.weights = x.rw, fe.k = g_.n_used, fe.n1 = w.post_ffw1, fe.n2 = w.post_ffw2;
            }
            if (l + 1 < g_.n_layers) {   // the next layer's attention input, or the final norm
                fe.next_w = m_.layer(l + 1).attn_norm, fe.act = x.act;
            } else {
                fe.next_w = m_.output_norm(), fe.xn = x.xn;
            }
            ffn_end(fe, rows, E, stream_);
        }
        if (tap) cudaMemcpyAsync(tap, x.xn, (size_t) rows * E * 4, cudaMemcpyDeviceToDevice, stream_);
        if (head <= 0) return;
        wg::pack(x.xn + (size_t) (rows - head) * E, E, E, head, x.act, stream_);
        {
            const Mat& hm = m_.head();
            wg::Call c;
            c.fmt = wg::Format::F8;
            c.K = hm.K;
            c.m[0] = {hm.w, hm.rows, nullptr, logits, (int64_t) g_.n_vocab};
            c.m[0].row_scales = hm.scale;
            c.n = 1;
            wg::gemm(c, x.act, head, sh_->ws, stream_);
        }
        softcap(logits, (int64_t) head * g_.n_vocab, g_.softcap, stream_);
    }

    void dproduct(const Mat& m, const float* x, int64_t ld_x, float* y, bool row_scales = false) {
        wg::pack(x, ld_x, m.K, 1, ds_.act, stream_);
        wg::Call c;
        c.fmt = wg::Format::F8;
        c.K = m.K;
        c.m[0] = {m.w, m.rows, row_scales ? nullptr : m.scale, y, m.rows};
        if (row_scales) c.m[0].row_scales = m.scale;
        c.n = 1;
        wg::gemm(c, ds_.act, 1, sh_->ws, stream_);
    }

    // `n` drafts in a chain (launches only): each step embeds its token beside the state, runs the drafter's layers at the
    // anchor's position over the model's caches, draws its pick greedily with its log probability, and hands its state on.
    void draft_chain(int n) {
        const DrafterWeights& dw = *m_.drafter();
        const int E = g_.n_embd, DE = dw.n_embd;
        const int32_t* pos = ds_.io;          // the anchor's position: the rotation
        const int32_t* seen = ds_.io + 1;     // the position before it: the keys' range
        bk::PdlScope pdl(true, true);
        bk::SamplerParams greedy;
        greedy.greedy = true;
        greedy.temperature = 0.f;
        for (int j = 0; j < n; ++j) {
            embed_rows(m_.embedding(), ds_.io + 2 + j, 1, E, g_.embed_scale, ds_.in, stream_);
            dproduct(dw.pre, ds_.in, 2 * E, ds_.hs);
            for (const DrafterLayer& w : dw.layers) {
                const int64_t qd = (int64_t) w.sh.n_head * w.sh.hd;
                wg::norm_pack(ds_.hs, w.attn_norm, g_.rms_eps, DE, 1, ds_.act, nullptr, stream_);
                wg::Call c;
                c.fmt = wg::Format::F8;
                c.K = DE;
                c.m[0] = {w.q.w, w.q.rows, w.q.scale, ds_.q, qd};
                c.n = 1;
                wg::gemm(c, ds_.act, 1, sh_->ws, stream_);
                Prep p;
                p.q = ds_.q, p.ldq = qd, p.pos = pos, p.q_norm = w.q_norm, p.rope = w.rope, p.eps = g_.rms_eps;
                p.sh = w.sh, p.kv_type = m_.kv_type(), p.q_out = ds_.qp, p.q_only = true;
                p.kv.c[0] = caches_[(size_t) w.source];
                attn_prep(p, 1, stream_);
                Attn a;
                a.q = ds_.qp, a.pos = seen, a.k_norm = m_.layer(w.source).k_norm, a.rope = w.rope, a.sh = w.sh;
                a.kv_type = m_.kv_type(), a.kv = p.kv, a.max_keys = w.sh.window > 0 ? w.sh.window : m_.context();
                a.scratch = ds_.attn_scr, a.out = ds_.att;
                attn_decode(a, 1, stream_);
                dproduct(w.o, ds_.att, qd, ds_.tmp);
                add_norm(ds_.hs, ds_.tmp, w.post_attn, g_.rms_eps, 1, DE, stream_);
                wg::norm_pack(ds_.hs, w.ffn_norm, g_.rms_eps, DE, 1, ds_.act, nullptr, stream_);
                wg::Call gu;
                gu.fmt = wg::Format::F8;
                gu.K = DE;
                gu.m[0] = {w.gate.w, w.gate.rows, w.gate.scale, ds_.gt, dw.n_ff_pad};
                gu.m[1] = {w.up.w, w.up.rows, w.up.scale, ds_.up, dw.n_ff_pad};
                gu.n = 2;
                wg::gemm(gu, ds_.act, 1, sh_->ws, stream_);
                gelu_mul(ds_.gt, ds_.up, 1, (int) dw.n_ff_pad, dw.n_ff_pad, dw.n_ff_pad, ds_.h, stream_);
                dproduct(w.down, ds_.h, dw.n_ff_pad, ds_.mlp);
                FfnEnd fe;
                fe.x = ds_.hs, fe.mlp = ds_.mlp, fe.n_post = w.post_ffw, fe.scalar = w.scalar, fe.eps = g_.rms_eps;
                ffn_end(fe, 1, DE, stream_);
            }
            bk::rms_norm_rows(ds_.hs, dw.out_norm, 1, DE, g_.rms_eps, ds_.xn, stream_);
            dproduct(dw.head, ds_.xn, DE, ds_.logits, true);
            dproduct(dw.post, ds_.xn, DE, ds_.in + E);
            bk::sample_tokens(ds_.logits, 1, g_.n_vocab, nullptr, 0, greedy, ds_.io + 3 + j, stream_);
            bk::row_logprobs(ds_.logits, 1, g_.n_vocab, ds_.io + 3 + j, 0, ds_.lp + (size_t) j * bk::kLogprobStride, ds_.lp_scr, stream_);
        }
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

    // The draws of the `rows` rows of logits_, queued on `st`; the host array they land in, read once `st` has run.
    const int32_t* draw(const model::DecodeRequest& r, int t0, int rows, std::string& err, cudaStream_t st) {
        bk::SamplerParams sp;
        if (r.sampling) sp = *r.sampling;
        else sp.greedy = true;
        if (!r.sampling) sp.temperature = 0.f;
        sp.counter += (uint64_t) t0;
        sp.logit_bias = bias_on_ ? d_bias_ : nullptr;
        history_len_ = r.history != nullptr ? r.history_len : 0;
        if (history_len_ > 0) {
            const size_t n = (size_t) kRows * history_len_;
            if (d_hist_len_ < n) {
                bool ok = true;
                d_hist_ = (int32_t*) alloc(n * 4, ok);
                if (!ok) {
                    cudaGetLastError();
                    err = "out of device memory for the sampling history";
                    return nullptr;
                }
                d_hist_len_ = n;
            }
            cudaMemcpyAsync(d_hist_, r.history + (size_t) t0 * history_len_, (size_t) rows * history_len_ * 4, cudaMemcpyHostToDevice, st);
        }
        sampling_ = sp;
        bk::sample_tokens(logits_, rows, g_.n_vocab, history_len_ > 0 ? d_hist_ : nullptr, history_len_, sp, pick_, st);
        if (cudaMemcpyAsync(pick_h_, pick_, (size_t) rows * 4, cudaMemcpyDeviceToHost, st) != cudaSuccess) {
            err = "the pick failed";
            return nullptr;
        }
        return pick_h_;
    }

    Gemma4Model& m_;
    const Geometry& g_;
    cudaStream_t stream_ = nullptr;
    std::vector<void*> bufs_;
    std::vector<void*> pinned_;
    std::vector<SegKv> caches_;              // [layer]
    std::shared_ptr<Shared> sh_;
    std::map<uint64_t, core::CapturedGraph> graphs_;   // by window shape: full layers' bound, head rows, rows
    std::set<uint64_t> direct_;                        // the shapes whose first window has run
    const model::BatchView* view_ = nullptr;
    float* logits_ = nullptr;
    float* lp_ = nullptr;
    float* tap_ = nullptr;                   // the last window's rows' final normed states (with a drafter)
    int64_t win_pos_ = 0;                    // the last window's first position and rows
    int win_rows_ = 0;
    Drafts ds_;
    int32_t *pick_ = nullptr, *pick_h_ = nullptr, *io_h_ = nullptr;
    cudaEvent_t stage_ev_[kStage] = {};
    uint64_t stage_n_ = 0;
    std::vector<int32_t> hist_;
    bool stale_ = false;
    float *d_bias_ = nullptr, *d_mask_ = nullptr;
    bool bias_on_ = false;
    int32_t* d_hist_ = nullptr;
    size_t d_hist_len_ = 0;
    int history_len_ = 0;
    bk::SamplerParams sampling_;
    int head_rows_ = 0;
    int64_t lookups_ = 0;
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(Gemma4Model& model, std::string& err) {
    auto s = std::make_unique<Session>(model);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::gemma4
