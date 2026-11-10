// src/models/deepseek41/prefill.cpp - the DeepSeek-V4.1 layer loop over a chunk of prompt tokens.
//
// Same per-layer sequence as the single-token loop (session.cpp), with every elementwise, mixing, norm, router and
// Engram step taken over T tokens at once. Attention and the experts go through the LayerOps batch entries, which
// keep the causal order inside the chunk.
#include "basalt/log.hpp"
#include "models/deepseek41/prefill.hpp"

#include "basalt/artifact/dequant.hpp"
#include "models/deepseek41/engram_rows.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/deepseek41/attn_index.hpp"
#include "basalt/models/deepseek41/kernels.hpp"
#include "models/deepseek41/layer_ops.hpp"
#include "models/deepseek41/model.hpp"
#include "models/deepseek41/ops.hpp"
#include "models/deepseek41/prefill_ffn.hpp"
#include "models/deepseek41/tier.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <thread>
#include <type_traits>
#include <utility>

namespace basalt::models::deepseek41 {
namespace {

#define DS_CUDA(call)                                                                                   \
    do {                                                                                                \
        cudaError_t e_ = (call);                                                                        \
        if (e_ != cudaSuccess) return fail(std::string(#call) + ": " + cudaGetErrorString(e_));         \
    } while (0)

class PrefillImpl final : public Prefill {
public:
    PrefillImpl(DsModel& m, LayerOps& ops, void* stream, int cap)
        : m_(m), g_(m.geo()), ops_(ops), stream_((cudaStream_t) stream), cap_(cap) {}
    ~PrefillImpl() override {
        finish_reads(ahead_);
        for (void* p : allocs_) cudaFree(p);
        if (batch_open_) ops_.close_batch();
    }

    bool init(std::string& err) {
        const size_t n = (size_t) g_.n_embd, C = (size_t) cap_;
        const size_t cols = (size_t) m_.pack().engram.hash.n_cols() * 256;
        auto buf = [&](auto*& p, size_t count) {
            void* v = nullptr;
            if (cudaMalloc(&v, count * sizeof(*p)) != cudaSuccess) return false;
            allocs_.push_back(v);
            p = (std::remove_reference_t<decltype(p)>) v;
            return true;
        };
        dump_dir_ = m_.diag_dump();
        ffn_ = make_prefill_ffn(m_, ops_, cap_, err);
        if (!ffn_) return false;
        ExpertTier* tier = ops_.expert_tier(err);   // the FFN has it open by now
        if (!tier) return false;
        host_ = &tier->host();
        batch_open_ = ops_.open_batch(cap_, err);
        if (!batch_open_) return false;
        const size_t q8 = std::max<size_t>(kernels::native_q8_1_bytes((int) std::max<size_t>(cols, n), 8), 1 << 16);
        bool ok = buf(x_, C * 4 * n) && buf(x2_, C * 4 * n) && buf(cur_, C * n) && buf(normed_, C * n) &&
                  buf(attn_, C * n) && buf(moe_, C * n) && buf(sh_, C * n) && buf(sum_, C * n) &&
                  buf(coef_a_, C * kHcCoef) && buf(coef_f_[0], C * kHcCoef) && buf(coef_f_[1], C * kHcCoef) &&
                  buf(hc_ws_, hc_mixes_workspace_floats(cap_)) && buf(rlogits_, C * (size_t) g_.n_expert) &&
                  buf(rw_, C * 8) && buf(ids_, C * 8) &&
                  buf(kv_dev_, C * 5 * n) && buf(q8_, q8) && buf(rows_all_, C * (cols ? cols : 1));
        if (!ok) err = "out of device memory for the prefill buffers";
        return ok;
    }

    bool run(const int32_t* tokens, int n, int64_t pos0, int64_t rows_after, std::vector<int32_t>& hist, std::string& err) override {
        // the layers past the first ratio-1 layer read earlier positions through its compressed rows only, and each looks
        // back one window more than the one before: rows further from the prompt's last token than that need nothing else
        const int yl = skip_layer();
        cut_ = yl > 0 && rows_after >= 0 ? prefillp::far_below(pos0 + n + rows_after, g_.n_layers, yl, kWindow) : 0;
        left_rows_ = cut_ > pos0;
        return run_rows(tokens, nullptr, n, pos0, hist, err);
    }
    bool run_embeddings(const float* rows, int n, int64_t pos0, std::vector<int32_t>& hist, std::string& err) override {
        cut_ = 0;
        left_rows_ = false;
        return run_rows(nullptr, rows, n, pos0, hist, err);
    }

    bool left_rows() const override { return left_rows_; }

    // The first ratio-1 layer when it is the last to own compressed rows and no Engram sits past it, else -1.
    int skip_layer() const {
        if (!m_.tiers().prompt_skip) return -1;
        int yl = -1;
        for (int il = 0; il < g_.n_layers && yl < 0; ++il)
            if (g_.compress_ratio[(size_t) il] == 1) yl = il;
        if (yl <= 0) return -1;
        for (int il = yl; il < g_.n_layers; ++il)
            if (g_.compress_ratio[(size_t) il] != 1) return -1;
        for (int l : m_.pack().engram.layers)
            if (l >= yl) return -1;
        return yl;
    }

    bool run_rows(const int32_t* tokens, const float* embd, int n, int64_t pos0, std::vector<int32_t>& hist,
                  std::string& err) {
        if (pos0 == 0) {
            hist.clear();
            ops_.reset();
        }
        if ((int64_t) hist.size() != pos0) {
            err = "positions must be consecutive from 0 (the session keeps one sequence)";
            return false;
        }
        for (int off = 0; off < n; off += cap_) {
            const int T = std::min(cap_, n - off);
            err_.clear();
            const int next_T = std::min(cap_, n - off - T);
            if (!chunk(tokens ? tokens + off : nullptr, embd ? embd + (size_t) off * g_.n_embd : nullptr, T, pos0 + off,
                       off, hist, tokens ? tokens + off + T : nullptr, next_T)) {
                cudaStreamSynchronize(stream_);   // the layer ticks queued so far still point at this object
                err = err_.empty() ? "deepseek41: prefill failed" : err_;
                return false;
            }
            if (progress) progress(off + T);
            if (stop && stop()) {
                finish_reads(ahead_);
                ahead_ = EngramReads{};
                break;
            }
        }
        return true;
    }

private:
    // The Engram rows of a chunk, requested ahead of it into one of the host's two read sets (`set`). A ticket left open
    // holds the reader's pointer into it.
    struct EngramReads {
        int64_t pos0 = -1;
        int T = 0;
        int set = -1;
        std::vector<ngram::PleReader::Ticket> tickets;
        std::vector<uint8_t> open;
    };
    uint8_t* raw(const EngramReads& r, size_t layer) const {
        return host_->engram_raw[r.set].data() + layer * host_->engram_layer;
    }

    // Reads into the set `other` does not hold.
    void issue_reads(EngramReads& r, const EngramReads& other, const int32_t* tokens, int T, int64_t pos0,
                     const std::vector<int32_t>& hist) {
        const EngramInfo& en = m_.pack().engram;
        const int n_cols = en.hash.n_cols();
        finish_reads(r);
        r.pos0 = pos0;
        r.T = T;
        r.set = other.set == 0 ? 1 : 0;
        r.tickets.assign(en.layers.size(), {});
        r.open.assign(en.layers.size(), 1);
        int32_t prev[8] = {0};
        const int np = en.hash.n_gram - 1;
        for (int j = 0; j < np; ++j) {
            const int64_t q = pos0 - np + j;
            prev[j] = q >= 0 ? hist[(size_t) q] : 0;
        }
        for (size_t k = 0; k < en.layers.size(); ++k) {
            std::vector<int32_t> ids((size_t) T * n_cols);
            engram_hash(en.hash, (int) k, tokens, T, pos0, pos0 == 0 ? nullptr : prev, ids.data());
            std::vector<uint32_t> rows(ids.begin(), ids.end());
            r.tickets[k] = m_.engram_reader().issue(rows.data(), rows.size(), raw(r, k));
        }
    }

    void finish_reads(EngramReads& r) {
        std::string ignored;
        for (size_t k = 0; k < r.open.size(); ++k)
            if (r.open[k]) m_.engram_reader().collect(r.tickets[k], ignored);
        r.open.clear();
    }

    struct Tick {
        const std::function<void(int)>* fn = nullptr;
        int tokens = 0;
    };
    static void CUDART_CB tick(void* p) {
        const Tick* t = (const Tick*) p;
        if (t->fn && *t->fn) (*t->fn)(t->tokens);
    }
    std::vector<Tick> ticks_;
    EngramReads ahead_;
    PromptHost* host_ = nullptr;   ///< the tier's: embedding and Engram pieces, the two Engram read sets
    int64_t cut_ = 0;   ///< positions before this one stop at skip_layer()'s compressor
    bool left_rows_ = false;
    static constexpr size_t kEngramThreads = 8;

    bool fail(const std::string& s) {
        err_ = s;
        return false;
    }
    const float* wf(int layer, const char* suffix) {
        const DevTensor* t = m_.tensor(layer, suffix);
        return t ? (const float*) t->ptr : nullptr;
    }

    // `embd` set: the chunk is input embeddings, not tokens (picture rows).
    // `next` and `next_T`: the chunk after this one, whose Engram reads start now.
    bool chunk(const int32_t* tokens_in, const float* embd, int T, int64_t pos0, int off, std::vector<int32_t>& hist,
               const int32_t* next, int next_T) {
        const size_t n = (size_t) g_.n_embd;
        void* st = stream_;
        std::string err;
        const std::vector<int32_t> dead((size_t) (embd ? T : 0), -1);
        const int32_t* tokens = embd ? dead.data() : tokens_in;
        hist.insert(hist.end(), tokens, tokens + T);

        // embeddings, repeated into the four streams, kEmbedRows tokens at a time
        {
            const int rows = prefillp::kEmbedRows;
            float* const emb = host_->embed.data();
            for (int t0 = 0; t0 < T; t0 += rows) {
                const int nt = std::min(rows, T - t0);
                for (int i = 0; i < nt; ++i) {
                    const int t = t0 + i;
                    float* row = emb + (size_t) i * kHcMult * n;
                    if (embd) std::copy(embd + (size_t) t * n, embd + (size_t) (t + 1) * n, row);
                    else if (!m_.embedding_row(tokens[t], row, err)) return fail(err);
                    for (int s = 1; s < kHcMult; ++s) std::copy(row, row + n, row + (size_t) s * n);
                }
                DS_CUDA(cudaMemcpyAsync(x_ + (size_t) t0 * kHcMult * n, emb, (size_t) nt * kHcMult * n * sizeof(float),
                                        cudaMemcpyHostToDevice, stream_));
                DS_CUDA(cudaStreamSynchronize(stream_));
            }
        }

        // Engram rows for every Engram layer: this chunk's were requested a chunk ago, so their SSD reads overlapped it
        const EngramInfo& en = m_.pack().engram;
        const int n_cols = en.hash.n_cols();
        EngramReads cur;
        struct Close {
            PrefillImpl* s;
            EngramReads* r;
            ~Close() { s->finish_reads(*r); }
        } close_reads{this, &cur};
        if (!en.layers.empty() && !embd) {
            if (ahead_.pos0 == pos0 && ahead_.T == T) {
                cur = std::move(ahead_);
                ahead_ = EngramReads{};
            } else {
                issue_reads(cur, ahead_, tokens, T, pos0, hist);
            }
            if (next && next_T > 0) issue_reads(ahead_, cur, next, next_T, pos0 + T, hist);
        }

        // rows before `cut` stop at layer yl's compressor; with every row before it the chunk ends there
        const int yl = skip_layer();
        const int cut = yl > 0 && cut_ > pos0 ? (int) std::min<int64_t>(cut_ - pos0, T) : 0;
        const int n_run = cut == T ? yl + 1 : g_.n_layers;
        if (ffn_) {
            std::vector<int> layer_tokens((size_t) (cut == T ? yl : g_.n_layers));
            for (size_t il = 0; il < layer_tokens.size(); ++il) layer_tokens[il] = T - (yl > 0 && (int) il >= yl ? cut : 0);
            ffn_->chunk_layers(layer_tokens);
        }

        const float* pre_prev = nullptr;
        int fi = 0;
        HcArgs hca;
        hca.n_embd = (int) n;
        hca.rms_eps = 1e-20f;
        hca.hc_eps = g_.hc_eps;
        hca.sinkhorn_iters = g_.sinkhorn_iters;
        RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = (int) n;
        ra.k = g_.n_used;
        ra.weight_scale = g_.weight_scale;
        const int K = g_.n_used;

        // GPU-timeline marks around the attention and FFN of every layer, summed into the split line at the end
        std::vector<cudaEvent_t> ev((size_t) g_.n_layers * 3 + 1);
        for (cudaEvent_t& e : ev) cudaEventCreate(&e);
        cudaEventRecord(ev.back(), stream_);

        // a host function queued after each layer reports the layers done when the GPU gets there: no synchronization
        ticks_.resize((size_t) g_.n_layers);
        for (int il = 0; il < n_run; ++il) {
            for (size_t k = 0; k < en.layers.size(); ++k) {
                if (en.layers[k] != il || embd) continue;   // picture rows get no Engram contribution
                cur.open[k] = 0;
                if (!m_.engram_reader().collect(cur.tickets[k], err)) return fail(err);
                // dequantized kEngramRows tokens at a time into one host piece, each piece copied on
                const size_t nr = (size_t) T * n_cols, piece = (size_t) prefillp::kEngramRows * n_cols;
                float* const rows_f = host_->engram_rows.data();
                for (size_t p0 = 0; p0 < nr; p0 += piece) {
                    const size_t np = std::min(piece, nr - p0);
                    auto dequant = [&](size_t r0, size_t r1) {
                        for (size_t r = r0; r < r1; ++r)
                            engram_row_f32(en.type, en.row_values, raw(cur, k) + (p0 + r) * en.row_bytes, rows_f + r * 256);
                    };
                    {
                        std::vector<std::thread> th;
                        const size_t step = np / kEngramThreads + 1;
                        for (size_t r0 = step; r0 < np; r0 += step) th.emplace_back(dequant, r0, std::min(np, r0 + step));
                        dequant(0, std::min(np, step));
                        for (std::thread& t : th) t.join();
                    }
                    DS_CUDA(cudaMemcpyAsync(rows_all_ + p0 * 256, rows_f, np * 256 * sizeof(float),
                                            cudaMemcpyHostToDevice, stream_));
                    DS_CUDA(cudaStreamSynchronize(stream_));
                }
                const DevTensor* kvw = m_.tensor(il, "engram_kv.weight");
                const DevTensor* gw = m_.tensor(il, "engram_gate.weight");
                if (!kvw || !gw) return fail("layer " + std::to_string(il) + " has no Engram tensors");
                const int in = n_cols * 256;
                for (int t0 = 0; t0 < T; t0 += 8) {
                    const int nc = std::min(8, T - t0);
                    dense_f32(*kvw, rows_all_ + (size_t) t0 * in, q8_, kv_dev_ + (size_t) t0 * 5 * n, in,
                                             (int) (5 * n), nc, st);
                }
                EngramArgs ea;
                ea.n_embd = (int) n;
                engram_apply(kv_dev_, (const float*) gw->ptr, nullptr, T, ea, x_, st);
            }

            // past layer yl only the rows from the cut on; at it every row reaches the attention sublayer's input
            const int r0 = yl > 0 && il > yl ? cut : 0;
            const int r1 = yl > 0 && il >= yl ? cut : 0;
            const int Ta = T - r0, Tf = T - r1;
            if (layer_tap) layer_tap(il, x_ + (size_t) r0 * 4 * n, Ta, st);
            // attention sublayer
            hc_mixes(x_ + (size_t) r0 * 4 * n, wf(il, "hc_attn_fn.weight"), wf(il, "hc_attn_scale.weight"), wf(il, "hc_attn_base.weight"), Ta,
                     hca, hc_ws_, coef_a_ + (size_t) r0 * kHcCoef, st);
            if (pre_prev) {
                hc_pre_rms_norm(x_ + (size_t) r0 * 4 * n, pre_prev + (size_t) r0 * kHcCoef, kHcCoef, wf(il, "attn_norm.weight"), Ta, (int) n, g_.rms_eps,
                                cur_ + (size_t) r0 * n, normed_ + (size_t) r0 * n, st);
            } else {
                hc_pre_first(T, n);
                rms_norm(cur_, wf(il, "attn_norm.weight"), T, (int) n, g_.rms_eps, normed_, st);
            }
            cudaEventRecord(ev[(size_t) il * 3], stream_);
            if (il == yl && cut > 0 && !ops_.compress_batch(il, pos0, cut, normed_, st, err)) return fail(err);
            if (Tf > 0 && !ops_.attention_batch(il, pos0 + r1, Tf, tokens + r1, normed_ + (size_t) r1 * n, attn_ + (size_t) r1 * n, (int) n, st, err))
                return fail(err);
            cudaEventRecord(ev[(size_t) il * 3 + 1], stream_);
            if (Tf > 0) {
                dump_rows("attn_out", il, pos0 + r1, Tf, attn_ + (size_t) r1 * n, n);
                hc_post(attn_ + (size_t) r1 * n, x_ + (size_t) r1 * 4 * n, coef_a_ + (size_t) r1 * kHcCoef, kHcCoef, Tf, (int) n, x2_ + (size_t) r1 * 4 * n, st);

                // FFN sublayer
                float* coef_f = coef_f_[fi];
                hc_mixes(x2_ + (size_t) r1 * 4 * n, wf(il, "hc_ffn_fn.weight"), wf(il, "hc_ffn_scale.weight"), wf(il, "hc_ffn_base.weight"), Tf,
                         hca, hc_ws_, coef_f + (size_t) r1 * kHcCoef, st);
                hc_pre_rms_norm(x2_ + (size_t) r1 * 4 * n, coef_a_ + (size_t) r1 * kHcCoef, kHcCoef, wf(il, "ffn_norm.weight"), Tf, (int) n, g_.rms_eps,
                                cur_ + (size_t) r1 * n, normed_ + (size_t) r1 * n, st);
                // picture rows select experts with the image bias
                const float* bias = embd ? wf(il, "exp_probs_b_vl.bias") : nullptr;
                if (!bias) bias = wf(il, "exp_probs_b.bias");
                const float* nx = normed_ + (size_t) r1 * n;
                router(nx, wf(il, "ffn_gate_inp.weight"), bias, Tf, ra, rlogits_, ids_, rw_, st);
                const bool ok_r = ffn_ ? ffn_->routed(il, Tf, nx, ids_, rw_, moe_, st, err)
                                       : ops_.routed_experts_batch(il, pos0 + r1, Tf, nx, ids_, rw_, K, moe_, (int) n, st, err);
                if (!ok_r) return fail(err);
                const bool ok_s = ffn_ ? ffn_->shared(il, Tf, nx, sh_, st, err)
                                       : ops_.shared_expert_batch(il, pos0 + r1, Tf, nx, sh_, (int) n, st, err);
                if (!ok_s) return fail(err);
                dump_rows("ffn_moe_out", il, pos0 + r1, Tf, moe_, n);
                dump_rows("ffn_shexp", il, pos0 + r1, Tf, sh_, n);
                add(moe_, sh_, sum_, (int) (Tf * n), st);
                hc_post(sum_, x2_ + (size_t) r1 * 4 * n, coef_f + (size_t) r1 * kHcCoef, kHcCoef, Tf, (int) n, x_ + (size_t) r1 * 4 * n, st);
                dump_rows("l_out", il, pos0 + r1, Tf, x_ + (size_t) r1 * 4 * n, 4 * n);
                pre_prev = coef_f;
                fi ^= 1;
            }
            cudaEventRecord(ev[(size_t) il * 3 + 2], stream_);
            if (live && il + 1 < n_run) {   // the chunk's end is reported by `progress`
                ticks_[(size_t) il] = {&live, off + (int) ((int64_t) T * (il + 1) / n_run)};
                cudaLaunchHostFunc(stream_, &PrefillImpl::tick, &ticks_[(size_t) il]);
            }
        }
        DS_CUDA(cudaStreamSynchronize(stream_));
        if (chunk_done && !chunk_done(T, pos0, err)) return fail(err);
        {
            float attn_ms = 0, ffn_ms = 0, all_ms = 0, t = 0;
            for (int il = 0; il < n_run; ++il) {
                cudaEventElapsedTime(&t, ev[(size_t) il * 3], ev[(size_t) il * 3 + 1]);
                attn_ms += t;
                cudaEventElapsedTime(&t, ev[(size_t) il * 3 + 1], ev[(size_t) il * 3 + 2]);
                ffn_ms += t;
            }
            cudaEventElapsedTime(&all_ms, ev.back(), ev[(size_t) (n_run - 1) * 3 + 2]);
            // the first layer also carries the one-time tier load; the rest scale to a whole pass
            float rest_ms = 0;
            cudaEventElapsedTime(&rest_ms, ev[2], ev[(size_t) (n_run - 1) * 3 + 2]);
            char far[48] = "";
            if (cut > 0) std::snprintf(far, sizeof far, ", %d of them to layer %d", cut, yl);
            basalt::log::info("basalt deepseek41", "prefill chunk of %d tokens%s: attention %.0f ms, ffn %.0f ms, other %.0f ms; "
                         "layers 1.. %.0f ms (%.0f ms per pass)\n", T, far, attn_ms, ffn_ms, all_ms - attn_ms - ffn_ms, rest_ms,
                         rest_ms * n_run / (n_run - 1));
        }
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
        return true;
    }

    // dev rows [T][width] -> `<dir>/<name>.<layer>.<pos>.bin` per token, as the single-token loop writes them
    void dump_rows(const char* name, int layer, int64_t pos0, int T, const float* dev, size_t width) {
        if (dump_dir_.empty()) return;
        std::vector<float> h((size_t) T * width);
        cudaStreamSynchronize(stream_);
        cudaMemcpy(h.data(), dev, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (int t = 0; t < T; ++t) {
            const std::string path = dump_dir_ + "/" + name + "." + std::to_string(layer) + "." +
                                     std::to_string(pos0 + t) + ".bin";
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(h.data() + (size_t) t * width, sizeof(float), width, f);
                std::fclose(f);
            }
        }
    }

    // First layer: the collapse input is the stream 0 of each token.
    void hc_pre_first(int T, size_t n) {
        cudaMemcpy2DAsync(cur_, n * sizeof(float), x_, 4 * n * sizeof(float), n * sizeof(float), (size_t) T,
                          cudaMemcpyDeviceToDevice, stream_);
    }

    DsModel& m_;
    const Geometry& g_;
    LayerOps& ops_;
    cudaStream_t stream_;
    int cap_;
    bool batch_open_ = false;   // the ops hold this read's batched attention
    std::vector<void*> allocs_;
    std::string err_;
    float *x_ = nullptr, *x2_ = nullptr, *cur_ = nullptr, *normed_ = nullptr, *attn_ = nullptr, *moe_ = nullptr,
          *sh_ = nullptr, *sum_ = nullptr, *coef_a_ = nullptr, *coef_f_[2] = {nullptr, nullptr}, *hc_ws_ = nullptr,
          *rlogits_ = nullptr, *rw_ = nullptr, *kv_dev_ = nullptr, *rows_all_ = nullptr;
    int32_t* ids_ = nullptr;
    uint8_t* q8_ = nullptr;
    std::string dump_dir_;
    std::unique_ptr<PrefillFfn> ffn_;
};

}  // namespace

std::unique_ptr<Prefill> make_prefill(DsModel& model, LayerOps& ops, void* stream, int max_chunk, std::string& err) {
    auto p = std::make_unique<PrefillImpl>(model, ops, stream, max_chunk);
    if (!p->init(err)) return nullptr;
    return p;
}

}  // namespace basalt::models::deepseek41
