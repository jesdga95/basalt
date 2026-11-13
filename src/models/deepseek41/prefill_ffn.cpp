// src/models/deepseek41/prefill_ffn.cpp - the routed and shared experts over a chunk of prompt tokens.
//
// Routed: the (token, slot) pairs of the chunk are grouped by expert; each expert's blob is staged once (copied from the
// VRAM tier or the pinned RAM tier, else read from the file) and runs over all its tokens on the MMQ products (gate/up,
// clamped SwiGLU, down). Staging of one launch overlaps the products of the previous one. A second card computes its own
// experts and a share of the RAM ones, read over its own link, beside the main card (prefill::PeerShare); in a short chunk
// the CPU also takes RAM experts with few rows. Their rows join the main card's before the weighted sum. Shared: the dense
// matvec over columns of up to eight tokens.
#include "basalt/log.hpp"
#include "models/deepseek41/prefill_ffn.hpp"

#include "basalt/core/on_device.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "models/deepseek41/model.hpp"
#include "models/deepseek41/layer_ops.hpp"
#include "models/deepseek41/ops.hpp"
#include "models/deepseek41/prefill_plan.hpp"
#include "models/deepseek41/tier.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <cstdio>
#include <thread>
#include <vector>

namespace basalt::models::deepseek41 {
namespace {

using prefillp::kStageExperts;
constexpr int kReadThreads = 8;     // concurrent blob reads
constexpr int kSharedRows = 1024;   // tokens per shared-expert sub-chunk
constexpr int kCpuRows = 8;         // a RAM expert with this few rows in a short chunk may stay on the CPU
constexpr int kNvfp4 = 40;          // ggml's NVFP4: its blob ends in the FP32 scale tail

// The scale tails of a launch's experts staged `staged_stride` apart from `blobs` (none for a format without them).
ExpertTails tails_of(const ExpertLayer& el, const uint8_t* blobs, const int32_t* bounds, int n) {
    ExpertTails t;
    if (el.gu_type != kNvfp4) return t;
    t.blobs = blobs;
    t.stride = staged_stride(el);
    t.tail_off = el.blob_bytes - kernels::cpu::kNvfp4Tail;
    t.bounds = bounds;
    t.n = n;
    return t;
}

class Impl final : public PrefillFfn {
public:
    Impl(const DsModel& m, LayerOps& ops, int cap) : m_(m), g_(m.geo()), ops_(ops), cap_(cap) {}
    ~Impl() override {
        stop_prefetch();
        for (cudaEvent_t e : pf_events_) cudaEventDestroy(e);
        for (const auto& o : pf_out_) cudaEventDestroy(o.first);
        for (void* p : dev_) cudaFree(p);
        for (Stage& b : stage_) {
            if (b.copied) cudaEventDestroy(b.copied);
            if (b.done) cudaEventDestroy(b.done);
        }
        if (copy_) cudaStreamDestroy(copy_);
    }

    bool init(std::string& err) {
        namespace mmq = basalt::prefill::mmq;
        for (const ExpertLayer& e : m_.pack().experts)
            if (!mmq::fits(e.gu_type, 2 * g_.n_ff_exp) || !mmq::fits(e.d_type, g_.n_embd)) {
                err = "deepseek41: the prompt MMQ path does not cover an expert format of this pack";
                return false;
            }
        // the host buffers were allocated with the tier, before it took the free RAM
        tier_tried_ = true;
        tier_ = ops_.expert_tier(err);
        if (!tier_) {
            if (err.empty()) err = "deepseek41: the prompt path needs the expert tier";
            return false;
        }
        PromptHost& host = tier_->host();
        slot_ = host.slot;   // the products step from expert to expert in whole blocks of the layer's two formats
        const size_t K = (size_t) g_.n_used, n = (size_t) g_.n_embd;
        max_rows_ = prefillp::stage_rows(K, cap_);
        const size_t E = (size_t) cap_ * K, R = (size_t) max_rows_, ff = (size_t) g_.n_ff_exp;
        // the largest q8 buffer either product reads
        bool ok = dev(xq1_, mmq::q8_bytes((int64_t) R, g_.n_embd)) &&
                  dev(xq2_, mmq::q8_bytes((int64_t) R, g_.n_ff_exp));
        ok = ok && dev(gu_, R * 2 * ff * 4) && dev(h_, R * ff * 4);
        ok = ok && dev(ident_, E * 4) && dev(rows_, E * n * 4) && dev(map_, E * 4) && dev(sbnd_, 8 * 4);
        for (int i = 0; i < 2; ++i) {
            Stage& b = stage_[i];
            b.host = host.stage[i];
            ok = ok && dev(b.blobs, slot_ * kStageExperts) && dev(b.tok, R * 4) &&
                 dev(b.bounds, (kStageExperts + 1) * 4) &&
                 cudaEventCreateWithFlags(&b.copied, cudaEventDisableTiming) == cudaSuccess &&
                 cudaEventCreateWithFlags(&b.done, cudaEventDisableTiming) == cudaSuccess;
        }
        ok = ok && cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) == cudaSuccess;
        if (!ok) {
            err = "out of memory for the prefill expert buffers";
            return false;
        }
        mmq::iota(ident_, (int64_t) E, nullptr);
        map_h_.resize(E);
        cudaDeviceSynchronize();
        // every tile the products may pick, on each card that runs them: a tile's first launch in a long chunk would wait
        // for the expert copies queued on the card
        std::vector<std::pair<int, int64_t>> routed, all;
        auto add = [](std::vector<std::pair<int, int64_t>>& v, int t, int64_t rows) {
            if (std::find(v.begin(), v.end(), std::make_pair(t, rows)) == v.end()) v.emplace_back(t, rows);
        };
        for (const ExpertLayer& e : m_.pack().experts) {
            add(routed, e.gu_type, 2 * (int64_t) g_.n_ff_exp);
            add(routed, e.d_type, g_.n_embd);
        }
        all = routed;
        const int64_t nf = (int64_t) g_.n_ff_exp * std::max(g_.n_shared, 1);
        const std::pair<const char*, int64_t> shexp[] = {{"ffn_gate_shexp.weight", nf}, {"ffn_up_shexp.weight", nf},
                                                         {"ffn_down_shexp.weight", g_.n_embd}};
        for (int il = 0; il < g_.n_layers; ++il)
            for (const auto& [name, rows] : shexp)
                if (const DevTensor* t = m_.tensor(il, name); t && mmq::fits(t->type, rows)) add(all, t->type, rows);
        for (const auto& [t, rows] : all) mmq::load(t, rows);
        if (tier_->peer_share()) {
            const core::OnDevice od(tier_->peer()->device());
            for (const auto& [t, rows] : routed) mmq::load(t, rows);
        }
        return true;
    }

    void chunk_layers(const std::vector<int>& tokens) override {
        run_layers_ = (int) tokens.size();
        layer_tokens_ = tokens;
    }

    bool routed(int il, int T, const float* x, const int32_t* ids, const float* w, float* out, void* stream,
                std::string& err) override {
        namespace mmq = basalt::prefill::mmq;
        const int K = g_.n_used;
        const ExpertLayer& el = m_.pack().experts[(size_t) il];
        const kernels::NativeExpertLayout L = kernels::native_expert_layout(el.gu_type, el.d_type, g_.n_embd, g_.n_ff_exp);
        if (L.bytes != el.blob_bytes) {
            err = "deepseek41: layer " + std::to_string(il) + " expert blob size does not match the layout";
            return false;
        }
        cudaStream_t st = (cudaStream_t) stream;
        if (!tier_ && !tier_tried_) {
            tier_tried_ = true;
            std::string terr;
            tier_ = ops_.expert_tier(terr);
            if (!tier_ && !terr.empty()) basalt::log::warn("basalt deepseek41", "prefill reads experts from the file (%s)", terr.c_str());
        }
        if (il == 0) {
            stop_prefetch();
            ahead_ = false;
            if (tier_ && (int64_t) T * K >= prefillp::kAheadPairs) ahead_ = start_ahead();
        }
        if (ahead_) return routed_ahead(il, T, x, ids, w, out, st, err);
        std::vector<int32_t> hid((size_t) T * K);
        cudaStreamSynchronize(st);
        cudaMemcpy(hid.data(), ids, hid.size() * 4, cudaMemcpyDeviceToHost);
        std::vector<std::vector<int32_t>> by((size_t) g_.n_expert);   // pair = t * K + slot
        for (int e = 0; e < T * K; ++e) {
            if (hid[(size_t) e] < 0 || hid[(size_t) e] >= g_.n_expert) {
                err = "deepseek41: router returned a bad expert id";
                return false;
            }
            by[(size_t) hid[(size_t) e]].push_back(e);
        }
        std::vector<int> used;
        for (int e = 0; e < g_.n_expert; ++e)
            if (!by[(size_t) e].empty()) used.push_back(e);
        Share share;
        if (tier_ && !plan_share(il, T, x, by, used, share, stream, err)) return false;

        // the main card's rows first, in launch order; the second card's and the CPU's blocks after them
        int64_t pos = 0;
        for (size_t b0 = 0; b0 < used.size();) {
            // experts of this launch: up to kStageExperts, up to max_rows_ rows
            int nb = 0;
            int64_t rows = 0, max_rows = 0;
            while (b0 + (size_t) nb < used.size() && nb < kStageExperts) {
                const int64_t c = (int64_t) by[(size_t) used[b0 + (size_t) nb]].size();
                if (nb > 0 && rows + c > max_rows_) break;
                rows += c;
                max_rows = std::max(max_rows, c);
                ++nb;
            }
            Stage& B = stage_[launch_++ & 1];
            cudaEventSynchronize(B.copied);                  // the pinned staging of the launch before last is free
            cudaStreamWaitEvent(copy_, B.done, 0);           // so is its device buffer
            std::vector<int> from_file;
            for (int q = 0; q < nb; ++q) {
                const int e = used[b0 + (size_t) q];
                uint8_t* dst = B.blobs + staged_stride(el) * (size_t) q;
                ExpertTier::Source src;
                if (tier_) src = tier_->locate(il, e);
                if (src.dev) {
                    cudaMemcpyAsync(dst, src.dev, L.bytes, cudaMemcpyDeviceToDevice, copy_);
                } else if (src.pinned) {
                    cudaMemcpyAsync(dst, src.pinned, L.bytes, cudaMemcpyHostToDevice, copy_);
                } else {
                    from_file.push_back(q);
                }
            }
            if (!from_file.empty()) {
                std::atomic<bool> read_ok{true};
                std::atomic<size_t> next{0};
                auto worker = [&] {
                    for (size_t i; (i = next.fetch_add(1)) < from_file.size();) {
                        const int q = from_file[i];
                        const int e = used[b0 + (size_t) q];
                        const uint64_t off = m_.pack().experts_range.offset + el.offset + (uint64_t) e * el.blob_bytes;
                        if (!platform::pread_all(m_.experts_fd(), B.host + slot_ * (size_t) q, L.bytes, off)) read_ok = false;
                    }
                };
                std::vector<std::thread> th;
                for (int i = 1; i < std::min<int>(kReadThreads, (int) from_file.size()); ++i) th.emplace_back(worker);
                worker();
                for (std::thread& t : th) t.join();
                if (!read_ok) {
                    err = "deepseek41: short read of an expert blob";
                    return false;
                }
                for (int q : from_file)
                    cudaMemcpyAsync(B.blobs + staged_stride(el) * (size_t) q, B.host + slot_ * (size_t) q, L.bytes,
                                    cudaMemcpyHostToDevice, copy_);
            }
            cudaEventRecord(B.copied, copy_);
            std::vector<int32_t> tok, bounds{0};
            for (int q = 0; q < nb; ++q) {
                for (int32_t p : by[(size_t) used[b0 + (size_t) q]]) {
                    map_h_[(size_t) p] = (int32_t) (pos + (int64_t) tok.size());
                    tok.push_back(p / K);
                }
                bounds.push_back((int32_t) tok.size());
            }
            cudaMemcpyAsync(B.tok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(B.bounds, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice, st);
            cudaStreamWaitEvent(st, B.copied, 0);

            mmq::quantize(x, B.tok, xq1_, el.gu_type, g_.n_embd, g_.n_embd, rows, stream);
            mmq::Product p;
            p.w = B.blobs;
            p.type = el.gu_type;
            p.w_rows = 2 * (int64_t) g_.n_ff_exp;
            p.w_cols = g_.n_embd;
            p.expert_bytes = staged_stride(el);
            p.n = nb;
            p.xq = xq1_;
            p.bounds = B.bounds;
            p.ids = ident_;
            p.total_rows = rows;
            p.max_rows = max_rows;
            p.dst = gu_;
            p.ld_dst = p.w_rows;
            ctx_.run(p, stream);
            const ExpertTails tails = tails_of(el, B.blobs, B.bounds, nb);
            swiglu_split_clamped(gu_, h_, rows, g_.n_ff_exp, g_.clamp_exp[(size_t) il], stream, &tails);
            mmq::quantize(h_, nullptr, xq2_, el.d_type, g_.n_ff_exp, g_.n_ff_exp, rows, stream);
            mmq::Product d;
            d.w = B.blobs + L.down_off;
            d.type = el.d_type;
            d.w_rows = g_.n_embd;
            d.w_cols = g_.n_ff_exp;
            d.expert_bytes = staged_stride(el);
            d.n = nb;
            d.xq = xq2_;
            d.bounds = B.bounds;
            d.ids = ident_ + pos;
            d.total_rows = rows;
            d.max_rows = max_rows;
            d.dst = rows_;
            d.ld_dst = g_.n_embd;
            ctx_.run(d, stream);
            cudaEventRecord(B.done, st);
            b0 += (size_t) nb;
            pos += rows;
        }
        if (!finish_share(share, stream, err)) return false;
        cudaMemcpyAsync(map_, map_h_.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, st);
        weighted_sum_tokens(rows_, map_, w, T, K, g_.n_embd, out, stream);
        return true;
    }

    bool shared(int il, int T, const float* x, float* out, void* stream, std::string& err) override {
        auto Tn = [&](const char* s) { return m_.tensor(il, s); };
        const DevTensor *gt = Tn("ffn_gate_shexp.weight"), *up = Tn("ffn_up_shexp.weight"), *dn = Tn("ffn_down_shexp.weight");
        if (!gt || !up || !dn) {
            err = "deepseek41: layer " + std::to_string(il) + " lacks shared expert tensors";
            return false;
        }
        const int n = g_.n_embd, nf = g_.n_ff_exp * std::max(g_.n_shared, 1);
        namespace mmq = basalt::prefill::mmq;
        if (!mmq::fits(gt->type, nf) || !mmq::fits(up->type, nf) || !mmq::fits(dn->type, n)) {
            err = "deepseek41: the prompt MMQ path does not cover a shared expert format of this pack";
            return false;
        }
        float* sg = gu_;
        float* su = gu_ + (size_t) kSharedRows * nf;
        // the three dense products over sub-chunks of kSharedRows tokens
        for (int t0 = 0; t0 < T; t0 += kSharedRows) {
            const int nc = std::min(kSharedRows, T - t0);
            const int32_t b[2] = {0, nc};
            cudaMemcpyAsync(sbnd_, b, sizeof b, cudaMemcpyHostToDevice, (cudaStream_t) stream);
            mmq::quantize(x + (size_t) t0 * n, nullptr, xq1_, gt->type, n, n, nc, stream);
            dense(gt, nf, n, xq1_, nc, sg, stream);
            if (up->type != gt->type) mmq::quantize(x + (size_t) t0 * n, nullptr, xq1_, up->type, n, n, nc, stream);
            dense(up, nf, n, xq1_, nc, su, stream);
            swiglu_clamped(sg, su, nf * nc, g_.clamp_shexp[(size_t) il], h_, stream);
            mmq::quantize(h_, nullptr, xq2_, dn->type, nf, nf, nc, stream);
            dense(dn, n, nf, xq2_, nc, out + (size_t) t0 * n, stream);
        }
        return true;
    }

private:
    struct Share {
        std::vector<kernels::cpu::ExpertJobMulti> jobs;
        std::vector<int32_t> act_tokens, pair;   // tokens whose activations the CPU needs; each CPU row's (token, slot) pair
        int cpu_rows = 0;
        int64_t cpu_at = 0;                      // the CPU rows' first row in the layer's rows
        bool peer = false;                       // the second card took a share
        std::thread cpu_thread;
        void join() {
            if (cpu_thread.joinable()) cpu_thread.join();
        }
        ~Share() { join(); }
    };

    // A layer's formats and its clamped SwiGLU, for the second card.
    prefill::PeerShare::Layer share_layer(int il) const {
        const ExpertLayer& el = m_.pack().experts[(size_t) il];
        const kernels::NativeExpertLayout L = kernels::native_expert_layout(el.gu_type, el.d_type, g_.n_embd, g_.n_ff_exp);
        prefill::PeerShare::Layer sl;
        sl.gu_type = el.gu_type;
        sl.d_type = el.d_type;
        sl.blob_bytes = el.blob_bytes;
        sl.stride = staged_stride(el);
        sl.gu_stride = sl.stride;
        sl.down_off = L.down_off;
        sl.d_stride = sl.stride;
        const int n_ff = g_.n_ff_exp;
        const float limit = g_.clamp_exp[(size_t) il];
        sl.act = [n_ff, limit, el](const float* gu, float* h, const prefill::PeerShare::Launch& l, void* stream) {
            const ExpertTails tails = tails_of(el, l.stage + staged_stride(el) * (size_t) l.q0, l.bounds, l.n);
            swiglu_split_clamped(gu, h, l.rows, n_ff, limit, stream, &tails);
        };
        return sl;
    }

    // Starts the second card on `peer` experts of layer `il` (its homes and the RAM ones it takes), rows by `by`; its rows
    // fill the layer's rows from `at` on. Returns that block's end in `end`.
    bool start_peer(int il, int T, const float* x, const std::vector<std::vector<int32_t>>& by, const std::vector<int>& peer,
                    int64_t at, int64_t& end, void* stream, std::string& err) {
        const int K = g_.n_used;
        std::vector<std::vector<int32_t>> tok(peer.size());
        std::vector<prefill::PeerShare::Expert> ex;
        end = at;
        for (size_t i = 0; i < peer.size(); ++i) {
            const int e = peer[i];
            const ExpertTier::Source src = tier_->locate(il, e);
            for (int32_t p : by[(size_t) e]) {
                tok[i].push_back(p / K);
                map_h_[(size_t) p] = (int32_t) end++;
            }
            prefill::PeerShare::Expert x1;
            x1.blob = src.peer ? src.peer : src.pinned;
            x1.on_peer = src.peer != nullptr;
            x1.tok = tok[i].data();
            x1.n = (int) tok[i].size();
            ex.push_back(x1);
        }
        return tier_->peer_share()->run(share_layer(il), x, T, ex, rows_ + (size_t) at * (size_t) g_.n_embd, stream, err);
    }

    // The second card's experts of layer `il` in a long chunk start their staging before its routing.
    void prepare_peer(int il) {
        if (!tier_->peer_share()) return;
        std::vector<prefill::PeerShare::Expert> ex;
        for (int e = 0; e < g_.n_expert; ++e) {
            if (!on_peer_[(size_t) il][(size_t) e]) continue;
            const ExpertTier::Source src = tier_->locate(il, e);
            prefill::PeerShare::Expert x1;
            x1.blob = src.peer ? src.peer : src.pinned;
            x1.on_peer = src.peer != nullptr;
            ex.push_back(x1);
        }
        if (!ex.empty()) tier_->peer_share()->prepare(share_layer(il), ex);
    }

    // Of the RAM experts `ram` (highest ids first, `rows` each), how many the second card takes beside its own experts. In a
    // long chunk (`tokens` > 0) the main card's link reads ahead while the layer's attention runs.
    int peer_count(int il, const std::vector<int64_t>& ram_rows, int64_t main_rows, double main_bytes, int64_t peer_rows,
                   int tokens = 0) const {
        const ExpertLayer& el = m_.pack().experts[(size_t) il];
        const prefill::ShareRates& r = tier_->share_rates(il);
        const double lead = r.sm_hz0 > 0 ? (double) tokens * prefillp::kPreRouteCycles / r.sm_hz0 : 0;
        return prefill::peer_take(r, ram_rows, main_rows, main_bytes, peer_rows, (double) el.blob_bytes,
                                  (double) (basalt::prefill::mmq::q8_bytes(2, g_.n_embd) - basalt::prefill::mmq::q8_bytes(1, g_.n_embd)),
                                  4.0 * (double) g_.n_embd, lead);
    }

    bool share_buffers(std::string& err) {
        if (hrows_) return true;
        (void) err;
        const PromptHost& host = tier_->host();
        hx_ = host.hx;
        hrows_ = host.hrows;
        act_ = host.act;
        return true;
    }

    // Takes out of `used` what the second card and the CPU compute, and starts them; the main card's bus carries the rest.
    bool plan_share(int il, int T, const float* x, const std::vector<std::vector<int32_t>>& by, std::vector<int>& used, Share& s,
                    void* stream, std::string& err) {
        const int K = g_.n_used;
        if ((int64_t) T * K >= prefillp::kAheadPairs) return true;
        int pinned = 0;
        for (int e : used) pinned += tier_->locate(il, e).pinned != nullptr;
        if (pinned == 0 && !tier_->peer()) return true;
        if (!share_buffers(err)) return false;
        const size_t n = (size_t) g_.n_embd;
        std::vector<uint8_t> stays((size_t) g_.n_expert, 0);
        int rows = 0;
        // the CPU's: one-row experts, then two-row ones, while the bus would still carry as many experts as it has rows
        std::vector<int32_t> tok_row((size_t) T, -1);
        for (int c = 1; c <= kCpuRows; ++c) {
            for (int e : used) {
                const ExpertTier::Source src = tier_->locate(il, e);
                if (!src.pinned || (int) by[(size_t) e].size() != c || stays[(size_t) e]) continue;
                if (s.cpu_rows + c > pinned - (int) s.jobs.size() - 1 || rows + c > prefillp::kShareRows) break;
                kernels::cpu::ExpertJobMulti j;
                j.blob = src.pinned;
                j.nt = c;
                for (int i = 0; i < c; ++i) {
                    const int32_t p = by[(size_t) e][(size_t) i], t = p / K;
                    if (tok_row[(size_t) t] < 0) {
                        tok_row[(size_t) t] = (int32_t) s.act_tokens.size();
                        s.act_tokens.push_back(t);
                    }
                    j.nact[i] = act_ + (size_t) tok_row[(size_t) t] * kernels::cpu::kNativeActBytes;
                    j.out[i] = hrows_ + (size_t) rows * n;
                    s.pair.push_back(p);
                    ++rows;
                }
                s.jobs.push_back(j);
                stays[(size_t) e] = 1;
                s.cpu_rows += c;
            }
        }
        // the second card's: its own experts, and the RAM ones it reads over its own link where that prices lower
        std::vector<int> peer;
        if (tier_->peer_share()) {
            const ExpertLayer& el = m_.pack().experts[(size_t) il];
            std::vector<int> ram;
            int64_t main_rows = 0, peer_rows = 0;
            double main_bytes = 0;
            for (int e : used) {
                if (stays[(size_t) e]) continue;
                const ExpertTier::Source src = tier_->locate(il, e);
                const int64_t c = (int64_t) by[(size_t) e].size();
                if (src.peer) {
                    peer.push_back(e);
                    peer_rows += c;
                } else if (src.pinned) {
                    ram.push_back(e);
                } else {
                    main_rows += c;
                    if (!src.dev) main_bytes += (double) el.blob_bytes;
                }
            }
            std::reverse(ram.begin(), ram.end());   // the main card walks ids upward, the second card takes from the top
            std::vector<int64_t> ram_rows;
            for (int e : ram) ram_rows.push_back((int64_t) by[(size_t) e].size());
            const int take = peer_count(il, ram_rows, main_rows, main_bytes, peer_rows);
            peer.insert(peer.end(), ram.begin(), ram.begin() + take);
            for (int e : peer) stays[(size_t) e] = 1;
        }
        if (rows == 0 && peer.empty()) return true;
        used.erase(std::remove_if(used.begin(), used.end(), [&](int e) { return stays[(size_t) e] != 0; }), used.end());
        int64_t at = 0;
        for (int e : used) at += (int64_t) by[(size_t) e].size();
        if (!peer.empty()) {
            if (!start_peer(il, T, x, by, peer, at, at, stream, err)) return false;
            s.peer = true;
        }
        s.cpu_at = at;
        for (size_t i = 0; i < s.pair.size(); ++i) map_h_[(size_t) s.pair[i]] = (int32_t) (at + (int64_t) i);
        if (rows == 0) return true;
        if (cudaMemcpy(hx_, x, (size_t) T * n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "deepseek41: reading a short read's activations failed";
            return false;
        }
        s.cpu_thread = std::thread([this, il, &s] {
            const kernels::cpu::NativeFmt& f = tier_->fmt(il);
            for (size_t i = 0; i < s.act_tokens.size(); ++i)
                kernels::cpu::native_quant_act(f, hx_ + (size_t) s.act_tokens[i] * g_.n_embd, act_ + i * kernels::cpu::kNativeActBytes);
            tier_->pool()->run_split_multi_native(f, s.jobs.data(), (int) s.jobs.size());
        });
        return true;
    }

    // The rows of the CPU and the second card join the others'.
    bool finish_share(Share& s, void* stream, std::string& err) {
        s.join();
        if (s.peer && !tier_->peer_share()->join(stream, err)) return false;
        if (s.pair.empty()) return true;
        const size_t rows = s.pair.size(), n = (size_t) g_.n_embd;
        cudaMemcpyAsync(rows_ + (size_t) s.cpu_at * n, hrows_, rows * n * 4, cudaMemcpyHostToDevice, (cudaStream_t) stream);
        return true;
    }

    // Which experts of each layer the second card computes in a long chunk: its own, and the RAM ones it reads over its
    // own link where that prices lower, at the layer's tokens with every expert taking an even share of the rows (the
    // groups are staged before the routing is known).
    void plan_peer() {
        const int nl = g_.n_layers, K = g_.n_used, NE = g_.n_expert;
        on_peer_.assign((size_t) nl, std::vector<uint8_t>((size_t) NE, 0));
        if (!tier_->peer_share()) return;
        for (int l = 0; l < run_layers_; ++l) {
            const int64_t each = (int64_t) layer_tokens_[(size_t) l] * K / NE;
            std::vector<int> ram;
            int64_t main_rows = 0, peer_rows = 0;
            double main_bytes = 0;
            for (int e = NE - 1; e >= 0; --e) {
                const ExpertTier::Source src = tier_->locate(l, e);
                if (src.peer) {
                    on_peer_[(size_t) l][(size_t) e] = 1;
                    peer_rows += each;
                } else if (src.pinned) {
                    ram.push_back(e);
                } else {
                    main_rows += each;
                    if (!src.dev) main_bytes += (double) m_.pack().experts[(size_t) l].blob_bytes;
                }
            }
            const int take = peer_count(l, std::vector<int64_t>(ram.size(), each), main_rows, main_bytes, peer_rows,
                                        layer_tokens_[(size_t) l]);
            for (int i = 0; i < take; ++i) on_peer_[(size_t) l][(size_t) ram[(size_t) i]] = 1;
        }
    }

    // Brings the experts only the file holds, in the order the products take them, a ring's worth ahead of them into
    // the pinned ring. Piece `seq` of that order sits in ring slot seq % ring_pieces_; a slot is read again once the copy
    // out of it is done.
    bool start_ahead() {
        const int nl = g_.n_layers;
        plan_peer();
        pf_idx_.assign((size_t) nl, std::vector<int32_t>((size_t) g_.n_expert, -1));
        pf_list_.assign((size_t) nl, {});
        pf_base_.assign((size_t) nl + 1, 0);
        for (int l = 0; l < nl; ++l) {
            for (int e = 0; e < g_.n_expert; ++e) {
                const ExpertTier::Source src = tier_->locate(l, e);
                if (src.dev || src.pinned || src.peer) continue;
                pf_idx_[(size_t) l][(size_t) e] = (int32_t) pf_list_[(size_t) l].size();
                pf_list_[(size_t) l].push_back(e);
            }
            pf_base_[(size_t) l + 1] = pf_base_[(size_t) l] + (int64_t) pf_list_[(size_t) l].size();
        }
        pf_on_ = false;
        const PromptHost& host = tier_->host();
        ring_ = host.ring;
        ring_pieces_ = host.ring_pieces;
        if (pf_base_.back() > 0 && ring_pieces_ < kStageExperts) return false;   // a launch's pieces must fit at once
        if (!ahead_buffers()) return false;
        cudaStreamSynchronize(copy_);
        for (const auto& o : pf_out_) pf_events_.push_back(o.first);
        pf_out_.clear();
        pf_ready_ = 0;
        pf_released_ = 0;
        pf_failed_ = false;
        pf_stop_ = false;
        issued_ = consumed_ = 0;
        if (pf_base_.back() > 0) {
            pf_on_ = true;
            pf_th_ = std::thread([this] { prefetch_loop(run_layers_); });
        }
        prepare_peer(0);
        return true;
    }

    // Hands the reader the slots whose copies are done: those already done, or with `wait_oldest` the oldest one too.
    void release_done(bool wait_oldest) {
        int64_t upto = -1;
        while (!pf_out_.empty()) {
            const auto [ev, end] = pf_out_.front();
            if (wait_oldest) {
                cudaEventSynchronize(ev);
                wait_oldest = false;
            } else if (cudaEventQuery(ev) != cudaSuccess) {
                break;
            }
            upto = end;
            pf_events_.push_back(ev);
            pf_out_.pop_front();
        }
        if (upto < 0) return;
        std::lock_guard<std::mutex> lk(pf_mu_);
        pf_released_ = upto;
        pf_cv_.notify_all();
    }

    // Waits until ring piece `seq` is read. While it waits, the slots of finished copies go back to the reader, which may
    // be waiting for one.
    bool wait_ready(int64_t seq, std::string& err) {
        for (;;) {
            release_done(false);
            {
                std::unique_lock<std::mutex> lk(pf_mu_);
                if (!pf_failed_ && pf_ready_ <= seq && pf_out_.empty())
                    pf_cv_.wait(lk, [&] { return pf_ready_ > seq || pf_failed_; });
                if (pf_failed_) {
                    err = "deepseek41: short read of an expert blob";
                    return false;
                }
                if (pf_ready_ > seq) return true;
            }
            release_done(true);
        }
    }

    bool ahead_buffers() {
        if (!dstage_[0].blobs) {
            const size_t E = (size_t) cap_ * g_.n_used;
            if (!dev(tok_all_, E * 4) || !dev(bnd_, (kStageExperts + 1) * 4)) return false;
            for (Stage& b : dstage_) {
                if (!dev(b.blobs, slot_ * kStageExperts) ||
                    cudaEventCreateWithFlags(&b.copied, cudaEventDisableTiming) != cudaSuccess ||
                    cudaEventCreateWithFlags(&b.done, cudaEventDisableTiming) != cudaSuccess)
                    return false;
            }
        }
        return true;
    }

    // Starts the copies of the next group of experts (in layer order) into its slot; they depend on no routing. The second
    // card's experts stay where they are.
    bool issue_next(std::string& err) {
        const int G = (g_.n_expert + kStageExperts - 1) / kStageExperts;
        const int l = (int) (issued_ / G), g = (int) (issued_ % G);
        Stage& D = dstage_[issued_ % prefillp::kAheadSlots];
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        int64_t last = -1;   // the group's last ring piece
        if (pf_on_)
            for (int q = 0; q < kStageExperts && g * kStageExperts + q < g_.n_expert; ++q)
                if (const int32_t i = pf_idx_[(size_t) l][(size_t) (g * kStageExperts + q)]; i >= 0) last = pf_base_[(size_t) l] + i;
        if (last >= 0 && !wait_ready(last, err)) return false;
        cudaStreamWaitEvent(copy_, D.done, 0);
        for (int q = 0; q < kStageExperts; ++q) {
            const int e = g * kStageExperts + q;
            if (e >= g_.n_expert) break;
            if (on_peer_[(size_t) l][(size_t) e]) continue;
            uint8_t* dst = D.blobs + staged_stride(el) * (size_t) q;
            const ExpertTier::Source src = tier_->locate(l, e);
            if (src.dev) {
                cudaMemcpyAsync(dst, src.dev, el.blob_bytes, cudaMemcpyDeviceToDevice, copy_);
            } else if (src.pinned) {
                cudaMemcpyAsync(dst, src.pinned, el.blob_bytes, cudaMemcpyHostToDevice, copy_);
            } else {
                const int64_t seq = pf_base_[(size_t) l] + pf_idx_[(size_t) l][(size_t) e];
                cudaMemcpyAsync(dst, ring_ + (size_t) (seq % ring_pieces_) * slot_, el.blob_bytes, cudaMemcpyHostToDevice, copy_);
            }
        }
        cudaEventRecord(D.copied, copy_);
        if (last >= 0) {
            cudaEvent_t ev = nullptr;
            if (!pf_events_.empty()) {
                ev = pf_events_.back();
                pf_events_.pop_back();
            } else if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
                err = "deepseek41: cannot create a ring event";
                return false;
            }
            cudaEventRecord(ev, copy_);
            pf_out_.emplace_back(ev, last + 1);
        }
        ++issued_;
        return true;
    }

    // The routed experts of one layer for a long chunk: every group of experts is staged ahead of its products, across
    // layer boundaries, so the bus copies overlap the attention and the products of the layers before. The second card
    // computes its share beside them.
    bool routed_ahead(int il, int T, const float* x, const int32_t* ids, const float* w, float* out, cudaStream_t st,
                      std::string& err) {
        namespace mmq = basalt::prefill::mmq;
        const int K = g_.n_used, NE = g_.n_expert;
        const int G = (NE + kStageExperts - 1) / kStageExperts;
        const ExpertLayer& el = m_.pack().experts[(size_t) il];
        const kernels::NativeExpertLayout L = kernels::native_expert_layout(el.gu_type, el.d_type, g_.n_embd, g_.n_ff_exp);
        if (L.bytes != el.blob_bytes) {
            err = "deepseek41: layer " + std::to_string(il) + " expert blob size does not match the layout";
            return false;
        }
        std::vector<int32_t> hid((size_t) T * K);
        cudaStreamSynchronize(st);
        cudaMemcpy(hid.data(), ids, hid.size() * 4, cudaMemcpyDeviceToHost);
        std::vector<std::vector<int32_t>> by((size_t) NE);   // pair = t * K + slot
        for (int e = 0; e < T * K; ++e) {
            if (hid[(size_t) e] < 0 || hid[(size_t) e] >= NE) {
                err = "deepseek41: router returned a bad expert id";
                return false;
            }
            by[(size_t) hid[(size_t) e]].push_back(e);
        }
        // the main card's rows first, expert by expert (the second card's experts take none here); the second card's block
        // after them
        const std::vector<uint8_t>& away = on_peer_[(size_t) il];
        auto here = [&](int e) { return away[(size_t) e] ? (int64_t) 0 : (int64_t) by[(size_t) e].size(); };
        std::vector<int32_t> tok;
        std::vector<int64_t> start((size_t) NE + 1, 0);
        std::vector<int> peer;
        for (int e = 0; e < NE; ++e) {
            if (away[(size_t) e]) {
                peer.push_back(e);   // with or without rows: its staging may have started already
            } else {
                for (int32_t p : by[(size_t) e]) {
                    map_h_[(size_t) p] = (int32_t) tok.size();
                    tok.push_back(p / K);
                }
            }
            start[(size_t) e + 1] = (int64_t) tok.size();
        }
        cudaMemcpyAsync(tok_all_, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice, st);
        int64_t end = 0;
        if (!peer.empty() && !start_peer(il, T, x, by, peer, (int64_t) tok.size(), end, st, err)) return false;
        if (il + 1 < run_layers_) prepare_peer(il + 1);

        for (int g = 0; g < G; ++g) {
            while (issued_ <= consumed_)
                if (!issue_next(err)) return false;
            Stage& D = dstage_[consumed_ % prefillp::kAheadSlots];
            const int e0 = g * kStageExperts, e1 = std::min(NE, e0 + kStageExperts);
            cudaStreamWaitEvent(st, D.copied, 0);
            // sub-ranges of experts of at most max_rows_ rows (one range unless the routing is very skewed); an expert
            // without rows here inside a range takes no tile
            for (int q0 = e0; q0 < e1;) {
                if (here(q0) == 0) {
                    ++q0;
                    continue;
                }
                int q1 = q0, last = q0;
                int64_t rows = 0, max_rows = 0;
                while (q1 < e1 && (q1 == q0 || rows + here(q1) <= max_rows_)) {
                    rows += here(q1);
                    max_rows = std::max<int64_t>(max_rows, here(q1));
                    if (here(q1) > 0) last = q1;
                    ++q1;
                }
                q1 = last + 1;
                {
                    int32_t b[kStageExperts + 1];
                    for (int q = q0; q <= q1; ++q) b[q - q0] = (int32_t) (start[(size_t) q] - start[(size_t) q0]);
                    cudaMemcpyAsync(bnd_, b, (size_t) (q1 - q0 + 1) * 4, cudaMemcpyHostToDevice, st);
                    const int32_t* tk = tok_all_ + start[(size_t) q0];
                    const int32_t* pr = ident_ + start[(size_t) q0];
                    uint8_t* blobs = D.blobs + staged_stride(el) * (size_t) (q0 - e0);
                    mmq::quantize(x, tk, xq1_, el.gu_type, g_.n_embd, g_.n_embd, rows, st);
                    mmq::Product p;
                    p.w = blobs;
                    p.type = el.gu_type;
                    p.w_rows = 2 * (int64_t) g_.n_ff_exp;
                    p.w_cols = g_.n_embd;
                    p.expert_bytes = staged_stride(el);
                    p.n = q1 - q0;
                    p.xq = xq1_;
                    p.bounds = bnd_;
                    p.ids = ident_;
                    p.total_rows = rows;
                    p.max_rows = max_rows;
                    p.dst = gu_;
                    p.ld_dst = p.w_rows;
                    ctx_.run(p, st);
                    const ExpertTails tails = tails_of(el, blobs, bnd_, q1 - q0);
                    swiglu_split_clamped(gu_, h_, rows, g_.n_ff_exp, g_.clamp_exp[(size_t) il], st, &tails);
                    mmq::quantize(h_, nullptr, xq2_, el.d_type, g_.n_ff_exp, g_.n_ff_exp, rows, st);
                    mmq::Product d;
                    d.w = blobs + L.down_off;
                    d.type = el.d_type;
                    d.w_rows = g_.n_embd;
                    d.w_cols = g_.n_ff_exp;
                    d.expert_bytes = staged_stride(el);
                    d.n = q1 - q0;
                    d.xq = xq2_;
                    d.bounds = bnd_;
                    d.ids = pr;
                    d.total_rows = rows;
                    d.max_rows = max_rows;
                    d.dst = rows_;
                    d.ld_dst = g_.n_embd;
                    ctx_.run(d, st);
                }
                q0 = q1;
            }
            cudaEventRecord(D.done, st);
            ++consumed_;
            while (issued_ < std::min<int64_t>(consumed_ + prefillp::kAheadSlots, (int64_t) run_layers_ * G))
                if (!issue_next(err)) return false;
        }
        if (!peer.empty() && !tier_->peer_share()->join(st, err)) return false;
        cudaMemcpyAsync(map_, map_h_.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, st);
        weighted_sum_tokens(rows_, map_, w, T, K, g_.n_embd, out, st);
        return true;
    }

    void stop_prefetch() {
        if (pf_th_.joinable()) {
            {
                std::lock_guard<std::mutex> lk(pf_mu_);
                pf_stop_ = true;
            }
            pf_cv_.notify_all();
            pf_th_.join();
        }
        pf_on_ = false;
    }

    void prefetch_loop(int nl) {
        for (int l = 0; l < nl; ++l) {
            const ExpertLayer& el = m_.pack().experts[(size_t) l];
            const std::vector<int>& list = pf_list_[(size_t) l];
            for (size_t i0 = 0; i0 < list.size();) {
                const int64_t s0 = pf_base_[(size_t) l] + (int64_t) i0;
                int64_t room = 0;
                {
                    std::unique_lock<std::mutex> lk(pf_mu_);
                    pf_cv_.wait(lk, [&] { return pf_stop_ || s0 < pf_released_ + ring_pieces_; });
                    if (pf_stop_) return;
                    room = pf_released_ + ring_pieces_ - s0;
                }
                const size_t i1 = std::min(list.size(), i0 + (size_t) room);
                auto at = [&](size_t i) { return ring_ + (size_t) ((pf_base_[(size_t) l] + (int64_t) i) % ring_pieces_) * slot_; };
                std::atomic<bool> ok{true};
                std::atomic<size_t> next{i0};
                auto worker = [&] {
                    for (size_t i; (i = next.fetch_add(1)) < i1;) {
                        const uint64_t off = m_.pack().experts_range.offset + el.offset + (uint64_t) list[i] * el.blob_bytes;
                        if (!platform::pread_all(m_.experts_fd(), at(i), el.blob_bytes, off)) ok = false;
                    }
                };
                std::vector<std::thread> th;
                for (int i = 1; i < std::min<int>(kReadThreads, (int) (i1 - i0)); ++i) th.emplace_back(worker);
                worker();
                for (std::thread& t : th) t.join();
                std::lock_guard<std::mutex> lk(pf_mu_);
                if (!ok) pf_failed_ = true;
                pf_ready_ = pf_base_[(size_t) l] + (int64_t) i1;
                pf_cv_.notify_all();
                i0 = i1;
            }
        }
    }

    // y[nc][rows] = x . W^T over bounds {0, nc}
    void dense(const DevTensor* w, int rows, int cols, const void* xq, int nc, float* dst, void* stream) {
        basalt::prefill::mmq::Product p;
        p.w = w->ptr;
        p.type = w->type;
        p.w_rows = rows;
        p.w_cols = cols;
        p.expert_bytes = basalt::prefill::mmq::matrix_bytes(w->type, rows, cols);
        p.n = 1;
        p.xq = xq;
        p.bounds = sbnd_;
        p.ids = ident_;
        p.total_rows = nc;
        p.max_rows = nc;
        p.dst = dst;
        p.ld_dst = rows;
        ctx_.run(p, stream);
    }

    template <class P>
    bool dev(P*& p, size_t bytes) {
        void* v = nullptr;
        if (cudaMalloc(&v, bytes) != cudaSuccess) return false;
        dev_.push_back(v);
        p = (P*) v;
        return true;
    }

    const DsModel& m_;
    const Geometry& g_;
    LayerOps& ops_;
    ExpertTier* tier_ = nullptr;
    bool tier_tried_ = false;
    struct Stage {
        uint8_t *blobs = nullptr, *host = nullptr;
        int32_t *tok = nullptr, *bounds = nullptr;
        cudaEvent_t copied = nullptr, done = nullptr;
    };
    Stage stage_[2];
    cudaStream_t copy_ = nullptr;
    uint64_t launch_ = 0;
    bool ahead_ = false;
    int run_layers_ = 0;
    std::vector<int> layer_tokens_;                 // [layer] the chunk's tokens through it
    std::vector<std::vector<uint8_t>> on_peer_;     // [layer][expert] the second card computes it in a long chunk
    int64_t issued_ = 0, consumed_ = 0;
    Stage dstage_[prefillp::kAheadSlots];
    int32_t *tok_all_ = nullptr, *bnd_ = nullptr;
    // read-ahead of the experts that only the file has
    std::thread pf_th_;
    std::mutex pf_mu_;
    std::condition_variable pf_cv_;
    int64_t pf_ready_ = 0, pf_released_ = 0;   // ring pieces read; pieces whose copies are done
    bool pf_on_ = false, pf_stop_ = false, pf_failed_ = false;
    uint8_t* ring_ = nullptr;                  // the tier's PromptHost ring
    int64_t ring_pieces_ = 0;
    std::deque<std::pair<cudaEvent_t, int64_t>> pf_out_;   // copies out of the ring in flight: event, pieces done after it
    std::vector<cudaEvent_t> pf_events_;                  // events free for reuse
    std::vector<std::vector<int32_t>> pf_idx_;
    std::vector<std::vector<int>> pf_list_;
    std::vector<int64_t> pf_base_;             // [layer] the ring order's first piece of the layer
    int cap_;
    int64_t max_rows_ = 0;               // (token, slot) rows one launch handles
    size_t slot_ = 0;
    std::vector<void*> dev_;
    void *xq1_ = nullptr, *xq2_ = nullptr;
    int32_t *ident_ = nullptr, *sbnd_ = nullptr;
    float *gu_ = nullptr, *h_ = nullptr, *rows_ = nullptr;
    float *hx_ = nullptr, *hrows_ = nullptr;
    int32_t* map_ = nullptr;                 // [pair] the row of the layer's rows that holds it
    std::vector<int32_t> map_h_;
    uint8_t* act_ = nullptr;
    basalt::prefill::mmq::Context ctx_;
};

}  // namespace

std::unique_ptr<PrefillFfn> make_prefill_ffn(const DsModel& model, LayerOps& ops, int max_tokens, std::string& err) {
    auto p = std::make_unique<Impl>(model, ops, max_tokens);
    if (!p->init(err)) return nullptr;
    return p;
}

}  // namespace basalt::models::deepseek41
