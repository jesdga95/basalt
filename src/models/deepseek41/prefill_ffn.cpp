// src/models/deepseek41/prefill_ffn.cpp - the routed and shared experts over a chunk of prompt tokens.
//
// Routed: the (token, slot) pairs of the chunk are grouped by expert; each expert's blob is staged once (copied from the
// VRAM tier or the pinned RAM tier, else read from the file) and runs over all its tokens on the MMQ products (gate/up,
// clamped SwiGLU, down). Staging of one launch overlaps the products of the previous one. In a short chunk the experts the
// bus would carry for a few rows stay where they are: the second card computes its own, the CPU the RAM ones with few rows,
// both beside the bus copies, and their rows come up at the end. Shared: the dense matvec over columns of up to eight tokens.
#include "basalt/log.hpp"
#include "models/deepseek41/prefill_ffn.hpp"

#include "basalt/ggml/ggml.h"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
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
#include <mutex>
#include <cstdio>
#include <numeric>
#include <thread>
#include <vector>

namespace basalt::models::deepseek41 {
namespace {

constexpr int kStageExperts = 32;   // experts staged on the device per launch
constexpr int kReadThreads = 8;     // concurrent blob reads
constexpr int kRing = 5;             // layers of file-read experts held ahead
constexpr int kSharedRows = 1024;   // tokens per shared-expert sub-chunk
constexpr int kCpuRows = 8;         // a RAM expert with this few rows in a short chunk may stay on the CPU
constexpr int kPeerRows = 16;       // a second card's expert with this few rows computes there; a copy through the host costs more

class Impl final : public PrefillFfn {
public:
    Impl(const DsModel& m, LayerOps& ops, int cap) : m_(m), g_(m.geo()), ops_(ops), cap_(cap) {}
    ~Impl() override {
        stop_prefetch();
        if (hx_) cudaFreeHost(hx_);
        if (hrows_) cudaFreeHost(hrows_);
        if (xb_) cudaFreeHost(xb_);
        platform::free_aligned(act_);
        if (ring_) cudaFreeHost(ring_);
        for (cudaEvent_t e : ring_ev_) if (e) cudaEventDestroy(e);
        for (void* p : dev_) cudaFree(p);
        for (Stage& b : stage_) {
            if (b.host) cudaFreeHost(b.host);
            if (b.copied) cudaEventDestroy(b.copied);
            if (b.done) cudaEventDestroy(b.done);
        }
        if (copy_) cudaStreamDestroy(copy_);
    }

    // Bytes between experts in a staging buffer: the layer's blob rounded up to whole blocks of its two formats.
    static uint64_t stride_of(const ExpertLayer& e) {
        uint64_t unit = 1;
        for (int t : {e.gu_type, e.d_type}) {
            const uint64_t bs = ggml_type_size((ggml_type) t);
            unit = unit / std::gcd(unit, bs) * bs;
        }
        return (e.blob_bytes + unit - 1) / unit * unit;
    }

    bool init(std::string& err) {
        namespace mmq = basalt::prefill::mmq;
        // the products step from expert to expert in whole blocks of the layer's two formats
        uint64_t slot = 0;
        for (const ExpertLayer& e : m_.pack().experts) {
            if (!mmq::fits(e.gu_type, 2 * g_.n_ff_exp) || !mmq::fits(e.d_type, g_.n_embd)) {
                err = "deepseek41: the prompt MMQ path does not cover an expert format of this pack";
                return false;
            }
            slot = std::max(slot, stride_of(e));
        }
        slot_ = (size_t) slot;
        const size_t K = (size_t) g_.n_used, n = (size_t) g_.n_embd;
        max_rows_ = prefillp::stage_rows(K, cap_);
        const size_t E = (size_t) cap_ * K, R = (size_t) max_rows_, ff = (size_t) g_.n_ff_exp;
        // the largest q8 buffer either product reads
        bool ok = dev(xq1_, mmq::q8_bytes((int64_t) R, g_.n_embd)) &&
                  dev(xq2_, mmq::q8_bytes((int64_t) R, g_.n_ff_exp));
        ok = ok && dev(gu_, R * 2 * ff * 4) && dev(h_, R * ff * 4);
        ok = ok && dev(ident_, R * 4) && dev(rows_, E * n * 4) && dev(sbnd_, 8 * 4);
        for (Stage& b : stage_)
            ok = ok && dev(b.blobs, slot_ * kStageExperts) && dev(b.tok, R * 4) && dev(b.pair, R * 4) &&
                 dev(b.bounds, (kStageExperts + 1) * 4) &&
                 cudaHostAlloc((void**) &b.host, slot_ * kStageExperts, cudaHostAllocPortable) == cudaSuccess &&
                 cudaEventCreateWithFlags(&b.copied, cudaEventDisableTiming) == cudaSuccess &&
                 cudaEventCreateWithFlags(&b.done, cudaEventDisableTiming) == cudaSuccess;
        ok = ok && cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) == cudaSuccess;
        if (!ok) {
            err = "out of memory for the prefill expert buffers";
            return false;
        }
        mmq::iota(ident_, (int64_t) R, nullptr);
        cudaDeviceSynchronize();
        return true;
    }

    void chunk_layers(int n_layers) override { run_layers_ = n_layers; }

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
        if (tier_ && !plan_share(il, T, x, by, used, share, err)) return false;

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
            bool from_peer = false;
            for (int q = 0; q < nb; ++q) {
                const int e = used[b0 + (size_t) q];
                uint8_t* dst = B.blobs + stride_of(el) * (size_t) q;
                ExpertTier::Source src;
                if (tier_) src = tier_->locate(il, e);
                if (src.dev) {
                    cudaMemcpyAsync(dst, src.dev, L.bytes, cudaMemcpyDeviceToDevice, copy_);
                } else if (src.pinned) {
                    cudaMemcpyAsync(dst, src.pinned, L.bytes, cudaMemcpyHostToDevice, copy_);
                } else if (src.peer) {
                    tier_->peer()->read_slot(src.peer, B.host + slot_ * (size_t) q, L.bytes);
                    from_peer = true;
                } else {
                    from_file.push_back(q);
                }
            }
            if (from_peer) {
                if (!tier_->peer()->read_wait(err)) return false;
                for (int q = 0; q < nb; ++q)
                    if (tier_->locate(il, used[b0 + (size_t) q]).peer)
                        cudaMemcpyAsync(B.blobs + stride_of(el) * (size_t) q, B.host + slot_ * (size_t) q, L.bytes,
                                        cudaMemcpyHostToDevice, copy_);
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
                    cudaMemcpyAsync(B.blobs + stride_of(el) * (size_t) q, B.host + slot_ * (size_t) q, L.bytes,
                                    cudaMemcpyHostToDevice, copy_);
            }
            cudaEventRecord(B.copied, copy_);
            std::vector<int32_t> tok, pair, bounds{0};
            for (int q = 0; q < nb; ++q) {
                for (int32_t p : by[(size_t) used[b0 + (size_t) q]]) {
                    tok.push_back(p / K);
                    pair.push_back(p);
                }
                bounds.push_back((int32_t) tok.size());
            }
            cudaMemcpyAsync(B.tok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(B.pair, pair.data(), pair.size() * 4, cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(B.bounds, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice, st);
            cudaStreamWaitEvent(st, B.copied, 0);

            mmq::quantize(x, B.tok, xq1_, el.gu_type, g_.n_embd, g_.n_embd, rows, stream);
            mmq::Product p;
            p.w = B.blobs;
            p.type = el.gu_type;
            p.w_rows = 2 * (int64_t) g_.n_ff_exp;
            p.w_cols = g_.n_embd;
            p.expert_bytes = stride_of(el);
            p.n = nb;
            p.xq = xq1_;
            p.bounds = B.bounds;
            p.ids = ident_;
            p.total_rows = rows;
            p.max_rows = max_rows;
            p.dst = gu_;
            p.ld_dst = p.w_rows;
            ctx_.run(p, stream);
            swiglu_split_clamped(gu_, h_, rows, g_.n_ff_exp, g_.clamp_exp[(size_t) il], stream);
            mmq::quantize(h_, nullptr, xq2_, el.d_type, g_.n_ff_exp, g_.n_ff_exp, rows, stream);
            mmq::Product d;
            d.w = B.blobs + L.down_off;
            d.type = el.d_type;
            d.w_rows = g_.n_embd;
            d.w_cols = g_.n_ff_exp;
            d.expert_bytes = stride_of(el);
            d.n = nb;
            d.xq = xq2_;
            d.bounds = B.bounds;
            d.ids = B.pair;
            d.total_rows = rows;
            d.max_rows = max_rows;
            d.dst = rows_;
            d.ld_dst = g_.n_embd;
            ctx_.run(d, stream);
            cudaEventRecord(B.done, st);
            b0 += (size_t) nb;
        }
        if (!finish_share(share, stream, err)) return false;
        weighted_sum_tokens(rows_, w, T, K, g_.n_embd, out, stream);
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
    struct PeerRow { int32_t expert, tok; };
    struct Share {
        std::vector<kernels::cpu::ExpertJobMulti> jobs;
        std::vector<int32_t> act_tokens, pair;   // tokens whose activations the CPU needs; each row's (token, slot) pair
        std::vector<PeerRow> peer;
        int cpu_rows = 0;
        std::thread cpu_thread, peer_thread;
        std::string peer_err;
        void join() {
            if (cpu_thread.joinable()) cpu_thread.join();
            if (peer_thread.joinable()) peer_thread.join();
        }
        ~Share() { join(); }
    };

    bool share_buffers(std::string& err) {
        if (hrows_) return true;
        const size_t n = (size_t) g_.n_embd, R = (size_t) prefillp::kShareRows;
        constexpr unsigned kMapped = cudaHostAllocPortable | cudaHostAllocMapped;
        bool ok = dev(drows_, R * n * 4) && dev(dpair_, R * 4) &&
                  cudaHostAlloc((void**) &hx_, (size_t) cap_ * n * 4, cudaHostAllocPortable) == cudaSuccess &&
                  cudaHostAlloc((void**) &hrows_, R * n * 4, kMapped) == cudaSuccess &&
                  cudaHostAlloc((void**) &xb_, (size_t) core::PeerExperts::kRowBatch * n * 4, cudaHostAllocPortable) == cudaSuccess;
        act_ = ok ? (uint8_t*) platform::alloc_aligned(kernels::cpu::kNativeActBytes * (size_t) cap_, 64) : nullptr;
        ok = ok && act_ != nullptr;
        if (ok && tier_->peer()) ok = (peer_rows_ = tier_->peer()->map_rows(hrows_, err)) != nullptr;
        if (!ok && err.empty()) err = "out of memory for a short read's rows";
        return ok;
    }

    // Takes out of `used` what stays on the second card and on the CPU, and starts them; the bus carries the rest.
    bool plan_share(int il, int T, const float* x, const std::vector<std::vector<int32_t>>& by, std::vector<int>& used, Share& s,
                    std::string& err) {
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
        // the second card's
        if (tier_->peer()) {
            for (int e : used) {
                if (!tier_->locate(il, e).peer) continue;
                const int c = (int) by[(size_t) e].size();
                if (c > kPeerRows || rows + c > prefillp::kShareRows) continue;
                for (int32_t p : by[(size_t) e]) {
                    s.peer.push_back({e, p / K});
                    s.pair.push_back(p);
                    ++rows;
                }
                stays[(size_t) e] = 1;
            }
        }
        if (rows == 0) return true;
        used.erase(std::remove_if(used.begin(), used.end(), [&](int e) { return stays[(size_t) e] != 0; }), used.end());
        if (cudaMemcpy(hx_, x, (size_t) T * n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "deepseek41: reading a short read's activations failed";
            return false;
        }
        if (!s.jobs.empty())
            s.cpu_thread = std::thread([this, il, &s] {
                const kernels::cpu::NativeFmt& f = tier_->fmt(il);
                for (size_t i = 0; i < s.act_tokens.size(); ++i)
                    kernels::cpu::native_quant_act(f, hx_ + (size_t) s.act_tokens[i] * g_.n_embd, act_ + i * kernels::cpu::kNativeActBytes);
                tier_->pool()->run_split_multi_native(f, s.jobs.data(), (int) s.jobs.size());
            });
        if (!s.peer.empty())
            s.peer_thread = std::thread([this, il, &s] {
                const ExpertLayer& el = m_.pack().experts[(size_t) il];
                kernels::NativeExpertLayout limited = kernels::native_expert_layout(el.gu_type, el.d_type, g_.n_embd, g_.n_ff_exp);
                limited.swiglu_limit = g_.clamp_exp[(size_t) il];
                core::PeerExperts* peer = tier_->peer();
                const size_t n_rows = s.peer.size();
                int32_t ids[core::PeerExperts::kRowBatch];
                for (size_t r0 = 0; r0 < n_rows; r0 += (size_t) core::PeerExperts::kRowBatch) {
                    const size_t nb = std::min<size_t>(n_rows - r0, (size_t) core::PeerExperts::kRowBatch);
                    for (size_t i = 0; i < nb; ++i) {
                        ids[i] = s.peer[r0 + i].expert;
                        std::copy_n(hx_ + (size_t) s.peer[r0 + i].tok * g_.n_embd, g_.n_embd, xb_ + i * g_.n_embd);
                    }
                    float* out = peer_rows_ + ((size_t) s.cpu_rows + r0) * g_.n_embd;
                    if (!peer->launch_rows(il, limited, xb_, ids, (int64_t) nb, out, s.peer_err) || !peer->finish(nullptr, s.peer_err)) return;
                }
            });
        return true;
    }

    // The rows of the CPU and the second card join the others'.
    bool finish_share(Share& s, void* stream, std::string& err) {
        s.join();
        if (!s.peer_err.empty()) {
            err = s.peer_err;
            return false;
        }
        if (s.pair.empty()) return true;
        cudaStream_t st = (cudaStream_t) stream;
        const size_t rows = s.pair.size(), n = (size_t) g_.n_embd;
        cudaMemcpyAsync(drows_, hrows_, rows * n * 4, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(dpair_, s.pair.data(), rows * 4, cudaMemcpyHostToDevice, st);
        kernels::scatter_rows_f32(drows_, rows_, dpair_, (int64_t) rows, (int64_t) n, stream);
        return true;
    }

    // Brings the experts the main card and RAM do not hold, layer by layer a few layers ahead of the products, into a
    // pinned ring: from the second card, or read from the file.
    bool start_ahead() {
        const int nl = g_.n_layers;
        pf_idx_.assign((size_t) nl, std::vector<int32_t>((size_t) g_.n_expert, -1));
        pf_list_.assign((size_t) nl, {});
        size_t mx = 0;
        for (int l = 0; l < nl; ++l) {
            for (int e = 0; e < g_.n_expert; ++e) {
                const ExpertTier::Source src = tier_->locate(l, e);
                if (src.dev || src.pinned) continue;
                pf_idx_[(size_t) l][(size_t) e] = (int32_t) pf_list_[(size_t) l].size();
                pf_list_[(size_t) l].push_back(e);
            }
            mx = std::max(mx, pf_list_[(size_t) l].size());
        }
        pf_on_ = false;
        if (mx > 0 && (!ring_ || mx > pf_max_)) {
            if (ring_) cudaFreeHost(ring_);
            ring_ = nullptr;
            pf_max_ = mx;
            if (cudaHostAlloc((void**) &ring_, (size_t) kRing * pf_max_ * slot_, cudaHostAllocPortable) != cudaSuccess) {
                ring_ = nullptr;
                return false;
            }
        }
        if (!ahead_buffers()) return false;
        for (cudaEvent_t& e : ring_ev_)
            if (!e) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        pf_ready_ = 0;
        pf_released_ = kRing - 1;
        pf_failed_ = false;
        pf_stop_ = false;
        cudaStreamSynchronize(copy_);
        issued_ = consumed_ = 0;
        if (mx > 0) {
            pf_on_ = true;
            pf_th_ = std::thread([this] { prefetch_loop(run_layers_); });
        }
        return true;
    }

    bool ahead_buffers() {
        if (!dstage_[0].blobs) {
            const size_t E = (size_t) cap_ * g_.n_used;
            if (!dev(tok_all_, E * 4) || !dev(pair_all_, E * 4) || !dev(bnd_, (kStageExperts + 1) * 4)) return false;
            for (Stage& b : dstage_) {
                if (!dev(b.blobs, slot_ * kStageExperts) ||
                    cudaEventCreateWithFlags(&b.copied, cudaEventDisableTiming) != cudaSuccess ||
                    cudaEventCreateWithFlags(&b.done, cudaEventDisableTiming) != cudaSuccess)
                    return false;
            }
        }
        return true;
    }

    // Starts the copies of the next group of experts (in layer order) into its slot; they depend on no routing.
    bool issue_next(std::string& err) {
        const int G = (g_.n_expert + kStageExperts - 1) / kStageExperts;
        const int l = (int) (issued_ / G), g = (int) (issued_ % G);
        Stage& D = dstage_[issued_ % prefillp::kAheadSlots];
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        if (g == 0 && pf_on_) {
            if (l >= 2) {
                cudaEventSynchronize(ring_ev_[(l - 2) % kRing]);
                std::lock_guard<std::mutex> lk(pf_mu_);
                pf_released_ = l - 2 + kRing;
                pf_cv_.notify_all();
            }
            std::unique_lock<std::mutex> lk(pf_mu_);
            pf_cv_.wait(lk, [&] { return pf_ready_ > l || pf_failed_; });
            if (pf_failed_) {
                err = "deepseek41: short read of an expert blob";
                return false;
            }
        }
        cudaStreamWaitEvent(copy_, D.done, 0);
        for (int q = 0; q < kStageExperts; ++q) {
            const int e = g * kStageExperts + q;
            if (e >= g_.n_expert) break;
            uint8_t* dst = D.blobs + stride_of(el) * (size_t) q;
            const ExpertTier::Source src = tier_->locate(l, e);
            if (src.dev) {
                cudaMemcpyAsync(dst, src.dev, el.blob_bytes, cudaMemcpyDeviceToDevice, copy_);
            } else if (src.pinned) {
                cudaMemcpyAsync(dst, src.pinned, el.blob_bytes, cudaMemcpyHostToDevice, copy_);
            } else {
                cudaMemcpyAsync(dst, ring_ + ((size_t) (l % kRing) * pf_max_ + (size_t) pf_idx_[(size_t) l][(size_t) e]) * slot_,
                                el.blob_bytes, cudaMemcpyHostToDevice, copy_);
            }
        }
        cudaEventRecord(D.copied, copy_);
        if (g == G - 1 && pf_on_) cudaEventRecord(ring_ev_[l % kRing], copy_);
        ++issued_;
        return true;
    }

    // The routed experts of one layer for a long chunk: every group of experts is staged ahead of its products, across
    // layer boundaries, so the bus copies overlap the attention and the products of the layers before.
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
        std::vector<int32_t> tok, pair;
        std::vector<int64_t> start((size_t) NE + 1, 0);
        for (int e = 0; e < NE; ++e) {
            for (int32_t p : by[(size_t) e]) {
                tok.push_back(p / K);
                pair.push_back(p);
            }
            start[(size_t) e + 1] = (int64_t) tok.size();
        }
        cudaMemcpyAsync(tok_all_, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(pair_all_, pair.data(), pair.size() * 4, cudaMemcpyHostToDevice, st);

        for (int g = 0; g < G; ++g) {
            while (issued_ <= consumed_)
                if (!issue_next(err)) return false;
            Stage& D = dstage_[consumed_ % prefillp::kAheadSlots];
            const int e0 = g * kStageExperts, e1 = std::min(NE, e0 + kStageExperts);
            cudaStreamWaitEvent(st, D.copied, 0);
            // sub-ranges of experts of at most max_rows_ rows (one range unless the routing is very skewed)
            for (int q0 = e0; q0 < e1;) {
                if (by[(size_t) q0].empty()) {   // runs of experts that have rows
                    ++q0;
                    continue;
                }
                int q1 = q0;
                int64_t rows = 0, max_rows = 0;
                while (q1 < e1 && !by[(size_t) q1].empty() && (q1 == q0 || rows + (int64_t) by[(size_t) q1].size() <= max_rows_)) {
                    rows += (int64_t) by[(size_t) q1].size();
                    max_rows = std::max<int64_t>(max_rows, (int64_t) by[(size_t) q1].size());
                    ++q1;
                }
                {
                    int32_t b[kStageExperts + 1];
                    for (int q = q0; q <= q1; ++q) b[q - q0] = (int32_t) (start[(size_t) q] - start[(size_t) q0]);
                    cudaMemcpyAsync(bnd_, b, (size_t) (q1 - q0 + 1) * 4, cudaMemcpyHostToDevice, st);
                    const int32_t* tk = tok_all_ + start[(size_t) q0];
                    const int32_t* pr = pair_all_ + start[(size_t) q0];
                    uint8_t* blobs = D.blobs + stride_of(el) * (size_t) (q0 - e0);
                    mmq::quantize(x, tk, xq1_, el.gu_type, g_.n_embd, g_.n_embd, rows, st);
                    mmq::Product p;
                    p.w = blobs;
                    p.type = el.gu_type;
                    p.w_rows = 2 * (int64_t) g_.n_ff_exp;
                    p.w_cols = g_.n_embd;
                    p.expert_bytes = stride_of(el);
                    p.n = q1 - q0;
                    p.xq = xq1_;
                    p.bounds = bnd_;
                    p.ids = ident_;
                    p.total_rows = rows;
                    p.max_rows = max_rows;
                    p.dst = gu_;
                    p.ld_dst = p.w_rows;
                    ctx_.run(p, st);
                    swiglu_split_clamped(gu_, h_, rows, g_.n_ff_exp, g_.clamp_exp[(size_t) il], st);
                    mmq::quantize(h_, nullptr, xq2_, el.d_type, g_.n_ff_exp, g_.n_ff_exp, rows, st);
                    mmq::Product d;
                    d.w = blobs + L.down_off;
                    d.type = el.d_type;
                    d.w_rows = g_.n_embd;
                    d.w_cols = g_.n_ff_exp;
                    d.expert_bytes = stride_of(el);
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
        weighted_sum_tokens(rows_, w, T, K, g_.n_embd, out, st);
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
            {
                std::unique_lock<std::mutex> lk(pf_mu_);
                pf_cv_.wait(lk, [&] { return pf_stop_ || l <= pf_released_; });
                if (pf_stop_) return;
            }
            const ExpertLayer& el = m_.pack().experts[(size_t) l];
            const std::vector<int>& list = pf_list_[(size_t) l];
            uint8_t* base = ring_ + (size_t) (l % kRing) * pf_max_ * slot_;
            std::atomic<bool> ok{true};
            std::vector<size_t> files;
            bool any_peer = false;
            for (size_t i = 0; i < list.size(); ++i) {
                const ExpertTier::Source src = tier_->locate(l, list[i]);
                if (src.peer) {
                    tier_->peer()->read_slot(src.peer, base + i * slot_, el.blob_bytes);
                    any_peer = true;
                } else {
                    files.push_back(i);
                }
            }
            std::atomic<size_t> next{0};
            auto worker = [&] {
                for (size_t n; (n = next.fetch_add(1)) < files.size();) {
                    const size_t i = files[n];
                    const uint64_t off = m_.pack().experts_range.offset + el.offset + (uint64_t) list[i] * el.blob_bytes;
                    if (!platform::pread_all(m_.experts_fd(), base + i * slot_, el.blob_bytes, off)) ok = false;
                }
            };
            std::vector<std::thread> th;
            for (int i = 1; i < std::min<int>(kReadThreads, (int) files.size()); ++i) th.emplace_back(worker);
            worker();
            for (std::thread& t : th) t.join();
            if (any_peer) {
                std::string e;
                if (!tier_->peer()->read_wait(e)) ok = false;
            }
            std::lock_guard<std::mutex> lk(pf_mu_);
            if (!ok) pf_failed_ = true;
            pf_ready_ = l + 1;
            pf_cv_.notify_all();
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
        int32_t *tok = nullptr, *pair = nullptr, *bounds = nullptr;
        cudaEvent_t copied = nullptr, done = nullptr;
    };
    Stage stage_[2];
    cudaStream_t copy_ = nullptr;
    uint64_t launch_ = 0;
    bool ahead_ = false;
    int run_layers_ = 0;
    int64_t issued_ = 0, consumed_ = 0;
    Stage dstage_[prefillp::kAheadSlots];
    int32_t *tok_all_ = nullptr, *pair_all_ = nullptr, *bnd_ = nullptr;
    // read-ahead of the experts that only the file has
    std::thread pf_th_;
    std::mutex pf_mu_;
    std::condition_variable pf_cv_;
    int pf_ready_ = 0, pf_released_ = 0;
    bool pf_on_ = false, pf_stop_ = false, pf_failed_ = false;
    size_t pf_max_ = 0;
    uint8_t* ring_ = nullptr;
    cudaEvent_t ring_ev_[kRing] = {};
    std::vector<std::vector<int32_t>> pf_idx_;
    std::vector<std::vector<int>> pf_list_;
    int cap_;
    int64_t max_rows_ = 0;               // (token, slot) rows one launch handles
    size_t slot_ = 0;
    std::vector<void*> dev_;
    void *xq1_ = nullptr, *xq2_ = nullptr;
    int32_t *ident_ = nullptr, *sbnd_ = nullptr;
    float *gu_ = nullptr, *h_ = nullptr, *rows_ = nullptr;
    float *hx_ = nullptr, *hrows_ = nullptr, *peer_rows_ = nullptr, *xb_ = nullptr, *drows_ = nullptr;
    int32_t* dpair_ = nullptr;
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
