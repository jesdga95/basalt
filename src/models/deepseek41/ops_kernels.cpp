// src/models/deepseek41/ops_kernels.cpp - the layer ops built from the kernel groups' entry points: attention
// (window ring, compressor, indexer, query path, sparse MQA, grouped output) with per-layer state owned here.
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/models/deepseek41/attn.hpp"
#include "basalt/models/deepseek41/attn_index.hpp"
#include "models/deepseek41/attn_batch.hpp"
#include "models/deepseek41/elastic.hpp"
#include "models/deepseek41/layer_ops.hpp"
#include "models/deepseek41/model.hpp"
#include "models/deepseek41/prefill_plan.hpp"
#include "models/deepseek41/ops.hpp"
#include "models/deepseek41/tier.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <vector>

namespace basalt::models::deepseek41 {
namespace {

constexpr uint64_t kTierReserve = 6ull << 30;   // device memory left free beside the expert tier: caches and prompt workspaces
constexpr uint64_t kModuleRoom = 7ull << 27;    // kernel modules still to load when the tier opens
constexpr uint64_t kVisionRoom = 3ull << 29;    // the image encoder, when the pack carries one
// One layer's GPU plan for the grouped expert kernel, uploaded with a single copy.
struct ExpertPlan {
    unsigned long long ptr[8];
    int32_t start[16];
    int32_t ng[4];
    int32_t dst[8];
    int32_t tok[8];
};

// A window's plan: the distinct experts of its rows, each with the rows it computes.
struct WindowPlan {
    unsigned long long ptr[ExpertTier::kMaxK];
    int32_t start[ExpertTier::kMaxK + 2];
    int32_t ng[4];
    int32_t dst[ExpertTier::kMaxK];
    int32_t tok[ExpertTier::kMaxK];
};

// Layer roles: the kv source whose compressed rows a layer reads, and the latest index source at or below it.
int kv_source(int il) {
    if (il < 2) return -1;
    if (il < 8) return 2;
    if (il < 14) return 8;
    if (il < 20) return 14;
    return 20;
}
int index_source(int il) {
    int s = -1;
    for (int c : {2, 8, 14, 20, 24, 28, 32, 36})
        if (c <= il) s = c;
    return s;
}
IndexMode index_mode(int il) { return il == 20 ? IndexMode::Level1 : il > 20 ? IndexMode::Level2 : IndexMode::Full; }

class KernelOps final : public LayerOps {
public:
    explicit KernelOps(const DsModel& m) : m_(m), g_(m.geo()), ctx_((int) m.context()) {}
    ~KernelOps() override {
        for (cudaGraphExec_t e : share_g_)
            if (e) cudaGraphExecDestroy(e);
        for (void* p : allocs_) cudaFree(p);
    }

    bool init(std::string& err) {
        const int nl = g_.n_layers;
        ring_.assign((size_t) nl, nullptr);
        comp_kv_.assign((size_t) nl, nullptr);
        comp_ik_.assign((size_t) nl, nullptr);
        el_kv_.resize((size_t) nl);
        el_ik_.resize((size_t) nl);
        st_kv_.assign((size_t) nl, nullptr);
        st_sc_.assign((size_t) nl, nullptr);
        cws_l_.assign((size_t) nl, nullptr);
        for (int il = 0; il < nl; ++il) {
            if (!alloc((void**) &ring_[il], (size_t) kWindowSlots * kLatent * 2)) return oom(err);
            if (g_.compress_ratio[(size_t) il] && kv_source(il) == il) {
                const int ratio = g_.compress_ratio[(size_t) il];
                // the caches reserve the address range of the whole context and are backed as positions arrive
                const size_t cap = (size_t) ctx_ / ratio + 1;
                el_kv_[il] = std::make_unique<ElasticBuf>();
                el_ik_[il] = std::make_unique<ElasticBuf>();
                if (!el_kv_[il]->open(cap * kLatent * 2, err) || !el_ik_[il]->open(cap * kIdxDim * 2, err)) return false;
                comp_kv_[il] = (uint16_t*) el_kv_[il]->data();
                comp_ik_[il] = (uint16_t*) el_ik_[il]->data();
                if (!alloc((void**) &st_kv_[il], (size_t) ratio * kLatent * 4) ||
                    !alloc((void**) &st_sc_[il], (size_t) ratio * kLatent * 4) || !alloc(&cws_l_[il], compress_workspace_bytes()))
                    return oom(err);
            }
        }
        float tab[kRopePairs];
        rope_freq_table(RopeParams{}, tab);
        if (!alloc((void**) &freq_[0], sizeof(tab))) return oom(err);
        cudaMemcpy(freq_[0], tab, sizeof(tab), cudaMemcpyHostToDevice);
        rope_freq_table(RopeParams{g_.comp_rope_base, 1.f / g_.yarn_factor, 1.f, g_.yarn_beta_fast, g_.yarn_beta_slow,
                                   g_.yarn_orig_ctx}, tab);
        if (!alloc((void**) &freq_[1], sizeof(tab))) return oom(err);
        cudaMemcpy(freq_[1], tab, sizeof(tab), cudaMemcpyHostToDevice);
        rope_[0] = rope_table_window();
        rope_[1] = rope_table_compressed(g_.comp_rope_base, g_.yarn_factor, g_.yarn_beta_fast, g_.yarn_beta_slow, g_.yarn_orig_ctx);

        const size_t xq = std::max<size_t>({kernels::native_q8_1_bytes(kAttnEmbd, kMaxT), attn_out_scratch_bytes(kMaxT), 1 << 16});
        const size_t ws_f = sparse_mqa_workspace_floats(kMaxT, kTopK) + 64;
        const size_t max_rows = (size_t) ctx_ + 1;
        if (!alloc(&xq_, xq) || !alloc(&xq2_, std::max<size_t>(q_path_scratch_bytes(kMaxT), 1 << 16)) ||
            !alloc((void**) &kvraw_, (size_t) kMaxT * kLatent * 4) || !alloc((void**) &qr_, (size_t) kMaxT * kAttnQRank * 4) ||
            !alloc((void**) &q_, (size_t) kMaxT * kAttnHeads * kAttnHeadDim * 4) ||
            !alloc((void**) &attn_, (size_t) kMaxT * kAttnHeads * kAttnHeadDim * 4) ||
            !alloc((void**) &oa_, attn_out_scratch_floats(kMaxT) * 4) || !alloc((void**) &mws_, ws_f * 4) ||
            !alloc(&cws_, compress_workspace_bytes()) || !alloc(&iws_, indexer_workspace_bytes((int) max_rows)) ||
            !alloc((void**) &dpos_, (size_t) nl * kPosStride * 4) || !alloc((void**) &ids_, (size_t) kMaxT * kTopK * 4) ||
            !alloc((void**) &idn_, (size_t) kMaxT * 4) || !alloc((void**) &cand_, (size_t) kMaxT * kMaxCandBlocks * 4) ||
            !alloc((void**) &cand_n_, (size_t) kMaxT * 4))
            return oom(err);
        if (cudaHostAlloc((void**) &h_pos_, (size_t) nl * kPosStride * 4, cudaHostAllocPortable) != cudaSuccess) return oom(err);
        const int nf = g_.n_ff_exp * std::max(g_.n_shared, 1);
        if (!alloc((void**) &sh_g_, (size_t) nf * kMaxT * 4) || !alloc((void**) &sh_u_, (size_t) nf * kMaxT * 4) ||
            !alloc((void**) &sh_h_, (size_t) nf * kMaxT * 4) || !alloc(&sh_q1_, kernels::native_q8_1_bytes(kAttnEmbd, kMaxT)) ||
            !alloc(&sh_q2_, kernels::native_q8_1_bytes(nf, kMaxT)))
            return oom(err);
        return true;
    }

    // each kv-source layer's latents and index keys, in layer order
    template <class Fn>
    void each_cache(int64_t positions, Fn fn) const {
        for (int il = 0; il < g_.n_layers; ++il) {
            if (!el_kv_[(size_t) il]) continue;
            const uint64_t rows = (uint64_t) positions / (uint64_t) g_.compress_ratio[(size_t) il] + 1;
            fn(el_kv_[(size_t) il]->data(), rows * ctxp::kLatentRowBytes);
            fn(el_ik_[(size_t) il]->data(), rows * ctxp::kIndexRowBytes);
        }
    }

    size_t cache_bytes(int64_t positions) const override {
        size_t total = 0;
        each_cache(positions, [&](uint8_t*, uint64_t bytes) { total += bytes; });
        return total;
    }

    // Each kv-source layer as a K/V entry: its latents and index keys by row, a row being `ratio` cells.
    bool save_cache(void* stream, int64_t positions, std::vector<core::ConversationKv>& kv, std::string& err) override {
        cudaStreamSynchronize((cudaStream_t) stream);
        kv.clear();
        for (int il = 0; il < g_.n_layers; ++il) {
            if (!el_kv_[(size_t) il]) continue;
            const int64_t ratio = g_.compress_ratio[(size_t) il];
            if (ctxp::kLatentRowBytes % (uint64_t) ratio || ctxp::kIndexRowBytes % (uint64_t) ratio) {
                err = "deepseek41: compress ratio " + std::to_string(ratio) + " does not divide a cache row";
                return false;
            }
            const uint64_t rows = ctxp::cache_rows(positions, (int) ratio);
            core::ConversationKv k;
            k.page_size = ratio;
            k.cells = (int64_t) rows * ratio;
            k.k.resize_for_overwrite(rows * ctxp::kLatentRowBytes);
            k.v.resize_for_overwrite(rows * ctxp::kIndexRowBytes);
            const auto down = [&](const uint8_t* dev, core::ConversationBuffer& to) {
                return to.visit(0, to.size(), [&](uint8_t* p, size_t n, size_t at) {
                    return cudaMemcpy(p, dev + at, n, cudaMemcpyDeviceToHost) == cudaSuccess;
                });
            };
            if (!down(el_kv_[(size_t) il]->data(), k.k) || !down(el_ik_[(size_t) il]->data(), k.v)) {
                err = "deepseek41: reading a cache back failed";
                return false;
            }
            kv.push_back(std::move(k));
        }
        return true;
    }

    bool restore_cache(void* stream, const std::vector<core::ConversationKv>& kv, std::string& err) override {
        cudaStreamSynchronize((cudaStream_t) stream);
        size_t i = 0;
        for (int il = 0; il < g_.n_layers; ++il) {
            if (!el_kv_[(size_t) il]) continue;
            if (i >= kv.size() || kv[i].page_size != g_.compress_ratio[(size_t) il]) {
                err = "deepseek41: a parked conversation has other caches";
                return false;
            }
            const auto up = [&](uint8_t* dev, const core::ConversationBuffer& from) {
                return from.visit(0, from.size(), [&](const uint8_t* p, size_t n, size_t at) {
                    return cudaMemcpy(dev + at, p, n, cudaMemcpyHostToDevice) == cudaSuccess;
                });
            };
            if (!up(el_kv_[(size_t) il]->data(), kv[i].k) || !up(el_ik_[(size_t) il]->data(), kv[i].v)) {
                err = "deepseek41: writing a cache back failed";
                return false;
            }
            ++i;
        }
        return true;
    }

    size_t turn_bytes() const override {
        const size_t ring_b = (size_t) kWindowSlots * kLatent * 2;
        size_t total = 0;
        for (size_t il = 0; il < ring_.size(); ++il) {
            if (ring_[il]) total += ring_b;
            if (st_kv_[il]) total += 2 * (size_t) g_.compress_ratio[il] * kLatent * 4;
        }
        return total;
    }

    void save_turn(void* stream, uint8_t* host) override {
        cudaStreamSynchronize((cudaStream_t) stream);
        const size_t ring_b = (size_t) kWindowSlots * kLatent * 2;
        size_t at = 0;
        auto put = [&](const void* dev, size_t bytes) {
            cudaMemcpy(host + at, dev, bytes, cudaMemcpyDeviceToHost);
            at += bytes;
        };
        for (size_t il = 0; il < ring_.size(); ++il) {
            if (ring_[il]) put(ring_[il], ring_b);
            if (st_kv_[il]) {
                const size_t n = (size_t) g_.compress_ratio[il] * kLatent * 4;
                put(st_kv_[il], n);
                put(st_sc_[il], n);
            }
        }
    }

    void restore_turn(void* stream, const uint8_t* host) override {
        cudaStreamSynchronize((cudaStream_t) stream);
        win_n_ = 0;
        const size_t ring_b = (size_t) kWindowSlots * kLatent * 2;
        size_t at = 0;
        auto get = [&](void* dev, size_t bytes) {
            cudaMemcpy(dev, host + at, bytes, cudaMemcpyHostToDevice);
            at += bytes;
        };
        for (size_t il = 0; il < ring_.size(); ++il) {
            if (ring_[il]) get(ring_[il], ring_b);
            if (st_kv_[il]) {
                const size_t n = (size_t) g_.compress_ratio[il] * kLatent * 4;
                get(st_kv_[il], n);
                get(st_sc_[il], n);
            }
        }
    }

    void reset() override {
        for (size_t il = 0; il < st_kv_.size(); ++il) {
            if (!st_kv_[il]) continue;
            const size_t n = (size_t) g_.compress_ratio[il] * kLatent * 4;
            cudaMemset(st_kv_[il], 0, n);
            cudaMemset(st_sc_[il], 0, n);
        }
        win_n_ = 0;
        cudaMemset(ids_, 0xff, (size_t) kMaxT * kTopK * 4);
    }

    void begin_token(int64_t, int n_tokens) override {
        if (tier_) tier_->begin_token(n_tokens);
    }

    bool attention_batch(int il, int64_t pos0, int n_tokens, const int32_t*, const float* x, float* out, int,
                         void* stream, std::string& err) override {
        if (!ensure_positions(pos0 + n_tokens, err) || !ensure_batch(err)) return false;
        return batch_->run(il, pos0, n_tokens, x, out, stream, err);
    }

    bool ensure_batch(std::string& err) {
        if (batch_) return true;
        AttnCaches c;
        c.ring = &ring_;
        c.comp_kv = &comp_kv_;
        c.comp_ik = &comp_ik_;
        c.st_kv = &st_kv_;
        c.st_sc = &st_sc_;
        c.freq[0] = freq_[0];
        c.freq[1] = freq_[1];
        c.rope = rope_;
        batch_ = make_attn_batch(m_, c, kMaxChunk, ctx_, err);
        return batch_ != nullptr;
    }

    bool compress_batch(int il, int64_t pos0, int n_tokens, const float* x, void* stream, std::string& err) override {
        return ensure_positions(pos0 + n_tokens, err) && ensure_batch(err) && batch_->compress_only(il, pos0, n_tokens, x, stream, err);
    }

    bool stage_token(int64_t pos, int T, void* stream, std::string& err) override {
        if (pos + T > ctx_) {
            err = "deepseek41: the position exceeds the context (" + std::to_string(ctx_) + " tokens)";
            return false;
        }
        if (!ensure_positions(pos + T, err)) return false;
        if (!ensure_tier(err)) return false;
        commit_window(pos, stream);
        if (T > 1) {
            win_pos0_ = pos;
            win_n_ = T;
        }
        return stage_positions(pos, T, stream);
    }

    void commit_window(int64_t next_pos, void* stream) override {
        if (!win_n_) return;
        const int q = (int) std::clamp<int64_t>(next_pos - win_pos0_, 0, win_n_);
        win_n_ = 0;
        for (int il = 0; il < g_.n_layers; ++il)
            if (g_.compress_ratio[(size_t) il] && kv_source(il) == il)
                compress_window_commit(compress_args(il), dpos_ + il * kPosStride, q, cws_l_[il], stream);
    }

    bool attention_window(int il, int T, const float* x, float* out, void* stream, std::string& err) override {
        if (T > staged_T_) {
            err = "deepseek41: window attention called for a window that was not staged";
            return false;
        }
        return attn_impl(il, T, x, out, stream, err);
    }

    bool shared_expert_window(int il, int T, const float* x, float* out, void* stream, std::string& err) override {
        return shared_impl(il, T, x, out, stream, err);
    }
    bool graph_capable() const override { return true; }

    // Backs the compressed caches up to `positions`. When the card is full the expert tier gives back its loan of
    // spare slots and the mapping is tried again.
    bool ensure_positions(int64_t positions, std::string& err) override {
        if (positions <= backed_positions_) return true;
        if (positions > ctx_) {
            err = "deepseek41: the position exceeds the context (" + std::to_string(ctx_) + " tokens)";
            return false;
        }
        for (int attempt = 0; attempt < 2; ++attempt) {
            std::string e;
            bool ok = true;
            const uint64_t keep = tier_ ? tier_->free_floor() : 0;
            for (int il = 0; il < g_.n_layers && ok; ++il) {
                if (!el_kv_[(size_t) il]) continue;
                const uint64_t rows = (uint64_t) positions / (uint64_t) g_.compress_ratio[(size_t) il] + 1;
                ok = el_kv_[(size_t) il]->ensure(rows * kLatent * 2, keep, e) && el_ik_[(size_t) il]->ensure(rows * kIdxDim * 2, keep, e);
            }
            if (ok) {
                backed_positions_ = positions;
                return true;
            }
            if (attempt == 0 && tier_) {
                cudaDeviceSynchronize();
                tier_->shrink();
                continue;
            }
            err = "deepseek41: not enough free device memory for a context of " + std::to_string(positions) +
                  " tokens (" + e + "); a smaller --max-context leaves more for the expert tier";
            return false;
        }
        return false;
    }

    void trim_positions() override {
        std::vector<int> ratios;
        for (int il = 0; il < g_.n_layers; ++il)
            if (el_kv_[(size_t) il]) ratios.push_back(g_.compress_ratio[(size_t) il]);
        const int64_t keep = prefillp::keep_positions(backed_positions_, ratios);
        if (keep >= backed_positions_) return;
        for (int il = 0; il < g_.n_layers; ++il) {
            if (!el_kv_[(size_t) il]) continue;
            const uint64_t rows = (uint64_t) keep / (uint64_t) g_.compress_ratio[(size_t) il] + 1;
            el_kv_[(size_t) il]->trim(rows * kLatent * 2);
            el_ik_[(size_t) il]->trim(rows * kIdxDim * 2);
        }
        backed_positions_ = keep;
    }

    bool stage_positions(int64_t pos0, int T, void* stream) {
        for (int il = 0; il < g_.n_layers; ++il) {
            const int ratio = g_.compress_ratio[(size_t) il];
            int32_t* h = h_pos_ + il * kPosStride;
            for (int t = 0; t < T; ++t) {
                const int32_t p32 = (int32_t) (pos0 + t);
                const int32_t start = std::max<int32_t>(0, p32 - (kWindow - 1));
                h[t] = p32;
                h[kMaxT + t] = start;
                h[2 * kMaxT + t] = p32 - start + 1;
                h[3 * kMaxT + t] = ratio ? (p32 + 1) / ratio : 0;
            }
        }
        cudaMemcpyAsync(dpos_, h_pos_, (size_t) g_.n_layers * kPosStride * 4, cudaMemcpyHostToDevice, (cudaStream_t) stream);
        staged_pos_ = pos0;
        staged_T_ = T;
        return true;
    }




    bool attention(int il, int64_t pos, int32_t token, const float* x, float* out, void* stream,
                   std::string& err) override {
        (void) token;
        if (staged_pos_ != pos || staged_T_ != 1) {
            err = "deepseek41: attention called for a position that was not staged";
            return false;
        }
        return attn_impl(il, 1, x, out, stream, err);
    }


    CompressArgs compress_args(int il) {
        const int ratio = g_.compress_ratio[(size_t) il];
        auto T = [&](const char* s) { return m_.tensor(il, s); };
        const DevTensor *wc = T("attn_compressor_kv.weight"), *cn = T("attn_compressor_norm.weight");
        const DevTensor *wg = ratio == 2 ? T("attn_compressor_gate.weight") : nullptr;
        const DevTensor *ik = T("indexer.attn_k.weight"), *ikn = T("indexer.k_norm.weight");
        CompressArgs ca;
        ca.ratio = ratio;
        ca.rms_eps = g_.rms_eps;
        ca.n_embd = kAttnEmbd;
        ca.w_kv = wc ? (const uint16_t*) wc->ptr : nullptr;
        ca.w_gate = wg ? (const uint16_t*) wg->ptr : nullptr;
        ca.norm_w = cn ? (const float*) cn->ptr : nullptr;
        ca.w_idx_k = ik ? (const uint16_t*) ik->ptr : nullptr;
        ca.idx_k_norm_w = ikn ? (const float*) ikn->ptr : nullptr;
        ca.rope_freq = freq_[1];
        ca.state_kv = st_kv_[il];
        ca.state_score = st_sc_[il];
        ca.comp_kv = comp_kv_[il];
        ca.comp_ik = comp_ik_[il];
        return ca;
    }

    // Attention of T staged tokens (T == 1: the single-token decode; T > 1 or a window: the verify pass).
    bool attn_impl(int il, int T, const float* x, float* out, void* stream, std::string& err) {
        const int ratio = g_.compress_ratio[(size_t) il];
        const int kind = ratio ? 1 : 0;
        auto W = [&](const char* s) { return m_.tensor(il, s); };
        const DevTensor *wkv = W("attn_kv.weight"), *kvn = W("attn_kv_a_norm.weight"), *qa = W("attn_q_a.weight"),
                        *qan = W("attn_q_a_norm.weight"), *qb = W("attn_q_b.weight"), *oa = W("attn_output_a.weight"),
                        *ob = W("attn_output_b.weight"), *sk = W("attn_sinks.weight");
        if (!wkv || !kvn || !qa || !qan || !qb || !oa || !ob || !sk) {
            err = "deepseek41: layer " + std::to_string(il) + " lacks attention tensors";
            return false;
        }

        const int32_t *d_pos = dpos_ + il * kPosStride, *d_start = d_pos + kMaxT, *d_n = d_pos + 2 * kMaxT,
                      *d_lim = d_pos + 3 * kMaxT;

        // window latent
        kernels::native_quantize_q8_1(x, xq_, kAttnEmbd, T, stream);
        kernels::native_mmvq(wkv->type, wkv->ptr, xq_, kvraw_, kAttnEmbd, kLatent, T, stream);
        kv_window(kvraw_, (const float*) kvn->ptr, d_pos, T, g_.rms_eps, freq_[kind], nullptr, ring_[il], stream);

        // query path (also yields the indexer's input)
        QPathWeights qw{qa->ptr, qa->type, (const float*) qan->ptr, qb->ptr, qb->type, g_.rms_eps};
        q_path(xq_, qw, T, d_pos, rope_[kind], xq2_, qr_, q_, stream);

        int sel = 0;
        if (ratio) {
            const int ks = kv_source(il);
            if (ks == il) {
                if (!W("attn_compressor_kv.weight") || !W("attn_compressor_norm.weight") ||
                    (ratio == 2 && !W("attn_compressor_gate.weight"))) {
                    err = "deepseek41: layer " + std::to_string(il) + " lacks compressor tensors";
                    return false;
                }
                const CompressArgs ca = compress_args(il);
                if (T > 1) compress_window(ca, x, d_pos, T, cws_l_[il], stream);
                else compress(ca, x, d_pos, 1, cws_, stream);
            }
            if (index_source(il) == il) {
                const DevTensor *iqb = W("indexer.attn_q_b.weight"), *ipj = W("indexer.proj.weight");
                if (!iqb || !ipj) {
                    err = "deepseek41: layer " + std::to_string(il) + " lacks indexer tensors";
                    return false;
                }
                IndexerArgs ia;
                ia.mode = index_mode(il);
                ia.ratio = ratio;
                ia.n_embd = kAttnEmbd;
                ia.n_comp = ctx_ / ratio + 1;   // the buffers' row capacity; only the visible rows are scored
                ia.w_q = (const uint16_t*) iqb->ptr;
                ia.w_proj = (const uint16_t*) ipj->ptr;
                ia.rope_freq = freq_[1];
                ia.comp_ik = comp_ik_[ks];
                ia.cand_ids = cand_;
                ia.cand_n = cand_n_;
                ia.out_ids = ids_;
                ia.out_n = idn_;
                indexer_select(ia, qr_, x, d_pos, T, iws_, stream);
            }
            sel = kTopK;
        }

        MqaArgs a;
        a.n_tokens = T;
        a.q = q_;
        a.win = ring_[il];
        a.win_type = LatentType::F16;
        a.win_cap = kWindowSlots;
        a.win_start = d_start;
        a.win_n = d_n;
        if (ratio) {
            a.comp = comp_kv_[kv_source(il)];
            a.comp_type = LatentType::F16;
            a.comp_stride = kLatent;
            a.sel = ids_;
            a.n_sel = sel;
            a.comp_limit = d_lim;
        }
        a.sinks = (const float*) sk->ptr;
        a.pos = d_pos;
        a.rope = rope_[kind];
        a.out = attn_;
        sparse_mqa(a, mws_, stream);

        AttnOutWeights ow{oa->ptr, oa->type, ob->ptr, ob->type};
        attn_out(attn_, ow, T, xq_, oa_, out, stream);
        return true;
    }

    /// The expert staging and the tier, built on first use.
    bool ensure_tier(std::string& err) {
        if (tier_) return true;
        constexpr int kRows = ExpertTier::kMaxK;   // routed outputs of a window: tokens x slots
        constexpr size_t kPlanBytes = std::max(sizeof(ExpertPlan), sizeof(WindowPlan));
        share_g_.assign((size_t) g_.n_layers * (kMaxK + 1), nullptr);
        if (!alloc(&ex_xq_, kernels::native_q8_1_bytes(g_.n_embd, kMaxT)) ||
            !alloc(&ex_scr_, kernels::native_expert_scratch_bytes(kRows, g_.n_ff_exp)) ||
            !alloc((void**) &ex_rows_, (size_t) kRows * g_.n_embd * 4) ||
            !alloc((void**) &ex_plan_, kPlanBytes) || !alloc((void**) &ex_plan2_, kPlanBytes))
            return oom(err);
        // the plan and the CPU's rows are mapped host memory read by kernels: no copy-engine job queues behind the
        // promotion copies, and the host sync at the top of each layer orders their reuse
        constexpr unsigned kMapped = cudaHostAllocPortable | cudaHostAllocMapped;
        if (cudaHostAlloc((void**) &h_plan_, kPlanBytes, kMapped) != cudaSuccess ||
            cudaHostAlloc((void**) &h_plan2_, kPlanBytes, kMapped) != cudaSuccess ||
            cudaHostAlloc((void**) &h_ids_, 64 * 4, kMapped) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &d_plan_host_, h_plan_, 0) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &d_plan2_host_, h_plan2_, 0) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &d_ids_host_, h_ids_, 0) != cudaSuccess) {
            err = "deepseek41: cannot allocate the pinned expert staging";
            return false;
        }
        for (int k = 0; k < 64; ++k) h_ids_[k] = -1;
        DsModel::SharedTier& shared = m_.shared_tier();
        if (shared.tier) {   // another session opened it: the sessions run one at a time over the same rows
            tier_ = shared.tier;
            h_x_ = shared.h_x;
            h_rows_ = shared.h_rows;
            if (cudaHostGetDevicePointer((void**) &d_rows_, h_rows_, 0) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &d_x_host_, h_x_, 0) != cudaSuccess) {
                err = "deepseek41: cannot map the expert rows";
                return false;
            }
            return true;
        }
        if (cudaHostAlloc((void**) &h_x_, (size_t) kMaxT * g_.n_embd * 4, kMapped) != cudaSuccess ||
            cudaHostAlloc((void**) &h_rows_, (size_t) kRows * g_.n_embd * 4, kMapped) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &d_rows_, h_rows_, 0) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &d_x_host_, h_x_, 0) != cudaSuccess) {
            err = "deepseek41: cannot allocate the pinned expert staging";
            return false;
        }
        auto t = std::make_shared<ExpertTier>(m_);
        const prefillp::Dims d = m_.prompt_dims();
        ExpertTier::Reserve r;
        // the caches grow into the reserve beside the tier, then into the spare arena the tier gives back
        r.copies = kTierReserve;
        r.homes = m_.context_bytes() * (uint64_t) m_.sessions() + prefillp::workspace_bytes(d, prefillp::kHomeChunk) + prefillp::ahead_bytes(d) +
                  prefillp::kMargin + kModuleRoom + (m_.pack().vision.bytes ? kVisionRoom : 0);
        r.beside_homes = prefillp::workspace_bytes(d, prefillp::kShortChunk) + prefillp::kShortMargin;
        r.modules = kModuleRoom;
        if (!t->open(r, h_rows_, err)) return false;
        tier_ = t;
        shared = {std::move(t), h_x_, h_rows_};
        return true;
    }

    ExpertTier* expert_tier(std::string& err) override { return ensure_tier(err) ? tier_.get() : nullptr; }

    // Kernels write the mapped ids and activation: no copy-engine job queues behind the copies of experts.
    void routed_stage(int, const float* x, const int32_t* ids, void* stream) override {
        copy_row(x, d_x_host_, g_.n_embd, stream);
        kernels::quantize_q8_1_rows(x, 1, g_.n_embd, ex_xq_, stream);
        copy_words(ids, d_ids_host_, g_.n_used, stream);   // last: the host goes on once all of them landed
    }



    bool routed_experts(int il, const float* x, const int32_t* ids, const float* w, float* out, void* stream,
                        std::string& err) override {
        if (!ensure_tier(err)) return false;
        routed_stage(il, x, ids, stream);
        return routed_experts_staged(il, x, ids, w, out, stream, err);
    }

    bool routed_experts_staged(int il, const float*, const int32_t*, const float* w, float* out, void* stream,
                               std::string& err) override {
        return routed_run(il, w, out, nullptr, stream, err);
    }

    bool routed_and_shared_staged(int il, const float*, const int32_t*, const float* w, float* out, float* shared_out,
                                  void* stream, std::string& err) override {
        return routed_run(il, w, out, shared_out, stream, err);
    }

    void routed_stage_window(int T, const float* x, const int32_t* ids, void* stream) override {
        for (int t = 0; t < T; ++t) copy_row(x + (size_t) t * g_.n_embd, d_x_host_ + (size_t) t * g_.n_embd, g_.n_embd, stream);
        kernels::quantize_q8_1_rows(x, T, g_.n_embd, ex_xq_, stream);
        copy_words(ids, d_ids_host_, T * g_.n_used, stream);   // last: the host goes on once all of them landed
    }

    // The routed experts of a window. The tier is asked once per distinct expert of the window; each card runs what it holds
    // over every row that routed to it, the CPU does the same for the rest.
    bool routed_shared_window(int il, int T, const float* w, float* out, float* shared_out, void* stream,
                              std::string& err) override {
        const ExpertLayer& el = m_.pack().experts[(size_t) il];
        const int K = g_.n_used;
        const int n = g_.n_embd;
        const kernels::NativeExpertLayout L = kernels::native_expert_layout(el.gu_type, el.d_type, n, g_.n_ff_exp);
        if (L.bytes != el.blob_bytes) {
            err = "deepseek41: layer " + std::to_string(il) + " expert blob size does not match the layout";
            return false;
        }
        cudaStream_t st = (cudaStream_t) stream;
        await_ids(st, T * K);
        int32_t uniq[ExpertTier::kMaxK];
        int nu = 0;
        int uidx[ExpertTier::kMaxK];   // [t * K + k] -> index in uniq
        ExpertTier::Use use[ExpertTier::kMaxK];
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k) {
                const int32_t e = h_ids_[t * K + k];
                int u = 0;
                while (u < nu && uniq[u] != e) ++u;
                if (u == nu) {
                    uniq[nu++] = e;
                    use[u] = ExpertTier::Use{};
                }
                ExpertTier::Use& us = use[u];
                us.tok[us.nt] = t;
                us.row[us.nt] = t * K + k;
                ++us.nt;
                uidx[t * K + k] = u;
            }
        ExpertTier::Lookup lk;
        if (!tier_->lookup(il, uniq, nu, lk, err)) return false;
        const int n_fetched = tier_->fetch(il, lk);

        // the second card's share first: its rows are the last of the two cards' to land
        core::PeerExperts* peer = tier_->peer();
        if (peer) {
            int32_t kind[ExpertTier::kMaxK] = {};
            for (int r = 0; r < T * K; ++r) kind[r] = lk.peer[uidx[r]] ? 2 : 0;
            kernels::NativeExpertLayout limited = L;
            limited.swiglu_limit = g_.clamp_exp[(size_t) il];
            if (!peer->launch_window(il, limited, h_x_, h_ids_, T, K, kind, err)) return false;
        }

        // the GPU's share: the experts the VRAM tier holds; the fetched ones in a second plan, once their copies landed
        const auto make_plan = [&](WindowPlan& pl, bool fetched, int& entries) {
            int ng = 0, ne = 0;
            for (int i = 0; i < nu; ++i) {
                if (!lk.dev[i] || lk.fetched[i] != fetched) continue;
                pl.ptr[ng] = (unsigned long long) lk.dev[i];
                pl.start[ng] = ne;
                for (int j = 0; j < use[i].nt; ++j) {
                    pl.dst[ne] = use[i].row[j];
                    pl.tok[ne] = use[i].tok[j];
                    ++ne;
                }
                ++ng;
            }
            pl.start[ng] = ne;
            pl.ng[0] = ng;
            entries = ne;
            return ng;
        };
        const auto gpu_share = [&](const void* host_plan, void* dev_plan, int ng, int ne) {
            copy_words(host_plan, dev_plan, (int) (sizeof(WindowPlan) / 4), stream);
            WindowPlan* dp = (WindowPlan*) dev_plan;
            kernels::native_expert_grouped_clamped(L, g_.clamp_exp[(size_t) il], dp->ptr, dp->start, dp->ng, dp->dst, dp->tok, ng, ne,
                                                   ex_xq_, ex_scr_, ex_rows_, stream);
        };
        int entries = 0;
        const int nhits = make_plan(*(WindowPlan*) h_plan_, false, entries);
        if (nhits) gpu_share(d_plan_host_, ex_plan_, nhits, entries);
        if (n_fetched) {
            const int ng = make_plan(*(WindowPlan*) h_plan2_, true, entries);
            tier_->await_fetch(st);
            gpu_share(d_plan2_host_, ex_plan2_, ng, entries);
        }
        // the CPU's share runs while the GPU works; its rows land in their rows
        if (nhits + n_fetched < nu) {
            if (!tier_->run_cpu_window(il, h_x_, lk, use, h_rows_, err)) return false;
            tier_->promote(il, lk);
        }
        if (peer && !peer->finish(nullptr, err)) return false;
        WindowRowPtrs rp{};
        for (int r = 0; r < T * K; ++r)
            rp.p[r] = lk.dev[uidx[r]] ? ex_rows_ + (size_t) r * n : d_rows_ + (size_t) r * n;
        weighted_sum_window(rp, w, T, K, n, out, stream, shared_out);
        for (int r = 0; r < T * K; ++r) h_ids_[r] = -1;
        ids_read_ = false;
        return true;
    }

    // Every id of the layer landed in the mapped ids (routed_stage writes them last); the stream goes on meanwhile. A launch
    // that failed never writes them.
    void await_ids(cudaStream_t st, int n_ids) {
        if (ids_read_) {   // a layer that failed left its ids behind
            cudaStreamSynchronize(st);
            return;
        }
        ids_read_ = true;
        constexpr int kSpins = 1 << 22;
        const volatile int32_t* ids = h_ids_;
        for (int s = 0; s < kSpins; ++s) {
            bool all = true;
            for (int k = 0; k < n_ids; ++k) all = all && ids[k] >= 0;
            if (all) {
                std::atomic_thread_fence(std::memory_order_acquire);
                return;
            }
            platform::cpu_relax();
        }
        cudaStreamSynchronize(st);
    }

    // With `shared_out` set, the head has queued the shared expert, and the weighted sum adds its result.
    bool routed_run(int il, const float* w, float* out, float* shared_out, void* stream,
                    std::string& err) {
        const ExpertLayer& el = m_.pack().experts[(size_t) il];
        const int K = g_.n_used;
        const kernels::NativeExpertLayout L =
            kernels::native_expert_layout(el.gu_type, el.d_type, g_.n_embd, g_.n_ff_exp);
        if (L.bytes != el.blob_bytes) {
            err = "deepseek41: layer " + std::to_string(il) + " expert blob size does not match the layout";
            return false;
        }
        cudaStream_t st = (cudaStream_t) stream;
        if (shared_out) await_ids(st, K);
        else cudaStreamSynchronize(st);
        ExpertTier::Lookup lk;
        if (!tier_->lookup(il, h_ids_, K, lk, err)) return false;
        const int n_fetched = tier_->fetch(il, lk);

        // the second card's share first: its rows are the last of the two cards' to land
        core::PeerExperts* peer = tier_->peer();
        if (peer) {
            int32_t kind[ExpertTier::kMaxK] = {};
            for (int k = 0; k < K; ++k) kind[k] = lk.peer[k] ? 2 : 0;
            kernels::NativeExpertLayout limited = L;
            limited.swiglu_limit = g_.clamp_exp[(size_t) il];
            if (!peer->launch_plain(il, limited, h_x_, lk.experts, K, kind, err)) return false;
        }

        // the GPU's share: the experts the VRAM tier holds, one group each, writing the row of their router index; the
        // fetched ones in a second plan, once their copies landed, so the hits and the shared expert do not wait for them
        const auto make_plan = [&](ExpertPlan& pl, bool fetched) {
            int m = 0;
            for (int k = 0; k < K; ++k) {
                if (!lk.dev[k] || lk.fetched[k] != fetched) continue;
                pl.ptr[m] = (unsigned long long) lk.dev[k];
                pl.start[m] = m;
                pl.dst[m] = k;
                pl.tok[m] = 0;
                ++m;
            }
            pl.start[m] = m;
            pl.ng[0] = m;
            return m;
        };
        const int nhits = make_plan(*h_plan_, false);
        const int nh = nhits + n_fetched;
        const auto gpu_share = [&] {
            copy_words(d_plan_host_, ex_plan_, (int) (sizeof(ExpertPlan) / 4), stream);
            ExpertPlan* dp = (ExpertPlan*) ex_plan_;
            kernels::native_expert_grouped_clamped(L, g_.clamp_exp[(size_t) il], dp->ptr, dp->start, dp->ng, dp->dst,
                                                   dp->tok, nhits, nhits, ex_xq_, ex_scr_, ex_rows_, stream);
        };
        if (shared_out && nhits) {
            cudaGraphExec_t& exec = share_g_[(size_t) il * (kMaxK + 1) + (size_t) nhits];
            if (!exec) {
                cudaGraph_t g = nullptr;
                if (cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
                    err = "deepseek41: cannot capture a layer's experts";
                    return false;
                }
                gpu_share();
                const cudaError_t ce = cudaStreamEndCapture(st, &g);
                const cudaError_t ci = ce == cudaSuccess ? cudaGraphInstantiateWithFlags(&exec, g, 0) : ce;
                if (g) cudaGraphDestroy(g);
                if (ci != cudaSuccess) {
                    err = std::string("deepseek41: capturing a layer's experts: ") + cudaGetErrorString(ci);
                    return false;
                }
            }
            if (cudaGraphLaunch(exec, st) != cudaSuccess) {
                err = "deepseek41: launching a layer's experts failed";
                return false;
            }
        } else if (!shared_out && nhits) {
            gpu_share();
        }
        if (n_fetched) {
            make_plan(*h_plan2_, true);
            tier_->await_fetch(st);
            copy_words(d_plan2_host_, ex_plan2_, (int) (sizeof(ExpertPlan) / 4), stream);
            ExpertPlan* dp = (ExpertPlan*) ex_plan2_;
            kernels::native_expert_grouped_clamped(L, g_.clamp_exp[(size_t) il], dp->ptr, dp->start, dp->ng, dp->dst, dp->tok, n_fetched,
                                                   n_fetched, ex_xq_, ex_scr_, ex_rows_, stream);
        }
        // the CPU's share runs while the GPU works; its rows land in the router index's row
        if (nh < K) {
            if (!tier_->run_cpu(il, h_x_, lk, h_rows_, err)) return false;
            tier_->promote(il, lk);
        }
        if (peer && !peer->finish(nullptr, err)) return false;
        RowPtrs rp{};
        for (int k = 0; k < K; ++k)
            rp.p[k] = lk.dev[k] ? ex_rows_ + (size_t) k * g_.n_embd : d_rows_ + (size_t) k * g_.n_embd;
        weighted_sum_row_ptrs(rp, w, K, g_.n_embd, out, stream, shared_out);
        for (int k = 0; k < K; ++k) h_ids_[k] = -1;
        ids_read_ = false;
        return true;
    }
    bool shared_expert(int il, const float* x, float* out, void* stream, std::string& err) override {
        return shared_impl(il, 1, x, out, stream, err);
    }
    bool shared_impl(int il, int T, const float* x, float* out, void* stream, std::string& err) {
        auto W = [&](const char* s) { return m_.tensor(il, s); };
        const DevTensor *gt = W("ffn_gate_shexp.weight"), *up = W("ffn_up_shexp.weight"), *dn = W("ffn_down_shexp.weight");
        if (!gt || !up || !dn) {
            err = "deepseek41: layer " + std::to_string(il) + " lacks shared expert tensors";
            return false;
        }
        const int nf = g_.n_ff_exp * std::max(g_.n_shared, 1);
        kernels::native_quantize_q8_1(x, sh_q1_, kAttnEmbd, T, stream);
        kernels::native_mmvq(gt->type, gt->ptr, sh_q1_, sh_g_, kAttnEmbd, nf, T, stream);
        kernels::native_mmvq(up->type, up->ptr, sh_q1_, sh_u_, kAttnEmbd, nf, T, stream);
        swiglu_clamped(sh_g_, sh_u_, nf * T, g_.clamp_shexp[(size_t) il], sh_h_, stream);
        kernels::native_quantize_q8_1(sh_h_, sh_q2_, nf, T, stream);
        kernels::native_mmvq(dn->type, dn->ptr, sh_q2_, out, nf, kAttnEmbd, T, stream);
        return true;
    }

private:
    bool alloc(void** p, size_t bytes) {
        void* v = nullptr;
        if (cudaMalloc(&v, bytes) != cudaSuccess) return false;
        cudaMemset(v, 0, bytes);
        allocs_.push_back(v);
        *p = v;
        return true;
    }
    static bool oom(std::string& err) {
        err = "out of device memory for the attention state";
        return false;
    }

    const DsModel& m_;
    const Geometry& g_;
    std::unique_ptr<AttnBatch> batch_;
    std::vector<void*> allocs_;
    std::vector<uint16_t*> ring_, comp_kv_, comp_ik_;
    std::vector<std::unique_ptr<ElasticBuf>> el_kv_, el_ik_;   // backing of comp_kv_ / comp_ik_ per kv-source layer
    int64_t backed_positions_ = 0;
    int ctx_ = 0;
    std::vector<float*> st_kv_, st_sc_;
    float* freq_[2] = {nullptr, nullptr};
    RopeTable rope_[2];
    void *xq_ = nullptr, *xq2_ = nullptr, *cws_ = nullptr, *iws_ = nullptr;
    float *kvraw_ = nullptr, *qr_ = nullptr, *q_ = nullptr, *attn_ = nullptr, *oa_ = nullptr, *mws_ = nullptr;
    void *ex_plan_ = nullptr, *d_plan_host_ = nullptr, *ex_xq_ = nullptr, *ex_scr_ = nullptr;
    void *ex_plan2_ = nullptr, *d_plan2_host_ = nullptr;
    bool ids_read_ = false;   ///< this layer took the mapped ids and has not cleared them yet
    ExpertPlan* h_plan2_ = nullptr;
    float* ex_rows_ = nullptr;
    ExpertPlan* h_plan_ = nullptr;
    float *h_x_ = nullptr, *h_rows_ = nullptr, *d_rows_ = nullptr, *d_x_host_ = nullptr;
    int32_t *h_ids_ = nullptr, *h_pos_ = nullptr, *d_ids_host_ = nullptr;
    int64_t staged_pos_ = -1;
    std::shared_ptr<ExpertTier> tier_;
    static constexpr int kMaxK = 8;                 // routed experts of a token
    std::vector<cudaGraphExec_t> share_g_;          // [layer * (kMaxK + 1) + experts on the card]: their launch and the shared expert's
    float *sh_g_ = nullptr, *sh_u_ = nullptr, *sh_h_ = nullptr;
    void *sh_q1_ = nullptr, *sh_q2_ = nullptr;
    static constexpr int kPosStride = 4 * kMaxT;   // per layer: pos, window start, window rows, visible compressed rows
    std::vector<void*> cws_l_;                      // [layer] compressor workspace of a window, kept until its commit
    int staged_T_ = 1;
    int win_n_ = 0;
    int64_t win_pos0_ = 0;
    int32_t *dpos_ = nullptr, *ids_ = nullptr, *idn_ = nullptr, *cand_ = nullptr, *cand_n_ = nullptr;
};

}  // namespace

std::unique_ptr<LayerOps> make_kernel_ops(const DsModel& model, std::string& err) {
    auto o = std::make_unique<KernelOps>(model);
    if (!o->init(err)) return nullptr;
    return o;
}

}  // namespace basalt::models::deepseek41
