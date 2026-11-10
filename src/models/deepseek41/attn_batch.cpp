// src/models/deepseek41/attn_batch.cpp - attention of a chunk of prompt tokens for one layer.
//
// The dense projections (wkv, wq_a, wq_b, wo_a, wo_b) run over sub-chunks of kSub tokens as MMQ GEMMs. Between them
// the window, compressor and indexer run over the whole sub-chunk and the sparse attention in groups of kMqaT tokens,
// with the positions, window bounds and top-k ids of the whole chunk held on the device. The window rows of a
// sub-chunk are staged in a larger ring than the decode ring, which still receives the rows decode will read.
#include "models/deepseek41/attn_batch.hpp"

#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/deepseek41/attn_index.hpp"
#include "basalt/models/deepseek41/prefill_attn.hpp"
#include "basalt/models/deepseek41/prefill_index.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "models/deepseek41/attn_dense.hpp"
#include "models/deepseek41/model.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace basalt::models::deepseek41 {
namespace {

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

constexpr int kSub = 1024;   // tokens per dense sub-chunk
constexpr int kMqaT = 64;    // tokens per sparse-attention launch
constexpr int kStageSlots = kSub + kWindow;   // staged window ring: a sub-chunk plus the rows before it
constexpr int kMqaChunks = (kWindow + kAttnChunk - 1) / kAttnChunk + (kTopK + kAttnChunk - 1) / kAttnChunk;
constexpr size_t kIndexBudget = 160u << 20;   // bytes of indexer workspace

// Device bytes of the buffers that do not grow with the chunk, in the order init allocates them.
struct Sizes {
    size_t xq, kvraw, qr, q, attn, oa, ident, stage, mws, cws, iws, fixed;
};
Sizes sizes(CacheFormat f) {
    namespace mmq = basalt::prefill::mmq;
    const size_t S = kSub;
    Sizes z;
    z.xq = std::max({mmq::q8_bytes((int64_t) S, kAttnEmbd), mmq::q8_bytes((int64_t) S, kAttnQRank),
                     mmq::q8_bytes((int64_t) S * kAttnGroups, kAttnGroupIn),
                     mmq::q8_bytes((int64_t) S, kAttnGroups * kAttnOutRank), (size_t) 1 << 16});
    z.kvraw = S * kLatent * 4;
    z.qr = S * kAttnQRank * 4;
    z.q = z.attn = S * kAttnHeads * kAttnHeadDim * 4;
    z.oa = S * kAttnGroups * kAttnOutRank * 4;
    z.ident = S * 4;
    z.stage = (size_t) kStageSlots * win_row_bytes(f);
    z.mws = ((size_t) kMqaT * kAttnHeads * kMqaChunks * (kAttnHeadDim + 2) + 64) * 4;
    z.cws = compress_batch_workspace_bytes(kSub);
    z.iws = kIndexBudget;
    z.fixed = (size_t) (kAttnGroups * S + 16) * 4;   // each of the two sub-chunk sizes' index arrays
    return z;
}
// per token of the chunk: positions, window bounds, both visible-row limits, top-k ids and count, candidates and count
constexpr size_t kTokenBytes = 4 * (5 + kTopK + 1 + kMaxCandBlocks + 1);

class Impl final : public AttnBatch {
public:
    Impl(const DsModel& m, const AttnCaches& c, int cap, int max_pos) : m_(m), g_(m.geo()), c_(c), cap_(cap), max_pos_(max_pos) {}
    ~Impl() override {
        for (void* p : allocs_) cudaFree(p);
        for (Fixed& f : fixed_)
            if (f.host) cudaFreeHost(f.host);
    }

    bool init(std::string& err) {
        namespace mmq = basalt::prefill::mmq;
        const size_t C = (size_t) cap_, S = kSub;
        const Sizes z = sizes(c_.format);
        bool ok = alloc(&xq_, z.xq) && alloc(&kvraw_, z.kvraw) && alloc(&qr_, z.qr) && alloc(&q_, z.q) &&
                  alloc(&attn_, z.attn) && alloc(&oa_, z.oa) && alloc(&ident_, z.ident) && alloc(&stage_, z.stage) &&
                  alloc(&mws_, z.mws) && alloc(&cws_, z.cws) && alloc(&iws_, z.iws) &&
                  alloc(&d_pos_, C * 4) && alloc(&d_start_, C * 4) && alloc(&d_n_, C * 4) && alloc(&d_lim_[0], C * 4) &&
                  alloc(&d_lim_[1], C * 4) && alloc(&ids_, C * kTopK * 4) && alloc(&idn_, C * 4) &&
                  alloc(&cand_, C * kMaxCandBlocks * 4) && alloc(&cand_n_, C * 4);
        // layer-independent index arrays of the full and the tail sub-chunk, uploaded once from pinned memory
        for (Fixed& f : fixed_) {
            f.dev = nullptr;
            ok = ok && alloc(&f.dev, z.fixed) && cudaMallocHost((void**) &f.host, z.fixed) == cudaSuccess;
        }
        if (ok) {
            mmq::iota(ident_, (int64_t) S, nullptr);
            cudaDeviceSynchronize();
        }
        if (!ok) err = "out of device memory for the batched attention";
        return ok;
    }

    bool run(int il, int64_t pos0, int T, const float* x, float* out, void* stream, std::string& err) override {
        cudaStream_t st = (cudaStream_t) stream;
        if (pos0 + T > max_pos_ || T > cap_) {
            err = "deepseek41: the chunk exceeds the compressed cache capacity";
            return false;
        }
        if (il == 0 || chunk_pos0_ != pos0 || chunk_n_ != T) upload_positions(pos0, T, st);
        for (int s0 = 0; s0 < T; s0 += kSub) {
            const int S = std::min(kSub, T - s0);
            const float* xs = x + (size_t) s0 * kAttnEmbd;
            select_fixed(S, stream);
            if (!dense_in(il, s0, S, xs, stream, err)) return false;
            if (!sub(il, pos0, s0, S, xs, stream, err)) return false;
            if (!dense_out(il, S, out + (size_t) s0 * kAttnEmbd, stream, err)) return false;
        }
        return true;
    }

    bool compress_only(int il, int64_t pos0, int T, const float* x, void* stream, std::string& err) override {
        if (pos0 + T > max_pos_ || T > cap_) {
            err = "deepseek41: the chunk exceeds the compressed cache capacity";
            return false;
        }
        if (chunk_pos0_ != pos0 || chunk_n_ != T) upload_positions(pos0, T, (cudaStream_t) stream);
        for (int s0 = 0; s0 < T; s0 += kSub)
            if (!compress_sub(il, x + (size_t) s0 * kAttnEmbd, d_pos_ + s0, std::min(kSub, T - s0), stream, err)) return false;
        return true;
    }

private:
    // positions, window bounds and visible compressed rows of the whole chunk
    void upload_positions(int64_t pos0, int T, cudaStream_t st) {
        std::vector<int32_t> pos((size_t) T), start((size_t) T), n((size_t) T), l1((size_t) T), l2((size_t) T);
        for (int t = 0; t < T; ++t) {
            const int32_t p = (int32_t) (pos0 + t);
            pos[(size_t) t] = p;
            start[(size_t) t] = std::max<int32_t>(0, p - (kWindow - 1));
            n[(size_t) t] = p - start[(size_t) t] + 1;
            l1[(size_t) t] = p + 1;
            l2[(size_t) t] = (p + 1) / 2;
        }
        cudaMemcpyAsync(d_pos_, pos.data(), (size_t) T * 4, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(d_start_, start.data(), (size_t) T * 4, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(d_n_, n.data(), (size_t) T * 4, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(d_lim_[0], l1.data(), (size_t) T * 4, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(d_lim_[1], l2.data(), (size_t) T * 4, cudaMemcpyHostToDevice, st);
        chunk_pos0_ = pos0;
        chunk_n_ = T;
    }

    struct Fixed {
        int32_t *dev = nullptr, *host = nullptr;
        int S = -1;
    };

    // Points wo_ids_ / bounds_ at the arrays for S tokens, uploading them the first time S is seen.
    void select_fixed(int S, void* stream) {
        Fixed& f = fixed_[S == kSub ? 0 : 1];
        if (f.S != S) {
            f.S = S;
            int32_t* h = f.host;
            for (int g = 0; g < kAttnGroups; ++g)
                for (int t = 0; t < S; ++t) h[(size_t) g * S + t] = t * kAttnGroups + g;
            int32_t* bd = h + (size_t) kAttnGroups * S;
            bd[0] = 0;
            bd[1] = S;
            for (int g = 0; g <= kAttnGroups; ++g) bd[2 + g] = g * S;
            cudaMemcpyAsync(f.dev, h, (size_t) (kAttnGroups * S + 16) * 4, cudaMemcpyHostToDevice, (cudaStream_t) stream);
        }
        wo_ids_ = f.dev;
        bounds_ = f.dev + (size_t) kAttnGroups * S;
    }

    // y[S][rows] = x . W^T for one dense matrix (x already quantized for its type)
    void gemm(const DevTensor* w, int rows, int cols, const void* xq, int S, float* dst, void* stream) {
        namespace mmq = basalt::prefill::mmq;
        mmq::Product p;
        p.w = w->ptr;
        p.type = w->type;
        p.w_rows = rows;
        p.w_cols = cols;
        p.expert_bytes = mmq::matrix_bytes(w->type, rows, cols);
        p.n = 1;
        p.xq = xq;
        p.bounds = bounds_;
        p.ids = ident_;
        p.total_rows = S;
        p.max_rows = S;
        p.dst = dst;
        p.ld_dst = rows;
        ctx_.run(p, stream);
    }

    bool dense_in(int il, int s0, int S, const float* x, void* stream, std::string& err) {
        namespace mmq = basalt::prefill::mmq;
        auto Tn = [&](const char* n) { return m_.tensor(il, n); };
        const DevTensor *wkv = Tn("attn_kv.weight"), *qa = Tn("attn_q_a.weight"), *qan = Tn("attn_q_a_norm.weight"),
                        *qb = Tn("attn_q_b.weight");
        if (!wkv || !qa || !qan || !qb) {
            err = "deepseek41: layer " + std::to_string(il) + " lacks attention tensors";
            return false;
        }
        mmq::quantize(x, nullptr, xq_, wkv->type, kAttnEmbd, kAttnEmbd, S, stream);
        gemm(wkv, kLatent, kAttnEmbd, xq_, S, kvraw_, stream);
        if (qa->type != wkv->type) mmq::quantize(x, nullptr, xq_, qa->type, kAttnEmbd, kAttnEmbd, S, stream);
        gemm(qa, kAttnQRank, kAttnEmbd, xq_, S, qr_, stream);
        rms_norm_rows(qr_, (const float*) qan->ptr, kAttnQRank, g_.rms_eps, S, stream);
        mmq::quantize(qr_, nullptr, xq_, qb->type, kAttnQRank, kAttnQRank, S, stream);
        gemm(qb, kAttnHeads * kAttnHeadDim, kAttnQRank, xq_, S, q_, stream);
        rope_rows(q_, S * kAttnHeads, kAttnHeadDim, kAttnHeads, d_pos_ + s0, c_.rope[g_.compress_ratio[(size_t) il] ? 1 : 0],
                  false, stream);
        return true;
    }

    bool dense_out(int il, int S, float* out, void* stream, std::string& err) {
        namespace mmq = basalt::prefill::mmq;
        const DevTensor *oa = m_.tensor(il, "attn_output_a.weight"), *ob = m_.tensor(il, "attn_output_b.weight");
        if (!oa || !ob) {
            err = "deepseek41: layer " + std::to_string(il) + " lacks attention tensors";
            return false;
        }
        // wo_a is block-diagonal: group g multiplies column g of every token's [8][4096] attention row
        mmq::quantize(attn_, wo_ids_, xq_, oa->type, kAttnGroupIn, kAttnGroupIn, (int64_t) kAttnGroups * S, stream);
        mmq::Product p;
        p.w = oa->ptr;
        p.type = oa->type;
        p.w_rows = kAttnOutRank;
        p.w_cols = kAttnGroupIn;
        p.expert_bytes = mmq::matrix_bytes(oa->type, kAttnOutRank, kAttnGroupIn);
        p.n = kAttnGroups;
        p.xq = xq_;
        p.bounds = bounds_ + 2;
        p.ids = wo_ids_;
        p.total_rows = (int64_t) kAttnGroups * S;
        p.max_rows = S;
        p.dst = oa_;
        p.ld_dst = kAttnOutRank;
        ctx_.run(p, stream);
        mmq::quantize(oa_, nullptr, xq_, ob->type, kAttnGroups * kAttnOutRank, kAttnGroups * kAttnOutRank, S, stream);
        gemm(ob, kAttnEmbd, kAttnGroups * kAttnOutRank, xq_, S, out, stream);
        return true;
    }

    // The compressed latents and index keys of S tokens of a kv-source layer.
    bool compress_sub(int il, const float* x, const int32_t* d_pos, int S, void* stream, std::string& err) {
        const int ratio = g_.compress_ratio[(size_t) il];
        auto Tn = [&](const char* s) { return m_.tensor(il, s); };
        const DevTensor *wc = Tn("attn_compressor_kv.weight"), *cn = Tn("attn_compressor_norm.weight");
        const DevTensor *wg = ratio == 2 ? Tn("attn_compressor_gate.weight") : nullptr;
        const DevTensor *ik = Tn("indexer.attn_k.weight"), *ikn = Tn("indexer.k_norm.weight");
        if (!wc || !cn || (ratio == 2 && !wg)) {
            err = "deepseek41: layer " + std::to_string(il) + " lacks compressor tensors";
            return false;
        }
        CompressArgs ca;
        ca.ratio = ratio;
        ca.format = c_.format;
        ca.rms_eps = g_.rms_eps;
        ca.n_embd = kAttnEmbd;
        ca.w_kv = (const uint16_t*) wc->ptr;
        ca.w_gate = wg ? (const uint16_t*) wg->ptr : nullptr;
        ca.norm_w = (const float*) cn->ptr;
        ca.w_idx_k = ik ? (const uint16_t*) ik->ptr : nullptr;
        ca.idx_k_norm_w = ikn ? (const float*) ikn->ptr : nullptr;
        ca.rope_freq = c_.freq[1];
        ca.state_kv = (*c_.st_kv)[(size_t) il];
        ca.state_score = (*c_.st_sc)[(size_t) il];
        ca.comp_kv = (*c_.comp_kv)[(size_t) il];
        ca.comp_ik = (*c_.comp_ik)[(size_t) il];
        compress_batch(ca, x, d_pos, S, cws_, stream);
        return true;
    }

    // One sub-chunk of S tokens starting at token s0 of the chunk: window, compressor, indexer, sparse attention.
    bool sub(int il, int64_t pos0, int s0, int S, const float* x, void* stream, std::string& err) {
        const int ratio = g_.compress_ratio[(size_t) il];
        const int kind = ratio ? 1 : 0;
        auto Tn = [&](const char* s) { return m_.tensor(il, s); };
        const DevTensor *kvn = Tn("attn_kv_a_norm.weight"), *sk = Tn("attn_sinks.weight");
        if (!kvn || !sk) {
            err = "deepseek41: layer " + std::to_string(il) + " lacks attention tensors";
            return false;
        }
        const int32_t *d_pos = d_pos_ + s0, *d_start = d_start_ + s0, *d_n = d_n_ + s0;
        const int32_t* d_lim = ratio ? d_lim_[ratio == 2 ? 1 : 0] + s0 : nullptr;
        const int64_t p0 = pos0 + s0, pos_last = p0 + S - 1;
        uint8_t* ring = (*c_.ring)[(size_t) il];

        const int64_t lo = std::max<int64_t>(0, p0 - (kWindow - 1));
        window_stage_copy(ring, (uint8_t*) stage_, kStageSlots, lo, (int) (p0 - lo), c_.format, stream);
        kv_window_stage(kvraw_, (const float*) kvn->ptr, d_pos, S, g_.rms_eps, c_.freq[kind], (uint8_t*) stage_, kStageSlots, ring,
                        std::max(0, S - kWindowSlots), c_.format, stream);

        int sel = 0;
        if (ratio) {
            const int ks = kv_source(il);
            if (ks == il) {
                if (!compress_sub(il, x, d_pos, S, stream, err)) return false;
            }
            if (index_source(il) == il) {
                const DevTensor *iqb = Tn("indexer.attn_q_b.weight"), *ipj = Tn("indexer.proj.weight");
                if (!iqb || !ipj) {
                    err = "deepseek41: layer " + std::to_string(il) + " lacks indexer tensors";
                    return false;
                }
                // tokens per indexer call: the score buffers of a call (rows x tokens) stay inside the workspace
                const size_t rows = (size_t) ((p0 + S) / ratio) + 1;
                const int index_tokens = (int) std::max<size_t>(16, std::min<size_t>(kSub, (kIndexBudget - 4096) /
                    ((kIdxHeads * kIdxDim + kIdxHeads) * 4 + rows * 9)));
                for (int g0 = 0; g0 < S; g0 += index_tokens) {
                    const int n = std::min(index_tokens, S - g0);
                    IndexerArgs ia;
                    ia.mode = index_mode(il);
                    ia.ratio = ratio;
                    ia.format = c_.format;
                    ia.n_embd = kAttnEmbd;
                    ia.n_comp = std::max<int>(1, (int) ((p0 + g0 + n) / ratio));
                    ia.w_q = (const uint16_t*) iqb->ptr;
                    ia.w_proj = (const uint16_t*) ipj->ptr;
                    ia.rope_freq = c_.freq[1];
                    ia.comp_ik = (*c_.comp_ik)[(size_t) ks];
                    ia.cand_ids = cand_ + (size_t) (s0 + g0) * kMaxCandBlocks;
                    ia.cand_n = cand_n_ + s0 + g0;
                    ia.out_ids = ids_ + (size_t) (s0 + g0) * kTopK;
                    ia.out_n = idn_ + s0 + g0;
                    indexer_select_batch(ia, qr_ + (size_t) g0 * kAttnQRank, x + (size_t) g0 * kAttnEmbd, d_pos + g0, n, iws_,
                                         stream);
                }
            }
            sel = kTopK;
        }
        (void) pos_last;

        for (int l0 = 0; l0 < S; l0 += kMqaT) {
            MqaArgs a;
            a.n_tokens = std::min(kMqaT, S - l0);
            a.q = q_ + (size_t) l0 * kAttnHeads * kAttnHeadDim;
            a.win = stage_;
            a.win_type = c_.format == CacheFormat::Fp8 ? LatentType::Fp8 : LatentType::F16;
            a.win_cap = kStageSlots;
            a.win_start = d_start + l0;
            a.win_n = d_n + l0;
            if (ratio) {
                a.comp = (*c_.comp_kv)[(size_t) kv_source(il)];
                a.comp_type = c_.format == CacheFormat::Fp8 ? LatentType::Fp4 : LatentType::F16;
                a.sel = ids_ + (size_t) (s0 + l0) * kTopK;
                a.n_sel = sel;
                a.comp_limit = d_lim + l0;
            }
            a.sinks = (const float*) sk->ptr;
            a.pos = d_pos + l0;
            a.rope = c_.rope[kind];
            a.out = attn_ + (size_t) l0 * kAttnHeads * kAttnHeadDim;
            sparse_mqa_tc(a, mws_, stream);
        }
        return true;
    }

    template <class P>
    bool alloc(P** p, size_t bytes) {
        void* v = nullptr;
        if (cudaMalloc(&v, bytes) != cudaSuccess) return false;
        cudaMemset(v, 0, bytes);
        allocs_.push_back(v);
        *p = (P*) v;
        return true;
    }

    const DsModel& m_;
    const Geometry& g_;
    AttnCaches c_;
    int cap_, max_pos_;
    int64_t chunk_pos0_ = -1;
    int chunk_n_ = 0;
    Fixed fixed_[2];
    void* stage_ = nullptr;
    std::vector<void*> allocs_;
    void *xq_ = nullptr, *cws_ = nullptr, *iws_ = nullptr;
    float *kvraw_ = nullptr, *qr_ = nullptr, *q_ = nullptr, *attn_ = nullptr, *oa_ = nullptr, *mws_ = nullptr;
    int32_t *d_pos_ = nullptr, *d_start_ = nullptr, *d_n_ = nullptr, *d_lim_[2] = {nullptr, nullptr}, *ids_ = nullptr,
            *idn_ = nullptr, *cand_ = nullptr, *cand_n_ = nullptr, *ident_ = nullptr, *wo_ids_ = nullptr, *bounds_ = nullptr;
    basalt::prefill::mmq::Context ctx_;
};

}  // namespace

uint64_t attn_batch_fixed_bytes(CacheFormat format) {
    const Sizes z = sizes(format);
    return z.xq + z.kvraw + z.qr + z.q + z.attn + z.oa + z.ident + z.stage + z.mws + z.cws + z.iws + 2 * z.fixed;
}

uint64_t attn_batch_token_bytes() { return kTokenBytes; }

std::unique_ptr<AttnBatch> make_attn_batch(const DsModel& model, const AttnCaches& caches, int max_tokens, int max_positions,
                                           std::string& err) {
    auto p = std::make_unique<Impl>(model, caches, max_tokens, max_positions);
    if (!p->init(err)) return nullptr;
    return p;
}

}  // namespace basalt::models::deepseek41
