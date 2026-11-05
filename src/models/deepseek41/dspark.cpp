// src/models/deepseek41/dspark.cpp - the DSpark drafter (see dspark.hpp). The forward reuses the DeepSeek decode
// kernels (mHC, router, norms, query path, sparse MQA, grouped output, Q8_0 MMVQ) at the drafter's geometry; the MXFP4
// experts, the latent rotation and the Markov and confidence heads are in dspark_kernels.cu.
//
// The stage structure follows llama.cpp's DeepSeek-V4.1 DSpark graph (MIT, Copyright (c) 2023-2026 The ggml authors).
#include "models/deepseek41/dspark.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/expert_cache.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/models/deepseek41/attn.hpp"
#include "basalt/models/deepseek41/attn_index.hpp"
#include "basalt/models/deepseek41/kernels.hpp"
#include "models/deepseek41/dspark_kernels.hpp"
#include "models/deepseek41/model.hpp"
#include "models/deepseek41/ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace basalt::models::deepseek41 {
namespace {

constexpr int kT = kDsparkBlock;
constexpr int kN = 5120;
constexpr int kStages = 3;
constexpr int kExperts = 128;
constexpr int kUsed = 3;
constexpr int kFF = 2304;
constexpr int kMarkovR = 256;
constexpr int kInjectRows = 8;       // rows of one inject launch
constexpr float kEps = 1e-20f;
constexpr int kMxfp4 = 39;

// What a DSpark GGUF must be for this build, whatever else the file holds.
std::string gguf_problem(const GgufFile& g) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch || arch->s != "dflash") return "not a DSpark GGUF (architecture is not dflash)";
    const auto u = [&](const char* k) -> uint64_t {
        const MetaValue* v = g.get(k);
        return v ? v->u : 0;
    };
    if (u("dflash.block_count") != kStages || u("dflash.embedding_length") != kN || u("dflash.expert_count") != kExperts ||
        u("dflash.expert_used_count") != kUsed || u("dflash.block_size") != kDsparkBlock || !u("dflash.hyper_connection.single_pass"))
        return "the GGUF geometry is not the 3-stage, 128-expert, block-5 DSpark this build runs";
    if (!g.get("dflash.target_layers")) return "no dflash.target_layers";
    if (!g.get("tokenizer.ggml.mask_token_id")) return "no mask token id";
    for (const char* name : {"fc.weight", "blk.0.attn_kv.weight", "blk.0.attn_q_a.weight", "blk.0.attn_q_b.weight", "blk.0.attn_output_a.weight",
                             "blk.0.attn_output_b.weight", "blk.0.ffn_gate_shexp.weight"}) {
        const TensorInfo* t = g.find(name);
        if (!t) return std::string("no tensor ") + name;
        if (!kernels::native_mmvq_supported((int) t->type)) return std::string(name) + " has a type no decode kernel reads";
    }
    return {};
}

struct W {
    void* p = nullptr;
    int type = 0;
};

struct Stage {
    W wq_a, wq_b, wkv, wo_a, wo_b, sh_gate, sh_up, sh_down, gate_exps, up_exps, down_exps;
    float *q_norm = nullptr, *kv_norm = nullptr, *sinks = nullptr, *attn_norm = nullptr, *ffn_norm = nullptr;
    float *hc_attn_fn = nullptr, *hc_attn_scale = nullptr, *hc_attn_base = nullptr;
    float *hc_ffn_fn = nullptr, *hc_ffn_scale = nullptr, *hc_ffn_base = nullptr;
    float *router = nullptr, *router_bias = nullptr;
    uint16_t* ring = nullptr;   ///< [kWindowSlots][512] fp16
    int n_exp = kExperts;       ///< routed experts kept, the router's choices
};

}  // namespace

struct DsparkDrafter::Impl {
    const DsModel& main;
    int device = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t ev0 = nullptr, ev_body = nullptr, ev1 = nullptr;
    std::vector<void*> allocs;
    std::vector<int> target_layers;
    int32_t mask_id = 0;
    Stage stage[kStages];
    W fc, head;
    float *enc_norm = nullptr, *out_norm = nullptr, *conf_w = nullptr;
    uint16_t *w1 = nullptr, *w2 = nullptr;
    float* freq = nullptr;
    RopeTable rope;
    std::vector<float> mask_emb;

    // scratch
    float *x = nullptr, *x2 = nullptr, *cur = nullptr, *normed = nullptr, *attn_o = nullptr, *ffn_sum = nullptr,
          *sh_out = nullptr, *coef_a = nullptr, *coef_f[2] = {nullptr, nullptr}, *hc_ws = nullptr, *rlogits = nullptr,
          *rw = nullptr, *kvraw = nullptr, *blockkv = nullptr, *qr = nullptr, *q = nullptr, *attn = nullptr,
          *oa = nullptr, *mws = nullptr, *sh_g = nullptr, *sh_u = nullptr, *sh_h = nullptr, *moe_h = nullptr,
          *moe_d = nullptr, *logits = nullptr, *col = nullptr, *w1p = nullptr, *dconf = nullptr, *hcout = nullptr,
          *feat = nullptr, *mainx = nullptr, *mainxn = nullptr;
    void *xq = nullptr, *xq2 = nullptr, *sh_q1 = nullptr, *sh_q2 = nullptr, *am_ws = nullptr;
    int32_t *ids = nullptr, *dpos = nullptr, *dstart = nullptr, *dn = nullptr, *dlim = nullptr, *dsel = nullptr,
            *dtok = nullptr, *dpos1 = nullptr;

    // pinned host staging
    float* h_feat = nullptr;     // [kInjectRows][feature floats]
    int32_t* h_pos = nullptr;    // [kInjectRows]
    float* h_emb = nullptr;
    int32_t* h_meta = nullptr;   // pos[5], start[5], n[5], lim[5], sel[25], tok0
    int32_t* h_out = nullptr;    // tok[5]
    float* h_conf = nullptr;

    // which position each ring slot holds
    std::vector<int64_t> ring_pos;
    bool block_live = false;
    std::string kept_note;
    uint64_t bytes_taken = 0;
    int64_t blocks = 0;
    double ms_body = 0, ms_head = 0;

    explicit Impl(const DsModel& m) : main(m) {}
    ~Impl() {
        core::OnDevice g(device);
        for (void* p : allocs) cudaFree(p);
        if (h_feat) cudaFreeHost(h_feat);
        if (h_pos) cudaFreeHost(h_pos);
        if (h_emb) cudaFreeHost(h_emb);
        if (h_meta) cudaFreeHost(h_meta);
        if (h_out) cudaFreeHost(h_out);
        if (h_conf) cudaFreeHost(h_conf);
        if (ev0) cudaEventDestroy(ev0);
        if (ev_body) cudaEventDestroy(ev_body);
        if (ev1) cudaEventDestroy(ev1);
        if (st) cudaStreamDestroy(st);
    }

    bool alloc(void** p, size_t bytes) {
        void* v = nullptr;
        if (cudaMalloc(&v, bytes) != cudaSuccess) return false;
        cudaMemset(v, 0, bytes);
        allocs.push_back(v);
        *p = v;
        return true;
    }
    template <class T>
    bool buf(T*& p, size_t count) {
        return alloc((void**) &p, count * sizeof(T));
    }

    bool upload(const void* src, size_t bytes, void** dst, size_t pad = 0) {
        if (bytes == 0 || !alloc(dst, bytes + pad)) return false;
        return cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    }

    bool load(const core::FileRange& section, const std::string& ranking, int keep, std::string& err) {
        size_t free0 = 0, free1 = 0, total_b = 0;
        cudaMemGetInfo(&free0, &total_b);
        const bool ok = read(section, ranking, keep, err);
        cudaMemGetInfo(&free1, &total_b);
        bytes_taken = free0 > free1 ? (uint64_t) (free0 - free1) : 0;
        return ok;
    }

    // The experts each stage keeps: the ranking's most routed, every one without a ranking that reads.
    std::vector<std::vector<int>> kept_experts(const std::string& ranking, int asked) {
        std::vector<std::pair<int32_t, int32_t>> ranked;
        int64_t slots = 0;
        uint64_t bytes = 0;
        std::string why;
        if (asked > 0 && asked < kExperts) {
            if (ranking.empty() || !platform::file_size(ranking, bytes) || bytes == 0) why = "no ranking at " + ranking;
            else core::read_expert_profile(core::FileRange{ranking, 0, bytes}, kStages, kExperts, ranked, slots, why);
        }
        const int keep = why.empty() && asked > 0 && asked < kExperts ? asked : kExperts;
        kept_note = std::to_string(keep) + " of " + std::to_string(kExperts) + " experts per stage" + (why.empty() ? "" : " (" + why + ")");
        return dspark_kept_experts(ranked, kStages, kExperts, keep);
    }

    bool read(const core::FileRange& section, const std::string& ranking, int keep_asked, std::string& err) {
        std::unique_ptr<GgufFile> g;
        try {
            g = std::make_unique<GgufFile>(section.path, section.offset, section.bytes);
        } catch (const std::exception& e) {
            err = std::string("dspark: ") + e.what();
            return false;
        }
        if (const std::string why = gguf_problem(*g); !why.empty()) {
            err = "dspark: " + why;
            return false;
        }
        for (const MetaValue& v : g->get("dflash.target_layers")->items) target_layers.push_back((int) v.u);
        mask_id = (int32_t) g->get("tokenizer.ggml.mask_token_id")->u;

        auto find = [&](const std::string& n) -> const TensorInfo* {
            const TensorInfo* t = g->find(n);
            if (!t) err = "dspark: missing tensor " + n;
            return t;
        };
        // the shared reader has no MXFP4 geometry: 32 elements in 17 bytes
        auto bytes_of = [&](const TensorInfo& t) {
            return t.type == kMxfp4 ? (size_t) (t.elements() / kMxfp4BlockElems) * kMxfp4BlockBytes : (size_t) tensor_payload_bytes(t);
        };
        auto wt = [&](const std::string& n, W& w) {
            const TensorInfo* t = find(n);
            if (!t) return false;
            w.type = (int) t->type;
            return upload(g->tensor_data(*t), bytes_of(*t), &w.p, t->type == kMxfp4 ? kMxfp4Pad : 0);
        };
        auto ft = [&](const std::string& n, float*& p, size_t expect) {
            const TensorInfo* t = find(n);
            if (!t) return false;
            if (t->type != 0 || t->elements() != expect) {
                err = "dspark: " + n + " is not the expected fp32 tensor";
                return false;
            }
            return upload(g->tensor_data(*t), bytes_of(*t), (void**) &p);
        };
        // a stage's kept experts, and their router rows and biases, in ascending order
        const std::vector<std::vector<int>> keep = kept_experts(ranking, keep_asked);
        auto experts_of = [&](const std::string& n, W& w, const std::vector<int>& k) {
            const TensorInfo* t = find(n);
            if (!t) return false;
            w.type = (int) t->type;
            const size_t one = bytes_of(*t) / kExperts;
            std::vector<uint8_t> h(one * k.size());
            const uint8_t* src = (const uint8_t*) g->tensor_data(*t);
            for (size_t i = 0; i < k.size(); ++i) std::memcpy(h.data() + i * one, src + (size_t) k[i] * one, one);
            return upload(h.data(), h.size(), &w.p, kMxfp4Pad);
        };
        auto rows_of = [&](const std::string& n, float*& p, size_t row_floats, const std::vector<int>& k) {
            const TensorInfo* t = find(n);
            if (!t) return false;
            if (t->type != 0 || t->elements() != row_floats * kExperts) {
                err = "dspark: " + n + " is not the expected fp32 tensor";
                return false;
            }
            std::vector<float> h(row_floats * k.size());
            const float* src = (const float*) g->tensor_data(*t);
            for (size_t i = 0; i < k.size(); ++i) std::memcpy(h.data() + i * row_floats, src + (size_t) k[i] * row_floats, row_floats * 4);
            return upload(h.data(), h.size() * 4, (void**) &p);
        };
        for (int l = 0; l < kStages; ++l) {
            Stage& s = stage[l];
            const std::string b = "blk." + std::to_string(l) + ".";
            const std::vector<int>& kp = keep[(size_t) l];
            s.n_exp = (int) kp.size();
            bool ok = wt(b + "attn_q_a.weight", s.wq_a) && wt(b + "attn_q_b.weight", s.wq_b) &&
                      wt(b + "attn_kv.weight", s.wkv) && wt(b + "attn_output_a.weight", s.wo_a) &&
                      wt(b + "attn_output_b.weight", s.wo_b) && wt(b + "ffn_gate_shexp.weight", s.sh_gate) &&
                      wt(b + "ffn_up_shexp.weight", s.sh_up) && wt(b + "ffn_down_shexp.weight", s.sh_down) &&
                      experts_of(b + "ffn_gate_exps.weight", s.gate_exps, kp) && experts_of(b + "ffn_up_exps.weight", s.up_exps, kp) &&
                      experts_of(b + "ffn_down_exps.weight", s.down_exps, kp);
            ok = ok && ft(b + "attn_q_a_norm.weight", s.q_norm, 1280) && ft(b + "attn_kv_a_norm.weight", s.kv_norm, 512) &&
                 ft(b + "attn_sinks.weight", s.sinks, 64) && ft(b + "attn_norm.weight", s.attn_norm, kN) &&
                 ft(b + "ffn_norm.weight", s.ffn_norm, kN) && ft(b + "hc_attn_fn.weight", s.hc_attn_fn, 20480 * 24) &&
                 ft(b + "hc_attn_scale.weight", s.hc_attn_scale, 3) && ft(b + "hc_attn_base.weight", s.hc_attn_base, 24) &&
                 ft(b + "hc_ffn_fn.weight", s.hc_ffn_fn, 20480 * 24) && ft(b + "hc_ffn_scale.weight", s.hc_ffn_scale, 3) &&
                 ft(b + "hc_ffn_base.weight", s.hc_ffn_base, 24) && rows_of(b + "ffn_gate_inp.weight", s.router, (size_t) kN, kp) &&
                 rows_of(b + "exp_probs_b.bias", s.router_bias, 1, kp);
            if (!ok) {
                if (err.empty()) err = "dspark: cannot upload stage " + std::to_string(l);
                return false;
            }
            if (s.gate_exps.type != kMxfp4 || s.up_exps.type != kMxfp4 || s.down_exps.type != kMxfp4) {
                err = "dspark: the routed experts are not MXFP4";
                return false;
            }
            if (!buf(s.ring, (size_t) kWindowSlots * kLatent)) {
                err = "dspark: out of device memory";
                return false;
            }
        }
        if (!wt("fc.weight", fc) || !ft("enc.output_norm.weight", enc_norm, kN) || !ft("output_norm.weight", out_norm, kN) ||
            !ft("conf_proj.weight", conf_w, kN + kMarkovR)) {
            if (err.empty()) err = "dspark: cannot upload the heads";
            return false;
        }
        const TensorInfo *t1 = find("markov_w1.weight"), *t2 = find("markov_w2.weight");
        if (!t1 || !t2) return false;
        if (t1->type != 30 || t2->type != 30 || t1->shape[0] != kMarkovR) {
            err = "dspark: the Markov tables are not bf16 rank-256";
            return false;
        }
        if (!upload(g->tensor_data(*t1), bytes_of(*t1), (void**) &w1) || !upload(g->tensor_data(*t2), bytes_of(*t2), (void**) &w2)) {
            err = "dspark: out of device memory (Markov tables)";
            return false;
        }

        // the main model's output matrix (the drafter has none of its own): read in place on its card, else staged through
        // the host
        const DevTensor* h = main.tensor("output.weight");
        if (!h) {
            err = "dspark: the main model has no output matrix";
            return false;
        }
        head.type = h->type;
        if (device == main.device()) {
            head.p = h->ptr;
        } else {
            if (!alloc(&head.p, h->bytes)) {
                err = "dspark: out of device memory (output matrix)";
                return false;
            }
            std::vector<char> stage_buf((size_t) std::min<uint64_t>(h->bytes, 64ull << 20));
            for (uint64_t off = 0; off < h->bytes; off += stage_buf.size()) {
                const size_t n = (size_t) std::min<uint64_t>(stage_buf.size(), h->bytes - off);
                {
                    core::OnDevice on_main(main.device());
                    if (cudaMemcpy(stage_buf.data(), (const char*) h->ptr + off, n, cudaMemcpyDeviceToHost) != cudaSuccess) {
                        err = "dspark: cannot copy the output matrix";
                        return false;
                    }
                }
                if (cudaMemcpy((char*) head.p + off, stage_buf.data(), n, cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "dspark: cannot copy the output matrix";
                    return false;
                }
            }
        }
        mask_emb.resize(kN);
        if (!main.embedding_row(mask_id, mask_emb.data(), err)) return false;

        // scratch
        const size_t V = (size_t) main.geo().n_vocab;
        float tab[kRopePairs];
        rope_freq_table(RopeParams{}, tab);
        if (!upload(tab, sizeof(tab), (void**) &freq)) return false;
        rope = rope_table_window();
        ring_pos.assign((size_t) kWindowSlots, -1);
        const size_t Tn = (size_t) kT * kN;
        const size_t ff = (size_t) kN * target_layers.size();
        const size_t Rows = std::max<size_t>(kT, kInjectRows);
        bool ok = buf(x, 4 * Tn) && buf(x2, 4 * Tn) && buf(cur, Tn) && buf(normed, Tn) && buf(attn_o, Tn) &&
                  buf(ffn_sum, Tn) && buf(sh_out, Tn) && buf(coef_a, kT * kHcCoef) && buf(coef_f[0], kT * kHcCoef) &&
                  buf(coef_f[1], kT * kHcCoef) && buf(hc_ws, hc_mixes_workspace_floats(kT)) &&
                  buf(rlogits, (size_t) kT * kExperts) && buf(rw, kT * kUsed) && buf(kvraw, Rows * kLatent) &&
                  buf(blockkv, (size_t) kT * kLatent) && buf(qr, (size_t) kT * kAttnQRank) &&
                  buf(q, (size_t) kT * kAttnHeads * kAttnHeadDim) && buf(attn, (size_t) kT * kAttnHeads * kAttnHeadDim) &&
                  buf(oa, attn_out_scratch_floats(kT)) && buf(mws, sparse_mqa_workspace_floats(kT, kT) + 64) &&
                  buf(sh_g, (size_t) kT * kFF) && buf(sh_u, (size_t) kT * kFF) && buf(sh_h, (size_t) kT * kFF) &&
                  buf(moe_h, (size_t) kT * kUsed * kFF) && buf(moe_d, (size_t) kT * kUsed * kN) && buf(logits, kT * V) &&
                  buf(col, V) && buf(w1p, kMarkovR) && buf(dconf, kT) && buf(hcout, Tn) && buf(feat, kInjectRows * ff) &&
                  buf(mainx, kInjectRows * (size_t) kN) && buf(mainxn, kInjectRows * (size_t) kN);
        const size_t xq_bytes = std::max<size_t>({kernels::native_q8_1_bytes((int) ff, kInjectRows),
                                                   kernels::native_q8_1_bytes(kN, 8), attn_out_scratch_bytes(kT), 1 << 16});
        ok = ok && alloc(&xq, xq_bytes) && alloc(&xq2, std::max<size_t>(q_path_scratch_bytes(kT), 1 << 16)) &&
             alloc(&sh_q1, kernels::native_q8_1_bytes(kN, kT)) && alloc(&sh_q2, kernels::native_q8_1_bytes(kFF, kT)) &&
             alloc(&am_ws, argmax_workspace_bytes()) && buf(ids, kT * kUsed) && buf(dpos, kT) && buf(dstart, kT) &&
             buf(dn, kT) && buf(dlim, kT) && buf(dsel, kT * kT) && buf(dtok, kT + 1) && buf(dpos1, kInjectRows);
        if (!ok) {
            err = "dspark: out of device memory (scratch)";
            return false;
        }
        if (cudaHostAlloc((void**) &h_feat, (size_t) kInjectRows * ff * 4, cudaHostAllocPortable) != cudaSuccess ||
            cudaHostAlloc((void**) &h_pos, kInjectRows * 4, cudaHostAllocPortable) != cudaSuccess ||
            cudaHostAlloc((void**) &h_emb, (size_t) 4 * Tn * 4, cudaHostAllocPortable) != cudaSuccess ||
            cudaHostAlloc((void**) &h_meta, 64 * 4, cudaHostAllocPortable) != cudaSuccess ||
            cudaHostAlloc((void**) &h_out, 8 * 4, cudaHostAllocPortable) != cudaSuccess ||
            cudaHostAlloc((void**) &h_conf, 8 * 4, cudaHostAllocPortable) != cudaSuccess) {
            err = "dspark: cannot allocate pinned staging";
            return false;
        }
        if (cudaStreamCreate(&st) != cudaSuccess || cudaEventCreate(&ev0) != cudaSuccess ||
            cudaEventCreate(&ev_body) != cudaSuccess || cudaEventCreate(&ev1) != cudaSuccess) {
            err = "dspark: cannot create a stream";
            return false;
        }
        return true;
    }

    // Window latents of n <= kInjectRows positions from the staged features.
    void inject_staged(int n) {
        const size_t ff = (size_t) kN * target_layers.size();
        cudaMemcpyAsync(feat, h_feat, (size_t) n * ff * 4, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(dpos1, h_pos, (size_t) n * 4, cudaMemcpyHostToDevice, st);
        kernels::native_quantize_q8_1(feat, xq, (int) ff, n, st);
        kernels::native_mmvq(fc.type, fc.p, xq, mainx, (int) ff, kN, n, st);
        rms_norm_rows(mainx, enc_norm, n, kN, kEps, mainxn, st);
        kernels::native_quantize_q8_1(mainxn, xq, kN, n, st);
        for (int l = 0; l < kStages; ++l) {
            kernels::native_mmvq(stage[l].wkv.type, stage[l].wkv.p, xq, kvraw, kN, kLatent, n, st);
            dspark_kv_rope(kvraw, stage[l].kv_norm, dpos1, n, kEps, freq, nullptr, stage[l].ring, kWindowSlots, st);
        }
    }

    void run_block() {
        HcArgs hca;
        hca.n_embd = kN;
        hca.rms_eps = kEps;
        hca.hc_eps = 1e-6f;
        hca.sinkhorn_iters = 20;
        RouterArgs ra;
        ra.n_expert = kExperts;
        ra.n_embd = kN;
        ra.k = kUsed;
        ra.weight_scale = 1.5f;
        const size_t V = (size_t) main.geo().n_vocab;

        cudaMemcpyAsync(x, h_emb, (size_t) 4 * kT * kN * 4, cudaMemcpyHostToDevice, st);
        for (int l = 0; l < kStages; ++l) {
            const Stage& s = stage[l];
            hc_mixes(x, s.hc_attn_fn, s.hc_attn_scale, s.hc_attn_base, kT, hca, hc_ws, coef_a, st);
            if (l == 0) {
                cudaMemcpy2DAsync(cur, kN * 4, x, 4 * kN * 4, kN * 4, kT, cudaMemcpyDeviceToDevice, st);
                rms_norm_rows(cur, s.attn_norm, kT, kN, kEps, normed, st);
            } else {
                hc_pre_rms_norm(x, coef_f[(l - 1) & 1], kHcCoef, s.attn_norm, kT, kN, kEps, cur, normed, st);
            }
            kernels::native_quantize_q8_1(normed, xq, kN, kT, st);
            kernels::native_mmvq(s.wkv.type, s.wkv.p, xq, kvraw, kN, kLatent, kT, st);
            dspark_kv_rope(kvraw, s.kv_norm, dpos, kT, kEps, freq, blockkv, nullptr, 0, st);
            QPathWeights qw{s.wq_a.p, s.wq_a.type, s.q_norm, s.wq_b.p, s.wq_b.type, kEps};
            q_path(xq, qw, kT, dpos, rope, xq2, qr, q, st);
            MqaArgs a;
            a.n_tokens = kT;
            a.q = q;
            a.win = s.ring;
            a.win_type = LatentType::F16;
            a.win_cap = kWindowSlots;
            a.win_start = dstart;
            a.win_n = dn;
            a.comp = blockkv;
            a.comp_type = LatentType::F32;
            a.comp_stride = kLatent;
            a.sel = dsel;
            a.n_sel = kT;
            a.comp_limit = dlim;
            a.sinks = s.sinks;
            a.pos = dpos;
            a.rope = rope;
            a.out = attn;
            sparse_mqa(a, mws, st);
            AttnOutWeights ow{s.wo_a.p, s.wo_a.type, s.wo_b.p, s.wo_b.type};
            attn_out(attn, ow, kT, xq, oa, attn_o, st);
            hc_post(attn_o, x, coef_a, kHcCoef, kT, kN, x2, st);

            hc_mixes(x2, s.hc_ffn_fn, s.hc_ffn_scale, s.hc_ffn_base, kT, hca, hc_ws, coef_f[l & 1], st);
            hc_pre_rms_norm(x2, coef_a, kHcCoef, s.ffn_norm, kT, kN, kEps, cur, normed, st);
            ra.n_expert = s.n_exp;
            router(normed, s.router, s.router_bias, kT, ra, rlogits, ids, rw, st);
            dspark_moe_gate_up(s.gate_exps.p, s.up_exps.p, kN, kFF, normed, ids, kT, kUsed, 10.f, moe_h, st);
            dspark_moe_down(s.down_exps.p, kFF, kN, moe_h, ids, kT, kUsed, moe_d, st);
            kernels::native_quantize_q8_1(normed, sh_q1, kN, kT, st);
            kernels::native_mmvq(s.sh_gate.type, s.sh_gate.p, sh_q1, sh_g, kN, kFF, kT, st);
            kernels::native_mmvq(s.sh_up.type, s.sh_up.p, sh_q1, sh_u, kN, kFF, kT, st);
            swiglu_clamped(sh_g, sh_u, kT * kFF, 10.f, sh_h, st);
            kernels::native_quantize_q8_1(sh_h, sh_q2, kFF, kT, st);
            kernels::native_mmvq(s.sh_down.type, s.sh_down.p, sh_q2, sh_out, kFF, kN, kT, st);
            dspark_moe_combine(moe_d, rw, sh_out, kT, kUsed, kN, ffn_sum, st);
            hc_post(ffn_sum, x2, coef_f[l & 1], kHcCoef, kT, kN, x, st);
        }
        // collapse with the last FFN's pre-mix (no head mix), norm, the main model's output matrix
        hc_pre(x, coef_f[(kStages - 1) & 1], kHcCoef, kT, kN, hcout, st);
        rms_norm_rows(hcout, out_norm, kT, kN, kEps, normed, st);
        cudaEventRecord(ev_body, st);
        kernels::native_quantize_q8_1(normed, xq, kN, kT, st);
        kernels::native_mmvq(head.type, head.p, xq, logits, kN, (int) V, kT, st);

        // the Markov chain: row i is biased by the token before it
        const int32_t* tok0 = dtok;   // [0] = anchor, [i + 1] = the argmax of row i
        for (int i = 0; i < kT; ++i) {
            dspark_markov_gather(w1, tok0 + i, w1p, st);
            dspark_markov_bias(w2, w1p, logits + (size_t) i * V, (int) V, col, st);
            argmax_index(col, (int) V, dtok + i + 1, am_ws, st);
            dspark_conf(conf_w, hcout + (size_t) i * kN, w1p, kN, dconf + i, st);
        }
    }
};

DsparkDrafter::DsparkDrafter(std::unique_ptr<Impl> i) : p_(std::move(i)) {}
DsparkDrafter::~DsparkDrafter() = default;

std::vector<std::vector<int>> dspark_kept_experts(const std::vector<std::pair<int32_t, int32_t>>& ranked, int stages, int experts,
                                                  int keep) {
    std::vector<std::vector<int>> out((size_t) stages);
    std::vector<uint8_t> taken((size_t) stages * (size_t) experts, 0);
    const auto take = [&](int s, int e) {
        if (s < 0 || s >= stages || e < 0 || e >= experts || (int) out[(size_t) s].size() >= keep) return;
        uint8_t& t = taken[(size_t) s * (size_t) experts + (size_t) e];
        if (t) return;
        t = 1;
        out[(size_t) s].push_back(e);
    };
    for (const auto& [s, e] : ranked) take(s, e);
    for (int s = 0; s < stages; ++s) {
        for (int e = 0; e < experts; ++e) take(s, e);
        std::sort(out[(size_t) s].begin(), out[(size_t) s].end());
    }
    return out;
}

std::unique_ptr<DsparkDrafter> DsparkDrafter::create(const DsModel& main, int device, const std::string& ranking, int keep,
                                                     std::string& err) {
    core::OnDevice g(device);
    auto i = std::make_unique<Impl>(main);
    i->device = device;
    try {
        if (!i->load(main.pack().dspark, ranking, keep, err)) return nullptr;
    } catch (const std::exception& e) {
        err = std::string("dspark: ") + e.what();
        return nullptr;
    }
    return std::unique_ptr<DsparkDrafter>(new DsparkDrafter(std::move(i)));
}

int DsparkDrafter::feature_floats() const { return kN * (int) p_->target_layers.size(); }
int DsparkDrafter::feature_slot(int il) const {
    for (size_t k = 0; k < p_->target_layers.size(); ++k)
        if (p_->target_layers[k] == il) return (int) k;
    return -1;
}
int DsparkDrafter::device() const { return p_->device; }
uint64_t DsparkDrafter::device_bytes() const { return p_->bytes_taken; }
const std::string& DsparkDrafter::kept() const { return p_->kept_note; }

void DsparkDrafter::forget_from(int64_t pos) {
    for (int64_t& rp : p_->ring_pos)
        if (rp >= pos) rp = -1;
}

void DsparkDrafter::reset() { std::fill(p_->ring_pos.begin(), p_->ring_pos.end(), -1); }

void DsparkDrafter::report() const {
    const Impl& d = *p_;
    if (d.blocks == 0) return;
    basalt::log::info("basalt deepseek41", "drafter: %lld blocks, %.2f ms each (stages %.2f, head %.2f)", (long long) d.blocks,
                      (d.ms_body + d.ms_head) / (double) d.blocks, d.ms_body / (double) d.blocks, d.ms_head / (double) d.blocks);
}

bool DsparkDrafter::inject(int64_t pos0, int n_rows, const float* features, std::string& err) {
    core::OnDevice g(p_->device);
    Impl& d = *p_;
    try {
        const size_t ff = (size_t) feature_floats();
        for (int r = 0; r < n_rows; r += kInjectRows) {
            const int n = std::min(kInjectRows, n_rows - r);
            cudaStreamSynchronize(d.st);   // the staging is reused
            std::memcpy(d.h_feat, features + (size_t) r * ff, (size_t) n * ff * 4);
            for (int j = 0; j < n; ++j) {
                const int64_t p = pos0 + r + j;
                d.h_pos[j] = (int32_t) p;
                d.ring_pos[(size_t) (p % kWindowSlots)] = p;
            }
            d.inject_staged(n);
        }
    } catch (const std::exception& e) {
        err = std::string("dspark inject: ") + e.what();
        return false;
    }
    return true;
}

bool DsparkDrafter::draft_begin(int32_t anchor, int64_t pos, std::string& err) {
    core::OnDevice g(p_->device);
    Impl& d = *p_;
    try {
        // the block's host inputs: four hyper-connection copies of each row's embedding, then the index arrays
        cudaStreamSynchronize(d.st);   // the staging and the metadata block are reused
        std::vector<float> emb(kN);
        if (!d.main.embedding_row(anchor, emb.data(), err)) return false;
        for (int t = 0; t < kT; ++t) {
            const float* src = t == 0 ? emb.data() : d.mask_emb.data();
            for (int s = 0; s < 4; ++s) std::memcpy(d.h_emb + ((size_t) t * 4 + s) * kN, src, (size_t) kN * 4);
        }
        // the window is the run of positions before `pos` the rings hold
        int64_t start = pos;
        while (start > 0 && pos - start < kDsparkWindow && d.ring_pos[(size_t) ((start - 1) % kWindowSlots)] == start - 1) --start;
        int32_t* m = d.h_meta;
        for (int t = 0; t < kT; ++t) {
            m[t] = (int32_t) (pos + t);
            m[kT + t] = (int32_t) start;
            m[2 * kT + t] = (int32_t) (pos - start);
            m[3 * kT + t] = kT;
            for (int j = 0; j < kT; ++j) m[4 * kT + t * kT + j] = j;
        }
        m[4 * kT + kT * kT] = anchor;
        cudaEventRecord(d.ev0, d.st);
        cudaMemcpyAsync(d.dpos, m, kT * 4, cudaMemcpyHostToDevice, d.st);
        cudaMemcpyAsync(d.dstart, m + kT, kT * 4, cudaMemcpyHostToDevice, d.st);
        cudaMemcpyAsync(d.dn, m + 2 * kT, kT * 4, cudaMemcpyHostToDevice, d.st);
        cudaMemcpyAsync(d.dlim, m + 3 * kT, kT * 4, cudaMemcpyHostToDevice, d.st);
        cudaMemcpyAsync(d.dsel, m + 4 * kT, kT * kT * 4, cudaMemcpyHostToDevice, d.st);
        cudaMemcpyAsync(d.dtok, m + 4 * kT + kT * kT, 4, cudaMemcpyHostToDevice, d.st);
        d.run_block();
        cudaMemcpyAsync(d.h_out, d.dtok + 1, kT * 4, cudaMemcpyDeviceToHost, d.st);
        cudaMemcpyAsync(d.h_conf, d.dconf, kT * 4, cudaMemcpyDeviceToHost, d.st);
        cudaEventRecord(d.ev1, d.st);
        d.block_live = true;
    } catch (const std::exception& e) {
        err = std::string("dspark draft: ") + e.what();
        return false;
    }
    return true;
}

bool DsparkDrafter::draft_end(DsparkDraft& out, std::string& err) {
    core::OnDevice g(p_->device);
    Impl& d = *p_;
    if (!d.block_live) {
        err = "dspark draft: no block was started";
        return false;
    }
    d.block_live = false;
    const cudaError_t e = cudaStreamSynchronize(d.st);
    if (e != cudaSuccess) {
        err = std::string("dspark draft: ") + cudaGetErrorString(e);
        return false;
    }
    float body = 0, all = 0;
    cudaEventElapsedTime(&body, d.ev0, d.ev_body);
    cudaEventElapsedTime(&all, d.ev0, d.ev1);
    ++d.blocks;
    d.ms_body += body;
    d.ms_head += all - body;
    for (int i = 0; i < kT; ++i) {
        out.token[i] = d.h_out[i];
        out.conf[i] = d.h_conf[i];
    }
    return true;
}

}  // namespace basalt::models::deepseek41
