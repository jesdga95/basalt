// src/models/ernie55/pack.cpp - reads an ERNIE-5.5-Flash .basalt pack's meta and experts.layout sections.
#include "models/ernie55/pack.hpp"

#include "basalt/ggml/ggml.h"
#include "basalt/models/ernie55/cache_layout.hpp"
#include "basalt/platform/file.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace basalt::models::ernie55 {
namespace {

using core::Json;

bool req_int(const Json& obj, const std::string& key, int& out, const std::string& path, std::string& err) {
    const Json* j = obj.get(key);
    if (!j || !j->is_number()) {
        err = path + ": missing or non-numeric meta key " + key;
        return false;
    }
    out = (int) j->num();
    return true;
}

bool req_i64(const Json& obj, const std::string& key, int64_t& out, const std::string& path, std::string& err) {
    const Json* j = obj.get(key);
    if (!j || !j->is_number()) {
        err = path + ": missing or non-numeric meta key " + key;
        return false;
    }
    out = (int64_t) j->num();
    return true;
}

bool req_u64(const Json& obj, const std::string& key, uint64_t& out, const std::string& path, std::string& err) {
    const Json* j = obj.get(key);
    if (!j || !j->is_number()) {
        err = path + ": missing or non-numeric meta key " + key;
        return false;
    }
    out = j->u64();
    return true;
}

bool req_float(const Json& obj, const std::string& key, float& out, const std::string& path, std::string& err) {
    const Json* j = obj.get(key);
    if (!j || !j->is_number()) {
        err = path + ": missing or non-numeric meta key " + key;
        return false;
    }
    out = (float) j->num();
    return true;
}

bool req_bool(const Json& obj, const std::string& key, bool& out, const std::string& path, std::string& err) {
    const Json* j = obj.get(key);
    if (!j || (j->type != Json::Type::Bool && !j->is_number())) {
        err = path + ": missing or non-boolean meta key " + key;
        return false;
    }
    out = j->is_number() ? j->num() != 0 : j->b;
    return true;
}

bool req_ints(const Json& obj, const std::string& key, std::vector<int>& out, const std::string& path,
              std::string& err) {
    const Json* j = obj.get(key);
    if (!j || !j->is_array()) {
        err = path + ": missing or non-array meta key " + key;
        return false;
    }
    out.clear();
    out.reserve(j->items.size());
    for (const Json& x : j->items) {
        if (!x.is_number()) {
            err = path + ": non-numeric element in meta key " + key;
            return false;
        }
        out.push_back((int) x.num());
    }
    return true;
}

bool req_unsigneds(const Json& obj, const std::string& key, std::vector<uint64_t>& out, const std::string& path,
                   std::string& err) {
    const Json* j = obj.get(key);
    if (!j || !j->is_array()) {
        err = path + ": missing or non-array meta key " + key;
        return false;
    }
    out.clear();
    out.reserve(j->items.size());
    for (const Json& x : j->items) {
        if (!x.is_number()) {
            err = path + ": non-numeric element in meta key " + key;
            return false;
        }
        out.push_back(x.u64());
    }
    return true;
}

bool check_dim(int got, int want, const char* name, const std::string& path, std::string& err) {
    if (got == want) return true;
    err = path + ": unsupported " + name + "=" + std::to_string(got) + " (kernels require " + std::to_string(want) + ")";
    return false;
}

}  // namespace

bool f32_range_is_bf16(int fd, uint64_t off, uint64_t bytes, bool& exact, std::string& err) {
    exact = false;
    if (fd < 0 || bytes == 0 || (bytes % 4) != 0) return true;
    constexpr size_t kChunk = 256 * 1024;
    std::vector<uint32_t> buf(kChunk / sizeof(uint32_t));
    uint64_t done = 0;
    while (done < bytes) {
        const size_t want = (size_t) std::min<uint64_t>(kChunk, bytes - done);
        if (!basalt::platform::pread_all(fd, buf.data(), want, off + done)) {
            err = "failed to read F32 tensor while verifying BF16 representation";
            return false;
        }
        const size_t n = want / sizeof(uint32_t);
        for (size_t i = 0; i < n; ++i) {
            if ((buf[i] & 0xFFFFu) != 0) {
                exact = false;
                return true;
            }
        }
        done += want;
    }
    exact = true;
    return true;
}

bool open_pack(const std::string& path, Pack& p, std::string& err) {
    if (!p.file.open(path, err) || !p.file.load_meta(err)) return false;
    const Json& meta = p.file.meta();
    const Json* arch = meta.get("arch");
    if (!arch || !arch->is_string() || arch->s != "ernie55") {
        err = path + ": not an ernie55 pack";
        return false;
    }
    const Json* kv = meta.get("gguf_kv");
    const Json* tj = meta.get("tensors");
    if (!kv || !kv->is_object() || !tj || !tj->is_object()) {
        err = path + ": meta has no gguf_kv / tensors";
        return false;
    }
    auto ek = [](const char* s) { return std::string("ernie55.") + s; };
    auto ri = [&](const char* s, int& out) { return req_int(*kv, ek(s), out, path, err); };
    auto ri64 = [&](const char* s, int64_t& out) { return req_i64(*kv, ek(s), out, path, err); };
    auto rf = [&](const char* s, float& out) { return req_float(*kv, ek(s), out, path, err); };
    auto rb = [&](const char* s, bool& out) { return req_bool(*kv, ek(s), out, path, err); };

    Geometry& g = p.geo;
    if (!ri("block_count", g.n_layers) || !ri("first_k_dense_replace", g.n_dense) ||
        !ri("embedding_length", g.n_embd) || !ri("feed_forward_length", g.n_ff) ||
        !ri64("context_length", g.context_length) || !rf("attention.layer_norm_rms_epsilon", g.rms_eps) ||
        !ri("attention.sliding_window", g.window) ||
        !req_ints(*kv, ek("attention.compress_ratios"), g.compress_ratios, path, err))
        return false;

    if (!ri("attention.head_count", g.hca_heads) || !ri("attention.head_count_kv", g.hca_kv_heads) ||
        !ri("attention.key_length", g.hca_head_dim) || !ri("attention.value_length", g.hca_v_dim) ||
        !ri("attention.q_lora_rank", g.hca_q_lora) || !ri("attention.kv_lora_rank", g.hca_kv_lora) ||
        !ri("attention.output_group_count", g.hca_o_groups) || !ri("attention.output_lora_rank", g.hca_o_lora) ||
        !ri("rope.dimension_count", g.hca_rope_dim) || !rf("rope.freq_base", g.rope_theta) ||
        !rf("attention.compress_rope_freq_base", g.comp_rope_theta) ||
        !rf("rope.scaling.factor", g.yarn_factor) || !ri("rope.scaling.original_context_length", g.yarn_orig_ctx) ||
        !rf("rope.scaling.beta_fast", g.yarn_beta_fast) || !rf("rope.scaling.beta_slow", g.yarn_beta_slow))
        return false;

    if (!ri("mla.head_count", g.mla_heads) || !ri("mla.q_lora_rank", g.mla_q_lora) ||
        !ri("mla.kv_lora_rank", g.mla_kv_lora) || !ri("mla.qk_nope_head_dim", g.mla_qk_nope) ||
        !ri("mla.qk_rope_head_dim", g.mla_qk_rope) || !ri("mla.v_head_dim", g.mla_v_dim) ||
        !ri("attention.indexer.head_count", g.idx_heads) || !ri("attention.indexer.key_length", g.idx_dim) ||
        !ri("attention.indexer.top_k", g.idx_top_k))
        return false;

    if (!ri("expert_count", g.n_expert) || !ri("expert_used_count", g.n_used) ||
        !ri("expert_shared_count", g.n_shared) || !ri("expert_feed_forward_length", g.n_ff_exp) ||
        !ri("expert_latent_length", g.n_latent) || !rb("expert_weights_norm", g.norm_topk_prob) ||
        !rf("expert_gate_softcap", g.gate_softcap) || !rf("expert_up_softcap", g.up_softcap))
        return false;

    if (!ri("hyper_connection.count", g.hc_mult) || !ri("hyper_connection.sinkhorn_iterations", g.sinkhorn_iters) ||
        !rf("hyper_connection.epsilon", g.hc_eps) || !rb("has_flashindexer", g.has_flashindexer))
        return false;

    g.n_vocab = 0;
    if (const Json* t = kv->get("tokenizer.ggml.tokens")) g.n_vocab = (int) t->count;
    if (g.n_vocab <= 0 && !ri("vocab_size", g.n_vocab)) return false;

    // Validate general bounds and every dimension hard-coded in the CUDA kernels and KV cache layouts.
    if (g.n_layers <= 0 || (int) g.compress_ratios.size() != g.n_layers || g.n_embd <= 0 || g.n_ff <= 0 ||
        g.n_vocab <= 0 || g.context_length <= 0 || g.n_dense < 0 || g.n_dense > g.n_layers || g.window <= 0 ||
        g.hca_heads <= 0 || g.hca_o_groups <= 0 || (g.hca_heads % g.hca_o_groups) != 0 || g.mla_heads <= 0 ||
        g.n_used <= 0 || g.n_used > 32 || g.n_used > g.n_expert || g.n_ff_exp <= 0 || g.n_latent <= 0 ||
        g.idx_heads <= 0 || g.idx_heads > layout::kIdxMaxHeads || g.idx_top_k <= 0 || g.sinkhorn_iters <= 0) {
        err = path + ": incomplete or unsupported ernie55 geometry";
        return false;
    }
    if (!check_dim(g.hc_mult, layout::kHcMult, "hyper_connection.count", path, err) ||
        !check_dim(g.hca_head_dim, layout::kHcaHeadDim, "attention.key_length", path, err) ||
        !check_dim(g.hca_v_dim, layout::kHcaVDim, "attention.value_length", path, err) ||
        !check_dim(g.hca_kv_heads, layout::kHcaKvHeads, "attention.head_count_kv", path, err) ||
        !check_dim(g.hca_kv_lora, layout::kHcaKvLora, "attention.kv_lora_rank", path, err) ||
        !check_dim(g.hca_rope_dim, layout::kHcaRopeDim, "rope.dimension_count", path, err) ||
        !check_dim(g.mla_kv_lora, layout::kMlaKvLora, "mla.kv_lora_rank", path, err) ||
        !check_dim(g.mla_qk_nope, layout::kMlaQkNope, "mla.qk_nope_head_dim", path, err) ||
        !check_dim(g.mla_qk_rope, layout::kMlaQkRope, "mla.qk_rope_head_dim", path, err) ||
        !check_dim(g.mla_v_dim, layout::kMlaVDim, "mla.v_head_dim", path, err) ||
        !check_dim(g.idx_dim, layout::kIdxDim, "attention.indexer.key_length", path, err) ||
        !check_dim(g.n_expert, layout::kNumExperts, "expert_count", path, err))
        return false;

    int seen_ratio = 0;
    for (int r : g.compress_ratios) {
        if (r > 0) {
            if (seen_ratio == 0) seen_ratio = r;
            else if (r != seen_ratio) {
                err = path + ": non-uniform HCA compress_ratios (" + std::to_string(r) + " vs " +
                      std::to_string(seen_ratio) + ")";
                return false;
            }
        }
    }

    const core::ModelFile::Section* gguf = p.file.find("gguf");
    const core::ModelFile::Section* exps = p.file.find("experts");
    if (!gguf || !exps) {
        err = path + ": pack misses gguf / experts";
        return false;
    }
    p.experts_range = p.file.range("experts");

    p.tensors.clear();
    for (const auto& [name, val] : tj->members) {
        if (!val.is_object()) continue;
        TensorRef tr;
        tr.name = name;
        uint64_t rel = 0;
        if (!req_int(val, "type_id", tr.type, path + " tensor " + name, err) ||
            !req_unsigneds(val, "shape", tr.shape, path + " tensor " + name, err) ||
            !req_u64(val, "offset", rel, path + " tensor " + name, err) ||
            !req_u64(val, "bytes", tr.bytes, path + " tensor " + name, err))
            return false;
        if (tr.type < 0 || tr.bytes == 0 || rel + tr.bytes > gguf->bytes) {
            err = path + ": malformed tensor entry " + name;
            return false;
        }
        tr.offset = gguf->offset + rel;
        p.tensors.emplace(name, std::move(tr));
    }

    p.experts.assign((size_t) g.n_layers, ExpertLayer{});
    std::vector<uint8_t> layout_raw;
    if (!p.file.read("experts.layout", layout_raw, err)) return false;
    std::istringstream iss(std::string(layout_raw.begin(), layout_raw.end()));
    std::string line;
    while (std::getline(iss, line)) {
        if (line.empty() || line[0] == '#') continue;
        int l = -1, gu = 0, dn = 0;
        uint64_t off = 0, blob = 0;
        std::istringstream ls(line);
        if (!(ls >> l >> gu >> dn >> off >> blob)) {
            err = path + ": malformed experts.layout line: " + line;
            return false;
        }
        if (l < 0 || l >= g.n_layers) {
            err = path + ": experts.layout layer out of range: " + std::to_string(l);
            return false;
        }
        if (gu != GGML_TYPE_MXFP4 || dn != GGML_TYPE_MXFP4) {
            err = path + ": ernie55 expects MXFP4 experts (type 39), got " + std::to_string(gu) + "/" + std::to_string(dn);
            return false;
        }
        if (off + blob * (uint64_t) g.n_expert > p.experts_range.bytes) {
            err = path + ": experts.layout exceeds experts section at layer " + std::to_string(l);
            return false;
        }
        p.experts[(size_t) l] = {true, gu, dn, off, blob};
    }
    for (int l = g.n_dense; l < g.n_layers; ++l) {
        if (!p.experts[(size_t) l].present) {
            err = path + ": experts.layout misses MoE layer " + std::to_string(l);
            return false;
        }
    }

    // Optional bundled DSpark speculative drafter section (`dspark`)
    p.dspark = DsparkPack{};
    const core::ModelFile::Section* dsec = p.file.find("dspark");
    const Json* dj = meta.get("dspark");
    if (dsec != nullptr || (dj != nullptr && dj->is_object())) {
        if (!dsec || !dj || !dj->is_object()) {
            err = path + ": dspark section and meta.dspark must both be present";
            return false;
        }
        DsparkPack& dp = p.dspark;
        const std::string dctx = path + " (meta.dspark)";
        if (!req_int(*dj, "layers", dp.n_stages, dctx, err) ||
            !req_int(*dj, "block_size", dp.block_size, dctx, err) ||
            !req_int(*dj, "noise_token_id", dp.noise_token_id, dctx, err) ||
            !req_int(*dj, "markov_rank", dp.markov_rank, dctx, err) ||
            !req_int(*dj, "window", dp.window, dctx, err) ||
            !req_float(*dj, "rope_theta", dp.rope_theta, dctx, err) ||
            !req_ints(*dj, "target_layers", dp.target_layers, dctx, err))
            return false;
        const Json* dtj = dj->get("tensors");
        const Json* dej = dj->get("experts");
        if (dp.n_stages <= 0 || dp.block_size <= 0 || dp.window <= 0 || dp.markov_rank <= 0 ||
            dp.target_layers.empty() || !dtj || !dtj->is_object() || !dej || !dej->is_object()) {
            err = path + ": malformed meta.dspark";
            return false;
        }
        for (int tl : dp.target_layers) {
            if (tl < 0 || tl >= g.n_layers) {
                err = path + ": meta.dspark target_layer out of range";
                return false;
            }
        }
        for (const auto& [name, val] : dtj->members) {
            if (!val.is_object()) continue;
            TensorRef tr;
            tr.name = name;
            uint64_t rel = 0;
            if (!req_int(val, "type_id", tr.type, dctx + " tensor " + name, err) ||
                !req_unsigneds(val, "shape", tr.shape, dctx + " tensor " + name, err) ||
                !req_u64(val, "offset", rel, dctx + " tensor " + name, err) ||
                !req_u64(val, "bytes", tr.bytes, dctx + " tensor " + name, err))
                return false;
            if (tr.type < 0 || tr.bytes == 0 || rel + tr.bytes > dsec->bytes) {
                err = path + ": malformed dspark tensor " + name;
                return false;
            }
            tr.offset = dsec->offset + rel;
            dp.tensors.emplace(name, std::move(tr));
        }
        uint64_t erel = 0;
        int gu = 0, dn = 0;
        const std::string dectx = path + " (meta.dspark.experts)";
        if (!req_u64(*dej, "offset", erel, dectx, err) ||
            !req_u64(*dej, "bytes", dp.experts_bytes, dectx, err) ||
            !req_int(*dej, "n_expert", dp.n_expert, dectx, err) ||
            !req_u64(*dej, "blob_bytes", dp.blob_bytes, dectx, err) ||
            !req_u64(*dej, "stage_stride", dp.stage_stride, dectx, err) ||
            !req_int(*dej, "gu_type", gu, dectx, err) ||
            !req_int(*dej, "d_type", dn, dectx, err))
            return false;
        if (gu != GGML_TYPE_MXFP4 || dn != GGML_TYPE_MXFP4 || dp.n_expert != g.n_expert || dp.blob_bytes == 0 ||
            dp.stage_stride != (uint64_t) dp.n_expert * dp.blob_bytes ||
            dp.experts_bytes != (uint64_t) dp.n_stages * dp.stage_stride || erel + dp.experts_bytes > dsec->bytes) {
            err = path + ": malformed meta.dspark.experts";
            return false;
        }
        dp.experts_offset = dsec->offset + erel;
        dp.present = true;
    }
    return true;
}

}  // namespace basalt::models::ernie55
