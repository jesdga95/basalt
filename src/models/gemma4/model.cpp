// src/models/gemma4/model.cpp - loads a Gemma 4 pack: every weight to the main card, the embedding to mapped host memory,
// then the context sized from what the weights leave.
#include "models/gemma4/model.hpp"

#include "models/gemma4/prompt.hpp"

#include "basalt/core/gpu_mem_info.hpp"
#include "basalt/core/device_copy.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace basalt::models::gemma4 {
namespace {

constexpr uint64_t kGiB = 1ull << 30;
constexpr uint64_t kMiB = 1ull << 20;
constexpr int kTypeF32 = 0;
constexpr int kTypeBF16 = 30;
constexpr int kTypeNvfp4 = 40;
constexpr int kTypeF8 = 44;
// what each session needs beside its caches (its graphs and staging), and what the card keeps free for the kernels'
// modules and the driver
constexpr uint64_t kSessionFixed = 96 * kMiB;
constexpr uint64_t kCardSpare = 1024 * kMiB;
constexpr uint64_t kExpertSlack = 4096;   // zeroed bytes after a layer's last blob (the kernels' last loads)

// A blob's stride on the card: whole NVFP4 blocks (36 bytes) and 16 bytes, so a kernel reads each expert's blocks in place.
uint64_t expert_stride(uint64_t blob) {
    constexpr uint64_t unit = 144;   // lcm(36, 16)
    return (blob + unit - 1) / unit * unit;
}

uint64_t layer_cache_bytes(const Geometry& g, int l, int64_t context, KvType kv) {
    const LayerShape s = g.shape(l);
    const uint64_t cell = (uint64_t) s.n_kv * s.hd * kv_elem_bytes(kv) * (s.one ? 1 : 2);
    return cell * (s.window > 0 ? (uint64_t) s.window + kChunk : (uint64_t) context);
}

}  // namespace

uint64_t session_bytes(const Geometry& g, int64_t context, KvType kv) {
    uint64_t caches = 0;
    for (int l = 0; l < g.n_layers; ++l) caches += layer_cache_bytes(g, l, context, kv);
    return caches + kSessionFixed + (uint64_t) kRows * (uint64_t) g.n_vocab * 4;
}

uint64_t shared_bytes(const Geometry& g, int64_t n_ff_pad, int64_t context, KvType kv) {
    const uint64_t R = kRows, E = (uint64_t) g.n_embd;
    const uint64_t qmax = (uint64_t) g.n_head * std::max(g.hd_swa, g.hd_full);
    const uint64_t rowbuf = R * (4 * E + 3 * qmax + 3 * (uint64_t) n_ff_pad) * 4;
    const uint64_t experts = g.moe() ? R * (uint64_t) g.n_used * (E * 4 + (uint64_t) g.n_ff_exp * 16) + (64ull << 20) : 0;
    const uint64_t attn = R * (uint64_t) g.n_head * 64 * (std::max(g.hd_swa, g.hd_full) + 2) * 4;
    return rowbuf + experts + attn + R * (uint64_t) g.n_vocab * 4 + 256 * kMiB + PromptPath::bytes(g, n_ff_pad, context, kv);
}

Gemma4Model::Gemma4Model() = default;

Gemma4Model::~Gemma4Model() {
    for (void* p : dev_) cudaFree(p);
    if (embd_host_) cudaFreeHost(embd_host_);
    if (fd_ >= 0) platform::close_fd(fd_);
}

void* Gemma4Model::device_buf(uint64_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    dev_.push_back(p);
    return p;
}

float* Gemma4Model::small(const std::string& name, uint64_t floats, core::LoadPlan& reads, int file, std::string& err) {
    const TensorRef* t = pack_.find(name);
    if (!t || t->type != kTypeF32 || t->bytes != floats * 4) {
        err = name + (t ? " is not F32 of the expected size" : " is missing");
        return nullptr;
    }
    void* p = device_buf(t->bytes);
    if (!p) {
        err = "out of device memory loading " + name;
        return nullptr;
    }
    core::LoadItem it;
    it.file = file;
    it.off = t->offset;
    it.bytes = t->bytes;
    it.dev = p;
    it.device = device_;
    reads.add(it);
    return static_cast<float*>(p);
}

bool Gemma4Model::f8(const std::string& name, int64_t rows, int64_t K, core::LoadPlan& reads, int file, F8Mat& m,
                     std::string& err, bool per_row) {
    const TensorRef* t = pack_.find(name + ".weight");
    if (!t || t->type != kTypeF8 || t->bytes != (uint64_t) rows * K || t->shape.size() != 2 || (int64_t) t->shape[0] != K) {
        err = name + ".weight" + (t ? " is not F8_E4M3 [" + std::to_string(rows) + "][" + std::to_string(K) + "]" : " is missing");
        return false;
    }
    void* p = device_buf(t->bytes);
    if (!p) {
        err = "out of device memory loading " + name;
        return false;
    }
    core::LoadItem it;
    it.file = file;
    it.off = t->offset;
    it.bytes = t->bytes;
    it.dev = p;
    it.device = device_;
    reads.add(it);
    m.w = static_cast<uint8_t*>(p);
    m.rows = rows;
    m.K = K;
    m.scale = small(name + ".scale", per_row ? (uint64_t) rows : 1, reads, file, err);
    return m.scale != nullptr;
}

model::Result Gemma4Model::plan(const model::LoadRequest& req, core::LoadPlan& reads,
                                std::vector<std::unique_ptr<model::ModelSession>>&) {
    if (req.paths.empty()) return {model::Status::Error, "gemma4: no model path"};
    std::string err;
    const std::string& path = req.paths.front();
    if (!open_pack(path, pack_, err)) return {model::Status::Error, "gemma4: " + err};
    flags_ = req;
    cudaGetDevice(&device_);
    kv_ = req.kv == "fp8" ? KvType::F8 : KvType::F16;
    const Geometry& g = pack_.geo;
    desc_.arch = "gemma4";
    desc_.n_layers = g.n_layers;
    desc_.n_embd = g.n_embd;
    desc_.n_vocab = g.n_vocab;
    desc_.n_expert = g.n_expert;
    desc_.n_expert_used = g.n_used;
    desc_.n_ff_exp = g.n_ff_exp;
    desc_.context_length = g.context_length;
    for (const ExpertLayer& e : pack_.experts) {
        desc_.expert_gate_up_type.push_back(e.gu_type);
        desc_.expert_down_type.push_back(e.d_type);
    }
    fd_ = platform::open_fd_utf8(path);
    if (fd_ < 0) return {model::Status::Error, "gemma4: cannot open " + path};
    const int file = reads.file_fd(fd_);
    auto fail = [&](const std::string& e) { return model::Result{model::Status::Error, "gemma4: " + e}; };
    const int64_t E = g.n_embd, V = g.n_vocab;

    // the token embedding: BF16 rows in mapped host memory, gathered by the card
    {
        const TensorRef* t = pack_.find("token_embd.weight");
        if (!t || t->type != kTypeBF16 || t->bytes != (uint64_t) V * E * 2) return fail("token_embd.weight is not BF16 [n_embd][n_vocab]");
        if (cudaHostAlloc((void**) &embd_host_, t->bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
            return fail("cannot pin the token embedding's host memory");
        if (cudaHostGetDevicePointer((void**) &embd_dev_, embd_host_, 0) != cudaSuccess) return fail("cannot map the token embedding");
        core::LoadItem it;
        it.file = file;
        it.off = t->offset;
        it.bytes = t->bytes;
        it.host = reinterpret_cast<uint8_t*>(embd_host_);
        reads.add(it);
    }
    if (!f8("output", V, E, reads, file, head_, err, true)) return fail(err);
    if (!(output_norm_ = small("output_norm.weight", (uint64_t) E, reads, file, err))) return fail(err);
    {   // the router input's unweighted norm
        std::vector<float> one((size_t) E, 1.f);
        if (!(ones_ = (float*) device_buf((uint64_t) E * 4)) ||
            core::device_copy(ones_, one.data(), one.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("out of device memory for the small tables");
    }
    // the rotary tables of the two layer kinds, for the longest context the model takes
    const int64_t want = req.max_context > 0 ? req.max_context : kDefaultContext;
    for (int kind = 0; kind < 2; ++kind) {
        const int l = (int) (std::find(g.swa.begin(), g.swa.end(), kind == 0) - g.swa.begin()) % g.n_layers;
        const LayerShape sh = g.shape(l);
        std::vector<float> lo(rope_lo_floats(sh.hd)), hi(rope_hi_floats(sh.hd, want));
        rope_tables(sh.hd, sh.rot, kind == 0 ? g.rope_base_swa : g.rope_base, want, lo.data(), hi.data());
        float* dl = (float*) device_buf(lo.size() * 4);
        float* dh = (float*) device_buf(hi.size() * 4);
        if (!dl || !dh || core::device_copy(dl, lo.data(), lo.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
            core::device_copy(dh, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("out of device memory for the rotary tables");
        (kind == 0 ? rope_swa_ : rope_full_) = {dl, dh};
    }
    {
        const TensorRef* t = pack_.find("blk.0.ffn_down.weight");
        if (!t || t->shape.size() != 2) return fail("blk.0.ffn_down.weight is missing");
        n_ff_pad_ = (int64_t) t->shape[0];
        if (n_ff_pad_ < g.n_ff || n_ff_pad_ % 128) return fail("the MLP's padded width is not a multiple of 128 at least its width");
    }

    uint64_t dense = (uint64_t) V * E, experts = 0;
    layers_.assign((size_t) g.n_layers, LayerWeights{});
    for (int l = 0; l < g.n_layers; ++l) {
        LayerWeights& w = layers_[(size_t) l];
        const std::string b = "blk." + std::to_string(l) + ".";
        w.sh = g.shape(l);
        w.rope = w.sh.window > 0 ? rope_swa_ : rope_full_;
        const int64_t qd = (int64_t) w.sh.n_head * w.sh.hd, kd = (int64_t) w.sh.n_kv * w.sh.hd;
        auto S = [&](const char* suffix, uint64_t floats) { return small(b + suffix, floats, reads, file, err); };
        if (!f8(b + "attn_q", qd, E, reads, file, w.q, err) || !f8(b + "attn_k", kd, E, reads, file, w.k, err) ||
            (!w.sh.one && !f8(b + "attn_v", kd, E, reads, file, w.v, err)) ||
            !f8(b + "attn_output", E, qd, reads, file, w.o, err) ||
            !f8(b + "ffn_gate", n_ff_pad_, E, reads, file, w.gate, err) || !f8(b + "ffn_up", n_ff_pad_, E, reads, file, w.up, err) ||
            !f8(b + "ffn_down", E, n_ff_pad_, reads, file, w.down, err))
            return fail(err);
        dense += (uint64_t) (qd + 2 * kd) * E + (uint64_t) qd * E + 3ull * n_ff_pad_ * E;
        if (!(w.attn_norm = S("attn_norm.weight", (uint64_t) E)) || !(w.q_norm = S("attn_q_norm.weight", (uint64_t) w.sh.hd)) ||
            !(w.k_norm = S("attn_k_norm.weight", (uint64_t) w.sh.hd)) || !(w.post_attn = S("post_attention_norm.weight", (uint64_t) E)) ||
            !(w.ffn_norm = S("ffn_norm.weight", (uint64_t) E)) || !(w.post_ffw = S("post_ffw_norm.weight", (uint64_t) E)) ||
            !(w.scalar = S("layer_scalar", 1)))
            return fail(err);
        if (!g.moe()) continue;
        if (!(w.post_ffw1 = S("post_ffw_norm_1.weight", (uint64_t) E)) || !(w.post_ffw2 = S("post_ffw_norm_2.weight", (uint64_t) E)) ||
            !(w.pre_ffw2 = S("pre_ffw_norm_2.weight", (uint64_t) E)) ||
            !(w.router = S("ffn_gate_inp.weight", (uint64_t) g.n_expert * E)))
            return fail(err);
        const ExpertLayer& el = pack_.experts[(size_t) l];
        if (el.gu_type != kTypeNvfp4 || el.d_type != kTypeNvfp4) return fail("experts of a type other than NVFP4 at layer " + std::to_string(l));
        w.gu_type = el.gu_type;
        w.d_type = el.d_type;
        w.blob_bytes = el.blob_bytes;
        w.expert_stride = expert_stride(el.blob_bytes);
        const uint64_t layer_b = (uint64_t) g.n_expert * w.expert_stride + kExpertSlack;
        if (!(w.experts = (uint8_t*) device_buf(layer_b)) || !(w.blob_table = (unsigned long long*) device_buf((uint64_t) g.n_expert * 8)))
            return fail("out of device memory loading the experts of layer " + std::to_string(l));
        cudaMemset(w.experts, 0, layer_b);
        std::vector<unsigned long long> table((size_t) g.n_expert);
        for (int e = 0; e < g.n_expert; ++e) {
            core::LoadItem it;
            it.file = file;
            it.off = pack_.experts_range.offset + el.offset + (uint64_t) e * el.blob_bytes;
            it.bytes = el.blob_bytes;
            it.dev = w.experts + (uint64_t) e * w.expert_stride;
            it.device = device_;
            reads.add(it);
            table[(size_t) e] = (unsigned long long) (uintptr_t) it.dev;
        }
        if (core::device_copy(w.blob_table, table.data(), table.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("cannot write the blob table of layer " + std::to_string(l));
        experts += layer_b;
    }

    // the assistant drafter, when the pack has one and the verify window drafts
    if (req.spec >= 2 && g.drafter.present) {
        const DrafterGeometry& dg = g.drafter;
        auto d = std::make_unique<DrafterWeights>();
        const int64_t DE = dg.n_embd;
        d->n_embd = dg.n_embd;
        {
            const TensorRef* t = pack_.find("assistant.blk.0.ffn_down.weight");
            if (!t || t->shape.size() != 2) return fail("assistant.blk.0.ffn_down.weight is missing");
            d->n_ff_pad = (int64_t) t->shape[0];
        }
        if (!f8("assistant.pre_proj", DE, 2 * E, reads, file, d->pre, err) || !f8("assistant.post_proj", E, DE, reads, file, d->post, err) ||
            !f8("assistant.head", V, DE, reads, file, d->head, err, true) ||
            !(d->out_norm = small("assistant.output_norm.weight", (uint64_t) DE, reads, file, err)))
            return fail(err);
        int last_swa = -1, last_full = -1;
        for (int l = 0; l < g.n_layers; ++l) (g.swa[(size_t) l] ? last_swa : last_full) = l;
        for (int l = 0; l < dg.n_layers; ++l) {
            DrafterLayer w;
            const bool sw = dg.swa[(size_t) l];
            w.source = sw ? last_swa : last_full;
            if (w.source < 0) return fail("the drafter reads a kind of layer the model does not have");
            w.sh = g.shape(w.source);
            w.sh.n_head = dg.n_head;
            w.rope = sw ? rope_swa_ : rope_full_;
            const std::string b = "assistant.blk." + std::to_string(l) + ".";
            const int64_t qd = (int64_t) w.sh.n_head * w.sh.hd;
            auto S = [&](const char* suffix, uint64_t floats) { return small(b + suffix, floats, reads, file, err); };
            if (!f8(b + "attn_q", qd, DE, reads, file, w.q, err) || !f8(b + "attn_output", DE, qd, reads, file, w.o, err) ||
                !f8(b + "ffn_gate", d->n_ff_pad, DE, reads, file, w.gate, err) || !f8(b + "ffn_up", d->n_ff_pad, DE, reads, file, w.up, err) ||
                !f8(b + "ffn_down", DE, d->n_ff_pad, reads, file, w.down, err) ||
                !(w.attn_norm = S("attn_norm.weight", (uint64_t) DE)) || !(w.q_norm = S("attn_q_norm.weight", (uint64_t) w.sh.hd)) ||
                !(w.post_attn = S("post_attention_norm.weight", (uint64_t) DE)) || !(w.ffn_norm = S("ffn_norm.weight", (uint64_t) DE)) ||
                !(w.post_ffw = S("post_ffw_norm.weight", (uint64_t) DE)) || !(w.scalar = S("layer_scalar", 1)))
                return fail(err);
            d->layers.push_back(w);
        }
        drafter_ = std::move(d);
    }

    // the context: what the weights leave, less the card's spare
    if (want < 1024 || (g.context_length > 0 && want > g.context_length))
        return fail("--max-context " + std::to_string(want) + " is outside 1,024 .. " + std::to_string(g.context_length));
    size_t free_b = 0, total_b = 0;
    basalt::core::gpu_mem_info(&free_b, &total_b);
    const uint64_t spare = req.vram_reserve_mib > 0 ? (uint64_t) req.vram_reserve_mib * kMiB : kCardSpare;
    const uint64_t n_sess = (uint64_t) std::max(req.sessions, 1);
    const uint64_t need = session_bytes(g, want, kv_) * n_sess + shared_bytes(g, n_ff_pad_, want, kv_) + spare;
    if (need > free_b) {
        const uint64_t per = (session_bytes(g, 2, kv_) - session_bytes(g, 1, kv_)) * n_sess +
                             shared_bytes(g, n_ff_pad_, 2, kv_) - shared_bytes(g, n_ff_pad_, 1, kv_);
        const uint64_t base = session_bytes(g, 0, kv_) * n_sess + shared_bytes(g, n_ff_pad_, 0, kv_) + spare;
        const int64_t fits = free_b > base ? (int64_t) ((free_b - base) / per) : 0;
        return fail("--max-context " + std::to_string(want) + " needs " + std::to_string(need >> 20) + " MiB beside the weights, " +
                    std::to_string(free_b >> 20) + " MiB are free: at most about " + std::to_string(fits) +
                    " positions fit (a smaller --max-context, --kv fp8 or fewer --batch slots)");
    }
    ctx_ = want;
    basalt::log::info("basalt gemma4", "weights: dense %.2f GiB (FP8), experts %.2f GiB (NVFP4), embedding %.2f GiB in RAM; "
                      "context %lld x %llu sessions, %s KV",
                      (double) dense / (double) kGiB, (double) experts / (double) kGiB, (double) V * E * 2 / (double) kGiB,
                      (long long) ctx_, (unsigned long long) n_sess, kv_ == KvType::F8 ? "fp8" : "fp16");
    return {};
}

model::Result Gemma4Model::attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    for (int i = 0; i < std::max(req.sessions, 1); ++i) {
        std::string err;
        auto s = make_session(*this, err);
        if (!s) return {model::Status::Error, "gemma4: " + err};
        sessions.push_back(std::move(s));
    }
    for (auto& s : sessions)
        if (model::Result r = s->prepare(); !r.ok()) return r;
    return {};
}

}  // namespace basalt::models::gemma4
