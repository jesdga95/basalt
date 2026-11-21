// src/models/inkling2/model.cpp - loads an Inkling 2 pack: every weight to the main card, the embedding to mapped host
// memory, then the context sized from what the weights leave.
#include "models/inkling2/model.hpp"

#include "basalt/core/loader.hpp"
#include "basalt/log.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/platform/file.hpp"
#include "models/inkling2/prompt.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace basalt::models::inkling2 {
namespace {
constexpr uint64_t kGiB = 1ull << 30;
constexpr uint64_t kMiB = 1ull << 20;
constexpr int kTypeF32 = 0;
constexpr int kTypeQ8_0 = 8;
constexpr int kTypeBF16 = 30;
constexpr int kTypeNvfp4 = 40;
constexpr uint64_t kQ8Row = (uint64_t) kWidth / 32 * 34;   // bytes of a Q8_0 row of the hidden width
// what a session needs beside its caches (its row buffers, the expert scratch, the head's logits) and what the card keeps
// free for the kernels' modules and the driver
constexpr uint64_t kSessionFixed = 768 * kMiB;
constexpr uint64_t kCardSpare = 1024 * kMiB;

// The fused projection rows of a layer: tensor suffix, first row, rows, needed on a full layer only.
struct ProjPart {
    const char* suffix;
    int row0, rows;
    bool full_only;
};
constexpr ProjPart kProjParts[] = {
    {"attn_q.weight", kOffQ, 3072, false},       {"attn_rel.weight", kOffR, 384, false},
    {"attn_gate.weight", kOffGate, 3072, false}, {"attn_sink.weight", kOffSink, 24, false},
    {"attn_k.weight", kRowK, 1024, false},       {"attn_k_conv_proj.weight", kRowK + kOffTaps, 32, true},
    {"attn_v.weight", kRowV, 1024, false},       {"attn_v_conv_proj.weight", kRowV + kOffTaps, 32, true},
};
const char* const kStreamNames[kStreams] = {"q", "k", "v", "ffn"};
// A blob's stride on the card: whole blocks of its type and 16 bytes, so a kernel can read each expert's blocks in place.
uint64_t expert_stride(int type, uint64_t blob) {
    const uint64_t block = type == kTypeQ8_0 ? 34 : type == kTypeNvfp4 ? 36 : 1;
    uint64_t a = block, b = 16;
    while (b) {
        const uint64_t t = a % b;
        a = b;
        b = t;
    }
    const uint64_t unit = block / a * 16;
    return (blob + unit - 1) / unit * unit;
}
}  // namespace

uint64_t session_bytes(const Geometry& g, int64_t context, bool fp8) {
    int global = 0, swa = 0;
    for (const Visit& v : g.visits) (g.swa[(size_t) v.layer] ? swa : global) += 1;
    const uint64_t elem = fp8 ? 1 : 2;
    const uint64_t per_pos = 2ull * kKvHeads * kHeadDim * elem;
    const uint64_t ring = (uint64_t) g.window + kDecodeRows;
    return (uint64_t) global * per_pos * (uint64_t) context + (uint64_t) swa * per_pos * ring + kSessionFixed +
           prompt_bytes(g);   // PROMPT HOOK
}

InklingModel::InklingModel() = default;

InklingModel::~InklingModel() {
    for (void* p : dev_) cudaFree(p);
    if (embd_host_) cudaFreeHost(embd_host_);
    if (fd_ >= 0) platform::close_fd(fd_);
}

float* InklingModel::small(const std::string& name, uint64_t floats, core::LoadPlan& reads, int file, std::string& err) {
    const TensorRef* t = pack_.find(name);
    if (!t || t->type != kTypeF32 || t->bytes != floats * 4) {
        err = name + (t ? " is not F32 of the expected size" : " is missing");
        return nullptr;
    }
    void* p = nullptr;
    if (cudaMalloc(&p, t->bytes) != cudaSuccess) {
        err = "out of device memory loading " + name;
        return nullptr;
    }
    dev_.push_back(p);
    core::LoadItem it;
    it.file = file;
    it.off = t->offset;
    it.bytes = t->bytes;
    it.dev = p;
    it.device = device_;
    reads.add(it);
    return static_cast<float*>(p);
}

uint16_t* InklingModel::small_bf16(const std::string& name, uint64_t values, core::LoadPlan& reads, int file, std::string& err,
                                   uint16_t* dst) {
    const TensorRef* t = pack_.find(name);
    if (!t || t->type != kTypeBF16 || t->bytes != values * 2) {
        err = name + (t ? " is not BF16 of the expected size" : " is missing");
        return nullptr;
    }
    void* p = dst;
    if (!p) {
        if (cudaMalloc(&p, t->bytes) != cudaSuccess) {
            err = "out of device memory loading " + name;
            return nullptr;
        }
        dev_.push_back(p);
    }
    core::LoadItem it;
    it.file = file;
    it.off = t->offset;
    it.bytes = t->bytes;
    it.dev = p;
    it.device = device_;
    reads.add(it);
    return static_cast<uint16_t*>(p);
}

model::Result InklingModel::plan(const model::LoadRequest& req, core::LoadPlan& reads,
                                 std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    if (req.paths.empty()) return {model::Status::Error, "inkling2: no model path"};
    std::string err;
    const std::string& path = req.paths.front();
    if (!open_pack(path, pack_, err)) return {model::Status::Error, "inkling2: " + err};
    flags_ = req;
    cudaGetDevice(&device_);
    const Geometry& g = pack_.geo;
    desc_.arch = "inkling2";
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
    if (fd_ < 0) return {model::Status::Error, "inkling2: cannot open " + path};
    const int file = reads.file_fd(fd_);
    auto fail = [&](const std::string& e) { return model::Result{model::Status::Error, "inkling2: " + e}; };
    auto device_buf = [&](uint64_t bytes) -> uint8_t* {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        dev_.push_back(p);
        return static_cast<uint8_t*>(p);
    };
    const uint64_t n_embd = (uint64_t) g.n_embd, n_vocab = (uint64_t) g.n_vocab;

    // the token embedding: BF16 rows in mapped host memory, gathered by the card
    {
        const TensorRef* t = pack_.find("token_embd.weight");
        if (!t || t->type != kTypeBF16 || t->bytes != n_vocab * n_embd * 2) return fail("token_embd.weight is not BF16 [n_embd][n_vocab]");
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
    // the head: the pack's BF16 rows rounded to Q8_0 as they load
    {
        const TensorRef* t = pack_.find("output.weight");
        if (!t || t->type != kTypeBF16 || t->bytes != n_vocab * n_embd * 2) return fail("output.weight is not BF16 [n_embd][n_vocab]");
        if (!(head_ = device_buf(n_vocab * kQ8Row))) return fail("out of device memory loading the head");
        core::LoadItem it;
        it.file = file;
        it.off = t->offset;
        it.bytes = t->bytes;
        it.dev = head_;
        it.device = device_;
        it.conv = core::Conv::Bf16ToQ8_0;
        reads.add(it);
    }
    if (!(embed_norm_ = small("token_embd_norm.weight", n_embd, reads, file, err)) ||
        !(output_norm_ = small("output_norm.weight", n_embd, reads, file, err)) ||
        !(output_query_ = small("output_res_query.weight", n_embd, reads, file, err)))
        return fail(err);

    uint64_t dense = n_vocab * kQ8Row, experts = 0;
    layers_.assign((size_t) g.n_layers, LayerWeights{});
    for (int l = 0; l < g.n_layers; ++l) {
        LayerWeights& w = layers_[(size_t) l];
        const bool full = !g.swa[(size_t) l];
        const std::string b = "blk." + std::to_string(l) + ".";
        // the fused projection
        if (!(w.proj = device_buf((uint64_t) kProj * kQ8Row))) return fail("out of device memory loading layer " + std::to_string(l));
        cudaMemset(w.proj, 0, (size_t) kProj * kQ8Row);
        for (const ProjPart& part : kProjParts) {
            if (part.full_only && !full) continue;
            const TensorRef* t = pack_.find(b + part.suffix);
            if (!t || t->type != kTypeQ8_0 || t->bytes != (uint64_t) part.rows * kQ8Row)
                return fail(b + part.suffix + (t ? " is not Q8_0 of the expected rows" : " is missing"));
            core::LoadItem it;
            it.file = file;
            it.off = t->offset;
            it.bytes = t->bytes;
            it.dev = w.proj + (uint64_t) part.row0 * kQ8Row;
            it.device = device_;
            reads.add(it);
        }
        {
            const TensorRef* t = pack_.find(b + "attn_output.weight");
            const uint64_t want = n_embd * (3072 / 32 * 34);
            if (!t || t->type != kTypeQ8_0 || t->bytes != want) return fail(b + "attn_output.weight is not Q8_0 [3072][n_embd]");
            if (!(w.out = device_buf(want))) return fail("out of device memory loading layer " + std::to_string(l));
            core::LoadItem it;
            it.file = file;
            it.off = t->offset;
            it.bytes = t->bytes;
            it.dev = w.out;
            it.device = device_;
            reads.add(it);
        }
        dense += (uint64_t) kProj * kQ8Row + n_embd * (3072 / 32 * 34);
        auto S = [&](const char* suffix, uint64_t floats) { return small(b + suffix, floats, reads, file, err); };
        if (full) {
            if (!(w.q_norm = S("attn_q_inp_norm.weight", n_embd)) || !(w.k_norm = S("attn_k_inp_norm.weight", n_embd)) ||
                !(w.v_norm = S("attn_v_inp_norm.weight", n_embd)))
                return fail(err);
        } else if (!(w.attn_norm = S("attn_norm.weight", n_embd))) {
            return fail(err);
        }
        if (!(w.q_clip = S("attn_q_clip.weight", kHeads * kHeadDim)) || !(w.r_clip = S("attn_rel_clip.weight", kHeads * kRel)) ||
            !(w.rel_table = S("attn_rel_bias.weight", kRel * kExtent)) || !(w.k_conv = S("attn_k_conv.weight", 1024 * kConv)) ||
            !(w.v_conv = S("attn_v_conv.weight", 1024 * kConv)) || !(w.attn_post_conv = S("attn_post_conv.weight", n_embd * kConv)) ||
            !(w.ffn_post_conv = S("ffn_post_conv.weight", n_embd * kConv)) || !(w.ffn_query = S("ffn_res_query.weight", n_embd)) ||
            !(w.ffn_norm = S("ffn_norm.weight", n_embd)) ||
            !(w.router = S("ffn_gate_inp.weight", (uint64_t) g.n_expert * n_embd)) ||
            !(w.router_bias = S("exp_probs_b.bias", (uint64_t) g.n_expert)))
            return fail(err);
        if (l > 0 && !(w.attn_query = S("attn_res_query.weight", (full ? 3 : 1) * n_embd))) return fail(err);
        if (g.looped(l)) {
            if (!(w.msg_query = S("loop_msg_query.weight", kStreams * n_embd))) return fail(err);
            const uint64_t rank = (uint64_t) g.gate_rank;
            for (int s = 0; s < kStreams; ++s) {
                const std::string c = b + "loop_" + kStreamNames[s] + "_ctrl_";
                if (!(w.ctrl_down[s] = small_bf16(c + "down.weight", rank * 2 * n_embd, reads, file, err))) return fail(err);
                uint16_t* gf = reinterpret_cast<uint16_t*>(device_buf(2 * n_embd * rank * 2));
                if (!gf) return fail("out of device memory loading layer " + std::to_string(l));
                if (!small_bf16(c + "gain.weight", n_embd * rank, reads, file, err, gf) ||
                    !small_bf16(c + "forget.weight", n_embd * rank, reads, file, err, gf + n_embd * rank))
                    return fail(err);
                w.ctrl_gf[s] = gf;
            }
        }
        // the experts: the layer's blobs in one run, each at a stride of whole blocks
        const ExpertLayer& el = pack_.experts[(size_t) l];
        w.gu_type = el.gu_type;
        w.d_type = el.d_type;
        w.blob_bytes = el.blob_bytes;
        w.expert_stride = expert_stride(el.gu_type, el.blob_bytes);
        const uint64_t eb = (uint64_t) g.n_expert * w.expert_stride;
        if (!(w.experts = device_buf(eb))) return fail("out of device memory loading layer " + std::to_string(l) + "'s experts");
        if (w.expert_stride == el.blob_bytes) {
            core::LoadItem it;
            it.file = file;
            it.off = pack_.experts_range.offset + el.offset;
            it.bytes = eb;
            it.dev = w.experts;
            it.device = device_;
            reads.add(it);
        } else {
            cudaMemset(w.experts, 0, eb);
            for (int e = 0; e < g.n_expert; ++e) {
                core::LoadItem it;
                it.file = file;
                it.off = pack_.experts_range.offset + el.offset + (uint64_t) e * el.blob_bytes;
                it.bytes = el.blob_bytes;
                it.dev = w.experts + (uint64_t) e * w.expert_stride;
                it.device = device_;
                reads.add(it);
            }
        }
        experts += eb;
        // the experts' addresses, for the plan
        std::vector<unsigned long long> table((size_t) g.n_expert);
        for (int e = 0; e < g.n_expert; ++e) table[(size_t) e] = (unsigned long long) (w.experts + (uint64_t) e * w.expert_stride);
        if (!(w.blob_table = reinterpret_cast<unsigned long long*>(device_buf(table.size() * 8))) ||
            cudaMemcpy(w.blob_table, table.data(), table.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("out of device memory loading layer " + std::to_string(l) + "'s expert table");
    }
    fp8_ = req.kv == "fp8";

    // the context: what the weights leave, less the card's spare
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t want = req.max_context > 0 ? req.max_context : kDefaultContext;
        if (want < 1024 || (g.context_length > 0 && want > g.context_length))
            return fail("--max-context " + std::to_string(want) + " is outside 1,024 .. " + std::to_string(g.context_length));
        const uint64_t avail = free_b > kCardSpare ? (free_b - kCardSpare) / (uint64_t) std::max(req.sessions, 1) : 0;
        const uint64_t need = session_bytes(g, want, fp8_);
        if (need > avail) {
            const uint64_t per = session_bytes(g, 2, fp8_) - session_bytes(g, 1, fp8_);
            const uint64_t base = session_bytes(g, 0, fp8_);
            const int64_t fits = avail > base ? (int64_t) ((avail - base) / per) : 0;
            return {model::Status::Error, "inkling2: --max-context " + std::to_string(want) + " needs " +
                                              std::to_string(need / kMiB) + " MiB a session; about " + std::to_string(fits) +
                                              " positions fit" + (fp8_ ? "" : " (--kv fp8 halves a position's bytes)")};
        }
        ctx_ = want;
    }
    basalt::log::info("basalt inkling2", "weights on the GPU: dense %.2f GiB, experts %.2f GiB; the embedding in host memory (%.2f GiB)",
                      (double) dense / (double) kGiB, (double) experts / (double) kGiB,
                      (double) (n_vocab * n_embd * 2) / (double) kGiB);
    model::Result r;
    for (int i = 0; r.ok() && i < std::max(req.sessions, 1); ++i) {
        std::unique_ptr<model::ModelSession> s = make_session(*this, err);
        if (!s) r = {model::Status::Error, "inkling2: " + err};
        else sessions.push_back(std::move(s));
    }
    return r;
}

model::Result InklingModel::attach(const model::LoadRequest&, std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    for (auto& s : sessions)
        if (model::Result r = s->prepare(); !r.ok()) return r;
    return {};
}

}  // namespace basalt::models::inkling2
