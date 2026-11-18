// src/models/hy4flash/model.cpp - loads a Hy4-flash pack: the dense tensors to the main card, the embedding to host memory,
// the routed experts to RAM.
#include "models/hy4flash/model.hpp"

#include "basalt/artifact/dequant.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "models/hy4flash/context_policy.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace basalt::models::hy4flash {
namespace {
constexpr uint64_t kGiB = 1ull << 30;
constexpr int kTypeQ8_0 = 8;
// CPU uses of a missed expert that earn it a slot on the card
constexpr int kAdmitUses = 2;
}  // namespace

Hy4Model::Hy4Model() = default;

Hy4Model::~Hy4Model() {
    for (auto& [name, t] : dev_)
        if (t.ptr) cudaFree(t.ptr);
    if (fd_ >= 0) platform::close_fd(fd_);
}

const DevTensor* Hy4Model::tensor(const std::string& name) const {
    auto it = dev_.find(name);
    return it == dev_.end() ? nullptr : &it->second;
}

bool Hy4Model::embedding_row(int32_t token, float* out) const {
    const uint64_t n = (uint64_t) geo().n_embd;
    if (token < 0 || (int64_t) token >= geo().n_vocab) return false;
    const uint8_t* row = embd_.data() + (uint64_t) token * (n / 32) * 34;
    for (uint64_t b = 0; b < n / 32; ++b) dequantize_q8_0(row + b * 34, out + b * 32);
    return true;
}

core::ExpertTierSpec Hy4Model::tier_spec() const {
    const Geometry& g = pack_.geo;
    core::ExpertTierSpec s;
    s.area = "basalt hy4flash";
    s.name = "hy4flash";
    s.n_layers = g.n_layers - g.n_dense;
    s.n_expert = g.n_expert;
    s.top_k = g.n_used;
    s.n_embd = g.n_latent;   // the experts run over the latent that routed_down makes
    s.n_ff = g.n_ff_exp;
    for (int l = g.n_dense; l < g.n_layers; ++l) {
        const ExpertLayer& e = pack_.experts[(size_t) l];
        s.layers.push_back({e.gu_type, e.d_type, e.offset, e.blob_bytes, g.clamp_exp[(size_t) l]});
        s.share_stride = std::max<uint64_t>(s.share_stride, e.blob_bytes);
    }
    s.path = path_;
    s.fd = fd_;
    s.range = pack_.experts_range;
    s.peer_devices = peer_devices_;
    s.diag_tier_ram = flags_.diag_tier_ram;
    s.diag_tier_vram = flags_.diag_tier_vram;
    s.diag_tier_frozen = flags_.diag_tier_frozen;
    s.vram_reserve_mib = flags_.vram_reserve_mib;
    s.peer_reserve_mib = flags_.peer_reserve_mib;
    s.expert_cache = flags_.expert_cache;
    s.resident_budget = flags_.resident_budget;
    s.pool_workers = flags_.pool_workers;
    s.pool_affinity = flags_.pool_affinity;
    s.pool_affinity_given = flags_.pool_affinity_given;
    s.expert_profile = flags_.expert_profile;
    s.admit_uses = kAdmitUses;
    s.share_max_tokens = 16384;
    return s;
}

model::Result Hy4Model::plan(const model::LoadRequest& req, core::LoadPlan& reads,
                             std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    if (req.paths.empty()) return {model::Status::Error, "hy4flash: no model path"};
    std::string err;
    const std::string& path = req.paths.front();
    if (!open_pack(path, pack_, err)) return {model::Status::Error, "hy4flash: " + err};
    cudaGetDevice(&device_);
    const Geometry& g = pack_.geo;
    desc_.arch = "hy4flash";
    desc_.n_layers = g.n_layers;
    desc_.n_embd = g.n_embd;
    desc_.n_vocab = g.n_vocab;
    desc_.n_expert = g.n_expert;
    desc_.n_expert_used = g.n_used;
    desc_.n_ff_exp = g.n_ff_exp;
    desc_.n_expert_shared = g.n_shared;
    desc_.context_length = g.context_length;
    for (const ExpertLayer& e : pack_.experts) {
        desc_.expert_gate_up_type.push_back(e.gu_type);
        desc_.expert_down_type.push_back(e.d_type);
    }
    diag_dump_ = req.diag_dump;
    if (!diag_dump_.empty()) basalt::log::info("basalt hy4flash", "--diag-dump: each layer's streams to %s", diag_dump_.c_str());

    fd_ = platform::open_fd_utf8(path);
    if (fd_ < 0) return {model::Status::Error, "hy4flash: cannot open " + path};
    const int file = reads.file_fd(fd_);

    uint64_t dense = 0;
    for (const auto& [name, t] : pack_.tensors) {
        if (name == "token_embd.weight") {
            if (t.type != kTypeQ8_0 || t.shape.size() != 2 || t.shape[0] != (uint64_t) g.n_embd) {
                return {model::Status::Error, "hy4flash: token_embd.weight is not Q8_0 [n_embd][vocab]"};
            }
            embd_.resize(t.bytes);
            core::LoadItem it;
            it.file = file;
            it.off = t.offset;
            it.bytes = t.bytes;
            it.host = embd_.data();
            reads.add(it);
            continue;
        }
        if (name == "per_layer_token_embd.weight") continue;
        DevTensor d;
        d.type = t.type;
        d.bytes = t.bytes;
        d.shape = t.shape;
        if (cudaMalloc(&d.ptr, d.bytes) != cudaSuccess) return {model::Status::Error, "hy4flash: out of device memory loading " + name};
        core::LoadItem it;
        it.file = file;
        it.off = t.offset;
        it.bytes = t.bytes;
        it.dev = d.ptr;
        it.device = device_;
        reads.add(it);
        dense += d.bytes;
        dev_.emplace(name, std::move(d));
    }
    fp8_ = req.kv == "fp8";
    f32_ = req.kv == "f32";
    // DFLY HOOK: the drafter's weights, with the dense ones so the context is sized after them (no pack drafter, or --spec 0: none)
    spec_ = req.spec;
    if (req.draft)
        if (model::Result r = dfly_plan(pack_, reads, file, device_, dfly_); !r.ok()) return r;
    // the context: refused when its caches would take more than half of what the weights leave
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        int sparse = 0;
        for (int l = 0; l < g.n_layers; ++l) sparse += !g.swa[(size_t) l];
        const ctxp::Plan p = ctxp::plan_context(req.max_context > 0 ? req.max_context : kDefaultContext, g.context_length,
                                                free_b / (uint64_t) std::max(req.sessions, 1), sparse, fp8_);
        if (!p.error.empty()) return {model::Status::Error, "hy4flash: " + p.error};
        if (!p.note.empty()) basalt::log::info("basalt hy4flash", "%s", p.note.c_str());
        ctx_ = p.context;
    }
    basalt::log::info("basalt hy4flash", "%zu dense tensors, %.2f GiB on the GPU", dev_.size(), (double) dense / (double) kGiB);

    if (pack_.engram.present) {
        engram_ = std::make_unique<ngram::PleReader>();
        const EngramInfo& e = pack_.engram;
        if (!engram_->open(path, e.file_offset, e.rows, 64, 1 << 16, err, e.row_bytes))
            return {model::Status::Error, "hy4flash: " + err};
    }
    // the sessions first, then the expert tier, sized from what they leave and opened by the first session; its reads are
    // queued in the start's plan
    flags_ = req;
    path_ = path;
    std::vector<int> peers = req.peer_devices;
    if (peers.size() > (size_t) kMaxSecondCards) {
        basalt::log::warn("basalt hy4flash", "%zu second cards given, %d hold experts; later cards idle", peers.size(), kMaxSecondCards);
        peers.resize((size_t) kMaxSecondCards);
    }
    peer_devices_ = std::move(peers);
    planning_ = &reads;
    model::Result r;
    for (int i = 0; r.ok() && i < std::max(req.sessions, 1); ++i) {
        std::unique_ptr<model::ModelSession> s = make_session(*this, err);
        if (!s) r = {model::Status::Error, "hy4flash: " + err};
        else sessions.push_back(std::move(s));
    }
    if (r.ok()) r = sessions.front()->prepare();
    planning_ = nullptr;
    return r;
}

model::Result Hy4Model::attach(const model::LoadRequest&, std::vector<std::unique_ptr<model::ModelSession>>&) {
    std::string err;
    if (shared_tier_.tier && !shared_tier_.tier->attach(err)) return {model::Status::Error, "hy4flash: " + err};
    return {};
}

}  // namespace basalt::models::hy4flash
