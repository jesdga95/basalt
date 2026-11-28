// src/models/deepseek41/model.cpp - loads a DeepSeek-V4.1 pack onto the GPU.
#include "basalt/core/device.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/expert_tier.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/log.hpp"
#include "models/deepseek41/model.hpp"

#include "basalt/artifact/dequant.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "models/deepseek41/attn_batch.hpp"
#include "models/deepseek41/context_policy.hpp"
#include "models/deepseek41/session.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace basalt::models::deepseek41 {

DsModel::~DsModel() {
    for (auto& [name, t] : dev_)
        if (t.ptr) cudaFree(t.ptr);
    if (embd_fd_ >= 0) platform::close_fd(embd_fd_);
    if (experts_fd_ >= 0) platform::close_fd(experts_fd_);
}

bool DsModel::load(const std::string& path, core::LoadPlan& plan, bool ple_ram, std::string& err) {
    if (!open_pack(path, pack_, err)) return false;
    path_ = path;
    cudaGetDevice(&device_);
    const Geometry& g = pack_.geo;
    desc_.arch = "deepseek41";
    desc_.n_layers = g.n_layers;
    desc_.n_embd = g.n_embd;
    desc_.n_vocab = g.n_vocab;
    desc_.n_expert = g.n_expert;
    desc_.n_expert_used = g.n_used;
    desc_.n_ff_exp = g.n_ff_exp;
    desc_.n_expert_shared = g.n_shared;
    desc_.context_length = g.context_length;
    // the grouped kernels take every layer's routed pair: refuse an unsupported pack at load, not at the first token
    for (int l = 0; l < g.n_layers; ++l) {
        const ExpertLayer& e = pack_.experts[(size_t) l];
        if (!basalt::kernels::native_expert_supported(e.gu_type, e.d_type, g.n_embd, g.n_ff_exp)) {
            err = "layer " + std::to_string(l) + ": no grouped expert kernels for gate/up " +
                  std::string(basalt::ggml_type_name((uint32_t) e.gu_type)) + " (" + std::to_string(e.gu_type) +
                  "), down " + std::string(basalt::ggml_type_name((uint32_t) e.d_type)) + " (" + std::to_string(e.d_type) + ")";
            return false;
        }
        desc_.expert_gate_up_type.push_back(e.gu_type);
        desc_.expert_down_type.push_back(e.d_type);
    }

    embd_fd_ = platform::open_fd_utf8(path);
    experts_fd_ = platform::open_fd_utf8(path);
    if (embd_fd_ < 0 || experts_fd_ < 0) {
        err = "cannot open " + path;
        return false;
    }

    // dense tensors to the device; the token embedding and the Engram table stay in the file
    uint64_t total = 0;
    const int dense_file = plan.file_fd(embd_fd_);
    int dense_device = 0;
    if (cudaGetDevice(&dense_device) != cudaSuccess) dense_device = 0;
    for (const auto& [name, t] : pack_.tensors) {
        if (name == "token_embd.weight" || name == "per_layer_token_embd.weight") continue;
        DevTensor d;
        d.type = t.type;
        d.bytes = t.bytes;
        d.shape = t.shape;
        // the router reads its gate as F32; BF16 widens exactly (the compressor and indexer kernels read BF16)
        const bool widen = t.type == 30 && name.find("ffn_gate_inp.weight") != std::string::npos;
        if (widen) {
            d.type = 0;
            d.bytes = t.bytes * 2;
        }
        if (cudaMalloc(&d.ptr, d.bytes) != cudaSuccess) {
            err = "out of device memory loading " + name;
            return false;
        }
        core::LoadItem it;
        it.file = dense_file;
        it.off = t.offset;
        it.bytes = t.bytes;
        it.dev = d.ptr;
        it.device = dense_device;
        it.conv = widen ? core::Conv::Bf16ToF32 : core::Conv::Copy;
        plan.add(it);
        total += d.bytes;
        dev_.emplace(name, std::move(d));
    }
    basalt::log::info("basalt deepseek41", "%zu dense tensors, %.2f GiB on the GPU", dev_.size(),
                 (double) total / (1024.0 * 1024.0 * 1024.0));

    if (pack_.engram.present) {
        engram_ = std::make_unique<ngram::PleReader>();
        const EngramInfo& e = pack_.engram;
        // Map now so tier_spec() reserves its RAM; lock/touch in attach() after the expert tier's reads finish so
        // buffered expert reads cannot evict unlocked Engram pages from the OS page cache.
        if (ple_ram) {
            if (!engram_->open_ram(path, e.file_offset, e.rows, err, e.row_bytes, /*lock=*/false)) return false;
        } else if (!engram_->open(path, e.file_offset, e.rows, 256, 1 << 20, err, e.row_bytes)) {
            return false;
        }
    }
    return true;
}

int64_t DsModel::ctxp_default() { return ctxp::kLegacyContext; }

static std::vector<int> cache_ratios(const Geometry& g) {
    std::vector<int> ratios;
    for (int il = 0; il < g.n_layers; ++il)
        if (ctxp::owns_cache(il, g.compress_ratio[(size_t) il])) ratios.push_back(g.compress_ratio[(size_t) il]);
    return ratios;
}

uint64_t DsModel::context_bytes() const { return ctxp::context_bytes(context(), cache_ratios(pack_.geo), cache_format_); }

int64_t DsModel::keep_positions(int64_t backed) const {
    return prefillp::keep_positions(backed, cache_ratios(pack_.geo), cache_format_);
}

core::ExpertTierSpec DsModel::tier_spec() const {
    const Geometry& g = pack_.geo;
    core::ExpertTierSpec s;
    s.area = "basalt deepseek41";
    s.name = "deepseek41";
    s.n_layers = g.n_layers;
    s.n_expert = g.n_expert;
    s.top_k = g.n_used;
    s.n_embd = g.n_embd;
    s.n_ff = g.n_ff_exp;
    for (int l = 0; l < g.n_layers; ++l) {
        const ExpertLayer& e = pack_.experts[(size_t) l];
        s.layers.push_back({e.gu_type, e.d_type, e.offset, e.blob_bytes, g.clamp_exp[(size_t) l]});
        s.share_stride = std::max<uint64_t>(s.share_stride, staged_stride(e));
    }
    s.path = path_;
    s.fd = experts_fd();
    s.range = pack_.experts_range;
    s.gpus = gpus_;
    s.diag_tier_ram = diag_tier_ram_;
    s.diag_tier_vram = diag_tier_vram_;
    s.diag_tier_frozen = tiers_.diag_tier_frozen;
    s.route_tail_skip = tiers_.route.tail_skip;
    s.route_prior = tiers_.route.prior;
    s.vram_reserve_mib = tiers_.vram_reserve_mib;
    s.peer_reserve_mib = tiers_.peer_reserve_mib;
    s.expert_cache = tiers_.expert_cache;
    s.resident_budget = tiers_.resident_budget;
    s.pool_workers = tiers_.pool_workers;
    s.pool_affinity = tiers_.pool_affinity;
    s.pool_affinity_given = tiers_.pool_affinity_given;
    s.expert_profile = tiers_.expert_profile;
    s.ring_pieces = prefillp::kRingPieces;
    s.share_max_tokens = prefillp::kMaxChunk;
    // the image encoder's cuBLAS handle takes host memory when created, so it opens before the RAM tier reads the free
    // memory, with the prompt path's buffers
    s.host.open = [this](std::string& err) {
        if (tiers_.vision && !vision_blas().open(err)) return false;
        return prompt_host_.open(*this, 0, sessions(), tiers_.vision, err);
    };
    s.host.bytes = [this] { return prompt_host_.bytes() + (engram_ ? engram_->ram_bytes() : 0); };
    s.host.set_ring = [this](int pieces, std::string& err) { return prompt_host_.set_ring(pieces, err); };
    s.host.ram_only = [this] { warm_token_table(); };
    s.host_cache_keep = engram_ ? engram_->ram_bytes() : 0;
    return s;
}

prefillp::Dims DsModel::prompt_dims() const {
    const Geometry& g = pack_.geo;
    prefillp::Dims d;
    d.n_embd = g.n_embd;
    d.n_used = g.n_used;
    d.n_expert = g.n_expert;
    d.n_ff_exp = g.n_ff_exp;
    d.engram_cols = (int64_t) pack_.engram.hash.n_cols() * 256;
    for (const ExpertLayer& e : pack_.experts) {
        d.expert_blob_bytes = std::max<uint64_t>(d.expert_blob_bytes, e.blob_bytes);
    }
    d.attn_fixed_bytes = attn_batch_fixed_bytes(cache_format_);
    d.attn_token_bytes = attn_batch_token_bytes();
    return d;
}

bool DsModel::resolve_context(int64_t requested, std::string& err) {
    const Geometry& g = pack_.geo;
    const std::vector<int> ratios = cache_ratios(g);
    uint64_t free_b = 0, total_b = 0;
    std::string why;
    if (!core::device_memory(-1, free_b, total_b, why)) {
        err = "deepseek41: cannot query the free device memory: " + why;
        return false;
    }
    const ctxp::Plan p = ctxp::plan_context(requested, g.context_length, free_b, ratios, cache_format_);
    if (!p.error.empty()) {
        err = "deepseek41: " + p.error;
        return false;
    }
    ctx_ = p.context;
    const uint64_t full = ctxp::context_bytes(ctx_, ratios, cache_format_);
    core::LoadProgress::kv(core::LoadProgress::cuda(0), ctxp::context_bytes(0, ratios, cache_format_), ctx_, "", full);
    basalt::log::info("basalt deepseek41", "context %lld tokens (caches of a full context: %.2f GiB, --kv %s)%s%s", (long long) ctx_,
                 (double) full / (1024.0 * 1024.0 * 1024.0), cache_format_ == CacheFormat::Fp8 ? "fp8" : "fp16",
                 p.note.empty() ? "" : "; ", p.note.c_str());
    return true;
}

const DevTensor* DsModel::tensor(const std::string& name) const {
    auto it = dev_.find(name);
    return it == dev_.end() ? nullptr : &it->second;
}

bool DsModel::embedding_row(int32_t token, float* out, std::string& err) const {
    const TensorRef* t = pack_.find("token_embd.weight");
    if (!t || (t->type != 14 && t->type != 13 && t->type != 8) || t->shape.size() != 2) {
        err = "token_embd.weight is missing or not Q5_K, Q6_K or Q8_0";
        return false;
    }
    const uint64_t n_embd = t->shape[0];
    if (token < 0 || (uint64_t) token >= t->shape[1]) {
        err = "token id out of range";
        return false;
    }
    const bool q8 = t->type == 8;
    const uint64_t q5_block = 176;
    const uint64_t row_bytes = q8 ? n_embd / 32 * 34 : n_embd / 256 * (t->type == 13 ? q5_block : 210);
    std::vector<uint8_t> row(row_bytes);
    if (!platform::pread_all(embd_fd_, row.data(), row_bytes, t->offset + (uint64_t) token * row_bytes)) {
        err = "cannot read the embedding row";
        return false;
    }
    if (q8) {
        for (uint64_t b = 0; b < n_embd / 32; ++b) dequantize_q8_0(row.data() + b * 34, out + b * 32);
    } else if (t->type == 13) {
        for (uint64_t b = 0; b < n_embd / 256; ++b) dequantize_q5_K(row.data() + b * q5_block, out + b * 256);
    } else {
        for (uint64_t b = 0; b < n_embd / 256; ++b) dequantize_q6_K(row.data() + b * 210, out + b * 256);
    }
    return true;
}

void DsModel::warm_token_table() const {
    if (const TensorRef* t = pack_.find("token_embd.weight")) platform::advise_willneed(embd_fd_, t->offset, t->bytes);
}

bool DsModel::open_mmproj(vision::Mmproj& mp, std::string& err) const {
    if (pack_.vision.empty() || pack_.vision.bytes == 0) {
        err = "this pack carries no vision projector (convert with --mmproj)";
        return false;
    }
    return mp.open(pack_.vision, err, "deepseek41v");
}

model::Result DsModel::plan(const model::LoadRequest& req, core::LoadPlan& reads,
                            std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    if (req.paths.empty()) return {model::Status::Error, "deepseek41: no model path"};
    if (!req.ple_io.empty() && req.ple_io != "direct" && req.ple_io != "ram")
        return {model::Status::Error, "deepseek41: --ple-io " + req.ple_io + ": direct or ram"};
    std::string err;
    if (!load(req.paths.front(), reads, req.ple_io == "ram", err)) return {model::Status::Error, "deepseek41: " + err};
    CacheFormat fmt = CacheFormat::Fp8;
    if (!req.kv.empty() && !ctxp::cache_format(req.kv, fmt)) return {model::Status::Error, "deepseek41: --kv " + req.kv + ": fp8 or fp16"};
    set_cache_format(fmt);
    if (!resolve_context(req.max_context, err)) return {model::Status::Error, err};
    set_diag_dump(req.diag_dump);
    set_diag_tier(req.diag_tier_ram, req.diag_tier_vram);
    set_tiers(req);
    gpus_ = req.gpus;
    // the sessions first (the drafter takes its memory with the first), then the expert tier, sized from what they leave,
    // opened by the first session; what those share queues its reads in the start's plan
    planning_ = &reads;
    model::Result r;
    for (int i = 0; r.ok() && i < std::max(req.sessions, 1); ++i) {
        std::unique_ptr<model::ModelSession> s = make_session_impl(*this, err);
        if (!s) r = {model::Status::Error, err};
        else sessions.push_back(std::move(s));
    }
    if (r.ok()) r = sessions.front()->prepare();
    planning_ = nullptr;
    return r;
}

model::Result DsModel::attach(const model::LoadRequest&, std::vector<std::unique_ptr<model::ModelSession>>&) {
    std::string err;
    if (shared_tier_.tier && !shared_tier_.tier->attach(err)) return {model::Status::Error, err};
    if (engram_ && engram_->ram_bytes() > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        engram_->lock_ram();
        basalt::log::info("basalt deepseek41", "Engram table %.2f GiB %s (--ple-io ram) in %.1f s",
                          (double) engram_->ram_bytes() / (1024.0 * 1024.0 * 1024.0),
                          engram_->locked() ? "locked in RAM" : "loaded (not locked)",
                          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return {};
}

model::PeerHooks* DsModel::peer() const { return shared_tier_.tier.get(); }

}  // namespace basalt::models::deepseek41
