// src/models/ernie55/model.cpp - loads an ERNIE-5.5-Flash .basalt pack: dense/attention/mHC weights to GPU 0,
// BF16 token embeddings to host RAM, output head BF16->Q8_0 on GPU 0, and MXFP4 routed experts via core::ExpertTier.
#include "models/ernie55/model.hpp"

#include "basalt/core/loader.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <string_view>

namespace basalt::models::ernie55 {
namespace {

constexpr uint64_t kGiB = 1ull << 30;
constexpr int kTypeQ8_0 = 8;
constexpr int kTypeBF16 = 30;
constexpr int kTypeF32 = 0;
constexpr std::string_view kRouterSuffix = "ffn_gate_inp.weight";
constexpr int kAdmitUses = 2;
constexpr int kClaimsPerRow = 128;
constexpr int kVictimHalfLife = 256;
constexpr int kAdmitMargin = 2;

inline float bf16_to_f32(uint16_t u) {
    uint32_t bits = (uint32_t) u << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

ErnieModel::ErnieModel() = default;

ErnieModel::~ErnieModel() {
    for (auto& [name, t] : dev_)
        if (t.ptr) cudaFree(t.ptr);
    if (shared_tier_.h_x) cudaFreeHost(shared_tier_.h_x);
    if (shared_tier_.h_rows) cudaFreeHost(shared_tier_.h_rows);
    if (fd_ >= 0) platform::close_fd(fd_);
}

const DevTensor* ErnieModel::tensor(const std::string& name) const {
    auto it = dev_.find(name);
    return it == dev_.end() ? nullptr : &it->second;
}

bool ErnieModel::embedding_row(int32_t token, float* out) const {
    const int64_t n = geo().n_embd;
    if (token < 0 || (int64_t) token >= geo().n_vocab) return false;
    const uint16_t* row = embd_ + (size_t) token * (size_t) n;
    for (int64_t i = 0; i < n; ++i) out[i] = bf16_to_f32(row[i]);
    return true;
}

size_t ErnieModel::expert_stride() const {
    size_t blob = 0, unit = 16;
    for (const ExpertLayer& el : pack_.experts) {
        blob = std::max<size_t>(blob, el.blob_bytes);
        if (!el.present) continue;
        for (int t : {el.gu_type, el.d_type}) unit = std::lcm(unit, ggml_type_size((ggml_type) t));
    }
    return (blob + unit - 1) / unit * unit;
}

core::ExpertTierSpec ErnieModel::tier_spec() const {
    const Geometry& g = pack_.geo;
    core::ExpertTierSpec s;
    s.area = "basalt ernie55";
    s.name = "ernie55";
    s.n_layers = g.n_layers - g.n_dense;
    s.n_expert = g.n_expert;
    s.top_k = g.n_used;
    s.n_embd = g.n_latent;
    s.n_ff = g.n_ff_exp;
    for (int l = g.n_dense; l < g.n_layers; ++l) {
        const ExpertLayer& e = pack_.experts[(size_t) l];
        core::ExpertTierSpec::Layer lay;
        lay.gu_type = e.gu_type;
        lay.d_type = e.d_type;
        lay.offset = e.offset;
        lay.blob_bytes = e.blob_bytes;
        lay.clamp = 0.0f;
        lay.gate_softcap = g.gate_softcap;
        lay.up_softcap = g.up_softcap;
        s.layers.push_back(lay);
    }
    s.path = path_;
    s.fd = fd_;
    s.range = pack_.experts_range;
    s.gpus = gpus_;
    s.diag_tier_ram = flags_.diag_tier_ram;
    s.diag_tier_vram = flags_.diag_tier_vram;
    s.diag_tier_frozen = flags_.diag_tier_frozen;
    s.route_tail_skip = flags_.route.tail_skip;
    s.route_prior = flags_.route.prior;
    s.vram_reserve_mib = flags_.vram_reserve_mib;
    s.peer_reserve_mib = flags_.peer_reserve_mib;
    s.expert_cache = flags_.expert_cache;
    s.resident_budget = flags_.resident_budget;
    s.pool_workers = flags_.pool_workers;
    s.pool_affinity = flags_.pool_affinity;
    s.pool_affinity_given = flags_.pool_affinity_given;
    s.expert_profile = flags_.expert_profile;
    s.admit_uses = kAdmitUses;
    s.claims_per_row = kClaimsPerRow;
    s.victim_half_life = kVictimHalfLife;
    s.admit_margin = kAdmitMargin;
    s.share_max_tokens = 16384;
    s.share_stride = expert_stride();
    s.lend_segment = kExpertLendSegment;
    if (dspark_) {
        const uint64_t dh = dspark_->host_bytes();
        s.host.bytes = [dh]() { return dh; };
    }
    return s;
}

model::Result ErnieModel::plan(const model::LoadRequest& req, core::LoadPlan& reads,
                               std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    if (req.paths.empty()) return {model::Status::Error, "ernie55: no model path"};
    std::string err;
    const std::string& path = req.paths.front();
    if (!open_pack(path, pack_, err)) return {model::Status::Error, "ernie55: " + err};
    cudaGetDevice(&device_);
    const Geometry& g = pack_.geo;

    desc_.arch = "ernie55";
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

    if (!ctxp::cache_format(req.kv, cache_fmt_)) {
        return {model::Status::Error, "ernie55: unsupported --kv '" + req.kv + "' (use fp8 or fp16)"};
    }

    fd_ = platform::open_fd_utf8(path);
    if (fd_ < 0) return {model::Status::Error, "ernie55: cannot open " + path};
    const int file = reads.file_fd(fd_);

    uint64_t dense_bytes = 0;
    for (const auto& [name, t] : pack_.tensors) {
        if (name == "token_embd.weight") {
            if (t.type != kTypeBF16 || t.shape.size() != 2 || t.shape[0] != (uint64_t) g.n_embd) {
                return {model::Status::Error, "ernie55: token_embd.weight is not BF16 [n_embd][vocab]"};
            }
            // a view of the file, not a copy: a token reads one row, and the pages stay with the OS file cache
            // instead of taking RAM the expert tier sizes itself from
            platform::MapOptions mo;
            mo.offset = t.offset / kEmbdMapAlign * kEmbdMapAlign;
            mo.bytes = t.offset + t.bytes - mo.offset;
            mo.shared = true;
            std::optional<platform::MappedFile> mf = platform::MappedFile::open_ro(path, &err, mo);
            if (!mf) return {model::Status::Error, "ernie55: cannot map token_embd.weight: " + err};
            embd_map_ = std::move(*mf);
            embd_ = reinterpret_cast<const uint16_t*>(embd_map_.data() + (t.offset - mo.offset));
            continue;
        }

        DevTensor d;
        d.type = t.type;
        d.bytes = t.bytes;
        d.shape = t.shape;
        // Load BF16 output.weight as Q8_0 on GPU 0
        const bool head_q8 = name == "output.weight" && t.type == kTypeBF16 && t.shape.size() == 2 && t.shape[0] % 32 == 0;
        if (head_q8) {
            d.type = kTypeQ8_0;
            d.bytes = t.shape[0] * t.shape[1] / 32 * 34;
        }
        // Pack router weights as BF16 via Bf16High only after verifying every F32 element is an exact BF16 value.
        bool router_bf16 = false;
        if (t.type == kTypeF32 && name.size() > kRouterSuffix.size() &&
            name.compare(name.size() - kRouterSuffix.size(), kRouterSuffix.size(), kRouterSuffix) == 0) {
            if (!f32_range_is_bf16(fd_, t.offset, t.bytes, router_bf16, err)) {
                return {model::Status::Error, "ernie55: " + name + ": " + err};
            }
        }
        if (router_bf16) {
            d.type = kTypeBF16;
            d.bytes = t.bytes / 2;
        }
        const uint64_t past =
            !t.shape.empty() && prefill::mmq::supported(d.type) ? prefill::mmq::read_past_bytes(d.type, (int64_t) t.shape[0]) : 0;
        if (cudaMalloc(&d.ptr, d.bytes + past) != cudaSuccess) {
            return {model::Status::Error, "ernie55: out of device memory loading " + name};
        }
        if (past > 0 && cudaMemset((uint8_t*) d.ptr + d.bytes, 0, past) != cudaSuccess) {
            return {model::Status::Error, "ernie55: failed to zero read-past tail of " + name};
        }
        core::LoadItem it;
        it.file = file;
        it.off = t.offset;
        it.bytes = t.bytes;
        it.dev = d.ptr;
        it.device = device_;
        if (head_q8) it.conv = core::Conv::Bf16ToQ8_0;
        if (router_bf16) it.conv = core::Conv::Bf16High;
        reads.add(it);
        dense_bytes += d.bytes;
        dev_.emplace(name, std::move(d));
    }

    // Load bundled DSpark drafter when present and requested (--spec >= 2)
    if (pack_.dspark.present && req.spec >= 2) {
        const int gpu_cache = std::clamp(req.draft_experts, 0, 128);
        dspark_ = ErnieDsparkDrafter::plan_load(*this, file, reads, std::max(req.sessions, 1), gpu_cache, err);
        if (!dspark_) return {model::Status::Error, err};
        basalt::log::info("basalt ernie55",
                          "dspark drafter: %d stages, block=%d, gpu-cache=%d/stage, %.2f GiB GPU %d + %.2f GiB RAM, windows <= %d (--spec 0: off)",
                          dspark_->n_stages(), dspark_->block_size(), gpu_cache,
                          (double) dspark_->device_bytes() / (double) kGiB, device_,
                          (double) dspark_->host_bytes() / (double) kGiB,
                          std::min(req.spec, dspark_->block_size() + 1));
    }

    // Plan context against remaining VRAM
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int hca_ratio = g.compress_ratios.empty() ? 128 : std::max(1, g.compress_ratios.front());
        const ctxp::Plan p = ctxp::plan_context(
            req.max_context > 0 ? req.max_context : kDefaultContext, g.context_length,
            free_b / (uint64_t) std::max(req.sessions, 1), g.num_hca_layers(), g.num_dsa_layers(), hca_ratio,
            g.hca_head_dim, g.mla_kv_lora, g.mla_qk_rope, g.idx_dim, cache_fmt_);
        if (!p.error.empty()) return {model::Status::Error, "ernie55: " + p.error};
        ctx_ = p.context;
    }

    basalt::log::info("basalt ernie55", "%zu dense tensors (%.2f GiB on GPU %d), ctx=%lld, kv=%s, %d HCA + %d MLA-DSA",
                      dev_.size(), (double) dense_bytes / (double) kGiB, device_, (long long) ctx_,
                      cache_fp8() ? "fp8" : "fp16", g.num_hca_layers(), g.num_dsa_layers());

    flags_ = req;
    path_ = path;
    gpus_ = req.gpus;
    planning_ = &reads;
    model::Result r;
    for (int i = 0; r.ok() && i < std::max(req.sessions, 1); ++i) {
        std::unique_ptr<model::ModelSession> s = make_session(*this, err);
        if (!s) r = {model::Status::Error, "ernie55: " + err};
        else sessions.push_back(std::move(s));
    }
    if (r.ok()) r = sessions.front()->prepare();
    planning_ = nullptr;
    return r;
}

model::Result ErnieModel::attach(const model::LoadRequest&, std::vector<std::unique_ptr<model::ModelSession>>&) {
    std::string err;
    if (shared_tier_.tier && !shared_tier_.tier->attach(err)) return {model::Status::Error, "ernie55: " + err};
    return {};
}

model::PeerHooks* ErnieModel::peer() const { return shared_tier_.tier.get(); }

}  // namespace basalt::models::ernie55
