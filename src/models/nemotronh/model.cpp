// src/models/nemotronh/model.cpp - loads a Nemotron-H pack: the dense tensors to the main card, the embedding to mapped
// host memory, the routed experts to the expert tier.
#include "models/nemotronh/model.hpp"

#include "basalt/core/loader.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/ssd.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace basalt::models::nemotronh {
namespace {
constexpr uint64_t kGiB = 1ull << 30;
// CPU uses of a missed expert that earn it a slot on the card, copies a decode window may claim per row, and the tokens
// after which the lookup counts that pick the main card's victims halve: Hy4-flash's values (512 experts, a sigmoid
// router, a latent), not yet measured on this model
constexpr int kAdmitUses = 2;
constexpr int kClaimsPerRow = 128;
constexpr int kVictimHalfLife = 256;
}  // namespace

NemotronModel::NemotronModel() = default;

NemotronModel::~NemotronModel() {
    for (void* p : dev_) cudaFree(p);
    if (embd_host_) cudaFreeHost(embd_host_);
    if (shared_tier_.h_x) cudaFreeHost(shared_tier_.h_x);
    if (shared_tier_.h_rows) cudaFreeHost(shared_tier_.h_rows);
    if (fd_ >= 0) platform::close_fd(fd_);
}

size_t NemotronModel::expert_stride() const {
    // MMQ steps from expert to expert in whole blocks (NVFP4's 36 bytes) and reads up to 256 values past a matrix's last
    // row: the blob, then room past it, rounded to 36 and 16 bytes
    size_t blob = 0;
    for (const ExpertLayer& el : pack_.experts) blob = std::max<size_t>(blob, el.blob_bytes);
    constexpr size_t kUnit = 144, kReadPast = 256 / 64 * 36;   // lcm(36, 16); the most an NVFP4 row is read past
    return (blob + kReadPast + kUnit - 1) / kUnit * kUnit;
}

core::ExpertTierSpec NemotronModel::tier_spec() const {
    const Geometry& g = pack_.geo;
    core::ExpertTierSpec s;
    s.area = "basalt nemotronh";
    s.name = "nemotronh";
    s.n_layers = g.count(Kind::Moe);
    s.n_expert = g.n_expert;
    s.top_k = g.n_used;
    s.n_embd = g.n_latent;   // the experts run over the latent ffn_latent_down makes
    s.n_ff = g.n_ff_exp;
    for (int l = 0; l < g.n_layers; ++l) {
        const ExpertLayer& e = pack_.experts[(size_t) l];
        if (!e.present) continue;
        core::ExpertTierSpec::Layer tl{e.gu_type, e.d_type, e.offset, e.blob_bytes, 0.0f};
        tl.relu2 = true;
        s.layers.push_back(tl);
    }
    s.path = path_;
    s.fd = fd_;
    s.range = pack_.experts_range;
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
    s.claims_per_row = kClaimsPerRow;
    s.victim_half_life = kVictimHalfLife;
    s.share_stride = expert_stride();
    return s;
}

model::Result NemotronModel::plan(const model::LoadRequest& req, core::LoadPlan& reads,
                                  std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    if (req.paths.empty()) return {model::Status::Error, "nemotronh: no model path"};
    std::string err;
    const std::string& path = req.paths.front();
    if (!open_pack(path, pack_, err)) return {model::Status::Error, "nemotronh: " + err};
    cudaGetDevice(&device_);
    const Geometry& g = pack_.geo;
    // the shapes the kernels take
    const kernels::SsdGeometry sg{g.ssm_heads, g.ssm_head_dim, g.ssm_state, g.ssm_groups, g.ssm_conv};
    if (std::string e = kernels::ssd_check(sg); !e.empty()) return {model::Status::Error, "nemotronh: " + e};
    if (g.head_dim != 128 || g.n_head % g.n_kv || g.n_head / g.n_kv > 16)
        return {model::Status::Error, "nemotronh: attention of 128-wide heads, at most 16 query heads a KV head, is run"};
    if (g.n_expert > 512 || g.n_expert % 32 || g.n_used > 32 || g.n_embd % 128 || g.n_latent % 128 ||
        g.n_ff_exp > kernels::cpu::kNativeMaxFF)
        return {model::Status::Error, "nemotronh: a mixture-of-experts shape the kernels do not take"};
    desc_.arch = "nemotronh";
    desc_.n_layers = g.n_layers;
    desc_.n_embd = g.n_embd;
    desc_.n_vocab = g.n_vocab;
    desc_.n_expert = g.n_expert;
    desc_.n_expert_used = g.n_used;
    desc_.n_ff_exp = g.n_ff_exp;
    desc_.n_expert_shared = 1;
    desc_.context_length = g.context_length;
    for (const ExpertLayer& e : pack_.experts)
        if (e.present) {
            desc_.expert_gate_up_type.push_back(e.gu_type);
            desc_.expert_down_type.push_back(e.d_type);
        }

    fd_ = platform::open_fd_utf8(path);
    if (fd_ < 0) return {model::Status::Error, "nemotronh: cannot open " + path};
    const int file = reads.file_fd(fd_);

    // every tensor but the embedding to the card, one allocation each
    std::map<std::string, void*> placed;
    uint64_t dense = 0;
    for (const auto& [name, t] : pack_.tensors) {
        core::LoadItem it;
        it.file = file;
        it.off = t.offset;
        it.bytes = t.bytes;
        if (name == "token_embd.weight") {
            if (t.type != kTypeBf16 || t.shape.size() != 2 || t.shape[0] != (uint64_t) g.n_embd)
                return {model::Status::Error, "nemotronh: token_embd.weight is not BF16 [n_embd][vocab]"};
            if (cudaHostAlloc((void**) &embd_host_, t.bytes, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &embd_dev_, embd_host_, 0) != cudaSuccess)
                return {model::Status::Error, "nemotronh: cannot pin the token embedding"};
            it.host = (uint8_t*) embd_host_;
            reads.add(it);
            continue;
        }
        void* p = nullptr;
        if (cudaMalloc(&p, (t.bytes + 255) & ~(uint64_t) 255) != cudaSuccess)
            return {model::Status::Error, "nemotronh: out of device memory loading " + name};
        dev_.push_back(p);
        it.dev = p;
        it.device = device_;
        reads.add(it);
        placed[name] = p;
        dense += t.bytes;
    }
    if (!embd_host_) return {model::Status::Error, "nemotronh: the pack has no token_embd.weight"};

    const auto f32 = [&](const std::string& n, uint64_t count, const float*& out) {
        const TensorRef* t = pack_.find(n);
        if (!t || t->type != kTypeF32 || t->bytes != count * 4) {
            err = n + " is missing or not F32 of " + std::to_string(count);
            return false;
        }
        out = (const float*) placed.at(n);
        return true;
    };
    // a projection `base`.weight [cols][rows] of `type` (and its `base`.scale for F8 and NVFP4)
    const auto mat = [&](const std::string& base, int type, int64_t rows, int64_t cols, Mat& m) {
        const TensorRef* t = pack_.find(base + ".weight");
        if (!t || t->type != type || t->shape.size() != 2 || (int64_t) t->shape[0] != cols || (int64_t) t->shape[1] != rows) {
            err = base + ".weight is missing, not of type " + std::to_string(type) + " or not [" + std::to_string(cols) + "][" +
                  std::to_string(rows) + "]";
            return false;
        }
        m.type = type;
        m.w = (const uint8_t*) placed.at(base + ".weight");
        m.rows = rows;
        m.cols = cols;
        if (type == kTypeF8 || type == kTypeNvfp4Planes) return f32(base + ".scale", 1, m.scale);
        return true;
    };
    const int64_t E = g.n_embd, PW = g.proj_width(), DI = g.ssm_inner, C = g.conv_dim(), H = g.ssm_heads;
    layers_.resize((size_t) g.n_layers);
    int tl = 0;
    for (int l = 0; l < g.n_layers; ++l) {
        Layer& L = layers_[(size_t) l];
        const std::string b = "blk." + std::to_string(l) + ".";
        L.kind = g.kinds[(size_t) l];
        bool ok = f32(b + "attn_norm.weight", (uint64_t) E, L.norm);
        if (L.kind == Kind::Mamba) {
            ok = ok && mat(b + "ssm_in", kTypeF8, PW, E, L.in) && mat(b + "ssm_out", kTypeF8, E, DI, L.out) &&
                 f32(b + "ssm_conv1d.weight", (uint64_t) C * g.ssm_conv, L.conv_w) && f32(b + "ssm_conv1d.bias", (uint64_t) C, L.conv_b) &&
                 f32(b + "ssm_dt.bias", (uint64_t) H, L.dt_bias) && f32(b + "ssm_a", (uint64_t) H, L.A) &&
                 f32(b + "ssm_d", (uint64_t) H, L.D) && f32(b + "ssm_norm.weight", (uint64_t) DI, L.ssm_norm);
        } else if (L.kind == Kind::Attention) {
            const int64_t qw = (int64_t) g.n_head * g.head_dim, kw = (int64_t) g.n_kv * g.head_dim;
            ok = ok && mat(b + "attn_q", kTypeF8, qw, E, L.q) && mat(b + "attn_k", kTypeF8, kw, E, L.k) &&
                 mat(b + "attn_v", kTypeF8, kw, E, L.v) && mat(b + "attn_output", kTypeF8, E, qw, L.o);
        } else {
            ok = ok && f32(b + "ffn_gate_inp.weight", (uint64_t) g.n_expert * E, L.router) &&
                 f32(b + "exp_probs_b.bias", (uint64_t) g.n_expert, L.router_bias) &&
                 mat(b + "ffn_up_shexp", kTypeNvfp4Planes, g.n_ff_shared, E, L.sh_up) &&
                 mat(b + "ffn_down_shexp", kTypeNvfp4Planes, E, g.n_ff_shared, L.sh_down);
            if (ok && g.n_latent != g.n_embd)
                ok = mat(b + "ffn_latent_down", kTypeBf16, g.n_latent, E, L.lat_down) &&
                     mat(b + "ffn_latent_up", kTypeBf16, E, g.n_latent, L.lat_up);
            if (ok && !pack_.experts[(size_t) l].present) {
                err = "layer " + std::to_string(l) + " has no experts in experts.layout";
                ok = false;
            }
            L.tier_layer = tl++;
        }
        if (!ok) return {model::Status::Error, "nemotronh: " + err};
    }
    if (!f32("output_norm.weight", (uint64_t) E, out_norm_) || !mat("output", kTypeF8, g.n_vocab, E, head_))
        return {model::Status::Error, "nemotronh: " + err};

    ctx_ = req.max_context > 0 ? req.max_context : kDefaultContext;
    if (g.context_length > 0 && ctx_ > g.context_length)
        return {model::Status::Error, "nemotronh: --max-context " + std::to_string(ctx_) + " is past the model's " +
                                          std::to_string(g.context_length)};
    basalt::log::info("basalt nemotronh", "%zu dense tensors, %.2f GiB on the GPU; %d Mamba-2, %d attention and %d MoE layers",
                      placed.size(), (double) dense / (double) kGiB, g.count(Kind::Mamba), g.count(Kind::Attention),
                      g.count(Kind::Moe));

    // the sessions first, then the expert tier, sized from what they leave and opened by the first session; its reads are
    // queued in the start's plan
    flags_ = req;
    path_ = path;
    if (!req.peer_devices.empty())
        basalt::log::warn("basalt nemotronh", "%zu second cards given: this model's experts stay on the main card and in RAM",
                          req.peer_devices.size());
    planning_ = &reads;
    model::Result r;
    for (int i = 0; r.ok() && i < std::max(req.sessions, 1); ++i) {
        std::unique_ptr<model::ModelSession> s = make_session(*this, err);
        if (!s) r = {model::Status::Error, "nemotronh: " + err};
        else sessions.push_back(std::move(s));
    }
    if (r.ok()) r = sessions.front()->prepare();
    planning_ = nullptr;
    return r;
}

model::Result NemotronModel::attach(const model::LoadRequest&, std::vector<std::unique_ptr<model::ModelSession>>&) {
    std::string err;
    if (shared_tier_.tier && !shared_tier_.tier->attach(err)) return {model::Status::Error, "nemotronh: " + err};
    return {};
}

}  // namespace basalt::models::nemotronh
