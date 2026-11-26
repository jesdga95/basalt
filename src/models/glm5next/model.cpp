// src/models/glm5next/model.cpp - loads a glm5-next pack onto the GPU, then sizes the caches and one session from
// the pack geometry (§5.1).
#include "basalt/core/loader.hpp"
#include "basalt/artifact/gguf_reader.hpp"   // ggml_type_name, for the expert types a pack carries
#include "basalt/core/expert_tier.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "models/glm5next/model.hpp"
#include "models/glm5next/cache.hpp"
#include "models/glm5next/prefill_plan.hpp"      // window_bytes/kMaxChunk: what a prompt's window costs beside the homes
#include "models/glm5next/forward.hpp"             // forward_footprint_bytes, what the pass keeps off its window
#include "models/glm5next/kernels/expert_sum.hpp"   // kMaxExpertsPerToken, the sum's own bound
#include "models/glm5next/session.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <string>

namespace basalt::models::glm5next {

namespace {
constexpr uint64_t kModuleRoom = 1ull << 28;   // kernel modules still to load when the tier opens (glm5next has few)
constexpr int kScratchReserveMib = 512;        // the card the forward's per-window scratch keeps for itself
/// The piece size the tier maps its card copies in, so a window can borrow part of the arena and the rest of the
/// experts stay on the card (core::ExpertTier::lend). One expert index spans ~300 MiB of the real pack's arena, so
/// this is a little under a slot: a lend of a few hundred MiB gives up experts, not the arena.
constexpr uint64_t kExpertLendSegment = 256ull << 20;
/// The window a first or short request runs, which is what the tier prices in `beside_homes` (deepseek41's
/// prefillp::kShortChunk is the same number for the same reason).
constexpr int kShortWindow = 1024;
const ExpertLayer kNoExperts{};                // a layer the pack's expert layout does not list
}  // namespace

GlmModel::~GlmModel() {
    // The tiers write through the pinned expert rows, so they go first.
    tier_.reset();
    draft_tier_.reset();
    for (auto& [name, t] : dev_)
        if (t.ptr) cudaFree(t.ptr);
    if (h_rows_) cudaFreeHost(h_rows_);
    if (read_fd_ >= 0) platform::close_fd(read_fd_);
}

bool GlmModel::load(const std::string& path, core::LoadPlan& plan, std::string& err) {
    if (!open_pack(path, pack_, err)) return false;
    path_ = path;
    if (cudaGetDevice(&device_) != cudaSuccess) device_ = 0;

    const Geometry& g = pack_.geo;
    desc_.arch = "glm5next";
    desc_.n_layers = g.n_model_layers();   // the drafter block is not a model layer the engine runs
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

    read_fd_ = platform::open_fd_utf8(path);
    if (read_fd_ < 0) {
        err = "cannot open " + path;
        return false;
    }

    // dense tensors to the device; the token embedding stays in the file (read per token), the experts land through the
    // tier path with the session layer
    uint64_t total = 0;
    const int dense_file = plan.file_fd(read_fd_);
    for (const auto& [name, t] : pack_.tensors) {
        if (name == "token_embd.weight") continue;
        DevTensor d;
        d.name = name;
        d.type = t.type;
        d.bytes = t.bytes;
        d.shape = t.shape;
        if (cudaMalloc(&d.ptr, d.bytes) != cudaSuccess) {
            err = "out of device memory loading " + name;
            return false;
        }
        core::LoadItem it;
        it.file = dense_file;
        it.off = t.offset;
        it.bytes = t.bytes;
        it.dev = d.ptr;
        it.device = device_;
        it.conv = core::Conv::Copy;
        plan.add(it);
        total += d.bytes;
        dev_.emplace(name, std::move(d));
    }

    // The drafter's block keeps its float weights in a pool of their own, `mtp/dense.index`, because the pack's tensor
    // table lists only quantized tensors and every one of the block's norms, its router gate and its merge projections
    // is fp32. They come to the device the same way the model's own do: same file, absolute offsets, one read each.
    for (const auto& [name, t] : pack_.mtp_tensors) {
        DevTensor d;
        d.name = name;
        d.type = t.type;
        d.bytes = t.bytes;
        d.shape = t.shape;
        if (cudaMalloc(&d.ptr, d.bytes) != cudaSuccess) {
            err = "out of device memory loading " + name;
            return false;
        }
        core::LoadItem it;
        it.file = dense_file;
        it.off = t.offset;
        it.bytes = t.bytes;
        it.dev = d.ptr;
        it.device = device_;
        it.conv = core::Conv::Copy;
        plan.add(it);
        total += d.bytes;
        dev_.emplace(name, std::move(d));
    }

    // The routed experts do not come to the device: they stay in the pack's experts section, and the expert tier opened
    // below reads the ones a token routes to into pinned RAM places for the CPU pool to compute, while the card
    // computes the ones it homes straight from their packed rows. Both of those read the blob by the layout the pack's
    // own geometry implies, so check that here, before anything is allocated.
    if (!check_expert_layouts(err)) return false;
    check_draft_weights();
    basalt::log::info("basalt glm5next", "%zu dense tensors, %.2f MiB on the GPU; %zu expert layers in the tier",
                      dev_.size(), (double) total / (1024.0 * 1024.0), pack_.experts.size());
    if (pack_.geo.nextn > 0)
        basalt::log::info("basalt glm5next",
                          "the pack describes %d next-n drafter block(s) from blk.%d: their indexer is %s, their "
                          "experts %s (the pass says whether it can draft with them)",
                          pack_.geo.nextn, pack_.geo.n_layers - pack_.geo.nextn,
                          pack_.geo.index_share_mtp ? "the model's own" : "their own, which this pass does not run",
                          has_draft_experts() ? "have a card path" : "have none for the card");
    return true;
}

bool GlmModel::check_expert_layouts(std::string& err) {
    const Geometry& g = pack_.geo;
    if (g.n_used > kMaxExpertsPerToken) {
        err = "glm5next: the pack routes every token to " + std::to_string(g.n_used) +
              " experts; the expert sum covers " + std::to_string(kMaxExpertsPerToken);
        return false;
    }
    // One entry per model layer, positional: the tier and the forward pass both index by layer, and a pack whose
    // layout stops listing before the last layer (a trailing layer with no experts) still has to have a row there.
    elayouts_.assign((size_t) g.n_model_layers(), {});
    for (int l = 0; l < g.n_model_layers(); ++l) {
        const ExpertLayer& e = l < (int) pack_.experts.size() ? pack_.experts[(size_t) l] : kNoExperts;
        if (!e.blob_bytes) continue;   // a dense lead layer routes to none
        const std::string types = std::string(ggml_type_name((uint32_t) e.gu_type)) + " gate/up, " +
                                  ggml_type_name((uint32_t) e.d_type) + " down";
        if (!kernels::native_expert_supported(e.gu_type, e.d_type, g.n_embd, g.n_ff_exp)) {
            err = "glm5next: layer " + std::to_string(l) + "'s experts (" + types + ", " + std::to_string(g.n_embd) +
                  " x " + std::to_string(g.n_ff_exp) +
                  ") have no grouped kernel for the card to compute them from a device blob";
            return false;
        }
        const kernels::NativeExpertLayout L = kernels::native_expert_layout(
            e.gu_type, e.d_type, g.n_embd, g.n_ff_exp, l < (int) g.clamp_exp.size() ? g.clamp_exp[(size_t) l] : 0.f);
        if (L.bytes != e.blob_bytes) {
            err = "glm5next: layer " + std::to_string(l) + "'s expert blob is " + std::to_string(e.blob_bytes) +
                  " bytes; its " + types + " rows at " + std::to_string(g.n_embd) + " x " + std::to_string(g.n_ff_exp) +
                  " come to " + std::to_string(L.bytes);
            return false;
        }
        elayouts_[(size_t) l] = L;
    }
    // The drafter's block keeps its experts in a section of their own, in formats of their own, so the same two checks
    // run again for it: the card must have a grouped kernel for the types, and the blob must be the rows the layout
    // describes. It is a warning rather than a refusal because a pack whose drafter the card cannot compute still runs
    // the model, token by token; only the drafting is lost, and the pass says so when it builds itself.
    const int dl = g.n_model_layers();
    if (dl < (int) pack_.mtp_experts.size() && pack_.mtp_experts[(size_t) dl].blob_bytes) {
        const ExpertLayer& e = pack_.mtp_experts[(size_t) dl];
        const std::string types = std::string(ggml_type_name((uint32_t) e.gu_type)) + " gate/up, " +
                                  ggml_type_name((uint32_t) e.d_type) + " down";
        const kernels::NativeExpertLayout L = kernels::native_expert_layout(
            e.gu_type, e.d_type, g.n_embd, g.n_ff_exp, dl < (int) g.clamp_exp.size() ? g.clamp_exp[(size_t) dl] : 0.f);
        if (!kernels::native_expert_supported(e.gu_type, e.d_type, g.n_embd, g.n_ff_exp)) {
            basalt::log::warn("basalt glm5next",
                              "blk.%d's experts (%s, %lld x %lld) have no grouped kernel for the card; the drafter "
                              "would have to compute them on the CPU pool alone",
                              dl, types.c_str(), (long long) g.n_embd, (long long) g.n_ff_exp);
        } else if (L.bytes != e.blob_bytes) {
            basalt::log::warn("basalt glm5next",
                              "blk.%d's expert blob is %llu bytes; its %s rows at %lld x %lld come to %llu, so the "
                              "drafter's experts are not the layout the card reads",
                              dl, (unsigned long long) e.blob_bytes, types.c_str(), (long long) g.n_embd,
                              (long long) g.n_ff_exp, (unsigned long long) L.bytes);
        } else {
            draft_elayout_ = L;
        }
    }
    return true;
}

/// What the drafter's block has to carry to run, named as the geometry has it: a tensor suffix and its GGUF-order
/// shape, innermost first - the order the pack's tables and the pass's products both read. Its indexer is deliberately
/// absent: a pack that states `index_share_mtp` (the released one does) selects with the model's own indexer, so the
/// block's indexer weights are never read. The block's float weights live in its own pool (`mtp/dense.index`), its
/// quantized ones in the pack's tensor table like every other one, so both tables are asked.
void GlmModel::check_draft_weights() const {
    const Geometry& g = geo();
    if (g.nextn <= 0) return;
    const int dl = g.n_model_layers();
    const uint64_t E = (uint64_t) g.n_embd, QL = (uint64_t) g.q_lora, KV = (uint64_t) g.kv_lora;
    const uint64_t QK = (uint64_t) g.qk_dim, V = (uint64_t) g.v_dim, nH = (uint64_t) g.n_head;
    const uint64_t SH = (uint64_t) g.shared_ffn() * (uint64_t) g.n_shared, NE = (uint64_t) g.n_expert;
    struct Want {
        const char* suffix;
        std::vector<uint64_t> shape;
    };
    const Want want[] = {
        {"attn_q_a.weight", {E, QL}},
        {"attn_q_a_norm.weight", {QL}},
        {"attn_q_b.weight", {QL, nH * QK}},
        {"attn_kv_a_mqa.weight", {E, KV}},
        {"attn_kv_a_norm.weight", {KV}},
        {"attn_k_b.weight", {QK, KV, nH}},
        {"attn_v_b.weight", {KV, V, nH}},
        {"attn_output.weight", {nH * V, E}},
        {"attn_norm.weight", {E}},
        {"ffn_norm.weight", {E}},
        {"ffn_gate_inp.weight", {E, NE}},
        {"exp_probs_b.bias", {NE}},
        {"ffn_gate_shexp.weight", {E, SH}},
        {"ffn_up_shexp.weight", {E, SH}},
        {"ffn_down_shexp.weight", {SH, E}},
        {"nextn.eh_proj.weight", {2 * E, E}},
        {"nextn.enorm.weight", {E}},
        {"nextn.hnorm.weight", {E}},
        {"nextn.shared_head_norm.weight", {E}},
    };
    std::string missing, wrong;
    // The pack's tensor table carries a per-head weight's full [K][N][H]; a pool row cannot, because the index states
    // only two extents, so it flattens everything past the first ([K][N*H]). The products read both the same way - K
    // inputs, the rest as rows - so that is what gets compared.
    auto rows = [](const std::vector<uint64_t>& s) {
        uint64_t n = 1;
        for (size_t i = 1; i < s.size(); ++i) n *= s[i];
        return n;
    };
    for (const Want& w : want) {
        const TensorRef* t = pack_.find(dl, w.suffix);
        if (!t) t = pack_.find_mtp(dl, w.suffix);
        if (!t) {
            if (!missing.empty()) missing += ", ";
            missing += w.suffix;
            continue;
        }
        if (t->shape[0] != w.shape[0] || rows(t->shape) != rows(w.shape)) {
            std::string got, asks;
            for (uint64_t d : t->shape) got += (got.empty() ? "" : "x") + std::to_string(d);
            for (uint64_t d : w.shape) asks += (asks.empty() ? "" : "x") + std::to_string(d);
            if (!wrong.empty()) wrong += ", ";
            wrong += std::string(w.suffix) + " is " + got + ", the geometry wants " + asks;
        }
    }
    if (missing.empty() && wrong.empty()) return;
    std::string msg = "blk." + std::to_string(dl) +
                      " is the drafter's block and is not the block the geometry describes";
    if (!missing.empty()) msg += ": it holds no " + missing;
    if (!wrong.empty()) msg += (missing.empty() ? ": " : "; ") + wrong;
    msg += " - the drafter refuses to run rather than draft from a block it cannot complete";
    basalt::log::warn("basalt glm5next", "%s", msg.c_str());
}

model::Result GlmModel::plan(const model::LoadRequest& req, core::LoadPlan& reads,
                             std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    if (req.paths.empty()) return {model::Status::Error, "glm5next: no model path"};
    std::string err;
    if (!load(req.paths.front(), reads, err)) return {model::Status::Error, "glm5next: " + err};

    // Size the three caches and one session from the pack geometry; the session owns its caches, stream and logits.
    const Geometry& g = geo();
    const CacheDims dims = cache_dims(g);
    if (dims.n_kda == 0 && dims.n_mla == 0) {
        return {model::Status::Error, "glm5next: geometry has no KDA or MLA layers"};
    }
    int64_t context = req.max_context > 0 ? req.max_context : g.context_length;
    if (context <= 0) context = 1024;
    // The start's flags and the caches' size, both of which the expert tier sizes itself against.
    req_ = req;
    req_.sessions = 1;
    cache_bytes_ = cache_bytes(dims, (uint64_t) context, 1);
    footprint_ = forward_footprint_bytes(g, pack_);
    if (!ensure_tier(reads, err)) return {model::Status::Error, "glm5next: " + err};
    // A drafter that cannot open its tier is a loss of speed, not of correctness: every draft the engine would verify
    // simply never happens, and the pass says so once at load.
    std::string derr;
    if (!ensure_draft_tier(reads, derr)) basalt::log::warn("basalt glm5next", "%s", derr.c_str());
    std::unique_ptr<model::ModelSession> s = make_session_impl(*this, dims, (uint64_t) context, 1, err);
    if (!s) return {model::Status::Error, err};
    sessions.push_back(std::move(s));
    return {};
}

core::ExpertTierSpec GlmModel::tier_spec() const {
    const Geometry& g = pack_.geo;
    core::ExpertTierSpec s;
    s.area = "basalt glm5next";
    s.name = "glm5next";
    s.n_layers = g.n_model_layers();   // the drafter's block is not a model layer, and its experts are a separate tier
    s.n_expert = g.n_expert;
    s.top_k = g.n_used;
    s.n_embd = g.n_embd;
    s.n_ff = g.n_ff_exp;
    // positional: the tier reads layers[layer], so a layer the layout does not list still gets its row (and the tier
    // takes a layer with no blob as a layer with no experts)
    for (int l = 0; l < g.n_model_layers(); ++l) {
        const ExpertLayer& e = l < (int) pack_.experts.size() ? pack_.experts[(size_t) l] : kNoExperts;
        s.layers.push_back({e.gu_type, e.d_type, e.offset, e.blob_bytes,
                            l < (int) g.clamp_exp.size() ? g.clamp_exp[(size_t) l] : 0.f});
    }
    s.path = path_;
    s.fd = read_fd_;
    s.range = pack_.experts_range;
    s.gpus = req_.gpus;
    s.diag_tier_ram = req_.diag_tier_ram;
    s.diag_tier_vram = req_.diag_tier_vram;
    s.diag_tier_frozen = req_.diag_tier_frozen;
    s.route_tail_skip = req_.route.tail_skip;
    s.route_prior = req_.route.prior;
    s.vram_reserve_mib = req_.vram_reserve_mib;
    s.peer_reserve_mib = req_.peer_reserve_mib;
    s.expert_cache = req_.expert_cache;
    s.resident_budget = req_.resident_budget;
    s.pool_workers = req_.pool_workers;
    s.pool_affinity = req_.pool_affinity;
    s.pool_affinity_given = req_.pool_affinity_given;
    s.expert_profile = req_.expert_profile;
    s.lend_segment = kExpertLendSegment;
    return s;
}

bool GlmModel::ensure_tier(core::LoadPlan& reads, std::string& err) {
    if (tier_ || pack_.experts.empty()) return true;
    // The pass's own buffers - the expert block's activations and scratch, the packed conv taps, and the fp32 copies it
    // makes of the packed weights it multiplies as rows - are allocated outside the tier, after it has planned, and a
    // window goes on top of them. Leave the card room for the pass and a window, or the first request runs out of
    // memory part way through the layers while the tier sits on the room they needed.
    const int own_mib = (int) (footprint_ / (1ull << 20));
    if (req_.vram_reserve_mib < kScratchReserveMib + own_mib) req_.vram_reserve_mib = kScratchReserveMib + own_mib;
    basalt::log::info("basalt glm5next", "the pass holds %d MiB on the card outside a window, so the tier leaves at least %d MiB free",
                      own_mib, req_.vram_reserve_mib);
    if (cudaHostAlloc(reinterpret_cast<void**>(&h_rows_),
                      (size_t) kExpertRows * geo().n_used * geo().n_embd * sizeof(float), cudaHostAllocMapped) !=
        cudaSuccess) {
        err = "cannot allocate the pinned expert rows";
        return false;
    }
    // The same rows as the card sees them: the block's sum reads the CPU pool's rows beside the ones the card's own
    // experts wrote, and one launch over both is what keeps the pool off the token path.
    if (cudaHostGetDevicePointer(reinterpret_cast<void**>(&h_rows_dev_), h_rows_, 0) != cudaSuccess) {
        err = "cannot map the pinned expert rows for the card";
        cudaFreeHost(h_rows_);
        h_rows_ = nullptr;
        return false;
    }
    tier_ = std::make_shared<core::ExpertTier>(tier_spec());
    const Geometry& g = pack_.geo;
    const int64_t ctx = req_.max_context > 0 ? req_.max_context : (int64_t) g.context_length;
    const int window_full = (int) std::min<int64_t>(prefillp::kMaxChunk, std::max<int64_t>(ctx, 1));
    const int window_short = (int) std::min<int64_t>(kShortWindow, std::max<int64_t>(ctx, 1));
    const uint64_t caches = cache_bytes_ * (uint64_t) req_.sessions;
    core::ExpertTier::Reserve r;
    // the caches are resident for the whole start, so the tier plans around them; kModuleRoom is the kernel modules the
    // first token still has to load (core/expert_tier.hpp's own reason for the field)
    r.copies = caches + kModuleRoom;
    // Beside the homes a prompt keeps the caches at the full context *and* a window, so the homes have to leave that
    // much room: the longest chunk the plan ever runs (prefillp::kMaxChunk) priced at this context. Ask for nothing and
    // the arena covers the card, so every long prompt lends the tier's card copies back at runtime - session.cpp's
    // window_cap logs "the expert tier gives up N MiB of card copies so a window fits" - and one that cannot lend
    // enough is refused outright. `beside_homes` is the floor kept while homes and their copies sit together, which is
    // what a first or short request runs its window against; the caches are out of it because they are already
    // allocated by then. deepseek41's ops_kernels.cpp prices the same two arrangements for its caches and staged groups.
    r.homes = caches + prefillp::window_bytes(g, (uint64_t) ctx, window_full) + prefillp::kMargin + kModuleRoom;
    r.beside_homes = prefillp::window_bytes(g, (uint64_t) ctx, window_short) + prefillp::kMargin;
    r.modules = kModuleRoom;
    if (!tier_->open(r, h_rows_, reads, err)) {
        tier_.reset();
        return false;
    }
    return true;
}

/// The drafter's tier: the same cards, pool and flags as the model's, one layer, and the block's own experts section.
core::ExpertTierSpec GlmModel::draft_spec() const {
    const Geometry& g = pack_.geo;
    core::ExpertTierSpec s = tier_spec();
    s.area = "basalt glm5next drafter";
    s.name = "glm5next-mtp";
    s.n_layers = 1;   // the block is not a model layer, and it is the only one this tier places
    // The pack lists the drafter's experts where the block sits: after the model's layers, by its own index.
    const int dl = g.n_model_layers();
    const ExpertLayer& e = dl < (int) pack_.mtp_experts.size() ? pack_.mtp_experts[(size_t) dl] : kNoExperts;
    s.layers.assign(1, {e.gu_type, e.d_type, e.offset, e.blob_bytes,
                        dl < (int) g.clamp_exp.size() ? g.clamp_exp[(size_t) dl] : 0.f});
    s.range = pack_.mtp_experts_range;
    // The card room it asks for is deliberately small. A draft is one row, its picks are `n_used` experts of a block
    // the model's own 45 layers compete with for the card, and the RAM tier reads the rest of them anyway: the room
    // the drafter leaves is room the model's tier keeps for the tokens that have to be computed either way.
    s.expert_cache = 2 * g.n_used;
    s.diag_tier_vram = -1;   // the flag that pins expert counts is about the model's layers, not this one
    return s;
}

bool GlmModel::ensure_draft_tier(core::LoadPlan& reads, std::string& err) {
    const Geometry& g = pack_.geo;
    if (draft_tier_ || g.nextn <= 0) return true;
    if (!drafting_asked()) return true;   // --spec off, or an oracle's tokens in the window's place: the block's experts stay in the file, GiB of them on a real pack
    if (!g.index_share_mtp) {
        err = "glm5next: blk." + std::to_string(g.n_layers - g.nextn) + " is a next-n block with an indexer of its own,"
              " which this pass does not run: the pack drafts nothing";
        return false;
    }
    if (!has_draft_experts()) return true;   // check_expert_layouts already said why the card has no rows for them
    const core::ExpertTierSpec spec = draft_spec();
    if (spec.layers.empty() || spec.layers[0].blob_bytes == 0) {
        err = "glm5next: blk." + std::to_string(g.n_layers - g.nextn) +
              " has an expert layout the card can read but no blob behind it: the pack drafts nothing";
        return false;
    }
    draft_tier_ = std::make_shared<core::ExpertTier>(spec);
    core::ExpertTier::Reserve r;
    // The caches, the pass and a window are already inside the model tier's promise, so this one asks for nothing
    // beside its own places and the modules it still has to load.
    r.copies = kModuleRoom;
    r.homes = r.copies;
    r.beside_homes = 0;
    r.modules = kModuleRoom;
    if (!draft_tier_->open(r, h_rows_, reads, err)) {
        draft_tier_.reset();
        err = "glm5next: the drafter's expert tier did not open: " + err;
        return false;
    }
    return true;
}

model::Result GlmModel::attach(const model::LoadRequest&, std::vector<std::unique_ptr<model::ModelSession>>&) {
    std::string err;
    if (tier_ && !tier_->attach(err)) return {model::Status::Error, "glm5next: " + err};
    if (draft_tier_ && !draft_tier_->attach(err)) return {model::Status::Error, "glm5next: " + err};
    return {};
}

const DevTensor* GlmModel::tensor(const std::string& name) const {
    auto it = dev_.find(name);
    return it == dev_.end() ? nullptr : &it->second;
}

}  // namespace basalt::models::glm5next
