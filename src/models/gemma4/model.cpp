// src/models/gemma4/model.cpp - loads a Gemma 4 pack: the dense weights to the main card, the embedding to mapped host
// memory, the context sized from what they leave, then the routed experts to core::ExpertTier, which keeps on the card as
// many as the rest of it holds and the others in RAM.
#include "models/gemma4/model.hpp"

#include "models/gemma4/experts.hpp"
#include "models/gemma4/prompt.hpp"

#include "basalt/core/gpu_mem_info.hpp"
#include "basalt/core/device_copy.hpp"
#include "basalt/core/expert_tier.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/nvfp4_planes.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace basalt::models::gemma4 {
namespace {

constexpr uint64_t kGiB = 1ull << 30;
constexpr uint64_t kMiB = 1ull << 20;
constexpr int kTypeF32 = 0;
constexpr int kTypeBF16 = 30;
constexpr int kTypeNvfp4 = 40;
constexpr int kTypeMxfp8 = 46;
constexpr int kTypeF8 = 44;
constexpr int kTypeNvfp4Planes = 45;
// what each session needs beside its caches (its graphs and staging), and what the card keeps free for the kernels'
// modules and the driver
constexpr uint64_t kSessionFixed = 96 * kMiB;
constexpr uint64_t kCardSpare = 1024 * kMiB;

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

void mlp_workspace(const Geometry& g, int64_t n_ff_pad, bool mlp_nvfp4, size_t& partial, size_t& tickets) {
    namespace wg = basalt::kernels::wgemm;
    partial = tickets = 0;
    if (!mlp_nvfp4) return;
    wg::Call gu;   // the gate and up products in one call, then the down product
    gu.fmt = wg::Format::Nvfp4;
    gu.K = g.n_embd;
    gu.m[0].rows = gu.m[1].rows = n_ff_pad;
    gu.n = 2;
    wg::Call dn;
    dn.fmt = wg::Format::Nvfp4;
    dn.K = n_ff_pad;
    dn.m[0].rows = g.n_embd;
    for (const wg::Call* c : {&gu, &dn}) {
        size_t p = 0, t = 0;
        wg::workspace_need(*c, p, t);
        partial = std::max(partial, p);
        tickets = std::max(tickets, t);
    }
}

uint64_t shared_bytes(const Geometry& g, int64_t n_ff_pad, bool mlp_nvfp4, int64_t context, KvType kv, bool mxfp8) {
    const uint64_t R = kRows, E = (uint64_t) g.n_embd;
    const uint64_t qmax = (uint64_t) g.n_head * std::max(g.hd_swa, g.hd_full);
    const uint64_t rowbuf = R * (4 * E + 3 * qmax + 3 * (uint64_t) n_ff_pad) * 4;
    const uint64_t experts = g.moe() ? R * (uint64_t) g.n_used * (E * 4 + (uint64_t) g.n_ff_exp * 16) + (64ull << 20) : 0;
    const uint64_t attn = attn_scratch_floats(kRows, g.n_head, std::max(g.hd_swa, g.hd_full)) * 4;
    size_t ws_p = 0, ws_t = 0;
    mlp_workspace(g, n_ff_pad, mlp_nvfp4, ws_p, ws_t);
    return rowbuf + experts + attn + (ws_p + ws_t) * 4 + R * (uint64_t) g.n_vocab * 4 + 256 * kMiB +
           PromptPath::bytes(g, n_ff_pad, mlp_nvfp4, context, kv, mxfp8);
}

Gemma4Model::Gemma4Model() = default;

ExpertRoom Gemma4Model::expert_room() const {
    ExpertRoom r;
    for (const ExpertLayer& el : pack_.experts) {
        const uint64_t slot = (el.blob_bytes + 255) / 256 * 256;   // the tier's slots are 256-byte aligned
        r.one_each += slot;
        r.slot = std::max(r.slot, slot);
    }
    return r;
}

// The homes a layer, h, leave the room of a prompt's staging of the other n_expert - h experts in two layers' buffers:
// h one_each + 2 (n_expert - h) slot <= what the context leaves. The tier sizes its homes from the same room, so it holds
// at least h a layer and the staging fits once its copies beside the homes give way (shrink).
bool Gemma4Model::open_tier(const model::LoadRequest& req, uint64_t need, uint64_t free_b, core::LoadPlan& reads, std::string& err) {
    const Geometry& g = pack_.geo;
    const ExpertRoom room = expert_room();
    const uint64_t left = free_b > need ? free_b - need : 0, all = 2 * (uint64_t) g.n_expert * room.slot;
    int64_t h = left > all && room.one_each > 2 * room.slot ? (int64_t) ((left - all) / (room.one_each - 2 * room.slot)) : 0;
    h = std::min<int64_t>(h, g.n_expert);
    // an exact --expert-cache or the diagnostic's homes cap them as the tier does
    if (req.expert_cache > 0) h = std::min<int64_t>(h, req.expert_cache / g.n_layers);
    if (req.diag_tier_vram >= 0) h = std::min<int64_t>(h, req.diag_tier_vram);
    stage_experts_ = (int) (g.n_expert - h);
    stage_slot_ = room.slot;
    core::ExpertTierSpec s;
    s.area = "basalt gemma4";
    s.name = "gemma4";
    s.n_layers = g.n_layers;
    s.n_expert = g.n_expert;
    s.top_k = g.n_used;
    s.n_embd = g.n_embd;
    s.n_ff = g.n_ff_exp;
    for (const ExpertLayer& el : pack_.experts) {
        core::ExpertTierSpec::Layer lay;
        lay.gu_type = el.gu_type;
        lay.d_type = el.d_type;
        lay.offset = el.offset;
        lay.blob_bytes = el.blob_bytes;
        lay.gelu = true;
        s.layers.push_back(lay);
    }
    s.path = req.paths.front();
    s.fd = fd_;
    s.range = pack_.experts_range;
    s.gpus = req.gpus;
    s.diag_tier_ram = req.diag_tier_ram;
    s.diag_tier_vram = req.diag_tier_vram;
    s.diag_tier_frozen = req.diag_tier_frozen;
    s.vram_reserve_mib = 0;   // the card's spare is in `need`
    s.expert_cache = req.expert_cache;
    s.resident_budget = req.resident_budget;
    s.pool_workers = req.pool_workers;
    s.pool_affinity = req.pool_affinity;
    s.pool_affinity_given = req.pool_affinity_given;
    s.expert_profile = req.expert_profile;
    s.keep_homes = true;   // a prompt stages the RAM experts beside the homes, it borrows no slot
    if (cudaHostAlloc((void**) &tier_rows_, (size_t) kMoePairs * g.n_embd * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable) !=
        cudaSuccess) {
        cudaGetLastError();
        tier_rows_ = nullptr;
        err = "cannot pin the experts' host rows";
        return false;
    }
    tier_ = std::make_unique<core::ExpertTier>(std::move(s));
    core::ExpertTier::Reserve res;
    res.copies = res.homes = need + 2 * (uint64_t) stage_experts_ * room.slot;
    res.beside_homes = need;
    return tier_->open(res, tier_rows_, reads, err);
}

Gemma4Model::~Gemma4Model() {
    experts_.reset();   // its thread first, then the tier it reads
    tier_.reset();
    if (tier_rows_) cudaFreeHost(tier_rows_);
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

bool Gemma4Model::mat(const std::string& name, int64_t rows, int64_t K, core::LoadPlan& reads, int file, Mat& m,
                      std::string& err, bool per_row, bool nvfp4) {
    namespace nvfp4p = basalt::kernels::nvfp4p;
    const TensorRef* t = pack_.find(name + ".weight");
    const int type = nvfp4 ? kTypeNvfp4Planes : kTypeF8;
    const uint64_t want = nvfp4 ? (uint64_t) nvfp4p::bytes(rows, K) : (uint64_t) rows * K;
    if (!t || t->type != type || t->bytes != want || t->shape.size() != 2 || (int64_t) t->shape[0] != K ||
        (nvfp4 && !nvfp4p::fits(rows, K))) {
        err = name + ".weight" + (t ? std::string(" is not ") + (nvfp4 ? "NVFP4_PLANES" : "F8_E4M3") + " [" + std::to_string(rows) +
                                          "][" + std::to_string(K) + "]"
                                    : " is missing");
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
    m.fmt = nvfp4 ? basalt::kernels::wgemm::Format::Nvfp4 : basalt::kernels::wgemm::Format::F8;
    weight_bytes_ += t->bytes;
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
    if (!mat("output", V, E, reads, file, head_, err, true)) return fail(err);
    const TensorRef* down0 = pack_.find("blk.0.ffn_down.weight");
    mlp_nvfp4_ = down0 && down0->type == kTypeNvfp4Planes;
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
        if (mlp_nvfp4_ && (n_ff_pad_ % 256 || E % 256))   // the prompt's FP4 products take K in multiples of 256
            return fail("an NVFP4 MLP whose widths are not multiples of 256");
    }

    layers_.assign((size_t) g.n_layers, LayerWeights{});
    for (int l = 0; l < g.n_layers; ++l) {
        LayerWeights& w = layers_[(size_t) l];
        const std::string b = "blk." + std::to_string(l) + ".";
        w.sh = g.shape(l);
        w.rope = w.sh.window > 0 ? rope_swa_ : rope_full_;
        const int64_t qd = (int64_t) w.sh.n_head * w.sh.hd, kd = (int64_t) w.sh.n_kv * w.sh.hd;
        auto S = [&](const char* suffix, uint64_t floats) { return small(b + suffix, floats, reads, file, err); };
        if (!mat(b + "attn_q", qd, E, reads, file, w.q, err) || !mat(b + "attn_k", kd, E, reads, file, w.k, err) ||
            (!w.sh.one && !mat(b + "attn_v", kd, E, reads, file, w.v, err)) ||
            !mat(b + "attn_output", E, qd, reads, file, w.o, err) ||
            !mat(b + "ffn_gate", n_ff_pad_, E, reads, file, w.gate, err, false, mlp_nvfp4_) ||
            !mat(b + "ffn_up", n_ff_pad_, E, reads, file, w.up, err, false, mlp_nvfp4_) ||
            !mat(b + "ffn_down", E, n_ff_pad_, reads, file, w.down, err, false, mlp_nvfp4_))
            return fail(err);
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
        if (el.gu_type != el.d_type || (el.gu_type != kTypeNvfp4 && el.gu_type != kTypeMxfp8))
            return fail("experts of a type other than NVFP4 or MXFP8 at layer " + std::to_string(l));
        if (el.gu_type == kTypeMxfp8 && E % 128)
            return fail("MXFP8 experts need a hidden size that is a multiple of 128 (layer " + std::to_string(l) + ")");
        if (el.blob_bytes != basalt::kernels::native_expert_layout(el.gu_type, el.d_type, E, g.n_ff_exp).bytes)
            return fail("experts.layout: the blobs of layer " + std::to_string(l) + " are not the size of their type");
        mxfp8_experts_ = mxfp8_experts_ || el.gu_type == kTypeMxfp8;
        w.gu_type = el.gu_type;
        w.d_type = el.d_type;
        w.blob_bytes = el.blob_bytes;
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
        if (!mat("assistant.pre_proj", DE, 2 * E, reads, file, d->pre, err) || !mat("assistant.post_proj", E, DE, reads, file, d->post, err) ||
            !mat("assistant.head", V, DE, reads, file, d->head, err, true) ||
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
            if (!mat(b + "attn_q", qd, DE, reads, file, w.q, err) || !mat(b + "attn_output", DE, qd, reads, file, w.o, err) ||
                !mat(b + "ffn_gate", d->n_ff_pad, DE, reads, file, w.gate, err) || !mat(b + "ffn_up", d->n_ff_pad, DE, reads, file, w.up, err) ||
                !mat(b + "ffn_down", DE, d->n_ff_pad, reads, file, w.down, err) ||
                !(w.attn_norm = S("attn_norm.weight", (uint64_t) DE)) || !(w.q_norm = S("attn_q_norm.weight", (uint64_t) w.sh.hd)) ||
                !(w.post_attn = S("post_attention_norm.weight", (uint64_t) DE)) || !(w.ffn_norm = S("ffn_norm.weight", (uint64_t) DE)) ||
                !(w.post_ffw = S("post_ffw_norm.weight", (uint64_t) DE)) || !(w.scalar = S("layer_scalar", 1)))
                return fail(err);
            d->layers.push_back(w);
        }
        drafter_ = std::move(d);
    }

    // [vision] the image encoder staged before the context is sized; it keeps none of the card: a request's pictures borrow
    // their region from the prompt path's buffers (lend_room) and give it back (refill). Only what those do not cover of
    // the region stays free beside the caches. What staging leaves on the card (the encoder's kernels, stream and product
    // handle) comes out of the card's spare, which is there for the kernels' modules.
    uint64_t vision_room = 0, vision_modules = 0;
    if (req.vision) {
        size_t before = 0, after = 0, total = 0;
        basalt::core::gpu_mem_info(&before, &total);
        if (!vis_.stage(pack_.vision, g.n_embd, g.image_cells, err)) return fail("vision: " + err);
        basalt::core::gpu_mem_info(&after, &total);
        vision_modules = std::min<uint64_t>(before > after ? before - after : 0, kCardSpare / 4);
        const uint64_t region = vis_.region_bytes(), lent = PromptPath::bytes(g, n_ff_pad_, mlp_nvfp4_, want, kv_, mxfp8_experts_);
        vision_room = region > lent ? region - lent : 0;
        vis_.set_lender({[this](uint64_t bytes, std::string& e) { return lend_room(bytes, e); }, [this] { refill(); }});
        basalt::log::info("basalt gemma4", "image encoder: %.2f GiB staged in page-locked RAM (its kernels' %.0f MiB inside the "
                          "card's spare), a request's pictures take %.2f GiB of the card while they are read, lent by the prompt "
                          "path's buffers (%.2f GiB)",
                          (double) vis_.weights_bytes() / (double) kGiB, (double) vision_modules / (double) kMiB,
                          (double) region / (double) kGiB, (double) lent / (double) kGiB);
    }

    // the context: what the weights leave, less the card's spare
    if (want < 1024 || (g.context_length > 0 && want > g.context_length))
        return fail("--max-context " + std::to_string(want) + " is outside 1,024 .. " + std::to_string(g.context_length));
    size_t free_b = 0, total_b = 0;
    basalt::core::gpu_mem_info(&free_b, &total_b);
    const uint64_t asked = req.vram_reserve_mib > 0 ? (uint64_t) req.vram_reserve_mib * kMiB : kCardSpare;
    card_spare_ = asked > vision_modules ? asked - vision_modules : 0;   // [vision]
    const uint64_t spare = card_spare_ + vision_room;
    const uint64_t n_sess = (uint64_t) std::max(req.sessions, 1);
    // the experts' least: top-k slots in every layer and a prompt's staging of every expert (two layers' worth)
    const ExpertRoom room = expert_room();
    const uint64_t least = g.moe() ? (uint64_t) g.n_used * room.one_each + 2 * (uint64_t) g.n_expert * room.slot : 0;
    const uint64_t need = session_bytes(g, want, kv_) * n_sess + shared_bytes(g, n_ff_pad_, mlp_nvfp4_, want, kv_, mxfp8_experts_) + spare;
    if (need + least > free_b) {
        const uint64_t per = (session_bytes(g, 2, kv_) - session_bytes(g, 1, kv_)) * n_sess +
                             shared_bytes(g, n_ff_pad_, mlp_nvfp4_, 2, kv_, mxfp8_experts_) -
                             shared_bytes(g, n_ff_pad_, mlp_nvfp4_, 1, kv_, mxfp8_experts_);
        const uint64_t base = session_bytes(g, 0, kv_) * n_sess + shared_bytes(g, n_ff_pad_, mlp_nvfp4_, 0, kv_, mxfp8_experts_) + spare + least;
        const int64_t fits = free_b > base ? (int64_t) ((free_b - base) / per) : 0;
        return fail("--max-context " + std::to_string(want) + " needs " + std::to_string((need + least) >> 20) +
                    " MiB beside the weights, " + std::to_string(free_b >> 20) + " MiB are free: at most about " + std::to_string(fits) +
                    " positions fit (a smaller --max-context, --kv fp8 or fewer --batch slots)");
    }
    ctx_ = want;
    uint64_t experts = 0;
    if (g.moe()) {   // the tier: the card's experts beside what the context needs, the rest in RAM
        if (!open_tier(req, need, free_b, reads, err)) return fail(err);
        for (const core::ExpertTier::CardSlots& c : tier_->slots_by_card()) experts += c.bytes;
    }
    int n_mx = 0;
    for (const ExpertLayer& el : pack_.experts) n_mx += el.gu_type == kTypeMxfp8;
    const std::string fmt = n_mx == 0 ? "NVFP4" : n_mx == (int) pack_.experts.size() ? "MXFP8"
                                                : "NVFP4, MXFP8 at " + std::to_string(n_mx) + " layers";
    char ex[160] = "";
    if (g.moe())
        std::snprintf(ex, sizeof ex, ", experts (%s) %.2f GiB on the card (%lld of %d a layer), %.2f GiB in RAM", fmt.c_str(),
                      (double) experts / (double) kGiB, (long long) (tier_->gpu_slots() / g.n_layers), g.n_expert,
                      (double) tier_->ram_bytes() / (double) kGiB);
    basalt::log::info("basalt gemma4", "weights: projections and head %.2f GiB (%s)%s, embedding %.2f GiB in RAM; context "
                      "%lld x %llu sessions, %s KV",
                      (double) weight_bytes_ / (double) kGiB, mlp_nvfp4_ ? "FP8, the MLP NVFP4" : "FP8", ex,
                      (double) V * E * 2 / (double) kGiB,
                      (long long) ctx_, (unsigned long long) n_sess, kv_ == KvType::F8 ? "fp8" : "fp16");
    return {};
}

model::Result Gemma4Model::attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    if (tier_) {
        std::string err;
        if (!tier_->attach(err) || !(experts_ = TierExperts::make(*this, *tier_, tier_rows_, err)))
            return {model::Status::Error, "gemma4: " + err};
    }
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

// [vision] ENC runs between calls, each of which syncs its stream: nothing reads the prompt path's buffers until refill.
bool Gemma4Model::lend_room(uint64_t bytes, std::string& err) {
    if (prompt_) {
        prompt_->park();
        parked_ = true;
    }
    size_t free_b = 0, total_b = 0;
    basalt::core::gpu_mem_info(&free_b, &total_b);
    if (free_b >= bytes + card_spare_) return true;
    refill();
    err = "no device memory for the image encoder (" + std::to_string((unsigned long long) (bytes >> 20)) + " MiB beside the card's " +
          std::to_string((unsigned long long) (card_spare_ >> 20)) + " MiB spare; " + std::to_string((unsigned long long) (free_b >> 20)) +
          " MiB free with the prompt path's buffers lent)";
    return false;
}

void Gemma4Model::refill() {
    if (!parked_) return;
    parked_ = false;
    std::string err;
    if (prompt_ && !prompt_->unpark(err)) {
        // no prompt may run without them: the card is out of memory it just had
        basalt::log::error("basalt gemma4", "the prompt path's buffers lent to the image encoder cannot come back to the card: %s",
                           err.c_str());
        std::abort();
    }
}

}  // namespace basalt::models::gemma4
