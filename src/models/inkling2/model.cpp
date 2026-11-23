// src/models/inkling2/model.cpp - loads an Inkling 2 pack: every weight to the main card (with the layer split the later
// layers' to the second), the embedding to mapped host memory, then the context sized from what the weights leave.
#include "models/inkling2/model.hpp"

#include "basalt/core/elastic_kv.hpp"
#include "basalt/core/expert_cache.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/log.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/prefill/q8_gemm.hpp"
#include "basalt/model/multi_gpu.hpp"
#include "models/inkling2/prompt.hpp"
#include "models/inkling2/walk.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <new>

namespace basalt::models::inkling2 {
namespace {
constexpr uint64_t kGiB = 1ull << 30;
constexpr uint64_t kMiB = 1ull << 20;
constexpr int kTypeF32 = 0;
constexpr int kTypeQ8_0 = 8;
constexpr int kTypeBF16 = 30;
constexpr int kTypeNvfp4 = 40;
constexpr int kTypeMxfp8 = 46;
constexpr uint64_t kQ8Row = (uint64_t) kWidth / 32 * 34;   // bytes of a Q8_0 row of the hidden width
// what the sessions share once (the forward's row buffers and expert scratch, the batch windows' logits), what each one
// needs beside its caches (its logits, its windows' graphs) and what the card keeps free for the kernels' modules and the
// driver
constexpr uint64_t kSharedFixed = 768 * kMiB;
constexpr uint64_t kSessionFixed = 96 * kMiB;
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
    const uint64_t block = type == kTypeQ8_0 ? 34 : type == kTypeNvfp4 ? 36 : type == kTypeMxfp8 ? 132 : 1;
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

namespace {
// The cached keys and values of one visit of layer `layer` for `context` positions (a sliding-window layer: its ring).
uint64_t visit_cache_bytes(const Geometry& g, int layer, int64_t context, bool fp8, bool int8) {
    const uint64_t per_pos = 2ull * kKvHeads * kv_cache_row_bytes(fp8, int8);
    return per_pos * (g.swa[(size_t) layer] ? (uint64_t) g.window + kDecodeRows : (uint64_t) context);
}
}  // namespace

uint64_t session_bytes(const Geometry& g, int64_t context, bool fp8, bool int8, int cut, int card) {
    uint64_t caches = 0;
    for (const Visit& v : g.visits)
        if ((cut > 0 && v.layer >= cut ? 1 : 0) == card) caches += visit_cache_bytes(g, v.layer, context, fp8, int8);
    return caches + kSessionFixed + (uint64_t) kDecodeRows * (uint64_t) g.n_vocab * 4;
}

uint64_t shared_bytes(const Geometry& g, bool drafter) {
    return kSharedFixed + prompt_bytes(g) + (drafter ? dspark_tap_bytes(std::max(kDecodeRows, kPromptChunk)) : 0);   // PROMPT, DSPARK HOOKS
}

InklingModel::InklingModel() = default;

InklingModel::~InklingModel() {
    peer_.reset();
    for (const auto& [d, p] : peer_bufs_) {
        cudaSetDevice(d);
        cudaFree(p);
    }
    if (!peer_bufs_.empty()) cudaSetDevice(device_);
    for (void* p : dev_) cudaFree(p);
    for (void* p : copy_)
        if (p) cudaFree(p);
    for (const Range& r : ranges_) {
        if (r.handle) core::vmm::unmap(r.va, r.own, r.handle);
        core::vmm::unmap_only(r.va + r.own, r.bytes - r.own);
        core::vmm::release_range(r.va, r.bytes);
    }
    for (unsigned long long h : staging_) core::vmm::release(h);
    if (spill_host_) cudaFreeHost(spill_host_);
    if (embd_host_) cudaFreeHost(embd_host_);
    if (fd_ >= 0) platform::close_fd(fd_);
}

float* InklingModel::small(const std::string& name, uint64_t floats, core::LoadPlan& reads, int file, std::string& err,
                           float* dst) {
    const TensorRef* t = pack_.find(name);
    if (!t || t->type != kTypeF32 || t->bytes != floats * 4) {
        err = name + (t ? " is not F32 of the expected size" : " is missing");
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
    it.device = ldev_;
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
    it.device = ldev_;
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
    ldev_ = device_;
    fp8_ = req.kv == "fp8";
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
    // two cards (gpu-mode split): the layer split's cut from the shares (model/multi_gpu.hpp)
    if (req.gpus.mode == program::GpuMode::Split)
        if (model::Result r = plan_cards(req); !r.ok()) return r;
    // the loop's message queries of every core layer in one array on each card that runs a core layer: a pass's messages
    // are one launch
    const uint64_t msg_floats = (uint64_t) kStreams * n_embd;
    for (int c = 0; c < cards(); ++c) {
        bool runs = false;
        for (int l = g.loop_start; l < g.loop_end; ++l) runs = runs || card_of(l) == c;
        if (!runs) continue;
        cudaSetDevice(card_device(c));
        if (!(msg_queries_[c] = reinterpret_cast<float*>(device_buf((uint64_t) (g.loop_end - g.loop_start) * msg_floats * 4))))
            return fail("out of device memory loading the loop's message queries");
    }
    for (int l = 0; l < g.n_layers; ++l) {
        LayerWeights& w = layers_[(size_t) l];
        ldev_ = card_device(card_of(l));   // the layer's weights on its card
        cudaSetDevice(ldev_);
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
            it.device = ldev_;
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
            it.device = ldev_;
            reads.add(it);
        }
        const uint64_t q_sc_b = prefill::q8::scale_plane_bytes(kProjQ, (int64_t) n_embd);
        const uint64_t kv_sc_b = prefill::q8::scale_plane_bytes(kProjKV, (int64_t) n_embd);
        const uint64_t o_sc_b = prefill::q8::scale_plane_bytes((int64_t) n_embd, 3072);
        if (!(w.proj_q_sc_t = reinterpret_cast<uint16_t*>(device_buf(q_sc_b))) ||
            !(w.proj_k_sc_t = reinterpret_cast<uint16_t*>(device_buf(kv_sc_b))) ||
            !(w.proj_v_sc_t = reinterpret_cast<uint16_t*>(device_buf(kv_sc_b))) ||
            !(w.out_sc_t = reinterpret_cast<uint16_t*>(device_buf(o_sc_b))))
            return fail("out of device memory loading layer " + std::to_string(l) + " scale planes");
        dense += (uint64_t) kProj * kQ8Row + n_embd * (3072 / 32 * 34) + q_sc_b + 2 * kv_sc_b + o_sc_b;
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
            for (int c = 0; c < cards(); ++c) {   // into the other card's array too, when it runs a core layer
                if (c == card_of(l) || !msg_queries_[c]) continue;
                ldev_ = card_device(c);
                if (!small(b + "loop_msg_query.weight", msg_floats, reads, file, err, msg_queries_[c] + (uint64_t) (l - g.loop_start) * msg_floats))
                    return fail(err);
            }
            ldev_ = card_device(card_of(l));
            if (!(w.msg_query = small(b + "loop_msg_query.weight", msg_floats, reads, file, err,
                                      msg_queries_[card_of(l)] + (uint64_t) (l - g.loop_start) * msg_floats)))
                return fail(err);
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
        // the experts' layout; they load once the context has said how many stay on the card
        const ExpertLayer& el = pack_.experts[(size_t) l];
        w.gu_type = el.gu_type;
        w.d_type = el.d_type;
        w.blob_bytes = el.blob_bytes;
        w.expert_stride = expert_stride(el.gu_type, el.blob_bytes);
    }
    ldev_ = device_;
    cudaSetDevice(device_);
    fp8_ = req.kv == "fp8";
    int8_ = req.kv == "int8";
    // DSPARK HOOK: the drafter's weights (none with --spec 1), before the context is sized from what the weights leave
    if (req.spec >= 2)
        if (model::Result r = dspark_plan(pack_, reads, file, device_, dspark_); !r.ok()) return r;

    // [vision] the image encoder staged before the context is sized; it keeps none of the card: a request's pictures borrow
    // their region from the experts (lend_room) and give it back (refill). What staging leaves on the card (the encoder's
    // kernels, stream and product handle) comes out of the card's spare, which is there for the kernels' modules.
    uint64_t vision_modules = 0;
    if (req.vision) {
        size_t before = 0, after = 0, total_b = 0;
        cudaMemGetInfo(&before, &total_b);
        if (!vis_.stage(pack_.vision, g.n_embd, err)) return fail("vision: " + err);
        cudaMemGetInfo(&after, &total_b);
        vision_modules = std::min<uint64_t>(before > after ? before - after : 0, kCardSpare / 4);
        vis_.set_lender({[this](uint64_t bytes, std::string& e) { return lend_room(bytes, e); }, [this] { refill(); }});
        basalt::log::info("basalt inkling2", "image encoder: %.2f GiB staged in page-locked RAM, 0 GiB of the card kept for it (its "
                          "kernels' %.0f MiB inside the card's spare; a request's pictures borrow up to %.2f GiB of expert memory "
                          "and give it back)",
                          (double) vis_.weights_bytes() / (double) kGiB, (double) vision_modules / (double) kMiB,
                          (double) vis_.region_bytes() / (double) kGiB);
    }

    // the context: what the weights leave, less the card's spare
    {
        const int64_t want = req.max_context > 0 ? req.max_context : kDefaultContext;
        if (want < 1024 || (g.context_length > 0 && want > g.context_length))
            return fail("--max-context " + std::to_string(want) + " is outside 1,024 .. " + std::to_string(g.context_length));
        const uint64_t spare = kCardSpare - vision_modules;   // [vision]
        const uint64_t n_sess = (uint64_t) std::max(req.sessions, 1);
        const auto bytes_at = [&](int64_t c) {   // DSPARK HOOK: a session's drafter beside its caches
            return session_bytes(g, c, fp8_, int8_) + (dspark_ ? dspark_session_bytes(pack_, c) : 0);
        };
        const uint64_t shared = shared_bytes(g, dspark_ != nullptr);
        if (cut_ > 0) {   // the layer split: each card holds its layers' experts (no spill), checked once they are placed
            if (req.expert_cache > 0) return fail("--expert-cache takes one card; the layer split keeps every expert on its card");
            if (model::Result r = load_experts(reads, file, experts); !r.ok()) return r;
            for (int c = 0; c < 2; ++c) {
                cudaSetDevice(card_device(c));
                size_t fb = 0, tb = 0;
                cudaMemGetInfo(&fb, &tb);
                const uint64_t need = (session_bytes(g, want, fp8_, int8_, cut_, c) +
                                       (c == 0 && dspark_ ? dspark_session_bytes(pack_, want) : 0)) * n_sess +
                                      (c == 0 ? shared + spare : shared_bytes(g, false) + kCardSpare);
                if (need > fb) {
                    cudaSetDevice(device_);
                    return fail("the layer split at layer " + std::to_string(cut_) + ": card " + std::to_string(card_device(c)) +
                                " needs " + std::to_string(need >> 20) + " MiB beside its weights for --max-context " + std::to_string(want) +
                                ", " + std::to_string(fb >> 20) + " MiB are free (a smaller --max-context, --kv fp8, or one card)");
                }
            }
            cudaSetDevice(device_);
            ctx_ = want;
        } else {
        // the experts: all on the card when they fit beside the asked context; else the least routed go to RAM, just enough
        // (spill.hpp). --expert-cache N keeps at most N on the card (a diagnostic).
        {
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            std::vector<uint64_t> stride;
            for (const LayerWeights& w : layers_) stride.push_back(w.expert_stride);
            uint64_t full = 0;
            for (uint64_t st : stride) full += st * (uint64_t) g.n_expert;
            const uint64_t want_b = full + bytes_at(want) * n_sess + shared + spare;
            // when some must go: a margin for the tables and maps allocated after the plan and the driver's own rounding
            const uint64_t short_b = want_b > free_b ? want_b - free_b + 128 * kMiB : 0;
            const int64_t keep = req.expert_cache > 0 ? req.expert_cache : -1;
            if (short_b > 0 || keep >= 0) {
                if (!core::vmm::supported()) return fail("the experts do not fit beside the context and this card has no virtual memory management");
                std::vector<std::pair<int32_t, int32_t>> rank;
                if (!req.expert_profile.empty()) {
                    uint64_t pb = 0;
                    int64_t slots = 0;
                    if (!platform::file_size(req.expert_profile, pb) ||
                        !core::read_expert_profile(core::FileRange{req.expert_profile, 0, pb}, g.n_layers, g.n_expert, rank, slots, err))
                        return fail("expert profile " + req.expert_profile + ": " + err);
                }
                if (!plan_spill(g.n_layers, g.n_expert, stride, core::vmm::granularity(), rank, short_b, keep, spill_, err))
                    return fail(err + " (--max-context " + std::to_string(want) + ")");
                // the RAM experts, the most routed first (the profile's order; without one, layer by layer)
                for (const auto& [l, e] : rank)
                    if (!spill_.slot[(size_t) l].empty() && spill_.slot[(size_t) l][(size_t) e] >= g.n_expert - spill_.n_spill[(size_t) l])
                        spill_order_.emplace_back(l, e);
                if (rank.empty())
                    for (int l = 0; l < g.n_layers; ++l)
                        for (int e = 0; e < g.n_expert; ++e)
                            if (!on_card(l, e)) spill_order_.emplace_back(l, e);
                elastic_ = keep < 0 && want > core::kElasticKvBase;
                uint64_t ram = 0;
                for (int l = 0; l < g.n_layers; ++l) ram += (uint64_t) spill_.n_spill[(size_t) l] * stride[(size_t) l];
                basalt::log::info("basalt inkling2",
                                  "experts: %lld of %lld in RAM (%.2f GiB, the least routed%s), %.2f GiB on the card with %.0f MiB of "
                                  "prompt staging, so --max-context %lld fits",
                                  (long long) spill_.spilled, (long long) g.n_layers * g.n_expert, (double) ram / (double) kGiB,
                                  rank.empty() ? ", no profile" : " by the profile", (double) spill_.device / (double) kGiB,
                                  (double) spill_.staging / (double) kMiB, (long long) want);
            }
            if (req.gpus.mode == program::GpuMode::Peer && req.gpus.cards.size() > 1) {   // the second card takes the hottest of what spilled
                if (!peer_open(req.gpus, err)) return {model::Status::Error, err};
                int64_t held = 0;
                for (int n : peer_n_) held += n;
                basalt::log::info("basalt inkling2", "gpu-mode peer: %lld of the %lld experts in RAM are on CUDA%d (%.2f GiB), %lld stay on the CPU",
                                  (long long) held, (long long) spill_.spilled, peer_->device(), (double) peer_bytes_ / (double) kGiB,
                                  (long long) (spill_.spilled - held));
            }
            if (model::Result r = load_experts(reads, file, experts); !r.ok()) return r;
        }
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const uint64_t avail = free_b > spare + shared ? (free_b - spare - shared) / n_sess : 0;
        const uint64_t need = bytes_at(want);
        if (need > avail) {
            const uint64_t per = bytes_at(2) - bytes_at(1);
            const uint64_t base = bytes_at(0);
            const int64_t fits = avail > base ? (int64_t) ((avail - base) / per) : 0;
            return {model::Status::Error, "inkling2: --max-context " + std::to_string(want) + " needs " +
                                              std::to_string(need / kMiB) + " MiB a session; about " + std::to_string(fits) +
                                              " positions fit" + (fp8_ || int8_ ? "" : " (--kv fp8 halves a position's bytes)")};
        }
        ctx_ = want;
        }
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

model::Result InklingModel::load_experts(core::LoadPlan& reads, int file, uint64_t& experts) {
    const Geometry& g = pack_.geo;
    auto fail = [&](const std::string& e) { return model::Result{model::Status::Error, "inkling2: " + e}; };
    auto device_buf = [&](uint64_t bytes) -> uint8_t* {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        dev_.push_back(p);
        return static_cast<uint8_t*>(p);
    };
    const uint64_t gran = spill_.any() ? core::vmm::granularity() : 0;
    if (spill_.any()) {
        copy_.assign((size_t) g.n_layers * g.n_expert, nullptr);
        layer_copies_.assign((size_t) g.n_layers, 0);
        layer_ram_copies_.assign((size_t) g.n_layers, 0);
    }
    // the RAM copies, every spilled layer's in slot order, and the staging their ranges share
    std::vector<uint64_t> host_at((size_t) g.n_layers, 0);
    if (spill_.any()) {
        uint64_t host = 0;
        for (int l = 0; l < g.n_layers; ++l) {
            host_at[(size_t) l] = host;
            host += (uint64_t) spill_.n_spill[(size_t) l] * layers_[(size_t) l].expert_stride;
        }
        if (cudaHostAlloc((void**) &spill_host_, host, cudaHostAllocPortable) != cudaSuccess)
            return fail("cannot pin " + std::to_string(host >> 20) + " MiB of RAM for the experts that leave the card");
        std::memset(spill_host_, 0, host);
        for (uint64_t at = 0; at < spill_.staging; at += gran) {
            const unsigned long long h = core::vmm::create(gran);
            if (!h) return fail("out of device memory for the experts' prompt staging");
            staging_.push_back(h);
        }
    }
    for (int l = 0; l < g.n_layers; ++l) {
        LayerWeights& w = layers_[(size_t) l];
        const ExpertLayer& el = pack_.experts[(size_t) l];
        const uint64_t eb = (uint64_t) g.n_expert * w.expert_stride;
        std::vector<unsigned long long> table((size_t) g.n_expert);
        const std::vector<int32_t>& slot = spill_.slot.empty() ? std::vector<int32_t>() : spill_.slot[(size_t) l];
        const int dev = card_device(card_of(l));   // the layer's card (a spill runs on one card only)
        cudaSetDevice(dev);
        if (slot.empty()) {
            // the prompt path reads the last expert's down rows (n_ff 1152 columns) on to a multiple of 256 values: zeros there
            const uint64_t past = std::max<uint64_t>(
                16, std::max(prefill::mmq::read_past_bytes(el.gu_type, g.n_embd),
                             prefill::mmq::read_past_bytes(el.d_type, g.n_ff_exp)));
            if (core::vmm::supported() && card_of(l) == 0) {   // a range of its own, so that a picture can borrow its memory (lend_room)
                const uint64_t vg = core::vmm::granularity();
                Range r;
                r.bytes = r.own = (eb + past + vg - 1) / vg * vg;
                if (!(r.va = core::vmm::reserve(r.bytes))) return fail("cannot reserve the address range of layer " + std::to_string(l) + "'s experts");
                if (!(r.handle = core::vmm::map(r.va, r.own))) {
                    core::vmm::release_range(r.va, r.bytes);
                    return fail("out of device memory loading layer " + std::to_string(l) + "'s experts");
                }
                ranges_.push_back(r);
                w.experts = r.va;
            } else if (!(w.experts = device_buf(eb + past))) {
                return fail("out of device memory loading layer " + std::to_string(l) + "'s experts");
            }
            if (past) cudaMemset(w.experts + eb, 0, past);
            if (w.expert_stride == el.blob_bytes) {
                core::LoadItem it;
                it.file = file;
                it.off = pack_.experts_range.offset + el.offset;
                it.bytes = eb;
                it.dev = w.experts;
                it.device = dev;
                reads.add(it);
            } else {
                cudaMemset(w.experts, 0, eb);
                for (int e = 0; e < g.n_expert; ++e) {
                    core::LoadItem it;
                    it.file = file;
                    it.off = pack_.experts_range.offset + el.offset + (uint64_t) e * el.blob_bytes;
                    it.bytes = el.blob_bytes;
                    it.dev = w.experts + (uint64_t) e * w.expert_stride;
                    it.device = dev;
                    reads.add(it);
                }
            }
            experts += eb;
            // the experts' addresses, for the plan
            for (int e = 0; e < g.n_expert; ++e) table[(size_t) e] = (unsigned long long) (w.experts + (uint64_t) e * w.expert_stride);
        } else {
            // the kept experts in their own backing, the RAM ones' slots over the shared staging; their blobs in pinned RAM
            const int keep = g.n_expert - spill_.n_spill[(size_t) l];
            Range r;
            r.bytes = (eb + kSlotGuard + gran - 1) / gran * gran;
            r.own = ((uint64_t) keep * w.expert_stride + gran - 1) / gran * gran;
            if (!(r.va = core::vmm::reserve(r.bytes))) return fail("cannot reserve the address range of layer " + std::to_string(l) + "'s experts");
            if (r.own && !(r.handle = core::vmm::map(r.va, r.own))) {
                core::vmm::release_range(r.va, r.bytes);
                return fail("out of device memory loading layer " + std::to_string(l) + "'s experts");
            }
            for (uint64_t at = r.own; at < r.bytes; at += gran) {
                if (core::vmm::map_handle(r.va + at, gran, staging_[(size_t) ((at - r.own) / gran)])) continue;
                core::vmm::unmap_only(r.va + r.own, at - r.own);
                if (r.handle) core::vmm::unmap(r.va, r.own, r.handle);
                core::vmm::release_range(r.va, r.bytes);
                return fail("cannot map the prompt staging under layer " + std::to_string(l) + "'s experts");
            }
            ranges_.push_back(r);
            w.experts = r.va;
            w.n_spill = spill_.n_spill[(size_t) l];
            uint8_t* host = spill_host_ + host_at[(size_t) l];
            w.spill_host = host;
            if (w.expert_stride != el.blob_bytes && r.own) cudaMemset(w.experts, 0, r.own);
            for (int e = 0; e < g.n_expert; ++e) {
                const int s = slot[(size_t) e];
                core::LoadItem it;
                it.file = file;
                it.off = pack_.experts_range.offset + el.offset + (uint64_t) e * el.blob_bytes;
                it.bytes = el.blob_bytes;
                if (s < keep) {
                    it.dev = w.experts + (uint64_t) s * w.expert_stride;
                    it.device = device_;
                    table[(size_t) e] = (unsigned long long) it.dev;
                } else if (peer_held(l, e)) {   // the second card's slot; the CPU's entry is 0 as for any expert off the card
                    it.dev = peer_->slot_ptr(peer_slot_[(size_t) l * g.n_expert + e]);
                    it.device = peer_->device();
                    table[(size_t) e] = 0;
                } else {
                    it.host = host + (uint64_t) (s - keep) * w.expert_stride;
                    table[(size_t) e] = 0;   // the CPU computes it (cpu_experts.hpp)
                }
                reads.add(it);
            }
            experts += (uint64_t) keep * w.expert_stride;
            std::vector<int32_t> by_slot = slot;   // a prompt chunk leaves the second card's experts out (-1)
            for (int e = 0; e < g.n_expert; ++e)
                if (peer_held(l, e)) by_slot[(size_t) e] = -1;
            int32_t* map = reinterpret_cast<int32_t*>(device_buf((uint64_t) g.n_expert * 4));
            if (!map || cudaMemcpy(map, by_slot.data(), (size_t) g.n_expert * 4, cudaMemcpyHostToDevice) != cudaSuccess)
                return fail("out of device memory loading layer " + std::to_string(l) + "'s expert slots");
            w.slot_map = map;
            int32_t* full = reinterpret_cast<int32_t*>(device_buf((uint64_t) g.n_expert * 4));
            if (!full || cudaMemcpy(full, slot.data(), (size_t) g.n_expert * 4, cudaMemcpyHostToDevice) != cudaSuccess)
                return fail("out of device memory loading layer " + std::to_string(l) + "'s expert slots");
            w.slot_full = full;
        }
        if (!(w.blob_table = reinterpret_cast<unsigned long long*>(device_buf(table.size() * 8))) ||
            cudaMemcpy(w.blob_table, table.data(), table.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("out of device memory loading layer " + std::to_string(l) + "'s expert table");
    }
    cudaSetDevice(device_);
    return {};
}

model::Result InklingModel::plan_cards(const model::LoadRequest& req) {
    req_ = &req;
    std::string err;
    const bool ok = split_open(req.gpus, err);
    req_ = nullptr;
    if (!ok) return {model::Status::Error, err};
    return {};
}

model::SplitSpec InklingModel::split_spec(const model::GpuCards& c) const {
    const model::LoadRequest& req = *req_;
    const Geometry& g = pack_.geo;
    // the bytes the share divides: per layer its experts (a split card holds every expert of its layers), its dense rows and
    // every slot's caches of its visits at the context; per card what it holds whatever its layers (the head is on the main
    // card already, so its room is what is free now)
    const int64_t want = req.max_context > 0 ? req.max_context : kDefaultContext;
    const uint64_t n_sess = (uint64_t) std::max(req.sessions, 1);
    const bool fp8 = req.kv == "fp8", int8 = req.kv == "int8";
    std::vector<int> visits((size_t) g.n_layers, 0);
    for (const Visit& v : g.visits) ++visits[(size_t) v.layer];
    model::SplitSpec s;
    for (int l = 0; l < g.n_layers; ++l) {
        const ExpertLayer& el = pack_.experts[(size_t) l];
        const uint64_t dense = (uint64_t) kProj * kQ8Row + (uint64_t) g.n_embd * (3072 / 32 * 34) +
                               prefill::q8::scale_plane_bytes(kProjQ, (int64_t) g.n_embd) +
                               2 * prefill::q8::scale_plane_bytes(kProjKV, (int64_t) g.n_embd) +
                               prefill::q8::scale_plane_bytes((int64_t) g.n_embd, 3072);
        s.layer.push_back((uint64_t) g.n_expert * expert_stride(el.gu_type, el.blob_bytes) + dense +
                          (uint64_t) visits[(size_t) l] * visit_cache_bytes(g, l, want, fp8, int8) * n_sess);
    }
    const uint64_t slot_rest = session_bytes(g, want, fp8, int8, g.n_layers, 1);   // a slot's own beyond its caches (no layer on card 1)
    s.fixed = {kCardSpare + shared_bytes(g, req.spec >= 2) + slot_rest * n_sess + (req.spec >= 2 ? 1000ull << 20 : 0),
               kCardSpare + shared_bytes(g, false) + slot_rest * n_sess};
    for (const model::GpuCards::Card& card : c.cards) {
        cudaSetDevice(card.device);
        size_t free_b = 0, total_b = 0;
        s.room.push_back(cudaMemGetInfo(&free_b, &total_b) == cudaSuccess ? free_b : 0);
    }
    cudaSetDevice(device_);
    return s;
}

bool InklingModel::split_place(const model::SplitCut& cut, const std::vector<int>& devices, std::string&) {
    const Geometry& g = pack_.geo;
    cut_ = cut.first[1];
    second_ = devices[1];
    split_need_ = cut.need;
    basalt::log::info("basalt inkling2", "layer split: layers 0..%d on CUDA%d, %d..%d on CUDA%d, %d hand-offs a window", cut_ - 1,
                      device_, cut_, g.n_layers - 1, second_, split_handoffs(g, cut_));
    return true;
}

std::string InklingModel::split_part(size_t card) const {
    return "with their experts and caches (" + model::gib(split_need_[card]) + ")" + (card == 0 ? ", head" : "");
}

std::string InklingModel::gpu_report(int card) const {
    if (card < 0) return {};
    if (cut_ > 0) return split_report((size_t) card);
    return peer_ ? peer_report((size_t) card) : std::string();
}

// ---- gpu-mode peer: the second card holds the hottest of the experts that spilled from the main card, in its VRAM

bool InklingModel::peer_card(size_t, int device, std::string& err) {
    const Geometry& g = pack_.geo;
    std::string e;
    auto p = std::make_unique<core::PeerExperts>();
    if (!p->open_plain(device, g.n_layers, g.n_expert, g.n_embd, g.n_ff_exp, nullptr, e)) {
        err = "gpu-mode peer: Inkling-2-Mini cannot open the second card: " + e;
        return false;
    }
    peer_ = std::move(p);
    return true;
}

namespace {
// What the second card keeps beyond the experts: the prompt path's buffers (a lane like the main card's), its kernels'
// modules and the driver's own rounding.
constexpr uint64_t kPeerSpare = 1ull << 30;
}  // namespace

uint64_t InklingModel::peer_room(size_t) const { return peer_->room(); }

model::PeerSpec InklingModel::peer_spec() const {
    model::PeerSpec s;
    s.deal = model::PeerDeal::Experts;
    for (const auto& [l, e] : spill_order_) s.unit.push_back((layers_[(size_t) l].expert_stride + 255) / 256 * 256);
    s.reserve = prompt_bytes(pack_.geo) + kPeerSpare;
    return s;
}

bool InklingModel::peer_place(const model::PeerPlacement& p, std::string& err) {
    const Geometry& g = pack_.geo;
    const int ne = g.n_expert;
    std::vector<uint8_t> mine((size_t) g.n_layers * ne, 0);
    peer_n_.assign((size_t) g.n_layers, 0);
    for (size_t i = 0; i < spill_order_.size(); ++i)
        if (p.owner[i] >= 1) {
            mine[(size_t) spill_order_[i].first * ne + spill_order_[i].second] = 1;
            ++peer_n_[(size_t) spill_order_[i].first];
        }
    // the slots: layer by layer, the layer's experts in expert order, then a zeroed guard (the products read past the last
    // expert's down rows, as kSlotGuard says)
    peer_slot_.assign((size_t) g.n_layers * ne, -1);
    peer_first_.assign((size_t) g.n_layers, 0);
    std::vector<int64_t> sizes;
    for (int l = 0; l < g.n_layers; ++l) {
        peer_first_[(size_t) l] = (int32_t) sizes.size();
        for (int e = 0; e < ne; ++e)
            if (mine[(size_t) l * ne + e]) {
                peer_slot_[(size_t) l * ne + e] = (int32_t) sizes.size();
                sizes.push_back((int64_t) peer_stride(l));
            }
    }
    sizes.push_back((int64_t) kSlotGuard);
    if (!peer_->open_slots(sizes, err)) {
        err = "gpu-mode peer: " + err;
        return false;
    }
    uint64_t bytes = 0;
    for (int64_t b : sizes) bytes += (uint64_t) (b + 255) / 256 * 256;
    peer_bytes_ = bytes - (uint64_t) ((int64_t) kSlotGuard);
    cudaSetDevice(peer_->device());
    cudaMemset(peer_->slot_ptr(0), 0, bytes);   // the padding after each blob and the guard
    peer_map_.assign((size_t) g.n_layers, nullptr);
    for (int l = 0; l < g.n_layers; ++l) {
        if (peer_n_[(size_t) l] == 0) continue;
        std::vector<int32_t> map((size_t) ne, -1);
        int j = 0;
        for (int e = 0; e < ne; ++e)
            if (mine[(size_t) l * ne + e]) {
                map[(size_t) e] = j++;
                peer_->admit(l, e, peer_slot_[(size_t) l * ne + e]);
            }
        void* d = nullptr;
        if (cudaMalloc(&d, (size_t) ne * 4) != cudaSuccess || cudaMemcpy(d, map.data(), (size_t) ne * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaSetDevice(device_);
            err = "gpu-mode peer: out of device memory for the expert maps";
            return false;
        }
        peer_bufs_.emplace_back(peer_->device(), d);
        peer_map_[(size_t) l] = static_cast<int32_t*>(d);
    }
    cudaSetDevice(device_);
    // the main card's slots: a layer's RAM experts first, the second card's last (a chunk stages the first run only)
    for (int l = 0; l < g.n_layers; ++l) {
        if (peer_n_[(size_t) l] == 0) continue;
        std::vector<int32_t>& slot = spill_.slot[(size_t) l];
        const int keep = ne - spill_.n_spill[(size_t) l];
        int ram = keep, held = ne - peer_n_[(size_t) l];
        for (int e = 0; e < ne; ++e)
            if (slot[(size_t) e] >= keep) slot[(size_t) e] = mine[(size_t) l * ne + e] ? held++ : ram++;
    }
    return true;
}

std::string InklingModel::peer_main() const {
    const int64_t all = (int64_t) pack_.geo.n_layers * pack_.geo.n_expert;
    return "dense, KV, " + std::to_string(all - spill_.spilled) + " of " + std::to_string(all) + " experts (" +
           model::gib(spill_.device) + ")";
}

model::Result InklingModel::attach(const model::LoadRequest&, std::vector<std::unique_ptr<model::ModelSession>>& sessions) {
    {   // Convert loaded Q8_0 w.proj and w.out blocks in-place to wgemm::q8_planes + coalesced prefill scale planes.
        const int64_t E = pack_.geo.n_embd;
        const size_t proj_b = (size_t) kProj * (size_t) (E / 32 * 34);
        const size_t out_b = (size_t) E * (size_t) (3072 / 32 * 34);
        const int64_t nb_e = E / 32;
        for (int c = 0; c < cards(); ++c) {
            cudaSetDevice(card_device(c));
            void* tmp = nullptr;
            if (cudaMalloc(&tmp, std::max(proj_b, out_b)) != cudaSuccess) {
                cudaSetDevice(device_);
                return {model::Status::Error, "inkling2: out of device memory converting Q8_0 projections to planes"};
            }
            for (int l = 0; l < pack_.geo.n_layers; ++l) {
                if (card_of(l) != c) continue;
                LayerWeights& w = layers_[(size_t) l];
                kernels::wgemm::q8_planes(w.proj, kProj, E, tmp, nullptr);
                cudaMemcpy(w.proj, tmp, proj_b, cudaMemcpyDeviceToDevice);
                kernels::wgemm::q8_planes(w.out, E, 3072, tmp, nullptr);
                cudaMemcpy(w.out, tmp, out_b, cudaMemcpyDeviceToDevice);
                const uint16_t* proj_sc = reinterpret_cast<const uint16_t*>(w.proj + (size_t) kProj * (size_t) E);
                prefill::q8::transpose_scales(proj_sc, kProjQ, E, w.proj_q_sc_t, nullptr);
                prefill::q8::transpose_scales(proj_sc + (size_t) kRowK * nb_e, kProjKV, E, w.proj_k_sc_t, nullptr);
                prefill::q8::transpose_scales(proj_sc + (size_t) kRowV * nb_e, kProjKV, E, w.proj_v_sc_t, nullptr);
                const uint16_t* out_sc = reinterpret_cast<const uint16_t*>(w.out + (size_t) E * 3072);
                prefill::q8::transpose_scales(out_sc, E, 3072, w.out_sc_t, nullptr);
            }
            cudaDeviceSynchronize();
            cudaFree(tmp);
        }
        cudaSetDevice(device_);
    }
    if (dspark_)   // DSPARK HOOK: the drafter's head folds, once the reads have landed
        if (model::Result r = dspark_->finish(head_, embd_dev_, embed_norm_, nullptr); !r.ok()) return r;
    for (auto& s : sessions)
        if (model::Result r = s->prepare(); !r.ok()) return r;
    if (elastic_) {   // the caches back only their base yet: the RAM experts come back while the card has room
        const uint64_t moved = promote(held_free(kCardSpare));
        basalt::log::info("basalt inkling2", "elastic spill: %lld of %lld RAM experts copied back to the card (%.2f GiB); they go back "
                          "to RAM, the least routed first, as a conversation's caches grow",
                          (long long) n_copy_, (long long) spill_.spilled, (double) moved / (double) kGiB);
    }
    return {};
}

uint64_t InklingModel::promote(uint64_t keep_free) {
    const int ne = geo().n_expert;
    uint64_t moved = 0;
    for (const auto& [l, e] : spill_order_) {
        const size_t i = (size_t) l * ne + e;
        if (copy_[i]) continue;
        const LayerWeights& w = layers_[(size_t) l];
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        if (free_b < keep_free + w.expert_stride) break;
        void* p = nullptr;
        if (cudaMalloc(&p, w.expert_stride) != cudaSuccess) {
            cudaGetLastError();
            break;
        }
        const int keep = ne - w.n_spill;
        const bool held = peer_held(l, e);   // the second card's copy is the source, there is none in RAM
        const uint8_t* src = held ? peer_->slot_ptr(peer_slot_[i])
                                  : w.spill_host + (uint64_t) (spill_.slot[(size_t) l][(size_t) e] - keep) * w.expert_stride;
        const unsigned long long at = (unsigned long long) p;
        if (cudaMemcpy(p, src, w.expert_stride, held ? cudaMemcpyDefault : cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(w.blob_table + e, &at, 8, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(p);
            break;
        }
        copy_[i] = p;
        ++n_copy_;
        ++layer_copies_[(size_t) l];
        layer_ram_copies_[(size_t) l] += !held;
        moved += w.expert_stride;
    }
    return moved;
}

bool InklingModel::resize_reserve(uint64_t free_bytes, void* stream) {
    reserve_ = free_bytes;
    if (lent_) return true;   // the wake takes the copies back against this reserve
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const uint64_t want = held_free(kCardSpare);
    if (free_b < want) demote(want - free_b, stream);
    else promote(want);
    cudaMemGetInfo(&free_b, &total_b);
    return free_b >= want;
}

void InklingModel::lend(bool low, void* stream) {
    if (low) {
        demote(~0ull, stream);
        lent_ = true;
    } else {
        lent_ = false;
        promote(held_free(kCardSpare));
    }
}

const uint8_t* InklingModel::ram_blob(int l, int e) const {
    const LayerWeights& w = layers_[(size_t) l];
    if (w.n_spill == 0 || copy_[(size_t) l * geo().n_expert + e] || peer_held(l, e)) return nullptr;
    const int slot = spill_.slot[(size_t) l][(size_t) e] - (geo().n_expert - w.n_spill);
    return slot < 0 ? nullptr : w.spill_host + (uint64_t) slot * w.expert_stride;
}

uint64_t InklingModel::cpu_layers() const {
    uint64_t mask = 0;
    for (size_t l = 0; l < layers_.size(); ++l)
        if (layers_[l].n_spill > (layer_copies_.empty() ? 0 : layer_copies_[l])) mask |= 1ull << l;
    return mask;
}

bool InklingModel::stage_from_card(int l, void* stream, bool with_peer) const {
    const LayerWeights& w = layers_[(size_t) l];
    if (copy_.empty() || w.n_spill == 0 || (with_peer ? layer_copies_[(size_t) l] < w.n_spill : layer_ram_copies_[(size_t) l] < ram_n(l)))
        return false;
    const int ne = geo().n_expert;
    for (int e = 0; e < ne; ++e)
        if (const void* c = !with_peer && peer_held(l, e) ? nullptr : copy_[(size_t) l * ne + e])
            cudaMemcpyAsync(w.experts + (uint64_t) spill_.slot[(size_t) l][(size_t) e] * w.expert_stride, c, w.expert_stride,
                            cudaMemcpyDeviceToDevice, (cudaStream_t) stream);
    return true;
}

uint64_t InklingModel::demote(uint64_t bytes, void* stream) {
    const int ne = geo().n_expert;
    std::vector<size_t> out;
    uint64_t freed = 0;
    for (auto it = spill_order_.rbegin(); it != spill_order_.rend() && freed < bytes; ++it) {
        const size_t i = (size_t) it->first * ne + it->second;
        if (!copy_[i]) continue;
        cudaMemsetAsync(layers_[(size_t) it->first].blob_table + it->second, 0, 8, (cudaStream_t) stream);
        out.push_back(i);
        freed += layers_[(size_t) it->first].expert_stride;
    }
    if (out.empty()) return 0;
    cudaStreamSynchronize((cudaStream_t) stream);
    for (size_t i : out) {
        cudaFree(copy_[i]);
        copy_[i] = nullptr;
        --n_copy_;
        --layer_copies_[i / (size_t) ne];
        layer_ram_copies_[i / (size_t) ne] -= !peer_held((int) (i / (size_t) ne), (int) (i % (size_t) ne));
    }
    return freed;
}

// [vision] ENC runs between requests and windows: once every stream is idle, nothing reads the experts until refill.
bool InklingModel::lend_room(uint64_t bytes, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = "the card failed before the image encoder";
        return false;
    }
    // the region beside the card's spare and the reserve the card holds now (a reserve it could not meet stays unmet)
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const uint64_t want = bytes + std::min<uint64_t>(held_free(kCardSpare), std::max<uint64_t>(free_b, kCardSpare));
    if (free_b < want && n_copy_ > 0) {
        const int64_t before = n_copy_;
        demote(want - free_b, nullptr);
        lend_copies_ += before - n_copy_;
        cudaMemGetInfo(&free_b, &total_b);
    }
    for (size_t i = ranges_.size(); i-- > 0 && free_b < want;) {
        Range& r = ranges_[i];
        if (!r.handle) continue;
        std::unique_ptr<uint8_t[]> keep(new (std::nothrow) uint8_t[r.own]);
        if (!keep || cudaMemcpy(keep.get(), r.va, r.own, cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "no RAM for the experts the image encoder borrows from";
            break;
        }
        if (!core::vmm::unmap(r.va, r.own, r.handle)) {
            err = "cannot lend expert memory to the image encoder";
            break;
        }
        r.handle = 0;
        lent_ranges_.push_back({i, std::move(keep)});
        cudaMemGetInfo(&free_b, &total_b);
    }
    lend_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (free_b >= want) return true;
    refill();
    if (err.empty())
        err = "no device memory for the image encoder (" + std::to_string((unsigned long long) (bytes >> 20)) +
              " MiB) even with the experts' memory lent";
    return false;
}

void InklingModel::refill() {
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t back = 0;
    for (auto it = lent_ranges_.rbegin(); it != lent_ranges_.rend(); ++it) {
        Range& r = ranges_[it->range];
        if (!(r.handle = core::vmm::map(r.va, r.own)) ||
            cudaMemcpy(r.va, it->bytes.get(), r.own, cudaMemcpyHostToDevice) != cudaSuccess) {
            // nothing may run on these experts now: the card is out of memory it just had
            basalt::log::error("basalt inkling2", "the experts lent to the image encoder cannot come back to the card");
            std::abort();
        }
        back += r.own;
    }
    const size_t layers = lent_ranges_.size();
    lent_ranges_.clear();
    const int64_t copies = n_copy_;
    if (elastic_ && !lent_) promote(held_free(kCardSpare));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (layers > 0 || lend_copies_ > 0)
        basalt::log::info("basalt inkling2", "image encoder: borrowed %lld expert copies and %zu layers' experts (%.2f GiB) in %.0f ms, "
                          "back in %.0f ms (%lld copies)",
                          (long long) lend_copies_, layers, (double) back / (double) kGiB, lend_ms_, ms, (long long) (n_copy_ - copies));
    lend_ms_ = 0;
    lend_copies_ = 0;
}

}  // namespace basalt::models::inkling2
