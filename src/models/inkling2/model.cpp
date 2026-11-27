// src/models/inkling2/model.cpp - loads an Inkling 2 pack: every weight to the main card (with the layer split the later
// layers' to the second), the embedding to mapped host memory, then the context sized from what the weights leave.
#include "basalt/core/gpu_mem_info.hpp"
#include "basalt/core/device_copy.hpp"
#include "models/inkling2/caches.hpp"
#include "models/inkling2/model.hpp"

#include "basalt/core/elastic_kv.hpp"
#include "basalt/core/expert_cache.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/log.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/prefill/q8_gemm.hpp"
#include "basalt/model/batch_driver.hpp"
#include "basalt/model/multi_gpu.hpp"
#include "models/inkling2/loan_layout.hpp"
#include "models/inkling2/prompt.hpp"
#include "models/inkling2/walk.hpp"
#include "models/inkling2/wide_chunk.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

namespace basalt::models::inkling2 {
namespace {
constexpr uint64_t kGiB = 1ull << 30;
constexpr uint64_t kMiB = 1ull << 20;
// The loan's uploads queued after one decode window: well under a window's time on the bus, so they overlap decode.
constexpr uint64_t kLoanBytesPerWindow = 192 * kMiB;
constexpr int kTypeF32 = 0;
constexpr int kTypeQ8_0 = 8;
constexpr int kTypeBF16 = 30;
constexpr uint64_t kQ8Row = (uint64_t) kWidth / 32 * 34;   // bytes of a Q8_0 row of the hidden width
// what the sessions share once (the forward's row buffers and expert scratch, the batch windows' logits), what each one
// needs beside its caches (its logits, its windows' graphs) and what the card keeps free for the kernels' modules and the
// driver
constexpr uint64_t kSharedFixed = 896 * kMiB;
constexpr uint64_t kSessionMargin = 12 * kMiB;   // a session's small buffers and the allocator's rounding
constexpr uint64_t kCardSpare = 1024 * kMiB;
// a batch window's graph on the main card (its kernel nodes): a batch of slots keeps up to BatchShared::kGraphs of them,
// instantiated as compositions come, so the plan holds them beside the KV pool
constexpr uint64_t kBatchGraphBytes = 3 * kMiB;
uint64_t batch_graphs_bytes(int n_sess) { return n_sess > 1 ? kBatchGraphBytes * model::BatchShared::kGraphs : 0; }
// with --vram-reserve-mib: what loads after the copies are placed (kernels at their first launch, the windows' graphs),
// kept free beside the reserve until that much has loaded
constexpr uint64_t kLateRoom = 256 * kMiB;

// A log line that cannot be written (out of host memory) is left out: the placement step it reports stands.
template <class F>
void quiet(F&& f) noexcept {
    try {
        f();
    } catch (...) {
    }
}

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
}  // namespace

namespace {
// The cached keys and values of one visit of layer `layer` for `context` positions (a sliding-window layer: its ring).
uint64_t visit_cache_bytes(const Geometry& g, int layer, int64_t context, bool fp8, bool int8) {
    const uint64_t per_pos = 2ull * kKvHeads * kv_cache_row_bytes(fp8, int8);
    return per_pos * (uint64_t) visit_cells(g.swa[(size_t) layer], (int64_t) g.window, context);
}
}  // namespace

uint64_t session_bytes(const Geometry& g, int64_t context, bool fp8, bool int8, int cut, int card) {
    // a visit's window caches, its four convolution rings, and the window's routed ids the copies' ranking keeps
    uint64_t own = kSessionMargin + 3ull * (uint64_t) g.n_layers * (uint64_t) g.n_expert * 4;
    for (const Visit& v : g.visits) {
        if ((cut > 0 && v.layer >= cut ? 1 : 0) != card) continue;
        if (g.swa[(size_t) v.layer]) own += visit_cache_bytes(g, v.layer, context, fp8, int8);
        own += (uint64_t) kConvRing * (2ull * kKvHeads * kHeadDim + 2ull * (uint64_t) g.n_embd) * 4;
        own += (uint64_t) kBatchRows * (uint64_t) g.n_used * 4;
    }
    return own;
}

uint64_t pool_bytes(const Geometry& g, int64_t positions, bool fp8, bool int8, uint64_t gran, int cut, int card) {
    const uint64_t cells = (uint64_t) kKvHeads * kv_cache_row_bytes(fp8, int8) * (uint64_t) positions;
    uint64_t n = 0;
    for (const Visit& v : g.visits)
        if (!g.swa[(size_t) v.layer] && (cut > 0 && v.layer >= cut ? 1 : 0) == card) n += 2 * ((cells + gran - 1) / gran * gran);
    return n;
}

uint64_t shared_bytes(const Geometry& g, bool drafter, bool head) {
    return kSharedFixed + prompt_bytes(g) + (drafter ? dspark_tap_bytes(std::max(kBatchRows, kPromptChunk)) + (32ull << 20) : 0) +   // PROMPT, DSPARK HOOKS
           (head ? kernels::sampler_wide_bytes((int) g.n_vocab) : 0);
}

InklingModel::InklingModel() = default;

InklingModel::~InklingModel() {
    // an upload that may still run keeps everything it reads or writes (and the stream it runs on)
    if (!drain_loan_upload() || loan_stuck_) return;
    if (loan_event_) cudaEventDestroy((cudaEvent_t) loan_event_);
    if (loan_stream_) cudaStreamDestroy((cudaStream_t) loan_stream_);
    peer_.reset();
    for (const auto& [d, p] : peer_bufs_) {
        cudaSetDevice(d);
        cudaFree(p);
    }
    if (!peer_bufs_.empty()) cudaSetDevice(device_);
    for (void* p : dev_) cudaFree(p);
    for (size_t g = 0; g < copy_maps_.size(); ++g) core::vmm::unmap(copies_va_ + g * copy_lay_.gran, copy_lay_.gran, copy_maps_[g]);
    if (copies_va_) core::vmm::release_range(copies_va_, copy_reserved_);
    if (loan_stage_) {
        core::vmm::unmap_only(loan_stage_, loan_stage_bytes_);
        core::vmm::release_range(loan_stage_, loan_stage_bytes_);
    }
    if (loan_addr_) cudaFreeHost(loan_addr_);
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
    if (req.vram_reserve_mib > 0) {
        const int floor = platform::gpu_memory_pages() ? 256 : 128;   // less makes the driver page GPU memory
        if (req.vram_reserve_mib < floor)
            return {model::Status::Error, "inkling2: --vram-reserve-mib " + std::to_string(req.vram_reserve_mib) + " is below " +
                                              std::to_string(floor) + " MiB"};
        spare_ = (uint64_t) req.vram_reserve_mib * kMiB;
    }
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
        w.expert_stride = kernels::native_expert_stride(el.gu_type, el.blob_bytes);
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
        basalt::core::gpu_mem_info(&before, &total_b);
        if (!vis_.stage(pack_.vision, g.n_embd, err)) return fail("vision: " + err);
        basalt::core::gpu_mem_info(&after, &total_b);
        vision_modules = std::min<uint64_t>(before > after ? before - after : 0, kCardSpare / 4);
        vis_.set_lender({[this](uint64_t bytes, std::string& e) { return lend_room(bytes, e); }, [this] { refill(); }});
        basalt::log::info("basalt inkling2", "image encoder: %.2f GiB staged in page-locked RAM, 0 GiB of the card kept for it (its "
                          "kernels' %.0f MiB inside the card's spare; a request's pictures borrow up to %.2f GiB of expert memory "
                          "and give it back)",
                          (double) vis_.weights_bytes() / (double) kGiB, (double) vision_modules / (double) kMiB,
                          (double) vis_.region_bytes() / (double) kGiB);
    }

    // the context: every session reaches the whole --max-context, its full visits' cells backed from the card's KV pool as
    // its conversation grows; the pool is what the weights, the sessions' own buffers and the card's spare leave, at least
    // one session at the whole context or every session at its first granule, whichever is more
    {
        const int64_t want = req.max_context > 0 ? req.max_context : kDefaultContext;
        if (want < 1024 || (g.context_length > 0 && want > g.context_length))
            return fail("--max-context " + std::to_string(want) + " is outside 1,024 .. " + std::to_string(g.context_length));
        // the spare the sessions are sized against: the asked reserve (--vram-reserve-mib, the card's own otherwise),
        // less what the image encoder staged on the card (its kernels' modules, capped at a quarter of the card's spare)
        const uint64_t spare_base = spare_ ? spare_ : kCardSpare;
        const uint64_t spare = spare_base > vision_modules ? spare_base - vision_modules : 0;   // [vision]
        const int n_sess = std::max(req.sessions, 1);
        if (!core::vmm::supported()) return fail("the KV pool needs virtual memory management, which this card does not offer");
        const uint64_t gran = core::vmm::granularity();
        // DSPARK HOOK: a session's drafter beside its own buffers, its full layers' cells in the main card's pool
        const auto own_on = [&](int c) { return session_bytes(g, want, fp8_, int8_, cut_, c) + (c == 0 && dspark_ ? dspark_session_bytes(pack_) : 0); };
        // a pool of `positions` positions of one session (its granules rounded as `round` says)
        const auto one_on = [&](int c, int64_t positions, uint64_t round) {
            return pool_bytes(g, positions, fp8_, int8_, round, cut_, c) + (c == 0 && dspark_ ? dspark_pool_bytes(pack_, positions, round, int8_) : 0);
        };
        // the least pool: every session at its first granule (the pool takes what is left after the experts; a conversation
        // that outgrows it waits for room or is refused); with --kv-pool off every session holds the whole context
        const auto pool_on = [&](int c) {
            return req.kv_pool ? (uint64_t) n_sess * one_on(c, 1, gran) : (uint64_t) n_sess * one_on(c, want, gran);
        };
        // a position's bytes in card c's pool, granules aside
        const auto per_on = [&](int c) { return one_on(c, 2, 1) - one_on(c, 1, 1); };
        const uint64_t shared = shared_bytes(g, dspark_ != nullptr, true);
        // what card c keeps beside its pool: the sessions' own buffers, what they share and the spare
        const auto kept_on = [&](int c) {
            return own_on(c) * (uint64_t) n_sess + (c == 0 ? shared + batch_graphs_bytes(n_sess) + spare : shared_bytes(g, false, false) + kCardSpare);
        };
        const auto size_pool = [&](int c, uint64_t free_b) -> model::Result {
            const uint64_t kept = kept_on(c);
            const uint64_t least = pool_on(c);
            if (free_b < kept + least) {
                const uint64_t per = per_on(c);
                const uint64_t room = free_b > kept ? free_b - kept : 0;
                return {model::Status::Error, "inkling2: --max-context " + std::to_string(want) + " with " + std::to_string(n_sess) +
                                                  " slot" + (n_sess == 1 ? "" : "s") + " needs a KV pool of " + std::to_string(least >> 20) +
                                                  " MiB on CUDA" + std::to_string(card_device(c)) + "; " + std::to_string(room >> 20) +
                                                  " MiB are left beside the weights and the slots' own buffers (about " +
                                                  std::to_string(per ? room / per : 0) + " positions" +
                                                  (fp8_ || int8_ ? ")" : "; --kv fp8 halves a position's bytes)")};
            }
            kv_pool_[c].set_budget(free_b - kept);
            return {};
        };
        if (cut_ > 0) {   // the layer split: each card holds its layers' experts (no spill), checked once they are placed
            if (req.expert_cache > 0) return fail("--expert-cache takes one card; the layer split keeps every expert on its card");
            if (model::Result r = load_experts(reads, file, experts); !r.ok()) return r;
            for (int c = 0; c < 2; ++c) {
                cudaSetDevice(card_device(c));
                size_t fb = 0, tb = 0;
                basalt::core::gpu_mem_info(&fb, &tb);
                if (model::Result r = size_pool(c, fb); !r.ok()) {
                    cudaSetDevice(device_);
                    return r;
                }
            }
            cudaSetDevice(device_);
            ctx_ = want;
        } else {
        // the experts: all on the card when they fit beside the slots' own buffers and the least pool; else the least routed
        // go to RAM, just enough (spill.hpp). --expert-cache N keeps at most N on the card (a diagnostic).
        {
            size_t free_b = 0, total_b = 0;
            basalt::core::gpu_mem_info(&free_b, &total_b);
            std::vector<uint64_t> stride;
            for (const LayerWeights& w : layers_) stride.push_back(w.expert_stride);
            const bool vmm = core::vmm::supported();
            const uint64_t vg = vmm ? core::vmm::granularity() : 0;
            uint64_t full = 0;
            for (int l = 0; l < g.n_layers; ++l) {
                const ExpertLayer& el = pack_.experts[(size_t) l];
                const uint64_t eb = (uint64_t) g.n_expert * stride[(size_t) l];
                const uint64_t past = std::max<uint64_t>(
                    16, std::max(prefill::mmq::read_past_bytes(el.gu_type, g.n_embd),
                                 prefill::mmq::read_past_bytes(el.d_type, g.n_ff_exp)));
                full += vg ? (eb + past + vg - 1) / vg * vg : eb + past;
            }
            // include the margin for tables, maps and driver rounding before deciding whether anything must spill
            const uint64_t want_b = full + kept_on(0) + pool_on(0) + (vmm ? 128 * kMiB : 0);
            const uint64_t short_b = want_b > free_b ? want_b - free_b : 0;
            basalt::log::info("basalt inkling2", "card plan: %.2f GiB free; experts %.2f GiB, the slots' own %.2f GiB (%.1f MiB each), "
                              "shared and spare %.2f GiB, the least pool %.2f GiB",
                              (double) free_b / (double) kGiB, (double) full / (double) kGiB,
                              (double) (own_on(0) * (uint64_t) n_sess) / (double) kGiB, (double) own_on(0) / (double) kMiB,
                              (double) (kept_on(0) - own_on(0) * (uint64_t) n_sess) / (double) kGiB, (double) pool_on(0) / (double) kGiB);
            const int64_t keep = req.expert_cache > 0 ? req.expert_cache : -1;
            if (short_b > 0 || keep >= 0) {
                if (!vmm) return fail("the experts do not fit beside the context and this card has no virtual memory management");
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
                has_spill_ = spill_.any();
                if (has_spill_) {
                    // the RAM experts, the most routed first (the profile's order; without one, layer by layer)
                    for (const auto& [l, e] : rank)
                        if (!spill_.slot[(size_t) l].empty() && spill_.slot[(size_t) l][(size_t) e] >= g.n_expert - spill_.n_spill[(size_t) l])
                            spill_order_.emplace_back(l, e);
                    if (rank.empty())
                        for (int l = 0; l < g.n_layers; ++l)
                            for (int e = 0; e < g.n_expert; ++e)
                                if (!on_card(l, e)) spill_order_.emplace_back(l, e);
                    elastic_ = keep < 0 && want > core::kElasticKvBase;
                    std::vector<int> visits((size_t) g.n_layers, 0);
                    for (const Visit& v : g.visits) ++visits[(size_t) v.layer];
                    heat_.start((size_t) g.n_layers * g.n_expert, g.n_expert, spill_order_, visits);
                    uint64_t ram = 0;
                    for (int l = 0; l < g.n_layers; ++l) ram += (uint64_t) spill_.n_spill[(size_t) l] * stride[(size_t) l];
                    basalt::log::info("basalt inkling2",
                                      "experts: %lld of %lld in RAM (%.2f GiB, the least routed%s), %.2f GiB on the card with %.0f MiB of "
                                      "prompt staging, so --max-context %lld fits",
                                      (long long) spill_.spilled, (long long) g.n_layers * g.n_expert, (double) ram / (double) kGiB,
                                      rank.empty() ? ", no profile" : " by the profile", (double) spill_.device / (double) kGiB,
                                      (double) spill_.staging / (double) kMiB, (long long) want);
                }
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
        basalt::core::gpu_mem_info(&free_b, &total_b);
        if (model::Result r = size_pool(0, free_b); !r.ok()) return r;
        ctx_ = want;
        }
        // the sampler's list for a request without a top-k limit (top_k 0 or above 64), counted in the first card's shared
        // bytes: made now, so no request allocates it beside a full pool
        if (!kernels::sampler_wide_reserve((int) g.n_vocab, nullptr))
            return fail("out of device memory for the sampler's list (" + std::to_string(kernels::sampler_wide_bytes((int) g.n_vocab) >> 20) +
                        " MiB)");
        for (int c = 0; c < cards(); ++c)
            if (!req.kv_pool)
                basalt::log::info("basalt inkling2", "KV%s: each of the %d slot%s holds its own %lld positions (--kv-pool off)",
                                  cards() > 1 ? (" on CUDA" + std::to_string(card_device(c))).c_str() : "", n_sess, n_sess == 1 ? "" : "s",
                                  (long long) want);
            else
            {
                // the positions a slot backs when `slots` slots back as many and the rest one granule each, the caches'
                // granules counted whole (each grow maps a granule of every cache, the drafter's rings too)
                const auto reach = [&](int64_t slots) {
                    const uint64_t rest = (uint64_t) (n_sess - slots) * one_on(c, 1, gran);
                    int64_t lo = 0, hi = want;
                    while (lo < hi) {
                        const int64_t mid = lo + (hi - lo + 1) / 2;
                        if (rest + (uint64_t) slots * one_on(c, mid, gran) <= kv_pool_[c].budget()) lo = mid;
                        else hi = mid - 1;
                    }
                    return (long long) lo;
                };
                const long long each = reach(n_sess);
                basalt::log::info("basalt inkling2",
                                  "KV pool%s: %.0f MiB, %lld positions across the %d slot%s (%lld a slot when all are full; one slot "
                                  "can reach %lld)",
                                  cards() > 1 ? (" on CUDA" + std::to_string(card_device(c))).c_str() : "",
                                  (double) kv_pool_[c].budget() / (double) kMiB, each * n_sess, n_sess, n_sess == 1 ? "" : "s", each,
                                  reach(1));
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
            if (!map || basalt::core::device_copy(map, by_slot.data(), (size_t) g.n_expert * 4, cudaMemcpyHostToDevice) != cudaSuccess)
                return fail("out of device memory loading layer " + std::to_string(l) + "'s expert slots");
            w.slot_map = map;
            int32_t* full = reinterpret_cast<int32_t*>(device_buf((uint64_t) g.n_expert * 4));
            if (!full || basalt::core::device_copy(full, slot.data(), (size_t) g.n_expert * 4, cudaMemcpyHostToDevice) != cudaSuccess)
                return fail("out of device memory loading layer " + std::to_string(l) + "'s expert slots");
            w.slot_full = full;
        }
        if (!(w.blob_table = reinterpret_cast<unsigned long long*>(device_buf(table.size() * 8))) ||
            basalt::core::device_copy(w.blob_table, table.data(), table.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess)
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
    const uint64_t gran = core::vmm::supported() ? core::vmm::granularity() : 1;
    std::vector<int> visits((size_t) g.n_layers, 0);
    for (const Visit& v : g.visits) ++visits[(size_t) v.layer];
    model::SplitSpec s;
    for (int l = 0; l < g.n_layers; ++l) {
        const ExpertLayer& el = pack_.experts[(size_t) l];
        const uint64_t dense = (uint64_t) kProj * kQ8Row + (uint64_t) g.n_embd * (3072 / 32 * 34) +
                               prefill::q8::scale_plane_bytes(kProjQ, (int64_t) g.n_embd) +
                               2 * prefill::q8::scale_plane_bytes(kProjKV, (int64_t) g.n_embd) +
                               prefill::q8::scale_plane_bytes((int64_t) g.n_embd, 3072);
        // a sliding-window visit's rings in every slot; a full visit's cells in the card's KV pool, at least one slot at the
        // whole context or every slot at one granule (with --kv-pool off, every slot at the whole context)
        const uint64_t cells = (uint64_t) kKvHeads * kv_cache_row_bytes(fp8, int8) * (uint64_t) want;
        const uint64_t full = (cells + gran - 1) / gran * gran;
        const uint64_t kv = g.swa[(size_t) l] ? visit_cache_bytes(g, l, want, fp8, int8) * n_sess
                                              : 2 * (req.kv_pool ? std::max(full, n_sess * gran) : n_sess * full);
        s.layer.push_back((uint64_t) g.n_expert * kernels::native_expert_stride(el.gu_type, el.blob_bytes) + dense + (uint64_t) visits[(size_t) l] * kv);
    }
    const uint64_t slot_rest = session_bytes(g, want, fp8, int8, g.n_layers, 1);   // a slot's own beyond its caches (no layer on card 1)
    s.fixed = {kCardSpare + shared_bytes(g, req.spec >= 2, true) + batch_graphs_bytes(n_sess) + slot_rest * n_sess + (req.spec >= 2 ? 1000ull << 20 : 0),
               kCardSpare + shared_bytes(g, false, false) + slot_rest * n_sess};
    for (const model::GpuCards::Card& card : c.cards) {
        cudaSetDevice(card.device);
        size_t free_b = 0, total_b = 0;
        s.room.push_back(basalt::core::gpu_mem_info(&free_b, &total_b) == cudaSuccess ? free_b : 0);
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
        if (cudaMalloc(&d, (size_t) ne * 4) != cudaSuccess || basalt::core::device_copy(d, map.data(), (size_t) ne * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
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
            if (const auto error = cudaSetDevice(card_device(c)); error != cudaSuccess) {
                cudaSetDevice(device_);
                return {model::Status::Error, std::string("inkling2: cannot select the layer's device: ") + cudaGetErrorString(error)};
            }
            if (int8_) {
                try {
                    attention_decode_int8_init();
                } catch (const std::runtime_error& error) {
                    cudaSetDevice(device_);
                    return {model::Status::Error, error.what()};
                }
            }
            void* tmp = nullptr;
            if (cudaMalloc(&tmp, std::max(proj_b, out_b)) != cudaSuccess) {
                cudaSetDevice(device_);
                return {model::Status::Error, "inkling2: out of device memory converting Q8_0 projections to planes"};
            }
            for (int l = 0; l < pack_.geo.n_layers; ++l) {
                if (card_of(l) != c) continue;
                LayerWeights& w = layers_[(size_t) l];
                kernels::wgemm::q8_planes(w.proj, kProj, E, tmp, nullptr);
                basalt::core::device_copy(w.proj, tmp, proj_b, cudaMemcpyDeviceToDevice);
                kernels::wgemm::q8_planes(w.out, E, 3072, tmp, nullptr);
                basalt::core::device_copy(w.out, tmp, out_b, cudaMemcpyDeviceToDevice);
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
    {   // Repack lm_head from Q8_0 blocks to aligned q8_planes in-place (zero extra persistent VRAM).
        const int64_t V = pack_.geo.n_vocab, E = pack_.geo.n_embd;
        const size_t head_b = (size_t) V * (size_t) (E / 32 * 34);
        cudaSetDevice(device_);
        void* tmp = nullptr;
        if (cudaMalloc(&tmp, head_b) != cudaSuccess)
            return {model::Status::Error, "inkling2: out of device memory converting lm_head to q8_planes"};
        kernels::wgemm::q8_planes(head_, V, E, tmp, nullptr);
        basalt::core::device_copy(head_, tmp, head_b, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        cudaFree(tmp);
    }
    for (auto& s : sessions)
        if (model::Result r = s->prepare(); !r.ok()) return r;
    // the asked reserve, after the whole load (the planes converted, the drafter folded, the sessions prepared): a load
    // that leaves less than the flag asks is refused rather than served at a smaller reserve, so a --vram-reserve-mib
    // that cannot be met is never silently served
    if (spare_) {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b < spare_) {
            return {model::Status::Error,
                    "inkling2: --vram-reserve-mib " + std::to_string(spare_ / kMiB) + " MiB cannot be met after loading: only " +
                        std::to_string(free_b / kMiB) + " MiB free; lower --max-context or --spec, or use a bigger card"};
        }
    }
    if (copies()) {   // the sessions' buffers are in place (an elastic spill's caches back only their base): the RAM experts
                      // come back while the card has room beyond its spare
        size_t was_b = 0, free_b = 0, total_b = 0;
        basalt::core::gpu_mem_info(&was_b, &total_b);
        sync_room();
        const int64_t before = card_copies();
        const uint64_t moved = promote(held_free(kCardSpare));
        note_shortage("start");
        if (wedged()) {
            log_copy_change("startup failed", before);
            return {model::Status::Error, "inkling2: expert copy placement failed"};
        }
        basalt::core::gpu_mem_info(&free_b, &total_b);
        if (elastic_)
            basalt::log::info("basalt inkling2", "elastic spill: %lld of %lld RAM experts copied back to the card (%.2f GiB), %.0f -> %.0f MiB "
                              "free; they go back to RAM, the least routed first, as a conversation's caches grow",
                              (long long) promoted(), (long long) spill_.spilled, (double) moved / (double) kGiB, (double) was_b / (double) kMiB,
                              (double) free_b / (double) kMiB);
        else
            basalt::log::info("basalt inkling2", "static spill: %lld of %lld RAM experts copied back to the card (%.2f GiB), %.0f -> %.0f MiB free",
                              (long long) promoted(), (long long) spill_.spilled, (double) moved / (double) kGiB, (double) was_b / (double) kMiB,
                              (double) free_b / (double) kMiB);
        log_copy_change("startup", before);
    }
    return {};
}

bool InklingModel::copy_range() {
    if (copies_va_) return true;
    const uint64_t gran = core::vmm::granularity();
    if (spill_order_.empty() || !gran) return false;
    std::vector<uint64_t> strides;
    for (const auto& [l, e] : spill_order_) strides.push_back(layers_[(size_t) l].expert_stride);
    copy_lay_.gran = gran;
    copy_reserved_ = copy_layout(strides, gran).bytes();
    copy_slots_.reserve(strides.size());
    copy_lay_.at.reserve(strides.size());
    copy_lay_.end.reserve(strides.size());
    copies_va_ = core::vmm::reserve(copy_reserved_);
    return copies_va_ != nullptr;
}

// The granules past the last copy go back to the driver. One that could not be unmapped (or whose handle could not be
// released) leaves the range in an unknown state: it stays tracked for the end, and the engine stops (wedge).
bool InklingModel::trim_copies() {
    const uint64_t gran = copy_lay_.gran, keep = copy_lay_.pages(promoted());
    while (copy_maps_.size() > keep) {
        if (!core::vmm::unmap(copies_va_ + (copy_maps_.size() - 1) * gran, gran, copy_maps_.back())) {
            copies_retired_ = true;
            wedge();
            basalt::log::error("basalt inkling2", "a granule of the expert copies could not be released; restart the engine");
            return false;
        }
        copy_maps_.pop_back();
        room_ += (int64_t) gran;
    }
    return true;
}

uint64_t InklingModel::late_room() const { return spare_ ? kLateRoom : 0; }

void InklingModel::sync_room() {
    size_t free_b = 0, total_b = 0;
    room_ = basalt::core::gpu_mem_info(&free_b, &total_b) == cudaSuccess ? (int64_t) free_b : 0;
}

std::vector<size_t> InklingModel::copy_candidates() const {
    const size_t ne = geo().n_expert;
    std::vector<size_t> pairs;
    for (const auto& [l, e] : spill_order_) {
        const size_t pair = (size_t) l * ne + e;
        if (!copy_[pair]) pairs.push_back(pair);
    }
    return pairs;
}

void InklingModel::learn(const std::vector<uint32_t>& lookups, int64_t rows) { heat_.learn(lookups, rows); }

uint64_t InklingModel::copy_stride(size_t pair) const {
    return layers_[pair / (size_t) geo().n_expert].expert_stride;
}

const uint8_t* InklingModel::copy_source(size_t pair) const {
    const size_t ne = geo().n_expert, l = pair / ne, e = pair % ne;
    const LayerWeights& w = layers_[l];
    if (peer_held((int) l, (int) e)) return peer_->slot_ptr(peer_slot_[pair]);   // the second card's copy: none in RAM
    const int slot = spill_.slot[l][e] - (geo().n_expert - w.n_spill);
    return w.spill_host + (uint64_t) slot * w.expert_stride;
}

bool InklingModel::join_placement() {
    if (wedged()) return false;
    // every reader of the tables and copies (the sessions' streams, the CPU experts' merge, the loan's uploads) is done
    if (cudaDeviceSynchronize() == cudaSuccess) return true;
    wedge();
    basalt::log::error("basalt inkling2", "expert copy placement could not be confirmed; restart the engine");
    return false;
}

bool InklingModel::publish_copy(size_t pair, const void* ptr) {
    const size_t ne = geo().n_expert, l = pair / ne, e = pair % ne;
    const unsigned long long addr = ptr ? (unsigned long long) ptr : 0;
    if (basalt::core::device_copy(layers_[l].blob_table + e, &addr, sizeof(addr), cudaMemcpyHostToDevice) == cudaSuccess) return true;
    wedge();
    basalt::log::error("basalt inkling2", "an expert copy table update failed; restart the engine");
    return false;
}

uint64_t InklingModel::promote(uint64_t keep_free) {
    if (copies_retired_ || wedged() || (size_t) promoted() == spill_order_.size()) return 0;
    return promote_pairs(by_score_desc(heat_, copy_candidates()), keep_free + late_room());
}

uint64_t InklingModel::promote_pairs(const std::vector<size_t>& pairs, uint64_t keep_free) {
    if (copies_retired_ || wedged() || pairs.empty()) return 0;
    if (!copy_range()) {   // no address range for the copies (with virtual memory management): placement leaves its rule
        map_short_ = copy_lay_.gran != 0;
        return 0;
    }
    if (!join_placement()) return 0;
    const size_t ne = geo().n_expert;
    const uint64_t gran = copy_lay_.gran;
    uint64_t moved = 0;
    for (const size_t pair : pairs) {
        const uint64_t stride = copy_stride(pair);
        const uint64_t start = copy_lay_.end.empty() ? 0 : CopyLayout::aligned(copy_lay_.end.back());
        const uint64_t mapped = (uint64_t) copy_maps_.size() * gran, end = start + stride;
        const uint64_t grow = end > mapped ? (end - mapped + gran - 1) / gran * gran : 0;
        if (room() < keep_free + grow) break;
        bool mapped_ok = true, out_of_memory = false;
        for (uint64_t g = 0; g < grow && mapped_ok; g += gran) {
            const unsigned long long h = core::vmm::map(copies_va_ + mapped + g, gran, &out_of_memory);
            if (h) {
                copy_maps_.push_back(h);
                room_ -= (int64_t) gran;
            } else {
                mapped_ok = false;
            }
        }
        if (!mapped_ok && !out_of_memory) {   // the range may hold a mapping nothing tracks
            wedge();
            basalt::log::error("basalt inkling2", "an expert copy's page could not be mapped; restart the engine");
            return moved;
        }
        if (!mapped_ok) {   // the card had less free memory than counted: the copies stop here and the count follows the card
            map_short_ = true;
            break;
        }
        uint8_t* ptr = copies_va_ + start;
        if (basalt::core::device_copy(ptr, copy_source(pair), stride, cudaMemcpyDefault) != cudaSuccess) {
            wedge();
            basalt::log::error("basalt inkling2", "an expert copy upload failed; restart the engine");
            return moved;
        }
        if (!publish_copy(pair, ptr)) return moved;
        copy_lay_.append(stride);
        copy_slots_.push_back(pair);
        copy_[pair] = ptr;
        ++layer_copies_[pair / ne];
        moved += stride;
    }
    if (join_placement()) trim_copies();   // a step that stopped part-way leaves no granule behind
    if (map_short_) {   // the count follows the card, and copies go back until it keeps `keep_free` free again
        sync_room();
        give_back(keep_free, [&] { return room(); }, [&] { return promoted(); },
                  [&](uint64_t bytes, uint64_t& freed) { return demote(bytes, freed); });
    }
    return moved;
}

void InklingModel::note_shortage(const std::string& where) {
    if (std::exchange(map_short_, false))
        basalt::log::warn("basalt inkling2", "elastic spill: placement changed by a memory shortage at %s", where.c_str());
}

void InklingModel::log_copy_change(const char* event, int64_t before, uint64_t to_map, bool always, bool warning) const {
    const int64_t after = card_copies();
    if (before == after && !always && !wedged()) return;
    size_t free_b = 0, total_b = 0;
    const std::string free_mib = basalt::core::gpu_mem_info(&free_b, &total_b) == cudaSuccess
        ? std::to_string(free_b / kMiB) : "unknown";
    const auto log = warning || wedged() ? basalt::log::warn : basalt::log::info;
    // which experts have a copy, paid or loaned, as one number: equal runs log equal sets
    uint64_t set = 1469598103934665603ull;
    for (size_t pair = 0; pair < copy_.size(); ++pair)
        if (copy_[pair]) set = (set ^ (pair * 2 + (loan_flag_.empty() ? 0 : loan_flag_[pair]))) * 1099511628211ull;
    log("basalt inkling2", "elastic spill: %s; expert copies %lld -> %lld%s (%lld loaned after); "
        "%s MiB free, %.2f MiB paid mapped, %.2f MiB to map; copy set %016llx",
        event, (long long) before, (long long) after, wedged() ? " (tracked; placement unconfirmed)" : "",
        (long long) loaned(), free_mib.c_str(), (double) copy_maps_.size() * copy_lay_.gran / kMiB, (double) to_map / kMiB,
        (unsigned long long) set);
}

bool InklingModel::resize_reserve(uint64_t free_bytes) {
    sync_room();   // a resize is an order to look at the card again
    const uint64_t free_b = room();
    const uint64_t want = std::max(spare_ ? spare_ : kCardSpare, free_bytes);
    if (!has_spill_) {
        if (free_b < want || wedged()) return false;
        reserve_ = free_bytes;
        return true;
    }
    const int64_t before = card_copies();
    const std::string event = "VRAM reserve " + std::to_string(free_bytes / kMiB) + " MiB";
    if (free_b < want) {
        const uint64_t needed = want - free_b;
        const uint64_t reclaimable = promoted() > 0 && !lent_ && !copies_retired_ && !wedged()
            ? (uint64_t) copy_maps_.size() * copy_lay_.gran : 0;
        if (needed > reclaimable) {
            log_copy_change((event + " refused").c_str(), before, 0, true, true);
            return false;
        }
        uint64_t freed = 0;
        if (!demote(needed, freed)) {
            log_copy_change((event + " failed").c_str(), before);
            return false;
        }
    } else if (!lent_) {
        promote(want);
        note_shortage(event);
    }
    log_copy_change(event.c_str(), before, 0, reserve_ != free_bytes);
    if (wedged()) return false;
    reserve_ = free_bytes;
    return true;
}

bool InklingModel::lend(bool low) {
    const int64_t before = card_copies();
    if (low) {
        if (loaned() > 0) return false;
        uint64_t freed = 0;
        if (!demote(~0ull, freed)) {
            log_copy_change("sleep failed", before);
            return false;
        }
        lent_ = true;
    } else {
        lent_ = false;
        sync_room();   // the driver brought the rest back: the card as it is now
        promote(held_free(kCardSpare));
        note_shortage("wake");
    }
    log_copy_change(low ? "sleep" : "wake", before);
    return !wedged();
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

bool InklingModel::stage(int l, const uint8_t* want, void* stream, bool with_peer) const {
    const LayerWeights& w = layers_[(size_t) l];
    if (wedged() || loaned() > 0) return false;   // a loan lives in the staging: a read takes it back first
    if (w.n_spill == 0) return true;
    const int ne = geo().n_expert, keep = ne - w.n_spill;
    const int n = with_peer ? w.n_spill : ram_n(l);   // the RAM experts' slots come first, the second card's last
    std::vector<const void*> card((size_t) w.n_spill, nullptr);   // [RAM slot] the expert's card copy, null: none
    std::vector<uint8_t> has((size_t) w.n_spill, 0);
    if (layer_copies_[(size_t) l] > 0)
        for (int e = 0; e < ne; ++e)
            if (const int s = spill_.slot[(size_t) l][(size_t) e]; s >= keep) {
                card[(size_t) (s - keep)] = copy_[(size_t) l * ne + e];
                has[(size_t) (s - keep)] = card[(size_t) (s - keep)] != nullptr;
            }
    const cudaStream_t st = (cudaStream_t) stream;
    for (const StageOp& op : stage_plan(n, has.data(), want ? want + keep : nullptr, keep > 0)) {
        uint8_t* dst = w.experts + (uint64_t) (keep + op.slot) * w.expert_stride;
        const cudaError_t e =
            op.kind == StageOp::Guard ? cudaMemsetAsync(dst, 0, kSlotGuard, st)
            : op.kind == StageOp::Card
                ? cudaMemcpyAsync(dst, card[(size_t) op.slot], w.expert_stride, cudaMemcpyDeviceToDevice, st)
                : cudaMemcpyAsync(dst, w.spill_host + (uint64_t) op.slot * w.expert_stride, (size_t) op.n * w.expert_stride,
                                  cudaMemcpyHostToDevice, st);
        if (e != cudaSuccess) return false;
    }
    return true;
}

bool InklingModel::demote(uint64_t bytes, uint64_t& freed) {
    freed = 0;
    if (wedged()) return false;
    if (copy_slots_.empty() || bytes == 0) return true;
    const size_t ne = geo().n_expert;
    const uint64_t mapped = copy_maps_.size(), gran = copy_lay_.gran;
    const std::vector<size_t> held = by_score_desc(heat_, copy_slots_);
    // The aligned sizes add up to the same granule count in rank order and in packed slot order.
    CopyLayout ranked;
    ranked.gran = gran;
    for (const size_t pair : held) ranked.append(copy_stride(pair));
    const size_t count = (size_t) ranked.keep_for((int64_t) held.size(), mapped, bytes);
    std::vector<uint8_t> evict(copy_.size(), 0);
    for (size_t i = count; i < held.size(); ++i) evict[held[i]] = 1;

    if (!join_placement()) return false;
    // Clear evicted entries first. Keep the survivors in slot order so each moves only left and cannot overwrite a later source.
    for (const size_t pair : copy_slots_)
        if (evict[pair] && !publish_copy(pair, nullptr)) return false;

    size_t keep = 0;
    uint64_t end = 0;
    for (const size_t pair : copy_slots_) {
        if (evict[pair]) {
            copy_[pair] = nullptr;
            --layer_copies_[pair / ne];
            continue;
        }
        const uint64_t start = CopyLayout::aligned(end), stride = copy_stride(pair);
        uint8_t* dst = copies_va_ + start;
        if (dst != copy_[pair]) {
            // Device copies cannot overlap; the original RAM blob is still available.
            const bool overlaps = dst + stride > static_cast<uint8_t*>(copy_[pair]);
            if (basalt::core::device_copy(dst, overlaps ? copy_source(pair) : copy_[pair], stride, cudaMemcpyDefault) != cudaSuccess) {
                wedge();
                basalt::log::error("basalt inkling2", "expert copy compaction failed; restart the engine");
                return false;
            }
            if (!publish_copy(pair, dst)) return false;
            copy_[pair] = dst;
        }
        end = start + stride;
        copy_slots_[keep] = pair;
        copy_lay_.at[keep] = start;
        copy_lay_.end[keep++] = end;
    }
    if (keep != copy_slots_.size()) ++epoch_;
    copy_slots_.resize(keep);
    copy_lay_.at.resize(keep);
    copy_lay_.end.resize(keep);
    if (!join_placement()) return false;
    const bool ok = trim_copies();
    freed = (mapped - (uint64_t) copy_maps_.size()) * gran;
    return ok;
}

int InklingModel::rebalance(int max_swaps) {
    if (!elastic_ || lent_ || wedged() || copies_retired_ || copy_slots_.empty() || max_swaps <= 0) return 0;
    const size_t ne = geo().n_expert;
    const auto plan = plan_swaps(heat_, copy_slots_, copy_candidates(),
                                 [&](size_t pair) { return copy_stride(pair); }, max_swaps);
    if (plan.empty() || !join_placement()) return 0;
    int done = 0;
    for (const CopySwap& swap : plan) {
        uint8_t* ptr = static_cast<uint8_t*>(copy_[swap.out]);
        const auto at = std::lower_bound(copy_lay_.at.begin(), copy_lay_.at.end(), (uint64_t) (ptr - copies_va_));
        if (!publish_copy(swap.out, nullptr)) break;
        if (basalt::core::device_copy(ptr, copy_source(swap.in), copy_stride(swap.in), cudaMemcpyDefault) != cudaSuccess) {
            wedge();
            basalt::log::error("basalt inkling2", "an expert copy swap failed; restart the engine");
            break;
        }
        if (!publish_copy(swap.in, ptr)) break;
        copy_slots_[(size_t) (at - copy_lay_.at.begin())] = swap.in;
        copy_[swap.out] = nullptr;
        copy_[swap.in] = ptr;
        --layer_copies_[swap.out / ne];
        ++layer_copies_[swap.in / ne];
        ++done;
    }
    if (done > 0) ++epoch_;
    join_placement();
    return done;
}

bool InklingModel::set_prompt_arena(uint8_t* base, uint64_t bytes) {
    if (!base) {
        // freed only once no upload writes it and no reader of a loaned copy can still run
        if (!drain_loan_upload() || !recall_loan(nullptr))
            basalt::log::error("basalt inkling2", "the prompt arena goes while expert copies are still loaned in it");
        if (loan_stuck_) return false;
        loan_arena_ = nullptr;
        loan_arena_bytes_ = 0;
        return true;
    }
    loan_arena_ = base;
    loan_arena_bytes_ = bytes;
    loan_due_ = loan_ready();
    return true;
}

// The shared staging's granules mapped a second time, at an address range of their own: the layers' ranges (the prompt
// path's view of the staging) stay as they are.
bool InklingModel::loan_ready() {
    if (loan_state_ != 0) return loan_state_ > 0;
    if (!elastic_ || sessions() != 1 || cards() != 1 || !loan_arena_ || spill_order_.empty() || !core::vmm::supported()) return false;
    loan_state_ = -1;
    if (cudaHostAlloc((void**) &loan_addr_, spill_order_.size() * sizeof(unsigned long long), cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        loan_addr_ = nullptr;
        return false;
    }
    cudaStream_t copy = nullptr;
    cudaEvent_t uploaded = nullptr;
    if (cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&uploaded, cudaEventDisableTiming) != cudaSuccess) {
        if (copy) cudaStreamDestroy(copy);
        cudaFreeHost(loan_addr_);
        loan_addr_ = nullptr;
        cudaGetLastError();
        return false;
    }
    loan_stream_ = copy;
    loan_event_ = uploaded;
    const uint64_t gran = core::vmm::granularity();
    if (!staging_.empty() && gran) {
        const uint64_t bytes = (uint64_t) staging_.size() * gran;
        if (uint8_t* va = core::vmm::reserve(bytes)) {
            size_t done = 0;
            while (done < staging_.size() && core::vmm::map_handle(va + done * gran, gran, staging_[done])) ++done;
            if (done == staging_.size()) {
                loan_stage_ = va;
                loan_stage_bytes_ = bytes;
            } else {
                core::vmm::unmap_only(va, done * gran);
                core::vmm::release_range(va, bytes);
            }
        }
    }
    loan_flag_.assign(copy_.size(), 0);
    loan_state_ = 1;
    return true;
}


// Queues the plan's next uploads on the copy stream, `budget` bytes of them (at least one). An upload is queued whether or
// not its expert has a paid copy by then: which ones publish is decided at publication, from the placement there.
bool InklingModel::queue_loan(uint64_t budget) {
    const cudaStream_t copy = (cudaStream_t) loan_stream_;
    uint64_t bytes = 0;
    while (loan_next_ < loan_plan_.size()) {
        const auto& [pair, ptr] = loan_plan_[loan_next_];
        const uint64_t stride = copy_stride(pair);
        if (bytes && (bytes >= budget || stride > budget - bytes)) break;
        loan_inflight_ = true;
        if (cudaMemcpyAsync(ptr, copy_source(pair), stride, cudaMemcpyDefault, copy) != cudaSuccess) {
            wedge();
            basalt::log::error("basalt inkling2", "a prompt loan upload could not be queued; restart the engine");
            return false;
        }
        ++loan_next_;
        bytes += stride;
    }
    if (bytes > 0 && cudaEventRecord((cudaEvent_t) loan_event_, copy) != cudaSuccess) {
        wedge();
        basalt::log::error("basalt inkling2", "a prompt loan completion event could not be queued; restart the engine");
        return false;
    }
    return true;
}

bool InklingModel::submit_loan(int64_t publish_at) {
    if (wedged()) return false;
    if (lent_) return true;
    if (loan_due_) {   // the plan: the highest-ranked RAM experts without a copy now, packed into the regions in rank order
        loan_due_ = false;
        if (loan_state_ != 1 || !loan_arena_) return true;
        const std::array regions{loan_arena_bytes_, loan_stage_bytes_};
        std::array<uint64_t, 2> at{};
        loan_plan_.clear();
        loan_next_ = 0;
        for (const size_t pair : by_score_desc(heat_, copy_candidates())) {
            const auto slot = next_loan_slot(copy_stride(pair), regions, at);
            if (!slot) break;
            loan_plan_.push_back({pair, (slot->region == 0 ? loan_arena_ : loan_stage_) + slot->offset});
        }
        loan_list_.reserve(loan_plan_.size());
        if (!loan_plan_.empty()) loan_publish_at_ = publish_at;
    }
    return !loan_pending() || queue_loan(kLoanBytesPerWindow);
}

bool InklingModel::publish_loan(void* stream, int64_t at) {
    if (wedged()) return false;
    if (!loan_pending()) return true;
    loan_publish_at_ = -1;
    if (!queue_loan(~0ull)) return false;
    // the whole plan becomes visible here, at a position the token stream fixes, however long its uploads took
    if (loan_inflight_ && cudaEventSynchronize((cudaEvent_t) loan_event_) != cudaSuccess) {
        wedge();
        basalt::log::error("basalt inkling2", "a prompt loan upload failed; restart the engine");
        return false;
    }
    loan_inflight_ = false;
    const int64_t before = card_copies();
    const size_t ne = geo().n_expert;
    for (const auto& [pair, ptr] : loan_plan_) {
        if (copy_[pair]) continue;   // a paid copy came after the plan: its place stays unused
        const size_t l = pair / ne, e = pair % ne;
        auto& addr = loan_addr_[loan_list_.size()];
        addr = (unsigned long long) ptr;
        // every reader of the tables runs on the decode stream or forks from it
        if (cudaMemcpyAsync(layers_[l].blob_table + e, &addr, sizeof(addr), cudaMemcpyHostToDevice,
                            (cudaStream_t) stream) != cudaSuccess) {
            wedge();
            log_copy_change("prompt loan failed", before);
            return false;
        }
        copy_[pair] = ptr;
        loan_flag_[pair] = 1;
        loan_list_.push_back(pair);
        ++layer_copies_[l];
    }
    loan_plan_.clear();
    if (before != card_copies()) ++epoch_;
    log_copy_change(("prompt loan at " + std::to_string(at)).c_str(), before);
    return true;
}

// The copy stream is idle again, or the uploads stay on record as running: their source and destination are then never
// released (the engine ends without the CUDA teardown).
bool InklingModel::drain_loan_upload() {
    if (loan_stuck_) return false;
    if (!loan_inflight_) return true;
    if (cudaStreamSynchronize((cudaStream_t) loan_stream_) == cudaSuccess) {
        loan_inflight_ = false;
        return true;
    }
    loan_stuck_ = true;
    core::gpu_left_running().store(true);
    wedge();
    basalt::log::error("basalt inkling2", "a prompt loan upload could not be drained; restart the engine");
    return false;
}

bool InklingModel::recall_loan(void* stream, const std::string& event) {
    if (wedged()) {   // nothing confirms a reader of a published loan finished: its memory is kept
        if (loaned() > 0 || loan_inflight_) loan_stuck_ = true;
        return false;
    }
    if (!drain_loan_upload()) return false;
    loan_plan_.clear();
    loan_next_ = 0;
    loan_publish_at_ = -1;
    loan_due_ = true;
    if (loaned() == 0) return true;
    const int64_t before = card_copies();
    const int ne = geo().n_expert;
    cudaStream_t st = (cudaStream_t) stream;
    if (!join_placement()) {   // a reader of the loaned copies may still run
        loan_stuck_ = true;
        return false;
    }
    // the entries go back to 0 (the CPU computes them again) before any loaned byte can be overwritten
    size_t queued = 0;
    for (auto it = loan_list_.rbegin(); it != loan_list_.rend(); ++it, ++queued) {
        const size_t i = *it;
        if (cudaMemsetAsync(layers_[i / (size_t) ne].blob_table + i % (size_t) ne, 0, 8, st) != cudaSuccess) {
            cudaGetLastError();
            break;
        }
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) {   // the cleared entries are not confirmed: a reader may still run
        loan_stuck_ = true;
        wedge();
        basalt::log::error("basalt inkling2", "the prompt loan's copies could not be recalled");
        log_copy_change((event + " failed").c_str(), before);
        return false;
    }
    for (size_t k = 0; k < queued; ++k) {
        const size_t i = loan_list_.back();
        loan_list_.pop_back();
        copy_[i] = nullptr;
        loan_flag_[i] = 0;
        --layer_copies_[i / (size_t) ne];
    }
    if (queued > 0) ++epoch_;
    if (loaned() > 0) {
        wedge();
        basalt::log::error("basalt inkling2", "%lld prompt loan copies could not be recalled", (long long) loaned());
        log_copy_change((event + " failed").c_str(), before);
        return false;
    }
    log_copy_change(event.c_str(), before);
    return true;
}

Take InklingModel::take_room(uint64_t bytes) {
    wide_out_.clear();
    if (!copies() || lent_ || wedged() || loaned() > 0) return Take::Refused;
    const uint64_t paid = (uint64_t) copy_maps_.size() * copy_lay_.gran;
    if (paid < bytes) return Take::Refused;
    // the bookkeeping first, in rank order (the order the copies come back in): nothing allocates once a copy moves
    const std::vector<size_t> held = by_score_desc(heat_, copy_slots_);
    std::vector<uint8_t> kept(copy_.size(), 0);
    wide_out_.reserve(held.size());
    const int64_t before = card_copies();
    const uint64_t room0 = room();
    uint64_t freed = 0;
    if (!demote(bytes, freed)) return Take::Failed;
    left_out(held, copy_slots_, kept, wide_out_);
    quiet([&] { log_copy_change("prompt wide chunk", before, bytes); });
    if (room() < room0 + bytes)   // short of the count: the copies come back as they were
        return give_room(0) ? Take::Refused : Take::Failed;
    room_ -= (int64_t) bytes;
    return Take::Taken;
}

bool InklingModel::give_room(uint64_t bytes) noexcept {
    room_ += (int64_t) bytes;
    if (wedged()) return false;
    if (wide_out_.empty()) return true;
    const int64_t before = card_copies();
    bool back = false;
    try {   // the copies' vectors keep the capacity the copies had, so their return allocates nothing
        promote_pairs(wide_out_, 0);
        back = restored(wide_out_, [&](size_t pair) { return copy_[pair] != nullptr; }, !wedged());
    } catch (...) {
        back = false;
    }
    wide_out_.clear();
    quiet([&] {
        note_shortage("prompt wide chunk back");
        log_copy_change("prompt wide chunk back", before, 0, false, !back);
    });
    return back;
}

// [vision] ENC runs between requests and windows: once every stream is idle, nothing reads the experts until refill.
bool InklingModel::lend_room(uint64_t bytes, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!join_placement()) {
        err = "the card failed before the image encoder";
        return false;
    }
    // the region beside the card's spare and the reserve the card holds now (a reserve it could not meet stays unmet)
    // (the copies go by the counted room, the layers below by the card's own free memory: those come back byte for byte)
    const uint64_t room0 = room();
    const uint64_t want = bytes + std::min<uint64_t>(held_free(kCardSpare), std::max<uint64_t>(room0, kCardSpare));
    size_t free_b = 0, total_b = 0;
    basalt::core::gpu_mem_info(&free_b, &total_b);
    if (has_spill_ && room0 < want && promoted() > 0) {
        const int64_t before = promoted();
        const int64_t before_total = card_copies();
        uint64_t freed = 0;
        const bool ok = demote(want - room0, freed);
        lend_copies_ += before - promoted();
        log_copy_change("image encoder lend", before_total, bytes);
        if (!ok) {
            refill();
            err = "the expert copies could not go back to RAM for the image encoder";
            return false;
        }
        basalt::core::gpu_mem_info(&free_b, &total_b);
    }
    for (size_t i = ranges_.size(); i-- > 0 && free_b < want;) {
        Range& r = ranges_[i];
        if (!r.handle) continue;
        std::unique_ptr<uint8_t[]> keep(new (std::nothrow) uint8_t[r.own]);
        if (!keep || basalt::core::device_copy(keep.get(), r.va, r.own, cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "no RAM for the experts the image encoder borrows from";
            break;
        }
        if (!core::vmm::unmap(r.va, r.own, r.handle)) {
            err = "cannot lend expert memory to the image encoder";
            break;
        }
        r.handle = 0;
        lent_ranges_.push_back({i, std::move(keep)});
        basalt::core::gpu_mem_info(&free_b, &total_b);
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
            basalt::core::device_copy(r.va, it->bytes.get(), r.own, cudaMemcpyHostToDevice) != cudaSuccess) {
            // nothing may run on these experts now: the card is out of memory it just had
            basalt::log::error("basalt inkling2", "the experts lent to the image encoder cannot come back to the card");
            std::abort();
        }
        back += r.own;
    }
    const size_t layers = lent_ranges_.size();
    lent_ranges_.clear();
    const int64_t before = copies() ? promoted() : 0;
    if (copies() && !lent_) {
        const int64_t before_total = card_copies();
        promote(held_free(kCardSpare));
        note_shortage("image encoder refill");
        log_copy_change("image encoder refill", before_total);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (layers > 0 || lend_copies_ > 0)
        basalt::log::info("basalt inkling2", "image encoder: borrowed %lld expert copies and %zu layers' experts (%.2f GiB) in %.0f ms, "
                          "back in %.0f ms (%lld copies)",
                          (long long) lend_copies_, layers, (double) back / (double) kGiB, lend_ms_, ms, (long long) (promoted() - before));
    lend_ms_ = 0;
    lend_copies_ = 0;
}

}  // namespace basalt::models::inkling2
