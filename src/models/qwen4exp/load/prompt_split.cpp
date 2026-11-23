#include "basalt/core/prompt_stage.hpp"
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "context.hpp"

#include "basalt/artifact/dequant.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/bf16_bits.hpp"
#include "basalt/models/qwen4exp/kernels/cvec.hpp"

namespace basalt::models::qwen4exp::generate {

int split_plan(Context& context) {
    auto& o = *context.o;
    auto& pack_files = *context.pack_files;
    auto& split_want = *context.split_want;
    auto& main_link_gbps = *context.main_link_gbps;
    auto& g = *context.g;
    auto& skip_base = *context.skip_base;
    auto& kMaxAutoChunk = *context.kMaxAutoChunk;
    auto& err = *context.err;

    // ---- the prompt-time split: what this engine can run it with, and each tier card's link (probed while the cards are
    // empty, for the card facts).  The stages in pipeline order from the last: pstages[0] reads the last layers on the last
    // card of gpu, each earlier card the block before it.  Probed before the RAM copy, so the probes' buffers come and go
    // while the RAM is still free.
    std::vector<std::unique_ptr<PromptStage>> pstages;
    if (split_want) {
        const char* off = o.peer_exclusive ? "the tier's experts have no RAM copy (--peer-exclusive)"
                        : !o.resident_cpu_experts ? "no RAM copy of the experts" : nullptr;
        if (off == nullptr && main_link_gbps <= 0) main_link_gbps = basalt::core::probe_h2d_gbps();   // (a set --pcie-frac skips it)
        for (size_t j = o.peer_devices.size(); off == nullptr && j-- > 0;) {
            auto st = std::make_unique<PromptStage>();
            st->dev = o.peer_devices[j];
            st->tier_at = j;
            const basalt::core::OnDevice on(st->dev);
            st->link_gbps = basalt::core::probe_h2d_gbps();
            if (!basalt::kernels::cvec_replicate(err)) {   // this stage applies the vector on its card
                basalt::log::warn("basalt generate", "prompt split: off (CUDA%d's control vector: %s)", st->dev, err.c_str());
                err.clear();
                off = "the control vector's tables on a tier card";
            }
            if (off == nullptr) pstages.push_back(std::move(st));
        }
        if (off != nullptr) {
            basalt::core::log_prompt_split_off("basalt generate", off);
            split_want = false;
            pstages.clear();
        }
    }
    // the RAM its page-locked buffers take at their most, held before the RAM copy is sized: each stage's weights for
    // the widest window (every layer past the second), the hand-off buffers of the main card and of every stage that
    // hands chunks on for the longest chunk a split reads, the stages' staging rings and their state copies
    if (o.serve && !pstages.empty()) {
        std::vector<uint64_t> pool_l, nat_l;
        uint64_t pool_sh = 0, nat_sh = 0;
        bool ok = basalt::core::WeightTable::pool_bytes_by_layer(pack_files, g.n_layers, pool_l, pool_sh, err, &skip_base) &&
                  basalt::core::NativeDense::plan_bytes_by_layer(o.native_shards, true, g.n_layers, nat_l, nat_sh, err);
        uint64_t widest = pool_sh + nat_sh;
        for (int64_t l = 2; ok && l < g.n_layers; ++l) widest += pool_l[(size_t) l] + nat_l[(size_t) l];
        const int64_t rows = o.prefill_auto_fit ? std::min<int64_t>(o.prefill_chunk, kMaxAutoChunk) : o.prefill_chunk;
        for (size_t j = 0; ok && j < pstages.size(); ++j) {
            ok = basalt::core::host_plan((size_t) widest) && basalt::core::host_plan(256u << 20);
            for (int b = 0; ok && b < 2; ++b) ok = basalt::core::host_plan((size_t) rows * (size_t) (g.hc * g.n_embd) * sizeof(float));
            for (int b = 0; ok && b < 2 * basalt::core::StagedCopy::kBufs; ++b) ok = basalt::core::host_plan(8u << 20);
        }
        if (!ok) {
            basalt::core::log_prompt_split_off("basalt generate", "RAM for its buffers");
            err.clear();
            split_want = false;
            pstages.clear();
        }
    }

    context.pstages = &pstages;
    return resident_load(context);
}

namespace {
// The stage's copies of the matrices the main card holds only as 8-bit copies (hc8_names), widened in its page-locked image
// exactly as hc8_dequant_bf16_kernel widens them: each value the int8 code times its group's fp16 scale (exact in fp32),
// rounded once to bf16, nearest even.
bool hc8_widen_into(PromptStage& st, const uint8_t* w_at, const std::set<std::string>& hc8_names, std::string& e) {
    constexpr uint32_t kGroup = 32;   // values per fp16 scale (hc_fused.cu G8)
    for (const std::string& n : hc8_names) {
        const basalt::core::WeightRef* r = st.wt.find(n);
        if (r == nullptr || r->data == nullptr) continue;   // not one of its layers
        const basalt::core::Hc8Copy* c = basalt::core::hc8_find(n, (int) r->ne1, (int) r->ne0);
        const uint8_t* at = (const uint8_t*) r->data;
        if (c == nullptr || at < w_at || at + (size_t) c->rows * c->cols * 2 > w_at + st.image_bytes) {
            e = "prompt split: no 8-bit copy of " + n + " in the stage's image";
            return false;
        }
        uint16_t* out = (uint16_t*) (st.image + (at - w_at));
        const int8_t* q = c->q();
        const uint16_t* s = c->s();
        for (uint32_t row = 0; row < c->rows; ++row)
            for (uint32_t k = 0; k < c->cols; ++k) {
                const float sc = basalt::fp16_to_fp32(s[(size_t) row * c->srow + k / kGroup]);
                out[(size_t) row * c->cols + k] = basalt::kernels::bf16_from_f32((float) q[(size_t) row * c->cols + k] * sc);
            }
    }
    return true;
}
}  // namespace

int prompt_split(Context& context) {
    auto& o = *context.o;
    auto& pack_files = *context.pack_files;
    auto& split_want = *context.split_want;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& skip_base = *context.skip_base;
    auto& hc8_names = *context.hc8_names;
    auto& add_foreign = context.add_foreign;
    auto& src = *context.src;
    auto& peer = *context.peer;
    auto& peers = *context.peers;
    auto& drive = *context.drive;
    auto& kMaxAutoChunk = *context.kMaxAutoChunk;
    auto& open_peer = context.open_peer;
    auto& pstages = *context.pstages;

    // the split reads the experts its cards lack from the RAM copy: without one it is off
    if (split_want && !src.complement_ready()) {
        basalt::core::log_prompt_split_off("basalt generate", "no RAM copy of the experts");
        split_want = false;
        pstages.clear();
    }
    // ---- the peer's experts: the ranked pairs the primary does not hold.  Without --peer-exclusive they are copies of
    // experts the RAM complement holds, so with the complement built they are read from it, not from the file.
    if (o.peer_device >= 1 && !peer.valid())
        if (const int rc = open_peer()) return rc;
    for (auto& st : pstages) st->tier = &peers.tier(st->tier_at + 1);
    // ---- the prompt-time split: a part of a prompt longer than a chunk is read across the cards at the cuts the shares give
    // (model::stage_cuts).  The split reads a part in the chunks the main card alone would, so the chunk boundaries and the
    // rounding of the K/V there are the same.  The largest chunk a read alone takes now (the loan's chunk follows the slots
    // left: see relend).
    auto read_cap = [&]() -> int64_t {
        return o.prefill_auto_fit ? std::min<int64_t>(o.prefill_chunk, kMaxAutoChunk) : o.prefill_chunk;
    };
    // the cells before a prompt's end the draft layer attends to (all of them without a window: no split then)
    const int64_t draft_window = o.mtp.empty() ? 0 : o.mtp_window > 0 ? o.mtp_window + 64 : INT64_MAX / 4;
    constexpr int64_t kSplitRefCells = 131072;   // the longest prompt the start sizes the stages' regions for
    int64_t split_hand_rows = 0;   // the rows each hand-off buffer holds (0 before they exist: a chunk's)
    int64_t split_chunk_cap = 0;   // the longest chunk every stage can hold (0 before they are laid out: no limit)
    int64_t split_peer_rows = 0;   // the longest routed-only chunk the tier card computes its experts' rows of
    // a part [a0, b0) of a prompt ending at `prompt_end`: the split's chunk and head chunk, or chunk 0 and why it is read on
    // the main card
    auto split_part = [&](int64_t a0, int64_t b0, int64_t prompt_end) -> engine_local::SplitPart {
        engine_local::SplitPart sp0;
        const int64_t n_tok = b0 - a0;
        // two chunks or more, the draft layer's window within the last two the main card hands on (their final rows
        // come back for it in the hand-off buffers)
        const auto [chunk, head] = serve_loans::one_card_chunks(n_tok, read_cap(), true, o.prefill_auto_fit);
        if (n_tok <= std::max(chunk, head)) return sp0;   // one chunk: read on the main card
        // the stages read chunks of at most what their lent slots hold
        const auto [schunk, shead] = split_chunk_cap > 0 && split_chunk_cap < chunk
                                         ? serve_loans::one_card_chunks(n_tok, split_chunk_cap, true, o.prefill_auto_fit)
                                         : std::make_pair(chunk, head);
        char why[160];
        // the routed-only chunks of a CUDA0 read have the tier card compute its experts' rows: those stay on CUDA0
        if (split_peer_rows > 0 && schunk <= split_peer_rows) {
            std::snprintf(why, sizeof why, "chunks of %lld tokens: CUDA0 alone reads them with the tier card computing its "
                          "experts' rows (up to %lld tokens)", (long long) schunk, (long long) split_peer_rows);
            sp0.why = why;
            return sp0;
        }
        if (schunk < 4096) {
            std::snprintf(why, sizeof why, "chunks of %lld tokens, below the 4096 minimum", (long long) schunk);
            sp0.why = why;
            return sp0;
        }
        const std::vector<int64_t> chunks = serve_loans::chunk_list(n_tok, schunk, shead);
        const int64_t c_last = b0 - chunks.back();
        const int64_t draft_rows = draft_window > 0 ? std::max<int64_t>(0, c_last - std::max(a0, prompt_end - draft_window)) : 0;
        // the draft layer's rows before the kept chunk come back in the hand-off buffers: the last two chunks handed on
        // in place, earlier ones past `chunk` rows in either buffer (see read_split)
        const int64_t kept_rows = basalt::prefill::split_draft_room(chunks, schunk, std::max(split_hand_rows, schunk));
        if (draft_rows > kept_rows) {
            if (draft_window >= INT64_MAX / 4)
                std::snprintf(why, sizeof why, "the draft layer attends to every cell");
            else
                std::snprintf(why, sizeof why, "chunks of %lld tokens: the draft layer's window needs %lld rows back, the "
                              "hand-off buffers hold %lld", (long long) schunk, (long long) draft_rows, (long long) kept_rows);
            sp0.why = why;
            return sp0;
        }
        sp0.chunk = schunk;
        sp0.head = shead;
        return sp0;
    };
    // a split's stages for the log: "layers 36-47 on CUDA1", "layers 12-23 on CUDA3, 24-35 on CUDA2, 36-47 on CUDA1"
    auto split_layers_text = [&](int stages, const std::vector<int>& cuts) {
        std::string t = "layers ";
        for (int k = 1; k < stages && (size_t) k <= cuts.size(); ++k) {
            const int l1 = k + 1 < stages ? cuts[(size_t) k] : (int) g.n_layers;
            t += (k > 1 ? ", " : "") + std::to_string(cuts[(size_t) k - 1]) + "-" + std::to_string(l1 - 1) + " on CUDA" +
                 std::to_string(pstages[(size_t) (stages - 1 - k)]->dev);
        }
        return t;
    };
    // the layers each stage reads, fixed for the run: the shared driver's cut of the cards' Stages shares over the layers
    // past the second (PeerPlacement::stage_first); card i of gpu is pstages[n - 1 - i]
    int split_stages = 0;   // the stages a read uses, the main card's included (0: no split)
    std::vector<int> split_cuts;   // stage k (1 .. split_stages - 1) reads [split_cuts[k - 1], split_cuts[k] or the last layer)
    if (split_want && peer.valid() && !pstages.empty() && peers.peer_placement().stage_first.size() == pstages.size() + 1) {
        bool ok = true;
        const int64_t longest = std::min<int64_t>(o.max_context, kSplitRefCells) - 1;
        if (split_part(0, longest, longest).chunk <= 0) {
            basalt::core::log_prompt_split_off("basalt generate", "a " + std::to_string((long long) o.max_context) +
                                                                      "-token context is too short for it");
        } else {
            const std::vector<int>& first = peers.peer_placement().stage_first;
            const size_t n = first.size();
            split_cuts.assign(first.begin() + 1, first.end());
            for (size_t i = 1; i < n; ++i) {
                PromptStage& st = *pstages[n - 1 - i];
                st.lo = first[i];
                st.hi = i + 1 < n ? first[i + 1] : g.n_layers;
            }
            uint64_t images = 0;
            for (size_t j = 0; ok && j < pstages.size(); ++j) {
                PromptStage& st = *pstages[j];
                // its layers' weights: the arena and the native projections
                uint64_t native = 0;
                st.skip = skip_base;
                add_foreign(st.lo, st.hi, st.skip);
                basalt::core::NativeDense::set_layer_range((int) st.lo, (int) st.hi);
                ok = basalt::core::WeightTable::pool_bytes(pack_files, st.pool, err, &st.skip) &&
                     basalt::core::NativeDense::plan_bytes(o.native_shards, true, 0, -1, native, err);
                basalt::core::NativeDense::set_layer_range(-1, -1);
                st.arena_bytes = (st.pool + 255) / 256 * 256;
                st.image_bytes = st.arena_bytes + basalt::core::NativeDense::slab_bytes(native);
                images += st.image_bytes;
            }
            if (!ok) {
                basalt::core::log_prompt_split_off("basalt generate", "its weights' sizes");
                err.clear();
            } else {
                // its pinned RAM (the weights' images, the two hand-off buffers of each card that hands chunks on, the
                // staging rings) takes the RAM held for it before the RAM copy and leaves the reserve free: without the
                // room the split is off
                const uint64_t need = images + (uint64_t) pstages.size() * (256ull << 20) +
                                      2ull * (uint64_t) pstages.size() * (uint64_t) read_cap() * (uint64_t) (g.hc * g.n_embd) *
                                          sizeof(float);
                const uint64_t keep = basalt::platform::ram_reserve();
                const uint64_t avail = basalt::platform::available_memory() + basalt::core::host_plan_bytes();
                if (avail < need + keep) {
                    char why[160];
                    std::snprintf(why, sizeof why, "RAM: %.1f GiB free, it takes %.1f GiB beside %.1f GiB kept",
                                  (double) avail / 1073741824.0, (double) need / 1073741824.0, (double) keep / 1073741824.0);
                    basalt::core::log_prompt_split_off("basalt generate", why);
                } else {
                    split_stages = (int) pstages.size() + 1;
                }
            }
        }
        if (split_stages == 0) pstages.clear();
    }
    // each stage's weights: the page-locked image a split read takes to the card's last slots, laid out for those slots,
    // and filled in the start's pass from the bytes it places on CUDA0 (a range CUDA0 does not hold is read from the file).
    // The images join the pass once every stage has its own.
    std::vector<basalt::core::LoadPlan> images(pstages.size());
    for (size_t j = 0; j < pstages.size() && split_stages > 0; ++j) {
        PromptStage& st = *pstages[j];
        const basalt::core::OnDevice on(st.dev);
        const basalt::core::ExpertCache& pc = st.tier->cache();
        uint8_t* base = const_cast<uint8_t*>(pc.device_slot(0));
        uint8_t* w_at = (uint8_t*) ((uintptr_t) (base + pc.bytes() - st.image_bytes) & ~(uintptr_t) 255);
        const int64_t k = serve_loans::cache_slots_for(pc, (uint64_t) (base + pc.bytes() - w_at));
        bool ok = k > 0 && k < pc.slots() / 2 &&
                  basalt::core::host_alloc((void**) &st.image, st.image_bytes, cudaHostAllocPortable) == cudaSuccess &&
                  st.wt.load(pack_files, w_at, st.pool, err, &st.skip, true, &images[j]);
        if (ok) {
            basalt::core::NativeDense::set_layer_range((int) st.lo, (int) st.hi);
            ok = st.dense.load(o.native_shards, st.wt, err, true, 0, -1, w_at + st.arena_bytes, st.image_bytes - st.arena_bytes,
                               &images[j]);
            basalt::core::NativeDense::set_layer_range(-1, -1);
        }
        if (ok) {
            images[j].to_host(w_at, st.image_bytes, st.image);
            // the main card holds the hyper-connection down and up matrices only as their 8-bit copies and reads the
            // prompt with them widened to bf16 (Gemm::bf16_w): the stage's image takes those widened values in place of
            // the file's bf16, so that its layers read the prompt as the main card would (bitwise)
            if (!hc8_names.empty())
                context.after->push_back([&st, w_at, &hc8_names](std::string& e) { return hc8_widen_into(st, w_at, hc8_names, e); });
            continue;
        }
        (void) cudaGetLastError();
        basalt::core::log_prompt_split_off("basalt generate", "CUDA" + std::to_string(st.dev) + "'s weights: " +
                                                                   (err.empty() ? std::string("no memory") : err), true);
        err.clear();
        for (auto& x : pstages)
            if (x->image != nullptr) cudaFreeHost(x->image);
        pstages.clear();
        split_stages = 0;
    }
    if (split_stages > 0) {
        std::vector<basalt::core::PromptStage> on;
        for (size_t j = 0; j < pstages.size(); ++j) {
            context.reads->append_reusing(images[j]);
            on.push_back({pstages[j]->dev, (int) pstages[j]->lo, (int) pstages[j]->hi, basalt::core::StageHome::PinnedRam,
                          pstages[j]->image_bytes});
        }
        basalt::core::log_prompt_split("basalt generate", on);
    }
    if (split_stages == 0) peers.peer_drop_stages();   // the start line shows no prompt layers
    // ---- the start's one pass: every byte the phases queued lands, then what needs them landed runs
    if (!basalt::core::place_start(*context.loader, *context.reads, err)) return fatal_start_error("reading the model: " + err);
    for (auto& step : *context.after)
        if (!step(err)) return fatal_start_error(err);
    context.after->clear();
    drive.d.peer = peer.valid() ? &peer : nullptr;

    context.read_cap = read_cap;
    context.kSplitRefCells = &kSplitRefCells;
    context.split_hand_rows = &split_hand_rows;
    context.split_chunk_cap = &split_chunk_cap;
    context.split_peer_rows = &split_peer_rows;
    context.split_part = split_part;
    context.split_layers_text = split_layers_text;
    context.split_cuts = &split_cuts;
    context.split_stages = &split_stages;
    return dispatch(context);
}

}
