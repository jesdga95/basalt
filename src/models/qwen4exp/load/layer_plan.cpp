#include "basalt/models/qwen4exp/shared_names.hpp"
#include "context.hpp"

namespace basalt::models::qwen4exp::generate {

int layer_plan(Context& context) {
    auto& o = *context.o;
    auto& pack_files = *context.pack_files;
    auto& split_at = *context.split_at;
    auto& split_devs = *context.split_devs;
    auto& multi_gpu = *context.multi_gpu;
    auto& pipe_want = *context.pipe_want;
    auto& kPairMibLater = *context.kPairMibLater;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& K = *context.K;
    auto& skip = *context.skip;

    // ---- layer split across GPUs: the cut follows the shares (model::split_cut) over what each card holds before its experts,
    // its layers' dense weights and session; each card's experts cache fills what is left.
    // THE PROMPT PATH'S BUFFERS ARE BORROWED FROM THE CACHE, NOT WITHHELD BESIDE IT.  With borrowing the cache is sized
    // first and at full size, and the prompt path is laid out in the tail of it (`Prefill::relayout`), so it withholds
    // no VRAM of its own and this reserve is zero.  Only without a profile to fill a cache from do the buffers take a reserve, and then this estimate stands in for buffers that cannot be
    // priced exactly yet because the sessions do not exist.  `plan_lend` uses the exact `Prefill::bytes_needed` as
    // soon as it can.
    const bool pf_borrow = !o.expert_profile.empty();
    // From 1024-token chunks the prompt path also holds a ring of whole expert blobs, which a split without borrowing
    // books here (without it the cards would fill and the draft head no longer fit).
    const int64_t split_ring_mib =
        (multi_gpu && !pf_borrow && o.prefill_chunk >= 1024)
            ? (int64_t) (((uint64_t) basalt::prefill::Prefill::ring_slots_for(o.prefill_chunk) *
                              (uint64_t) basalt::kernels::cpu::expert_layout().max_blob + (1ull << 20) - 1) >> 20)
            : 0;
    const int64_t split_pf_mib =
        (o.prefill_chunk > 0 && !pf_borrow) ? 160 + (o.prefill_chunk * 680) / 1024 + split_ring_mib : 0;
    // ---- WHAT A STAGE RESERVES, AND ON WHICH STAGE.  Every stage after the first books room for its verify windows,
    // the first stage books room for the drafter and its head copy, the last for the head.  All of these allocations are
    // already made before a stage's cache is sized, so what has to be held back here is the windows and, on those
    // stages, the drafter and the head.
    const int64_t kWindowMib = 96;       // the verify windows, with margin
    const int64_t kDrafterMib = 1000;    // the MTP drafter and its copy of the head, on the first stage
    const int64_t kHeadMib = 600;        // the head, on the last stage
    auto stage_room = [&](int dev, bool later, int64_t extra_mib) -> int64_t {
        const basalt::core::OnDevice on(dev);
        size_t fb = 0, tb = 0;
        if (const cudaError_t e = cudaMemGetInfo(&fb, &tb); e != cudaSuccess)
            basalt::log::warn("basalt generate", "layer split: CUDA%d free memory: %s", dev < 0 ? 0 : dev,
                         cudaGetErrorString(e));
        const int64_t reserve = ((int64_t) o.vram_reserve_mib + split_pf_mib + (later ? kWindowMib + (pipe_want ? kPairMibLater : 0) : 0) +
                                 extra_mib) << 20;
        return std::max<int64_t>((int64_t) fb - reserve, 0);
    };
    if (multi_gpu || *context.split_same) {
        // the cut by the shares (or --diag-gpu-cut) over what each card holds before its experts: the shared driver over
        // LayerSplit (model::SplitHooks), which gives split_at and the stages' devices
        const size_t ns = o.gpus.cards.size();
        LayerSplit& split = *context.split;
        LayerSplit::Open in;
        in.pack = &pack_files;
        in.native_shards = &o.native_shards;
        in.skip = &skip;
        in.g = &g;
        in.k = K;
        in.max_context = o.max_context;
        in.room = [&, ns](size_t i) -> uint64_t {
            return (uint64_t) stage_room(i == 0 ? -1 : o.gpus.cards[i].device, i > 0, i == 0 ? kDrafterMib : i + 1 == ns ? kHeadMib : 0);
        };
        in.split_at = &split_at;
        in.split_devs = &split_devs;
        split.set_open(in);
        if (!split.split_open(o.gpus, err)) return fatal_start_error(err);
        const basalt::model::SplitCut& cut = split.split_placement();
        std::string ks, needs;
        for (const int64_t k : split_at) ks += (ks.empty() ? "" : ",") + std::to_string(k);
        for (size_t i = 0; i < ns; ++i) needs += (i ? ", " : "") + basalt::model::gib(cut.need[i]);
        basalt::log::info("basalt generate", "layer split: K=%s %s; dense weights and sessions %s, the rest expert cache", ks.c_str(),
                          o.gpus.diag_same ? "at --diag-gpu-cut, every stage on CUDA0" : "by the shares", needs.c_str());
    }
    // Layer split: every GPU holds only the dense weights of ITS layers (the PLE tensors stay everywhere), the cut being
    // known before any weight is placed.  The VRAM a full copy would take goes back to the expert cache.
    const std::set<std::string> skip_base = skip;
    const bool stage_trim = multi_gpu;
    auto add_foreign = [&](int64_t lb, int64_t le, std::set<std::string>& out) {
        for_each_index_name(pack_files.index, [&](const std::string& n) {
            if (n.rfind("blk.", 0) != 0 || n.find("ple") != std::string::npos) return;
            const int64_t l = std::atoll(n.c_str() + 4);
            if (l < lb || l >= le) out.insert(n);
        });
    };
    if (stage_trim) {
        add_foreign(0, split_at[0], skip);
        basalt::core::NativeDense::set_layer_range(0, (int) split_at[0]);
        basalt::log::info("basalt generate", "layer split: CUDA0 loads the dense weights of layers 0-%lld only",
                     (long long) split_at[0] - 1);
    }
    // One GPU: the hyper-connection down/up matrices are held only as 8-bit copies - the arena leaves their bf16 form
    // out and they are staged, copied and dropped right after the load.  A split keeps their bf16 form on every card, and
    // so does its reference with both stages on CUDA0 (--diag-gpu-same), so that the two read a prompt alike.
    std::set<std::string> hc8_names;
    if (basalt::kernels::hc8_only() && !multi_gpu && !*context.split_same) {
        for_each_index_name(pack_files.index, [&](const std::string& n) {
            const bool blk = n.rfind("blk.", 0) == 0 &&
                             (n.ends_with(".hc_attn_down.weight") || n.ends_with(".hc_attn_up.weight") ||
                              n.ends_with(".hc_ffn_down.weight") || n.ends_with(".hc_ffn_up.weight"));
            if (blk || n == "output_hc_down.weight" || n == "output_hc_up.weight") hc8_names.insert(n);
        });
        for (const std::string& n : hc8_names) skip.insert(n);
    }
    uint64_t pool_bytes = 0;
    if (!basalt::core::WeightTable::pool_bytes(pack_files, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        return fatal_start_error(err);
    }

    context.pf_borrow = &pf_borrow;
    context.kWindowMib = &kWindowMib;
    context.stage_room = stage_room;
    context.skip_base = &skip_base;
    context.stage_trim = &stage_trim;
    context.add_foreign = add_foreign;
    context.hc8_names = &hc8_names;
    context.pool_bytes = &pool_bytes;
    return weights(context);
}

}
