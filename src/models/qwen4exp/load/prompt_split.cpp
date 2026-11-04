#include "context.hpp"

namespace basalt::program::generate {

int prompt_split(Context& context) {
    auto& o = *context.o;
    auto& chosen_cards = *context.chosen_cards;
    auto& pack_files = *context.pack_files;
    auto& peer_streamed_share = *context.peer_streamed_share;
    auto& split_want = *context.split_want;
    auto& split_cap = *context.split_cap;
    auto& err = *context.err;
    auto& main_link_gbps = *context.main_link_gbps;
    auto& g = *context.g;
    auto& skip_base = *context.skip_base;
    auto& add_foreign = context.add_foreign;
    auto& ss = *context.ss;
    auto& src = *context.src;
    auto& xcache = *context.xcache;
    auto& peer = *context.peer;
    auto& xpeers = *context.xpeers;
    auto& drive = *context.drive;
    auto& host_res = *context.host_res;
    auto& kMaxAutoChunk = *context.kMaxAutoChunk;
    auto& open_peer = context.open_peer;

    // ---- the prompt-time split: what this engine can run it with, and each tier card's link (probed while the cards are
    // empty).  The stages, in the order their cards join a split (prompt_split_order).
    std::vector<std::unique_ptr<PromptStage>> pstages;
    double split_host_gbps = 0;   // the host's bandwidth to the main card and every tier card at once
    if (split_want) {
        const char* off = o.peer_exclusive ? "the tier's experts have no RAM copy (--peer-exclusive)"
                        : !src.complement_ready() ? "no RAM copy of the experts"
                        : !o.cvec_files.empty() ? "a control vector"
                        : !basalt::prefill::Prefill::fp4_experts() ? "priced for NVFP4 experts only" : nullptr;
        if (off == nullptr && main_link_gbps <= 0) main_link_gbps = probe_pcie_h2d_gbps();   // (a set --pcie-frac skips it)
        if (off == nullptr) {
            std::vector<double> bw;
            for (int dev : o.peer_devices) bw.push_back((size_t) dev < chosen_cards.size() ? chosen_cards[(size_t) dev].mem_gbps : 0);
            for (size_t j : basalt::program::prompt_split_order(bw)) {
                auto st = std::make_unique<PromptStage>();
                st->dev = o.peer_devices[j];
                st->tier_at = j;
                st->mem_gbps = bw[j];
                const basalt::core::OnDevice on(st->dev);
                st->link_gbps = probe_pcie_h2d_gbps();
                if (st->link_gbps <= 0) off = "a link probe failed";
                pstages.push_back(std::move(st));
            }
        }
        if (off == nullptr && main_link_gbps <= 0) off = "a link probe failed";
        if (off == nullptr) {
            std::vector<int> devs = {0};
            for (const auto& st : pstages) devs.push_back(st->dev);
            split_host_gbps = probe_h2d_together_gbps(devs);
            if (split_host_gbps <= 0) off = "a link probe failed";
            else
                basalt::log::info("basalt generate", "prompt split: %.1f GB/s host->device with %zu cards copying at once",
                                  split_host_gbps, devs.size());
        }
        if (off != nullptr) {
            basalt::log::info("basalt generate", "prompt split: off (%s)", off);
            split_want = false;
            pstages.clear();
        }
    }
    // ---- the peer's experts: the ranked pairs the primary does not hold.  Without --peer-exclusive they are copies of
    // experts the RAM complement holds, so with the complement built they are read from it, not from the file.
    if (o.peer_device >= 1 && !peer.valid() && !open_peer(false)) return 1;
    for (auto& st : pstages) st->tier = st->tier_at == 0 ? &peer : xpeers[st->tier_at - 1].get();
    // ---- the prompt-time split's price (price_prompt_split): each card's side of a part read, from what its slots hold
    // The largest chunk a read alone takes now (the loan's chunk follows the slots left: see relend).  The split reads a
    // part in the chunks the main card alone would, so the chunk boundaries and the rounding of the K/V there are the same.
    auto read_cap = [&]() -> int64_t {
        return o.prefill_auto_fit ? std::min<int64_t>(o.prefill_chunk, kMaxAutoChunk) : o.prefill_chunk;
    };
    // the cells before a prompt's end the draft layer attends to (all of them without a window: no split then)
    const int64_t draft_window = o.mtp.empty() ? 0 : o.mtp_window > 0 ? o.mtp_window + 64 : INT64_MAX / 4;
    constexpr int64_t kSplitRefCells = 131072;   // the longest prompt the start prices the split for
    // the main card with a loan for chunks of `chunk`: every expert of a layer not in its slots below the loan streams
    auto split_main_side = [&](int64_t chunk) {
        basalt::program::PsCard c;
        c.mem_gbps = chosen_cards.empty() ? 0 : chosen_cards[0].mem_gbps;
        c.link_gbps = main_link_gbps;
        c.stream_bytes.assign((size_t) g.n_layers, 0.0);
        const int64_t first = xcache.slots() - serve_loans::cache_slots_for(xcache, basalt::prefill::Prefill::bytes_needed(g, ss, chunk));
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const int32_t r = host_res.empty() ? -1 : host_res[(size_t) (l * g.n_expert + e)];
                if (r < 0 || r >= first) c.stream_bytes[(size_t) l] += (double) basalt::kernels::cpu::expert_layout().blob_bytes(l);
            }
        return c;
    };
    // a tier card with `region` bytes of its last slots lent to its stage: what its layers do not hold streams to it
    auto split_peer_side = [&](const PromptStage& st, uint64_t region) {
        basalt::program::PsCard c;
        c.mem_gbps = st.mem_gbps;
        c.link_gbps = st.link_gbps;
        c.stream_bytes.assign((size_t) g.n_layers, 0.0);
        const basalt::core::ExpertCache& pc = st.tier->cache();
        const int64_t first = pc.slots() - serve_loans::cache_slots_for(pc, region);
        const std::vector<int32_t>& r = st.tier->residency();
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const int32_t sl = r[(size_t) (l * g.n_expert + e)];
                if (sl < 0 || sl >= first) c.stream_bytes[(size_t) l] += (double) basalt::kernels::cpu::expert_layout().blob_bytes(l);
            }
        return c;
    };
    // a part [a0, b0) of a prompt ending at `prompt_end` over up to `max_stages` stages, each stage with its weights'
    // layers [lo, hi) and image bytes: both prices, and the split's chunk and head chunk
    int64_t split_hand_rows = 0;   // the rows each hand-off buffer holds (0 before they exist: a chunk's)
    int64_t split_peer_rows = 0;   // the longest routed-only chunk a slower tier card computes its experts' rows of
    auto split_price = [&](int64_t a0, int64_t b0, int64_t prompt_end, int max_stages) -> engine_local::SplitPricing {
        engine_local::SplitPricing sp0;
        const int64_t n_tok = b0 - a0;
        // two chunks or more, the draft layer's window within the last two the main card hands on (their final rows
        // come back for it in the hand-off buffers)
        const auto [chunk, head] = serve_loans::one_card_chunks(n_tok, read_cap(), true, o.prefill_auto_fit);
        if (n_tok <= std::max(chunk, head)) return sp0;   // one chunk: read on the main card, no reason to give
        char why[160];
        // a slower tier card that computes its experts' rows of the routed-only chunks of a CUDA0 read: the split
        // gives that share up and the price does not count it, so those chunks stay on CUDA0; any other tier set keeps
        // the 4096-token floor
        if (!peer_streamed_share && split_peer_rows > 0) {
            if (chunk <= split_peer_rows) {
                std::snprintf(why, sizeof why, "chunks of %lld tokens: CUDA0 alone reads them with the tier card "
                              "computing its experts' rows (up to %lld tokens)", (long long) chunk,
                              (long long) split_peer_rows);
                sp0.why = why;
                return sp0;
            }
        } else if (chunk < 4096) {
            std::snprintf(why, sizeof why, "chunks of %lld tokens, below the 4096 minimum", (long long) chunk);
            sp0.why = why;
            return sp0;
        }
        basalt::program::PsPart part;
        part.cell0 = a0;
        part.tier_chunks = serve_loans::chunk_list(n_tok, chunk, head);
        part.split_chunks = part.tier_chunks;
        part.row_bytes = (uint64_t) (g.hc * g.n_embd) * sizeof(float);
        const int64_t c_last = b0 - part.split_chunks.back();
        part.draft_rows = draft_window > 0 ? std::max<int64_t>(0, c_last - std::max(a0, prompt_end - draft_window)) : 0;
        // the draft layer's rows before the kept chunk come back in the hand-off buffers: the last two chunks handed on
        // in place, earlier ones past `chunk` rows in either buffer (see read_split)
        const int64_t kept_rows =
            basalt::program::prompt_split_draft_room(part.split_chunks, chunk, std::max(split_hand_rows, chunk));
        if (part.draft_rows > kept_rows) {
            if (draft_window >= INT64_MAX / 4)
                std::snprintf(why, sizeof why, "the draft layer attends to every cell");
            else
                std::snprintf(why, sizeof why, "chunks of %lld tokens: the draft layer's window needs %lld rows back, the "
                              "hand-off buffers hold %lld", (long long) chunk, (long long) part.draft_rows,
                              (long long) kept_rows);
            sp0.why = why;
            return sp0;
        }
        sp0.chunk = chunk;
        sp0.head = head;
        part.host_gbps = split_host_gbps;
        std::vector<basalt::program::PsCard> peers;
        for (const auto& st : pstages) {
            const uint64_t region = st->image_bytes + basalt::core::prompt_session_bytes(g, b0 + 64, st->lo, st->hi) +
                                    basalt::prefill::Prefill::bytes_needed(g, ss, chunk);
            basalt::program::PsCard c = split_peer_side(*st, region);
            c.setup_ms = st->link_gbps > 0 ? (double) st->image_bytes / (0.85 * st->link_gbps * 1e9) * 1000.0 + 10.0 : 0.0;
            c.layer_lo = (int) st->lo;
            c.layer_hi = (int) st->hi;
            peers.push_back(std::move(c));
        }
        const basalt::program::PsCard main_side = split_main_side(chunk);
        sp0.price = basalt::program::price_prompt_split(main_side, main_side, peers, part, max_stages, (int) g.n_layers);
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
    // the price of every stage count past two, for the log: " (2 stages 9850 ms, 3 stages 8710 ms)"; empty on two cards
    auto split_stages_text = [&](const basalt::program::PsPrice& pr) {
        std::string t;
        for (size_t s = 2; pr.stage_ms.size() > 3 && s < pr.stage_ms.size(); ++s)
            if (pr.stage_ms[s] > 0) {
                char b[64];
                std::snprintf(b, sizeof b, "%s%zu stages %.0f ms", t.empty() ? " (" : ", ", s, pr.stage_ms[s]);
                t += b;
            }
        return t.empty() ? t : t + ")";
    };
    // the windows the start gives the stages: for the longest prompt, the best cuts at every stage count worth it, each
    // card's weights holding the layers those give it and a few more; none when no split is worth reading with
    int split_stages = 0;   // the most stages a read may use, the main card's included (0: no split)
    if (split_want && peer.valid() && !pstages.empty()) {
        std::vector<uint64_t> pool_l, nat_l;
        uint64_t pool_sh = 0, nat_sh = 0;
        bool ok = basalt::core::WeightTable::pool_bytes_by_layer(pack_files, g.n_layers, pool_l, pool_sh, err, &skip_base) &&
                  basalt::core::NativeDense::plan_bytes_by_layer(o.native_shards, true, g.n_layers, nat_l, nat_sh, err);
        auto image_from = [&](int64_t lb) {
            uint64_t b = pool_sh + nat_sh;
            for (int64_t l = lb; l < g.n_layers; ++l) b += pool_l[(size_t) l] + nat_l[(size_t) l];
            return b;
        };
        const int64_t longest = std::min<int64_t>(o.max_context, kSplitRefCells) - 1;
        const int all = (int) pstages.size() + 1;
        for (auto& st : pstages) {   // priced with any cut, each card's weights the later half's
            st->lo = 2;
            st->hi = g.n_layers;
            st->image_bytes = ok ? image_from(g.n_layers / 2) : 0;
        }
        const engine_local::SplitPricing ref_all = ok ? split_price(0, longest, longest, all) : engine_local::SplitPricing{};
        const engine_local::SplitPricing ref = ok && split_cap >= 2 && split_cap < all ? split_price(0, longest, longest, split_cap) : ref_all;
        for (auto& st : pstages) st->lo = st->hi = 0, st->image_bytes = 0;
        if (!ok) {
            basalt::log::info("basalt generate", "prompt split: off (its weights' sizes: %s)", err.c_str());
            err.clear();
        } else if (ref.chunk <= 0) {
            basalt::log::info("basalt generate", "prompt split: off (a %lld-token context is too short for it)",
                              (long long) o.max_context);
        } else if (split_cap < 2) {
            basalt::log::info("basalt generate", "prompt split: off on %zu unlike cards until measured there (a %lld-token "
                                 "prompt: %.0f ms on CUDA0, %.0f ms with %s%s)",
                              pstages.size() + 1, (long long) longest, ref_all.price.tier_ms, ref_all.price.split_ms,
                              split_layers_text(ref_all.price.stages, ref_all.price.cuts).c_str(),
                              split_stages_text(ref_all.price).c_str());
        } else if (!ref.price.split) {
            basalt::log::info("basalt generate", "prompt split: off (a %lld-token prompt: %.0f ms on CUDA0, %.0f ms across%s)",
                              (long long) longest, ref.price.tier_ms, ref.price.split_ms, split_stages_text(ref.price).c_str());
        } else {
            // each card's window, and the stages that have one (a prefix of the join order)
            const std::vector<std::pair<int, int>> win =
                basalt::program::prompt_split_windows(ref.price, pstages.size(), 3, (int) g.n_layers);
            size_t used = 0;
            while (used < pstages.size() && win[used].second > win[used].first) ++used;
            pstages.resize(used);
            uint64_t images = 0;
            for (size_t j = 0; ok && j < pstages.size(); ++j) {
                PromptStage& st = *pstages[j];
                st.lo = win[j].first;
                st.hi = win[j].second;
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
                basalt::log::info("basalt generate", "prompt split: off (its weights' sizes)");
                err.clear();
            } else {
                // its pinned RAM (the weights' images, the two hand-off buffers of each card that hands chunks on, the
                // staging rings) comes out of what the RAM copy left free, never out of the copy: without the room the
                // split is off
                const uint64_t need = images + (uint64_t) pstages.size() * (256ull << 20) +
                                      2ull * (uint64_t) pstages.size() * (uint64_t) read_cap() * (uint64_t) (g.hc * g.n_embd) *
                                          sizeof(float);
                const uint64_t keep = basalt::platform::ram_margin() + basalt::platform::kLateHostBytes +
                                      serve_growth_bytes(o, g);
                const uint64_t avail = basalt::platform::available_memory();
                if (avail < need + keep) {
                    basalt::log::info("basalt generate", "prompt split: off (RAM: %.1f GiB free, it takes %.1f GiB beside "
                                         "%.1f GiB kept)", (double) avail / 1073741824.0, (double) need / 1073741824.0,
                                      (double) keep / 1073741824.0);
                } else {
                    split_stages = std::min((int) pstages.size() + 1, split_cap);
                }
            }
        }
        if (split_stages == 0) pstages.clear();
    }
    // each stage's weights are loaded where a split read places them, in tier slots lent for the load, copied into pinned
    // RAM, and the slots get their experts back: the tiers are as without the split
    for (size_t j = 0; j < pstages.size(); ++j) {
        PromptStage& st = *pstages[j];
        const basalt::core::OnDevice on(st.dev);
        const basalt::core::ExpertCache& pc = st.tier->cache();
        uint8_t* base = const_cast<uint8_t*>(pc.device_slot(0));
        uint8_t* w_at = (uint8_t*) ((uintptr_t) (base + pc.bytes() - st.image_bytes) & ~(uintptr_t) 255);
        const int64_t k = serve_loans::cache_slots_for(pc, (uint64_t) (base + pc.bytes() - w_at));
        bool ok = k > 0 && k < pc.slots() / 2 && st.tier->lend((int32_t) (pc.slots() - k), st.lent, err) &&
                  st.wt.load(pack_files, w_at, st.pool, err, &st.skip, false);
        if (ok) {
            basalt::core::NativeDense::set_layer_range((int) st.lo, (int) st.hi);
            ok = st.dense.load(o.native_shards, st.wt, err, true, 0, -1, w_at + st.arena_bytes,
                               st.image_bytes - st.arena_bytes);
            basalt::core::NativeDense::set_layer_range(-1, -1);
        }
        ok = ok && cudaDeviceSynchronize() == cudaSuccess &&   // (the uploads ran on their own streams)
             cudaHostAlloc((void**) &st.image, st.image_bytes, cudaHostAllocPortable) == cudaSuccess &&
             cudaMemcpy(st.image, w_at, st.image_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        std::string ge;
        if (!(st.tier->give_back(st.lent, ge) && st.tier->give_back_wait(st.lent, ge))) {
            basalt::log::error("basalt generate", "prompt split: refilling the tier: %s", ge.c_str());
            return 1;
        }
        if (!ok) {
            (void) cudaGetLastError();
            basalt::log::warn("basalt generate", "prompt split: off (CUDA%d's weights: %s)", st.dev,
                              err.empty() ? "no memory" : err.c_str());
            err.clear();
            for (auto& x : pstages)
                if (x->image != nullptr) cudaFreeHost(x->image);
            pstages.clear();
            split_stages = 0;
        }
    }
    drive.d.peer = peer.valid() ? &peer : nullptr;

    context.pstages = &pstages;
    context.read_cap = read_cap;
    context.kSplitRefCells = &kSplitRefCells;
    context.split_hand_rows = &split_hand_rows;
    context.split_peer_rows = &split_peer_rows;
    context.split_price = split_price;
    context.split_layers_text = split_layers_text;
    context.split_stages_text = split_stages_text;
    context.split_stages = &split_stages;
    return dispatch(context);
}

}
