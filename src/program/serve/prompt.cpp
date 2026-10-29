// src/program/serve/prompt.cpp - prompt operations for the persistent engine.
#include "request.hpp"

namespace basalt::program::serve {

int64_t Runtime::part_slots(const PfPart& p, int64_t c) {
    return serve_loans::cache_slots_for(*p.cache, basalt::prefill::Prefill::bytes_needed(g, *p.ses, c));
}

uint64_t Runtime::part_bytes(const PfPart& p, int32_t first) {
    basalt::core::ExpertCache& xc = *p.cache;
    return xc.slot_offsets() ? (uint64_t) (xc.bytes() - (int64_t) xc.slot_offsets()[first])
                             : (uint64_t) (xc.slots() - first) *
                                   (uint64_t) basalt::kernels::cpu::expert_layout().max_blob;
}

int Runtime::init_prompt_paths() {   // 0: ready; 1: failed (err set); 2: failed with "do not fit"
    for (size_t i = 0; i < stages.size(); ++i) {
        GpuStage& st = *stages[i];
        st.sp.set_stage(st.lb, i + 1 < stages.size() ? st.le : -1, i + 1 < stages.size() ? &stages[i + 1]->sp : nullptr);
        const basalt::core::OnDevice on(st.dev);
        void* sb = nullptr;              // this stage's own loan, out of its own cache
        uint64_t sbb = 0;
        // `first < 0`: no loan was taken (nothing was lendable), so this stage allocates its own buffers
        if (i + 1 < pf_parts.size() && pf_parts[i + 1].first >= 0) {
            sb = st.cache.device_slot(pf_parts[i + 1].first);
            sbb = part_bytes(pf_parts[i + 1], pf_parts[i + 1].first);
        }
        if (!st.sp.init(st.wt, g, st.ss, srcp, &st.cache, host_res.data(), o.prefill_chunk, (void*) st.stream,
                        err, sb, sbb)) {
            err = "layer split, CUDA" + std::to_string(st.dev) + " prompt path: " + err;
            return err.find("do not fit") != std::string::npos ? 2 : 1;
        }
    }
    if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
    if (!multi_gpu && o.batch <= 0 && !o.no_pool) { sp.set_cpu_pool(&pool); sp.set_cpu_share(o.prefill_cpu_share); }   // the pool is idle while a prompt is read
    if (!sp.init(wt, g, ss, srcp, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes))
        return err.find("do not fit") != std::string::npos ? 2 : 1;
    return 0;
}

bool Runtime::split_region(PromptStage& st, int64_t cells, int64_t chunk, int64_t l0, int64_t l1, uint8_t*& w_at, uint8_t*& s_at, uint8_t*& b_at) {
    const basalt::core::ExpertCache& pc = st.tier->cache();
    uint8_t* base = const_cast<uint8_t*>(pc.device_slot(0));
    auto down = [](uint8_t* p) { return (uint8_t*) ((uintptr_t) p & ~(uintptr_t) 255); };
    w_at = down(base + pc.bytes() - st.image_bytes);
    s_at = down(w_at - basalt::core::prompt_session_bytes(g, cells, l0, l1));
    basalt::core::prompt_session_carve(g, cells, K, s_at, st.ss, l0, l1);
    b_at = down(s_at - basalt::prefill::Prefill::bytes_needed(g, st.ss, chunk));
    // the loan leaves the tier a tenth of its slots, as the main card's
    return b_at >= base + pc.bytes() / 10;
}

bool Request::refill_issue(PfPart& p, std::string& e, bool assign) {
    const basalt::core::OnDevice on(p.dev);
    for (const auto& [i, slot] : p.lent) {   // queued, one wait
        const uint8_t* b = runtime_.srcp->blob(i / runtime_.g.n_expert, i % runtime_.g.n_expert);
        const int64_t nb = (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(i / runtime_.g.n_expert);
        if (b == nullptr || !(p.cache->fill_slot_queued(slot, b, e, nb)))
            return false;
        if (assign) runtime_.host_res[(size_t) i] = slot;
    }
    return true;
}

bool Request::refill_wait(PfPart& p, std::string& e) {
    const basalt::core::OnDevice on(p.dev);
    if (!p.cache->sync_queued(e)) return false;
    p.lent.clear();
    p.lent_chunk = 0;
    return true;
}

bool Request::refill_one(PfPart& p, std::string& e) {
    if (p.lent.empty()) return true;
    if (ov_copied) {
        for (const auto& [i, slot] : p.lent) runtime_.host_res[(size_t) i] = slot;
        p.lent.clear();
        p.lent_chunk = 0;
        ov_copied = false;
        return true;
    }
    return refill_issue(p, e) && refill_wait(p, e);
}

bool Request::refill(std::string& e) {
    const auto t_rf = Clock::now();
    int64_t n_lent = 0, n_parts = 0;
    for (PfPart& p : runtime_.pf_parts)
        if (!p.lent.empty()) { n_lent += (int64_t) p.lent.size(); ++n_parts; }
    if (n_parts > 1) {
        for (PfPart& p : runtime_.pf_parts)
            if (!p.lent.empty() && !refill_issue(p, e)) return false;
        for (PfPart& p : runtime_.pf_parts)
            if (!p.lent.empty() && !refill_wait(p, e)) return false;
    } else {
        for (PfPart& p : runtime_.pf_parts)
            if (!p.lent.empty() && !refill_one(p, e)) return false;
    }
    if (n_parts > 0) runtime_.res_upload();
    ov_copied = false;
    return true;
}

bool Request::lend(int64_t tokens, std::string& e, int64_t chunk, int64_t head) {
    if (runtime_.pf_parts.empty()) return true;                     // its own buffers: nothing to lend
    const auto t_ln = Clock::now();
    // the ring depth this read's layout uses; every size computed below (and the layout) follows it, and
    // the full depth is restored on the way out for the sizing done elsewhere
    const bool beside = runtime_.batch_on() && (tokens > runtime_.kReadAlone || resumed_chunk > 0);
    basalt::prefill::Prefill::set_ring_depth(beside ? runtime_.kRingDecoding : 1 << 20);
    struct RingReset { ~RingReset() { basalt::prefill::Prefill::set_ring_depth(1 << 20); } } ring_reset;
    // what this segment needs, capped by the configured chunk: a request lends only what its own
    // prompt needs, so a large chunk costs a short prompt nothing
    // A bare `auto` reads a prompt in equal chunks of at most this many tokens: past a few thousand tokens
    // a chunk's cost per token stops falling (its experts stream once whatever its size) and then rises
    // with the chunk, so the rule is the fewest equal parts of at most this size, never a small tail.
    // `auto:N` and a fixed N keep the operator's number.  (Chunking changes the GEMM tiles and so the rounding of
    // a prompt's K/V; the answers can differ from another chunking by greedy near-ties.)
    int64_t cap = runtime_.read_cap();
    // Loan limits while slots decode: the loan takes only slots whose expert has a RAM copy (a lent slot
    // is refilled from RAM, never read from the file, before the slots decode again), and a read beside
    // them stays within kLendCapDecoding slots (they keep their experts in the cache).  A read on an idle
    // engine may refill lent slots from the file, so a larger chunk is worth it.  The chunk shrinks to the
    // loan, but not below the smallest chunk the prompt path runs.
    int64_t loan_lim = std::numeric_limits<int64_t>::max();
    if (runtime_.batch_on()) {
        if (beside) loan_lim = runtime_.kLendCapDecoding;
        // the RAM-backed slots are the cache's last ones: those still live after the KV took its share
        if (runtime_.src.complement_ready())
            loan_lim = std::min<int64_t>(loan_lim, std::max<int64_t>(0, runtime_.src.resident_lent_slots() -
                                                                         (runtime_.xcache.full_slots() - runtime_.xcache.slots())));
    }
    if (loan_lim != std::numeric_limits<int64_t>::max() && !runtime_.pf_parts.empty() && runtime_.pf_parts[0].first >= 0) {
        const int64_t lc = std::max<int64_t>(256, engine_local::biggest_chunk(cap, [&](int64_t t) {
            return runtime_.part_slots(runtime_.pf_parts[0], t) <= loan_lim;
        }));
        cap = std::min(cap, lc);
    }
    auto [want_full, head_tok] = serve_loans::one_card_chunks(tokens, cap, !beside && runtime_.stages.empty(),
                                                           runtime_.o.prefill_auto_fit);
    // a chunk the read must have: a split read's, or the one the read this continues had (if it fits)
    const int64_t fixed_chunk = chunk > 0 ? chunk : (resumed_chunk > 0 && resumed_chunk <= cap) ? resumed_chunk : 0;
    if (fixed_chunk > 0) {
        want_full = fixed_chunk;
        head_tok = chunk > 0 ? head : 0;
    }
    runtime_.sp.set_head_chunk(head_tok);
    const int64_t want = want_full;
    if (want <= 0) {
        e = "prefill: cannot lend buffers for an empty request segment";
        return false;
    }
    bool any = false;
    const int ring_now = (int) basalt::prefill::Prefill::ring_slots_for(want);
    for (PfPart& p : runtime_.pf_parts) {
        if (p.first < 0) continue;
        const int32_t first = std::max<int32_t>(p.first, (int32_t) (p.cache->slots() - runtime_.part_slots(p, want)));
        std::vector<serve_loans::LentEntry> again;   // entries whose slots the prompt path still holds
        if (!p.lent.empty()) {
            if (want <= p.lent_chunk && p.sp->ring_depth() == ring_now && !ov_copied &&
                (fixed_chunk <= 0 || want == p.sp->chunk()))
                continue;   // its loan covers this (a fixed chunk must be the layout's)
            // the slots the new loan takes again are not refilled first (their experts would be
            // overwritten by the new layout): only their table entries go back, and the loop below
            // lends them again; the new loan's refill brings their experts back after the prompt
            if (!ov_copied) {
                serve_loans::LoanSplit split = serve_loans::split_loan(p.lent, first);
                for (const auto& [i, slot] : split.again) runtime_.host_res[(size_t) i] = slot;
                again = std::move(split.again);
                p.lent = std::move(split.back);
            }
            // ONLY this participant's loan goes back: `refill` would return the other participants'
            // loans too, and their buffers are still laid out in their caches - marking those slots
            // resident again would hand the next window a prompt buffer in place of an expert
            if (!refill_one(p, e)) return false;
        }
        const basalt::core::OnDevice on(p.dev);
        if (beside && &p == &runtime_.pf_parts[0] && p.lent.empty()) {
            if (runtime_.ba_job) { runtime_.ba_thr.join(); runtime_.ba_job = false; }   // no adaptive copy lands in a slot it moves
            if (!runtime_.xq_drain()) { e = "an adaptive refill failed"; return false; }
            if (!runtime_.move_routed_below(first, "prompt loan", e, &again)) return false;
            any = true;
        }
        if (want != p.sp->chunk() || first != p.first_now || p.sp->ring_depth() != ring_now) {
            if (!p.sp->relayout(want, p.cache->device_slot(first), runtime_.part_bytes(p, first), e)) return false;
            p.first_now = first;
        }
        for (int64_t l = p.lb; l < p.le; ++l)              // THIS participant's layers only
            for (int64_t ex = 0; ex < runtime_.g.n_expert; ++ex) {
                const size_t i = (size_t) (l * runtime_.g.n_expert + ex);
                if (runtime_.host_res[i] >= first) {
                    p.lent.emplace_back((int32_t) i, runtime_.host_res[i]);
                    runtime_.host_res[i] = basalt::core::kNotResident;
                    any = true;
                }
            }
        p.lent_chunk = want;
    }
    if (any) runtime_.res_upload();
    return true;
}

bool Request::can_yield(int64_t q, int64_t b0, int ys, bool beside) {
    return ys >= 0 && ys < (int) runtime_.bs.size() && !runtime_.bs[(size_t) ys].active &&
           n + (runtime_.admit_slot >= 0 ? runtime_.admit_max_new : (long long) max_new) + 8 <= runtime_.slot_cap[(size_t) ys] &&
           (runtime_.admit_slot < 0 || ys == runtime_.admit_slot) && runtime_.req_imgs.empty() && runtime_.o.prompt_cache > 0 &&
           (!in_slot || ys == slot_b) &&
           b0 - q > (beside ? runtime_.o.short_read : std::max<int64_t>(std::max<int64_t>(runtime_.sp.chunk(), 1), runtime_.o.short_read));
}

bool Request::park_read(int64_t q, int ys, std::string& ye, int64_t chunk) {
    std::vector<int32_t> pre(ids.begin(), ids.begin() + q);
    if (!runtime_.copy_to_slot(ys, pre, ye)) {
        runtime_.bs[(size_t) ys].cached = false;   // half written
        return false;
    }
    if (in_slot) {   // the read ran in this slot's own state: its PLE token window goes with it
        basalt::core::SessionState& t = *runtime_.bslot_ss[0][(size_t) ys];
        t.ple_prev[0] = (*rview).ple_prev[0];
        t.ple_prev[1] = (*rview).ple_prev[1];
        t.ple_token = (*rview).ple_token;
    }
    BSlot& sl = runtime_.bs[(size_t) ys];
    sl = BSlot{};
    sl.ids = std::move(pre);
    sl.cached = true;
    sl.partial = true;
    sl.partial_from0 = read_from == 0 || resumed_from0;
    sl.partial_chunk = chunk;
    sl.cvec = runtime_.cvec_cached;
    for (const ConvCheckpoint& c : runtime_.checks)   // the root / periodic checkpoints of this read
        if ((int64_t) c.ids.size() <= q &&
            basalt::core::conversation_ids_equal(c.ids.data(), c.ids.size(), ids32.data()))
            sl.checks.push_back(c);
    yielded_at = q;
    yielded_slot = ys;
    return true;
}

int Runtime::init_prompt() {
    if ((!no_spec && (o.spec < 2 || o.mtp.empty())) || o.prefill_chunk <= 0 ||
        (graph_hits && (thits.d_res == nullptr || host_res.empty()))) {
        basalt::log::error("basalt serve", "needs --spec T (0 for none; T >= 2 also needs the model's draft layer) "
                             "and --prefill CHUNK (and a fillable "
                             "--expert-cache; the graphed hit path additionally needs an expert profile)");
        return 2;
    }

    borrow = nullptr;
    borrow_bytes = 0;
    lend_first = -1;          // the first slot the prompt path may borrow (its largest chunk)
    // ---- WHO BORROWS, AND FROM WHOSE CACHE.  One entry per prompt path: CUDA0's (layers [0, split_at[0]), which is
    // the whole model without a split) borrowing the tail of CUDA0's cache, then one per stage borrowing the tail
    // of ITS OWN cache.  A loan is sized by the exact `Prefill::bytes_needed` for the chunk, is laid out by
    // `Prefill::relayout`, and is refilled before any window reads - so outside the prompt the whole cache is
    // expert cache.  A stage may only lend the rows for ITS OWN layers: `host_res` is one table whose slot values
    // are indices into whichever cache owns the layer, so a loan that marked rows by slot number alone would hand
    // CUDA0 a slot belonging to another stage's cache.

    // A prompt read beside decoding slots lends at most this many cache slots, so their experts stay in the
    // cache, and its streamed ring is this deep (the loan shrinks with it); otherwise the ring is full depth.

    // A read of at most this many tokens runs alone even while slots decode: they wait for it (a few seconds at
    // most) and it lends and reads as on an idle engine.  A longer read goes beside them, within the limits above,
    // and so does the rest of one that gave way to a shorter request (in its chunk).

    // a cache too small to lend the prompt path its buffers would make it allocate them on top - on a card
    // whose cache already filled its reserve, that is the over-subscription the auto sizing avoids - so the
    // chunk is the largest one EVERY participant can lend (a smaller chunk only reads slower)
    if (pf_borrow && d_res != nullptr) {
        pf_parts.push_back({&xcache, &ss, &sp, -1, 0, multi_gpu ? split_at[0] : g.n_layers, -1, -1, 0, {}});
        for (auto& st : stages)
            pf_parts.push_back({&st->cache, &st->ss, &st->sp, st->dev, st->lb, st->le, -1, -1, 0, {}});
        // The two tests plan_lend makes for CUDA0 alone, one participant at a time: a loan must leave the
        // 128-slot floor.  The percentage cap is an AUTO-chunk rule and only the auto scan applies it - an
        // explicit --prefill is the operator's number, and a loan of it only has to fit.  With one participant
        // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
        auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
            const int64_t k = part_slots(p, c);
            if (k <= 0 || k + 128 > p.cache->slots()) return false;
            return !(cap && k * 100 > auto_lend_pct * p.cache->slots());
        };
        // `only`: CUDA0's cache alone (what one GPU would choose, for the log below); null: every one
        auto fits = [&](int64_t c, bool cap, const PfPart* only = nullptr) -> bool {
            if (only != nullptr) return fits_one(*only, c, cap);
            for (const PfPart& p : pf_parts)
                if (!fits_one(p, c, cap)) return false;
            return true;
        };
        auto scan = [&](const PfPart* only) -> int64_t {
            if (!o.prefill_auto) {   // an explicit --prefill is the operator's number, and a loan of it only has to fit
                for (int64_t c = o.prefill_chunk; c >= 256; c /= 2)
                    if (fits(c, false, only)) return c;
                return 0;
            }
            // the largest chunk on the 256-token grid that `only` - or, with null, every stage - can lend, under
            // the operator's ceiling (auto_ceiling keeps a bare `auto` at 8192)
            return engine_local::biggest_chunk(auto_ceiling, [&](int64_t t) -> bool { return fits(t, true, only); });
        };
        const int64_t chunk = scan(nullptr);
        // A small card in a layer split caps every stage's chunk.  Named when it bites, so the regression is one
        // log line: each stage that cannot fund the chunk CUDA0 alone would read in.
        const int64_t alone = pf_parts.size() > 1 ? scan(&pf_parts[0]) : chunk;
        if (chunk > 0 && alone > chunk) {
            for (size_t i = 1; i < pf_parts.size(); ++i) {
                const PfPart& p = pf_parts[i];
                if (fits_one(p, alone, o.prefill_auto)) continue;
                const int dev = p.dev < 0 ? 0 : p.dev;
                cudaDeviceProp prop{};
                if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
                    (void) cudaGetLastError();
                    prop.name[0] = 0;
                }
                const std::string pct =
                    o.prefill_auto ? ", and lend at most " + std::to_string(auto_lend_pct) + "%" : "";
                basalt::log::warn("basalt serve", "prompt chunk %lld tokens, not %lld: CUDA%d (%s) "
                                     "has %lld expert-cache slots, and a %lld-token chunk borrows %lld of them "
                                     "(it must keep 128%s) - prompts read slower than on CUDA0 alone",
                             (long long) chunk, (long long) alone, dev, prop.name,
                             (long long) p.cache->slots(), (long long) alone,
                             (long long) part_slots(p, alone), pct.c_str());
                // the helper tiers start at CUDA1 without a split (and are enabled in order)
                basalt::log::info("basalt serve", "  a card this small can serve as a helper expert cache "
                                     "instead of a split stage: without --layer-split, with %s",
                             dev == 1 ? "--expert-cache-device1 N" : "--expert-cache-device1..3 N, in order");
            }
        }
        if (chunk > 0) {
            if (o.prefill_auto)
                // the RESOLVED ring: a chunk under stream_all_min() gets the staging slots and those for copies
                // ahead of its routing.  This is what `init` lays out, and what the INFO line reports to the Monitor tab.
                basalt::log::info("basalt serve", "prompt chunk auto: %lld tokens, a %lld-slot ring",
                             (long long) chunk,
                             (long long) basalt::prefill::Prefill::ring_slots_for(chunk));
            else if (chunk != o.prefill_chunk)
                basalt::log::info("basalt serve", "prompt chunk %lld -> %lld tokens so its buffers fit in "
                                     "every expert cache", (long long) o.prefill_chunk, (long long) chunk);
            o.prefill_chunk = chunk;
            for (PfPart& p : pf_parts) {
                p.first = (int32_t) (p.cache->slots() - part_slots(p, chunk));
                p.first_now = p.first;
            }
            // a split stage whose card still has room for the chunk's buffers (its cache already holds every
            // expert of its layers, so auto stopped short of its VRAM) keeps them as its own instead of borrowing
            // cache slots: nothing of it is lent, streamed during the prompt or refilled after it.  Room = the
            // buffers + 1.5 GiB (the verify windows, the draft head, cuBLAS workspaces made after this).  Layer
            // splits only - one GPU keeps its loan.
            if (pf_parts.size() > 1) {
                for (PfPart& p : pf_parts) {
                    const basalt::core::OnDevice on(p.dev);
                    size_t fb = 0, tb = 0;
                    if (cudaMemGetInfo(&fb, &tb) != cudaSuccess) { (void) cudaGetLastError(); continue; }
                    const uint64_t need = basalt::prefill::Prefill::bytes_needed(g, *p.ses, chunk);
                    if ((uint64_t) fb >= need + (3ull << 29)) {
                        basalt::log::info("basalt serve", "  CUDA%d keeps its own prompt buffers (%.2f GiB of "
                                             "%.2f GiB free): no loan", p.dev < 0 ? 0 : p.dev,
                                     (double) need / 1073741824.0, (double) fb / 1073741824.0);
                        p.first = -1;
                        p.first_now = -1;
                    }
                }
            }
            lend_first = pf_parts[0].first;
            borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
            borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
        } else if (o.prefill_auto) {
            o.prefill_chunk = 1024;   // nothing lendable: small buffers of its own
        } else if (pf_parts.size() > 1) {
            // An explicit chunk no stage can lend in full.  main falls back to the prompt path's own buffers
            // here and so do we, rather than refusing to start - but say what every stage has, because a split
            // stage that has to allocate these on top of a cache that already filled its VRAM will not fit,
            // and `init` would otherwise report only that the buffers do not fit.
            basalt::log::info("basalt serve", "no stage can lend the prompt path its %lld-token buffers, so "
                                 "each stage allocates its own:", (long long) o.prefill_chunk);
            for (const PfPart& p : pf_parts)
                basalt::log::info("basalt serve", "  CUDA%d has %lld slots, and a %lld-token chunk needs "
                                     "the last %lld of them", p.dev < 0 ? 0 : p.dev,
                             (long long) p.cache->slots(), (long long) o.prefill_chunk,
                             (long long) part_slots(p, o.prefill_chunk));
        }
    } else if (o.prefill_auto && d_res == nullptr) {
        o.prefill_chunk = 1024;   // no expert cache at all: small buffers of its own
    }
    any_loan = borrow != nullptr;
    for (size_t i = 1; i < pf_parts.size(); ++i) any_loan = any_loan || pf_parts[i].first >= 0;
    if (any_loan) {
        if (borrow != nullptr)
            basalt::log::info("basalt serve", "the prompt path borrows %lld CUDA0 cache slots (%.2f GiB)",
                         (long long) (xcache.slots() - lend_first), (double) borrow_bytes / 1073741824.0);
        for (size_t i = 1; i < pf_parts.size(); ++i)   // one loan per stage, from that stage's own cache
            if (pf_parts[i].first >= 0) basalt::log::info("basalt serve", "  CUDA%d prompt path borrows %lld of its %lld slots (%.2f GiB)",
                         pf_parts[i].dev, (long long) (pf_parts[i].cache->slots() - pf_parts[i].first),
                         (long long) pf_parts[i].cache->slots(),
                         (double) part_bytes(pf_parts[i], pf_parts[i].first) / 1073741824.0);
    } else {
        basalt::log::info("basalt serve", "the prompt path allocates its own buffers (too few cache slots to borrow)");
    }
    // layer split across GPUs: a prompt path per stage, each handing its chunk's rows to the next.
    // THE CHUNK STEPS DOWN INSTEAD OF EXITING.  A split's stage caches are sized after the arena is registered, and
    // the prompt path's own buffers (no loan) are not priced into them: with the whole arena pinned a
    // `--prefill auto` split could stop at start with "device buffers for a chunk of 2048 tokens do not fit".  A
    // chunk that does not fit is tried again one size smaller, down to 512 tokens (a smaller chunk only reads slower).

    {
        // First by arithmetic: a prompt path without a loan allocates its buffers, so the chunk must leave headroom
        // on that device (a chunk that fits to the last MiB leaves the GEMM library nothing: its GEMMs would then
        // fail to launch).  `bytes_needed` is the same count `init` makes.
        const int64_t kHeadroom = 512ll << 20;
        auto own_fits = [&](int64_t c, int& dev_out, int64_t& need_out, int64_t& free_out) -> bool {
            for (size_t i = 0; i <= stages.size(); ++i) {
                const bool loan = i == 0 ? borrow != nullptr : (i < pf_parts.size() && pf_parts[i].first >= 0);
                if (loan) continue;
                const int dev = i == 0 ? -1 : stages[i - 1]->dev;
                const basalt::core::OnDevice on(dev);
                size_t fb = 0, tb = 0;
                cudaMemGetInfo(&fb, &tb);
                const int64_t need = (int64_t) basalt::prefill::Prefill::bytes_needed(g, i == 0 ? ss : stages[i - 1]->ss, c);
                if (need + kHeadroom > (int64_t) fb) {
                    dev_out = dev < 0 ? 0 : dev; need_out = need; free_out = (int64_t) fb;
                    return false;
                }
            }
            return true;
        };
        int dev = 0;
        int64_t need = 0, fb = 0;
        if (!own_fits(o.prefill_chunk, dev, need, fb)) {
            int64_t c = 0;
            for (const int64_t s : kStepChunks) {
                int d2 = 0;
                int64_t n2 = 0, f2 = 0;
                if (s < o.prefill_chunk && own_fits(s, d2, n2, f2)) { c = s; break; }
            }
            if (c > 0) {
                basalt::log::warn("basalt serve", "a %lld-token chunk's prompt buffers need %lld MiB on CUDA%d, "
                                     "%lld MiB free: %lld-token chunks", (long long) o.prefill_chunk,
                             (long long) (need >> 20), dev, (long long) (fb >> 20), (long long) c);
                o.prefill_chunk = c;
            }
        }
    }
    for (;;) {
        const int r = init_prompt_paths();
        if (r == 0) break;
        int64_t next = 0;
        for (const int64_t c : kStepChunks)
            if (c < o.prefill_chunk) { next = c; break; }
        if (r == 2 && next > 0) {
            basalt::log::warn("basalt serve", "%s: trying a %lld-token chunk", err.c_str(), (long long) next);
            // start the prompt paths over (a failed init may hold buffers), and consume the failed allocation's
            // error: it is sticky, and the next launch check would report "out of memory" for a kernel
            for (auto& stp : stages) {
                const basalt::core::OnDevice on(stp->dev);
                stp->sp.reset();
                (void) cudaGetLastError();
            }
            sp.reset();
            (void) cudaGetLastError();
            o.prefill_chunk = next;
            if (any_loan) {                  // smaller loans for the smaller chunk
                for (PfPart& p : pf_parts)
                    if (p.first >= 0) {
                        p.first = (int32_t) (p.cache->slots() - part_slots(p, next));
                        p.first_now = p.first;
                    }
                lend_first = pf_parts[0].first;
                borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
                borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
            }
            err.clear();
            continue;
        }
        basalt::log::error("basalt serve", "%s", err.c_str());
        if (err.find("fit") != std::string::npos)   // say what frees VRAM
            basalt::log::error("basalt serve", "the GPU has too little free VRAM for the prompt path: move images "
                                 "to the CPU (--vision cpu) or off, close other programs using the GPU, use a shorter "
                                 "context, or read prompts in smaller chunks (--prefill 512)");
        return 1;
    }
    if (peer.valid() && o.peer_exclusive) sp.set_ring_peer(&peer);
    if (peer.valid() && o.peer_prefill_rows != 0) {
        const int64_t rows = o.peer_prefill_rows > 0 ? o.peer_prefill_rows : o.prefill_chunk * K / 2;
        if (!sp.set_peer(&peer, rows, peer_streamed_share, err)) {
            basalt::log::warn("basalt serve", "%s - the prompt path stays on the primary GPU", err.c_str());
            err.clear();
        }
        split_peer_rows = sp.peer_routed_rows_max();
    }
    // ---- the prompt-time split's stages on the tier cards.  A stage's place, from the end of its tier's slots: the
    // weights, the session (for the whole context here; for its prompt's cells and layers at each read), then the
    // prompt buffers.  The layout is made over live slots and writes nothing until a read lends them.

    // its buffers' and the hand-off buffers' size; a larger chunk reads on CUDA0
    split_chunk_max = read_cap();
    split_ready = split_stages >= 2 && !pstages.empty();
    for (size_t j = 0; split_ready && j < pstages.size(); ++j) {
        PromptStage& st = *pstages[j];
        const basalt::core::OnDevice on(st.dev);
        size_t fb0 = 0, tb0 = 0;
        cudaMemGetInfo(&fb0, &tb0);
        uint8_t *w_at = nullptr, *s_at = nullptr, *b_at = nullptr;
        // (a read of the longest prompt the start priced must fit; a longer one checks its own fit)
        bool ok = split_chunk_max >= 1024 &&
                  split_region(st, std::min<int64_t>(o.max_context, kSplitRefCells), split_chunk_max, st.lo, st.hi,
                               w_at, s_at, b_at);
        ok = ok && cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking) == cudaSuccess &&
             st.state_in.init(0, st.dev, 8u << 20, err) && st.state_back.init(st.dev, 0, 8u << 20, err);
        if (ok) {
            st.res.assign((size_t) (g.n_layers * g.n_expert), basalt::core::kNotResident);
            st.sp.set_stage(st.lo, -1, nullptr);
            ok = st.sp.init(st.wt, g, st.ss, srcp, &st.tier->cache(), st.res.data(), split_chunk_max,
                            (void*) st.stream, err, b_at, (uint64_t) (s_at - b_at));
        }
        // a stage that may hand its chunks on (every card but the one that reads the last layers) holds two pinned
        // buffers of the stage's chunk, as the main card does
        ok = ok && (j == 0 || st.sp.reserve_hand_off(split_chunk_max, err));
        // a read's kernels and small buffers load on the card outside the lent slots: a card without room for them
        // beside its stage declines here, before a read could fail on them
        constexpr size_t kStageFreeMin = 256ull << 20;
        size_t fb = 0, tb = 0;
        if (ok && cudaMemGetInfo(&fb, &tb) == cudaSuccess && fb < kStageFreeMin) {
            ok = false;
            err = std::to_string(fb >> 20) + " MiB free beside its stage, a read needs " +
                  std::to_string(kStageFreeMin >> 20) + "; --peer-reserve-mib";
        }
        if (ok) {
            st.sp.set_ring_remote(host_res.data(), &xcache, 0);
            basalt::log::info("basalt serve", "prompt split: CUDA%d can take layers %lld-%lld of a long prompt (chunks up "
                                 "to %lld tokens, links %.1f/%.1f GB/s); %zu MiB free on it (%zu before)", st.dev,
                              (long long) st.lo, (long long) st.hi - 1, (long long) split_chunk_max,
                              main_link_gbps, st.link_gbps, fb >> 20, fb0 >> 20);
        } else {
            basalt::log::warn("basalt serve", "prompt split: off (CUDA%d: %s)", st.dev,
                              err.empty() ? "too few tier slots" : err.c_str());
            err.clear();
            (void) cudaGetLastError();
            split_ready = false;
        }
    }
    // the main card hands its chunks on through two pinned buffers of the stage's chunk
    if (split_ready) split_hand_rows = split_chunk_max;
    if (split_ready && !sp.reserve_hand_off(split_chunk_max, err)) {
        basalt::log::warn("basalt serve", "prompt split: off (%s)", err.c_str());
        err.clear();
        split_ready = false;
    }
       // the routed entries the stages counted in the last split read
    split_hist_tokens = 0;
    if (split_ready && pstages.size() >= 2)
        basalt::log::info("basalt serve", "prompt split: up to %d stages over CUDA0 and %zu tier cards", split_stages,
                          pstages.size());    return 0;
}

}  // namespace basalt::program::serve
