#include "context.hpp"

namespace basalt::program::generate {

int prefill(Context& context) {
    auto& o = *context.o;
    auto& multi_gpu = *context.multi_gpu;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& wt = *context.wt;
    auto& ss = *context.ss;
    auto& main_cs = *context.main_cs;
    auto& mtp = *context.mtp;
    auto& native_head = *context.native_head;
    auto& srcp = *context.srcp;
    auto& xcache = *context.xcache;
    auto& prefill_ms = *context.prefill_ms;
    auto& n_prompt = *context.n_prompt;
    auto& tok = *context.tok;
    auto& host_res = *context.host_res;
    auto& res_put = context.res_put;
    auto& d_res = *context.d_res;
    auto& auto_lend_pct = *context.auto_lend_pct;
    auto& auto_ceiling = *context.auto_ceiling;
    auto& pool = *context.pool;

    // ---- the prompt's conditioning positions [0, n_prompt - 1) in batched chunks.  The token loop then starts at the
    // last prompt position, whose prediction is the first generated token.
    int64_t pos_start = 0;
    basalt::prefill::Prefill prefill;
    double prefill_batched_ms = 0;
    std::FILE* final_r = o.dump_final_r.empty() ? nullptr : std::fopen(o.dump_final_r.c_str(), "wb");
    std::vector<float> final_r_host(final_r ? (size_t) (g.hc * g.n_embd) : 0);
    std::vector<std::pair<int32_t, int32_t>> lent;     // (residency index, slot) lent to the prompt path
    const int64_t n_batched = (o.prefill_until > 0 && o.prefill_until < n_prompt - 1) ? o.prefill_until : n_prompt - 1;
    if (o.prefill_chunk > 0 && n_prompt > 1) {
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        if (!host_res.empty() && d_res != nullptr) {
            int64_t chunk = o.prefill_chunk;
            int64_t k = serve_loans::plan_lend(chunk, xcache, g, ss, o.prefill_auto, auto_ceiling,
                                               (int) auto_lend_pct);   // auto: the largest that fits; fixed: halved
            const int64_t request_sized = serve_loans::request_chunk(n_batched, chunk);
            if (k > 0 && request_sized < chunk) {                     // no bigger than this prompt segment needs
                chunk = request_sized;
                k = serve_loans::lend_slots(xcache, g, ss, chunk);
                if (k + 128 > xcache.slots()) k = 0;
                if (!o.prefill_auto) o.prefill_chunk = chunk;
            }
            if (o.prefill_auto) {
                o.prefill_chunk = k > 0 ? chunk : serve_loans::request_chunk(n_batched, 1024);
                basalt::log::info("basalt generate", "prompt chunk auto: %lld tokens", (long long) o.prefill_chunk);
            } else if (chunk != o.prefill_chunk) {
                k = 0;                                 // a fixed chunk that does not fit: its own buffers, as before
            }
            const int64_t blob = (int64_t) basalt::kernels::cpu::expert_layout().max_blob;
            if (k > 0) {   // the lent slots are refilled after the prompt
                const int32_t first = (int32_t) (xcache.slots() - k);
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= first) {
                        lent.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = basalt::core::kNotResident;
                    }
                res_put(d_res);
                borrow = xcache.device_slot(first);
                borrow_bytes = xcache.slot_offsets() ? (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[first])
                                                     : (uint64_t) k * (uint64_t) blob;
                basalt::log::info("basalt generate", "prompt path borrows %lld cache slots (%.2f GiB)", (long long) k,
                             (double) borrow_bytes / 1073741824.0);
            }
        }
        if (borrow == nullptr)
            basalt::log::info("basalt generate", "prompt path allocates its own buffers (no cache slots to borrow)");
        if (!multi_gpu && o.batch <= 0 && !o.no_pool) { prefill.set_cpu_pool(&pool); prefill.set_cpu_share(o.prefill_cpu_share); }
        if (!prefill.init(wt, g, ss, srcp, o.expert_cache > 0 ? &xcache : nullptr,
                          host_res.empty() ? nullptr : host_res.data(), o.prefill_chunk, main_cs, err, borrow,
                          borrow_bytes)) {
            return fatal_start_error(err);
        }
        if (!o.mtp.empty()) {
            if (!mtp.bind(wt, &native_head, nullptr, err)) {
                return fatal_start_error(err);
            }
            prefill.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
                // cell i pairs R_i with the token at i + 1 (every such token is in the prompt)
                std::vector<int32_t> nxt((size_t) T);
                for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) o.tokens[(size_t) (p0 + t + 1)];
                if (prefill.draft_kv(mtp, R_rows, nxt.data(), T, p0, e)) return true;   // the draft layer's K/V from the batched prompt path
                return e.empty() && mtp.prefill(R_rows, nxt.data(), T, p0, e);
            };
        }
        const Clock::time_point tp0 = Clock::now();
        if (!prefill.run(o.tokens.data(), n_batched, 0, err)) {
            return fatal_start_error(err);
        }
        // refill the lent slots from the arena and give them back to the decode tier
        if (!lent.empty()) {
            const Clock::time_point tr = Clock::now();
            for (const auto& [i, slot] : lent) {   // queued, one wait
                const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
                const int64_t nb = (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
                if (b == nullptr || !(xcache.fill_slot_queued(slot, b, err, nb))) {
                    basalt::log::error("basalt generate", "refilling a lent slot failed: %s", err.c_str());
                    return 1;
                }
                host_res[(size_t) i] = slot;
            }
            if (!xcache.sync_queued(err)) {
                basalt::log::error("basalt generate", "refilling the lent slots failed: %s", err.c_str());
                return 1;
            }
            res_put(d_res);
            basalt::log::info("basalt generate", "%zu lent slots refilled in %.1f ms", lent.size(),
                         std::chrono::duration<double, std::milli>(Clock::now() - tr).count());
        }
        prefill_batched_ms = std::chrono::duration<double, std::milli>(Clock::now() - tp0).count();
        prefill_ms += prefill_batched_ms;
        pos_start = n_batched;
        tok = o.tokens[(size_t) pos_start];
        // the PLE window of the token path: the two tokens before `pos_start`
        ss.ple_prev[0] = pos_start >= 2 ? (int32_t) o.tokens[(size_t) (pos_start - 2)] : -1;
        ss.ple_prev[1] = pos_start >= 1 ? (int32_t) o.tokens[(size_t) (pos_start - 1)] : -1;
        const basalt::prefill::PrefillStats& ps = prefill.stats();
        basalt::log::info("basalt generate", "prefill %lld tokens in %lld chunks, %.1f ms (%.1f tok/s); experts "
                             "streamed %lld (%lld by DMA, host %.1f ms), resident %lld, on the CPU %lld (share %.2f); PLE %.1f ms",
                     (long long) ps.tokens, (long long) ps.chunks, ps.ms_total,
                     ps.ms_total > 0 ? 1000.0 * (double) ps.tokens / ps.ms_total : 0.0, (long long) ps.experts_streamed,
                     (long long) ps.experts_dma, ps.ms_experts_host, (long long) ps.experts_resident,
                     (long long) ps.experts_cpu, ps.cpu_share, ps.ms_ple);
    }

    // The last prompt token is the first verify window (T = 1).
    context.spec_pos = &pos_start;
    return decode(context);
}

}
