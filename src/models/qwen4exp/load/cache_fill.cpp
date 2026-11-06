#include "basalt/models/qwen4exp/shared_names.hpp"
#include "context.hpp"

namespace basalt::models::qwen4exp::generate {

int cache_fill(Context& context) {
    auto& o = *context.o;
    auto& split_at = *context.split_at;
    auto& multi_gpu = *context.multi_gpu;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& profile = *context.profile;
    auto& stage_room = context.stage_room;
    auto& model_bytes = *context.model_bytes;
    auto& expert_total_bytes = *context.expert_total_bytes;
    auto& cuda0_fixed_bytes = *context.cuda0_fixed_bytes;
    auto& wt = *context.wt;
    auto& stages = *context.stages;
    auto& n_vocab = *context.n_vocab;
    auto& native_head = *context.native_head;
    auto& src = *context.src;
    auto& srcp = *context.srcp;
    auto& peer_prompt_bytes = context.peer_prompt_bytes;
    auto& xcache = *context.xcache;

    // ---- fill the tier from the profile.  This is the only place the plan is applied, and it runs
    // ONCE: with `slots` pairs and `slots` slots the cache is full when this returns, so the decode-time
    // admission finds no room and every non-profiled expert stays a CPU miss.  That is what makes the profile
    // the policy rather than a hint.
    int64_t prefilled = 0;
    uint64_t cache_fill_bytes = 0;   // what the GPU cache took from the profile (counted by its load phase)
    // The model's bytes against what the tiers will hold, from the same quantities the loader uses: the primary card
    // keeps its weights and the cache just sized, the peer card its free memory less its buffers and reserve, and the
    // RAM copy the room past the headroom.  With the file-backed modes the rest is read from the SSD (the default
    // arena mode was checked against the RAM above).
    // what the peer card can hold of experts: its free memory less its buffers and reserve, capped by --peer-slots
    // what the verify window's splits (the head by vocabulary, the GDN mixer by key head) load on the peer after its
    // expert tier is filled, when the GDN mixer split is taken; without it the head's peer rows fit in what the tier
    // leaves free
    auto peer_split_bytes = [&](int dev0) -> int64_t {
        const int64_t gs = basalt::core::Verifier::gs_peer_bytes(wt, g, dev0, o.peer_device);
        if (gs == 0) return 0;
        const int64_t hs = native_head.loaded()
            ? basalt::core::Verifier::hs_peer_bytes(native_head.row_bytes(), native_head.type(), n_vocab, g.n_embd,
                                                    std::max({o.spec, o.batch, 2}), dev0, o.peer_device) : 0;
        return hs + gs;
    };
    auto peer_expert_room = [&]() -> uint64_t {
        uint64_t room = 0;
        if (o.peer_device < 1) return 0;
        int cur = 0, count = 0;
        if (cudaGetDevice(&cur) != cudaSuccess || cudaGetDeviceCount(&count) != cudaSuccess) {
            (void) cudaGetLastError();
            return 0;
        }
        std::vector<int> seen;   // each card once; its tiers share its free memory
        for (int dev : o.peer_devices) {
            if (dev >= count || std::find(seen.begin(), seen.end(), dev) != seen.end()) continue;
            seen.push_back(dev);
            const uint64_t tiers = (uint64_t) std::count(o.peer_devices.begin(), o.peer_devices.end(), dev);
            size_t pfree = 0, ptotal = 0;
            if (cudaSetDevice(dev) == cudaSuccess && cudaMemGetInfo(&pfree, &ptotal) == cudaSuccess) {
                const uint64_t reserve = ((uint64_t) std::max(o.peer_reserve_mib, 128) << 20) +
                                         (dev == o.peer_device ? peer_prompt_bytes() : 0) +
                                         tiers * basalt::core::PeerExperts::device_overhead_bytes(g.n_embd, g.n_ff) +
                                         (dev == o.peer_device ? (uint64_t) peer_split_bytes(cur) : 0);
                room += (uint64_t) pfree > reserve ? (uint64_t) pfree - reserve : 0;
            }
        }
        (void) cudaSetDevice(cur);
        (void) cudaGetLastError();
        return room;
    };
    if (model_bytes > 0 && o.mmap_experts) {
        const auto free_ram = basalt::core::conversation_available_memory();
        if (free_ram) {
            uint64_t cache_bytes_now = 0;
            if (!profile.empty() && srcp != nullptr) {
                const int64_t n = xcache.per_layer_admission()
                                      ? (int64_t) profile.size()
                                      : std::min<int64_t>((int64_t) profile.size(), xcache.slots());
                for (int64_t i = 0; i < n; ++i)
                    cache_bytes_now += basalt::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first);
            }
            cache_bytes_now = std::min(cache_bytes_now, expert_total_bytes);
            const uint64_t vram = cuda0_fixed_bytes + cache_bytes_now;
            uint64_t outside = expert_total_bytes - cache_bytes_now;   // the experts no cache on the primary holds
            uint64_t ram = basalt::core::ram_tier_room(
                *free_ram, std::max(o.resident_headroom, basalt::platform::ram_reserve()), o.resident_pin);
            if (o.resident_cpu_experts && (o.resident_budget > 0 || o.resident_soft)) {
                const uint64_t margin = 256ull << 20;
                ram = ram > margin ? ram - margin : 0;
                if (o.resident_budget > 0) ram = std::min(ram, o.resident_budget);
            }
            // the peer tier holds experts too: what its card has free beyond its buffers and the reserve it leaves
            uint64_t peer_vram = peer_expert_room();
            peer_vram = std::min(peer_vram, outside);
            outside -= peer_vram;
            if (model_bytes > vram + peer_vram + ram) {
                ram = std::min(ram, outside);
                if (model_bytes > vram + peer_vram + ram)
                    basalt::log::warn("basalt", "model %.1f GiB, free VRAM + RAM %.1f GiB: the rest is read from "
                                         "the SSD while generating (slow)",
                                 (double) model_bytes / 1073741824.0,
                                 (double) (vram + peer_vram + ram) / 1073741824.0);
            }
        }
    }
    if (!profile.empty() && srcp != nullptr) {
        // per layer, a full layer skips only its own pairs - each layer takes its hottest experts until its range is
        // full
        const bool per_layer = xcache.per_layer_admission();
        const int64_t want = per_layer ? (int64_t) profile.size()
                                       : std::min<int64_t>((int64_t) profile.size(), xcache.slots());
        const auto fill_t0 = std::chrono::steady_clock::now();
        uint64_t fill_bytes = 0;
        uint64_t fill_total = 0;
        for (int64_t i = 0; i < want; ++i)
            fill_total += basalt::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first);
        cache_fill_bytes = fill_total;
        if (model_bytes > 0) {
            if (srcp == &src) {
                basalt::core::LoadProgress::plan("cuda0", cuda0_fixed_bytes + fill_total);
                basalt::core::LoadProgress::plan("ram", expert_total_bytes - fill_total);
            } else {
                basalt::core::LoadProgress::plan("cuda0", fill_total, true);
            }
        }
        basalt::core::LoadProgress fill_progress(basalt::core::LoadProgress::cuda(0), fill_total, srcp != &src);
        // the slots are handed out in profile order, then the blobs are read and copied together
        std::vector<std::pair<int32_t, int32_t>> fill_pairs;
        std::vector<int32_t> fill_slots;
        for (int64_t i = 0; i < want; ++i) {
            const int32_t slot = xcache.admit(profile[(size_t) i].first, profile[(size_t) i].second);
            if (slot == basalt::core::kNotResident) {
                if (per_layer) continue;
                break;
            }
            fill_pairs.push_back(profile[(size_t) i]);
            fill_slots.push_back(slot);
            fill_bytes += basalt::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first);
        }
        if (!basalt::core::fill_cache_slots(*srcp, xcache, 0, fill_pairs.data(), fill_slots.data(),
                                            (int64_t) fill_pairs.size(),
                                            [&](uint64_t bytes) { fill_progress.add(bytes); }, err)) {
            basalt::log::error("basalt generate", "the profile fill failed: %s", err.c_str());
            return 1;
        }
        prefilled += (int64_t) fill_pairs.size();
        fill_progress.finish();
        const double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - fill_t0).count();
        // **AND ONE SLOT IS READ BACK AND COMPARED.**  A residency table that is right about indices and wrong
        // about bytes produces a plausible token, which is this project's most expensive failure mode; the
        // cache's own `verify_slot` is the check and it costs one blob-sized D2H at startup.
        if (prefilled > 0 && !xcache.verify_slot(xcache.slot_of(profile[0].first, profile[0].second),
                                srcp->blob(profile[0].first, profile[0].second), err,
                                (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(profile[0].first))) {
            return fatal_start_error(err);
        }
        basalt::log::info("basalt generate", "pre-filled %lld of %lld slots from the profile in %.1f s (%.0f MB/s); "
                             "slot 0 verified",
                     (long long) prefilled, (long long) (per_layer ? xcache.slots() : want), fill_s,
                     fill_s > 0 ? (double) fill_bytes / 1e6 / fill_s : 0.0);
    }

    for (auto& stp : stages) {
        GpuStage& st = *stp;
        const auto& lay = basalt::kernels::cpu::expert_layout();
        // the drafter and the head are already allocated by now (they load above, before this), so what is left
        // to hold back is the windows - and `free_b` has already lost the drafter.
        const int64_t room = stage_room(st.dev, true, 0);
        const basalt::core::OnDevice on(st.dev);
        {
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            basalt::log::info("basalt generate", "layer split, CUDA%d: %.2f GiB free of %.2f, room for experts %.2f GiB",
                         st.dev, (double) fb / 1073741824.0, (double) tb / 1073741824.0, (double) room / 1073741824.0);
        }
        std::vector<int64_t> sized;
        int64_t used = 0;
        for (const auto& pr : st.profile) {
            const int64_t b = ((int64_t) lay.blob_bytes(pr.first) + 255) / 256 * 256;
            if (used + b > room) break;
            used += b;
            sized.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        if (sized.empty() ||
            !st.cache.open_sized(sized, g.n_layers, g.n_expert, err)) {
            basalt::log::error("basalt generate", "layer split, CUDA%d expert cache: %s", st.dev,
                         sized.empty() ? "no room" : err.c_str());
            return 1;
        }
        int64_t filled = 0;
        uint64_t stage_fill_total = 0;
        for (size_t i = 0; i < st.profile.size() && (int64_t) i < st.cache.slots(); ++i)
            stage_fill_total += lay.blob_bytes(st.profile[i].first);
        basalt::core::LoadProgress stage_progress(basalt::core::LoadProgress::cuda(st.dev), stage_fill_total);
        std::vector<std::pair<int32_t, int32_t>> stage_pairs;
        std::vector<int32_t> stage_slots;
        for (const auto& pr : st.profile) {
            if (filled >= st.cache.slots()) break;
            const int32_t slot = st.cache.admit(pr.first, pr.second);
            if (slot == basalt::core::kNotResident) break;
            stage_pairs.push_back(pr);
            stage_slots.push_back(slot);
            ++filled;
        }
        if (!basalt::core::fill_cache_slots(*srcp, st.cache, st.dev, stage_pairs.data(), stage_slots.data(),
                                            (int64_t) stage_pairs.size(),
                                            [&](uint64_t bytes) { stage_progress.add(bytes); }, err)) {
            basalt::log::error("basalt generate", "layer split, CUDA%d profile fill failed: %s", st.dev, err.c_str());
            return 1;
        }
        stage_progress.finish();
        if (filled == 0 || !st.cache.verify_slot(st.cache.slot_of(st.profile[0].first, st.profile[0].second),
                                                 srcp->blob(st.profile[0].first, st.profile[0].second), err,
                                                 (int64_t) lay.blob_bytes(st.profile[0].first))) {
            basalt::log::error("basalt generate", "layer split, CUDA%d expert cache: %s", st.dev,
                         filled == 0 ? "nothing filled" : err.c_str());
            return 1;
        }
        basalt::log::info("basalt generate", "layer split: CUDA%d runs layers %lld-%lld, expert cache %lld slots "
                             "(%.2f GiB), %lld of its %zu profiled pairs; slot 0 verified",
                     st.dev, (long long) st.lb, (long long) (st.le - 1), (long long) st.cache.slots(), st.cache.gib(),
                     (long long) filled, st.profile.size());
    }
    if (multi_gpu)
        basalt::log::info("basalt generate", "layer split: CUDA0 runs layers 0-%lld", (long long) (split_at[0] - 1));
    // the experts the stages' caches hold are on a GPU, so the RAM copy leaves them out
    for (const auto& stp : stages)
        for (const auto& pr : stp->profile)
            if (stp->cache.slot_of(pr.first, pr.second) >= 0)
                cache_fill_bytes += basalt::kernels::cpu::expert_layout().blob_bytes(pr.first);

    context.cache_fill_bytes = &cache_fill_bytes;
    context.peer_split_bytes = peer_split_bytes;
    context.peer_expert_room = peer_expert_room;
    return token_path(context);
}

}
