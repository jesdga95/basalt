#include "context.hpp"

namespace basalt::program::generate {

int layer_plan(Context& context) {
    auto& o = *context.o;
    auto& pack_files = *context.pack_files;
    auto& split_at = *context.split_at;
    auto& split_devs = *context.split_devs;
    auto& split_auto = *context.split_auto;
    auto& multi_gpu = *context.multi_gpu;
    auto& pipe_want = *context.pipe_want;
    auto& kPairMibLater = *context.kPairMibLater;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& K = *context.K;
    auto& skip = *context.skip;
    auto& profile = *context.profile;

    // ---- layer split across GPUs: "auto" places the split points by a cost model of one decode window:
    //   - every layer costs its GPU a time inversely proportional to SMs x clock (the per-layer round trip and
    //     kernels, not the bytes);
    //   - an expert no cache holds costs a fixed time per unit of routed mass: the CPU pool in decode and the PCIe
    //     stream in prompts;
    //   - which experts a cache holds: its layers' profiled pairs, hottest first, until its free VRAM (less the
    //     reserve, the prompt path's buffers and, on a later GPU, room for its windows and the drafter) is used; the
    //     routed mass of rank r is taken as (r+1)^-1.2.
    // Up to 3 GPUs every placement is tried; beyond, the layers are shared in proportion to speed.
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
    const uint64_t kStage0OtherBytes = 100ull << 20;   // what CUDA0 holds beside its dense weights before the experts: buffers and tables
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
    if (multi_gpu && split_auto) {
        const auto& lay = basalt::kernels::cpu::expert_layout();
        const int ns = (int) split_devs.size() + 1;
        std::vector<int64_t> cap((size_t) ns), used((size_t) ns);
        std::vector<double> layer_ms((size_t) ns);
        for (int i = 0; i < ns; ++i) {
            const int dev = i == 0 ? 0 : split_devs[(size_t) i - 1];
            cap[(size_t) i] = stage_room(i == 0 ? -1 : dev, i > 0, i == 0 ? kDrafterMib : i + 1 == ns ? kHeadMib : 0);
            int sms = 0, khz = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            if (cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev) != cudaSuccess || khz <= 0) khz = 1800000;
            cudaGetLastError();
            const double speed = std::max(1.0, (double) sms * (double) khz / 1e6);   // SMs x GHz
            layer_ms[(size_t) i] = 0.33 * (84.0 * 2.617) / speed;
            basalt::log::info("basalt generate", "layer split auto: CUDA%d %d SMs at %.2f GHz -> %.2f ms per layer, "
                                 "%.2f GiB free before its weights", dev, sms, khz / 1e6, layer_ms[(size_t) i],
                         (double) cap[(size_t) i] / 1073741824.0);
        }
        // What a stage's card holds before its experts, by block: its layers' dense weights (the arena, the native
        // projections, the 8-bit hyper-connection copies) and what every stage holds whatever its layers.
        std::vector<uint64_t> pool_l, nat_l, hc8_l((size_t) g.n_layers, 0);
        uint64_t pool_sh = 0, nat_sh = 0, hc8_sh = 0;
        if (!basalt::core::WeightTable::pool_bytes_by_layer(pack_files, g.n_layers, pool_l, pool_sh, err, &skip) ||
            !basalt::core::NativeDense::plan_bytes_by_layer(o.native_shards, true, g.n_layers, nat_l, nat_sh, err)) {
            return fatal_start_error(err);
        }
        auto block_of = [](const std::string& n) -> long {
            return n.rfind("blk.", 0) == 0 ? std::strtol(n.c_str() + 4, nullptr, 10) : -1;
        };
        for (int64_t l = 0; l < g.n_layers; ++l)
            hc8_l[(size_t) l] = basalt::core::Verifier::hc8_plan_bytes([&](const std::string& n) { return block_of(n) == l; }, g);
        hc8_sh = basalt::core::Verifier::hc8_plan_bytes([&](const std::string& n) { return block_of(n) < 0; }, g);
        auto dense_bytes = [&](int i, int64_t lb, int64_t le) -> int64_t {
            uint64_t b = pool_sh + nat_sh + hc8_sh + (i == 0 ? kStage0OtherBytes : 0);
            for (int64_t l = lb; l < le; ++l) b += pool_l[(size_t) l] + nat_l[(size_t) l] + hc8_l[(size_t) l];
            return (int64_t) b;
        };
        const double miss_ms = 190.0;
        std::vector<double> mass(profile.size());
        double total_mass = 0;
        for (size_t r = 0; r < profile.size(); ++r) total_mass += (mass[r] = std::pow((double) r + 1.0, -1.2));
        auto cost = [&](int64_t l) -> int64_t {
            return ((int64_t) lay.blob_bytes(l) + 255) / 256 * 256;
        };
        // the predicted window time (ms) of a placement, and the routed mass its caches hold
        auto predict = [&](const std::vector<int64_t>& at, double& held_mass, int64_t& held) -> double {
            // THE CARVE, PRICED: a placement gives stage i the layers [lb, le), and that range's session is a
            // real cost on its device - subtracted here so the search knows what it leaves for experts.  This
            // is why the sessions are allocated after the search: `session_bytes` is pure arithmetic.
            std::vector<int64_t> capr((size_t) ns);
            for (int i = 0; i < ns; ++i) {
                const int64_t lb = i == 0 ? 0 : at[(size_t) i - 1];
                const int64_t le = i + 1 < ns ? at[(size_t) i] : g.n_layers;
                capr[(size_t) i] = cap[(size_t) i] - dense_bytes(i, lb, le) -
                                   (int64_t) basalt::core::session_bytes(g, o.max_context, K, lb, le);
            }
            std::fill(used.begin(), used.end(), 0);
            held_mass = 0;
            held = 0;
            std::vector<bool> full((size_t) ns, false);
            for (size_t r = 0; r < profile.size(); ++r) {
                const int64_t l = profile[r].first;
                int st = 0;
                while (st + 1 < ns && l >= at[(size_t) st]) ++st;
                if (full[(size_t) st]) continue;
                if (used[(size_t) st] + cost(l) > capr[(size_t) st]) { full[(size_t) st] = true; continue; }   // as the fill
                used[(size_t) st] += cost(l);
                held_mass += mass[r];
                ++held;
            }
            held_mass /= std::max(total_mass, 1e-9);
            double ms = miss_ms * (1.0 - held_mass);
            for (int i = 0; i < ns; ++i) {
                const int64_t lb = i == 0 ? 0 : at[(size_t) i - 1], le = i + 1 < ns ? at[(size_t) i] : g.n_layers;
                ms += (double) (le - lb) * layer_ms[(size_t) i];
            }
            return ms;
        };
        std::vector<int64_t> best, at((size_t) ns - 1);
        double best_ms = 1e30, best_mass = 0;
        int64_t best_held = 0;
        auto consider = [&]() {
            double hm = 0;
            int64_t held = 0;
            const double ms = predict(at, hm, held);
            if (ms < best_ms) { best = at; best_ms = ms; best_mass = hm; best_held = held; }
        };
        const int64_t L = g.n_layers;
        if (ns == 2) {
            for (int64_t k = 2; k < L; ++k) { at[0] = k; consider(); }
        } else if (ns == 3) {
            for (int64_t k1 = 2; k1 + 1 < L; ++k1)
                for (int64_t k2 = k1 + 1; k2 < L; ++k2) { at[0] = k1; at[1] = k2; consider(); }
        } else {
            double total = 0;
            for (const double c : layer_ms) total += 1.0 / c;
            double acc = 0;
            for (int i = 0; i + 1 < ns; ++i) {
                acc += 1.0 / layer_ms[(size_t) i];
                at[(size_t) i] = std::clamp<int64_t>((int64_t) std::llround(acc / total * (double) L),
                                                     i == 0 ? 2 : at[(size_t) i - 1] + 1, L - (ns - 1 - i));
            }
            consider();
        }
        split_at = best;
        std::string ks;
        for (const int64_t k : split_at) ks += (ks.empty() ? "" : ",") + std::to_string(k);
        basalt::log::info("basalt generate", "layer split auto: K=%s - predicted %.1f ms per decode window; the caches "
                             "hold %lld of %zu profiled pairs (~%.1f%% of the routed mass)", ks.c_str(), best_ms,
                     (long long) best_held, profile.size(), 100.0 * best_mass);
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
    // out and they are staged, copied and dropped right after the load
    std::set<std::string> hc8_names;
    if (basalt::kernels::hc8_only() && !multi_gpu) {
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
