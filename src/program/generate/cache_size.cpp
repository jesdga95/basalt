#include "context.hpp"

namespace basalt::program::generate {

int cache_size(Context& context) {
    auto& o = *context.o;
    auto& peer_streamed_share = *context.peer_streamed_share;
    auto& multi_gpu = *context.multi_gpu;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& K = *context.K;
    auto& profile = *context.profile;
    auto& pf_borrow = *context.pf_borrow;
    auto& kWindowMib = *context.kWindowMib;
    auto& expert_total_bytes = *context.expert_total_bytes;
    auto& wt = *context.wt;
    auto& ss = *context.ss;
    auto& kv_floor = *context.kv_floor;
    auto& bslot_ss = *context.bslot_ss;
    auto& slot_kv_cuda0 = *context.slot_kv_cuda0;
    auto& slot0_is_solo = *context.slot0_is_solo;
    auto& mtp = *context.mtp;
    auto& n_vocab = *context.n_vocab;
    auto& native_head = *context.native_head;
    auto& mtp_head = *context.mtp_head;

    // ---- R4's slot storage.  Allocated AFTER the weights and the session, so `cudaMemGetInfo` inside `open`
    // sees the memory this process actually has left rather than the card's idle figure - and refuses with both
    // numbers if the slots do not fit, instead of handing back a cache smaller than it was asked for.
    basalt::core::Verifier::prepare_hc8(wt, g); // before the cache sizes itself
    if (!o.dump_hc8.empty()) {
        // --dump-hc8: the copies, as a .basalt's `hc8` section holds them; nothing else of the startup is kept
        cudaDeviceSynchronize();
        const auto extra = o.mtp.empty() ? std::vector<std::pair<std::string, const void*>>{} : mtp.hc8_keys();
        if (!basalt::core::hc8_dump(wt, g, extra, o.dump_hc8, err)) {
            basalt::log::error("basalt generate", "--dump-hc8: %s", err.c_str());
            return 1;
        }
        return 0;
    }
    // what the peer's prompt buffers take on its card: they are allocated after its tier is filled, so the tier leaves them free
    auto peer_prompt_bytes = [&]() -> uint64_t {
        return o.serve && o.peer_device >= 1 && o.peer_prefill_rows != 0
                   ? basalt::prefill::Prefill::peer_bytes(o.prefill_chunk, peer_streamed_share) : 0;
    };
    // the GDN mixer split is priced once, before either card sizes its experts: its peer reserve displaces experts
    // only when the cards cannot hold them all
    if (o.peer_device >= 1) {
        int d0 = 0, count = 0;
        size_t f0 = 0, t0 = 0, f1 = 0, t1 = 0;
        cudaGetDevice(&d0);
        cudaMemGetInfo(&f0, &t0);
        if (cudaGetDeviceCount(&count) == cudaSuccess && o.peer_device < count && cudaSetDevice(o.peer_device) == cudaSuccess) {
            cudaMemGetInfo(&f1, &t1);
            (void) cudaSetDevice(d0);
        }
        (void) cudaGetLastError();
        const int max_t = std::max({o.spec, o.batch, 2});
        const int64_t hs = native_head.loaded()
            ? basalt::core::Verifier::hs_peer_bytes(native_head.row_bytes(), native_head.type(), n_vocab, g.n_embd, max_t,
                                                    d0, o.peer_device) : 0;
        const int64_t peer_room = (int64_t) f1 - (((int64_t) std::max(o.peer_reserve_mib, 128) << 20) +
                                                  (int64_t) basalt::core::PeerExperts::device_overhead_bytes() + hs +
                                                  (int64_t) peer_prompt_bytes());
        // CUDA0's cache: its free memory less the reserve and what loads after the cache
        const int64_t cache0 = std::max<int64_t>(0, (int64_t) f0 - (((int64_t) o.vram_reserve_mib + 1024) << 20));
        const int64_t outside = std::max<int64_t>(0, (int64_t) expert_total_bytes - cache0 - std::max<int64_t>(peer_room, 0));
        const int64_t split = basalt::core::Verifier::gs_peer_bytes(wt, g, d0, o.peer_device);
        basalt::core::Verifier::gs_decide(wt, g, d0, o.peer_device, max_t, std::min(split, outside));
    }
    basalt::core::ExpertCache xcache;
    // THE HEAD BEFORE THE CACHE: the native head and the logits are allocated above, before the expert arena
    // Profiled slots are padded; sizing and allocation must charge the same stride.
    const uint64_t max_cache_blob = basalt::kernels::cpu::expert_layout().max_blob;
    const uint64_t cache_slot_bytes = profile.empty() ? max_cache_blob : (max_cache_blob + 255) / 256 * 256;
    const bool auto_cache = o.expert_cache < 0;
    if (o.expert_cache < 0) {
        // the prompt path's cuBLAS handle is made now, so the free figure below already has its device memory out
        if (o.serve && o.prefill_chunk > 0 && !basalt::prefill::Gemm::prepare_handle(err)) return fatal_start_error(err);
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        // The batched prompt path's chunk buffers are allocated later, so they are reserved here - an over-subscribed
        // allocation can page to system memory and crawl.  With borrowing - the default with a profile - the prompt
        // path lends cache slots instead (`pf_borrow`, hoisted above so both cache-size branches read the same one).
        const int64_t prefill_mib = (o.prefill_chunk > 0 && !pf_borrow) ? 160 + (o.prefill_chunk * 680) / 1024 : 0;
        // the draft layer's head and logits are allocated when it binds, after this, so they come out of the reserve
        const basalt::core::NativeHead& draft_head = multi_gpu ? mtp_head : native_head;
        const int64_t mtp_bind = (!o.mtp.empty() && draft_head.loaded())
                                     ? (int64_t) mtp.bind_bytes(draft_head.row_bytes(), draft_head.type(), n_vocab) : 0;
        // a server allocates its verify windows after the cache is sized: they come out of the reserve too, so the
        // free VRAM after loading is the reserve itself
        int64_t late_b = o.serve ? (kWindowMib << 20) : 0;
        // the kernels that first run after the cache is sized (the prompt path's, the verify windows') load into the
        // reserve when modules load lazily.  A server loading every kernel up front still loads cuBLAS's GEMM kernels on
        // their first use and captures its window graphs after the cache, so the same room stays back for those, and
        // batched serving adds the further slot sessions' solo windows and the batch windows.  This room loads into the
        // reserve: the reserve is raised only when it cannot hold the room and still leave the floor free beside it.
        int64_t late_room = 0;
        {
            const char* ml = std::getenv("CUDA_MODULE_LOADING");
            constexpr int64_t kLazyKernelsMib = 96;
            constexpr int64_t kSlotWindowsMib = 24;   // one session's solo windows, or the batch windows
            if (o.serve || ml == nullptr || std::strcmp(ml, "EAGER") != 0) late_room += kLazyKernelsMib << 20;
            if (o.serve && o.batch >= 2 && slot0_is_solo) late_room += ((int64_t) o.batch * kSlotWindowsMib) << 20;
        }
        const int64_t reserve_b = std::max<int64_t>((int64_t) o.vram_reserve_mib << 20,
                                                    late_room + ((int64_t) vram_floor_mib() << 20));
        // the GDN mixer split by key head loads CUDA0's share of the GDN layers after the cache is sized
        if (o.peer_device >= 1) {
            int d0 = 0;
            cudaGetDevice(&d0);
            late_b += basalt::core::Verifier::gs_cuda0_bytes(wt, g, d0, o.peer_device);
        }
        // the deferred checkpoint copy's spare device buffer is allocated after the cache, so it comes out of the reserve
        if (o.serve && o.prompt_cache > 0 && !multi_gpu) {
            basalt::core::ConversationStateSizes zc;
            std::string ce;
            if (basalt::core::conversation_state_sizes(g, zc, ce)) late_b += (int64_t) zc.gdn;
        }
        const int64_t base_b = reserve_b + ((int64_t) prefill_mib << 20) + mtp_bind + late_b;
        // the RAM copy of the experts the cache does not hold is page-locked and mapped, which takes device memory per
        // GiB mapped: a 1 GiB probe gives the rate, applied to the experts that stay out of the cache
        if (o.serve && o.resident_cpu_experts && expert_total_bytes > 0) {
            void* probe = nullptr;
            const size_t pb = (size_t) 1 << 30;
            cudaDeviceSynchronize();
            size_t f0 = 0, f1 = 0, t0 = 0;
            cudaMemGetInfo(&f0, &t0);
            if (cudaHostAlloc(&probe, pb, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess) {
                cudaMemGetInfo(&f1, &t0);
                cudaFreeHost(probe);
                const double per_gib = f0 > f1 ? (double) (f0 - f1) : 0.0;
                const double outside = std::max(0.0, (double) expert_total_bytes -
                                                         (double) std::max<int64_t>((int64_t) free_b - base_b, 0));
                late_b += (int64_t) (per_gib * outside / (double) pb);
            }
            cudaGetLastError();
        }
        const int64_t reserve = reserve_b + ((int64_t) prefill_mib << 20) + mtp_bind + late_b;
        const int64_t blob = (int64_t) cache_slot_bytes;
        int64_t slots = ((int64_t) free_b - reserve) / blob;
        if (!profile.empty()) slots = std::min<int64_t>(slots, (int64_t) profile.size());
        o.expert_cache = (int) std::max<int64_t>(slots, 0);
        // a reserve raised to hold the late room says what was asked
        const std::string raised = reserve_b > ((int64_t) o.vram_reserve_mib << 20)
            ? " (" + std::to_string(o.vram_reserve_mib) + " asked, raised to hold " + std::to_string(late_room >> 20) +
                  " MiB that loads into it and the " + std::to_string(vram_floor_mib()) + " MiB floor)"
            : std::string();
        basalt::log::info("basalt generate", "expert cache auto: %.2f GiB free, %d MiB reserved%s (+%lld MiB for the "
                             "draft head, +%lld MiB for what loads after) -> %d slots",
                     (double) free_b / 1073741824.0, (int) (reserve_b >> 20), raised.c_str(), (long long) (mtp_bind >> 20),
                     (long long) (late_b >> 20), o.expert_cache);
        // The batch slots' sessions came out of the same VRAM: say when that leaves the cache short of the expert set
        // (experts then swap in and out per token) or of half of what it would hold without them.
        if (slot_kv_cuda0 > 0) {
            int64_t without = ((int64_t) free_b + (int64_t) slot_kv_cuda0 - reserve) / blob;
            if (!profile.empty()) without = std::min<int64_t>(without, (int64_t) profile.size());
            const bool swaps = !profile.empty() && o.expert_cache < (int64_t) profile.size() && without >= (int64_t) profile.size();
            if (without > 0 && (swaps || o.expert_cache * 2 < without))
                basalt::log::warn("basalt", "the %d batch slots' sessions (the first is the solo one) take %.1f GiB of CUDA0 beyond it, leaving an "
                                     "expert cache of %d slots (%.1f GiB) where %lld would fit without them%s. Lower "
                                     "--max-context or drop --slot-full-context (or use fewer --parallel slots)",
                             o.batch, (double) slot_kv_cuda0 / 1073741824.0, o.expert_cache,
                             (double) o.expert_cache * (double) blob / 1073741824.0, (long long) without,
                             swaps ? "; the configured expert set no longer fits, so experts swap per token" : "");
        }
        // The verify window cannot start without a cache, and a cache too small to lend the prompt path a 256-token
        // chunk's buffers (plus the 128 slots a loan leaves; one slot without --prefill) makes it allocate its own on
        // top - more than the reserve.  When the default reserve leaves less than that (a small card), the reserve
        // shrinks to what leaves exactly that cache, down to kSmallReserveMib: what is allocated after the cache - the
        // prompt path's own part, the verify buffers, the draft head - comes out of the reserve, so the cache gets no
        // more than it needs, and the serve check says so when it ends low.  A reserve
        // given on the command line is kept.  No slot at all: the start stops, saying what is short and what makes
        // room.  A card the default reserve leaves that much is sized as before.
        constexpr int kSmallReserveMib = 300;
        const int64_t min_slots = (o.prefill_chunk > 0 && pf_borrow)
            ? ((int64_t) basalt::prefill::Prefill::bytes_needed(g, ss, 256) + blob - 1) / blob + 128 : 1;
        if (o.expert_cache < min_slots && !o.vram_reserve_given && o.vram_reserve_mib > kSmallReserveMib) {
            // the largest reserve (in MiB) that still leaves min_slots
            const int64_t fit_mib = ((int64_t) free_b - mtp_bind - late_b - min_slots * blob) / (1 << 20) - prefill_mib;
            if (fit_mib >= kSmallReserveMib) {
                const int r = (int) std::min<int64_t>(fit_mib, o.vram_reserve_mib);
                const int64_t rb = std::max<int64_t>((int64_t) r << 20, late_room + ((int64_t) vram_floor_mib() << 20));
                int64_t s2 = ((int64_t) free_b - (rb + ((int64_t) prefill_mib << 20) + mtp_bind + late_b)) / blob;
                if (!profile.empty()) s2 = std::min<int64_t>(s2, (int64_t) profile.size());
                basalt::log::warn("basalt generate", "expert cache auto: the %d MiB reserve leaves too few slots on "
                                     "this card (a working cache needs %lld): a %d MiB reserve instead -> %lld slots",
                             o.vram_reserve_mib, (long long) min_slots, r, (long long) s2);
                o.vram_reserve_mib = r;
                o.expert_cache = (int) s2;
            }
        }
        if (o.expert_cache == 0) {
            // what is short, and what makes room: the numbers a small card picks from
            const int64_t at_reserve = o.vram_reserve_given ? o.vram_reserve_mib
                                                            : std::min(o.vram_reserve_mib, kSmallReserveMib);
            const int64_t need_b = std::max<int64_t>(at_reserve << 20, late_room + ((int64_t) vram_floor_mib() << 20)) +
                                   ((int64_t) prefill_mib << 20) + mtp_bind + late_b + min_slots * blob;
            const int64_t short_mib = std::max<int64_t>(1, (need_b - (int64_t) free_b + (1 << 20) - 1) >> 20);
            const int64_t session_mib =
                (int64_t) (basalt::core::session_bytes(g, o.max_context, K, 0, g.n_layers) >> 20);
            const std::string reserve_tip =
                o.vram_reserve_given && o.vram_reserve_mib > kSmallReserveMib
                    ? ", a smaller --vram-reserve-mib (" + std::to_string(o.vram_reserve_mib) + " now; " +
                          std::to_string(kSmallReserveMib) + " is enough on a small card)"
                    : std::string();
            basalt::log::error("basalt generate", "no VRAM is left for the expert cache: it needs at least %lld slots "
                                 "(%lld MiB), about %lld MiB more than this card has free. To make room: a smaller "
                                 "--max-context (the session, mostly its KV cache, takes %lld MiB at %lld tokens), "
                                 "--kv q4_0, a smaller draft vocabulary (the draft head takes "
                                 "%lld MiB now)%s, images on the CPU, or close other programs that use the GPU%s",
                         (long long) min_slots, (long long) ((min_slots * blob) >> 20), (long long) short_mib,
                         (long long) session_mib, (long long) o.max_context, (long long) (mtp_bind >> 20),
                         reserve_tip.c_str(),
                         slot_kv_cuda0 == 0 ? "" : (", or fewer --parallel slots / no --slot-full-context (the slot sessions take " +
                                                    std::to_string(slot_kv_cuda0 >> 20) + " MiB)").c_str());
        }
    } else if (multi_gpu && o.expert_cache > 0) {
        // an explicit cache size leaves room for the prompt path's buffers and the reserve, or the first prompt
        // fails with "device buffers ... do not fit" (with borrowing - the default with a profile - the path lends
        // slots instead and `prefill_mib` is 0, so only the reserve is checked)
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t prefill_mib = (o.prefill_chunk > 0 && !pf_borrow) ? 160 + (o.prefill_chunk * 680) / 1024 : 0;
        const int64_t reserve = ((int64_t) o.vram_reserve_mib + prefill_mib) << 20;
        const int64_t fit = std::max<int64_t>(((int64_t) free_b - reserve) / (int64_t) cache_slot_bytes, 0);
        if (o.expert_cache > fit) {
            // a WARNING that names the knob: the user asked for this size, and gets fewer slots
            basalt::log::warn("basalt generate", "layer split: --expert-cache %d leaves no room for the "
                                 "prompt path's buffers (%lld MiB) and the %d MiB reserve on CUDA0: %lld slots instead "
                                 "(a smaller --vram-reserve-mib leaves more of them)", o.expert_cache,
                         (long long) prefill_mib, o.vram_reserve_mib, (long long) fit);
            o.expert_cache = (int) fit;
        }
    }
    // A native pack's blobs differ per layer, so with a profile its slots are sized per pair: the same VRAM holds more
    // IQ3_XXS experts than slots of the largest blob would.
    std::vector<int64_t> sized_slots;
    if (o.expert_cache > 0 && !profile.empty()) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const auto& lay = basalt::kernels::cpu::expert_layout();
        const uint64_t budget = (uint64_t) o.expert_cache * cache_slot_bytes;   // what the uniform sizing granted
        uint64_t used = 0;
        size_t free_room = free_b > ((size_t) o.vram_reserve_mib << 20) ? free_b - ((size_t) o.vram_reserve_mib << 20) : 0;
        const uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);
        for (const auto& pr : profile) {
            const uint64_t b = (lay.blob_bytes(pr.first) + 255) / 256 * 256;
            if (used + b > cap) break;
            used += b;
            sized_slots.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        o.expert_cache = (int) sized_slots.size();
    }
    // --vram-elastic: the cache in physical segments (the VRAM command resizes it between requests).  One GPU, no
    // helper caches, serve mode: anything else keeps the one cudaMalloc, said once.
    if (o.vram_elastic) {
        const char* why = !o.serve ? "it works between requests of --serve"
                        : multi_gpu ? "a layer split has a cache per GPU"
                        : std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                      [](int n) { return n > 0; }) ? "the helper caches on other GPUs"
                        : o.vram_segment_mib < 64 ? "--vram-segment-mib is below 64" : nullptr;
        if (why != nullptr) {
            basalt::log::warn("basalt generate", "--vram-elastic is off: %s", why);
            o.vram_elastic = false;
        } else {
            xcache.set_segment_bytes(o.vram_segment_mib << 20);
        }
    }
    // the elastic KV: what the sessions' unbacked cells would take is the cache's tail, lent back to them on demand
    uint64_t kv_tail_bytes = 0;
    if (kv_floor > 0) {
        auto add = [&](const basalt::core::SessionState& s) {
            kv_tail_bytes += basalt::core::session_kv_bytes_for(s, s.max_cells) - basalt::core::session_kv_bytes_for(s, kv_floor);
        };
        add(ss);
        if (!bslot_ss.empty())
            for (const auto& u : bslot_ss[0])
                if (u.get() != &ss) add(*u);
        xcache.set_elastic_tail((int64_t) kv_tail_bytes);
    }
    if (o.expert_cache > 0) {
        // keep the first `keep_bytes` of the cache (the profile's hottest experts first); false when nothing is left
        auto shrink_to = [&](int64_t keep_bytes) -> bool {
            if (keep_bytes <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            if (!sized_slots.empty()) {
                int64_t used = 0;
                size_t keep = 0;
                while (keep < sized_slots.size() && used + (sized_slots[keep] + 255) / 256 * 256 <= keep_bytes)
                    used += (sized_slots[keep++] + 255) / 256 * 256;
                sized_slots.resize(keep);
                o.expert_cache = (int) keep;
            } else {
                o.expert_cache = (int) (keep_bytes / (int64_t) cache_slot_bytes);
            }
            if (o.expert_cache <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            return true;
        };
        auto cache_bytes = [&]() -> int64_t {
            if (sized_slots.empty()) return (int64_t) o.expert_cache * (int64_t) cache_slot_bytes;
            int64_t b = 0;
            for (const int64_t s : sized_slots) b += (s + 255) / 256 * 256;
            return b;
        };
        // With `--expert-cache auto` the reserve must still be free once the slots are WRITTEN: an allocation may not
        // be resident until it is touched, and the free figure read before it can be too high.  A cache sized from it
        // could fill the card, the driver would then page, and a request that needed a page back while the verify
        // graph spun on a host flag would never finish.  So the slots are zeroed and the free figure read again; while
        // it is short of the reserve the cache is reopened smaller.
        int failed = 0;
        int zero_reads = 0;
        for (int attempt = 0;; ++attempt) {
            bool ok = false;
            {
                ok = sized_slots.empty()
                    ? xcache.open(o.expert_cache, g.n_layers, g.n_expert, (int64_t) basalt::kernels::cpu::expert_layout().max_blob, err)
                    : xcache.open_sized(sized_slots, g.n_layers, g.n_expert, err);
            }
            if (!ok) {
                // A device allocation can also be charged to the system commit (RAM + page file), so with a small page
                // file the cache's one big cudaMalloc fails while the VRAM is free.  An auto cache then tries three
                // quarters of the size, a few times, instead of stopping the engine.
                char commit[96] = "";
                if (auto_cache && failed < 8 && shrink_to(cache_bytes() / 4 * 3)) {
                    ++failed;
                    basalt::log::warn("basalt generate", "%s%s; trying a smaller expert cache: %d slots", err.c_str(),
                                 commit, o.expert_cache);
                    continue;
                }
                return fatal_start_error(err);
            }
            if (!auto_cache || attempt - failed >= 6) break;
            cudaMemset(xcache.device_slot(0), 0, (size_t) xcache.bytes());
            cudaDeviceSynchronize();
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const int64_t want = (int64_t) o.vram_reserve_mib << 20;
            if ((int64_t) free_b >= want - (64ll << 20)) break;
            // short by (want - free); a figure of 0 only says "at least": the first two such reads give back 1 GiB
            // each (the free figure read before the allocation can run high), later ones a quarter
            int64_t give = want - (int64_t) free_b + (64ll << 20);
            if (free_b < ((size_t) 16 << 20))
                give = std::max<int64_t>(give, ++zero_reads <= 2 ? 1ll << 30 : xcache.bytes() / 4);
            const int64_t keep_bytes = xcache.bytes() - give;
            basalt::log::warn("basalt generate", "only %lld MiB free once the slots are written (reserve %d MiB); "
                                 "shrinking the expert cache", (long long) (free_b >> 20), o.vram_reserve_mib);
            xcache.close();
            if (!shrink_to(keep_bytes)) break;
        }
        if (failed > 0 && o.expert_cache > 0)
            basalt::log::warn("basalt generate", "expert cache: %d slots (%.2f GiB) after %d smaller tries - a bigger "
                                 "page file lets it use more of the free VRAM",
                         o.expert_cache, (double) xcache.bytes() / 1073741824.0, failed);
    }
    // the slots wholly or partly in the elastic tail: the ones the KV may take back
    const int64_t kv_tail_slots = xcache.valid() && xcache.elastic_bytes() > 0
        ? xcache.full_slots() - xcache.slots_within(xcache.full_bytes() - xcache.elastic_bytes()) : 0;
    if (kv_tail_slots > 0)
        basalt::log::info("basalt generate", "elastic KV: %lld cells backed per session, the other %.2f GiB of KV are "
                             "the expert cache's last %lld slots until a request needs them",
                     (long long) kv_floor, (double) kv_tail_bytes / 1073741824.0, (long long) kv_tail_slots);
    if (o.expert_cache > 0) {
        basalt::log::info("basalt generate", "expert cache %lld slots, %.2f GiB of VRAM; policy is",
                     (long long) xcache.slots(), xcache.gib());
        xcache.set_per_layer_admission(false);
        // A GPU expert and the CPU's compute the same quantized expert with different float order, so a near-tie can
        // flip against a cache-off run; neither output is more correct than the other.  (native_expert_parity,
        // expert_parity and the grouped kernels' tests check the kernels themselves.)
        basalt::log::info("basalt generate", "the GPU computes the experts in the cache; it rounds differently from the CPU,\n"
                     "                 so a reply can differ slightly from a run without the cache (same quality).");
        if (profile.empty()) {
            basalt::log::raw("                 compulsory-miss (fills with whatever the run routes first).\n");
        } else {
            basalt::log::raw("                 PROFILE, ranked by routing frequency, no eviction.\n");
        }
    }

    context.peer_prompt_bytes = peer_prompt_bytes;
    context.xcache = &xcache;
    context.kv_tail_slots = &kv_tail_slots;
    return cache_fill(context);
}

}
