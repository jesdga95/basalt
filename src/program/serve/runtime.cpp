// src/program/serve/runtime.cpp - lifecycle of the persistent engine.
#include "state.hpp"

#include "basalt/core/on_device.hpp"

namespace basalt::program::serve {

void Runtime::start_guard() {
    basalt::platform::stop_memory_watchdog();
    guard_floor = basalt::platform::ram_reserve();
    if (guard_floor > 0) basalt::platform::start_ram_guard(guard_floor);
    disk_floor = guard_floor + (1ull << 30);
    // what a stop's error compares against, to tell the engine's own growth from other programs'
    ready_avail = basalt::platform::available_memory();
    ready_rss = (uint64_t) basalt::platform::process_memory().rss_mib << 20;
    shed_drop = 0;
}

void Runtime::shed_if_low() {
    if (basalt::platform::ram_guard_level() == basalt::platform::RamLevel::Ok) return;
    const uint64_t drop = basalt::platform::ram_guard_drops();
    if (drop == shed_drop) return;
    shed_drop = drop;
    if (conversations) {
        const size_t parked = conversations->size();
        const size_t freed = conversations->clear();
        size_t to_disk = 0;
        for (auto& sc : conversations->take_spilled())
            if (disk.enabled() && !disk_has(sc.live.ids, sc.live.imgs, sc.checkpoints, sc.cvec) &&
                disk.put(std::move(sc), kDiskQueueCap))
                ++to_disk;
        if (freed > 0)
            basalt::log::info("basalt serve", "RAM guard: %zu parked conversations (%zu MiB) leave RAM, %zu of them "
                              "to disk", parked, freed >> 20, to_disk);
    }
    if (const size_t pooled = basalt::core::CheckpointBytes::allocator_type::release_pooled(); pooled > 0)
        basalt::log::info("basalt serve", "RAM guard: freed %zu MiB of checkpoint blocks kept for reuse",
                          pooled >> 20);
    const uint64_t before = basalt::platform::available_memory();
    basalt::platform::release_free_heap();
    const uint64_t after = basalt::platform::available_memory();
    basalt::log::info("basalt serve", "RAM guard: free heap returned to the OS, %lld MiB free now (%+lld MiB)",
                      (long long) (after >> 20), ((long long) after - (long long) before) / (1ll << 20));
}

void Runtime::ckd_finish() {
    const void* dst = ckd.pending();
    if (dst == nullptr) return;
    std::string ce;
    bool ok = ckd.finish(ce);
    if (ckd_peer.pending()) {
        const basalt::core::OnDevice on(ckd_peer_dev);
        ok = ckd_peer.finish(ce) && ok;
    }
    if (ok) return;
    basalt::log::error("basalt serve", "%s", ce.c_str());
    checks.erase(std::remove_if(checks.begin(), checks.end(),
                                [&](const ConvCheckpoint& k) { return k.state_a.data() == dst; }), checks.end());
}

bool Runtime::touch_check(int64_t to, ConvCheckpoint::Source source) {
    for (ConvCheckpoint& c : checks)
        if ((int64_t) c.ids.size() == to) {   // a turn end or the root outranks a stride that landed on it
            c.used = ++check_clock;
            if (source > c.source) c.source = source;
            return true;
        }
    return false;
}

void Runtime::keep_check(ConvCheckpoint&& c) {
    c.used = ++check_clock;
    checks.push_back(std::move(c));
    while ((int) checks.size() > o.prompt_cache) {
        std::vector<uint64_t> stamps;
        stamps.reserve(checks.size());
        for (const ConvCheckpoint& k : checks) stamps.push_back(k.used);
        const size_t victim = basalt::program::conv_cache::eviction_victim(stamps.data(), stamps.size(), o.prompt_cache);
        if (checks[victim].state_a.data() == ckd.pending()) {
            ckd_finish();   // a failed save drops its own checkpoint: look again
            continue;
        }
        checks.erase(checks.begin() + (std::ptrdiff_t) victim);
    }
    trim_check_pool();
}

bool Runtime::capture_check(basalt::model::ModelSession& s, int64_t to, ConvCheckpoint::Source source, const int32_t* ids,
                            std::vector<ImgKey> imgs, ConvCheckpoint& c) {
    c.source = source;
    c.ids.assign(ids, ids + to);
    c.imgs = std::move(imgs);
    basalt::core::RunningState rs;
    if (const basalt::model::Result r = s.running_state(to, true, rs); !r.ok()) {
        err = r.message;
        return false;
    }
    c.state_a.resize(rs.bytes());
    std::memcpy(c.state_a.data(), rs.head.data(), rs.head.size());
    // each card's part on its own device
    const std::vector<basalt::core::SaveSegment> seg{{c.state_a.data() + rs.head.size(), rs.regions}};
    if (!ckd.save(seg, (cudaStream_t) rs.stream, c.state_a.data(), err)) return false;
    if (!rs.peer_regions.empty()) {
        const basalt::core::OnDevice on(rs.peer_device);
        ckd_peer_dev = rs.peer_device;
        const std::vector<basalt::core::SaveSegment> ps{{c.state_a.data() + rs.head.size() + rs.regions_bytes(), rs.peer_regions}};
        if (!ckd_peer.save(ps, (cudaStream_t) rs.peer_stream, c.state_a.data(), err)) return false;
    }
    return true;
}

bool Runtime::slot_turn_checkpoint(int b) {
    BSlot& sl = bs[(size_t) b];
    if (o.prompt_cache <= 0) return true;
    ckd_finish();   // the save before this one lands first
    ConvCheckpoint c;
    if (!capture_check(slot_session(b), sl.p, ConvCheckpoint::kTurnEnd, sl.ids.data(), sl.imgs, c)) return false;
    c.used = ++check_clock;
    // the slot's shortest checkpoint stays (a system prompt other requests may share); its longest gives way to this one
    if (sl.checks.size() > 1) {
        size_t longest = 0;
        for (size_t i = 1; i < sl.checks.size(); ++i)
            if (sl.checks[i].ids.size() > sl.checks[longest].ids.size()) longest = i;
        sl.checks.erase(sl.checks.begin() + (std::ptrdiff_t) longest);
    }
    sl.checks.push_back(std::move(c));
    trim_check_pool();
    return true;
}

namespace {
// A checkpoint's running-state bytes, every card's part.
uint64_t state_bytes(const ConvCheckpoint& c) {
    uint64_t n = c.state_a.size() + c.state_b.size() + c.state_c.size() + c.state_d.size() + c.state_e.size();
    for (const ConvCheckpoint& p : c.stage_parts) n += state_bytes(p);
    return n;
}
}  // namespace

void Runtime::trim_check_pool() {
    for (;;) {   // the pool every chain shares: the solo session's, the one a request in another slot set aside, every slot's
        std::vector<std::vector<ConvCheckpoint>*> chains{&checks};
        if (solo_out != nullptr) chains.push_back(&solo_out->checks);
        for (BSlot& s : bs) chains.push_back(&s.checks);
        std::vector<std::vector<uint64_t>> stamps(chains.size());
        size_t held = 0;
        uint64_t held_bytes = 0;
        for (size_t i = 0; i < chains.size(); ++i) {
            for (const ConvCheckpoint& k : *chains[i]) {
                stamps[i].push_back(k.used);
                held_bytes += state_bytes(k);
            }
            held += chains[i]->size();
        }
        const size_t most = basalt::program::conv_cache::pool_most(o.prompt_cache, o.batch);
        if (held <= most && (check_bytes == 0 || held_bytes <= check_pool * check_bytes)) return;
        const auto [chain, at] = basalt::program::conv_cache::pool_victim(stamps);
        std::vector<ConvCheckpoint>& ch = *chains[chain];
        if (ch[at].state_a.data() == ckd.pending()) {
            ckd_finish();   // a failed save drops its own checkpoint: look again
            continue;
        }
        ch.erase(ch.begin() + (std::ptrdiff_t) at);
    }
}

int Runtime::execute() {
    if (const int code = init_before_input()) return code;
    if (const int code = init_protocol()) return code;
    if (const int code = announce()) return code;
    if (const int code = init_after_input()) return code;
    return run_requests();
}

int Runtime::run_requests() {
    for (;;) {
        const Flow flow = dispatch();
        if (flow == Flow::error) {
            if (basalt::core::gpu_left_running().load()) {   // the teardown would wait for the stuck window forever
                basalt::log::error("basalt serve", "the GPU did not finish a released window: ending without the CUDA teardown");
                std::fflush(stdout);
                std::_Exit(1);
            }
            return 1;
        }
        if (flow == Flow::quit) return shutdown();
    }
}

// Whether the disk tier needs no new copy of a conversation: it is short (below the disk's minimum, or below
// --conversation-cache-min-tokens when that is higher), or a stored prefix of it reaches within
// kDiskStaleTokens of the chain's deepest checkpoint (the point its next turn goes on from; a restore
// from that copy reads the rest again).  Each turn of a growing conversation is not written again in full.
bool Runtime::disk_has(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs, const std::vector<ConvCheckpoint>& cks, bool cvec) {
    if ((int64_t) ids.size() < std::max<int64_t>(disk_min_tokens(), o.conversation_cache_min_tokens)) return true;
    int64_t deepest = 0;
    for (const ConvCheckpoint& c : cks) deepest = std::max<int64_t>(deepest, (int64_t) c.ids.size());
    if (deepest <= 0) deepest = (int64_t) ids.size();
    const auto m = disk.best(ids.data(), ids.size(), imgs, cvec);
    return m.tokens > 0 && m.tokens + kDiskStaleTokens >= deepest;
}

int Runtime::shutdown() {
    if (basalt::core::gpu_left_running().load()) {   // the teardown would wait for unconfirmed GPU work forever
        basalt::log::error("basalt serve", "the GPU did not confirm earlier work: ending without the CUDA teardown");
        std::fflush(stdout);
        std::_Exit(1);
    }
    // the conversations this process holds go to the disk tier before it ends
    if (disk.enabled()) {
        const auto t0 = Clock::now();
        const size_t written = spill_to_disk(true, SIZE_MAX);
        const auto st = disk.stats();
        basalt::log::info("basalt serve", "conversation cache: %zu conversations written to disk at exit in %.0f ms; "
                          "on disk %zu (%.2f GiB); since the start written %zu (%.2f GiB, %.2f GiB with the shared "
                          "files they reuse), read back %zu (%.2f GiB), evicted %zu, dropped %zu", written,
                          std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), st.files,
                          (double) st.bytes / 1073741824.0, st.written, (double) st.written_bytes / 1073741824.0,
                          (double) st.whole_bytes / 1073741824.0, st.restored, (double) st.read_bytes / 1073741824.0,
                          st.evicted, st.dropped);
    }
    at_exit();
    return 0;
}

}  // namespace basalt::program::serve
