// src/program/serve/conversation.cpp - conversation operations for the persistent engine.
#include "state.hpp"

namespace basalt::program::serve {

uint64_t fnv1a(const void* data, size_t n, uint64_t h) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

ConvStateSizes conv_state_sizes(const basalt::core::ModelGeometry& g, const basalt::core::SessionState& ss) {
    ConvStateSizes z;
    std::string error;
    // a split stage's session owns only its layer range's state (see SessionState's carve note); the geometry
    // and the carve have already passed engine validation
    basalt::core::conversation_session_sizes(g, ss, z, error);
    return z;
}

/// Copies the running state out (this session's carve only).  The caller has synchronized the device.
bool checkpoint_save(ConvCheckpoint& c, const basalt::core::SessionState& ss, const basalt::core::ModelGeometry& g) {
    std::string error;
    if (basalt::core::conversation_checkpoint_save(c, ss, g, error)) return true;
    basalt::log::error("basalt serve", "checkpoint save: %s", error.c_str());   // the caller's ERR has no reason
    return false;
}

/// Puts a checkpoint's running state back; the positional cells below it are the caller's to guarantee.
bool checkpoint_restore(const ConvCheckpoint& c, basalt::core::SessionState& ss, const basalt::core::ModelGeometry& g) {
    std::string error;
    if (basalt::core::conversation_checkpoint_restore(c, ss, g, error)) return true;
    basalt::log::error("basalt serve", "checkpoint restore: %s", error.c_str());
    return false;
}

/// The identity parked conversations on disk are written under (conversation_store.hpp): the model file
/// (its size and samples of its bytes) and the settings that change a state's bytes - the K/V type, the rope, the draft
/// layer, the image encoder, control vectors.  A file of another identity is never restored.
uint64_t conversation_identity(const Options& o, const basalt::core::ModelGeometry& g) {
    using basalt::core::conversation_store_hash;
    char num[512];
    std::snprintf(num, sizeof num, " rope %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g cvec %d %d %d %d", o.rope_scale,
                  o.rope_freq_base, o.rope_freq_scale, o.yarn_orig_ctx, o.yarn_ext_factor, o.yarn_attn_factor,
                  o.yarn_beta_fast, o.yarn_beta_slow, o.cvec_first, o.cvec_last, o.cvec_mode, o.cvec_single);
    std::string text = std::string("basalt conversation state kv ") + o.kv + " rope " +
                       o.rope_scaling + num + " mtp " + o.mtp + " mmproj " + o.vision_mmproj;
    for (const auto& [path, scale] : o.cvec_files) text += " cvec " + path + " " + std::to_string(scale);
    uint64_t h = conversation_store_hash(text.data(), text.size(), 1);
    const int64_t geo[] = {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_v_heads, g.n_head_kv,
                           g.head_dim, g.idx_key_dim, g.n_expert, g.n_ff};
    h = conversation_store_hash(geo, sizeof geo, h);
    uint64_t size = 0;
    basalt::platform::file_size(o.model, size);
    h = conversation_store_hash(&size, sizeof size, h);
    const int fd = basalt::platform::open_fd_utf8(o.model);
    std::vector<uint8_t> buf((size_t) 64 << 10);
    const uint64_t n = std::min<uint64_t>(buf.size(), size);
    for (int i = 0; i < 16 && fd >= 0 && n > 0; ++i) {
        const uint64_t off = (size - n) / 15 * (uint64_t) i;
        if (!basalt::platform::pread_all(fd, buf.data(), n, off)) break;
        h = conversation_store_hash(buf.data(), (size_t) n, h);
    }
    basalt::platform::close_fd(fd);
    return h;
}

// finishes the in-flight host copy; a failed one drops its checkpoint
void Runtime::ckd_finish() {
    const void* dst = ckd.pending();
    if (dst == nullptr) return;
    std::string ce;
    if (ckd.finish(ce)) return;
    basalt::log::error("basalt serve", "%s", ce.c_str());
    checks.erase(std::remove_if(checks.begin(), checks.end(),
                                [&](const ConvCheckpoint& k) { return k.gdn.data() == dst; }), checks.end());
}

// Whether the disk tier needs no new copy of a conversation: it is short, or a stored prefix of it reaches
// within kDiskStaleTokens of the chain's deepest checkpoint (the point its next turn goes on from; a restore
// from that copy reads the rest again).  Each turn of a growing conversation is not written again in full.
bool Runtime::disk_has(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs, const std::vector<ConvCheckpoint>& cks, bool cvec) {
    if ((int64_t) ids.size() < kDiskMinTokens) return true;
    int64_t deepest = 0;
    for (const ConvCheckpoint& c : cks) deepest = std::max<int64_t>(deepest, (int64_t) c.ids.size());
    if (deepest <= 0) deepest = (int64_t) ids.size();
    const auto m = disk.best(ids.data(), ids.size(), imgs, cvec);
    return m.tokens > 0 && m.tokens + kDiskStaleTokens >= deepest;
}

// A capture of a conversation's state, or a read of one from disk, takes `need` bytes of RAM: only while the RAM
// guard reads its level as ok, and with `floor` left free after it (the disk tier's floor sits above the guard's).
bool Runtime::capture_admit(uint64_t need, uint64_t floor) {
    return basalt::platform::ram_guard_level() == basalt::platform::RamLevel::Ok &&
           basalt::core::conversation_memory_admit(basalt::core::conversation_available_memory(), need, floor);
}

// Save only on a switch/rewind, not on each continuing request. No graph
// addresses change: all parked images live in ordinary host vectors.
// `kept`: the tokens of the outgoing conversation the session goes on holding (the request continues from its
// own chain), 0 when another conversation takes the session
bool Runtime::park_current(size_t held, int64_t kept) {
    ckd_finish();
    if ((!conversations->enabled() && !disk.enabled()) || !live_ok || live.empty()) return true;
    // the disk tier takes the image when there is no RAM tier or it cannot make room, unless the conversation is
    // short or stored already, or the request goes on from the chain's deepest checkpoint (the same
    // conversation's next turn: all it drops is the last reply)
    bool to_disk = !conversations->enabled();
    int64_t deepest = 0;
    for (const ConvCheckpoint& c : checks) deepest = std::max<int64_t>(deepest, (int64_t) c.ids.size());
    const bool disk_wants = disk.enabled() && !(kept > 0 && kept >= deepest) &&
                            !disk_has(live, live_imgs, checks, cvec_cached);
    if (to_disk && !disk_wants) return true;
    // before make_room evicts oldest-first, the copies of this conversation a turn back go (they hold nothing
    // the outgoing chain does not, apart from the tail this conversation rewrote)
    if (const size_t dropped = conversations->drop_superseded(live, live_imgs, checks, cvec_cached))
        basalt::log::info("basalt serve", "conversation cache: dropped %zu superseded cop%s of this "
                     "conversation; parked=%zu", dropped, dropped == 1 ? "y" : "ies", conversations->size());
    // A layer split parks one image per stage: the first stage's (with the draft layer's K/V, the drafter
    // runs on CUDA0) and one per later stage (without it).  The checkpoints are moved apart into their
    // stage parts for the capture and put back together whichever way this ends (no running state copied).
    const size_t n_st = stages.size();
    const basalt::core::QsaState* draft0 = mtp_kv;
    auto draft_of = [&](size_t) -> const basalt::core::QsaState* { return nullptr; };
    basalt::core::ConversationCheckpointSplit cs;
    if (n_st > 0) cs = basalt::core::conversation_checkpoints_split(std::move(checks), n_st);
    struct MergeBack {
        basalt::core::ConversationCheckpointSplit& cs; std::vector<ConvCheckpoint>& checks; bool on;
        ~MergeBack() { if (on && !basalt::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
    } merge_back{cs, checks, n_st > 0};
    const basalt::core::ConversationView view{live, live_imgs, n_st > 0 ? cs.stage0 : checks, cvec_cached};
    auto reuse = conversations->take_reuse(rsp);
    std::vector<basalt::core::ConversationKvReuse> stage_reuse = std::move(reuse.stages);
    stage_reuse.resize(n_st);
    reuse.stages.clear();
    // the later stages' estimates, each with its own retained K/V when it still fits
    size_t stage_estimate = 0, stage_retained = 0;
    for (size_t k = 0; k < n_st; ++k) {
        auto& st = stages[k];
        const basalt::core::OnDevice on(st->dev);
        const basalt::core::ConversationView view_k{live, live_imgs, cs.parts[k], cvec_cached};
        size_t b = 0;
        if (!basalt::core::conversation_snapshot_bytes(view_k, st->ss, g, draft_of(k), b, err)) {
            basalt::log::warn("basalt serve", "conversation cache: skip parking (stage CUDA%d: %s)",
                         st->dev, err.c_str());
            err.clear();
            return true;
        }
        if (!stage_reuse[k].kv.empty()) {
            size_t r = 0;
            if (basalt::core::conversation_snapshot_capture_bytes(stage_reuse[k], view_k, st->ss, g, draft_of(k),
                                                                  r, err)) {
                b = r;
                stage_retained += stage_reuse[k].bytes();
            } else {
                stage_reuse[k] = {};
                err.clear();
            }
        }
        stage_estimate += b;
    }
    size_t estimate = 0;
    if (!basalt::core::conversation_snapshot_bytes(view, *rsp, g, draft0, estimate, err)) {
        basalt::log::warn("basalt serve", "conversation cache: skip parking (%s)", err.c_str());
        err.clear(); // A recoverable miss must not poison the batched draft prefill's error channel.
        return true;
    }
    const size_t fresh_estimate = estimate;
    if (!reuse.kv.empty() && !basalt::core::conversation_snapshot_capture_bytes(
            reuse, view, *rsp, g, draft0, estimate, err)) {
        reuse = {};
        estimate = fresh_estimate;
        err.clear();
    }
    estimate += stage_estimate;
    // A park carrying its own retained K/V replaces memory the cache
    // already held, so capacity is make_room's call - it runs next either
    // way, and put()'s accounting still bounds the budget. The with-reuse
    // estimate must stay uncapped: it counts the retained buffers'
    // capacity and directories, and put() charges that same true size -
    // a capped figure would under-evict and overfill the budget.
    const bool ram_room = !to_disk && conversations->make_room(estimate, held);
    if (!to_disk && !ram_room && disk_wants) to_disk = true;   // larger than the RAM tier: the disk tier takes it
    if (to_disk ? disk.queued_bytes() + estimate > kDiskQueueCap : !ram_room) {
        basalt::log::info("basalt serve", "conversation cache: skip parking (snapshot %zu MiB exceeds available budget)",
                     estimate >> 20);
        return true;
    }
    const auto t0 = Clock::now();
    try {
        const uint64_t floor = std::max<uint64_t>((uint64_t) o.conversation_cache_min_free_mib * 1024 * 1024,
                                                  disk.enabled() ? disk_floor : 0);
        const size_t retained = reuse.bytes() + stage_retained;
        const size_t additional = estimate > retained ? estimate - retained : 0;
        if (!capture_admit(additional, floor)) {
            basalt::log::info("basalt serve", "conversation cache: skip parking (physical RAM admission; need %zu MiB plus %lld MiB floor, or telemetry unavailable, or RAM below the guard)",
                         additional >> 20, (long long) o.conversation_cache_min_free_mib);
            return true;
        }
        basalt::core::SavedConversation image;
        size_t reused_bytes = 0;
        if (!basalt::core::conversation_snapshot_save(image, view, *rsp, g, draft0, err,
                std::move(reuse), &reused_bytes)) return false;
        for (size_t k = 0; k < n_st; ++k) {   // the later stages
            auto& st = stages[k];
            const basalt::core::OnDevice on(st->dev);
            if (cudaDeviceSynchronize() != cudaSuccess) { err = "stage sync failed"; return false; }
            const basalt::core::ConversationView view_k{live, live_imgs, cs.parts[k], cvec_cached};
            basalt::core::SavedConversation part;
            if (!basalt::core::conversation_snapshot_save(part, view_k, st->ss, g, draft_of(k), err,
                    std::move(stage_reuse[k]), &reused_bytes))
                return false;
            image.stage_images.push_back(std::move(part));
        }
        if (!basalt::core::conversation_memory_admit(basalt::core::conversation_available_memory(), 0, floor)) {
            basalt::log::info("basalt serve", "conversation cache: skip parking (physical RAM floor after capture, or telemetry unavailable)");
            return true;
        }
        const size_t snapshot_bytes = image.bytes();
        const bool stored = to_disk ? disk.put(std::move(image), kDiskQueueCap)
                                    : conversations->put(std::move(image), held);
        basalt::log::info("basalt serve", "conversation cache: %s %zu tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu snapshot_bytes=%zu reused_kv_bytes=%zu%s",
                     stored ? "parked" : "skipped", live.size(),
                     std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                     conversations->size(), conversations->bytes(), conversations->evictions(), snapshot_bytes, reused_bytes,
                     to_disk ? " (to disk)" : "");
        // what the RAM tier evicted to make room goes to the disk tier
        for (auto& s : conversations->take_spilled())
            if (!disk_has(s.live.ids, s.live.imgs, s.checkpoints, s.cvec)) disk.put(std::move(s), kDiskQueueCap);
    } catch (const std::bad_alloc&) {
        // The active state has not been touched. Continue with normal
        // prompt processing rather than killing a serving process.
        basalt::log::warn("basalt serve", "conversation cache: allocation failed; skip parking");
    }
    return true;
}

std::vector<ImgKey> Runtime::imgs_below(const std::vector<ImgKey>& all, int64_t L) {
    std::vector<ImgKey> v;
    for (const ImgKey& k : all) if (k.start < L) v.push_back(k);
    return v;
}

bool Runtime::gpu_sync_ok() {
    const cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) return true;
    ckpt_why = std::string(": the GPU stopped responding before it (") + cudaGetErrorString(e) +
               ") - a GPU hang";
    return false;
}

// a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed.  `parts`: the stages'
// states saved at L (a split's mid-prompt checkpoint); without, they are read now (everything is at L)
bool Runtime::checkpoint_at(int64_t L, std::vector<ConvCheckpoint>* parts, bool defer) {
    ckpt_why.clear();
    ckd_finish();
    if (o.prompt_cache <= 0 || L < 1) return true;
    for (ConvCheckpoint& c : checks)
        if ((int64_t) c.ids.size() == L) { c.used = ++check_clock; return true; }
    ConvCheckpoint c;
    c.ids.assign(cur.begin(), cur.begin() + L);
    c.imgs = imgs_below(req_imgs, L);
    if (parts != nullptr) {
        if (parts->size() != stages.size() + 1) return false;
        c.gdn = std::move((*parts)[0].gdn);
        c.ple = std::move((*parts)[0].ple);
        c.tails = std::move((*parts)[0].tails);
        c.dead = std::move((*parts)[0].dead);
        c.block_pos = std::move((*parts)[0].block_pos);
        for (size_t i = 1; i < parts->size(); ++i) c.stage_parts.push_back(std::move((*parts)[i]));
    } else {
        if (!gpu_sync_ok()) return false;
        std::string ce;
        if (defer && stages.empty()) {
            if (!ckd.save(c, *rsp, g, ce)) { basalt::log::error("basalt serve", "checkpoint save: %s", ce.c_str()); return false; }
        } else if (!checkpoint_save(c, *rsp, g)) return false;
        for (auto& st : stages) {   // a layer split's later stages: their sessions' part
            const basalt::core::OnDevice on(st->dev);
            ConvCheckpoint part;
            part.ids = c.ids;
            if (!gpu_sync_ok() || !checkpoint_save(part, st->ss, g)) return false;
            c.stage_parts.push_back(std::move(part));
        }
    }
    c.used = ++check_clock;
    checks.push_back(std::move(c));
    while ((int) checks.size() > o.prompt_cache) {
        std::vector<uint64_t> stamps;
        stamps.reserve(checks.size());
        for (const ConvCheckpoint& k : checks) stamps.push_back(k.used);
        const size_t victim = basalt::program::conv_cache::eviction_victim(stamps.data(), stamps.size(),
                                                                           o.prompt_cache);
        if (checks[victim].gdn.data() == ckd.pending()) ckd_finish();
        checks.erase(checks.begin() + (std::ptrdiff_t) victim);
    }
    return true;
}

// The conversations the engine holds go to the disk tier when no stored copy reaches within kDiskStaleTokens of
// them: the RAM tier's (at exit), the solo session's and every idle slot's, each with its draft layer's K/V.  Between
// requests (no slot decoding) this bounds what a crash or a kill loses; a slot's draft K/V is copied into the main
// store for the capture, which every request refills from its own store first, so between requests that runs only
// when the solo session holds no conversation of its own there.  At most `limit` images; returns those queued.
size_t Runtime::spill_to_disk(bool at_exit, size_t limit) {
    ckd_finish();   // a turn checkpoint's state may still be on its way to the host
    size_t written = 0;
    const uint64_t floor = std::max<uint64_t>((uint64_t) o.conversation_cache_min_free_mib << 20, disk_floor);
    if (at_exit)
        while (conversations->size() > 0) {
            basalt::core::SavedConversation s = conversations->take(conversations->size() - 1);
            if (!disk_has(s.live.ids, s.live.imgs, s.checkpoints, s.cvec) && disk.put(std::move(s), kDiskQueueCap))
                ++written;
        }
    auto spill = [&](const basalt::core::SessionState& s, const std::vector<int32_t>& ids,
                     const std::vector<ImgKey>& imgs, const std::vector<ConvCheckpoint>& cks, bool cvec) {
        const basalt::core::ConversationView view{ids, imgs, cks, cvec};
        size_t bytes = 0;
        std::string se;
        if (!basalt::core::conversation_snapshot_bytes(view, s, g, mtp_kv, bytes, se)) {
            basalt::log::warn("basalt serve", "conversation cache: not written to disk (%s)", se.c_str());
            return;
        }
        if (at_exit) disk.flush();   // one image in RAM at a time
        if (!capture_admit(bytes, floor)) {
            basalt::log::info("basalt serve", "conversation cache: %zu tokens not written to disk (physical RAM "
                              "admission)", ids.size());
            return;
        }
        try {
            const auto t0 = Clock::now();
            basalt::core::SavedConversation image;
            if (!basalt::core::conversation_snapshot_save(image, view, s, g, mtp_kv, se)) {
                basalt::log::warn("basalt serve", "conversation cache: not written to disk (%s)", se.c_str());
                return;
            }
            if (disk.put(std::move(image), kDiskQueueCap)) {
                ++written;
                if (!at_exit)
                    basalt::log::info("basalt serve", "conversation cache: %zu tokens to disk between requests "
                                      "(captured in %.0f ms)", ids.size(),
                                      std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            }
        } catch (const std::bad_alloc&) {
            basalt::log::warn("basalt serve", "conversation cache: allocation failed; not written to disk");
        }
    };
    std::string se;
    if (mtp_on && !mtp.use_main(se)) se.clear();
    const bool slot0_cached = slot0_is_solo && !bs.empty() && bs[0].cached;
    const bool solo_holds = live_ok && !live.empty() && !slot0_cached;
    if (solo_holds && !disk_has(live, live_imgs, checks, cvec_cached)) spill(ss, live, live_imgs, checks, cvec_cached);
    if (solo_holds && !at_exit && batch_spec) return written;   // the main store holds its draft K/V
    for (int b = 0; b < (int) bs.size() && written < limit; ++b) {
        BSlot& sl = bs[(size_t) b];
        if (!sl.cached || sl.active || sl.partial || disk_has(sl.ids, sl.imgs, sl.checks, sl.cvec)) continue;
        if (batch_spec && !mtp.copy_store(b, mtp.main_store(), (int64_t) sl.ids.size(), se)) {
            se.clear();
            continue;
        }
        make_view(rviews[(size_t) b], b);
        spill(rviews[(size_t) b], sl.ids, sl.imgs, sl.checks, sl.cvec);
    }
    if (at_exit) disk.flush();
    return written;
}

int Runtime::init_conversations() {
    live_ok = false;

    check_clock = 0;   // the checkpoints' LRU clock; creation and every use advance it

    {
        ConvCheckpoint warm;   // one set of pooled blocks, page-faulted now instead of on the first request
        const ConvStateSizes zw = conv_state_sizes(g, ss);
        if (o.prompt_cache > 0) {
            warm.gdn.resize(zw.gdn);
            std::memset(warm.gdn.data(), 0, warm.gdn.size());
            if (ss.ple_hist) { warm.ple.resize(zw.ple); std::memset(warm.ple.data(), 0, warm.ple.size()); }
        }
        if (o.prompt_cache > 0 && !multi_gpu) {
            std::string ce;
            if (!ckd.reserve(zw.gdn, ce))
                basalt::log::warn("basalt serve", "deferred checkpoint copy off: %s", ce.c_str());
        }
    }

    cvec_cached = true;   // the control vector's state the live session and the checkpoints were read with
    // the session the request path reads, parks and restores: the solo one, or (a --batch request for a slot other
    // than 0) a view of it carrying that slot's state, so the prompt is read in the slot itself
    // (one view per slot, each at a fixed address: the verifier keeps solo window graphs per view)
    rviews = std::vector<basalt::core::SessionState>((size_t) std::max(o.batch, 1));
    rsp = &ss;
    conversations.emplace(
        o.prompt_cache > 0 ? (size_t) o.conversation_cache_mib * 1024 * 1024 : 0,
        (size_t) o.conversation_cache_slots);

    disk_floor = 0;                              // the RAM a capture or a read leaves free (set below)

    if (o.conversation_cache_disk_mib > 0) {
        std::string de;
        const auto t0 = Clock::now();
        if (o.prompt_cache <= 0 || !stages.empty()) {
            basalt::log::warn("basalt serve", "conversation cache on disk off: %s",
                              o.prompt_cache <= 0 ? "--prompt-cache 0" : "a layer split");
        } else if (!disk.open(o.conversation_cache_dir, conversation_identity(o, g),
                              (uint64_t) o.conversation_cache_disk_mib << 20, de)) {
            basalt::log::warn("basalt serve", "conversation cache on disk off: %s", de.c_str());
        } else {
            conversations->set_spill(true);
            const auto st = disk.stats();
            basalt::log::info("basalt serve", "conversation cache on disk: %s, %zu conversations of this model "
                              "(%.2f GiB), budget %lld MiB, indexed in %.0f ms", o.conversation_cache_dir.c_str(),
                              st.files, (double) st.bytes / 1073741824.0, (long long) o.conversation_cache_disk_mib,
                              std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        }
    }

    pp_total = 0;
    pp_from = 0;
    pp_next_check = 0;
    // the end of the part being read when the part loop saves a checkpoint there (else -1): the periodic
    // checkpoint skips that cell, the loop's copy runs beside what follows
    pp_part_ckpt = -1;
    // the position the prompt pass has read up to (a chunk's or a window's end): what a request cancelled mid-read
    // reports as read, instead of the whole prompt
    pp_reached = 0;
    pp_t0 = Clock::now();

    part_next = std::vector<int64_t>(stages.size() + 1, INT64_MAX);

    // live progress inside a chunk: PL <estimated position> <prompt tokens> <ms> <tokens/s>, at most twice a second
    sp.on_layer = [&, last_ms = 0.0](int64_t layers_done, int64_t n_layers, int64_t T, int64_t p0) mutable {
        if (pp_total <= 0 || layers_done <= 0 || n_layers <= 0) return;
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
        if (ms - last_ms < 500.0 && ms >= last_ms) return;
        last_ms = ms;
        const int64_t at = p0 + T * layers_done / n_layers;
        std::printf("PL %lld %lld %.0f %.1f\n", (long long) at, (long long) pp_total, ms,
                    ms > 0.0 ? 1000.0 * (double) (at - pp_from) / ms : 0.0);
        std::fflush(stdout);
    };
    sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
        std::vector<int32_t> nxt((size_t) T);
        for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) cur[(size_t) (p0 + t + 1)];
        // batched through the prompt path when it can (one GPU: a layer split's drafter is on the last stage)
        const bool batched = mtp_on && !multi_gpu && sp.draft_kv(mtp, R_rows, nxt.data(), T, p0, e);
        if (!e.empty() || (mtp_on && !batched && !mtp.prefill(R_rows, nxt.data(), T, p0, e))) return false;
        // progress for the server window: PP <position reached> <prompt tokens> <ms> <fresh tokens/s>
        const int64_t done = p0 + T;
        pp_reached = done;
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
        std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) pp_total, ms,
                    ms > 0.0 ? 1000.0 * (double) (done - pp_from) / ms : 0.0);
        basalt::core::progress_at("reading the prompt (batched), done up to token", done);
        basalt::core::progress_beat();
        std::fflush(stdout);
        if (o.prompt_cache_every > 0 && done >= pp_next_check) {
            if (done == pp_part_ckpt && !multi_gpu) {
                pp_next_check = done + o.prompt_cache_every;
                return true;
            }
            bool saved = false;
            if (multi_gpu) {   // the stages' parts, saved when each of them read this chunk
                std::vector<ConvCheckpoint> parts;
                {
                    std::lock_guard<std::mutex> lk(part_mu);
                    auto it = part_at.find(done);
                    if (it != part_at.end()) parts = std::move(it->second);
                    part_at.erase(part_at.begin(), part_at.upper_bound(done));
                }
                bool complete = parts.size() == stages.size() + 1;
                for (const ConvCheckpoint& k : parts) complete = complete && !k.gdn.empty();
                saved = !complete || checkpoint_at(done, &parts);   // an incomplete set: no checkpoint here
            } else {
                saved = checkpoint_at(done);
            }
            if (!saved) { e = "saving a conversation checkpoint failed" + ckpt_why; return false; }
            pp_next_check = done + o.prompt_cache_every;
        }
        return true;
    };
    if (multi_gpu) {   // the batched prompt is reported by its last stage (the drafter's rows are there)
        stages.back()->sp.on_chunk = std::move(sp.on_chunk);
        sp.on_chunk = nullptr;
        for (size_t i = 0; i <= stages.size(); ++i) {
            basalt::prefill::Prefill& stage_sp = i == 0 ? sp : stages[i - 1]->sp;
            basalt::core::SessionState& stage_ss = i == 0 ? ss : stages[i - 1]->ss;
            stage_sp.on_stage_chunk = [&, i](int64_t done, std::string& e) -> bool {
                if (o.prompt_cache <= 0 || o.prompt_cache_every <= 0 || done < part_next[i]) return true;
                part_next[i] = done + o.prompt_cache_every;
                ConvCheckpoint part;   // this stage's state at `done` (its stream is synchronized)
                part.ids.assign(cur.begin(), cur.begin() + done);
                if (!checkpoint_save(part, stage_ss, g)) { e = "saving a checkpoint part failed"; return false; }
                std::lock_guard<std::mutex> lk(part_mu);
                auto& v = part_at[done];
                v.resize(stages.size() + 1);
                v[i] = std::move(part);
                return true;
            };
        }
    }
    drive.d.plan = ver.plan_sink();
    drive.d.pcie_num = std::max(0, std::min(256, (int) (o.pcie_frac * 256.0 + 0.5)));
    return 0;
}

}  // namespace basalt::program::serve
