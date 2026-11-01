#include "context.hpp"

namespace basalt::program::generate {

int sessions(Context& context) {
    auto& o = *context.o;
    auto& srcs = *context.srcs;
    auto& model_file = *context.model_file;
    auto& pack_files = *context.pack_files;
    auto& split_at = *context.split_at;
    auto& split_devs = *context.split_devs;
    auto& split_same = *context.split_same;
    auto& multi_gpu = *context.multi_gpu;
    auto& pipe_want = *context.pipe_want;
    auto& remote_dev = *context.remote_dev;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& K = *context.K;
    auto& skip = *context.skip;
    auto& profile = *context.profile;
    auto& skip_base = *context.skip_base;
    auto& stage_trim = *context.stage_trim;
    auto& add_foreign = context.add_foreign;
    auto& pool_bytes = *context.pool_bytes;
    auto& wt = *context.wt;
    auto& mrope_host = *context.mrope_host;
    auto& ss = *context.ss;
    auto& sbuf = *context.sbuf;
    auto& main_cs = *context.main_cs;
    auto& ple_table = *context.ple_table;

    // ---- layer split across GPUs: each later stage's own copy of the dense weights, its session and (the last) the
    // head, made on its device before the host arena is mapped (as the drafter below, for the same reason)
    std::vector<std::unique_ptr<GpuStage>> stages;
    for (size_t i = 0; multi_gpu && i < split_devs.size(); ++i) {
        stages.push_back(std::make_unique<GpuStage>());
        GpuStage& st = *stages.back();
        st.dev = split_devs[i];
        double free_gib = 0;
        if (!basalt::core::RemoteExperts::preflight(st.dev, free_gib, err)) {
            return fatal_start_error(err);
        }
        const basalt::core::OnDevice on(st.dev);
        std::set<std::string> skip_s = skip_base;
        uint64_t pool_s = pool_bytes;
        if (stage_trim) {
            const int64_t lb = split_at[i], le = i + 1 < split_at.size() ? split_at[i + 1] : g.n_layers;
            add_foreign(lb, le, skip_s);
            basalt::core::NativeDense::set_layer_range((int) lb, (int) le);
            if (!basalt::core::WeightTable::pool_bytes(pack_files, pool_s, err, &skip_s)) {
                return fatal_start_error(err);
            }
            basalt::log::info("basalt generate", "layer split: CUDA%d loads the dense weights of layers %lld-%lld only",
                         st.dev, (long long) lb, (long long) le - 1);
        } else {
            skip_s = skip;
        }
        void* arena_s = nullptr;
        if (cudaMalloc(&arena_s, pool_s) != cudaSuccess ||
            !st.wt.load(pack_files, arena_s, pool_s, err, skip_s.empty() ? nullptr : &skip_s)) {
            cudaGetLastError();
            size_t free_b = 0, total_b = 0;   // what that card had free
            cudaMemGetInfo(&free_b, &total_b);
            basalt::log::error("basalt generate", "layer split, CUDA%d weights: %s (%llu MiB needed, %llu MiB of %llu "
                                 "MiB free on that card)", st.dev,
                         err.empty() ? "the weight arena does not fit" : err.c_str(),
                         (unsigned long long) (pool_s >> 20), (unsigned long long) (free_b >> 20),
                         (unsigned long long) (total_b >> 20));
            return 1;
        }
        if (!st.dense.load(o.native_shards, st.wt, err, true)) {
            basalt::log::error("basalt generate", "layer split, CUDA%d native dense projections: %s", st.dev,
                         err.c_str());
            return 1;
        }
        basalt::core::Verifier::prepare_hc8(st.wt, g);   // this stage's fused hyper-connection reads need its device's copies
        // THE SESSION AND THE HEAD WAIT FOR THE SPLIT SEARCH.  `session_bytes` prices a stage's session by its LAYER
        // RANGE (the carve), so the sessions are allocated after the search below has set `st.lb`/`st.le`; the last
        // stage's head follows.
        if (cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&st.adapt_stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&st.adapt_ev, cudaEventDisableTiming) != cudaSuccess) {
            basalt::log::error("basalt generate", "layer split, CUDA%d: its streams failed", st.dev);
            return 1;
        }
        // a control vector (the speed projection): its tables on this device too - the stage's layers apply it here
        if (!basalt::kernels::cvec_replicate(err)) {
            basalt::log::error("basalt generate", "layer split, CUDA%d: %s", st.dev, err.c_str());
            return 1;
        }
        // --vision: this device's image-position table (the identity until a picture request), read by every rope
        // kernel its stage runs - set before any of its graphs is captured
        if (o.vision) {
            if (cudaMalloc(&st.mrope, mrope_host.size() * sizeof(int32_t)) != cudaSuccess ||
                cudaMemcpy(st.mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                    cudaSuccess) {
                basalt::log::error("basalt generate", "layer split, CUDA%d: the image position table failed", st.dev);
                return 1;
            }
            basalt::kernels::mrope_table_set(st.mrope);
        }
        // its own PCIe share of the missed experts (the same rule as CUDA0's above: its link is probed)
        st.pcie_frac = o.pcie_frac;
        {
            std::string bursts;
            const double bw = probe_pcie_h2d_gbps(&bursts);
            if (bw > 0.0) st.pcie_frac = pcie_frac_for_gbps(bw, 0.55);
            basalt::log::info("basalt generate", "layer split: CUDA%d PCIe probe %.1f GB/s (best of %s) -> pcie_frac "
                                 "%.2f", st.dev, bw, bursts.c_str(), st.pcie_frac);
        }
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        basalt::log::info("basalt generate", "layer split: CUDA%d holds its weights; %.2f GiB free (its session "
                             "follows the split search)", st.dev, (double) fb / 1073741824.0);
    }
    GpuStage* const last_st = stages.empty() ? nullptr : stages.back().get();

    for (size_t i = 0; i < split_at.size(); ++i)
        if (split_at[i] >= g.n_layers) {
            basalt::log::error("basalt generate", "--layer-split: layer %lld is past the last (%lld)",
                         (long long) split_at[i], (long long) (g.n_layers - 1));
            return 2;
        }
    // the stage that runs a layer (0: CUDA0's)
    auto stage_of = [&](int64_t l) -> int {
        int st = 0;
        while (st < (int) split_at.size() && l >= split_at[(size_t) st]) ++st;
        return st;
    };
    std::vector<std::pair<int32_t, int32_t>> profile_all;   // a layer split: the whole ranking (`profile` keeps CUDA0's share)
    if (multi_gpu) {
        profile_all = profile;
        std::vector<std::pair<int32_t, int32_t>> mine;
        for (const auto& pr : profile) {
            const int st = stage_of(pr.first);
            (st == 0 ? mine : stages[(size_t) st - 1]->profile).push_back(pr);
        }
        profile.swap(mine);
        for (size_t i = 0; i < stages.size(); ++i) {
            stages[i]->lb = split_at[i];
            stages[i]->le = i + 1 < stages.size() ? split_at[i + 1] : g.n_layers;
        }
    }

    uint64_t solo_kv_bytes = 0;   // the solo session's state on CUDA0, reported once it is known to be a session of its own
    // ---- THE ELASTIC KV.  A server on one GPU backs only the first kKvElasticFloor cells of each session's K/V pools at
    // start; the rest of their VRAM goes to the expert cache as its tail, and a request that needs more cells takes
    // them back from that tail (kv_fit below), so a short request decodes with the larger cache.  Not with KV
    // streaming (its pools are a fixed set of VRAM pages) nor with --vram-elastic (that resizes the same tail).
    // With --batch a session's base is smaller: the slots' sessions are many, and an idle or short one would hold its
    // whole base out of the expert cache.
    constexpr int64_t kKvElasticFloor = 32768, kKvElasticFloorBatch = 4096;
    const int64_t kv_base = o.batch > 0 ? kKvElasticFloorBatch : kKvElasticFloor;
    const int64_t kv_floor = o.serve && !multi_gpu && o.kv_resident == 0 && !o.vram_elastic &&
                             o.max_context > kv_base ? kv_base : 0;
    basalt::core::qsa_set_kv_elastic(kv_floor);
    // ---- CUDA0's session, and the stages' sessions: sized to each device's own layer range (the carve).  A
    // stage that runs [lb, le) carves only those layers' GDN rows and QSA pools - before the carve every stage
    // held all 48 layers' state whatever layers it ran, which is the same disease the chunked-QSA-prefill PR
    // fixed in llama.cpp: allocation sized by the whole model instead of the device's own work.
    {
        const basalt::core::OnDevice on0(0);
        const int64_t hi0 = multi_gpu ? split_at[0] : -1;
        if (cudaMalloc(&sbuf, basalt::core::session_bytes(g, o.max_context, K, 0, hi0)) != cudaSuccess) {
            basalt::log::error("basalt generate", "session state allocation failed");
            return 1;
        }
        solo_kv_bytes = basalt::core::session_bytes(g, o.max_context, K, 0, hi0);
        if (basalt::core::session_init(g, o.max_context, K, sbuf, ss, 0, hi0) == 0) {
            basalt::log::error("basalt generate", "session_init failed");
            return 1;
        }
        if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1)
            basalt::log::info("basalt generate", "KV streaming: %lld of %lld cells per QSA layer in VRAM, the K/V in "
                                 "%.2f GiB of pinned RAM",
                         (long long) (ss.qsa_states[ss.qsa_primary()].n_slots * 4),
                         (long long) o.max_context, (double) basalt::core::qsa_kv_host_bytes() / 1073741824.0);
        // the PLE block above built everything but the history, which session_init has just carved
        ss.ple.hist = ss.ple_hist;
        // generate mode starts from an empty sequence, and nothing else zeroes this state before the prompt path or
        // the verifier reads it (--serve zeroes it per request when nothing is reused)
        basalt::core::session_zero(ss, g, nullptr, main_cs);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            basalt::log::error("basalt generate", "zeroing the session state failed");
            return 1;
        }
        if (!ss.ple.ready()) {
            basalt::log::error("basalt generate", "the PLE run is not ready after construction");
            return 1;
        }
        basalt::log::info("basalt generate", "PLE on, table %llu rows of %s",
                     (unsigned long long) ple_table.rows(), o.model.c_str());
    }
    for (size_t i = 0; i < stages.size(); ++i) {
        GpuStage& st = *stages[i];
        const basalt::core::OnDevice on(st.dev);
        void* sbuf_s = nullptr;
        if (cudaMalloc(&sbuf_s, basalt::core::session_bytes(g, o.max_context, K, st.lb, st.le)) != cudaSuccess ||
            basalt::core::session_init(g, o.max_context, K, sbuf_s, st.ss, st.lb, st.le) == 0) {
            basalt::log::error("basalt generate", "layer split, CUDA%d: the session state failed", st.dev);
            return 1;
        }
        basalt::core::LoadProgress::kv(basalt::core::LoadProgress::cuda(st.dev),
                                       basalt::core::session_bytes(g, o.max_context, K, st.lb, st.le), o.max_context);
        const bool last = i + 1 == stages.size();
        const basalt::core::WeightRef* wo_s = st.wt.find("output.weight");
        if (wo_s == nullptr ||
            (last && !st.head.load(o.native_shards, g.n_embd, wo_s->ne1, err))) {
            basalt::log::error("basalt generate", "layer split, CUDA%d head: %s%s", st.dev,
                         wo_s == nullptr ? "output.weight is missing" : err.c_str(),
                         wo_s == nullptr ? "" : vram_free_note().c_str());
            return 1;
        }
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        basalt::log::info("basalt generate", "layer split: CUDA%d holds its weights, session [%lld, %lld)%s; "
                             "%.2f GiB free", st.dev, (long long) st.lb, (long long) st.le,
                     last ? " and the head" : "", (double) fb / 1073741824.0);
    }

    // --batch: every stage carves o.batch more sessions like its own (same layer range and context), one per
    // slot of the batch windows.  Before the expert cache is sized, so `--expert-cache auto` leaves them room.
    // Recommend, never force: a count the engine cannot run is said and adjusted (or batching left off) - the
    // engine still starts and serves one request at a time.
    std::vector<std::vector<SlotSession>> bslot_ss;
    std::vector<int64_t> slot_cap;   // each batch slot's context (cells)
    uint64_t slot_kv_cuda0 = 0;      // the batch slots' sessions on CUDA0 beyond the solo one: VRAM the expert cache does not get
    bool slot0_is_solo = false;      // slot 0 is the solo session (no separate solo state)
    if (o.batch != 0) {
        const int cap = basalt::kernels::kVerifyMaxT;
        const char* off = !o.serve ? "it needs --serve" : o.batch < 2 ? "it needs 2 or more slots"
                        : split_same ? "it needs each stage of a layer split on its own GPU" : nullptr;
        if (off != nullptr) {
            basalt::log::warn("basalt generate", "--batch %d is off: %s", o.batch, off);
            o.batch = 0;
        } else if (o.batch > cap) {
            basalt::log::warn("basalt generate", "--batch %d: a batch window holds at most %d rows, so %d "
                                 "slots", o.batch, cap, cap);
            o.batch = cap;
        }
    }
    if (o.batch > 0 && o.batch_groups > 1 && (stages.empty() || o.batch % o.batch_groups != 0)) {
        basalt::log::warn("basalt generate", "--batch-groups %d needs a layer split and to divide --batch %d; "
                             "one group", o.batch_groups, o.batch);
        o.batch_groups = 1;
    }
    if (o.batch > 0) {
        bslot_ss.resize(1 + stages.size());
        slot_cap.assign((size_t) o.batch, o.max_context);
        slot0_is_solo = stages.empty();   // one GPU: the solo session's state is slot 0's
        int fit = o.batch;   // the slots every stage could carve
        for (size_t k = 0; k < bslot_ss.size(); ++k) {
            const int dev = k == 0 ? 0 : stages[k - 1]->dev;
            const int64_t lo = k == 0 ? 0 : stages[k - 1]->lb;
            const int64_t hi = k == 0 ? (multi_gpu ? split_at[0] : -1) : stages[k - 1]->le;
            const basalt::core::OnDevice on_k(dev);
            uint64_t total = 0;
            for (int b = 0; b < fit; ++b) {
                const uint64_t bytes = basalt::core::session_bytes(g, slot_cap[(size_t) b], K, lo, hi);
                if (k == 0 && b == 0 && slot0_is_solo) {
                    bslot_ss[0].push_back(SlotSession(&ss, SlotDelete{false}));
                    basalt::log::info("basalt generate", "--batch: slot 0 on CUDA%d is the solo session (%.2f GiB)", dev,
                                 (double) bytes / 1073741824.0);
                    continue;
                }
                SlotSession u(new basalt::core::SessionState());
                void* buf = nullptr;
                if (cudaMalloc(&buf, bytes) != cudaSuccess || basalt::core::session_init(g, slot_cap[(size_t) b], K, buf, *u, lo, hi) == 0) {
                    cudaGetLastError();
                    if (buf != nullptr) cudaFree(buf);
                    size_t fb = 0, tb = 0;
                    cudaMemGetInfo(&fb, &tb);
                    basalt::log::warn("basalt generate", "--batch %d: slot %d's session does not fit on CUDA%d "
                                         "(%.2f GiB, %.2f GiB free); %d slot%s", o.batch, b, dev,
                                 (double) bytes / 1073741824.0, (double) fb / 1073741824.0, b, b == 1 ? "" : "s");
                    fit = b;
                    break;
                }
                basalt::core::session_zero(*u, g, nullptr, nullptr);
                bslot_ss[k].push_back(std::move(u));
                total += bytes;
                basalt::log::info("basalt generate", "--batch: slot %d on CUDA%d holds %lld tokens (%.2f GiB)", b, dev,
                             (long long) slot_cap[(size_t) b], (double) bytes / 1073741824.0);
            }
            cudaDeviceSynchronize();
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            basalt::log::info("basalt generate", "--batch: %d slot sessions on CUDA%d (%.2f GiB in all); %.2f GiB free",
                         (int) bslot_ss[k].size(), dev, (double) total / 1073741824.0, (double) fb / 1073741824.0);
        }
        for (auto& v : bslot_ss) if ((int) v.size() > fit) v.resize((size_t) fit);   // the same count on every stage
        if (fit < 2) {
            basalt::log::warn("basalt generate", "--batch is off: not two slot sessions fit (a smaller "
                                 "--max-context, or KV streaming, --kv-resident, makes them smaller)");
            bslot_ss.clear();
            slot_cap.clear();
            o.batch = 0;
            slot0_is_solo = false;
        } else {
            o.batch = fit;
            for (size_t k = 0; k < bslot_ss.size(); ++k)   // every slot session allocated, solo state included
                for (int b = 0; b < o.batch; ++b)
                {
                    const uint64_t sb = basalt::core::session_bytes(g, slot_cap[(size_t) b], K, k == 0 ? 0 : stages[k - 1]->lb,
                                                                    k == 0 ? (multi_gpu ? split_at[0] : -1) : stages[k - 1]->le);
                    const basalt::core::SessionState& u = *bslot_ss[k][(size_t) b];
                    basalt::core::LoadProgress::kv(
                        basalt::core::LoadProgress::cuda(k == 0 ? 0 : stages[k - 1]->dev),
                        sb + basalt::core::session_kv_mapped_bytes(u), slot_cap[(size_t) b], "slot",
                        sb + basalt::core::session_kv_bytes_for(u, u.max_cells));
                }
            slot_cap.resize((size_t) fit);
            if (o.batch_groups > 1 && o.batch % o.batch_groups != 0) o.batch_groups = 1;
            // the sessions' VRAM is the expert cache's: say what it costs (the slots carved on CUDA0 may be fewer than
            // were first carved when a later stage ran out)
            uint64_t kept = 0;
            for (int b = slot0_is_solo ? 1 : 0; b < o.batch; ++b)
                kept += basalt::core::session_bytes(g, slot_cap[(size_t) b], K, 0, multi_gpu ? split_at[0] : -1);
            slot_kv_cuda0 = kept;
            const uint64_t per_slot = basalt::core::session_bytes(g, slot_cap[0], K, 0, multi_gpu ? split_at[0] : -1);
            basalt::log::info("basalt generate", "--batch %d: %.2f GiB of KV cache and state per slot (%lld tokens), "
                                 "%.2f GiB in all; the slots beyond the solo one take %.2f GiB of VRAM on CUDA0 that "
                                 "the expert cache would otherwise hold", o.batch, (double) per_slot / 1073741824.0,
                         (long long) slot_cap[0], (double) (per_slot * (uint64_t) o.batch) / 1073741824.0,
                         (double) kept / 1073741824.0);
        }
    }

    // the drafter's K/V is elastic like the sessions' (MtpDrafter::load keeps it whole with a K/V per batch slot)
    basalt::core::qsa_set_kv_elastic(kv_floor);
    if (!slot0_is_solo)
        basalt::core::LoadProgress::kv(basalt::core::LoadProgress::cuda(0),
                                       solo_kv_bytes + basalt::core::session_kv_mapped_bytes(ss), o.max_context, "",
                                       solo_kv_bytes + basalt::core::session_kv_bytes_for(ss, ss.max_cells));
    // Secure MTP's CUDA0 allocations before the large host arena is registered with both CUDA contexts: the driver can
    // refuse the draft weights after mapping tens of GiB of host pages.
    basalt::core::MtpDrafter mtp;
    if (!o.mtp.empty()) {
        if (o.spec < 2) o.mtp.clear();
        if (!o.mtp.empty()) mtp.set_prompt_len((int64_t) o.tokens.size());
        if (o.batch > 0) mtp.set_stores(o.batch, slot_cap);   // a draft K/V per batch slot
        // the draft layer is the canonical model's MTP head (512 experts) even when the target is pruned,
        // so it always sees the canonical geometry; `static` because MtpDrafter keeps a reference
        static const basalt::core::ModelGeometry draft_geometry{};
        // with a layer split across GPUs the drafter stays on CUDA0 (the fastest card) and reads the last stage's
        // residual rows through a relay
        if (!o.mtp.empty() && !mtp.load(basalt::core::MtpFiles::from(srcs), draft_geometry, ss, pipe_want ? basalt::core::kMtpChainMax : o.spec, err, o.mtp_window)) { basalt::log::error("basalt generate", "%s%s", err.c_str(), vram_free_note().c_str()); return 1; }
        mtp.set_ple_session(&ss);
    }
    basalt::core::qsa_set_kv_elastic(0);   // every later state: whole
    const bool mtp_on = !o.mtp.empty();
    const basalt::core::QsaState* const mtp_kv = mtp_on ? &mtp.kv_state() : nullptr;   // the draft layer's K/V, if any
    // THE HEAD BEFORE THE CACHE, AND BEFORE THE ARENA.  The expert cache takes what is free minus the reserve, so
    // everything allocated after it comes out of the reserve.  Loaded first, the cache is sized around the native
    // head, and the head is not refused after the expert arena has registered tens of GiB of host pages, like the
    // drafter above.
    const basalt::core::WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { basalt::log::error("basalt generate", "output.weight is missing"); return 1; }
    const int64_t n_vocab = wo->ne1;
    // --grammar on: the token texts the grammar walks, read from the pack's tokenizer files
    std::optional<basalt::program::grammar::PackVocab> grammar_vocab;
    if (o.serve && o.grammar) {
        std::vector<uint8_t> vj, tj;
        std::string gerr;
        basalt::program::grammar::PackVocab pv;
        if (!basalt::core::read_range(model_file.range("tokenizer/vocab.json"), vj, gerr) ||
            !basalt::core::read_range(model_file.range("tokenizer/token_type.json"), tj, gerr))
            return fatal_start_error("--grammar on: the pack has no tokenizer files: " + gerr);
        std::vector<int32_t> ends;
        for (int64_t e : o.eos_ids) ends.push_back((int32_t) e);
        if (!basalt::program::grammar::vocab_from_pack(std::string_view((const char*) vj.data(), vj.size()),
                                                       std::string_view((const char*) tj.data(), tj.size()),
                                                       (size_t) n_vocab, ends, pv, gerr))
            return fatal_start_error("--grammar on: " + gerr);
        grammar_vocab = std::move(pv);
    }
    basalt::core::NativeHead native_head;
    basalt::core::NativeHead mtp_head;   // a layer split: the drafter's copy of the head on CUDA0
    if (multi_gpu && mtp_on) {
        if (!mtp_head.load(o.native_shards, g.n_embd, n_vocab, err)) {
            basalt::log::error("basalt generate", "%s%s", err.c_str(), vram_free_note().c_str());
            return 1;
        }
    }
    if (!multi_gpu) {   // a layer split's head is on its last stage
        const auto head_t0 = std::chrono::steady_clock::now();
        if (!native_head.load(o.native_shards, g.n_embd, n_vocab, err)) {
            basalt::log::error("basalt generate", "%s%s", err.c_str(), vram_free_note().c_str());
            return 1;
        }
        basalt::log::info("basalt generate", "native head: %s, %llu bytes, in %.1f s",
                     basalt::ggml_type_name((uint32_t) native_head.type()),
                     (unsigned long long) native_head.weight_bytes(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - head_t0).count());
    }
    std::vector<float> logits((size_t) n_vocab);
    float* d_logits = nullptr;
    if (cudaMalloc(&d_logits, (size_t) n_vocab * 4) != cudaSuccess) {
        (void) cudaGetLastError();
        basalt::log::error("basalt generate", "the logits buffer failed%s", vram_free_note().c_str());
        return 1;
    }
    // Create the additional contexts after MTP has secured CUDA0 memory, but
    // before the host arena maps its expert pages into their address spaces.
    for (int r = 1; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0) {
        double free_gib = 0;
        if (!basalt::core::RemoteExperts::preflight(remote_dev[r], free_gib, err)) {
            return fatal_start_error(err);
        }
        basalt::log::info("basalt generate", "CUDA%d context ready, %.2f GiB free before expert arena registration",
                     remote_dev[r], free_gib);
    }


    context.stages = &stages;
    context.last_st = &last_st;
    context.stage_of = stage_of;
    context.profile_all = &profile_all;
    context.kv_floor = &kv_floor;
    context.bslot_ss = &bslot_ss;
    context.slot_cap = &slot_cap;
    context.slot_kv_cuda0 = &slot_kv_cuda0;
    context.slot0_is_solo = &slot0_is_solo;
    context.mtp = &mtp;
    context.mtp_on = &mtp_on;
    context.mtp_kv = &mtp_kv;
    context.n_vocab = &n_vocab;
    context.grammar_vocab = &grammar_vocab;
    context.native_head = &native_head;
    context.mtp_head = &mtp_head;
    context.d_logits = &d_logits;
    return sources(context);
}

}
