#include "context.hpp"
#include "run.hpp"

namespace basalt::program::generate {

int run(int argc, char** argv) {
    Context context;

    Options o;
    std::string ppl_base, ppl_out;
    if (const auto code = cli::start(argc, argv, o, ppl_base, ppl_out)) return *code;
    // ---- the GPUs: chosen before any CUDA call, which then sees only them, the main card as device 0
    std::vector<basalt::program::GpuCard> chosen_cards;
    if (const auto code = cli::select_gpus(o, chosen_cards, basalt::kernels::kDmMaxX + 1)) return *code;
    // ---- where every loader reads from: the sections of the .basalt file
    if (const auto code = cli::prepare_model_process(o)) return *code;
    basalt::core::ModelSources srcs;
    basalt::core::ModelFile model_file;
    {
        std::string merr;
        if (!model_file.open(o.model, merr) || !model_file.load_meta(merr) || !model_file.register_gguf_view(merr)) {
            return fatal_start_error("--model: " + merr);
        }
        basalt::core::model_sources(model_file, srcs);
        if (srcs.dense.empty() || srcs.dense_index.empty() || srcs.experts.empty() || srcs.experts_layout.empty()) {
            return fatal_start_error(o.model + " lacks the dense, dense.index, experts or experts.layout section");
        }
        // the source tensors (embedding, PLE table, native projections, head) are served through the GGUF view of the file
        o.native_shards = {o.model};
        if (!srcs.mtp_dense.empty() && !srcs.mtp_dense_index.empty() && !srcs.mtp_experts.empty()) o.mtp = o.model;
        if (!srcs.profile.empty()) o.expert_profile = o.model + "#profile";
        // the 8-bit hyper-connection copies, unless this run is making them
        if (!srcs.hc8.empty() && o.dump_hc8.empty()) {
            std::map<std::string, basalt::core::Hc8Copy> table;
            if (!basalt::core::hc8_read(srcs.hc8, table, merr)) {
                return fatal_start_error(o.model + ": " + merr);
            }
            basalt::core::hc8_set_table(std::move(table));
        }
    }
    const basalt::core::PackFiles pack_files = basalt::core::PackFiles::from(srcs);
    const basalt::core::ExpertFiles expert_files = basalt::core::ExpertFiles::from(srcs);
    // ---- two or more cards: what they do together is priced and the cheapest mechanism the engine has runs (one card:
    // nothing to choose, and no other card is touched)
    if (chosen_cards.size() >= 2) {
        basalt::program::MgModel mm;
        mm.pairs = (int64_t) mm.layers * 512;
        mm.blob_bytes = srcs.experts.bytes > 0 ? srcs.experts.bytes / (uint64_t) mm.pairs : mm.blob_bytes;
        mm.main_overhead = srcs.dense.bytes + (7ull << 29);
        mm.peer_overhead = 5ull << 28;
        const basalt::program::MgPlan mp = basalt::program::plan_multi_gpu(chosen_cards, mm);
        basalt::log::info("basalt", "%s", mp.line.c_str());
        if (mp.pick == basalt::program::Mechanism::LayerSplit && o.peer_device >= 1) {
            o.layer_split = "auto";
            o.peer_devices.clear();
            o.peer_device = -1;
        }
    }
    // the peer card takes expert rows of a long prompt chunk only when it is as fast as the main one: a slower card costs the
    // chunk more than the experts it saves the main card's link, so it keeps to the short chunks it helps
    const bool peer_streamed_share = chosen_cards.size() >= 2 && basalt::program::cards_alike(chosen_cards[0], chosen_cards[1]);
    // the prompt-time layer split over the expert tier cards (PromptStage), and the most stages it may use, the main
    // card's included: every card on two cards or on three or more alike ones, none on three or more unlike cards
    // until measured on such a set.  With --batch a read splits only while no slot decodes.
    bool split_want = o.serve && o.peer_device >= 1;
    bool split_alike = true;
    for (size_t i = 1; i < chosen_cards.size(); ++i)
        split_alike = split_alike && basalt::program::cards_alike(chosen_cards[0], chosen_cards[i]);
    const int split_cap = o.peer_devices.size() <= 1 ? 2 : split_alike ? (int) o.peer_devices.size() + 1 : 1;
    basalt::core::set_peer_portable(o.peer_device >= 1 || !o.layer_split.empty());   // multi-GPU: the Portable flag on mapped host buffers only with a peer device (before any allocation)
    // Chunks below this take the routed-only path with the peer card computing its experts' rows; from it on
    // every expert streams through the ring.  The routed-only path costs ~0.3 ms per token, streaming ~0.12 ms
    // plus the time to move the experts the primary lacks over its link; with --peer-exclusive that includes the
    // peer-only experts, staged through the host, so streaming loses until later.
    auto set_stream_all_min = [&]() {
        if (o.serve && o.peer_device >= 1 && o.peer_prefill_rows != 0 && peer_streamed_share)
            basalt::prefill::Prefill::set_stream_all_min(o.peer_exclusive ? 4096 : 2560);
    };
    set_stream_all_min();
    if (o.serve && o.conversation_cache_disk_mib > 0 && o.conversation_cache_dir.empty()) {
        basalt::log::error("basalt", "--conversation-cache-disk-mib needs --conversation-cache-dir");
        return 2;
    }
    if (o.serve && o.conversation_cache_mib > 0 && (o.prompt_cache == 0 || o.conversation_cache_slots == 0))
        basalt::log::warn("basalt serve", "conversation caching is disabled by %s",
                     o.prompt_cache == 0 ? "--prompt-cache 0" : "--conversation-cache-slots 0");
    // parking with --layer-split saves every stage (SavedConversation::stage_images)
    // Layer split (multi-GPU): the later stages run layers [K_i, K_i+1) on their own GPUs (--split-device, default
    // the next visible ones); "auto" places the K from each GPU's free VRAM once the weights are in (below).  KV
    // streaming and images (the server's GPU encoder) run across a split, also with parking and --batch.  Across
    // GPUs, not supported: control vectors, the helper caches (--expert-cache-remote), and lending cache slots to the
    // prompt path (each stage's prompt path has its own buffers).
    std::vector<int64_t> split_at;
    std::vector<int> split_devs;
    bool split_auto = false, split_same = false;
    if (!o.layer_split.empty()) {
        int n_dev = 1;
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) n_dev = 1;
        cudaGetLastError();
        auto ints = [](const std::string& str, auto& out) -> bool {
            using V = typename std::decay_t<decltype(out)>::value_type;
            size_t a = 0;
            while (a < str.size()) {
                size_t b = str.find(',', a);
                if (b == std::string::npos) b = str.size();
                const std::string t = str.substr(a, b - a);
                if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) return false;
                out.push_back((V) std::atoll(t.c_str()));
                a = b + 1;
            }
            return !out.empty();
        };
        split_auto = o.layer_split == "auto";
        bool ok = o.serve && (split_auto || ints(o.layer_split, split_at));
        if (ok && !o.split_device.empty()) ok = ints(o.split_device, split_devs);
        else if (ok)
            for (int d = 1; d < n_dev && (split_auto || split_devs.size() < split_at.size()); ++d) split_devs.push_back(d);
        if (ok && !split_auto && split_devs.empty() && split_at.size() == 1) split_devs.push_back(0);   // one GPU
        split_same = ok && split_devs.size() == 1 && split_devs[0] == 0 && !split_auto;
        if (split_same && o.conversation_cache_mib > 0 && o.conversation_cache_slots > 0 && o.prompt_cache > 0) {
            basalt::log::error("basalt serve", "conversation parking does not support a layer split on one GPU "
                                 "(--split-device 0); disable parking with --conversation-cache-mib 0");
            return 2;
        }
        if (ok && split_auto && split_devs.empty()) {
            basalt::log::info("basalt generate", "--layer-split auto: one GPU visible, so no split");
            o.layer_split.clear();
            split_auto = false;
        } else if (ok) {
            ok = (split_auto || split_at.size() == split_devs.size()) && split_devs.size() < (size_t) SplitDrive::kMax;
            for (size_t i = 0; ok && i < split_at.size(); ++i) ok = split_at[i] >= 2 && (i == 0 || split_at[i] > split_at[i - 1]);
            for (size_t i = 0; ok && !split_same && i < split_devs.size(); ++i) {
                ok = split_devs[i] > 0 && split_devs[i] < n_dev;
                for (size_t j = 0; ok && j < i; ++j) ok = split_devs[i] != split_devs[j];
            }
        }
        if (!ok) {
            basalt::log::error("basalt generate", "--layer-split K[,K2..]|auto needs --serve, rising K from 2, and one "
                                 "distinct GPU per K in --split-device (1..%d; or 0 with one K: the same GPU). K is the "
                                 "FIRST LAYER of each later GPU, not a count of layers per card: \"24,36,42\" for 4 "
                                 "GPUs, not \"24,12,6,6\" (got \"%s\")", n_dev - 1, o.layer_split.c_str());
            return 2;
        }
    }
    bool multi_gpu = !split_devs.empty() && !split_same;
    // two stages with the drafter and one conversation at a time pipeline their windows: a second chain of verifiers over
    // the same sessions (a window's buffers on each card) and, on the first card, the snapshot a rolled-back guess needs
    const bool pipe_want = multi_gpu && split_devs.size() == 1 && !o.mtp.empty() && o.serve && o.batch <= 0;
    constexpr int kPairMib0 = 280, kPairMibLater = 160;
    if (pipe_want) o.vram_reserve_mib += kPairMib0;
    if (const auto code = cli::resolve_model_options(o, multi_gpu)) return *code;
    if (o.serve) basalt::platform::watch_resident_ram(o.resident_cpu_experts, 0);
    // the helper-GPU expert caches (--expert-cache-remote): CUDA1..3 on one GPU; with a layer
    // split, the visible GPUs no stage runs on, in order
    int remote_dev[3] = {1, 2, 3};
    if (multi_gpu) {
        if (o.expert_profile.empty()) {
            basalt::log::error("basalt generate", "a layer split across GPUs needs a model with an expert profile");
            return 2;
        }
        // A stage's prompt path borrows from that stage's own expert cache: the loan is the tail of the stage's own
        // cache (see `PfPart` in the serve block), and outside the prompt that tail is expert cache, so a large chunk
        // costs a stage nothing permanent.
        int n_vis = 1;
        if (cudaGetDeviceCount(&n_vis) != cudaSuccess || n_vis < 1) n_vis = 1;
        cudaGetLastError();
        int next_free = 1;
        for (int r = 0; r < 3; ++r) {
            if (o.expert_cache_remote[(size_t) r] <= 0) continue;
            while (next_free < n_vis &&
                   std::find(split_devs.begin(), split_devs.end(), next_free) != split_devs.end()) ++next_free;
            if (next_free >= n_vis) {
                basalt::log::error("basalt generate", "--expert-cache-remote with a layer split needs a GPU that runs no "
                                     "stage (%d visible, %zu used by the split)", n_vis, split_devs.size() + 1);
                return 2;
            }
            remote_dev[r] = next_free++;
        }
        std::string devs;
        for (const int d : split_devs) devs += (devs.empty() ? "" : ",") + std::to_string(d);
        basalt::log::info("basalt generate", "layer split across %zu GPUs: CUDA0, then CUDA%s (split %s)",
                     split_devs.size() + 1, devs.c_str(), o.layer_split.c_str());
    }
    if (const auto code = cli::prepare_input_options(o)) return *code;
    basalt::core::qsa_set_kv_int8(o.kv == "int8");
    basalt::core::qsa_set_kv_q4(o.kv == "q4_0");   // 4-bit codes after a Hadamard rotation (kv_q4.hpp)
    basalt::core::qsa_set_kv_hybrid(o.kv == "k8v4");   // K8V4: INT8 K + rotated Q4_0 V, 816 B/cell
    if (const auto code = cli::validate_kv_streaming(o)) return *code;
    basalt::core::qsa_set_kv_resident(o.kv_resident);
    const bool no_spec = cli::normalize_speculation_options(o);
    basalt::core::layer_set_shared_early(true);
    {
        // the source tensors hold the PLE table
        try {
            const basalt::GgufModel model(o.native_shards);
            if (model.find("per_layer_token_embd.weight") == nullptr) {
                basalt::log::error("basalt generate", "%s has no per_layer_token_embd.weight", o.model.c_str());
                return 2;
            }
        } catch (const std::exception& e) {
            basalt::log::error("basalt generate", "%s: %s", o.model.c_str(), e.what());
            return 2;
        }
    }
    basalt::kernels::RopeScaling rope_cfg;
    if (const auto code = cli::validate_generation_options(o, multi_gpu, vram_floor_mib(), rope_cfg)) return *code;

    // **BEFORE ANYTHING ELSE.**  The CPU expert kernel is AVX-512 (VNNI + VBMI), so on a CPU without those features it
    // does not fail - it executes an illegal instruction at some unpredictable token.  Refusing at second zero is the
    // whole point of this check.
    // ... and nothing runs on a CPU without AVX2: every CPU expert kernel is AVX2 at least (the AVX-512 ones are
    // chosen above it), and so is ggml-cpu, which the native pack's layout load initializes next.  Refused here, by name, rather than an illegal instruction in the first expert.
    if (!basalt::kernels::cpu::cpu_avx2_ok()) {
        basalt::log::error("basalt generate", "this CPU (%s) does not support AVX2 with FMA and F16C, which this engine's "
                             "CPU expert kernels need",
                     basalt::kernels::cpu::cpu_name().c_str());
        return 2;
    }

    std::string err;
    // Where the experts live.  The model keeps every quantized tensor in its GGUF form, so its experts run in verify
    // windows only (--spec) and its dense projections, head and embedding come from the source tensors.
    {
        const basalt::core::ModelGeometry g0;
        std::vector<uint8_t> text;
        bool layout_ok = basalt::core::read_range(srcs.experts_layout, text, err) &&
                         basalt::kernels::cpu::expert_layout_load_text(std::string(text.begin(), text.end()), g0.n_layers,
                                                                       g0.n_expert, err);
        if (!layout_ok) {
            return fatal_start_error(err);
        }
        // Every layer's formats must have GPU expert kernels and a prompt-path dequantizer, checked here, before
        // anything is allocated: an unsupported type would otherwise fail from inside the first verify window or leave
        // the prompt path's fp16 buffer unwritten.
        const auto& lay = basalt::kernels::cpu::expert_layout();
        for (int64_t l = 0; lay.native && l < (int64_t) lay.fmt.size(); ++l) {
            const auto& f = lay.fmt[(size_t) l];
            if (!basalt::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
                char msg[224];
                std::snprintf(msg, sizeof msg, "layer %lld's experts are %s/%s (ggml types %d/%d), which this "
                                               "engine has no GPU kernels for: format not supported for decode",
                              (long long) l,
                              basalt::ggml_type_name((uint32_t) f.gu_type), basalt::ggml_type_name((uint32_t) f.d_type),
                              f.gu_type, f.d_type);
                return fatal_start_error(msg);
            }
            // Q6_K experts run on the fast kernels only (decode-once grouped / device-planned tail, MMQ prefill, CPU
            // dot): a pair or shape outside them is refused rather than routed to a general path.
            if ((f.gu_type == 14 || f.d_type == 14) &&
                (!basalt::kernels::dec_moe_formats_ok(f.gu_type, f.d_type, f.n_embd, f.n_ff) ||
                 !basalt::prefill::mmq::fits(f.gu_type, 2 * f.n_ff) || !basalt::prefill::mmq::fits(f.d_type, f.n_embd))) {
                char msg[224];
                std::snprintf(msg, sizeof msg, "layer %lld's experts are %s/%s: Q6_K experts have fast kernels only "
                                               "for the covered pairs and shapes",
                              (long long) l, basalt::ggml_type_name((uint32_t) f.gu_type),
                              basalt::ggml_type_name((uint32_t) f.d_type));
                return fatal_start_error(msg);
            }
        }
        // one line per run of layers with the same formats: the kernels each decode and prefill path uses
        for (int64_t l = 0; lay.native && l < (int64_t) lay.fmt.size();) {
            const auto& f = lay.fmt[(size_t) l];
            int64_t e = l;
            while (e + 1 < (int64_t) lay.fmt.size() && lay.fmt[(size_t) e + 1].gu_type == f.gu_type &&
                   lay.fmt[(size_t) e + 1].d_type == f.d_type)
                ++e;
            basalt::log::info("basalt generate", "experts %s/%s, layers %lld-%lld: decode %s, prefill %s, CPU tier dot",
                         basalt::ggml_type_name((uint32_t) f.gu_type), basalt::ggml_type_name((uint32_t) f.d_type),
                         (long long) l, (long long) e,
                         basalt::kernels::dec_moe_formats_ok(f.gu_type, f.d_type, f.n_embd, f.n_ff)
                             ? "device-planned tail + grouped kernels" : "grouped kernels",
                         basalt::prefill::mmq::fits(f.gu_type, 2 * f.n_ff) && basalt::prefill::mmq::fits(f.d_type, f.n_embd)
                             ? "MMQ" : "dequant GEMM");
            l = e + 1;
        }
    }
    // Only CUDA1's context is created here, before the large allocations: the later helpers' come after the arena.
    for (int r = 0; r < 1; ++r) if (o.expert_cache_remote[(size_t) r] > 0) {
        // Keep CUDA1's proven startup order: initialise its context before
        // allocating GPU0 weights or mapping the large host expert arena.
        double free_gib = 0;
        if (!basalt::core::RemoteExperts::preflight(remote_dev[r], free_gib, err)) {
            return fatal_start_error(err);
        }
        basalt::log::info("basalt generate", "CUDA%d context ready, %.2f GiB free before expert arena registration",
                     remote_dev[r], free_gib);
    }
    // The PCIe share of the missed experts.  The GPU's SMs read that share over the link (the copy kernel), so on a
    // slower link it must shrink or the window waits for it.  The real H2D bandwidth is probed once; from 20 GB/s up
    // the default stays.
    double main_link_gbps = 0;   // the probed host->device bandwidth (the prompt-time split's price reads it)
    if (o.pcie_frac < 0.0) {
        const double base = 0.55;
        std::string bursts;
        const double bw = probe_pcie_h2d_gbps(&bursts);
        main_link_gbps = bw;
        if (bw > 0.0) {
            o.pcie_frac = pcie_frac_for_gbps(bw, base);
            basalt::log::info("basalt generate", "PCIe probe: %.1f GB/s host->device (best of %s) -> pcie_frac %.2f "
                                 "(default %.2f)", bw, bursts.c_str(), o.pcie_frac, base);
        } else {
            o.pcie_frac = base;
            basalt::log::warn("basalt generate", "PCIe probe failed -> pcie_frac default %.2f", base);
        }
    }
    if (!basalt::kernels::cpu::cpu_avx512_ok())
        basalt::log::warn("basalt generate", "this CPU has no AVX-512: the expert kernels run on %s "
                             "(multi-token for the i-quant gate/up rows)",
                     !basalt::kernels::cpu::cpu_avx2_ok() ? "ggml-cpu vec_dot (no AVX2)" : "AVX-2");

    context.o = &o;
    context.ppl_base = &ppl_base;
    context.ppl_out = &ppl_out;
    context.chosen_cards = &chosen_cards;
    context.srcs = &srcs;
    context.model_file = &model_file;
    context.pack_files = &pack_files;
    context.expert_files = &expert_files;
    context.peer_streamed_share = &peer_streamed_share;
    context.split_want = &split_want;
    context.split_cap = &split_cap;
    context.set_stream_all_min = set_stream_all_min;
    context.split_at = &split_at;
    context.split_devs = &split_devs;
    context.split_auto = &split_auto;
    context.split_same = &split_same;
    context.multi_gpu = &multi_gpu;
    context.pipe_want = &pipe_want;
    context.kPairMibLater = &kPairMibLater;
    context.remote_dev = &remote_dev;
    context.no_spec = &no_spec;
    context.rope_cfg = &rope_cfg;
    context.err = &err;
    context.main_link_gbps = &main_link_gbps;
    return model(context);
}

}
