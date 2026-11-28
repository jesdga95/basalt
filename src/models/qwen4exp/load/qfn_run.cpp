// The whole start of a Flash-Next engine, in order: its stack frames hold the resources while the server runs.
#include "basalt/core/diag.hpp"
#include "basalt/core/device.hpp"
#include "basalt/core/shipped_profiles.hpp"
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/platform/file.hpp"
#include "context.hpp"
#include "../../../program/generate/run.hpp"
#include "models/qwen4exp/adapter.hpp"

namespace basalt::models::qwen4exp {

int run_engine(Options& o, std::vector<program::GpuCard>& chosen_cards) {
    using namespace basalt::models::qwen4exp::generate;
    Context context;
    std::string& ppl_base = o.ppl_base;
    std::string& ppl_out = o.ppl_out;
    basalt::core::ModelSources srcs;
    basalt::core::ModelFile model_file;
    {
        std::string merr;
        if (!model_file.open(o.model, merr) || !model_file.load_meta(merr) || !model_file.register_gguf_view(merr)) {
            return fatal_start_error("--model: " + merr);
        }
        basalt::core::model_sources(model_file, srcs);
        // the source tensors (embedding, PLE table, native projections, head) are served through the GGUF view of the file
        o.native_shards = {o.model};
        if (srcs.dense.empty() || srcs.dense_index.empty() || srcs.experts.empty() || srcs.experts_layout.empty() ||
            srcs.hc8.empty()) {
            return fatal_start_error(o.model + " lacks the dense, dense.index, experts, experts.layout or hc8 section");
        }
        if (!srcs.mtp_dense.empty() && !srcs.mtp_dense_index.empty() && !srcs.mtp_experts.empty()) o.mtp = o.model;
        // the expert profile: the one --expert-profile names, else the engine's for this model when it has one
        uint64_t prof_bytes = 0;
        if (!o.expert_profile.empty()) {
            if (!basalt::platform::file_size(o.expert_profile, prof_bytes) || prof_bytes == 0) {
                return fatal_start_error("--expert-profile " + o.expert_profile + ": no such file, or it is empty");
            }
        } else {
            const std::string key = cli::profile_key(model_file.meta());
            const std::string miss = basalt::core::missing_shipped_profile(basalt::platform::install_root() + "/data/profiles", cli::shipped_profiles(key));
            if (!miss.empty()) return fatal_start_error(miss);
            const std::string prof = basalt::platform::install_root() + "/data/profiles/" + key + ".prof";
            if (!key.empty() && basalt::platform::file_size(prof, prof_bytes) && prof_bytes > 0) o.expert_profile = prof;
            else {
                basalt::log::info("basalt generate", "no expert profile for %s", key.empty() ? "this model" : key.c_str());
                if (o.resident_cpu_experts) {
                    basalt::log::error("basalt generate", "expert profile not found: %s; ship data/profiles with the engine "
                                       "(or pass --mmap-experts without --resident-experts)",
                                       key.empty() ? "the model names no architecture or layer and expert counts" : prof.c_str());
                    return 2;
                }
            }
        }
        // the 8-bit hyper-connection copies
        std::map<std::string, basalt::core::Hc8Copy> table;
        if (!basalt::core::hc8_read(srcs.hc8, table, merr)) {
            return fatal_start_error(o.model + ": " + merr);
        }
        basalt::core::hc8_set_table(std::move(table));
    }
    const basalt::core::PackFiles pack_files = basalt::core::PackFiles::from(srcs);
    const basalt::core::ExpertFiles expert_files = basalt::core::ExpertFiles::from(srcs);
    // ---- two or more cards: what they do together is the user's gpu-mode (one card: nothing to choose, and no other card is
    // touched). Peer: the cards after the main one hold the expert tier and read prompt stages; split: they run the later
    // layers, cut by the shares once the weights' sizes are known (layer_plan).
    const bool split_mode = o.gpus.mode == basalt::program::GpuMode::Split;
    // --diag-main-card-bits (the bitwise A/B against one card): a split card divides its kernels' work as the main card
    // (CUDA0) does, so a stage on a smaller card computes its bits. Otherwise every card plans by its own SM count
    if (split_mode && basalt::core::diagnostics().main_card_bits) basalt::core::plan_sms_device(0);
    if (o.gpus.mode == basalt::program::GpuMode::Peer) {
        o.peer_devices = o.gpus.peers();
        o.peer_device = o.peer_devices.front();
    }
    // the prompt-time layer split over the expert tier cards (PromptStage), every card a stage; with --batch a read splits
    // only while no slot decodes
    bool split_want = o.serve && o.peer_device >= 1;
    basalt::core::PeerExperts::set_min_scratch((size_t) basalt::kernels::moe_hit_grouped_scratch_bytes(basalt::kernels::cpu::MAXT * 10, basalt::kernels::cpu::H, basalt::kernels::cpu::FF));
    basalt::core::set_peer_portable(o.peer_device >= 1 || split_mode);   // multi-GPU: the Portable flag on mapped host buffers only with a peer device (before any allocation)
    if (o.serve && o.conversation_cache_disk_mib > 0 && o.conversation_cache_dir.empty()) {
        basalt::log::error("basalt", "--conversation-cache-disk-mib needs --conversation-cache-dir");
        return 2;
    }
    if (o.serve && o.conversation_cache_mib > 0 && (o.prompt_cache == 0 || o.conversation_cache_slots == 0))
        basalt::log::warn("basalt serve", "conversation caching is disabled by %s",
                     o.prompt_cache == 0 ? "--prompt-cache 0" : "--conversation-cache-slots 0");
    // Layer split (gpu-mode split): the later stages run layers [K_i, K_i+1) on the cards after the main one, cut by the shared
    // driver (LayerSplit, model::SplitHooks) once the weights are sized (layer_plan), which also gives their devices.  KV
    // streaming and images (the server's GPU encoder) run across a split, also with parking and --batch.  Across GPUs, not supported: lending cache slots to the prompt path (each stage's prompt
    // path has its own buffers).  Parking saves every stage (SavedConversation::stage_images).
    std::vector<int64_t> split_at;
    std::vector<int> split_devs;
    LayerSplit split;
    bool split_same = false;   // both stages on one GPU, sharing everything: the hand-off's bit-exact A/B (--diag-gpu-same)
    if (o.gpus.diag_same && !split_mode) {
        basalt::log::error("basalt generate", "--diag-gpu-same needs gpu-mode split (and --diag-gpu-cut L, the layer the second stage starts at)");
        return 2;
    }
    if (split_mode) {
        if (!o.serve) {
            basalt::log::error("basalt generate", "gpu-mode split runs Qwen3.8-Flash-Next with --serve");
            return 2;
        }
        if (o.gpus.diag_same) {   // the split's reference: the same cut, the second stage on CUDA0 too, the second card untouched
            if (o.gpus.diag_cut < 2) {
                basalt::log::error("basalt generate", "--diag-gpu-same takes its cut from --diag-gpu-cut L, a layer from 2");
                return 2;
            }
            if (o.conversation_cache_mib > 0 && o.conversation_cache_slots > 0 && o.prompt_cache > 0) {
                basalt::log::error("basalt serve", "conversation parking does not support a layer split on one GPU (--diag-gpu-same); "
                                   "disable parking with --conversation-cache-mib 0");
                return 2;
            }
            split_same = true;
        }
    }
    bool multi_gpu = split_mode && !split_same;
    // two stages with the drafter and one conversation at a time pipeline their windows: a second chain of verifiers over
    // the same sessions (a window's buffers on each card) and, on the first card, the snapshot a rolled-back guess needs
    const bool pipe_want = multi_gpu && o.gpus.cards.size() == 2 && !o.mtp.empty() && o.serve && o.batch <= 0;
    constexpr int kPairMib0 = 280, kPairMibLater = 160;
    if (pipe_want) o.vram_reserve_mib += kPairMib0;
    if (const auto code = cli::resolve_model_options(o, multi_gpu)) return *code;
    if (o.serve) basalt::platform::watch_resident_ram(o.resident_cpu_experts, 0);
    if (multi_gpu) {
        if (o.expert_profile.empty()) {
            basalt::log::error("basalt generate", "a layer split across GPUs needs a model with an expert profile");
            return 2;
        }
        // A stage's prompt path borrows from that stage's own expert cache: the loan is the tail of the stage's own
        // cache (see `PfPart` in the serve block), and outside the prompt that tail is expert cache, so a large chunk
        // costs a stage nothing permanent.
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
    if (const auto code = cli::validate_generation_options(o, multi_gpu, vram_floor_mib())) return *code;

    // **BEFORE ANYTHING ELSE.**  The CPU expert kernel is AVX-512 (VNNI + VBMI), so on a CPU without those features it
    // does not fail - it executes an illegal instruction at some unpredictable token.  Refusing at second zero is the
    // whole point of this check.
    // ... and nothing runs on a CPU without AVX2: every CPU expert kernel is AVX2 at least (the AVX-512 ones are
    // chosen above it), and so is ggml-cpu, which the native pack's layout load initializes next.  Refused here, by name, rather than an illegal instruction in the first expert.
    if (!basalt::kernels::cpu::cpu_kernels_ok()) {
        basalt::log::error("basalt generate", "this CPU (%s) does not support %s, which this engine's "
                             "CPU expert kernels need",
                     basalt::kernels::cpu::cpu_name().c_str(),
                     basalt::platform::kArm64 ? "NEON with the Armv8.2 dot product" : "AVX2 with FMA and F16C");
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
                                                                       g0.n_expert, err, g0.n_embd, g0.n_ff);
        if (!layout_ok) {
            return fatal_start_error(err);
        }
        // Every layer's formats must have GPU expert kernels and a prompt-path dequantizer, checked here, before
        // anything is allocated: an unsupported type would otherwise fail from inside the first verify window or leave
        // the prompt path's fp16 buffer unwritten.
        const auto& lay = basalt::kernels::cpu::expert_layout();
        for (int64_t l = 0; lay.native && l < (int64_t) lay.fmt.size(); ++l) {
            const auto& f = lay.fmt[(size_t) l];
            if (!basalt::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type)) {
                char msg[224];
                std::snprintf(msg, sizeof msg, "layer %lld's experts are %s%s%s/%s (ggml types %d/%d/%d), which "
                                               "this engine has no GPU kernels for: format not supported for decode",
                              (long long) l, basalt::ggml_type_name((uint32_t) f.gu_type),
                              f.up_type != f.gu_type ? "+" : "",
                              f.up_type != f.gu_type ? basalt::ggml_type_name((uint32_t) f.up_type) : "",
                              basalt::ggml_type_name((uint32_t) f.d_type), f.gu_type, f.up_type, f.d_type);
                return fatal_start_error(msg);
            }
            // Q6_K experts run on the fast kernels only (decode-once grouped / device-planned tail, MMQ prefill, CPU
            // dot): a pair or shape outside them is refused rather than routed to a general path.
            if ((f.gu_type == 14 || f.d_type == 14) &&
                (!basalt::kernels::dec_moe_formats_ok(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type) ||
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
                   lay.fmt[(size_t) e + 1].up_type == f.up_type && lay.fmt[(size_t) e + 1].d_type == f.d_type)
                ++e;
            // a layer whose up rows have their own type shows both: gate+up/down
            const bool mixed = f.up_type != f.gu_type;
            basalt::log::info("basalt generate", "experts %s%s%s/%s, layers %lld-%lld: decode %s, prefill %s, CPU tier dot",
                         basalt::ggml_type_name((uint32_t) f.gu_type), mixed ? "+" : "",
                         mixed ? basalt::ggml_type_name((uint32_t) f.up_type) : "",
                         basalt::ggml_type_name((uint32_t) f.d_type), (long long) l, (long long) e,
                         basalt::kernels::dec_moe_formats_ok(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type)
                             ? "device-planned tail + grouped kernels" : "grouped kernels",
                         basalt::prefill::mmq::fits(f.gu_type, 2 * f.n_ff) && basalt::prefill::mmq::fits(f.up_type, 2 * f.n_ff) &&
                                 basalt::prefill::mmq::fits(f.d_type, f.n_embd)
                             ? "MMQ" : "dequant GEMM");
            l = e + 1;
        }
    }
    // The PCIe share of the missed experts.  The GPU's SMs read that share over the link (the copy kernel), so on a
    // slower link it must shrink or the window waits for it.  The real H2D bandwidth is probed once; at x16 rates
    // the default stays.
    double main_link_gbps = 0;   // the probed host->device bandwidth
    if (o.diag_fixed_experts) {   // a prompt's missed experts over PCIe on the GPU, no CPU rows (the windows: serve/verify)
        o.pcie_frac = 1.0;
        o.prefill_cpu_share = 0.0;
    }
    if (o.pcie_frac < 0.0) {
        const double base = 0.55;
        std::string bursts;
        const double bw = basalt::core::probe_h2d_gbps(&bursts);
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
    if (basalt::platform::kX86 && !basalt::kernels::cpu::cpu_avx512_ok())
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
    context.split_want = &split_want;
    context.split_at = &split_at;
    context.split_devs = &split_devs;
    context.split_same = &split_same;
    context.split = &split;
    context.multi_gpu = &multi_gpu;
    context.pipe_want = &pipe_want;
    context.kPairMibLater = &kPairMibLater;
    context.no_spec = &no_spec;
    context.rope_cfg = &rope_cfg;
    context.err = &err;
    context.main_link_gbps = &main_link_gbps;
    return basalt::models::qwen4exp::generate::model(context);
}

}  // namespace basalt::models::qwen4exp
