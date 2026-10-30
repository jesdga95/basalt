#include "options.hpp"
#include "help.hpp"

#include "basalt/kernels/rope_scaling.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/log.hpp"
#include "basalt/program/options.hpp"

#include <algorithm>
#include <cmath>

namespace basalt::program::cli {

std::optional<int> resolve_model_options(Options& o, bool multi_gpu) {
    // Built-in defaults for what is given nowhere: the fastest settings that need no knowledge of the machine.
    // Speculation needs the pack's drafter; prompt chunks are sized from what the expert cache can lend; the experts
    // the GPU does not hold go to page-locked RAM when they fit (a plain file mapping when not); the adaptive tier
    // of a server on one card runs a round every window.  An explicit flag always wins.
    if (!o.spec_given && !o.mtp.empty()) o.spec = 8;
    if (!o.prefill_given) {
        o.prefill_auto = o.prefill_auto_fit = true;
        o.prefill_auto_max = 32768;
        o.prefill_chunk = o.prefill_auto_max;
    }
    if (!o.mmap_experts && !o.resident_cpu_experts && o.shared_expert_arena.empty() && !o.expert_profile.empty() &&
        o.expert_cache_remote[0] <= 0 && o.expert_cache_remote[1] <= 0 && o.expert_cache_remote[2] <= 0) {
        o.mmap_experts = o.resident_cpu_experts = o.resident_pin = o.resident_soft = true;
        o.resident_headroom = 0;
    }
    if (!o.adapt_every_given && o.serve && !multi_gpu && !o.remote_expert_opt) o.adapt_every = 1;
    if (o.mmap_experts && !o.shared_expert_arena.empty()) {
        basalt::log::error("basalt generate", "--shared-expert-arena backs the resident arena and cannot be used with --mmap-experts");
        return 2;
    }
    if (o.resident_cpu_experts && (!o.mmap_experts || o.expert_profile.empty())) {
        basalt::log::error("basalt generate", "--resident-cpu-experts requires --mmap-experts and a model with an expert profile");
        return 2;
    }
    const bool remote_caches = o.expert_cache_remote[0] > 0 || o.expert_cache_remote[1] > 0 ||
                               o.expert_cache_remote[2] > 0;
    if (o.resident_cpu_experts && remote_caches) {
        basalt::log::error("basalt generate", "--resident-cpu-experts does not support remote expert caches");
        return 2;
    }
    return std::nullopt;
}

std::optional<int> prepare_input_options(Options& o) {
    bool have_tokens = !o.tokens.empty();
    if (o.prefill_auto && o.expert_profile.empty()) {
        o.prefill_auto = false;       // nothing to lend from: the buffers are reserved for the session, so keep them small
        o.prefill_chunk = 2048;
    }
    if (!have_tokens && o.serve) {   // requests bring their own tokens
        o.tokens = {248045};
        o.max_new = 1;
        have_tokens = true;
        o.stop_eos = true;
    }
    if (!have_tokens) {
        basalt::log::error("basalt generate", "--tokens is required (this build has no tokenizer; see the "
                             "header of src/program/generate.cpp)");
        usage();
        return 2;
    }

    if ((o.ple_io != "direct" && o.ple_io != "ram") || o.ple_row_cache < 0 || o.ple_inflight < 1 ||
        o.ple_inflight > 1024 || !(o.ple_delay_us >= 0)) {
        basalt::log::error("basalt generate", "invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us");
        return 2;
    }
    if (o.kv == "q4") o.kv = "q4_0";
    if (o.kv != "fp16" && o.kv != "int8" && o.kv != "q4_0" && o.kv != "k8v4") {
        basalt::log::error("basalt generate", "--kv must be fp16, int8, q4_0 or k8v4");
        return 2;
    }
    return std::nullopt;
}

std::optional<int> validate_generation_options(const Options& o, bool multi_gpu, int min_vram_reserve_mib,
                                              basalt::kernels::RopeScaling& rope_cfg) {
    // positions, cells and pooled-block indices are cast to int32 on the device path.
    if (o.max_context > 2147483647LL - 8) {
        basalt::log::error("basalt generate", "--max-context must be below 2^31");
        return 2;
    }
    if (o.max_new <= 0 || o.max_context <= 0 || o.max_new > o.max_context ||
        o.tokens.size() > (size_t) (o.max_context - o.max_new)) {
        basalt::log::error("basalt generate", "positive --max-new and --max-context must fit the prompt and generation");
        return 2;
    }
    // THE ROPE KNOBS (rope_scaling.hpp).  Anything invalid dies here, at second zero, rather than becoming a
    // NaN angle inside one of the twelve QSA layers.  Only the RANGES are checked - the config itself is
    // resolved after the model file has had its say, right before session_init.
    {
        using RST = basalt::kernels::RopeScalingType;
        // an absent --rope-scaling (the empty default) leaves the type to the model file's rope keys,
        // resolved below; anything present must be one of the three names
        if (o.rope_scaling == "none") rope_cfg.type = RST::None;
        else if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
        else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
        else if (!o.rope_scaling.empty()) {
            basalt::log::error("basalt generate", "--rope-scaling must be none, linear or yarn (got '%s')",
                         o.rope_scaling.c_str());
            return 2;
        }
        // every knob FINITE first: `atof("nan")` is NaN, and a NaN passes every range comparison below
        for (const double v : {o.rope_scale, o.rope_freq_base, o.rope_freq_scale, o.yarn_orig_ctx, o.yarn_ext_factor,
                               o.yarn_attn_factor, o.yarn_beta_fast, o.yarn_beta_slow})
            if (!std::isfinite(v)) {
                basalt::log::error("basalt generate", "a rope scaling knob is not a finite number (%g)", v);
                return 2;
            }
        // 0 is the absent default; an explicit factor must extend, not shrink
        if (o.rope_scale != 0 && o.rope_scale < 1.0) {
            basalt::log::error("basalt generate", "--rope-scale %g must be >= 1 (it extends the context, not shrinks it)",
                         o.rope_scale);
            return 2;
        }
        if (o.rope_freq_base != 0 && o.rope_freq_base <= 1.0) {
            basalt::log::error("basalt generate", "--rope-freq-base must be a base above 1 (0 = the model's)");
            return 2;
        }
        if (o.rope_freq_scale < 0 || o.yarn_orig_ctx < 0 || o.yarn_ext_factor < -1.0 || o.yarn_attn_factor <= 0 ||
            o.yarn_beta_fast <= 0 || o.yarn_beta_slow <= 0) {
            basalt::log::error("basalt generate", "invalid rope scaling knob (see usage: --yarn-ext-factor <0 = auto, "
                                 "--yarn-orig-ctx 0 = default, the rest positive)");
            return 2;
        }
    }
    if (!std::isfinite(o.temperature) || o.temperature < 0 || !std::isfinite(o.top_p) ||
        o.top_p <= 0 || o.top_p > 1 || o.top_k < 0 || o.expert_cache < -1 ||
        o.pool_workers < 0 || std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                           [](int slots) { return slots < 0; }) ||
        (o.expert_cache_remote[1] > 0 && o.expert_cache_remote[0] == 0) ||
        (o.expert_cache_remote[2] > 0 && o.expert_cache_remote[1] == 0)) {
        basalt::log::error("basalt generate", "invalid sampling or resource parameter");
        return 2;
    }
    if (!(o.prefill_cpu_share >= 0.0 && o.prefill_cpu_share <= 1.0)) {
        basalt::log::error("basalt generate", "--prefill-cpu-share must be a number from 0 to 1");
        return 2;
    }
    if (o.vram_reserve_given && o.vram_reserve_mib < min_vram_reserve_mib) {
        basalt::log::error("basalt", "--vram-reserve-mib %d leaves less VRAM free than the engine needs once "
                             "everything is loaded; the smallest safe value is %d", o.vram_reserve_mib, min_vram_reserve_mib);
        return 2;
    }
    if (o.expert_cache_remote_placement != "stripe" && o.expert_cache_remote_placement != "layer") {
        basalt::log::error("basalt generate", "--expert-cache-remote-placement must be stripe or layer");
        return 2;
    }
    if (o.remote_expert_opt && !o.serve) {
        basalt::log::error("basalt generate", "--remote-expert-opt requires --serve");
        return 2;
    }
    if (o.peer_exclusive && (o.peer_device < 1 || !o.resident_cpu_experts || !o.serve)) {
        basalt::log::error("basalt generate", "--peer-exclusive needs a second GPU (--gpu 0,1), --resident-experts and --serve");
        return 2;
    }
    if (o.peer_exclusive && o.peer_devices.size() > 1) {
        basalt::log::error("basalt generate", "--peer-exclusive runs with one expert tier card (--gpu 0,1)");
        return 2;
    }
    // the peer tier is the second card's only user: a layer split or a remote expert cache would put a second engine
    // part (and a second copy of the same experts) on it
    if (o.peer_device >= 1 && (!o.layer_split.empty() || o.expert_cache_remote[0] > 0)) {
        basalt::log::error("basalt generate", "--peer-device cannot be combined with %s",
                     !o.layer_split.empty() ? "--layer-split (use one or the other)"
                                            : "--expert-cache-device1..3 (the peer tier already caches experts there)");
        return 2;
    }

    // The whole-model graph measurements replay every layer through CUDA0's session, which a layer split carves to
    // CUDA0's own range - the answer would read another stage's state.  Refused here rather than at the call, so
    // the reason is visible before the model is loaded.
    if (multi_gpu && (o.gpu_only_full || o.gpu_stages)) {
        basalt::log::error("basalt generate", "--gpu-only-full and --gpu-stages replay the whole model through one "
                             "session, which a layer split does not have; run them on one GPU (--gpu 0)");
        return 2;
    }

    return std::nullopt;
}

std::optional<int> validate_kv_streaming(const Options& o) {
    if (o.kv_resident < 0) {
        basalt::log::error("basalt generate", "--kv-resident must be >= 0");
        return 2;
    }
    if (o.kv == "k8v4" && o.kv_resident > 0) {
        basalt::log::error("basalt generate", "--kv k8v4 does not support --kv-resident streaming (yet)");
        return 2;
    }
    return std::nullopt;
}

bool normalize_speculation_options(Options& o) {
    // --serve with --spec 0 or 1: no speculation.  The drafter is not loaded, prompt lookup is off, and every step
    // is a one-token window through the verify path.
    const bool no_spec = o.serve && o.spec < 2;
    if (no_spec) {
        o.spec = 1;   // one-token windows; the verifier is still sized for 2 or more tokens
        o.suffix_draft = 0;
        o.mtp.clear();
    }
    // Prompt lookup (the suffix drafter, on by default): the MTP keeps its --spec windows and a lookup window may be
    // up to 2 tokens longer (at most the verifier width); the draft policy (basalt/spec/draft_policy.hpp) takes one only where it pays.
    // --suffix-draft 0 turns it off.
    if (o.suffix_draft > 0 && o.spec >= 2 && o.mtp_max_t == 0) {
        o.mtp_max_t = o.spec;
        o.spec = std::min(o.spec + 2, 8);   // kVerifyMaxT
    }
    return no_spec;
}

basalt::kernels::SamplerParams sampling_options(const Options& o) {
    basalt::kernels::SamplerParams sp;
    sp.greedy = o.greedy;
    sp.seed = o.seed;
    sp.top_k = o.top_k;
    sp.top_p = o.top_p;
    sp.temperature = o.temperature;
    // what this run actually samples with (the speculative loop below gets the same parameters); serve samples
    // per request instead
    if (!o.serve) {
        if (sp.greedy || sp.temperature <= 0.0f)
            basalt::log::info("basalt generate", "sampling greedy");
        else
            basalt::log::info("basalt generate", "sampling temperature=%g top_k=%d top_p=%g seed=%llu",
                         (double) sp.temperature, sp.top_k > 0 && sp.top_k < 64 ? sp.top_k : 64, (double) sp.top_p,
                         (unsigned long long) sp.seed);
    }


    return sp;
}

}  // namespace basalt::program::cli
