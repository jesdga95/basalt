#include "context.hpp"

namespace basalt::program::generate {

int model(Context& context) {
    auto& o = *context.o;
    auto& chosen_cards = *context.chosen_cards;
    auto& srcs = *context.srcs;
    auto& pack_files = *context.pack_files;
    auto& no_spec = *context.no_spec;
    auto& rope_cfg = *context.rope_cfg;
    auto& err = *context.err;

    basalt::core::ModelGeometry g;   // canonical defaults; the model file overrides the MoE shape below
    int64_t K = 10;
    // THE ROPE CONFIG RESOLVES HERE, BEFORE ANY WEIGHT MOVES - the CLI and the model file have both spoken,
    // and `session_init` below builds the rope table from it and captures the kernels reading its constants
    // (rope_scaling.hpp); the only hard constraint is "set before that", and dying on a bad rope key beats
    // scanning gigabytes of shards first.  Precedence: an EXPLICIT flag over the model file's rope keys over
    // the struct defaults.  The empty --rope-scaling and the 0 --rope-scale mean the flag is absent, so the
    // model file decides; an explicit value - `none` and `1` included, the opt-outs - wins over the model file.
    {
        // The model file's rope keys (llama.cpp's names under the arch prefix), when it carries any.
        std::string gguf_rope_type;
        double gguf_rope_base = 0, gguf_rope_factor = 0, gguf_rope_orig_ctx = 0;
        {
            // the model file is the authority on its own MoE shape (a pruned variant has fewer experts);
            // everything else in the geometry is unchanged
            try {
                basalt::GgufFile model_gguf(o.native_shards.front());   // the metadata shard
                if (const basalt::MetaValue* v = model_gguf.get("qwen4exp.expert_count")) g.n_expert = (int64_t) v->u;
                if (const basalt::MetaValue* v = model_gguf.get("qwen4exp.expert_used_count")) K = (int64_t) v->u;
                if (const basalt::MetaValue* v = model_gguf.get("qwen4exp.rope.freq_base")) gguf_rope_base = v->num();
                if (const basalt::MetaValue* v = model_gguf.get("qwen4exp.rope.scaling.type")) gguf_rope_type = v->s;
                if (const basalt::MetaValue* v = model_gguf.get("qwen4exp.rope.scaling.factor")) gguf_rope_factor = v->num();
                if (const basalt::MetaValue* v = model_gguf.get("qwen4exp.rope.scaling.original_context_length"))
                    gguf_rope_orig_ctx = v->num();
            } catch (const std::exception& e) {
                basalt::log::error("basalt generate", "reading the model's expert shape from %s: %s",
                             o.model.c_str(), e.what());
                return 1;
            }
        }
        using RST = basalt::kernels::RopeScalingType;
        if (!o.rope_scaling.empty()) {
            // the early validation pinned the spelling; `none` here is the CLI opting OUT of the model file's keys
            if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
            else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
            else rope_cfg.type = RST::None;
        } else if (!gguf_rope_type.empty()) {
            if (gguf_rope_type == "linear") rope_cfg.type = RST::Linear;
            else if (gguf_rope_type == "yarn") rope_cfg.type = RST::YaRN;
            else if (gguf_rope_type != "none") {
                basalt::log::error("basalt generate", "%s carries rope.scaling.type '%s' - none, linear or yarn only",
                             o.model.c_str(), gguf_rope_type.c_str());
                return 2;
            }
        }
        if (o.rope_scale > 0) rope_cfg.factor = o.rope_scale;            // an explicit factor, 1 included
        else if (gguf_rope_factor > 1.0) rope_cfg.factor = gguf_rope_factor;
        if (o.rope_freq_base > 0) rope_cfg.freq_base = o.rope_freq_base;
        else if (gguf_rope_base > 1.0) rope_cfg.freq_base = gguf_rope_base;
        if (o.yarn_orig_ctx > 0) rope_cfg.orig_ctx = o.yarn_orig_ctx;
        else if (gguf_rope_orig_ctx >= 1) rope_cfg.orig_ctx = gguf_rope_orig_ctx;
        rope_cfg.freq_scale_in = o.rope_freq_scale;
        rope_cfg.ext_factor = o.yarn_ext_factor >= 0 ? o.yarn_ext_factor
                                                     : (rope_cfg.type == RST::YaRN ? 1.0 : 0.0);
        rope_cfg.attn_factor = o.yarn_attn_factor;
        rope_cfg.beta_fast = o.yarn_beta_fast;
        rope_cfg.beta_slow = o.yarn_beta_slow;
        if (rope_cfg.type == RST::None) {
            // none is the trained rotation, exactly: the scaling knobs are inert (the table builder and
            // `kernel_args` ignore them), and resetting them keeps the logged/queried config honest.  Only the
            // frequency base survives - it is the rotation itself, not a scaling knob.
            const bool knobs = o.rope_scale > 1.0 || o.rope_freq_scale > 0 || o.yarn_ext_factor > 0 ||
                               o.yarn_attn_factor != 1.0;
            const double base = rope_cfg.freq_base;
            rope_cfg = basalt::kernels::RopeScaling{};
            rope_cfg.freq_base = base;
            if (knobs)
                basalt::log::info("basalt generate", "note: no rope scaling is active (none), so --rope-scale, "
                                     "--rope-freq-scale and the --yarn-* knobs have no effect");
        }
        // THE RESOLVED CONFIG IS VALIDATED AS A WHOLE, with the one rule every rotation site also applies
        // (rope_scaling.hpp): the CLI ranges above cannot see a model-file value, nor a combination such as a
        // --rope-freq-scale that turns the resolved factor non-finite.
        if (const char* why = basalt::kernels::rope_scaling_invalid(rope_cfg)) {
            basalt::log::error("basalt generate", "invalid rope scaling configuration: %s (type %s, factor %g, "
                                 "freq_scale %g, base %g, original context %g)",
                         why, rope_cfg.type == RST::YaRN ? "yarn" : rope_cfg.type == RST::Linear ? "linear" : "none",
                         rope_cfg.factor, rope_cfg.freq_scale(), rope_cfg.freq_base, rope_cfg.orig_ctx);
            return 2;
        }
        basalt::kernels::rope_scaling_set(rope_cfg);
        if (rope_cfg.type != RST::None) {
            const char* tn = rope_cfg.type == RST::YaRN ? "yarn" : "linear";
            basalt::log::info("basalt generate", "rope scaling %s, factor %.6g (freq_scale %.6g, base %.6g, mscale %.6f), "
                         "--max-context %lld against a trained context of %.0f",
                         tn, rope_cfg.factor, rope_cfg.freq_scale(), rope_cfg.freq_base, rope_cfg.mscale(),
                         (long long) o.max_context, rope_cfg.orig_ctx);
            if ((double) o.max_context <= rope_cfg.orig_ctx)
                basalt::log::info("basalt generate", "note: the context is within the trained %.0f - no position needs the "
                             "extension, and the resolved scaling still applies to every angle",
                             rope_cfg.orig_ctx);
            // only when there IS a magnitude correction: YaRN's log term (ext_factor != 0) or an explicit
            // --yarn-attn-factor; plain linear (mscale 1) has none, and saying otherwise was TODO 22
            if (rope_cfg.mscale() != 1.0)
                basalt::log::info("basalt generate", "note: the %s magnitude correction scales cos and sin by %.6f "
                             "at every position",
                             rope_cfg.type == RST::YaRN ? "YaRN" : "--yarn-attn-factor", rope_cfg.mscale());
        }
    }
    basalt::core::NativeEmbed native_embed;
    {
        if ((o.spec < 2 && !no_spec) || (o.prefill_chunk <= 0 && o.tokens.size() > 1)) {
            basalt::log::error("basalt generate", "%s runs its experts in verify windows: it needs --spec T (T >= 2; "
                                 "--serve also takes 0) and --prefill CHUNK", o.model.c_str());
            return 2;
        }
        const basalt::core::ModelGeometry g0;
        const auto embed_t0 = std::chrono::steady_clock::now();
        if (!native_embed.load(o.native_shards, g0.n_embd, 248320, err)) {
            return fatal_start_error(err);
        }
        basalt::core::set_native_embed(&native_embed);
        basalt::log::info("basalt generate", "%s experts (largest blob %.2f MB), token embedding "
                             "%s in mapped host memory (%.0f MiB, %.1f s), its first %lld rows in VRAM (%.1f MiB)",
                     o.model.c_str(), (double) basalt::kernels::cpu::expert_layout().max_blob / 1e6,
                     basalt::ggml_type_name((uint32_t) native_embed.type()), (double) native_embed.bytes() / 1048576.0,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - embed_t0).count(),
                     (long long) native_embed.hot_rows(), (double) native_embed.hot_bytes() / 1048576.0);
    }
    // Tensors served from the source tensors are not also loaded into the weight arena.
    std::set<std::string> skip;
    {
        if (!basalt::core::NativeDense::served_names(o.native_shards, true, skip, err)) {
            return fatal_start_error(err);
        }
        skip.insert("output.weight");
        // a PLE key the dense index holds in BF16 stays in the arena
        if (!basalt::core::NativeDense::keep_unquantized_ple_key(pack_files, skip, err)) {
            return fatal_start_error(err);
        }
        skip.insert("token_embd.weight");
    }
    std::vector<std::pair<int32_t, int32_t>> profile;
    bool peer_hot_share = false;   // the profile was reordered for the peer's share of the hot pairs
    if (!o.expert_profile.empty()) {
        int64_t pslots = 0;
        // --expert-profile-save's file from an earlier run of this model: the start places what that run's adaptive
        // tier learned instead of the model's profile (the same format; another model's file is refused by its shape)
        bool learned = false;
        if (o.serve && !o.expert_profile_save.empty()) {
            uint64_t bytes = 0;
            std::string perr;
            if (basalt::platform::file_size(o.expert_profile_save, bytes) && bytes > 0) {
                if (basalt::core::read_expert_profile(basalt::core::FileRange(o.expert_profile_save), g.n_layers,
                                                      g.n_expert, profile, pslots, perr)) {
                    learned = true;
                    o.expert_profile = o.expert_profile_save;
                } else {
                    basalt::log::warn("basalt generate", "%s not used (%s): the model's profile instead",
                                      o.expert_profile_save.c_str(), perr.c_str());
                    profile.clear();
                }
            }
        }
        if (!learned && !basalt::core::read_expert_profile(srcs.profile, g.n_layers, g.n_expert, profile, pslots, err)) {
            return fatal_start_error(err);
        }
        // An explicit number truncates the ranked list ("what would 2,000 slots give" without rebuilding the file).
        // `--expert-cache 0` means auto: the profile ranks every pair.
        if (o.expert_cache == 0) o.expert_cache = -1;
        // Multi-GPU: a peer card as fast as the main one gets a share of the hot pairs (peer_hot_share_order).  A
        // slower peer card takes the pairs after the primary's: its fixed hand-off plus a hot expert's entries at its
        // bandwidth end the layer after the primary.
        const bool slow_peer = chosen_cards.size() == 2 && chosen_cards[0].mem_gbps > 0 && chosen_cards[1].mem_gbps > 0 &&
                               chosen_cards[1].mem_gbps < 0.9 * chosen_cards[0].mem_gbps;
        if (o.peer_device >= 1 && slow_peer)
            basalt::log::info("basalt generate", "peer hot share 0: the peer card's memory is slower than the main card's");
        if (o.peer_device >= 1 && !slow_peer) {   // the peer's share of the hot pairs
            size_t at = 0, fill = 0;
            const size_t n_moved = peer_hot_share_order(profile, at, fill);
            peer_hot_share = true;
            if (at > 0)
                basalt::log::info("basalt generate", "peer hot share %.2f: %zu of the first %zu ranked pairs moved "
                                     "behind rank %zu (the peer's)", 0.45, n_moved, at, fill);
        }
        basalt::log::info("basalt generate", "profile %s: %zu ranked pairs, built for %lld slots",
                     o.expert_profile.c_str(), profile.size(), (long long) pslots);
    }
    // the whole ranking as loaded, the prior of --expert-profile-save's order (a layer split keeps only CUDA0's pairs
    // in `profile` below).  Empty without --expert-profile-save.
    std::vector<std::pair<int32_t, int32_t>> profile_loaded;
    if (!o.expert_profile_save.empty()) profile_loaded = profile;

    context.g = &g;
    context.K = &K;
    context.skip = &skip;
    context.profile = &profile;
    context.peer_hot_share = &peer_hot_share;
    context.profile_loaded = &profile_loaded;
    return layer_plan(context);
}

}
