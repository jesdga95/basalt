// src/model/rope_flags.cpp - see basalt/model/rope_flags.hpp.
#include "basalt/model/rope_flags.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/log.hpp"
#include "basalt/model/flags.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace basalt::model {

bool parse_rope_flag(const std::string& a, FlagArgs& args, RopeFlags& r) {
    if (a == "--rope-scaling") r.type = args.next("--rope-scaling");
    else if (a == "--rope-scale") r.scale = std::atof(args.next("--rope-scale"));
    else if (a == "--rope-freq-base") r.freq_base = std::atof(args.next("--rope-freq-base"));
    else if (a == "--rope-freq-scale") r.freq_scale = std::atof(args.next("--rope-freq-scale"));
    else if (a == "--yarn-orig-ctx") r.orig_ctx = std::atof(args.next("--yarn-orig-ctx"));
    else if (a == "--yarn-ext-factor") r.ext_factor = std::atof(args.next("--yarn-ext-factor"));
    else if (a == "--yarn-attn-factor") r.attn_factor = std::atof(args.next("--yarn-attn-factor"));
    else if (a == "--yarn-beta-fast") r.beta_fast = std::atof(args.next("--yarn-beta-fast"));
    else if (a == "--yarn-beta-slow") r.beta_slow = std::atof(args.next("--yarn-beta-slow"));
    else return false;
    return true;
}

std::string rope_flag_words(const RopeFlags& r) {
    const RopeFlags d;
    if (r.type.empty() && r.scale == d.scale && r.freq_base == d.freq_base && r.freq_scale == d.freq_scale &&
        r.orig_ctx == d.orig_ctx && r.ext_factor == d.ext_factor && r.attn_factor == d.attn_factor &&
        r.beta_fast == d.beta_fast && r.beta_slow == d.beta_slow)
        return {};
    char num[256];
    std::snprintf(num, sizeof num, " %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g", r.scale, r.freq_base, r.freq_scale, r.orig_ctx,
                  r.ext_factor, r.attn_factor, r.beta_fast, r.beta_slow);
    return "rope " + r.type + num;
}

// Only the ranges: the configuration is resolved once the model file has had its say.  Anything invalid stops the
// start here rather than becoming a NaN angle in a rotation.
std::optional<int> check_rope_flags(const RopeFlags& r, const char* area) {
    if (!r.type.empty() && r.type != "none" && r.type != "linear" && r.type != "yarn") {
        basalt::log::error(area, "--rope-scaling must be none, linear or yarn (got '%s')", r.type.c_str());
        return 2;
    }
    // every knob FINITE first: `atof("nan")` is NaN, and a NaN passes every range comparison below
    for (const double v : {r.scale, r.freq_base, r.freq_scale, r.orig_ctx, r.ext_factor, r.attn_factor, r.beta_fast, r.beta_slow})
        if (!std::isfinite(v)) {
            basalt::log::error(area, "a rope scaling knob is not a finite number (%g)", v);
            return 2;
        }
    // 0 is the absent default; an explicit factor must extend, not shrink
    if (r.scale != 0 && r.scale < 1.0) {
        basalt::log::error(area, "--rope-scale %g must be >= 1 (it extends the context, not shrinks it)", r.scale);
        return 2;
    }
    if (r.freq_base != 0 && r.freq_base <= 1.0) {
        basalt::log::error(area, "--rope-freq-base must be a base above 1 (0 = the model's)");
        return 2;
    }
    if (r.freq_scale < 0 || r.orig_ctx < 0 || r.ext_factor < -1.0 || r.attn_factor <= 0 || r.beta_fast <= 0 || r.beta_slow <= 0) {
        basalt::log::error(area, "invalid rope scaling knob (see usage: --yarn-ext-factor <0 = auto, "
                                 "--yarn-orig-ctx 0 = default, the rest positive)");
        return 2;
    }
    return std::nullopt;
}

RopeKeys rope_keys(const GgufFile& g, const std::string& arch) {
    RopeKeys k;
    if (const MetaValue* v = g.get(arch + ".rope.freq_base")) k.freq_base = v->num();
    if (const MetaValue* v = g.get(arch + ".rope.scaling.type")) k.type = v->s;
    if (const MetaValue* v = g.get(arch + ".rope.scaling.factor")) k.factor = v->num();
    if (const MetaValue* v = g.get(arch + ".rope.scaling.original_context_length")) k.orig_ctx = v->num();
    return k;
}

bool resolve_rope_scaling(const RopeFlags& f, const RopeKeys& k, int64_t max_context, const char* area,
                          kernels::RopeScaling& out, std::string& err) {
    using RST = kernels::RopeScalingType;
    kernels::RopeScaling c = out;
    if (!f.type.empty()) {
        // check_rope_flags pinned the spelling; `none` here is the flag opting OUT of the model file's keys
        c.type = f.type == "linear" ? RST::Linear : f.type == "yarn" ? RST::YaRN : RST::None;
    } else if (!k.type.empty()) {
        if (k.type == "linear") c.type = RST::Linear;
        else if (k.type == "yarn") c.type = RST::YaRN;
        else if (k.type != "none") {
            err = "the model file carries rope.scaling.type '" + k.type + "' - none, linear or yarn only";
            return false;
        }
    }
    if (f.scale > 0) c.factor = f.scale;   // an explicit factor, 1 included
    else if (k.factor > 1.0) c.factor = k.factor;
    if (f.freq_base > 0) c.freq_base = f.freq_base;
    else if (k.freq_base > 1.0) c.freq_base = k.freq_base;
    if (f.orig_ctx > 0) c.orig_ctx = f.orig_ctx;
    else if (k.orig_ctx >= 1) c.orig_ctx = k.orig_ctx;
    c.freq_scale_in = f.freq_scale;
    c.ext_factor = f.ext_factor >= 0 ? f.ext_factor : (c.type == RST::YaRN ? 1.0 : 0.0);
    c.attn_factor = f.attn_factor;
    c.beta_fast = f.beta_fast;
    c.beta_slow = f.beta_slow;
    if (c.type == RST::None) {
        // none is the trained rotation, exactly: the scaling knobs are inert, and resetting them keeps the logged
        // config honest.  Only the frequency base and the trained context survive: they are the model's, not knobs.
        const bool knobs = f.scale > 1.0 || f.freq_scale > 0 || f.ext_factor > 0 || f.attn_factor != 1.0;
        const double base = c.freq_base, orig = c.orig_ctx;
        c = kernels::RopeScaling{};
        c.freq_base = base;
        c.orig_ctx = orig;
        if (knobs)
            basalt::log::info(area, "note: no rope scaling is active (none), so --rope-scale, --rope-freq-scale and the "
                                    "--yarn-* knobs have no effect");
    }
    // the resolved config as a whole, with the one rule every rotation site also applies: the ranges cannot see a model
    // file value, nor a combination such as a --rope-freq-scale that turns the resolved factor non-finite
    if (const char* why = kernels::rope_scaling_invalid(c)) {
        char buf[384];
        std::snprintf(buf, sizeof buf, "invalid rope scaling configuration: %s (type %s, factor %g, freq_scale %g, base %g, "
                      "original context %g)", why, c.type == RST::YaRN ? "yarn" : c.type == RST::Linear ? "linear" : "none",
                      c.factor, c.freq_scale(), c.freq_base, c.orig_ctx);
        err = buf;
        return false;
    }
    out = c;
    if (c.type == RST::None) return true;
    basalt::log::info(area, "rope scaling %s, factor %.6g (freq_scale %.6g, base %.6g, mscale %.6f), --max-context %lld "
                            "against a trained context of %.0f", c.type == RST::YaRN ? "yarn" : "linear", c.factor,
                      c.freq_scale(), c.freq_base, c.mscale(), (long long) max_context, c.orig_ctx);
    if ((double) max_context <= c.orig_ctx)
        basalt::log::info(area, "note: the context is within the trained %.0f - no position needs the extension, and the "
                                "resolved scaling still applies to every angle", c.orig_ctx);
    // only when there IS a magnitude correction: YaRN's log term (ext_factor != 0) or an explicit --yarn-attn-factor
    if (c.mscale() != 1.0)
        basalt::log::info(area, "note: the %s magnitude correction scales cos and sin by %.6f at every position",
                          c.type == RST::YaRN ? "YaRN" : "--yarn-attn-factor", c.mscale());
    return true;
}

}  // namespace basalt::model
