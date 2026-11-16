// include/basalt/model/rope_flags.hpp - the rope scaling flags of the models that extend their context past the trained
// one (--rope-scaling, --rope-scale, --rope-freq-base, --rope-freq-scale, --yarn-*: llama.cpp's names), and how they
// resolve with the model file's rope keys into the run's one scaling (basalt/kernels/rope_scaling.hpp).
//
// Precedence: an explicit flag over the model file's key over the model's own values; `none` and `1` are explicit values
// (the opt-outs), the absent flag is not.  A model takes these flags in its own parse_flag; a model that carries its own
// scaling leaves them unknown.
#pragma once

#include "basalt/kernels/rope_scaling.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace basalt {
class GgufFile;
}

namespace basalt::model {

class FlagArgs;

struct RopeFlags {
    std::string type;          ///< --rope-scaling none|linear|yarn; empty = the flag is absent
    double scale = 0;          ///< --rope-scale F: the extension factor; 0 = absent (the model file's, else 1 = off)
    double freq_base = 0;      ///< --rope-freq-base N: 0 = the model's
    double freq_scale = 0;     ///< --rope-freq-scale F: the raw ggml knob; 0 = 1/--rope-scale
    double orig_ctx = 0;       ///< --yarn-orig-ctx N: 0 = the model's trained context
    double ext_factor = -1.0;  ///< --yarn-ext-factor F: <0 = auto (1 for yarn, 0 otherwise)
    double attn_factor = 1.0;  ///< --yarn-attn-factor F
    double beta_fast = 32.0;   ///< --yarn-beta-fast F
    double beta_slow = 1.0;    ///< --yarn-beta-slow F
};

/// The flags as words for an identity (a parked conversation's K is stored rotated): empty when none was given.
std::string rope_flag_words(const RopeFlags& r);

/// Takes one of the flags above; false for any other.
bool parse_rope_flag(const std::string& a, FlagArgs& args, RopeFlags& r);
/// The ranges, before the model file is read: exit code 2 after the error line for a bad value.
std::optional<int> check_rope_flags(const RopeFlags& r, const char* area);

/// The model file's rope keys: `<arch>.rope.freq_base`, `<arch>.rope.scaling.type`, `.factor` and
/// `.original_context_length`; 0 or empty when absent.
struct RopeKeys {
    std::string type;
    double factor = 0, orig_ctx = 0, freq_base = 0;
};
RopeKeys rope_keys(const GgufFile& g, const std::string& arch);

/// The run's scaling: `out` holds the model's own values (its frequency base and trained context) and takes the keys,
/// then the flags.  False with `err` for a configuration that cannot run.  The info lines (area `area`) say what was
/// resolved, against `max_context`.
bool resolve_rope_scaling(const RopeFlags& f, const RopeKeys& k, int64_t max_context, const char* area,
                          kernels::RopeScaling& out, std::string& err);

}  // namespace basalt::model
