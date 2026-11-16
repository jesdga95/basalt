// include/basalt/model/control_vectors.hpp - the control-vector flags a model with one takes, and the vector they
// name (see basalt/kernels/cvec.hpp).
#pragma once

#include "basalt/model/flags.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace basalt::model {

/// What `--control-vector*` selected.  The defaults are llama.cpp's: additive, every layer but the first, each
/// layer's own direction.  The `experimental-speed-projection` profile passes
/// `--control-vector-scaled FILE:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer`.
/// None by default; a model with one switches a loaded vector per request (cvec=0|1).
struct CvecFlags {
    std::vector<std::pair<std::string, float>> files;   ///< the vector files and the scale each is read with
    int first = -1, last = -1;   ///< llama.cpp's defaults: 1 .. the last layer
    int mode = 1;                ///< 0 = project, 1 = add (llama.cpp's default)
    int single = -1;             ///< --cvec-dir single:L (project mode): layer L's direction everywhere
    bool given() const { return !files.empty(); }
};

/// Takes one of the control-vector flags, with the values llama.cpp's flags of the same names take; false when `a`
/// is another flag.  `code`: an exit code for a bad value.
bool parse_cvec_flag(const std::string& a, FlagArgs& args, CvecFlags& f, int& code);

/// The words a conversation cache keys a vector under: the range, mode and direction the flags name, then each file
/// and the scale it is read with.  Empty when none was asked for.  A model's conversation identity builds these in
/// (see its `conversation_identity`), and hashes each file's bytes too: the same path can hold a different vector.
std::string cvec_identity_words(const CvecFlags& f);

/// Reads the files llama.cpp's `common_control_vector_load` would (every file's `direction.<l>` times its scale,
/// summed; layer 0 has none), applies the range, the mode and the direction the flags name, and uploads the result
/// (see `kernels::cvec_upload`).  `n_layers` and `n_embd` are the model's geometry, `hc` the residual streams its
/// layers keep per token (1 for a dense model), `arch` its name: a file whose `controlvector.model_hint` names
/// another model warns.  `summary` says what was loaded, for the model to report.
bool load_control_vectors(const CvecFlags& f, int64_t n_layers, int64_t n_embd, int64_t hc, const char* arch,
                          std::string& summary, std::string& err);

}  // namespace basalt::model
