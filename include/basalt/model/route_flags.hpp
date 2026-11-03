// include/basalt/model/route_flags.hpp - the opt-in routing flags of the mixture-of-experts models, which trade accuracy for
// fewer experts that no GPU holds (basalt/kernels/route_lossy.hpp):
//
//   --route-tail-skip [R]   in a decode window an expert no GPU holds is neither copied nor computed when every entry of the
//                           window that names it ranks R or lower by router weight (0-based of the model's k); its entries
//                           get weight 0 and the others keep theirs.  R in 1..k-1; the flag alone = k-1.
//   --route-prior [LAMBDA]  a decode window picks its k experts as if every expert a GPU holds had LAMBDA added to the
//                           score its router takes the top k of (the log probability for a softmax router); the weights stay
//                           the router's own, normalised over the picks as the model normalises them.  LAMBDA in (0, 16];
//                           the flag alone = the model's measured default (its scores' scale differs).
//
// Both are off by default and change the output: each one given is named in the start's lossy warning.  A model takes them
// in its own parse_flag and resolves them in its finish_flags; a model without routed experts leaves them unknown.
#pragma once

#include <optional>
#include <string>

namespace basalt::program {
struct CommonOptions;
}

namespace basalt::model {

class FlagArgs;

struct RouteFlags {
    int tail_skip = 0;     ///< R; 0 = off, kTailSkipBare = given without a value (finish_route_flags makes it k-1)
    float prior = 0.0f;    ///< LAMBDA; 0 = off, kPriorBare = given without a value (finish_route_flags gives the model's)
    static constexpr int kTailSkipBare = -1;
    static constexpr float kPriorBare = -1.0f;
};

/// Takes --route-tail-skip or --route-prior with its optional value; false for any other flag.  A value out of range is
/// an error line and `code` 2.
bool parse_route_flag(const std::string& a, FlagArgs& args, RouteFlags& r, int& code);
/// Once the model's routed experts per token (`top_k`) are known: the bare R becomes top_k - 1, an R of top_k or more is
/// refused (exit code 2 after the error line, area `area`), the bare LAMBDA becomes `prior_bare`, and each flag on is
/// added to o.lossy.
std::optional<int> finish_route_flags(program::CommonOptions& o, int top_k, float prior_bare, const char* area);

}  // namespace basalt::model
