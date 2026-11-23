// include/basalt/model/gpu_check.hpp - the several-GPUs conformance check (docs/design/multi-gpu.md): a prompt and its greedy
// continuation through the session interface only, so it names no model. `basalt --model FILE --gpu 0,1 --gpu-mode peer
// --tokens-file F --diag-gpu-check --diag-gpu-trace OUT` records the run; the same command on the reference (one card, or a
// split with both stages on one GPU) with `--diag-gpu-ref OUT` compares the two. tools/gpu_identity.py runs the pair.
#pragma once

#include "basalt/model/adapter.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::model {

/// What a run produced: the logits after the prompt (step 0) and after each greedy token, and those tokens.
struct GpuTrace {
    int64_t n_vocab = 0;
    std::vector<int32_t> tokens;     ///< the pick of each step
    std::vector<float> logits;       ///< steps x n_vocab
    int steps() const { return (int) tokens.size(); }
};

/// Reads `prompt` through the session (one prefill call over all but its last token, as the CLI does, so a long prompt
/// takes the model's own chunks and any prompt stage), then `steps` greedy single-token decode steps, keeping every
/// step's logits. With `forced` (the reference's picks) each step feeds the reference's token instead of its own, so every
/// step is compared on the same context: a greedy split does not turn the later steps' KLD into two texts' distance.
/// Error: the session failed.
Result record_gpu_trace(ModelSession& s, const std::vector<int32_t>& prompt, int steps, int64_t n_vocab, GpuTrace& out,
                        const std::vector<int32_t>* forced = nullptr);

bool write_gpu_trace(const std::string& path, const GpuTrace& t, std::string& err);
bool read_gpu_trace(const std::string& path, GpuTrace& t, std::string& err);

struct GpuCompare {
    int steps = 0;                   ///< steps compared
    int bitwise = 0;                 ///< steps whose logits are the same bits
    int first_diff = -1;             ///< the first step whose logits differ (-1: none)
    int first_token_diff = -1;       ///< the first step whose pick differs (-1: greedy identical)
    double max_kld = 0;              ///< the largest step KL(reference || run), nats
    double max_abs = 0;              ///< the largest logit difference
    double kld = 0;                  ///< the mean KL(reference || run) over the steps, nats
    double top1 = 0;                 ///< the share of steps whose argmax is the reference's
    bool same() const { return steps > 0 && bitwise == steps && first_token_diff < 0; }
};

/// The run against the reference, over the steps both hold.
GpuCompare compare_gpu_traces(const GpuTrace& ref, const GpuTrace& run);

/// "gpu check: 33 steps, 33 bitwise equal, greedy identical, ...: BITWISE", or DIFFERS with the first differences.
std::string gpu_compare_line(const GpuCompare& c);

}  // namespace basalt::model
