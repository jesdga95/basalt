// src/models/deepseek41/prefill.hpp - batched prompt processing for DeepSeek-V4.1: a chunk of tokens per layer call.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

class DsModel;
class LayerOps;

class Prefill {
public:
    virtual ~Prefill() = default;
    /// Optional: called after each chunk with the tokens of the current run read so far.
    std::function<void(int)> progress;
    /// Optional: called from a driver thread as each layer of a chunk finishes on the GPU, with an estimate of the
    /// tokens of the current run read so far (the chunk's start plus its length times the share of layers done).
    std::function<void(int)> live;
    /// Optional: asked after each chunk; true ends the read there, and the state is good for nothing.
    std::function<bool()> stop;
    /// Runs tokens[0..n) at positions pos0.. through every layer, chunk by chunk, leaving the layer state exactly as
    /// n single-token steps would. `hist` is the session's token history (extended here).
    virtual bool run(const int32_t* tokens, int n, int64_t pos0, std::vector<int32_t>& hist, std::string& err) = 0;
    /// The same for n rows of input embeddings [n][n_embd] (an encoded picture) in place of token embeddings. The
    /// rows take the image router bias and no Engram; they enter `hist` as -1 (a dead id for the n-gram hash).
    virtual bool run_embeddings(const float* rows, int n, int64_t pos0, std::vector<int32_t>& hist, std::string& err) = 0;
};

/// `ops` and `stream` (a cudaStream_t) outlive the result.
std::unique_ptr<Prefill> make_prefill(DsModel& model, LayerOps& ops, void* stream, int max_chunk, std::string& err);

}  // namespace basalt::models::deepseek41
