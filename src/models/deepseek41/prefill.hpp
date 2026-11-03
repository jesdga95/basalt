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
    /// Optional: queued at the start of each layer with its streams [T][4][n_embd], on the chunk's stream.
    std::function<void(int layer, const float* x, int T, void* stream)> layer_tap;
    /// Optional: a chunk of T tokens at pos0 has run; false with `err` fails the read.
    std::function<bool(int T, int64_t pos0, std::string& err)> chunk_done;
    /// Runs tokens[0..n) at positions pos0.. through the layers, chunk by chunk, leaving the layer state exactly as
    /// n single-token steps would for what any token from the read's stop on reads. `rows_after`: positions between these
    /// tokens and the stop (-1 unknown): rows further from it than the layers past the first ratio-1 layer can see stop at
    /// that layer's compressor. `hist` is the session's token history (extended here).
    virtual bool run(const int32_t* tokens, int n, int64_t pos0, int64_t rows_after, std::vector<int32_t>& hist,
                     std::string& err) = 0;
    /// The last run left rows short of the last layers.
    virtual bool left_rows() const = 0;
    /// The same for n rows of input embeddings [n][n_embd] (an encoded picture) in place of token embeddings. The
    /// rows take the image router bias and no Engram; they enter `hist` as -1 (a dead id for the n-gram hash).
    virtual bool run_embeddings(const float* rows, int n, int64_t pos0, std::vector<int32_t>& hist, std::string& err) = 0;
};

/// `ops` and `stream` (a cudaStream_t) outlive the result.
std::unique_ptr<Prefill> make_prefill(DsModel& model, LayerOps& ops, void* stream, int max_chunk, std::string& err);

}  // namespace basalt::models::deepseek41
