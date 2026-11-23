// src/models/inkling2/cpu_experts.hpp - the RAM experts of a window computed on the CPU, beside the card's experts.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

namespace basalt::models::inkling2 {

class InklingModel;

/// The experts that spilled to RAM (and have no card copy) run on the CPU pool instead of being read over the bus: a layer's
/// rows and choices go to RAM after its router, the pool computes their RAM pairs while the card computes the rest, and the
/// card takes the pool's rows into the experts' output before it combines them. One for every session of the model (the
/// sessions' Shared): a call's rows, a decode window's or a batch window's, are at most kDecodeRows. Every step is queued
/// on the caller's stream, so a captured window replays it. Which experts the CPU computes is read when the window runs:
/// the model's blob table (0: in RAM, InklingModel::ram_blob) on the card, ram_blob on the host; both change only between
/// windows.
class CpuExperts {
public:
    ~CpuExperts();
    /// Null with `err` empty when no expert spilled.
    static std::unique_ptr<CpuExperts> make(const InklingModel& model, std::string& err);
    /// After layer l's router: `rows` rows of its MLP input [rows][n_embd] and their choices [rows][n_used] to RAM, then the
    /// pool computes the pairs whose expert is in RAM.
    void start(int l, int rows, const float* input, const int32_t* ids, cudaStream_t stream);
    /// After the card's experts: once the pool is done, each RAM pair's row into `out` (row r * n_used + slot) and its lookup
    /// into `tally`.
    void finish(int l, int rows, const int32_t* ids, float* out, uint32_t* tally, cudaStream_t stream);

private:
    struct Impl;
    CpuExperts();
    std::unique_ptr<Impl> impl_;
};

}  // namespace basalt::models::inkling2
