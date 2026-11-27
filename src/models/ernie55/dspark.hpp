// src/models/ernie55/dspark.hpp - ERNIE-5.5-Flash bundled DSpark speculative drafter:
//   - 3 SWA-only HCA + mHC + Latent MXFP4 MoE stages (512 experts, top-10, shared expert, SiTU-GLU 4/25)
//   - Context injection from stream-averaged main model layers [40, 41, 42] into per-stage 128-slot FP16 rings
//   - Block-parallel pass over [anchor, noise x6] (kDsparkBlock = 7) + autoregressive Markov head + multimax + confidence
//   - Dense/attention/markov_w2 weights on GPU 0; markov_w1 & stage MXFP4 experts in host RAM
#pragma once

#include "basalt/core/loader.hpp"
#include "models/ernie55/pack.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::ernie55 {

class ErnieModel;

constexpr int kDsparkBlock = 7;     ///< block rows: [anchor, noise x6]
constexpr int kDsparkWindow = 128;  ///< context positions attended before pos0
constexpr int kDsparkWinCap = 256;  ///< ring buffer capacity (> kDsparkWindow + verify window)

struct DsparkDraft {
    int n = 0;
    int32_t token[kDsparkBlock] = {};
    float conf[kDsparkBlock] = {};
};

class ErnieDsparkDrafter {
public:
    ~ErnieDsparkDrafter();

    /// Plans asynchronous file reads into `reads` for all DSpark GPU tensors, host markov_w1, and host MXFP4 experts.
    /// `gpu_cache_per_stage`: lossless GPU-0 LRU cache slots per stage for hot routed MXFP4 experts (0: CPU-only).
    /// Returns null with `err` on error. Must be called during ErnieModel::plan before ExpertTier::open so host_bytes()
    /// and GPU 0 VRAM are accounted for.
    static std::unique_ptr<ErnieDsparkDrafter> plan_load(const ErnieModel& main, int file_handle, core::LoadPlan& reads,
                                                         int max_windows, int gpu_cache_per_stage, std::string& err);

    int open_window();
    int feature_slot(int layer) const;
    int feature_floats() const;
    int block_size() const;
    int n_stages() const;
    uint64_t device_bytes() const;
    uint64_t host_bytes() const;

    /// Projects `n_rows` committed target feature rows on GPU (`d_features [n_rows, feature_floats]`) at positions
    /// `pos0 .. pos0 + n_rows - 1` into window `w`'s per-stage FP16 context rings, executing on `stream`.
    bool inject_device(int w, int64_t pos0, int n_rows, const float* d_features, void* stream, std::string& err);

    /// Runs the DSpark block pass + autoregressive Markov/multimax/confidence chain for window `w` after `anchor` at
    /// position `pos`, producing `n_out` drafts (1..block_size). Waits for `caller_stream` (e.g. inject_device).
    bool draft(int w, int32_t anchor, int64_t pos, int n_out, void* caller_stream, DsparkDraft& out, std::string& err);

    void forget_from(int w, int64_t pos);
    void reset(int w);

private:
    struct Impl;
    explicit ErnieDsparkDrafter(std::unique_ptr<Impl> p);
    std::unique_ptr<Impl> p_;
};

}  // namespace basalt::models::ernie55
