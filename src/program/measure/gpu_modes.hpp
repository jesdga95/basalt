// basalt/program/measure/gpu_modes.hpp - the GPU-only measurement modes and the per-stage event names.
//
// These are diagnostic entry points: they replay the captured graphs to time the GPU alone, and they print a table
// instead of decoding.  Nothing in the decode path calls them; each mode runs once and the engine exits.
// `arm_stage_timing` is the exception: it only names the stages so a normal `--no-capture` run reports them.
//
// The header names the engine's types without including their headers: this is program-layer code, and the core
// headers pull CUDA in.  The one translation unit that needs the definitions includes them.
#pragma once

#include <functional>
#include <string>

namespace basalt::core {
struct ModelGeometry;
struct SessionState;
struct SessionGraphs;
struct Doorbell;
}  // namespace basalt::core

namespace basalt::program {

struct Options;

namespace measure {

/// Returned by `run_gpu_modes` when no measurement flag was given: the engine should carry on decoding.
inline constexpr int kNotRequested = -1;

/// What the modes need from the engine: the captured graphs, the stream they replay on, the doorbell, and the head
/// runner that `--gpu-only-full` adds to the layer floor.
struct GpuModeContext {
    const Options* o = nullptr;
    const basalt::core::ModelGeometry* g = nullptr;
    basalt::core::SessionState* ss = nullptr;
    basalt::core::SessionGraphs* gr = nullptr;
    void* stream = nullptr;                        ///< the main stream
    basalt::core::Doorbell* db = nullptr;             ///< `--gpu-stages` frees it, as it did inline
    void* d_next = nullptr;                        ///< `--gpu-stages` frees it, as it did inline
    std::function<bool(void*)> run_head;           ///< the LM head on a stream
    std::string* err = nullptr;
};

/// `--gpu-stages`, `--graph-only`, `--gpu-only-full`: replay and print, then the engine exits.  Returns
/// `kNotRequested` when none of them was asked for, else the process exit code (0 ran, 1 failed, 2 refused).
int run_gpu_modes(const GpuModeContext& c);

/// `--stage-timing`: refuse it without `--no-capture` (a captured event record is silently dropped), then name the
/// stages.  Returns 0 when the run may go on, else the exit code.
int arm_stage_timing(const Options& o);

}  // namespace measure
}  // namespace basalt::program
