#pragma once

#include <optional>

namespace basalt::program { struct Options; }
namespace basalt::kernels { struct RopeScaling; struct SamplerParams; }

namespace basalt::program::cli {

// Resolve defaults after the model's sections and GPU roles are known.
std::optional<int> resolve_model_options(Options& o, bool multi_gpu);

// Supply the server's seed prompt and check the token input, PLE options and KV format.
std::optional<int> prepare_input_options(Options& o);

std::optional<int> validate_kv_streaming(const Options& o);

// Normalize the draft window options; returns whether serving uses one-token windows.
bool normalize_speculation_options(Options& o);

// Check CLI ranges and conflicts before allocating weights; seed the RoPE type for model resolution.
std::optional<int> validate_generation_options(const Options& o, bool multi_gpu, int min_vram_reserve_mib,
                                              basalt::kernels::RopeScaling& rope_cfg);
basalt::kernels::SamplerParams sampling_options(const Options& o);

}  // namespace basalt::program::cli
