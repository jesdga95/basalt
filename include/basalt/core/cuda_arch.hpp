// include/basalt/core/cuda_arch.hpp - the one compute capability this build's kernels are compiled for
// (CMAKE_CUDA_ARCHITECTURES): 12.0 or 12.1.  A card of any other compute capability is refused before it runs one.
#pragma once

#include <string>

namespace basalt::core {

/// The compute capability as major * 10 + minor.
inline constexpr int kCudaCc = BASALT_CUDA_CC;
inline constexpr int kCudaCcMajor = kCudaCc / 10;
inline constexpr int kCudaCcMinor = kCudaCc % 10;

/// "12.1": the compute capability, for messages.
inline std::string cuda_cc_text() { return std::to_string(kCudaCcMajor) + "." + std::to_string(kCudaCcMinor); }

/// "sm_121": the architecture's name, for messages.
inline std::string cuda_sm_text() { return "sm_" + std::to_string(kCudaCc); }

}  // namespace basalt::core
