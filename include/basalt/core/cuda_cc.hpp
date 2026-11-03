// include/basalt/core/cuda_cc.hpp - the one compute capability this build's kernels are compiled for.
//
// 12.0 is a discrete Blackwell card; 12.1 is the integrated Blackwell GPU of the N1X and GB10 systems-on-chip. The
// build sets BASALT_CUDA_CC (cmake/target_arch.cmake) from CMAKE_CUDA_ARCHITECTURES; every "is this card ours" check
// reads these names, so no file spells the number.
#pragma once

#include <string>

#ifndef BASALT_CUDA_CC
#define BASALT_CUDA_CC 120
#endif

namespace basalt::core {

inline constexpr int kCudaCC = BASALT_CUDA_CC;          ///< 120 or 121
inline constexpr int kCudaCCMajor = kCudaCC / 10;
inline constexpr int kCudaCCMinor = kCudaCC % 10;

/// Whether a card of compute capability major.minor runs this build's main kernels.
inline constexpr bool cuda_cc_is_ours(int major, int minor) { return major == kCudaCCMajor && minor == kCudaCCMinor; }

/// "sm_120" / "sm_121", for messages.
inline std::string cuda_cc_name() { return "sm_" + std::to_string(kCudaCC); }
/// "12.0" / "12.1", for messages.
inline std::string cuda_cc_dotted() { return std::to_string(kCudaCCMajor) + "." + std::to_string(kCudaCCMinor); }

}  // namespace basalt::core
