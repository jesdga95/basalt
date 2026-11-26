// include/basalt/kernels/mxfp4_scale.hpp - the MXFP4 E8M0 scale byte as the float the CUDA dots and dequantizers use.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

/// MXFP4's E8M0 scale as ggml decodes it (half the value, kvalues_mxfp4 being doubled; the byte is the fp32
/// exponent): the same float as the CPU tier's GGML_E8M0_TO_FP32_HALF, built from its bits.
__device__ __forceinline__ float mxfp4_scale(uint8_t e) {
    const uint32_t bits = e < 2 ? (uint32_t) 0x00200000 << e : (uint32_t) (e - 1) << 23;
    return __uint_as_float(bits);
}
