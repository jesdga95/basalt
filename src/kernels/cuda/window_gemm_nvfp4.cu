// src/kernels/cuda/window_gemm_nvfp4.cu - the decode window products against NVFP4 matrices in planes, alone and as a
// SwiGLU pair (window_gemm.cuh).
#include "window_gemm_kernels.hpp"

namespace basalt::kernels::wgemm::detail {
namespace {
constexpr int F = (int) Format::Nvfp4, CH = 256, ST = 4, MW = 1;
}

const Kernel& nvfp4(int nt) {
    static const Kernel k1 = make_kernel<F, 64, 1, CH, ST, 1, MW>(), k2 = make_kernel<F, 64, 1, CH, ST, 2, MW>();
    return nt > 1 ? k2 : k1;
}

const Kernel& nvfp4_swiglu(int nt) {
    static const Kernel k1 = make_kernel<F, 32, 2, CH, ST, 1, MW>(), k2 = make_kernel<F, 32, 2, CH, ST, 2, MW>();
    return nt > 1 ? k2 : k1;
}

}  // namespace basalt::kernels::wgemm::detail
