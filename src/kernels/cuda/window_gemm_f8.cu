// src/kernels/cuda/window_gemm_f8.cu - the decode window products against FP8 E4M3 and BF16 matrices (window_gemm.cuh).
#include "window_gemm_kernels.hpp"

namespace basalt::kernels::wgemm::detail {
namespace {
constexpr int F = (int) Format::F8, RS = 64, CH = 128, ST = 4, MW = 1;
}

const Kernel& f8(int nt) {
    static const Kernel k1 = make_kernel<F, RS, 1, CH, ST, 1, MW>(), k2 = make_kernel<F, RS, 1, CH, ST, 2, MW>();
    return nt > 1 ? k2 : k1;
}

const Kernel& bf16(int nt) {
    constexpr int B = (int) Format::Bf16;
    static const Kernel k1 = make_kernel<B, RS, 1, CH, ST, 1, MW>(), k2 = make_kernel<B, RS, 1, CH, ST, 2, MW>();
    return nt > 1 ? k2 : k1;
}

}  // namespace basalt::kernels::wgemm::detail
