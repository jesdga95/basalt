// src/kernels/cuda/window_gemm_kernels.hpp - the decode window products' kernel instantiations, one .cu per format.
#pragma once

#include "window_gemm.cuh"

namespace basalt::kernels::wgemm::detail {

/// The kernel for windows of up to 8 (nt 1) or 16 (nt 2) rows.  Both nt of a kind have the same block shape.
const Kernel& f8(int nt);
const Kernel& bf16(int nt);
const Kernel& nvfp4(int nt);
const Kernel& nvfp4_swiglu(int nt);

template <int F, int RS, int NM, int CH, int ST, int NT, int MW> Kernel make_kernel() {
    using C = Cfg<F, RS, NM, CH, ST>;
    Kernel k;
    k.fn = wgemm_kernel<F, RS, NM, CH, ST, NT, MW>;
    k.threads = C::kThreads / MW;
    k.smem = C::kSmem;
    k.rs = RS;
    k.ch = CH;
    k.nm = NM;
    return k;
}

}  // namespace basalt::kernels::wgemm::detail
