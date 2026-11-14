// src/kernels/cuda/window_gemm_f8.cu - the decode window products against FP8 E4M3 (window_gemm_sliced.cuh), BF16 and Q8
// matrices (window_gemm.cuh).
#include "window_gemm_kernels.hpp"
#include "window_gemm_sliced.cuh"

namespace basalt::kernels::wgemm::detail {
namespace {
constexpr int RS = 64, CH = 128, ST = 4, MW = 1;
}

template <int KW, int NT> Kernel make_sliced() {
    // the stages change only the overlap, never the sums: two where few wide blocks stream short slices
    constexpr int KB = 1, SST = KW == 8 && NT <= 2 ? 2 : 1;
    using SC = SlicedCfg<KW, KB, NT, SST>;
    Kernel k;
    k.fn = wgemm_sliced_kernel<KW, KB, NT, SST>;
    k.threads = SC::kThreads;
    k.smem = SC::kSmem;
    k.rs = 16;
    k.ch = SC::kGroupK;
    return k;
}

const Kernel& f8(int kw, int nt) {
    static const Kernel k[3][3] = {{make_sliced<2, 1>(), make_sliced<2, 2>(), make_sliced<2, 4>()},
                                   {make_sliced<4, 1>(), make_sliced<4, 2>(), make_sliced<4, 4>()},
                                   {make_sliced<8, 1>(), make_sliced<8, 2>(), make_sliced<8, 4>()}};
    return k[kw > 4 ? 2 : kw > 2 ? 1 : 0][nt > 2 ? 2 : nt > 1 ? 1 : 0];
}

const Kernel& bf16(int nt) {
    constexpr int B = (int) Format::Bf16;
    static const Kernel k1 = make_kernel<B, RS, 1, CH, ST, 1, MW>(), k2 = make_kernel<B, RS, 1, CH, ST, 2, MW>(),
                        k4 = make_kernel<B, RS, 1, CH, ST, 4, MW>();
    return nt > 2 ? k4 : nt > 1 ? k2 : k1;
}

const Kernel& q8(int nt) {
    constexpr int Q = (int) Format::Q8;
    static const Kernel k1 = make_kernel<Q, RS, 1, CH, ST, 1, MW>(), k2 = make_kernel<Q, RS, 1, CH, ST, 2, MW>(),
                        k4 = make_kernel<Q, RS, 1, CH, ST, 4, MW>();
    return nt > 2 ? k4 : nt > 1 ? k2 : k1;
}

}  // namespace basalt::kernels::wgemm::detail
