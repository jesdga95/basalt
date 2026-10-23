// src/kernels/cuda/gr.cu - the gated residual / hyper-connection, `gr_read` and `gr_write`.
//
// See include/basalt/kernels/gr.hpp for the semantics and for why the weights are bf16.
//
// `gr_read` is a weighted RMSNorm, the down MMVF, SiLU, the up MMVF, the gate + mean and the inject MMVF; the
// projections are the BF16 MMVF taking FP32 activations, and the norm, nonlinearity and combine live in
// `native_gr_norm.hpp` and `native_gr_postops.hpp`.  `gr_write` is one kernel.
//
// Layouts are the MANIFEST's: `hc_attn_down.weight` is [10240, 320] with ne0 = hc*n_embd fast, so it holds
// `w_down[k][i]` with i contiguous - the reference's (hc_lr, hc*n_embd) row-major.  `hc_attn_up.weight` is
// [320, 10240] so it holds `w_up[i][k]` with k contiguous - the reference's (hc*n_embd, hc_lr) row-major.  Both index
// with no permutation at all.
#include "basalt/kernels/gr.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/fused_gr.hpp"
#include "basalt/kernels/native_gr_norm.hpp"
#include "basalt/kernels/native_gr_postops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "basalt/log.hpp"

namespace basalt::kernels {

size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out) {
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    // ONE table, indexed by the same `k` that assigns the pointers below.  The first version of this function
    // sized three locals named `xn`, `xq`, `lq` in a DIFFERENT order from the pointer assignments - `xq` got
    // hc_lr*2 while `lq` got hc_dim*4 - so for the real geometry xq was given 640 bytes where it needed
    // 20,480 and every region overlapped its neighbour.
    //
    // What that looked like is worth recording: `inject` came out ~20% wrong, and rewriting its reduction
    // from block-wide to warp-wide produced BYTE-IDENTICAL wrong answers.  Two different reductions agreeing
    // exactly is not a coincidence to investigate - it means the reduction is not where the bug is, and the
    // inputs are.  Chasing the reduction instead cost most of a round.
    const size_t sz[3] = {
        hc_dim * sizeof(float),                // 0: xn
        hc_dim * sizeof(float),                // 1: gated
        (size_t) s.hc_lr * sizeof(float),      // 2: lo
    };
    size_t al[3], bytes = 0;
    for (int k = 0; k < 3; ++k) {
        al[k] = (sz[k] + 15) & ~(size_t) 15;
        bytes += al[k];
    }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char* p = (unsigned char*) base;
        void* ptr[3];
        for (int k = 0; k < 3; ++k) {
            ptr[k] = p;
            p += al[k];
        }
        out.xn = (float*) ptr[0];
        out.gated = (float*) ptr[1];
        out.lo = (float*) ptr[2];
    }
    return bytes;
}

void gr_read(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
             const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws, float* mixed,
             float* inject, void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.hc_lr <= 0) return;
    if (ws.xn == nullptr || ws.gated == nullptr || ws.lo == nullptr) {
        basalt::log::error("basalt", "gr_read: GrWorkspace is not initialised (see gr_workspace_init)");
        std::exit(1);
    }
    if (ws.bytes < gr_workspace_bytes(s)) {
        basalt::log::error("basalt", "gr_read: GrWorkspace is %zu bytes but this geometry needs %zu",
                     ws.bytes, gr_workspace_bytes(s));
        std::exit(1);
    }
    {   // matrices with 8-bit copies: the read on the copies
        FusedGrArgs fa;
        fa.R = R; fa.apply = false; fa.w_norm = w_norm; fa.w_down = w_down; fa.w_up = w_up; fa.w_inject = w_inject;
        fa.eps = eps; fa.lo = ws.lo; fa.rs = ws.gated; fa.inject_out = inject; fa.mixed = mixed;
        if (hc8_gr_read_multi(&fa, 1, s.n_embd, s.hc, stream)) {
            if (stream == nullptr && cudaDeviceSynchronize() != cudaSuccess) {
                basalt::log::error("basalt", "gr_read: the 8-bit read failed");
                std::exit(1);
            }
            return;
        }
    }
    const int n_embd = (int) s.n_embd, hc = (int) s.hc, hc_lr = (int) s.hc_lr;
    const int hc_dim = (int) (s.hc * s.n_embd);

    if ((hc_dim & 1) != 0 || (hc_lr & 1) != 0)
        throw std::invalid_argument("gr_read MMVF requires even hc*n_embd and hc_lr");
    native_gr_rms_norm_weighted(R, w_norm, ws.xn, n_embd, hc, eps, stream);
    bf16_gemv_fp32_mmvf(ws.xn, w_down, ws.lo, hc_dim, hc_lr, stream);
    native_gr_down_silu(ws.lo, hc_lr, hc, stream);
    bf16_gemv_fp32_mmvf(ws.lo, w_up, ws.gated, hc_lr, hc_dim, stream);
    // The null-injection contract identifies the final mixer: layers use the fused HC pre, il=-1 the unfused graph.
    native_gr_pre_gated(ws.xn, ws.gated, mixed, n_embd, hc, w_inject != nullptr, stream);
    // Absent for the final mixer, and then nothing is written - computing an injection nobody reads would be
    // a claim, not a convenience.
    if (w_inject != nullptr) bf16_gemv_fp32_mmvf(ws.xn, w_inject, inject, hc_dim, hc, stream);

    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "gr_read launch: %s", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) {
        const cudaError_t se = cudaDeviceSynchronize();
        if (se != cudaSuccess) {
            basalt::log::error("basalt", "gr_read: %s", cudaGetErrorString(se));
            std::exit(1);
        }
    }
}

void gr_write(const float* R, const float* block_out, const float* inject, const GrShapes& s, float* R_out,
              void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0) return;
    native_gr_post(R, block_out, inject, R_out, (int) s.n_embd, (int) s.hc, stream);
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            basalt::log::error("basalt", "gr_write: %s", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

}  // namespace basalt::kernels
