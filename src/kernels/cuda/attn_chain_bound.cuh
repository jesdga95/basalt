// Device-side wait bound for the persistent chain kernel (wait_abort.hpp): a wait that passes MKB_NS of globaltimer
// sets the sticky abort word and falls through, and a wait that has waited kWaitPollNs falls through once the word is
// set, so a lost hand-off ends the launch with garbage results (the host fails the window) instead of a trap.
#pragma once
#include "basalt/kernels/wait_abort.hpp"
#ifndef MKB_NS
#define MKB_NS 2000000000ull
#endif
static __device__ unsigned long long* mkb_word;   // the abort word's alias on this device (set by the host at init)
__device__ __forceinline__ unsigned long long mkb_now() { unsigned long long t; asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(t)); return t; }
// a wait started at t0 that has run kWaitPollNs: true when it gives up
__device__ __forceinline__ bool mkb_give_up(unsigned long long t0, int kind, int line) {
    return basalt::kernels::wait_give_up(mkb_word, mkb_now() - t0, (unsigned long long) MKB_NS,
                                         basalt::kernels::wait_abort_code_of(basalt::kernels::kWaitMidChain, (unsigned) kind, (unsigned) line, blockIdx.x));
}
__device__ __forceinline__ bool mkb_try(unsigned bar, unsigned ph) {
    unsigned d;
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(d) : "r"(bar), "r"(ph) : "memory");
    return d != 0;
}
__device__ __forceinline__ void mkb_mwait(unsigned bar, unsigned ph, int line) {
    if (mkb_try(bar, ph)) return;
    const unsigned long long t0 = mkb_now();
    while (!mkb_try(bar, ph))
        if (mkb_now() - t0 > basalt::kernels::kWaitPollNs && mkb_give_up(t0, 1, line)) return;
}
