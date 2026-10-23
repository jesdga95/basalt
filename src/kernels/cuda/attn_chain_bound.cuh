// Device-side wait bound for the persistent attention chain: every wait loop gives up after MKB_NS of globaltimer,
// reports where, and traps, so a lost hand-off ends the launch with an error instead of hanging the GPU.
#pragma once
#ifndef MKB_NS
#define MKB_NS 2000000000u
#endif
[[noreturn]] __device__ __noinline__ void mkb_hang(int code, int line) {   // out of line, never returns
    printf("basalt: error: attn_chain: wait timed out (kind %d, line %d) cta %d thread %d\n", code, line, (int) blockIdx.x, (int) threadIdx.x);
    asm volatile("membar.sys;\ntrap;" ::: "memory");
    while (true) {}
}
__device__ __forceinline__ unsigned mkb_now() { unsigned t; asm volatile("mov.u32 %0,%%globaltimer_lo;" : "=r"(t)); return t; }
__device__ __forceinline__ bool mkb_try(unsigned bar, unsigned ph) {
    unsigned d;
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(d) : "r"(bar), "r"(ph) : "memory");
    return d != 0;
}
__device__ __forceinline__ void mkb_mwait(unsigned bar, unsigned ph, int line) {
    if (mkb_try(bar, ph)) return;
    const unsigned t0 = mkb_now();
    while (!mkb_try(bar, ph)) if (mkb_now() - t0 > MKB_NS) mkb_hang(1, line);
}
