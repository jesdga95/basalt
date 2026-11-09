// src/models/qwen4exp/kernels/dec_moe_peer.cuh - the peer window kernel, on dec_moe.cuh's device code.  One TU a variant (dec_moe_peer_*.cu): this kernel is by far the slowest to compile.
#pragma once

namespace basalt::kernels {
namespace {

// The peer's shared memory: the expert rings' or the mixer recurrence's (never both at once in a block).
union DmPeerShared {
    DmShared d;
    RecSmem r;
};

// The peer window.  Per ring: block 0 waits for CUDA0's doorbell and copies the ring's header and q8_1 rows to the
// device (only block 0 reads host memory); a grid barrier; every block groups the peer's entries (the same lists) and
// runs its work items with the rows written straight to mapped memory; the last block to finish raises `done`.
// Every wait inside a ring is bounded: a down wait past kDmWaitLimitNs, a corrupt ring or the host's release aborts
// the window (a record in diag_m, done_m past every ring) and the kernel exits.
// Q2: Q2_0 gate/up rows two a warp as well (a build of its own, so the other formats' kernel stays as it was)
// MIX: the GDN mixer split's item (a build of its own, so the window without the split keeps its kernel)
template<bool Q2, bool MIX>
__global__ void __launch_bounds__(kDmThreads, 4) dm_peer_kernel(DmPeerArgs a) {
    __shared__ DmPeerShared su;
    DmShared& sm = su.d;
    __shared__ uint32_t s_seq;
    __shared__ int32_t s_ok;
    __shared__ float s_part[MIX ? kDecMaxT : 1][kDmThreads / 32];
    const int tid = threadIdx.x;
    const int nb = gridDim.x;
    float* gate = (float*) a.scratch;
    float* up = gate + (size_t) kDmCap * a.n_ff;
    block_q8_1* xq0 = (block_q8_1*) (up + (size_t) kDmCap * a.n_ff);
    DmRing* hdr = (DmRing*) (xq0 + (size_t) kVerifyMaxT * (a.n_embd / 32));
    unsigned int* done = (unsigned int*) (hdr + 1);
    float* wlog = (float*) (done + kDmCap + 4);
    unsigned long long* wlist = (unsigned long long*) (wlog + kVerifyMaxT * kDmPredExperts);
    // with the warm-up the rings alternate two row buffers: the router pass may still read a ring's rows while
    // block 0 fetches the next ring's
    block_q8_1* xq2 = (block_q8_1*) (wlist + kDmCap);
    const bool warm = a.rq != nullptr && a.rs != nullptr && nb > 1;
    // a.arrive (zeroed before every launch): [0] ring arrivals, [1] the grid barrier, [2] the abort word, [3, 8) the
    // warm-up's words
    unsigned int* bar = a.arrive + 1;
    unsigned int* abort = a.arrive + 2;
    unsigned int served = 0, bars = 0;
    if (blockIdx.x == 0) {   // whatever an earlier window left: the group counters start at zero
        for (int i = tid; i <= kDmCap; i += kDmThreads) done[i] = 0u;
        __threadfence();
    }
    DmWatch w;
    w.abort = abort; w.diag_m = a.diag_m; w.seq_m = a.seq_m; w.done_m = a.done_m; w.hdr_ids = hdr->ids;
    w.mdone_m = a.mix != nullptr ? a.mdone_m : nullptr;
    auto stamp = [&](int s, int i) {
        if (a.stamps_m != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            a.stamps_m[s * 8 + i] = t;
        }
    };
    for (int s = 0; s < a.steps; ++s) {
        const uint32_t ring = (uint32_t) s + 1u;
        block_q8_1* xq = warm && (s & 1) ? xq2 : xq0;
        w.ring = ring; w.layer = -1; w.n_tok = 0; w.n = 0;
        if (a.mix != nullptr && a.mix[s]) {   // the layer's mixer doorbell, ahead of its MoE ring
            if (blockIdx.x == 0) {
                if (tid == 0) {
                    uint32_t v = 0xffffffffu;
                    if (*(volatile unsigned int*) abort == 0u)
                        while (((v = *a.mseq_m) & 0xffffu) < ring) spin_pause();
                    __threadfence_system();
                    s_seq = v;
                    if (warm && s > 0)   // the warm-up stops
                        atomicMax(a.arrive + kDmWarmGo, v == 0xffffffffu ? 0xffffffffu : ring);
                }
                __syncthreads();
                const uint32_t v = s_seq;
                if (v == 0xffffffffu) {
                    if (tid == 0) hdr->n_tok = -1;
                } else {
                    const int nt = min((int) ((v >> 16) & 0xffu), kVerifyMaxT);
                    if (MIX && ((v >> 30) & 1u)) {   // the q8_1 rows and the fp32 rows (the gates' input)
                        const uint4* xs = (const uint4*) (a.mxq_m + (size_t) s * a.xq_stride);
                        const int n16 = nt * (a.n_embd / 32) * (int) sizeof(block_q8_1) / 16;
                        for (int i = tid; i < n16; i += kDmThreads) ((uint4*) a.mxq)[i] = __ldcv(xs + i);
                        const uint4* fs = (const uint4*) (a.mx_m + (size_t) s * (a.xq_stride / sizeof(block_q8_1) * 32));
                        float* xd = (float*) a.mixl[s].rec.x;
                        for (int i = tid; i < nt * a.n_embd / 4; i += kDmThreads) ((uint4*) xd)[i] = __ldcv(fs + i);
                    }
                    if (tid == 0) hdr->n_tok = (v >> 30) & 1u ? nt : 0;
                }
                __threadfence();
            }
            if (!dm_grid_sync(w, bar, ++bars * (unsigned int) nb, &s_ok)) return;
            if (hdr->n_tok < 0) return;          // released
            const int mt = hdr->n_tok;
            if constexpr (MIX) stamp(a.steps + s, 0);   // the mixer's phases: after the doorbell and each phase
            if constexpr (MIX) if (mt > 0) {   // the peer's key heads: qkv + z, the recurrence, the norm, the out partial
                const DmMixLayer& L = a.mixl[s];
                const DecRecArgs& R = L.rec;
                const int lane = tid & 31, gw = blockIdx.x * (kDmThreads / 32) + tid / 32, nw = nb * (kDmThreads / 32);
                const int ZV = R.h_v * 128;
                const block_q8_1* xin = (const block_q8_1*) a.mxq;
                dm_mix_rows(L.tq, L.wqkv, R.C, a.n_embd, xin, mt, (float*) R.qkv, R.C, gw, nw, lane);
                dm_mix_rows(L.tz, L.wz, ZV, a.n_embd, xin, mt, (float*) R.z, ZV, ((gw - R.C) % nw + nw) % nw, nw, lane);
                if (!dm_grid_sync(w, bar, ++bars * (unsigned int) nb, &s_ok)) return;
                stamp(a.steps + s, 1);
                if (tid < 128)   // one 128-thread item per (value head, rank), grid-strided
                    for (int it = blockIdx.x; it < 4 * R.h_v; it += nb) {
                        if (it != (int) blockIdx.x) rec_sync<true>();
                        rec_block<kDecMaxT, true, true>(R, mt, it >> 2, it & 3, su.r, nullptr);
                    }
                if (!dm_grid_sync(w, bar, ++bars * (unsigned int) nb, &s_ok)) return;
                stamp(a.steps + s, 2);
                for (int u = gw; u < mt * R.h_v; u += nw) {   // the gated norm and q8_1, one warp per (token, head)
                    const int t = u / R.h_v, head = u % R.h_v;
                    const size_t base = (size_t) t * ZV + head * 128 + lane;
                    float oc[4], q[4];
#pragma unroll
                    for (int r = 0; r < 4; ++r) oc[r] = R.o[base + 32 * r];
#pragma unroll
                    for (int r = 0; r < 4; ++r) q[r] = sq32(oc[r]);
                    const float ss = q[0] + q[1] + q[2] + q[3];
#pragma unroll
                    for (int r = 0; r < 4; ++r)
                        norm_q8(R, t, head, R.h_v, r, lane, oc[r], ss, R.gamma[32 * r + lane], R.z[base + 32 * r]);
                }
                if (!dm_grid_sync(w, bar, ++bars * (unsigned int) nb, &s_ok)) return;
                stamp(a.steps + s, 3);
                dm_mix_rows_m(L.to, L.wo, a.n_embd, ZV, (const block_q8_1*) R.yq, mt, a.part_m, a.n_embd, s_part);
                __threadfence_system();
            }
            if (!dm_grid_sync(w, bar, ++bars * (unsigned int) nb, &s_ok)) return;
            if constexpr (MIX) stamp(a.steps + s, 4);
            if (blockIdx.x == 0 && tid == 0) {
                __threadfence_system();
                *(volatile uint32_t*) a.mdone_m = ring;
                __threadfence_system();
                if (*(volatile unsigned int*) abort != 0u) *(volatile uint32_t*) a.mdone_m = 0xffffffffu;
            }
        }
        if (blockIdx.x == 0) {
            if (tid == 0) {
                uint32_t v = 0xffffffffu;   // an aborted window stops here
                if (*(volatile unsigned int*) abort == 0u)
                    while (((v = *a.seq_m) & 0xffffu) < ring) spin_pause();
                __threadfence_system();
                s_seq = v;
                if (warm && s > 0)   // the warm-up stops
                    atomicMax(a.arrive + kDmWarmGo, v == 0xffffffffu ? 0xffffffffu : ring);
            }
            __syncthreads();
            stamp(s, 0);
            const uint32_t v = s_seq;
            if (v == 0xffffffffu) {
                if (tid == 0) hdr->n_tok = -1;
            } else if ((v >> 28) & 3u) {         // the header and the ring's q8_1 rows, in one pass
                const int nt = (int) ((v >> 16) & 0xffu);
                const uint4* src = (const uint4*) (a.rings_m + s);
                const uint4* xs = (const uint4*) (a.xq_m + (size_t) s * a.xq_stride);
                constexpr int h16 = (int) (sizeof(DmRing) / 16);
                const int n16 = h16 + min(nt, kVerifyMaxT) * (a.n_embd / 32) * (int) sizeof(block_q8_1) / 16;
                // a thread's loads all go out before its first store: one round trip over the link, not one a pass
                constexpr int kLd = 4;
                for (int i0 = tid; i0 < n16; i0 += kLd * kDmThreads) {
                    uint4 t[kLd];
#pragma unroll
                    for (int r = 0; r < kLd; ++r) {
                        const int i = i0 + r * kDmThreads;
                        if (i < n16) t[r] = __ldcv(i < h16 ? src + i : xs + (i - h16));
                    }
#pragma unroll
                    for (int r = 0; r < kLd; ++r) {
                        const int i = i0 + r * kDmThreads;
                        if (i < h16) ((uint4*) hdr)[i] = t[r];
                        else if (i < n16) ((uint4*) xq)[i - h16] = t[r];
                    }
                }
            } else if (tid == 0) {
                hdr->n_tok = 0;
                hdr->n_peer = 0;
            }
            __threadfence();
        }
        if (!dm_grid_sync(w, bar, ++bars * (unsigned int) nb, &s_ok)) return;
        stamp(s, 1);
        if (hdr->n_tok < 0) return;              // released: the window is being abandoned
        const bool warm_next = warm && s + 1 < a.steps && blockIdx.x != 0;
        const int w_tok = min(hdr->n_tok, kVerifyMaxT), w_layer = hdr->layer;
        if (hdr->n_peer == 0) {                  // block 0 rewrites the header next ring
            if (!dm_grid_sync(w, bar, ++bars * (unsigned int) nb, &s_ok)) return;
            if (warm_next) dm_warm(a, sm, xq, w_tok, w_layer, ring + 1u, wlog, wlist, 0u);
            continue;
        }
        const int n_tok = min(hdr->n_tok, kVerifyMaxT), layer = hdr->layer, n = n_tok * a.k;
        w.layer = layer; w.n_tok = n_tok; w.n = n;
        unsigned long long blob = 0ull;   // a peer entry's blob, loaded while the block groups
        if (tid < n) {
            const int32_t id = hdr->ids[tid], kd = hdr->kind[tid];
            sm.ids[tid] = id;
            sm.kind[tid] = kd;
            if (kd == 2) blob = a.ptr[(size_t) layer * a.n_expert + id];
        }
        __syncthreads();
        dm_group(sm.ids, sm.kind, n, 2, tid, sm.first, sm.cnt, sm.gstart, sm.gidx, sm.pos, sm.start, &sm.ng, &sm.ne);
        if (tid < n && sm.kind[tid] == 2) {
            sm.tok[sm.pos[tid]] = tid / a.k;
            sm.dst[sm.pos[tid]] = tid;
            if (sm.first[tid]) sm.gptr[sm.gidx[tid]] = blob;
        }
        __syncthreads();
        // a group of more entries than the shared rows hold (a ring the route kernel did not write) aborts the window
        bool bad = false, big = false;
        for (int g = 0; g < sm.ng; ++g) {
            bad |= sm.gptr[g] == 0ull;
            big |= sm.start[g + 1] - sm.start[g] > kDmMaxEnt;
        }
        if (big && tid == 0) dm_abort(w, &sm, 2u, 0u, -1, 0u, 0u);
        stamp(s, 2);
        if (!bad && !big) {
            dm_experts_any<Q2>(sm, xq, a.lay[layer], gate, up, a.rows_m, done, blockIdx.x, nb, a.gu2 != 0, &w);
        }
        stamp(s, 3);
        __threadfence_system();
        ++served;
        if (dm_arrive(a.arrive, served * (unsigned int) nb, done, sm.ng)) {
            __threadfence_system();
            *(volatile uint32_t*) a.done_m = ring;
            __threadfence_system();
            // an abort raised meanwhile: its own store past every ring may have landed first
            if (*(volatile unsigned int*) abort != 0u) *(volatile uint32_t*) a.done_m = 0xffffffffu;
            if (a.stamps_m != nullptr) {
                unsigned long long t;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                a.stamps_m[s * 8 + 4] = t;
            }
        }
        if (warm_next) dm_warm(a, sm, xq, w_tok, w_layer, ring + 1u, wlog, wlist, served * (unsigned int) nb);
    }
}

}  // namespace
}  // namespace basalt::kernels
