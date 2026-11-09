// src/models/qwen4exp/kernels/dec_moe_kernels.cuh - the decode MoE tail's route, stage, wait and test kernels (one TU: dec_moe.cu), on dec_moe.cuh's device code.
#pragma once

namespace basalt::kernels {
namespace {

// Block 0 plans CUDA0's groups and publishes the ring's header; the other blocks quantize the FFN input (256 values
// each) into the device q8_1 rows and, for the tiers that need them, into mapped memory.  The last block to finish
// rings both doorbells.  Mode 1 keeps only the plan and the device rows, mode 2 only the ring (DmRouteArgs::mode).
__global__ void __launch_bounds__(kDmThreads) dm_route_kernel(DmRouteArgs a) {
    __shared__ int32_t s_ids[kDmCap], s_kind[kDmCap], s_first[kDmCap], s_cnt[kDmCap], s_gstart[kDmCap];
    __shared__ int32_t s_gidx[kDmCap], s_pos[kDmCap];
    __shared__ int32_t s_ncpu, s_npeer, s_ng, s_ne, s_nx[kDmMaxX];
    if (a.sh_gw != nullptr) pdl_trigger();   // the next kernel may start (only weights before its wait)
    const int tid = threadIdx.x, n = a.n_tok * a.k;
    pdl_wait();   // DEC_PDL2: launched during the router's tail
    if (tid == 0) { s_ncpu = 0; s_npeer = 0; }
    if (tid < kDmMaxX) s_nx[tid] = 0;
    if (tid < n) s_ids[tid] = a.ids[tid];
    __syncthreads();
    if (tid < n) {
        const int32_t e = s_ids[tid];
        // an id past the table or repeated within its token goes to the CPU pool: a device group then holds at most
        // one entry per token (kDmMaxEnt, the persistent kernels' shared rows)
        bool ok = e >= 0 && e < a.n_expert;
        for (int j = tid - tid % a.k; ok && j < tid; ++j) ok = s_ids[j] != e;
        s_kind[tid] = ok ? dm_tier(a, e) : -1;
    }
    __syncthreads();
    if (a.peer_cap > 0) {   // the peer's groups past peer_cap (distinct ids in routing order) go to the CPU pool
        __shared__ int32_t s_pf[kDmCap];
        int first = tid;
        const bool pe = tid < n && s_kind[tid] == 2;
        if (pe)
            for (int j = 0; j < tid; ++j)
                if (s_ids[j] == s_ids[tid] && s_kind[j] == 2) { first = j; break; }
        if (tid < kDmCap) s_pf[tid] = pe && first == tid;
        __syncthreads();
        int rank = 0;
        if (pe)
            for (int j = 0; j < first; ++j) rank += s_pf[j];
        __syncthreads();
        if (pe && rank >= a.peer_cap) s_kind[tid] = -1;
        __syncthreads();
    }
    dm_stage_split(a, s_ids, s_kind, n, tid, s_first);
    if (tid < n) {
        if (s_kind[tid] < 0) atomicAdd(&s_ncpu, 1);
        else if (s_kind[tid] == 2) atomicAdd(&s_npeer, 1);
        else if (s_kind[tid] >= 3) atomicAdd(&s_nx[s_kind[tid] - 3], 1);
    }
    __syncthreads();
    int npeers = s_npeer;   // every peer's entries
    for (int q = 0; q < a.n_x; ++q) npeers += s_nx[q];
    const bool plan = a.mode != 2, pub = a.mode != 1;
    if (blockIdx.x == 0 && !plan) {   // the ring's header only
        if (tid < n) {
            a.ring_m->ids[tid] = s_ids[tid];
            a.ring_m->kind[tid] = s_kind[tid];
            if (a.kinds) a.plan->kind[tid] = s_kind[tid];
        }
        if (a.kinds && tid == 0) {
            a.plan->counts[2] = npeers;
            a.plan->counts[3] = s_ncpu;
        }
        int ne = 0;
        for (int j = 0; j < n; ++j) ne += s_kind[j] == 0;
        if (tid == 0) {
            DmRing* r = a.ring_m;
            r->n_tok = a.n_tok;
            r->layer = a.layer;
            r->n_vram = ne;
            r->n_peer = s_npeer;
            r->n_cpu = s_ncpu;
        }
        dm_xring_headers(a, s_ids, s_kind, n, tid, s_nx, ne, s_ncpu);
    } else if (blockIdx.x == 0) {
        DmPlan* p = a.plan;
        dm_group(s_ids, s_kind, n, 0, tid, s_first, s_cnt, s_gstart, s_gidx, s_pos, p->start, &s_ng, &s_ne);
        if (tid < n) {
            const int32_t kd = s_kind[tid];
            p->kind[tid] = kd;
            if (kd == 0) {
                p->dst[s_pos[tid]] = tid;
                p->tok[s_pos[tid]] = tid / a.k;
                if (s_first[tid]) {
                    const int32_t slot = a.res0[s_ids[tid]];
                    p->ptr[s_gidx[tid]] = (unsigned long long) (a.cache_base + (a.slot_off ? (size_t) a.slot_off[slot]
                                                                                              : (size_t) slot * (size_t) a.blob));
                }
            }
            if (pub) {
                a.ring_m->ids[tid] = s_ids[tid];
                a.ring_m->kind[tid] = kd;
            }
        }
        if (tid == 0) {
            p->counts[0] = s_ng;
            p->counts[1] = s_ne;
            p->counts[2] = npeers;
            p->counts[3] = s_ncpu;
        }
        if (tid == 0 && pub) {
            DmRing* r = a.ring_m;
            r->n_tok = a.n_tok;
            r->layer = a.layer;
            r->n_vram = s_ne;
            r->n_peer = s_npeer;
            r->n_cpu = s_ncpu;
        }
        if (pub) {
            __syncthreads();   // s_ne is final
            dm_xring_headers(a, s_ids, s_kind, n, tid, s_nx, s_ne, s_ncpu);
        }
    } else {
        const long long i = (long long) (blockIdx.x - 1) * kDmThreads + tid;
        const long long nv = (long long) a.n_tok * a.n_embd;
        void* xq = plan ? a.xq : a.xq_pub;
        if (i < nv) {   // n_embd is a multiple of 256: whole warps
            const float xi = a.x[i];
            if (plan || npeers > 0 || a.rows) q8_1_store(xi, (block_q8_1*) xq, i);
            if (pub && s_ncpu > 0) a.x_m[i] = xi;
        }
        if (plan && a.sh_gw != nullptr) {   // the shared expert's gate: this block's 256 products, warps summed in order
            __shared__ float s_gp[kDmThreads / 32];
            float pr = 0.0f;
            if (i < nv) pr = __fmul_rn(a.x[i], __uint_as_float((uint32_t) a.sh_gw[i % a.n_embd] << 16));
            pr = warp_sum(pr);
            if ((tid & 31) == 0) s_gp[tid >> 5] = pr;
            __syncthreads();
            if (tid == 0) {
                float gs = s_gp[0];
#pragma unroll
                for (int w = 1; w < kDmThreads / 32; ++w) gs += s_gp[w];
                a.sh_part[blockIdx.x - 1] = gs;
            }
        }
        if (pub && (npeers > 0 || a.rows)) {
            __syncthreads();
            // this block's 8 q8_1 blocks (288 bytes) to mapped memory, 16 bytes per thread
            const long long b0 = (long long) (blockIdx.x - 1) * (kDmThreads / 32);
            if (b0 * 32 < nv && tid < 18) {
                const uint4* src = (const uint4*) ((const block_q8_1*) xq + b0);
                uint4* dst = (uint4*) ((block_q8_1*) a.xq_m + b0);
                dst[tid] = src[tid];
            }
        }
    }
    if (!pub) return;
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        const unsigned int arrived = atomicAdd(a.ctr, 1u) + 1u;
        if (arrived == gridDim.x) {
            *a.ctr = 0u;
            if (a.stats != nullptr) a.stats[0] = dm_clock();
            __threadfence_system();
            // the peer's word: the ring, the tokens and whether it has entries (it fetches nothing otherwise)
            *(volatile uint32_t*) a.peer_seq_m = a.ring | (uint32_t) a.n_tok << 16 | (s_npeer > 0 ? 1u << 28 : 0u) |
                                                 (a.rows ? 1u << 29 : 0u);
            dm_xring_doorbells(a, s_nx);
            *(volatile uint32_t*) a.host_seq_m = a.ring;
            __threadfence_system();
        }
    }
}

// The staged share's copy and plan (dec_moe_stage): every block derives the layer's kinds (the ring's on a card without
// peers) and the split, then copies its part of each staged blob from host RAM into its staging slot, whole 128-byte
// lines (a blob starts 16-byte aligned: its first line holds the bytes before it); block 0 writes the staged groups.
constexpr int kDmStageUnroll = 4;
__global__ void __launch_bounds__(kDmThreads) dm_stage_kernel(DmRouteArgs a, uint8_t* staging, long long stride,
                                                              long long blob) {
    __shared__ int32_t s_ids[kDmCap], s_kind[kDmCap], s_first[kDmCap], s_cnt[kDmCap], s_gstart[kDmCap];
    __shared__ int32_t s_gidx[kDmCap], s_pos[kDmCap], s_start[kDmCap + 1];
    __shared__ int32_t s_ng, s_ne;
    __shared__ unsigned long long s_src[kDmStageMax];
    const int tid = threadIdx.x, n = a.n_tok * a.k;
    if (tid < n) s_ids[tid] = __ldcg(a.ids + tid);
    __syncthreads();
    if (tid < n) {
        const int32_t e = s_ids[tid];
        bool ok = e >= 0 && e < a.n_expert;
        for (int j = tid - tid % a.k; ok && j < tid; ++j) ok = s_ids[j] != e;
        s_kind[tid] = ok && a.res0[e] >= 0 ? 0 : -1;
    }
    __syncthreads();
    dm_stage_split(a, s_ids, s_kind, n, tid, s_first);
    dm_group(s_ids, s_kind, n, 1, tid, s_first, s_cnt, s_gstart, s_gidx, s_pos, s_start, &s_ng, &s_ne);
    const int ng = min(s_ng, kDmStageMax);
    if (tid < n && s_first[tid] && s_gidx[tid] < kDmStageMax) s_src[s_gidx[tid]] = a.stage_src[s_ids[tid]];
    __syncthreads();
    if (blockIdx.x == 0) {
        DmPlan* p = a.plan;
        if (tid < n && s_kind[tid] == 1 && s_pos[tid] >= 0) {
            p->sdst[s_pos[tid]] = tid;
            p->stok[s_pos[tid]] = tid / a.k;
        }
        if (tid <= ng) p->sstart[tid] = s_start[tid];
        if (tid < ng) p->sptr[tid] = (unsigned long long) (staging + (long long) tid * stride) + (s_src[tid] & 127ull);
        if (tid == 0) {
            p->scounts[0] = ng;
            p->scounts[1] = s_ne;
        }
    }
    const long long step = (long long) gridDim.x * kDmThreads;
    for (int g = 0; g < ng; ++g) {
        const unsigned long long src = s_src[g];
        const uint4* s = (const uint4*) (src & ~127ull);
        uint4* d = (uint4*) (staging + (long long) g * stride);
        const long long chunks = (long long) ((src + (unsigned long long) blob + 127ull) / 128ull - src / 128ull) * 8;
        long long i = (long long) blockIdx.x * kDmThreads + tid;
        for (; i + (kDmStageUnroll - 1) * step < chunks; i += kDmStageUnroll * step) {
            uint4 v[kDmStageUnroll];
#pragma unroll
            for (int u = 0; u < kDmStageUnroll; ++u) v[u] = __ldcv(s + i + u * step);
#pragma unroll
            for (int u = 0; u < kDmStageUnroll; ++u) d[i + u * step] = v[u];
        }
        for (; i < chunks; i += step) d[i] = __ldcv(s + i);
    }
}

__global__ void dm_wait_kernel(const DmPlan* plan, const volatile uint32_t* peer_done, const volatile uint32_t* host_flag,
                               uint32_t ring, unsigned long long* stats, int layer, int trigger) {
    if ((trigger & 1) != 0) pdl_trigger();
    if (stats != nullptr) {
        stats[1] = dm_clock();
        dm_poll_tiers(plan, peer_done, host_flag, ring, stats);
        pdl_wait();
        dm_stat_record(stats, plan, layer);
        return;
    }
    if (plan->counts[2] > 0 && (trigger & 2) == 0) while (*peer_done < ring) spin_pause();
    if (plan->counts[3] > 0) while (*host_flag < ring) spin_pause();
    __threadfence_system();
    pdl_wait();   // started during the down kernel: its rows are complete once this returns
}

// the tests: the work items over given groups, without rings (one launch)
__global__ void __launch_bounds__(kDmThreads) dm_test_kernel(NativeExpertLayout L, const unsigned long long* grp_ptr,
                                                              const int32_t* start, int n_groups, const int32_t* tok,
                                                              const int32_t* dst, const block_q8_1* xq, float* gate,
                                                              float* up, float* out, unsigned int* sync, bool gu2) {
    __shared__ DmShared sm;
    if (threadIdx.x == 0) sm.ng = n_groups;
    for (int i = threadIdx.x; i <= n_groups; i += kDmThreads) {
        sm.start[i] = start[i];
        if (i < n_groups) sm.gptr[i] = grp_ptr[i];
    }
    for (int i = threadIdx.x; i < kDmCap; i += kDmThreads) {
        sm.tok[i] = tok[i];
        sm.dst[i] = dst[i];
    }
    __syncthreads();
    dm_experts_any(sm, xq, L, gate, up, out, sync + 1, blockIdx.x, gridDim.x, gu2);
    if (dm_arrive(sync, gridDim.x, sync + 1, n_groups)) sync[0] = 0u;
}


}  // namespace
}  // namespace basalt::kernels
