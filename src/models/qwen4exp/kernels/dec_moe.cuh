// src/models/qwen4exp/kernels/dec_moe.cuh - see include/basalt/models/qwen4exp/kernels/dec_moe.hpp.  The decode MoE tail's device code: types, helpers and
// kernel templates, with no kernel of its own in it, so a TU that includes it compiles only the kernels it names.  The kernels
// are in dec_moe_kernels.cuh and dec_moe_peer.cuh, the launchers in dec_moe_host.cuh.
#pragma once
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int kDmThreads = 256;

__device__ __forceinline__ unsigned long long dm_clock() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// DM_WAIT_STATS: one layer's times (S[0] doorbell, S[1] CUDA0's work done, S[2] the peer landed, S[3] every
// tier landed, S[4] the CPU pool landed) into the counters; one thread, after every time is stored
__device__ void dm_stat_record(unsigned long long* S, const DmPlan* plan, int layer) {
    const int np = plan->counts[2], nc = plan->counts[3];
    const unsigned long long tr = S[0], tc = S[1];
    const unsigned long long tp = np > 0 ? S[2] : tr, tl = (np > 0 || nc > 0) ? S[3] : tc;
    if (nc > 0) {
        unsigned long long* c = S + kDmStatCpu + 2 * min(nc, 7);
        c[0] += 1; c[1] += S[4] > tr ? S[4] - tr : 0ull;
    }
    const unsigned long long w = tl > tc ? tl - tc : 0ull;
    unsigned long long* b = S + kDmStatBucket + 8 * min(np, 15);
    const unsigned long long wp = np > 0 && tp > tc ? tp - tc : 0ull, wc = nc > 0 && S[4] > tc ? S[4] - tc : 0ull;
    b[0] += 1; b[1] += w > 0; b[2] += w; b[3] += tp > tr ? tp - tr : 0ull; b[4] += tc > tr ? tc - tr : 0ull;
    b[5] += nc > 0; b[6] += wp; b[7] += wc;
    if (layer >= 0 && layer < kDmStatLayers) {
        unsigned long long* l = S + kDmStatLayer + kDmStatPerLayer * layer;
        l[0] += (unsigned long long) np; l[1] += 1; l[2] += w; l[3] += wc; l[4] += wp;
        l[5] += tc > tr ? tc - tr : 0ull; l[6] += np > 0 && tp > tr ? tp - tr : 0ull; l[7] += np > 0;
    }
}

// ---------------------------------------------------------------- tier plan
// Entry i (tid < n) of kind `want`: its group (distinct ids in routing order) and position.  s_* are block arrays.
// Writes start[0..ng], and per entry pos[i] (its place in the entry list) for entries of kind `want`, -1 else.
__device__ __forceinline__ void dm_group(const int32_t* s_ids, const int32_t* s_kind, int n, int want, int tid,
                                         int32_t* s_first, int32_t* s_cnt, int32_t* s_gstart, int32_t* s_gidx,
                                         int32_t* s_pos, int32_t* start, int32_t* s_ng, int32_t* s_ne) {
    int first = tid, rank = 0, cnt = 0;
    const bool mine = tid < n && s_kind[tid] == want;
    if (mine) {
        const int32_t e = s_ids[tid];
        for (int j = 0; j < n; ++j)
            if (s_ids[j] == e && s_kind[j] == want) {
                if (j < first) first = j;
                if (j < tid) ++rank;
                ++cnt;
            }
    }
    if (tid < kDmCap) {
        s_first[tid] = mine && first == tid;
        s_cnt[tid] = mine && first == tid ? cnt : 0;
    }
    __syncthreads();
    if (mine && first == tid) {
        int gi = 0, es = 0;
        for (int j = 0; j < tid; ++j) { gi += s_first[j]; es += s_cnt[j]; }
        s_gidx[tid] = gi;
        s_gstart[tid] = es;
        start[gi] = es;
    }
    __syncthreads();
    if (tid < kDmCap) s_pos[tid] = mine ? s_gstart[first] + rank : -1;
    if (tid == 0) {
        int ng = 0, ne = 0;
        for (int j = 0; j < n; ++j) { ng += s_first[j]; ne += s_cnt[j]; }
        start[ng] = ne;
        *s_ng = ng;
        *s_ne = ne;
    }
    __syncthreads();
}

// dm_group's plan from a key array (the id of a device entry, -1 else; padded to a multiple of four) read four
// entries at a time.  Leaves s_pos to the caller (no closing barrier): entry tid's place is s_gstart[first] + rank.
__device__ __forceinline__ void dm_group_keys(const int32_t* s_key, int n, int tid, int32_t* s_first, int32_t* s_cnt,
                                              int32_t* s_gstart, int32_t* s_gidx, int32_t* start, int32_t* s_ng,
                                              int32_t* s_ne, int& first_out, int& rank_out) {
    static_assert(kDmCap == 128, "the scan takes four entries a lane");
    int first = tid, rank = 0, cnt = 0;
    const int32_t e = tid < n ? s_key[tid] : -1;
    const bool mine = e >= 0;
    if (mine) {
        const int n4 = (n + 3) & ~3;
#pragma unroll 4
        for (int j = 0; j < n4; j += 4) {
            const int4 k4 = *(const int4*) (s_key + j);
            const bool m0 = k4.x == e, m1 = k4.y == e, m2 = k4.z == e, m3 = k4.w == e;
            first = m3 && j + 3 < first ? j + 3 : first;
            first = m2 && j + 2 < first ? j + 2 : first;
            first = m1 && j + 1 < first ? j + 1 : first;
            first = m0 && j < first ? j : first;
            rank += (m0 && j < tid) + (m1 && j + 1 < tid) + (m2 && j + 2 < tid) + (m3 && j + 3 < tid);
            cnt += (int) m0 + (int) m1 + (int) m2 + (int) m3;
        }
    }
    const bool is_first = mine && first == tid;
    if (tid < kDmCap) {
        s_first[tid] = is_first;
        s_cnt[tid] = is_first ? cnt : 0;
    }
    __syncthreads();
    if (tid < 32) {
        int f[4], c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            f[q] = s_first[4 * tid + q];
            c[q] = s_cnt[4 * tid + q];
        }
        int fi = f[0] + f[1] + f[2] + f[3], ci = c[0] + c[1] + c[2] + c[3];
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int fu = __shfl_up_sync(0xffffffffu, fi, d), cu = __shfl_up_sync(0xffffffffu, ci, d);
            if (tid >= d) { fi += fu; ci += cu; }
        }
        const int ng = __shfl_sync(0xffffffffu, fi, 31), ne = __shfl_sync(0xffffffffu, ci, 31);
        int gi = fi - (f[0] + f[1] + f[2] + f[3]), es = ci - (c[0] + c[1] + c[2] + c[3]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (f[q]) {
                s_gidx[4 * tid + q] = gi;
                s_gstart[4 * tid + q] = es;
                start[gi] = es;
            }
            gi += f[q];
            es += c[q];
        }
        if (tid == 0) {
            start[ng] = ne;
            *s_ng = ng;
            *s_ne = ne;
        }
    }
    __syncthreads();
    first_out = first;
    rank_out = rank;
}

// The staged share (a card without peers): of the layer's distinct CPU misses (an entry of kind -1 that is its expert's
// first such entry), the last stage_k[m] in routing order whose blob has a host-RAM address go to kind 1 with every
// entry of theirs (m: the layer's distinct CPU misses).  Every thread of the block calls it; q is kDmCap words of
// block scratch.  The ring's kinds and the staged copy (dm_stage_kernel) make the same split from the same ids and
// residency.
__device__ void dm_stage_split(const DmRouteArgs& a, const int32_t* s_ids, int32_t* s_kind, int n, int tid, int32_t* q) {
    if (a.stage_src == nullptr) return;
    const bool in = tid < n;
    const int32_t e = in ? s_ids[tid] : -1;
    const bool miss = in && s_kind[tid] < 0 && e >= 0 && e < a.n_expert;
    bool first = miss;
    for (int j = 0; first && j < tid; ++j)
        if (s_ids[j] == e && s_kind[j] < 0) first = false;
    if (tid < kDmCap) q[tid] = first;
    __syncthreads();
    int m = 0;
    for (int j = 0; j < n; ++j) m += q[j];
    const int k = a.stage_k[m];
    if (k <= 0) {
        __syncthreads();
        return;
    }
    bool el = first && a.stage_src[e] != 0ull;
    for (int j = 0; el && j < n; ++j)   // a group holds one entry per token
        if (j != tid && s_ids[j] == e && j / a.k == tid / a.k) el = false;
    __syncthreads();
    if (tid < kDmCap) q[tid] = el;
    __syncthreads();
    bool st = false;
    if (el) {
        int after = 0;
        for (int j = tid + 1; j < n; ++j) after += q[j];
        st = after < k;
    }
    __syncthreads();
    if (tid < kDmCap) q[tid] = st;
    __syncthreads();
    bool staged = false;
    for (int j = 0; miss && j < n; ++j)
        if (q[j] && s_ids[j] == e) staged = true;
    __syncthreads();
    if (staged) s_kind[tid] = 1;
    __syncthreads();
}

// The tier of expert e from CUDA0's and the peer's residency rows res0 / res1 (global or shared memory; res1 null: no
// peer) and the further peers' rows: CUDA0, the peer, a further peer (3 + q) or the CPU pool (-1).
__device__ __forceinline__ int dm_tier_rows(const DmRouteArgs& a, const int32_t* res0, const int32_t* res1, int32_t e) {
    if (res0[e] >= 0) return 0;
    if (res1 != nullptr && res1[e] >= 0) return 2;
    for (int q = 0; q < a.n_x; ++q)
        if (a.xres[q][e] >= 0) return 3 + q;
    return -1;
}
__device__ __forceinline__ int dm_tier(const DmRouteArgs& a, int32_t e) { return dm_tier_rows(a, a.res0, a.res1, e); }

// The further peers' ring headers: each holds its own entries as kind 2 and every other GPU's as 0.
__device__ void dm_xring_headers(const DmRouteArgs& a, const int32_t* ids, const int32_t* kind, int n, int tid,
                                 const int32_t* nx, int n_vram, int n_cpu) {
    for (int q = 0; q < a.n_x; ++q) {
        DmRing* r = a.xring_m[q];
        if (tid < n) {
            const int32_t kd = kind[tid];
            r->ids[tid] = ids[tid];
            r->kind[tid] = kd == 3 + q ? 2 : kd >= 0 ? 0 : -1;
        }
        if (tid == 0) {
            r->n_tok = a.n_tok;
            r->layer = a.layer;
            r->n_vram = n_vram;
            r->n_peer = nx[q];
            r->n_cpu = n_cpu;
        }
    }
}

__device__ void dm_xring_doorbells(const DmRouteArgs& a, const int32_t* nx) {
    for (int q = 0; q < a.n_x; ++q)
        *(volatile uint32_t*) a.xseq_m[q] = a.ring | (uint32_t) a.n_tok << 16 | (nx[q] > 0 ? 1u << 28 : 0u) |
                                            (a.rows ? 1u << 29 : 0u);
}

// both tiers' words polled together, each one's landing time kept (S[2] the peer, S[4] the CPU pool, S[3] both)
__device__ void dm_poll_tiers(const DmPlan* plan, const volatile uint32_t* peer_done, const volatile uint32_t* host_flag,
                              uint32_t ring, unsigned long long* S) {
    bool pd = plan->counts[2] == 0, cd = plan->counts[3] == 0;
    while (!pd || !cd) {
        if (!pd && *peer_done >= ring) { pd = true; S[2] = dm_clock(); }
        if (!cd && *host_flag >= ring) { cd = true; S[4] = dm_clock(); }
        if (!pd || !cd) __nanosleep(32);
    }
    __threadfence_system();
    S[3] = dm_clock();
}

// ---------------------------------------------------------------- DEC_GLUE & 1: the route inside gate/up
// dm_route_kernel's mode 2 kinds (with the peer's cap) for the ring's entries r_ids (shared memory) from the
// residency rows res0 / res1 (dm_tier_rows), the counts into r_cnt (zeroed: CPU, peer, VRAM, then each further
// peer's), then the ring's header and every entry's kind (mapped, the further peers' rings, and the plan's with
// a.kinds).  Ends without a barrier after the header's stores.
__device__ void dm_ring_head(const DmRouteArgs& a, const int32_t* res0, const int32_t* res1, int tid,
                             const int32_t* r_ids, int32_t* r_kind, int32_t* r_pf, int32_t* r_cnt) {
    const int n = a.n_tok * a.k;
    if (tid < n) {
        const int32_t e = r_ids[tid];
        bool ok = e >= 0 && e < a.n_expert;
        for (int j = tid - tid % a.k; ok && j < tid; ++j) ok = r_ids[j] != e;
        r_kind[tid] = ok ? dm_tier_rows(a, res0, res1, e) : -1;
    }
    __syncthreads();
    if (a.peer_cap > 0) {   // dm_route_kernel's cap: the peer's groups past peer_cap go to the CPU pool
        int first = tid;
        const bool pe = tid < n && r_kind[tid] == 2;
        if (pe)
            for (int j = 0; j < tid; ++j)
                if (r_ids[j] == r_ids[tid] && r_kind[j] == 2) { first = j; break; }
        if (tid < kDmCap) r_pf[tid] = pe && first == tid;
        __syncthreads();
        int rank = 0;
        if (pe)
            for (int j = 0; j < first; ++j) rank += r_pf[j];
        __syncthreads();
        if (pe && rank >= a.peer_cap) r_kind[tid] = -1;
        __syncthreads();
    }
    dm_stage_split(a, r_ids, r_kind, n, tid, r_pf);
    if (tid < n) {
        if (r_kind[tid] < 0) atomicAdd(&r_cnt[0], 1);
        else if (r_kind[tid] == 2) atomicAdd(&r_cnt[1], 1);
        else if (r_kind[tid] >= 3) atomicAdd(&r_cnt[r_kind[tid]], 1);   // further peer q at 3 + q
        else atomicAdd(&r_cnt[2], 1);
    }
    __syncthreads();
    int npeers = r_cnt[1];   // every peer's entries
    for (int q = 0; q < a.n_x; ++q) npeers += r_cnt[3 + q];
    dm_xring_headers(a, r_ids, r_kind, n, tid, r_cnt + 3, r_cnt[2], r_cnt[0]);
    if (tid < n) {
        a.ring_m->ids[tid] = r_ids[tid];
        a.ring_m->kind[tid] = r_kind[tid];
        if (a.kinds) a.plan->kind[tid] = r_kind[tid];
    }
    if (tid == 0) {
        DmRing* r = a.ring_m;
        r->n_tok = a.n_tok;
        r->layer = a.layer;
        r->n_vram = r_cnt[2];
        r->n_peer = r_cnt[1];
        r->n_cpu = r_cnt[0];
        if (a.kinds) {
            a.plan->counts[2] = npeers;
            a.plan->counts[3] = r_cnt[0];
        }
    }
}

// DEC_GLUE & 2: dm_route_kernel's mode 2 (with kinds) in one block, the mapped q8_1 rows copied from `xq` (the
// same q8_1_store bits); called after the block's pdl_wait
__device__ void dm_glue_ring(const DmRouteArgs& a, const void* xq, int tid) {
    __shared__ int32_t r_ids[kDmCap], r_kind[kDmCap], r_pf[kDmCap], r_cnt[3 + kDmMaxX];
    const int n = a.n_tok * a.k;
    if (tid < 3 + kDmMaxX) r_cnt[tid] = 0;
    const int n16 = a.n_tok * (a.n_embd / 32) * 36 / 16;
    if (tid < n) r_ids[tid] = __ldcg(a.ids + tid);
    __syncthreads();
    dm_ring_head(a, a.res0, a.res1, tid, r_ids, r_kind, r_pf, r_cnt);
    int npeers = r_cnt[1];
    for (int q = 0; q < a.n_x; ++q) npeers += r_cnt[3 + q];
    if (npeers > 0 || a.rows) {   // the q8_1 rows, 16 bytes at a time (a row is n_embd / 32 blocks of 36 bytes)
        for (int i = tid; i < n16; i += blockDim.x) ((uint4*) a.xq_m)[i] = __ldcg((const uint4*) xq + i);
    }
    if (r_cnt[0] > 0) {
        const int n4 = a.n_tok * a.n_embd / 4;
        for (int i = tid; i < n4; i += blockDim.x) ((float4*) a.x_m)[i] = __ldcg((const float4*) a.x + i);
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        if (a.stats != nullptr) a.stats[0] = dm_clock();
        __threadfence_system();
        *(volatile uint32_t*) a.peer_seq_m = a.ring | (uint32_t) a.n_tok << 16 | (r_cnt[1] > 0 ? 1u << 28 : 0u) |
                                             (a.rows ? 1u << 29 : 0u);
        dm_xring_doorbells(a, r_cnt + 3);
        *(volatile uint32_t*) a.host_seq_m = a.ring;
        __threadfence_system();
    }
}

__device__ __forceinline__ unsigned dm_ld_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// the layer's landed word, read at the kernel's start (`have`) and spun on only while it is short of `need`
__device__ __forceinline__ void dm_landed_read(const DmGlueArgs& a, unsigned& need, unsigned& have) {
    if (a.landed == nullptr) return;
    need = __ldcg(a.landed_need);
    have = dm_ld_acquire(a.landed);
}
// (the mapped stop word is read once in a while: every block polling it would crowd the link the copies come over)
__device__ __forceinline__ void dm_landed_wait(const DmGlueArgs& a, unsigned need, unsigned have) {
    if (a.landed == nullptr || have >= need) return;
    for (unsigned i = 1; dm_ld_acquire(a.landed) < need; ++i) {
        if ((i & 1023u) == 0u && a.landed_stop != nullptr && *(const volatile unsigned*) a.landed_stop == 0xffffffffu) break;
        __nanosleep(128);
    }
}

// The ring's block after a read that counts its progress (DmRouteArgs::pub_ctr), started with the read: the layer's
// residency rows into shared memory (s_res: 2 * n_expert words); once the up slices are through, every token's FFN
// input rows to mapped memory while the router runs; once the router items have written xq, its q8_1 rows while the
// top-10 runs; once the ids are out, the kinds and the header, then one fence and every doorbell.  It reads nothing the
// read writes before those counts, so it needs no pdl_wait; with `reset` it zeroes them for the layer's next window.
__device__ void dm_glue_ring_pub(const DmRouteArgs& a, const void* xq, int tid, int32_t* s_res, bool reset) {
    __shared__ int32_t r_ids[kDmCap], r_kind[kDmCap], r_pf[kDmCap], r_cnt[3 + kDmMaxX];
    const int n = a.n_tok * a.k;
    int32_t* const res0 = s_res;
    int32_t* const res1 = a.res1 != nullptr ? s_res + a.n_expert : nullptr;
    for (int i = tid; i < a.n_expert; i += blockDim.x) {
        res0[i] = a.res0[i];
        if (res1 != nullptr) res1[i] = a.res1[i];
    }
    if (tid < 3 + kDmMaxX) r_cnt[tid] = 0;
    if (tid == 0)
        while (dm_ld_acquire(a.pub_ctr + 2) == 0u) __nanosleep(32);
    __syncthreads();
    const int n4 = a.n_tok * (a.n_embd / 4);   // the FFN input rows the CPU pool reads, 16 bytes at a time
    for (int i = tid; i < n4; i += blockDim.x) ((float4*) a.x_m)[i] = __ldcg((const float4*) a.x + i);
    if (tid == 0)
        while (dm_ld_acquire(a.pub_ctr) < a.pub_target) __nanosleep(32);
    __syncthreads();
    const int n16 = a.n_tok * (a.n_embd / 32) * 36 / 16;   // the q8_1 rows, 16 bytes at a time
    for (int i = tid; i < n16; i += blockDim.x) ((uint4*) a.xq_m)[i] = __ldcg((const uint4*) xq + i);
    if (tid == 0) {
        while (dm_ld_acquire(a.pub_ctr + 1) == 0u) __nanosleep(32);
        if (reset) {
            a.pub_ctr[0] = 0u;
            a.pub_ctr[1] = 0u;
            a.pub_ctr[2] = 0u;
        }
    }
    __syncthreads();
    if (tid < n) r_ids[tid] = __ldcg(a.ids + tid);
    __syncthreads();
    dm_ring_head(a, res0, res1, tid, r_ids, r_kind, r_pf, r_cnt);
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        if (a.stats != nullptr) a.stats[0] = dm_clock();
        *(volatile uint32_t*) a.peer_seq_m = a.ring | (uint32_t) a.n_tok << 16 | (r_cnt[1] > 0 ? 1u << 28 : 0u) |
                                             (a.rows ? 1u << 29 : 0u);
        dm_xring_doorbells(a, r_cnt + 3);
        *(volatile uint32_t*) a.host_seq_m = a.ring;
        __threadfence_system();
    }
}

// native_gu2_kernel's rows with the VRAM plan built in the block (dm_route_kernel's kinds and dm_group for the VRAM
// tier: an entry is CUDA0's when its id is valid, not repeated within its token and ptab holds its blob).  The blocks
// are co-resident and loop over the (row block, group) items, so each plans once.  Two blocks an SM: the row loop
// keeps its full register budget.
//
// The plan keeps one mask of tokens per expert id (shared atomics, so no pass over the other entries); each entry
// reads its rank, its group's size and whether it opens the group off that mask.  Group indices and starts come from
// warp scans and the warp totals.  Entry order, group order and places are those of dm_group_keys.
constexpr int kDmGlueExperts = 1024;   // expert ids the mask table covers
constexpr int kDmGlueTokens = 16;      // tokens a mask holds

template<int TG, int TU = TG>
__global__ void __launch_bounds__(256, 2) dm_glue_gu_kernel(DmGlueArgs a, NativeExpertLayout L,
                                                            float* __restrict__ gate, float* __restrict__ up,
                                                            block_q8_1* __restrict__ hq) {
    __shared__ int32_t s_ids[kDmCap], s_start[kDmCap + 1], s_tok[kDmCap];
    __shared__ unsigned long long s_ptr[kDmCap];
    __shared__ uint32_t s_tab[kDmGlueExperts];
    __shared__ int2 s_wtot[4];
    __shared__ int s_dup;
    __shared__ alignas(16) uint32_t s_grid_buf[gu2::Fmt2<TG>::kGridWords];
    const int tid = threadIdx.x, n = a.n_tok * a.k;
    const bool lead = blockIdx.x == 0;
    if (!lead) pdl_trigger();   // the lead block lets the SwiGLU and down launches start once the plan is out
    const int nwork = (int) gridDim.x - (a.ring_on ? 1 : 0);
    if (a.ring_on && (int) blockIdx.x == nwork) {   // the ring's block
        if (a.ring.pub_ctr != nullptr) {
            dm_glue_ring_pub(a.ring, a.xq, tid, (int32_t*) s_tab, true);
            return;
        }
        if (tid < 32 && tid * 32 < a.n_expert) {   // the layer's residency rows into L2
            asm volatile("prefetch.global.L2 [%0];" ::"l"(a.ring.res0 + tid * 32));
            if (a.ring.res1 != nullptr) asm volatile("prefetch.global.L2 [%0];" ::"l"(a.ring.res1 + tid * 32));
        }
        pdl_wait();
        dm_glue_ring(a.ring, a.xq, tid);
        return;
    }
    unsigned lw_need = 0u, lw_have = 0u;
    if (tid == 0) dm_landed_read(a, lw_need, lw_have);
    for (int i = tid; i < a.n_expert; i += 256) s_tab[i] = 0u;
    if (tid == 0) s_dup = 0;
    const uint32_t* s_grid = gu2::stage_grid<TG>(s_grid_buf, tid);
    if (tid == 0 && (int) blockIdx.x * 16 < a.n_expert)   // the layer's table into L2
        asm volatile("prefetch.global.L2 [%0];" ::"l"(a.ptab + blockIdx.x * 16));
    const int tok = tid / a.k, warp = tid >> 5, lane = tid & 31;
    __syncthreads();   // the masks are zero before any entry marks one
    pdl_wait();
    if (tid == 0) dm_landed_wait(a, lw_need, lw_have);   // before the plan: no launch after this one reads a blob early
    // the table entry is read at once; the marking and the barrier hide its latency
    const int32_t e0 = tid < n ? __ldcg(a.ids + tid) : -1;
    const bool inr = e0 >= 0 && e0 < a.n_expert;
    const unsigned long long praw = inr ? a.ptab[e0] : 0ull;
    if (tid < n) s_ids[tid] = e0;
    if (inr && (atomicOr(&s_tab[e0], 1u << tok) >> tok & 1u) != 0u) s_dup = 1;   // a repeat within a token
    __syncthreads();
    // an entry counts once per token and only when ptab holds its blob (a function of the id: a mask never mixes)
    bool dup = false;
    if (s_dup != 0)
        for (int j = inr ? tok * a.k : tid; j < tid; ++j) dup |= s_ids[j] == e0;
    const bool valid = inr && praw != 0ull && !dup;
    const uint32_t bits = inr ? s_tab[e0] : 0u;
    const uint32_t low = bits & ((1u << tok) - 1u);   // the earlier tokens that route to the expert
    const bool is_first = valid && low == 0u;
    const int gcnt = is_first ? __popc(bits) : 0;
    int cinc = gcnt;
    unsigned fb = 0u;
    if (warp < 4) {   // entries are at most kDmCap
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, cinc, d);
            if (lane >= d) cinc += v;
        }
        fb = __ballot_sync(0xffffffffu, is_first);
        if (lane == 31) s_wtot[warp] = make_int2(__popc(fb), cinc);
    }
    __syncthreads();
    int g0 = 0, e_off = 0, ng = 0, ne = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const int2 t = s_wtot[w];
        if (w < warp) { g0 += t.x; e_off += t.y; }
        ng += t.x;
        ne += t.y;
    }
    if (is_first) {
        const int gi = g0 + __popc(fb & ((1u << lane) - 1u)), gs = e_off + cinc - gcnt;
        s_ptr[gi] = praw;
        s_start[gi] = gs;
        int r = 0;
        for (uint32_t m = bits; m != 0u; m &= m - 1u) s_tok[gs + r++] = __ffs(m) - 1;   // the group's tokens in order
        s_tab[e0] = bits | (uint32_t) gs << 16;   // the group's start, for the lead block's places
    }
    if (tid == 0) s_start[ng] = ne;
    __syncthreads();
    if (lead) {
        DmPlan* P = a.plan;
        if (valid) {
            const int pos = (int) (s_tab[e0] >> 16) + __popc(low);
            P->dst[pos] = tid;
            P->tok[pos] = tok;
        }
        if (tid < ng) P->ptr[tid] = s_ptr[tid];
        if (tid <= ng) P->start[tid] = s_start[tid];
        if (tid == 0) {
            P->counts[0] = ng;
            P->counts[1] = ne;
        }
        __threadfence();
        __syncthreads();
        pdl_trigger();
    }
    const int nx = (int) (2 * L.n_ff + GU2_ROWS - 1) / GU2_ROWS;
    const int items = nx * ng;   // one (row block, group) item each: every group's rows spread over all blocks
    const auto* xq = (const block_q8_1*) a.xq;
    for (int it = blockIdx.x; it < items; it += nwork) {
        const int bx = it % nx, g = it / nx;
        const int row = bx * GU2_ROWS + 2 * warp;   // rows row, row + 1: both gate or both up
        if (row < 2 * L.n_ff)
            gu2::pair_gu<TG, TU>((const uint8_t*) s_ptr[g], row, s_start[g], s_start[g + 1], s_tok, xq, L, gate, up,
                                 s_grid, lane);
    }
}

// co-resident blocks on the current device, an eighth of the SMs left to the side streams (shared expert, ring)
template<int TG, int TU = TG> int dm_glue_blocks() {
    static int b[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    if (b[dev] == 0) {
        int sms = 0, per = 0;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, dm_glue_gu_kernel<TG, TU>, 256, 0) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        b[dev] = std::max(1, (sms - sms / 8) * per);
    }
    return b[dev];
}

// dm_glue_gu_kernel for gate/up formats without gu2 row pairs (NVFP4): the same VRAM plan built in the block, then
// native_gu_multi_kernel's block rows (dm_gu_item, the same calls in the same order) as the items.  Four blocks an SM:
// the row loop is latency-bound and needs the occupancy more than the registers.  The blocks stay resident until the
// items run out, so an eighth of the SMs is left to the side streams (the shared expert); the items are handed out by
// a counter as blocks free up, so the blocks finish together.  The ring's block is block 1, resident from the start.
// dm_glue_gu1_kernel's item counter and finished-block count (one gate/up launch at a time on a device)
__device__ unsigned int g_dm_glue1_work[2];

template<int TG>
__global__ void __launch_bounds__(256, 4) dm_glue_gu1_kernel(DmGlueArgs a, NativeExpertLayout L,
                                                             float* __restrict__ gate, float* __restrict__ up,
                                                             block_q8_1* __restrict__ hq);

// ---------------------------------------------------------------- the grouped rows, one work item at a time
// native_gu_multi_kernel's body for block row `rb` of one group (entries [e0, e1)): the same calls in the same order.
// gate/up row `row` (0 .. 2 n_ff: gate rows, then up rows) at `wr`, for entries [e0, e1)
// `blob` is the expert's blob (its output scales: expert_out_scale).
template<int TG, bool STAGE_GRID>
__device__ __forceinline__ void dm_gu_row(const uint8_t* wr, int row, int e0, int e1, const int32_t* __restrict__ ent_tok,
                                          const block_q8_1* __restrict__ xq, const NativeExpertLayout& L,
                                          float* __restrict__ gate, float* __restrict__ up, const uint32_t* s_grid,
                                          int lane, unsigned long long blob) {
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    float* dst = is_up ? up : gate;
    [[maybe_unused]] const float sc = expert_out_scale<TG>(blob, L, is_up ? 1 : 0);
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        if (n == 1) {
            const int off1[1] = { ent_tok[e] * xb };
            float s1[1];
            row_dot_multi<TG, 1, true, STAGE_GRID>(wr, xq, off1, 1, nb, lane, s1, s_grid);
            tail_scale<TG>(s1, sc);
            if (lane == 0) dst[(size_t) e * L.n_ff + r] = s1[0];
        } else if (n == 2) {
            const int off2[2] = { ent_tok[e] * xb, ent_tok[e + 1] * xb };
            float s2[2];
            row_dot_multi<TG, 2, true, STAGE_GRID>(wr, xq, off2, 2, nb, lane, s2, s_grid);
            tail_scale<TG>(s2, sc);
            if (lane < 2) dst[(size_t) (e + lane) * L.n_ff + r] = s2[lane];
        } else {
            int off[GRP_NC];
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = ent_tok[e + min(c, n - 1)] * xb;
            float s[GRP_NC];
            row_dot_multi<TG, GRP_NC, false, STAGE_GRID>(wr, xq, off, n, nb, lane, s, s_grid);
            tail_scale<TG>(s, sc);
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) dst[(size_t) (e + c) * L.n_ff + r] = s[c];
        }
    }
}

// TU: the up rows' type; a mixed layer's up rows read their grid from the device table (s_grid is the gate's)
template<int TG, bool STAGE_GRID, int TU = TG>
__device__ __forceinline__ void dm_gu_item(const uint8_t* blob, int e0, int e1, const int32_t* __restrict__ ent_tok,
                                           const block_q8_1* __restrict__ xq, const NativeExpertLayout& L, int rb,
                                           float* __restrict__ gate, float* __restrict__ up, const uint32_t* s_grid,
                                           int warp, int lane) {
    const int row = rb * GU_ROWS + warp;
    if (row >= 2 * L.n_ff) return;
    if constexpr (TG != TU) {
        if (row >= L.n_ff) {
            dm_gu_row<TU, false>(blob + L.up_off + (size_t) (row - L.n_ff) * L.up_row, row, e0, e1, ent_tok, xq, L,
                                 gate, up, nullptr, lane, (unsigned long long) blob);
            return;
        }
    }
    // a uniform layer's gate and up rows are contiguous in the blob: row `row` is at row * gu_row
    dm_gu_row<TG, STAGE_GRID>(blob + (size_t) row * L.gu_row, row, e0, e1, ent_tok, xq, L, gate, up, s_grid, lane,
                              (unsigned long long) blob);
}

template<int TG>
__global__ void __launch_bounds__(256, 4) dm_glue_gu1_kernel(DmGlueArgs a, NativeExpertLayout L,
                                                             float* __restrict__ gate, float* __restrict__ up,
                                                             block_q8_1* __restrict__ hq) {
    (void) hq;
    __shared__ int32_t s_ids[kDmCap], s_start[kDmCap + 1], s_tok[kDmCap];
    __shared__ unsigned long long s_ptr[kDmCap];
    __shared__ uint32_t s_tab[kDmGlueExperts];
    __shared__ int2 s_wtot[4];
    __shared__ int s_dup;
    const int tid = threadIdx.x, n = a.n_tok * a.k;
    const bool lead = blockIdx.x == 0;
    if (!lead) pdl_trigger();
    const int nwork = (int) gridDim.x - (a.ring_on ? 1 : 0);
    // after a read that counts its progress the blocks start on its counts, not its end, and the last block of the
    // launch to finish (the ring's block among them) zeroes the counts
    const bool counted = a.ring_on && a.ring.pub_ctr != nullptr;
    const unsigned finishers = (unsigned) nwork + (counted ? 1u : 0u);
    auto finish = [&] {
        if (tid == 0) {
            __threadfence();
            if (atomicAdd(&g_dm_glue1_work[1], 1u) == finishers - 1u) {
                g_dm_glue1_work[0] = 0u;
                g_dm_glue1_work[1] = 0u;
                if (counted) {
                    a.ring.pub_ctr[0] = 0u;
                    a.ring.pub_ctr[1] = 0u;
                    a.ring.pub_ctr[2] = 0u;
                }
            }
        }
    };
    if (a.ring_on && blockIdx.x == 1) {   // the ring's block
        if (counted) {
            dm_glue_ring_pub(a.ring, a.xq, tid, (int32_t*) s_tab, false);
            finish();
            return;
        }
        if (tid < 32 && tid * 32 < a.n_expert) {   // the layer's residency rows into L2
            asm volatile("prefetch.global.L2 [%0];" ::"l"(a.ring.res0 + tid * 32));
            if (a.ring.res1 != nullptr) asm volatile("prefetch.global.L2 [%0];" ::"l"(a.ring.res1 + tid * 32));
        }
        pdl_wait();
        dm_glue_ring(a.ring, a.xq, tid);
        return;
    }
    const int wb = (int) blockIdx.x - (a.ring_on && blockIdx.x > 1 ? 1 : 0);   // the block's place among the workers
    unsigned lw_need = 0u, lw_have = 0u;
    if (tid == 0) dm_landed_read(a, lw_need, lw_have);
    for (int i = tid; i < a.n_expert; i += 256) s_tab[i] = 0u;
    if (tid == 0) s_dup = 0;
    if (tid == 0 && wb * 16 < a.n_expert)   // the layer's table into L2
        asm volatile("prefetch.global.L2 [%0];" ::"l"(a.ptab + wb * 16));
    const int tok = tid / a.k, warp = tid >> 5, lane = tid & 31;
    if (counted) {   // xq and the ids are out once both counts are in
        if (tid == 0) {
            while (dm_ld_acquire(a.ring.pub_ctr) < a.ring.pub_target) __nanosleep(32);
            while (dm_ld_acquire(a.ring.pub_ctr + 1) == 0u) __nanosleep(32);
            dm_landed_wait(a, lw_need, lw_have);   // before the plan: no launch after this one reads a blob early
        }
        __syncthreads();
    } else {
        __syncthreads();
        pdl_wait();
        if (tid == 0) dm_landed_wait(a, lw_need, lw_have);
    }
    const int32_t e0 = tid < n ? __ldcg(a.ids + tid) : -1;
    const bool inr = e0 >= 0 && e0 < a.n_expert;
    const unsigned long long praw = inr ? a.ptab[e0] : 0ull;
    if (tid < n) s_ids[tid] = e0;
    if (inr && (atomicOr(&s_tab[e0], 1u << tok) >> tok & 1u) != 0u) s_dup = 1;
    __syncthreads();
    bool dup = false;
    if (s_dup != 0)
        for (int j = inr ? tok * a.k : tid; j < tid; ++j) dup |= s_ids[j] == e0;
    const bool valid = inr && praw != 0ull && !dup;
    const uint32_t bits = inr ? s_tab[e0] : 0u;
    const uint32_t low = bits & ((1u << tok) - 1u);
    const bool is_first = valid && low == 0u;
    const int gcnt = is_first ? __popc(bits) : 0;
    int cinc = gcnt;
    unsigned fb = 0u;
    if (warp < 4) {
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, cinc, d);
            if (lane >= d) cinc += v;
        }
        fb = __ballot_sync(0xffffffffu, is_first);
        if (lane == 31) s_wtot[warp] = make_int2(__popc(fb), cinc);
    }
    __syncthreads();
    int g0 = 0, e_off = 0, ng = 0, ne = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const int2 t = s_wtot[w];
        if (w < warp) { g0 += t.x; e_off += t.y; }
        ng += t.x;
        ne += t.y;
    }
    if (is_first) {
        const int gi = g0 + __popc(fb & ((1u << lane) - 1u)), gs = e_off + cinc - gcnt;
        s_ptr[gi] = praw;
        s_start[gi] = gs;
        int r = 0;
        for (uint32_t m = bits; m != 0u; m &= m - 1u) s_tok[gs + r++] = __ffs(m) - 1;
        s_tab[e0] = bits | (uint32_t) gs << 16;
    }
    if (tid == 0) s_start[ng] = ne;
    __syncthreads();
    if (lead) {
        DmPlan* P = a.plan;
        if (valid) {
            const int pos = (int) (s_tab[e0] >> 16) + __popc(low);
            P->dst[pos] = tid;
            P->tok[pos] = tok;
        }
        if (tid < ng) P->ptr[tid] = s_ptr[tid];
        if (tid <= ng) P->start[tid] = s_start[tid];
        if (tid == 0) {
            P->counts[0] = ng;
            P->counts[1] = ne;
        }
        __threadfence();
        __syncthreads();
        pdl_trigger();
    }
    const int nx = (int) (2 * L.n_ff + GU_ROWS - 1) / GU_ROWS;
    const int items = nx * ng;   // one (row block, group) item each
    const auto* xq = (const block_q8_1*) a.xq;
    // the first item is the block's own, the rest are taken off the launch's counter as blocks free up; the
    // last block to finish zeroes the counters for the next launch
    __shared__ int s_next[2];
    int it = wb;
    for (int i = 0; it < items; ++i) {
        if (tid == 0) s_next[i & 1] = nwork + (int) atomicAdd(&g_dm_glue1_work[0], 1u);
        const int bx = it % nx, g = it / nx;
        dm_gu_item<TG, false>((const uint8_t*) s_ptr[g], s_start[g], s_start[g + 1], s_tok, xq, L, bx, gate, up, nullptr,
                              warp, lane);
        __syncthreads();
        it = s_next[i & 1];
    }
    finish();
}

template<int TG> int dm_glue1_blocks() {
    static int b[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    if (b[dev] == 0) {
        int sms = 0, per = 0;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, dm_glue_gu1_kernel<TG>, 256, 0) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        b[dev] = std::max(1, (sms - sms / 8) * per);   // as dm_glue_blocks: the side streams keep an eighth
    }
    return b[dev];
}

// rows per down work item: native_down_multi_kernel's block rows at n_ff == 640
template<int TD> __host__ __device__ constexpr int dm_down_rows() { return kDown40<TD> ? 32 : TD == 42 ? 16 : 8; }

// The SwiGLU product and its q8_1 blocks for entries [e0, e1) into shared memory (swiglu_q8_1_entries_kernel's values).
__device__ __forceinline__ void dm_swiglu_q8(const float* __restrict__ gate, const float* __restrict__ up, int e0,
                                             int e1, int n_ff, block_q8_1* s_hq, int tid) {
    const int nv = (e1 - e0) * n_ff;
    const size_t base = (size_t) e0 * n_ff;
    for (int i = tid; i < nv; i += kDmThreads) {   // n_ff is a multiple of 32: whole warps
        const float g = __ldcg(gate + base + i);   // written by other blocks of this launch
        const float h = __fmul_rn(g / (1.0f + __expf(-g)), __ldcg(up + base + i));
        q8_1_store(h, s_hq, i);
    }
}

// native_down_multi_kernel's body (hb == 20) for block row `rb` of one group (`rows`: the address of its first row),
// from the group's q8_1 rows in s_hq
// (entry e at s_hq + (e - e0) * 20); row r of entry e lands in s_out[(e - e0) * ROWS + r - rb * R], R the item's
// rows (dm_down_rows) and ROWS the stride of s_out's entries.  `blob` is the expert's blob (expert_out_scale).
template<int TD>
__device__ __forceinline__ void dm_down_item(const uint8_t* rows, int e0, int e1, const block_q8_1* s_hq,
                                             const NativeExpertLayout& L, int rb, float* s_out, int tid, int ROWS,
                                             unsigned long long blob) {
    const int nb = (int) (L.n_ff / Fmt<TD>::qk);
    (void) nb;
    [[maybe_unused]] const float sc = expert_out_scale<TD>(blob, L, 2);
    if constexpr (kDown40<TD>) {   // K = 40 calls per row: 8 lanes a row
        const int rl = tid >> 3, r = rb * 32 + rl, t = tid & 7;
        if (r >= L.n_embd) return;
        const uint8_t* wr = rows + (size_t) rl * L.d_row;
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e), b = (e - e0) * 20;
            if (n == 1) {
                const int off1[1] = { b };
                float s1[1];
                row_dot_40_sub8<TD, 1, true>(wr, s_hq, off1, 1, t, s1);
                tail_scale<TD>(s1, sc);
                if (t == 0) s_out[(e - e0) * ROWS + rl] = s1[0];
            } else if (n == 2) {
                const int off2[2] = { b, b + 20 };
                float s2[2];
                row_dot_40_sub8<TD, 2, true>(wr, s_hq, off2, 2, t, s2);
                tail_scale<TD>(s2, sc);
                if (t < 2) s_out[(e - e0 + t) * ROWS + rl] = s2[t];
            } else {
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) off[c] = b + min(c, n - 1) * 20;
                float s[GRP_NC];
                row_dot_40_sub8<TD, GRP_NC>(wr, s_hq, off, n, t, s);
                tail_scale<TD>(s, sc);
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n && t == c) s_out[(e - e0 + c) * ROWS + rl] = s[c];
            }
        }
    } else if constexpr (TD == 42) {
        const int rl = tid >> 4, r = rb * 16 + rl, t = tid & 15;
        if (r >= L.n_embd) return;
        const uint8_t* wr = rows + (size_t) rl * L.d_row;
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e), b = (e - e0) * 20;
            if (n == 1) {
                const int off1[1] = { b };
                float s1[1];
                row_dot_20_sub16<TD, 1, true>(wr, s_hq, off1, 1, t, s1);
                tail_scale<TD>(s1, sc);
                if (t == 0) s_out[(e - e0) * ROWS + rl] = s1[0];
            } else if (n == 2) {
                const int off2[2] = { b, b + 20 };
                float s2[2];
                row_dot_20_sub16<TD, 2, true>(wr, s_hq, off2, 2, t, s2);
                tail_scale<TD>(s2, sc);
                if (t < 2) s_out[(e - e0 + t) * ROWS + rl] = s2[t];
            } else {
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) off[c] = b + min(c, n - 1) * 20;
                float s[GRP_NC];
                row_dot_20_sub16<TD, GRP_NC>(wr, s_hq, off, n, t, s);
                tail_scale<TD>(s, sc);
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n && t == c) s_out[(e - e0 + c) * ROWS + rl] = s[c];
            }
        }
    } else {
        const int warp = tid >> 5, lane = tid & 31, r = rb * 8 + warp;
        if (r >= L.n_embd) return;
        const uint8_t* wr = rows + (size_t) warp * L.d_row;
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e), b = (e - e0) * 20;
            if (n == 1) {
                const int off1[1] = { b };
                float s1[1];
                row_dot_multi<TD, 1, true>(wr, s_hq, off1, 1, nb, lane, s1);
                tail_scale<TD>(s1, sc);
                if (lane == 0) s_out[(e - e0) * ROWS + warp] = s1[0];
            } else if (n == 2) {
                const int off2[2] = { b, b + 20 };
                float s2[2];
                row_dot_multi<TD, 2, true>(wr, s_hq, off2, 2, nb, lane, s2);
                tail_scale<TD>(s2, sc);
                if (lane < 2) s_out[(e - e0 + lane) * ROWS + warp] = s2[lane];
            } else {
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) off[c] = b + min(c, n - 1) * 20;
                float s[GRP_NC];
                row_dot_multi<TD, GRP_NC>(wr, s_hq, off, n, nb, lane, s);
                tail_scale<TD>(s, sc);
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n && lane == c) s_out[(e - e0 + c) * ROWS + warp] = s[c];
            }
        }
    }
}

constexpr int kDmMaxEnt = kVerifyMaxT;   // entries of one group: at most one per token
constexpr int kDmSpan = 64;               // down rows per work item

struct DmShared {
    alignas(16) uint32_t grid[2048];                       // the format's codebook grid (stage_iq_grid)
    alignas(16) block_q8_1 hq[kDmMaxEnt * 20];             // one group's SwiGLU rows (n_ff == 640)
    alignas(16) float out[kDmMaxEnt * kDmSpan];            // one item's rows per entry
    int32_t ids[kDmCap], kind[kDmCap], first[kDmCap], cnt[kDmCap], gstart[kDmCap], gidx[kDmCap], pos[kDmCap];
    int32_t start[kDmCap + 1], tok[kDmCap], dst[kDmCap];
    unsigned long long gptr[kDmCap];
    int32_t ng, ne;
    int32_t stop;                                          // this block left its ring (the window aborted)
};

// The peer's waits inside a window: the abort word (device), the record (mapped), the host's doorbell (its release:
// UINT32_MAX) and what a record names.
struct DmWatch {
    unsigned int* abort = nullptr;
    uint32_t* diag_m = nullptr;
    const volatile uint32_t* seq_m = nullptr;
    uint32_t* done_m = nullptr;
    uint32_t* mdone_m = nullptr;                           // the mixer's done word, or null
    uint32_t ring = 0;
    int layer = 0, n_tok = 0, n = 0;
    const int32_t* hdr_ids = nullptr;                      // the device copy of the ring's ids (the record's checksum)
};

__device__ __forceinline__ unsigned long long dm_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// The first block to give up raises the abort word and writes the record: reason (1 a wait past kDmWaitLimitNs,
// 2 a group of more entries than tokens, 3 released by the host), ring, layer, tokens, block, phase (1 a down wait,
// 2 a grid barrier), group, count / wanted, the block's groups / entries, the group's entries, a checksum of the ids
// the block grouped and of the device header.  Thread 0 of a block calls it.
__device__ void dm_abort(const DmWatch& w, const DmShared* sm, uint32_t reason, uint32_t phase, int g, uint32_t have,
                         uint32_t want) {
    if (atomicCAS(w.abort, 0u, 1u) != 0u) return;
    __threadfence_system();
    *(volatile uint32_t*) w.done_m = 0xffffffffu;   // CUDA0's waits on this window pass
    if (w.mdone_m != nullptr) *(volatile uint32_t*) w.mdone_m = 0xffffffffu;
    __threadfence_system();
    if (w.diag_m == nullptr) return;
    uint32_t h0 = 0, h1 = 0;
    for (int i = 0; sm != nullptr && i < w.n; ++i) {
        h0 = h0 * 16777619u ^ (uint32_t) sm->ids[i] ^ (uint32_t) sm->kind[i] << 20;
        h1 = h1 * 16777619u ^ (uint32_t) __ldcg(w.hdr_ids + i) ^ (uint32_t) __ldcg(w.hdr_ids + kDmCap + i) << 20;
    }
    volatile uint32_t* d = w.diag_m;
    d[1] = w.ring; d[2] = (uint32_t) w.layer; d[3] = (uint32_t) w.n_tok; d[4] = blockIdx.x; d[5] = phase;
    d[6] = (uint32_t) g; d[7] = have; d[8] = want;
    d[9] = sm != nullptr ? (uint32_t) sm->ng : 0u; d[10] = sm != nullptr ? (uint32_t) sm->ne : 0u;
    d[11] = sm != nullptr && g >= 0 && g < sm->ng ? (uint32_t) (sm->start[g + 1] - sm->start[g]) : 0u;
    d[12] = h0; d[13] = h1;
    __threadfence_system();
    d[0] = reason;
    __threadfence_system();
}

// Thread 0's spin until *word >= want (acquire): false when the window was aborted or released meanwhile (the
// release and a wait past kDmWaitLimitNs, when `limit`, abort it with a record).  Only block 0 reads the host's word:
// many blocks reading host memory queue on the bus ahead of the doorbell and the ring's rows; the others follow the
// abort word.
__device__ __forceinline__ bool dm_wait_ge(const DmWatch& w, const DmShared* sm, const unsigned int* word,
                                           unsigned int want, uint32_t phase, int g, bool limit) {
    unsigned int v, spins = 0;
    unsigned long long t0 = 0;
    while ((v = *(const volatile unsigned int*) word) < want) {
        __nanosleep(20);
        if ((++spins & 255u) != 0u) continue;
        if (*(volatile unsigned int*) w.abort != 0u) return false;
        if (blockIdx.x == 0 && *w.seq_m == 0xffffffffu) { dm_abort(w, sm, 3u, phase, g, v, want); return false; }
        if (!limit) continue;
        const unsigned long long t = dm_now();
        if (t0 == 0) t0 = t;
        else if (t - t0 > kDmWaitLimitNs) { dm_abort(w, sm, 1u, phase, g, v, want); return false; }
    }
    __threadfence();
    return true;
}

// The grouped experts of sm's lists (ng groups: blob gptr[g], entries [start[g], start[g+1]) reading q8_1 row tok[e]
// and writing row dst[e] of `out`): block b of nb takes gate/up items (group, 8 rows) b, b + nb, ..., each one
// counted in done[g] when finished, then down items (group, kDmSpan rows) the same way, a group's first down item
// on a block waiting until done[g] holds all of its gate/up items.  A block never waits during its gate/up items, so
// with a co-resident grid every wait ends.  The rows are native_expert_grouped's, bit for bit.
template<int TG, int TD, bool Q2 = false, int TU = TG>
__device__ void dm_experts(DmShared& sm, const block_q8_1* __restrict__ xq, const NativeExpertLayout& L, float* gate,
                           float* up, float* out, unsigned int* done, int b, int nb, bool gu2 = false,
                           const DmWatch* watch = nullptr) {
    const int tid = threadIdx.x;
    if (tid == 0) sm.stop = 0;
    constexpr bool SG = kStageIqGrid<TG>;
    // gu2: gate/up items of GU2_ROWS rows, two a warp (native_gu2_kernel's rows, the same bits); Q2: Q2_0's too
    constexpr bool K2 = Q2 ? kGu2Q2<TG> : kGu2<TG>;
    const bool g2 = K2 && gu2 && gu2_usable(L);
    const uint32_t* s_grid;
    if constexpr (K2) s_grid = g2 ? gu2::stage_grid<TG>(sm.grid, tid) : stage_iq_grid<TG, SG>(sm.grid, tid, kDmThreads);
    else s_grid = stage_iq_grid<TG, SG>(sm.grid, tid, kDmThreads);
    const int warp = tid >> 5, lane = tid & 31;
    const int ng = sm.ng;
    const int ir = g2 ? GU2_ROWS : GU_ROWS;
    const int nrb = (int) ((2 * L.n_ff + ir - 1) / ir);
    for (int it = b; it < ng * nrb; it += nb) {
        const int g = it / nrb, rb = it - g * nrb;
        if constexpr (K2) {
            const int row = rb * GU2_ROWS + 2 * warp;
            if (g2) {
                if (row < 2 * L.n_ff)
                    gu2::pair_gu<TG, TU>((const uint8_t*) sm.gptr[g], row, sm.start[g], sm.start[g + 1], sm.tok, xq, L,
                                         gate, up, s_grid, lane);
            } else {
                dm_gu_item<TG, SG, TU>((const uint8_t*) sm.gptr[g], sm.start[g], sm.start[g + 1], sm.tok, xq, L, rb,
                                       gate, up, s_grid, warp, lane);
            }
        } else {
            dm_gu_item<TG, SG, TU>((const uint8_t*) sm.gptr[g], sm.start[g], sm.start[g + 1], sm.tok, xq, L, rb, gate,
                                   up, s_grid, warp, lane);
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) atomicAdd(done + g, 1u);
    }
    constexpr int R = dm_down_rows<TD>(), SUB = kDmSpan / R;
    const int nsp = (int) ((L.n_embd + kDmSpan - 1) / kDmSpan);
    int g_staged = -1;
    for (int it = b; it < ng * nsp; it += nb) {
        const int g = it / nsp, sp = it - g * nsp;
        const int e0 = sm.start[g], e1 = sm.start[g + 1];
        __syncthreads();                         // the previous item's reads of hq and out are done
        if (g != g_staged) {
            if (tid == 0) {
                if (watch == nullptr) {
                    while (*(volatile unsigned int*) (done + g) < (unsigned int) nrb) __nanosleep(20);
                    __threadfence();
                } else if (!dm_wait_ge(*watch, &sm, done + g, (unsigned int) nrb, 1u, g, true)) {
                    sm.stop = 1;   // the window is aborted: its rows are not needed
                }
            }
            __syncthreads();
            if (sm.stop) return;
            dm_swiglu_q8(gate, up, e0, e1, (int) L.n_ff, sm.hq, tid);
            g_staged = g;
            __syncthreads();
        }
        for (int sb = 0; sb < SUB; ++sb)
            dm_down_item<TD>((const uint8_t*) sm.gptr[g] + L.down_off + (size_t) ((sp * SUB + sb) * R) * L.d_row, e0, e1,
                             sm.hq, L, sp * SUB + sb, sm.out + sb * R, tid, kDmSpan, sm.gptr[g]);
        __syncthreads();
        constexpr int q = kDmSpan / 4;           // kDmSpan floats per entry, 16 bytes per thread
        const int nq = (e1 - e0) * q;
        for (int i = tid; i < nq; i += kDmThreads) {
            const int e = i / q, c = i - e * q;
            const int r0 = sp * kDmSpan + c * 4;
            if (r0 < L.n_embd)
                *(float4*) (out + (size_t) sm.dst[e0 + e] * L.n_embd + r0) = *(const float4*) &sm.out[e * kDmSpan + c * 4];
        }
    }
}

#define DM_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(40) X(12) X(13) X(14) X(8)
#define DM_D_FMTS(X) X(20) X(42) X(40) X(7) X(6) X(8)
#define DM_PAIRS(X) \
    X(16, 20) X(17, 20) X(18, 20) X(21, 20) X(22, 20) X(23, 20) X(29, 20) X(42, 20) \
    X(16, 42) X(17, 42) X(18, 42) X(21, 42) X(22, 42) X(23, 42) X(29, 42) X(42, 42) \
    X(16, 6) X(17, 6) X(18, 6) X(21, 6) X(22, 6) X(23, 6) X(29, 6) X(42, 6) \
    X(40, 40) X(12, 7) X(12, 8) X(13, 8) X(14, 8) X(8, 8) X(21, 8) X(23, 8)   // IQ3_S / IQ4_XS gate/up, Q8_0 down
// the mixed gate/up layers (GU_MIX) as (gate, up, down)
#define DM_MIX(X) X(22, 16, 42)

template<bool Q2 = false>
__device__ void dm_experts_any(DmShared& sm, const block_q8_1* xq, const NativeExpertLayout& L, float* gate, float* up,
                               float* out, unsigned int* done, int b, int nb, bool gu2 = false,
                               const DmWatch* watch = nullptr) {
    if (L.up_type != L.gu_type) {
        switch ((L.gu_type * 100 + L.up_type) * 100 + L.d_type) {
#define DM_MIX_CASE(TG, TU, TD) \
            case (TG * 100 + TU) * 100 + TD: dm_experts<TG, TD, Q2, TU>(sm, xq, L, gate, up, out, done, b, nb, gu2, watch); break;
            DM_MIX(DM_MIX_CASE)
#undef DM_MIX_CASE
            default: break;
        }
        return;
    }
    switch (L.gu_type * 100 + L.d_type) {
#define DM_CASE(TG, TD) \
        case TG * 100 + TD: dm_experts<TG, TD, Q2>(sm, xq, L, gate, up, out, done, b, nb, gu2, watch); break;
        DM_PAIRS(DM_CASE)
#undef DM_CASE
        default: break;
    }
}

// The last block of a launch (or of a peer ring) to arrive: zero the group counters for the next one.
__device__ __forceinline__ bool dm_arrive(unsigned int* arrive, unsigned int target, unsigned int* done, int ng) {
    __syncthreads();
    bool last = false;
    if (threadIdx.x == 0) {
        __threadfence();
        last = atomicAdd(arrive, 1u) + 1u == target;
        if (last) {
            for (int g = 0; g < ng; ++g) done[g] = 0u;
            __threadfence();
        }
    }
    return last;
}

// The peer's grid barrier (the k-th waits for k * nb arrivals on `bar`): false when the window was aborted or released
// meanwhile (block 0 watches the host's word, as in dm_wait_ge).  A block that gets false leaves the kernel; the
// others follow at their next wait.
__device__ __forceinline__ bool dm_grid_sync(const DmWatch& w, unsigned int* bar, unsigned int target, int32_t* s_ok) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(bar, 1u);
        bool ok = true;
        unsigned int v, spins = 0;
        while (true) {
            asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(bar) : "memory");
            if (v >= target) break;
            spin_pause();
            if ((++spins & 255u) != 0u) continue;
            if (*(volatile unsigned int*) w.abort != 0u) { ok = false; break; }
            if (blockIdx.x == 0 && *w.seq_m == 0xffffffffu) {
                dm_abort(w, nullptr, 3u, 2u, -1, v, target);
                ok = false;
                break;
            }
        }
        *s_ok = ok;
    }
    __syncthreads();
    return *s_ok != 0;
}

// The peer's warm-up words (DmPeerArgs::arrive, zeroed per launch): the ring whose doorbell block 0 has seen, the
// predictor's arrivals, the ring whose expert list is ready, its length, a sink for the touch loads.
constexpr int kDmWarmGo = 3, kDmWarmArr = 4, kDmWarmRing = 5, kDmWarmN = 6, kDmWarmSink = 7;
constexpr int kDmWarmTok = 5;   // tokens of a ring the predictor reads

__device__ __forceinline__ unsigned int dm_ld(const unsigned int* p) { return *(const volatile unsigned int*) p; }

// Between ring `next - 1` (rows xq of n_tok tokens, layer `layer`) and ring `next` (layer + 1), on blocks 1..nb-1
// while block 0 waits for the next doorbell, once every block has left the ring: the router of layer + 1 on these
// rows (int8 rows, a prediction only), every token's top k, the predicted experts the peer holds rank by rank (logits,
// list: the scratch past the counters), then one plain-load pass over those blobs so the next ring's expert reads find
// them in the L2.  Each part is skipped or cut short once the doorbell is in (kDmWarmGo).  Writes nothing the rings
// read.  Not inlined: the rings' expert loop keeps its own register allocation.  `a` is taken by value: a reference to
// the kernel parameter makes ptxas build a local copy that the non-inlined callee reads through an unset base.
__device__ __noinline__ void dm_warm(DmPeerArgs a, DmShared& sm, const block_q8_1* xq, int n_tok, int layer,
                                     uint32_t next, float* logits, unsigned long long* list, unsigned int ring_arrivals) {
    __shared__ int32_t s_last, s_go;   // block-uniform decisions (go moves while the block reads it)
    __shared__ unsigned int s_n;
    unsigned int* go = a.arrive + kDmWarmGo;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    constexpr int WB = kDmThreads / 32;
    const int nw = (int) (gridDim.x - 1) * WB, gw = (int) (blockIdx.x - 1) * WB + warp;
    const int ne = a.n_expert, nblk = a.n_embd / 32, l1 = layer + 1;
    const bool on = n_tok > 0 && ne <= kDmPredExperts && a.k <= 32 && n_tok * a.k <= kDmCap;
    if (tid == 0) {
        if (on && ring_arrivals != 0u)   // the router pass waits until every block has left the ring
            while (dm_ld(a.arrive) < ring_arrivals && dm_ld(go) < next && dm_ld(a.arrive + 2) == 0u) __nanosleep(100);
        s_go = dm_ld(go) >= next;
    }
    __syncthreads();
    // the predictor reads at most kDmWarmTok tokens' rows, staged in the block's shared memory (int8 values and the
    // block scales) over the rings' grid, SwiGLU and output buffers
    constexpr int kSx = (int) (sizeof(sm.grid) + sizeof(sm.hq) + sizeof(sm.out));
    const int pt = min(min(n_tok, kDmWarmTok), kSx / (a.n_embd + nblk * 4));
    if (on && (int) (blockIdx.x - 1) * WB < ne && !s_go) {
        int8_t* sx = (int8_t*) sm.grid;
        float* sd = (float*) (sx + (size_t) pt * a.n_embd);
        for (int i = tid; i < pt * nblk; i += kDmThreads) {
            const int* xb = (const int*) (xq + i);   // tokens' rows back to back, nblk blocks each
            sd[i] = __half2float(__ushort_as_half((unsigned short) (__ldcg(xb) & 0xffff)));
            int* d = (int*) (sx + (size_t) i * 32);
#pragma unroll
            for (int j = 0; j < 8; ++j) d[j] = __ldcg(xb + 1 + j);
        }
        __syncthreads();
        const int8_t* rq = a.rq + (size_t) l1 * ne * a.n_embd;
        const float* rs = a.rs + (size_t) l1 * ne * nblk;
        const int nc = a.n_embd / 16;   // 16-value chunks a row
        for (int e = gw; e < ne; e += nw) {
            float acc[kDmWarmTok];
#pragma unroll
            for (int t = 0; t < kDmWarmTok; ++t) acc[t] = 0.0f;
            const int4* wr = (const int4*) (rq + (size_t) e * a.n_embd);
            for (int c = lane; c < nc; c += 32) {
                const int4 w = __ldcs(wr + c);
                const float sw = __ldcs(rs + (size_t) e * nblk + c / 2);
#pragma unroll
                for (int t = 0; t < kDmWarmTok; ++t) {
                    if (t < pt) {
                        const int4 xv = ((const int4*) (sx + (size_t) t * a.n_embd))[c];
                        int si = __dp4a(w.x, xv.x, 0);
                        si = __dp4a(w.y, xv.y, si); si = __dp4a(w.z, xv.z, si); si = __dp4a(w.w, xv.w, si);
                        acc[t] += sw * sd[t * nblk + c / 2] * (float) si;
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < kDmWarmTok; ++t) {
                if (t < pt) {
                    const float v = warp_sum(acc[t]);
                    if (lane == 0) logits[t * kDmPredExperts + e] = v;
                }
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        s_last = atomicAdd(a.arrive + kDmWarmArr, 1u) + 1u == (next - 1u) * (gridDim.x - 1);   // once a ring
        __threadfence();
        s_go = dm_ld(go) >= next;
    }
    __syncthreads();
    if (s_last && on && !s_go) {   // the last to arrive: the list
        int32_t* first = (int32_t*) sm.grid;   // restaged by the next ring's experts
        for (int i = tid; i < kDmPredExperts; i += kDmThreads) first[i] = INT32_MAX;
        const int k = a.k, n = pt * k;
        if (warp < pt) {
            float v[kDmPredExperts / 32];
#pragma unroll
            for (int i = 0; i < kDmPredExperts / 32; ++i) {
                const int e = lane + i * 32;
                v[i] = e < ne ? __ldcg(logits + warp * kDmPredExperts + e) : -INFINITY;
            }
            for (int r = 0; r < k; ++r) {
                float best = v[0];
                int ex = lane;
#pragma unroll
                for (int i = 1; i < kDmPredExperts / 32; ++i)
                    if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
#pragma unroll
                for (int m = 16; m; m >>= 1) {
                    const float ob = __shfl_xor_sync(0xffffffffu, best, m);
                    const int oe = __shfl_xor_sync(0xffffffffu, ex, m);
                    if (ob > best || (ob == best && oe < ex)) { best = ob; ex = oe; }
                }
#pragma unroll
                for (int i = 0; i < kDmPredExperts / 32; ++i)
                    if (ex == lane + i * 32) v[i] = -INFINITY;
                if (lane == 0) sm.ids[r * pt + warp] = ex;   // rank by rank across the tokens
            }
        }
        __syncthreads();
        if (tid < n) atomicMin(first + sm.ids[tid], tid);
        __syncthreads();
        if (tid < n) {
            const int e = sm.ids[tid];
            const unsigned long long p = first[e] == tid ? a.ptr[(size_t) l1 * ne + e] : 0ull;
            sm.gptr[tid] = p;
        }
        __syncthreads();
        if (warp == 0) {
            unsigned int base = 0;
            for (int c = 0; c < n; c += 32) {
                const int i = c + lane;
                const unsigned long long p = i < n ? sm.gptr[i] : 0ull;
                const unsigned int bal = __ballot_sync(0xffffffffu, p != 0ull);
                if (p != 0ull) list[base + __popc(bal & ((1u << lane) - 1u))] = p;
                base += __popc(bal);
            }
            __syncwarp();   // the lanes' list entries before the ring word
            if (lane == 0) {
                a.arrive[kDmWarmN] = base;
                __threadfence();
                atomicMax(a.arrive + kDmWarmRing, next);
            }
        }
    }
    // the touch: wait for the list or the doorbell
    if (tid == 0) {
        unsigned int spins = 0;
        while (dm_ld(a.arrive + kDmWarmRing) < next && dm_ld(go) < next) {
            __nanosleep(64);
            if ((++spins & 63u) == 0u && dm_ld(a.arrive + 2) != 0u) break;
        }
        __threadfence();
        s_n = dm_ld(go) < next && dm_ld(a.arrive + kDmWarmRing) >= next ? dm_ld(a.arrive + kDmWarmN) : 0u;
    }
    __syncthreads();
    const unsigned int nh = s_n;
    if (nh == 0u) return;
    const unsigned int per = (unsigned int) (a.lay[l1].bytes / 16);   // 16-byte units a blob
    const unsigned int total = nh * per;
    unsigned int x = 0;
    for (unsigned int u = (unsigned int) gw * 32u + (unsigned int) lane; u < total; u += (unsigned int) nw * 32u) {
        const unsigned int j = u / per;
        const uint4 q = __ldcg((const uint4*) __ldcg(list + j) + (u - j * per));
        x ^= q.x ^ q.w;
        if (dm_ld(go) >= next) break;
    }
    if (x == 0x9e3779b9u) a.arrive[kDmWarmSink] = x;
}

// The mixer item's projections: rows [0, rows) of w (rb bytes each) against the item's nt q8_1 rows (xs blocks
// apart), one warp a row; token t's sum to out[t * ld + r].
template<int TY>
__device__ __forceinline__ void dm_mix_rows_t(const uint8_t* w, size_t rb, int rows, int n_in, const block_q8_1* x, int xs,
                                              int nt, float* out, int ld, int gw, int nw, int lane) {
    const int nbk = n_in / Fmt<TY>::qk;
    int off[kDecMaxT];
#pragma unroll
    for (int c = 0; c < kDecMaxT; ++c) off[c] = min(c, nt - 1) * xs;
    for (int r = gw; r < rows; r += nw) {
        float s[kDecMaxT];
        row_dot_multi<TY, kDecMaxT>(w + (size_t) r * rb, x, off, nt, nbk, lane, s);
#pragma unroll
        for (int c = 0; c < kDecMaxT; ++c)
            if (c < nt && lane == c) out[(size_t) c * ld + r] = s[c];
    }
}

__device__ __forceinline__ void dm_mix_rows(int type, const void* w, int rows, int n_in, const block_q8_1* x, int nt,
                                            float* out, int ld, int gw, int nw, int lane) {
    const int xs = n_in / 32;
    switch (type) {
        case 12: dm_mix_rows_t<12>((const uint8_t*) w, (size_t) (n_in / 256) * sizeof(block_q4_K), rows, n_in, x, xs, nt, out, ld, gw, nw, lane); break;
        case 13: dm_mix_rows_t<13>((const uint8_t*) w, (size_t) (n_in / 256) * sizeof(block_q5_K), rows, n_in, x, xs, nt, out, ld, gw, nw, lane); break;
        case 14: dm_mix_rows_t<14>((const uint8_t*) w, (size_t) (n_in / 256) * sizeof(block_q6_K), rows, n_in, x, xs, nt, out, ld, gw, nw, lane); break;
        case 23: dm_mix_rows_t<23>((const uint8_t*) w, (size_t) (n_in / 256) * sizeof(block_iq4_xs), rows, n_in, x, xs, nt, out, ld, gw, nw, lane); break;
        default: break;
    }
}

// dm_mix_rows_t for an output in mapped memory: the block's warps take kDmThreads / 32 consecutive rows, their sums
// meet in shared memory and leave as one contiguous run per token, so each store is a whole segment, not 4 bytes.
template<int TY>
__device__ __forceinline__ void dm_mix_rows_m_t(const uint8_t* w, size_t rb, int rows, int n_in, const block_q8_1* x,
                                                int xs, int nt, float* out, int ld, float (*sp)[kDmThreads / 32]) {
    constexpr int WB = kDmThreads / 32;
    const int nbk = n_in / Fmt<TY>::qk, lane = threadIdx.x & 31, wi = threadIdx.x / 32;
    int off[kDecMaxT];
#pragma unroll
    for (int c = 0; c < kDecMaxT; ++c) off[c] = min(c, nt - 1) * xs;
    for (int r0 = blockIdx.x * WB; r0 < rows; r0 += gridDim.x * WB) {
        const int r = r0 + wi;
        if (r < rows) {
            float s[kDecMaxT];
            row_dot_multi<TY, kDecMaxT>(w + (size_t) r * rb, x, off, nt, nbk, lane, s);
#pragma unroll
            for (int c = 0; c < kDecMaxT; ++c)
                if (c < nt && lane == c) sp[c][wi] = s[c];
        }
        __syncthreads();
        const int c = threadIdx.x / WB, j = threadIdx.x % WB;
        if (c < nt && r0 + j < rows) out[(size_t) c * ld + r0 + j] = sp[c][j];
        __syncthreads();
    }
}

__device__ __forceinline__ void dm_mix_rows_m(int type, const void* w, int rows, int n_in, const block_q8_1* x, int nt,
                                              float* out, int ld, float (*sp)[kDmThreads / 32]) {
    const int xs = n_in / 32;
    switch (type) {
        case 12: dm_mix_rows_m_t<12>((const uint8_t*) w, (size_t) (n_in / 256) * sizeof(block_q4_K), rows, n_in, x, xs, nt, out, ld, sp); break;
        case 13: dm_mix_rows_m_t<13>((const uint8_t*) w, (size_t) (n_in / 256) * sizeof(block_q5_K), rows, n_in, x, xs, nt, out, ld, sp); break;
        case 14: dm_mix_rows_m_t<14>((const uint8_t*) w, (size_t) (n_in / 256) * sizeof(block_q6_K), rows, n_in, x, xs, nt, out, ld, sp); break;
        case 23: dm_mix_rows_m_t<23>((const uint8_t*) w, (size_t) (n_in / 256) * sizeof(block_iq4_xs), rows, n_in, x, xs, nt, out, ld, sp); break;
        default: break;
    }
}

// CUDA0's VRAM share in one launch over the device plan (a co-resident grid; `sync`: [0] arrivals, [1..] per group).
template<int TG, int TD>
__global__ void __launch_bounds__(kDmThreads) dm_vram_kernel(const DmPlan* __restrict__ plan,
                                                             const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                             float* gate, float* up, float* out, unsigned int* sync) {
    __shared__ DmShared sm;
    const int tid = threadIdx.x;
    const int ng = plan->counts[0];
    for (int i = tid; i <= ng; i += kDmThreads) {
        sm.start[i] = plan->start[i];
        if (i < ng) sm.gptr[i] = plan->ptr[i];
    }
    for (int i = tid; i < kDmCap; i += kDmThreads) {
        sm.tok[i] = plan->tok[i];
        sm.dst[i] = plan->dst[i];
    }
    if (tid == 0) sm.ng = ng;
    __syncthreads();
    dm_experts<TG, TD>(sm, xq, L, gate, up, out, sync + 1, blockIdx.x, gridDim.x);
    if (dm_arrive(sync, gridDim.x, sync + 1, ng)) sync[0] = 0u;
}

// ---------------------------------------------------------------- the VRAM share through the copy engine (TMA)
// Each block walks its work items (b, b + nb, ...: gate/up items of 8 rows, then down items of 32 rows) with every
// item's weight rows brought into shared memory by a bulk copy issued kTmaStages - 1 items ahead, so a block keeps
// several items in flight whatever its warps are doing; the rows are then read from shared memory by the same row
// dots (the same bits).  A down item waits for its group's gate/up items (counters, as dm_experts).
constexpr int kTmaStages = 3;
constexpr int kTmaStage = 11776;          // bytes: the largest item (32 IQ4_NL down rows, 8 IQ4_XS gate/up rows)
// The TMA bulk-copy pipeline (mbarrier + cp.async.bulk + the cluster fence) needs sm_90: a build for an
// older peer card compiles none of it, and the host side must not choose kind 1 there (the plain kernel
// path below covers the same formats).
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
constexpr int kTmaGuRows = 8, kTmaDownRows = 32;

struct DmTmaShared {
    alignas(128) uint8_t stage[kTmaStages][kTmaStage];
    alignas(8) unsigned long long bar[kTmaStages];
    alignas(16) block_q8_1 hq[kDmMaxEnt * 20];
    alignas(16) float out[kDmMaxEnt * kTmaDownRows];
    int32_t start[kDmCap + 1], tok[kDmCap], dst[kDmCap];
    unsigned long long gptr[kDmCap];
};

__device__ __forceinline__ uint32_t dm_smem(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void dm_bulk(void* dst, const void* src, uint32_t bytes, unsigned long long* bar) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(dm_smem(bar)), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dm_smem(dst)), "l"(src), "r"(bytes), "r"(dm_smem(bar)) : "memory");
}
__device__ __forceinline__ void dm_bar_wait(unsigned long long* bar, uint32_t parity) {
    uint32_t ok = 0;
    while (!ok)
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                     : "=r"(ok) : "r"(dm_smem(bar)), "r"(parity) : "memory");
}

// sm's groups (ng, start, gptr, tok, dst set) through the bulk-copy pipeline: block b of nb takes items b, b + nb, ...
// (gate/up items of 8 rows, then down items of 32 rows); `base` counts the block's earlier items on these barriers
// (their phases).  Returns the items it took.
template<int TG, int TD>
__device__ int dm_experts_tma(DmTmaShared& sm, int ng, const block_q8_1* __restrict__ xq, const NativeExpertLayout& L,
                              float* gate, float* up, float* out, unsigned int* done, int b, int nb, int base) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nga = (int) (2 * L.n_ff / kTmaGuRows), ndn = (int) (L.n_embd / kTmaDownRows);
    const int na = ng * nga, total = na + ng * ndn;
    auto issue = [&](int k) {
        const int it = b + k * nb;
        if (it >= total) return;
        const uint8_t* src;
        uint32_t bytes;
        if (it < na) {
            const int g = it / nga, c = it - g * nga;
            bytes = (uint32_t) (kTmaGuRows * L.gu_row);
            src = (const uint8_t*) sm.gptr[g] + (size_t) c * bytes;
        } else {
            const int j = it - na, g = j / ndn, c = j - g * ndn;
            bytes = (uint32_t) (kTmaDownRows * L.d_row);
            src = (const uint8_t*) sm.gptr[g] + L.down_off + (size_t) c * bytes;
        }
        const int st = (base + k) % kTmaStages;
        dm_bulk(sm.stage[st], src, bytes, &sm.bar[st]);
    };
    __syncthreads();                             // the stages' earlier contents are consumed
    if (tid == 0)
        for (int k = 0; k < kTmaStages - 1; ++k) issue(k);
    int g_staged = -1, k = 0;
    for (; b + k * nb < total; ++k) {
        const int it = b + k * nb;
        if (k > 0) __syncthreads();              // stage (k - 1) is consumed: refill it
        if (tid == 0) {
            // the warps read the stage with generic loads; the refill is an async-proxy write
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            issue(k + kTmaStages - 1);
        }
        const int st = (base + k) % kTmaStages;
        const uint32_t parity = (uint32_t) (((base + k) / kTmaStages) & 1);
        const uint8_t* rows = sm.stage[st];
        if (it < na) {
            const int g = it / nga, c = it - g * nga;
            dm_bar_wait(&sm.bar[st], parity);
            dm_gu_row<TG, false>(rows + (size_t) warp * L.gu_row, c * kTmaGuRows + warp, sm.start[g], sm.start[g + 1],
                                 sm.tok, xq, L, gate, up, nullptr, lane, sm.gptr[g]);
            __threadfence();
            __syncthreads();
            if (tid == 0) atomicAdd(done + g, 1u);
        } else {
            const int j = it - na, g = j / ndn, c = j - g * ndn;
            const int e0 = sm.start[g], e1 = sm.start[g + 1];
            if (g != g_staged) {
                if (tid == 0) {
                    while (*(volatile unsigned int*) (done + g) < (unsigned int) nga) __nanosleep(20);
                    __threadfence();
                }
                __syncthreads();
                dm_swiglu_q8(gate, up, e0, e1, (int) L.n_ff, sm.hq, tid);
                g_staged = g;
                __syncthreads();
            }
            dm_bar_wait(&sm.bar[st], parity);
            constexpr int R = dm_down_rows<TD>();
#pragma unroll
            for (int sb = 0; sb < kTmaDownRows / R; ++sb)
                dm_down_item<TD>(rows + (size_t) (sb * R) * L.d_row, e0, e1, sm.hq, L, c * (kTmaDownRows / R) + sb,
                                 sm.out + sb * R, tid, kTmaDownRows, sm.gptr[g]);
            __syncthreads();
            constexpr int q = kTmaDownRows / 4;
            const int nq = (e1 - e0) * q;
            for (int i = tid; i < nq; i += kDmThreads) {
                const int e = i / q, cc = i - e * q;
                *(float4*) (out + (size_t) sm.dst[e0 + e] * L.n_embd + c * kTmaDownRows + cc * 4) =
                    *(const float4*) &sm.out[e * kTmaDownRows + cc * 4];
            }
        }
    }
    return k;
}

__device__ __forceinline__ void dm_bars_init(DmTmaShared& sm) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < kTmaStages; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(dm_smem(&sm.bar[i])) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
}

template<int TG, int TD>
__global__ void __launch_bounds__(kDmThreads) dm_vram_tma_kernel(const DmPlan* __restrict__ plan,
                                                                 const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                                 float* gate, float* up, float* out, unsigned int* sync) {
    __shared__ DmTmaShared sm;
    const int tid = threadIdx.x;
    const int ng = plan->counts[0];
    for (int i = tid; i <= ng; i += kDmThreads) {
        sm.start[i] = plan->start[i];
        if (i < ng) sm.gptr[i] = plan->ptr[i];
    }
    for (int i = tid; i < kDmCap; i += kDmThreads) {
        sm.tok[i] = plan->tok[i];
        sm.dst[i] = plan->dst[i];
    }
    dm_bars_init(sm);
    dm_experts_tma<TG, TD>(sm, ng, xq, L, gate, up, out, sync + 1, blockIdx.x, gridDim.x, 0);
    if (dm_arrive(sync, gridDim.x, sync + 1, ng)) sync[0] = 0u;
}
#endif  // sm_90+ (the TMA pipeline)

// ---------------------------------------------------------------- the VRAM share as three launches, NR rows a warp
// row_dot_multi over NR rows at once: each row's sums are row_dot_multi's (the same calls, k lane-strided, added in the
// same order, the same warp_sum); the rows only share the loop, so a warp has NR rows' loads in flight.
template<int TY, int NC, int NR, bool EXACT_N, bool STAGE_GRID>
__device__ __forceinline__ void rows_dot_multi(const uint8_t* const (&rows)[NR], const block_q8_1* x, const int (&off)[NC],
                                               int n, int nb, int lane, float (&s)[NR][NC],
                                               const uint32_t* __restrict__ s_grid) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) s[r][c] = 0.0f;
    const int nk = nb * F::ipb;
    if (nk <= 3 * 32) {   // every weight part of the lane loaded first (up to 3 per row), then the same adds in k order
        typename S::W w[3][NR];
#pragma unroll
        for (int it = 0; it < 3; ++it) {
            const int k = lane + 32 * it;
            if (k < nk) {
                const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
#pragma unroll
                for (int r = 0; r < NR; ++r) w[it][r] = S::template load<STAGE_GRID>(rows[r], kbx, iqs, s_grid);
            }
        }
#pragma unroll
        for (int it = 0; it < 3; ++it) {
            const int k = lane + 32 * it;
            if (k < nk) {
                const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
#pragma unroll
                for (int r = 0; r < NR; ++r)
#pragma unroll
                    for (int c = 0; c < NC; ++c)
                        if (EXACT_N || c < n) s[r][c] += S::apply(w[it][r], x + off[c] + kbx * (F::qk / 32), iqs);
            }
        }
    } else
    for (int k = lane; k < nk; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        typename S::W w[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) w[r] = S::template load<STAGE_GRID>(rows[r], kbx, iqs, s_grid);
#pragma unroll
        for (int r = 0; r < NR; ++r)
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (EXACT_N || c < n) s[r][c] += S::apply(w[r], x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[r][c] = warp_sum(s[r][c]);
}

// native_gu_multi_kernel with NR rows a warp (rows row0 + 8 i): block x covers 8 NR rows, block row y a group
template<int TG, int NR, bool STAGE_GRID = kStageIqGrid<TG>>
__global__ void __launch_bounds__(256) dm_gu_kernel(const DmPlan* __restrict__ plan, const block_q8_1* __restrict__ xq,
                                                    NativeExpertLayout L, float* __restrict__ gate, float* __restrict__ up) {
    const int ng = plan->counts[0];
    if ((int) blockIdx.y >= ng) return;
    __shared__ alignas(16) uint32_t s_grid_buf[IqGridWords<TG, STAGE_GRID>::value];
    const uint32_t* s_grid = stage_iq_grid<TG, STAGE_GRID>(s_grid_buf, threadIdx.x, 256);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * (GU_ROWS * NR) + warp;
    const int nrow = 2 * (int) L.n_ff;
    if (row0 >= nrow) return;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int g = blockIdx.y;
    const uint8_t* blob = (const uint8_t*) plan->ptr[g];
    const uint8_t* rows[NR];
    int rr[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) {
        rr[i] = row0 + GU_ROWS * i < nrow ? row0 + GU_ROWS * i : row0;
        rows[i] = blob + (size_t) rr[i] * L.gu_row;
    }
    [[maybe_unused]] float sc[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) sc[i] = expert_out_scale<TG>(plan->ptr[g], L, rr[i] >= L.n_ff ? 1 : 0);
    auto put = [&](int i, int e, float v) {
        if (row0 + GU_ROWS * i >= nrow) return;
        if constexpr (TG == 40) v *= sc[i];
        const int row = rr[i];
        if (row >= L.n_ff) up[(size_t) e * L.n_ff + row - (int) L.n_ff] = v;
        else gate[(size_t) e * L.n_ff + row] = v;
    };
    const int32_t* ent_tok = plan->tok;
    const int e0 = plan->start[g], e1 = plan->start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        if (n == 1) {
            const int off1[1] = { ent_tok[e] * xb };
            float s1[NR][1];
            rows_dot_multi<TG, 1, NR, true, STAGE_GRID>(rows, xq, off1, 1, nb, lane, s1, s_grid);
            if (lane == 0)
#pragma unroll
                for (int i = 0; i < NR; ++i) put(i, e, s1[i][0]);
        } else if (n == 2) {
            const int off2[2] = { ent_tok[e] * xb, ent_tok[e + 1] * xb };
            float s2[NR][2];
            rows_dot_multi<TG, 2, NR, true, STAGE_GRID>(rows, xq, off2, 2, nb, lane, s2, s_grid);
            if (lane < 2)
#pragma unroll
                for (int i = 0; i < NR; ++i) put(i, e + lane, s2[i][lane]);
        } else {
            int off[GRP_NC];
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = ent_tok[e + min(c, n - 1)] * xb;
            float sm[NR][GRP_NC];
            rows_dot_multi<TG, GRP_NC, NR, false, STAGE_GRID>(rows, xq, off, n, nb, lane, sm, s_grid);
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c)
#pragma unroll
                    for (int i = 0; i < NR; ++i) put(i, e + c, sm[i][c]);
        }
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900  // the TMA kernel exists on this build
template<int TG, int TD>
int dm_vram_tma_blocks() {
    static int known[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    if (known[dev] == 0) {
        int sms = 0, per = 0;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, dm_vram_tma_kernel<TG, TD>, kDmThreads, 0) != cudaSuccess) {
            cudaGetLastError();
            per = 0;
        }
        known[dev] = per > 0 ? sms * per : -1;
    }
    return known[dev];
}
#endif  // the TMA kernel exists on this build

template<int TG, int TD>
int dm_vram_blocks() {
    static int known[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    if (known[dev] == 0) {
        int sms = 0, per = 0;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, dm_vram_kernel<TG, TD>, kDmThreads, 0) != cudaSuccess) {
            cudaGetLastError();
            per = 0;
        }
        known[dev] = per > 0 ? sms * per : -1;
    }
    return known[dev];
}

}  // namespace
}  // namespace basalt::kernels
