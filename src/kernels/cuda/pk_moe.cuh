// src/kernels/cuda/pk_moe.cuh - see include/basalt/kernels/pk_moe.hpp.  Included at the end of iq_kernels.cu, after
// dec_moe.cuh: it runs that file's gate/up items (gu2::pair), SwiGLU (dm_swiglu_q8) and down items (dm_down_item).
#include "basalt/kernels/pk_moe.hpp"
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int kPkSpan = 64;                         // output columns per down item
constexpr int kPkSyncArrive = 0, kPkSyncShared = 1, kPkSyncGroups = 2, kPkSyncSpans = 2 + kDmCap;
static_assert(kPkSyncSpans + 64 <= kPkSyncWords, "sync words");

struct PkShared {
    alignas(16) uint32_t grid[2048];
    alignas(16) block_q8_1 hq[kDmMaxEnt * 20];     // one group's SwiGLU rows (n_ff == 640)
    alignas(16) float out[kDmMaxEnt * kPkSpan];    // one down item's rows per entry
    alignas(16) int32_t key[kDmCap];
    int32_t ids[kDmCap], kind[kDmCap], first[kDmCap], cnt[kDmCap], gstart[kDmCap], gidx[kDmCap], pos[kDmCap];
    int32_t start[kDmCap + 1], tok[kDmCap], dst[kDmCap];
    unsigned long long p[kDmCap], gptr[kDmCap];
    int32_t ng, ne, last;
};

// optional stage stamps (PkMoeArgs::trace): device nanosecond clock folded into min/max/sum slots
__device__ __forceinline__ unsigned long long pk_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ void pk_tmin(unsigned long long* t, int slot) { atomicMin(t + slot, pk_now()); }
__device__ __forceinline__ void pk_tmax(unsigned long long* t, int slot) { atomicMax(t + slot, pk_now()); }

__device__ __forceinline__ unsigned int pk_arrive(unsigned int* p) {
    unsigned int v;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// every block builds the window's grouping from the ids (dm_glue_gu_kernel's plan for the VRAM tier)
__device__ __forceinline__ void pk_plan(PkShared& sm, const PkMoeArgs& a, int tid) {
    const int n = a.n_tok * a.k;
    const int32_t e0 = tid < n ? __ldcg(a.ids + tid) : -1;
    const unsigned long long praw = (e0 >= 0 && e0 < a.n_expert) ? a.ptab[e0] : 0ull;
    if (tid < n) sm.ids[tid] = e0;
    __syncthreads();
    int32_t key = -1;
    if (tid < n) {
        bool ok = e0 >= 0 && e0 < a.n_expert;
        for (int j = tid - tid % a.k; ok && j < tid; ++j) ok = sm.ids[j] != e0;
        const unsigned long long p = ok ? praw : 0ull;
        sm.p[tid] = p;
        sm.kind[tid] = p != 0ull ? 0 : -1;
        key = p != 0ull ? e0 : -1;
    }
    if (tid < kDmCap) sm.key[tid] = key;
    __syncthreads();
    int first = 0, rank = 0;
    dm_group_keys(sm.key, n, tid, sm.first, sm.cnt, sm.gstart, sm.gidx, sm.start, &sm.ng, &sm.ne, first, rank);
    if (tid < n && sm.kind[tid] == 0) {
        const int pos = sm.gstart[first] + rank;
        sm.pos[tid] = pos;
        sm.tok[pos] = tid / a.k;
        sm.dst[pos] = tid;
        if (sm.first[tid]) sm.gptr[sm.gidx[tid]] = sm.p[tid];
    }
    __syncthreads();
}

// the weighted sum of one 64-column span (dm_combine4_kernel's arithmetic: rows added to +0.0, the products
// accumulated in expert order, the shared row last)
__device__ __forceinline__ void pk_combine_span(const PkMoeArgs& a, const NativeExpertLayout& L, int sp, int tid) {
    const int i = tid;
    if (i >= a.n_tok * (kPkSpan / 4)) return;
    const int tk = i / (kPkSpan / 4), col = sp * kPkSpan + (i % (kPkSpan / 4)) * 4;
    const int64_t N = L.n_embd;
    float4 v[10];
    float w[10];
#pragma unroll
    for (int j = 0; j < 10; ++j) {
        if (j >= a.k) break;
        w[j] = __ldcg(a.weights + tk * a.k + j);
        const float4 p = __ldcg(reinterpret_cast<const float4*>(a.parts + (tk * a.k + j) * N + col));
        v[j] = make_float4(__fadd_rn(0.0f, p.x), __fadd_rn(0.0f, p.y), __fadd_rn(0.0f, p.z), __fadd_rn(0.0f, p.w));
    }
    float4 h = a.shared ? __ldcg(reinterpret_cast<const float4*>(a.shared + tk * N + col)) : make_float4(0.f, 0.f, 0.f, 0.f);
    float sx = v[0].x * w[0], sy = v[0].y * w[0], sz = v[0].z * w[0], sw = v[0].w * w[0];
#pragma unroll
    for (int j = 1; j < 10; ++j) {
        if (j >= a.k) break;
        sx += v[j].x * w[j];
        sy += v[j].y * w[j];
        sz += v[j].z * w[j];
        sw += v[j].w * w[j];
    }
    if (a.shared) { sx += h.x; sy += h.y; sz += h.z; sw += h.w; }
    *reinterpret_cast<float4*>(a.out + tk * N + col) = make_float4(sx, sy, sz, sw);
}

template<int TG, int TD>
__global__ void __launch_bounds__(256, 2) pk_moe_kernel(PkMoeArgs a, NativeExpertLayout L, float* __restrict__ gate,
                                                     float* __restrict__ up) {
    __shared__ PkShared sm;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    unsigned int* done = a.sync + kPkSyncGroups;
    unsigned int* span = a.sync + kPkSyncSpans;
    unsigned long long* const tr = tid == 0 ? a.trace : nullptr;
    if (tr) { pk_tmin(tr, PK_T_ENTRY_MIN); pk_tmax(tr, PK_T_ENTRY_MAX); }
    const uint32_t* s_grid = gu2::stage_grid<TG>(sm.grid, tid);
    if (tid == 0 && (int) blockIdx.x * 16 < a.n_expert)   // the layer's table into L2
        asm volatile("prefetch.global.L2 [%0];" ::"l"(a.ptab + blockIdx.x * 16));
    pdl_wait();
    if (tr) { pk_tmin(tr, PK_T_WAIT_MIN); pk_tmax(tr, PK_T_WAIT_MAX); }
    pk_plan(sm, a, tid);
    if (tr) { pk_tmin(tr, PK_T_PLAN_MIN); pk_tmax(tr, PK_T_PLAN_MAX); }
    const int ng = sm.ng;
    const int nx = (int) (2 * L.n_ff + GU2_ROWS - 1) / GU2_ROWS;
    const auto* xq = (const block_q8_1*) a.xq;

    // gate/up: item (row block bx, group by); the group's counter takes one per finished item
    for (int it = blockIdx.x; it < nx * ng; it += gridDim.x) {
        const int bx = it % nx, by = it / nx;
        const int row = bx * GU2_ROWS + 2 * warp;
        unsigned long long t0 = 0;
        if (tr) { t0 = pk_now(); atomicMin(tr + PK_T_GU_FIRST, t0); }
        if (row < 2 * L.n_ff)
            gu2::pair<TG>((const uint8_t*) sm.gptr[by], row, sm.start[by], sm.start[by + 1], sm.tok, xq, L, gate, up,
                          s_grid, lane);
        __syncthreads();
        if (tr) {
            const unsigned long long t1 = pk_now();
            atomicMax(tr + PK_T_GU_LAST, t1);
            atomicMin(tr + PK_T_GU_END_MIN, t1);
            atomicAdd(tr + PK_T_GU_BUSY, t1 - t0);
            atomicAdd(tr + PK_T_GU_ITEMS, 1ull);
        }
        if (tid == 0) red_release_add_gpu(done + by, 1u);
    }

    // down: item (group, 64-column span); a block takes the group's SwiGLU once its gate/up rows are all in
    constexpr int R = dm_down_rows<TD>(), SUB = kPkSpan / R;
    const int nsp = (int) ((L.n_embd + kPkSpan - 1) / kPkSpan);
    int g_staged = -1;
    for (int it = blockIdx.x; it < ng * nsp; it += gridDim.x) {
        const int g = it / nsp, sp = it - g * nsp;
        const int e0 = sm.start[g], e1 = sm.start[g + 1];
        __syncthreads();
        unsigned long long tw0 = 0;
        if (tr) tw0 = pk_now();
        if (g != g_staged) {
            if (tid == 0)
                while (ld_acquire_gpu(done + g) < (unsigned) nx) __nanosleep(20);
            __syncthreads();
            dm_swiglu_q8(gate, up, e0, e1, (int) L.n_ff, sm.hq, tid);
            g_staged = g;
            __syncthreads();
        }
        unsigned long long td0 = 0;
        if (tr) {
            td0 = pk_now();
            atomicAdd(tr + PK_T_SWIGLU_WAIT, td0 - tw0);
            atomicMin(tr + PK_T_DOWN_FIRST, td0);
        }
#pragma unroll
        for (int sb = 0; sb < SUB; ++sb)
            dm_down_item<TD>((const uint8_t*) sm.gptr[g] + L.down_off + (size_t) ((sp * SUB + sb) * R) * L.d_row, e0, e1,
                             sm.hq, L, sp * SUB + sb, sm.out + sb * R, tid, kPkSpan, sm.gptr[g]);
        __syncthreads();
        constexpr int q = kPkSpan / 4;
        const int nq = (e1 - e0) * q;
        for (int i = tid; i < nq; i += 256) {
            const int e = i / q, c = i - e * q;
            const int r0 = sp * kPkSpan + c * 4;
            if (r0 < L.n_embd)
                *(float4*) (a.parts + (size_t) sm.dst[e0 + e] * L.n_embd + r0) = *(const float4*) &sm.out[e * kPkSpan + c * 4];
        }
        __syncthreads();
        if (tr) {
            const unsigned long long t1 = pk_now();
            atomicMax(tr + PK_T_DOWN_LAST, t1);
            atomicAdd(tr + PK_T_DOWN_BUSY, t1 - td0);
            atomicAdd(tr + PK_T_DOWN_ITEMS, 1ull);
        }
        if (tid == 0) sm.last = pk_arrive(span + sp) + 1u == (unsigned) ng;
        __syncthreads();
        if (sm.last) {   // every group's rows of this span are in: the combine
            if (a.shared_flag && a.shared) {
                if (tid == 0)
                    while (ld_acquire_gpu(a.sync + kPkSyncShared) == 0u) __nanosleep(32);
                __syncthreads();
            }
            if (tr) pk_tmax(tr, PK_T_SHARED_SEEN);
            pk_combine_span(a, L, sp, tid);
            if (tr) pk_tmax(tr, PK_T_COMBINE_END);
            if (tid == 0) span[sp] = 0u;
        }
    }
    if (tr) { pk_tmax(tr, PK_T_BLOCK_END); pk_tmin(tr, PK_T_BLOCK_END_MIN); }
    pdl_trigger();   // the next layer's read may start on the SMs this launch leaves (every shared-row wait is past)
    // the last block to arrive zeroes the counters for the next launch
    __syncthreads();
    if (tid == 0) {
        if (pk_arrive(a.sync + kPkSyncArrive) + 1u == gridDim.x) {
            for (int g = 0; g < ng; ++g) done[g] = 0u;
            a.sync[kPkSyncShared] = 0u;
            a.sync[kPkSyncArrive] = 0u;
        }
    }
}

template<int TG, int TD>
int pk_moe_blocks() {
    static int b[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    if (b[dev] == 0) {
        int sms = 0, per = 0;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, pk_moe_kernel<TG, TD>, 256, 0) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        b[dev] = std::max(1, (sms - sms / 8) * per);
    }
    return b[dev];
}

__global__ void pk_shared_done_kernel(unsigned int* sync) { st_release_gpu(sync + kPkSyncShared, 1u); }

#define PK_PAIRS(X) \
    X(16, 20) X(17, 20) X(18, 20) X(21, 20) X(22, 20) X(16, 42) X(17, 42) X(18, 42) X(21, 42) X(22, 42)

}  // namespace

bool pk_moe_ok(const NativeExpertLayout& L) {
    if (L.n_ff != 640 || L.n_embd != 2560) return false;
    switch (L.gu_type * 100 + L.d_type) {
#define PK_OK(TG, TD) case TG * 100 + TD: return true;
        PK_PAIRS(PK_OK)
#undef PK_OK
        default: return false;
    }
}

size_t pk_moe_scratch_bytes(int n_ff) { return (size_t) kDmCap * n_ff * 2 * sizeof(float) + 256; }

void pk_moe_experts(const NativeExpertLayout& L, const PkMoeArgs& a, void* scratch, void* stream) {
    float* gate = (float*) scratch;
    float* up = gate + (size_t) kDmCap * L.n_ff;
    const int n = a.n_tok * a.k;
    if (n > kDmCap || a.k != 10 || a.n_tok < 1) {
        basalt::log::error("basalt", "pk_moe_experts: %d tokens, k %d", a.n_tok, a.k);
        std::exit(1);
    }
    const int items = (int) ((2 * L.n_ff + GU2_ROWS - 1) / GU2_ROWS) * n;
    cudaStream_t s = (cudaStream_t) stream;
    bool done = false;
    switch (L.gu_type * 100 + L.d_type) {
#define PK_GO(TG, TD) \
    case TG * 100 + TD: { \
        const int blocks = std::max(1, std::min(items, pk_moe_blocks<TG, TD>())); \
        launch_dependent(pk_moe_kernel<TG, TD>, dim3((unsigned) blocks), s, a, L, gate, up); \
        done = true; \
        break; \
    }
        PK_PAIRS(PK_GO)
#undef PK_GO
        default: break;
    }
    if (!done) {
        basalt::log::error("basalt", "pk_moe_experts: formats %d / %d", L.gu_type, L.d_type);
        std::exit(1);
    }
    check("pk_moe_experts");
}

void pk_moe_shared_done(unsigned int* sync, void* stream) {
    pk_shared_done_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(sync);
    check("pk_moe_shared_done");
}

}  // namespace basalt::kernels
