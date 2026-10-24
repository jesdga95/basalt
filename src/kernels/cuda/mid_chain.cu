// src/kernels/cuda/mid_chain.cu - see include/basalt/kernels/mid_chain.hpp.
//
// Its own translation unit: the device code of dense_tc.cu / dec_gdn_rec.cu / hc_fused.cu (and the helpers of
// attn_chain_kernel.cuh's phase code) without their host sides.  Built WITHOUT fast math, like them.
#define FZ_DEV_ONLY
#include "hc_fused.cu"
#define smem smem_dtc
#include "dense_tc.cu"
#undef smem
#define div_approx_ftz div_approx_ftz_rec
#include "dec_gdn_rec.cu"
#undef div_approx_ftz
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/mid_chain.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

using namespace basalt::kernels;
// the wait helpers of the bound header: this translation unit keeps its own copies
#define mkb_hang mid_mkb_hang
#define mkb_mwait mid_mkb_mwait
#define mkb_now mid_mkb_now
#define mkb_try mid_mkb_try
#include "attn_chain_bound.cuh"
#define mkc mkmid
#include "attn_chain_kernel.cuh"
#include "basalt/log.hpp"

namespace basalt::kernels {
bool tmap_2d(CUtensorMap* m, const void* base, CUtensorMapDataType ty, uint64_t cols, uint64_t rows, uint64_t stride, uint32_t box_c, uint32_t box_r);
}

namespace mkc {   // mkmid

// the kernel's argument record
struct MP {
    Geo geo;
    const uint8_t* xq;                   // the first read's q8_1 rows
    const uint8_t *wq, *wz, *wo;
    DecRecArgs rec;
    float *qkv, *z;                      // the pair's outputs
    float* bo;                           // the output projection's result
    float* abv;                          // the rec heads' alpha/beta [R_HV][2 XT]
    unsigned* cnt;                       // [2][NCTR] hand-off counters, then the arrival ticket
    unsigned par;                        // the counter set of this launch
    int pair_only;                       // bench: stop after the pair
    TmaDesc stm;                         // rec state [S rows][R_HV * S], box S x S
    unsigned long long* stamps;          // bench: [launch][CTA][point] phase times
    int lay;                             // bench: the launch's index in the stamp buffer
};
// the stamp points
enum { ST_ENTRY, ST_WAIT, ST_TILE0, ST_PAIR_REL = ST_TILE0 + 7, ST_PZ_REL = ST_PAIR_REL + 2, ST_PAIR_END = ST_PZ_REL + 2, ST_OUT_REQ0, ST_OUT_REQ1,
       ST_OFULL, ST_R_PAIR, ST_R_STATE, ST_R_TOKENS, ST_R_PZ, ST_R_REC, ST_O_REC, ST_O_DONE, ST_N };
static_assert(ST_N == kMidStampPts, "stamp points");
// the CTA's shared-memory tail: barriers, the IQ3_S grid, the layer's pointers and arguments
struct MPerm {
    uint64_t slot[P_SLOTS], ofull, oxbar, rbar;
    uint32_t grid[512];
    const uint8_t *lwq, *lwz, *lwo, *lyq;
    float* lbo;
    DecRecArgs ra;
    Geo geo;
    int vb;   // logical CTA id (arrival ticket)
};
static_assert(sizeof(MPerm) <= 4096, "perm");

__device__ __forceinline__ void m_tile(const MP& p, const MPerm& L, int tile, const uint8_t*& src, int& bytes, int& type, float*& y, int& n_out, int& row0) {
    if (tile < P_NTQ) { src = L.lwq + (size_t) tile * L.geo.tbq; bytes = L.geo.tbq; type = L.geo.tq; y = p.qkv; n_out = P_NQ; row0 = tile * 16; }
    else { const int t = tile - P_NTQ; src = L.lwz + (size_t) t * L.geo.tbz; bytes = L.geo.tbz; type = L.geo.tz; y = p.z; n_out = P_NZ; row0 = t * 16; }
}
// tile k of this CTA (k < ns: issued at the start) into slot k % ns
__device__ __forceinline__ void m_fill(const MP& p, MPerm& P, const PSplit& ps, int k, char* sm) {
    const uint8_t* s; int by, ty, no, r0; float* y;
    m_tile(p, P, ps.tile(k), s, by, ty, y, no, r0);
    const int sl = k % P.geo.ns;
    mbar_expect_tx(&P.slot[sl], by);
    bulk_w2s(sm + (size_t) sl * P.geo.slotb, s, by, &P.slot[sl]);
}

__device__ __forceinline__ void stamp_at(const MP& p, int b, int pt) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    p.stamps[((size_t) p.lay * kMidStampCtas + b) * kMidStampPts + pt] = t;
}

template <int T, int FQ, int FZ, int FO, bool ST>
__global__ void __launch_bounds__(CT, 1) k_mid(const __grid_constant__ MP p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define STP(cond, pt) do { if constexpr (ST) { if (cond) stamp_at(p, b, pt); } } while (0)
    extern __shared__ __align__(128) char sm_dyn[];
    char* sm = sm_dyn;
    MPerm& P = *reinterpret_cast<MPerm*>(sm + OFF_PERM);
    RecS& RS_ = *reinterpret_cast<RecS*>(sm + OFF_REC);
    const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, G = gridDim.x;
    unsigned tk = 0;
    if (tid == 0) tk = atomicInc(&p.cnt[2 * NCTR], (unsigned) (G - 1));   // arrival ticket; wraps, never cleared
    if (tid == 0) {
        for (int s = 0; s < P_SLOTS; ++s) mbar_init(&P.slot[s], 1);
        mbar_init(&P.ofull, 1); mbar_init(&P.oxbar, 1); mbar_init(&P.rbar, 1);
    }
    {   // the layer's arguments: one batch of independent loads, issued before any bulk copy
        constexpr int NR = (int) (sizeof(DecRecArgs) / 4);
        static_assert(NR <= CT, "argument batch");
        int v = 0;
        if (tid < NR) v = reinterpret_cast<const int*>(&p.rec)[tid];
        for (int i = tid; i < 512; i += CT) P.grid[i] = iq3s_grid[i];
        if (tid < NR) reinterpret_cast<int*>(&P.ra)[tid] = v;
        if (tid == 0) { P.geo = p.geo; P.lwq = p.wq; P.lwz = p.wz; P.lwo = p.wo; P.lyq = static_cast<const uint8_t*>(p.rec.yq); P.lbo = p.bo; }
    }
    if (tid == 0) P.vb = (int) (tk % (unsigned) G);
    __syncthreads();
    const int b = P.vb;
    STP(tid == 0, ST_ENTRY);
    unsigned PH = 0;   // mbarrier parities (PB_*)
    unsigned* c = p.cnt + (size_t) p.par * NCTR;
    const PSplit ps = p_split(b, G, -3);
    const int mine = ps.n;
    const bool rec_role = b >= R_CTA0 && b < R_CTA0 + R_HV, out_role = b < O_NOUT / 16;
    // the weights do not depend on the first read: the CTA's first pair tiles and the rec head's state and conv history
    // are requested before the grid dependency wait
    if (tid == 0) { pfence(); for (int k = 0; k < P.geo.ns && k < mine; ++k) m_fill(p, P, ps, k, sm); }
    if (rec_role && tid == 96) {
        const int head = b - R_CTA0, qh = head % R_HK, qk = S * R_HK;
        const float* st0 = p.rec.state;
        const float* hist = st0 + (size_t) S * R_HV * S;
        pf_l2(st0 + (size_t) head * S * S, (size_t) S * S * 4);
        pf_l2(hist + (size_t) qh * S * 3, (size_t) S * 3 * 4);
        pf_l2(hist + (size_t) (qk + qh * S) * 3, (size_t) S * 3 * 4);
        pf_l2(hist + (size_t) (2 * qk + head * S) * 3, (size_t) S * 3 * 4);
    }
    pdl_wait();
    STP(tid == 0, ST_WAIT);
    if (b == 0 && tid == 0) for (int k = 0; k < NCTR; ++k) p.cnt[(1 - p.par) * NCTR + k] = 0;   // the set the next launch uses
    pdl_launch_dependents();
    // ======== pair
    {
        const int gi = wp >> 3, lw = wp & 7, ltid = tid & 255, g = lane >> 2, t = lane & 3;
        const int hb = lw & 3, n = hb & 1, nchunk = P_NIN / 512, nb32 = P_NIN / 32;
        float* gred = reinterpret_cast<float*>(sm + OFF_PRED) + gi * 1024;
        XFrag xs[3];
#pragma unroll
        for (int j = 0; j < 3; ++j)
            if (mine > 0 && (lw >> 2) + 2 * j < nchunk) pair_xfrag_cg(xs[j], (lw >> 2) + 2 * j, hb, g, t, p.xq, nb32, T);
        if (ltid == 0 && (gi >= mine || ps.tile(gi) >= P_NTQ)) { STP(true, ST_PAIR_REL + gi); redrel(&c[C_PAIR]); }   // no qkv tile in this group
        for (int k = gi; k < mine; k += 2) {
            const int ns = P.geo.ns, slot = k % ns;
            const uint8_t* s; int by, ty, no, row0; float* y;
            m_tile(p, P, ps.tile(k), s, by, ty, y, no, row0);
            mwait(&P.slot[slot], ((PH >> (PB_SLOT + slot)) ^ (unsigned) (k / ns)) & 1u);   // slot uses before tile k: k / ns
            const uint8_t* sl = reinterpret_cast<const uint8_t*>(sm) + (size_t) slot * P.geo.slotb;
            const int row_bytes = (P_NIN / 256) * sb_bytes(ty);
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            by_tq_tz<FQ, FZ>(ty, P.geo.tq, [&](auto TY) { pair_terms_xr<decltype(TY)::value>(acc, sl + g * row_bytes, sl + (g + 8) * row_bytes, lw, hb, n, t, nchunk, xs, P.grid); });
            float* rw = gred + lw * 16 * kMaxCols;
            rw[g * kMaxCols + 2 * t] = acc[0];
            rw[g * kMaxCols + 2 * t + 1] = acc[1];
            rw[(g + 8) * kMaxCols + 2 * t] = acc[2];
            rw[(g + 8) * kMaxCols + 2 * t + 1] = acc[3];
            barn(2 + gi, 256);
            STP(ltid == 0, ST_TILE0 + k);
            const bool qrel = ps.tile(k) < P_NTQ && (k + 2 >= mine || ps.tile(k + 2) >= P_NTQ);   // this group's last qkv tile
            if (ltid == 0 && k + ns < mine && !qrel) { pfence(); m_fill(p, P, ps, k + ns, sm); }   // the slot's next tile
            if (ltid < 16 * kMaxCols) {
                const int row = ltid / kMaxCols, col = ltid % kMaxCols;
                if (col < T) {
                    const float* rr2 = gred + row * kMaxCols + col;
                    constexpr int WS = 16 * kMaxCols;
                    const float v = ((rr2[0] + rr2[4 * WS]) + (rr2[WS] + rr2[5 * WS])) + ((rr2[2 * WS] + rr2[6 * WS]) + (rr2[3 * WS] + rr2[7 * WS]));
                    y[(size_t) col * no + row0 + row] = v;
                }
            }
            barn(2 + gi, 256);
            if (b >= Q3_0 && b < Q3_0 + 40 && gi == 1 && k == 1) {   // the rec heads' alpha/beta (job = head * 2T + 2t + beta)
                for (int j = (b - Q3_0) * 8 + lw; j < R_HV * 2 * T; j += 320) {
                    const int head = j / (2 * T), r = j % (2 * T);
                    const float v = ab_one_g(P.ra, head, r & 1, r >> 1, lane);
                    if (lane == 0) p.abv[head * 2 * XT + r] = v;
                }
                barn(2 + gi, 256);
            }
            if (ltid == 0 && qrel) { STP(true, ST_PAIR_REL + gi); redrel(&c[C_PAIR]); }
            if (ltid == 0 && k + ns < mine && qrel) { pfence(); m_fill(p, P, ps, k + ns, sm); }   // after the release
        }
        if (ltid == 0) { STP(true, ST_PZ_REL + gi); redrel(&c[C_PZ]); }
        // every slot's uses by both groups
        for (int k = 0; k < mine; ++k) PH ^= 1u << (PB_SLOT + k % P.geo.ns);
        cbar();
        STP(tid == 0, ST_PAIR_END);
    }
    if (p.pair_only != 0) return;
    // ======== out tile into smem (the pair's ring is free; rec lives above it)
    if (out_role && !rec_role && tid == 32) {
        pfence();
        STP(true, ST_OUT_REQ0);
        mbar_expect_tx(&P.ofull, 16 * P.geo.orb);
        for (int r = 0; r < 16; ++r) bulk_w2s(sm + r * P.geo.ors, P.lwo + (size_t) (b * 16 + r) * P.geo.orow, P.geo.orb, &P.ofull);
        if constexpr (FO == 8) for (int r = 0; r < 16; ++r) pf_l2(P.lwo + (size_t) (b * 16 + r) * P.geo.orow + P.geo.orb, P.geo.orow - P.geo.orb);   // the row tails
        STP(true, ST_OUT_REQ1);
    }
    // ======== rec (48 CTAs, head = b - R_CTA0; rec_rank's code with RecS)
    if (rec_role) {
        const DecRecArgs& a = P.ra;
        const int head = b - R_CTA0, rank = tid / 128, ltid = tid % 128, rg = ltid / 32, bid = 8 + rank, col = rank * 32 + lane;
        const int qh = head % a.h_k, qk = S * a.h_k, value_dim = S * a.h_v, Cc = a.C;
        const int cq = qh * S + ltid, ck = qk + qh * S + ltid, cv = 2 * qk + head * S + col;
        float wq[4], wk[4], wv[4], hq[3], hk[3], hv[3];
        {
            const float* hist = a.state + (size_t) S * a.h_v * S;
#pragma unroll
            for (int j = 0; j < 4; ++j) { wq[j] = __ldg(a.conv_w + cq * 4 + j); wk[j] = __ldg(a.conv_w + ck * 4 + j); wv[j] = __ldg(a.conv_w + cv * 4 + j); }
#pragma unroll
            for (int j = 0; j < 3; ++j) { hq[j] = hist[cq * 3 + j]; hk[j] = hist[ck * 3 + j]; hv[j] = hist[cv * 3 + j]; }
        }
        const float gam = __ldg(a.gamma + col);
        if (tid == 0) spin_ge(&c[C_PAIR], (unsigned) (2 * G));
        cbar();
        STP(tid == 0, ST_R_PAIR);
        if (rg == 0 && lane < 2 * T) {   // every rank's copy of the head's alpha/beta (computed in the pair phase)
            const float v = __uint_as_float(ldca32(p.abv + head * 2 * XT + lane));
            ((lane & 1) ? RS_.b[rank] : RS_.g[rank])[lane >> 1] = v;
        }
        float xq[XT], xk[XT], xv[XT], zz[XT];
#pragma unroll
        for (int t = 0; t < XT; ++t) {
            xq[t] = t < T ? a.qkv[(size_t) t * Cc + cq] : 0.0f;
            xk[t] = t < T ? a.qkv[(size_t) t * Cc + ck] : 0.0f;
            xv[t] = t < T && rg == 1 ? a.qkv[(size_t) t * Cc + cv] : 0.0f;
            zz[t] = 0.0f;
        }
        // the state tile: after the hop and the qkv loads (an in-flight 64 KB copy delays this SM's loads and polls)
        if (tid == 0) { pfence(); mbar_expect_tx(&P.rbar, S * S * 4); tma_2d(sm, reinterpret_cast<const CUtensorMap*>(&p.stm), head * S, 0, &P.rbar); }
        float yq[XT], yk[XT];
#pragma unroll
        for (int t = 0; t < XT; ++t) {
            if (t >= T) break;
            yq[t] = conv_silu_x(hq, wq, xq, t);
            yk[t] = conv_silu_x(hk, wk, xk, t);
            float sq = yq[t] * yq[t], sk = yk[t] * yk[t];
            for (int o = 16; o > 0; o >>= 1) { sq += __shfl_xor_sync(0xffffffffu, sq, o); sk += __shfl_xor_sync(0xffffffffu, sk, o); }
            if (lane == 0) { RS_.part[rank][0][t][rg] = sq; RS_.part[rank][1][t][rg] = sk; }
            if (rg == 1) {
                const float yv = conv_silu_x(hv, wv, xv, t);
                RS_.v[rank][t][lane] = yv;
                a.h[(size_t) t * Cc + cv] = yv;
            }
        }
        barn(bid, 128);
        const bool write_qk = rank == 0 && head < a.h_k;
#pragma unroll
        for (int t = 0; t < XT; ++t) {
            if (t >= T) break;
            const float ssq = RS_.part[rank][0][t][0] + RS_.part[rank][0][t][1] + RS_.part[rank][0][t][2] + RS_.part[rank][0][t][3];
            const float ssk = RS_.part[rank][1][t][0] + RS_.part[rank][1][t][1] + RS_.part[rank][1][t][2] + RS_.part[rank][1][t][3];
            yq[t] *= rsqrtf(ssq + a.eps);
            yk[t] *= rsqrtf(ssk + a.eps);
            if (rank == 0) { RS_.qn[t][ltid] = yq[t]; RS_.kn[t][ltid] = yk[t]; }
            if (write_qk) { a.h[(size_t) t * Cc + cq] = yq[t]; a.h[(size_t) t * Cc + ck] = yk[t]; }
        }
        if (ltid < T) {
            if (rank == 0) a.gate[(size_t) ltid * a.h_v + head] = RS_.g[rank][ltid];
            if (rank == 1) a.beta[(size_t) ltid * a.h_v + head] = RS_.b[rank][ltid];
        }
        MW(&P.rbar, PB_REC);   // the state tile (tensor copy issued above)
        STP(tid == 0, ST_R_STATE);
        float sreg[RPG];
        {
            const float* stile = reinterpret_cast<const float*>(sm);
#pragma unroll
            for (int r = 0; r < RPG; ++r) sreg[r] = stile[(rg * RPG + r) * S + col];
        }
        cbar();   // the state tile is consumed (the out tile lands there); also the ranks' qn/kn
        if (out_role && tid == 32) {
            pfence();
            STP(true, ST_OUT_REQ0);
            mbar_expect_tx(&P.ofull, 16 * P.geo.orb);
            for (int r = 0; r < 16; ++r) bulk_w2s(sm + r * P.geo.ors, P.lwo + (size_t) (b * 16 + r) * P.geo.orow, P.geo.orb, &P.ofull);
            if constexpr (FO == 8) for (int r = 0; r < 16; ++r) pf_l2(P.lwo + (size_t) (b * 16 + r) * P.geo.orow + P.geo.orb, P.geo.orow - P.geo.orb);   // the row tails
            STP(true, ST_OUT_REQ1);
        }
        float wsum_t = 0.0f;
        for (int t = 0; t < T; ++t) {
            const float gg = __expf(RS_.g[rank][t]);
            float kv = 0.0f;
#pragma unroll
            for (int r = 0; r < RPG; ++r) kv = fmaf(sreg[r], RS_.kn[t][rg * RPG + r], kv);
            RS_.red_kv[rank][rg][lane] = kv;
            barn(bid, 128);
            const float kv_col = RS_.red_kv[rank][0][lane] + RS_.red_kv[rank][1][lane] + RS_.red_kv[rank][2][lane] + RS_.red_kv[rank][3][lane];
            const float delta = (RS_.v[rank][t][lane] - gg * kv_col) * RS_.b[rank][t];
            float o = 0.0f;
#pragma unroll
            for (int r = 0; r < RPG; ++r) {
                sreg[r] = fmaf(gg, sreg[r], RS_.kn[t][rg * RPG + r] * delta);
                o = fmaf(sreg[r], RS_.qn[t][rg * RPG + r], o);
            }
            RS_.red_o[rank][rg][lane] = o;
            barn(bid, 128);
            if (rg == 0)
                RS_.oc[rank][t][lane] = (RS_.red_o[rank][0][lane] + RS_.red_o[rank][1][lane] + RS_.red_o[rank][2][lane] + RS_.red_o[rank][3][lane]) * rsqrtf((float) S);
        }
        if (rg == 0)   // the sums of squares after the chain (the same butterfly per token)
            for (int t = 0; t < T; ++t) {
                const float sq_part = sq32(RS_.oc[rank][t][lane]);
                if (lane == t) wsum_t = sq_part;
            }
        if (rg == 0 && lane < T) RS_.wsum[lane][rank] = wsum_t;
        STP(tid == 0, ST_R_TOKENS);
        if (tid == 0) spin_ge(&c[C_PZ], (unsigned) (2 * G));
        cbar();
        STP(tid == 0, ST_R_PZ);
        if (rg == 0)
#pragma unroll
            for (int t = 0; t < XT; ++t) zz[t] = t < T ? a.z[(size_t) t * value_dim + head * S + col] : 0.0f;
        if (rg == 0) {
#pragma unroll
            for (int t = 0; t < XT; ++t) {
                if (t >= T) break;
                const float ss = RS_.wsum[t][0] + RS_.wsum[t][1] + RS_.wsum[t][2] + RS_.wsum[t][3];
                norm_q8(a, t, head, rank, lane, RS_.oc[rank][t][lane], ss, gam, zz[t]);
            }
        }
        cbar();
        if (tid == 0) { STP(true, ST_R_REC); redrel(&c[C_REC]); }
    }
    // ======== out (dense_tc_one<type, 24 logical warps> on 16 warps)
    if (out_role) {
        const int g = lane >> 2, t = lane & 3, nb32 = O_NIN / 32;
        const int ors = P.geo.ors, osb = sb_bytes(P.geo.to);
        constexpr bool oxs_fixed = FO != 0 && make_geo(23, 23, FO).ox + T * (O_NIN / 32) * kQ81 <= OFF_PERM;
        const bool oxs = FO != 0 ? oxs_fixed : P.geo.ox + T * nb32 * kQ81 <= OFF_PERM;   // x staged in smem (else read from L2 through L1)
        const uint8_t* xsm = oxs ? reinterpret_cast<const uint8_t*>(sm + P.geo.ox) : P.lyq;
        float* red = reinterpret_cast<float*>(sm + P.geo.ored);
        constexpr int RC = O_RC;
        if (oxs) {
            if (tid == 0) hop_copy(&c[C_REC], R_HV, sm + P.geo.ox, P.lyq, (unsigned) (T * nb32 * kQ81), &P.oxbar);
            MW(&P.oxbar, PB_OX);
        } else {
            if (tid == 0) spin_ge(&c[C_REC], R_HV);
            cbar();
        }
        STP(tid == 0, ST_O_REC);
        MW(&P.ofull, PB_OW);
        STP(tid == 0, ST_OFULL);
        const uint8_t* slot = reinterpret_cast<const uint8_t*>(sm);
        if constexpr (FO == 8) {
            // Q8_0: the warp order of the one-kernel projection it replaces (dense_tc's 1024-value chunks over two warp
            // groups): warp 4 * ks + hb adds its half super-block of the 512-value chunks 2 j + ks in order
            if (wp < 8) {
                const int hb = wp & 3, ks = wp >> 2, sbi = hb >> 1, n = hb & 1;
                const uint8_t* rowA = slot + g * ors + sbi * osb;
                const uint8_t* rowB = slot + (g + 8) * ors + sbi * osb;
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
                for (int j = 0; j < O_NIN / 1024; ++j) {
                    const int cch = 2 * j + ks;
                    XFrag xf;
#pragma unroll
                    for (int k4 = 0; k4 < 4; ++k4) {
                        const int k = 16 * cch + 4 * hb + k4;
                        xf.b0[k4] = xf.b1[k4] = 0;
                        if (g < T) {
                            const uint8_t* xb = xsm + ((size_t) g * nb32 + k) * kQ81;
                            if (oxs) {
                                xf.b0[k4] = *(reinterpret_cast<const unsigned int*>(xb + 4) + t);
                                xf.b1[k4] = *(reinterpret_cast<const unsigned int*>(xb + 20) + t);
                            } else {
                                xf.b0[k4] = ldca32(xb + 4 + 4 * t);
                                xf.b1[k4] = ldca32(xb + 20 + 4 * t);
                            }
                        }
#pragma unroll
                        for (int e2 = 0; e2 < 2; ++e2) {
                            const int col = 2 * t + e2;
                            const uint8_t* xd = xsm + ((size_t) col * nb32 + k) * kQ81;
                            xf.d8[k4][e2] = col < T ? (oxs ? __low2float(*reinterpret_cast<const half2*>(xd)) : __half2float(__ushort_as_half(ldca16(xd)))) : 0.0f;
                        }
                    }
                    constexpr int SB = Fmt<8>::SB;
                    const uint8_t* pp[2] = {rowA + cch * 2 * SB, rowB + cch * 2 * SB};
                    if (cch >= O_Q8_SMEM_CH) {   // past the smem part: this CTA's rows in the weights (prefetched to L2)
                        const uint8_t* gw = P.lwo + (size_t) (b * 16) * P.geo.orow + sbi * osb + cch * 2 * SB;
                        pp[0] = gw + (size_t) g * P.geo.orow; pp[1] = gw + (size_t) (g + 8) * P.geo.orow;
                    }
                    half_sb_terms<8>(acc, pp, n, t, xf, 0u, 0u, 0u, 0u, P.grid);
                }
                float* rw = red + wp * 16 * RC;
                if (2 * t + 1 < RC) {
                    rw[g * RC + 2 * t] = acc[0];
                    rw[g * RC + 2 * t + 1] = acc[1];
                    rw[(g + 8) * RC + 2 * t] = acc[2];
                    rw[(g + 8) * RC + 2 * t + 1] = acc[3];
                }
            }
        } else {
            for (int lw = wp; lw < O_NLW; lw += 16) {
                const int hb = lw & 3, sbi = hb >> 1, n = hb & 1;
                const uint8_t* rowA = slot + g * ors + sbi * osb;
                const uint8_t* rowB = slot + (g + 8) * ors + sbi * osb;
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                XFrag xf[2];
    #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int cch = (lw >> 2) + cc * (O_NLW / 4);
    #pragma unroll
                    for (int k4 = 0; k4 < 4; ++k4) {
                        const int k = 16 * cch + 4 * hb + k4;
                        xf[cc].b0[k4] = xf[cc].b1[k4] = 0;
                        if (g < T) {
                            const uint8_t* xb = xsm + ((size_t) g * nb32 + k) * kQ81;
                            if (oxs) {
                                xf[cc].b0[k4] = *(reinterpret_cast<const unsigned int*>(xb + 4) + t);
                                xf[cc].b1[k4] = *(reinterpret_cast<const unsigned int*>(xb + 20) + t);
                            } else {
                                xf[cc].b0[k4] = ldca32(xb + 4 + 4 * t);
                                xf[cc].b1[k4] = ldca32(xb + 20 + 4 * t);
                            }
                        }
    #pragma unroll
                        for (int e2 = 0; e2 < 2; ++e2) {
                            const int col = 2 * t + e2;
                            const uint8_t* xd = xsm + ((size_t) col * nb32 + k) * kQ81;
                            xf[cc].d8[k4][e2] = col < T ? (oxs ? __low2float(*reinterpret_cast<const half2*>(xd)) : __half2float(__ushort_as_half(ldca16(xd)))) : 0.0f;
                        }
                    }
                }
                by_to<FO>(P.geo.to, [&](auto TY) {
                    constexpr int TT = decltype(TY)::value, SB = Fmt<TT>::SB;
                    uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
                    if constexpr (TT == 23) {
                        const uint32_t* tbl = reinterpret_cast<const uint32_t*>(dtc_iq4nl);
                        t0 = tbl[0]; t1 = tbl[1]; t2 = tbl[2]; t3 = tbl[3];
                    }
    #pragma unroll
                    for (int cc = 0; cc < 2; ++cc) {
                        const int cch = (lw >> 2) + cc * (O_NLW / 4);
                        const uint8_t* const pp[2] = {rowA + cch * 2 * SB, rowB + cch * 2 * SB};
                        half_sb_terms<TT>(acc, pp, n, t, xf[cc], t0, t1, t2, t3, P.grid);
                    }
                });
                float* rw = red + lw * 16 * RC;
                if (2 * t + 1 < RC) {
                    rw[g * RC + 2 * t] = acc[0];
                    rw[g * RC + 2 * t + 1] = acc[1];
                    rw[(g + 8) * RC + 2 * t] = acc[2];
                    rw[(g + 8) * RC + 2 * t + 1] = acc[3];
                }
            }
        }
        cbar();
        if (tid < 16 * RC) {
            const int row = tid / RC, col = tid % RC;
            if (col < T) {
                constexpr int WS = 16 * RC;
                const float* rr2 = red + row * RC + col;
                float bov;
                if constexpr (FO == 8) {   // ((hb0 + hb1) + hb2) + hb3 per warp group, then the groups
                    bov = (((rr2[0] + rr2[WS]) + rr2[2 * WS]) + rr2[3 * WS]) + (((rr2[4 * WS] + rr2[5 * WS]) + rr2[6 * WS]) + rr2[7 * WS]);
                } else {
                    float h[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        float v = rr2[q * WS];
#pragma unroll
                        for (int j = 1; j < O_NLW / 4; ++j) v += rr2[(4 * j + q) * WS];
                        h[q] = v;
                    }
                    bov = (h[0] + h[1]) + (h[2] + h[3]);
                }
                P.lbo[(size_t) col * O_NOUT + b * 16 + row] = bov;
            }
        }
        STP(tid == 0, ST_O_DONE);
    }
#undef STP
#endif
}

}  // namespace mkc

namespace basalt::kernels {
namespace {
using namespace mkmid;

struct MidDev {
    bool checked = false, ok = false;
    unsigned* cnt = nullptr;   // [2][NCTR] hand-off counters, then the arrival ticket
    float* abv = nullptr;
};
MidDev g_mid[16];
MidDev& mid_dev() {
    int dev = 0;
    cudaGetDevice(&dev);
    static MidDev none;
    if (dev < 0 || dev >= 16) return none;
    return g_mid[dev];
}

constexpr int kMidCtas = R_CTA0 + R_HV;
using MFn = void (*)(MP);
bool type_ok(int t) { return t == 8 || t == 12 || t == 13 || t == 14 || t == 21 || t == 23; }
// Q8_0 has its own instance: all three types Q8_0, not the stamping build
bool types_ok(const MidChainArgs& a) {
    if (!type_ok(a.type_q) || !type_ok(a.type_z) || !type_ok(a.type_o)) return false;
    const int n8 = (a.type_q == 8) + (a.type_z == 8) + (a.type_o == 8);
    return n8 == 0 || (n8 == 3 && a.stamps == nullptr);
}

// One kernel per (window rows, projection types) for the projection type triples of the packs, and the generic one (the
// types picked at run time) for any other.
#define MID_TRIPLES X(14, 14, 14) X(23, 21, 12) X(8, 8, 8)
template <int T>
MFn mid_pick_t(int q, int z, int o) {
#define X(a, b, c) if (q == a && z == b && o == c) return k_mid<T, a, b, c, false>;
    MID_TRIPLES
#undef X
    return k_mid<T, 0, 0, 0, false>;
}
// the stamping instances (bench only): (14, 14, 14) and the generic one
template <int T>
MFn mid_pick_st(int q, int z, int o) {
    if (q == 14 && z == 14 && o == 14) return k_mid<T, 14, 14, 14, true>;
    return k_mid<T, 0, 0, 0, true>;
}
MFn mid_pick(int T, int q, int z, int o, bool st = false) {
    switch (T) {
        case 1: return st ? mid_pick_st<1>(q, z, o) : mid_pick_t<1>(q, z, o);
        case 2: return st ? mid_pick_st<2>(q, z, o) : mid_pick_t<2>(q, z, o);
        case 3: return st ? mid_pick_st<3>(q, z, o) : mid_pick_t<3>(q, z, o);
        default: return st ? mid_pick_st<4>(q, z, o) : mid_pick_t<4>(q, z, o);
    }
}
}  // namespace

void mid_chain_init() {
    MidDev& d = mid_dev();
    if (d.checked) return;
    d.checked = true;
    int dev = 0, optin = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (!hc_fused_supported() || sms < kMidCtas || optin < SMEM_MAX) {
        basalt::log::warn("basalt", "mid_chain: device not covered (%d SMs, %d B optin smem; need %d, %d): the unfused kernels run", sms, optin, kMidCtas, SMEM_MAX);
        return;
    }
    bool ok = cudaMalloc((void**) &d.cnt, (2 * NCTR + 1) * 4) == cudaSuccess &&
              cudaMemset(d.cnt, 0, (2 * NCTR + 1) * 4) == cudaSuccess &&
              cudaMalloc((void**) &d.abv, (size_t) R_HV * 2 * XT * 4) == cudaSuccess;
    for (int t = 1; ok && t <= kMidChainMaxT; ++t) {
        ok = cudaFuncSetAttribute((const void*) mid_pick(t, 0, 0, 0), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX) == cudaSuccess;
        if (ok) ok = cudaFuncSetAttribute((const void*) mid_pick(t, 0, 0, 0, true), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX) == cudaSuccess;
        if (ok) ok = cudaFuncSetAttribute((const void*) mid_pick(t, 14, 14, 14, true), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX) == cudaSuccess;
#define X(a, b, c) if (ok) ok = cudaFuncSetAttribute((const void*) mid_pick(t, a, b, c), cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX) == cudaSuccess;
        MID_TRIPLES
#undef X
    }
    cudaDeviceSynchronize();
    d.ok = ok && cudaGetLastError() == cudaSuccess;
    if (!d.ok) {
        basalt::log::error("basalt", "mid_chain: FATAL: kernel initialisation failed on a covered device (%s)", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
}

int mid_chain_ring_slots(int type_q, int type_z, int type_o) { return make_geo(type_q, type_z, type_o).ns; }

bool mid_chain_supported(const MidChainArgs& a) {
    const MidDev& d = mid_dev();
    return d.ok && a.T >= 1 && a.T <= kMidChainMaxT && types_ok(a) &&
           a.xq != nullptr && a.bo != nullptr && a.rec.yq != nullptr && a.rec.qkv != nullptr && a.rec.z != nullptr &&
           ((uintptr_t) a.wq & 15) == 0 && ((uintptr_t) a.wz & 15) == 0 && ((uintptr_t) a.wo & 15) == 0 &&
           ((uintptr_t) a.rec.state & 15) == 0 && ((uintptr_t) a.xq & 3) == 0 && a.rec.n_embd == 2560 && a.rec.h_v == R_HV &&
           a.rec.h_k == R_HK && a.rec.C == R_C;
}

void mid_chain(const MidChainArgs& a, void* stream) {
    MidDev& d = mid_dev();
    if (!mid_chain_supported(a)) {
        basalt::log::error("basalt", "mid_chain: unsupported arguments or device");
        std::exit(1);
    }
    MP p{};
    p.geo = make_geo(a.type_q, a.type_z, a.type_o);
    p.xq = static_cast<const uint8_t*>(a.xq);
    p.wq = static_cast<const uint8_t*>(a.wq);
    p.wz = static_cast<const uint8_t*>(a.wz);
    p.wo = static_cast<const uint8_t*>(a.wo);
    p.rec = a.rec;
    p.qkv = const_cast<float*>(a.rec.qkv);
    p.z = const_cast<float*>(a.rec.z);
    p.bo = a.bo;
    p.abv = d.abv;
    p.cnt = d.cnt;
    p.pair_only = a.pair_only ? 1 : 0;
    p.stamps = a.stamps;
    p.lay = a.ord;
    CUtensorMap stm_map;
    if (!tmap_2d(&stm_map, a.rec.state, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, (uint64_t) R_HV * S, S, (uint64_t) R_HV * S * 4, S, S)) {
        basalt::log::error("basalt", "mid_chain: the state's tensor map failed");
        std::exit(1);
    }
    std::memcpy(&p.stm, &stm_map, sizeof stm_map);
    // consecutive launches of a graph alternate between two counter sets; the first zeroes both, the others clear the
    // set the next one uses once their predecessor has completed (see the kernel)
    p.par = (unsigned) (a.ord & 1);
    if (a.ord == 0) cudaMemsetAsync(d.cnt, 0, (2 * NCTR + 1) * 4, (cudaStream_t) stream);
    launch_pdl(mid_pick(a.T, a.type_q, a.type_z, a.type_o, a.stamps != nullptr), dim3(kMidCtas), dim3(CT), (size_t) SMEM_MAX, (cudaStream_t) stream, p);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "mid_chain: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace basalt::kernels
