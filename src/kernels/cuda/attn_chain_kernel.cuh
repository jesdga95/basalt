namespace mkc {
constexpr int CT = 512, XT = 4, SMEM_MAX = 101376;
// ---- pair
// the projections' types are per layer: Q8_0 8, Q4_K 12, Q5_K 13, Q6_K 14, IQ3_S 21, IQ4_XS 23 (sb_bytes: a
// 256-value super-block's bytes; Q8_0 is eight 34-byte blocks)
constexpr int P_NIN = 2560, P_NQ = 10240, P_NZ = 6144, P_NTQ = P_NQ / 16, P_NT = P_NTQ + P_NZ / 16;
constexpr int P_SLOTS = 4, OFF_PRED = 87040;   // the ring: up to 4 slots of the layer's larger tile below OFF_PRED
// ---- rec
constexpr int R_HV = 48, R_HK = 16, R_C = 2 * R_HK * S + R_HV * S, R_ZV = R_HV * S, R_CTA0 = 122;   // [0, 64 KB): the head's state tile
// ---- out
// out: the tile [0, 16 * row stride), red (4 columns) after it, x staged after red where it fits below OFF_PERM
constexpr int O_Q8_SMEM_CH = 9;   // Q8_0 out: 512-value chunks of a row kept in smem (of 12)
constexpr int O_NIN = 6144, O_NOUT = 2560, O_NLW = 24, O_RC = 4, O_REDB = O_NLW * 16 * O_RC * 4;
// ---- permanent tail: barriers + the IQ3_S grid
constexpr int OFF_PERM = SMEM_MAX - 4096;
// the per-layer geometry of the typed phases (host-computed)
struct Geo {
    int tq, tz, to;     // types
    int tbq, tbz;       // pair tile bytes (16 rows)
    int ns, slotb;      // pair ring: slots, slot bytes
    int orb, ors, ored, ox;   // out: row bytes in smem, row stride, red offset, staged x offset
    int orow;           // out: a row's bytes in the weights (orb, or more when the row's tail is read from L2)
};
__host__ __device__ constexpr Geo make_geo(int tq, int tz, int to) {
    Geo g{};
    g.tq = tq; g.tz = tz; g.to = to;
    g.tbq = 16 * (P_NIN / 256) * sb_bytes(tq); g.tbz = 16 * (P_NIN / 256) * sb_bytes(tz);
    g.slotb = g.tbq > g.tbz ? g.tbq : g.tbz;
    g.ns = OFF_PRED / g.slotb < P_SLOTS ? OFF_PRED / g.slotb : P_SLOTS;
    // the two tile groups take alternate tiles: an odd ring would give a slot its consecutive tiles from different groups,
    // and one group's mbarrier wait for the slot's second use would pass before the first use's data arrived
    if (g.ns & 1) --g.ns;
    g.orow = (O_NIN / 256) * sb_bytes(to);
    // Q8_0 rows (6528 bytes) do not fit 16 to a tile: the first O_Q8_SMEM_CH chunks go to smem, the rest is prefetched to
    // L2 and read from there
    g.orb = to == 8 ? O_Q8_SMEM_CH * 2 * sb_bytes(8) : g.orow; g.ors = one_row_stride(g.orb);
    g.ored = (16 * g.ors + 127) & ~127; g.ox = g.ored + O_REDB;
    return g;
}
// f(integral_constant<TYPE>) for a projection type
template <typename F> __device__ __forceinline__ void by_type(int ty, F f) {
    switch (ty) {
        case 12: f(std::integral_constant<int, 12>()); break;
        case 13: f(std::integral_constant<int, 13>()); break;
        case 14: f(std::integral_constant<int, 14>()); break;
        case 21: f(std::integral_constant<int, 21>()); break;
        default: f(std::integral_constant<int, 23>()); break;
    }
}
// the kernel instance's types: FQ/FZ/FO fixed (a specialized instance; the layer's types are those), or 0 (generic:
// the layer's types at run time)
template <int FQ, int FZ, typename F> __device__ __forceinline__ void by_tq_tz(int ty, int tq, F f) {
    if constexpr (FQ == 0) by_type(ty, f);
    else if (FQ == FZ || ty == tq) f(std::integral_constant<int, FQ>()); else f(std::integral_constant<int, FZ>());
}
template <int FO, typename F> __device__ __forceinline__ void by_to(int ty, F f) {
    if constexpr (FO == 0) by_type(ty, f); else f(std::integral_constant<int, FO>());
}

__device__ __forceinline__ unsigned ldacq(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void redrel(unsigned* p) { asm volatile("red.release.gpu.global.add.u32 [%0],1;" ::"l"(p) : "memory"); }
__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, 512;" ::: "memory"); }
__device__ __forceinline__ void barn(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
#define mwait(b, p) mkb_mwait(smem_u32(b), p, __LINE__)
// thread / block index read opaquely: values derived from them are recomputed per phase instead of being hoisted
// out of the layer loop (and spilled: a local-memory reload after an acquire is an L2 round trip)
__device__ __forceinline__ void pfence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void spin_ge(const unsigned* c, unsigned target) {
    if (ldacq(c) < target) { const unsigned t0_ = mkb_now(); while (ldacq(c) < target) if (mkb_now() - t0_ > MKB_NS) mkb_hang(2, __LINE__); }
}
__device__ __forceinline__ void hop_copy(const unsigned* ctr, unsigned target, void* dst, const void* src, unsigned bytes, uint64_t* bar) {
    spin_ge(ctr, target);
    asm volatile("fence.proxy.async.global;" ::: "memory");
    pfence();
    mbar_expect_tx(bar, bytes);
    bulk_g2s(dst, src, bytes, bar);
}
__device__ __forceinline__ void pf_l2(const void* p, size_t n) {   // n multiple of 16
    const char* c = (const char*) p;
#pragma unroll 1
    for (size_t o = 0; o < n; o += 32768) {
        const unsigned sz = (unsigned) (n - o < 32768 ? n - o : 32768);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(c + o), "r"(sz) : "memory");
    }
}

enum { PB_WD = 0, PB_WU = 1, PB_A = 2, PB_SLOT = 3, PB_REC = 7, PB_OX = 8, PB_OW = 9 };   // bits of the parity register
#define MW(bar, bit) do { mwait(bar, (PH >> (bit)) & 1u); PH ^= 1u << (bit); } while (0)
enum { C_H1DN, C_H1UP, C_PAIR, C_REC, C_OUT, C_H2DN, C_H2UP, C_RT, C_END, C_REL, C_PZ, NCTR = 16 };
struct RecS {   // rec_rank's RecSmem for T <= XT, the alpha/beta rows shared by the 4 ranks
    float qn[XT][S], kn[XT][S];
    float v[4][XT][32], oc[4][XT][32];
    float red_kv[4][RG][32], red_o[4][RG][32];
    float part[4][2][XT][RG];
    float g[4][XT], b[4][XT];
    float wsum[XT][4];
    DecRecArgs a;
};
constexpr int OFF_REC = (OFF_PERM - (int) sizeof(RecS)) & ~127;
static_assert(OFF_REC >= 16 * 5040, "rec above the largest out tile (Q6_K)");
static_assert(OFF_REC >= 16 * one_row_stride(O_Q8_SMEM_CH * 2 * 272), "rec above the Q8_0 out tile");

extern __shared__ __align__(128) char sm_dyn[];
// weight copies: L2 evict-first policy (read once per layer; keeps the kernel's code and the activations in L2)
__device__ __forceinline__ void bulk_w2s(void* dst, const void* src, unsigned bytes, uint64_t* bar) {
#ifdef MK_NOEF
    bulk_g2s(dst, src, bytes, bar);
#else
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol) : "memory");
#endif
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ unsigned ldca32(const void* p) { unsigned v; asm volatile("ld.global.ca.u32 %0,[%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ unsigned short ldca16(const void* p) { unsigned short v; asm volatile("ld.global.ca.u16 %0,[%1];" : "=h"(v) : "l"(p)); return v; }
// pair_xfrag with L1-cached loads (L1 invalidated by the acquire): the replicas are written in this kernel (the nc path is only for kernel-constant data)
__device__ __forceinline__ void pair_xfrag_cg(XFrag& xf, int c, int hb, int g, int t, const uint8_t* x, int nb32, int ncols) {
#pragma unroll
    for (int k4 = 0; k4 < 4; ++k4) {
        const int k = 16 * c + 4 * hb + k4;
        xf.b0[k4] = xf.b1[k4] = 0;
        if (g < ncols) {
            const uint8_t* xb = x + ((size_t) g * nb32 + k) * kQ81;
            xf.b0[k4] = ldca32(reinterpret_cast<const unsigned int*>(xb + 4) + t);
            xf.b1[k4] = ldca32(reinterpret_cast<const unsigned int*>(xb + 20) + t);
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2 * t + e;
            unsigned short h = 0;
            if (col < ncols) h = ldca16((x + ((size_t) col * nb32 + k) * kQ81));
            xf.d8[k4][e] = col < ncols ? __half2float(__ushort_as_half(h)) : 0.0f;
        }
    }
}
__device__ __forceinline__ float4 ldca128(const void* p) { float4 v; asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3},[%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p)); return v; }
// ab_one with the row from global (kernel-constant: nc) and x from L1-cached loads (x is written in this kernel)
__device__ __forceinline__ float ab_one_g(const DecRecArgs& a, int head, bool is_beta, int t, int lane) {
    const float4* xt = reinterpret_cast<const float4*>(a.x + (size_t) t * a.n_embd);
    const uint4* w_g = reinterpret_cast<const uint4*>((is_beta ? a.w_beta : a.w_alpha) + (size_t) head * a.n_embd);
    float4 xa[NJ], xb[NJ];
    uint4 wv[NJ];
#pragma unroll
    for (int q = 0; q < NJ; ++q) {
        const int j = lane + 32 * q;
        xa[q] = ldca128(xt + 2 * j);
        xb[q] = ldca128(xt + 2 * j + 1);
        wv[q] = __ldg(w_g + j);
    }
    float v = 0.0f;
#pragma unroll
    for (int q = 0; q < NJ; ++q) {
        const uint4 w = wv[q];
        v = fmaf(__uint_as_float(w.x << 16), xa[q].x, v); v = fmaf(__uint_as_float(w.x & 0xffff0000u), xa[q].y, v);
        v = fmaf(__uint_as_float(w.y << 16), xa[q].z, v); v = fmaf(__uint_as_float(w.y & 0xffff0000u), xa[q].w, v);
        v = fmaf(__uint_as_float(w.z << 16), xb[q].x, v); v = fmaf(__uint_as_float(w.z & 0xffff0000u), xb[q].y, v);
        v = fmaf(__uint_as_float(w.w << 16), xb[q].z, v); v = fmaf(__uint_as_float(w.w & 0xffff0000u), xb[q].w, v);
    }
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (is_beta) return 1.0f / (1.0f + __expf(-v));
    const float u = v + a.dt[head];
    const float sp = u > 20.0f ? u : log1pf(__expf(u));
    return sp * a.ssm_a[head];
}
#endif
// dec_gdn_rec's conv_silu with its contraction spelled out (fma(v0, w0, v1*w1), then + v2*w2, + x*w3)
__device__ __forceinline__ float conv_silu_x(const float (&hist)[3], const float (&w)[4], const float (&xv)[XT], int t) {
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) { const int src = t + j; win[j] = src < 3 ? hist[src] : xv[src - 3]; }
    const float sum = fmaf(xv[t], w[3], fmaf(win[2], w[2], fmaf(win[0], w[0], win[1] * w[1])));
    return sum / (1.0f + __expf(-sum));
}
// ------------------------------------------------------------ pair (k_pair<2>: two 8-warp tile groups)
struct PSplit {
    int first, step, n, fz, nq;   // tile k: k < nq (or nq < 0): first + k step, else fz + k - nq
    __device__ __forceinline__ int tile(int k) const { return nq >= 0 && k >= nq ? fz + k - nq : first + k * step; }
};
constexpr int Q3_0 = 80;   // prt -3: CTAs [Q3_0, Q3_0 + 40) take 3 qkv tiles and compute the rec heads' alpha/beta
__device__ __forceinline__ PSplit p_split(int b, int G, int prt) {
    if (prt == -3) {   // qkv on every CTA (CTAs < 40: 3 tiles, the rest 4 = 2 rounds); z on the non-rec CTAs after their qkv
        const bool three = b >= Q3_0 && b < Q3_0 + 40;
        const int nq = three ? 3 : 4, fq = 4 * b - (b < Q3_0 ? 0 : b < Q3_0 + 40 ? b - Q3_0 : 40);
        if (b >= R_CTA0 && b < R_CTA0 + R_HV) return {fq, 1, nq, 0, nq};
        const int nn = R_CTA0, nzt = P_NT - P_NTQ, pz = nzt / nn, ez = nzt % nn;   // CTAs [Q3_0, Q3_0 + ez) take one more z
        const int xb = b < Q3_0 ? 0 : b < Q3_0 + ez ? b - Q3_0 : ez;
        return {fq, 1, nq + pz + (b >= Q3_0 && b < Q3_0 + ez ? 1 : 0), P_NTQ + b * pz + xb, nq};
    }
    if (prt == -2) {
        if (b >= R_CTA0 && b < R_CTA0 + R_HV) return {0, 1, 0, 0, 0};
        const int nn = G - R_HV, pq = P_NTQ / nn, eq = P_NTQ % nn, nzt = P_NT - P_NTQ, pz = nzt / nn, ez = nzt % nn;
        const int nq = pq + (b < eq ? 1 : 0);
        const int zb = (b - eq + nn) % nn;   // the z extras go to the CTAs with fewer qkv tiles first
        const int nz = pz + (zb < ez ? 1 : 0);
        return {b * pq + (b < eq ? b : eq), 1, nq + nz, P_NTQ + zb * pz + (zb < ez ? zb : ez), nq};
    }
    if (prt < 0) return {b, G, (b < P_NT ? (P_NT - 1 - b) / G + 1 : 0), 0, -1};
    if (b >= R_CTA0 && b < R_CTA0 + R_HV) return {P_NT - R_HV * prt + (b - R_CTA0) * prt, 1, prt, 0, -1};
    const int nn = G - R_HV, tiles = P_NT - R_HV * prt, per = tiles / nn, ex = tiles % nn;
    return {b * per + (b < ex ? b : ex), 1, per + (b < ex ? 1 : 0), 0, -1};
}

}  // namespace mkc
