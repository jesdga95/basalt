// include/basalt/kernels/hc_fused.hpp - the hyper-connection read of a verify window as ONE kernel.
//
// The same operations as fused_gr_read_multi (and, optionally, the router after it), on the matrices' 8-bit copies:
//
//   R' = R + bo_prev * 2 sigmoid(inj_prev / hc)                  (when apply)
//   rs[c] = rsqrt(mean(R'[c]^2) + eps),  xn = R' * w_norm * rs
//   lo[k] = silu((w_down[k] . xn) / hc),  inject[c] = w_inject[c] . xn
//   mixed[d] = mean_c xn[c,d] * sigmoid(w_up[c*n_embd + d] . lo),  R <- R'
//   logits = w_router . mixed,  (ids, weights) = top-10 of softmax(logits)   (when w_router)
//   gate_sh = sigmoid(w_gate_sh . mixed)                                       (when w_gate_sh)
//
// One CTA per SM; the work is a queue of items in dependency order, handed on through counters in global memory
// instead of kernel boundaries: 88 down items (the matrices' 8-bit rows over half-streams of the stream rows, each
// forming R' and x itself), one up item per remaining CTA (its slice of w_up copied into shared memory by tensor
// copies), 81 router items and the top-10 item.  A CTA claims items in order and waits only on earlier items, so the
// read finishes even when other work holds some of the SMs.  The matrices are read from their 8-bit copies, so the
// outputs are not the bf16 kernels' bits; every token's outputs are independent of the window width and its column.
//
// `pf` / `pf_bytes`: up to four ranges (e.g. the next read's weights) prefetched into L2 once this read is done.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

struct HcFusedArgs {
    int T = 0;                          ///< tokens, 1..8; token t's rows at the fixed strides below
    const float* R = nullptr;           ///< [T][hc * n_embd]
    float* R_out = nullptr;             ///< == R for the in-place update
    bool apply = false;                 ///< fold the previous half's write
    const float* bo_prev = nullptr;     ///< [T][n_embd]
    const float* inj_prev = nullptr;    ///< [T][hc]
    const float* w_norm = nullptr;      ///< [hc * n_embd] f32
    const uint16_t* w_down = nullptr;   ///< bf16 [hc_lr][hc * n_embd]
    const uint16_t* w_up = nullptr;     ///< bf16 [hc * n_embd][hc_lr]
    const uint16_t* w_inject = nullptr; ///< bf16 [hc][hc * n_embd], or null
    float eps = 1e-6f;
    float* lo = nullptr;                ///< [T][hc_lr]
    float* rs = nullptr;                ///< [T][hc]
    float* inject_out = nullptr;        ///< [T][hc] (when w_inject)
    float* mixed = nullptr;             ///< [T][n_embd]
    float* xn = nullptr;                ///< scratch [T][hc * n_embd]
    const uint16_t* w_router = nullptr; ///< bf16 [512][n_embd], or null (no router)
    float* logits = nullptr;            ///< [T][512]
    int32_t* ids = nullptr;             ///< [T][10]
    float* weights = nullptr;           ///< [T][10]
    /// mixed's q8_1 rows (36-byte blocks, [T][n_embd / 32]), or null: written by the router items, or without a
    /// router by the up slices
    void* xq = nullptr;
    /// With a router: the shared expert's gate-input row (bf16 [n_embd], 16-byte aligned), or null.  The router items
    /// dot it with mixed as a router row (the MMVF's arithmetic) and write sigmoid of the dot per token to gate_sh, as
    /// shared_expert_multi's gate kernel does.
    const uint16_t* w_gate_sh = nullptr;
    float* gate_sh = nullptr;           ///< [T]
    /// With a router: mixed's q8_1 rows as native_quantize_q8_1 writes them ([T][n_embd / 32]), or null
    void* xq_sh = nullptr;
    const void* pf[4] = {nullptr, nullptr, nullptr, nullptr};
    size_t pf_bytes[4] = {0, 0, 0, 0};
    /// let the next kernel in at the start (its blocks take the SMs this read's blocks leave)
    bool next_early = false;
    /// A read with a router and xq may also count its progress for a kernel that starts before it ends (device, two
    /// words): each router item adds one to pub_ctr[0] once its share of xq is written (hc_fused_pub_arrivals() in
    /// all), the top-10 adds one to pub_ctr[1] once the ids and weights are.  Null: no counts.
    unsigned* pub_ctr = nullptr;
    /// 8-bit copies of w_down / w_up / w_inject (int8, one f16 scale per 32 consecutive values of a row); null: found
    /// from the copies hc8_prepare / hc8_upload registered for the bf16 matrices (every read needs them).
    const int8_t* q_down = nullptr;
    const uint16_t* s_down = nullptr;   ///< f16 [hc_lr][hc * n_embd / 32]
    const int8_t* q_up = nullptr;
    const uint16_t* s_up = nullptr;     ///< f16 [hc * n_embd][16] (hc_lr / 32 used)
    const int8_t* q_inject = nullptr;
    const uint16_t* s_inject = nullptr; ///< f16 [hc][hc * n_embd / 32]
};

/// The hyper-connection reads use 8-bit copies of their matrices (every token's outputs stay independent of T and
/// column).
bool hc8_enabled();
/// Whether a read writes HcFusedArgs::xq.
bool hc_fused_q8_rows_ok();
/// The arrivals a read adds to HcFusedArgs::pub_ctr[0] (its router items').
unsigned hc_fused_pub_arrivals();
/// Quantize the bf16 matrix w [rows][cols] (cols a multiple of 32) to an 8-bit copy on the current device, once per
/// pointer, outside any stream capture.  A row's scales start every (cols / 32 rounded up to 8) f16; the copy and its
/// scales are one allocation (one prefetch range).
void hc8_prepare(const uint16_t* w, int rows, int cols);
/// Registers a copy made elsewhere (a hc8 table): `q` [rows][cols] int8 and the f16 scales [rows][srow] (the layout
/// hc8_prepare makes), uploaded to the current device.  Keyed by the bf16 matrix `w` like hc8_prepare, or, with
/// w == nullptr, by the copy itself like hc8_adopt (the key is returned).  nullptr: no VRAM, or a bad shape.
const void* hc8_upload(const uint16_t* w, const int8_t* q, const uint16_t* s, int rows, int cols);
/// The 8-bit copy of w (q, its scales, the bytes of both), or false when there is none.
bool hc8_lookup(const void* w, const int8_t** q, const uint16_t** s, size_t* bytes = nullptr);
/// Bytes held by the 8-bit copies.
size_t hc8_bytes();
/// The bf16 matrices are dropped from VRAM: the 8-bit copies are all there is.
bool hc8_only();
/// Quantize the bf16 matrix staged at w (the caller frees it afterwards) and register the copy under the copy's own
/// address, which stands for the matrix from then on: every reader resolves it with hc8_view.  nullptr: no VRAM.
const void* hc8_adopt(const uint16_t* w, int rows, int cols);
/// An 8-bit copy as its readers see it: q [rows][cols], f16 scales [rows][srow] (cols / 32 used).  `owned`: the key
/// is the copy itself (no bf16 matrix behind it).
struct Hc8View {
    const int8_t* q = nullptr;
    const uint16_t* s = nullptr;
    int rows = 0, cols = 0, srow = 0;
    size_t bytes = 0;
    bool owned = false;
};
bool hc8_view(const void* w, Hc8View& v);
/// Rows [row0, row0 + n) of w's copy as bf16 (q * scale rounded once), row-major `cols` per row, on `stream`.
void hc8_dequant_bf16(const Hc8View& v, int64_t row0, int64_t n, uint16_t* out, void* stream);

/// Allocate the read's scratch on the current device (outside a stream capture, once).
void hc_fused_init();
/// The current device runs the fused read (enough SMs for the role split); hc_fused_init ran there.
bool hc_fused_supported();
/// Enqueue the read on `stream` (a PdlScope applies).  Aborts on invalid arguments.
void hc_fused_read(const HcFusedArgs& a, void* stream);
/// Prefetch byte ranges into L2 with a one-CTA kernel (bulk prefetches; no data moves to the SMs).
void hc_l2_prefetch(const void* const* ptrs, const size_t* bytes, int n, void* stream);
/// The bench only: per-CTA phase times (globaltimer ns) into dbg[nb][16]: start, after the dependency wait, down/up
/// input ready, down/up done, router input ready, router rows done, top-10 done (last CTA), exit; [15] = 1 for a CTA
/// that ran a down item, 2 for an up item.  nullptr = off.
void hc_fused_set_debug(unsigned long long* dbg);

/// Tests only: k CTAs, one per SM, that hold their SM until *go (mapped host memory) is nonzero; each adds one to
/// *started once running.
void hc_test_hold_sms(int k, int* go, int* started, void* stream);

}  // namespace basalt::kernels
