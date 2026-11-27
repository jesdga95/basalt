// include/basalt/kernels/gqa_attn.hpp - causal grouped-query attention over an fp16 KV cache, head width 256.
//
// The cache holds one layer's cells as [cell][n_kv][256] fp16, K and V apart; query heads h and h + 1 ... share KV
// head h / (n_head / n_kv). Row t of a window or chunk sits at cell pos0 + t and sees cells 0 .. pos0 + t.  q is fp32
// [T][n_head][256], already normed and rotated; out fp32 [T][n_head][256]; scale 1/16.
//
//   gqa_decode_attn   a decode window (T <= 8): cells split in chunks over the grid, fp32 dot products, the chunks'
//                     partial softmax sums merged by a second kernel.  A row's output does not depend on T.
//   gqa_prompt_attn   a prompt chunk.  An int8 cache: blocks of 128 positions (64 for narrow launches) of one query
//                     head, a warp per 16 positions for the whole sweep; 64-cell K/V tiles as raw codes, QK^T on int8
//                     tensor cores (q rounded to int8 per 64 values), V decoded to fp16 in registers, P V on fp16
//                     tensor cores summed in fp16 per tile; with split scratch, the cells of a launch whose blocks
//                     would leave SMs idle split over more blocks and merged.  An fp16 cache: blocks of 96 rows
//                     (positions x the KV group's query heads), QK^T on fp16 tensor cores with fp32 accumulation, P V
//                     summed in fp16 per 32-cell tile.  Online softmax and the output in fp32.
#pragma once

#include "basalt/kernels/window_segs.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

constexpr int kGqaMaxWindow = 64;

/// An int8 K/V cache in place of the fp16 one: codes [cell][n_kv][256], fp16 scales [cell][n_kv][4] (one per 64 values,
/// amax / 127, codes rounded to nearest).  `k` null: the fp16 cache.
/// `fp4`: the same four buffers hold NVFP4 instead: E2M1 codes [cell][n_kv][128] bytes (value 2i in byte i's low nibble),
/// E4M3 scales [cell][n_kv][16] bytes (one per 16 values, amax / 6) in `ks` / `vs`.  K (and every query read against it)
/// is turned by the normalized 256-point Hadamard transform before it is rounded; V is stored as it is.
struct KvInt8 {
    int8_t* k = nullptr;
    int8_t* v = nullptr;
    uint16_t* ks = nullptr;
    uint16_t* vs = nullptr;
    bool fp4 = false;
};
/// Bytes of a run of `values` cache values' codes, and of their scales, in either quantized format.
constexpr size_t kv_code_bytes(size_t values, bool fp4) { return fp4 ? values / 2 : values; }
constexpr size_t kv_scale_bytes(size_t values, bool fp4) { return fp4 ? values / 16 : values / 64 * 2; }

/// A cache in two places: cells from `at` on are in these buffers instead (cell c at row c - at there, the same layout:
/// fp16 `k`, `v`, or the int8 `q8`), cells below `at` in the main ones.  A cache streamed from page-locked host memory
/// keeps its first `at` cells on the card and the rest here (device-mapped); a prompt chunk reads a VRAM copy of them.
/// Every kernel that takes one reads and writes each cell where it lives and computes exactly what it computes on one
/// buffer.  The default: one buffer.
struct KvFar {
    int64_t at = INT64_MAX;
    uint16_t* k = nullptr;
    uint16_t* v = nullptr;
    KvInt8 q8 = {};
};

/// Scratch floats gqa_decode_attn needs for up to `max_cells` cells and `T` rows (gqa_window_attn: cells its window
/// spans plus the window's rows).
size_t gqa_decode_scratch_floats(int64_t max_cells, int T, int n_head, int head_dim = 256);
/// `pos_dev` (optional): pos0 read from the device instead, the grid sized for `max_cells` (a captured window).
void gqa_decode_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int T, float* scratch, int64_t max_cells, float* out, void* stream,
                     const int32_t* pos_dev = nullptr, KvInt8 q8 = {}, const float* gate_q = nullptr,
                     uint16_t* gated16 = nullptr, KvFar far_kv = {});
/// (gate_q, gated16: also each head's output times sigmoid of its gate, q_full [T][n_head][512]'s second half, as bf16
/// rows [T][n_head * 256] into gated16 (bitwise attn_gate, then rounded); `out` may then be null.)
/// One window of each of several sequences (a batch window's segments) in one launch: segment k's rows follow segment
/// k - 1's in q, out, gate_q and gated16, with its own cache and position; every row is what gqa_decode_attn of its
/// segment alone gives.  At most kGqaMaxWindow rows in all.
struct GqaSeg {
    int T = 0;
    const uint16_t* k_cache = nullptr;
    const uint16_t* v_cache = nullptr;
    int64_t pos0 = 0;
    const int32_t* pos_dev = nullptr;
    KvInt8 q8 = {};
    KvFar far_kv = {};
};
void gqa_decode_attn_segs(const float* q, int n_head, int n_kv, const GqaSeg* segs, int n_seg, float* scratch,
                          int64_t max_cells, float* out, void* stream, const float* gate_q = nullptr,
                          uint16_t* gated16 = nullptr);
/// gqa_decode_attn for a block drafter's rows: head width 128 or 256, row t at position pos0 + t sees positions
/// [pos0 + t - window + 1, hi) (not causal), position p in cache cell p % cap. `block`: every row sees the first row's
/// window, [pos0 - window + 1, hi) (a drafter whose block shares its anchor's window). `q8`: the int8 cache instead (its
/// codes decoded to the fp16 the fp16 path reads; not NVFP4).
void gqa_window_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv, int head_dim,
                     int64_t pos0, int T, int64_t hi, int64_t window, int64_t cap, float* scratch, int64_t scratch_cells,
                     float* out, void* stream, bool block = false, KvInt8 q8 = {});
struct GqaWindowSeg {
    const uint16_t* k_cache = nullptr;
    const uint16_t* v_cache = nullptr;
    int64_t pos0 = 0;
    int T = 0;
    int64_t hi = 0;
    int64_t window = 0;
    int64_t cap = 0;
    bool block = false;
    KvInt8 q8 = {};   ///< the int8 cache instead: every segment of a call int8 or none
};
/// gqa_window_attn for up to kWindowSegsMax segments of any rows in one launch (segment k's rows after segment k - 1's in
/// q and out), each row what its segment alone gives; `scratch` holds gqa_decode_scratch_floats(scratch_cells, the rows of
/// every segment, n_head, head_dim).  `wide` (a drafter, whose rows need not match a lone segment's bits): a launch of
/// many segments splits each segment's keys less, so the splits depend on the segment count.
void gqa_window_attn_segs(const float* q, int n_head, int n_kv, int head_dim, const GqaWindowSeg* segs, int n_seg,
                          float* scratch, int64_t scratch_cells, float* out, void* stream, bool wide = false);
/// The device whose SM count plans the int8 prompt attention's launch (its key splits, so its rounding); -1 (the default):
/// the current one.  A card that reads some of the main card's layers plans as the main card, so its rows are the main card's.
void gqa_prompt_plan_device(int device);
/// The first `cells` cells of an NVFP4 cache `f4` (every KV head, K and V) into the int8 buffers `out`: the decoded
/// values rounded by the int8 cache's rule.  A prompt chunk reads them through the int8 kernel.
void kv4_to_int8(const KvInt8& f4, const KvInt8& out, int64_t cells, int n_kv, void* stream);
/// Scratch floats gqa_prompt_attn needs to split a T-row chunk's cells `splits` ways (an int8 cache).
size_t gqa_prompt_split_floats(int64_t T, int n_head, int splits);
/// `split_scratch` (optional, `split_floats` floats): lets an int8-cache launch split its cells over more blocks.
void gqa_prompt_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int64_t T, float* out, void* stream, KvInt8 q8 = {}, float* split_scratch = nullptr,
                     size_t split_floats = 0, KvFar far_kv = {});
/// The prompt chunk's form on an int8 cache: BF16 queries and BF16 output rows (the same arithmetic; the output rounded once).
void gqa_prompt_attn_bf16(const uint16_t* q, int n_head, int n_kv, int64_t pos0, int64_t T, uint16_t* out, void* stream,
                          KvInt8 q8, float* split_scratch = nullptr, size_t split_floats = 0, KvFar far_kv = {});

}  // namespace basalt::kernels
