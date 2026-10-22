// include/basalt/prefill/moe_fp4.hpp - the prompt path's NVFP4 experts on the FP4 block-scaled tensor cores.
// Weights stay in the pack's NVFP4 blocks (read in place from each expert's blob); activations are rounded to FP4
// codes with a UE4M3 scale per 16 values and one FP32 scale per token (W4A4).  A layer's input is rounded once per
// token (token planes: the gate/up product reads each row's token); H is laid out per 64-token tile of one expert
// (each expert's rows padded to whole tiles), so a tile's codes and scales are one contiguous, bank-swizzled run.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace basalt::prefill::fp4 {

constexpr int kTileTokens = 64;
constexpr int kMaxExperts = 16;   // experts per launch

/// Bytes of the activation planes for `rows` rows of `cols` values (multiple of 128) split over up to `experts`
/// segments, each padded to whole tiles.
size_t a4_bytes(int64_t rows, int64_t cols, int experts);
/// Floats of the per-token scale array for the same split.
size_t scale_count(int64_t rows, int experts);

/// tile_off[0..n]: the first tile of each of the n segments of `bounds` (n + 1 row bounds on the device) and the
/// total.  One tiny launch.
void plan(const int32_t* bounds, int n, int32_t* tile_off, void* stream);

/// out[i] = bounds[i] - bounds[0] for i in 0..n: a run of segments' own row bounds.
void rebase(const int32_t* bounds, int n, int32_t* out, void* stream);

/// Rounds rows to FP4: row i of the segments is row ids[i] of x (or i), `ld` floats per row.  `bounds`/`tile_off` as
/// planned; writes the planes `a4` and the token scales `gs`.
void quantize(const float* x, const int32_t* ids, const int32_t* bounds, const int32_t* tile_off, int n, void* a4,
              float* gs, int64_t cols, int64_t ld, int64_t rows, void* stream);

/// Bytes of the token planes of `tokens` rows of `cols` values: the codes [tokens][cols / 2] (each 16 values' 8 bytes
/// in order), then the UE4M3 scales [tokens][cols / 16].
size_t token_bytes(int64_t tokens, int64_t cols);
/// Rounds each of `tokens` rows of x (`ld` floats apart) to FP4 once, into the token planes `a4` and the token scales
/// gs[tokens]: the same codes and scales quantize() gives every row that reads that token.
void quantize_tokens(const float* x, int64_t tokens, int64_t cols, int64_t ld, void* a4, float* gs, void* stream);

/// n experts' products: expert j's matrix [w_rows, w_cols] of NVFP4 blocks starts at blob[j] + w_off; its rows
/// [bounds[j], bounds[j+1]) of the planes go to dst rows of the same indices (dst + row * ld_dst).  The result is the
/// plain product (no per-expert tail scale).  `max_rows`: the most rows one segment has.
/// `swiglu` (the gate/up product: gate rows, then as many up rows): row i reads token ids[i] of token planes over
/// `tokens` rows (a4, gs from quantize_tokens; no tile_off), and dst gets [w_rows / 2] values per row, the tail-scaled
/// SwiGLU silu(g s_gate) (u s_up) s_down of the blob's 16-byte tail at tail_off, the same bits as the plain product
/// followed by mmq::swiglu.
struct Product {
    const uint8_t* blob[kMaxExperts] = {};
    size_t w_off = 0;
    int64_t w_rows = 0, w_cols = 0;
    int n = 0;
    const void* a4 = nullptr;
    const float* gs = nullptr;
    const int32_t* bounds = nullptr;
    const int32_t* tile_off = nullptr;
    float* dst = nullptr;
    int64_t ld_dst = 0, max_rows = 0;
    bool swiglu = false;
    const int32_t* ids = nullptr;
    int64_t tokens = 0;
    size_t tail_off = 0;
};
void run(const Product& p, void* stream);
/// The 16-byte scale tail at `tail_off` of each of p's n blobs to dst[4 j .. 4 j + 3] (the tail swiglu reads).
void gather_tails(const Product& p, size_t tail_off, float* dst, void* stream);

/// A group of up to kMaxExperts consecutive experts of one layer, as the prompt path runs them: the layer's token
/// planes (already rounded to FP4) and, per run of the group, the gate/up product as the tail-scaled SwiGLU, the FP4
/// rounding of H and the down product.  `h` holds one run's rows only (its row 0 is the run's first row); `dm` is the
/// layer's own output, indexed by absolute row.
struct Group {
    int n = 0;
    const uint8_t* blob[kMaxExperts] = {};
    const int32_t* bounds_host = nullptr;   ///< n + 1 absolute row bounds, on the host
    const int32_t* bounds_dev = nullptr;    ///< the same bounds on the device
    const int32_t* ids = nullptr;           ///< the layer's token of each absolute row
    const void* a4x = nullptr;              ///< the layer's token planes and token scales (quantize_tokens)
    const float* gsx = nullptr;
    int64_t tokens = 0;
    size_t down_off = 0, tail_off = 0;      ///< the down matrix and the 16-byte scale tail inside a blob
    float *h = nullptr, *dm = nullptr;
    void* a4h = nullptr;                    ///< a run's H planes, token scales, tile table and relative bounds
    float* gsh = nullptr;
    int32_t *toh = nullptr, *rel = nullptr; ///< kMaxExperts + 1 entries each
};
/// Runs consecutive experts [i0, i0 + n) of the group (n >= 1): rows of `bounds_host` rows [i0, i0 + n].
void run_experts(const Group& g, int i0, int n, void* stream);
/// How many experts from i0 form the next run: as many consecutive ones as hold at most `row_cap` rows, and at least
/// one (an expert's rows are never split).
int run_length(const Group& g, int i0, int64_t row_cap);
/// The whole group as runs of at most `row_cap` rows each (H holds a run's rows); returns the number of runs.
int run_group(const Group& g, int64_t row_cap, void* stream);

/// The rows of n consecutive experts ([bounds[j], bounds[j + 1]), on the host) in pieces for buffers of `cap` rows
/// (cap >= kTileTokens): an expert above the cap is cut at whole tiles from its first row, so each piece's tiles hold
/// the rows the expert's own tiles hold.  The pieces form groups of up to kMaxExperts pieces and at most `cap` rows.
struct Pieces {
    std::vector<int32_t> expert;   ///< per piece: its expert's index j
    std::vector<int32_t> bounds;   ///< per piece its first row, then the last piece's end
    std::vector<int32_t> group;    ///< per group its first piece, then the piece count
};
void cut_pieces(const int32_t* bounds, int n, int64_t cap, Pieces& out);

}  // namespace basalt::prefill::fp4
