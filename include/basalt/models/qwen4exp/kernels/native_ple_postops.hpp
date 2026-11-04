#pragma once
#include "basalt/kernels/ple.hpp"

namespace basalt::kernels {

struct SegRow;

// Fixed qwen4exp single-token geometry: embd=2560, streams=4, history=9,
// convolution kernel=4/dilation=3. All buffers are caller-owned and capture-safe.
struct NativePlePostopsBuffers {
    float* key;        // 10240, normalized projected key
    float* query;      // 10240, temporary normalized hidden; may equal normalized
    float* gate;       // 4
    float* gated;      // 10240
    float* normalized; // 10240, append this to history AFTER this function
    float* conv;       // 10240
    float* result;     // 10240, may equal hidden exactly
};

// Matches pinned llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d CUDA
// build_ple AFTER the two projections. Only norm_key/query/conv and conv1d_f16
// fields of weights are used. Gamma is full [2560,4]; history is row-fastest
// [9,10240], and F16 taps are [4,10240]. Inputs must be finite.
// Require an explicit stream, nonnull aligned pointers and disjoint writable
// spans, with only the two exact aliases documented above allowed. No allocation
// or synchronization; retain every buffer through any captured graph execution.
void native_ple_postops(const float* projected_key, const float* hidden,
                        const float* value, const float* history,
                        const PleWeights& weights, const NativePlePostopsBuffers& buffers,
                        void* stream);

/// The prompt path: T tokens at once, element for element the arithmetic of T calls of native_ple_postops with
/// the history advanced after each.  `key` (T x 10240) holds the projected keys and is normalized in place;
/// `hidden` (T x 10240) becomes the result; `value` T x 2560; `query_norm` (T x 10240, the normalized hidden, then
/// the normalized gated values), `gated` (T x 10240) and `gate` (T x 4) are scratch; `history` (9 rows per
/// channel) is advanced past the T tokens.
void native_ple_postops_batch(float* key, float* hidden, const float* value, float* history, const PleWeights& w,
                              float* query_norm, float* gated, float* gate, int T, void* stream);

/// A verify window's PLE after its projections, T rows in two launches: per row, element for element, native_gr_post's
/// pending write into the hidden rows, native_ple_postops, the history advance and a copy of the advanced history.
struct PleWindowArgs {
    float* R = nullptr;                 ///< [T][10240] hidden rows: the pending write, then the result
    const float* block_out = nullptr;   ///< [T][2560] the pending write's block output
    const float* inject = nullptr;      ///< [T][4] its inject
    const float* key = nullptr;         ///< [T][10240] projected keys
    const float* value = nullptr;       ///< [T][2560] projected values
    float* history = nullptr;           ///< 9 rows per channel, advanced past the T rows ([T][9 * 10240] if segs)
    float* hist_snap = nullptr;         ///< [T][9 * 10240]: the history after each row
    /// The rows are segments of different sequences (device table, one entry per row): a segment's first row holds its
    /// history, which its rows advance in order, as a window of that segment alone would.
    const SegRow* segs = nullptr;
    float* scratch = nullptr;           ///< 2 * T * 10240 floats
    int T = 0;
};
void native_ple_window(const PleWindowArgs& a, const PleWeights& w, void* stream);

} // namespace basalt::kernels
