// include/basalt/kernels/window_handoff.hpp - the handoff between a decode window's host and its card: a plan that
// reaches the device in a kernel's parameters, a weighted sum over row pointers (each the card's or the CPU's mapped row),
// a wait for a word the host writes once the CPU's rows landed, and the next weights read into L2 while the card waits.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

/// Most words put_words carries: a decode window's expert plan and its row pointers (the kernel's parameters stay under 4 KB).
constexpr int kPutWords = 960;
/// dst[0..n_words) = src[0..n_words) for `src` in host memory, n_words <= kPutWords: the words travel in the kernel's
/// parameters, so a plan reaches the device in stream order without a copy-engine job (which would queue behind the tier's
/// copies) and the host may reuse `src` at once.
void put_words(const void* src, void* dst, int n_words, void* stream);

/// out[t][i] = sum_k w[t * K + k] * rows[t * K + k][i] for k < K, in that order (rows: device array of T * K row pointers, each
/// device or mapped host memory).
void weighted_sum_window(const float* const* rows, const float* w, int n_tok, int K, int n, float* out, void* stream);

/// Holds the stream until *word (mapped host memory the host writes) equals `value`, fences, then stores `value` in mark[0]
/// (device memory), which an l2_read_ahead on another stream stops at.
void wait_word_mark(const uint32_t* word, uint32_t value, uint32_t* mark, void* stream);
/// Byte ranges of device memory (16-byte aligned, 16-byte multiples) in the order the kernels after a wait read them; empty
/// ones last.
constexpr int kL2Ranges = 4;
struct L2Ranges {
    const void* p[kL2Ranges] = {};
    uint64_t n[kL2Ranges] = {};
};
/// Requests the ranges into L2, in order, until mark[0] equals `value` (mark: 2 words of device memory; mark[1] is scratch
/// the kernel never writes in practice). Reads only: nothing queued after it waits for it.
void l2_read_ahead(const L2Ranges& r, uint32_t* mark, uint32_t value, void* stream);

}  // namespace basalt::kernels
