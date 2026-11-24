// include/basalt/prefill/chunk_plan.hpp - the chunks a prompt part is read in: on one card, and what a split read across the
// cards can hand back for the draft layer.  Pure arithmetic, shared by the prompt paths and their tests.
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace basalt::prefill {

/// The rows of the last card's output a split read can hold for the draft layer before the kept last chunk: the last
/// two chunks handed on stay in the two hand-off buffers they came in, and earlier ones go past `chunk` rows in either
/// buffer of `hand_rows` rows, which no chunk writes.  `chunks`: the part's chunk sizes, the kept last one included.
inline int64_t split_draft_room(const std::vector<int64_t>& chunks, int64_t chunk, int64_t hand_rows) {
    int64_t room = 2 * std::max<int64_t>(0, hand_rows - chunk);
    for (size_t i = chunks.size() >= 3 ? chunks.size() - 3 : 0; i + 1 < chunks.size(); ++i) room += chunks[i];
    return room;
}

/// `tokens` up on the 256-token grid, at most `cap`.
inline int64_t grid_chunk(int64_t tokens, int64_t cap) {
    if (tokens <= 0 || cap <= 0) return 0;
    return std::min(cap, (tokens + 255) / 256 * 256);
}

/// How one card reads a part of `tokens` in chunks of at most `cap`: equal chunks on the 256-token grid for the fewest
/// chunks, and with `head_ok` a long part starts with a chunk of `head_chunk`.  The chunk, and the head chunk (0: none;
/// never one the chunk does not exceed, as the prompt path reads no first chunk longer than the rest).
inline std::pair<int64_t, int64_t> one_card_chunks(int64_t tokens, int64_t cap, bool head_ok, int64_t head_chunk) {
    int64_t want = grid_chunk(tokens, cap);
    if (want > 0 && tokens > cap) {
        const int64_t parts = (tokens + cap - 1) / cap;
        want = grid_chunk((tokens + parts - 1) / parts, cap);
    }
    int64_t head = 0;
    if (head_ok && tokens > cap && tokens >= 2 * head_chunk && cap > head_chunk) {
        const int64_t rem = tokens - head_chunk, parts = (rem + cap - 1) / cap;
        want = grid_chunk((rem + parts - 1) / parts, cap);
        head = head_chunk;
    }
    if (head >= want) head = 0;
    return {want, head};
}

/// The sizes of the chunks the prompt path reads a part of `tokens` in, for a chunk and head chunk: a first chunk of
/// min(head, chunk) when the part is longer than one chunk, then chunks of `chunk`, the last one what is left.
inline std::vector<int64_t> chunk_list(int64_t tokens, int64_t chunk, int64_t head) {
    std::vector<int64_t> v;
    const int64_t first = tokens > chunk && head > 0 ? std::min(head, chunk) : chunk;
    for (int64_t at = 0; chunk > 0 && at < tokens;) {
        const int64_t T = std::min(at == 0 ? first : chunk, tokens - at);
        v.push_back(T);
        at += T;
    }
    return v;
}

}  // namespace basalt::prefill
