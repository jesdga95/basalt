// include/basalt/spec/suffix_drafter.hpp - the suffix-lookup drafter (no weights, no GPU).
//
// Proposes the tokens that followed the longest earlier occurrence of the sequence's current suffix: when the
// output quotes its input (edits, refactors, repeated code) the continuation is usually exact, and a verify pass
// accepts many tokens at once. It needs no model and costs microseconds.
//
// Index: every trigram of the history (prompt + accepted output) maps to its WAYS most recent end positions in an
// open-addressing table, and so does every LONG_ORDER-gram in a second one: a common trigram's few slots
// fill with recent repeats, while the longer key still finds an old source of the text being quoted. The tables double as
// the history grows, up to 128 to 256 bytes per token of capacity (8 more per token used); appends are amortized O(1).
// A proposal checks the candidates of both, extends each match backwards up to `max_match`, and takes the longest
// (most recent on ties). Matches shorter than `min_match` propose nothing.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace basalt::spec {

class SuffixDrafter {
public:
    static constexpr int WAYS = 8;
    static constexpr int LONG_ORDER = 8;

    explicit SuffixDrafter(int min_match = 3, int max_match = 32, size_t capacity_tokens = 1u << 19);

    void reset();
    /// Synchronize the history to `tokens[0..n)`: extends incrementally if the current history is a prefix, else resets.
    void sync(const int32_t* tokens, size_t n);
    /// Add tokens to the history (the prompt, then every accepted token).
    void append(const int32_t* tokens, size_t n);
    void append(int32_t token) { append(&token, 1); }

    /// Write up to `max_k` proposed next tokens to `out`; returns how many (0 = no match of at least min_match).
    int propose(int max_k, int32_t* out);
    /// Propose next tokens as if `extra[0..extra_n)` were appended to the history, without mutating the index.
    int propose_after(const int32_t* extra, size_t extra_n, int max_k, int32_t* out, int* match_out = nullptr) const;

    /// Length of the match behind the last proposal (0 if none).
    int last_match() const { return last_match_; }
    size_t size() const { return hist_.size(); }

private:
    struct Slot {
        uint64_t key = 0;             // trigram hash + 1 (0 = empty)
        uint32_t pos[WAYS] = {};      // end positions, most recent first
        uint8_t n = 0;
    };
    Slot* find_slot(std::vector<Slot>& table, uint64_t key, bool insert);
    const Slot* find_slot_ro(const std::vector<Slot>& table, uint64_t key) const;
    static void push(Slot* s, size_t end);
    uint64_t key_at(size_t end) const;
    uint64_t long_key_at(size_t end) const;
    void grow();   // both tables twice as wide, the keys in again

    int min_match_, max_match_;
    std::vector<int32_t> hist_;
    static constexpr uint32_t kLongBit = 1u << 31;
    std::vector<Slot> table_, long_;
    std::vector<uint32_t> dirty_;   // the slots in use (long table flagged), so a reset clears only those
    size_t mask_ = 0;
    static constexpr size_t kFirstCap = 1u << 13;   // the tables start this wide and double with the history
    size_t max_cap_ = 1;
    size_t used_[2] = {0, 0};   // the slots in use in table_ and long_
    int last_match_ = 0;
};

}  // namespace basalt::spec
