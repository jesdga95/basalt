// src/program/serve/prefix_anchor.hpp - a held conversation met against a prompt (header-only, CPU).
//
// A conversation a session holds is the prompt it read, then every token the model wrote. The model can write a text
// in other pieces than the tokenizer cuts it into (two ids where the tokenizer makes one); a client that sends the
// reply back unchanged is tokenized canonically, so the prompt stops matching the held ids inside the reply although
// the bytes agree. `meet_held` finds that case: the held ids past the common prefix spell exactly the bytes of the
// prompt's next tokens and end on a prompt token boundary, so the held ids stand for those tokens.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace basalt::program::serve {

/// How a held sequence meets a prompt.
struct HeldMeet {
    int64_t held = 0;     ///< the tokens of the held sequence the prompt continues; 0: it does not
    int64_t resume = 0;   ///< where the prompt's own tokens go on after them
    bool exact = false;   ///< the held ids are the prompt's first tokens as they are
    bool respelled() const { return held > 0 && !exact; }
};

/// The held ids past the common prefix that may be respelled: a reply's worth.
constexpr int64_t kRespellMax = 1 << 17;

/// `bytes(id)` is the raw bytes a token stands for (empty when it has none). An exact prefix meets a prompt only when it is
/// shorter than it: the prompt's last token always starts the first window. With `respell` false only an exact
/// prefix meets the prompt.
template <class Bytes>
HeldMeet meet_held(const int32_t* prompt, int64_t n, const int32_t* held, int64_t h, bool respell, Bytes bytes) {
    if (h < 1 || n < 1) return {};
    const int64_t k = std::mismatch(held, held + std::min(h, n), prompt).first - held;
    if (k == h) return h < n ? HeldMeet{h, h, true} : HeldMeet{};
    if (!respell || k == 0 || h - k > kRespellMax) return {};
    std::string want;
    for (int64_t i = k; i < h; ++i) {
        const std::string_view b = bytes(held[i]);
        if (b.empty()) return {};
        want.append(b);
    }
    size_t got = 0;
    int64_t j = k;
    while (got < want.size() && j < n) {
        const std::string_view piece = bytes(prompt[j]);
        if (piece.empty() || piece.size() > want.size() - got || want.compare(got, piece.size(), piece) != 0) return {};
        got += piece.size();
        ++j;
    }
    if (got != want.size() || j >= n) return {};
    return {h, j, false};
}

/// Whether a prompt that grows from `n` to `grown` tokens still leaves `want` tokens of answer in a context of `ctx`
/// (the engine keeps 8 free). A prompt that does not grow always fits: the request was admitted with it.
inline bool respell_fits(int64_t grown, int64_t n, int64_t want, int64_t ctx) { return grown <= n || grown + want + 8 <= ctx; }

/// The ids two sequences share from the start.
inline int64_t shared_start(const int32_t* a, int64_t an, const int32_t* b, int64_t bn) {
    const int64_t lim = std::min(an, bn);
    return lim > 0 ? std::mismatch(a, a + lim, b).first - a : 0;
}

/// Whether the held ids from `k` on spell the bytes of the prompt's tokens from `k` on: the model wrote the start of
/// the prompt in other pieces than the tokenizer cuts it into (a reply the client sends back), so the ids part where
/// the bytes agree.  The prompt may end inside the held ids' last token, or run on past them into the next message.
template <class Bytes>
bool spells_the_prompt(const int32_t* prompt, int64_t n, const int32_t* held, int64_t h, int64_t k, Bytes bytes) {
    int64_t hi = k;
    size_t hoff = 0;
    for (int64_t j = k; j < n; ++j) {
        const std::string_view pb = bytes(prompt[j]);
        if (pb.empty()) return false;
        size_t po = 0;
        while (po < pb.size()) {
            if (hi == h) return true;                       // the held ids end inside this prompt token
            const std::string_view hb = bytes(held[hi]);
            if (hb.empty()) return false;
            const size_t take = std::min(hb.size() - hoff, pb.size() - po);
            if (hb.compare(hoff, take, pb.data() + po, take) != 0) return false;
            po += take;
            hoff += take;
            if (hoff == hb.size()) {
                hoff = 0;
                ++hi;
            }
        }
    }
    return true;                                            // the prompt ends inside (or at) the held ids: they spell it
}

/// The prompt tokens a slot's held conversation continues: the ids the two share from the start when the slot's
/// conversation goes on as this prompt (its reply the client sent back, cut into other pieces - `meet_held`, or the
/// bytes of its tail spelling the prompt's; `shared` is the ids the two share), else 0 - two conversations that share
/// a system prompt continue nothing of each other.  Never the prompt's last token, which starts the first window.
template <class Bytes>
int64_t slot_cover(const int32_t* prompt, int64_t n, const int32_t* held, int64_t h, int64_t shared, bool respell,
                   Bytes bytes) {
    if (n <= 0 || h <= 0) return 0;
    const HeldMeet m = meet_held(prompt, n, held, h, respell, bytes);
    if (m.held > 0) return std::min(m.held, n - 1);
    if (shared >= n) return n - 1;                          // the prompt is the start of the slot's conversation
    if (respell && shared > 0 && spells_the_prompt(prompt, n, held, h, shared, bytes)) return std::min(shared, n - 1);
    return 0;
}

/// The prompt with the held ids standing for the tokens they spell: the held sequence, then the prompt from `m.resume`.
template <class Out>
void splice_held(const int32_t* held, const HeldMeet& m, const int32_t* prompt, int64_t n, std::vector<Out>& out) {
    std::vector<Out> next;
    next.reserve((size_t) (m.held + n - m.resume));
    next.insert(next.end(), held, held + m.held);
    next.insert(next.end(), prompt + m.resume, prompt + n);
    out.swap(next);
}

}  // namespace basalt::program::serve
