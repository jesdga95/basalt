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
