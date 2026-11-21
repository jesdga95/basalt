// src/program/grammar/guard.cpp - the pack's vocabulary and the per-request window check.
#include "basalt/program/grammar/guard.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>

#include "basalt/program/grammar/schema.hpp"
#include "basalt/program/json_scan.hpp"
#include "basalt/log.hpp"

namespace basalt::program::grammar {

namespace {

using json::next_cp;
using json::put_utf8;
using json::Scan;

// the byte-level BPE spelling: a printable byte stands for itself, the others for U+0100 upward in byte order
const std::unordered_map<uint32_t, uint8_t>& char_to_byte() {
    static const std::unordered_map<uint32_t, uint8_t> table = [] {
        std::unordered_map<uint32_t, uint8_t> t;
        uint32_t next = 256;
        for (int b = 0; b < 256; ++b) {
            const bool plain = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            t[plain ? (uint32_t) b : next++] = (uint8_t) b;
        }
        return t;
    }();
    return table;
}

}  // namespace

bool vocab_from_pack(std::string_view vocab_json, std::string_view types_json, size_t n_vocab,
                     const std::vector<int32_t>& end_ids, PackVocab& out, std::string& err) {
    std::vector<int8_t> types(n_vocab, -1);
    {
        Scan sc{types_json};
        if (!sc.eat('[')) { err = "token_type.json: not an array"; return false; }
        size_t id = 0;
        if (!sc.eat(']')) {
            do {
                int64_t t = 0;
                if (!sc.integer(t)) { err = "token_type.json: bad entry"; return false; }
                if (id < n_vocab) types[id] = (int8_t) t;
                ++id;
            } while (sc.eat(','));
        }
    }
    std::vector<std::string> bytes(n_vocab);
    out.think_open = out.think_close = -1;
    const auto& to_byte = char_to_byte();
    Scan sc{vocab_json};
    if (!sc.eat('{')) { err = "vocab.json: not an object"; return false; }
    std::string key;
    if (!sc.eat('}')) {
        do {
            int64_t id = 0;
            if (!sc.str(key) || !sc.eat(':') || !sc.integer(id)) { err = "vocab.json: bad entry"; return false; }
            if (id < 0 || (size_t) id >= n_vocab) continue;
            const int ty = types[(size_t) id];
            std::string& b = bytes[(size_t) id];
            if (ty == 3 || ty == 5 || ty == 6) continue;   // control, unused and byte entries never appear as text
            if (ty == 4) {
                b = key;   // an added token keeps its literal text
                if (key == "<think>") out.think_open = (int32_t) id;
                else if (key == "</think>") out.think_close = (int32_t) id;
                continue;
            }
            for (size_t i = 0; i < key.size();) {
                const uint32_t cp = next_cp(key, i);
                const auto it = to_byte.find(cp);
                if (it != to_byte.end()) b += (char) it->second;
                else put_utf8(b, cp);
            }
        } while (sc.eat(','));
    }
    if (!sc.eat('}')) { err = "vocab.json: unterminated"; return false; }
    out.vocab = std::make_shared<const Vocab>(std::move(bytes), end_ids);
    return true;
}

bool Guard::begin(const std::string& grammar_json, const int64_t* prompt, size_t n_prompt, std::string& err) {
    on_ = false;
    windows_ = misses_ = fallbacks_ = ns_ = 0;
    RequestGrammar rg = grammar_from_request(grammar_json, err);
    if (!err.empty()) return false;
    if (!rg.grammar) return true;
    st_ = GrammarState(rg.grammar, rg.triggers);
    // generation starts inside a thinking block when the prompt's last think tag opens one
    thinking_ = false;
    for (size_t i = n_prompt; i-- > 0;) {
        const int64_t t = prompt[i];
        if (t == pv_.think_close) break;
        if (t == pv_.think_open) { thinking_ = true; break; }
    }
    on_ = true;
    return true;
}

bool Guard::feed(int32_t token) {
    if (thinking_) {
        if (token == pv_.think_close) thinking_ = false;
        return true;
    }
    const auto& ends = pv_.vocab->end_ids();
    if (std::find(ends.begin(), ends.end(), token) != ends.end()) return !st_.active() || st_.allows_end();
    if (token < 0 || (size_t) token >= pv_.vocab->size()) return !st_.active();
    const std::string& b = pv_.vocab->bytes(token);
    if (b.empty()) return !st_.active();   // a control token is not text
    return st_.accept(b, token);
}

int Guard::scan(const int32_t* out, int rows, int first) {
    const auto t0 = std::chrono::steady_clock::now();
    int miss = -1;
    for (int t = first; t < rows; ++t)
        if (!feed(out[t])) { miss = t; break; }
    ++windows_;
    if (miss >= 0) ++misses_;
    ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    return miss;
}

void Guard::mask_bias(const std::vector<float>& user_bias, bool use_cache, std::vector<float>& bias) {
    st_.fill_mask(*pv_.vocab, mask_, use_cache);
    const size_t n = mask_.size();
    bias.resize(n);
    const bool u = user_bias.size() == n;
    for (size_t i = 0; i < n; ++i) bias[i] = mask_[i] ? (u ? user_bias[i] : 0.0f) : -INFINITY;
}

std::string Guard::describe() const {
    char buf[160];
    std::snprintf(buf, sizeof buf, "%zu live stacks, %s, %s", st_.stack_elements().size(),
                  st_.accepting() ? "complete" : "incomplete", st_.dead() ? "dead end" : "open");
    return buf;
}

Redraw Guard::redraw(const std::vector<float>& user_bias, const DrawFn& draw, const LogitsFn& logits, int32_t& token) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto done = [&](Redraw r) {
        ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        return r;
    };
    const auto allowed = [](const std::vector<float>& b) {
        return std::any_of(b.begin(), b.end(), [](float v) { return v > -INFINITY; });
    };
    const auto dead = [&] {
        basalt::log::error("basalt grammar", "no token is allowed here (%s); the reply ends", describe().c_str());
        on_ = false;
        return done(Redraw::dead);
    };
    std::vector<float> bias;
    mask_bias(user_bias, true, bias);
    if (!allowed(bias)) return dead();
    int32_t t = 0;
    if (!draw(bias, t)) return done(Redraw::failed);
    if (feed(t)) { token = t; return done(Redraw::drawn); }
    basalt::log::warn("basalt grammar", "a masked draw was refused by the grammar (token %d, %s); drawing again", (int) t,
                      describe().c_str());
    mask_bias(user_bias, false, bias);
    if (!allowed(bias)) return dead();
    if (!draw(bias, t)) return done(Redraw::failed);
    if (feed(t)) { token = t; return done(Redraw::drawn); }
    // still refused: the likeliest token the grammar accepts, tried best first (a few, in case the mask and accept differ)
    std::vector<float> lg;
    if (!logits(lg) || lg.size() != bias.size()) return done(Redraw::failed);
    std::vector<uint32_t> cand;
    for (size_t i = 0; i < bias.size(); ++i) {
        lg[i] += bias[i];
        if (lg[i] > -INFINITY && lg[i] < INFINITY) cand.push_back((uint32_t) i);   // not -inf, not nan
    }
    const size_t k = std::min<size_t>(cand.size(), 8);
    std::partial_sort(cand.begin(), cand.begin() + (std::ptrdiff_t) k, cand.end(),
                      [&](uint32_t a, uint32_t b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    for (size_t i = 0; i < k; ++i) {
        if (!feed((int32_t) cand[i])) continue;
        token = (int32_t) cand[i];
        ++fallbacks_;
        basalt::log::warn("basalt grammar", "the draw was refused again; token %d is the likeliest the grammar accepts", (int) token);
        return done(Redraw::fallback);
    }
    return dead();
}

}  // namespace basalt::program::grammar
