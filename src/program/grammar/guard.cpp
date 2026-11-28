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

ReplyForm g_reply_form;

// True when a control token's literal text appears in the active tool-call form (so the grammar trie and lazy trigger
// can match it by bytes). Single-character tokens are excluded so punctuation cannot collide.
bool in_tool_form(const ToolForm& tf, const std::string& key) {
    if (key.size() < 2) return false;
    for (const std::string* s : {&tf.call_open, &tf.call_close, &tf.fn_open, &tf.fn_mid, &tf.fn_close,
                                 &tf.param_open, &tf.param_mid_text, &tf.param_mid_value, &tf.param_close,
                                 &tf.text_end, &tf.trigger}) {
        if (!s->empty() && s->find(key) != std::string::npos) return true;
    }
    return false;
}

}  // namespace

void set_reply_form(const ReplyForm& form) { g_reply_form = form; }
const ReplyForm& reply_form() { return g_reply_form; }

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
    std::vector<uint8_t> is_end_id(n_vocab, 0);
    for (int32_t e : end_ids) {
        if (e >= 0 && (size_t) e < n_vocab) is_end_id[(size_t) e] = 1;
    }
    std::vector<std::string> bytes(n_vocab);
    out.think_open = out.think_close = out.block_header = out.answer_open = -1;
    const ReplyForm& form = reply_form();
    const ToolForm& tform = tool_form();
    const bool blocks = !form.header.empty();
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
            if (ty == 3 || ty == 4) {   // thinking and block markers may be control or user-defined added tokens
                if (blocks && key == form.header) out.block_header = (int32_t) id;
                else if (key == form.think_open) out.think_open = (int32_t) id;
                else if (key == form.think_close) out.think_close = (int32_t) id;
                else if (blocks && key == form.answer_open) out.answer_open = (int32_t) id;
            }
            if (ty == 5 || ty == 6) continue;   // unused and byte entries never appear as text
            if (ty == 3) {
                // Control tokens referenced by the active tool form keep their literal text so tool grammars and
                // triggers can match them; turn-ending control tokens stay non-text.
                if (!is_end_id[(size_t) id] && in_tool_form(tform, key)) b = key;
                continue;
            }
            if (ty == 4) {
                b = key;   // an added token keeps its literal text
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
    std::vector<int32_t> ends = end_ids;
    if (blocks && (out.block_header < 0 || out.think_close < 0))   // not this vocabulary's form: no thinking known
        out.think_open = out.think_close = out.block_header = out.answer_open = -1;
    // When the block end is also a literal tool-form delimiter, keep it out of Vocab::end_ids so tool grammars can
    // match it in the trie; Guard gates it directly during block-answer phases.
    if (out.block_header >= 0 && bytes[(size_t) out.think_close].empty() &&
        std::find(ends.begin(), ends.end(), out.think_close) == ends.end())
        ends.push_back(out.think_close);   // the answer block's end: allowed where the grammar can end
    out.vocab = std::make_shared<const Vocab>(std::move(bytes), std::move(ends));
    return true;
}

bool Guard::begin(const std::string& grammar_json, const int64_t* prompt, size_t n_prompt, std::string& err) {
    err.clear();
    on_ = false;
    windows_ = misses_ = fallbacks_ = ns_ = 0;
    header_starts_call_ = false;
    RequestGrammar rg = grammar_from_request(grammar_json, err);
    if (!err.empty()) return false;
    // A reply of blocks that may not think, outside a grammar that holds the block structure: the guard only keeps the
    // thinking block from opening.
    ban_think_ = pv_.block_header >= 0 && !rg.think;
    call_names_ = rg.call_names;
    must_call_ = rg.must_call;
    first_mask_.clear();
    after_header_ = n_prompt > 0 && prompt[n_prompt - 1] == pv_.block_header;
    has_grammar_ = (bool) rg.grammar;
    if (!rg.grammar) {
        if (!ban_think_) return true;
        blocks_ = thinking_ = false;
        text_only_ = !rg.calls && pv_.answer_open >= 0;
        phase_ = Phase::answer;
        on_ = true;
        return true;
    }
    st_ = GrammarState(rg.grammar, rg.triggers);
    // A reply of blocks under a grammar from the first answer token: thinking blocks, then the answer block. A lazy
    // grammar (tool calls) finds its own start.
    blocks_ = pv_.block_header >= 0 && !rg.triggers.lazy();
    const bool has_header_text = pv_.block_header >= 0 && (size_t) pv_.block_header < pv_.vocab->size() &&
                                 !pv_.vocab->bytes(pv_.block_header).empty();
    if (blocks_ && has_header_text && st_.allows(pv_.vocab->bytes(pv_.block_header), pv_.block_header)) {
        header_starts_call_ = true;
    } else if (!blocks_ && after_header_ && has_header_text) {
        st_.accept(pv_.vocab->bytes(pv_.block_header), pv_.block_header);
    }
    allow_think_ = rg.think;
    ban_think_ = ban_think_ && !blocks_;   // a grammar of blocks bans thinking through its phases
    phase_ = blocks_ ? Phase::between : Phase::answer;
    // generation starts inside a thinking block when the prompt's last think tag opens one (a reply of blocks: after
    // the prompt's last block marker)
    thinking_ = false;
    for (size_t i = n_prompt; i-- > 0;) {
        const int64_t t = prompt[i];
        if (t == pv_.think_close) break;
        if (t == pv_.think_open) { thinking_ = true; break; }
        if (blocks_ && t == pv_.block_header) { phase_ = Phase::opened; break; }
        if (blocks_ && t == pv_.answer_open) { phase_ = Phase::answer; break; }
    }
    on_ = true;
    return true;
}

bool Guard::is_end(int32_t token) const {
    const auto& ends = pv_.vocab->end_ids();
    return token != pv_.think_close && std::find(ends.begin(), ends.end(), token) != ends.end();
}

void Guard::build_first_mask() {
    if (!first_mask_.empty()) return;
    const size_t n = pv_.vocab->size();
    first_mask_.assign(n, (uint8_t) 0);
    for (size_t i = 0; i < n; ++i) {
        const std::string& b = pv_.vocab->bytes((int32_t) i);
        if (b.empty()) continue;
        for (const std::string& name : call_names_)
            if (name.compare(0, b.size(), b) == 0) { first_mask_[i] = 1; break; }
    }
}

// the first token of a block that is not a text block: the start of a tool call's name
bool Guard::opens_block(int32_t token) {
    if (text_only_ || call_names_.empty()) return !text_only_;
    build_first_mask();
    return token >= 0 && (size_t) token < first_mask_.size() && first_mask_[(size_t) token];
}

bool Guard::feed(int32_t token) {
    const bool next_after = token == pv_.block_header;
    if (ban_think_) {
        if (token == pv_.think_open) return false;
        if (after_header_ && !(token == pv_.answer_open && !must_call_) && !opens_block(token)) return false;
        if (!has_grammar_) {
            after_header_ = next_after;
            return true;
        }
    }
    auto accept_ok = [&]() {
        if (ban_think_) after_header_ = next_after;
        return true;
    };
    if (thinking_) {
        if (token == pv_.think_close) {
            thinking_ = false;
            if (blocks_) phase_ = Phase::between;
            return accept_ok();
        }
        if (!blocks_) return accept_ok();
        if (token == pv_.block_header) {   // the next block opened without closing this one
            thinking_ = false;
            phase_ = Phase::opened;
            return accept_ok();
        }
        return !is_end(token) ? accept_ok() : false;   // the reply does not end before its answer
    }
    if (blocks_) {
        if (phase_ == Phase::between) {
            if (token != pv_.block_header) return false;
            phase_ = Phase::opened;
            return accept_ok();
        }
        if (phase_ == Phase::done) return is_end(token) ? accept_ok() : false;
        if (phase_ == Phase::opened) {
            if (token == pv_.think_open && allow_think_) { thinking_ = true; return accept_ok(); }
            if (token == pv_.answer_open) { phase_ = Phase::answer; return accept_ok(); }
            // else the answer's first text, or a tool call starting right after the header
        }
        if (token == pv_.block_header) return false;
        if (token == pv_.think_close) {
            if (st_.active() && !st_.allows_end()) return false;
            phase_ = Phase::done;   // the answer block ended
            return accept_ok();
        }
    }
    const auto& ends = pv_.vocab->end_ids();
    if (std::find(ends.begin(), ends.end(), token) != ends.end()) {
        if (st_.active() && !st_.allows_end()) return false;
        if (blocks_ && token == pv_.think_close) phase_ = Phase::done;   // the answer block ended
        return accept_ok();
    }
    if (token < 0 || (size_t) token >= pv_.vocab->size()) return !st_.active() ? accept_ok() : false;
    const std::string& b = pv_.vocab->bytes(token);
    if (b.empty()) {
        if (st_.active()) return false;   // a control token is not text
        // A non-text control token (such as a block content tag) breaks any partial trigger prefix in tail_.
        st_.reset();
        return accept_ok();
    }
    if (st_.accept(b, token)) {
        if (blocks_) phase_ = Phase::answer;
        return accept_ok();
    }
    if (blocks_ && phase_ == Phase::opened && header_starts_call_) {
        GrammarState cs = st_.clone();
        if (cs.accept(pv_.vocab->bytes(pv_.block_header), pv_.block_header) && cs.accept(b, token)) {
            st_ = std::move(cs);
            blocks_ = false;
            header_starts_call_ = false;
            phase_ = Phase::answer;
            return accept_ok();
        }
    }
    return false;
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

bool Guard::mask_bias(const std::vector<float>& user_bias, bool use_cache) {
    const size_t n = pv_.vocab->size();
    const bool has_u = user_bias.size() == n;
    bias_.resize(n);
    bool any = false;
    bool cleared = false;
    auto allow_bias = [&](int32_t t) {
        if (t < 0 || (size_t) t >= n) return;
        const float v = has_u ? user_bias[(size_t) t] : 0.0f;
        bias_[(size_t) t] = v;
        if (v > -INFINITY) any = true;
    };
    auto forbid_bias = [&](int32_t t) {
        if (t < 0 || (size_t) t >= n) return;
        if (bias_[(size_t) t] > -INFINITY) cleared = true;
        bias_[(size_t) t] = -INFINITY;
    };
    auto fill_from_mask = [&](const uint8_t* m) {
        if (has_u) {
            for (size_t i = 0; i < n; ++i) {
                const float v = m[i] ? user_bias[i] : -INFINITY;
                bias_[i] = v;
                if (v > -INFINITY) any = true;
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                if (m[i]) {
                    bias_[i] = 0.0f;
                    any = true;
                } else {
                    bias_[i] = -INFINITY;
                }
            }
        }
    };
    auto fill_all_allowed = [&]() {
        if (has_u) {
            for (size_t i = 0; i < n; ++i) {
                const float v = user_bias[i];
                bias_[i] = v;
                if (v > -INFINITY) any = true;
            }
        } else {
            std::fill(bias_.begin(), bias_.end(), 0.0f);
            any = n > 0;
        }
    };
    auto finish = [&]() {
        if (cleared) {
            any = false;
            for (size_t i = 0; i < n; ++i) {
                if (bias_[i] > -INFINITY) { any = true; break; }
            }
        }
        return any;
    };
    if (ban_think_) {
        if (after_header_ && (text_only_ || !call_names_.empty()) && !(has_grammar_ && st_.active())) {
            std::fill(bias_.begin(), bias_.end(), -INFINITY);
            if (!must_call_) allow_bias(pv_.answer_open);
            if (!text_only_) {
                build_first_mask();
                for (size_t i = 0; i < n; ++i) if (first_mask_[i]) allow_bias((int32_t) i);
            }
        } else if (has_grammar_) {
            fill_from_mask(st_.fill_mask_view(*pv_.vocab, mask_, use_cache));
        } else {
            fill_all_allowed();
        }
        forbid_bias(pv_.think_open);
        return finish();
    }
    if (blocks_ && thinking_) {
        fill_all_allowed();
        for (int32_t e : pv_.vocab->end_ids())
            if (is_end(e)) forbid_bias(e);
        return finish();
    }
    if (blocks_ && phase_ == Phase::between) {
        std::fill(bias_.begin(), bias_.end(), -INFINITY);
        allow_bias(pv_.block_header);
        return any;
    }
    if (blocks_ && phase_ == Phase::done) {
        std::fill(bias_.begin(), bias_.end(), -INFINITY);
        for (int32_t e : pv_.vocab->end_ids())
            if (is_end(e)) allow_bias(e);
        return any;
    }
    if (blocks_ && phase_ == Phase::opened && header_starts_call_) {
        st_.fill_mask(*pv_.vocab, mask_, use_cache);
        GrammarState cs = st_.clone();
        if (cs.accept(pv_.vocab->bytes(pv_.block_header), pv_.block_header)) {
            std::vector<uint8_t> call_scratch;
            const uint8_t* cm = cs.fill_mask_view(*pv_.vocab, call_scratch, use_cache);
            for (size_t i = 0; i < n; ++i) mask_[i] |= cm[i];
        }
        fill_from_mask(mask_.data());
        forbid_bias(pv_.block_header);
        if (st_.allows_end()) allow_bias(pv_.think_close);
        else forbid_bias(pv_.think_close);
        allow_bias(pv_.answer_open);
        if (allow_think_) allow_bias(pv_.think_open);
        return finish();
    }
    fill_from_mask(st_.fill_mask_view(*pv_.vocab, mask_, use_cache));
    if (blocks_) {
        forbid_bias(pv_.block_header);
        if (st_.allows_end()) allow_bias(pv_.think_close);
        else forbid_bias(pv_.think_close);
        if (phase_ == Phase::opened) {   // the grammar's first text, or a block's tag
            allow_bias(pv_.answer_open);
            if (allow_think_) allow_bias(pv_.think_open);
        }
    }
    return finish();
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
    const auto dead = [&] {
        basalt::log::error("basalt grammar", "no token is allowed here (%s); the reply ends", describe().c_str());
        on_ = false;
        return done(Redraw::dead);
    };
    if (!mask_bias(user_bias, true)) return dead();
    int32_t t = 0;
    if (!draw(bias_, t)) return done(Redraw::failed);
    if (feed(t)) { token = t; return done(Redraw::drawn); }
    basalt::log::warn("basalt grammar", "a masked draw was refused by the grammar (token %d, %s); drawing again", (int) t,
                      describe().c_str());
    if (!mask_bias(user_bias, false)) return dead();
    if (!draw(bias_, t)) return done(Redraw::failed);
    if (feed(t)) { token = t; return done(Redraw::drawn); }
    // still refused: the likeliest token the grammar accepts, tried best first (a few, in case the mask and accept differ)
    std::vector<float> lg;
    if (!logits(lg) || lg.size() != bias_.size()) return done(Redraw::failed);
    std::vector<uint32_t> cand;
    for (size_t i = 0; i < bias_.size(); ++i) {
        if (!(bias_[i] > -INFINITY)) continue;
        lg[i] += bias_[i];
        if (lg[i] > -INFINITY) cand.push_back((uint32_t) i);   // finite or +inf (not -inf, not nan)
    }
    if (cand.empty()) {
        basalt::log::error("basalt grammar", "every allowed token has a non-finite logit (%s); the reply ends",
                           describe().c_str());
        on_ = false;
        return done(Redraw::dead);
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
