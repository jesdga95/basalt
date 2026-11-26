// include/basalt/program/grammar/guard.hpp - one request's grammar, checked against the tokens a verify window produced.
//
// vocab_from_pack builds token id -> bytes from the pack's tokenizer files. A Guard holds the request's GrammarState and,
// after each window, scans the sampled tokens in order: the first one the grammar does not allow is reported, and the
// caller hands redraw() two callbacks to draw that row again under the grammar's mask and to read its logits. The guard
// never switches itself off over a refused draw: it redraws under a fresh mask, then picks the likeliest token the
// grammar accepts, and only a state with no allowed token at all ends the request. Nothing here knows a GPU or an OS.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "basalt/program/grammar/grammar.hpp"
#include "basalt/program/grammar/reply_form.hpp"

namespace basalt::program::grammar {

struct PackVocab {
    std::shared_ptr<const Vocab> vocab;
    int32_t think_open = -1;    ///< the token that opens a thinking block, or -1
    int32_t think_close = -1;   ///< the token that closes it (a reply of blocks: that ends any block), or -1
    int32_t block_header = -1;  ///< a reply of blocks: the token that opens a block, else -1
    int32_t answer_open = -1;   ///< a reply of blocks: the tag of the answer block, or -1
};

/// `vocab_json` is tokenizer/vocab.json ({"token text": id}, byte-level BPE spelling), `types_json` is
/// tokenizer/token_type.json (one type per id: 3 control, 4 user-defined added token, 5 unused, 6 byte). Ids from
/// 0 to n_vocab - 1; an id without an entry has no text. `end_ids` are the tokens that finish generation. The thinking
/// tokens are reply_form()'s: the default form's are added tokens, a reply of blocks' may be control tokens, and its
/// block end joins the end tokens (the answer block ends only where the grammar can).
bool vocab_from_pack(std::string_view vocab_json, std::string_view types_json, size_t n_vocab,
                     const std::vector<int32_t>& end_ids, PackVocab& out, std::string& err);

/// Draws the refused row again with `bias` (n_vocab floats, added to the logits) and returns the token; false on a failure.
using DrawFn = std::function<bool(const std::vector<float>& bias, int32_t& token)>;
/// Reads the refused row's logits (n_vocab floats); false on a failure.
using LogitsFn = std::function<bool(std::vector<float>& logits)>;

/// What redraw() settled on.
enum class Redraw {
    drawn,      ///< the draw under the mask was allowed
    fallback,   ///< it was refused twice: the token is the likeliest one the grammar accepts
    dead,       ///< no token is allowed (a grammar dead end, or the request's bias rules every allowed one out): end the request
    failed,     ///< a callback failed
};

class Guard {
public:
    Guard(PackVocab vocab) : pv_(std::move(vocab)) {}

    /// Starts a request. `grammar_json` is the request's "grammar" object. `prompt` ends where generation starts: an
    /// open thinking block there defers the grammar until the block is closed. False with `err` on an unusable grammar.
    /// A request whose grammar is null (tool_choice "none") leaves the guard off. With a reply of blocks and a grammar
    /// from the first answer token, the reply's thinking blocks pass (unless the request says "think": false) and the
    /// grammar holds the answer block.
    bool begin(const std::string& grammar_json, const int64_t* prompt, size_t n_prompt, std::string& err);
    bool on() const { return on_; }

    /// Feeds out[first..rows) in order and returns the first row the grammar does not allow, or -1. The state has then
    /// advanced over every row from `first` before it. Rows before `first` are not generated (a window's prompt
    /// tokens) and are skipped.
    int scan(const int32_t* out, int rows, int first);
    /// After a miss from scan(): draws the row again under the grammar's mask (bias[id] = user_bias[id] where the
    /// grammar allows id, else -inf; `user_bias` may be empty). A draw that is still refused is drawn once more under a
    /// mask built without the cache, then the token is the highest logit (plus user_bias) the grammar accepts. On
    /// drawn and fallback `token` is fed to the grammar; on dead the guard turns itself off and `token` is untouched.
    Redraw redraw(const std::vector<float>& user_bias, const DrawFn& draw, const LogitsFn& logits, int32_t& token);

    int64_t windows() const { return windows_; }
    int64_t fallbacks() const { return fallbacks_; }
    int64_t misses() const { return misses_; }
    double ms() const { return (double) ns_ / 1e6; }

private:
    // 1: fed, 0: not allowed
    bool feed(int32_t token);
    void build_first_mask();
    bool opens_block(int32_t token);
    void mask_bias(const std::vector<float>& user_bias, bool use_cache, std::vector<float>& bias);
    std::string describe() const;

    // a reply of blocks, outside a thinking block: where it is before (and after) the answer
    enum class Phase { answer, between, opened, done };   // in the answer; a block may open; after a header; answered
    bool is_end(int32_t token) const;   // a token that ends the reply (not the block end)

    PackVocab pv_;
    GrammarState st_;
    std::vector<uint8_t> mask_;
    bool on_ = false;
    bool thinking_ = false;
    bool blocks_ = false;        // this request's reply is read as blocks
    bool ban_think_ = false;     // a reply of blocks that may not think, outside a grammar of blocks: no thinking block opens
    bool after_header_ = false;  // the last token opened a block
    bool has_grammar_ = false;
    std::vector<std::string> call_names_;   // the tools a reply may call, when known
    std::vector<uint8_t> first_mask_;       // the tokens that start one of their names
    bool must_call_ = false;                // every block that opens is a call
    bool text_only_ = false;     // no grammar and no calls: every block of a reply of blocks is a text block
    bool allow_think_ = true;    // a reply of blocks may think
    Phase phase_ = Phase::answer;
    int64_t windows_ = 0, misses_ = 0, fallbacks_ = 0, ns_ = 0;
};

}  // namespace basalt::program::grammar
