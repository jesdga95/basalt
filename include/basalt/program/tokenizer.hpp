// include/basalt/program/tokenizer.hpp - the byte-level BPE tokenizer the engine runs on request text.
//
// Tables load from the pack's tokenizer files (vocab.json, merges.txt, token_type.json). encode() splits the text at
// special-token literals, pre-tokenizes the plain spans with the qwen35 rules, and merges each piece by lowest rank
// first. The ids equal tools/tokenizer.py's. Nothing here knows a GPU or an OS. One thread encodes at a time.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace basalt::program {

class Tokenizer {
public:
    /// A byte range of the text in which special-token literals stay ordinary text.
    struct Span {
        size_t begin = 0;
        size_t end = 0;
    };

    /// `vocab_json` is {"token text": id}, `merges_txt` one "left right" pair per line, best rank first, `types_json`
    /// one type per id (3 control, 4 user-defined). False with `err` on a malformed table.
    bool load(std::string_view vocab_json, std::string_view merges_txt, std::string_view types_json, std::string& err);
    bool loaded() const { return !raw_.empty(); }
    size_t vocab_size() const { return raw_.size(); }

    /// Tokenizes UTF-8 `text` into `out`. Type-4 literals always become their token, type-3 ones only with
    /// `parse_special`; a literal that starts inside a `plain` span is text. False with `err` if a piece cannot be built
    /// from the vocabulary.
    bool encode(std::string_view text, bool parse_special, const std::vector<Span>& plain, std::vector<int32_t>& out,
                std::string& err);

    /// The raw bytes one token stands for (empty for an id without a token).
    const std::string& token_bytes(int32_t id) const { return raw_[(size_t) id]; }

private:
    struct Merge {
        uint64_t key = ~0ull;
        uint32_t rank = 0;
        int32_t result = -1;
    };
    struct Literal {
        std::string text;
        int32_t id = -1;
        bool control = false;   // type 3: a literal only when parse_special is set
    };
    struct LiteralSet {
        std::vector<Literal> lits;
        std::vector<uint32_t> by_first[256];   // indexes into lits, longest first
        bool first[256] = {};
        int distinct_first = 0;
        int single_first = -1;
    };
    struct MemoEntry {
        uint64_t hash = 0;
        uint32_t key_off = 0, key_len = 0, ids_off = 0, ids_len = 0;   // key_len 0: empty slot
    };

    int32_t find_merge(int32_t left, int32_t right, uint32_t& rank) const;
    bool merge_piece(const uint8_t* p, size_t n, std::vector<int32_t>& out, std::string& err);
    bool encode_plain(std::string_view text, std::vector<int32_t>& out, std::string& err);
    void memo_grow();
    static void build_literals(LiteralSet& set, const std::vector<Literal>& all);

    std::vector<std::string> raw_;          // token id -> bytes
    int32_t byte_id_[256];                  // byte -> its single-byte token, or -1
    std::vector<Merge> merges_;             // open addressing, power-of-two size
    uint64_t merge_mask_ = 0;
    LiteralSet always_, all_;               // type 4 only, and types 3 and 4

    std::vector<MemoEntry> memo_;
    size_t memo_used_ = 0;
    std::string memo_keys_;
    std::vector<int32_t> memo_ids_;

    // scratch for the pre-tokenizer and the merger, kept between calls
    std::vector<size_t> ends_;
    std::vector<int32_t> sym_;
    std::vector<uint32_t> rank_;
};

}  // namespace basalt::program
