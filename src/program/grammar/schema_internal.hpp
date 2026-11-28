// src/program/grammar/schema_internal.hpp - the JSON schema tree and the converter from it to GBNF. A port of
// llama.cpp's common/json-schema.cpp (schema -> tree) and common/json-schema-to-grammar.cpp (tree -> GBNF); the
// grammar text it writes is the same. Internal to the grammar library.
#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "json_value.hpp"

namespace basalt::program::grammar {

/// Unescapes one RFC 6901 JSON Pointer reference token (~1 -> /, ~0 -> ~) in a single left-to-right pass.
inline bool json_pointer_unescape(std::string_view token, std::string& out) {
    out.clear();
    out.reserve(token.size());
    for (size_t i = 0; i < token.size(); ++i) {
        if (token[i] != '~') {
            out += token[i];
        } else {
            if (i + 1 >= token.size()) return false;
            const char next = token[++i];
            if (next == '0') out += '~';
            else if (next == '1') out += '/';
            else return false;
        }
    }
    return true;
}

/// Strict RFC 6901 array index parser ("0" or [1-9][0-9]* within [0, limit)).
inline bool json_pointer_index(std::string_view token, size_t limit, size_t& idx) {
    if (token.empty() || limit == 0) return false;
    if (token.size() > 1 && token[0] == '0') return false;
    size_t v = 0;
    for (char ch : token) {
        if (ch < '0' || ch > '9') return false;
        const size_t d = (size_t) (ch - '0');
        if (v > ((std::numeric_limits<size_t>::max)() - d) / 10) return false;
        v = v * 10 + d;
        if (v >= limit) return false;
    }
    idx = v;
    return true;
}

struct SchemaNode;
using SchemaPtr = std::unique_ptr<SchemaNode>;

struct SchemaProperty {
    std::string name;
    SchemaPtr schema;
    bool required = false;
};

struct SchemaNode {
    enum Kind { Any, Ref, AnyOf, AllOf, Const, Enum, Null, Boolean, Number, Integer, String, Array, Tuple, Object };
    enum Format { FormatNone, FormatUuid, FormatDate, FormatTime, FormatDateTime };

    Kind kind = Any;
    // Ref: the $ref text and its target (owned by SchemaDocument::refs)
    std::string ref;
    const SchemaNode* target = nullptr;
    // AnyOf / AllOf alternatives, Tuple items
    std::vector<SchemaPtr> children;
    // Const (one value) / Enum
    std::vector<JsonValue> values;
    // Integer bounds, inclusive
    bool has_minimum = false;
    bool has_maximum = false;
    int64_t minimum = INT64_MIN;
    int64_t maximum = INT64_MAX;
    // String
    std::string pattern;
    Format format = FormatNone;
    int min_length = 0;
    int max_length = -1;
    // Array: the item schema
    SchemaPtr items;
    int min_items = 0;
    int max_items = -1;
    // Object
    std::vector<SchemaProperty> properties;
    SchemaPtr additional_properties;   ///< null when not allowed
};

struct SchemaDocument {
    SchemaPtr root;
    std::map<std::string, SchemaPtr> refs;
};

/// Builds the tree for `schema`; `root` is the document that local $refs point into (the schema itself when null).
/// Throws std::runtime_error with llama.cpp's message when the schema is outside the supported subset.
SchemaDocument build_schema_document(const JsonValue& schema, const JsonValue* root = nullptr);

/// GBNF text for a literal string, quoted and escaped.
std::string gbnf_format_literal(const std::string& literal);

class SchemaConverter {
public:
    explicit SchemaConverter(bool dotall = false);

    /// Adds a rule (the name is made safe; an existing rule with other content gets a numeric suffix); returns the name.
    std::string add_rule(const std::string& name, const std::string& rule);
    /// Adds the rules for a schema under `name` ("" for the root) and returns the rule name.
    std::string add_schema(const std::string& name, const SchemaNode& schema);
    /// Throws std::invalid_argument when the schema produced errors.
    void check_errors();
    std::string format_grammar() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

/// The grammar for a whole document (rule `root`), as llama.cpp's json_schema_to_grammar. Throws std::invalid_argument.
std::string schema_document_to_gbnf(const SchemaDocument& doc);

}  // namespace basalt::program::grammar
