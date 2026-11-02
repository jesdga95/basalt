// include/basalt/program/grammar/schema.hpp - GBNF text built from a JSON schema or from a tool list.
//
// json_schema_to_gbnf is llama.cpp's json-schema-to-grammar: the same schema keywords and the same grammar text (type,
// enum, const, properties/required/additionalProperties, items/prefixItems, minItems/maxItems, minLength/maxLength,
// integer minimum/maximum, pattern, format, anyOf/oneOf/allOf, local ). What llama.cpp ignores is ignored here
// too: uniqueItems, number bounds, a pattern that is not anchored or has no grammar form (any string is accepted);
// what it refuses (a  outside the document) is refused.
//
// tool_call_gbnf writes the tool-call text the server parses:
//   <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call>
// where VALUE follows the declared type: raw text for strings (anything but a closing </parameter>), JSON otherwise.
// The required parameters come in any order, each once (up to max_permute of them), then the optional ones in any order,
// repeatable: the construction of llama.cpp's Qwen3-Coder parser.
#pragma once

#include <memory>
#include <string>

#include "basalt/program/grammar/grammar.hpp"

namespace basalt::program::grammar {

/// GBNF with a `root` rule for the schema (JSON text). Empty with `err` set on bad JSON or an unresolvable $ref.
std::string json_schema_to_gbnf(const std::string& schema_json, std::string& err);

struct ToolGrammarOptions {
    size_t max_permute = 6;          ///< required parameters are permuted up to this many; more keep the declared order
    bool multiple_calls = true;      ///< more than one <tool_call> block
    std::string answer_schema;       ///< a JSON schema: the root also takes one JSON value of it instead of calls
};

/// `tools_json` is an array of {name, parameters:{properties, required}} (or the OpenAI {type:"function",
/// function:{...}} wrapper). Empty with `err` set on bad input or an empty list.
std::string tool_call_gbnf(const std::string& tools_json, const ToolGrammarOptions& opt, std::string& err);

/// The text that starts a tool call: the lazy trigger for tool_call_gbnf.
inline const char* tool_call_trigger() { return "<tool_call>"; }

/// What the engine needs for one request: the parsed grammar and when it starts.
struct RequestGrammar {
    std::shared_ptr<const Grammar> grammar;   ///< null: nothing to enforce
    Triggers triggers;                        ///< empty: enforce from the first answer token
};

/// Builds the grammar from the request's "grammar" JSON object:
///   {"kind":"tools","tools":[<OpenAI tool objects>],"tool_choice":<as received>}   lazy, trigger "<tool_call>"
///   {"kind":"json_schema","schema":{...}}                                          from the first answer token
///   {"kind":"json_object"}                                                         any JSON object
/// A json_schema or json_object request may carry "tools" too: from the first answer token, either tool calls or the
/// JSON value. tool_choice "none" gives a null grammar; a named choice keeps only that tool. A null grammar with empty `err` is
/// not an error; an unusable request sets `err`.
RequestGrammar grammar_from_request(const std::string& grammar_json, std::string& err);

}  // namespace basalt::program::grammar
