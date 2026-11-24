// basalt/program/grammar/tool_form.hpp - how a model writes a tool call, for the tool-call grammar.
#pragma once

#include <string>

namespace basalt::program::grammar {

struct ToolForm {
    std::string call_open, call_close;
    std::string fn_open, fn_mid, fn_close;            ///< fn_open NAME fn_mid PARAMETERS fn_close
    std::string param_open, param_mid_text, param_mid_value, param_close;   ///< param_open NAME param_mid_* VALUE param_close
    std::string text_end;                             ///< what ends a parameter's free text
    bool calls_share_block = false;                   ///< several calls in one call_open .. call_close, joined by a newline
    std::string trigger;                              ///< the text that starts a call: the lazy trigger
    /// A regex in place of `trigger`, for a call whose start is told apart only by what follows a shared opening; the
    /// grammar then runs from where the match starts.
    std::string trigger_pattern;
    /// The arguments as one JSON object of the tool's parameter schema (the param_* fields unused): fn_open NAME fn_mid
    /// ARGUMENTS fn_close, "{name}" in fn_mid standing for the name as a JSON string.
    bool json_args = false;
};

/// The running model's form; a tool grammar is refused until a model set it.
void set_tool_form(const ToolForm& form);
const ToolForm& tool_form();

}  // namespace basalt::program::grammar
