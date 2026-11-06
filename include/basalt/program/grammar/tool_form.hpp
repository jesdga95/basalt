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
};

/// The running model's form; a tool grammar is refused until a model set it.
void set_tool_form(const ToolForm& form);
const ToolForm& tool_form();

}  // namespace basalt::program::grammar
