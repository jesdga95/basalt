// A tool form for the grammar tests: <tool_call> blocks with <function=..> and <parameter=..> elements.
#pragma once

#include "basalt/program/grammar/tool_form.hpp"

inline void set_test_tool_form() {
    basalt::program::grammar::ToolForm f;
    f.call_open = "<tool_call>\n";
    f.call_close = "\n</tool_call>";
    f.fn_open = "<function=";
    f.fn_mid = ">\n";
    f.fn_close = "</function>";
    f.param_open = "<parameter=";
    f.param_mid_text = f.param_mid_value = ">\n";
    f.param_close = "\n</parameter>\n";
    f.text_end = "</parameter>";
    f.trigger = "<tool_call>";
    basalt::program::grammar::set_tool_form(f);
}
