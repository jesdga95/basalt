// include/basalt/program/qwen_forms.hpp - what every model of the Qwen family shares at the text level: the qwen35
// pre-tokenizer, the `<tool_call><function=...>` tool-call form and the names of its special tokens.
#pragma once

#include "basalt/model/adapter.hpp"
#include "basalt/program/grammar/tool_form.hpp"

#include <cstddef>
#include <vector>

namespace basalt::program::tokenizer {
/// The qwen35 pre-tokenizer: `ends` gets where each piece of the text ends.
void pretok_qwen35(const char* text, size_t size, std::vector<size_t>& ends);
}  // namespace basalt::program::tokenizer

namespace basalt::program {

/// `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call>`
inline grammar::ToolForm qwen_tool_form() {
    grammar::ToolForm f;
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
    return f;
}

/// The end-of-turn, turn, image-pad and video-pad tokens.
inline model::TokenNames qwen_token_names() {
    return {{"<|endoftext|>", "<|im_end|>"}, "<|im_start|>", "<|image_pad|>", "<|video_pad|>"};
}

}  // namespace basalt::program
