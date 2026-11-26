#include "models/deepseek4/adapter.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main() {
    const auto adapter = basalt::models::deepseek4::make_adapter();
    if (!adapter || std::string(adapter->arch()) != "deepseek4") {
        std::puts("deepseek4 adapter: wrong architecture");
        return 1;
    }

    const auto spec = adapter->tokenizer_spec();
    if (spec.vocab_key != "tokenizer.ggml.tokens" || spec.chat_format != "deepseek") {
        std::puts("deepseek4 adapter: tokenizer metadata mismatch");
        return 1;
    }
    const auto pre = adapter->pretokenizer();
    if (std::string(pre.name) != "joyai-llm" || !pre.split || pre.ignore_merges) {
        std::puts("deepseek4 adapter: pre-tokenizer metadata mismatch");
        return 1;
    }
    const std::string text = "abc1234567中文";
    std::vector<size_t> ends;
    pre.split(text.data(), text.size(), ends);
    if (ends != std::vector<size_t>({3, 6, 9, 10, 16})) {
        std::puts("deepseek4 adapter: pre-tokenizer callback mismatch");
        return 1;
    }

    const auto names = adapter->token_names();
    if (names.eos != std::vector<std::string>({"<｜end▁of▁sentence｜>"}) || names.turn != "<｜User｜>" ||
        names.reply_role != "<｜Assistant｜>" || names.image_pad != "<｜deepseek_image｜>" || !names.video_pad.empty()) {
        std::puts("deepseek4 adapter: control-token names mismatch");
        return 1;
    }

    const auto tools = adapter->tool_form();
    if (tools.call_open != "<｜DSML｜tool_calls>\n" || tools.call_close != "\n</｜DSML｜tool_calls>" ||
        tools.fn_open != "<｜DSML｜invoke name=\"" || tools.fn_mid != "\">\n" ||
        tools.fn_close != "</｜DSML｜invoke>" || tools.param_open != "<｜DSML｜parameter name=\"" ||
        tools.param_mid_text != "\" string=\"true\">" || tools.param_mid_value != "\" string=\"false\">" ||
        tools.param_close != "</｜DSML｜parameter>\n" || !tools.calls_share_block ||
        tools.trigger != "<｜DSML｜tool_calls>") {
        std::puts("deepseek4 adapter: DSML grammar form mismatch");
        return 1;
    }
    const auto modes = adapter->gpu_modes();
    if (modes.peer || modes.split || modes.tensor) {
        std::puts("deepseek4 adapter: unsupported multi-GPU mode was advertised");
        return 1;
    }
    std::puts("deepseek4 adapter metadata: ok");
    return 0;
}
