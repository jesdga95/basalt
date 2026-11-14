"""The Qwen dialect: the model's Jinja chat template and `<tool_call><function=...>` calls."""
from __future__ import annotations

from serve import frontend
from serve.models import Dialect
from serve.util.text import IM_END


class Qwen4Exp(Dialect):
    name = "qwen4exp"
    stop_texts = (IM_END, "<|endoftext|>")
    reasoning_close = "\n</think>\n\n"
    effort_turn = "<|im_start|>system\n{}<|im_end|>\n"
    video_pad = "<|video_pad|>"
    video_marker = "<|vision_start|><|video_pad|><|vision_end|>"
    # chat_template.jinja: tools and several calls per turn, the efforts xhigh/medium/low, a leading system turn,
    # and `preserve_thinking`
    caps = {"supports_tool_calls": True, "supports_parallel_tool_calls": True, "supports_reasoning_effort": True,
            "supports_system_role": True, "supports_preserve_reasoning": True}

    def template(self, source):
        return frontend.ChatTemplate(source)

    def parser(self, thinking, tools, stream_tools=True):
        return frontend.OutputParser(thinking=thinking, tools=tools, stream_tools=stream_tools)

    def literal_tags(self, controls):
        return frontend.literal_tags(controls)

    def video_text(self, times, rows):
        # Qwen3-VL's processor at the model's release replaces the whole marker: per group "<t seconds>", then the
        # group's pads between its own start and end
        return "".join(f"<{t:.1f} seconds><|vision_start|>{self.video_pad * rows}<|vision_end|>" for t in times)

    def forced_opening(self, name):
        return frontend.CALL_START + (f"\n<function={name}>\n" if name else "\n<function=")
