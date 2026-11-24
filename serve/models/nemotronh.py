"""The Nemotron-H dialect (NVIDIA Nemotron 3 / 3.5): the checkpoint's Jinja chat template, ChatML turns with `<think>`
and Qwen's `<tool_call><function=...>` calls; the template's own switches for the reasoning effort (`medium_effort`)
and for earlier turns' thinking (`truncate_history_thinking`)."""
from __future__ import annotations

from serve.models.qwen import Qwen
from serve.util.text import IM_END


class NemotronH(Qwen):
    name = "nemotronh"
    stop_texts = (IM_END, "</s>")
    reasoning_close = "\n</think>"
    effort_turn = ""                    # the effort is a template switch, not a trailing turn
    video_pad = ""
    video_marker = ""
    caps = {"supports_tool_calls": True, "supports_parallel_tool_calls": True, "supports_reasoning_effort": True,
            "supports_system_role": True, "supports_preserve_reasoning": True}

    def template_kwargs(self, kwargs: dict) -> dict:
        out = dict(kwargs)
        if "preserve_reasoning" in out:
            out["truncate_history_thinking"] = not out.pop("preserve_reasoning")
        if out.get("reasoning_effort") in ("low", "medium"):
            out["medium_effort"] = True
        return out

    def effort_level(self, level):
        """low and medium are the template's efficient reasoning, high and max its default."""
        if isinstance(level, int):
            raise ValueError(f"unknown reasoning effort {level!r}: use none, low, medium, high or max")
        return "high" if level in ("high", "max", "xhigh") else level
