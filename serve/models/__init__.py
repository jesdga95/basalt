"""Per-model chat dialects: how a model's prompt is rendered, its output parsed and its turn ended.

The server picks one from the model file's architecture; a file of an unknown architecture is refused.
"""
from __future__ import annotations


class Dialect:
    name = ""
    stop_texts: tuple[str, ...] = ()        # texts that end the model's turn
    reasoning_close = ""                    # closes thinking the model left open, or that reached its budget or repeats one token
    forced_call_ends_reply = False          # a named tool_choice ends the reply after its one call
    nothink_lead = ""                       # text between a no-thinking prompt and a forced call's opening
    image_pad = "<|image_pad|>"             # the prompt token that stands for one row of an image's embeddings
    image_start = "<|vision_start|>"        # the token an image's pad run follows ("" : every pad is an image row)
    video_pad = ""                          # the prompt token for one row of a video group ("": the model reads no video)
    video_marker = ""                       # what the template writes for a video, replaced by video_text
    in_engine_vision = False                # the image encoder always lives in the engine (only --vision dynamic)
    effort_turn = ""                        # a trailing turn that states the reasoning effort ("{}" for its text)
    # what the model's template does with a conversation, as llama.cpp's /props `chat_template_caps` names it
    caps = {"supports_tool_calls": False, "supports_parallel_tool_calls": False, "supports_reasoning_effort": False,
            "supports_system_role": False, "supports_preserve_reasoning": False}

    def template_kwargs(self, kwargs: dict) -> dict:
        """The frontend's template kwargs as this model's template names them: `preserve_reasoning` (keep earlier
        turns' thinking in the prompt) becomes the template's own switch."""
        if "preserve_reasoning" not in kwargs:
            return kwargs
        out = dict(kwargs)
        out["preserve_thinking"] = out.pop("preserve_reasoning")
        return out

    def effort_level(self, level):
        """The frontend's reasoning level (low, medium, high, max or a number) as this model's template takes it."""
        if isinstance(level, int):
            raise ValueError(f"unknown reasoning effort {level!r}: use none, low, medium, high or max")
        return "xhigh" if level in ("high", "max") else level

    def template(self, source: str):
        """The prompt renderer: an object with `render(messages, tools=, add_generation_prompt=, **kwargs)` and
        `source`. `source` is the model's exported template text, which a dialect may ignore."""
        raise NotImplementedError

    def parser(self, thinking: bool, tools: list[dict] | None, stream_tools: bool = True):
        """An incremental output parser: `feed(text)` and `finish()` return events; `state` and `buf` tell the
        server where it is."""
        raise NotImplementedError

    def literal_tags(self, controls) -> dict[str, str]:
        """Texts that stay plain text inside a message, with the private-use mark standing for each while the
        prompt is rendered."""
        raise NotImplementedError

    def video_text(self, times: list[float], rows: int) -> str:
        """The prompt text for a video in place of video_marker: per two-frame group its time (seconds) and its run of
        `rows` video pads."""
        raise NotImplementedError

    def forced_opening(self, name: str | None) -> str:
        """The text that opens a call to tool `name` (None: any tool), for a forced tool choice."""
        raise NotImplementedError


def dialect_for(arch: str | None) -> Dialect:
    """The dialect of a model file's architecture; None (no file: the mock engine) is the plain chat-template one."""
    if arch == "deepseek41":
        from serve.models.deepseek41 import DeepSeek41
        return DeepSeek41()
    if arch in ("qwen4exp", None):
        from serve.models.qwen4exp import Qwen4Exp
        return Qwen4Exp()
    raise ValueError(f"no dialect for architecture {arch!r}")
