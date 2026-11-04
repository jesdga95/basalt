"""Per-model chat dialects: how a model's prompt is rendered, its output parsed and its turn ended.

The server picks one from the model file's architecture; an unknown or missing architecture gets the Qwen dialect.
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

    def template(self, path):
        """The prompt renderer: an object with `render(messages, tools=, add_generation_prompt=, **kwargs)` and
        `source`. `path` is the model's exported template file, which a dialect may ignore."""
        raise NotImplementedError

    def parser(self, thinking: bool, tools: list[dict] | None, stream_tools: bool = True):
        """An incremental output parser: `feed(text)` and `finish()` return events; `state` and `buf` tell the
        server where it is."""
        raise NotImplementedError

    def literal_tags(self, controls) -> dict[str, str]:
        """Texts that stay plain text inside a message, with the private-use mark standing for each while the
        prompt is rendered."""
        raise NotImplementedError

    def forced_opening(self, name: str | None) -> str:
        """The text that opens a call to tool `name` (None: any tool), for a forced tool choice."""
        raise NotImplementedError


def dialect_for(arch: str | None) -> Dialect:
    if arch == "deepseek41":
        from serve.models.deepseek41 import DeepSeek41
        return DeepSeek41()
    from serve.models.qwen4exp import Qwen4Exp
    return Qwen4Exp()
