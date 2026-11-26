"""The DeepSeek-V4.1 dialect: prompts from DeepSeek's own encoder, `<think>` reasoning and DSML tool calls.

The call format follows the encoder (serve/models/deepseek41_encoding.py) and llama.cpp's parser
(common/parsers/deepseek.cpp, MIT): a reply is optional reasoning ending in `</think>` (or in the start of a call
block), the answer, then `<｜DSML｜ calls>` holding `<｜DSML｜ invoke name="N">` elements whose
`<｜DSML｜ parameter name="P" string="true|false">` values are text or JSON.
"""
from __future__ import annotations

from serve import frontend
from serve.models import Dialect
from serve.models import deepseek41_encoding as enc
from serve.models.dsml_parser import DsmlParser

EOS = enc.eos_token
THINK_END = enc.thinking_end_token
CALLS_START = f"<{enc.dsml_token}{enc.tool_calls_block_name}>"
CALLS_END = f"</{enc.dsml_token}{enc.tool_calls_block_name}>"
INVOKE_START = f"<{enc.dsml_token}{enc.tool_call_tag_name}"
INVOKE_END = f"</{enc.dsml_token}{enc.tool_call_tag_name}>"
PARAM_START = f"<{enc.dsml_token}{enc.tool_parameter_tag_name}"
PARAM_END = f"</{enc.dsml_token}{enc.tool_parameter_tag_name}>"
CALLS_GAP = "\n\n"                      # between the answer and the call block
DSML_MARK = "\U000F2000"                # stands for the DSML marker while a message is rendered

# the frontend's levels -> the encoder's scale (low 50, high 75, max 100, or 1-100); medium is the model's default (high)
EFFORTS = {"low": "low", "medium": None, "high": "high", "max": "max"}


class DeepseekTemplate:
    """Renders messages with DeepSeek's encoder, taking the same keyword arguments as the Jinja template."""

    source = "deepseek41 encoder (serve/models/deepseek41_encoding.py)"

    def render(self, messages, tools=None, add_generation_prompt=True, enable_thinking=True, reasoning_effort=None,
               drop_thinking=True, **_):
        msgs = [dict(m) for m in messages]
        if tools:
            wrapped = [{"type": "function", "function": t} for t in tools]
            if msgs and msgs[0].get("role") == "system":
                msgs[0]["tools"] = wrapped
            else:
                msgs.insert(0, {"role": "system", "content": "", "tools": wrapped})
        mode = "thinking" if enable_thinking is not False else "chat"
        effort = EFFORTS.get(reasoning_effort, reasoning_effort)
        text = enc.encode_messages(msgs, thinking_mode=mode, reasoning_effort=effort, drop_thinking=drop_thinking)
        if not add_generation_prompt and msgs and msgs[-1].get("role") in ("user", "tool", "system"):
            head = enc.ASSISTANT_SP_TOKEN
            cut = text.rfind(head)
            if cut >= 0 and text[cut + len(head):] in (enc.thinking_start_token, THINK_END):
                text = text[:cut]
        return text


class DeepseekParser(DsmlParser):
    """DeepSeek-V4.1's DSML spacing configured on the shared incremental parser."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = True):
        super().__init__(
            thinking=thinking, tools=tools, stream_tools=stream_tools,
            calls_start=CALLS_START, calls_end=CALLS_END,
            invoke_start=INVOKE_START, invoke_end=INVOKE_END,
            param_start=PARAM_START, param_end=PARAM_END,
            thinking_end=THINK_END, calls_gap=CALLS_GAP,
        )


class DeepSeek41(Dialect):
    name = "deepseek41"
    # the encoder: DSML tools with several invokes per block, effort 1-100 or low/high/max, a system message,
    # and `drop_thinking` (off keeps earlier turns' thinking)
    caps = {"supports_tool_calls": True, "supports_parallel_tool_calls": True, "supports_reasoning_effort": True,
            "supports_system_role": True, "supports_preserve_reasoning": True}

    def effort_level(self, level):
        return level

    def template_kwargs(self, kwargs):
        if "preserve_reasoning" not in kwargs:
            return kwargs
        out = dict(kwargs)
        out["drop_thinking"] = not out.pop("preserve_reasoning")
        return out
    in_engine_vision = True
    image_pad = enc.IMAGE_PLACEHOLDER
    image_start = ""
    stop_texts = (EOS,)
    reasoning_close = "\n" + THINK_END
    forced_call_ends_reply = True
    nothink_lead = CALLS_GAP

    def template(self, source):
        return DeepseekTemplate()

    def parser(self, thinking, tools, stream_tools=True):
        return DeepseekParser(thinking=thinking, tools=tools, stream_tools=stream_tools)

    def literal_tags(self, controls):
        tags = {**frontend.literal_tags(controls), enc.dsml_token: DSML_MARK}
        return dict(sorted(tags.items(), key=lambda t: -len(t[0])))

    def forced_opening(self, name):
        head = f"{CALLS_START}\n{INVOKE_START} name=\""
        return head + (f'{name}">\n' if name else "")
