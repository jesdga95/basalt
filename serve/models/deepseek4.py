"""Serving dialect for DeepSeek-V4 Vision."""
from __future__ import annotations

from serve import frontend
from serve.models import Dialect
from serve.models import deepseek41_encoding as enc
from serve.models.dsml_parser import DsmlParser

EOS = enc.eos_token
THINK_END = enc.thinking_end_token
DSML = "｜DSML｜"
CALLS_START = f"<{DSML}tool_calls>"
CALLS_END = f"</{DSML}tool_calls>"
INVOKE_START = f"<{DSML}invoke"
INVOKE_END = f"</{DSML}invoke>"
PARAM_START = f"<{DSML}parameter"
PARAM_END = f"</{DSML}parameter>"
CALLS_GAP = "\n\n"
DSML_MARK = "\U000F2000"


class Deepseek4Template:
    """Render through the shared DeepSeek encoder, then use the V4 DSML spelling."""

    source = "DeepSeek V4-compatible encoder with V4 DSML tags"

    def render(self, messages, tools=None, add_generation_prompt=True, enable_thinking=True,
               reasoning_effort=None, drop_thinking=True, **_):
        msgs = [dict(m) for m in messages]
        if tools:
            wrapped = [{"type": "function", "function": t} for t in tools]
            if msgs and msgs[0].get("role") == "system":
                msgs[0]["tools"] = wrapped
            else:
                msgs.insert(0, {"role": "system", "content": "", "tools": wrapped})
        mode = "thinking" if enable_thinking is not False else "chat"
        # V4 defaults to low effort. V4.1's numeric effort prefixes and its
        # high/max prompts are not part of the V4 prompt format.
        text = enc.encode_messages(msgs, thinking_mode=mode, reasoning_effort="low",
                                   drop_thinking=drop_thinking)
        text = (text.replace(f"<{DSML} calls>", CALLS_START)
                    .replace(f"</{DSML} calls>", CALLS_END)
                    .replace(f"<{DSML} invoke", INVOKE_START)
                    .replace(f"</{DSML} invoke>", INVOKE_END)
                    .replace(f"<{DSML} parameter", PARAM_START)
                    .replace(f"</{DSML} parameter>", PARAM_END))
        # The upstream V4/V4.1 tool guide asks for full private deliberation;
        # keep the section marker but do not ask the model to expose it.
        text = text.replace(
            "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside "
            "<think>...</think> BEFORE any tool calls or final response.",
            "In thinking mode, use the thinking section before your answer or tool calls.",
        )
        if not add_generation_prompt and msgs and msgs[-1].get("role") in ("user", "tool", "system"):
            head = enc.ASSISTANT_SP_TOKEN
            cut = text.rfind(head)
            if cut >= 0 and text[cut + len(head):] in (enc.thinking_start_token, THINK_END):
                text = text[:cut]
        return text


class Deepseek4Parser(DsmlParser):
    def __init__(self, thinking=True, tools=None, stream_tools=True):
        super().__init__(thinking=thinking, tools=tools, stream_tools=stream_tools,
                         calls_start=CALLS_START, calls_end=CALLS_END,
                         invoke_start=INVOKE_START, invoke_end=INVOKE_END,
                         param_start=PARAM_START, param_end=PARAM_END,
                         thinking_end=THINK_END, calls_gap=CALLS_GAP)


class DeepSeek4(Dialect):
    name = "deepseek4"
    # the encoder: DSML tools, a leading or mid-conversation system message, and `drop_thinking` (off keeps earlier
    # turns' thinking); its reasoning effort is the fixed low prompt, not a template argument
    caps = {"supports_tool_calls": True, "supports_parallel_tool_calls": True, "supports_reasoning_effort": False,
            "supports_system_role": True, "supports_preserve_reasoning": True}
    in_engine_vision = True
    image_pad = enc.IMAGE_PLACEHOLDER
    image_start = ""
    stop_texts = (EOS,)
    reasoning_close = "\n" + THINK_END
    forced_call_ends_reply = True
    nothink_lead = CALLS_GAP

    def template(self, source):
        return Deepseek4Template()

    def parser(self, thinking, tools, stream_tools=True):
        return Deepseek4Parser(thinking=thinking, tools=tools, stream_tools=stream_tools)

    def template_kwargs(self, kwargs):
        if "preserve_reasoning" not in kwargs:
            return kwargs
        out = dict(kwargs)
        out["drop_thinking"] = not out.pop("preserve_reasoning")
        return out

    def literal_tags(self, controls):
        tags = {**frontend.literal_tags(controls), DSML: DSML_MARK}
        return dict(sorted(tags.items(), key=lambda t: -len(t[0])))

    def forced_opening(self, name):
        head = f"{CALLS_START}\n{INVOKE_START} name=\""
        return head + (f'{name}\">\n' if name else "")
