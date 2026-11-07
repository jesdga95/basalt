"""The DeepSeek-V4.1 dialect: prompts from DeepSeek's own encoder, `<think>` reasoning and DSML tool calls.

The call format follows the encoder (serve/models/deepseek41_encoding.py) and llama.cpp's parser
(common/parsers/deepseek.cpp, MIT): a reply is optional reasoning ending in `</think>` (or in the start of a call
block), the answer, then `<｜DSML｜ calls>` holding `<｜DSML｜ invoke name="N">` elements whose
`<｜DSML｜ parameter name="P" string="true|false">` values are text or JSON.
"""
from __future__ import annotations

import json
import re

from serve import frontend
from serve.frontend import Event, ToolCall
from serve.models import Dialect
from serve.models import deepseek41_encoding as enc

EOS = enc.eos_token
THINK_END = enc.thinking_end_token
CALLS_START = f"<{enc.dsml_token}{enc.tool_calls_block_name}>"
CALLS_END = f"</{enc.dsml_token}{enc.tool_calls_block_name}>"
INVOKE_START = f"<{enc.dsml_token}{enc.tool_call_tag_name}"
INVOKE_END = f"</{enc.dsml_token}{enc.tool_call_tag_name}>"
PARAM_START = f"<{enc.dsml_token}{enc.tool_parameter_tag_name}"
PARAM_END = f"</{enc.dsml_token}{enc.tool_parameter_tag_name}>"
CALLS_GAP = "\n\n"                      # between the answer and the call block
INVOKE_HEAD = re.compile(r'^' + re.escape(INVOKE_START) + r' name="([^"]*)">')
PARAM_HEAD = re.compile(r'^' + re.escape(PARAM_START) + r' name="([^"]*)" string="(true|false)">')
DSML_MARK = "\U000F2000"                # stands for the DSML marker while a message is rendered

# the frontend's levels -> the encoder's scale (low, high, max or 1-100); medium is its default
EFFORTS = {"low": "low", "medium": None, "high": "high", "max": "max"}


class DeepseekTemplate:
    """Renders messages with DeepSeek's encoder, taking the same keyword arguments as the Jinja template."""

    source = "deepseek41 encoder (serve/models/deepseek41_encoding.py)"

    def render(self, messages, tools=None, add_generation_prompt=True, enable_thinking=True, reasoning_effort=None,
               **_):
        msgs = [dict(m) for m in messages]
        if tools:
            wrapped = [{"type": "function", "function": t} for t in tools]
            if msgs and msgs[0].get("role") == "system":
                msgs[0]["tools"] = wrapped
            else:
                msgs.insert(0, {"role": "system", "content": "", "tools": wrapped})
        mode = "thinking" if enable_thinking is not False else "chat"
        effort = EFFORTS.get(reasoning_effort, reasoning_effort)
        text = enc.encode_messages(msgs, thinking_mode=mode, reasoning_effort=effort)
        if not add_generation_prompt and msgs and msgs[-1].get("role") in ("user", "tool", "system"):
            head = enc.ASSISTANT_SP_TOKEN
            cut = text.rfind(head)
            if cut >= 0 and text[cut + len(head):] in (enc.thinking_start_token, THINK_END):
                text = text[:cut]
        return text


def _hold(text: str, tags: tuple[str, ...]) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
    best = 0
    for tag in tags:
        for n in range(len(tag) - 1, best, -1):
            if text.endswith(tag[:n]):
                best = n
                break
    return best


def _value(text: str, is_string: bool):
    if is_string:
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


class DeepseekParser:
    """Incremental parser of the model's text. `state` is "reasoning", "content" or "call"; text is held back
    while it may still be the start of a marker."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = True):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.stream_tools = stream_tools
        self.sub = "between"            # inside a call block: between | invoke | value
        self.call: ToolCall | None = None
        self.args: dict = {}
        self.first = True               # no argument written yet
        self.pname, self.pstring, self.pvalue = "", True, ""
        self.calls_seen = False

    # ------------------------------------------------------------------ helpers
    def _args(self, out: list, piece: str) -> None:
        if piece and self.stream_tools:
            out.append(Event("tool_args", piece, call=self.call))

    def _end_call(self, out: list) -> None:
        self._args(out, "}")
        out.append(Event("tool_call", call=ToolCall(name=self.call.name, arguments=self.args, id=self.call.id)))
        self.call = None

    # ------------------------------------------------------------------ text outside the call block
    def _reasoning(self, out: list) -> bool:
        """Advance reasoning; True when the state changed."""
        end, calls = self.buf.find(THINK_END), self.buf.find(CALLS_START)
        if end >= 0 and (calls < 0 or end < calls):
            if end:
                out.append(Event("reasoning", self.buf[:end]))
            self.buf, self.state = self.buf[end + len(THINK_END):], "content"
            return True
        if calls >= 0:                  # the thinking ran straight into the calls
            text = self.buf[:calls].rstrip()
            if text:
                out.append(Event("reasoning", text))
            self.buf, self.state = self.buf[calls + len(CALLS_START):], "call"
            return True
        keep = _hold(self.buf, (THINK_END, CALLS_GAP + CALLS_START, CALLS_START))
        if len(self.buf) > keep:
            out.append(Event("reasoning", self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    def _content(self, out: list) -> bool:
        calls = self.buf.find(CALLS_START)
        if calls >= 0:
            text = self.buf[:calls].rstrip("\n")
            if text:
                out.append(Event("content", text))
            self.buf, self.state = self.buf[calls + len(CALLS_START):], "call"
            return True
        keep = _hold(self.buf, (CALLS_GAP + CALLS_START, CALLS_START))
        if len(self.buf) > keep:
            out.append(Event("content", self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    # ------------------------------------------------------------------ the call block
    def _call(self, out: list) -> bool:
        """Advance inside the call block; True when more may follow in the buffer."""
        if self.sub == "between":
            s = self.buf.lstrip()
            if s.startswith(CALLS_END):
                self.buf, self.state, self.calls_seen = s[len(CALLS_END):], "content", True
                return True
            m = INVOKE_HEAD.match(s)
            if m:
                self.call = ToolCall(name=m.group(1), arguments={})
                self.args, self.first, self.sub = {}, True, "invoke"
                self.buf = s[m.end():]
                if self.stream_tools:
                    out.append(Event("tool_start", call=self.call))
                self._args(out, "{")
                return True
            if s and not (CALLS_END.startswith(s) or INVOKE_START.startswith(s) or
                          (s.startswith(INVOKE_START) and ">" not in s)):
                self.buf, self.state = CALLS_START + self.buf, "content"     # not a call block: it is text
                self.sub, self.calls_seen = "between", True
                out.append(Event("content", self.buf))
                self.buf = ""
            return False
        if self.sub == "invoke":
            s = self.buf.lstrip()
            if s.startswith(INVOKE_END):
                self.buf, self.sub = s[len(INVOKE_END):], "between"
                self._end_call(out)
                return True
            m = PARAM_HEAD.match(s)
            if m:
                self.pname, self.pstring, self.pvalue = m.group(1), m.group(2) == "true", ""
                self._args(out, ("" if self.first else ",") + json.dumps(m.group(1), ensure_ascii=False) + ":")
                self.first = False
                if self.pstring:
                    self._args(out, '"')
                self.buf, self.sub = s[m.end():], "value"
                return True
            return False                # a tag still arriving
        end = self.buf.find(PARAM_END)  # a value
        if end >= 0:
            text = self.buf[:end]
            self.pvalue += text
            self.args[self.pname] = _value(self.pvalue, self.pstring)
            if self.pstring:
                self._args(out, json.dumps(text, ensure_ascii=False)[1:-1] + '"')
            else:
                self._args(out, json.dumps(self.args[self.pname], ensure_ascii=False))
            self.buf, self.sub = self.buf[end + len(PARAM_END):], "invoke"
            return True
        keep = _hold(self.buf, (PARAM_END,))
        text, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        self.pvalue += text
        if self.pstring:
            self._args(out, json.dumps(text, ensure_ascii=False)[1:-1])
        return False

    # ------------------------------------------------------------------ interface
    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
        out: list[Event] = []
        while True:
            if self.state == "reasoning":
                more = self._reasoning(out)
            elif self.state == "content":
                more = self._content(out)
            else:
                more = self._call(out)
            if not more:
                return out

    def finish(self) -> list[Event]:
        """End of generation: flush what is held. A call cut off mid-way is closed without a "tool_call"; text
        that was only the start of a marker is text."""
        out: list[Event] = []
        if self.state == "call":
            if self.call is not None:
                if self.sub == "value":
                    self.pvalue += self.buf
                    if self.pstring:
                        self._args(out, json.dumps(self.buf, ensure_ascii=False)[1:-1] + '"')
                    else:
                        self._args(out, json.dumps(self.pvalue))
                self._args(out, "}")
                self.call = None
            elif self.buf.strip() and not self.calls_seen:
                out.append(Event("content", CALLS_START + self.buf))
        elif self.buf:
            out.append(Event("reasoning" if self.state == "reasoning" else "content", self.buf))
        self.buf = ""
        return out


class DeepSeek41(Dialect):
    def effort_level(self, level):
        return level

    name = "deepseek41"
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
