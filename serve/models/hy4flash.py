"""The Hy4-flash dialect: the model's Jinja chat template, `<think>` reasoning and `<tool_calls>` calls.

Every special token of the template carries one suffix (`:opensource` in the Hugging Face release, another in a
quantized release's template); the dialect reads it from the template text it is given. A reply is optional
reasoning ending in `</think>`, the answer, then `<tool_calls>` holding `<tool_call>NAME<arg_key>K</arg_key>
<arg_value>V</arg_value>...</tool_call>` elements: a string value is written as it is, any other value as JSON.
"""
from __future__ import annotations

import json
import re

from serve import frontend
from serve.frontend import Event, ToolCall
from serve.models import Dialect

OFFICIAL_SUFFIX = ":opensource"
SUFFIX = re.compile(r"""\{%-?\s*set\s+HYTK\s*=\s*(['"])(.*?)\1""")
MARK0 = 0xF2100                         # the marks of the dialect's own tokens while a message is rendered


def suffix_of(source: str) -> str:
    """The token suffix a template names (its HYTK variable)."""
    m = SUFFIX.search(source)
    if m is None:
        raise ValueError("this chat template has no HYTK token suffix: not a Hy4-flash template")
    return m.group(2)


class Tags:
    """The template's special tokens for one suffix."""

    def __init__(self, suffix: str):
        self.suffix = suffix
        self.start, self.middle, self.end = (f"<｜hy_{n}{suffix}｜>" for n in ("start", "middle", "end"))
        self.image_start, self.image_pad, self.image_end = (f"<｜hy_image▁{n}{suffix}｜>" for n in ("start", "pad", "end"))
        self.video_start, self.video_pad, self.video_end = (f"<｜hy_video▁{n}{suffix}｜>" for n in ("start", "pad", "end"))
        self.reasoning_mode = f"<｜reasoning_mode{suffix}｜>"
        (self.think, self.think_end, self.calls, self.calls_end, self.call, self.call_end, self.key, self.key_end,
         self.value, self.value_end, self.response, self.response_end) = (
            f"<{n}{suffix}>" for n in ("think", "/think", "tool_calls", "/tool_calls", "tool_call", "/tool_call",
                                      "arg_key", "/arg_key", "arg_value", "/arg_value", "tool_response",
                                      "/tool_response"))

    def all(self) -> list[str]:
        return [v for k, v in vars(self).items() if k != "suffix"]


class Hy4Parser:
    """Incremental parser of the model's text. `state` is "reasoning", "content" or "call"; text is held back while
    it may still be the start of a marker."""

    def __init__(self, tags: Tags, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = True):
        self.t = tags
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.stream_tools = stream_tools
        self.props = {t.get("name"): frontend._tool_properties(t) for t in tools or []}
        self.sub = "between"            # inside a call block: between | name | args | key | open | value
        self.call: ToolCall | None = None
        self.args: dict = {}
        self.first = True               # no argument written yet
        self.key, self.schema, self.text_value, self.raw = "", None, False, ""
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
        t = self.t
        end, calls = self.buf.find(t.think_end), self.buf.find(t.calls)
        if end >= 0 and (calls < 0 or end < calls):
            if end:
                out.append(Event("reasoning", self.buf[:end]))
            self.buf, self.state = self.buf[end + len(t.think_end):], "content"
            return True
        if calls >= 0:                  # the thinking ran straight into the calls
            if calls:
                out.append(Event("reasoning", self.buf[:calls]))
            self.buf, self.state = self.buf[calls + len(t.calls):], "call"
            return True
        keep = frontend.hold_back(self.buf, (t.think_end, t.calls))
        if len(self.buf) > keep:
            out.append(Event("reasoning", self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    def _content(self, out: list) -> bool:
        calls = self.buf.find(self.t.calls)
        if calls >= 0:
            if calls:
                out.append(Event("content", self.buf[:calls]))
            self.buf, self.state = self.buf[calls + len(self.t.calls):], "call"
            return True
        keep = frontend.hold_back(self.buf, (self.t.calls,))
        if len(self.buf) > keep:
            out.append(Event("content", self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    # ------------------------------------------------------------------ the call block
    def _call(self, out: list) -> bool:
        """Advance inside the call block; True when more may follow in the buffer."""
        t = self.t
        if self.sub == "between":
            s = self.buf.lstrip()
            if s.startswith(t.calls_end):
                self.buf, self.state, self.calls_seen = s[len(t.calls_end):], "content", True
                return True
            if s.startswith(t.call):
                self.buf, self.sub = s[len(t.call):], "name"
                return True
            if s and not (t.calls_end.startswith(s) or t.call.startswith(s)):
                self.buf, self.state, self.calls_seen = t.calls + self.buf, "content", True     # not a call block: text
                out.append(Event("content", self.buf))
                self.buf = ""
            return False
        if self.sub == "name":          # the name ends at the first argument or at the close of the call
            ends = [i for i in (self.buf.find(t.key), self.buf.find(t.call_end)) if i >= 0]
            if not ends:
                return False
            end = min(ends)
            self.call = ToolCall(name=self.buf[:end].strip(), arguments={})
            self.args, self.first, self.sub, self.buf = {}, True, "args", self.buf[end:]
            if self.stream_tools:
                out.append(Event("tool_start", call=self.call))
            self._args(out, "{")
            return True
        if self.sub == "args":
            s = self.buf.lstrip()
            if s.startswith(t.call_end):
                self.buf, self.sub = s[len(t.call_end):], "between"
                self._end_call(out)
                return True
            if s.startswith(t.key):
                self.buf, self.sub, self.key = s[len(t.key):], "key", ""
                return True
            return False                # a tag still arriving
        if self.sub == "key":
            end = self.buf.find(t.key_end)
            if end < 0:
                return False
            self.key, self.buf, self.sub = self.buf[:end], self.buf[end + len(t.key_end):], "open"
            return True
        if self.sub == "open":          # the value's opening follows the key
            s = self.buf.lstrip()
            if not s.startswith(t.value):
                if s and not t.value.startswith(s):     # a key with no value: skip it
                    self.buf, self.sub = s, "args"
                    return True
                return False
            self.buf, self.sub, self.raw = s[len(t.value):], "value", ""
            self.schema = self.props.get(self.call.name, {}).get(self.key)
            text, null = frontend._text_kinds(self.schema)
            self.text_value = text and not null         # streamed as the text it is; any other value is decoded whole
            self._args(out, ("" if self.first else ",") + json.dumps(self.key, ensure_ascii=False) + ":")
            self.first = False
            if self.text_value:
                self._args(out, '"')
            return True
        end = self.buf.find(t.value_end)    # a value
        if end >= 0:
            self.raw += self.buf[:end]
            self.args[self.key] = frontend._decode_param(self.raw, self.schema)
            if self.text_value:
                self._args(out, json.dumps(self.buf[:end], ensure_ascii=False)[1:-1] + '"')
            else:
                self._args(out, json.dumps(self.args[self.key], ensure_ascii=False))
            self.buf, self.sub = self.buf[end + len(t.value_end):], "args"
            return True
        keep = frontend.hold_back(self.buf, (t.value_end,))
        text, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        self.raw += text
        if self.text_value:
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
                    self.raw += self.buf
                    if self.text_value:
                        self._args(out, json.dumps(self.buf, ensure_ascii=False)[1:-1] + '"')
                    else:
                        self._args(out, json.dumps(frontend._decode_param(self.raw, self.schema), ensure_ascii=False))
                self._args(out, "}")
                self.call = None
            elif self.buf.strip() and not self.calls_seen:
                out.append(Event("content", self.t.calls + self.buf))
        elif self.buf:
            out.append(Event("reasoning" if self.state == "reasoning" else "content", self.buf))
        self.buf = ""
        return out


class Hy4Flash(Dialect):
    name = "hy4flash"
    in_engine_vision = True             # the encoder places its rows by the model's own grid (newline column): no sidecar
    # chat_template.jinja: tools and several calls per turn, an effort of no_think/high/max, a leading system turn,
    # and `preserved_thinking`
    caps = {"supports_tool_calls": True, "supports_parallel_tool_calls": True, "supports_reasoning_effort": True,
            "supports_system_role": True, "supports_preserve_reasoning": True}
    call_newlines = 0                   # the template writes the answer and the calls with nothing between

    def __init__(self, suffix: str = OFFICIAL_SUFFIX):
        self.tags = Tags(suffix)
        self._tokens()

    def _tokens(self) -> None:
        t = self.tags
        self.stop_texts = (t.end,)
        self.reasoning_close = t.think_end
        self.image_pad, self.image_start = t.image_pad, t.image_start

    def effort_level(self, level):
        """The frontend's level as the template takes it: `no_think` is the thinking switch's, `max` is its own, and
        every other level is `high`."""
        if isinstance(level, int):
            raise ValueError(f"unknown reasoning effort {level!r}: use none, low, medium, high or max")
        return "max" if level == "max" else "high"

    def template_kwargs(self, kwargs):
        out = dict(kwargs)
        if out.pop("enable_thinking", True) is False:
            out["reasoning_effort"] = "no_think"
        if "preserve_reasoning" in out:
            out["preserved_thinking"] = out.pop("preserve_reasoning")
        return out

    def template(self, source):
        self.tags = Tags(suffix_of(source))
        self._tokens()
        return frontend.ChatTemplate(source)

    def parser(self, thinking, tools, stream_tools=True):
        return Hy4Parser(self.tags, thinking=thinking, tools=tools, stream_tools=stream_tools)

    def literal_tags(self, controls):
        own = {tok: chr(MARK0 + k) for k, tok in enumerate(self.tags.all())}
        tags = {**frontend.literal_tags(controls), **own}
        return dict(sorted(tags.items(), key=lambda kv: -len(kv[0])))

    def forced_opening(self, name):
        return self.tags.calls + self.tags.call + (name or "")
