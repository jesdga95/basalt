"""The Gemma 4 dialect: the model's Jinja chat template, `<|channel>thought` reasoning and `<|tool_call>` calls.

A reply is optional reasoning, `<|channel>thought\\n...<channel|>`, then the answer, then any number of calls
`<|tool_call>call:NAME{key:value,...}<tool_call|>`. A string value is wrapped in `<|"|>`, a number, `true`, `false`
and `null` are bare, and `[...]` and `{...}` nest by the same rules (an object's keys are bare, or wrapped when the
model writes them as the declarations do). After its calls the model writes `<|tool_response>` and stops to wait for
the results. The template has a thinking switch and no levels; it asks for a thought channel with `<|think|>` in the
first system turn and, with thinking off, ends the prompt with the empty channel.

The call grammar follows the Gemma 4 tool parsers of vLLM (Apache-2.0) and llama.cpp (MIT), which read the same text.
"""
from __future__ import annotations

import json
import re

from serve import frontend
from serve.frontend import Event, ToolCall
from serve.models import Dialect

BOS = "<bos>"                           # the template's bos_token: the prompt opens with it
CHANNEL = "<|channel>"
HEAD = CHANNEL + "thought\n"            # what opens the model's thinking
CLOSE = "<channel|>"
CALL = "<|tool_call>"
CALL_END = "<tool_call|>"
QUOTE = '<|"|>'                         # wraps a string value
END_TURN = "<turn|>"
EOS = "<eos>"
RESPONSE = "<|tool_response>"
IMAGE = "<|image>"
IMAGE_PAD = "<|image|>"
IMAGE_END = "<image|>"
OWN_TOKENS = (BOS, EOS, "<pad>", "<|turn>", END_TURN, "<|think|>", CHANNEL, CLOSE, "<|tool>", "<tool|>", CALL, CALL_END,
              RESPONSE, "<tool_response|>", QUOTE, IMAGE, IMAGE_PAD, IMAGE_END, "<|audio>", "<|audio|>", "<audio|>", "<|video|>")
MARK0 = 0xF2500                         # the marks of the dialect's own tokens while a message is rendered
_NUMBER = re.compile(r"-?\d+(\.\d+)?([eE][+-]?\d+)?")
_WORDS = {"true": True, "false": False, "null": None}


def _colon(s: str, i: int):
    """The index of the `:` ending the bare key at `s[i:]`, or None while it may still arrive; a key holds none of the
    structure characters, so one before the colon is a ValueError."""
    for j in range(i, len(s)):
        if s[j] == ":":
            return j
        if s[j] in ",{}[]":
            raise ValueError("an object key is not followed by :")
    return None


def _value(s: str, i: int):
    """(value, end) of the value written at `s[i:]`, or None while the text may still grow into one. A bare value ends
    before the `,`, `}` or `]` that follows it. Text that cannot be a value is a ValueError."""
    if i >= len(s):
        return None
    if s.startswith(QUOTE, i):
        j = s.find(QUOTE, i + len(QUOTE))
        return None if j < 0 else (s[i + len(QUOTE):j], j + len(QUOTE))
    if s[i] == "[":
        pos, items = i + 1, []
        while True:
            if pos >= len(s):
                return None
            if s[pos] == "]":
                return items, pos + 1
            r = _value(s, pos)
            if r is None:
                return None
            items.append(r[0])
            pos = r[1]
            if pos >= len(s):
                return None
            if s[pos] == ",":
                pos += 1
            elif s[pos] != "]":
                raise ValueError("an array item is followed by neither , nor ]")
    if s[i] == "{":
        pos, obj = i + 1, {}
        while True:
            if pos >= len(s):
                return None
            if s[pos] == "}":
                return obj, pos + 1
            if s.startswith(QUOTE, pos):
                j = s.find(QUOTE, pos + len(QUOTE))
                if j < 0:
                    return None
                key, pos = s[pos + len(QUOTE):j], j + len(QUOTE)
            else:
                j = _colon(s, pos)
                if j is None:
                    return None
                key, pos = s[pos:j], j
            if pos >= len(s):
                return None
            if s[pos] != ":":
                raise ValueError("an object key is not followed by :")
            r = _value(s, pos + 1)
            if r is None:
                return None
            obj[key], pos = r
            if pos >= len(s):
                return None
            if s[pos] == ",":
                pos += 1
            elif s[pos] != "}":
                raise ValueError("an object value is followed by neither , nor }")
    ends = [k for k in (s.find(c, i) for c in ",}]") if k >= 0]
    if not ends:
        return None
    raw = s[i:min(ends)].strip()
    if raw in _WORDS:
        return _WORDS[raw], min(ends)
    if _NUMBER.fullmatch(raw):
        return (float(raw) if any(c in raw for c in ".eE") else int(raw)), min(ends)
    raise ValueError(f"{raw!r} is no value")


class Gemma4Parser:
    """Incremental parser of the model's text. `state` is "reasoning", "content" or "call"; text is held back while it
    may still be the start of a marker. A call's arguments are streamed as JSON, a string value as it arrives and any
    other value once it is whole; a call that stops making sense is closed without a "tool_call" and its text is
    dropped up to its end."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = True):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.stream_tools = stream_tools
        self.calls = bool(tools)        # no tools offered (none in the request, or tool_choice "none"): no calls
        self.fresh = thinking           # the thinking's own opening may still arrive (a prompt may have written it)
        self.sub = "name"               # inside a call: name | key | value | string | after | tail | skip
        self.call: ToolCall | None = None
        self.args: dict = {}
        self.first = True               # no argument written yet
        self.key = ""

    # ------------------------------------------------------------------ helpers
    def _args(self, out: list, piece: str) -> None:
        if piece and self.stream_tools:
            out.append(Event("tool_args", piece, call=self.call))

    def _put(self, out: list, value) -> None:
        """Writes the pending key and `value` into the arguments."""
        self.args[self.key] = value
        self._args(out, ("" if self.first else ",") + json.dumps(self.key, ensure_ascii=False) + ":" +
                   json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        self.first = False

    def _close(self, out: list) -> None:
        """The call's arguments closed into valid JSON; no "tool_call" follows."""
        if self.call is not None:
            self._args(out, "}")
            self.call = None
        self.sub = "skip"

    # ------------------------------------------------------------------ text outside a call
    def _reasoning(self, out: list) -> bool:
        if self.fresh:
            if self.buf.startswith(HEAD):
                self.buf = self.buf[len(HEAD):]
            elif HEAD.startswith(self.buf):
                return False
            self.fresh = False
        end = self.buf.find(CLOSE)
        if end >= 0:
            if end:
                out.append(Event("reasoning", self.buf[:end]))
            self.buf, self.state = self.buf[end + len(CLOSE):], "content"
            return True
        keep = frontend.hold_back(self.buf, (CLOSE,))
        if len(self.buf) > keep:
            out.append(Event("reasoning", self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    def _content(self, out: list) -> bool:
        tags = (CHANNEL, CALL) if self.calls else (CHANNEL,)
        found = [(i, t) for t in tags if (i := self.buf.find(t)) >= 0]
        if not found:
            keep = frontend.hold_back(self.buf, tags)
            if len(self.buf) > keep:
                out.append(Event("content", self.buf[:len(self.buf) - keep]))
                self.buf = self.buf[len(self.buf) - keep:]
            return False
        at, tag = min(found)
        if at:
            out.append(Event("content", self.buf[:at]))
        rest = self.buf[at:]
        if tag == CALL:
            self.buf, self.state, self.sub = rest[len(CALL):], "call", "name"
            return True
        if rest.startswith(HEAD):       # a thought channel the model opens in its answer
            self.buf, self.state, self.fresh = rest[len(HEAD):], "reasoning", False
            return True
        if HEAD.startswith(rest):
            self.buf = rest
            return False
        out.append(Event("content", CHANNEL))     # a channel of another name is text
        self.buf = rest[len(CHANNEL):]
        return True

    # ------------------------------------------------------------------ a call
    def _begin(self, out: list, name: str) -> None:
        self.call = ToolCall(name=name, arguments={})
        self.args, self.first = {}, True
        if self.stream_tools:
            out.append(Event("tool_start", call=self.call))
        self._args(out, "{")

    def _end(self, out: list) -> None:
        self._args(out, "}")
        out.append(Event("tool_call", call=ToolCall(name=self.call.name, arguments=self.args, id=self.call.id)))
        self.call, self.sub = None, "tail"

    def _call(self, out: list) -> bool:
        """Advance inside a call; True when more may follow in the buffer."""
        sub = self.sub
        if sub == "name":
            if not self.buf.startswith("call:"):
                if "call:".startswith(self.buf):
                    return False
                out.append(Event("content", CALL))      # not a call: text
                self.state = "content"
                return True
            ends = [i for i in (self.buf.find("{"), self.buf.find(CALL_END)) if i >= 0]
            if not ends:
                return False
            at = min(ends)
            name = self.buf[len("call:"):at].strip()
            if not name:
                self.buf, self.sub = self.buf[at:], "skip"
                return True
            self._begin(out, name)
            if self.buf[at] == "{":
                self.buf, self.sub = self.buf[at + 1:], "key"
            else:                       # no braces: a call with no arguments
                self.buf = self.buf[at:]
                self._end(out)
            return True
        if sub == "key":
            if self.buf.startswith("}"):
                self.buf = self.buf[1:]
                self._end(out)
                return True
            if self.buf.startswith(QUOTE):
                j = self.buf.find(QUOTE, len(QUOTE))
                colon = j + len(QUOTE)
                if j < 0 or colon >= len(self.buf):
                    return False
                key = self.buf[len(QUOTE):j]
            elif QUOTE.startswith(self.buf):
                return False
            else:
                try:
                    colon = _colon(self.buf, 0)
                except ValueError:
                    self._close(out)
                    return True
                if colon is None:
                    return False
                key = self.buf[:colon]
            if self.buf[colon] != ":":
                self._close(out)
                return True
            self.key, self.buf, self.sub = key, self.buf[colon + 1:], "value"
            return True
        if sub == "value":
            if self.buf.startswith(QUOTE):
                self._args(out, ("" if self.first else ",") + json.dumps(self.key, ensure_ascii=False) + ':"')
                self.first = False
                self.buf, self.sub, self.args[self.key] = self.buf[len(QUOTE):], "string", ""
                return True
            if QUOTE.startswith(self.buf):
                return False
            try:
                r = _value(self.buf, 0)
            except ValueError:
                self._close(out)
                return True
            if r is None:
                return False
            self._put(out, r[0])
            self.buf, self.sub = self.buf[r[1]:], "after"
            return True
        if sub == "string":
            end = self.buf.find(QUOTE)
            keep = 0 if end >= 0 else frontend.hold_back(self.buf, (QUOTE,))
            text = self.buf[:end] if end >= 0 else self.buf[:len(self.buf) - keep]
            self.args[self.key] += text
            self._args(out, json.dumps(text, ensure_ascii=False)[1:-1] + ('"' if end >= 0 else ""))
            if end < 0:
                self.buf = self.buf[len(self.buf) - keep:]
                return False
            self.buf, self.sub = self.buf[end + len(QUOTE):], "after"
            return True
        if sub == "after":
            if not self.buf:
                return False
            if self.buf[0] == ",":
                self.buf, self.sub = self.buf[1:], "key"
            elif self.buf[0] == "}":
                self.buf = self.buf[1:]
                self._end(out)
            else:
                self._close(out)
            return True
        if sub == "tail":               # the call's own end follows its closing brace
            if self.buf.startswith(CALL_END):
                self.buf = self.buf[len(CALL_END):]
            elif CALL_END.startswith(self.buf):
                return False
            self.state = "content"
            return True
        end = self.buf.find(CALL_END)   # skip: the rest of a call that made no sense
        if end < 0:
            self.buf = self.buf[len(self.buf) - frontend.hold_back(self.buf, (CALL_END,)):]
            return False
        self.buf, self.state = self.buf[end + len(CALL_END):], "content"
        return True

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
        """End of generation: flush what is held. A call cut off mid-way is closed into valid JSON without a "tool_call"
        (a string value keeps the text it has; a key or a nested value not yet whole is left out); text that was only
        the start of a marker is text."""
        out: list[Event] = []
        if self.state == "call":
            if self.call is not None:
                if self.sub == "string":
                    self.args[self.key] += self.buf
                    self._args(out, json.dumps(self.buf, ensure_ascii=False)[1:-1] + '"')
                self._args(out, "}")
                self.call = None
            elif self.sub == "name" and self.buf.strip():
                out.append(Event("content", CALL + self.buf))
        elif self.buf:
            out.append(Event("reasoning" if self.state == "reasoning" else "content", self.buf))
        self.buf = ""
        return out


class _Renderer:
    """The model's template, with three things done for it.
    Each tool result is tied to its call: the template names a result after the call whose id it carries, and the server
    drops the ids after ordering a turn's results by their calls, so the n-th result of a run gets the id of the turn's
    n-th call (a call with no id gets one made up).
    The server hands the template each tool as its bare function object; this template reads the OpenAI wrapper around
    it, `{"type": "function", "function": {...}}`.
    Every image the template writes as `<|image|>` becomes the block the model reads, `<|image>` ... `<image|>`, which
    the server expands to one pad per row."""

    def __init__(self, source: str):
        self.inner = frontend.ChatTemplate(source)
        self.source = source

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        out, calls, n = [], [], 0
        for m in messages:
            if m.get("role") == "tool":
                if not m.get("tool_call_id") and n < len(calls):
                    m = dict(m, tool_call_id=calls[n]["id"])
                n += 1
            elif m.get("role") == "assistant" and m.get("tool_calls"):
                calls = [dict(c, id=c.get("id") or f"call_{k}") for k, c in enumerate(m["tool_calls"])]
                m, n = dict(m, tool_calls=calls), 0
            else:
                calls, n = [], 0
            out.append(m)
        wrapped = [{"type": "function", "function": t} for t in tools] if tools else tools
        text = self.inner.render(out, tools=wrapped, add_generation_prompt=add_generation_prompt, **kwargs)
        return text.replace(IMAGE_PAD, IMAGE + IMAGE_PAD + IMAGE_END)


class Gemma4(Dialect):
    name = "gemma4"
    stop_texts = (END_TURN, EOS, RESPONSE)  # the turn's end, the end of text, and the wait for the calls' results
    reasoning_close = CLOSE
    call_newlines = 0                   # a call follows the thinking's close or the answer directly
    image_pad = IMAGE_PAD
    image_start = IMAGE
    in_engine_vision = True             # the engine's encoder reads the picture at the size the processor gives it: no sidecar
    # chat_template.jinja: tools and several calls per turn, a thinking switch with no levels, a leading system turn,
    # and `preserve_thinking`
    caps = {"supports_tool_calls": True, "supports_parallel_tool_calls": True, "supports_reasoning_effort": False,
            "supports_system_role": True, "supports_preserve_reasoning": True}

    def effort_level(self, level):
        """The frontend's level as the model takes it: every level thinks, `none` is the thinking switch's."""
        if isinstance(level, int):
            raise ValueError(f"unknown reasoning effort {level!r}: use none, low, medium, high or max")
        return level

    def template_kwargs(self, kwargs):
        """The template's thinking switch, on unless the request turned it off (the template's own default is off); it
        has no levels. The template's `bos_token` too, which opens the prompt."""
        out = super().template_kwargs(kwargs)
        out.pop("reasoning_effort", None)
        out.setdefault("enable_thinking", True)
        out["bos_token"] = BOS
        return out

    def template(self, source):
        return _Renderer(source)

    def parser(self, thinking, tools, stream_tools=True):
        return Gemma4Parser(thinking=thinking, tools=tools, stream_tools=stream_tools)

    def literal_tags(self, controls):
        own = {tok: chr(MARK0 + k) for k, tok in enumerate(OWN_TOKENS)}
        tags = {**frontend.literal_tags(controls), **own}
        return dict(sorted(tags.items(), key=lambda kv: -len(kv[0])))

    def forced_opening(self, name):
        return CALL + "call:" + (name + "{" if name else "")
