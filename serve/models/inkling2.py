"""The Inkling 2 dialect: the model's Jinja chat template, and a reply made of blocks that each end in `<|end_message|>`.

A reply is a list of blocks, every one opened by `<|message_model|>`: `<|content_thinking|>` reasoning,
`<|content_text|>` text, or a tool call `NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}`. The calls come
last; `<|content_model_end_sampling|>` ends the turn. The template takes `reasoning_effort` as a level name or a number
from 0 to 0.99 and writes it in a system message; it adds no generation prompt, the model opens its own blocks.
"""
from __future__ import annotations

import json
import re

from serve import frontend
from serve.frontend import Event, ToolCall
from serve.models import Dialect

HEADER = "<|message_model|>"
THINK = "<|content_thinking|>"
TEXT = "<|content_text|>"
TOOL_ERROR = "<|content_tool_error|>"
TOOL_TEXT = "<|content_invoke_tool_text|>"
TOOL_JSON = "<|content_invoke_tool_json|>"
END = "<|end_message|>"
END_TURN = "<|content_model_end_sampling|>"
IMAGE = "<|content_image|>"
IMAGE_PAD = "<|unused_200054|>"
OWN_TOKENS = ("<|endoftext|>", "<|message_user|>", HEADER, "<|message_system|>", "<|message_tool|>", TEXT, IMAGE,
              END_TURN, THINK, END, TOOL_ERROR, "<|content_xml|>", "<|begin_of_text|>", TOOL_JSON, TOOL_TEXT, IMAGE_PAD)
MARK0 = 0xF2400                         # the marks of the dialect's own tokens while a message is rendered
MAX_EFFORT = 0.99
_NAME = re.compile(r'"name"\s*:\s*("(?:[^"\\]|\\.)*")')


class Inkling2Parser:
    """Incremental parser of the model's text. `state` is "between" (a block may open), "reasoning", "content",
    "name" (a tool call's name) or "call" (its JSON); text is held back while it may still be the start of a marker."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = True):
        self.state = "between"
        self.buf = ""
        self.stream_tools = stream_tools
        self.call: ToolCall | None = None
        self.head = ""                  # the call's `{"name":...,"args":` prefix, once read
        self.raw = ""                   # the call's JSON so far
        self.streamed = 0               # characters of its arguments already streamed
        self.started = False            # its "tool_start" sent (a bare call waits for the name in its JSON)

    # ------------------------------------------------------------------ a block's text
    def _text(self, out: list, kind: str) -> bool:
        end = self.buf.find(END)
        # the model opened the next block (or wrote a call's JSON marker) without closing this one: it ends here
        cut = min((i for i in (self.buf.find(HEADER), self.buf.find(TOOL_JSON)) if i >= 0), default=-1)
        if cut >= 0 and (end < 0 or cut < end):
            if cut:
                out.append(Event(kind, self.buf[:cut]))
            self.buf, self.state = self.buf[cut:], "between"
            return True
        if end >= 0:
            if end:
                out.append(Event(kind, self.buf[:end]))
            self.buf, self.state = self.buf[end + len(END):], "between"
            return True
        keep = frontend.hold_back(self.buf, (END, HEADER, TOOL_JSON))
        if len(self.buf) > keep:
            out.append(Event(kind, self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    def _between(self, out: list) -> bool:
        if not self.buf:
            return False
        if self.buf.startswith(TOOL_JSON):     # a call with no header and no name: the name is in its JSON
            self._open_call(out, "")
            self.buf = self.buf[len(TOOL_JSON):]
            return True
        if not self.buf.startswith(HEADER):
            if HEADER.startswith(self.buf) or TOOL_JSON.startswith(self.buf):
                return False            # a header or a call marker still arriving
            stray = min((i for i in (self.buf.find(HEADER), self.buf.find(TOOL_JSON)) if i >= 0), default=-1)
            text = self.buf if stray < 0 else self.buf[:stray]
            keep = 0 if stray >= 0 else frontend.hold_back(text, (HEADER, TOOL_JSON))
            if len(text) > keep:        # text outside a block: shown as it is
                out.append(Event("content", text[:len(text) - keep]))
            self.buf = self.buf[len(text) - keep:]
            return stray >= 0
        rest = self.buf[len(HEADER):]
        for tag, state in ((THINK, "reasoning"), (TEXT, "content"), (TOOL_ERROR, "content"), (TOOL_TEXT, "content")):
            if rest.startswith(tag):
                self.buf, self.state = rest[len(tag):], state
                return True
            if tag.startswith(rest):
                return False
        if rest.startswith("<|"):       # another block's marker: not a call
            if any(t.startswith(rest) for t in (TOOL_JSON, END)):
                return False
            self.buf, self.state = rest, "content"
            return True
        self.buf, self.state = rest, "name"
        return True

    # ------------------------------------------------------------------ a tool call
    def _name(self, out: list) -> bool:
        at = self.buf.find(TOOL_JSON)
        if at < 0:
            if END in self.buf:         # a name with no call: text
                end = self.buf.find(END)
                out.append(Event("content", self.buf[:end]))
                self.buf, self.state = self.buf[end + len(END):], "between"
                return True
            return False
        self._open_call(out, self.buf[:at].strip())
        self.buf = self.buf[at + len(TOOL_JSON):]
        return True

    def _open_call(self, out: list, name: str) -> None:
        self.call = ToolCall(name=name, arguments={})
        self.state, self.head, self.raw, self.streamed, self.started = "call", "", "", 0, False
        self._start(out)

    def _start(self, out: list) -> None:
        """Sends "tool_start" once the call has a name: its header's, or the JSON's for a bare call."""
        if self.started or not self.stream_tools:
            return
        if not self.call.name:
            m = _NAME.search(self.raw)
            if not m:
                return
            try:
                self.call.name = json.loads(m.group(1))
            except ValueError:
                return
        self.started = True
        out.append(Event("tool_start", call=self.call))

    def _args_text(self) -> str | None:
        """The arguments' JSON text read so far, once the `{"name":...,"args":` prefix has passed."""
        if not self.head:
            i = self.raw.find('"args":')
            if i < 0:
                return None
            self.head = self.raw[:i + len('"args":')]
        return self.raw[len(self.head):]

    def _call(self, out: list) -> bool:
        end = self.buf.find(END)
        keep = 0 if end >= 0 else frontend.hold_back(self.buf, (END,))
        take = self.buf[:end] if end >= 0 else self.buf[:len(self.buf) - keep]
        self.raw += take
        self.buf = self.buf[end + len(END):] if end >= 0 else self.buf[len(self.buf) - keep:]
        self._start(out)
        args = self._args_text()
        if args is not None and self.stream_tools and self.started:
            # the last character may be the call object's own closing brace: streamed once the block ends
            # trailing whitespace and the last character may be the call object's own closing brace: held back
            upto = len(args) if end >= 0 else max(self.streamed, len(args.rstrip()) - 1)
            if end >= 0 and args.rstrip().endswith("}"):
                upto = len(args.rstrip()) - 1
            if upto > self.streamed:
                out.append(Event("tool_args", args[self.streamed:upto], call=self.call))
                self.streamed = upto
        if end < 0:
            return False
        self._end_call(out)
        self.state = "between"
        return True

    def _end_call(self, out: list) -> None:
        try:
            obj = json.loads(self.raw)
            args = obj.get("args", {}) if isinstance(obj, dict) else {}
            name = obj.get("name") if isinstance(obj, dict) and isinstance(obj.get("name"), str) else self.call.name
        except ValueError:
            args, name = {}, self.call.name
        if not isinstance(args, dict):
            args = {}
        if not (name or self.call.name):        # a bare call whose JSON names nothing: its text, not a call
            out.append(Event("content", TOOL_JSON + self.raw))
            self.call = None
            return
        out.append(Event("tool_call", call=ToolCall(name=name or self.call.name, arguments=args, id=self.call.id)))
        self.call = None

    def _whole(self) -> str | None:
        """The call's name when its JSON parses whole (a name and an object of arguments), else None."""
        try:
            obj = json.loads(self.raw)
        except ValueError:
            return None
        if not isinstance(obj, dict) or not isinstance(obj.get("args", {}), dict):
            return None
        name = obj.get("name") if isinstance(obj.get("name"), str) else self.call.name if self.call else ""
        return name or None

    # ------------------------------------------------------------------ interface
    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
        out: list[Event] = []
        while True:
            if self.state == "between":
                more = self._between(out)
            elif self.state in ("reasoning", "content"):
                more = self._text(out, self.state)
            elif self.state == "name":
                more = self._name(out)
            else:
                more = self._call(out)
            if not more:
                return out

    def finish(self) -> list[Event]:
        """End of generation: flush what is held. A call cut off mid-way is closed without a "tool_call"; text that
        was only the start of a marker is text."""
        out: list[Event] = []
        if self.state == "call":
            whole = self._whole()
            if whole and self.stream_tools and not self.started:
                self.call.name = self.call.name or whole
                self._start(out)
            if self.stream_tools and self.call is not None and self.started:
                args = self._args_text()
                if args is not None:
                    upto = len(args)
                    if whole and args.rstrip().endswith("}"):     # the call object's own closing brace, as in _call
                        upto = len(args.rstrip()) - 1
                    if upto > self.streamed:
                        out.append(Event("tool_args", args[self.streamed:upto], call=self.call))
            if whole:                   # the JSON is whole, only the block's end is missing: still a call
                self._end_call(out)
            self.call = None
        elif self.state == "name":
            if self.buf.strip():
                out.append(Event("content", self.buf))
        elif self.buf and self.buf != HEADER:
            kind = "reasoning" if self.state == "reasoning" else "content"
            out.append(Event(kind, self.buf[len(HEADER):] if self.buf.startswith(HEADER) else self.buf))
        self.buf = ""
        return out


class _Renderer:
    """The pack's template, each tool result named after its call so its header names the tool: the server orders a
    turn's results by their calls (when their ids allow it) and drops the ids, so the n-th result of a run answers the
    turn's n-th call."""

    def __init__(self, source: str):
        self.inner = frontend.ChatTemplate(source)
        self.source = source

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        out, calls, n = [], [], 0
        for m in messages:
            if m.get("role") == "tool":
                if not m.get("name") and not m.get("tool_call_id") and n < len(calls):
                    name = (calls[n].get("function") or {}).get("name")
                    if name:
                        m = dict(m, name=name)
                n += 1
            else:
                calls, n = (m.get("tool_calls") or []) if m.get("role") == "assistant" else [], 0
            out.append(m)
        return self.inner.render(out, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)


class Inkling2(Dialect):
    name = "inkling2"
    stop_texts = (END_TURN,)
    reasoning_close = END               # a thinking block closes with its own end; the model opens the next block
    call_newlines = 0
    image_pad = IMAGE_PAD
    image_start = IMAGE
    in_engine_vision = True             # the engine's encoder reads the picture at its own size: no sidecar
    # chat_template.jinja: tools and several calls per turn, an effort level or number, system messages, and every
    # assistant message's reasoning_content kept in the prompt
    caps = {"supports_tool_calls": True, "supports_parallel_tool_calls": True, "supports_reasoning_effort": True,
            "supports_system_role": True, "supports_preserve_reasoning": True}

    def effort_level(self, level):
        """The frontend's level as the template takes it: the level names are the template's own; a number from 1
        to 100 is that many hundredths, at most 0.99."""
        if isinstance(level, int):
            return min(level / 100.0, MAX_EFFORT)
        return level

    def template_kwargs(self, kwargs):
        out = dict(kwargs)
        if out.pop("enable_thinking", True) is False:
            out["reasoning_effort"] = "none"
        out.pop("preserve_reasoning", None)    # the template keeps every message's reasoning_content
        return out

    def template(self, source):
        return _Renderer(source)

    def parser(self, thinking, tools, stream_tools=True):
        return Inkling2Parser(thinking=thinking, tools=tools, stream_tools=stream_tools)

    def literal_tags(self, controls):
        own = {tok: chr(MARK0 + k) for k, tok in enumerate(OWN_TOKENS)}
        tags = {**frontend.literal_tags(controls), **own}
        return dict(sorted(tags.items(), key=lambda kv: -len(kv[0])))

    def forced_opening(self, name):
        if name is None:
            return HEADER
        return HEADER + name + TOOL_JSON + '{"name":' + json.dumps(name, ensure_ascii=False) + ',"args":'
