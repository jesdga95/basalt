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
_NAME = re.compile(r'\s*\{\s*"name"\s*:\s*("(?:[^"\\]|\\.)*")')   # a call object's own name, first key
_ARGS = re.compile(r'"args"\s*:')


class _Blocks:
    """The block structure of the model's text. `state` is "between" (a block may open), "reasoning", "content",
    "name" (a tool call's name), "call" (its JSON) or "literal_call" (the JSON of a call the request cannot make, passed
    on as text); text is held back while it may still be the start of a marker."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = True):
        self.state = "between"
        self.buf = ""
        self.stream_tools = stream_tools
        self.calls = bool(tools)        # no tools offered (none in the request, or tool_choice "none"): no calls
        self.call: ToolCall | None = None
        self.head = ""                  # the call's `{"name":...,"args":` prefix, once read
        self.raw = ""                   # the call's JSON so far
        self.streamed = 0               # characters of its arguments already streamed
        self.json = frontend.JsonPrefix()   # the arguments read: up to where they can be closed
        self.fed = 0                    # characters of the arguments fed to it
        self.started = False            # its "tool_start" sent (a call with no name in its header waits for its JSON's)

    # ------------------------------------------------------------------ a block's text
    def _text(self, out: list, kind: str) -> bool:
        end = self.buf.find(END)
        # a header or a call marker before the block's end: the model opened the next block without closing this one
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
        if self.buf.startswith(END):    # the end of a block already closed
            self.buf = self.buf[len(END):]
            return True
        if self.buf.startswith(TOOL_JSON):      # a call with no header: its name is in its JSON
            self._open_call(out, "")
            self.buf = self.buf[len(TOOL_JSON):]
            return True
        if not self.buf.startswith(HEADER):
            if any(m.startswith(self.buf) for m in (HEADER, TOOL_JSON, END)):
                return False            # a header, a call marker or an end still arriving
            starts = [i for m in (HEADER, TOOL_JSON, END) if (i := self.buf.find(m)) >= 0]
            stray = min(starts, default=-1)
            text = self.buf if stray < 0 else self.buf[:stray]
            keep = 0 if stray >= 0 else frontend.hold_back(text, (HEADER, TOOL_JSON, END))
            if len(text) > keep:        # text outside a block: shown as it is
                out.append(Event("content", text[:len(text) - keep]))
            self.buf = self.buf[len(text) - keep:]
            return stray >= 0
        rest = self.buf[len(HEADER):]
        if rest.startswith(TOOL_JSON):          # a header with no name: the name is in the call's JSON
            self._open_call(out, "")
            self.buf = rest[len(TOOL_JSON):]
            return True
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
        self.buf, self.state = rest, "name" if self.calls else "content"
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
        if not self.calls:              # the request offers no tools: the call is text, its marker and JSON as written
            out.append(Event("content", TOOL_JSON))
            self.state = "literal_call"
            return
        self.call = ToolCall(name=name, arguments={})
        self.state, self.head, self.raw, self.streamed, self.started = "call", "", "", 0, False
        self.json, self.fed = frontend.JsonPrefix(), 0
        self._start(out)

    def _start(self, out: list) -> None:
        """Sends "tool_start" once the call has a name: its header's, or the one its JSON opens with."""
        if self.started or not self.stream_tools:
            return
        if not self.call.name:
            m = _NAME.match(self.raw)
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
            m = _ARGS.search(self.raw)
            if not m:
                return None
            self.head = self.raw[:m.end()]
        return self.raw[len(self.head):]

    def _stream_args(self, out: list) -> None:
        """Streams the arguments up to where they can be closed into valid JSON (the call object's own closing brace
        and the whitespace around the arguments stay out)."""
        args = self._args_text()
        if args is None:
            return
        self.json.feed(args[self.fed:])
        self.fed = len(args)
        if self.started and self.json.safe > self.streamed:
            out.append(Event("tool_args", args[self.streamed:self.json.safe], call=self.call))
            self.streamed = self.json.safe

    def _call(self, out: list) -> bool:
        end = self.buf.find(END)
        keep = 0 if end >= 0 else frontend.hold_back(self.buf, (END,))
        take = self.buf[:end] if end >= 0 else self.buf[:len(self.buf) - keep]
        self.buf = self.buf[end + len(END):] if end >= 0 else self.buf[len(self.buf) - keep:]
        if self.state == "literal_call":
            if take:
                out.append(Event("content", take))
            if end >= 0:
                self.state = "between"
            return end >= 0
        self.raw += take
        self._start(out)
        self._stream_args(out)
        if end < 0:
            return False
        self._end_call(out)
        self.state = "between"
        return True

    def _end_call(self, out: list) -> None:
        try:
            obj = json.loads(self.raw)
        except ValueError:
            obj = None
        name = obj.get("name") if isinstance(obj, dict) and isinstance(obj.get("name"), str) else ""
        name = name or self.call.name
        args = obj.get("args", {}) if isinstance(obj, dict) else {}
        if not isinstance(args, dict):
            args = {}
        if not name or (not self.call.name and not isinstance(obj, dict)):
            out.append(Event("content", TOOL_JSON + self.raw))      # a marker whose JSON names no function: text
            self.call = None
            return
        self.call.name = name
        if self.stream_tools:
            if not self.started:      # the name came with the end: the call is told whole
                self.started = True
                out.append(Event("tool_start", call=self.call))
            if self.streamed == 0:    # zero-arg call (no "args" key) or args never reached a safe prefix before end
                out.append(Event("tool_args", json.dumps(args, ensure_ascii=False, separators=(",", ":")), call=self.call))
        out.append(Event("tool_call", call=ToolCall(name=name, arguments=args, id=self.call.id)))
        self.call = None

    def _whole(self) -> bool:
        """The call's JSON parses whole: a name and an object of arguments."""
        try:
            obj = json.loads(self.raw)
        except ValueError:
            return False
        return isinstance(obj, dict) and isinstance(obj.get("args"), dict) \
            and bool(obj.get("name") if isinstance(obj.get("name"), str) else self.call.name)

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
        """End of generation: flush what is held. A call cut off mid-way gets no "tool_call", and the arguments it
        streamed are closed into valid JSON (a dangling key, separator or partial literal dropped); text that was only
        the start of a marker is text."""
        out: list[Event] = []
        if self.state == "call":
            pending, self.buf = self.buf, ""    # at most the start of an end marker: not the call's
            if self._whole():           # the JSON is whole, only the block's end is missing: still a call
                self._start(out)
                self._stream_args(out)
                self._end_call(out)
            elif self.stream_tools and self.call is not None and self.started:
                self._stream_args(out)
                close = self.json.closer if self.streamed else "{}"
                if close:
                    out.append(Event("tool_args", close, call=self.call))
            elif not self.call.name and not self.started:
                out.append(Event("content", TOOL_JSON + self.raw + pending))
            self.call = None
        elif self.state == "name":
            if self.buf.strip():
                out.append(Event("content", self.buf))
        elif self.buf and self.buf != HEADER:
            kind = "reasoning" if self.state == "reasoning" else "content"
            out.append(Event(kind, self.buf[len(HEADER):] if self.buf.startswith(HEADER) else self.buf))
        self.buf = ""
        return out


# ------------------------------------------------------------------------------------------------ calls written as XML
# The model sometimes writes a call in a notation it was not given, inside a text block:
#   <invoke name="NAME"><parameter name="KEY">VALUE</parameter>...</invoke>
# bare, or wrapped in <function_calls> and <tool_call>, a parameter with or without string="true". The name and the
# arguments are right; only the notation is off. Such text is a call when it is outside code (a fence or backticks, as
# in the frontend), names a tool of the request and is whole; anything else stays text, so a message that talks
# about these tags is not a call.
XML_OPENERS = ("<tool_call>", "<function_calls>", "<invoke name=")
_INVOKE = re.compile(r'<invoke\s+name=(["\'])([^"\'<>]*)\1\s*>')
_PARAM = re.compile(r'<parameter\s+name=(["\'])([^"\'<>]*)\1(?:\s+string=(["\'])(true|false)\3)?\s*>')
MORE, NO = "more", "no"                 # parse_xml: the text may still become a call; it cannot


def _open_tag(rest: str, pattern, head: str):
    """(match, None) when `rest` starts with the tag; (None, MORE) while it may still arrive; (None, NO)."""
    m = pattern.match(rest)
    if m:
        return m, None
    if head.startswith(rest) or (rest.startswith(head) and ">" not in rest and len(rest) < 400):
        return None, MORE
    return None, NO


def _skip(s: str, pos: int) -> int:
    """`pos` moved past the whitespace there."""
    return len(s) - len(s[pos:].lstrip())


def parse_xml(s: str, tools: dict, final: bool):
    """The calls written at the start of `s` (which begins with an opener): ("call", [(name, params)], consumed), (MORE,)
    when the text may still grow into one, or (NO,). params: [(key, text, string flag or None)]. `final`: no more text
    follows, so calls that are whole are calls even where more could have followed."""
    pos, wrap, calls = 0, [], []
    for tag in ("<tool_call>", "<function_calls>"):     # the wrappers, in this order
        if s.startswith(tag, pos):
            wrap.append(tag)
            pos = _skip(s, pos + len(tag))
        elif tag.startswith(s[pos:]):
            return (NO,) if final else (MORE,)
    while True:
        m, wait = _open_tag(s[pos:], _INVOKE, "<invoke name=")
        if m is None:
            if calls:
                if wait == MORE and not final:
                    return (MORE,)
                break               # the invokes are over: what follows is not one
            return (NO,) if final else (wait,)
        if m.group(2) not in tools:
            return (NO,)
        name, pos, params = m.group(2), pos + m.end(), []
        while True:
            pos = _skip(s, pos)
            if s.startswith("</invoke>", pos):
                pos += len("</invoke>")
                break
            m, wait = _open_tag(s[pos:], _PARAM, "<parameter name=")
            if m is None:
                return (NO,) if final or wait == NO else (MORE,)
            start = at = pos + m.end()
            while True:         # a close counts when a parameter or the invoke's end follows it
                i = s.find("</parameter>", at)
                if i < 0:
                    return (NO,) if final else (MORE,)
                tail = s[_skip(s, i + len("</parameter>")):]
                if tail.startswith(("<parameter ", "</invoke>")):
                    break
                if not tail:
                    if final:
                        return (NO,)
                    return (MORE,)
                if "<parameter ".startswith(tail) or "</invoke>".startswith(tail):
                    if final:
                        return (NO,)
                    return (MORE,)
                at = i + 1      # quoted inside the value
            params.append((m.group(2), s[start:i], m.group(4)))
            pos = i + len("</parameter>")
        calls.append((name, params))
        pos = _skip(s, pos)
        if pos >= len(s):
            if not final:
                return (MORE,)
            break
    for tag in reversed(wrap):
        close = tag.replace("<", "</")
        rest = s[pos:]
        if rest.startswith(close):
            pos = _skip(s, pos + len(close))
        elif close.startswith(rest) and not final:
            return (MORE,)
        # a wrapper left open: the calls inside it are whole, taken as they are
    return "call", calls, pos


class _XmlCalls(frontend.CodeTracker):
    """Turns the XML-written calls in the answer text into call events; every other text goes on as it came."""

    def __init__(self, tools: dict, stream_tools: bool):
        self.tools = tools
        self.stream_tools = stream_tools
        self.buf = ""
        self.pend = False               # buf starts (after whitespace) with an opener that begins a call's text
        self.lead = False               # a call just ended: the whitespace after it is dropped
        self.fence, self.line, self.ticks = "", "", 0

    def _call(self, name: str, params: list) -> list:
        props = frontend._tool_properties(self.tools.get(name))
        args = {}
        for key, text, flag in params:
            if flag == "true":
                args[key] = text
            elif flag == "false":
                try:
                    args[key] = json.loads(text)
                except ValueError:
                    args[key] = text
            else:
                args[key] = frontend._decode_param(text, props.get(key))
        call = ToolCall(name=name, arguments=args)
        out = [Event("tool_start", call=call),
               Event("tool_args", json.dumps(args, ensure_ascii=False, separators=(",", ":")), call=call)] \
            if self.stream_tools else []
        return out + [Event("tool_call", call=call)]

    def push(self, text: str, final: bool = False) -> list:
        if self.pend and not final and ">" not in text:     # nothing that can end a tag came
            self.buf += text
            return []
        self.buf += text
        out: list[Event] = []

        def content(t):
            if t:
                out.append(Event("content", self._track(t)))
        while self.buf:
            if self.lead and not self.pend:
                self.buf = self.buf.lstrip()
                self.lead = not self.buf
                if not self.buf:
                    break
            if not self.pend:
                found = [(i, op) for op in XML_OPENERS if (i := self.buf.find(op)) >= 0]
                if not found:
                    keep = len(self.buf[:len(self.buf) - frontend.hold_back(self.buf, XML_OPENERS)].rstrip())   # a call drops the whitespace before it
                    content(self.buf[:keep])
                    self.buf = self.buf[keep:]
                    break
                i, op = min(found)
                snap = (self.fence, self.line, self.ticks)
                self._track(self.buf[:i])
                quoted = self._in_code()
                self.fence, self.line, self.ticks = snap
                if quoted:                      # an example in code (a fence or backticks) is text
                    content(self.buf[:i + len(op)])
                    self.buf = self.buf[i + len(op):]
                    continue
                cut = len(self.buf[:i].rstrip())
                content(self.buf[:cut])
                self.buf, self.pend = self.buf[cut:], True
            body = self.buf.lstrip()
            ws = len(self.buf) - len(body)
            r = parse_xml(body, self.tools, final)
            if r[0] == MORE:
                break
            self.pend = False
            if r[0] == NO:              # text after all: the whitespace and the opener
                op = next(o for o in XML_OPENERS if body.startswith(o))
                content(self.buf[:ws + len(op)])
                self.buf = self.buf[ws + len(op):]
                continue
            for name, params in r[1]:
                out += self._call(name, params)
            self.buf, self.lead = body[r[2]:], True
            self.fence, self.line, self.ticks = "", "", 0
        if final and self.buf:
            content(self.buf)
            self.buf, self.pend = "", False
        return out


class Inkling2Parser(_Blocks):
    """The block parser, and the XML-written calls of the answer text read as calls (see _XmlCalls) when the request
    offers tools. A request that offers none (tool_choice "none" offers none) gets no calls in either notation: what
    the model writes as one is text."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = True):
        super().__init__(thinking=thinking, tools=tools, stream_tools=stream_tools)
        names = {t.get("name"): t for t in tools or [] if isinstance(t, dict) and t.get("name")}
        self.xml = _XmlCalls(names, stream_tools) if names else None

    def _through(self, events: list[Event], final: bool = False) -> list[Event]:
        if self.xml is None:
            return events
        out: list[Event] = []
        for e in events:
            if e.kind == "content":
                out += self.xml.push(e.text)
            else:                       # another kind of event ends the text before it
                out += self.xml.push("", final=True)
                out.append(e)
        if final:
            out += self.xml.push("", final=True)
        return out

    def feed(self, delta: str) -> list[Event]:
        return self._through(super().feed(delta))

    def finish(self) -> list[Event]:
        return self._through(super().finish(), final=True)


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
    thinks_in_blocks = True             # the grammar guard lets its thinking blocks through, unless thinking is off
    minimal_effort = True               # the template's effort map has minimal (0.1)
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
