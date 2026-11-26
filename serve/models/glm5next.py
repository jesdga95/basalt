"""The GLM-5.3-Flash dialect: the release's Jinja chat template, <think> reasoning, and one block per call:

    <|im_start|>assistant
    <think>REASONING</think>
    ANSWER<tool_call>
    <arg_key>PARAM</arg_key>
    <arg_value>VALUE</arg_value>
    </tool_call>
    <|im_end|>

the form the templatethe form the template's own tool instruction spells out, whose every marker is a single token of the release's
vocabulary (154843..154850). The template writes each role as its own token (`<|system|>`, `<|user|>`,
`<|assistant|>`, `<|observation|>`), closes a reply with `<|endoftext|>`, and asks for a tool result to come back
inside `
...
`. A reply is reasoning in a ``
 block, optional because the template switches it off with `enable_thinking`.
"""
from __future__ import annotations

import json

from serve import frontend
from serve.frontend import Event, ToolCall
from serve.models import Dialect

CALL, CALL_END = "<tool_call>", "</tool_call>"
KEY, KEY_END = "<arg_key>", "</arg_key>"
VALUE, VALUE_END = "<arg_value>", "</arg_value>"
THINK, THINK_END = "<think>", "</think>"
# the template wraps a tool result in these; the parser needs no state for them, the tests assert the prompt
RESPONSE, RESPONSE_END = "<tool_response>", "</tool_response>"
# what a held tail is tested against: the parser's own markers, and the reply's end token the server stops on
HOLD = (CALL, CALL_END, KEY, KEY_END, VALUE, VALUE_END, THINK, THINK_END, "<think>", "</think>", "<|endoftext|>", "<|user|>",
        "<|observation|>")


class GlmParser:
    """Incremental parser of the model's text. `state` is "reasoning", "content" or "call"; text is held back while
    it may still be the start of a marker, so a tag split across deltas is never read as the answer."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = True):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.stream_tools = stream_tools
        self.props = {t.get("name"): frontend._tool_properties(t) for t in tools or []}
        self.call: ToolCall | None = None
        self.args: dict = {}
        self.key, self.schema, self.raw = "", None, ""
        self.first = True                  # no argument written yet
        self.lead = True                   # nothing of the answer written yet: its leading newlines are format
        self.calls_seen = False

    # ------------------------------------------------------------------ helpers
    def _text(self, out: list, kind: str, upto: int) -> None:
        if upto > 0:
            out.append(Event(kind, self.buf[:upto]))
            self.buf = self.buf[upto:]

    def _emit(self, out: list, upto: int) -> None:
        """Report buf[:upto] as the current state's text."""
        if upto > 0 and self.state == "content":
            self.lead = False                       # the answer has begun: from here on, text is text
        self._text(out, self.state, upto)

    def _start(self, out: list) -> None:
        """The call as it begins: the client gets its index before the argument pieces stream into it."""
        if self.stream_tools:
            out.append(Event("tool_start", call=self.call))

    def _frag(self, out: list, name: str, value) -> None:
        """One argument as a piece of the JSON object the client is streamed: the pieces of a call join into it."""
        if not self.stream_tools:
            return
        piece = json.dumps(name, ensure_ascii=False) + ": " + json.dumps(value, ensure_ascii=False)
        out.append(Event("tool_args", ("{" if self.first else ", ") + piece, call=self.call))
        self.first = False

    def _commit(self, out: list) -> None:
        """Close the current argument: a string stays as written, anything else the schema's type reads it as."""
        text = self.raw.strip("\n")
        if self.schema and self.schema.get("type") != "string":
            try:
                value = json.loads(text)
            except ValueError:
                value = text                      # the model wrote a number or object the schema type cannot parse
        else:
            value = text
        self.args[self.key] = value
        self._frag(out, self.key, value)
        self.key, self.schema, self.raw = "", None, ""

    def _end_call(self, out: list) -> None:
        if self.call is not None:
            if self.stream_tools:
                out.append(Event("tool_args", "}" if not self.first else "{}", call=self.call))
            out.append(Event("tool_call", call=ToolCall(name=self.call.name, arguments=self.args,
                                                        id=self.call.id)))
        self.call, self.args, self.first = None, {}, True

    # ------------------------------------------------------------------ outside a call
    def _drop_open(self) -> None:
        """A model that writes the `` the prompt already opened itself: not reasoning text."""
        if self.buf.startswith(THINK):
            self.buf = self.buf[len(THINK):]

    def _reasoning(self, out: list) -> bool:
        """Advance reasoning; True when the state changed and the buffer is worth looking at again."""
        self._drop_open()
        end = self.buf.find(THINK_END)
        if end >= 0:
            self._emit(out, end)                                  # buf now starts at the close itself
            self.buf = self.buf[len(THINK_END):]
            self.state = "content"
            return True
        keep = frontend.hold_back(self.buf, HOLD)
        self._emit(out, len(self.buf) - keep)
        return False

    def _content(self, out: list) -> bool:
        if self.lead:                               # the newline the template puts after the thinking is not text
            head = self.buf.lstrip("\n")
            self.buf = head
            if not self.buf:
                return False
        call, think = self.buf.find(CALL), self.buf.find(THINK)
        if think >= 0 and (call < 0 or think < call):
            self._emit(out, think)                                # buf now starts at the opener
            self.buf = self.buf[len(THINK):]
            self.state = "reasoning"
            return True
        if call >= 0:
            head = self.buf[:call].rstrip("\n")               # the newline ahead of a call is the format's, not text
            if head:
                self.lead = False
            out.append(Event("content", head))
            self.buf, self.state, self.calls_seen = self.buf[call + len(CALL):], "call", True
            return True
        keep = frontend.hold_back(self.buf, HOLD)
        # a run of newlines at the end may be the one the format puts ahead of a call: it waits for what follows
        self._emit(out, len(self.buf[:len(self.buf) - keep].rstrip("\n")))
        return False

    # ------------------------------------------------------------------ a call: name, parameters, end
    def _call(self, out: list) -> bool:
        """Advance inside a call: the name, then each parameter between its own pair of markers, then the call's own
        end. A parameter or an end that the model has not finished writing is waited for, never guessed at, so a
        marker split across two pieces cannot reach past it to a later one. True when the buffer moved on."""
        key_at, call_end_at = self.buf.find(KEY), self.buf.find(CALL_END)
        if call_end_at < 0 and key_at >= 0 and key_at >= len(self.buf) - frontend.hold_back(self.buf, (CALL_END,)):
            return False                        # the tail may still grow into this call's end: not decided yet
        if self.call is None:                       # the function name, up to the first parameter or the end
            if key_at < 0 and call_end_at < 0:
                return False
            if call_end_at >= 0 and (key_at < 0 or call_end_at < key_at):
                self.call = ToolCall(self.buf[:call_end_at].strip(), {})      # a call that takes no arguments
                self.buf = self.buf[call_end_at + len(CALL_END):]
                self._start(out)
                self._end_call(out)
                self.state = "content"
                return True
            self.call = ToolCall(self.buf[:key_at].strip(), {})
            self._start(out)
            self.buf = self.buf[key_at:]            # the parameter's opener stays for the step that reads it
            return True
        if not self.key:                            # the next parameter, or the end of the call
            if call_end_at >= 0 and (key_at < 0 or call_end_at < key_at):
                self.buf = self.buf[call_end_at + len(CALL_END):]
                self._end_call(out)
                self.state = "content"
                return True
            if key_at < 0:
                return False                        # nothing of the next parameter has arrived to choose from
            end = self.buf.find(KEY_END, key_at + len(KEY))
            if end < 0:
                return False
            name = self.buf[key_at + len(KEY):end].strip()
            at = self.buf.find(VALUE, end)          # from the name's end marker, so a split marker is not passed by
            if at < 0:
                return False                        # the value has not opened yet: the name stays unread as well
            self.key = name
            self.schema = (self.props.get(self.call.name) or {}).get(self.key)
            self.buf = self.buf[at + len(VALUE):]   # the value, from here to its own end marker
            return True
        end = self.buf.find(VALUE_END)              # the argument's value, up to its own end marker
        if end < 0:
            keep = frontend.hold_back(self.buf, (VALUE_END,))
            self.raw += self.buf[:len(self.buf) - keep]       # value for certain; a half-written end stays to be read
            self.buf = self.buf[len(self.buf) - keep:]
            return False
        self.raw += self.buf[:end]
        self.buf = self.buf[end + len(VALUE_END):]
        self._commit(out)
        return True

    # ------------------------------------------------------------------ interface
    def feed(self, delta: str) -> list:
        self.buf += delta
        out: list = []
        while True:
            moved = {"reasoning": self._reasoning, "content": self._content, "call": self._call}[self.state](out)
            if not moved:
                return out

    def finish(self) -> list:
        """End of generation: flush what is held. A call cut off mid-way is dropped rather than reported, so an
        incomplete argument never reaches the client as a call."""
        out: list = []
        while self.state == "call" and self._call(out):
            pass
        if self.state == "call":
            self.call, self.args, self.key, self.raw = None, {}, "", ""
            self.state = "content"
        if self.buf:
            self._emit(out, len(self.buf))
        return out


class Glm5Next(Dialect):
    name = "glm5next"
    # generation_config.json's eos_token_id [154820, 154827, 154829] in this vocabulary: the end of a reply, and the
    # two role tokens that open the next turn when the model stops speaking
    stop_texts = ("<|endoftext|>", "<|user|>", "<|observation|>")
    reasoning_close = "</think>"                              # closes a <think> the model left open
    call_newlines = 1                             # one newline between the answer and the first call, none inside
    image_pad = "<|image|>"                       # the pad the template's emit_image() writes, between its own pair
    image_start = "<|begin_of_image|>"
    video_pad = "<|video|>"
    video_marker = "<|begin_of_video|><|video|><|end_of_video|>"   # the template's emit_video()
    # chat_template.jinja: tools and several calls per turn, a leading system turn, and a `reasoning_effort` of only
    # `low` and `high` (anything else is the template's own `max`), which it states in its first line itself. Its
    # `clear_thinking` is the switch for the thinking of turns before the last user one. There is no thinking switch:
    # `` is how the template writes no thinking, and `reasoning_close` closes the block the generation prompt opens.
    caps = {"supports_tool_calls": True, "supports_parallel_tool_calls": True, "supports_reasoning_effort": True,
            "supports_system_role": True, "supports_preserve_reasoning": True}

    def effort_level(self, level):
        """The frontend's level as this template names it: it knows only `low` and `high`, `max` is its own default,
        and a numbered budget splits the way the frontend's automatic effort does."""
        if isinstance(level, int):
            return "low" if level <= 2048 else "high"
        return {"minimal": "low", "low": "low", "medium": "high", "high": "high", "max": "max"}[level]

    def template_kwargs(self, kwargs: dict) -> dict:
        if "preserve_reasoning" not in kwargs:
            return kwargs
        out = dict(kwargs)
        out["clear_thinking"] = not out.pop("preserve_reasoning")
        return out

    def template(self, source):
        return frontend.ChatTemplate(source)

    def parser(self, thinking, tools, stream_tools=True):
        return GlmParser(thinking=thinking, tools=tools, stream_tools=stream_tools)

    def literal_tags(self, controls):
        return frontend.literal_tags(controls)

    def video_text(self, times, rows):
        # the template's emit_video() writes one group's marker; each group keeps its own time ahead of it
        return "".join(f"<{t:.1f} seconds>{self.video_marker}" for t in times)

    def forced_opening(self, name):
        return CALL + (f"\n{name}\n{KEY}" if name else "")
