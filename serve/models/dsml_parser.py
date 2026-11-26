"""Incremental parser for DeepSeek-style DSML call blocks."""
from __future__ import annotations

import json
import re

from serve import frontend
from serve.frontend import Event, ToolCall


def _value(text: str, is_string: bool):
    if is_string:
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


class DsmlParser:
    """Parse reasoning, answer text, and parameterized calls across arbitrary stream chunks."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = True,
                 *, calls_start: str, calls_end: str, invoke_start: str, invoke_end: str,
                 param_start: str, param_end: str, thinking_end: str = "</think>", calls_gap: str = "\n\n"):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.stream_tools = stream_tools
        self.calls_start = calls_start
        self.calls_end = calls_end
        self.invoke_start = invoke_start
        self.invoke_end = invoke_end
        self.param_start = param_start
        self.param_end = param_end
        self.thinking_end = thinking_end
        self.calls_gap = calls_gap
        self.invoke_head = re.compile(r'^' + re.escape(invoke_start) + r' name="([^"]*)">')
        self.param_head = re.compile(r'^' + re.escape(param_start) + r' name="([^"]*)" string="(true|false)">')
        self.sub = "between"
        self.call: ToolCall | None = None
        self.args: dict = {}
        self.first = True
        self.pname, self.pstring, self.pvalue = "", True, ""
        self.calls_seen = False

    def _args(self, out: list, piece: str) -> None:
        if piece and self.stream_tools:
            out.append(Event("tool_args", piece, call=self.call))

    def _end_call(self, out: list) -> None:
        self._args(out, "}")
        out.append(Event("tool_call", call=ToolCall(name=self.call.name, arguments=self.args, id=self.call.id)))
        self.call = None

    def _reasoning(self, out: list) -> bool:
        end, calls = self.buf.find(self.thinking_end), self.buf.find(self.calls_start)
        if end >= 0 and (calls < 0 or end < calls):
            if end:
                out.append(Event("reasoning", self.buf[:end]))
            self.buf, self.state = self.buf[end + len(self.thinking_end):], "content"
            return True
        if calls >= 0:
            text = self.buf[:calls].rstrip()
            if text:
                out.append(Event("reasoning", text))
            self.buf, self.state = self.buf[calls + len(self.calls_start):], "call"
            return True
        keep = frontend.hold_back(self.buf, (self.thinking_end, self.calls_gap + self.calls_start, self.calls_start))
        if len(self.buf) > keep:
            out.append(Event("reasoning", self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    def _content(self, out: list) -> bool:
        calls = self.buf.find(self.calls_start)
        if calls >= 0:
            text = self.buf[:calls].rstrip("\n")
            if text:
                out.append(Event("content", text))
            self.buf, self.state = self.buf[calls + len(self.calls_start):], "call"
            return True
        keep = frontend.hold_back(self.buf, (self.calls_gap + self.calls_start, self.calls_start))
        if len(self.buf) > keep:
            out.append(Event("content", self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    def _call(self, out: list) -> bool:
        if self.sub == "between":
            s = self.buf.lstrip()
            if s.startswith(self.calls_end):
                self.buf, self.state, self.calls_seen = s[len(self.calls_end):], "content", True
                return True
            m = self.invoke_head.match(s)
            if m:
                self.call = ToolCall(name=m.group(1), arguments={})
                self.args, self.first, self.sub = {}, True, "invoke"
                self.buf = s[m.end():]
                if self.stream_tools:
                    out.append(Event("tool_start", call=self.call))
                self._args(out, "{")
                return True
            if s and not (self.calls_end.startswith(s) or self.invoke_start.startswith(s) or
                          (s.startswith(self.invoke_start) and ">" not in s)):
                self.buf, self.state = self.calls_start + self.buf, "content"
                self.sub, self.calls_seen = "between", True
                out.append(Event("content", self.buf))
                self.buf = ""
            return False
        if self.sub == "invoke":
            s = self.buf.lstrip()
            if s.startswith(self.invoke_end):
                self.buf, self.sub = s[len(self.invoke_end):], "between"
                self._end_call(out)
                return True
            m = self.param_head.match(s)
            if m:
                self.pname, self.pstring, self.pvalue = m.group(1), m.group(2) == "true", ""
                self._args(out, ("" if self.first else ",") + json.dumps(m.group(1), ensure_ascii=False) + ":")
                self.first = False
                if self.pstring:
                    self._args(out, '"')
                self.buf, self.sub = s[m.end():], "value"
                return True
            return False
        end = self.buf.find(self.param_end)
        if end >= 0:
            text = self.buf[:end]
            self.pvalue += text
            self.args[self.pname] = _value(self.pvalue, self.pstring)
            if self.pstring:
                self._args(out, json.dumps(text, ensure_ascii=False)[1:-1] + '"')
            else:
                self._args(out, json.dumps(self.args[self.pname], ensure_ascii=False))
            self.buf, self.sub = self.buf[end + len(self.param_end):], "invoke"
            return True
        keep = frontend.hold_back(self.buf, (self.param_end,))
        text, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        self.pvalue += text
        if self.pstring:
            self._args(out, json.dumps(text, ensure_ascii=False)[1:-1])
        return False

    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
        out: list[Event] = []
        while True:
            more = self._reasoning(out) if self.state == "reasoning" else \
                self._content(out) if self.state == "content" else self._call(out)
            if not more:
                return out

    def finish(self) -> list[Event]:
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
                out.append(Event("content", self.calls_start + self.buf))
        elif self.buf:
            out.append(Event("reasoning" if self.state == "reasoning" else "content", self.buf))
        self.buf = ""
        return out
