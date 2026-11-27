"""The ERNIE 5.5 Flash dialect: pure-Python prompt renderer matching Baidu's `encoding_ernie.py`,
`<ernie:think>` reasoning, and `<ernie:tool_call>` XML-parameter tool calls.

A reply is optional reasoning ending in `</ernie:think>` (or in the start of a call block), the answer,
then `<ernie:tool_call>` holding `<invoke name="N"><parameter name="K">V</parameter>...</invoke>`
elements: string values are written raw, non-string values as JSON (resolved via tool schema when present).
"""
from __future__ import annotations

import json
import re

from serve import frontend
from serve.frontend import Event, ToolCall
from serve.models import Dialect

PAD = "<|ERNIE_PAD|>"
BEGIN = "<|ERNIE_BEGIN|>"
MASK = "<|ERNIE_MASK|>"
START = "<|ernie:im_start|>"
END = "<|ernie:im_end|>"
SYS_START = "<ernie:system_setting>"
SYS_END = "</ernie:system_setting>"
GLOBAL_START = "<ernie:global_setting>"
GLOBAL_END = "</ernie:global_setting>"
TOOL_LIST_START = "<ernie:tool_list>"
TOOL_LIST_END = "</ernie:tool_list>"
OUTPUT = "<ernie:tool_output>"
OUTPUT_END = "</ernie:tool_output>"
CALL = "<ernie:tool_call>"
CALL_END = "</ernie:tool_call>"
THINK = "<ernie:think>"
THINK_END = "</ernie:think>"

OWN_TOKENS = (
    PAD, BEGIN, MASK, START, END,
    SYS_START, SYS_END, GLOBAL_START, GLOBAL_END,
    TOOL_LIST_START, TOOL_LIST_END, OUTPUT, OUTPUT_END,
    CALL, CALL_END, THINK, THINK_END,
)
MARK0 = 0xF2500                         # private-use marks standing for ERNIE tokens while rendering

INVOKE_START = "<invoke"
INVOKE_END = "</invoke>"
PARAM_START = "<parameter"
PARAM_END = "</parameter>"
INVOKE_HEAD = re.compile(r'^<invoke\s+name="([^"]*)">')
PARAM_HEAD = re.compile(r'^<parameter\s+name="([^"]*)">')

REASONING_EFFORTS = ("low", "medium", "high")


def _json(value, sort_keys: bool = True) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys)


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "".join(parts)


class Ernie55Template:
    """Renders conversations in ERNIE 5.5's chat protocol without Jinja, matching `encoding_ernie.py`."""

    source = "ernie55 encoder (serve/models/ernie55.py)"

    def render(
        self,
        messages,
        tools=None,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        reasoning_effort: str | None = None,
        preserve_thinking: bool = True,
        parallel_tool_calls: bool = True,
        sort_keys: bool = True,
        **_,
    ) -> str:
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("messages must be a non-empty list of message dictionaries")

        thinking_off = enable_thinking in (False, 0, "false")
        wrapped_tools = []
        for t in tools or []:
            if isinstance(t, dict) and "function" not in t and "type" not in t:
                wrapped_tools.append({"type": "function", "function": t})
            else:
                wrapped_tools.append(t)

        # System header block
        parts: list[str] = [MASK, START, "system\n"]
        if messages[0].get("role") == "system":
            sys_text = _content_text(messages[0].get("content"))
            parts.extend([SYS_START, "\n", sys_text, "\n", SYS_END, "\n"])

        parts.append(GLOBAL_START)
        if not thinking_off:
            effort = "high" if reasoning_effort is None else reasoning_effort
            if effort not in REASONING_EFFORTS:
                raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}, got {effort!r}")
            parts.extend(["\nreasoning_effort=", effort])
        if wrapped_tools and parallel_tool_calls not in (False, 0, "false"):
            parts.append("\nparallel_tool_calls=True")
        parts.extend(["\n", GLOBAL_END])

        if wrapped_tools:
            parts.extend([
                "\n", TOOL_LIST_START, "\n[",
                ", ".join(_json(t, sort_keys) for t in wrapped_tools),
                "]\n", TOOL_LIST_END,
            ])
        parts.extend([END, "\n"])

        # Index of the last user message so preserve_thinking=False drops earlier turns' thoughts
        last_user = -1
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                last_user = i

        # Conversation turns
        body: list[str] = []
        n = len(messages)
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                raise ValueError(f"unsupported role {role!r}")
            text = _content_text(msg.get("content"))
            if role == "user" or (role == "system" and i > 0):
                body.extend([START, role, "\n", text, END, "\n"])
            elif role == "assistant":
                reasoning = msg.get("thoughts")
                if not isinstance(reasoning, str):
                    reasoning = msg.get("reasoning_content")
                if not isinstance(reasoning, str):
                    if THINK_END in text:
                        first = text.index(THINK_END)
                        start = text.rfind(THINK, 0, first)
                        reasoning = text[start + len(THINK) if start >= 0 else 0:first]
                        text = text[text.rindex(THINK_END) + len(THINK_END):]
                    else:
                        reasoning = ""
                keep_thought = (not thinking_off) and (preserve_thinking or i > last_user)
                body.extend([START, "assistant\n", THINK])
                if keep_thought and reasoning.strip():
                    body.append(reasoning)
                body.extend([THINK_END, text])

                calls = msg.get("tool_calls")
                if calls:
                    body.append(CALL + "\n")
                    for call in calls:
                        fn = call.get("function") or call
                        body.extend(['<invoke name="', str(fn["name"]), '">'])
                        arguments = fn.get("arguments", {})
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        if not isinstance(arguments, dict):
                            raise TypeError("tool call arguments must be a JSON object or JSON object string")
                        keys = sorted(arguments) if sort_keys else list(arguments)
                        for k in keys:
                            val = arguments[k]
                            val_text = val if isinstance(val, str) else _json(val, sort_keys)
                            body.extend(['<parameter name="', str(k), '">', val_text, PARAM_END])
                        body.append(INVOKE_END + "\n")
                    body.append(CALL_END)
                body.extend([END, "\n"])
            elif role == "tool":
                if i == 0 or messages[i - 1].get("role") != "tool":
                    body.append(START + "tool")
                body.extend(["\n", OUTPUT, text, OUTPUT_END])
                if i == n - 1 or messages[i + 1].get("role") != "tool":
                    body.append(END + "\n")

        body_str = "".join(body).strip("\n")
        if body_str:
            parts.append(body_str)

        if add_generation_prompt:
            prefix = (THINK + THINK_END) if thinking_off else THINK
            parts.extend(["\n", START, "assistant\n", prefix])

        return "".join(parts)


class Ernie55Parser:
    """Incremental parser of ERNIE 5.5 assistant output.

    `state` is "reasoning", "content", or "call"; text is held back while it may still be the
    prefix of a protocol marker.
    """

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = True):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.stream_tools = stream_tools
        self.props = {t.get("name"): frontend._tool_properties(t) for t in tools or [] if isinstance(t, dict)}
        self.sub = "between"            # inside a call block: between | invoke | value
        self.call: ToolCall | None = None
        self.args: dict = {}
        self.first = True
        self.key, self.schema, self.text_value, self.raw = "", None, False, ""
        self.calls_seen = False

    def _args(self, out: list, piece: str) -> None:
        if piece and self.stream_tools:
            out.append(Event("tool_args", piece, call=self.call))

    def _end_call(self, out: list) -> None:
        self._args(out, "}")
        out.append(Event("tool_call", call=ToolCall(name=self.call.name, arguments=self.args, id=self.call.id)))
        self.call = None

    def _reasoning(self, out: list) -> bool:
        marks = [(i, tag) for tag in (THINK, THINK_END, CALL) if (i := self.buf.find(tag)) >= 0]
        if marks:
            i, tag = min(marks, key=lambda x: x[0])
            if i:
                out.append(Event("reasoning", self.buf[:i]))
            self.buf = self.buf[i + len(tag):]
            if tag == THINK_END:
                self.state = "content"
            elif tag == CALL:
                self.state = "call"
            return True
        keep = frontend.hold_back(self.buf, (THINK, THINK_END, CALL))
        if len(self.buf) > keep:
            out.append(Event("reasoning", self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    def _content(self, out: list) -> bool:
        marks = [(i, tag) for tag in (THINK, THINK_END, CALL) if (i := self.buf.find(tag)) >= 0]
        if marks:
            i, tag = min(marks, key=lambda x: x[0])
            if i:
                out.append(Event("content", self.buf[:i]))
            self.buf = self.buf[i + len(tag):]
            if tag == THINK:
                self.state = "reasoning"
            elif tag == CALL:
                self.state = "call"
            return True
        keep = frontend.hold_back(self.buf, (THINK, THINK_END, CALL))
        if len(self.buf) > keep:
            out.append(Event("content", self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return False

    def _call(self, out: list) -> bool:
        if self.sub == "between":
            s = self.buf.lstrip()
            if s.startswith(CALL_END):
                self.buf, self.state, self.calls_seen = s[len(CALL_END):], "content", True
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
            if s and not (
                CALL_END.startswith(s)
                or INVOKE_START.startswith(s)
                or (s.startswith(INVOKE_START) and ">" not in s)
            ):
                out.append(Event("content", CALL))
                self.state, self.calls_seen = "content", True
                self.sub = "between"
                return True
            return False

        if self.sub == "invoke":
            s = self.buf.lstrip()
            if s.startswith(INVOKE_END):
                self.buf, self.sub = s[len(INVOKE_END):], "between"
                self._end_call(out)
                return True
            m = PARAM_HEAD.match(s)
            if m:
                self.key = m.group(1)
                self.schema = self.props.get(self.call.name, {}).get(self.key)
                text, null = frontend._text_kinds(self.schema)
                self.text_value = text and not null
                self.raw = ""
                self._args(out, ("" if self.first else ",") + json.dumps(self.key, ensure_ascii=False) + ":")
                self.first = False
                if self.text_value:
                    self._args(out, '"')
                self.buf, self.sub = s[m.end():], "value"
                return True
            return False

        end = self.buf.find(PARAM_END)
        if end >= 0:
            chunk = self.buf[:end]
            self.raw += chunk
            self.args[self.key] = frontend._decode_param(self.raw, self.schema)
            if self.text_value:
                self._args(out, json.dumps(chunk, ensure_ascii=False)[1:-1] + '"')
            else:
                self._args(out, json.dumps(self.args[self.key], ensure_ascii=False))
            self.buf, self.sub = self.buf[end + len(PARAM_END):], "invoke"
            return True

        keep = frontend.hold_back(self.buf, (PARAM_END,))
        text, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        self.raw += text
        if self.text_value:
            self._args(out, json.dumps(text, ensure_ascii=False)[1:-1])
        return False

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
                out.append(Event("content", CALL + self.buf))
        elif self.buf:
            out.append(Event("reasoning" if self.state == "reasoning" else "content", self.buf))
        self.buf = ""
        return out


class Ernie55(Dialect):
    name = "ernie55"
    stop_texts = (END, MASK)
    reasoning_close = THINK_END
    forced_call_ends_reply = True
    call_newlines = 0
    caps = {
        "supports_tool_calls": True,
        "supports_parallel_tool_calls": True,
        "supports_reasoning_effort": True,
        "supports_system_role": True,
        "supports_preserve_reasoning": True,
    }

    def effort_level(self, level):
        if isinstance(level, int):
            raise ValueError(f"unknown reasoning effort {level!r}: use none, low, medium, high or max")
        return "high" if level in ("high", "max", "xhigh") else level

    def template(self, source):
        return Ernie55Template()

    def parser(self, thinking, tools, stream_tools=True):
        return Ernie55Parser(thinking=thinking, tools=tools, stream_tools=stream_tools)

    def literal_tags(self, controls):
        own = {tok: chr(MARK0 + k) for k, tok in enumerate(OWN_TOKENS)}
        tags = {**frontend.literal_tags(controls), **own}
        return dict(sorted(tags.items(), key=lambda kv: -len(kv[0])))

    def forced_opening(self, name):
        head = f'{CALL}\n<invoke name="'
        return head + (f'{name}">' if name else "")
