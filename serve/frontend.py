"""Text-only layer between API clients and the engine (no model, no GPU).

  * Normalizes OpenAI and Anthropic requests into the chat template's message form.
  * Renders the model's Jinja chat template as Hugging Face does.
  * Parses streamed output into reasoning, content and `<tool_call><function=NAME><parameter=P>...` tool calls,
    never emitting a partial tag.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


# ------------------------------------------------------------------------------------------------ template
class TemplateRequestError(jinja2.exceptions.TemplateError, ValueError):
    """The template refused the messages; a ValueError so the client gets a 400 with its message."""


class ChatTemplate:
    """The model's chat template, rendered with the Jinja settings of transformers' apply_chat_template."""

    def __init__(self, path: str | Path):
        def raise_exception(message):
            raise TemplateRequestError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        self.source = Path(path).read_text(encoding="utf-8")
        self.template = env.from_string(self.source)

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                    **kwargs)


# ------------------------------------------------------------------------------------------------ requests
def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in
                   ("text", "input_text", None))


IMAGE_PARTS = ("image_url", "input_image", "image")

# client spellings of a thinking level -> the template's low, medium, xhigh (default) or none
EFFORT = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
          "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh", "maximum": "xhigh"}


def effort_kwargs(value) -> dict:
    """A client's reasoning effort as template kwargs; unknown values are a 400."""
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    key = str(value).strip().lower()
    if key not in EFFORT:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, medium or high")
    level = EFFORT[key]
    return {"enable_thinking": False} if level is None else {"reasoning_effort": level}


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget (budget_tokens) -> a level: under 2K low, under 8K medium, else high."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "low" if n < 2048 else "medium" if n < 8192 else "xhigh"}


def _has_image(content) -> bool:
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS for p in content)


def _image_source(part: dict) -> str:
    """An image part's source as a data: URL, http(s) URL or local path (OpenAI, Responses and Anthropic shapes)."""
    if part.get("type") == "image":
        src = part.get("source") or {}
        if src.get("type") == "base64":
            return f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
        return src.get("url") or src.get("path") or ""
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return url or ""


def _parts_of(content):
    """Message content for the template: a string without images, else its list of text and image items."""
    if not _has_image(content):
        return _text_of(content)
    items = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in IMAGE_PARTS:
            items.append({"type": "image", "source": _image_source(part)})
        elif part.get("type") in ("text", "input_text", None) and "text" in part:
            items.append({"type": "text", "text": part.get("text", "")})
    return items


def images_of(messages: list[dict]) -> list[str]:
    """The image sources of the conversation in prompt order."""
    return [item["source"] for m in messages if isinstance(m.get("content"), list)
            for item in m["content"] if item.get("type") == "image"]


# Literal think and vision tags in message text must stay plain text, but the tokenizer would turn them into
# special tokens. Before rendering they are swapped for these private-use marks and encoded as ordinary text.
THINK_TAGS = {"<think>": "\U000F0E01", "</think>": "\U000F0E02"}
# vision markers are special only where the template writes them for an image item
VISION_TAGS = {"<|vision_start|>": "\U000F0E03", "<|image_pad|>": "\U000F0E04", "<|vision_end|>": "\U000F0E05",
               "<|video_pad|>": "\U000F0E06"}
# The tool-call tags are special tokens the tokenizer matches in any text; a quoted one would hand the model the real
# marker. An assistant message keeps its own, since a client replaying raw call text sends the tokens the model wrote.
TOOL_LITERAL_TAGS = {"</tool_call>": "\U000F0E07", "<tool_call>": "\U000F0E08", "<tool_response>": "\U000F0E09",
                     "</tool_response>": "\U000F0E0A"}
LITERAL_TAGS = {**THINK_TAGS, **VISION_TAGS, **TOOL_LITERAL_TAGS}
THINK_MARKS = {v: k for k, v in LITERAL_TAGS.items()}
CONTROL_MARK0 = 0xF0E10      # control-token marks start here


def literal_tags(controls) -> dict[str, str]:
    """LITERAL_TAGS plus a mark for each control token's text in `controls`, longest first like the tokenizer."""
    tags = {**LITERAL_TAGS, **{c: chr(CONTROL_MARK0 + k) for k, c in enumerate(c for c in controls
                                                                                if c not in LITERAL_TAGS)}}
    return dict(sorted(tags.items(), key=lambda t: -len(t[0])))


def _mark(text: str, tags: dict[str, str] = LITERAL_TAGS) -> str:
    for tag, mark in tags.items():
        text = text.replace(tag, mark)
    return text


def _mark_deep(v, tags: dict[str, str] = LITERAL_TAGS):
    if isinstance(v, str):
        return _mark(v, tags)
    if isinstance(v, dict):
        return {k: _mark_deep(x, tags) for k, x in v.items()}
    if isinstance(v, list):
        return [_mark_deep(x, tags) for x in v]
    return v


def _has_tag(v, tags: dict[str, str] = LITERAL_TAGS) -> bool:
    if isinstance(v, str):
        return any(tag in v for tag in tags)
    if isinstance(v, dict):
        return any(_has_tag(x, tags) for x in v.values())
    if isinstance(v, list):
        return any(_has_tag(x, tags) for x in v)
    return False


_TOOL_RESULT_OPEN, _TOOL_RESULT_CLOSE = "<tool_response>", "</tool_response>"


def _keep_tool_result_wrap(content, tags: dict[str, str]):
    """A user message whose whole text is <tool_response>...</tool_response> (a client that formats a tool result the way
    the template does) keeps that outer pair as the markers, since the template tells such a turn from a question by
    exactly that pair; only the text inside is marked. None: not such a message."""
    o, c = _TOOL_RESULT_OPEN, _TOOL_RESULT_CLOSE

    def wrapped(text: str) -> bool:
        body = text.strip()
        return len(body) >= len(o) + len(c) and body.startswith(o) and body.endswith(c)

    def wrap(text: str, first: bool, last: bool) -> str:
        i = text.index(o) + len(o) if first else 0
        j = text.rindex(c) if last else len(text)
        head = _mark(text[:i - len(o)], tags) + o if first else ""
        tail = c + _mark(text[j + len(c):], tags) if last else ""
        return head + _mark(text[i:j], tags) + tail

    if isinstance(content, str):
        return wrap(content, True, True) if wrapped(content) else None
    if not (isinstance(content, list) and content and all(isinstance(x, dict) and isinstance(x.get("text"), str)
                                                          for x in content)):
        return None
    texts = [x["text"] for x in content]
    if not wrapped("".join(texts)) or not (texts[0].lstrip().startswith(o) and texts[-1].rstrip().endswith(c)):
        return None          # the pair is split inside a part: mark it all
    if len(content) == 1:
        return [{**content[0], "text": wrap(texts[0], True, True)}]
    return ([{**content[0], "text": wrap(texts[0], True, False)}] + [_mark_deep(x, tags) for x in content[1:-1]] +
            [{**content[-1], "text": wrap(texts[-1], False, True)}])


def mark_think_literals(messages: list[dict], tools: list[dict] | None, tags: dict[str, str] = LITERAL_TAGS):
    """(messages, tools, changed) with literal tags in their text swapped for `tags`' marks; unchanged objects come
    back when there are none. An assistant message opening with a whole <think>...</think> block keeps it."""
    if not _has_tag(messages, tags) and not _has_tag(tools, tags):
        return messages, tools, False
    out = []
    own = {t: v for t, v in tags.items() if t not in TOOL_LITERAL_TAGS}     # what an assistant message keeps as written
    for m in messages:
        m = dict(m)
        content = m.get("content")
        mine = own if m.get("role") == "assistant" else tags
        for k, v in m.items():
            if k != "role":
                m[k] = _mark_deep(v, mine)
        if m.get("role") == "user":
            kept = _keep_tool_result_wrap(content, tags)
            if kept is not None:
                m["content"] = kept
        if m.get("role") == "assistant" and isinstance(content, str) and content.lstrip().startswith("<think>") \
                and "</think>" in content:
            i, j = content.index("<think>") + len("<think>"), content.index("</think>")
            m["content"] = (content[:i] + _mark(content[i:j], own) + "</think>" +
                            _mark(content[j + len("</think>"):], own))
        out.append(m)
    return out, _mark_deep(tools, tags), True


_THINK_MARK_RE = re.compile("|".join(THINK_MARKS))


def unmark_think_literals(prompt: str, tags: dict[str, str] = LITERAL_TAGS) -> tuple[str, list[tuple[int, int]]]:
    """The prompt with marks restored to tag text, and those tags' (start, end) spans to encode as plain text."""
    marks = THINK_MARKS if tags is LITERAL_TAGS else {v: k for k, v in tags.items()}
    mark_re = _THINK_MARK_RE if tags is LITERAL_TAGS else re.compile("|".join(map(re.escape, marks)))
    out, spans, pos, n = [], [], 0, 0
    for m in mark_re.finditer(prompt):
        out.append(prompt[pos:m.start()])
        n += m.start() - pos
        tag = marks[m.group(0)]
        out.append(tag)
        spans.append((n, n + len(tag)))
        n += len(tag)
        pos = m.end()
    out.append(prompt[pos:])
    return "".join(out), spans


def _late_system_to_user(messages: list[dict]) -> list[dict]:
    """Turn system messages after the first into user messages in place; the template allows system only first,
    and merging would change the prompt start and lose the cache."""
    return [dict(m, role="user") if m.get("role") == "system" and i > 0 else m for i, m in enumerate(messages)]


def _object_list(value, name: str) -> list[dict]:
    """A list of objects from `value` (a double-encoded JSON string is decoded); None is empty, anything else a
    ValueError naming `name`."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError(f"{name} must be a list of objects (a string was sent that is not JSON)") from None
    if not isinstance(value, list) or not all(isinstance(m, dict) for m in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


_TOOL_ORDERS: "OrderedDict[tuple, None]" = OrderedDict()   # orders of tool names seen, least recently used first
_TOOL_ORDERS_LOCK = threading.Lock()
_TOOL_ORDERS_MAX = 64


def stable_tool_order(tools: list[dict], name_of) -> list[dict]:
    """Put `tools` in the order their names were first seen, so a client that lists the same tools in another order
    keeps the prompt prefix (the template renders tools before the system prompt, so one swap re-reads the whole
    conversation). The closest remembered order that shares at least half the names decides: its tools keep its
    order and tools it lacks follow in the order sent, which moves the prefix back the least. A tool set with no such
    relative, a single tool, or duplicate names is left as sent. Based on Strata 1a62f6d."""
    if len(tools) < 2:
        return tools
    names = [name_of(t) for t in tools]
    have = set(names)
    if len(have) != len(names):
        return tools
    with _TOOL_ORDERS_LOCK:
        best, shared = None, 0
        for order in reversed(_TOOL_ORDERS):         # most recent first, so ties go to the newest
            n = len(have.intersection(order))
            if n > shared:
                best, shared = order, n
        if best is None or shared * 2 < len(names):
            result = tuple(names)
        else:
            known = set(best)
            result = tuple([n for n in best if n in have] + [n for n in names if n not in known])
        _TOOL_ORDERS.pop(result, None)
        _TOOL_ORDERS[result] = None
        while len(_TOOL_ORDERS) > _TOOL_ORDERS_MAX:
            _TOOL_ORDERS.popitem(last=False)
    if list(result) == names:
        return tools
    pos = {n: k for k, n in enumerate(result)}
    return sorted(tools, key=lambda t: pos[name_of(t)])


def _tool_list(value, wrapper: str | None) -> list[dict]:
    """A request's tools as named objects, in the OpenAI shape (`wrapper` "function") or Anthropic's (None);
    anything else is a ValueError, no value is no tools. A tool's schema ("parameters" / "input_schema") is an object
    when it is there: a string or a list would otherwise fail on the model's first call of that tool."""
    if not value:
        return []
    shape = ('{"type": "function", "function": {"name": ..., "parameters": {...}}}' if wrapper else
             '{"name": ..., "input_schema": {...}}')
    try:
        tools = _object_list(value, "tools")
    except ValueError:
        raise ValueError(f"tools must be a list of tool objects, each {shape}") from None
    key = "parameters" if wrapper else "input_schema"
    for i, t in enumerate(tools):
        fn = t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ValueError(f"tools[{i}] has no name: each tool must be {shape}")
        schema = fn.get(key)
        if schema is not None and not isinstance(schema, dict):
            raise ValueError(f'tools[{i}] ({fn["name"]}): "{key}" must be an object (the JSON schema of its '
                             f"parameters), not {type(schema).__name__}")
    return stable_tool_order(tools, lambda t: (t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t)["name"])


def openai_to_messages(req: dict) -> tuple[list[dict], list[dict] | None, dict]:
    """OpenAI Chat Completions -> (template messages, template tools, template kwargs)."""
    messages = []
    for m in _object_list(req.get("messages"), "messages"):
        role = m.get("role")
        if role == "developer":
            role = "system"
        out = {"role": role, "content": _parts_of(m.get("content")) if role in ("user", "tool", "assistant") else _text_of(m.get("content"))}
        if m.get("reasoning_content"):
            out["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            calls = []
            for c in _object_list(m["tool_calls"], "tool_calls"):
                fn = c.get("function", c)
                if not isinstance(fn, dict):
                    raise ValueError("tool_calls must be a list of objects (each with a \"function\" object)")
                args = fn.get("arguments")
                if isinstance(args, str):               # the template needs a mapping
                    args = json.loads(args) if args.strip() else {}
                calls.append({"function": {"name": fn.get("name"), "arguments": args or {}}})
            out["tool_calls"] = calls
        messages.append(out)
    tools = [t.get("function", t) if t.get("type") == "function" else t
             for t in _tool_list(req.get("tools"), "function")] or None
    kwargs = {}
    # "reasoning_effort", or Responses-style "reasoning": {"effort": ...}
    reasoning = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else {}
    kwargs.update(effort_kwargs(req.get("reasoning_effort") or reasoning.get("effort")))
    # chat_template_kwargs: enable_thinking / reasoning_effort
    for k, v in (req.get("chat_template_kwargs") or {}).items():
        if k == "enable_thinking" and not v:
            kwargs = {"enable_thinking": False}
        elif k == "reasoning_effort" and "enable_thinking" not in kwargs:
            kwargs.update(effort_kwargs(v))
    return _late_system_to_user(messages), tools, kwargs


BILLING_HEADER = "x-anthropic-billing-header:"


def pin_billing_stamp(system: str) -> str:
    """Pin Claude Code's per-request billing stamp to f's. The header at the start of its system prompt carries a
    `cch=` value that changes on every request and a `cc_version` 4th part that changes on every session; left alone
    they change the prompt ahead of the cached prefix. Only a header at the very start of the system text is
    touched, and only inside its first 160 characters."""
    if not system.startswith(BILLING_HEADER):
        return system
    s = list(system)
    cch = system.find("cch=", len(BILLING_HEADER))
    if 0 <= cch <= 160:
        v, end = cch + 4, system.find(";", cch + 4)
        if end > v and end - v <= 16:
            s[v:end] = "f" * (end - v)
    cv = system.find("cc_version=")
    if 0 <= cv <= 160:
        v, end = cv + len("cc_version="), system.find(";", cv)
        if v < end and end - v <= 64:
            parts = system[v:end].split(".")
            if len(parts) > 3:
                tail = v + len(".".join(parts[:3])) + 1
                s[tail:end] = "f" * (end - tail)
    return "".join(s)


def anthropic_to_messages(req: dict, think_unasked: bool = True) -> tuple[list[dict], list[dict] | None, dict]:
    """Anthropic Messages -> (template messages, tools, kwargs). `think_unasked`: a request that asks nothing about
    thinking gets the template default; False renders it without thinking."""
    messages = []
    system = req.get("system")
    if system:
        messages.append({"role": "system", "content": pin_billing_stamp(_text_of(system))})
    for m in _object_list(req.get("messages"), "messages"):
        content = m.get("content")
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        if m.get("role") == "user" and _has_image(content) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            messages.append({"role": "user", "content": _parts_of(content)})
            continue
        text, reasoning, calls = [], [], []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                text.append(block.get("text", ""))
            elif kind == "thinking":
                reasoning.append(block.get("thinking", ""))
            elif kind == "tool_use":
                calls.append({"function": {"name": block.get("name"), "arguments": block.get("input") or {}}})
            elif kind == "tool_result":
                messages.append({"role": "tool", "content": _text_of(block.get("content"))})
        if text or calls or reasoning:
            out = {"role": m["role"], "content": "".join(text)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
    tools = [{"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}
             for t in _tool_list(req.get("tools"), None)] or None
    kwargs = {}
    # "thinking": {"type": "disabled" | "enabled", "budget_tokens"} and "output_config": {"effort"}
    thinking = req.get("thinking")
    effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        kwargs["enable_thinking"] = False
    elif effort:
        kwargs.update(effort_kwargs(effort))
    elif isinstance(thinking, dict) and thinking.get("budget_tokens"):
        kwargs.update(budget_effort(thinking["budget_tokens"]))
    elif thinking is None and not (req.get("reasoning_budget_tokens") or req.get("thinking_budget_tokens")) and not think_unasked:
        # opt-in thinking: short helper calls that ask for none would otherwise spend their tokens thinking
        kwargs["enable_thinking"] = False
    return _late_system_to_user(messages), tools, kwargs


# ------------------------------------------------------------------------------------------------ output parser
@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])


@dataclass
class Event:
    kind: str                     # "reasoning" | "content" | "tool_call"
    text: str = ""
    call: ToolCall | None = None


THINK_END = "</think>"
CALL_START = "<tool_call>"
CALL_END = "</tool_call>"


PARAM_END = "</parameter>"
FUNC_END = "</function>"


def param_end(text: str, final: bool = False) -> int:
    """Where a parameter value ends: the first `</parameter>` followed by `<parameter=` or `</function>`, so the
    same text inside a value is kept. -1: none yet; -2: candidate awaiting its follower (`final` accepts it)."""
    at = text.find(PARAM_END)
    while at >= 0:
        after = text[at + len(PARAM_END):].lstrip()
        if after.startswith(("<parameter=", FUNC_END)):
            return at
        if not after or "<parameter=".startswith(after) or FUNC_END.startswith(after):
            return at if final else -2
        at = text.find(PARAM_END, at + 1)
    return -1


def call_end(text: str) -> int:
    """Where a call body ends: the `</tool_call>` after its own `</function>` (found via param_end); -1: incomplete.
    A body not in the call format ends at the first `</tool_call>`."""
    s = text.lstrip()
    pos = len(text) - len(s)
    if not s.startswith("<function="):
        return text.find(CALL_END) if not "<function=".startswith(s) else -1
    gt = text.find(">", pos)
    if gt < 0:
        return -1
    pos = gt + 1
    while True:
        rest = text[pos:]
        s = rest.lstrip()
        pos += len(rest) - len(s)
        if s.startswith("<parameter="):
            gt = text.find(">", pos)
            if gt < 0:
                return -1
            end = param_end(text[gt + 1:])
            if end < 0:
                return -1
            pos = gt + 1 + end + len(PARAM_END)
        elif s.startswith(FUNC_END):
            rest = text[pos + len(FUNC_END):]
            s = rest.lstrip()
            if s.startswith(CALL_END):
                return pos + len(FUNC_END) + len(rest) - len(s)
            return -1 if CALL_END.startswith(s) else text.find(CALL_END, pos)
        elif not s or "<parameter=".startswith(s) or FUNC_END.startswith(s):
            return -1
        else:
            return text.find(CALL_END, pos)


def tool_choice_of(tool_choice) -> tuple[str, str | None]:
    """A client's `tool_choice` in any API's shape -> ("auto" | "none" | "required" | "named", name).
    OpenAI: "auto" / "none" / "required", {"type": "function", "function": {"name": N}} or the flat
    {"type": "function", "name": N} (Responses, with an optional "namespace"); Anthropic: {"type": "auto" | "any" |
    "none"} and {"type": "tool", "name": N}.  A value it does not know is a ValueError."""
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        if tool_choice in ("auto", "none", "required"):
            return tool_choice, None
    elif isinstance(tool_choice, dict):
        kind = tool_choice.get("type")
        if kind in ("auto", "none"):
            return kind, None
        if kind in ("any", "required"):
            return "required", None
        if kind in ("function", "tool", "custom"):
            fn = tool_choice.get("function")
            name = (fn.get("name") if isinstance(fn, dict) else None) or tool_choice.get("name")
            if isinstance(name, str) and name:
                space = tool_choice.get("namespace")
                return "named", f"{space}.{name}" if isinstance(space, str) and space else name
            raise ValueError("tool_choice: a named choice needs the tool's name")
    raise ValueError(f"tool_choice: unsupported value {json.dumps(tool_choice)[:120]}; use auto, none, required "
                     "(Anthropic: any) or a named tool")


def forced_call(tool_choice, tools: list[dict] | None) -> tuple[list[dict] | None, str | None]:
    """`tool_choice` -> (the tools to offer, the text that opens the call the reply must make or None).
    "none" offers no tools.  There is no grammar: the server writes the opening itself, so the model can only go on
    with a call.  "required" / Anthropic "any": any tool (the only one is named); a named tool: that one.  A forced
    choice with no tools, or naming a tool the request lacks, is a ValueError."""
    kind, name = tool_choice_of(tool_choice)
    if kind == "none":
        return None, None
    if kind in ("auto", "none"):
        return tools, None
    names = [t.get("name") for t in tools or []]
    if kind == "required":
        if not names:
            raise ValueError("tool_choice: required needs at least one tool in the request")
        if len(names) == 1:
            name = names[0]
    elif name not in names:
        raise ValueError(f"tool_choice: the request has no tool named {name!r}")
    return tools, CALL_START + (f"\n<function={name}>\n" if name else "\n<function=")


def _text_kinds(schema) -> tuple[bool, bool]:
    """(text, null) for a parameter schema: whether the grammar lets the model write it as raw text (type "string",
    a type list naming "string", or an anyOf/oneOf/one-branch allOf of plain strings and null), and whether null is
    one of its types."""
    if not isinstance(schema, dict):
        return False, False
    t = schema.get("type")
    if isinstance(t, str):
        return t == "string", t == "null"
    if isinstance(t, list):
        return "string" in t, "null" in t
    if t is not None or any(k in schema for k in ("enum", "const", "properties", "items")):
        return False, False
    union = [schema[k] for k in ("anyOf", "oneOf", "allOf") if isinstance(schema.get(k), list)]
    if len(union) != 1 or not union[0] or ("allOf" in schema and len(union[0]) != 1):
        return False, False
    text = null = False
    for b in union[0]:
        bt = b.get("type") if isinstance(b, dict) else None
        types = bt if isinstance(bt, list) else [bt]
        if (not types or any(x not in ("string", "null") for x in types)
                or any(k in b for k in ("enum", "const", "anyOf", "oneOf", "allOf", "properties", "items", "$ref"))):
            return False, False
        text = text or "string" in types
        null = null or "null" in types
    return text, null


def _decode_param(value: str, schema) -> object:
    """A parameter's raw text as its declared value: kept as text when the schema allows a string (`null` is None only
    where null is a type), else JSON-decoded when it parses (text when it does not)."""
    text, null = _text_kinds(schema)
    if text:
        return None if null and value.strip() == "null" else value
    try:
        return json.loads(value)
    except ValueError:
        return value


def parse_tool_call(body: str, schema: dict | None = None) -> ToolCall:
    """`<function=NAME>...</function>` -> ToolCall. Values are JSON-decoded when the schema says non-string
    (without a schema: when they parse as JSON objects, arrays, numbers or booleans)."""
    body = body.strip()
    if not body.startswith("<function=") or ">" not in body:
        raise ValueError("malformed tool call: " + body[:80])
    name = body[len("<function="):body.index(">")]
    rest = body[body.index(">") + 1:]
    props = ((schema or {}).get("parameters") or {}).get("properties") or {}
    args = {}
    while "<parameter=" in rest:
        rest = rest[rest.index("<parameter=") + len("<parameter="):]
        pname = rest[:rest.index(">")]
        rest = rest[rest.index(">") + 1:]
        end = param_end(rest, final=True)
        value = rest[:end] if end >= 0 else rest
        rest = rest[end + len(PARAM_END):] if end >= 0 else ""
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        args[pname] = _decode_param(value, props.get(pname))
    return ToolCall(name=name, arguments=args)


class OutputParser:
    """Incremental parser of model text: feed deltas, get events; a split tag is held until complete."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.lead = False
        self.rtail = ""              # last reasoning character sent
        self.schemas = {t.get("name"): t for t in tools or []}
        aliases = {}
        for name in self.schemas:
            if not isinstance(name, str):
                continue
            if "." in name:
                namespace, member = name.rsplit(".", 1)
                for alias in (f"{namespace}__{member}", f"{namespace}={member}", f"{namespace}_{member}", member):
                    aliases.setdefault(alias, set()).add(name)
            if "__" in name:
                aliases.setdefault(name.replace("__", "_"), set()).add(name)
        self.aliases = {alias: next(iter(names)) for alias, names in aliases.items()
                        if len(names) == 1 and alias not in self.schemas}
        # stream_tools: also report a call while it is written ("tool_start", "tool_args" pieces) before "tool_call"
        self.stream_tools = stream_tools
        self.fence, self.line, self.ticks = "", "", 0     # code fence, current line and backticks of the answer so far
        self._reset_scan()

    def _tool_name(self, name: str) -> str | None:
        """Resolve only an exact declaration or an unambiguous spelling of one."""
        name = name.strip()
        name = self.aliases.get(name, name)
        return name if not self.schemas or name in self.schemas else None

    def _track(self, text: str) -> str:
        """Follow the answer text that has gone out: the open code fence (``` or ~~~), the current line, and the
        backticks of the current paragraph. Returns the text."""
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k < len(parts) - 1:
                line, self.line = self.line + part, ""
                s = line.lstrip()
                if self.fence:
                    if s.startswith(self.fence * 3):
                        self.fence = ""
                elif s[:3] in ("```", "~~~") and (s[0] * 3) not in s[3:]:
                    self.fence = s[0]
                elif not s:
                    self.ticks = 0
                else:
                    self.ticks += line.count("`")
            else:
                self.line += part
        return text

    def _in_code(self) -> bool:
        """The text so far leaves the next character inside a code fence or inline code."""
        return bool(self.fence) or (self.ticks + self.line.count("`")) % 2 == 1

    def _reset_scan(self):
        self.sp = 0                  # consumed part of self.buf (the call body)
        self.ss = "name"             # name -> between -> str|raw -> ... -> done
        self.scall = None            # the ToolCall being streamed
        self.sfirst = True
        self.sval_started = False
        self.sdeclared = {}
        self.spname = ""

    def _scan(self) -> list[Event]:
        """Advance the streaming view of the call body."""
        out = []

        def args(s):
            if s:
                out.append(Event("tool_args", s, call=self.scall))
        while True:
            rest = self.buf[self.sp:]
            if self.ss == "name":
                a = rest.find("<function=")
                b = rest.find(">", a + 10) if a >= 0 else -1
                if b < 0:
                    return out
                name = self._tool_name(rest[a + 10:b])
                if name is None:
                    return out
                self.scall = ToolCall(name=name, arguments={})
                props = ((self.schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
                self.sdeclared = props
                out.append(Event("tool_start", call=self.scall))
                args("{")
                self.sp += b + 1
                self.ss = "between"
            elif self.ss == "between":
                stripped = rest.lstrip()
                self.sp += len(rest) - len(stripped)
                if stripped.startswith("<parameter="):
                    b = stripped.find(">")
                    if b < 0:
                        return out
                    pname = stripped[11:b]
                    args(("" if self.sfirst else ",") + json.dumps(pname) + ":")
                    self.sfirst = False
                    self.sp += b + 1
                    self.spname = pname
                    text, null = _text_kinds(self.sdeclared.get(pname))
                    if text and not null:
                        args('"')
                        self.ss, self.sval_started = "str", False
                    else:
                        self.ss = "raw"
                elif stripped.startswith("</function>"):
                    args("}")
                    self.sp += len("</function>")
                    self.ss = "done"
                else:
                    return out          # a tag still arriving: wait
            elif self.ss == "str":
                if not self.sval_started:
                    if not rest:
                        return out
                    if rest[0] == "\n":          # the leading newline is not part of the value
                        self.sp += 1
                        rest = rest[1:]
                    self.sval_started = True
                end = param_end(rest)
                if end >= 0:
                    value = rest[:end]
                    if value.endswith("\n"):
                        value = value[:-1]
                    args(json.dumps(value)[1:-1] + '"')
                    self.sp += end + len(PARAM_END)
                    self.ss = "between"
                    continue
                # an undecided </parameter> is held from its start
                safe = rest.find(PARAM_END) if end == -2 else len(rest) - self._hold(rest, (PARAM_END,))
                if safe > 0 and rest[safe - 1] == "\n":   # maybe the newline before </parameter>
                    safe -= 1
                if safe > 0:
                    args(json.dumps(rest[:safe])[1:-1])
                    self.sp += safe
                return out
            elif self.ss == "raw":
                end = param_end(rest)
                if end < 0:
                    return out
                value = rest[:end]
                if value.startswith("\n"):
                    value = value[1:]
                if value.endswith("\n"):
                    value = value[:-1]
                args(json.dumps(_decode_param(value, self.sdeclared.get(self.spname)), ensure_ascii=False))
                self.sp += end + len(PARAM_END)
                self.ss = "between"
            else:
                return out

    def _close_scan(self) -> list[Event]:
        """Close the JSON of a streamed call that ended without a clean </function>."""
        out = []
        if self.scall is None or self.ss == "done":
            return out
        tail = ""
        if self.ss == "str":
            tail += '"'
        elif self.ss == "raw":
            tail += json.dumps(self.buf[self.sp:].strip("\n"))
        tail += "}"
        out.append(Event("tool_args", tail, call=self.scall))
        self.ss = "done"
        return out

    def _reasoning(self, text: str) -> Event:
        """A reasoning event for text going out; remembers its last character for the quote check."""
        self.rtail = text[-1:] or self.rtail
        return Event("reasoning", text)

    def _quoted_think_end(self, i: int) -> bool | None:
        """Whether the `</think>` at self.buf[i] is the model quoting the tag while it thinks: written right after a
        quote or backtick and followed by anything but a line break. None while the character after it has not
        arrived."""
        before = self.buf[i - 1] if i else self.rtail
        if before not in ('"', "'", "`"):
            return False
        after = self.buf[i + len(THINK_END):]
        if not after:
            return None
        return after[0] not in "\r\n"

    def _hold(self, text: str, tags: tuple[str, ...]) -> int:
        """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
        best = 0
        for tag in tags:
            for n in range(1, len(tag)):
                if text.endswith(tag[:n]):
                    best = max(best, n)
        return best

    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
        out: list[Event] = []
        while True:
            if self.state == "reasoning":
                i = self.buf.find(THINK_END)
                if i < 0:
                    keep = self._hold(self.buf, (THINK_END,))
                    if len(self.buf) > keep:
                        out.append(self._reasoning(self.buf[:len(self.buf) - keep]))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                quoted = self._quoted_think_end(i)
                if quoted is None:                   # held like a partial tag until the character after it is here
                    if i:
                        out.append(self._reasoning(self.buf[:i]))
                        self.buf = self.buf[i:]
                    return out
                if quoted:                           # a quoted tag is reasoning text
                    out.append(self._reasoning(self.buf[:i + len(THINK_END)]))
                    self.buf = self.buf[i + len(THINK_END):]
                    continue
                if i:
                    out.append(self._reasoning(self.buf[:i]))
                self.buf = self.buf[i + len(THINK_END):]
                self.state, self.lead = "content", True
                self.fence, self.line, self.ticks = "", "", 0     # the answer's own text starts here
            elif self.state == "content":
                if self.lead:                                   # newlines after </think> or a call
                    stripped = self.buf.lstrip("\n")
                    if not stripped:
                        self.buf = ""
                        return out
                    self.buf, self.lead = stripped, False
                i = self.buf.find(CALL_START)
                if i < 0:
                    # hold a partial tag and the newlines before it: a following tool call drops them
                    j = len(self.buf) - self._hold(self.buf, (CALL_START,))
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                # an opener inside a code fence or inline code is a quoted example: text, decided by the text before it
                snap = (self.fence, self.line, self.ticks)
                self._track(self.buf[:i])
                in_code = self._in_code()
                self.fence, self.line, self.ticks = snap
                if in_code:
                    out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                    self.buf = self.buf[i + len(CALL_START):]
                    continue
                if self.schemas:
                    body = self.buf[i + len(CALL_START):]
                    header = body.lstrip()
                    if not header or "<function=".startswith(header):
                        return out
                    if header.startswith("<function="):
                        gt = header.find(">")
                        if gt < 0:
                            return out
                        if self._tool_name(header[len("<function="):gt]) is None:
                            end = call_end(body)
                            if end < 0:
                                return out
                            end += i + len(CALL_START) + len(CALL_END)
                            out.append(Event("content", self._track(self.buf[:end])))
                            self.buf = self.buf[end:]
                            continue
                if i and self.buf[:i].strip():
                    out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                self.buf = self.buf[i + len(CALL_START):]
                self.state = "call"
            else:
                i = call_end(self.buf)
                if self.stream_tools:
                    if i >= 0:
                        whole, self.buf = self.buf, self.buf[:i]     # scan only the body
                        out += self._scan()
                        out += self._close_scan()
                        self.buf = whole
                    else:
                        out += self._scan()
                if i < 0:
                    return out
                body = self.buf[:i]
                self.buf = self.buf[i + len(CALL_END):]
                name = self._tool_name(body.strip()[len("<function="):].split(">", 1)[0])
                call = parse_tool_call(body, self.schemas.get(name))
                call.name = name
                if self.scall is not None:
                    call.id = self.scall.id
                out.append(Event("tool_call", call=call))
                self._reset_scan()
                self.state, self.lead = "content", True

    def finish(self) -> list[Event]:
        """End of generation: flush what is held (an unterminated call becomes content; an announced one is left
        unfinished with no "tool_call")."""
        out = []
        if self.state == "call" and self.stream_tools and self.scall is not None:
            out += self._scan()                 # ended inside an announced call
            if self.ss == "done":               # only </tool_call> is missing
                out.append(Event("tool_call", call=self.scall))
            self.buf = ""
            self._reset_scan()
            return out
        if self.state == "reasoning" and self.buf == THINK_END:     # the output ends on a held tag: the thinking's end
            self.buf, self.state = "", "content"
        if self.buf:
            kind = {"reasoning": "reasoning", "content": "content"}.get(self.state, "content")
            text = self.buf if self.state != "call" else CALL_START + self.buf
            out.append(Event(kind, text))
            self.buf = ""
        return out
