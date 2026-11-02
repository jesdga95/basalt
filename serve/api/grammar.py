"""The `grammar` key of an engine request: what the engine builds its constraint from.

Sent only for a request with tools or a JSON response format, and only when the server option is on or the request
sets strict: true on the schema or on a tool function. A JSON format with tools offered (tool_choice auto) holds the
answer to tool calls or the JSON value; a forced call keeps the tools grammar alone."""
from __future__ import annotations

import json

from serve.frontend import tool_choice_of

REQUEST_KEY = "_grammar"        # in the request dict handed to the engine


def _strict(value) -> bool:
    """True when a tool list (any API shape, namespaces included) or a schema spec has strict: true."""
    if isinstance(value, dict):
        return value.get("strict") is True or _strict(value.get("function")) or _strict(value.get("tools"))
    if isinstance(value, list):
        return any(_strict(v) for v in value)
    return False


def _as_function(tool: dict) -> dict:
    """A tool in the OpenAI function-object shape."""
    return tool if tool.get("type") == "function" and isinstance(tool.get("function"), dict) \
        else {"type": "function", "function": tool}


def grammar_spec(enabled: bool, tools, tool_choice, raw_tools, response_format) -> dict | None:
    """The `grammar` value for a request, or None. `tools`: the tools offered (None when there are none or
    tool_choice is none); `raw_tools`: the request's tools as received, for strict; `response_format`: the chat
    path's shape ({"type": "json_schema", "json_schema": {...}} or {"type": "json_object"}), None for text.
    A JSON format with tools under tool_choice auto: the answer is tool calls or the JSON value."""
    answer = None
    kind = response_format.get("type") if isinstance(response_format, dict) else None
    if kind == "json_schema":
        spec = response_format.get("json_schema") or {}
        if enabled or spec.get("strict") is True:
            answer = {"kind": "json_schema", "schema": spec["schema"]}
    elif kind == "json_object" and enabled:
        answer = {"kind": "json_object"}
    if not tools:
        return answer
    calls = [_as_function(t) for t in tools]
    if answer and tool_choice_of(tool_choice)[0] == "auto":
        return dict(answer, tools=calls)
    if enabled or _strict(raw_tools):
        return {"kind": "tools", "tools": calls, "tool_choice": "auto" if tool_choice is None else tool_choice}
    return None


def unenforced(enabled: bool, response_format) -> bool:
    """True when response_format asks for JSON output but the request would get no grammar key."""
    kind = response_format.get("type") if isinstance(response_format, dict) else None
    return kind in ("json_schema", "json_object") and grammar_spec(enabled, None, None, None, response_format) is None


def engine_key(spec: dict | None) -> str:
    """The engine protocol's ` grammar=<JSON>`: compact, ASCII, no space (the line is split at spaces)."""
    if not spec:
        return ""
    text = json.dumps(spec, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return " grammar=" + text.replace(" ", "\\u0020")

