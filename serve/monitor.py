"""API request monitor: the recent calls' input, output, reasoning and raw response.

Always on, within RECORDS per model. Self-contained on purpose: the hooks in `serve/handler.py` and `serve/service.py`
are one call each, so a server update touches at most those few lines. The page lives at /api-monitor, data at
/api/monitor, assets at serve/web/monitor.{html,js}; the page uses the app's stylesheet.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque

RECORDS = 100                  # requests kept in memory, per model
CHARS = 131072                 # per field (input, output, reasoning, response, each tool call's arguments)
TOOLS = 128                    # offered tools kept per request
TOOL_PARAMS = 64               # parameters kept per tool
TOOL_TEXT = 240                # characters of a tool's or parameter's description, one line
TOOL_CALLS = 64                # tool calls kept per request


def _truncate(value: str) -> tuple[str, bool]:
    return value[:CHARS], len(value) > CHARS


def _line(value, limit: int = TOOL_TEXT) -> str:
    """A description cut to one bounded line."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _schema_type(spec) -> str:
    if not isinstance(spec, dict):
        return ""
    kind = spec.get("type")
    if isinstance(kind, list):
        kind = " | ".join(str(k) for k in kind)
    if not kind:
        options = spec.get("anyOf") or spec.get("oneOf")
        if isinstance(options, list):
            kind = " | ".join(t for t in (_schema_type(o) for o in options) if t)
        elif "enum" in spec:
            kind = "enum"
    if kind == "array" and isinstance(spec.get("items"), dict) and spec["items"].get("type"):
        kind = f"array of {_schema_type(spec['items'])}"
    return str(kind or "")


def _offered(tools) -> list:
    """The request's tools as [{name, description, params: [{name, type, required, description}]}], bounded."""
    out = []
    for tool in tools[:TOOLS] if isinstance(tools, list) else []:
        if not isinstance(tool, dict):
            continue
        fn = tool["function"] if isinstance(tool.get("function"), dict) else tool      # chat nests it, the others do not
        name = fn.get("name") or tool.get("type")
        if not name:
            continue
        schema = fn.get("parameters") or fn.get("input_schema") or {}
        props = schema.get("properties") if isinstance(schema, dict) else None
        needed = schema.get("required") if isinstance(schema, dict) else None
        needed = needed if isinstance(needed, list) else []
        params = []
        for pname, spec in list((props if isinstance(props, dict) else {}).items())[:TOOL_PARAMS]:
            spec = spec if isinstance(spec, dict) else {}
            params.append({"name": str(pname), "type": _schema_type(spec), "required": pname in needed,
                           "description": _line(spec.get("description"))})
        out.append({"name": str(name), "description": _line(fn.get("description")), "params": params})
    return out


def _argument_text(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return repr(value)


class Monitor:
    """The last RECORDS requests of one model, kept in memory until the server restarts."""

    def __init__(self, svc):
        self.svc = svc
        self.records = deque(maxlen=RECORDS)
        self.trace = threading.local()                  # .record: this request's record (thread per connection)
        self.lock = svc.status_lock

    # ------------------------------------------------------------------ lifecycle
    def begin(self, path: str, req: dict):
        try:
            raw = json.dumps(req, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            raw = repr(req)
        text, cut = _truncate(raw)
        fmt = req.get("response_format")
        record = {"id": uuid.uuid4().hex[:12], "path": path, "model": req.get("model") or self.svc.model,
                  "started_at": time.time(), "finished_at": None, "state": "queued",
                  "stream": bool(req.get("stream")),
                  "response_format": fmt.get("type") if isinstance(fmt, dict) else None,
                  "input": text, "input_truncated": cut,
                  "output": "", "output_truncated": False,
                  "reasoning": "", "reasoning_truncated": False,
                  "response": "", "response_truncated": False,
                  "tools": _offered(req.get("tools")),
                  "tool_calls": [], "tool_calls_truncated": False, "_slots": {},
                  "error": None, "http_status": None, "usage": {}, "timings": {},
                  "wallclock_s": None, "first_token_s": None,
                  "_clock": time.perf_counter()}
        with self.lock:
            self.records.append(record)
        self.trace.record = record
        return record

    def clear(self) -> int:
        """Forget the finished requests; those still running stay until they finish. Returns how many remain."""
        with self.lock:
            running = [r for r in self.records if r["finished_at"] is None]
            for record in running:
                record["_cleared"] = True               # finish() drops it
            self.records.clear()
            self.records.extend(running)
            return len(running)

    def finish(self):
        """Close this request's record; safe to call more than once."""
        record = getattr(self.trace, "record", None)
        if record is None:
            return
        with self.lock:
            if record.get("_cleared") and record in self.records:
                self.records.remove(record)
            if record["finished_at"] is None:
                record["finished_at"] = time.time()
                record["wallclock_s"] = round(time.perf_counter() - record["_clock"], 3)
                if record["http_status"] is None and record["state"] != "error":
                    record["http_status"] = 200          # SSE replies never go through _json
                if record["state"] != "error":
                    record["state"] = "completed"
                if not record["timings"]:
                    last = getattr(self.svc, "last_timings", None)
                    if isinstance(last, dict):
                        record["timings"] = dict(last)
        self.trace.record = None

    # ------------------------------------------------------------------ capture
    def capture(self, items, api: str):
        """Yield the API's chunks unchanged while recording them; `api` is "openai", "anthropic" or "responses"."""
        record = getattr(self.trace, "record", None)
        if record is None:
            yield from items
            return
        try:
            for item in items:
                if item is not None:
                    self._eat(record, api, item)
                yield item
        finally:
            close = getattr(items, "close", None)
            if close is not None:
                close()

    def _call(self, record: dict, slot, call_id=None, name=None, arguments: str = "", whole: bool = False) -> None:
        """Fold one tool-call fragment into the record's call for `slot` (the API's index or item id); lock held."""
        calls = record["tool_calls"]
        call = record["_slots"].get(slot)
        if call is None:
            if len(calls) >= TOOL_CALLS:
                return
            call = {"id": "", "name": "", "arguments": ""}
            calls.append(call)
            record["_slots"][slot] = call
        if call_id:
            call["id"] = call_id
        if name:
            call["name"] = name
        if arguments:
            joined = arguments if whole else call["arguments"] + arguments
            call["arguments"], cut = _truncate(joined)
            record["tool_calls_truncated"] = record["tool_calls_truncated"] or cut

    def _eat(self, record: dict, api: str, item) -> None:
        content = reasoning = ""
        usage = timings = None
        calls = []                                      # (slot, id, name, arguments fragment, whole)
        try:
            if api == "openai" and isinstance(item, dict):
                choice = (item.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content, reasoning = delta.get("content") or "", delta.get("reasoning_content") or ""
                usage, timings = item.get("usage"), item.get("timings")
                for part in delta.get("tool_calls") or []:
                    fn = part.get("function") or {}
                    calls.append((part.get("index", 0), part.get("id"), fn.get("name"), fn.get("arguments") or "", False))
            elif api == "anthropic" and isinstance(item, tuple) and len(item) == 2:
                name, data = item
                data = data if isinstance(data, dict) else {}
                delta = data.get("delta") or {}
                kind = delta.get("type")
                if kind == "text_delta":
                    content = delta.get("text") or ""
                elif kind == "thinking_delta":
                    reasoning = delta.get("thinking") or ""
                elif kind == "input_json_delta":
                    calls.append((data.get("index", 0), None, None, delta.get("partial_json") or "", False))
                block = data.get("content_block") or {}
                if name == "content_block_start" and block.get("type") == "tool_use":
                    calls.append((data.get("index", 0), block.get("id"), block.get("name"), "", False))
                if name == "message_delta":
                    usage = data.get("usage")
            elif api == "responses" and isinstance(item, dict):
                kind = item.get("type")
                if kind == "response.output_text.delta":
                    content = item.get("delta") or ""
                elif kind == "response.reasoning_text.delta":
                    reasoning = item.get("delta") or ""
                elif kind == "response.output_item.added" and (item.get("item") or {}).get("type") == "function_call":
                    it = item["item"]
                    calls.append((item.get("output_index", it.get("id")), it.get("call_id") or it.get("id"), it.get("name"), "", False))
                elif kind == "response.function_call_arguments.delta":
                    calls.append((item.get("output_index", item.get("item_id")), None, None, item.get("delta") or "", False))
                elif kind == "response.output_item.done" and (item.get("item") or {}).get("type") == "function_call":
                    it = item["item"]
                    calls.append((item.get("output_index", it.get("id")), it.get("call_id") or it.get("id"), it.get("name"),
                                  it.get("arguments") or "", True))
                elif kind in ("response.completed", "response.incomplete"):
                    usage = (item.get("response") or {}).get("usage")
        except (AttributeError, IndexError, TypeError, KeyError):
            return
        if not (content or reasoning or usage or timings or calls):
            return
        with self.lock:
            for slot, call_id, name, arguments, whole in calls:
                self._call(record, slot, call_id, name, arguments, whole)
            for field, value in (("output", content), ("reasoning", reasoning)):
                if value:
                    joined = record[field] + value
                    record[field], record[field + "_truncated"] = _truncate(joined)
            if usage:
                record["usage"] = usage
            if timings:
                record["timings"] = timings
            if (content or reasoning or calls) and record["first_token_s"] is None:
                record["first_token_s"] = round(time.perf_counter() - record["_clock"], 3)
                record["state"] = "generating"

    def json(self, code: int, obj) -> None:
        """The handler's JSON reply for the request in flight, if any: raw body, status, usage and errors."""
        record = getattr(self.trace, "record", None)
        if record is None:
            return
        try:
            raw = json.dumps(obj, ensure_ascii=False)
        except (TypeError, ValueError):
            raw = repr(obj)
        body, cut = _truncate(raw)
        error = None
        if code >= 400 and isinstance(obj, dict):
            detail = obj.get("error")
            if isinstance(detail, dict):
                error = {"message": detail.get("message") or detail.get("type") or "error"}
            elif detail:
                error = {"message": str(detail)}
        with self.lock:
            record["http_status"], record["response"], record["response_truncated"] = code, body, cut
            if error:
                record["error"] = error
                record["state"] = "error"
            if isinstance(obj, dict):
                if not record["tool_calls"]:
                    for n, (call_id, name, arguments) in enumerate(self._replied_calls(obj)):
                        self._call(record, n, call_id, name or "tool", arguments, whole=True)
                if isinstance(obj.get("usage"), dict):
                    record["usage"] = obj["usage"]
                if isinstance(obj.get("timings"), dict):
                    record["timings"] = obj["timings"]

    @staticmethod
    def _replied_calls(obj: dict):
        """(id, name, arguments) of the tool calls in a finished reply of any of the three APIs."""
        try:
            for choice in obj.get("choices") or []:
                for call in (choice.get("message") or {}).get("tool_calls") or []:
                    fn = call.get("function") or {}
                    yield call.get("id"), fn.get("name"), _argument_text(fn.get("arguments"))
            for block in obj.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    yield block.get("id"), block.get("name"), _argument_text(block.get("input"))
            for part in obj.get("output") or []:
                if isinstance(part, dict) and part.get("type") == "function_call":
                    yield part.get("call_id") or part.get("id"), part.get("name"), _argument_text(part.get("arguments"))
        except (AttributeError, TypeError):
            return

    # ------------------------------------------------------------------ read side
    def summary(self, record: dict) -> dict:
        return {key: value for key, value in record.items()
                if key not in ("input", "output", "reasoning", "response", "tools", "tool_calls") and not key.startswith("_")}

    def detail(self, record: dict) -> dict:
        out = {key: value for key, value in record.items() if not key.startswith("_")}
        timings = out.get("timings") or {}
        if isinstance(timings, dict) and "predicted_per_second" not in timings:
            generated, decode_ms = timings.get("generated"), timings.get("decode_ms")
            if isinstance(generated, (int, float)) and isinstance(decode_ms, (int, float)) and decode_ms:
                timings = dict(timings, predicted_per_second=round(generated / (decode_ms / 1000), 1))
        out["timings"] = timings
        return out

    def snapshot(self, request_id: str | None = None):
        """This model's records, newest first; with an id, that one record in full, or None."""
        with self.lock:
            records = list(self.records)
            in_flight = len(getattr(self.svc, "live_reqs", {}) or {})
        if request_id:
            for record in reversed(records):
                if record["id"] == request_id:
                    return self.detail(record)
            return None
        return {"requests": [self.summary(r) for r in reversed(records)], "in_flight": in_flight}


def clear(services) -> dict:
    """Clear every configured model's finished requests; the count left are those still in flight."""
    return {"remaining": sum(svc.monitor.clear() for svc in services)}


def gather(services, request_id: str | None = None):
    """The monitor's answer over every configured model: one list, newest first, each record naming its model."""
    if request_id:
        for svc in services:
            found = svc.monitor.snapshot(request_id)
            if found is not None:
                return found
        return None
    parts = [svc.monitor.snapshot() for svc in services]
    requests = sorted((r for part in parts for r in part["requests"]), key=lambda r: r["started_at"], reverse=True)
    return {"requests": requests,
            "in_flight": sum(part["in_flight"] for part in parts), "capacity": RECORDS * len(services)}
