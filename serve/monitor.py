"""API request monitor: the last 100 calls' input, output, reasoning and raw response.

Off by default (--api-monitor). Self-contained on purpose: the hooks in `serve/handler.py` and `serve/service.py`
are one call each, so a server update touches at most those few lines. The page lives at /api-monitor, data at
/api/monitor, assets at serve/web/monitor.{html,js,css}.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque

RECORDS = 100                  # requests kept in memory
CHARS = 131072                 # per field (input, output, reasoning, response)


def _truncate(value: str) -> tuple[str, bool]:
    return value[:CHARS], len(value) > CHARS


class Monitor:
    """Off by default; `enabled` is set from --api-monitor / "api-monitor": true in the config."""

    def __init__(self, svc):
        self.svc = svc
        self.enabled = False
        self.records = deque(maxlen=RECORDS)
        self.trace = threading.local()                  # .record: this request's record (thread per connection)
        self.lock = svc.status_lock

    # ------------------------------------------------------------------ lifecycle
    def begin(self, path: str, req: dict):
        if not self.enabled:
            return None
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
                  "error": None, "http_status": None, "usage": {}, "timings": {},
                  "wallclock_s": None, "load_s": 0.0, "queue_s": 0.0, "first_token_s": None,
                  "_clock": time.perf_counter()}
        with self.lock:
            self.records.append(record)
        self.trace.record = record
        return record

    def finish(self):
        """Close this request's record; safe to call more than once and when the monitor is off."""
        record = getattr(self.trace, "record", None)
        if record is None:
            return
        with self.lock:
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

    def _eat(self, record: dict, api: str, item) -> None:
        content = reasoning = ""
        usage = timings = None
        try:
            if api == "openai" and isinstance(item, dict):
                choice = (item.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content, reasoning = delta.get("content") or "", delta.get("reasoning_content") or ""
                usage, timings = item.get("usage"), item.get("timings")
            elif api == "anthropic" and isinstance(item, tuple) and len(item) == 2:
                name, data = item
                data = data if isinstance(data, dict) else {}
                delta = data.get("delta") or {}
                kind = delta.get("type")
                if kind == "text_delta":
                    content = delta.get("text") or ""
                elif kind == "thinking_delta":
                    reasoning = delta.get("thinking") or ""
                if name == "message_delta":
                    usage = data.get("usage")
            elif api == "responses" and isinstance(item, dict):
                kind = item.get("type")
                if kind == "response.output_text.delta":
                    content = item.get("delta") or ""
                elif kind == "response.reasoning_text.delta":
                    reasoning = item.get("delta") or ""
                elif kind in ("response.completed", "response.incomplete"):
                    usage = (item.get("response") or {}).get("usage")
        except (AttributeError, IndexError, TypeError, KeyError):
            return
        if not (content or reasoning or usage or timings):
            return
        with self.lock:
            for field, value in (("output", content), ("reasoning", reasoning)):
                if value:
                    joined = record[field] + value
                    record[field], record[field + "_truncated"] = _truncate(joined)
            if usage:
                record["usage"] = usage
            if timings:
                record["timings"] = timings
            if (content or reasoning) and record["first_token_s"] is None:
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
                if isinstance(obj.get("usage"), dict):
                    record["usage"] = obj["usage"]
                if isinstance(obj.get("timings"), dict):
                    record["timings"] = obj["timings"]

    # ------------------------------------------------------------------ read side
    def summary(self, record: dict) -> dict:
        return {key: value for key, value in record.items()
                if key not in ("input", "output", "reasoning", "response") and not key.startswith("_")}

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
        with self.lock:
            records = list(self.records)
            loaded, model = self.svc.loaded(), self.svc.model
            in_flight = len(getattr(self.svc, "live_reqs", {}) or {})
        if request_id:
            for record in reversed(records):
                if record["id"] == request_id:
                    return self.detail(record)
            return None
        return {"requests": [self.summary(r) for r in reversed(records)],
                "loaded": loaded, "auto_load": True, "model": model, "in_flight": in_flight,
                "capacity": RECORDS}
