"""Bounded, authenticated-by-the-caller SSE access to the server's log ring."""
import json
import time
from urllib.parse import parse_qs, urlsplit


def stream_logs(handler, logs):
    query = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
    try:
        after = int(query["after"][0]) if "after" in query else None
        limit = int(query.get("limit", ["300" if after is None else "500"])[0])
        if (after is not None and after < 0) or not 1 <= limit <= 500:
            raise ValueError()
    except ValueError:
        handler._json(400, {"error": {"message": "after must be non-negative; limit must be 1..500"}})
        return
    if not logs.clients.acquire(blocking=False):
        handler._json(503, {"error": {"message": "Too many log streams (maximum 8)"}})
        return
    timeout = handler.connection.gettimeout()
    try:
        handler.connection.settimeout(5)
        handler._sse()
        handler.close_connection = True
        instance = query.get("instance", [None])[0]
        initial, heartbeat = True, time.monotonic()
        while not handler.server.log_stop.is_set():
            data = logs.snapshot(after, limit, instance)
            if initial or data["lines"] or data["reset"]:
                handler.wfile.write(b"event: logs\ndata: " + json.dumps(data, ensure_ascii=False).encode() + b"\n\n")
                handler.wfile.flush()
                after, instance = data["next"], data["instance_id"]
                initial, heartbeat = False, time.monotonic()
            elif time.monotonic() - heartbeat >= 10:
                handler.wfile.write(b": heartbeat\n\n")
                handler.wfile.flush()
                heartbeat = time.monotonic()
            handler.server.log_stop.wait(.2)
    except OSError:
        pass
    finally:
        handler.connection.settimeout(timeout)
        logs.clients.release()
