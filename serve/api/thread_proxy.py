"""Optional localhost bridge to the Rust Stateful Threads API.

No public caller can choose the upstream host or owner identity. The existing Basalt
handler checks API-key and Origin before proxying. Without configuration, the feature
is disabled and all original model routes remain unchanged.
"""
from __future__ import annotations

import http.client
import json
import os
import re

_THREAD_PATH = re.compile(r"^/v1/threads(?:/[0-9a-fA-F-]{36}(?:/(?:messages|turns|metrics|branches(?:/[0-9a-fA-F-]{36}(?:/messages)?)?))?)?$")


def is_thread_path(path: str) -> bool:
    return bool(_THREAD_PATH.fullmatch(path))


def forward(method: str, path: str, payload=None) -> tuple[int, dict]:
    if not is_thread_path(path):
        return 404, {"error": {"message": "thread route not found"}}
    secret = os.environ.get("BASALT_THREAD_API_KEY")
    port = os.environ.get("BASALT_THREAD_API_PORT", "8095")
    if not secret or len(secret) < 24:
        return 503, {"error": {"message": "stateful thread service not configured"}}
    try:
        portnum = int(port)
        if not 1 <= portnum <= 65535:
            raise ValueError("port out of range")
    except ValueError:
        return 503, {"error": {"message": "invalid thread service port"}}
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    host = os.environ.get("BASALT_THREAD_API_HOST") or "127.0.0.1"
    conn = http.client.HTTPConnection(host, portnum, timeout=3600)
    try:
        conn.request(method, path, body=body, headers={
            "Authorization": f"Bearer {secret}",
            "Content-Type": "application/json",
        })
        response = conn.getresponse()
        raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            return 502, {"error": {"message": "thread service returned oversized response"}}
        decoded = json.loads(raw) if raw else {}
        if not isinstance(decoded, dict):
            return 502, {"error": {"message": "invalid thread service response"}}
        return response.status, decoded
    except (OSError, TimeoutError, ValueError, http.client.HTTPException) as exc:
        return 503, {"error": {"type": "server_error",
                               "message": "thread service unavailable",
                               "detail": type(exc).__name__}}
    finally:
        conn.close()
