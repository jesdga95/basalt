"""Optional localhost bridge to the Rust context registry (`/v1/contexts*`).

Same sidecar, secret and loopback-only rule as ``thread_proxy``: callers cannot choose
the upstream host or the owner identity, and the existing Basalt handler authenticates
the request before calling ``forward``. Without configuration the feature is disabled.
"""
from __future__ import annotations

import http.client
import json
import os
import re

_ID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
# (compiled path pattern, allowed methods) — mirrors the Rust router exactly.
_ROUTES = (
    (re.compile(r"^/v1/contexts$"), frozenset({"POST"})),
    (re.compile(rf"^/v1/contexts/{_ID}$"), frozenset({"GET", "DELETE"})),
    (re.compile(rf"^/v1/contexts/{_ID}/(?:fork|resume|transition)$"), frozenset({"POST"})),
    (re.compile(rf"^/v1/contexts/{_ID}/lease/release$"), frozenset({"POST"})),
    (re.compile(rf"^/v1/contexts/{_ID}/payload/released$"), frozenset({"POST"})),
)
_MAX_RESPONSE = 8 * 1024 * 1024
_IDEMPOTENCY_KEY = re.compile(r"^[\x20-\x7e]{1,256}$")


def _error(status: int, code: str, message: str, **extra) -> tuple[int, dict]:
    return status, {"error": {"code": code, "message": message, **extra}}


def is_context_path(path: str) -> bool:
    return any(pattern.fullmatch(path) for pattern, _ in _ROUTES)


def allowed_methods(path: str) -> frozenset:
    for pattern, methods in _ROUTES:
        if pattern.fullmatch(path):
            return methods
    return frozenset()


def forward(method: str, path: str, payload=None, idempotency_key: str | None = None) -> tuple[int, dict]:
    methods = allowed_methods(path)
    if not methods:
        return _error(404, "not_found", "context route not found")
    if method not in methods:
        return _error(405, "method_not_allowed", "method not allowed for this context route")
    if idempotency_key is not None and not _IDEMPOTENCY_KEY.fullmatch(idempotency_key):
        return _error(422, "invalid_request", "Idempotency-Key must be 1..256 printable ASCII characters")
    secret = os.environ.get("BASALT_THREAD_API_KEY")
    port = os.environ.get("BASALT_THREAD_API_PORT", "8095")
    if not secret or len(secret) < 24:
        return _error(503, "unavailable", "context service not configured")
    try:
        portnum = int(port)
        if not 1 <= portnum <= 65535:
            raise ValueError("port out of range")
    except ValueError:
        return _error(503, "unavailable", "invalid context service port")
    headers = {"Authorization": f"Bearer {secret}", "Content-Type": "application/json"}
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    host = os.environ.get("BASALT_THREAD_API_HOST") or "127.0.0.1"
    conn = http.client.HTTPConnection(host, portnum, timeout=60)
    try:
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        raw = response.read(_MAX_RESPONSE + 1)
        if len(raw) > _MAX_RESPONSE:
            return _error(502, "bad_gateway", "context service returned oversized response")
        decoded = json.loads(raw) if raw else {}
        if not isinstance(decoded, dict):
            return _error(502, "bad_gateway", "invalid context service response")
        return response.status, decoded
    except (OSError, TimeoutError, ValueError, http.client.HTTPException) as exc:
        return _error(503, "unavailable", "context service unavailable", detail=type(exc).__name__)
    finally:
        conn.close()
