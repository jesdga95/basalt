"""Adapter from Chat/Responses thread bindings to the /v1/contexts REST contract.

Not wired into the handler yet. ``attach`` resolves a request's thread binding
(``thread_id`` or ``previous_response_id`` via an injected resolver) into a leased
context. Any registry failure (transport error, 404, 409, 422, 5xx, bad body) yields an
explicit replay fallback: the caller must re-send the full transcript, nothing is
silently dropped.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_ID = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_-]{0,127}$")

# transport(method, url, headers, body, timeout) -> (status, response_bytes)
Transport = Callable[[str, str, Mapping[str, str], Optional[bytes], float], "tuple[int, bytes]"]
# resolver(previous_response_id) -> (thread_id, branch_id|None, upto_message_id|None) | None
ResponseResolver = Callable[[str], Optional["tuple[str, Optional[str], Optional[str]]"]]
TranscriptLoader = Callable[[str, Optional[str], Optional[str]], list]


class TransportError(Exception):
    """Network-level failure (connect, timeout, reset)."""


def urllib_transport(method: str, url: str, headers: Mapping[str, str],
                     body: Optional[bytes], timeout: float) -> "tuple[int, bytes]":
    req = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (base_url is operator config)
            return resp.status, resp.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(MAX_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise TransportError(type(exc).__name__) from exc


@dataclass(frozen=True)
class Binding:
    thread_id: str
    branch_id: Optional[str] = None
    upto_message_id: Optional[str] = None


@dataclass
class ReplayFallback:
    reason: str
    status: Optional[int]
    binding: Optional[Binding]
    messages: Optional[list] = None  # full transcript when a loader is injected


@dataclass
class AttachResult:
    """mode: 'none' (no binding), 'context' (leased context), 'replay' (use fallback)."""
    mode: str
    context_id: Optional[str] = None
    lease_token: Optional[int] = None  # the registry issues a 64-bit number
    source: Optional[str] = None  # hot|warm|cold|replay as reported by the registry
    replay_required: bool = False  # True when mode=='replay' or registry source=='replay'
    fallback: Optional[ReplayFallback] = None
    context: dict = field(default_factory=dict)


class RegistryError(Exception):
    def __init__(self, reason: str, status: Optional[int] = None):
        super().__init__(reason)
        self.reason = reason
        self.status = status


_STATUS_REASON = {404: "context_not_found", 409: "conflict", 422: "invalid_request"}


class ContextAdapter:
    def __init__(self, base_url: str, headers: Optional[Mapping[str, str]] = None,
                 transport: Transport = urllib_transport, timeout: float = 10.0,
                 resolver: Optional[ResponseResolver] = None,
                 transcript_loader: Optional[TranscriptLoader] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers or {})
        self.transport = transport
        self.timeout = timeout
        self.resolver = resolver
        self.transcript_loader = transcript_loader

    @classmethod
    def from_env(cls, **kw) -> Optional["ContextAdapter"]:
        """Same configuration as thread_proxy; None when the service is not configured."""
        secret = os.environ.get("BASALT_THREAD_API_KEY")
        if not secret or len(secret) < 24:
            return None
        host = os.environ.get("BASALT_THREAD_API_HOST") or "127.0.0.1"
        port = int(os.environ.get("BASALT_THREAD_API_PORT", "8095"))
        return cls(f"http://{host}:{port}", {"Authorization": f"Bearer {secret}"}, **kw)

    # -- binding ---------------------------------------------------------------
    def extract_binding(self, request: Mapping[str, Any]) -> Optional[Binding]:
        meta = request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
        thread_id = request.get("thread_id") or meta.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            return Binding(thread_id,
                           request.get("branch_id") or meta.get("branch_id"),
                           request.get("upto_message_id") or meta.get("upto_message_id"))
        prev = request.get("previous_response_id")
        if isinstance(prev, str) and prev and self.resolver is not None:
            try:
                resolved = self.resolver(prev)
            except Exception:
                resolved = None
            if resolved:
                return Binding(*resolved)
        return None

    # -- operations ------------------------------------------------------------
    def attach(self, request: Mapping[str, Any], idempotency_key: Optional[str] = None,
               lease_ttl_ms: Optional[int] = None) -> AttachResult:
        """Create + resume a context for the request's binding, or explain why to replay."""
        wants_binding = bool(request.get("previous_response_id") or request.get("thread_id")
                             or (isinstance(request.get("metadata"), dict)
                                 and request["metadata"].get("thread_id")))
        binding = self.extract_binding(request)
        if binding is None:
            if wants_binding:  # previous_response_id we cannot map: replay, never guess
                return AttachResult("replay", replay_required=True, fallback=ReplayFallback(
                    "unresolvable_previous_response", None, None))
            return AttachResult("none")
        key = idempotency_key or self._derive_key(binding)
        try:
            payload = {"thread_id": binding.thread_id}
            if binding.branch_id:
                payload["branch_id"] = binding.branch_id
            if binding.upto_message_id:
                payload["upto_message_id"] = binding.upto_message_id
            ctx = self._call("POST", "/v1/contexts", payload, f"{key}:create", (200, 201))
            context_id = self._id(ctx.get("context_id"))
            resume_body = {"lease_ttl_ms": lease_ttl_ms} if lease_ttl_ms else {}
            res = self._call("POST", f"/v1/contexts/{context_id}/resume", resume_body,
                             f"{key}:resume", (200,))
            lease = res.get("lease")
            token = lease.get("token") if isinstance(lease, dict) else None
            # The registry issues the lease token as a 64-bit number; transitions and releases send it back as one.
            if isinstance(token, bool) or not isinstance(token, (int, str)):
                raise RegistryError("invalid_response")
            if isinstance(token, str):
                if not token.isdigit():
                    raise RegistryError("invalid_response")
                token = int(token)
        except RegistryError as err:
            return self._replay(binding, err)
        source = res.get("source")
        return AttachResult("context", context_id=context_id, lease_token=token, source=source,
                            replay_required=source == "replay", context=ctx)

    def fork(self, context_id: str, expected_version: int, idempotency_key: str) -> dict:
        """Idempotent child creation; raises RegistryError (409 on conflicting payload)."""
        cid = self._id(context_id)
        if not idempotency_key:
            raise ValueError("Idempotency-Key required for fork")
        return self._call("POST", f"/v1/contexts/{cid}/fork",
                          {"expected_version": expected_version}, idempotency_key, (200, 201))

    def release(self, context_id: str, lease_token: int) -> bool:
        try:
            self._call("POST", f"/v1/contexts/{self._id(context_id)}/lease/release",
                       {"token": lease_token}, None, (200, 202, 204))
            return True
        except RegistryError:
            return False

    # -- internals -------------------------------------------------------------
    @staticmethod
    def _id(value: Any) -> str:
        if not isinstance(value, str) or not _ID.fullmatch(value):
            raise RegistryError("invalid_response")
        return value

    @staticmethod
    def _derive_key(b: Binding) -> str:
        raw = json.dumps([b.thread_id, b.branch_id, b.upto_message_id])
        return "adapter-" + hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _call(self, method: str, path: str, payload: Optional[dict], idem: Optional[str],
              ok: "tuple[int, ...]") -> dict:
        headers = {"Accept": "application/json", **self.headers}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if idem and method == "POST":
            headers["Idempotency-Key"] = idem
        try:
            status, raw = self.transport(method, self.base_url + path, headers, body, self.timeout)
        except (TransportError, OSError, TimeoutError) as exc:
            raise RegistryError("registry_unavailable") from exc
        if status not in ok:
            reason = _STATUS_REASON.get(status, "registry_unavailable" if status >= 500
                                        else "unexpected_status")
            raise RegistryError(reason, status)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise RegistryError("invalid_response", status)
        try:
            decoded = json.loads(raw) if raw else {}
        except ValueError as exc:
            raise RegistryError("invalid_response", status) from exc
        if not isinstance(decoded, dict):
            raise RegistryError("invalid_response", status)
        return decoded

    def _replay(self, binding: Binding, err: RegistryError) -> AttachResult:
        messages = None
        if self.transcript_loader is not None:
            try:
                messages = self.transcript_loader(binding.thread_id, binding.branch_id,
                                                  binding.upto_message_id)
            except Exception:
                messages = None
        return AttachResult("replay", replay_required=True, fallback=ReplayFallback(
            err.reason, err.status, binding, messages))
