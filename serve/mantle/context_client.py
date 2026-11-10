"""ContextClient for the Mantle orchestrator, backed by the context registry through ContextAdapter, and the
previous_response_id -> thread map that ContextAdapter's resolver hook needs.

prefill() binds the handoff's transcript (``transcript_ref`` = thread id) to a new context and asks the registry to
materialize it; the engine's own materialization (re-prefill from the transcript, never raw KV) completes the
transition to HOT. start() takes the fenced writer lease. discard() tombstones the context."""
from __future__ import annotations

import collections
import threading
from typing import Optional

from serve.api.context_adapter import ContextAdapter, RegistryError
from .handoff import Handoff


class RegistryContextClient:
    def __init__(self, adapter: ContextAdapter):
        self.adapter = adapter
        self._leases: dict = {}

    def prefill(self, model: str, handoff: Handoff) -> str:
        ctx = self.adapter._call("POST", "/v1/contexts", {"thread_id": handoff.transcript_ref},
                                 f"mantle:{model}:{handoff.transcript_ref}:create", (200, 201))
        cid = self.adapter._id(ctx.get("context_id"))
        version = ctx.get("version")
        if not isinstance(version, int):
            raise RegistryError("invalid_response")
        self.adapter._call("POST", f"/v1/contexts/{cid}/transition",
                           {"event": "materialize", "expected_version": version},
                           f"mantle:{model}:{handoff.transcript_ref}:materialize", (200,))
        return cid

    def start(self, context_id: str) -> None:
        res = self.adapter._call("POST", f"/v1/contexts/{self.adapter._id(context_id)}/resume", {},
                                 f"mantle:{context_id}:resume", (200,))
        lease = res.get("lease")
        token = lease.get("token") if isinstance(lease, dict) else None
        if not isinstance(token, str) or not token:
            raise RegistryError("invalid_response")
        self._leases[context_id] = token

    def discard(self, context_id: str) -> None:
        token = self._leases.pop(context_id, None)
        if token:
            self.adapter.release(context_id, token)
        try:
            self.adapter._call("DELETE", f"/v1/contexts/{self.adapter._id(context_id)}", None, None, (200, 202, 204, 404))
        except RegistryError:
            pass  # a discard that cannot reach the registry is retried by the orchestrator's recovery


class ResponseThreadMap:
    """Bounded, thread-safe previous_response_id -> (thread_id, branch_id, upto_message_id). Entries only come from
    responses this server produced for a thread-bound request, so an unknown id is a replay, never a guess."""

    def __init__(self, capacity: int = 4096):
        self.capacity = max(1, int(capacity))
        self._d: "collections.OrderedDict[str, tuple]" = collections.OrderedDict()
        self._lock = threading.Lock()

    def remember(self, response_id: str, thread_id: str, branch_id: Optional[str] = None,
                 upto_message_id: Optional[str] = None) -> None:
        if not (isinstance(response_id, str) and response_id and isinstance(thread_id, str) and thread_id):
            return
        with self._lock:
            self._d[response_id] = (thread_id, branch_id, upto_message_id)
            self._d.move_to_end(response_id)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)

    def __call__(self, response_id: str):
        with self._lock:
            v = self._d.get(response_id)
            if v is not None:
                self._d.move_to_end(response_id)
            return v
