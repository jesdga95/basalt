"""ContextClient for the Mantle orchestrator, backed by the context registry through ContextAdapter, and the
previous_response_id -> thread map that ContextAdapter's resolver hook needs.

prefill() binds the handoff's transcript (``transcript_ref`` = thread id) to the head's context, takes the fenced
writer lease and starts materializing; the engine's own re-prefill from the transcript (never raw KV) completes the
transition to HOT under that lease. start() checks the context is building or ready. discard() releases the lease and
tombstones the context."""
from __future__ import annotations

import collections
import threading
from typing import Optional

from serve.api.context_adapter import ContextAdapter, RegistryError
from .handoff import Handoff


class RegistryContextClient:
    def __init__(self, adapter: ContextAdapter):
        self.adapter = adapter
        self._leases: dict = {}   # context id -> numeric lease token

    @staticmethod
    def _lease_token(res: dict) -> int:
        lease = res.get("lease")
        token = lease.get("token") if isinstance(lease, dict) else None
        # the registry issues the token as a 64-bit number (a digit string is accepted); release sends it back as one
        if isinstance(token, bool) or not isinstance(token, (int, str)) or (isinstance(token, str) and not token.isdigit()):
            raise RegistryError("invalid_response")
        return int(token)

    def prefill(self, model: str, handoff: Handoff) -> str:
        """Create (or reuse) the transcript head's context, take its writer lease and start materializing it. The
        engine's re-prefill from the transcript completes the transition to HOT under the same lease."""
        key = f"mantle:{model}:{handoff.transcript_ref}"
        ctx = self.adapter._call("POST", "/v1/contexts", {"thread_id": handoff.transcript_ref}, f"{key}:create", (200, 201))
        cid = self.adapter._id(ctx.get("context_id"))
        res = self.adapter._call("POST", f"/v1/contexts/{cid}/resume", {}, f"{key}:resume", (200,))
        token = self._lease_token(res)
        self._leases[cid] = token
        try:
            cur = self.adapter._call("GET", f"/v1/contexts/{cid}", None, None, (200,))
            version = cur.get("version")
            if not isinstance(version, int) or isinstance(version, bool):
                raise RegistryError("invalid_response")
            if cur.get("state") == "RECORDED":
                self.adapter._call("POST", f"/v1/contexts/{cid}/transition",
                                   {"event": "materialize", "expected_version": version, "lease_token": token},
                                   f"{key}:materialize", (200,))
        except RegistryError:
            self.discard(cid)
            raise
        return cid

    def start(self, context_id: str) -> None:
        """The context must be building or ready under this client's lease; anything else means the prefill was lost."""
        if context_id not in self._leases:
            raise RegistryError("lease_missing")
        cur = self.adapter._call("GET", f"/v1/contexts/{self.adapter._id(context_id)}", None, None, (200,))
        if cur.get("state") not in ("MATERIALIZING", "HOT"):
            raise RegistryError("not_materialized")

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
