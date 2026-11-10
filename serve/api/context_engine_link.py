"""Links a successful registry fork to the engine's native thread tapes.

Tape names are the context ids and heads are ``v<version>``, so the registry's CAS version is the tape's head: a stale
parent is refused by the engine exactly as the registry refuses it. Best effort by design: no parent tape in this
engine epoch (restart, never attached) is not an error, the child then simply replays and the exact-prefix KV check
decides what is reused."""
from __future__ import annotations

import re

_FORK_PATH = re.compile(r"^/v1/contexts/([0-9a-fA-F-]{36})/fork$")


def head_of(version) -> str | None:
    return f"v{version}" if isinstance(version, int) and not isinstance(version, bool) and version >= 0 else None


def after_registry_fork(path: str, request: dict, status: int, body: dict, fork) -> dict | None:
    """Return the ``engine_fork`` annotation for the response, or None when this was not a successful fork.
    `fork(parent, parent_head, child, child_head, keep) -> shared_prefix_tokens` is Service.thread_fork."""
    m = _FORK_PATH.match(path)
    if not m or status not in (200, 201) or not isinstance(body, dict):
        return None
    child = body.get("context_id")
    parent_head = head_of(request.get("expected_version"))
    child_head = head_of(body.get("version"))
    if not (isinstance(child, str) and parent_head and child_head):
        return {"status": "skipped", "reason": "registry response lacks ids or versions"}
    try:
        shared = fork(m.group(1), parent_head, child, child_head, 0)
    except ValueError as e:                      # no parent tape, stale head, child already differs
        return {"status": "skipped", "reason": str(e)}
    except Exception as e:                       # engine down or busy: the registry fork stands
        return {"status": "skipped", "reason": type(e).__name__}
    return {"status": "forked", "shared_prefix_tokens": shared}
