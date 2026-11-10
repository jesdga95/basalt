"""BIG<->LITTLE model transition state machine with a crash-safe journal.

States (role neutral: "source" is the model being left, "target" the one entered)::

    PLAN_PERSISTED -> BIG_RELEASED -> LITTLE_LOADED -> PREFILLED -> RUNNING

``BIG_RELEASED`` means the source model was released, ``LITTLE_LOADED`` that the target
was loaded, for both demotion (BIG->LITTLE) and promotion (LITTLE->BIG). A step that
exhausts its retries triggers rollback (discard prefill context, release target, reload
source) ending in ROLLED_BACK; a rollback that itself fails ends in ROLLBACK_FAILED and
is retried by ``recover()``.

Effects go through ``ModelController`` / ``ContextClient``. Their methods must be
idempotent: a crash between an effect and its journal write re-runs that step.
The journal is one JSON file per idempotency key, replaced via tmp + fsync + rename +
directory fsync.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from .handoff import Handoff

PLAN_PERSISTED = "PLAN_PERSISTED"
BIG_RELEASED = "BIG_RELEASED"
LITTLE_LOADED = "LITTLE_LOADED"
PREFILLED = "PREFILLED"
RUNNING = "RUNNING"
ROLLED_BACK = "ROLLED_BACK"
ROLLBACK_FAILED = "ROLLBACK_FAILED"

FORWARD = (PLAN_PERSISTED, BIG_RELEASED, LITTLE_LOADED, PREFILLED, RUNNING)
TERMINAL = frozenset({RUNNING, ROLLED_BACK})
DIRECTIONS = frozenset({"demote", "promote"})  # demote: BIG->LITTLE, promote: LITTLE->BIG


class ModelController(Protocol):
    def load(self, model: str) -> None: ...
    def release(self, model: str) -> None: ...


class ContextClient(Protocol):
    def prefill(self, model: str, handoff: Handoff) -> str:
        """Prefill ``model`` from the handoff's transcript; return a context id."""
    def start(self, context_id: str) -> None: ...
    def discard(self, context_id: str) -> None: ...


class IdempotencyConflict(Exception):
    """Same idempotency key reused with a different request."""


@dataclass
class Transition:
    key: str
    request_digest: str
    direction: str
    source_model: str
    target_model: str
    handoff: str  # Handoff.dumps() envelope
    state: str = PLAN_PERSISTED
    context_id: Optional[str] = None
    error: Optional[str] = None
    attempts: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Journal:
    def __init__(self, directory: str):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json")

    def save(self, t: Transition) -> None:
        path = self._path(t.key)
        tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        data = json.dumps(t.__dict__, sort_keys=True, ensure_ascii=False).encode("utf-8")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        _fsync_dir(self.dir)

    def load(self, key: str) -> Optional[Transition]:
        return self._read(self._path(key))

    def all(self) -> list:
        out = []
        for name in sorted(os.listdir(self.dir)):
            if name.endswith(".json"):
                t = self._read(os.path.join(self.dir, name))
                if t is not None:
                    out.append(t)
        return out

    @staticmethod
    def _read(path: str) -> Optional[Transition]:
        try:
            with open(path, "rb") as f:
                return Transition(**json.loads(f.read()))
        except FileNotFoundError:
            return None


class Orchestrator:
    def __init__(self, journal_dir: str, controller: ModelController, client: ContextClient,
                 max_attempts: int = 3, backoff_s: float = 0.0,
                 sleep: Callable[[float], None] = time.sleep):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.journal = Journal(journal_dir)
        self.controller = controller
        self.client = client
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self._sleep = sleep
        self._lock = threading.RLock()

    # -- public API -------------------------------------------------------------
    def begin(self, key: str, direction: str, source_model: str, target_model: str,
              handoff: Handoff) -> Transition:
        """Persist the plan (PLAN_PERSISTED). Same key + request returns the existing record."""
        if not key:
            raise ValueError("idempotency key required")
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}")
        if not source_model or not target_model or source_model == target_model:
            raise ValueError("source_model and target_model must be set and differ")
        if handoff.target_model != target_model:
            raise ValueError("handoff.target_model does not match target_model")
        envelope = handoff.dumps()  # validates
        digest = hashlib.sha256(json.dumps(
            [direction, source_model, target_model, handoff.digest]).encode()).hexdigest()
        with self._lock:
            existing = self.journal.load(key)
            if existing is not None:
                if existing.request_digest != digest:
                    raise IdempotencyConflict(key)
                return existing
            t = Transition(key=key, request_digest=digest, direction=direction,
                           source_model=source_model, target_model=target_model,
                           handoff=envelope)
            t.history.append(PLAN_PERSISTED)
            self.journal.save(t)
            return t

    def run(self, key: str) -> Transition:
        """Drive a journaled transition to a terminal state (or ROLLBACK_FAILED)."""
        with self._lock:
            t = self.journal.load(key)
            if t is None:
                raise KeyError(key)
            if t.state == ROLLBACK_FAILED:
                return self._rollback(t)
            while not t.terminal:
                if not self._advance(t):
                    return self._rollback(t)
            return t

    def submit(self, key: str, direction: str, source_model: str, target_model: str,
               handoff: Handoff) -> Transition:
        self.begin(key, direction, source_model, target_model, handoff)
        return self.run(key)

    def get(self, key: str) -> Optional[Transition]:
        return self.journal.load(key)

    def recover(self) -> list:
        """After a restart, finish every non-terminal journaled transition."""
        with self._lock:
            return [self.run(t.key) for t in self.journal.all() if not t.terminal]

    # -- internals --------------------------------------------------------------
    def _retry(self, t: Transition, step: str, fn: Callable[[], object]):
        last: Optional[BaseException] = None
        for attempt in range(1, self.max_attempts + 1):
            t.attempts[step] = t.attempts.get(step, 0) + 1
            try:
                return fn()
            except Exception as exc:  # effect failures are data, not crashes
                last = exc
                t.error = f"{step}: {type(exc).__name__}: {exc}"
                self.journal.save(t)
                if attempt < self.max_attempts and self.backoff_s > 0:
                    self._sleep(self.backoff_s * (2 ** (attempt - 1)))
        raise last  # type: ignore[misc]

    def _advance(self, t: Transition) -> bool:
        step = t.state
        try:
            if step == PLAN_PERSISTED:
                self._retry(t, step, lambda: self.controller.release(t.source_model))
                nxt = BIG_RELEASED
            elif step == BIG_RELEASED:
                self._retry(t, step, lambda: self.controller.load(t.target_model))
                nxt = LITTLE_LOADED
            elif step == LITTLE_LOADED:
                handoff = Handoff.loads(t.handoff)
                t.context_id = self._retry(
                    t, step, lambda: self.client.prefill(t.target_model, handoff))
                nxt = PREFILLED
            else:  # PREFILLED
                self._retry(t, step, lambda: self.client.start(t.context_id))
                nxt = RUNNING
        except Exception:
            return False
        t.state = nxt
        t.error = None
        t.history.append(nxt)
        self.journal.save(t)
        return True

    def _rollback(self, t: Transition) -> Transition:
        reached = FORWARD.index(t.state) if t.state in FORWARD else FORWARD.index(LITTLE_LOADED)
        try:
            if t.context_id:
                self._retry(t, "rollback.discard", lambda: self.client.discard(t.context_id))
                t.context_id = None
            if reached >= FORWARD.index(BIG_RELEASED):  # target load may have partly happened
                self._retry(t, "rollback.release_target",
                            lambda: self.controller.release(t.target_model))
            self._retry(t, "rollback.reload_source",
                        lambda: self.controller.load(t.source_model))
        except Exception:
            # Remember how far we had got so recover() resumes the same rollback.
            if t.state != ROLLBACK_FAILED:
                t.history.append(ROLLBACK_FAILED)
            t.state = ROLLBACK_FAILED
            self.journal.save(t)
            return t
        t.state = ROLLED_BACK
        t.history.append(ROLLED_BACK)
        self.journal.save(t)
        return t
