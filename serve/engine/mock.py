"""The Engine protocol and MockEngine, a scripted engine that needs no GPU."""
from __future__ import annotations

import threading
import time
from typing import Iterator, Protocol

from serve.engine.slots import AdmissionQueue
from serve.util.text import IM_END


class Engine(Protocol):
    max_context: int
    def generate(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event, embeddings=None,
                 session: str | None = None) -> Iterator[int]: ...


class MockEngine:
    """Replays scripted completions as token ids plus the end-of-turn token; a list gives one script per request, the last repeating."""

    def __init__(self, tokenizer, script: str | list[str], max_context: int = 32768, delay_s: float = 0.0,
                 batch: int = 0, route=None):
        self.tok, self.max_context, self.delay = tokenizer, max_context, delay_s
        # batch >= 2: that many requests generate at once, the rest wait their turn (first come, first served)
        self.batch = batch if batch >= 2 else 0
        self.cv, self.admission, self.busy = threading.Condition(), AdmissionQueue(), 0
        self.route = route                              # prompt ids -> index of the script to answer with
        self.peak = 0                                   # most requests generating at once
        self.requests: list[tuple[list[int], dict]] = []    # (prompt, sampling) of each request, in arrival order
        end = tokenizer.encode(IM_END, parse_special=True)
        self.scripts = [tokenizer.encode(x, parse_special=True) + end for x in ([script] if isinstance(script, str)
                                                                                 else script)]
        self.script, self.turns = self.scripts[0], 0
        self.last_prompt: list[int] = []

    def queue_depth(self) -> int:
        with self.cv:
            return len(self.admission)

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):   # one script: a session picks no slot
        if not self.batch:
            yield from self._run(ids, max_new, sampling, cancel, embeddings)
            return
        with self.cv:
            ticket = self.admission.join()
            try:
                while not (self.busy < self.batch and self.admission.may_go(ticket)):
                    self.cv.wait(timeout=0.05)
                    if cancel.is_set():
                        return
                self.busy += 1
                self.peak = max(self.peak, self.busy)
            finally:
                self.admission.leave(ticket)
                self.cv.notify_all()
        try:
            yield from self._run(ids, max_new, sampling, cancel, embeddings)
        finally:
            with self.cv:
                self.busy -= 1
                self.cv.notify_all()

    def _run(self, ids, max_new, sampling, cancel, embeddings=None):
        self.requests.append((list(ids), dict(sampling or {})))
        self.last_prompt = list(ids)
        self.last_embeddings = embeddings
        script = self.script
        if self.route:
            script = self.scripts[self.route(list(ids))]
        elif len(self.scripts) > 1:
            script = self.scripts[min(self.turns, len(self.scripts) - 1)]
            self.turns += 1
        self.script = script
        for t in script[:max_new]:
            if cancel.is_set():
                return
            if self.delay:
                time.sleep(self.delay)
            yield t
