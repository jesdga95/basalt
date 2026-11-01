"""Demand-driven score waves, with bounded callers and in-flight row fan-out."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from serve.decision.admission import DecisionError


@dataclass
class Job:
    row: tuple
    controls: list = field(default_factory=list)
    done: bool = False
    value: object = None
    error: Exception | None = None


class Live:
    def __init__(self, jobs, lock):
        self.jobs, self.lock = jobs, lock

    def is_set(self):
        with self.lock:
            return all(c.is_set() for job in self.jobs for c in job.controls)

    def check(self):
        if self.is_set():
            raise DecisionError(499, "all decision wave callers cancelled")


class Batches:
    def __init__(self, execute):
        self.execute = execute
        self.cv = threading.Condition()
        self.jobs = {}
        self.pending = []
        self.running = False

    def submit(self, rows, control):
        control.check()
        waiting = []
        with self.cv:
            for row in rows:
                ids, options, points, fresh = row
                key = (tuple(ids), tuple(options), object() if fresh else None)
                job = self.jobs.get(key)
                if job is not None and all(c.is_set() for c in job.controls):
                    key = (*key[:2], object())
                    job = None
                owner = job is None
                if owner:
                    job = Job((list(ids), list(options), list(points), fresh))
                    self.jobs[key] = job
                    self.pending.append((key, job))
                job.controls.append(control)
                waiting.append((job, owner))
            self.cv.notify_all()
            if not self.running and waiting:
                self.running = True
                threading.Thread(target=self._drain, name="jev-wave", daemon=True).start()
            while not all(job.done for job, _ in waiting):
                control.check()
                self.cv.wait(0.05)
            control.check()
            result = []
            for job, owner in waiting:
                if job.error:
                    raise job.error
                logits, stats = job.value
                stats = dict(stats)
                if not owner:
                    stats.update(reused=len(job.row[0]), prompt_ms=0, decode_ms=0, scored_windows=0, deduplicated_rows=1)
                result.append((list(logits), stats))
            return result

    def _drain(self):
        while True:
            with self.cv:
                if not self.pending:
                    self.running = False
                    return
                wave, self.pending = self.pending[:256], self.pending[256:]
            # Admission to the native lease can collect more work, without an artificial batching delay.
            def collect():
                with self.cv:
                    room = 256 - len(wave)
                    wave.extend(self.pending[:room])
                    del self.pending[:room]
                    control.jobs[:] = [job for _, job in wave]
                    return [job.row for _, job in wave]
            control = Live([job for _, job in wave], self.cv)
            try:
                values = self.execute(collect, control)
                if len(values) != len(wave):
                    raise RuntimeError("Jev wave result count mismatch")
                error = None
            except Exception as exc:
                values, error = [], exc
            with self.cv:
                for i, (key, job) in enumerate(wave):
                    job.value = values[i] if error is None else None
                    job.error, job.done = error, True
                    if self.jobs.get(key) is job:
                        self.jobs.pop(key)
                finished = not self.pending
                if finished:
                    self.running = False
                self.cv.notify_all()
                if finished:
                    return
