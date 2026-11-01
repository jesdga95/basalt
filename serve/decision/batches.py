"""Demand-driven sequence batches with independent caller completion."""
from __future__ import annotations

import collections
import threading

from serve.decision.admission import DecisionError


class BatchControl:
    def __init__(self, jobs):
        self.jobs = jobs

    def is_set(self):
        return all(job.done.is_set() or job.control.is_set() for job in self.jobs)

    def check(self):
        if self.is_set():
            raise DecisionError(499, "decision batch has no waiting callers")


class Job:
    def __init__(self, rows, control):
        self.rows, self.control = rows, control
        self.values = [None] * len(rows)
        self.done = threading.Event()
        self.error = None


class Batches:
    def __init__(self, owner):
        self.owner = owner
        self.lock = threading.Lock()
        self.pending = collections.deque()
        self.running = False

    def score(self, rows, control):
        control.check()
        if not rows:
            return []
        job = Job(rows, control)
        with self.lock:
            self.pending.append(job)
            if not self.running:
                self.running = True
                threading.Thread(target=self._run, daemon=True, name="jev-sequences").start()
        while not job.done.wait(0.1):
            control.check()
        control.check()
        if job.error:
            raise job.error
        return job.values

    def _run(self):
        while True:
            with self.lock:
                if not self.pending:
                    self.running = False
                    return
                first = self.pending.popleft()
            jobs = [first]
            try:
                with self.owner.lease(first.control):
                    # Collect only work already waiting at the engine boundary; no batching timer.
                    count = len(first.rows)
                    with self.lock:
                        while (self.pending and not any(row[3] for row in first.rows) and
                               not any(row[3] for row in self.pending[0].rows) and
                               count + len(self.pending[0].rows) <= 256):
                            job = self.pending.popleft()
                            jobs.append(job)
                            count += len(job.rows)
                    live = [job for job in jobs if not job.control.is_set()]
                    rows, owners = [], []
                    for job in live:
                        for i, row in enumerate(job.rows):
                            rows.append(row)
                            owners.append((job, i))

                    def completed(index, value):
                        job, i = owners[index]
                        job.values[i] = value
                        if all(v is not None for v in job.values):
                            job.done.set()

                    engine = self.owner.svc.engine
                    if rows:
                        control = BatchControl(live)
                        if callable(getattr(engine, "score_many", None)):
                            engine.score_many(rows, control, completed=completed)
                        else:
                            for i, row in enumerate(rows):
                                control.check()
                                values, stats = engine.score(*row, control)
                                completed(i, (values, dict(stats, batch_size=1, scored_windows=1, workspace_bytes=0)))
            except Exception as error:
                for job in jobs:
                    if not job.done.is_set():
                        job.error = error
            finally:
                for job in jobs:
                    if not job.done.is_set():
                        job.error = job.error or DecisionError(499, "decision cancelled before scoring")
                        job.done.set()
