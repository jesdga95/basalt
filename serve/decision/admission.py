"""Bounded decision admission and writer-priority access to the resident engine."""
from __future__ import annotations

import contextlib
import threading
import time


class DecisionError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Control:
    def __init__(self, cancel, seconds):
        self.cancel = cancel
        self.deadline = time.monotonic() + seconds

    def is_set(self):
        return self.cancel.is_set() or time.monotonic() >= self.deadline

    def check(self):
        if self.cancel.is_set():
            raise DecisionError(499, "decision cancelled")
        if time.monotonic() >= self.deadline:
            raise DecisionError(504, "decision deadline exceeded")


class Gate:
    def __init__(self, pending):
        self.cv = threading.Condition()
        self.readers = self.writers = self.outstanding = 0
        self.writing = False
        self.limit = pending + 1

    @contextlib.contextmanager
    def shared(self, cancel):
        with self.cv:
            while self.writers or self.writing:
                if cancel.is_set():
                    raise DecisionError(499, "request cancelled while waiting for decision work")
                self.cv.wait(0.1)
            self.readers += 1
        try:
            yield
        finally:
            with self.cv:
                self.readers -= 1
                self.cv.notify_all()

    @contextlib.contextmanager
    def admit(self):
        with self.cv:
            if self.outstanding >= self.limit:
                raise DecisionError(429, "decision queue is full")
            self.outstanding += 1
        try:
            yield
        finally:
            with self.cv:
                self.outstanding -= 1
                self.cv.notify_all()

    @contextlib.contextmanager
    def exclusive(self, control):
        with self.cv:
            self.writers += 1
            try:
                while self.readers or self.writing:
                    control.check()
                    self.cv.wait(0.1)
                control.check()
                self.writing = True
            finally:
                self.writers -= 1
                self.cv.notify_all()
        try:
            yield
        finally:
            with self.cv:
                self.writing = False
                self.cv.notify_all()
