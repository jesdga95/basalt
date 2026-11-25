"""SSE stream semaphore cleanup on socket errors and UIState worker resilience."""
import contextlib
import io
import json
import threading
import time
import types
import unittest

from serve.api.logs import stream_logs
from serve.util.logbuf import LogBuffer
from serve.util.ui_state import UIState, stream_state


class BrokenConn:
    def __init__(self, *, timeout=None, fail_get=False, fail_restore=False):
        self.timeout = timeout
        self.fail_get = fail_get
        self.fail_restore = fail_restore
        self.set_calls = []

    def gettimeout(self):
        if self.fail_get:
            raise OSError(9, "Bad file descriptor")
        return self.timeout

    def settimeout(self, value):
        self.set_calls.append(value)
        if self.fail_restore and len(self.set_calls) >= 2:
            raise OSError(9, "Socket closed")


class BrokenWriter:
    def __init__(self, *, fail_write=True):
        self.fail_write = fail_write
        self.chunks = []

    def write(self, data):
        if self.fail_write:
            raise BrokenPipeError("peer disconnected")
        self.chunks.append(data)

    def flush(self):
        pass


class FakeHandler:
    def __init__(self, path, conn, writer, log_stop=None):
        self.path = path
        self.connection = conn
        self.wfile = writer
        self.close_connection = False
        self.server = types.SimpleNamespace(log_stop=log_stop or threading.Event())
        self.json_responses = []
        self.sse_started = False

    def _json(self, status, body):
        self.json_responses.append((status, body))

    def _sse(self):
        self.sse_started = True


class FakeService:
    def __init__(self):
        self.status_lock = threading.Lock()
        self.status = {}
        self.live_reqs = {}
        self.loading = False
        self.requests_done = 0
        self.waiting = []
        self.engine = types.SimpleNamespace(asleep=False)

    def loaded(self):
        return True


class FakeRouter:
    def __init__(self, metrics_fn=None):
        self.cv = threading.Condition()
        self.current = FakeService()
        self.services = [self.current]
        self.queue = []
        self.swap = None
        self._metrics_fn = metrics_fn or (lambda svc, include_slots=False: {"ok": True, "slots": []})

    def shown(self):
        return self.current

    def metrics(self, svc, include_slots=False):
        return self._metrics_fn(svc, include_slots=include_slots)


def assert_all_slots_available(tc, sem, limit=8):
    acquired = 0
    try:
        for _ in range(limit):
            tc.assertTrue(sem.acquire(blocking=False), f"leaked semaphore slot after {acquired} acquires")
            acquired += 1
        tc.assertFalse(sem.acquire(blocking=False))
    finally:
        for _ in range(acquired):
            sem.release()


class LogStreamSemaphore(unittest.TestCase):
    def test_releases_slot_when_gettimeout_raises_oserror(self):
        logs = LogBuffer()
        for _ in range(10):
            conn = BrokenConn(fail_get=True)
            handler = FakeHandler("/api/logs/stream", conn, BrokenWriter())
            stream_logs(handler, logs)
            self.assertEqual(handler.json_responses, [])
            self.assertEqual(conn.set_calls, [])
        assert_all_slots_available(self, logs.clients)

    def test_releases_slot_when_finally_settimeout_raises_oserror(self):
        logs = LogBuffer()
        for _ in range(10):
            conn = BrokenConn(timeout=12.0, fail_restore=True)
            handler = FakeHandler("/api/logs/stream", conn, BrokenWriter(fail_write=True))
            stream_logs(handler, logs)
            self.assertEqual(handler.json_responses, [])
            self.assertEqual(conn.set_calls, [5, 12.0])
        assert_all_slots_available(self, logs.clients)

    def test_restores_none_timeout_when_original_socket_was_blocking(self):
        logs = LogBuffer()
        conn = BrokenConn(timeout=None)
        handler = FakeHandler("/api/logs/stream", conn, BrokenWriter(fail_write=True))
        stream_logs(handler, logs)
        self.assertEqual(conn.set_calls, [5, None])
        assert_all_slots_available(self, logs.clients)


class StateStreamSemaphore(unittest.TestCase):
    def test_releases_slot_when_gettimeout_raises_oserror(self):
        state = UIState(FakeRouter())
        self.addCleanup(state.close)
        for _ in range(10):
            conn = BrokenConn(fail_get=True)
            handler = FakeHandler("/api/state/stream", conn, BrokenWriter())
            stream_state(handler, state)
            self.assertEqual(handler.json_responses, [])
            self.assertEqual(conn.set_calls, [])
            self.assertIsNone(state.worker)
        assert_all_slots_available(self, state.clients)

    def test_releases_slot_when_finally_settimeout_raises_oserror(self):
        state = UIState(FakeRouter())
        self.addCleanup(state.close)
        for _ in range(10):
            conn = BrokenConn(timeout=15.0, fail_restore=True)
            handler = FakeHandler("/api/state/stream", conn, BrokenWriter(fail_write=True))
            stream_state(handler, state)
            self.assertEqual(handler.json_responses, [])
            self.assertEqual(conn.set_calls, [5, 15.0])
        assert_all_slots_available(self, state.clients)

    def test_restores_none_timeout_when_original_socket_was_blocking(self):
        state = UIState(FakeRouter())
        self.addCleanup(state.close)
        conn = BrokenConn(timeout=None)
        handler = FakeHandler("/api/state/stream", conn, BrokenWriter(fail_write=True))
        stream_state(handler, state)
        self.assertEqual(conn.set_calls, [5, None])
        assert_all_slots_available(self, state.clients)


class UIStateWorkerResilience(unittest.TestCase):
    def wait_version(self, state, target, timeout=3.0):
        deadline = time.monotonic() + timeout
        with state.cv:
            while state.version < target:
                remaining = deadline - time.monotonic()
                self.assertGreater(remaining, 0, f"timed out waiting for version {target} (at {state.version})")
                state.cv.wait(remaining)
            return state.frame, state.version

    def test_worker_survives_transient_metrics_exception_and_deduplicates_warning(self):
        calls = 0

        def flaky_metrics(svc, include_slots=False):
            nonlocal calls
            calls += 1
            if calls <= 2:
                raise RuntimeError("transient metrics failure")
            return {"ok": True, "calls": calls, "slots": []}

        state = UIState(FakeRouter(metrics_fn=flaky_metrics))
        self.addCleanup(state.close)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            state.start()
            frame, version = self.wait_version(state, 1)
            state.close()

        self.assertGreaterEqual(version, 1)
        self.assertTrue(frame.startswith(b"event: state\ndata: "))
        payload = json.loads(frame.removeprefix(b"event: state\ndata: "))
        self.assertEqual(payload, {"ok": True, "calls": 3, "slots": []})
        warnings = [line for line in buf.getvalue().splitlines() if "UI state update failed" in line]
        self.assertEqual(warnings, ["[server] warning: UI state update failed (transient metrics failure)"])

    def test_worker_survives_transient_key_exception(self):
        state = UIState(FakeRouter())
        self.addCleanup(state.close)
        orig_key = state._key
        failed = False

        def flaky_key():
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("transient key error")
            return orig_key()

        state._key = flaky_key
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            state.start()
            frame, version = self.wait_version(state, 1)
            self.assertTrue(state.worker.is_alive())
            state.close()

        self.assertGreaterEqual(version, 1)
        self.assertEqual(json.loads(frame.removeprefix(b"event: state\ndata: ")), {"ok": True, "slots": []})
        self.assertIn("[server] warning: UI state update failed (transient key error)", buf.getvalue())

    def test_start_restarts_dead_worker_and_ignores_after_close(self):
        state = UIState(FakeRouter())
        self.addCleanup(state.close)

        dead = threading.Thread(target=lambda: None, daemon=True)
        dead.start()
        dead.join(timeout=2)
        self.assertFalse(dead.is_alive())

        state.worker = dead
        state.start()
        live = state.worker
        self.assertIsNot(live, dead)
        self.assertTrue(live.is_alive())
        frame, version = self.wait_version(state, 1)
        self.assertGreaterEqual(version, 1)
        self.assertEqual(json.loads(frame.removeprefix(b"event: state\ndata: ")), {"ok": True, "slots": []})

        state.start()
        self.assertIs(state.worker, live)

        state.close()
        self.assertFalse(live.is_alive())
        state.start()
        self.assertIs(state.worker, live)


if __name__ == "__main__":
    unittest.main()
