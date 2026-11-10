"""One bounded snapshot publisher for all UI clients; no per-client queues or engine work."""
import json
import threading
import time


class UIState:
    def __init__(self, router):
        self.router = router
        self.cv = threading.Condition()
        self.stop = threading.Event()
        self.clients = threading.BoundedSemaphore(8)
        self.version = 0
        self.frame = None
        self.worker = None

    def start(self):
        with self.cv:
            if self.worker is None:
                self.worker = threading.Thread(target=self._run, daemon=True)
                self.worker.start()

    def _key(self):
        router = self.router
        with router.cv:
            key = [id(router.current), tuple((id(t), t.state) for t in router.queue),
                   None if not router.swap else (id(router.swap["svc"]), router.swap["phase"])]
        for svc in router.services:
            with svc.status_lock:
                states = [svc.status] + [st for st, _ in svc.live_reqs.values()]
                key.append((svc.loading, svc.requests_done, tuple(svc.waiting),
                            tuple((s.get("started"), s.get("busy"), s.get("first_token"),
                                   s.get("phase"), s.get("queued")) for s in states)))
            key.append((svc.loaded(), getattr(svc.engine, "asleep", False)))
        return key

    def _run(self):
        previous, deadline = None, 0
        while not self.stop.is_set():
            key = self._key()
            if key != previous or time.monotonic() >= deadline:
                data = self.router.metrics(self.router.shown(), include_slots=True)
                frame = b"event: state\ndata: " + json.dumps(data, ensure_ascii=False).encode() + b"\n\n"
                with self.cv:
                    self.frame = frame
                    self.version += 1
                    self.cv.notify_all()
                previous, deadline = key, time.monotonic() + 1
            self.stop.wait(.1)

    def close(self):
        self.stop.set()
        with self.cv:
            self.cv.notify_all()
        if self.worker is not None:
            self.worker.join(timeout=5)


def stream_state(handler, state):
    if not state.clients.acquire(blocking=False):
        handler._json(503, {"error": {"message": "Too many UI streams (maximum 8)"}})
        return
    timeout = handler.connection.gettimeout()
    try:
        state.start()
        handler.connection.settimeout(5)
        handler._sse()
        handler.close_connection = True
        version = 0
        while not state.stop.is_set():
            with state.cv:
                state.cv.wait_for(lambda: state.stop.is_set() or state.version != version, timeout=10)
                if state.stop.is_set():
                    break
                frame = state.frame if version != state.version else None
                version = state.version
            handler.wfile.write(frame or b": heartbeat\n\n")
            handler.wfile.flush()
    except OSError:
        pass
    finally:
        handler.connection.settimeout(timeout)
        state.clients.release()
