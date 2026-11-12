"""Router: one Service per configured model, one model loaded at a time, requests picking theirs by name.

A swap ends one engine process and starts the next. A request holds a turn from before its model is used until its reply
ends. Turns follow one rule, so mixed traffic cannot starve or thrash:
  * A request for the loaded model goes at once while nothing waits. Once a request for another model waits, later
    arrivals queue behind it, whatever their model.
  * The queue's head decides the next model. Its swap starts when every running request has finished.
  * A loaded model takes all the requests queued for it at that moment, in one batch.
  * A failed load fails that batch with the load's error and the queue moves on; the next request for it tries again.
"""
from __future__ import annotations

import collections
import threading
import time

from serve.engine.errors import EngineDied, EngineStartError, EngineStuck, GpuBusy
from serve.live import LiveRequests

WAITING, GRANTED, FAILED, LEFT = "waiting", "granted", "failed", "left"


class Turn:
    """A request's place: waiting in the queue, then granted until it leaves. A preload holds nothing once granted."""
    __slots__ = ("svc", "state", "error", "holds", "at")

    def __init__(self, svc, holds: bool = True):
        self.svc, self.holds, self.state, self.error, self.at = svc, holds, WAITING, None, time.time()


def load_failure(name: str, e: Exception) -> Exception:
    """The error a waiting request gets for a failed load: the kind the API maps to its status, naming the model."""
    text = f"{name} did not load: {e}"
    if isinstance(e, EngineStartError):
        return EngineStartError(text, e.log)
    return (type(e) if isinstance(e, (GpuBusy, EngineStuck)) else EngineDied)(text)


class Router:
    def __init__(self, services: list):
        self.services = list(services)
        self.front = self.services[0]                   # holds the settings every model shares: key, hosts, image sources
        self.current = self.front if len(self.services) == 1 else None
        self.cv = threading.Condition()
        self.queue: collections.deque[Turn] = collections.deque()
        self.running = 0                                # granted requests that have not left
        self.swap = None                                # the swap under way: {"from", "to", "phase", "since"}
        self.swaps = 0                                  # swaps done since start
        self.errors: dict[int, str] = {}                # id(service) -> its last load's failure, until one succeeds
        self.stopping = False
        self.worker = None
        self.live = LiveRequests()                      # the running requests, for answer now

    @property
    def multi(self) -> bool:
        return len(self.services) > 1

    def service_for(self, name) -> object | None:
        """The service answering to `name`: a listed name or a model file's name."""
        if isinstance(name, str):
            for svc in self.services:
                if name in svc.served_names():
                    return svc
        return None

    def shown(self):
        """The service the status routes describe: the swap's target, else the loaded model, else the first."""
        with self.cv:
            target = self.swap["svc"] if self.swap else self.current
        return target or self.front

    def models(self) -> list[dict]:
        """What /v1/models lists: every configured model, loaded or not."""
        out = []
        for svc in self.services:
            model = svc.model_entry()
            why = self.errors.get(id(svc))
            if why and model["status"]["value"] == "unloaded":
                model["status"] = {"value": "failed", "error": why}
            out.append(model)
        return out

    def swap_status(self) -> dict | None:
        """The swap under way, for /metrics and /health."""
        with self.cv:
            s = self.swap
            if not s:
                return None
            return {"from": s["from"], "to": s["svc"].model, "phase": s["phase"],
                    "elapsed_s": round(time.time() - s["since"], 1), "waiting": len(self.queue)}

    def metrics(self, svc) -> dict:
        """GET /metrics for `svc`, with the requests waiting for any model's turn and every model's row."""
        m = svc.metrics()
        now = time.time()
        with self.cv:
            waiting = [{"api": None, "model": t.svc.model, "prompt_tokens": None, "waited_s": round(now - t.at, 1)}
                       for t in self.queue if t.holds]
        m["live"]["queued"] += len(waiting)
        m["live"]["queue"] = m["live"]["queue"] + waiting
        return {**m, "models": self.models(), "swap": self.swap_status()}

    def join(self, svc) -> Turn:
        """Take a turn for `svc`: granted at once when its model is loaded and nothing waits, else queued."""
        turn = Turn(svc)
        with self.cv:
            if self.stopping:
                turn.state, turn.error = FAILED, EngineDied("the server is stopping")
            elif self.current is svc and not self.swap and not self.queue:
                turn.state = GRANTED
                self.running += 1
            else:
                self.queue.append(turn)
                self._start_worker()
                self.cv.notify_all()
        return turn

    def preload(self, svc) -> None:
        """Load `svc` through the queue, with no request behind it."""
        with self.cv:
            self.queue.append(Turn(svc, holds=False))
            self._start_worker()
            self.cv.notify_all()

    def wait(self, turn: Turn, timeout: float | None) -> bool:
        """Whether `turn` is granted after up to `timeout` seconds; raises the load's error when it failed."""
        with self.cv:
            self.cv.wait_for(lambda: turn.state != WAITING, timeout)
            if turn.state == FAILED:
                raise turn.error
            return turn.state == GRANTED

    def leave(self, turn: Turn) -> None:
        """The request ended, or left while queued."""
        with self.cv:
            if turn.state == GRANTED and turn.holds:
                self.running -= 1
            elif turn.state == WAITING:
                self.queue.remove(turn)
            turn.state = LEFT
            self.cv.notify_all()

    def stop(self) -> None:
        """Shutdown: the waiting requests get an error and no swap starts."""
        with self.cv:
            self.stopping = True
            for turn in self.queue:
                turn.state, turn.error = FAILED, EngineDied("the server is stopping")
            self.queue.clear()
            self.cv.notify_all()
        for svc in self.services:
            svc.stop()
        self.stop_telemetry()

    def vram(self, reserve_mib):
        """POST /v1/vram: the reserve is the machine's, so a model loaded later applies it too."""
        active = self.shown()
        for svc in self.services:
            if svc is not active:
                svc.vram_reserve = reserve_mib
        return active.vram(reserve_mib)

    def start_telemetry(self) -> None:
        """One hardware sampler over the cards of every model; the speeds are the shown model's."""
        current = getattr(self.front, "telemetry", None)
        if current is None or current.stopped():
            from serve.util.telemetry import Telemetry
            cards = list(dict.fromkeys(i for svc in self.services for i in (getattr(svc, "gpu_indices", None) or [0])))
            sampler = Telemetry(extra=lambda: self.shown().telemetry_extra(), gpu_indices=cards)
            for svc in self.services:
                svc.telemetry = sampler

    def stop_telemetry(self) -> None:
        sampler = getattr(self.front, "telemetry", None)
        if sampler is not None:
            sampler.stop()

    def _start_worker(self) -> None:
        if self.worker is None:
            self.worker = threading.Thread(target=self._run, daemon=True, name="router")
            self.worker.start()

    def _grant_batch(self, svc) -> None:
        """Every queued request for `svc` goes now. Caller holds the lock."""
        for turn in [t for t in self.queue if t.svc is svc]:
            self.queue.remove(turn)
            turn.state = GRANTED
            self.running += bool(turn.holds)
        self.cv.notify_all()

    def _fail_batch(self, svc, error: Exception) -> None:
        for turn in [t for t in self.queue if t.svc is svc]:
            self.queue.remove(turn)
            turn.state, turn.error = FAILED, error
        self.cv.notify_all()

    def _run(self) -> None:
        while True:
            with self.cv:
                while True:
                    if self.stopping:
                        return
                    head = self.queue[0].svc if self.queue else None
                    if head is not None and head is self.current:
                        self._grant_batch(head)
                    elif head is not None and self.running == 0:
                        break
                    else:
                        self.cv.wait()
                self.swap = {"svc": head, "from": self.current.model if self.current else None,
                             "phase": "unloading" if self.current else "loading", "since": time.time()}
                old, waiting = self.current, len(self.queue)
            failure = self._swap(old, head, waiting)
            with self.cv:
                if failure is None:
                    self.current = head
                    self.swaps += 1
                    self.errors.pop(id(head), None)
                    self._grant_batch(head)
                else:
                    self.errors[id(head)] = str(failure)
                    self._fail_batch(head, load_failure(head.model, failure))
                self.swap = None
                self.cv.notify_all()

    def _swap(self, old, new, waiting: int) -> Exception | None:
        """End `old`, start `new`; the failure, or None."""
        t = time.time()
        new.load_kind, new.loading = "weights", True    # the status routes show the load from now on
        if hasattr(new.engine, "load_status"):
            new.engine.load_status = {}
        print(f"[server] {f'swap {old.model} ->' if old else 'loading'} {new.model} ({waiting} waiting)", flush=True)
        try:
            if old is not None:
                old.swap_out()                          # at its QUIT the engine spills its conversations to disk
                with self.cv:
                    self.current = None
                    self.swap["phase"] = "loading"
            new.load()
        except Exception as e:  # noqa: BLE001 - any failure ends in the batch's error, and the server stays up for the others
            print(f"[server] error: {new.model} did not load: {e}", flush=True)
            return e
        finally:
            new.loading = False
        print(f"[server] swap: {new.model} ready in {time.time() - t:.1f} s", flush=True)
        return None
