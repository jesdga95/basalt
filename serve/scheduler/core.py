"""Fair agent admission: bounded per-tenant / per-agent queues, deficit round robin, aging, prefill/decode interleave.

A `Grant` is a lease on one admission slot plus `cost_tokens` of its kind's in-flight token budget; it is held until
`release()`. A request that finds nothing in flight and nothing queued takes the solo fast path: one lock acquisition,
no queue, no pressure probe. Disabled (the default) the scheduler returns a shared, already granted no-op `Grant`.

Order of service when requests have to wait:
  1. aged: a queued head older than `aging_s` goes first, oldest first (starvation guard, deficits may go negative);
  2. kind: decode and prefill take turns by per-round token budgets (`round_tokens`), decode first within a round;
  3. tenant, then agent: deficit round robin by `cost_tokens` (tenant quantum scaled by its weight), FIFO per agent.
Everything is under one lock; waiting is on per-grant events, so `wait()` is thread safe and `await grant` works.
"""
from __future__ import annotations

import asyncio
import collections
import heapq
import itertools
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

PREFILL, DECODE = "prefill", "decode"
KINDS = (DECODE, PREFILL)                  # service order inside a round: decode is latency sensitive
QUEUED, GRANTED, RELEASED, CANCELLED, EXPIRED = "queued", "granted", "released", "cancelled", "expired"
WAIT_SAMPLES = 512                         # queue-wait samples kept per kind for the percentiles
TENANTS_TRACKED = 512                      # idle tenants' counters kept before the oldest are dropped


class SchedulerError(Exception):
    pass


class AdmissionRejected(SchedulerError):
    """The queue is full (or the scheduler closed): `reason` is queue_full_agent|queue_full_tenant|queue_full_total|closed."""

    def __init__(self, reason: str, tenant: str = "", agent: str = ""):
        super().__init__(f"admission rejected: {reason} (tenant={tenant!r} agent={agent!r})")
        self.reason, self.tenant, self.agent = reason, tenant, agent


class AdmissionCancelled(SchedulerError):
    pass


class AdmissionExpired(SchedulerError):
    """The deadline passed while the request was still queued."""


@dataclass(frozen=True)
class Request:
    tenant: str
    agent: str
    cost_tokens: int
    kind: str = PREFILL                      # "prefill" | "decode"
    deadline: float | None = None            # absolute `time.monotonic()` second by which it must be admitted

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.cost_tokens < 0:
            raise ValueError("cost_tokens must be >= 0")


@dataclass
class SchedulerConfig:
    enabled: bool = False                    # off: submit() is a no-op and returns an already granted Grant
    max_inflight: int = 0                    # concurrent grants (batch slots); 0: unlimited
    max_queue_per_agent: int = 32
    max_queue_per_tenant: int = 128
    max_queue_total: int = 1024
    prefill_inflight_tokens: int = 0         # in-flight cost cap per kind, 0: none; a lone grant may exceed it
    decode_inflight_tokens: int = 0
    round_tokens: dict = field(default_factory=lambda: {PREFILL: 8192, DECODE: 2048})  # interleave budget per round
    quantum_tokens: int = 256                # DRR quantum of a weight-1 flow: its burst before the next flow goes
    aging_s: float = 30.0                    # a head waiting this long goes first (oldest first): a starvation floor
    tenant_weights: dict = field(default_factory=dict)
    pressure_poll_s: float = 0.05            # how often a pressure hold is re-probed without other events
    background: bool = True                  # timer thread for deadlines / pressure; False: call poll() yourself


class Grant:
    """Admission result. Blocking: `wait()` / `with grant:`; asyncio: `await grant`. Always `release()` when done."""
    __slots__ = ("request", "kind", "cost", "seq", "t_submit", "t_grant", "bypass", "state", "_sched", "_event",
                 "_listeners", "__weakref__")

    def __init__(self, sched, request: Request, seq: int, now: float):
        self._sched, self.request, self.seq, self.t_submit = sched, request, seq, now
        self.kind, self.cost = request.kind, max(1, int(request.cost_tokens))
        self.t_grant, self.bypass, self.state = None, False, QUEUED
        self._event, self._listeners = threading.Event(), []

    @property
    def queue_wait_s(self) -> float | None:
        return None if self.t_grant is None else self.t_grant - self.t_submit

    def done(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """True once admitted; False on timeout (still queued). Raises AdmissionCancelled / AdmissionExpired."""
        if not self._event.wait(timeout):
            return False
        self._raise_if_failed()
        return True

    def _raise_if_failed(self) -> None:
        if self.state == CANCELLED:
            raise AdmissionCancelled("admission cancelled")
        if self.state == EXPIRED:
            raise AdmissionExpired("deadline passed while queued")

    def cancel(self) -> bool:
        """Leave the queue; True if it was still queued. A granted request is released instead (False)."""
        return self._sched._cancel(self) if self._sched else False

    def release(self) -> None:
        """Give the slot and tokens back (idempotent). Releasing while queued cancels."""
        if self._sched is not None:
            self._sched._release(self)

    def __enter__(self):
        self.wait()
        return self

    def __exit__(self, *exc):
        self.release()

    def _listen(self, cb: Callable[[], None]) -> None:
        if self._sched is None or self._event.is_set():
            return cb()
        with self._sched._lock:
            if self._event.is_set():
                return cb()
            self._listeners.append(cb)

    def __await__(self):
        return self._await().__await__()

    async def _await(self):
        if not self._event.is_set():
            loop = asyncio.get_running_loop()
            fut = loop.create_future()

            def wake():
                try:
                    loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(None))
                except RuntimeError:         # loop closed
                    pass
            self._listen(wake)
            try:
                await fut
            except asyncio.CancelledError:
                self.cancel()
                raise
        self._raise_if_failed()
        return self


_NOOP = Grant(None, Request("", "", 0), 0, 0.0)
_NOOP.state = GRANTED
_NOOP._event.set()


class _Agent:
    __slots__ = ("q", "deficit")

    def __init__(self):
        self.q, self.deficit = collections.deque(), 0.0


class _Tenant:
    __slots__ = ("agents", "active", "cursor", "deficit", "pend")

    def __init__(self):
        self.agents: dict[str, _Agent] = {}
        self.active: list[str] = []          # agents with a queue, DRR order
        self.cursor, self.deficit, self.pend = 0, 0.0, 0


class _Tier:
    """The queue of one kind: tenants -> agents -> FIFO of grants."""
    __slots__ = ("tenants", "active", "cursor", "count")

    def __init__(self):
        self.tenants: dict[str, _Tenant] = {}
        self.active: list[str] = []
        self.cursor, self.count = 0, 0


def _drr_pick(n: int, cursor: int, deficit: Callable[[int], float], cost: Callable[[int], float],
              top_up: Callable[[float], None], need: Callable[[int], float]) -> int:
    """Index of the first flow from `cursor` whose deficit covers its head cost; when none does, every flow's
    deficit grows by the number of whole rounds the nearest one needs (no per-round spinning)."""
    while True:
        for i in range(n):
            j = (cursor + i) % n
            if deficit(j) >= cost(j):
                return j
        top_up(min(math.ceil(need(j)) for j in range(n)))


class AgentScheduler:
    def __init__(self, config: SchedulerConfig | None = None, *, pressure: Callable[[], object] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = config or SchedulerConfig()
        self.pressure, self.clock = pressure, clock
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._tiers = {k: _Tier() for k in KINDS}
        self._credit = {k: float(self.cfg.round_tokens.get(k, 0)) for k in KINDS}
        self._heap: list = []                # (deadline, seq, grant), lazily cleaned
        self._seq = itertools.count(1)
        self._queued = self._inflight = 0
        self._inflight_tokens = {k: 0 for k in KINDS}
        self._weights = dict(self.cfg.tenant_weights)
        self._reaper: threading.Thread | None = None
        self._closed = False
        self._held = False                   # last pressure verdict
        self._waits = {k: collections.deque(maxlen=WAIT_SAMPLES) for k in KINDS}
        self._tenants: dict[str, dict] = {}
        self._n = collections.Counter()      # granted, bypassed, cancelled, expired, rejected, pressure_errors, ...
        self._rejected = collections.Counter()

    # ----- public API -----------------------------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def set_weight(self, tenant: str, weight: float) -> None:
        if weight <= 0:
            raise ValueError("weight must be > 0")
        with self._lock:
            self._weights[tenant] = float(weight)

    def submit(self, req: Request) -> Grant:
        """Admission for `req`. Raises AdmissionRejected when its queues are full; otherwise returns a Grant that is
        either already granted (solo / free capacity) or queued (`wait()`, `await`, or poll `done()`)."""
        if not self.cfg.enabled:
            return _NOOP
        now = self.clock()
        with self._lock:
            if self._closed:
                self._reject("closed", req)
            g = Grant(self, req, next(self._seq), now)
            if req.deadline is not None and req.deadline <= now:
                self._resolve(g, EXPIRED)
                self._n["expired"] += 1
                self._tstat(req.tenant)["expired"] += 1
                return g
            if self._inflight == 0 and self._queued == 0:    # solo: nothing to be fair to
                g.bypass = True
                self._n["bypassed"] += 1
                self._admit(g, now)
                return g
            self._enqueue(g, now)
            self._pump(now)
            return g

    def poll(self) -> None:
        """Expire due deadlines and admit whatever now fits (the timer thread's job; callable by hand)."""
        if not self.cfg.enabled:
            return
        with self._lock:
            now = self.clock()
            self._expire_due(now)
            self._pump(now)

    def close(self) -> None:
        """Reject new requests and cancel every queued one."""
        with self._lock:
            self._closed = True
            for k in KINDS:
                for t in list(self._tiers[k].tenants.values()):
                    for a in list(t.agents.values()):
                        for g in list(a.q):
                            self._dequeue(g)
                            self._resolve(g, CANCELLED)
                            self._n["cancelled"] += 1
            self._cv.notify_all()

    # ----- internals; everything below runs under self._lock ----------------------------------------------------

    def _reject(self, reason: str, req: Request):
        self._n["rejected"] += 1
        self._rejected[reason] += 1
        self._tstat(req.tenant)["rejected"] += 1
        raise AdmissionRejected(reason, req.tenant, req.agent)

    def _tstat(self, tenant: str) -> dict:
        s = self._tenants.get(tenant)
        if s is None:
            if len(self._tenants) >= TENANTS_TRACKED:
                for name in [n for n, v in self._tenants.items() if not v["queued"] and not v["inflight"]]:
                    del self._tenants[name]
                    if len(self._tenants) < TENANTS_TRACKED * 3 // 4:
                        break
            s = self._tenants[tenant] = {"queued": 0, "inflight": 0, "inflight_tokens": 0, "granted": 0,
                                         "granted_tokens": 0, "rejected": 0, "cancelled": 0, "expired": 0}
        return s

    def _resolve(self, g: Grant, state: str) -> None:
        g.state = state
        g._event.set()
        ls, g._listeners = g._listeners, []
        for cb in ls:
            cb()

    def _enqueue(self, g: Grant, now: float) -> None:
        r, cfg = g.request, self.cfg
        tiers = self._tiers
        total_agent = total_tenant = 0
        for k in KINDS:                       # bounds count a tenant's / agent's queued requests of both kinds
            t = tiers[k].tenants.get(r.tenant)
            if t:
                total_tenant += sum(len(a.q) for a in t.agents.values())
                a = t.agents.get(r.agent)
                total_agent += len(a.q) if a else 0
        if total_agent >= cfg.max_queue_per_agent:
            self._reject("queue_full_agent", r)
        if total_tenant >= cfg.max_queue_per_tenant:
            self._reject("queue_full_tenant", r)
        if self._queued >= cfg.max_queue_total:
            self._reject("queue_full_total", r)
        tier = tiers[g.kind]
        t = tier.tenants.get(r.tenant)
        if t is None:
            t = tier.tenants[r.tenant] = _Tenant()
        a = t.agents.get(r.agent)
        if a is None:
            a = t.agents[r.agent] = _Agent()
        if not a.q:
            if not t.active:
                tier.active.append(r.tenant)
            t.active.append(r.agent)
        a.q.append(g)
        tier.count += 1
        self._queued += 1
        self._tstat(r.tenant)["queued"] += 1
        if r.deadline is not None:
            heapq.heappush(self._heap, (r.deadline, g.seq, g))
        if self._reaper is None and cfg.background and (r.deadline is not None or self.pressure is not None):
            self._reaper = threading.Thread(target=self._reap_loop, name="agent-scheduler", daemon=True)
            self._reaper.start()
        else:
            self._cv.notify_all()

    def _dequeue(self, g: Grant) -> None:
        """Take a queued grant out of its agent queue, dropping emptied flows (their deficit resets, as in DRR)."""
        r, tier = g.request, self._tiers[g.kind]
        t = tier.tenants[r.tenant]
        a = t.agents[r.agent]
        a.q.remove(g)
        tier.count -= 1
        self._queued -= 1
        self._tstat(r.tenant)["queued"] -= 1
        if a.q:
            return
        del t.agents[r.agent]
        i = t.active.index(r.agent)
        t.active.pop(i)
        if i < t.cursor:
            t.cursor -= 1
        if t.active:
            return
        del tier.tenants[r.tenant]
        i = tier.active.index(r.tenant)
        tier.active.pop(i)
        if i < tier.cursor:
            tier.cursor -= 1

    def _weight(self, tenant: str) -> float:
        return self._weights.get(tenant, 1.0)

    def _head(self, tier: _Tier) -> Grant:
        """DRR choice of the next grant of one kind, tenant first, then that tenant's agents. Top-ups only add
        credit, so asking without then granting (budget refusal) is harmless."""
        q = float(self.cfg.quantum_tokens)
        names, tenants = tier.active, tier.tenants

        def agent_of(t: _Tenant) -> int:
            ags = t.active
            ag = [t.agents[n] for n in ags]
            return _drr_pick(len(ag), t.cursor, lambda j: ag[j].deficit, lambda j: ag[j].q[0].cost,
                             lambda k: [setattr(x, "deficit", x.deficit + k * q) for x in ag],
                             lambda j: (ag[j].q[0].cost - ag[j].deficit) / q)

        def cost(j: int) -> float:
            t = tenants[names[j]]
            t.pend = agent_of(t)
            return t.agents[t.active[t.pend]].q[0].cost

        def top_up(k: int) -> None:
            for n in names:
                tenants[n].deficit += k * q * self._weight(n)

        j = _drr_pick(len(names), tier.cursor, lambda j: tenants[names[j]].deficit, cost, top_up,
                      lambda j: (cost(j) - tenants[names[j]].deficit) / (q * self._weight(names[j])))
        t = tenants[names[j]]
        t.pend = agent_of(t)                 # settled: the tenant's deficit already covers this head
        tier.cursor, t.cursor = j, t.pend
        return t.agents[t.active[t.pend]].q[0]

    def _charge(self, g: Grant) -> None:
        """Pay a grant's cost out of its flows' deficits (aged grants may drive them negative) and dequeue it."""
        t = self._tiers[g.kind].tenants[g.request.tenant]
        t.deficit -= g.cost
        t.agents[g.request.agent].deficit -= g.cost
        self._dequeue(g)

    def _oldest_head(self, kinds) -> Grant | None:
        best = None
        for k in kinds:
            for t in self._tiers[k].tenants.values():
                for a in t.agents.values():
                    h = a.q[0]
                    if best is None or h.t_submit < best.t_submit or (h.t_submit == best.t_submit and h.seq < best.seq):
                        best = h
        return best

    def _fits(self, g: Grant) -> bool:
        cap = getattr(self.cfg, f"{g.kind}_inflight_tokens")
        tok = self._inflight_tokens[g.kind]
        return not cap or tok == 0 or tok + g.cost <= cap

    def _select(self, kinds: list, now: float) -> Grant | None:
        cand = [k for k in kinds if self._tiers[k].count]
        if not cand:
            return None
        aged = self._oldest_head(cand)
        if aged is not None and now - aged.t_submit >= self.cfg.aging_s:
            return aged
        pick = next((k for k in cand if self._credit[k] > 0), None)
        if pick is None:                     # round over: every kind starts again with its full budget
            self._credit = {k: float(self.cfg.round_tokens.get(k, 0)) for k in KINDS}
            pick = next((k for k in cand if self._credit[k] > 0), cand[0])
        return self._head(self._tiers[pick])

    def _pump(self, now: float) -> None:
        if not self._queued:
            self._held = False
            return
        self._held = self._pressured()
        if self._held:
            return
        kinds = list(KINDS)
        cap = self.cfg.max_inflight
        while self._queued and kinds and (not cap or self._inflight < cap):
            g = self._select(kinds, now)
            if g is None:
                break
            if not self._fits(g):            # block that kind behind its selected head; the other kind may go on
                kinds.remove(g.kind)
                continue
            self._charge(g)
            self._credit[g.kind] -= g.cost
            self._admit(g, now)

    def _pressured(self) -> bool:
        if self.pressure is None or self._inflight == 0:   # nothing in flight to drain: always make progress
            return False
        try:
            return bool(self.pressure())
        except Exception:                    # a broken probe must not stop serving
            self._n["pressure_errors"] += 1
            return False

    def _admit(self, g: Grant, now: float) -> None:
        g.t_grant = now
        self._inflight += 1
        self._inflight_tokens[g.kind] += g.cost
        s = self._tstat(g.request.tenant)
        s["inflight"] += 1
        s["inflight_tokens"] += g.cost
        s["granted"] += 1
        s["granted_tokens"] += g.cost
        self._n["granted"] += 1
        if not g.bypass:
            self._waits[g.kind].append(now - g.t_submit)
        self._resolve(g, GRANTED)

    def _expire_due(self, now: float) -> None:
        h = self._heap
        while h and (h[0][2].state != QUEUED or h[0][0] <= now):
            _, _, g = heapq.heappop(h)
            if g.state == QUEUED:
                self._dequeue(g)
                self._resolve(g, EXPIRED)
                self._n["expired"] += 1
                self._tstat(g.request.tenant)["expired"] += 1
        if len(h) > 4 * self._queued + 64:   # granted requests leave stale entries behind: compact
            h[:] = [e for e in h if e[2].state == QUEUED]
            heapq.heapify(h)

    def _cancel(self, g: Grant) -> bool:
        if g.state == QUEUED:
            self._release(g)                 # releasing a queued grant is a cancel
            return g.state == CANCELLED
        self._release(g)                     # a granted one gives its slot back
        return False

    def _release(self, g: Grant) -> None:
        with self._lock:
            if g.state == QUEUED:
                self._dequeue(g)
                self._resolve(g, CANCELLED)
                self._n["cancelled"] += 1
                self._tstat(g.request.tenant)["cancelled"] += 1
            elif g.state == GRANTED:
                g.state = RELEASED
                self._inflight -= 1
                self._inflight_tokens[g.kind] -= g.cost
                s = self._tstat(g.request.tenant)
                s["inflight"] -= 1
                s["inflight_tokens"] -= g.cost
            else:
                return
            self._pump(self.clock())

    def _reap_loop(self) -> None:
        with self._lock:
            while not self._closed:
                now = self.clock()
                self._expire_due(now)
                self._pump(now)
                wait = self._heap[0][0] - now if self._heap else None
                if self._held or (self.pressure is not None and self._queued):
                    wait = self.cfg.pressure_poll_s if wait is None else min(wait, self.cfg.pressure_poll_s)
                if not self._queued or (wait is None):
                    break                    # nothing queued, or nothing time-driven left to watch
                self._cv.wait(max(0.0, wait))
            self._reaper = None

    # ----- metrics ----------------------------------------------------------------------------------------------

    @staticmethod
    def _pct(samples, p: float) -> float | None:
        if not samples:
            return None
        s = sorted(samples)
        return round(s[min(len(s) - 1, max(0, math.ceil(p * len(s)) - 1))] * 1000.0, 3)

    def metrics(self) -> dict:
        """JSON-ready snapshot for /metrics (`metrics["scheduler"] = scheduler.metrics()`)."""
        if not self.cfg.enabled:
            return {"enabled": False}
        with self._lock:
            waits = {k: list(v) for k, v in self._waits.items()}
            tenants = {n: dict(v) for n, v in self._tenants.items()}
            n, rej = dict(self._n), dict(self._rejected)
            out = {"enabled": True, "paused": self._held, "queue_depth": self._queued,
                   "queue_depth_by_kind": {k: self._tiers[k].count for k in KINDS},
                   "inflight": self._inflight, "inflight_tokens": dict(self._inflight_tokens),
                   "max_inflight": self.cfg.max_inflight}
        wait_ms = {k: {"n": len(v), "p50": self._pct(v, 0.5), "p95": self._pct(v, 0.95),
                       "max": self._pct(v, 1.0)} for k, v in waits.items()}
        out.update({"wait_ms": wait_ms, "ttft_wait_ms": {"p50": wait_ms[PREFILL]["p50"], "p95": wait_ms[PREFILL]["p95"]},
                    "granted": n.get("granted", 0), "bypassed": n.get("bypassed", 0), "cancelled": n.get("cancelled", 0),
                    "expired": n.get("expired", 0), "rejected": n.get("rejected", 0), "rejected_by_reason": rej,
                    "pressure_errors": n.get("pressure_errors", 0), "tenants": tenants})
        return out
