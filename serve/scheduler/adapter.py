"""Thin glue for serve.service.Service (wired by the integration step, not here). All of it is a no-op while the
scheduler is None or disabled, so the existing request path is unchanged."""
from __future__ import annotations

import threading
import time

from serve.scheduler.core import (PREFILL, AdmissionCancelled, AdmissionExpired, AgentScheduler, Grant, Request,
                                  SchedulerConfig)


def scheduler_for_service(service, *, enabled: bool = True, min_free_vram_mib: int = 0, **config) -> AgentScheduler:
    """A scheduler sized to the service's engine: `max_inflight` = its batch slots (unless given), and admission
    paused while NVML free VRAM is below `min_free_vram_mib` (0: no VRAM probe) and something is in flight."""
    config.setdefault("max_inflight", int(getattr(service.engine, "batch", 0) or 0))
    pressure = None
    if min_free_vram_mib > 0:
        def pressure() -> bool:
            free = service.free_vram_mib()
            return free is not None and free < min_free_vram_mib
    return AgentScheduler(SchedulerConfig(enabled=enabled, **config), pressure=pressure)


def submit(scheduler: AgentScheduler | None, tenant: str, agent: str, cost_tokens: int, kind: str = PREFILL,
           timeout_s: float | None = None) -> Grant | None:
    """Request admission; None when there is no scheduler (the caller proceeds as before). `timeout_s` becomes the
    queued deadline. Raises AdmissionRejected when the queues are full."""
    if scheduler is None:
        return None
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    return scheduler.submit(Request(tenant, agent, cost_tokens, kind, deadline))


def step(grant: Grant | None, cancel: threading.Event | None = None, beat_s: float = 1.0) -> str:
    """One bounded wait for the admission loop in Service.run: "granted", "cancelled" (the client's `cancel` event
    fired; the grant is cancelled) or "waiting" (send a keep-alive ping and call again). Raises AdmissionExpired."""
    if grant is None:
        return "granted"
    try:
        if grant.wait(beat_s):
            return "granted"
    except AdmissionCancelled:
        return "cancelled"
    if cancel is not None and cancel.is_set():
        grant.cancel()
        return "cancelled"
    return "waiting"


def release(grant: Grant | None) -> None:
    if grant is not None:
        grant.release()


def add_metrics(metrics: dict, scheduler: AgentScheduler | None) -> dict:
    """Attach the scheduler's counters to the GET /metrics body (in place) when a scheduler is configured."""
    if scheduler is not None and scheduler.enabled:
        metrics["scheduler"] = scheduler.metrics()
    return metrics


__all__ = ["scheduler_for_service", "submit", "step", "release", "add_metrics", "AdmissionExpired"]
