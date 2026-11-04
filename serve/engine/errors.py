"""Errors raised for a request by the engine process and the server."""
from __future__ import annotations



class EngineDied(RuntimeError):
    """The engine process ended mid-request."""


class EngineStartError(EngineDied):
    """The engine ended before READY; the message is its error lines, else its last log line."""

    def __init__(self, message: str, log: str | None = None):
        super().__init__(message)
        self.log = log


class EngineStarting(RuntimeError):
    """The engine has not said READY yet, so there is no context size to plan with (a 503)."""


class ModelBusy(RuntimeError):
    """A model control was refused because requests are active or queued."""


class EngineSilent(EngineDied):
    """The engine was silent too long (or ignored a STOP) and the server ended it; the next request restarts it."""


class EngineError(ValueError):
    """The engine failed a request for a reason of its own (a CUDA error, a timed-out device wait): a 500, not the
    request's fault. A ValueError, so every path that ends a request on the engine's ERR line still does."""


class EngineStuck(RuntimeError):
    """The engine survived QUIT, terminate and kill."""


class GpuBusy(RuntimeError):
    """The model is unloaded and free VRAM is below min_free_vram_mib, so the engine is not started."""
