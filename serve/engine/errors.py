"""Errors raised for a request by the engine process and the server."""
from __future__ import annotations

# The ERR codes the engine gives a machine-readable form: `ERR <code> key=N ...: <message>`
ERR_CODES = ("context_overflow", "context_no_room")


def split_err(text: str) -> tuple[str | None, dict[str, int], str]:
    """(code, fields, message) of an ERR text (the line without `ERR `): the code and integer fields of a coded error,
    else (None, {}, text)."""
    head, sep, message = text.partition(": ")
    words = head.split()
    if sep and len(words) > 1 and words[0] in ERR_CODES:
        fields = {}
        for w in words[1:]:
            k, _, v = w.partition("=")
            if not v.lstrip("-").isdigit():
                return None, {}, text
            fields[k] = int(v)
        return words[0], fields, message
    return None, {}, text


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


class ClientGone(Exception):
    """The client left while its request waited for its model's turn."""


class EngineStuck(RuntimeError):
    """The engine survived QUIT, terminate and kill."""


class GpuBusy(RuntimeError):
    """The model is unloaded and free VRAM is below min_free_vram_mib, so the engine is not started."""
