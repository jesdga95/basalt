"""Suspending a running engine's GPU state with the driver's process checkpoint.

The CUDA driver can lock a process, move the contents of its GPU memory to host memory and release the VRAM, then put
them back at the same addresses (cuCheckpointProcessLock / Checkpoint / Restore / Unlock). A sleeping engine keeps its
process, its RAM copy of the experts, its conversations and its learned placement, and holds no VRAM. The driver offers
this on Linux only; elsewhere `available()` is false and the server unloads instead.
"""
from __future__ import annotations

import ctypes
import sys
import threading

_lock = threading.Lock()
_cu = None
_tried = False


def _driver():
    """The CUDA driver with the checkpoint entry points, initialized once; None when absent."""
    global _cu, _tried
    with _lock:
        if _tried:
            return _cu
        _tried = True
        if not sys.platform.startswith("linux"):
            return None
        try:
            cu = ctypes.CDLL("libcuda.so.1")
            for name in ("cuCheckpointProcessLock", "cuCheckpointProcessCheckpoint", "cuCheckpointProcessRestore",
                         "cuCheckpointProcessUnlock"):
                fn = getattr(cu, name)
                fn.argtypes = [ctypes.c_int, ctypes.c_void_p]
                fn.restype = ctypes.c_int
            cu.cuInit.argtypes = [ctypes.c_uint]
            cu.cuInit.restype = ctypes.c_int
            if cu.cuInit(0) != 0:
                return None
        except (OSError, AttributeError):
            return None
        _cu = cu
        return _cu


def available() -> bool:
    return _driver() is not None


def suspend(pid: int) -> str | None:
    """Lock `pid` and move its GPU state to host memory; None on success, else why not (the process runs on)."""
    cu = _driver()
    if cu is None:
        return "the driver cannot suspend a process here"
    rc = cu.cuCheckpointProcessLock(int(pid), None)
    if rc != 0:
        return f"the driver could not lock the engine (CUresult {rc})"
    rc = cu.cuCheckpointProcessCheckpoint(int(pid), None)
    if rc != 0:
        cu.cuCheckpointProcessUnlock(int(pid), None)
        return f"the driver could not save the engine's GPU state (CUresult {rc})"
    return None


def resume(pid: int) -> str | None:
    """Put `pid`'s GPU state back and unlock it; None on success, else why not."""
    cu = _driver()
    if cu is None:
        return "the driver cannot resume a process here"
    rc = cu.cuCheckpointProcessRestore(int(pid), None)
    if rc != 0:
        return f"the driver could not restore the engine's GPU state (CUresult {rc})"
    rc = cu.cuCheckpointProcessUnlock(int(pid), None)
    if rc != 0:
        return f"the driver could not unlock the engine (CUresult {rc})"
    return None
