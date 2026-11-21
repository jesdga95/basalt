"""Who holds the GPUs' memory: per-process figures from NVML, for the sleep and wake checks and their messages."""
from __future__ import annotations

import ctypes
import os

from serve.util.telemetry import _Nvml


class _ProcInfo(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong), ("gpu_instance", ctypes.c_uint),
                ("compute_instance", ctypes.c_uint)]


def _command(pid: int) -> str:
    """The first words of a process's command line, else its pid alone."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            words = f.read().decode("utf-8", "replace").split("\0")
        return " ".join(w for w in words[:3] if w)[:80] or f"pid {pid}"
    except OSError:
        return f"pid {pid}"


_NOT_AVAILABLE = 2**64 - 1      # NVML's answer for a process whose memory it cannot read (Windows, containers)


def processes(index: int) -> list[tuple[int, int | None]]:
    """(pid, bytes of memory or None when NVML cannot read it) of the compute processes on GPU `index`; [] when NVML
    cannot say."""
    nv = _Nvml(index)
    if not nv.ok():
        return []
    for fn in ("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses_v2"):
        try:
            call = getattr(nv.lib, fn)
        except AttributeError:
            continue
        n = ctypes.c_uint(64)
        arr = (_ProcInfo * 64)()
        if call(nv.dev, ctypes.byref(n), arr) == 0:
            return [(int(arr[i].pid), None if arr[i].used == _NOT_AVAILABLE else int(arr[i].used))
                    for i in range(min(n.value, 64))]
    return []


def gpu_count() -> int:
    """The GPUs NVML has, by trying indices (the reader has no count call of its own)."""
    n = 0
    while n < 16 and _Nvml(n).ok():
        n += 1
    return n


def process_by_gpu(pid: int, gpus: list[int] | None = None) -> dict[int, int] | None:
    """The memory process `pid` holds on each GPU, in MiB ({gpu: MiB}, the GPUs it is on); None when NVML does not list
    the process anywhere or cannot read its memory (a container, Windows): the caller must not take that for zero."""
    gpus = list(range(gpu_count())) if gpus is None else gpus
    out: dict[int, int] = {}
    for g in gpus:
        mine = [b for p, b in processes(g) if p == pid]
        if not mine:
            continue
        if any(b is None for b in mine):
            return None
        out[g] = sum(mine) >> 20
    return out or None


def process_mib(pid: int, gpus: list[int] | None = None) -> int | None:
    """The memory process `pid` holds across `gpus` in MiB; None as for process_by_gpu."""
    by = process_by_gpu(pid, gpus)
    return None if by is None else sum(by.values())


def free_mib(index: int) -> int | None:
    """Free memory on GPU `index` in MiB; None when unreadable."""
    try:
        nv = _Nvml(index)
        if not nv.ok():
            return None
        m = nv.Mem()
        if nv.lib.nvmlDeviceGetMemoryInfo(nv.dev, ctypes.byref(m)) != 0:
            return None
        return int(m.free >> 20)
    except Exception:
        return None


def holders(gpus: list[int] | None = None, ignore: int | None = None) -> str:
    """The processes holding GPU memory, as "pid 123 (python -m ...) 20.0 GiB on GPU 0; ..."; "" when none is listed."""
    gpus = list(range(gpu_count())) if gpus is None else gpus
    rows = []
    for g in gpus:
        for pid, used in processes(g):
            if pid != ignore and pid != os.getpid():
                size = "an unreadable amount" if used is None else f"{used / 2**30:.1f} GiB"
                rows.append(f"pid {pid} ({_command(pid)}) {size} on GPU {g}")
    return "; ".join(rows)
