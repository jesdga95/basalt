"""Start the engine so that it ends with the server, however the server ends.

The server's own path is graceful (QUIT, wait, terminate, kill). This is the guarantee behind it for the cases where
none of that runs: SIGKILL, a crash, a closed console, also while the model is still loading.

* Linux: the child gets PR_SET_PDEATHSIG = SIGKILL. The kernel ties the signal to the thread that forked the child,
  not to the process, so every start is made by one long-lived anchor thread; a request thread that starts an engine
  and then finishes would otherwise take the engine down with it.
* Windows: the child joins a job object that kills its members when the last handle closes; the only handle is ours.
* Everywhere: the child gets its own process group, so a terminal Ctrl+C reaches the server only, which then ends
  the engine the clean way.
"""
from __future__ import annotations

import concurrent.futures
import os
import queue
import signal
import subprocess
import sys
import threading

WINDOWS = os.name == "nt"
LINUX = sys.platform.startswith("linux")
NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)

PR_SET_PDEATHSIG = 1
_prctl = None
if LINUX:
    try:
        import ctypes
        _prctl = ctypes.CDLL(None, use_errno=True).prctl
        _prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
    except (OSError, AttributeError):
        _prctl = None


class _Anchor:
    """A thread that lives as long as the process and makes every start, so the death signal's parent is it."""

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="engine-anchor", daemon=True).start()

    def _run(self) -> None:
        while True:
            fn, future = self._jobs.get()
            try:
                future.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - handed to the caller
                future.set_exception(exc)

    def call(self, fn):
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._jobs.put((fn, future))
        return future.result()


_anchor: _Anchor | None = None
_anchor_lock = threading.Lock()


def _the_anchor() -> _Anchor:
    global _anchor
    with _anchor_lock:
        if _anchor is None:
            _anchor = _Anchor()
        return _anchor


def _child_setup(parent: int, then=None):
    """The child's preexec function: its own process group, and (Linux) SIGKILL when the starting thread ends."""
    def run() -> None:
        os.setpgid(0, 0)
        if _prctl is not None:
            _prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
            if os.getppid() != parent:                  # the parent died between fork and prctl
                os.kill(os.getpid(), signal.SIGKILL)
        if then is not None:
            then()
    return run


_job = None
_job_lock = threading.Lock()


def _contain(child: subprocess.Popen) -> bool:
    """Windows: put the child in the kill-on-close job. False, harmlessly, when that is not possible."""
    global _job
    import ctypes
    from ctypes import wintypes

    class Limits(ctypes.Structure):             # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD), ("IoCounters", ctypes.c_ulonglong * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
        k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        with _job_lock:
            if _job is None:
                job = k32.CreateJobObjectW(None, None)  # no security attributes: the handle is not inherited
                if not job:
                    return False
                info = Limits()
                info.LimitFlags = 0x2000                # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                    k32.CloseHandle(job)
                    return False
                _job = job
        return bool(k32.AssignProcessToJobObject(_job, int(child._handle)))
    except (AttributeError, OSError, ValueError):
        return False


def popen(args, **kw) -> subprocess.Popen:
    """``subprocess.Popen(args, **kw)`` for a process that must not outlive the server."""
    if WINDOWS:
        kw["creationflags"] = kw.get("creationflags", 0) | NEW_PROCESS_GROUP
        child = subprocess.Popen(args, **kw)
        _contain(child)
        return child
    kw["preexec_fn"] = _child_setup(os.getpid(), kw.get("preexec_fn"))
    return _the_anchor().call(lambda: subprocess.Popen(args, **kw))
