"""Hardware readings for GET /metrics, sampled once a second on a background thread.

GPU readings come from NVML through ctypes; CPU, RAM and disk from psutil, falling back to /proc (no disk rate).
Unreadable values are None and never stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import os
import platform
import sys
import threading
import time

HISTORY = 60


# ------------------------------------------------------------------------------------------------ NVML
class _Nvml:
    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, index=0):
        self.lib = self.dev = None
        names = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        for n in names:
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if self.lib is None:
            return
        try:
            init = getattr(self.lib, "nvmlInit_v2", None) or self.lib.nvmlInit
            if init() != 0:
                self.lib = None
                return
            h = ctypes.c_void_p()
            get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
                self.lib = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def name(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def driver_version(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlSystemGetDriverVersion(buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def cuda_version(self):
        """The newest CUDA the driver supports, as "13.3"."""
        v = ctypes.c_int()
        for fn in ("nvmlSystemGetCudaDriverVersion_v2", "nvmlSystemGetCudaDriverVersion"):
            try:
                if getattr(self.lib, fn)(ctypes.byref(v)) == 0:
                    return f"{v.value // 1000}.{v.value % 1000 // 10}"
            except (AttributeError, OSError):
                continue
        return None

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")        # drops at idle
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        return out


def gpu_reader(index=0):
    """The card's readings, from NVML."""
    return _Nvml(index)


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_name():
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        try:
            f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
        except (OSError, ValueError):
            return None

    def cpu(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ the sampler
GPU_KEYS = ("util", "mem_used", "mem_total", "temp", "power", "power_limit", "pcie_gen", "pcie_gen_max", "pcie_width",
            "pcie_rx_mb", "pcie_tx_mb")
SERIES = ("tok_s", "prefill_tok_s_mean")      # series kept as history
GPU_SERIES = ("util", "power", "temp")         # per-card series kept as history


def _r2(v):
    return round(v, 2) if isinstance(v, float) else v


class Telemetry:
    def __init__(self, extra=None, gpu_indices=None, gpu_factory=gpu_reader):
        """`extra()` returns the server's own speeds, recorded each second. `gpu_indices`: the engine's cards
        (NVML numbering); `gpu_factory(index)` returns a reader with ok(), read(), name() and optionally
        driver_version(), cuda_version()."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {"gpus": [], "cpu": None, "ram_used": None, "ram_total": None, "disk_read_mb": None}
        self.hist = {k: collections.deque(maxlen=HISTORY) for k in SERIES}
        self.slot_hist = collections.deque(maxlen=HISTORY)     # per sample: each slot's tok/s
        self.gpus = [(i, g) for i, g in ((i, gpu_factory(i)) for i in dict.fromkeys(gpu_indices or [0])) if g.ok()]
        self.gpu_hist = {i: {k: collections.deque(maxlen=HISTORY) for k in GPU_SERIES} for i, _ in self.gpus}
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        first = self.gpus[0][1] if self.gpus else None
        self.static = {
            "gpus": [{"index": i, "name": g.name()} for i, g in self.gpus],
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "driver": first.driver_version() if hasattr(first, "driver_version") else None,
            "cuda": first.cuda_version() if hasattr(first, "cuda_version") else None,
        }
        self._disk_prev = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _disk(self):
        """Disk read MB/s since the last call; None without psutil or on the first call."""
        if not self.ps:
            return None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes)
        if prev is None or t <= prev[0]:
            return None
        return (c.read_bytes - prev[1]) / (t - prev[0]) / 2**20

    def sample(self):
        s = {"gpus": []}
        for i, g in self.gpus:
            try:
                r = g.read()
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                r = {}
            s["gpus"].append({"index": i, **{k: r.get(k) for k in GPU_KEYS}})
        s["cpu"] = s["ram_used"] = s["ram_total"] = None
        if self.ps:
            try:
                s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        s["disk_read_mb"] = self._disk()
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001
                pass
        return s

    def record(self, s: dict) -> None:
        """Make sample `s` the current reading and add it to the history."""
        with self.lock:
            self.now = s
            for k in SERIES:
                self.hist[k].append(_r2(s.get(k)))
            if "slot_tok_s" in s:
                self.slot_hist.append([_r2(v) for v in s.get("slot_tok_s") or []])
            for g in s["gpus"]:
                h = self.gpu_hist.get(g["index"])
                if h:
                    for k in GPU_SERIES:
                        h[k].append(_r2(g.get(k)))

    def _loop(self):
        while True:
            self.record(self.sample())
            time.sleep(1.0)

    def snapshot(self):
        with self.lock:
            now = {k: v for k, v in self.now.items() if k not in SERIES and k != "slot_tok_s"}
            hist = {k: list(v) for k, v in self.hist.items()}
            # one series per slot, oldest first; a sample from before a slot existed is None
            n = max((len(x) for x in self.slot_hist), default=0)
            hist["slot_tok_s"] = [[x[j] if j < len(x) else None for x in self.slot_hist] for j in range(n)]
            hist["gpus"] = [{"index": i, **{k: list(v) for k, v in h.items()}} for i, h in self.gpu_hist.items()]
            return {"now": now, "history": hist, "static": dict(self.static)}
