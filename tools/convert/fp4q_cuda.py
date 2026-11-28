"""checkpoint_qwen35.fp4q_rows on the GPU (fp4q_cuda.cu), bit-identical to it. `rows_bf16` returns None when the library is
not built or no GPU answers, and the caller rounds on the CPU instead. Build: see the header of fp4q_cuda.cu."""
from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_LIB = None
_TRIED = False


def _load():
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    path = pathlib.Path(__file__).resolve().parent / "fp4q_cuda.dll"
    try:
        import checkpoint_qwen35 as Q
        lib = ctypes.CDLL(str(path))
        lib.fp4q_init.argtypes = [ctypes.c_void_p]
        lib.fp4q_rows.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p]
        ue = np.ascontiguousarray(Q.UE4M3, dtype=np.float32)
        if lib.fp4q_init(ue.ctypes.data) == 0:
            _LIB = lib
    except OSError:
        pass
    return _LIB


def rows_bf16(u16: np.ndarray, rows: int, cols: int, gs) -> tuple[np.ndarray, np.ndarray] | None:
    """BF16 bits [rows, cols] -> (E2M1 codes [rows, cols / 2], UE4M3 scales [rows, cols / 16]), or None without the GPU path."""
    lib = _load()
    if lib is None:
        return None
    u16 = np.ascontiguousarray(u16, dtype="<u2")
    codes = np.empty((rows, cols // 2), np.uint8)
    scales = np.empty((rows, cols // 16), np.uint8)
    if lib.fp4q_rows(u16.ctypes.data, rows, cols, float(gs), codes.ctypes.data, scales.ctypes.data) != 0:
        return None
    return codes, scales
