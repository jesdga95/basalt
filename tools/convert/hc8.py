"""The `hc8` section: the 8-bit copies of the hyper-connection matrices the engine's decode reads.

The layout is in include/basalt/core/model_file.hpp (Hc8Header, Hc8Entry).  Each copy is the matrix's int8 codes, then
its f16 scales: q = rint(w / s) clamped to +-127, s = f16(absmax / 127) per 32 consecutive values of a row, in fp32
arithmetic, as the engine's quantizer (hc8_quant_kernel) computes them.

    blob = hc8.section(read, dense_index_text, mtp_index_text)   # read(section, offset, size) -> bytes
"""
from __future__ import annotations

import re
import struct

import numpy as np

G8 = 32
HEADER = struct.Struct("<8sII")                 # magic, version, n_entries
ENTRY = struct.Struct("<128sIIIIQQ")             # name, rows, cols, srow, reserved, offset, bytes
MAGIC = b"BASALTH8"
VERSION = 1
DATA_ALIGN = 256
# the drafter's matrices: bf16 tensors of its hyper-connection blocks
MTP_HC = re.compile(r".*hyper_connection.*")


class Hc8Error(Exception):
    pass


def srow_of(cols: int) -> int:
    return ((cols // G8) + 7) & ~7


def quantize(w: np.ndarray) -> bytes:
    """One copy of the bf16 matrix `w` (uint16, rows x cols): the codes, then rows x srow f16 scales (zero padded)."""
    rows, cols = w.shape
    if cols % G8:
        raise Hc8Error(f"{cols} columns are not whole groups of {G8}")
    x = (w.astype(np.uint32) << 16).view(np.float32).reshape(rows, cols // G8, G8)
    amax = np.abs(x).max(axis=2)
    with np.errstate(over="ignore"):
        h = (amax / np.float32(127.0)).astype(np.float16)
    s = h.astype(np.float32)[:, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.clip(np.rint(x / s), np.float32(-127.0), np.float32(127.0))
    q = np.where(s > 0, v, np.float32(0.0)).astype(np.int8)
    scales = np.zeros((rows, srow_of(cols)), dtype=np.uint16)
    scales[:, : cols // G8] = h.view(np.uint16)
    return q.tobytes() + scales.astype("<u2").tobytes()


def table(copies: list[tuple[str, int, int, bytes]]) -> bytes:
    """The section from (name, rows, cols, copy) in order."""
    at = (HEADER.size + len(copies) * ENTRY.size + DATA_ALIGN - 1) // DATA_ALIGN * DATA_ALIGN
    entries, data = [], []
    for name, rows, cols, blob in copies:
        nb = name.encode()
        if len(nb) >= 128:
            raise Hc8Error(f"name too long: {name}")
        srow = srow_of(cols)
        if len(blob) != rows * cols + rows * srow * 2:
            raise Hc8Error(f"wrong size for {name}")
        entries.append(ENTRY.pack(nb, rows, cols, srow, 0, at, len(blob)))
        data.append((at, blob))
        at = (at + len(blob) + DATA_ALIGN - 1) // DATA_ALIGN * DATA_ALIGN
    out = bytearray(HEADER.pack(MAGIC, VERSION, len(copies)) + b"".join(entries))
    for off, blob in data:
        out += b"\0" * (off - len(out))
        out += blob
    return bytes(out)


def read_table(blob: bytes) -> dict[str, tuple[int, int, int, bytes]]:
    """name -> (rows, cols, srow, copy)."""
    magic, version, n = HEADER.unpack_from(blob, 0)
    if magic != MAGIC or version != VERSION:
        raise Hc8Error("bad magic or version")
    out = {}
    for i in range(n):
        name, rows, cols, srow, _, off, nb = ENTRY.unpack_from(blob, HEADER.size + i * ENTRY.size)
        out[name.rstrip(b"\0").decode()] = (rows, cols, srow, blob[off:off + nb])
    return out


def _dense_rows(index_text: str) -> dict[str, list[str]]:
    rows = {}
    for line in index_text.splitlines():
        if line and not line.startswith("#"):
            f = line.split()
            rows[f[0]] = f
    return rows


def _main_names(rows: dict) -> list[str]:
    """The model's hyper-connection matrices in the engine's order: per layer attn then ffn (down, up, inject), then the
    output head's down and up."""
    layers = sorted({int(m.group(1)) for n in rows for m in [re.match(r"blk\.(\d+)\.hc_", n)] if m})
    names = [f"blk.{l}.hc_{p}_{w}.weight" for l in layers for p in ("attn", "ffn") for w in ("down", "up", "inject")]
    names += ["output_hc_down.weight", "output_hc_up.weight"]
    return [n for n in names if n in rows]


def _bf16(raw: bytes, kind: int, rows: int, cols: int, name: str) -> np.ndarray:
    if kind == 4:                                 # bf16 as stored
        a = np.frombuffer(raw, dtype="<u2")
    elif kind == 1:                               # bf16 promoted to f32: the high 16 bits
        a = (np.frombuffer(raw, dtype="<u4") >> 16).astype(np.uint16)
    else:
        raise Hc8Error(f"{name} is dense index kind {kind}, not bf16")
    if a.size != rows * cols:
        raise Hc8Error(f"{name}: {a.size} values, not {rows} x {cols}")
    return a.reshape(rows, cols)


def section(read, dense_index: bytes, mtp_index: bytes | None) -> bytes:
    """The `hc8` section of a pack: `read(section, offset, size)` reads the pack's `dense` and `mtp/dense` bytes."""
    copies = []
    rows = _dense_rows(dense_index.decode("utf-8"))
    for name in _main_names(rows):
        f = rows[name]
        kind, off, nb, ne0, ne1 = int(f[2]), int(f[3]), int(f[4]), int(f[7]), int(f[8])
        if f[1] != "0":
            raise Hc8Error(f"{name} is not in the dense section")
        w = _bf16(read("dense", off, nb), kind, ne1, ne0, name)
        copies.append((name, ne1, ne0, quantize(w)))
    for line in (mtp_index or b"").decode("utf-8").splitlines():
        f = line.split()
        if len(f) != 6 or f[1] != "bf16" or not MTP_HC.match(f[0]):
            continue
        r, c, off, nb = int(f[2]), int(f[3]), int(f[4]), int(f[5])
        if r * c * 2 != nb:
            continue
        w = _bf16(read("mtp/dense", off, nb), 4, r, c, f[0])
        copies.append(("mtp." + f[0], r, c, quantize(w)))
    return table(copies)
