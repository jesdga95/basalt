"""A minimal GGUF v3 writer including `Q2_0`, which gguf-py cannot write (no type 42).

Enough to build the tiny synthetic model; the only quantizer is Q2_0:
    block = 64 elements, fp16 scale, 16 bytes of 2-bit codes, LSB-first, 4 codes per byte
    d = amax, code = clamp(round(w/d) + 1, 0, 3), symbol = code - 1  ->  {-1, 0, +1, +2}

`python tools/gguf_writer.py` writes a synthetic file, reads it back with `gguf_reader.py` and checks it byte-exactly.
"""
from __future__ import annotations

import dataclasses
import pathlib
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUF_MAGIC, GGUFFile  # noqa: E402

# writable ggml_type ids, matching GGML_TYPES in gguf_reader.py
TYPE_IDS = {"F32": 0, "F16": 1, "Q5_K": 13, "Q6_K": 14, "Q8_0": 8, "Q2_0": 42}
DEFAULT_ALIGNMENT = 32

_META_IDS = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6,
             "bool": 7, "string": 8, "array": 9, "u64": 10, "i64": 11, "f64": 12}


# ------------------------------------------------------------------ Q2_0, per the Q2_0 contract
def quantize_q2_0(w: np.ndarray) -> bytes:
    """(n,) float32 -> concatenated 18-byte blocks. n must be a multiple of 64."""
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if w.size % 64:
        raise ValueError(f"Q2_0 needs a multiple of 64 elements, got {w.size}")
    blocks = w.reshape(-1, 64)
    out = bytearray()
    for b in blocks:
        amax = float(np.max(np.abs(b)))
        d = amax
        # d = amax (not amax/2): the grid is asymmetric, symbols are {-1,0,+1,+2}
        code = np.zeros(64, dtype=np.uint8) if d == 0 else \
            np.clip(np.rint(b / d).astype(np.int32) + 1, 0, 3).astype(np.uint8)
        packed = np.zeros(16, dtype=np.uint8)
        # code j -> byte j//4, bit (j%4)*2
        for j in range(64):
            packed[j // 4] |= np.uint8(code[j] << ((j % 4) * 2))
        out += struct.pack("<e", d)      # fp16 scale
        out += packed.tobytes()
    return bytes(out)


def dequantize_q2_0(raw: bytes) -> np.ndarray:
    """Inverse of the above, for the round-trip check. Mirrors `dequantize_row_q2_0`."""
    nblocks = len(raw) // 18
    out = np.empty(nblocks * 64, dtype=np.float32)
    for i in range(nblocks):
        d = struct.unpack_from("<e", raw, i * 18)[0]
        qs = raw[i * 18 + 2: i * 18 + 18]
        for j in range(64):
            c = (qs[j // 4] >> ((j % 4) * 2)) & 0x03
            out[i * 64 + j] = (c - 1) * d
    return out


# ------------------------------------------------------------------ writer
@dataclasses.dataclass
class _Tensor:
    name: str
    shape: list[int]          # GGUF order: dim 0 varies fastest
    type_name: str
    data: bytes


class GGUFWriter:
    def __init__(self, alignment: int = DEFAULT_ALIGNMENT):
        self.alignment = alignment
        self.metadata: dict[str, tuple[str, object]] = {}
        self.tensors: list[_Tensor] = []

    # ---- metadata
    def add(self, key: str, value, type_name: str | None = None) -> None:
        if type_name is None:
            type_name = {bool: "bool", int: "i64", float: "f64", str: "string"}.get(type(value))
        if type_name is None:
            if isinstance(value, list) and value:
                elem = {bool: "bool", int: "i32", float: "f32", str: "string"}.get(type(value[0]))
                if elem == "i32" and any(v < -(1 << 31) or v > (1 << 31) - 1 for v in value):
                    elem = "i64"        # 64-bit values (the n-gram multipliers are) do not fit an int32 array
                if elem is None:
                    raise TypeError(f"unsupported array element {type(value[0])} for {key}")
                type_name = "array:" + elem
            else:
                raise TypeError(f"cannot infer a GGUF type for {key}")
        self.metadata[key] = (type_name, value)

    # ---- tensors
    def add_f32(self, name: str, arr: np.ndarray) -> None:
        a = np.ascontiguousarray(arr, dtype="<f4")
        self.tensors.append(_Tensor(name, list(a.shape), "F32", a.tobytes()))

    def add_raw(self, name: str, shape: list[int], type_name: str, data: bytes) -> None:
        """A tensor whose blocks are given as they are stored (shape in GGUF order)."""
        self.tensors.append(_Tensor(name, list(shape), type_name, bytes(data)))

    def add_q2_0(self, name: str, arr: np.ndarray, shape: list[int]) -> None:
        """`arr` is (rows, in_features) row-major; `shape` is the full GGUF shape, passed explicitly because a
        fused expert tensor is [in, out, experts] and cannot be inferred. Blocks run along dim 0 only.
        """
        a = np.asarray(arr, dtype=np.float32)
        in_features = shape[0]
        if a.ndim != 2 or a.shape[1] != in_features:
            raise ValueError(f"{name}: expected (rows, {in_features}), got {a.shape}")
        if a.shape[0] * in_features != int(np.prod(shape)):
            raise ValueError(f"{name}: {a.shape} does not cover shape {shape}")
        per_row = quantize_q2_0(a.reshape(-1))          # row-major: row 0 first
        self.tensors.append(_Tensor(name, list(shape), "Q2_0", per_row))

    # ---- serialisation
    def _kv_bytes(self, key: str, type_name: str, value) -> bytes:
        out = bytearray()
        out += struct.pack("<Q", len(key)) + key.encode("utf-8")
        if type_name.startswith("array:"):
            elem = type_name.split(":", 1)[1]
            out += struct.pack("<I", _META_IDS["array"])
            out += struct.pack("<I", _META_IDS[elem])
            out += struct.pack("<Q", len(value))
            for v in value:
                out += self._scalar(elem, v)
            return bytes(out)
        out += struct.pack("<I", _META_IDS[type_name])
        out += self._scalar(type_name, value)
        return bytes(out)

    @staticmethod
    def _scalar(type_name: str, v) -> bytes:
        if type_name == "string":
            b = v.encode("utf-8")
            return struct.pack("<Q", len(b)) + b
        fmt = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I", "i32": "<i",
               "f32": "<f", "bool": "<?", "u64": "<Q", "i64": "<q", "f64": "<d"}[type_name]
        return struct.pack(fmt, v)

    def blob(self) -> bytes:
        """The whole file: header, metadata, tensor directory, then the tensor data aligned to `alignment`."""
        body = bytearray()
        body += struct.pack("<I", GGUF_MAGIC)
        body += struct.pack("<I", 3)                       # GGUF v3
        body += struct.pack("<Q", len(self.tensors))
        body += struct.pack("<Q", len(self.metadata))
        for k, (t, v) in self.metadata.items():
            body += self._kv_bytes(k, t, v)

        # tensor directory: offsets are relative to the data section
        offset = 0
        infos = bytearray()
        for t in self.tensors:
            infos += struct.pack("<Q", len(t.name)) + t.name.encode("utf-8")
            infos += struct.pack("<I", len(t.shape))
            infos += struct.pack(f"<{len(t.shape)}Q", *t.shape)
            infos += struct.pack("<I", TYPE_IDS[t.type_name])
            infos += struct.pack("<Q", offset)
            offset += len(t.data)
            offset = (offset + self.alignment - 1) // self.alignment * self.alignment
        body += infos

        pad = (-len(body)) % self.alignment
        body += b"\0" * pad
        for t in self.tensors:
            body += t.data
            pad = (-len(t.data)) % self.alignment
            body += b"\0" * pad

        return bytes(body)

    def write(self, path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.write_bytes(self.blob())
        return path


# ------------------------------------------------------------------ self-check
def _selfcheck() -> int:
    rng = np.random.default_rng(7)
    w = GGUFWriter()
    w.add("general.architecture", "qwen4exp")
    w.add("qwen4exp.block_count", 8, "u32")
    w.add("qwen4exp.test.array", [1, 2, 3, 4], "array:i32")
    w.add("general.name", "tiny")

    A = rng.standard_normal((16, 64)).astype(np.float32)
    B = rng.standard_normal((4, 8)).astype(np.float32)
    w.add_q2_0("blk.0.ffn_gate_exps.weight", A, [64, 16])
    w.add_f32("blk.0.ssm_a", B)

    p = pathlib.Path(tempfile.mkdtemp(prefix="gguf-selfcheck-")) / "tiny-selftest.gguf"
    w.write(p)
    print(f"wrote {p.name}  ({p.stat().st_size} B)")

    g = GGUFFile(p)
    ok = True
    print(f"  version {g.version}  tensors {len(g.tensors)}  metadata {len(g.metadata)}")
    if g.metadata.get("general.architecture") != "qwen4exp": print("  FAIL arch"); ok = False
    if g.metadata.get("qwen4exp.block_count") != 8:         print("  FAIL u32"); ok = False
    if g.metadata.get("qwen4exp.test.array") != [1, 2, 3, 4]: print("  FAIL array"); ok = False

    by = {t.name: t for t in g.tensors}
    t = by.get("blk.0.ffn_gate_exps.weight")
    if t is None or t.type_name != "Q2_0" or t.shape != [64, 16]:
        print(f"  FAIL tensor dir: {t}"); ok = False
    else:
        with p.open("rb") as fh:
            fh.seek(g.data_start + t.offset)
            raw = fh.read(16 * 18)
        back = dequantize_q2_0(raw)
        err = np.abs(back - A.reshape(-1)).max()
        rel = float(np.abs(back - A.reshape(-1)).mean() / np.abs(A).mean())
        print(f"  Q2_0 round trip: max abs {err:.4f}  mean rel {rel*100:.2f}%  "
              f"(quantization error, not a write bug)")
        # re-quantizing the dequantized values must be a fixed point
        again = dequantize_q2_0(quantize_q2_0(back))
        if not np.array_equal(again, back):
            print("  FAIL Q2_0 is not a fixed point under requantization"); ok = False
        else:
            print("  Q2_0 fixed point: PASS (requantizing the dequantized values is exact)")
    print("SELF-CHECK " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
