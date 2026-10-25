"""Deterministic i-quant fixtures for tests/kernels/iq_parity.cpp.

For each of the ten types iq_parity tests, writes `<out>/<name>.bin` (int32 header: the KERNEL type id,
rows, cols - then the raw GGUF block bytes) and `<out>/<name>.f32` (rows*cols float32 reference values,
dequantized by the same library the comparison treats as ground truth):

    python tools/iq_fixture.py --out build/iq_fixture
    python tools/iq_fixture.py --out build/iq_fixture --seed 7 --rows 32 --cols 256

Deterministic: a seed and fixed dimensions drive every random choice. The i-quants are only dequantized (tools/ggufq.py), so
each block pairs a sane fp16 scale (at the layout's scale offsets) with seeded index and sub-scale bytes.

Needs numpy. Q2_0 (GGML type 42, not in ggufq) is
encoded by tools/gguf_writer.py. The kernel ids in the .bin headers match the repo's i-quant dispatch table and the
ggufq enum ids.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

try:
    import ggufq
    from ggufq import dequantize as gguf_dequantize
    from gguf_writer import dequantize_q2_0, quantize_q2_0
except ImportError as e:  # say so plainly instead of a traceback
    sys.stderr.write(f"iq_fixture: missing dependency: {e}\n"
                     "numpy is required.\n")
    sys.exit(3)

QT = ggufq.GGMLQuantizationType
# name -> (block values, block bytes), mirroring ggufq's dequantize_blocks; scales sit at fixed offsets
# and the rest of a block is seeded random indices and sub-scales
LAYOUT = {   # equals ggufq's GGML_QUANT_SIZES and the kernels' sizes
    "IQ2_XXS": (256,  66),
    "IQ2_XS":  (256,  74),
    "IQ2_S":   (256,  82),
    "IQ3_XXS": (256,  98),
    "IQ3_S":   (256, 110),
    "IQ1_M":   (256,  56),
    "IQ4_NL":   (32,  18),
    "IQ4_XS":  (256, 136),
    "Q3_K":    (256, 110),
}
HALF_ONE = struct.pack("<e", 1.0)      # fp16 1.0 = 0x3c00


def repair_scales(name: str, raw: np.ndarray) -> None:
    """Overwrite every block's fp16 scale with a sane value, in place. IQ*_X* and IQ4_* keep it at offset 0; Q3_K
    after the 32-byte hmask, 64-byte qs and 12 scale bytes; IQ1_M splits it over the high nibbles of the four
    uint16s at bytes 48..56 (1.0 is nibbles 3, C, 0, 0 on bytes 49, 51, 54, 55)."""
    if name == "Q3_K":
        raw[:, 108:110] = np.frombuffer(HALF_ONE, dtype=np.uint8)
        return
    if name == "IQ1_M":
        raw[:, 48:56] &= np.uint8(0x0F)            # clear the four scale nibbles
        for byte, nib in ((49, 0x30), (51, 0xC0), (54, 0x00), (55, 0x00)):
            raw[:, byte] |= np.uint8(nib)
        return
    raw[:, 0:2] = np.frombuffer(HALF_ONE, dtype=np.uint8)
# name -> kernel type id for the .bin header (equal to the ggufq enum id)
KERNEL_IDS = {"IQ2_XXS": 16, "IQ2_XS": 17, "IQ2_S": 22, "IQ3_XXS": 18, "IQ3_S": 21,
              "IQ1_M": 29, "IQ4_NL": 20, "IQ4_XS": 23, "Q2_0": 42, "Q3_K": 11}


def build(name: str, rows: int, cols: int, seed: int):
    """The .bin body (header + raw blocks) and the .f32 reference for one type."""
    kernel_type = KERNEL_IDS[name]
    n = rows * cols
    rng = np.random.RandomState((seed, hash(name) & 0xFFFF) if False else (seed, KERNEL_IDS[name]))
    if name == "Q2_0":                                 # ggufq has no type 42
        w = rng.standard_normal(n).astype(np.float32)
        raw = np.frombuffer(quantize_q2_0(w), dtype=np.uint8)
        ref = dequantize_q2_0(raw.tobytes()).astype(np.float32)
        flat = raw
    else:
        block_values, block_bytes = LAYOUT[name]
        blocks = n // block_values
        raw = rng.randint(0, 256, (blocks, block_bytes), dtype=np.uint8)
        repair_scales(name, raw)                       # sane scales; indices stay random
        flat = raw.reshape(-1)
        ref = np.asarray(gguf_dequantize(flat, getattr(QT, name)), dtype=np.float32)
    if not np.isfinite(ref).all() or float(np.abs(ref).max()) > 1e4:
        raise RuntimeError(f"{name}: the dequantized reference is not finite/tame - the layout table "
                           f"(block bytes, scale offsets) does not match ggufq's dequantizer")
    header = struct.pack("<3i", kernel_type, rows, cols)
    return header + flat.tobytes(), ref


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="logs/iq_fixture", help="the fixture directory (created; may live in the "
                                                             "build directory - the source tree is not written)")
    ap.add_argument("--seed", type=int, default=7, help="the fixture RNG seed (deterministic generation)")
    ap.add_argument("--rows", type=int, default=32, help="weight rows (rows*cols stays a multiple of 256)")
    ap.add_argument("--cols", type=int, default=256, help="the reduction dimension, a multiple of 256")
    args = ap.parse_args()
    if args.cols % 256 or args.rows * args.cols % 256:
        ap.error("--cols must be a multiple of 256 and rows*cols a multiple of 256")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in KERNEL_IDS:
        blob, ref = build(name, args.rows, args.cols, args.seed)
        (out / f"{name}.bin").write_bytes(blob)
        (out / f"{name}.f32").write_bytes(np.ascontiguousarray(ref, dtype="<f4").tobytes())
        print(f"  {name:<8} type {KERNEL_IDS[name]:>2}  {args.rows} x {args.cols}  "
              f"{len(blob) - 12} block bytes, {ref.size} reference floats")
    print(f"iq_fixture: {len(KERNEL_IDS)} fixtures in {out} (seed {args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
