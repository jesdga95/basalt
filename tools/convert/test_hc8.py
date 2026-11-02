"""Tests for hc8.py: the copies against a scalar reference of the engine's quantizer, and the section's layout.

    python3 -m unittest discover -s tools/convert -p 'test_hc8.py'"""
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hc8 as H  # noqa: E402


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def reference(w: np.ndarray) -> bytes:
    """hc8_quant_kernel one value at a time: fp32 arithmetic (a double quotient of two floats rounds to the fp32
    quotient), f16 by round-to-nearest-even, rint ties to even."""
    rows, cols = w.shape
    codes, scales = bytearray(), bytearray()
    for r in range(rows):
        row_s = []
        for g in range(cols // 32):
            x = [struct.unpack("<f", struct.pack("<I", int(b) << 16))[0] for b in w[r, g * 32:(g + 1) * 32]]
            amax = max(abs(v) for v in x)
            try:
                h = struct.pack("<e", f32(amax / 127.0))
            except OverflowError:
                h = struct.pack("<H", 0x7C00)
            s = struct.unpack("<e", h)[0]
            row_s.append(h)
            for v in x:
                q = min(max(round(f32(v / s)), -127), 127) if s > 0 else 0
                codes += struct.pack("<b", q)
        scales += b"".join(row_s) + b"\0\0" * (H.srow_of(cols) - len(row_s))
    return bytes(codes + scales)


def bf16(values) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


class Quantize(unittest.TestCase):
    def test_random_rows_match_the_reference(self):
        rng = np.random.default_rng(3)
        w = bf16(rng.standard_normal((5, 96)).astype(np.float32) * 0.02)
        self.assertEqual(H.quantize(w), reference(w))

    def test_edges(self):
        g = np.zeros((6, 32), dtype=np.float32)
        g[0, :] = 0.0                                          # an all-zero group: scale 0, codes 0
        g[1, 0], g[1, 1:5] = 127.0, [0.5, 1.5, 2.5, -2.5]      # ties at scale 1: even codes
        g[2, 0], g[2, 1] = 1e-30, -3e-31                       # an absmax whose scale underflows f16
        g[3, 0], g[3, 1] = 3e-6, 1e-6                          # a subnormal f16 scale
        g[4, :] = -1e30                                        # a scale beyond f16: inf, codes 0
        g[5, :] = np.linspace(-3, 7, 32)
        w = bf16(g)
        self.assertEqual(H.quantize(w), reference(w))
        q = np.frombuffer(H.quantize(w)[:6 * 32], dtype=np.int8).reshape(6, 32)
        self.assertEqual(q[1, :5].tolist(), [127, 0, 2, 2, -2])
        self.assertFalse(q[0].any())

    def test_scale_padding(self):
        w = bf16(np.ones((3, 64), dtype=np.float32))
        blob = H.quantize(w)
        self.assertEqual(len(blob), 3 * 64 + 3 * 8 * 2)
        s = np.frombuffer(blob[3 * 64:], dtype="<u2").reshape(3, 8)
        self.assertTrue((s[:, 2:] == 0).all())
        self.assertTrue((s[:, :2] == np.float16(1 / 127).view(np.uint16)).all())


class Section(unittest.TestCase):
    def test_table_layout(self):
        a, b = H.quantize(bf16(np.ones((2, 32)))), H.quantize(bf16(np.full((1, 64), 2.0)))
        blob = H.table([("x", 2, 32, a), ("mtp.y", 1, 64, b)])
        self.assertEqual(blob[:8], b"BASALTH8")
        t = H.read_table(blob)
        self.assertEqual(list(t), ["x", "mtp.y"])
        self.assertEqual(t["x"], (2, 32, 8, a))
        self.assertEqual(t["mtp.y"], (1, 64, 8, b))
        first = H.ENTRY.unpack_from(blob, H.HEADER.size)[5]
        self.assertEqual(first, 512)                         # 16 + 2 x 160, rounded up to 256
        self.assertEqual(H.ENTRY.size, 160)

    def test_section_order_and_sources(self):
        rng = np.random.default_rng(5)
        mats = {"blk.1.hc_attn_down.weight": (2, 64), "blk.0.hc_ffn_inject.weight": (4, 64),
                "output_hc_up.weight": (64, 32), "blk.0.hc_attn_up.weight": (64, 32)}
        pool, lines = b"", []
        for n, (r, c) in mats.items():
            w = bf16(rng.standard_normal((r, c)))
            kind, data = ("1", (w.astype(np.uint32) << 16).astype("<u4").tobytes()) if "inject" in n \
                else ("4", w.astype("<u2").tobytes())
            lines.append(f"{n} 0 {kind} {len(pool)} {len(data)} 0 {r * c * 2} {c} {r} 0 0 1 0 0 0 0 0 0 0")
            pool += data
        lines.append("blk.0.attn_q.weight 0 2 0 4 0 4 1 1 0 0 1 0 0 0 0 0 0 0")
        mw = bf16(rng.standard_normal((2, 32)))
        mtp_pool = b"\0" * 256 + mw.astype("<u2").tobytes()
        mtp_index = ("hyper_connection_mixer.hc_norm.weight f32 1 32 0 128\n"
                     f"attn_hyper_connection.input_mix_weight_down.weight bf16 2 32 256 {mw.nbytes}\n"
                     "mlp.gate.weight bf16 1 32 0 64\n").encode()
        secs = {"dense": pool, "mtp/dense": mtp_pool}
        blob = H.section(lambda s, o, n: secs[s][o:o + n], ("# dense index v3\n" + "\n".join(lines)).encode(), mtp_index)
        t = H.read_table(blob)
        self.assertEqual(list(t), ["blk.0.hc_attn_up.weight", "blk.0.hc_ffn_inject.weight", "blk.1.hc_attn_down.weight",
                                   "output_hc_up.weight", "mtp.attn_hyper_connection.input_mix_weight_down.weight"])
        self.assertEqual(t["mtp.attn_hyper_connection.input_mix_weight_down.weight"][3], H.quantize(mw))
        self.assertEqual(t["output_hc_up.weight"][:2], (64, 32))


if __name__ == "__main__":
    unittest.main()
