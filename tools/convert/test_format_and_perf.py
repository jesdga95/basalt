"""Unit tests for gguf_writer.py, model_file.py, and bit-exact parity of lossless converter optimizations."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import checkpoint as C  # noqa: E402
import checkpoint_deepseek41 as D  # noqa: E402
import model_sections as S  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import HEADER_BYTES, ModelFile, ModelWriter, append_section, undo_append  # noqa: E402


# ---- verbatim reference implementations prior to lossless perf optimizations
def _ref_q2_0(x: np.ndarray) -> np.ndarray:
    x = x.reshape(-1, 64).astype(np.float32)
    amax = np.abs(x).max(axis=1, keepdims=True)
    best_err = np.full((x.shape[0], 1), np.inf, dtype=np.float32)
    best_d = np.zeros_like(amax)
    for f in np.linspace(0.5, 1.0, 17, dtype=np.float32):
        d = amax * f
        inv = np.where(d > 0, np.float32(1.0) / np.where(d > 0, d, np.float32(1.0)), np.float32(0.0))
        q = np.clip(np.rint(x * inv), np.float32(-1), np.float32(2))
        err = ((q * d - x) ** 2).sum(axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)
    d16 = best_d.astype(np.float16)
    d = d16.astype(np.float32)
    inv = np.where(d > 0, np.float32(1.0) / np.where(d > 0, d, np.float32(1.0)), np.float32(0.0))
    codes = (np.clip(np.rint(x * inv), np.float32(-1), np.float32(2)) + np.float32(1)).astype(np.uint8)
    c = codes.reshape(-1, 16, 4)
    packed = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    out = np.empty((x.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.reshape(-1)


def _ref_q8_0(x: np.ndarray) -> bytes:
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(b).max(axis=1) / np.float32(127.0)
    inv = np.where(d > 0, np.float32(1.0) / np.where(d > 0, d, np.float32(1.0)), np.float32(0.0)).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def _ref_fp8_rows_to_q8_0(raw: np.ndarray, scale: float) -> bytes:
    x = S.fp8_e4m3_table()[raw.reshape(-1)] * np.float32(scale)
    if not np.isfinite(x).all():
        raise S.ConvertError("the FP8 table holds a non-finite value")
    return _ref_q8_0(x)


def _ref_nvfp4_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    rows, cols = codes.shape[0], codes.shape[1] * 2
    v = np.empty((rows, cols), np.uint8)
    v[:, 0::2] = codes & 15
    v[:, 1::2] = codes >> 4
    v = v.reshape(rows, cols // 64, 4, 16)
    out = np.empty((rows, cols // 64, 36), np.uint8)
    out[..., :4] = scales.reshape(rows, cols // 64, 4)
    out[..., 4:] = (v[..., :8] | (v[..., 8:] << 4)).reshape(rows, cols // 64, 32)
    return out.reshape(-1)


def _ref_mxfp4_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    rows = codes.shape[0]
    v = np.empty((rows, D.ENGRAM_ROW), np.uint8)
    v[:, 0::2] = codes & 15
    v[:, 1::2] = codes >> 4
    v = v.reshape(rows, D.ENGRAM_ROW // 32, 2, 16)
    out = np.empty((rows, D.ENGRAM_ROW // 32, 17), np.uint8)
    out[..., 0] = scales
    out[..., 1:] = v[:, :, 0, :] | (v[:, :, 1, :] << 4)
    return out.reshape(-1)


def _ref_csf_decode(fixed: np.ndarray, exceptions: np.ndarray) -> np.ndarray:
    rows, cols = fixed.shape[0] * 16, (fixed.shape[1] // 16 - 1) * 2
    base = fixed[:, :16].reshape(rows, 1).astype(np.int16)
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    out = (base + np.stack((packed & 15, packed >> 4), axis=-1).reshape(rows, cols)).reshape(-1)
    pos = exceptions & 0xFFFFFF
    out[pos] = exceptions >> 24
    return out.astype(np.uint8).reshape(rows, cols)


class GgufWriterUtf8Test(unittest.TestCase):
    def test_utf8_keys_and_tensor_names_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "utf8.gguf"
            w = GGUFWriter()
            w.add("general.architecture", "qwen4exp")
            w.add("custom.clé_日本語", "valeur_αβγ")
            arr = np.arange(8, dtype=np.float32)
            w.add_f32("blk.0.tenseur_λ.weight", arr)
            w.write(p)

            g = GGUFFile(p)
            self.assertEqual(g.metadata["custom.clé_日本語"], "valeur_αβγ")
            self.assertEqual([t.name for t in g.tensors], ["blk.0.tenseur_λ.weight"])
            with p.open("rb") as f:
                f.seek(g.data_start + g.tensors[0].offset)
                got = np.frombuffer(f.read(arr.nbytes), dtype="<f4")
            np.testing.assert_array_equal(got, arr)


class ModelFileAppendUndoTest(unittest.TestCase):
    def _make_pack(self, path: Path, payload: bytes, meta: bytes) -> None:
        w = ModelWriter(str(path))
        w.add_bytes("dense", payload)
        w.add_bytes("meta", meta)
        w.close()

    def test_append_mid_copy_failure_truncates_and_stays_appendable(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            pack = t / "model.basalt"
            self._make_pack(pack, b"A" * 1024, b'{"v":1}')
            before = pack.read_bytes()
            src = t / "extra.bin"
            src.write_bytes(b"X" * 4096)

            real_open = open

            class FailingReader:
                def __init__(self, f):
                    self._f = f
                    self._calls = 0

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    self._f.close()

                def read(self, n=-1):
                    self._calls += 1
                    if self._calls == 1:
                        return self._f.read(512)
                    raise OSError("simulated disk read error")

            def patched_open(p, mode="r", *a, **kw):
                f = real_open(p, mode, *a, **kw)
                if str(p) == str(src) and "rb" in mode:
                    return FailingReader(f)
                return f

            with mock.patch("builtins.open", side_effect=patched_open):
                with self.assertRaises(OSError):
                    append_section(str(pack), "extra", str(src), b'{"v":2}', str(t / "undo1"))

            # File must be truncated back to its original size and remain valid and appendable.
            self.assertEqual(pack.read_bytes(), before)
            append_section(str(pack), "extra", str(src), b'{"v":2}', str(t / "undo2"))
            m = ModelFile(str(pack))
            self.assertTrue(m.verify("extra"))
            self.assertEqual(m.read("extra"), src.read_bytes())

    def test_undo_append_refuses_wrong_pack_and_corrupt_undo_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            pack_a = t / "a.basalt"
            pack_b = t / "b.basalt"
            self._make_pack(pack_a, b"A" * 1024, b'{"id":"a"}')
            self._make_pack(pack_b, b"B" * 1024, b'{"id":"b"}')
            before_a = pack_a.read_bytes()
            before_b = pack_b.read_bytes()
            src = t / "extra.bin"
            src.write_bytes(b"E" * 512)

            undo = t / "undo"
            append_section(str(pack_a), "extra", str(src), b'{"id":"a2"}', str(undo))

            # Wrong target file must be rejected without modifying pack_b.
            with self.assertRaisesRegex(ValueError, "does not match"):
                undo_append(str(pack_b), str(undo))
            self.assertEqual(pack_b.read_bytes(), before_b)

            # Corrupt meta.bin must be rejected without modifying pack_a.
            after_a = pack_a.read_bytes()
            saved_meta = (undo / "meta.bin").read_bytes()
            (undo / "meta.bin").write_bytes(b"X" * len(saved_meta))
            with self.assertRaisesRegex(ValueError, "meta.bin"):
                undo_append(str(pack_a), str(undo))
            self.assertEqual(pack_a.read_bytes(), after_a)
            (undo / "meta.bin").write_bytes(saved_meta)

            # Truncated header.bin must be rejected without modifying pack_a.
            saved_head = (undo / "header.bin").read_bytes()
            (undo / "header.bin").write_bytes(saved_head[:100])
            with self.assertRaises(ValueError):
                undo_append(str(pack_a), str(undo))
            self.assertEqual(pack_a.read_bytes(), after_a)
            (undo / "header.bin").write_bytes(saved_head)

            # Valid rollback restores pack_a bit for bit.
            self.assertEqual(len(saved_head), HEADER_BYTES)
            undo_append(str(pack_a), str(undo))
            self.assertEqual(pack_a.read_bytes(), before_a)


class BitExactPerfParityTest(unittest.TestCase):
    def test_q2_0_bit_exact_across_chunks_and_edges(self):
        rng = np.random.default_rng(42)
        # > 2048 blocks so multi-chunk path runs, plus zeros, negatives, ties, subnormals
        x = rng.standard_normal((2300, 64)).astype(np.float32)
        x[0] = 0.0
        x[1] = -0.0
        x[2] = np.float32(1e-7)
        x[3] = np.linspace(-3.0, 3.0, 64, dtype=np.float32)
        x[2047] = np.array(([0.5, -0.5, 1.5, -1.5] * 16), dtype=np.float32)
        x[2048] = x[2047] * np.float32(-1.0)
        np.testing.assert_array_equal(S.q2_0(x), _ref_q2_0(x))
        np.testing.assert_array_equal(S.q2_0(np.empty(0, dtype=np.float32)), np.empty(0, dtype=np.uint8))

    def test_q8_0_bit_exact_across_chunks_and_edges(self):
        rng = np.random.default_rng(43)
        # > 4096 blocks so multi-chunk path runs
        x = rng.standard_normal((4300, 32)).astype(np.float32)
        x[0] = 0.0
        x[1] = -0.0
        x[2] = np.array(([0.5, -0.5, 1.5, -1.5, 126.5, -126.5, 127.0, -127.0] * 4), dtype=np.float32)
        x[4095] = np.float32(1e-8)
        x[4096] = rng.uniform(-100.0, 100.0, 32).astype(np.float32)
        self.assertEqual(S.q8_0(x), _ref_q8_0(x))
        self.assertEqual(S.q8_0(np.empty(0, dtype=np.float32)), b"")

    def test_fp8_rows_to_q8_0_bit_exact_and_refuses_non_finite(self):
        finite = np.array([c for c in range(256) if (c & 0x7F) != 0x7F], dtype=np.uint8)
        # Repeat to multiple of 32 and PLE_DIM
        codes = np.resize(finite, (64, S.PLE_DIM))
        for scale in (0.0, 1e-4, 0.0125, 1.0, 17.5):
            with self.subTest(scale=scale):
                self.assertEqual(S.fp8_rows_to_q8_0(codes, scale), _ref_fp8_rows_to_q8_0(codes, scale))
        bad = codes.copy()
        bad[3, 5] = 0x7F
        with self.assertRaises(S.ConvertError):
            S.fp8_rows_to_q8_0(bad, 1.0)
        bad[3, 5] = 0xFF
        with self.assertRaises(S.ConvertError):
            S.fp8_rows_to_q8_0(bad, 1.0)
        with self.assertRaises(S.ConvertError):
            S.fp8_rows_to_q8_0(codes, float("inf"))

    def test_sha256_array_matches_tobytes(self):
        arr = np.random.default_rng(44).integers(0, 256, size=100_000, dtype=np.uint8)
        self.assertEqual(S._sha256_array(arr), hashlib.sha256(arr.tobytes()).hexdigest())

    def test_nvfp4_rows_bit_exact(self):
        rng = np.random.default_rng(45)
        codes = rng.integers(0, 256, size=(17, 128), dtype=np.uint8)
        # Cover every single byte value 0..255 deterministically
        codes[0, :] = np.arange(0, 128, dtype=np.uint8)
        codes[1, :] = np.arange(128, 256, dtype=np.uint8)
        scales = rng.integers(0, 0x7F, size=(17, 16), dtype=np.uint8)
        np.testing.assert_array_equal(C.nvfp4_rows(codes, scales), _ref_nvfp4_rows(codes, scales))
        # Also verify non-contiguous strided views (as checkpoint_inkling2 passes c13[e, 0::2]).
        np.testing.assert_array_equal(C.nvfp4_rows(codes[0::2], scales[0::2]), _ref_nvfp4_rows(codes[0::2], scales[0::2]))

    def test_mxfp4_rows_bit_exact(self):
        rng = np.random.default_rng(46)
        codes = rng.integers(0, 256, size=(17, 128), dtype=np.uint8)
        codes[0, :] = np.arange(0, 128, dtype=np.uint8)
        codes[1, :] = np.arange(128, 256, dtype=np.uint8)
        scales = rng.integers(0, 255, size=(17, 8), dtype=np.uint8)
        np.testing.assert_array_equal(D.mxfp4_rows(codes, scales), _ref_mxfp4_rows(codes, scales))
        np.testing.assert_array_equal(D.mxfp4_rows(codes[1::2], scales[1::2]), _ref_mxfp4_rows(codes[1::2], scales[1::2]))

    def test_csf_decode_bit_exact(self):
        rng = np.random.default_rng(47)
        fixed = rng.integers(0, 256, size=(4, 16 + 8 * 32), dtype=np.uint8)
        fixed[:, :16] = rng.integers(0, 241, size=(4, 16), dtype=np.uint8)
        fixed[0, 0] = 240  # max base + max nibble 15 == 255, must not wrap
        fixed[0, 16] = 0xFF
        exc = np.array([0 | (12 << 24), 63 | (255 << 24), (64 * 32 - 1) | (0 << 24)], dtype=np.uint32)
        np.testing.assert_array_equal(C.csf_decode(fixed, exc), _ref_csf_decode(fixed, exc))
        empty_exc = np.empty(0, dtype=np.uint32)
        np.testing.assert_array_equal(C.csf_decode(fixed, empty_exc), _ref_csf_decode(fixed, empty_exc))


if __name__ == "__main__":
    unittest.main()
