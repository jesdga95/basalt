"""Tests for tools/iq_fixture.py: IQ1_M scale packing, sub-scale preservation, and fixture generation."""
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import iq_fixture as iqf  # noqa: E402


class IqFixtureTest(unittest.TestCase):
    def test_iq1_m_scale_is_one_and_preserves_subscales(self):
        rng = np.random.RandomState(42)
        raw = rng.randint(0, 256, (16, 56), dtype=np.uint8)
        before = raw.copy()
        iqf.repair_scales("IQ1_M", raw)
        # Bytes 0..47, even scale bytes (48, 50, 52, 54), and low nibbles of odd bytes (49, 51, 53, 55) are untouched.
        np.testing.assert_array_equal(raw[:, :48], before[:, :48])
        np.testing.assert_array_equal(raw[:, 48:56:2], before[:, 48:56:2])
        np.testing.assert_array_equal(raw[:, 49:56:2] & np.uint8(0x0F), before[:, 49:56:2] & np.uint8(0x0F))
        # Reconstruct the 16-bit scale the exact way both ggufq.py and iq_kernels.cu do.
        sc = raw[:, 48:56].copy().view(np.uint16)
        d_bits = (sc[:, 0] >> 12) | ((sc[:, 1] >> 8) & 0x00F0) | ((sc[:, 2] >> 4) & 0x0F00) | (sc[:, 3] & 0xF000)
        np.testing.assert_array_equal(d_bits, np.full(16, 0x3C00, dtype=np.uint16))
        d_f32 = d_bits.astype(np.uint16).view(np.float16).astype(np.float32)
        np.testing.assert_array_equal(d_f32, np.ones(16, dtype=np.float32))

    def test_build_all_types_finite_and_iq1_m_nontrivial(self):
        for name, type_id in iqf.KERNEL_IDS.items():
            with self.subTest(name=name):
                blob, ref = iqf.build(name, rows=4, cols=256, seed=7)
                got_type, got_rows, got_cols = struct.unpack_from("<3i", blob, 0)
                self.assertEqual((got_type, got_rows, got_cols), (type_id, 4, 256))
                self.assertEqual(ref.shape, (4 * 256,))
                self.assertTrue(np.isfinite(ref).all())
                if name == "IQ1_M":
                    # With fp16 scale 1.0 (0x3C00) and intact 3-bit sub-scales (2*s+1 in 1..15), magnitudes are O(1..15).
                    self.assertGreater(float(np.abs(ref).max()), 1.0)
                    raw = np.frombuffer(blob[12:], dtype=np.uint8).reshape(-1, 56)
                    sc = raw[:, 48:56].copy().view(np.uint16)
                    d_bits = (sc[:, 0] >> 12) | ((sc[:, 1] >> 8) & 0x00F0) | ((sc[:, 2] >> 4) & 0x0F00) | (sc[:, 3] & 0xF000)
                    np.testing.assert_array_equal(d_bits, np.full(raw.shape[0], 0x3C00, dtype=np.uint16))

    def test_cli_generates_all_fixtures(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, str(TOOLS / "iq_fixture.py"), "--out", d, "--rows", "4", "--cols", "256"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            for name in iqf.KERNEL_IDS:
                self.assertTrue((Path(d) / f"{name}.bin").is_file(), name)
                self.assertEqual((Path(d) / f"{name}.f32").stat().st_size, 4 * 256 * 4, name)


if __name__ == "__main__":
    unittest.main()
