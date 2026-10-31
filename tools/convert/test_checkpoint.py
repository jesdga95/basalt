"""Tests for checkpoint.py on small synthetic safetensors files.

    python3 -m unittest discover -s tools/convert -p 'test_checkpoint.py'"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checkpoint as C  # noqa: E402
import model_sections as S  # noqa: E402

E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def bf16(x: np.ndarray) -> np.ndarray:
    """float32 values that are exactly BF16 -> their u16 codes."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    assert not np.count_nonzero(u & 0xFFFF)
    return (u >> 16).astype("<u2")


def write_safetensors(path: Path, tensors: dict) -> None:
    """name -> (dtype, shape, raw bytes), in the given order."""
    head, blobs, at = {}, [], 0
    for k, (dt, shape, raw) in tensors.items():
        head[k] = {"dtype": dt, "shape": list(shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    hb = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def ue4m3(x) -> float:
    x = int(x)
    e, m = (x >> 3) & 15, x & 7
    return m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)


def modelopt_values(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """What Model Optimizer's NVFP4 rows mean: value 2i is the low nibble of byte i, 2i+1 the high one."""
    rows = codes.shape[0]
    nib = np.empty((rows, codes.shape[1] * 2), np.uint8)
    nib[:, 0::2], nib[:, 1::2] = codes & 15, codes >> 4
    mag = np.array(E2M1, np.float64)[nib & 7] * np.where(nib & 8, -1, 1)
    return mag * np.vectorize(ue4m3)(scales).repeat(16, axis=1)


def basalt_values(blocks: np.ndarray, cols: int) -> np.ndarray:
    """Basalt's NVFP4 block decode (dequantize_nvfp4): sub-block s reads byte 8 s + j, low nibble value j,
    high nibble value j + 8."""
    b = blocks.reshape(-1, cols // 64, 36)
    out = np.empty((b.shape[0], cols // 64, 64))
    for s in range(4):
        sc = np.vectorize(ue4m3)(b[:, :, s])
        q = b[:, :, 4 + 8 * s:12 + 8 * s]
        for half, nib in ((0, q & 15), (8, q >> 4)):
            out[:, :, 16 * s + half:16 * s + half + 8] = (np.array(E2M1)[nib & 7] * np.where(nib & 8, -1, 1)) * sc[..., None]
    return out.reshape(b.shape[0], cols)


def csf_encode(scales: np.ndarray, shift: int = 0):
    """The inverse of `csf_scales`: per slab of 16 rows the 16 base bytes (the row's lowest value, lifted by `shift`
    so that some values fall below their base) and each row's 4-bit offsets, value 2i in the low nibble of byte i;
    a scale outside base..base+15 becomes a (value << 24 | flat position) word."""
    rows, cols = scales.shape
    assert rows % 16 == 0 and cols % 2 == 0
    slabs, exc = [], []
    for r in range(0, rows, 16):
        w = scales[r:r + 16].astype(np.int32)
        base = w.min(axis=1) + shift
        d = w - base[:, None]
        code = np.where((d >= 0) & (d <= 15), d, 0).astype(np.uint8)
        packed = (code[:, 0::2] | (code[:, 1::2] << 4)).reshape(-1)
        slabs.append(np.concatenate((base.astype(np.uint8), packed)))
        for i, j in zip(*np.where((d < 0) | (d > 15))):
            exc.append((int(w[i, j]) << 24) | ((r + i) * cols + int(j)))
    return np.array(slabs, np.uint8), np.array(exc, np.uint32)


def nvfp4_ple_values(row: bytes, scale_2: float) -> np.ndarray:
    """Basalt's 90-byte NVFP4 PLE row (`nvfp4_dequant_row`): per 16 values one UE4M3 scale byte, then 8 code bytes
    with value j in the low nibble of byte j and value j + 8 in its high one - the arrangement of an expert block -
    and the table's F32 scale multiplies everything."""
    b = np.frombuffer(row, np.uint8).reshape(-1, 9)
    sc = np.vectorize(ue4m3)(b[:, 0]) * scale_2
    q, out = b[:, 1:], np.empty((b.shape[0], 16))
    out[:, :8] = (np.array(E2M1)[q & 7] * np.where(q & 8, -1, 1)) * sc[:, None]
    out[:, 8:] = (np.array(E2M1)[(q >> 4) & 7] * np.where(q >> 4 & 8, -1, 1)) * sc[:, None]
    return out.reshape(-1)


class Nvfp4RepackTest(unittest.TestCase):
    def test_values_survive_the_repack(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 256, size=(5, 64), dtype=np.uint8)            # 5 rows of 128 values
        scales = rng.integers(1, 0x7F, size=(5, 8), dtype=np.uint8)
        got = C.nvfp4_rows(codes, scales)
        self.assertEqual(got.size, 5 * S.nvfp4_row_bytes(128))
        np.testing.assert_array_equal(basalt_values(got, 128), modelopt_values(codes, scales))

    def test_expert_blob_and_tail(self):
        rng = np.random.default_rng(1)
        h, ff = 128, 64
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.safetensors"
            tens, want = {}, []
            shapes = {"gate_proj": (ff, h), "up_proj": (ff, h), "down_proj": (h, ff)}
            for i, (r, (rows, cols)) in enumerate(shapes.items()):
                w = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
                s = rng.integers(1, 0x7F, size=(rows, cols // 16), dtype=np.uint8)
                tens[f"{r}.weight"] = ("U8", w.shape, w.tobytes())
                tens[f"{r}.weight_scale"] = ("F8_E4M3", s.shape, s.tobytes())
                tens[f"{r}.weight_scale_2"] = ("F32", (), np.float32(0.5 + i).tobytes())
                want.append(C.nvfp4_rows(w, s).tobytes())
            write_safetensors(p, tens)
            head, base = S.read_safetensors_header(p)
            ent = []
            for r, (rows, cols) in shapes.items():
                ent.append(((p, base + head[f"{r}.weight"]["data_offsets"][0], [rows, cols // 2]),
                            (p, base + head[f"{r}.weight_scale"]["data_offsets"][0], [rows, cols // 16]),
                            None,
                            (p, base + head[f"{r}.weight_scale_2"]["data_offsets"][0])))
            blob = C._expert_job(([ent, ent], 2))
            one = b"".join(want) + np.array([0.5, 1.5, 2.5, 0], "<f4").tobytes()
            self.assertEqual(blob, one + one)
            # a scale byte with the sign bit set is refused
            raw = bytearray(p.read_bytes())
            raw[base + head["up_proj.weight_scale"]["data_offsets"][0]] |= 0x80
            p.write_bytes(bytes(raw))
            self.assertIsInstance(C._expert_job(([ent], 1)), str)

    def test_ple_rows_hold_the_checkpoint_values(self):
        """The pack's 90-byte PLE rows must decode (in Basalt's block order) to exactly what the checkpoint's own
        nibble order means - the converter moves nibbles, the engine never sees the checkpoint's layout."""
        rng = np.random.default_rng(7)
        rows = 33                                       # not a multiple of a chunk or a slab
        codes = rng.integers(0, 256, size=(rows, S.PLE_DIM // 2), dtype=np.uint8)
        scales = rng.integers(1, 0x7F, size=(rows, S.PLE_DIM // 16), dtype=np.uint8)
        scale_2 = 3.324236240587197e-05                 # the QAD checkpoint's own table scale
        with tempfile.TemporaryDirectory() as tmp:
            cp, sp = Path(tmp) / "c.bin", Path(tmp) / "s.bin"
            cp.write_bytes(b"\xa5" * 7 + codes.tobytes())
            sp.write_bytes(b"\xa5" * 3 + scales.tobytes())
            got = S._ple_nvfp4_job((str(cp), 7, str(sp), 3, rows))         # a non-zero offset in each file
            self.assertEqual(len(got), rows * S.PleNvfp4.ROW_BYTES)
            want = modelopt_values(codes, scales) * scale_2
            for r in range(rows):
                row = got[r * S.PleNvfp4.ROW_BYTES:(r + 1) * S.PleNvfp4.ROW_BYTES]
                np.testing.assert_allclose(nvfp4_ple_values(row, scale_2), want[r], rtol=1e-15)
            bad = scales.copy()
            bad[5, 2] |= 0x80                           # a signed or NaN UE4M3 scale is refused, not ignored
            sp.write_bytes(b"\xa5" * 3 + bad.tobytes())
            self.assertIsInstance(S._ple_nvfp4_job((str(cp), 7, str(sp), 3, rows)), str)

    def test_csf_scales_decode_to_the_plain_bytes(self):
        rng = np.random.default_rng(1)
        rows, cols = 32, 8                      # scale bytes of two slabs of 16 rows
        scales = rng.integers(1, 0x7F, size=(rows, cols), dtype=np.uint8)
        for shift in (0, 6):                    # 6 puts values below the base as well as above base + 15
            fixed, exc = csf_encode(scales, shift)
            self.assertEqual(fixed.shape, (rows // 16, 16 * (1 + cols // 2)))
            np.testing.assert_array_equal(C.csf_scales(fixed, exc, rows, cols), scales)
        self.assertTrue(csf_encode(scales, 6)[1].size, "the shifted window should need exceptions")
        # a base that cannot hold its window, or an exception past the matrix, is refused
        fixed, exc = csf_encode(scales, 0)
        self.assertRaises(S.ConvertError, C.csf_scales, np.full_like(fixed, 241), exc, rows, cols)
        self.assertRaises(S.ConvertError, C.csf_scales, fixed, np.array([rows * cols], np.uint32), rows, cols)

    def test_csf_and_plain_scales_build_the_same_blob(self):
        rng = np.random.default_rng(4)
        h, ff = 128, 64
        shapes = {"gate_proj": (ff, h), "up_proj": (ff, h), "down_proj": (h, ff)}
        tens, want = {}, []
        for i, (r, (rows, cols)) in enumerate(shapes.items()):
            w = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
            s = rng.integers(1, 0x7F, size=(rows, cols // 16), dtype=np.uint8)
            fixed, exc = csf_encode(s, 6)
            tens[f"{r}.weight"] = ("U8", w.shape, w.tobytes())
            tens[f"{r}.weight_scale"] = ("F8_E4M3", s.shape, s.tobytes())
            tens[f"{r}.weight_scale.nvfp4_csf_fixed"] = ("U8", fixed.shape, fixed.tobytes())
            tens[f"{r}.weight_scale.nvfp4_csf_exceptions"] = ("U32", exc.shape, exc.astype("<u4").tobytes())
            tens[f"{r}.weight_scale_2"] = ("F32", (), np.float32(0.5 + i).tobytes())
            want.append(C.nvfp4_rows(w, s).tobytes())
        want = b"".join(want) + np.array([0.5, 1.5, 2.5, 0], "<f4").tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.safetensors"
            write_safetensors(p, tens)
            head, base = S.read_safetensors_header(p)

            def entry(r, skey, xkey):
                rows, cols = shapes[r]
                s, x = head[f"{r}.{skey}"], head[f"{r}.{xkey}"] if xkey else None
                return ((p, base + head[f"{r}.weight"]["data_offsets"][0], [rows, cols // 2]),
                        (p, base + s["data_offsets"][0], list(s["shape"])),
                        None if x is None else (p, base + x["data_offsets"][0], x["shape"][0]),
                        (p, base + head[f"{r}.weight_scale_2"]["data_offsets"][0]))

            for keys in ((("weight_scale", None),) * 3,
                         (("weight_scale.nvfp4_csf_fixed", "weight_scale.nvfp4_csf_exceptions"),) * 3,
                         (("weight_scale", None), ("weight_scale.nvfp4_csf_fixed",
                                                   "weight_scale.nvfp4_csf_exceptions"), ("weight_scale", None))):
                entries = [entry(r, sk, xk) for r, (sk, xk) in zip(shapes, keys)]
                self.assertEqual(C._expert_job(([entries], 1)), want)


CSF_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "csf"


def csf_fixture(path: Path):
    """A frozen CSF case: rows, cols, the fixed slabs, the exception words, and the scale bytes a second
    implementation of the codec says those streams mean."""
    rows = cols = 0
    parts = {"fixed": [], "exceptions": [], "expected": []}
    for line in path.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if key in ("rows", "cols"):
            rows, cols = (int(value), cols) if key == "rows" else (rows, int(value))
        else:
            parts[key].append(value)
    fixed = np.frombuffer(bytes.fromhex("".join(parts["fixed"])), np.uint8).reshape(rows // 16, -1)
    exc = np.array([int(w, 16) for w in parts["exceptions"]], np.uint32)
    want = np.frombuffer(bytes.fromhex("".join(parts["expected"])), np.uint8).reshape(rows, cols)
    return rows, cols, fixed, exc, want


class CsfCodecTest(unittest.TestCase):
    """`csf_scales` against frozen streams.  The expected bytes are the independent decoder in Strata's
    tools/nvfp4_local_pack.py, which follows the same published format (vLLM PR 60646, flashinfer's b12x kernel,
    codec byte-window4-fixed-stream-u24-exceptions/1), so the fixture checks the format and not just ourselves.
    The streams are synthetic scale bytes, not a checkpoint's data."""

    def cases(self):
        self.assertTrue(CSF_DIR.is_dir(), f"{CSF_DIR} is missing: the CSF fixtures are a required input")
        cases = sorted(CSF_DIR.glob("*.txt"))
        self.assertEqual(len(cases), 3, f"expected three CSF fixtures in {CSF_DIR}, found {len(cases)}")
        return cases

    def test_frozen_streams_decode_to_the_frozen_bytes(self):
        for path in self.cases():
            rows, cols, fixed, exc, want = csf_fixture(path)
            np.testing.assert_array_equal(C.csf_scales(fixed, exc, rows, cols), want, err_msg=path.name)

    def test_the_exception_stream_is_load_bearing(self):
        rows, cols, fixed, exc, want = csf_fixture(CSF_DIR / "window.txt")
        self.assertTrue(exc.size, "the frozen window case should carry exception words")
        without = C.csf_scales(fixed, np.zeros(0, np.uint32), rows, cols)
        self.assertFalse(np.array_equal(without, want), "an empty exception stream must not give the expected bytes")


GDN = 2 * S.GDN_K_HEADS * S.GDN_HEAD + S.GDN_V_HEADS * S.GDN_HEAD      # q, k, then the value channels


class CheckpointDenseTest(unittest.TestCase):
    """A one-layer checkpoint at the real GDN head geometry with narrow matrices."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        rng = np.random.default_rng(2)
        H = 256
        self.v = {}

        def t(name, shape, scale=1.0):
            x = (rng.integers(-64, 64, size=shape) * scale / 64).astype(np.float32)
            self.v[name] = x
            return ("BF16", shape, bf16(x).tobytes())

        def t32(name, shape, scale=1.0):
            """The same tensor as F32, which is how some checkpoints store the norms the engine reads as F32."""
            x = (rng.integers(-64, 64, size=shape) * scale / 64).astype(np.float32)
            self.v[name] = x
            return ("F32", shape, x.astype("<f4").tobytes())

        L = "model.language_model.layers.0."
        tens = {
            L + "linear_attn.A_log": t(L + "linear_attn.A_log", (48,)),
            L + "linear_attn.conv1d.weight": t(L + "linear_attn.conv1d.weight", (GDN, 1, 4)),
            L + "linear_attn.in_proj_a.weight": t(L + "linear_attn.in_proj_a.weight", (48, H)),
            L + "linear_attn.in_proj_z.weight": t(L + "linear_attn.in_proj_z.weight", (48 * 128, H)),
            L + "linear_attn.norm.weight": t(L + "linear_attn.norm.weight", (128,)),
            L + "linear_attn.out_proj.weight": t(L + "linear_attn.out_proj.weight", (H, 48 * 128)),
            L + "attn_hyper_connection.hc_norm.weight": t32(L + "attn_hyper_connection.hc_norm.weight", (H,)),
            L + "mlp.shared_expert.down_proj.weight": t(L + "mlp.shared_expert.down_proj.weight", (H, 640), 0.0625),
            L + "self_attn.indexer.index_qk_proj.weight": t(L + "self_attn.indexer.index_qk_proj.weight", (640, H)),
            L + "mlp.experts.0.gate_proj.weight": ("U8", (2, 2), b"\0" * 4),
            "mtp.layers.0.mlp.gate.weight": ("BF16", (1,), b"\0\0"),
        }
        write_safetensors(d / "model-00001-of-00001.safetensors", tens)
        cfg = {"text_config": {"linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
                               "linear_value_head_dim": 128, "indexer_n_heads": 4, "indexer_head_dim": 128}}
        (d / "config.json").write_text(json.dumps(cfg))
        self.d, self.L = d, L

    def tearDown(self):
        self.tmp.cleanup()

    def test_names_order_types(self):
        dn = C.CheckpointDense(C.Checkpoint(self.d), q6k=True, require_complete=False)
        self.assertEqual(list(dn.tensors), [
            "blk.0.ssm_a", "blk.0.ssm_conv1d.weight", "blk.0.ssm_alpha.weight", "blk.0.attn_gate.weight",
            "blk.0.ssm_norm.weight", "blk.0.ssm_out.weight", "blk.0.hc_attn_norm.weight",
            "blk.0.ffn_down_shexp.weight", "blk.0.indexer.q_proj.weight", "blk.0.indexer.k_proj.weight"])
        ty = {n: (t.type_name, t.ne) for n, t in dn.tensors.items()}
        self.assertEqual(ty["blk.0.attn_gate.weight"], ("Q6_K", [256, 6144]))
        self.assertEqual(ty["blk.0.ffn_down_shexp.weight"], ("Q8_0", [640, 256]))    # 640-value rows: not Q6_K
        self.assertEqual(ty["blk.0.ssm_conv1d.weight"], ("F32", [4, GDN]))
        self.assertEqual(ty["blk.0.indexer.k_proj.weight"], ("BF16", [256, 128]))
        q8 = C.CheckpointDense(C.Checkpoint(self.d), q6k=False, require_complete=False)
        self.assertEqual(q8.tensors["blk.0.attn_gate.weight"].type_name, "Q8_0")

    def test_transforms(self):
        dn = C.CheckpointDense(C.Checkpoint(self.d), q6k=True, require_complete=False)
        L, v = self.L, self.v
        heads = S.gdn_block_source()
        f32 = lambda n: dn.bytes(n).view("<f4")
        a = v[L + "linear_attn.A_log"]
        np.testing.assert_array_equal(f32("blk.0.ssm_a"), (-np.exp(a.astype(np.float64))).astype(np.float32)[heads])
        np.testing.assert_array_equal(f32("blk.0.hc_attn_norm.weight"), v[L + "attn_hyper_connection.hc_norm.weight"] + 1)
        np.testing.assert_array_equal(f32("blk.0.ssm_norm.weight"), v[L + "linear_attn.norm.weight"])
        conv = v[L + "linear_attn.conv1d.weight"][:, 0, :]
        np.testing.assert_array_equal(f32("blk.0.ssm_conv1d.weight").reshape(GDN, 4), conv[S.gdn_index(GDN, 4096)])
        self.assertEqual(dn.bytes("blk.0.ssm_alpha.weight").tobytes(),
                         bf16(v[L + "linear_attn.in_proj_a.weight"][heads]).tobytes())
        qk = v[L + "self_attn.indexer.index_qk_proj.weight"]
        self.assertEqual(dn.bytes("blk.0.indexer.k_proj.weight").tobytes(), bf16(qk[512:]).tobytes())

    def test_quantized_streams(self):
        import multiprocessing
        dn = C.CheckpointDense(C.Checkpoint(self.d), q6k=True, require_complete=False)
        L, v = self.L, self.v
        with multiprocessing.Pool(2) as pool:
            z = b"".join(dn.chunks("blk.0.attn_gate.weight", pool))
            o = b"".join(dn.chunks("blk.0.ssm_out.weight", pool))
            dw = b"".join(dn.chunks("blk.0.ffn_down_shexp.weight", pool))
        zv = v[L + "linear_attn.in_proj_z.weight"][S.gdn_index(6144, 0)]
        self.assertEqual(z, S.q6_k(zv.reshape(-1)))
        ov = v[L + "linear_attn.out_proj.weight"][:, S.gdn_index(6144, 0)]
        self.assertEqual(o, S.q6_k(np.ascontiguousarray(ov).reshape(-1)))
        self.assertEqual(dw, S.q8_0(v[L + "mlp.shared_expert.down_proj.weight"].reshape(-1)))
        self.assertEqual(len(z), dn.tensors["blk.0.attn_gate.weight"].expected_bytes())

    def test_unknown_tensor_is_refused(self):
        p = self.d / "model-00001-of-00001.safetensors"
        head, base = S.read_safetensors_header(p)
        tens = {k: (h["dtype"], h["shape"], p.read_bytes()[base + h["data_offsets"][0]:base + h["data_offsets"][1]])
                for k, h in head.items()}
        tens[self.L + "linear_attn.new_thing.weight"] = ("BF16", (2,), b"\0" * 4)
        write_safetensors(p, tens)
        with self.assertRaises(S.ConvertError):
            C.CheckpointDense(C.Checkpoint(self.d), require_complete=False)


GDN_CFG = {"text_config": {"linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
                           "linear_value_head_dim": 128, "indexer_n_heads": 4, "indexer_head_dim": 128}}


def small_checkpoint(d: Path, tens: dict, cfg=None) -> None:
    write_safetensors(d / "model-00001-of-00001.safetensors", tens)
    (d / "config.json").write_text(json.dumps(cfg or GDN_CFG))


class F16OverflowTest(unittest.TestCase):
    def test_overflowing_conv_weight_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            x = np.full((2, 1, 4), 2.0 ** 17, np.float32)          # exactly BF16, beyond F16's range
            small_checkpoint(d, {"model.language_model.layers.0.ple.conv1d.weight": ("BF16", x.shape, bf16(x).tobytes())})
            dn = C.CheckpointDense(C.Checkpoint(d), require_complete=False)
            with self.assertRaisesRegex(S.ConvertError, "beyond F16"):
                dn.bytes("blk.0.ple_conv1d.weight")


class Mxfp8DenseTest(unittest.TestCase):
    """A dense projection the checkpoint stores as MXFP8: E4M3 values with one UE8M0 scale per 32 values."""

    NAME = "model.language_model.layers.0.mlp.shared_expert.down_proj.weight"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        rng = np.random.default_rng(6)
        self.v = rng.integers(0, 256, size=(256, 640), dtype=np.uint8)
        self.v[self.v == 0x7F] = 0x7E                   # the two E4M3 NaN codes are not weights
        self.v[self.v == 0xFF] = 0xFE
        self.s = rng.integers(110, 120, size=(256, 20), dtype=np.uint8)
        self.write()
        self.exact = (S.fp8_e4m3_table()[self.v]
                      * np.float32(2.0) ** (self.s.astype(np.int16) - 127).repeat(32, axis=1)).astype(np.float32)

    def write(self):
        small_checkpoint(self.d, {self.NAME: ("F8_E4M3", self.v.shape, self.v.tobytes()),
                                  self.NAME + "_scale": ("U8", self.s.shape, self.s.tobytes())})

    def test_dequantized_exactly_then_quantized_once(self):
        import multiprocessing
        dn = C.CheckpointDense(C.Checkpoint(self.d), require_complete=False)
        t = dn.tensors["blk.0.ffn_down_shexp.weight"]
        self.assertEqual((t.type_name, t.scale is not None), ("Q8_0", True))
        with multiprocessing.Pool(2) as pool:
            got = b"".join(dn.chunks("blk.0.ffn_down_shexp.weight", pool))
        self.assertEqual(got, S.q8_0(self.exact.reshape(-1)))
        self.assertEqual(len(got), t.expected_bytes())

    def test_nan_ue8m0_scale_is_refused(self):
        import multiprocessing
        self.s[7, 3] = 255
        self.write()
        dn = C.CheckpointDense(C.Checkpoint(self.d), require_complete=False)
        with multiprocessing.Pool(1) as pool:
            with self.assertRaisesRegex(S.ConvertError, "UE8M0 scale of 255"):
                b"".join(dn.chunks("blk.0.ffn_down_shexp.weight", pool))

    def test_missing_or_misshapen_scale_is_refused(self):
        small_checkpoint(self.d, {self.NAME: ("F8_E4M3", self.v.shape, self.v.tobytes())})
        with self.assertRaisesRegex(S.ConvertError, "_scale"):
            C.CheckpointDense(C.Checkpoint(self.d), require_complete=False)
        small_checkpoint(self.d, {self.NAME: ("F8_E4M3", self.v.shape, self.v.tobytes()),
                                  self.NAME + "_scale": ("U8", (256, 21), bytes(256 * 21))})
        with self.assertRaisesRegex(S.ConvertError, "_scale"):
            C.CheckpointDense(C.Checkpoint(self.d), require_complete=False)


def expert_tensors(l, e, h=128, ff=64):
    out = {}
    for r, (rows, cols) in {"gate_proj": (ff, h), "up_proj": (ff, h), "down_proj": (h, ff)}.items():
        p = f"model.language_model.layers.{l}.mlp.experts.{e}.{r}."
        out[p + "weight"] = ("U8", (rows, cols // 2), bytes(rows * cols // 2))
        out[p + "weight_scale"] = ("F8_E4M3", (rows, cols // 16), bytes(rows * cols // 16))
        out[p + "weight_scale_2"] = ("F32", (), np.float32(1).tobytes())
        out[p + "input_scale"] = ("F32", (), np.float32(1).tobytes())
    return out


class UnexpectedTensorTest(unittest.TestCase):
    CFG = {"text_config": {"num_hidden_layers": 1, "num_experts": 2, "hidden_size": 128, "moe_intermediate_size": 64}}

    def build(self, extra=None, other=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        tens = {**expert_tensors(0, 0), **expert_tensors(0, 1), **(extra or {})}
        small_checkpoint(d, tens, self.CFG)
        if other:
            write_safetensors(d / "extra.safetensors", other)
        return d

    def test_complete_set_is_accepted(self):
        C.CheckpointExperts(C.Checkpoint(self.build()))

    def test_expert_index_out_of_range(self):
        d = self.build(expert_tensors(0, 2))
        with self.assertRaisesRegex(S.ConvertError, "unexpected expert tensor.*experts\\.2\\."):
            C.CheckpointExperts(C.Checkpoint(d))

    def test_layer_out_of_range(self):
        d = self.build(expert_tensors(1, 0))
        with self.assertRaisesRegex(S.ConvertError, "unexpected expert tensor"):
            C.CheckpointExperts(C.Checkpoint(d))

    def test_extra_expert_tensor(self):
        k = "model.language_model.layers.0.mlp.experts.0.gate_proj.bias"
        d = self.build({k: ("BF16", (2,), bytes(4))})
        with self.assertRaisesRegex(S.ConvertError, "unexpected expert tensor"):
            C.CheckpointExperts(C.Checkpoint(d))

    def test_csf_exceptions_must_be_one_dimensional(self):
        """The word count is read from the tensor's first dimension, so a 2-D exception stream would decode only
        its first row's words - a shorter exception list that still leaves plausible scales."""
        rng = np.random.default_rng(11)
        s = rng.integers(1, 0x7F, size=(64, 8), dtype=np.uint8)          # gate_proj of CFG: ff rows, h / 16 columns
        fixed, exc = csf_encode(s, 6)
        exc = exc[:exc.size - exc.size % 2]
        self.assertTrue(exc.size, "the lifted window should need exceptions")
        p = "model.language_model.layers.0.mlp.experts.0.gate_proj."
        both = {p + "weight_scale.nvfp4_csf_fixed": ("U8", fixed.shape, fixed.tobytes()),
                p + "weight_scale.nvfp4_csf_exceptions": ("U32", (exc.size,), exc.astype("<u4").tobytes())}
        C.CheckpointExperts(C.Checkpoint(self.build(extra=both)))        # the shape a checkpoint really writes
        both[p + "weight_scale.nvfp4_csf_exceptions"] = ("U32", (exc.size // 2, 2), exc.astype("<u4").tobytes())
        with self.assertRaisesRegex(S.ConvertError, "csf_exceptions.*1-D"):
            C.CheckpointExperts(C.Checkpoint(self.build(extra=both)))

    def test_tensor_in_a_non_shard_file(self):
        with self.assertRaisesRegex(S.ConvertError, "extra.safetensors.*dropped"):
            C.Checkpoint(self.build(other={"lm_head.weight": ("BF16", (2,), bytes(4))}))
        ok = {"model.language_model.layers.0.ple.ple_embedding.layer_multipliers": ("I64", (1,), bytes(8))}
        C.Checkpoint(self.build(other=ok))


class CompleteSetTest(unittest.TestCase):
    """A two-layer checkpoint (linear attention, then full attention with the PLE) with every dense tensor."""
    CFG = {"text_config": {**GDN_CFG["text_config"], "indexer_n_heads": 1, "indexer_head_dim": 32,
                           "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"]}}

    @staticmethod
    def tensors():
        L = "model.language_model.layers."
        out = {}
        for l, kind in enumerate(("linear_attention", "full_attention")):
            for key, (_, ty, how) in C.LAYER.items():
                part = key.split(".")[0]
                if part == "linear_attn" and kind != "linear_attention" or part == "self_attn" and kind != "full_attention" \
                        or part == "ple" and l != 1:
                    continue
                shape = (4, 1, 4) if how in ("conv", "gconv") else (4, 64) if ty in ("MAT", "Q8_0") else (64,)
                out[f"{L}{l}.{key}"] = ("BF16", shape, bytes(2 * int(np.prod(shape))))
            if kind == "full_attention":
                out[f"{L}{l}.{C.INDEXER_QK}"] = ("BF16", (64, 64), bytes(2 * 64 * 64))
        for k, (_, ty, how) in C.GLOBAL.items():
            shape = (4, 64) if ty == "Q8_0" else (64,)
            out[k] = ("BF16", shape, bytes(2 * int(np.prod(shape))))
        return out

    def build(self, drop=()):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        small_checkpoint(d, {k: v for k, v in self.tensors().items() if k not in drop}, self.CFG)
        return d

    def test_complete_checkpoint_is_accepted(self):
        dn = C.CheckpointDense(C.Checkpoint(self.build()))
        self.assertIn("output.weight", dn.tensors)
        self.assertIn("blk.0.ssm_a", dn.tensors)
        self.assertIn("blk.1.attn_q.weight", dn.tensors)
        self.assertNotIn("blk.0.attn_q.weight", dn.tensors)

    def test_each_missing_tensor_is_refused(self):
        L = "model.language_model.layers."
        for gone, want in (("lm_head.weight", "output.weight"),
                           ("model.language_model.embed_tokens.weight", "token_embd.weight"),
                           ("model.language_model.hyper_connection_mixer.hc_norm.weight", "output_hc_norm.weight"),
                           (L + "0.linear_attn.in_proj_qkv.weight", "blk.0.attn_qkv.weight"),
                           (L + "1.self_attn.q_norm.weight", "blk.1.attn_q_norm.weight"),
                           (L + "1.self_attn.indexer.index_qk_proj.weight", "blk.1.indexer.k_proj.weight"),
                           (L + "1.ple.key_proj.weight", "blk.1.ple_key.weight"),
                           (L + "0.mlp.gate.weight", "blk.0.ffn_gate_inp.weight")):
            with self.subTest(gone=gone):
                with self.assertRaisesRegex(S.ConvertError, "lacks \\d+ tensors.*" + want.replace(".", "\\.")):
                    C.CheckpointDense(C.Checkpoint(self.build(drop=(gone,))))

    def test_layer_beyond_the_config_is_refused(self):
        d = self.build()
        p = d / "model-00001-of-00001.safetensors"
        t = {k: v for k, v in self.tensors().items()}
        t["model.language_model.layers.2.mlp.gate.weight"] = ("BF16", (64,), bytes(128))
        write_safetensors(p, t)
        with self.assertRaisesRegex(S.ConvertError, "2 layers"):
            C.CheckpointDense(C.Checkpoint(d))


class TruncationTest(unittest.TestCase):
    def test_short_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.safetensors"
            write_safetensors(p, {"a": ("BF16", (4,), b"\1" * 8), "b": ("F32", (2,), b"\2" * 8)})
            self.assertEqual(S.read_safetensors_header(p)[0]["b"]["data_offsets"], [8, 16])
            full = p.read_bytes()
            for cut in (1, 9):
                p.write_bytes(full[:-cut])
                with self.assertRaisesRegex(S.ConvertError, "truncated"):
                    S.read_safetensors_header(p)
            p.write_bytes(full[:20])                       # inside the header
            with self.assertRaisesRegex(S.ConvertError, "truncated"):
                S.read_safetensors_header(p)
            p.write_bytes(full[:5])
            with self.assertRaises(S.ConvertError):
                S.read_safetensors_header(p)


class TokenizerKvTest(unittest.TestCase):
    def test_types_gaps_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            vocab = {"a": 0, "b": 1, "ab": 2}
            tj = {"model": {"type": "BPE", "vocab": vocab, "merges": [["a", "b"]]},
                  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                      {"type": "Split", "pattern": {"Regex": C.QWEN35_HF_SPLIT}}, {"type": "ByteLevel"}]},
                  "added_tokens": [{"id": 4, "content": "<|end|>", "special": True},
                                   {"id": 5, "content": "<|fim|>", "special": False},
                                   {"id": 6, "content": "<think>", "special": False}]}
            (d / "tokenizer.json").write_text(json.dumps(tj))
            (d / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|end|>", "pad_token": "<|end|>",
                                                                 "bos_token": None, "add_bos_token": False}))
            (d / "generation_config.json").write_text(json.dumps({"bos_token_id": 4}))
            (d / "chat_template.jinja").write_text("{{ x }}")
            (d / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 8}}))
            (d / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", 2) + b"{}")
            kv = C.tokenizer_kv(C.Checkpoint(d))
            self.assertEqual(kv["tokenizer.ggml.tokens"], ["a", "b", "ab", "[PAD3]", "<|end|>", "<|fim|>", "<think>", "[PAD7]"])
            self.assertEqual(kv["tokenizer.ggml.token_type"], [1, 1, 1, 5, 3, 3, 4, 5])
            self.assertEqual(kv["tokenizer.ggml.merges"], ["a b"])
            self.assertEqual([kv[f"tokenizer.ggml.{k}_token_id"] for k in ("eos", "padding", "bos")], [4, 4, 4])
            self.assertEqual(kv["tokenizer.chat_template"], "{{ x }}")
            tj["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = r"\s+"
            (d / "tokenizer.json").write_text(json.dumps(tj))
            with self.assertRaises(S.ConvertError):
                C.tokenizer_kv(C.Checkpoint(d))


if __name__ == "__main__":
    unittest.main()
