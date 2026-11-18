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
                            (p, base + head[f"{r}.weight_scale_2"]["data_offsets"][0])))
            blob = C._expert_job(([ent, ent], 2))
            one = b"".join(want) + np.array([0.5, 1.5, 2.5, 0], "<f4").tobytes()
            self.assertEqual(blob, one + one)
            # a scale byte with the sign bit set is refused
            raw = bytearray(p.read_bytes())
            raw[base + head["up_proj.weight_scale"]["data_offsets"][0]] |= 0x80
            p.write_bytes(bytes(raw))
            self.assertIsInstance(C._expert_job(([ent], 1)), str)


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

        L = "model.language_model.layers.0."
        tens = {
            L + "linear_attn.A_log": t(L + "linear_attn.A_log", (48,)),
            L + "linear_attn.conv1d.weight": t(L + "linear_attn.conv1d.weight", (GDN, 1, 4)),
            L + "linear_attn.in_proj_a.weight": t(L + "linear_attn.in_proj_a.weight", (48, H)),
            L + "linear_attn.in_proj_z.weight": t(L + "linear_attn.in_proj_z.weight", (48 * 128, H)),
            L + "linear_attn.norm.weight": t(L + "linear_attn.norm.weight", (128,)),
            L + "linear_attn.out_proj.weight": t(L + "linear_attn.out_proj.weight", (H, 48 * 128)),
            L + "attn_hyper_connection.hc_norm.weight": t(L + "attn_hyper_connection.hc_norm.weight", (H,)),
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


class QuantizedSourceTest(unittest.TestCase):
    """Dense tensors that the checkpoint stores as MXFP8 or NVFP4 (NVIDIA's quantization-aware distillation export):
    the values decode as the codes say, and the engine forms are quantized once from those values."""

    H = 256

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        rng = np.random.default_rng(7)
        H = self.H
        L = "model.language_model.layers.0."
        self.mx = {}
        self.nv = {}
        tens = {}

        def mxfp8(name, rows, cols):
            codes = rng.integers(0, 0x7F, size=(rows, cols), dtype=np.uint8)
            scales = rng.integers(120, 134, size=(rows, cols // 32), dtype=np.uint8)
            self.mx[name] = (codes, scales)
            tens[name + ".weight"] = ("F8_E4M3", (rows, cols), codes.tobytes())
            tens[name + ".weight_scale"] = ("U8", (rows, cols // 32), scales.tobytes())

        def nvfp4(name, rows, cols, scale2=0.5):
            codes = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
            scales = rng.integers(120, 0x7F, size=(rows, cols // 16), dtype=np.uint8)
            self.nv[name] = (codes, scales, scale2)
            tens[name + ".weight"] = ("U8", (rows, cols // 2), codes.tobytes())
            tens[name + ".weight_scale"] = ("F8_E4M3", (rows, cols // 16), scales.tobytes())
            tens[name + ".weight_scale_2"] = ("F32", (), np.float32(scale2).tobytes())

        mxfp8(L + "linear_attn.in_proj_a", 48, H)          # a BF16 target
        mxfp8(L + "linear_attn.in_proj_z", 48 * S.GDN_HEAD, H)    # a Q6_K target, zrows (real geometry)
        nvfp4(L + "linear_attn.out_proj", H, 48 * S.GDN_HEAD)     # a Q6_K target, vcols
        small_checkpoint(self.d, tens)
        self.L = L

    def tearDown(self):
        self.tmp.cleanup()

    def decode_mx(self, name):
        codes, scales = self.mx[name]
        return S.fp8_e4m3_table()[codes] * np.repeat(np.exp2(scales.astype(np.float32) - np.float32(127)), 32, axis=1)

    def decode_nv(self, name):
        codes, scales, scale2 = self.nv[name]
        v = np.empty((codes.shape[0], codes.shape[1] * 2), np.float32)
        v[:, 0::2] = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)[codes & 15]
        v[:, 1::2] = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)[codes >> 4]
        return v * np.repeat(S.fp8_e4m3_table()[scales] * np.float32(scale2), 16, axis=1)

    def test_mxfp8_decodes_and_rounds_once(self):
        import multiprocessing
        dn = C.CheckpointDense(C.Checkpoint(self.d), q6k=True, require_complete=False)
        L = self.L
        # a BF16 target: E4M3 codes times a power of two are exactly BF16, so the bytes must be exact
        self.assertEqual(dn.bytes("blk.0.ssm_alpha.weight").tobytes(),
                         bf16(self.decode_mx(L + "linear_attn.in_proj_a")[S.gdn_block_source()]).tobytes())
        with multiprocessing.Pool(2) as pool:
            got = b"".join(dn.chunks("blk.0.attn_gate.weight", pool))
        want = self.decode_mx(L + "linear_attn.in_proj_z")[S.gdn_index(48 * S.GDN_HEAD, 0)]
        self.assertEqual(got, S.q6_k(want.reshape(-1)))

    def test_nvfp4_decodes_and_rounds_once(self):
        import multiprocessing
        dn = C.CheckpointDense(C.Checkpoint(self.d), q6k=True, require_complete=False)
        with multiprocessing.Pool(2) as pool:
            got = b"".join(dn.chunks("blk.0.ssm_out.weight", pool))
        want = np.ascontiguousarray(self.decode_nv(self.L + "linear_attn.out_proj")[:, S.gdn_index(48 * S.GDN_HEAD, 0)])
        self.assertEqual(got, S.q6_k(want.reshape(-1)))

    def test_a_wrong_scale_shape_is_refused(self):
        p = self.d / "model-00001-of-00001.safetensors"
        head, base = S.read_safetensors_header(p)
        tens = {k: (h["dtype"], h["shape"], p.read_bytes()[base + h["data_offsets"][0]:base + h["data_offsets"][1]])
                for k, h in head.items()}
        k = self.L + "linear_attn.in_proj_a.weight_scale"
        dt, shape, raw = tens[k]
        tens[k] = (dt, (24, 8), raw)
        write_safetensors(p, tens)
        with self.assertRaisesRegex(S.ConvertError, "weight_scale"):
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


def csf_encode(scales: np.ndarray):
    """The CSF stream pair of scale bytes [rows, cols]: a row's base is the start of its fullest 16-wide window (at
    most 240); a byte outside base..base+15 becomes an exception word."""
    rows, cols = scales.shape
    base = np.zeros(rows, np.int16)
    for r in range(rows):
        v = scales[r].astype(np.int16)
        counts = [int(((v >= b) & (v <= b + 15)).sum()) for b in range(int(v.min()), int(v.max()) + 1)]
        base[r] = min(int(v.min()) + int(np.argmax(counts)), 240)
    code = scales.astype(np.int16) - base[:, None]
    exc = (code < 0) | (code > 15)
    code = np.where(exc, 0, code).astype(np.uint8)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    fixed = np.concatenate([base.reshape(-1, 16).astype(np.uint8), packed.reshape(rows // 16, -1)], axis=1)
    pos = np.flatnonzero(exc)
    return fixed, (pos | scales.reshape(-1)[pos].astype(np.int64) << 24).astype(np.uint32)


class CsfTest(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(3)
        scales = (rng.integers(0, 16, size=(48, 40)) + rng.integers(20, 90, size=(48, 1))).astype(np.uint8)
        for r, c, b in ((5, 7, 0), (30, 0, 126), (47, 39, 1)):         # outside their rows' windows
            scales[r, c] = b
        fixed, exc = csf_encode(scales)
        self.assertEqual((fixed.shape, exc.size), ((3, 16 + 8 * 40), 3))
        np.testing.assert_array_equal(C.csf_decode(fixed, exc), scales)

    def test_by_hand(self):
        fixed = np.zeros((1, 32), np.uint8)                            # 16 rows of 2 scales
        fixed[0, :16] = 40                                             # every row's base
        fixed[0, 16] = 0x21                                            # row 0: codes 1, 2
        fixed[0, 17] = 0xF0                                            # row 1: codes 0, 15
        got = C.csf_decode(fixed, np.array([3 | 7 << 24], np.uint32))  # row 1, column 1 holds 7
        self.assertEqual(got[:2].tolist(), [[41, 42], [40, 7]])
        self.assertTrue((got[2:] == 40).all())

    def test_bad_streams_are_refused(self):
        fixed, exc = csf_encode(np.full((16, 4), 50, np.uint8))
        high = fixed.copy()
        high[0, 3] = 241
        for f, x, want in ((high, exc, "above 240"), (fixed, np.array([16 * 4], np.uint32), "past the end"),
                           (fixed[:, :-8], exc, "shape"), (fixed.reshape(-1), exc, "shape")):
            with self.subTest(want=want), self.assertRaisesRegex(S.ConvertError, want):
                C.csf_decode(f, x)


def csf_expert_tensors(l, e, rng, h=128, ff=64):
    """One expert's tensors twice: with plain scales, and with each scale matrix as a CSF stream pair."""
    plain, csf = {}, {}
    for r, (rows, cols) in {"gate_proj": (ff, h), "up_proj": (ff, h), "down_proj": (h, ff)}.items():
        p = f"model.language_model.layers.{l}.mlp.experts.{e}.{r}."
        w = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
        s = (rng.integers(0, 16, size=(rows, cols // 16)) + 30).astype(np.uint8)
        s[0, 0] = 0x7E                                                 # an exception
        fixed, exc = csf_encode(s)
        for t in (plain, csf):
            t[p + "weight"] = ("U8", w.shape, w.tobytes())
            t[p + "weight_scale_2"] = ("F32", (), np.float32(0.25 + e).tobytes())
            t[p + "input_scale"] = ("F32", (), np.float32(1).tobytes())
        plain[p + "weight_scale"] = ("F8_E4M3", s.shape, s.tobytes())
        csf[p + "weight_scale" + C.CSF_FIXED] = ("U8", fixed.shape, fixed.tobytes())
        csf[p + "weight_scale" + C.CSF_EXC] = ("U32", exc.shape, exc.tobytes())
    return plain, csf


class CsfExpertTest(unittest.TestCase):
    """Experts whose scales are CSF stream pairs, with the shards in a `tensors/` folder."""

    def setUp(self):
        rng = np.random.default_rng(4)
        self.plain, self.csf = {}, {}
        for e in range(2):
            p, c = csf_expert_tensors(0, e, rng)
            self.plain.update(p)
            self.csf.update(c)

    def experts(self, files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        (d / "tensors").mkdir()
        for name, tens in files.items():
            write_safetensors(d / "tensors" / name, tens)
        (d / "config.json").write_text(json.dumps(UnexpectedTensorTest.CFG))
        return C.CheckpointExperts(C.Checkpoint(d))

    def test_same_blob_as_plain_scales(self):
        plain = self.experts({"model-00001-of-00001.safetensors": self.plain})
        exc = {k: v for k, v in self.csf.items() if k.endswith(C.CSF_EXC)}
        rest = {k: v for k, v in self.csf.items() if k not in exc}
        csf = self.experts({"model-00001-of-00002.safetensors": rest, "model-00002-of-00002.safetensors": exc})
        self.assertEqual((plain.csf, csf.csf), (False, True))
        self.assertEqual(C._expert_job(next(csf.jobs(32))), C._expert_job(next(plain.jobs(32))))

    def test_bad_stream_is_reported(self):
        k = "model.language_model.layers.0.mlp.experts.1.down_proj.weight_scale" + C.CSF_FIXED
        dt, shape, raw = self.csf[k]
        ex = self.experts({"model-00001-of-00001.safetensors": {**self.csf, k: (dt, shape, b"\xfa" + raw[1:])}})
        self.assertRegex(C._expert_job(next(ex.jobs(32))), "above 240")

    def test_wrong_shape_or_missing_exceptions_are_refused(self):
        k = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale"
        dt, shape, raw = self.csf[k + C.CSF_FIXED]
        with self.assertRaisesRegex(S.ConvertError, "nvfp4_csf_fixed is U8"):
            self.experts({"model-00001-of-00001.safetensors": {**self.csf, k + C.CSF_FIXED: (dt, (8, 40), raw)}})
        with self.assertRaisesRegex(S.ConvertError, "no .*nvfp4_csf_exceptions"):
            self.experts({"model-00001-of-00001.safetensors": {n: v for n, v in self.csf.items() if n != k + C.CSF_EXC}})


def e4m3(code: int) -> float:
    """An E4M3 byte's value by the format's definition (bias 7, no infinities); the two NaN codes are NaN."""
    sign, e, m = -1.0 if code >> 7 else 1.0, (code >> 3) & 15, code & 7
    if e == 15 and m == 7:
        return float("nan")
    return sign * (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))



def mxfp8(rng, rows, cols):
    """Random MXFP8 rows (no NaN code) and the float32 values they hold, decoded by the format's definition."""
    codes = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x38
    scales = rng.integers(118, 124, size=(rows, cols // 32), dtype=np.uint8)
    table = np.array([e4m3(c) for c in range(256)])
    return codes, scales, (table[codes] * np.exp2(scales - 127.0).repeat(32, axis=1)).astype(np.float32)


class Mxfp8DenseTest(unittest.TestCase):
    """CheckpointDenseTest's layer with MXFP8 projections and an F32 norm, as the FP4-Local layout stores them."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d, self.L, H = Path(tmp.name), "model.language_model.layers.0.", 256
        rng = np.random.default_rng(5)
        self.v, tens = {}, {}
        for name, shape in {"linear_attn.in_proj_a.weight": (48, H), "linear_attn.in_proj_z.weight": (48 * 128, H),
                            "linear_attn.out_proj.weight": (H, 48 * 128), "mlp.shared_expert.down_proj.weight": (H, 640),
                            "self_attn.indexer.index_qk_proj.weight": (640, H)}.items():
            codes, scales, self.v[name] = mxfp8(rng, *shape)
            tens[self.L + name] = ("F8_E4M3", codes.shape, codes.tobytes())
            tens[self.L + name + "_scale"] = ("U8", scales.shape, scales.tobytes())
        self.v["norm"] = (rng.integers(-64, 64, size=(H,)) / 64).astype(np.float32)
        tens[self.L + "attn_hyper_connection.hc_norm.weight"] = ("F32", (H,), self.v["norm"].tobytes())
        self.tens = tens
        small_checkpoint(self.d, tens)

    def dense(self):
        return C.CheckpointDense(C.Checkpoint(self.d), q6k=True, require_complete=False)

    def test_values_widen_exactly(self):
        dn, v = self.dense(), self.v
        self.assertEqual(dn.bytes("blk.0.ssm_alpha.weight").tobytes(),
                         bf16(v["linear_attn.in_proj_a.weight"][S.gdn_block_source()]).tobytes())
        self.assertEqual(dn.bytes("blk.0.indexer.k_proj.weight").tobytes(),
                         bf16(v["self_attn.indexer.index_qk_proj.weight"][512:]).tobytes())
        np.testing.assert_array_equal(dn.bytes("blk.0.hc_attn_norm.weight").view("<f4"), v["norm"] + 1)

    def test_quantized_streams(self):
        import multiprocessing
        dn, v = self.dense(), self.v
        with multiprocessing.Pool(2) as pool:
            z = b"".join(dn.chunks("blk.0.attn_gate.weight", pool))
            o = b"".join(dn.chunks("blk.0.ssm_out.weight", pool))
            dw = b"".join(dn.chunks("blk.0.ffn_down_shexp.weight", pool))
        self.assertEqual(z, S.q6_k(v["linear_attn.in_proj_z.weight"][S.gdn_index(6144, 0)].reshape(-1)))
        o_want = v["linear_attn.out_proj.weight"][:, S.gdn_index(6144, 0)]     # scales apply before the columns move
        self.assertEqual(o, S.q6_k(np.ascontiguousarray(o_want).reshape(-1)))
        self.assertEqual(dw, S.q8_0(v["mlp.shared_expert.down_proj.weight"].reshape(-1)))

    def test_nan_scale_is_refused(self):
        for name in ("linear_attn.in_proj_a.weight_scale", "linear_attn.in_proj_z.weight_scale"):
            dt, shape, raw = self.tens[self.L + name]
            self.tens[self.L + name] = (dt, shape, b"\xff" + raw[1:])
        small_checkpoint(self.d, self.tens)
        dn = self.dense()
        with self.assertRaisesRegex(S.ConvertError, "non-finite"):
            dn.bytes("blk.0.ssm_alpha.weight")
        self.assertIn("a non-finite value", [C._mat_job(j) for j in dn.jobs("blk.0.attn_gate.weight")])

    def test_missing_scale_and_f32_projection_are_refused(self):
        del self.tens[self.L + "linear_attn.out_proj.weight_scale"]
        small_checkpoint(self.d, self.tens)
        with self.assertRaisesRegex(S.ConvertError, "out_proj\\.weight is F8_E4M3 and has no weight_scale"):
            self.dense()
        self.tens[self.L + "linear_attn.out_proj.weight"] = ("F32", (2, 32), bytes(256))
        small_checkpoint(self.d, self.tens)
        with self.assertRaisesRegex(S.ConvertError, "out_proj\\.weight is F32"):
            self.dense()


class PleNvfp4Test(unittest.TestCase):
    """The NVFP4 n-gram table as Q8_0 rows, its two shards in two files."""
    STEM = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = Path(tmp.name)
        rng = np.random.default_rng(6)
        self.codes = [rng.integers(0, 256, size=(r, S.PLE_DIM // 2), dtype=np.uint8) for r in (3, 5)]
        self.scales = [rng.integers(0x20, 0x50, size=(r, S.PLE_DIM // 16), dtype=np.uint8) for r in (3, 5)]
        self.s2 = np.float32(0.0123)
        files = [{}, {}]
        for i in range(2):
            files[i][f"{self.STEM}shard_{i}.weight"] = ("U8", self.codes[i].shape, self.codes[i].tobytes())
            files[i][f"{self.STEM}shard_{i}.weight_scale"] = ("F8_E4M3", self.scales[i].shape, self.scales[i].tobytes())
        files[1][self.STEM + "weight_scale_2"] = ("F32", (1,), self.s2.tobytes())
        for i, t in enumerate(files):
            write_safetensors(self.d / f"model-0000{i + 1}-of-00002.safetensors", t)
        (self.d / "config.json").write_text("{}")

    def test_rows_from_both_files(self):
        ple = C.ple_table(C.Checkpoint(self.d))
        self.assertIsInstance(ple, C.PleNvfp4)
        self.assertEqual((ple.rows, ple.nbytes), (8, 8 * 170))
        self.assertEqual([p.name for p in ple.paths], ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
        want = []
        for codes, scales in zip(self.codes, self.scales):
            nib = np.empty((codes.shape[0], S.PLE_DIM), np.uint8)
            nib[:, 0::2], nib[:, 1::2] = codes & 15, codes >> 4
            mag = np.array(E2M1, np.float32)[nib & 7] * np.where(nib & 8, -1, 1).astype(np.float32)
            sc = np.vectorize(ue4m3)(scales).astype(np.float32)
            want.append(S.q8_0(mag * np.repeat(sc, 16, axis=1) * self.s2))
        ple.chunk_rows = 2
        self.assertEqual(b"".join(C._ple_nvfp4_job(j) for j in ple.jobs()), b"".join(want))

    def test_sign_bit_scale_is_refused(self):
        p = self.d / "model-00002-of-00002.safetensors"
        head, base = S.read_safetensors_header(p)
        raw = bytearray(p.read_bytes())
        raw[base + head[self.STEM + "shard_1.weight_scale"]["data_offsets"][0]] |= 0x80
        p.write_bytes(bytes(raw))
        ple = C.ple_table(C.Checkpoint(self.d))
        with self.assertRaisesRegex(S.ConvertError, "sign bit"):
            for j in ple.jobs():
                C._ple_nvfp4_job(j)

    def test_fp8_table_keeps_its_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            small_checkpoint(d, {self.STEM + "weight_scale": ("BF16", (1,), np.array([0x3C23], "<u2").tobytes()),
                                 self.STEM + "shard_0.weight": ("F8_E4M3", (2, S.PLE_DIM), bytes(2 * S.PLE_DIM))})
            ple = C.ple_table(C.Checkpoint(d))
            self.assertIsInstance(ple, S.PleFp8)
            self.assertEqual(ple.paths, [d / "model-00001-of-00001.safetensors"])
            self.assertEqual(ple.conversion["src_type"], "F8_E4M3")


if __name__ == "__main__":
    unittest.main()

