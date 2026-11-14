"""Tests for checkpoint_qwen35.py (convert.py --checkpoint DIR, no --sections-from) on a tiny synthetic Model Optimizer
checkpoint: two layers (GDN, then gated attention) at the real GDN head geometry with narrow matrices, FP8 projections,
NVFP4 MLPs and head, a BF16 drafter, a two-block vision tower and tokenizer files; and on its compressed-tensors
variant: every projection NVFP4 (weight_packed, weight_global_scale), the a / b projections too, the head BF16 (FP8 rows
by default, NVFP4 with --head nvfp4).

    python3 tools/convert/test_checkpoint_qwen35.py
    python3 -m unittest discover -s tools/convert -p 'test_checkpoint_qwen35.py'"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import checkpoint as C  # noqa: E402
import checkpoint_qwen35 as Q  # noqa: E402
import model_sections as S  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402
from model_file import ModelFile  # noqa: E402
from test_checkpoint import basalt_values, bf16, modelopt_values, ue4m3, write_safetensors  # noqa: E402

H = 256                    # hidden size (Q8_0 rows and 64-value NVFP4 blocks need multiples of 32 and 64)
FF = 128                   # dense intermediate size
VOCAB = 128                # NVFP4 planes tile rows in 128s (the head)
NHEAD, NKV, HD = 2, 1, 64  # the attention layer
QKV = 2 * S.GDN_K_HEADS * S.GDN_HEAD + S.GDN_V_HEADS * S.GDN_HEAD      # 10240 GDN qkv channels
VALUE = S.GDN_V_HEADS * S.GDN_HEAD                                      # 6144
VH, VDEPTH, VHEADS, VFF, VOUT = 32, 2, 2, 64, H                         # the vision tower
NPOS = 9


def e4m3(c: int) -> float:
    e, m = (c >> 3) & 15, c & 7
    return (-1 if c & 0x80 else 1) * (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


E4M3 = np.array([e4m3(c) for c in range(256)], dtype=np.float64)

TEXT_CONFIG = {
    "model_type": "qwen3_5_text", "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
    "hidden_size": H, "intermediate_size": FF, "num_attention_heads": NHEAD, "num_key_value_heads": NKV, "head_dim": HD,
    "rms_norm_eps": 1e-6, "max_position_embeddings": 4096, "vocab_size": VOCAB, "attn_output_gate": True,
    "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]},
    "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "full_attention_interval": 2, "mtp_num_hidden_layers": 1,
}
VISION_CONFIG = {"model_type": "qwen3_5_vision", "depth": VDEPTH, "hidden_size": VH, "num_heads": VHEADS,
                 "intermediate_size": VFF, "out_hidden_size": VOUT, "patch_size": 16, "temporal_patch_size": 2,
                 "spatial_merge_size": 2, "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": []}
TOKENS = list("abcdefghijkl") + ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>"]
PADDED = TOKENS + [f"[PAD{i}]" for i in range(len(TOKENS), VOCAB)]   # the vocabulary padded to the head's rows


def q8_values(blob: bytes, rows: int, cols: int):
    """A Q8_0 tensor of `rows` rows -> (values, per-value block step d)."""
    b = np.frombuffer(blob, np.uint8).reshape(rows * cols // 32, 34)
    d = np.ascontiguousarray(b[:, :2]).view("<f2").astype(np.float64).reshape(-1)
    q = b[:, 2:].view(np.int8).astype(np.float64)
    return (q * d[:, None]).reshape(rows, cols), np.repeat(d, 32).reshape(rows, cols)


class Fixture:
    """Writes the checkpoint and keeps the values it holds."""

    def __init__(self, d: Path, extra: dict | None = None, ct: bool = False):
        self.ct = ct                            # compressed-tensors: every projection NVFP4, the head BF16
        self.rng = np.random.default_rng(7)
        self.v: dict[str, np.ndarray] = {}      # BF16 / F32 tensors as float32
        self.fp8: dict[str, tuple] = {}         # name -> (codes, scale)
        self.fp4: dict[str, tuple] = {}         # name -> (codes, scales, weight_scale_2)
        self.tens: dict = {}
        r = self.rng
        HD = 128 if ct else globals()["HD"]      # NVFP4 planes need the k / v rows in 128s
        P = "model.language_model."
        L0, L1, M = P + "layers.0.", P + "layers.1.", "mtp.layers.0."
        self.bf(P + "embed_tokens.weight", (VOCAB, H))
        self.bf(P + "norm.weight", (H,))
        if ct:
            self.head(VOCAB, H)
        else:
            self.fp4w("lm_head.weight", VOCAB, H)
        self.bf(L0 + "input_layernorm.weight", (H,))
        self.bf(L0 + "post_attention_layernorm.weight", (H,))
        self.proj(L0 + "linear_attn.in_proj_qkv.weight", (QKV, H))
        self.proj(L0 + "linear_attn.in_proj_z.weight", (VALUE, H))
        self.proj(L0 + "linear_attn.out_proj.weight", (H, VALUE))
        for n in ("in_proj_a", "in_proj_b"):
            if ct:
                self.fp4w(L0 + f"linear_attn.{n}.weight", 48, H)
            else:
                self.bf(L0 + f"linear_attn.{n}.weight", (48, H))
        self.bf(L0 + "linear_attn.A_log", (48,))
        self.bf(L0 + "linear_attn.dt_bias", (48,))
        self.bf(L0 + "linear_attn.conv1d.weight", (QKV, 1, 4))
        self.bf(L0 + "linear_attn.norm.weight", (128,))
        for n, rows, cols in (("gate_proj", FF, H), ("up_proj", FF, H), ("down_proj", H, FF)):
            self.fp4w(L0 + f"mlp.{n}.weight", rows, cols)
        self.bf(L1 + "input_layernorm.weight", (H,))
        self.bf(L1 + "post_attention_layernorm.weight", (H,))
        self.proj(L1 + "self_attn.q_proj.weight", (2 * NHEAD * HD, H))      # query and its output gate
        self.proj(L1 + "self_attn.k_proj.weight", (NKV * HD, H))
        self.proj(L1 + "self_attn.v_proj.weight", (NKV * HD, H))
        self.proj(L1 + "self_attn.o_proj.weight", (H, NHEAD * HD))
        self.bf(L1 + "self_attn.q_norm.weight", (HD,))
        self.bf(L1 + "self_attn.k_norm.weight", (HD,))
        for n, rows, cols in (("gate_proj", FF, H), ("up_proj", FF, H), ("down_proj", H, FF)):
            self.fp4w(L1 + f"mlp.{n}.weight", rows, cols)
        # the drafter, all BF16
        self.bf("mtp.fc.weight", (H, 2 * H))
        for n in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
            self.bf(f"mtp.{n}.weight", (H,))
        self.bf(M + "input_layernorm.weight", (H,))
        self.bf(M + "post_attention_layernorm.weight", (H,))
        for n, shape in (("q_proj", (2 * NHEAD * HD, H)), ("k_proj", (NKV * HD, H)), ("v_proj", (NKV * HD, H)),
                         ("o_proj", (H, NHEAD * HD))):
            self.bf(M + f"self_attn.{n}.weight", shape)
        self.bf(M + "self_attn.q_norm.weight", (HD,))
        self.bf(M + "self_attn.k_norm.weight", (HD,))
        for n, shape in (("gate_proj", (FF, H)), ("up_proj", (FF, H)), ("down_proj", (H, FF))):
            self.bf(M + f"mlp.{n}.weight", shape)
        # the vision tower
        V = "model.visual."
        for l in range(VDEPTH):
            B = V + f"blocks.{l}."
            self.bf(B + "attn.qkv.weight", (3 * VH, VH))
            self.bf(B + "attn.qkv.bias", (3 * VH,))
            self.bf(B + "attn.proj.weight", (VH, VH))
            self.bf(B + "attn.proj.bias", (VH,))
            self.bf(B + "mlp.linear_fc1.weight", (VFF, VH))
            self.bf(B + "mlp.linear_fc1.bias", (VFF,))
            self.bf(B + "mlp.linear_fc2.weight", (VH, VFF))
            self.bf(B + "mlp.linear_fc2.bias", (VH,))
            for n in ("norm1", "norm2"):
                self.bf(B + n + ".weight", (VH,))
                self.bf(B + n + ".bias", (VH,))
        self.bf(V + "merger.norm.weight", (VH,))
        self.bf(V + "merger.norm.bias", (VH,))
        self.bf(V + "merger.linear_fc1.weight", (4 * VH, 4 * VH))
        self.bf(V + "merger.linear_fc1.bias", (4 * VH,))
        self.bf(V + "merger.linear_fc2.weight", (VOUT, 4 * VH))
        self.bf(V + "merger.linear_fc2.bias", (VOUT,))
        self.bf(V + "patch_embed.proj.weight", (VH, 3, 2, 16, 16))
        self.bf(V + "patch_embed.proj.bias", (VH,))
        self.bf(V + "pos_embed.weight", (NPOS, VH))
        for k, v in (extra or {}).items():
            self.tens[k] = v
        write_safetensors(d / "model-00001-of-00001.safetensors", self.tens)
        cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "image_token_id": 11, "video_token_id": 10,
               "vision_start_token_id": 15, "text_config": dict(TEXT_CONFIG, head_dim=HD), "vision_config": VISION_CONFIG}
        (d / "config.json").write_text(json.dumps(cfg))
        (d / "generation_config.json").write_text(json.dumps({"top_k": 20, "top_p": 0.95, "temperature": 0.6}))
        (d / "README.md").write_text("---\nlicense: apache-2.0\nbase_model:\n- Qwen/Qwen3.8-27B\n---\n# card\n")
        (d / "preprocessor_config.json").write_text(json.dumps(
            {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.25, 0.25, 0.25]}))
        vocab = {t: i for i, t in enumerate(TOKENS[:12])}
        (d / "tokenizer.json").write_text(json.dumps({
            "model": {"type": "BPE", "vocab": vocab, "merges": [["a", "b"], ["c", "d"]]},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": C.QWEN35_HF_SPLIT}}, {"type": "ByteLevel"}]},
            "added_tokens": [{"id": 12 + i, "content": t, "special": True} for i, t in enumerate(TOKENS[12:])]}))
        (d / "tokenizer_config.json").write_text(json.dumps(
            {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "add_bos_token": False,
             "chat_template": "{{ messages }}"}))

    def bf(self, name, shape):
        x = (self.rng.integers(-64, 64, size=shape) / 64).astype(np.float32)
        self.v[name] = x
        u = x.view(np.uint32)
        self.tens[name] = ("BF16", shape, (u >> 16).astype("<u2").tobytes())

    def proj(self, name, shape):
        if self.ct:
            self.fp4w(name, *shape)
        else:
            self.f8(name, shape)

    def head(self, rows, cols):
        # BF16 values with a spread of group magnitudes and an outlier per row, so the scale search has work to do
        x = self.rng.standard_normal((rows, cols)) * np.exp(self.rng.uniform(-3, 0, (rows, cols // 16))).repeat(16, 1)
        x[:, 5] *= 8
        u = (x.astype(np.float32).view(np.uint32) + 0x8000) & 0xFFFF0000
        self.v["lm_head.weight"] = u.view(np.float32)
        self.tens["lm_head.weight"] = ("BF16", (rows, cols), (u >> 16).astype("<u2").tobytes())

    def f8(self, name, shape):
        codes = self.rng.integers(0, 127, size=shape, dtype=np.uint8) | (self.rng.integers(0, 2, size=shape, dtype=np.uint8) << 7)
        scale = np.float32(0.0123 + 0.001 * len(self.fp8))
        self.fp8[name] = (codes, scale)
        base = name[:-len(".weight")]
        self.tens[name] = ("F8_E4M3", shape, codes.tobytes())
        self.tens[base + ".weight_scale"] = ("F32", (), scale.tobytes())
        self.tens[base + ".input_scale"] = ("F32", (), np.float32(1).tobytes())

    def fp4w(self, name, rows, cols):
        codes = self.rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
        sc = self.rng.integers(1, 0x7F, size=(rows, cols // 16), dtype=np.uint8)
        g = np.float32(0.25 + 0.125 * len(self.fp4))
        if self.ct:
            g = np.float32(3000 + 7 * len(self.fp4))      # weight_global_scale: the divisor
        self.fp4[name] = (codes, sc, g)
        base = name[:-len(".weight")]
        if self.ct:
            self.tens[base + ".weight_packed"] = ("U8", codes.shape, codes.tobytes())
            self.tens[base + ".weight_scale"] = ("F8_E4M3", sc.shape, sc.tobytes())
            self.tens[base + ".weight_global_scale"] = ("F32", (1,), g.tobytes())
            self.tens[base + ".input_global_scale"] = ("F32", (1,), np.float32(100).tobytes())
            return
        self.tens[name] = ("U8", codes.shape, codes.tobytes())
        self.tens[base + ".weight_scale"] = ("F8_E4M3", sc.shape, sc.tobytes())
        self.tens[base + ".weight_scale_2"] = ("F32", (), g.tobytes())
        self.tens[base + ".input_scale"] = ("F32", (), np.float32(1).tobytes())


class Pack:
    """A converted pack: the `gguf` section's tensors by name, meta, the mmproj and text_vocab GGUFs."""

    def __init__(self, path: Path, tmp: Path):
        self.mf = ModelFile(str(path))
        self.meta = json.loads(self.mf.read("meta"))
        self.gguf = self.mf.read("gguf")
        self.mm = GGUFFile(self._dump("vision/mmproj", tmp / "mmproj.gguf"))
        self.mmraw = (tmp / "mmproj.gguf").read_bytes()
        self.tv = GGUFFile(self._dump("vision/text_vocab", tmp / "text_vocab.gguf"))

    def _dump(self, section, to: Path) -> Path:
        to.write_bytes(self.mf.read(section))
        return to

    def raw(self, name: str) -> bytes:
        t = self.meta["tensors"][name]
        return self.gguf[t["offset"]:t["offset"] + t["bytes"]]

    def f32(self, name: str) -> np.ndarray:
        return np.frombuffer(self.raw(name), "<f4").reshape(list(reversed(self.meta["tensors"][name]["shape"])))

    def bf16(self, name: str) -> np.ndarray:
        u = np.frombuffer(self.raw(name), "<u2").astype(np.uint32) << 16
        return u.view(np.float32).reshape(list(reversed(self.meta["tensors"][name]["shape"])))

    def mm_raw(self, name: str) -> bytes:
        t = next(t for t in self.mm.tensors if t.name == name)
        n = int(np.prod(t.shape)) * {"F32": 4, "BF16": 2}[t.type_name]
        return self.mmraw[self.mm.data_start + t.offset:self.mm.data_start + t.offset + n]


class Qwen35Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        (t / "ck").mkdir()
        cls.fx = Fixture(t / "ck")
        out = t / "x.basalt"
        r = subprocess.run([sys.executable, str(HERE / "convert.py"), "--checkpoint", str(t / "ck"), "--out", str(out),
                            "--workers", "2"], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stdout + r.stderr)
        cls.out = out
        cls.p = Pack(out, t)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.fx, self.p = type(self).fx, type(self).p

    # ---- names, order, types
    def test_tensor_names_order_types(self):
        P = "model.language_model."
        want = ["token_embd.weight", "output_norm.weight", "output.weight", "output.scale",
                "blk.0.attn_norm.weight", "blk.0.post_attention_norm.weight", "blk.0.attn_qkv.weight", "blk.0.attn_qkv.scale",
                "blk.0.attn_gate.weight", "blk.0.attn_gate.scale", "blk.0.ssm_out.weight", "blk.0.ssm_out.scale",
                "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight",
                "blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_conv1d.weight", "blk.0.ssm_norm.weight"]
        for n in ("gate", "up", "down"):
            want += [f"blk.0.ffn_{n}.weight", f"blk.0.ffn_{n}.scale"]
        want += ["blk.1.attn_norm.weight", "blk.1.post_attention_norm.weight", "blk.1.attn_q.weight", "blk.1.attn_q.scale",
                 "blk.1.attn_k.weight", "blk.1.attn_k.scale", "blk.1.attn_v.weight", "blk.1.attn_v.scale",
                 "blk.1.attn_output.weight", "blk.1.attn_output.scale", "blk.1.attn_q_norm.weight", "blk.1.attn_k_norm.weight"]
        for n in ("gate", "up", "down"):
            want += [f"blk.1.ffn_{n}.weight", f"blk.1.ffn_{n}.scale"]
        want += ["mtp.fc.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
                 "mtp.attn_norm.weight", "mtp.post_attention_norm.weight", "mtp.attn_q.weight", "mtp.attn_k.weight",
                 "mtp.attn_v.weight", "mtp.attn_output.weight", "mtp.attn_q_norm.weight", "mtp.attn_k_norm.weight",
                 "mtp.ffn_gate.weight", "mtp.ffn_up.weight", "mtp.ffn_down.weight"]
        # the checkpoint orders the tensors by file offset: embedding and norm, head, layer 0, layer 1, drafter
        self.assertEqual(list(self.p.meta["tensors"]), want)
        T = self.p.meta["tensors"]
        self.assertEqual((T["token_embd.weight"]["type"], T["token_embd.weight"]["shape"]), ("BF16", [H, VOCAB]))
        self.assertEqual((T["output.weight"]["type"], T["output.weight"]["shape"]), ("NVFP4_PLANES", [H, VOCAB]))
        self.assertEqual((T["output.scale"]["type"], T["output.scale"]["shape"], T["output.scale"]["bytes"]), ("F32", [1], 4))
        self.assertEqual((T["blk.0.attn_qkv.weight"]["type"], T["blk.0.attn_qkv.weight"]["shape"]), ("F8_E4M3", [H, QKV]))
        self.assertEqual((T["blk.0.attn_qkv.scale"]["type"], T["blk.0.attn_qkv.scale"]["bytes"]), ("F32", 4))
        self.assertEqual(T["blk.0.ssm_out.weight"]["shape"], [VALUE, H])
        self.assertEqual(T["blk.0.ssm_conv1d.weight"]["shape"], [4, QKV])
        self.assertEqual((T["blk.0.ssm_a"]["type"], T["blk.0.ssm_a"]["shape"]), ("F32", [48]))
        self.assertEqual(T["blk.0.ssm_alpha.weight"]["type"], "BF16")
        self.assertEqual(T["blk.1.attn_q.weight"]["shape"], [H, 2 * NHEAD * HD])
        self.assertEqual((T["blk.0.ffn_down.weight"]["type"], T["blk.0.ffn_down.weight"]["shape"]), ("NVFP4_PLANES", [FF, H]))
        self.assertEqual(T["mtp.ffn_gate.weight"]["type"], "Q8_0")        # the drafter's MLP is BF16 in the checkpoint
        self.assertEqual(T["mtp.fc.weight"]["shape"], [2 * H, H])
        for n, t in T.items():
            self.assertEqual(len(self.p.raw(n)), t["bytes"], n)
            self.assertEqual(t["offset"] % Q.TENSOR_ALIGN, 0, n)
            self.assertEqual(t["type_id"], Q.TYPE_ID[t["type"]], n)

    # ---- NVFP4
    def test_nvfp4_moved_bit_for_bit(self):
        P = "model.language_model."
        for src, name in (("lm_head.weight", "output.weight"), (P + "layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"),
                          (P + "layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
                          (P + "layers.1.mlp.up_proj.weight", "blk.1.ffn_up.weight")):
            codes, sc, g = self.fx.fp4[src]
            got = self.p.raw(name)
            rows, cols = codes.shape[0], codes.shape[1] * 2
            self.assertEqual(len(got), rows * S.nvfp4_row_bytes(cols))
            # the code plane as stored, then each scale at its tile slot (nvfp4_planes.hpp)
            self.assertEqual(got[:rows * cols // 2], codes.tobytes(), name)
            tiles = np.frombuffer(got[rows * cols // 2:], np.uint8)
            r, q = np.meshgrid(np.arange(rows), np.arange(cols // 16), indexing="ij")
            idx = ((r // 128) * (cols // 64) + q // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + q % 4
            np.testing.assert_array_equal(tiles[idx], sc, name)
            scale = self.p.f32(name[:-len(".weight")] + ".scale")
            self.assertEqual(scale.tolist(), [float(g)], name)

    # ---- FP8 kept, BF16 to Q8_0
    def check_f8(self, name, src, rows=None, cols=None):
        # the codes as stored (rows or columns in the GGUF head order) and the tensor scale beside them
        codes, scale = self.fx.fp8[src]
        if rows is not None:
            codes = codes[rows]
        if cols is not None:
            codes = codes[:, cols]
        self.assertEqual(self.p.meta["tensors"][name]["shape"], [codes.shape[1], codes.shape[0]])
        self.assertEqual(self.p.raw(name), np.ascontiguousarray(codes).tobytes(), name)
        self.assertEqual(self.p.f32(name[:-len(".weight")] + ".scale").tolist(), [float(np.float32(scale))], name)

    def check_q8(self, name, want, tol=1.0):
        rows, cols = want.shape
        self.assertEqual(self.p.meta["tensors"][name]["shape"], [cols, rows])
        got, d = q8_values(self.p.raw(name), rows, cols)
        self.assertTrue((np.abs(got - want) <= d * tol + 1e-12).all(), f"{name}: off by more than one Q8_0 step")
        return got

    def fp8_values(self, src):
        codes, scale = self.fx.fp8[src]
        return E4M3[codes] * np.float64(scale)

    def test_fp8_kept(self):
        P = "model.language_model.layers.1."
        for hf, g in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
            self.check_f8(f"blk.1.{g}.weight", P + f"self_attn.{hf}.weight")

    def test_bf16_drafter_to_q8_0(self):
        v = self.fx.v
        self.check_q8("mtp.fc.weight", v["mtp.fc.weight"])
        self.check_q8("mtp.ffn_up.weight", v["mtp.layers.0.mlp.up_proj.weight"])
        self.check_q8("mtp.attn_q.weight", v["mtp.layers.0.self_attn.q_proj.weight"])

    # ---- the GDN value-head reorder
    def test_head_order_formula(self):
        # the GGUF lists value heads as (ratio r, key head k) = r * 16 + k, the checkpoint as k * 3 + r
        g = np.arange(48)
        np.testing.assert_array_equal(S.gdn_block_source(), (g % 16) * 3 + g // 16)
        idx = S.gdn_index(QKV, 4096)
        np.testing.assert_array_equal(idx[:4096], np.arange(4096))
        np.testing.assert_array_equal(idx[4096 + 128 * 17:4096 + 128 * 18], 4096 + 128 * (1 * 3 + 1) + np.arange(128))
        self.assertEqual(sorted(idx), list(range(QKV)))

    def test_gdn_reorder_in_the_pack(self):
        L = "model.language_model.layers.0.linear_attn."
        v = self.fx.v
        qkv = self.fp8_values(L + "in_proj_qkv.weight")[S.gdn_index(QKV, S.GDN_QK_ROWS)]
        self.check_f8("blk.0.attn_qkv.weight", L + "in_proj_qkv.weight", rows=S.gdn_index(QKV, S.GDN_QK_ROWS))
        np.testing.assert_array_equal(qkv[:4096], self.fp8_values(L + "in_proj_qkv.weight")[:4096])     # q and k stay
        z = self.fp8_values(L + "in_proj_z.weight")[S.gdn_index(VALUE, 0)]
        self.check_f8("blk.0.attn_gate.weight", L + "in_proj_z.weight", rows=S.gdn_index(VALUE, 0))
        self.check_f8("blk.0.ssm_out.weight", L + "out_proj.weight", cols=S.gdn_index(VALUE, 0))
        # GGUF value head 17 (r = 1, k = 1) is checkpoint head 4, in z rows and out_proj columns alike
        zraw = self.fp8_values(L + "in_proj_z.weight")
        np.testing.assert_array_equal(z[17 * 128:18 * 128], zraw[4 * 128:5 * 128])
        heads = S.gdn_block_source()
        for hf, g in (("in_proj_a", "ssm_alpha"), ("in_proj_b", "ssm_beta")):
            np.testing.assert_array_equal(self.p.bf16(f"blk.0.{g}.weight"), v[L + hf + ".weight"][heads])
        np.testing.assert_array_equal(self.p.f32("blk.0.ssm_dt.bias"), v[L + "dt_bias"][heads])
        np.testing.assert_array_equal(self.p.f32("blk.0.ssm_a"),
                                      (-np.exp(v[L + "A_log"].astype(np.float64))).astype(np.float32)[heads])
        conv = v[L + "conv1d.weight"][:, 0, :]
        np.testing.assert_array_equal(self.p.f32("blk.0.ssm_conv1d.weight"), conv[S.gdn_index(QKV, S.GDN_QK_ROWS)])

    # ---- norms
    def test_norms_plus_one_and_ssm_norm_as_stored(self):
        v = self.fx.v
        P, L0, L1 = "model.language_model.", "model.language_model.layers.0.", "model.language_model.layers.1."
        for src, name in ((P + "norm.weight", "output_norm.weight"),
                          (L0 + "input_layernorm.weight", "blk.0.attn_norm.weight"),
                          (L0 + "post_attention_layernorm.weight", "blk.0.post_attention_norm.weight"),
                          (L1 + "self_attn.q_norm.weight", "blk.1.attn_q_norm.weight"),
                          (L1 + "self_attn.k_norm.weight", "blk.1.attn_k_norm.weight"),
                          ("mtp.norm.weight", "mtp.norm.weight"),
                          ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_embedding.weight"),
                          ("mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_hidden.weight"),
                          ("mtp.layers.0.input_layernorm.weight", "mtp.attn_norm.weight"),
                          ("mtp.layers.0.self_attn.k_norm.weight", "mtp.attn_k_norm.weight")):
            self.assertEqual(self.p.meta["tensors"][name]["type"], "F32", name)
            np.testing.assert_array_equal(self.p.f32(name), v[src] + np.float32(1), name)
        np.testing.assert_array_equal(self.p.f32("blk.0.ssm_norm.weight"), v[L0 + "linear_attn.norm.weight"])
        np.testing.assert_array_equal(self.p.bf16("token_embd.weight"), v[P + "embed_tokens.weight"])

    # ---- metadata
    def test_gguf_kv(self):
        kv = self.p.meta["gguf_kv"]
        A = "qwen35."
        want = {
            "general.architecture": "qwen35", "general.license": "apache-2.0", "general.file_type": 40,
            "general.base_model.count": 1, "general.base_model.0.name": "Qwen3.8 27B",
            "general.base_model.0.organization": "Qwen", "general.sampling.top_k": 20,
            A + "block_count": 2, A + "context_length": 4096, A + "embedding_length": H, A + "feed_forward_length": FF,
            A + "attention.head_count": NHEAD, A + "attention.head_count_kv": NKV, A + "attention.key_length": HD,
            A + "attention.value_length": HD, A + "rope.dimension_count": 16,
            A + "rope.dimension_sections": [11, 11, 10, 0], A + "rope.freq_base": 10000000.0,
            A + "ssm.conv_kernel": 4, A + "ssm.state_size": 128, A + "ssm.group_count": 16, A + "ssm.time_step_rank": 48,
            A + "ssm.inner_size": VALUE, A + "attention.recurrent_layers": [True, False],
            A + "full_attention_interval": 2, A + "nextn_predict_layers": 1,
            A + "image_token_id": 11, A + "video_token_id": 10, A + "vision_start_token_id": 15,
            "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35", "tokenizer.ggml.eos_token_id": 14,
            "tokenizer.ggml.padding_token_id": 12, "tokenizer.ggml.add_bos_token": False,
        }
        for k, x in want.items():
            self.assertEqual(kv[k], x, k)
        self.assertAlmostEqual(kv[A + "attention.layer_norm_rms_epsilon"], 1e-6, 12)
        self.assertEqual(kv["tokenizer.ggml.tokens"], PADDED)
        self.assertEqual(kv["tokenizer.ggml.merges"], ["a b", "c d"])
        self.assertEqual(kv["tokenizer.ggml.token_type"][:12], [1] * 12)
        self.assertEqual(kv["tokenizer.ggml.token_type"][12:16], [3] * 4)
        self.assertEqual(kv["tokenizer.chat_template"], "{{ messages }}")
        self.assertEqual(self.p.meta["arch"], "qwen35")
        self.assertEqual(self.p.meta["mtp"]["layers"], 1)
        self.assertEqual(self.p.meta["name"], "Qwen3.8-27B NVFP4")
        self.assertEqual(set(self.p.meta["checkpoint"]["transforms"]),
                         {"id", "norm", "heads", "negexp", "gconv", "vrows", "zrows", "vcols"})

    def test_text_vocab_and_sections(self):
        md = self.p.tv.metadata
        self.assertEqual(md["general.architecture"], "qwen35")
        self.assertEqual(md["tokenizer.ggml.tokens"], PADDED)
        self.assertEqual(md["qwen35.block_count"], 2)
        self.assertEqual(self.p.tv.tensors, [])
        names = self.p.mf.names()
        for s in ("gguf", "meta", "vision/mmproj", "vision/text_vocab", "tokenizer/vocab.json", "tokenizer/merges.txt",
                  "tokenizer/tokenizer.json", "tokenizer/chat_template.jinja"):
            self.assertIn(s, names)
            self.assertTrue(self.p.mf.verify(s), s)

    # ---- the vision projector
    def test_mmproj_names_shapes_values(self):
        T = {t.name: t for t in self.p.mm.tensors}
        md = self.p.mm.metadata
        self.assertEqual(md["general.architecture"], "clip")
        self.assertEqual(md["clip.projector_type"], "qwen3vl_merger")
        self.assertEqual(md["clip.vision.block_count"], VDEPTH)
        self.assertEqual(md["clip.vision.embedding_length"], VH)
        self.assertEqual(md["clip.vision.projection_dim"], VOUT)
        self.assertEqual(md["clip.vision.spatial_merge_size"], 2)
        self.assertEqual(md["clip.vision.image_mean"], [0.5, 0.5, 0.5])
        want = {"v.patch_embd.weight": ([16, 16, 3, VH], "F32"), "v.patch_embd.weight.1": ([16, 16, 3, VH], "F32"),
                "v.patch_embd.bias": ([VH], "F32"), "v.position_embd.weight": ([VH, NPOS], "F32"),
                "v.post_ln.weight": ([VH], "F32"), "v.post_ln.bias": ([VH], "F32"),
                "mm.0.weight": ([4 * VH, 4 * VH], "BF16"), "mm.0.bias": ([4 * VH], "F32"),
                "mm.2.weight": ([4 * VH, VOUT], "BF16"), "mm.2.bias": ([VOUT], "F32")}
        for l in range(VDEPTH):
            B = f"v.blk.{l}."
            want.update({B + "attn_qkv.weight": ([VH, 3 * VH], "BF16"), B + "attn_qkv.bias": ([3 * VH], "F32"),
                         B + "attn_out.weight": ([VH, VH], "BF16"), B + "attn_out.bias": ([VH], "F32"),
                         B + "ffn_up.weight": ([VH, VFF], "BF16"), B + "ffn_up.bias": ([VFF], "F32"),
                         B + "ffn_down.weight": ([VFF, VH], "BF16"), B + "ffn_down.bias": ([VH], "F32"),
                         B + "ln1.weight": ([VH], "F32"), B + "ln1.bias": ([VH], "F32"),
                         B + "ln2.weight": ([VH], "F32"), B + "ln2.bias": ([VH], "F32")})
        self.assertEqual(sorted(T), sorted(want))
        for n, (ne, ty) in want.items():
            self.assertEqual((T[n].shape, T[n].type_name), (ne, ty), n)
        v = self.fx.v
        V = "model.visual."
        pw = v[V + "patch_embed.proj.weight"]
        for t, n in ((0, "v.patch_embd.weight"), (1, "v.patch_embd.weight.1")):
            got = np.frombuffer(self.p.mm_raw(n), "<f4").reshape(VH, 3, 16, 16)
            np.testing.assert_array_equal(got, pw[:, :, t])
        np.testing.assert_array_equal(np.frombuffer(self.p.mm_raw("v.position_embd.weight"), "<f4").reshape(NPOS, VH),
                                      v[V + "pos_embed.weight"])
        np.testing.assert_array_equal(np.frombuffer(self.p.mm_raw("v.blk.1.attn_qkv.bias"), "<f4"), v[V + "blocks.1.attn.qkv.bias"])
        u = np.frombuffer(self.p.mm_raw("mm.2.weight"), "<u2").astype(np.uint32) << 16
        np.testing.assert_array_equal(u.view(np.float32).reshape(VOUT, 4 * VH), v[V + "merger.linear_fc2.weight"])
        np.testing.assert_array_equal(np.frombuffer(self.p.mm_raw("v.post_ln.weight"), "<f4"), v[V + "merger.norm.weight"])


def planes(codes: np.ndarray, sc: np.ndarray) -> bytes:
    """NVFP4_PLANES bytes of codes [rows, K / 2] and UE4M3 scales [rows, K / 16] (nvfp4_planes.hpp's formula)."""
    rows, cols = codes.shape[0], codes.shape[1] * 2
    tiles = np.zeros(rows * cols // 16, np.uint8)
    r, q = np.meshgrid(np.arange(rows), np.arange(cols // 16), indexing="ij")
    tiles[((r // 128) * (cols // 64) + q // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + q % 4] = sc
    return codes.tobytes() + tiles.tobytes()


def nvfp4_dequant(codes: np.ndarray, sc: np.ndarray) -> np.ndarray:
    v = np.empty((codes.shape[0], codes.shape[1] * 2), np.float64)
    v[:, 0::2], v[:, 1::2] = Q.E2M1[codes & 15], Q.E2M1[codes >> 4]
    return v * np.repeat(E4M3[sc], 16, axis=1)


class Qwen35CompressedTensorsTest(unittest.TestCase):
    """The compressed-tensors checkpoint: GDN, attention and MLP projections move as planes with 1 / weight_global_scale,
    the a / b projections become BF16 of their values, the BF16 head is rounded to NVFP4 (--head nvfp4)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        (t / "ck").mkdir()
        cls.fx = Fixture(t / "ck", ct=True)
        out = t / "x.basalt"
        r = subprocess.run([sys.executable, str(HERE / "convert.py"), "--checkpoint", str(t / "ck"), "--out", str(out),
                            "--head", "nvfp4", "--workers", "2"], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stdout + r.stderr)
        cls.p = Pack(out, t)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.fx, self.p = type(self).fx, type(self).p

    def check_planes(self, name, src, rows=None, cols=None):
        codes, sc, g = self.fx.fp4[src]
        if rows is not None:
            codes, sc = codes[rows], sc[rows]
        if cols is not None:   # whole 128-value heads move: their code bytes and their group scales
            codes, sc = codes[:, cols[::2] // 2], sc[:, cols[::16] // 16]
        T = self.p.meta["tensors"][name]
        self.assertEqual((T["type"], T["shape"]), ("NVFP4_PLANES", [codes.shape[1] * 2, codes.shape[0]]), name)
        self.assertEqual(self.p.raw(name), planes(np.ascontiguousarray(codes), np.ascontiguousarray(sc)), name)
        self.assertEqual(self.p.f32(name[:-len(".weight")] + ".scale").tolist(), [float(np.float32(1 / np.float64(g)))], name)

    def test_projections_as_planes(self):
        L0, L1 = "model.language_model.layers.0.", "model.language_model.layers.1."
        self.check_planes("blk.0.attn_qkv.weight", L0 + "linear_attn.in_proj_qkv.weight", rows=S.gdn_index(QKV, S.GDN_QK_ROWS))
        self.check_planes("blk.0.attn_gate.weight", L0 + "linear_attn.in_proj_z.weight", rows=S.gdn_index(VALUE, 0))
        self.check_planes("blk.0.ssm_out.weight", L0 + "linear_attn.out_proj.weight", cols=S.gdn_index(VALUE, 0))
        for hf, g in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
            self.check_planes(f"blk.1.{g}.weight", L1 + f"self_attn.{hf}.weight")
        for l in (0, 1):
            for n in ("gate", "up", "down"):
                self.check_planes(f"blk.{l}.ffn_{n}.weight", f"model.language_model.layers.{l}.mlp.{n}_proj.weight")
        self.assertEqual(self.p.meta["tensors"]["mtp.ffn_gate.weight"]["type"], "Q8_0")

    def test_gdn_gates_as_bf16(self):
        L = "model.language_model.layers.0.linear_attn."
        for hf, g in (("in_proj_a", "ssm_alpha"), ("in_proj_b", "ssm_beta")):
            codes, sc, gs = self.fx.fp4[L + hf + ".weight"]
            x = (nvfp4_dequant(codes, sc) / np.float64(gs))[S.gdn_block_source()]
            got = self.p.bf16(f"blk.0.{g}.weight").astype(np.float64)
            self.assertEqual(self.p.meta["tensors"][f"blk.0.{g}.weight"]["type"], "BF16")
            # to nearest: within half a BF16 step (8 significant bits)
            self.assertTrue((np.abs(got - x) <= np.abs(x) * 2.0 ** -8 + 1e-30).all(), g)
            np.testing.assert_array_equal(got, Q.bf16_round(x).astype(np.uint32).__lshift__(16).view(np.float32))

    def test_bf16_rounding(self):
        x = np.array([1 + 2 ** -8, 1 + 3 * 2 ** -8, 1 + 2 ** -8 + 2 ** -30, -(1 + 2 ** -8), 0.0, 3.0])
        u = Q.bf16_round(x).astype(np.uint32) << 16
        np.testing.assert_array_equal(u.view(np.float32), [1.0, 1 + 4 * 2 ** -8, 1 + 2 * 2 ** -8, -1.0, 0.0, 3.0])

    def test_head_rounded_to_nvfp4(self):
        x = self.fx.v["lm_head.weight"]
        T = self.p.meta["tensors"]["output.weight"]
        self.assertEqual((T["type"], T["shape"]), ("NVFP4_PLANES", [H, VOCAB]))
        gs = np.float32(np.abs(x).max()) / np.float32(2688)
        self.assertEqual(self.p.f32("output.scale").tolist(), [float(gs)])
        raw = np.frombuffer(self.p.raw("output.weight"), np.uint8)
        codes = raw[:VOCAB * H // 2].reshape(VOCAB, H // 2)
        r, q = np.meshgrid(np.arange(VOCAB), np.arange(H // 16), indexing="ij")
        sc = raw[VOCAB * H // 2:][((r // 128) * (H // 64) + q // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + q % 4]
        got = nvfp4_dequant(codes, sc) * np.float64(gs)
        # each group's scale is the least-error one of those tried, so no worse than amax / 6 (round to nearest)
        g = x.reshape(VOCAB, H // 16, 16).astype(np.float64)
        s0 = E4M3[Q.ue4m3_round(np.abs(g).max(-1).astype(np.float32) / (np.float32(6) * gs))] * gs
        q0 = Q.E2M1[Q.e2m1_round((g / np.where(s0 > 0, s0, 1)[..., None]).astype(np.float32))] * s0[..., None]
        err = ((got.reshape(g.shape) - g) ** 2).sum(-1)
        err0 = ((q0 - g) ** 2).sum(-1)
        self.assertTrue((err <= err0 * (1 + 1e-6) + 1e-30).all())
        self.assertLess(err.sum(), err0.sum())
        self.assertIn("bf16_to_nvfp4", self.p.meta["checkpoint"])
        self.assertIn("nvfp4_bf16", self.p.meta["checkpoint"])

    def test_e2m1_rounding(self):
        x = np.array([0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 100, -0.75, -0.1], np.float32)
        np.testing.assert_array_equal(Q.e2m1_round(x), [0, 1, 2, 2, 4, 4, 6, 6, 7, 7, 10, 0])


class Qwen35Fp8HeadTest(unittest.TestCase):
    """The compressed-tensors checkpoint by default: the BF16 head becomes E4M3 rows, each with its FP32 scale."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        (t / "ck").mkdir()
        cls.fx = Fixture(t / "ck", ct=True)
        out = t / "x.basalt"
        r = subprocess.run([sys.executable, str(HERE / "convert.py"), "--checkpoint", str(t / "ck"), "--out", str(out),
                            "--workers", "2"], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stdout + r.stderr)
        cls.p = Pack(out, t)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_head_as_fp8_rows(self):
        x = type(self).fx.v["lm_head.weight"]
        p = type(self).p
        T = p.meta["tensors"]["output.weight"]
        self.assertEqual((T["type"], T["type_id"], T["shape"]), ("F8_E4M3", 44, [H, VOCAB]))
        self.assertEqual(p.meta["tensors"]["output.scale"]["shape"], [VOCAB])
        sc = p.f32("output.scale")
        np.testing.assert_array_equal(sc, (np.abs(x).max(-1).astype(np.float32) / np.float32(448)).astype(np.float32))
        codes = np.frombuffer(p.raw("output.weight"), np.uint8).reshape(VOCAB, H)
        self.assertFalse(((codes & 0x7F) == 0x7F).any())
        got = E4M3[codes].astype(np.float64)
        ref = x.astype(np.float64) / sc.astype(np.float64)[:, None]
        # to nearest: no E4M3 value is closer than the one written
        cand = np.sort(np.unique(E4M3[np.isfinite(E4M3)]).astype(np.float64))
        i = np.clip(np.searchsorted(cand, ref), 1, len(cand) - 1)
        best = np.minimum(np.abs(cand[i] - ref), np.abs(cand[i - 1] - ref))
        np.testing.assert_array_equal(np.abs(got - ref), best)
        self.assertEqual(np.abs(got).max(-1).tolist(), [448.0] * VOCAB)   # each row's amax lands on the largest code
        self.assertIn("bf16_to_fp8", p.meta["checkpoint"])
        self.assertEqual(p.meta["tensors"]["blk.0.ffn_gate.weight"]["type"], "NVFP4_PLANES")   # only the head changes

    def test_e4m3_rounding(self):
        x = np.array([0.0, -0.0, 1.0625, 1.1875, -1.0625, 448.0, 500.0, -1000.0, 2.0 ** -10, 3 * 2.0 ** -10])
        np.testing.assert_array_equal(E4M3[Q.e4m3_round(x)], [0, 0, 1.0, 1.25, -1.0, 448, 448, -448, 0, 2.0 ** -8])


class Qwen35RefusalTest(unittest.TestCase):
    """The plan refuses checkpoints it would convert wrongly; Plan reads headers only, so no conversion runs."""

    def plan(self, extra=None, edit=None):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            Fixture(d, extra)
            if edit:
                cfg = json.loads((d / "config.json").read_text())
                edit(cfg)
                (d / "config.json").write_text(json.dumps(cfg))
            return Q.Plan(C.Checkpoint(d))

    def test_the_fixture_plans(self):
        t = self.plan().tensors
        self.assertIn("blk.1.attn_q.weight", t)
        self.assertEqual(t["output.weight"].form, "FP4")   # an NVFP4 head is kept whatever the --head default

    def test_unknown_tensor(self):
        with self.assertRaisesRegex(S.ConvertError, "no GGUF counterpart.*linear_attn.new_thing"):
            self.plan({"model.language_model.layers.0.linear_attn.new_thing.weight": ("BF16", (2,), b"\0" * 4)})

    def test_moe_checkpoint(self):
        with self.assertRaisesRegex(S.ConvertError, "num_experts"):
            self.plan(edit=lambda c: c["text_config"].update(num_experts=8))

    def test_other_gdn_geometry(self):
        with self.assertRaisesRegex(S.ConvertError, "GDN head geometry"):
            self.plan(edit=lambda c: c["text_config"].update(linear_num_value_heads=32))

    def test_wrong_model_type(self):
        with self.assertRaisesRegex(S.ConvertError, "qwen3_5_text"):
            self.plan(edit=lambda c: c["text_config"].update(model_type="qwen3_moe"))

    def test_fp8_head_needs_a_bf16_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            Fixture(d)
            with self.assertRaisesRegex(S.ConvertError, "--head fp8 rounds a BF16 head; lm_head.weight"):
                Q.Plan(C.Checkpoint(d), "fp8")

    def test_missing_layer_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            fx = Fixture(d)
            del fx.tens["model.language_model.layers.1.self_attn.v_proj.weight"]
            del fx.tens["model.language_model.layers.1.self_attn.v_proj.weight_scale"]
            write_safetensors(d / "model-00001-of-00001.safetensors", fx.tens)
            with self.assertRaisesRegex(S.ConvertError, "lacks 1 tensors.*blk.1.attn_v.weight"):
                Q.Plan(C.Checkpoint(d))

    def test_nvfp4_scale_with_the_sign_bit(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            fx = Fixture(d)
            codes, sc, g = fx.fp4["lm_head.weight"]
            sc = sc.copy()
            sc[0, 0] |= 0x80
            data = Q._job(("fp4s", str(d / "model-00001-of-00001.safetensors"),
                           C.Checkpoint(d).t["lm_head.weight_scale"][1], H, VOCAB, 0, VOCAB, None, 0))
            self.assertIsInstance(data, bytes)      # the intact scales convert
            fx.tens["lm_head.weight_scale"] = ("F8_E4M3", sc.shape, sc.tobytes())
            write_safetensors(d / "model-00001-of-00001.safetensors", fx.tens)
            ck = C.Checkpoint(d)
            bad = Q._job(("fp4s", str(d / "model-00001-of-00001.safetensors"), ck.t["lm_head.weight_scale"][1], H, VOCAB, 0,
                          VOCAB, None, 0))
            self.assertIsInstance(bad, str)


if __name__ == "__main__":
    unittest.main()
