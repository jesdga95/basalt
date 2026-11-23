"""Tests for checkpoint_nemotronh.py (convert.py --checkpoint DIR on a Nemotron-H checkpoint) on a tiny synthetic
checkpoint shaped as NVIDIA's VL NVFP4 export: three layers (Mamba-2, attention, latent MoE) under
`language_model.backbone.`, FP8 projections and head with a scale each, NVFP4 routed and shared experts, BF16 everything
else, an MTP head and vision tensors the converter leaves out, and tokenizer files.

    python3 tools/convert/test_checkpoint_nemotronh.py
    python3 -m unittest discover -s tools/convert -p 'test_checkpoint_nemotronh.py'"""
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
import checkpoint_nemotronh as N  # noqa: E402
import model_sections as S  # noqa: E402
from model_file import ModelFile  # noqa: E402
from test_checkpoint import basalt_values, modelopt_values, write_safetensors  # noqa: E402

H, NH, NKV, HD = 256, 2, 1, 128
MH, MHD, DS, NG, DC = 4, 64, 16, 2, 4          # d_inner 256, conv_dim 320
LAT, FF, FFS, NE, TOPK = 128, 64, 128, 4, 2
KINDS = ["mamba", "attention", "moe"]
TOKENS = list("abcdefghijklmnop") + ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
VOCAB = len(TOKENS)
E4M3 = S.fp8_e4m3_table()
DI, CD = MH * MHD, MH * MHD + 2 * NG * DS


def llm_config(**over):
    c = {"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h", "num_hidden_layers": len(KINDS),
         "layers_block_type": KINDS, "hidden_size": H, "vocab_size": VOCAB, "num_attention_heads": NH,
         "num_key_value_heads": NKV, "head_dim": HD, "mamba_num_heads": MH, "mamba_head_dim": MHD,
         "ssm_state_size": DS, "n_groups": NG, "conv_kernel": DC, "n_routed_experts": NE, "num_experts_per_tok": TOPK,
         "moe_intermediate_size": FF, "moe_shared_expert_intermediate_size": FFS, "moe_latent_size": LAT,
         "mamba_hidden_act": "silu", "mlp_hidden_act": "relu2", "use_conv_bias": True, "mamba_proj_bias": False,
         "use_bias": False, "attention_bias": False, "mlp_bias": False, "n_group": 1, "topk_group": 1,
         "norm_topk_prob": True, "tie_word_embeddings": False, "residual_in_fp32": False, "n_shared_experts": 1,
         "sliding_window": None, "routed_scaling_factor": 5.0, "layer_norm_epsilon": 1e-5,
         "max_position_embeddings": 1048576}
    c.update(over)
    return c


class Fixture:
    def __init__(self, d: Path):
        self.rng = np.random.default_rng(7)
        self.v, self.tens, self.f8, self.nv = {}, {}, {}, {}
        B = "language_model.backbone."
        self.bf(B + "embeddings.weight", (VOCAB, H))
        self.bf(B + "norm_f.weight", (H,))
        self.fp8("language_model.lm_head.weight", (VOCAB, H))
        for l, kind in enumerate(KINDS):
            p = f"{B}layers.{l}."
            self.bf(p + "norm.weight", (H,))
            m = p + "mixer."
            if kind == "mamba":
                self.fp8(m + "in_proj.weight", (2 * DI + 2 * NG * DS + MH, H))
                self.bf(m + "conv1d.weight", (CD, 1, DC))
                self.bf(m + "conv1d.bias", (CD,))
                self.bf(m + "dt_bias", (MH,))
                self.bf(m + "A_log", (MH,))
                self.bf(m + "D", (MH,))
                self.bf(m + "norm.weight", (DI,))
                self.fp8(m + "out_proj.weight", (H, DI))
            elif kind == "attention":
                for n, rows in (("q_proj", NH * HD), ("k_proj", NKV * HD), ("v_proj", NKV * HD)):
                    self.fp8(m + n + ".weight", (rows, H))
                self.fp8(m + "o_proj.weight", (H, NH * HD))
                self.tens[m + "k_scale"] = ("F32", (), np.array(1.5, np.float32).tobytes())
            else:
                self.bf(m + "gate.weight", (NE, H))
                self.f32(m + "gate.e_score_correction_bias", (NE,))
                self.bf(m + "fc1_latent_proj.weight", (LAT, H))
                self.bf(m + "fc2_latent_proj.weight", (H, LAT))
                self.nvfp4(m + "shared_experts.up_proj", FFS, H)
                self.nvfp4(m + "shared_experts.down_proj", H, FFS)
                for e in range(NE):
                    self.nvfp4(m + f"experts.{e}.up_proj", FF, LAT)
                    self.nvfp4(m + f"experts.{e}.down_proj", LAT, FF)
        # left out: the MTP head and the vision tower
        self.bf("language_model.mtp.layers.0.eh_proj.weight", (H, 2 * H))
        self.bf("vision_model.radio_model.model.blocks.0.norm1.weight", (32,))
        self.bf("mlp1.0.weight", (64,))
        names = list(self.tens)
        half = len(names) // 2
        write_safetensors(d / "model-00001-of-00002.safetensors", {k: self.tens[k] for k in names[:half]})
        write_safetensors(d / "model-00002-of-00002.safetensors", {k: self.tens[k] for k in names[half:]})
        llm = llm_config()
        (d / "config.json").write_text(json.dumps({"architectures": ["NemotronH_Omni_Reasoning_V3"], "model_type": "nemotron_h_omni",
                                                   "llm_config": llm}))
        (d / "generation_config.json").write_text(json.dumps({"temperature": 1.0, "top_p": 0.95, "eos_token_id": [2, 20]}))
        (d / "README.md").write_text("---\nlicense: other\nlicense_name: openmdw-1.1\n---\n# card\n")
        (d / "chat_template.jinja").write_text("{{ messages }}")
        vocab = {t: i for i, t in enumerate(TOKENS[:16])}
        (d / "tokenizer.json").write_text(json.dumps({
            "model": {"type": "BPE", "vocab": vocab, "merges": [["a", "b"], ["c", "d"]]},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": N.NEMOTRONH_HF_SPLIT}}, {"type": "ByteLevel"}]},
            "added_tokens": [{"id": 16 + i, "content": t, "special": True} for i, t in enumerate(TOKENS[16:])]}))
        (d / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>", "eos_token": "<|im_end|>", "pad_token": "<|im_end|>"}))

    def bf(self, name, shape):
        x = (self.rng.integers(-64, 64, size=shape) / 64).astype(np.float32)
        self.v[name] = x
        self.tens[name] = ("BF16", shape, (x.view(np.uint32) >> 16).astype("<u2").tobytes())

    def f32(self, name, shape):
        x = self.rng.standard_normal(shape).astype(np.float32)
        self.v[name] = x
        self.tens[name] = ("F32", shape, x.astype("<f4").tobytes())

    def fp8(self, name, shape):
        codes = self.rng.integers(0, 0x7F, size=shape, dtype=np.uint8) | (self.rng.integers(0, 2, size=shape, dtype=np.uint8) << 7)
        s = np.float32(self.rng.uniform(0.001, 0.01))
        self.f8[name] = (codes, s)
        self.tens[name] = ("F8_E4M3", shape, codes.tobytes())
        base = name[:-len(".weight")]
        self.tens[base + ".weight_scale"] = ("F32", (), np.array(s, np.float32).tobytes())
        self.tens[base + ".input_scale"] = ("F32", (1,), np.array([0.5], np.float32).tobytes())

    def nvfp4(self, base, rows, cols):
        codes = self.rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
        scales = self.rng.integers(32, 120, size=(rows, cols // 16), dtype=np.uint8)
        s2 = np.float32(self.rng.uniform(0.001, 0.01))
        self.nv[base] = (codes, scales, s2)
        self.tens[base + ".weight"] = ("U8", (rows, cols // 2), codes.tobytes())
        self.tens[base + ".weight_scale"] = ("F8_E4M3", (rows, cols // 16), scales.tobytes())
        self.tens[base + ".weight_scale_2"] = ("F32", (), np.array(s2, np.float32).tobytes())
        self.tens[base + ".input_scale"] = ("F32", (), np.array(0.25, np.float32).tobytes())


class ConvertTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name) / "ckpt"
        d.mkdir()
        cls.fx = Fixture(d)
        cls.out = Path(cls.tmp.name) / "tiny.basalt"
        r = subprocess.run([sys.executable, str(HERE / "convert.py"), "--checkpoint", str(d), "--out", str(cls.out), "--workers", "2"],
                           capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(r.stdout + r.stderr)
        cls.mf = ModelFile(str(cls.out))
        cls.meta = json.loads(cls.mf.read("meta"))
        cls.gguf = cls.mf.read("gguf")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tensor(self, name):
        t = self.meta["tensors"][name]
        return t, self.gguf[t["offset"]:t["offset"] + t["bytes"]]

    def f32(self, name):
        return np.frombuffer(self.tensor(name)[1], "<f4")

    def test_the_fp8_projections_keep_their_codes_and_scale(self):
        B = "language_model.backbone.layers."
        for name, src in (("blk.0.ssm_in.weight", B + "0.mixer.in_proj.weight"), ("blk.0.ssm_out.weight", B + "0.mixer.out_proj.weight"),
                          ("blk.1.attn_q.weight", B + "1.mixer.q_proj.weight"), ("blk.1.attn_k.weight", B + "1.mixer.k_proj.weight"),
                          ("blk.1.attn_output.weight", B + "1.mixer.o_proj.weight"), ("output.weight", "language_model.lm_head.weight")):
            with self.subTest(tensor=name):
                codes, s = self.fx.f8[src]
                t, raw = self.tensor(name)
                self.assertEqual((t["type"], t["type_id"], t["shape"]), ("F8_E4M3", 44, [codes.shape[1], codes.shape[0]]))
                self.assertEqual(raw, codes.tobytes())
                self.assertEqual(self.f32(name[:-len(".weight")] + ".scale").tolist(), [s])

    def test_the_mamba_tensors(self):
        m = "language_model.backbone.layers.0.mixer."
        np.testing.assert_array_equal(self.f32("blk.0.ssm_conv1d.weight").reshape(CD, DC), self.fx.v[m + "conv1d.weight"][:, 0])
        np.testing.assert_array_equal(self.f32("blk.0.ssm_conv1d.bias"), self.fx.v[m + "conv1d.bias"])
        np.testing.assert_array_equal(self.f32("blk.0.ssm_dt.bias"), self.fx.v[m + "dt_bias"])
        np.testing.assert_array_equal(self.f32("blk.0.ssm_d"), self.fx.v[m + "D"])
        np.testing.assert_array_equal(self.f32("blk.0.ssm_norm.weight"), self.fx.v[m + "norm.weight"])
        np.testing.assert_array_equal(self.f32("blk.0.ssm_a"), (-np.exp(self.fx.v[m + "A_log"].astype(np.float64))).astype(np.float32))
        self.assertEqual(self.meta["tensors"]["blk.0.ssm_conv1d.weight"]["shape"], [DC, CD])

    def test_the_moe_tensors(self):
        m = "language_model.backbone.layers.2.mixer."
        np.testing.assert_array_equal(self.f32("blk.2.ffn_gate_inp.weight").reshape(NE, H), self.fx.v[m + "gate.weight"])
        np.testing.assert_array_equal(self.f32("blk.2.exp_probs_b.bias"), self.fx.v[m + "gate.e_score_correction_bias"])
        for name, src, shape in (("blk.2.ffn_latent_down.weight", m + "fc1_latent_proj.weight", (LAT, H)),
                                 ("blk.2.ffn_latent_up.weight", m + "fc2_latent_proj.weight", (H, LAT)),
                                 ("token_embd.weight", "language_model.backbone.embeddings.weight", (VOCAB, H))):
            t, raw = self.tensor(name)
            self.assertEqual(t["type"], "BF16", name)
            got = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
            np.testing.assert_array_equal(got, self.fx.v[src], name)

    def test_the_shared_expert_is_nvfp4_planes(self):
        m = "language_model.backbone.layers.2.mixer.shared_experts."
        for name, src, rows, K in (("blk.2.ffn_up_shexp.weight", m + "up_proj", FFS, H), ("blk.2.ffn_down_shexp.weight", m + "down_proj", H, FFS)):
            codes, scales, s2 = self.fx.nv[src]
            t, raw = self.tensor(name)
            self.assertEqual((t["type"], t["type_id"], t["shape"]), ("NVFP4_PLANES", 45, [K, rows]))
            raw = np.frombuffer(raw, np.uint8)
            self.assertEqual(raw[:rows * K // 2].tobytes(), codes.tobytes())
            tiles = raw[rows * K // 2:]
            for r in (0, 37, rows - 1):
                for g in (0, 5, K // 16 - 1):
                    at = ((r // 128) * (K // 64) + g // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + g % 4
                    self.assertEqual(tiles[at], scales[r, g])
            self.assertEqual(self.f32(name[:-len(".weight")] + ".scale").tolist(), [s2])

    def test_the_experts_move_bit_for_bit(self):
        lay = [l.split() for l in self.mf.read("experts.layout").decode().splitlines() if l and l[0] != "#"]
        self.assertEqual(len(lay), 1)
        l, gu, d, off, blob = map(int, lay[0])
        self.assertEqual((l, gu, d, off), (2, 40, 40, 0))
        up_bytes = FF * S.nvfp4_row_bytes(LAT)
        self.assertEqual(blob, up_bytes + LAT * S.nvfp4_row_bytes(FF) + 16)
        experts = self.mf.read("experts")
        self.assertEqual(len(experts), NE * blob)
        m = "language_model.backbone.layers.2.mixer.experts."
        for e in range(NE):
            b = np.frombuffer(experts[off + e * blob:off + (e + 1) * blob], np.uint8)
            uc, us, us2 = self.fx.nv[m + f"{e}.up_proj"]
            dc, ds, ds2 = self.fx.nv[m + f"{e}.down_proj"]
            np.testing.assert_array_equal(basalt_values(b[:up_bytes], LAT), modelopt_values(uc, us))
            np.testing.assert_array_equal(basalt_values(b[up_bytes:blob - 16], FF), modelopt_values(dc, ds))
            self.assertEqual(np.frombuffer(b[blob - 16:].tobytes(), "<f4").tolist(), [us2, 0.0, ds2, 0.0])

    def test_the_metadata(self):
        kv = self.meta["gguf_kv"]
        self.assertEqual((self.meta["arch"], kv["general.architecture"]), ("nemotronh", "nemotronh"))
        self.assertEqual(kv["nemotronh.layer_kinds"], [0, 1, 2])
        for k, want in (("block_count", 3), ("embedding_length", H), ("vocab_size", VOCAB), ("attention.head_count", NH),
                        ("attention.head_count_kv", NKV), ("attention.key_length", HD), ("ssm.conv_kernel", DC),
                        ("ssm.state_size", DS), ("ssm.head_count", MH), ("ssm.head_dim", MHD), ("ssm.group_count", NG),
                        ("ssm.inner_size", DI), ("expert_count", NE), ("expert_used_count", TOPK),
                        ("expert_feed_forward_length", FF), ("expert_shared_feed_forward_length", FFS),
                        ("expert_latent_size", LAT), ("expert_weights_scale", 5.0), ("context_length", 1048576)):
            self.assertEqual(kv["nemotronh." + k], want, k)
        self.assertAlmostEqual(kv["nemotronh.attention.layer_norm_rms_epsilon"], 1e-5, places=9)
        self.assertEqual(kv["tokenizer.ggml.pre"], "nemotronh")
        self.assertEqual(len(kv["tokenizer.ggml.tokens"]), VOCAB)
        self.assertEqual(kv["tokenizer.ggml.eos_token_id"], TOKENS.index("<|im_end|>"))
        self.assertEqual(kv["general.license.name"], "openmdw-1.1")

    def test_the_mtp_head_and_vision_are_left_out(self):
        names = set(self.meta["tensors"])
        self.assertFalse(any(n.startswith(("mtp", "v.", "mm.")) or "eh_proj" in n for n in names))
        self.assertNotIn("vision/mmproj", self.mf.names())
        self.assertFalse(any(n.endswith(("k_scale", "input_scale")) for n in names))


class RefusalTest(unittest.TestCase):
    def test_a_sliding_window_is_refused(self):
        N.Geometry(llm_config())
        with self.assertRaisesRegex(S.ConvertError, "sliding window"):
            N.Geometry(llm_config(sliding_window=4096))
        with self.assertRaisesRegex(S.ConvertError, "mlp_hidden_act"):
            N.Geometry(llm_config(mlp_hidden_act="silu"))

    def test_the_pattern_string_is_read(self):
        self.assertEqual(N.layer_kinds({"num_hidden_layers": 4, "hybrid_override_pattern": "ME*E"}),
                         ["mamba", "moe", "attention", "moe"])
        with self.assertRaises(S.ConvertError):
            N.layer_kinds({"num_hidden_layers": 2, "hybrid_override_pattern": "M-"})


if __name__ == "__main__":
    unittest.main()
