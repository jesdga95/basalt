"""Tests for checkpoint_gemma4.py's vision tower (the pack's vision/mmproj) on a tiny synthetic checkpoint shaped as
Google's release: one block of the gemma4_vision tower, its position tables, standardisation and embedding projection.

    python3 tools/convert/test_checkpoint_gemma4.py
    python3 -m unittest discover -s tools/convert -p 'test_checkpoint_gemma4.py'"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import checkpoint as C  # noqa: E402
import checkpoint_gemma4 as G  # noqa: E402
import model_sections as S  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402
from test_checkpoint import write_safetensors  # noqa: E402

VH, VHEADS, VHD, VFF, PATCH, POOL, TABLE, TEXT_H = 8, 2, 4, 12, 2, 3, 5, 16
rng = np.random.default_rng(5)


def bf16_values(shape) -> np.ndarray:
    x = rng.standard_normal(shape).astype(np.float32)
    return (x.view(np.uint32) & 0xFFFF0000).view(np.float32)


class Fixture:
    def __init__(self, d: Path, extra: dict | None = None, vision_cfg: dict | None = None):
        self.v = {}
        V, B = G.VISION, G.VISION + "encoder.layers.0."
        shapes = {V + "patch_embedder.input_proj.weight": (VH, 3 * PATCH * PATCH),
                  V + "patch_embedder.position_embedding_table": (2, TABLE, VH),
                  V + "std_bias": (VH,), V + "std_scale": (VH,),
                  "model.embed_vision.embedding_projection.weight": (TEXT_H, VH),
                  B + "input_layernorm.weight": (VH,), B + "post_attention_layernorm.weight": (VH,),
                  B + "pre_feedforward_layernorm.weight": (VH,), B + "post_feedforward_layernorm.weight": (VH,),
                  B + "self_attn.q_norm.weight": (VHD,), B + "self_attn.k_norm.weight": (VHD,),
                  B + "self_attn.q_proj.linear.weight": (VH, VH), B + "self_attn.k_proj.linear.weight": (VH, VH),
                  B + "self_attn.v_proj.linear.weight": (VH, VH), B + "self_attn.o_proj.linear.weight": (VH, VH),
                  B + "mlp.gate_proj.linear.weight": (VFF, VH), B + "mlp.up_proj.linear.weight": (VFF, VH),
                  B + "mlp.down_proj.linear.weight": (VH, VFF),
                  "model.language_model.norm.weight": (TEXT_H,)}
        shapes.update(extra or {})
        tens = {}
        for k, sh in shapes.items():
            self.v[k] = bf16_values(sh)
            tens[k] = ("BF16", sh, (self.v[k].view(np.uint32) >> 16).astype("<u2").tobytes())
        write_safetensors(d / "model-00001-of-00001.safetensors", tens)
        vc = {"model_type": "gemma4_vision", "hidden_activation": "gelu_pytorch_tanh", "attention_bias": False,
              "use_clipped_linears": False, "standardize": True, "hidden_size": VH, "patch_size": PATCH,
              "pooling_kernel_size": POOL, "num_attention_heads": VHEADS, "head_dim": VHD, "intermediate_size": VFF,
              "num_hidden_layers": 1, "rms_norm_eps": 1e-6, "position_embedding_size": TABLE, "default_output_length": 280,
              "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"}}
        vc.update(vision_cfg or {})
        (d / "config.json").write_text(json.dumps({"model_type": "gemma4", "vision_config": vc,
                                                   "text_config": {"model_type": "gemma4_text", "hidden_size": TEXT_H}}))
        (d / "processor_config.json").write_text(json.dumps({"image_processor": {
            "patch_size": PATCH, "pooling_kernel_size": POOL, "do_normalize": False, "image_mean": [0.0] * 3,
            "image_std": [1.0] * 3, "max_soft_tokens": 280}}))


def build(fx_args=None) -> tuple[Fixture, GGUFFile, bytes]:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fx = Fixture(d, **(fx_args or {}))
        ck = C.Checkpoint(d)
        out = d / "mmproj.gguf"
        G.mmproj(ck, out, "Tiny")
        return fx, GGUFFile(out), out.read_bytes()


class VisionTower(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx, cls.g, cls.raw = build()
        cls.t = {x.name: x for x in cls.g.tensors}

    def values(self, name):
        x = self.t[name]
        at = self.g.data_start + x.offset
        if x.type_name == "F32":
            return np.frombuffer(self.raw[at:at + 4 * x.elements], "<f4")
        return (np.frombuffer(self.raw[at:at + 2 * x.elements], "<u2").astype(np.uint32) << 16).view(np.float32)

    def test_the_metadata_is_llama_cpps_gemma4v(self):
        md = self.g.metadata
        self.assertEqual((md["general.architecture"], md["clip.vision.projector_type"]), ("clip", "gemma4v"))
        self.assertEqual([md["clip.vision." + k] for k in ("patch_size", "embedding_length", "feed_forward_length", "block_count",
                                                           "attention.head_count", "projector.scale_factor", "projection_dim")],
                         [PATCH, VH, VFF, 1, VHEADS, POOL, TEXT_H])
        self.assertEqual(md["clip.vision.rope.freq_base"], 100.0)
        self.assertEqual((md["clip.vision.image_mean"], md["clip.vision.image_std"]), ([0.0] * 3, [1.0] * 3))
        self.assertEqual(len(self.t), 1 + 4 + 13)   # the patch embedding, the globals, one block

    def test_the_patch_embedding_is_a_convolution(self):
        # [out][(row, column, channel)] as llama.cpp's convolution [out][channel][row][column]
        pw = self.fx.v[G.VISION + "patch_embedder.input_proj.weight"].reshape(VH, PATCH, PATCH, 3).transpose(0, 3, 1, 2)
        self.assertEqual((self.t["v.patch_embd.weight"].type_name, self.t["v.patch_embd.weight"].shape), ("F32", [PATCH, PATCH, 3, VH]))
        np.testing.assert_array_equal(self.values("v.patch_embd.weight"), pw.reshape(-1))

    def test_the_tensors_move_as_stored(self):
        V, B = G.VISION, G.VISION + "encoder.layers.0."
        self.assertEqual(self.t["v.position_embd.weight"].shape, [VH, TABLE, 2])
        for name, src in (("v.position_embd.weight", V + "patch_embedder.position_embedding_table"), ("v.std_bias", V + "std_bias"),
                          ("v.std_scale", V + "std_scale"), ("mm.input_projection.weight", "model.embed_vision.embedding_projection.weight"),
                          ("v.blk.0.ln1.weight", B + "input_layernorm.weight"),
                          ("v.blk.0.attn_post_norm.weight", B + "post_attention_layernorm.weight"),
                          ("v.blk.0.ln2.weight", B + "pre_feedforward_layernorm.weight"),
                          ("v.blk.0.ffn_post_norm.weight", B + "post_feedforward_layernorm.weight"),
                          ("v.blk.0.attn_q_norm.weight", B + "self_attn.q_norm.weight"),
                          ("v.blk.0.attn_q.weight", B + "self_attn.q_proj.linear.weight"),
                          ("v.blk.0.attn_out.weight", B + "self_attn.o_proj.linear.weight"),
                          ("v.blk.0.ffn_gate.weight", B + "mlp.gate_proj.linear.weight"),
                          ("v.blk.0.ffn_down.weight", B + "mlp.down_proj.linear.weight")):
            np.testing.assert_array_equal(self.values(name), self.fx.v[src].reshape(-1), err_msg=name)
        for name in ("v.blk.0.ln1.weight", "v.std_bias", "v.position_embd.weight"):
            self.assertEqual(self.t[name].type_name, "F32")
        for name in ("v.blk.0.attn_q.weight", "v.blk.0.ffn_up.weight", "mm.input_projection.weight"):
            self.assertEqual(self.t[name].type_name, "BF16")

    def test_an_unknown_vision_tensor_is_refused(self):
        with self.assertRaisesRegex(S.ConvertError, "no pack counterpart"):
            build({"extra": {G.VISION + "encoder.layers.0.layer_scalar": (1,)}})

    def test_clipped_linears_are_refused(self):
        with self.assertRaisesRegex(S.ConvertError, "use_clipped_linears"):
            build({"vision_cfg": {"use_clipped_linears": True}})


if __name__ == "__main__":
    unittest.main()
