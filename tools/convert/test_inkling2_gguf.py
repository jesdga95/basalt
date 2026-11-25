"""Unit tests for tools/convert/inkling2_gguf.py (Inkling 2 GGUF -> .basalt conversion)."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "convert"))

from gguf_reader import GGUFFile  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import ModelFile  # noqa: E402
import inkling2_gguf as IG  # noqa: E402
import model_sections as S  # noqa: E402


def _write_synthetic_inkling2_gguf(path: pathlib.Path, mode: str = "BF16") -> None:
    """Create a tiny valid inkling2_loop GGUF (2 layers, 2 experts, hidden=64, inter=64)."""
    rng = np.random.RandomState(42)
    n_layers = 2
    n_expert = 2
    hidden = 64
    inter = 64
    vocab_padded = 16
    vocab_unpadded = 12

    gw = GGUFWriter()
    gw.add("general.architecture", "inkling2_loop")
    gw.add("general.type", "model")
    gw.add("general.name", f"Inkling2-Tiny-{mode}")
    gw.add("general.license", "apache-2.0")
    gw.add("general.sampling.temp", 1.0, "f32")
    gw.add("inkling2_loop.block_count", n_layers, "u32")
    gw.add("inkling2_loop.context_length", 1024, "u32")
    gw.add("inkling2_loop.embedding_length", hidden, "u32")
    gw.add("inkling2_loop.vocab_size", vocab_padded, "u32")
    gw.add("inkling2_loop.vocab_size_unpadded", vocab_unpadded, "u32")
    gw.add("inkling2_loop.attention.head_count", [2] * n_layers, "array:u32")
    gw.add("inkling2_loop.attention.head_count_kv", [1] * n_layers, "array:u32")
    gw.add("inkling2_loop.attention.key_length", 32, "u32")
    gw.add("inkling2_loop.attention.key_length_swa", 32, "u32")
    gw.add("inkling2_loop.attention.layer_norm_rms_epsilon", 1e-6, "f32")
    gw.add("inkling2_loop.attention.sliding_window", 64, "u32")
    gw.add("inkling2_loop.attention.sliding_window_pattern", [True, False], "array:bool")
    gw.add("inkling2_loop.attention.relative_length", 8, "u32")
    gw.add("inkling2_loop.attention.relative_extent", 64, "u32")
    gw.add("inkling2_loop.attention.qk_clip", 22.6, "f32")
    gw.add("inkling2_loop.attention.relative_clip", 8.0, "f32")
    gw.add("inkling2_loop.attention.log_scale_floor", 64, "u32")
    gw.add("inkling2_loop.attention.log_scale_alpha", 0.1, "f32")
    gw.add("inkling2_loop.shortconv.l_cache", 4, "u32")
    gw.add("inkling2_loop.attn_res.block_size", 2, "u32")
    gw.add("inkling2_loop.expert_count", n_expert, "u32")
    gw.add("inkling2_loop.expert_used_count", 1, "u32")
    gw.add("inkling2_loop.expert_feed_forward_length", inter, "u32")
    gw.add("inkling2_loop.expert_weights_scale", 8.0, "f32")
    gw.add("inkling2_loop.activation.situ_beta", 4.0, "f32")
    gw.add("inkling2_loop.activation.situ_linear_beta", 25.0, "f32")
    gw.add("inkling2_loop.final_logit_softcapping", 30.0, "f32")
    gw.add("inkling2_loop.loop.start", 1, "u32")
    gw.add("inkling2_loop.loop.end", 2, "u32")
    gw.add("inkling2_loop.loop.count", 2, "u32")
    gw.add("inkling2_loop.loop.gate_rank", 16, "u32")

    # Minimal tokenizer metadata
    toks = [f"t{i}" for i in range(vocab_unpadded)]
    gw.add("tokenizer.ggml.model", "gpt2")
    gw.add("tokenizer.ggml.tokens", toks, "array:string")
    gw.add("tokenizer.ggml.token_type", [1] * vocab_unpadded, "array:i32")
    gw.add("tokenizer.ggml.merges", [], "array:string")
    gw.add("tokenizer.chat_template", "{{ messages }}")

    def add_mat(name: str, shape: list[int], fmt: str):
        arr = rng.randn(*reversed(shape)).astype(np.float32) * 0.05
        if fmt == "F32":
            gw.add_raw(name, shape, "F32", arr.tobytes())
        elif fmt == "BF16":
            gw.add_raw(name, shape, "BF16", IG._f32_to_bf16(arr).tobytes())
        elif fmt == "Q8_0":
            gw.add_raw(name, shape, "Q8_0", S.q8_0(arr.reshape(-1, shape[0])))
        elif fmt == "Q4_K":
            # Dummy Q4_K bytes for refusal test
            gw.add_raw(name, shape, "Q4_K", b"\x00" * (arr.size // 256 * 144))

    add_mat("token_embd.weight", [hidden, vocab_padded], "BF16")
    add_mat("output.weight", [hidden, vocab_padded], "BF16")
    add_mat("token_embd_norm.weight", [hidden], "F32")
    add_mat("output_norm.weight", [hidden], "F32")
    add_mat("output_res_query.weight", [hidden], "F32")

    proj_fmt = "Q8_0" if mode == "Q8_0" else "BF16"
    exp_fmt = mode

    for l in range(n_layers):
        add_mat(f"blk.{l}.attn_q.weight", [hidden, hidden], proj_fmt)
        add_mat(f"blk.{l}.attn_k.weight", [hidden, 32], proj_fmt)
        add_mat(f"blk.{l}.attn_v.weight", [hidden, 32], proj_fmt)
        add_mat(f"blk.{l}.attn_output.weight", [hidden, hidden], proj_fmt)
        add_mat(f"blk.{l}.attn_sink_proj.weight", [hidden, 2], proj_fmt)
        add_mat(f"blk.{l}.attn_rel_bias.weight", [8, 64], "F32")
        add_mat(f"blk.{l}.attn_norm.weight", [hidden], "F32")
        add_mat(f"blk.{l}.ffn_norm.weight", [hidden], "F32")
        add_mat(f"blk.{l}.ffn_gate_inp.weight", [hidden, n_expert], "F32")
        add_mat(f"blk.{l}.exp_probs_b.bias", [n_expert], "F32")
        add_mat(f"blk.{l}.ffn_gate_up_exps.weight", [hidden, 2 * inter, n_expert], exp_fmt)
        add_mat(f"blk.{l}.ffn_down_exps.weight", [inter, hidden, n_expert], exp_fmt)
        if l == 1:
            add_mat(f"blk.{l}.loop_q_ctrl_down.weight", [2 * hidden, 16], "F32")

    gw.write(path)


class Inkling2GgufTest(unittest.TestCase):
    def _run_convert(self, gguf_path: pathlib.Path, out_path: pathlib.Path):
        return subprocess.run(
            [sys.executable, str(ROOT / "tools" / "convert" / "convert.py"),
             "--gguf", str(gguf_path), "--out", str(out_path), "--workers", "2", "--force"],
            capture_output=True, text=True,
        )

    def test_bf16_gguf_to_nvfp4_basalt(self):
        with tempfile.TemporaryDirectory() as d:
            td = pathlib.Path(d)
            src = td / "tiny-bf16.gguf"
            out = td / "tiny.basalt"
            _write_synthetic_inkling2_gguf(src, "BF16")
            r = self._run_convert(src, out)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

            m = ModelFile(str(out))
            for sec in ("experts", "experts.layout", "gguf", "meta"):
                self.assertIn(sec, m.sections)
                self.assertTrue(m.verify(sec))
            meta = json.loads(m.read("meta"))
            self.assertEqual(meta["arch"], "inkling2")
            kv = meta["gguf_kv"]
            self.assertEqual(kv["general.architecture"], "inkling2")
            self.assertEqual(kv["inkling2.attention.head_count"], 2)
            self.assertEqual(kv["inkling2.attention.head_count_kv"], 1)
            self.assertEqual(kv["inkling2.expert_gate_softcap"], 4.0)
            self.assertEqual(kv["inkling2.expert_up_softcap"], 25.0)

            tensors = meta["tensors"]
            # Vocab trimmed from 16 -> 12
            self.assertEqual(tensors["token_embd.weight"]["shape"], [64, 12])
            self.assertEqual(tensors["output.weight"]["shape"], [64, 12])
            # attn_sink_proj renamed to attn_sink and quantized to Q8_0
            self.assertIn("blk.0.attn_sink.weight", tensors)
            self.assertEqual(tensors["blk.0.attn_sink.weight"]["type"], "Q8_0")
            self.assertEqual(tensors["blk.0.attn_q.weight"]["type"], "Q8_0")
            # Loop controller converted F32 -> BF16
            self.assertEqual(tensors["blk.1.loop_q_ctrl_down.weight"]["type"], "BF16")
            # attn_rel_bias transposed [d_rel, rel_extent] -> [rel_extent, d_rel]
            self.assertEqual(tensors["blk.0.attn_rel_bias.weight"]["shape"], [64, 8])

            # Experts layout is NVFP4 (type 40)
            layout = m.read("experts.layout").decode()
            self.assertIn("0 40 40 0 ", layout)

    def test_q8_0_gguf_passthrough(self):
        with tempfile.TemporaryDirectory() as d:
            td = pathlib.Path(d)
            src = td / "tiny-q8.gguf"
            out = td / "tiny-q8.basalt"
            _write_synthetic_inkling2_gguf(src, "Q8_0")
            r = self._run_convert(src, out)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

            m = ModelFile(str(out))
            layout = m.read("experts.layout").decode()
            self.assertIn("0 8 8 0 ", layout)

    def test_unsupported_quant_refused(self):
        with tempfile.TemporaryDirectory() as d:
            td = pathlib.Path(d)
            src = td / "tiny-q4k.gguf"
            out = td / "tiny-q4k.basalt"
            _write_synthetic_inkling2_gguf(src, "Q4_K")
            r = self._run_convert(src, out)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("supports BF16, F16, and Q8_0", r.stderr)


if __name__ == "__main__":
    unittest.main()
