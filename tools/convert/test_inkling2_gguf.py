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


# mode MIXED: an ISTA GSQ-RCO-shaped GGUF. Per layer (gate_up, down) expert types, and the dense types.
MIXED_EXPERTS = [("IQ1_S", "Q2_0"), ("IQ3_S", "IQ4_NL")]
MIXED_DENSE = {"attn_q.weight": "Q5_K", "attn_k.weight": "IQ4_XS", "attn_v.weight": "Q8_0",
               "attn_output.weight": "Q6_K", "attn_sink_proj.weight": "BF16"}
MIXED_OUTPUT = "Q6_K"
# byte offsets of each block's fp16 scales, set to a small finite value so the dequantized weights are finite
_FP16_FIELDS = {"Q5_K": (0, 2), "Q6_K": (208,), "IQ4_XS": (0,)}


def _random_blocks(rng: np.random.RandomState, type_name: str, n_values: int) -> bytes:
    """Random blocks of a quantized type; their scale fields finite where the type is dequantized by the test."""
    block_elems, block_bytes = IG.BLOCK_GEOMETRY[type_name]
    b = rng.randint(0, 256, size=(n_values // block_elems, block_bytes)).astype(np.uint8)
    for off in _FP16_FIELDS.get(type_name, ()):
        d = (rng.rand(b.shape[0]) * 2e-3 + 1e-4).astype(np.float16)
        b[:, off:off + 2] = d.view(np.uint8).reshape(-1, 2)
    return b.tobytes()


def _write_synthetic_inkling2_gguf(path: pathlib.Path, mode: str = "BF16") -> None:
    """Create a tiny valid inkling2_loop GGUF (2 layers, 2 experts, hidden=64 (MIXED: 256), inter=64)."""
    rng = np.random.RandomState(42)
    n_layers = 2
    n_expert = 2
    hidden = 256 if mode == "MIXED" else 64
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
        elif fmt == "Q4_K" and mode == "Q4_K":
            # Dummy Q4_K bytes for refusal test
            gw.add_raw(name, shape, "Q4_K", b"\x00" * (arr.size // 256 * 144))
        else:
            gw.add_raw(name, shape, fmt, _random_blocks(rng, fmt, arr.size))

    mixed = mode == "MIXED"
    add_mat("token_embd.weight", [hidden, vocab_padded], "BF16")
    add_mat("output.weight", [hidden, vocab_padded], MIXED_OUTPUT if mixed else "BF16")
    add_mat("token_embd_norm.weight", [hidden], "F32")
    add_mat("output_norm.weight", [hidden], "F32")
    add_mat("output_res_query.weight", [hidden], "F32")

    proj_fmt = "Q8_0" if mode == "Q8_0" else "BF16"
    exp_fmt = mode

    def dense(suffix: str) -> str:
        return MIXED_DENSE[suffix] if mixed else proj_fmt

    for l in range(n_layers):
        add_mat(f"blk.{l}.attn_q.weight", [hidden, hidden], dense("attn_q.weight"))
        add_mat(f"blk.{l}.attn_k.weight", [hidden, 32], dense("attn_k.weight"))
        add_mat(f"blk.{l}.attn_v.weight", [hidden, 32], dense("attn_v.weight"))
        add_mat(f"blk.{l}.attn_output.weight", [hidden, hidden], dense("attn_output.weight"))
        add_mat(f"blk.{l}.attn_sink_proj.weight", [hidden, 2], dense("attn_sink_proj.weight"))
        add_mat(f"blk.{l}.attn_rel_bias.weight", [8, 64], "F32")
        add_mat(f"blk.{l}.attn_norm.weight", [hidden], "F32")
        add_mat(f"blk.{l}.ffn_norm.weight", [hidden], "F32")
        add_mat(f"blk.{l}.ffn_gate_inp.weight", [hidden, n_expert], "F32")
        add_mat(f"blk.{l}.exp_probs_b.bias", [n_expert], "F32")
        gu_fmt, dn_fmt = MIXED_EXPERTS[l] if mixed else (exp_fmt, exp_fmt)
        add_mat(f"blk.{l}.ffn_gate_up_exps.weight", [hidden, 2 * inter, n_expert], gu_fmt)
        add_mat(f"blk.{l}.ffn_down_exps.weight", [inter, hidden, n_expert], dn_fmt)
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

    def test_mixed_quant_gguf(self):
        """ISTA GSQ-RCO shape: IQ experts kept as stored (Q2_0/IQ4_NL down), K-quant/IQ4_XS attention -> Q8_0,
        Q6_K output head -> BF16."""
        with tempfile.TemporaryDirectory() as d:
            td = pathlib.Path(d)
            src = td / "tiny-mixed.gguf"
            out = td / "tiny-mixed.basalt"
            _write_synthetic_inkling2_gguf(src, "MIXED")
            r = self._run_convert(src, out)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

            g = GGUFFile(src)
            by_name = {t.name: t for t in g.tensors}

            def src_raw(name: str) -> bytes:
                t = by_name[name]
                with open(src, "rb") as f:
                    f.seek(g.data_start + t.offset)
                    return f.read(t.expected_bytes())

            m = ModelFile(str(out))
            for sec in ("experts", "experts.layout", "gguf", "meta"):
                self.assertTrue(m.verify(sec))

            # experts: [gate_up of e | down of e] per expert, the GGUF's bytes, each layer in its own types
            hidden, inter, n_expert = 256, 64, 2
            want, offset, lines = b"", 0, []
            for l, (gt, dt) in enumerate(MIXED_EXPERTS):
                gu, dn = src_raw(f"blk.{l}.ffn_gate_up_exps.weight"), src_raw(f"blk.{l}.ffn_down_exps.weight")
                gu_eb, dn_eb = len(gu) // n_expert, len(dn) // n_expert
                self.assertEqual(gu_eb, 2 * inter * IG._row_bytes(gt, hidden))
                self.assertEqual(dn_eb, hidden * IG._row_bytes(dt, inter))
                for e in range(n_expert):
                    want += gu[e * gu_eb:(e + 1) * gu_eb] + dn[e * dn_eb:(e + 1) * dn_eb]
                ids = {"IQ1_S": 19, "Q2_0": 42, "IQ3_S": 21, "IQ4_NL": 20}
                lines.append(f"{l} {ids[gt]} {ids[dt]} {offset} {gu_eb + dn_eb}")
                offset += n_expert * (gu_eb + dn_eb)
            self.assertEqual(m.read("experts"), want)
            layout = m.read("experts.layout").decode().splitlines()
            self.assertEqual(layout[1:], lines)

            meta = json.loads(m.read("meta"))
            tensors = meta["tensors"]
            packed = m.read("gguf")

            def pack_raw(name: str) -> bytes:
                t = tensors[name]
                return packed[t["offset"]:t["offset"] + t["bytes"]]

            def deq(raw: bytes, type_name: str) -> np.ndarray:
                return IG.ggufq.dequantize(np.frombuffer(raw, dtype=np.uint8),
                                           IG.ggufq.GGMLQuantizationType[type_name]).reshape(-1)

            for l in range(2):
                for suffix, st in MIXED_DENSE.items():
                    dst = f"blk.{l}." + IG.RENAME_SUFFIX.get(suffix, suffix)
                    self.assertEqual(tensors[dst]["type"], "Q8_0", dst)
                    raw = src_raw(f"blk.{l}.{suffix}")
                    if st == "Q8_0":
                        self.assertEqual(pack_raw(dst), raw, dst)
                        continue
                    ref = deq(raw, st) if st != "BF16" else IG._read_f32(str(src), g.data_start + by_name[
                        f"blk.{l}.{suffix}"].offset, len(raw), "BF16")
                    got = deq(pack_raw(dst), "Q8_0")
                    self.assertTrue(np.isfinite(ref).all(), dst)
                    # within Q8_0 rounding: d = amax / 127 per 32-value block, codes rounded against d, then d stored
                    # as fp16 (relative error <= 2^-11): |err| <= d / 2 + 127 * d * 2^-11
                    step = np.repeat(np.abs(ref.reshape(-1, 32)).max(axis=1) / 127.0, 32)
                    self.assertTrue((np.abs(got - ref) <= step * (0.5 + 127 / 2048) + 1e-12).all(), dst)

            # output head: Q6_K -> BF16 (round to nearest even), padded vocab rows dropped
            self.assertEqual(tensors["output.weight"]["type"], "BF16")
            self.assertEqual(tensors["output.weight"]["shape"], [hidden, 12])
            ref = deq(src_raw("output.weight"), MIXED_OUTPUT)[:hidden * 12]
            self.assertEqual(pack_raw("output.weight"), IG._f32_to_bf16(ref).tobytes())

    def test_unsupported_quant_refused(self):
        with tempfile.TemporaryDirectory() as d:
            td = pathlib.Path(d)
            src = td / "tiny-q4k.gguf"
            out = td / "tiny-q4k.basalt"
            _write_synthetic_inkling2_gguf(src, "Q4_K")
            r = self._run_convert(src, out)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("blk.0.ffn_gate_up_exps.weight is Q4_K", r.stderr)
            self.assertIn("supports BF16, F16 (rounded to NVFP4) and IQ1_M, IQ1_S", r.stderr)

    def test_q2_0_gate_up_refused(self):
        """Q2_0 gate/up has no softcapped SwiGLU on the CPU: refused, naming the layer."""
        with tempfile.TemporaryDirectory() as d:
            td = pathlib.Path(d)
            src = td / "tiny-q2gu.gguf"
            out = td / "tiny-q2gu.basalt"
            saved = list(MIXED_EXPERTS)
            MIXED_EXPERTS[1] = ("Q2_0", "Q2_0")
            try:
                _write_synthetic_inkling2_gguf(src, "MIXED")
            finally:
                MIXED_EXPERTS[:] = saved
            r = self._run_convert(src, out)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("layer 1 gate_up experts are Q2_0", r.stderr)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
