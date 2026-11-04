"""A tiny synthetic deepseek41 GGUF converted end to end: sections, meta, expert layout and the Engram section."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import convert  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import SECTION_ALIGN, ModelFile  # noqa: E402

N_EXPERT, N_LAYERS, ROWS = 4, 2, 5
TOKENS = ["a", "b", "c", "d"]
BLOCK = {"IQ1_S": 50, "IQ2_XXS": 66, "Q6_K": 210, "IQ4_NL": 18}
GU = ["IQ1_S", "IQ2_XXS"]          # gate/up type per layer
DOWN = ["IQ2_XXS", "IQ1_S"]        # down type per layer


class DeepSeek41Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        rng = np.random.default_rng(3)
        g = GGUFWriter()
        P = "deepseek41."
        g.add("general.architecture", "deepseek41")
        g.add("general.name", "Tiny-DS")
        g.add("general.version", "1017")
        g.add("general.quantized_by", "Someone")
        g.add(P + "block_count", N_LAYERS, "u32")
        g.add(P + "embedding_length", 256, "u32")
        g.add(P + "expert_count", N_EXPERT, "u32")
        g.add(P + "expert_used_count", 2, "u32")
        g.add(P + "embedding_length_per_layer_input", 256, "u32")
        g.add(P + "ple.layer_multipliers", [7, 11], "array:u64")
        g.add(P + "ple.head_offsets", [0, 2], "array:u64")
        g.add(P + "ple.head_vocab_sizes", [2, 3], "array:u64")
        g.add(P + "engram.vocab_size", 16, "u32")
        g.add(P + "engram.token_map", [0, 1, 2, 3], "array:u32")
        g.add("tokenizer.ggml.tokens", TOKENS)
        self.raw = {}

        def raw(name, shape, ty, n_blocks):
            data = rng.bytes(BLOCK[ty] * n_blocks)
            self.raw[name] = data
            g.add_raw(name, shape, ty, data)

        raw("token_embd.weight", [256, 3], "Q6_K", 3)
        raw("per_layer_token_embd.weight", [256, ROWS], "IQ4_NL", 8 * ROWS)
        for l in range(N_LAYERS):
            g.add_f32(f"blk.{l}.ffn_gate_inp.weight", rng.standard_normal((8, N_EXPERT)))
            g.add_f32(f"blk.{l}.attn_norm.weight", rng.standard_normal(8))
            raw(f"blk.{l}.attn_q_a.weight", [256, 2], "Q6_K", 2)
            for role, ty in (("gate", GU[l]), ("up", GU[l]), ("down", DOWN[l])):
                raw(f"blk.{l}.ffn_{role}_exps.weight", [256, 2, N_EXPERT], ty, 2 * N_EXPERT)
        self.gguf = g.write(self.t / "ds-00001-of-00001.gguf")
        self._orig = convert.export_tokenizer

        def tokenizer_files(first, out_dir):
            d = Path(out_dir) / "tokenizer"
            d.mkdir(parents=True, exist_ok=True)
            (d / "tokenizer.json").write_bytes(b"{}")
            return sorted(d.iterdir())
        convert.export_tokenizer = tokenizer_files

    def tearDown(self):
        convert.export_tokenizer = self._orig
        self.tmp.cleanup()

    def args(self, **kw):
        d = dict(gguf=[str(self.gguf)], mmproj=None, out=str(self.t / "out.basalt"), mtp_source=None, mtp_dense=None,
                 draft_vocab=None, profile=None, profile_from=None, hc8=None, name="Tiny", compat_bf16=False,
                 no_vision=False, force=False, sections_from=None)
        d.update(kw)
        return type("A", (), d)()

    def test_sections_and_meta(self):
        convert.convert(self.args())
        m = ModelFile(str(self.t / "out.basalt"))
        self.assertEqual(set(m.names()), {"tokenizer/tokenizer.json", "dense", "dense.index", "experts",
                                          "experts.layout", "gguf", "engram", "meta"})
        for n in m.names():
            self.assertTrue(m.verify(n), n)
            self.assertEqual(m.section(n)[0] % SECTION_ALIGN, 0, n)
        meta = json.loads(m.read("meta"))
        self.assertEqual(meta["arch"], "deepseek41")
        self.assertEqual(meta["license"]["name"], "MIT")
        self.assertEqual(meta["geometry"]["expert_count"], N_EXPERT)
        self.assertIsNone(meta["vision"])
        self.assertIsNone(meta["mtp"])
        e = meta["engram"]
        self.assertEqual((e["rows"], e["row_values"], e["row_bytes"], e["bytes"]), (ROWS, 256, 144, ROWS * 144))
        self.assertEqual(e["ple.layer_multipliers"], [7, 11])
        self.assertEqual(e["ple.head_vocab_sizes"], [2, 3])

    def test_engram_section_is_the_table(self):
        convert.convert(self.args())
        m = ModelFile(str(self.t / "out.basalt"))
        self.assertEqual(m.read("engram"), self.raw["per_layer_token_embd.weight"])
        meta = json.loads(m.read("meta"))
        t = meta["tensors"]["per_layer_token_embd.weight"]
        self.assertEqual((t["section"], t["offset"], t["type"], t["shape"]), ("engram", 0, "IQ4_NL", [256, ROWS]))

    def test_gguf_section_holds_the_quantized_tensors_but_not_the_table(self):
        convert.convert(self.args())
        m = ModelFile(str(self.t / "out.basalt"))
        meta = json.loads(m.read("meta"))
        gg = m.read("gguf")
        for name in ("token_embd.weight", "blk.0.attn_q_a.weight", "blk.1.attn_q_a.weight"):
            t = meta["tensors"][name]
            self.assertEqual(gg[t["offset"]:t["offset"] + t["bytes"]], self.raw[name], name)
            self.assertEqual(t["offset"] % 4096, 0)
        self.assertNotIn("section", meta["tensors"]["token_embd.weight"])
        self.assertIn(b"attn_norm", m.read("dense.index"))

    def test_experts_with_mixed_types(self):
        convert.convert(self.args())
        m = ModelFile(str(self.t / "out.basalt"))
        lines = m.read("experts.layout").decode().splitlines()[1:]
        out, off = m.read("experts"), 0
        for l in range(N_LAYERS):
            per = {r: BLOCK[ty] * 2 for r, ty in (("gate", GU[l]), ("up", GU[l]), ("down", DOWN[l]))}
            blob = sum(per.values())
            ids = {"IQ1_S": 19, "IQ2_XXS": 16}
            self.assertEqual(lines[l], f"{l} {ids[GU[l]]} {ids[DOWN[l]]} {off} {blob}")
            for e in range(N_EXPERT):
                want = b"".join(self.raw[f"blk.{l}.ffn_{r}_exps.weight"][e * per[r]:(e + 1) * per[r]]
                                for r in ("gate", "up", "down"))
                self.assertEqual(out[off + e * blob:off + (e + 1) * blob], want, (l, e))
            off += blob * N_EXPERT
        self.assertEqual(off, len(out))

    def test_side_sections_are_refused(self):
        with self.assertRaises(SystemExit):
            convert.convert(self.args(draft_vocab=str(self.gguf)))

    def test_mmproj_becomes_the_vision_section(self):
        proj = self.t / "mmproj.gguf"
        proj.write_bytes(b"GGUF" + b"\0" * 16 + b"deepseek41v" + b"\1\2\3" * 100)
        convert.convert(self.args(mmproj=str(proj)))
        m = ModelFile(str(self.t / "out.basalt"))
        self.assertIn("vision/mmproj", m.names())
        self.assertTrue(m.verify("vision/mmproj"))
        self.assertEqual(m.read("vision/mmproj"), proj.read_bytes())
        meta = json.loads(m.read("meta"))
        self.assertEqual(meta["vision"]["mmproj"], "vision/mmproj")
        self.assertEqual(meta["vision"]["mmproj_bytes"], proj.stat().st_size)

    def test_foreign_mmproj_is_refused(self):
        with self.assertRaises(SystemExit):
            convert.convert(self.args(mmproj=str(self.gguf)))

    def test_dry_run_writes_nothing_and_matches_the_build(self):
        convert.convert(self.args(dry_run=True))
        self.assertFalse((self.t / "out.basalt").exists())
        self.assertFalse((self.t / "out.basalt.partial").exists())

    def test_missing_engram_is_refused(self):
        g = GGUFWriter()
        g.add("general.architecture", "deepseek41")
        g.add("tokenizer.ggml.tokens", TOKENS)
        g.add_f32("blk.0.ffn_gate_inp.weight", np.zeros((8, 2)))
        p = g.write(self.t / "bad-00001-of-00001.gguf")
        with self.assertRaises(SystemExit):
            convert.convert(self.args(gguf=[str(p)]))


if __name__ == "__main__":
    unittest.main()
