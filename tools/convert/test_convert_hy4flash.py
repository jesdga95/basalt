"""A tiny synthetic hy_v4_flash GGUF converted end to end: sections, the expert layout (layer 0 is dense), the 3-D tensors, the
Engram section, the next-n block's mtp/* sections, the Hugging Face metadata, and tools/convert/verify_pack.py on the result."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import convert  # noqa: E402
import verify_pack  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import SECTION_ALIGN, ModelFile  # noqa: E402

N_EXPERT, N_BLOCKS, ROWS, WIDTH = 4, 4, 8, 96     # blocks 0 (dense), 1, 2 (MoE), 3 (next-n)
BLOCK = {"Q4_K": 144, "Q6_K": 210, "Q8_0": 34}
DOWN = {1: "Q6_K", 2: "Q4_K"}
TYPE_ID = {"Q4_K": 12, "Q6_K": 14}
P = "hy_v4_flash."
TEMPLATE = "{{ messages }}"


class Hy4FlashTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        rng = np.random.default_rng(3)
        g = GGUFWriter()
        g.add("general.architecture", "hy_v4_flash")
        g.add("general.name", "Huggingface")
        g.add(P + "block_count", N_BLOCKS, "u32")
        g.add(P + "embedding_length", 256, "u32")
        g.add(P + "vocab_size", 4, "u32")
        g.add(P + "expert_count", N_EXPERT, "u32")
        g.add(P + "nextn_predict_layers", 1, "u32")
        g.add(P + "attention.head_count", [2, 2, 4, 2], "array:u32")
        g.add(P + "embedding_length_per_layer_input", WIDTH, "u32")
        g.add(P + "ple.layers", [1, 11, 21, 31], "array:u32")
        g.add(P + "ple.ngram_size", 2, "u32")
        g.add(P + "ple.heads_per_ngram", 2, "u32")
        g.add(P + "ple.layer_multipliers", [7, 11, 13, 17], "array:u64")
        g.add(P + "ple.head_offsets", list(range(8)), "array:u64")
        g.add(P + "ple.head_vocab_sizes", [1] * 8, "array:u64")
        g.add(P + "engram.pad_token_id", 2, "u32")
        g.add(P + "engram.token_map", [0, 1, 2, 3], "array:u32")
        g.add("tokenizer.ggml.tokens", ["a", "b", "c", "d"])
        g.add("tokenizer.chat_template", TEMPLATE)
        self.raw = {}

        def raw(name, shape, ty, n_blocks=None):
            """n_blocks blocks of a quantized type, or one value per element for BF16 and F8_E4M3_RAW (n_blocks = values)."""
            if ty in ("BF16", "F8_E4M3_RAW"):
                n = n_blocks if n_blocks is not None else int(np.prod(shape))
                data = rng.bytes(n * (2 if ty == "BF16" else 1))
            else:
                data = rng.bytes(BLOCK[ty] * n_blocks)
            self.raw[name] = data
            g.add_raw(name, shape, ty, data)

        g.add_f32("output_norm.weight", rng.standard_normal(8))
        raw("token_embd.weight", [256, 3], "Q8_0", 24)
        raw("output.weight", [8, 4], "BF16")
        raw("per_layer_token_embd.weight", [WIDTH, ROWS], "F8_E4M3_RAW")
        for l in range(N_BLOCKS):
            g.add_f32(f"blk.{l}.attn_norm.weight", rng.standard_normal(8))
            if l == 0:
                raw("blk.0.ffn_gate.weight", [256, 2], "Q8_0", 16)
            else:
                g.add_f32(f"blk.{l}.ffn_gate_inp.weight", rng.standard_normal((8, N_EXPERT)))
            if l < N_BLOCKS - 1:
                raw(f"blk.{l}.attn_k_b.weight", [64, 2, 2], "Q8_0", 8)
            else:       # the next-n block is BF16
                raw(f"blk.{l}.attn_k_b.weight", [64, 2, 2], "BF16")
                raw(f"blk.{l}.nextn.eh_proj.weight", [16, 4], "BF16")
            for role in ("gate", "up", "down"):
                if l == 0:
                    continue
                ty = "BF16" if l == N_BLOCKS - 1 else DOWN[l] if role == "down" else "Q4_K"
                if ty == "BF16":
                    raw(f"blk.{l}.ffn_{role}_exps.weight", [256, 2, N_EXPERT], ty, 512 * N_EXPERT)
                else:
                    raw(f"blk.{l}.ffn_{role}_exps.weight", [256, 2, N_EXPERT], ty, 2 * N_EXPERT)
        self.gguf = g.write(self.t / "hy-00001-of-00001.gguf")
        self.hf = self.t / "hf"
        self.hf.mkdir()
        (self.hf / "LICENSE").write_text("Tencent Hy License, Version 1.0\n")
        rp = {"mrope_interleaved": True, "mrope_section": [12, 10, 10], "rope_type": "default"}
        (self.hf / "config.json").write_text(json.dumps({"model_type": "hy_v4_flash", "text_config": {
            "rope_parameters": {"sliding_attention": rp, "deepseek_sparse_attention": rp}}}))
        self._orig = convert.export_tokenizer

        def tokenizer_files(first, out_dir):
            d = Path(out_dir) / "tokenizer"
            d.mkdir(parents=True, exist_ok=True)
            (d / "vocab.json").write_text(json.dumps({"a": 0, "b": 1, "c": 2, "d": 3}))
            (d / "merges.txt").write_text("")
            (d / "token_type.json").write_text("[1,1,1,1]")
            (d / "tokenizer.json").write_text("{}")
            (d / "chat_template.jinja").write_text(TEMPLATE)
            return sorted(d.iterdir())
        convert.export_tokenizer = tokenizer_files

    def tearDown(self):
        convert.export_tokenizer = self._orig
        self.tmp.cleanup()

    def args(self, **kw):
        d = dict(gguf=[str(self.gguf)], mmproj=None, out=str(self.t / "out.basalt"), mtp_source=None, mtp_dense=None,
                 draft_vocab=None, name="Tiny", compat_bf16=False, no_vision=False, force=False, sections_from=None,
                 hf_dir=str(self.hf))
        d.update(kw)
        return type("A", (), d)()

    def build(self):
        convert.convert(self.args())
        return ModelFile(str(self.t / "out.basalt"))

    def test_sections_and_meta(self):
        m = self.build()
        self.assertEqual(set(m.names()), {
            "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/token_type.json", "tokenizer/tokenizer.json",
            "tokenizer/chat_template.jinja", "dense", "dense.index", "experts", "experts.layout", "gguf", "engram",
            "mtp/dense", "mtp/dense.index", "mtp/experts", "mtp/experts.layout", "meta"})
        for n in m.names():
            self.assertTrue(m.verify(n), n)
            self.assertEqual(m.section(n)[0] % SECTION_ALIGN, 0, n)
        meta = json.loads(m.read("meta"))
        self.assertEqual(meta["arch"], "hy4flash")
        self.assertEqual(meta["license"]["text"], "Tencent Hy License, Version 1.0\n")
        self.assertEqual(meta["rope"], {"mrope_section": [12, 10, 10], "interleaved": True})
        self.assertEqual(meta["geometry"]["expert_count"], N_EXPERT)
        self.assertNotIn("ple.layers", meta["geometry"])
        self.assertIsNone(meta["vision"])
        self.assertIsNone(meta["dfly2"])
        e = meta["engram"]
        self.assertEqual((e["type"], e["type_id"], e["rows"], e["row_bytes"], e["bytes"]), ("F8_E4M3_RAW", 43, ROWS, WIDTH, ROWS * WIDTH))
        self.assertEqual(e["ple.layers"], [1, 11, 21, 31])
        self.assertEqual(meta["mtp"]["layers"], [3])
        self.assertEqual(m.read("engram"), self.raw["per_layer_token_embd.weight"])

    def test_experts_layout_lists_the_moe_layers_only(self):
        m = self.build()
        lines = m.read("experts.layout").decode().splitlines()[1:]
        out, off = m.read("experts"), 0
        self.assertEqual(len(lines), 2)
        for line, l in zip(lines, (1, 2)):
            per = {r: BLOCK["Q6_K" if (r == "down" and l == 1) else "Q4_K"] * 2 for r in ("gate", "up", "down")}
            blob = sum(per.values())
            self.assertEqual(line, f"{l} 12 {TYPE_ID[DOWN[l]]} {off} {blob}")
            for e in range(N_EXPERT):
                want = b"".join(self.raw[f"blk.{l}.ffn_{r}_exps.weight"][e * per[r]:(e + 1) * per[r]] for r in ("gate", "up", "down"))
                self.assertEqual(out[off + e * blob:off + (e + 1) * blob], want, (l, e))
            off += blob * N_EXPERT
        self.assertEqual(off, len(out))

    def test_mtp_block_has_its_own_sections(self):
        m = self.build()
        idx = m.read("dense.index").decode()
        self.assertNotIn("blk.3.", idx)
        mi = m.read("mtp/dense.index").decode()
        self.assertIn("blk.3.nextn.eh_proj.weight", mi)
        self.assertNotIn("blk.1.", mi)
        lines = m.read("mtp/experts.layout").decode().splitlines()[1:]
        blob = sum(len(self.raw[f"blk.3.ffn_{r}_exps.weight"]) // N_EXPERT for r in ("gate", "up", "down"))
        self.assertEqual(lines, [f"3 30 30 0 {blob}"])
        self.assertEqual(len(m.read("mtp/experts")), blob * N_EXPERT)
        meta = json.loads(m.read("meta"))
        self.assertEqual(meta["mtp"]["dense_shapes"]["blk.3.attn_k_b.weight"], [64, 2, 2])

    def test_three_dimensional_native_tensor_flattens_in_the_index_only(self):
        m = self.build()
        meta = json.loads(m.read("meta"))
        t = meta["tensors"]["blk.1.attn_k_b.weight"]
        self.assertEqual(t["shape"], [64, 2, 2])
        gg = m.read("gguf")
        self.assertEqual(gg[t["offset"]:t["offset"] + t["bytes"]], self.raw["blk.1.attn_k_b.weight"])
        row = next(x for x in m.read("dense.index").decode().splitlines() if x.startswith("blk.1.attn_k_b.weight "))
        self.assertEqual(row.split()[7:9], ["64", "4"])

    def test_verify_agrees_and_catches_a_flipped_byte(self):
        m = self.build()
        pack = self.t / "out.basalt"
        self.assertTrue(verify_pack.verify(str(pack), str(self.gguf)))
        off, _ = m.section("experts")
        with open(pack, "r+b") as f:
            f.seek(off + 5)
            b = f.read(1)
            f.seek(off + 5)
            f.write(bytes([b[0] ^ 1]))
        self.assertFalse(verify_pack.verify(str(pack), str(self.gguf)))

    def test_hf_dir_is_needed_and_the_wrong_side_inputs_are_refused(self):
        with self.assertRaises(SystemExit):
            convert.convert(self.args(hf_dir=None))
        # --mmproj is the vision projector (vision/mmproj), accepted for this arch; a file that is not a projector GGUF is refused
        notgguf = self.t / "not-a-projector.gguf"
        notgguf.write_bytes(b"not a gguf")
        with self.assertRaises(SystemExit):
            convert.convert(self.args(mmproj=str(notgguf)))
        with self.assertRaises(SystemExit):
            convert.convert(self.args(draft_vocab=str(self.gguf)))

    def test_dry_run_writes_nothing(self):
        convert.convert(self.args(dry_run=True))
        self.assertFalse((self.t / "out.basalt").exists())
        self.assertFalse((self.t / "out.basalt.partial").exists())


if __name__ == "__main__":
    unittest.main()
