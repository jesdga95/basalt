"""A tiny synthetic BF16 hy_v4_flash GGUF converted by `--nvfp4`: the experts' NVFP4 blobs decoded against the source, the Q8_0 and
Q6_K projections, the head, the E4M3 Engram table and its error figures, the float tensors as stored, the sections and meta,
the tokenizer check, and `--verify` on the result (it must catch a flipped byte in every rounded section)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import convert  # noqa: E402
import ggufq  # noqa: E402
import hy4flash_bf16 as H  # noqa: E402
import model_sections as S  # noqa: E402
import verify_pack  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import SECTION_ALIGN, ModelFile  # noqa: E402

N_EXPERT, ROWS, COLS, WIDTH, TABLE_ROWS = 4, 8, 128, 96, 40
P = "hy_v4_flash."
TOKENS = ["a", "b", "c", "d"]
TEMPLATE = "{{ messages }}"
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)


def bf16_bytes(x: np.ndarray) -> bytes:
    """Round-to-nearest-even BF16 of float32 values."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype("<u2").tobytes()


def bf16_to_f32(b: bytes) -> np.ndarray:
    return S._bf16_to_f32(np.frombuffer(b, "<u2"))


def decode_nvfp4(blob: bytes, rows: int, cols: int, gs: float) -> np.ndarray:
    """An NVFP4 matrix by the format's definition: blocks of 64 values as 4 UE4M3 scales then 32 code bytes, sub-block s
    holding value j in the low nibble of byte 8 s + j and value j + 8 in the high one."""
    b = np.frombuffer(blob, np.uint8).reshape(rows, cols // 64, 36)
    sc = S.fp8_e4m3_table()[b[..., :4]].astype(np.float64) * gs
    codes = b[..., 4:].reshape(rows, cols // 64, 4, 8)
    v = np.empty((rows, cols // 64, 4, 16), np.float64)
    v[..., :8], v[..., 8:] = E2M1[codes & 15], E2M1[codes >> 4]
    return (v * sc[..., None]).reshape(rows, cols)


class Hy4FlashBf16Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        rng = np.random.default_rng(5)
        self.rng, self.src = rng, {}
        g = GGUFWriter()
        g.add("general.architecture", "hy_v4_flash")
        g.add("general.name", "Hy4-flash")
        g.add("general.file_type", 32, "u32")
        g.add(P + "block_count", 3, "u32")
        g.add(P + "embedding_length", 64, "u32")
        g.add(P + "vocab_size", 4, "u32")
        g.add(P + "expert_count", N_EXPERT, "u32")
        g.add(P + "attention.head_count", [2, 2, 4], "array:u32")
        g.add(P + "embedding_length_per_layer_input", WIDTH, "u32")
        g.add(P + "ple.layers", [1, 11, 21, 31], "array:u32")
        g.add(P + "ple.ngram_size", 2, "u32")
        g.add(P + "ple.heads_per_ngram", 2, "u32")
        g.add(P + "ple.layer_multipliers", [7, 11, 13, 17], "array:u64")
        g.add(P + "ple.head_offsets", [5 * i for i in range(8)], "array:u64")
        g.add(P + "ple.head_vocab_sizes", [5] * 8, "array:u64")
        g.add(P + "engram.pad_token_id", 2, "u32")
        g.add(P + "engram.token_map", [0, 1, 2, 3], "array:u32")
        g.add("tokenizer.ggml.tokens", TOKENS)
        g.add("tokenizer.ggml.merges", [], "array:string")
        g.add("tokenizer.ggml.bos_token_id", 0, "u32")
        g.add("tokenizer.ggml.eos_token_id", 1, "u32")
        g.add("tokenizer.ggml.padding_token_id", 2, "u32")
        g.add("tokenizer.chat_template", TEMPLATE)

        def bf16(name, shape, scale=0.05, f32_head=False):
            x = (rng.standard_normal(shape) * scale).astype(np.float32)
            data = bf16_bytes(x)
            self.src[name] = data
            if f32_head:
                data = (bf16_to_f32(data)).astype("<f4").tobytes()
                self.src[name] = data
                g.add_raw(name, list(reversed(shape)), "F32", data)
            else:
                g.add_raw(name, list(reversed(shape)), "BF16", data)

        def f32(name, shape):
            a = rng.standard_normal(shape).astype(np.float32)
            self.src[name] = a.tobytes()
            g.add_f32(name, a)

        bf16("token_embd.weight", (4, 64))
        bf16("output.weight", (4, 64), f32_head=True)
        f32("output_norm.weight", (64,))
        for l in range(3):
            f32(f"blk.{l}.attn_norm.weight", (64,))
            bf16(f"blk.{l}.attn_q_a.weight", (4, 64))             # 64 values a row: Q8_0 even with --dense-q6k
            bf16(f"blk.{l}.attn_output.weight", (2, 256))          # whole Q6_K blocks
            bf16(f"blk.{l}.attn_k_b.weight", (2, 2, 64))           # 3-D, not a Q6_K target
            if l == 0:
                bf16("blk.0.ffn_gate.weight", (3, 256))
            else:
                f32(f"blk.{l}.ffn_gate_inp.weight", (64, N_EXPERT))
                for r in H.ROLES:
                    bf16(f"blk.{l}.ffn_{r}_exps.weight", (N_EXPERT, ROWS, COLS))
        # table values: a spread of magnitudes, some exact E4M3 numbers, a tiny one and a zero
        tab = (rng.standard_normal((TABLE_ROWS, WIDTH)) * 1.2).astype(np.float32)
        tab[0, :4] = [0.5, -1.75, 0.0, 448.0]
        tab[1, 0] = 1e-4
        self.table = bf16_bytes(tab)
        self.src[H.ENGRAM_TENSOR] = self.table
        g.add_raw(H.ENGRAM_TENSOR, [WIDTH, TABLE_ROWS], "BF16", self.table)
        self.gguf = g.write(self.t / "hy-bf16.gguf")
        self.hf = self.t / "hf"
        self.hf.mkdir()
        (self.hf / "LICENSE").write_text("Tencent Hy License, Version 1.0\n")
        rp = {"mrope_interleaved": True, "mrope_section": [12, 10, 10], "rope_type": "default"}
        (self.hf / "config.json").write_text(json.dumps({"model_type": "hy_v4_flash", "text_config": {
            "rope_parameters": {"sliding_attention": rp, "deepseek_sparse_attention": rp}}, "dfly_num_hidden_layers": 2,
            "dfly_mask_token_id": 7, "dfly_target_kv_layers": [3, 5], "dfly_target_index_layer": [3, 5]}))
        (self.hf / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": {t: i for i, t in enumerate(TOKENS)},
                                                                      "merges": []}, "added_tokens": []}))
        (self.hf / "tokenizer_config.json").write_text(json.dumps({"bos_token": "a", "eos_token": "b", "pad_token": "c"}))
        (self.hf / "chat_template.jinja").write_text(TEMPLATE)
        self._orig = convert.export_tokenizer

        def tokenizer_files(first, out_dir):
            d = Path(out_dir) / "tokenizer"
            d.mkdir(parents=True, exist_ok=True)
            (d / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(TOKENS)}))
            (d / "merges.txt").write_text("")
            (d / "token_type.json").write_text("[1,1,1,1]")
            (d / "tokenizer.json").write_text("{}")
            (d / "chat_template.jinja").write_text(TEMPLATE)
            return sorted(d.iterdir())
        convert.export_tokenizer = tokenizer_files
        self.mmproj = self.t / "mmproj.gguf"
        self.mmproj.write_bytes(b"GGUF" + b"\0" * 20 + b"hy_v4_flash" + b"\0" * 100)
        self.dfly = self.t / "dfly.safetensors"
        import dfly_attach
        names = [dfly_attach.PREFIX + n for n in sorted(dfly_attach.wanted(2))]
        head = {n: {"dtype": "BF16", "shape": [2], "data_offsets": [4 * i, 4 * i + 4]} for i, n in enumerate(names)}
        hj = json.dumps(head).encode()
        self.dfly.write_bytes(len(hj).to_bytes(8, "little") + hj + b"\1\2\3\4" * len(names))

    def tearDown(self):
        convert.export_tokenizer = self._orig
        self.tmp.cleanup()

    def args(self, **kw):
        d = dict(gguf=[str(self.gguf)], mmproj=str(self.mmproj), out=str(self.t / "out.basalt"), name=None, no_vision=False,
                 force=False, hf_dir=str(self.hf), dfly=str(self.dfly), dense_q6k=False, workers=2, dry_run=False)
        d.update(kw)
        return type("A", (), d)()

    def build(self, **kw):
        convert.convert_bf16(self.args(**kw))
        return ModelFile(str(self.t / "out.basalt"))

    def tensor(self, m, meta, name):
        e = meta["tensors"][name]
        return m.read("gguf")[e["offset"]:e["offset"] + e["bytes"]], e

    def test_sections_and_meta(self):
        m = self.build()
        self.assertEqual(set(m.names()), {
            "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/token_type.json", "tokenizer/tokenizer.json",
            "tokenizer/chat_template.jinja", "dense", "dense.index", "experts", "experts.layout", "gguf", "engram",
            "vision/mmproj", "dfly", "meta"})
        for n in m.names():
            self.assertTrue(m.verify(n), n)
            self.assertEqual(m.section(n)[0] % SECTION_ALIGN, 0, n)
        meta = json.loads(m.read("meta"))
        self.assertEqual((meta["arch"], meta["gguf_kv"]["general.architecture"], meta["name"]), ("hy4flash",) * 2 + ("Hy4-flash NVFP4",))
        self.assertNotIn("general.file_type", meta["gguf_kv"])
        self.assertEqual(meta["vision"]["mmproj"], "vision/mmproj")
        self.assertEqual(meta["dfly2"]["layers"], 2)
        self.assertEqual(m.read("dfly"), self.dfly.read_bytes())
        self.assertIsNone(meta["mtp"])
        self.assertEqual(meta["geometry"]["expert_count"], N_EXPERT)
        self.assertEqual(meta["rope"], {"mrope_section": [12, 10, 10], "interleaved": True})
        e = meta["engram"]
        self.assertEqual((e["type"], e["type_id"], e["rows"], e["row_bytes"], e["bytes"]), ("F8_E4M3_RAW", 43, TABLE_ROWS, WIDTH, TABLE_ROWS * WIDTH))
        self.assertEqual(e["token_map"], "gguf_kv:hy_v4_flash.engram.token_map")
        self.assertEqual(meta["quantized_from"]["dense_variant"], "Q8_0")

    def test_experts_are_nvfp4_blobs_of_the_sources_values(self):
        m = self.build()
        lines = m.read("experts.layout").decode().splitlines()
        blob = 3 * ROWS * (COLS // 64 * 36) + 16
        self.assertEqual(lines[1:], [f"1 40 40 0 {blob}", f"2 40 40 {N_EXPERT * blob} {blob}"])
        self.assertIn(f"(n_expert {N_EXPERT}, total {2 * N_EXPERT * blob})", lines[0])
        out = m.read("experts")
        self.assertEqual(len(out), 2 * N_EXPERT * blob)
        worst = 0.0
        for i, l in enumerate((1, 2)):
            for e in range(N_EXPERT):
                b = out[(i * N_EXPERT + e) * blob:(i * N_EXPERT + e + 1) * blob]
                tail = np.frombuffer(b[-16:], "<f4")
                self.assertEqual(tail[3], 0)
                for k, r in enumerate(H.ROLES):
                    x = bf16_to_f32(self.src[f"blk.{l}.ffn_{r}_exps.weight"])[e * ROWS * COLS:(e + 1) * ROWS * COLS].reshape(ROWS, COLS)
                    # the tensor scale is the matrix's amax / (6 x 448), in float32
                    self.assertEqual(tail[k], np.float32(np.abs(x).max()) / np.float32(6 * 448))
                    mat = decode_nvfp4(b[k * ROWS * 72:(k + 1) * ROWS * 72], ROWS, COLS, float(tail[k]))
                    worst = max(worst, np.sqrt(((mat - x) ** 2).sum() / (x ** 2).sum()))
        self.assertLess(worst, 0.15)          # NVFP4 of a Gaussian by RTN: about 0.09 relative RMS
        self.assertGreater(worst, 0.02)

    def test_projections_head_and_floats(self):
        m = self.build()
        meta = json.loads(m.read("meta"))
        for name, shape in (("token_embd.weight", [64, 4]), ("blk.1.attn_k_b.weight", [64, 2, 2]), ("blk.0.ffn_gate.weight", [256, 3])):
            raw, e = self.tensor(m, meta, name)
            self.assertEqual((e["type"], e["shape"], e["offset"] % 4096), ("Q8_0", shape, 0), name)
            x = bf16_to_f32(self.src[name])
            blocks = np.frombuffer(raw, np.uint8).reshape(-1, 34)
            d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
            got = d * blocks[:, 2:].view(np.int8)
            self.assertTrue((np.abs(got - x.reshape(-1, 32)) <= d * 0.6 + 1e-7).all(), name)
        raw, e = self.tensor(m, meta, "output.weight")
        self.assertEqual((e["type"], e["type_id"], e["bytes"]), ("BF16", 30, 4 * 64 * 2))
        self.assertTrue(np.array_equal(bf16_to_f32(raw), np.frombuffer(self.src["output.weight"], "<f4")))
        for name in ("output_norm.weight", "blk.1.ffn_gate_inp.weight"):
            raw, e = self.tensor(m, meta, name)
            self.assertEqual((e["type"], raw), ("F32", self.src[name]))
        index = {x.split()[0]: x.split() for x in m.read("dense.index").decode().splitlines() if not x.startswith("#")}
        row = index["blk.1.attn_k_b.weight"]      # a native row: no bytes in the pool, the type's code bits and group
        self.assertEqual(row[1:5] + row[6:], ["0", "0", "0", "0", "0", "64", "4", "8", "0", "32"] + ["0"] * 7)
        self.assertEqual(index["output.weight"][2], "4")      # BF16 in the pool
        self.assertEqual(index["blk.1.ffn_gate_inp.weight"][2], "2")
        pool, off = m.read("dense"), int(index["output.weight"][5])
        self.assertEqual(pool[off:off + 512], bf16_bytes(np.frombuffer(self.src["output.weight"], "<f4")))
        self.assertNotIn("blk.1.ffn_gate_exps.weight", meta["tensors"])
        self.assertEqual(set(meta["tensors"]) - {H.ENGRAM_TENSOR}, set(index))

    def test_dense_q6k_variant(self):
        m = self.build(dense_q6k=True)
        meta = json.loads(m.read("meta"))
        self.assertEqual(meta["quantized_from"]["dense_variant"], "Q6_K")
        self.assertEqual(meta["name"], "Hy4-flash NVFP4 Q6K")
        types = {n: e["type"] for n, e in meta["tensors"].items()}
        self.assertEqual(types["blk.1.attn_output.weight"], "Q6_K")
        self.assertEqual(types["blk.0.ffn_gate.weight"], "Q6_K")
        for n in ("blk.1.attn_q_a.weight", "blk.1.attn_k_b.weight", "token_embd.weight"):
            self.assertEqual(types[n], "Q8_0", n)
        raw, e = self.tensor(m, meta, "blk.1.attn_output.weight")
        self.assertEqual(e["bytes"], 2 * 210)
        got = ggufq.dequantize(np.frombuffer(raw, np.uint8), ggufq.GGMLQuantizationType.Q6_K).astype(np.float32)
        x = bf16_to_f32(self.src["blk.1.attn_output.weight"])
        self.assertLess(np.sqrt(((got - x) ** 2).sum() / (x ** 2).sum()), 0.03)
        self.assertTrue(verify_pack.verify(str(self.t / "out.basalt"), str(self.gguf), 2))

    def test_engram_table_is_e4m3_rounded_once_and_the_error_is_reported(self):
        m = self.build()
        meta = json.loads(m.read("meta"))
        codes = np.frombuffer(m.read("engram"), np.uint8)
        self.assertEqual(len(codes), TABLE_ROWS * WIDTH)
        x = bf16_to_f32(self.table).astype(np.float64)
        dec = S.fp8_e4m3_table().astype(np.float64)[codes]
        self.assertEqual(list(dec[:4]), [0.5, -1.75, 0.0, 448.0])         # E4M3 numbers stay as they are
        self.assertEqual(dec[WIDTH], 0.0)                                  # 1e-4 is below half the least subnormal
        big = np.abs(x) >= 2.0 ** -6
        self.assertLessEqual((np.abs(dec - x)[big] / np.abs(x)[big]).max(), 2.0 ** -4 + 1e-9)
        q = meta["quantized_from"]["engram"]
        self.assertEqual(q["values"], TABLE_ROWS * WIDTH)
        self.assertAlmostEqual(q["rms_relative_error"], float(np.sqrt(((dec - x) ** 2).sum() / (x ** 2).sum())), places=12)
        self.assertAlmostEqual(q["exact_fraction"], float((dec == x).mean()), places=12)
        self.assertEqual(q["nonzero_rounded_to_zero"], int(((dec == 0) & (x != 0)).sum()))
        self.assertLess(q["exact_fraction"], 0.2)

    def test_a_table_that_is_exactly_e4m3_has_no_error(self):
        v = S.fp8_e4m3_table()[:127].astype(np.float32)
        data = bf16_bytes(np.resize(v, TABLE_ROWS * WIDTH))
        e = H.engram_error(np.bincount(np.frombuffer(data, "<u2"), minlength=65536))
        self.assertEqual((e["exact_fraction"], e["rms_relative_error"], e["max_abs_error"]), (1.0, 0.0, 0.0))

    def test_verify_agrees_and_catches_a_flipped_byte_in_each_rounded_section(self):
        m = self.build()
        pack = self.t / "out.basalt"
        self.assertTrue(verify_pack.verify(str(pack), str(self.gguf), 2))
        meta = json.loads(m.read("meta"))
        for section, at in (("experts", 5), ("engram", 7), ("dense", 3), ("gguf", meta["tensors"]["blk.1.attn_q_a.weight"]["offset"] + 9),
                            ("gguf", meta["tensors"]["blk.1.attn_norm.weight"]["offset"] + 2)):
            off, _ = m.section(section)
            with open(pack, "r+b") as f:
                f.seek(off + at)
                b = f.read(1)
                f.seek(off + at)
                f.write(bytes([b[0] ^ 1]))
            self.assertFalse(verify_pack.verify(str(pack), str(self.gguf), 2), (section, at))
            with open(pack, "r+b") as f:
                f.seek(off + at)
                f.write(b)
        self.assertTrue(verify_pack.verify(str(pack), str(self.gguf), 2))

    def test_refusals(self):
        with self.assertRaises(SystemExit):
            convert.convert_bf16(self.args(hf_dir=str(self.t / "nowhere")))
        (self.hf / "chat_template.jinja").write_text("other")
        with self.assertRaises(SystemExit):
            convert.convert_bf16(self.args())
        (self.hf / "chat_template.jinja").write_text(TEMPLATE)
        (self.hf / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1, "c": 3, "d": 2}, "merges": []}, "added_tokens": []}))
        with self.assertRaises(SystemExit):
            convert.convert_bf16(self.args())
        self.assertFalse((self.t / "out.basalt").exists())
        self.assertFalse((self.t / "out.basalt.partial").exists())

    def test_dry_run_writes_nothing_and_sizes_match_a_real_build(self):
        convert.convert_bf16(self.args(dry_run=True))
        self.assertFalse((self.t / "out.basalt").exists())
        self.assertFalse((self.t / "out.basalt.partial").exists())
        m = self.build()
        self.assertEqual(m.section("experts")[1], 2 * N_EXPERT * (3 * ROWS * 72 + 16))


if __name__ == "__main__":
    unittest.main()
