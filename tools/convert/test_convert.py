"""Tests for convert.py and model_sections.py on a tiny synthetic model, drafter and side files.

    python3 -m unittest discover -s tools/convert -p 'test_convert.py'"""
import contextlib
import hashlib
import io
import json
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert  # noqa: E402
import ggufq  # noqa: E402
import hc8 as H  # noqa: E402
import model_sections as S  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import ModelFile, ModelWriter  # noqa: E402

N_EXPERT, N_LAYERS = 4, 2
TOKENS = ["a", "b", "c", "d", "e", "f"]


def small_ints(rng, shape):
    """Float32 values that are exactly BF16."""
    return rng.integers(-8, 9, size=shape).astype(np.float32)


class ConvertTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.t = t
        rng = np.random.default_rng(0)
        g = GGUFWriter()
        g.add("general.architecture", "qwen")
        g.add("general.quantized_by", "Someone")
        g.add("qwen.block_count", N_LAYERS, "u32")
        g.add("qwen.rope.freqs", [0.5, 1.5], "array:f32")
        g.add("qwen.nan", float("inf"), "f32")
        g.add("tokenizer.ggml.tokens", TOKENS)
        self.router = small_ints(rng, (8, N_EXPERT))
        self.attn = rng.standard_normal((3, 8)).astype(np.float32)
        self.embd = rng.standard_normal((5, 8)).astype(np.float32)
        self.conv = rng.standard_normal((2, 4)).astype(np.float32)
        self.ple = rng.standard_normal((2, 6)).astype(np.float32)
        self.q2 = rng.standard_normal((2, 64)).astype(np.float32)
        self.hc_down = small_ints(rng, (64, 2)) / 4         # GGUF order: ne0 (columns) first
        self.hc_up = small_ints(rng, (32, 64)) / 8
        self.experts = {}
        g.add_f32("token_embd.weight", self.embd)
        g.add_f32("per_layer_token_embd.weight", self.ple)
        for l in range(N_LAYERS):
            g.add_f32(f"blk.{l}.ffn_gate_inp.weight", self.router)
            g.add_f32(f"blk.{l}.attn_q.weight", self.attn)
            for role, shape in (("gate", (3, 2, N_EXPERT)), ("up", (3, 2, N_EXPERT)), ("down", (2, 3, N_EXPERT))):
                a = rng.standard_normal(shape).astype(np.float32)
                self.experts[(l, role)] = a
                g.add_f32(f"blk.{l}.ffn_{role}_exps.weight", a)
        g.add_f32("blk.0.ple_conv1d.weight", self.conv)
        g.add_f32("blk.1.hc_attn_down.weight", self.hc_down)
        g.add_f32("output_hc_up.weight", self.hc_up)
        g.add_q2_0("blk.0.attn_gate.weight", self.q2, [64, 2])
        self.gguf = g.write(t / "m-00001-of-00001.gguf")
        self.tok = {"tokenizer.json": b"{}", "vocab.json": b'{"a":0}', "merges.txt": b"a b\n",
                    "token_type.json": b"[1]", "chat_template.jinja": b"{{ x }}"}
        self.mmproj = rng.bytes(4321)
        (t / "mmproj.gguf").write_bytes(self.mmproj)
        self.vision = None
        self._orig = (convert.load_vision_helper, convert.export_tokenizer)
        convert.load_vision_helper = lambda: self.vision

        def tokenizer_files(first, out_dir):
            d = Path(out_dir) / "tokenizer"
            d.mkdir(parents=True, exist_ok=True)
            for k, v in self.tok.items():
                (d / k).write_bytes(v)
            return sorted(d.iterdir())
        convert.export_tokenizer = tokenizer_files

    def tearDown(self):
        convert.load_vision_helper, convert.export_tokenizer = self._orig
        self.tmp.cleanup()

    def args(self, **kw):
        t = self.t
        d = dict(gguf=[str(self.gguf)], mmproj=str(t / "mmproj.gguf"), out=str(t / "out.basalt"), mtp_source=None, mtp_dense=None,
                 draft_vocab=None, name="Tiny",
                 compat_bf16=False, no_vision=False, force=False, sections_from=None)
        d.update(kw)
        return type("A", (), d)()

    def model(self):
        return ModelFile(str(self.t / "out.basalt"))

    def planned(self, **kw):
        """Section name -> size, from the listing a --dry-run prints."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            convert.convert(self.args(dry_run=True, **kw))
        return {m.group(1): int(m.group(3)) for m in re.finditer(r"^ {9}(\S+) +(\d+) +(\d+)$", out.getvalue(), re.M)}

    # ---------------------------------------------------------------- sections
    def test_sections(self):
        convert.convert(self.args())
        m = self.model()
        self.assertFalse((self.t / "out.basalt.partial").exists())
        want = {"hc8", "vision/mmproj", "dense", "dense.index", "experts", "experts.layout", "gguf", "meta"}
        want |= {"tokenizer/" + k for k in self.tok}
        self.assertEqual(set(m.names()), want)
        for k in m.names():
            self.assertTrue(m.verify(k), k)
        with open(m.path, "rb") as f:
            self.assertEqual(f.read(12)[8:], struct.pack("<I", 2))
        self.assertEqual(m.read("vision/mmproj"), self.mmproj)
        for k, v in self.tok.items():
            self.assertEqual(m.read("tokenizer/" + k), v)

    def test_experts_and_layout(self):
        convert.convert(self.args())
        m = self.model()
        blobs = b""
        for l in range(N_LAYERS):
            parts = [self.experts[(l, r)].reshape(N_EXPERT, -1) for r in ("gate", "up", "down")]
            blobs += np.concatenate(parts, axis=1).tobytes()
        self.assertEqual(m.read("experts"), blobs)
        blob = (24 * 3) // N_EXPERT * 4              # bytes per expert: gate + up + down, F32
        lines = m.read("experts.layout").decode().splitlines()
        self.assertEqual(lines[0], f"# experts v5: layer gu_type d_type offset blob_bytes "
                                   f"(n_expert {N_EXPERT}, total {2 * blob * N_EXPERT})")
        self.assertEqual(lines[1:], [f"0 0 0 0 {blob}", f"1 0 0 {blob * N_EXPERT} {blob}"])

    def test_dense(self):
        convert.convert(self.args())
        m = self.model()
        rows = [l.split() for l in m.read("dense.index").decode().splitlines()]
        self.assertEqual(rows[0], ["#", "dense", "index", "v3"])
        by = {r[0]: r for r in rows[2:]}
        pool = m.read("dense")
        # the router is BF16 in the pool (exact narrowing); a plain float tensor is stored as it is
        r = by["blk.0.ffn_gate_inp.weight"]
        self.assertEqual(r[2], "4")
        off, n = int(r[3]), int(r[4])
        self.assertEqual(pool[off:off + n], (self.router.view(np.uint32) >> 16).astype("<u2").tobytes())
        r = by["blk.0.attn_q.weight"]
        self.assertEqual(r[2], "2")
        self.assertEqual(pool[int(r[3]):int(r[3]) + int(r[4])], self.attn.astype("<f4").tobytes())
        # F32 PLE conv: kind 3 (narrowed by the loader), 2 bytes an element in the arena
        r = by["blk.0.ple_conv1d.weight"]
        self.assertEqual((r[2], int(r[4]), int(r[6])), ("3", self.conv.nbytes, self.conv.nbytes // 2))
        # a quantized tensor is only listed; the PLE table and the experts are not listed at all
        self.assertEqual(by["blk.0.attn_gate.weight"][4], "0")
        self.assertNotIn("per_layer_token_embd.weight", by)
        self.assertFalse(any("_exps" in k for k in by))
        self.assertEqual(rows[1][:2], ["#", "align"])
        arena = [int(r[5]) for r in rows[2:]]
        self.assertEqual(arena, sorted(arena))
        self.assertTrue(all(a % 64 == 0 for a in arena))
        self.assertEqual(int(rows[1][4]), (arena[-1] + (int(rows[-1][6]) + 63) // 64 * 64))

    def test_gguf_section_and_meta(self):
        convert.convert(self.args())
        m = self.model()
        meta = json.loads(m.read("meta"))
        self.assertEqual(meta["format"], "basalt")
        self.assertEqual(meta["name"], "Tiny")
        self.assertEqual(meta["gguf_kv"]["qwen.block_count"], N_LAYERS)
        self.assertEqual(meta["gguf_kv"]["qwen.rope.freqs"], [0.5, 1.5])
        self.assertEqual(meta["gguf_kv"]["qwen.nan"], "inf")
        tens = meta["tensors"]
        self.assertFalse(any("_exps" in k for k in tens))
        self.assertIn("per_layer_token_embd.weight", tens)
        self.assertEqual(tens["token_embd.weight"]["offset"], 0)
        gg = m.read("gguf")
        for name, arr in (("token_embd.weight", self.embd), ("per_layer_token_embd.weight", self.ple)):
            o = tens[name]["offset"]
            self.assertEqual(gg[o:o + arr.nbytes], arr.astype("<f4").tobytes(), name)
            self.assertEqual(o % 4096, 0)
        self.assertEqual(tens["blk.0.attn_gate.weight"]["type"], "Q2_0")
        self.assertEqual(meta["vision"]["mmproj"], "vision/mmproj")
        self.assertEqual(meta["sources"][0]["size"], self.gguf.stat().st_size)
        self.assertEqual(meta["version"], 2)
        for k in ("profile", "hc8", "sampling", "engine_args"):
            self.assertNotIn(k, meta)
        self.assertEqual(meta["converter"], {"tool": "tools/convert/convert.py"})
        self.assertEqual(meta["gguf_kv"]["general.sampling.top_k"], 20)
        self.assertIsNone(meta["mtp"])
        lic = meta["license"]
        self.assertEqual(lic["name"], "Qwen Community License 1.0")
        self.assertEqual(lic["text"], (Path(convert.__file__).parent / "qwen-community-license-1.0.txt").read_text(encoding="utf-8"))
        self.assertEqual(lic["quantized_by"], "Someone")
        self.assertEqual(set(lic), {"name", "text", "quantized_by"})

    def test_without_drafter(self):
        convert.convert(self.args())
        m = self.model()
        self.assertFalse([n for n in m.names() if n.startswith("mtp/")])

    def test_hc8_from_the_pack_matrices(self):
        convert.convert(self.args())
        t = H.read_table(self.model().read("hc8"))
        self.assertEqual(list(t), ["blk.1.hc_attn_down.weight", "output_hc_up.weight"])
        for name, a in (("blk.1.hc_attn_down.weight", self.hc_down), ("output_hc_up.weight", self.hc_up)):
            rows, cols = a.shape[1], a.shape[0]
            u16 = (a.view(np.uint32) >> 16).astype(np.uint16).reshape(rows, cols)
            self.assertEqual(t[name][:2], (rows, cols))
            self.assertEqual(t[name][3], H.quantize(u16), name)

    def test_a_dry_run_plans_the_sections_a_build_writes(self):
        planned = self.planned()
        self.assertFalse((self.t / "out.basalt").exists() or (self.t / "out.basalt.partial").exists())
        convert.convert(self.args())
        m = self.model()
        self.assertEqual(planned, {k: m.sections[k][1] for k in m.names()})

    # ---------------------------------------------------------------- refusals and options
    def test_no_vision_and_overwrite(self):
        convert.convert(self.args(no_vision=True))
        self.assertNotIn("vision/mmproj", self.model().names())
        with self.assertRaises(SystemExit):
            convert.convert(self.args())
        convert.convert(self.args(force=True))
        self.assertIn("vision/mmproj", self.model().names())

    def test_vision_sections(self):
        src = self.t / "extra.bin"
        src.write_bytes(b"0123456789")
        self.vision = lambda mm, nat: [("vision/a", b"inline", None, 0, 6), ("vision/b", None, str(src), 2, 5)]
        convert.convert(self.args())
        m = self.model()
        self.assertEqual(m.read("vision/a"), b"inline")
        self.assertEqual(m.read("vision/b"), b"23456")
        self.assertEqual(json.loads(m.read("meta"))["vision"]["extra"], ["vision/a", "vision/b"])

    def test_router_not_bf16_is_refused(self):
        g = GGUFWriter()
        g.add("general.architecture", "qwen")
        g.add_f32("blk.0.ffn_gate_inp.weight", np.full((8, N_EXPERT), 0.10001, dtype=np.float32))
        for role, shape in (("gate", (3, 2, N_EXPERT)), ("up", (3, 2, N_EXPERT)), ("down", (2, 3, N_EXPERT))):
            g.add_f32(f"blk.0.ffn_{role}_exps.weight", np.zeros(shape, dtype=np.float32))
        path = g.write(self.t / "bad.gguf")
        with self.assertRaises(SystemExit) as cm:
            convert.convert(self.args(gguf=[str(path)]))
        self.assertIn("requires BF16", str(cm.exception))
        self.assertFalse((self.t / "out.basalt").exists())
        self.assertFalse((self.t / "out.basalt.partial").exists())

    def test_draft_vocab_needs_the_drafter(self):
        (self.t / "dv.bin").write_bytes(np.array([0, 1], dtype="<i4").tobytes())
        with self.assertRaises(SystemExit):
            convert.convert(self.args(draft_vocab=str(self.t / "dv.bin")))

    # ---------------------------------------------------------------- per-layer expert formats
    def mixed_gguf(self, gu_types):
        """A GGUF whose layer l has gate/up of gu_types[l] (K-quant blocks, random bytes) and Q8_0 down."""
        rng = np.random.default_rng(7)
        blk = {"Q5_K": 176, "Q6_K": 210}
        g = GGUFWriter()
        g.add("general.architecture", "qwen")
        g.add("qwen.block_count", len(gu_types), "u32")
        g.add("tokenizer.ggml.tokens", TOKENS)
        g.add_f32("token_embd.weight", self.embd)
        g.add_f32("per_layer_token_embd.weight", self.ple)
        raw = {}
        for l, gt in enumerate(gu_types):
            g.add_f32(f"blk.{l}.ffn_gate_inp.weight", self.router)
            for role, t, cols, nb in (("gate", gt, 256, blk[gt]), ("up", gt, 256, blk[gt]), ("down", "Q8_0", 32, 34)):
                data = rng.bytes(nb * 2 * N_EXPERT)
                raw[(l, role)] = data
                g.add_raw(f"blk.{l}.ffn_{role}_exps.weight", [cols, 2, N_EXPERT], t, data)
        return g.write(self.t / "mixed-00001-of-00001.gguf"), raw

    def test_mixed_k_quant_layers_are_copied_byte_for_byte(self):
        types = ["Q5_K", "Q6_K", "Q5_K"]
        path, raw = self.mixed_gguf(types)
        convert.convert(self.args(gguf=[str(path)]))
        m = self.model()
        out, off = m.read("experts"), 0
        lines = m.read("experts.layout").decode().splitlines()[1:]
        sizes = {"Q5_K": 176, "Q6_K": 210}
        for l, gt in enumerate(types):
            blob = 2 * (2 * sizes[gt]) + 2 * 34
            self.assertEqual(lines[l], f"{l} {13 if gt == 'Q5_K' else 14} 8 {off} {blob}")
            for e in range(N_EXPERT):
                want = b"".join(raw[(l, r)][e * (len(raw[(l, r)]) // N_EXPERT):(e + 1) * (len(raw[(l, r)]) // N_EXPERT)]
                                for r in ("gate", "up", "down"))
                self.assertEqual(out[off + e * blob:off + (e + 1) * blob], want, (l, e))
            off += blob * N_EXPERT
        self.assertEqual(off, len(out))

    def test_gate_up_of_two_types_write_a_v7_layout(self):
        g = GGUFWriter()
        rng = np.random.default_rng(1)
        g.add("general.architecture", "qwen")
        g.add("qwen.block_count", 2, "u32")
        g.add("tokenizer.ggml.tokens", TOKENS)
        g.add_f32("token_embd.weight", self.embd)
        g.add_f32("per_layer_token_embd.weight", self.ple)
        raw = {}
        for l, (gt, ut) in enumerate((("Q5_K", "Q5_K"), ("Q5_K", "Q6_K"))):
            g.add_f32(f"blk.{l}.ffn_gate_inp.weight", self.router)
            for role, t, cols, nb in (("gate", gt, 256, 176), ("up", ut, 256, 176 if ut == "Q5_K" else 210),
                                      ("down", "Q8_0", 32, 34)):
                raw[(l, role)] = rng.bytes(nb * 2 * N_EXPERT)
                g.add_raw(f"blk.{l}.ffn_{role}_exps.weight", [cols, 2, N_EXPERT], t, raw[(l, role)])
        path = g.write(self.t / "mix-00001-of-00001.gguf")
        convert.convert(self.args(gguf=[str(path)]))
        m = self.model()
        b0, b1 = 2 * (176 + 176 + 34), 2 * (176 + 210 + 34)
        lines = m.read("experts.layout").decode().splitlines()
        self.assertEqual(lines[0], f"# experts v7: layer gu_type d_type offset blob_bytes up_type "
                                   f"(n_expert {N_EXPERT}, total {(b0 + b1) * N_EXPERT})")
        self.assertEqual(lines[1:], [f"0 13 8 0 {b0} 13", f"1 13 8 {b0 * N_EXPERT} {b1} 14"])
        out = m.read("experts")
        e = 1
        want = b"".join(raw[(1, r)][e * (len(raw[(1, r)]) // N_EXPERT):(e + 1) * (len(raw[(1, r)]) // N_EXPERT)]
                        for r in ("gate", "up", "down"))
        self.assertEqual(out[b0 * N_EXPERT + e * b1:b0 * N_EXPERT + (e + 1) * b1], want)

    def test_sections_from_with_gguf(self):
        import pack_vision
        self.vision = pack_vision.vision_sections
        convert.convert(self.args(out=str(self.t / "base.basalt")))
        base = ModelFile(str(self.t / "base.basalt"))
        path, _ = self.mixed_gguf(["Q6_K", "Q5_K"])
        convert.convert(self.args(gguf=[str(path)], out=str(self.t / "new.basalt"), sections_from=str(self.t / "base.basalt"),
                                  mmproj=None, name="Mixed"))
        m = ModelFile(str(self.t / "new.basalt"))
        for n in m.names():
            self.assertTrue(m.verify(n), n)
        self.assertEqual(m.read("vision/mmproj"), base.read("vision/mmproj"))
        self.assertNotIn("profile", m.names())
        self.assertEqual(H.read_table(m.read("hc8")), {})        # the mixed GGUF has no hyper-connection matrices
        meta = json.loads(m.read("meta"))
        self.assertEqual(meta["sections_from"]["copied"], [n for n in base.names() if n != "vision/text_vocab"
                                                           and n.startswith(("mtp/", "vision/"))])
        self.assertEqual(m.read("vision/text_vocab"), pack_vision.vocab_only_gguf(str(path)))
        self.assertEqual(meta["vision"], json.loads(base.read("meta"))["vision"])
        self.assertEqual(m.read("experts.layout").decode().splitlines()[1].split()[:3], ["0", "14", "8"])

    def test_sections_from_dry_run_matches_real_build(self):
        import pack_vision
        self.vision = pack_vision.vision_sections
        convert.convert(self.args(out=str(self.t / "base.basalt")))
        path, _ = self.mixed_gguf(["Q6_K", "Q5_K"])
        planned = self.planned(gguf=[str(path)], out=str(self.t / "new.basalt"),
                               sections_from=str(self.t / "base.basalt"), mmproj=None, name="Mixed")
        self.assertFalse((self.t / "new.basalt").exists())
        convert.convert(self.args(gguf=[str(path)], out=str(self.t / "new.basalt"),
                                  sections_from=str(self.t / "base.basalt"), mmproj=None, name="Mixed"))
        m = ModelFile(str(self.t / "new.basalt"))
        self.assertEqual(planned, {k: m.sections[k][1] for k in m.names()})

    def test_verify_pack_handles_converted_dense_v7_up_type_and_multi_shard(self):
        import verify_pack
        # 1) Converted dense tensors (F32 -> BF16 router/hc + kind-3 ple_conv1d) must verify cleanly.
        self.tok["vocab.json"] = json.dumps({t: i for i, t in enumerate(TOKENS)}).encode()
        convert.convert(self.args())
        # Populate chat_template in gguf_kv match: self.gguf has no tokenizer.chat_template, so check our verify_pack
        # on a pack whose gguf has matching chat_template.
        from gguf_reader import GGUFFile
        g0 = GGUFFile(self.gguf)
        raw = self.gguf.read_bytes()
        gw = GGUFWriter()
        for k, v in g0.metadata.items():
            if isinstance(v, int) and not isinstance(v, bool):
                gw.add(k, v, "u32")
            elif isinstance(v, float):
                gw.add(k, v, "f32")
            else:
                gw.add(k, v)
        gw.add("tokenizer.chat_template", "{{ x }}")
        for t in g0.tensors:
            nb = t.expected_bytes()
            gw.add_raw(t.name, t.shape, t.type_name, raw[g0.data_start + t.offset:g0.data_start + t.offset + nb])
        g_tpl = gw.write(self.t / "with-tpl.gguf")
        out_tpl = self.t / "with-tpl.basalt"
        convert.convert(self.args(gguf=[str(g_tpl)], out=str(out_tpl)))
        self.assertTrue(verify_pack.verify(str(out_tpl), str(g_tpl), workers=1))

        # 2) v7 experts.layout: valid v7 passes, corrupted up_type fails, and multi-shard --gguf list passes.
        rng = np.random.default_rng(11)
        g1, g2 = GGUFWriter(), GGUFWriter()
        g1.add("general.architecture", "qwen")
        g1.add("qwen.block_count", 2, "u32")
        g1.add("tokenizer.ggml.tokens", TOKENS)
        g1.add("tokenizer.chat_template", "{{ x }}")
        for i, g in enumerate((g1, g2)):
            g.add("split.no", i, "u32")
            g.add("split.count", 2, "u32")
        g1.add_f32("token_embd.weight", self.embd)
        g1.add_f32("per_layer_token_embd.weight", self.ple)
        for l, (gt, ut) in enumerate((("Q5_K", "Q5_K"), ("Q5_K", "Q6_K"))):
            dst = g1 if l == 0 else g2
            dst.add_f32(f"blk.{l}.ffn_gate_inp.weight", self.router)
            for role, t, cols, nb in (("gate", gt, 256, 176), ("up", ut, 256, 176 if ut == "Q5_K" else 210),
                                      ("down", "Q8_0", 32, 34)):
                dst.add_raw(f"blk.{l}.ffn_{role}_exps.weight", [cols, 2, N_EXPERT], t, rng.bytes(nb * 2 * N_EXPERT))
        total_t = len(g1.tensors) + len(g2.tensors)
        g1.add("split.tensors.count", total_t, "u32")
        g2.add("split.tensors.count", total_t, "u32")
        s1 = g1.write(self.t / "part_a.gguf")
        s2 = g2.write(self.t / "part_b.gguf")
        out_v7 = self.t / "v7.basalt"
        convert.convert(self.args(gguf=[str(s1), str(s2)], out=str(out_v7)))
        self.assertTrue(verify_pack.verify(str(out_v7), [str(s1), str(s2)], workers=1))
        # CLI --verify with repeated --gguf must pass all shards, not drop a.gguf[1:].
        r = subprocess.run([sys.executable, str(Path(convert.__file__)), "--verify", str(out_v7),
                            "--gguf", str(s1), "--gguf", str(s2), "--workers", "1"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        # Corrupt up_type in experts.layout (rewrite pack with wrong 6th column -> verify must fail).
        m_v7 = ModelFile(str(out_v7))
        bad_v7 = self.t / "bad_v7.basalt"
        w = ModelWriter(str(bad_v7))
        for n in m_v7.names():
            data = m_v7.read(n)
            if n == "experts.layout":
                data = data.replace(b" 14\n", b" 13\n")
            w.add_bytes(n, data)
        w.close()
        self.assertFalse(verify_pack.verify(str(bad_v7), [str(s1), str(s2)], workers=1))

    def test_help(self):
        r = subprocess.run([sys.executable, str(Path(convert.__file__)), "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        for flag in ("--gguf", "--mtp-source", "--mtp-dense", "--mmproj", "--draft-vocab"):
            self.assertIn(flag, r.stdout)
        for flag in ("--profile", "--hc8"):
            self.assertNotIn(flag, r.stdout)

    def test_checkpoint_path_takes_mtp_dense_only(self):
        run = lambda *x: subprocess.run([sys.executable, str(Path(convert.__file__)), "--sections-from", str(self.t / "b"),
                                         "--checkpoint", str(self.t), "--out", str(self.t / "o"), *x],
                                        capture_output=True, text=True)
        r = run("--mtp-dense", str(self.t / "missing.bin"))
        self.assertEqual(r.returncode, 2)
        self.assertIn("missing input", r.stderr)
        r = run("--mtp-source", str(self.t))
        self.assertEqual(r.returncode, 2)
        self.assertIn("--mtp-dense", r.stderr)
        r = run("--dry-run")
        self.assertEqual(r.returncode, 2)
        self.assertIn("takes only", r.stderr)


class DrafterTest(unittest.TestCase):
    """The draft layer's sections at tiny dimensions."""

    H, FF, NE = 64, 64, 2

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name)
        self._dims = (S.MTP_H, S.MTP_FF, S.MTP_NE)
        S.MTP_H, S.MTP_FF, S.MTP_NE = self.H, self.FF, self.NE
        rng = np.random.default_rng(1)
        manifest, self.vals = [], {}

        def put(name, shape):
            u16 = (rng.standard_normal(shape).astype(np.float32).view(np.uint32) >> 16).astype("<u2")
            f = name.replace(".", "_") + ".bin"
            u16.tofile(self.src / f)
            self.vals[name] = (u16.astype(np.uint32) << 16).view(np.float32)
            manifest.append({"name": name, "shape": list(shape), "file": f, "sha256": "0" * 64})

        put("mtp.fc_embedding.weight", (4, 64))                       # Q8_0
        put("mtp.hyper_connection_mixer.hc_norm.weight", (8,))        # norm: F32, +1
        put("mtp.layers.0.mlp.gate.weight", (4, 64))                  # BF16 as it is
        put("mtp.layers.0.mlp.experts.gate_up_proj", (self.NE, 2 * self.FF, self.H))
        put("mtp.layers.0.mlp.experts.down_proj", (self.NE, self.H, self.FF))
        (self.src / "mtp-manifest.json").write_text(json.dumps(manifest))

    def tearDown(self):
        S.MTP_H, S.MTP_FF, S.MTP_NE = self._dims
        self.tmp.cleanup()

    def test_dense(self):
        m = S.Mtp(self.src)
        blob = b"".join(m.dense_chunks())
        lines = [l.split() for l in m.dense_index().decode().splitlines()]
        self.assertEqual([l[0] for l in lines],
                         ["fc_embedding.weight", "hyper_connection_mixer.hc_norm.weight", "mlp.gate.weight"])
        by = {l[0]: l for l in lines}
        kind, rows, cols, off, n = by["fc_embedding.weight"][1:]
        self.assertEqual((kind, rows, cols, off), ("q8_0", "4", "64", "0"))
        self.assertEqual(blob[:int(n)], S.q8_0(self.vals["mtp.fc_embedding.weight"]))
        kind, rows, cols, off, n = by["hyper_connection_mixer.hc_norm.weight"][1:]
        self.assertEqual((kind, rows, cols), ("f32", "1", "8"))
        self.assertEqual(int(off) % 256, 0)
        got = np.frombuffer(blob[int(off):int(off) + int(n)], dtype="<f4")
        np.testing.assert_array_equal(got, self.vals["mtp.hyper_connection_mixer.hc_norm.weight"] + 1.0)
        kind, rows, cols, off, n = by["mlp.gate.weight"][1:]
        self.assertEqual(kind, "bf16")
        got = np.frombuffer(blob[int(off):int(off) + int(n)], dtype="<u2")
        np.testing.assert_array_equal((got.astype(np.uint32) << 16).view(np.float32).reshape(4, 64),
                                      self.vals["mtp.layers.0.mlp.gate.weight"])

    def test_experts(self):
        m = S.Mtp(self.src)
        blobs = list(m.expert_chunks())
        self.assertEqual(len(blobs), self.NE)
        size = 3 * (self.H * self.FF * 18 // 64)
        self.assertTrue(all(len(b) == size for b in blobs))
        # layout: the gate/up code plane first (rows interleaved: gate r, up r), then the down codes, then the scales
        gu = self.vals["mtp.layers.0.mlp.experts.gate_up_proj"][1]
        q_gate0 = S.q2_0(gu[0]).reshape(1, 18)
        q_up0 = S.q2_0(gu[self.FF]).reshape(1, 18)
        codes = np.frombuffer(blobs[1], dtype=np.uint8)
        self.assertEqual(codes[0:16].tobytes(), q_gate0[0, 2:].tobytes())
        self.assertEqual(codes[16:32].tobytes(), q_up0[0, 2:].tobytes())
        scales_at = 2 * self.FF * 16 + self.H * (self.FF // 64) * 16
        self.assertEqual(blobs[1][scales_at:scales_at + 2], q_gate0[0, :2].tobytes())
        self.assertEqual(blobs[1][scales_at + 2:scales_at + 4], q_up0[0, :2].tobytes())

    def test_q2_0_is_a_valid_quantization(self):
        x = np.random.default_rng(2).standard_normal(256).astype(np.float32)
        b = S.q2_0(x).reshape(-1, 18)
        d = b[:, :2].copy().view(np.float16).astype(np.float32)
        qs = b[:, 2:]
        codes = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(-1, 64).astype(np.float32)
        back = ((codes - 1.0) * d).reshape(-1)
        self.assertLess(float(np.abs(back - x).mean()), 0.5 * float(np.abs(x).mean()))

    def test_in_a_model(self):
        t = ConvertTest("test_sections")
        t.setUp()
        try:
            (t.t / "dv.bin").write_bytes(np.array([0, 1, 5], dtype="<i4").tobytes())
            convert.convert(t.args(mtp_source=str(self.src), draft_vocab=str(t.t / "dv.bin")))
            m = t.model()
            self.assertEqual({n for n in m.names() if n.startswith("mtp/")},
                             {"mtp/dense", "mtp/dense.index", "mtp/experts", "mtp/draft_vocab"})
            self.assertEqual(len(m.read("mtp/experts")), self.NE * 3 * (self.H * self.FF * 18 // 64))
            self.assertEqual(m.read("mtp/draft_vocab"), (t.t / "dv.bin").read_bytes())
            self.assertEqual(t.planned(mtp_source=str(self.src), draft_vocab=str(t.t / "dv.bin"), force=True),
                             {k: m.sections[k][1] for k in m.names()})
            # verify_pack must handle 6-col external mtp/dense.index without falsely flagging MTP tensors.
            import verify_pack
            from gguf_reader import GGUFFile
            g0 = GGUFFile(t.gguf)
            raw = t.gguf.read_bytes()
            gw = GGUFWriter()
            for k, v in g0.metadata.items():
                if isinstance(v, int) and not isinstance(v, bool):
                    gw.add(k, v, "u32")
                elif isinstance(v, float):
                    gw.add(k, v, "f32")
                else:
                    gw.add(k, v)
            gw.add("tokenizer.chat_template", "{{ x }}")
            for ti in g0.tensors:
                nb = ti.expected_bytes()
                gw.add_raw(ti.name, ti.shape, ti.type_name, raw[g0.data_start + ti.offset:g0.data_start + ti.offset + nb])
            g_tpl = gw.write(t.t / "mtp-tpl.gguf")
            t.tok["vocab.json"] = json.dumps({tok: i for i, tok in enumerate(TOKENS)}).encode()
            out_mtp = t.t / "mtp-verify.basalt"
            convert.convert(t.args(gguf=[str(g_tpl)], out=str(out_mtp), mtp_source=str(self.src),
                                   draft_vocab=str(t.t / "dv.bin")))
            self.assertTrue(verify_pack.verify(str(out_mtp), str(g_tpl), workers=1))

            # Mtp.expert_chunks(pool) must be bit-exact against single-threaded Mtp.expert_chunks().
            import multiprocessing
            mtp = S.Mtp(self.src)
            serial = list(mtp.expert_chunks())
            with multiprocessing.Pool(2) as pool:
                parallel = list(mtp.expert_chunks(pool))
            self.assertEqual(parallel, serial)

            (t.t / "dv.bin").write_bytes(np.array([0, 6], dtype="<i4").tobytes())      # 6 is outside the 6 tokens
            with self.assertRaises(SystemExit):
                convert.convert(t.args(mtp_source=str(self.src), draft_vocab=str(t.t / "dv.bin"), force=True))
        finally:
            t.tearDown()

    def test_tuned_dense(self):
        t = ConvertTest("test_sections")
        t.setUp()
        try:
            mtp = S.Mtp(self.src)
            plain = b"".join(mtp.dense_chunks())
            tuned = bytes(b ^ 0x5A for b in plain)
            f = t.t / "tuned.bin"
            f.write_bytes(tuned)
            convert.convert(t.args(mtp_source=str(self.src), mtp_dense=str(f)))
            m = t.model()
            self.assertEqual(m.read("mtp/dense"), tuned)
            self.assertEqual(m.read("mtp/dense.index"), mtp.dense_index())
            meta = json.loads(m.read("meta"))
            self.assertEqual(meta["mtp"]["dense"], {"tuning": "self_distilled", "sha256": hashlib.sha256(tuned).hexdigest()})
            f.write_bytes(tuned[:-1])
            with self.assertRaises(SystemExit):
                convert.convert(t.args(mtp_source=str(self.src), mtp_dense=str(f), force=True))
            with self.assertRaises(SystemExit):
                convert.convert(t.args(mtp_dense=str(f), force=True))
        finally:
            t.tearDown()


class PleFp8Test(unittest.TestCase):
    """The FP8 n-gram table as Q8_0 rows: every code decodes as the plain formula says, and the Q8_0 values stay
    within about half a step (plus the fp16 scale rounding) of the exact products."""

    @staticmethod
    def decode(code: int) -> float:
        sign, e, m = -1.0 if code >> 7 else 1.0, (code >> 3) & 15, code & 7
        if e == 15 and m == 7:
            return float("nan")
        return sign * (m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))

    def test_table_matches_formula(self):
        want = np.array([self.decode(c) for c in range(256)], dtype=np.float32)
        np.testing.assert_array_equal(S.fp8_e4m3_table(), want)

    def test_q8_0_rows(self):
        rng = np.random.default_rng(1)
        codes = rng.integers(0, 256, size=(7, S.PLE_DIM), dtype=np.uint8)
        codes[(codes & 0x7F) == 0x7F] = 0x38
        scale = 0.0125
        got = S.fp8_rows_to_q8_0(codes, scale)
        self.assertEqual(len(got), 7 * S.PLE_DIM // 32 * 34)
        deq = ggufq.dequantize(np.frombuffer(got, dtype=np.uint8), ggufq.GGMLQuantizationType.Q8_0).reshape(-1, 32)
        ref = (S.fp8_e4m3_table()[codes] * np.float32(scale)).reshape(-1, 32)
        step = np.abs(ref).max(axis=1, keepdims=True) / 127.0
        self.assertTrue((np.abs(deq - ref) <= step * 0.6 + 1e-9).all())

    def test_reads_safetensors_shards(self):
        rng = np.random.default_rng(2)
        shards = [rng.integers(0, 0x70, size=(r, S.PLE_DIM), dtype=np.uint8) for r in (3, 5)]
        stem = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
        scale_bf16 = np.array([0x3C23], dtype="<u2")            # 0.0098...
        head, data, at = {}, b"", 0
        for k, arr, dt in [(stem + "weight_scale", scale_bf16, "BF16"), (stem + "shard_0.weight", shards[0], "F8_E4M3"),
                           (stem + "shard_1.weight", shards[1], "F8_E4M3")]:
            raw = arr.tobytes()
            head[k] = {"dtype": dt, "shape": list(arr.shape) if arr.ndim > 1 else [1], "data_offsets": [at, at + len(raw)]}
            data += raw
            at += len(raw)
        hb = json.dumps(head).encode()
        want = b"".join(S.fp8_rows_to_q8_0(a, float(np.array([0x3C23 << 16], np.uint32).view(np.float32)[0]))
                        for a in shards)

        def write(path, entries):
            head, data, at = {}, b"", 0
            for k, arr, dt in entries:
                raw = arr.tobytes()
                head[k] = {"dtype": dt, "shape": list(arr.shape) if arr.ndim > 1 else [1],
                           "data_offsets": [at, at + len(raw)]}
                data += raw
                at += len(raw)
            hb = json.dumps(head).encode()
            path.write_bytes(struct.pack("<Q", len(hb)) + hb + data)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.safetensors"
            path.write_bytes(struct.pack("<Q", len(hb)) + hb + data)
            ple = S.PleFp8(str(path), chunk_rows=4)
            self.assertEqual(ple.rows, 8)
            self.assertAlmostEqual(ple.scale, float(np.array([0x3C23 << 16], np.uint32).view(np.float32)[0]), places=9)
            got = b"".join(S._ple_job(j) for j in ple.jobs())
            self.assertEqual(len(got), ple.nbytes)
            self.assertEqual(got, want)
            import multiprocessing
            with multiprocessing.Pool(2) as pool:
                self.assertEqual(b"".join(ple.chunks(pool, depth=2)), want)

            # a consolidated release spreads the shards over several files; the scale sits in one of them
            a, b = Path(tmp) / "a.safetensors", Path(tmp) / "b.safetensors"
            write(a, [(stem + "shard_1.weight", shards[1], "F8_E4M3")])
            write(b, [(stem + "weight_scale", scale_bf16, "BF16"), (stem + "shard_0.weight", shards[0], "F8_E4M3")])
            split = S.PleFp8([str(a), str(b)], chunk_rows=4)
            self.assertEqual(split.rows, 8)
            self.assertEqual(b"".join(S._ple_job(j) for j in split.jobs()), want)
            # a missing shard is refused, not silently dropped
            write(a, [(stem + "shard_2.weight", shards[1], "F8_E4M3")])
            with self.assertRaises(S.ConvertError):
                S.PleFp8([str(a), str(b)])


class BoundedImapTest(unittest.TestCase):
    def test_order_and_depth(self):
        class Pool:
            live = peak = 0

            def apply_async(self, fn, args):
                Pool.live += 1
                Pool.peak = max(Pool.peak, Pool.live)
                return self

            def get(self):
                Pool.live -= 1
                return 1

        self.assertEqual(sum(S.bounded_imap(Pool(), None, range(20), 3)), 20)
        self.assertEqual(Pool.peak, 3)


class Q6KRequantTest(unittest.TestCase):
    """Q6_K blocks decode (through the reference dequantizer) to within the 6-bit error of their source, and a pack
    rewrite changes only the Q8_0 projections it can."""

    def dequant(self, blob: bytes) -> np.ndarray:
        return ggufq.dequantize(np.frombuffer(blob, dtype=np.uint8), ggufq.GGMLQuantizationType.Q6_K).reshape(-1)

    def test_error_bound(self):
        rng = np.random.default_rng(3)
        x = (rng.standard_normal((64, 512)) * 0.02).astype(np.float32)
        x[3, :256] = 0.0
        x[4, 7] = 0.5
        got = S.q6_k(x)
        self.assertEqual(len(got), 64 * 2 * S.Q6K_BYTES)
        deq = self.dequant(got).reshape(x.shape)
        err = deq - x
        self.assertLess(np.sqrt((err ** 2).mean()), 0.025 * x.std())
        blk = np.abs(x).reshape(-1, 256).max(axis=1)
        self.assertTrue((np.abs(err).reshape(-1, 256).max(axis=1) <= blk * 0.035 + 1e-7).all())
        np.testing.assert_array_equal(deq[3, :256], 0.0)

    def test_from_q8_0_is_within_both_steps(self):
        rng = np.random.default_rng(4)
        x = (rng.standard_normal((8, 256)) * 0.05).astype(np.float32)
        q8 = S.q8_0(x)
        vals = S.q8_0_values(np.frombuffer(q8, dtype=np.uint8)).reshape(x.shape)
        self.assertLessEqual(np.abs(vals - x).max(), np.abs(x).max() / 127 * 0.51)
        deq = self.dequant(S.q6_k(vals)).reshape(x.shape)
        self.assertLess(np.sqrt(((deq - x) ** 2).mean()), 0.025 * x.std())

    def test_large_input_quantizes_each_block_once(self):
        """A matrix of many blocks must quantize each block the same as it does alone: a per-group term that goes
        wrong only at scale zeroes whole 16-value groups, which both checks below catch."""
        rng = np.random.default_rng(6)
        x = (rng.standard_normal((512, 512)) * 0.03).astype(np.float32)       # 1024 blocks
        blob = S.q6_k(x)
        self.assertEqual(blob[:len(S.q6_k(x[:8]))], S.q6_k(x[:8]))
        deq = self.dequant(blob).reshape(x.shape)
        self.assertLess(np.sqrt(((deq - x) ** 2).mean()), 0.02 * x.std())
        self.assertLess((deq == 0).mean(), 0.06)

    def test_gdn_head_order(self):
        src = S.gdn_block_source()
        self.assertEqual(sorted(src), list(range(48)))
        self.assertEqual(list(src[:4]), [0, 3, 6, 9])           # GGUF head r * 16 + k comes from checkpoint head k * 3 + r
        self.assertEqual(src[16], 1)
        idx = S.gdn_index(4096 + 6144, 4096)
        np.testing.assert_array_equal(idx[:4096], np.arange(4096))
        self.assertEqual(sorted(idx), list(range(4096 + 6144)))
        self.assertEqual(idx[4096 + 128], 4096 + 3 * 128)

    def test_pack_rewrite(self):
        rng = np.random.default_rng(5)
        a = (rng.standard_normal((6, 512)) * 0.03).astype(np.float32)       # blk.0.attn_qkv: ne0 512, 6 rows
        b = (rng.standard_normal((4, 160)) * 0.03).astype(np.float32)       # blk.0.attn_k: rows not whole blocks
        qa, qb = S.q8_0(a), S.q8_0(b)
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            w = ModelWriter(str(t / "base.basalt"))
            w.add_bytes("dense", b"d" * 100)
            offs = w.add_parts("gguf", [qa, qb], 32)
            w.add_bytes("experts", b"e" * 300)
            tens = {"blk.0.attn_qkv.weight": {"type": "Q8_0", "type_id": 8, "shape": [512, 6], "bytes": len(qa), "offset": offs[0]},
                    "blk.0.attn_k.weight": {"type": "Q8_0", "type_id": 8, "shape": [160, 4], "bytes": len(qb), "offset": offs[1]}}
            w.add_bytes("meta", json.dumps({"tensors": tens}).encode())
            w.close()
            out = t / "out.basalt"
            args = type("A", (), {"out": str(out), "requant_dense_q6k": str(t / "base.basalt"), "checkpoint": None,
                                  "workers": 2, "force": False})
            convert.requant_dense(args)
            m = ModelFile(str(out))
            meta = json.loads(m.read("meta"))
            self.assertEqual(m.read("dense"), b"d" * 100)
            self.assertEqual(m.read("experts"), b"e" * 300)
            ta, tb = meta["tensors"]["blk.0.attn_qkv.weight"], meta["tensors"]["blk.0.attn_k.weight"]
            self.assertEqual((ta["type"], ta["type_id"], ta["bytes"]), ("Q6_K", 14, 6 * 2 * S.Q6K_BYTES))
            self.assertEqual(tb["type"], "Q8_0")
            g = m.read("gguf")
            self.assertEqual(g[tb["offset"]:tb["offset"] + tb["bytes"]], qb)
            deq = self.dequant(g[ta["offset"]:ta["offset"] + ta["bytes"]]).reshape(a.shape)
            self.assertLess(np.sqrt(((deq - a) ** 2).mean()), 0.03 * a.std())

    def test_pack_rewrite_preserves_own_section_tensors(self):
        rng = np.random.default_rng(7)
        a = (rng.standard_normal((6, 512)) * 0.03).astype(np.float32)
        qa = S.q8_0(a)
        engram_blob = bytes(range(200))
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            w = ModelWriter(str(t / "base.basalt"))
            w.add_bytes("dense", b"d" * 100)
            offs = w.add_parts("gguf", [qa], 32)
            w.add_bytes("engram", engram_blob)
            tens = {
                "blk.0.attn_qkv.weight": {"type": "Q8_0", "type_id": 8, "shape": [512, 6], "bytes": len(qa), "offset": offs[0]},
                "blk.0.engram.weight": {"section": "engram", "type": "Q8_0", "type_id": 8, "shape": [64, 2],
                                        "bytes": len(engram_blob), "offset": 0},
            }
            w.add_bytes("meta", json.dumps({"tensors": tens}).encode())
            w.close()
            out = t / "out.basalt"
            args = type("A", (), {"out": str(out), "requant_dense_q6k": str(t / "base.basalt"), "checkpoint": None,
                                  "workers": 2, "force": False})
            convert.requant_dense(args)
            m = ModelFile(str(out))
            meta = json.loads(m.read("meta"))
            self.assertEqual(m.read("engram"), engram_blob)
            self.assertEqual(meta["tensors"]["blk.0.engram.weight"], tens["blk.0.engram.weight"])
            ta = meta["tensors"]["blk.0.attn_qkv.weight"]
            self.assertEqual((ta["type"], ta["type_id"], ta["bytes"]), ("Q6_K", 14, 6 * 2 * S.Q6K_BYTES))
            # gguf must hold only the gguf tensor (not a spurious copy of engram)
            self.assertEqual(len(m.read("gguf")), ta["offset"] + ta["bytes"])

    def test_verify_pack_kind3_dense_file_offset(self):
        import verify_pack
        rng = np.random.default_rng(9)
        t1 = rng.standard_normal((16, 32)).astype(np.float32)       # 2048 B stored F32 -> 1024 B F16 runtime (kind 3)
        t2 = rng.standard_normal((16, 32)).astype(np.float32)       # 2048 B stored F32 -> 2048 B F32 runtime (kind 2)
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            gw = GGUFWriter()
            gw.add("general.architecture", "qwen4exp")
            gw.add_f32("blk.0.ssm_a", t1)
            gw.add_f32("blk.0.attn_norm.weight", t2)
            gguf_path = gw.write(t / "src.gguf")

            src = S.SourceModel([gguf_path])
            # Custom arch: ssm_a wants F16 (stored as kind "3", F32->F16 narrowing), attn_norm wants F32 (kind "2")
            class Kind3Arch:
                not_in_dense = frozenset()
                @staticmethod
                def form_of(name):
                    return "F16" if name == "blk.0.ssm_a" else "F32"
                @staticmethod
                def native(name, tname):
                    return False

            dense = S.Dense(src, arch=Kind3Arch())
            blob = b"".join(dense.chunks())
            idx = dense.index_text()
            # Sanity: r[3] (file offset = 2048) != r[5] (pool offset = 1024) for the second tensor!
            rows = verify_pack.parse_index(idx)
            self.assertNotEqual(int(rows["blk.0.attn_norm.weight"][3]), int(rows["blk.0.attn_norm.weight"][5]))

            pack = t / "pack.basalt"
            w = ModelWriter(str(pack))
            w.add_bytes("dense", blob)
            w.add_bytes("dense.index", idx)
            w.add_bytes("tokenizer/vocab.json", b"{}")
            w.add_bytes("tokenizer/merges.txt", b"")
            w.add_bytes("tokenizer/token_type.json", b"[]")
            w.add_bytes("tokenizer/tokenizer.json", b"{}")
            w.add_bytes("tokenizer/chat_template.jinja", b"tpl")
            w.add_bytes("meta", json.dumps({
                "tensors": {},
                "gguf_kv": {"tokenizer.ggml.tokens": [], "tokenizer.chat_template": "tpl"},
                "license": {"text": "Apache-2.0"},
            }).encode())
            w.close()
            del dense, src
            import gc
            gc.collect()
            self.assertTrue(verify_pack.verify(pack, gguf_path, workers=1))
            gc.collect()

    def test_dense_q6k_rejects_nan(self):
        rng = np.random.default_rng(12)
        a = (rng.standard_normal((2, 256)) * 0.03).astype(np.float32)
        qa = bytearray(S.q8_0(a))
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            # 1) NaN in BF16 checkpoint source must raise ConvertError (not be masked by nan > 1.05 == False).
            w = ModelWriter(str(t / "base.basalt"))
            offs = w.add_parts("gguf", [bytes(qa)], 32)
            tens = {"blk.0.attn_q.weight": {"type": "Q8_0", "type_id": 8, "shape": [256, 2], "bytes": len(qa), "offset": offs[0]}}
            w.add_bytes("meta", json.dumps({"tensors": tens}).encode())
            w.close()
            ck = t / "ck"
            ck.mkdir()
            u16 = (a.view(np.uint32) >> 16).astype("<u2")
            u16[0, 0] = 0x7FC0  # BF16 NaN
            raw_bf16 = u16.tobytes()
            head = json.dumps({"model.language_model.layers.0.self_attn.q_proj.weight": {
                "dtype": "BF16", "shape": [2, 256], "data_offsets": [0, len(raw_bf16)]}}).encode()
            (ck / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", len(head)) + head + raw_bf16)
            plan = S.DenseQ6K(tens, str(ck))
            m = ModelFile(str(t / "base.basalt"))
            gguf_off, _ = m.section("gguf")
            import multiprocessing
            with multiprocessing.Pool(1) as pool:
                with self.assertRaises(S.ConvertError):
                    list(plan.chunks("blk.0.attn_q.weight", tens["blk.0.attn_q.weight"], str(t / "base.basalt"), gguf_off, pool))

            # 2) NaN fp16 scale in pack Q8_0 (st_path=None) must also raise ConvertError.
            qa[0:2] = b"\x00\x7e"  # fp16 NaN
            w2 = ModelWriter(str(t / "nan_q8.basalt"))
            offs2 = w2.add_parts("gguf", [bytes(qa)], 32)
            tens2 = {"blk.0.attn_q.weight": {"type": "Q8_0", "type_id": 8, "shape": [256, 2], "bytes": len(qa), "offset": offs2[0]}}
            w2.add_bytes("meta", json.dumps({"tensors": tens2}).encode())
            w2.close()
            plan2 = S.DenseQ6K(tens2, None)
            m2 = ModelFile(str(t / "nan_q8.basalt"))
            gguf_off2, _ = m2.section("gguf")
            with multiprocessing.Pool(1) as pool:
                with self.assertRaises(S.ConvertError):
                    list(plan2.chunks("blk.0.attn_q.weight", tens2["blk.0.attn_q.weight"], str(t / "nan_q8.basalt"), gguf_off2, pool))


class Qwen4ExpDefaultNameTest(unittest.TestCase):
    """Qwen4Exp.default_name takes the pack name and its quant from the GGUF file name (both the plain and the
    --sections-from path); an unrecognised name carries no quant rather than a guessed one."""

    def setUp(self):
        self.arch = convert.A.for_architecture("qwen4exp")

    def name(self, gguf, sections_from=False):
        return self.arch.default_name(gguf, {}, sections_from)

    def test_shard_quant_names_both_paths(self):
        gguf = "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
        self.assertEqual(self.name(gguf), "Qwen3.8-Flash-Next UD-Q4_K_XL")
        self.assertEqual(self.name(gguf, sections_from=True), "Qwen3.8-Flash-Next UD-Q4_K_XL")

    def test_iq3_shard_plain_name_matches_the_sections_from_name(self):
        gguf = "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
        self.assertEqual(self.name(gguf), "Qwen3.8-Flash-Next UD-IQ3_XXS")
        self.assertEqual(self.name(gguf, sections_from=True), "Qwen3.8-Flash-Next UD-IQ3_XXS")

    def test_single_file_quant(self):
        self.assertEqual(self.name("Qwen3.8-Flash-Next-Q8_0.gguf"), "Qwen3.8-Flash-Next Q8_0")

    def test_sections_from_single_file_quant(self):
        self.assertEqual(self.name("Qwen3.8-Flash-Next-Q8_0.gguf", sections_from=True), "Qwen3.8-Flash-Next Q8_0")

    def test_dot_separated_tag(self):
        self.assertEqual(self.name("Qwen3.8-Flash-Next.Q4_K_M.gguf"), "Qwen3.8-Flash-Next Q4_K_M")

    def test_lowercase_name_keeps_the_tag_case(self):
        self.assertEqual(self.name("qwen3.8-flash-next-q4_k_m.gguf"), "Qwen3.8-Flash-Next q4_k_m")

    def test_uppercase_extension_and_shard(self):
        self.assertEqual(self.name("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.GGUF"), "Qwen3.8-Flash-Next UD-Q4_K_XL")

    def test_shard_suffix_of_any_digit_count(self):
        self.assertEqual(self.name("Qwen3.8-Flash-Next-UD-Q4_K_XL-1-of-4.gguf"), "Qwen3.8-Flash-Next UD-Q4_K_XL")

    def test_shard_suffix_uppercase_of(self):
        self.assertEqual(self.name("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-OF-00004.gguf"), "Qwen3.8-Flash-Next UD-Q4_K_XL")

    def test_unrelated_name_has_no_quant(self):
        self.assertEqual(self.name("model.gguf"), "Qwen3.8-Flash-Next")
        self.assertEqual(self.name("model.gguf", sections_from=True), "Qwen3.8-Flash-Next")

    def test_only_the_file_name_is_used(self):
        gguf = "/models/unsloth/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
        self.assertEqual(self.name(gguf), "Qwen3.8-Flash-Next UD-Q4_K_XL")


class GgufStemTest(unittest.TestCase):
    """arch.gguf_stem: the file name alone, a trailing `.gguf` (any case) and a shard suffix (any case) of any digit
    count stripped."""

    def test_stems(self):
        self.assertEqual(convert.A.gguf_stem("/models/unsloth/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.GGUF"),
                         "Qwen3.8-Flash-Next-UD-Q4_K_XL")
        self.assertEqual(convert.A.gguf_stem("Qwen3.8-Flash-Next-UD-Q4_K_XL-1-of-4.gguf"), "Qwen3.8-Flash-Next-UD-Q4_K_XL")
        self.assertEqual(convert.A.gguf_stem("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-OF-00004.gguf"),
                         "Qwen3.8-Flash-Next-UD-Q4_K_XL")
        self.assertEqual(convert.A.gguf_stem("model.gguf"), "model")
        self.assertEqual(convert.A.gguf_stem("model"), "model")


if __name__ == "__main__":
    unittest.main()
