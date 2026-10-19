"""Tests for convert.py and model_sections.py on a tiny synthetic model, drafter and side files.

    python3 -m unittest discover -s tools/convert -p 'test_convert.py'"""
import hashlib
import json
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
        g.add_q2_0("blk.0.attn_gate.weight", self.q2, [64, 2])
        self.gguf = g.write(t / "m-00001-of-00001.gguf")
        self.tok = {"tokenizer.json": b"{}", "vocab.json": b'{"a":0}', "merges.txt": b"a b\n",
                    "token_type.json": b"[1]", "chat_template.jinja": b"{{ x }}"}
        self.hc8 = rng.bytes(123)
        (t / "hc8.bin").write_bytes(self.hc8)
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
                 draft_vocab=None, profile=None, profile_from=None, hc8=str(t / "hc8.bin"), name="Tiny",
                 compat_bf16=False, no_vision=False, force=False, sections_from=None)
        d.update(kw)
        return type("A", (), d)()

    def model(self):
        return ModelFile(str(self.t / "out.basalt"))

    # ---------------------------------------------------------------- sections
    def test_sections(self):
        convert.convert(self.args())
        m = self.model()
        self.assertFalse((self.t / "out.basalt.partial").exists())
        want = {"hc8", "vision/mmproj", "profile", "dense", "dense.index", "experts", "experts.layout", "gguf", "meta"}
        want |= {"tokenizer/" + k for k in self.tok}
        self.assertEqual(set(m.names()), want)
        for k in m.names():
            self.assertTrue(m.verify(k), k)
        self.assertEqual(m.read("hc8"), self.hc8)
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
        self.assertEqual(meta["hc8"], {"bytes": 123})
        self.assertEqual(meta["vision"]["mmproj"], "vision/mmproj")
        self.assertEqual(meta["sources"][0]["size"], self.gguf.stat().st_size)
        self.assertEqual(meta["profile"]["n_pairs"], N_LAYERS * N_EXPERT)
        self.assertNotIn("engine_args", meta)
        self.assertEqual(meta["sampling"]["top_k"], 20)
        self.assertIsNone(meta["mtp"])
        lic = meta["license"]
        self.assertEqual(lic["name"], "Qwen Community License 1.0")
        self.assertEqual(lic["text"], (Path(convert.__file__).parent / "qwen-community-license-1.0.txt").read_text(encoding="utf-8"))
        self.assertEqual(lic["quantized_by"], "Someone")
        self.assertEqual(set(lic), {"name", "text", "quantized_by"})

    def test_without_hc8_or_drafter(self):
        convert.convert(self.args(hc8=None))
        m = self.model()
        self.assertNotIn("hc8", m.names())
        self.assertFalse([n for n in m.names() if n.startswith("mtp/")])
        self.assertIsNone(json.loads(m.read("meta"))["hc8"])

    # ---------------------------------------------------------------- profile
    def test_profile_fill_order_and_traces(self):
        convert.convert(self.args())
        blob = self.model().read("profile")
        info = S.profile_info(blob, N_LAYERS, N_EXPERT)
        self.assertEqual(info["slots"], 8)
        pairs = struct.unpack_from("<16H", blob, 20)
        self.assertEqual(pairs[:4], (0, 0, 1, 0))                    # expert-major fill: (0,0) (1,0) (0,1) ...
        # three records, each routing its expert and expert 0
        trace = b"".join(struct.pack("<ii2i2f", l, 2, e, 0, 1.0, 1.0) for l, e in ((1, 3), (0, 2), (1, 3)))
        (self.t / "trace.bin").write_bytes(trace)
        convert.convert(self.args(profile_from=[str(self.t / "trace.bin")], force=True))
        blob = self.model().read("profile")
        pairs = struct.unpack_from("<16H", blob, 20)
        self.assertEqual(pairs[:8], (1, 0, 1, 3, 0, 0, 0, 2))   # by count, then by pair
        table = struct.unpack_from("<8i", blob, 20 + 4 * 8)
        self.assertEqual(table[1 * N_EXPERT + 3], 1)

    def test_profile_file(self):
        ranked = [(l, e) for e in range(N_EXPERT) for l in range(N_LAYERS)][::-1]
        (self.t / "p.bin").write_bytes(S.build_profile(ranked, N_LAYERS, N_EXPERT))
        convert.convert(self.args(profile=str(self.t / "p.bin")))
        self.assertEqual(self.model().read("profile"), (self.t / "p.bin").read_bytes())

    def test_bad_profile(self):
        good = S.build_profile([(0, 0)], N_LAYERS, N_EXPERT)
        for name, blob in (("short", good[:-4]), ("other model", S.build_profile([(0, 0)], N_LAYERS, N_EXPERT + 1))):
            (self.t / "p.bin").write_bytes(blob)
            with self.assertRaises(SystemExit, msg=name):
                convert.convert(self.args(profile=str(self.t / "p.bin")))
            self.assertFalse((self.t / "out.basalt").exists())

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

    def test_gate_up_type_mismatch_is_refused(self):
        g = GGUFWriter()
        rng = np.random.default_rng(1)
        g.add("general.architecture", "qwen")
        g.add("qwen.block_count", 1, "u32")
        g.add("tokenizer.ggml.tokens", TOKENS)
        g.add_f32("blk.0.ffn_gate_inp.weight", self.router)
        g.add_raw("blk.0.ffn_gate_exps.weight", [256, 2, N_EXPERT], "Q5_K", rng.bytes(176 * 2 * N_EXPERT))
        g.add_raw("blk.0.ffn_up_exps.weight", [256, 2, N_EXPERT], "Q6_K", rng.bytes(210 * 2 * N_EXPERT))
        g.add_raw("blk.0.ffn_down_exps.weight", [32, 2, N_EXPERT], "Q8_0", rng.bytes(34 * 2 * N_EXPERT))
        bad = g.write(self.t / "bad-00001-of-00001.gguf")
        with self.assertRaises(SystemExit) as c:
            convert.convert(self.args(gguf=[str(bad)]))
        self.assertIn("gate and up differ in type", str(c.exception))

    def test_sections_from_with_gguf(self):
        import pack_vision
        self.vision = pack_vision.vision_sections
        convert.convert(self.args(out=str(self.t / "base.basalt")))
        base = ModelFile(str(self.t / "base.basalt"))
        path, _ = self.mixed_gguf(["Q6_K", "Q5_K"])
        convert.convert(self.args(gguf=[str(path)], out=str(self.t / "new.basalt"), sections_from=str(self.t / "base.basalt"),
                                  mmproj=None, hc8=None, name="Mixed"))
        m = ModelFile(str(self.t / "new.basalt"))
        for n in m.names():
            self.assertTrue(m.verify(n), n)
        for n in ("profile", "hc8", "vision/mmproj"):
            self.assertEqual(m.read(n), base.read(n), n)
        meta = json.loads(m.read("meta"))
        self.assertEqual(meta["sections_from"]["copied"], [n for n in base.names() if n != "vision/text_vocab"
                                                           and (n.startswith(("mtp/", "vision/")) or n in ("profile", "hc8"))])
        self.assertEqual(m.read("vision/text_vocab"), pack_vision.vocab_only_gguf(str(path)))
        self.assertEqual(meta["vision"], json.loads(base.read("meta"))["vision"])
        self.assertEqual(m.read("experts.layout").decode().splitlines()[1].split()[:3], ["0", "14", "8"])

    def test_help(self):
        r = subprocess.run([sys.executable, str(Path(convert.__file__)), "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        for flag in ("--gguf", "--mtp-source", "--mtp-dense", "--mmproj", "--profile-from", "--profile ", "--hc8", "--draft-vocab"):
            self.assertIn(flag, r.stdout)

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
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.safetensors"
            path.write_bytes(struct.pack("<Q", len(hb)) + hb + data)
            ple = S.PleFp8(str(path), chunk_rows=4)
            self.assertEqual(ple.rows, 8)
            self.assertAlmostEqual(ple.scale, float((np.uint32(0x3C23) << 16).view(np.float32)), places=9)
            got = b"".join(S._ple_job(j) for j in ple.jobs())
            self.assertEqual(len(got), ple.nbytes)
            want = b"".join(S.fp8_rows_to_q8_0(a, ple.scale) for a in shards)
            self.assertEqual(got, want)
            import multiprocessing
            with multiprocessing.Pool(2) as pool:
                self.assertEqual(b"".join(ple.chunks(pool, depth=2)), want)


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


if __name__ == "__main__":
    unittest.main()
