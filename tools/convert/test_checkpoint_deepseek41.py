"""A tiny synthetic DeepSeek-V4.1 NVFP4 checkpoint (NVIDIA Model Optimizer's names and forms) converted against a tiny
deepseek41 BASE pack: experts, dense tensors and the Engram table, then with the shards streamed (deleted once used); a
tensor whose layout disagrees with BASE's is refused."""
import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import checkpoint as C  # noqa: E402
import checkpoint_deepseek41 as D  # noqa: E402
import convert  # noqa: E402
import model_sections as S  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import ModelFile  # noqa: E402

L, E, H, FF, ROWS, KV = 2, 4, 64, 64, 6, 320
TOKENS = ["a", "b", "c", "d"]
IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)


def iq4_nl(x: np.ndarray) -> bytes:
    """IQ4_NL blocks of 32 (fp16 d = amax / 127, nearest codebook value): BASE's Engram rows for the test."""
    b = x.reshape(-1, 32)
    d = (np.abs(b).max(axis=1) / 127).astype(np.float16)
    q = np.abs(b[:, :, None] - IQ4NL[None, None, :] * d.astype(np.float32)[:, None, None]).argmin(axis=2).astype(np.uint8)
    out = np.empty((b.shape[0], 18), np.uint8)
    out[:, :2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out.tobytes()


def write_safetensors(path: Path, tensors: dict):
    """tensors: name -> (dtype, shape, bytes)."""
    head, data, at = {}, [], 0
    for name, (dt, shape, raw) in tensors.items():
        head[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [at, at + len(raw)]}
        data.append(raw)
        at += len(raw)
    h = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(data))


class DeepSeekCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        rng = np.random.default_rng(5)
        self.rng = rng
        # the checkpoint's values: FP8 projections, BF16 / F32 tensors, NVFP4 experts, the MXFP4 Engram table
        self.fp8 = {}   # name -> (codes, scales)

        def fp8(name, rows, cols):
            codes = rng.integers(0, 0x7E, (rows, cols), dtype=np.uint8) | (rng.integers(0, 2, (rows, cols), dtype=np.uint8) << 7)
            scales = rng.integers(118, 126, ((rows + 31) // 32, (cols + 31) // 32), dtype=np.uint8)
            self.fp8[name] = (codes, scales)
            return {name: ("F8_E4M3", (rows, cols), codes.tobytes()), name[:-7] + ".scale": ("F8_E8M0", scales.shape, scales.tobytes())}

        def bf16(shape):
            x = rng.standard_normal(shape).astype(np.float32)
            return (x.view(np.uint32) >> 16).astype("<u2")

        shards = [{}, {}, {}]
        self.bf = {}
        for l in range(L):
            s = shards[l]
            s.update(fp8(f"layers.{l}.attn.wq_a.weight", 32, H))
            self.bf[f"layers.{l}.attn_norm.weight"] = bf16((H,))
            s[f"layers.{l}.attn_norm.weight"] = ("BF16", (H,), self.bf[f"layers.{l}.attn_norm.weight"].tobytes())
            self.bf[f"layers.{l}.ffn.gate.weight"] = bf16((E, H))
            s[f"layers.{l}.ffn.gate.weight"] = ("BF16", (E, H), self.bf[f"layers.{l}.ffn.gate.weight"].tobytes())
            for e in range(E):
                for r, (rows, cols) in (("w1", (FF, H)), ("w3", (FF, H)), ("w2", (H, FF))):
                    p = f"layers.{l}.ffn.experts.{e}.{r}."
                    s[p + "weight"] = ("U8", (rows, cols // 2), rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8).tobytes())
                    s[p + "weight_scale"] = ("F8_E4M3", (rows, cols // 16),
                                             (0x20 + rng.integers(0, 24, (rows, cols // 16))).astype(np.uint8).tobytes())
                    s[p + "weight_scale_2"] = ("F32", (), np.float32(0.01 * (1 + e + l)).tobytes())
                    s[p + "input_scale"] = ("F32", (), np.float32(1).tobytes())
        self.bf["embed.weight"] = bf16((len(TOKENS), H))
        shards[0]["embed.weight"] = ("BF16", (len(TOKENS), H), self.bf["embed.weight"].tobytes())
        self.bf["head.weight"] = bf16((len(TOKENS), H))
        shards[1]["head.weight"] = ("BF16", (len(TOKENS), H), self.bf["head.weight"].tobytes())
        self.qk = {k: bf16((4, H)) for k in ("q_weight", "k_weight")}
        for k, v in self.qk.items():
            shards[2][f"layers.1.engram.{k}"] = ("BF16", (4, H), v.tobytes())
        shards[2].update(fp8("layers.1.engram.wkv.weight", KV, 256))
        self.ecodes = rng.integers(0, 256, (ROWS, 128), dtype=np.uint8)
        self.escales = rng.integers(115, 125, (ROWS, 8), dtype=np.uint8)
        shards[2]["layers.1.engram.embed.weight"] = ("U8", (ROWS, 128), self.ecodes.tobytes())
        shards[2]["layers.1.engram.embed.scale"] = ("F8_E8M0", (ROWS, 8), self.escales.tobytes())
        ck = self.t / "ck"
        ck.mkdir()
        where = {}
        for i, s in enumerate(shards):
            f = f"model-{i + 1:05d}-of-00003.safetensors"
            write_safetensors(ck / f, s)
            where.update({n: f for n in s})
        (ck / "model.safetensors.index.json").write_text(json.dumps({"weight_map": where}))
        (ck / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": L, "n_routed_experts": E,
                                                                    "hidden_size": H, "moe_intermediate_size": FF}}))
        self.ck = ck
        self.base = self.make_base()

    def fp8_values(self, name):
        codes, scales = self.fp8[name]
        return D.fp8_values(codes, scales)

    def make_base(self, swap_q_a=False):
        """BASE: the same model as an Unsloth-style GGUF (Q8_0 projections, F32 norms, IQ4_NL Engram rows)."""
        g = GGUFWriter()
        P = "deepseek41."
        g.add("general.architecture", "deepseek41")
        g.add("general.name", "Tiny-DS")
        g.add("general.quantized_by", "Someone")
        g.add("general.tags", ["someone"])
        for k, v in (("block_count", L), ("embedding_length", H), ("expert_count", E), ("expert_used_count", 2),
                     ("expert_feed_forward_length", FF), ("embedding_length_per_layer_input", 256), ("engram.vocab_size", 16)):
            g.add(P + k, v, "u32")
        g.add(P + "ple.layers", [1], "array:u32")
        g.add(P + "ple.layer_multipliers", [7, 11, 13], "array:u64")
        g.add(P + "ple.head_offsets", [0], "array:u64")
        g.add(P + "ple.head_vocab_sizes", [ROWS], "array:u64")
        g.add(P + "engram.token_map", [0, 1, 2, 3], "array:u32")
        g.add("tokenizer.ggml.tokens", TOKENS)
        q8 = lambda x: np.frombuffer(S.q8_0(np.ascontiguousarray(x, np.float32)), np.uint8)
        f32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
        g.add_raw("token_embd.weight", [H, len(TOKENS)], "Q8_0", q8(f32(self.bf["embed.weight"])).tobytes())
        g.add_raw("output.weight", [H, len(TOKENS)], "Q8_0", q8(f32(self.bf["head.weight"])).tobytes())
        e = D.mxfp4_values(D.mxfp4_rows(self.ecodes, self.escales))
        g.add_raw("per_layer_token_embd.weight", [256, ROWS], "IQ4_NL", iq4_nl(e))
        for l in range(L):
            qa = self.fp8_values(f"layers.{l}.attn.wq_a.weight")
            if swap_q_a:
                qa = qa[::-1].copy()
            g.add_raw(f"blk.{l}.attn_q_a.weight", [H, 32], "Q8_0", q8(qa).tobytes())
            g.add_f32(f"blk.{l}.attn_norm.weight", f32(self.bf[f"layers.{l}.attn_norm.weight"]))
            g.add_raw(f"blk.{l}.ffn_gate_inp.weight", [H, E], "F32", f32(self.bf[f"layers.{l}.ffn.gate.weight"]).tobytes())
            for role, rows in (("gate", FF), ("up", FF), ("down", H)):
                g.add_raw(f"blk.{l}.ffn_{role}_exps.weight", [64, rows, E], "IQ4_NL", self.rng.bytes(18 * 2 * rows * E))
        g.add_raw("blk.1.engram_gate.weight", [H, 4], "F32", (f32(self.qk["q_weight"]) * f32(self.qk["k_weight"])).tobytes())
        g.add_raw("blk.1.engram_kv.weight", [256, KV], "Q8_0", q8(self.fp8_values("layers.1.engram.wkv.weight")).tobytes())
        gguf = g.write(self.t / ("swapped.gguf" if swap_q_a else "base.gguf"))
        orig = convert.export_tokenizer

        def tokenizer_files(first, out_dir):
            d = Path(out_dir) / "tokenizer"
            d.mkdir(parents=True, exist_ok=True)
            (d / "tokenizer.json").write_bytes(b"{}")
            return sorted(d.iterdir())
        convert.export_tokenizer = tokenizer_files
        try:
            out = self.t / ("swapped.basalt" if swap_q_a else "base.basalt")
            a = dict(gguf=[str(gguf)], mmproj=None, out=str(out), mtp_source=None, mtp_dense=None, draft_vocab=None,
                     name="Tiny", compat_bf16=False, no_vision=False, force=False, sections_from=None)
            convert.convert(type("A", (), a)())
        finally:
            convert.export_tokenizer = orig
        return out

    def tearDown(self):
        self.tmp.cleanup()

    def run_ck(self, out, stream=False, base=None, ck=None):
        a = dict(sections_from=str(base or self.base), checkpoint=str(ck or self.ck), out=str(self.t / out),
                 stream_shards=stream, workers=2, name="Tiny NVFP4", force=False,
                 dense_q8_0=False, mtp_dense=None)
        convert.from_checkpoint(type("A", (), a)())
        return ModelFile(str(self.t / out))

    def expected_experts(self):
        out = []
        for l in range(L):
            f = self.ck / f"model-{l + 1:05d}-of-00003.safetensors"
            head, base = S.read_safetensors_header(f)
            for e in range(E):
                parts, tail = [], []
                for r in D.ROLES:
                    p = f"layers.{l}.ffn.experts.{e}.{r}."
                    get = lambda k: np.fromfile(f, np.uint8, head[p + k]["data_offsets"][1] - head[p + k]["data_offsets"][0],
                                                offset=base + head[p + k]["data_offsets"][0])
                    shp = head[p + "weight"]["shape"]
                    parts.append(C.nvfp4_rows(get("weight").reshape(shp), get("weight_scale").reshape(shp[0], -1)))
                    tail.append(get("weight_scale_2").view("<f4")[0])
                out.append(np.concatenate(parts).tobytes() + np.array(tail + [0], "<f4").tobytes())
        return b"".join(out)

    def test_dense_and_engram(self):
        m = self.run_ck("nv.basalt")
        meta = json.loads(m.read("meta"))
        self.assertEqual(m.read("experts"), self.expected_experts())
        self.assertNotIn("dense", m.names())
        gg = m.read("gguf")
        t = meta["tensors"]

        def tensor(n):
            return gg[t[n]["offset"]:t[n]["offset"] + t[n]["bytes"]]
        for l in range(L):
            self.assertEqual(tensor(f"blk.{l}.attn_q_a.weight"), S.q8_0(self.fp8_values(f"layers.{l}.attn.wq_a.weight")))
            self.assertEqual(t[f"blk.{l}.attn_q_a.weight"]["type"], "Q8_0")
            f32 = (self.bf[f"layers.{l}.attn_norm.weight"].astype(np.uint32) << 16).view(np.float32)
            self.assertEqual(tensor(f"blk.{l}.attn_norm.weight"), f32.astype("<f4").tobytes())
        q, k = ((self.qk[x].astype(np.uint32) << 16).view(np.float32) for x in ("q_weight", "k_weight"))
        self.assertEqual(tensor("blk.1.engram_gate.weight"), (q * k).astype("<f4").tobytes())
        self.assertEqual(t["output.weight"]["type"], "Q8_0")
        self.assertEqual(m.read("engram"), D.mxfp4_rows(self.ecodes, self.escales).tobytes())
        e = meta["engram"]
        self.assertEqual((e["type"], e["type_id"], e["row_bytes"], e["bytes"]), ("MXFP4", 39, 136, ROWS * 136))
        self.assertNotIn("general.quantized_by", meta["gguf_kv"])
        self.assertNotIn("quantized_by", meta["license"])
        self.assertEqual(meta["gguf_kv"]["tokenizer.ggml.tokens"], TOKENS)
        self.assertLess(meta["checkpoint"]["engram"]["rel_rms_vs_sections_from"], 0.3)
        self.assertEqual(len(meta["sources"]), 3)

    def test_mxfp4_rows_are_the_definition(self):
        v = D.mxfp4_values(D.mxfp4_rows(self.ecodes, self.escales))
        e2m1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
        codes = np.empty((ROWS, 256), np.uint8)
        codes[:, 0::2], codes[:, 1::2] = self.ecodes & 15, self.ecodes >> 4
        want = e2m1[codes] * np.repeat(2.0 ** (self.escales.astype(np.float64) - 127), 32, axis=1)
        self.assertTrue(np.array_equal(v, want))

    def test_streamed_shards_are_deleted(self):
        ck = self.t / "ck2"
        shutil.copytree(self.ck, ck)
        self.run_ck("nv2.basalt", stream=True, ck=ck)
        self.assertEqual(sorted(p.name for p in ck.iterdir()), ["config.json", "model.safetensors.index.json"])
        self.assertEqual(ModelFile(str(self.t / "nv2.basalt")).read("experts"), self.expected_experts())

    def test_a_layout_error_is_refused(self):
        swapped = self.make_base(swap_q_a=True)
        with self.assertRaises(SystemExit) as cm:
            self.run_ck("bad.basalt", base=swapped)
        self.assertIn("attn_q_a", str(cm.exception))
        self.assertFalse((self.t / "bad.basalt").exists())


if __name__ == "__main__":
    unittest.main()
