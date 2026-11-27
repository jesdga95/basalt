"""Tests for attaching a checkpoint's mtp.safetensors DSpark to an existing inkling2 pack."""
from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import dspark_graft as G  # noqa: E402
import model_sections as S  # noqa: E402
from model_file import ModelFile, ModelWriter  # noqa: E402

TOOL = HERE / "dspark_graft.py"
N_VOCAB, PADDED = 12, 16


def config() -> dict:
    return {
        "text_config": {"hidden_size": 8, "unpadded_vocab_size": N_VOCAB},
        "mtp_config": {
            "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 4, "intermediate_size": 8, "max_position_embeddings": 262144,
            "local_layer_ids": [0], "sliding_window_size": 8, "block_size": 2, "markov_rank": 2,
            "markov_head_type": "rnn_hidden", "q_bias": False, "o_bias": False,
            "rope_theta": 1e6, "rms_norm_eps": 1e-6, "target_layer_ids": [3, 7, 11, 15],
        },
    }


def write_safetensors(path: pathlib.Path, cfg: dict) -> dict[str, np.ndarray]:
    """A tiny mtp.safetensors: the tool's own map, with deterministic finite BF16 values."""
    items = {}
    for _, _, key, shape, trim in G.wanted(cfg["mtp_config"], N_VOCAB):
        shape = [PADDED if trim and x is None else x for x in shape]
        n = int(np.prod(shape))
        vals = (np.arange(n, dtype=np.uint32) % 977 + 0x3C00).astype("<u2")
        items[key] = (shape, vals)
    off, entries, data = 0, {}, bytearray()
    for key, (shape, vals) in items.items():
        raw = vals.tobytes()
        entries[key] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + len(raw)]}
        data += raw
        off += len(raw)
    head = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(head)) + head + data)
    return {k: v for k, (_, v) in items.items()}


def write_pack(path: pathlib.Path, n_vocab: int = N_VOCAB) -> bytes:
    w = ModelWriter(str(path))
    w.add_bytes("gguf", b"an existing inkling2 pack's dense section")
    meta = {
        "format": "basalt", "version": 2, "arch": "inkling2", "name": "test",
        "gguf_kv": {"tokenizer.ggml.tokens": [f"t{i}" for i in range(n_vocab)]},
        "tensors": {"token_embd.weight": {"type": "BF16", "type_id": 30, "shape": [8, n_vocab],
                                          "bytes": 16 * n_vocab, "offset": 0}},
        "sources": ["an older fingerprint string, as the GGUF converter writes"],
    }
    w.add_bytes("meta", json.dumps(meta).encode("utf-8"))
    w.close()
    return path.read_bytes()


class GraftTest(unittest.TestCase):
    def test_graft_and_roll_back(self):
        with tempfile.TemporaryDirectory() as d:
            td = pathlib.Path(d)
            cfg = config()
            (td / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            mtp = td / "mtp.safetensors"
            source = write_safetensors(mtp, cfg)
            pack = td / "model.basalt"
            before = write_pack(pack)
            undo = td / "undo"
            r = subprocess.run([sys.executable, str(TOOL), "--mtp", str(mtp), "--config", str(td / "config.json"),
                                "--into", str(pack), "--undo-dir", str(undo)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

            m = ModelFile(str(pack))
            self.assertTrue(m.verify(G.SECTION))
            self.assertTrue(m.verify("meta"))
            meta = json.loads(m.read("meta"))
            entries = meta["tensors"]
            self.assertEqual(len(entries), 1 + 24)   # 9 global + 15 layer tensors, plus the pack's own
            kv = meta["gguf_kv"]
            self.assertEqual(kv["inkling2.dspark.block_count"], 1)
            self.assertEqual(kv["inkling2.dspark.target_layers"], [3, 7, 11, 15])
            self.assertEqual(kv["inkling2.dspark.sliding_window_pattern"], [True])
            self.assertEqual(kv["inkling2.dspark.markov_rank"], 2)
            self.assertEqual(meta["dspark"]["source"], "mtp.safetensors")

            blob = m.read(G.SECTION)
            for name, e in entries.items():
                if not name.startswith("dspark."):
                    continue
                self.assertEqual(e["section"], G.SECTION)
                self.assertEqual(e["offset"] % G.ALIGN, 0)
                self.assertEqual(len(blob[e["offset"]:e["offset"] + e["bytes"]]), e["bytes"])
            # BF16 copied as stored, with the padded vocabulary rows dropped
            got = np.frombuffer(blob[entries["dspark.markov.w1.weight"]["offset"]:
                                     entries["dspark.markov.w1.weight"]["offset"] +
                                     entries["dspark.markov.w1.weight"]["bytes"]], dtype="<u2")
            np.testing.assert_array_equal(got, source["markov_head.markov_w1.weight"][:N_VOCAB * 2])
            self.assertEqual(entries["dspark.markov.w1.weight"]["shape"], [2, N_VOCAB])
            # F32 form: widened, not raw BF16
            got32 = np.frombuffer(blob[entries["dspark.fuse_weight"]["offset"]:
                                       entries["dspark.fuse_weight"]["offset"] + entries["dspark.fuse_weight"]["bytes"]],
                                  dtype="<f4")
            np.testing.assert_array_equal(got32, S._bf16_to_f32(source["fuse_weight"]))
            self.assertEqual(entries["dspark.blk.0.attn_q.weight"]["shape"], [8, 8])  # reversed [8, 8] here

            rollback = subprocess.run([sys.executable, str(TOOL), "--roll-back", "--into", str(pack),
                                       "--undo-dir", str(undo)], capture_output=True, text=True)
            self.assertEqual(rollback.returncode, 0, rollback.stderr + rollback.stdout)
            self.assertEqual(hashlib.sha256(pack.read_bytes()).digest(), hashlib.sha256(before).digest())

    def test_refuses_a_second_drafter(self):
        with tempfile.TemporaryDirectory() as d:
            td = pathlib.Path(d)
            cfg = config()
            (td / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            write_safetensors(td / "mtp.safetensors", cfg)
            pack = td / "model.basalt"
            write_pack(pack)
            m = ModelFile(str(pack))
            meta = json.loads(m.read("meta"))
            meta["gguf_kv"]["inkling2.dspark.block_count"] = 1
            # rewrite the pack's meta section in place is unnecessary: graft reads it from the file
            # served by ModelFile, so copy it into a fresh writer with that meta instead
            p2 = td / "drafter.basalt"
            w = ModelWriter(str(p2))
            w.add_bytes("gguf", b"x")
            w.add_bytes("meta", json.dumps(meta).encode("utf-8"))
            w.close()
            r = subprocess.run([sys.executable, str(TOOL), "--mtp", str(td / "mtp.safetensors"), "--config",
                                str(td / "config.json"), "--into", str(p2), "--undo-dir", str(td / "undo")],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("already carries a DSpark drafter", r.stderr)


if __name__ == "__main__":
    unittest.main()
