"""Tests for basalt_v1_to_v2.py on a tiny version 1 file.

    python3 -m unittest discover -s tools/migrations -p 'test_*.py'"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "convert"))
import basalt_v1_to_v2 as M  # noqa: E402
import hc8 as H  # noqa: E402
import model_file  # noqa: E402
from model_file import ModelFile, ModelWriter  # noqa: E402


class Migrate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        rng = np.random.default_rng(2)
        self.w = (rng.standard_normal((2, 64)).astype(np.float32).view(np.uint32) >> 16).astype("<u2")
        self.mw = (rng.standard_normal((1, 32)).astype(np.float32).view(np.uint32) >> 16).astype("<u2")
        dense = b"\1" * 64 + self.w.tobytes()
        index = ("# dense index v3\n# align 64 pool 384 tensors 2\n"
                 "blk.0.attn_q.weight 0 2 0 64 0 64 16 1 0 0 1 0 0 0 0 0 0 0\n"
                 f"blk.0.hc_attn_down.weight 0 4 64 256 64 256 64 2 0 0 1 0 0 0 0 0 0 0\n").encode()
        mtp_index = f"attn_hyper_connection.input_mix_weight_down.weight bf16 1 32 0 {self.mw.nbytes}\n".encode()
        self.meta = {"format": "basalt", "version": 1, "name": "Tiny", "converter": {"tool": "c.py", "git": "abc"},
                     "gguf_kv": {"general.sampling.top_k": 20}, "profile": {"n_pairs": 1}, "hc8": {"bytes": 3},
                     "sampling": {"temperature": 1.0, "top_k": 40, "top_p": 0.95, "presence_penalty": 0.0}}
        self.sections = [("tokenizer/vocab.json", b'{"a":0}'), ("dense", dense), ("dense.index", index),
                         ("mtp/dense", self.mw.tobytes()), ("mtp/dense.index", mtp_index), ("profile", b"P" * 20),
                         ("hc8", b"old"), ("vision/mmproj", b"M" * 100)]
        self.v1 = self.t / "old.basalt"
        saved, model_file.VERSION = model_file.VERSION, 1
        try:
            w = ModelWriter(str(self.v1))
            for n, b in self.sections:
                w.add_bytes(n, b)
            w.add_bytes("meta", json.dumps(self.meta).encode())
            w.close()
        finally:
            model_file.VERSION = saved

    def tearDown(self):
        self.tmp.cleanup()

    def test_migrate(self):
        out = self.t / "new.basalt"
        M.migrate(self.v1, out)
        m = ModelFile(str(out))
        self.assertEqual(m.version, 2)
        self.assertEqual(m.names(), [n for n, _ in self.sections if n not in ("profile", "hc8")] + ["hc8", "meta"])
        for n, b in self.sections:
            if n not in ("profile", "hc8"):
                self.assertEqual(m.read(n), b, n)
        for n in m.names():
            self.assertTrue(m.verify(n), n)
        t = H.read_table(m.read("hc8"))
        self.assertEqual(list(t), ["blk.0.hc_attn_down.weight", "mtp.attn_hyper_connection.input_mix_weight_down.weight"])
        self.assertEqual(t["blk.0.hc_attn_down.weight"][3], H.quantize(self.w))
        self.assertEqual(t["mtp.attn_hyper_connection.input_mix_weight_down.weight"][3], H.quantize(self.mw))
        meta = json.loads(m.read("meta"))
        self.assertEqual(meta["version"], 2)
        for k in ("profile", "hc8", "sampling"):
            self.assertNotIn(k, meta)
        self.assertEqual(meta["converter"], {"tool": "c.py"})
        kv = meta["gguf_kv"]
        self.assertEqual(kv["general.sampling.top_k"], 20)          # the GGUF's own value stays
        self.assertEqual((kv["general.sampling.temp"], kv["general.sampling.top_p"]), (1.0, 0.95))
        self.assertNotIn("general.sampling.presence_penalty", kv)

    def test_refusals(self):
        out = self.t / "new.basalt"
        M.migrate(self.v1, out)
        with self.assertRaises(SystemExit) as c:
            M.migrate(out, self.t / "again.basalt")
        self.assertIn("version 2", str(c.exception))
        with self.assertRaises(ValueError) as c:
            ModelFile(str(self.v1))
        self.assertIn("tools/migrations/basalt_v1_to_v2.py", str(c.exception))
        with open(self.v1, "rb") as f:
            self.assertEqual(struct.unpack_from("<I", f.read(12), 8)[0], 1)
        self.assertFalse((self.t / "again.basalt.partial").exists())


if __name__ == "__main__":
    unittest.main()
