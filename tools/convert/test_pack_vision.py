import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_vision import vision_sections  # noqa: E402


def s(x: bytes) -> bytes:
    return struct.pack("<Q", len(x)) + x


def kv(key, vtype, payload):
    return s(key.encode()) + struct.pack("<I", vtype) + payload


def gguf(entries, tensors=b"", n_tensors=0):
    return b"GGUF" + struct.pack("<IQQ", 3, n_tensors, len(entries)) + b"".join(entries) + tensors


ENTRIES = [kv("general.architecture", 8, s(b"qwen")), kv("split.count", 4, struct.pack("<I", 3)),
           kv("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + s(b"a") + s(b"bb")),
           kv("x.f", 9, struct.pack("<IQ", 6, 2) + b"\0" * 8)]


class Pack(unittest.TestCase):
    def test_sections(self):
        with tempfile.TemporaryDirectory() as d:
            mm, shard = os.path.join(d, "mm.gguf"), os.path.join(d, "s1.gguf")
            with open(mm, "wb") as f:
                f.write(b"M" * 10)
            with open(shard, "wb") as f:
                f.write(gguf(ENTRIES, b"T" * 100, n_tensors=1))
            (n1, b1, p1, o1, z1), (n2, b2, p2, o2, z2) = vision_sections(mm, [shard])
            self.assertEqual((n1, b1, p1, o1, z1), ("vision/mmproj", None, mm, 0, 10))
            self.assertEqual((n2, p2, o2, z2), ("vision/text_vocab", None, 0, len(b2)))
            self.assertEqual(b2[:4], b"GGUF")
            self.assertEqual(struct.unpack_from("<IQQ", b2, 4), (3, 0, 3))
            self.assertNotIn(b"split.count", b2)
            self.assertIn(b"tokenizer.ggml.tokens", b2)
            self.assertEqual(len(b2) % 32, 0)

    def test_no_tokenizer(self):
        with tempfile.TemporaryDirectory() as d:
            mm, shard = os.path.join(d, "mm.gguf"), os.path.join(d, "s1.gguf")
            with open(mm, "wb") as f:
                f.write(b"M")
            with open(shard, "wb") as f:
                f.write(gguf(ENTRIES[:1]))
            with self.assertRaises(ValueError):
                vision_sections(mm, [shard])


class VocabFromKv(unittest.TestCase):
    """The vocabulary section of a pack built from a checkpoint: the pack's own metadata, read back as GGUF."""

    KV = {"general.architecture": "qwen4exp",
          "tokenizer.ggml.tokens": ["a", "bb"],
          "tokenizer.ggml.bos_token_id": 151643,
          "qwen4exp.ple.nvfp4_scale_2": 3.324236240587197e-05,
          "qwen4exp.ngram.multipliers": [1469598103934665603, 1099511628211, 2 ** 40],   # past int32
          "split.count": 3}

    def test_values_survive_the_round_trip(self):
        from pack_vision import vocab_gguf_bytes
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from gguf_reader import GGUFFile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "v.gguf")
            with open(p, "wb") as f:
                f.write(vocab_gguf_bytes(self.KV))
            g = GGUFFile(p)
        self.assertEqual(g.tensors, [])
        self.assertEqual(g.metadata["tokenizer.ggml.tokens"], ["a", "bb"])
        self.assertEqual(g.metadata["tokenizer.ggml.bos_token_id"], 151643)
        self.assertEqual(g.metadata["qwen4exp.ngram.multipliers"], self.KV["qwen4exp.ngram.multipliers"])
        self.assertAlmostEqual(g.metadata["qwen4exp.ple.nvfp4_scale_2"], 3.324236240587197e-05, places=18)
        self.assertNotIn("split.count", g.metadata)


if __name__ == "__main__":
    unittest.main()
