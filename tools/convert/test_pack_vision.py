import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pack_vision  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402
from pack_vision import vision_sections, vocab_only_gguf  # noqa: E402


def s(x: bytes) -> bytes:
    return struct.pack("<Q", len(x)) + x


def kv(key, vtype, payload):
    return s(key.encode()) + struct.pack("<I", vtype) + payload


def gguf(entries, tensors=b"", n_tensors=0):
    return b"GGUF" + struct.pack("<IQQ", 3, n_tensors, len(entries)) + b"".join(entries) + tensors


def tensor_info(name=b"w"):
    return s(name) + struct.pack("<IQ", 1, 4) + struct.pack("<IQ", 0, 0) + b"\0" * 64


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

    def _assert_roundtrip(self, entries, initial_read):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.gguf"
            out = Path(d) / "vocab.gguf"
            src.write_bytes(gguf(entries, tensor_info(), n_tensors=1))
            expected = {k: v for k, v in GGUFFile(src).metadata.items() if not k.startswith("split.")}
            with mock.patch.object(pack_vision, "_INITIAL_READ", initial_read):
                data = vocab_only_gguf(str(src))
            self.assertIsNotNone(data)
            self.assertEqual(len(data) % 32, 0)
            out.write_bytes(data)
            parsed = GGUFFile(out)
            self.assertEqual(parsed.tensors, [])
            self.assertEqual(parsed.metadata, expected)

    def test_chunk_boundary_crossing_last_entry(self):
        base = ENTRIES[:3]
        scores = [float(i) * 0.25 for i in range(64)]
        fixed_array = kv("tokenizer.ggml.scores", 9, struct.pack(f"<IQ{len(scores)}f", 6, len(scores), *scores))
        long_string = kv("tokenizer.chat_template", 8, s(b"{% for m in messages %}" * 12))
        merges = [f"tok{i} tok{i + 1}".encode() for i in range(20)]
        string_array = kv("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, len(merges)) + b"".join(s(m) for m in merges))
        fixed_scalar = kv("general.quantization_version", 10, struct.pack("<Q", 0x0102030405060708))

        cases = [
            ("fixed_array", base + [fixed_array], len(gguf(base + [fixed_array])) - 100),
            ("string", base + [long_string], len(gguf(base + [long_string])) - 50),
            ("string_array_last_elem", base + [string_array], len(gguf(base + [string_array])) - 5),
            ("fixed_scalar", base + [fixed_scalar], len(gguf(base + [fixed_scalar])) - 3),
            ("key_mid_name", base + [fixed_scalar], len(gguf(base)) + 16),
            ("multi_growth", base + [fixed_array, long_string, string_array], 32),
        ]
        for label, entries, initial_read in cases:
            with self.subTest(label=label, initial_read=initial_read):
                self._assert_roundtrip(entries, initial_read)

    def test_truncated_file_raises(self):
        base = ENTRIES[:3]
        fixed_array = kv("tokenizer.ggml.scores", 9, struct.pack("<IQ4f", 6, 4, 1.0, 2.0, 3.0, 4.0))
        long_string = kv("tokenizer.chat_template", 8, s(b"template-payload"))
        string_array = kv("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 2) + s(b"a b") + s(b"c defg"))
        fixed_scalar = kv("general.quantization_version", 10, struct.pack("<Q", 42))

        cases = [
            ("fixed_array", gguf(base + [fixed_array])[:-4]),
            ("string", gguf(base + [long_string])[:-5]),
            ("string_array", gguf(base + [string_array])[:-3]),
            ("fixed_scalar", gguf(base + [fixed_scalar])[:-3]),
            ("key", gguf(base + [fixed_scalar])[:len(gguf(base)) + 12]),
            ("header", b"GGUF" + struct.pack("<I", 3)),
        ]
        with tempfile.TemporaryDirectory() as d:
            for label, payload in cases:
                for exact_chunk in (False, True):
                    with self.subTest(label=label, exact_chunk=exact_chunk):
                        p = Path(d) / f"{label}_{exact_chunk}.gguf"
                        p.write_bytes(payload)
                        initial_read = len(payload) if exact_chunk else pack_vision._INITIAL_READ
                        with mock.patch.object(pack_vision, "_INITIAL_READ", initial_read):
                            with self.assertRaises(struct.error):
                                vocab_only_gguf(str(p))


if __name__ == "__main__":
    unittest.main()
