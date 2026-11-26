"""Regression test for DeepSeek V4's raw I32 routing tensor."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import convert  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import ModelFile  # noqa: E402
from arch.deepseek4 import DeepSeek4  # noqa: E402
from model_sections import ConvertError  # noqa: E402


GEOMETRY = {
    "block_count": 43,
    "context_length": 1048576,
    "embedding_length": 4096,
    "attention.head_count": 64,
    "attention.head_count_kv": 1,
    "attention.key_length": 512,
    "attention.q_lora_rank": 1024,
    "attention.output_group_count": 8,
    "attention.output_lora_rank": 1024,
    "attention.layer_norm_rms_epsilon": 1e-20,
    "attention.compress_rope_freq_base": 160000.0,
    "attention.compress_ratios": [0, 0] + [4 if i % 2 == 0 else 128 for i in range(2, 43)] + [0, 0, 0],
    "attention.sliding_window": 128,
    "rope.dimension_count": 64,
    "rope.freq_base": 10000.0,
    "rope.scaling.factor": 16.0,
    "rope.scaling.original_context_length": 65536,
    "rope.scaling.yarn_beta_fast": 32.0,
    "rope.scaling.yarn_beta_slow": 1.0,
    "expert_count": 256,
    "expert_used_count": 6,
    "expert_shared_count": 1,
    "expert_feed_forward_length": 2048,
    "expert_gating_func": 4,
    "expert_weights_scale": 1.5,
    "expert_weights_norm": True,
    "hyper_connection.count": 4,
    "hyper_connection.epsilon": 1e-6,
    "hyper_connection.sinkhorn_iterations": 20,
    "embedding_length_out": 16384,
    "hash_layer_count": 3,
    "attention.indexer.head_count": 64,
    "attention.indexer.key_length": 128,
    "attention.indexer.top_k": 512,
}


def add_geometry(gguf):
    for key, value in GEOMETRY.items():
        gguf.add(f"deepseek4.{key}", value)


class DeepSeek4I32Test(unittest.TestCase):
    def test_incomplete_architecture_metadata_is_refused(self):
        with self.assertRaisesRegex(ConvertError, "missing required deepseek4.block_count"):
            DeepSeek4().check({}, {})

    def test_unsupported_compression_schedule_is_refused(self):
        kv = {f"deepseek4.{key}": value for key, value in GEOMETRY.items()}
        ratios = list(GEOMETRY["attention.compress_ratios"])
        ratios[2] = 2
        kv["deepseek4.attention.compress_ratios"] = ratios
        with self.assertRaisesRegex(ConvertError, "unsupported layer ratio"):
            DeepSeek4().check(kv, {})

    def test_i32_tensor_is_preserved_and_architecture_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gguf = GGUFWriter()
            gguf.add("general.architecture", "deepseek4")
            gguf.add("general.name", "Tiny DeepSeek V4 Vision")
            add_geometry(gguf)
            gguf.add("tokenizer.ggml.tokens", ["a", "b"])
            gguf.add_f32("blk.0.ffn_gate_inp.weight", np.zeros((8, 4), dtype=np.float32))
            raw = np.asarray([0, 1, 2, 3], dtype="<i4").tobytes()
            gguf.add_raw("blk.0.ffn_gate_tid2eid.weight", [4], "I32", raw)
            for role in ("gate", "up", "down"):
                gguf.add_f32(f"blk.0.ffn_{role}_exps.weight", np.zeros((2, 2, 4), dtype=np.float32))
            source = gguf.write(root / "tiny.gguf")
            parsed = GGUFFile(source)
            self.assertEqual(parsed.tensors[1].expected_bytes(), len(raw))

            tok_dir = root / "tokens"
            tok_dir.mkdir()
            token_file = tok_dir / "tokenizer.json"
            token_file.write_text("{}", encoding="utf-8")
            original = convert.export_tokenizer
            convert.export_tokenizer = lambda first, out_dir: [token_file]
            try:
                out = root / "tiny.basalt"
                args = type("Args", (), {
                    "gguf": [str(source)], "mmproj": None, "out": str(out),
                    "mtp_source": None, "mtp_dense": None, "draft_vocab": None,
                    "name": "Tiny DeepSeek V4 Vision", "compat_bf16": False, "no_vision": True,
                    "force": False, "sections_from": None, "dspark": None,
                    "dry_run": False, "hf_dir": None,
                })()
                convert.convert(args)
            finally:
                convert.export_tokenizer = original

            pack = ModelFile(str(out))
            meta = json.loads(pack.read("meta"))
            self.assertEqual(meta["arch"], "deepseek4")
            self.assertEqual(meta["name"], "Tiny DeepSeek V4 Vision")
            info = meta["tensors"]["blk.0.ffn_gate_tid2eid.weight"]
            self.assertEqual(info["type"], "I32")
            self.assertEqual(info["type_id"], 26)
            section = pack.read("gguf")
            self.assertEqual(section[info["offset"]:info["offset"] + info["bytes"]], raw)


if __name__ == "__main__":
    unittest.main()
