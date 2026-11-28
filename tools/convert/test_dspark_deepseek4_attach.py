"""Unit tests for tools/convert/dspark_deepseek4_attach.py: synthetic mini deepseek4 pack + MTP
checkpoint attach, tensor roundtrip, and bit-for-bit rollback."""
from __future__ import annotations

import json
import pathlib
import struct
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import dspark_deepseek4_attach as D  # noqa: E402
from model_file import ModelFile, ModelWriter, undo_append  # noqa: E402


def _write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    head = {}
    off = 0
    chunks = []
    for name, (dt, shape, raw) in tensors.items():
        head[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        chunks.append(raw)
        off += len(raw)
    hb = json.dumps(head, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for c in chunks:
            f.write(c)


def _f32_to_bf16_bytes(x: np.ndarray) -> bytes:
    u32 = np.ascontiguousarray(x, dtype="<f4").view("<u4")
    return ((u32 >> 16).astype("<u2")).tobytes()


class TestDsparkDeepseek4Attach(unittest.TestCase):
    def test_attach_and_rollback(self):
        rng = np.random.default_rng(7)
        H = 64
        V = 128
        NH = 4
        HD = 64
        Q_LORA = 32
        O_GROUPS = 2
        O_LORA = 32
        N_EXP = 2
        FF_EXP = 32
        MARKOV_RANK = 32
        N_STAGES = 1
        BN, BK = 32, 32

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            ck_dir = root / "hf"
            ck_dir.mkdir()
            pack_path = root / "mini.basalt"
            undo_dir = root / "undo"

            meta = {
                "arch": "deepseek4",
                "gguf_kv": {
                    "deepseek4.embedding_length": H,
                    "deepseek4.vocab_size": V,
                    "deepseek4.block_count": 3,
                },
                "tensors": {},
            }
            w = ModelWriter(str(pack_path))
            w.add_bytes("meta", json.dumps(meta).encode("utf-8"))
            w.add_bytes("gguf", b"\x00" * 64)
            w.add_bytes("experts", b"\x00" * 64)
            w.add_bytes("experts.layout", b"")
            w.close()
            orig_bytes = pack_path.read_bytes()

            cfg = {
                "model_type": "deepseek_v4",
                "hidden_size": H,
                "vocab_size": V,
                "num_hidden_layers": 3,
                "num_attention_heads": NH,
                "n_routed_experts": N_EXP,
                "moe_intermediate_size": FF_EXP,
                "num_nextn_predict_layers": N_STAGES,
                "dspark_num_stages": N_STAGES,
                "dspark_block_size": 5,
                "dspark_noise_token_id": 127,
                "dspark_markov_rank": MARKOV_RANK,
                "dspark_target_layer_ids": [1, 2],
                "quantization_config": {"weight_block_size": [BN, BK]},
            }
            (ck_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

            tensors: dict[str, tuple[str, list[int], bytes]] = {}

            def add_fp8(name: str, rows: int, cols: int):
                codes = rng.integers(0, 120, size=(rows, cols), dtype=np.uint8)
                scales = np.full((rows // BN, cols // BK), 127, dtype=np.uint8)
                tensors[name] = ("F8_E4M3", [rows, cols], codes.tobytes())
                tensors[name.replace(".weight", ".scale")] = ("U8", [rows // BN, cols // BK], scales.tobytes())

            def add_bf16(name: str, shape: list[int]):
                arr = rng.normal(0.0, 0.1, size=shape).astype(np.float32)
                tensors[name] = ("BF16", shape, _f32_to_bf16_bytes(arr))

            def add_f32(name: str, shape: list[int]):
                arr = rng.normal(0.0, 0.1, size=shape).astype(np.float32)
                tensors[name] = ("F32", shape, arr.tobytes())

            add_bf16("mtp.0.main_norm.weight", [H])
            add_fp8("mtp.0.main_proj.weight", H, H * 2)

            for s in range(N_STAGES):
                p = f"mtp.{s}"
                add_bf16(f"{p}.attn_norm.weight", [H])
                add_bf16(f"{p}.ffn_norm.weight", [H])
                add_f32(f"{p}.hc_attn_base", [4, 4])
                add_f32(f"{p}.hc_attn_fn", [4, 4])
                add_f32(f"{p}.hc_attn_scale", [4])
                add_f32(f"{p}.hc_ffn_base", [4, 4])
                add_f32(f"{p}.hc_ffn_fn", [4, 4])
                add_f32(f"{p}.hc_ffn_scale", [4])
                add_f32(f"{p}.attn.attn_sink", [NH])
                add_bf16(f"{p}.attn.q_norm.weight", [Q_LORA])
                add_bf16(f"{p}.attn.kv_norm.weight", [512])
                add_fp8(f"{p}.attn.wq_a.weight", Q_LORA, H)
                add_fp8(f"{p}.attn.wq_b.weight", NH * HD, Q_LORA)
                add_fp8(f"{p}.attn.wkv.weight", 512, H)
                add_fp8(f"{p}.attn.wo_a.weight", O_GROUPS * O_LORA, NH * HD)
                add_fp8(f"{p}.attn.wo_b.weight", H, O_GROUPS * O_LORA)
                add_bf16(f"{p}.ffn.gate.weight", [N_EXP, H])
                add_f32(f"{p}.ffn.gate.bias", [N_EXP])
                add_f32(f"{p}.ffn.gate.bias_vl", [N_EXP])
                add_fp8(f"{p}.ffn.shared_experts.w1.weight", FF_EXP, H)
                add_fp8(f"{p}.ffn.shared_experts.w2.weight", H, FF_EXP)
                add_fp8(f"{p}.ffn.shared_experts.w3.weight", FF_EXP, H)
                for e in range(N_EXP):
                    for wname, r, c in (("w1", FF_EXP, H), ("w2", H, FF_EXP), ("w3", FF_EXP, H)):
                        q = rng.integers(0, 255, size=(r, c // 2), dtype=np.uint8)
                        sc = np.full((r, c // 32), 127, dtype=np.uint8)
                        tensors[f"{p}.ffn.experts.{e}.{wname}.weight"] = ("U8", [r, c // 2], q.tobytes())
                        tensors[f"{p}.ffn.experts.{e}.{wname}.scale"] = ("U8", [r, c // 32], sc.tobytes())

            add_f32("mtp.0.hc_head_base", [4, 4])
            add_f32("mtp.0.hc_head_fn", [4, 4])
            add_f32("mtp.0.hc_head_scale", [4])
            add_bf16("mtp.0.norm.weight", [H])
            add_bf16("mtp.0.markov_head.markov_w1.weight", [V, MARKOV_RANK])
            add_bf16("mtp.0.markov_head.markov_w2.weight", [V, MARKOV_RANK])
            add_bf16("mtp.0.confidence_head.proj.weight", [H])

            _write_safetensors(ck_dir / "model.safetensors", tensors)
            (ck_dir / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}),
                encoding="utf-8",
            )

            old_argv = sys.argv
            try:
                sys.argv = [
                    "dspark_deepseek4_attach.py",
                    "--checkpoint", str(ck_dir),
                    "--into", str(pack_path),
                    "--undo-dir", str(undo_dir),
                    "--workers", "1",
                ]
                D.main()
            finally:
                sys.argv = old_argv

            mf = ModelFile(str(pack_path))
            self.assertIn(D.SECTION, mf.sections)
            undo_append(str(pack_path), str(undo_dir))
            self.assertEqual(pack_path.read_bytes(), orig_bytes)


if __name__ == "__main__":
    unittest.main()
