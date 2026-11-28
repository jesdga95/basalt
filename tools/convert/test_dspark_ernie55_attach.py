"""Unit tests for tools/convert/dspark_ernie55_attach.py: synthetic mini ERNIE-5.5 pack + MTP checkpoint
attach, tensor roundtrip (FP8->Q8_0, BF16->BF16_T, BF16->Q8_0, VHA fold, split router, MXFP4 experts),
SHA-256 verification, and bit-for-bit rollback."""
from __future__ import annotations

import hashlib
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
import checkpoint_ernie55 as E  # noqa: E402
import dspark_ernie55_attach as D  # noqa: E402
import model_sections as S  # noqa: E402
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


def _dequant_q8_0(raw: bytes, rows: int, cols: int) -> np.ndarray:
    nb = cols // 32
    b = np.frombuffer(raw, dtype=np.uint8).reshape(rows, nb, 34)
    d = b[..., :2].copy().view("<f2").astype(np.float32)
    q = b[..., 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(rows, cols)


class TestDsparkErnie55Attach(unittest.TestCase):
    def test_attach_and_rollback(self):
        rng = np.random.default_rng(42)
        H = 64
        V = 128
        NH = 4
        HD = 64
        Q_LORA = 32
        O_GROUPS = 2
        O_LORA = 32
        N_EXP = 4
        LAT = 32
        FF_EXP = 32
        MARKOV_RANK = 32
        N_STAGES = 2
        TARGET_LAYERS = [1, 2]
        BN, BK = 32, 32

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            ck_dir = root / "hf"
            ck_dir.mkdir()
            pack_path = root / "mini.basalt"
            undo_dir = root / "undo"

            # Minimal ernie55 .basalt pack
            meta = {
                "arch": "ernie55",
                "gguf_kv": {
                    "ernie55.embedding_length": H,
                    "ernie55.vocab_size": V,
                    "ernie55.block_count": 3,
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
                "model_type": "ernie5_5_moe",
                "hidden_size": H,
                "vocab_size": V,
                "num_hidden_layers": 3,
                "num_attention_heads": NH,
                "n_routed_experts": N_EXP,
                "moe_intermediate_size": FF_EXP,
                "moe_latent_size": LAT,
                "sliding_window_size": 128,
                "rope_theta": 8000000.0,
                "num_dspark_layers": N_STAGES,
                "dspark_block_size": 7,
                "dspark_noise_token_id": 1024,
                "dspark_markov_rank": MARKOV_RANK,
                "dspark_target_layer_ids": TARGET_LAYERS,
                "quantization_config": {"weight_block_size": [BN, BK], "expert_dtype": "mxfp4"},
            }
            (ck_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

            tensors: dict[str, tuple[str, list[int], bytes]] = {}

            def add_fp8(name: str, rows: int, cols: int):
                codes = rng.integers(0, 120, size=(rows, cols), dtype=np.uint8)
                sr = (rows + BN - 1) // BN
                sc = (cols + BK - 1) // BK
                scales = rng.uniform(0.05, 0.25, size=(sr, sc)).astype("<f4")
                tensors[name] = ("F8_E4M3", [rows, cols], codes.tobytes())
                tensors[name + "_scale_inv"] = ("F32", [sr, sc], scales.tobytes())
                return codes, scales

            def add_bf16(name: str, shape: list[int]):
                x = rng.normal(0.0, 0.5, size=shape).astype("<f4")
                raw = _f32_to_bf16_bytes(x)
                tensors[name] = ("BF16", shape, raw)
                return np.frombuffer(raw, dtype="<u2").reshape(shape).copy()

            def add_f32(name: str, shape: list[int]):
                x = rng.normal(0.0, 0.5, size=shape).astype("<f4")
                tensors[name] = ("F32", shape, x.tobytes())
                return x

            add_fp8("model.mtp.feature_projection.weight", H, len(TARGET_LAYERS) * H)
            add_bf16("model.mtp.feature_norm.weight", [H])
            hc_head_fn = add_f32("model.mtp.mhc_contract.hc_head_fn", [4 * H, 4])
            add_f32("model.mtp.mhc_contract.hc_head_base", [4])
            add_f32("model.mtp.mhc_contract.hc_head_scale", [1])
            add_bf16("model.mtp.norm.weight", [H])
            w1_u16 = add_bf16("model.mtp.markov_head.markov_w1.weight", [MARKOV_RANK, V])
            w2_u16 = add_bf16("model.mtp.markov_head.markov_w2.weight", [V, MARKOV_RANK])
            conf_u16 = add_bf16("model.mtp.confidence_head.projection.weight", [1, H + MARKOV_RANK])

            saved_exp0 = None
            for s in range(N_STAGES):
                p = f"model.mtp.{s}"
                add_bf16(f"{p}.input_layernorm.weight", [H])
                add_bf16(f"{p}.post_attention_layernorm.weight", [H])
                for hc in ("self_attention_hyper_connection", "mlp_hyper_connection"):
                    add_f32(f"{p}.{hc}.mapping_proj.weight", [4 * H, 24])
                    add_f32(f"{p}.{hc}.bias", [24])
                    for a in ("pre", "post", "res"):
                        add_f32(f"{p}.{hc}.alpha_{a}", [1])
                add_fp8(f"{p}.self_attn.gate_proj.weight", O_GROUPS * O_LORA, Q_LORA)
                add_fp8(f"{p}.self_attn.linear_q_down_proj.weight", Q_LORA, H)
                add_bf16(f"{p}.self_attn.q_layernorm.weight", [Q_LORA])
                add_fp8(f"{p}.self_attn.linear_q_up_proj.weight", NH * HD, Q_LORA)
                add_fp8(f"{p}.self_attn.linear_kv_proj.weight", HD, H)
                add_bf16(f"{p}.self_attn.kv_layernorm.weight", [HD])
                add_fp8(f"{p}.self_attn.linear_o_group_proj", O_GROUPS * O_LORA, (NH // O_GROUPS) * HD)
                add_fp8(f"{p}.self_attn.o_proj.weight", H, O_GROUPS * O_LORA)
                add_f32(f"{p}.self_attn.core_attention.attn_sink", [NH])
                add_bf16(f"{p}.self_attn.vha_postmix_U", [NH, 2])
                add_bf16(f"{p}.self_attn.vha_postmix_V", [NH, 2])
                add_f32(f"{p}.block_sparse_moe.e_score_correction_bias", [N_EXP])
                add_f32(f"{p}.block_sparse_moe.gate.routed_scaling_factor_param", [N_EXP])
                add_bf16(f"{p}.block_sparse_moe.gate.weight", [N_EXP, H])
                add_bf16(f"{p}.block_sparse_moe.gate.weight_1", [N_EXP, H])
                add_fp8(f"{p}.block_sparse_moe.fc1_latent_proj.weight", LAT, H)
                add_bf16(f"{p}.block_sparse_moe.latent_norm.weight", [LAT])
                add_fp8(f"{p}.block_sparse_moe.fc2_latent_proj.weight", H, LAT)
                add_fp8(f"{p}.block_sparse_moe.shared_experts.w1.weight", FF_EXP, H)
                add_fp8(f"{p}.block_sparse_moe.shared_experts.w3.weight", FF_EXP, H)
                add_fp8(f"{p}.block_sparse_moe.shared_experts.w2.weight", H, FF_EXP)

                for e in range(N_EXP):
                    for role, (r, c) in (("w1", (FF_EXP, LAT)), ("w3", (FF_EXP, LAT)), ("w2", (LAT, FF_EXP))):
                        codes = rng.integers(0, 256, size=(r, c // 2), dtype=np.uint8)
                        scales = rng.integers(120, 134, size=(r, c // 32), dtype=np.uint8)
                        base = f"{p}.block_sparse_moe.experts.{e}.{role}"
                        tensors[f"{base}.weight"] = ("I8", [r, c // 2], codes.tobytes())
                        tensors[f"{base}.weight_scale_inv"] = ("F8_E8M0", [r, c // 32], scales.tobytes())
                        if s == 0 and e == 0 and role == "w1":
                            saved_exp0 = (codes.copy(), scales.copy())

            _write_safetensors(ck_dir / "model-00001-of-00001.safetensors", tensors)

            dmeta = D.attach_dspark(ck_dir, pack_path, undo_dir=undo_dir, workers=2)
            self.assertEqual(dmeta["layers"], N_STAGES)
            self.assertEqual(dmeta["block_size"], 7)
            self.assertEqual(dmeta["noise_token_id"], 1024)
            self.assertEqual(dmeta["markov_rank"], MARKOV_RANK)
            self.assertEqual(dmeta["target_layers"], TARGET_LAYERS)

            mf = ModelFile(str(pack_path))
            self.assertIn("dspark", mf.names())
            self.assertTrue(mf.verify("dspark"))
            self.assertTrue(mf.verify("meta"))
            blob = mf.read("dspark")

            # Verify markov_w1 was transposed [rank, vocab] -> [vocab, rank]
            mw1_m = dmeta["tensors"]["markov_w1.weight"]
            self.assertEqual(mw1_m["shape"], [MARKOV_RANK, V])
            mw1_got = np.frombuffer(blob[mw1_m["offset"]:mw1_m["offset"] + mw1_m["bytes"]], dtype="<u2").reshape(V, MARKOV_RANK)
            np.testing.assert_array_equal(mw1_got, w1_u16.T)

            # Verify markov_w2 Q8_0 dequant matches original BF16 closely
            mw2_m = dmeta["tensors"]["markov_w2.weight"]
            mw2_got = _dequant_q8_0(blob[mw2_m["offset"]:mw2_m["offset"] + mw2_m["bytes"]], V, MARKOV_RANK)
            mw2_ref = S._bf16_to_f32(w2_u16)
            self.assertLess(float(np.max(np.abs(mw2_got - mw2_ref))), 0.03)

            # Verify hc_head_fn transpose
            hcf_m = dmeta["tensors"]["hc_head_fn.weight"]
            hcf_got = np.frombuffer(blob[hcf_m["offset"]:hcf_m["offset"] + hcf_m["bytes"]], dtype="<f4").reshape(4, 4 * H)
            np.testing.assert_allclose(hcf_got, hc_head_fn.T, atol=0.0)

            # Verify conf_proj squeezed to 1D
            cp_m = dmeta["tensors"]["conf_proj.weight"]
            self.assertEqual(cp_m["shape"], [H + MARKOV_RANK])
            cp_got = np.frombuffer(blob[cp_m["offset"]:cp_m["offset"] + cp_m["bytes"]], dtype="<f4")
            np.testing.assert_allclose(cp_got, S._bf16_to_f32(conf_u16).reshape(-1), atol=0.0)

            # Verify first expert MXFP4 w1 bytes match bit-exact
            eoff = dmeta["experts"]["offset"]
            w1_bytes = FF_EXP * (LAT // 32) * 17
            exp0_got = np.frombuffer(blob[eoff:eoff + w1_bytes], dtype=np.uint8)
            exp0_ref = E.mxfp4_rows(saved_exp0[0], saved_exp0[1])
            np.testing.assert_array_equal(exp0_got, exp0_ref)

            # Verify rollback restores original file bit-for-bit
            undo_append(str(pack_path), str(undo_dir))
            self.assertEqual(pack_path.read_bytes(), orig_bytes)


if __name__ == "__main__":
    unittest.main()
