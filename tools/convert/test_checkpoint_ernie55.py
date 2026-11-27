"""End-to-end tests for tools/convert/checkpoint_ernie55.py: builds a synthetic mini ERNIE-5.5-Flash checkpoint
(Layer 0 dense HCA, Layer 1 MoE HCA, Layer 2 MoE MLA-DSA, optional FlashIndexer, skipped MTP, tokenizer), runs
convert.py --checkpoint, and checks every section, tensor shape/type, bit-exact MXFP4 repacking, blockwise FP8 -> Q8_0
(including partial 128-row blocks), mHC transpose + alpha stacking, VHA fold, and split-feature router concat.
"""
from __future__ import annotations

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

import checkpoint_ernie55 as E  # noqa: E402
import model_sections as S  # noqa: E402
from model_file import ModelFile  # noqa: E402


def write_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """Write a safetensors file from {name: (safetensors_dtype, raw_ndarray)}."""
    header = {}
    body = bytearray()
    for name, (dt, arr) in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        off = len(body)
        body.extend(raw)
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
    hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        f.write(body)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(x, dtype=np.float32).view(np.uint32) >> 16).astype("<u2")


def dequant_q8_0(raw: bytes, cols: int) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, cols // 32, 34)
    d = np.frombuffer(b[..., :2].tobytes(), dtype="<f2").astype(np.float32).reshape(b.shape[0], b.shape[1], 1)
    q = b[..., 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(-1, cols)


def make_fp8_pair(rng: np.random.Generator, rows: int, cols: int, block=(128, 128)):
    # Avoid E4M3 NaN bytes (0x7F, 0xFF)
    codes = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x38
    bn, bk = block
    sn = (rows + bn - 1) // bn
    sk = (cols + bk - 1) // bk
    exps = rng.integers(-6, 1, size=(sn, sk))
    scales = np.ldexp(np.ones((sn, sk), dtype=np.float32), exps).astype("<f4")
    return codes, scales


def make_mini_checkpoint(root: pathlib.Path, rng: np.random.Generator):
    H = 256
    FF_DENSE = 512
    LATENT = 128
    FF_EXP = 128
    N_EXP = 4
    VOCAB = 64
    NH = 4
    HD = 64
    Q_LORA = 64
    KV_LORA = 64
    O_GROUPS = 2
    O_LORA = 64
    MLA_NOPE = 32
    MLA_ROPE = 32
    MLA_V = 64
    IDX_H = 4
    IDX_D = 32
    WIN = 16

    cfg = {
        "architectures": ["Ernie5_5_MoeForCausalLM"],
        "model_type": "ernie5_5_moe",
        "num_hidden_layers": 3,
        "first_k_dense_replace": 1,
        "compress_ratios": [128, 128, -2],
        "hidden_size": H,
        "intermediate_size": FF_DENSE,
        "vocab_size": VOCAB,
        "max_position_embeddings": 1048576,
        "rms_norm_eps": 1e-6,
        "sliding_window": WIN,
        "num_attention_heads": NH,
        "num_key_value_heads": 1,
        "head_dim": HD,
        "v_head_dim": HD,
        "q_lora_rank": Q_LORA,
        "kv_lora_rank": KV_LORA,
        "o_groups": O_GROUPS,
        "o_lora_rank": O_LORA,
        "qk_rope_head_dim": 32,
        "rope_theta": 8000000.0,
        "compress_rope_theta": 160000.0,
        "rope_scaling": {"factor": 16, "original_max_position_embeddings": 65536, "beta_fast": 32, "beta_slow": 1, "type": "yarn"},
        "hybrid_mla_num_attention_heads": NH,
        "hybrid_mla_q_lora_rank": Q_LORA,
        "hybrid_mla_kv_lora_rank": KV_LORA,
        "hybrid_mla_qk_nope_head_dim": MLA_NOPE,
        "hybrid_mla_qk_rope_head_dim": MLA_ROPE,
        "hybrid_mla_v_head_dim": MLA_V,
        "index_n_heads": IDX_H,
        "index_head_dim": IDX_D,
        "index_topk": 16,
        "n_routed_experts": N_EXP,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_intermediate_size": FF_EXP,
        "moe_latent_size": LATENT,
        "norm_topk_prob": True,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "enable_hyper_connections": True,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "moe_split_feature_routing": True,
        "scoring_func": "sigmoid",
        "bos_token_id": 1,
        "eos_token_id": 8,
        "pad_token_id": 0,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "expert_dtype": "mxfp4",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (root / "generation_config.json").write_text(json.dumps({"temperature": 1.0, "top_p": 0.95, "eos_token_id": [2, 8]}), encoding="utf-8")
    (root / "README.md").write_text("---\nlicense: apache-2.0\n---\n# mini ernie55\n", encoding="utf-8")
    vocab = {f"tok{i}": i for i in range(VOCAB)}
    added = [
        {"id": 0, "content": "<|ERNIE_PAD|>", "special": True},
        {"id": 1, "content": "<|ERNIE_BEGIN|>", "special": True},
        {"id": 7, "content": "<|ernie:im_start|>", "special": True},
        {"id": 8, "content": "<|ernie:im_end|>", "special": True},
    ]
    for a in added:
        vocab[a["content"]] = a["id"]
    (root / "tokenizer.json").write_text(json.dumps({
        "version": "1.0",
        "added_tokens": added,
        "model": {"type": "BPE", "vocab": vocab, "merges": ["tok10 tok11"]},
    }), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<|ERNIE_BEGIN|>", "eos_token": "<|ernie:im_end|>", "pad_token": "<|ERNIE_PAD|>",
    }), encoding="utf-8")

    ref = {}
    t: dict[str, tuple[str, np.ndarray]] = {}

    def add_fp8(name: str, rows: int, cols: int):
        c, s = make_fp8_pair(rng, rows, cols)
        t[name] = ("F8_E4M3", c)
        t[name + "_scale_inv"] = ("F32", s)
        ref[name] = E.fp8_block_dequant(c, s, (128, 128))

    def add_bf16(name: str, shape):
        x = rng.normal(0, 0.25, size=shape).astype(np.float32)
        b = f32_to_bf16(x)
        t[name] = ("BF16", b)
        ref[name] = S._bf16_to_f32(b)

    def add_f32(name: str, shape):
        x = rng.normal(0, 0.25, size=shape).astype("<f4")
        t[name] = ("F32", x)
        ref[name] = x

    add_bf16("model.embed_tokens.weight", (VOCAB, H))
    add_bf16("lm_head.weight", (VOCAB, H))
    add_bf16("model.norm.weight", (H,))
    add_bf16("model.lm_head.multimax_ranges", (4,))
    add_bf16("model.lm_head.multimax_ts", (4,))
    add_f32("model.mhc_contract.hc_head_fn", (4 * H, 4))
    add_f32("model.mhc_contract.hc_head_base", (4,))
    add_f32("model.mhc_contract.hc_head_scale", (1,))
    # Dummy MTP tensor (must be ignored cleanly by Stage 1 converter)
    add_bf16("model.mtp.norm.weight", (H,))

    for l in range(3):
        p = f"model.layers.{l}."
        is_dsa = cfg["compress_ratios"][l] <= 0
        add_bf16(p + "input_layernorm.weight", (H,))
        add_bf16(p + "post_attention_layernorm.weight", (H,))
        for hc in ("self_attention_hyper_connection", "mlp_hyper_connection"):
            add_f32(p + f"{hc}.mapping_proj.weight", (4 * H, 24))
            add_f32(p + f"{hc}.bias", (24,))
            for a in ("pre", "post", "res"):
                add_f32(p + f"{hc}.alpha_{a}", (1,))
        add_bf16(p + "self_attn.vha_postmix_U", (NH, 4))
        add_bf16(p + "self_attn.vha_postmix_V", (NH, 4))

        if not is_dsa:
            add_fp8(p + "self_attn.linear_q_down_proj.weight", Q_LORA, H)
            add_bf16(p + "self_attn.q_layernorm.weight", (Q_LORA,))
            add_fp8(p + "self_attn.linear_q_up_proj.weight", NH * HD, Q_LORA)
            add_fp8(p + "self_attn.linear_kv_proj.weight", KV_LORA, H)
            add_bf16(p + "self_attn.kv_layernorm.weight", (KV_LORA,))
            add_fp8(p + "self_attn.linear_o_group_proj", O_GROUPS * O_LORA, NH * HD // O_GROUPS)
            add_fp8(p + "self_attn.gate_proj.weight", O_GROUPS * O_LORA, Q_LORA)
            add_fp8(p + "self_attn.o_proj.weight", H, O_GROUPS * O_LORA)
            add_f32(p + "self_attn.core_attention.attn_sink", (NH,))
            add_f32(p + "self_attn.core_attention.compressor.ape", (WIN, KV_LORA))
            add_bf16(p + "self_attn.core_attention.compressor.linear_wkv.weight", (KV_LORA, H))
            add_bf16(p + "self_attn.core_attention.compressor.linear_wgate.weight", (KV_LORA, H))
            add_bf16(p + "self_attn.core_attention.compressor.norm.weight", (KV_LORA,))
        else:
            add_fp8(p + "self_attn.q_a_proj.weight", Q_LORA, H)
            add_bf16(p + "self_attn.q_a_layernorm.weight", (Q_LORA,))
            add_fp8(p + "self_attn.q_b_proj.weight", NH * (MLA_NOPE + MLA_ROPE), Q_LORA)
            # Partial 128-row block: 96 rows (KV_LORA + MLA_ROPE = 64 + 32 = 96) with 128x128 scale [1, 2]
            add_fp8(p + "self_attn.kv_a_proj_with_mqa.weight", KV_LORA + MLA_ROPE, H)
            add_bf16(p + "self_attn.kv_a_layernorm.weight", (KV_LORA,))
            add_fp8(p + "self_attn.kv_b_proj.weight", NH * (MLA_NOPE + MLA_V), KV_LORA)
            add_fp8(p + "self_attn.gate_proj.weight", NH * MLA_V, Q_LORA)
            add_fp8(p + "self_attn.o_proj.weight", H, NH * MLA_V)
            add_bf16(p + "self_attn.core_attention.softmax_offset", (NH,))
            add_fp8(p + "self_attn.core_attention.indexer.wq_b.weight", IDX_H * IDX_D, Q_LORA)
            add_bf16(p + "self_attn.core_attention.indexer.wk.weight", (IDX_D, H))
            add_bf16(p + "self_attn.core_attention.indexer.k_norm.weight", (IDX_D,))
            add_bf16(p + "self_attn.core_attention.indexer.k_norm.bias", (IDX_D,))
            add_bf16(p + "self_attn.core_attention.indexer.weights_proj.weight", (IDX_H, H))

        if l < 1:
            add_fp8(p + "mlp.gate_proj.weight", FF_DENSE, H)
            add_fp8(p + "mlp.up_proj.weight", FF_DENSE, H)
            add_fp8(p + "mlp.down_proj.weight", H, FF_DENSE)
        else:
            add_bf16(p + "block_sparse_moe.gate.weight", (N_EXP, H))
            add_bf16(p + "block_sparse_moe.gate.weight_1", (N_EXP, H))
            add_f32(p + "block_sparse_moe.e_score_correction_bias", (N_EXP,))
            add_f32(p + "block_sparse_moe.gate.routed_scaling_factor_param", (N_EXP,))
            add_fp8(p + "block_sparse_moe.fc1_latent_proj.weight", LATENT, H)
            add_bf16(p + "block_sparse_moe.latent_norm.weight", (LATENT,))
            add_fp8(p + "block_sparse_moe.fc2_latent_proj.weight", H, LATENT)
            add_fp8(p + "block_sparse_moe.shared_experts.w1.weight", FF_EXP, H)
            add_fp8(p + "block_sparse_moe.shared_experts.w3.weight", FF_EXP, H)
            add_fp8(p + "block_sparse_moe.shared_experts.w2.weight", H, FF_EXP)
            for e in range(N_EXP):
                ep = p + f"block_sparse_moe.experts.{e}."
                for role, rows, cols in (("w1", FF_EXP, LATENT), ("w3", FF_EXP, LATENT), ("w2", LATENT, FF_EXP)):
                    c = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
                    sc = rng.integers(118, 126, size=(rows, cols // 32), dtype=np.uint8)
                    t[ep + f"{role}.weight"] = ("I8", c.view(np.int8))
                    t[ep + f"{role}.weight_scale_inv"] = ("F8_E8M0", sc)
                    ref[ep + role] = (c, sc)

    write_safetensors(root / "model-00001-of-00001.safetensors", t)
    # Optional FlashIndexer subfolder for DSA layer 2
    fi_t = {}
    fg = f32_to_bf16(rng.normal(0, 0.25, size=(IDX_H, H)).astype(np.float32))
    fb = f32_to_bf16(rng.normal(0, 0.25, size=(IDX_H,)).astype(np.float32))
    fi_t["model.layers.2.self_attn.core_attention.indexer.linear_gate.weight"] = ("BF16", fg)
    fi_t["model.layers.2.self_attn.core_attention.indexer.flashindexer_bias"] = ("BF16", fb)
    write_safetensors(root / "flashindexer" / "flashindexer.safetensors", fi_t)
    return cfg, ref


class TestCheckpointErnie55(unittest.TestCase):
    def test_mxfp4_repack_bit_exact(self):
        rng = np.random.default_rng(7)
        rows, cols = 17, 128
        codes = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
        scales = rng.integers(100, 140, size=(rows, cols // 32), dtype=np.uint8)
        packed = E.mxfp4_rows(codes, scales)
        got = E.mxfp4_dequant(packed, cols)
        # Reference directly from HF paired nibbles
        nib = np.empty((rows, cols), dtype=np.uint8)
        nib[:, 0::2] = codes & 0x0F
        nib[:, 1::2] = codes >> 4
        sc = np.repeat(np.ldexp(np.float32(1.0), scales.astype(np.int32) - 127), 32, axis=1)
        want = E.MXFP4_VALUES[nib] * sc
        np.testing.assert_array_equal(got, want)

    def test_end_to_end_convert_and_verify(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "ckpt"
            out = pathlib.Path(td) / "mini-ernie55.basalt"
            root.mkdir()
            _, ref = make_mini_checkpoint(root, rng)

            proc = subprocess.run(
                [sys.executable, str(HERE / "convert.py"), "--checkpoint", str(root), "--out", str(out), "--workers", "2"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

            mf = ModelFile(str(out))
            meta = json.loads(mf.read("meta").decode("utf-8"))
            self.assertEqual(meta["arch"], "ernie55")
            kv = meta["gguf_kv"]
            self.assertEqual(kv["general.architecture"], "ernie55")
            self.assertEqual(kv["ernie55.block_count"], 3)
            self.assertEqual(kv["ernie55.attention.compress_ratios"], [128, 128, -2])
            self.assertEqual(kv["ernie55.expert_gate_softcap"], 4.0)
            self.assertEqual(kv["ernie55.expert_up_softcap"], 25.0)
            self.assertTrue(kv["ernie55.has_flashindexer"])
            self.assertEqual(kv["tokenizer.ggml.pre"], "ernie55")
            self.assertEqual(kv["tokenizer.ggml.eos_token_id"], 8)

            # Check experts.layout
            layout_lines = [ln for ln in mf.read("experts.layout").decode().splitlines() if not ln.startswith("#")]
            self.assertEqual(len(layout_lines), 2)   # layers 1 and 2
            blob_bytes = (2 * 128 + 128) * (128 // 32 * 17)
            self.assertEqual(layout_lines[0], f"1 39 39 0 {blob_bytes}")
            self.assertEqual(layout_lines[1], f"2 39 39 {4 * blob_bytes} {blob_bytes}")

            # Verify bit-exact MXFP4 expert payload in pack
            exp_raw = mf.read("experts")
            c_w1, s_w1 = ref["model.layers.1.block_sparse_moe.experts.0.w1"]
            got_w1 = E.mxfp4_dequant(np.frombuffer(exp_raw[:128 * 4 * 17], dtype=np.uint8), 128)
            want_w1 = E.mxfp4_dequant(E.mxfp4_rows(c_w1, s_w1), 128)
            np.testing.assert_array_equal(got_w1, want_w1)

            # Verify gguf tensors: shapes, types, transforms, and numerical fidelity
            tens = meta["tensors"]
            gguf_sec_off = mf.sections["gguf"][0]

            def read_tensor(name: str) -> bytes:
                info = tens[name]
                with open(out, "rb") as f:
                    f.seek(gguf_sec_off + info["offset"])
                    return f.read(info["bytes"])

            # 1. Partial-block 128x128 FP8 -> Q8_0 (kv_a_proj_with_mqa: 96 x 256)
            kv_a_got = dequant_q8_0(read_tensor("blk.2.attn_kv_a_mqa.weight"), 256)
            kv_a_ref = ref["model.layers.2.self_attn.kv_a_proj_with_mqa.weight"]
            rel = np.linalg.norm(kv_a_got - kv_a_ref) / np.linalg.norm(kv_a_ref)
            self.assertLess(rel, 1e-2)

            # 2. mHC transpose: [4*H, 24] -> [24, 4*H]
            hc_got = np.frombuffer(read_tensor("blk.0.hc_attn_fn.weight"), dtype="<f4").reshape(24, 4 * 256)
            np.testing.assert_array_equal(hc_got, ref["model.layers.0.self_attention_hyper_connection.mapping_proj.weight"].T)

            # 3. mHC alpha stack: [pre, post, res]
            al_got = np.frombuffer(read_tensor("blk.0.hc_attn_scale.weight"), dtype="<f4")
            al_ref = np.concatenate([
                ref["model.layers.0.self_attention_hyper_connection.alpha_pre"],
                ref["model.layers.0.self_attention_hyper_connection.alpha_post"],
                ref["model.layers.0.self_attention_hyper_connection.alpha_res"],
            ])
            np.testing.assert_array_equal(al_got, al_ref)

            # 4. VHA fold: I + V @ U^T
            vha_got = np.frombuffer(read_tensor("blk.1.attn_vha_mix.weight"), dtype="<f4").reshape(4, 4)
            u = ref["model.layers.1.self_attn.vha_postmix_U"]
            v = ref["model.layers.1.self_attn.vha_postmix_V"]
            np.testing.assert_allclose(vha_got, np.eye(4, dtype=np.float32) + v @ u.T, rtol=1e-6, atol=1e-6)

            # 5. Split-feature router concat: [gate.weight; gate.weight_1]
            r_got = np.frombuffer(read_tensor("blk.1.ffn_gate_inp.weight"), dtype="<f4").reshape(8, 256)
            r_ref = np.concatenate([
                ref["model.layers.1.block_sparse_moe.gate.weight"],
                ref["model.layers.1.block_sparse_moe.gate.weight_1"],
            ], axis=0)
            np.testing.assert_array_equal(r_got, r_ref)

            # 6. FlashIndexer tensors present on DSA layer 2
            self.assertIn("blk.2.indexer.flash_gate.weight", tens)
            self.assertIn("blk.2.indexer.flash_bias.weight", tens)
            self.assertEqual(tens["output.weight"]["type"], "BF16")
            self.assertEqual(tens["token_embd.weight"]["type"], "BF16")

    def test_missing_required_config_keys_fail_loudly(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as td:
            ckpt = pathlib.Path(td) / "ckpt"
            ckpt.mkdir()
            make_mini_checkpoint(ckpt, rng)
            orig_cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
            for key in ("first_k_dense_replace", "activation_situ_beta", "rope_scaling"):
                bad = dict(orig_cfg)
                del bad[key]
                (ckpt / "config.json").write_text(json.dumps(bad), encoding="utf-8")
                ck = E.ErnieCheckpoint(ckpt)
                with self.assertRaises(S.ConvertError) as ctx:
                    E.Experts(ck)
                    E.Plan(ck)
                    E.model_kv(ck, "test")
                self.assertIn(key, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
