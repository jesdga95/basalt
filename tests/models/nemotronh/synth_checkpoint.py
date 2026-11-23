"""A small Nemotron-H checkpoint in the layout of NVIDIA's NVFP4 export (Model Optimizer's mixed precision: FP8
projections and head with a per-tensor scale, NVFP4 routed and shared experts, BF16 the rest), with weights drawn so
the activations stay in range: the engine's end-to-end parity test (model_parity.cpp) converts it and compares the
engine's logits with reference.py's.

Every shape satisfies the kernels the real model runs: 128 SSM heads of 64 (the FP8 product's rows come in 128s),
d_state 128, 8 groups, 128-wide attention heads, a latent and widths in whole blocks.

    python3 tests/models/nemotronh/synth_checkpoint.py OUT_DIR [--seed N]"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools" / "convert"))
sys.path.insert(0, str(HERE.parents[2] / "tools"))
import checkpoint_nemotronh as N  # noqa: E402
import model_sections as S  # noqa: E402
from test_checkpoint import write_safetensors  # noqa: E402

PATTERN = "ME*MEM*E"
H, V = 512, 256                      # hidden, vocabulary
NH, NKV, HD = 4, 2, 128              # attention
MH, MHD, DS, NG, DC = 128, 64, 128, 8, 4
NE, TOPK, LAT, FF, FFS = 32, 4, 256, 128, 256
DI = MH * MHD
CD = DI + 2 * NG * DS
PW = 2 * DI + 2 * NG * DS + MH

E4M3 = S.fp8_e4m3_table()
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float64)
_pos = np.array([E4M3[c] for c in range(0x7F)], np.float64)   # codes 0..126: the finite non-negative values, ascending


def bf16(x: np.ndarray) -> np.ndarray:
    """x rounded to BF16 (nearest even), as float32."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16
    return u.astype(np.uint32).view(np.float32)


def e4m3_codes(x: np.ndarray) -> np.ndarray:
    """Nearest E4M3 codes of x (|x| <= 448)."""
    a = np.minimum(np.abs(x), 448.0)
    i = np.clip(np.searchsorted(_pos, a), 1, len(_pos) - 1)
    lo, hi = _pos[i - 1], _pos[i]
    c = np.where(a - lo <= hi - a, i - 1, i).astype(np.uint8)
    return c | (np.signbit(x).astype(np.uint8) << 7)


class Writer:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.t = {}

    def bf(self, name, x):
        x = bf16(np.asarray(x, np.float32))
        self.t[name] = ("BF16", x.shape, (x.view(np.uint32) >> 16).astype("<u2").tobytes())

    def f32(self, name, x):
        x = np.asarray(x, np.float32)
        self.t[name] = ("F32", x.shape, x.astype("<f4").tobytes())

    def fp8(self, base, w):
        s = float(np.abs(w).max()) / 448.0
        codes = e4m3_codes(w / s)
        self.t[base + ".weight"] = ("F8_E4M3", w.shape, codes.tobytes())
        self.t[base + ".weight_scale"] = ("F32", (), np.array(s, np.float32).tobytes())
        self.t[base + ".input_scale"] = ("F32", (1,), np.array([1.0], np.float32).tobytes())

    def nvfp4(self, base, w):
        rows, cols = w.shape
        s2 = np.float32(np.abs(w).max() / (6.0 * 448.0))
        blocks = w.reshape(rows, cols // 16, 16)
        bs = np.abs(blocks).max(axis=2) / 6.0 / np.float64(s2)
        sc = e4m3_codes(np.maximum(bs, _pos[1]))                    # positive UE4M3, at least the smallest subnormal
        scale = (E4M3[sc].astype(np.float64) * np.float64(s2))[..., None]
        v = blocks / scale
        mag = np.abs(v)
        i = np.argmin(np.abs(mag[..., None] - E2M1), axis=-1).astype(np.uint8)
        nib = (i | (np.signbit(v).astype(np.uint8) << 3)).reshape(rows, cols)
        codes = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
        self.t[base + ".weight"] = ("U8", (rows, cols // 2), codes.tobytes())
        self.t[base + ".weight_scale"] = ("F8_E4M3", (rows, cols // 16), sc.astype(np.uint8).tobytes())
        self.t[base + ".weight_scale_2"] = ("F32", (), np.array(s2, np.float32).tobytes())
        self.t[base + ".input_scale"] = ("F32", (), np.array(1.0, np.float32).tobytes())

    def normal(self, shape, std):
        return self.rng.standard_normal(shape) * std


def write(out: pathlib.Path, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    w = Writer(seed)
    rng = w.rng
    B = "backbone."
    w.bf(B + "embeddings.weight", w.normal((V, H), 1.0))
    w.bf(B + "norm_f.weight", rng.uniform(0.5, 1.5, H))
    w.fp8("lm_head", w.normal((V, H), 1.0 / np.sqrt(H)))
    for l, c in enumerate(PATTERN):
        p = f"{B}layers.{l}."
        w.bf(p + "norm.weight", rng.uniform(0.5, 1.5, H))
        m = p + "mixer."
        if c == "M":
            w.fp8(m + "in_proj", w.normal((PW, H), 1.0 / np.sqrt(H)))
            w.bf(m + "conv1d.weight", w.normal((CD, 1, DC), 0.5))
            w.bf(m + "conv1d.bias", w.normal((CD,), 0.1))
            w.bf(m + "dt_bias", rng.uniform(-4.0, -1.0, MH))
            w.bf(m + "A_log", np.log(rng.uniform(1.0, 16.0, MH)))
            w.bf(m + "D", rng.uniform(0.5, 1.5, MH))
            w.bf(m + "norm.weight", rng.uniform(0.5, 1.5, DI))
            w.fp8(m + "out_proj", w.normal((H, DI), 0.5 / np.sqrt(DI)))
        elif c == "*":
            for n, rows in (("q_proj", NH * HD), ("k_proj", NKV * HD), ("v_proj", NKV * HD)):
                w.fp8(m + n, w.normal((rows, H), 1.0 / np.sqrt(H)))
            w.fp8(m + "o_proj", w.normal((H, NH * HD), 0.5 / np.sqrt(NH * HD)))
        else:
            w.bf(m + "gate.weight", w.normal((NE, H), 1.0 / np.sqrt(H)))
            w.f32(m + "gate.e_score_correction_bias", w.normal((NE,), 0.02))
            w.bf(m + "fc1_latent_proj.weight", w.normal((LAT, H), 1.0 / np.sqrt(H)))
            w.bf(m + "fc2_latent_proj.weight", w.normal((H, LAT), 0.5 / np.sqrt(LAT)))
            w.nvfp4(m + "shared_experts.up_proj", w.normal((FFS, H), 1.0 / np.sqrt(H)))
            w.nvfp4(m + "shared_experts.down_proj", w.normal((H, FFS), 0.5 / np.sqrt(FFS)))
            for e in range(NE):
                w.nvfp4(m + f"experts.{e}.up_proj", w.normal((FF, LAT), 1.0 / np.sqrt(LAT)))
                w.nvfp4(m + f"experts.{e}.down_proj", w.normal((LAT, FF), 0.5 / np.sqrt(FF)))
    names = list(w.t)
    half = len(names) // 2
    write_safetensors(out / "model-00001-of-00002.safetensors", {k: w.t[k] for k in names[:half]})
    write_safetensors(out / "model-00002-of-00002.safetensors", {k: w.t[k] for k in names[half:]})
    cfg = {"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h", "num_hidden_layers": len(PATTERN),
           "hybrid_override_pattern": PATTERN, "hidden_size": H, "vocab_size": V, "num_attention_heads": NH,
           "num_key_value_heads": NKV, "head_dim": HD, "mamba_num_heads": MH, "mamba_head_dim": MHD, "ssm_state_size": DS,
           "n_groups": NG, "conv_kernel": DC, "n_routed_experts": NE, "num_experts_per_tok": TOPK, "moe_intermediate_size": FF,
           "moe_shared_expert_intermediate_size": FFS, "moe_latent_size": LAT, "mamba_hidden_act": "silu",
           "mlp_hidden_act": "relu2", "use_conv_bias": True, "mamba_proj_bias": False, "use_bias": False,
           "attention_bias": False, "mlp_bias": False, "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
           "tie_word_embeddings": False, "residual_in_fp32": False, "n_shared_experts": 1, "sliding_window": None,
           "routed_scaling_factor": 2.5, "layer_norm_epsilon": 1e-5, "max_position_embeddings": 262144}
    (out / "config.json").write_text(json.dumps(cfg))
    (out / "generation_config.json").write_text(json.dumps({"temperature": 1.0, "top_p": 0.95}))
    (out / "README.md").write_text("---\nlicense: other\nlicense_name: synthetic\n---\n# a synthetic Nemotron-H\n")
    (out / "chat_template.jinja").write_text("{{ messages }}")
    specials = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    plain = [chr(c) for c in range(0x21, 0x7F)] + [chr(c) for c in range(0xA1, 0xAD)] + [chr(c) for c in range(0xAE, 0x100)]
    plain = plain[:V - len(specials)]
    vocab = {t: i for i, t in enumerate(plain)}
    (out / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": N.NEMOTRONH_HF_SPLIT}}, {"type": "ByteLevel"}]},
        "added_tokens": [{"id": len(plain) + i, "content": t, "special": True} for i, t in enumerate(specials)]}))
    (out / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>", "eos_token": "<|im_end|>"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    write(pathlib.Path(a.out), a.seed)


if __name__ == "__main__":
    main()
