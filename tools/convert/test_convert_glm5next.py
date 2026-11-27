"""A tiny synthetic glm5-next GGUF converted end to end: sections, meta, the expert layout that skips the dense
layer, the drafter block's own sections, and what the converter refuses."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import convert  # noqa: E402
from arch import glm5next  # noqa: E402
import ggufq  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
import iq_fixture  # noqa: E402
from model_file import SECTION_ALIGN, ModelFile  # noqa: E402

N_EXPERT = 32             # expert_count: the router takes a multiple of 32 and throws otherwise
TOKENS = ["a", "b", "c"]
# The tiny model's geometry, as the GGUF below declares it. The placeholder tensors `gguf()` write stand in for these
# with a fixed block-sized blob; `gguf_runnable()` writes the real shapes instead, so a forward pass can run on them.
#
# The numbers are not free, and four separate kernels enforce it. f32_rows_dot (the engine's fp32 projection) wants
# every input width a multiple of 16; mla_sparse fixes the latent at 512 and takes heads 8 at a time; and router()
# throws "unsupported geometry" unless n_expert is a multiple of 32 and n_embd a multiple of 128. The expert tier's
# native formats want a whole quantization block per expert row: 256 values on both sides of the gate/up product and
# 256 on the down product's, which is what sets E and N_FF_EXP. So: 256-wide embeddings and expert hidden widths,
# 32 experts, a 512-wide latent, 8 heads, and every other width a multiple of 16.
E = 256                   # embedding_length: the router wants a multiple of 128, the i-quant expert rows a whole
                          # 256-value block (native_expert_supported: n_embd % qg == 0 && n_embd % 256 == 0)
VOCAB = 3                 # vocab_size (the token list below)
HEAD = 8                  # attention.head_count: mla_sparse wants 8 or 16 divisible
HD = 16                   # kda.head_dim
D_INNER = HEAD * HD       # a KDA block's q/k/v width
CONV = 4                  # ssm.conv_kernel
Q_LORA = 16               # attention.q_lora_rank
KV_LORA = 512             # attention.kv_lora_rank: mla_sparse's latent width, fixed
QK_MLA = 16               # attention.key_length_mla: the MLA head key width (rope dimension_count is 0, so all nope)
V_MLA = 16                # attention.value_length_mla: the MLA head value width
QK_LEN = 512              # attention.key_length: the checkpoint's own length, the latent's (512 in the release). The
V_LEN = 512               # attention.value_length: same. Neither is the head width; the weights want *_length_mla.
IDX_HEAD = 2              # attention.indexer.head_count
IDX_DIM = 16              # attention.indexer.key_length
KPOOL = 2                 # attention.indexer.kpool
N_FF = 32                 # feed_forward_length: the dense lead MLP's hidden width
N_FF_EXP = 256            # expert_feed_forward_length: a routed expert's hidden width, and a down row is quantized
                          # whole (n_ff % qd == 0), so a multiple of the 256-value i-quant block
N_FF_SHEXP = 48           # expert_shared_feed_forward_length: a shared expert's, apart from the routed ones (and a
                          # multiple of 16, the width the fp32 products take)
CLAMP_EXP = 10.0          # swiglu_clamp_exp: the routed experts' SwiGLU clamp, per block
CLAMP_SHEXP = 8.0         # swiglu_clamp_shexp: the shared experts' clamp (a different value, so the two are not confused)
LN_EPS = 1e-6             # attention.layer_norm_epsilon: the indexer key LayerNorm's epsilon
N_SHARED = 1              # expert_shared_count
HC = 4                    # hyper_connection.count
HC_MIX = (2 + HC) * HC    # the mHC mixes: hc_attn_fn's rows and hc_attn_base's length
# ggml block geometry (values, bytes) per type, as gguf_reader.py spells it
BLOCK = {"IQ1_S": (256, 50), "IQ2_XXS": (256, 66), "IQ3_XXS": (256, 98), "Q5_K": (256, 176), "Q6_K": (256, 210),
         "Q8_0": (32, 34)}
KINDS = [0, 0, 1, 1]          # linear-attention, linear-attention, sparse MLA, sparse MLA (the drafter)
N_LAYERS = len(KINDS)
DENSE_LAYERS = 1              # blk.0 has a dense MLP, blk.1 the first with routed experts
MTP = N_LAYERS - 1
GU = {1: "IQ1_S", 2: "IQ2_XXS", MTP: "IQ1_S"}
DOWN = {1: "IQ3_XXS", 2: "IQ2_XXS", MTP: "IQ3_XXS"}
# The runnable form's routed experts, at the type mix the release pack carries: IQ2_XXS gate/up with IQ3_XXS down. The
# tier computes experts only in its native formats, so this is the shape a real expert tensor can reach the card in.
IQ_GU, IQ_DOWN = "IQ2_XXS", "IQ3_XXS"
# The RMS iq_fixture's blocks dequantize to when only the fp16 main scale is repaired (measured, seed-independent):
# 53.5 for IQ2_XXS, 137.2 for IQ3_XXS, 1257.8 for IQ4_XS - the sub-scales and indices stay random, so a "sane 1.0"
# block is nothing like a weight. Dividing by this lands the blob at unit RMS, and by sqrt(fan-in) on top of that
# puts it in the same convention as the fixture's f32() matrices, which is the one a running model lives in.
IQ_DEQ_RMS = {IQ_GU: 53.5, IQ_DOWN: 137.2}
# The dense projections a pack can store packed instead of fp32 - 645 of the release pack's 1270 tensors, of which
# 346 are Q8_0. Q8_0 is the type this repo's Python side can encode as well as read (`ggufq` quantizes BF16 and
# dequantizes everything), and it is wide enough to cover what `proj` multiplies: the widths the fixture gives
# ssm_f_b, attn_q_b, attn_k_b and the conv taps (16 values) do not fill the type's 32-value block, so those stay
# fp32 - as do the norms, the biases and the expert shards. The mHC mixes are packed too: the forward pass multiplies
# them as fp32 rows, so a pack that stores them packed has them dequantized at the first window that needs them.
QUANT_DENSE = ("output.weight", "attn_output.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
               "attn_q_a.weight", "attn_kv_a_mqa.weight", "attn_v_b.weight",
               "hc_attn_fn.weight", "hc_ffn_fn.weight",
               "ssm_beta.weight", "ssm_f_a.weight", "ssm_g_a.weight", "indexer.attn_k.weight",
               "indexer.proj.weight", "indexer_compressor_gate.weight", "ffn_gate.weight", "ffn_up.weight",
               "ffn_down.weight")


def q8_0_blocks(values: np.ndarray) -> bytes:
    """GGML Q8_0 blocks over dim 0 - one float16 scale per 32 values, then those 32 as int8 - in the order the
    engine reads a weight: dim 0 contiguous, row 0 first. `ggufq.dequantize` is the reference for the same bytes."""
    if values.shape[0] % 32:
        raise ValueError(f"{values.shape[0]}-wide rows do not fill Q8_0's 32-value blocks")
    rows = np.ascontiguousarray(values.T, dtype="<f4").reshape(-1, 32)
    d = (np.max(np.abs(rows), axis=1) / 127.0).astype("<f2")
    scale = d.astype("<f4")[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(scale > 0, np.rint(rows / scale), 0)
    q = np.clip(q, -127, 127).astype(np.int8)
    head = d.view(np.uint8).reshape(len(d), 2)
    return np.concatenate([head, q.view(np.uint8)], axis=1).tobytes()


def base_name(name: str) -> str:
    """A layer tensor's name without its block: `blk.7.attn_q_a.weight` -> `attn_q_a.weight`."""
    parts = name.split(".")
    return ".".join(parts[2:]) if parts[0] == "blk" else name


P = "glm5-next."


class Glm5NextTest(unittest.TestCase):
    def gguf(self, path, kinds=KINDS, kinds_meta=None, nextn=1, drop=(), runnable=False, quant_dense=False,
             embed_type="F32"):
        rng = np.random.default_rng(3)
        g = GGUFWriter()
        self.raw = {}

        def raw(name, shape, ty):
            vals, bpb = BLOCK[ty]
            data = rng.bytes(math.prod(shape) // vals * bpb)
            self.raw[name] = data
            g.add_raw(name, shape, ty, data)

        def f32(name, shape):
            """Matrices scaled by 1/sqrt(fan-in). The converter's own assertions do not care what the values are, but
            the runnable fixture multiplies them, and a plain standard normal inflates this fixture's residual about
            a thousandfold per block - which turns every comparison between two slightly different arithmetic paths
            into chaos, and makes the forward test's agreement assertions meaningless."""
            vals = rng.standard_normal(shape)
            if len(shape) > 1:
                vals = vals / math.sqrt(shape[-1])
            g.add_f32(name, vals)

        g.add("general.architecture", "glm5-next")
        g.add("general.name", "GLM 5.3 Flash")
        g.add("general.quantized_by", "Someone")
        g.add(P + "block_count", len(kinds), "u32")
        g.add(P + "embedding_length", E, "u32")
        g.add(P + "feed_forward_length", N_FF, "u32")
        g.add(P + "context_length", 1024, "u32")
        g.add(P + "vocab_size", VOCAB, "u32")
        g.add(P + "attention.head_count", HEAD, "u32")
        g.add(P + "attention.head_count_kv", list(kinds_meta if kinds_meta is not None else kinds))
        g.add(P + "attention.key_length", QK_LEN, "u32")
        g.add(P + "attention.value_length", V_LEN, "u32")
        g.add(P + "attention.key_length_mla", QK_MLA, "u32")
        g.add(P + "attention.value_length_mla", V_MLA, "u32")
        g.add(P + "attention.q_lora_rank", Q_LORA, "u32")
        g.add(P + "attention.kv_lora_rank", KV_LORA, "u32")
        g.add(P + "attention.indexer.head_count", IDX_HEAD, "u32")
        g.add(P + "attention.indexer.key_length", IDX_DIM, "u32")
        g.add(P + "attention.indexer.top_k", 2, "u32")
        g.add(P + "attention.indexer.kpool", KPOOL, "u32")
        g.add(P + "attention.layer_norm_rms_epsilon", 1e-5, "f32")
        g.add(P + "attention.layer_norm_epsilon", LN_EPS, "f32")
        g.add(P + "kda.head_dim", HD, "u32")
        g.add(P + "kda.gate_lower_bound", -5.0, "f32")
        g.add(P + "ssm.conv_kernel", CONV, "u32")
        g.add(P + "expert_count", N_EXPERT, "u32")
        g.add(P + "expert_used_count", 2, "u32")
        g.add(P + "expert_shared_count", N_SHARED, "u32")
        g.add(P + "expert_feed_forward_length", N_FF_EXP, "u32")
        g.add(P + "expert_shared_feed_forward_length", N_FF_SHEXP, "u32")
        g.add(P + "swiglu_clamp_exp", [CLAMP_EXP] * len(kinds))
        g.add(P + "swiglu_clamp_shexp", [CLAMP_SHEXP] * len(kinds))
        g.add(P + "expert_gating_func", 2, "u32")
        g.add(P + "expert_weights_scale", 2.5, "f32")
        g.add(P + "expert_weights_norm", True)
        g.add(P + "leading_dense_block_count", DENSE_LAYERS, "u32")
        g.add(P + "nextn_predict_layers", nextn, "u32")
        g.add(P + "hyper_connection.count", 4, "u32")
        g.add(P + "hyper_connection.epsilon", 1e-6, "f32")
        g.add(P + "hyper_connection.sinkhorn_iterations", 20, "u32")
        g.add(P + "rope.dimension_count", 0, "u32")
        g.add("tokenizer.ggml.tokens", TOKENS)
        if runnable:
            return self._runnable_tensors(g, rng, kinds, path, quant_dense, embed_type)
        raw("token_embd.weight", [256, 3], "Q6_K")
        raw("output.weight", [256, 3], "Q6_K")
        f32("output_norm.weight", [8])

        for l in range(len(kinds)):
            def keep(name):
                return name not in drop

            if keep(f"blk.{l}.attn_norm.weight"):
                f32(f"blk.{l}.attn_norm.weight", [8])
            if keep(f"blk.{l}.ffn_norm.weight"):
                f32(f"blk.{l}.ffn_norm.weight", [8])
            if keep(f"blk.{l}.attn_output.weight"):
                raw(f"blk.{l}.attn_output.weight", [256, 2], "Q5_K")
            if kinds[l] == 0:
                for p in ("q", "k", "v"):
                    if keep(f"blk.{l}.attn_{p}.weight"):
                        raw(f"blk.{l}.attn_{p}.weight", [256, 2], "Q5_K")
                f32(f"blk.{l}.ssm_a", [8])
                f32(f"blk.{l}.ssm_dt.bias", [16])
                if keep(f"blk.{l}.ssm_norm.weight"):
                    f32(f"blk.{l}.ssm_norm.weight", [4])
                for p in ("q", "k", "v"):
                    f32(f"blk.{l}.ssm_conv1d_{p}.weight", [4, 1, 16])
                if keep(f"blk.{l}.ssm_beta.weight"):
                    raw(f"blk.{l}.ssm_beta.weight", [256, 2], "Q8_0")
                for r in ("f_a", "f_b", "g_a", "g_b"):
                    if keep(f"blk.{l}.ssm_{r}.weight"):
                        raw(f"blk.{l}.ssm_{r}.weight", [256, 2], "Q8_0")
            else:
                if keep(f"blk.{l}.attn_q_a.weight"):
                    raw(f"blk.{l}.attn_q_a.weight", [256, 2], "Q5_K")
                for p in ("attn_q_b", "attn_kv_a_mqa", "attn_k_b", "attn_v_b"):
                    if keep(f"blk.{l}.{p}.weight"):
                        raw(f"blk.{l}.{p}.weight", [256, 2], "Q8_0")
                f32(f"blk.{l}.attn_q_a_norm.weight", [4])
                f32(f"blk.{l}.attn_kv_a_norm.weight", [4])
                if keep(f"blk.{l}.indexer.attn_q_b.weight"):
                    raw(f"blk.{l}.indexer.attn_q_b.weight", [256, 2], "Q8_0")
                if keep(f"blk.{l}.indexer.attn_k.weight"):
                    raw(f"blk.{l}.indexer.attn_k.weight", [256, 2], "Q8_0")
                if keep(f"blk.{l}.indexer_compressor_gate.weight"):
                    raw(f"blk.{l}.indexer_compressor_gate.weight", [256, 2], "Q8_0")
                f32(f"blk.{l}.indexer.proj.weight", [2, 8])
                f32(f"blk.{l}.indexer.k_norm.weight", [4])
                f32(f"blk.{l}.indexer.k_norm.bias", [4])
                f32(f"blk.{l}.indexer_compressor_ape.weight", [4, 2])
            if l != MTP:
                # The blobs below, F32 ones included, are placeholders: they carry a stand-in size, not the geometry's
                # shape, because the converter test asserts byte spans against them. Only gguf_runnable() writes the
                # shapes a forward pass can actually use.
                for p in ("attn", "ffn"):
                    f32(f"blk.{l}.hc_{p}_base.weight", [HC_MIX])
                    f32(f"blk.{l}.hc_{p}_scale.weight", [3])
                    raw(f"blk.{l}.hc_{p}_fn.weight", [256, 3], "Q8_0")
            if l < DENSE_LAYERS:
                for p in ("gate", "up"):
                    raw(f"blk.{l}.ffn_{p}.weight", [256, 2], "Q5_K")
                raw(f"blk.{l}.ffn_down.weight", [256, 2], "Q6_K")
            else:
                f32(f"blk.{l}.ffn_gate_inp.weight", [8, N_EXPERT])
                f32(f"blk.{l}.exp_probs_b.bias", [N_EXPERT])
                for role in ("gate", "up"):
                    raw(f"blk.{l}.ffn_{role}_exps.weight", [256, 2, N_EXPERT], GU[l])
                raw(f"blk.{l}.ffn_down_exps.weight", [256, 2, N_EXPERT], DOWN[l])
                for role in ("gate", "up"):
                    raw(f"blk.{l}.ffn_{role}_shexp.weight", [256, 2], "Q5_K")
                raw(f"blk.{l}.ffn_down_shexp.weight", [256, 2], "Q6_K")
            if l == MTP:
                raw(f"blk.{l}.nextn.eh_proj.weight", [256, 4], "Q8_0")
                f32(f"blk.{l}.nextn.enorm.weight", [8])
                f32(f"blk.{l}.nextn.hnorm.weight", [8])
                f32(f"blk.{l}.nextn.shared_head_norm.weight", [8])
        return g.write(path)

    def _runnable_tensors(self, g, rng, kinds, path, quant_dense=False, embed_type="F32"):
        """The same tensors at the shapes the geometry says, so a forward pass can actually multiply them: F32
        throughout except the routed experts, which carry the tier's native formats (the release pack's IQ2_XXS
        gate/up and IQ3_XXS down) because that is the only form the tier and the card's grouped kernels can compute.

        `gguf()` stands every quantized tensor in with a fixed-size random blob, which is what the converter's own
        byte-layout assertions need; a session test cannot read a matrix out of it. `glm5next_tiny_pack.py` asks for
        this form instead. Shapes are GGUF order (innermost first), and the bytes are written with dim 0 contiguous to
        match, which is how the engine lays a weight row out (`row o` = `o * n_in + i`). The i-quant tensors are the
        tier's blob order: a whole quantization block per row of the reduction dimension.

        `quant_dense` stores the projections in `QUANT_DENSE` as Q8_0 instead of fp32 - what the release pack does to
        645 of its tensors - from the same draw, so the two forms hold the same numbers underneath. `embed_type`
        stores the token embedding as that type instead of fp32, which is how the release pack stores its 154880 rows
        (Q4_K); the fixture uses Q8_0 because that is the one quantized form it can write from values itself drew.
        """
        self.f32vals, self.packed_bytes = {}, {}

        def f32(name, shape):
            """One draw either way, so a packed tensor and its fp32 twin hold the same values; matrices scaled by
            1/sqrt(fan-in) so the runnable fixture's forward pass stays bounded rather than inflating per block."""
            vals = rng.standard_normal(shape)
            if len(shape) > 1:
                vals = vals / math.sqrt(shape[-1])
            self.f32vals[name] = vals
            base = base_name(name)
            if quant_dense and base in QUANT_DENSE:
                data = q8_0_blocks(vals)
                self.packed_bytes[name] = data
                g.add_raw(name, list(shape), "Q8_0", data)
            else:
                g.add_raw(name, list(shape), "F32", np.asfortranarray(vals, dtype="<f4").tobytes(order="F"))

        self.iq = {}
        iq_seed = iter(range(1, 1 << 20))

        def iq(name, shape, ty):
            """dim 0 is the reduction, so the blocks are `prod(shape[1:])` rows of `shape[0]` values. The block scale
            lands the dequantized blob at 1/sqrt(fan-in), the convention f32() uses: without it a repaired block
            dequantizes to an RMS of tens and every block's expert FFN sums them into a runaway residual."""
            scale = 1.0 / (IQ_DEQ_RMS[ty] * math.sqrt(shape[0])) if ty in IQ_DEQ_RMS else 1.0
            data, _ref = iq_fixture.blocks(ty, int(np.prod(shape[1:])), shape[0], next(iq_seed), scale)
            self.iq[name] = data.tobytes()
            g.add_raw(name, list(shape), ty, self.iq[name])

        if embed_type == "F32":
            f32("token_embd.weight", [E, VOCAB])
        else:
            vals = rng.standard_normal([E, VOCAB])
            self.f32vals["token_embd.weight"] = vals
            self.packed_bytes["token_embd.weight"] = data = q8_0_blocks(vals)
            g.add_raw("token_embd.weight", [E, VOCAB], embed_type, data)
        f32("output.weight", [E, VOCAB])
        f32("output_norm.weight", [E])

        for l in range(len(kinds)):
            b = f"blk.{l}."
            f32(b + "attn_norm.weight", [E])
            f32(b + "ffn_norm.weight", [E])
            if kinds[l] == 0:                                    # KDA: linear attention
                f32(b + "attn_output.weight", [D_INNER, E])
                for p in ("q", "k", "v"):
                    f32(b + f"attn_{p}.weight", [E, D_INNER])
                    f32(b + f"ssm_conv1d_{p}.weight", [CONV, 1, D_INNER])
                f32(b + "ssm_a", [HEAD])
                f32(b + "ssm_dt.bias", [D_INNER])
                f32(b + "ssm_norm.weight", [HD])
                f32(b + "ssm_beta.weight", [E, HEAD])
                for p in ("f", "g"):
                    f32(b + f"ssm_{p}_a.weight", [E, HD])
                    f32(b + f"ssm_{p}_b.weight", [HD, D_INNER])
            else:                                                # sparse MLA with the DSA indexer
                f32(b + "attn_output.weight", [HEAD * V_MLA, E])
                f32(b + "attn_q_a.weight", [E, Q_LORA])
                f32(b + "attn_q_a_norm.weight", [Q_LORA])
                f32(b + "attn_q_b.weight", [Q_LORA, HEAD * QK_MLA])
                f32(b + "attn_kv_a_mqa.weight", [E, KV_LORA])   # rope dimension_count is 0: kv_lora only
                f32(b + "attn_kv_a_norm.weight", [KV_LORA])
                f32(b + "attn_k_b.weight", [QK_MLA, KV_LORA, HEAD])
                f32(b + "attn_v_b.weight", [KV_LORA, V_MLA, HEAD])
                f32(b + "indexer.attn_q_b.weight", [Q_LORA, IDX_HEAD * IDX_DIM])
                f32(b + "indexer.attn_k.weight", [E, IDX_DIM])
                f32(b + "indexer.proj.weight", [E, IDX_HEAD])
                f32(b + "indexer.k_norm.weight", [IDX_DIM])
                f32(b + "indexer.k_norm.bias", [IDX_DIM])
                f32(b + "indexer_compressor_gate.weight", [E, IDX_DIM])
                f32(b + "indexer_compressor_ape.weight", [IDX_DIM, KPOOL])

            if l != MTP:                                         # the mHC streams, on every model layer
                for p in ("attn", "ffn"):
                    f32(b + f"hc_{p}_fn.weight", [HC * E, HC_MIX])
                    f32(b + f"hc_{p}_base.weight", [HC_MIX])
                    f32(b + f"hc_{p}_scale.weight", [3])

            if l < DENSE_LAYERS:                                 # the dense lead MLP
                for p in ("gate", "up"):
                    f32(b + f"ffn_{p}.weight", [E, N_FF])
                f32(b + "ffn_down.weight", [N_FF, E])
            else:                                                # routed experts, plus the shared ones
                f32(b + "ffn_gate_inp.weight", [E, N_EXPERT])
                f32(b + "exp_probs_b.bias", [N_EXPERT])
                for role, ty in (("gate", IQ_GU), ("up", IQ_GU), ("down", IQ_DOWN)):
                    shape = [N_FF_EXP, E, N_EXPERT] if role == "down" else [E, N_FF_EXP, N_EXPERT]
                    iq(b + f"ffn_{role}_exps.weight", shape, ty)
                sh = N_FF_SHEXP * N_SHARED
                f32(b + "ffn_gate_shexp.weight", [E, sh])
                f32(b + "ffn_up_shexp.weight", [E, sh])
                f32(b + "ffn_down_shexp.weight", [sh, E])

            if l == MTP:                                         # the drafter block
                f32(b + "nextn.eh_proj.weight", [2 * E, E])
                for p in ("enorm", "hnorm", "shared_head_norm"):
                    f32(b + f"nextn.{p}.weight", [E])
        return g.write(path)

    def test_runnable_pack_can_carry_packed_dense_projections(self):
        """The dense projections as Q8_0, which is how 346 of the release pack's tensors arrive: the type the pack
        records, a byte count the forward pass accepts at open, and the same values underneath - so the answer that
        comes out of the mmvq kernels can be held against the one that comes out of the fp32 product."""
        file = self.gguf(self.t / "packed-dense.gguf", runnable=True, quant_dense=True)
        want = {n for n in self.f32vals if base_name(n) in QUANT_DENSE}
        self.assertTrue(want)
        self.assertEqual(set(self.packed_bytes), want)
        on_disk = {t.name: t for t in GGUFFile(file).tensors}
        for name, data in sorted(self.packed_bytes.items()):
            t = on_disk[name]
            self.assertEqual(t.type_name, "Q8_0", name)
            self.assertEqual(len(data), t.elements // 32 * 34, f"{name}: 34 bytes per 32 values")
            rows = np.ascontiguousarray(self.f32vals[name].T, dtype=np.float32).reshape(-1, 32)
            got = ggufq.dequantize(np.frombuffer(data, dtype=np.uint8),
                                   ggufq.GGMLQuantizationType.Q8_0).reshape(-1, 32)
            self.assertEqual(got.shape, rows.shape, name)
            # The block's own scale, read back out of its first two bytes: no value may move by more than half of it.
            d = np.frombuffer(data, dtype=np.float16).reshape(-1, 17)[:, 0].astype(np.float32)[:, None]
            self.assertTrue(np.all(np.abs(got - rows) <= 0.5 * np.abs(d) + 1e-9), name)
        for name in ("token_embd.weight", "blk.1.ssm_f_b.weight", "blk.1.attn_norm.weight",
                     "blk.1.hc_attn_base.weight", "blk.1.ffn_gate_shexp.weight"):
            self.assertEqual(on_disk[name].type_name, "F32", f"{name} stays fp32")

    def test_runnable_pack_can_carry_a_packed_embedding(self):
        """The token embedding as Q8_0, which is the form the forward pass dequantizes on the host: one row per
        token, contiguous, so a token costs one read. The bytes are checked against `ggufq.dequantize` row by row -
        the same read order the engine performs - and against the values the fixture drew for the same tensor."""
        file = self.gguf(self.t / "packed-embed.gguf", runnable=True, embed_type="Q8_0")
        on_disk = {t.name: t for t in GGUFFile(file).tensors}
        t = on_disk["token_embd.weight"]
        data = self.packed_bytes["token_embd.weight"]
        self.assertEqual(t.type_name, "Q8_0")
        self.assertEqual(len(data), t.elements // 32 * 34, "34 bytes per 32 values")
        self.assertEqual(len(data) // int(t.shape[1]), int(t.shape[0]) // 32 * 34, "one row per vocab id")
        rows = np.ascontiguousarray(self.f32vals["token_embd.weight"].T, dtype=np.float32).reshape(-1, 32)
        got = ggufq.dequantize(np.frombuffer(data, dtype=np.uint8),
                               ggufq.GGMLQuantizationType.Q8_0).reshape(-1, 32)
        # Read back as [vocab][E], the rows land in the order the forward pass preads them - token 0 first.
        self.assertTrue(np.allclose(got.reshape(-1, int(t.shape[0])), self.f32vals["token_embd.weight"].T, atol=0.3),
                        "the rows come out in token order")
        # The block's own scale, read back out of its first two bytes: no value may move by more than half of it.
        d = np.frombuffer(data, dtype=np.float16).reshape(-1, 17)[:, 0].astype(np.float32)[:, None]
        self.assertTrue(np.all(np.abs(got - rows) <= 0.5 * np.abs(d) + 1e-9))
        self.assertEqual(on_disk["output.weight"].type_name, "F32", "the lm head keeps its own type")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.gguf_file = self.gguf(self.t / "GLM-5.3-Flash-UD-IQ1_S.gguf")
        self._orig = convert.export_tokenizer

        def tokenizer_files(first, out_dir):
            d = Path(out_dir) / "tokenizer"
            d.mkdir(parents=True, exist_ok=True)
            (d / "tokenizer.json").write_bytes(b"{}")
            return sorted(d.iterdir())
        convert.export_tokenizer = tokenizer_files

    def tearDown(self):
        convert.export_tokenizer = self._orig
        self.tmp.cleanup()

    def args(self, **kw):
        d = dict(gguf=[str(self.gguf_file)], mmproj=None, out=str(self.t / "out.basalt"), mtp_source=None,
                 mtp_dense=None, draft_vocab=None, profile=None, profile_from=None, hc8=None, name="Tiny",
                 compat_bf16=False, no_vision=False, force=False, sections_from=None)
        d.update(kw)
        return type("A", (), d)()

    def convert(self, **kw):
        convert.convert(self.args(**kw))
        return ModelFile(str(self.t / "out.basalt"))

    def rows(self, model, section):
        """`section`'s dense index as name -> row fields."""
        lines = model.read(section).decode().splitlines()
        return {c[0]: c for c in (l.split() for l in lines if l and not l.startswith("#"))}

    def test_sections_and_meta(self):
        m = self.convert()
        self.assertEqual(set(m.names()), {"tokenizer/tokenizer.json", "dense", "dense.index", "experts",
                                          "experts.layout", "gguf", "meta", "mtp/dense", "mtp/dense.index",
                                          "mtp/experts", "mtp/experts.layout"})
        for n in m.names():
            self.assertTrue(m.verify(n), n)
            self.assertEqual(m.section(n)[0] % SECTION_ALIGN, 0, n)
        meta = json.loads(m.read("meta"))
        self.assertEqual(meta["arch"], "glm5next")
        self.assertEqual(meta["license"]["name"], "MIT")
        self.assertEqual(meta["license"]["quantized_by"], "Someone")
        self.assertIsNone(meta["vision"])
        self.assertEqual(meta["mtp"]["layers"], [MTP])
        g = meta["geometry"]
        self.assertEqual(g["attention.head_count_kv"], KINDS)
        # the two lengths the engine separates: the checkpoint's own, and the MLA heads' the weights are built to
        self.assertEqual((g["attention.key_length"], g["attention.value_length"]), (QK_LEN, V_LEN))
        self.assertEqual((g["attention.key_length_mla"], g["attention.value_length_mla"]), (QK_MLA, V_MLA))
        self.assertEqual(g["expert_shared_feed_forward_length"], N_FF_SHEXP)
        self.assertEqual(g["swiglu_clamp_exp"], [CLAMP_EXP] * len(KINDS))
        self.assertEqual(g["swiglu_clamp_shexp"], [CLAMP_SHEXP] * len(KINDS))
        self.assertAlmostEqual(g["attention.layer_norm_epsilon"], LN_EPS, places=9)
        self.assertEqual(g["leading_dense_block_count"], DENSE_LAYERS)
        self.assertEqual(g["expert_count"], N_EXPERT)
        self.assertEqual(g["attention.indexer.kpool"], 2)
        self.assertEqual(g["hyper_connection.count"], 4)

    def test_experts_section_skips_the_dense_layer(self):
        m = self.convert()
        lines = m.read("experts.layout").decode().splitlines()
        self.assertEqual([int(l.split()[0]) for l in lines if not l.startswith("#")],
                         [l for l in range(DENSE_LAYERS, MTP)])
        # the drafter's experts are their own layout, naming only its layer
        mtp_lines = [l.split() for l in m.read("mtp/experts.layout").decode().splitlines() if not l.startswith("#")]
        self.assertEqual([int(c[0]) for c in mtp_lines], [MTP])
        out, off = m.read("experts"), 0
        for l in range(DENSE_LAYERS, MTP):
            per = {r: math.prod([256, 2]) // BLOCK[ty][0] * BLOCK[ty][1]
                   for r, ty in (("gate", GU[l]), ("up", GU[l]), ("down", DOWN[l]))}
            blob = sum(per.values())
            for e in range(N_EXPERT):
                want = b"".join(self.raw[f"blk.{l}.ffn_{r}_exps.weight"][e * per[r]:(e + 1) * per[r]]
                                for r in ("gate", "up", "down"))
                self.assertEqual(out[off + e * blob:off + (e + 1) * blob], want, (l, e))
            off += blob * N_EXPERT
        self.assertEqual(off, len(out))

    def test_the_runnable_fixture_carries_the_release_packs_i_quant_experts(self):
        """The runnable form, whose routed experts are the tier's native formats (IQ2_XXS gate/up, IQ3_XXS down, the
        release pack's own mix): the layout names those types and one expert's blob, and the pool is that blob - gate
        rows, up rows, down rows - once per expert."""
        m = self.convert(gguf=[str(self.gguf(self.t / "iq-experts.gguf", runnable=True))])
        rows = [l.split() for l in m.read("experts.layout").decode().splitlines() if not l.startswith("#")]
        self.assertEqual([int(c[0]) for c in rows], [l for l in range(DENSE_LAYERS, MTP)])

        def row_bytes(ty, n):
            vals, by = BLOCK[ty]
            self.assertEqual(n % vals, 0, (ty, n))          # a whole quantization block per expert row
            return n // vals * by

        gu_row, d_row = row_bytes(IQ_GU, E), row_bytes(IQ_DOWN, N_FF_EXP)
        blob = 2 * N_FF_EXP * gu_row + E * d_row
        want_gu, want_d = iq_fixture.KERNEL_IDS[IQ_GU], iq_fixture.KERNEL_IDS[IQ_DOWN]
        for c in rows:                                      # layer gu down offset blob [up]
            self.assertEqual([int(c[1]), int(c[2])], [want_gu, want_d], c[0])
            if len(c) > 5:                                  # v7 names the up rows' type apart
                self.assertEqual(int(c[5]), want_gu, c[0])
            self.assertEqual(int(c[4]), blob, c[0])
        out = m.read("experts")
        self.assertEqual(len(out), blob * N_EXPERT * len(rows))
        l = int(rows[0][0])
        parts = [(self.iq[f"blk.{l}.ffn_{r}_exps.weight"], N_FF_EXP * gu_row if r != "down" else E * d_row)
                 for r in ("gate", "up", "down")]
        base = int(rows[0][3])
        for e in range(N_EXPERT):
            want = b"".join(data[e * n:(e + 1) * n] for data, n in parts)
            self.assertEqual(out[base + e * blob:base + (e + 1) * blob], want, (l, e))

    def test_drafter_floats_are_in_mtp_dense_and_its_rows_stay_in_gguf(self):
        m = self.convert()
        meta = json.loads(m.read("meta"))
        mtp_rows = self.rows(m, "mtp/dense.index")
        self.assertIn("blk.3.nextn.enorm.weight", mtp_rows)         # F32: bytes in the mtp/dense pool
        self.assertNotEqual(mtp_rows["blk.3.nextn.enorm.weight"][2], "0")
        self.assertEqual(mtp_rows["blk.3.attn_q_b.weight"][2], "0")  # quantized: served from the gguf section
        self.assertNotIn("blk.3.attn_q_b.weight", self.rows(m, "dense.index"))
        t = meta["tensors"]["blk.3.attn_q_b.weight"]
        self.assertNotIn("section", t)
        self.assertEqual(m.read("gguf")[t["offset"]:t["offset"] + t["bytes"]], self.raw["blk.3.attn_q_b.weight"])

    def test_mmproj_becomes_the_vision_section(self):
        proj = self.t / "mmproj.gguf"
        proj.write_bytes(b"GGUF" + b"\0" * 16 + b"glm5v" + b"\1\2\3" * 100)
        m = self.convert(mmproj=str(proj))
        meta = json.loads(m.read("meta"))
        self.assertIn("vision/mmproj", m.names())
        self.assertTrue(m.verify("vision/mmproj"))
        self.assertEqual(m.read("vision/mmproj"), proj.read_bytes())
        self.assertEqual(meta["vision"]["mmproj"], "vision/mmproj")

    def test_a_wrong_projector_is_refused(self):
        proj = self.t / "mmproj.gguf"
        proj.write_bytes(b"GGUF" + b"\0" * 16 + b"deepseek41v" + b"\1\2\3" * 100)
        with self.assertRaises(SystemExit):
            self.convert(mmproj=str(proj))

    def test_side_sections_are_refused(self):
        with self.assertRaises(SystemExit):
            self.convert(mtp_source=str(self.gguf_file))

    def test_layer_kinds_must_cover_every_layer(self):
        bad = self.gguf(self.t / "bad-kinds.gguf", kinds_meta=[0, 0, 1])
        with self.assertRaises(SystemExit) as cm:
            convert.convert(self.args(gguf=[str(bad)]))
        self.assertIn("attention.head_count_kv", str(cm.exception))

    def test_a_linear_attention_layer_needs_its_delta_net(self):
        bad = self.gguf(self.t / "bad-kda.gguf", drop=("blk.0.ssm_f_b.weight",))
        with self.assertRaises(SystemExit) as cm:
            convert.convert(self.args(gguf=[str(bad)]))
        self.assertIn("blk.0.ssm_f_b.weight", str(cm.exception))

    def test_a_sparse_layer_without_its_indexer_is_refused(self):
        bad = self.gguf(self.t / "bad-indexer.gguf", drop=("blk.2.indexer_compressor_gate.weight",))
        with self.assertRaises(SystemExit) as cm:
            convert.convert(self.args(gguf=[str(bad)]))
        self.assertIn("indexer_compressor_gate", str(cm.exception))

    def test_no_drafter_block_is_refused(self):
        bad = self.gguf(self.t / "bad-nextn.gguf", nextn=0)
        with self.assertRaises(SystemExit) as cm:
            convert.convert(self.args(gguf=[str(bad)]))
        self.assertIn("next-n", str(cm.exception))

    def test_default_name(self):
        arch = glm5next.Glm5Next()
        self.assertEqual(arch.default_name("GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf", {}), "GLM-5.3-Flash UD-IQ1_S")
        self.assertEqual(arch.default_name("GLM-5.3-Flash-UD-Q3_K_XL.gguf", {}), "GLM-5.3-Flash UD-Q3_K_XL")
        self.assertEqual(arch.default_name("GLM-5.3-Flash.gguf", {}), "GLM-5.3-Flash")
        self.assertEqual(arch.default_name("something-else.gguf", {}), "GLM-5.3-Flash")


if __name__ == "__main__":
    unittest.main()
