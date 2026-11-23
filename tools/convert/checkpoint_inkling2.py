"""An inkling2 pack (Thinking Machines' Inkling 2 Mini: a looped mixture-of-experts transformer) straight from the
official checkpoint, for tools/convert/convert.py.

  experts             the routed experts per layer, [gate | up | down] (the checkpoint's w13 rows alternate gate and up):
                        NVFP4 layers: type 40 blocks, the E2M1 codes and E4M3 scales as stored (nibbles moved into
                          64-value blocks), then the FP32 tail {s, s, s_down, 0}, s = amax / outer_scale_denominator
                        MXFP8 layers: type 46 blocks, the E4M3 codes and UE8M0 scales as stored (four 32-value
                          sub-blocks per 132-byte block: 4 UE8M0 scales then 128 E4M3 codes)
  experts.layout      one `layer gu_type d_type offset blob_bytes` line per layer
  gguf                every other text tensor in llama.cpp-style names: the attention projections Q8_0 (ggml's
                        reference rounding of the BF16 values), the token embedding and the output head BF16 as stored
                        (their padded rows past the vocabulary dropped), the loop controllers BF16 as stored, norms,
                        block-residual and message queries, router, its bias, convolution taps, relative-bias tables
                        and clip scales F32 (exact); the DSpark drafter's tensors (`mtp.safetensors`) as `dspark.*`,
                        BF16 and F32 as stored
  vision/mmproj       the vision tower as llama.cpp's clip GGUF (projector inkling2v), BF16 / F32 as stored
  tokenizer/*, meta   from config.json, generation_config.json, the tokenizer files and the model card

Needs numpy.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import pathlib
import re
import sys
import tempfile
import time

import numpy as np

import checkpoint as C
import model_sections as S

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import TENSOR_ALIGN, ModelWriter, Stream  # noqa: E402

ARCH = "inkling2"
TYPE_ID = {"F32": 0, "Q8_0": 8, "BF16": 30, "NVFP4": 40, "MXFP8": 46}
Q8_0 = 8
MXFP8 = 46
ROWS_PER_JOB = 1 << 22
EXPERTS_PER_JOB = 16
E4M3 = S.fp8_e4m3_table()
# the o200k split of the checkpoint's tokenizer.json
INKLING2_HF_SPLIT = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
                     r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
                     r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")

# checkpoint tensor after `model.llm.layers.N.` -> (GGUF name after `blk.N.`, form, transform)
#   forms: Q8 (Q8_0 from BF16), BF16 (as stored), F32 (BF16 widened, or F32 as stored)
#   transforms: id; conv ([C, 1, K] -> [C, K], tap K-1 the current token); flip (a [K, C] table whose row 0 weighs the
#   current token -> [C, K] in the conv order)
LAYER = {
    "attn.wq_du.weight": ("attn_q.weight", "Q8", "id"),
    "attn.wk_dv.weight": ("attn_k.weight", "Q8", "id"),
    "attn.wv_dv.weight": ("attn_v.weight", "Q8", "id"),
    "attn.wr_du.weight": ("attn_rel.weight", "Q8", "id"),
    "attn.gate_attn.weight": ("attn_gate.weight", "Q8", "id"),
    "attn.attn_sink_proj.weight": ("attn_sink.weight", "Q8", "id"),
    "attn.wo_ud.weight": ("attn_output.weight", "Q8", "id"),
    "attn.q_l2_clip.scale": ("attn_q_clip.weight", "F32", "id"),
    "attn.rel_l2_clip.scale": ("attn_rel_clip.weight", "F32", "id"),
    "attn.rel_logits_proj.proj": ("attn_rel_bias.weight", "F32", "id"),
    "attn.k_sconv.weight": ("attn_k_conv.weight", "F32", "conv"),
    "attn.v_sconv.weight": ("attn_v_conv.weight", "F32", "conv"),
    "attn.k_sconv.static_w": ("attn_k_conv.weight", "F32", "flip"),
    "attn.v_sconv.static_w": ("attn_v_conv.weight", "F32", "flip"),
    "attn.k_sconv.weight_proj": ("attn_k_conv_proj.weight", "Q8", "id"),
    "attn.v_sconv.weight_proj": ("attn_v_conv_proj.weight", "Q8", "id"),
    "attn_norm.weight": ("attn_norm.weight", "F32", "id"),
    "q_input_norm.weight": ("attn_q_inp_norm.weight", "F32", "id"),
    "k_input_norm.weight": ("attn_k_inp_norm.weight", "F32", "id"),
    "v_input_norm.weight": ("attn_v_inp_norm.weight", "F32", "id"),
    "attn_sconv.weight": ("attn_post_conv.weight", "F32", "conv"),
    "mlp_sconv.weight": ("ffn_post_conv.weight", "F32", "conv"),
    "mlp_norm.weight": ("ffn_norm.weight", "F32", "id"),
    "mlp.gate.weight": ("ffn_gate_inp.weight", "F32", "id"),
    "mlp.gate.bias": ("exp_probs_b.bias", "F32", "id"),
}
# the loop controllers of the core layers: stream attn (the query input), k, v and mlp
for _src, _dst in (("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "ffn")):
    for _part in ("down", "gain", "forget"):
        LAYER[f"controllers.{_src}.{_part}_weight"] = (f"loop_{_dst}_ctrl_{_part}.weight", "BF16", "id")
GLOBAL = {
    "model.llm.embed.weight": ("token_embd.weight", "BF16", "vocab"),
    "model.llm.embed_norm.weight": ("token_embd_norm.weight", "F32", "id"),
    "model.llm.norm.weight": ("output_norm.weight", "F32", "id"),
    "model.llm.unembed.weight": ("output.weight", "BF16", "vocab"),
    "model.llm.attn_res.output_query": ("output_res_query.weight", "F32", "id"),
}
DRAFTER_LAYER = {
    "input_layernorm.weight": ("attn_norm.weight", "F32"),
    "post_attention_layernorm.weight": ("post_attention_norm.weight", "F32"),
    "dflare_norm.weight": ("dflare_norm.weight", "F32"),
    "self_attn.q_proj.weight": ("attn_q.weight", "BF16"),
    "self_attn.k_proj.weight": ("attn_k.weight", "BF16"),
    "self_attn.v_proj.weight": ("attn_v.weight", "BF16"),
    "self_attn.o_proj.weight": ("attn_output.weight", "BF16"),
    "self_attn.k_proj_ctx.weight": ("attn_k_ctx.weight", "BF16"),
    "self_attn.v_proj_ctx.weight": ("attn_v_ctx.weight", "BF16"),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "F32"),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", "F32"),
    "self_attn.k_norm_ctx.weight": ("attn_k_ctx_norm.weight", "F32"),
    "mlp.gate_proj.weight": ("ffn_gate.weight", "BF16"),
    "mlp.up_proj.weight": ("ffn_up.weight", "BF16"),
    "mlp.down_proj.weight": ("ffn_down.weight", "BF16"),
}
DRAFTER_GLOBAL = {
    "fuse_weight": ("fuse_weight", "F32"),
    "mask_embed": ("mask_embed", "F32"),
    "norm.weight": ("norm.weight", "F32"),
    "markov_head.markov_w1.weight": ("markov.w1.weight", "BF16"),
    "markov_head.rnn_proj.weight": ("markov.rnn.weight", "BF16"),
    "markov_head.rnn_proj.bias": ("markov.rnn.bias", "F32"),
    "markov_head.hidden_out.weight": ("markov.out.weight", "BF16"),
    "markov_head.hidden_out.bias": ("markov.out.bias", "F32"),
    "markov_head.dflare_norm.weight": ("markov.dflare_norm.weight", "F32"),
}
VISION_BLOCK = {
    "ln_1.weight": ("ln1.weight", "F32"), "ln_1.bias": ("ln1.bias", "F32"),
    "ln_2.weight": ("ln2.weight", "F32"), "ln_2.bias": ("ln2.bias", "F32"),
    "ls_1.gamma": ("ls1.weight", "F32"), "ls_2.gamma": ("ls2.weight", "F32"),
    "self_attn.q_proj.weight": ("attn_q.weight", "BF16"), "self_attn.q_proj.bias": ("attn_q.bias", "F32"),
    "self_attn.k_proj.weight": ("attn_k.weight", "BF16"), "self_attn.k_proj.bias": ("attn_k.bias", "F32"),
    "self_attn.v_proj.weight": ("attn_v.weight", "BF16"), "self_attn.v_proj.bias": ("attn_v.bias", "F32"),
    "self_attn.out_proj.weight": ("attn_out.weight", "BF16"), "self_attn.out_proj.bias": ("attn_out.bias", "F32"),
    "mlp.c_fc.weight": ("ffn_up.weight", "BF16"), "mlp.c_fc.bias": ("ffn_up.bias", "F32"),
    "mlp.c_proj.weight": ("ffn_down.weight", "BF16"), "mlp.c_proj.bias": ("ffn_down.bias", "F32"),
}
VISION_GLOBAL = {
    "model.visual.backbone.embeddings.class_embedding": ("v.class_embd", "F32"),
    "model.visual.backbone.embeddings.cls_position_embedding": ("v.class_position_embd.weight", "F32"),
    "model.visual.backbone.embeddings.position_embedding": ("v.position_embd.weight", "F32"),
    "model.visual.backbone.ln_pre.weight": ("v.pre_ln.weight", "F32"),
    "model.visual.backbone.ln_pre.bias": ("v.pre_ln.bias", "F32"),
    "model.visual.proj_in.weight": ("mm.1.weight", "BF16"),
    "model.visual.proj_out.weight": ("mm.2.weight", "BF16"),
}
EXPERT_PARTS = ("w13_weight", "w13_weight_scale", "w13_weight_amax", "w13_weight_outer_scale_denominator",
                "w2_weight", "w2_weight_scale", "w2_weight_amax", "w2_weight_outer_scale_denominator")
METHODS = {"id": "as stored", "conv": "[C, 1, K] -> [C, K]", "flip": "[K, C] (row 0 the current token) -> [C, K] reversed",
           "vocab": "rows past the vocabulary dropped", "stack": "per core layer (q, k, v, mlp) message queries"}


class Tensor:
    def __init__(self, name, form, src, how, ne, rows=None):
        self.name, self.form, self.src, self.how, self.ne, self.rows = name, form, src, how, ne, rows
        self.type_name = {"Q8": "Q8_0"}.get(form, form)
        self.type_id = TYPE_ID[self.type_name]

    def expected_bytes(self) -> int:
        n = int(np.prod(self.ne))
        return {"F32": 4 * n, "BF16": 2 * n, "Q8_0": n // 32 * 34}[self.type_name]


def _schemes(ck: C.Checkpoint) -> dict[str, str]:
    """hf_quant_config.json: each expert tensor's weight transform (nvfp4_nearest or mxfp8_nearest)."""
    qc = ck.dir / "hf_quant_config.json"
    if not qc.is_file():
        raise S.ConvertError(f"{qc} not found")
    out = {}
    for g in json.loads(qc.read_text(encoding="utf-8")).get("model", {}).get("parameter_groups", []):
        for n in g.get("tensors", []):
            if n.endswith((".w13_weight", ".w2_weight")):
                out[n] = g.get("weight_transform")
                if out[n] == "nvfp4_nearest" and g.get("weight_scale_format") != "amax_denominator":
                    raise S.ConvertError(f"hf_quant_config.json: {n} has NVFP4 scales other than amax_denominator")
    return out


class Experts:
    """The `experts` section: per layer the blobs of its routed experts."""

    def __init__(self, ck: C.Checkpoint):
        tc = ck.text
        self.ck = ck
        self.n_layers, self.n_expert = int(tc["num_hidden_layers"]), int(tc["n_routed_experts"])
        self.h, self.ff = int(tc["hidden_size"]), int(tc["intermediate_size"])
        schemes = _schemes(ck)
        self.layers, offset = [], 0
        h, ff, E = self.h, self.ff, self.n_expert
        for l in range(self.n_layers):
            p = f"model.llm.layers.{l}.mlp.experts."
            s13, s2 = schemes.get(p + "w13_weight"), schemes.get(p + "w2_weight")
            if s13 != s2 or s13 not in ("nvfp4_nearest", "mxfp8_nearest"):
                raise S.ConvertError(f"layer {l}: experts quantized as {s13} / {s2}")
            if s13 == "nvfp4_nearest":
                ck.need(p + "w13_weight", "U8", (E, 2 * ff, h // 2))
                ck.need(p + "w13_weight_scale", "F8_E4M3", (E, 2 * ff, h // 16))
                ck.need(p + "w2_weight", "U8", (E, h, ff // 2))
                ck.need(p + "w2_weight_scale", "F8_E4M3", (E, h, ff // 16))
                for w in ("w13", "w2"):
                    ck.need(p + w + "_weight_amax", "F32", (E,))
                    ck.need(p + w + "_weight_outer_scale_denominator", "F32", (1,))
                blob = 2 * ff * S.nvfp4_row_bytes(h) + h * S.nvfp4_row_bytes(ff) + S.NVFP4_TAIL
                kind, gu = "nvfp4", S.NVFP4
            else:
                ck.need(p + "w13_weight", "F8_E4M3", (E, 2 * ff, h))
                ck.need(p + "w13_weight_scale", "U8", (E, 2 * ff, h // 32))
                ck.need(p + "w2_weight", "F8_E4M3", (E, h, ff))
                ck.need(p + "w2_weight_scale", "U8", (E, h, ff // 32))
                for w in ("w13", "w2"):
                    if p + w + "_weight_amax" in ck.t or p + w + "_weight_outer_scale_denominator" in ck.t:
                        raise S.ConvertError(f"layer {l}: MXFP8 experts with NVFP4 companions")
                if h % 128 or ff % 128:
                    raise S.ConvertError(f"layer {l}: MXFP8 experts require hidden_size and intermediate_size multiples of 128")
                blob = 2 * ff * (h // 128 * 132) + h * (ff // 128 * 132)
                kind, gu = "mxfp8", MXFP8
            self.layers.append((l, kind, gu, offset, blob))
            offset += blob * E
        self.total = offset

    def layout_text(self) -> bytes:
        head = ("# experts v6: layer gu_type d_type offset blob_bytes (n_expert %d, total %d); "
                "an NVFP4 blob ends in FP32 {s_gate, s_up, s_down, 0}\n" % (self.n_expert, self.total))
        return (head + "".join("%d %d %d %d %d\n" % (l, gu, gu, off, blob) for l, _, gu, off, blob in self.layers)).encode()

    def jobs(self):
        ck, E = self.ck, self.n_expert
        for l, kind, _, _, _ in self.layers:
            p = f"model.llm.layers.{l}.mlp.experts."
            src = {k: (str(ck.t[p + k][0]), ck.t[p + k][1]) for k in EXPERT_PARTS if p + k in ck.t}
            for s in range(0, E, EXPERTS_PER_JOB):
                yield ("experts", kind, src, self.h, self.ff, s, min(EXPERTS_PER_JOB, E - s))

    def chunks(self, pool, depth: int = 8):
        for b in S.bounded_imap(pool, _job, self.jobs(), depth):
            if isinstance(b, str):
                raise S.ConvertError(b)
            yield b


def _expert_blobs(kind, src, h, ff, e0, n) -> bytes:
    """Blobs of experts [e0, e0 + n) of one layer."""
    def rows(key, shape, dtype=np.uint8):
        path, off = src[key]
        count = int(np.prod(shape))
        return np.fromfile(path, dtype=dtype, count=count * n, offset=off + e0 * count * np.dtype(dtype).itemsize).reshape(n, *shape)

    out = []
    if kind == "nvfp4":
        c13, s13 = rows("w13_weight", (2 * ff, h // 2)), rows("w13_weight_scale", (2 * ff, h // 16))
        c2, s2 = rows("w2_weight", (h, ff // 2)), rows("w2_weight_scale", (h, ff // 16))
        if (s13 & 0x80).any() or (s13 == 0x7F).any() or (s2 & 0x80).any() or (s2 == 0x7F).any():
            return f"experts {e0}..: an E4M3 block scale with the sign bit set or NaN"
        g = {w: (np.fromfile(src[w + "_weight_amax"][0], "<f4", count=n, offset=src[w + "_weight_amax"][1] + 4 * e0),
                 np.fromfile(src[w + "_weight_outer_scale_denominator"][0], "<f4", count=1,
                             offset=src[w + "_weight_outer_scale_denominator"][1])[0]) for w in ("w13", "w2")}
        for e in range(n):
            s_gu = np.float32(g["w13"][0][e]) / np.float32(g["w13"][1])
            s_d = np.float32(g["w2"][0][e]) / np.float32(g["w2"][1])
            tail = np.array([s_gu, s_gu, s_d, 0.0], dtype="<f4")
            if not (np.isfinite(tail).all() and (tail[:3] > 0).all()):
                return f"expert {e0 + e}: an amax / denominator that is not finite and positive"
            out.append(C.nvfp4_rows(c13[e, 0::2], s13[e, 0::2]).tobytes())   # gate: the even rows of w13
            out.append(C.nvfp4_rows(c13[e, 1::2], s13[e, 1::2]).tobytes())   # up: the odd rows
            out.append(C.nvfp4_rows(c2[e], s2[e]).tobytes())
            out.append(tail.tobytes())
        return b"".join(out)
    c13, s13 = rows("w13_weight", (2 * ff, h)), rows("w13_weight_scale", (2 * ff, h // 32))
    c2, s2 = rows("w2_weight", (h, ff)), rows("w2_weight_scale", (h, ff // 32))
    if (s13 == 255).any() or (s2 == 255).any():
        return f"experts {e0}..: a UE8M0 scale byte of 255 (NaN)"

    def pack_mxfp8(c, s):   # 128-value blocks: 4 UE8M0 scales then 128 E4M3 codes (132 bytes)
        nb = c.shape[1] // 128
        return np.concatenate([s.reshape(c.shape[0], nb, 4), c.reshape(c.shape[0], nb, 128)], axis=2).tobytes()

    for e in range(n):
        if ((c13[e] & 0x7F) == 0x7F).any() or ((c2[e] & 0x7F) == 0x7F).any():
            return f"expert {e0 + e}: an E4M3 NaN code"
        out.append(pack_mxfp8(c13[e, 0::2], s13[e, 0::2]))   # gate: the even rows of w13
        out.append(pack_mxfp8(c13[e, 1::2], s13[e, 1::2]))   # up: the odd rows
        out.append(pack_mxfp8(c2[e], s2[e]))
    return b"".join(out)


class Plan:
    """The pack's `gguf` tensors in checkpoint order (the language model, then the drafter), each with how it is made."""

    def __init__(self, ck: C.Checkpoint):
        tc = ck.text
        self.ck = ck
        if ck.config.get("model_type") != "inkling2_loop" or tc.get("model_type") != "inkling2_loop_text":
            raise S.ConvertError(f"config.json: model_type {ck.config.get('model_type')} / {tc.get('model_type')}, "
                                 "expected inkling2_loop / inkling2_loop_text")
        self.n_layers = int(tc["num_hidden_layers"])
        self.h = int(tc["hidden_size"])
        self.n_vocab = int(tc["unpadded_vocab_size"])
        self.local = set(int(x) for x in tc["local_layer_ids"])
        loop = tc["loop_config"]
        self.core = (int(loop["core_start"]), int(loop["core_end"]))
        for k, want in (("q_bias", False), ("o_bias", False), ("n_shared_experts", 0), ("use_value_norm_no_gamma", True),
                        ("attn_res_normalize_values", True), ("use_gate_bias", True)):
            if tc.get(k, want) != want:
                raise S.ConvertError(f"config.json: {k} = {tc.get(k)!r}; the engine runs {want!r}")
        if loop.get("is_varloop"):
            raise S.ConvertError("config.json: a variable loop (is_varloop); the engine runs a fixed number of passes")
        if tc.get("scoring_func", "sigmoid") != "sigmoid":
            raise S.ConvertError(f"config.json: router scoring {tc.get('scoring_func')}")
        self.tensors: dict[str, Tensor] = {}
        unknown = []
        for k in ck.order:
            if k.startswith("model.visual.") or re.search(r"\.mlp\.experts\.", k):
                continue
            m = re.fullmatch(r"model\.llm\.layers\.(\d+)\.(.+)", k)
            mq = re.fullmatch(r"model\.llm\.attn_res\.(attn|mlp)_queries\.(\d+)", k)
            _, _, dt, shape, _ = ck.t[k]
            if mq:   # attn_queries.i feeds layer i + 1 (layer 0 has no attention mix); mlp_queries.i layer i
                l = int(mq.group(2)) + (1 if mq.group(1) == "attn" else 0)
                name = f"blk.{l}.{'attn' if mq.group(1) == 'attn' else 'ffn'}_res_query.weight"
                self._add(name, "F32", k, "id", list(reversed(shape)), dt)
                continue
            if k in ("model.llm.middle_cycle_cross_pass_queries", "model.llm.middle_cycle_cross_pass_kv_queries"):
                n_core = self.core[1] - self.core[0]
                if dt != "F32" or shape != [2 * n_core, self.h]:
                    raise S.ConvertError(f"{k} is {dt} {shape}, expected F32 {[2 * n_core, self.h]}")
                continue
            if m:
                rule = LAYER.get(m.group(2))
                if rule is None:
                    unknown.append(k)
                    continue
                gname, form, how = f"blk.{m.group(1)}.{rule[0]}", rule[1], rule[2]
            elif k in GLOBAL:
                gname, form, how = GLOBAL[k]
            else:
                unknown.append(k)
                continue
            if how == "conv":
                if len(shape) != 3 or shape[1] != 1:
                    raise S.ConvertError(f"{k}: a conv weight of shape {shape}")
                ne = [shape[2], shape[0]]
            elif how == "flip":
                ne = [shape[0], shape[1]]
            elif how == "vocab":
                if shape[0] < self.n_vocab:
                    raise S.ConvertError(f"{k} has {shape[0]} rows, fewer than the vocabulary's {self.n_vocab}")
                ne = [shape[1], self.n_vocab]
            else:
                ne = list(reversed(shape))
            self._add(gname, form, k, how, ne, dt)
        if unknown:
            raise S.ConvertError("checkpoint tensors with no pack counterpart: " + ", ".join(unknown[:8]))
        n_core = self.core[1] - self.core[0]
        for i in range(n_core):
            name = f"blk.{self.core[0] + i}.loop_msg_query.weight"
            self.tensors[name] = Tensor(name, "F32", None, "stack", [self.h, 4])
        for k in sorted((k for k in ck.t if k.startswith("mtp.")), key=lambda k: ck.t[k][1]):
            _, _, dt, shape, _ = ck.t[k]
            m = re.fullmatch(r"mtp\.layers\.(\d+)\.(.+)", k)
            rule = DRAFTER_LAYER.get(m.group(2)) if m else DRAFTER_GLOBAL.get(k[len("mtp."):])
            if rule is None:
                raise S.ConvertError(f"mtp.safetensors: {k[len('mtp.'):]} has no counterpart")
            gname = "dspark." + (f"blk.{m.group(1)}.{rule[0]}" if m else rule[0])
            how = "vocab" if k == "mtp.markov_head.markov_w1.weight" else "id"
            ne = [shape[1], self.n_vocab] if how == "vocab" else list(reversed(shape))
            self._add(gname, rule[1], k, how, ne, dt)
        self._check_complete(tc)

    def _add(self, gname, form, src, how, ne, dt):
        if dt not in ("BF16", "F32") or (form == "BF16" and dt != "BF16"):
            raise S.ConvertError(f"{src} is {dt}, expected BF16" + (" or F32" if form == "F32" else ""))
        if form == "Q8" and (len(ne) != 2 or ne[0] % 32):
            raise S.ConvertError(f"{src}: rows of {ne[0]} values cannot be Q8_0")
        if gname in self.tensors:
            raise S.ConvertError(f"two checkpoint tensors map to {gname}")
        self.tensors[gname] = Tensor(gname, form, src, how, ne)

    def _check_complete(self, tc):
        need = [r[0] for r in GLOBAL.values()]
        for l in range(self.n_layers):
            full = l not in self.local
            core = self.core[0] <= l < self.core[1]
            for key, rule in LAYER.items():
                if key.startswith("controllers.") and not core:
                    continue
                if key in ("attn_norm.weight", "attn.k_sconv.weight", "attn.v_sconv.weight") and full:
                    continue
                if (key.startswith(("q_input_norm", "k_input_norm", "v_input_norm")) or key.endswith(("static_w", "weight_proj"))) \
                        and not full:
                    continue
                need.append(f"blk.{l}.{rule[0]}")
            if l > 0:
                need.append(f"blk.{l}.attn_res_query.weight")
            need.append(f"blk.{l}.ffn_res_query.weight")
        missing = [n for n in need if n not in self.tensors]
        if missing:
            raise S.ConvertError(f"the checkpoint lacks {len(missing)} tensors the engine reads: " + ", ".join(missing[:8]))
        mc = self.ck.config.get("mtp_config") or {}
        if mc:
            n = int(mc["num_hidden_layers"])
            dneed = ["dspark." + r[0] for r in DRAFTER_GLOBAL.values()]
            dneed += [f"dspark.blk.{l}.{r[0]}" for l in range(n) for r in DRAFTER_LAYER.values()]
            dmissing = [x for x in dneed if x not in self.tensors]
            if dmissing:
                raise S.ConvertError("mtp.safetensors lacks " + ", ".join(dmissing[:8]))

    # ---- float tensors
    def values(self, name) -> np.ndarray:
        t = self.tensors[name]
        if t.how == "stack":   # core layer start + i mixes its q, k, v and mlp messages with these four queries
            i = int(name.split(".")[1]) - self.core[0]
            q = self.ck.values("model.llm.middle_cycle_cross_pass_queries")
            kv = self.ck.values("model.llm.middle_cycle_cross_pass_kv_queries")
            return np.ascontiguousarray(np.stack([q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]]), dtype=np.float32)
        x = self.ck.values(t.src)
        if t.how == "conv":
            x = x[:, 0, :]
        elif t.how == "flip":
            x = x[::-1].T
        elif t.how == "vocab":
            x = x[:self.n_vocab]
        return np.ascontiguousarray(x, dtype=np.float32)

    def float_bytes(self, name) -> bytes:
        t = self.tensors[name]
        x = self.values(name)
        if not np.isfinite(x).all():
            raise S.ConvertError(f"{name} has a non-finite value")
        if t.type_name == "F32":
            return x.astype("<f4").tobytes()
        u = x.view(np.uint32)
        if np.count_nonzero(u & 0xFFFF):
            raise S.ConvertError(f"{name}: a value that is not BF16")
        return (u >> 16).astype("<u2").tobytes()

    # ---- Q8_0 projections, streamed
    def chunks(self, name, pool, depth: int = 64):
        t = self.tensors[name]
        p, off, _, _, _ = self.ck.t[t.src]
        ne0, ne1 = t.ne
        step = max(1, ROWS_PER_JOB // ne0)
        jobs = (("q8", str(p), off, ne0, r, min(step, ne1 - r)) for r in range(0, ne1, step))
        for b in S.bounded_imap(pool, _job, jobs, depth):
            if isinstance(b, str):
                raise S.ConvertError(f"{name}: {b}")
            yield b


def _job(args):
    if args[0] == "experts":
        return _expert_blobs(*args[1:])
    _, path, off, ne0, row0, rows = args    # BF16 rows [row0, row0 + rows) as Q8_0
    mm = np.memmap(path, dtype="<u2", mode="r", offset=off + row0 * ne0 * 2, shape=(rows, ne0))
    x = S._bf16_to_f32(np.ascontiguousarray(mm)).reshape(-1)
    return S.q8_0(x) if np.isfinite(x).all() else "a non-finite value"


# ------------------------------------------------------------------ metadata
def model_kv(ck: C.Checkpoint, name: str | None) -> dict:
    c, tc = ck.config, ck.text
    A = ARCH + "."
    d = ck.dir
    gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8")) if (d / "generation_config.json").is_file() else {}
    card = C.front_matter(d / "README.md")
    n_layers = int(tc["num_hidden_layers"])
    local = set(int(x) for x in tc["local_layer_ids"])
    loop = tc["loop_config"]
    kv = {"general.architecture": ARCH, "general.type": "model"}
    if "top_k" in gen:
        kv["general.sampling.top_k"] = int(gen["top_k"])
    if "top_p" in gen:
        kv["general.sampling.top_p"] = C.f32(gen["top_p"])
    if "temperature" in gen:
        kv["general.sampling.temp"] = C.f32(gen["temperature"])
    if name:
        kv["general.name"] = name
    if "license" in card:
        kv["general.license"] = card["license"]
    if (int(tc["swa_head_dim"]), int(tc["swa_num_attention_heads"]), int(tc["swa_num_key_value_heads"])) != \
            (int(tc["head_dim"]), int(tc["num_attention_heads"]), int(tc["num_key_value_heads"])):
        raise S.ConvertError("config.json: the sliding-window layers' heads differ from the full layers'")
    if int(tc["rel_extent"]) != int(tc["sliding_window_size"]):
        raise S.ConvertError("config.json: rel_extent differs from the window (the tables are one width)")
    kv.update({
        A + "block_count": n_layers,
        A + "context_length": int(tc["model_max_length"]),
        A + "embedding_length": int(tc["hidden_size"]),
        A + "vocab_size": int(tc["vocab_size"]),
        A + "attention.head_count": int(tc["num_attention_heads"]),
        A + "attention.head_count_kv": int(tc["num_key_value_heads"]),
        A + "attention.key_length": int(tc["head_dim"]),
        A + "attention.layer_norm_rms_epsilon": C.f32(tc["rms_norm_eps"]),
        A + "attention.sliding_window": int(tc["sliding_window_size"]),
        A + "attention.sliding_window_pattern": [l in local for l in range(n_layers)],
        A + "attention.relative_length": int(tc["d_rel"]),
        A + "attention.relative_extent": int(tc["rel_extent"]),
        A + "attention.qk_clip": C.f32(tc["qk_l2_clip_threshold"]),
        A + "attention.relative_clip": C.f32(tc["rel_l2_clip_threshold"]),
        A + "attention.log_scale_floor": int(tc["log_scaling_n_floor"]),
        A + "attention.log_scale_alpha": C.f32(tc["log_scaling_alpha"]),
        A + "shortconv.l_cache": int(tc["sconv_kernel_size"]),
        A + "attn_res.block_size": int(tc["n_layers_per_attn_res_block"]),
        A + "expert_count": int(tc["n_routed_experts"]),
        A + "expert_used_count": int(tc["num_experts_per_tok"]),
        A + "expert_feed_forward_length": int(tc["intermediate_size"]),
        A + "expert_weights_scale": C.f32(tc["route_scale"]),
        A + "expert_gate_softcap": C.f32(tc["mlp_gate_softcap"]),
        A + "expert_up_softcap": C.f32(tc["mlp_up_softcap"]),
        A + "final_logit_softcapping": C.f32(tc["final_logit_softcapping"]),
        A + "loop.start": int(loop["core_start"]),
        A + "loop.end": int(loop["core_end"]),
        A + "loop.count": int(loop["num_passes"]),
        A + "loop.gate_rank": int(loop["gate_rank"]),
        A + "image_token_id": 200054,
        "general.file_type": 40,
        "general.quantization_version": 2,
    })
    mc = c.get("mtp_config") or {}
    if mc:
        D = A + "dspark."
        kv.update({
            D + "block_count": int(mc["num_hidden_layers"]),
            D + "block_size": int(mc["block_size"]),
            D + "target_layers": [int(x) for x in mc["target_layer_ids"]],
            D + "sliding_window_pattern": [l in set(mc["local_layer_ids"]) for l in range(int(mc["num_hidden_layers"]))],
            D + "sliding_window": int(mc["sliding_window_size"]),
            D + "attention.head_count": int(mc["num_attention_heads"]),
            D + "attention.head_count_kv": int(mc["num_key_value_heads"]),
            D + "attention.key_length": int(mc["head_dim"]),
            D + "feed_forward_length": int(mc["intermediate_size"]),
            D + "rope.freq_base": C.f32(mc["rope_theta"]),
            D + "attention.layer_norm_rms_epsilon": C.f32(mc["rms_norm_eps"]),
            D + "markov_rank": int(mc["markov_rank"]),
        })
        if mc.get("markov_head_type") != "rnn_hidden" or mc.get("q_bias") or mc.get("o_bias"):
            raise S.ConvertError("config.json: a DSpark drafter other than the rnn_hidden one without biases")
    kv.update(C.tokenizer_kv(ck, "inkling2", INKLING2_HF_SPLIT, int(tc["unpadded_vocab_size"])))
    return kv


def mmproj(ck: C.Checkpoint, out: pathlib.Path, name: str) -> None:
    """The vision tower as the clip GGUF the engine's encoder reads (llama.cpp's inkling2v layout)."""
    vc = ck.config["vision_config"]
    if vc.get("model_type") != "inkling2_vision" or int(vc.get("temporal_patch_size", 2)) != 2:
        raise S.ConvertError("config.json: the vision tower is not Inkling 2's")
    pp = json.loads((ck.dir / "processor_config.json").read_text(encoding="utf-8")).get("image_processor", {})
    h, patch, grid = int(vc["hidden_size"]), int(vc["native_patch_size"]), int(vc["posemb_grid_size"])
    w = GGUFWriter()
    for k, v, t in (("general.architecture", "clip", "string"), ("general.type", "mmproj", "string"),
                    ("general.name", name + " (mmproj)", "string"), ("clip.has_vision_encoder", True, "bool"),
                    ("clip.vision.projector_type", "inkling2v", "string"),
                    ("clip.vision.projection_dim", int(vc["text_hidden_size"]), "u32"),
                    ("clip.vision.image_size", grid * patch, "u32"), ("clip.vision.patch_size", patch, "u32"),
                    ("clip.vision.embedding_length", h, "u32"),
                    ("clip.vision.feed_forward_length", int(h * float(vc.get("mlp_ratio", 4.0))), "u32"),
                    ("clip.vision.block_count", int(vc["num_hidden_layers"]), "u32"),
                    ("clip.vision.attention.head_count", int(vc["num_attention_heads"]), "u32"),
                    ("clip.vision.attention.layer_norm_epsilon", C.f32(vc["layer_norm_eps"]), "f32"),
                    ("clip.vision.spatial_merge_size", int(vc["adapter_merge_size"]), "u32")):
        w.add(k, v, t)
    w.add("clip.vision.image_mean", [float(x) for x in pp.get("image_mean", [0.5] * 3)], "array:f32")
    w.add("clip.vision.image_std", [float(x) for x in pp.get("image_std", [0.5] * 3)], "array:f32")
    if vc.get("projector_bias"):
        raise S.ConvertError("config.json: a vision projector with biases")

    def put(gname, x, ty):
        if ty == "BF16":
            u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32) >> 16
            w.add_raw(gname, list(reversed(x.shape)), "BF16", u.astype("<u2").tobytes())
        else:
            a = np.ascontiguousarray(x, dtype="<f4")
            w.add_raw(gname, list(reversed(a.shape)), "F32", a.tobytes())

    pw = ck.values("model.visual.backbone.embeddings.patch_embedding.weight")   # [out, (py, px, c)]
    if pw.shape != (h, 3 * patch * patch):
        raise S.ConvertError(f"the patch embedding has shape {list(pw.shape)}")
    put("v.patch_embd.weight", pw.reshape(h, patch, patch, 3).transpose(0, 3, 1, 2), "F32")
    for k, (g, ty) in VISION_GLOBAL.items():
        put(g, ck.values(k), ty)
    for l in range(int(vc["num_hidden_layers"])):
        for k, (g, ty) in VISION_BLOCK.items():
            put(f"v.blk.{l}.{g}", ck.values(f"model.visual.backbone.layers.{l}.{k}"), ty)
    w.write(out)


# ------------------------------------------------------------------ the pack
LICENSE = {"apache-2.0": "Apache License 2.0"}


def build(a, Job, fingerprint, die) -> None:
    import os
    import tokenizer
    out = pathlib.Path(a.out)
    partial = pathlib.Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    try:
        ck = C.Checkpoint(a.checkpoint, side={"mtp.safetensors": "mtp."})
        plan = Plan(ck)
        experts = Experts(ck)
        name = a.name or "Inkling 2 Mini"
        kv = model_kv(ck, name)
    except S.ConvertError as e:
        die(str(e))
    partial.unlink(missing_ok=True)
    w = ModelWriter(str(partial))
    job = Job(w)
    ok = False
    try:
        with tempfile.TemporaryDirectory() as tmp, multiprocessing.Pool(a.workers) as pool:
            tokenizer.extract_metadata(kv, pathlib.Path(tmp))
            print("writing sections:", flush=True)
            for p in sorted(q for q in (pathlib.Path(tmp) / "tokenizer").iterdir() if q.is_file()):
                job.file(f"tokenizer/{p.name}", p)
            job.chunks("experts", experts.chunks(pool))
            job.bytes("experts.layout", experts.layout_text())
            t = time.time()
            parts, names = [], []
            for n, info in plan.tensors.items():
                parts.append(Stream(info.expected_bytes(), plan.chunks(n, pool)) if info.type_name == "Q8_0" else plan.float_bytes(n))
                names.append(n)
            offs = w.add_parts("gguf", parts, TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            mp = pathlib.Path(tmp) / "mmproj.gguf"
            mmproj(ck, mp, name)
            job.file("vision/mmproj", mp)
            tens = {}
            for n, off in zip(names, offs):
                t_ = plan.tensors[n]
                tens[n] = {"type": t_.type_name, "type_id": t_.type_id, "shape": [int(d) for d in t_.ne],
                           "bytes": t_.expected_bytes(), "offset": off}
            lic = kv.get("general.license", "")
            mx = [l for l, kind, _, _, _ in experts.layers if kind == "q8_0"]
            meta = {
                "format": "basalt", "version": 2, "name": name, "arch": ARCH,
                "converter": {"tool": "tools/convert/convert.py"},
                "sources": [{"path": p.name, "size": p.stat().st_size,
                             "fingerprint_sha256_first_last_64mib_and_size": fingerprint(p)} for p in ck.files],
                "gguf_kv": kv,
                "tensors": tens,
                "license": {"name": LICENSE.get(lic, lic)},
                "hf_config": ck.config,
                "vision": {"projector": "inkling2v", "source": "the checkpoint's BF16 vision tower"},
                "dspark": {"source": "mtp.safetensors", "form": "BF16 and F32 as stored"} if ck.config.get("mtp_config") else None,
                "checkpoint": {
                    "dir": pathlib.Path(a.checkpoint).name,
                    "config_sha256": hashlib.sha256((ck.dir / "config.json").read_bytes()).hexdigest(),
                    "experts_nvfp4": "w13 / w2 E2M1 codes and E4M3 block scales as stored (gate the even rows of w13, up "
                                     "the odd), FP32 tail amax / outer_scale_denominator (rounded once)",
                    **({"experts_mxfp8": "layers " + ", ".join(map(str, mx)) + ": native MXFP8 (type 46: 4 UE8M0 scales + "
                                         "128 E4M3 codes per 132-byte block, codes and scales as stored)"} if mx else {}),
                    "bf16_projections": "Q8_0 (ggml reference rounding)",
                    "transforms": {k: METHODS[k] for k in sorted({t_.how for t_ in plan.tensors.values()})}},
            }
            job.bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        w.close()
        ok = True
    except S.ConvertError as e:
        die(str(e))
    finally:
        if not ok:
            try:
                w.f.close()
            except OSError:
                pass
            partial.unlink(missing_ok=True)
    os.replace(partial, out)
    print(f"done: {len(job.rows)} sections, {out.stat().st_size / (1 << 30):.3f} GiB, {time.time() - job.t0:.1f} s -> {out}")
