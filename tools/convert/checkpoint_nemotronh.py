"""A nemotronh pack (NVIDIA Nemotron-H: a hybrid of Mamba-2, attention and latent mixture-of-experts layers) straight
from NVIDIA's NVFP4 checkpoint (Model Optimizer's mixed-precision export, as in
nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4), for tools/convert/convert.py.

  experts             per MoE layer, each routed expert's NVFP4 rows moved into Basalt's 64-value blocks (E2M1 codes and
                        UE4M3 scales unchanged): [up rows (n_ff x latent) | down rows (latent x n_ff)], then the FP32 tail
                        {s_up, 0, s_down, 0} from each matrix's weight_scale_2.  The experts are squared-ReLU, h =
                        relu(up x)^2, with no gate rows.
  experts.layout      one `layer gu_type d_type offset blob_bytes` line per MoE layer (gu_type: the up rows' type)
  layer kinds         nemotronh.layer_pattern, one letter a layer: M Mamba-2, * attention, E MoE
  gguf                every other language-model tensor in llama.cpp's nemotron_h_moe names:
                        FP8 projections (the Mamba in / out, attention q / k / v / o and the head): the E4M3 codes as
                          stored (F8_E4M3), the FP32 weight_scale as `<name>.scale` (F32 [1])
                        the NVFP4 shared experts: NVFP4_PLANES (the code plane as stored, then the UE4M3 scales in
                          128-row x 4-group tiles), the FP32 weight_scale_2 as `<name>.scale`
                        the BF16 latent projections and token embedding as stored; the router, its bias, the norms, the
                          convolution (weights and bias), dt bias and D as F32 (exact); ssm_a = -exp(A_log) as F32, the
                          exponential of the BF16 value in float64, rounded once
  tokenizer/*, meta   from config.json, generation_config.json, the tokenizer files and the model card

The vision tower and the multi-token-prediction head are not converted.  Activation scales (input_scale) are unused:
the engine scales activations at run time.  Needs numpy.
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
import checkpoint_qwen35 as Q
import model_sections as S

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from model_file import TENSOR_ALIGN, ModelWriter, Stream  # noqa: E402

ARCH = "nemotronh"
TYPE_ID = {"F32": 0, "BF16": 30, "NVFP4": 40, "F8_E4M3": 44, "NVFP4_PLANES": 45}
ROWS_PER_JOB = 1 << 22
EXPERTS_PER_JOB = 16
NVFP4_TAIL = 16
E4M3 = S.fp8_e4m3_table()
# the layer kinds and the letters the pack records them by (nemotronh.layer_pattern: a string, as a GGUF array of 88
# would pass the pack's meta parser's 64 items)
PATTERN = {"M": "mamba", "*": "attention", "E": "moe"}
KIND = {v: k for k, v in PATTERN.items()}
# the Mistral (tekken) split of the checkpoint's tokenizer.json: o200k's expression without the contractions, one
# digit a piece
NEMOTRONH_HF_SPLIT = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
                      r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
                      r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")

# checkpoint tensor after `<prefix>layers.N.` -> (GGUF name after `blk.N.`, form, transform)
#   forms: F8 (E4M3 codes as stored, with <name>.scale), FP4 (NVFP4_PLANES, with <name>.scale), BF16 (as stored),
#          F32 (BF16 widened, or F32 as stored)
#   transforms: id; conv ([C, 1, K] -> [C, K], tap K-1 the current token); neg_exp (-exp(x))
LAYER = {
    "norm.weight": ("attn_norm.weight", "F32", "id"),
    "mixer.in_proj.weight": ("ssm_in.weight", "F8", "id"),
    "mixer.conv1d.weight": ("ssm_conv1d.weight", "F32", "conv"),
    "mixer.conv1d.bias": ("ssm_conv1d.bias", "F32", "id"),
    "mixer.dt_bias": ("ssm_dt.bias", "F32", "id"),
    "mixer.A_log": ("ssm_a", "F32", "neg_exp"),
    "mixer.D": ("ssm_d", "F32", "id"),
    "mixer.norm.weight": ("ssm_norm.weight", "F32", "id"),
    "mixer.out_proj.weight": ("ssm_out.weight", "F8", "id"),
    "mixer.q_proj.weight": ("attn_q.weight", "F8", "id"),
    "mixer.k_proj.weight": ("attn_k.weight", "F8", "id"),
    "mixer.v_proj.weight": ("attn_v.weight", "F8", "id"),
    "mixer.o_proj.weight": ("attn_output.weight", "F8", "id"),
    "mixer.gate.weight": ("ffn_gate_inp.weight", "F32", "id"),
    "mixer.gate.e_score_correction_bias": ("exp_probs_b.bias", "F32", "id"),
    "mixer.fc1_latent_proj.weight": ("ffn_latent_down.weight", "BF16", "id"),
    "mixer.fc2_latent_proj.weight": ("ffn_latent_up.weight", "BF16", "id"),
    "mixer.shared_experts.up_proj.weight": ("ffn_up_shexp.weight", "FP4", "id"),
    "mixer.shared_experts.down_proj.weight": ("ffn_down_shexp.weight", "FP4", "id"),
}
GLOBAL = {
    "embeddings.weight": ("token_embd.weight", "BF16", "id"),
    "norm_f.weight": ("output_norm.weight", "F32", "id"),
}
HEAD = ("output.weight", "F8", "id")
# what each layer kind reads, by the checkpoint names above
NEEDS = {
    "mamba": ("norm.weight", "mixer.in_proj.weight", "mixer.conv1d.weight", "mixer.conv1d.bias", "mixer.dt_bias",
              "mixer.A_log", "mixer.D", "mixer.norm.weight", "mixer.out_proj.weight"),
    "attention": ("norm.weight", "mixer.q_proj.weight", "mixer.k_proj.weight", "mixer.v_proj.weight", "mixer.o_proj.weight"),
    "moe": ("norm.weight", "mixer.gate.weight", "mixer.gate.e_score_correction_bias", "mixer.fc1_latent_proj.weight",
            "mixer.fc2_latent_proj.weight", "mixer.shared_experts.up_proj.weight", "mixer.shared_experts.down_proj.weight"),
}
QUANT_SIDE = re.compile(r"\.(weight_scale|weight_scale_2|input_scale|k_scale|v_scale)$")
SKIP = re.compile(r"^(vision_model\.|vision_projector\.|mlp1\.|(language_model\.)?mtp\.)")


def text_config(cfg: dict) -> dict:
    """The language model's config: a VL checkpoint's llm_config, else the file's own."""
    return cfg.get("llm_config", cfg.get("text_config", cfg))


def layer_kinds(tc: dict) -> list[str]:
    """The layer kinds; the count is num_hidden_layers where the config states it (the VL export's does not)."""
    if "layers_block_type" in tc:
        kinds = [str(k) for k in tc["layers_block_type"]]
    elif "hybrid_override_pattern" in tc:
        pat = str(tc["hybrid_override_pattern"])
        if any(c not in PATTERN for c in pat):
            raise S.ConvertError(f"config.json: hybrid_override_pattern {pat!r} has a layer kind other than M, * and E")
        kinds = [PATTERN[c] for c in pat]
    else:
        raise S.ConvertError("config.json: neither layers_block_type nor hybrid_override_pattern")
    n = int(tc.get("num_hidden_layers", len(kinds)))
    if len(kinds) != n or any(k not in KIND for k in kinds):
        raise S.ConvertError(f"config.json: {len(kinds)} layer kinds for {n} layers, or a kind other than mamba, "
                             "attention and moe")
    return kinds


class Tensor:
    def __init__(self, name, form, src, how, ne):
        self.name, self.form, self.src, self.how, self.ne = name, form, src, how, ne
        self.type_name = {"F8": "F8_E4M3", "FP4": "NVFP4_PLANES"}.get(form, form)
        self.type_id = TYPE_ID[self.type_name]

    def expected_bytes(self) -> int:
        n = int(np.prod(self.ne))
        return {"F32": 4 * n, "BF16": 2 * n, "F8_E4M3": n, "NVFP4_PLANES": n // 64 * 36}[self.type_name]


class Geometry:
    """The checked shape of the language model, from its config."""

    def __init__(self, tc: dict):
        def need(key, want=None):
            if key not in tc:
                raise S.ConvertError(f"config.json: no {key}")
            if want is not None and tc[key] != want:
                raise S.ConvertError(f"config.json: {key} = {tc[key]!r}; the engine runs {want!r}")
            return tc[key]

        self.kinds = layer_kinds(tc)
        self.n_layers = len(self.kinds)
        self.h = int(need("hidden_size"))
        self.vocab = int(need("vocab_size"))
        self.n_head, self.n_kv, self.head_dim = int(need("num_attention_heads")), int(need("num_key_value_heads")), int(need("head_dim"))
        self.m_heads, self.m_head_dim = int(need("mamba_num_heads")), int(need("mamba_head_dim"))
        self.d_state, self.n_groups, self.d_conv = int(need("ssm_state_size")), int(need("n_groups")), int(need("conv_kernel"))
        self.d_inner = self.m_heads * self.m_head_dim
        self.conv_dim = self.d_inner + 2 * self.n_groups * self.d_state
        self.n_expert, self.top_k = int(need("n_routed_experts")), int(need("num_experts_per_tok"))
        self.ff, self.ff_sh = int(need("moe_intermediate_size")), int(need("moe_shared_expert_intermediate_size"))
        lat = tc.get("moe_latent_size")
        self.latent = int(lat) if lat else self.h
        self.has_latent = bool(lat)
        for k, want in (("mamba_hidden_act", "silu"), ("mlp_hidden_act", "relu2"), ("use_conv_bias", True),
                        ("mamba_proj_bias", False), ("use_bias", False), ("attention_bias", False), ("mlp_bias", False),
                        ("n_group", 1), ("topk_group", 1), ("norm_topk_prob", True), ("tie_word_embeddings", False),
                        ("residual_in_fp32", False)):
            need(k, want)
        if int(tc.get("n_shared_experts", 1)) != 1:
            raise S.ConvertError(f"config.json: n_shared_experts = {tc.get('n_shared_experts')}; the engine runs one")
        if tc.get("sliding_window") is not None:
            raise S.ConvertError("config.json: a sliding window; the engine runs full attention")
        if self.d_inner % self.n_groups or self.m_heads % self.n_groups or self.n_head % self.n_kv:
            raise S.ConvertError("config.json: heads that do not divide into their groups")
        self.scale = float(need("routed_scaling_factor"))
        self.eps = float(tc.get("layer_norm_epsilon", tc.get("norm_eps", 1e-5)))
        self.context = int(need("max_position_embeddings"))


class Experts:
    """The `experts` section: per MoE layer the blobs of its routed experts, from the per-expert NVFP4 tensors."""

    def __init__(self, ck: C.Checkpoint, g: Geometry, prefix: str):
        self.ck, self.g, self.prefix = ck, g, prefix
        self.layers, offset = [], 0
        self.blob = g.ff * S.nvfp4_row_bytes(g.latent) + g.latent * S.nvfp4_row_bytes(g.ff) + NVFP4_TAIL
        for l, kind in enumerate(g.kinds):
            if kind != "moe":
                continue
            for e in range(g.n_expert):
                p = f"{prefix}layers.{l}.mixer.experts.{e}."
                ck.need(p + "up_proj.weight", "U8", (g.ff, g.latent // 2))
                ck.need(p + "up_proj.weight_scale", "F8_E4M3", (g.ff, g.latent // 16))
                ck.need(p + "up_proj.weight_scale_2", "F32", ())
                ck.need(p + "down_proj.weight", "U8", (g.latent, g.ff // 2))
                ck.need(p + "down_proj.weight_scale", "F8_E4M3", (g.latent, g.ff // 16))
                ck.need(p + "down_proj.weight_scale_2", "F32", ())
            self.layers.append((l, offset))
            offset += self.blob * g.n_expert
        self.total = offset

    def layout_text(self) -> bytes:
        head = ("# experts v7 relu2: layer gu_type d_type offset blob_bytes (n_expert %d, total %d); a blob is [up | down] "
                "with no gate rows, then FP32 {s_up, 0, s_down, 0}\n" % (self.g.n_expert, self.total))
        return (head + "".join("%d %d %d %d %d\n" % (l, S.NVFP4, S.NVFP4, off, self.blob)
                               for l, off in self.layers)).encode()

    def jobs(self):
        ck, g = self.ck, self.g
        for l, _ in self.layers:
            for s in range(0, g.n_expert, EXPERTS_PER_JOB):
                ents = []
                for e in range(s, min(g.n_expert, s + EXPERTS_PER_JOB)):
                    p = f"{self.prefix}layers.{l}.mixer.experts.{e}."
                    ents.append(tuple((str(ck.t[p + w + k][0]), ck.t[p + w + k][1])
                                      for w in ("up_proj.", "down_proj.") for k in ("weight", "weight_scale", "weight_scale_2")))
                yield ("experts", ents, g.latent, g.ff)

    def chunks(self, pool, depth: int = 8):
        for b in S.bounded_imap(pool, _job, self.jobs(), depth):
            if isinstance(b, str):
                raise S.ConvertError(b)
            yield b


def _expert_blobs(ents, lat, ff) -> bytes | str:
    """Blobs of a run of experts: [up | down] blocks, then the FP32 tail {s_up, 0, s_down, 0}."""
    out = []
    for (uw, us, us2, dw, ds, ds2) in ents:
        parts, tail = [], []
        for (wp, wo), (sp, so), (tp, to), rows, cols in ((uw, us, us2, ff, lat), (dw, ds, ds2, lat, ff)):
            w = np.fromfile(wp, dtype=np.uint8, count=rows * cols // 2, offset=wo).reshape(rows, cols // 2)
            sc = np.fromfile(sp, dtype=np.uint8, count=rows * cols // 16, offset=so).reshape(rows, cols // 16)
            if (sc & 0x80).any() or (sc == 0x7F).any():
                return f"a UE4M3 scale with the sign bit set or NaN at {pathlib.Path(sp).name}+{so}"
            tail.append(np.fromfile(tp, dtype="<f4", count=1, offset=to)[0])
            parts.append(C.nvfp4_rows(w, sc))
        t = np.array([tail[0], 0.0, tail[1], 0.0], dtype="<f4")
        if not (np.isfinite(t).all() and t[0] > 0 and t[2] > 0):
            return f"a weight_scale_2 that is not finite and positive: {[float(tail[0]), float(tail[1])]}"
        out.append(np.concatenate(parts).tobytes())
        out.append(t.tobytes())
    return b"".join(out)


class Plan:
    """The pack's `gguf` tensors in checkpoint order, each with how it is made."""

    def __init__(self, ck: C.Checkpoint, g: Geometry, prefix: str, head_src: str):
        self.ck, self.g, self.prefix = ck, g, prefix
        self.tensors: dict[str, Tensor] = {}
        self.scales: dict[str, str] = {}   # <name>.scale -> the checkpoint's scalar
        unknown = []
        lp = re.escape(prefix) + r"layers\.(\d+)\.(.+)"
        for k in ck.order:
            if SKIP.match(k) or QUANT_SIDE.search(k) or re.search(r"\.mixer\.experts\.\d+\.", k):
                continue
            _, _, dt, shape, _ = ck.t[k]
            m = re.fullmatch(lp, k)
            if m:
                rule = LAYER.get(m.group(2))
                if rule is None:
                    unknown.append(k)
                    continue
                gname, form, how = f"blk.{m.group(1)}.{rule[0]}", rule[1], rule[2]
            elif k.startswith(prefix) and k[len(prefix):] in GLOBAL:
                gname, form, how = GLOBAL[k[len(prefix):]]
            elif k == head_src:
                gname, form, how = HEAD
            else:
                unknown.append(k)
                continue
            if how == "conv":
                if len(shape) != 3 or shape[1] != 1:
                    raise S.ConvertError(f"{k}: a conv weight of shape {shape}")
                ne = [shape[2], shape[0]]
            else:
                ne = list(reversed(shape))
            self._add(gname, form, k, how, ne, dt)
        if unknown:
            raise S.ConvertError("checkpoint tensors with no pack counterpart: " + ", ".join(unknown[:8]))
        self._check_complete()

    def _add(self, gname, form, src, how, ne, dt):
        want = {"F8": ("F8_E4M3",), "FP4": ("U8",), "BF16": ("BF16",), "F32": ("BF16", "F32")}[form]
        if dt not in want:
            raise S.ConvertError(f"{src} is {dt}, expected {' or '.join(want)}")
        if gname in self.tensors:
            raise S.ConvertError(f"two checkpoint tensors map to {gname}")
        base = src[:-len(".weight")]
        if form == "F8":
            if len(ne) != 2 or ne[0] % 256:
                raise S.ConvertError(f"{src}: F8_E4M3 rows of {ne[0]} values (the kernels read multiples of 256)")
            self.ck.need(base + ".weight_scale", "F32", ())
            self.scales[gname[:-len(".weight")] + ".scale"] = base + ".weight_scale"
        elif form == "FP4":
            rows, K = ne[1], ne[0] * 2   # U8 [rows, K / 2]
            if rows % 128 or K % 64:
                raise S.ConvertError(f"{src}: an NVFP4 matrix of {rows} x {K} (NVFP4_PLANES needs rows in 128s, K in 64s)")
            self.ck.need(base + ".weight_scale", "F8_E4M3", (rows, K // 16))
            self.ck.need(base + ".weight_scale_2", "F32", ())
            self.scales[gname[:-len(".weight")] + ".scale"] = base + ".weight_scale_2"
            ne = [K, rows]
        self.tensors[gname] = Tensor(gname, form, src, how, ne)

    def _check_complete(self):
        g = self.g
        need = [r[0] for r in GLOBAL.values()] + [HEAD[0]]
        for l, kind in enumerate(g.kinds):
            need += [f"blk.{l}.{LAYER[k][0]}" for k in NEEDS[kind]]
        missing = [n for n in need if n not in self.tensors]
        if missing:
            raise S.ConvertError(f"the checkpoint lacks {len(missing)} tensors the engine reads: " + ", ".join(missing[:8]))
        shapes = {"token_embd.weight": [g.h, g.vocab], "output.weight": [g.h, g.vocab], "output_norm.weight": [g.h]}
        for l, kind in enumerate(g.kinds):
            b = f"blk.{l}."
            shapes[b + "attn_norm.weight"] = [g.h]
            if kind == "mamba":
                shapes.update({b + "ssm_in.weight": [g.h, 2 * g.d_inner + 2 * g.n_groups * g.d_state + g.m_heads],
                               b + "ssm_conv1d.weight": [g.d_conv, g.conv_dim], b + "ssm_conv1d.bias": [g.conv_dim],
                               b + "ssm_dt.bias": [g.m_heads], b + "ssm_a": [g.m_heads], b + "ssm_d": [g.m_heads],
                               b + "ssm_norm.weight": [g.d_inner], b + "ssm_out.weight": [g.d_inner, g.h]})
            elif kind == "attention":
                shapes.update({b + "attn_q.weight": [g.h, g.n_head * g.head_dim], b + "attn_k.weight": [g.h, g.n_kv * g.head_dim],
                               b + "attn_v.weight": [g.h, g.n_kv * g.head_dim], b + "attn_output.weight": [g.n_head * g.head_dim, g.h]})
            else:
                shapes.update({b + "ffn_gate_inp.weight": [g.h, g.n_expert], b + "exp_probs_b.bias": [g.n_expert],
                               b + "ffn_up_shexp.weight": [g.h, g.ff_sh], b + "ffn_down_shexp.weight": [g.ff_sh, g.h]})
                if g.has_latent:
                    shapes.update({b + "ffn_latent_down.weight": [g.h, g.latent], b + "ffn_latent_up.weight": [g.latent, g.h]})
        for n, want in shapes.items():
            if self.tensors[n].ne != want:
                raise S.ConvertError(f"{self.tensors[n].src} has shape {list(reversed(self.tensors[n].ne))}, "
                                     f"expected {list(reversed(want))}")
        if not g.has_latent and any(n.endswith(("ffn_latent_down.weight", "ffn_latent_up.weight")) for n in self.tensors):
            raise S.ConvertError("latent projections in a checkpoint whose config has no moe_latent_size")

    # ---- float tensors
    def float_bytes(self, name) -> bytes:
        t = self.tensors[name]
        x = self.ck.values(t.src)
        if t.how == "conv":
            x = x[:, 0, :]
        elif t.how == "neg_exp":
            x = -np.exp(x.astype(np.float64))
        x = np.ascontiguousarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            raise S.ConvertError(f"{name} has a non-finite value")
        if t.type_name == "F32":
            return x.astype("<f4").tobytes()
        u = x.view(np.uint32)
        if np.count_nonzero(u & 0xFFFF):
            raise S.ConvertError(f"{name}: a value that is not BF16")
        return (u >> 16).astype("<u2").tobytes()

    def scale_bytes(self, sname) -> bytes:
        v = np.float64(self.ck.values(self.scales[sname]).reshape(-1)[0])
        if not (np.isfinite(v) and v > 0):
            raise S.ConvertError(f"{self.scales[sname]} is not finite and positive")
        return np.array([v], dtype="<f4").tobytes()

    # ---- large tensors, streamed
    def jobs(self, name):
        t = self.tensors[name]
        p, off, _, _, _ = self.ck.t[t.src]
        ne0, ne1 = t.ne
        if t.form == "FP4":   # the code plane, then the scale plane in 128-row tiles
            sp, soff = self.ck.t[t.src[:-len(".weight")] + ".weight_scale"][:2]
            step = max(128, ROWS_PER_JOB // ne0 // 128 * 128)
            for r in range(0, ne1, step):
                yield ("fp4c", str(p), off, ne0, ne1, r, min(step, ne1 - r), None, 0)
            for r in range(0, ne1, step):
                yield ("fp4s", str(sp), soff, ne0, ne1, r, min(step, ne1 - r), None, 0)
            return
        step = max(1, ROWS_PER_JOB // ne0)
        for r in range(0, ne1, step):
            yield ("rows", t.form, str(p), off, ne0, r, min(step, ne1 - r))

    def chunks(self, name, pool, depth: int = 64):
        for b in S.bounded_imap(pool, _job, self.jobs(name), depth):
            if isinstance(b, str):
                raise S.ConvertError(f"{name}: {b}")
            yield b


def _job(args):
    if args[0] == "experts":
        return _expert_blobs(*args[1:])
    if args[0] in ("fp4c", "fp4s"):
        return Q._job(args)
    _, form, path, off, ne0, row0, rows = args
    if form == "F8":   # the E4M3 codes as stored
        u = np.fromfile(path, dtype=np.uint8, count=rows * ne0, offset=off + row0 * ne0)
        if ((u & 0x7F) == 0x7F).any():
            return "an E4M3 NaN code"
        return u.tobytes()
    u = np.fromfile(path, dtype="<u2", count=rows * ne0, offset=off + row0 * ne0 * 2)   # BF16 rows as stored
    if not np.isfinite(S._bf16_to_f32(u)).all():
        return "a non-finite value"
    return u.tobytes()


# ------------------------------------------------------------------ metadata
def model_kv(ck: C.Checkpoint, g: Geometry, tc: dict, name: str | None) -> dict:
    A = ARCH + "."
    d = ck.dir
    gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8")) if (d / "generation_config.json").is_file() else {}
    card = C.front_matter(d / "README.md")
    kv = {"general.architecture": ARCH, "general.type": "model"}
    for k, gk in (("top_k", "general.sampling.top_k"), ("top_p", "general.sampling.top_p"), ("temperature", "general.sampling.temp")):
        if k in gen:
            kv[gk] = int(gen[k]) if k == "top_k" else C.f32(gen[k])
    if name:
        kv["general.name"] = name
    if "license" in card:
        kv["general.license"] = card["license"]
    if "license_name" in card:
        kv["general.license.name"] = card["license_name"]
    kv.update({
        A + "block_count": g.n_layers,
        A + "layer_pattern": "".join(KIND[k] for k in g.kinds),
        A + "context_length": g.context,
        A + "embedding_length": g.h,
        A + "vocab_size": g.vocab,
        A + "attention.head_count": g.n_head,
        A + "attention.head_count_kv": g.n_kv,
        A + "attention.key_length": g.head_dim,
        A + "attention.layer_norm_rms_epsilon": C.f32(g.eps),
        A + "ssm.conv_kernel": g.d_conv,
        A + "ssm.state_size": g.d_state,
        A + "ssm.head_count": g.m_heads,
        A + "ssm.head_dim": g.m_head_dim,
        A + "ssm.group_count": g.n_groups,
        A + "ssm.inner_size": g.d_inner,
        A + "expert_count": g.n_expert,
        A + "expert_used_count": g.top_k,
        A + "expert_feed_forward_length": g.ff,
        A + "expert_shared_feed_forward_length": g.ff_sh,
        A + "expert_latent_size": g.latent if g.has_latent else 0,
        A + "expert_weights_scale": C.f32(g.scale),
        "general.file_type": 40,
        "general.quantization_version": 2,
    })
    ck.text = tc   # tokenizer_kv reads the vocabulary size from it
    kv.update(C.tokenizer_kv(ck, "nemotronh", NEMOTRONH_HF_SPLIT, g.vocab))
    return kv


# ------------------------------------------------------------------ the pack
def checkpoint_kind(cfg: dict) -> str | None:
    """'nemotronh' for a Nemotron-H checkpoint (plain or the VL one), else None."""
    tc = text_config(cfg)
    if tc.get("model_type") == "nemotron_h" or "NemotronHForCausalLM" in tc.get("architectures", []):
        return "nemotronh"
    return None


def build(a, Job, fingerprint, die) -> None:
    import os
    import tokenizer
    out = pathlib.Path(a.out)
    partial = pathlib.Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    try:
        ck = C.Checkpoint(a.checkpoint)
        tc = text_config(ck.config)
        g = Geometry(tc)
        vl = any(k.startswith("language_model.") for k in ck.t)
        prefix = "language_model.backbone." if vl else "backbone."
        head_src = ("language_model." if vl else "") + "lm_head.weight"
        plan = Plan(ck, g, prefix, head_src)
        experts = Experts(ck, g, prefix)
        name = a.name or str(ck.config.get("_name_or_path") or ck.dir.name)
        kv = model_kv(ck, g, tc, name)
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
            parts, names, sizes = [], [], {}
            for n, info in plan.tensors.items():
                if info.form in ("F8", "FP4") or (info.form == "BF16" and info.expected_bytes() > ROWS_PER_JOB):
                    parts.append(Stream(info.expected_bytes(), plan.chunks(n, pool)))
                else:
                    parts.append(plan.float_bytes(n))
                names.append(n)
                if info.form in ("F8", "FP4"):
                    s = n[:-len(".weight")] + ".scale"
                    parts.append(plan.scale_bytes(s))
                    names.append(s)
                    sizes[s] = 1
            offs = w.add_parts("gguf", parts, TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            tens = {}
            for n, off in zip(names, offs):
                if n in sizes:
                    tens[n] = {"type": "F32", "type_id": 0, "shape": [1], "bytes": 4, "offset": off}
                    continue
                t_ = plan.tensors[n]
                tens[n] = {"type": t_.type_name, "type_id": t_.type_id, "shape": [int(d) for d in t_.ne],
                           "bytes": t_.expected_bytes(), "offset": off}
            meta = {
                "format": "basalt", "version": 2, "name": name, "arch": ARCH,
                "converter": {"tool": "tools/convert/convert.py"},
                "sources": [{"path": p.name, "size": p.stat().st_size,
                             "fingerprint_sha256_first_last_64mib_and_size": fingerprint(p)} for p in ck.shards],
                "gguf_kv": kv,
                "tensors": tens,
                "license": {"name": kv.get("general.license.name", kv.get("general.license", ""))},
                "hf_config": ck.config,
                "checkpoint": {
                    "dir": pathlib.Path(a.checkpoint).name,
                    "config_sha256": hashlib.sha256((ck.dir / "config.json").read_bytes()).hexdigest(),
                    "experts_nvfp4": "up_proj / down_proj E2M1 codes and E4M3 block scales as stored, FP32 tail "
                                     "{weight_scale_2 of up, 0, weight_scale_2 of down, 0}; squared-ReLU, no gate rows",
                    "fp8": "weight (E4M3 codes) as stored, weight_scale as <name>.scale; input_scale unused",
                    "nvfp4_shared": "NVFP4_PLANES: weight (E2M1 pairs) as stored, then weight_scale (UE4M3 per 16) in "
                                    "128-row x 4-group tiles, unchanged; weight_scale_2 as <name>.scale",
                    "ssm_a": "-exp(A_log), the exponential in float64, rounded once to F32",
                    "not_converted": "the vision tower and the MTP head"},
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
