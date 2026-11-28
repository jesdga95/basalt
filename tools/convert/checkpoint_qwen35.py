"""A whole qwen35 pack (Qwen3.5-family dense hybrid: GDN and gated attention layers, dense SwiGLU) straight from an
NVFP4 checkpoint, for tools/convert/convert.py: NVIDIA Model Optimizer's (as nvidia/Qwen3.8-27B-NVFP4: weight,
weight_scale, weight_scale_2) or compressed-tensors' (as QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4: weight_packed,
weight_scale, weight_global_scale, its reciprocal the tensor scale).

  gguf                every language-model and drafter tensor in GGUF names:
                        NVFP4 projections (MLP, head; GDN and attention where the checkpoint has them in NVFP4):
                          NVFP4_PLANES (type 45): the E2M1 code plane as stored, then the UE4M3 scales in 128-row x
                          4-group tiles (unchanged values; rows / columns in the GGUF head order); the FP32 tensor
                          scale as `<name>.scale` (F32 [1])
                        FP8 projections: the E4M3 codes as stored (type 44, F8_E4M3; rows / columns in the GGUF
                          head order), the FP32 per-tensor weight_scale as `<name>.scale` (F32 [1])
                        a BF16 head: rounded once to F8_E4M3 rows, each row's FP32 scale its amax / 448 as
                          `output.scale` (F32 [rows]), the codes to nearest; with --head nvfp4 to NVFP4_PLANES (per
                          16 values the UE4M3 scale of least squared error, tensor scale amax / (6 x 448))
                        BF16 projections (the drafter's): Q8_0
                        norms F32 with the offset "+1" folded in (exact), the GDN gated norm as stored
                        GDN value heads in the GGUF order (value head h meets key head h % 16)
                        a / b projections, token embedding: BF16 as stored (NVFP4 a / b: their values rounded once
                          to BF16, as the GDN kernels read them)
  vision/mmproj       the vision tower as llama.cpp's clip GGUF (projector qwen3vl_merger), BF16 / F32 as stored
  vision/text_vocab   the text model's metadata alone, for the vision sidecar's tokenizer
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

ARCH = "qwen35"
TYPE_ID = {"F32": 0, "F16": 1, "Q8_0": 8, "BF16": 30, "NVFP4": 40, "F8_E4M3": 44, "NVFP4_PLANES": 45}
ROWS_PER_JOB = 1 << 22
E4M3 = S.fp8_e4m3_table()
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)
UE4M3 = E4M3[:127].astype(np.float32)     # the finite non-negative codes, ascending

# checkpoint tensor (after `model.language_model.layers.N.` or `mtp.layers.0.`) -> (GGUF name, form, transform)
#   forms: Q8 (Q8_0 from BF16; the main model's FP8 projections become F8, their codes kept, NVFP4 ones FP4), FP4 (NVFP4
#   kept, as planes; F8Q when the checkpoint has the head in BF16: E4M3 rows with a scale per row; FP4Q, that head with
#   --head nvfp4: rounded once to NVFP4), BF16, F32
#   transforms as checkpoint.py names them: id, norm, heads, negexp, gconv, vrows, zrows, vcols
LAYER = {
    "input_layernorm.weight": ("attn_norm.weight", "F32", "norm"),
    "post_attention_layernorm.weight": ("post_attention_norm.weight", "F32", "norm"),
    "linear_attn.in_proj_qkv.weight": ("attn_qkv.weight", "Q8", "vrows"),
    "linear_attn.in_proj_z.weight": ("attn_gate.weight", "Q8", "zrows"),
    "linear_attn.out_proj.weight": ("ssm_out.weight", "Q8", "vcols"),
    "linear_attn.in_proj_a.weight": ("ssm_alpha.weight", "BF16", "heads"),
    "linear_attn.in_proj_b.weight": ("ssm_beta.weight", "BF16", "heads"),
    "linear_attn.A_log": ("ssm_a", "F32", "negexp"),
    "linear_attn.dt_bias": ("ssm_dt.bias", "F32", "heads"),
    "linear_attn.conv1d.weight": ("ssm_conv1d.weight", "F32", "gconv"),
    "linear_attn.norm.weight": ("ssm_norm.weight", "F32", "id"),
    "self_attn.q_proj.weight": ("attn_q.weight", "Q8", "id"),
    "self_attn.k_proj.weight": ("attn_k.weight", "Q8", "id"),
    "self_attn.v_proj.weight": ("attn_v.weight", "Q8", "id"),
    "self_attn.o_proj.weight": ("attn_output.weight", "Q8", "id"),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "F32", "norm"),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", "F32", "norm"),
    "mlp.gate_proj.weight": ("ffn_gate.weight", "FP4", "id"),
    "mlp.up_proj.weight": ("ffn_up.weight", "FP4", "id"),
    "mlp.down_proj.weight": ("ffn_down.weight", "FP4", "id"),
}
GLOBAL = {
    "model.language_model.embed_tokens.weight": ("token_embd.weight", "BF16", "id"),
    "model.language_model.norm.weight": ("output_norm.weight", "F32", "norm"),
    "lm_head.weight": ("output.weight", "FP4", "id"),
    "mtp.fc.weight": ("mtp.fc.weight", "Q8", "id"),
    "mtp.norm.weight": ("mtp.norm.weight", "F32", "norm"),
    "mtp.pre_fc_norm_embedding.weight": ("mtp.pre_fc_norm_embedding.weight", "F32", "norm"),
    "mtp.pre_fc_norm_hidden.weight": ("mtp.pre_fc_norm_hidden.weight", "F32", "norm"),
}
QUANT_SIDE = re.compile(r"\.(weight_scale|weight_scale_2|input_scale|weight_global_scale|input_global_scale)$")
METHODS = dict(C.METHODS, id="as stored")


class Tensor:
    def __init__(self, name, form, src, how, ne):
        self.name, self.form, self.src, self.how, self.ne = name, form, src, how, ne
        self.type_name = {"Q8": "Q8_0", "FP4": "NVFP4_PLANES", "FP4Q": "NVFP4_PLANES", "F8": "F8_E4M3", "F8Q": "F8_E4M3"}.get(form, form)
        self.type_id = TYPE_ID[self.type_name]
        self.packed = False      # an NVFP4 source (codes, weight_scale and a tensor scale beside them)

    def expected_bytes(self) -> int:
        n = int(np.prod(self.ne))
        return {"F32": 4 * n, "BF16": 2 * n, "Q8_0": n // 32 * 34, "NVFP4_PLANES": n // 64 * 36, "F8_E4M3": n}[self.type_name]


class Plan:
    """The pack's `gguf` tensors in checkpoint order, each with how it is made."""

    def __init__(self, ck: C.Checkpoint, head: str | None = None):
        tc = ck.text
        self.ck = ck
        if tc.get("model_type") != "qwen3_5_text":
            raise S.ConvertError(f"config.json: text model_type {tc.get('model_type')}, expected qwen3_5_text")
        for k in ("num_experts", "hc_count"):
            if k in tc:
                raise S.ConvertError(f"config.json: {k} is not a dense qwen3_5 field")
        if int(tc["linear_num_key_heads"]) != S.GDN_K_HEADS or int(tc["linear_num_value_heads"]) != S.GDN_V_HEADS \
                or int(tc["linear_key_head_dim"]) != S.GDN_HEAD or int(tc["linear_value_head_dim"]) != S.GDN_HEAD:
            raise S.ConvertError("the GDN head geometry differs from the GGUF head order this converter knows")
        self.tensors: dict[str, Tensor] = {}
        # NVFP4 tensor -> (its tensor-scale source, True when that holds the reciprocal: compressed-tensors'
        # weight_global_scale, Model Optimizer's weight_scale_2 is the scale); F8 tensor -> (its weight_scale, False)
        self.scales: dict[str, tuple[str, bool]] = {}
        self.amax: dict[str, float] = {}      # FP4Q tensor -> its largest |value|
        self.row_scale: dict[str, np.ndarray] = {}   # F8Q tensor -> its rows' scales (float32)
        unknown = []
        for k in ck.order:
            if k.startswith("model.visual.") or QUANT_SIDE.search(k):
                continue
            key = k[:-len("_packed")] if k.endswith(".weight_packed") else k    # compressed-tensors' NVFP4 codes
            m = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", key)
            mm = re.fullmatch(r"mtp\.layers\.0\.(.+)", key)
            if m:
                rule, name = LAYER.get(m.group(2)), lambda r: f"blk.{m.group(1)}.{r}"
            elif mm:
                rule, name = LAYER.get(mm.group(1)), lambda r: f"mtp.{r}"
            else:
                rule, name = GLOBAL.get(key), lambda r: r
            if rule is None:
                unknown.append(k)
                continue
            gname, form, how = rule[0], rule[1], rule[2]
            gname = name(gname)
            _, _, dt, shape, _ = ck.t[k]
            packed = dt == "U8"
            if mm and form == "FP4":
                form = "Q8"           # the drafter's MLP is BF16 in the checkpoint
            if form == "Q8" and dt == "F8_E4M3":
                form = "F8"           # FP8 projections keep their codes and scale
            if form == "Q8" and packed and not mm:
                form = "FP4"          # NVFP4 projections keep their codes and scales
            if gname == "output.weight" and form == "FP4" and dt == "BF16":
                form = "FP4Q" if head == "nvfp4" else "F8Q"   # a BF16 head: rounded once to FP8 rows or NVFP4
            elif gname == "output.weight" and head == "fp8":
                raise S.ConvertError(f"--head fp8 rounds a BF16 head; {k} is {dt}")
            if form == "FP4" and not packed:
                raise S.ConvertError(f"{k} is {dt}; NVFP4 codes are U8")
            cols = shape[1] * 2 if packed and len(shape) == 2 else shape[-1]
            if form in ("FP4", "FP4Q") and (shape[0] % 128 or cols % 64):
                raise S.ConvertError(f"{k}: NVFP4 planes need rows in 128s and 64-value groups, not {shape}")
            if how == "gconv":
                if len(shape) != 3 or shape[1] != 1:
                    raise S.ConvertError(f"{k}: a conv weight of shape {shape}")
                ne = [shape[2], shape[0]]
            elif packed:
                ne = [cols, shape[0]]
            else:
                ne = list(reversed(shape))
            if form == "Q8" and (len(ne) != 2 or ne[0] % 32):
                raise S.ConvertError(f"{k}: rows of {ne[0]} values cannot be Q8_0")
            if form == "Q8" and dt != "BF16":
                raise S.ConvertError(f"{k} is {dt}; projections are read from FP8 or BF16")
            if form == "BF16" and packed and len(shape) != 2:
                raise S.ConvertError(f"{k}: NVFP4 codes of shape {shape}")
            if form in ("F32", "BF16") and dt != "BF16" and not (form == "BF16" and packed):
                raise S.ConvertError(f"{k} is {dt}, expected BF16")
            if gname in self.tensors:
                raise S.ConvertError(f"two checkpoint tensors map to {gname}")
            t = self.tensors[gname] = Tensor(gname, form, k, how, ne)
            if packed:
                t.packed = True
                base = key[:-len(".weight")]
                ck.need(base + ".weight_scale", "F8_E4M3", (shape[0], cols // 16))
                if base + ".weight_global_scale" in ck.t:
                    ck.need(base + ".weight_global_scale", "F32", (1,))
                    self.scales[gname] = (base + ".weight_global_scale", True)
                else:
                    ck.need(base + ".weight_scale_2", "F32", ())
                    self.scales[gname] = (base + ".weight_scale_2", False)
            if form == "FP4Q":
                self.amax[gname] = _bf16_amax(ck.t[k][0], ck.t[k][1], shape)
            if form == "F8Q":
                if cols % 256:
                    raise S.ConvertError(f"{k}: F8_E4M3 rows of {cols} values (the kernels read multiples of 256)")
                self.row_scale[gname] = f8q_row_scales(_bf16_row_amax(ck.t[k][0], ck.t[k][1], shape))
            if dt == "F8_E4M3":
                ck.need(k[:-len(".weight")] + ".weight_scale", "F32", ())
                self.scales[gname] = (k[:-len(".weight")] + ".weight_scale", False)
        if unknown:
            raise S.ConvertError("checkpoint tensors with no GGUF counterpart: " + ", ".join(unknown[:8]))
        self._check_complete(tc)

    def _check_complete(self, tc):
        n_layers, kinds = int(tc["num_hidden_layers"]), tc["layer_types"]
        need = [r[0] for r in GLOBAL.values()]
        for l in range(n_layers):
            if kinds[l] not in ("linear_attention", "full_attention"):
                raise S.ConvertError(f"config.json: layer {l} is {kinds[l]}")
            for key, rule in LAYER.items():
                part = key.split(".")[0]
                if part == "linear_attn" and kinds[l] != "linear_attention" or part == "self_attn" and kinds[l] != "full_attention":
                    continue
                need.append(f"blk.{l}.{rule[0]}")
        if int(tc.get("mtp_num_hidden_layers", 0)) == 1:
            need += [f"mtp.{r[0]}" for k, r in LAYER.items() if not k.startswith("linear_attn")]
        missing = [n for n in need if n not in self.tensors]
        if missing:
            raise S.ConvertError(f"the checkpoint lacks {len(missing)} tensors the engine reads: " + ", ".join(missing[:8]))

    # ---- float tensors
    def values(self, name) -> np.ndarray:
        t = self.tensors[name]
        x = self.nvfp4_values(name) if t.packed else self.ck.values(t.src)
        if t.how == "gconv":
            x = x[:, 0, :]
        if t.how == "norm":
            x = x + np.float32(1)
        if t.how == "negexp":
            x = (-np.exp(x.astype(np.float64))).astype(np.float32)
        idx = C._index(t.how, x.shape[0], S.GDN_QK_ROWS)
        if idx is not None:
            x = x[idx]
        return np.ascontiguousarray(x, dtype=np.float32)

    def float_bytes(self, name) -> bytes:
        t = self.tensors[name]
        if t.type_name == "BF16" and t.packed:   # NVFP4 values as BF16, rounded once (to nearest, ties to even)
            x64 = self.nvfp4_values(name, reorder=True)
            if not np.isfinite(x64).all():
                raise S.ConvertError(f"{name} has a non-finite value")
            out = bf16_round(x64)
            if not np.isfinite(S._bf16_to_f32(out)).all():
                raise S.ConvertError(f"{name} has a non-finite value")
            return out.tobytes()
        x = self.values(name)
        if not np.isfinite(x).all():
            raise S.ConvertError(f"{name} has a non-finite value")
        if t.type_name == "F32":
            return x.astype("<f4").tobytes()
        u = x.view(np.uint32)
        if np.count_nonzero(u & 0xFFFF):
            raise S.ConvertError(f"{name}: a value that is not BF16")
        return (u >> 16).astype("<u2").tobytes()

    def scale_value(self, name) -> np.ndarray:
        if self.tensors[name].form == "FP4Q":
            return np.array([fp4q_global(self.amax[name])], "<f4")
        if self.tensors[name].form == "F8Q":
            return self.row_scale[name].astype("<f4")
        src, reciprocal = self.scales[name]
        v = self.ck.values(src).astype(np.float64).reshape(1)
        if not (np.isfinite(v).all() and v[0] > 0):
            raise S.ConvertError(f"{src} is not finite and positive")
        return (1 / v if reciprocal else v).astype("<f4")

    def nvfp4_values(self, name, reorder=False) -> np.ndarray:
        """An NVFP4 tensor's values (code x group scale x tensor scale, exact in float64), [rows, K]; with `reorder`
        its rows in the GGUF order (the a / b projections' value heads)."""
        t = self.tensors[name]
        rows, K = t.ne[1], t.ne[0]
        codes = self.ck.raw(t.src).reshape(rows, K // 2)
        base = t.src[:-len(".weight_packed")] if t.src.endswith("_packed") else t.src[:-len(".weight")]
        sc = self.ck.raw(base + ".weight_scale").reshape(rows, K // 16)
        if (sc & 0x80).any() or (sc == 0x7F).any():
            raise S.ConvertError(f"{t.src}: a UE4M3 scale with the sign bit set or NaN")
        v = np.empty((rows, K), np.float64)
        v[:, 0::2], v[:, 1::2] = E2M1[codes & 15], E2M1[codes >> 4]
        src, reciprocal = self.scales[name]
        g = np.float64(self.ck.values(src).reshape(1)[0])
        x = v * np.repeat(E4M3[sc], 16, axis=1)
        x = x / g if reciprocal else x * g
        if reorder:
            idx = C._index(t.how, rows, S.GDN_QK_ROWS)
            if idx is not None:
                x = x[idx]
        return x

    # ---- quantized tensors, streamed
    def jobs(self, name):
        t = self.tensors[name]
        p, off, dt, shape, _ = self.ck.t[t.src]
        ne0, ne1 = t.ne
        kind = {"vrows": "rows", "zrows": "rows", "vcols": "cols"}.get(t.how)
        first = S.GDN_QK_ROWS if t.how == "vrows" else 0
        step = max(128, ROWS_PER_JOB // ne0 // 128 * 128)
        if t.form == "FP4":   # the code plane, then the scale plane in 128-row tiles
            base = t.src[:-len(".weight_packed")] if t.src.endswith("_packed") else t.src[:-len(".weight")]
            sp, soff = self.ck.t[base + ".weight_scale"][:2]
            for r in range(0, ne1, step):
                yield ("fp4c", str(p), off, ne0, ne1, r, min(step, ne1 - r), kind, first)
            for r in range(0, ne1, step):
                yield ("fp4s", str(sp), soff, ne0, ne1, r, min(step, ne1 - r), kind, first)
            return
        if t.form == "FP4Q":   # (codes, scales) per row block: chunks() writes the codes, then the scale plane
            gs = float(fp4q_global(self.amax[name]))
            for r in range(0, ne1, step):
                yield ("fp4q", str(p), off, ne0, r, min(step, ne1 - r), gs)
            return
        if t.form == "F8Q":   # BF16 rows rounded to E4M3 against their row scales
            step = max(1, ROWS_PER_JOB // ne0)
            for r in range(0, ne1, step):
                yield ("f8q", str(p), off, ne0, r, self.row_scale[name][r:r + step])
            return
        scale = float(self.ck.values(t.src[:-len(".weight")] + ".weight_scale")) if dt == "F8_E4M3" else None
        step = max(1, ROWS_PER_JOB // ne0)
        for r in range(0, ne1, step):
            yield ("f8" if t.form == "F8" else "q8", str(p), off, dt, scale, ne0, ne1, r, min(step, ne1 - r), kind, first)

    def chunks(self, name, pool, depth: int = 64):
        scales = []
        for b in S.bounded_imap(pool, _job, self.jobs(name), depth):
            if isinstance(b, str):
                raise S.ConvertError(f"{name}: {b}")
            if isinstance(b, tuple):   # FP4Q: the codes now, the scales once every row is done
                scales.append(b[1])
                b = b[0]
            yield b
        if scales:
            yield nvfp4_scale_tiles(np.concatenate(scales)).tobytes()


def _bf16_amax(path, off, shape) -> float:
    """The largest |value| of a BF16 matrix."""
    rows, cols = shape
    mm = np.memmap(path, dtype="<u2", mode="r", offset=off, shape=(rows, cols))
    step = max(1, ROWS_PER_JOB // cols)
    return max(float(np.abs(S._bf16_to_f32(np.ascontiguousarray(mm[r:r + step]))).max()) for r in range(0, rows, step))


def _bf16_row_amax(path, off, shape) -> np.ndarray:
    """Each row's largest |value| of a BF16 matrix (float32)."""
    rows, cols = shape
    mm = np.memmap(path, dtype="<u2", mode="r", offset=off, shape=(rows, cols))
    step = max(1, ROWS_PER_JOB // cols)
    return np.concatenate([np.abs(S._bf16_to_f32(np.ascontiguousarray(mm[r:r + step]))).max(-1) for r in range(0, rows, step)])


def f8q_row_scales(amax: np.ndarray) -> np.ndarray:
    """Rows' FP32 scales for E4M3 codes: amax / 448 (a row's largest |value| maps to the largest code), 1 for a zero row."""
    s = amax.astype(np.float32) / np.float32(448)
    return np.where(s > 0, s, np.float32(1)).astype(np.float32)


def e4m3_round(x: np.ndarray) -> np.ndarray:
    """float64 values -> E4M3 codes, to nearest with ties to the even code, saturating at 448 (no -0)."""
    c = ue4m3_round(np.abs(x))
    return c | np.where((x < 0) & (c > 0), 0x80, 0).astype(np.uint8)


def fp4q_global(amax: float) -> np.float32:
    """The tensor scale of a matrix rounded to NVFP4: its largest |value| maps to E2M1 6 at the largest UE4M3 scale."""
    return np.float32(amax) / np.float32(6.0 * 448.0) if amax > 0 else np.float32(1)


def bf16_round(x: np.ndarray) -> np.ndarray:
    """float64 values -> BF16 bits, to nearest with ties to even, rounded once (no subnormals: rows of weights)."""
    m, e = np.frexp(x)
    y = np.ldexp(np.round(m * 256.0), e - 8).astype(np.float32)   # 8 significant bits: exact in float32
    return (y.view(np.uint32) >> 16).astype("<u2")


def ue4m3_round(x: np.ndarray) -> np.ndarray:
    """Non-negative float32 values -> UE4M3 codes, to nearest with ties to the even code, saturating at 448."""
    x = np.minimum(x, np.float32(448))
    i = np.clip(np.searchsorted(UE4M3, x), 1, 126)
    lo, hi = UE4M3[i - 1], UE4M3[i]
    up = (x - lo > hi - x) | ((x - lo == hi - x) & (i % 2 == 0))
    return np.where(up, i, i - 1).astype(np.uint8)


def e2m1_round(x: np.ndarray) -> np.ndarray:
    """float32 values -> E2M1 codes, to nearest with ties to the even code, saturating (as nvfp4_planes.cu rounds)."""
    a = np.abs(x)
    c = ((a > 0.25).astype(np.uint8) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5) + (a > 5.0)).astype(np.uint8)
    return c | np.where((x < 0) & (c > 0), 8, 0).astype(np.uint8)


FP4Q_STEPS = range(-8, 17)    # UE4M3 codes tried around amax / 6 per group: half to four times that scale


def fp4q_rows(x: np.ndarray, gs: np.float32):
    """Rows [rows, K] (float32) -> (E2M1 codes [rows, K / 2], UE4M3 scales [rows, K / 16]): each 16-value group takes the
    scale of least squared error among the UE4M3 codes near amax / (6 gs) (the first such when tied, amax / 6 first)."""
    rows, K = x.shape
    g = x.reshape(rows, K // 16, 16)
    c0 = ue4m3_round(np.abs(g).max(-1) / (np.float32(6) * gs)).astype(np.int32)
    best_err = best_code = None
    for d in sorted(FP4Q_STEPS, key=abs):
        c = np.clip(c0 + d, 0, 126)
        sd = UE4M3[c] * gs
        inv = np.where(sd > 0, np.float32(1) / np.where(sd > 0, sd, 1), 0).astype(np.float32)
        q = e2m1_round(g * inv[..., None])
        err = ((E2M1[q].astype(np.float32) * sd[..., None] - g) ** 2).sum(-1)
        if best_err is None:
            best_err, best_code = err, c
        else:
            better = err < best_err
            best_err, best_code = np.where(better, err, best_err), np.where(better, c, best_code)
    sd = UE4M3[best_code] * gs
    inv = np.where(sd > 0, np.float32(1) / np.where(sd > 0, sd, 1), 0).astype(np.float32)
    q = e2m1_round(g * inv[..., None]).reshape(rows, K)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8), best_code.astype(np.uint8)


def _job(args):
    if args[0] == "slice":   # BF16 rows [row0, row0 + rows), columns [c0, c0 + ne0) of a [.][width] matrix, as Q8_0
        _, path, off, width, c0, ne0, row0, rows = args
        mm = np.memmap(path, dtype="<u2", mode="r", offset=off + row0 * width * 2, shape=(rows, width))
        x = S._bf16_to_f32(np.ascontiguousarray(mm[:, c0:c0 + ne0])).reshape(-1)
        return S.q8_0(x) if np.isfinite(x).all() else "a non-finite value"
    if args[0] in ("fp4c", "fp4s"):   # code rows as stored (value 2i in the low nibble of byte i); scale rows into their
        # tiles (rows in 128s); both with rows / columns in the GGUF head order (whole 128-value heads move)
        op, path, off, ne0, ne1, row0, rows, kind, first = args
        per = 2 if op == "fp4c" else 16          # values per byte
        mm = np.memmap(path, dtype=np.uint8, mode="r", offset=off, shape=(ne1, ne0 // per))
        if kind == "rows":
            u = mm[S.gdn_index(ne1, first)[row0:row0 + rows]]
        elif kind == "cols":
            u = mm[row0:row0 + rows][:, S.gdn_index(ne0, 0)[::per] // per]
        else:
            u = mm[row0:row0 + rows]
        u = np.ascontiguousarray(u)
        if op == "fp4c":
            return u.tobytes()
        if (u & 0x80).any() or (u == 0x7F).any():
            return "a UE4M3 scale with the sign bit set or NaN"
        return nvfp4_scale_tiles(u).tobytes()
    if args[0] == "f8q":   # BF16 rows rounded to E4M3 codes against their row scales (the division exact in float64)
        _, path, off, ne0, row0, sc = args
        rows = len(sc)
        mm = np.memmap(path, dtype="<u2", mode="r", offset=off + row0 * ne0 * 2, shape=(rows, ne0))
        x = S._bf16_to_f32(np.ascontiguousarray(mm)).reshape(rows, ne0)
        if not np.isfinite(x).all():
            return "a non-finite value"
        return e4m3_round(x.astype(np.float64) / sc.astype(np.float64)[:, None]).tobytes()
    if args[0] == "fp4q":   # BF16 rows rounded to NVFP4: (codes, scales)
        _, path, off, ne0, row0, rows, gs = args
        mm = np.memmap(path, dtype="<u2", mode="r", offset=off + row0 * ne0 * 2, shape=(rows, ne0))
        x = S._bf16_to_f32(np.ascontiguousarray(mm)).reshape(rows, ne0)
        if not np.isfinite(x).all():
            return "a non-finite value"
        codes, sc = fp4q_rows(x, np.float32(gs))
        return codes.tobytes(), sc
    op, path, off, dt, scale, ne0, ne1, row0, rows, kind, first = args
    if dt == "F8_E4M3":
        mm = np.memmap(path, dtype=np.uint8, mode="r", offset=off, shape=(ne1, ne0))
    else:
        mm = np.memmap(path, dtype="<u2", mode="r", offset=off, shape=(ne1, ne0))
    if kind == "rows":
        u = mm[S.gdn_index(ne1, first)[row0:row0 + rows]]
    elif kind == "cols":
        u = mm[row0:row0 + rows][:, S.gdn_index(ne0, 0)]
    else:
        u = mm[row0:row0 + rows]
    u = np.ascontiguousarray(u)
    if op == "f8":   # the codes as stored, in the GGUF head order
        if ((u & 0x7F) == 0x7F).any():
            return "an E4M3 NaN code"
        return u.tobytes()
    x = E4M3[u.reshape(-1)] * np.float32(scale) if dt == "F8_E4M3" else S._bf16_to_f32(u).reshape(-1)
    if not np.isfinite(x).all():
        return "a non-finite value"
    return S.q8_0(x)


def nvfp4_scale_tiles(sc: np.ndarray) -> np.ndarray:
    """UE4M3 scales [rows, K/16] (rows in 128s) -> the NVFP4_PLANES scale tiles: per 128 rows and 4 groups a 512-byte tile
    (tiles row-block major), byte (r % 32) * 16 + ((r % 128) // 32) * 4 + g % 4 (include/basalt/kernels/nvfp4_planes.hpp)."""
    rows, groups = sc.shape
    return np.ascontiguousarray(sc.reshape(rows // 128, 4, 32, groups // 4, 4).transpose(0, 3, 2, 1, 4)).reshape(-1)


# ------------------------------------------------------------------ metadata
def model_kv(ck: C.Checkpoint, name: str | None) -> dict:
    c, tc = ck.config, ck.text
    A = ARCH + "."
    d = ck.dir
    gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8")) if (d / "generation_config.json").is_file() else {}
    card = C.front_matter(d / "README.md")
    kinds = tc["layer_types"]
    rope = tc.get("rope_parameters", {})
    head = int(tc["head_dim"])
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
    bases = card.get("base_model") or []
    bases = [bases] if isinstance(bases, str) else bases
    if bases:
        kv["general.base_model.count"] = len(bases)
        for i, b in enumerate(bases):
            org, _, repo = b.rpartition("/")
            kv[f"general.base_model.{i}.name"] = repo.replace("-", " ")
            if org:
                kv[f"general.base_model.{i}.organization"] = org
            kv[f"general.base_model.{i}.repo_url"] = f"https://huggingface.co/{b}"
    if tc.get("attn_output_gate") is not True:
        raise S.ConvertError("config.json: attention without the output gate")
    kv.update({
        A + "block_count": int(tc["num_hidden_layers"]),
        A + "context_length": int(tc["max_position_embeddings"]),
        A + "embedding_length": int(tc["hidden_size"]),
        A + "feed_forward_length": int(tc["intermediate_size"]),
        A + "attention.head_count": int(tc["num_attention_heads"]),
        A + "attention.head_count_kv": int(tc["num_key_value_heads"]),
        A + "attention.key_length": head,
        A + "attention.value_length": head,
        A + "attention.layer_norm_rms_epsilon": C.f32(tc["rms_norm_eps"]),
        A + "rope.dimension_count": int(head * float(rope.get("partial_rotary_factor", tc.get("partial_rotary_factor")))),
        A + "rope.dimension_sections": (list(rope.get("mrope_section", [])) + [0, 0, 0, 0])[:4],
        A + "rope.freq_base": C.f32(rope.get("rope_theta", tc.get("rope_theta"))),
        A + "ssm.conv_kernel": int(tc["linear_conv_kernel_dim"]),
        A + "ssm.state_size": int(tc["linear_key_head_dim"]),
        A + "ssm.group_count": int(tc["linear_num_key_heads"]),
        A + "ssm.time_step_rank": int(tc["linear_num_value_heads"]),
        A + "ssm.inner_size": int(tc["linear_num_value_heads"]) * int(tc["linear_value_head_dim"]),
        A + "attention.recurrent_layers": [k == "linear_attention" for k in kinds],
        A + "full_attention_interval": int(tc["full_attention_interval"]),
        A + "nextn_predict_layers": int(tc.get("mtp_num_hidden_layers", 0)),
        A + "image_token_id": int(c["image_token_id"]),
        A + "video_token_id": int(c["video_token_id"]),
        A + "vision_start_token_id": int(c["vision_start_token_id"]),
        "general.file_type": 40,
        "general.quantization_version": 2,
    })
    kv.update(C.tokenizer_kv(ck))
    return kv


# ------------------------------------------------------------------ vision
VISION_BLOCK = {
    "attn.qkv.weight": ("attn_qkv.weight", "BF16"), "attn.qkv.bias": ("attn_qkv.bias", "F32"),
    "attn.proj.weight": ("attn_out.weight", "BF16"), "attn.proj.bias": ("attn_out.bias", "F32"),
    "mlp.linear_fc1.weight": ("ffn_up.weight", "BF16"), "mlp.linear_fc1.bias": ("ffn_up.bias", "F32"),
    "mlp.linear_fc2.weight": ("ffn_down.weight", "BF16"), "mlp.linear_fc2.bias": ("ffn_down.bias", "F32"),
    "norm1.weight": ("ln1.weight", "F32"), "norm1.bias": ("ln1.bias", "F32"),
    "norm2.weight": ("ln2.weight", "F32"), "norm2.bias": ("ln2.bias", "F32"),
}
VISION_GLOBAL = {
    "model.visual.merger.norm.weight": ("v.post_ln.weight", "F32"), "model.visual.merger.norm.bias": ("v.post_ln.bias", "F32"),
    "model.visual.merger.linear_fc1.weight": ("mm.0.weight", "BF16"), "model.visual.merger.linear_fc1.bias": ("mm.0.bias", "F32"),
    "model.visual.merger.linear_fc2.weight": ("mm.2.weight", "BF16"), "model.visual.merger.linear_fc2.bias": ("mm.2.bias", "F32"),
    "model.visual.patch_embed.proj.bias": ("v.patch_embd.bias", "F32"),
    "model.visual.pos_embed.weight": ("v.position_embd.weight", "F32"),
}


def mmproj(ck: C.Checkpoint, out: pathlib.Path, name: str) -> None:
    """The vision tower as the clip GGUF the engine's encoder reads (llama.cpp's qwen3vl_merger layout)."""
    vc = ck.config["vision_config"]
    if vc.get("model_type") not in ("qwen3_5_vision", "qwen3_5") or vc.get("deepstack_visual_indexes"):
        raise S.ConvertError("config.json: the vision tower is not the Qwen3-VL one without deepstack layers")
    pp = json.loads((ck.dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    w = GGUFWriter()
    for k, v, t in (("general.architecture", "clip", "string"), ("general.type", "mmproj", "string"),
                    ("general.name", name + " (mmproj)", "string"), ("clip.has_vision_encoder", True, "bool"),
                    ("clip.projector_type", "qwen3vl_merger", "string"), ("clip.use_gelu", True, "bool"),
                    ("clip.vision.projection_dim", int(vc["out_hidden_size"]), "u32"),
                    ("clip.vision.image_size", 768, "u32"), ("clip.vision.patch_size", int(vc["patch_size"]), "u32"),
                    ("clip.vision.embedding_length", int(vc["hidden_size"]), "u32"),
                    ("clip.vision.feed_forward_length", int(vc["intermediate_size"]), "u32"),
                    ("clip.vision.block_count", int(vc["depth"]), "u32"),
                    ("clip.vision.attention.head_count", int(vc["num_heads"]), "u32"),
                    ("clip.vision.attention.layer_norm_epsilon", 1e-6, "f32"),
                    ("clip.vision.spatial_merge_size", int(vc["spatial_merge_size"]), "u32")):
        w.add(k, v, t)
    w.add("clip.vision.image_mean", [float(x) for x in pp["image_mean"]], "array:f32")
    w.add("clip.vision.image_std", [float(x) for x in pp["image_std"]], "array:f32")
    if vc.get("hidden_act") != "gelu_pytorch_tanh" or int(vc["temporal_patch_size"]) != 2:
        raise S.ConvertError("config.json: the vision tower's activation or temporal patch differs from Qwen3-VL's")

    def put(gname, src, ty):
        x = ck.values(src)
        if ty == "BF16":
            u = x.astype(np.float32).view(np.uint32) >> 16
            w.add_raw(gname, list(reversed(x.shape)), "BF16", u.astype("<u2").tobytes())
        else:
            a = np.ascontiguousarray(x, dtype="<f4")
            w.add_raw(gname, list(reversed(a.shape)), "F32", a.tobytes())

    for l in range(int(vc["depth"])):
        for k, (g, ty) in VISION_BLOCK.items():
            put(f"v.blk.{l}.{g}", f"model.visual.blocks.{l}.{k}", ty)
    for k, (g, ty) in VISION_GLOBAL.items():
        put(g, k, ty)
    pw = ck.values("model.visual.patch_embed.proj.weight")     # [out, 3, t=2, 16, 16]
    if pw.ndim != 5 or pw.shape[2] != 2:
        raise S.ConvertError(f"model.visual.patch_embed.proj.weight has shape {list(pw.shape)}")
    for t, g in ((0, "v.patch_embd.weight"), (1, "v.patch_embd.weight.1")):
        a = np.ascontiguousarray(pw[:, :, t], dtype="<f4")      # [out, 3, 16, 16] -> ne [16, 16, 3, out]
        w.add_raw(g, list(reversed(a.shape)), "F32", a.tobytes())
    w.write(out)


def text_vocab(kv: dict, out: pathlib.Path) -> None:
    """The text model's metadata alone (no tensors), as vision/text_vocab."""
    w = GGUFWriter()
    for k, v in kv.items():
        if isinstance(v, bool):
            w.add(k, v, "bool")
        elif isinstance(v, int):
            w.add(k, v, "u32" if v >= 0 else "i32")
        elif isinstance(v, float):
            w.add(k, v, "f32")
        elif isinstance(v, str):
            w.add(k, v, "string")
        elif isinstance(v, list) and v:
            w.add(k, v, {bool: "array:bool", int: "array:i32", float: "array:f32", str: "array:string"}[type(v[0])])
    w.write(out)



# ------------------------------------------------------------------ the block drafter (DFlash 2)
DRAFTER_LAYER = {
    "input_layernorm.weight": ("attn_norm.weight", "F32"),
    "post_attention_layernorm.weight": ("post_attention_norm.weight", "F32"),
    "self_attn.q_proj.weight": ("attn_q.weight", "Q8"), "self_attn.k_proj.weight": ("attn_k.weight", "Q8"),
    "self_attn.v_proj.weight": ("attn_v.weight", "Q8"), "self_attn.o_proj.weight": ("attn_output.weight", "Q8"),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "F32"), "self_attn.k_norm.weight": ("attn_k_norm.weight", "F32"),
    "mlp.gate_proj.weight": ("ffn_gate.weight", "Q8"), "mlp.up_proj.weight": ("ffn_up.weight", "Q8"),
    "mlp.down_proj.weight": ("ffn_down.weight", "Q8"),
    "attention_conv.base_kernel": ("attn_conv.base", "F32"), "attention_conv.kernel_projection.weight": ("attn_conv.weight", "Q8"),
    "mlp_conv.base_kernel": ("ffn_conv.base", "F32"), "mlp_conv.kernel_projection.weight": ("ffn_conv.weight", "Q8"),
}
DRAFTER_GLOBAL = {
    "hidden_norm.weight": ("hidden_norm.weight", "F32"), "norm.weight": ("norm.weight", "F32"),
    "candidate_selector.hidden_projection.weight": ("selector.hidden.weight", "BF16"),
    "candidate_selector.predecessor_codebook": ("selector.pred", "BF16"),
    "candidate_selector.successor_codebook": ("selector.succ", "BF16"),
}


class Drafter:
    """A DFlash 2 block drafter (z-lab's DFlash2DraftModel safetensors) as `dflash.*` tensors: projections Q8_0 from
    BF16, norms (plain RMS weights, no offset) and the convolutions' base kernels F32, the selector BF16."""

    def __init__(self, d, ck: C.Checkpoint):
        self.dir = pathlib.Path(d)
        self.config = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        if self.config.get("architectures") != ["DFlash2DraftModel"]:
            raise S.ConvertError(f"{self.dir}: not a DFlash2DraftModel")
        dc = self.config["dflash_config"]
        if int(self.config["hidden_size"]) != int(ck.text["hidden_size"]) or int(self.config["vocab_size"]) != int(ck.text["vocab_size"]):
            raise S.ConvertError(f"{self.dir}: the drafter's width or vocabulary differs from the model's")
        if any(t >= int(ck.text["num_hidden_layers"]) for t in dc["target_layer_ids"]):
            raise S.ConvertError(f"{self.dir}: a target layer past the model's layers")
        if self.config.get("is_causal") is not False or set(self.config.get("layer_types", [])) != {"sliding_attention"}:
            raise S.ConvertError(f"{self.dir}: only the non-causal sliding-window drafter is converted")
        self.files, self.t = sorted(self.dir.glob("*.safetensors")), {}
        for f in self.files:
            head, base = S.read_safetensors_header(f)
            for k, v in sorted(((k, v) for k, v in head.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0]):
                a_, b_ = v["data_offsets"]
                self.t[k] = (f, base + a_, v["dtype"], list(v["shape"]), b_ - a_)
        self.tensors, self.slices = {}, {}
        H = int(self.config["hidden_size"])
        for k, (f, off, dt, shape, n) in self.t.items():
            if k == "fc.weight":   # one [H][H] block per target layer, so the features project as each layer ends
                if shape != [H, H * len(dc["target_layer_ids"])]:
                    raise S.ConvertError(f"{self.dir}: fc.weight is {shape}")
                for j in range(len(dc["target_layer_ids"])):
                    name = f"dflash.fc.{j}.weight"
                    self.tensors[name] = Tensor(name, "Q8", k, "id", [H, H])
                    self.slices[name] = (j * H, shape[1])
                continue
            m = re.fullmatch(r"layers\.(\d+)\.(.+)", k)
            rule = DRAFTER_LAYER.get(m.group(2)) if m else DRAFTER_GLOBAL.get(k)
            if rule is None:
                raise S.ConvertError(f"{self.dir}: drafter tensor {k} has no counterpart")
            if dt != "BF16":
                raise S.ConvertError(f"{k} is {dt}, expected BF16")
            name = "dflash." + (f"blk.{m.group(1)}.{rule[0]}" if m else rule[0])
            self.tensors[name] = Tensor(name, rule[1], k, "id", list(reversed(shape)))

    def kv(self) -> dict:
        c, dc = self.config, self.config["dflash_config"]
        A = ARCH + ".dflash."
        return {A + "block_count": int(c["num_hidden_layers"]), A + "block_size": int(dc["block_size"]),
                A + "mask_token_id": int(dc["mask_token_id"]), A + "target_layers": [int(x) for x in dc["target_layer_ids"]],
                A + "conv_kernel": int(dc["conv_kernel_size"]), A + "conv_group": int(dc["conv_group_size"]),
                A + "selector_rank": int(dc["selector_rank"]), A + "selector_top_k": int(dc["selector_top_k"]),
                A + "sliding_window": int(c["sliding_window"]), A + "attention.head_count": int(c["num_attention_heads"]),
                A + "attention.head_count_kv": int(c["num_key_value_heads"]), A + "attention.key_length": int(c["head_dim"]),
                A + "feed_forward_length": int(c["intermediate_size"]),
                A + "rope.freq_base": C.f32(c["rope_parameters"]["rope_theta"]), A + "attention.layer_norm_rms_epsilon": C.f32(c["rms_norm_eps"])}

    def values(self, name) -> np.ndarray:
        f, off, dt, shape, n = self.t[self.tensors[name].src]
        u = np.fromfile(f, dtype="<u2", count=n // 2, offset=off)
        return S._bf16_to_f32(u).reshape(shape)

    def float_bytes(self, name) -> bytes:
        t = self.tensors[name]
        x = self.values(name)
        if t.type_name == "F32":
            return np.ascontiguousarray(x, dtype="<f4").tobytes()
        f, off, dt, shape, n = self.t[t.src]
        return np.fromfile(f, dtype=np.uint8, count=n, offset=off).tobytes()

    def chunks(self, name, pool, depth: int = 64):
        t = self.tensors[name]
        f, off, dt, shape, n = self.t[t.src]
        ne0, ne1 = t.ne
        step = max(1, ROWS_PER_JOB // ne0)
        if name in self.slices:
            c0, width = self.slices[name]
            jobs = (("slice", str(f), off, width, c0, ne0, r, min(step, ne1 - r)) for r in range(0, ne1, step))
        else:
            jobs = (("q8", str(f), off, "BF16", None, ne0, ne1, r, min(step, ne1 - r), None, 0) for r in range(0, ne1, step))
        for b in S.bounded_imap(pool, _job, jobs, depth):
            if isinstance(b, str):
                raise S.ConvertError(f"{name}: {b}")
            yield b


# ------------------------------------------------------------------ the pack
LICENSE = {"apache-2.0": "Apache License 2.0"}


def build(a, Job, fingerprint, die) -> None:
    def source_entry(p):
        return {"path": p.name, "size": p.stat().st_size, "fingerprint_sha256_first_last_64mib_and_size": fingerprint(p)}

    import os
    import tokenizer
    out = pathlib.Path(a.out)
    partial = pathlib.Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    try:
        ck = C.Checkpoint(a.checkpoint)
        plan = Plan(ck, getattr(a, "head", None))
        bm = C.front_matter(ck.dir / "README.md").get("base_model")
        base = (bm if isinstance(bm, str) else bm[0]) if bm else "Qwen3.5"
        name = a.name or base.split("/")[-1] + " NVFP4"
        kv = model_kv(ck, name)
        drafter = Drafter(a.drafter, ck) if getattr(a, "drafter", None) else None
        if drafter:
            kv.update(drafter.kv())
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
            t = time.time()
            parts, names, scale_n = [], [], {}
            for n, info in plan.tensors.items():
                if info.type_name in ("Q8_0", "NVFP4_PLANES", "F8_E4M3"):
                    parts.append(Stream(info.expected_bytes(), plan.chunks(n, pool)))
                else:
                    parts.append(plan.float_bytes(n))
                names.append(n)
                if info.type_name in ("NVFP4_PLANES", "F8_E4M3"):
                    sv = plan.scale_value(n)
                    parts.append(sv.tobytes())
                    names.append(n[:-len(".weight")] + ".scale")
                    scale_n[names[-1]] = sv.size
            for n, info in (drafter.tensors.items() if drafter else []):
                parts.append(Stream(info.expected_bytes(), drafter.chunks(n, pool)) if info.type_name == "Q8_0" else drafter.float_bytes(n))
                names.append(n)
            offs = w.add_parts("gguf", parts, TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            mp = pathlib.Path(tmp) / "mmproj.gguf"
            mmproj(ck, mp, name)
            job.file("vision/mmproj", mp)
            tv = pathlib.Path(tmp) / "text_vocab.gguf"
            text_vocab(kv, tv)
            job.file("vision/text_vocab", tv)
            tens = {}
            every = dict(plan.tensors, **(drafter.tensors if drafter else {}))
            for n, off in zip(names, offs):
                if n in every:
                    t_ = every[n]
                    tens[n] = {"type": t_.type_name, "type_id": t_.type_id, "shape": [int(d) for d in t_.ne],
                               "bytes": t_.expected_bytes(), "offset": off}
                else:
                    tens[n] = {"type": "F32", "type_id": 0, "shape": [scale_n[n]], "bytes": 4 * scale_n[n], "offset": off}
            used = sorted({t_.how for t_ in plan.tensors.values()})
            lic = kv.get("general.license", "")
            meta = {
                "format": "basalt", "version": 2, "name": name, "arch": ARCH,
                "converter": {"tool": "tools/convert/convert.py"},
                "sources": [{"path": p.name, "size": p.stat().st_size,
                             "fingerprint_sha256_first_last_64mib_and_size": fingerprint(p)} for p in ck.shards],
                "gguf_kv": kv,
                "tensors": tens,
                "license": {"name": LICENSE.get(lic, lic)},
                "vision": {"projector": "qwen3vl_merger", "source": "the checkpoint's BF16 vision tower"},
                "mtp": {"layers": int(ck.text.get("mtp_num_hidden_layers", 0)), "source": "the checkpoint's BF16 mtp.*, Q8_0"},
                **({"dflash": {"source": pathlib.Path(a.drafter).name, "files": [source_entry(p) for p in drafter.files],
                               "form": "projections Q8_0 from BF16, norms and conv base kernels F32, selector BF16"}}
                   if drafter else {}),
                "checkpoint": {
                    "dir": pathlib.Path(a.checkpoint).name,
                    "config_sha256": hashlib.sha256((ck.dir / "config.json").read_bytes()).hexdigest(),
                    "nvfp4": "NVFP4_PLANES: weight / weight_packed (E2M1 pairs) as stored, then weight_scale (UE4M3 per "
                             "16) in 128-row x 4-group tiles, unchanged; weight_scale_2, or 1 / weight_global_scale "
                             "(FP32, rounded once), as <name>.scale; input_scale / input_global_scale unused",
                    **({"nvfp4_bf16": "NVFP4 a / b projections: their values rounded once to BF16"}
                       if any(t_.packed and t_.form == "BF16" for t_ in plan.tensors.values()) else {}),
                    **({"bf16_to_nvfp4": "BF16 " + ", ".join(n for n, t_ in plan.tensors.items() if t_.form == "FP4Q") +
                                         ": NVFP4_PLANES, tensor scale amax / (6 x 448), per 16 values the UE4M3 scale "
                                         "of least squared error among half to four times amax / 6, E2M1 to nearest"}
                       if plan.amax else {}),
                    **({"bf16_to_fp8": "BF16 " + ", ".join(plan.row_scale) + ": F8_E4M3 rows, each row's FP32 scale "
                                       "its amax / 448 as <name>.scale (F32 [rows]), E4M3 codes to nearest"}
                       if plan.row_scale else {}),
                    "fp8": "weight (E4M3 codes) kept as stored, rows / columns in the GGUF head order; weight_scale as "
                           "<name>.scale; input_scale unused (activations get a scale per row at run time)",
                    "bf16_projections": "Q8_0 (ggml reference rounding)",
                    "transforms": {k: METHODS[k] for k in used}},
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
