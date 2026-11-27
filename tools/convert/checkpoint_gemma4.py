"""A gemma4 pack (Google's Gemma 4: interleaved sliding-window and full-attention layers, a dense MLP beside optional
routed experts) straight from the official BF16 checkpoint, for tools/convert/convert.py.

  experts             the routed experts per layer, [gate | up | down] (the checkpoint's gate_up_proj rows: gate the
                        first half, up the second): NVFP4 (type 40) blocks rounded once from BF16 (per matrix the
                        tensor scale amax / (6 x 448), per 16 values the UE4M3 scale of least squared error among half
                        to four times amax / 6), then the FP32 tail {s_gate, s_up, s_down * per_expert_scale, 0};
                        the layers of `--experts-mxfp8` as MXFP8 (type 46) blocks (per 32 values the UE8M0 scale that
                        brings the largest magnitude within 448; the down rows padded with zero columns to whole 128-value
                        blocks; the per-expert scale folded into the down weights, no tail)
  experts.layout      one `layer gu_type d_type offset blob_bytes` line per layer that has experts
  gguf                every other text tensor in llama.cpp-style names: the attention and MLP projections F8_E4M3
                        (type 44) with one FP32 scale per matrix (amax / 448, `<name>.scale`), the MLP's inner width
                        padded with zero rows and columns to a multiple of 128; with --mlp-nvfp4 the MLP's three
                        projections NVFP4_PLANES (type 45: the experts' rounding, the scales in 128-row tiles) with the
                        tensor scale as `<name>.scale`; the token embedding BF16 as stored; the
                        output head (the tied embedding) F8_E4M3 rows with an FP32 scale per row (`output.scale`); the
                        router F32 with its input scale and 1 / sqrt(hidden) folded into the columns; norms and layer
                        scalars F32 (exact). The assistant drafter's tensors (`--drafter DIR`) as `assistant.*`.
  vision/mmproj       the vision tower as llama.cpp's clip GGUF (projector gemma4v): the patch embedding as the F32
                        convolution [out][channel][row][column], the column and row position tables F32, the products
                        BF16 as stored, the norms and the standardisation F32, the embedding projection as
                        mm.input_projection.weight
  tokenizer/*, meta   the vocabulary respelled for the engine's BPE (the SentencePiece space mark as a space, the byte
                        fallback tokens as type 6), the chat template and the geometry from config.json

Needs numpy.
"""
from __future__ import annotations

import hashlib
import json
import math
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
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import TENSOR_ALIGN, ModelWriter, Stream  # noqa: E402

ARCH = "gemma4"
TYPE_ID = {"F32": 0, "BF16": 30, "NVFP4": 40, "F8_E4M3": 44, "NVFP4_PLANES": 45}
NVFP4 = 40
MXFP8 = 46
BYTE = 6                      # llama.cpp's token type of a byte-fallback token
ROWS_PER_JOB = 4096
FP4_ROWS_PER_JOB = 512        # an NVFP4 job's rows (its scale search holds several float32 copies of them)
FP4_K = 256                   # the prompt's FP4 products take K in multiples of 256
EXPERTS_PER_JOB = 8
PAD = 128                     # the dense products take K in multiples of 128
TEXT = "model.language_model."
ASSIST = "model."             # the assistant's own text model
# config.json model_type of the model / its text model, and of the assistant: Gemma 4 and the 12B's Gemma 4 Unified, whose
# text model is the same decoder (only its vision embedder differs, which the pack does not carry)
MODEL_TYPES = {"gemma4": "gemma4_text", "gemma4_unified": "gemma4_unified_text"}
ASSIST_TYPES = {"gemma4_assistant": "gemma4_text", "gemma4_unified_assistant": "gemma4_unified_text"}


def pad_to(n: int, m: int = PAD) -> int:
    return (n + m - 1) // m * m


class Tensor:
    def __init__(self, name, form, src, how, ne, extra=None):
        self.name, self.form, self.src, self.how, self.ne, self.extra = name, form, src, how, ne, extra or {}
        self.type_id = TYPE_ID[form]

    def expected_bytes(self) -> int:
        n = int(np.prod(self.ne))
        return {"F32": 4 * n, "BF16": 2 * n, "F8_E4M3": n, "NVFP4_PLANES": n // 64 * 36}[self.form]


# ------------------------------------------------------------------ the dense tensors
class Plan:
    """The pack's `gguf` tensors: the language model's, then the drafter's."""

    def __init__(self, ck: C.Checkpoint, drafter: C.Checkpoint | None, mlp_nvfp4: bool = False):
        tc = ck.text
        self.ck, self.dr = ck, drafter
        mt = ck.config.get("model_type")
        if mt not in MODEL_TYPES or tc.get("model_type") != MODEL_TYPES[mt]:
            raise S.ConvertError(f"config.json: model_type {mt} / {tc.get('model_type')}, expected one of "
                                 + ", ".join(f"{a} / {b}" for a, b in MODEL_TYPES.items()))
        for k, want in (("attention_bias", False), ("hidden_size_per_layer_input", 0), ("num_kv_shared_layers", 0),
                        ("use_double_wide_mlp", False), ("hidden_activation", "gelu_pytorch_tanh"),
                        ("tie_word_embeddings", True)):
            if tc.get(k, want) != want:
                raise S.ConvertError(f"config.json: {k} = {tc.get(k)!r}; the engine runs {want!r}")
        self.n_layers = int(tc["num_hidden_layers"])
        self.h = int(tc["hidden_size"])
        self.n_vocab = int(tc["vocab_size"])
        self.ff = int(tc["intermediate_size"])
        self.ff_pad = pad_to(self.ff)
        self.mlp_nvfp4 = mlp_nvfp4
        if mlp_nvfp4 and (self.ff % FP4_K or self.h % FP4_K):
            raise S.ConvertError(f"--mlp-nvfp4: hidden_size {self.h} and intermediate_size {self.ff} must be multiples of "
                                 f"{FP4_K}")
        self.kinds = list(tc["layer_types"])
        self.moe = bool(tc.get("enable_moe_block"))
        self.kev = bool(tc.get("attention_k_eq_v"))
        self.tensors: dict[str, Tensor] = {}
        P = TEXT
        self._f32("token_embd.weight", P + "embed_tokens.weight", [self.h, self.n_vocab], form="BF16")
        self.tensors["output.weight"] = Tensor("output.weight", "F8_E4M3", P + "embed_tokens.weight", "rows",
                                               [self.h, self.n_vocab])
        self.tensors["output.scale"] = Tensor("output.scale", "F32", P + "embed_tokens.weight", "row_scales",
                                              [self.n_vocab])
        self._f32("output_norm.weight", P + "norm.weight", [self.h])
        for l in range(self.n_layers):
            full = self.kinds[l] == "full_attention"
            hd = int(tc["global_head_dim"] if full else tc["head_dim"])
            nkv = int(tc["num_global_key_value_heads"] if full and self.kev else tc["num_key_value_heads"])
            nq = int(tc["num_attention_heads"])
            L = f"{P}layers.{l}."
            b = f"blk.{l}."
            self._f32(b + "attn_norm.weight", L + "input_layernorm.weight", [self.h])
            self._f8(b + "attn_q.weight", L + "self_attn.q_proj.weight", [self.h, nq * hd])
            self._f8(b + "attn_k.weight", L + "self_attn.k_proj.weight", [self.h, nkv * hd])
            if not (full and self.kev):
                self._f8(b + "attn_v.weight", L + "self_attn.v_proj.weight", [self.h, nkv * hd])
            elif L + "self_attn.v_proj.weight" in ck.t:
                raise S.ConvertError(f"layer {l}: attention_k_eq_v with a v_proj")
            self._f8(b + "attn_output.weight", L + "self_attn.o_proj.weight", [nq * hd, self.h])
            self._f32(b + "attn_q_norm.weight", L + "self_attn.q_norm.weight", [hd])
            self._f32(b + "attn_k_norm.weight", L + "self_attn.k_norm.weight", [hd])
            self._f32(b + "post_attention_norm.weight", L + "post_attention_layernorm.weight", [self.h])
            self._f32(b + "ffn_norm.weight", L + "pre_feedforward_layernorm.weight", [self.h])
            if mlp_nvfp4:
                self._fp4(b + "ffn_gate.weight", L + "mlp.gate_proj.weight", [self.h, self.ff])
                self._fp4(b + "ffn_up.weight", L + "mlp.up_proj.weight", [self.h, self.ff])
                self._fp4(b + "ffn_down.weight", L + "mlp.down_proj.weight", [self.ff, self.h])
            else:
                self._f8(b + "ffn_gate.weight", L + "mlp.gate_proj.weight", [self.h, self.ff_pad], pad_rows=self.ff)
                self._f8(b + "ffn_up.weight", L + "mlp.up_proj.weight", [self.h, self.ff_pad], pad_rows=self.ff)
                self._f8(b + "ffn_down.weight", L + "mlp.down_proj.weight", [self.ff_pad, self.h], pad_cols=self.ff)
            self._f32(b + "post_ffw_norm.weight", L + "post_feedforward_layernorm.weight", [self.h])
            self._f32(b + "layer_scalar", L + "layer_scalar", [1])
            if self.moe:
                self._f32(b + "post_ffw_norm_1.weight", L + "post_feedforward_layernorm_1.weight", [self.h])
                self._f32(b + "post_ffw_norm_2.weight", L + "post_feedforward_layernorm_2.weight", [self.h])
                self._f32(b + "pre_ffw_norm_2.weight", L + "pre_feedforward_layernorm_2.weight", [self.h])
                self.tensors[b + "ffn_gate_inp.weight"] = Tensor(b + "ffn_gate_inp.weight", "F32", L + "router.proj.weight",
                                                                 "router", [self.h, int(tc["num_experts"])],
                                                                 {"scale": L + "router.scale"})
        if drafter is not None:
            self._drafter(drafter)
        missing = [k for k in ck.t if k.startswith(P) and ".experts." not in k and ".router.per_expert_scale" not in k
                   and ".router.scale" not in k and not any(t.src == k for t in self.tensors.values())]
        if missing:
            raise S.ConvertError("checkpoint tensors with no pack counterpart: " + ", ".join(missing[:8]))

    def _drafter(self, dr: C.Checkpoint):
        c, tc = dr.config, dr.text
        mt = c.get("model_type")
        if mt not in ASSIST_TYPES or tc.get("model_type") != ASSIST_TYPES[mt]:
            raise S.ConvertError(f"{dr.dir}/config.json: model_type {mt} / {tc.get('model_type')}, expected one of "
                                 + ", ".join(f"{a} / {b}" for a, b in ASSIST_TYPES.items()))
        if int(c["backbone_hidden_size"]) != self.h:
            raise S.ConvertError("the drafter's backbone width is not the model's")
        if c.get("use_ordered_embeddings"):
            raise S.ConvertError("the drafter's ordered (centroid) head: not converted")
        if int(tc["num_kv_shared_layers"]) != int(tc["num_hidden_layers"]) or tc.get("enable_moe_block"):
            raise S.ConvertError("the drafter does not read every layer's keys from the model, or has experts")
        h = int(tc["hidden_size"])
        ff = pad_to(int(tc["intermediate_size"]))
        nq = int(tc["num_attention_heads"])
        self.d_h, self.d_ff, self.d_kinds = h, ff, list(tc["layer_types"])
        A = "assistant."
        self._f32(A + "output_norm.weight", ASSIST + "norm.weight", [h], ck=dr)
        self._f8(A + "head.weight", "lm_head.weight" if "lm_head.weight" in dr.t else ASSIST + "embed_tokens.weight",
                 [h, self.n_vocab], ck=dr, per_row=True)
        self._f8(A + "pre_proj.weight", "pre_projection.weight", [2 * self.h, h], ck=dr)
        self._f8(A + "post_proj.weight", "post_projection.weight", [h, self.h], ck=dr)
        for l, kind in enumerate(self.d_kinds):
            hd = int(tc["global_head_dim"] if kind == "full_attention" else tc["head_dim"])
            L = f"{ASSIST}layers.{l}."
            b = f"{A}blk.{l}."
            self._f32(b + "attn_norm.weight", L + "input_layernorm.weight", [h], ck=dr)
            self._f8(b + "attn_q.weight", L + "self_attn.q_proj.weight", [h, nq * hd], ck=dr)
            self._f8(b + "attn_output.weight", L + "self_attn.o_proj.weight", [nq * hd, h], ck=dr)
            self._f32(b + "attn_q_norm.weight", L + "self_attn.q_norm.weight", [hd], ck=dr)
            self._f32(b + "post_attention_norm.weight", L + "post_attention_layernorm.weight", [h], ck=dr)
            self._f32(b + "ffn_norm.weight", L + "pre_feedforward_layernorm.weight", [h], ck=dr)
            f = int(tc["intermediate_size"])
            self._f8(b + "ffn_gate.weight", L + "mlp.gate_proj.weight", [h, ff], ck=dr, pad_rows=f)
            self._f8(b + "ffn_up.weight", L + "mlp.up_proj.weight", [h, ff], ck=dr, pad_rows=f)
            self._f8(b + "ffn_down.weight", L + "mlp.down_proj.weight", [ff, h], ck=dr, pad_cols=f)
            self._f32(b + "post_ffw_norm.weight", L + "post_feedforward_layernorm.weight", [h], ck=dr)
            self._f32(b + "layer_scalar", L + "layer_scalar", [1], ck=dr)

    def _src(self, ck, src, ne, dtypes=("BF16", "F32")):
        if src not in ck.t:
            raise S.ConvertError(f"{ck.dir.name}: no tensor {src}")
        _, _, dt, shape, _ = ck.t[src]
        if dt not in dtypes:
            raise S.ConvertError(f"{src} is {dt}")
        return shape

    def _f32(self, name, src, ne, form="F32", ck=None):
        ck = ck or self.ck
        shape = self._src(ck, src, ne)
        if int(np.prod(shape)) != int(np.prod(ne)):
            raise S.ConvertError(f"{src} has shape {shape}, expected {list(reversed(ne))}")
        self.tensors[name] = Tensor(name, form, src, "id", ne, {"ck": ck})

    def _f8(self, name, src, ne, ck=None, pad_rows=None, pad_cols=None, per_row=False):
        """A [rows][K] projection as E4M3 codes; ne = [K, rows] in the pack (padded), the source [rows'][K']."""
        ck = ck or self.ck
        shape = self._src(ck, src, ne, ("BF16",))
        want = [pad_rows or ne[1], pad_cols or ne[0]]
        if shape != want:
            raise S.ConvertError(f"{src} has shape {shape}, expected {want}")
        if ne[0] % PAD:
            raise S.ConvertError(f"{src}: rows of {ne[0]} values (the products take multiples of {PAD})")
        self.tensors[name] = Tensor(name, "F8_E4M3", src, "rows" if per_row else "f8", ne,
                                    {"ck": ck, "src_shape": shape})
        sname = name[: -len(".weight")] + ".scale"
        self.tensors[sname] = Tensor(sname, "F32", src, "row_scales" if per_row else "scale",
                                     [ne[1]] if per_row else [1], {"ck": ck, "src_shape": shape})

    def _fp4(self, name, src, ne):
        """A [rows][K] projection as NVFP4 planes; ne = [K, rows] as the source [rows][K]."""
        shape = self._src(self.ck, src, ne, ("BF16",))
        if shape != [ne[1], ne[0]]:
            raise S.ConvertError(f"{src} has shape {shape}, expected {[ne[1], ne[0]]}")
        if ne[1] % 128 or ne[0] % FP4_K:
            raise S.ConvertError(f"{src}: NVFP4 planes take rows in 128s and K in multiples of {FP4_K}, not {shape}")
        self.tensors[name] = Tensor(name, "NVFP4_PLANES", src, "fp4", ne)
        sname = name[: -len(".weight")] + ".scale"
        self.tensors[sname] = Tensor(sname, "F32", src, "fp4_scale", [1])

    # ---- values
    def float_bytes(self, name) -> bytes:
        t = self.tensors[name]
        ck = t.extra.get("ck", self.ck)
        if t.how == "router":   # proj [E][h] * scale[h] / sqrt(h), in float64, rounded once to FP32
            p = ck.values(t.src).astype(np.float64)
            s = ck.values(t.extra["scale"]).astype(np.float64)
            x = (p * s[None, :] / math.sqrt(self.h)).astype(np.float32)
        elif t.how == "scale":
            x = np.array([_f8_scale(_amax(ck, t.src))], np.float32)
        elif t.how == "fp4_scale":
            x = np.array([Q.fp4q_global(_amax(ck, t.src))], np.float32)
        elif t.how == "row_scales":
            x = Q.f8q_row_scales(_row_amax(ck, t.src))
        else:
            x = np.ascontiguousarray(ck.values(t.src), dtype=np.float32).reshape(-1)
        if not np.isfinite(x).all():
            raise S.ConvertError(f"{name} has a non-finite value")
        if t.form == "F32":
            return x.astype("<f4").tobytes()
        u = x.view(np.uint32)
        if np.count_nonzero(u & 0xFFFF):
            raise S.ConvertError(f"{name}: a value that is not BF16")
        return (u >> 16).astype("<u2").tobytes()

    def chunks(self, name, pool, depth: int = 32):
        """An F8 matrix's codes, padded, in row chunks; an NVFP4 matrix's code plane in row chunks, then its scale plane."""
        t = self.tensors[name]
        ck = t.extra.get("ck", self.ck)
        p, off, _, (src_rows, src_cols), _ = ck.t[t.src]
        K, rows = t.ne
        if t.how == "fp4":   # the shared NVFP4 rounding (checkpoint_qwen35's fp4q job): (codes, scales) per row block
            gs = float(Q.fp4q_global(_amax(ck, t.src)))
            jobs = (("fp4q", str(p), off, K, r, min(FP4_ROWS_PER_JOB, rows - r), gs) for r in range(0, rows, FP4_ROWS_PER_JOB))
            scales = []
            for b in S.bounded_imap(pool, Q._job, jobs, depth):
                if isinstance(b, str):
                    raise S.ConvertError(f"{name}: {b}")
                scales.append(b[1])
                yield b[0]
            yield Q.nvfp4_scale_tiles(np.concatenate(scales)).tobytes()
            return
        if t.how == "rows":
            sc = Q.f8q_row_scales(_row_amax(ck, t.src))
            jobs = (("f8rows", str(p), off, src_cols, K, r, min(ROWS_PER_JOB, rows - r), src_rows,
                     sc[r:r + ROWS_PER_JOB]) for r in range(0, rows, ROWS_PER_JOB))
        else:
            s = _f8_scale(_amax(ck, t.src))
            jobs = (("f8", str(p), off, src_cols, K, r, min(ROWS_PER_JOB, rows - r), src_rows, s)
                    for r in range(0, rows, ROWS_PER_JOB))
        for b in S.bounded_imap(pool, _job, jobs, depth):
            if isinstance(b, str):
                raise S.ConvertError(f"{name}: {b}")
            yield b


def _f8_scale(amax: float) -> float:
    return float(np.float32(amax) / np.float32(448)) if amax > 0 else 1.0


_AMAX: dict = {}


def _amax(ck, src) -> float:
    key = (str(ck.dir), src)
    if key not in _AMAX:
        _AMAX[key] = float(_row_amax(ck, src).max())
    return _AMAX[key]


_ROW_AMAX: dict = {}


def _row_amax(ck, src) -> np.ndarray:
    key = (str(ck.dir), src)
    if key not in _ROW_AMAX:
        p, off, _, (rows, cols), _ = ck.t[src]
        mm = np.memmap(p, dtype="<u2", mode="r", offset=off, shape=(rows, cols))
        step = max(1, (1 << 24) // cols)
        _ROW_AMAX[key] = np.concatenate([np.abs(S._bf16_to_f32(np.ascontiguousarray(mm[r:r + step]))).max(-1)
                                         for r in range(0, rows, step)])
    return _ROW_AMAX[key]


# ------------------------------------------------------------------ routed experts
class Experts:
    """The `experts` section: per MoE layer the NVFP4 or MXFP8 blobs of its routed experts."""

    def __init__(self, ck: C.Checkpoint, mxfp8: frozenset = frozenset()):
        tc = ck.text
        self.ck = ck
        self.n_layers, self.n_expert = int(tc["num_hidden_layers"]), int(tc["num_experts"])
        self.h, self.ff = int(tc["hidden_size"]), int(tc["moe_intermediate_size"])
        if self.h % 64 or self.ff % 64:
            raise S.ConvertError("NVFP4 experts need hidden_size and moe_intermediate_size multiples of 64")
        if mxfp8 and self.h % 128:
            raise S.ConvertError("MXFP8 experts need a hidden_size that is a multiple of 128")
        if not mxfp8 <= set(range(self.n_layers)):
            raise S.ConvertError(f"--experts-mxfp8: layers {sorted(mxfp8 - set(range(self.n_layers)))} do not exist")
        self.mxfp8 = sorted(mxfp8)
        self.fmt = {l: MXFP8 if l in mxfp8 else NVFP4 for l in range(self.n_layers)}
        ffk = pad_to(self.ff)   # an MXFP8 down row holds whole 128-value blocks: its columns past ff are zero
        self.blobs = {NVFP4: 2 * self.ff * S.nvfp4_row_bytes(self.h) + self.h * S.nvfp4_row_bytes(self.ff) + S.NVFP4_TAIL,
                      MXFP8: 2 * self.ff * (self.h // 128 * 132) + self.h * (ffk // 128 * 132)}
        self.layers, offset = [], 0
        for l in range(self.n_layers):
            p = f"{TEXT}layers.{l}."
            ck.need(p + "experts.gate_up_proj", "BF16", (self.n_expert, 2 * self.ff, self.h))
            ck.need(p + "experts.down_proj", "BF16", (self.n_expert, self.h, self.ff))
            ck.need(p + "router.per_expert_scale", "BF16", (self.n_expert,))
            self.layers.append((l, offset))
            offset += self.blobs[self.fmt[l]] * self.n_expert
        self.total = offset

    def layout_text(self) -> bytes:
        head = ("# experts v6: layer gu_type d_type offset blob_bytes (n_expert %d, total %d); "
                "an NVFP4 blob ends in FP32 {s_gate, s_up, s_down, 0}\n" % (self.n_expert, self.total))
        return (head + "".join("%d %d %d %d %d\n" % (l, self.fmt[l], self.fmt[l], off, self.blobs[self.fmt[l]])
                               for l, off in self.layers)).encode()

    def jobs(self):
        ck = self.ck
        for l, _ in self.layers:
            p = f"{TEXT}layers.{l}."
            gu, dn = ck.t[p + "experts.gate_up_proj"], ck.t[p + "experts.down_proj"]
            pes = ck.values(p + "router.per_expert_scale").astype(np.float32)
            for s in range(0, self.n_expert, EXPERTS_PER_JOB):
                n = min(EXPERTS_PER_JOB, self.n_expert - s)
                yield ("experts", self.fmt[l], str(gu[0]), gu[1], str(dn[0]), dn[1], self.h, self.ff, s, n, pes[s:s + n])

    def chunks(self, pool, depth: int = 8):
        for b in S.bounded_imap(pool, _job, self.jobs(), depth):
            if isinstance(b, str):
                raise S.ConvertError(b)
            yield b


def _nvfp4_matrix(x: np.ndarray):
    gs = Q.fp4q_global(float(np.abs(x).max()))
    codes, sc = Q.fp4q_rows(x, gs)
    return C.nvfp4_rows(codes, sc).tobytes(), gs


def _mxfp8_matrix(x: np.ndarray) -> bytes:
    """[rows][K] values (K a multiple of 128) as MXFP8: per 32 values the UE8M0 scale, the smallest power of two that
    brings the largest magnitude within E4M3's 448, then the codes to nearest."""
    rows, K = x.shape
    b = x.reshape(rows, K // 32, 32).astype(np.float64)
    amax = np.abs(b).max(axis=2)
    e = np.frexp(amax / 448.0)[1].astype(np.int64)   # within one of the answer: settled by exact comparisons
    e = np.where(amax <= np.ldexp(448.0, e - 1), e - 1, e)
    e = np.where(amax > np.ldexp(448.0, e), e + 1, e)
    e = np.clip(np.where(amax > 0, e, -127), -127, 127)
    codes = Q.e4m3_round(np.ldexp(b, -e[:, :, None])).reshape(rows, K)
    return C.mxfp8_rows(codes, (e + 127).astype(np.uint8)).tobytes()


def _expert_blobs_mxfp8(gu_path, gu_off, dn_path, dn_off, h, ff, e0, n, pes) -> bytes:
    out = []
    for i in range(n):
        e = e0 + i
        gu = S._bf16_to_f32(np.fromfile(gu_path, "<u2", count=2 * ff * h, offset=gu_off + e * 2 * ff * h * 2)).reshape(2 * ff, h)
        dn = S._bf16_to_f32(np.fromfile(dn_path, "<u2", count=h * ff, offset=dn_off + e * h * ff * 2)).reshape(h, ff)
        if not (np.isfinite(gu).all() and np.isfinite(dn).all()):
            return f"expert {e}: a non-finite value"
        if not (np.isfinite(pes[i]) and pes[i] > 0):
            return f"expert {e}: a per-expert scale that is not finite and positive"
        dk = np.zeros((h, pad_to(ff)), np.float64)
        dk[:, :ff] = dn * np.float64(pes[i])   # no tail in an MXFP8 blob: the per-expert scale rides in the down weights (exact)
        out += [_mxfp8_matrix(gu[:ff]), _mxfp8_matrix(gu[ff:]), _mxfp8_matrix(dk)]
    return b"".join(out)


def _expert_blobs(gu_path, gu_off, dn_path, dn_off, h, ff, e0, n, pes) -> bytes:
    out = []
    for i in range(n):
        e = e0 + i
        gu = S._bf16_to_f32(np.fromfile(gu_path, "<u2", count=2 * ff * h, offset=gu_off + e * 2 * ff * h * 2)).reshape(2 * ff, h)
        dn = S._bf16_to_f32(np.fromfile(dn_path, "<u2", count=h * ff, offset=dn_off + e * h * ff * 2)).reshape(h, ff)
        if not (np.isfinite(gu).all() and np.isfinite(dn).all()):
            return f"expert {e}: a non-finite value"
        bg, sg = _nvfp4_matrix(gu[:ff])
        bu, su = _nvfp4_matrix(gu[ff:])
        bd, sd = _nvfp4_matrix(dn)
        tail = np.array([sg, su, np.float32(sd) * np.float32(pes[i]), 0.0], dtype="<f4")
        if not (np.isfinite(tail).all() and (tail[:3] > 0).all()):
            return f"expert {e}: a scale that is not finite and positive"
        out += [bg, bu, bd, tail.tobytes()]
    return b"".join(out)


def _job(args):
    if args[0] == "experts":
        return (_expert_blobs_mxfp8 if args[1] == MXFP8 else _expert_blobs)(*args[2:])
    op, path, off, src_cols, K, row0, rows, src_rows, sc = args
    # rows [row0, row0 + rows) of the padded [.][K] matrix; source rows past src_rows and columns past src_cols are 0
    x = np.zeros((rows, K), np.float64)
    have = max(0, min(rows, src_rows - row0))
    if have:
        mm = np.memmap(path, dtype="<u2", mode="r", offset=off + row0 * src_cols * 2, shape=(have, src_cols))
        v = S._bf16_to_f32(np.ascontiguousarray(mm))
        if not np.isfinite(v).all():
            return "a non-finite value"
        x[:have, :src_cols] = v
    if op == "f8rows":
        x /= np.asarray(sc, np.float64)[:, None]
    else:
        x /= np.float64(sc)
    return Q.e4m3_round(x).tobytes()


# ------------------------------------------------------------------ tokenizer
def _byte_spelling():
    """GPT-2's byte alphabet: a printable byte stands for itself, the others for U+0100 upward in byte order."""
    out, nxt = {}, 256
    for b in range(256):
        plain = 33 <= b <= 126 or 161 <= b <= 172 or 174 <= b <= 255
        out[b] = chr(b) if plain else chr(nxt)
        if not plain:
            nxt += 1
    return out


def tokenizer_kv(ck: C.Checkpoint) -> dict:
    """The checkpoint's SentencePiece-style BPE (normalizer: space -> U+2581, byte fallback) as the engine's BPE tables:
    a token's bytes are its text with U+2581 as a space, spelled in GPT-2's byte alphabet; a byte-fallback token keeps
    its `<0xNN>` text and type 6 (the engine reads it as that byte). The engine starts a piece from its characters (a byte-fallback token
    for a character the vocabulary lacks) and merges by rank, as the checkpoint's tokenizer does."""
    d = ck.dir
    tj = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    tcfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    model = tj.get("model", {})
    if model.get("type") != "BPE" or not model.get("byte_fallback"):
        raise S.ConvertError("tokenizer.json: not a BPE model with byte fallback")
    norm = tj.get("normalizer") or {}
    if norm.get("type") != "Replace" or norm.get("pattern", {}).get("String") != " " or norm.get("content") != "▁":
        raise S.ConvertError("tokenizer.json: the normalizer is not the space -> U+2581 replace")
    pre = tj.get("pre_tokenizer") or {}
    if pre and not (pre.get("type") == "Split" and pre.get("pattern", {}).get("String") == " "):
        raise S.ConvertError("tokenizer.json: a pre-tokenizer other than the split on spaces (a no-op after the normalizer)")
    n_vocab = int(ck.text["vocab_size"])
    spell = _byte_spelling()
    vocab = model["vocab"]
    rev = {int(i): t for t, i in vocab.items()}
    added = {int(a["id"]): a for a in tj.get("added_tokens", [])}
    if max(rev) >= n_vocab:
        raise S.ConvertError(f"tokenizer.json: id {max(rev)} is beyond the vocabulary of {n_vocab}")
    tokens, types, raw_of = [], [], {}
    seen = {}
    for i in range(n_vocab):
        if i in added:
            tokens.append(added[i]["content"])
            types.append(C.CONTROL if added[i].get("special") else C.USER_DEFINED)
            continue
        if i not in rev:
            tokens.append(f"[PAD{i}]")
            types.append(C.UNUSED)
            continue
        t = rev[i]
        m = re.fullmatch(r"<0x([0-9A-F]{2})>", t)
        raw = bytes([int(m.group(1), 16)]) if m else t.replace("▁", " ").encode("utf-8")
        s = "".join(spell[b] for b in raw)
        if not m:
            if s in seen:
                raise S.ConvertError(f"tokenizer.json: tokens {seen[s]} and {i} have the same bytes")
            seen[s] = i
        raw_of[t] = s
        tokens.append(t if m else s)   # a byte-fallback token keeps its <0xNN> text (its byte spells a character's)
        types.append(BYTE if m else C.NORMAL)
    merges = []
    for mg in model["merges"]:
        a, b = mg if isinstance(mg, list) else mg.split(" ")
        if a not in raw_of or b not in raw_of or (a + b) not in raw_of:
            raise S.ConvertError(f"tokenizer.json: merge {a!r} {b!r} outside the vocabulary")
        merges.append(raw_of[a] + " " + raw_of[b])

    def token_id(key):
        v = tcfg.get(key)
        v = v.get("content") if isinstance(v, dict) else v
        return tokens.index(v) if v in tokens[:1024] else None

    kv = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": ARCH, "tokenizer.ggml.tokens": tokens,
          "tokenizer.ggml.token_type": types, "tokenizer.ggml.merges": merges}
    for k, key in (("eos", "eos_token"), ("padding", "pad_token"), ("bos", "bos_token")):
        v = token_id(key)
        if v is not None:
            kv[f"tokenizer.ggml.{k}_token_id"] = int(v)
    kv["tokenizer.ggml.add_bos_token"] = False   # the chat template writes <bos>
    tpl = d / "chat_template.jinja"
    template = tpl.read_text(encoding="utf-8") if tpl.is_file() else tcfg.get("chat_template")
    if template:
        kv["tokenizer.chat_template"] = template
    return kv


# ------------------------------------------------------------------ metadata
def model_kv(ck: C.Checkpoint, drafter: C.Checkpoint | None, name: str | None) -> dict:
    c, tc = ck.config, ck.text
    A = ARCH + "."
    d = ck.dir
    gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8")) if (d / "generation_config.json").is_file() else {}
    card = C.front_matter(d / "README.md")
    rope = tc["rope_parameters"]
    full_r, swa_r = rope["full_attention"], rope["sliding_attention"]
    if swa_r.get("rope_type") != "default" or full_r.get("rope_type") != "proportional":
        raise S.ConvertError("config.json: rope types other than default (sliding) and proportional (full)")
    if tc.get("final_logit_softcapping") is None:
        raise S.ConvertError("config.json: no final_logit_softcapping")
    kinds = tc["layer_types"]
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
    h = int(tc["hidden_size"])
    kv.update({
        A + "block_count": int(tc["num_hidden_layers"]),
        A + "context_length": int(tc["max_position_embeddings"]),
        A + "embedding_length": h,
        # the embedding is scaled by sqrt(hidden) rounded to BF16, as the checkpoint's BF16 model multiplies it
        A + "embedding_scale": _bf16_value(math.sqrt(h)),
        A + "vocab_size": int(tc["vocab_size"]),
        A + "feed_forward_length": int(tc["intermediate_size"]),
        A + "attention.head_count": int(tc["num_attention_heads"]),
        A + "attention.head_count_kv": int(tc["num_key_value_heads"]),
        A + "attention.head_count_kv_full": int(tc["num_global_key_value_heads"] if tc.get("attention_k_eq_v")
                                                else tc["num_key_value_heads"]),
        A + "attention.key_length": int(tc["head_dim"]),
        A + "attention.key_length_full": int(tc["global_head_dim"]),
        A + "attention.k_eq_v": bool(tc.get("attention_k_eq_v")),
        A + "attention.layer_norm_rms_epsilon": C.f32(tc["rms_norm_eps"]),
        A + "attention.sliding_window": int(tc["sliding_window"]),
        A + "attention.sliding_window_pattern": [k == "sliding_attention" for k in kinds],
        A + "rope.freq_base": C.f32(full_r["rope_theta"]),
        A + "rope.freq_base_swa": C.f32(swa_r["rope_theta"]),
        A + "rope.partial_rotary_factor": C.f32(full_r.get("partial_rotary_factor", 1.0)),
        A + "final_logit_softcapping": C.f32(tc["final_logit_softcapping"]),
        A + "image_token_id": int(c["image_token_id"]),
        "general.file_type": 40,
        "general.quantization_version": 2,
    })
    if tc.get("enable_moe_block"):
        kv.update({A + "expert_count": int(tc["num_experts"]), A + "expert_used_count": int(tc["top_k_experts"]),
                   A + "expert_feed_forward_length": int(tc["moe_intermediate_size"])})
    if drafter is not None:
        dc, dt = drafter.config, drafter.text
        D = A + "assistant."
        gen_d = drafter.dir / "generation_config.json"
        n_draft = json.loads(gen_d.read_text(encoding="utf-8")).get("num_assistant_tokens", 0) if gen_d.is_file() else 0
        kv.update({
            D + "block_count": int(dt["num_hidden_layers"]),
            D + "embedding_length": int(dt["hidden_size"]),
            D + "feed_forward_length": int(dt["intermediate_size"]),
            D + "attention.head_count": int(dt["num_attention_heads"]),
            D + "attention.sliding_window_pattern": [k == "sliding_attention" for k in dt["layer_types"]],
            D + "draft_tokens": int(n_draft),
        })
        for k in ("head_dim", "global_head_dim", "num_key_value_heads", "num_global_key_value_heads", "sliding_window",
                  "rms_norm_eps"):
            if dt.get(k) != tc.get(k):
                raise S.ConvertError(f"the drafter's {k} ({dt.get(k)}) differs from the model's ({tc.get(k)})")
        if dt.get("rope_parameters") != tc.get("rope_parameters"):
            raise S.ConvertError("the drafter's rope differs from the model's")
    kv.update(tokenizer_kv(ck))
    return kv


def _bf16_value(v: float) -> float:
    b = Q.bf16_round(np.array([v], np.float64))[0]
    return float(np.array([int(b) << 16], np.uint32).view(np.float32)[0])


# ------------------------------------------------------------------ the vision tower
VISION = "model.vision_tower."
# checkpoint tensor after `model.vision_tower.encoder.layers.N.` -> (GGUF name after `v.blk.N.`, form)
VISION_BLOCK = {
    "input_layernorm.weight": ("ln1.weight", "F32"),
    "post_attention_layernorm.weight": ("attn_post_norm.weight", "F32"),
    "pre_feedforward_layernorm.weight": ("ln2.weight", "F32"),
    "post_feedforward_layernorm.weight": ("ffn_post_norm.weight", "F32"),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "F32"),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", "F32"),
    "self_attn.q_proj.linear.weight": ("attn_q.weight", "BF16"),
    "self_attn.k_proj.linear.weight": ("attn_k.weight", "BF16"),
    "self_attn.v_proj.linear.weight": ("attn_v.weight", "BF16"),
    "self_attn.o_proj.linear.weight": ("attn_out.weight", "BF16"),
    "mlp.gate_proj.linear.weight": ("ffn_gate.weight", "BF16"),
    "mlp.up_proj.linear.weight": ("ffn_up.weight", "BF16"),
    "mlp.down_proj.linear.weight": ("ffn_down.weight", "BF16"),
}
VISION_GLOBAL = {
    VISION + "patch_embedder.position_embedding_table": ("v.position_embd.weight", "F32"),   # [2][size][hidden]: columns, then rows
    VISION + "std_bias": ("v.std_bias", "F32"),
    VISION + "std_scale": ("v.std_scale", "F32"),
    "model.embed_vision.embedding_projection.weight": ("mm.input_projection.weight", "BF16"),
}


def has_vision(ck: C.Checkpoint) -> bool:
    return bool(ck.config.get("vision_config")) and VISION + "patch_embedder.input_proj.weight" in ck.t


def mmproj(ck: C.Checkpoint, out: pathlib.Path, name: str) -> None:
    """The vision tower as the clip GGUF the engine's encoder reads (llama.cpp's gemma4v layout)."""
    vc = ck.config["vision_config"]
    pp = json.loads((ck.dir / "processor_config.json").read_text(encoding="utf-8")).get("image_processor", {})
    for k, want in (("model_type", "gemma4_vision"), ("hidden_activation", "gelu_pytorch_tanh"), ("attention_bias", False),
                    ("use_clipped_linears", False), ("standardize", True)):
        if vc.get(k) != want:
            raise S.ConvertError(f"config.json: vision_config.{k} = {vc.get(k)!r}; the engine runs {want!r}")
    h, patch, pool = int(vc["hidden_size"]), int(vc["patch_size"]), int(vc["pooling_kernel_size"])
    if int(pp.get("patch_size", patch)) != patch or int(pp.get("pooling_kernel_size", pool)) != pool or pp.get("do_normalize"):
        raise S.ConvertError("processor_config.json: an image processor other than the tower's (patch, pooling, no normalisation)")
    if int(vc["num_attention_heads"]) * int(vc["head_dim"]) != h or vc.get("rope_parameters", {}).get("rope_type") != "default":
        raise S.ConvertError("config.json: a vision tower whose heads do not tile its width, or a rope other than the default")
    w = GGUFWriter()
    for k, v, t in (("general.architecture", "clip", "string"), ("general.type", "mmproj", "string"),
                    ("general.name", name + " (mmproj)", "string"), ("clip.has_vision_encoder", True, "bool"),
                    ("clip.vision.projector_type", "gemma4v", "string"),
                    ("clip.vision.projection_dim", int(ck.text["hidden_size"]), "u32"),
                    ("clip.vision.image_size", 224, "u32"),   # unused, as llama.cpp's converter writes it
                    ("clip.vision.patch_size", patch, "u32"),
                    ("clip.vision.embedding_length", h, "u32"),
                    ("clip.vision.feed_forward_length", int(vc["intermediate_size"]), "u32"),
                    ("clip.vision.block_count", int(vc["num_hidden_layers"]), "u32"),
                    ("clip.vision.attention.head_count", int(vc["num_attention_heads"]), "u32"),
                    ("clip.vision.attention.layer_norm_epsilon", C.f32(vc["rms_norm_eps"]), "f32"),
                    ("clip.vision.projector.scale_factor", pool, "u32"),
                    ("clip.vision.rope.freq_base", C.f32(vc["rope_parameters"]["rope_theta"]), "f32")):
        w.add(k, v, t)
    w.add("clip.vision.image_mean", [float(x) for x in pp.get("image_mean", [0.0] * 3)], "array:f32")
    w.add("clip.vision.image_std", [float(x) for x in pp.get("image_std", [1.0] * 3)], "array:f32")

    def put(gname, x, ty):
        if ty == "BF16":
            u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32) >> 16
            w.add_raw(gname, list(reversed(x.shape)), "BF16", u.astype("<u2").tobytes())
        else:
            a = np.ascontiguousarray(x, dtype="<f4")
            w.add_raw(gname, list(reversed(a.shape)), "F32", a.tobytes())

    used = {VISION + "patch_embedder.input_proj.weight", *VISION_GLOBAL}
    used |= {f"{VISION}encoder.layers.{l}.{k}" for l in range(int(vc["num_hidden_layers"])) for k in VISION_BLOCK}
    left = [k for k in ck.t if k.startswith((VISION, "model.embed_vision.")) and k not in used]
    if left:
        raise S.ConvertError("vision tensors with no pack counterpart: " + ", ".join(left[:8]))
    pw = ck.values(VISION + "patch_embedder.input_proj.weight")   # [out, (row, column, channel)]
    if pw.shape != (h, 3 * patch * patch):
        raise S.ConvertError(f"the patch embedding has shape {list(pw.shape)}")
    put("v.patch_embd.weight", pw.reshape(h, patch, patch, 3).transpose(0, 3, 1, 2), "F32")
    for k, (g, ty) in VISION_GLOBAL.items():
        put(g, ck.values(k), ty)
    for l in range(int(vc["num_hidden_layers"])):
        for k, (g, ty) in VISION_BLOCK.items():
            put(f"v.blk.{l}.{g}", ck.values(f"{VISION}encoder.layers.{l}.{k}"), ty)
    w.write(out)


# ------------------------------------------------------------------ the pack
LICENSE = {"apache-2.0": "Apache License 2.0", "gemma": "Gemma Terms of Use"}


def mxfp8_note(layers: list[int]) -> str:
    """The meta's account of the layers whose experts are MXFP8."""
    return ("layers " + ", ".join(map(str, layers)) + ": MXFP8 (type 46) rounded once from BF16 instead: per 32 values "
            "the UE8M0 scale that brings the largest magnitude within 448, E4M3 to nearest (ties to even); the down rows "
            "padded with zero columns to whole 128-value blocks; the router's per_expert_scale folded into the down "
            "weights (no tail)")


def parse_layers(text: str | None) -> list[int]:
    """`--experts-mxfp8 0,5,10-12` -> [0, 5, 10, 11, 12]."""
    out = []
    for part in (text or "").split(","):
        if part.strip():
            lo, _, hi = part.partition("-")
            out += range(int(lo), int(hi or lo) + 1)
    return out


def build(a, Job, fingerprint, die) -> None:
    import os
    import tokenizer
    out = pathlib.Path(a.out)
    partial = pathlib.Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    try:
        ck = C.Checkpoint(a.checkpoint, single=(pathlib.Path(a.checkpoint) / "model.safetensors").is_file())
        dr = C.Checkpoint(a.drafter, single=True) if a.drafter else None
        plan = Plan(ck, dr, a.mlp_nvfp4)
        experts = Experts(ck, frozenset(parse_layers(a.experts_mxfp8))) if plan.moe else None
        name = a.name or "Gemma 4"
        kv = model_kv(ck, dr, name)
        vision = has_vision(ck)
        for k in ("vision_config", "audio_config"):   # the towers the pack does not carry
            if ck.config.get(k) and not (k == "vision_config" and vision):
                print(f"not converted: the checkpoint's {ck.config[k].get('model_type')} (the pack reads text only)",
                      flush=True)
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
            if experts:
                job.chunks("experts", experts.chunks(pool))
                job.bytes("experts.layout", experts.layout_text())
            t = time.time()
            parts, names = [], []
            for n, info in plan.tensors.items():
                if info.form in ("F8_E4M3", "NVFP4_PLANES"):
                    parts.append(Stream(info.expected_bytes(), plan.chunks(n, pool)))
                else:
                    parts.append(plan.float_bytes(n))
                names.append(n)
            offs = w.add_parts("gguf", parts, TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            if vision:
                mp = pathlib.Path(tmp) / "mmproj.gguf"
                mmproj(ck, mp, name)
                job.file("vision/mmproj", mp)
            tens = {}
            for n, off in zip(names, offs):
                t_ = plan.tensors[n]
                tens[n] = {"type": t_.form, "type_id": t_.type_id, "shape": [int(x) for x in t_.ne],
                           "bytes": t_.expected_bytes(), "offset": off}
            lic = kv.get("general.license", "")
            meta = {
                "format": "basalt", "version": 2, "name": name, "arch": ARCH,
                "converter": {"tool": "tools/convert/convert.py"},
                "sources": [{"path": p.name, "size": p.stat().st_size,
                             "fingerprint_sha256_first_last_64mib_and_size": fingerprint(p)}
                            for p in ck.files + (dr.files if dr else [])],
                "gguf_kv": kv,
                "tensors": tens,
                "license": {"name": LICENSE.get(lic, lic)},
                "hf_config": ck.config,
                "assistant": {"source": dr.dir.name, "config": dr.config} if dr else None,
                "vision": {"projector": "gemma4v", "source": "the checkpoint's BF16 vision tower"} if vision else None,
                "checkpoint": {
                    "dir": pathlib.Path(a.checkpoint).name,
                    "config_sha256": hashlib.sha256((ck.dir / "config.json").read_bytes()).hexdigest(),
                    "experts": "NVFP4 rounded once from BF16: per matrix amax / (6 x 448), per 16 values the UE4M3 scale "
                               "of least squared error among half to four times amax / 6, E2M1 to nearest; the router's "
                               "per_expert_scale folded into s_down",
                    **({"experts_mxfp8": mxfp8_note(experts.mxfp8)} if experts and experts.mxfp8 else {}),
                    "projections": "F8_E4M3 to nearest (ties to even) with one FP32 scale per matrix, amax / 448; the "
                                   "MLP's inner width padded with zeros to a multiple of 128",
                    **({"mlp": "NVFP4_PLANES rounded once from BF16: per matrix amax / (6 x 448), per 16 values the UE4M3 "
                               "scale of least squared error among half to four times amax / 6, E2M1 to nearest"}
                       if plan.mlp_nvfp4 else {}),
                    "head": "the tied embedding as F8_E4M3 rows with an FP32 scale per row, amax / 448",
                    "router": "proj x scale / sqrt(hidden) in float64, rounded once to FP32"},
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
