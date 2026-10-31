"""A .basalt pack's language-model sections read straight from an NVFP4 safetensors checkpoint (NVIDIA Model
Optimizer's layout, as in nvidia/Qwen3.8-Flash-Next-NVFP4), for tools/convert/convert.py.

  experts / experts.layout  each routed expert's NVFP4 gate | up | down rows moved into Basalt's blocks (codes and
                            UE4M3 scales unchanged) with the FP32 tail {s_gate, s_up, s_down, 0} from `weight_scale_2`.
                            A projection's scales are either `weight_scale` (UE4M3 per 16 values) or the CSF pair
                            `weight_scale.nvfp4_csf_fixed` and `weight_scale.nvfp4_csf_exceptions`, which holds the
                            same scale bytes in 4-bit windows over a per-row base plus an exception list
  gguf, dense, dense.index  every other language-model tensor in the GGUF names, order and forms the engine reads:
                            BF16 kept, norms as F32 with the "+1" folded in, GDN value heads in the GGUF head order,
                            projections quantized once from BF16 or from MXFP8 (E4M3 values, one UE8M0 scale per 32
                            values, which dequantizes exactly) to Q6_K, or Q8_0 where rows are not whole 256-value
                            blocks, or for the 8-bit variant
  gguf_kv, tokenizer        the model metadata and tokenizer from config.json, generation_config.json, README.md and
                            the tokenizer files

The checkpoint's tensors are visited in file order, each file in header order, so the output order follows the
checkpoint; the shards sit in the directory itself or in its `tensors/` subdirectory, where a
`model.safetensors.index.json` names the shard of every tensor. Needs numpy.
"""
from __future__ import annotations

import json
import pathlib
import re
import struct

import numpy as np

import model_sections as S

NVFP4 = S.NVFP4
UNUSED, NORMAL, CONTROL, USER_DEFINED = 5, 1, 3, 4
# The pre-tokenizer split of the qwen35 vocabulary as Hugging Face tokenizers writes it.
QWEN35_HF_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
                   r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
VALUES_PER_JOB = 1 << 22
TYPE_ID = {"F32": 0, "F16": 1, "Q8_0": 8, "Q6_K": 14, "BF16": 30}


class Checkpoint:
    """The safetensors files of a checkpoint directory: name -> (path, absolute offset, dtype, shape, bytes)."""

    def __init__(self, d):
        self.dir = pathlib.Path(d)
        cfg = self.dir / "config.json"
        if not cfg.is_file():
            raise S.ConvertError(f"{cfg} not found")
        self.config = json.loads(cfg.read_text(encoding="utf-8"))
        self.text = self.config.get("text_config", self.config)
        self.files = sorted(set(self.dir.glob("*.safetensors")) | set((self.dir / "tensors").glob("*.safetensors")))
        self.shards = [p for p in self.files if re.fullmatch(r"model-\d+-of-\d+\.safetensors", p.name)]
        if not self.shards:
            raise S.ConvertError(f"{self.dir}: no model-NNNNN-of-NNNNN.safetensors shards, in the directory or in tensors/")
        self.t, self.order = {}, []
        for p in self.files:
            head, base = S.read_safetensors_header(p)
            items = sorted(((k, v) for k, v in head.items() if k != "__metadata__"),
                           key=lambda kv: kv[1]["data_offsets"][0])
            for k, v in items:
                if k in self.t:
                    raise S.ConvertError(f"{p.name}: duplicate tensor {k}")
                a, b = v["data_offsets"]
                self.t[k] = (p, base + a, v["dtype"], list(v["shape"]), b - a)
                if p in self.shards:
                    self.order.append(k)
                elif not SKIP.match(k):
                    raise S.ConvertError(f"{p.name} is not a model-NNNNN-of-NNNNN shard and holds {k}, which would be "
                                         "dropped; only the n-gram table, drafter and vision tensors may live there")
        self._check_index()

    def _check_index(self):
        """`model.safetensors.index.json`, when the checkpoint ships one: every shard it names is on disk and every
        tensor it names is in one of them, so a half-fetched checkpoint is refused before a section is written."""
        f = self.dir / "model.safetensors.index.json"
        if not f.is_file():
            return
        try:
            weight_map = json.loads(f.read_text(encoding="utf-8"))["weight_map"]
        except (ValueError, KeyError) as e:
            raise S.ConvertError(f"{f}: {e}") from None
        named = {self.dir / str(v) for v in weight_map.values()}
        missing = sorted(str(p.relative_to(self.dir)) for p in named - set(self.files))
        if missing:
            raise S.ConvertError(f"{f} names shards that are not in the checkpoint: " + ", ".join(missing[:4])
                                 + (f" ... and {len(missing) - 4} more" if len(missing) > 4 else ""))
        absent = [k for k in weight_map if k not in self.t]
        if absent:
            raise S.ConvertError(f"{f} names {len(absent)} tensors no shard holds, first: {absent[0]}")

    def raw(self, name) -> np.ndarray:
        p, off, _, _, n = self.t[name]
        return np.fromfile(p, dtype=np.uint8, count=n, offset=off)

    def values(self, name) -> np.ndarray:
        """A BF16, F32 or I64 tensor as float32 (int64 for I64), shaped."""
        p, off, dt, shape, n = self.t[name]
        raw = self.raw(name)
        if dt == "BF16":
            return S._bf16_to_f32(raw.view("<u2")).reshape(shape)
        if dt == "F32":
            return raw.view("<f4").reshape(shape)
        if dt == "I64":
            return raw.view("<i8").reshape(shape)
        raise S.ConvertError(f"{name} is {dt}")

    def need(self, name, dtype, shape=None):
        if name not in self.t:
            raise S.ConvertError(f"the checkpoint has no {name}")
        _, _, dt, sh, _ = self.t[name]
        if dt != dtype or (shape is not None and sh != list(shape)):
            raise S.ConvertError(f"{name} is {dt} {sh}, expected {dtype} {list(shape) if shape else ''}")


# ------------------------------------------------------------------ routed experts
def nvfp4_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Model Optimizer's NVFP4 rows -> Basalt's blocks. `codes` [rows, cols/2]: value 2i in the low nibble of byte i,
    2i+1 in the high one; `scales` [rows, cols/16] UE4M3. A Basalt block of 64 values is 4 scale bytes then 32 code
    bytes where sub-block s keeps value j in the low nibble of byte 8 s + j and value j + 8 in its high nibble."""
    rows, cols = codes.shape[0], codes.shape[1] * 2
    v = np.empty((rows, cols), np.uint8)
    v[:, 0::2] = codes & 15
    v[:, 1::2] = codes >> 4
    v = v.reshape(rows, cols // 64, 4, 16)
    out = np.empty((rows, cols // 64, 36), np.uint8)
    out[..., :4] = scales.reshape(rows, cols // 64, 4)
    out[..., 4:] = (v[..., :8] | (v[..., 8:] << 4)).reshape(rows, cols // 64, 32)
    return out.reshape(-1)


def csf_scales(fixed: np.ndarray, exceptions: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Model Optimizer's CSF scale stream -> the UE4M3 scale bytes as uint8 [rows, cols] (cols = values / 16).
    `fixed` is uint8 [rows / 16, 16 * (1 + cols / 2)]: per slab of 16 rows the 16 rows' base bytes, then each row's
    `cols` 4-bit window offsets, value 2i in the low nibble of byte i; a scale byte is base + offset. `exceptions`
    is uint32 words whose low 24 bits are the flat position in the matrix and whose top 8 bits are the scale byte
    that belongs there, applied last because it lies outside the row's 16-wide window."""
    if fixed.dtype != np.uint8 or fixed.ndim != 2 or fixed.shape[0] != rows // 16 or fixed.shape[1] % 16:
        raise S.ConvertError(f"a CSF fixed stream of {fixed.dtype} {fixed.shape} for {rows} rows")
    if (fixed.shape[1] // 16 - 1) * 2 != cols:
        raise S.ConvertError(f"a CSF fixed stream of {fixed.shape} for {cols} scale columns")
    if exceptions.dtype != np.uint32:
        raise S.ConvertError(f"a CSF exception stream of {exceptions.dtype}")
    bases = fixed[:, :16].reshape(rows, 1).astype(np.int16)
    if (bases > 240).any():      # a base above 240 cannot hold its window without carrying out of the byte
        raise S.ConvertError("a CSF row base above 240")
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(rows, cols)
    out = (bases + codes).reshape(-1)
    words = exceptions.reshape(-1).astype(np.int64)
    pos = words & 0xFFFFFF
    if pos.size and int(pos.max()) >= rows * cols:
        raise S.ConvertError("a CSF exception position outside the scale matrix")
    out[pos] = words >> 24
    return out.astype(np.uint8).reshape(rows, cols)


def _expert_job(args):
    """Blobs of experts [s, s + n) of one layer: [gate | up | down] blocks and the FP32 scale tail."""
    entries, n = args
    out = []
    for e in range(n):
        parts, tail = [], []
        for (wp, wo, wshape), (sp, so, sshape), xinfo, (tp, to) in entries[e]:
            w = np.fromfile(wp, dtype=np.uint8, count=wshape[0] * wshape[1], offset=wo).reshape(wshape)
            if xinfo is None:
                sc = np.fromfile(sp, dtype=np.uint8, count=sshape[0] * sshape[1], offset=so).reshape(sshape)
            else:
                xp, xo, xn = xinfo
                fixed = np.fromfile(sp, dtype=np.uint8, count=sshape[0] * sshape[1], offset=so).reshape(sshape)
                exc = np.fromfile(xp, dtype="<u4", count=xn, offset=xo)
                sc = csf_scales(fixed, exc, wshape[0], wshape[1] // 8)
            if (sc & 0x80).any() or (sc == 0x7F).any():
                return f"a UE4M3 scale with the sign bit set or NaN at {wp.name}+{wo}"
            parts.append(nvfp4_rows(w, sc))
            tail.append(np.fromfile(tp, dtype="<f4", count=1, offset=to)[0])
        t = np.array(tail + [0.0], dtype="<f4")
        if not (np.isfinite(t).all() and (t[:3] > 0).all()):
            return f"a weight_scale_2 that is not finite and positive: {t[:3].tolist()}"
        out.append(np.concatenate(parts).tobytes())
        out.append(t.tobytes())
    return b"".join(out)


bounded_imap = S.bounded_imap


class CheckpointExperts:
    """The `experts` section from the checkpoint's per-expert NVFP4 tensors (`input_scale` is not used): codes as
    `weight`, scales plain (`weight_scale`) or CSF-encoded (`weight_scale.nvfp4_csf_fixed` and its exceptions)."""

    ROLES = ("gate_proj", "up_proj", "down_proj")

    def __init__(self, ck: Checkpoint):
        tc = ck.text
        self.ck = ck
        self.n_layers, self.n_expert = int(tc["num_hidden_layers"]), int(tc["num_experts"])
        h, ff = int(tc["hidden_size"]), int(tc["moe_intermediate_size"])
        shapes = {"gate_proj": (ff, h), "up_proj": (ff, h), "down_proj": (h, ff)}
        self.blob = 2 * ff * S.nvfp4_row_bytes(h) + h * S.nvfp4_row_bytes(ff) + S.NVFP4_TAIL
        known = {f"{r}.{k}" for r in self.ROLES for k in ("weight", "weight_scale", "weight_scale_2", "input_scale",
                                                           "weight_scale.nvfp4_csf_fixed",
                                                           "weight_scale.nvfp4_csf_exceptions")}
        for k in ck.t:
            m = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)", k)
            if m and (int(m.group(1)) >= self.n_layers or int(m.group(2)) >= self.n_expert or m.group(3) not in known):
                raise S.ConvertError(f"unexpected expert tensor {k} (the model has {self.n_layers} layers of "
                                     f"{self.n_expert} experts)")
        self.entries, self.csf = [], 0
        for l in range(self.n_layers):
            layer = []
            for e in range(self.n_expert):
                ex = []
                for r in self.ROLES:
                    p = f"model.language_model.layers.{l}.mlp.experts.{e}.{r}."
                    rows, cols = shapes[r]
                    ck.need(p + "weight", "U8", (rows, cols // 2))
                    fixed = p + "weight_scale.nvfp4_csf_fixed"
                    if fixed in ck.t:      # a checkpoint may keep a few projections' scales plain
                        ck.need(fixed, "U8", (rows // 16, 16 * (1 + cols // 32)))
                        exc = p + "weight_scale.nvfp4_csf_exceptions"
                        ck.need(exc, "U32")
                        if len(ck.t[exc][3]) != 1:    # the word count is the shape's first entry
                            raise S.ConvertError(f"{exc} is {ck.t[exc][2]} {ck.t[exc][3]}, expected a 1-D array of words")
                        sn, xn = fixed, exc
                        self.csf += 1
                    else:
                        ck.need(p + "weight_scale", "F8_E4M3", (rows, cols // 16))
                        sn, xn = p + "weight_scale", None
                    ck.need(p + "weight_scale_2", "F32", ())
                    w, s, t = ck.t[p + "weight"], ck.t[sn], ck.t[p + "weight_scale_2"]
                    x = None if xn is None else (ck.t[xn][0], ck.t[xn][1], ck.t[xn][3][0])
                    ex.append(((w[0], w[1], w[3]), (s[0], s[1], s[3]), x, (t[0], t[1])))
                layer.append(ex)
            self.entries.append(layer)
        self.layout = [(l, NVFP4, NVFP4, l * self.blob * self.n_expert, self.blob) for l in range(self.n_layers)]
        self.total = self.n_layers * self.n_expert * self.blob

    def layout_text(self) -> bytes:
        return S.nvfp4_layout_text(self.n_expert, self.total, self.layout)

    def jobs(self, group: int):
        for l in range(self.n_layers):
            for s in range(0, self.n_expert, group):
                n = min(group, self.n_expert - s)
                yield self.entries[l][s:s + n], n

    def chunks(self, pool, group: int = 32, depth: int = 32):
        for b in bounded_imap(pool, _expert_job, self.jobs(group), depth):
            if isinstance(b, str):
                raise S.ConvertError(b)
            yield b


# ------------------------------------------------------------------ every other language-model tensor
# How each checkpoint tensor becomes a GGUF tensor: (GGUF name, stored type, transform). "MAT" is a projection
# quantized from BF16 (Q6_K when its rows are whole 256-value blocks and Q6_K is asked for, else Q8_0).
#   id     the values as they are             norm   + 1 (the RMSNorm weights are stored as offsets from 1)
#   heads  rows in the GGUF value-head order  negexp -exp, in the GGUF value-head order (A from A_log)
#   conv   [C, 1, K] -> [C, K]                gconv  conv, value-channel rows in the GGUF order
#   vrows  rows after the q and k parts in the GGUF value-head order;  zrows  all rows in it;  vcols  columns in it
LAYER = {
    "attn_hyper_connection.block_inject_weight.weight": ("hc_attn_inject.weight", "BF16", "id"),
    "attn_hyper_connection.hc_norm.weight": ("hc_attn_norm.weight", "F32", "norm"),
    "attn_hyper_connection.input_mix_weight_down.weight": ("hc_attn_down.weight", "BF16", "id"),
    "attn_hyper_connection.input_mix_weight_up.weight": ("hc_attn_up.weight", "BF16", "id"),
    "mlp_hyper_connection.block_inject_weight.weight": ("hc_ffn_inject.weight", "BF16", "id"),
    "mlp_hyper_connection.hc_norm.weight": ("hc_ffn_norm.weight", "F32", "norm"),
    "mlp_hyper_connection.input_mix_weight_down.weight": ("hc_ffn_down.weight", "BF16", "id"),
    "mlp_hyper_connection.input_mix_weight_up.weight": ("hc_ffn_up.weight", "BF16", "id"),
    "linear_attn.A_log": ("ssm_a", "F32", "negexp"),
    "linear_attn.conv1d.weight": ("ssm_conv1d.weight", "F32", "gconv"),
    "linear_attn.dt_bias": ("ssm_dt.bias", "F32", "heads"),
    "linear_attn.in_proj_a.weight": ("ssm_alpha.weight", "BF16", "heads"),
    "linear_attn.in_proj_b.weight": ("ssm_beta.weight", "BF16", "heads"),
    "linear_attn.in_proj_qkv.weight": ("attn_qkv.weight", "MAT", "vrows"),
    "linear_attn.in_proj_z.weight": ("attn_gate.weight", "MAT", "zrows"),
    "linear_attn.norm.weight": ("ssm_norm.weight", "F32", "id"),
    "linear_attn.out_proj.weight": ("ssm_out.weight", "MAT", "vcols"),
    "mlp.gate.weight": ("ffn_gate_inp.weight", "BF16", "id"),
    "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp.weight", "MAT", "id"),
    "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp.weight", "MAT", "id"),
    "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp.weight", "MAT", "id"),
    "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", "BF16", "id"),
    "ple.conv1d.weight": ("ple_conv1d.weight", "F16", "conv"),
    "ple.key_proj.weight": ("ple_key.weight", "BF16", "id"),
    "ple.value_proj.weight": ("ple_value.weight", "BF16", "id"),
    "ple.norm_conv.weight": ("ple_norm_conv.weight", "F32", "norm"),
    "ple.norm_key.weight": ("ple_norm_key.weight", "F32", "norm"),
    "ple.norm_query.weight": ("ple_norm_query.weight", "F32", "norm"),
    "self_attn.indexer.k_layernorm.weight": ("indexer.k_norm.weight", "F32", "norm"),
    "self_attn.indexer.q_layernorm.weight": ("indexer.q_norm.weight", "F32", "norm"),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", "F32", "norm"),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "F32", "norm"),
    "self_attn.k_proj.weight": ("attn_k.weight", "MAT", "id"),
    "self_attn.o_proj.weight": ("attn_output.weight", "MAT", "id"),
    "self_attn.q_proj.weight": ("attn_q.weight", "MAT", "id"),
    "self_attn.v_proj.weight": ("attn_v.weight", "MAT", "id"),
}
# the indexer's fused projection: its first rows are q (heads x head width), the rest k
INDEXER_QK = "self_attn.indexer.index_qk_proj.weight"
GLOBAL = {
    "lm_head.weight": ("output.weight", "Q8_0", "id"),
    "model.language_model.embed_tokens.weight": ("token_embd.weight", "Q8_0", "id"),
    "model.language_model.hyper_connection_mixer.hc_norm.weight": ("output_hc_norm.weight", "F32", "norm"),
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": ("output_hc_down.weight", "BF16", "id"),
    "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": ("output_hc_up.weight", "BF16", "id"),
}
# read for the metadata or written elsewhere (experts, the n-gram table); the drafter and vision come from the base pack
SKIP = re.compile(r"^(mtp\.|model\.visual\.|model\.language_model\.layers\.\d+\.(mlp\.experts\.|ple\.ple_embedding\.))")
METHODS = {
    "id": "as stored",
    "norm": "bf16 + 1 in f32 (exact)",
    "heads": "value-head rows in the GGUF order",
    "negexp": "-exp(A_log), computed in f64 and rounded once to f32, value heads in the GGUF order",
    "conv": "[C, 1, K] -> [C, K]",
    "gconv": "[C, 1, K] -> [C, K], value channels in the GGUF head order",
    "vrows": "rows after q and k in the GGUF value-head order",
    "zrows": "rows in the GGUF value-head order",
    "vcols": "columns in the GGUF value-head order",
    "split": "the fused indexer projection split into q and k rows",
}


class Tensor:
    """One GGUF tensor made from a checkpoint tensor: `ne` is the GGUF shape (fastest dimension first)."""

    def __init__(self, name, type_name, src, how, ne, rows=None, scale=None):
        self.name, self.type_name, self.src, self.how, self.ne, self.rows = name, type_name, src, how, ne, rows
        self.scale = scale          # the MXFP8 UE8M0 scale tensor of `src`, or None when `src` is BF16
        self.type_id = TYPE_ID[type_name]
        self.shape = ne

    def expected_bytes(self) -> int:
        n = int(np.prod(self.ne))
        return {"F32": 4 * n, "F16": 2 * n, "BF16": 2 * n, "Q8_0": n // 32 * 34, "Q6_K": n // 256 * 210}[self.type_name]


def _index(how: str, n: int, first: int) -> np.ndarray | None:
    if how in ("heads", "negexp"):
        return S.gdn_block_source()
    if how in ("vrows", "gconv"):
        return S.gdn_index(n, first)
    if how in ("zrows", "vcols"):
        return S.gdn_index(n, 0)
    return None


class CheckpointDense:
    """Every language-model tensor but the routed experts and the n-gram table, in the checkpoint's order, with
    `bytes(name)` the stored bytes of a float tensor and `chunks(name, pool)` the streamed bytes of a quantized one.
    `where` and `bytes` are what model_sections.Dense reads."""

    def __init__(self, ck: Checkpoint, q6k: bool = True, require_complete: bool = True):
        tc = ck.text
        self.ck, self.q6k = ck, q6k
        if int(tc["linear_num_key_heads"]) != S.GDN_K_HEADS or int(tc["linear_num_value_heads"]) != S.GDN_V_HEADS \
                or int(tc["linear_key_head_dim"]) != S.GDN_HEAD or int(tc["linear_value_head_dim"]) != S.GDN_HEAD:
            raise S.ConvertError("the GDN head geometry differs from the GGUF head order this converter knows")
        self.qk_rows = int(tc["indexer_n_heads"]) * int(tc["indexer_head_dim"])
        self.tensors: dict[str, Tensor] = {}
        unknown = []
        # an MXFP8 weight is a pair: the E4M3 values and their UE8M0 scales, which the weight's rule consumes
        consumed = {k + "_scale" for k in ck.order if ck.t[k][2] == "F8_E4M3" and not SKIP.match(k)}
        for k in ck.order:
            _, _, dt, shape, _ = ck.t[k]
            if SKIP.match(k) or k in consumed:
                continue
            m = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", k)
            if m and m.group(2) == INDEXER_QK:
                if dt not in ("BF16", "F8_E4M3"):
                    raise S.ConvertError(f"{k} is {dt}; the indexer projection is read from BF16, or from MXFP8")
                rows = shape[0] - self.qk_rows
                sc = self._mxfp8_scale(k, dt, shape)
                for nm, r in (("indexer.q_proj.weight", (0, self.qk_rows)), ("indexer.k_proj.weight", (self.qk_rows, shape[0]))):
                    self._add(f"blk.{m.group(1)}.{nm}", "BF16", k, "split", [shape[1], r[1] - r[0]], r, sc)
                continue
            rule = LAYER.get(m.group(2)) if m else GLOBAL.get(k)
            if rule is None:
                unknown.append(k)
                continue
            if dt == "F8_E4M3" and rule[1] == "F32":
                raise S.ConvertError(f"{k} is MXFP8; the engine reads this tensor as raw F32, which has no scale")
            if dt not in ("BF16", "F8_E4M3") and not (dt == "F32" and rule[1] == "F32"):
                raise S.ConvertError(f"{k} is {dt}; a dense tensor is read from BF16, or from F32 where the engine "
                                     "reads F32")
            name = f"blk.{m.group(1)}.{rule[0]}" if m else rule[0]
            ty, how = rule[1], rule[2]
            if how in ("conv", "gconv"):
                if len(shape) != 3 or shape[1] != 1:
                    raise S.ConvertError(f"{k}: a conv weight of shape {shape}")
                ne = [shape[2], shape[0]]
            else:
                ne = list(reversed(shape))
            if ty == "MAT":
                ty = "Q6_K" if q6k and ne[0] % S.Q6K_BLOCK == 0 else "Q8_0"
            if ty in ("Q8_0", "Q6_K") and (len(ne) != 2 or ne[0] % 32):
                raise S.ConvertError(f"{k}: rows of {ne[0]} values cannot be {ty}")
            self._add(name, ty, k, how, ne, scale=self._mxfp8_scale(k, dt, shape))
        if unknown:
            raise S.ConvertError("checkpoint tensors with no GGUF counterpart: " + ", ".join(unknown[:8])
                                 + (f" ... and {len(unknown) - 8} more" if len(unknown) > 8 else ""))
        if require_complete:
            self._check_complete(tc)
        self.where = {n: (None, t, ck.t[t.src][0]) for n, t in self.tensors.items()}

    def _check_complete(self, tc):
        """Refuses a checkpoint that lacks a tensor the engine reads (each layer's by its type) or has a layer the
        config does not."""
        n_layers, kinds = int(tc["num_hidden_layers"]), tc["layer_types"]
        for name in self.tensors:
            m = re.match(r"blk\.(\d+)\.", name)
            if m and int(m.group(1)) >= n_layers:
                raise S.ConvertError(f"{self.tensors[name].src}: the config has {n_layers} layers")
        need = [rule[0] for rule in GLOBAL.values()]
        ple_layers = {int(m.group(1)) for name in self.tensors for m in [re.match(r"blk\.(\d+)\.ple_", name)] if m}
        for l in range(n_layers):
            if kinds[l] not in ("linear_attention", "full_attention"):
                raise S.ConvertError(f"config.json: layer {l} is {kinds[l]}")
            for key, rule in LAYER.items():
                part = key.split(".")[0]
                if part == "linear_attn" and kinds[l] != "linear_attention" \
                        or part == "self_attn" and kinds[l] != "full_attention" \
                        or part == "ple" and l not in ple_layers:
                    continue
                need.append(f"blk.{l}.{rule[0]}")
            if kinds[l] == "full_attention":
                need += [f"blk.{l}.indexer.q_proj.weight", f"blk.{l}.indexer.k_proj.weight"]
        missing = [n for n in need if n not in self.tensors]
        if missing:
            raise S.ConvertError(f"the checkpoint lacks {len(missing)} tensors the engine reads: " + ", ".join(missing[:8])
                                 + (f" ... and {len(missing) - 8} more" if len(missing) > 8 else ""))

    def _add(self, name, ty, src, how, ne, rows=None, scale=None):
        if name in self.tensors:
            raise S.ConvertError(f"two checkpoint tensors map to {name}")
        self.tensors[name] = Tensor(name, ty, src, how, ne, rows, scale)

    def _mxfp8_scale(self, name, dt, shape):
        """The UE8M0 scale tensor that belongs to an MXFP8 weight (one byte per 32 values); None for a weight that is
        BF16, or F32 where the engine reads F32 - the caller's type gate has already refused anything else."""
        if dt != "F8_E4M3":
            return None
        if len(shape) != 2 or shape[1] % 32:
            raise S.ConvertError(f"{name}: MXFP8 rows of {shape} are not whole 32-value blocks")
        self.ck.need(name + "_scale", "U8", (shape[0], shape[1] // 32))
        return name + "_scale"

    def values(self, name) -> np.ndarray:
        """A float tensor's values in the GGUF order, as float32 [ne1, ne0]."""
        t = self.tensors[name]
        x = mxfp8_values(self.ck, t.src, t.scale) if t.scale else self.ck.values(t.src)
        if t.how in ("conv", "gconv"):
            x = x[:, 0, :]
        if t.how == "split":
            x = x[t.rows[0]:t.rows[1]]
        if t.how == "norm":
            x = x + np.float32(1)
        if t.how == "negexp":
            x = (-np.exp(x.astype(np.float64))).astype(np.float32)
        idx = _index(t.how, x.shape[0], 2 * S.GDN_K_HEADS * S.GDN_HEAD)
        if idx is not None:
            x = x[idx]
        return np.ascontiguousarray(x, dtype=np.float32)

    def bytes(self, name) -> np.ndarray:
        t = self.tensors[name]
        if t.type_name not in ("F32", "F16", "BF16"):
            raise S.ConvertError(f"{name} is {t.type_name}; read it with chunks()")
        x = self.values(name)
        if not np.isfinite(x).all():
            raise S.ConvertError(f"{name} has a non-finite value")
        if t.type_name == "F32":
            out = x.astype("<f4")
        elif t.type_name == "F16":
            try:
                out = S.f16_values(x, name).astype("<f2")
            except ValueError as e:
                raise S.ConvertError(str(e)) from None
        else:
            u = x.view(np.uint32)
            if np.count_nonzero(u & 0xFFFF):
                raise S.ConvertError(f"{name}: a value that is not BF16")
            out = (u >> 16).astype("<u2")
        return np.frombuffer(out.tobytes(), dtype=np.uint8)

    def jobs(self, name):
        t = self.tensors[name]
        p, off, _, shape, _ = self.ck.t[t.src]
        ne0, ne1 = t.ne
        step = max(1, VALUES_PER_JOB // ne0)
        kind = {"vrows": "rows", "zrows": "rows", "vcols": "cols"}.get(t.how)
        sp = None if t.scale is None else (str(self.ck.t[t.scale][0]), self.ck.t[t.scale][1])
        for r in range(0, ne1, step):
            yield (str(p), off, ne0, ne1, r, min(step, ne1 - r), kind,
                   2 * S.GDN_K_HEADS * S.GDN_HEAD if t.how == "vrows" else 0, t.type_name, sp)

    def chunks(self, name, pool, depth: int = 64):
        fn = _mxfp8_quant_job if self.tensors[name].scale else _quant_job
        for b in bounded_imap(pool, fn, self.jobs(name), depth):
            if isinstance(b, str):
                raise S.ConvertError(f"{name}: {b}")
            yield b


def _quant_job(args):
    """Rows [row0, row0 + rows) of a BF16 matrix [ne1, ne0] (rows or columns in the GGUF order) as Q6_K or Q8_0."""
    path, off, ne0, ne1, row0, rows, kind, first, ty, _ = args
    mm = np.memmap(path, dtype="<u2", mode="r", offset=off, shape=(ne1, ne0))
    if kind == "rows":
        u = mm[S.gdn_index(ne1, first)[row0:row0 + rows]]
    elif kind == "cols":
        u = mm[row0:row0 + rows][:, S.gdn_index(ne0, 0)]
    else:
        u = mm[row0:row0 + rows]
    x = S._bf16_to_f32(np.ascontiguousarray(u)).reshape(-1)
    if not np.isfinite(x).all():
        return "a non-finite value"
    return S.q6_k(x) if ty == "Q6_K" else S.q8_0(x)


def mxfp8_dequant(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """MXFP8 as float32: E4M3 values times the UE8M0 scale byte of their 32-value block. Exact for every finite
    value, since a 4-bit significand times a power of two is a BF16 number."""
    if (scales == 255).any():
        raise S.ConvertError("an UE8M0 scale of 255 (NaN)")
    e = scales.astype(np.int16) - 127
    return S.fp8_e4m3_table()[values] * np.float32(2.0) ** e.repeat(32, axis=e.ndim - 1)


def mxfp8_values(ck: Checkpoint, name: str, scale: str) -> np.ndarray:
    """An MXFP8 checkpoint tensor (its E4M3 values and their UE8M0 scales) as shaped float32."""
    ck.need(name, "F8_E4M3")
    shape = ck.t[name][3]
    ck.need(scale, "U8", (shape[0], shape[1] // 32))
    v = ck.raw(name).reshape(shape)
    s = ck.raw(scale).reshape(shape[0], shape[1] // 32)
    return mxfp8_dequant(v, s)


def _mxfp8_quant_job(args):
    """Rows [row0, row0 + rows) of an MXFP8 matrix [ne1, ne0] (rows or columns in the GGUF order) as Q6_K or Q8_0:
    dequantized exactly, then quantized once, so the pack holds one rounding of these weights."""
    path, off, ne0, ne1, row0, rows, kind, first, ty, spair = args
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=off, shape=(ne1, ne0))
    ms = np.memmap(spair[0], dtype=np.uint8, mode="r", offset=spair[1], shape=(ne1, ne0 // 32))
    if kind == "rows":
        sel = S.gdn_index(ne1, first)[row0:row0 + rows]
        v, s = np.ascontiguousarray(mm[sel]), np.ascontiguousarray(ms[sel])
    else:
        v = np.ascontiguousarray(mm[row0:row0 + rows])
        s = np.ascontiguousarray(ms[row0:row0 + rows])
    try:
        x = mxfp8_dequant(v, s)
    except S.ConvertError as e:
        return str(e)
    if kind == "cols":
        x = x[:, S.gdn_index(ne0, 0)]
    x = np.ascontiguousarray(x).reshape(-1)
    if not np.isfinite(x).all():
        return "a non-finite value"
    return S.q6_k(x) if ty == "Q6_K" else S.q8_0(x)


# ------------------------------------------------------------------ metadata and tokenizer
def front_matter(readme: pathlib.Path) -> dict:
    """The simple `key: value` / `key:\\n- item` YAML block at the top of a model card."""
    out, key = {}, None
    if not readme.is_file():
        return out
    lines = readme.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return out
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line.startswith("- ") and key:
            out.setdefault(key, []).append(line[2:].strip())
        elif ":" in line:
            key, v = (s.strip() for s in line.split(":", 1))
            if v:
                out[key] = v
    return out


def f32(v) -> float:
    return float(np.float32(v))


def tokenizer_kv(ck: Checkpoint) -> dict:
    """The tokenizer as GGUF metadata (llama.cpp's gpt2 conversion): ids without a token become `[PADn]` (unused),
    added tokens are control tokens when marked special or written `<|...|>`, other added tokens user-defined."""
    d = ck.dir
    tj = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    tcfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    model = tj.get("model", {})
    if model.get("type") != "BPE":
        raise S.ConvertError(f"tokenizer.json: a {model.get('type')} model; only byte-level BPE is converted")
    pre = tj.get("pre_tokenizer") or {}
    splits = [p.get("pattern", {}).get("Regex") for p in pre.get("pretokenizers", []) if p.get("type") == "Split"]
    if splits != [QWEN35_HF_SPLIT]:
        raise S.ConvertError("tokenizer.json: the pre-tokenizer is not the qwen35 split")
    n_vocab = int(ck.text["vocab_size"])
    rev = {int(i): t for t, i in model["vocab"].items()}
    added = {int(a["id"]): a for a in tj.get("added_tokens", [])}
    for i, a in added.items():
        rev[i] = a["content"]
    if max(rev) >= n_vocab:
        raise S.ConvertError(f"tokenizer.json: id {max(rev)} is beyond the vocabulary of {n_vocab}")
    tokens, types = [], []
    for i in range(n_vocab):
        if i not in rev:
            tokens.append(f"[PAD{i}]")
            types.append(UNUSED)
            continue
        t = rev[i]
        tokens.append(t)
        if i in added:
            special = added[i].get("special") or (t.startswith("<|") and t.endswith("|>"))
            types.append(CONTROL if special else USER_DEFINED)
        else:
            types.append(NORMAL)
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    ids = {t: i for i, t in enumerate(tokens)}

    def token_id(key):
        v = tcfg.get(key)
        v = v.get("content") if isinstance(v, dict) else v
        return ids.get(v) if v else None

    gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8")) if (d / "generation_config.json").is_file() else {}
    kv = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen35", "tokenizer.ggml.tokens": tokens,
          "tokenizer.ggml.token_type": types, "tokenizer.ggml.merges": merges}
    eos, pad = token_id("eos_token"), token_id("pad_token")
    bos = token_id("bos_token")
    if bos is None:
        bos = gen.get("bos_token_id", ck.text.get("bos_token_id"))
    for k, v in (("eos", eos), ("padding", pad), ("bos", bos)):
        if v is not None:
            kv[f"tokenizer.ggml.{k}_token_id"] = int(v)
    kv["tokenizer.ggml.add_bos_token"] = bool(tcfg.get("add_bos_token", False))
    tpl = d / "chat_template.jinja"
    template = tpl.read_text(encoding="utf-8") if tpl.is_file() else tcfg.get("chat_template")
    if template:
        kv["tokenizer.chat_template"] = template
    return kv


def model_kv(ck: Checkpoint, ple_dim: int, file_type: int, name: str | None) -> dict:
    """The GGUF metadata the engine reads, from config.json, generation_config.json and the model card."""
    c, tc = ck.config, ck.text
    if tc.get("model_type") != "qwen4_exp_text":
        raise S.ConvertError(f"config.json: model_type {tc.get('model_type')}, expected qwen4_exp_text")
    A = "qwen4exp."
    d = ck.dir
    gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8")) if (d / "generation_config.json").is_file() else {}
    card = front_matter(d / "README.md")
    kinds = tc["layer_types"]
    rope = tc.get("rope_parameters", {})
    head = int(tc["head_dim"])
    ple_layers = sorted({int(m.group(1)) for k in ck.t for m in [re.match(r"model\.language_model\.layers\.(\d+)\.ple\.key_proj\.weight$", k)] if m})
    if len(ple_layers) != 1:
        raise S.ConvertError(f"the checkpoint has PLE weights in layers {ple_layers}")
    pe = f"model.language_model.layers.{ple_layers[0]}.ple.ple_embedding."
    kv = {"general.architecture": "qwen4exp", "general.type": "model"}
    if "top_k" in gen:
        kv["general.sampling.top_k"] = int(gen["top_k"])
    if "top_p" in gen:
        kv["general.sampling.top_p"] = f32(gen["top_p"])
    if "temperature" in gen:
        kv["general.sampling.temp"] = f32(gen["temperature"])
    if name:
        kv["general.name"] = name
    if "license" in card:
        kv["general.license"] = card["license"]
    if "license_name" in card:
        kv["general.license.name"] = card["license_name"]
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
    tags = list(card.get("tags") or [])
    if card.get("pipeline_tag") and card["pipeline_tag"] not in tags:
        tags.append(card["pipeline_tag"])
    if tags:
        kv["general.tags"] = tags
    kv.update({
        A + "block_count": int(tc["num_hidden_layers"]),
        A + "context_length": int(tc["max_position_embeddings"]),
        A + "embedding_length": int(tc["hidden_size"]),
        A + "attention.head_count": int(tc["num_attention_heads"]),
        A + "attention.head_count_kv": int(tc["num_key_value_heads"]),
        A + "rope.dimension_sections": (list(rope.get("mrope_section", [])) + [0, 0, 0, 0])[:4],
        A + "rope.freq_base": f32(rope.get("rope_theta", tc.get("rope_theta"))),
        A + "attention.layer_norm_rms_epsilon": f32(tc["rms_norm_eps"]),
        A + "expert_count": int(tc["num_experts"]),
        A + "expert_used_count": int(tc["num_experts_per_tok"]),
        A + "attention.key_length": head,
        A + "attention.value_length": head,
        "general.file_type": file_type,
        A + "expert_feed_forward_length": int(tc["moe_intermediate_size"]),
        A + "expert_shared_feed_forward_length": int(tc["shared_expert_intermediate_size"]),
        A + "ssm.conv_kernel": int(tc["linear_conv_kernel_dim"]),
        A + "ssm.state_size": int(tc["linear_key_head_dim"]),
        A + "ssm.group_count": int(tc["linear_num_key_heads"]),
        A + "ssm.time_step_rank": int(tc["linear_num_value_heads"]),
        A + "ssm.inner_size": int(tc["linear_num_value_heads"]) * int(tc["linear_value_head_dim"]),
        A + "attention.recurrent_layers": [k == "linear_attention" for k in kinds],
        A + "full_attention_interval": int(tc["full_attention_interval"]),
        A + "rope.dimension_count": int(head * float(rope.get("partial_rotary_factor", tc.get("partial_rotary_factor")))),
        A + "hyper_connection.count": int(tc["hc_count"]),
        A + "hyper_connection.low_rank": int(tc["hc_lowrank"]),
        A + "attention.indexer.head_count": int(tc["indexer_n_heads"]),
        A + "attention.indexer.key_length": int(tc["indexer_head_dim"]),
        A + "attention.indexer.top_k": int(tc["indexer_budget"]),
        A + "attention.compress_ratios": [int(tc["indexer_compress_ratio"]) if k == "full_attention" else 0 for k in kinds],
        A + "ple.layers": ple_layers,
        A + "ple.ngram_size": int(tc["ngram_size"]),
        A + "ple.heads_per_ngram": int(tc["heads_per_ngram"]),
        A + "ple.conv_kernel": int(tc["ple_conv_kernel_size"]),
        A + "ple.eos_token_id": int(tc["eos_token_id"]),
        A + "ple.image_token_id": int(c["image_token_id"]),
        A + "embedding_length_per_layer_input": ple_dim,
        A + "ple.layer_multipliers": [int(v) for v in ck.values(pe + "layer_multipliers")],
        A + "ple.head_offsets": [int(v) for v in ck.values(pe + "ngram_heads_offsets")],
        A + "ple.head_vocab_sizes": [int(v) for v in ck.values(pe + "ngram_heads_vocab_sizes")],
        "general.quantization_version": 2,
    })
    kv.update(tokenizer_kv(ck))
    return kv


def ple_nvfp4(ck: Checkpoint, stem: str) -> S.PleNvfp4:
    """The n-gram table whose tensors begin with `stem`, stored as NVFP4: `shard_K.weight` E2M1 codes with
    `shard_K.weight_scale` UE4M3 per 16 values and one F32 `weight_scale_2` for the table."""
    ck.need(stem + "weight_scale_2", "F32", (1,))
    scale = float(ck.raw(stem + "weight_scale_2").view("<f4")[0])
    shards = []
    while True:
        c, s = f"{stem}shard_{len(shards)}.weight", f"{stem}shard_{len(shards)}.weight_scale"
        if c not in ck.t:
            break
        if ck.t[c][2] != "U8" or len(ck.t[c][3]) != 2 or ck.t[c][3][1] != S.PLE_DIM // 2:
            raise S.ConvertError(f"{c}: expected uint8 [rows, {S.PLE_DIM // 2}] of E2M1 codes")
        if ck.t[s][2] != "F8_E4M3" or ck.t[s][3] != [ck.t[c][3][0], S.PLE_DIM // 16]:
            raise S.ConvertError(f"{s}: expected one UE4M3 scale per 16 values of {c}")
        shards.append((ck.t[c][0], ck.t[c][1], ck.t[s][0], ck.t[s][1], ck.t[c][3][0]))
    if not shards:
        raise S.ConvertError(f"{ck.dir}: no ngram_embedding shards beside {stem}")
    return S.PleNvfp4(shards, scale)


def ple_table(ck: Checkpoint):
    """The checkpoint's n-gram table, in whichever of the two forms a checkpoint keeps it: FP8 shards with one BF16
    scale (requantized to Basalt's Q8_0 rows), or NVFP4 shards with UE4M3 scales and one F32 scale (carried over as
    Basalt's 90-byte NVFP4 rows, which is what they are)."""
    stems = {k[: -len("weight_scale_2")] for k in ck.t if k.endswith("ngram_embedding.weight_scale_2")}
    if len(stems) > 1:
        raise S.ConvertError(f"{ck.dir}: {len(stems)} ngram_embedding.weight_scale_2 tensors")
    if stems:
        return ple_nvfp4(ck, stems.pop())
    hits = sorted({ck.t[k][0] for k in ck.t if k.endswith("ngram_embedding.weight_scale")})
    if len(hits) != 1:
        raise S.ConvertError(f"{ck.dir}: {len(hits)} files hold an ngram_embedding.weight_scale")
    return S.PleFp8(hits[0])
