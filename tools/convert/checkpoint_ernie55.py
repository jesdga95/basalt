"""An ernie55 pack (Baidu's ERNIE-5.5-Flash: hybrid HCA / MLA-DSA, mHC, latent MXFP4 MoE) straight from the official
Hugging Face safetensors checkpoint (`baidu/ERNIE-5.5-Flash-Preview`), for tools/convert/convert.py.

  experts             the routed experts of MoE layers (l >= first_k_dense_replace), [w1 (gate) | w3 (up) | w2 (down)]
                      packed bit-exact as ggml MXFP4 (type 39: 17-byte blocks of 1 UE8M0 scale + 16 split-half nibble
                      bytes per 32 values) from the checkpoint's I8 paired nibbles and F8_E8M0 scales
  experts.layout      one `layer 39 39 offset blob_bytes` line per MoE layer
  gguf                every other language-model tensor under Basalt's ernie55 names:
                        - 128 x 128 blockwise FP8 projections -> exact f32 dequant -> Q8_0 (ggml reference rounding)
                        - token_embd.weight and output.weight kept as BF16
                        - compressor projections and indexer wk / weights_proj / flash_gate kept as BF16
                        - norms, mHC mixers (transposed to [out, in]) / biases / stacked [pre, post, res] scales,
                          VHA PostMix (folded I + V @ U^T), attention sinks, compressor APE, router [w0; w1],
                          router bias & per-expert scale, and lm_head multimax params as F32
  tokenizer/*, meta   from config.json, generation_config.json, tokenizer.json, tokenizer_config.json and README.md

Optional FlashIndexer tensors (`flashindexer/flashindexer.safetensors`) are included automatically when present.
Needs numpy.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
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
from model_file import TENSOR_ALIGN, ModelWriter, Stream  # noqa: E402

ARCH = "ernie55"
MXFP4 = 39
TYPE_ID = {"F32": 0, "Q8_0": 8, "BF16": 30, "MXFP4": 39}
ROLES = ("w1", "w3", "w2")   # gate, up, down
EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.(weight|weight_scale_inv)$")
MTP_RE = re.compile(r"^model\.mtp\.")
FLASH_RE = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\.core_attention\.indexer\.(linear_gate\.weight|flashindexer_bias)$"
)
EXPERTS_PER_JOB = 16
ROWS_PER_JOB = 1 << 22
E4M3 = S.fp8_e4m3_table()
MXFP4_VALUES = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)

# Checkpoint tensor (without `model.layers.N.`) -> (GGUF name after `blk.N.`, target form)
# Forms:
#   "FP8_Q8"  128x128 blockwise FP8 (pair with *_scale_inv) -> Q8_0
#   "BF16"    BF16 as stored
#   "F32"     F32 as stored or BF16 widened to F32
#   "F32_T"   F32 transposed [in, out] -> [out, in] (mHC mapping_proj)
LAYER_COMMON = {
    "input_layernorm.weight": ("attn_norm.weight", "F32"),
    "post_attention_layernorm.weight": ("ffn_norm.weight", "F32"),
    "self_attention_hyper_connection.mapping_proj.weight": ("hc_attn_fn.weight", "F32_T"),
    "self_attention_hyper_connection.bias": ("hc_attn_base.weight", "F32"),
    "mlp_hyper_connection.mapping_proj.weight": ("hc_ffn_fn.weight", "F32_T"),
    "mlp_hyper_connection.bias": ("hc_ffn_base.weight", "F32"),
    "self_attn.gate_proj.weight": ("attn_gate.weight", "FP8_Q8"),
}

LAYER_HCA = {
    "self_attn.linear_q_down_proj.weight": ("attn_q_a.weight", "FP8_Q8"),
    "self_attn.q_layernorm.weight": ("attn_q_a_norm.weight", "F32"),
    "self_attn.linear_q_up_proj.weight": ("attn_q_b.weight", "FP8_Q8"),
    "self_attn.linear_kv_proj.weight": ("attn_kv.weight", "FP8_Q8"),
    "self_attn.kv_layernorm.weight": ("attn_kv_a_norm.weight", "F32"),
    "self_attn.linear_o_group_proj": ("attn_output_a.weight", "FP8_Q8"),
    "self_attn.o_proj.weight": ("attn_output_b.weight", "FP8_Q8"),
    "self_attn.core_attention.attn_sink": ("attn_sinks.weight", "F32"),
    "self_attn.core_attention.compressor.ape": ("attn_compressor_ape.weight", "F32"),
    "self_attn.core_attention.compressor.linear_wkv.weight": ("attn_compressor_kv.weight", "BF16"),
    "self_attn.core_attention.compressor.linear_wgate.weight": ("attn_compressor_gate.weight", "BF16"),
    "self_attn.core_attention.compressor.norm.weight": ("attn_compressor_norm.weight", "F32"),
}

LAYER_DSA = {
    "self_attn.q_a_proj.weight": ("attn_q_a.weight", "FP8_Q8"),
    "self_attn.q_a_layernorm.weight": ("attn_q_a_norm.weight", "F32"),
    "self_attn.q_b_proj.weight": ("attn_q_b.weight", "FP8_Q8"),
    "self_attn.kv_a_proj_with_mqa.weight": ("attn_kv_a_mqa.weight", "FP8_Q8"),
    "self_attn.kv_a_layernorm.weight": ("attn_kv_a_norm.weight", "F32"),
    "self_attn.kv_b_proj.weight": ("attn_kv_b.weight", "FP8_Q8"),
    "self_attn.o_proj.weight": ("attn_output.weight", "FP8_Q8"),
    "self_attn.core_attention.softmax_offset": ("attn_sinks.weight", "F32"),
    "self_attn.core_attention.indexer.wq_b.weight": ("indexer.attn_q_b.weight", "FP8_Q8"),
    "self_attn.core_attention.indexer.wk.weight": ("indexer.attn_k.weight", "BF16"),
    "self_attn.core_attention.indexer.k_norm.weight": ("indexer.k_norm.weight", "F32"),
    "self_attn.core_attention.indexer.k_norm.bias": ("indexer.k_norm.bias", "F32"),
    "self_attn.core_attention.indexer.weights_proj.weight": ("indexer.proj.weight", "BF16"),
}

LAYER_DENSE_MLP = {
    "mlp.gate_proj.weight": ("ffn_gate.weight", "FP8_Q8"),
    "mlp.up_proj.weight": ("ffn_up.weight", "FP8_Q8"),
    "mlp.down_proj.weight": ("ffn_down.weight", "FP8_Q8"),
}

LAYER_MOE = {
    "block_sparse_moe.e_score_correction_bias": ("exp_probs_b.bias", "F32"),
    "block_sparse_moe.gate.routed_scaling_factor_param": ("exp_probs_scale.weight", "F32"),
    "block_sparse_moe.fc1_latent_proj.weight": ("ffn_routed_down.weight", "FP8_Q8"),
    "block_sparse_moe.latent_norm.weight": ("ffn_latent_norm.weight", "F32"),
    "block_sparse_moe.fc2_latent_proj.weight": ("ffn_routed_up.weight", "FP8_Q8"),
    "block_sparse_moe.shared_experts.w1.weight": ("ffn_gate_shexp.weight", "FP8_Q8"),
    "block_sparse_moe.shared_experts.w3.weight": ("ffn_up_shexp.weight", "FP8_Q8"),
    "block_sparse_moe.shared_experts.w2.weight": ("ffn_down_shexp.weight", "FP8_Q8"),
}

GLOBAL = {
    "model.embed_tokens.weight": ("token_embd.weight", "BF16"),
    "lm_head.weight": ("output.weight", "BF16"),
    "model.norm.weight": ("output_norm.weight", "F32"),
    "model.lm_head.multimax_ranges": ("output_multimax_ranges", "F32"),
    "model.lm_head.multimax_ts": ("output_multimax_ts", "F32"),
    "model.mhc_contract.hc_head_fn": ("hc_head_fn.weight", "F32_T"),
    "model.mhc_contract.hc_head_base": ("hc_head_base.weight", "F32"),
    "model.mhc_contract.hc_head_scale": ("hc_head_scale.weight", "F32"),
}

# Composite per-layer tensors assembled from multiple checkpoint keys
HC_ALPHAS = (
    ("self_attention_hyper_connection", "hc_attn_scale.weight"),
    ("mlp_hyper_connection", "hc_ffn_scale.weight"),
)
CONSUMED_SUFFIXES = frozenset({
    "self_attention_hyper_connection.alpha_pre",
    "self_attention_hyper_connection.alpha_post",
    "self_attention_hyper_connection.alpha_res",
    "mlp_hyper_connection.alpha_pre",
    "mlp_hyper_connection.alpha_post",
    "mlp_hyper_connection.alpha_res",
    "self_attn.vha_postmix_U",
    "self_attn.vha_postmix_V",
    "block_sparse_moe.gate.weight",
    "block_sparse_moe.gate.weight_1",
})

METHODS = {
    "fp8_block_q8_0": "128x128 blockwise E4M3 x FP32 scale_inv (exact in f32), then Q8_0 (ggml reference rounding)",
    "bf16": "BF16 as stored",
    "f32": "F32 as stored or BF16 widened to F32",
    "f32_t": "F32 transposed [in, out] -> [out, in]",
    "hc_scale": "stacked F32 [alpha_pre, alpha_post, alpha_res]",
    "vha_fold": "F32 I + V @ U^T folded from BF16 vha_postmix_V and vha_postmix_U",
    "gate_concat": "F32 [gate.weight; gate.weight_1] stacked for split-feature sigmoid routing",
}


def scale_name_for(weight_name: str) -> str:
    """Block-FP8 companion scale name for `weight_name` (handles both `*.weight` and `linear_o_group_proj`)."""
    return weight_name + "_scale_inv"


def mxfp4_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Repack HF MXFP4 rows (codes [rows, cols // 2] uint8: value 2i in low nibble, 2i+1 in high nibble;
    E8M0 scales [rows, cols // 32] uint8) into ggml MXFP4 (type 39): per 32 values, 1 E8M0 scale byte then
    16 bytes whose low nibble is value j and high nibble is value j + 16."""
    rows, half = codes.shape
    cols = half * 2
    if cols % 32 != 0 or scales.shape != (rows, cols // 32):
        raise S.ConvertError(f"mxfp4_rows: shape mismatch codes={codes.shape} scales={scales.shape}")
    v = np.empty((rows, cols), np.uint8)
    v[:, 0::2] = codes & 0x0F
    v[:, 1::2] = codes >> 4
    v = v.reshape(rows, cols // 32, 2, 16)
    out = np.empty((rows, cols // 32, 17), np.uint8)
    out[..., 0] = scales
    out[..., 1:] = v[:, :, 0, :] | (v[:, :, 1, :] << 4)
    return out.reshape(-1)


def mxfp4_dequant(raw: np.ndarray, cols: int) -> np.ndarray:
    """Dequantize ggml MXFP4 rows (17 bytes / 32 values) to float32 [rows, cols]."""
    b = raw.reshape(-1, cols // 32, 17)
    d = np.ldexp(np.float32(1.0), b[..., 0].astype(np.int32) - 127).astype(np.float32)[..., None]
    q = b[..., 1:]
    v = np.concatenate([MXFP4_VALUES[q & 0x0F], MXFP4_VALUES[q >> 4]], axis=-1) * d
    return v.reshape(-1, cols)


def fp8_block_dequant(codes: np.ndarray, scales: np.ndarray, block: tuple[int, int], row0: int = 0) -> np.ndarray:
    """Dequantize 2-D E4M3 codes [n_rows, cols] using FP32 block scales [ceil(N/bn), ceil(cols/bk)]."""
    bn, bk = block
    v = E4M3[codes]
    if np.isnan(v).any():
        raise S.ConvertError("a NaN E4M3 weight code")
    if not np.isfinite(scales).all():
        raise S.ConvertError("a non-finite FP8 block scale")
    n, c = codes.shape
    b0 = row0 // bn
    skip = row0 - b0 * bn
    nb_r = (skip + n + bn - 1) // bn
    sr = np.repeat(scales[b0:b0 + nb_r], bn, axis=0)[skip:skip + n]
    sc = np.repeat(sr, bk, axis=1)[:, :c]
    return (v * sc).astype(np.float32, copy=False)


def _pread_bytes(path: str, off: int, n: int) -> bytearray:
    buf = bytearray(n)
    mv = memoryview(buf)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        got = 0
        while got < n:
            if hasattr(os, "preadv"):
                k = os.preadv(fd, [mv[got:]], off + got)
            else:
                os.lseek(fd, off + got, os.SEEK_SET)
                k = os.read(fd, n - got)
                mv[got:got + len(k)] = k
                k = len(k)
            if k <= 0:
                raise IOError(f"{path}: short read at {off + got}")
            got += k
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return buf


class ErnieCheckpoint:
    """Safetensors directory reader for ERNIE 5.5 (model-NNNNN-of-NNNNN.safetensors + optional flashindexer/)."""

    def __init__(self, d):
        self.dir = pathlib.Path(d)
        cfg = self.dir / "config.json"
        if not cfg.is_file():
            raise S.ConvertError(f"{cfg} not found")
        self.config = json.loads(cfg.read_text(encoding="utf-8"))
        self.text = self.config
        if self.config.get("model_type") != "ernie5_5_moe":
            raise S.ConvertError(f"config.json: model_type {self.config.get('model_type')!r}, expected 'ernie5_5_moe'")
        self.shards = sorted(p for p in self.dir.glob("*.safetensors")
                             if re.fullmatch(r"model-\d+-of-\d+\.safetensors", p.name))
        if not self.shards:
            raise S.ConvertError(f"{self.dir}: no model-NNNNN-of-NNNNN.safetensors shards")
        self.files = list(self.shards)
        fi = self.dir / "flashindexer" / "flashindexer.safetensors"
        self.has_flashindexer = fi.is_file()
        if self.has_flashindexer:
            self.files.append(fi)
        self.t: dict[str, tuple[pathlib.Path, int, str, list[int], int]] = {}
        self.order: list[str] = []
        for p in self.files:
            head, base = S.read_safetensors_header(p)
            items = sorted(((k, v) for k, v in head.items() if k != "__metadata__"),
                           key=lambda kv: kv[1]["data_offsets"][0])
            for k, v in items:
                if k in self.t:
                    raise S.ConvertError(f"{p.name}: duplicate tensor {k}")
                a, b = v["data_offsets"]
                self.t[k] = (p, base + a, v["dtype"], list(v["shape"]), b - a)
                self.order.append(k)

    def raw(self, name: str) -> np.ndarray:
        p, off, _, _, n = self.t[name]
        return np.frombuffer(_pread_bytes(str(p), off, n), dtype=np.uint8)

    def values(self, name: str) -> np.ndarray:
        _, _, dt, shape, _ = self.t[name]
        raw = self.raw(name)
        if dt == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
        if dt == "BF16":
            u16 = np.frombuffer(raw, dtype="<u2").reshape(shape)
            return S._bf16_to_f32(np.ascontiguousarray(u16))
        raise S.ConvertError(f"{name}: expected F32 or BF16, got {dt}")

    def need(self, name: str, dtype: str | tuple[str, ...], shape: tuple[int, ...] | None = None):
        if name not in self.t:
            raise S.ConvertError(f"missing tensor {name}")
        _, _, dt, sh, _ = self.t[name]
        ok = (dt in dtype) if isinstance(dtype, tuple) else (dt == dtype)
        if not ok:
            raise S.ConvertError(f"{name}: dtype {dt}, expected {dtype}")
        if shape is not None and tuple(sh) != tuple(shape):
            raise S.ConvertError(f"{name}: shape {sh}, expected {list(shape)}")
        return self.t[name]


def _req(d: dict, key: str, where: str = "config.json"):
    if not isinstance(d, dict) or key not in d or d[key] is None:
        raise S.ConvertError(f"{where}: missing required key {key!r}")
    return d[key]


class Experts:
    """The `experts` section: per MoE layer, n_routed_experts blobs of [w1 | w3 | w2] in GGML MXFP4 (type 39)."""

    def __init__(self, ck: ErnieCheckpoint):
        c = ck.config
        qc = _req(c, "quantization_config")
        exp_dtype = _req(qc, "expert_dtype", "config.json.quantization_config")
        if exp_dtype != "mxfp4":
            raise S.ConvertError(f"config.json.quantization_config: expert_dtype is {exp_dtype!r}; expected 'mxfp4'")
        self.ck = ck
        self.n_layers = int(_req(c, "num_hidden_layers"))
        self.first_dense = int(_req(c, "first_k_dense_replace"))
        self.n_expert = int(_req(c, "n_routed_experts"))
        self.latent = int(_req(c, "moe_latent_size"))
        self.ff = int(_req(c, "moe_intermediate_size"))
        if self.latent % 32 or self.ff % 32:
            raise S.ConvertError(f"MXFP4 experts require multiples of 32, got latent={self.latent} ff={self.ff}")
        self.gu_row = self.latent // 32 * 17
        self.d_row = self.ff // 32 * 17
        self.blob_bytes = 2 * self.ff * self.gu_row + self.latent * self.d_row
        self.layers: list[tuple[int, int, int]] = []
        offset = 0
        for l in range(self.first_dense, self.n_layers):
            for e in range(self.n_expert):
                p = f"model.layers.{l}.block_sparse_moe.experts.{e}."
                for role, rows, cols in (("w1", self.ff, self.latent),
                                         ("w3", self.ff, self.latent),
                                         ("w2", self.latent, self.ff)):
                    ck.need(p + f"{role}.weight", ("I8", "U8"), (rows, cols // 2))
                    ck.need(p + f"{role}.weight_scale_inv", ("F8_E8M0", "U8"), (rows, cols // 32))
            self.layers.append((l, offset, self.blob_bytes))
            offset += self.blob_bytes * self.n_expert
        self.total = offset

    def layout_text(self) -> bytes:
        head = (f"# experts v6: layer gu_type d_type offset blob_bytes "
                f"(n_expert {self.n_expert}, total {self.total})\n")
        return (head + "".join(f"{l} {MXFP4} {MXFP4} {off} {blob}\n" for l, off, blob in self.layers)).encode()

    def jobs(self):
        ck = self.ck
        for l, _, _ in self.layers:
            for e0 in range(0, self.n_expert, EXPERTS_PER_JOB):
                batch = []
                for e in range(e0, min(self.n_expert, e0 + EXPERTS_PER_JOB)):
                    p = f"model.layers.{l}.block_sparse_moe.experts.{e}."
                    spec = []
                    for role, rows, cols in (("w1", self.ff, self.latent),
                                             ("w3", self.ff, self.latent),
                                             ("w2", self.latent, self.ff)):
                        wp, woff, _, _, wn = ck.t[p + f"{role}.weight"]
                        sp, soff, _, _, sn = ck.t[p + f"{role}.weight_scale_inv"]
                        spec.append((str(wp), woff, wn, str(sp), soff, sn, rows, cols))
                    batch.append((l, e, spec))
                yield ("experts", batch)

    def chunks(self, pool, depth: int = 8):
        for b in S.bounded_imap(pool, _job, self.jobs(), depth):
            if isinstance(b, str):
                raise S.ConvertError(b)
            yield b


class Tensor:
    def __init__(self, name: str, form: str, src, how: str, ne: list[int], scale_src: str | None = None):
        self.name = name
        self.form = form
        self.src = src
        self.how = how
        self.ne = ne   # GGUF order: innermost first [cols, rows]
        self.scale_src = scale_src
        self.type_name = "Q8_0" if form == "FP8_Q8" else ("F32" if form in ("F32", "F32_T") else form)
        self.type_id = TYPE_ID[self.type_name]

    def expected_bytes(self) -> int:
        n = int(np.prod(self.ne))
        return {"F32": 4 * n, "BF16": 2 * n, "Q8_0": n // 32 * 34}[self.type_name]


class Plan:
    """All non-expert `gguf` tensors of an ernie55 pack."""

    def __init__(self, ck: ErnieCheckpoint):
        c = ck.config
        self.ck = ck
        self.n_layers = int(_req(c, "num_hidden_layers"))
        self.first_dense = int(_req(c, "first_k_dense_replace"))
        self.compress_ratios = [int(x) for x in _req(c, "compress_ratios")]
        if len(self.compress_ratios) != self.n_layers:
            raise S.ConvertError("config.json: len(compress_ratios) != num_hidden_layers")
        qc = _req(c, "quantization_config")
        wb = _req(qc, "weight_block_size", "config.json.quantization_config")
        if not isinstance(wb, list) or len(wb) != 2:
            raise S.ConvertError(f"config.json.quantization_config: weight_block_size={wb!r}, expected [rows, cols]")
        self.fp8_block = (int(wb[0]), int(wb[1]))
        hidden_act = _req(c, "hidden_act")
        if hidden_act != "situ":
            raise S.ConvertError(f"config.json: hidden_act={hidden_act!r}, expected 'situ'")
        if not _req(c, "enable_hyper_connections") or int(_req(c, "hc_mult")) != 4:
            raise S.ConvertError("config.json: expected enable_hyper_connections=true and hc_mult=4")
        if not _req(c, "moe_split_feature_routing"):
            raise S.ConvertError("config.json: expected moe_split_feature_routing=true")
        scoring = _req(c, "scoring_func")
        if scoring != "sigmoid":
            raise S.ConvertError(f"config.json: scoring_func={scoring!r}, expected 'sigmoid'")

        self.tensors: dict[str, Tensor] = {}
        unknown: list[str] = []
        scale_keys: set[str] = set()
        for k in ck.order:
            if EXPERT_RE.match(k) or MTP_RE.match(k):
                continue
            if k.endswith("_scale_inv"):
                scale_keys.add(k)
                continue
            fm = FLASH_RE.match(k)
            if fm:
                l, leaf = int(fm.group(1)), fm.group(2)
                if leaf == "linear_gate.weight":
                    self._add_simple(f"blk.{l}.indexer.flash_gate.weight", "BF16", k)
                else:
                    self._add_simple(f"blk.{l}.indexer.flash_bias.weight", "F32", k)
                continue
            if k in GLOBAL:
                gname, form = GLOBAL[k]
                self._add_simple(gname, form, k)
                continue
            m = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", k)
            if not m:
                unknown.append(k)
                continue
            l, sub = int(m.group(1)), m.group(2)
            if sub in CONSUMED_SUFFIXES:
                continue
            is_dsa = self.compress_ratios[l] <= 0
            rule = LAYER_COMMON.get(sub)
            if rule is None:
                rule = LAYER_DSA.get(sub) if is_dsa else LAYER_HCA.get(sub)
            if rule is None:
                rule = LAYER_DENSE_MLP.get(sub) if l < self.first_dense else LAYER_MOE.get(sub)
            if rule is None:
                unknown.append(k)
                continue
            gname, form = f"blk.{l}.{rule[0]}", rule[1]
            self._add_simple(gname, form, k)

        if unknown:
            raise S.ConvertError("checkpoint tensors with no pack counterpart: " + ", ".join(unknown[:8]))

        # Composite per-layer tensors: hc scales, VHA folded mix, split-feature router
        n_heads_hca = int(c["num_attention_heads"])
        n_heads_dsa = int(c["hybrid_mla_num_attention_heads"])
        h = int(c["hidden_size"])
        n_exp = int(c["n_routed_experts"])
        for l in range(self.n_layers):
            is_dsa = self.compress_ratios[l] <= 0
            for src_pref, dst_name in HC_ALPHAS:
                srcs = tuple(f"model.layers.{l}.{src_pref}.alpha_{a}" for a in ("pre", "post", "res"))
                for s in srcs:
                    ck.need(s, ("F32", "BF16"), (1,))
                self.tensors[f"blk.{l}.{dst_name}"] = Tensor(f"blk.{l}.{dst_name}", "F32", srcs, "hc_scale", [3])
            nh = n_heads_dsa if is_dsa else n_heads_hca
            u_k = f"model.layers.{l}.self_attn.vha_postmix_U"
            v_k = f"model.layers.{l}.self_attn.vha_postmix_V"
            _, _, udt, ush, _ = ck.need(u_k, ("BF16", "F32"))
            _, _, vdt, vsh, _ = ck.need(v_k, ("BF16", "F32"))
            if ush[0] != nh or vsh != ush:
                raise S.ConvertError(f"layer {l}: vha_postmix shapes {ush} / {vsh}, expected [{nh}, rank]")
            self.tensors[f"blk.{l}.attn_vha_mix.weight"] = Tensor(
                f"blk.{l}.attn_vha_mix.weight", "F32", (u_k, v_k), "vha_fold", [nh, nh]
            )
            if l >= self.first_dense:
                g0 = f"model.layers.{l}.block_sparse_moe.gate.weight"
                g1 = f"model.layers.{l}.block_sparse_moe.gate.weight_1"
                ck.need(g0, ("BF16", "F32"), (n_exp, h))
                ck.need(g1, ("BF16", "F32"), (n_exp, h))
                self.tensors[f"blk.{l}.ffn_gate_inp.weight"] = Tensor(
                    f"blk.{l}.ffn_gate_inp.weight", "F32", (g0, g1), "gate_concat", [h, 2 * n_exp]
                )

        # Verify every FP8 scale tensor was paired
        used_scales = {t.scale_src for t in self.tensors.values() if t.scale_src}
        stray_scales = sorted(scale_keys - used_scales)
        if stray_scales:
            raise S.ConvertError("unpaired FP8 scale tensors: " + ", ".join(stray_scales[:8]))
        self._check_complete()

    def _add_simple(self, gname: str, form: str, src: str):
        _, _, dt, shape, _ = self.ck.t[src]
        if gname in self.tensors:
            raise S.ConvertError(f"two checkpoint tensors map to {gname}")
        if form == "FP8_Q8":
            if dt != "F8_E4M3" or len(shape) != 2:
                raise S.ConvertError(f"{src}: expected 2-D F8_E4M3, got {dt} {shape}")
            if shape[1] % 32 != 0:
                raise S.ConvertError(f"{src}: cols {shape[1]} not divisible by 32 for Q8_0")
            sk = scale_name_for(src)
            bn, bk = self.fp8_block
            exp_s = ((shape[0] + bn - 1) // bn, (shape[1] + bk - 1) // bk)
            self.ck.need(sk, ("F32", "BF16"), exp_s)
            self.tensors[gname] = Tensor(gname, form, src, "fp8_block_q8_0", [shape[1], shape[0]], scale_src=sk)
        elif form == "F32_T":
            if dt not in ("F32", "BF16") or len(shape) != 2:
                raise S.ConvertError(f"{src}: expected 2-D F32/BF16, got {dt} {shape}")
            # Checkpoint stores [in, out]; transposed to row-major [out, in] -> GGUF ne = [in, out]
            self.tensors[gname] = Tensor(gname, form, src, "f32_t", [shape[0], shape[1]])
        elif form == "BF16":
            if dt != "BF16":
                raise S.ConvertError(f"{src}: expected BF16, got {dt}")
            self.tensors[gname] = Tensor(gname, form, src, "bf16", list(reversed(shape)))
        elif form == "F32":
            if dt not in ("F32", "BF16"):
                raise S.ConvertError(f"{src}: expected F32 or BF16, got {dt}")
            self.tensors[gname] = Tensor(gname, form, src, "f32", list(reversed(shape)))
        else:
            raise S.ConvertError(f"unknown form {form}")

    def _check_complete(self):
        need = [r[0] for r in GLOBAL.values()]
        for l in range(self.n_layers):
            is_dsa = self.compress_ratios[l] <= 0
            for rule in LAYER_COMMON.values():
                need.append(f"blk.{l}.{rule[0]}")
            for rule in (LAYER_DSA if is_dsa else LAYER_HCA).values():
                need.append(f"blk.{l}.{rule[0]}")
            for rule in (LAYER_DENSE_MLP if l < self.first_dense else LAYER_MOE).values():
                need.append(f"blk.{l}.{rule[0]}")
            need.extend([f"blk.{l}.hc_attn_scale.weight", f"blk.{l}.hc_ffn_scale.weight", f"blk.{l}.attn_vha_mix.weight"])
            if l >= self.first_dense:
                need.append(f"blk.{l}.ffn_gate_inp.weight")
        missing = [n for n in need if n not in self.tensors]
        if missing:
            raise S.ConvertError(f"checkpoint lacks {len(missing)} required tensors: " + ", ".join(missing[:8]))

    def float_bytes(self, name: str) -> bytes:
        t = self.tensors[name]
        if t.how == "bf16":
            raw = self.ck.raw(t.src)
            u16 = np.frombuffer(raw, dtype="<u2")
            f32 = S._bf16_to_f32(u16)
            if not np.isfinite(f32).all():
                raise S.ConvertError(f"{name} has a non-finite BF16 value")
            return raw.tobytes()
        if t.how == "f32":
            x = self.ck.values(t.src)
        elif t.how == "f32_t":
            x = np.ascontiguousarray(self.ck.values(t.src).T, dtype=np.float32)
        elif t.how == "hc_scale":
            x = np.concatenate([self.ck.values(s).reshape(1) for s in t.src]).astype(np.float32)
        elif t.how == "vha_fold":
            u = self.ck.values(t.src[0]).astype(np.float32)
            v = self.ck.values(t.src[1]).astype(np.float32)
            nh = u.shape[0]
            x = (np.eye(nh, dtype=np.float32) + v @ u.T).astype(np.float32)
        elif t.how == "gate_concat":
            g0 = self.ck.values(t.src[0]).astype(np.float32)
            g1 = self.ck.values(t.src[1]).astype(np.float32)
            x = np.ascontiguousarray(np.concatenate([g0, g1], axis=0), dtype=np.float32)
        else:
            raise S.ConvertError(f"float_bytes: unexpected how={t.how}")
        if not np.isfinite(x).all():
            raise S.ConvertError(f"{name} has a non-finite value")
        return np.ascontiguousarray(x, dtype="<f4").tobytes()

    def chunks(self, name: str, pool, depth: int = 32):
        t = self.tensors[name]
        wp, woff, _, wsh, _ = self.ck.t[t.src]
        sp, soff, sdt, ssh, sn = self.ck.t[t.scale_src]
        rows, cols = wsh
        step = max(self.fp8_block[0], (ROWS_PER_JOB // cols // self.fp8_block[0]) * self.fp8_block[0])
        jobs = (
            ("fp8_q8", str(wp), woff, rows, cols, str(sp), soff, sdt, ssh, sn, self.fp8_block, r0, min(step, rows - r0))
            for r0 in range(0, rows, step)
        )
        for b in S.bounded_imap(pool, _job, jobs, depth):
            if isinstance(b, str):
                raise S.ConvertError(f"{name}: {b}")
            yield b


def _job(args):
    kind = args[0]
    if kind == "experts":
        batch = args[1]
        out = []
        for l, e, spec in batch:
            for wp, woff, wn, sp, soff, sn, rows, cols in spec:
                codes = np.frombuffer(_pread_bytes(wp, woff, wn), dtype=np.uint8).reshape(rows, cols // 2)
                scales = np.frombuffer(_pread_bytes(sp, soff, sn), dtype=np.uint8).reshape(rows, cols // 32)
                if (scales == 255).any():
                    return f"layer {l} expert {e}: UE8M0 scale byte 255 (NaN)"
                out.append(mxfp4_rows(codes, scales).tobytes())
        return b"".join(out)
    if kind == "fp8_q8":
        _, wp, woff, rows, cols, sp, soff, sdt, ssh, sn, block, r0, n = args
        codes = np.frombuffer(_pread_bytes(wp, woff + r0 * cols, n * cols), dtype=np.uint8).reshape(n, cols)
        sraw = _pread_bytes(sp, soff, sn)
        if sdt == "F32":
            scales = np.frombuffer(sraw, dtype="<f4").reshape(ssh)
        else:
            scales = S._bf16_to_f32(np.frombuffer(sraw, dtype="<u2").reshape(ssh))
        try:
            vals = fp8_block_dequant(codes, scales, block, row0=r0).reshape(-1)
        except S.ConvertError as exc:
            return str(exc)
        if not np.isfinite(vals).all():
            return "non-finite value after FP8 dequant"
        return S.q8_0(vals)
    return f"unknown job {kind}"


def tokenizer_kv(ck: ErnieCheckpoint) -> dict:
    """Extract ERNIE 5.5 byte-level BPE tokenizer into Basalt's gguf_kv keys (`tokenizer.ggml.pre = 'ernie55'`)."""
    d = ck.dir
    tj = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    tcfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    model = tj.get("model", {})
    if model.get("type") != "BPE":
        raise S.ConvertError(f"tokenizer.json: a {model.get('type')} model; expected BPE")
    n_vocab = int(ck.config["vocab_size"])
    rev = {int(i): t for t, i in model["vocab"].items()}
    added = {int(a["id"]): a for a in tj.get("added_tokens", [])}
    for i, a in added.items():
        rev[i] = a["content"]
    if max(rev) >= n_vocab:
        raise S.ConvertError(f"tokenizer.json: id {max(rev)} >= vocab_size {n_vocab}")
    tokens, types = [], []
    for i in range(n_vocab):
        if i not in rev:
            tokens.append(f"[PAD{i}]")
            types.append(C.UNUSED)
            continue
        tok = rev[i]
        tokens.append(tok)
        if i in added:
            special = bool(added[i].get("special")) or (tok.startswith("<|") and tok.endswith("|>"))
            types.append(C.CONTROL if special else C.USER_DEFINED)
        else:
            types.append(C.NORMAL)
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    ids = {tok: i for i, tok in enumerate(tokens)}

    def tok_id(key):
        v = tcfg.get(key)
        v = v.get("content") if isinstance(v, dict) else v
        return ids.get(v) if v else None

    gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8")) if (d / "generation_config.json").is_file() else {}
    kv = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": ARCH,
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": merges,
    }
    eos = tok_id("eos_token")
    if eos is None:
        g_eos = gen.get("eos_token_id", ck.config.get("eos_token_id"))
        eos = g_eos[-1] if isinstance(g_eos, list) else g_eos
    bos = tok_id("bos_token")
    if bos is None:
        bos = gen.get("bos_token_id", ck.config.get("bos_token_id"))
    pad = tok_id("pad_token")
    if pad is None:
        pad = gen.get("pad_token_id", ck.config.get("pad_token_id"))
    for k, v in (("eos", eos), ("bos", bos), ("padding", pad)):
        if v is not None:
            kv[f"tokenizer.ggml.{k}_token_id"] = int(v)
    kv["tokenizer.ggml.add_bos_token"] = bool(tcfg.get("add_bos_token", False))
    return kv


def model_kv(ck: ErnieCheckpoint, name: str) -> dict:
    c = ck.config
    A = ARCH + "."
    d = ck.dir
    gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8")) if (d / "generation_config.json").is_file() else {}
    card = C.front_matter(d / "README.md")
    rs = _req(c, "rope_scaling")
    kv = {"general.architecture": ARCH, "general.type": "model", "general.name": name}
    if "license" in card:
        kv["general.license"] = card["license"]
    if "temperature" in gen:
        kv["general.sampling.temp"] = C.f32(gen["temperature"])
    if "top_p" in gen:
        kv["general.sampling.top_p"] = C.f32(gen["top_p"])
    kv["general.sampling.top_k"] = int(gen.get("top_k", 0))
    kv.update({
        A + "block_count": int(_req(c, "num_hidden_layers")),
        A + "first_k_dense_replace": int(_req(c, "first_k_dense_replace")),
        A + "context_length": int(_req(c, "max_position_embeddings")),
        A + "embedding_length": int(_req(c, "hidden_size")),
        A + "feed_forward_length": int(_req(c, "intermediate_size")),
        A + "vocab_size": int(_req(c, "vocab_size")),
        A + "attention.layer_norm_rms_epsilon": C.f32(_req(c, "rms_norm_eps")),
        A + "attention.sliding_window": int(_req(c, "sliding_window")),
        A + "attention.compress_ratios": [int(x) for x in _req(c, "compress_ratios")],
        # HCA geometry
        A + "attention.head_count": int(_req(c, "num_attention_heads")),
        A + "attention.head_count_kv": int(_req(c, "num_key_value_heads")),
        A + "attention.key_length": int(_req(c, "head_dim")),
        A + "attention.value_length": int(_req(c, "v_head_dim")),
        A + "attention.q_lora_rank": int(_req(c, "q_lora_rank")),
        A + "attention.kv_lora_rank": int(_req(c, "kv_lora_rank")),
        A + "attention.output_group_count": int(_req(c, "o_groups")),
        A + "attention.output_lora_rank": int(_req(c, "o_lora_rank")),
        A + "rope.dimension_count": int(_req(c, "qk_rope_head_dim")),
        A + "rope.freq_base": C.f32(_req(c, "rope_theta")),
        A + "attention.compress_rope_freq_base": C.f32(_req(c, "compress_rope_theta")),
        A + "rope.scaling.factor": C.f32(_req(rs, "factor", "config.json.rope_scaling")),
        A + "rope.scaling.original_context_length": int(_req(rs, "original_max_position_embeddings", "config.json.rope_scaling")),
        A + "rope.scaling.beta_fast": C.f32(_req(rs, "beta_fast", "config.json.rope_scaling")),
        A + "rope.scaling.beta_slow": C.f32(_req(rs, "beta_slow", "config.json.rope_scaling")),
        # MLA-DSA geometry
        A + "mla.head_count": int(_req(c, "hybrid_mla_num_attention_heads")),
        A + "mla.q_lora_rank": int(_req(c, "hybrid_mla_q_lora_rank")),
        A + "mla.kv_lora_rank": int(_req(c, "hybrid_mla_kv_lora_rank")),
        A + "mla.qk_nope_head_dim": int(_req(c, "hybrid_mla_qk_nope_head_dim")),
        A + "mla.qk_rope_head_dim": int(_req(c, "hybrid_mla_qk_rope_head_dim")),
        A + "mla.v_head_dim": int(_req(c, "hybrid_mla_v_head_dim")),
        A + "attention.indexer.head_count": int(_req(c, "index_n_heads")),
        A + "attention.indexer.key_length": int(_req(c, "index_head_dim")),
        A + "attention.indexer.top_k": int(_req(c, "index_topk")),
        # MoE geometry
        A + "expert_count": int(_req(c, "n_routed_experts")),
        A + "expert_used_count": int(_req(c, "num_experts_per_tok")),
        A + "expert_shared_count": int(_req(c, "n_shared_experts")),
        A + "expert_feed_forward_length": int(_req(c, "moe_intermediate_size")),
        A + "expert_latent_length": int(_req(c, "moe_latent_size")),
        A + "expert_weights_norm": bool(_req(c, "norm_topk_prob")),
        A + "expert_gate_softcap": C.f32(_req(c, "activation_situ_beta")),
        A + "expert_up_softcap": C.f32(_req(c, "activation_situ_linear_beta")),
        # mHC
        A + "hyper_connection.count": int(_req(c, "hc_mult")),
        A + "hyper_connection.sinkhorn_iterations": int(_req(c, "hc_sinkhorn_iters")),
        A + "hyper_connection.epsilon": C.f32(_req(c, "hc_eps")),
        A + "has_flashindexer": bool(ck.has_flashindexer),
        "general.file_type": MXFP4,
        "general.quantization_version": 2,
    })
    kv.update(tokenizer_kv(ck))
    return kv


LICENSE = {"apache-2.0": "Apache License 2.0"}


def build(a, Job, fingerprint, die) -> None:
    import tokenizer
    out = pathlib.Path(a.out)
    partial = pathlib.Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    try:
        ck = ErnieCheckpoint(a.checkpoint)
        plan = Plan(ck)
        experts = Experts(ck)
        name = a.name or "ERNIE-5.5-Flash-Preview MXFP4"
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
                if info.type_name == "Q8_0":
                    parts.append(Stream(info.expected_bytes(), plan.chunks(n, pool)))
                else:
                    parts.append(plan.float_bytes(n))
                names.append(n)
            offs = w.add_parts("gguf", parts, TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            tens = {}
            for n, off in zip(names, offs):
                t_ = plan.tensors[n]
                tens[n] = {
                    "type": t_.type_name,
                    "type_id": t_.type_id,
                    "shape": [int(d) for d in t_.ne],
                    "bytes": t_.expected_bytes(),
                    "offset": off,
                }
            lic = kv.get("general.license", "apache-2.0")
            meta = {
                "format": "basalt",
                "version": 2,
                "name": name,
                "arch": ARCH,
                "converter": {"tool": "tools/convert/convert.py"},
                "sources": [
                    {
                        "path": p.name,
                        "size": p.stat().st_size,
                        "fingerprint_sha256_first_last_64mib_and_size": fingerprint(p),
                    }
                    for p in ck.files
                ],
                "gguf_kv": kv,
                "tensors": tens,
                "license": {"name": LICENSE.get(lic, lic)},
                "hf_config": ck.config,
                "checkpoint": {
                    "dir": pathlib.Path(a.checkpoint).name,
                    "config_sha256": hashlib.sha256((ck.dir / "config.json").read_bytes()).hexdigest(),
                    "experts_mxfp4": "w1 / w3 / w2 E2M1 nibbles + UE8M0 scales repacked bit-exact to ggml MXFP4 (type 39)",
                    "fp8_projections": "128x128 blockwise E4M3 x FP32 scale_inv -> Q8_0 (ggml reference rounding)",
                    "has_flashindexer": ck.has_flashindexer,
                    "transforms": {k: METHODS[k] for k in sorted({t_.how for t_ in plan.tensors.values()})},
                },
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
