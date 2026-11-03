"""A DeepSeek-V4.1 pack from NVIDIA Model Optimizer's NVFP4 checkpoint (nvidia/DeepSeek-V4.1-Flash-1017-NVFP4), for
tools/convert/convert.py --sections-from BASE.basalt --checkpoint DIR.

  experts / experts.layout  each routed expert's NVFP4 w1 (gate) | w3 (up) | w2 (down) rows moved into Basalt's blocks
                            (codes and UE4M3 scales unchanged) with the FP32 tail {s_gate, s_up, s_down, 0} from
                            `weight_scale_2` (checkpoint.py's repack; `input_scale` is not used)
  gguf                      every other language-model tensor under the GGUF name BASE has it, from the checkpoint: the
                            FP8 projections (E4M3 codes with E8M0 scales per 32 x 32 block) as Q8_0 of their exact
                            values, or BF16 where BASE reads BF16; BF16 and F32 tensors kept, or widened to F32 where BASE
                            reads F32; the token embedding and the output head Q8_0 from BF16; the Engram gate the product
                            of its query and key weights. Each tensor is checked against BASE's tensor of the same name.
  engram                    the MXFP4 Engram tables (E2M1 codes, E8M0 scales per 32 values) as ggml MXFP4 rows, codes
                            and scales unchanged, the layers' tables back to back (checked against BASE's rows)
The tokenizer, the metadata, the vision projector and the DSpark drafter come from BASE byte for byte (BASE: a
deepseek41 pack of the same model, such as the UD-IQ1_M one).

The checkpoint's shards are read in file order. With --stream-shards they may arrive while the conversion runs (another
process downloads them in that order; a shard appears whole, as `hf download` writes it) and each is deleted once its
tensors are in the pack, so the disk holds the pack and a shard or two rather than the whole checkpoint. Needs numpy.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import pathlib
import re
import tempfile
import time

import numpy as np

import ggufq
import model_sections as S
from checkpoint import _expert_job, bounded_imap
from model_file import TENSOR_ALIGN, ModelFile, ModelWriter

MXFP4 = 39
ENGRAM_ROW = 256
ROLES = ("w1", "w3", "w2")           # gate, up, down
EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|weight_scale|weight_scale_2|input_scale)$")
ENGRAM_RE = re.compile(r"^layers\.(\d+)\.engram\.embed\.(weight|scale)$")
# checkpoint tensor (without `layers.N.`) -> GGUF name (without `blk.N.`)
LAYER = {
    "attn.attn_sink": "attn_sinks.weight", "attn.wq_a.weight": "attn_q_a.weight", "attn.wq_b.weight": "attn_q_b.weight",
    "attn.q_norm.weight": "attn_q_a_norm.weight", "attn.wkv.weight": "attn_kv.weight",
    "attn.kv_norm.weight": "attn_kv_a_norm.weight", "attn.wo_a.weight": "attn_output_a.weight",
    "attn.wo_b.weight": "attn_output_b.weight", "attn.compressor.wkv.weight": "attn_compressor_kv.weight",
    "attn.compressor.wgate.weight": "attn_compressor_gate.weight", "attn.compressor.norm.weight": "attn_compressor_norm.weight",
    "attn.indexer.wq_b.weight": "indexer.attn_q_b.weight", "attn.indexer.wk.weight": "indexer.attn_k.weight",
    "attn.indexer.k_norm.weight": "indexer.k_norm.weight", "attn.indexer.weights_proj.weight": "indexer.proj.weight",
    "attn_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight", "ffn.gate.weight": "ffn_gate_inp.weight",
    "ffn.gate.bias": "exp_probs_b.bias", "ffn.gate.bias_vl": "exp_probs_b_vl.bias",
    "ffn.shared_experts.w1.weight": "ffn_gate_shexp.weight", "ffn.shared_experts.w3.weight": "ffn_up_shexp.weight",
    "ffn.shared_experts.w2.weight": "ffn_down_shexp.weight", "engram.wkv.weight": "engram_kv.weight",
    "hc_attn_fn": "hc_attn_fn.weight", "hc_attn_base": "hc_attn_base.weight", "hc_attn_scale": "hc_attn_scale.weight",
    "hc_ffn_fn": "hc_ffn_fn.weight", "hc_ffn_base": "hc_ffn_base.weight", "hc_ffn_scale": "hc_ffn_scale.weight",
}
GLOBAL = {"embed.weight": "token_embd.weight", "head.weight": "output.weight", "norm.weight": "output_norm.weight"}
ENGRAM_GATE = ("engram.q_weight", "engram.k_weight")   # their product is the GGUF engram_gate.weight
SKIP = re.compile(r"^(mtp\.|vision\.|aligner\.|image_)")   # the drafter and the vision encoder come from BASE
TYPE_ID = {"F32": 0, "Q8_0": 8, "BF16": 30}
QUANTIZED = {"Q8_0", "Q6_K", "Q5_K", "Q4_K", "IQ4_XS", "IQ4_NL"}   # BASE types the checkpoint's tensor becomes Q8_0 for
ROWS_PER_JOB = 1 << 13
ENGRAM_ROWS_PER_JOB = 1 << 20
CHECK_RMS = 0.15                     # a tensor further than this from BASE's (relative RMS) is a layout error
METHODS = {
    "fp8_q8_0": "E4M3 code x 2^(E8M0 - 127) per 32 x 32 block (exact in f32), then Q8_0 (ggml reference rounding)",
    "fp8_bf16": "E4M3 code x 2^(E8M0 - 127) per 32 x 32 block, exact in BF16",
    "bf16_q8_0": "BF16 to Q8_0 (ggml reference rounding)",
    "bf16": "as stored", "f32": "as stored", "bf16_f32": "BF16 widened to F32 (exact)",
    "gate_product": "q_weight x k_weight, exact in F32",
}


def _e4m3_table() -> np.ndarray:
    t = np.zeros(256, np.float32)
    for b in range(256):
        e, m = (b >> 3) & 15, b & 7
        v = m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        t[b] = -v if b >> 7 else v
    t[0x7F] = t[0xFF] = np.nan
    return t


E4M3 = _e4m3_table()


def fp8_values(codes: np.ndarray, scales: np.ndarray, skip: int = 0) -> np.ndarray:
    """E4M3 codes [n, C] times 2^(E8M0 - 127) of their 32 x 32 block, in f32 (exact). `scales` holds the block rows from
    the one the codes' first row is in, `skip` rows into it."""
    v = E4M3[codes]
    if np.isnan(v).any() or (scales == 255).any():
        raise S.ConvertError("a NaN E4M3 weight or E8M0 block scale")
    n, c = codes.shape
    s = np.ldexp(np.float32(1), scales.astype(np.int32) - 127).astype(np.float32)
    return v * np.repeat(np.repeat(s, 32, 0)[skip:skip + n], 32, 1)[:, :c]


def mxfp4_rows(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """The checkpoint's MXFP4 rows (codes [rows, 128]: value 2i in the low nibble of byte i, 2i + 1 in the high one;
    E8M0 scales [rows, 8]) as ggml MXFP4 rows: per 32 values the scale byte, then 16 bytes whose low nibble is value j
    and high nibble value j + 16."""
    rows = codes.shape[0]
    v = np.empty((rows, ENGRAM_ROW), np.uint8)
    v[:, 0::2] = codes & 15
    v[:, 1::2] = codes >> 4
    v = v.reshape(rows, ENGRAM_ROW // 32, 2, 16)
    out = np.empty((rows, ENGRAM_ROW // 32, 17), np.uint8)
    out[..., 0] = scales
    out[..., 1:] = v[:, :, 0, :] | (v[:, :, 1, :] << 4)
    return out.reshape(-1)


MXFP4_VALUES = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def mxfp4_values(rows: np.ndarray) -> np.ndarray:
    """ggml MXFP4 rows (17-byte blocks, ENGRAM_ROW values a row) as f32 [rows, ENGRAM_ROW]."""
    b = rows.reshape(-1, ENGRAM_ROW // 32, 17)
    d = np.ldexp(np.float32(1), b[..., 0].astype(np.int32) - 127).astype(np.float32)[..., None]
    q = b[..., 1:]
    v = np.concatenate([MXFP4_VALUES[q & 15], MXFP4_VALUES[q >> 4]], axis=-1) * d
    return v.reshape(-1, ENGRAM_ROW)


def _engram_job(args):
    path, w_off, s_off, r0, n = args
    codes = np.fromfile(path, np.uint8, n * 128, offset=w_off + r0 * 128).reshape(n, 128)
    scales = np.fromfile(path, np.uint8, n * 8, offset=s_off + r0 * 8).reshape(n, 8)
    if (scales == 255).any():
        return f"{pathlib.Path(path).name}: an Engram scale is NaN"
    return mxfp4_rows(codes, scales).tobytes()


def _dense_job(args):
    """Rows [r0, r0 + n) of a 2-D BF16 or FP8 tensor [R, C] as `how` makes them."""
    path, w_off, dt, c, s_off, how, r0, n = args
    try:
        if dt == "F8_E4M3":
            codes = np.fromfile(path, np.uint8, n * c, offset=w_off + r0 * c).reshape(n, c)
            sc_cols = (c + 31) // 32
            b0, b1 = r0 // 32, (r0 + n + 31) // 32
            sc = np.fromfile(path, np.uint8, (b1 - b0) * sc_cols, offset=s_off + b0 * sc_cols).reshape(b1 - b0, sc_cols)
            x = fp8_values(codes, sc, r0 - b0 * 32)
        else:
            x = S._bf16_to_f32(np.fromfile(path, "<u2", n * c, offset=w_off + r0 * c * 2)).reshape(n, c)
    except S.ConvertError as e:
        return str(e)
    if not np.isfinite(x).all():
        return "a non-finite value"
    if how in ("fp8_q8_0", "bf16_q8_0"):
        return S.q8_0(x)
    u = np.ascontiguousarray(x).view(np.uint32)
    if np.count_nonzero(u & 0xFFFF):
        return "a value that is not BF16"
    return (u >> 16).astype("<u2").tobytes()


class Shards:
    """The checkpoint's shards by `model.safetensors.index.json`; a shard's header is read when it is first used (with
    `stream`, once the shard is there), and `release` deletes it (with `stream`)."""

    def __init__(self, d, stream: bool, record):
        self.dir, self.stream, self.record = pathlib.Path(d), stream, record
        for f in ("config.json", "model.safetensors.index.json"):
            if not (self.dir / f).is_file():
                raise S.ConvertError(f"{self.dir / f} not found")
        self.config = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        self.where = json.loads((self.dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        self.heads: dict[str, tuple[dict, int]] = {}
        self.records: list[dict] = []

    def header(self, f: str):
        if f not in self.heads:
            p = self.dir / f
            if self.stream and not p.is_file():
                print(f"  waiting for {f} ...", flush=True)
                while not p.is_file():
                    time.sleep(5)
            if not p.is_file():
                raise S.ConvertError(f"{p} not found")
            self.heads[f] = S.read_safetensors_header(p)
            self.records.append(self.record(p))
        return self.heads[f]

    def t(self, name: str):
        """(path, absolute offset, dtype, shape, bytes) of a tensor, its shard's header read."""
        f = self.where[name]
        head, base = self.header(f)
        v = head[name]
        a, b = v["data_offsets"]
        return str(self.dir / f), base + a, v["dtype"], list(v["shape"]), b - a

    def need(self, name, dtype, shape=None):
        _, _, dt, sh, _ = self.t(name)
        if dt != dtype or (shape is not None and sh != list(shape)):
            raise S.ConvertError(f"{name} is {dt} {sh}, expected {dtype} {list(shape) if shape is not None else ''}")

    def release(self, f: str):
        if self.stream:
            (self.dir / f).unlink(missing_ok=True)


class Plan:
    """What the pack takes from which shard: the routed experts by layer, the Engram tables, the other tensors."""

    def __init__(self, ck: Shards, base_meta: dict):
        tc = ck.config.get("text_config", ck.config)
        self.n_layers, self.n_expert = int(tc["num_hidden_layers"]), int(tc["n_routed_experts"])
        self.h, self.ff = int(tc["hidden_size"]), int(tc["moe_intermediate_size"])
        geo = base_meta["geometry"]
        mine = (self.n_layers, self.n_expert, self.h, self.ff)
        theirs = (geo.get("block_count"), geo.get("expert_count"), geo.get("embedding_length"), geo.get("expert_feed_forward_length"))
        if mine != theirs:
            raise S.ConvertError(f"the checkpoint is {mine} (layers, experts, hidden, expert FF), BASE {theirs}")
        self.blob = 2 * self.ff * S.nvfp4_row_bytes(self.h) + self.h * S.nvfp4_row_bytes(self.ff) + S.NVFP4_TAIL
        self.layer_file = {}
        for name, f in ck.where.items():
            m = EXPERT_RE.match(name)
            if not m:
                continue
            l, e = int(m.group(1)), int(m.group(2))
            if l >= self.n_layers or e >= self.n_expert:
                raise S.ConvertError(f"unexpected expert tensor {name}")
            if self.layer_file.setdefault(l, f) != f:
                raise S.ConvertError(f"layer {l}'s experts span two shards ({self.layer_file[l]}, {f})")
        if sorted(self.layer_file) != list(range(self.n_layers)):
            raise S.ConvertError("the checkpoint lacks the experts of some layers")
        files = [self.layer_file[l] for l in range(self.n_layers)]
        if files != sorted(files) or len(set(files)) != len(files):
            raise S.ConvertError("each layer's experts are read from a shard of their own, in file order")
        self.tensors: dict[str, tuple] = {}     # GGUF name -> (how, source names)
        self.engram: list[tuple[int, str]] = []  # (layer, shard), in layer order
        need = set(files)
        base_t = base_meta["tensors"]
        unknown = []
        gate: dict[int, list[str]] = {}
        for name, f in ck.where.items():
            if SKIP.match(name) or EXPERT_RE.match(name) or name.endswith(".scale") and name[:-6] + ".weight" in ck.where:
                continue
            m = ENGRAM_RE.match(name)
            if m:
                if m.group(2) == "weight":
                    self.engram.append((int(m.group(1)), f))
                need.add(f)
                continue
            m = re.match(r"^layers\.(\d+)\.(.+)$", name)
            if m and m.group(2) in ENGRAM_GATE:
                gate.setdefault(int(m.group(1)), []).append(name)
                need.add(f)
                continue
            gname = (f"blk.{m.group(1)}.{LAYER[m.group(2)]}" if m and m.group(2) in LAYER else
                     GLOBAL.get(name) if not m else None)
            if gname is None:
                unknown.append(name)
                continue
            self.tensors[gname] = ("src", [name])
            need.add(f)
        for l, names in gate.items():
            self.tensors[f"blk.{l}.engram_gate.weight"] = ("gate_product", sorted(names, key=lambda n: n.endswith("k_weight")))
        if unknown:
            raise S.ConvertError("checkpoint tensors with no GGUF counterpart: " + ", ".join(sorted(unknown)[:8]))
        missing = sorted(set(base_t) - set(self.tensors) - {base_meta["engram"]["tensor"]})
        extra = sorted(set(self.tensors) - set(base_t))
        if missing or extra:
            raise S.ConvertError(f"the checkpoint's tensors differ from BASE's: missing {missing[:8]}, extra {extra[:8]}")
        self.engram.sort()
        if [l for l, _ in self.engram] != list(base_meta["engram"]["ple.layers"]):
            raise S.ConvertError(f"the checkpoint's Engram layers {[l for l, _ in self.engram]} are not BASE's")
        self.files = sorted(need)
        last = self.files.index(files[-1])
        self.before, self.after = self.files[:last + 1], self.files[last + 1:]
        if any(f in self.before for _, f in self.engram):
            raise S.ConvertError("an Engram table comes before the last layer's experts in file order")

    def layout(self):
        return [(l, S.NVFP4, S.NVFP4, l * self.blob * self.n_expert, self.blob) for l in range(self.n_layers)]


def gguf_how(src_dtype: str, base_type: str) -> tuple[str, str]:
    """(method, stored type) of a checkpoint tensor whose BASE counterpart is `base_type`."""
    if base_type in QUANTIZED:
        return ("fp8_q8_0" if src_dtype == "F8_E4M3" else "bf16_q8_0"), "Q8_0"
    if base_type == "BF16":
        return ("fp8_bf16" if src_dtype == "F8_E4M3" else "bf16"), "BF16"
    if base_type == "F32":
        return ("f32" if src_dtype == "F32" else "bf16_f32"), "F32"
    raise S.ConvertError(f"no conversion of {src_dtype} to a tensor BASE keeps as {base_type}")


class Converter:
    def __init__(self, a, base: ModelFile, base_meta: dict, pool, record):
        self.a, self.base, self.base_meta, self.pool = a, base, base_meta, pool
        self.ck = Shards(a.checkpoint, a.stream_shards, record)
        self.plan = Plan(self.ck, base_meta)
        self.spool = pathlib.Path(tempfile.mkdtemp(prefix="ds41-dense-"))
        self.out: dict[str, tuple[str, int, str]] = {}   # GGUF name -> (spool file, bytes, stored type)
        self.records: list[dict] = []

    # ---------------------------------------------------------------- the other tensors
    def spool_shard(self, f: str):
        """Every non-expert tensor of shard `f` the plan takes, converted, into the spool."""
        for gname, (how, srcs) in self.plan.tensors.items():
            if self.ck.where[srcs[0]] != f or gname in self.out:
                continue
            bt = self.base_meta["tensors"][gname]
            if how == "gate_product":
                q, k = (self.ck_values(n) for n in srcs)
                x = (q * k).astype("<f4")
                self.put(gname, x.tobytes(), "F32", how, srcs, list(reversed(q.shape)))
                continue
            path, off, dt, shape, _ = self.ck.t(srcs[0])
            if list(reversed(shape)) != [int(d) for d in bt["shape"]]:
                raise S.ConvertError(f"{srcs[0]} {shape} does not match BASE's {gname} {bt['shape']}")
            method, ty = gguf_how(dt, bt["type"])
            if len(shape) != 2 or method in ("f32", "bf16_f32"):
                x = self.ck_values(srcs[0])
                data = x.astype("<f4").tobytes() if ty == "F32" else None
                if data is None:
                    raise S.ConvertError(f"{srcs[0]}: a {len(shape)}-D tensor that BASE keeps as {ty}")
                self.put(gname, data, ty, method, srcs, list(reversed(shape)))
                continue
            s_off = self.ck.t(srcs[0][:-7] + ".scale")[1] if dt == "F8_E4M3" else 0
            if dt == "F8_E4M3":
                self.ck.need(srcs[0][:-7] + ".scale", "F8_E8M0", [(shape[0] + 31) // 32, (shape[1] + 31) // 32])
            elif dt != "BF16":
                raise S.ConvertError(f"{srcs[0]} is {dt}")
            jobs = [(path, off, dt, shape[1], s_off, method, r, min(ROWS_PER_JOB, shape[0] - r))
                    for r in range(0, shape[0], ROWS_PER_JOB)]
            p = self.spool / gname
            n = 0
            with open(p, "wb") as fo:
                for b in bounded_imap(self.pool, _dense_job, jobs, 32):
                    if isinstance(b, str):
                        raise S.ConvertError(f"{srcs[0]}: {b}")
                    fo.write(b)
                    n += len(b)
            self.out[gname] = (str(p), n, ty)
            self.records.append({"name": gname, "from": srcs, "src_type": dt, "dst_type": ty, "method": method})
            self.check(gname)

    def ck_values(self, name) -> np.ndarray:
        path, off, dt, shape, n = self.ck.t(name)
        raw = np.fromfile(path, np.uint8, n, offset=off)
        if dt == "BF16":
            return S._bf16_to_f32(raw.view("<u2")).reshape(shape)
        if dt == "F32":
            return raw.view("<f4").reshape(shape)
        raise S.ConvertError(f"{name} is {dt}")

    def put(self, gname, data: bytes, ty, method, srcs, ne):
        if ty == "F32" and not np.isfinite(np.frombuffer(data, "<f4")).all():
            raise S.ConvertError(f"{gname}: a non-finite value")
        p = self.spool / gname
        p.write_bytes(data)
        self.out[gname] = (str(p), len(data), ty)
        self.records.append({"name": gname, "from": srcs, "dst_type": ty, "method": method})
        self.check(gname)

    def base_values(self, gname, rows: int | None = None) -> np.ndarray:
        bt = self.base_meta["tensors"][gname]
        off, _ = self.base.section(bt.get("section", "gguf"))
        ne0 = int(bt["shape"][0])
        n = bt["bytes"] if rows is None else bt["bytes"] // int(np.prod(bt["shape"][1:] or [1])) * rows
        raw = np.fromfile(self.base.path, np.uint8, n, offset=off + bt["offset"])
        return ggufq.dequantize(raw, ggufq.GGMLQuantizationType[bt["type"]]).astype(np.float32).reshape(-1, ne0)

    def check(self, gname):
        """The converted tensor against BASE's of the same name: a layout or naming error shows as a large distance."""
        p, n, ty = self.out[gname]
        bt = self.base_meta["tensors"][gname]
        ne0 = int(bt["shape"][0])
        rows = min(int(np.prod(bt["shape"][1:] or [1])), 4096)
        raw = np.fromfile(p, np.uint8, min(n, n // int(np.prod(bt["shape"][1:] or [1])) * rows))
        mine = ggufq.dequantize(raw, ggufq.GGMLQuantizationType[ty]).astype(np.float64).reshape(-1, ne0)
        theirs = self.base_values(gname, rows).astype(np.float64)
        d = float(np.sqrt(((mine - theirs) ** 2).sum() / max((theirs ** 2).sum(), 1e-30)))
        if not d < CHECK_RMS:
            raise S.ConvertError(f"{gname}: {d:.3f} relative RMS from BASE's {bt['type']} values; a layout error")
        self.records[-1]["rel_rms_vs_sections_from"] = round(d, 5)

    # ---------------------------------------------------------------- experts and Engram
    def expert_chunks(self):
        P, ck = self.plan, self.ck
        for f in P.before:
            ck.header(f)
            self.spool_shard(f)
            for l in [l for l, lf in P.layer_file.items() if lf == f]:
                entries = []
                for e in range(P.n_expert):
                    ex = []
                    for r in ROLES:
                        p = f"layers.{l}.ffn.experts.{e}.{r}."
                        rows, cols = (P.ff, P.h) if r != "w2" else (P.h, P.ff)
                        ck.need(p + "weight", "U8", (rows, cols // 2))
                        ck.need(p + "weight_scale", "F8_E4M3", (rows, cols // 16))
                        ck.need(p + "weight_scale_2", "F32", ())
                        w, s, t = (ck.t(p + k) for k in ("weight", "weight_scale", "weight_scale_2"))
                        ex.append(((pathlib.Path(w[0]), w[1], w[3]), (pathlib.Path(s[0]), s[1], s[3]), (pathlib.Path(t[0]), t[1])))
                    entries.append(ex)
                jobs = ((entries[s:s + 16], min(16, P.n_expert - s)) for s in range(0, P.n_expert, 16))
                for b in bounded_imap(self.pool, _expert_job, jobs, 32):
                    if isinstance(b, str):
                        raise S.ConvertError(f"layer {l}: {b}")
                    yield b
                print(f"  layer {l} experts written", flush=True)
            ck.release(f)

    def engram_chunks(self):
        P, ck = self.plan, self.ck
        sizes = []
        for f in P.after:
            ck.header(f)
            self.spool_shard(f)
            for l in [l for l, lf in P.engram if lf == f]:
                w, s = f"layers.{l}.engram.embed.weight", f"layers.{l}.engram.embed.scale"
                path, w_off, dt, shape, _ = ck.t(w)
                rows = shape[0]
                ck.need(w, "U8", (rows, ENGRAM_ROW // 2))
                ck.need(s, "F8_E8M0", (rows, ENGRAM_ROW // 32))
                s_off = ck.t(s)[1]
                self.check_engram(path, w_off, s_off, rows, sum(sizes))
                jobs = ((path, w_off, s_off, r, min(ENGRAM_ROWS_PER_JOB, rows - r)) for r in range(0, rows, ENGRAM_ROWS_PER_JOB))
                for b in bounded_imap(self.pool, _engram_job, jobs, 8):
                    if isinstance(b, str):
                        raise S.ConvertError(b)
                    yield b
                sizes.append(rows)
                print(f"  layer {l} Engram table written ({rows} rows)", flush=True)
            ck.release(f)
        self.engram_rows = sizes

    def check_engram(self, path, w_off, s_off, rows, first):
        """Rows of the table spread over it against BASE's rows of the same ids."""
        e = self.base_meta["engram"]
        off, _ = self.base.section(e["section"])
        ids = np.linspace(0, rows - 1, 64).astype(np.int64)
        mine, theirs = [], []
        for r in ids:
            mine.append(mxfp4_values(np.frombuffer(_engram_job((path, w_off, s_off, int(r), 1)), np.uint8))[0])
            raw = np.fromfile(self.base.path, np.uint8, e["row_bytes"], offset=off + (first + int(r)) * e["row_bytes"])
            theirs.append(ggufq.dequantize(raw, ggufq.GGMLQuantizationType[e["type"]]).astype(np.float32))
        m, t = np.array(mine, np.float64), np.array(theirs, np.float64)
        d = float(np.sqrt(((m - t) ** 2).sum() / max((t ** 2).sum(), 1e-30)))
        if not d < 0.3:
            raise S.ConvertError(f"the Engram rows of {pathlib.Path(path).name} are {d:.3f} relative RMS from BASE's; a layout error")
        print(f"  Engram rows vs BASE's: {d:.4f} relative RMS", flush=True)
        self.engram_check = round(d, 5)


def build(a, Job, fingerprint, source_record) -> None:
    """--sections-from BASE.basalt --checkpoint DIR for a deepseek41 BASE."""
    out, src = pathlib.Path(a.out), pathlib.Path(a.sections_from)
    partial = pathlib.Path(str(out) + ".partial")
    if out.exists() and not a.force:
        raise SystemExit(f"convert: {out} exists (use --force to overwrite)")
    base = ModelFile(str(src))
    base_meta = json.loads(base.read("meta"))
    if partial.exists():
        partial.unlink()
    with multiprocessing.Pool(a.workers) as pool:
        try:
            c = Converter(a, base, base_meta, pool, source_record)
        except S.ConvertError as e:
            raise SystemExit(f"convert: {e}")
        P = c.plan
        w = ModelWriter(str(partial))
        job = Job(w)
        ok = False
        try:
            print("writing sections:", flush=True)
            for name in base.names():
                if name.startswith("tokenizer/"):
                    job.file(name, src, *base.section(name))
            job.chunks("experts", c.expert_chunks())
            if w.entries[-1][2] != P.n_layers * P.n_expert * P.blob:
                raise S.ConvertError(f"experts: {w.entries[-1][2]} bytes written, {P.n_layers * P.n_expert * P.blob} planned")
            job.bytes("experts.layout", S.nvfp4_layout_text(P.n_expert, P.n_layers * P.n_expert * P.blob, P.layout()))
            replaced = {"experts", "experts.layout", "meta"}
            meta = dict(base_meta)
            job.chunks("engram", c.engram_chunks())
            order = sorted(base_meta["tensors"], key=lambda n: base_meta["tensors"][n]["offset"])
            order = [n for n in order if base_meta["tensors"][n].get("section", "gguf") == "gguf"]
            t = time.time()
            offs = w.add_parts("gguf", [(c.out[n][0], 0, c.out[n][1]) for n in order], TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            tens = {}
            for n, o in zip(order, offs):
                _, nb, ty = c.out[n]
                tens[n] = {**base_meta["tensors"][n], "type": ty, "type_id": TYPE_ID[ty], "bytes": nb, "offset": o}
            e = dict(base_meta["engram"])
            rows = sum(c.engram_rows)
            if rows != e["rows"]:
                raise S.ConvertError(f"the Engram tables hold {rows} rows, BASE's {e['rows']}")
            e.update(type="MXFP4", type_id=MXFP4, row_bytes=ENGRAM_ROW // 32 * 17, bytes=rows * (ENGRAM_ROW // 32 * 17))
            tens[e["tensor"]] = {**base_meta["tensors"][e["tensor"]], "type": "MXFP4", "type_id": MXFP4, "bytes": e["bytes"]}
            meta["tensors"], meta["engram"] = tens, e
            meta["conversions"] = {"compat_bf16": False, "tensors": c.records}
            kv = dict(base_meta["gguf_kv"])
            for k in ("general.quantized_by", "general.tags", "general.file_type"):
                kv.pop(k, None)
            meta["gguf_kv"] = kv
            meta["license"] = {k: v for k, v in base_meta["license"].items() if k != "quantized_by"}
            replaced |= {"gguf", "engram", "dense", "dense.index"}
            copied = []
            for name in base.names():
                if name in replaced or name.startswith("tokenizer/"):
                    continue
                job.file(name, src, *base.section(name))
                if w.entries[-1][3] != base.sections[name][2]:
                    raise S.ConvertError(f"section {name}: the copy's SHA-256 differs from {src}'s")
                copied.append(name)
            meta.update({
                "name": a.name or base_meta.get("name"),
                "converter": {"tool": "tools/convert/convert.py"},
                "sources": c.ck.records,
                "checkpoint": {
                    "dir": pathlib.Path(a.checkpoint).name,
                    "config_sha256": hashlib.sha256((pathlib.Path(a.checkpoint) / "config.json").read_bytes()).hexdigest(),
                    "experts": {"format": "nvfp4", "source": "w1 (gate), w3 (up), w2 (down): weight (E2M1 pairs), "
                                "weight_scale (UE4M3 per 16), weight_scale_2 (FP32); input_scale unused",
                                "repack": "codes and scales unchanged, nibbles moved into 64-value blocks",
                                "tail": "fp32 {s_gate, s_up, s_down, 0} per expert"},
                    "dense": {"transforms": {k: METHODS[k] for k in sorted({r["method"] for r in c.records})},
                              "checked": f"every tensor within {CHECK_RMS} relative RMS of {src.name}'s"},
                    "engram": {"format": "mxfp4", "source": "embed.weight (E2M1 pairs), embed.scale (E8M0 per 32)",
                               "repack": "codes and scales unchanged, into ggml MXFP4 blocks",
                               "rel_rms_vs_sections_from": c.engram_check},
                },
                "sections_from": {"path": src.name, "size": src.stat().st_size,
                                  "fingerprint_sha256_first_last_64mib_and_size": fingerprint(src), "copied": copied},
            })
            meta.pop("dense_requant", None)
            job.bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            w.close()
            ok = True
        except S.ConvertError as e:
            raise SystemExit(f"convert: {e}")
        finally:
            for p in c.spool.iterdir():
                p.unlink()
            c.spool.rmdir()
            if not ok:
                try:
                    w.f.close()
                except OSError:
                    pass
                partial.unlink(missing_ok=True)
    os.replace(partial, out)
    print(f"done: {len(job.rows)} sections, {out.stat().st_size / (1 << 30):.3f} GiB, {time.time() - job.t0:.1f} s -> {out}")
