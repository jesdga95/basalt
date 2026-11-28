"""Attach the bundled ERNIE-5.5-Flash DSpark speculative drafter (`model.mtp.*`) to an existing
ernie55 `.basalt` pack as the `dspark` section, without reconverting the main model weights.

Usage:
    python3 tools/convert/dspark_ernie55_attach.py \\
        --checkpoint HF_DIR --into PACK.basalt [--undo-dir DIR] [--workers N]
    python3 tools/convert/dspark_ernie55_attach.py \\
        --roll-back --into PACK.basalt --undo-dir DIR

Layout of the `dspark` section:
  1. Dense / attention / mHC / Markov / confidence tensors (each 64-byte aligned), indexed in
     `meta["dspark"]["tensors"]` with `{type_id, shape, offset, bytes}` (GGUF innermost-first
     shape, `offset` relative to the start of the `dspark` section):
       - FP8 128x128 projections -> exact f32 dequant -> Q8_0 (type 8)
       - markov_w1.weight [rank, vocab] BF16 -> transposed [vocab, rank] BF16 (for O(1) row gather)
       - markov_w2.weight [vocab, rank] BF16 -> Q8_0 (type 8, fast GPU MMVQ)
       - confidence_head.projection.weight [1, n_embd + rank] BF16 -> F32 [n_embd + rank]
       - norms, mHC mixers (transposed to [out, in]) / biases / stacked scales, VHA PostMix
         (folded I + V @ U^T), attention sinks, split-feature router [w0; w1], router bias &
         per-expert scale -> F32
  2. Padded to 4096 bytes: contiguous MXFP4 (type 39) routed expert blobs for all stages
     `s = 0..num_dspark_layers-1`, `e = 0..n_routed_experts-1`, each `[w1 | w3 | w2]` bit-exact
     (same format as main `experts` section). Indexed in `meta["dspark"]["experts"]`.

No existing section moves or changes; `--roll-back` restores the pack bit for bit.
"""
from __future__ import annotations

import argparse
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

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import checkpoint_ernie55 as E  # noqa: E402
import model_sections as S  # noqa: E402
from model_file import TENSOR_ALIGN, ModelFile, append_section, undo_append  # noqa: E402

SECTION = "dspark"
DENSE_ALIGN = 64
MTP_EXPERT_RE = re.compile(
    r"^model\.mtp\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.(weight|weight_scale_inv)$"
)
EXPERTS_PER_JOB = 16
ROWS_PER_JOB = 1 << 22

MTP_GLOBAL = {
    "model.mtp.feature_projection.weight": ("feature_proj.weight", "FP8_Q8"),
    "model.mtp.feature_norm.weight": ("feature_norm.weight", "F32"),
    "model.mtp.mhc_contract.hc_head_fn": ("hc_head_fn.weight", "F32_T"),
    "model.mtp.mhc_contract.hc_head_base": ("hc_head_base.weight", "F32"),
    "model.mtp.mhc_contract.hc_head_scale": ("hc_head_scale.weight", "F32"),
    "model.mtp.norm.weight": ("output_norm.weight", "F32"),
    "model.mtp.markov_head.markov_w1.weight": ("markov_w1.weight", "BF16_T"),
    "model.mtp.markov_head.markov_w2.weight": ("markov_w2.weight", "BF16_Q8"),
    "model.mtp.confidence_head.projection.weight": ("conf_proj.weight", "F32_1D"),
}

MTP_STAGE = {
    "input_layernorm.weight": ("attn_norm.weight", "F32"),
    "post_attention_layernorm.weight": ("ffn_norm.weight", "F32"),
    "self_attention_hyper_connection.mapping_proj.weight": ("hc_attn_fn.weight", "F32_T"),
    "self_attention_hyper_connection.bias": ("hc_attn_base.weight", "F32"),
    "mlp_hyper_connection.mapping_proj.weight": ("hc_ffn_fn.weight", "F32_T"),
    "mlp_hyper_connection.bias": ("hc_ffn_base.weight", "F32"),
    "self_attn.gate_proj.weight": ("attn_gate.weight", "FP8_Q8"),
    "self_attn.linear_q_down_proj.weight": ("attn_q_a.weight", "FP8_Q8"),
    "self_attn.q_layernorm.weight": ("attn_q_a_norm.weight", "F32"),
    "self_attn.linear_q_up_proj.weight": ("attn_q_b.weight", "FP8_Q8"),
    "self_attn.linear_kv_proj.weight": ("attn_kv.weight", "FP8_Q8"),
    "self_attn.kv_layernorm.weight": ("attn_kv_a_norm.weight", "F32"),
    "self_attn.linear_o_group_proj": ("attn_output_a.weight", "FP8_Q8"),
    "self_attn.o_proj.weight": ("attn_output_b.weight", "FP8_Q8"),
    "self_attn.core_attention.attn_sink": ("attn_sinks.weight", "F32"),
    "block_sparse_moe.e_score_correction_bias": ("exp_probs_b.bias", "F32"),
    "block_sparse_moe.gate.routed_scaling_factor_param": ("exp_probs_scale.weight", "F32"),
    "block_sparse_moe.fc1_latent_proj.weight": ("ffn_routed_down.weight", "FP8_Q8"),
    "block_sparse_moe.latent_norm.weight": ("ffn_latent_norm.weight", "F32"),
    "block_sparse_moe.fc2_latent_proj.weight": ("ffn_routed_up.weight", "FP8_Q8"),
    "block_sparse_moe.shared_experts.w1.weight": ("ffn_gate_shexp.weight", "FP8_Q8"),
    "block_sparse_moe.shared_experts.w3.weight": ("ffn_up_shexp.weight", "FP8_Q8"),
    "block_sparse_moe.shared_experts.w2.weight": ("ffn_down_shexp.weight", "FP8_Q8"),
}


def die(msg: str):
    raise SystemExit(f"dspark-ernie55: {msg}")


def _worker(job):
    kind = job[0]
    try:
        if kind == "fp8_q8":
            _, wp, woff, rows, cols, sp, soff, sdt, ssh, sn, block, r0, nr = job
            codes = np.frombuffer(E._pread_bytes(wp, woff + r0 * cols, nr * cols), dtype=np.uint8).reshape(nr, cols)
            sraw = E._pread_bytes(sp, soff, sn)
            if sdt == "F32":
                scales = np.frombuffer(sraw, dtype="<f4").reshape(ssh)
            else:
                scales = S._bf16_to_f32(np.ascontiguousarray(np.frombuffer(sraw, dtype="<u2").reshape(ssh)))
            x = E.fp8_block_dequant(codes, scales, block, row0=r0)
            return S.q8_0(x)
        if kind == "bf16_q8":
            _, wp, woff, cols, r0, nr = job
            raw = E._pread_bytes(wp, woff + r0 * cols * 2, nr * cols * 2)
            u16 = np.frombuffer(raw, dtype="<u2").reshape(nr, cols)
            x = S._bf16_to_f32(np.ascontiguousarray(u16))
            if not np.isfinite(x).all():
                return "non-finite BF16 value in markov_w2"
            return S.q8_0(x)
        if kind == "mxfp4":
            _, batch = job
            out = bytearray()
            for items in batch:
                for cp, coff, cn, csh, sp, soff, sn, ssh in items:
                    codes = np.frombuffer(E._pread_bytes(cp, coff, cn), dtype=np.uint8).reshape(csh)
                    scales = np.frombuffer(E._pread_bytes(sp, soff, sn), dtype=np.uint8).reshape(ssh)
                    out += E.mxfp4_rows(codes, scales).tobytes()
            return bytes(out)
        return f"unknown job kind {kind}"
    except Exception as ex:
        return f"{type(ex).__name__}: {ex}"


def _run_jobs(pool, jobs, depth: int = 32):
    if pool is None:
        for j in jobs:
            b = _worker(j)
            if isinstance(b, str):
                raise S.ConvertError(b)
            yield b
        return
    it = iter(jobs)
    pending = []
    for _ in range(depth):
        j = next(it, None)
        if j is None:
            break
        pending.append(pool.apply_async(_worker, (j,)))
    while pending:
        b = pending.pop(0).get()
        if isinstance(b, str):
            raise S.ConvertError(b)
        yield b
        j = next(it, None)
        if j is not None:
            pending.append(pool.apply_async(_worker, (j,)))


class DsparkPlan:
    """Validates and stages all `model.mtp.*` dense tensors and MXFP4 experts from an ERNIE-5.5 HF checkpoint."""

    def __init__(self, ck: E.ErnieCheckpoint):
        c = ck.config
        self.ck = ck
        stage_ids = sorted({
            int(m.group(1))
            for k in ck.order
            if (m := re.match(r"^model\.mtp\.(\d+)\.", k)) is not None
        })
        self.n_stages = int(c.get("num_dspark_layers") or (max(stage_ids) + 1 if stage_ids else 0))
        if self.n_stages <= 0 or stage_ids != list(range(self.n_stages)):
            raise S.ConvertError(f"cannot determine contiguous model.mtp stages (found {stage_ids})")
        self.block_size = int(c.get("dspark_block_size") or 7)
        self.noise_token_id = int(c.get("dspark_noise_token_id") or 1024)
        self.markov_rank = int(c.get("dspark_markov_rank") or 256)
        self.target_layers = [int(x) for x in (c.get("dspark_target_layer_ids") or [])]
        if not self.target_layers:
            raise S.ConvertError("config.json: dspark_target_layer_ids missing")
        self.window = int(c.get("sliding_window") or c.get("sliding_window_size") or 128)
        self.rope_theta = float(c.get("rope_theta") or 8000000.0)
        self.n_embd = int(c["hidden_size"])
        self.n_vocab = int(c["vocab_size"])
        self.n_heads = int(c["num_attention_heads"])
        self.n_expert = int(c["n_routed_experts"])
        self.n_ff_exp = int(c["moe_intermediate_size"])
        self.n_latent = int(c["moe_latent_size"])
        qc = c.get("quantization_config") or {}
        wb = qc.get("weight_block_size") or [128, 128]
        self.fp8_block = (int(wb[0]), int(wb[1]))

        self.tensors: dict[str, dict] = {}
        unknown: list[str] = []
        scale_keys: set[str] = set()
        for k in ck.order:
            if not k.startswith("model.mtp."):
                continue
            if MTP_EXPERT_RE.match(k):
                continue
            if k.endswith("_scale_inv"):
                scale_keys.add(k)
                continue
            if k in MTP_GLOBAL:
                gname, form = MTP_GLOBAL[k]
                self._add(gname, form, k)
                continue
            m = re.fullmatch(r"model\.mtp\.(\d+)\.(.+)", k)
            if not m:
                unknown.append(k)
                continue
            s, sub = int(m.group(1)), m.group(2)
            if s < 0 or s >= self.n_stages:
                raise S.ConvertError(f"stage {s} out of range 0..{self.n_stages - 1} in {k}")
            if sub in E.CONSUMED_SUFFIXES:
                continue
            rule = MTP_STAGE.get(sub)
            if rule is None:
                unknown.append(k)
                continue
            self._add(f"blk.{s}.{rule[0]}", rule[1], k)

        if unknown:
            raise S.ConvertError("unexpected model.mtp tensors: " + ", ".join(unknown[:8]))

        # Composite per-stage tensors (mHC alpha scales, VHA folded mix, split-feature router)
        for s in range(self.n_stages):
            for src_pref, dst_name in E.HC_ALPHAS:
                srcs = tuple(f"model.mtp.{s}.{src_pref}.alpha_{a}" for a in ("pre", "post", "res"))
                for sk in srcs:
                    ck.need(sk, ("F32", "BF16"), (1,))
                self.tensors[f"blk.{s}.{dst_name}"] = {
                    "form": "hc_scale", "type_id": E.TYPE_ID["F32"], "ne": [3], "src": srcs,
                }
            u_k = f"model.mtp.{s}.self_attn.vha_postmix_U"
            v_k = f"model.mtp.{s}.self_attn.vha_postmix_V"
            _, _, _, ush, _ = ck.need(u_k, ("BF16", "F32"))
            _, _, _, vsh, _ = ck.need(v_k, ("BF16", "F32"))
            if ush[0] != self.n_heads or vsh != ush:
                raise S.ConvertError(f"mtp stage {s}: vha_postmix shapes {ush} / {vsh}")
            self.tensors[f"blk.{s}.attn_vha_mix.weight"] = {
                "form": "vha_fold", "type_id": E.TYPE_ID["F32"], "ne": [self.n_heads, self.n_heads], "src": (u_k, v_k),
            }
            g0 = f"model.mtp.{s}.block_sparse_moe.gate.weight"
            g1 = f"model.mtp.{s}.block_sparse_moe.gate.weight_1"
            ck.need(g0, ("BF16", "F32"), (self.n_expert, self.n_embd))
            ck.need(g1, ("BF16", "F32"), (self.n_expert, self.n_embd))
            self.tensors[f"blk.{s}.ffn_gate_inp.weight"] = {
                "form": "gate_concat", "type_id": E.TYPE_ID["F32"], "ne": [self.n_embd, 2 * self.n_expert], "src": (g0, g1),
            }

        used_scales = {t["scale_src"] for t in self.tensors.values() if t.get("scale_src")}
        stray = sorted(scale_keys - used_scales)
        if stray:
            raise S.ConvertError("unpaired model.mtp FP8 scale tensors: " + ", ".join(stray[:6]))

        # Verify completeness
        need = [r[0] for r in MTP_GLOBAL.values()]
        for s in range(self.n_stages):
            for r in MTP_STAGE.values():
                need.append(f"blk.{s}.{r[0]}")
            need.extend([
                f"blk.{s}.hc_attn_scale.weight", f"blk.{s}.hc_ffn_scale.weight",
                f"blk.{s}.attn_vha_mix.weight", f"blk.{s}.ffn_gate_inp.weight",
            ])
        missing = [n for n in need if n not in self.tensors]
        if missing:
            raise S.ConvertError(f"model.mtp lacks {len(missing)} required tensors: " + ", ".join(missing[:8]))

        # Verify all stage MXFP4 experts
        self.shapes = {
            "w1": (self.n_ff_exp, self.n_latent),
            "w3": (self.n_ff_exp, self.n_latent),
            "w2": (self.n_latent, self.n_ff_exp),
        }
        self.blob_bytes = sum(r * (c // 32) * 17 for r, c in self.shapes.values())
        self.stage_stride = self.n_expert * self.blob_bytes
        for s in range(self.n_stages):
            for e in range(self.n_expert):
                for role in E.ROLES:
                    rows, cols = self.shapes[role]
                    base = f"model.mtp.{s}.block_sparse_moe.experts.{e}.{role}"
                    ck.need(f"{base}.weight", ("I8", "U8"), (rows, cols // 2))
                    ck.need(f"{base}.weight_scale_inv", ("F8_E8M0", "U8"), (rows, cols // 32))

    def _add(self, gname: str, form: str, src: str):
        _, _, dt, shape, _ = self.ck.t[src]
        if gname in self.tensors:
            raise S.ConvertError(f"duplicate mtp target {gname}")
        if form == "FP8_Q8":
            if dt != "F8_E4M3" or len(shape) != 2 or shape[1] % 32 != 0:
                raise S.ConvertError(f"{src}: expected 2-D F8_E4M3 with cols%32==0, got {dt} {shape}")
            sk = E.scale_name_for(src)
            bn, bk = self.fp8_block
            self.ck.need(sk, ("F32", "BF16"), ((shape[0] + bn - 1) // bn, (shape[1] + bk - 1) // bk))
            self.tensors[gname] = {
                "form": "fp8_q8", "type_id": E.TYPE_ID["Q8_0"], "ne": [shape[1], shape[0]],
                "src": src, "scale_src": sk,
            }
        elif form == "BF16_Q8":
            if dt != "BF16" or len(shape) != 2 or shape[1] % 32 != 0:
                raise S.ConvertError(f"{src}: expected 2-D BF16 with cols%32==0, got {dt} {shape}")
            self.tensors[gname] = {
                "form": "bf16_q8", "type_id": E.TYPE_ID["Q8_0"], "ne": [shape[1], shape[0]], "src": src,
            }
        elif form == "BF16_T":
            if dt != "BF16" or len(shape) != 2:
                raise S.ConvertError(f"{src}: expected 2-D BF16, got {dt} {shape}")
            # Checkpoint [rank, vocab] -> transposed row-major [vocab, rank] -> GGUF ne = [rank, vocab]
            self.tensors[gname] = {
                "form": "bf16_t", "type_id": E.TYPE_ID["BF16"], "ne": [shape[0], shape[1]], "src": src,
            }
        elif form == "F32_T":
            if dt not in ("F32", "BF16") or len(shape) != 2:
                raise S.ConvertError(f"{src}: expected 2-D F32/BF16, got {dt} {shape}")
            self.tensors[gname] = {
                "form": "f32_t", "type_id": E.TYPE_ID["F32"], "ne": [shape[0], shape[1]], "src": src,
            }
        elif form == "F32_1D":
            if dt not in ("F32", "BF16"):
                raise S.ConvertError(f"{src}: expected F32/BF16, got {dt}")
            n = int(np.prod(shape))
            self.tensors[gname] = {
                "form": "f32_1d", "type_id": E.TYPE_ID["F32"], "ne": [n], "src": src,
            }
        elif form == "F32":
            if dt not in ("F32", "BF16"):
                raise S.ConvertError(f"{src}: expected F32/BF16, got {dt}")
            self.tensors[gname] = {
                "form": "f32", "type_id": E.TYPE_ID["F32"], "ne": list(reversed(shape)), "src": src,
            }
        else:
            raise S.ConvertError(f"unknown form {form}")

    def _small_bytes(self, t: dict) -> bytes:
        form = t["form"]
        if form == "bf16_t":
            _, _, _, shape, _ = self.ck.t[t["src"]]
            raw = self.ck.raw(t["src"])
            u16 = np.frombuffer(raw, dtype="<u2").reshape(shape)
            f32 = S._bf16_to_f32(np.ascontiguousarray(u16))
            if not np.isfinite(f32).all():
                raise S.ConvertError(f"{t['src']} has a non-finite BF16 value")
            return np.ascontiguousarray(u16.T, dtype="<u2").tobytes()
        if form == "f32":
            x = self.ck.values(t["src"])
        elif form == "f32_1d":
            x = self.ck.values(t["src"]).reshape(-1)
        elif form == "f32_t":
            x = np.ascontiguousarray(self.ck.values(t["src"]).T, dtype=np.float32)
        elif form == "hc_scale":
            x = np.concatenate([self.ck.values(s).reshape(1) for s in t["src"]]).astype(np.float32)
        elif form == "vha_fold":
            u = self.ck.values(t["src"][0]).astype(np.float32)
            v = self.ck.values(t["src"][1]).astype(np.float32)
            nh = u.shape[0]
            x = (np.eye(nh, dtype=np.float32) + v @ u.T).astype(np.float32)
        elif form == "gate_concat":
            g0 = self.ck.values(t["src"][0]).astype(np.float32)
            g1 = self.ck.values(t["src"][1]).astype(np.float32)
            x = np.ascontiguousarray(np.concatenate([g0, g1], axis=0), dtype=np.float32)
        else:
            raise S.ConvertError(f"unexpected small form {form}")
        if not np.isfinite(x).all():
            raise S.ConvertError(f"non-finite value in {t['src']}")
        return np.ascontiguousarray(x, dtype="<f4").tobytes()

    def write_blob(self, out_path: pathlib.Path, workers: int) -> dict:
        """Writes the complete `dspark` section blob to `out_path` and returns the `meta["dspark"]` dict."""
        t0 = time.monotonic()
        pool = multiprocessing.Pool(workers) if workers > 1 else None
        tensor_meta: dict[str, dict] = {}
        try:
            with out_path.open("wb") as f:
                pos = 0

                def pad_to(align: int):
                    nonlocal pos
                    rem = (-pos) % align
                    if rem:
                        f.write(b"\0" * rem)
                        pos += rem

                for name in sorted(self.tensors):
                    t = self.tensors[name]
                    pad_to(DENSE_ALIGN)
                    off = pos
                    form = t["form"]
                    if form == "fp8_q8":
                        wp, woff, _, wsh, _ = self.ck.t[t["src"]]
                        sp, soff, sdt, ssh, sn = self.ck.t[t["scale_src"]]
                        rows, cols = wsh
                        step = max(self.fp8_block[0], (ROWS_PER_JOB // cols // self.fp8_block[0]) * self.fp8_block[0])
                        jobs = (
                            ("fp8_q8", str(wp), woff, rows, cols, str(sp), soff, sdt, ssh, sn,
                             self.fp8_block, r0, min(step, rows - r0))
                            for r0 in range(0, rows, step)
                        )
                        for b in _run_jobs(pool, jobs):
                            f.write(b)
                            pos += len(b)
                    elif form == "bf16_q8":
                        wp, woff, _, wsh, _ = self.ck.t[t["src"]]
                        rows, cols = wsh
                        step = max(1, ROWS_PER_JOB // cols)
                        jobs = (
                            ("bf16_q8", str(wp), woff, cols, r0, min(step, rows - r0))
                            for r0 in range(0, rows, step)
                        )
                        for b in _run_jobs(pool, jobs):
                            f.write(b)
                            pos += len(b)
                    else:
                        b = self._small_bytes(t)
                        f.write(b)
                        pos += len(b)
                    nbytes = pos - off
                    tensor_meta[name] = {
                        "type_id": t["type_id"],
                        "shape": t["ne"],
                        "offset": off,
                        "bytes": nbytes,
                    }

                dense_end = pos
                pad_to(TENSOR_ALIGN)
                exp_off = pos

                def expert_jobs():
                    for s in range(self.n_stages):
                        for e0 in range(0, self.n_expert, EXPERTS_PER_JOB):
                            batch = []
                            for e in range(e0, min(self.n_expert, e0 + EXPERTS_PER_JOB)):
                                items = []
                                for role in E.ROLES:
                                    base = f"model.mtp.{s}.block_sparse_moe.experts.{e}.{role}"
                                    cp, coff, _, csh, cn = self.ck.t[f"{base}.weight"]
                                    sp, soff, _, ssh, sn = self.ck.t[f"{base}.weight_scale_inv"]
                                    items.append((str(cp), coff, cn, csh, str(sp), soff, sn, ssh))
                                batch.append(items)
                            yield ("mxfp4", batch)

                for b in _run_jobs(pool, expert_jobs()):
                    f.write(b)
                    pos += len(b)
                exp_bytes = pos - exp_off
                expected_exp = self.n_stages * self.stage_stride
                if exp_bytes != expected_exp:
                    raise S.ConvertError(f"expert bytes {exp_bytes} != expected {expected_exp}")
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        dt = time.monotonic() - t0
        print(
            f"dspark blob written: {pos / (1 << 30):.2f} GiB "
            f"(dense {dense_end / (1 << 20):.1f} MiB, experts {exp_bytes / (1 << 30):.2f} GiB) in {dt:.1f}s",
            flush=True,
        )
        return {
            "section": SECTION,
            "format": "ernie55_dspark_v1",
            "bytes": pos,
            "layers": self.n_stages,
            "block_size": self.block_size,
            "noise_token_id": self.noise_token_id,
            "markov_rank": self.markov_rank,
            "target_layers": self.target_layers,
            "window": self.window,
            "rope_theta": self.rope_theta,
            "tensors": tensor_meta,
            "experts": {
                "offset": exp_off,
                "bytes": exp_bytes,
                "gu_type": E.MXFP4,
                "d_type": E.MXFP4,
                "n_expert": self.n_expert,
                "blob_bytes": self.blob_bytes,
                "stage_stride": self.stage_stride,
            },
        }


def attach_dspark(checkpoint_dir: str | pathlib.Path, pack_path: str | pathlib.Path,
                  undo_dir: str | pathlib.Path | None = None, workers: int = 8) -> dict:
    pack = pathlib.Path(pack_path)
    if not pack.is_file():
        die(f"missing pack: {pack}")
    m = ModelFile(str(pack))
    meta = json.loads(m.read("meta"))
    if meta.get("arch") != "ernie55":
        die(f"{pack} is not an ernie55 pack (arch={meta.get('arch')!r})")
    if SECTION in m.names() or meta.get("dspark"):
        die(f"{pack} already has a dspark section")

    ck = E.ErnieCheckpoint(checkpoint_dir)
    plan = DsparkPlan(ck)
    kv = meta.get("gguf_kv") or {}
    if int(kv.get("ernie55.embedding_length", 0)) != plan.n_embd:
        die("checkpoint hidden_size does not match pack")
    if int(kv.get("ernie55.vocab_size", 0)) != plan.n_vocab:
        die("checkpoint vocab_size does not match pack")

    tmp_dir = pack.parent
    work_dir = pathlib.Path(tempfile.mkdtemp(prefix=".dspark_work_", dir=str(tmp_dir)))
    tmp_blob = work_dir / "blob.bin"
    owned_undo = undo_dir is None
    actual_undo = str(undo_dir) if undo_dir is not None else str(work_dir / "undo")
    try:
        dspark_meta = plan.write_blob(tmp_blob, max(1, workers))
        meta["dspark"] = dspark_meta
        new_meta = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        append_section(str(pack), SECTION, str(tmp_blob), new_meta, actual_undo)
    finally:
        if tmp_blob.exists():
            tmp_blob.unlink()
        if not owned_undo and work_dir.exists():
            try:
                work_dir.rmdir()
            except OSError:
                pass

    m2 = ModelFile(str(pack))
    if not m2.verify(SECTION) or not m2.verify("meta"):
        die(f"{pack}: verification failed after append; restore with --roll-back --undo-dir {actual_undo}")
    if owned_undo:
        if os.path.isdir(actual_undo):
            for fn in os.listdir(actual_undo):
                os.remove(os.path.join(actual_undo, fn))
            os.rmdir(actual_undo)
        if work_dir.exists():
            work_dir.rmdir()
    print(f"done: {pack} is {pack.stat().st_size / (1 << 30):.3f} GiB, sections={m2.names()}", flush=True)
    return dspark_meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Attach ERNIE-5.5 DSpark drafter (`model.mtp.*`) to an ernie55 .basalt pack")
    ap.add_argument("--checkpoint", help="HF snapshot directory containing config.json and model-*.safetensors")
    ap.add_argument("--into", required=True, help="Existing ernie55 .basalt pack")
    ap.add_argument("--undo-dir", help="Directory to store rollback state (created new); optional for attach, required for --roll-back")
    ap.add_argument("--roll-back", action="store_true", help="Restore the pack bit-for-bit using --undo-dir")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4), help="Worker processes (default: min(16, cpu_count))")
    a = ap.parse_args(argv)
    if a.roll_back:
        if not a.undo_dir:
            die("--roll-back requires --undo-dir")
        undo_append(a.into, a.undo_dir)
        print(f"rolled back: {a.into}", flush=True)
        return 0
    if not a.checkpoint:
        die("--checkpoint is required")
    attach_dspark(a.checkpoint, a.into, undo_dir=a.undo_dir, workers=a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
