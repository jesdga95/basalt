"""Attach the bundled DeepSeek-V4 DSpark speculative drafter (`mtp.0`..`mtp.2`) to an existing
deepseek4 `.basalt` pack as the `dspark` section, without reconverting the main model weights.

Usage:
    python3 tools/convert/dspark_deepseek4_attach.py \
        --checkpoint HF_DIR --into PACK.basalt [--undo-dir DIR] [--workers N]
    python3 tools/convert/dspark_deepseek4_attach.py \
        --roll-back --into PACK.basalt --undo-dir DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import pathlib
import re
import struct
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
MTP_EXPERT_RE = re.compile(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
EXPERTS_PER_JOB = 16
ROWS_PER_JOB = 1 << 22

MTP_FIRST = {
    "main_norm.weight": ("main_norm.weight", "BF16_F32"),
    "main_proj.weight": ("main_proj.weight", "FP8_Q8"),
}

MTP_STAGE = {
    "attn_norm.weight": ("attn_norm.weight", "BF16_F32"),
    "ffn_norm.weight": ("ffn_norm.weight", "BF16_F32"),
    "hc_attn_base": ("hc_attn_base.weight", "F32"),
    "hc_attn_fn": ("hc_attn_fn.weight", "F32"),
    "hc_attn_scale": ("hc_attn_scale.weight", "F32"),
    "hc_ffn_base": ("hc_ffn_base.weight", "F32"),
    "hc_ffn_fn": ("hc_ffn_fn.weight", "F32"),
    "hc_ffn_scale": ("hc_ffn_scale.weight", "F32"),
    "attn.attn_sink": ("attn_sinks.weight", "F32"),
    "attn.q_norm.weight": ("attn_q_a_norm.weight", "BF16_F32"),
    "attn.kv_norm.weight": ("attn_kv_a_norm.weight", "BF16_F32"),
    "attn.wq_a.weight": ("attn_q_a.weight", "FP8_Q8"),
    "attn.wq_b.weight": ("attn_q_b.weight", "FP8_Q8"),
    "attn.wkv.weight": ("attn_kv.weight", "FP8_Q8"),
    "attn.wo_a.weight": ("attn_output_a.weight", "FP8_Q8"),
    "attn.wo_b.weight": ("attn_output_b.weight", "FP8_Q8"),
    "ffn.gate.weight": ("ffn_gate_inp.weight", "BF16_F32"),
    "ffn.gate.bias": ("exp_probs_b.bias", "F32"),
    "ffn.gate.bias_vl": ("exp_probs_b_vl.bias", "F32"),
    "ffn.shared_experts.w1.weight": ("ffn_gate_shexp.weight", "FP8_Q8"),
    "ffn.shared_experts.w2.weight": ("ffn_down_shexp.weight", "FP8_Q8"),
    "ffn.shared_experts.w3.weight": ("ffn_up_shexp.weight", "FP8_Q8"),
}

MTP_HEAD = {
    "hc_head_base": ("hc_head_base.weight", "F32"),
    "hc_head_fn": ("hc_head_fn.weight", "F32"),
    "hc_head_scale": ("hc_head_scale.weight", "F32"),
    "norm.weight": ("output_norm.weight", "BF16_F32"),
    "markov_head.markov_w1.weight": ("markov_w1.weight", "BF16"),
    "markov_head.markov_w2.weight": ("markov_w2.weight", "BF16_Q8"),
    "confidence_head.proj.weight": ("conf_proj.weight", "BF16_F32_1D"),
}


def _e8m0_to_f32(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    u8 = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    return np.ldexp(np.ones(shape, dtype=np.float32), u8.astype(np.int32) - 127)


def _bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _load_index(ckpt_dir: pathlib.Path) -> tuple[dict, dict[str, tuple[str, str, list[int], int, int]]]:
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    wmap = json.loads((ckpt_dir / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(v for k, v in wmap.items() if k.startswith("mtp.")))
    tensors: dict[str, tuple[str, str, list[int], int, int]] = {}
    for sh in shards:
        p = ckpt_dir / sh
        with open(p, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
        data_off = 8 + hlen
        for k, info in hdr.items():
            if not k.startswith("mtp."):
                continue
            s0, s1 = info["data_offsets"]
            tensors[k] = (str(p), info["dtype"], list(info["shape"]), data_off + s0, s1 - s0)
    return cfg, tensors


def _worker_q8(args):
    kind, wp, woff, cols, r0, nr, sp, soff, sn, ssh = args
    if kind == "fp8_q8":
        codes = np.frombuffer(E._pread_bytes(wp, woff + r0 * cols, nr * cols), dtype=np.uint8).reshape(nr, cols)
        scales = _e8m0_to_f32(E._pread_bytes(sp, soff, sn), tuple(ssh))
        x = E.fp8_block_dequant(codes, scales, (128, 128), row0=r0)
    else:
        raw = E._pread_bytes(wp, woff + r0 * cols * 2, nr * cols * 2)
        x = _bf16_to_f32(raw).reshape(nr, cols)
    return r0, S.q8_0(x)


def _worker_mxfp4(args):
    stage, e0, items = args
    out = bytearray()
    for role_entries in items:
        for cp, coff, cn, csh, sp, soff, sn, ssh in role_entries:
            codes = np.frombuffer(E._pread_bytes(cp, coff, cn), dtype=np.uint8).reshape(csh)
            scales = np.frombuffer(E._pread_bytes(sp, soff, sn), dtype=np.uint8).reshape(ssh)
            out += E.mxfp4_rows(codes, scales).tobytes()
    return stage, e0, bytes(out)


def _pad(f, align: int, pos: int) -> int:
    rem = pos % align
    if rem:
        n = align - rem
        f.write(b"\x00" * n)
        pos += n
    return pos


def attach(ckpt_dir: pathlib.Path, pack_path: pathlib.Path, undo_dir: pathlib.Path | None, workers: int) -> None:
    t0 = time.monotonic()
    cfg, tensors = _load_index(ckpt_dir)
    n_stages = int(cfg["num_nextn_predict_layers"])
    n_expert = int(cfg["n_routed_experts"])
    block_size = int(cfg["dspark_block_size"])
    target_layers = [int(x) for x in cfg["dspark_target_layer_ids"]]
    markov_rank = int(cfg["dspark_markov_rank"])
    noise_token_id = int(cfg["dspark_noise_token_id"])

    mf = ModelFile(str(pack_path))
    if SECTION in mf.sections:
        raise S.ConvertError(f"{pack_path} already has a '{SECTION}' section")

    dense_plan = []
    for sub, (dst_name, form) in MTP_FIRST.items():
        src = f"mtp.0.{sub}"
        if src not in tensors:
            raise S.ConvertError(f"missing DSpark input tensor: {src}")
        dense_plan.append((dst_name, form, src))
    for s in range(n_stages):
        pref = f"mtp.{s}."
        for sub, (dst_sub, form) in MTP_STAGE.items():
            src = pref + sub
            if src not in tensors:
                raise S.ConvertError(f"missing DSpark tensor: {src}")
            dense_plan.append((f"stage.{s}.{dst_sub}", form, src))
    last_pref = f"mtp.{n_stages - 1}."
    for sub, (dst_name, form) in MTP_HEAD.items():
        src = last_pref + sub
        if src not in tensors:
            raise S.ConvertError(f"missing DSpark head tensor: {src}")
        dense_plan.append((dst_name, form, src))

    pool = multiprocessing.Pool(workers) if workers > 1 else None
    meta_tensors: dict[str, dict] = {}
    experts_meta: list[dict] = []
    tmp_fd, tmp_name = tempfile.mkstemp(prefix="dspark_ds4_", suffix=".bin", dir=str(pack_path.parent))
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            pos = 0
            for dst_name, form, src in dense_plan:
                pos = _pad(out, DENSE_ALIGN, pos)
                wp, dt, shape, woff, wn = tensors[src]
                if form in ("FP8_Q8", "BF16_Q8"):
                    rows, cols = shape
                    sp, soff, sn, ssh = "", 0, 0, ()
                    if form == "FP8_Q8":
                        sk = src.replace(".weight", ".scale")
                        sp, _, ssh, soff, sn = tensors[sk]
                    step = max(1, ROWS_PER_JOB // max(1, cols))
                    jobs = [
                        ("fp8_q8" if form == "FP8_Q8" else "bf16_q8", wp, woff, cols, r0, min(step, rows - r0), sp, soff, sn, ssh)
                        for r0 in range(0, rows, step)
                    ]
                    parts = pool.map(_worker_q8, jobs) if pool else [_worker_q8(j) for j in jobs]
                    parts.sort(key=lambda x: x[0])
                    blob = b"".join(p[1] for p in parts)
                    type_id, ne = E.TYPE_ID["Q8_0"], [cols, rows]
                elif form == "BF16_F32":
                    blob = _bf16_to_f32(E._pread_bytes(wp, woff, wn)).tobytes()
                    type_id, ne = E.TYPE_ID["F32"], list(reversed(shape))
                elif form == "BF16_F32_1D":
                    blob = _bf16_to_f32(E._pread_bytes(wp, woff, wn)).reshape(-1).tobytes()
                    type_id, ne = E.TYPE_ID["F32"], [int(np.prod(shape))]
                elif form == "BF16":
                    blob = E._pread_bytes(wp, woff, wn)
                    type_id, ne = E.TYPE_ID["BF16"], list(reversed(shape))
                elif form == "F32":
                    blob = E._pread_bytes(wp, woff, wn)
                    type_id, ne = E.TYPE_ID["F32"], list(reversed(shape))
                else:
                    raise S.ConvertError(f"unknown form {form}")
                out.write(blob)
                meta_tensors[dst_name] = {"type_id": type_id, "shape": ne, "offset": pos, "bytes": len(blob)}
                pos += len(blob)

            pos = _pad(out, TENSOR_ALIGN, pos)
            for s in range(n_stages):
                stage_off = pos
                jobs = []
                for e0 in range(0, n_expert, EXPERTS_PER_JOB):
                    e1 = min(n_expert, e0 + EXPERTS_PER_JOB)
                    items = []
                    for e in range(e0, e1):
                        roles = []
                        for r in ("w1", "w3", "w2"):
                            cp, _, csh, coff, cn = tensors[f"mtp.{s}.ffn.experts.{e}.{r}.weight"]
                            sp, _, ssh, soff, sn = tensors[f"mtp.{s}.ffn.experts.{e}.{r}.scale"]
                            roles.append((cp, coff, cn, tuple(csh), sp, soff, sn, tuple(ssh)))
                        items.append(roles)
                    jobs.append((s, e0, items))
                res = pool.map(_worker_mxfp4, jobs) if pool else [_worker_mxfp4(j) for j in jobs]
                res.sort(key=lambda x: x[1])
                blob_bytes = 0
                for _, _, chunk in res:
                    out.write(chunk)
                    pos += len(chunk)
                    if not blob_bytes:
                        blob_bytes = len(chunk) // min(EXPERTS_PER_JOB, n_expert)
                experts_meta.append({
                    "stage": s, "gu_type": 39, "d_type": 39,
                    "offset": stage_off, "blob_bytes": blob_bytes, "n_expert": n_expert,
                })
        if pool:
            pool.close()
            pool.join()

        dspark_meta = {
            "version": 1,
            "num_stages": n_stages,
            "block_size": block_size,
            "target_layer_ids": target_layers,
            "markov_rank": markov_rank,
            "noise_token_id": noise_token_id,
            "tensors": meta_tensors,
            "experts": experts_meta,
        }
        new_meta = json.loads(mf.read("meta").decode("utf-8"))
        new_meta["dspark"] = dspark_meta
        actual_undo = str(undo_dir) if undo_dir else tempfile.mkdtemp(prefix="dspark_ds4_undo_", dir=str(pack_path.parent))
        append_section(str(pack_path), SECTION, tmp_name, json.dumps(new_meta).encode("utf-8"), actual_undo)
        dt = time.monotonic() - t0
        sz_gib = os.path.getsize(tmp_name) / (1 << 30)
        print(f"Attached '{SECTION}' ({sz_gib:.2f} GiB, {n_stages} stages) to {pack_path} in {dt:.1f}s")
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=pathlib.Path)
    ap.add_argument("--into", type=pathlib.Path, required=True)
    ap.add_argument("--undo-dir", type=pathlib.Path, default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    ap.add_argument("--roll-back", action="store_true")
    args = ap.parse_args()
    if args.roll_back:
        if not args.undo_dir:
            ap.error("--roll-back requires --undo-dir")
        undo_append(str(args.into), str(args.undo_dir))
        print(f"Rolled back {args.into} from {args.undo_dir}")
        return
    if not args.checkpoint:
        ap.error("--checkpoint is required")
    attach(args.checkpoint, args.into, args.undo_dir, args.workers)


if __name__ == "__main__":
    main()
