"""An inkling2 pack (Inkling 2 Mini) from an inkling2_loop GGUF (`general.architecture = inkling2_loop`): BF16, F16,
Q8_0, or a mixed-type quantization such as ISTA-DASLab's GSQ-RCO builds (IQ/Q2_0 experts, K-quant attention), for
tools/convert/convert.py.

  experts             routed experts per layer, [gate | up | down] (llama.cpp's fused gate_up is gate rows, then up rows):
                        BF16/F16 source:   NVFP4 (type 40) blocks + FP32 scale tail {s_gate, s_up, s_down, 0}
                        quantized source:  the GGUF's blocks as stored, each layer in its own gate_up and down types
                                           (STORED_EXPERT_TYPES), no scale tail
  experts.layout      one `layer gu_type d_type offset blob_bytes` line per layer
  gguf                all other tensors in Basalt's inkling2 names:
                        attention projections -> Q8_0 (passthrough if Q8_0, else ggml reference rounding of the
                                                 BF16/F16 values or of the dequantized K-quant/IQ4_XS values)
                        token_embd, output    -> BF16 (dequantized if quantized; trimmed to vocab_size_unpadded)
                        loop_*_ctrl_*         -> BF16 (rounded from F32/F16 if needed)
                        norms, convs, clips, router, bias, queries -> F32
  vision/mmproj       copied from --mmproj or --sections-from if supplied
  tokenizer/*, meta   extracted from the source GGUF

Needs numpy.
"""
from __future__ import annotations

import json
import multiprocessing
import pathlib
import re
import sys
import tempfile
import time

import numpy as np

import checkpoint as C
import hy4flash_bf16 as H
import model_sections as S

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ggufq  # noqa: E402
from gguf_reader import BLOCK_GEOMETRY, GGUFFile, TensorInfo  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import ModelFile, TENSOR_ALIGN, ModelWriter, Stream  # noqa: E402

ARCH = "inkling2"
TYPE_ID = {"F32": 0, "Q8_0": 8, "BF16": 30, "NVFP4": 40}
Q8_0 = 8
NVFP4 = 40
EXPERTS_PER_JOB = 16
FLOAT_TYPES = ("F32", "BF16", "F16")
# Expert encodings copied as stored: the engine's native expert kernels (CUDA kernels::native_expert_layout and the CPU
# cpu::native_fmt) read them in place, gate/up and down each in its own type.
STORED_EXPERT_TYPES = frozenset({
    "Q8_0", "Q2_0", "IQ4_NL", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S",
})
# Quantized types tools/ggufq.py dequantizes (dense tensors are rounded from their values to the engine's types).
DEQUANT_TYPES = frozenset(t.name for t in ggufq.GGMLQuantizationType if t.name not in FLOAT_TYPES)

# Attention projection suffixes (after `blk.N.`) that the inkling2 engine requires in Q8_0.
Q8_SUFFIXES = frozenset({
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_rel.weight",
    "attn_gate.weight",
    "attn_sink.weight",
    "attn_output.weight",
    "attn_k_conv_proj.weight",
    "attn_v_conv_proj.weight",
})

# GGUF tensor name rewrites (suffix after `blk.N.`).
RENAME_SUFFIX = {
    "attn_sink_proj.weight": "attn_sink.weight",
}

CTRL_RE = re.compile(r"^loop_(q|k|v|ffn)_ctrl_(down|gain|forget)\.weight$")
EXPERT_RE = re.compile(r"^blk\.(\d+)\.(ffn_gate_up_exps|ffn_down_exps)\.weight$")


def _f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 array to BF16 (uint16) using round-to-nearest-even."""
    u = np.ascontiguousarray(x, dtype="<f4").view("<u4")
    # Preserve NaN payloads; round-to-nearest-even for finite/inf
    lsb = (u >> 16) & 1
    rounded = u + np.uint32(0x7FFF) + lsb
    is_nan = (u & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    out = (rounded >> 16).astype("<u2")
    if is_nan.any():
        out = np.where(is_nan, ((u >> 16) | np.uint32(0x0040)).astype("<u2"), out)
    return out


def _read_f32(path: str, offset: int, nbytes: int, type_name: str) -> np.ndarray:
    buf = H.pread(path, offset, nbytes)
    if type_name == "F32":
        return np.frombuffer(buf, dtype="<f4").copy()
    if type_name == "BF16":
        return H.bf16_values(buf)
    if type_name == "F16":
        return np.frombuffer(buf, dtype="<f2").astype(np.float32)
    if type_name in DEQUANT_TYPES:
        raw = np.frombuffer(buf, dtype=np.uint8)
        return ggufq.dequantize(raw, ggufq.GGMLQuantizationType[type_name]).astype(np.float32).reshape(-1)
    raise S.ConvertError(f"unsupported tensor type {type_name} for float read")


def _row_bytes(type_name: str, cols: int) -> int:
    """Bytes of one row of `cols` values in a ggml type."""
    block_elems, block_bytes = BLOCK_GEOMETRY[type_name]
    if cols % block_elems:
        raise S.ConvertError(f"{type_name} width {cols} is not a multiple of {block_elems}")
    return cols // block_elems * block_bytes


def _read_bf16_bytes(path: str, offset: int, nbytes: int, type_name: str) -> bytes:
    if type_name == "BF16":
        return bytes(H.pread(path, offset, nbytes))
    # whole blocks at a time (a block dequantizes on its own), so a quantized output head never sits in RAM as F32
    step = BLOCK_GEOMETRY[type_name][1] * ((64 << 20) // BLOCK_GEOMETRY[type_name][1])
    return b"".join(_f32_to_bf16(_read_f32(path, offset + o, min(step, nbytes - o), type_name)).tobytes()
                    for o in range(0, nbytes, step))


def _expert_nvfp4_job(args):
    """Quantize experts [e0, e1) of one layer from stacked BF16/F16 [gate; up] and down to NVFP4."""
    gu_path, gu_off, gu_type, dn_path, dn_off, dn_type, inter, hidden, e0, e1 = args
    eb_bps = 2  # BF16 or F16
    gu_eb = 2 * inter * hidden * eb_bps
    dn_eb = hidden * inter * eb_bps
    out = []
    for e in range(e0, e1):
        gu = _read_f32(gu_path, gu_off + e * gu_eb, gu_eb, gu_type).reshape(2 * inter, hidden)
        dn = _read_f32(dn_path, dn_off + e * dn_eb, dn_eb, dn_type).reshape(hidden, inter)
        gate, up = gu[:inter], gu[inter:]
        tail = []
        for mat in (gate, up, dn):
            if not np.isfinite(mat).all():
                return f"expert {e}: a non-finite value"
            gs = H.Q.fp4q_global(float(np.abs(mat).max()))
            codes, sc = H.Q.fp4q_rows(mat, gs)
            out.append(C.nvfp4_rows(codes, sc).tobytes())
            tail.append(gs)
        out.append(np.array(tail + [0.0], dtype="<f4").tobytes())
    return b"".join(out)


def _expert_stored_job(args):
    """Copy experts [e0, e1) of one layer as stored: [gate | up | down] blobs (gate_up is gate rows, then up rows)."""
    gu_path, gu_off, gu_eb, dn_path, dn_off, dn_eb, e0, e1 = args
    out = []
    for e in range(e0, e1):
        out.append(bytes(H.pread(gu_path, gu_off + e * gu_eb, gu_eb)))
        out.append(bytes(H.pread(dn_path, dn_off + e * dn_eb, dn_eb)))
    return b"".join(out)


def _scalar(val):
    if isinstance(val, list):
        if not val:
            raise S.ConvertError("empty per-layer metadata array")
        if len(set(val)) != 1:
            raise S.ConvertError(f"varying per-layer metadata {val[:4]}... is not supported")
        return val[0]
    return val


def _map_kv(src_kv: dict, name: str, expert_type: int) -> dict:
    """Translate `inkling2_loop.*` GGUF metadata to Basalt's `inkling2.*` metadata."""
    src_arch = src_kv.get("general.architecture", "inkling2_loop")
    P = src_arch + "."
    A = ARCH + "."

    def req(key: str):
        if P + key not in src_kv:
            raise S.ConvertError(f"missing GGUF metadata key {P + key}")
        return src_kv[P + key]

    n_layers = int(req("block_count"))
    n_head = int(_scalar(req("attention.head_count")))
    n_head_kv = int(_scalar(req("attention.head_count_kv")))
    head_dim = int(req("attention.key_length"))
    if P + "attention.key_length_swa" in src_kv and int(src_kv[P + "attention.key_length_swa"]) != head_dim:
        raise S.ConvertError("sliding-window key_length differs from full key_length")
    rel_extent = int(req("attention.relative_extent"))
    swa_window = int(req("attention.sliding_window"))
    if rel_extent != swa_window:
        raise S.ConvertError("attention.relative_extent differs from attention.sliding_window")

    gate_cap = float(src_kv.get(P + "activation.situ_beta", src_kv.get(P + "expert_gate_softcap", 4.0)))
    up_cap = float(src_kv.get(P + "activation.situ_linear_beta", src_kv.get(P + "expert_up_softcap", 25.0)))

    kv: dict = {
        "general.architecture": ARCH,
        "general.type": "model",
        "general.name": name,
        "general.file_type": expert_type,
        "general.quantization_version": 2,
    }
    for k in ("general.sampling.temp", "general.license", "general.license.link"):
        if k in src_kv:
            kv[k] = src_kv[k]

    kv.update({
        A + "block_count": n_layers,
        A + "context_length": int(req("context_length")),
        A + "embedding_length": int(req("embedding_length")),
        A + "vocab_size": int(req("vocab_size")),
        A + "attention.head_count": n_head,
        A + "attention.head_count_kv": n_head_kv,
        A + "attention.key_length": head_dim,
        A + "attention.layer_norm_rms_epsilon": C.f32(req("attention.layer_norm_rms_epsilon")),
        A + "attention.sliding_window": swa_window,
        A + "attention.sliding_window_pattern": [bool(x) for x in req("attention.sliding_window_pattern")],
        A + "attention.relative_length": int(req("attention.relative_length")),
        A + "attention.relative_extent": rel_extent,
        A + "attention.qk_clip": C.f32(req("attention.qk_clip")),
        A + "attention.relative_clip": C.f32(req("attention.relative_clip")),
        A + "attention.log_scale_floor": int(req("attention.log_scale_floor")),
        A + "attention.log_scale_alpha": C.f32(req("attention.log_scale_alpha")),
        A + "shortconv.l_cache": int(req("shortconv.l_cache")),
        A + "attn_res.block_size": int(req("attn_res.block_size")),
        A + "expert_count": int(req("expert_count")),
        A + "expert_used_count": int(req("expert_used_count")),
        A + "expert_feed_forward_length": int(req("expert_feed_forward_length")),
        A + "expert_weights_scale": C.f32(req("expert_weights_scale")),
        A + "expert_gate_softcap": C.f32(gate_cap),
        A + "expert_up_softcap": C.f32(up_cap),
        A + "final_logit_softcapping": C.f32(req("final_logit_softcapping")),
        A + "loop.start": int(req("loop.start")),
        A + "loop.end": int(req("loop.end")),
        A + "loop.count": int(req("loop.count")),
        A + "loop.gate_rank": int(req("loop.gate_rank")),
        A + "image_token_id": int(src_kv.get(P + "image_token_id", 200054)),
    })

    # Copy tokenizer.* keys from source GGUF
    for k, v in src_kv.items():
        if k.startswith("tokenizer."):
            kv[k] = v
    return kv


def _dst_name(src_name: str) -> str:
    m = re.match(r"^(blk\.\d+\.)(.+)$", src_name)
    if not m:
        return src_name
    prefix, suffix = m.group(1), m.group(2)
    return prefix + RENAME_SUFFIX.get(suffix, suffix)


def _target_form(dst_name: str, src_type: str) -> str:
    """Target tensor format inside the pack's `gguf` section."""
    if dst_name in ("token_embd.weight", "output.weight"):
        if src_type not in FLOAT_TYPES and src_type not in DEQUANT_TYPES:
            raise S.ConvertError(f"{dst_name} is {src_type}; inkling2 GGUF conversion supports BF16/F16/F32 or "
                                 f"{'/'.join(sorted(DEQUANT_TYPES))} embeddings/head (rounded to BF16)")
        return "BF16"
    suffix = re.sub(r"^blk\.\d+\.", "", dst_name)
    if suffix in Q8_SUFFIXES:
        if src_type not in FLOAT_TYPES and src_type not in DEQUANT_TYPES:
            raise S.ConvertError(
                f"{dst_name} is {src_type}; inkling2 dense projections are rounded to Q8_0 from BF16/F16/F32 or "
                f"{'/'.join(sorted(DEQUANT_TYPES))}")
        return "Q8_0"
    if CTRL_RE.match(suffix):
        if src_type not in ("BF16", "F16", "F32"):
            raise S.ConvertError(f"{dst_name} is {src_type}; inkling2 loop controllers require BF16/F16/F32")
        return "BF16"
    if src_type not in ("F32", "BF16", "F16"):
        raise S.ConvertError(f"{dst_name} is {src_type}; expected F32/BF16/F16")
    return "F32"


def build(a, Job, source_paths, export_tokenizer, fingerprint, die) -> None:
    out = pathlib.Path(a.out)
    partial = pathlib.Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    try:
        paths = source_paths(a.gguf)
    except S.ConvertError as e:
        die(str(e))

    shards = [GGUFFile(p) for p in paths]
    src_kv = shards[0].metadata
    src_arch = src_kv.get("general.architecture", "")
    if src_arch not in ("inkling2_loop", "inkling2"):
        die(f"{paths[0].name}: expected architecture inkling2_loop, got {src_arch!r}")

    name = a.name or src_kv.get("general.name") or paths[0].stem
    P = src_arch + "."
    n_layers = int(src_kv[P + "block_count"])
    n_expert = int(src_kv[P + "expert_count"])
    hidden = int(src_kv[P + "embedding_length"])
    inter = int(src_kv[P + "expert_feed_forward_length"])
    unpadded_vocab = int(src_kv.get(P + "vocab_size_unpadded", src_kv[P + "vocab_size"]))

    # Index all tensors across shards
    experts_gu: dict[int, tuple[str, TensorInfo, int]] = {}
    experts_dn: dict[int, tuple[str, TensorInfo, int]] = {}
    gguf_items: list[tuple[str, str, TensorInfo, int, str]] = []  # (dst_name, path, ti, file_off, dst_form)

    for sh in shards:
        pstr = str(sh.path)
        for ti in sh.tensors:
            foff = sh.data_start + ti.offset
            em = EXPERT_RE.match(ti.name)
            if em:
                layer, kind = int(em.group(1)), em.group(2)
                if kind == "ffn_gate_up_exps":
                    experts_gu[layer] = (pstr, ti, foff)
                else:
                    experts_dn[layer] = (pstr, ti, foff)
                continue
            dname = _dst_name(ti.name)
            try:
                form = _target_form(dname, ti.type_name)
            except S.ConvertError as e:
                die(str(e))
            gguf_items.append((dname, pstr, ti, foff, form))

    # Validate expert layers & determine target expert type
    if len(experts_gu) != n_layers or len(experts_dn) != n_layers:
        die(f"expected {n_layers} expert layers, found {len(experts_gu)} gate_up and {len(experts_dn)} down")
    sample_type = experts_gu[0][1].type_name
    nvfp4 = sample_type in ("BF16", "F16")
    if not nvfp4 and sample_type not in STORED_EXPERT_TYPES:
        die(f"blk.0.ffn_gate_up_exps.weight is {sample_type}; inkling2 GGUF conversion supports BF16, F16 "
            f"(rounded to NVFP4) and {', '.join(sorted(STORED_EXPERT_TYPES))} (as stored)")

    # per layer: (gate_up type name, down type name, per-expert gate_up bytes, per-expert down bytes)
    layer_types: list[tuple[str, str, int, int]] = []
    for l in range(n_layers):
        gt, dt = experts_gu[l][1].type_name, experts_dn[l][1].type_name
        if nvfp4:
            for kind, t in (("gate_up", gt), ("down", dt)):
                if t != sample_type:
                    die(f"layer {l} {kind} experts are {t}; a {sample_type} source must be {sample_type} throughout")
            layer_types.append((gt, dt, 0, 0))
            continue
        for kind, t in (("gate_up", gt), ("down", dt)):
            if t not in STORED_EXPERT_TYPES:
                die(f"layer {l} {kind} experts are {t}; quantized inkling2 experts must be one of "
                    f"{', '.join(sorted(STORED_EXPERT_TYPES))}")
        try:
            gu_eb, dn_eb = 2 * inter * _row_bytes(gt, hidden), hidden * _row_bytes(dt, inter)
        except S.ConvertError as e:
            die(f"layer {l} experts: {e}")
        for kind, (_, ti, _), eb in (("gate_up", experts_gu[l], gu_eb), ("down", experts_dn[l], dn_eb)):
            if ti.expected_bytes() != n_expert * eb:
                die(f"layer {l} {kind} experts: {ti.expected_bytes()} bytes, expected {n_expert} x {eb}")
        layer_types.append((gt, dt, gu_eb, dn_eb))

    if nvfp4:
        expert_type = NVFP4
        expert_blob = (2 * inter * S.nvfp4_row_bytes(hidden)
                       + hidden * S.nvfp4_row_bytes(inter)
                       + S.NVFP4_TAIL)
    else:
        expert_type = experts_gu[0][1].type_id

    kv = _map_kv(src_kv, name, expert_type)
    base = ModelFile(a.sections_from) if a.sections_from else None
    mmproj = pathlib.Path(a.mmproj) if a.mmproj else None
    if mmproj and not mmproj.is_file():
        die(f"missing --mmproj: {mmproj}")

    t0 = time.monotonic()
    workers = max(1, a.workers)

    with tempfile.TemporaryDirectory(prefix="basalt_ink_gguf_") as tmp:
        tmp_dir = pathlib.Path(tmp)
        tok_files = export_tokenizer(paths[0], tmp_dir)

        mw = ModelWriter(str(partial))
        try:
            job = Job(mw)
            for p in tok_files:
                job.file(f"tokenizer/{p.name}", p)

            # 1. experts + experts.layout
            layout_lines = []
            offset = 0
            for l in range(n_layers):
                gt, dt, gu_eb, dn_eb = layer_types[l]
                if nvfp4:
                    layout_lines.append((l, NVFP4, NVFP4, offset, expert_blob))
                    offset += n_expert * expert_blob
                else:
                    gu_ti, dn_ti = experts_gu[l][1], experts_dn[l][1]
                    layout_lines.append((l, gu_ti.type_id, dn_ti.type_id, offset, gu_eb + dn_eb))
                    offset += n_expert * (gu_eb + dn_eb)
            total_exp_bytes = offset

            def expert_chunks():
                with multiprocessing.Pool(workers) as pool:
                    for l in range(n_layers):
                        gu_p, _, gu_off = experts_gu[l]
                        dn_p, _, dn_off = experts_dn[l]
                        _, _, gu_eb, dn_eb = layer_types[l]
                        tasks = []
                        for e0 in range(0, n_expert, EXPERTS_PER_JOB):
                            e1 = min(e0 + EXPERTS_PER_JOB, n_expert)
                            if nvfp4:
                                tasks.append((gu_p, gu_off, sample_type, dn_p, dn_off, sample_type,
                                              inter, hidden, e0, e1))
                            else:
                                tasks.append((gu_p, gu_off, gu_eb, dn_p, dn_off, dn_eb, e0, e1))
                        fn = _expert_nvfp4_job if nvfp4 else _expert_stored_job
                        for chunk in pool.imap(fn, tasks):
                            if isinstance(chunk, str):
                                die(f"layer {l}: {chunk}")
                            yield chunk

            job.chunks("experts", expert_chunks(), total_exp_bytes)

            if nvfp4:
                layout_txt = S.nvfp4_layout_text(n_expert, total_exp_bytes, layout_lines)
            else:
                hdr = f"# experts v6: layer gu_type d_type offset blob_bytes (n_expert {n_expert}, total {total_exp_bytes})\n"
                layout_txt = (hdr + "".join(
                    f"{l} {gt} {dt} {off} {nb}\n" for l, gt, dt, off, nb in layout_lines
                )).encode("utf-8")
            job.bytes("experts.layout", layout_txt)

            # 2. gguf section (all non-expert tensors)
            gguf_entries: dict[str, dict] = {}
            gw = GGUFWriter()
            for k, v in kv.items():
                if isinstance(v, list) and not v:
                    gw.add(k, v, "array:string")
                else:
                    gw.add(k, v)

            prepared: list[tuple[str, list[int], str, bytes]] = []
            for dname, pstr, ti, foff, form in gguf_items:
                shape = list(ti.shape)
                nbytes = ti.expected_bytes()
                if dname in ("token_embd.weight", "output.weight") and len(shape) == 2 and shape[1] > unpadded_vocab:
                    # Trim padded vocab rows (GGUF shape is [cols, rows])
                    cols = shape[0]
                    keep_bytes = unpadded_vocab * _row_bytes(ti.type_name, cols)
                    data = _read_bf16_bytes(pstr, foff, keep_bytes, ti.type_name)
                    shape[1] = unpadded_vocab
                elif form == "Q8_0":
                    if ti.type_name == "Q8_0":
                        data = bytes(H.pread(pstr, foff, nbytes))
                    else:
                        arr = _read_f32(pstr, foff, nbytes, ti.type_name).reshape(-1, shape[0])
                        data = S.q8_0(arr)
                elif form == "BF16":
                    data = _read_bf16_bytes(pstr, foff, nbytes, ti.type_name)
                else:  # F32
                    if dname.endswith(".attn_rel_bias.weight") and len(shape) == 2 and shape[0] < shape[1]:
                        # llama.cpp stores [d_rel, rel_extent]; Basalt expects [rel_extent, d_rel]
                        arr = _read_f32(pstr, foff, nbytes, ti.type_name).reshape(shape[1], shape[0])
                        data = np.ascontiguousarray(arr.T, dtype="<f4").tobytes()
                        shape = [shape[1], shape[0]]
                    elif ti.type_name == "F32":
                        data = bytes(H.pread(pstr, foff, nbytes))
                    else:
                        data = _read_f32(pstr, foff, nbytes, ti.type_name).tobytes()
                prepared.append((dname, shape, form, data))

            # Write gguf section
            gguf_tmp = tmp_dir / "model.gguf"
            for dname, shape, form, data in prepared:
                gw.add_raw(dname, shape, form, data)
            gw.write(gguf_tmp)
            # Record tensor offsets for meta
            g_out = GGUFFile(gguf_tmp)
            for ti in g_out.tensors:
                gguf_entries[ti.name] = {
                    "type": ti.type_name,
                    "type_id": ti.type_id,
                    "shape": list(ti.shape),
                    "bytes": ti.expected_bytes(),
                    "offset": g_out.data_start + ti.offset,
                }
            job.file("gguf", gguf_tmp)

            # 3. Optional vision/mmproj
            vision_meta = None
            if mmproj:
                job.file("vision/mmproj", mmproj)
                vision_meta = {"projector": "inkling2v", "source": mmproj.name}
            elif base and "vision/mmproj" in base.sections:
                off, sz = base.sections["vision/mmproj"][:2]
                def _copy_mmproj():
                    with open(base.path, "rb") as bf:
                        bf.seek(off)
                        rem = sz
                        while rem > 0:
                            chunk = bf.read(min(rem, 64 << 20))
                            if not chunk:
                                break
                            rem -= len(chunk)
                            yield chunk
                job.chunks("vision/mmproj", _copy_mmproj(), sz)
                bmeta = json.loads(base.read("meta"))
                vision_meta = bmeta.get("vision")

            # 4. meta
            meta = {
                "format": "basalt",
                "version": 2,
                "name": name,
                "arch": ARCH,
                "converter": {"tool": "tools/convert/convert.py"},
                "sources": [fingerprint(p) for p in paths],
                "gguf_kv": kv,
                "tensors": gguf_entries,
                "license": {"name": str(kv.get("general.license", "apache-2.0"))},
            }
            if vision_meta:
                meta["vision"] = vision_meta
            job.bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            mw.close()
        except BaseException:
            if partial.exists():
                partial.unlink()
            raise

        partial.replace(out)
    print(f"done: {out} ({out.stat().st_size / (1 << 30):.2f} GiB) in {time.monotonic() - t0:.1f} s")
