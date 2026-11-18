"""--dfly FILE --into PACK.basalt: the DFly2 drafter joins a hy4flash pack as the `dfly` section.

FILE is a safetensors file holding the drafter's tensors (`model.dfly_layers.*`, 87 BF16 tensors, 1.69 GB): the last shard of
tencent/Hy4-flash, or the side file extracted from the checkpoint. It is stored whole; meta.dfly2 records where each tensor
lies (offsets from the start of the section), the drafter's settings from the pack's own meta.hf_config (config.json
`dfly_*` keys) and the shapes the engine checks. Tensors of the file that are not the drafter's are left unread.
Like --dspark: no other section moves, and --roll-back restores the pack bit for bit.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

PREFIX = "model.dfly_layers."
SECTION = "dfly"


def die(msg: str):
    raise SystemExit(f"convert: {msg}")


def read_header(path: Path) -> tuple[dict, int]:
    """The safetensors header (without __metadata__) and the file offset where the tensor data starts."""
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            die(f"{path}: not a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        if n > 64 << 20:
            die(f"{path}: not a safetensors file")
        try:
            head = json.loads(f.read(n))
        except ValueError:
            die(f"{path}: not a safetensors file")
    head.pop("__metadata__", None)
    return head, 8 + n


def wanted(layers: int) -> set[str]:
    """The drafter's tensor names, without the model.dfly_layers. prefix."""
    names = {f"{i}.{s}" for i in range(layers) for s in (
        "input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "mlp.down_proj.weight", "self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight", "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_a_layernorm.weight", "self_attn.kv_b_proj.weight",
        "self_attn.linear_gate.weight", "self_attn.o_proj.weight", "self_attn.post_proj_norm.weight",
        "self_attn.learnable_sink_param")}
    names |= {"final_norm.weight", "mtp_fusion.eh_proj.weight", "mtp_fusion.enorm.weight", "mtp_fusion.hnorm.weight",
              "mtp_fusion.gate", "hidden_correction.gate_proj.weight", "hidden_correction.up_proj.weight",
              "hidden_correction.down_proj.weight", "hidden_correction.hidden_norm.weight",
              "hidden_correction.embed_norm.weight", "confidence_head.proj.weight", "confidence_head.proj.bias"}
    return names


def dfly_record(path: Path, hf_config: dict | None) -> dict:
    """meta.dfly2 for the file: what the engine loads by, from the file's header and the checkpoint's config."""
    cfg = hf_config or {}
    text = {**cfg.get("text_config", {}), **cfg}      # the released config.json keeps the dfly_* keys at its top level
    for k in ("dfly_num_hidden_layers", "dfly_mask_token_id", "dfly_target_kv_layers", "dfly_target_index_layer"):
        if text.get(k) is None:
            die(f"the pack's meta.hf_config has no {k}: the pack was made without --hf-dir, or not from Hy4-flash")
    layers = int(text["dfly_num_hidden_layers"])
    if text.get("dfly_mtp_fusion", "slot0") != "slot0" or text.get("dfly_enable_hidden_correction", True) is not True:
        die("the engine runs slot0 fusion with hidden correction, the only drafter configuration there is")
    head, base = read_header(path)
    tensors = {}
    for name, t in head.items():
        if not name.startswith(PREFIX):
            continue
        if t["dtype"] != "BF16":
            die(f"{path}: {name} is {t['dtype']}; the engine reads the drafter's BF16 tensors")
        lo, hi = t["data_offsets"]
        tensors[name[len(PREFIX):]] = {"dtype": "BF16", "shape": t["shape"], "offset": base + lo, "bytes": hi - lo}
    if not tensors:
        die(f"{path} holds no {PREFIX}* tensors")
    want = wanted(layers)
    if missing := sorted(want - tensors.keys()):
        die(f"{path} lacks drafter tensors: {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
    window = text.get("dfly_context_local_window")
    return {
        "section": SECTION, "format": "safetensors", "bytes": path.stat().st_size, "layers": layers,
        "mask_token_id": int(text["dfly_mask_token_id"]),
        "target_kv_layers": [int(x) for x in text["dfly_target_kv_layers"]],
        "target_index_layers": [int(x) for x in text["dfly_target_index_layer"]],
        "target_index_topk": int(text.get("dfly_target_index_topk") or 2048),
        "context_local_window": int(window) if window else 0,
        "tensors": {k: v for k, v in tensors.items() if k in want},
    }


def add_dfly(a, source_record=None) -> None:
    """--dfly with --into: the drafter joins a pack that exists, whose other sections stay where and as they are."""
    from model_file import ModelFile, append_section
    pack = Path(a.into)
    if not pack.is_file():
        die(f"missing input: {pack}")
    src = Path(a.dfly)
    if not src.is_file():
        die(f"missing input: {src}")
    m = ModelFile(str(pack))
    meta = json.loads(m.read("meta"))
    if meta.get("arch") != "hy4flash":
        die(f"{pack} is no hy4flash pack")
    if SECTION in m.names() or meta.get("dfly2"):
        die(f"{pack} has a drafter already")
    meta["dfly2"] = dfly_record(src, meta.get("hf_config"))
    if source_record is not None:
        meta.setdefault("sources", []).append(source_record(src))
    try:
        append_section(str(pack), SECTION, str(src), json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), a.undo_dir)
    except (ValueError, FileExistsError) as e:
        die(str(e))
    m = ModelFile(str(pack))
    if not m.verify(SECTION) or not m.verify("meta"):
        die(f"{pack}: the new sections do not read back as written; --roll-back puts the file back")
    print(f"done: {pack} is {pack.stat().st_size} bytes, {len(m.names())} sections")
