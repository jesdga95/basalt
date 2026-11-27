"""Attach Inkling 2's DSpark drafter from `mtp.safetensors` to an existing inkling2 pack.

    python3 tools/convert/dspark_graft.py --mtp mtp.safetensors --config config.json \
        --into PACK.basalt --undo-dir DIR
    python3 tools/convert/dspark_graft.py --roll-back --into PACK.basalt --undo-dir DIR

The checkpoint converter writes the drafter's tensors into the pack's `gguf` section as `dspark.*`.
A pack that already exists has its `gguf` section, so this tool appends a separate `dspark/weights`
section and points `meta.tensors` at it: `pack.cpp` resolves a tensor through the section named in
its meta entry. No other section moves, and --roll-back restores the file bit for bit.

The tensor map is the checkpoint converter's DRAFTER_LAYER / DRAFTER_GLOBAL, so the drafter this
writes is the one the engine reads. The source BF16 bytes are copied as they are; the few F32
tensors are widened, and the padded vocabulary rows of `markov.w1.weight` are dropped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import checkpoint_inkling2 as I  # noqa: E402
from convert import source_record  # noqa: E402
from dfly_attach import read_header  # noqa: E402
from gguf_writer import TYPE_IDS  # noqa: E402
from model_file import ModelFile, append_section, undo_append  # noqa: E402
import model_sections as S  # noqa: E402

SECTION = "dspark/weights"
ALIGN = 4096


def die(msg: str) -> None:
    raise SystemExit("convert: " + msg)


def wanted(mc: dict, n_vocab: int) -> list[tuple[str, str, str, list[int], bool]]:
    """(pack name, target form, safetensors key, expected source shape, trim vocabulary) per tensor."""
    n_layers = int(mc["num_hidden_layers"])
    e = int(mc["hidden_size"])
    n_ff = int(mc["intermediate_size"])
    nq = int(mc["num_attention_heads"]) * int(mc["head_dim"])
    nkv = int(mc["num_key_value_heads"]) * int(mc["head_dim"])
    head = int(mc["head_dim"])
    rank = int(mc["markov_rank"])
    targets = len(mc["target_layer_ids"])
    out = []
    for key, (name, form) in I.DRAFTER_GLOBAL.items():
        shape = {
            "fuse_weight": [4, targets],
            "mask_embed": [e],
            "norm.weight": [e],
            "markov_head.markov_w1.weight": [None, rank],
            "markov_head.rnn_proj.weight": [3 * rank, 2 * rank + 2 * e],
            "markov_head.rnn_proj.bias": [3 * rank],
            "markov_head.hidden_out.weight": [e, rank],
            "markov_head.hidden_out.bias": [e],
            "markov_head.dflare_norm.weight": [e],
        }[key]
        out.append(("dspark." + name, form, key, shape, key == "markov_head.markov_w1.weight"))
    for layer in range(n_layers):
        shape = {
            "input_layernorm.weight": [e],
            "post_attention_layernorm.weight": [e],
            "dflare_norm.weight": [e],
            "self_attn.q_proj.weight": [nq, e],
            "self_attn.k_proj.weight": [nkv, e],
            "self_attn.v_proj.weight": [nkv, e],
            "self_attn.o_proj.weight": [e, nq],
            "self_attn.k_proj_ctx.weight": [nkv, e],
            "self_attn.v_proj_ctx.weight": [nkv, e],
            "self_attn.q_norm.weight": [head],
            "self_attn.k_norm.weight": [head],
            "self_attn.k_norm_ctx.weight": [head],
            "mlp.gate_proj.weight": [n_ff, e],
            "mlp.up_proj.weight": [n_ff, e],
            "mlp.down_proj.weight": [e, n_ff],
        }
        for key, (name, form) in I.DRAFTER_LAYER.items():
            out.append((f"dspark.blk.{layer}.{name}", form, f"layers.{layer}.{key}", shape[key], False))
    return out


def dspark_kv(mc: dict) -> dict:
    """The `inkling2.dspark.*` geometry, the same keys `checkpoint_inkling2.model_kv` writes."""
    n = int(mc["num_hidden_layers"])
    local = set(int(x) for x in mc["local_layer_ids"])
    a = I.ARCH + ".dspark."
    return {
        a + "block_count": n,
        a + "block_size": int(mc["block_size"]),
        a + "target_layers": [int(x) for x in mc["target_layer_ids"]],
        a + "sliding_window_pattern": [l in local for l in range(n)],
        a + "sliding_window": int(mc["sliding_window_size"]),
        a + "attention.head_count": int(mc["num_attention_heads"]),
        a + "attention.head_count_kv": int(mc["num_key_value_heads"]),
        a + "attention.key_length": int(mc["head_dim"]),
        a + "feed_forward_length": int(mc["intermediate_size"]),
        a + "rope.freq_base": float(mc["rope_theta"]),
        a + "attention.layer_norm_rms_epsilon": float(mc["rms_norm_eps"]),
        a + "markov_rank": int(mc["markov_rank"]),
    }


def graft(mtp: Path, config: Path, pack: Path, undo_dir: Path) -> None:
    if not mtp.is_file():
        die(f"missing input: {mtp}")
    if not config.is_file():
        die(f"missing input: {config}")
    if undo_dir.exists():
        die(f"undo directory exists: {undo_dir}")
    m = ModelFile(str(pack))
    meta = json.loads(m.read("meta"))
    if meta.get("arch") != "inkling2":
        die(f"{pack}: not an inkling2 pack")
    kv = meta.setdefault("gguf_kv", {})
    if "inkling2.dspark.block_count" in kv:
        die(f"{pack}: already carries a DSpark drafter")
    tokens = kv.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or not tokens:
        die(f"{pack}: meta has no tokenizer vocabulary")
    n_vocab = len(tokens)
    ck = json.loads(config.read_text(encoding="utf-8"))
    mc = ck.get("mtp_config") or {}
    if not mc:
        die(f"{config}: no mtp_config")
    if mc.get("markov_head_type") != "rnn_hidden" or mc.get("q_bias") or mc.get("o_bias"):
        die("a DSpark drafter other than the rnn_hidden one without biases")
    text = ck.get("text_config") or {}
    if int(text.get("unpadded_vocab_size", n_vocab)) != n_vocab:
        die(f"{config}: unpadded vocabulary differs from the pack's {n_vocab} tokens")
    if int(text.get("hidden_size", mc["hidden_size"])) != int(mc["hidden_size"]):
        die(f"{config}: the text and drafter widths differ")

    raw_head, base = read_header(mtp)
    head = {k[4:] if k.startswith("mtp.") else k: v for k, v in raw_head.items()}
    tensors = wanted(mc, n_vocab)
    missing = [t[2] for t in tensors if t[2] not in head]
    if missing:
        die(f"{mtp}: lacks {len(missing)} drafter tensors: " + ", ".join(missing[:8]))

    entries: dict[str, dict] = {}
    digests: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="basalt_dspark_graft_", dir=str(pack.parent)) as tmp:
        section = Path(tmp) / "dspark.section"
        with section.open("wb") as out:
            for name, form, key, shape, trim in tensors:
                info = head[key]
                if info["dtype"] != "BF16":
                    die(f"{mtp}: {key} is {info['dtype']}, expected BF16")
                got = list(info["shape"])
                want = [n_vocab if (trim and x is None) else x for x in shape]
                if got[1:] != want[1:] or (not trim and got != want) or (trim and got[0] < n_vocab):
                    die(f"{mtp}: {key} is {got}, expected {want}" + (" or more rows" if trim else ""))
                lo, hi = info["data_offsets"]
                x = np.fromfile(mtp, dtype="<u2", count=(hi - lo) // 2, offset=base + lo)
                if trim:
                    x = x[:n_vocab * got[1]]
                if form == "F32":
                    data = S._bf16_to_f32(x).astype("<f4").tobytes()
                else:
                    data = x.astype("<u2", copy=False).tobytes()
                pos = out.tell()
                out.write(b"\0" * ((-pos) % ALIGN))
                off = out.tell()
                out.write(data)
                entries[name] = {"type": form, "type_id": TYPE_IDS[form],
                                 "shape": [got[1], n_vocab] if trim else list(reversed(got)),
                                 "bytes": len(data), "offset": off, "section": SECTION}
                digests[name] = hashlib.sha256(data).hexdigest()
        meta["tensors"].update(entries)
        kv.update(dspark_kv(mc))
        meta["dspark"] = {"source": mtp.name, "form": "BF16 as stored, F32 where the engine reads F32",
                          "section": SECTION}
        for s in meta.get("sources", []):
            if isinstance(s, dict) and s.get("path") == mtp.name:
                break
        else:
            meta.setdefault("sources", []).append(source_record(mtp))
        new_meta = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            append_section(str(pack), SECTION, str(section), new_meta, str(undo_dir))
        except (ValueError, OSError, FileExistsError) as e:
            die(str(e))

    m = ModelFile(str(pack))
    if not m.verify(SECTION) or not m.verify("meta"):
        die(f"{pack}: the new section does not read back as written; --roll-back puts the file back")
    blob = m.read(SECTION)
    for name, entry in entries.items():
        got = blob[entry["offset"]:entry["offset"] + entry["bytes"]]
        if hashlib.sha256(got).hexdigest() != digests[name]:
            die(f"{pack}: {name} reads back differently; --roll-back puts the file back")
    print(f"done: {pack} carries {len(entries)} dspark tensors in {SECTION}, "
          f"{pack.stat().st_size / (1 << 30):.2f} GiB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtp", metavar="FILE", help="the checkpoint's mtp.safetensors")
    ap.add_argument("--config", metavar="FILE", help="the checkpoint's config.json (its mtp_config)")
    ap.add_argument("--into", metavar="PACK.basalt", required=True, help="the inkling2 pack to add the drafter to")
    ap.add_argument("--undo-dir", metavar="DIR", required=True, help="a new directory for what --roll-back restores")
    ap.add_argument("--roll-back", action="store_true", help="restore the pack as it was before --into")
    a = ap.parse_args()
    pack = Path(a.into)
    undo = Path(a.undo_dir)
    if a.roll_back:
        if a.mtp or a.config:
            ap.error("--roll-back takes only --into and --undo-dir")
        if not pack.is_file() or not undo.is_dir():
            ap.error("--roll-back needs the pack and its undo directory")
        try:
            undo_append(str(pack), str(undo))
        except (ValueError, OSError) as e:
            die(str(e))
        print(f"done: {pack} is back to {pack.stat().st_size} bytes")
        return
    if not a.mtp or not a.config:
        ap.error("--mtp and --config are required without --roll-back")
    graft(Path(a.mtp), Path(a.config), pack, undo)


if __name__ == "__main__":
    main()
