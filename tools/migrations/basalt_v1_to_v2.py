"""Write a version 2 .basalt file from a version 1 file, without re-quantizing anything.

    python3 tools/migrations/basalt_v1_to_v2.py OLD.basalt NEW.basalt [--force]

Every model section is copied byte for byte (its SHA-256 checked against OLD's table); `profile` is dropped; `hc8` is
computed from the file's own hyper-connection matrices (tools/convert/hc8.py); `meta` loses `profile`, `hc8`,
`sampling` and the converter's git revision, and records version 2. Sampling defaults that only `meta.sampling` held
move to `gguf_kv` as llama.cpp's general.sampling.* keys. Needs numpy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "convert"))
import hc8 as H  # noqa: E402
from model_file import ModelFile, ModelWriter  # noqa: E402

DROPPED = ("profile", "hc8", "meta")
# meta.sampling key -> GGUF key
SAMPLING_KV = {"temperature": "general.sampling.temp", "top_p": "general.sampling.top_p",
               "top_k": "general.sampling.top_k", "min_p": "general.sampling.min_p",
               "repetition_penalty": "general.sampling.penalty_repeat",
               "penalty_last_n": "general.sampling.penalty_last_n"}


def die(msg: str):
    raise SystemExit(f"migration: {msg}")


def new_meta(meta: dict) -> dict:
    meta = dict(meta)
    kv = dict(meta.get("gguf_kv") or {})
    for key, value in (meta.pop("sampling", None) or {}).items():
        if key in SAMPLING_KV and value is not None:
            kv.setdefault(SAMPLING_KV[key], value)
    meta["gguf_kv"] = kv
    for key in ("profile", "hc8"):
        meta.pop(key, None)
    if isinstance(meta.get("converter"), dict):
        meta["converter"] = {k: v for k, v in meta["converter"].items() if k != "git"}
    meta["version"] = 2
    return meta


def migrate(src: Path, out: Path) -> None:
    try:
        old = ModelFile(str(src), versions=(1, 2))
    except ValueError as e:
        die(str(e))
    if old.version != 1:
        die(f"{src} is .basalt version {old.version}; this script reads version 1")
    meta = json.loads(old.read("meta"))
    partial = Path(str(out) + ".partial")
    partial.unlink(missing_ok=True)
    w = ModelWriter(str(partial))
    t0 = time.time()
    ok = False
    try:
        with open(src, "rb") as f:
            def read(name, off, size):
                f.seek(old.section(name)[0] + off)
                return f.read(size)
            for name in old.names():
                if name in DROPPED:
                    continue
                off, size = old.section(name)
                w.add_file(name, str(src), off, size)
                if w.entries[-1][3] != old.sections[name][2]:
                    die(f"section {name}: its bytes do not match the SHA-256 in {src}'s table")
                print(f"  {name:<44} {size / (1 << 30):9.3f} GiB", flush=True)
            whole = lambda n: read(n, 0, old.section(n)[1]) if n in old.sections else None
            try:
                table = H.section(read, whole("dense.index"), whole("mtp/dense.index"))
            except H.Hc8Error as e:
                die(f"hc8: {e}")
        w.add_bytes("hc8", table)
        print(f"  {'hc8':<44} {len(table) / (1 << 30):9.3f} GiB", flush=True)
        w.add_bytes("meta", json.dumps(new_meta(meta), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        w.close()
        ok = True
    finally:
        if not ok:
            w.f.close()
            partial.unlink(missing_ok=True)
    os.replace(partial, out)
    print(f"done: {len(w.entries)} sections, {out.stat().st_size / (1 << 30):.3f} GiB, {time.time() - t0:.1f} s -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", help="the version 1 .basalt file (read only)")
    ap.add_argument("new", help="the version 2 .basalt file to write")
    ap.add_argument("--force", action="store_true", help="overwrite NEW")
    a = ap.parse_args()
    src, out = Path(a.old), Path(a.new)
    if not src.is_file():
        die(f"missing input: {src}")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    if out.exists() and out.resolve() == src.resolve():
        die("NEW must be another file than OLD")
    migrate(src, out)


if __name__ == "__main__":
    main()
