"""Engine settings, tokenizer and vision inputs for a .basalt model file, unpacked once into the cache directory."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from model_file import ModelFile  # noqa: E402

VISION_EXE = "basalt-vision"


def cache_root() -> Path:
    return Path.home() / ".cache" / "basalt"


def cache_dir(mf: ModelFile) -> Path:
    """One cache directory per model, named by the meta section's sha256 prefix."""
    return cache_root() / mf.sections["meta"][2].hex()[:16]


def write_section(mf: ModelFile, name: str, dest: Path) -> None:
    """Copy a section to `dest` via a temporary name so a partial file is never used."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    off, size = mf.section(name)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".")
    try:
        with os.fdopen(fd, "wb") as out, open(mf.path, "rb") as src:
            src.seek(off)
            left = size
            while left > 0:
                b = src.read(min(left, 64 << 20))
                if not b:
                    raise IOError(f"{mf.path}: short read in {name}")
                out.write(b)
                left -= len(b)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def extract_tokenizer(mf: ModelFile) -> Path:
    """The `tokenizer/*` sections as a directory, reused when present."""
    out = cache_dir(mf) / "tokenizer"
    if (out / "vocab.json").exists():
        return out
    names = [n for n in mf.names() if n.startswith("tokenizer/")]
    if not any(n == "tokenizer/vocab.json" for n in names):
        raise ValueError(f"{mf.path}: no tokenizer sections")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out.parent, prefix="tokenizer."))
    try:
        for n in names:
            (tmp / n.split("/", 1)[1]).write_bytes(mf.read(n))
        try:
            os.rename(tmp, out)
        except OSError:                          # another process finished first
            if not (out / "vocab.json").exists():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def vision_config(mf: ModelFile, engine_bin: str, sidecar: bool = True) -> dict:
    """The vision sidecar's config (the projector alone without `sidecar`); ValueError names what is missing."""
    side = Path(engine_bin).resolve().parent / VISION_EXE
    if "vision/mmproj" not in mf.sections:
        raise ValueError("this model file has no vision projector (vision/mmproj); drop --vision")
    if sidecar and not side.is_file():
        raise ValueError(f"{side} not found (build the vision sidecar), or drop --vision")
    if "vision/text_vocab" not in mf.sections:
        raise ValueError("this model file has no vision/text_vocab section; drop --vision")
    vdir = cache_dir(mf) / "vision"
    mmproj, vocab = vdir / "mmproj.gguf", vdir / "text_vocab.gguf"
    for name, dest in (("vision/mmproj", mmproj), ("vision/text_vocab", vocab)):
        if not dest.exists() or dest.stat().st_size != mf.section(name)[1]:
            write_section(mf, name, dest)
    return {"vision_bin": str(side), "mmproj": str(mmproj), "model": str(vocab)}


def model_config(path: str, engine_bin: str, vision: bool | str = False,
                 engine_words: list[str] | None = None) -> dict:
    """The engine settings the file implies; `engine_words` follow its arguments as given (the engine's parser lets a
    later flag win), `vision` (True, "peer"
    or "cpu") adds the sidecar and "dynamic" the in-engine encoder (ValueError if it cannot be set up)."""
    mf = ModelFile(path)
    meta = json.loads(mf.read("meta"))
    path = os.path.abspath(path)
    words = list(engine_words or [])
    args = ["--model", path]   # settings come from the engine defaults and the server
    args += words
    cdir = cache_dir(mf)
    cdir.mkdir(parents=True, exist_ok=True)
    cfg = {"engine_bin": os.path.abspath(engine_bin), "args": args, "model_name": meta.get("name") or "basalt",
           "sampling": dict(meta.get("sampling") or {}), "log": str(cdir / "engine.log"),
           "tokenizer": str(extract_tokenizer(mf))}
    v = vision_config(mf, engine_bin, sidecar=vision != "dynamic") if vision else None
    if v:
        v["mode"] = "on" if vision is True else vision
        v["gpu"] = vision != "cpu"
        cfg["vision"] = v
        if vision == "dynamic":     # the engine encodes: it reads the projector itself
            args += ["--vision-mmproj", v["mmproj"]]
        if "--vision" not in args:   # image embeddings need --vision
            args.append("--vision")
    return cfg
