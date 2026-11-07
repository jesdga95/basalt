"""Engine settings, tokenizer and vision inputs for a .basalt model file. Nothing is unpacked: the tokenizer is read from
its sections into memory, and the vision programs open the projector inside the file themselves."""
from __future__ import annotations

import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from model_file import ModelFile  # noqa: E402
from serve.models import dialect_for  # noqa: E402
from serve.engine.flags import flag_value  # noqa: E402

VISION_EXE = "basalt-vision"
# llama.cpp's sampling keys in the GGUF metadata -> the request field each one defaults
GGUF_SAMPLING = {"general.sampling.temp": "temperature", "general.sampling.top_p": "top_p",
                 "general.sampling.top_k": "top_k", "general.sampling.min_p": "min_p",
                 "general.sampling.penalty_repeat": "repetition_penalty",
                 "general.sampling.penalty_last_n": "penalty_last_n"}


def cache_root() -> Path:
    return Path.home() / ".cache" / "basalt"


def cache_dir(mf: ModelFile) -> Path:
    """One cache directory per model, named by the meta section's sha256 prefix."""
    return cache_root() / mf.sections["meta"][2].hex()[:16]


def with_conversation_dir(words: list[str], cdir: Path) -> list[str]:
    """`words` plus the default conversation directory (inside the model cache folder `cdir`) when the disk budget
    is above zero and no directory is named."""
    budget = flag_value(words, "--conversation-cache-disk-mib")
    if budget and budget.isdigit() and int(budget) > 0 and not flag_value(words, "--conversation-cache-dir"):
        return words + ["--conversation-cache-dir", str(cdir / "conversations")]
    return words


def _short(v):
    """A GGUF float32 as the shortest decimal that is the same float32 (0.949999988079071 -> 0.95)."""
    if not isinstance(v, float):
        return v
    bits = struct.pack("<f", v)
    for digits in range(1, 10):
        s = float(f"{v:.{digits}g}")
        if struct.pack("<f", s) == bits:
            return s
    return v


def sampling_defaults(meta: dict) -> dict:
    """The model's sampling defaults: the general.sampling.* keys of meta.gguf_kv, by request field."""
    kv = meta.get("gguf_kv") or {}
    return {field: _short(kv[key]) for key, field in GGUF_SAMPLING.items() if kv.get(key) is not None}


TOKENIZER_SECTIONS = ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja")


def tokenizer_files(mf: ModelFile) -> dict[str, bytes]:
    """The `tokenizer/*` sections the server reads, by file name (those a pack lacks are absent)."""
    if "tokenizer/vocab.json" not in mf.sections:
        raise ValueError(f"{mf.path}: no tokenizer sections")
    return {n: mf.read("tokenizer/" + n) for n in TOKENIZER_SECTIONS if "tokenizer/" + n in mf.sections}


def vision_config(mf: ModelFile, engine_bin: str, sidecar: bool = True) -> dict:
    """The vision sidecar's config, which names the model file (the sidecar opens `vision/mmproj` inside it);
    ValueError names what is missing."""
    engine = Path(engine_bin).resolve()
    side = engine.parent / (VISION_EXE + engine.suffix)   # with the engine's suffix: basalt-vision.exe on Windows
    if "vision/mmproj" not in mf.sections:
        raise ValueError("this model file has no vision projector (vision/mmproj); drop --vision")
    if sidecar and not side.is_file():
        raise ValueError(f"{side} not found (build the vision sidecar), or drop --vision")
    return {"vision_bin": str(side), "model": os.path.abspath(mf.path)}


@dataclass(frozen=True)
class ModelSpec:
    """One model the server runs. `engine_words` are its engine flags as given. `vision`: a GPU number as a string, "cpu"
    or "dynamic". `gpu`: as --gpu."""
    path: str
    alias: str | None = None
    gpu: object = None
    vision: bool | str = False
    engine_words: tuple[str, ...] = ()


def model_config(spec: ModelSpec, engine_bin: str) -> dict:
    """The engine settings a model implies; its `engine_words` follow its arguments as given (the engine's parser lets
    a later flag win), its `vision` adds the sidecar or, "dynamic", the in-engine encoder (ValueError if it cannot be
    set up)."""
    path, vision = spec.path, spec.vision
    mf = ModelFile(path)
    meta = json.loads(mf.read("meta"))
    path = os.path.abspath(path)
    cdir = cache_dir(mf)
    words = with_conversation_dir(list(spec.engine_words), cdir)
    args = ["--model", path]   # settings come from the engine defaults and the server
    args += words
    cdir.mkdir(parents=True, exist_ok=True)
    cfg = {"engine_bin": os.path.abspath(engine_bin), "args": args, "model_name": spec.alias or meta.get("name") or "basalt",
           "sampling": sampling_defaults(meta),
           "arch": (meta.get("gguf_kv") or {}).get("general.architecture"), "log": str(cdir / "engine.log"),
           "tokenizer": tokenizer_files(mf)}
    if vision and dialect_for(cfg["arch"]).in_engine_vision:
        if vision != "dynamic":
            raise ValueError(f"this model's images are encoded in the engine: --vision dynamic, not {vision}")
        if "vision/mmproj" not in mf.sections:
            raise ValueError("this model file has no vision projector (vision/mmproj); drop --vision")
        args.append("--vision")
        cfg["vision"] = {"mode": "dynamic", "gpu": True}
        return cfg
    v = vision_config(mf, engine_bin, sidecar=vision != "dynamic") if vision else None
    if v:
        v["mode"] = vision
        v["gpu"] = vision != "cpu"
        cfg["vision"] = v
        # dynamic: the engine encodes, opening the projector inside the model file itself; else the sidecar's
        # embeddings come in with each image
        if vision == "dynamic" and "--vision" not in args:
            args.append("--vision")
        elif vision != "dynamic" and "--vision-external" not in args:
            args.append("--vision-external")
    return cfg
