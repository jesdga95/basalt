"""Build the single .basalt model file from the source GGUF shards.

    python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --out MODEL.basalt \\
        [--mtp-source DIR] [--mtp-dense FILE] [--profile FILE | --profile-from TRACE ...] [--draft-vocab FILE] [--hc8 FILE] \\
        [--name NAME] [--compat-bf16] [--no-vision] [--force]
    python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --sections-from BASE.basalt --out MODEL.basalt [--name NAME] [--force]
    python3 tools/convert/convert.py --checkpoint DIR --out MODEL.basalt [--dense-q8-0] [--sections-from BASE.basalt]
        [--mmproj FILE | --no-vision] [--mtp-source DIR] [--draft-vocab FILE] [--profile FILE | --profile-from TRACE ...]
        [--hc8 FILE] [--name NAME] [--workers N] [--force]

Inputs
  --gguf FILE        the model's GGUF: shard 1 of a split model (the others are found beside it) or --gguf per shard
  --sections-from BASE.basalt  with --gguf: the drafter, profile, hc8 and vision sections are copied from BASE byte for
                     byte (BASE's vocabulary must be the GGUF's); replaces --mmproj, --mtp-*, --draft-vocab, --profile*, --hc8
  --mmproj FILE      the vision projector GGUF (required unless --no-vision)
  --mtp-source DIR   the draft layer's BF16 tensors from `mtp_fetch.py fetch --out DIR`; without it there is no drafter
  --mtp-dense FILE   a tuned `mtp/dense` section (raw bytes, same layout as the source; needs --mtp-source); the packs
                     with a self-distilled MTP layer carry one. Its size is checked and its SHA-256 goes to meta.mtp.
                     With --sections-from and --checkpoint it replaces the mtp/dense copied from BASE (same size required)
  --draft-vocab FILE the token subset the draft head may propose (int32 ids; needs --mtp-source)
  --profile FILE     an expert profile stored as is (`--expert-profile-save`'s file)
  --profile-from TRACE ...  `--dump-routing` traces: routed experts first, most frequent first, then the rest;
                     with neither option the ranking is the plain fill order
  --hc8 FILE         8-bit hyper-connection copies from `--dump-hc8`; without them the engine quantizes at every start

From a quantized checkpoint (--checkpoint)
  --checkpoint DIR   NVIDIA Model Optimizer's safetensors (config.json, the model-NNNNN-of-NNNNN shards, in the
                     directory or in tensors/, the n-gram table, tokenizer files, generation_config.json, README.md).
                     The routed experts keep their E2M1 codes and UE4M3 scales, read plain (`weight_scale`) or from
                     Model Optimizer's CSF scale stream (`weight_scale.nvfp4_csf_fixed` with its exceptions), moved
                     into Basalt's blocks with the FP32 tail {s_gate, s_up, s_down, 0} from weight_scale_2; the
                     projections are quantized once from BF16, or from MXFP8 (E4M3 x UE8M0, which is exact), to Q6_K
                     (Q8_0 where rows are not whole 256-value blocks, and for the output head and token embedding);
                     the other tensors keep BF16 or become F32/F16 as the engine reads them; the n-gram table is FP8
                     code x scale as Q8_0, or carried over as NVFP4 rows when the checkpoint stores it that way. The
                     GGUF metadata and the tokenizer come from the checkpoint's files.
  --dense-q8-0       Q8_0 projections instead of Q6_K (the 8-bit dense variant)
  With --sections-from BASE the drafter, expert profile and vision sections come from BASE byte for byte (BASE's
  vocabulary must be the checkpoint's); without BASE they are built here from --mtp-source, --mmproj and --profile-from.

Dense requantization
    python3 tools/convert/convert.py --requant-dense-q6k PACK.basalt [--checkpoint DIR] --out MODEL.basalt [--workers N] [--force]
  Every Q8_0 dense projection whose rows are whole 256-value blocks (GDN qkv/z/out, attention q/k/v/o, the shared
  experts' gate and up) is written as Q6_K; everything else is copied byte for byte. --checkpoint DIR quantizes from the
  BF16 shards (each tensor is checked against the Q8_0 copy in PACK); without it the Q8_0 values are requantized.

Options
  --name NAME        the model's name, reported by the server
  --compat-bf16      convert small projections stored in another form to BF16 (round-to-nearest-even, recorded in
                     meta.conversions)
  --no-vision        leave out the vision projector
  --force            overwrite --out

Sections are listed in include/basalt/core/model_format.hpp. Everything is streamed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))   # for model_file, gguf_reader, tokenizer
import model_sections as S  # noqa: E402
from model_file import TENSOR_ALIGN, ModelFile, ModelWriter, Stream  # noqa: E402

FINGERPRINT_BYTES = 64 << 20

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
LICENSE_TEXT = (HERE / "qwen-community-license-1.0.txt").read_text(encoding="utf-8")


def model_license(kv: dict) -> dict:
    """The model's license, with its full text (the license requires it in every copy)."""
    lic = {"name": "Qwen Community License 1.0", "text": LICENSE_TEXT}
    if kv.get("general.quantized_by"):
        lic["quantized_by"] = kv["general.quantized_by"]
    return lic


def die(msg: str):
    raise SystemExit(f"convert: {msg}")


# ------------------------------------------------------------------ helpers
def fingerprint(path: Path) -> str:
    """sha256 over the first and last 64 MiB and the size: fast, and enough to tell two files apart."""
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(FINGERPRINT_BYTES))
        f.seek(max(0, size - FINGERPRINT_BYTES))
        h.update(f.read(FINGERPRINT_BYTES))
    h.update(struct.pack("<Q", size))
    return h.hexdigest()


def git_rev():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except OSError:
        return None


def json_safe(v):
    """GGUF metadata as JSON: non-finite floats become strings, which JSON cannot carry."""
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    if isinstance(v, list):
        return [json_safe(x) for x in v]
    return v


def load_vision_helper():
    import pack_vision
    return pack_vision.vision_sections


def export_tokenizer(first_gguf: Path, out_dir: Path) -> list[Path]:
    """The tokenizer files (tools/tokenizer.py) of the model's first shard, written under `out_dir`/tokenizer."""
    import tokenizer
    tokenizer.extract(first_gguf, out_dir)
    return sorted(p for p in (out_dir / "tokenizer").iterdir() if p.is_file())


def source_tensors(model: S.SourceModel):
    """(kv, [(name, shard path, absolute offset, size, TensorInfo)]) over every shard, expert tensors excluded."""
    kv, tensors = {}, []
    for p, g in zip(model.paths, model.files):
        for k, v in g.metadata.items():
            kv.setdefault(k, json_safe(v))
    for name, (g, t, p) in model.where.items():
        if S.is_expert(name):
            continue
        a, n = g.data_start + t.offset, t.expected_bytes()
        tensors.append((name, p, a, n, t))
    order = {p: i for i, p in enumerate(model.paths)}
    tensors.sort(key=lambda x: (order[x[1]], x[2]))      # sequential reads per shard
    return kv, tensors


# ------------------------------------------------------------------ the conversion
class Job:
    def __init__(self, writer: ModelWriter):
        self.w = writer
        self.rows: list[tuple[str, int]] = []
        self.t0 = time.time()

    def _report(self, name, nbytes, secs):
        mbs = nbytes / (1 << 20) / secs if secs > 0 else float("inf")
        print(f"  {name:<44} {nbytes / (1 << 30):9.3f} GiB {mbs:9.0f} MB/s", flush=True)
        self.rows.append((name, nbytes))

    def bytes(self, name, data: bytes):
        t = time.time()
        self.w.add_bytes(name, data)
        self._report(name, len(data), time.time() - t)

    def file(self, name, src, offset=0, size=None):
        t = time.time()
        self.w.add_file(name, str(src), offset, size)
        self._report(name, self.w.entries[-1][2], time.time() - t)

    def chunks(self, name, chunks):
        t = time.time()
        n = self.w.add_chunks(name, chunks)
        self._report(name, n, time.time() - t)


def source_paths(given) -> list[Path]:
    paths = [Path(p) for p in given]
    for p in paths:
        if not p.is_file():
            die(f"missing input: {p}")
    return S.expand_shards(paths[0]) if len(paths) == 1 else paths


def convert(a) -> None:
    out = Path(a.out)
    partial = Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    base = None
    if a.sections_from:
        base = ModelFile(a.sections_from) if Path(a.sections_from).is_file() else die(f"missing input: {a.sections_from}")
        if a.mtp_source or a.mtp_dense or a.draft_vocab or a.profile or a.profile_from or a.hc8 or a.mmproj or a.no_vision:
            die("--sections-from supplies the drafter, profile, hc8 and vision sections; do not combine it with "
                "--mtp-source, --mtp-dense, --draft-vocab, --profile, --profile-from, --hc8, --mmproj or --no-vision")
        base_meta = json.loads(base.read("meta"))
    mmproj = Path(a.mmproj) if a.mmproj else None
    side = [Path(x) for x in (a.hc8, a.draft_vocab, a.profile, a.mtp_dense) if x] + [Path(t) for t in (a.profile_from or [])]
    if not a.no_vision and base is None:
        side.append(mmproj if mmproj else Path("<--mmproj>"))
    for p in side:
        if not p.is_file():
            die(f"missing input: {p}")
    if a.draft_vocab and not a.mtp_source:
        die("--draft-vocab needs --mtp-source")
    if a.mtp_dense and not a.mtp_source:
        die("--mtp-dense needs --mtp-source")
    if a.profile and a.profile_from:
        die("give --profile or --profile-from, not both")

    err = None
    try:
        print("reading the source GGUFs ...", flush=True)
        model = S.SourceModel(source_paths(a.gguf))
        kv, tensors = source_tensors(model)
        dense = S.Dense(model, a.compat_bf16)
        experts = S.Experts(model)
        mtp = S.Mtp(a.mtp_source, a.mtp_dense) if a.mtp_source else None
        n_vocab = len(kv.get("tokenizer.ggml.tokens") or []) or None
        draft_vocab = S.draft_vocab_bytes(a.draft_vocab, n_vocab) if a.draft_vocab else None
        if base is not None:
            if base_meta.get("gguf_kv", {}).get("tokenizer.ggml.tokens") != kv.get("tokenizer.ggml.tokens"):
                die(f"{a.sections_from}s vocabulary differs from the GGUFs; its drafter and vision sections would not fit")
            profile_blob = base.read("profile")
        elif a.profile:
            profile_blob = Path(a.profile).read_bytes()
        else:
            freq = {}
            for t in a.profile_from or []:
                for p, c in S.read_trace(t, experts.n_layers, experts.n_expert).items():
                    freq[p] = freq.get(p, 0) + c
            if not a.profile_from:
                print("note: no profile and no routing traces: the ranking is the plain fill order", flush=True)
            profile_blob = S.build_profile(S.rank_pairs(freq, experts.n_layers, experts.n_expert),
                                           experts.n_layers, experts.n_expert)
        prof = S.profile_info(profile_blob, experts.n_layers, experts.n_expert)
    except S.ConvertError as e:
        err = str(e)
    if err is not None:
        # Drop every view of the source files so they are unmapped (and deletable) before the process exits.
        model = dense = experts = mtp = None
        gc.collect()
        die(err)

    vision_extra = []
    if not a.no_vision and base is None:
        fn = load_vision_helper()
        if fn is None:
            print("note: pack_vision.py (vision_sections) not found; only vision/mmproj is written")
        else:
            vision_extra = list(fn(str(mmproj), [str(p) for p in model.paths]))

    if out.parent and not out.parent.exists():
        die(f"{out.parent} does not exist")
    if partial.exists():
        partial.unlink()
    w = ModelWriter(str(partial))
    job = Job(w)
    ok = False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            print("writing sections:", flush=True)
            for p in export_tokenizer(model.paths[0], Path(tmp)):
                job.file(f"tokenizer/{p.name}", p)
            job.chunks("dense", dense.chunks())
            job.bytes("dense.index", dense.index_text())
            job.chunks("experts", experts.chunks())
            job.bytes("experts.layout", experts.layout_text())
            t = time.time()
            offs = w.add_parts("gguf", [(src, off, size) for _, src, off, size, _ in tensors], TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            tensor_offsets = dict(zip((x[0] for x in tensors), offs))
            copied = []
            if base is not None:
                for name in base.names():
                    if name != "vision/text_vocab" and (name.startswith(("mtp/", "vision/")) or name in ("profile", "hc8")):
                        off, size = base.section(name)
                        job.file(name, a.sections_from, off, size)
                        if w.entries[-1][3] != base.sections[name][2]:
                            die(f"section {name}: the copys SHA-256 differs from {a.sections_from}s")
                        copied.append(name)
                if "vision/text_vocab" in base.sections:
                    import pack_vision
                    for p in model.paths:
                        data = pack_vision.vocab_only_gguf(str(p))
                        if data is not None:
                            job.bytes("vision/text_vocab", data)
                            break
                    else:
                        die("none of the source GGUF files carries the tokenizer metadata the vision sidecar needs")
            if mtp is not None:
                job.chunks("mtp/dense", mtp.dense_chunks())
                job.bytes("mtp/dense.index", mtp.dense_index())
                job.chunks("mtp/experts", mtp.expert_chunks())
                if draft_vocab is not None:
                    job.bytes("mtp/draft_vocab", draft_vocab)
            if base is None:
                job.bytes("profile", profile_blob)
            if a.hc8:
                job.file("hc8", a.hc8)
            written = {e[0] for e in w.entries}
            if not a.no_vision and base is None:
                job.file("vision/mmproj", mmproj)
                for name, data, src, off, size in vision_extra:
                    if name in written or name == "vision/mmproj":
                        print(f"note: vision section {name} already written, skipped")
                        continue
                    if data is not None:
                        job.bytes(name, data)
                    else:
                        job.file(name, src, off, size)

            sources = [{"path": p.name, "size": p.stat().st_size,
                        "fingerprint_sha256_first_last_64mib_and_size": fingerprint(p)}
                       for p in model.paths + ([mmproj] if not a.no_vision and base is None else [])]
            meta = {
                "format": "basalt", "version": 1, "name": a.name,
                "converter": {"tool": "tools/convert/convert.py", "git": git_rev()},
                "sources": sources,
                "gguf_kv": kv,
                "tensors": {n: {"type": t.type_name, "type_id": t.type_id, "shape": [int(d) for d in t.shape],
                                "bytes": nb, "offset": tensor_offsets[n]}
                            for n, _, _, nb, t in tensors},
                "conversions": {"compat_bf16": a.compat_bf16, "tensors": dense.records},
                "sampling": SAMPLING, "license": model_license(kv),
                "profile": prof,
                "hc8": base_meta.get("hc8") if base is not None else {"bytes": Path(a.hc8).stat().st_size} if a.hc8 else None,
                "mtp": base_meta.get("mtp") if base is not None else None if mtp is None else {"source_sha256": mtp.source_sha256(),
                                                  "expert_format": "q2_0", "draft_vocab": draft_vocab is not None,
                                                  "dense": ({"tuning": "self_distilled", "sha256": mtp.tuned_sha256}
                                                            if mtp.tuned else {"tuning": None})},
                "vision": base_meta.get("vision") if base is not None else None if a.no_vision else {"mmproj": "vision/mmproj", "mmproj_bytes": mmproj.stat().st_size,
                                                    "extra": [v[0] for v in vision_extra]},
            }
            if base is not None:
                meta["sections_from"] = {"path": Path(a.sections_from).name, "size": Path(a.sections_from).stat().st_size,
                                         "fingerprint_sha256_first_last_64mib_and_size": fingerprint(Path(a.sections_from)),
                                         "copied": copied, "regenerated": ["vision/text_vocab"]}
            job.bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        w.close()
        ok = True
    except S.ConvertError as e:
        err = str(e)
    finally:
        if not ok:
            try:
                w.f.close()
            except OSError:
                pass
            partial.unlink(missing_ok=True)
    if err is not None:
        model = dense = experts = mtp = job = None
        gc.collect()
        die(err)
    os.replace(partial, out)
    total = out.stat().st_size
    print(f"done: {len(job.rows)} sections, {total / (1 << 30):.3f} GiB, {time.time() - job.t0:.1f} s -> {out}")


def warn_unofficial():
    msg = ("WARNING: packs converted yourself are significantly slower than the official .basalt files. "
           "Download them from https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt")
    if sys.stderr.isatty():
        msg = f"\033[1;31m{msg}\033[0m"
    print(msg, file=sys.stderr, flush=True)


def from_checkpoint(a) -> None:
    """The language model straight from an NVFP4 safetensors checkpoint (routed experts moved into Basalt's layout,
    every other tensor converted once, the n-gram table in the form the checkpoint keeps it, metadata and tokenizer
    from the checkpoint's files). With BASE (`--sections-from`) the drafter, expert profile, hc8 and vision sections
    are copied from it byte for byte; without BASE they are built here: the drafter from `--mtp-source`, the projector
    from `--mmproj`, the ranking from routing traces or the plain fill order."""
    import multiprocessing
    import checkpoint as C
    out = Path(a.out)
    partial = Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    base = None
    src = Path(a.sections_from) if a.sections_from else None
    if src is not None and not src.is_file():
        die(f"missing input: {src}")
    mmproj = Path(a.mmproj) if a.mmproj else None
    if a.mtp_source and not Path(a.mtp_source).is_dir():
        die(f"missing input: {a.mtp_source}")
    for p in [x for x in (a.draft_vocab, a.hc8, a.mtp_dense) if x] + [Path(t) for t in (a.profile_from or [])] \
            + ([mmproj] if src is None and not a.no_vision else []):
        if not Path(p).is_file():
            die(f"missing input: {p}")
    if a.draft_vocab and not a.mtp_source:
        die("--draft-vocab needs --mtp-source")
    q6k = not a.dense_q8_0
    try:
        ck = C.Checkpoint(a.checkpoint)
        experts = C.CheckpointExperts(ck)
        dense_src = C.CheckpointDense(ck, q6k)
        ple = C.ple_table(ck)
        kv = C.model_kv(ck, S.PLE_DIM, 18 if q6k else 7, a.name)
        if not ple.folded:      # an NVFP4 table keeps its scales; the engine needs the table's F32 one with them
            kv["qwen4exp.ple.nvfp4_scale_2"] = C.f32(ple.scale)
        dense = S.Dense(dense_src, False)
        mtp = draft_vocab = vocab_blob = None
        profile_blob = prof = None
        if src is None:
            if a.mtp_source:
                mtp = S.Mtp(a.mtp_source, None)
                if a.draft_vocab:
                    draft_vocab = S.draft_vocab_bytes(a.draft_vocab, int(ck.text["vocab_size"]))
            freq = {}
            for t in a.profile_from or []:
                for p, c in S.read_trace(t, experts.n_layers, experts.n_expert).items():
                    freq[p] = freq.get(p, 0) + c
            if not a.profile_from:
                print("note: no routing traces: the expert ranking is the plain fill order", flush=True)
            profile_blob = S.build_profile(S.rank_pairs(freq, experts.n_layers, experts.n_expert),
                                           experts.n_layers, experts.n_expert)
            prof = S.profile_info(profile_blob, experts.n_layers, experts.n_expert)
            if not a.no_vision:
                import pack_vision
                # made before any weight is written: a tokenizer metadata that will not serialize is found in
                # seconds rather than after an hour of weights
                vocab_blob = pack_vision.vocab_gguf_bytes(kv)
    except S.ConvertError as e:
        die(str(e))
    if src is not None:
        base = ModelFile(str(src))
        base_meta = json.loads(base.read("meta"))
        old = base.read("experts.layout").decode("utf-8").splitlines()
        m = re.search(r"\(n_expert (\d+)", old[0])
        n_layers = sum(1 for x in old if x.strip() and not x.startswith("#"))
        if not m or int(m.group(1)) != experts.n_expert or n_layers != experts.n_layers:
            die(f"{src} has {n_layers} layers of {m.group(1) if m else '?'} experts, the checkpoint {experts.n_layers} "
                f"of {experts.n_expert}")
        if base_meta.get("gguf_kv", {}).get("tokenizer.ggml.tokens") != kv["tokenizer.ggml.tokens"]:
            die(f"{src}'s vocabulary differs from the checkpoint's; its drafter and vision sections would not fit")
    else:
        base_meta = {}
    if partial.exists():
        partial.unlink()
    w = ModelWriter(str(partial))
    job = Job(w)
    ok = False
    try:
        with tempfile.TemporaryDirectory() as tmp, multiprocessing.Pool(a.workers) as pool:
            import tokenizer
            tokenizer.extract_metadata(kv, Path(tmp))
            print("writing sections:", flush=True)
            for p in sorted(q for q in (Path(tmp) / "tokenizer").iterdir() if q.is_file()):
                job.file(f"tokenizer/{p.name}", p)
            job.chunks("dense", dense.chunks())
            job.bytes("dense.index", dense.index_text())
            # the queued expert blobs are ~90 MB a chunk: bound how many are in flight by how many workers make them
            job.chunks("experts", experts.chunks(pool, depth=2 * a.workers))
            job.bytes("experts.layout", experts.layout_text())
            t = time.time()
            parts = []
            for n, info in dense_src.tensors.items():
                if info.type_name in ("Q8_0", "Q6_K"):
                    parts.append(Stream(info.expected_bytes(), dense_src.chunks(n, pool)))
                else:
                    parts.append(dense_src.bytes(n).tobytes())
            parts.append(Stream(ple.nbytes, ple.chunks(pool)))
            offs = w.add_parts("gguf", parts, TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            names = list(dense_src.tensors) + ["per_layer_token_embd.weight"]
            tensor_offsets = dict(zip(names, offs))
            tuned_sha = None
            if src is not None:
                for name in base.names():
                    if name.startswith(("mtp/", "vision/")) or name in ("profile", "hc8"):
                        off, size = base.section(name)
                        if name == "mtp/dense" and a.mtp_dense:
                            tuned = Path(a.mtp_dense)
                            if tuned.stat().st_size != size:
                                die(f"{tuned}: {tuned.stat().st_size} bytes, {src}'s mtp/dense is {size}")
                            h = hashlib.sha256()
                            with open(tuned, "rb") as f:
                                for b in iter(lambda: f.read(1 << 24), b""):
                                    h.update(b)
                            tuned_sha = h.hexdigest()
                            job.file(name, tuned)
                            continue
                        job.file(name, src, off, size)
                for name, _, _, digest in w.entries:
                    if name in base.sections and name.startswith(("mtp/", "vision/", "profile", "hc8")) \
                            and not (name == "mtp/dense" and a.mtp_dense) \
                            and digest != base.sections[name][2]:
                        die(f"section {name}: the copy's SHA-256 differs from {src}'s")
            else:
                if mtp is not None:
                    job.chunks("mtp/dense", mtp.dense_chunks())
                    job.bytes("mtp/dense.index", mtp.dense_index())
                    job.chunks("mtp/experts", mtp.expert_chunks())
                    if draft_vocab is not None:
                        job.bytes("mtp/draft_vocab", draft_vocab)
                job.bytes("profile", profile_blob)
                if a.hc8:
                    job.file("hc8", a.hc8)
                if not a.no_vision:
                    job.file("vision/mmproj", mmproj)
                    job.bytes("vision/text_vocab", vocab_blob)
            tens = {n: {"type": t.type_name, "type_id": t.type_id, "shape": [int(d) for d in t.ne],
                        "bytes": t.expected_bytes(), "offset": tensor_offsets[n]} for n, t in dense_src.tensors.items()}
            tens["per_layer_token_embd.weight"] = {"type": ple.gguf_type, "type_id": ple.gguf_type_id,
                                                  "shape": [S.PLE_DIM, ple.rows], "bytes": ple.nbytes,
                                                  "offset": tensor_offsets["per_layer_token_embd.weight"]}
            ck_files = sorted(set(ck.shards) | set(ple.files))
            meta = {
                "format": "basalt", "version": 1, "name": a.name or base_meta.get("name"),
                "converter": {"tool": "tools/convert/convert.py", "git": git_rev()},
                "sources": [{"path": p.name, "size": p.stat().st_size,
                             "fingerprint_sha256_first_last_64mib_and_size": fingerprint(p)} for p in ck_files],
                "gguf_kv": kv,
                "tensors": tens,
                "conversions": {"compat_bf16": False,
                                "tensors": [{**ple.conversion, "scale": ple.scale}]},
                "sampling": SAMPLING, "license": model_license(kv),
            }
            if src is not None:
                for k in ("profile", "hc8", "mtp", "vision"):
                    meta[k] = base_meta.get(k)
                if tuned_sha:
                    meta["mtp"] = {**meta["mtp"], "dense": {"tuning": "self_distilled", "sha256": tuned_sha}}
            else:
                meta["profile"] = prof
                meta["hc8"] = {"bytes": Path(a.hc8).stat().st_size} if a.hc8 else None
                meta["mtp"] = None if mtp is None else {
                    "source_sha256": mtp.source_sha256(), "expert_format": "q2_0",
                    "draft_vocab": draft_vocab is not None, "dense": {"tuning": None}}
                meta["vision"] = None if a.no_vision else {
                    "mmproj": "vision/mmproj", "mmproj_bytes": mmproj.stat().st_size,
                    "extra": ["vision/text_vocab"]}
            n_mxfp8 = sum(1 for t in dense_src.tensors.values() if t.scale)
            used = sorted({t.how for t in dense_src.tensors.values()})
            meta["checkpoint"] = {
                "dir": Path(a.checkpoint).name, "config_sha256": hashlib.sha256((ck.dir / "config.json").read_bytes()).hexdigest(),
                "experts": {"format": "nvfp4",
                            "source": "weight (E2M1 pairs), weight_scale.nvfp4_csf_fixed and "
                                      ".nvfp4_csf_exceptions (UE4M3 per 16 values, decoded to bytes), "
                                      "weight_scale_2 (FP32); input_scale unused" if experts.csf else
                                      "weight (E2M1 pairs), weight_scale (UE4M3 per 16), weight_scale_2 (FP32); "
                                      "input_scale unused",
                            "csf_decoded": experts.csf,
                            "repack": "codes and scales unchanged, "
                            "nibbles moved into 64-value blocks", "tail": "fp32 {s_gate, s_up, s_down, 0} per expert"},
                "dense": {"projections": "Q6_K where rows are whole 256-value blocks, else Q8_0" if q6k else "Q8_0",
                          "q6_k_method": "least-squares 16-value group scales, 8-bit group scales, fp16 super-scale",
                          "q8_0_method": "ggml reference rounding",
                          "from": "BF16, quantized once" if not n_mxfp8 else "BF16, or MXFP8 dequantized exactly, "
                                  "quantized once",
                          "mxfp8_dequantized": n_mxfp8,
                          "transforms": {k: C.METHODS[k] for k in used}}}
            if src is not None:
                meta["checkpoint"]["sections_from"] = {
                    "path": src.name, "size": src.stat().st_size,
                    "fingerprint_sha256_first_last_64mib_and_size": fingerprint(src),
                    "copied": [n for n in base.names() if n.startswith(("mtp/", "vision/")) or
                               n in ("profile", "hc8")]}
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


def requant_dense(a) -> None:
    """BASE with its Q8_0 dense projections written as Q6_K; every other section and tensor is copied byte for byte."""
    import multiprocessing
    out, src = Path(a.out), Path(a.requant_dense_q6k)
    partial = Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    if not src.is_file():
        die(f"missing input: {src}")
    if a.checkpoint and not Path(a.checkpoint).is_dir():
        die(f"missing input: {a.checkpoint}")
    base = ModelFile(str(src))
    meta = json.loads(base.read("meta"))
    gguf_off, _ = base.section("gguf")
    tensors = meta["tensors"]
    try:
        plan = S.DenseQ6K(tensors, a.checkpoint)
    except S.ConvertError as e:
        die(str(e))
    if not plan.plan:
        die(f"{src} has no Q8_0 dense projection to requantize")
    order = sorted(tensors, key=lambda n: tensors[n]["offset"])
    if partial.exists():
        partial.unlink()
    w = ModelWriter(str(partial))
    job = Job(w)
    ok = False
    try:
        with multiprocessing.Pool(a.workers) as pool:
            for name in base.names():
                if name == "meta":
                    continue
                if name != "gguf":
                    off, size = base.section(name)
                    job.file(name, src, off, size)
                    continue
                t = time.time()
                parts, sizes = [], {}
                for n in order:
                    info = tensors[n]
                    if n in plan.plan:
                        sizes[n] = plan.nbytes(info)
                        parts.append(Stream(sizes[n], plan.chunks(n, info, str(src), gguf_off, pool)))
                    else:
                        sizes[n] = info["bytes"]
                        parts.append((str(src), gguf_off + info["offset"], info["bytes"]))
                offs = w.add_parts("gguf", parts, TENSOR_ALIGN)
                job._report("gguf", w.entries[-1][2], time.time() - t)
            new = {n: dict(tensors[n]) for n in tensors}
            for n, o in zip(order, offs):
                new[n]["offset"] = o
                if n in plan.plan:
                    new[n].update({"type": "Q6_K", "type_id": 14, "bytes": sizes[n]})
            meta["tensors"] = new
            meta["converter"] = {"tool": "tools/convert/convert.py", "git": git_rev()}
            meta["dense_requant"] = {
                "from": {"path": src.name, "size": src.stat().st_size,
                         "fingerprint_sha256_first_last_64mib_and_size": fingerprint(src)},
                "to": "Q6_K", "source": "checkpoint BF16" if a.checkpoint else "pack Q8_0 values (two roundings)",
                "tensors": sorted(plan.plan), "max_bf16_to_q8_0_steps": plan.worst,
                "method": "least-squares 16-value group scales, 8-bit group scales, fp16 super-scale"}
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
    print(f"done: {len(job.rows)} sections, {len(plan.plan)} tensors to Q6_K, {out.stat().st_size / (1 << 30):.3f} GiB,"
          f" max BF16-to-Q8_0 distance {plan.worst:.3f} steps -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections-from", metavar="PACK.basalt",
                    help="with --checkpoint: the drafter, vision projector and expert profile (and hc8 if it has them) come from "
                         "this .basalt; everything else from the checkpoint")
    ap.add_argument("--gguf", action="append", metavar="FILE",
                    help="the model's GGUF: shard 1 of a split model (the others are found beside it), or repeat per shard")
    ap.add_argument("--mmproj", help="vision projector GGUF (required unless --no-vision)")
    ap.add_argument("--out", required=True, help="the .basalt file to write")
    ap.add_argument("--mtp-source", metavar="DIR", help="the drafter's BF16 tensors written by tools/convert/mtp_fetch.py fetch")
    ap.add_argument("--mtp-dense", metavar="FILE", help="a tuned (self-distilled) mtp/dense section to embed (needs --mtp-source)")
    ap.add_argument("--draft-vocab", metavar="FILE", help="int32 token ids the draft head may propose (needs --mtp-source)")
    ap.add_argument("--profile", metavar="FILE", help="an expert profile to store as it is (--expert-profile-save's file)")
    ap.add_argument("--profile-from", nargs="+", metavar="TRACE",
                    help="routing traces from `basalt generate --dump-routing`: the ranking is built from them")
    ap.add_argument("--hc8", metavar="FILE", help="file written by `basalt generate --model ... --dump-hc8`")
    ap.add_argument("--name")
    ap.add_argument("--compat-bf16", action="store_true",
                    help="convert small projections stored in another form to BF16 (rounds)")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--requant-dense-q6k", metavar="PACK.basalt",
                    help="copy PACK with its Q8_0 dense projections requantized to Q6_K")
    ap.add_argument("--checkpoint", metavar="DIR",
                    help="a quantized safetensors checkpoint (NVFP4 experts, BF16 or MXFP8 projections): with "
                         "--sections-from the language model is read from it and the other sections from that pack; "
                         "alone it builds every section; with --requant-dense-q6k, the BF16 projections")
    ap.add_argument("--dense-q8-0", action="store_true",
                    help="with --checkpoint: Q8_0 projections instead of Q6_K")
    ap.add_argument("--workers", type=int, default=16, help="with --checkpoint: converter processes")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    a = ap.parse_args()
    if a.requant_dense_q6k:
        if a.sections_from or a.gguf or a.dense_q8_0:
            ap.error("--requant-dense-q6k takes only --checkpoint, --out, --workers and --force")
        requant_dense(a)
        return
    if a.checkpoint:
        if a.gguf:
            ap.error("--checkpoint reads the language model from the checkpoint; it takes no --gguf")
        if not Path(a.checkpoint).is_dir():
            ap.error(f"missing input: {a.checkpoint}")
        if a.mtp_dense and not Path(a.mtp_dense).is_file():
            ap.error(f"missing input: {a.mtp_dense}")
        if a.mtp_dense and not a.sections_from:
            ap.error("--mtp-dense replaces the mtp/dense copied from --sections-from")
        if a.sections_from:
            if a.mmproj or a.mtp_source or a.draft_vocab or a.profile \
                    or a.profile_from or a.hc8:
                ap.error("--checkpoint with --sections-from takes only --out, --name, --mtp-dense, --dense-q8-0, --workers and --force")
        elif not a.no_vision and not a.mmproj:
            ap.error("--checkpoint without --sections-from needs --mmproj (or --no-vision); the drafter comes from "
                     "--mtp-source and the expert ranking from --profile-from")
        from_checkpoint(a)
        return
    if a.dense_q8_0:
        ap.error("--dense-q8-0 needs --checkpoint")
    if not a.gguf:
        ap.error("--gguf is required")
    a.name = a.name or ("Qwen3.8-Flash-Next " + Path(a.gguf[0]).name.split("Next-")[-1].split("-0000")[0] if a.sections_from
                        else "Qwen3.8-Flash-Next IQ3_XXS")
    if not a.no_vision and not a.mmproj and not a.sections_from:
        ap.error("--mmproj is required unless --no-vision")
    warn_unofficial()
    convert(a)


if __name__ == "__main__":
    main()
