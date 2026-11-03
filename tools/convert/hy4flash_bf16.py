"""The Hy4-flash release pack from the BF16 GGUF (`convert.py --gguf Hy4-flash-BF16.gguf --nvfp4 ...`), for tools/convert/convert.py.

  experts / experts.layout  the routed experts rounded once to NVFP4 (ggml type 40), the 27B pack's rounding: per matrix
                            the tensor scale amax / (6 x 448), per 16 values the UE4M3 scale of least squared error
                            near amax / 6; blobs [gate | up | down] plus the FP32 tail {s_gate, s_up, s_down, 0}
  gguf / dense / dense.index  every other tensor: BF16 projections as Q8_0 (Q6_K with --dense-q6k where rows are whole
                            256-value blocks and the tensor is one of Q6K_TARGETS), F32 tensors as stored, the head
                            (F32 holding BF16 values) as BF16
  engram                    the Engram table rounded once to E4M3 codes (llama.cpp's type 43, no scale), 96 bytes a row
  tokenizer/*               the GGUF's, checked against the Hugging Face directory's tokenizer.json and chat template

Nothing is held whole: tensors are read in pieces by worker processes, the page cache of the source is dropped as it is
read, and `verify` repeats every rounding (they are deterministic) and compares the bytes with the pack's.
Needs numpy.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import checkpoint as C  # noqa: E402
import checkpoint_qwen35 as Q  # noqa: E402
import model_sections as S  # noqa: E402
from arch.hy4flash import ENGRAM_SECTION, ENGRAM_TENSOR, F8_E4M3_RAW, Hy4Flash  # noqa: E402
from model_file import SECTION_ALIGN, TENSOR_ALIGN, ModelFile, ModelWriter, Stream  # noqa: E402

TYPE_ID = {"F32": 0, "Q8_0": 8, "Q6_K": 14, "BF16": 30, "NVFP4": 40}
ROLES = ("gate", "up", "down")
ENGRAM_VALUES_PER_JOB = 1 << 22
ROWS_VALUES_PER_JOB = 1 << 22
EXPERTS_PER_JOB = 4
DEFAULT_NAME = "Hy4-flash NVFP4"

# BF16 tensors the --dense-q6k variant rounds to Q6_K (when their rows are whole 256-value blocks); every other BF16
# tensor, the token embedding included, is Q8_0
Q6K_TARGETS = re.compile(r"blk\.\d+\.(attn_q_a|attn_q_b|attn_kv_a_mqa|attn_gate|attn_output|ffn_(gate|up|down)_shexp|"
                         r"ffn_routed_(down|up)|ffn_(gate|up|down))\.weight")


class Hy4FlashBf16(Hy4Flash):
    """Hy4Flash for the BF16 GGUF: a BF16 Engram table, no next-n block, the head stored as BF16."""
    engram_type = TYPE_ID["BF16"]
    needs_nextn = False
    form_table = {"output.weight": "BF16"}


ARCH = Hy4FlashBf16()


# ------------------------------------------------------------------ reading and the rounding of one piece
def pread(path, off: int, n: int) -> bytearray:
    """`n` bytes at `off`. Where the OS has it (Linux), the source's page cache is dropped behind the read so a long conversion
    keeps the box's RAM; elsewhere (Windows) a plain seek and read."""
    buf = bytearray(n)
    mv = memoryview(buf)
    if not hasattr(os, "preadv"):
        with open(path, "rb") as f:
            f.seek(off)
            got = 0
            while got < n:
                k = f.readinto(mv[got:])
                if not k:
                    raise IOError(f"{path}: short read at {off + got}")
                got += k
        return buf
    fd = os.open(path, os.O_RDONLY)
    try:
        got = 0
        while got < n:
            k = os.preadv(fd, [mv[got:]], off + got)
            if k <= 0:
                raise IOError(f"{path}: short read at {off + got}")
            got += k
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return buf


def bf16_values(buf) -> np.ndarray:
    return S._bf16_to_f32(np.frombuffer(buf, dtype="<u2"))


def _expert_job(args):
    """Blobs of experts [e0, e1) of one layer: each [gate | up | down] NVFP4 blocks and the FP32 scale tail."""
    path, mats, e0, e1 = args       # mats: three (offset, rows, cols) of the whole [expert, rows, cols] tensors
    out = []
    for e in range(e0, e1):
        tail = []
        for off, rows, cols in mats:
            nb = rows * cols * 2
            x = bf16_values(pread(path, off + e * nb, nb)).reshape(rows, cols)
            if not np.isfinite(x).all():
                return f"expert {e}: a non-finite value"
            gs = Q.fp4q_global(float(np.abs(x).max()))
            codes, sc = Q.fp4q_rows(x, gs)
            out.append(C.nvfp4_rows(codes, sc).tobytes())
            tail.append(gs)
        out.append(np.array(tail + [0.0], dtype="<f4").tobytes())
    return b"".join(out)


def _rows_job(args):
    """BF16 values [row0, row0 + rows) of a matrix of ne0 columns as Q8_0 or Q6_K blocks."""
    path, off, ne0, row0, rows, ty = args
    x = bf16_values(pread(path, off + row0 * ne0 * 2, rows * ne0 * 2))
    if not np.isfinite(x).all():
        return "a non-finite value"
    return S.q8_0(x) if ty == "Q8_0" else S.q6_k(x)


def _head_job(args):
    """F32 values holding BF16 numbers as BF16."""
    path, off, n = args
    u = np.frombuffer(pread(path, off, n * 4), dtype="<u4")
    if (u & 0xFFFF).any():
        return "an F32 value that is not a BF16 number"
    return (u >> 16).astype("<u2").tobytes()


_LUT = None


def bf16_grid():
    """The 65536 BF16 patterns' values (float64; 0 for the non-finite ones) and which are finite."""
    x = S._bf16_to_f32(np.arange(65536, dtype=np.uint16))
    fin = np.isfinite(x)
    return np.where(fin, x, np.float32(0)).astype(np.float64), fin


def engram_lut() -> np.ndarray:
    """E4M3 code of each of the 65536 BF16 patterns: to nearest, ties to the even code, saturating (non-finite patterns
    get the code of 0; the caller counts them)."""
    global _LUT
    if _LUT is None:
        _LUT = Q.e4m3_round(bf16_grid()[0])
    return _LUT


def _engram_job(args):
    """BF16 values of a range of the table as E4M3 codes, and the histogram of the BF16 patterns seen."""
    path, off, n = args
    u = np.frombuffer(pread(path, off, n * 2), dtype="<u2")
    return engram_lut()[u].tobytes(), np.bincount(u, minlength=65536)


def engram_error(hist: np.ndarray) -> dict:
    """What rounding the table to E4M3 cost, from the histogram of its BF16 patterns."""
    bf, fin = bf16_grid()
    dec = S.fp8_e4m3_table().astype(np.float64)[engram_lut()]
    h = hist.astype(np.float64)
    err = np.where(fin, dec - bf, 0.0)
    n, sx = h.sum(), (h * bf ** 2).sum()
    exact = h[fin & (err == 0)].sum()
    return {"values": int(n), "non_finite": int(h[~fin].sum()), "exact_fraction": float(exact / n),
            "rms_relative_error": float(np.sqrt((h * err ** 2).sum() / sx)) if sx > 0 else 0.0,
            "max_abs_error": float(np.abs(err[(h > 0) & fin]).max()),
            "max_abs_value": float(np.abs(bf[(h > 0) & fin]).max()),
            "nonzero_rounded_to_zero": int(h[fin & (bf != 0) & (dec == 0)].sum())}


# ------------------------------------------------------------------ the plan
def die_on(x):
    if isinstance(x, str):
        raise S.ConvertError(x)
    return x


class Plan:
    """What the pack holds, from the BF16 GGUF's tensor table alone."""

    def __init__(self, model: S.SourceModel, q6k: bool):
        self.model, self.q6k = model, q6k
        T = {n: w[1] for n, w in model.where.items()}
        self.paths = {n: w[2] for n, w in model.where.items()}
        exps = sorted(int(m.group(1)) for n in T if (m := re.fullmatch(r"blk\.(\d+)\.ffn_gate_exps\.weight", n)))
        if not exps:
            raise S.ConvertError("the model has no expert tensors (blk.N.ffn_{gate,up,down}_exps.weight)")
        self.layers = exps
        self.n_expert = int(T[f"blk.{exps[0]}.ffn_gate_inp.weight"].shape[1])
        self.dims = None
        for l in exps:
            dims = []
            for r in ROLES:
                t = T.get(f"blk.{l}.ffn_{r}_exps.weight")
                if t is None:
                    raise S.ConvertError(f"layer {l}: missing blk.{l}.ffn_{r}_exps.weight")
                if t.type_name != "BF16" or len(t.shape) != 3 or int(t.shape[2]) != self.n_expert:
                    raise S.ConvertError(f"{t.name} is {t.type_name} {list(t.shape)}; expected BF16 [cols, rows, {self.n_expert}]")
                dims.append((int(t.shape[1]), int(t.shape[0])))
            if self.dims not in (None, dims):
                raise S.ConvertError(f"layer {l}: the expert matrices differ in shape from layer {exps[0]}'s")
            self.dims = dims
        self.blob = sum(rows * S.nvfp4_row_bytes(cols) for rows, cols in self.dims) + S.NVFP4_TAIL
        self.expert_bytes = len(exps) * self.n_expert * self.blob
        self.layout = [(l, TYPE_ID["NVFP4"], TYPE_ID["NVFP4"], i * self.n_expert * self.blob, self.blob)
                       for i, l in enumerate(exps)]
        # every other tensor: name -> (how, stored type)
        self.how: dict[str, tuple[str, str]] = {}
        for n, t in T.items():
            if S.is_expert(n) or n == ENGRAM_TENSOR:
                continue
            self.how[n] = self.classify(n, t)

    def classify(self, name, t):
        tn, ne0 = t.type_name, int(t.shape[0])
        if name == "output.weight":
            if tn not in ("F32", "BF16"):
                raise S.ConvertError(f"{name} is {tn}; expected F32 or BF16 numbers")
            return ("raw", "BF16") if tn == "BF16" else ("head", "BF16")
        if tn == "F32":
            return "raw", "F32"
        if tn != "BF16":
            raise S.ConvertError(f"{name} is {tn}; the BF16 GGUF holds F32 and BF16 tensors")
        if self.q6k and Q6K_TARGETS.fullmatch(name) and ne0 % S.Q6K_BLOCK == 0:
            return "quant", "Q6_K"
        if ne0 % 32:
            raise S.ConvertError(f"{name}: rows of {ne0} values are not whole Q8_0 blocks")
        return "quant", "Q8_0"

    @staticmethod
    def stored_bytes(t, ty: str) -> int:
        n = int(np.prod([int(d) for d in t.shape]))
        return {"Q8_0": n // 32 * 34, "Q6_K": n // S.Q6K_BLOCK * S.Q6K_BYTES, "BF16": n * 2, "F32": n * 4}[ty]

    # -- experts
    def expert_jobs(self, l):
        names = [f"blk.{l}.ffn_{r}_exps.weight" for r in ROLES]
        mats = [(self.model.abs_range(n)[1], rows, cols) for n, (rows, cols) in zip(names, self.dims)]
        path = self.model.abs_range(names[0])[0]
        if any(self.model.abs_range(n)[0] != path for n in names):
            raise S.ConvertError(f"layer {l}: the expert tensors are in different files")
        return [(path, mats, e, min(e + EXPERTS_PER_JOB, self.n_expert)) for e in range(0, self.n_expert, EXPERTS_PER_JOB)]

    def expert_chunks(self, pool):
        for l in self.layers:
            for b in S.bounded_imap(pool, _expert_job, self.expert_jobs(l), 64):
                yield die_on(b)
            print(f"  layer {l} experts", flush=True)

    # -- the `gguf` section's tensors, in file order
    def parts(self, pool):
        """[(name, stored type, part)] for `add_parts`: a (path, offset, size) copy or a Stream."""
        out = []
        for n, (how, ty) in self.how.items():
            path, off, size = self.model.abs_range(n)
            t = self.model.where[n][1]
            nb = self.stored_bytes(t, ty)
            if how == "raw":
                out.append((n, ty, (path, off, size)))
                continue
            if how == "head":
                n_val = size // 4
                step = ROWS_VALUES_PER_JOB
                jobs = [(path, off + v * 4, min(step, n_val - v)) for v in range(0, n_val, step)]
                fn = _head_job
            else:
                ne0, ne1 = int(t.shape[0]), int(np.prod([int(d) for d in t.shape[1:]]))
                step = max(1, ROWS_VALUES_PER_JOB // ne0)
                jobs = [(path, off, ne0, r, min(step, ne1 - r), ty) for r in range(0, ne1, step)]
                fn = _rows_job
            out.append((n, ty, Stream(nb, (die_on(b) for b in S.bounded_imap(pool, fn, jobs, 32)) if pool else iter(()))))
        return out

    def dense(self):
        return QuantDense(self.model, ARCH, self.how)

    # -- the Engram table
    def engram_chunks(self, pool, hist: list):
        path, off, size = self.model.abs_range(ENGRAM_TENSOR)
        n_val = size // 2
        jobs = [(path, off + v * 2, min(ENGRAM_VALUES_PER_JOB, n_val - v)) for v in range(0, n_val, ENGRAM_VALUES_PER_JOB)]
        done = 0
        for b in S.bounded_imap(pool, _engram_job, jobs, 16):
            if isinstance(b, str):
                raise S.ConvertError(b)
            codes, h = b
            hist[0] = h if hist[0] is None else hist[0] + h
            done += len(codes)
            yield codes
            if done // (8 << 30) != (done - len(codes)) // (8 << 30):
                print(f"  engram {done / 1e9:.0f} of {n_val / 1e9:.0f} GB", flush=True)


class QuantDense(S.Dense):
    """`dense` of the float tensors (the others are native rows, as for a GGUF whose projections are quantized), rows in file order."""

    def __init__(self, model, arch, how):
        super().__init__(model, False, arch, only=lambda n: n in how and how[n][0] in ("raw", "head"))
        have = {x[0]: x for x in self.todo}
        self.todo = [have.get(n) or (n, t, p, True) for n, (g, t, p) in model.where.items() if n in how]


# ------------------------------------------------------------------ the tokenizer
def check_tokenizer(kv: dict, hf_dir: Path) -> None:
    """The GGUF's vocabulary, merges, special ids and chat template against the checkpoint's own files."""
    tj = json.loads((hf_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tc = json.loads((hf_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    ids = {v: k for k, v in tj["model"]["vocab"].items()}
    ids.update({a["id"]: a["content"] for a in tj["added_tokens"]})
    toks = kv.get("tokenizer.ggml.tokens") or []
    bad = [i for i, t in enumerate(toks) if ids.get(i) != t]
    if bad or len(toks) != len(ids):
        raise S.ConvertError(f"the GGUF's vocabulary differs from {hf_dir.name}/tokenizer.json at {len(bad)} ids "
                             f"(first: {bad[:4]}); {len(toks)} tokens against {len(ids)}")
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    if merges != list(kv.get("tokenizer.ggml.merges") or []):
        raise S.ConvertError(f"the GGUF's merges differ from {hf_dir.name}/tokenizer.json")
    by_tok = {t: i for i, t in enumerate(toks)}
    for key, field in (("bos", "bos_token"), ("eos", "eos_token"), ("padding", "pad_token")):
        if by_tok.get(tc.get(field)) != kv.get(f"tokenizer.ggml.{key}_token_id"):
            raise S.ConvertError(f"the GGUF's {key} token id is not {tc.get(field)!r}'s id in tokenizer.json")
    if (hf_dir / "chat_template.jinja").read_text(encoding="utf-8") != kv.get("tokenizer.chat_template"):
        raise S.ConvertError(f"the GGUF's chat template differs from {hf_dir.name}/chat_template.jinja")


# ------------------------------------------------------------------ the build
def build(a, Job, DryWriter, fingerprint, source_record, source_tensors, export_tokenizer) -> None:
    from dfly_attach import SECTION as DFLY, dfly_record
    out = Path(a.out)
    partial = Path(str(out) + ".partial")
    dry = getattr(a, "dry_run", False)
    q6k = bool(a.dense_q6k)
    if out.exists() and not a.force:
        raise SystemExit(f"convert: {out} exists (use --force to overwrite)")
    paths = S.expand_shards(Path(a.gguf[0])) if len(a.gguf) == 1 else [Path(p) for p in a.gguf]
    hf_dir = Path(a.hf_dir)
    mmproj = Path(a.mmproj) if a.mmproj and not a.no_vision else None
    dfly = Path(a.dfly) if getattr(a, "dfly", None) else None
    try:
        print("reading the source GGUF ...", flush=True)
        model = S.SourceModel(paths, frozenset())
        kv, _ = source_tensors(model, ARCH.own_sections)
        if kv.get("general.architecture") != ARCH.gguf_architecture:
            raise S.ConvertError(f"{paths[0].name} is architecture {kv.get('general.architecture')!r}, not {ARCH.gguf_architecture}")
        own = {ENGRAM_TENSOR: model.where[ENGRAM_TENSOR][1]} if ENGRAM_TENSOR in model.where else {}
        ARCH.check(kv, own)
        hf_meta = ARCH.hf_meta(hf_dir)
        check_tokenizer(kv, hf_dir)
        plan = Plan(model, q6k)
        dense = plan.dense()
    except (S.ConvertError, KeyError) as e:
        raise SystemExit(f"convert: {e}")
    if mmproj is not None:
        with open(mmproj, "rb") as f:
            head = f.read(1 << 20)
        if head[:4] != b"GGUF" or ARCH.projector_type.encode() not in head:
            raise SystemExit(f"convert: {mmproj} is not a {ARCH.projector_type} projector GGUF")
    for p in (mmproj, dfly):
        if p is not None and not p.is_file():
            raise SystemExit(f"convert: missing input: {p}")
    if partial.exists():
        partial.unlink()
    name = a.name or DEFAULT_NAME + (" Q6K" if q6k else "")
    w = DryWriter() if dry else ModelWriter(str(partial))
    job = Job(w)
    hist = [None]
    ok = False
    pool = None if dry else multiprocessing.Pool(a.workers)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            print("planning sections:" if dry else "writing sections:", flush=True)
            for p in export_tokenizer(model.paths[0], Path(tmp)):
                job.file(f"tokenizer/{p.name}", p)
            job.chunks("dense", dense.chunks())
            job.bytes("dense.index", dense.index_text())
            job.chunks("experts", plan.expert_chunks(pool) if pool else iter(()), plan.expert_bytes)
            if not dry and w.entries[-1][2] != plan.expert_bytes:
                raise S.ConvertError(f"experts: {w.entries[-1][2]} bytes written, {plan.expert_bytes} planned")
            job.bytes("experts.layout", S.nvfp4_layout_text(plan.n_expert, plan.expert_bytes, plan.layout))
            t = time.time()
            parts = plan.parts(pool)
            offs = w.add_parts("gguf", [p for _, _, p in parts], TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            tensors = {}
            for (n, ty, _), off in zip(parts, offs):
                g, t_, _p = model.where[n]
                tensors[n] = {"type": ty, "type_id": TYPE_ID[ty], "shape": [int(d) for d in t_.shape],
                              "bytes": plan.stored_bytes(t_, ty), "offset": off}
            et = own[ENGRAM_TENSOR]
            eng_bytes = int(et.shape[0]) * int(et.shape[1])
            job.chunks(ENGRAM_SECTION, plan.engram_chunks(pool, hist) if pool else iter(()), eng_bytes)
            tensors[ENGRAM_TENSOR] = {"type": "F8_E4M3_RAW", "type_id": F8_E4M3_RAW, "shape": [int(d) for d in et.shape],
                                      "bytes": eng_bytes, "offset": 0, "section": ENGRAM_SECTION}
            engram_cost = engram_error(hist[0]) if hist[0] is not None else None
            if engram_cost:
                if engram_cost["non_finite"]:
                    raise S.ConvertError(f"the Engram table holds {engram_cost['non_finite']} non-finite values")
                print(f"  engram table to E4M3: {engram_cost['exact_fraction'] * 100:.2f} % of the values exact, "
                      f"relative RMS error {engram_cost['rms_relative_error'] * 100:.3f} %, "
                      f"max |value| {engram_cost['max_abs_value']:.3f}, {engram_cost['nonzero_rounded_to_zero']} non-zero values to zero",
                      flush=True)
            vision = None
            if mmproj is not None:
                job.file("vision/mmproj", mmproj)
                vision = {"mmproj": "vision/mmproj", "mmproj_bytes": mmproj.stat().st_size, "extra": []}
            if dfly is not None:
                job.file(DFLY, dfly)
            kv = dict(kv)
            kv["general.architecture"] = ARCH.name
            kv.pop("general.file_type", None)
            sources = [source_record(p) for p in model.paths + [x for x in (mmproj, dfly) if x is not None]]
            sources += [{"path": f"hf/{n}", "size": (hf_dir / n).stat().st_size,
                         "sha256": hashlib.sha256((hf_dir / n).read_bytes()).hexdigest()}
                        for n in ("config.json", "LICENSE", "tokenizer.json", "chat_template.jinja")]
            meta = {
                "format": "basalt", "version": 2, "name": name,
                "converter": {"tool": "tools/convert/convert.py"},
                "sources": sources,
                "gguf_kv": kv,
                "tensors": tensors,
                "conversions": {"compat_bf16": False, "tensors": dense.records},
                "license": ARCH.license(kv),
                "mtp": None,
                **ARCH.meta_extra(kv, {n: t for n, t in tensors.items() if n != ENGRAM_TENSOR}, own),
                "vision": vision,
                "quantized_from": {
                    "source": "BF16 GGUF", "experts": "NVFP4: per matrix tensor scale amax / (6 x 448), per 16 values the "
                    "UE4M3 scale of least squared error among half to four times amax / 6 (as the 27B pack); blobs "
                    "[gate | up | down] and the FP32 tail {s_gate, s_up, s_down, 0}",
                    "dense": "Q6_K (least-squares 16-value group scales) where Q6K_TARGETS names the tensor and rows are whole "
                    "256-value blocks, else Q8_0 (ggml reference rounding)" if q6k else "Q8_0 (ggml reference rounding)",
                    "dense_variant": "Q6_K" if q6k else "Q8_0", "head": "BF16 (the F32 values are BF16 numbers)",
                    "engram": {"format": "E4M3 codes (llama.cpp type 43), to nearest, ties to even, no scale", **(engram_cost or {})},
                },
                **({"dfly2": dfly_record(dfly, hf_meta["hf_config"])} if dfly is not None else {}),
                **hf_meta,
            }
            job.bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        w.close()
        ok = True
    except S.ConvertError as e:
        raise SystemExit(f"convert: {e}")
    finally:
        if pool is not None:
            pool.terminate()
        if not ok and not dry:
            try:
                w.f.close()
            except OSError:
                pass
            partial.unlink(missing_ok=True)
    if dry:
        end = w.pos
        print(f"dry run: {'section':<28} {'offset':>16} {'bytes':>16}")
        for n, off, size, _ in w.entries:
            print(f"         {n:<28} {off:>16} {size:>16}")
        print(f"dry run: {len(w.entries)} sections, {end} bytes ({end / (1 << 30):.3f} GiB) -> {out}")
        return
    os.replace(partial, out)
    print(f"done: {len(job.rows)} sections, {out.stat().st_size / (1 << 30):.3f} GiB, {time.time() - job.t0:.1f} s -> {out}")


# ------------------------------------------------------------------ verify
def compare(section: np.ndarray, at: int, chunks, what: str, bad) -> int:
    """The bytes `chunks` yield against the pack's section from `at`; returns the end offset."""
    for b in chunks:
        n = len(b)
        if at + n > len(section) or not np.array_equal(section[at:at + n], np.frombuffer(b, np.uint8)):
            bad(f"{what}: the bytes at {at} differ from the rounding of the source")
            return -1
        at += n
    return at


def verify(pack: str, first_gguf: str, workers: int = 16) -> bool:
    """Every section's SHA-256, then each section's bytes against the roundings of the source, repeated."""
    t0 = time.time()
    problems: list[str] = []

    def bad(msg):
        problems.append(msg)
        print("  FAIL", msg, flush=True)

    m = ModelFile(pack)
    meta = json.loads(m.read("meta"))
    q = meta["quantized_from"]
    paths = S.expand_shards(Path(first_gguf))
    model = S.SourceModel(paths, frozenset())
    plan = Plan(model, q["dense_variant"] == "Q6_K")
    print(f"{pack}: {len(m.names())} sections, {Path(pack).stat().st_size} bytes; source {len(paths)} GGUFs, {len(model.where)} tensors")
    for n in m.names():
        off, size = m.section(n)
        ok = m.verify(n)
        print(f"  {n:<24} {off:>16} {size:>16}  {'sha256 ok' if ok else 'sha256 MISMATCH'}")
        if not ok or off % SECTION_ALIGN:
            bad(f"section {n}: " + ("checksum differs" if not ok else "not 2 MiB aligned"))
    file_map = np.memmap(pack, dtype=np.uint8, mode="r")

    def view(name):
        off, size = m.section(name)
        return file_map[off:off + size]

    # tensor table: every source tensor once, in its planned type; nothing else
    tens = meta["tensors"]
    want = set(plan.how) | {ENGRAM_TENSOR}
    if set(tens) != want:
        bad(f"meta.tensors lacks {sorted(want - set(tens))[:4]} / has extra {sorted(set(tens) - want)[:4]}")
    for n, (how, ty) in plan.how.items():
        e = tens.get(n)
        t = model.where[n][1]
        if e and (e["type"] != ty or [int(x) for x in e["shape"]] != [int(x) for x in t.shape] or
                  e["bytes"] != plan.stored_bytes(t, ty) or e["offset"] % TENSOR_ALIGN):
            bad(f"{n}: meta.tensors says {e['type']} {e['shape']} {e['bytes']} B at {e['offset']}")
    # the layout and the dense index
    if m.read("experts.layout") != S.nvfp4_layout_text(plan.n_expert, plan.expert_bytes, plan.layout):
        bad("experts.layout differs from the source's expert tensors")
    if m.section("experts")[1] != plan.expert_bytes:
        bad("the experts section has the wrong size")
    dense = plan.dense()
    with multiprocessing.Pool(workers) as pool:
        if compare(view("dense"), 0, dense.chunks(), "dense", bad) != m.section("dense")[1]:
            bad("the dense section has the wrong size")
        if m.read("dense.index") != dense.index_text():
            bad("dense.index differs from the source's tensors")
        print("  dense compared", flush=True)
        compare(view("experts"), 0, plan.expert_chunks(pool), "experts", bad)
        print("  experts compared", flush=True)
        gg = view("gguf")
        parts = plan.parts(pool)
        for n, ty, part in parts:
            e = tens[n]
            at = e["offset"]
            if isinstance(part, tuple):
                chunks = (bytes(pread(part[0], part[1] + o, min(64 << 20, part[2] - o))) for o in range(0, part[2], 64 << 20))
            else:
                chunks = part.chunks
            if compare(gg, at, chunks, n, bad) != at + e["bytes"]:
                bad(f"{n}: wrong length")
        print("  gguf compared", flush=True)
        h = [None]
        compare(view(ENGRAM_SECTION), 0, plan.engram_chunks(pool, h), "engram", bad)
        if engram_error(h[0]) != {k: v for k, v in q["engram"].items() if k in engram_error(h[0])}:
            bad("meta's Engram rounding error differs from a repeat")
        print("  engram compared", flush=True)
    kv = meta.get("gguf_kv", {})
    toks = kv.get("tokenizer.ggml.tokens") or []
    if len(toks) != int(kv.get("hy_v4_flash.vocab_size", len(toks))):
        bad(f"vocabulary of {len(toks)} tokens, vocab_size {kv.get('hy_v4_flash.vocab_size')}")
    if kv.get("general.architecture") != ARCH.name or meta.get("arch") != ARCH.name:
        bad("the pack does not say hy4flash in general.architecture and arch")
    for f in ("tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/token_type.json", "tokenizer/tokenizer.json",
              "tokenizer/chat_template.jinja", "meta"):
        if f not in m.sections:
            bad(f"no {f} section")
    if "tokenizer/chat_template.jinja" in m.sections and \
            m.read("tokenizer/chat_template.jinja").decode("utf-8") != kv.get("tokenizer.chat_template"):
        bad("tokenizer/chat_template.jinja differs from the metadata's template")
    if not meta.get("license", {}).get("text"):
        bad("meta.license has no text")
    for key, sec in (("vision", "vision/mmproj"), ("dfly2", "dfly")):
        if bool(meta.get(key)) != (sec in m.sections):
            bad(f"meta.{key} and the {sec} section disagree")
    print(f"verify: {'OK' if not problems else str(len(problems)) + ' problems'} in {time.time() - t0:.0f} s")
    return not problems
