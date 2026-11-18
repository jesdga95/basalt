"""Read a .basalt file built from a GGUF back and check it against the GGUF (tools/convert/convert.py --verify).

  * every section's SHA-256 (the header's) against its bytes;
  * every source tensor found exactly once, where the pack says: the dense index (`dense`, `mtp/dense`), the expert
    layouts (`experts`, `mtp/experts`: each expert's slice of the role), or meta.tensors (`gguf` and the own sections);
  * its type, shape and size as the pack records them, and its bytes by SHA-256 against the source's;
  * nothing in the pack's indexes that the source lacks; the section and tensor alignments;
  * the tokenizer and metadata the engine reads (vocabulary size, chat template, license, tensor table).
Writes nothing; returns True when everything agrees. Needs numpy.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import model_sections as S  # noqa: E402
from model_file import SECTION_ALIGN, TENSOR_ALIGN, ModelFile  # noqa: E402

CHUNK = 64 << 20
ROLES = ("gate", "up", "down")
EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def sha(view: np.ndarray) -> str:
    h = hashlib.sha256()
    for i in range(0, view.size, CHUNK):
        h.update(view[i:i + CHUNK])
    return h.hexdigest()


def parse_layout(text: bytes):
    """-> {layer: (gate_type, down_type, offset, blob)}, n_expert, total"""
    lines = text.decode("utf-8").splitlines()
    m = re.search(r"\(n_expert (\d+), total (\d+)\)", lines[0])
    layers = {}
    for line in lines[1:]:
        if line.strip() and not line.startswith("#"):
            f = [int(x) for x in line.split()]
            layers[f[0]] = (f[1], f[2], f[3], f[4])
    return layers, int(m.group(1)), int(m.group(2))


def parse_index(text: bytes) -> dict:
    """dense.index -> {name: row} for the tensors the pool holds (the others are served from `gguf`)."""
    out = {}
    for line in text.decode("utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        r = line.split()
        out[r[0]] = r
    return out


def verify(pack: str, first_gguf: str, workers: int = 16) -> bool:
    if "quantized_from" in json.loads(ModelFile(pack).read("meta")):
        import hy4flash_bf16
        return hy4flash_bf16.verify(pack, first_gguf, workers)
    t0 = time.time()
    problems: list[str] = []

    def bad(msg):
        problems.append(msg)
        print("  FAIL", msg, flush=True)

    m = ModelFile(pack)
    meta = json.loads(m.read("meta"))
    paths = S.expand_shards(Path(first_gguf))
    src = S.SourceModel(paths, frozenset())
    file_map = np.memmap(pack, dtype=np.uint8, mode="r")
    print(f"{pack}: {len(m.names())} sections, {Path(pack).stat().st_size} bytes; source {len(paths)} GGUFs, {len(src.where)} tensors")

    # sections: alignment and checksums
    print("sections:")
    with ThreadPoolExecutor(8) as ex:
        sums = dict(zip(m.names(), ex.map(m.verify, m.names())))
    for name in m.names():
        off, size = m.section(name)
        ok = sums[name] and off % SECTION_ALIGN == 0
        print(f"  {name:<24} {off:>16} {size:>16}  {'sha256 ok' if sums[name] else 'sha256 MISMATCH'}")
        if not ok:
            bad(f"section {name}: " + ("checksum differs" if not sums[name] else "not 2 MiB aligned"))

    def section_view(name):
        off, size = m.section(name)
        return file_map[off:off + size]

    # where each source tensor is: name -> (kind, args)
    # (the `gguf` section holds every tensor outside the experts and the own sections, so a float tensor is there and
    # in the dense pool; any other second place is an error)
    where: dict[str, list] = {}

    def place(name, loc):
        if name not in src.where:
            bad(f"{loc[0]} holds {name}, which the source does not have")
            return
        have = where.setdefault(name, [])
        if have and {have[0][0], loc[0]} not in ({"dense", "gguf"}, {"mtp/dense", "gguf"}):
            bad(f"{name} is stored twice ({have[0][0]} and {loc[0]})")
        else:
            have.append(loc)

    for pre in ("", "mtp/"):
        if pre + "experts.layout" in m.sections:
            layers, n_exp, total = parse_layout(m.read(pre + "experts.layout"))
            if total != m.section(pre + "experts")[1]:
                bad(f"{pre}experts.layout totals {total} bytes, the section is {m.section(pre + 'experts')[1]}")
            for l, (gt, dt, off, blob) in layers.items():
                names = [f"blk.{l}.ffn_{r}_exps.weight" for r in ROLES]
                if any(n not in src.where for n in names):
                    bad(f"{pre}experts layer {l}: the source lacks {names}")
                    continue
                ts = [src.where[n][1] for n in names]
                per = [t.expected_bytes() // n_exp for t in ts]
                if sum(per) != blob or (ts[0].type_id, ts[2].type_id) != (gt, dt) or off % 1:
                    bad(f"{pre}experts layer {l}: layout {gt}/{dt}/{blob} does not match the source tensors")
                role_off = [0, per[0], per[0] + per[1]]
                for n, ro, p in zip(names, role_off, per):
                    place(n, (pre + "experts", off, blob, n_exp, ro, p))
            n_in = sum(1 for n in src.where if EXPERT.match(n) and int(EXPERT.match(n).group(1)) in
                       {l for l in layers})
            if n_in != 3 * len(layers):
                bad(f"{pre}experts: {len(layers)} layers listed, {n_in} source expert tensors of those layers")
    for pre in ("", "mtp/"):
        if pre + "dense.index" in m.sections:
            for name, r in parse_index(m.read(pre + "dense.index")).items():
                if r[2] == "0" and r[4] == "0":
                    continue           # served from `gguf`; meta.tensors places it
                place(name, (pre + "dense", int(r[5]), int(r[4]), r[2]))
    for name, e in meta["tensors"].items():
        place(name, (e.get("section", "gguf"), e["offset"], e["bytes"], e))
    missing = [n for n in src.where if n not in where]
    if missing:
        bad(f"{len(missing)} source tensors are nowhere in the pack: {missing[:6]}")

    # type, shape, size against the source
    kind = {v: k for k, v in S.KIND.items()}
    for name, loc in ((n, l) for n, ls in where.items() for l in ls):
        g, t, p = src.where[name]
        nb = t.expected_bytes()
        if loc[0] in ("dense", "mtp/dense"):
            _, off, ln, k = loc
            if ln != nb or kind.get(k) != t.type_name:
                bad(f"{name}: the dense index has kind {k} of {ln} bytes, the source {t.type_name} of {nb}")
        elif loc[0] in ("experts", "mtp/experts"):
            pass
        else:
            e = loc[3]
            if e["bytes"] != nb or e["type_id"] != t.type_id or [int(x) for x in e["shape"]] != [int(x) for x in t.shape]:
                bad(f"{name}: meta.tensors says {e['type']} {e['shape']} {e['bytes']} bytes, the source {t.type_name} {list(t.shape)} {nb}")
            if loc[0] == "gguf" and loc[1] % TENSOR_ALIGN:
                bad(f"{name}: offset {loc[1]} in gguf is not 4096 aligned")

    # bytes: one task per tensor
    def pack_bytes_hash(name, loc):
        if loc[0] in ("experts", "mtp/experts"):
            _, off, blob, n_exp, ro, per = loc
            sec = section_view(loc[0])
            h = hashlib.sha256()
            for e in range(n_exp):
                a = off + e * blob + ro
                h.update(sec[a:a + per])
            return h.hexdigest()
        sec = section_view(loc[0])
        off, n = loc[1], loc[2]
        return sha(sec[off:off + n])

    def one(name):
        a = sha(src.bytes(name))
        return name, all(a == pack_bytes_hash(name, loc) for loc in where[name])

    todo = [n for n in where if n in src.where]
    done = 0
    mism = []
    with ThreadPoolExecutor(12) as ex:
        for name, ok in ex.map(one, todo):
            done += 1
            if not ok:
                mism.append(name)
                bad(f"{name}: bytes differ from the source")
            if done % 200 == 0:
                print(f"  {done}/{len(todo)} tensors compared", flush=True)
    print(f"tensors: {len(todo)} compared, {len(mism)} differ")

    # summary by role
    role = defaultdict(lambda: [0, 0])
    for name, locs in where.items():
        t = src.where[name][1]
        for loc in locs:
            key = loc[0] + (" (bf16/f32 pool)" if loc[0].endswith("dense") else "")
            if loc[0] == "gguf":
                key = "gguf (" + t.type_name + ")"
            role[key][0] += 1
            role[key][1] += t.expected_bytes()
    print("by section:")
    for k, (n, b) in sorted(role.items()):
        print(f"  {k:<28} {n:>5} tensors {b / (1 << 30):>10.3f} GiB")
    print(f"  source: {len(src.where)} tensors, {sum(t.expected_bytes() for _, t, _ in src.where.values()) / (1 << 30):.3f} GiB;"
          f" located {len(where)} (float tensors are in the dense pool and in gguf)")

    # what the engine reads from the metadata and the tokenizer
    kv = meta.get("gguf_kv", {})
    toks = kv.get("tokenizer.ggml.tokens") or []
    if len(toks) != int(kv.get("hy_v4_flash.vocab_size", len(toks))):
        bad(f"vocabulary of {len(toks)} tokens, vocab_size {kv.get('hy_v4_flash.vocab_size')}")
    for f in ("tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/token_type.json", "tokenizer/tokenizer.json",
              "tokenizer/chat_template.jinja", "meta"):
        if f not in m.sections:
            bad(f"no {f} section")
    if "tokenizer/vocab.json" in m.sections:
        vocab = json.loads(m.read("tokenizer/vocab.json"))
        if len(vocab) != len(toks):
            bad(f"tokenizer/vocab.json has {len(vocab)} entries, the GGUF {len(toks)}")
    if "tokenizer/chat_template.jinja" in m.sections and \
            m.read("tokenizer/chat_template.jinja").decode("utf-8") != kv.get("tokenizer.chat_template"):
        bad("tokenizer/chat_template.jinja differs from the GGUF's template")
    if not meta.get("license", {}).get("text"):
        bad("meta.license has no text")
    eng = meta.get("engram")
    if eng and eng["bytes"] != m.section(eng["section"])[1]:
        bad(f"meta.engram says {eng['bytes']} bytes, section {eng['section']} is {m.section(eng['section'])[1]}")

    print(f"verify: {'OK' if not problems else str(len(problems)) + ' problems'} in {time.time() - t0:.0f} s")
    return not problems
