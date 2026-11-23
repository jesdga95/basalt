"""Does a model on several cards give the greedy tokens it gives on one card (or with both stages on one GPU)?

    python3 tools/gpu_identity.py ENGINE --model FILE.basalt --mode-flags "--gpu 0,1 --gpu-mode peer"
        [--ref-flags "--gpu 0"] [--ref-engine BIN] [--prompts conv,struct,prose,prefill] [--tokens N] [--fixtures DIR]
        [-- COMMON ENGINE ARGS...]   (Flash-Next: -- --adapt-swaps 0 --peer-adapt-swaps 0)

The serve-level multi-GPU check of any model (docs/design/multi-gpu.md), the sibling of batch_identity.py. The reference
engine (ENGINE, or --ref-engine for another build: v0.5.6 against the branch) and the mode's engine each run as `--serve`
with the common args; each prompt (the bench fixtures conv, struct, prose and the long prefill, which exceeds a chunk so a
prompt stage or a split's hand-off runs) is greedy at temperature 0 and every token is compared. A split too large for one
card takes `--ref-flags "--gpu 0,1 --gpu-mode split --diag-gpu-cut L --diag-gpu-same"` (Qwen3.8-Flash-Next).

For the logits themselves (bitwise, KLD and top-1) run the engine's own `--diag-gpu-check --diag-gpu-trace F` on the reference
and `--diag-gpu-ref F` on the mode: the same prompt through `--tokens-file`.

Prints IDENTICAL or DIFFERS per prompt (exit 1 when any differs, 2 when an engine fails); `--out` keeps every token.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from greedy_identity import run  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode-flags", required=True)
    ap.add_argument("--ref-flags", default="--gpu 0")
    ap.add_argument("--ref-engine", default="")
    ap.add_argument("--prompts", default="conv,struct,prose,prefill")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--fixtures", default=str(HERE.parent / "data" / "bench"))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    names = a.prompts.split(",")
    prompts = []
    for name in names:
        path = Path(a.fixtures) / f"{name}.ids"
        if not path.is_file():
            print(f"gpu_identity: {path} is missing", file=sys.stderr)
            return 2
        prompts.append(",".join(path.read_text().split()))   # prefill.ids: its lines (before, body, after) are one prompt
    common = ["--model", a.model, "--max-context", "32768", *extra]
    logs = Path(tempfile.mkdtemp(prefix="gpu-identity-"))
    results = {}
    for label, engine_bin, flags in (("ref", a.ref_engine or a.engine, a.ref_flags), ("mode", a.engine, a.mode_flags)):
        r = run(label, engine_bin, common + shlex.split(flags), prompts, a.tokens, logs)
        if r is None:
            tail = (logs / f"{label}.err").read_text().strip().splitlines()[-5:]
            print(f"gpu_identity: the {label} engine failed (logs: {logs})", *tail, sep="\n  ", file=sys.stderr)
            return 2
        print(f"{label}: {sum(map(len, r))} tokens, {len(r)} requests")
        results[label] = r
    differs = False
    for name, ref, got in zip(names, results["ref"], results["mode"]):
        if ref == got:
            print(f"{name}: IDENTICAL ({len(got)} tokens)")
            continue
        differs = True
        i = next((k for k, (x, y) in enumerate(zip(ref, got)) if x != y), min(len(ref), len(got)))
        print(f"{name}: DIFFERS at token {i}: ref {ref[i:i + 5]} mode {got[i:i + 5]}")
    if a.out:
        Path(a.out).write_text(json.dumps({"prompts": names, **results}))
    print("GPU DIFFERS" if differs else "GPU IDENTICAL" + f" (logs: {logs})")
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main())
