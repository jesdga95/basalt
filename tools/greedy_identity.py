"""Do two engine builds produce the same greedy tokens?

    python3 tools/greedy_identity.py REF_BIN NEW_BIN --model FILE.basalt [--fixtures DIR] [--tokens N] [-- ENGINE ARGS...]

Each binary runs as `--serve` with BASE plus any flags after `--` (which replace BASE values). The bench fixtures'
prompts (conv, struct, prose: comma-separated token ids per line) run at temperature 0 and every token is compared.
Adaptive swaps and the PCIe share are off so placement cannot drift; pin the main card's tier too (`-- --expert-cache
N`) when the builds' VRAM use differs, since auto-sized tiers change rounding. A second card's tier holds what fits:
check that its expert count ("peer GPU 1: N experts" in both logs) matches.

Prints "GREEDY IDENTICAL" (exit 0) or "GREEDY DIFFERS" with the first difference (exit 1); exit 2 when an engine fails.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROMPTS = ("conv", "struct", "prose")
# flags after `--` replace these
BASE = ["--spec", "7", "--max-context", "8192"]
PINNED = ["--adapt-swaps", "0", "--peer-adapt-swaps", "0"]


def set_args(args: list[str], extra: list[str]) -> list[str]:
    """`args` with each flag of `extra` replacing the same flag (and its value) in `args`, or added."""
    out = list(args)
    i = 0
    while i < len(extra):
        flag = extra[i]
        has_value = i + 1 < len(extra) and not extra[i + 1].startswith("--")
        value = [extra[i + 1]] if has_value else []
        if flag in out:
            j = out.index(flag)
            old_has_value = j + 1 < len(out) and not out[j + 1].startswith("--")
            out[j:j + 1 + old_has_value] = [flag, *value]
        else:
            out += [flag, *value]
        i += 1 + has_value
    return out


def run(label: str, engine_bin: str, args: list[str], prompts: list[str], tokens: int, logs: Path) -> list[list[int]] | None:
    """One engine session over the prompts: the generated ids of each, or None when the engine failed."""
    err = open(logs / f"{label}.err", "w")
    proc = subprocess.Popen([engine_bin, "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                            text=True, bufsize=1)
    out: list[list[int]] = []
    try:
        for line in proc.stdout:
            if line.startswith("READY"):
                break
        else:
            return None
        for ids in prompts:
            proc.stdin.write(f"GEN {tokens} temperature=0 {ids}\n")
            proc.stdin.flush()
            got: list[int] = []
            for line in proc.stdout:
                if line.startswith("T "):
                    got.append(int(line.split()[1]))
                elif line.startswith("DONE"):
                    break
                elif line.startswith("ERR"):
                    print(f"{label}: {line.strip()}", file=sys.stderr)
                    return None
            else:
                return None
            out.append(got)
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.wait(timeout=120)
        return out
    except (BrokenPipeError, subprocess.TimeoutExpired):
        return None
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
        err.close()


def main() -> int:
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", default=str(HERE.parent / "data" / "bench"))
    ap.add_argument("--tokens", type=int, default=300)
    a = ap.parse_args(argv)

    prompts = []
    for name in PROMPTS:
        path = Path(a.fixtures) / f"{name}.ids"
        if not path.is_file():
            print(f"greedy_identity: {path} is missing", file=sys.stderr)
            return 2
        prompts.append(path.read_text().strip())
    args = set_args(["--model", a.model, *BASE], PINNED + extra)

    logs = Path(tempfile.mkdtemp(prefix="greedy-identity-"))
    results = {}
    for label, engine_bin in (("ref", a.ref), ("new", a.new)):
        r = run(label, engine_bin, args, prompts, a.tokens, logs)
        if r is None:
            tail = (logs / f"{label}.err").read_text().strip().splitlines()[-5:]
            print(f"greedy_identity: the {label} engine failed (logs: {logs})", *tail, sep="\n  ", file=sys.stderr)
            return 2
        print(f"{label}: {sum(map(len, r))} tokens, {len(r)} requests")
        results[label] = r

    for name, ref, new in zip(PROMPTS, results["ref"], results["new"]):
        if ref != new:
            i = next((k for k, (x, y) in enumerate(zip(ref, new)) if x != y), min(len(ref), len(new)))
            print(f"GREEDY DIFFERS: {name}, token {i}: ref {ref[i:i + 5]} new {new[i:i + 5]} (logs: {logs})")
            return 1
    print("GREEDY IDENTICAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
