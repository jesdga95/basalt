"""Do two engine builds produce the same greedy tokens? Does a request's output depend on batching or its slot?

    python3 tools/greedy_identity.py REF_BIN NEW_BIN --model FILE.basalt [--fixtures DIR] [--tokens N] [-- ENGINE ARGS...]
    python3 tools/greedy_identity.py --slots BIN --model FILE.basalt [--fixtures DIR] [--tokens N] [-- ENGINE ARGS...]

Each binary runs as `--serve` with BASE plus any flags after `--` (which replace BASE values). The bench fixtures'
prompts (conv, struct, prose: comma-separated token ids per line) run at temperature 0 and every token is compared.
For qwen4exp packs, adaptive swaps are off so placement cannot drift; pin the main card's tier too (`-- --expert-cache
N`) when the builds' VRAM use differs, since auto-sized tiers change rounding. A second card's tier holds what fits:
check that its expert count ("peer GPU 1: N experts" in both logs) matches.

--slots: one build, one engine with batch slots. Each prompt runs alone as a solo request, then alone in every slot, then
all prompts together with the prompt-to-slot order rotated through every slot; each run is compared token by token with
the solo one. A slot reads its prompt through the batched prompt path, so BASE_SLOTS sets `--short-read 0` and the solo
request reads it that way too (a short read through the decode windows is a different computation, see
docs/precision/qwen38-flash-next.md#prefill).

Prints "GREEDY IDENTICAL" (exit 0) or "GREEDY DIFFERS" with the first difference (exit 1); exit 2 when an engine fails.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model_file import ModelFile  # noqa: E402

PROMPTS = ("conv", "struct", "prose")
# flags after `--` replace these
BASE = ["--spec", "7", "--max-context", "8192"]
PINNED = ["--adapt-swaps", "0", "--peer-adapt-swaps", "0"]
# --slots: drafting and the prompt and conversation caches off, so every run starts from the prompt alone
SLOTS = 4
BASE_SLOTS = ["--batch", str(SLOTS), "--spec", "0", "--max-context", "8192", "--prompt-cache", "0",
              "--conversation-cache-mib", "0", "--short-read", "0"]


def model_arch(model_path: str) -> str | None:
    """Read general.architecture from a .basalt pack's meta section (None when unreadable or absent)."""
    try:
        mf = ModelFile(model_path, versions=(1, 2))
        meta = json.loads(mf.read("meta"))
    except (OSError, ValueError, KeyError, struct.error):
        return None
    return (meta.get("gguf_kv") or {}).get("general.architecture")


def build_args(model_path: str, extra: list[str]) -> list[str]:
    """Engine args for `model_path`: qwen4exp-only adaptive-swap pinning is added only for qwen4exp packs."""
    pinned = PINNED if model_arch(model_path) == "qwen4exp" else []
    return set_args(["--model", model_path, *BASE], pinned + extra)


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
        for pipe in (proc.stdin, proc.stdout):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        err.close()


class Engine:
    """One `--serve` engine: solo requests (GEN) and batch-slot requests (BGEN), read line by line."""

    def __init__(self, label: str, engine_bin: str, args: list[str], logs: Path):
        self.err = open(logs / f"{label}.err", "w")
        self.proc = subprocess.Popen([engine_bin, "--serve", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.err, text=True, bufsize=1)

    def line(self) -> str:
        line = self.proc.stdout.readline()
        if not line:
            raise EOFError("the engine ended")
        if line.startswith("ERR"):
            raise EOFError(line.strip())
        return line

    def ready(self) -> None:
        while not self.line().startswith("READY"):
            pass

    def send(self, text: str) -> None:
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def gen(self, ids: str, tokens: int) -> list[int]:
        self.send(f"GEN {tokens} temperature=0 {ids}")
        got: list[int] = []
        while not (line := self.line()).startswith("DONE"):
            if line.startswith("T "):
                got.append(int(line.split()[1]))
        return got

    def batch(self, placement: dict[int, str], tokens: int) -> dict[int, list[int]]:
        """The prompts of `placement` ({slot: ids}) admitted in that order and decoded together until each is done."""
        got: dict[int, list[int]] = {b: [] for b in placement}
        done: set[int] = set()

        def take(line: str, admitting: int) -> None:
            f = line.split()
            if f[0] == "T":   # the admission's first token, when the slot's read samples it
                got[admitting].append(int(f[1]))
            elif f[0] == "BT":
                got[int(f[1])] += [int(x) for x in f[2:]]
            elif f[0] == "BDONE":
                done.add(int(f[1]))

        for b, ids in placement.items():
            self.send(f"BGEN {b} {tokens} temperature=0 {ids}")
            while not (line := self.line()).startswith("BADM"):
                take(line, b)
            if line.split()[2] == "0":   # finished in its read
                done.add(b)
        while done != set(placement):
            take(self.line(), -1)
        return got

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.send("QUIT")
                self.proc.wait(timeout=120)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()
        self.err.close()


def first_diff(ref: list[int], new: list[int]) -> int | None:
    if ref == new:
        return None
    return next((k for k, (x, y) in enumerate(zip(ref, new)) if x != y), min(len(ref), len(new)))


def slots_check(engine_bin: str, args: list[str], names: tuple[str, ...], prompts: list[str], tokens: int,
                logs: Path) -> int:
    """--slots: every prompt solo, alone in each slot, and all together under each rotation of the slot order."""
    e = Engine("slots", engine_bin, args, logs)
    runs: list[tuple[str, int, list[int]]] = []   # (run, prompt index, ids)
    try:
        e.ready()
        solo = [e.gen(p, tokens) for p in prompts]
        for b in range(SLOTS):
            for i, p in enumerate(prompts):
                runs.append((f"alone in slot {b}", i, e.batch({b: p}, tokens)[b]))
        for r in range(SLOTS):
            place = {(i + r) % SLOTS: i for i in range(len(prompts))}
            got = e.batch({b: prompts[i] for b, i in place.items()}, tokens)
            order = " ".join(f"{names[i]}@{b}" for b, i in place.items())
            for b, i in place.items():
                runs.append((f"together ({order})", i, got[b]))
    except (EOFError, BrokenPipeError, OSError) as x:
        tail = (logs / "slots.err").read_text().strip().splitlines()[-5:]
        print(f"greedy_identity: the engine failed: {x} (logs: {logs})", *tail, sep="\n  ", file=sys.stderr)
        return 2
    finally:
        e.close()
    print(f"solo: {sum(map(len, solo))} tokens; {len(runs)} batch runs")
    bad = 0
    for run_name, i, ids in runs:
        k = first_diff(solo[i], ids)
        if k is not None:
            bad += 1
            print(f"GREEDY DIFFERS: {names[i]} {run_name}, token {k}: solo {solo[i][k:k + 5]} batch {ids[k:k + 5]}")
    if bad:
        print(f"{bad} of {len(runs)} batch runs differ from solo (logs: {logs})")
        return 1
    print("GREEDY IDENTICAL")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    extra: list[str] = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", nargs="?")
    ap.add_argument("new", nargs="?")
    ap.add_argument("--slots", metavar="BIN", help="check one build's batch slots against its solo requests")
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", default=str(HERE.parent / "data" / "bench"))
    ap.add_argument("--tokens", type=int, default=300)
    a = ap.parse_args(argv)
    if not ((a.slots is not None and a.ref is None) or (a.slots is None and a.ref is not None and a.new is not None)):
        ap.error("give REF_BIN NEW_BIN, or --slots BIN")

    prompts = []
    for name in PROMPTS:
        path = Path(a.fixtures) / f"{name}.ids"
        if not path.is_file():
            print(f"greedy_identity: {path} is missing", file=sys.stderr)
            return 2
        prompts.append(path.read_text().strip())
    args = build_args(a.model, extra)
    logs = Path(tempfile.mkdtemp(prefix="greedy-identity-"))
    if a.slots is not None:
        pinned = PINNED if model_arch(a.model) == "qwen4exp" else []
        return slots_check(a.slots, set_args(["--model", a.model, *BASE_SLOTS], pinned + extra), PROMPTS, prompts,
                           a.tokens, logs)
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
