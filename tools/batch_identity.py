"""Do requests decoded together give the tokens they give alone, and how fast are they together?

    python3 tools/batch_identity.py ENGINE --model FILE.basalt [--slots 2,4,8] [--tokens N] [--fixtures DIR]
                                    [--out FILE.json] [-- ENGINE ARGS...]

The serve-level batch check of any model (docs/architecture/batching.md). First each prompt (the bench fixtures conv, struct,
prose) runs in an engine of one slot (`--serve --batch 1`, the same code as `--parallel 1`): the reference. Then for each
slot count N the engine runs as `--serve --batch N`: each of N prompts (the fixtures in turn) runs alone in slot 0, then
all N run together, one per slot, at temperature 0. Every slot's tokens, alone and together, must equal the reference.

Prints per slot count the aggregate tokens per second of the together run (every slot's tokens over the wall time from the
first request to the last one's end), the experts the engine put in RAM, and IDENTICAL or DIFFERS per slot; `--out` keeps
every token, so two builds' runs can be compared. Exit 1 when any slot differs, 2 when the engine fails.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = ("conv", "struct", "prose")


def fixture_ids(fixtures: Path, name: str) -> str:
    line = (fixtures / f"{name}.ids").read_text().splitlines()[0].strip()
    return ",".join(t.strip() for t in line.split(",") if t.strip())


class Engine:
    def __init__(self, cmd: list[str], log: Path):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "w"), text=True, bufsize=1)
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"the engine exited before READY (see {log})")
            if line.startswith("READY"):
                break

    def send(self, line: str) -> None:
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def read(self) -> str:
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("the engine exited")
        return line.rstrip("\n")

    def stop(self) -> None:
        try:
            self.send("QUIT")
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()


def ram_experts(log: Path) -> int:
    """The experts the engine's start put in RAM: its "experts: N of M in RAM" line, or the expert tier's "expert homes of
    N per layer: ... RAM R" line (R, a layer's); none: every one on the card."""
    text = log.read_text(errors="replace")
    m = re.search(r"experts: (\d+) of \d+ in RAM", text)
    if m:
        return int(m.group(1))
    m = re.search(r"expert homes of \d+ per layer: .*?RAM (\d+)", text)
    return int(m.group(1)) if m else 0


def run_slots(eng: Engine, jobs: list[tuple[int, str]], max_new: int) -> tuple[dict[int, list[int]], float]:
    """Starts every (slot, ids) request, one BGEN after the other as each is admitted, and reads until all are done: each
    slot's tokens and the wall time."""
    toks: dict[int, list[int]] = {b: [] for b, _ in jobs}
    live: set[int] = set()
    t0 = time.monotonic()

    def handle(line: str, admitting: int) -> int | None:
        f = line.split()
        if not f:
            return None
        if f[0] == "T" and admitting >= 0:
            toks[admitting].append(int(f[1]))
        elif f[0] == "BT":
            toks[int(f[1])].extend(int(x) for x in f[2:])
        elif f[0] == "BDONE":
            live.discard(int(f[1]))
        elif f[0] == "BADM":
            b = int(f[1])
            if f[2] != "0":
                live.add(b)
            return b
        elif f[0] == "ERR":
            raise RuntimeError(line)
        return None

    for b, ids in jobs:
        eng.send(f"BGEN {b} {max_new} {ids}")
        while handle(eng.read(), b) != b:
            pass
    while live:
        handle(eng.read(), -1)
    return toks, time.monotonic() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("--model", required=True)
    ap.add_argument("--slots", default="2,4,8")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--fixtures", default="")
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="each fixture's first N tokens only (0: all of it): a short prompt takes the windowed read")
    ap.add_argument("--out", default="")
    ap.add_argument("--log-dir", default=".")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:cut])
    extra = argv[cut + 1:]
    fixtures = Path(a.fixtures) if a.fixtures else HERE.parent / "data" / "bench"

    def prompt_ids(name: str) -> str:
        ids = fixture_ids(fixtures, name)
        return ",".join(ids.split(",")[:a.prompt_tokens]) if a.prompt_tokens > 0 else ids
    report: dict = {"slots": {}}
    differs = False
    # the reference: each prompt in an engine of one slot
    log = Path(a.log_dir) / "batch_identity.ref.log"
    try:
        eng = Engine([a.engine, "--model", a.model, "--serve", "--batch", "1", "--prompt-cache", "0"] + extra, log)
        ref = {name: run_slots(eng, [(0, prompt_ids(name))], a.tokens)[0][0] for name in PROMPTS}
        eng.stop()
    except RuntimeError as e:
        print(f"one slot: {e} (see {log})")
        return 2
    report["reference"] = ref
    for n in [int(s) for s in a.slots.split(",")]:
        ids = [prompt_ids(PROMPTS[j % len(PROMPTS)]) for j in range(n)]
        log = Path(a.log_dir) / f"batch_identity.{n}.log"
        cmd = [a.engine, "--model", a.model, "--serve", "--batch", str(max(n, 2)), "--prompt-cache", "0"] + extra
        try:
            eng = Engine(cmd, log)
        except RuntimeError as e:
            print(e)
            return 2
        try:
            alone = [run_slots(eng, [(0, ids[j])], a.tokens)[0][0] for j in range(n)]
            if n == 1:
                together, wall = run_slots(eng, [(0, ids[0])], a.tokens)
            else:
                together, wall = run_slots(eng, list(enumerate(ids)), a.tokens)
        except RuntimeError as e:
            print(f"{n} slots: {e} (see {log})")
            eng.stop()
            return 2
        eng.stop()
        total = sum(len(t) for t in together.values())
        same = [together[b] == alone[b] == ref[PROMPTS[b % len(PROMPTS)]] for b in range(n)]
        differs |= not all(same)
        rate = total / wall if wall > 0 else 0.0
        ram = ram_experts(log)
        print(f"{n} slots: {total} tokens in {wall:.2f} s = {rate:.1f} tok/s together, {ram} experts in RAM; " +
              ", ".join(f"slot {b} {PROMPTS[b % len(PROMPTS)]} {'IDENTICAL' if s else 'DIFFERS'}" for b, s in enumerate(same)))
        report["slots"][str(n)] = {"tok_s": rate, "tokens": total, "wall_s": wall, "ram_experts": ram,
                                   "alone": alone, "together": [together[b] for b in range(n)]}
    if a.out:
        Path(a.out).write_text(json.dumps(report))
    print("BATCH IDENTICAL" if not differs else "BATCH DIFFERS")
    return 1 if differs else 0


if __name__ == "__main__":
    sys.exit(main())
