#!/usr/bin/env python3
"""Verify fixed-text replay with a conflicting grammar and relaxed drafting request.

    ppl_force_grammar_test.py ENGINE MODEL.basalt BASE.kld [ENGINE ARGS...]

Requires a GPU, a model with an MTP drafter, and a matching KL base. The grammar-only
control must change the first reference token; otherwise this fixture proves nothing.
"""
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
from ppl_gate import read_base


def request(proc, prompt, count, keys):
    proc.stdin.write(f"GEN {count} {keys} {','.join(map(str, prompt))}\n")
    proc.stdin.flush()
    tokens = []
    for line in proc.stdout:
        if line.startswith("T "):
            tokens.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            return tokens
        elif line.startswith("ERR"):
            raise RuntimeError(line.strip())
    raise RuntimeError("engine exited during request")


def main():
    engine, model, base, *extra = sys.argv[1:]
    n_ctx, _, _, chunks = read_base(base)
    first = n_ctx // 2
    prompt = chunks[0][:first + 1]
    expected = list(chunks[0][first + 1:])
    schema = {"type": "string", "enum": ["fixed grammar control"]}
    grammar = "grammar=" + json.dumps({"kind": "json_schema", "schema": schema}, separators=(",", ":")).replace(" ", "\\u0020")
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryFile(mode="w+") as log:
        proc = subprocess.Popen([engine, "--serve", "--model", model, "--grammar", "on", "--spec", "8",
                                 "--ppl-base", base, "--ppl-out", str(pathlib.Path(tmp) / "rows"), *extra],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                if line.startswith("READY"):
                    break
            else:
                raise RuntimeError("engine exited before READY")
            control = request(proc, prompt, 1, f"temperature=0 {grammar}")
            if not control or control[0] == expected[0]:
                raise RuntimeError("grammar control did not conflict with the first reference token; use another base")
            for keys in ("", "relax=0.01", grammar, f"relax=0.01 {grammar}"):
                got = request(proc, prompt, len(expected), f"ppl=0 ppl_force=1 {keys}")
                if got != expected:
                    raise RuntimeError(f"forced replay differs with {keys!r}: {got!r} != {expected!r}")
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=120)
            if proc.returncode:
                raise RuntimeError(f"engine exited with {proc.returncode}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=120)
    print("ppl_force_grammar_test: fixed reference preserved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
