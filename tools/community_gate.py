#!/usr/bin/env python3
"""The deterministic gate for community benchmark pull requests (stdlib only).

A pull request may be merged without a human only if this gate passes; the model review that follows can only
downgrade a pass. Everything here is a pure function of the base file, the head file, the PR author and the list of
changed files, so the same code runs in CI and in the tests. The head file is data: nothing in it is executed.

Status: "pass" (every check holds and the row has comparable rows to be judged against), "fail" (a check failed, the
reasons say which), "human" (every check holds but nothing comparable exists, e.g. a card with no earlier row, so only
a person can judge the numbers).
"""
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_benchmarks as cb  # noqa: E402

FILE = "docs/community/benchmarks.json"
MAX_NEW_ROWS = 12
PRESET = "community-1"
# The engine arguments the community preset passes (bench_main.cpp): anything else is a lossy or off-preset run.
ENGINE_FLAGS = {"--gpu": r"\d+(,\d+)*", "--max-context": r"\d{4,7}", "--kv": "|".join(re.escape(k) for k in cb.KV_FORMATS)}
# Band: a new row's decode_tps and prefill_16k_tps must lie in [lowest/BAND, highest*BAND] of the earlier rows of the
# same model, GPU name and GPU count (different quants of one card spread about 4.5x in the table).
BAND = 2.5
# Hard ceilings, whatever the card: the table's best rows are 656 tok/s decode and 18,710 tok/s prefill.
CAP_DECODE, CAP_PREFILL = 2000, 40000
# prefill_64k_tps over prefill_16k_tps: the table spans 0.70 to 4.06.
RATIO_LOW, RATIO_HIGH = 0.3, 6.0
CAPS = {"submitter": 40, "notes": 200, "cpu": 80, "engine_args": 300, "quant": 40, "preset": 30, "gpu_name": 80}
TEXT_RE = re.compile(r"^[^<>`\\]*$")
URL_RE = re.compile(
    r"(?i)(https?|ftp|file|data|javascript|ssh|git)\s*:|//|www\.|\bt\.me\b|@"
    r"|\b[a-z0-9-]+\.(com|net|org|io|dev|ai|app|xyz|co|me|gg|ly|sh|to|cc|info|ru|cn|link|site|top|tk|ml|ga|cf|gq|zip|mov)\b"
    r"|\]\(")
BIDI_RE = re.compile(r"[​-‏‪-‮⁠-⁩﻿]")
QUANT_RE = re.compile(r"[A-Za-z0-9._+-]+")


@dataclass
class Gate:
    status: str = "pass"
    reasons: list = field(default_factory=list)   # why it failed (status fail) or why a person must look (human)
    notes: list = field(default_factory=list)     # facts that held, given to the model review
    new_rows: list = field(default_factory=list)

    def fail(self, msg):
        self.status = "fail"
        self.reasons.append(msg)

    def to_json(self):
        return {"status": self.status, "reasons": self.reasons, "notes": self.notes, "new_rows": self.new_rows}


def free_texts(row):
    """Every string of a row, as (name, text)."""
    out = [(k, v) for k, v in row.items() if isinstance(v, str)]
    out += [(f"gpus[{i}].{k}", v) for i, g in enumerate(row.get("gpus") or []) if isinstance(g, dict)
            for k, v in g.items() if isinstance(v, str)]
    return out


def check_text(row, label, gate):
    for name, text in free_texts(row):
        base = name.split(".")[-1]
        cap = CAPS.get("gpu_name" if base == "name" else base)
        if cap and len(text) > cap:
            gate.fail(f"{label}: {name} is longer than {cap} characters")
        if URL_RE.search(text):
            gate.fail(f"{label}: {name} contains a link, address or domain")
        if not TEXT_RE.match(text):
            gate.fail(f"{label}: {name} contains markup characters (< > ` or backslash)")
        if BIDI_RE.search(text):
            gate.fail(f"{label}: {name} contains invisible or direction-changing characters")
    if re.search(r"(?i)lossy", " ".join(t for _, t in free_texts(row))):
        gate.fail(f"{label}: the row mentions LOSSY; lossy runs are not comparable and are not accepted")
    if isinstance(row.get("quant"), str) and not QUANT_RE.fullmatch(row["quant"]):
        gate.fail(f"{label}: quant has characters outside letters, digits and ._+-")
    if row.get("preset") != PRESET:
        gate.fail(f"{label}: preset must be {PRESET}")


def check_engine_args(row, label, gate):
    args = row.get("engine_args")
    if args is None:
        return
    toks = args.split()
    i = 0
    while i < len(toks):
        pat = ENGINE_FLAGS.get(toks[i])
        if pat is None or i + 1 >= len(toks) or not re.fullmatch(pat, toks[i + 1]):
            gate.fail(f"{label}: engine_args has '{toks[i]}' outside the community preset's --gpu, --max-context and --kv (a lossy or off-preset run)")
            return
        i += 2


def plausible(row, earlier, label, gate):
    for key in ("decode_tps", "prefill_16k_tps", "prefill_64k_tps"):
        v = row.get(key)
        if cb.is_pos(v) and v > (CAP_DECODE if key == "decode_tps" else CAP_PREFILL):
            gate.fail(f"{label}: {key} {v} is above the ceiling of any card")
    a, b = row.get("prefill_16k_tps"), row.get("prefill_64k_tps")
    if cb.is_pos(a) and cb.is_pos(b) and not RATIO_LOW <= b / a <= RATIO_HIGH:
        gate.fail(f"{label}: prefill_64k_tps / prefill_16k_tps is {b / a:.2f}, outside {RATIO_LOW} to {RATIO_HIGH}")
    gpus = row.get("gpus")
    if not isinstance(gpus, list) or not gpus or not isinstance(gpus[0], dict):
        return
    def key_of(r):
        g = r.get("gpus")
        return (r.get("model"), g[0].get("name"), len(g)) if isinstance(g, list) and g and isinstance(g[0], dict) else None
    mine = key_of(row)
    peers = [r for r in earlier if key_of(r) == mine]
    if not peers:
        gate.reasons.append(f"{label}: no earlier row of {mine[0]} on {'x'.join([str(mine[2]), mine[1]])}; only a person can judge these numbers")
        if gate.status == "pass":
            gate.status = "human"
        return
    for key in ("decode_tps", "prefill_16k_tps"):
        xs = [r[key] for r in peers if cb.is_pos(r.get(key))]
        v = row.get(key)
        if xs and cb.is_pos(v):
            lo, hi = min(xs) / BAND, max(xs) * BAND
            if not lo <= v <= hi:
                gate.fail(f"{label}: {key} {v} is outside {lo:.1f} to {hi:.1f}, the band around {len(xs)} earlier "
                          f"row(s) of {mine[0]} on {mine[2]}x {mine[1]} ({min(xs)} to {max(xs)})")
    gate.notes.append(f"{label}: compared with {len(peers)} earlier row(s) of the same model and card")


def run_gate(base_text, head_text, author, changed, known_commit):
    """base_text/head_text: the file before and after; author: the PR author's login; changed: [(filename, status)];
    known_commit(sha) -> bool: the commit is a build of this repository."""
    g = Gate()
    if [c[0] for c in changed] != [FILE]:
        g.fail(f"the pull request must change only {FILE} (changes: {', '.join(c[0] for c in changed) or 'none'})")
        return g
    if changed[0][1] != "modified":
        g.fail(f"{FILE} must be modified, not {changed[0][1]}")
        return g
    try:
        base, head = json.loads(base_text), json.loads(head_text)
    except ValueError as e:
        g.fail(f"the file is not valid JSON: {e}")
        return g
    for p in cb.validate(head):
        g.fail(p)
    try:
        if cb.canonical(head) != head_text:
            g.fail(f"not in the canonical format; run: {cb.FIX_HINT}")
    except Exception:
        g.fail("the file is not in the canonical format")
    if g.status == "fail":
        return g
    old, new = base["runs"], head["runs"]
    if head.get("preset") != base.get("preset"):
        g.fail("the top-level preset changed")
    # Rows are matched by setup (submitter, model, quant and hardware), not by position: a row whose setup is in the
    # base file is an update of that run (the validator asks for exactly that: "update your run instead").
    def find(rows, r):
        return next((x for x in rows if cb.same_setup(x, r)), None)
    added, updated, removed = [], [], []
    for r in new:
        before = find(old, r)
        if before is None:
            added.append(r)
        elif before != r:
            updated.append((before, r))
    for r in old:
        if find(new, r) is None:
            removed.append(r)
    for r in removed:
        human(g, f"the row of {r.get('submitter')} ({r.get('model')}) was removed; only a person can approve a removal")
    changed_rows = added + [r for _, r in updated]
    if not changed_rows and not removed:
        g.fail("no row was added or updated")
    if len(changed_rows) > MAX_NEW_ROWS:
        g.fail(f"{len(changed_rows)} rows added or updated; at most {MAX_NEW_ROWS} per pull request")
    if g.status == "fail":
        return g
    g.new_rows = changed_rows
    others = [r for r in old if find(changed_rows, r) is None]
    for i, r in enumerate(changed_rows):
        is_update = i >= len(added)
        label = f"{'updated' if is_update else 'new'} row {i + 1} ({r['submitter']})"
        check_text(r, label, g)
        check_engine_args(r, label, g)
        if not known_commit(r["commit"]):
            g.fail(f"{label}: commit {r['commit']} is not a build of this repository's main branch")
        if is_update:
            g.notes.append(f"{label}: updates the existing run of the same submitter, model, quant and hardware")
            if r["submitter"].lower() != author.lower():
                human(g, f"{label}: updates the run of '{r['submitter']}', which is not the pull request author "
                         f"'{author}'; only a person can approve changing someone else's numbers")
        before = [b for b, x in updated if x is r]
        plausible(r, others + before + changed_rows[:i], label, g)
    return g


def human(gate, msg):
    """A reason only a person can judge; a failure stays a failure."""
    gate.reasons.append(msg)
    if gate.status == "pass":
        gate.status = "human"


def git_known_commit(repo="."):
    """True for a commit reachable from origin/main of a full checkout. Fails closed."""
    def known(sha):
        try:
            return subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", sha + "^{commit}", "origin/main"],
                                  capture_output=True, timeout=30).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
    return known


def main(argv):
    """community_gate.py BASE.json HEAD.json AUTHOR CHANGED.json OUT.json  (CHANGED: [[filename, status], ...])"""
    base, head, author, changed, out = argv[1:6]
    g = run_gate(Path(base).read_text(encoding="utf-8"), Path(head).read_text(encoding="utf-8"), author,
                 [tuple(c) for c in json.loads(Path(changed).read_text())], git_known_commit())
    Path(out).write_text(json.dumps(g.to_json()))
    print(g.status)
    for r in g.reasons:
        print(" -", r)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
