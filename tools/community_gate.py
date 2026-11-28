#!/usr/bin/env python3
"""The deterministic gate for community benchmark pull requests (stdlib only).

A pull request may be merged without a human only if this gate passes; the model review that follows can only
downgrade a pass. Everything here is a pure function of the run files at the base, the run files the pull request adds
or modifies, the PR author and the list of changed files, so the same code runs in CI and in the tests. The head files are
data: nothing in them is executed.

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

RUNS_PREFIX = "docs/community/runs/"
LEGACY_FILE = "docs/community/benchmarks.json"
PATH_RE = re.compile(re.escape(RUNS_PREFIX) + r"[^/]+/[0-9a-f]{%d}\.json" % cb.ID_LEN)
MAX_NEW_ROWS = 12
PRESET = "community-1"
# The engine arguments the community preset passes (bench_main.cpp; the cards, their mode and shares are the submitter's): anything else is a lossy or off-preset run.
ENGINE_FLAGS = {"--gpu": r"\d+(,\d+)*", "--gpu-mode": "peer|split|tensor", "--gpu-share": r"\d+(\.\d+)?(,\d+(\.\d+)?)*",
                "--max-context": r"\d{4,7}", "--kv": "|".join(re.escape(k) for k in cb.KV_FORMATS)}
# Band: a new row's decode_tps and prefill_16k_tps must lie in [lowest/BAND, highest*BAND] of the earlier rows of the
# same model, GPU name and GPU count (different quants of one card spread widely).
BAND = 2.5
# Hard ceilings, whatever the card: above the best rows of the table.
CAP_DECODE, CAP_PREFILL = 2000, 40000
# prefill_64k_tps over prefill_16k_tps, bounded around the spread of the table.
RATIO_LOW, RATIO_HIGH = 0.3, 6.0
CAPS = {"submitter": 40, "notes": 200, "cpu": 80, "engine_args": 300, "quant": 40, "preset": 30, "gpu_name": 80}
TEXT_RE = re.compile(r"^[^<>`\\]*$")
URL_RE = re.compile(
    r"(?i)\b(https?|ftp|file|data|javascript|ssh|git)\s*:|//|www\.|\bt\.me\b|@"
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
            gate.fail(f"{label}: engine_args has '{toks[i]}' outside the community preset's --gpu, --gpu-mode, --gpu-share, --max-context and --kv (a lossy or off-preset run)")
            return
        i += 2


def plausible(row, earlier, label, gate):
    for key in ("decode_tps", "prefill_16k_tps", "prefill_64k_tps"):
        v = row.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not cb.is_pos(v):
            gate.fail(f"{label}: {key} {v} must be a finite number > 0")
        elif cb.is_pos(v) and v > (CAP_DECODE if key == "decode_tps" else CAP_PREFILL):
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


def parse_run(text):
    """The row of a run file's text, or None when it is not a JSON object."""
    try:
        row = cb.loads_strict(text)
    except ValueError:
        return None
    return row if isinstance(row, dict) else None


def run_gate(base_files, head_files, author, changed, known_commit):
    """base_files: {path: text} of every run file at the commit the branch started from; head_files: {path: text} of the
    added and modified run files in the pull request; author: the PR author's login; changed: [(filename, status)];
    known_commit(sha) -> bool: the commit is a build of this repository."""
    g = Gate()
    if not changed:
        g.fail("the pull request changes no file")
        return g
    for name, status in changed:
        if name == LEGACY_FILE:
            g.fail(f"{LEGACY_FILE} is no longer used: add your run as its own file under {RUNS_PREFIX}")
        elif not PATH_RE.fullmatch(name):
            g.fail(f"{name} is not a run file: a community pull request may only add or modify {RUNS_PREFIX}MODEL/ID.json files")
        elif status not in ("added", "modified", "removed"):
            g.fail(f"{name} is {status}; a run file can only be added, modified or removed")
    if g.status == "fail":
        return g
    added = [n for n, st in changed if st == "added"]
    modified = [n for n, st in changed if st == "modified"]
    removed = [n for n, st in changed if st == "removed"]
    if len(added) + len(modified) > MAX_NEW_ROWS:
        g.fail(f"{len(added) + len(modified)} run files added or modified; at most {MAX_NEW_ROWS} per pull request")
        return g
    base_rows = {}
    for path, text in base_files.items():
        row = parse_run(text)
        if row is not None:
            base_rows[path] = row
    # every changed file is checked as a file: it parses, is valid, in the house format and named by its content
    head_rows = {}
    for path in added + modified:
        label = f"{'new' if path in added else 'modified'} file {path.rsplit('/', 1)[-1]}"
        text = head_files.get(path)
        if text is None:
            g.fail(f"{label}: the file could not be read")
            continue
        row = parse_run(text)
        if row is None:
            g.fail(f"{label}: not a JSON object")
            continue
        problems = []
        cb.check_row(row, problems.append)
        for p in problems:
            g.fail(f"{label}: {p}")
        if problems:
            continue
        if RUNS_PREFIX + cb.run_path(row) != path:
            g.fail(f"{label}: the file belongs at {RUNS_PREFIX}{cb.run_path(row)} (the model's folder and the id of its setup)")
        if cb.canonical_row(row) != text:
            g.fail(f"{label}: not in the house format; run: {cb.FIX_HINT}")
        head_rows[path] = row
    if g.status == "fail":
        return g
    # one run per setup: no other file at the base (not removed, not changed here) or in this pull request has this setup
    untouched = {p: r for p, r in base_rows.items() if p not in removed and p not in head_rows}
    for path, row in head_rows.items():
        clash = next((p for p, r in {**untouched, **{q: x for q, x in head_rows.items() if q != path}}.items() if cb.same_setup(r, row)), None)
        if clash is not None:
            g.fail(f"{path}: same submitter, model, quant and hardware as {clash}: one run per setup, update that file instead")
    if g.status == "fail":
        return g
    for path in removed:
        old = base_rows.get(path, {})
        human(g, f"{path} ({old.get('submitter')}, {old.get('model')}) was deleted; only a person can approve a deletion")
    g.new_rows = [head_rows[p] for p in added + modified]
    # Judge every PR row only against trusted base rows (never against other untrusted rows in the same PR).
    earlier = [r for p, r in base_rows.items() if p not in removed]
    for i, path in enumerate(added + modified):
        r = head_rows[path]
        is_update = path in modified
        label = f"{'updated' if is_update else 'new'} row {i + 1} ({r['submitter']})"
        check_text(r, label, g)
        check_engine_args(r, label, g)
        if not known_commit(r["commit"]):
            g.fail(f"{label}: commit {r['commit']} is not a build of this repository's main branch")
        before = base_rows.get(path) if is_update else None
        if is_update:
            g.notes.append(f"{label}: modifies the run file of the same submitter, model, quant and hardware")
            owners = {r["submitter"].lower(), str((before or {}).get("submitter")).lower()}
            if before is None:
                human(g, f"{label}: the earlier file could not be read; only a person can approve this change")
            elif owners != {author.lower()}:
                human(g, f"{label}: modifies the run of '{before.get('submitter')}', which is not the pull request author "
                         f"'{author}'; only a person can approve changing someone else's numbers")
        plausible(r, earlier, label, g)
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


def read_tree(base, rel):
    """{repo path: text} of every file under base/<rel>; nothing when it does not exist."""
    out = {}
    top = Path(base) / rel
    for f in sorted(top.rglob("*.json")) if top.is_dir() else []:
        out[f.relative_to(base).as_posix()] = f.read_text(encoding="utf-8")
    return out


def main(argv):
    """community_gate.py BASE_DIR HEAD_DIR AUTHOR CHANGED.json OUT.json

    BASE_DIR holds docs/community/runs as it was at the commit the branch started from, HEAD_DIR the pull request's added
    and modified run files at the same paths; CHANGED: [[filename, status], ...]."""
    base, head, author, changed, out = argv[1:6]
    changed = [tuple(c) for c in json.loads(Path(changed).read_text())]
    head_files = {}
    for name, status in changed:
        f = Path(head) / name
        if status in ("added", "modified") and PATH_RE.fullmatch(name) and f.is_file():
            head_files[name] = f.read_text(encoding="utf-8")
    g = run_gate(read_tree(base, RUNS_PREFIX.rstrip("/")), head_files, author, changed, git_known_commit())
    Path(out).write_text(json.dumps(g.to_json(), allow_nan=False))
    print(g.status)
    for r in g.reasons:
        print(" -", r)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
