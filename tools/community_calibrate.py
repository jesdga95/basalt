#!/usr/bin/env python3
"""Run the community gate (and, with a key, the model review) over past community pull requests (stdlib + gh).

    community_calibrate.py [--limit N] [--state all|merged|closed|open] [--review]

Prints one row per pull request: the gate status and reasons and, with --review and LLM_API_KEY, LLM_BASE_URL and
LLM_DEFAULT_MODEL set, the model's verdict for the ones whose gate passed. Needs `gh` logged in and a full clone with
origin/main fetched (for the commit check). Nothing is changed on GitHub.
"""
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import community_gate as gate  # noqa: E402
import llm_review as llm  # noqa: E402

REPO = "jesdga95/basalt"
STATUS_MAP = {"ADDED": "added", "MODIFIED": "modified", "DELETED": "removed"}


def gh(*args):
    p = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=120)
    if p.returncode:
        raise RuntimeError(p.stderr.strip()[:200])
    return p.stdout


def file_at(path, sha):
    encoded = urllib.parse.quote(path, safe="/")
    return gh("api", f"repos/{REPO}/contents/{encoded}?ref={sha}", "-H", "Accept: application/vnd.github.raw+json")


def is_community_pr(files):
    return any(f["path"].startswith(gate.RUNS_PREFIX) or f["path"] == gate.LEGACY_FILE for f in files)


def normalize_changed(files):
    return [(f["path"], STATUS_MAP.get(f["changeType"], f["changeType"].lower())) for f in files]


def base_tree(sha, repo="."):
    """{repo path: text} of every run file under docs/community/runs at commit `sha`."""
    prefix = gate.RUNS_PREFIX.rstrip("/")
    listed = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "-z", "--name-only", sha, "--", prefix],
                            capture_output=True, timeout=60)
    if listed.returncode != 0:
        raise RuntimeError(listed.stderr.decode("utf-8", errors="replace").strip()[:200] or f"git ls-tree failed for {sha}")
    out = {}
    for raw in listed.stdout.split(b"\0"):
        if not raw:
            continue
        path = raw.decode("utf-8")
        if not path.endswith(".json"):
            continue
        blob = subprocess.run(["git", "-C", repo, "show", f"{sha}:{path}"], capture_output=True, timeout=60)
        if blob.returncode != 0:
            raise RuntimeError(blob.stderr.decode("utf-8", errors="replace").strip()[:200] or f"git show failed for {sha}:{path}")
        out[path] = blob.stdout.decode("utf-8")
    return out


def head_files_at(sha, changed, fetch_fn=file_at):
    """Fetch added and modified run files at `sha`, matching community-review.yml's cap and path filter."""
    touched = [path for path, status in changed if status in ("added", "modified")]
    if len(touched) > gate.MAX_NEW_ROWS:
        return {}
    return {path: fetch_fn(path, sha) for path in touched if gate.PATH_RE.fullmatch(path)}


def main(argv):
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 40
    state = argv[argv.index("--state") + 1] if "--state" in argv else "all"
    review = "--review" in argv
    prs = json.loads(gh("pr", "list", "--repo", REPO, "--state", state, "--search", "community in:title",
                        "--limit", str(limit * 3), "--json", "number,title,state,author"))
    known = gate.git_known_commit()
    print(f"{'PR':>5} {'state':7} {'author':18} {'gate':5} {'model':12} notes")
    shown = 0
    for pr in prs:
        if shown >= limit:
            break
        try:
            d = json.loads(gh("pr", "view", str(pr["number"]), "--repo", REPO, "--json", "files,baseRefOid,headRefOid,mergeCommit"))
            if not is_community_pr(d["files"]):
                continue
            changed = normalize_changed(d["files"])
            mb = json.loads(gh("api", f"repos/{REPO}/compare/{d['baseRefOid']}...{d['headRefOid']}", "--jq", "{m: .merge_base_commit.sha}"))["m"]
            g = gate.run_gate(base_tree(mb), head_files_at(d["headRefOid"], changed), pr["author"]["login"], changed, known)
            verdict = "-"
            if review and g.status == "pass" and os.environ.get("LLM_API_KEY"):
                verdict = llm.community_review(g.new_rows, g.notes)["verdict"]
            note = "; ".join(g.reasons)[:110]
        except Exception as e:
            g, verdict, note = type("G", (), {"status": "error"}), "-", str(e)[:110]
        shown += 1
        print(f"{pr['number']:>5} {pr['state'].lower():7} {pr['author']['login'][:18]:18} {g.status:5} {verdict:12} {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
