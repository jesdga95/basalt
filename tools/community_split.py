#!/usr/bin/env python3
"""One-off: split docs/community/benchmarks.json into preset.json and runs/<model slug>/<id>.json (stdlib only).

    community_split.py [BENCHMARKS.json] [OUT_DIR]      OUT_DIR is docs/community by default

Each row is written unchanged in the house format. When two rows map to the same file the script writes nothing and
names both. It reads the file back afterwards and compares every row with the original.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_benchmarks as cb  # noqa: E402


def main(argv):
    src = argv[1] if len(argv) > 1 else "docs/community/benchmarks.json"
    out = argv[2] if len(argv) > 2 else cb.DEFAULT_DIR
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    seen = {}
    clash = False
    for i, row in enumerate(data["runs"]):
        path = cb.run_path(row)
        if path in seen:
            clash = True
            print(f"runs[{i}] ({row['submitter']}) and runs[{seen[path]}] ({data['runs'][seen[path]]['submitter']}) are the same file {path}")
        seen.setdefault(path, i)
    if clash:
        return 1
    root = os.path.join(out, cb.RUNS_DIR)
    for path, i in seen.items():
        full = os.path.join(root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        cb.write_text(full, cb.canonical_row(data["runs"][i]))
    cb.write_text(os.path.join(out, cb.PRESET_FILE), cb.canonical_preset(data["preset"]))
    back = {}
    for rel, full in cb.list_runs(out)[0]:
        back[rel.split("/", 1)[1]] = json.loads(cb.read_text(full))
    assert len(back) == len(data["runs"]), "file count differs from the row count"
    for path, i in seen.items():
        assert back[path] == data["runs"][i], f"{path} differs from its row"
    print(f"{len(seen)} run files written to {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
