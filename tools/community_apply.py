#!/usr/bin/env python3
"""Turn a community outcome (llm_review.decide) into the actions the workflow takes (stdlib only).

    community_apply.py OUTCOME.json MODE_VALUE PLAN.json

Only the exact mode value "live" merges. Any other value (unset, shadow, test, a typo) is shadow: labels, review
requests and comments only, and the comment says which mode ran.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_review import plain  # noqa: E402

LABELS = ("community: needs review", "community: rejected", "community: needs changes", "community: would auto-merge")
OWNER = "jesdga95"


def mode_of(value):
    return "live" if value == "live" else "shadow"


def plan(outcome, mode_value):
    mode = mode_of(mode_value)
    shown = "live" if mode == "live" else f"shadow (COMMUNITY_AUTOMERGE is '{plain(mode_value or '', 20)}'; only 'live' merges)"
    p = {"mode": mode, "merge": False, "labels": [], "request_review": False, "comment": ""}
    o = outcome["outcome"]
    if o == "merge":
        if mode == "live":
            p["merge"] = True
        else:
            p["labels"] = ["community: would auto-merge"]
            p["comment"] = "Automatic review: this row passed every check and would be merged."
    elif o == "reject":
        p["labels"] = ["community: rejected"]
        p["comment"] = outcome["comment"]
    elif o == "gate_fail":
        p["labels"] = ["community: needs changes"]
        p["comment"] = outcome["comment"]
    else:
        p["labels"] = ["community: needs review"]
        p["request_review"] = True
    if p["comment"]:
        p["comment"] += f"\n\nMode: {shown}."
    return p


def main(argv):
    outcome = json.load(open(argv[1], encoding="utf-8"))
    out = plan(outcome, argv[2] if len(argv) > 2 else "")
    json.dump(out, open(argv[3], "w", encoding="utf-8"))
    print(json.dumps({k: v for k, v in out.items() if k != "comment"}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
