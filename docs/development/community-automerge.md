# Community benchmark auto-merge and /review

Two workflows (community auto-merge and the `/review` comment) use one cheap OpenAI-compatible model through `tools/llm_review.py` (urllib, no dependencies).

## Community auto-merge

`community.yml` (on `pull_request`, no secrets) keeps validating the table. `community-review.yml` (on `pull_request_target`) decides what happens to a pull request that touches `docs/community/`. It checks out only the base commit and reads the pull request's run files through the API as data; nothing from the head is run.

Order, enforced by the workflow and by `llm_review.decide`:

1. **Gate** (`tools/community_gate.py`, no secrets). Any failure stops here: no model call, label `community: needs changes` and a comment with the reasons.
   - only files under `docs/community/runs/` changed (`MODEL/ID.json`, one row each), and each was added, modified or deleted. Anything else is a failure with its reason, including `preset.json` and the old `docs/community/benchmarks.json`. A modified or added file must be valid for `check_benchmarks.py`: the house format byte for byte, the folder of the model's slug and the id of its setup as its name, and no other file with the same setup (submitter, model, quant and hardware). At most 12 files are added or modified. The base is the commit the branch started from (the merge base), read from that commit's `docs/community/runs/`. What each change does:
     - an added file is a new row;
     - a modified file is an updated row, checked like a new one; it can merge automatically only when the pull request author's login is the `submitter` of the row both before and after (case-insensitive), otherwise it goes to a person (`community: needs review`);
     - a deleted file always goes to a person;
   - `preset` is `community-1` (the `submitter` is a display name and may differ from the PR author; the model review rejects an abusive one);
   - free text (submitter, notes, cpu, quant, GPU names, engine_args) has no link, domain, `@`, markup characters or invisible/direction characters, stays within its length cap, and never says LOSSY;
   - `engine_args` holds only `--gpu`, `--gpu-mode`, `--gpu-share`, `--max-context` and `--kv` with valid values (the preset's own arguments; `basalt-bench --community` refuses any other flag);
   - `commit` is an ancestor of `origin/main`, so it is a build of this repository;
   - numbers, for each added or modified file: `decode_tps` and `prefill_16k_tps` within 2.5x outside the lowest and highest earlier row of the same model, GPU name and GPU count (one card's quants spread about 4.5x in the table); decode at most 2000 and prefill at most 40000 tok/s; `prefill_64k_tps / prefill_16k_tps` between 0.3 and 6 (the table spans 0.70 to 4.06). A card or model with no earlier row has nothing to compare with, so it goes to a person without a model call.
2. **Model review** (only after a pass, `tools/llm_review.py`). It sees the added and modified rows and the check results, temperature 0, with the rows declared untrusted data. It answers `{"verdict": "auto_merge" | "needs_human" | "reject", "reason": "..."}`; anything else, a timeout, an API error or a missing key is `needs_human`. It can only downgrade: the one merge needs a passing gate and `auto_merge`.
3. **Act** (`community_apply.py`, job `act`).
   - `auto_merge`: squash-merge, pinned to the reviewed head commit (`--match-head-commit`), with the App token.
   - `needs_human`: label `community: needs review` and a review request to jesdga95.
   - `reject`: label `community: rejected` and one neutral comment. The pull request is never closed.

`COMMUNITY_AUTOMERGE`: only the exact value `live` merges. Unset, `shadow`, `test` or a typo is shadow: labels, review requests and comments only (a row that would merge gets `community: automatic merge` and a comment; in live it gets the same label and is merged). A shadow comment ends with the mode and why nothing merged; a live comment has no mode note.

Security model: the review runs from base code with secrets only in the one model step; a fork pull request cannot change what runs. Jobs have minimal `permissions` (the deciding job is read-only; the acting job has no `contents: write`); the merge token is a GitHub App token minted only in a merge run; the comments use a second App token scoped to pull requests only. Model output only selects a label. Reasons shown in comments are stripped of links, mentions and markup. Actions are pinned by SHA.

### Setup (once, on GitHub)

1. Secret `LLM_API_KEY` (repository secrets). Variables `LLM_BASE_URL` (the API root, such as `https://api.openai.com/v1`) and `LLM_DEFAULT_MODEL` (`gpt-6-luna`).
2. Variable `COMMUNITY_AUTOMERGE` = `test` (shadow). The workflow creates the labels itself.
3. GitHub App (`basalt-code-review-bot`), which merges and also posts the community and `/review` comments (they fall back to `github-actions` while it is not configured): Settings, Developer settings, GitHub Apps, New. Name it such as `basalt-code-review-bot`, no webhook, repository permissions Contents: read and write, Pull requests: read and write, installable on this account only. Generate a private key. Install it on `jesdga95/basalt` only.
4. Repository variable `BASALT_REVIEW_APP_CLIENT_ID` (the App's client ID) and secret `BASALT_REVIEW_APP_PRIVATE_KEY` (the whole `.pem`).
5. Ruleset `main` (Settings, Rules, Rulesets): Bypass list, Add bypass, choose the App, mode "Always". The ruleset requires a pull request, which the merge API call would otherwise fail; the bypass lets the App's squash merge through while everyone else still needs a pull request. No status checks are required.
6. Run `python3 tools/community_calibrate.py --review` with the three `LLM_*` values in the environment and look at the table.
7. After shadow runs on real pull requests look right, set `COMMUNITY_AUTOMERGE` to `live`. Setting anything else turns merging off again.

## /review

A comment on a pull request that starts with `/review` and is written by `jesdga95` with author association OWNER runs `review.yml`. The command takes optional settings before the focus text:

```
/review                                    the defaults: LLM_REVIEW_EFFORT (medium if unset) and LLM_DEFAULT_MODEL (gpt-6-luna)
/review effort=high                        low | medium | high (16k, 32k, 64k output tokens, reasoning included)
/review effort=low model=gpt-6.2-sol check the locking in cpu_experts.cpp
```

`model` must be `LLM_DEFAULT_MODEL` or listed in the variable `LLM_REVIEW_MODELS` (comma-separated, such as `gpt-6.1-terra,gpt-6.2-sol`); an unknown effort or model is answered with a one-line comment and no model call. Settings stop at the first word that is not one, so `check that x=5` stays focus text. The rest of the comment, newlines kept, is the focus. The footer names the model and the effort. It also approves the pull request's CI runs that wait for a maintainer (a fork's first runs, `action_required` on its head commit), so `/review` is the "Approve and run workflows" click too. The workflow reacts with 👀 while it runs, reads the diff with `gh pr diff` (data; no checkout of the pull request), and posts one comment.

`tools/code_review.py` splits the diff per file, skips generated, data and binary files (`data/`, `*.ids`, lock files, images, the community table), numbers the new-file lines, cuts a file past about 6k tokens at a hunk, sends chunks of about 7k tokens within a 40k token budget, and tells the model the rules of `AGENTS.md` and `CONTRIBUTING.md`. Replies must match the review schema (a two-to-three sentence plain summary, a verdict and the findings); the comment opens with the verdict in bold (**Merge**, **Merge with changes**, **Product decision** or **Needs additional review**; with several parts the worst one, and a part that failed makes it Needs additional review), then the summary, a **Product decision:** line naming any changed default, user-visible behavior, removed option or new dependency (flagged even when the code is right), then the validated findings ranked high to low with `file:line`, or `No findings.` Parts that failed or were skipped are named.

## Community table sync

`community-sync.yml` runs on every push to `main` that touches `docs/community/runs/**` or `docs/community/preset.json`, and by hand (`workflow_dispatch`). It runs `tools/community_sync.py`, which rebuilds the Cloudflare D1 table the landing page's `/api/runs` reads: all runs are deleted and every run file inserted again, so the table never drifts from the files.

- Setup (once): secret `CLOUDFLARE_D1_TOKEN` (a token that may edit D1), variables `CLOUDFLARE_ACCOUNT_ID` and `COMMUNITY_D1_ID` (the `basalt-community` database). A missing value fails the run by name; there is no fallback.
- Tables, made by the script when missing: `runs(gen, id, model, data)` with the row's JSON in `data`, and `meta(key, value)` holding the preset and the live generation. A change of the row format needs no migration.
- No half-filled table: the new rows go in under the next generation number, which no reader selects, and one `UPDATE` of `meta.gen` makes them live. The REST endpoint does not document its multi-statement requests as transactions, and a table rename cannot replace a table in one statement, so the flip is a single statement instead. The generation before the live one is deleted by the next run. A failed run leaves the live table as it was.
- Limits, from the D1 limits page: 100 bound parameters per query (so a few rows per INSERT), 100 KB per statement, 30 seconds per request. A request carries a bounded number of statements and bytes.
- The landing Worker reads with `SELECT data FROM runs WHERE gen = (SELECT CAST(value AS INTEGER) FROM meta WHERE key = 'gen')`.

## Tests

`tools/test_community_gate.py` (gate with fixture files, the mode rule, workflow structure), `tools/test_community_sync.py` (the table sync) and `tools/test_llm_review.py` (client against a local HTTP server, verdict parsing, review chunking) run in the CI lint job.
