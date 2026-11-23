#!/usr/bin/env python3
"""Mirror the community run files into the Cloudflare D1 table the landing page reads (stdlib only).

    community_sync.py [DIR]       DIR is docs/community by default

Rebuilds the table from scratch on every run, so there is no drift and no diff logic. It reads three settings from the
environment and fails without them: CLOUDFLARE_D1_TOKEN (a token with D1 edit permission), CLOUDFLARE_ACCOUNT_ID and
COMMUNITY_D1_ID (the database's id). Nothing it prints holds the token.

Schema, made here when missing (a change of the row format never needs a migration: `data` is the row's JSON):
    runs(gen INTEGER, id TEXT, model TEXT, data TEXT, PRIMARY KEY (gen, id))
    meta(key TEXT PRIMARY KEY, value TEXT)           'gen': the generation readers use, 'preset': the table's preset

No reader sees a half-filled table. The REST endpoint does not say whether the statements of one request run in a
transaction (only a Worker's db.batch() is documented as one), and a table rename cannot replace an existing table in one
statement, so the swap is one statement that is atomic on its own: the rows of a new generation are inserted under
gen N+1, which nothing reads, and then meta.gen is updated from N to N+1 in a single UPDATE. Readers select
`WHERE gen = (SELECT value FROM meta WHERE key = 'gen')`, one statement, so they see the old table or the new one. The
generation before the live one is deleted at the start of the next run, so a reader that picked the live one just before
a flip still finds its rows.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_benchmarks as cb  # noqa: E402

API = "https://api.cloudflare.com/client/v4/accounts/{account}/d1/database/{database}/query"
ENV = ("CLOUDFLARE_D1_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "COMMUNITY_D1_ID")

# D1 limits (https://developers.cloudflare.com/d1/platform/limits/): at most 100 bound parameters per query, 100,000
# bytes per SQL statement, 2,000,000 bytes per row, 30 seconds per query and per whole batch. The REST API documents no
# limit on the statements of one batch request, so a request carries a modest number of them and a bounded body.
PARAMS_PER_QUERY = 100
COLUMNS = 3                                   # id, model, data are bound; the generation is written into the SQL
ROWS_PER_STATEMENT = PARAMS_PER_QUERY // COLUMNS
STATEMENT_BYTES = 90_000                      # under the 100,000 byte statement limit, counting the bound values too to be safe
STATEMENTS_PER_REQUEST = 20
REQUEST_BYTES = 4_000_000                     # bound values per request, far under any request size limit and the 30 s
RETRIES = 4

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS runs (gen INTEGER NOT NULL, id TEXT NOT NULL, model TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (gen, id))",
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)
READ_GEN = "SELECT value FROM meta WHERE key = 'gen'"
# What the landing Worker runs to read the live generation, and the preset beside it (tests run both against SQLite).
READER_SQL = ("SELECT data FROM runs WHERE gen = (SELECT CAST(value AS INTEGER) FROM meta WHERE key = 'gen')",
              "SELECT value FROM meta WHERE key = 'preset'")
UPSERT_META = "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value"


def run_rows(runs):
    """[(id, model, data)] for the (relative path, row) pairs of check_benchmarks.check_dir: data is the row as compact JSON."""
    return [(Path(rel).stem, row["model"], json.dumps(row, ensure_ascii=False, separators=(",", ":"))) for rel, row in runs]


def insert_statements(gen, rows):
    """Multi-row INSERTs of the rows under generation `gen`: [(sql, params)]. A statement holds at most
    ROWS_PER_STATEMENT rows (the bound-parameter limit) and, counting its bound values too, STATEMENT_BYTES. The
    generation is an integer written into the SQL; everything else is bound."""
    if not isinstance(gen, int) or isinstance(gen, bool):
        raise TypeError("gen must be an integer")
    out, chunk, size = [], [], 0

    def flush():
        sql = f"INSERT INTO runs (gen, id, model, data) VALUES " + ", ".join(f"({gen}, ?, ?, ?)" for _ in chunk)
        out.append((sql, [v for r in chunk for v in r]))

    for r in rows:
        n = sum(len(v.encode("utf-8")) for v in r) + 40
        if n > STATEMENT_BYTES:
            raise ValueError(f"run {r[0]} is {n} bytes, over the statement limit")
        if chunk and (len(chunk) >= ROWS_PER_STATEMENT or size + n > STATEMENT_BYTES):
            flush()
            chunk, size = [], 0
        chunk.append(r)
        size += n
    if chunk:
        flush()
    return out


def batches(statements):
    """Groups of statements for one request: at most STATEMENTS_PER_REQUEST and REQUEST_BYTES of bound values."""
    group, size = [], 0
    for sql, params in statements:
        n = len(sql) + sum(len(p.encode("utf-8")) for p in params)
        if group and (len(group) >= STATEMENTS_PER_REQUEST or size + n > REQUEST_BYTES):
            yield group
            group, size = [], 0
        group.append((sql, params))
        size += n
    if group:
        yield group


class D1:
    """The one POST endpoint of the D1 REST API: {"sql", "params"} for a query, {"batch": [...]} for several."""

    def __init__(self, account, database, token, opener=None, sleep=time.sleep):
        self.url = API.format(account=account, database=database)
        self.token = token
        self.opener, self.sleep = opener or urllib.request.urlopen, sleep

    def post(self, body):
        """The `result` list of a successful call. The 429 and 5xx answers are retried; anything else raises RuntimeError
        with Cloudflare's own error messages (never the request's headers)."""
        data = json.dumps(body).encode("utf-8")
        for attempt in range(RETRIES + 1):
            req = urllib.request.Request(self.url, data=data, method="POST",
                                         headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
            try:
                with self.opener(req, timeout=60) as resp:
                    answer = json.load(resp)
            except urllib.error.HTTPError as e:
                try:
                    answer = json.load(e)
                except ValueError:
                    answer = {}
                finally:
                    e.close()
                if e.code in (429, 500, 502, 503, 504) and attempt < RETRIES:
                    self.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"D1 answered HTTP {e.code}: {self.errors(answer)}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < RETRIES:
                    self.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"D1 request failed: {getattr(e, 'reason', e)}") from None
            if not answer.get("success"):
                raise RuntimeError(f"D1 refused the request: {self.errors(answer)}")
            result = answer.get("result") or []
            for r in result:
                if r.get("success") is False:
                    raise RuntimeError(f"D1 refused a statement: {self.errors(answer)}")
            return result
        raise RuntimeError("D1 request failed")

    @staticmethod
    def errors(answer):
        msgs = [str(e.get("message", e)) if isinstance(e, dict) else str(e) for e in answer.get("errors") or []]
        return "; ".join(msgs) or "no error message"

    def query(self, sql, params=()):
        return self.post({"sql": sql, "params": list(params)})

    def batch(self, statements):
        return self.post({"batch": [{"sql": s, "params": list(p)} for s, p in statements]})


def live_generation(result):
    rows = result[0].get("results") if result else None
    return int(rows[0]["value"]) if rows else 0


def sync(db, preset, rows, log=print):
    """Rebuild the table. Returns the new generation."""
    db.batch([(s, []) for s in SCHEMA])
    old = live_generation(db.query(READ_GEN))
    new = old + 1
    # rows of an unfinished earlier run and of the generation before the live one; the live one stays for readers in flight
    db.query(f"DELETE FROM runs WHERE gen <> {old}")
    for group in batches(insert_statements(new, rows)):
        db.batch(group)
    got = db.query(f"SELECT COUNT(*) AS n FROM runs WHERE gen = {new}")[0]["results"][0]["n"]
    if got != len(rows):
        raise RuntimeError(f"generation {new} holds {got} rows, expected {len(rows)}; it was not made live")
    db.query(UPSERT_META, ["preset", preset])
    db.query(UPSERT_META, ["gen", str(new)])   # the flip: one statement
    log(f"community table: generation {new} live with {len(rows)} runs (generation {old} kept until the next sync)")
    return new


def main(argv):
    root = argv[1] if len(argv) > 1 else cb.DEFAULT_DIR
    missing = [k for k in ENV if not os.environ.get(k)]
    if missing:
        print("missing setting(s): " + ", ".join(missing))
        return 2
    problems, runs = cb.check_dir(root)
    if problems:
        print("\n".join(problems))
        print("the run files are not valid; nothing was synced")
        return 1
    preset = json.loads(cb.read_text(os.path.join(root, cb.PRESET_FILE)))["preset"]
    db = D1(os.environ["CLOUDFLARE_ACCOUNT_ID"], os.environ["COMMUNITY_D1_ID"], os.environ["CLOUDFLARE_D1_TOKEN"])
    try:
        sync(db, preset, run_rows(runs))
    except RuntimeError as e:
        print(f"sync failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
