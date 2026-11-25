"""Tests of the community sync: the SQL and batches it builds, the D1 REST client against a fake server, the rebuild against
SQLite (the dialect D1 runs), and the workflow's structure.

    python3 tools/test_community_sync.py
"""
import io
import json
import os
import re
import sqlite3
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_benchmarks as cb  # noqa: E402
import community_sync as cs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOKEN = "tok-SECRET-123"


def rows(n, size=50):
    return [(f"{i:012x}", "m", json.dumps({"i": i, "pad": "x" * size})) for i in range(n)]


class FakeDB:
    """The D1 client's two calls, run on SQLite; records every statement. `fail_at` raises on the n-th insert batch."""

    def __init__(self, fail_at=None):
        self.c = sqlite3.connect(":memory:")
        self.log, self.inserts, self.fail_at = [], 0, fail_at
        unittest.addModuleCleanup(self.c.close)

    def run(self, sql, params):
        self.log.append(sql)
        cur = self.c.execute(sql, params)
        cols = [d[0] for d in cur.description or []]
        return {"results": [dict(zip(cols, r)) for r in cur.fetchall()], "success": True}

    def query(self, sql, params=()):
        return [self.run(sql, params)]

    def batch(self, statements):
        if statements and statements[0][0].startswith("INSERT OR REPLACE INTO runs"):
            self.inserts += 1
            if self.fail_at == self.inserts:
                raise RuntimeError("boom")
        out = [self.run(s, p) for s, p in statements]
        self.c.commit()
        return out

    def read(self):
        """What the Worker's two queries return."""
        data = [r[0] for r in self.c.execute(cs.READER_SQL[0])]
        preset = [r[0] for r in self.c.execute(cs.READER_SQL[1])]
        return data, preset


class Statements(unittest.TestCase):
    def test_every_row_once_within_the_limits(self):
        data = rows(1000)
        stmts = cs.insert_statements(7, data)
        self.assertEqual([p for _, ps in stmts for p in [ps[i:i + 3] for i in range(0, len(ps), 3)] for p in [tuple(p)]], data)
        for sql, params in stmts:
            self.assertLessEqual(len(params), cs.PARAMS_PER_QUERY)
            self.assertEqual(sql.count("?"), len(params))
            self.assertEqual(sql.count("(7, ?, ?, ?)"), len(params) // 3)
            self.assertLessEqual(len(sql), cs.STATEMENT_BYTES)
        self.assertEqual(len(stmts), -(-1000 // cs.ROWS_PER_STATEMENT))

    def test_big_rows_split_a_statement_by_size(self):
        big = rows(10, size=30_000)
        for sql, params in cs.insert_statements(1, big):
            self.assertLessEqual(sum(len(p) for p in params), cs.STATEMENT_BYTES)
        self.assertGreater(len(cs.insert_statements(1, big)), 2)
        with self.assertRaises(ValueError):
            cs.insert_statements(1, rows(1, size=cs.STATEMENT_BYTES))

    def test_nothing_unbound_in_the_sql(self):
        evil = [("id'; DROP TABLE runs; --", "m'x", "d\"'\\")]
        sql, params = cs.insert_statements(1, evil)[0]
        self.assertNotIn("DROP", sql)
        self.assertEqual(params, list(evil[0]))
        with self.assertRaises(TypeError):
            cs.insert_statements("1; DROP TABLE runs", evil)

    def test_100k_rows_make_bounded_requests(self):
        groups = list(cs.batches(cs.insert_statements(1, rows(100_000, size=700))))
        self.assertEqual(sum(len(g) for g in groups), len(cs.insert_statements(1, rows(100_000, size=700))))
        for g in groups:
            self.assertLessEqual(len(g), cs.STATEMENTS_PER_REQUEST)
            self.assertLessEqual(sum(len(s) + sum(len(p) for p in ps) for s, ps in g), cs.REQUEST_BYTES + 100_000)

    def test_run_rows_are_the_files(self):
        problems, runs = cb.check_dir(str(ROOT / "docs" / "community"))
        self.assertEqual(problems, [])
        out = cs.run_rows(runs)
        self.assertEqual(len(out), len(runs))
        for (rel, row), (i, model, data) in zip(runs, out):
            self.assertEqual((i, model, json.loads(data)), (cb.row_id(row), row["model"], row))
            self.assertNotIn("\n", data)

    def test_run_rows_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            cs.run_rows([("runs/m/0123456789ab.json", {"model": "m", "decode_tps": float("inf")})])
        with self.assertRaises(ValueError):
            cs.run_rows([("runs/m/0123456789ab.json", {"model": "m", "decode_tps": float("nan")})])


class Client(unittest.TestCase):
    def client(self, answers):
        calls = []

        def opener(req, timeout=None):
            calls.append(req)
            a = answers.pop(0)
            if isinstance(a, Exception):
                raise a
            return io.BytesIO(json.dumps(a).encode())
        return cs.D1("acct", "db", TOKEN, opener=opener, sleep=lambda s: None), calls

    @staticmethod
    def http(code, body):
        return urllib.error.HTTPError("https://api.cloudflare.com/x", code, "x", {}, io.BytesIO(json.dumps(body).encode()))

    def test_request_shape(self):
        d1, calls = self.client([{"success": True, "result": [{"results": [], "success": True}]}] * 2)
        d1.query("SELECT ?", ["a"])
        d1.batch([("INSERT 1", ["x"]), ("INSERT 2", [])])
        self.assertEqual(calls[0].full_url, "https://api.cloudflare.com/client/v4/accounts/acct/d1/database/db/query")
        self.assertEqual(calls[0].get_method(), "POST")
        self.assertEqual(calls[0].get_header("Authorization"), "Bearer " + TOKEN)
        self.assertEqual(json.loads(calls[0].data), {"sql": "SELECT ?", "params": ["a"]})
        self.assertEqual(json.loads(calls[1].data), {"batch": [{"sql": "INSERT 1", "params": ["x"]}, {"sql": "INSERT 2", "params": []}]})

    def test_retries_rate_limits_and_server_errors(self):
        ok = {"success": True, "result": []}
        d1, calls = self.client([self.http(429, {}), self.http(503, {}), urllib.error.URLError("reset"), ok])
        self.assertEqual(d1.query("x"), [])
        self.assertEqual(len(calls), 4)

    def test_failures_are_loud_and_never_hold_the_token(self):
        for answers in ([self.http(400, {"errors": [{"message": "bad sql"}]})],
                        [self.http(403, {"errors": [{"code": 10000, "message": "Authentication error"}]})],
                        [{"success": False, "errors": [{"message": "nope"}], "result": []}],
                        [{"success": True, "result": [{"success": False}]}],
                        [self.http(500, {})] * 5, [urllib.error.URLError("down")] * 5):
            d1, calls = self.client(list(answers))
            with self.assertRaises(RuntimeError) as cm:
                d1.query("x")
            self.assertNotIn(TOKEN, str(cm.exception))
        d1, _ = self.client([self.http(400, {"errors": [{"message": "bad sql"}]})])
        with self.assertRaises(RuntimeError) as cm:
            d1.query("x")
        self.assertIn("HTTP 400", str(cm.exception))
        self.assertIn("bad sql", str(cm.exception))


class Rebuild(unittest.TestCase):
    PRESET = "basalt-bench --community"

    def test_first_sync_fills_and_flips(self):
        db = FakeDB()
        data = rows(100)
        self.assertEqual(cs.sync(db, self.PRESET, data, log=lambda m: None), 1)
        got, preset = db.read()
        self.assertEqual(sorted(got), sorted(r[2] for r in data))
        self.assertEqual(preset, [self.PRESET])

    def test_a_rebuild_replaces_everything_and_keeps_one_generation_back(self):
        db = FakeDB()
        cs.sync(db, self.PRESET, rows(40), log=lambda m: None)
        second = rows(10, size=7)
        self.assertEqual(cs.sync(db, "p2", second, log=lambda m: None), 2)
        got, preset = db.read()
        self.assertEqual(sorted(got), sorted(r[2] for r in second))   # the removed runs are gone, nothing is stale
        self.assertEqual(preset, ["p2"])
        self.assertEqual(sorted(r[0] for r in db.c.execute("SELECT DISTINCT gen FROM runs")), [1, 2])
        cs.sync(db, "p2", second, log=lambda m: None)
        self.assertEqual(sorted(r[0] for r in db.c.execute("SELECT DISTINCT gen FROM runs")), [2, 3])

    def test_the_flip_is_the_last_statement_and_one_update(self):
        db = FakeDB()
        cs.sync(db, self.PRESET, rows(70), log=lambda m: None)
        flips = [i for i, s in enumerate(db.log) if s == cs.UPSERT_META]
        self.assertEqual(flips[-1], len(db.log) - 1)
        self.assertTrue(all(not s.startswith("INSERT OR REPLACE INTO runs") for s in db.log[flips[-1]:]))
        self.assertTrue(all(s.startswith("INSERT OR REPLACE INTO runs") or not s.startswith("DELETE") or "gen <>" in s for s in db.log))

    def test_insert_retry_is_idempotent(self):
        db = FakeDB()
        db.batch([(s, []) for s in cs.SCHEMA])
        data = rows(50)
        stmts = cs.insert_statements(1, data)
        db.batch(stmts)
        # Simulates D1 executing a batch before a transient 502/timeout triggers a retry of the exact same statements.
        db.batch(stmts)
        got = db.query("SELECT COUNT(*) AS n FROM runs WHERE gen = 1")[0]["results"][0]["n"]
        self.assertEqual(got, len(data))

    def test_a_failed_rebuild_leaves_the_live_table_untouched(self):
        db = FakeDB()
        cs.sync(db, self.PRESET, rows(40), log=lambda m: None)
        before = db.read()
        db.fail_at = db.inserts + 2   # the second insert request of the next run
        with self.assertRaises(RuntimeError):
            cs.sync(db, "other", rows(3000), log=lambda m: None)
        self.assertEqual(db.read(), before)
        # the next run cleans the partial generation and goes on
        db.fail_at = None
        self.assertEqual(cs.sync(db, self.PRESET, rows(5), log=lambda m: None), 2)
        self.assertEqual(len(db.read()[0]), 5)

    def test_a_count_mismatch_is_not_made_live(self):
        db = FakeDB()
        cs.sync(db, self.PRESET, rows(10), log=lambda m: None)
        before = db.read()
        original = db.batch

        def lossy(statements):
            out = original(statements)
            if statements[0][0].startswith("INSERT OR REPLACE INTO runs"):
                db.c.execute("DELETE FROM runs WHERE gen = 2 AND rowid = (SELECT MIN(rowid) FROM runs WHERE gen = 2)")
            return out
        db.batch = lossy
        with self.assertRaises(RuntimeError):
            cs.sync(db, self.PRESET, rows(10), log=lambda m: None)
        self.assertEqual(db.read(), before)

    def test_an_empty_table_is_a_valid_rebuild(self):
        db = FakeDB()
        cs.sync(db, self.PRESET, [], log=lambda m: None)
        self.assertEqual(db.read(), ([], [self.PRESET]))

    def test_the_real_runs_load(self):
        problems, runs = cb.check_dir(str(ROOT / "docs" / "community"))
        db = FakeDB()
        cs.sync(db, self.PRESET, cs.run_rows(runs), log=lambda m: None)
        got, _ = db.read()
        self.assertEqual(sorted(json.loads(g)["commit"] + json.loads(g)["submitter"] for g in got),
                         sorted(r["commit"] + r["submitter"] for _, r in runs))


class Main(unittest.TestCase):
    def test_missing_settings_fail_by_name(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("builtins.print") as pr:
            self.assertEqual(cs.main(["x"]), 2)
        out = " ".join(str(c.args[0]) for c in pr.call_args_list)
        for name in cs.ENV:
            self.assertIn(name, out)
        env = {"CLOUDFLARE_D1_TOKEN": TOKEN, "CLOUDFLARE_ACCOUNT_ID": "a", "COMMUNITY_D1_ID": ""}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("builtins.print") as pr:
            self.assertEqual(cs.main(["x"]), 2)
        out = " ".join(str(c.args[0]) for c in pr.call_args_list)
        self.assertIn("COMMUNITY_D1_ID", out)
        self.assertNotIn(TOKEN, out)

    def test_invalid_run_files_stop_before_any_call(self):
        env = {"CLOUDFLARE_D1_TOKEN": TOKEN, "CLOUDFLARE_ACCOUNT_ID": "a", "COMMUNITY_D1_ID": "d"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("builtins.print"), \
                mock.patch.object(cs, "D1", side_effect=AssertionError("called")):
            self.assertEqual(cs.main(["x", str(ROOT / "docs" / "nonexistent")]), 1)

    def test_the_names_and_the_token_never_reach_the_output(self):
        env = {"CLOUDFLARE_D1_TOKEN": TOKEN, "CLOUDFLARE_ACCOUNT_ID": "a", "COMMUNITY_D1_ID": "d"}
        err = urllib.error.HTTPError("https://x", 401, "x", {"Authorization": "Bearer " + TOKEN}, io.BytesIO(b'{"errors":[{"message":"denied"}]}'))
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("builtins.print") as pr, \
                mock.patch("urllib.request.urlopen", side_effect=err):
            self.assertEqual(cs.main(["x", str(ROOT / "docs" / "community")]), 1)
        out = " ".join(str(c.args[0]) for c in pr.call_args_list)
        self.assertIn("denied", out)
        self.assertNotIn(TOKEN, out)


class Workflow(unittest.TestCase):
    text = (ROOT / ".github/workflows/community-sync.yml").read_text()

    def test_triggers(self):
        self.assertRegex(self.text, r"push:\s+branches: \[main\]\s+paths:\s+- \"docs/community/runs/\*\*\"\s+- \"docs/community/preset.json\"")
        self.assertIn("workflow_dispatch:", self.text)
        self.assertIn("github.ref == 'refs/heads/main'", self.text)

    def test_the_three_settings_by_their_names(self):
        self.assertIn("CLOUDFLARE_D1_TOKEN: ${{ secrets.CLOUDFLARE_D1_TOKEN }}", self.text)
        self.assertIn("CLOUDFLARE_ACCOUNT_ID: ${{ vars.CLOUDFLARE_ACCOUNT_ID }}", self.text)
        self.assertIn("COMMUNITY_D1_ID: ${{ vars.COMMUNITY_D1_ID }}", self.text)
        self.assertEqual(self.text.count("secrets."), 1)
        self.assertNotIn("||", self.text)   # no fallback for a missing value
        self.assertNotIn("set -x", self.text)

    def test_runs_one_at_a_time_and_is_pinned(self):
        self.assertRegex(self.text, r"concurrency:\s+group: community-sync\s+cancel-in-progress: false")
        for m in re.finditer(r"uses: (\S+)", self.text):
            self.assertRegex(m.group(1), r"@[0-9a-f]{40}$")
        self.assertIn("contents: read", self.text)
        self.assertIn("python3 tools/community_sync.py docs/community", self.text)


if __name__ == "__main__":
    unittest.main()
