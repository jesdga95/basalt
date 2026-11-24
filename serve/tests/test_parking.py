"""The model cache: the budget and LRU books, and the router parking a model it switches away from instead of ending it."""
from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve import server
from serve.frontend import ChatTemplate
from serve.parking import ModelCache
from serve.router import Router
from serve.service import Service
from serve.tests.test_router import Base, SwapEngine, TEMPLATE
from serve.util.text import ByteTokenizer


class Books(unittest.TestCase):
    def test_a_budget_below_one_mib_is_refused(self):
        with self.assertRaises(ValueError):
            ModelCache(0)

    def test_what_fits_evicts_nothing(self):
        c = ModelCache(1000)
        c.add("a", 400)
        self.assertEqual(c.eviction_for(600), [])
        self.assertEqual(c.total_mib(), 400)

    def test_the_least_recently_used_goes_first_and_only_as_many_as_needed(self):
        c = ModelCache(1000)
        for k, mib in (("a", 300), ("b", 300), ("c", 300)):
            c.add(k, mib)
        self.assertEqual(c.eviction_for(200), ["a"])
        self.assertEqual(c.eviction_for(500), ["a", "b"])
        self.assertEqual(c.eviction_for(1000), ["a", "b", "c"])

    def test_parking_again_makes_it_the_most_recent(self):
        c = ModelCache(1000)
        c.add("a", 300)
        c.add("b", 300)
        c.add("a", 350)
        self.assertEqual(c.keys(), ["b", "a"])
        self.assertEqual(c.total_mib(), 650)

    def test_more_than_the_budget_never_fits(self):
        c = ModelCache(1000)
        c.add("a", 100)
        self.assertIsNone(c.eviction_for(1001))
        self.assertEqual(c.eviction_for(1000), ["a"])

    def test_taking_a_model_returns_what_it_held(self):
        c = ModelCache(1000)
        c.add("a", 300)
        self.assertEqual(c.take("a"), 300)
        self.assertNotIn("a", c)
        self.assertEqual(c.total_mib(), 0)

    def test_over_budget_names_the_oldest_until_it_fits(self):
        c = ModelCache(1000)
        for k, mib in (("a", 500), ("b", 500), ("c", 500)):
            c.add(k, mib)
        self.assertEqual(c.over_budget(), ["a"])
        c.add("d", 900)
        self.assertEqual(c.over_budget(), ["a", "b", "c"])

    def test_low_ram_names_the_oldest_only_when_it_is_low_and_known(self):
        c = ModelCache(1000)
        self.assertIsNone(c.short_of_ram(10, 100))            # nothing parked
        c.add("a", 300)
        c.add("b", 300)
        self.assertEqual(c.short_of_ram(99, 100), "a")
        self.assertIsNone(c.short_of_ram(100, 100))
        self.assertIsNone(c.short_of_ram(None, 100))          # unreadable: nothing is guessed


class ParkEngine(SwapEngine):
    """A swap engine that can sleep: sleeping and waking are recorded; an evicted one is gone."""

    def can_sleep(self):
        return True

    def sleep(self, ram_check=None, timeout=120.0):
        why = ram_check() if ram_check is not None else None
        if why is not None:
            raise ValueError(why)
        self.asleep = True
        self.parked = {0: 1}
        self.events.append(("sleep", self.name))

    def wake(self, timeout=120.0, vram_check=None):
        self.asleep = False
        self.events.append(("wake", self.name))

    def discard(self):
        self.events.append(("evict", self.name))
        self.up, self.unloaded, self.asleep = False, True, False

    def held_mib(self):
        return 0

    def close(self):
        self.events.append(("close", self.name))
        self.up, self.asleep = False, False


class Parking(Base):
    NAMES = ("A", "B", "C")
    BUDGET = 10_000
    HELD = {"A": 3000, "B": 3000, "C": 3000}
    REFUSE = {}
    FLOOR = 3072

    def setUp(self):
        tok = ByteTokenizer()
        self.events = []
        self.engines = {n: ParkEngine(tok, n, self.events) for n in self.NAMES}
        self.svcs = {n: Service(e, tok, ChatTemplate.from_file(TEMPLATE), model_name=n) for n, e in self.engines.items()}
        for n, svc in self.svcs.items():
            self.engines[n].model_path = f"/packs/{n}.basalt"
            svc.park_need_mib = lambda n=n: self.HELD[n]
            svc.held_mib = lambda n=n: self.HELD[n]
            svc.park_refusal = lambda floor, n=n: self.REFUSE.get(n)
        self.router = Router(list(self.svcs.values()), self.BUDGET, self.FLOOR)
        self.out = io.StringIO()
        self.printing = mock.patch("builtins.print", lambda *a, **k: self.out.write(" ".join(map(str, a)) + "\n"))
        self.printing.start()
        from serve.handler import serve
        self.httpd = serve(self.router, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.printing.stop()
        super().tearDown()

    def parked(self):
        return [s.model for s in self.router.cache.keys()]

    def test_a_model_swapped_away_from_sleeps_and_wakes_for_its_next_turn(self):
        for name in ("A", "B", "A"):
            self.assertEqual(self.chat(name)[0], 200)
        self.assertEqual([e for e in self.events if e[0] in ("start", "end", "sleep", "wake", "evict")],
                         [("start", "A"), ("sleep", "A"), ("start", "B"), ("sleep", "B"), ("wake", "A")])
        self.assertEqual(self.parked(), ["B"])
        self.assertIn("parked A: 2.9 GiB in RAM, 2.9 GiB of 9.8 GiB parked", self.out.getvalue())

    def test_models_report_what_they_hold_parked(self):
        self.chat("A")
        self.chat("B")
        rows = {m["id"]: m for m in self.get("/v1/models")[1]["data"]}
        self.assertEqual((rows["A"]["parked_mib"], rows["B"]["parked_mib"], rows["C"]["parked_mib"]), (3000, None, None))

    def test_the_least_recently_used_is_ended_when_the_budget_is_full(self):
        self.router.cache.budget_mib = 5000
        for name in ("A", "B", "C"):
            self.assertEqual(self.chat(name)[0], 200)
        self.assertIn(("evict", "A"), self.events)
        self.assertEqual(self.parked(), ["B"])
        self.assertIn("unparked A: the budget needs", self.out.getvalue())

    def test_a_model_over_the_whole_budget_is_ended_and_says_so(self):
        self.router.cache.budget_mib = 2000
        self.chat("A")
        self.chat("B")
        self.assertEqual(self.parked(), [])
        self.assertIn(("end", "A"), self.events)
        self.assertIn("not parked: A: it holds 2.9 GiB, over the 2.0 GiB budget, ended", self.out.getvalue())

    def test_a_ram_refusal_ends_the_model_and_says_why(self):
        self.REFUSE = {"A": "RAM: 100 MiB free"}
        self.chat("A")
        self.chat("B")
        self.assertEqual(self.parked(), [])
        self.assertEqual(self.events.count(("end", "A")), 1)
        self.assertIn("not parked: A: RAM: 100 MiB free, ended", self.out.getvalue())

    def test_low_ram_ends_the_least_recently_used_parked_model(self):
        for name in ("A", "B", "C"):
            self.chat(name)
        self.assertEqual(self.parked(), ["A", "B"])
        with mock.patch("serve.router.available_ram_mib", return_value=1000):
            self.router.guard_once()
        self.assertEqual(self.parked(), ["B"])
        self.assertIn(("evict", "A"), self.events)
        self.assertIn("under the 3072 MiB floor", self.out.getvalue())
        with mock.patch("serve.router.available_ram_mib", return_value=50_000):
            self.router.guard_once()
        self.assertEqual(self.parked(), ["B"])

    def test_a_parked_engine_that_ended_by_itself_leaves_the_books(self):
        self.chat("A")
        self.chat("B")
        self.engines["A"].up = False                           # an idle unload
        self.chat("C")
        self.assertEqual(self.parked(), ["B"])
        self.chat("A")
        self.assertEqual(self.events.count(("start", "A")), 2)

    def test_stopping_ends_the_loaded_model_before_the_parked_ones(self):
        for name in ("A", "B"):
            self.chat(name)
        closed = []
        for n, e in self.engines.items():
            e.close = lambda n=n: closed.append(n)
        self.router.stop()
        self.assertEqual(closed, ["B", "A"])


class Off(Base):
    def test_without_a_cache_a_swap_ends_the_engine_and_the_rows_say_nothing_of_parking(self):
        self.chat("A")
        self.chat("B")
        self.assertIn(("end", "A"), self.events)
        self.assertIsNone(self.router.cache)
        for row in self.get("/v1/models")[1]["data"]:
            self.assertNotIn("parked_mib", row)


class Setting(unittest.TestCase):
    def services(self, n=2, reserve=None):
        return [SimpleNamespace(engine=SimpleNamespace(spawn=(None, ["--ram-reserve-mib", str(reserve)] if reserve else [])))
                for _ in range(n)]

    def plan(self, mib, services, mock_engine=False, linux=True, total=65536):
        a = SimpleNamespace(model_cache_mib=mib, mock_engine=mock_engine)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch("serve.server.suspend.available", return_value=linux), \
                mock.patch("serve.server.meminfo_mib", return_value=total):
            return server.model_cache_plan(a, services), out.getvalue()

    def refused(self, *args, **kw):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
            self.plan(*args, **kw)
        return err.getvalue()

    def test_unset_is_off_and_prints_nothing(self):
        self.assertEqual(self.plan(0, self.services()), ((0, 0), ""))
        self.assertEqual(self.plan(0, self.services(1)), ((0, 0), ""))

    def test_the_floor_is_the_highest_reserve_and_a_margin_and_the_start_line_says_so(self):
        (budget, floor), line = self.plan(20000, self.services())
        self.assertEqual((budget, floor), (20000, 3072))
        self.assertIn("up to 20000 MiB", line)
        self.assertIn("under 3072 MiB", line)
        self.assertEqual(self.plan(20000, self.services(reserve=8192))[0], (20000, 9216))

    def test_a_setting_that_cannot_work_stops_the_start(self):
        self.assertIn("0 (off) or more", self.refused(-1, self.services()))
        self.assertIn("two or more models", self.refused(100, self.services(1)))
        self.assertIn("not --mock-engine", self.refused(100, self.services(), mock_engine=True))
        self.assertIn("Linux only", self.refused(100, self.services(), linux=False))
        self.assertIn("more than this machine can park", self.refused(63000, self.services()))


if __name__ == "__main__":
    unittest.main()
