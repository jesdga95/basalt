"""The engine's console lines (XSLOT, XMAP, INFO cards= and cpu_l3=) and what /slots and /metrics show of them."""
import threading
import unittest
from pathlib import Path

from serve.engine.console import (ConsoleLines, engine_facts, hardware_view, parse_cards, parse_l3, parse_xmap,
                                  parse_xslot)
from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.service import Service
from serve.util.text import ByteTokenizer

TEMPLATE = Path(__file__).resolve().parents[2] / "serve" / "chat_template.jinja"


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class Lines(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_xslot("XSLOT 2 10 25 80 100 20 3 9/10,5/9,1/4"),
                         (2, {"windows": 10, "tokens": 25, "hits": 80, "lookups": 100, "offloaded": 20, "files": 3,
                              "accept": [(9, 10), (5, 9), (1, 4)]}))
        self.assertEqual(parse_xslot("XSLOT 0 3 3 0 0 0 0 -")[1]["accept"], [])
        self.assertIsNone(parse_xslot("XSLOT 0 x"))
        self.assertEqual(parse_xmap("XMAP 20480 0123 4,0,1,2"), (20480, [0, 1, 2, 3], [4, 0, 1, 2]))
        self.assertIsNone(parse_xmap("XMAP 20480 012 4,0"))                 # one count per sampled expert

    def test_a_slots_acceptance_split_and_rates(self):
        clk = Clock()
        c = ConsoleLines(clock=clk)
        self.assertFalse(c.feed("T 5"))
        self.assertEqual(c.slot(0), {"accept_pos": None, "accept_hist": None, "hit": None})
        c.feed("XSLOT 0 10 30 60 90 10 0 8/10,4/8")
        clk.t += 1.0
        c.feed("XSLOT 0 20 60 120 180 20 0 16/20,8/16")
        v = c.slot(0)
        self.assertEqual(v["accept_pos"], [0.8, 0.5])
        self.assertEqual(v["hit"], {"vram": 0.6, "peer": 0.1, "cpu": 0.3, "lookups_s": 100, "peer_s": 10, "ram_s": 30})
        self.assertEqual(v["accept_hist"][-1], 3.0)                           # 30 tokens over 10 windows this second
        self.assertEqual(len(v["accept_hist"]), 60)
        clk.t += 5.0                                                          # its windows stopped: no rate
        v = c.slot(0)
        self.assertEqual((v["hit"]["lookups_s"], v["hit"]["ram_s"]), (0, 0))
        self.assertEqual(v["hit"]["vram"], 0.6)                               # the request's split stays

    def test_a_new_request_counts_from_zero(self):
        clk = Clock()
        c = ConsoleLines(clock=clk)
        c.feed("XSLOT 1 50 100 10 10 0 0 -")
        clk.t += 1
        c.feed("XSLOT 1 0 0 0 0 0 0 -")                                       # the next request's start
        clk.t += 1
        c.feed("XSLOT 1 4 12 0 0 0 0 8/8")
        v = c.slot(1)
        self.assertEqual(v["accept_hist"][-1], 3.0)
        self.assertEqual(v["accept_pos"], [1.0])
        self.assertIsNone(v["hit"])                                           # no lookups: a dense model

    def test_a_request_in_several_engine_runs_adds_them_up(self):
        clk = Clock()
        c = ConsoleLines(clock=clk)
        c.feed("XSLOT 0 9 9 9 10 0 0 4/8")                                    # the previous request
        clk.t += 10
        started = clk.t
        clk.t += 0.1
        c.feed("XSLOT 0 0 0 0 0 0 0 -")
        c.feed("XSLOT 0 10 30 60 90 10 0 8/10,4/8")                           # its first run
        clk.t += 0.5
        c.feed("XSLOT 0 0 0 0 0 0 0 -")                                       # the next pass, a run of its own
        c.feed("XSLOT 0 5 10 30 40 0 0 2/4")
        v = c.slot(0, since=started)
        self.assertEqual(v["accept_pos"], [round(10 / 14, 3), 0.5])
        self.assertEqual((v["hit"]["vram"], v["hit"]["cpu"]), (round(90 / 140, 3), round(40 / 140, 3)))
        self.assertEqual((v["drafts_accepted"], v["drafts_offered"]), (14, 22))   # every run's drafts
        self.assertEqual(c.slot(0)["accept_pos"], [0.5])                     # no start given: the last run
        clk.t += 5
        self.assertEqual(c.slot(0, since=clk.t - 1), {"accept_pos": None, "accept_hist": None, "hit": None})   # still reading

    def test_the_expert_map_fades(self):
        clk = Clock()
        c = ConsoleLines(clock=clk)
        self.assertIsNone(c.expert_map())
        c.feed("XMAP 8 0123 2,0,0,4")
        m = c.expert_map()
        self.assertEqual((m["total"], m["tier"], m["routed"], m["recent"]), (8, [0, 1, 2, 3], [2.0, 0, 0, 4.0], [2.0, 0, 0, 4.0]))
        clk.t += 0.5
        c.feed("XMAP 8 0123 1,0,0,0")
        m = c.expert_map()
        self.assertEqual(m["routed"], [2.0, 0, 0, 2.67])                      # a second over this line and the one before
        self.assertAlmostEqual(m["recent"][0], 2 * 0.85 ** 0.5 + 1, places=2)
        clk.t += 10                                                           # nothing since: no lookups now, faded
        m = c.expert_map()
        self.assertEqual(m["routed"], [0, 0, 0, 0])
        self.assertLess(m["recent"][3], 1.0)

    def test_expert_map_is_thread_safe_under_concurrent_xmap_feeds(self):
        c = ConsoleLines()
        c.feed("XMAP 8 0123 1,2,3,4")
        stop = threading.Event()
        errors = []

        def writer():
            try:
                for i in range(1000):
                    c.feed(f"XMAP 8 0123 {i % 5},1,2,3")
            except BaseException as exc:
                errors.append(exc)
            finally:
                stop.set()

        def reader():
            try:
                while not stop.is_set():
                    m = c.expert_map()
                    if m is not None:
                        self.assertEqual(m["total"], 8)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer), *(threading.Thread(target=reader) for _ in range(3))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(errors, [])


class Hardware(unittest.TestCase):
    def test_cards_and_groups(self):
        self.assertEqual(parse_cards("0000:01:00.0/9052/52.1/23869/512;0000:08:00.0/5497/-/14500/0;bad/1/2"),
                         {"0:1:0.0": {"expert_slots": 9052, "h2d_gbps": 52.1, "experts": 23869 << 20, "kv": 512 << 20},
                          "0:8:0.0": {"expert_slots": 5497, "h2d_gbps": None, "experts": 14500 << 20, "kv": 0}})
        self.assertEqual(parse_l3("0-2,8;3-4"), [[0, 1, 2, 8], [3, 4]])
        info = {"cards": "0000:01:00.0/9052/52.1/20480/-;0000:08:00.0/5497/-/10240/0", "cpu_l3": "0-1;2-3", "expert_slots": 388}
        cards, l3 = engine_facts(info)
        self.assertEqual(info, {"expert_slots": 14549})                      # every card's experts, one count
        now = {"gpus": [{"index": 0, "mem_used": 30 << 30}, {"index": 1, "mem_used": 12 << 30}], "cpu_cores": [10.0, 30.0, 50.0, 50.0], "ram_used": 100 << 30,
               "ram_cached": 20 << 30}
        static = {"gpus": [{"index": 0, "bus": "0000:01:00.0"}, {"index": 1, "bus": "00000000:08:00.0"}]}
        hw = hardware_view(now, static, {"ram_experts_mib": 60 << 10}, cards, l3)
        self.assertEqual([(g["expert_slots"], g["h2d_gbps"]) for g in hw["gpus"]], [(9052, 52.1), (5497, None)])
        self.assertEqual([g["vram_parts"] for g in hw["gpus"]], [{"experts": 20 << 30, "kv": None, "other": 10 << 30},
                                                                 {"experts": 10 << 30, "kv": 0, "other": 2 << 30}])
        self.assertEqual(hw["cpu_ccd"], [20.0, 50.0])
        self.assertEqual(hw["ram_parts"], {"experts": 60 << 30, "page_cache": 20 << 30, "other": 40 << 30})
        self.assertNotIn("cpu_cores", hw)
        bare = hardware_view({"gpus": [{"index": 0}], "ram_used": 1, "ram_cached": None}, {}, {}, {}, [])
        self.assertEqual((bare["cpu_ccd"], bare["ram_parts"], bare["gpus"][0]["expert_slots"]), (None, None, None))


class Served(unittest.TestCase):
    def test_slots_and_metrics_carry_the_fields(self):
        tok = ByteTokenizer()
        eng = MockEngine(tok, "Hi.")
        eng.console = ConsoleLines()
        eng.console.feed("XSLOT 0 10 30 60 90 10 0 8/10")
        eng.console.feed("XMAP 4 0122 1,0,0,3")
        svc = Service(eng, tok, ChatTemplate.from_file(TEMPLATE))
        row = svc.slot_table()[0][0]
        for k in ("started", "path", "prompt_ms"):
            self.assertIn(k, row)
        self.assertEqual(row["accept_pos"], [0.8])
        self.assertEqual(row["hit"]["vram"], 0.6)
        self.assertEqual(svc.metrics()["experts_map"]["tier"], [0, 1, 2, 2])

    def test_a_running_slot_shows_its_start_path_and_read_time(self):
        tok = ByteTokenizer()
        seen = []

        class Peek(MockEngine):
            def prompt_progress(self):
                return {"read": 5, "total": 5, "ms": 31.5, "cache": 0}

            def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):
                for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
                    if i == 2:
                        seen.append(svc.slot_table()[0][0])
                    yield t
        svc = Service(Peek(tok, "Hello there, friend."), tok, ChatTemplate.from_file(TEMPLATE))
        svc.request_ctx.path = "/v1/messages"
        for _ in svc.run(tok.encode("hi"), False, None, 16, {}, threading.Event()):
            pass
        row = seen[0]
        self.assertEqual((row["path"], row["prompt_ms"]), ("/v1/messages", 31.5))
        self.assertEqual(row["reused"], 0)                                   # RESUME 0: nothing cached, a 0% hit
        self.assertIsNotNone(row["started"])

    def test_a_running_request_shows_only_its_own_drafts(self):
        tok = ByteTokenizer()
        seen = []

        class Peek(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):
                for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
                    if i == 2:
                        seen.append(svc.slot_table()[0][0])
                    yield t
        eng = Peek(tok, "Hello there, friend.")
        eng.console = ConsoleLines()
        eng.console.feed("XSLOT 0 9 20 0 0 0 0 7/9")                      # the slot's last request, before this one
        svc = Service(eng, tok, ChatTemplate.from_file(TEMPLATE))
        svc.slot_last[0] = {"ctx": 5, "offered": 9, "accepted": 7, "at": 0}
        for _ in svc.run(tok.encode("hi"), False, None, 16, {}, threading.Event()):
            pass
        self.assertEqual((seen[0]["drafts_offered"], seen[0]["accept_pos"]), (None, None))   # no window of its own yet

if __name__ == "__main__":
    unittest.main()
