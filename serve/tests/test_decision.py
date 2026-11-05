"""Finite-domain math, protocol, admission and resident scoring boundaries."""
from __future__ import annotations

import io
import json
import math
import queue
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.config import build_parser, parse_settings
from serve.decision import schema, systemone
from serve.decision.admission import Control, DecisionError, Gate
from serve.decision.engine import Decisions
from serve.decision.batches import BatchControl, Job
from serve.decision.trie import Trie, summary
from serve.engine.mock import MockEngine
from serve.engine.process import ProcessEngine
from serve.engine.errors import EngineError
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
REQUEST = {"contexts": ["charged twice"], "schema": {"urgent": {"type": "boolean", "description": "Needs action?"}}}


class Finite(unittest.TestCase):
    def test_fields_and_schema_wrappers(self):
        for name in ("urgent", "properties", "type", "required"):
            body = {**REQUEST, "schema": {name: REQUEST["schema"]["urgent"]}}
            self.assertEqual(schema.parse(body)[1][0].values, [True, False])
        self.assertEqual(schema.parse({**REQUEST, "schema": {"type": "object", "properties": REQUEST["schema"]}})[1][0].name, "urgent")
        self.assertEqual(schema.parse({**REQUEST, "schema": {"properties": {"enum": ["one"], "description": "n"}}})[1][0].name, "properties")

    def test_precise_number_grids(self):
        f = schema.field("n", {"type": "number", "description": "n", "minimum": -0.1, "maximum": 0.3, "step": 0.1})
        self.assertEqual(f.values, [-0.1, 0.0, 0.1, 0.2, 0.3])
        for lo, hi, step in ((9007199254740992, 9007199254740993, 1), (0, 254000000001, 1000000000), (0, 0.31, 0.1)):
            with self.subTest(lo=lo), self.assertRaises(ValueError):
                schema.field("n", {"type": "number", "description": "n", "minimum": lo, "maximum": hi, "step": step})
        f = schema.field("n", {"type": "number", "description": "n", "minimum": 1000000000000000.1,
                                "maximum": 1000000000000001.1, "step": 1})
        self.assertEqual(f.values, [1000000000000000.1, 1000000000000001.1])

    def test_unsupported_shapes_and_limits(self):
        for patch in ({"contexts": []}, {"contexts": [True]}, {"mode": "magic"}, {"tree_max": True},
                      {"cache_prompt": 1}, {"unknown": 1}, {"schema": {"x": {"type": "string"}}}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                schema.parse({**REQUEST, **patch})
        with self.assertRaises(ValueError):
            schema.field("n", {"type": "number", "description": "n", "minimum": 10 ** 1000, "maximum": 1, "step": 1})

    def test_compact_trie_probability_oracle(self):
        prefix = tuple(range(10000))
        trie = Trie(iter([prefix + (10001, 1, 0), prefix + (10001, 2, 0), prefix + (10002, 0)]))
        self.assertEqual(trie.prefix, prefix)
        self.assertLess(sum(len(p) for p in trie.paths), 12)
        ps = trie.distribution({0: [math.log(0.6), math.log(0.4)], 1: [math.log(0.1), math.log(0.9)]})
        for a, b in zip(ps, [0.06, 0.54, 0.4]):
            self.assertAlmostEqual(a, b)
        for paths in (([1, 2], [1, 2]), ([1, 2], [1, 2, 3])):
            with self.assertRaises(ValueError):
                Trie(paths)

    def test_numeric_aggregates(self):
        f = schema.field("n", {"type": "integer", "description": "n", "minimum": 0, "maximum": 2, "aggregate": "mean"})
        self.assertEqual(summary(f, [0.1, 0.2, 0.7], 1, True)["value"], 2)
        f.aggregate = "median"
        self.assertEqual(summary(f, [0.1, 0.7, 0.2], 1, True)["value"], 1)
        f.values, f.aggregate = [2 ** 63 - 3, 2 ** 63 - 2, 2 ** 63 - 1], "mean"
        self.assertEqual(summary(f, [0.1, 0.2, 0.7], 1, True)["value"], 2 ** 63 - 1)

    def test_systemone_noul_is_not_winning_probability(self):
        p = systemone.parse({"state": None, "questions": {"private": {"type": "noul"}}})
        result = {"results": [{"fields": {"private": {"probabilities": [0.1, 0.9]}}}], "usage": {"prompt_tokens": 13}}
        out = systemone.format_response(result, p[1], p[6], "model")
        self.assertEqual(out["answers"], {"private": {"type": "noul", "noul": 0.1, "confidence": 0.1}})
        self.assertEqual(out["usage"]["output_tokens"], 0)

    def test_config_does_not_raise_chat_parallelism(self):
        a, _, file_words, cli_words = parse_settings(build_parser(), ["--jev", "1"])
        self.assertEqual(a.jev, 1)
        self.assertFalse(a.parallel)
        self.assertEqual(file_words + cli_words, [])
        self.assertEqual(parse_settings(build_parser(), ["--jev", "0"])[0].jev, 0)
        self.assertEqual(parse_settings(build_parser(), [])[0].jev, 0)

    def test_json_opt_in_preserves_engine_settings(self):
        from serve.tests.test_config import conf
        engine = {"kv": "fp16", "expert-cache": "auto", "prefill": "auto", "spec": 7, "resident-experts": True}
        plain = conf({"server": {"parallel": 1}, "engine": engine})
        enabled = conf({"server": {"parallel": 1, "jev": 1}, "engine": engine})
        a, _, fw, cw = parse_settings(build_parser(), ["--config", plain])
        b, _, jfw, jcw = parse_settings(build_parser(), ["--config", enabled])
        self.assertEqual((fw, cw, a.parallel), (jfw, jcw, b.parallel))
        self.assertEqual(b.jev, 1)
        self.assertEqual(parse_settings(build_parser(), ["--config", enabled, "--jev", "0"])[0].jev, 0)
        with self.assertRaisesRegex(ValueError, 'expected a string or number'):
            parse_settings(build_parser(), ["--config", conf({"server": {"jev": True}})])
        with self.assertRaisesRegex(ValueError, 'server option'):
            parse_settings(build_parser(), ["--config", conf({"engine": {"jev": True}})])

    def test_systemone_score_and_choice_math(self):
        req = {"state": "text", "questions": {"q": {"type": "score", "criteria": ["low", "mid", "high"]}}}
        p = systemone.parse(req)
        result = {"results": [{"fields": {"q": {"probabilities": [0.1, 0.2, 0.7]}}}], "usage": {"prompt_tokens": 1}}
        answer = systemone.format_response(result, p[1], p[6], "model")["answers"]["q"]
        self.assertAlmostEqual(answer["score"], 1.6)
        self.assertAlmostEqual(answer["confidence"], 0.4)
        result["results"][0]["fields"]["q"]["probabilities"] = [float("nan"), 0, 1]
        with self.assertRaises(EngineError):
            systemone.format_response(result, p[1], p[6], "model")

    def test_systemone_box_sorts_its_four_grid_edges(self):
        req = {"state": "a screenshot", "images": ["shot.png"],
               "questions": {"cta": {"type": "box", "instructions": "Where is the primary button?"}}}
        p = systemone.parse(req)
        self.assertEqual([f.name for f in p[1]], [f"cta\u0000{edge}" for edge in systemone.EDGES])
        self.assertTrue(all(f.values[0] == 0.0 and f.values[-1] == 1.0 and len(f.values) == 101 for f in p[1]))
        self.assertEqual(p[7], ("shot.png",))

        def edge(at):
            return [1.0 if i == at else 0.0 for i in range(101)]

        # top is asked below bottom: the answer sorts each pair so the box is never inverted
        fields = {f.name: {"probabilities": edge(at)} for f, at in zip(p[1], (60, 20, 24, 74))}
        answer = systemone.format_response({"results": [{"fields": fields}], "usage": {"prompt_tokens": 9}},
                                           p[1], p[6], "model")["answers"]["cta"]
        self.assertEqual(answer["box"], {"top": 0.2, "bottom": 0.6, "left": 0.24, "right": 0.74})
        self.assertEqual(answer["confidence"], 1.0)                 # every edge is one grid point
        self.assertEqual(answer["aggregate"], "mode")
        self.assertEqual(answer["edges"]["top"],
                         {"probability": 1.0, "modal": 0.6, "mean": 0.6, "median": 0.6, "moved": True})
        self.assertFalse(answer["edges"]["left"]["moved"])
        fields[p[1][2].name]["probabilities"] = [0.5 if i in (20, 28) else 0.0 for i in range(101)]
        spread = systemone.format_response({"results": [{"fields": fields}], "usage": {"prompt_tokens": 9}},
                                           p[1], p[6], "model")["answers"]["cta"]
        self.assertLess(spread["confidence"], 1.0)                  # the least certain edge decides
        self.assertAlmostEqual(spread["edges"]["left"]["modal"], 0.2)

    def test_systemone_box_aggregate_choice(self):
        def ask(criteria):
            p = systemone.parse({"state": "s", "images": ["shot.png"],
                                 "questions": {"q": {"type": "box", "criteria": criteria}}})
            # one edge's mass: 0.6 on the peak, 0.4 two fifths of the way down
            fields = {f.name: {"probabilities": [0.6 if i == 60 else 0.4 if i == 20 else 0.0 for i in range(101)]}
                      for f in p[1]}
            return systemone.format_response({"results": [{"fields": fields}], "usage": {"prompt_tokens": 9}},
                                             p[1], p[6], "model")["answers"]["q"]

        peak, average, middle = ask(None), ask({"aggregate": "mean"}), ask({"aggregate": "median"})
        self.assertEqual((peak["aggregate"], peak["box"]["bottom"]), ("mode", 0.6))
        self.assertEqual((average["aggregate"], average["box"]["bottom"]), ("mean", 0.44))
        self.assertEqual((middle["aggregate"], middle["box"]["bottom"]), ("median", 0.6))
        self.assertEqual(average["edges"]["bottom"]["modal"], 0.6)  # the peak stays visible under an average
        for criteria in ({"aggregate": "middle"}, {"aggregate": "mode", "steps": 2}):
            with self.subTest(criteria=criteria), self.assertRaises(ValueError):
                systemone.parse({"state": "s", "images": ["shot.png"],
                                 "questions": {"q": {"type": "box", "criteria": criteria}}})

    def test_systemone_images_and_grid_are_validated(self):
        base = {"state": "s", "questions": {"q": {"type": "noul"}}}
        for patch in ({"images": "shot.png"}, {"images": [None]}, {"images": [""]}, {"images": ["p"] * 9},
                      {"box": 1}, {"questions": {"q": {"type": "box"}}}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                systemone.parse({**base, **patch})       # a box without a picture asks where nothing is
        self.assertEqual(systemone.parse({**base, "images": []})[7], ())
        for criteria in ({"step": 0.03}, {"step": 0.01, "unit": "px"}, {"step": "0.01"}, {"edge": 1}):
            with self.subTest(criteria=criteria), self.assertRaises(ValueError):
                systemone.parse({"state": "s", "images": ["shot.png"],
                                 "questions": {"q": {"type": "box", "criteria": criteria}}})


class Admission(unittest.TestCase):
    def test_bounded_admission_and_cleanup(self):
        gate = Gate(0)
        with gate.admit():
            with self.assertRaises(DecisionError) as got:
                with gate.admit():
                    pass
            self.assertEqual(got.exception.status, 429)
        self.assertEqual(gate.outstanding, 0)
        with gate.exclusive(Control(threading.Event(), 1)):
            self.assertTrue(gate.writing)
        self.assertFalse(gate.writing)

    def test_expired_writer_releases_waiting_readers(self):
        gate, cancel = Gate(1), threading.Event()
        with gate.shared(cancel):
            with self.assertRaises(DecisionError):
                with gate.exclusive(Control(cancel, 0.01)):
                    pass
        with gate.shared(cancel):
            self.assertEqual(gate.readers, 1)
        self.assertEqual((gate.readers, gate.writers), (0, 0))

    def test_active_chat_finishes_before_decision_and_failure_releases_gate(self):
        gate, cancel = Gate(1), threading.Event()
        acquired, finished = threading.Event(), threading.Event()
        def decide():
            try:
                with gate.exclusive(Control(cancel, 2)):
                    acquired.set()
                    raise RuntimeError("scorer failed")
            except RuntimeError:
                finished.set()
        with gate.shared(cancel):
            thread = threading.Thread(target=decide)
            thread.start()
            self.assertFalse(acquired.wait(0.05))
        self.assertTrue(finished.wait(2))
        thread.join()
        with gate.shared(cancel):
            self.assertFalse(gate.writing)

    def test_transactions_are_fifo_and_cancelled_waiters_leave_the_queue(self):
        gate = Gate(3)
        cancels = [threading.Event() for _ in range(3)]
        order, errors, threads = [], [], []
        def writer(i):
            try:
                with gate.exclusive(Control(cancels[i], 3)):
                    order.append(i)
            except DecisionError as error:
                errors.append((i, error.status))
        with gate.shared(threading.Event()):
            for i in range(3):
                thread = threading.Thread(target=writer, args=(i,))
                threads.append(thread)
                thread.start()
                with gate.cv:
                    self.assertTrue(gate.cv.wait_for(lambda: len(gate.waiters) == i + 1, 2))
            cancels[1].set()
            with gate.cv:
                self.assertTrue(gate.cv.wait_for(lambda: len(gate.waiters) == 2, 2))
        for thread in threads:
            thread.join(3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(order, [0, 2])
        self.assertEqual(errors, [(1, 499)])
        self.assertEqual(gate.waiters, [])
        self.assertEqual((gate.readers, gate.writers, gate.writing), (0, 0, False))


class Scorer(MockEngine):
    """Host oracle only: these deterministic scores never enter the production engine."""
    def __init__(self, tok):
        super().__init__(tok, "ok", max_context=16384)
        self.calls = []

    def score(self, ids, candidates, points, fresh, control, embeddings=None):
        control.check()
        self.calls.append((tuple(ids), list(candidates), list(points), fresh, embeddings))
        return [float(-i) for i in range(len(candidates))], {"reused": 0, "prompt_ms": 0.0}


class Waves(unittest.TestCase):
    def wire(self, lines, window=8):
        engine = ProcessEngine.__new__(ProcessEngine)
        engine.proc = mock.Mock()
        engine.proc.poll.return_value = None
        engine.info, engine.max_context, engine.batch = {"jev": 8, "jev_lanes": 8, "jev_window": window}, 4096, 0
        engine.lines, engine.wlock, engine.read_ms = queue.Queue(), threading.Lock(), 0.0
        for line in lines:
            engine.lines.put(line + "\n")
        return engine

    def test_two_suffixes_share_one_native_window(self):
        engine = self.wire(["JWAVE 2 1000", "JLOGITS 3 1", "JLOGITS 4 2", "JTIME 0.015250", "DONE 0 4 1.0 0.0 length 0 0 2"])
        rows = [([1, 2, 3, 4], [8, 9], [1, 3], False), ([1, 2, 3, 5], [6, 7], [1, 3], False)]
        values = engine.score_many(rows, Control(threading.Event(), 2))
        self.assertEqual([v for v, _ in values], [[3, 1], [4, 2]])
        self.assertEqual([s["reused"] for _, s in values], [2, 3])
        self.assertEqual(sum(s["scored_windows"] for _, s in values), 1)
        self.assertAlmostEqual(sum(s["decode_ms"] for _, s in values), 0.015250)
        self.assertEqual([s["batch_size"] for _, s in values], [2, 2])
        self.assertTrue(engine.lines.empty())
        sent = "".join(c.args[0] for c in engine.proc.stdin.write.call_args_list)
        self.assertEqual(sent.count("GEN 1"), 1)
        self.assertIn("score_branches=4:8,9/5:6,7", sent)

    def test_an_image_prompt_asks_through_the_image_verb(self):
        for case, embeddings in (("text", None), ("picture", "/tmp/basalt-vision/emb.bin")):
            with self.subTest(case=case):
                engine = self.wire(["JLOGITS 1.5 -0.5", "DONE 0 3 1.0 0.0 length 0 0 1"])
                values, stats = engine.score([1, 2, 3], [4, 5], [], False, Control(threading.Event(), 2),
                                             embeddings=embeddings)
                self.assertEqual(values, [1.5, -0.5])
                self.assertEqual(stats["generated"], 0)
                self.assertTrue(engine.lines.empty())
                sent = "".join(c.args[0] for c in engine.proc.stdin.write.call_args_list)
                self.assertIn(("GENI 1 score=4,5 /tmp/basalt-vision/emb.bin 1,2,3" if embeddings
                               else "GEN 1 score=4,5 1,2,3"), sent)
                self.assertEqual(sent.count("GEN 1 "), 0 if embeddings else 1)

    def test_a_wave_carries_its_prompt_picture(self):
        engine = self.wire(["JWAVE 2 1000", "JLOGITS 3 1", "JLOGITS 4 2", "JTIME 0.015250",
                            "DONE 0 4 1.0 0.0 length 0 0 2"])
        values = engine.score_many([([1, 2, 3, 4], [8, 9], [1, 3], False), ([1, 2, 3, 5], [6, 7], [1, 3], False)],
                                   Control(threading.Event(), 2), embeddings="/tmp/emb.bin")
        self.assertEqual([v for v, _ in values], [[3, 1], [4, 2]])
        sent = "".join(c.args[0] for c in engine.proc.stdin.write.call_args_list)
        self.assertEqual(sent.count("GENI 1"), 1)                   # one image prefill, two scored suffixes
        self.assertIn("score_branches=4:8,9/5:6,7 /tmp/emb.bin 1,2,3", sent)

    def test_memory_fallback_drains_before_serial_scoring(self):
        engine = self.wire(["JFALLBACK", "DONE 0 3 1.0 0 length 0 0 1",
                            "JLOGITS 1 2", "DONE 0 3 0 1 length 0 0 2",
                            "JLOGITS 3 4", "DONE 0 3 0 1 length 0 0 2"])
        rows = [((1, 2, 3), [4, 5], [], False), ((1, 2, 4), [4, 5], [], False)]
        values = engine.score_many(rows, Control(threading.Event(), 2))
        self.assertEqual([v for v, _ in values], [[1, 2], [3, 4]])
        self.assertTrue(all(s["batch_size"] == 1 for _, s in values))
        self.assertTrue(engine.lines.empty())

    def test_wave_planner_respects_suffix_sum_and_output_order(self):
        engine = self.wire([], window=3)
        calls = []
        def score(ids, options, points, fresh, control, _branches=None, embeddings=None):
            calls.append((ids, _branches))
            if _branches:
                self.assertLessEqual(sum(len(t) for t, _ in _branches), 3)
                return [[float(t[-1])] for t, _ in _branches], {"reused": 1, "workspace_bytes": 10}
            return [float(ids[-1])], {"reused": 0}
        engine.score = score
        ids = [[1, 2, 4], [1, 2, 3], [1, 9, 9, 9], [1, 2, 5]]
        result = engine.score_many([(p, [6], [], False) for p in ids], Control(threading.Event(), 2))
        self.assertEqual([x[0][0] for x in result], [4, 3, 9, 5])
        self.assertEqual(sum(x[1]["scored_windows"] for x in result), 2)

    def test_workspace_limit_retries_smaller_waves_and_keeps_prefill_accounting(self):
        engine = self.wire(["JFALLBACK", "JTIME 0.001", "DONE 0 3 9 0 length 0 0 1",
                            "JWAVE 2 100", "JLOGITS 3", "JLOGITS 4", "JTIME 0.015", "DONE 0 3 1 0 length 0 0 2",
                            "JWAVE 2 100", "JLOGITS 5", "JLOGITS 6", "JTIME 0.015", "DONE 0 3 1 0 length 0 0 2"])
        result = engine.score_many([([1, 2, i], [7], [], False) for i in range(3, 7)], Control(threading.Event(), 2))
        self.assertEqual([v[0] for v, _ in result], [3, 4, 5, 6])
        self.assertEqual([s["batch_size"] for _, s in result], [2, 2, 2, 2])
        self.assertEqual(sum(s["scored_windows"] for _, s in result), 2)
        self.assertEqual(sum(s["prompt_ms"] for _, s in result), 11)
        self.assertAlmostEqual(sum(s["decode_ms"] for _, s in result), 0.031)
        self.assertEqual(result[0][1]["reused"], 1)
        self.assertTrue(engine.lines.empty())

    def test_long_distinct_contexts_use_sequence_batches(self):
        engine = self.wire(["JWAVE 2 500000000", "JWINDOWS 100", "JLOGITS 3", "JLOGITS 4",
                            "JTIME 10", "DONE 0 2 1 0 length 0 0 1"], window=2)
        engine.info["jev_sequence"] = 1
        rows = [((1, 2) + (i,) * 100, [7], [2], False) for i in (3, 4)]
        result = engine.score_many(rows, Control(threading.Event(), 2))
        self.assertEqual([v[0] for v, _ in result], [3, 4])
        self.assertEqual([s["batch_size"] for _, s in result], [2, 2])
        self.assertEqual(sum(s["scored_windows"] for _, s in result), 100)
        sent = "".join(c.args[0] for c in engine.proc.stdin.write.call_args_list)
        self.assertEqual(sent.count("GEN 1"), 1)

    def test_physical_groups_publish_before_unrelated_scoring(self):
        engine = self.wire([], window=2)
        engine.info.update(jev_lanes=2, jev_sequence=1)
        published = []
        def score(ids, options, points, fresh, control, _branches=None, embeddings=None):
            if _branches and _branches[0][0][-1] == 5:
                self.assertEqual(published, [0, 1])
            return [[float(t[-1])] for t, _ in _branches], {"reused": 1}
        engine.score = score
        rows = [([1, i], [7], [], False) for i in range(3, 7)]
        result = engine.score_many(rows, Control(threading.Event(), 2), completed=lambda i, _: published.append(i))
        self.assertEqual(published, [0, 1, 2, 3])
        self.assertEqual([v[0] for v, _ in result], [3, 4, 5, 6])

    def test_completed_callers_do_not_prevent_cancel_of_remaining_work(self):
        controls = [Control(threading.Event(), 2) for _ in range(2)]
        jobs = [Job([None], control) for control in controls]
        batch = BatchControl(jobs)
        self.assertFalse(batch.is_set())
        jobs[0].done.set()
        controls[1].cancel.set()
        self.assertTrue(batch.is_set())


class Integration(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = Scorer(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.jev = Decisions(self.svc)

    def test_zero_output_and_refresh_only_first_row(self):
        req = {**REQUEST, "cache_prompt": False, "schema": {"a": {"type": "enum", "description": "Pick", "choices": ["aa", "ab", "b"]}}}
        out = self.svc.jev.run(req, threading.Event())
        self.assertGreater(len(self.engine.calls), 1)
        self.assertTrue(self.engine.calls[0][3])
        self.assertTrue(all(not c[3] for c in self.engine.calls[1:]))
        self.assertEqual(out["usage"]["prompt_tokens"], out["usage"]["computed_tokens"] + out["usage"]["cached_tokens"])
        self.assertEqual(out["timings"]["largest_batch"], 1)

    def test_memory_first_execution_reports_no_private_kv(self):
        self.engine.info = {"jev": 8, "kv": "fp16"}
        out = self.svc.jev.run(REQUEST, threading.Event())
        self.assertEqual(out["execution"], {"requested_lanes": 8, "effective_lanes": 1,
                                            "dedicated_kv_bytes": 0, "kv": "fp16"})
        self.assertEqual(self.engine.info, {"jev": 8, "kv": "fp16"})

    def test_trie_tuples_reach_native_sequence_planner(self):
        self.engine.info = {"jev": 2, "jev_lanes": 2, "jev_window": 7, "jev_sequence": 1}
        self.engine.score_many = ProcessEngine.score_many.__get__(self.engine)
        waves, original = [], self.engine.score
        def score(ids, options, points, fresh, control, _branches=None, embeddings=None):
            if not _branches:
                return original(ids, options, points, fresh, control)
            control.check()
            self.assertIsInstance(ids, tuple)
            self.assertTrue(all(isinstance(tokens, tuple) for tokens, _ in _branches))
            waves.append(_branches)
            return [[float(-i) for i in range(len(choices))] for _, choices in _branches], {"reused": 0}
        self.engine.score = score
        requests = [(False, {**REQUEST, "contexts": ["charged twice", "refund please"]}),
                    (True, {"state": "charged twice", "questions": {
                        name: {"type": "noul", "instructions": text}
                        for name, text in (("urgent", "Needs action?"), ("billing", "Is this billing?"))}})]
        for system_one, body in requests:
            with self.subTest(system_one=system_one):
                out = self.svc.jev.run(body, threading.Event(), system_one)
                self.assertEqual(out["timings"]["largest_batch"], 2)
        self.assertGreaterEqual(len(waves), 2)

    def test_identical_callers_score_and_complete_independently(self):
        entered = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        completed = [threading.Event(), threading.Event()]
        original = self.engine.score
        calls = 0
        def score(*args, **kw):
            nonlocal calls
            i, calls = calls, calls + 1
            entered[i].set()
            self.assertTrue(release[i].wait(3))
            return original(*args, **kw)
        self.engine.score = score
        output, errors = [], []
        def run(i):
            try:
                output.append(self.svc.jev.run(REQUEST, threading.Event()))
                completed[i].set()
            except Exception as exc:
                errors.append(exc)
        a, b = threading.Thread(target=run, args=(0,)), threading.Thread(target=run, args=(1,))
        try:
            a.start()
            self.assertTrue(entered[0].wait(2))
            b.start()
            batches = self.svc.jev.batches
            for _ in range(100):
                with batches.lock:
                    if batches.pending:
                        break
                threading.Event().wait(0.01)
            with batches.lock:
                self.assertEqual(len(batches.pending), 1)
            release[0].set()
            self.assertTrue(entered[1].wait(2))
            self.assertTrue(completed[0].wait(2))
            self.assertFalse(completed[1].is_set())
        finally:
            for event in release:
                event.set()
            for thread in (a, b):
                if thread.ident is not None:
                    thread.join(3)
                    self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(self.engine.calls), 2)
        self.assertEqual(len(output), 2)
        self.assertTrue(all(x["timings"]["deduplicated_rows"] == 0 for x in output))

    def test_waiting_requests_batch_without_a_completion_barrier(self):
        entered, release = threading.Event(), threading.Event()
        done = [threading.Event() for _ in range(3)]
        values, errors, sizes = [None] * 3, [], []
        def score_many(rows, control, completed=None, embeddings=None):
            sizes.append(len(rows))
            for i, row in enumerate(rows):
                if i == 2:
                    entered.set()
                    self.assertTrue(release.wait(3))
                completed(i, ([row[0][-1]], {"reused": 0}))
        self.engine.score_many = score_many
        def run(i):
            try:
                values[i] = self.svc.jev._score([([1, i], [7], [], False)], Control(threading.Event(), 3))
                done[i].set()
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
        try:
            with self.svc.jev.gate.shared(threading.Event()):
                threads[0].start()
                gate = self.svc.jev.gate
                with gate.cv:
                    self.assertTrue(gate.cv.wait_for(lambda: len(gate.waiters) == 1, 2))
                for thread in threads[1:]:
                    thread.start()
                for _ in range(100):
                    with self.svc.jev.batches.lock:
                        if len(self.svc.jev.batches.pending) == 2:
                            break
                    threading.Event().wait(0.01)
                with self.svc.jev.batches.lock:
                    self.assertEqual(len(self.svc.jev.batches.pending), 2)
            self.assertTrue(entered.wait(2))
            self.assertTrue(done[0].wait(2))
            self.assertTrue(done[1].wait(2))
            self.assertFalse(done[2].is_set())
        finally:
            release.set()
            for thread in threads:
                if thread.ident is not None:
                    thread.join(3)
                    self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(sizes, [3])
        self.assertEqual([row[0][0][0] for row in values], [0, 1, 2])

    def test_systemone_ids_never_enter_prompt(self):
        req = {"state": "new state", "questions": {"private-id": {"type": "choice", "instructions": "Pick", "criteria": {"A": "first", "B": "second"}}}}
        first = self.svc.jev.run(req, threading.Event(), True)
        tokens = [c[0] for c in self.engine.calls]
        self.engine.calls.clear()
        req["questions"]["other-id"] = req["questions"].pop("private-id")
        second = self.svc.jev.run(req, threading.Event(), True)
        self.assertEqual(tokens, [c[0] for c in self.engine.calls])
        self.assertEqual(first["answers"]["private-id"], second["answers"]["other-id"])

    def test_ready_row_limit_keeps_later_fields_and_contexts(self):
        req = {"contexts": [f"state {i}" for i in range(100)],
               "schema": {name: {"type": "boolean", "description": "Pick"} for name in ("a", "b", "c")}}
        out = self.svc.jev.run(req, threading.Event())
        self.assertEqual(len(out["results"]), 100)
        self.assertTrue(all(set(r["fields"]) == {"a", "b", "c"} for r in out["results"]))
        self.assertEqual(out["usage"]["scored_rows"], 300)
        self.assertEqual(out["timings"]["rounds"], 2)

    def test_http_routes_opt_in_and_auth(self):
        self.svc.api_key = "secret"
        httpd = serve(self.svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        origin = f"http://127.0.0.1:{httpd.server_address[1]}"
        for path in ("/decision", "/v1/decision", "/v1/systemone"):
            req = urllib.request.Request(origin + path, data=json.dumps(REQUEST).encode(), headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as got:
                urllib.request.urlopen(req)
            self.assertEqual(got.exception.code, 401)
        req.add_header("Authorization", "Bearer secret")
        req.full_url = origin + "/decision"
        with urllib.request.urlopen(req) as response:
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertEqual(json.load(response)["object"], "decision")
        self.svc.jev = None
        with self.assertRaises(urllib.error.HTTPError) as got:
            urllib.request.urlopen(req)
        self.assertEqual(got.exception.code, 404)

    def test_native_protocol_does_not_generate_or_leave_unread_lines(self):
        engine = ProcessEngine.__new__(ProcessEngine)
        engine.proc = mock.Mock()
        engine.proc.poll.return_value = None
        engine.info, engine.max_context, engine.batch = {"jev": 1}, 4096, 0
        engine.lines, engine.wlock, engine.read_ms = queue.Queue(), threading.Lock(), 0.0
        for _ in range(2):
            engine.lines.put("JLOGITS 1.25 -2.5\n")
            engine.lines.put("DONE 0 3 1.0 2.0 length 0 0 1\n")
        for fresh in (True, False):
            values, stats = engine.score([1, 2, 3], [4, 5], [1], fresh, Control(threading.Event(), 1))
            self.assertEqual(values, [1.25, -2.5])
            self.assertEqual(stats["generated"], 0)
        self.assertTrue(engine.lines.empty())
        sent = "".join(c.args[0] for c in engine.proc.stdin.write.call_args_list)
        self.assertEqual(sent.count("score_fresh=1"), 1)
        self.assertNotIn("GENT", sent)

    def test_native_cancellation_drains_done_and_preserves_process(self):
        cancel = threading.Event()
        class Lines(queue.Queue):
            def get(self, **kwargs):
                line = super().get(**kwargs)
                cancel.set()
                return line
        engine = ProcessEngine.__new__(ProcessEngine)
        engine.proc = mock.Mock()
        engine.proc.poll.return_value = None
        engine.info, engine.max_context, engine.batch = {"jev": 1}, 4096, 0
        engine.lines, engine.wlock, engine.read_ms = Lines(), threading.Lock(), 0.0
        engine.lines.put("PP 1 3\n")
        engine.lines.put("DONE 0 3 1.0 2.0 cancel 0 0 1\n")
        with self.assertRaises(DecisionError) as got:
            engine.score([1, 2, 3], [4, 5], [], False, Control(cancel, 1))
        self.assertEqual(got.exception.status, 499)
        engine.proc.kill.assert_not_called()
        self.assertTrue(engine.lines.empty())
        sent = "".join(c.args[0] for c in engine.proc.stdin.write.call_args_list)
        self.assertIn("STOP\n", sent)

    def test_single_question_rubric_stays_in_static_prefix(self):
        req = {"state": "first state", "questions": {"q": {"type": "noul", "instructions": "Is this urgent?"}}}
        self.svc.jev.run(req, threading.Event(), True)
        a = self.engine.calls[0]
        self.engine.calls.clear()
        req["state"] = "totally different state"
        self.svc.jev.run(req, threading.Event(), True)
        b = self.engine.calls[0]
        self.assertGreater(a[2][0], 0)
        self.assertEqual(a[0][:a[2][0]], b[0][:b[2][0]])

    def picture(self, counts=(3,), vision=True):
        """A service with a scripted vision encoder: real embeddings file, invented rows."""
        tok = ByteTokenizer()
        directory = tempfile.mkdtemp(prefix="basalt-jev-vision-")
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)

        class Eyes:
            calls, path = 0, None

            def read(self, sources, allowed=None):
                return [f"pixels:{s}".encode() for s in sources]      # what a real encoder reads back per source

            def encode(self, images):
                self.calls += 1
                self.path = Path(directory) / f"emb-{self.calls}.bin"
                self.path.write_bytes(b"rows")
                return self.path, list(counts) * len(images)

        eyes = Eyes()
        svc = Service(Scorer(tok), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                      vision=eyes if vision else None)
        svc.jev = Decisions(svc)
        return svc, tok, eyes

    def test_a_picture_is_encoded_once_for_every_scored_row(self):
        svc, tok, eyes = self.picture()
        pad = tok.encode("<|image_pad|>", parse_special=True)[0]
        request = {"state": "a settings screen", "images": ["settings.png"],
                   "questions": {"warn": {"type": "noul", "instructions": "Is a warning visible?"},
                                 "cta": {"type": "box", "instructions": "Where is the primary button?"}}}
        out = svc.jev.run(request, threading.Event(), True)
        self.assertEqual(set(out["answers"]), {"warn", "cta"})
        self.assertEqual(out["answers"]["warn"]["type"], "noul")
        self.assertEqual(set(out["answers"]["cta"]["box"]), set(systemone.EDGES))
        box = out["answers"]["cta"]["box"]
        self.assertTrue(box["top"] <= box["bottom"] and box["left"] <= box["right"])
        self.assertEqual(out["usage"]["output_tokens"], 0)
        self.assertEqual(eyes.calls, 1)                             # one encode, whatever the row count
        self.assertEqual({call[4] for call in svc.engine.calls}, {eyes.path})
        self.assertTrue(all(call[0].count(pad) == 3 for call in svc.engine.calls))
        self.assertIsNone(svc.embeddings.path)                      # the request took its embeddings with it
        self.assertFalse(eyes.path.exists())

    def test_a_picture_stays_in_the_reused_prefix(self):
        svc, tok, eyes = self.picture()
        pad = tok.encode("<|image_pad|>", parse_special=True)[0]
        request = {"state": "a settings screen", "images": ["settings.png"],
                   "questions": {"warn": {"type": "noul", "instructions": "Is a warning visible?"}}}
        svc.jev.run(request, threading.Event(), True)
        a, svc.engine.calls = svc.engine.calls[0], []
        request["state"] = "a checkout screen"
        svc.jev.run(request, threading.Event(), True)
        b = svc.engine.calls[0]
        self.assertGreater(a[2][0], 0)
        self.assertEqual(a[0][:a[2][0]], b[0][:b[2][0]])
        self.assertEqual(a[0][:a[2][0]].count(pad), 3)              # the picture is what the next ask reuses
        self.assertEqual(eyes.calls, 2)                             # a new state re-encodes only its own picture

    def test_a_picture_asked_without_a_vision_encoder_is_refused(self):
        svc, _, _ = self.picture(vision=False)
        with self.assertRaises(ValueError) as got:
            svc.jev.run({"state": "s", "images": ["shot.png"], "questions": {"q": {"type": "noul"}}},
                        threading.Event(), True)
        self.assertIn("without image input", str(got.exception))

    def test_two_pictures_never_share_one_scoring_transaction(self):
        sizes, seen, errors = [], [], []

        def score_many(rows, control, completed=None, embeddings=None):
            sizes.append(len(rows))
            seen.append(embeddings)
            for i, row in enumerate(rows):
                completed(i, ([row[0][-1]], {"reused": 0}))

        self.engine.score_many = score_many

        def run(path):
            try:
                self.svc.jev._score([([1, 2], [7], [], False)], Control(threading.Event(), 3), path)
            except Exception as error:                              # noqa: BLE001 - reported below
                errors.append(error)

        threads = [threading.Thread(target=run, args=(path,)) for path in ("/tmp/a.bin", "/tmp/b.bin")]
        try:
            with self.svc.jev.gate.shared(threading.Event()):        # hold the lane: both requests queue
                for thread in threads:
                    thread.start()
                for _ in range(200):
                    with self.svc.jev.batches.lock:
                        if len(self.svc.jev.batches.pending) == 1:   # the second request waits behind the first
                            break
                    threading.Event().wait(0.01)
                self.assertEqual(len(self.svc.jev.batches.pending), 1)
        finally:
            for thread in threads:
                if thread.ident is not None:
                    thread.join(3)
        self.assertEqual(errors, [])
        self.assertEqual(sizes, [1, 1])                             # a transaction scores one picture only
        self.assertEqual(seen, ["/tmp/a.bin", "/tmp/b.bin"])


if __name__ == "__main__":
    unittest.main()
