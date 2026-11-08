"""Opt-in native counters observe the protocol without changing generated tokens."""
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from serve.config import build_parser
from serve.engine.process import ProcessEngine
from serve.tests.test_parallel import ParallelService


class NativeCounters(unittest.TestCase):
    def engine(self, enabled=True):
        e = ProcessEngine('missing-executable', [], lazy=True)
        e.native_slot_metrics = enabled
        e._native_slots = [None, None]
        return e

    def test_default_off_and_parser_opt_in(self):
        self.assertFalse(build_parser().parse_args([]).native_slot_metrics)
        self.assertTrue(build_parser().parse_args(['--native-slot-metrics']).native_slot_metrics)
        e = self.engine(False)
        e._native_admission(0, 20)
        e._native_line('BT 0 1 2')
        self.assertEqual(e.native_slots_view(), [])

    def test_independent_slots_and_readmission(self):
        e = self.engine()
        e._native_admission(0, 20)
        e._native_admission(1, 30)
        e._native_line('BT 0 11 12 13')
        e._native_line('BT 1 14')
        view = e.native_slots_view()
        self.assertEqual([s['generated'] for s in view], [3, 1])
        view[0]['generated'] = 1000
        self.assertEqual(e.native_slots_view()[0]['generated'], 3)
        e._native_line('BDONE 0 4 stop 1.0')
        self.assertEqual(e.native_slots_view()[0]['state'], 'idle')
        e._native_admission(0, 40)
        self.assertEqual(e.native_slots_view()[0]['generated'], 0)

    def test_malformed_and_unadmitted_lines_do_not_progress(self):
        e = self.engine()
        e._native_admission(None, 20)
        e._native_admission(0, 20)
        for line in ['BT 8 3', 'BT 1 3', 'BT -1 3', 'BT x 3', 'BT 0 x', 'BT 0', 'BDONE 0', 'T 8']:
            e._native_line(line)
        self.assertEqual(e.native_slots_view()[0]['generated'], 0)


class NativeProtocol(ParallelService):
    # Only run this focused integration, rather than inherit all protocol tests twice.
    def test_native_metrics_preserve_replies(self):
        self.start(2)
        self.engine.native_slot_metrics = True
        barrier = threading.Barrier(2)
        def one(i):
            barrier.wait()
            return self.chat(f'question {i} LONGREPLY')
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(one, range(2)))
        self.assertTrue(all(r['choices'][0]['message']['content'] == 'ok, ' + 'la ' * 15 + 'done.' for r in results))
        live = self.get('/metrics')['live']
        self.assertEqual(live['native_slot_source'], 'engine BT/BDONE protocol')
        self.assertTrue(any(s['generated'] > 0 for s in live['native_slots']))
        self.engine.native_slot_metrics = False
        self.assertNotIn('native_slots', self.get('/metrics')['live'])
        self.assertEqual(self.chat('hi')['choices'][0]['message']['content'], 'ok, done.')


# Reuse the existing fixture helpers without rediscovering its tests in this module.
for _name in tuple(vars(NativeProtocol.__bases__[0])):
    if _name.startswith('test_') and _name != 'test_native_metrics_preserve_replies':
        setattr(NativeProtocol, _name, None)
del ParallelService
