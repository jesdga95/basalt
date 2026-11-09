"""AgentScheduler: bypass, bounds, fairness, aging, interleave, cancel, deadline, pressure, threads."""
import asyncio
import collections
import json
import threading
import time
import unittest

from serve.scheduler import (AdmissionCancelled, AdmissionExpired, AdmissionRejected, AgentScheduler, Request,
                             SchedulerConfig)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def sched(clock=None, **kw):
    kw.setdefault("background", clock is None)
    pressure = kw.pop("pressure", None)
    return AgentScheduler(SchedulerConfig(enabled=True, **kw), pressure=pressure, clock=clock or time.monotonic)


def drain(s, holder, grants):
    """Release `holder`, then keep releasing whichever queued grant is admitted; returns the admission order."""
    order, cur = [], holder
    pending = list(grants)
    while True:
        cur.release()
        nxt = next((g for g in pending if g.done()), None)
        if nxt is None:
            return order
        pending.remove(nxt)
        order.append(nxt)
        cur = nxt


class Basics(unittest.TestCase):
    def test_disabled_is_noop(self):
        s = AgentScheduler()
        g = s.submit(Request("t", "a", 10))
        self.assertTrue(g.wait(0))
        g.release()
        g.cancel()
        self.assertEqual(s.metrics(), {"enabled": False})
        self.assertIs(s.submit(Request("t", "b", 5)), g)       # shared no-op grant, nothing accounted

    def test_solo_bypass_then_queue(self):
        s = sched(max_inflight=1)
        a = s.submit(Request("t", "a", 100))
        self.assertTrue(a.done() and a.bypass)
        b = s.submit(Request("t", "b", 100))
        self.assertFalse(b.done())
        self.assertEqual(s.metrics()["queue_depth"], 1)
        a.release()
        self.assertTrue(b.wait(1))
        b.release()
        m = s.metrics()
        self.assertEqual((m["bypassed"], m["granted"], m["inflight"], m["queue_depth"]), (1, 2, 0, 0))
        c = s.submit(Request("t", "a", 1))                      # idle again: solo again
        self.assertTrue(c.bypass)

    def test_solo_ignores_pressure_probe(self):
        calls = []
        s = sched(max_inflight=2, pressure=lambda: calls.append(1) or True)
        g = s.submit(Request("t", "a", 1))
        self.assertTrue(g.done())
        self.assertEqual(calls, [])                              # idle: no probe, no hold

    def test_context_manager_releases(self):
        s = sched(max_inflight=1)
        with s.submit(Request("t", "a", 1)):
            self.assertEqual(s.metrics()["inflight"], 1)
        self.assertEqual(s.metrics()["inflight"], 0)

    def test_negative_cost_and_bad_kind(self):
        with self.assertRaises(ValueError):
            Request("t", "a", -1)
        with self.assertRaises(ValueError):
            Request("t", "a", 1, kind="embed")

    def test_unlimited_slots_admit_all(self):
        s = sched(max_inflight=0)
        gs = [s.submit(Request("t", f"a{i}", 5)) for i in range(20)]
        self.assertTrue(all(g.done() for g in gs))
        self.assertEqual(s.metrics()["inflight"], 20)

    def test_metrics_json_and_percentiles(self):
        clk = Clock()
        s = sched(clk, max_inflight=1)
        h = s.submit(Request("t", "a", 1))
        qs = [s.submit(Request("t", "a", 1)) for _ in range(4)]
        for i, q in enumerate(qs):
            clk.t += 0.1
            h.release()
            self.assertTrue(q.done())
            h = q
        h.release()
        m = s.metrics()
        json.dumps(m)
        w = m["wait_ms"]["prefill"]
        self.assertEqual(w["n"], 4)
        self.assertAlmostEqual(w["p50"], 200.0, delta=1)        # waits 100,200,300,400 ms
        self.assertAlmostEqual(w["p95"], 400.0, delta=1)
        self.assertEqual(m["ttft_wait_ms"]["p95"], w["p95"])
        self.assertEqual(m["tenants"]["t"]["granted"], 5)


class Bounds(unittest.TestCase):
    def test_agent_tenant_total_rejection(self):
        s = sched(max_inflight=1, max_queue_per_agent=2, max_queue_per_tenant=3, max_queue_total=4)
        s.submit(Request("t", "hold", 1))
        s.submit(Request("t", "a", 1)); s.submit(Request("t", "a", 1))
        with self.assertRaises(AdmissionRejected) as e:
            s.submit(Request("t", "a", 1))
        self.assertEqual(e.exception.reason, "queue_full_agent")
        s.submit(Request("t", "b", 1))
        with self.assertRaises(AdmissionRejected) as e:
            s.submit(Request("t", "c", 1))
        self.assertEqual(e.exception.reason, "queue_full_tenant")
        s.submit(Request("u", "a", 1))
        with self.assertRaises(AdmissionRejected) as e:
            s.submit(Request("v", "a", 1))
        self.assertEqual(e.exception.reason, "queue_full_total")
        m = s.metrics()
        self.assertEqual(m["rejected"], 3)
        self.assertEqual(m["rejected_by_reason"]["queue_full_total"], 1)
        self.assertEqual(m["queue_depth"], 4)

    def test_bound_counts_both_kinds(self):
        s = sched(max_inflight=1, max_queue_per_agent=1)
        s.submit(Request("t", "hold", 1))
        s.submit(Request("t", "a", 1, kind="prefill"))
        with self.assertRaises(AdmissionRejected):
            s.submit(Request("t", "a", 1, kind="decode"))

    def test_closed_rejects_and_cancels(self):
        s = sched(max_inflight=1)
        s.submit(Request("t", "hold", 1))
        q = s.submit(Request("t", "a", 1))
        s.close()
        with self.assertRaises(AdmissionCancelled):
            q.wait(0.1)
        with self.assertRaises(AdmissionRejected) as e:
            s.submit(Request("t", "a", 1))
        self.assertEqual(e.exception.reason, "closed")


class Fairness(unittest.TestCase):
    def test_tenant_round_robin_not_fifo(self):
        s = sched(max_inflight=1)
        hold = s.submit(Request("hog", "h", 1))
        hog = [s.submit(Request("hog", "h", 100)) for _ in range(30)]
        mice = [s.submit(Request("mouse", "m", 100)) for _ in range(5)]
        order = drain(s, hold, hog + mice)
        first = [g.request.tenant for g in order[:12]]
        self.assertEqual(first.count("mouse"), 5, first)         # FIFO would put all five after 30 hogs
        self.assertEqual(len(order), 35)

    def test_weights(self):
        s = sched(max_inflight=1, quantum_tokens=100, max_queue_per_agent=100)
        s.set_weight("big", 3.0)
        hold = s.submit(Request("big", "x", 1))
        gs = [s.submit(Request(t, "x", 100)) for _ in range(40) for t in ("big", "small")]
        order = drain(s, hold, gs)
        c = collections.Counter(g.request.tenant for g in order[:40])
        self.assertAlmostEqual(c["big"] / c["small"], 3.0, delta=0.6, msg=str(c))

    def test_agents_within_tenant_share(self):
        s = sched(max_inflight=1, quantum_tokens=100)
        hold = s.submit(Request("t", "x", 1))
        gs = [s.submit(Request("t", "a", 100)) for _ in range(20)] + [s.submit(Request("t", "b", 100)) for _ in range(4)]
        order = drain(s, hold, gs)
        self.assertEqual([g.request.agent for g in order[:10]].count("b"), 4)

    def test_cost_is_tokens_not_requests(self):
        s = sched(max_inflight=1, quantum_tokens=500)
        hold = s.submit(Request("t0", "x", 1))
        gs = []
        for _ in range(30):
            gs.append(s.submit(Request("long", "x", 2000)))
            gs.extend(s.submit(Request("short", "x", 200)) for _ in range(1))
        order = drain(s, hold, gs)
        tok = collections.Counter()
        for g in order[:24]:
            tok[g.request.tenant] += g.cost
        self.assertLess(abs(tok["long"] - tok["short"]), 4000, tok)  # about equal token service

    def test_idle_flow_resets_deficit(self):
        s = sched(max_inflight=1)
        hold = s.submit(Request("a", "x", 1))
        q = s.submit(Request("a", "x", 5000))
        drain(s, hold, [q])
        self.assertEqual(s._tiers["prefill"].tenants, {})        # no leftover flow state


class Aging(unittest.TestCase):
    def test_aged_request_goes_first(self):
        clk = Clock()
        s = sched(clk, max_inflight=1, aging_s=5.0, quantum_tokens=10)
        s.set_weight("slow", 0.001)                              # DRR alone would serve it last
        hold = s.submit(Request("fast", "x", 1))
        victim = s.submit(Request("slow", "x", 1000))
        clk.t += 6.0                                             # victim is aged; the rest are fresh
        fresh = [s.submit(Request("fast", "x", 1000)) for _ in range(10)]
        hold.release()
        self.assertTrue(victim.done())
        self.assertFalse(any(g.done() for g in fresh))

    def test_without_aging_weight_starves_longer(self):
        clk = Clock()
        s = sched(clk, max_inflight=1, aging_s=1e9, quantum_tokens=10)
        s.set_weight("slow", 0.001)
        hold = s.submit(Request("fast", "x", 1))
        victim = s.submit(Request("slow", "x", 1000))
        fresh = [s.submit(Request("fast", "x", 1000)) for _ in range(10)]
        order = drain(s, hold, [victim] + fresh)
        self.assertIs(order[-1], victim)                         # control: aging is what changes this


class Interleave(unittest.TestCase):
    def test_prefill_not_starved_by_decode_flood(self):
        s = sched(max_inflight=1, round_tokens={"prefill": 100, "decode": 100}, aging_s=1e9)
        hold = s.submit(Request("t", "x", 1, kind="decode"))
        dec = [s.submit(Request("d", f"a{i % 4}", 50, kind="decode")) for i in range(40)]
        pre = s.submit(Request("p", "x", 80, kind="prefill"))
        order = drain(s, hold, dec + [pre])
        self.assertLess(order.index(pre), 6, [g.kind for g in order[:8]])

    def test_decode_not_starved_by_prefill_flood(self):
        s = sched(max_inflight=1, round_tokens={"prefill": 1000, "decode": 100}, aging_s=1e9)
        hold = s.submit(Request("t", "x", 1))
        pre = [s.submit(Request("p", f"a{i % 4}", 400, kind="prefill")) for i in range(40)]
        dec = s.submit(Request("d", "x", 50, kind="decode"))
        order = drain(s, hold, pre + [dec])
        self.assertLess(order.index(dec), 5)

    def test_kind_share_follows_budgets(self):
        s = sched(max_inflight=1, round_tokens={"prefill": 400, "decode": 100}, aging_s=1e9)
        hold = s.submit(Request("t", "x", 1))
        gs = [s.submit(Request("p", f"a{i % 5}", 100, kind="prefill")) for i in range(60)]
        gs += [s.submit(Request("d", f"a{i % 5}", 100, kind="decode")) for i in range(60)]
        order = drain(s, hold, gs)
        c = collections.Counter(g.kind for g in order[:50])
        self.assertAlmostEqual(c["prefill"] / c["decode"], 4.0, delta=1.0, msg=str(c))

    def test_inflight_token_budget(self):
        s = sched(max_inflight=0, prefill_inflight_tokens=1000)
        a = s.submit(Request("t", "a", 600))
        b = s.submit(Request("t", "b", 600))
        self.assertTrue(a.done() and not b.done())
        d = s.submit(Request("t", "c", 600, kind="decode"))      # the other kind has its own budget
        self.assertTrue(d.done())
        a.release()
        self.assertTrue(b.done())

    def test_oversized_request_runs_alone_and_blocks_followers(self):
        s = sched(max_inflight=0, prefill_inflight_tokens=1000, aging_s=1e9)
        a = s.submit(Request("t", "a", 100))
        big = s.submit(Request("t", "b", 5000))
        small = s.submit(Request("u", "c", 100))
        self.assertFalse(big.done())
        a.release()
        self.assertTrue(big.done())
        self.assertFalse(small.done())                           # not past a big one that is next in line
        big.release()
        self.assertTrue(small.done())


class Invariants(unittest.TestCase):
    def test_random_ops_keep_accounting_and_drain(self):
        import random
        rnd = random.Random(7)
        clk = Clock()
        s = sched(clk, max_inflight=3, prefill_inflight_tokens=3000, decode_inflight_tokens=500, aging_s=2.0,
                  max_queue_per_agent=6, max_queue_per_tenant=20, max_queue_total=40)
        live, queued = [], []
        for _ in range(4000):
            clk.t += rnd.random() * 0.2
            op = rnd.random()
            if op < 0.45:
                try:
                    r = Request(f"t{rnd.randrange(4)}", f"a{rnd.randrange(5)}", rnd.choice([1, 50, 800, 5000]),
                                rnd.choice(["prefill", "decode"]),
                                clk.t + rnd.uniform(0.5, 6) if rnd.random() < 0.3 else None)
                    queued.append(s.submit(r))
                except AdmissionRejected:
                    pass
            elif op < 0.75 and live:
                live.pop(rnd.randrange(len(live))).release()
            elif op < 0.85 and queued:
                queued.pop(rnd.randrange(len(queued))).cancel()
            s.poll()
            for g in list(queued):
                if g.state == "granted":
                    queued.remove(g); live.append(g)
                elif g.done():
                    queued.remove(g)
            m = s.metrics()
            self.assertLessEqual(m["inflight"], 3)
            self.assertEqual(m["inflight"], len(live))
            self.assertEqual(m["queue_depth"], len(queued))
            self.assertEqual(sum(m["queue_depth_by_kind"].values()), m["queue_depth"])
            self.assertEqual(sum(v["queued"] for v in m["tenants"].values()), m["queue_depth"])
        while live or queued:                                   # everything drains: nothing is stuck
            clk.t += 0.1
            if live:
                live.pop().release()
            s.poll()
            for g in list(queued):
                if g.state == "granted":
                    queued.remove(g); live.append(g)
                elif g.done():
                    queued.remove(g)
        m = s.metrics()
        self.assertEqual((m["inflight"], m["queue_depth"], m["inflight_tokens"]), (0, 0, {"decode": 0, "prefill": 0}))


class CancelDeadline(unittest.TestCase):
    def test_cancel_queued(self):
        s = sched(max_inflight=1)
        hold = s.submit(Request("t", "a", 1))
        q = s.submit(Request("t", "b", 1))
        other = s.submit(Request("t", "c", 1))
        err = []
        th = threading.Thread(target=lambda: err.append(self._catch(q)))
        th.start()
        time.sleep(0.05)
        self.assertTrue(q.cancel())
        th.join(2)
        self.assertEqual(err, ["cancelled"])
        hold.release()
        self.assertTrue(other.done())                            # the cancelled one left no hole
        other.release()
        m = s.metrics()
        self.assertEqual((m["cancelled"], m["inflight"], m["queue_depth"]), (1, 0, 0))

    @staticmethod
    def _catch(g):
        try:
            g.wait(5)
        except AdmissionCancelled:
            return "cancelled"
        return "ok"

    def test_cancel_granted_releases(self):
        s = sched(max_inflight=1)
        g = s.submit(Request("t", "a", 1))
        self.assertFalse(g.cancel())
        self.assertEqual(s.metrics()["inflight"], 0)

    def test_release_idempotent(self):
        s = sched(max_inflight=1)
        g = s.submit(Request("t", "a", 1))
        g.release(); g.release()
        self.assertEqual(s.metrics()["inflight"], 0)

    def test_deadline_expires_in_background(self):
        s = sched(max_inflight=1)
        hold = s.submit(Request("t", "a", 1))
        q = s.submit(Request("t", "b", 1, deadline=time.monotonic() + 0.1))
        t0 = time.monotonic()
        with self.assertRaises(AdmissionExpired):
            q.wait(3)
        self.assertLess(time.monotonic() - t0, 1.0)
        hold.release()
        m = s.metrics()
        self.assertEqual((m["expired"], m["queue_depth"], m["inflight"]), (1, 0, 0))

    def test_deadline_poll_with_fake_clock(self):
        clk = Clock()
        s = sched(clk, max_inflight=1)
        hold = s.submit(Request("t", "a", 1))
        q = s.submit(Request("t", "b", 1, deadline=clk.t + 2))
        live = s.submit(Request("t", "c", 1, deadline=clk.t + 10))
        clk.t += 3
        s.poll()
        self.assertTrue(q.done() and q.state == "expired")
        self.assertFalse(live.done())
        hold.release()
        self.assertTrue(live.done())

    def test_already_expired_at_submit(self):
        s = sched(max_inflight=1)
        g = s.submit(Request("t", "a", 1, deadline=time.monotonic() - 1))
        with self.assertRaises(AdmissionExpired):
            g.wait(0)
        self.assertEqual(s.metrics()["inflight"], 0)

    def test_granted_before_deadline_not_expired(self):
        s = sched(max_inflight=1)
        hold = s.submit(Request("t", "a", 1))
        q = s.submit(Request("t", "b", 1, deadline=time.monotonic() + 0.3))
        hold.release()
        time.sleep(0.5)
        self.assertEqual(q.state, "granted")
        self.assertEqual(s.metrics()["expired"], 0)


class Pressure(unittest.TestCase):
    def test_hold_and_resume(self):
        flag = {"p": False}
        s = sched(max_inflight=4, pressure=lambda: flag["p"], pressure_poll_s=0.01)
        a = s.submit(Request("t", "a", 1))
        flag["p"] = True
        b = s.submit(Request("t", "b", 1))
        self.assertFalse(b.wait(0.2))
        self.assertTrue(s.metrics()["paused"])
        flag["p"] = False
        self.assertTrue(b.wait(2))                               # the timer thread re-probes
        a.release(); b.release()

    def test_drained_scheduler_always_progresses(self):
        s = sched(max_inflight=4, pressure=lambda: True, pressure_poll_s=0.01)
        a = s.submit(Request("t", "a", 1))
        b = s.submit(Request("t", "b", 1))
        self.assertFalse(b.done())
        a.release()
        self.assertTrue(b.wait(1))                               # nothing in flight to drain -> admit anyway

    def test_broken_probe_fails_open(self):
        def boom():
            raise RuntimeError("nvml")
        s = sched(max_inflight=4, pressure=boom)
        s.submit(Request("t", "a", 1))
        b = s.submit(Request("t", "b", 1))
        self.assertTrue(b.wait(1))
        self.assertGreaterEqual(s.metrics()["pressure_errors"], 1)


class Concurrency(unittest.TestCase):
    def test_threads_never_exceed_slots_and_leak_nothing(self):
        s = sched(max_inflight=3, aging_s=0.5)
        live, peak, lock, done = 0, 0, threading.Lock(), []

        def worker(n):
            nonlocal live, peak
            for i in range(60):
                kind = "decode" if i % 3 == 0 else "prefill"
                g = s.submit(Request(f"t{n % 3}", f"a{n}", 10 + i, kind=kind, deadline=time.monotonic() + 30))
                g.wait(30)
                with lock:
                    live += 1
                    peak = max(peak, live)
                time.sleep(0.0005)
                with lock:
                    live -= 1
                g.release()
                done.append(1)
        ts = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        [t.start() for t in ts]
        [t.join(60) for t in ts]
        self.assertEqual(len(done), 480)
        self.assertLessEqual(peak, 3)
        m = s.metrics()
        self.assertEqual((m["inflight"], m["queue_depth"], m["inflight_tokens"]),
                         (0, 0, {"decode": 0, "prefill": 0}))
        self.assertEqual(m["granted"], 480)

    def test_await_grant(self):
        s = sched(max_inflight=1)
        hold = s.submit(Request("t", "a", 1))

        async def main():
            q = s.submit(Request("t", "b", 1))
            asyncio.get_running_loop().call_later(0.05, hold.release)
            g = await q
            self.assertIs(g, q)
            q.release()
            c = s.submit(Request("t", "c", 1))
            d = s.submit(Request("t", "d", 1))
            task = asyncio.ensure_future(d._await())
            await asyncio.sleep(0.02)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(d.state, "cancelled")
            c.release()
        asyncio.run(main())
        self.assertEqual(s.metrics()["inflight"], 0)

    def test_solo_overhead_small(self):
        s = sched(max_inflight=4)
        n = 20000
        t0 = time.perf_counter()
        for _ in range(n):
            s.submit(Request("t", "a", 10)).release()
        us = (time.perf_counter() - t0) / n * 1e6
        print(f"\nsolo submit+release: {us:.1f} us")
        self.assertLess(us, 100)
        self.assertEqual(s.metrics()["bypassed"], n)


if __name__ == "__main__":
    unittest.main()
