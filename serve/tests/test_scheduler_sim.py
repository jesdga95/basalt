"""Mixed short/long load on a virtual clock: AgentScheduler against a plain FIFO with the same slots."""
import collections
import heapq
import unittest

from serve.scheduler import AdmissionRejected, AgentScheduler, Request, SchedulerConfig


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


class _H:
    def __init__(self, fifo, req, now):
        self.fifo, self.request, self.t_submit, self.t_grant = fifo, req, now, None

    def done(self):
        return self.t_grant is not None

    def release(self):
        self.fifo.free += 1
        self.fifo.pump()


class Fifo:
    """Baseline: first come first served on `slots` slots, no fairness."""

    def __init__(self, clk, slots):
        self.clk, self.free, self.q = clk, slots, collections.deque()

    def submit(self, req):
        h = _H(self, req, self.clk.t)
        self.q.append(h)
        self.pump()
        return h

    def pump(self):
        while self.free and self.q:
            self.free -= 1
            h = self.q.popleft()
            h.t_grant = self.clk.t


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else 0.0


def simulate(make, workload, horizon, service):
    """workload(arrive, finished) seeds events; service(req) -> seconds the grant is held."""
    clk = Clock()
    sys_ = make(clk)
    ev, seq = [], iter(range(10 ** 9))
    pending, waits, rejected = [], collections.defaultdict(list), collections.Counter()

    def push(t, kind, payload):
        heapq.heappush(ev, (t, next(seq), kind, payload))

    def submit(req, tag):
        try:
            h = sys_.submit(req)
        except AdmissionRejected:
            rejected[tag] += 1
            return
        pending.append((h, tag))

    def settle():
        for item in list(pending):
            h, tag = item
            if h.done():
                pending.remove(item)
                t_grant = h.t_grant
                waits[tag].append(t_grant - h.t_submit)
                push(t_grant + service(h.request), "finish", (h, tag))

    workload(push, submit)
    done = collections.Counter()
    while ev:
        t, _, kind, payload = heapq.heappop(ev)
        if t > horizon:
            break
        clk.t = t
        if kind == "arrive":
            payload()
        else:
            h, tag = payload
            h.release()
            done[tag] += 1
            if tag in closed_loop:
                closed_loop[tag](h)
        settle()
    return waits, done, rejected, sys_


closed_loop = {}


def build_scenario(kind_short="prefill"):
    """12 closed-loop bulk jobs (3 agents x 4, 16k-token prefills) + 8 chat agents sending a short request every 0.5 s."""
    closed_loop.clear()

    def workload(push, submit):
        def bulk(agent):
            return Request("bulk", agent, 16000, "prefill")
        for a in range(3):
            for _ in range(4):
                push(0.0, "arrive", lambda a=a: submit(bulk(f"b{a}"), "bulk"))
        closed_loop["bulk"] = lambda h: submit(bulk(h.request.agent), "bulk")
        for c in range(8):
            t = 0.3 + c * 0.06
            while t < 60:
                push(t, "arrive", lambda c=c: submit(Request("chat", f"c{c}", 200 if kind_short == "prefill" else 20, kind_short), "chat"))
                t += 0.5
    return workload


def service(req):
    return req.cost_tokens / 4000.0 if req.kind == "prefill" else req.cost_tokens / 400.0


class MixedLoad(unittest.TestCase):
    def run_both(self, kind_short):
        wl = build_scenario(kind_short)
        sch = lambda clk: AgentScheduler(SchedulerConfig(enabled=True, max_inflight=4, aging_s=30.0, background=False,
                                                          max_queue_per_agent=64, max_queue_per_tenant=256),
                                         clock=clk)
        fifo = lambda clk: Fifo(clk, 4)
        a = simulate(sch, wl, 60, service)
        wl = build_scenario(kind_short)
        b = simulate(fifo, wl, 60, service)
        return a, b

    def test_short_requests_not_stuck_behind_long_prefills(self):
        for kind in ("prefill", "decode"):
            (sw, sd, sr, s), (fw, fd, fr, _) = self.run_both(kind)
            s_p95, f_p95 = pct(sw["chat"], .95), pct(fw["chat"], .95)
            print(f"\n[{kind}] chat wait p50/p95  sched {pct(sw['chat'], .5):.2f}/{s_p95:.2f}s  fifo "
                  f"{pct(fw['chat'], .5):.2f}/{f_p95:.2f}s; bulk done sched {sd['bulk']} fifo {fd['bulk']}; "
                  f"chat done sched {sd['chat']} fifo {fd['chat']}")
            self.assertLess(s_p95, 5.0)           # bounded by a slot freeing up: the long jobs cannot be preempted
            self.assertLess(s_p95 * 3, f_p95)
            self.assertEqual(sr["chat"], 0)

    def test_bulk_not_starved_and_throughput_kept(self):
        (sw, sd, sr, s), (fw, fd, fr, _) = self.run_both("prefill")
        self.assertGreaterEqual(sd["bulk"], 0.7 * fd["bulk"])          # fairness costs little bulk throughput
        self.assertLess(max(sw["bulk"]), 30.0)                         # no bulk request waits unboundedly
        self.assertGreater(len(sw["bulk"]), 30)
        self.assertLessEqual(s.metrics()["inflight"], 4)

    def test_aging_bounds_wait_of_a_starved_flow(self):
        """A near-zero-weight tenant never wins the deficit round robin against a saturating one; aging alone serves it,
        within the time the older backlog takes to drain (18 jobs x 2 s / 3 slots)."""
        def run(aging_s):
            closed_loop.clear()

            def wl(push, submit):
                for a in range(6):
                    for _ in range(3):
                        push(0.0, "arrive", lambda a=a: submit(Request("heavy", f"h{a}", 8000), "heavy"))
                closed_loop["heavy"] = lambda h: submit(Request("heavy", h.request.agent, 8000), "heavy")
                for t in (1.0, 5.0, 11.0, 17.0, 23.0):
                    push(t, "arrive", lambda: submit(Request("tiny", "t", 8000), "tiny"))
            make = lambda clk: AgentScheduler(SchedulerConfig(enabled=True, max_inflight=3, aging_s=aging_s,
                                                              background=False, tenant_weights={"tiny": 0.001}), clock=clk)
            return simulate(make, wl, 60, service)
        w, d, _, _ = run(4.0)
        self.assertEqual(d["tiny"], 5)
        self.assertLessEqual(max(w["tiny"]), 18 * 2 / 3 + 0.5, w["tiny"])
        _, d0, _, _ = run(1e9)
        self.assertEqual(d0["tiny"], 0)                                 # control: without aging it starves


if __name__ == "__main__":
    unittest.main()
