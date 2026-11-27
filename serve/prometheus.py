"""Prometheus text exposition format (0.0.4) for GET /metrics.

Every metric is named basalt_* in Prometheus's own convention (underscores, the unit in the name, _total on counters);
a dashboard built for another engine's names maps onto them with relabeling rules.
"""
from __future__ import annotations

import math

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
E2E_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
TPOT_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
STATES = ("idle", "reading", "generating", "loading", "asleep", "unloaded")


class Histogram:
    """Cumulative Prometheus histogram with fixed upper bounds."""

    def __init__(self, buckets: tuple[float, ...]):
        self.buckets = buckets
        self.counts = [0] * (len(buckets) + 1)
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            return
        self.sum += value
        self.count += 1
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1
        self.counts[-1] += 1

    def snapshot(self) -> tuple[tuple[float, ...], list[int], float, int]:
        return self.buckets, list(self.counts), self.sum, self.count


class PromStats:
    """Per-service cumulative counters and latency histograms for Prometheus."""

    def __init__(self):
        self.ttft = Histogram(TTFT_BUCKETS)
        self.e2e = Histogram(E2E_BUCKETS)
        self.tpot = Histogram(TPOT_BUCKETS)
        self.finished: dict[str, int] = {}
        self.drafts_offered = 0
        self.drafts_accepted = 0

    def record(self, *, finish: str, ttft_s: float | None, duration_s: float | None,
               decode_ms: float | None, output_tokens: int,
               drafts_offered: int | None, drafts_accepted: int | None) -> None:
        reason = finish or "unknown"
        self.finished[reason] = self.finished.get(reason, 0) + 1
        if ttft_s is not None:
            self.ttft.observe(ttft_s)
        if duration_s is not None:
            self.e2e.observe(duration_s)
        if output_tokens > 0 and decode_ms is not None and decode_ms > 0:
            self.tpot.observe((decode_ms / 1000.0) / output_tokens)
        if drafts_offered:
            self.drafts_offered += int(drafts_offered)
        if drafts_accepted:
            self.drafts_accepted += int(drafts_accepted)

    def snapshot(self) -> dict:
        return {
            "ttft": self.ttft.snapshot(),
            "e2e": self.e2e.snapshot(),
            "tpot": self.tpot.snapshot(),
            "finished": dict(self.finished),
            "drafts_offered": self.drafts_offered,
            "drafts_accepted": self.drafts_accepted,
        }


def wants_prometheus(accept: str | None, query_format: str | None) -> bool:
    """True when the request asks for Prometheus text format via ?format=prometheus or Accept header."""
    if (query_format or "").lower() in ("prometheus", "prom", "openmetrics"):
        return True
    if not accept:
        return False
    best_prom, best_json = -1.0, -1.0
    for part in accept.split(","):
        items = [p.strip() for p in part.split(";") if p.strip()]
        if not items:
            continue
        mime = items[0].lower()
        q = 1.0
        for param in items[1:]:
            if param.lower().startswith("q="):
                try:
                    q = float(param[2:])
                except ValueError:
                    q = 0.0
        if q <= 0:
            continue
        if mime in ("text/plain", "application/openmetrics-text"):
            best_prom = max(best_prom, q)
        elif mime == "application/json":
            best_json = max(best_json, q)
    return best_prom > 0 and best_prom >= best_json


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return ""
    return "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in pairs) + "}"


def _num(v: float | int) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if v == 0:
        return "0"
    s = repr(float(v))
    return s[:-2] if s.endswith(".0") else s


class _Builder:
    """Collects samples by metric family so each # HELP / # TYPE is emitted exactly once."""

    def __init__(self):
        self._order: list[str] = []
        self._meta: dict[str, tuple[str, str]] = {}
        self._lines: dict[str, list[str]] = {}

    def add(self, name: str, kind: str, help_text: str, labels: list[tuple[str, str]], value: float | int) -> None:
        if name not in self._meta:
            self._order.append(name)
            self._meta[name] = (kind, help_text)
            self._lines[name] = []
        self._lines[name].append(f"{name}{_labels(labels)} {_num(value)}")

    def add_histogram(self, name: str, help_text: str, labels: list[tuple[str, str]],
                      snap: tuple[tuple[float, ...], list[int], float, int]) -> None:
        if name not in self._meta:
            self._order.append(name)
            self._meta[name] = ("histogram", help_text)
            self._lines[name] = []
        buckets, counts, total, count = snap
        for i, bound in enumerate(buckets):
            self._lines[name].append(f"{name}_bucket{_labels(labels + [('le', _num(bound))])} {counts[i]}")
        self._lines[name].append(f"{name}_bucket{_labels(labels + [('le', '+Inf')])} {counts[-1]}")
        self._lines[name].append(f"{name}_sum{_labels(labels)} {_num(total)}")
        self._lines[name].append(f"{name}_count{_labels(labels)} {count}")

    def render(self) -> str:
        out: list[str] = []
        for name in self._order:
            kind, help_text = self._meta[name]
            out.append(f"# HELP {name} {help_text}")
            out.append(f"# TYPE {name} {kind}")
            out.extend(self._lines[name])
        return "\n".join(out) + "\n"


def render_prometheus(entries: list[tuple[dict, dict]]) -> str:
    """Render one or more `(metrics_dict, prom_stats_snapshot)` pairs in Prometheus 0.0.4 text format."""
    b = _Builder()
    hw_emitted = False

    for m, p in entries:
        engine = m.get("engine") or {}
        live = m.get("live") or {}
        slots = live.get("slots") or {}
        totals = m.get("totals") or {}
        model = str(engine.get("model") or "default")
        ml = [("model_name", model)]

        # ---- requests and tokens: gauges and counters ----
        running = int(slots.get("active") or 0)
        waiting = int(live.get("queued") or 0)
        b.add("basalt_requests_running", "gauge", "Number of requests currently running.", ml, running)
        b.add("basalt_requests_waiting", "gauge", "Number of requests waiting to be processed.", ml, waiting)

        prompt_tok = int(totals.get("prompt") or 0)
        reused_tok = int(totals.get("reused") or 0)
        gen_tok = int(totals.get("written") or 0)
        b.add("basalt_prompt_tokens_total", "counter", "Number of prefill tokens processed.", ml, prompt_tok)
        b.add("basalt_generation_tokens_total", "counter", "Number of generation tokens processed.", ml, gen_tok)
        b.add("basalt_prefix_cache_hit_tokens_total", "counter", "Number of prefix cache hit tokens.", ml, reused_tok)
        b.add("basalt_prefix_cache_query_tokens_total", "counter", "Number of prefix cache queried tokens.", ml, prompt_tok)

        finished = p.get("finished") or {}
        for reason, count in sorted(finished.items()):
            b.add("basalt_requests_finished_total", "counter", "Count of finished requests by finish reason.",
                  ml + [("finished_reason", reason)], count)

        b.add("basalt_draft_tokens_total", "counter",
              "Total number of speculative draft tokens proposed.", ml, int(p.get("drafts_offered") or 0))
        b.add("basalt_draft_tokens_accepted_total", "counter",
              "Total number of speculative draft tokens accepted.", ml, int(p.get("drafts_accepted") or 0))

        # ---- latency histograms ----
        b.add_histogram("basalt_time_to_first_token_seconds", "Histogram of time to first token in seconds.", ml, p["ttft"])
        b.add_histogram("basalt_request_duration_seconds", "Histogram of end to end request latency in seconds.", ml, p["e2e"])
        b.add_histogram("basalt_time_per_output_token_seconds", "Histogram of time per output token in seconds.", ml, p["tpot"])

        # ---- Basalt-specific engine metrics ----
        cur_state = str(live.get("state") or "idle")
        for st in STATES:
            b.add("basalt_model_state", "gauge", "Current model state (1 for active state).",
                  ml + [("state", st)], 1 if cur_state == st else 0)
        b.add("basalt_decode_tokens_per_second", "gauge", "Current live decode throughput in tokens per second.",
              ml, float(live.get("tok_s") or 0.0))
        b.add("basalt_prefill_tokens_per_second", "gauge", "Current live prefill throughput in tokens per second.",
              ml, float(live.get("prefill_tok_s_mean") or 0.0))
        b.add("basalt_max_context_tokens", "gauge", "Configured maximum context length in tokens.",
              ml, int(engine.get("max_context") or 0))
        b.add("basalt_prompt_seconds_total", "counter", "Total time spent in prompt processing in seconds.",
              ml, float(totals.get("prompt_ms") or 0.0) / 1000.0)
        b.add("basalt_decode_seconds_total", "counter", "Total time spent in token generation in seconds.",
              ml, float(totals.get("decode_ms") or 0.0) / 1000.0)
        spill = engine.get("spill")
        if isinstance(spill, dict):
            spill = spill.get("experts")
        if spill is not None:
            b.add("basalt_experts_spilled", "gauge", "Number of routed experts spilled to disk.",
                  ml, int(spill))
        if engine.get("expert_total") is not None:
            b.add("basalt_experts", "gauge", "Total number of routed experts.",
                  ml, int(engine["expert_total"]))

        refused = (m.get("conversation") or {}).get("refused")
        if refused is not None:
            b.add("basalt_conversation_parking_refused_total", "counter",
                  "Conversation-cache parkings refused by RAM admission since the engine started.", ml, int(refused))

        # ---- Host / GPU hardware metrics (emitted once per scrape) ----
        if not hw_emitted:
            hw = m.get("hardware") or {}
            hw_static_gpus = (m.get("hardware_static") or {}).get("gpus") or []
            for i, g in enumerate(hw.get("gpus") or []):
                gname = g.get("name") or (hw_static_gpus[i].get("name") if i < len(hw_static_gpus) else None) or f"gpu{i}"
                gl = [("gpu", str(i)), ("name", str(gname))]
                if g.get("mem_used") is not None:
                    b.add("basalt_gpu_memory_used_bytes", "gauge", "GPU memory used in bytes.", gl, int(g["mem_used"]))
                if g.get("mem_total") is not None:
                    b.add("basalt_gpu_memory_total_bytes", "gauge", "GPU memory total in bytes.", gl, int(g["mem_total"]))
                if g.get("util") is not None:
                    b.add("basalt_gpu_utilization_ratio", "gauge", "GPU utilization ratio (0 to 1).",
                          gl, float(g["util"]) / 100.0)
                if g.get("temp") is not None:
                    b.add("basalt_gpu_temperature_celsius", "gauge", "GPU temperature in Celsius.", gl, float(g["temp"]))
                if g.get("power") is not None:
                    b.add("basalt_gpu_power_watts", "gauge", "GPU power draw in Watts.", gl, float(g["power"]))
            if hw.get("ram_used") is not None:
                b.add("basalt_ram_used_bytes", "gauge", "Host RAM used in bytes.", [], int(hw["ram_used"]))
            if hw.get("ram_total") is not None:
                b.add("basalt_ram_total_bytes", "gauge", "Host RAM total in bytes.", [], int(hw["ram_total"]))
            hw_emitted = True

    return b.render()
