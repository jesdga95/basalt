"""The data contract between GET /metrics and GET /slots and the stats page (serve/web/app.js).

A field the page reads that the server stops sending is silent: the JavaScript draws "-" and nothing fails. These
lists are what the page reads, so a renamed or dropped field fails here instead of in someone's browser. The request
record's own fields are covered by test_metrics, which drives real requests; slot fields that only exist while a
request runs are left to it for the same reason.
"""
import unittest
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.service import Service
from serve.util import telemetry
from serve.util.text import ByteTokenizer

TEMPLATE = Path(__file__).resolve().parents[1] / "chat_template.jinja"

# what app.js reads from each section of /metrics
SECTIONS = {
    "engine": ("model", "max_context", "path", "version", "quant", "kv", "expert_slots", "expert_cache_mib",
               "expert_slots_primary", "expert_cache_primary_mib", "conversation_cache_slots",
               "conversation_cache_mib"),
    "live": ("state", "load", "queued", "slots", "queue", "prompt_tokens", "generated", "elapsed_s", "tok_s",
             "phase", "prefill_tok_s_mean"),
    "hardware": ("gpus", "cpu", "ram_used", "ram_total", "disk_read_mb", "disk_write_mb"),
    "hardware_static": ("gpus", "cpu_name", "cores", "threads", "driver", "cuda"),
    "history": ("tok_s", "prefill_tok_s_mean", "slot_tok_s", "gpus"),
}
TOP = ("requests", "best", "totals", "conversation")
# the slot table the page draws, for a slot that is not running
SLOT = ("id", "n_ctx", "ctx_used", "is_processing", "state", "generated", "max_tokens")


class FakeGpu:
    """An NVML reader that answers the fields the sampler keeps."""

    def __init__(self, index=0):
        self.index = index

    def ok(self):
        return True

    def name(self):
        return "Card 0"

    def driver_version(self):
        return "610.57.04"

    def cuda_version(self):
        return "13.3"

    def read(self):
        return {"util": 10, "mem_used": 100, "mem_total": 1000, "temp": 40, "power": 50.5, "power_limit": 400.0,
                "pcie_gen": 5, "pcie_gen_max": 5, "pcie_width": 16, "pcie_rx_mb": 1.0, "pcie_tx_mb": 2.0}


class PageContract(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "hi"), tok, ChatTemplate(TEMPLATE))
        # the figures a real engine reports on its INFO line; the page reads all of them
        self.svc.engine.info = {"version": "1.2.3", "quant": "NVFP4", "kv": "int8", "expert_slots": 100,
                                "expert_cache_mib": 1024, "expert_slots_primary": 100,
                                "expert_cache_primary_mib": 1024, "conversation_cache_slots": 4,
                                "conversation_cache_mib": 16384}
        self.svc.telemetry = telemetry.Telemetry(gpu_indices=[0], gpu_factory=FakeGpu)
        self.telemetry = self.svc.telemetry

    def test_every_metrics_section_the_page_reads_is_there(self):
        m = self.svc.metrics()
        for section, fields in SECTIONS.items():
            self.assertIn(section, m, f"the page reads /metrics {section}")
            missing = [f for f in fields if f not in m[section]]
            self.assertEqual(missing, [], f"/metrics {section} lost {missing}")
        missing = [k for k in TOP if k not in m]
        self.assertEqual(missing, [], f"/metrics lost {missing}")

    def test_the_card_and_history_fields_the_charts_plot(self):
        self.telemetry.record(self.telemetry.sample())
        m = self.svc.metrics()
        self.assertTrue(m["hardware"]["gpus"], "a card must appear for the sidebar to draw one")
        missing = [k for k in telemetry.GPU_KEYS if k not in m["hardware"]["gpus"][0]]
        self.assertEqual(missing, [], f"/metrics hardware.gpus lost {missing}")
        card = m["history"]["gpus"][0]
        missing = [k for k in telemetry.GPU_SERIES if k not in card]
        self.assertEqual(missing, [], f"/metrics history.gpus lost {missing}")

    def test_the_slot_fields_the_page_draws(self):
        rows, _ = self.svc.slot_table()
        self.assertTrue(rows, "the page draws one row per slot")
        missing = [k for k in SLOT if k not in rows[0]]
        self.assertEqual(missing, [], f"/slots lost {missing}")


if __name__ == "__main__":
    unittest.main()
