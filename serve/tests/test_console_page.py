"""The page contract: what the console's app.js reads of /metrics is what the server sends."""
import re
import unittest
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "serve" / "chat_template.jinja"
APP_JS = (ROOT / "serve" / "web" / "app.js").read_text(encoding="utf-8")


class PageContract(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "Hi.")
        self.engine.conversation = {"parked": 2, "bytes": 5, "evictions": 1, "restored": 9, "restored_ms": 1.5}
        self.svc = Service(self.engine, tok, ChatTemplate.from_file(TEMPLATE))
        self.m = self.svc.metrics()

    def reads(self, var, text=APP_JS):
        return set(re.findall(rf"\b{var}\.([a-z_]+)", text))

    def test_the_top_level_fields(self):
        top = self.reads("M") - {"slots", "models", "swap"}          # /slots, and the router's
        self.assertTrue({"live", "engine", "history", "requests", "best", "hardware", "totals", "conversation"} <= top)
        self.assertEqual(top - set(self.m), set())

    def test_the_live_fields(self):
        live = self.reads("live")
        self.assertTrue({"state", "phase", "slots"} <= live)
        self.assertEqual(live - set(self.m["live"]), set())

    def test_the_conversation_and_totals_fields(self):
        self.assertEqual(self.reads("conv") - set(self.m["conversation"]), set())
        totals = APP_JS.split("function drawTotals")[1].split("\n}\n")[0]
        self.assertEqual(self.reads("n", totals) - set(self.m["totals"]), set())
        self.assertEqual(self.reads("t", totals) - set(self.m["totals"]), set())
