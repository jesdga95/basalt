"""A thread-bound request on an engine without CONTEXT_* support must still answer (found by a real-server run:
settle() raised AttributeError after the reply was made and the connection dropped)."""
import os
import unittest
from pathlib import Path
from unittest import mock

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]


class Needs(unittest.TestCase):
    def make(self, engine_cls):
        tok = ByteTokenizer()
        with mock.patch.dict(os.environ, {"BASALT_THREAD_API_KEY": "k" * 32, "BASALT_THREAD_API_PORT": "9"}):
            return Service(engine_cls(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def test_mock_engine_gets_no_lifecycle(self):
        self.assertIsNone(self.make(MockEngine).context)

    def test_engine_with_context_command_gets_one(self):
        class WithCommands(MockEngine):
            def context_command(self, command, timeout=900.0):
                return "CONTEXT_OK"
        self.assertIsNotNone(self.make(WithCommands).context)


if __name__ == "__main__":
    unittest.main()
