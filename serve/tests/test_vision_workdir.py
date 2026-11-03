"""The vision work folder: kept by an unload, removed at shutdown."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.engine.vision import EngineVision, Vision  # noqa: E402


class WorkDir(unittest.TestCase):
    def test_shutdown_removes_the_folder_and_an_unload_keeps_it(self):
        v = Vision({"vision_bin": "basalt-vision", "model": "m.basalt"}, start=False)
        (v.dir / "k.sve").write_bytes(b"e")
        v.unload()                                   # the cache outlives an unload
        self.assertTrue((v.dir / "k.sve").is_file())
        v.close()
        self.assertFalse(v.dir.exists())
        v.close()                                    # a second close finds nothing to remove

    def test_shutdown_removes_the_in_engine_encoders_folder(self):
        v = EngineVision({})
        (v.dir / "k.sve").write_bytes(b"e")
        v.unload()
        self.assertTrue(v.dir.is_dir())
        v.close()
        self.assertFalse(v.dir.exists())


if __name__ == "__main__":
    unittest.main()
