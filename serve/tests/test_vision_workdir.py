"""The vision work folder: kept by an unload, removed at shutdown."""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_owned_log_is_closed_on_close_and_start_failure_cleans_up_child(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / "vision.log"
            v = Vision({"vision_bin": "basalt-vision", "model": "m.basalt"}, log=str(log_path), start=False)
            log_fp = v._log
            self.assertFalse(log_fp.closed)
            v.close()
            self.assertTrue(log_fp.closed)
            log_path.unlink()

            # A sidecar that prints VIDEO and then exits (no READY) must be cleaned up, along with its log & workdir.
            proc = mock.Mock()
            proc.stdin = io.StringIO()
            proc.stdout = io.StringIO("VIDEO on fake\n")
            proc.stderr = None
            proc.poll.return_value = None
            with mock.patch("serve.engine.vision.lifeline.popen", return_value=proc):
                with self.assertRaisesRegex(RuntimeError, "did not start"):
                    Vision({"vision_bin": "basalt-vision", "model": "m.basalt"}, log=str(log_path), start=True)
            proc.kill.assert_called_once()
            proc.wait.assert_called_once()
            self.assertTrue(proc.stdin.closed)
            self.assertTrue(proc.stdout.closed)
            log_path.unlink()

    def test_broken_pipe_in_enc_and_encv_marks_stopped_and_raises_value_error(self):
        class BrokenStdin:
            closed = False

            def write(self, _):
                raise BrokenPipeError("pipe closed")

            def flush(self):
                pass

            def close(self):
                self.closed = True

        for kind in ("image", "video"):
            with self.subTest(kind=kind):
                v = Vision({"vision_bin": "basalt-vision", "model": "m.basalt"}, start=False)
                try:
                    proc = mock.Mock()
                    proc.stdin = BrokenStdin()
                    proc.stdout = io.StringIO()
                    proc.stderr = None
                    proc.poll.return_value = None
                    v.proc = proc
                    v.stopped = False
                    with self.assertRaisesRegex(ValueError, "the vision encoder stopped"):
                        if kind == "image":
                            v.encode([b"\x89PNG\r\n\x1a\n"])
                        else:
                            v.encode_media([("video", b"clip", {})])
                    self.assertTrue(v.stopped)
                    self.assertFalse(v.alive())
                    self.assertIsNone(v.proc)
                    self.assertTrue(proc.stdin.closed)
                    self.assertTrue(proc.stdout.closed)
                    self.assertEqual(list(v.dir.glob("*.part")), [])
                finally:
                    v.close()


if __name__ == "__main__":
    unittest.main()
