"""--lazy drops only a sidecar image encoder: --vision dynamic encodes inside the engine, which a request loads first."""
import contextlib
import io
import unittest

from serve.server import drop_sidecar_vision


def drop(cfg):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        drop_sidecar_vision(cfg)
    return out.getvalue()


class LazyVision(unittest.TestCase):
    def test_dynamic_stays(self):
        cfg = {"vision": {"mode": "dynamic"}, "args": ["--vision", "dynamic"]}
        self.assertEqual(drop(cfg), "")
        self.assertEqual(cfg["vision"], {"mode": "dynamic"})
        self.assertEqual(cfg["args"], ["--vision", "dynamic"])

    def test_sidecar_goes_with_its_flags(self):
        for mode in ("cpu", "1"):
            cfg = {"vision": {"mode": mode}, "args": ["--ctx", "4096", "--vision-mmproj", "a.gguf", "--vision-mmproj", "b.gguf"]}
            self.assertIn("vision off", drop(cfg))
            self.assertNotIn("vision", cfg)
            self.assertEqual(cfg["args"], ["--ctx", "4096"])

    def test_no_vision_says_nothing(self):
        cfg = {"args": []}
        self.assertEqual(drop(cfg), "")
        self.assertEqual(cfg, {"args": []})


if __name__ == "__main__":
    unittest.main()
