"""The encoded-image cache on the sidecar and `--vision dynamic` paths."""
import hashlib
import os
import shutil
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.engine import vision
from serve.engine.vision import EngineVision, Vision

SVE = b"sve"
BUDGET = 64 * len(SVE)


class Eng:
    def encode_images(self, jobs):
        for _, out in jobs:
            out.write_bytes(SVE)
        return ["OK 7 2 2 5"] * len(jobs)


def sidecar():
    v = Vision.__new__(Vision)
    v.dir = EngineVision({}).dir
    v.lock, v.cache, v.cache_bytes, v.pinned = threading.Lock(), {}, 0, {}

    def enc(img, out):
        out.write_bytes(SVE)
        return "OK 7 2 2 5"
    v._enc = enc
    return v


def dynamic():
    v = EngineVision({})
    v.attach(Eng())
    return v


def key(source):
    return hashlib.sha256(source.encode()).hexdigest()[:32]


PATHS = {"sidecar": sidecar, "dynamic": dynamic}


class VisionCache(unittest.TestCase):
    def setUp(self):
        for cls in (Vision, EngineVision):
            for name, fake in (("load", lambda s: s.encode()), ("normalize", lambda d: d)):
                p = mock.patch.object(cls, name, side_effect=fake)
                p.start()
                self.addCleanup(p.stop)
        p = mock.patch.object(vision, "VISION_CACHE_BYTES", BUDGET)
        p.start()
        self.addCleanup(p.stop)

    def each_path(self):
        for name, make in PATHS.items():
            with self.subTest(path=name):
                v = make()
                try:
                    yield v
                finally:
                    shutil.rmtree(v.dir, ignore_errors=True)

    def readable(self, v, sources):
        combined, counts = v.encode(sources)
        return combined.read_bytes() == SVE * len(sources) and counts == [7] * len(sources)

    def test_a_request_with_more_images_than_fit(self):
        for v in self.each_path():
            images = [f"a{i}" for i in range(80)]
            for _ in range(3):
                self.assertTrue(self.readable(v, images))
            self.assertLessEqual(len(v.cache), 64)

    def test_one_chat_past_the_old_cap(self):
        for v in self.each_path():
            history = []
            for turn in range(1, 67):
                history.append(f"c{turn}")
                self.assertTrue(self.readable(v, history), f"turn {turn}")

    def test_one_chat_does_not_break_another(self):
        for v in self.each_path():
            chats = {c: [] for c in range(8)}
            for c, history in chats.items():
                for n in range(8):
                    history.append(f"chat{c}-{n}")
                    self.assertTrue(self.readable(v, history))
            chats[0].append("chat0-new")
            self.assertTrue(self.readable(v, chats[0]))
            self.assertTrue(self.readable(v, chats[1]))

    def test_least_recently_used_goes_first(self):
        for v in self.each_path():
            for images in ([f"i{n}" for n in range(64)], ["i0"], ["new"]):
                v.encode(images)
            self.assertIn(key("i0"), v.cache)
            self.assertNotIn(key("i1"), v.cache)
            self.assertFalse((v.dir / f"{key('i1')}.sve").exists())

    def test_the_budget_is_bytes(self):
        for v in self.each_path():
            with mock.patch.object(vision, "VISION_CACHE_BYTES", 10 * len(SVE)):
                v.encode([f"i{n}" for n in range(30)])
                self.assertEqual(len(v.cache), 10)
                self.assertEqual(sum(path.stat().st_size for path, _, _ in v.cache.values()), 10 * len(SVE))
                self.assertEqual(v.cache_bytes, 10 * len(SVE))
                self.assertEqual(len([p for p in v.dir.glob("*.sve") if not p.name.startswith("req-")]), 10)

    def test_concurrent_requests_with_overlapping_images(self):
        # A pauses between encoding its images and copying them while B, sharing some, fills the cache twice over
        for v in self.each_path():
            a_images = [f"a{i}" for i in range(10)]
            b_images = a_images[5:] + [f"b{i}" for i in range(128)]
            done = {}
            holding, b_done = threading.Event(), threading.Event()
            encode_pinned = v._encode_pinned

            def pausing(sources, pins):
                encoded = encode_pinned(sources, pins)
                if sources is a_images:
                    holding.set()
                    b_done.wait(30)
                return encoded
            v._encode_pinned = pausing

            def request_a():
                done["a"] = self.readable(v, a_images)

            def request_b():
                holding.wait(30)
                done["b"] = self.readable(v, b_images)
                b_done.set()

            threads = [threading.Thread(target=request_a), threading.Thread(target=request_b)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(60)
            self.assertEqual(done, {"a": True, "b": True})
            self.assertEqual(v.pinned, {})
            self.assertLessEqual(len(v.cache), 64)


    def test_a_failed_write_leaves_no_combined_file(self):
        for v in self.each_path():
            encode_pinned = v._encode_pinned

            def vanishing(sources, pins):
                encoded = encode_pinned(sources, pins)
                encoded[-1][0].unlink()                 # a cached file gone before the copy
                return encoded
            v._encode_pinned = vanishing
            with self.assertRaises(FileNotFoundError):
                v.encode(["a", "b"])
            self.assertEqual(list(v.dir.glob("req-*.sve")), [])
            self.assertEqual(v.pinned, {})

    def test_input_cleanup_retries_sharing_violations(self):
        for v in self.each_path():
            unlink = Path.unlink
            attempts = {}

            def temporarily_locked(path, *args, **kwargs):
                if path.suffix == ".img":
                    attempts[path] = attempts.get(path, 0) + 1
                    if attempts[path] < 3:
                        error = PermissionError("image is still in use")
                        error.winerror = 32 if attempts[path] == 1 else 33
                        raise error
                return unlink(path, *args, **kwargs)

            with mock.patch.object(Path, "unlink", temporarily_locked), mock.patch.object(vision.time, "sleep"):
                self.assertTrue(self.readable(v, ["a", "b"]))
            self.assertEqual(list(v.dir.glob("*.img")), [])
            self.assertEqual(v.pinned, {})

    def test_input_cleanup_does_not_hide_permanent_errors(self):
        v = sidecar()
        try:
            image = v.dir / "input.img"
            for code, expected_attempts in ((32, 8), (5, 1)):
                with self.subTest(error=code):
                    error = PermissionError("cannot remove image")
                    error.winerror = code
                    with mock.patch.object(Path, "unlink", side_effect=error) as unlink, \
                            mock.patch.object(vision.time, "sleep") as sleep:
                        with self.assertRaises(PermissionError):
                            v._unlink_image(image)
                    self.assertEqual(unlink.call_count, expected_attempts)
                    self.assertEqual(sleep.call_count, expected_attempts - 1)
        finally:
            shutil.rmtree(v.dir)

    @unittest.skipUnless(os.name == "nt", "Windows sharing locks")
    def test_input_cleanup_waits_for_an_actual_open_handle(self):
        import ctypes
        from ctypes import wintypes
        create = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
        create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                           wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        create.restype = wintypes.HANDLE
        close = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        v = sidecar()
        timers = []
        enc = v._enc

        def locked(img, out):
            handle = create(str(img), 0x80000000, 1, None, 3, 0, None)
            self.assertNotEqual(handle, wintypes.HANDLE(-1).value)
            timer = threading.Timer(0.05, close, (handle,))
            timers.append(timer)
            timer.start()
            return enc(img, out)

        v._enc = locked
        try:
            self.assertTrue(self.readable(v, ["locked-image"]))
            self.assertEqual(list(v.dir.glob("*.img")), [])
            self.assertEqual(v.pinned, {})
        finally:
            for timer in timers:
                timer.join()
            shutil.rmtree(v.dir)


if __name__ == "__main__":
    unittest.main()
