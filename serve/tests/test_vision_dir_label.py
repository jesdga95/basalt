"""The vision scratch dir: its Low mandatory label, its private DACL, and the mkdtemp fallback."""
import shutil
import unittest

from serve.engine import vision


@unittest.skipUnless(vision.os.name == "nt", "mandatory labels are a Windows thing")
class Label(unittest.TestCase):
    def sddl(self, directory):
        import ctypes
        from ctypes import wintypes
        adv = ctypes.windll.advapi32
        adv.GetNamedSecurityInfoW.restype = wintypes.DWORD
        adv.GetNamedSecurityInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.POINTER(ctypes.c_void_p)]
        sd = ctypes.c_void_p()
        self.assertEqual(adv.GetNamedSecurityInfoW(str(directory), 1, 0x14, None, None, None, None,
                                                   ctypes.byref(sd)), 0)         # file object, DACL|label
        adv.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = \
            [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.c_wchar_p),
             ctypes.POINTER(wintypes.DWORD)]
        text, size = ctypes.c_wchar_p(), wintypes.DWORD()
        self.assertEqual(adv.ConvertSecurityDescriptorToStringSecurityDescriptorW(sd, 1, 0x14, ctypes.byref(text),
                                                                                  ctypes.byref(size)), 1)
        return text.value

    def test_labeled_low(self):
        directory = vision._labeled_dir()
        self.assertIsNotNone(directory)
        try:
            sddl = self.sddl(directory)
            self.assertIn("(ML;OICI;NW;;;LW)", sddl)                  # Low, inherited by everything placed here
            self.assertIn("FA;;;OW", sddl)                            # and still private to the owner
            (directory / "probe.bin").write_bytes(b"x")               # the server writes down into it
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_vision_dir_falls_back_when_the_helper_refuses(self):
        """What this pins: `_vision_dir`'s None-handling, not the OS refusal itself."""
        original = vision._labeled_dir
        vision._labeled_dir = lambda: None
        directory = None
        try:
            directory = vision._vision_dir()
            self.assertTrue(directory.is_dir())
        finally:
            vision._labeled_dir = original
            if directory is not None:
                shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
