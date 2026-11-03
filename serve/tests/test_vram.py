"""Per-process GPU memory from NVML: an unlisted or unreadable process is unknown, never zero."""
import unittest
from unittest import mock

from serve.util import vram


class ProcessMemory(unittest.TestCase):
    def test_a_listed_process_sums_its_cards(self):
        rows = {0: [(10, 3 << 30), (7, 1 << 30)], 1: [(7, 2 << 30)]}
        with mock.patch.object(vram, "processes", side_effect=lambda g: rows[g]):
            self.assertEqual(vram.process_by_gpu(7, [0, 1]), {0: 1024, 1: 2048})
            self.assertEqual(vram.process_mib(7, [0, 1]), 3072)

    def test_an_unlisted_process_is_unknown_not_zero(self):
        with mock.patch.object(vram, "processes", return_value=[(10, 3 << 30)]):
            self.assertIsNone(vram.process_by_gpu(7, [0]))
            self.assertIsNone(vram.process_mib(7, [0]))
        with mock.patch.object(vram, "processes", return_value=[]):
            self.assertIsNone(vram.process_mib(7, [0]))

    def test_an_unreadable_amount_is_unknown(self):
        with mock.patch.object(vram, "processes", return_value=[(7, None)]):
            self.assertIsNone(vram.process_mib(7, [0]))

    def test_holders_say_when_a_size_is_unreadable(self):
        with mock.patch.object(vram, "processes", return_value=[(7, None), (8, 2 << 30)]), \
                mock.patch.object(vram, "_command", side_effect=lambda pid: f"cmd{pid}"):
            text = vram.holders([0], ignore=None)
        self.assertIn("pid 7 (cmd7) an unreadable amount on GPU 0", text)
        self.assertIn("pid 8 (cmd8) 2.0 GiB on GPU 0", text)

    def test_the_not_available_marker_reads_as_unknown(self):
        self.assertEqual(vram._NOT_AVAILABLE, 2**64 - 1)


if __name__ == "__main__":
    unittest.main()
