"""The image pixel budget: a header over 64 MP is refused before its pixels are loaded."""
import struct
import unittest

from serve.engine.vision import IMAGE_MAX_PIXELS, Vision

# A PNG's signature is all normalize() needs to pass it to the decoder, which enforces the budget itself.
PNG = b"\x89PNG\r\n\x1a\n" + b"body"


def tiff(w: int, h: int) -> bytes:
    """A TIFF header with the given size and no pixel data: Pillow reads the size without loading pixels."""
    entries = [
        (256, 4, 1, w),      # ImageWidth
        (257, 4, 1, h),      # ImageLength
        (258, 3, 3, None),   # BitsPerSample (three shorts, at an offset)
        (259, 3, 1, 1),      # Compression: none
        (262, 3, 1, 2),      # Photometric: RGB
        (273, 4, 1, None),   # StripOffsets
        (277, 3, 1, 3),      # SamplesPerPixel
        (278, 4, 1, h),      # RowsPerStrip
        (279, 4, 1, 0),      # StripByteCounts: no rows follow
    ]
    ifd_offset = 8
    data_offset = ifd_offset + 2 + 12 * len(entries) + 4
    bps_offset, strip_offset = data_offset, data_offset + 6
    out = bytearray(b"II" + struct.pack("<HI", 42, ifd_offset))
    out += struct.pack("<H", len(entries))
    for tag, kind, count, value in entries:
        if tag == 258:
            value = bps_offset
        if tag == 273:
            value = strip_offset
        raw = struct.pack("<HH", value, 0) if kind == 3 and count == 1 else struct.pack("<I", value)
        out += struct.pack("<HHI", tag, kind, count) + raw
    out += struct.pack("<I", 0)
    out += struct.pack("<HHH", 8, 8, 8) + b"\x00" * 16
    return bytes(out)


class PixelBudgetTest(unittest.TestCase):
    def test_an_image_past_the_budget_is_refused(self):
        # 9000 x 9000 = 81,000,000 pixels: past the 64 Mi limit and below Pillow's own bomb warning.
        with self.assertRaisesRegex(ValueError, "megapixels"):
            Vision.normalize(tiff(9000, 9000))

    def test_pillows_bomb_limit_is_reworded(self):
        # 20000 x 20000 is past Pillow's own open-time limit; the message still names the pixel budget.
        with self.assertRaisesRegex(ValueError, "megapixels"):
            Vision.normalize(tiff(20000, 20000))

    def test_the_budget_matches_the_decoder(self):
        self.assertEqual(IMAGE_MAX_PIXELS, 64 << 20)

    def test_a_png_still_passes_through(self):
        self.assertEqual(Vision.normalize(PNG), PNG)


if __name__ == "__main__":
    unittest.main()
