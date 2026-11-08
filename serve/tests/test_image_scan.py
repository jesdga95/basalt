"""An image's source (often megabytes of base64) is never scanned or copied for literal tags; the text around it is."""
import unittest

from serve.frontend import mark_think_literals


class Source(str):
    """A source that fails when it is searched or rewritten."""

    def __contains__(self, item):
        raise AssertionError("the image source was scanned")

    def replace(self, *args):
        raise AssertionError("the image source was rewritten")


class ImageScan(unittest.TestCase):
    def messages(self, text):
        src = Source("data:image/png;base64,AAAA")
        return src, [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image", "source": src}]}]

    def test_no_tag_leaves_the_messages_and_never_reads_the_source(self):
        _, messages = self.messages("what is this?")
        out, _, changed = mark_think_literals(messages, None)
        self.assertFalse(changed)
        self.assertIs(out, messages)

    def test_a_tag_in_the_text_is_marked_and_the_source_passes_through(self):
        src, messages = self.messages("quote <think> here")
        out, _, changed = mark_think_literals(messages, None)
        self.assertTrue(changed)
        text, image = out[0]["content"]
        self.assertNotIn("<think>", text["text"])
        self.assertIs(image["source"], src)


if __name__ == "__main__":
    unittest.main()
