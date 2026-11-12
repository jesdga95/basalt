"""Documents in message content: PDFs and text through Anthropic document blocks, Chat Completions file parts and
Responses input_file parts, as their text and their scanned pages' images."""
from __future__ import annotations

import base64
import http.client
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from serve.api import documents
from serve.api.responses import input_messages
from serve.engine.image_source import ImageSources
from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer

TEMPLATE = ChatTemplate.from_file(Path(__file__).resolve().parents[1] / "chat_template.jinja")


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list) -> bytes:
    """A PDF with one page per item: a str is lines of Helvetica text, bytes is a JPEG drawn over the page, a
    (str, bytes) pair is both."""
    objects = {1: b"<< /Type /Catalog /Pages 2 0 R >>", 3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    kids, n = [], 4
    for page in pages:
        page_id, content_id, image_id = n, n + 1, n + 2
        n += 3
        kids.append(page_id)
        words, picture = (page, None) if isinstance(page, str) else (None, page) if isinstance(page, bytes) else page
        stream, resources = b"", []
        if picture is not None:
            from PIL import Image
            w, h = Image.open(io.BytesIO(picture)).size
            stream += b"q 468 0 0 600 72 100 cm /Im1 Do Q "
            resources.append(f"/XObject << /Im1 {image_id} 0 R >>")
            objects[image_id] = (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB "
                                 f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(picture)} >>\nstream\n").encode() \
                + picture + b"\nendstream"
        else:
            objects[image_id] = b"<< >>"
        if words is not None:
            ops = ["BT /F1 12 Tf 72 720 Td"]
            for k, line in enumerate(words.split("\n")):
                ops.append(("0 -14 Td " if k else "") + f"({_escape(line)}) Tj")
            stream += (" ".join(ops) + " ET").encode("latin-1")
            resources.append("/Font << /F1 3 0 R >>")
        resources = f"<< {' '.join(resources)} >>".encode()
        objects[content_id] = f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        objects[page_id] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                            "/Resources ").encode() + resources + b" >>"
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()
    out, offsets = io.BytesIO(), {}
    out.write(b"%PDF-1.4\n")
    for k in sorted(objects):
        offsets[k] = out.tell()
        out.write(f"{k} 0 obj\n".encode() + objects[k] + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {n}\n0000000000 65535 f \n".encode())
    for k in range(1, n):
        out.write(f"{offsets[k]:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def jpeg(w: int = 160, h: int = 120) -> bytes:
    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(out, format="JPEG")
    return out.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


PAPER = make_pdf(["Mixture of experts\nRouting picks 8 of 256 experts.", "Results\nDecode reached 120 tokens/s."])


def anthropic_doc(data: bytes, title: str | None = "paper.pdf", media: str = "application/pdf") -> dict:
    block = {"type": "document", "source": {"type": "base64", "media_type": media, "data": b64(data)}}
    if title:
        block["title"] = title
    return block


class PdfText(unittest.TestCase):
    def setUp(self):
        documents._cache.clear()

    def test_pages_in_order_inside_one_frame(self):
        messages, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [
            anthropic_doc(PAPER), {"type": "text", "text": "What does page 2 say?"}]}]})
        text = messages[0]["content"]
        self.assertIsInstance(text, str)
        self.assertTrue(text.startswith('<document name="paper.pdf" pages="2">\n[page 1]\nMixture of experts'), text)
        self.assertIn("[page 2]\nResults\nDecode reached 120 tokens/s.\n</document>\n\nWhat does page 2 say?", text)
        self.assertLess(text.index("Routing picks"), text.index("[page 2]"))

    def test_claude_code_read_tool_result(self):
        """Claude Code's Read of a .pdf: a caption line, then the PDF as a document block in the tool result."""
        messages, _, _ = anthropic_to_messages({"messages": [
            {"role": "user", "content": "Summarize paper.pdf"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read",
                                               "input": {"file_path": "/w/paper.pdf"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "text", "text": "PDF file read: /w/paper.pdf (1.2KB)"},
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                "data": b64(PAPER)}}]}]}]})
        tool = [m for m in messages if m["role"] == "tool"][0]["content"]
        self.assertTrue(tool.startswith("PDF file read: /w/paper.pdf (1.2KB)\n\n<document name=\"document\" "
                                        "pages=\"2\">"), tool)
        self.assertIn("Routing picks 8 of 256 experts.", tool)

    def test_every_api_renders_the_same_prompt(self):
        data_url = "data:application/pdf;base64," + b64(PAPER)
        anthropic, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [anthropic_doc(PAPER)]}]})
        chat, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [
            {"type": "file", "file": {"filename": "paper.pdf", "file_data": data_url}}]}]})
        responses = input_messages({"input": [{"role": "user", "content": [
            {"type": "input_file", "filename": "paper.pdf", "file_data": data_url}]}]})
        self.assertEqual(anthropic[0]["content"], chat[0]["content"])
        self.assertEqual(anthropic[0]["content"], responses[0]["content"])

    def test_context_and_untitled(self):
        messages, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [
            {**anthropic_doc(PAPER, title=None), "context": "from the design review"}]}]})
        self.assertTrue(messages[0]["content"].startswith(
            '<document name="document" pages="2">\n<context>from the design review</context>\n[page 1]'))

    def test_a_resent_document_is_parsed_once(self):
        import pypdf
        with mock.patch.object(pypdf, "PdfReader", wraps=pypdf.PdfReader) as reader:
            first = documents.expand_documents([anthropic_doc(PAPER)])
            again = documents.expand_documents([anthropic_doc(PAPER, title="other.pdf")])
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(first[0]["text"].replace('"paper.pdf"', '"other.pdf"'), again[0]["text"])


class Pages(unittest.TestCase):
    """A Basalt extension: "pages" on a document names the PDF pages to read."""
    BOOK = make_pdf(["Preface", "Chapter one\nThe router.", "Chapter two\nThe experts.", "Index"])

    def setUp(self):
        documents._cache.clear()

    def read(self, pages) -> str:
        return documents.expand_documents([{**anthropic_doc(self.BOOK, "book.pdf"), "pages": pages}])[0]["text"]

    def test_named_pages_only_with_their_numbers(self):
        self.assertEqual(self.read("2-3"), '<document name="book.pdf" pages="4" read="2-3">\n[page 2]\nChapter one\n'
                                           'The router.\n\n[page 3]\nChapter two\nThe experts.\n</document>')
        self.assertIn('read="1,3-4"', self.read(" 4, 1 ,3 "))
        self.assertIn('read="3"', self.read(3))
        self.assertNotIn("Preface", self.read("2"))

    def test_every_api_takes_pages(self):
        data_url = "data:application/pdf;base64," + b64(self.BOOK)
        chat, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [
            {"type": "file", "file": {"filename": "book.pdf", "file_data": data_url, "pages": "2"}}]}]})
        responses = input_messages({"input": [{"role": "user", "content": [
            {"type": "input_file", "filename": "book.pdf", "file_data": data_url, "pages": "2"}]}]})
        self.assertEqual(chat[0]["content"], self.read("2"))
        self.assertEqual(responses[0]["content"], self.read("2"))

    def test_pages_lift_the_page_cap(self):
        with mock.patch.object(documents, "MAX_PAGES", 2):
            self.assertIn('read="1-2"', self.read("1-2"))
            with self.assertRaises(ValueError) as e:
                self.read(None)
            self.assertIn("name the ones to read", str(e.exception))

    def test_bad_pages(self):
        for pages, words in (("5", "page 5 is past its end (4 pages)"), ("0", "such as"), ("3-2", "such as"),
                             ("a", "such as"), ("", "such as"), (True, "such as"), ([1], "such as")):
            with self.assertRaises(ValueError) as e:
                self.read(pages)
            self.assertIn(words, str(e.exception))
        with self.assertRaises(ValueError) as e:
            documents.expand_documents([{**anthropic_doc(b"plain", "a.txt", "text/plain"), "pages": "1"}])
        self.assertIn("pages are named in PDF documents only", str(e.exception))


class Names(unittest.TestCase):
    def test_names_from_urls_and_paths(self):
        for source, name in (("C:\\docs\\a.pdf", "a.pdf"), ("https://x/y/b%20c.pdf?q=1", "b c.pdf"), ("/tmp/z.pdf", "z.pdf"),
                             ("file:///tmp/d.pdf", "d.pdf"), ("https://x/", "document")):
            self.assertEqual(documents._url_name(source), name)

    def test_a_name_stays_on_its_line(self):
        text = documents.expand_documents([{"type": "document", "title": 'a\n"b"', "source": {
            "type": "text", "data": "x"}}])[0]["text"]
        self.assertEqual(text, "<document name=\"a 'b'\">\nx\n</document>")
        text = documents.expand_documents([{"type": "document", "title": 7, "source": {"type": "text", "data": "x"}}])[0]
        self.assertTrue(text["text"].startswith('<document name="document">'))


class Scans(unittest.TestCase):
    def setUp(self):
        documents._cache.clear()

    def test_a_scanned_page_is_its_own_image(self):
        scan = jpeg()
        pdf = make_pdf(["Cover text", scan, "Last page"])
        messages, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [
            anthropic_doc(pdf, "scan.pdf"), {"type": "text", "text": "Read it."}]}]})
        items = messages[0]["content"]
        self.assertEqual([i["type"] for i in items], ["text", "image", "text", "text"])
        self.assertTrue(items[0]["text"].endswith("[page 2: scanned, as an image]\n"), items[0]["text"])
        self.assertEqual(items[1]["source"], "data:image/jpeg;base64," + b64(scan))   # the scan's own bytes
        self.assertEqual(items[2]["text"], "[page 3]\nLast page\n</document>\n\n")
        self.assertEqual(items[3]["text"], "Read it.")

    def test_a_page_with_little_text_keeps_its_picture(self):
        """A slide or a stamped scan: its few words, then its main image; a small picture stays out."""
        picture = jpeg(640, 480)
        pdf = make_pdf([("Quarterly results", picture), ("Logo page", jpeg(200, 100)), ("words " * 60, picture)])
        items = documents.expand_documents([anthropic_doc(pdf, "deck.pdf")])
        self.assertEqual([i["type"] for i in items], ["text", "image_url", "text"])
        self.assertTrue(items[0]["text"].endswith("[page 1, with its image]\nQuarterly results\n"), items[0]["text"])
        self.assertEqual(items[1]["image_url"], "data:image/jpeg;base64," + b64(picture))
        self.assertTrue(items[2]["text"].startswith("[page 2]\nLogo page\n\n[page 3]\nwords words"), items[2]["text"])

    def test_a_tiny_image_is_not_a_scan(self):
        messages, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [
            anthropic_doc(make_pdf([jpeg(16, 16)]))]}]})
        self.assertEqual(messages[0]["content"], '<document name="paper.pdf" pages="1">\n[page 1: no text]\n</document>')


class TextDocuments(unittest.TestCase):
    def test_text_source_and_base64_text(self):
        plain, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [
            {"type": "document", "title": "notes.md", "source": {"type": "text", "media_type": "text/plain",
                                                                "data": "line one\r\nline two\n\n\n\nend"}}]}]})
        self.assertEqual(plain[0]["content"], '<document name="notes.md">\nline one\nline two\n\nend\n</document>')
        coded, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [
            anthropic_doc(b"line one\r\nline two\n\n\n\nend", "notes.md", "text/plain")]}]})
        self.assertEqual(coded, plain)

    def test_content_source_keeps_its_blocks(self):
        messages, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [
            {"type": "document", "title": "deck", "source": {"type": "content", "content": [
                {"type": "text", "text": "slide one"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]}}]}]})
        self.assertEqual(messages[0]["content"], [
            {"type": "text", "text": '<document name="deck">\n'},
            {"type": "text", "text": "slide one"},
            {"type": "image", "source": "data:image/png;base64,AA=="},
            {"type": "text", "text": "\n</document>"}])


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
PR = 'xmlns="http://schemas.openxmlformats.org/package/2006/relationships"'
S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'


def office(parts: dict) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path, text in parts.items():
            z.writestr(path, text)
    return out.getvalue()


DOCX = office({"word/document.xml": f'<w:document {W}><w:body>'
               '<w:p><w:r><w:t>Spec</w:t></w:r></w:p>'
               '<w:p><w:r><w:t xml:space="preserve">Port </w:t></w:r><w:r><w:t>9473</w:t><w:tab/><w:t>fixed</w:t></w:r></w:p>'
               '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>key</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>value</w:t></w:r></w:p>'
               '</w:tc></w:tr><w:tr><w:tc><w:p><w:r><w:t>workers</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>6</w:t>'
               '</w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>'})
PPTX = office({"ppt/presentation.xml": f'<p:presentation xmlns:p="x" {R}><p:sldIdLst><p:sldId r:id="rId3"/>'
               '<p:sldId r:id="rId2"/></p:sldIdLst></p:presentation>',
               "ppt/_rels/presentation.xml.rels": f'<Relationships {PR}><Relationship Id="rId2" Target="slides/slide1.xml"/>'
               '<Relationship Id="rId3" Target="slides/slide2.xml"/></Relationships>',
               "ppt/slides/slide1.xml": f'<p:sld xmlns:p="x" {A}><a:p><a:r><a:t>Second in order</a:t></a:r></a:p></p:sld>',
               "ppt/slides/slide2.xml": f'<p:sld xmlns:p="x" {A}><a:p><a:r><a:t>Roadmap</a:t></a:r></a:p>'
               '<a:p><a:r><a:t>Ship </a:t></a:r><a:r><a:t>documents</a:t></a:r></a:p></p:sld>'})
XLSX = office({"xl/workbook.xml": f'<workbook {S} {R}><sheets><sheet name="Sales" r:id="rId1"/></sheets></workbook>',
               "xl/_rels/workbook.xml.rels": f'<Relationships {PR}><Relationship Id="rId1" '
               'Target="worksheets/sheet1.xml"/></Relationships>',
               "xl/sharedStrings.xml": f'<sst {S}><si><t>Region</t></si><si><t>Units</t></si><si><t>South</t></si></sst>',
               "xl/worksheets/sheet1.xml": f'<worksheet {S}><sheetData><row><c t="s"><v>0</v></c><c t="s"><v>1</v></c></row>'
               '<row><c t="s"><v>2</v></c><c><v>52</v></c><c t="b"><v>1</v></c><c t="inlineStr"><is><t>top</t></is></c>'
               '</row></sheetData></worksheet>'})


class OfficeFiles(unittest.TestCase):
    def read(self, data: bytes, media: str = "", name: str = "f") -> str:
        return documents.expand_documents([anthropic_doc(data, name, media)])[0]["text"]

    def test_word(self):
        self.assertEqual(self.read(DOCX, name="spec.docx"),
                         '<document name="spec.docx">\nSpec\nPort 9473\tfixed\nkey | value\nworkers | 6\n</document>')

    def test_powerpoint_in_presentation_order(self):
        self.assertEqual(self.read(PPTX, name="deck.pptx"), '<document name="deck.pptx" slides="2">\n[slide 1]\n'
                         'Roadmap\nShip documents\n\n[slide 2]\nSecond in order\n</document>')

    def test_excel(self):
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        self.assertEqual(self.read(XLSX, media, "q3.xlsx"), '<document name="q3.xlsx" sheets="1">\n[sheet Sales]\n'
                         'Region | Units\nSouth | 52 | TRUE | top\n</document>')

    def test_every_api_reads_them(self):
        data_url = "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64," + b64(DOCX)
        responses = input_messages({"input": [{"role": "user", "content": [
            {"type": "input_file", "filename": "spec.docx", "file_data": data_url}]}]})
        self.assertEqual(responses[0]["content"], self.read(DOCX, name="spec.docx"))

    def test_refusals(self):
        word = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        for data, media, words in ((b"PK\x03\x04 not a zip", word, "is not a Word (.docx) file"),
                                   (office({"other.xml": "<x/>"}), word, "is not a Word (.docx) file"),
                                   (office({"word/document.xml": "<broken"}), "", "could not be read as a docx file")):
            with self.assertRaises(ValueError) as e:
                self.read(data, media)
            self.assertIn(words, str(e.exception))
        with mock.patch.object(documents, "MAX_PART_BYTES", 100):
            with self.assertRaises(ValueError) as e:
                self.read(office({"word/document.xml": f"<w:document {W}>" + " " * 200 + "</w:document>"}))
            self.assertIn("larger than", str(e.exception))


class Refusals(unittest.TestCase):
    def refused(self, block: dict, words: str):
        with self.assertRaises(ValueError) as e:
            anthropic_to_messages({"messages": [{"role": "user", "content": [block]}]})
        self.assertIn(words, str(e.exception))

    def test_password_protected(self):
        from pypdf import PdfReader, PdfWriter
        writer = PdfWriter(clone_from=PdfReader(io.BytesIO(PAPER)))
        writer.encrypt(user_password="secret", owner_password="owner", algorithm="RC4-128")
        out = io.BytesIO()
        writer.write(out)
        self.refused(anthropic_doc(out.getvalue()), "password-protected")

    def test_owner_only_encryption_is_read(self):
        from pypdf import PdfReader, PdfWriter
        writer = PdfWriter(clone_from=PdfReader(io.BytesIO(PAPER)))
        writer.encrypt(user_password="", owner_password="owner", algorithm="RC4-128")
        out = io.BytesIO()
        writer.write(out)
        documents._cache.clear()
        messages, _, _ = anthropic_to_messages({"messages": [{"role": "user", "content": [
            anthropic_doc(out.getvalue())]}]})
        self.assertIn("Routing picks 8 of 256 experts.", messages[0]["content"])

    def test_damaged_and_mislabelled(self):
        self.refused(anthropic_doc(b"%PDF-1.4\n garbage"), "could not be read as a PDF")
        self.refused(anthropic_doc(b"plain words"), "is not a PDF")
        self.refused(anthropic_doc(b"\x89PNG....", media="image/png"), "media type 'image/png' is not read")
        self.refused({"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                     "data": "@@@"}}, "not valid base64")

    def test_files_are_named(self):
        self.refused({"type": "document", "source": {"type": "file", "file_id": "file_1"}}, "keeps no files")
        with self.assertRaises(ValueError) as e:
            openai_to_messages({"messages": [{"role": "user", "content": [
                {"type": "file", "file": {"file_id": "file-1"}}]}]})
        self.assertIn("keeps no files", str(e.exception))

    def test_too_many_pages(self):
        with mock.patch.object(documents, "MAX_PAGES", 1):
            documents._cache.clear()
            self.refused(anthropic_doc(PAPER), "2 pages; at most 1 are read")


class ByUrlOrPath(unittest.TestCase):
    """A document named by URL or path is read under the request's image-source policy."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "paper.pdf")
        with open(self.path, "wb") as f:
            f.write(PAPER)

    def tearDown(self):
        self.dir.cleanup()

    def convert(self, block: dict, allowed=None):
        if allowed is None:
            return anthropic_to_messages({"messages": [{"role": "user", "content": [block]}]})[0]
        with documents.sources(allowed):
            return anthropic_to_messages({"messages": [{"role": "user", "content": [block]}]})[0]

    def test_a_local_client_names_a_file(self):
        local = ImageSources(local=True, why="")
        for url in (self.path, "file://" + self.path):
            text = self.convert({"type": "document", "source": {"type": "url", "url": url}}, local)[0]["content"]
            self.assertTrue(text.startswith('<document name="paper.pdf" pages="2">'), text)
        responses = None
        with documents.sources(local):
            responses = input_messages({"input": [{"role": "user", "content": [
                {"type": "input_file", "file_url": "file://" + self.path}]}]})
        self.assertEqual(responses[0]["content"], text)

    def test_a_remote_client_needs_the_image_dir(self):
        block = {"type": "document", "title": "p.pdf", "source": {"type": "url", "url": self.path}}
        with self.assertRaises(ValueError) as e:
            self.convert(block, ImageSources(why="peer"))
        self.assertIn("document 'p.pdf': image source refused by server policy", str(e.exception))
        allowed = ImageSources(dirs=(os.path.realpath(self.dir.name),), why="peer")
        self.assertIn("Routing picks", self.convert(block, allowed)[0]["content"])

    def test_no_policy_reads_nothing(self):
        with self.assertRaises(ValueError) as e:
            self.convert({"type": "document", "source": {"type": "url", "url": self.path}})
        self.assertIn("refused by server policy", str(e.exception))

    def test_a_text_file_by_path(self):
        notes = os.path.join(self.dir.name, "notes.txt")
        with open(notes, "w") as f:
            f.write("port 9473")
        text = self.convert({"type": "document", "source": {"type": "url", "url": notes}},
                            ImageSources(local=True, why=""))[0]["content"]
        self.assertEqual(text, '<document name="notes.txt">\nport 9473\n</document>')


class OverHttp(unittest.TestCase):
    def post(self, path: str, body: dict):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "ok", max_context=16384)
        service = Service(engine, tok, TEMPLATE)
        httpd = serve(service, port=0)
        connection = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
        try:
            connection.request("POST", path, json.dumps({"model": service.model, **body}),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read()), tok.decode(engine.last_prompt)
        finally:
            connection.close()
            httpd.shutdown()
            httpd.server_close()

    def test_messages_and_count_tokens(self):
        body = {"max_tokens": 16, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": [
            anthropic_doc(PAPER), {"type": "text", "text": "Summarize."}]}]}
        status, reply, prompt = self.post("/v1/messages", body)
        self.assertEqual(status, 200, reply)
        self.assertIn("Decode reached 120 tokens/s.", prompt)
        status, counted, _ = self.post("/v1/messages/count_tokens", body)
        self.assertEqual(status, 200, counted)
        self.assertGreater(counted["input_tokens"], len("Decode reached 120 tokens/s."))

    def test_this_machine_names_a_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "paper.pdf")
            with open(path, "wb") as f:
                f.write(PAPER)
            status, reply, prompt = self.post("/v1/messages", {"max_tokens": 16, "messages": [{"role": "user", "content": [
                {"type": "document", "source": {"type": "url", "url": path}}]}]})
        self.assertEqual(status, 200, reply)
        self.assertIn("Routing picks 8 of 256 experts.", prompt)

    def test_a_bad_document_is_a_400(self):
        status, reply, _ = self.post("/v1/chat/completions", {"max_tokens": 16, "messages": [{"role": "user", "content": [
            {"type": "file", "file": {"filename": "x.pdf", "file_data": "data:application/pdf;base64," + b64(b"%PDF-1.4 x")}}]}]})
        self.assertEqual(status, 400, reply)
        self.assertIn("could not be read as a PDF", reply["error"]["message"])


if __name__ == "__main__":
    unittest.main()
