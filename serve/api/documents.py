"""Documents in message content: Anthropic `document` blocks, Chat Completions `file` parts and Responses `input_file`
parts become the text and image items every API's content is made of. A PDF gives its text layer page by page; a page
with little or no text gives its main embedded image (a scan, a slide's picture) to the vision encoder too. Plain-text documents
give their text. Reading the same bytes again is a cache hit, so a document resent on every turn of a conversation
renders the same prompt and costs no second parse."""
from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import io
import logging
import re
import threading
import time
import urllib.parse
import zipfile
from collections import OrderedDict
from xml.etree import ElementTree

from serve.engine.image_source import ImageRefused, ImageSources, load_image

DOCUMENT_PARTS = ("document", "file", "input_file")
MAX_DOCUMENT_BYTES = 64 << 20       # one document's bytes after base64
MAX_PAGES = 600                     # pages read from one PDF
MIN_SCAN_PIXELS = 64 * 64           # a smaller image on a page without text is decoration, not the page
SPARSE_TEXT = 200                   # a page with less text is read with its main image too (a stamped scan, a slide)
MIN_PICTURE_PIXELS = 512 * 512      # the main image of a page that has some text must be this large
_CACHE_ENTRIES = 32

_cache: OrderedDict[str, tuple] = OrderedDict()
_cache_lock = threading.Lock()
_request = threading.local()                # the image-source policy of the request this thread serves
_parses = threading.Semaphore(2)            # PDFs are parsed outside the request turn: this bounds the CPU and memory
_OFFICE_TYPES = {"application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word (.docx)",
                 "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PowerPoint (.pptx)",
                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Excel (.xlsx)"}
MAX_PART_BYTES = 64 << 20           # one XML part of an Office file, unpacked
_TEXT_TYPES = ("application/json", "application/xml", "application/x-yaml", "application/yaml", "application/csv",
               "application/javascript", "application/x-sh")


def is_document(part) -> bool:
    return isinstance(part, dict) and part.get("type") in DOCUMENT_PARTS


def has_document(content) -> bool:
    return isinstance(content, list) and any(is_document(p) for p in content)


def expand_documents(content: list) -> list:
    """`content` with each document part replaced by its text and image items, in place. A document beside other
    parts is set off from them by blank lines, so a tool result's caption and the document text stay apart."""
    out = []
    for k, part in enumerate(content):
        if not is_document(part):
            out.append(part)
            continue
        items = [dict(i) for i in _document_items(part)]
        if items and items[0]["type"] == "text" and k > 0:
            items[0]["text"] = "\n\n" + items[0]["text"]
        if items and items[-1]["type"] == "text" and k < len(content) - 1:
            items[-1]["text"] += "\n\n"
        out.extend(items)
    return out


@contextlib.contextmanager
def sources(allowed: ImageSources):
    """While a request is turned into messages: a document named by URL or path is read under `allowed`, the policy
    its images get. Outside one, every such document is refused."""
    _request.allowed = allowed
    try:
        yield
    finally:
        _request.allowed = None


def _fetch(source: str, name: str) -> bytes:
    try:
        return load_image(source, getattr(_request, "allowed", None) or ImageSources())
    except ImageRefused as e:
        print(f"[server] error: document {_quoted(name)}: {e.log_line}", flush=True)
        raise ValueError(f"document {name!r}: {e}") from None
    except ValueError as e:
        raise ValueError(f"document {name!r}: {e}") from None


def _given(value) -> str:
    return value if isinstance(value, str) else ""


def _url_name(source: str) -> str:
    """The file name a URL or a path ends in (either separator)."""
    path = source
    if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]+://", source):
        try:
            path = urllib.parse.unquote(urllib.parse.urlsplit(source).path)
        except ValueError:
            path = ""
    return re.split(r"[\\/]", path.rstrip("\\/"))[-1] or "document"


# ------------------------------------------------------------------------------------------------ the part shapes
def _document_items(part: dict) -> tuple:
    kind = part["type"]
    if kind == "document":                                   # Anthropic Messages
        return _anthropic(part)
    spec = part.get("file") if kind == "file" else part      # Chat Completions nests the fields under "file"
    if not isinstance(spec, dict):
        raise ValueError(f"a {kind} part needs a \"file\" object with file_data")
    name = _given(spec.get("filename")) or "document"
    if spec.get("file_data"):
        media, data = _data_url(spec["file_data"], kind)
        return _read(data, media, name, None, spec.get("pages"))
    if spec.get("file_id"):
        raise ValueError(f"{kind}: this server keeps no files, so a file_id cannot be read; send the file's bytes "
                         "as file_data (a data: URL)")
    if isinstance(spec.get("file_url"), str) and spec["file_url"]:
        name = _given(spec.get("filename")) or _url_name(spec["file_url"])
        return _read(_fetch(spec["file_url"], name), "", name, None, spec.get("pages"))
    raise ValueError(f"a {kind} part needs file_data (a data: URL with the file's bytes) or file_url")


def _anthropic(part: dict) -> tuple:
    src = part.get("source")
    if not isinstance(src, dict):
        raise ValueError("a document block needs a source object")
    name = _given(part.get("title")) or "document"
    context = part.get("context") if isinstance(part.get("context"), str) else None
    kind = src.get("type")
    if kind == "base64":
        return _read(_b64(src.get("data"), "document source.data"), src.get("media_type") or "", name, context,
                     part.get("pages"))
    if kind == "text":
        if not isinstance(src.get("data"), str):
            raise ValueError("a text document's source.data must be a string")
        return (_text_item(_wrap(name, None, context, _clean(src["data"]))),)
    if kind == "content":                                   # the client's own text and image blocks
        body = src.get("content")
        if isinstance(body, str):
            return (_text_item(_wrap(name, None, context, body)),)
        if not isinstance(body, list):
            raise ValueError("a content document's source.content must be a string or a list of blocks")
        head, tail = _wrap(name, None, context, "\0").split("\0")
        items = [_text_item(head)]
        for block in body:
            if isinstance(block, dict) and block.get("type") == "text":
                items.append(_text_item(block.get("text") or ""))
            elif isinstance(block, dict) and block.get("type") == "image":
                items.append(block)
            else:
                raise ValueError("a content document holds text and image blocks only")
        items.append(_text_item(tail))
        return tuple(items)
    if kind == "file":
        raise ValueError("document: this server keeps no files, so a file_id cannot be read; send the file's bytes "
                         "as a base64 source")
    if kind == "url":
        if not isinstance(src.get("url"), str) or not src["url"]:
            raise ValueError("a url document needs source.url")
        name = _given(part.get("title")) or _url_name(src["url"])
        return _read(_fetch(src["url"], name), "", name, context, part.get("pages"))
    raise ValueError(f"document source type {kind!r} is not supported (base64, text and content are)")


def _data_url(value, kind: str) -> tuple[str, bytes]:
    """file_data as (media type, bytes): a data: URL, or bare base64 as some clients send it."""
    if not isinstance(value, str):
        raise ValueError(f"{kind}: file_data must be a string")
    m = re.match(r"data:([^;,]*)(;[^,]*)?,", value)
    if not m:
        return "", _b64(value, f"{kind} file_data")
    body = value[m.end():]
    if ";base64" not in (m.group(2) or ""):
        from urllib.parse import unquote_to_bytes
        return m.group(1), unquote_to_bytes(body)
    return m.group(1), _b64(body, f"{kind} file_data")


def _b64(value, what: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{what} must be a base64 string")
    if len(value) > MAX_DOCUMENT_BYTES * 4 // 3 + 4:
        raise ValueError(f"{what}: a document is limited to {MAX_DOCUMENT_BYTES >> 20} MiB")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(f"{what} is not valid base64") from None


# ------------------------------------------------------------------------------------------------ reading
def _read(data: bytes, media: str, name: str, context: str | None, want=None) -> tuple:
    """A document's items; `want` (a Basalt extension: "1-5,8") names the PDF pages to read, None all of them."""
    media = media.split(";")[0].strip().lower()
    pdf = data[:1024].lstrip().startswith(b"%PDF-")
    if media == "application/pdf" or (pdf and media in ("", "application/octet-stream")):
        if not pdf:
            raise ValueError(f"document {name!r} is not a PDF (it does not start with %PDF-)")
        key = f"pdf:{hashlib.sha256(data).hexdigest()}:{want!r}"
        with _cache_lock:
            hit = _cache.get(key)
            if hit is not None:
                _cache.move_to_end(key)
        if hit is None:
            with _parses:
                hit = _read_pdf(data, name, want)
            with _cache_lock:
                _cache[key] = hit
                while len(_cache) > _CACHE_ENTRIES:
                    _cache.popitem(last=False)
        pages, read, items = hit
        return _framed(name, pages, read, context, items)
    if want is not None:
        raise ValueError(f"document {name!r}: pages are named in PDF documents only")
    office = _office_kind(data) if media in _OFFICE_TYPES or media in ("", "application/octet-stream") else None
    if office:
        body, count = _read_office(data, name, office)
        return (_text_item(_wrap(name, None, context, body, count)),)
    if media in _OFFICE_TYPES:
        raise ValueError(f"document {name!r} is not a {_OFFICE_TYPES[media]} file")
    if media.startswith("text/") or media in _TEXT_TYPES or (media in ("", "application/octet-stream") and not pdf):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError(f"document {name!r} is not UTF-8 text") from None
        return (_text_item(_wrap(name, None, context, _clean(text))),)
    raise ValueError(f"document {name!r}: media type {media!r} is not read (PDF, Word, PowerPoint, Excel and text "
                     "documents are)")


def _read_pdf(data: bytes, name: str, want) -> tuple[int, str | None, tuple]:
    """(pages, the pages read as named or None for all, items) for a PDF: a text item per run of text pages and an
    image item per page read as its image. Items carry page text only; _framed adds the frame."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ValueError("this server cannot read PDF documents: pypdf is missing (pip install -r requirements.txt)") \
            from None
    logging.getLogger("pypdf").setLevel(logging.ERROR)     # damaged-but-readable files warn on every object
    t0 = time.monotonic()
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError(f"document {name!r} is a password-protected PDF")
        n = len(reader.pages)
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001  pypdf raises many kinds on a damaged file
        raise ValueError(f"document {name!r} could not be read as a PDF: {e}") from None
    chosen, read = _page_list(want, n, name)
    if len(chosen) > MAX_PAGES:
        raise ValueError(f"document {name!r}: {len(chosen)} pages; at most {MAX_PAGES} are read per document "
                         "(name the ones to read with \"pages\")")
    items, text, scans, chars = [], [], 0, 0
    for k in chosen:
        try:
            page = reader.pages[k]
            body = _clean(page.extract_text() or "")
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"document {name!r}: page {k + 1} could not be read: {e}") from None
        chars += len(body)
        image = None if len(body) >= SPARSE_TEXT else _scan_of(page, MIN_PICTURE_PIXELS if body else MIN_SCAN_PIXELS)
        if image is None:
            text.append(f"[page {k + 1}]\n{body}\n" if body else f"[page {k + 1}: no text]\n")
            continue
        scans += 1
        text.append(f"[page {k + 1}, with its image]\n{body}\n" if body else f"[page {k + 1}: scanned, as an image]\n")
        items.append(_text_item("\n".join(text)))
        items.append({"type": "image_url", "image_url": image})
        text = []
    if text:
        items.append(_text_item("\n".join(text)))
    print(f"[server] document: {_quoted(name)} {n} pages" + (f", read {read}" if read else "") +
          f", {scans} as images, {chars} chars of text, {time.monotonic() - t0:.2f}s", flush=True)
    return n, read, tuple(items)


def _page_list(want, n: int, name: str) -> tuple[list[int], str | None]:
    """The 0-based pages `want` names ("3", "1-5", "1-5,8", or a number), in order without repeats, and the 1-based
    ranges as text; None names every page."""
    if want is None:
        return list(range(n)), None
    if isinstance(want, int) and not isinstance(want, bool):
        want = str(want)
    usage = f"document {name!r}: pages must name pages and ranges such as \"1-5,8\", not {want!r}"
    if not isinstance(want, str) or not want.strip():
        raise ValueError(usage)
    chosen = set()
    for piece in want.split(","):
        m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", piece)
        if not m:
            raise ValueError(usage)
        first, last = int(m.group(1)), int(m.group(2) or m.group(1))
        if first < 1 or last < first:
            raise ValueError(usage)
        if last > n:
            raise ValueError(f"document {name!r}: page {last} is past its end ({n} pages)")
        chosen.update(range(first - 1, last))
    pages = sorted(chosen)
    runs, start = [], pages[0]
    for a, b in zip(pages, pages[1:] + [None]):
        if b != a + 1:
            runs.append(f"{start + 1}" if start == a else f"{start + 1}-{a + 1}")
            start = b
    return pages, ",".join(runs)


def _scan_of(page, min_pixels: int) -> str | None:
    """The largest embedded image of a page as a data: URL, or None when it has fewer than `min_pixels`."""
    try:
        images = list(page.images)
    except Exception:  # noqa: BLE001  an image filter pypdf cannot decode
        return None
    best, size = None, 0
    for img in images:
        try:
            pil = img.image
        except Exception:  # noqa: BLE001
            continue
        if pil is None:
            continue
        pixels = pil.width * pil.height
        if pixels > size:
            best, size = (img, pil), pixels
    if best is None or size < min_pixels:
        return None
    img, pil = best
    if (pil.format or "").upper() == "JPEG" and img.data[:2] == b"\xff\xd8":
        raw, media = img.data, "image/jpeg"           # the scan's own bytes, not a second encoding
    else:
        out = io.BytesIO()
        if pil.mode not in ("RGB", "L", "RGBA", "LA"):
            pil = pil.convert("RGB")
        pil.save(out, format="PNG")
        raw, media = out.getvalue(), "image/png"
    return f"data:{media};base64,{base64.b64encode(raw).decode()}"


# ------------------------------------------------------------------------------------------------ Office files
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PR = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _office_kind(data: bytes) -> str | None:
    if not data.startswith(b"PK\x03\x04"):
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())
    except (zipfile.BadZipFile, ValueError):
        return None
    for part, kind in (("word/document.xml", "docx"), ("ppt/presentation.xml", "pptx"), ("xl/workbook.xml", "xlsx")):
        if part in names:
            return kind
    return None


def _read_office(data: bytes, name: str, kind: str) -> tuple[str, str]:
    """(text, the frame's count attribute) of a Word, PowerPoint or Excel file."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            part = lambda path: _xml(z, path)   # noqa: E731
            if kind == "docx":
                return _docx(part("word/document.xml")), ""
            if kind == "pptx":
                slides = _targets(part, "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels", "ppt/",
                                  f"{_PR}Relationship", "sldIdLst")
                body = "\n\n".join(f"[slide {k}]\n" + _paragraphs(part(path), f"{_A}p", f"{_A}t")
                                    for k, (_, path) in enumerate(slides, 1))
                return body, f'slides="{len(slides)}"'
            sheets = _targets(part, "xl/workbook.xml", "xl/_rels/workbook.xml.rels", "xl/", f"{_PR}Relationship",
                              "sheets")
            shared = [_text_of_node(si, f"{_S}t") for si in part("xl/sharedStrings.xml").iter(f"{_S}si")] \
                if "xl/sharedStrings.xml" in z.namelist() else []
            body = "\n\n".join(f"[sheet {title}]\n" + _sheet(part(path), shared) for title, path in sheets)
            return body, f'sheets="{len(sheets)}"'
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001  a damaged archive or XML part
        raise ValueError(f"document {name!r} could not be read as a {kind} file: {e}") from None


def _xml(z: zipfile.ZipFile, path: str):
    info = z.getinfo(path)
    if info.file_size > MAX_PART_BYTES:
        raise ValueError(f"an Office part is larger than {MAX_PART_BYTES >> 20} MiB unpacked")
    with z.open(info) as f:
        raw = f.read(MAX_PART_BYTES + 1)
    if len(raw) > MAX_PART_BYTES:
        raise ValueError(f"an Office part is larger than {MAX_PART_BYTES >> 20} MiB unpacked")
    return ElementTree.fromstring(raw)


def _targets(part, main: str, rels: str, base: str, rel_tag: str, list_tag: str) -> list[tuple[str, str]]:
    """(name, part path) of the slides or sheets `main` lists, in its order, through its relationships."""
    links = {r.get("Id"): r.get("Target", "") for r in part(rels).iter(rel_tag)}
    out = []
    for node in next((n for n in part(main).iter() if n.tag.endswith("}" + list_tag)), ElementTree.Element("x")):
        target = links.get(node.get(f"{_R}id"), "")
        if target:
            out.append((node.get("name") or "", target.lstrip("/") if target.startswith("/") else base + target))
    return out


def _text_of_node(node, tag: str) -> str:
    return "".join(t.text or "" for t in node.iter(tag))


def _paragraphs(root, p_tag: str, t_tag: str) -> str:
    return "\n".join(line for line in (_text_of_node(p, t_tag) for p in root.iter(p_tag)) if line.strip())


def _docx(root) -> str:
    def para(p) -> str:
        out = []
        for node in p.iter():
            if node.tag == f"{_W}t":
                out.append(node.text or "")
            elif node.tag == f"{_W}tab":
                out.append("\t")
            elif node.tag in (f"{_W}br", f"{_W}cr"):
                out.append("\n")
        return "".join(out)

    body = root.find(f"{_W}body")
    lines = []
    for block in (body if body is not None else []):
        if block.tag == f"{_W}p":
            lines.append(para(block))
        elif block.tag == f"{_W}tbl":
            for row in block.iter(f"{_W}tr"):
                lines.append(" | ".join(" ".join(para(p) for p in cell.iter(f"{_W}p")).strip()
                                        for cell in row.iter(f"{_W}tc")))
    return _clean("\n".join(lines))


def _sheet(root, shared: list[str]) -> str:
    rows = []
    for row in root.iter(f"{_S}row"):
        cells = []
        for c in row.iter(f"{_S}c"):
            kind, v = c.get("t"), c.find(f"{_S}v")
            if kind == "s" and v is not None and (v.text or "").isdigit() and int(v.text) < len(shared):
                cells.append(shared[int(v.text)])
            elif kind == "inlineStr":
                cells.append(_text_of_node(c, f"{_S}t"))
            elif kind == "b" and v is not None:
                cells.append("TRUE" if v.text == "1" else "FALSE")
            else:
                cells.append((v.text or "") if v is not None else "")
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


# ------------------------------------------------------------------------------------------------ text
def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\0", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _quoted(name: str) -> str:
    """A client-given name as one quoted line."""
    return '"' + re.sub(r"[\x00-\x1f\x7f]+", " ", name).replace('"', "'") + '"'


def _header(name: str, pages: int | None, context: str | None, read: str | None = None) -> str:
    head = (f"<document name={_quoted(name)}" + (f" pages=\"{pages}\"" if pages is not None else "") +
            (f" read=\"{read}\"" if read else "") + ">\n")
    return head + (f"<context>{context}</context>\n" if context else "")


def _wrap(name: str, pages: int | None, context: str | None, body: str, count: str = "") -> str:
    head = _header(name, pages, context)
    if count:
        head = head.replace(">\n", f" {count}>\n", 1)
    return f"{head}{body}\n</document>"


def _framed(name: str, pages: int, read: str | None, context: str | None, items: tuple) -> tuple:
    """A PDF's cached page items inside its document frame."""
    items = [dict(i) for i in items]
    head = _header(name, pages, context, read)
    if items and items[0]["type"] == "text":
        items[0]["text"] = head + items[0]["text"]
    else:
        items.insert(0, _text_item(head))
    if items[-1]["type"] == "text":
        items[-1]["text"] = items[-1]["text"].rstrip("\n") + "\n</document>"
    else:
        items.append(_text_item("</document>"))
    return tuple(items)


def _text_item(text: str) -> dict:
    return {"type": "text", "text": text}
