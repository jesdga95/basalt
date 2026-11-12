"""Documents in message content: Anthropic `document` blocks, Chat Completions `file` parts and Responses `input_file`
parts become the text and image items every API's content is made of. A PDF gives its text layer page by page; a page
without text whose content is an embedded image (a scan) gives that image to the vision encoder. Plain-text documents
give their text. Reading the same bytes again is a cache hit, so a document resent on every turn of a conversation
renders the same prompt and costs no second parse."""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import logging
import re
import threading
import time
from collections import OrderedDict

DOCUMENT_PARTS = ("document", "file", "input_file")
MAX_DOCUMENT_BYTES = 64 << 20       # one document's bytes after base64
MAX_PAGES = 600                     # pages read from one PDF
MIN_SCAN_PIXELS = 64 * 64           # a smaller image on a page without text is decoration, not the page
_CACHE_ENTRIES = 32

_cache: OrderedDict[str, tuple] = OrderedDict()
_cache_lock = threading.Lock()
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


# ------------------------------------------------------------------------------------------------ the part shapes
def _document_items(part: dict) -> tuple:
    kind = part["type"]
    if kind == "document":                                   # Anthropic Messages
        return _anthropic(part)
    spec = part.get("file") if kind == "file" else part      # Chat Completions nests the fields under "file"
    if not isinstance(spec, dict):
        raise ValueError(f"a {kind} part needs a \"file\" object with file_data")
    name = spec.get("filename") or "document"
    if spec.get("file_data"):
        media, data = _data_url(spec["file_data"], kind)
        return _read(data, media, name, None)
    if spec.get("file_id"):
        raise ValueError(f"{kind}: this server keeps no files, so a file_id cannot be read; send the file's bytes "
                         "as file_data (a data: URL)")
    if spec.get("file_url"):
        raise ValueError(f"{kind}: a document by URL is not fetched; send the file's bytes as file_data (a data: URL)")
    raise ValueError(f"a {kind} part needs file_data (a data: URL with the file's bytes)")


def _anthropic(part: dict) -> tuple:
    src = part.get("source")
    if not isinstance(src, dict):
        raise ValueError("a document block needs a source object")
    name = part.get("title") or "document"
    context = part.get("context") if isinstance(part.get("context"), str) else None
    kind = src.get("type")
    if kind == "base64":
        return _read(_b64(src.get("data"), "document source.data"), src.get("media_type") or "", name, context)
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
        raise ValueError("document: a document by URL is not fetched; send the file's bytes as a base64 source")
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
def _read(data: bytes, media: str, name: str, context: str | None) -> tuple:
    media = media.split(";")[0].strip().lower()
    pdf = data[:1024].lstrip().startswith(b"%PDF-")
    if media == "application/pdf" or (pdf and media in ("", "application/octet-stream")):
        if not pdf:
            raise ValueError(f"document {name!r} is not a PDF (it does not start with %PDF-)")
        key = "pdf:" + hashlib.sha256(data).hexdigest()
        with _cache_lock:
            hit = _cache.get(key)
            if hit is not None:
                _cache.move_to_end(key)
        if hit is None:
            hit = _read_pdf(data, name)
            with _cache_lock:
                _cache[key] = hit
                while len(_cache) > _CACHE_ENTRIES:
                    _cache.popitem(last=False)
        pages, items = hit
        return _framed(name, pages, context, items)
    if media.startswith("text/") or media in _TEXT_TYPES or (media in ("", "application/octet-stream") and not pdf):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError(f"document {name!r} is not UTF-8 text") from None
        return (_text_item(_wrap(name, None, context, _clean(text))),)
    raise ValueError(f"document {name!r}: media type {media!r} is not read (PDF and text documents are)")


def _read_pdf(data: bytes, name: str) -> tuple[int, tuple]:
    """(pages, items) for a PDF: a text item per run of text pages and an image item per scanned page. Items carry
    page text only; _framed adds the name and the page count."""
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
    if n > MAX_PAGES:
        raise ValueError(f"document {name!r} has {n} pages; at most {MAX_PAGES} are read per document")
    items, text, scans, chars = [], [], 0, 0
    for k in range(n):
        try:
            page = reader.pages[k]
            body = _clean(page.extract_text() or "")
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"document {name!r}: page {k + 1} could not be read: {e}") from None
        if body:
            chars += len(body)
            text.append(f"[page {k + 1}]\n{body}\n")
            continue
        image = _scan_of(page)
        if image is None:
            text.append(f"[page {k + 1}: no text]\n")
            continue
        scans += 1
        text.append(f"[page {k + 1}: scanned, as an image]\n")
        items.append(_text_item("\n".join(text)))
        items.append({"type": "image_url", "image_url": image})
        text = []
    if text:
        items.append(_text_item("\n".join(text)))
    print(f"[server] document: {_quoted(name)} {n} pages, {scans} scanned, {chars} chars of text, "
          f"{time.monotonic() - t0:.2f}s", flush=True)
    return n, tuple(items)


def _scan_of(page) -> str | None:
    """The largest embedded image of a page as a data: URL, or None when the page has none worth reading."""
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
    if best is None or size < MIN_SCAN_PIXELS:
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


# ------------------------------------------------------------------------------------------------ text
def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\0", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _quoted(name: str) -> str:
    return '"' + name.replace('"', "'") + '"'


def _header(name: str, pages: int | None, context: str | None) -> str:
    head = f"<document name={_quoted(name)}" + (f" pages=\"{pages}\"" if pages is not None else "") + ">\n"
    return head + (f"<context>{context}</context>\n" if context else "")


def _wrap(name: str, pages: int | None, context: str | None, body: str) -> str:
    return f"{_header(name, pages, context)}{body}\n</document>"


def _framed(name: str, pages: int, context: str | None, items: tuple) -> tuple:
    """A PDF's cached page items inside its document frame."""
    items = [dict(i) for i in items]
    head = _header(name, pages, context)
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
