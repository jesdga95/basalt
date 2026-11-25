"""The sections the vision sidecar reads from a .basalt file:

    vision/mmproj       the projector GGUF, whole
    vision/text_vocab   a vocab-only GGUF of the text model (metadata without tensors or split keys), used for the
                        image-marker tokenizer

vision_sections(mmproj_path, native_paths) -> [(section_name, bytes_or_None, src_path_or_None, offset, size)]"""
from __future__ import annotations

import os
import struct

GGUF_MAGIC = b"GGUF"
ALIGNMENT = 32
_INITIAL_READ = 8 << 20
# value type id -> fixed size; strings and arrays are walked
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _check_bounds(buf: bytes, end: int) -> int:
    if end > len(buf):
        raise struct.error("truncated GGUF metadata")
    return end


def _skip_value(buf: bytes, pos: int, vtype: int) -> int:
    if vtype in _FIXED:
        return _check_bounds(buf, pos + _FIXED[vtype])
    if vtype == 8:
        (n,) = struct.unpack_from("<Q", buf, pos)
        return _check_bounds(buf, pos + 8 + n)
    if vtype == 9:
        etype, count = struct.unpack_from("<IQ", buf, pos)
        pos += 12
        if etype in _FIXED:
            return _check_bounds(buf, pos + _FIXED[etype] * count)
        for _ in range(count):
            pos = _skip_value(buf, pos, etype)
        return pos
    raise ValueError(f"unknown GGUF value type {vtype}")


def _kv_entries(buf: bytes):
    """(key, raw bytes of the whole entry) for each metadata entry of a GGUF v3 header in `buf`."""
    if buf[:4] != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version, _n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
    if version != 3:
        raise ValueError(f"GGUF v{version}: only v3 is handled")
    pos = 24
    for _ in range(n_kv):
        start = pos
        (klen,) = struct.unpack_from("<Q", buf, pos)
        key_end = _check_bounds(buf, pos + 8 + klen)
        key = buf[pos + 8:key_end].decode("utf-8", "replace")
        pos = key_end
        (vtype,) = struct.unpack_from("<I", buf, pos)
        pos = _skip_value(buf, pos + 4, vtype)
        yield key, buf[start:pos]


def _read_header_prefix(path: str) -> bytes:
    """The start of a GGUF up to the end of its metadata; tensor info and data follow and are not read."""
    size = _INITIAL_READ
    with open(path, "rb") as f:
        while True:
            f.seek(0)
            buf = f.read(size)
            try:
                list(_kv_entries(buf))
                return buf
            except struct.error:
                if len(buf) < size:
                    raise
                size *= 4


def vocab_only_gguf(path: str) -> bytes | None:
    """The GGUF at `path` with its metadata only (no tensors, no split.* keys), or None when it has no tokenizer."""
    buf = _read_header_prefix(path)
    entries = list(_kv_entries(buf))
    if not any(k == "tokenizer.ggml.tokens" for k, _ in entries):
        return None
    kept = [raw for key, raw in entries if not key.startswith("split.")]
    head = GGUF_MAGIC + struct.pack("<IQQ", 3, 0, len(kept))
    body = head + b"".join(kept)
    return body + b"\0" * (-len(body) % ALIGNMENT)


def vision_sections(mmproj_path, native_paths):
    out = [("vision/mmproj", None, str(mmproj_path), 0, os.path.getsize(mmproj_path))]
    for p in native_paths:
        data = vocab_only_gguf(str(p))
        if data is not None:
            out.append(("vision/text_vocab", data, None, 0, len(data)))
            return out
    raise ValueError("none of the source GGUF files carries the tokenizer metadata the vision sidecar needs")
