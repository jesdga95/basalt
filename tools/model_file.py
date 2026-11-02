"""Read and write the .basalt model file (the layout in include/basalt/core/model_format.hpp).

    w = ModelWriter(path)
    w.add_bytes("meta", json_bytes)
    w.add_file("experts", "/path/experts.bin")        # streamed and hashed while copied
    w.add_chunks("dense", chunk_iterator)             # streamed
    w.close()

    m = ModelFile(path)                               # version 2 only; versions=(1,) for the migration's input
    m.names(); m.section("meta") -> (offset, bytes); m.read("meta") -> bytes; m.verify("dense") -> bool
"""
from __future__ import annotations

import hashlib
import os
import struct

MAGIC = b"BASALTMF"
VERSION = 2
MIGRATION = "tools/migrations/basalt_v1_to_v2.py"
SECTION_ALIGN = 2 << 20
HEADER_BYTES = 64 << 10
TENSOR_ALIGN = 4096                            # inside the `gguf` section
HEADER = struct.Struct("<8sIIQ40s")           # 64 bytes
ENTRY = struct.Struct("<64sQQ32s16s")         # 128 bytes
MAX_SECTIONS = (HEADER_BYTES - HEADER.size) // ENTRY.size
COPY_CHUNK = 64 << 20


def _align(n: int) -> int:
    return (n + SECTION_ALIGN - 1) // SECTION_ALIGN * SECTION_ALIGN


class Stream:
    """A part of `add_parts` produced on the fly: `size` bytes yielded by the iterable `chunks`."""

    def __init__(self, size: int, chunks):
        self.size, self.chunks = size, chunks


class ModelWriter:
    """Sections are appended in order; the header and table are written by close()."""

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "wb")
        self.f.write(b"\0" * HEADER_BYTES)
        self.entries: list[tuple[str, int, int, bytes]] = []

    def _start(self, name: str) -> int:
        if len(name.encode()) >= 64:
            raise ValueError(f"section name too long: {name}")
        if any(e[0] == name for e in self.entries):
            raise ValueError(f"duplicate section: {name}")
        if len(self.entries) >= MAX_SECTIONS:
            raise ValueError("too many sections")
        off = _align(self.f.tell())
        self.f.write(b"\0" * (off - self.f.tell()))
        return off

    def add_bytes(self, name: str, data: bytes) -> None:
        off = self._start(name)
        self.f.write(data)
        self.entries.append((name, off, len(data), hashlib.sha256(data).digest()))

    def add_file(self, name: str, src: str, offset: int = 0, size: int | None = None) -> None:
        """Copy `size` bytes of `src` from `offset` (the whole rest of the file by default)."""
        off = self._start(name)
        h = hashlib.sha256()
        left = (os.path.getsize(src) - offset) if size is None else size
        n = 0
        with open(src, "rb") as s:
            s.seek(offset)
            while left > 0:
                b = s.read(min(COPY_CHUNK, left))
                if not b:
                    raise IOError(f"{src}: short read")
                h.update(b)
                self.f.write(b)
                left -= len(b)
                n += len(b)
        self.entries.append((name, off, n, h.digest()))

    def add_chunks(self, name: str, chunks) -> int:
        """One section from an iterator of byte chunks (streamed, hashed as written).  Returns its size."""
        off = self._start(name)
        h = hashlib.sha256()
        n = 0
        for b in chunks:
            h.update(b)
            self.f.write(b)
            n += len(b)
        self.entries.append((name, off, n, h.digest()))
        return n

    def add_parts(self, name: str, parts, align: int = TENSOR_ALIGN) -> list[int]:
        """One section of several parts, each at a multiple of `align` (zero padded). A part is bytes, a Stream or
        (src_path, offset, size); returns each part's offset. The hash covers the whole section."""
        off = self._start(name)
        h = hashlib.sha256()
        pos, offsets = 0, []

        def put(b: bytes) -> None:
            nonlocal pos
            h.update(b)
            self.f.write(b)
            pos += len(b)

        for part in parts:
            pad = (-pos) % align
            if pad:
                put(b"\0" * pad)
            offsets.append(pos)
            if isinstance(part, (bytes, bytearray)):
                put(bytes(part))
                continue
            if isinstance(part, Stream):
                n0 = pos
                for b in part.chunks:
                    put(b)
                if pos - n0 != part.size:
                    raise IOError(f"stream part: {pos - n0} bytes, expected {part.size}")
                continue
            src, o, size = part
            left = size
            with open(src, "rb") as s:
                s.seek(o)
                while left > 0:
                    b = s.read(min(COPY_CHUNK, left))
                    if not b:
                        raise IOError(f"{src}: short read")
                    put(b)
                    left -= len(b)
        self.entries.append((name, off, pos, h.digest()))
        return offsets

    def close(self) -> None:
        end = self.f.tell()
        self.f.seek(0)
        self.f.write(HEADER.pack(MAGIC, VERSION, len(self.entries), end, b"\0" * 40))
        for name, off, n, digest in self.entries:
            self.f.write(ENTRY.pack(name.encode(), off, n, digest, b"\0" * 16))
        self.f.close()


class ModelFile:
    def __init__(self, path: str, versions: tuple[int, ...] = (VERSION,)):
        self.path = path
        with open(path, "rb") as f:
            head = f.read(HEADER_BYTES)
        magic, version, n, file_bytes, _ = HEADER.unpack_from(head, 0)
        if magic != MAGIC:
            raise ValueError(f"{path}: not a .basalt model file")
        if version not in versions:
            hint = f"; upgrade it with {MIGRATION}" if version < VERSION else ""
            raise ValueError(f"{path}: .basalt version {version}, this build reads version {VERSION}{hint}")
        self.version = version
        if os.path.getsize(path) < file_bytes:
            raise ValueError(f"{path}: truncated ({os.path.getsize(path)} of {file_bytes} bytes)")
        self.sections: dict[str, tuple[int, int, bytes]] = {}
        for i in range(n):
            name, off, size, digest, _ = ENTRY.unpack_from(head, HEADER.size + i * ENTRY.size)
            self.sections[name.rstrip(b"\0").decode()] = (off, size, digest)

    def names(self) -> list[str]:
        return list(self.sections)

    def section(self, name: str) -> tuple[int, int]:
        off, size, _ = self.sections[name]
        return off, size

    def read(self, name: str) -> bytes:
        off, size = self.section(name)
        with open(self.path, "rb") as f:
            f.seek(off)
            return f.read(size)

    def verify(self, name: str) -> bool:
        off, size, digest = self.sections[name]
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            f.seek(off)
            left = size
            while left > 0:
                b = f.read(min(COPY_CHUNK, left))
                if not b:
                    return False
                h.update(b)
                left -= len(b)
        return h.digest() == digest
