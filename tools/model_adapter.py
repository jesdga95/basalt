"""Standalone .basalt whole-tensor replacements, without runtime model binding."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
import tempfile

if __package__:
    from .gguf_reader import BLOCK_GEOMETRY, GGML_TYPES
    from .model_file import HEADER_BYTES, SECTION_ALIGN, TENSOR_ALIGN, ModelFile, ModelWriter, Stream
else:
    from gguf_reader import BLOCK_GEOMETRY, GGML_TYPES
    from model_file import HEADER_BYTES, SECTION_ALIGN, TENSOR_ALIGN, ModelFile, ModelWriter, Stream


_UINT64_MAX = (1 << 64) - 1
_SUPPORTED_TYPES = frozenset((0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                              16, 17, 18, 20, 21, 22, 23, 24, 29, 30, 40, 42))
_DATA_SECTION = "adapter/data"


def _uint64(value, field: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT64_MAX:
        raise ValueError(f"{field}: expected a uint64 integer")
    return value


def _text(value, field: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not empty) or "\0" in value:
        raise ValueError(f"{field}: expected {'a' if empty else 'a nonempty'} string without NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field}: invalid UTF-8 string") from exc
    return value


def _object(value, keys, field: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{field}: expected exactly {', '.join(keys)}")
    return value


def _geometry(shape, type_id) -> tuple[tuple[int, ...], int]:
    if not isinstance(shape, (list, tuple)) or not 1 <= len(shape) <= 4:
        raise ValueError("target.shape: expected 1..4 dimensions")
    _uint64(type_id, "storage.type_id")
    if type_id not in _SUPPORTED_TYPES:
        raise ValueError(f"unsupported adapter type_id: {type_id}")
    elements = 1
    for extent in shape:
        _uint64(extent, "target.shape extent")
        if extent == 0:
            raise ValueError("target.shape: zero extent")
        elements = _uint64(elements * extent, "target.shape product")
    block, block_bytes = BLOCK_GEOMETRY[GGML_TYPES[type_id]]
    if shape[0] % block:
        raise ValueError("target.shape: fastest axis must contain whole dtype blocks")
    size = _uint64(elements // block * block_bytes, "tensor byte count")
    return tuple(shape), size


@dataclass(frozen=True)
class AdapterTensor:
    """Validated replacement range; shape is GGUF order and data offset is absolute."""

    tensor: str
    shape: tuple[int, ...]
    type_id: int
    data: tuple[str, int, int]


class AdapterFile:
    """Read replacements only; selectors and storage methods require explicit support."""

    def __init__(self, path):
        source = os.path.abspath(os.fsdecode(path))
        model = ModelFile(source)
        if "meta" not in model.names() or _DATA_SECTION not in model.names():
            raise ValueError("adapter requires meta and adapter/data sections")
        meta = model.load_meta()
        if not isinstance(meta, dict):
            raise ValueError("adapter metadata must be an object")
        if meta.get("format") != "basalt" or meta.get("kind") != "adapter":
            raise ValueError("not a basalt adapter artifact")
        if _uint64(meta.get("version"), "version") != 1:
            raise ValueError("unsupported metadata version")
        if "name" in meta:
            _text(meta["name"], "name", empty=True)
        adapter = _object(meta.get("adapter"), ("version", "entries"), "adapter")
        if _uint64(adapter["version"], "adapter.version") != 1:
            raise ValueError("unsupported adapter version")
        entries = adapter["entries"]
        if not isinstance(entries, dict) or not entries:
            raise ValueError("adapter.entries must be a nonempty object")
        section_offset, section_size = model.section(_DATA_SECTION)
        replacements = {}
        for entry_id, entry in entries.items():
            _text(entry_id, "entry ID")
            entry = _object(entry, ("target", "storage"), "entry")
            target = _object(entry["target"], ("tensor", "shape", "region"), "target")
            tensor = _text(target["tensor"], "target.tensor")
            if tensor in replacements:
                raise ValueError(f"duplicate replacement target: {tensor}")
            region = _object(target["region"], ("kind",), "target.region")
            if region["kind"] != "whole":
                raise ValueError("unsupported target region")
            storage = _object(entry["storage"],
                              ("kind", "type_id", "section", "offset", "bytes"), "storage")
            if storage["kind"] != "replace" or storage["section"] != _DATA_SECTION:
                raise ValueError("unsupported adapter storage")
            shape, size = _geometry(target["shape"], storage["type_id"])
            offset = _uint64(storage["offset"], "storage.offset")
            if offset % TENSOR_ALIGN:
                raise ValueError("storage.offset must be tensor-aligned")
            if _uint64(storage["bytes"], "storage.bytes") != size:
                raise ValueError("storage.bytes disagrees with tensor geometry")
            end = _uint64(offset + size, "payload extent")
            if end > section_size:
                raise ValueError("adapter payload extends beyond adapter/data")
            absolute = _uint64(section_offset + offset, "absolute payload offset")
            _uint64(absolute + size, "absolute payload extent")
            replacements[tensor] = AdapterTensor(tensor, shape, storage["type_id"],
                                                 (source, absolute, size))
        self.replacements = replacements

    def find(self, tensor: str) -> AdapterTensor | None:
        return self.replacements.get(tensor)

    def read_tensor(self, tensor: str) -> bytes:
        source, offset, size = self.replacements[tensor].data
        with open(source, "rb") as stream:
            stream.seek(offset)
            payload = stream.read(size)
        if len(payload) != size:
            raise IOError(f"{source}: short read for {tensor}")
        return payload


def _part(part, destination: str, expected: int):
    if isinstance(part, (bytes, bytearray)):
        size = len(part)
        part = bytes(part)
    elif isinstance(part, Stream):
        size = _uint64(part.size, "stream size")
        part = Stream(size, part.chunks)
    elif isinstance(part, tuple) and len(part) == 3:
        source, offset, size = part
        source = os.path.abspath(os.fsdecode(source))
        offset = _uint64(offset, "source offset")
        size = _uint64(size, "source size")
        end = _uint64(offset + size, "source extent")
        if (os.path.normcase(os.path.realpath(source)) == os.path.normcase(os.path.realpath(destination))
                or (os.path.exists(destination) and os.path.samefile(source, destination))):
            raise ValueError("adapter destination is also an input segment")
        if end > os.path.getsize(source):
            raise IOError(f"{source}: source segment exceeds file size")
        part = (source, offset, size)
    else:
        raise ValueError("part must be bytes, bytearray, Stream, or (source, offset, size)")
    if size != expected:
        raise ValueError(f"part has {size} bytes, tensor geometry requires {expected}")
    return part


def write_adapter(path, replacements, *, name=None) -> None:
    """Atomically write raw replacement parts, streaming each source exactly once."""
    destination = os.path.abspath(os.fsdecode(path))
    if not isinstance(replacements, Mapping) or not replacements:
        raise ValueError("replacements must be a nonempty mapping")
    if name is not None:
        _text(name, "name", empty=True)
    entries, parts, targets = {}, [], set()
    position = 0
    for tensor, replacement in replacements.items():
        _text(tensor, "target.tensor")
        if tensor in targets:
            raise ValueError(f"duplicate replacement target: {tensor}")
        targets.add(tensor)
        if not isinstance(replacement, (tuple, list)) or len(replacement) != 3:
            raise ValueError("replacement must be (shape, type_id, part)")
        shape, type_id, part = replacement
        shape, size = _geometry(shape, type_id)
        part = _part(part, destination, size)
        offset = _uint64(position + (-position) % TENSOR_ALIGN, "storage.offset")
        position = _uint64(offset + size, "adapter/data size")
        entries[str(len(entries))] = {
            "target": {"tensor": tensor, "shape": list(shape), "region": {"kind": "whole"}},
            "storage": {"kind": "replace", "type_id": type_id, "section": _DATA_SECTION,
                        "offset": offset, "bytes": size},
        }
        parts.append(part)
    meta = {"format": "basalt", "version": 1, "kind": "adapter",
            "adapter": {"version": 1, "entries": entries}}
    if name is not None:
        meta["name"] = name
    encoded = json.dumps(meta, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    meta_offset = HEADER_BYTES + (-HEADER_BYTES) % SECTION_ALIGN
    data_offset = meta_offset + len(encoded)
    data_offset += (-data_offset) % SECTION_ALIGN
    _uint64(data_offset + position, "adapter file size")

    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(destination)}.", suffix=".tmp",
                                     dir=os.path.dirname(destination))
    writer = None
    try:
        os.close(fd)
        writer = ModelWriter(temporary)
        writer.add_bytes("meta", encoded)
        writer.add_parts(_DATA_SECTION, parts, align=TENSOR_ALIGN)
        writer.close()
        os.replace(temporary, destination)
    finally:
        try:
            if writer is not None and not writer.f.closed:
                writer.f.close()
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
