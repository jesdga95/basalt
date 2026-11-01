"""Standalone adapter format behavior; no model, GPU, or third-party packages required."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import builtins
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from . import model_file
    from .gguf_reader import BLOCK_GEOMETRY, GGML_TYPES
    from .model_adapter import AdapterFile, write_adapter
else:
    import model_file
    from gguf_reader import BLOCK_GEOMETRY, GGML_TYPES
    from model_adapter import AdapterFile, write_adapter


class AdapterFormatTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "adapter.basalt"

    @staticmethod
    def manifest():
        return {
            "format": "basalt", "version": 1, "kind": "adapter",
            "adapter": {"version": 1, "entries": {
                "opaque entry ID": {
                    "target": {"tensor": "blk.0.attn_q.weight", "shape": [2, 2],
                               "region": {"kind": "whole"}},
                    "storage": {"kind": "replace", "type_id": 0,
                                "section": "adapter/data", "offset": 0, "bytes": 16},
                },
            }},
        }

    def fixture(self, manifest=None, *, raw=None, payload=bytes(range(16))):
        if raw is None:
            raw = json.dumps(self.manifest() if manifest is None else manifest).encode("utf-8")
        writer = model_file.ModelWriter(self.path)
        try:
            writer.add_bytes("meta", raw)
            writer.add_bytes("adapter/data", payload)
            writer.close()
        finally:
            if not writer.f.closed:
                writer.f.close()
        return self.path

    def test_raw_and_quantized_parts_preserve_exact_bytes(self):
        cases = [("f32", (3, 2), 0, 24), ("bf16", (3, 2), 30, 12),
                 ("q6", (256, 2), 14, 420), ("q8", (32, 3), 8, 102),
                 ("nvfp4", (64, 2), 40, 72)]
        payloads = {name: bytes((i * 37 + type_id) % 256 for i in range(size))
                    for name, _, type_id, size in cases}
        source = self.directory / "source.bin"
        source.write_bytes(b"prefix" + payloads["q6"] + b"suffix")
        calls = []

        def chunks():
            calls.append("consumed")
            yield payloads["q8"][:17]
            yield payloads["q8"][17:]

        parts = {"f32": payloads["f32"], "bf16": bytearray(payloads["bf16"]),
                 "q6": (source, 6, len(payloads["q6"])),
                 "q8": model_file.Stream(len(payloads["q8"]), chunks()),
                 "nvfp4": payloads["nvfp4"]}
        write_adapter(self.path, {name: (shape, type_id, parts[name])
                                  for name, shape, type_id, _ in cases}, name="replacement 雪")
        adapter = AdapterFile(self.path)
        container = model_file.ModelFile(self.path)
        self.assertTrue(container.verify("meta"))
        self.assertTrue(container.verify("adapter/data"))
        section_offset, _ = container.section("adapter/data")
        position = 0
        expected_section = bytearray()
        for name, shape, type_id, size in cases:
            padding = (-position) % model_file.TENSOR_ALIGN
            expected_section.extend(bytes(padding))
            position += padding
            tensor = adapter.find(name)
            self.assertEqual((tensor.tensor, tensor.shape, tensor.type_id), (name, shape, type_id))
            self.assertEqual(tensor.data, (str(self.path), section_offset + position, size))
            self.assertEqual(adapter.read_tensor(name), payloads[name])
            expected_section.extend(payloads[name])
            position += size
        self.assertEqual(container.read("adapter/data"), bytes(expected_section))
        self.assertEqual(calls, ["consumed"])
        self.assertEqual(source.read_bytes(), b"prefix" + payloads["q6"] + b"suffix")
        self.assertEqual(container.load_meta()["name"], "replacement 雪")
        with self.assertRaises(FrozenInstanceError):
            adapter.find("f32").type_id = 30
        self.assertIsNone(adapter.find("missing"))
        with self.assertRaises(KeyError):
            adapter.read_tensor("missing")

    def test_all_current_native_types_and_four_dimensions(self):
        replacements, expected = {}, {}
        for type_id in (0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                        16, 17, 18, 20, 21, 22, 23, 24, 29, 30, 40, 42):
            block, block_bytes = BLOCK_GEOMETRY[GGML_TYPES[type_id]]
            tensor = f"type.{type_id}"
            payload = bytes([type_id]) * (block_bytes * 2)
            replacements[tensor] = ((block, 1, 2, 1), type_id, payload)
            expected[tensor] = payload
        write_adapter(self.path, replacements)
        adapter = AdapterFile(self.path)
        self.assertEqual({name: adapter.read_tensor(name) for name in adapter.replacements}, expected)

    def test_more_than_sixty_four_entries_and_arbitrary_ids(self):
        replacements = {f"blk.{i}.weight": ((1,), 0, i.to_bytes(4, "little")) for i in range(79)}
        write_adapter(self.path, replacements)
        adapter = AdapterFile(self.path)
        self.assertEqual({name: adapter.read_tensor(name) for name in adapter.replacements},
                         {name: replacement[2] for name, replacement in replacements.items()})
        manifest = self.manifest()
        manifest["description"] = {"source": "metadata remains descriptive"}
        self.fixture(manifest)
        self.assertEqual(AdapterFile(self.path).read_tensor("blk.0.attn_q.weight"), bytes(range(16)))

    def test_invalid_manifests_and_geometry(self):
        entry = ("adapter", "entries", "opaque entry ID")
        target, storage = entry + ("target",), entry + ("storage",)
        changes = [
            (("format",), "other"), (("kind",), "model"),
            (("version",), 2), (("version",), True), (("version",), 1.0),
            (("adapter", "version"), 2), (("adapter", "version"), True),
            (("adapter", "version"), 1.0), (("adapter", "future"), "selector"),
            (("adapter", "entries"), {}), (("adapter", "entries"), []),
            (("adapter", "entries"), None), (("adapter",), None),
            (("name",), 1), (("name",), "bad\0name"),
            (entry + ("future",), True), (entry, []),
            (target + ("tensor",), ""), (target + ("tensor",), "bad\0tensor"),
            (target + ("tensor",), 12), (target + ("future",), "selector"),
            (target + ("region", "kind"), "slice"),
            (target + ("region", "offset",), 0), (target + ("region",), "whole"),
            (storage + ("kind",), "add"), (storage + ("kind",), "lora"),
            (storage + ("section",), "gguf"), (storage + ("scale",), 1),
            (storage + ("type_id",), True), (storage + ("type_id",), 0.0),
            (storage + ("type_id",), -1), (storage + ("type_id",), 999),
            (storage + ("type_id",), 15), (storage + ("type_id",), 19),
            (storage + ("type_id",), 39),
            (storage + ("offset",), 1), (storage + ("offset",), 4096),
            (storage + ("offset",), -4096), (storage + ("offset",), False),
            (storage + ("offset",), 0.0), (storage + ("offset",), (1 << 64)),
            (storage + ("offset",), (1 << 64) - 4096),
            (storage + ("bytes",), 15), (storage + ("bytes",), 17),
            (storage + ("bytes",), False), (storage + ("bytes",), 16.0),
            (storage + ("bytes",), -1), (storage + ("bytes",), (1 << 64)),
        ]
        for shape in ([], [0], [-1], [True], [2.0, 2], "2,2", [1] * 5,
                      [(1 << 64)], [(1 << 64) - 1, 2], [(1 << 63)]):
            changes.append((target + ("shape",), shape))
        for keys, value in changes:
            with self.subTest(keys=keys, value=value):
                manifest = self.manifest()
                node = manifest
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = value
                self.fixture(manifest)
                with self.assertRaises(ValueError):
                    AdapterFile(self.path)

    def test_missing_required_fields_and_nonobject_metadata(self):
        for keys in (("adapter",), ("version",), ("kind",), ("format",),
                     ("adapter", "version"), ("adapter", "entries"),
                     ("adapter", "entries", "opaque entry ID", "target", "shape"),
                     ("adapter", "entries", "opaque entry ID", "storage", "bytes")):
            with self.subTest(keys=keys):
                manifest = self.manifest()
                node = manifest
                for key in keys[:-1]:
                    node = node[key]
                del node[keys[-1]]
                self.fixture(manifest)
                with self.assertRaises(ValueError):
                    AdapterFile(self.path)
        for raw in (b"[]", b"null", b"{", b"\xff"):
            with self.subTest(raw=raw):
                self.fixture(raw=raw)
                with self.assertRaises(ValueError):
                    AdapterFile(self.path)

    def test_quantized_rows_and_nvfp4_auxiliary_tail_are_rejected(self):
        for type_id, shape, size in ((8, [16, 2], 34), (14, [128, 2], 210),
                                     (40, [64], 40)):
            with self.subTest(type_id=type_id, shape=shape):
                manifest = self.manifest()
                entry = manifest["adapter"]["entries"]["opaque entry ID"]
                entry["target"]["shape"] = shape
                entry["storage"].update(type_id=type_id, bytes=size)
                self.fixture(manifest, payload=bytes(size))
                with self.assertRaises(ValueError):
                    AdapterFile(self.path)

    def test_duplicate_targets_ids_and_json_keys_are_rejected(self):
        manifest = self.manifest()
        entries = manifest["adapter"]["entries"]
        entries["other"] = deepcopy(entries["opaque entry ID"])
        self.fixture(manifest)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            AdapterFile(self.path)
        raw = json.dumps(self.manifest(), separators=(",", ":"))
        for original, duplicate in (("\"version\":1", "\"version\":1,\"version\":1"),
                                    ("\"kind\":\"whole\"", "\"kind\":\"whole\",\"kind\":\"whole\""),
                                    ("\"bytes\":16", "\"bytes\":16,\"bytes\":16")):
            with self.subTest(key=original):
                self.fixture(raw=raw.replace(original, duplicate, 1).encode())
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    AdapterFile(self.path)
        one_entry = json.dumps(self.manifest()["adapter"]["entries"]["opaque entry ID"])
        duplicated_ids = ('{"format":"basalt","version":1,"kind":"adapter",'
                          '"adapter":{"version":1,"entries":{"id":' + one_entry
                          + ',"id":' + one_entry + '}}}')
        self.fixture(raw=duplicated_ids.encode())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            AdapterFile(self.path)
        for entry_id in ("", "bad\0id"):
            manifest = self.manifest()
            entries = manifest["adapter"]["entries"]
            entries[entry_id] = entries.pop("opaque entry ID")
            self.fixture(manifest)
            with self.assertRaises(ValueError):
                AdapterFile(self.path)

    def test_payload_extent_overflow_is_rejected(self):
        manifest = self.manifest()
        entry = manifest["adapter"]["entries"]["opaque entry ID"]
        entry["target"]["shape"] = [2048]
        entry["storage"].update(offset=(1 << 64) - 4096, bytes=8192)
        self.fixture(manifest, payload=bytes(8192))
        with self.assertRaises(ValueError):
            AdapterFile(self.path)

    def test_required_sections_cannot_be_omitted(self):
        for section in ("meta", "adapter/data"):
            with self.subTest(section=section):
                writer = model_file.ModelWriter(self.path)
                writer.add_bytes(section, json.dumps(self.manifest()).encode()
                                 if section == "meta" else bytes(16))
                writer.close()
                with self.assertRaises(ValueError):
                    AdapterFile(self.path)

    def test_corrupt_section_tables_are_rejected_before_payload_reads(self):
        for label in ("too many sections", "invalid file size", "duplicate section",
                      "empty section", "unterminated name", "unaligned section",
                      "out of bounds section", "overlapping sections"):
            with self.subTest(label=label):
                self.fixture()
                with self.path.open("r+b") as stream:
                    header = list(model_file.HEADER.unpack(stream.read(model_file.HEADER.size)))
                    first = list(model_file.ENTRY.unpack(stream.read(model_file.ENTRY.size)))
                    second = list(model_file.ENTRY.unpack(stream.read(model_file.ENTRY.size)))
                    if label == "too many sections":
                        header[2] = model_file.MAX_SECTIONS + 1
                    elif label == "invalid file size":
                        header[3] = model_file.HEADER_BYTES - 1
                    elif label == "duplicate section":
                        second[0] = first[0]
                    elif label == "empty section":
                        second[0] = bytes(64)
                    elif label == "unterminated name":
                        second[0] = b"x" * 64
                    elif label == "unaligned section":
                        second[1] += 1
                    elif label == "out of bounds section":
                        second[2] += 1
                    else:
                        second[1] = first[1]
                    stream.seek(0)
                    stream.write(model_file.HEADER.pack(*header))
                    stream.write(model_file.ENTRY.pack(*first))
                    stream.write(model_file.ENTRY.pack(*second))
                with self.assertRaises(ValueError):
                    AdapterFile(self.path)

    def test_short_payload_and_truncation_after_open(self):
        self.fixture(payload=bytes(15))
        with self.assertRaises(ValueError):
            AdapterFile(self.path)
        self.fixture()
        adapter = AdapterFile(self.path)
        with self.path.open("r+b") as stream:
            stream.truncate(self.path.stat().st_size - 1)
        with self.assertRaises(IOError):
            adapter.read_tensor("blk.0.attn_q.weight")
        with self.assertRaises(ValueError):
            AdapterFile(self.path)

    def test_writer_validation_preserves_destination(self):
        self.path.write_bytes(b"existing destination")
        invalid = [({}, None), ({"x": ((0,), 0, b"")}, None),
                   ({"x": ((16, 2), 8, bytes(34))}, None),
                   ({"x": ((1,), 0, b"bad")}, None),
                   ({"x": ((1,), True, bytes(4))}, None),
                   ({"x": ((1,), 0, model_file.Stream(True, []))}, None),
                   ({"x": ((1,), 0, object())}, None),
                   ({"x": ((1,), 0, bytes(4))}, "bad\0name"),
                   ({"bad\0tensor": ((1,), 0, bytes(4))}, None)]
        for replacements, name in invalid:
            with self.subTest(replacements=replacements, name=name):
                with self.assertRaises(ValueError):
                    write_adapter(self.path, replacements, name=name)
                self.assertEqual(self.path.read_bytes(), b"existing destination")
                self.assertEqual(set(self.directory.iterdir()), {self.path})

        class DuplicateMapping(dict):
            def items(self):
                return [("x", ((1,), 0, bytes(4))), ("x", ((1,), 0, bytes(4)))]

        with self.assertRaisesRegex(ValueError, "duplicate"):
            write_adapter(self.path, DuplicateMapping(x=None))
        self.assertEqual(self.path.read_bytes(), b"existing destination")

    def test_stream_failures_are_atomic_and_remove_temporary_files(self):
        def broken():
            yield b"ab"
            raise RuntimeError("producer failed")

        for chunks, error in (([b"abc"], IOError), ([b"abcde"], IOError),
                              (broken(), RuntimeError)):
            with self.subTest(error=error, chunks=type(chunks)):
                self.path.write_bytes(b"existing destination")
                with self.assertRaises(error):
                    write_adapter(self.path, {"x": ((1,), 0, model_file.Stream(4, chunks))})
                self.assertEqual(self.path.read_bytes(), b"existing destination")
                self.assertEqual(set(self.directory.iterdir()), {self.path})
        self.path.unlink()
        with self.assertRaises(IOError):
            write_adapter(self.path, {"x": ((1,), 0, model_file.Stream(4, []))})
        self.assertEqual(set(self.directory.iterdir()), set())

    def test_publication_failure_removes_temporary_file(self):
        self.path.write_bytes(b"existing destination")
        with patch.object(os, "replace", side_effect=PermissionError("publication denied")):
            with self.assertRaises(PermissionError):
                write_adapter(self.path, {"x": ((1,), 0, bytes(4))})
        self.assertEqual(self.path.read_bytes(), b"existing destination")
        self.assertEqual(set(self.directory.iterdir()), {self.path})

    def test_input_segments_validate_bounds_and_destination_aliases(self):
        source = self.directory / "source.bin"
        source.write_bytes(bytes(range(16)))
        self.path.write_bytes(b"existing destination")
        for offset, size, error in ((-1, 4, ValueError), (False, 4, ValueError),
                                     (0, 4.0, ValueError), (13, 4, IOError),
                                     ((1 << 64) - 1, 4, ValueError)):
            with self.subTest(offset=offset, size=size):
                with self.assertRaises(error):
                    write_adapter(self.path, {"x": ((1,), 0, (source, offset, size))})
                self.assertEqual(self.path.read_bytes(), b"existing destination")
        with self.assertRaises(ValueError):
            write_adapter(self.path, {"x": ((1,), 0, (self.path, 0, 4))})
        alias = self.directory / "alias.bin"
        os.link(self.path, alias)
        with self.assertRaises(ValueError):
            write_adapter(self.path, {"x": ((1,), 0, (alias, 0, 4))})
        self.assertEqual(self.path.read_bytes(), b"existing destination")
        self.assertEqual(source.read_bytes(), bytes(range(16)))
        self.assertEqual(set(self.directory.iterdir()), {source, alias, self.path})

    def test_source_short_read_during_write_is_atomic(self):
        source = self.directory / "source.bin"
        source.write_bytes(b"abcd")
        self.path.write_bytes(b"existing destination")

        def truncate_source():
            source.write_bytes(b"a")
            yield b"1234"

        replacements = {"first": ((1,), 0, model_file.Stream(4, truncate_source())),
                        "second": ((1,), 0, (source, 0, 4))}
        with self.assertRaises(IOError):
            write_adapter(self.path, replacements)
        self.assertEqual(self.path.read_bytes(), b"existing destination")
        self.assertEqual(set(self.directory.iterdir()), {self.path, source})

    def test_file_segments_are_copied_with_bounded_reads(self):
        source = self.directory / "source.bin"
        payload = bytes(range(256)) * 17
        source.write_bytes(b"prefix" + payload + b"suffix")
        original_open = builtins.open
        requests = []

        class BoundedReader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def seek(self, offset):
                return self.stream.seek(offset)

            def read(self, size=-1):
                if not 0 < size <= 97:
                    raise AssertionError(f"unbounded source read: {size}")
                requests.append(size)
                return self.stream.read(size)

        def bounded_open(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            if os.fspath(path) == str(source):
                return BoundedReader(stream)
            return stream

        with patch.object(model_file, "COPY_CHUNK", 97), patch.object(builtins, "open", bounded_open):
            write_adapter(self.path, {"x": ((len(payload) // 4,), 0, (source, 6, len(payload)))})
        self.assertEqual(sum(requests), len(payload))
        self.assertGreater(len(requests), 1)
        self.assertEqual(AdapterFile(self.path).read_tensor("x"), payload)
        self.assertEqual(source.read_bytes(), b"prefix" + payload + b"suffix")


if __name__ == "__main__":
    unittest.main()
