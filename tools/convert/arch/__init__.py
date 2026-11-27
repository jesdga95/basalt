"""Per-architecture conversion rules for tools/convert/convert.py, keyed by the GGUF's `general.architecture`.

An architecture says how its source tensors are stored in the pack (the float form the engine reads, tensors that get
a section of their own), which optional sections it carries, and the license, sampling defaults and extra `meta` keys.
Everything else (tokenizer, dense/gguf, experts, container) is shared.
"""
from __future__ import annotations

import re
from pathlib import Path


def gguf_stem(path) -> str:
    """The file name alone, with a trailing `.gguf` (any case) and a shard suffix `-<digits>-of-<digits>` (any case) stripped."""
    name = re.sub(r"\.gguf$", "", Path(path).name, flags=re.IGNORECASE)
    return re.sub(r"-\d+-of-\d+$", "", name, flags=re.IGNORECASE)


class Arch:
    name = ""
    # float form the engine reads each dense tensor in (name without `blk.N.`); empty = tensors are stored as they are
    form_table: dict[str, str] = {}
    # tensors kept out of the dense pool (they are served from `gguf` or from a section of their own)
    not_in_dense: frozenset = frozenset()
    # tensor name -> section that holds it alone (2 MiB aligned); such tensors are left out of `gguf`
    own_sections: dict[str, str] = {}

    def own_section_bytes(self, name, tensor):
        """Bytes of a tensor that gets a section of its own (default: the GGUF block geometry)."""
        return tensor.expected_bytes()

    # drafter, hc8 and vision sections
    side_sections = True
    # only the vision projector section (vision/mmproj), written when --mmproj is given
    mmproj_only = False
    projector_type = ""
    # needs the model's Hugging Face directory (--hf-dir) for its license text and config.json
    needs_hf_dir = False
    # takes a DSpark drafter GGUF (--dspark) as the dspark/gguf section
    dspark_ok = False
    # the model's sampling defaults, for a GGUF that does not carry them (llama.cpp's general.sampling.* keys)
    sampling_kv: dict = {}

    def form_of(self, name: str):
        return self.form_table.get(re.sub(r"^blk\.\d+\.", "", name))

    def native(self, name: str, type_name: str) -> bool:
        """A quantized tensor the engine reads from the source bytes as they are."""
        return self.form_of(name) is None

    def mtp_layers(self, kv: dict) -> list[int]:
        """Layers of the source that are the drafter's: written to mtp/* sections instead of the main ones."""
        return []

    def hf_meta(self, hf_dir) -> dict:
        """Meta keys read from the Hugging Face directory (--hf-dir): the license text and the config."""
        return {}

    def license(self, kv: dict) -> dict:
        raise NotImplementedError

    def default_name(self, first_gguf, kv: dict, sections_from: bool = False) -> str:
        raise NotImplementedError

    def meta_extra(self, kv: dict, tensors: dict, own: dict) -> dict:
        """Keys added to `meta`: `tensors` are the `gguf` tensors' entries, `own` the own-section tensors' ones."""
        return {}

    def check(self, kv: dict, own: dict) -> None:
        """Raise ConvertError when the source does not fit this architecture."""


REGISTRY: dict[str, Arch] = {}


def register(arch: Arch, *names: str) -> None:
    for n in names or (arch.name,):
        REGISTRY[n] = arch


def for_architecture(name) -> Arch:
    """The rules for `general.architecture`; an architecture without its own entry is converted as Flash-Next."""
    from . import qwen4exp, deepseek41, hy4flash  # noqa: F401  (register themselves)
    return REGISTRY.get(name) or REGISTRY["qwen4exp"]
