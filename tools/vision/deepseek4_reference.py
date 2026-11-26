#!/usr/bin/env python3
"""Write a small DeepSeek-V4 Vision/Aligner reference block from the source HF checkpoint."""

import argparse
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import safe_open


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="DeepSeek-V4 HF checkpoint directory")
    parser.add_argument("--output", required=True, type=Path, help="Output [int32 rows,width][float32 payload] file")
    args = parser.parse_args()

    source = args.source.resolve()
    inference = source / "inference"
    sys.path.insert(0, str(inference))
    from vision import Aligner, ViT

    config = json.loads((source / "config.json").read_text())
    config["dim"] = config["hidden_size"]
    model_args = SimpleNamespace(**config)
    vision = ViT(model_args).eval()
    aligner = Aligner(model_args).eval()

    with (source / "model.safetensors.index.json").open() as stream:
        weight_map = json.load(stream)["weight_map"]
    names = [name for name in weight_map if name.startswith(("vision.", "aligner."))]
    shards = sorted({weight_map[name] for name in names})
    vision_state: dict[str, torch.Tensor] = {}
    aligner_state: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name.startswith("vision."):
                    vision_state[name.removeprefix("vision.")] = handle.get_tensor(name)
                elif name.startswith("aligner."):
                    aligner_state[name.removeprefix("aligner.")] = handle.get_tensor(name)

    if vision_state.keys() != vision.state_dict().keys() or aligner_state.keys() != aligner.state_dict().keys():
        raise ValueError("source Vision/Aligner tensor names do not match the reference implementation")
    vision.load_state_dict(vision_state)
    aligner.load_state_dict(aligner_state)

    device = "cuda"
    torch.set_default_device(device)
    vision = vision.to(device=device, dtype=torch.bfloat16)
    aligner = aligner.to(device=device, dtype=torch.bfloat16)
    patch_values = 9 * 3 * 14 * 14
    patches = (((torch.arange(patch_values, device=device) % 17) - 8).float() / 8).to(torch.bfloat16)
    patches = patches.view(9, 3 * 14 * 14)
    with torch.inference_mode():
        output = aligner(vision(patches, 3, 3), 3, 3).float().cpu().contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(struct.pack("ii", output.shape[0], output.shape[1]))
        stream.write(output.numpy().tobytes())
    print(f"wrote {output.shape[0]}x{output.shape[1]} source reference from {len(shards)} shards")


if __name__ == "__main__":
    main()
