#!/usr/bin/env python3
"""Convert Wan 2.2 UMT5-XXL .pth weights to safetensors format.

Usage:
    python3 scripts/convert-wan-t5-to-safetensors.py \
        /Volumes/Bolt/Models/wan22-i2v/models_t5_umt5-xxl-enc-bf16.pth \
        /Volumes/Bolt/Models/wan22-i2v/models_t5_umt5-xxl-enc-bf16.safetensors

The Wan .pth checkpoint uses its own naming convention (blocks.N.attn.q, etc.)
which maps directly to the Swift module hierarchy. No key remapping is needed.

Requires: pip install torch safetensors
"""

import sys
import torch
from safetensors.torch import save_file


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.pth> <output.safetensors>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading {input_path}...")
    state_dict = torch.load(input_path, map_location="cpu", weights_only=True)

    print(f"Found {len(state_dict)} tensors:")
    total_params = 0
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        params = tensor.numel()
        total_params += params
        print(f"  {key}: {list(tensor.shape)} {tensor.dtype}")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Saving to {output_path}...")

    # safetensors requires contiguous tensors
    contiguous = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(contiguous, output_path)

    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! Output size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
