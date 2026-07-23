#!/usr/bin/env python3
"""
Pre-convert Wan 2.2 I2V PyTorch safetensors weights to MLX-native layout.

Eliminates runtime transposition as a variable in the purple/magenta color
cast investigation. Produces weights that can be loaded directly by the
Swift code without any transposition.

Components converted:
  1. VAE (Wan2.1_VAE.safetensors)
     - Conv3d weights (5D): transpose (0,2,3,4,1) -- PyTorch [O,I,kT,kH,kW] -> MLX [O,kT,kH,kW,I]
     - Conv2d weights (4D): transpose (0,2,3,1)   -- PyTorch [O,I,kH,kW]     -> MLX [O,kH,kW,I]
     - Key remapping: insert "layers" after container names before digit indices
       Containers: downsamples, upsamples, middle, head, residual, resample

  2. Transformer shards (high_noise_model/, low_noise_model/)
     - Only patch_embedding.weight (5D Conv3d): transpose (0,2,3,4,1)
     - Key remapping: insert "layers" after container names before digit indices
       Containers: text_embedding, time_embedding, time_projection, ffn

  3. T5 text encoder (models_t5_umt5-xxl-enc-bf16.safetensors)
     - No transpositions (all linear/embedding)
     - Key remapping: ffn.gate.0.weight -> ffn.gate.weight

Usage:
    python3 convert_to_mlx.py [--src ~/Models/wan22-i2v] [--dst ~/Models/wan22-i2v-mlx]
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


# ---------------------------------------------------------------------------
# Key remapping
# ---------------------------------------------------------------------------

VAE_CONTAINERS = {"downsamples", "upsamples", "middle", "head", "residual", "resample"}
TRANSFORMER_CONTAINERS = {"text_embedding", "time_embedding", "time_projection", "ffn"}


def remap_key_insert_layers(key: str, containers: set[str]) -> str:
    """Insert 'layers' between a container name and a following digit index."""
    parts = key.split(".")
    remapped = []
    for i, part in enumerate(parts):
        remapped.append(part)
        if part in containers and i + 1 < len(parts) and parts[i + 1].isdigit():
            remapped.append("layers")
    return ".".join(remapped)


def remap_t5_gate_key(key: str) -> str:
    """Remap ffn.gate.0.weight -> ffn.gate.weight for T5 encoder."""
    return key.replace(".ffn.gate.0.", ".ffn.gate.")


# ---------------------------------------------------------------------------
# Transposition helpers
# ---------------------------------------------------------------------------

def transpose_conv3d(tensor: np.ndarray) -> np.ndarray:
    """PyTorch [O, I, kT, kH, kW] -> MLX [O, kT, kH, kW, I]"""
    return np.ascontiguousarray(np.transpose(tensor, (0, 2, 3, 4, 1)))


def transpose_conv2d(tensor: np.ndarray) -> np.ndarray:
    """PyTorch [O, I, kH, kW] -> MLX [O, kH, kW, I]"""
    return np.ascontiguousarray(np.transpose(tensor, (0, 2, 3, 1)))


# ---------------------------------------------------------------------------
# VAE conversion
# ---------------------------------------------------------------------------

def convert_vae(src_path: Path, dst_path: Path) -> dict:
    """Convert VAE weights with transposition and key remapping."""
    stats = {"transposed_5d": 0, "transposed_4d": 0, "kept": 0, "total_elements": 0}
    examples = []

    print(f"\n{'='*70}")
    print(f"VAE: {src_path.name}")
    print(f"{'='*70}")

    output_tensors = {}
    with safe_open(str(src_path), framework="numpy") as f:
        keys = sorted(f.keys())
        print(f"  Keys: {len(keys)}")

        for key in keys:
            tensor = f.get_tensor(key)
            orig_shape = tensor.shape
            stats["total_elements"] += tensor.size

            # Transpose convolution weights
            if key.endswith(".weight") and not key.endswith(".bias"):
                if tensor.ndim == 5:
                    tensor = transpose_conv3d(tensor)
                    stats["transposed_5d"] += 1
                    if len(examples) < 5:
                        examples.append((key, orig_shape, tensor.shape, "Conv3d"))
                elif tensor.ndim == 4:
                    tensor = transpose_conv2d(tensor)
                    stats["transposed_4d"] += 1
                    if len(examples) < 5:
                        examples.append((key, orig_shape, tensor.shape, "Conv2d"))
                else:
                    stats["kept"] += 1
            else:
                stats["kept"] += 1

            # Remap key
            new_key = remap_key_insert_layers(key, VAE_CONTAINERS)
            output_tensors[new_key] = tensor

    # Show examples
    if examples:
        print(f"\n  Example transpositions:")
        for key, orig, new, typ in examples:
            new_key = remap_key_insert_layers(key, VAE_CONTAINERS)
            print(f"    {key}")
            print(f"      {orig} -> {new}  ({typ})")
            if new_key != key:
                print(f"      key remapped -> {new_key}")

    # Save
    print(f"\n  Saving to {dst_path}...")
    save_file(output_tensors, str(dst_path))
    file_size = dst_path.stat().st_size
    print(f"  Saved: {file_size / 1024**2:.1f} MB")

    print(f"\n  Stats: {stats['transposed_5d']} Conv3d transposed, "
          f"{stats['transposed_4d']} Conv2d transposed, "
          f"{stats['kept']} kept as-is")
    print(f"  Total elements: {stats['total_elements']:,}")

    return stats


# ---------------------------------------------------------------------------
# Transformer conversion (sharded)
# ---------------------------------------------------------------------------

def convert_transformer_shards(src_dir: Path, dst_dir: Path, label: str) -> dict:
    """Convert transformer shards with transposition and key remapping."""
    stats = {"transposed_5d": 0, "transposed_4d": 0, "kept": 0,
             "total_elements": 0, "shards": 0}

    index_path = src_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.exists():
        print(f"\n  WARNING: No index file at {index_path}, skipping")
        return stats

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    print(f"\n{'='*70}")
    print(f"Transformer ({label}): {len(weight_map)} keys across {len(shard_files)} shards")
    print(f"{'='*70}")

    # Copy config.json
    config_src = src_dir / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, dst_dir / "config.json")

    # Build new weight map with remapped keys
    new_weight_map = {}

    for shard_idx, shard_file in enumerate(shard_files):
        shard_path = src_dir / shard_file
        print(f"\n  Shard {shard_idx + 1}/{len(shard_files)}: {shard_file}")

        # Keys belonging to this shard
        shard_keys = [k for k, v in weight_map.items() if v == shard_file]
        print(f"    Keys: {len(shard_keys)}")

        output_tensors = {}
        shard_transposed = 0
        t0 = time.time()

        with safe_open(str(shard_path), framework="numpy") as f:
            for key in sorted(shard_keys):
                tensor = f.get_tensor(key)
                orig_shape = tensor.shape
                stats["total_elements"] += tensor.size

                # Only patch_embedding.weight is Conv3d in the transformer
                if key == "patch_embedding.weight" and tensor.ndim == 5:
                    print(f"    Transposing: {key} {orig_shape}", end="")
                    tensor = transpose_conv3d(tensor)
                    print(f" -> {tensor.shape}")
                    stats["transposed_5d"] += 1
                    shard_transposed += 1
                else:
                    stats["kept"] += 1

                # Remap key
                new_key = remap_key_insert_layers(key, TRANSFORMER_CONTAINERS)
                output_tensors[new_key] = tensor
                new_weight_map[new_key] = shard_file

        # Save shard
        out_path = dst_dir / shard_file
        save_file(output_tensors, str(out_path))
        elapsed = time.time() - t0
        file_size = out_path.stat().st_size
        print(f"    Saved: {file_size / 1024**3:.2f} GB in {elapsed:.1f}s "
              f"({shard_transposed} transposed)")
        stats["shards"] += 1

    # Write new index
    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    index_out = dst_dir / "diffusion_pytorch_model.safetensors.index.json"
    with open(index_out, "w") as f:
        json.dump(new_index, f, indent=2, sort_keys=True)
    print(f"\n  Index written: {index_out}")
    print(f"  Stats: {stats['transposed_5d']} Conv3d transposed, "
          f"{stats['kept']} kept as-is across {stats['shards']} shards")

    return stats


# ---------------------------------------------------------------------------
# T5 encoder conversion
# ---------------------------------------------------------------------------

def convert_t5(src_path: Path, dst_path: Path) -> dict:
    """Convert T5 encoder weights with key remapping only (no transposition)."""
    stats = {"remapped_gate": 0, "kept": 0, "total_elements": 0}

    print(f"\n{'='*70}")
    print(f"T5 Encoder: {src_path.name}")
    print(f"{'='*70}")

    # T5 is BF16 which numpy doesn't support directly.
    # safetensors can load it but numpy will see it as uint16.
    # We need to handle this carefully -- just pass through as bytes.
    # Actually, safetensors.numpy handles bf16 by converting to float32.
    # Let's try with torch framework if available, else raw copy.
    try:
        output_tensors = {}
        with safe_open(str(src_path), framework="numpy") as f:
            keys = sorted(f.keys())
            print(f"  Keys: {len(keys)}")

            for key in keys:
                tensor = f.get_tensor(key)
                stats["total_elements"] += tensor.size

                # T5 has no convolutions -- all linear/embedding (2D or 1D)
                # Only remap gate keys
                new_key = remap_t5_gate_key(key)
                if new_key != key:
                    stats["remapped_gate"] += 1
                    print(f"  Remapped: {key} -> {new_key}")
                else:
                    stats["kept"] += 1

                output_tensors[new_key] = tensor

        save_file(output_tensors, str(dst_path))
        file_size = dst_path.stat().st_size
        print(f"  Saved: {file_size / 1024**3:.2f} GB")
        print(f"  Stats: {stats['remapped_gate']} gate keys remapped, "
              f"{stats['kept']} kept as-is")

    except Exception as e:
        # If numpy can't handle bf16, just copy the file and note it
        print(f"  WARNING: numpy can't load bf16 tensors: {e}")
        print(f"  Falling back to raw file copy + key remap via safetensors metadata...")

        # Try with torch if available
        try:
            import torch
            from safetensors.torch import load_file, save_file as torch_save_file

            tensors = load_file(str(src_path))
            output_tensors = {}
            for key in sorted(tensors.keys()):
                new_key = remap_t5_gate_key(key)
                if new_key != key:
                    stats["remapped_gate"] += 1
                    print(f"  Remapped: {key} -> {new_key}")
                else:
                    stats["kept"] += 1
                output_tensors[new_key] = tensors[new_key] if new_key in tensors else tensors[key]
                stats["total_elements"] += tensors[key].numel()

            torch_save_file(output_tensors, str(dst_path))
            file_size = dst_path.stat().st_size
            print(f"  Saved: {file_size / 1024**3:.2f} GB (via torch)")

        except ImportError:
            # Last resort: copy file as-is, document that gate keys need runtime remap
            print(f"  WARNING: torch not available either. Copying file as-is.")
            print(f"  Gate key remapping will still need to happen at runtime.")
            shutil.copy2(src_path, dst_path)
            file_size = dst_path.stat().st_size
            print(f"  Copied: {file_size / 1024**3:.2f} GB (no key remapping applied)")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert Wan 2.2 I2V weights to MLX layout")
    parser.add_argument("--src", type=Path, default=Path.home() / "Models" / "wan22-i2v",
                        help="Source model directory")
    parser.add_argument("--dst", type=Path, default=Path.home() / "Models" / "wan22-i2v-mlx",
                        help="Destination directory for MLX-layout weights")
    parser.add_argument("--skip-transformer", action="store_true",
                        help="Skip transformer shard conversion (VAE + T5 only)")
    parser.add_argument("--skip-t5", action="store_true",
                        help="Skip T5 encoder conversion")
    parser.add_argument("--vae-only", action="store_true",
                        help="Convert VAE only")
    args = parser.parse_args()

    src = args.src
    dst = args.dst

    if not src.exists():
        print(f"ERROR: Source directory not found: {src}")
        sys.exit(1)

    print(f"Source:      {src}")
    print(f"Destination: {dst}")
    print(f"Time:        {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directories
    dst.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    t_start = time.time()

    # --- VAE ---
    vae_src = src / "Wan2.1_VAE.safetensors"
    if vae_src.exists():
        vae_dst = dst / "Wan2.1_VAE.safetensors"
        all_stats["vae"] = convert_vae(vae_src, vae_dst)
    else:
        print(f"\nWARNING: VAE not found at {vae_src}")

    if args.vae_only:
        _print_summary(all_stats, t_start)
        return

    # --- T5 Encoder ---
    if not args.skip_t5:
        t5_src = src / "models_t5_umt5-xxl-enc-bf16.safetensors"
        if t5_src.exists():
            t5_dst = dst / "models_t5_umt5-xxl-enc-bf16.safetensors"
            all_stats["t5"] = convert_t5(t5_src, t5_dst)
        else:
            print(f"\nWARNING: T5 encoder not found at {t5_src}")

    # --- Transformer (high_noise_model) ---
    if not args.skip_transformer:
        for model_name in ["high_noise_model", "low_noise_model"]:
            model_src = src / model_name
            if model_src.exists():
                model_dst = dst / model_name
                model_dst.mkdir(parents=True, exist_ok=True)
                all_stats[model_name] = convert_transformer_shards(
                    model_src, model_dst, model_name
                )
            else:
                print(f"\nWARNING: {model_name} not found at {model_src}")

    # --- Copy configuration.json if exists ---
    config_src = src / "configuration.json"
    if config_src.exists():
        shutil.copy2(config_src, dst / "configuration.json")
        print(f"\nCopied configuration.json")

    _print_summary(all_stats, t_start)


def _print_summary(all_stats: dict, t_start: float):
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print()

    total_elements = 0
    total_transposed = 0
    for name, stats in all_stats.items():
        t5d = stats.get("transposed_5d", 0)
        t4d = stats.get("transposed_4d", 0)
        kept = stats.get("kept", 0)
        elems = stats.get("total_elements", 0)
        gate = stats.get("remapped_gate", 0)
        total_elements += elems
        total_transposed += t5d + t4d

        parts = []
        if t5d: parts.append(f"{t5d} Conv3d")
        if t4d: parts.append(f"{t4d} Conv2d")
        if gate: parts.append(f"{gate} gate keys")
        parts.append(f"{kept} kept")

        print(f"  {name:25s}: {', '.join(parts)}  ({elems:,} elements)")

    total_gb = total_elements * 4 / 1024**3  # FP32
    print(f"\n  Total: {total_transposed} tensors transposed, "
          f"{total_elements:,} elements ({total_gb:.1f} GB FP32)")

    print(f"\nSwift modification needed:")
    print(f"  1. Point weight paths to the -mlx directory")
    print(f"  2. Remove transposition code in WanI2VPipeline (lines ~97-107)")
    print(f"  3. Remove remapVAEKeys() call (line ~116)")
    print(f"  4. Remove remapTransformerKeys() call in WanTransformerWeightLoader")
    print(f"  5. Remove remapGateKeys() call in WanUMT5Encoder")
    print(f"  Or: add a flag like 'preConverted: true' to skip all remapping/transposition")


if __name__ == "__main__":
    main()
