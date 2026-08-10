#!/usr/bin/env python3
"""Bake LoRA(s) into an LTX-2 Echo monolith safetensors, low-memory.

Replicates ComfyBox's runtime merge (LTX2VideoGenerator.swift ~line 322):
for every `diffusion_model.X.lora_A/B.weight` pair, skipping audio branches
(`audio_`, `av_ca_`, `video_to_audio_attn`, `audio_to_video_attn`),
W' = W + scale * (B @ A) onto `model.diffusion_model.X.weight`.

Instead of rewriting the 46GB file through safetensors (needs all tensors in
RAM), this copies the monolith once, then patches ONLY the merged tensors'
bytes in place at their header offsets — bf16 in, bf16 out, same byte size.
Peak memory: one layer at a time.

Usage:
  python bake_ltx2_lora.py --base <monolith.safetensors> \
      --lora <lora.safetensors> --scale 0.6 --out <out.safetensors>

History: same procedure as the 2026-07-23 sexgod distil bake (that script
lived in /tmp and died in a reboot — this one is checked in).
"""

import argparse
import json
import os
import shutil
import struct
import sys

import torch
from safetensors import safe_open

SKIP_SUBSTRINGS = ("audio_", "av_ca_", "video_to_audio_attn", "audio_to_video_attn")
MONO_PREFIX = "model.diffusion_model."
LORA_PREFIX = "diffusion_model."
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    header, data_start = read_header(args.base)

    print(f"[bake] copying {args.base} -> {args.out}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    shutil.copyfile(args.base, args.out)

    merged, skipped_audio, unmatched = [], [], []
    with safe_open(args.lora, framework="pt") as lf, \
         safe_open(args.base, framework="pt") as bf, \
         open(args.out, "r+b") as out:
        keys = lf.keys()
        for key in sorted(keys):
            if not key.endswith(".lora_A.weight"):
                continue
            base_key = key[: -len(".lora_A.weight")]
            if base_key.startswith(LORA_PREFIX):
                base_key = base_key[len(LORA_PREFIX):]
            if any(s in base_key for s in SKIP_SUBSTRINGS):
                skipped_audio.append(base_key)
                continue
            b_key = key.replace(".lora_A.weight", ".lora_B.weight")
            if b_key not in keys:
                unmatched.append((base_key, "no lora_B"))
                continue
            target = MONO_PREFIX + base_key + ".weight"
            meta = header.get(target)
            if meta is None:
                unmatched.append((base_key, "no monolith target"))
                continue
            if meta["dtype"] != "BF16":
                unmatched.append((base_key, f"dtype {meta['dtype']}"))
                continue

            lora_a = lf.get_tensor(key).to(torch.float32)
            lora_b = lf.get_tensor(b_key).to(torch.float32)
            base_w = bf.get_tensor(target).to(torch.float32)
            delta = (lora_b @ lora_a) * args.scale
            if delta.shape != base_w.shape:
                unmatched.append((base_key, f"shape {tuple(delta.shape)} vs {tuple(base_w.shape)}"))
                continue
            new_w = (base_w + delta).to(torch.bfloat16).contiguous()

            off_start, off_end = meta["data_offsets"]
            raw = new_w.view(torch.uint16).numpy().tobytes()
            expected = off_end - off_start
            if len(raw) != expected:
                print(f"[bake] FATAL byte-size mismatch on {target}: {len(raw)} vs {expected}")
                sys.exit(1)
            out.seek(data_start + off_start)
            out.write(raw)
            merged.append(base_key)
            if len(merged) % 50 == 0:
                print(f"[bake] merged {len(merged)} layers…", flush=True)

    print(f"[bake] DONE: merged {len(merged)}, audio-skipped {len(skipped_audio)}, unmatched {len(unmatched)}")
    for bk, why in unmatched[:20]:
        print(f"[bake]   unmatched: {bk} ({why})")
    if unmatched:
        print("[bake] WARNING: unmatched pairs above — verify against the runtime merge count before shipping.")
    manifest = {
        "base": args.base, "lora": args.lora, "scale": args.scale,
        "merged_layers": len(merged), "audio_skipped": len(skipped_audio),
        "unmatched": [f"{bk} ({why})" for bk, why in unmatched],
    }
    with open(args.out + ".bake-manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
