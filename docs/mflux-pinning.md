# mflux Commit Pinning

ZImage's image pipeline is built on [mflux](https://github.com/filipstrand/mflux) -- a Swift/MLX implementation of Flux image generation. This document tracks the mflux commits we depend on for reproducibility and debugging.

## Current Pins

| Component | Repository | Commit | Date | Notes |
|-----------|-----------|--------|------|-------|
| mflux (upstream) | filipstrand/mflux | `156c398` | 2026-04-30 | fix: warm worker done response includes peak_memory_gb and duration_ms |
| mflux (fork) | twalderman/mflux | `156c398` | 2026-04-30 | Synced with upstream |

## How to Update

1. Pull latest upstream:
   ```bash
   cd ~/Projects/mflux
   git fetch origin
   git log --oneline origin/main -5
   ```

2. Test against ZImage:
   ```bash
   cd ~/Projects/zimage.swift
   swift build -c release
   ZImageCLI -p "test prompt" -m ~/.cache/huggingface/hub/z-image-turbo-q8 --steps 2
   ```

3. Update this table with the new commit hash.

## Why Pin?

mflux is a fast-moving project. Pinning commits ensures:
- Reproducible builds across machines
- Known-good baselines for regression testing
- Clear audit trail when debugging generation quality changes
