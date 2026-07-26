# Checkpoint: video pipeline known-good (2026-07-26 ~04:00)

Locked immediately after the first fully-successful end-to-end Kira video:
12s / 3 chunks / 448x832 / zero crashes / filed 03:59.

- Binary: branch claude/ltx2-haze-optimization @ 68b0ffa (tag: checkpoint-video-20260726)
- Weights: ~/LocalModels/sexgod-distill06-bf16 (workflow-equivalent distill @0.6 bake)
- Config: sibling .plist (env = workflow-faithful: 20-step sigmas, comp 2 = CRF 2,
  flat cfg, TWO_STAGE=1, refine sigmas 0.85/0.725/0.4219, streamed exact decode)
- Daemon: coffeeshop-server main @ 179e6b5 (tool cap 80min, hardened fail-fast,
  deadline 25min+20/chunk)
- Anchor metrics at this checkpoint: i2v sharp 35.3 / flicker 0.239 / identity 0.905
  (GT 58.8 / 0.157); t2v sharp 103.7 / flicker 0.096 / motion 0.78.
- Rollback: this plist + `git checkout checkpoint-video-20260726 && swift build -c release`
  + xattr/codesign + bootout/bootstrap.
