# DRAFT — upstream issue for ml-explore/mlx (needs Todd's OK before posting)

Title: [Metal] convGeneral silently corrupts boundary frames of large-spatial
5D convolutions (int32 offset overflow class)

## Summary
On Apple Silicon (M3 Max, 128GB), `conv_general` over 5D video tensors
produces silently wrong results in the leading and trailing frame regions when
the spatial extent is large, with no error raised. Interior frames are exact.

## Reproduction (measured, mlx-swift 0.29.1 / mlx core 0.29.1; also present
with the core bundled by mlx-swift 0.31.6 per changelog analysis)
- Conv: k=(3,3,3), stride 1, pad (0,1,1), C_in=128, C_out=128, bf16 weights.
- Input A: (1, 128, 27, 320, 192) — output frames 0-2 and 20-24 WRONG
  (rel err 0.7-1.0 vs ground truth), frames 5-14 EXACT (rel err 0.0).
- Input B: identical conv, spatial crop (1, 128, 27, 128, 128): ALL frames exact.
- Ground truth built from overlapping spatial quadrants of the same conv
  (each quadrant call small; halo-trimmed reassembly) — quadrants agree with
  each other and with small-tensor calls.
- The corruption pattern moves with dtype (bf16: tail-weighted; f32: different
  frame subset), consistent with kernel-selection thresholds crossing an
  int32 offset boundary (same family as #3836/#3609/#3524).
- Empirical onset: between "rows x im2col-stride" ~1.7e9 (clean) and ~3.2e9
  (corrupt) for this shape family; 2^31 = 2.147e9 sits in that bracket.

## Impact
Video VAE decoders (LTX-2 class: 128-ch latents, 3D convs at up to
320x192 x dozens of frames) silently produce garbage frames; the only safe
workaround is chunking every conv call to keep offsets under 2^31, which we
implemented (streaming decode with per-conv temporal caches).

## Ask
Confirm whether the #3836 strided-copy fix (post-v0.32.0) covers the
conv_general path, or whether this is a separate kernel; happy to run a
minimal python repro against main if useful.
