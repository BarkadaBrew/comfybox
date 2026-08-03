# SPEC: LTX-2 audio — joint audio-video generation

**Task:** #21 · **Status:** draft for Codex review, 2026-08-03
**Goal:** ComfyBox LTX renders produce synchronized audio in the mp4, matching
the ComfyUI reference behavior (the last major LTX parity gap after today's
#9/#19 work).

## 1. Verified inventory (2026-08-03, against the shipping code + prod weights)

**Weights: fully present.** The production int8 monolith
(`pinkcherry-v18-distill06-int8`) retains 2,896 `audio*` keys + 1,251
`av_ca*/vocoder*` keys — the distill bake and quantize both preserved the
audio branch. No re-bake, no second checkpoint, no new disk/memory story.

**Ported and believed-complete:**
- `LTX2Vocoder` (BigVGAN-style: SnakeBeta, AMP blocks, anti-aliased
  resampling; `synthesize(mel [B,128,T]) -> stereo waveform`)
- `LTX2AudioVAE` (2D mel VAE, 399 lines, decoder path) — **caveat: never
  validated against golden tensors** (noted Phase-4 item in the file header)
- Transformer top-level audio branch (patchify/proj/AdaLN/scale-shift tables),
  built when the checkpoint `hasAudio`
- Audio embeddings connector in the text encoder (loads at every prod boot:
  "audio_embeddings_connector: 128 weights")
- Checkpoint loader keeps audio tensors lazy; LoRA merge skips audio branches

**NOT implemented (the honest correction to earlier estimates):**
- `LTX2CrossModalAttention` is a STUB — scaffolding + weight slots exist, but
  `callAsFunction` returns inputs unchanged. The A2V/V2A attention math
  (Q-video/KV-audio and inverse, each with 5-param AdaLN gating) must be
  written. This was "deferred to Phase 5"; this spec IS Phase 5.
- `LTX2Pipeline` has zero audio wiring: no audio latents, no audio sigma
  schedule, no audio decode call, no mux.

## 2. The four wires

### Wire 1 — audio latents through the denoising loop (the project)
- Allocate audio latents alongside video latents (shapes/token layout per
  the reference `transformer.py`; audio token count derives from clip
  duration at the audio latent rate).
- Implement the cross-modal attention math in `LTX2CrossModalAttention`
  (replace the stub; weights already load into its slots).
- Audio AdaLN/timestep conditioning via the existing top-level audio branch.
- NAG: the reference author feeds dedicated NEGATIVE audio conditioning
  (08-02 handoff note) — audio joins the NAG positive-pass extrapolation the
  same way video context does.
- Two-stage: audio rides the BASE pass only; the refine pass is spatial
  upsampling and must not touch audio latents. (Verify against the reference
  graphs before assuming — acceptance test 5.3.)
- Scheduler: same sigma schedule as video per reference (verify; if audio
  uses its own schedule, it becomes a registry param via #9's machinery).

### Wire 2 — decode: latents → mel → waveform
`audio latents → LTX2AudioVAE.decode → mel [B,128,T] → LTX2Vocoder.synthesize
→ stereo PCM`. The AudioVAE golden-tensor validation happens HERE, first:
decode a reference latent from the ComfyUI implementation and compare mel
outputs before trusting the whole chain (the upsampler taught us what
partially-bound/unvalidated modules do).

### Wire 3 — mux into the mp4
`LTX2PostProcess.writeMP4` gains an optional PCM track (AVFoundation:
`AVAssetWriterInput` audio alongside the existing video input). Sample rate
from the vocoder config. No audio → byte-identical current behavior.

### Wire 4 — API plumbing
- `LTX2VideoRequest.audio: Bool` (default false — silent renders stay the
  default until validated), `LocalVideoRequest.audio`, MCP schema flag.
- `audio` joins the #9 registry (Tier A, boolExactOne env `LTX2_AUDIO`,
  request/preset-overridable) so provenance/readout come free.
- Trace payload records `audio=true` and the audio decode duration.

## 3. Sequencing (validation-first, per the don't-generalize rule)

1. **Wire 2 first, standalone:** golden-tensor the AudioVAE + vocoder against
   reference outputs (export mel/waveform from the ComfyUI graph on Bolt).
   No pipeline work until the decode chain is proven.
2. Wire 3 (mux) with a synthetic tone — testable without the model.
3. Wire 1 (the transformer/denoise work), behind `audio: false` default.
4. Wire 4 last (it's mechanical once 1–3 exist).
5. A/B: same seed with/without audio — VIDEO output must be unchanged when
   audio=false (byte-comparable latents), and visually equivalent when
   audio=true (cross-modal attention changes the video pathway; the
   reference behavior is "audio-aware video", so equivalence is judged, not
   byte-asserted — Todd's eyes).

## 4. Acceptance

1. AudioVAE mel output matches reference golden tensor (tolerance TBD from
   fp16 comparison), vocoder waveform audibly clean on a reference mel.
2. Muxed mp4 plays with audio in QuickTime/Telegram; A/V duration match
   within one frame.
3. audio=false renders are unchanged vs today (the no-regression gate).
4. audio=true at production config (832x448, 241f single-pass, i2v 0.5 +
   NAG): clip sounds plausibly synchronized — speech/foley matches visible
   action. Judged by ear (no automated metric exists here).
5. Two-stage + audio: refine leaves the audio track intact.
6. Memory: peak RSS during an audio render stays within the 40GB admission
   gate (audio branch adds active compute the gate was not sized for —
   measure, then adjust the gate if needed).

## 5. Risks

- **AudioVAE was never validated** — could be subtly wrong (the mesh-artifact
  lesson). Mitigated by golden-tensor-first sequencing.
- Cross-modal attention is new math, not a port-and-call — the largest
  correctness surface. Port line-by-line from reference `transformer.py`
  with shape asserts at each seam.
- int8: audio-branch linears were quantized with everything else, but no
  audio render has ever exercised them. If quality is off, the A/B vs a
  bf16 monolith isolates quantization (plist repoint, no code).
- Distil LoRA was baked with audio branches SKIPPED at merge — matches the
  reference author's LoRA (video-only deltas), but verify his v1.8 audio
  demos used the distilled checkpoint at all.

## 6. Estimate

Wire 2 + 3: ~1 day (mostly golden-tensor harness). Wire 1: 2–3 days (the
cross-modal math + denoise integration). Wire 4: hours. Total 4–5 days —
up from the earlier 3–4 day guess because the cross-modal stub was
discovered to be a stub, not a port.
