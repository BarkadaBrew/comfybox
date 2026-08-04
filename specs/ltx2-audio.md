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

---

## Rev 2 — Codex findings resolution (2026-08-03, 16 findings / 5 blockers in reviews/codex-audio-review-2026-08-03.txt)

Superseding corrections — the rev-1 wires understated the work:

1. **Wire 1 targets the WRONG code (blocker #1/#2).** `LTX2CrossModalAttention`
   is an unused orphan. The operative path is `LTX2TransformerBlock.callDualStream`
   — half-built and structurally wrong vs reference (`av_model.py:307`):
   AdaLN row semantics inverted (rows 0–1 = A2V, 2–3 = V2A, 4 = gate), no
   cross-modal temporal RoPE, and V2A computed from already-updated video
   (direction contamination). Plus the top-level transformer forward is
   video-only: it needs a joint (video, audio) contract, four cross-AdaLN
   parameter generators, audio caption projection/connectors, and the audio
   output head. This is a replacement, not a completion.
2. **Conditioning/timesteps are per-modality (blocker #3, finding #4):**
   shared scalar sigma SCHEDULE, but separate per-token timestep tensors per
   modality, zero-timestep reference-audio tokens, cross-AdaLN conditioned on
   the OTHER modality's max timestep, audio 1-D self-RoPE + real-time cross
   temporal RoPE.
3. **Two-stage is JOINT (blocker #5, inverts rev-1):** stage two re-noises
   stage-one audio at the refine sigma and jointly denoises. Acceptance
   inverted: stage-two audio differs deterministically, never passes through.
4. **Codec contract (blocker #7/#8):** AudioVAE must denormalize latents
   per-channel FIRST, use causal temporal padding (not symmetric), causal
   upsample with leading-sample removal, crop to 4*T−3; output is 2-channel
   64-bin (adapter needed before the vocoder's 128-mel input). Vocoder chain
   is base 16kHz → BWE residual → 3x resample → **48kHz stereo**; Swift
   currently stops at the 16kHz base. Sample rate is metadata to add, not
   config to read.
5. **Guidance mapping** from the Python API, not workflow GUI labels (the
   author's Negative-audio/video nodes are cross-wired): specify audio text
   connectors for positive AND negative, CFG++/STG modality behavior,
   independent scales, isolated RNG so audio=false keeps video determinism.
6. **Chunked/extended renders are a design decision (finding #9):** audio is
   25 tok/s with no integer 1-frame overlap at 24fps. Chosen approach:
   single-pass audio for ≤289f (the folded common case); for chunked
   continuations, carry reference-audio tokens (`ref_audio`, negative
   temporal positions) across chunks; PCM crossfade as fallback. Global
   audio positions, not chunk-local.
7. **Lifecycle is Wire 5 (finding #10):** production constructs the
   transformer WITHOUT hasAudio and sanitizes audio weights away. Needs
   conditional construction keyed into the warm-load identity (audio=true
   after a video-only warm load = rebuild), exact expected-key manifest
   binding, unload policy.
8. **Memory truth (finding #15/#16):** audio branch ≈ **+11.3GiB BF16
   static** (audio DiT 10.92 + codec 0.34; quantizer deliberately excludes
   audio weights). Either extend quantize-ltx2 to audio blocks (validated
   separately) or budget the BF16. Admission threshold re-derived from
   MEASURED peak; the 40GB constant is not evidence.
9. **Oracle before code (findings #12–14):** ComfyUI's public nodes can't
   export stage tensors; ltx-2-mlx's T8 parity harness is the right shape
   but has hardcoded absent paths. First build step = a pinned PyTorch
   exporter producing deterministic fixtures for: text connectors (pos/neg),
   positions (self + cross), per-modality timesteps, one dual block, 1-layer
   E2E transformer, one joint sampler step (fixed RNG), latent denorm +
   causal decode, base vocoder, BWE residual, final 48kHz. Neither Swift nor
   ltx-2-mlx is the oracle.
10. **Mux (finding #11):** AAC config, channel layout, PTS/timebase,
    two-input backpressure, priming, trim policy. "No audio track + identical
    frame hashes" replaces byte-identical mp4.

**Revised estimate: 1.5–2 weeks** (was 4–5 days). Ranked order: architecture
statement → oracle → codec parity → sampler semantics → joint two-stage →
lifecycle/memory → extended renders → mux → API/registry/trace.

---

## Wire 4 integration plan (pinned 2026-08-04, transformer path complete)

Transformer-side DONE at reference parity: callDualStream (block oracle
0.999+), AudioPatchifier, prepareAVConditioning / projectAudioTokens /
processAudioOutput (top-level oracle, crossed gates verified), callAV +
precomputeAVPositionalEmbeddings (shape-tested). Mux deadlock fixed
(demand-driven two-track appends). Commits d7e1261, a05186b, 14e4116,
e4282dd, 191e993.

### Pipeline edit (LTX2Pipeline.swift denoise loop)
- Audio latent rate: sampleRate/(hop*ds) = 16000/640 = 25 latent frames/s.
  Ta = ceil(seconds*25). Init noise [B,8,Ta,16] from the ISOLATED audio RNG
  (spec: separate MLXRandom key so video seeds reproduce with/without audio).
- Thread an optional AV state through the step loop: {audioLatents,
  audioContext (textEncoder.audioEmbeddings), av PEs (precomputed once),
  audioSigma == video sigma schedule}.
- POSITIVE pass -> callAV; audio x0 = ax - sigma*velA; same Euler update
  as video. v1: CFG/STG/NAG passes stay VIDEO-ONLY (callAsFunction) and
  only steer the video x0 — audio rides the conditional prediction.
  OPEN: verify against ComfyUI whether STG/CFG passes in the AV recipe
  run run_ax=false (transformer_options) — believed yes for distilled.
- videoSigmaMax for av_ca = sigma (uniform t2v); i2v per-token av_ca video
  ss deferred (OPEN — reference uses per-token timestep_flat there).
- Two-stage refine: re-noise audio JOINTLY at stage-2 strength (spec rev 2).
- Decode: LTX2AudioVAE.load(~/.comfybox models dir), decodeToWaveform
  (proven chain) -> writeMP4(audio: AudioTrack(samples, 48000)).

### Lifecycle / API
- Warm-key gains hasAudio (+11.3GiB gate); registry flag audio=true;
  RenderTraceStore event field has_audio; /v1/video request param
  `audio: bool` (default false), server passes through.
- Weights: monolith already carries audio branch; loadWeights must use
  sanitizeWeightsWithAudio when hasAudio (exists, branch-tested).

### Acceptance
- t2v 2s clip with audible, prompt-relevant sound; A/V duration agree
  within one frame; video-only path byte-identical when audio=false;
  judge clip vs ComfyUI same-seed render.

### Wire 4 SHIPPED (2026-08-04)

Full path live and listen-validated: `/v1/video/generate {"audio": true}` →
dual-stream load (audio in warm key, admission audio-aware) → joint denoise →
codec decode → AAC mux; MCP `generate_video.audio`; `has_audio` +
`audio_seconds` on traces. First clip: prompt-conditioned ocean audio,
Todd-validated. Codex review same day — 10/10 findings folded
(reviews/codex-audio-wire-review-2026-08-04.md), incl. the cross-RoPE double
fps division (positions grid is already seconds) and audio-aware admission.

**v2 backlog:** i2v audio (per-token av_ca video timesteps), NAG/STG on the
joint path, chunked audio, desktop toggle, ComfyUI same-seed judge run,
exact-manifest weight verification (Codex #7 shipped as a 95% audio-branch
coverage gate; full manifest still open).

### i2v audio SHIPPED (2026-08-04, pulled forward)

Per-token av_ca video conditioning (reference timestep_flat semantics),
joint audio re-noise through applyTwoStageRefine with refine-grid AV PEs,
single-chunk i2v accepted (chunked + multi-keyframe still reject).
E2E-validated: i2v surf clip with generated audio. Speech probe underway
(16 steps, quoted dialogue) — quality tuning tracked separately from
architecture.
