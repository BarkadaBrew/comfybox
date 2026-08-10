# LTX Director: Timeline-Directed Video (task #24)

**Status:** DRAFT rev 1 — 2026-08-04. For Codex review.
**Reference:** ~/Projects/WhatDreamsCost-ComfyUI (ltx_director.py PromptRelay
attention masking; ltx_director.js — an 11.7k-line NLE timeline editor;
ltx_keyframer/sequencer; speech_length_calculator; FFLF example workflows).

## Core principle (Todd 2026-08-04)

**The timeline DOCUMENT is the product; UIs and agents are just authors.**
One schema consumed by the renderer, produced by: the desktop segment
editor (v1), Krita's Storyboard Docker (#237/#247), or Kira's daemon
(agent-authored direction — first-class, not an afterthought).

## Timeline document schema (the contract)

```json
{
  "version": 1,
  "fps": 24, "width": 480, "height": 832, "seconds": 8,
  "tracks": {
    "prompt": [
      {"start": 0.0, "end": 3.0, "text": "she walks to the window, morning light"},
      {"start": 3.0, "end": 8.0, "text": "she turns and laughs, hair catching the sun"}
    ],
    "keyframes": [
      {"at": 0.0, "image": "<path>", "strength": 1.0},
      {"at": 8.0, "image": "<path>", "strength": 0.6}
    ],
    "audio": [
      {"start": 1.0, "end": 3.0, "sound": "street noise through the window"},
      {"start": 4.0, "end": 6.5, "voice": "come look at this", "sizeToSpeech": true}
    ]
  },
  "directions": {"camera": "slow push in", "seed": 4242, "preset": "krea-kira"}
}
```

- Segments may overlap only within a track's defined blend rule (v1: prompt
  segments non-overlapping, hard cuts; crossfade masks later).
- `sizeToSpeech`: segment end derived from the speech-length calculation
  (port of speech_length_calculator: text → seconds at fps + padding).
- Validation is typed + total (reject, never guess): coverage gaps in the
  prompt track, out-of-range times, keyframes off the 1+8k frame grid.

## Renderer (engine, ~1 week)

1. **Timeline prompts** = per-segment cross-attention masking: encode each
   segment text; build additive attention masks scoping each context block
   to its token time range (reference PromptRelay/_encode_relay). Our
   seams already exist: contextMask plumbing, per-token conditioning
   (#21), real-seconds time axis.
2. **Keyframes** = existing generateMultiKeyframe (FFLF is a 2-keyframe
   case; reference 2/3-stage workflows become recipes).
3. **Audio track** = segment-scoped audio text joins the audio context with
   the same time masking; voice lines ride the joint A/V path (#21). The
   real-seconds cross-modal RoPE means masked segments naturally scope
   sound to picture time.
4. API: POST /v1/video/generate accepts `timeline` (full document) as an
   alternative to `prompt`; validation errors are typed.

## Authors

- **Kira (daemon):** timeline composer module — extends the shipped
  vision-enrichment: VLM reads the seed, the composer emits a 2-4 beat
  timeline (establish/action/reaction) with sized voice lines instead of
  a single prompt. Config-gated like i2vVoiceEnrichment; objectives
  ledger later decides WHAT to direct.
- **Desktop v1:** segment-list editor in Motion (rows: range/text/voice) —
  NOT the NLE. Full drag-drop timeline is a separate roadmap item after
  the document has mileage; renderer doesn't care who authored.
- **Krita:** storyboard docker emits the same document (#237 alignment).

## Phases

- P0: schema + validator + speech-length port (pure, TDD) — engine-side.
- P1: renderer (masked segments t2v/i2v single-chunk; keyframe track;
  audio track scoping). Oracle: reference ComfyUI Director workflow,
  same-seed judge comparison (validation-only doctrine).
- P2: Kira timeline composer (daemon) + Motion segment editor.
- Non-goals v1: overlap blends, multi-chunk timelines, camera-control
  maps (directive text only), the desktop NLE.

## Open questions for review

1. Mask granularity: per-latent-frame (8-frame quanta) vs per-token —
   reference masks at token level; confirm boundary handling at segment
   edges (hard cut on a latent frame boundary).
2. NAG/CFG interaction with per-segment contexts (negative prompt stays
   global v1?).
3. Segment count ceiling vs encode cost (each segment = one text-encoder
   pass; Gemma is ~seconds each — cap at 6 segments v1?).
