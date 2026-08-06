# codex-director-spec-review-2026-08-04

Distilled from the raw codex transcript (raw discarded to keep the repo light).

   - keyframe points as closed `0...seconds`;
   - whether non-latent-aligned prompt boundaries are rejected or rounded;
   - the exact seconds-to-frame tolerance;
   - whether segment allocation follows absolute boundaries or the reference’s largest-remainder conversion ([ltx_director.py:755](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director.py:755)).
8. **HIGH — `sizeToSpeech` cannot be implemented deterministically from the proposed schema.**
   The reference counts only words inside straight or smart quotation marks ([speech_length_calculator.py:29](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/speech_length_calculator.py:29)). The example’s raw `voice: "come look at this"` would therefore count zero words if passed directly. It returns three alternatives—100, 130, and 160 WPM—and accepts `additional_time` padding ([speech_length_calculator.py:39](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/speech_length_calculator.py:39), [speech_length_calculator.py:46](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/speech_length_calculator.py:46)).
   Pin one deterministic rule. Recommended v1:
   - `voice` is already the spoken payload, so count all whitespace-delimited words in it.
   - Default to 130 WPM.
   - Add `speechWpm` or a `speechRate` enum and `paddingSeconds`.
   - Compute `frames = ceil(((words / wpm) × 60 + padding) × fps)`.
   - Make `end` absent when `sizeToSpeech=true`, or require it to exactly equal the derived end. The current example supplies both without defining precedence.
9. **HIGH — “Typed + total” needs substantially more validation rules.**
   At minimum, rev 2 should require:
   - `timeline` and legacy `prompt` to be mutually exclusive, with explicit conflict rules for request-level width, height, fps, seed, preset, negative prompt, audio, and init image.
   - `fps` in the engine-supported range, dimensions positive and divisible by 32, and a valid single-chunk frame count/window.
   - Strictly ordered prompt coverage after time-to-frame quantization, not merely in floating-point seconds.
   - Nonempty prompt text, `strength ∈ [0,1]`, no duplicate keyframes after latent-frame conversion, and each prompt segment receiving at least one latent frame.
   - Exactly one of `sound` or `voice`, a defined audio-overlap rule, and no unresolved `sizeToSpeech` overflow.
   - Combined post-tokenization length within 128 tokens, including global text, separators, and special tokens; reject instead of allowing tokenizer truncation.
   - Stable asset references or allowlisted path resolution for keyframe files—reject traversal, unreadable files, and unsupported formats.
   - Unknown-field/version policy and bounded document/segment/text sizes for the public API.
10. **HIGH — Phase ordering will otherwise load the wrong model and build masks for the wrong grid.**
    Timeline validation and compilation must occur before warm-model admission, because the compiled plan determines whether the audio branch is required and whether the render is T2V, I2V, or multi-keyframe. The present generator chooses `load(... audio:)` immediately after request validation ([LTX2VideoGenerator.swift:629](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2VideoGenerator.swift:629)).
    Masks also cannot be compiled once as a base-grid tensor. Two-stage refine changes the number of video query tokens, while audio query geometry differs again. The reference solves this with a shape-aware mask closure and cache ([prompt_relay.py:49](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:49), [prompt_relay.py:95](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:95)). The Swift equivalent should be a compiled temporal-mask plan that materializes video-base, video-refine, and audio matrices on demand.
11. **MEDIUM — `generateMultiKeyframe` is not automatically parity with the reference keyframer/sequencer recipes.**
    The reference has two materially different mechanisms:
    - `LTXKeyframer` overwrites dense latent slices and sets their noise mask to `1−strength`, without changing text conditioning ([ltx_keyframer.py:111](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_keyframer.py:111)).
    - `LTXSequencer` calls `append_keyframe`, modifying positive conditioning, negative conditioning, latent state, and noise mask together ([ltx_sequencer.py:116](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_sequencer.py:116), [ltx_sequencer.py:122](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_sequencer.py:122)). The Director guide follows this appended-keyframe route ([ltx_director_guide.py:523](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director_guide.py:523), [ltx_director_guide.py:536](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director_guide.py:536)).
    The Swift method performs dense latent-grid conditioning. It may be the right v1 choice, but “reference 2/3-stage workflows become recipes” needs an explicit semantic mapping and acceptance run, not an assumption of equivalence.
12. **MEDIUM — The proposed visual oracle is too late and too weak for deterministic mask logic.**
    Before same-seed judge comparisons, export golden fixtures from the reference for:
    - full token IDs and local ranges;
    - latent segment-length allocation;
    - video mask rows around every boundary;
    - audio scaled-mask rows;
    - conditional versus unconditional behavior;
    - base versus refine query shapes;
    - speech-sizing cases.
    These are deterministic and should be exact or tolerance-tested in P0. Visual A/B remains useful for final behavioral acceptance, but it cannot diagnose a one-column token offset or a wrong boundary row.
## Concrete answers to the three open questions
1. **Mask granularity:** Video masking is materialized per flattened query token, but its temporal value is per latent frame: every spatial token in a latent frame receives the same row. Audio uses individual audio query-token positions scaled continuously onto the latent-frame axis ([prompt_relay.py:23](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:23)). The reference does not implement a hard cut; it uses the Gaussian boundary behavior above.
2. **NAG/CFG:** Keep the v1 CFG/CFG++ negative prompt global and unmasked. NAG remains part of the masked conditional forward, not the unconditional pass. Per-segment negative prompts should be out of scope. Audio+NAG needs either implementation or an explicit validation rejection.
3. **Segment ceiling:** Do not cap at six because of encoder cost; the reference performs one concatenated encoder call, not one per segment. Enforce the real limits: combined token count ≤128, every segment maps to at least one latent frame, and `segmentCount ≤ latentFrameCount`. A small authoring default of 2–4 beats is sensible, but it should not become a renderer cap.
## Build-readiness verdict
**Not build-ready.** P0 should not freeze the schema until rev 2 defines the conditioning layout, global/negative fields, exact mask and frame semantics, speech sizing, audio modality masks, and API conflict rules. After that, the first engine work should be a unified timeline compiler plus reference mask fixtures, followed by a renderer entry point that genuinely composes keyframes, AV, guidance, and two-stage refinement.
tokens used
185,654
Rev 1 is not build-ready. The reference contradicts several core assumptions in the renderer plan, and the shipped masking, multi-keyframe, guidance, and A/V paths do not yet compose behind one entry point.
## Findings
1. **BLOCKER — The text-encoding architecture is described incorrectly.**
   The reference does not run one text-encoder pass per segment. It concatenates the global prompt and every local prompt, computes token ranges incrementally, and invokes scheduled encoding once ([prompt_relay.py:186](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:186), [ltx_director.py:836](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director.py:836), [ltx_director.py:842](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director.py:842)). Separate per-segment encodes would change contextual embeddings, duplicate special tokens, increase context length, and invalidate same-seed parity.
   This needs an explicit conditioning layout in the spec: one concatenated encode for parity, final-context token ranges, and a rule for global/special/padding columns. The Swift tokenizer left-pads and silently truncates at 128 tokens ([LTX2GemmaTokenizer.swift:127](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/TextEncoder/LTX2GemmaTokenizer.swift:127), [LTX2GemmaTokenizer.swift:161](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/TextEncoder/LTX2GemmaTokenizer.swift:161)), so raw unpadded ranges copied from the Python code would mask the wrong columns unless shifted onto the final padded sequence.
2. **BLOCKER — “Hard cuts” do not match the reference masking math.**
   The reference constructs a soft Gaussian additive penalty:
   `M[q,k] = -strength × relu(|frame(q)-midpoint|−window)² / (2σ²)`
   with `σ = 1/ln(1/epsilon)`, `midpoint = (2×cursor+L)//2`, and `window = max(L//2−2, 0)` ([prompt_relay.py:11](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:11), [prompt_relay.py:17](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:17), [prompt_relay.py:123](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:123), [prompt_relay.py:153](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:153)).
   With the Director default `epsilon=0.001`, a one-frame excursion outside the zero-cost window is about `−23.9`; two frames is about `−95.4`. For adjacent even-length segments, the first latent frame at the new cursor gives both local contexts approximately `−95.4`, while global context remains unpenalized. That is a narrow global-dominated transition, not a hard ownership switch.
   Rev 2 must either:
   - pin this exact Gaussian behavior, including `epsilon=0.001`; or
   - declare hard half-open masks as an intentional non-parity mode and stop using the reference Director as a same-seed oracle.
3. **BLOCKER — The proposed independent audio track cannot use the current shared-mask AV contract.**
   The reference applies the same local-prompt token layout to both video `attn2` and audio `audio_attn2` ([patches.py:192](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/patches.py:192), [patches.py:200](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/patches.py:200)). It does not have an independent generated-audio-text timeline: the Director’s audio segments are uploaded waveform/video media that are trimmed, mixed, encoded, and optionally inpainted ([ltx_director.py:585](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director.py:585), [ltx_director.py:727](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director.py:727), [ltx_director.py:1249](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director.py:1249)).
   The shipped `callAV` accepts separate video/audio contexts but only one mask and forwards it to both streams ([LTX2TransformerAV.swift:157](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/Transformer/LTX2TransformerAV.swift:157), [LTX2TransformerAV.swift:207](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/Transformer/LTX2TransformerAV.swift:207)). Independent prompt and audio tracks require at least `videoContextMask` and `audioContextMask`, with defined token layouts for each.
   Real-seconds cross-modal RoPE does not itself enforce segment scope. The reference masks only direct text cross-attention; temporal self-attention and A2V/V2A attention remain able to propagate information outside the segment.
4. **BLOCKER — Multi-keyframe, audio, masks, and guidance are not currently one composable renderer path.**
   `generateMultiKeyframe` explicitly encodes without audio embeddings ([LTX2Pipeline.swift:820](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:820)), creates no AV state, exposes no context mask, and returns no audio latents. It also prepares NAG embeddings but omits them from its denoising call ([LTX2Pipeline.swift:834](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:834), [LTX2Pipeline.swift:906](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:906)). The generator already rejects audio when its multi-keyframe re-anchoring path is selected ([LTX2VideoGenerator.swift:285](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2VideoGenerator.swift:285)).
   P1 therefore needs a new unified timeline generation path carrying:
   - positive video/audio contexts and separate masks;
   - global CFG/NAG conditioning;
   - arbitrary keyframe conditions;
   - AV state through base and refine;
   - mask regeneration for the refine grid.
   Calling existing methods sequentially will not produce those semantics.
5. **HIGH — Global positive and negative conditioning are absent from the document contract.**
   The reference’s global prompt is deliberately persistent: local token ranges start after it, and the temporal matrix writes penalties only into local-token columns, leaving global columns unpenalized ([prompt_relay.py:219](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:219), [prompt_relay.py:228](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:228), [prompt_relay.py:18](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:18)). The Director exposes this explicitly as `global_prompt` ([ltx_director.py:888](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director.py:888)).
   The schema cannot express persistent character/scene context independently of a beat, nor the global negative/NAG prompt. Add an explicit conditioning section, for example `global`, `negative`, and optionally `nagNegative`, rather than overloading `directions.camera`.
6. **HIGH — Guidance behavior needs to be normative, not an open implementation detail.**
   The reference skips relay masking on a purely unconditional CFG pass ([prompt_relay.py:65](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:65)). Its LTX wrapper is designed to wrap a pre-existing NAG forward and supply the relay mask to that conditional attention operation ([patches.py:94](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/patches.py:94), [patches.py:111](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/patches.py:111)).
   The v1 contract should state:
   - CFG/CFG++ negative conditioning is global and unmasked.
   - Positive and STG conditional passes receive temporal masks.
   - NAG remains inside the masked conditional pass; it is not the CFG unconditional pass.
   - No per-segment negative prompts in v1.
   - Joint AV currently has no NAG support, so either reject `timeline + audio + NAG` or implement it before claiming full composition.
7. **HIGH — Frame-count, duration, boundary, and keyframe-grid semantics are underspecified.**
   The clean canonical interpretation for the example is:
   `timelineSpanFrames = seconds × fps = 192`  
   `renderedFrameCount = timelineSpanFrames + 1 = 193`
   That makes the final keyframe at 8.0 seconds frame index 192. The reference similarly snaps a 120-frame timeline span to a 121-frame LTX render ([ltx_director.py:1178](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director.py:1178), [ltx_director.py:1187](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director.py:1187)).
   “Keyframes off the 1+8k frame grid” is inaccurate. `1+8k` constrains total frame count. If the validator wants non-aliased keyframes, their zero-based indices must be `0 mod 8`, including the final index. The reference itself accepts arbitrary pixel-frame indices and floors them to latent indices ([ltx_keyframer.py:94](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_keyframer.py:94), [ltx_keyframer.py:103](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_keyframer.py:103)).
   Also define:
   - prompt/audio ranges as half-open `[start,end)`;
   - keyframe points as closed `0...seconds`;
   - whether non-latent-aligned prompt boundaries are rejected or rounded;
   - the exact seconds-to-frame tolerance;
   - whether segment allocation follows absolute boundaries or the reference’s largest-remainder conversion ([ltx_director.py:755](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director.py:755)).
8. **HIGH — `sizeToSpeech` cannot be implemented deterministically from the proposed schema.**
   The reference counts only words inside straight or smart quotation marks ([speech_length_calculator.py:29](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/speech_length_calculator.py:29)). The example’s raw `voice: "come look at this"` would therefore count zero words if passed directly. It returns three alternatives—100, 130, and 160 WPM—and accepts `additional_time` padding ([speech_length_calculator.py:39](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/speech_length_calculator.py:39), [speech_length_calculator.py:46](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/speech_length_calculator.py:46)).
   Pin one deterministic rule. Recommended v1:
   - `voice` is already the spoken payload, so count all whitespace-delimited words in it.
   - Default to 130 WPM.
   - Add `speechWpm` or a `speechRate` enum and `paddingSeconds`.
   - Compute `frames = ceil(((words / wpm) × 60 + padding) × fps)`.
   - Make `end` absent when `sizeToSpeech=true`, or require it to exactly equal the derived end. The current example supplies both without defining precedence.
9. **HIGH — “Typed + total” needs substantially more validation rules.**
   At minimum, rev 2 should require:
   - `timeline` and legacy `prompt` to be mutually exclusive, with explicit conflict rules for request-level width, height, fps, seed, preset, negative prompt, audio, and init image.
   - `fps` in the engine-supported range, dimensions positive and divisible by 32, and a valid single-chunk frame count/window.
   - Strictly ordered prompt coverage after time-to-frame quantization, not merely in floating-point seconds.
   - Nonempty prompt text, `strength ∈ [0,1]`, no duplicate keyframes after latent-frame conversion, and each prompt segment receiving at least one latent frame.
   - Exactly one of `sound` or `voice`, a defined audio-overlap rule, and no unresolved `sizeToSpeech` overflow.
   - Combined post-tokenization length within 128 tokens, including global text, separators, and special tokens; reject instead of allowing tokenizer truncation.
   - Stable asset references or allowlisted path resolution for keyframe files—reject traversal, unreadable files, and unsupported formats.
   - Unknown-field/version policy and bounded document/segment/text sizes for the public API.
10. **HIGH — Phase ordering will otherwise load the wrong model and build masks for the wrong grid.**
    Timeline validation and compilation must occur before warm-model admission, because the compiled plan determines whether the audio branch is required and whether the render is T2V, I2V, or multi-keyframe. The present generator chooses `load(... audio:)` immediately after request validation ([LTX2VideoGenerator.swift:629](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2VideoGenerator.swift:629)).
    Masks also cannot be compiled once as a base-grid tensor. Two-stage refine changes the number of video query tokens, while audio query geometry differs again. The reference solves this with a shape-aware mask closure and cache ([prompt_relay.py:49](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:49), [prompt_relay.py:95](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:95)). The Swift equivalent should be a compiled temporal-mask plan that materializes video-base, video-refine, and audio matrices on demand.
11. **MEDIUM — `generateMultiKeyframe` is not automatically parity with the reference keyframer/sequencer recipes.**
    The reference has two materially different mechanisms:
    - `LTXKeyframer` overwrites dense latent slices and sets their noise mask to `1−strength`, without changing text conditioning ([ltx_keyframer.py:111](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_keyframer.py:111)).
    - `LTXSequencer` calls `append_keyframe`, modifying positive conditioning, negative conditioning, latent state, and noise mask together ([ltx_sequencer.py:116](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_sequencer.py:116), [ltx_sequencer.py:122](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_sequencer.py:122)). The Director guide follows this appended-keyframe route ([ltx_director_guide.py:523](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director_guide.py:523), [ltx_director_guide.py:536](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/ltx_director_guide.py:536)).
    The Swift method performs dense latent-grid conditioning. It may be the right v1 choice, but “reference 2/3-stage workflows become recipes” needs an explicit semantic mapping and acceptance run, not an assumption of equivalence.
12. **MEDIUM — The proposed visual oracle is too late and too weak for deterministic mask logic.**
    Before same-seed judge comparisons, export golden fixtures from the reference for:
    - full token IDs and local ranges;
    - latent segment-length allocation;
    - video mask rows around every boundary;
    - audio scaled-mask rows;
    - conditional versus unconditional behavior;
    - base versus refine query shapes;
    - speech-sizing cases.
    These are deterministic and should be exact or tolerance-tested in P0. Visual A/B remains useful for final behavioral acceptance, but it cannot diagnose a one-column token offset or a wrong boundary row.
## Concrete answers to the three open questions
1. **Mask granularity:** Video masking is materialized per flattened query token, but its temporal value is per latent frame: every spatial token in a latent frame receives the same row. Audio uses individual audio query-token positions scaled continuously onto the latent-frame axis ([prompt_relay.py:23](/Users/toddwalderman/Projects/WhatDreamsCost-ComfyUI/prompt_relay.py:23)). The reference does not implement a hard cut; it uses the Gaussian boundary behavior above.
2. **NAG/CFG:** Keep the v1 CFG/CFG++ negative prompt global and unmasked. NAG remains part of the masked conditional forward, not the unconditional pass. Per-segment negative prompts should be out of scope. Audio+NAG needs either implementation or an explicit validation rejection.
3. **Segment ceiling:** Do not cap at six because of encoder cost; the reference performs one concatenated encoder call, not one per segment. Enforce the real limits: combined token count ≤128, every segment maps to at least one latent frame, and `segmentCount ≤ latentFrameCount`. A small authoring default of 2–4 beats is sensible, but it should not become a renderer cap.
## Build-readiness verdict
**Not build-ready.** P0 should not freeze the schema until rev 2 defines the conditioning layout, global/negative fields, exact mask and frame semantics, speech sizing, audio modality masks, and API conflict rules. After that, the first engine work should be a unified timeline compiler plus reference mask fixtures, followed by a renderer entry point that genuinely composes keyframes, AV, guidance, and two-stage refinement.
CODEX-DIRECTOR-EXIT:0
