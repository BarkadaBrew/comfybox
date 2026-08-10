# Codex adversarial review — audio handoff (2026-08-05)

Confirmed ROOT CAUSE: the 128-token cap deletes the audio clause from every Kira prompt (0 of 16 survive).

  ✓ Trace tokenizer truncation and audio-embedding/AV-conditioning behavior
  ✓ Identify missing hypotheses and rank cheap discriminating tests
  ✓ Deliver the adversarial numbered review with file-and-line evidence
codex
## Verdict

The handoff’s ordering is wrong. Prompt truncation is not merely “plausible”; the current T2V assembly demonstrably places every authored audio clause beyond the 128-token boundary. P0 is not discriminating because it changes modality, audio semantics, prompt length, negative conditioning, duration policy, and likely dimensions at once.

The most likely causal chain is:

`daemon rewrite / scene prefix → character or EXIF prefix → audio clause last → prefix(128) truncation → audio model never sees the requested sound/dialogue`

### Findings

1. **Critical — the current T2V path deletes the audio clause.**

   The generator hardcodes `maxLength: 128`, then encodes both positive and negative prompts at that limit ([LTX2VideoGenerator.swift:552](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2VideoGenerator.swift:552), [LTX2VideoGenerator.swift:663](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2VideoGenerator.swift:663)). Truncation keeps `tokens.prefix(128)`, so it removes the tail, not the head ([LTX2GemmaTokenizer.swift:152](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/TextEncoder/LTX2GemmaTokenizer.swift:152)).

   Kira T2V appends `audio:` last ([content-scheduler.ts:1736](/Users/toddwalderman/Projects/coffeeshop-server/src/kira/content-scheduler.ts:1736), [content-scheduler.ts:1741](/Users/toddwalderman/Projects/coffeeshop-server/src/kira/content-scheduler.ts:1741)). MCP drops the explicit `character`, but WarmServer defaults T2V to Kira and prepends her description ([WarmServer.swift:2108](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/Server/WarmServer.swift:2108), [WarmServer.swift:2149](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/Server/WarmServer.swift:2149)).

   Using the installed Gemma tokenizer and current character configuration, I counted the conservative, no-optimizer case:

   - Kira’s neutral base: 101 tokens.
   - The 16 current `buildT2VScene` prompts after assembly: 168–186 tokens.
   - `audio:` begins around token 155–178.
   - Audio clauses surviving the 128-token cap: **0 of 16**.

   That is already sufficient to explain missing English and uncontrolled sound. The “no quoted line causes vocalization” theory is downstream of a more basic issue: the model often receives no audio instruction at all.

2. **Critical — the daemon optimizer and the 128-token encoder have contradictory contracts.**

   The LTX T2V optimizer mandates at least 120 words and explicitly says denser is better ([engine-templates.ts:331](/Users/toddwalderman/Projects/coffeeshop-server/src/image/engine-templates.ts:331), [engine-templates.ts:361](/Users/toddwalderman/Projects/coffeeshop-server/src/image/engine-templates.ts:361)). Its LLM output replaces the original prompt wholesale ([prompt-optimizer.ts:734](/Users/toddwalderman/Projects/coffeeshop-server/src/image/prompt-optimizer.ts:734)). There is no instruction or postcondition requiring the audio clause—or quoted dialogue—to survive or move near the front.

   I2V has two similarly unsafe branches:

   - If source metadata is readable, the complete, unclamped EXIF scene prompt is prepended, then optimization is skipped ([video-tools.ts:970](/Users/toddwalderman/Projects/coffeeshop-server/src/tools/video-tools.ts:970)). Audio remains last.
   - Otherwise, the LLM rewrites the whole request as “motion-only,” again with no audio-preservation rule ([prompt-optimizer.ts:696](/Users/toddwalderman/Projects/coffeeshop-server/src/image/prompt-optimizer.ts:696)).

   WarmServer’s own enhancer is not the normal Kira culprit because the daemon submits `enhance:false` ([video-tools.ts:1081](/Users/toddwalderman/Projects/coffeeshop-server/src/tools/video-tools.ts:1081), [video-tools.ts:1351](/Users/toddwalderman/Projects/coffeeshop-server/src/tools/video-tools.ts:1351)). But it is another reference-vs-daemon confound: it replaces the prompt completely ([WarmServer.swift:2138](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/Server/WarmServer.swift:2138)), and its user message incorrectly asks for a “Z-Image Turbo” rewrite even when the selected system template is LTX video ([PromptOptimizer.swift:462](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/Telegram/PromptOptimizer.swift:462)). A direct request that omitted `enhance:false` followed a materially different path.

3. **High — long and short prompts have the same audio-embedding shape, but materially different content.**

   There is no audio-specific tokenizer or audio-clause parser. Gemma hidden states feed separate video and audio projections/connectors ([LTX2TextEncoder.swift:98](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/TextEncoder/LTX2TextEncoder.swift:98)). With the generator’s tokenizer, both short and long prompts therefore produce:

   - Video context: `[B, 128, 4096]`
   - Audio context: `[B, 128, 2048]`

   A character prefix does not alter the shape, but it:

   - moves tail tokens beyond the cutoff;
   - changes positions;
   - changes later hidden states because Gemma is causal ([LTX2GemmaModel.swift:339](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/TextEncoder/LTX2GemmaModel.swift:339)).

   Short prompts also get padding replaced by learned register tokens; a full 128-token prompt has no register positions ([LTX2Connector1D.swift:420](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/TextEncoder/LTX2Connector1D.swift:420)). Thus “short reference” and “128-token-saturated Kira prompt” differ even when their visible audio words are nominally similar.

   The connector itself is not inherently restricted to exactly 128 tokens: it tiles 128 registers across any divisible sequence length ([LTX2Connector1D.swift:442](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/TextEncoder/LTX2Connector1D.swift:442)); the [official Lightricks connector](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/embeddings_connector.py) uses the same divisibility rule. The exact 128 cap is a generator policy, though raising it should still be validated against the trained recipe.

4. **High — empty LoRA arrays do not make the presets inert.**

   The active Kira presets have empty LoRA arrays but nonempty negative prompts ([presets.json:185](/Users/toddwalderman/.comfybox/presets.json:185), [presets.json:193](/Users/toddwalderman/.comfybox/presets.json:193)). Token counts are 23 for neutral and 77 for the longer Kira video presets.

   WarmServer applies a preset negative whenever the request does not override it ([WarmServer.swift:2187](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/Server/WarmServer.swift:2187)). The pipeline produces separate negative audio embeddings ([LTX2Pipeline.swift:203](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:203)), runs the dual-stream negative pass ([LTX2Pipeline.swift:1161](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1161)), and CFG++ uses `x0Neg` even at audio CFG 1.0 ([LTX2Guidance.swift:176](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Guidance.swift:176)).

   Therefore the known-good no-preset request and a Kira-preset request do not have equivalent audio conditioning. “Preset ID” is also not one variable for P3; it is a bundle.

5. **High — I2V is not a valid proxy for the known-good T2V path.**

   T2V and I2V create audio text embeddings similarly, but I2V additionally encodes the source image and normally appends IC-control reference frames ([LTX2Pipeline.swift:635](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:635), [LTX2Pipeline.swift:675](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:675)).

   More importantly, every AV transformer block updates audio from video tokens through video-to-audio cross-modal attention ([LTX2TransformerBlock.swift:238](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/Transformer/LTX2TransformerBlock.swift:238), [LTX2TransformerBlock.swift:269](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/Transformer/LTX2TransformerBlock.swift:269)). Source image, IC frames, dimensions, and video-token count can therefore all change audio.

   “Both P0 I2V arms are bad” would not distinguish truncation from I2V coupling. It certainly would not prove a generic request-path bug.

6. **High — the daemon’s logged request shape is not the engine’s request shape.**

   The MCP schema and forwarding layer omit `character`, `content_mode`, `frame_rate`, `mode`, and `model` ([MCPToolRegistry.swift:419](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/MCP/MCPToolRegistry.swift:419), [MCPToolExecutor.swift:512](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/MCP/MCPToolExecutor.swift:512)).

   Consequences:

   - T2V defaults to Kira with **neutral** character content because `content_mode` is lost.
   - I2V gets no WarmServer character injection.
   - The daemon’s `frame_rate` value is ignored.
   - MCP forces T2V to 448×704 or 704×448, which overrides the named resolution ([MCPToolExecutor.swift:529](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/MCP/MCPToolExecutor.swift:529)).
   - I2V dimensions are derived from source aspect and the resolution budget, not necessarily 512×832 ([WarmServer.swift:2041](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/Server/WarmServer.swift:2041)).

   P0’s “exact request shape” and P3’s dimension assumptions are therefore unproven.

7. **High — P2 is not a duration-plumbing bug. It is an intentional policy override.**

   For banana/avocado I2V, `computeI2VMotionRecipe` explicitly floors every ≤12-second request at 241 frames ([video-tools.ts:287](/Users/toddwalderman/Projects/coffeeshop-server/src/tools/video-tools.ts:287)). Submission then sends `frames` and deliberately omits `duration` ([video-tools.ts:1101](/Users/toddwalderman/Projects/coffeeshop-server/src/tools/video-tools.ts:1101)).

   A unit test asserts that 4 seconds must become 241 frames ([video-motion-recipe.test.ts:79](/Users/toddwalderman/Projects/coffeeshop-server/src/tools/video-motion-recipe.test.ts:79)). At 24 fps, that is about 10.04 seconds.

   The long duration may still harm audio, but the handoff has misidentified the mechanism. Achieving 97 frames requires changing the policy and its test, not repairing lost plumbing.

8. **Medium — P1’s proposed log check cannot prove that dialogue reaches the engine.**

   `composeI2VPrompt` puts sounds and voice last ([i2v-voice-enrichment.ts:74](/Users/toddwalderman/Projects/coffeeshop-server/src/kira/vision/i2v-voice-enrichment.ts:74)), while the scheduler logs only the first 160 characters of `finalIntent` ([content-scheduler.ts:1191](/Users/toddwalderman/Projects/coffeeshop-server/src/kira/content-scheduler.ts:1191)). Grepping `enriched intent` can therefore miss the very clause it is supposed to verify.

   Log structured facts instead: pre-truncation token count, audio-marker token index, whether quoted dialogue survived, and the hash of the final effective prompt.

## P0–P4 critique

P0 is not discriminating:

- Both arms may lose their audio clause after token 128.
- It switches from the proven T2V modality to I2V.
- Spoken dialogue versus ambience naturally changes voice/floor and ZCR; those metrics are not comparable semantic controls.
- “Neutral subject” does not specify whether the non-neutral preset, 241-frame floor, and prompt trigger remain active.
- Both-bad does not imply “request-path bug”; A-clean/B-bad does not isolate why the prompt mattered.

The fastest first test costs zero render minutes:

1. Capture the daemon’s final submitted prompt and WarmServer’s final effective prompt.
2. Tokenize without truncating.
3. Report total tokens, `audio:` token index, whether any quoted line survives tokens 0–127, and the decoded neutral first 128 tokens.
4. Add an invariant: when `audio:true`, reject or reorder any request whose audio clause would be truncated.

Then run one cheap 97-frame **T2V** pair, matching the known-good modality:

- Same short neutral request, seed, dimensions, negative, and no preset.
- A: audio clause near the beginning.
- B: the same clause after a neutral character-length prefix.
- `enhance:false`, explicit `frames:97`.

If A is good and B is bad, truncation is isolated. Next run the same short T2V request with only the negative prompt swapped to the Kira preset negative. Test I2V only after those two variables are settled.

P1 should not author more audio text until survival is enforced. P2 should be reframed as a duration-policy decision. P3 needs the exact captured known-good JSON/effective prompt/config—the handoff only preserves a prose reconstruction ([handoff:120](/Users/toddwalderman/Projects/zimage.swift/specs/prepared/ltx-audio-tuning-handoff.md:120))—and preset, modality, and dimensions must be decomposed rather than treated as single attributes. P4’s NAG and sigma work should remain behind prompt-survival and negative-conditioning checks.

No media was opened or evaluated, and no files were changed.
tokens used
423,280
## Verdict

The handoff’s ordering is wrong. Prompt truncation is not merely “plausible”; the current T2V assembly demonstrably places every authored audio clause beyond the 128-token boundary. P0 is not discriminating because it changes modality, audio semantics, prompt length, negative conditioning, duration policy, and likely dimensions at once.

The most likely causal chain is:

`daemon rewrite / scene prefix → character or EXIF prefix → audio clause last → prefix(128) truncation → audio model never sees the requested sound/dialogue`

### Findings

1. **Critical — the current T2V path deletes the audio clause.**

   The generator hardcodes `maxLength: 128`, then encodes both positive and negative prompts at that limit ([LTX2VideoGenerator.swift:552](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2VideoGenerator.swift:552), [LTX2VideoGenerator.swift:663](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2VideoGenerator.swift:663)). Truncation keeps `tokens.prefix(128)`, so it removes the tail, not the head ([LTX2GemmaTokenizer.swift:152](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/TextEncoder/LTX2GemmaTokenizer.swift:152)).

   Kira T2V appends `audio:` last ([content-scheduler.ts:1736](/Users/toddwalderman/Projects/coffeeshop-server/src/kira/content-scheduler.ts:1736), [content-scheduler.ts:1741](/Users/toddwalderman/Projects/coffeeshop-server/src/kira/content-scheduler.ts:1741)). MCP drops the explicit `character`, but WarmServer defaults T2V to Kira and prepends her description ([WarmServer.swift:2108](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/Server/WarmServer.swift:2108), [WarmServer.swift:2149](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/Server/WarmServer.swift:2149)).

   Using the installed Gemma tokenizer and current character configuration, I counted the conservative, no-optimizer case:

   - Kira’s neutral base: 101 tokens.
   - The 16 current `buildT2VScene` prompts after assembly: 168–186 tokens.
   - `audio:` begins around token 155–178.
   - Audio clauses surviving the 128-token cap: **0 of 16**.

   That is already sufficient to explain missing English and uncontrolled sound. The “no quoted line causes vocalization” theory is downstream of a more basic issue: the model often receives no audio instruction at all.

2. **Critical — the daemon optimizer and the 128-token encoder have contradictory contracts.**

   The LTX T2V optimizer mandates at least 120 words and explicitly says denser is better ([engine-templates.ts:331](/Users/toddwalderman/Projects/coffeeshop-server/src/image/engine-templates.ts:331), [engine-templates.ts:361](/Users/toddwalderman/Projects/coffeeshop-server/src/image/engine-templates.ts:361)). Its LLM output replaces the original prompt wholesale ([prompt-optimizer.ts:734](/Users/toddwalderman/Projects/coffeeshop-server/src/image/prompt-optimizer.ts:734)). There is no instruction or postcondition requiring the audio clause—or quoted dialogue—to survive or move near the front.

   I2V has two similarly unsafe branches:

   - If source metadata is readable, the complete, unclamped EXIF scene prompt is prepended, then optimization is skipped ([video-tools.ts:970](/Users/toddwalderman/Projects/coffeeshop-server/src/tools/video-tools.ts:970)). Audio remains last.
   - Otherwise, the LLM rewrites the whole request as “motion-only,” again with no audio-preservation rule ([prompt-optimizer.ts:696](/Users/toddwalderman/Projects/coffeeshop-server/src/image/prompt-optimizer.ts:696)).

   WarmServer’s own enhancer is not the normal Kira culprit because the daemon submits `enhance:false` ([video-tools.ts:1081](/Users/toddwalderman/Projects/coffeeshop-server/src/tools/video-tools.ts:1081), [video-tools.ts:1351](/Users/toddwalderman/Projects/coffeeshop-server/src/tools/video-tools.ts:1351)). But it is another reference-vs-daemon confound: it replaces the prompt completely ([WarmServer.swift:2138](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/Server/WarmServer.swift:2138)), and its user message incorrectly asks for a “Z-Image Turbo” rewrite even when the selected system template is LTX video ([PromptOptimizer.swift:462](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/Telegram/PromptOptimizer.swift:462)). A direct request that omitted `enhance:false` followed a materially different path.

3. **High — long and short prompts have the same audio-embedding shape, but materially different content.**

   There is no audio-specific tokenizer or audio-clause parser. Gemma hidden states feed separate video and audio projections/connectors ([LTX2TextEncoder.swift:98](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/TextEncoder/LTX2TextEncoder.swift:98)). With the generator’s tokenizer, both short and long prompts therefore produce:

   - Video context: `[B, 128, 4096]`
   - Audio context: `[B, 128, 2048]`

   A character prefix does not alter the shape, but it:

   - moves tail tokens beyond the cutoff;
   - changes positions;
   - changes later hidden states because Gemma is causal ([LTX2GemmaModel.swift:339](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/TextEncoder/LTX2GemmaModel.swift:339)).

   Short prompts also get padding replaced by learned register tokens; a full 128-token prompt has no register positions ([LTX2Connector1D.swift:420](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/TextEncoder/LTX2Connector1D.swift:420)). Thus “short reference” and “128-token-saturated Kira prompt” differ even when their visible audio words are nominally similar.

   The connector itself is not inherently restricted to exactly 128 tokens: it tiles 128 registers across any divisible sequence length ([LTX2Connector1D.swift:442](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/TextEncoder/LTX2Connector1D.swift:442)); the [official Lightricks connector](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/embeddings_connector.py) uses the same divisibility rule. The exact 128 cap is a generator policy, though raising it should still be validated against the trained recipe.

4. **High — empty LoRA arrays do not make the presets inert.**

   The active Kira presets have empty LoRA arrays but nonempty negative prompts ([presets.json:185](/Users/toddwalderman/.comfybox/presets.json:185), [presets.json:193](/Users/toddwalderman/.comfybox/presets.json:193)). Token counts are 23 for neutral and 77 for the longer Kira video presets.

   WarmServer applies a preset negative whenever the request does not override it ([WarmServer.swift:2187](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/Server/WarmServer.swift:2187)). The pipeline produces separate negative audio embeddings ([LTX2Pipeline.swift:203](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:203)), runs the dual-stream negative pass ([LTX2Pipeline.swift:1161](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1161)), and CFG++ uses `x0Neg` even at audio CFG 1.0 ([LTX2Guidance.swift:176](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Guidance.swift:176)).

   Therefore the known-good no-preset request and a Kira-preset request do not have equivalent audio conditioning. “Preset ID” is also not one variable for P3; it is a bundle.

5. **High — I2V is not a valid proxy for the known-good T2V path.**

   T2V and I2V create audio text embeddings similarly, but I2V additionally encodes the source image and normally appends IC-control reference frames ([LTX2Pipeline.swift:635](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:635), [LTX2Pipeline.swift:675](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:675)).

   More importantly, every AV transformer block updates audio from video tokens through video-to-audio cross-modal attention ([LTX2TransformerBlock.swift:238](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/Transformer/LTX2TransformerBlock.swift:238), [LTX2TransformerBlock.swift:269](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/Transformer/LTX2TransformerBlock.swift:269)). Source image, IC frames, dimensions, and video-token count can therefore all change audio.

   “Both P0 I2V arms are bad” would not distinguish truncation from I2V coupling. It certainly would not prove a generic request-path bug.

6. **High — the daemon’s logged request shape is not the engine’s request shape.**

   The MCP schema and forwarding layer omit `character`, `content_mode`, `frame_rate`, `mode`, and `model` ([MCPToolRegistry.swift:419](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/MCP/MCPToolRegistry.swift:419), [MCPToolExecutor.swift:512](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/MCP/MCPToolExecutor.swift:512)).

   Consequences:

   - T2V defaults to Kira with **neutral** character content because `content_mode` is lost.
   - I2V gets no WarmServer character injection.
   - The daemon’s `frame_rate` value is ignored.
   - MCP forces T2V to 448×704 or 704×448, which overrides the named resolution ([MCPToolExecutor.swift:529](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/MCP/MCPToolExecutor.swift:529)).
   - I2V dimensions are derived from source aspect and the resolution budget, not necessarily 512×832 ([WarmServer.swift:2041](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/Server/WarmServer.swift:2041)).

   P0’s “exact request shape” and P3’s dimension assumptions are therefore unproven.

7. **High — P2 is not a duration-plumbing bug. It is an intentional policy override.**

   For banana/avocado I2V, `computeI2VMotionRecipe` explicitly floors every ≤12-second request at 241 frames ([video-tools.ts:287](/Users/toddwalderman/Projects/coffeeshop-server/src/tools/video-tools.ts:287)). Submission then sends `frames` and deliberately omits `duration` ([video-tools.ts:1101](/Users/toddwalderman/Projects/coffeeshop-server/src/tools/video-tools.ts:1101)).

   A unit test asserts that 4 seconds must become 241 frames ([video-motion-recipe.test.ts:79](/Users/toddwalderman/Projects/coffeeshop-server/src/tools/video-motion-recipe.test.ts:79)). At 24 fps, that is about 10.04 seconds.

   The long duration may still harm audio, but the handoff has misidentified the mechanism. Achieving 97 frames requires changing the policy and its test, not repairing lost plumbing.

8. **Medium — P1’s proposed log check cannot prove that dialogue reaches the engine.**

   `composeI2VPrompt` puts sounds and voice last ([i2v-voice-enrichment.ts:74](/Users/toddwalderman/Projects/coffeeshop-server/src/kira/vision/i2v-voice-enrichment.ts:74)), while the scheduler logs only the first 160 characters of `finalIntent` ([content-scheduler.ts:1191](/Users/toddwalderman/Projects/coffeeshop-server/src/kira/content-scheduler.ts:1191)). Grepping `enriched intent` can therefore miss the very clause it is supposed to verify.

   Log structured facts instead: pre-truncation token count, audio-marker token index, whether quoted dialogue survived, and the hash of the final effective prompt.

## P0–P4 critique

P0 is not discriminating:

- Both arms may lose their audio clause after token 128.
- It switches from the proven T2V modality to I2V.
- Spoken dialogue versus ambience naturally changes voice/floor and ZCR; those metrics are not comparable semantic controls.
- “Neutral subject” does not specify whether the non-neutral preset, 241-frame floor, and prompt trigger remain active.
- Both-bad does not imply “request-path bug”; A-clean/B-bad does not isolate why the prompt mattered.

The fastest first test costs zero render minutes:

1. Capture the daemon’s final submitted prompt and WarmServer’s final effective prompt.
2. Tokenize without truncating.
3. Report total tokens, `audio:` token index, whether any quoted line survives tokens 0–127, and the decoded neutral first 128 tokens.
4. Add an invariant: when `audio:true`, reject or reorder any request whose audio clause would be truncated.

Then run one cheap 97-frame **T2V** pair, matching the known-good modality:

- Same short neutral request, seed, dimensions, negative, and no preset.
- A: audio clause near the beginning.
- B: the same clause after a neutral character-length prefix.
- `enhance:false`, explicit `frames:97`.

If A is good and B is bad, truncation is isolated. Next run the same short T2V request with only the negative prompt swapped to the Kira preset negative. Test I2V only after those two variables are settled.

P1 should not author more audio text until survival is enforced. P2 should be reframed as a duration-policy decision. P3 needs the exact captured known-good JSON/effective prompt/config—the handoff only preserves a prose reconstruction ([handoff:120](/Users/toddwalderman/Projects/zimage.swift/specs/prepared/ltx-audio-tuning-handoff.md:120))—and preset, modality, and dimensions must be decomposed rather than treated as single attributes. P4’s NAG and sigma work should remain behind prompt-survival and negative-conditioning checks.

No media was opened or evaluated, and no files were changed.
EXIT:0
