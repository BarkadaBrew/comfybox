# codex-audio-fixes-review-2026-08-05

Distilled from the raw codex transcript (raw discarded to keep the repo light).

Sources/ComfyBox/main.swift:3664:        seed: actualSeed,
Sources/ComfyBox/main.swift:3680:        seed: actualSeed,
Sources/ComfyBox/main.swift:4110:      steps: steps, seed: UInt64(seed),
Sources/ComfyBox/main.swift:4561:        steps: steps, seed: chunkSeed,
Sources/ComfyBox/main.swift:4628:    var seed: UInt64 = 0
Sources/ComfyBox/main.swift:4657:      .init(prompt: prompt, width: width, height: height, steps: steps, seed: seed)
Sources/ZImage/Server/PresetStore.swift:88:  public var seed: Int?
Sources/ZImage/Server/PresetStore.swift:115:    seed: Int? = nil,
Sources/ZImage/Server/PresetStore.swift:244:  public var seed: Int?
Sources/ZImage/Server/WarmServer.swift:1681:    let seed: UInt64?
Sources/ZImage/Server/WarmServer.swift:2195:      seed: req.seed ?? videoPreset?.seed.map(UInt64.init) ?? 42,
Sources/ZImage/Server/WarmServer.swift:2759:        seed: request.seed,
Sources/ZImage/Server/WarmServer.swift:2864:      seed: request.seed,
Sources/ZImage/Server/WarmServer.swift:2908:            seed: baseSeed + UInt64(i),
Sources/ZImage/Server/WarmServer.swift:3155:      seed: payload.seed,
Sources/ZImage/Server/WarmServer.swift:3273:    let seed: UInt64?
Sources/ZImage/Server/WarmServer.swift:3292:      ? WorkflowRunPayload(prompt: nil, negativePrompt: nil, seed: nil, outputPath: nil, timeoutS: nil)
Sources/ZImage/Server/WarmServer.swift:3419:      let seed: UInt64?
Sources/ZImage/Server/WarmServer.swift:3427:      let seed: UInt64?
Sources/ZImage/Server/WarmServer.swift:3470:            seed: ins.seed)
Sources/ZImage/Server/WarmServer.swift:3473:        seed: s.seed)
Sources/ZImage/Server/WarmServer.swift:3529:          seed: insert.seed,
Sources/ZImage/Server/WarmServer.swift:5682:        seed: payload.seed,
Sources/ZImage/Server/WarmServer.swift:5787:                steps: steps, seed: seed, strength: strength, dyPE: krea2DyPE)
Sources/ZImage/Server/WarmServer.swift:5793:          .init(prompt: payload.prompt, width: width, height: height, steps: steps, seed: seed,
Sources/ZImage/Server/WarmServer.swift:5801:        seed: seed,
Sources/ZImage/Server/WarmServer.swift:5863:        seed: payload.seed,
Sources/ZImage/Server/WarmServer.swift:6000:      seed: seed,
Sources/ZImage/Server/WarmServer.swift:6012:        seed: seed, steps: steps, guidance: guidance, width: width, height: height,
Sources/ZImage/Server/WarmServer.swift:6437:  let seed: UInt64?
Sources/ZImage/Server/WarmServer.swift:6503:    guidance: Float? = nil, seed: UInt64? = nil, outputPath: String? = nil,
Sources/ZImage/Server/WarmServer.swift:6655:      seed: seed,
Sources/ZImage/Server/WarmServer.swift:6729:      seed: seed,
Sources/ZImage/Server/WarmServer.swift:6834:  let seed: Int?
Sources/ZImage/Server/ComfyBridge/ComfyBridgeWorkflowParser.swift:23:  var seed: UInt64?
Sources/ZImage/Server/ComfyBridge/ComfyBridgeWorkflowParser.swift:457:      seed: seed,
exec
/bin/zsh -lc "nl -ba .claude/skills/kira-ltx-tuning/SKILL.md | sed -n '20,75p'
nl -ba Sources/ZImage/LTX2/LTX2Pipeline.swift | sed -n '1306,1334p;1438,1450p'
nl -ba Sources/ZImage/LTX2/LTX2Pipeline.swift | sed -n '320,399p;1550,1610p;1652,1678p'
nl -ba Sources/ZImage/LTX2/LTX2Guidance.swift | sed -n '155,210p'" in /Users/toddwalderman/Projects/zimage.swift
 succeeded in 1ms:
   155	public func ltx2AudioStep(
   156	  audio ax: MLXArray,
   157	  velocityPos: MLXArray,
   158	  velocityNeg: MLXArray?,
   159	  sigma: Float,
   160	  sigmaNext: Float,
   161	  cfgScale: Float,
   162	  useCfgPP: Bool,
   163	  useSDE: Bool,
   164	  ancestralNoise: MLXArray?
   165	) -> MLXArray {
   166	  let sig = MLXArray(sigma)
   167	  let x0Cond = ax - sig * velocityPos.asType(.float32)
   168	  guard sigmaNext > 0 else {
   169	    if let vn = velocityNeg, cfgScale > 1.0 {
   170	      let x0Neg = ax - sig * vn.asType(.float32)
   171	      return LTX2Guidance.applyCFG(conditioned: x0Cond, unconditioned: x0Neg, scale: cfgScale)
   172	    }
   173	    return x0Cond
   174	  }
   175	
   176	  guard let vn = velocityNeg else {
   177	    // No negative context: plain Euler on the positive velocity (v1 behavior).
   178	    return ax + MLXArray(sigmaNext - sigma) * velocityPos.asType(.float32)
   179	  }
   180	  let x0Neg = ax - sig * vn.asType(.float32)
   181	  var x0Guided = x0Cond
   182	  if cfgScale > 1.0 {
   183	    x0Guided = LTX2Guidance.applyCFG(conditioned: x0Cond, unconditioned: x0Neg, scale: cfgScale)
   184	  }
   185	
   186	  if useCfgPP {
   187	    let alphaS = max(1.0 - sigma, Float(1e-4))
   188	    let alphaT = 1.0 - sigmaNext
   189	    let d = (ax - MLXArray(alphaS) * x0Neg) / sig
   190	    var sigmaDown = sigmaNext
   191	    var sigmaUp: Float = 0
   192	    if useSDE {
   193	      let sf = sigma / alphaS
   194	      let st = sigmaNext / alphaT
   195	      let inner = st * st * (sf * sf - st * st) / (sf * sf)
   196	      let up = min(st, (inner > 0 ? inner : 0).squareRoot())
   197	      sigmaDown = alphaT * (max(st * st - up * up, 0)).squareRoot()
   198	      sigmaUp = up
   199	    }
   200	    var next = MLXArray(alphaT) * x0Guided + MLXArray(sigmaDown) * d
   201	    if useSDE, sigmaUp > 0, let noise = ancestralNoise {
   202	      next = next + MLXArray(alphaT) * noise * MLXArray(sigmaUp)
   203	    }
   204	    return next
   205	  }
   206	  // Classic CFG without cfg_pp: Euler toward the guided x0.
   207	  return x0Guided + MLXArray(sigmaNext) * (ax - x0Guided) / sig
   208	}
exec
/bin/zsh -lc 'swiftc -frontend -parse Sources/ZImage/LTX2/LTX2Pipeline.swift Sources/ZImage/LTX2/LTX2Guidance.swift Sources/ZImage/LTX2/LTX2AudioEnhance.swift Tests/ZImageTests/LTX2AudioEnhanceTests.swift' in /Users/toddwalderman/Projects/zimage.swift
 succeeded in 4375ms:
swiftc: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
swiftc: error: couldn't create cache file '/tmp/xcrun_db-DINEmHjv' (errno=Operation not permitted)
2026-08-05 15:44:32.346 xcodebuild[30951:4502501]  DVTFilePathFSEvents: Failed to start fs event stream.
2026-08-05 15:44:32.658 xcodebuild[30951:4502500] [MT] DVTDeveloperPaths: Failed to get length of DARWIN_USER_CACHE_DIR from confstr(3), error = Error Domain=NSPOSIXErrorDomain Code=5 "Input/output error". Using NSCachesDirectory instead.
2026-08-05 15:44:33.365 xcodebuild[30954:4502548]  DVTFilePathFSEvents: Failed to start fs event stream.
2026-08-05 15:44:33.542 xcodebuild[30954:4502547] [MT] DVTDeveloperPaths: Failed to get length of DARWIN_USER_CACHE_DIR from confstr(3), error = Error Domain=NSPOSIXErrorDomain Code=5 "Input/output error". Using NSCachesDirectory instead.
swiftc: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
swiftc: error: couldn't create cache file '/tmp/xcrun_db-IBMgjh6a' (errno=Operation not permitted)
2026-08-05 15:44:34.168 xcodebuild[30958:4502569]  DVTFilePathFSEvents: Failed to start fs event stream.
2026-08-05 15:44:34.367 xcodebuild[30958:4502567] [MT] DVTDeveloperPaths: Failed to get length of DARWIN_USER_CACHE_DIR from confstr(3), error = Error Domain=NSPOSIXErrorDomain Code=5 "Input/output error". Using NSCachesDirectory instead.
codex
Static validation is complete. The default seeded `euler_ancestral_cfg_pp` math matches the video path and upstream ComfyUI CFG++ equations; the remaining defects cluster in supported edge/override paths: non-CFG++ or unseeded ancestral audio, incomplete state copied into audio refine, and one asymmetric gate override. I also confirmed the limiter knee itself is continuous and transparent below the knee.
Completed the review using commit-by-commit diffs, final-code tracing, upstream sampler comparison, and static algebra. The scoped unit-test command was attempted but could not start because the read-only sandbox prevents Xcode/xcrun cache and result-bundle writes.
  ✓ Read repository instructions and identify the last 10 commits/change surface
  ✓ Audit sampler math, RNG isolation, refine gate paths, limiter knee, and state lifetimes
  ✓ Inspect or run scoped unit tests and validate suspected issues
  ✓ Report numbered findings with severity, citations, and deploy confidence
codex
## Findings
1. **Medium — `euler_ancestral` audio ignores ancestral sampling.** `useSDE` is only honored inside the CFG++ branch. Without a negative velocity it returns plain Euler immediately; with a negative but `useCfgPP == false`, it still returns deterministic Euler. Video injects ancestral noise in the equivalent path. [LTX2Guidance.swift:176](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Guidance.swift:176), [LTX2Guidance.swift:186](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Guidance.swift:186), [LTX2Pipeline.swift:1295](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1295)
2. **Medium — unseeded ancestral CFG++ applies `sigmaDown` without its compensating noise.** With `seed == nil`, `audioNoiseKey` is nil. The wiring therefore supplies no `ancestralNoise`, but `ltx2AudioStep` still computes the stochastic `sigmaDown` and merely omits `sigmaUp * noise`. That is neither ancestral sampling nor deterministic CFG++; it progressively removes variance. [LTX2Pipeline.swift:299](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:299), [LTX2Pipeline.swift:1315](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1315), [LTX2Guidance.swift:192](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Guidance.swift:192)
3. **Medium — audio-refine loses its negative context and RNG state at both refine sites.** Both reconstructed `refineAVState` objects copy only positive context and latents, dropping `negativeAudioContext` and `audioNoiseKey`. With `LTX2_AUDIO_REFINE=1`, the negative pass consequently falls back to the video-only transformer, audio gets no negative velocity/CFG++ participation, and nondeterministic refine cannot use its keyed chain. It also mixes dual-stream positive video with video-only unconditional video. [LTX2Pipeline.swift:378](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:378), [LTX2Pipeline.swift:1163](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1163), [LTX2Pipeline.swift:1662](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1662)
4. **Low — T2V’s new early gate regresses `LTX2_REFINE_DECODE_ONLY`.** The shared I2V/refine-anchor path exempts `REFINE_DECODE_ONLY=1` from early native-size return so it can produce the requested upsample-only decode. The bespoke T2V gate does not. Above the volume threshold, T2V now returns native-size unless `REFINE_UPSCALE_ON_SKIP=1` is also set. [LTX2Pipeline.swift:329](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:329), [LTX2Pipeline.swift:1563](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1563), [LTX2Pipeline.swift:1601](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1601)
5. **Low — “bit-identical video with audio on/off” overstates the RNG guarantee.** The keyed chain does preserve the seeded global RNG stream, but video output cannot be bit-identical because audio-on uses `callAV` and receives cross-modal contributions; the adjacent comment already acknowledges it is a different joint model. Reword the invariant to “video RNG sequence remains identical.” [LTX2Pipeline.swift:38](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:38), [LTX2Pipeline.swift:48](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:48), [LTX2Pipeline.swift:1123](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1123)
6. **Low — the updated tuning skill still recommends the placebo `steps` lever.** The new rule correctly says distilled `steps` is a no-op, but the audio-lever table still says speech wants 16 versus foley 8. [.claude/skills/kira-ltx-tuning/SKILL.md:39](/Users/toddwalderman/Projects/zimage.swift/.claude/skills/kira-ltx-tuning/SKILL.md:39), [.claude/skills/kira-ltx-tuning/SKILL.md:52](/Users/toddwalderman/Projects/zimage.swift/.claude/skills/kira-ltx-tuning/SKILL.md:52)
The default seeded `euler_ancestral_cfg_pp` equations match the video path and [upstream ComfyUI CFG++ implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/k_diffusion/sampling.py#L1149-L1191). The limiter knee is value- and slope-continuous, linear through the exact boundary, and asymptotes correctly to 0.85. Both refine gates consistently use strict `>`, so exact-boundary volume refines; `LTX2_REFINE_UPSCALE_ON_SKIP=1` works as intended; the I2V anchor is bypassed only when refine itself is bypassed. The default audio-CFG cap and steps warning are correct.
Tests could not start because the read-only sandbox prevented Xcode/xcrun cache and result-bundle writes; `git diff --check` passed.
**Deploy confidence: conditional GO / moderate.** Current production is safe if it always supplies a seed, uses `euler_ancestral_cfg_pp`, leaves `LTX2_AUDIO_REFINE` off, and does not depend on T2V `REFINE_DECODE_ONLY`. I would fix findings 1–4 before treating the full configurable surface as deployable.
tokens used
232,381
## Findings
1. **Medium — `euler_ancestral` audio ignores ancestral sampling.** `useSDE` is only honored inside the CFG++ branch. Without a negative velocity it returns plain Euler immediately; with a negative but `useCfgPP == false`, it still returns deterministic Euler. Video injects ancestral noise in the equivalent path. [LTX2Guidance.swift:176](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Guidance.swift:176), [LTX2Guidance.swift:186](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Guidance.swift:186), [LTX2Pipeline.swift:1295](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1295)
2. **Medium — unseeded ancestral CFG++ applies `sigmaDown` without its compensating noise.** With `seed == nil`, `audioNoiseKey` is nil. The wiring therefore supplies no `ancestralNoise`, but `ltx2AudioStep` still computes the stochastic `sigmaDown` and merely omits `sigmaUp * noise`. That is neither ancestral sampling nor deterministic CFG++; it progressively removes variance. [LTX2Pipeline.swift:299](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:299), [LTX2Pipeline.swift:1315](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1315), [LTX2Guidance.swift:192](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Guidance.swift:192)
3. **Medium — audio-refine loses its negative context and RNG state at both refine sites.** Both reconstructed `refineAVState` objects copy only positive context and latents, dropping `negativeAudioContext` and `audioNoiseKey`. With `LTX2_AUDIO_REFINE=1`, the negative pass consequently falls back to the video-only transformer, audio gets no negative velocity/CFG++ participation, and nondeterministic refine cannot use its keyed chain. It also mixes dual-stream positive video with video-only unconditional video. [LTX2Pipeline.swift:378](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:378), [LTX2Pipeline.swift:1163](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1163), [LTX2Pipeline.swift:1662](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1662)
4. **Low — T2V’s new early gate regresses `LTX2_REFINE_DECODE_ONLY`.** The shared I2V/refine-anchor path exempts `REFINE_DECODE_ONLY=1` from early native-size return so it can produce the requested upsample-only decode. The bespoke T2V gate does not. Above the volume threshold, T2V now returns native-size unless `REFINE_UPSCALE_ON_SKIP=1` is also set. [LTX2Pipeline.swift:329](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:329), [LTX2Pipeline.swift:1563](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1563), [LTX2Pipeline.swift:1601](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1601)
5. **Low — “bit-identical video with audio on/off” overstates the RNG guarantee.** The keyed chain does preserve the seeded global RNG stream, but video output cannot be bit-identical because audio-on uses `callAV` and receives cross-modal contributions; the adjacent comment already acknowledges it is a different joint model. Reword the invariant to “video RNG sequence remains identical.” [LTX2Pipeline.swift:38](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:38), [LTX2Pipeline.swift:48](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:48), [LTX2Pipeline.swift:1123](/Users/toddwalderman/Projects/zimage.swift/Sources/ZImage/LTX2/LTX2Pipeline.swift:1123)
6. **Low — the updated tuning skill still recommends the placebo `steps` lever.** The new rule correctly says distilled `steps` is a no-op, but the audio-lever table still says speech wants 16 versus foley 8. [.claude/skills/kira-ltx-tuning/SKILL.md:39](/Users/toddwalderman/Projects/zimage.swift/.claude/skills/kira-ltx-tuning/SKILL.md:39), [.claude/skills/kira-ltx-tuning/SKILL.md:52](/Users/toddwalderman/Projects/zimage.swift/.claude/skills/kira-ltx-tuning/SKILL.md:52)
The default seeded `euler_ancestral_cfg_pp` equations match the video path and [upstream ComfyUI CFG++ implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/k_diffusion/sampling.py#L1149-L1191). The limiter knee is value- and slope-continuous, linear through the exact boundary, and asymptotes correctly to 0.85. Both refine gates consistently use strict `>`, so exact-boundary volume refines; `LTX2_REFINE_UPSCALE_ON_SKIP=1` works as intended; the I2V anchor is bypassed only when refine itself is bypassed. The default audio-CFG cap and steps warning are correct.
Tests could not start because the read-only sandbox prevented Xcode/xcrun cache and result-bundle writes; `git diff --check` passed.
**Deploy confidence: conditional GO / moderate.** Current production is safe if it always supplies a seed, uses `euler_ancestral_cfg_pp`, leaves `LTX2_AUDIO_REFINE` off, and does not depend on T2V `REFINE_DECODE_ONLY`. I would fix findings 1–4 before treating the full configurable surface as deployable.
EXIT:0
