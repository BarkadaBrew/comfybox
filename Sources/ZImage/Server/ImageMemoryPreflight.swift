// ImageMemoryPreflight.swift — memory-aware pre-flight for image requests,
// gating high-resolution renders BEFORE any model load (issue #22).
//
// Problem (issue #22): a 2048px DyPE render maxed out 128GB unified memory on
// an M3 Max and swapped to SSD (9.5min denoise vs 1:45 at 1024px); 3072px+
// renders are impossible today without tiling. ComfyBox has no live-memory
// admission check for image requests — `HeavyModelAdmission` (see
// `HeavyModelAdmission.swift`) only gates whether a HEAVY MODEL may be loaded
// (image-vs-video residency), never how much activation memory a specific
// render's resolution will need once a model is already resident.
//
// ## PR #363 review round 1 — what changed and why
//
// The original version of this file modeled a `[heads, tokens, tokens]`
// self-attention score tensor as an O(tokens²) term. That allocation does not
// exist: every attention call in this codebase (`ZImageTransformerBlock`,
// `Krea2Transformer`) goes through `MLXFast.scaledDotProductAttention`, a
// fused/tiled kernel that never materializes the full score matrix. That term
// was deleted; the two things below replace it. Because this file's numbers
// are still UNCALIBRATED against any live memory trace, the estimate is
// ADVISORY by default (`ImageMemoryCapsConfig.enforceMemoryEstimate = false`)
// — see `validate(...)`.
//
// ## Memory model (read before changing any constant below)
//
// 1. **Transformer per-layer working set — O(tokens).** The fused attention
//    kernel's working set is the Q/K/V/O projections plus the SwiGLU FFN
//    intermediate, each `[tokens, hiddenSize]`-ish — linear in tokens, not
//    quadratic. See `estimateBytes` for the exact terms.
// 2. **VAE decode — O(pixels), but only at the resolution each stage actually
//    runs at.** The decoder's widest channel count (512) occurs at LATENT
//    resolution, near the input; only the un-widened RGB output (3 channels)
//    runs at FULL resolution. Modeling the wide channel count at full
//    resolution (the original, since-corrected version of this file) grossly
//    over-estimated — by roughly the VAE's own upsample factor squared.
//
// Both terms are computed from constants that already exist elsewhere in this
// codebase — every one is cited at its definition site below. Where no
// profiling backs a number (there is no live memory trace to calibrate
// against yet — that is what this preflight exists to start collecting), the
// comment says so explicitly; those are ASSUMPTIONS, not measurements, and
// are the first things to recalibrate once real `/health` memory samples
// exist for a range of resolutions (see `docs/user-guide.md`'s "DyPE /
// high-resolution pre-flight" section for the sampling recipe).
//
// This is a heuristic GATE, not a memory predictor: the resolution cap
// (`ImageMemoryCapsConfig.maxLongEdge`/`maxPixels`) is a hard, always-enforced
// refusal that needs no calibration to be trustworthy (it never even computes
// a byte estimate); the live-memory-budget check is the UNCALIBRATED part,
// hence advisory-by-default.

import Foundation

enum ImageMemoryPreflight {

  // MARK: - Overflow-safe UInt64 arithmetic (I6, PR #363 review)

  /// Saturates to `UInt64.max` on overflow rather than trapping. A trap on
  /// pathological input (an out-of-range width/height reaching this pure
  /// function directly, bypassing the resolution cap) would crash the render
  /// process; saturating just makes the estimate huge, which the resolution
  /// cap — itself now bounded, see `ServerConfigStore.validate` — refuses
  /// long before this runs in the wired path.
  static func mulSat(_ a: UInt64, _ b: UInt64) -> UInt64 {
    let (result, overflow) = a.multipliedReportingOverflow(by: b)
    return overflow ? UInt64.max : result
  }

  static func addSat(_ a: UInt64, _ b: UInt64) -> UInt64 {
    let (result, overflow) = a.addingReportingOverflow(b)
    return overflow ? UInt64.max : result
  }

  // MARK: - Family transformer profile

  /// The handful of per-family constants the estimate formula needs. Not a
  /// general model-config type — just what this estimate reads.
  struct TransformerProfile: Equatable {
    let hiddenSize: UInt64
    let ffnHiddenDim: UInt64
    /// Sum of `axesDims / 2` across the RoPE axes — the width of one
    /// per-token frequency-table row DyPE recomputes.
    let axesHalfDimSum: UInt64
  }

  /// Z-Image-Turbo ("flux1", the base/default family). Every field is an
  /// existing engine constant — see `ZImageModelMetadata.Transformer`
  /// (`Sources/ZImage/Support/ModelMetadata.swift:35-46`).
  static let flux1Profile = TransformerProfile(
    hiddenSize: UInt64(ZImageModelMetadata.Transformer.hiddenSize),  // 3840 — ModelMetadata.swift:36
    // SwiGLU hidden width = dim/3*8 (`ZImageTransformerBlock.swift:42`,
    // `ZImageControlTransformerBlock.swift:50`): 3840/3*8 = 10240.
    ffnHiddenDim: UInt64(Float(ZImageModelMetadata.Transformer.hiddenSize) / 3.0 * 8.0),
    // axesDims = [32, 48, 48] (ModelMetadata.swift:44); the image-frequency
    // table's per-token width is `axesDims.reduce { $0 + $1/2 }`
    // (`ZImageRopeEmbedder.swift:128`) = 16+24+24 = 64.
    axesHalfDimSum: UInt64(ZImageModelMetadata.Transformer.axesDims.reduce(0) { $0 + $1 / 2 }))

  /// Krea-2 — the production image model (`intent.md`). Constants are
  /// `Krea2Config`'s own defaults (`Sources/ZImage/Krea2/Transformer/Krea2Transformer.swift:15-38`).
  static let krea2Profile = TransformerProfile(
    hiddenSize: 6144,  // Krea2Config.features — Krea2Transformer.swift:16
    // SwiGLU hidden width = (2*features/3)*multiplier (`Krea2Transformer.swift:91-92`),
    // multiplier=4 (`Krea2Transformer.swift:21`): (2*6144/3)*4 = 16384.
    ffnHiddenDim: (2 * 6144 / 3) * 4,
    // Krea2Config.headDim = features/heads = 6144/48 = 128 (`Krea2Transformer.swift:32`);
    // Krea2Config.axes sums to headDim (`Krea2Transformer.swift:33-37`), so the
    // per-token frequency-table width (half of that) is 64.
    axesHalfDimSum: 64)

  /// Family → profile. `flux2`/`fibo`/`chroma` have no compiled-in transformer
  /// constants in this codebase today — their configs are inferred from the
  /// actual weight-file tensor shapes at load time
  /// (`ModelConfigs.inferTransformerConfig`, `Sources/ZImage/Weights/ModelConfigs.swift:333-375`),
  /// not known ahead of a model load. Defaults to `krea2Profile` (the larger
  /// of the two this file can cite) for an unrecognized family — see I4 in
  /// `resolvedFamily(model:warmFamily:)` for why an EXPLICIT `model` still
  /// resolves accurately and only a truly unknown one falls back like this.
  static func profile(for family: WarmModelFamily) -> TransformerProfile {
    switch family {
    case .flux1: return flux1Profile
    case .krea2: return krea2Profile
    case .flux2, .fibo, .chroma: return krea2Profile
    }
  }

  // MARK: - Shared constants

  /// Compute dtype for the denoising loop — bf16 (`Krea2Pipeline.swift:961`,
  /// `ZImagePipeline.swift:687`). Applied uniformly to the VAE decode estimate
  /// too, rather than guessing a second unverified dtype.
  static let bytesPerElement: UInt64 = 2

  /// Combined pixel→token divisor: VAE spatial downsample (8 —
  /// `Krea2VAE.swift:312`) × DiT patch size (2 — `ZImageTransformer2D.swift:221`,
  /// `Krea2Transformer.swift:23`). Both families share this arithmetic — it is
  /// the same "1024/16 = 64 base tokens" computation already inline at
  /// `Krea2Pipeline.swift:136-141` and `ZImageTransformer2D.swift:244-245`.
  static let pixelsPerToken: UInt64 = 16

  /// VAE spatial downsample factor alone (8 — `Krea2VAE.swift:312`) — the
  /// LATENT resolution the widest decoder channel count actually runs at,
  /// distinct from `pixelsPerToken` above (which also folds in the DiT patch
  /// size and is a transformer-token count, not a VAE-latent count).
  static let vaeSpatialScale: UInt64 = 8

  /// DyPE's own base training resolution default (`DyPEConfig.baseResolution`,
  /// `Sources/ZImage/Weights/ModelConfigs.swift:26,44` = 1024) — the same
  /// threshold `resolvedDyPEConfig` auto-enables above
  /// (`WarmServer.swift:9698`, `> 1024`) and `ZImageTransformer2D.forward`
  /// gates its RoPE rescale on (`hScale > 1.0`, `ZImageTransformer2D.swift:249`).
  static let dyPEBaseResolution = 1024

  /// ASSUMPTION — not a measurement. MLX evaluates its compute graph lazily
  /// and functionally (ops build new buffers rather than mutating in place),
  /// so more than exactly one transformer layer's activations can be live at
  /// a render's peak: residual/skip references, async graph scheduling, and
  /// the buffer-pool's own retention all push the true peak above a naive
  /// single-layer estimate. `2` is a deliberately conservative safety
  /// multiplier picked to round UP, not a profiled figure — retune it against
  /// real `/health` memory samples (`StatsProvider.swift`) once this gate has
  /// live traffic to learn from.
  static let concurrentLayerSafetyMultiplier: UInt64 = 2

  /// ASSUMPTION — not a measurement. The decoder's widest block (512
  /// channels, `ZImageModelMetadata.VAE.blockOutChannels.max()`,
  /// `Sources/ZImage/Support/ModelMetadata.swift:32`) is modeled as running
  /// at LATENT resolution (see `vaeSpatialScale` above) — the resolution it
  /// is actually closest to in a standard 4-stage decoder (channels shrink
  /// 512→512→256→128 as each stage doubles H/W) — rather than at full output
  /// resolution, which the original version of this file assumed and which
  /// PR #363's review identified as a large over-estimate.
  static let vaeLatentStageChannelWidth: UInt64 = UInt64(ZImageModelMetadata.VAE.blockOutChannels.max() ?? 512)

  /// RGB output channels — the only term of the VAE estimate that runs at
  /// FULL output resolution. Standard for an image decoder; not itself a
  /// per-checkpoint constant in this codebase (`ZImageVAEConfig.outChannels`
  /// is JSON-configured per checkpoint, but every shipped VAE decodes to RGB).
  static let vaeOutputChannels: UInt64 = 3

  /// Defensive ceiling on the per-axis token count: pure functions still need
  /// to behave for a fuzzed/direct call with an out-of-range width/height, so
  /// `estimateBytes` saturates to `UInt64.max` past this point rather than
  /// relying on the overflow-safe multiplies alone to keep the number
  /// meaningful. In the wired path this never binds — the resolution cap
  /// (now bounded, `ServerConfigStore.validate`) refuses long before a
  /// request could reach it.
  static let tokensPerAxisOverflowGuard: UInt64 = 20_000

  // MARK: - Token / latent counts

  /// Latent-patch token count along one axis. Rounds UP: a request whose
  /// pixel dimension is not a multiple of 16 still allocates for the padded
  /// token grid, so truncating would under-estimate.
  static func tokensPerAxis(_ pixels: Int) -> UInt64 {
    guard pixels > 0 else { return 0 }
    return (UInt64(pixels) + pixelsPerToken - 1) / pixelsPerToken
  }

  /// VAE latent-resolution extent along one axis (spatial downsample only,
  /// no patch folding — see `vaeSpatialScale`).
  static func latentAxis(_ pixels: Int) -> UInt64 {
    guard pixels > 0 else { return 0 }
    return (UInt64(pixels) + vaeSpatialScale - 1) / vaeSpatialScale
  }

  /// Same auto-enable threshold `GeneratePayload.resolvedDyPEConfig` applies
  /// (`WarmServer.swift:9698`, `max(width,height) > 1024`) — exposed here so a
  /// caller with no explicit `dype` field of its own (the ComfyUI bridge's
  /// `ComfyBridgeGenerateRequest` has none) can derive the same default this
  /// gate assumes, without duplicating the threshold constant.
  static func autoDyPEEnabled(width: Int, height: Int) -> Bool {
    max(width, height) > dyPEBaseResolution
  }

  // MARK: - Estimate

  /// Peak activation-memory estimate for one image render at `width`×`height`,
  /// in bytes. Pure — no live memory probing, no file/model I/O — so it is
  /// safe to call before any model load. UNCALIBRATED against any live memory
  /// trace (see the file header) — this is why callers treat it as advisory
  /// by default.
  ///
  /// Formula (every constant is cited above; PR #363 review round 1 — the
  /// O(tokens²) attention term was deleted, it modeled an allocation the
  /// fused `MLXFast.scaledDotProductAttention` kernel never makes):
  /// ```
  /// tokens = ceil(width/16) * ceil(height/16)
  /// perLayer = tokens * (4*hiddenSize + 2*ffnHiddenDim) * bytesPerElement   // Q/K/V/O projections + SwiGLU gate/up — O(tokens)
  /// transformer = perLayer * concurrentLayerSafetyMultiplier
  /// dype extra (only when dype && tokens > baseTokens, matching ZImageTransformer2D.swift:249):
  ///   tokens*axesHalfDimSum*2*bytesPerElement                             // recomputed NTK freq table (ZImageRopeEmbedder.swift:118-138)
  ///   + tokens*3*4                                                        // retained image position ids (TransformerCacheBuilder.swift:37,72)
  /// latentPixels = ceil(width/8) * ceil(height/8)
  /// vae = latentPixels*vaeLatentStageChannelWidth*bytesPerElement          // the decoder's widest block, at LATENT resolution
  ///     + width*height*vaeOutputChannels*bytesPerElement                  // the RGB output, at FULL resolution
  /// estimate = transformer + dype extra + vae
  /// ```
  static func estimateBytes(width: Int, height: Int, family: WarmModelFamily, dype: Bool) -> UInt64 {
    guard width > 0, height > 0 else { return 0 }
    let tokensH = tokensPerAxis(height)
    let tokensW = tokensPerAxis(width)
    guard tokensH <= tokensPerAxisOverflowGuard, tokensW <= tokensPerAxisOverflowGuard else {
      return UInt64.max
    }
    let tokens = mulSat(tokensH, tokensW)
    let p = profile(for: family)

    // Q, K, V, O projections (each ~hiddenSize-wide — GQA narrows K/V in
    // practice, e.g. krea2's kvheads=12 vs heads=48, but this simplified
    // O(tokens·heads·headDim) grouping treats all four uniformly, which
    // over-estimates K/V slightly; conservative, not exact) + SwiGLU
    // gate/up (each ffnHiddenDim-wide).
    let perTokenLinearUnits = addSat(mulSat(4, p.hiddenSize), mulSat(2, p.ffnHiddenDim))
    let linearBytes = mulSat(mulSat(tokens, perTokenLinearUnits), bytesPerElement)
    let transformerBytes = mulSat(linearBytes, concurrentLayerSafetyMultiplier)

    let baseTokensPerAxis = tokensPerAxis(dyPEBaseResolution)
    let baseTokens = mulSat(baseTokensPerAxis, baseTokensPerAxis)
    var dypeExtraBytes: UInt64 = 0
    if dype, tokens > baseTokens {
      let freqTableBytes = mulSat(mulSat(tokens, p.axesHalfDimSum), mulSat(2, bytesPerElement))
      let posIdsBytes = mulSat(mulSat(tokens, 3), 4)
      dypeExtraBytes = addSat(freqTableBytes, posIdsBytes)
    }

    let latH = latentAxis(height)
    let latW = latentAxis(width)
    let latentPixels = mulSat(latH, latW)
    let vaeLatentStageBytes = mulSat(mulSat(latentPixels, vaeLatentStageChannelWidth), bytesPerElement)
    let fullResPixels = mulSat(UInt64(width), UInt64(height))
    let vaeRGBBytes = mulSat(mulSat(fullResPixels, vaeOutputChannels), bytesPerElement)
    let vaeBytes = addSat(vaeLatentStageBytes, vaeRGBBytes)

    return addSat(addSat(transformerBytes, dypeExtraBytes), vaeBytes)
  }

  // MARK: - Admission decisions

  struct Decision: Sendable, Equatable {
    let allow: Bool
    let reason: String
  }

  /// Pure resolution-cap decision — no memory probing at all, so an obviously
  /// oversized request (issue #22's test plan: 6000×6000) fails fast before
  /// `MemoryProbe` is even consulted. ALWAYS a hard refusal when violated —
  /// unlike the memory-budget check below, this does not have an advisory
  /// mode (C1b, PR #363 review): it needs no calibration to be trustworthy.
  static func decideResolution(width: Int, height: Int, caps: ImageMemoryCapsConfig) -> Decision {
    // I6: non-positive or pathologically large dimensions refuse here too —
    // `width * height` as plain `Int` would otherwise risk a TRAP (not just
    // an under/over-count) on a deliberately huge value before the cap
    // comparison ever runs, which is exactly backwards for the ONE check
    // that is supposed to fail fast with no risk at all.
    guard width > 0, height > 0 else {
      return Decision(allow: false, reason: "\(width)x\(height): width and height must both be > 0")
    }
    let longEdge = max(width, height)
    if longEdge > caps.maxLongEdge {
      return Decision(
        allow: false,
        reason: "\(width)x\(height): long edge \(longEdge)px exceeds the \(caps.maxLongEdge)px resolution cap")
    }
    let pixels = mulSat(UInt64(width), UInt64(height))
    if pixels > safeMaxPixels(caps) {
      return Decision(
        allow: false,
        reason: "\(width)x\(height) = \(pixels) pixels exceeds the \(caps.maxPixels)-pixel resolution cap")
    }
    return Decision(allow: true, reason: "\(width)x\(height) within resolution caps")
  }

  /// Point-of-use defensive clamp (fix round 2, PR #363 review): `caps`
  /// SHOULD already be valid — `ServerConfigStore` validates on write and
  /// now sanitizes on load (`sanitizeImageMemoryCaps`) — but this function is
  /// also reachable with a directly-constructed `ImageMemoryCapsConfig`
  /// (tests, or a future caller that bypasses the store), and `UInt64(-1)`
  /// TRAPS. `max(0, …)` before the conversion turns a negative/garbage
  /// `maxPixels` into "refuse everything" instead of a crash — never a
  /// silent pass-through of nonsense.
  static func safeMaxPixels(_ caps: ImageMemoryCapsConfig) -> UInt64 {
    UInt64(max(0, caps.maxPixels))
  }

  /// Same defensive posture as `safeMaxPixels` for the headroom fraction:
  /// `min`/`max` do NOT reliably filter out NaN (a `<`/`>` comparison against
  /// NaN is always false, so `max(.nan, 0.0)` can return `.nan` right back
  /// out) — an explicit `.isFinite` check is required, not just a range
  /// clamp. A non-finite value falls back to `ImageMemoryCapsConfig.default`'s
  /// headroom fraction, per the review.
  static func safeHeadroomFraction(_ caps: ImageMemoryCapsConfig) -> Double {
    let raw = caps.minAvailableHeadroomFraction
    guard raw.isFinite else { return ImageMemoryCapsConfig.default.minAvailableHeadroomFraction }
    return min(max(raw, 0.0), 1.0)
  }

  /// Pure memory-budget decision: does `estimate` fit under `cap` bytes, given
  /// `available` live free system memory right now? Mirrors
  /// `HeavyModelAdmission.decide` (`HeavyModelAdmission.swift:83-100`) — pure
  /// and unit-testable, with no probing inside the function itself. `cap` is
  /// supplied by the caller (typically `available * (1 - headroomFraction)`,
  /// see `validate(...)` below) so this stays a simple, injectable comparison.
  static func decide(estimate: UInt64, available: UInt64, cap: UInt64) -> Decision {
    if estimate > cap {
      return Decision(
        allow: false,
        reason: "estimated \(estimate >> 20)MB exceeds the \(cap >> 20)MB memory cap "
          + "(\(available >> 20)MB available right now)")
    }
    return Decision(
      allow: true,
      reason: "estimated \(estimate >> 20)MB within the \(cap >> 20)MB memory cap "
        + "(\(available >> 20)MB available right now)")
  }

  // MARK: - Outcome

  /// What `validate(...)` learned, for a caller to log/surface even when
  /// nothing was refused (C1b, PR #363 review: the memory estimate is
  /// advisory by default, so a caller needs this to warn and to stamp
  /// `memory_estimate_bytes`/`memory_available_bytes` on the eventual
  /// response — see `GeneratePayload.memoryEstimateBytes`).
  struct Outcome: Sendable, Equatable {
    let estimateBytes: UInt64
    let availableBytes: UInt64
    let capBytes: UInt64
    /// Whether the estimate fit the live memory budget. `false` here did NOT
    /// necessarily throw — see `enforceMemoryEstimate`.
    let withinBudget: Bool
    let reason: String
  }

  /// The full request-validation gate (issue #22). Two independent checks:
  ///
  /// 1. **Resolution cap** — no probing, ALWAYS a hard refusal when violated
  ///    (413/400 depending on caller — see `WarmServerError.imageMemoryPreflightRefused`).
  /// 2. **Live memory budget** — computed and returned as an `Outcome` either
  ///    way; only THROWS when `caps.enforceMemoryEstimate` is true (C1b: the
  ///    estimate formula is uncalibrated, so refusing on it by default risked
  ///    refusing every krea2 2K render outright — krea2 alone footprints
  ///    ~75GB resident, leaving well under this file's advisory cap on a
  ///    128GB machine even before any activation estimate). When not
  ///    enforced, the caller is expected to log `outcome.reason` when
  ///    `!outcome.withinBudget` and forward `outcome` into the response.
  ///
  /// Callers run this BEFORE any model load, and ONLY at submission — never
  /// on already-accepted/replayed jobs (C2) — see
  /// `WarmServer.decodedGeneratePayload`'s `gateSubmission` parameter and
  /// `ComfyBridge.handlePrompt`.
  static func validate(
    width: Int, height: Int, family: WarmModelFamily, dype: Bool,
    caps: ImageMemoryCapsConfig, availableBytes: UInt64
  ) throws -> Outcome {
    let resolutionDecision = decideResolution(width: width, height: height, caps: caps)
    guard resolutionDecision.allow else {
      throw WarmServerError.imageMemoryPreflightRefused(
        code: "resolution_cap", reason: resolutionDecision.reason,
        estimateBytes: nil, availableBytes: nil, capBytes: safeMaxPixels(caps))
    }

    let estimate = estimateBytes(width: width, height: height, family: family, dype: dype)
    let headroomFraction = safeHeadroomFraction(caps)
    let budgetCap = UInt64(Double(availableBytes) * (1.0 - headroomFraction))
    let memoryDecision = decide(estimate: estimate, available: availableBytes, cap: budgetCap)
    let outcome = Outcome(
      estimateBytes: estimate, availableBytes: availableBytes, capBytes: budgetCap,
      withinBudget: memoryDecision.allow, reason: memoryDecision.reason)

    if !memoryDecision.allow, caps.enforceMemoryEstimate {
      throw WarmServerError.imageMemoryPreflightRefused(
        code: "insufficient_memory", reason: memoryDecision.reason,
        estimateBytes: estimate, availableBytes: availableBytes, capBytes: budgetCap)
    }
    return outcome
  }

  /// `payload.model` resolved to a `WarmModelFamily` via the same canonical-name
  /// matching the desktop's sampling-option facade uses
  /// (`SamplingRecipeCatalog.canonicalFamily`, `FamilyRecipeMatrix.swift:256-279`).
  ///
  /// I4 (PR #363 review): when no explicit `model` is given, use the
  /// WARM/active family when the caller knows it (`warmFamily` — the render
  /// will actually run on it, so this is the ACCURATE answer, not a
  /// conservative guess), else fall back to `.flux1`, the LIGHTER of the two
  /// profiles this file can cite. Never default to the heaviest — the
  /// original version of this file defaulted an unspecified model to
  /// `.krea2` specifically to avoid under-estimating, but with the O(tokens²)
  /// term gone (and the estimate advisory by default besides) that
  /// conservatism bought little and cost every flux1 render an inflated
  /// number; correctness (using the real active family when known) is a
  /// strictly better trade now.
  static func resolvedFamily(model: String?, warmFamily: WarmModelFamily? = nil) -> WarmModelFamily {
    if let raw = model,
       let canonical = SamplingRecipeCatalog.canonicalFamily(raw),
       let resolved = WarmModelFamily(rawValue: canonical) {
      return resolved
    }
    return warmFamily ?? .flux1
  }
}
