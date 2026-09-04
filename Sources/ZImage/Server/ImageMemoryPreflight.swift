// ImageMemoryPreflight.swift — memory-aware pre-flight for image requests,
// gating DyPE / high-resolution renders BEFORE any model load (issue #22).
//
// Problem (issue #22): a 2048px DyPE render maxed out 128GB unified memory on
// an M3 Max and swapped to SSD (9.5min denoise vs 1:45 at 1024px); 3072px+
// renders are impossible today without tiling. ComfyBox has no live-memory
// admission check for image requests — `HeavyModelAdmission` (see
// `HeavyModelAdmission.swift`) only gates whether a HEAVY MODEL may be loaded
// (image-vs-video residency), never how much activation memory a specific
// render's resolution will need once a model is already resident. This file
// is the missing gate: a fast, pure, unit-testable estimate of a render's
// peak activation memory, plus the admission decision built on top of it.
//
// ## Memory model (read before changing any constant below)
//
// Two terms dominate a render's peak activation footprint:
//
// 1. **Transformer joint-attention — O(tokens²).** DyPE (`DyPEConfig`,
//    `Sources/ZImage/Weights/ModelConfigs.swift:17-67`) does not change the
//    transformer architecture; it rescales RoPE frequencies so the SAME joint
//    self-attention runs over MORE tokens at high resolution
//    (`ZImageTransformer2D.forward`, `Sources/ZImage/Model/Transformer/ZImageTransformer2D.swift:239-255`).
//    The attention score tensor is `[heads, tokens, tokens]` — quadratic in
//    tokens — which is the direct mechanism behind the reported 2048px
//    slowdown/swap: token count only grows 4x from 1024→2048px, but the
//    attention term grows 16x.
//
// 2. **VAE decode — O(pixels).** `AutoencoderKL`'s decoder upsamples the
//    latent back to full resolution; its peak intermediate feature map scales
//    with output pixel count (`ZImageModelMetadata.VAE.blockOutChannels`,
//    `Sources/ZImage/Support/ModelMetadata.swift:32`).
//
// Both terms are computed from constants that already exist elsewhere in this
// codebase — every one is cited at its definition site below. Where no
// profiling backs a number (there is no live memory trace to calibrate
// against yet — that is what this preflight exists to start collecting), the
// comment says so explicitly; those are ASSUMPTIONS, not measurements, and
// are the first things to recalibrate once real `/health` memory samples
// exist for a range of resolutions.
//
// This is a heuristic GATE, not a memory predictor: the resolution cap
// (`ImageMemoryCapsConfig.maxLongEdge`/`maxPixels`) is what actually keeps
// pathological requests out fast (no probing needed); the live-memory check
// is what adapts to whatever headroom the shared Mac happens to have right
// now, alongside LM Studio / the embeddings service / a resident video model
// (`intent.md` — "Memory is a shared resource").

import Foundation

enum ImageMemoryPreflight {

  // MARK: - Family transformer profile

  /// The handful of per-family constants the O(tokens²)/O(tokens) formula
  /// needs. Not a general model-config type — just what this estimate reads.
  struct TransformerProfile: Equatable {
    let hiddenSize: Int
    let heads: Int
    /// SwiGLU FFN hidden width (gate + up projections, each this wide).
    let ffnHiddenDim: Int
    /// Sum of `axesDims / 2` across the RoPE axes — the width of one
    /// per-token frequency-table row DyPE recomputes.
    let axesHalfDimSum: Int
  }

  /// Z-Image-Turbo ("flux1", the base/default family). Every field is an
  /// existing engine constant — see `ZImageModelMetadata.Transformer`
  /// (`Sources/ZImage/Support/ModelMetadata.swift:35-46`).
  static let flux1Profile = TransformerProfile(
    hiddenSize: ZImageModelMetadata.Transformer.hiddenSize,  // 3840 — ModelMetadata.swift:36
    heads: ZImageModelMetadata.Transformer.heads,             // 30 — ModelMetadata.swift:39
    // SwiGLU hidden width = dim/3*8 (`ZImageTransformerBlock.swift:42`,
    // `ZImageControlTransformerBlock.swift:50`): 3840/3*8 = 10240.
    ffnHiddenDim: Int(Float(ZImageModelMetadata.Transformer.hiddenSize) / 3.0 * 8.0),
    // axesDims = [32, 48, 48] (ModelMetadata.swift:44); the image-frequency
    // table's per-token width is `axesDims.reduce { $0 + $1/2 }`
    // (`ZImageRopeEmbedder.swift:128`) = 16+24+24 = 64.
    axesHalfDimSum: ZImageModelMetadata.Transformer.axesDims.reduce(0) { $0 + $1 / 2 })

  /// Krea-2 — the production image model (`intent.md`). Constants are
  /// `Krea2Config`'s own defaults (`Sources/ZImage/Krea2/Transformer/Krea2Transformer.swift:15-38`).
  static let krea2Profile = TransformerProfile(
    hiddenSize: 6144,  // Krea2Config.features — Krea2Transformer.swift:16
    heads: 48,          // Krea2Config.heads — Krea2Transformer.swift:19
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
  /// not known ahead of a model load. ASSUMPTION: default the unresolved
  /// families to `krea2Profile` — the larger of the two profiles this file
  /// can actually cite — so an unrecognized family never UNDER-estimates.
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
  static let bytesPerElement = 2

  /// Combined pixel→token divisor: VAE spatial downsample (8 —
  /// `Krea2VAE.swift:312`) × DiT patch size (2 — `ZImageTransformer2D.swift:221`,
  /// `Krea2Transformer.swift:23`). Both families share this arithmetic — it is
  /// the same "1024/16 = 64 base tokens" computation already inline at
  /// `Krea2Pipeline.swift:136-141` and `ZImageTransformer2D.swift:244-245`.
  static let pixelsPerToken = 16

  /// DyPE's own base training resolution default (`DyPEConfig.baseResolution`,
  /// `Sources/ZImage/Weights/ModelConfigs.swift:26,44` = 1024) — the same
  /// threshold `resolvedDyPEConfig` auto-enables above
  /// (`WarmServer.swift:9691`, `> 1024`) and `ZImageTransformer2D.forward`
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
  static let concurrentLayerSafetyMultiplier = 2

  /// ASSUMPTION — not a measurement. The VAE decoder's peak intermediate
  /// feature map is modeled as if the decoder's WIDEST channel count
  /// (`ZImageModelMetadata.VAE.blockOutChannels.max()`,
  /// `Sources/ZImage/Support/ModelMetadata.swift:32` = 512) persisted at the
  /// FULL output resolution. It does not — each upsample stage halves
  /// channels while doubling H/W, so the real peak is lower. That is
  /// deliberate: a preflight gate should round up, not down.
  static let vaeDecodeChannelWidth = ZImageModelMetadata.VAE.blockOutChannels.max() ?? 512

  /// Defensive ceiling on the per-axis token count: beyond this, `heads *
  /// tokens²` would overflow `Int` (64-bit) before the resolution cap ever
  /// gets a chance to refuse the request in the wired path. Pure functions
  /// still need to behave for a fuzzed/direct call, so `estimateBytes`
  /// saturates to `UInt64.max` past this point instead of trapping.
  static let tokensPerAxisOverflowGuard = 20_000

  // MARK: - Token count

  /// Latent-patch token count along one axis. Rounds UP: a request whose
  /// pixel dimension is not a multiple of 16 still allocates for the padded
  /// token grid, so truncating would under-estimate.
  static func tokensPerAxis(_ pixels: Int) -> Int {
    guard pixels > 0 else { return 0 }
    return (pixels + pixelsPerToken - 1) / pixelsPerToken
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
  /// safe to call before any model load (issue #22, requirement 4).
  ///
  /// Formula (every constant is cited above):
  /// ```
  /// tokens = ceil(width/16) * ceil(height/16)
  /// perLayer = tokens*(3*hiddenSize + 2*ffnHiddenDim)*bytesPerElement   // QKV + SwiGLU gate/up
  ///          + heads*tokens²*bytesPerElement                           // joint self-attention scores (O(tokens²) — the DyPE-inflated term)
  /// transformer = perLayer * concurrentLayerSafetyMultiplier
  /// dype extra (only when dype && tokens > baseTokens, matching ZImageTransformer2D.swift:249):
  ///   tokens*axesHalfDimSum*2*bytesPerElement                          // recomputed NTK freq table (ZImageRopeEmbedder.swift:118-138)
  ///   + tokens*3*4                                                     // retained image position ids (TransformerCacheBuilder.swift:37,72)
  /// vae = width*height*vaeDecodeChannelWidth*bytesPerElement           // O(pixels)
  /// estimate = transformer + dype extra + vae
  /// ```
  static func estimateBytes(width: Int, height: Int, family: WarmModelFamily, dype: Bool) -> UInt64 {
    guard width > 0, height > 0 else { return 0 }
    let tokensH = tokensPerAxis(height)
    let tokensW = tokensPerAxis(width)
    guard tokensH <= tokensPerAxisOverflowGuard, tokensW <= tokensPerAxisOverflowGuard else {
      return UInt64.max
    }
    let tokens = tokensH * tokensW
    let p = profile(for: family)

    let qkvAndFFNBytes = tokens * (3 * p.hiddenSize + 2 * p.ffnHiddenDim) * bytesPerElement
    let attnBytes = p.heads * tokens * tokens * bytesPerElement
    let transformerBytes = (qkvAndFFNBytes + attnBytes) * concurrentLayerSafetyMultiplier

    let baseTokensPerAxis = tokensPerAxis(dyPEBaseResolution)
    let baseTokens = baseTokensPerAxis * baseTokensPerAxis
    var dypeExtraBytes = 0
    if dype, tokens > baseTokens {
      let freqTableBytes = tokens * p.axesHalfDimSum * 2 * bytesPerElement
      let posIdsBytes = tokens * 3 * 4
      dypeExtraBytes = freqTableBytes + posIdsBytes
    }

    let vaeDecodeBytes = width * height * vaeDecodeChannelWidth * bytesPerElement

    let total = transformerBytes + dypeExtraBytes + vaeDecodeBytes
    return UInt64(max(0, total))
  }

  // MARK: - Admission decisions

  struct Decision: Sendable, Equatable {
    let allow: Bool
    let reason: String
  }

  /// Pure resolution-cap decision — no memory probing at all, so an obviously
  /// oversized request (issue #22's test plan: 6000×6000) fails fast before
  /// `MemoryProbe` is even consulted.
  static func decideResolution(width: Int, height: Int, caps: ImageMemoryCapsConfig) -> Decision {
    let longEdge = max(width, height)
    if longEdge > caps.maxLongEdge {
      return Decision(
        allow: false,
        reason: "\(width)x\(height): long edge \(longEdge)px exceeds the \(caps.maxLongEdge)px resolution cap")
    }
    let pixels = width * height
    if pixels > caps.maxPixels {
      return Decision(
        allow: false,
        reason: "\(width)x\(height) = \(pixels) pixels exceeds the \(caps.maxPixels)-pixel resolution cap")
    }
    return Decision(allow: true, reason: "\(width)x\(height) within resolution caps")
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

  /// The full request-validation gate (issue #22): resolution cap first (no
  /// probing), then a live-memory budget check. Throws
  /// `WarmServerError.imageMemoryPreflightRefused` naming the estimate, the
  /// available budget and the cap. Callers run this BEFORE any model load —
  /// see `WarmServer.decodedGeneratePayload` and `ComfyBridge.handlePrompt`.
  static func validate(
    width: Int, height: Int, family: WarmModelFamily, dype: Bool,
    caps: ImageMemoryCapsConfig, availableBytes: UInt64
  ) throws {
    let resolutionDecision = decideResolution(width: width, height: height, caps: caps)
    guard resolutionDecision.allow else {
      throw WarmServerError.imageMemoryPreflightRefused(
        code: "resolution_cap", reason: resolutionDecision.reason,
        estimateBytes: nil, availableBytes: nil, capBytes: UInt64(caps.maxPixels))
    }

    let estimate = estimateBytes(width: width, height: height, family: family, dype: dype)
    let headroomFraction = min(max(caps.minAvailableHeadroomFraction, 0.0), 1.0)
    let budgetCap = UInt64(Double(availableBytes) * (1.0 - headroomFraction))
    let memoryDecision = decide(estimate: estimate, available: availableBytes, cap: budgetCap)
    guard memoryDecision.allow else {
      throw WarmServerError.imageMemoryPreflightRefused(
        code: "insufficient_memory", reason: memoryDecision.reason,
        estimateBytes: estimate, availableBytes: availableBytes, capBytes: budgetCap)
    }
  }

  /// `payload.model` resolved to a `WarmModelFamily` via the same canonical-name
  /// matching the desktop's sampling-option facade uses
  /// (`SamplingRecipeCatalog.canonicalFamily`, `FamilyRecipeMatrix.swift:256-279`).
  /// ASSUMPTION: when no model override is given, the actual family the
  /// coordinator will dequeue against is live actor state
  /// (`WarmServerCoordinator.currentModelFamily`, `WarmServer.swift:6189`) that
  /// this SYNCHRONOUS pre-model-load gate cannot read without an actor hop.
  /// Defaulting to `.krea2` — the larger of the two profiles this file can
  /// cite, and the production model per `intent.md` — is the conservative
  /// choice: it never under-estimates a real krea2 render, at the cost of
  /// occasionally over-refusing a flux1 render that would have fit.
  static func resolvedFamily(model: String?) -> WarmModelFamily {
    SamplingRecipeCatalog.canonicalFamily(model).flatMap(WarmModelFamily.init(rawValue:)) ?? .krea2
  }
}
