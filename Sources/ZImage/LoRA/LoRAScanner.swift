// LoRAScanner.swift — Filesystem scanner + auto-detection for LoRA files
//
// Part of the LoRA Library Manager (#73). Reads safetensors headers to
// extract metadata, detect model compatibility, infer format and rank.
//
// Uses the existing SafeTensorsReader to read file headers without loading
// full tensor data into memory.

import Foundation

// MARK: - Scan Result

/// Result of scanning a single safetensors LoRA file.
public struct LoRAScanResult: Sendable {
  /// Detected model compatibility tags (e.g. ["z-image"], ["klein-9b"]).
  public let compatibility: [String]
  /// Adapter format: .lora, .lokr, or .slider.
  public let format: LoRAFormat
  /// Inferred LoRA rank.
  public let rank: Int
  /// LoRA alpha from metadata (nil if not present).
  public let alpha: Float?
  /// Total tensor key count (excluding __metadata__).
  public let keyCount: Int
  /// Which layer types are targeted.
  public let layerTargets: [String]
  /// Candidate trigger/activation words auto-extracted from kohya-style
  /// `ss_tag_frequency` training metadata, if present. Empty if the LoRA
  /// wasn't trained with that tooling or no tags were captured.
  public let triggerWords: [String]
  /// Raw metadata from the safetensors __metadata__ header.
  public let safetensorsMetadata: [String: String]?
}

// MARK: - Scanner Errors

public enum LoRAScannerError: Error, LocalizedError {
  case notSafetensors(URL)
  case readFailed(URL, Error)

  public var errorDescription: String? {
    switch self {
    case .notSafetensors(let url):
      return "Not a safetensors file: \(url.lastPathComponent)"
    case .readFailed(let url, let error):
      return "Failed to read \(url.lastPathComponent): \(error.localizedDescription)"
    }
  }
}

// MARK: - Scanner

/// Scans safetensors files and extracts LoRA metadata without loading tensor data.
public enum LoRAScanner {

  /// #313 (review round 1): the complete vocabulary `detectCompatibility`/
  /// `detectCompatibilityFromKeys` can emit — every literal string returned
  /// anywhere in this file, plus `"unknown"`. Single source of truth so the
  /// `/v1/loras/{id}/update` route can reject a manually-supplied
  /// `model_compatibility` tag the scanner itself could never produce,
  /// mirroring how `krea2_relative` already 400s on an unrecognized value.
  public static let knownCompatibilityTags: Set<String> = [
    "z-image", "klein-9b", "klein-4b", "chroma", "flux1", "krea2", "ltx", "unknown",
  ]

  /// Scan a safetensors file and extract all discoverable metadata.
  ///
  /// - Parameter url: Path to a .safetensors file.
  /// - Returns: A `LoRAScanResult` with detected compatibility, format, rank, etc.
  /// - Throws: `LoRAScannerError` if the file cannot be read or is not safetensors.
  public static func analyze(_ url: URL) throws -> LoRAScanResult {
    guard url.pathExtension.lowercased() == "safetensors" else {
      throw LoRAScannerError.notSafetensors(url)
    }

    let reader: SafeTensorsReader
    do {
      reader = try SafeTensorsReader(fileURL: url)
    } catch {
      throw LoRAScannerError.readFailed(url, error)
    }

    let allKeys = reader.tensorNames
    let metadata = reader.fileMetadata.isEmpty ? nil : reader.fileMetadata

    let compatibility = detectCompatibility(metadata: metadata, sampleKeys: allKeys)
    let format = detectFormat(keys: allKeys)
    let rank = inferRank(keys: allKeys, metadata: metadata, reader: reader)
    let alpha = extractAlpha(metadata: metadata)
    let layerTargets = detectLayerTargets(keys: allKeys)
    // CivitAI downloads (Desktop's CivitAIBrowserView) write a sidecar with
    // curated trigger words — prefer that over ss_tag_frequency auto-
    // extraction when present, since it's the model author's own list
    // rather than a training-caption tag histogram.
    let triggerWords = civitaiSidecarTriggerWords(for: url) ?? extractTriggerWords(metadata: metadata)

    return LoRAScanResult(
      compatibility: compatibility,
      format: format,
      rank: rank,
      alpha: alpha,
      keyCount: allKeys.count,
      layerTargets: layerTargets,
      triggerWords: triggerWords,
      safetensorsMetadata: metadata
    )
  }

  // MARK: - Trigger Word Extraction

  /// Read trigger words from a `{basename}.civitai.json` sidecar next to the
  /// LoRA file, if one exists (written by Desktop's CivitAI download flow).
  /// Returns nil (not empty) when no sidecar exists, so callers can fall
  /// back to auto-extraction; returns [] only if the sidecar exists but
  /// genuinely lists no trigger words.
  private static func civitaiSidecarTriggerWords(for url: URL) -> [String]? {
    let sidecar = url.deletingPathExtension().appendingPathExtension("civitai.json")
    guard let data = try? Data(contentsOf: sidecar),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    return (json["triggerWords"] as? [String]) ?? []
  }

  /// Extract candidate trigger words from kohya-style `ss_tag_frequency`
  /// training metadata: `{"1_Pinay": {"Pinay": 1, ...}, ...}` — a JSON object
  /// mapping dataset subfolder name to a tag→count histogram. Aggregates
  /// counts across all datasets and returns tags sorted by frequency,
  /// highest first, capped to avoid dumping an entire caption vocabulary.
  private static func extractTriggerWords(metadata: [String: String]?) -> [String] {
    guard let raw = metadata?["ss_tag_frequency"],
          let data = raw.data(using: .utf8),
          let datasets = try? JSONSerialization.jsonObject(with: data) as? [String: [String: Int]]
    else { return [] }

    var counts: [String: Int] = [:]
    for (_, tags) in datasets {
      for (tag, count) in tags {
        let trimmed = tag.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { continue }
        counts[trimmed, default: 0] += count
      }
    }

    return counts.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
      .prefix(10)
      .map { $0.key }
  }

  // MARK: - Compatibility Detection

  /// Detect model compatibility from safetensors metadata and key patterns.
  ///
  /// Priority order:
  /// 1. `ss_base_model_version` metadata (most reliable)
  /// 2. `modelspec.architecture` metadata
  /// 3. Layer name heuristics (fallback)
  public static func detectCompatibility(
    metadata: [String: String]?,
    sampleKeys: [String]
  ) -> [String] {
    // 1. Check ss_base_model_version
    if let baseVersion = metadata?["ss_base_model_version"] {
      let lower = baseVersion.lowercased()
      switch lower {
      // #313: sd-scripts' `networks.lora_zimage` (real vault file
      // Anneliese_Zbase3.safetensors) writes `ss_base_model_version: "z_image"`
      // — underscore, not one of the two forms this switch recognized.
      case "zimage", "z-image", "z_image", "lumina2":
        return ["z-image"]
      // #402 fix round 1 (Critical 2, coffeeshop-server#1681 review): a real
      // vault file (FK_adrianoanal.safetensors) writes
      // `ss_base_model_version: "flux2_klein_9b"` — this switch only knew
      // `flux_klein_9b` (no "2"), so it fell through to the modelspec
      // fallback below, which used to confidently mistag it `["flux1"]`.
      case "flux_klein_9b", "flux2_klein_9b":
        return ["klein-9b"]
      case "flux_klein_4b", "flux2_klein_4b":
        return ["klein-4b"]
      case "flux1":
        // Could be Chroma or generic Flux 1 — check keys for Chroma patterns
        if hasChromaKeyPatterns(sampleKeys) {
          return ["chroma"]
        }
        return ["flux1"]
      case "chroma":
        return ["chroma"]
      case "krea2", "krea-2", "krea-2-turbo":
        return ["krea2"]
      case "ltx2", "ltx-2", "ltxv", "ltx-video", "ltx_video":
        return ["ltx"]
      default:
        break
      }
    }

    // 2. Check modelspec.architecture
    if let arch = metadata?["modelspec.architecture"] {
      let lower = arch.lowercased()
      // #313: this branch had no z-image case at all (only klein/chroma/
      // ltx/flux) — `modelspec.architecture: "Z-Image/lora"` (the same real
      // file above) fell through to key heuristics unnecessarily.
      if lower.contains("z-image") || lower.contains("zimage") || lower.contains("z_image") {
        return ["z-image"]
      }
      if lower.contains("klein_9b") || lower.contains("klein-9b") {
        return ["klein-9b"]
      }
      if lower.contains("klein_4b") || lower.contains("klein-4b") {
        return ["klein-4b"]
      }
      if lower.contains("chroma") {
        return ["chroma"]
      }
      if lower.contains("ltx") {
        return ["ltx"]
      }
      // #402 fix round 1 (Critical 2): "flux-2/lora" (Flux 2 Klein, no
      // klein_9b/klein_4b marker in ITS modelspec — the real bug: the
      // ss_base_model_version switch above should have caught it first, and
      // now does) used to fall into a BARE `contains("flux")` check here and
      // come out a confident `["flux1"]` — the wrong architecture, and the
      // #402 guard makes a confident wrong tag fatal at swap/generate/preset
      // time. `"flux"` alone (flux2, flux-2, flux.2, or an unqualified
      // mention) is no longer enough for a CONFIDENT "flux1" — only an
      // explicit Flux.1 marker is. Anything else falls through to the
      // key-pattern heuristics below (step 3), which end in `["unknown"]`
      // rather than a guess when they too come up empty.
      if lower.contains("flux1") || lower.contains("flux.1") || lower.contains("flux-1") || lower.contains("flux_1") {
        return ["flux1"]
      }
    }

    // 3. Layer name heuristics
    return detectCompatibilityFromKeys(sampleKeys)
  }

  /// Detect compatibility purely from tensor key patterns.
  private static func detectCompatibilityFromKeys(_ keys: [String]) -> [String] {
    // Strip LoRA suffixes to get base keys
    let baseKeys = keys.compactMap { stripLoRASuffix($0) }

    var hasLayers = false
    var hasContextRefiner = false
    var hasNoiseRefiner = false
    var hasDoubleBlocks = false
    var hasSingleBlocks = false
    var hasImgAttn = false
    var hasTxtAttn = false
    var hasLTX2AudioVideoBranch = false
    var hasKrea2Signature = false
    // #313: LTX2TransformerBlock (Sources/ZImage/LTX2/Transformer/LTX2TransformerBlock.swift)
    // is the only architecture in this codebase with BOTH `attn1` (self) and
    // `attn2` (cross) under one numbered block — Z-Image's block has a
    // single `attention` and never an `attn2`, even under the generic
    // diffusers-style `transformer_blocks` naming LoRAKeyMapper also accepts
    // for Z-Image. Real vault files confirming the layout (all were
    // `unknown` before this fix): phool-realism-ltx-2.3-v1.0,
    // ltx-2.3-22b-ic-lora-union-control-ref0.5,
    // LTX2.3_reasoning_Sulphur-2_I2V_V4 — plain `diffusion_model.
    // transformer_blocks.N.attn1/attn2/ff...`, no audio branch, no metadata.
    var hasTransformerBlocksAttn1 = false
    var hasTransformerBlocksAttn2 = false

    for key in baseKeys {
      // Krea-2 SingleStreamDiT (WP-E6, FDD §3.6): `blocks.N.attn.w{q,k,v,o}`
      // is the DiT's own attention naming (no other family uses `attn.w*`),
      // and `txtfusion.` is its text-fusion stack — the bypass LoRA carries
      // ONLY `txtfusion.projector.diff`, so either signature alone suffices.
      if key.contains("txtfusion.") || isKrea2AttentionKey(key) {
        hasKrea2Signature = true
      }
      // LTX-2's transformer is the only architecture with audio/video
      // cross-attention branches — a reliable, distinctive signal since its
      // 48 numbered `layers.N.` otherwise look like Z-Image's.
      if key.contains("audio_") || key.contains("av_ca_")
          || key.contains("video_to_audio_attn") || key.contains("audio_to_video_attn") {
        hasLTX2AudioVideoBranch = true
      }
      if key.contains("transformer_blocks.") && key.contains(".attn1.") {
        hasTransformerBlocksAttn1 = true
      }
      if key.contains("transformer_blocks.") && key.contains(".attn2.") {
        hasTransformerBlocksAttn2 = true
      }
      // #313: comfy-export/kohya (`lora_unet_...`) keys are underscore-joined
      // throughout, not dot-separated — `double_blocks_`/`single_blocks_`
      // already had this fallback below; `layers_`/`context_refiner_`/
      // `noise_refiner_` did not (real vault file confirming the gap, was
      // `unknown` before this fix: Anneliese_Zbase3.safetensors —
      // `lora_unet_layers_0_attention_out...`).
      if key.contains("diffusion_model.layers.") || key.contains("layers.") || key.contains("layers_") {
        hasLayers = true
      }
      if key.contains("context_refiner.") || key.contains("context_refiner_") {
        hasContextRefiner = true
      }
      if key.contains("noise_refiner.") || key.contains("noise_refiner_") {
        hasNoiseRefiner = true
      }
      if key.contains("double_blocks.") || key.contains("double_blocks_") {
        hasDoubleBlocks = true
      }
      if key.contains("single_blocks.") || key.contains("single_blocks_") {
        hasSingleBlocks = true
      }
      if key.contains("img_attn") {
        hasImgAttn = true
      }
      if key.contains("txt_attn") {
        hasTxtAttn = true
      }
    }

    // LTX-2 video: distinctive audio/video cross-attention branches. Checked
    // first — LTX-2's plain numbered `layers.N.` would otherwise fall
    // through to the Z-Image heuristics below.
    if hasLTX2AudioVideoBranch {
      return ["ltx"]
    }

    // #313: LTX-2 LoRAs that never touch the audio/video branch (T2V/I2V-only
    // adapters, IC-LoRA reference/control adapters, sliders) still carry this
    // block's OWN two-attention signature — see the field comment above.
    //
    // ORDERING IS LOAD-BEARING (review round 1, minor finding): this check
    // must stay ABOVE the "Z-Image: layers + context_refiner/noise_refiner"
    // check below. In practice `hasTransformerBlocksAttn1`/`Attn2` and
    // `hasLayers` never both go true for the files this codebase has seen —
    // `transformer_blocks.` never contains the `layers.`/`layers_` substring
    // that check looks for — but that is a property of today's fixtures, not
    // a Swift-enforced invariant. If a future signal broadens `hasLayers` (or
    // a hybrid/mislabeled file ever satisfies both), whichever branch runs
    // FIRST wins the classification: keep the LTX-2 signature first so such a
    // file reads as `ltx`, not silently as `z-image`.
    if hasTransformerBlocksAttn1 && hasTransformerBlocksAttn2 {
      return ["ltx"]
    }

    if hasKrea2Signature {
      return ["krea2"]
    }

    // Z-Image: layers + context_refiner + noise_refiner. Runs AFTER both
    // LTX-2 checks above — see the ordering note there.
    if hasLayers && (hasContextRefiner || hasNoiseRefiner) {
      return ["z-image"]
    }

    // Flux-family (Klein/Chroma): double_blocks + single_blocks with img/txt attention
    if hasDoubleBlocks && hasSingleBlocks && (hasImgAttn || hasTxtAttn) {
      // Distinguish Klein vs Chroma by block counts
      let doubleBlockCount = countBlocks(keys: baseKeys, prefix: "double_blocks")
      let singleBlockCount = countBlocks(keys: baseKeys, prefix: "single_blocks")

      // Klein 9B: ~8 double + ~24 single; Klein 4B: fewer
      // Chroma: different block structure
      if doubleBlockCount >= 6 && singleBlockCount >= 20 {
        return ["klein-9b"]
      }
      if doubleBlockCount > 0 {
        return ["klein-9b"]  // Default to Klein for ambiguous flux-family
      }
    }

    // Z-Image without refiners (some LoRAs only target main layers)
    if hasLayers && !hasDoubleBlocks && !hasSingleBlocks {
      return ["z-image"]
    }

    return ["unknown"]
  }

  /// `…blocks.<n>.attn.w{q,k,v,o}…` — Krea-2's attention projections.
  private static func isKrea2AttentionKey(_ key: String) -> Bool {
    guard let range = key.range(of: "blocks.") else { return false }
    var rest = key[range.upperBound...]
    let digits = rest.prefix { $0.isNumber }
    guard !digits.isEmpty else { return false }
    rest = rest.dropFirst(digits.count)
    guard rest.hasPrefix(".attn.w") else { return false }
    let proj = rest.dropFirst(".attn.w".count)
    guard let first = proj.first, "qkvo".contains(first) else { return false }
    let after = proj.dropFirst()
    return after.isEmpty || after.hasPrefix(".")
  }

  /// Check for Chroma-specific key patterns (approximator, specific block structure).
  private static func hasChromaKeyPatterns(_ keys: [String]) -> Bool {
    for key in keys {
      if key.contains("distilled_guidance_layer") || key.contains("approximator") {
        return true
      }
    }
    return false
  }

  /// Count unique block indices for a given prefix (e.g. "double_blocks").
  private static func countBlocks(keys: [String], prefix: String) -> Int {
    var indices = Set<Int>()
    let dotPrefix = prefix + "."
    let underscorePrefix = prefix + "_"

    for key in keys {
      var afterPrefix: Substring?
      if key.contains(dotPrefix), let range = key.range(of: dotPrefix) {
        afterPrefix = key[range.upperBound...]
      } else if key.contains(underscorePrefix), let range = key.range(of: underscorePrefix) {
        afterPrefix = key[range.upperBound...]
      }

      if let rest = afterPrefix {
        let indexStr: Substring
        if let dotIdx = rest.firstIndex(of: ".") {
          indexStr = rest[..<dotIdx]
        } else if let underIdx = rest.firstIndex(of: "_") {
          indexStr = rest[..<underIdx]
        } else {
          continue
        }
        if let idx = Int(indexStr) {
          indices.insert(idx)
        }
      }
    }
    return indices.count
  }

  // MARK: - Format Detection

  /// Detect LoRA format from key patterns.
  ///
  /// - `.lokr` if any key contains `.lokr_w1`
  /// - `.slider` if key count <= 50 (very small adapters)
  /// - `.lora` otherwise (standard lora_down/lora_up or lora_A/lora_B)
  public static func detectFormat(keys: [String]) -> LoRAFormat {
    for key in keys {
      if key.contains(".lokr_w1") || key.contains(".lokr_w2") {
        return .lokr
      }
    }

    // Very small key count → likely a slider
    if keys.count <= 50 {
      return .slider
    }

    return .lora
  }

  // MARK: - Rank Inference

  /// Infer LoRA rank from metadata or tensor shapes.
  ///
  /// Priority:
  /// 1. `ss_network_dim` from metadata
  /// 2. Shape of first lora_down/lora_A tensor (second dimension)
  /// 3. Default to 0 (unknown)
  public static func inferRank(
    keys: [String],
    metadata: [String: String]?,
    reader: SafeTensorsReader
  ) -> Int {
    // 1. From metadata
    if let dimStr = metadata?["ss_network_dim"], let dim = Int(dimStr) {
      return dim
    }

    // 2. From tensor shapes — find a lora_down or lora_A tensor
    for key in keys {
      if key.contains("lora_down.weight") || key.contains("lora_A.weight") {
        if let meta = reader.metadata(for: key) {
          // For lora_down, shape is [rank, input_dim] — rank is dim 0
          if meta.shape.count >= 2 {
            return meta.shape[0]
          }
        }
      }
    }

    // 3. For LoKr, check lokr_w1 shapes
    for key in keys {
      if key.hasSuffix(".lokr_w1") {
        if let meta = reader.metadata(for: key) {
          if meta.shape.count >= 2 {
            return min(meta.shape[0], meta.shape[1])
          }
        }
      }
    }

    return 0
  }

  // MARK: - Alpha Extraction

  /// Extract alpha value from safetensors metadata.
  private static func extractAlpha(metadata: [String: String]?) -> Float? {
    guard let alphaStr = metadata?["ss_network_alpha"] else { return nil }
    return Float(alphaStr)
  }

  // MARK: - Layer Target Detection

  /// Detect which layer types are targeted by the LoRA.
  public static func detectLayerTargets(keys: [String]) -> [String] {
    var targets = Set<String>()

    for key in keys {
      let lower = key.lowercased()
      if lower.contains("attn") || lower.contains("attention") ||
         lower.contains("to_q") || lower.contains("to_k") || lower.contains("to_v") ||
         lower.contains("qkv") || lower.contains("proj") {
        targets.insert("attention")
      }
      if lower.contains("feed_forward") || lower.contains("ff.") || lower.contains("ff_") ||
         lower.contains("mlp") || lower.contains("linear") {
        targets.insert("feed_forward")
      }
      if lower.contains("adaln") || lower.contains("modulation") {
        targets.insert("adaLN_modulation")
      }
      if lower.contains("norm") {
        targets.insert("norm")
      }
    }

    return targets.sorted()
  }

  // MARK: - Helpers

  /// Strip LoRA-specific suffixes from a key to get the base model path.
  private static func stripLoRASuffix(_ key: String) -> String? {
    let suffixes = [
      ".lora_down.weight", ".lora_up.weight",
      ".lora_A.weight", ".lora_B.weight",
      ".lokr_w1", ".lokr_w2",
      ".alpha",
    ]
    for suffix in suffixes {
      if key.hasSuffix(suffix) {
        return String(key.dropLast(suffix.count))
      }
    }
    return key
  }

  /// Generate a slug ID from a filename.
  public static func slugify(_ filename: String) -> String {
    var name = (filename as NSString).deletingPathExtension
    // Replace non-alphanumeric with hyphens
    name = name.replacingOccurrences(of: "_", with: "-")
    name = name.replacingOccurrences(of: " ", with: "-")
    // Collapse multiple hyphens
    while name.contains("--") {
      name = name.replacingOccurrences(of: "--", with: "-")
    }
    // Trim leading/trailing hyphens
    name = name.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
    return name.lowercased()
  }
}
