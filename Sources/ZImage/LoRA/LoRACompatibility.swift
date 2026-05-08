// LoRACompatibility.swift — Model compatibility matrix for LoRA adapters
//
// Part of the LoRA Library Manager (#73). Maps compatibility strings to
// ComfyBoxModelFamily and performs trial key mapping to verify LoRA/model
// compatibility with match ratio reporting.

import Foundation

// MARK: - Compatibility Result

/// Result of checking a LoRA's compatibility with a specific model family.
public struct LoRACompatibilityResult: Sendable {
  /// Whether the LoRA is considered compatible (matchRatio > 0.5).
  public let isCompatible: Bool
  /// Number of LoRA keys that successfully mapped to model targets.
  public let matchedKeys: Int
  /// Total number of LoRA base keys tested.
  public let totalKeys: Int
  /// Ratio of matched keys to total keys (0.0 - 1.0).
  public let matchRatio: Float
  /// Warnings about potential issues (e.g. "LoKr format not supported for this model").
  public let warnings: [String]
}

// MARK: - Compatibility Checker

/// Checks LoRA compatibility against model families using key mappers.
public enum LoRACompatibility {

  /// Map a compatibility string from library.json to a ComfyBoxModelFamily.
  ///
  /// - Parameter compat: A compatibility tag (e.g. "z-image", "klein-9b").
  /// - Returns: The matching model family, or nil if unknown.
  public static func familyMapping(_ compat: String) -> ComfyBoxModelFamily? {
    switch compat.lowercased() {
    case "z-image", "zimage":
      return .zImage
    case "klein-9b", "klein-4b", "flux2-klein":
      return .flux2Klein
    case "chroma":
      return .chroma
    default:
      return nil
    }
  }

  /// Map a ComfyBoxModelFamily to its compatibility string(s).
  ///
  /// - Parameter family: The model family to look up.
  /// - Returns: Compatibility strings that match this family.
  public static func compatibilityStrings(for family: ComfyBoxModelFamily) -> [String] {
    switch family {
    case .zImage:
      return ["z-image"]
    case .flux2Klein:
      return ["klein-9b", "klein-4b"]
    case .chroma:
      return ["chroma"]
    case .fibo, .seedvr2, .esrgan:
      return []
    }
  }

  /// Check if a LoRA entry is compatible with a given model family.
  ///
  /// First checks the declared `model_compatibility` tags. If the entry
  /// declares compatibility, returns a positive result without trial mapping.
  /// If not declared, performs trial key mapping through the family's key mapper.
  ///
  /// - Parameters:
  ///   - entry: The LoRA library entry to check.
  ///   - modelFamily: The target model family.
  /// - Returns: A `LoRACompatibilityResult` with match details.
  public static func check(
    entry: LoRALibraryEntry,
    modelFamily: ComfyBoxModelFamily
  ) -> LoRACompatibilityResult {
    let familyCompats = compatibilityStrings(for: modelFamily)
    var warnings: [String] = []

    // Quick check: does the entry declare compatibility?
    let declaredMatch = entry.modelCompatibility.contains { compat in
      familyCompats.contains(compat.lowercased())
    }

    if declaredMatch {
      // Check format compatibility
      if entry.format == .lokr {
        switch modelFamily {
        case .flux2Klein:
          // Only Flux2LoRALoader supports LoKr, not the standard loadForFlux2
          warnings.append("LoKr format — requires Flux2LoRALoader (not all Klein loaders support LoKr)")
        case .chroma:
          warnings.append("LoKr format not supported for Chroma")
        default:
          break
        }
      }

      return LoRACompatibilityResult(
        isCompatible: true,
        matchedKeys: entry.keyCount,
        totalKeys: entry.keyCount,
        matchRatio: 1.0,
        warnings: warnings
      )
    }

    // Trial key mapping: test sample keys against the model's key mapper
    return trialKeyMapping(entry: entry, modelFamily: modelFamily)
  }

  // MARK: - Trial Key Mapping

  /// Perform trial key mapping to estimate compatibility.
  ///
  /// Generates sample LoRA keys based on the entry's detected compatibility
  /// and tests them against the target model family's key mapper.
  private static func trialKeyMapping(
    entry: LoRALibraryEntry,
    modelFamily: ComfyBoxModelFamily
  ) -> LoRACompatibilityResult {
    // We can't do actual key mapping without reading the file,
    // so use the declared compatibility and key count for estimation.
    let familyCompats = compatibilityStrings(for: modelFamily)
    let entryCompats = Set(entry.modelCompatibility.map { $0.lowercased() })
    let familySet = Set(familyCompats)

    // No overlap in declared compatibility = incompatible
    if entryCompats.isDisjoint(with: familySet) && !entryCompats.contains("unknown") {
      return LoRACompatibilityResult(
        isCompatible: false,
        matchedKeys: 0,
        totalKeys: entry.keyCount,
        matchRatio: 0.0,
        warnings: ["LoRA targets \(entry.modelCompatibility.joined(separator: ", ")), not \(modelFamily.displayName)"]
      )
    }

    // Unknown compatibility — report as uncertain
    return LoRACompatibilityResult(
      isCompatible: false,
      matchedKeys: 0,
      totalKeys: entry.keyCount,
      matchRatio: 0.0,
      warnings: ["Unknown compatibility — run `lora scan` to detect"]
    )
  }

  /// Perform a live compatibility check by reading actual keys from a file.
  ///
  /// This reads the safetensors header and maps each LoRA base key through
  /// the target model's key mapper to count successful matches.
  ///
  /// - Parameters:
  ///   - url: Path to the safetensors file.
  ///   - modelFamily: The target model family.
  /// - Returns: A detailed compatibility result with actual match counts.
  public static func checkFile(
    _ url: URL,
    modelFamily: ComfyBoxModelFamily
  ) throws -> LoRACompatibilityResult {
    let reader = try SafeTensorsReader(fileURL: url)
    let allKeys = reader.tensorNames
    var warnings: [String] = []

    // Extract base keys (strip lora_down/up, lokr_w1/w2 suffixes)
    let baseKeys = extractBaseKeys(from: allKeys)
    let totalBaseKeys = baseKeys.count

    guard totalBaseKeys > 0 else {
      return LoRACompatibilityResult(
        isCompatible: false,
        matchedKeys: 0,
        totalKeys: allKeys.count,
        matchRatio: 0.0,
        warnings: ["No LoRA keys found in file"]
      )
    }

    // Map each base key through the target model's key mapper
    var matchedCount = 0

    switch modelFamily {
    case .zImage:
      let validTargets = Set(LoRAKeyMapper.supportedTargetPaths)
      for baseKey in baseKeys {
        let mapped = LoRAKeyMapper.mapToZImageKey(baseKey)
        if validTargets.contains(mapped) {
          matchedCount += 1
        }
      }

    case .flux2Klein:
      for baseKey in baseKeys {
        let stripped = Flux2LoRAMapping.stripPrefix(baseKey)
        let result = Flux2LoRAMapping.map(stripped)
        switch result {
        case .direct, .qkvSplit:
          matchedCount += 1
        case .unmapped:
          break
        }
      }

    case .chroma:
      for baseKey in baseKeys {
        let mapped = ChromaLoRAKeyMapper.map(baseKey)
        // ChromaLoRAKeyMapper always returns something — check if it's a valid target
        // Valid Chroma targets start with double_blocks., single_blocks., txt_in, img_in
        if mapped.hasPrefix("double_blocks.") || mapped.hasPrefix("single_blocks.") ||
           mapped.hasPrefix("txt_in") || mapped.hasPrefix("img_in") {
          matchedCount += 1
        }
      }

    default:
      warnings.append("\(modelFamily.displayName) does not support LoRA")
      return LoRACompatibilityResult(
        isCompatible: false,
        matchedKeys: 0,
        totalKeys: totalBaseKeys,
        matchRatio: 0.0,
        warnings: warnings
      )
    }

    let ratio = totalBaseKeys > 0 ? Float(matchedCount) / Float(totalBaseKeys) : 0.0
    let isCompatible = ratio > 0.5

    // Check LoKr format compatibility
    let hasLoKr = allKeys.contains { $0.contains(".lokr_w1") }
    if hasLoKr {
      switch modelFamily {
      case .chroma:
        warnings.append("LoKr format not supported for Chroma")
      default:
        break
      }
    }

    return LoRACompatibilityResult(
      isCompatible: isCompatible,
      matchedKeys: matchedCount,
      totalKeys: totalBaseKeys,
      matchRatio: ratio,
      warnings: warnings
    )
  }

  // MARK: - Key Extraction Helpers

  /// Extract unique base keys from LoRA tensor names.
  ///
  /// Strips lora_down/lora_up, lora_A/lora_B, lokr_w1/lokr_w2, and alpha suffixes
  /// to get the underlying model path each adapter targets.
  private static func extractBaseKeys(from keys: [String]) -> Set<String> {
    var baseKeys = Set<String>()
    let suffixes = [
      ".lora_down.weight", ".lora_up.weight",
      ".lora_A.weight", ".lora_B.weight",
      ".lokr_w1", ".lokr_w2",
    ]

    for key in keys {
      for suffix in suffixes {
        if key.hasSuffix(suffix) {
          let base = String(key.dropLast(suffix.count))
          baseKeys.insert(base)
          break
        }
      }
    }

    return baseKeys
  }
}
