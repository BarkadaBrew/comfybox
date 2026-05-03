// Flux2WeightDefinition.swift — Defines which safetensors files contain which components
// Ported from mflux: flux2_weight_definition.py

import Foundation

/// Defines the safetensors weight file layout for Flux 2 Klein models.
///
/// Flux 2 Klein models store weights in three component subdirectories:
///   - `vae/` — VAE encoder/decoder weights
///   - `transformer/` — Transformer denoising backbone weights
///   - `text_encoder/` — Qwen3 text encoder weights
///
/// Each subdirectory contains one or more `.safetensors` shards plus a
/// `config.json` describing the component's architecture.
public enum Flux2WeightDefinition {

  /// Known Flux 2 Klein model identifiers on HuggingFace.
  public enum ModelID: String {
    case klein4B = "black-forest-labs/FLUX.2-klein-4B"
    case klein9B = "black-forest-labs/FLUX.2-klein-9B"
  }

  /// Component subdirectory names within the model snapshot.
  public enum Component: String, CaseIterable {
    case vae = "vae"
    case transformer = "transformer"
    case textEncoder = "text_encoder"
  }

  /// File patterns to download for a complete Flux 2 Klein model.
  public static let downloadPatterns: [String] = [
    "vae/*.safetensors",
    "vae/*.json",
    "transformer/*.safetensors",
    "transformer/*.json",
    "text_encoder/*.safetensors",
    "text_encoder/*.json",
    "tokenizer/**",
    "added_tokens.json",
    "chat_template.jinja",
  ]

  /// Resolve weight shard files for a given component within a snapshot directory.
  ///
  /// Uses the same shard resolution logic as the existing `ZImageFiles` for consistency:
  /// checks index JSON, then scans directory for `.safetensors` files.
  ///
  /// - Parameters:
  ///   - component: Which model component to resolve.
  ///   - snapshot: Root URL of the model snapshot.
  /// - Returns: Array of URLs pointing to safetensors shard files.
  public static func resolveWeightFiles(
    for component: Component,
    at snapshot: URL
  ) -> [URL] {
    let componentDir = snapshot.appendingPathComponent(component.rawValue)
    return ZImageFiles.resolveWeightFiles(in: componentDir, componentName: component.rawValue)
  }

  /// Check whether a snapshot directory contains all required Flux 2 components.
  ///
  /// - Parameter snapshot: Root URL of the model snapshot.
  /// - Returns: `true` if all three component directories contain safetensors files.
  public static func hasAllComponents(at snapshot: URL) -> Bool {
    Component.allCases.allSatisfy { component in
      !resolveWeightFiles(for: component, at: snapshot).isEmpty
    }
  }
}
