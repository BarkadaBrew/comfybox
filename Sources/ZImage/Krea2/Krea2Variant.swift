// Krea2Variant.swift — The physical Krea-2 checkpoint variant (WP-E5, FDD §3.5, D7).
//
// `Krea2Variant` is a PHYSICAL FACT read from the checkpoint file the engine
// actually loaded (`raw.safetensors` vs `turbo.safetensors`); it is never
// requested, only reported. Step budgets, guidance ranges and sampler pairings
// are the client's `CheckpointFamily` policy layer — `raw-accel` and
// `raw-stock` are two policies over the one physical variant `raw`.

import Foundation

public enum Krea2Variant: String, Sendable, Codable, CaseIterable, Equatable {
  case turbo
  case raw

  /// The transformer filename a model root carries for this variant.
  public var transformerFilename: String {
    switch self {
    case .turbo: return "turbo.safetensors"
    case .raw: return "raw.safetensors"
    }
  }

  /// Turbo is guidance-distilled (CFG is a no-op by design); Raw honours CFG.
  public var supportsGuidance: Bool {
    switch self {
    case .turbo: return false
    case .raw: return true
    }
  }

  /// Engine default when a request omits `steps`. 30 on Raw is a neutral,
  /// CFG-free default for a direct Raw call; 6 / 10 / 52 are client policies.
  public var defaultSteps: Int {
    switch self {
    case .turbo: return 9
    case .raw: return 30
    }
  }

  /// Engine default when a request omits `guidance`. 1.0 == off on BOTH
  /// variants. 3.5 (Krea's stock Raw recipe) is deliberately NOT the engine
  /// default: it fires for every Raw request that omits guidance, doubling
  /// model evals and activating an empty negative prompt — a surprise. It
  /// belongs to the client's `raw-stock` family policy, which sends it
  /// explicitly and records it (FDD §3.5, corrected in v2).
  public var defaultGuidance: Float {
    switch self {
    case .turbo: return 1.0
    case .raw: return 1.0
    }
  }

  /// The Krita bridge's runaway-KSampler clamp. Turbo keeps today's 12;
  /// Raw has no clamp (the bridge arm itself is WP-E19).
  public var bridgeStepClamp: Int? {
    switch self {
    case .turbo: return 12
    case .raw: return nil
    }
  }

  /// `payload.steps ?? variant.defaultSteps` — the generate-path resolution
  /// (`runKrea2Generate`), kept a pure function so AC-5b is testable without
  /// weights.
  public func resolvedSteps(_ requested: Int?) -> Int {
    requested ?? defaultSteps
  }

  /// `payload.guidance ?? variant.defaultGuidance`.
  public func resolvedGuidance(_ requested: Float?) -> Float {
    requested ?? defaultGuidance
  }
}

/// Fail-closed model-directory / spec resolution errors (FDD §3.5, AC-34/34a).
/// Every case names what was looked at; none of them is ever recovered from by
/// substituting a different checkpoint.
public enum Krea2ModelPathsError: Error, Equatable, CustomStringConvertible {
  /// Both `raw.safetensors` and `turbo.safetensors` are present. Never guess.
  case ambiguousVariant(URL)
  /// The path/spec does not resolve to a Krea-2 model root.
  case notAKrea2ModelDirectory(String, reason: Reason)

  public enum Reason: Equatable, Sendable {
    /// The path is not an existing directory.
    case notADirectory
    /// Neither `raw.safetensors` nor `turbo.safetensors`, and no
    /// `model_index.json` escape hatch.
    case noTransformer
    /// The DiT is there but `text_encoder/model.safetensors` is not.
    case missingTextEncoder
    /// The DiT is there but `vae/diffusion_pytorch_model.safetensors` is not.
    case missingVAE
    /// `model_index.json` exists but does not declare a loadable
    /// `krea2_variant` + `transformer_file` pair (detail in the payload).
    case invalidModelIndex(String)
    /// A spec that is neither an existing path, a declared alias in the
    /// spec→directory table, nor one of the four Krea-2-Turbo HF aliases.
    case unmappedSpec
  }

  public var description: String {
    switch self {
    case .ambiguousVariant(let dir):
      return "Krea2: \(dir.path) holds BOTH raw.safetensors and turbo.safetensors — refusing to guess the variant"
    case .notAKrea2ModelDirectory(let what, let reason):
      let why: String
      switch reason {
      case .notADirectory: why = "not an existing directory"
      case .noTransformer: why = "no raw.safetensors / turbo.safetensors and no model_index.json naming one"
      case .missingTextEncoder: why = "missing text_encoder/model.safetensors"
      case .missingVAE: why = "missing vae/diffusion_pytorch_model.safetensors"
      case .invalidModelIndex(let detail): why = "model_index.json is not usable: \(detail)"
      case .unmappedSpec:
        why = "not an existing path, not a declared alias (config.json krea2Models), and not one of the "
          + "Krea-2-Turbo aliases \(Krea2ModelDetection.turboAliases.sorted())"
      }
      return "Krea2: '\(what)' is not a Krea-2 model directory — \(why)"
    }
  }
}
