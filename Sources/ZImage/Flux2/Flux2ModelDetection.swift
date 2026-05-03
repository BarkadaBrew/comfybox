// Flux2ModelDetection.swift — Detect Flux 2 Klein models from snapshot directories

import Foundation

/// Model family detection for routing to the correct pipeline.
public enum ZImageModelFamily: String, Sendable {
  case flux1 = "flux1"
  case flux2 = "flux2"
  case fibo = "fibo"
}

/// Detected Flux 2 Klein model variant with its configuration.
public struct Flux2DetectedModel {
  /// The model variant (e.g., "klein-4b", "klein-base-4b").
  public let variant: String
  /// Whether this is a base (non-distilled) model that supports guidance > 1.0.
  public let isBaseModel: Bool
  /// Transformer config parsed from the model directory.
  public let transformerConfig: Flux2TransformerConfig
  /// Text encoder config inferred from the model directory.
  public let textEncoderConfig: Qwen3TextEncoderConfiguration
}

/// Detects Flux 2 Klein models from snapshot directories.
public enum Flux2ModelDetection {

  /// Detect whether a model snapshot directory contains a Flux 2 model.
  ///
  /// Detection signals:
  /// 1. `transformer/config.json` contains `"_class_name": "Flux2Transformer2DModel"`
  /// 2. `text_encoder/config.json` contains `"Qwen3ForCausalLM"` in architectures
  ///
  /// - Parameter snapshot: Root URL of the model snapshot directory.
  /// - Returns: Detected model info, or nil if not a Flux 2 model.
  public static func detect(at snapshot: URL) -> Flux2DetectedModel? {
    let fm = FileManager.default

    // Check transformer config for Flux2Transformer2DModel
    let transformerConfigURL = snapshot
      .appendingPathComponent("transformer")
      .appendingPathComponent("config.json")

    guard fm.fileExists(atPath: transformerConfigURL.path),
          let configData = try? Data(contentsOf: transformerConfigURL),
          let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
          let className = configDict["_class_name"] as? String,
          className == "Flux2Transformer2DModel" else {
      return nil
    }

    // Parse transformer config from JSON
    let numLayers = configDict["num_layers"] as? Int ?? 5
    let numSingleLayers = configDict["num_single_layers"] as? Int ?? 20
    let numAttentionHeads = configDict["num_attention_heads"] as? Int ?? 24
    let attentionHeadDim = configDict["attention_head_dim"] as? Int ?? 128
    let jointAttentionDim = configDict["joint_attention_dim"] as? Int ?? 7680
    let inChannels = configDict["in_channels"] as? Int ?? 128
    let patchSize = configDict["patch_size"] as? Int ?? 1
    let ropeTheta = configDict["rope_theta"] as? Int ?? 2000
    let guidanceEmbeds = configDict["guidance_embeds"] as? Bool ?? false
    let mlpRatio = (configDict["mlp_ratio"] as? NSNumber)?.floatValue ?? 3.0

    let axesDimsRope: [Int]
    if let axes = configDict["axes_dims_rope"] as? [Int] {
      axesDimsRope = axes
    } else {
      axesDimsRope = [32, 32, 32, 32]
    }

    let transformerConfig = Flux2TransformerConfig(
      patchSize: patchSize,
      inChannels: inChannels,
      numLayers: numLayers,
      numSingleLayers: numSingleLayers,
      attentionHeadDim: attentionHeadDim,
      numAttentionHeads: numAttentionHeads,
      jointAttentionDim: jointAttentionDim,
      mlpRatio: mlpRatio,
      axesDimsRope: axesDimsRope,
      ropeTheta: ropeTheta,
      guidanceEmbeds: guidanceEmbeds
    )

    // Parse text encoder config if available
    let textEncoderConfig = parseTextEncoderConfig(at: snapshot) ?? Qwen3TextEncoderConfiguration()

    // Determine variant from model dimensions and whether it's a base model.
    // Base (non-distilled) models are detected by "base" in the directory path,
    // matching mflux behavior (is_distilled = "base" not in model_name.lower()).
    // The config.json guidance_embeds field is false for both distilled and base.
    let isBase = snapshot.path.lowercased().contains("base")
    let variant: String
    if numLayers == 8 && numSingleLayers == 24 && numAttentionHeads == 32 {
      variant = isBase ? "klein-base-9b" : "klein-9b"
    } else {
      variant = isBase ? "klein-base-4b" : "klein-4b"
    }

    return Flux2DetectedModel(
      variant: variant,
      isBaseModel: isBase,
      transformerConfig: transformerConfig,
      textEncoderConfig: textEncoderConfig
    )
  }

  /// Detect model family from a snapshot directory.
  ///
  /// - Parameter snapshot: Root URL of the model snapshot directory.
  /// - Returns: `.flux2` if a Flux2 model is detected, `.flux1` otherwise.
  public static func detectFamily(at snapshot: URL) -> ZImageModelFamily {
    if FiboModelDetection.detect(at: snapshot) != nil {
      return .fibo
    }
    if detect(at: snapshot) != nil {
      return .flux2
    }
    return .flux1
  }

  // MARK: - Text Encoder Config Parsing

  private static func parseTextEncoderConfig(at snapshot: URL) -> Qwen3TextEncoderConfiguration? {
    let configURL = snapshot
      .appendingPathComponent("text_encoder")
      .appendingPathComponent("config.json")

    guard let data = try? Data(contentsOf: configURL),
          let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      return nil
    }

    // Verify it's a Qwen3 model
    if let architectures = dict["architectures"] as? [String],
       !architectures.contains("Qwen3ForCausalLM") {
      return nil
    }

    let hiddenSize = dict["hidden_size"] as? Int ?? 2560
    let numHiddenLayers = dict["num_hidden_layers"] as? Int ?? 36
    let numAttentionHeads = dict["num_attention_heads"] as? Int ?? 32
    let numKeyValueHeads = dict["num_key_value_heads"] as? Int ?? 8
    let intermediateSize = dict["intermediate_size"] as? Int ?? 9728
    let maxPositionEmbeddings = dict["max_position_embeddings"] as? Int ?? 40960
    let ropeTheta = (dict["rope_theta"] as? NSNumber)?.floatValue ?? 1_000_000.0
    let vocabSize = dict["vocab_size"] as? Int ?? 151936
    let rmsNormEps = (dict["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6
    let headDim = dict["head_dim"] as? Int ?? 128

    return Qwen3TextEncoderConfiguration(
      vocabSize: vocabSize,
      hiddenSize: hiddenSize,
      numHiddenLayers: numHiddenLayers,
      numAttentionHeads: numAttentionHeads,
      numKeyValueHeads: numKeyValueHeads,
      intermediateSize: intermediateSize,
      maxPositionEmbeddings: maxPositionEmbeddings,
      ropeTheta: ropeTheta,
      rmsNormEps: rmsNormEps,
      headDim: headDim
    )
  }

  /// Known Flux 2 Klein HuggingFace model IDs.
  public static let knownModelIds: [String] = [
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-9B",
    "black-forest-labs/FLUX.2-klein-base-4B",
    "black-forest-labs/FLUX.2-klein-base-9B",
  ]

  /// Check if a model spec string refers to a known Flux 2 model.
  public static func isKnownFlux2Model(_ modelSpec: String) -> Bool {
    let normalized = modelSpec.lowercased()
    return knownModelIds.contains { $0.lowercased() == normalized }
      || normalized.contains("flux.2-klein")
      || normalized.contains("flux2-klein")
  }
}
