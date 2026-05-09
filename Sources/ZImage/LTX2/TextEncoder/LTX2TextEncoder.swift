// LTX2TextEncoder.swift — Top-level text encoder for the LTX-2 video generation model
// Phase 2 of the LTX-2 Swift/MLX port
//
// Orchestrates the full text encoding pipeline:
//   Tokenize -> Gemma 3 forward (all hidden states) -> Feature Extract -> 1D Connector -> Embeddings
//
// Output:
//   - videoEmbeddings: (B, seqLen, 4096) — for cross-attention in the video transformer
//   - audioEmbeddings: (B, seqLen, 2048) — for cross-attention in the audio transformer
//   - captionProjection: (B, innerDim) — for AdaLN conditioning (via text projection)
//
// Supports both LTX-2 (original) and LTX-2.3 (prompt adaln) variants.
//
// Reference: text_encoder.py class LTX2TextEncoder

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Audio Embeddings Connector (V1 only)

/// Simple linear projection from shared feature space to audio cross-attention dimension.
///
/// Only used in LTX-2 original (V1) where video and audio share the same feature
/// extractor output and a simple projection maps to the audio dimension.
///
/// Weight key mapping:
/// - `audio_embeddings_connector_projection.linear.weight`
/// - `audio_embeddings_connector_projection.linear.bias`
public final class LTX2AudioEmbeddingsProjection: Module {
  @ModuleInfo(key: "linear") var linear: Linear

  public init(inputDim: Int = 3840, outputDim: Int = 2048) {
    self._linear.wrappedValue = Linear(inputDim, outputDim, bias: true)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    linear(x)
  }
}

// MARK: - Text Encoder Output

/// Output from the LTX-2 text encoder.
public struct LTX2TextEncoderOutput {
  /// Video embeddings for cross-attention `[B, S, videoDim]`.
  public let videoEmbeddings: MLXArray

  /// Audio embeddings for cross-attention `[B, S, audioDim]`.
  public let audioEmbeddings: MLXArray

  /// Attention mask `[B, S]` (1 = valid, 0 = pad).
  public let attentionMask: MLXArray
}

// MARK: - LTX-2 Text Encoder

/// Top-level LTX-2 text encoder.
///
/// Orchestrates the complete text encoding pipeline:
/// 1. Tokenize input prompt
/// 2. Run Gemma 3 forward pass extracting all 49 hidden states
/// 3. Feature extraction (V1 or V2 normalization + projection)
/// 4. 1D connector (transformer blocks + register tokens + RoPE)
/// 5. Output dual video/audio embeddings
///
/// The text encoder is a heavyweight module (~24 GB at fp16, ~6 GB at Q4).
/// It is loaded separately from the main diffusion transformer.
public final class LTX2TextEncoder: Module {
  public let config: LTX2TextEncoderConfig

  /// Gemma 3 12B language model backbone
  let languageModel: LTX2GemmaModel

  /// V1 feature extractor (LTX-2 original)
  @ModuleInfo(key: "feature_extractor") var featureExtractorV1: LTX2FeatureExtractorV1?

  /// V2 feature extractor (LTX-2.3)
  @ModuleInfo(key: "feature_extractor_v2") var featureExtractorV2: LTX2FeatureExtractorV2?

  /// Video embeddings connector
  @ModuleInfo(key: "video_embeddings_connector") var videoConnector: LTX2Connector1D

  /// Audio embeddings connector
  @ModuleInfo(key: "audio_embeddings_connector") var audioConnector: LTX2Connector1D

  public init(config: LTX2TextEncoderConfig = LTX2TextEncoderConfig()) {
    self.config = config

    self.languageModel = LTX2GemmaModel(config: config.gemma)

    if config.hasPromptAdaLN {
      // LTX-2.3: V2 feature extractor with separate video/audio projections
      self._featureExtractorV2.wrappedValue = LTX2FeatureExtractorV2(
        config: config.featureExtractor)
      self._featureExtractorV1.wrappedValue = nil
    } else {
      // LTX-2: V1 feature extractor with shared projection
      self._featureExtractorV1.wrappedValue = LTX2FeatureExtractorV1(
        inputDim: config.featureExtractor.inputDim,
        outputDim: config.hiddenDim
      )
      self._featureExtractorV2.wrappedValue = nil
    }

    self._videoConnector.wrappedValue = LTX2Connector1D(config: config.videoConnector)
    self._audioConnector.wrappedValue = LTX2Connector1D(config: config.audioConnector)
  }

  /// Encode a tokenized input to video and audio embeddings.
  ///
  /// This is the main encoding function that takes pre-tokenized input
  /// (input IDs and attention mask) and runs the full pipeline.
  ///
  /// - Parameters:
  ///   - inputIds: Token IDs `[B, S]`.
  ///   - attentionMask: Padding mask `[B, S]` (1 = valid, 0 = pad).
  ///   - returnAudioEmbeddings: If true, computes audio embeddings too.
  /// - Returns: `LTX2TextEncoderOutput` with video and audio embeddings.
  public func encode(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    returnAudioEmbeddings: Bool = true
  ) -> LTX2TextEncoderOutput {
    // Step 1: Gemma 3 forward pass — extract ALL hidden states
    let allHiddenStates = languageModel(
      inputIds: inputIds,
      attentionMask: attentionMask,
      outputHiddenStates: true
    )

    let videoEmbeddings: MLXArray
    let audioEmbeddings: MLXArray

    if config.hasPromptAdaLN {
      // LTX-2.3: V2 feature extraction with separate video/audio paths
      guard let extractor = featureExtractorV2 else {
        fatalError("V2 feature extractor not initialized for prompt adaln mode")
      }

      // Video features
      let videoFeatures = extractor.videoFeatures(
        hiddenStates: allHiddenStates,
        attentionMask: attentionMask
      )

      // Build additive mask for connector
      let additiveMask = buildAdditiveMask(
        attentionMask: attentionMask, dtype: videoFeatures.dtype)

      let (videoOut, _) = videoConnector(
        hiddenStates: videoFeatures, attentionMask: additiveMask)
      videoEmbeddings = videoOut

      if returnAudioEmbeddings {
        let audioFeatures = extractor.audioFeatures(
          hiddenStates: allHiddenStates,
          attentionMask: attentionMask
        )
        let audioMask = buildAdditiveMask(
          attentionMask: attentionMask, dtype: audioFeatures.dtype)
        let (audioOut, _) = audioConnector(
          hiddenStates: audioFeatures, attentionMask: audioMask)
        audioEmbeddings = audioOut
      } else {
        audioEmbeddings = MLX.zeros([inputIds.dim(0), inputIds.dim(1), config.audioDim])
      }

    } else {
      // LTX-2 original: V1 feature extraction with shared features
      guard let extractor = featureExtractorV1 else {
        fatalError("V1 feature extractor not initialized for non-prompt-adaln mode")
      }

      let features = extractor(
        hiddenStates: allHiddenStates,
        attentionMask: attentionMask
      )

      let additiveMask = buildAdditiveMask(
        attentionMask: attentionMask, dtype: features.dtype)

      let (videoOut, _) = videoConnector(
        hiddenStates: features, attentionMask: additiveMask)
      videoEmbeddings = videoOut

      if returnAudioEmbeddings {
        let (audioOut, _) = audioConnector(
          hiddenStates: features, attentionMask: additiveMask)
        audioEmbeddings = audioOut
      } else {
        audioEmbeddings = MLX.zeros([inputIds.dim(0), inputIds.dim(1), config.audioDim])
      }
    }

    return LTX2TextEncoderOutput(
      videoEmbeddings: videoEmbeddings,
      audioEmbeddings: audioEmbeddings,
      attentionMask: attentionMask
    )
  }

  /// Build additive attention mask for the connector.
  ///
  /// Converts binary mask `[B, S]` to additive mask `[B, 1, 1, S]`:
  /// - 1 -> 0 (attend)
  /// - 0 -> -1e9 (mask)
  private func buildAdditiveMask(
    attentionMask: MLXArray, dtype: DType
  ) -> MLXArray {
    let additive = (attentionMask.asType(dtype) - 1) * 1e9
    return additive
      .reshaped(attentionMask.dim(0), 1, 1, attentionMask.dim(1))
  }

  // MARK: - Weight Loading

  /// Load weights for the complete text encoder.
  ///
  /// Handles two weight layout formats:
  /// 1. Reformatted (text_projections/ subdirectory): pre-sanitized keys
  /// 2. Monolithic (single safetensors at root): raw PyTorch key names
  ///
  /// Gemma 3 weights are loaded separately from a text_encoder subdirectory
  /// or from a pre-trained model path.
  ///
  /// - Parameters:
  ///   - modelPath: Path to the LTX-2 model directory.
  ///   - textEncoderPath: Path to the Gemma 3 weights directory.
  ///     If nil, looks for `text_encoder/` subdirectory under modelPath.
  public func loadWeights(
    modelPath: URL,
    textEncoderPath: URL? = nil
  ) throws {
    // Load Gemma 3 language model weights
    let gemmaPath: URL
    if let tePath = textEncoderPath {
      let subDir = tePath.appendingPathComponent("text_encoder")
      gemmaPath = FileManager.default.fileExists(atPath: subDir.path) ? subDir : tePath
    } else {
      gemmaPath = modelPath.appendingPathComponent("text_encoder")
    }

    try loadGemmaWeights(from: gemmaPath)

    // Load connector and feature extractor weights
    try loadProjectionWeights(from: modelPath)
  }

  /// Load Gemma 3 weights from safetensors files.
  private func loadGemmaWeights(from path: URL) throws {
    let fm = FileManager.default
    guard fm.fileExists(atPath: path.path) else {
      print("WARNING: Gemma 3 weights directory not found at \(path.path)")
      print("  Language model will use uninitialized weights.")
      return
    }

    let contents = try fm.contentsOfDirectory(at: path, includingPropertiesForKeys: nil)
    let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }.sorted {
      $0.lastPathComponent < $1.lastPathComponent
    }

    guard !safetensorFiles.isEmpty else {
      print("WARNING: No safetensors files found in \(path.path)")
      return
    }

    var allWeights: [String: MLXArray] = [:]
    for file in safetensorFiles {
      let weights = try MLX.loadArrays(url: file)
      for (key, value) in weights {
        allWeights[key] = value
      }
    }

    // Sanitize: strip language_model. prefix, convert float32 -> bfloat16
    let sanitized = LTX2GemmaModel.sanitizeWeights(allWeights)

    // Check for quantization config
    let configFile = path.appendingPathComponent("config.json")
    var quantization: [String: Any]? = nil
    if let configData = try? Data(contentsOf: configFile),
       let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] {
      quantization = configDict["quantization"] as? [String: Any]
    }

    // Apply quantization if configured
    if let quant = quantization,
       let bits = quant["bits"] as? Int,
       let groupSize = quant["group_size"] as? Int {
      // Quantize the language model layers
      let availableKeys = Set(sanitized.keys)
      MLXNN.quantize(model: languageModel) { path, _ in
        let scalesKey = "\(path).scales"
        guard availableKeys.contains(scalesKey) else { return nil }
        return (groupSize, bits, .affine)
      }
    }

    // Load weights (non-strict to handle quantized layers with missing keys)
    let weightList = sanitized.map { ($0.key, $0.value) }
    let params = ModuleParameters.unflattened(weightList)
    try languageModel.update(parameters: params, verify: [.shapeMismatch])
  }

  /// Load connector and feature extractor weights from model directory.
  private func loadProjectionWeights(from modelPath: URL) throws {
    let fm = FileManager.default
    var weights: [String: MLXArray] = [:]
    var isReformatted = false

    // Try reformatted layout: text_projections/ subdirectory
    let textProjDir = modelPath.appendingPathComponent("text_projections")
    if fm.fileExists(atPath: textProjDir.path) {
      isReformatted = true
      let contents = try fm.contentsOfDirectory(at: textProjDir, includingPropertiesForKeys: nil)
      for file in contents where file.pathExtension == "safetensors" {
        let w = try MLX.loadArrays(url: file)
        for (key, value) in w {
          weights[key] = value
        }
      }
    }

    // Fall back to monolithic layout
    if weights.isEmpty {
      let contents = try fm.contentsOfDirectory(at: modelPath, includingPropertiesForKeys: nil)
      let ltxFiles = contents.filter {
        $0.lastPathComponent.hasPrefix("ltx-2-19") && $0.pathExtension == "safetensors"
      }
      if let file = ltxFiles.first {
        weights = try MLX.loadArrays(url: file)
      }
    }

    guard !weights.isEmpty else {
      print("WARNING: No transformer weights found for text projection connectors.")
      print("  Text conditioning will use uninitialized weights!")
      return
    }

    // Load feature extractor weights
    loadFeatureExtractorWeights(weights, isReformatted: isReformatted)

    // Load connector weights
    loadConnectorWeights(name: "video_embeddings_connector",
                         connector: videoConnector,
                         weights: weights,
                         isReformatted: isReformatted)
    loadConnectorWeights(name: "audio_embeddings_connector",
                         connector: audioConnector,
                         weights: weights,
                         isReformatted: isReformatted)
  }

  /// Load feature extractor weights.
  private func loadFeatureExtractorWeights(
    _ weights: [String: MLXArray],
    isReformatted: Bool
  ) {
    if config.hasPromptAdaLN {
      // V2: separate video/audio aggregate embeds
      guard let extractor = featureExtractorV2 else { return }

      var feWeights: [(String, MLXArray)] = []
      for prefix in ["video_aggregate_embed", "audio_aggregate_embed"] {
        if let w = weights["\(prefix).weight"] {
          feWeights.append(("\(prefix).weight", w))
        }
        if let b = weights["\(prefix).bias"] {
          feWeights.append(("\(prefix).bias", b))
        }
      }
      if !feWeights.isEmpty {
        let params = ModuleParameters.unflattened(feWeights)
        try? extractor.update(parameters: params, verify: [.shapeMismatch])
      }
    } else {
      // V1: single aggregate_embed
      guard let extractor = featureExtractorV1 else { return }
      let key = isReformatted
        ? "aggregate_embed.weight"
        : "text_embedding_projection.aggregate_embed.weight"
      if let w = weights[key] {
        let params = ModuleParameters.unflattened([("aggregate_embed.weight", w)])
        try? extractor.update(parameters: params, verify: [.shapeMismatch])
      }
    }
  }

  /// Load connector weights with key sanitization.
  private func loadConnectorWeights(
    name: String,
    connector: LTX2Connector1D,
    weights: [String: MLXArray],
    isReformatted: Bool
  ) {
    // Extract connector-specific weights
    var connectorWeights: [String: MLXArray] = [:]
    let prefix: String
    if isReformatted {
      prefix = "\(name)."
    } else {
      prefix = "model.diffusion_model.\(name)."
    }

    for (key, value) in weights {
      guard key.hasPrefix(prefix) else { continue }
      connectorWeights[String(key.dropFirst(prefix.count))] = value
    }

    guard !connectorWeights.isEmpty else { return }

    // Sanitize key names (only for monolithic/raw PyTorch keys)
    var mapped: [String: MLXArray] = [:]
    for (key, value) in connectorWeights {
      var newKey = key
      if !isReformatted {
        newKey = newKey.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
        newKey = newKey.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
        newKey = newKey.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
      }
      mapped[newKey] = value
    }

    // Load weights into the connector (non-strict for optional gate_logits)
    let weightList = mapped.map { ($0.key, $0.value) }
    let params = ModuleParameters.unflattened(weightList)
    try? connector.update(parameters: params, verify: [])

    // Manually load learnable_registers (plain array, not a Module parameter)
    if let registers = connectorWeights["learnable_registers"] {
      connector.learnableRegisters = registers
    }
  }
}
