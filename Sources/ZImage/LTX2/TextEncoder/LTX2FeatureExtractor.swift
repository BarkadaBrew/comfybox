// LTX2FeatureExtractor.swift — Gemma 3 hidden state feature extraction
// Phase 2 of the LTX-2 Swift/MLX port
//
// Extracts and combines hidden states from all 49 Gemma 3 layers into
// features suitable for the 1D connector. Two variants:
//
// V1 (LTX-2 original): 8 * (x - mean) / range normalization, single linear projection
// V2 (LTX-2.3): Per-token RMSNorm, rescale normalization, separate video/audio projections
//
// Reference: text_encoder.py classes GemmaFeaturesExtractor, GemmaFeaturesExtractorV2,
//            and functions norm_and_concat_hidden_states, norm_and_concat_per_token_rms

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - V1 Feature Extractor

/// V1 feature extractor (LTX-2): 8 * (x - mean) / range normalization + linear projection.
///
/// Process:
/// 1. Stack all hidden states: (B, T, D, L) where L = number of layers
/// 2. Compute masked mean and range per layer
/// 3. Normalize: 8 * (x - mean) / range
/// 4. Flatten: (B, T, D*L)
/// 5. Linear projection: (B, T, D*L) -> (B, T, outputDim)
///
/// Weight key mapping:
/// - `aggregate_embed.weight`
public final class LTX2FeatureExtractorV1: Module {
  @ModuleInfo(key: "aggregate_embed") var aggregateEmbed: Linear

  public init(inputDim: Int, outputDim: Int) {
    self._aggregateEmbed.wrappedValue = Linear(inputDim, outputDim, bias: false)
  }

  /// Extract features from stacked hidden states.
  ///
  /// - Parameters:
  ///   - hiddenStates: List of hidden state arrays, each `[B, T, D]`.
  ///   - attentionMask: Binary attention mask `[B, T]` (1 = valid, 0 = pad).
  /// - Returns: Projected features `[B, T, outputDim]`.
  public func callAsFunction(
    hiddenStates: [MLXArray],
    attentionMask: MLXArray
  ) -> MLXArray {
    let normed = LTX2FeatureExtractor.normAndConcatHiddenStates(
      hiddenStates: hiddenStates,
      attentionMask: attentionMask,
      paddingSide: "left"
    )
    return aggregateEmbed(normed)
  }
}

// MARK: - V2 Feature Extractor

/// V2 feature extractor (LTX-2.3): per-token RMSNorm + rescale normalization
/// with separate video and audio projection heads.
///
/// Process:
/// 1. Stack hidden states: (B, T, D, L)
/// 2. Per-token RMSNorm across hidden dimension
/// 3. Flatten: (B, T, D*L)
/// 4. Rescale: x * sqrt(target_dim / embedding_dim)
/// 5. Project through video or audio linear head
///
/// Weight key mapping:
/// - `video_aggregate_embed.weight`, `video_aggregate_embed.bias`
/// - `audio_aggregate_embed.weight`, `audio_aggregate_embed.bias`
public final class LTX2FeatureExtractorV2: Module {
  let embeddingDim: Int

  @ModuleInfo(key: "video_aggregate_embed") var videoAggregateEmbed: Linear
  @ModuleInfo(key: "audio_aggregate_embed") var audioAggregateEmbed: Linear

  public init(config: LTX2FeatureExtractorConfig) {
    self.embeddingDim = config.embeddingDim
    self._videoAggregateEmbed.wrappedValue = Linear(
      config.inputDim, config.videoOutputDim, bias: config.bias)
    self._audioAggregateEmbed.wrappedValue = Linear(
      config.inputDim, config.audioOutputDim, bias: config.bias)
  }

  /// Extract video features from hidden states.
  ///
  /// - Parameters:
  ///   - hiddenStates: List of hidden state arrays, each `[B, T, D]`.
  ///   - attentionMask: Binary attention mask `[B, T]` (1 = valid, 0 = pad).
  /// - Returns: Projected video features `[B, T, videoOutputDim]`.
  public func videoFeatures(
    hiddenStates: [MLXArray],
    attentionMask: MLXArray
  ) -> MLXArray {
    let normed = LTX2FeatureExtractor.normAndConcatPerTokenRMS(
      hiddenStates: hiddenStates,
      attentionMask: attentionMask
    )
    let targetDim = videoAggregateEmbed.weight.dim(0)
    let rescaled = LTX2FeatureExtractor.rescaleNorm(
      normed, targetDim: targetDim, sourceDim: embeddingDim)
    return videoAggregateEmbed(rescaled)
  }

  /// Extract audio features from hidden states.
  ///
  /// - Parameters:
  ///   - hiddenStates: List of hidden state arrays, each `[B, T, D]`.
  ///   - attentionMask: Binary attention mask `[B, T]` (1 = valid, 0 = pad).
  /// - Returns: Projected audio features `[B, T, audioOutputDim]`.
  public func audioFeatures(
    hiddenStates: [MLXArray],
    attentionMask: MLXArray
  ) -> MLXArray {
    let normed = LTX2FeatureExtractor.normAndConcatPerTokenRMS(
      hiddenStates: hiddenStates,
      attentionMask: attentionMask
    )
    let targetDim = audioAggregateEmbed.weight.dim(0)
    let rescaled = LTX2FeatureExtractor.rescaleNorm(
      normed, targetDim: targetDim, sourceDim: embeddingDim)
    return audioAggregateEmbed(rescaled)
  }
}

// MARK: - Static Normalization Functions

/// Static helper functions for feature extraction normalization.
public enum LTX2FeatureExtractor {

  /// V1 normalization: 8 * (x - mean) / range, with masking for padded positions.
  ///
  /// Matches Python `norm_and_concat_hidden_states` function.
  ///
  /// - Parameters:
  ///   - hiddenStates: List of hidden state arrays from Gemma layers.
  ///   - attentionMask: Binary attention mask `[B, T]`.
  ///   - paddingSide: "left" or "right" padding convention.
  /// - Returns: Normalized and concatenated tensor `[B, T, D*L]`.
  public static func normAndConcatHiddenStates(
    hiddenStates: [MLXArray],
    attentionMask: MLXArray,
    paddingSide: String = "left"
  ) -> MLXArray {
    // Stack: (B, T, D, L)
    let stacked = MLX.stacked(hiddenStates, axis: -1)
    let dtype = stacked.dtype
    let b = stacked.dim(0)
    let t = stacked.dim(1)
    let d = stacked.dim(2)
    let numLayers = stacked.dim(3)

    // Compute sequence lengths from attention mask
    let sequenceLengths = MLX.sum(attentionMask, axis: -1)  // (B,)

    // Build mask based on padding side
    let tokenIndices = MLXArray(0..<t).expandedDimensions(axis: 0)  // (1, T)

    let mask: MLXArray
    if paddingSide == "right" {
      mask = tokenIndices .< sequenceLengths.expandedDimensions(axis: 1)
    } else {
      let startIndices = MLXArray(t) - sequenceLengths.expandedDimensions(axis: 1)
      mask = tokenIndices .>= startIndices
    }

    // (B, T) -> (B, T, 1, 1) for broadcasting
    let mask4D = mask
      .expandedDimensions(axis: 2)
      .expandedDimensions(axis: 3)
    let eps = MLXArray(Float(1e-6)).asType(dtype)

    // Compute masked mean per layer
    let masked = MLX.where(mask4D, stacked, MLX.zeros(like: stacked))
    let denom = (sequenceLengths * MLXArray(d))
      .reshaped(b, 1, 1, 1)
      .asType(dtype)
    let mean = MLX.sum(masked, axes: [1, 2], keepDims: true) / (denom + eps)

    // Compute masked min/max per layer
    let posInf = MLX.full(stacked.shape, values: MLXArray(Float.infinity), dtype: dtype)
    let negInf = MLX.full(stacked.shape, values: MLXArray(-Float.infinity), dtype: dtype)
    let xForMin = MLX.where(mask4D, stacked, posInf)
    let xForMax = MLX.where(mask4D, stacked, negInf)
    let xMin = MLX.min(xForMin, axes: [1, 2], keepDims: true)
    let xMax = MLX.max(xForMax, axes: [1, 2], keepDims: true)
    let rangeVal = xMax - xMin

    // Normalize: 8 * (x - mean) / range
    let normed = 8 * (stacked - mean) / (rangeVal + eps)

    // Flatten layers into feature dimension: (B, T, D*L)
    var result = normed.reshaped(b, t, d * numLayers)

    // Zero out padded positions
    let maskFlat = MLX.broadcast(
      mask.expandedDimensions(axis: 2),
      to: [b, t, d * numLayers]
    )
    result = MLX.where(maskFlat, result, MLX.zeros(like: result))

    return result
  }

  /// V2 normalization: Per-token RMSNorm across hidden dimension.
  ///
  /// Matches Python `norm_and_concat_per_token_rms` function.
  ///
  /// - Parameters:
  ///   - hiddenStates: List of hidden state arrays from Gemma layers.
  ///   - attentionMask: Binary attention mask `[B, T]`.
  /// - Returns: Normalized and concatenated tensor `[B, T, D*L]`.
  public static func normAndConcatPerTokenRMS(
    hiddenStates: [MLXArray],
    attentionMask: MLXArray
  ) -> MLXArray {
    // Stack: (B, T, D, L)
    let encoded = MLX.stacked(hiddenStates, axis: -1)
    let dtype = encoded.dtype
    let b = encoded.dim(0)
    let t = encoded.dim(1)
    let d = encoded.dim(2)
    let numLayers = encoded.dim(3)

    // Per-token RMSNorm across hidden dimension: variance = mean(x^2) over dim D
    let encodedF32 = encoded.asType(.float32)
    let variance = MLX.mean(encodedF32 * encodedF32, axis: 2, keepDims: true)  // (B, T, 1, L)
    let normed = (encodedF32 * MLX.rsqrt(variance + MLXArray(Float(1e-6)))).asType(dtype)

    // Flatten layers: (B, T, D*L)
    var result = normed.reshaped(b, t, d * numLayers)

    // Zero out padded positions
    let mask3D = attentionMask.expandedDimensions(axis: 2)  // (B, T, 1)
    let maskBool = mask3D .> MLXArray(Float(0))
    result = MLX.where(
      MLX.broadcast(maskBool, to: result.shape),
      result,
      MLX.zeros(like: result)
    )

    return result
  }

  /// Rescale normalization: x * sqrt(target_dim / source_dim).
  ///
  /// Used in V2 feature extraction to compensate for dimension changes.
  public static func rescaleNorm(
    _ x: MLXArray, targetDim: Int, sourceDim: Int
  ) -> MLXArray {
    let scale = Float(targetDim) / Float(sourceDim)
    return x * MLXArray(scale.squareRoot())
  }
}
