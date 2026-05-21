// LTX2TextProjection.swift — PixArtAlpha-style text projection for AdaLN conditioning
// Phase 2 of the LTX-2 Swift/MLX port
//
// Simple 2-layer MLP that projects caption embeddings into the transformer's
// inner dimension for adaptive layer norm conditioning. Used in the main
// diffusion transformer's timestep conditioning path.
//
// Architecture:
//   Linear(caption_channels, inner_dim) -> GELU(tanh) -> Linear(inner_dim, inner_dim)
//
// Reference: text_projection.py class PixArtAlphaTextProjection

import MLX
import MLXNN

/// PixArtAlpha-style text projection for AdaLN conditioning.
///
/// Projects caption embeddings (from the text encoder pipeline) into the
/// diffusion transformer's conditioning space. The output is used alongside
/// timestep embeddings for adaptive layer normalization.
///
/// Weight key mapping (safetensors -> model):
/// - `caption_projection.linear1.weight`
/// - `caption_projection.linear1.bias`
/// - `caption_projection.linear2.weight`
/// - `caption_projection.linear2.bias`
public final class LTX2TextProjection: Module {
  @ModuleInfo(key: "linear1") var linear1: Linear
  @ModuleInfo(key: "linear2") var linear2: Linear

  /// Initialize the text projection MLP.
  ///
  /// - Parameters:
  ///   - inFeatures: Input dimension (caption_channels, e.g. 3840).
  ///   - hiddenSize: Hidden and output dimension (inner_dim, e.g. 4096).
  ///   - outFeatures: Output dimension. Defaults to hiddenSize if nil.
  ///   - bias: Whether to use bias in linear layers. Default true.
  public init(
    inFeatures: Int,
    hiddenSize: Int,
    outFeatures: Int? = nil,
    bias: Bool = true
  ) {
    let outputDim = outFeatures ?? hiddenSize
    self._linear1.wrappedValue = Linear(inFeatures, hiddenSize, bias: bias)
    self._linear2.wrappedValue = Linear(hiddenSize, outputDim, bias: bias)
  }

  /// Convenience initializer from config.
  public convenience init(config: LTX2TextProjectionConfig) {
    self.init(
      inFeatures: config.captionChannels,
      hiddenSize: config.innerDim
    )
  }

  /// Forward pass: Linear -> GELU(tanh) -> Linear
  ///
  /// - Parameter x: Caption embeddings `[B, caption_channels]` or `[B, S, caption_channels]`.
  /// - Returns: Projected embeddings `[B, inner_dim]` or `[B, S, inner_dim]`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = linear1(x)
    h = geluApproximate(h)
    h = linear2(h)
    return h
  }
}
