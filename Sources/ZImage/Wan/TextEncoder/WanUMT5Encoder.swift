import Foundation
import MLX
import MLXNN

/// Wan 2.2 UMT5-XXL text encoder — full encoder stack.
///
/// Encodes tokenized text into contextual embeddings for conditioning the
/// Wan 2.2 video diffusion model.
///
/// ## Architecture
///
/// ```
/// Input: [B, seqLen] (token IDs)
///   → Embedding(256384, 4096)
///   → 24 × WanT5EncoderLayer (pre-norm attention + gated GELU FFN)
///   → RMSNorm(4096)
///   → Output: [B, seqLen, 4096]
/// ```
///
/// ## Weight File
///
/// Loads from safetensors (converted from the Wan .pth checkpoint).
/// Key mapping:
///
/// | Safetensors Key | Module |
/// |----------------|--------|
/// | `token_embedding.weight` | embedding |
/// | `blocks.{i}.*` | layers[i] |
/// | `norm.weight` | finalNorm |
public final class WanUMT5Encoder: Module {

  /// Token embedding: [vocabSize, hiddenSize].
  @ModuleInfo(key: "token_embedding") var embedding: Embedding

  /// Encoder layers.
  @ModuleInfo(key: "blocks") var layers: [WanT5EncoderLayer]

  /// Final RMS normalization.
  @ModuleInfo(key: "norm") var finalNorm: RMSNorm

  /// Model configuration.
  public let config: WanUMT5Config

  /// Creates the UMT5-XXL encoder.
  ///
  /// - Parameter config: Model configuration. Default: `.wan22I2V`.
  public init(config: WanUMT5Config = .wan22I2V) {
    self.config = config
    self._embedding.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
    self._layers.wrappedValue = (0..<config.numLayers).map { _ in
      WanT5EncoderLayer(config: config)
    }
    self._finalNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    super.init()
  }

  /// Encodes token IDs into contextual embeddings.
  ///
  /// - Parameters:
  ///   - tokenIds: Token ID tensor of shape `[B, seqLen]`.
  ///   - attentionMask: Attention mask of shape `[B, seqLen]`. 1 = attend, 0 = mask.
  /// - Returns: Encoded representations of shape `[B, seqLen, hiddenSize]`.
  public func callAsFunction(tokenIds: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
    // Embed tokens: [B, seqLen] -> [B, seqLen, hiddenSize]
    var hidden = embedding(tokenIds)

    // Build causal-compatible mask from attention mask if provided
    // Convert from [B, seqLen] with 1/0 to [B, 1, 1, seqLen] with 0/-inf
    let mask: MLXArray?
    if let attentionMask = attentionMask {
      // (1 - mask) * -1e9 gives 0 for attend positions, large negative for masked
      let expandedMask = attentionMask.reshaped(attentionMask.dim(0), 1, 1, attentionMask.dim(1))
      mask = (1.0 - expandedMask.asType(.float32)) * MLXArray(Float(-1e9))
    } else {
      mask = nil
    }

    // Apply encoder layers
    for layer in layers {
      hidden = layer(hidden, mask: mask)
    }

    // Final normalization
    hidden = finalNorm(hidden)

    return hidden
  }

  /// Loads weights from a safetensors file.
  ///
  /// The safetensors file should contain keys matching the Wan naming convention
  /// (e.g., `blocks.0.attn.q.weight`, `token_embedding.weight`, `norm.weight`).
  ///
  /// - Parameter url: Path to the safetensors file.
  /// - Throws: If the file cannot be read or weights cannot be applied.
  public func loadWeights(from url: URL) throws {
    let weights = try MLX.loadArrays(url: url)
    try loadWeights(weights)
  }

  /// Loads weights from a pre-loaded dictionary.
  ///
  /// - Parameter weights: Dictionary mapping weight keys to MLXArrays.
  /// - Throws: If weights cannot be applied.
  public func loadWeights(_ weights: [String: MLXArray]) throws {
    // Build nested parameter dictionary for Module.update
    let params = ModuleParameters.unflattened(weights.map { ($0.key, $0.value) })
    try self.update(parameters: params, verify: [.shapeMismatch])
    eval(self.parameters())
  }
}
