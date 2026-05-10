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
///   -> Embedding(256384, 4096)
///   -> 24 x WanT5EncoderLayer (pre-norm attention + gated GELU FFN)
///   -> RMSNorm(4096)
///   -> Output: [B, seqLen, 4096]
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
///
/// Note: `blocks.{i}.ffn.gate.0.weight` is remapped to `blocks.{i}.ffn.gate.weight`
/// at load time because MLX-Swift unflatten treats numeric path segments as
/// array indices, which is incompatible with Module.update.
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
    var hidden = embedding(tokenIds)

    let mask: MLXArray?
    if let attentionMask = attentionMask {
      let expandedMask = attentionMask.reshaped(attentionMask.dim(0), 1, 1, attentionMask.dim(1))
      mask = (1.0 - expandedMask.asType(.float32)) * MLXArray(Float(-1e9))
    } else {
      mask = nil
    }

    for layer in layers {
      hidden = layer(hidden, mask: mask)
    }

    hidden = finalNorm(hidden)

    return hidden
  }

  /// Loads weights from a safetensors file.
  ///
  /// - Parameter url: Path to the safetensors file.
  /// - Throws: If the file cannot be read or weights cannot be applied.
  public func loadWeights(from url: URL) throws {
    let weights = try MLX.loadArrays(url: url)
    try loadWeights(weights)
  }

  /// Loads weights from a pre-loaded dictionary.
  ///
  /// Remaps Wan checkpoint keys before unflattening:
  /// - `ffn.gate.0.weight` -> `ffn.gate.weight` (numeric segment breaks MLX-Swift unflatten)
  /// - `ffn.gate.0.bias` -> `ffn.gate.bias` (if present)
  ///
  /// - Parameter weights: Dictionary mapping weight keys to MLXArrays.
  /// - Throws: If weights cannot be applied.
  public func loadWeights(_ weights: [String: MLXArray]) throws {
    let remapped = Self.remapGateKeys(weights)
    let params = ModuleParameters.unflattened(remapped.map { ($0.key, $0.value) })
    try self.update(parameters: params, verify: [.shapeMismatch])
    eval(self.parameters())
  }

  /// Remaps `ffn.gate.0.{suffix}` -> `ffn.gate.{suffix}` in weight keys.
  ///
  /// The Wan checkpoint stores the gate projection under PyTorch nn.ModuleList
  /// naming convention (`gate.0.weight`). MLX-Swift unflattenedRecurse() treats
  /// the numeric `0` segment as an array index, creating an `.array` structure that
  /// `Module.update()` cannot merge (it only handles `.module + .dictionary`).
  ///
  /// By stripping the `.0.` segment, the key becomes `gate.weight` which unflattens
  /// into a `.dictionary` structure compatible with the plain `Linear` property.
  ///
  /// - Parameter weights: Original weight dictionary from safetensors.
  /// - Returns: Dictionary with gate keys remapped.
  static func remapGateKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var result = [String: MLXArray]()
    result.reserveCapacity(weights.count)
    for (key, value) in weights {
      if key.contains(".ffn.gate.0.") {
        let remapped = key.replacingOccurrences(of: ".ffn.gate.0.", with: ".ffn.gate.")
        result[remapped] = value
      } else {
        result[key] = value
      }
    }
    return result
  }
}
