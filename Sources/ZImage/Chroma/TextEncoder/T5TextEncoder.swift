import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Relative Position Bias

/// T5 relative position bias — binned, bidirectional.
///
/// Computes position-dependent attention bias using bucketed relative positions.
/// Weight key: `encoder.relative_attention_bias.embeddings.weight`
final class T5RelativePositionBias: Module {
  let bidirectional: Bool
  let numBuckets: Int
  let maxDistance: Int
  let nHeads: Int

  @ModuleInfo(key: "embeddings") var embeddings: Embedding

  init(config: T5Config, bidirectional: Bool = true) {
    self.bidirectional = bidirectional
    self.numBuckets = config.relativeAttentionNumBuckets
    self.maxDistance = config.relativeAttentionMaxDistance
    self.nHeads = config.numHeads
    self._embeddings.wrappedValue = Embedding(embeddingCount: numBuckets, dimensions: nHeads)
    super.init()
  }

  func callAsFunction(queryLength: Int, keyLength: Int) -> MLXArray {
    let contextPos = MLXArray(stride(from: 0, to: queryLength, by: 1).map { Int32($0) })
      .expandedDimensions(axis: 1)
    let memoryPos = MLXArray(stride(from: 0, to: keyLength, by: 1).map { Int32($0) })
      .expandedDimensions(axis: 0)
    let relativePos = memoryPos - contextPos

    let buckets = relativeBucket(relativePos)
    // [queryLen, keyLen, nHeads] -> [nHeads, queryLen, keyLen]
    return embeddings(buckets).transposed(2, 0, 1)
  }

  private func relativeBucket(_ rpos: MLXArray) -> MLXArray {
    let effectiveBuckets = bidirectional ? numBuckets / 2 : numBuckets
    let maxExact = effectiveBuckets / 2

    let absPos = MLX.abs(rpos)
    let isSmall = absPos .< MLXArray(Int32(maxExact))

    let scale = Float(effectiveBuckets - maxExact) / log(Float(maxDistance) / Float(maxExact))
    let bucketsLarge = MLX.minimum(
      MLXArray(Int32(maxExact)) + (MLX.log(absPos.asType(.float32) / Float(maxExact)) * scale).asType(.int32),
      MLXArray(Int32(effectiveBuckets - 1))
    )

    var buckets = MLX.where(isSmall, absPos, bucketsLarge)
    if bidirectional {
      buckets = buckets + (rpos .> MLXArray(Int32(0))).asType(.int32) * MLXArray(Int32(numBuckets))
    }
    return buckets
  }
}

// MARK: - T5 Multi-Head Attention

/// T5 multi-head attention with relative position bias.
final class T5MultiHeadAttention: Module {
  let numHeads: Int
  let dKV: Int

  @ModuleInfo(key: "query_proj") var queryProj: Linear
  @ModuleInfo(key: "key_proj") var keyProj: Linear
  @ModuleInfo(key: "value_proj") var valueProj: Linear
  @ModuleInfo(key: "out_proj") var outProj: Linear

  init(config: T5Config) {
    self.numHeads = config.numHeads
    self.dKV = config.dKV
    let innerDim = config.dKV * config.numHeads
    self._queryProj.wrappedValue = Linear(config.dModel, innerDim, bias: false)
    self._keyProj.wrappedValue = Linear(config.dModel, innerDim, bias: false)
    self._valueProj.wrappedValue = Linear(config.dModel, innerDim, bias: false)
    self._outProj.wrappedValue = Linear(innerDim, config.dModel, bias: false)
    super.init()
  }

  func callAsFunction(queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray) -> MLXArray {
    let b = queries.dim(0)
    let l = queries.dim(1)
    let s = keys.dim(1)

    var q = queryProj(queries).reshaped(b, l, numHeads, dKV).transposed(0, 2, 1, 3)
    var k = keyProj(keys).reshaped(b, s, numHeads, dKV).transposed(0, 2, 1, 3)
    var v = valueProj(values).reshaped(b, s, numHeads, dKV).transposed(0, 2, 1, 3)

    let attn = MLXFast.scaledDotProductAttention(
      queries: q, keys: k, values: v,
      scale: 1.0, mask: mask.asType(q.dtype)
    )
    let out = attn.transposed(0, 2, 1, 3).reshaped(b, l, -1)
    return outProj(out)
  }
}

// MARK: - T5 Dense Activation (Gated GELU FFN)

/// T5 feed-forward network with gated activation.
///
/// For T5 v1.1: gated-gelu with two input projections and one output projection.
final class T5DenseActivation: Module {
  let gated: Bool
  @ModuleInfo(key: "wi_0") var wi0: Linear
  @ModuleInfo(key: "wi_1") var wi1: Linear
  @ModuleInfo(key: "wo") var wo: Linear

  init(config: T5Config) {
    let mlpDims = config.dFF
    self.gated = config.feedForwardProj.hasPrefix("gated")
    self._wi0.wrappedValue = Linear(config.dModel, mlpDims, bias: false)
    self._wi1.wrappedValue = Linear(config.dModel, mlpDims, bias: false)
    self._wo.wrappedValue = Linear(mlpDims, config.dModel, bias: false)
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    if gated {
      let hiddenAct = gelu(wi0(x))
      let hiddenLinear = wi1(x)
      return wo(hiddenAct * hiddenLinear)
    } else {
      return wo(gelu(wi0(x)))
    }
  }
}

// MARK: - T5 Encoder Layer

/// Single T5 encoder layer: self-attention + feed-forward with pre-norm.
final class T5EncoderLayer: Module {
  @ModuleInfo(key: "attention") var attention: T5MultiHeadAttention
  @ModuleInfo(key: "ln1") var ln1: RMSNorm
  @ModuleInfo(key: "ln2") var ln2: RMSNorm
  @ModuleInfo(key: "dense") var dense: T5DenseActivation

  init(config: T5Config) {
    self._attention.wrappedValue = T5MultiHeadAttention(config: config)
    self._ln1.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
    self._ln2.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
    self._dense.wrappedValue = T5DenseActivation(config: config)
    super.init()
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
    var h = x
    let y1 = ln1(h)
    let attnOut = attention(queries: y1, keys: y1, values: y1, mask: mask)
    h = h + attnOut
    let y2 = ln2(h)
    let ffOut = dense(y2)
    return h + ffOut
  }
}

// MARK: - T5 Encoder

/// Full T5 encoder stack: N layers + final layer norm + relative position bias.
final class T5TransformerEncoder: Module {
  @ModuleInfo(key: "layers") var layers: [T5EncoderLayer]
  @ModuleInfo(key: "ln") var ln: RMSNorm
  @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: T5RelativePositionBias

  init(config: T5Config) {
    var encoderLayers: [T5EncoderLayer] = []
    for _ in 0..<config.numLayers {
      encoderLayers.append(T5EncoderLayer(config: config))
    }
    self._layers.wrappedValue = encoderLayers
    self._ln.wrappedValue = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
    self._relativeAttentionBias.wrappedValue = T5RelativePositionBias(config: config)
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let seqLen = x.dim(1)
    let posBias = relativeAttentionBias(queryLength: seqLen, keyLength: seqLen).asType(x.dtype)
    var h = x
    for layer in layers {
      h = layer(h, mask: posBias)
    }
    return ln(h)
  }
}

// MARK: - T5 Encoder Model

/// T5-XXL text encoder for Chroma.
///
/// Embeds token IDs and runs through the encoder stack.
/// Output: `[B, seqLen, 4096]` text embeddings.
///
/// Weight key prefix: `wte` (embedding), `encoder` (transformer)
public final class T5Encoder: Module {
  @ModuleInfo(key: "wte") var wte: Embedding
  @ModuleInfo(key: "encoder") var encoder: T5TransformerEncoder

  public init(config: T5Config = .xxl) {
    self._wte.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.dModel)
    self._encoder.wrappedValue = T5TransformerEncoder(config: config)
    super.init()
  }

  public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
    encoder(wte(inputIds))
  }

  /// Sanitize weight keys from HuggingFace T5 format to our module paths.
  public static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    let shared: [(String, String)] = [
      (".block.", ".layers."),
      (".k.", ".key_proj."),
      (".o.", ".out_proj."),
      (".q.", ".query_proj."),
      (".v.", ".value_proj."),
      ("shared.", "wte."),
      (".layer.0.layer_norm.", ".ln1."),
      (".layer.1.layer_norm.", ".ln2."),
      (".layer.2.layer_norm.", ".ln3."),
      (".final_layer_norm.", ".ln."),
      ("layers.0.layer.0.SelfAttention.relative_attention_bias.",
       "relative_attention_bias.embeddings."),
    ]
    let encoder: [(String, String)] = [
      (".layer.0.SelfAttention.", ".attention."),
      (".layer.1.DenseReluDense.", ".dense."),
    ]

    var sanitized: [String: MLXArray] = [:]
    for (key, value) in weights {
      var k = key
      for (old, new) in shared {
        k = k.replacingOccurrences(of: old, with: new)
      }
      if k.hasPrefix("encoder.") {
        for (old, new) in encoder {
          k = k.replacingOccurrences(of: old, with: new)
        }
      }
      sanitized[k] = value
    }
    return sanitized
  }
}
