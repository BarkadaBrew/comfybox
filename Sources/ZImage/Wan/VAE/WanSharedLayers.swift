import Foundation
import MLX
import MLXNN

// MARK: - Sequential Layer Container

/// Flat sequential container with integer-indexed sub-modules.
/// Holds an array of Module (heterogeneous: WanResidualBlock and WanResample).
/// Weight loading uses integer indices: downsamples.0.*, downsamples.1.*, etc.
public final class WanSequentialLayers: Module {

  /// The layers in sequential order.
  public let layers: [Module]

  public init(layers: [Module]) {
    self.layers = layers
    super.init()
  }

  /// Standard forward (no cache).
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = x
    for layer in layers {
      if let resBlock = layer as? WanResidualBlock {
        h = resBlock(h)
      } else if let resample = layer as? WanResample {
        h = resample(h)
      }
    }
    return h
  }

  /// Cached forward for chunk-by-chunk encoding.
  public func forward(_ x: MLXArray, cache: WanEncoderCache) -> MLXArray {
    var h = x
    for layer in layers {
      if let resBlock = layer as? WanResidualBlock {
        h = resBlock.forward(h, cache: cache)
      } else if let resample = layer as? WanResample {
        h = resample.forward(h, cache: cache)
      }
    }
    return h
  }
}

// MARK: - Middle Block

/// Middle block: ResBlock(dim) + AttentionBlock(dim) + ResBlock(dim).
/// Weight keys: middle.0.*, middle.1.*, middle.2.*
public final class WanMiddleLayers: Module {

  public let layers: [Module]

  public init(dim: Int) {
    self.layers = [
      WanResidualBlock(inDim: dim, outDim: dim),
      WanAttentionBlock(dim: dim),
      WanResidualBlock(inDim: dim, outDim: dim),
    ]
    super.init()
  }

  /// Standard forward (no cache).
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = (layers[0] as! WanResidualBlock)(x)
    h = (layers[1] as! WanAttentionBlock)(h)
    h = (layers[2] as! WanResidualBlock)(h)
    return h
  }

  /// Cached forward for chunk-by-chunk encoding.
  public func forward(_ x: MLXArray, cache: WanEncoderCache) -> MLXArray {
    var h = (layers[0] as! WanResidualBlock).forward(x, cache: cache)
    h = (layers[1] as! WanAttentionBlock)(h)
    h = (layers[2] as! WanResidualBlock).forward(h, cache: cache)
    return h
  }
}

// MARK: - Encoder/Decoder Heads

/// Encoder output head: WanVAENorm + SiLU + CausalConv3d.
/// Weight keys: head.0.gamma, head.2.weight/bias
/// Index 1 is SiLU (no params) -- we use a placeholder Module.
public final class WanEncoderHead: Module {

  public let layers: [Module]

  public init(dim: Int, zDim: Int) {
    self.layers = [
      WanVAENorm(dim: dim, images: false),
      SiLUPlaceholder(),
      WanCausalConv3d(inChannels: dim, outChannels: zDim, kernelSize: 3, padding: 1),
    ]
    super.init()
  }

  /// Standard forward (no cache).
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = (layers[0] as! WanVAENorm)(x)
    h = silu(h)
    h = (layers[2] as! WanCausalConv3d)(h)
    return h
  }

  /// Cached forward for chunk-by-chunk encoding.
  public func forward(_ x: MLXArray, cache: WanEncoderCache) -> MLXArray {
    var h = (layers[0] as! WanVAENorm)(x)
    h = silu(h)
    h = (layers[2] as! WanCausalConv3d).forward(h, cache: cache)
    return h
  }
}

/// Decoder output head: WanVAENorm + SiLU + CausalConv3d.
public final class WanDecoderHead: Module {

  public let layers: [Module]

  public init(dim: Int, outChannels: Int) {
    self.layers = [
      WanVAENorm(dim: dim, images: false),
      SiLUPlaceholder(),
      WanCausalConv3d(inChannels: dim, outChannels: outChannels, kernelSize: 3, padding: 1),
    ]
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = (layers[0] as! WanVAENorm)(x)
    h = silu(h)
    h = (layers[2] as! WanCausalConv3d)(h)
    return h
  }
}

// MARK: - Placeholder

/// Placeholder for parameter-free layers in Sequential index mapping.
public final class SiLUPlaceholder: Module {
  public override init() {
    super.init()
  }
}
