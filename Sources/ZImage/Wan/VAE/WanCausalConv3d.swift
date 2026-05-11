import Foundation
import MLX
import MLXNN

/// Causal 3D convolution for the Wan 2.1 VAE.
///
/// Directly holds weight and bias parameters so that weight keys match
/// PyTorch: e.g. encoder.conv1.weight maps to this module's weight.
///
/// Applies asymmetric causal padding on the temporal dimension:
/// 2 * padding frames prepended at the start, zero at the end.
public final class WanCausalConv3d: Module {

  public var weight: MLXArray
  public var bias: MLXArray

  public let paddingT: Int
  public let paddingH: Int
  public let paddingW: Int
  public let strideT: Int
  public let strideH: Int
  public let strideW: Int
  public let outChannels: Int

  public init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: (Int, Int, Int),
    stride: (Int, Int, Int) = (1, 1, 1),
    padding: (Int, Int, Int) = (0, 0, 0)
  ) {
    self.paddingT = padding.0
    self.paddingH = padding.1
    self.paddingW = padding.2
    self.strideT = stride.0
    self.strideH = stride.1
    self.strideW = stride.2
    self.outChannels = outChannels

    // MLX Conv3d channels-last: weight shape (outCh, kT, kH, kW, inCh)
    self.weight = MLXArray.zeros([outChannels, kernelSize.0, kernelSize.1, kernelSize.2, inChannels])
    self.bias = MLXArray.zeros([outChannels])

    super.init()
  }

  /// Convenience initializer with isotropic kernel/stride/padding.
  public convenience init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    padding: Int = 0
  ) {
    self.init(
      inChannels: inChannels,
      outChannels: outChannels,
      kernelSize: (kernelSize, kernelSize, kernelSize),
      stride: (stride, stride, stride),
      padding: (padding, padding, padding)
    )
  }

  /// Standard forward pass (no cache).
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var input = x

    if paddingT > 0 || paddingH > 0 || paddingW > 0 {
      input = MLX.padded(
        input,
        widths: [
          IntOrPair((0, 0)),
          IntOrPair((0, 0)),
          IntOrPair((2 * paddingT, 0)),
          IntOrPair((paddingH, paddingH)),
          IntOrPair((paddingW, paddingW)),
        ]
      )
    }

    // Transpose BCTHW -> BTHWC for MLX conv3d
    input = input.transposed(0, 2, 3, 4, 1)

    let output = MLX.conv3d(
      input, weight,
      stride: IntOrTriple((strideT, strideH, strideW)),
      padding: IntOrTriple(0)
    )

    let biased = output + bias

    // Transpose BTHWC -> BCTHW
    return biased.transposed(0, 4, 1, 2, 3)
  }

  /// Cached forward pass for chunk-by-chunk encoding.
  ///
  /// Matches Python's CausalConv3d.forward(x, cache_x=...)
  /// where cache_x replaces some of the zero padding.
  public func forward(_ x: MLXArray, cache: WanEncoderCache) -> MLXArray {
    let slotIdx = cache.advance()
    var input = x

    // Save last CACHE_T=2 frames for next chunk
    let cacheT = 2
    var newCache = input[0..., 0..., (input.dim(2) - cacheT)..., 0..., 0...]
    if newCache.dim(2) < 2, let prev = cache.slots[slotIdx] {
      // If we only have 1 frame and there's a previous cache, prepend
      // the last frame from the previous cache
      let prevLast = prev[0..., 0..., (prev.dim(2) - 1)...(prev.dim(2) - 1), 0..., 0...]
      newCache = MLX.concatenated([prevLast, newCache], axis: 2)
    }

    if paddingT > 0 || paddingH > 0 || paddingW > 0 {
      var causalPad = 2 * paddingT
      if let cachedX = cache.slots[slotIdx], causalPad > 0 {
        // Prepend cached frames instead of zeros
        input = MLX.concatenated([cachedX, input], axis: 2)
        causalPad = max(0, causalPad - cachedX.dim(2))
      }
      if causalPad > 0 || paddingH > 0 || paddingW > 0 {
        input = MLX.padded(
          input,
          widths: [
            IntOrPair((0, 0)),
            IntOrPair((0, 0)),
            IntOrPair((causalPad, 0)),
            IntOrPair((paddingH, paddingH)),
            IntOrPair((paddingW, paddingW)),
          ]
        )
      }
    }

    cache.slots[slotIdx] = newCache

    // Transpose BCTHW -> BTHWC for MLX conv3d
    input = input.transposed(0, 2, 3, 4, 1)

    let output = MLX.conv3d(
      input, weight,
      stride: IntOrTriple((strideT, strideH, strideW)),
      padding: IntOrTriple(0)
    )

    let biased = output + bias

    // Transpose BTHWC -> BCTHW
    return biased.transposed(0, 4, 1, 2, 3)
  }
}
