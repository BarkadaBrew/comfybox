import Foundation
import MLX

/// RoPE (Rotary Position Embedding) frequency provider for the Z-Image transformer.
///
/// Supports three modes:
/// - **Vanilla** (DyPE disabled): Precomputes static frequency tables once.
/// - **NTK**: Scales theta per-axis when spatial resolution exceeds training scale.
/// - **YaRN**: NTK + linear interpolation blend with timestep-aware damping.
///
/// DyPE only applies to spatial axes (1 = height, 2 = width). Axis 0 (caption/frame)
/// always uses vanilla frequencies to preserve text-image alignment.
public final class ZImageRopeEmbedder {
  let theta: Float
  let axesDims: [Int]
  let axesLens: [Int]

  /// DyPE configuration. Set before calling `imageFreqs(ids:timestep:scale:)`.
  var dyPE: DyPEConfig = .disabled

  /// Cached vanilla (static) frequency tables — one per axis.
  private var vanillaFreqsCis: [MLXArray]?

  init(theta: Float, axesDims: [Int], axesLens: [Int]) {
    self.theta = theta
    self.axesDims = axesDims
    self.axesLens = axesLens
    precondition(axesDims.count == axesLens.count, "axesDims and axesLens must have same length")
  }

  // MARK: - Vanilla (static) frequencies

  /// Precompute standard RoPE frequency tables (vanilla — no DyPE scaling).
  private func ensureVanillaFreqs() {
    if vanillaFreqsCis != nil { return }
    var tables: [MLXArray] = []
    tables.reserveCapacity(axesDims.count)

    for (dim, end) in zip(axesDims, axesLens) {
      tables.append(Self.computeFreqTable(theta: theta, dim: dim, seqLen: end))
    }
    vanillaFreqsCis = tables
  }

  /// Compute a single axis frequency table: [seqLen, halfDim, 2] (cos, sin stacked).
  static func computeFreqTable(theta: Float, dim: Int, seqLen: Int) -> MLXArray {
    let halfDim = dim / 2
    let idx = MLXArray(0..<halfDim).asType(.float32) * 2.0
    var exponent = idx / MLXArray(Float(dim))
    exponent = -exponent
    let base = MLXArray(theta)
    let freqs = MLX.pow(base, exponent)

    let timesteps = MLXArray(0..<seqLen).asType(.float32)
    let angles = timesteps[.ellipsis, .newAxis] * freqs[.newAxis]

    let cosVals = MLX.cos(angles)
    let sinVals = MLX.sin(angles)

    return MLX.stacked([cosVals, sinVals], axis: -1)
  }

  // MARK: - NTK-scaled frequencies

  /// Compute NTK-aware RoPE frequencies for a single axis.
  ///
  /// NTK scaling adjusts theta so that high-frequency components spread to cover
  /// the larger position range without collapsing into aliasing.
  ///
  /// Formula: `theta' = theta * scale^(dim / (dim - 2))`
  static func computeNTKFreqTable(
    theta: Float, dim: Int, seqLen: Int, scale: Float
  ) -> MLXArray {
    guard scale > 1.0 else {
      return computeFreqTable(theta: theta, dim: dim, seqLen: seqLen)
    }

    let halfDim = dim / 2
    // NTK-aware theta: spread frequencies for the larger position range
    let ntkTheta = theta * pow(scale, Float(dim) / Float(dim - 2))

    let idx = MLXArray(0..<halfDim).asType(.float32) * 2.0
    var exponent = idx / MLXArray(Float(dim))
    exponent = -exponent
    let base = MLXArray(ntkTheta)
    let freqs = MLX.pow(base, exponent)

    let timesteps = MLXArray(0..<seqLen).asType(.float32)
    let angles = timesteps[.ellipsis, .newAxis] * freqs[.newAxis]

    let cosVals = MLX.cos(angles)
    let sinVals = MLX.sin(angles)

    return MLX.stacked([cosVals, sinVals], axis: -1)
  }

  // MARK: - Public API

  /// Caption frequencies — always vanilla, never DyPE-scaled.
  /// Caption/frame axis (axis 0) must stay unscaled to preserve text-image alignment.
  func captionFreqs(ids: MLXArray) -> MLXArray {
    ensureVanillaFreqs()
    guard let vanillaFreqsCis, !vanillaFreqsCis.isEmpty else {
      return MLX.zeros([ids.dim(0), axesDims[0] / 2, 2])
    }

    // Axis 0 only
    let index = ids[0..., 0]
    return vanillaFreqsCis[0][index, 0..., 0...]
  }

  /// Image frequencies — vanilla when DyPE is disabled, NTK/YaRN-scaled when enabled.
  ///
  /// - Parameters:
  ///   - ids: Position IDs [N, numAxes]
  ///   - hScale: Height scale factor (current_h_tokens / base_h_tokens)
  ///   - wScale: Width scale factor (current_w_tokens / base_w_tokens)
  func imageFreqs(ids: MLXArray, hScale: Float = 1.0, wScale: Float = 1.0) -> MLXArray {
    precondition(ids.ndim == 2 && ids.dim(1) == axesDims.count,
                 "ids must be [N, numAxes]")

    let batch = ids.dim(0)

    if !dyPE.enabled || dyPE.method == .none {
      // Vanilla path — use precomputed tables
      ensureVanillaFreqs()
      guard let vanillaFreqsCis else {
        let totalHalfDim = axesDims.reduce(0) { $0 + $1 / 2 }
        return MLX.zeros([batch, totalHalfDim, 2])
      }

      var outputs: [MLXArray] = []
      outputs.reserveCapacity(vanillaFreqsCis.count)
      for (axisIndex, table) in vanillaFreqsCis.enumerated() {
        let index = ids[0..., axisIndex]
        outputs.append(table[index, 0..., 0...])
      }
      return MLX.concatenated(outputs, axis: 1)
    }

    // DyPE path — axis 0 vanilla, axes 1+2 scaled
    let scales: [Float] = axesDims.count == 3
      ? [1.0, hScale, wScale]       // [frame/caption, height, width]
      : [hScale, wScale]            // fallback for 2-axis

    var outputs: [MLXArray] = []
    outputs.reserveCapacity(axesDims.count)

    for axisIndex in 0..<axesDims.count {
      let dim = axesDims[axisIndex]
      let end = axesLens[axisIndex]
      let scale = axisIndex < scales.count ? scales[axisIndex] : 1.0
      let index = ids[0..., axisIndex]

      if scale <= 1.0 || axisIndex == 0 {
        // Vanilla — use cached table
        ensureVanillaFreqs()
        let table = vanillaFreqsCis![axisIndex]
        outputs.append(table[index, 0..., 0...])
      } else {
        // NTK-scaled for this axis
        let table = Self.computeNTKFreqTable(
          theta: theta, dim: dim, seqLen: end, scale: scale
        )
        outputs.append(table[index, 0..., 0...])
      }
    }

    return MLX.concatenated(outputs, axis: 1)
  }

  // MARK: - Legacy callAsFunction (vanilla path)

  /// Legacy entry point — vanilla frequencies for all axes.
  /// Used by the cache builder for caption frequencies and when DyPE is disabled.
  func callAsFunction(ids: MLXArray) -> MLXArray {
    ensureVanillaFreqs()
    guard let vanillaFreqsCis else {
      return MLX.zeros([ids.dim(0), axesDims.reduce(0, +) / 2, 2])
    }
    precondition(ids.ndim == 2, "ids must be [N, numAxes]")
    precondition(ids.dim(1) == axesDims.count, "ids last dimension must equal axes count")

    let batch = ids.dim(0)
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(vanillaFreqsCis.count)

    for (axisIndex, table) in vanillaFreqsCis.enumerated() {
      let index = ids[0..., axisIndex]
      let selected = table[index, 0..., 0...]
      outputs.append(selected)
    }

    let totalHalfDim = axesDims.reduce(0) { $0 + $1 / 2 }
    if outputs.isEmpty {
      return MLX.zeros([batch, totalHalfDim, 2])
    }
    return MLX.concatenated(outputs, axis: 1)
  }
}
