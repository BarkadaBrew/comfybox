import Foundation
import MLX

/// 3D Rotary Position Embeddings for the Wan 2.2 transformer.
///
/// Wan uses a unique 3D RoPE where the head dimension is split into three
/// frequency sets for temporal, height, and width axes:
///
/// ```
/// headDim = 128 -> 64 complex pairs
/// Split: [22, 21, 21] complex pairs for [temporal, height, width]
/// ```
///
/// Each token's position is encoded based on its (f, h, w) coordinate in
/// the 3D patch grid. Applied to BOTH q and k after qk-norm, before attention.
/// NOT applied in cross-attention.
public enum WanRoPE {

  // MARK: - Frequency Generation

  /// Computes complex rotation frequencies for a single axis.
  ///
  /// Matches the Python `rope_params(max_seq_len, dim, theta=10000)`:
  /// ```python
  /// freqs = outer(arange(max_seq_len), 1/pow(theta, arange(0, dim, 2)/dim))
  /// freqs = polar(ones_like(freqs), freqs)  # complex exponential
  /// ```
  ///
  /// - Parameters:
  ///   - maxSeqLen: Maximum sequence length.
  ///   - dim: Full dimension for this frequency set (must be even).
  ///   - theta: Base frequency. Default 10000.
  /// - Returns: Real/imag frequency tensor of shape `[maxSeqLen, dim/2, 2]`.
  public static func ropeParams(maxSeqLen: Int, dim: Int, theta: Float = 10000.0) -> MLXArray {
    precondition(dim % 2 == 0, "RoPE dim must be even")
    let halfDim = dim / 2

    // positions: [maxSeqLen]
    let positions = MLXArray(Array(0..<maxSeqLen).map { Float($0) })

    // frequencies: 1/theta^(2k/dim) for k in 0..<halfDim
    let exponents = MLXArray(Array(0..<halfDim).map { Float($0 * 2) / Float(dim) })
    let invFreqs = MLXArray(Float(1.0)) / MLX.pow(MLXArray(theta), exponents)

    // outer product -> angles: [maxSeqLen, halfDim]
    let angles = MLX.matmul(positions.reshaped(-1, 1), invFreqs.reshaped(1, -1))

    // Convert to complex: [cos(angle), sin(angle)] -> [maxSeqLen, halfDim, 2]
    let cosA = MLX.cos(angles).expandedDimensions(axis: -1)
    let sinA = MLX.sin(angles).expandedDimensions(axis: -1)
    let freqs = MLX.concatenated([cosA, sinA], axis: -1)

    return freqs
  }

  /// Builds combined 3D RoPE frequencies for the Wan transformer.
  ///
  /// Concatenates temporal, height, and width frequency sets:
  /// ```
  /// d = headDim (128)
  /// temporal: d - 4*(d//6) = 44 real dims -> 22 complex pairs
  /// height:   2*(d//6) = 42 real dims -> 21 complex pairs
  /// width:    2*(d//6) = 42 real dims -> 21 complex pairs
  /// Total: 22 + 21 + 21 = 64 complex pairs
  /// ```
  ///
  /// - Parameters:
  ///   - maxSeqLen: Maximum sequence length per axis. Default 1024.
  ///   - headDim: Per-head dimension. Default 128.
  /// - Returns: Combined frequencies of shape `[maxSeqLen, headDim/2, 2]`.
  public static func buildFrequencies(maxSeqLen: Int = 1024, headDim: Int = 128) -> MLXArray {
    let d = headDim
    let temporalDim = d - 4 * (d / 6)  // 44 for headDim=128
    let heightDim = 2 * (d / 6)         // 42
    let widthDim = 2 * (d / 6)          // 42

    let temporalFreqs = ropeParams(maxSeqLen: maxSeqLen, dim: temporalDim)
    let heightFreqs = ropeParams(maxSeqLen: maxSeqLen, dim: heightDim)
    let widthFreqs = ropeParams(maxSeqLen: maxSeqLen, dim: widthDim)

    // Concatenate along the complex-pair axis
    let combined = MLX.concatenated([temporalFreqs, heightFreqs, widthFreqs], axis: 1)
    return combined
  }

  // MARK: - RoPE Application

  /// Applies 3D rotary position embeddings to queries or keys.
  ///
  /// Matches the Python `rope_apply(x, grid_sizes, freqs)`:
  /// - Splits frequencies into [temporal, height, width] sets
  /// - For each sample, expands frequencies to match 3D grid
  /// - Applies complex rotation: x_complex * freqs_complex
  /// - Padding tokens (beyond seq_len) are left unchanged
  ///
  /// - Parameters:
  ///   - x: Input tensor of shape `[B, seqLen, numHeads, headDim]`.
  ///   - gridSizes: 3D grid dimensions per sample, each `[F, H, W]`.
  ///   - freqs: Precomputed frequencies of shape `[maxSeqLen, headDim/2, 2]`.
  /// - Returns: Rotated tensor of shape `[B, seqLen, numHeads, headDim]`.
  public static func ropeApply(
    _ x: MLXArray,
    gridSizes: [[Int]],
    freqs: MLXArray
  ) -> MLXArray {
    let b = x.dim(0)
    let s = x.dim(1)
    let n = x.dim(2)
    let headDim = x.dim(3)
    let c = headDim / 2  // number of complex pairs

    // Split frequency dimensions: [temporal, height, width]
    let splitT = c - 2 * (c / 3)  // 22 for c=64
    let splitH = c / 3              // 21
    let splitW = c / 3              // 21

    // freqs is [maxSeqLen, c, 2] -- split along axis 1
    let freqsT = freqs[0..., 0..<splitT, 0...]
    let freqsH = freqs[0..., splitT..<(splitT + splitH), 0...]
    let freqsW = freqs[0..., (splitT + splitH)..<(splitT + splitH + splitW), 0...]

    var outputs: [MLXArray] = []

    for i in 0..<b {
      let f = gridSizes[i][0]
      let h = gridSizes[i][1]
      let w = gridSizes[i][2]
      let seqLen = f * h * w

      // Extract real tokens: [seqLen, n, headDim]
      let xi = x[i, 0..<seqLen]

      // Reshape to complex pairs: [seqLen, n, c, 2]
      let xiComplex = xi.reshaped(seqLen, n, c, 2)

      // Build per-position frequencies by expanding 3D grid
      // temporal: [f, splitT, 2] -> [f, 1, 1, splitT, 2] -> broadcast to [f, h, w, splitT, 2]
      let ftExp = MLX.broadcast(
        freqsT[0..<f].reshaped(f, 1, 1, splitT, 2),
        to: [f, h, w, splitT, 2]
      )
      // height: [h, splitH, 2] -> [1, h, 1, splitH, 2] -> broadcast to [f, h, w, splitH, 2]
      let fhExp = MLX.broadcast(
        freqsH[0..<h].reshaped(1, h, 1, splitH, 2),
        to: [f, h, w, splitH, 2]
      )
      // width: [w, splitW, 2] -> [1, 1, w, splitW, 2] -> broadcast to [f, h, w, splitW, 2]
      let fwExp = MLX.broadcast(
        freqsW[0..<w].reshaped(1, 1, w, splitW, 2),
        to: [f, h, w, splitW, 2]
      )

      // Concatenate frequencies: [f, h, w, c, 2] -> [seqLen, 1, c, 2]
      let freqsCombined = MLX.concatenated([ftExp, fhExp, fwExp], axis: 3)
        .reshaped(seqLen, 1, c, 2)

      // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
      let xReal = xiComplex[0..., 0..., 0..., 0...0]  // [seqLen, n, c, 1]
      let xImag = xiComplex[0..., 0..., 0..., 1...1]
      let fReal = freqsCombined[0..., 0..., 0..., 0...0]  // [seqLen, 1, c, 1]
      let fImag = freqsCombined[0..., 0..., 0..., 1...1]

      let outReal = xReal * fReal - xImag * fImag
      let outImag = xReal * fImag + xImag * fReal

      // Recombine: [seqLen, n, c, 2] -> [seqLen, n, headDim]
      let rotated = MLX.concatenated([outReal, outImag], axis: -1)
        .reshaped(seqLen, n, headDim)

      // Append padding tokens unchanged
      if seqLen < s {
        let padding = x[i, seqLen..<s]  // [padLen, n, headDim]
        let combined = MLX.concatenated([rotated, padding], axis: 0)
        outputs.append(combined)
      } else {
        outputs.append(rotated)
      }
    }

    return MLX.stacked(outputs, axis: 0).asType(.float32)
  }
}
