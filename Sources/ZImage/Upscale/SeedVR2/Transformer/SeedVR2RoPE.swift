import Foundation
import MLX
import MLXFast
import MLXNN

/// 3D axial rotary position embeddings for the SeedVR2 transformer.
///
/// Applies learned-frequency RoPE along temporal, height, and width axes independently.
/// For video tokens, the three axes produce separate frequency grids that are concatenated.
/// For text tokens, a 1D frequency grid is tiled across all three axes.
///
/// ## Key Dimensions
///
/// - `dim = 128` (head_dim)
/// - `freq_dim = dim / 3 = 42` (frequencies per axis)
/// - `freqs` parameter: shape `(freq_dim / 2,)` = `(21,)` — shared across axes
/// - Rotation covers first `3 * 42 = 126` dims; remaining 2 pass through
///
/// ## Text Temporal Offset
///
/// Video temporal positions are offset by the text sequence length, so that
/// text occupies positions `[0, txt_len)` and video occupies `[txt_len, txt_len+T)`.
///
/// ## Weight Key Paths
///
/// - `rope.freqs` — shape `(21,)`
public final class SeedVR2RoPE: Module {

  /// Feature dimension (128 = head_dim).
  public let dim: Int

  /// Number of spatial axes for RoPE (3: temporal, height, width).
  public let ropeDim: Int

  /// Features per axis: dim / ropeDim = 42.
  public let freqDim: Int

  /// Learned base frequencies, shape `(freqDim / 2,)` = `(21,)`.
  @ParameterInfo(key: "freqs") var freqs: MLXArray

  /// Creates a 3D RoPE module.
  ///
  /// - Parameter dim: Head dimension. Default `128`.
  public init(dim: Int = 128) {
    self.dim = dim
    self.ropeDim = 3
    self.freqDim = dim / 3  // 42

    // Initialize with standard geometric frequencies:
    // freqs = 1 / (theta^(arange(0, freq_dim, 2)[:freq_dim//2] / freq_dim))
    // where theta = 10000
    let theta: Float = 10000.0
    let halfFreqDim = freqDim / 2  // 21
    // arange(0, freq_dim, 2)[:halfFreqDim] as float
    var idxArray: [Float] = []
    for i in stride(from: 0, to: freqDim, by: 2).prefix(halfFreqDim) {
      idxArray.append(Float(i))
    }
    let indices = MLXArray(idxArray)
    let exponents = indices / Float(freqDim)
    self._freqs.wrappedValue = MLXArray(1.0) / MLXArray(theta).pow(exponents)

    super.init()
  }

  /// Applies 3D axial RoPE to video and text Q/K tensors.
  ///
  /// - Parameters:
  ///   - vidQ: Video queries, shape `(N_vid, heads, head_dim)`.
  ///   - vidK: Video keys, shape `(N_vid, heads, head_dim)`.
  ///   - vidShape: Window shapes, `(num_windows, 3)` containing `[T, H, W]`.
  ///   - txtQ: Text queries, shape `(N_txt, heads, head_dim)`.
  ///   - txtK: Text keys, shape `(N_txt, heads, head_dim)`.
  ///   - txtShape: Text shapes, `(num_windows, 1)` containing text length per window.
  /// - Returns: Tuple of rotated (vidQ, vidK, txtQ, txtK).
  public func callAsFunction(
    vidQ: MLXArray,
    vidK: MLXArray,
    vidShape: MLXArray,
    txtQ: MLXArray,
    txtK: MLXArray,
    txtShape: MLXArray
  ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    return SeedVR2RoPE.applyMMRoPE3D(
      vidQ: vidQ, vidK: vidK, vidShape: vidShape,
      txtQ: txtQ, txtK: txtK, txtShape: txtShape,
      freqs: freqs, ropeDim: ropeDim
    )
  }

  // MARK: - Static Helpers

  /// Rotates the second half of each pair: `[-x2, x1]` interleaved.
  private static func rotateHalf(_ x: MLXArray) -> MLXArray {
    // Reshape last dim into pairs: (..., dim/2, 2)
    var shape = x.shape
    let lastDim = shape[shape.count - 1]
    shape[shape.count - 1] = lastDim / 2
    shape.append(2)

    let paired = x.reshaped(shape)
    let x1 = paired[.ellipsis, 0]
    let x2 = paired[.ellipsis, 1]

    // Stack [-x2, x1] and flatten back
    let rotated = MLX.stacked([-x2, x1], axis: -1)

    var outShape = x.shape
    return rotated.reshaped(outShape)
  }

  /// Computes axial frequency grids for N-dimensional positions.
  ///
  /// For 3D (T, H, W): produces a `(T, H, W, 3*freq_dim)` frequency tensor.
  /// For 1D (L): produces a `(L, freq_dim)` frequency tensor.
  ///
  /// Each axis contributes `freq_dim` features (half-cos, half-sin alternating pairs).
  private static func getAxialFreqs(freqs: MLXArray, dims: [Int]) -> MLXArray {
    let freqDimPerAxis = freqs.dim(0) * 2  // 42
    var targetShape = dims + [freqDimPerAxis]

    var allFreqs: [MLXArray] = []

    for (ind, dimSize) in dims.enumerated() {
      // pos = arange(dimSize, float32)
      let pos = MLXArray(Int32(0) ..< Int32(dimSize)).asType(.float32)

      // axis_freqs = outer(pos, freqs) → (dimSize, freqDim/2)
      let axisFreqsRaw = MLX.outer(pos, freqs.asType(.float32))

      // Repeat each freq value → (dimSize, freqDim) = interleaved cos/sin positions
      let axisFreqs = MLX.repeated(axisFreqsRaw, count: 2, axis: 1)

      // Reshape to broadcast shape: [1, ..., dimSize, ..., 1, freqDimPerAxis]
      var shape = [Int](repeating: 1, count: dims.count) + [freqDimPerAxis]
      shape[ind] = dimSize
      let reshaped = axisFreqs.reshaped(shape)

      allFreqs.append(reshaped)
    }

    // Broadcast all to target shape and concatenate along last axis
    let broadcasted = allFreqs.map { MLX.broadcast($0, to: targetShape) }
    return MLX.concatenated(broadcasted, axis: -1)
  }

  /// Applies rotary embedding to a tensor using precomputed frequencies.
  ///
  /// Rotates the first `rot_dim` features, passes the rest through unchanged.
  private static func applyRotaryEmb(freqs: MLXArray, t: MLXArray) -> MLXArray {
    let rotDim = freqs.dim(-1)
    let tRotate = t[.ellipsis, 0 ..< rotDim]
    let tPass = t[.ellipsis, rotDim...]

    let freqsF = freqs.asType(.float32)
    let tF = tRotate.asType(.float32)

    let cosFreqs = freqsF.cos()
    let sinFreqs = freqsF.sin()
    var transformed = (tF * cosFreqs) + (rotateHalf(tF) * sinFreqs)
    transformed = transformed.asType(t.dtype)

    if tPass.dim(-1) > 0 {
      return MLX.concatenated([transformed, tPass], axis: -1)
    }
    return transformed
  }

  /// Full multi-modal 3D RoPE application.
  ///
  /// Computes separate 3D frequency grids for video (with text offset on temporal axis)
  /// and 1D grids for text, then applies rotary embeddings per window.
  private static func applyMMRoPE3D(
    vidQ: MLXArray, vidK: MLXArray, vidShape: MLXArray,
    txtQ: MLXArray, txtK: MLXArray, txtShape: MLXArray,
    freqs: MLXArray, ropeDim: Int
  ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    // Compute max dimensions for the precomputed grid
    let maxTemporal = Int(MLX.max(vidShape[0..., 0] + txtShape[0..., 0]).item(Int32.self))
    let maxHeight = Int(MLX.max(vidShape[0..., 1]).item(Int32.self))
    let maxWidth = Int(MLX.max(vidShape[0..., 2]).item(Int32.self))
    let maxTxtLen = Int(MLX.max(txtShape[0..., 0]).item(Int32.self))

    // Clamp with safety margin
    let clampTemporal = min(maxTemporal + 16, 1024)
    let clampHeight = min(maxHeight + 4, 128)
    let clampWidth = min(maxWidth + 4, 128)

    // Precompute full 3D freq grid: (clampT, clampH, clampW, 3*freqDim)
    let vidFreqsFull = getAxialFreqs(freqs: freqs, dims: [clampTemporal, clampHeight, clampWidth])

    // Precompute 1D freq grid for text: (maxTxtLen+margin, freqDim)
    let txtFreqs1D = getAxialFreqs(freqs: freqs, dims: [min(maxTxtLen + 16, 1024)])

    let numWindows = vidShape.dim(0)
    var vidFreqList: [MLXArray] = []
    var txtFreqList: [MLXArray] = []

    for b in 0 ..< numWindows {
      let f = Int(vidShape[b, 0].item(Int32.self))
      let h = Int(vidShape[b, 1].item(Int32.self))
      let w = Int(vidShape[b, 2].item(Int32.self))
      let txtLen = Int(txtShape[b, 0].item(Int32.self))

      // Video: offset temporal by txtLen to avoid position collision
      // vid_freq = vid_freqs_full[txtLen:txtLen+f, :h, :w].reshape(-1, D)
      let vidFreq = vidFreqsFull[txtLen ..< (txtLen + f), 0 ..< h, 0 ..< w]
        .reshaped(-1, vidFreqsFull.dim(-1))

      // Text: tile 1D freqs ropeDim times → (txtLen, ropeDim * freqDim)
      let txtFreqSlice = txtFreqs1D[0 ..< txtLen]
      let txtFreq = MLX.tiled(txtFreqSlice, repetitions: [1, ropeDim])

      vidFreqList.append(vidFreq)
      txtFreqList.append(txtFreq)
    }

    // Concatenate all windows
    let vidFreqsAll = MLX.concatenated(vidFreqList, axis: 0)
    let txtFreqsAll = MLX.concatenated(txtFreqList, axis: 0)

    // Apply rotary embeddings: freqs[:, None, :] broadcasts over heads
    let vidFreqsExpanded = vidFreqsAll.expandedDimensions(axis: 1)
    let txtFreqsExpanded = txtFreqsAll.expandedDimensions(axis: 1)

    let rotVidQ = applyRotaryEmb(freqs: vidFreqsExpanded, t: vidQ)
    let rotVidK = applyRotaryEmb(freqs: vidFreqsExpanded, t: vidK)
    let rotTxtQ = applyRotaryEmb(freqs: txtFreqsExpanded, t: txtQ)
    let rotTxtK = applyRotaryEmb(freqs: txtFreqsExpanded, t: txtK)

    return (rotVidQ, rotVidK, rotTxtQ, rotTxtK)
  }
}
