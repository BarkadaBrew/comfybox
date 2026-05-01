import Foundation
import MLX
import MLXFast
import MLXNN

/// Multi-modal windowed attention for the SeedVR2 transformer.
///
/// ## Forward Pass
///
/// 1. Project vid and txt to fused QKV (`dim -> 3 * heads * head_dim`)
/// 2. Window partition vid tokens via ``SeedVR2WindowPartitioner``
/// 3. QK-normalize (per-head RMSNorm)
/// 4. Repeat text tokens for each window
/// 5. Apply 3D RoPE to vid, 1D to txt
/// 6. Concatenate vid+txt per window
/// 7. Scaled dot-product attention per window
/// 8. Split vid and txt from attention output
/// 9. Coalesce text (mean across window copies)
/// 10. Reverse window partition for vid
/// 11. Project out
///
/// ## Shared Mode (blocks 10--31)
///
/// When `sharedWeights=true`, the vid projections are used for BOTH modalities
/// during forward. Separate txt modules still exist for weight loading compatibility,
/// but are not called.
///
/// ## Weight Key Paths
///
/// - `attn.proj_qkv_vid.weight`, `attn.proj_qkv_txt.weight`
/// - `attn.proj_out_vid.{weight,bias}`, `attn.proj_out_txt.{weight,bias}`
/// - `attn.norm_q_vid.weight`, `attn.norm_k_vid.weight`
/// - `attn.norm_q_txt.weight`, `attn.norm_k_txt.weight`
/// - `attn.rope.freqs`
public final class SeedVR2MMAttention: Module {

  public let sharedWeights: Bool
  public let heads: Int
  public let headDim: Int
  public let scale: Float
  public let window: (Int, Int, Int)
  public let shift: Bool

  // Video projections (always present)
  @ModuleInfo(key: "proj_qkv_vid") var projQkvVid: Linear
  @ModuleInfo(key: "proj_out_vid") var projOutVid: Linear
  @ModuleInfo(key: "norm_q_vid") var normQVid: RMSNorm
  @ModuleInfo(key: "norm_k_vid") var normKVid: RMSNorm

  // Text projections (always present for weight loading; aliased to vid in shared mode forward)
  @ModuleInfo(key: "proj_qkv_txt") var projQkvTxt: Linear
  @ModuleInfo(key: "proj_out_txt") var projOutTxt: Linear
  @ModuleInfo(key: "norm_q_txt") var normQTxt: RMSNorm
  @ModuleInfo(key: "norm_k_txt") var normKTxt: RMSNorm

  // 3D RoPE
  @ModuleInfo(key: "rope") var rope: SeedVR2RoPE

  /// Creates a multi-modal attention module.
  ///
  /// - Parameters:
  ///   - vidDim: Video feature dimension. Default `2560`.
  ///   - txtDim: Text feature dimension. Default `2560`.
  ///   - heads: Number of attention heads. Default `20`.
  ///   - headDim: Dimension per head. Default `128`.
  ///   - qkBias: Whether QKV projections have bias. Default `false`.
  ///   - qkNormEps: Epsilon for QK normalization. Default `1e-5`.
  ///   - ropeDim: RoPE dimension. Default `128`.
  ///   - sharedWeights: If true, vid projections serve both modalities. Default `false`.
  ///   - window: Number of windows per axis. Default `(4, 3, 3)`.
  ///   - shift: Whether to shift windows. Default `false`.
  public init(
    vidDim: Int = 2560,
    txtDim: Int = 2560,
    heads: Int = 20,
    headDim: Int = 128,
    qkBias: Bool = false,
    qkNormEps: Float = 1e-5,
    ropeDim: Int = 128,
    sharedWeights: Bool = false,
    window: (Int, Int, Int) = (4, 3, 3),
    shift: Bool = false
  ) {
    self.sharedWeights = sharedWeights
    self.heads = heads
    self.headDim = headDim
    self.scale = pow(Float(headDim), -0.5)
    self.window = window
    self.shift = shift

    let innerDim = heads * headDim

    // Video modules
    self._projQkvVid.wrappedValue = Linear(vidDim, 3 * innerDim, bias: qkBias)
    self._projOutVid.wrappedValue = Linear(innerDim, vidDim, bias: true)
    self._normQVid.wrappedValue = RMSNorm(dimensions: headDim, eps: qkNormEps)
    self._normKVid.wrappedValue = RMSNorm(dimensions: headDim, eps: qkNormEps)

    // Text modules (always created for checkpoint compatibility)
    self._projQkvTxt.wrappedValue = Linear(txtDim, 3 * innerDim, bias: qkBias)
    self._projOutTxt.wrappedValue = Linear(innerDim, txtDim, bias: true)
    self._normQTxt.wrappedValue = RMSNorm(dimensions: headDim, eps: qkNormEps)
    self._normKTxt.wrappedValue = RMSNorm(dimensions: headDim, eps: qkNormEps)

    // RoPE
    self._rope.wrappedValue = SeedVR2RoPE(dim: ropeDim)

    super.init()
  }

  /// Applies multi-modal windowed attention.
  ///
  /// - Parameters:
  ///   - vid: Video tokens, shape `(B, L_vid, vid_dim)`.
  ///   - txt: Text tokens, shape `(B, L_txt, txt_dim)`.
  ///   - vidShape: Video spatial shape, `(B, 3)`.
  ///   - txtShape: Text length, `(B, 1)`.
  /// - Returns: Tuple of (vid_out, txt_out) with same shapes as input.
  public func callAsFunction(
    _ vid: MLXArray,
    _ txt: MLXArray,
    _ vidShape: MLXArray,
    _ txtShape: MLXArray
  ) -> (MLXArray, MLXArray) {
    let bSize = vid.dim(0)
    let lVid = vid.dim(1)
    let lTxt = txt.dim(1)

    // Select projections: in shared mode, use vid projections for both
    let qkvProjVid = projQkvVid
    let qkvProjTxt = sharedWeights ? projQkvVid : projQkvTxt
    let outProjVid = projOutVid
    let outProjTxt = sharedWeights ? projOutVid : projOutTxt
    let nqVid = normQVid
    let nkVid = normKVid
    let nqTxt = sharedWeights ? normQVid : normQTxt
    let nkTxt = sharedWeights ? normKVid : normKTxt

    // 1. Project to QKV
    // vid: (B, L, D) → flatten → (B*L, D) → proj → (B*L, 3*heads*head_dim) → (B*L, 3, heads, head_dim)
    let vidFlat = vid.reshaped(-1, vid.dim(-1))
    var qkvVid = qkvProjVid(vidFlat).reshaped(-1, 3, heads, headDim)

    let txtFlat = txt.reshaped(-1, txt.dim(-1))
    let qkvTxt = qkvProjTxt(txtFlat).reshaped(-1, 3, heads, headDim)

    // 2. Window partition vid tokens
    let partitioner = SeedVR2WindowPartitioner(shape: vidShape, windowSize: window, shift: shift)
    qkvVid = partitioner.partition(qkvVid)

    // 3. Split Q/K/V and normalize Q,K
    // qkvVid shape: (N_vid_windowed, 3, heads, head_dim)
    let qVid = nqVid(qkvVid[0..., 0])   // (N, heads, head_dim)
    let kVid = nkVid(qkvVid[0..., 1])
    let vVid = qkvVid[0..., 2]

    let qTxt = nqTxt(qkvTxt[0..., 0])
    let kTxt = nkTxt(qkvTxt[0..., 1])
    let vTxt = qkvTxt[0..., 2]

    // 4. Repeat text for each window
    let counts = partitioner.windowCounts
    let txtLen = txtShape[0..., 0]  // (B,) text lengths

    // Stack Q/K/V for text: (N_txt, 3, heads, head_dim)
    let qkvTxtStacked = MLX.stacked([qTxt, kTxt, vTxt], axis: 1)

    // Repeat text per window: reshape to (B, L_txt, 3, heads, head_dim) then repeat along batch
    let qkvTxtRepeated = SeedVR2MMAttention.repeatTextForWindows(
      qkvTxtStacked, txtLen: txtLen, counts: counts
    )
    let qTxtRep = qkvTxtRepeated[0..., 0]
    let kTxtRep = qkvTxtRepeated[0..., 1]
    let vTxtRep = qkvTxtRepeated[0..., 2]

    // 5. Apply RoPE
    let txtShapeRepeated = SeedVR2MMAttention.perElementRepeat(txtShape, counts: counts)
    let (rqVid, rkVid, rqTxt, rkTxt) = rope(
      vidQ: qVid, vidK: kVid, vidShape: partitioner.windowShapes,
      txtQ: qTxtRep, txtK: kTxtRep, txtShape: txtShapeRepeated
    )

    // 6 & 7. Per-window attention
    let vidLens = partitioner.windowShapes.product(axis: 1)  // (num_windows,)
    let winToBatch = SeedVR2MMAttention.repeatIndices(counts: counts)
    let txtLens = txtLen[winToBatch]  // text length per window

    // Concatenate vid+txt per window, run attention, split back
    let vidQKV = MLX.stacked([rqVid, rkVid, vVid], axis: 1)
    let txtQKV = MLX.stacked([rqTxt, rkTxt, vTxtRep], axis: 1)

    let combined = SeedVR2MMAttention.concatWithText(
      vid: vidQKV, txt: txtQKV, vidLens: vidLens, txtLens: txtLens, counts: counts
    )

    let winLens = vidLens + txtLens
    let attnOut = SeedVR2MMAttention.perWindowAttention(
      combined: combined, winLens: winLens, heads: heads, headDim: headDim, scale: scale
    )

    // 8 & 9. Split vid/txt and coalesce text
    let (vidOut, txtOut) = SeedVR2MMAttention.unconcatAndCoalesce(
      combined: attnOut, vidLens: vidLens, txtLens: txtLens, counts: counts
    )

    // 10. Reverse window partition and project out
    let vidResult = outProjVid(partitioner.reverse(vidOut)).reshaped(bSize, lVid, -1)
    let txtResult = outProjTxt(txtOut).reshaped(bSize, lTxt, -1)

    return (vidResult, txtResult)
  }

  // MARK: - Static Helpers

  /// Repeats text tokens for each window.
  ///
  /// Input: (N_txt_total, 3, heads, head_dim), where N_txt_total = sum(B * L_txt)
  /// Output: (sum(counts * L_txt), 3, heads, head_dim)
  private static func repeatTextForWindows(
    _ txt: MLXArray,
    txtLen: MLXArray,
    counts: [Int]
  ) -> MLXArray {
    let batchSize = counts.count
    let lTxt = Int(txtLen[0].item(Int32.self))
    let restShape = Array(txt.shape[1...])

    // Reshape to (B, L_txt, ...)
    let reshaped = txt.reshaped([batchSize, lTxt] + restShape)

    // Per-batch repeat: repeat batch element b by counts[b] times
    var parts: [MLXArray] = []
    for b in 0 ..< batchSize {
      let batchSlice = reshaped[b].expandedDimensions(axis: 0)
      let rep = MLX.repeated(batchSlice, count: counts[b], axis: 0)
      parts.append(rep)
    }
    let joined = MLX.concatenated(parts, axis: 0)

    // Flatten: (sum_counts, L_txt, ...) -> (sum_counts * L_txt, ...)
    let totalWindows = counts.reduce(0, +)
    return joined.reshaped([totalWindows * lTxt] + restShape)
  }

  /// Creates an index array that maps each window to its batch element.
  /// e.g., counts=[3, 2] → [0, 0, 0, 1, 1]
  private static func repeatIndices(counts: [Int]) -> MLXArray {
    var indices: [Int32] = []
    for (batchIdx, count) in counts.enumerated() {
      indices.append(contentsOf: [Int32](repeating: Int32(batchIdx), count: count))
    }
    return MLXArray(indices)
  }

  /// Interleaves vid and txt tokens per window.
  ///
  /// Result: [vid_win0, txt_win0, vid_win1, txt_win1, ...]
  private static func concatWithText(
    vid: MLXArray,
    txt: MLXArray,
    vidLens: MLXArray,
    txtLens: MLXArray,
    counts: [Int]
  ) -> MLXArray {
    let numWindows = counts.reduce(0, +)
    var parts: [MLXArray] = []

    var vidOffset = 0
    var txtOffset = 0

    for w in 0 ..< numWindows {
      let vLen = Int(vidLens[w].item(Int32.self))
      let tLen = Int(txtLens[w].item(Int32.self))

      parts.append(vid[vidOffset ..< (vidOffset + vLen)])
      parts.append(txt[txtOffset ..< (txtOffset + tLen)])

      vidOffset += vLen
      txtOffset += tLen
    }

    return MLX.concatenated(parts, axis: 0)
  }

  /// Runs scaled dot-product attention independently per window.
  ///
  /// Input `combined`: concatenated (vid+txt) tokens per window with shape
  /// `(total_tokens, 3, heads, head_dim)`, where the `3` axis is [Q, K, V].
  private static func perWindowAttention(
    combined: MLXArray,
    winLens: MLXArray,
    heads: Int,
    headDim: Int,
    scale: Float
  ) -> MLXArray {
    let numWindows = winLens.dim(0)

    // Compute cumulative split indices
    var splitIndices: [Int] = []
    var cumLen = 0
    for w in 0 ..< numWindows - 1 {
      cumLen += Int(winLens[w].item(Int32.self))
      splitIndices.append(cumLen)
    }

    let windows = combined.split(indices: splitIndices, axis: 0)
    var outputs: [MLXArray] = []

    for w in windows {
      // w shape: (winLen, 3, heads, head_dim)
      // Extract Q, K, V and reshape for SDPA: (1, heads, winLen, head_dim)
      let q = w[0..., 0].expandedDimensions(axis: 0).transposed(0, 2, 1, 3)
      let k = w[0..., 1].expandedDimensions(axis: 0).transposed(0, 2, 1, 3)
      let v = w[0..., 2].expandedDimensions(axis: 0).transposed(0, 2, 1, 3)

      // SDPA: (1, heads, winLen, head_dim) → squeeze → (winLen, heads, head_dim)
      var o = MLXFast.scaledDotProductAttention(
        queries: q, keys: k, values: v, scale: scale, mask: nil
      )
      o = o.transposed(0, 2, 1, 3).squeezed(axis: 0)

      outputs.append(o)
    }

    // Concatenate all windows: (total_tokens, heads, head_dim)
    return MLX.concatenated(outputs, axis: 0)
  }

  /// Splits combined attention output into vid and txt, coalescing text across windows.
  ///
  /// Text tokens appear in every window. We average them across all windows that
  /// contain the same batch element.
  private static func unconcatAndCoalesce(
    combined: MLXArray,
    vidLens: MLXArray,
    txtLens: MLXArray,
    counts: [Int]
  ) -> (MLXArray, MLXArray) {
    // Flatten to (total, heads*head_dim) for output
    let combinedFlat = combined.reshaped(-1, combined.dim(-2) * combined.dim(-1))

    let numWindows = counts.reduce(0, +)

    // Build alternating split indices: [vLen0, tLen0, vLen1, tLen1, ...]
    var splitLens: [Int] = []
    for w in 0 ..< numWindows {
      splitLens.append(Int(vidLens[w].item(Int32.self)))
      splitLens.append(Int(txtLens[w].item(Int32.self)))
    }

    // Compute cumulative split indices
    var splitIndices: [Int] = []
    var cumLen = 0
    for i in 0 ..< (splitLens.count - 1) {
      cumLen += splitLens[i]
      splitIndices.append(cumLen)
    }

    let parts = combinedFlat.split(indices: splitIndices, axis: 0)

    // Even indices are vid, odd are txt
    var vidParts: [MLXArray] = []
    var txtParts: [MLXArray] = []
    for (i, part) in parts.enumerated() {
      if i % 2 == 0 {
        vidParts.append(part)
      } else {
        txtParts.append(part)
      }
    }

    let vidOut = MLX.concatenated(vidParts, axis: 0)

    // Coalesce text: average across windows per batch element
    var coalescedTxt: [MLXArray] = []
    var windowOffset = 0
    for count in counts {
      // Stack the text from `count` windows and take mean
      let windowTxts = Array(txtParts[windowOffset ..< (windowOffset + count)])
      let stacked = MLX.stacked(windowTxts, axis: 0)  // (count, L_txt, D)
      let averaged = stacked.mean(axis: 0)  // (L_txt, D)
      coalescedTxt.append(averaged)
      windowOffset += count
    }

    let txtOut = MLX.concatenated(coalescedTxt, axis: 0)

    return (vidOut, txtOut)
  }

  /// Repeats rows of an array with per-element counts along axis 0.
  private static func perElementRepeat(_ array: MLXArray, counts: [Int]) -> MLXArray {
    var parts: [MLXArray] = []
    for (i, count) in counts.enumerated() {
      let slice = array[i].expandedDimensions(axis: 0)
      parts.append(MLX.repeated(slice, count: count, axis: 0))
    }
    return MLX.concatenated(parts, axis: 0)
  }
}
