// LTX2RoPE.swift — 3D Rotary Position Embedding (SPLIT mode)
// Phase 3 of the LTX-2 Swift/MLX port
//
// LTX-2 uses SPLIT RoPE for the main transformer (not INTERLEAVED like some
// other DiT models). The 3D positions (time, height, width) each get their own
// frequency band, and frequencies are concatenated.
//
// SPLIT mode:
//   first_half  = x1 * cos - x2 * sin
//   second_half = x2 * cos + x1 * sin
//   output = [first_half, second_half]
//
// The frequency computation uses log-spaced indices from 1.0 to theta,
// scaled by pi/2, then applied to fractional positions in [-1, 1].
//
// Reference: rope.py — precompute_freqs_cis, apply_split_rotary_emb, split_freqs_cis

import Foundation
import MLX

// MARK: - RoPE Application

/// Apply SPLIT rotary position embeddings to an input tensor.
///
/// Splits the last dimension in two halves and applies rotation:
///   out1 = x1 * cos - x2 * sin
///   out2 = x2 * cos + x1 * sin
///
/// - Parameters:
///   - input: Tensor, either `(B, T, H*D)` (flat) or `(B, H, T, D)` (per-head).
///   - cos: Cosine frequencies `(B, H, T, D//2)`.
///   - sin: Sine frequencies `(B, H, T, D//2)`.
/// - Returns: Tensor with SPLIT rotary embeddings applied, same shape as input.
public func ltx2ApplySplitRoPE(
  _ input: MLXArray,
  cos cosFreqs: MLXArray,
  sin sinFreqs: MLXArray
) -> MLXArray {
  let inputDtype = input.dtype
  var x = input
  var needsReshape = false

  // Handle flat format: (B, T, H*D) -> (B, H, T, D)
  if x.ndim != 4 && cosFreqs.ndim == 4 {
    let b = cosFreqs.dim(0)
    let h = cosFreqs.dim(1)
    let t = cosFreqs.dim(2)
    x = x.reshaped(b, t, h, -1).transposed(0, 2, 1, 3)
    needsReshape = true
  }

  let xF = x.asType(.float32)
  let cosF = cosFreqs.asType(.float32)
  let sinF = sinFreqs.asType(.float32)

  let halfDim = x.dim(-1) / 2
  let x1 = xF[.ellipsis, ..<halfDim]
  let x2 = xF[.ellipsis, halfDim...]

  let out1 = x1 * cosF - x2 * sinF
  let out2 = x2 * cosF + x1 * sinF

  var output = MLX.concatenated([out1, out2], axis: -1)

  if needsReshape {
    // (B, H, T, D) -> (B, T, H*D)
    let b = output.dim(0)
    let h = output.dim(1)
    let t = output.dim(2)
    output = output.transposed(0, 2, 1, 3).reshaped(b, t, h * output.dim(3))
  }

  return output.asType(inputDtype)
}

/// Apply INTERLEAVED rotary position embeddings to an input tensor.
///
/// Pairs adjacent dimensions: [x0, x1, x2, x3, ...] -> rotate pairs (x0,x1), (x2,x3), ...
///
/// - Parameters:
///   - input: Tensor `(B, T, dim)` or similar.
///   - cos: Cosine frequencies `(B, T, dim)`.
///   - sin: Sine frequencies `(B, T, dim)`.
/// - Returns: Tensor with interleaved rotary embeddings applied.
public func ltx2ApplyInterleavedRoPE(
  _ input: MLXArray,
  cos cosFreqs: MLXArray,
  sin sinFreqs: MLXArray
) -> MLXArray {
  let inputDtype = input.dtype
  let xF = input.asType(.float32)
  let cosF = cosFreqs.asType(.float32)
  let sinF = sinFreqs.asType(.float32)

  // Reshape to pairs: (..., dim) -> (..., dim/2, 2)
  let shape = xF.shape
  let lastDim = shape[shape.count - 1]
  var pairedShape = Array(shape.dropLast())
  pairedShape.append(contentsOf: [lastDim / 2, 2])
  let paired = xF.reshaped(pairedShape)

  // Extract even (index 0) and odd (index 1) from the pair axis
  // Use split to get the two halves of the pair dimension
  let splits = paired.split(parts: 2, axis: paired.ndim - 1)
  let t1 = splits[0].squeezed(axis: -1)  // Even indices: (..., dim/2)
  let t2 = splits[1].squeezed(axis: -1)  // Odd indices: (..., dim/2)

  // Build rotated: [-t2, t1] interleaved back to full dim
  let negT2 = MLXArray(Float(0)) - t2
  let rotatedPairs = MLX.stacked([negT2, t1], axis: -1)  // (..., dim/2, 2)
  let rotated = rotatedPairs.reshaped(shape)

  let output = xF * cosF + rotated * sinF
  return output.asType(inputDtype)
}

/// Dispatch RoPE application based on mode.
///
/// - Parameters:
///   - input: Input tensor.
///   - freqsCIS: Tuple of `(cos, sin)` frequency tensors.
///   - mode: Either `.split` or `.interleaved`.
/// - Returns: Tensor with rotary embeddings applied.
public func ltx2ApplyRoPE(
  _ input: MLXArray,
  freqsCIS: (cos: MLXArray, sin: MLXArray),
  mode: LTX2RoPEMode
) -> MLXArray {
  switch mode {
  case .split:
    return ltx2ApplySplitRoPE(input, cos: freqsCIS.cos, sin: freqsCIS.sin)
  case .interleaved:
    return ltx2ApplyInterleavedRoPE(input, cos: freqsCIS.cos, sin: freqsCIS.sin)
  }
}

// MARK: - RoPE Mode

/// RoPE mode selection.
public enum LTX2RoPEMode: String, Sendable {
  case split
  case interleaved
}

// MARK: - Frequency Precomputation

/// Precompute RoPE cos/sin frequencies from a 3D position grid.
///
/// Steps:
/// 1. Generate log-spaced frequency indices from 1.0 to theta, scaled by pi/2
/// 2. Compute fractional positions scaled to [-1, 1]
/// 3. Outer product of positions and frequencies
/// 4. Compute cos/sin, pad if needed, reshape to per-head layout
///
/// - Parameters:
///   - indicesGrid: Position grid `(B, nDims, T, ...)`.
///   - dim: Inner dimension (e.g. 4096).
///   - theta: Base frequency (e.g. 10000.0).
///   - maxPos: Maximum position per dimension (e.g. [20, 2048, 2048]).
///   - useMiddleIndicesGrid: Whether position grid has start/end pairs.
///   - numAttentionHeads: Number of attention heads (e.g. 32).
///   - ropeMode: SPLIT or INTERLEAVED.
/// - Returns: `(cos, sin)` frequency tensors.
public func ltx2PrecomputeFreqsCIS(
  indicesGrid: MLXArray,
  dim: Int,
  theta: Float = 10000.0,
  maxPos: [Int] = [20, 2048, 2048],
  useMiddleIndicesGrid: Bool = true,
  numAttentionHeads: Int = 32,
  ropeMode: LTX2RoPEMode = .split,
  doublePrecision: Bool = false
) -> (cos: MLXArray, sin: MLXArray) {
  // Keep positions in float32 for precision
  let gridF32 = indicesGrid.asType(.float32)

  let nPosDims = gridF32.dim(1)
  let nElem = 2 * nPosDims

  // Generate log-spaced frequency indices
  var numIndices = dim / nElem
  if numIndices == 0 { numIndices = 1 }

  let freqIndices = ltx2GenerateFreqGrid(
    theta: theta,
    maxPosCount: nPosDims,
    innerDim: dim,
    doublePrecision: doublePrecision
  )

  // Generate frequencies from positions
  let freqs = ltx2GenerateFreqs(
    indices: freqIndices,
    indicesGrid: gridF32,
    maxPos: maxPos,
    useMiddleIndicesGrid: useMiddleIndicesGrid
  )

  // Compute cos/sin based on rope mode
  switch ropeMode {
  case .split:
    let expectedFreqs = dim / 2
    let currentFreqs = freqs.dim(-1)
    let padSize = expectedFreqs - currentFreqs
    return ltx2SplitFreqsCIS(freqs: freqs, padSize: padSize, numAttentionHeads: numAttentionHeads)

  case .interleaved:
    let padSize = dim % nElem
    return ltx2InterleavedFreqsCIS(freqs: freqs, padSize: padSize)
  }
}

// MARK: - Frequency Grid Generation

/// Generate log-spaced frequency indices.
///
/// Computes: theta^linspace(0, 1, numIndices) * pi/2
///
/// - Parameters:
///   - theta: Base theta value (e.g. 10000.0).
///   - maxPosCount: Number of position dimensions.
///   - innerDim: Inner dimension of the model.
/// - Returns: Frequency indices tensor `(numIndices,)`.
func ltx2GenerateFreqGrid(
  theta: Float,
  maxPosCount: Int,
  innerDim: Int,
  doublePrecision: Bool = false
) -> MLXArray {
  let nElem = 2 * maxPosCount
  var numIndices = innerDim / nElem
  if numIndices == 0 { numIndices = 1 }

  if doublePrecision {
    // LTX-2.3 uses float64 for the critical frequency grid computation.
    // This matches PyTorch's generate_freq_grid_np which uses numpy float64
    // for the log-spaced values before converting to float32. The high
    // frequencies (up to ~15708) need the extra precision to avoid sign
    // flips in cos/sin that destroy temporal coherence.
    let thetaD = Double(theta)
    let logStart = Foundation.log(1.0) / Foundation.log(thetaD)
    let logEnd = 1.0  // log(theta)/log(theta) = 1.0

    var powValues = [Float](repeating: 0, count: numIndices)
    for i in 0..<numIndices {
      let t = logStart + (logEnd - logStart) * Double(i) / Double(max(numIndices - 1, 1))
      let powVal = Foundation.pow(thetaD, t)
      powValues[i] = Float(powVal * (Double.pi / 2.0))
    }
    return MLXArray(powValues)
  } else {
    // Standard float32 path
    let logStart = Foundation.log(1.0) / Foundation.log(theta)
    let logEnd = Foundation.log(theta) / Foundation.log(theta)  // = 1.0

    let linSpace = MLXArray.linspace(Float(logStart), Float(logEnd), count: numIndices)
    let powIndices = MLXArray(theta).pow(linSpace)

    return powIndices * Float(Float.pi / 2.0)
  }
}

/// Generate frequencies from position indices and frequency grid.
///
/// Computes fractional positions, scales to [-1, 1], then outer products
/// with frequency indices.
///
/// - Parameters:
///   - indices: Frequency indices `(numIndices,)`.
///   - indicesGrid: Position grid `(B, nDims, T, ...)`.
///   - maxPos: Maximum position per dimension.
///   - useMiddleIndicesGrid: Whether to average start/end pairs.
/// - Returns: Frequency tensor `(B, T, numIndices * nDims)`.
func ltx2GenerateFreqs(
  indices: MLXArray,
  indicesGrid: MLXArray,
  maxPos: [Int],
  useMiddleIndicesGrid: Bool
) -> MLXArray {
  var grid = indicesGrid

  // Handle middle indices grid
  if useMiddleIndicesGrid {
    // grid: (B, nDims, T, 2)
    let gridStart = grid[.ellipsis, 0]
    let gridEnd = grid[.ellipsis, 1]
    grid = (gridStart + gridEnd) / 2.0
  } else if grid.ndim == 4 {
    grid = grid[.ellipsis, 0]
  }
  // grid: (B, nDims, T)

  let nPosDims = grid.dim(1)

  // Fractional positions: divide each dim by its max_pos
  var fractionalList: [MLXArray] = []
  for i in 0..<nPosDims {
    let frac = grid[0..., i] / Float(maxPos[i])  // (B, T)
    fractionalList.append(frac)
  }
  // Stack: (B, T, nDims)
  let fractionalPositions = MLX.stacked(fractionalList, axis: -1)

  // Scale to [-1, 1]
  let scaledPositions = fractionalPositions * 2.0 - 1.0

  // Outer product: (B, T, nDims, 1) * (1, 1, 1, numIndices) -> (B, T, nDims, numIndices)
  let freqs = scaledPositions.expandedDimensions(axis: -1) *
    indices.reshaped(1, 1, 1, -1)

  // Transpose and flatten: (B, T, nDims, numIndices) -> (B, T, numIndices, nDims) -> (B, T, numIndices*nDims)
  let freqsT = freqs.transposed(0, 1, 3, 2)
  return freqsT.reshaped(freqsT.dim(0), freqsT.dim(1), -1)
}

// MARK: - Split Frequency Preparation

/// Prepare cos/sin frequencies for SPLIT RoPE.
///
/// Computes cos/sin, pads if needed, reshapes to per-head layout.
///
/// - Parameters:
///   - freqs: Frequency tensor `(B, T, freqDim)`.
///   - padSize: Number of padding elements at the front.
///   - numAttentionHeads: Number of attention heads.
/// - Returns: `(cos, sin)` each shaped `(B, H, T, D//2)`.
func ltx2SplitFreqsCIS(
  freqs: MLXArray,
  padSize: Int,
  numAttentionHeads: Int
) -> (cos: MLXArray, sin: MLXArray) {
  var cosFreq = freqs.cos()
  var sinFreq = freqs.sin()

  // Pad at the front if needed
  if padSize > 0 {
    let cosPad = MLX.ones(like: cosFreq[.ellipsis, ..<padSize])
    let sinPad = MLX.zeros(like: sinFreq[.ellipsis, ..<padSize])
    cosFreq = MLX.concatenated([cosPad, cosFreq], axis: -1)
    sinFreq = MLX.concatenated([sinPad, sinFreq], axis: -1)
  }

  // Reshape for multi-head: (B, T, dim//2) -> (B, T, H, dim//2//H) -> (B, H, T, dim//2//H)
  let b = cosFreq.dim(0)
  let t = cosFreq.dim(1)

  cosFreq = cosFreq.reshaped(b, t, numAttentionHeads, -1).transposed(0, 2, 1, 3)
  sinFreq = sinFreq.reshaped(b, t, numAttentionHeads, -1).transposed(0, 2, 1, 3)

  return (cosFreq, sinFreq)
}

// MARK: - Interleaved Frequency Preparation

/// Prepare cos/sin frequencies for INTERLEAVED RoPE.
///
/// Repeats each frequency value twice and pads if needed.
///
/// - Parameters:
///   - freqs: Frequency tensor `(B, T, dim//2)`.
///   - padSize: Padding size for dimension alignment.
/// - Returns: `(cos, sin)` each shaped `(B, T, dim)`.
func ltx2InterleavedFreqsCIS(
  freqs: MLXArray,
  padSize: Int
) -> (cos: MLXArray, sin: MLXArray) {
  // Compute cos/sin then repeat each value twice
  var cosFreq = MLX.repeated(freqs.cos(), count: 2, axis: -1)
  var sinFreq = MLX.repeated(freqs.sin(), count: 2, axis: -1)

  // Pad at the front if needed
  if padSize > 0 {
    let cosPad = MLX.ones(like: cosFreq[.ellipsis, ..<padSize])
    let sinPad = MLX.zeros(like: sinFreq[.ellipsis, ..<padSize])
    cosFreq = MLX.concatenated([cosPad, cosFreq], axis: -1)
    sinFreq = MLX.concatenated([sinPad, sinFreq], axis: -1)
  }

  return (cosFreq, sinFreq)
}
