import Foundation
import MLX

/// Spatial window partitioner for the SeedVR2 transformer attention mechanism.
///
/// Splits a flat token sequence into windows for local attention, analogous to the
/// Swin Transformer's shifted window strategy. This is NOT a `Module` — it is
/// created fresh for each forward pass based on the current spatial shape.
///
/// ## Key Concepts
///
/// 1. **Adaptive window sizing**: Window dimensions are computed relative to a
///    reference resolution of 45x80 (720p after 2x2 patching).
/// 2. **Forward index**: Reorders tokens from spatial order to window-local order.
/// 3. **Reverse index**: Undoes the reordering after attention.
/// 4. **Shifted windows**: Odd-numbered blocks shift windows by half a window size
///    (like Swin Transformer) to enable cross-window information flow.
/// 5. **Variable edge windows**: Windows near spatial boundaries may be smaller.
///
/// ## Usage
///
/// ```swift
/// let partitioner = SeedVR2WindowPartitioner(shape: vidShape, windowSize: (4, 3, 3), shift: true)
/// let windowed = partitioner.partition(qkvTokens)
/// // ... per-window attention ...
/// let restored = partitioner.reverse(windowedOutput)
/// ```
public struct SeedVR2WindowPartitioner {

  /// Index array that reorders tokens from spatial to window-local order.
  public let forwardIdx: MLXArray

  /// Index array that restores tokens from window-local to spatial order.
  public let reverseIdx: MLXArray

  /// Shape of each window: `(num_windows, 3)` containing `[T, H, W]`.
  public let windowShapes: MLXArray

  /// Number of windows per batch element.
  public let windowCounts: [Int]

  /// Creates a window partitioner for the given spatial layout.
  ///
  /// - Parameters:
  ///   - shape: Spatial shape tensor, `(B, 3)` containing `[T, H, W]` per batch element.
  ///   - windowSize: Number of windows per axis `(nt, nh, nw)`. Default `(4, 3, 3)`.
  ///   - shift: Whether to apply half-window shifting. Default `false`.
  public init(shape: MLXArray, windowSize: (Int, Int, Int), shift: Bool = false) {
    let result = SeedVR2WindowPartitioner.createWindowIndices(
      shape: shape,
      windowSize: windowSize,
      shift: shift
    )
    self.forwardIdx = result.forwardIdx
    self.reverseIdx = result.reverseIdx
    self.windowShapes = result.windowShapes
    self.windowCounts = result.windowCounts
  }

  /// Reorders tokens from spatial order to window-local order.
  ///
  /// - Parameter tensor: Token tensor of shape `(N, ...)`.
  /// - Returns: Reordered tensor of the same shape.
  public func partition(_ tensor: MLXArray) -> MLXArray {
    return tensor[forwardIdx]
  }

  /// Restores tokens from window-local order back to spatial order.
  ///
  /// - Parameter tensor: Windowed tensor of shape `(N, ...)`.
  /// - Returns: Spatially ordered tensor.
  public func reverse(_ tensor: MLXArray) -> MLXArray {
    return tensor[reverseIdx]
  }

  // MARK: - Window Computation

  /// Computes window slices for a single spatial volume.
  ///
  /// The window boundaries are determined by:
  /// 1. Scaling the spatial dims to a reference 45x80 resolution
  /// 2. Computing per-axis window size from num_windows
  /// 3. Optionally shifting by half a window
  /// 4. Generating all (temporal, height, width) slice combinations
  ///
  /// - Parameters:
  ///   - size: Spatial extent `(T, H, W)`.
  ///   - numWindows: Number of windows per axis `(nt, nh, nw)`.
  ///   - shift: Whether to shift windows by half.
  /// - Returns: Array of `(tRange, hRange, wRange)` slice tuples.
  private static func makeWindows(
    size: (Int, Int, Int),
    numWindows: (Int, Int, Int),
    shift: Bool
  ) -> [(tRange: Range<Int>, hRange: Range<Int>, wRange: Range<Int>)] {
    let (t, h, w) = size
    let (resizedNt, resizedNh, resizedNw) = numWindows

    // Scale to reference resolution (45x80)
    let scale = sqrt(Float(45 * 80) / Float(h * w))
    let resizedH = Int(Float(h) * scale + 0.5)  // round
    let resizedW = Int(Float(w) * scale + 0.5)

    // Window sizes (ceil division)
    let wh = ceilDiv(resizedH, resizedNh)
    let ww = ceilDiv(resizedW, resizedNw)
    let wt = ceilDiv(min(t, 30), resizedNt)

    // Shift amounts and window counts
    let st: Float, sh: Float, sw: Float
    let nt: Int, nh: Int, nw: Int

    if shift {
      st = wt < t ? 0.5 : 0
      sh = wh < h ? 0.5 : 0
      sw = ww < w ? 0.5 : 0
      nt = st > 0 ? ceilDiv(t, wt, offset: st) + 1 : 1
      nh = sh > 0 ? ceilDiv(h, wh, offset: sh) + 1 : 1
      nw = sw > 0 ? ceilDiv(w, ww, offset: sw) + 1 : 1
    } else {
      st = 0; sh = 0; sw = 0
      nt = ceilDiv(t, wt)
      nh = ceilDiv(h, wh)
      nw = ceilDiv(w, ww)
    }

    var windows: [(Range<Int>, Range<Int>, Range<Int>)] = []

    for iw in 0 ..< nw {
      let wStart = max(Int((Float(iw) - sw) * Float(ww)), 0)
      let wEnd = min(Int((Float(iw) - sw + 1) * Float(ww)), w)
      guard wEnd > wStart else { continue }

      for ih in 0 ..< nh {
        let hStart = max(Int((Float(ih) - sh) * Float(wh)), 0)
        let hEnd = min(Int((Float(ih) - sh + 1) * Float(wh)), h)
        guard hEnd > hStart else { continue }

        for it in 0 ..< nt {
          let tStart = max(Int((Float(it) - st) * Float(wt)), 0)
          let tEnd = min(Int((Float(it) - st + 1) * Float(wt)), t)
          guard tEnd > tStart else { continue }

          windows.append((tStart ..< tEnd, hStart ..< hEnd, wStart ..< wEnd))
        }
      }
    }

    return windows
  }

  /// Integer ceiling division.
  private static func ceilDiv(_ a: Int, _ b: Int) -> Int {
    return (a + b - 1) / b
  }

  /// Ceiling division with fractional offset: ceil((a - offset) / b).
  private static func ceilDiv(_ a: Int, _ b: Int, offset: Float) -> Int {
    return Int(ceil((Float(a) - offset) / Float(b)))
  }

  /// Unflattens a 2D tensor (N, D) into a list of 3D tensors (T, H, W, D) using shape info.
  private static func unflattenList(
    _ tensor: MLXArray,
    shapes: MLXArray
  ) -> [(array: MLXArray, shape: (Int, Int, Int))] {
    let numBatch = shapes.dim(0)
    var results: [(MLXArray, (Int, Int, Int))] = []
    var offset = 0

    for b in 0 ..< numBatch {
      let t = Int(shapes[b, 0].item(Int32.self))
      let h = Int(shapes[b, 1].item(Int32.self))
      let w = Int(shapes[b, 2].item(Int32.self))
      let length = t * h * w
      let d = tensor.dim(-1)

      let slice = tensor[offset ..< (offset + length)]
      let reshaped = slice.reshaped(t, h, w, d)
      results.append((reshaped, (t, h, w)))
      offset += length
    }

    return results
  }

  /// Creates forward and reverse index arrays for window partitioning.
  ///
  /// This is the core algorithm:
  /// 1. Create a flat index array `[0, 1, ..., totalLen-1]`
  /// 2. Unflatten it using the spatial shapes
  /// 3. Apply window slicing to gather indices per window
  /// 4. The reordered indices form the forward index
  /// 5. argsort of forward gives the reverse index
  private static func createWindowIndices(
    shape: MLXArray,
    windowSize: (Int, Int, Int),
    shift: Bool
  ) -> (forwardIdx: MLXArray, reverseIdx: MLXArray, windowShapes: MLXArray, windowCounts: [Int]) {
    // Total number of tokens across all batch elements
    let totalLen = Int(MLX.sum(shape.product(axis: 1)).item(Int32.self))

    // Create flat index tensor: (totalLen, 1)
    let idx = MLXArray(Int32(0) ..< Int32(totalLen)).reshaped(-1, 1)

    // Unflatten into per-batch 3D volumes
    let volumes = unflattenList(idx, shapes: shape)

    // Partition each volume into windows
    var windowedParts: [MLXArray] = []
    var windowShapeList: [[Int32]] = []
    var windowCounts: [Int] = []

    for (volume, volumeShape) in volumes {
      let slices = makeWindows(size: volumeShape, numWindows: windowSize, shift: shift)
      windowCounts.append(slices.count)

      for (tRange, hRange, wRange) in slices {
        // Extract the window using ranges
        let window = volume[tRange, hRange, wRange]
        let tLen = tRange.count
        let hLen = hRange.count
        let wLen = wRange.count

        // Flatten to (tLen*hLen*wLen, 1)
        windowedParts.append(window.reshaped(-1, 1))
        windowShapeList.append([Int32(tLen), Int32(hLen), Int32(wLen)])
      }
    }

    // Concatenate all windowed indices
    let windowed = MLX.concatenated(windowedParts, axis: 0)
    let targetIdx = windowed.reshaped(-1)

    // Build window shapes array
    let windowShapes = MLXArray(windowShapeList.flatMap { $0 })
      .reshaped(windowShapeList.count, 3)

    // Reverse index = argsort of forward index
    let reverseIdx = argSort(targetIdx)

    return (targetIdx, reverseIdx, windowShapes, windowCounts)
  }
}
