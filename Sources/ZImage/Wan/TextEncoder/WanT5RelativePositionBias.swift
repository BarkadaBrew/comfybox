import Foundation
import MLX
import MLXNN

/// T5-style relative position bias for the Wan 2.2 UMT5 encoder.
///
/// Computes learned bias values based on the relative distance between
/// query and key positions. Distances are bucketed: exact for small
/// distances, log-scaled for larger ones.
///
/// In the Wan 2.2 checkpoint, every layer has its own position bias
/// (unlike standard T5 where only layer 0 computes it).
///
/// ## Bucketing Scheme
///
/// For bidirectional attention with `numBuckets=32`:
/// - Buckets 0-15: negative relative positions (key before query)
/// - Buckets 16-31: positive relative positions (key after query)
/// - Within each half: exact for |d| < 8, log-scaled for |d| >= 8 up to maxDistance
///
/// ## Weight Shape
///
/// The embedding weight has shape `[numBuckets, numHeads]` = `[32, 64]`,
/// mapping each bucket to a per-head bias value.
public final class WanT5RelativePositionBias: Module {

  /// Learned bias embedding: [numBuckets, numHeads].
  @ModuleInfo(key: "embedding") var embedding: Embedding

  /// Number of relative position buckets.
  public let numBuckets: Int

  /// Maximum distance for bucketing.
  public let maxDistance: Int

  /// Number of attention heads.
  public let numHeads: Int

  /// Creates a relative position bias layer.
  ///
  /// - Parameters:
  ///   - numBuckets: Number of relative position buckets. Default 32.
  ///   - numHeads: Number of attention heads. Default 64.
  ///   - maxDistance: Maximum distance for log bucketing. Default 128.
  public init(numBuckets: Int = 32, numHeads: Int = 64, maxDistance: Int = 128) {
    self.numBuckets = numBuckets
    self.numHeads = numHeads
    self.maxDistance = maxDistance
    self._embedding.wrappedValue = Embedding(embeddingCount: numBuckets, dimensions: numHeads)
    super.init()
  }

  /// Computes the relative position bias matrix.
  ///
  /// - Parameters:
  ///   - queryLength: Number of query positions.
  ///   - keyLength: Number of key positions.
  /// - Returns: Bias tensor of shape `[1, numHeads, queryLength, keyLength]`.
  public func computeBias(queryLength: Int, keyLength: Int) -> MLXArray {
    // Build relative position matrix: key_pos - query_pos
    // Shape: [queryLength, keyLength]
    let queryPositions = MLXArray(Array(0..<Int32(queryLength)))  // [Q]
    let keyPositions = MLXArray(Array(0..<Int32(keyLength)))      // [K]

    // relativePosition[i, j] = j - i
    let relativePosition = keyPositions.reshaped(1, keyLength) - queryPositions.reshaped(queryLength, 1)

    // Convert to bucket indices
    let buckets = Self.relativeBuckets(
      relativePosition: relativePosition,
      numBuckets: numBuckets,
      maxDistance: maxDistance,
      bidirectional: true
    )

    // Look up embeddings: [Q, K] -> [Q, K, numHeads]
    let values = embedding(buckets)

    // Transpose to [numHeads, Q, K] then add batch dim -> [1, numHeads, Q, K]
    let transposed = values.transposed(2, 0, 1)
    return transposed.reshaped(1, numHeads, queryLength, keyLength)
  }

  /// Computes bucket indices for relative positions.
  ///
  /// Implements T5's bucketing: exact for small distances, log-scaled for larger.
  ///
  /// - Parameters:
  ///   - relativePosition: Relative position values (key - query). Any shape.
  ///   - numBuckets: Total number of buckets.
  ///   - maxDistance: Maximum distance for log bucketing.
  ///   - bidirectional: If true, use separate buckets for positive/negative positions.
  /// - Returns: Bucket indices with same shape as input. Values in `[0, numBuckets)`.
  public static func relativeBuckets(
    relativePosition: MLXArray,
    numBuckets: Int,
    maxDistance: Int,
    bidirectional: Bool
  ) -> MLXArray {
    var effectiveBuckets = numBuckets
    var relPos = relativePosition
    var result = MLXArray.zeros(like: relativePosition).asType(.int32)

    if bidirectional {
      effectiveBuckets = numBuckets / 2

      // Positive positions get offset by effectiveBuckets
      let isPositive = MLX.which(relPos .> 0, relPos, MLXArray(Int32(0)))
      let positiveOffset = MLX.which(
        relPos .> 0,
        MLXArray(Int32(effectiveBuckets)),
        MLXArray(Int32(0))
      )
      result = result + positiveOffset

      // Work with absolute values
      relPos = MLX.abs(relPos)
    } else {
      // Clamp to non-positive
      relPos = MLX.minimum(relPos, MLXArray(Int32(0)))
      relPos = MLX.abs(relPos)
    }

    // Exact buckets for small values
    let halfBuckets = effectiveBuckets / 2
    let isSmall = relPos .< Int32(halfBuckets)

    // Log-scaled buckets for larger values
    let logBase = Float(maxDistance) / Float(halfBuckets)
    let relPosFloat = relPos.asType(.float32)
    let halfBucketsFloat = MLXArray(Float(halfBuckets))

    // log(relPos / halfBuckets) / log(maxDistance / halfBuckets) * halfBuckets + halfBuckets
    let logVal = MLX.log(relPosFloat / halfBucketsFloat) / log(logBase)
    let logBuckets = (logVal * halfBucketsFloat + halfBucketsFloat).asType(.int32)
    let clampedLogBuckets = MLX.minimum(logBuckets, MLXArray(Int32(effectiveBuckets - 1)))

    // Select: small positions use exact bucket, large use log bucket
    let bucketContribution = MLX.which(isSmall, relPos.asType(.int32), clampedLogBuckets)

    return result + bucketContribution
  }
}
