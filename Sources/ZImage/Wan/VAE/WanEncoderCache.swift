import Foundation
import MLX

/// Feature cache for chunk-by-chunk VAE encoding.
///
/// The Wan 2.1 VAE encoder processes video frames in temporal chunks
/// (1 frame, then 4 frames) to achieve correct temporal downsampling.
/// Each CausalConv3d and downsample3d Resample layer needs to cache
/// state between chunks.
///
/// The cache is indexed by layer position (matching Python's feat_idx).
public final class WanEncoderCache {
  /// Per-layer cached feature maps. nil = first use.
  public var slots: [MLXArray?]
  /// Current layer index (mutable, reset per chunk).
  public var idx: Int

  public init(layerCount: Int) {
    self.slots = Array(repeating: nil, count: layerCount)
    self.idx = 0
  }

  /// Advances to next slot and returns the current index.
  @discardableResult
  public func advance() -> Int {
    let current = idx
    idx += 1
    return current
  }
}
