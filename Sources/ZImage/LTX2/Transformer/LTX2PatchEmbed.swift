// LTX2PatchEmbed.swift — Linear patch embedding and position grid computation
// Phase 3 of the LTX-2 Swift/MLX port
//
// The patchify projection is a simple Linear(in_channels, inner_dim) that
// projects flattened spatial tokens into the transformer's hidden dimension.
//
// Position grid computation creates 3D pixel-space coordinates [time, height, width]
// for each token, used by the RoPE system.
//
// Reference: ltx_2.py patchify_proj (Linear) and position grid setup

import Foundation
import MLX

/// Linear projection from latent channels to inner dimension.
///
/// Also provides static utilities for computing the 3D position grid
/// required by the RoPE system.
///
/// Note: The patchify projection is a plain Linear layer on LTX2Transformer.
/// This class provides static utilities for position grid computation.
/// It is not a Module for weight loading purposes.
public enum LTX2PatchEmbed {

  // MARK: - Position Grid

  /// Compute a 3D position index grid for RoPE.
  ///
  /// Given the spatial dimensions of the latent tensor (frames, height, width),
  /// creates a position grid where each token gets a 3D coordinate.
  ///
  /// For `useMiddleIndicesGrid = true` (LTX-2 default), each position is
  /// represented as a (start, end) pair, and the RoPE system takes their midpoint.
  ///
  /// - Parameters:
  ///   - frames: Number of temporal frames in the latent.
  ///   - height: Spatial height of the latent.
  ///   - width: Spatial width of the latent.
  ///   - batchSize: Batch size.
  ///   - useMiddleIndicesGrid: Whether to use start/end pairs. Default true.
  /// - Returns: Position grid:
  ///   - If `useMiddleIndicesGrid`: `(B, 3, F*H*W, 2)` where last dim is [start, end].
  ///   - Otherwise: `(B, 3, F*H*W, 1)`.
  public static func makePositionGrid(
    frames: Int,
    height: Int,
    width: Int,
    batchSize: Int,
    useMiddleIndicesGrid: Bool = true
  ) -> MLXArray {
    let numTokens = frames * height * width

    if useMiddleIndicesGrid {
      // Create (start, end) pairs for each position dimension
      // For each token at (f, h, w), the start is the index and end is index + 1
      var positions = [Float](repeating: 0, count: batchSize * 3 * numTokens * 2)

      for b in 0..<batchSize {
        for f in 0..<frames {
          for h in 0..<height {
            for w in 0..<width {
              let tokenIdx = f * height * width + h * width + w
              let baseIdx = b * 3 * numTokens * 2

              // Time dimension (axis 0)
              positions[baseIdx + 0 * numTokens * 2 + tokenIdx * 2 + 0] = Float(f)
              positions[baseIdx + 0 * numTokens * 2 + tokenIdx * 2 + 1] = Float(f + 1)

              // Height dimension (axis 1)
              positions[baseIdx + 1 * numTokens * 2 + tokenIdx * 2 + 0] = Float(h)
              positions[baseIdx + 1 * numTokens * 2 + tokenIdx * 2 + 1] = Float(h + 1)

              // Width dimension (axis 2)
              positions[baseIdx + 2 * numTokens * 2 + tokenIdx * 2 + 0] = Float(w)
              positions[baseIdx + 2 * numTokens * 2 + tokenIdx * 2 + 1] = Float(w + 1)
            }
          }
        }
      }

      return MLXArray(positions, [batchSize, 3, numTokens, 2])
    } else {
      // Simple integer positions
      var positions = [Float](repeating: 0, count: batchSize * 3 * numTokens)

      for b in 0..<batchSize {
        for f in 0..<frames {
          for h in 0..<height {
            for w in 0..<width {
              let tokenIdx = f * height * width + h * width + w
              let baseIdx = b * 3 * numTokens

              positions[baseIdx + 0 * numTokens + tokenIdx] = Float(f)
              positions[baseIdx + 1 * numTokens + tokenIdx] = Float(h)
              positions[baseIdx + 2 * numTokens + tokenIdx] = Float(w)
            }
          }
        }
      }

      return MLXArray(positions, [batchSize, 3, numTokens, 1])
    }
  }
}
