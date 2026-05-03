// FiboTextProjection.swift — DimFusion caption projection (2048 → 1536)
// Ported from mflux: text_projection.py (BriaFiboTextProjection)
//
// Each of the 46 transformer blocks (8 joint + 38 single) has a corresponding
// caption_projection entry. These project per-layer text encoder hidden states
// (2048-dim from SmolLM3-3B) down to 1536-dim, which then gets concatenated
// with the first half of the context embedder output to form the full 3072-dim
// DimFusion conditioning signal.

import Foundation
import MLX
import MLXNN

/// DimFusion text projection layer.
///
/// Projects a text encoder hidden state from `inFeatures` (2048) to `hiddenSize` (1536).
/// Used in FIBO's per-layer text conditioning (DimFusion), where each transformer
/// block receives its own projected text encoder layer output.
///
/// Weight key path: `caption_projection.{i}.linear.weight`
final class FiboTextProjection: Module {
  @ModuleInfo(key: "linear") var linear: Linear

  init(inFeatures: Int = 2048, hiddenSize: Int = 1536) {
    self._linear.wrappedValue = Linear(inFeatures, hiddenSize, bias: false)
    super.init()
  }

  func callAsFunction(_ caption: MLXArray) -> MLXArray {
    linear(caption)
  }
}
