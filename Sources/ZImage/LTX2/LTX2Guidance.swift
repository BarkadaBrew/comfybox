// LTX2Guidance.swift -- Classifier-free guidance for LTX-2 video generation
// Phase 4 of the LTX-2 Swift/MLX port
//
// Implements standard CFG and Adaptive Projected Guidance (APG) for the
// denoising loop. CFG combines conditional and unconditional predictions
// to steer generation toward the prompt. APG decomposes guidance into
// parallel and orthogonal components for more stable I2V generation.
//
// Reference: generate.py functions cfg_delta, apg_delta

import MLX

/// Classifier-free guidance utilities for LTX-2.
public enum LTX2Guidance {

  /// Apply standard classifier-free guidance.
  ///
  /// Formula: guided = cond + (scale - 1) * (cond - uncond)
  ///
  /// This is equivalent to: uncond + scale * (cond - uncond)
  ///
  /// - Parameters:
  ///   - conditioned: Conditional denoised prediction (x0_pos).
  ///   - unconditioned: Unconditional denoised prediction (x0_neg).
  ///   - scale: CFG guidance scale. 1.0 = no guidance.
  /// - Returns: Guided prediction.
  public static func applyCFG(
    conditioned: MLXArray,
    unconditioned: MLXArray,
    scale: Float
  ) -> MLXArray {
    if scale == 1.0 { return conditioned }
    let delta = (scale - 1.0) * (conditioned - unconditioned)
    return conditioned + delta
  }

  /// Compute CFG delta for adding to the positive prediction.
  ///
  /// - Parameters:
  ///   - conditioned: Conditional prediction.
  ///   - unconditioned: Unconditional prediction.
  ///   - scale: Guidance scale.
  /// - Returns: Delta to add to conditioned prediction.
  public static func cfgDelta(
    conditioned: MLXArray,
    unconditioned: MLXArray,
    scale: Float
  ) -> MLXArray {
    return (scale - 1.0) * (conditioned - unconditioned)
  }

  /// Apply Adaptive Projected Guidance (APG).
  ///
  /// Decomposes guidance into parallel and orthogonal components relative to
  /// the conditional prediction, providing more stable guidance for I2V.
  ///
  /// Based on: https://arxiv.org/abs/2407.12173
  ///
  /// - Parameters:
  ///   - conditioned: Conditional prediction (x0_pos).
  ///   - unconditioned: Unconditional prediction (x0_neg).
  ///   - scale: Guidance strength (same as CFG scale).
  ///   - eta: Weight for parallel component (1.0 = keep full parallel).
  ///   - normThreshold: Clamp guidance norm to this value (0 = no clamping).
  /// - Returns: Guided prediction.
  public static func applyAPG(
    conditioned: MLXArray,
    unconditioned: MLXArray,
    scale: Float,
    eta: Float = 1.0,
    normThreshold: Float = 0.0
  ) -> MLXArray {
    if scale == 1.0 { return conditioned }

    var guidance = conditioned - unconditioned

    // Optionally clamp guidance norm for stability
    if normThreshold > 0 {
      let guidanceNorm = MLX.sqrt(
        MLX.sum(guidance * guidance, axes: [-1, -2, -3], keepDims: true) + 1e-8
      )
      let scaleFactor = MLX.minimum(
        MLXArray(Float(1.0)),
        MLXArray(normThreshold) / guidanceNorm
      )
      guidance = guidance * scaleFactor
    }

    let batchSize = conditioned.dim(0)
    let condFlat = conditioned.reshaped(batchSize, -1)
    let guidanceFlat = guidance.reshaped(batchSize, -1)

    // Projection coefficient: (guidance . cond) / (cond . cond)
    let dotProduct = MLX.sum(guidanceFlat * condFlat, axis: 1, keepDims: true)
    let squaredNorm = MLX.sum(condFlat * condFlat, axis: 1, keepDims: true) + 1e-8
    var projCoeff = dotProduct / squaredNorm

    // Reshape for broadcasting
    var expandShape = [batchSize]
    for _ in 1..<conditioned.ndim {
      expandShape.append(1)
    }
    projCoeff = projCoeff.reshaped(expandShape)

    // Parallel and orthogonal components
    let gParallel = projCoeff * conditioned
    let gOrth = guidance - gParallel

    // Combine with eta weighting
    let gAPG = gParallel * eta + gOrth
    let delta = gAPG * (scale - 1.0)

    return conditioned + delta
  }

  /// Apply CFG rescale to reduce over-saturation.
  ///
  /// Normalizes guided prediction variance relative to conditional prediction.
  /// factor = rescale * (cond_std / pred_std) + (1 - rescale)
  ///
  /// - Parameters:
  ///   - guided: Guided prediction.
  ///   - conditioned: Original conditional prediction.
  ///   - rescale: Rescale factor (0.0 - 1.0). 0.0 = no rescale.
  /// - Returns: Rescaled prediction.
  public static func cfgRescale(
    guided: MLXArray,
    conditioned: MLXArray,
    rescale: Float
  ) -> MLXArray {
    guard rescale > 0 else { return guided }
    // Compute std manually: sqrt(var)
    let condF32 = conditioned.asType(.float32)
    let guidedF32 = guided.asType(.float32)
    let condVar = condF32.variance()
    let guidedVar = guidedF32.variance()
    let condStd = MLX.sqrt(condVar)
    let guidedStd = MLX.sqrt(guidedVar) + 1e-8
    let vFactor = MLXArray(rescale) * (condStd / guidedStd) + MLXArray(1.0 - rescale)
    return guided * vFactor
  }
}
