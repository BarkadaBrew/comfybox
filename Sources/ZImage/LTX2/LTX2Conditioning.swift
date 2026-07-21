// LTX2Conditioning.swift -- I2V conditioning for LTX-2 video generation
// Phase 4 of the LTX-2 Swift/MLX port
//
// Handles image conditioning for Image-to-Video mode. An input image is
// encoded via the VAE encoder, injected at frame 0 of the noise latent,
// and a denoise mask controls which frames are denoised vs conditioned.
//
// Reference: conditioning/latent.py — LatentState, apply_conditioning,
//            apply_denoise_mask, VideoConditionByLatentIndex

import MLX
import MLXRandom

// MARK: - Latent State

/// State for latent diffusion with I2V conditioning support.
///
/// Carries the noisy latent alongside a clean conditioning latent and a
/// per-frame mask that controls how much denoising is applied at each frame.
public struct LTX2LatentState {
  /// Current noisy latent `(B, C, F, H, W)`.
  public var latent: MLXArray

  /// Clean conditioning latent `(B, C, F, H, W)`.
  public var cleanLatent: MLXArray

  /// Per-frame denoising mask `(B, 1, F, 1, 1)`.
  /// 1.0 = full denoise, 0.0 = keep clean.
  public var denoiseMask: MLXArray

  /// Face-region latent mask `(1,1,1,H,W)` (1 inside face boxes) + the source
  /// face reference latent `(1,C,1,H,W)` + pull strength. When set, the denoise
  /// loop softly pulls ONLY the masked face latents (all frames) toward the
  /// source face — holds every detected face (esp. a stationary partner) over a
  /// long pass without touching body/motion latents. #partnered
  public var faceMask: MLXArray? = nil
  public var faceRef: MLXArray? = nil
  public var faceAnchorStrength: Float = 0

  public init(latent: MLXArray, cleanLatent: MLXArray, denoiseMask: MLXArray) {
    self.latent = latent
    self.cleanLatent = cleanLatent
    self.denoiseMask = denoiseMask
  }

  /// Create a copy of this state.
  public func copy() -> LTX2LatentState {
    var c = LTX2LatentState(
      latent: latent,
      cleanLatent: cleanLatent,
      denoiseMask: denoiseMask
    )
    c.faceMask = faceMask; c.faceRef = faceRef; c.faceAnchorStrength = faceAnchorStrength
    return c
  }
}

// MARK: - Conditioning Item

/// Condition video generation by injecting an encoded image latent at a frame index.
public struct LTX2VideoCondition {
  /// Encoded image latent of shape `(B, C, 1, H, W)`.
  public let latent: MLXArray
  /// Frame index to condition (0 = first frame).
  public let frameIndex: Int
  /// Denoising strength (1.0 = full denoise, 0.0 = keep original).
  public let strength: Float

  public init(latent: MLXArray, frameIndex: Int = 0, strength: Float = 1.0) {
    self.latent = latent
    self.frameIndex = frameIndex
    self.strength = strength
  }
}

// MARK: - Conditioning Utilities

/// Utilities for I2V latent conditioning.
public enum LTX2Conditioning {

  /// Create initial noisy latent state for T2V (unconditioned).
  ///
  /// - Parameters:
  ///   - shape: Shape of latent `(B, C, F, H, W)`.
  ///   - noiseScale: Scale for initial noise (sigma_max, typically 1.0).
  ///   - seed: Optional random seed.
  /// - Returns: Initial state with random noise and full denoise mask.
  public static func createInitialState(
    shape: [Int],
    noiseScale: Float = 1.0,
    seed: UInt64? = nil
  ) -> LTX2LatentState {
    if let seed = seed {
      MLXRandom.seed(seed)
    }

    let noise = MLXRandom.normal(shape)
    return LTX2LatentState(
      latent: noise * noiseScale,
      cleanLatent: MLXArray.zeros(shape),
      denoiseMask: MLXArray.ones([shape[0], 1, shape[2], 1, 1])
    )
  }

  /// Apply I2V conditioning to a latent state.
  ///
  /// Replaces the latent at the specified frame index with the conditioned
  /// image latent and sets the denoise mask accordingly.
  ///
  /// - Parameters:
  ///   - state: Current latent state.
  ///   - conditions: List of conditioning items to apply.
  /// - Returns: Updated state with conditioning applied.
  public static func applyConditioning(
    state: LTX2LatentState,
    conditions: [LTX2VideoCondition]
  ) -> LTX2LatentState {
    var result = state.copy()
    let f = result.latent.dim(2)

    for cond in conditions {
      let frameIdx = cond.frameIndex
      let strength = cond.strength
      let condLatent = cond.latent
      let condFrames = condLatent.dim(2)
      let endIdx = min(frameIdx + condFrames, f)

      // Build frame-by-frame arrays for the updated state
      var latentSlices: [MLXArray] = []
      var cleanSlices: [MLXArray] = []
      var maskSlices: [MLXArray] = []

      for i in 0..<f {
        if i >= frameIdx && i < endIdx {
          let condIdx = i - frameIdx
          latentSlices.append(condLatent[0..., 0..., condIdx..<(condIdx + 1)])
          cleanSlices.append(condLatent[0..., 0..., condIdx..<(condIdx + 1)])
          // strength=1.0 means full denoise -> mask=0.0 for conditioned frame
          // mask = 1.0 - strength, so strength=1.0 -> mask=0.0 (keep clean)
          let maskValue = MLXArray(1.0 - strength).reshaped(1, 1, 1, 1, 1)
          maskSlices.append(maskValue)
        } else {
          latentSlices.append(result.latent[0..., 0..., i..<(i + 1)])
          cleanSlices.append(result.cleanLatent[0..., 0..., i..<(i + 1)])
          maskSlices.append(result.denoiseMask[0..., 0..., i..<(i + 1)])
        }
      }

      result.latent = MLX.concatenated(latentSlices, axis: 2)
      result.cleanLatent = MLX.concatenated(cleanSlices, axis: 2)
      result.denoiseMask = MLX.concatenated(maskSlices, axis: 2)
    }

    return result
  }

  /// Blend denoised output with clean state based on denoise mask.
  ///
  /// For conditioned frames (mask=0.0), keeps the clean latent.
  /// For unconditioned frames (mask=1.0), uses the denoised output.
  ///
  /// - Parameters:
  ///   - denoised: Denoised latent `(B, C, F, H, W)`.
  ///   - clean: Clean conditioning latent `(B, C, F, H, W)`.
  ///   - denoiseMask: Mask where 1.0 = use denoised, 0.0 = use clean.
  /// - Returns: Blended latent.
  public static func applyDenoiseMask(
    denoised: MLXArray,
    clean: MLXArray,
    denoiseMask: MLXArray
  ) -> MLXArray {
    return denoised * denoiseMask + clean * (1.0 - denoiseMask)
  }

  /// Add noise to state while respecting conditioning.
  ///
  /// For conditioned frames, adds noise proportionally to the denoise mask.
  ///
  /// - Parameters:
  ///   - state: Current latent state.
  ///   - noiseScale: Scale for noise (sigma).
  /// - Returns: Updated state with noise added.
  public static func addNoiseWithState(
    state: LTX2LatentState,
    noiseScale: Float
  ) -> LTX2LatentState {
    var result = state.copy()
    let noise = MLXRandom.normal(result.latent.shape)
    let effectiveScale = MLXArray(noiseScale) * result.denoiseMask
    result.latent = noise * effectiveScale + result.latent * (1.0 - effectiveScale)
    return result
  }
}
