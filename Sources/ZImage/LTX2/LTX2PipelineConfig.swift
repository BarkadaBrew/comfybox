// LTX2PipelineConfig.swift -- Pipeline configuration for LTX-2 video generation
// Phase 4 of the LTX-2 Swift/MLX port
//
// Defines configuration for the complete T2V/I2V pipeline, including:
// - Model paths (VAE, text encoder, transformer, upsampler)
// - Pipeline type selection (distilled vs dev)
// - Sampler selection (euler vs res2s)
// - Generation parameters (guidance, fps, quantization)
//
// Distilled pipeline: 8-step fixed sigma schedule, no CFG
// Dev pipeline: configurable steps with dynamic sigma schedule + CFG

import Foundation

/// Pipeline type selection.
public enum LTX2PipelineType: String, Sendable {
  /// Distilled: two-stage with fixed sigmas, no CFG, 8 steps
  case distilled
  /// Dev: single-stage, dynamic sigmas, CFG
  case dev
  /// Dev two-stage: dev at half resolution + distilled LoRA upscale
  case devTwoStage
}

/// Sampler type selection.
public enum LTX2SamplerType: String, Sendable {
  /// First-order Euler ODE solver
  case euler
  /// Second-order Rosenbrock-Runge-Kutta (res_2s) integrator
  case res2s
}

/// Quantization configuration for transformer weights.
public struct LTX2QuantConfig: Sendable {
  /// Number of quantization bits (4 or 8).
  public let bits: Int
  /// Group size for quantization.
  public let groupSize: Int

  public init(bits: Int = 4, groupSize: Int = 64) {
    self.bits = bits
    self.groupSize = groupSize
  }
}

/// Configuration for the LTX-2 video generation pipeline.
public struct LTX2PipelineConfig: Sendable {

  /// Path to the model weights directory.
  public let modelPath: String

  /// Optional path to text encoder weights (if separate from model).
  public let textEncoderPath: String?

  /// Optional path to upsampler weights for two-stage pipeline.
  public let upsamplerPath: String?

  /// Optional path to LoRA weights.
  public let loraPath: String?

  /// LoRA merge strength (default 1.0).
  public let loraStrength: Float

  /// Pipeline type: distilled (fast, 8 steps) or dev (quality, 25+ steps).
  public let pipelineType: LTX2PipelineType

  /// Sampler type: euler (1st order) or res2s (2nd order).
  public let sampler: LTX2SamplerType

  /// CFG guidance scale. Default 3.5 for dev, 1.0 (disabled) for distilled.
  public let guidance: Float

  /// Output frames per second. Default 24.
  public let fps: Int

  /// Optional quantization for transformer weights.
  public let quantization: LTX2QuantConfig?

  /// Whether to use LTX-2.3 prompt AdaLN variant.
  public let hasPromptAdaLN: Bool

  /// Whether to use tiled VAE decoding (for large resolutions).
  public let tiledDecode: Bool

  /// Negative prompt for CFG (dev pipeline only).
  public let negativePrompt: String

  /// Default negative prompt matching the Python reference.
  public static let defaultNegativePrompt = """
    blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, \
    excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted \
    proportions, unnatural skin tones, deformed facial features, asymmetrical face, \
    missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts \
    around text, inconsistent perspective, camera shake, incorrect depth of field, \
    background too sharp, background clutter, distracting reflections, harsh shadows, \
    inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, \
    unrealistic materials, uncanny valley effect
    """

  public init(
    modelPath: String,
    textEncoderPath: String? = nil,
    upsamplerPath: String? = nil,
    loraPath: String? = nil,
    loraStrength: Float = 1.0,
    pipelineType: LTX2PipelineType = .distilled,
    sampler: LTX2SamplerType = .euler,
    guidance: Float? = nil,
    fps: Int = 24,
    quantization: LTX2QuantConfig? = nil,
    hasPromptAdaLN: Bool = false,
    tiledDecode: Bool = false,
    negativePrompt: String? = nil
  ) {
    self.modelPath = modelPath
    self.textEncoderPath = textEncoderPath
    self.upsamplerPath = upsamplerPath
    self.loraPath = loraPath
    self.loraStrength = loraStrength
    self.pipelineType = pipelineType
    self.sampler = sampler
    self.fps = fps
    self.quantization = quantization
    self.hasPromptAdaLN = hasPromptAdaLN
    self.tiledDecode = tiledDecode
    self.negativePrompt = negativePrompt ?? Self.defaultNegativePrompt

    // Default guidance: 1.0 for distilled (no CFG), 3.5 for dev
    if let g = guidance {
      self.guidance = g
    } else {
      switch pipelineType {
      case .distilled: self.guidance = 1.0
      case .dev, .devTwoStage: self.guidance = 3.5
      }
    }
  }

  // MARK: - Distilled Sigma Schedules

  /// Stage 1 sigma schedule for distilled pipeline (8 steps).
  public static let stage1Sigmas: [Float] = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
  ]

  /// Stage 2 sigma schedule for distilled pipeline upscale (3 steps).
  public static let stage2Sigmas: [Float] = [
    0.909375, 0.725, 0.421875, 0.0,
  ]

  // Motion-quality tuning (2026-07-18): the 10Eros/LTX-2.3 authors sample with
  // euler_ancestral + a DMD-specific stage-1 schedule; ComfyBox defaulted to
  // deterministic euler + the front-clustered stock schedule, which under-
  // resolves moving regions (motion haze). These env overrides let the plist
  // drive sampler + schedule per loaded base with NO rebuild. Unset => identical
  // to prior behavior.

  /// Override the distilled stage-1 sigma schedule via env `LTX2_STAGE1_SIGMAS`
  /// (comma-separated floats, e.g. "1.0,0.955,0.893,...,0.0"). nil if unset.
  public static var envStage1Sigmas: [Float]? {
    guard let s = ProcessInfo.processInfo.environment["LTX2_STAGE1_SIGMAS"], !s.isEmpty else { return nil }
    let vals = s.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
    return vals.count >= 2 ? vals : nil
  }

  /// Ancestral (SDE) sampling toggle via env `LTX2_SAMPLER` containing "ancestral".
  public static var envAncestral: Bool {
    (ProcessInfo.processInfo.environment["LTX2_SAMPLER"] ?? "").lowercased().contains("ancestral")
  }

  // MARK: - Dev Sigma Schedule

  /// Scheduling constants for dev pipeline.
  public static let baseShiftAnchor: Float = 1024
  public static let maxShiftAnchor: Float = 4096

  /// Compute dev pipeline sigma schedule with token-dependent shifting.
  ///
  /// - Parameters:
  ///   - steps: Number of inference steps.
  ///   - numTokens: Number of latent tokens (F * H * W). If nil, uses maxShiftAnchor.
  ///   - maxShift: Maximum shift factor. Default 2.05.
  ///   - baseShift: Base shift factor. Default 0.95.
  ///   - stretch: Whether to stretch sigmas to terminal value. Default true.
  ///   - terminal: Terminal sigma value for stretching. Default 0.1.
  /// - Returns: Array of `steps + 1` sigma values.
  public static func devSigmaSchedule(
    steps: Int,
    numTokens: Int? = nil,
    maxShift: Float = 2.05,
    baseShift: Float = 0.95,
    stretch: Bool = true,
    terminal: Float = 0.1
  ) -> [Float] {
    let tokens = Float(numTokens ?? Int(maxShiftAnchor))

    // Linear spacing from 1.0 to 0.0
    var sigmas = (0...steps).map { i in
      1.0 - Float(i) / Float(steps)
    }

    // Compute shift based on token count
    let x1 = baseShiftAnchor
    let x2 = maxShiftAnchor
    let mm = (maxShift - baseShift) / (x2 - x1)
    let b = baseShift - mm * x1
    let sigmaShift = tokens * mm + b

    // Apply shift transformation
    let expShift = exp(sigmaShift)
    for i in 0..<sigmas.count {
      if sigmas[i] != 0 {
        sigmas[i] = expShift / (expShift + pow(1.0 / sigmas[i] - 1.0, 1.0))
      }
    }

    // Stretch sigmas to terminal value
    if stretch {
      var nonZeroIndices: [Int] = []
      for i in 0..<sigmas.count where sigmas[i] != 0 {
        nonZeroIndices.append(i)
      }
      if let lastIdx = nonZeroIndices.last {
        let lastNonZero = sigmas[lastIdx]
        let scaleFactor = (1.0 - lastNonZero) / (1.0 - terminal)
        for i in nonZeroIndices {
          sigmas[i] = 1.0 - (1.0 - sigmas[i]) / scaleFactor
        }
      }
    }

    return sigmas
  }
}
