// FiboPipeline.swift — Pipeline orchestration for FIBO image generation
// Ported from mflux: fibo.py (FIBO.generate_image)
//
// FIBO generation flow:
// 1. Encode prompts via FiboPromptEncoder (SmolLM3-3B, all 37 hidden states)
// 2. Create noise latents: (B, 48, H/16, W/16), pack to (B, H/16*W/16, 48)
// 3. Compute linear sigma schedule (no shift, unlike Flux)
// 4. Denoise loop: transformer predict + CFG + Euler step
// 5. Unpack latents to (B, 48, H/16, W/16)
// 6. VAE decode to image
// 7. Save PNG

import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
#endif

// MARK: - Generation Request

/// Configuration for a FIBO generation request.
public struct FiboGenerationRequest: Sendable {
  public var prompt: String
  public var negativePrompt: String?
  public var width: Int
  public var height: Int
  public var steps: Int
  public var guidanceScale: Float
  public var seed: UInt64?
  public var outputPath: URL
  public var levelsMin: Float
  public var levelsMax: Float

  /// Fruit mode (neutral | banana | avocado) — stamped into render metadata.
  public var contentMode: String?

  public init(
    prompt: String,
    negativePrompt: String? = nil,
    width: Int = 1024,
    height: Int = 1024,
    steps: Int = 30,
    guidanceScale: Float = 4.0,
    seed: UInt64? = nil,
    outputPath: URL = URL(fileURLWithPath: "fibo-output.png"),
    levelsMin: Float = 0.0,
    levelsMax: Float = 1.0,
    contentMode: String? = nil
  ) {
    self.prompt = prompt
    self.negativePrompt = negativePrompt
    self.width = width
    self.height = height
    self.steps = steps
    self.guidanceScale = guidanceScale
    self.seed = seed
    self.outputPath = outputPath
    self.levelsMin = levelsMin
    self.levelsMax = levelsMax
    self.contentMode = contentMode
  }
}

// MARK: - FiboPipeline

/// Orchestrates FIBO image generation.
///
/// ## Pipeline Stages
///
/// 1. Load model components via `FiboInitializer`
/// 2. Encode prompt via `FiboPromptEncoder` (SmolLM3-3B, all 37 hidden states)
/// 3. Create noise latents in FIBO format: (B, 48, H/16, W/16), packed to (B, seqLen, 48)
/// 4. Run linear flow-matching denoising loop with CFG
/// 5. Unpack latents, denormalize, and decode via `FiboVAE`
/// 6. Save output image
///
/// Key differences from Flux2Pipeline:
/// - 48 latent channels (not 128 packed)
/// - DimFusion: per-layer text encoder hidden states fed to transformer blocks
/// - Linear sigma schedule (no empirical mu shift)
/// - CFG with negative prompts (guidance > 1.0 always applies)
/// - No batch norm normalization
public final class FiboPipeline {

  public enum FiboPipelineError: Error, LocalizedError {
    case modelNotLoaded
    case invalidDimensions(String)
    case generationFailed(String)

    public var errorDescription: String? {
      switch self {
      case .modelNotLoaded: return "FIBO model not loaded"
      case .invalidDimensions(let msg): return "Invalid dimensions: \(msg)"
      case .generationFailed(let msg): return "Generation failed: \(msg)"
      }
    }
  }

  /// Progress reporting for FIBO generation.
  public struct GenerationProgress: Sendable {
    public let stage: Stage
    public let stepIndex: Int
    public let totalSteps: Int

    public enum Stage: String, Sendable {
      case loadingModel = "Loading model"
      case encodingText = "Encoding text"
      case denoising = "Denoising"
      case decoding = "Decoding"
      case saving = "Saving"
    }

    public var fractionCompleted: Double {
      guard totalSteps > 0 else { return 0 }
      return Double(stepIndex) / Double(totalSteps)
    }
  }

  public typealias ProgressHandler = (GenerationProgress) -> Void

  private var logger: Logger
  private var components: FiboComponents?
  private var modelSnapshot: URL?
  private var isLoaded: Bool = false

  public init(logger: Logger = Logger(label: "z-image.fibo-pipeline")) {
    self.logger = logger
  }

  /// Whether the model is currently loaded.
  public var loaded: Bool { isLoaded }

  /// Unload model components and free memory.
  public func unload() {
    components = nil
    modelSnapshot = nil
    isLoaded = false
    GPU.clearCache()
    logger.info("FIBO model unloaded")
  }

  /// Load the FIBO model from a snapshot directory.
  ///
  /// - Parameters:
  ///   - snapshot: Root URL of the model snapshot directory.
  ///   - transformerConfig: Transformer architecture config.
  ///   - vaeConfig: VAE architecture config.
  ///   - textEncoderConfig: Text encoder architecture config.
  ///   - progressHandler: Optional progress callback.
  public func loadModel(
    from snapshot: URL,
    transformerConfig: FiboTransformerConfig = FiboTransformerConfig(),
    vaeConfig: FiboVAEConfig = FiboVAEConfig(),
    textEncoderConfig: FiboTextEncoderConfig = FiboTextEncoderConfig(),
    progressHandler: ProgressHandler? = nil
  ) throws {
    if isLoaded && modelSnapshot == snapshot { return }

    progressHandler?(GenerationProgress(stage: .loadingModel, stepIndex: 0, totalSteps: 1))

    self.components = try FiboInitializer.load(
      from: snapshot,
      transformerConfig: transformerConfig,
      vaeConfig: vaeConfig,
      textEncoderConfig: textEncoderConfig,
      dtype: .bfloat16,
      logger: logger
    )

    self.modelSnapshot = snapshot
    self.isLoaded = true

    progressHandler?(GenerationProgress(stage: .loadingModel, stepIndex: 1, totalSteps: 1))
    logger.info("FIBO model loaded from \(snapshot.path)")
  }

  /// Generate an image and save it to disk.
  ///
  /// - Parameters:
  ///   - request: Generation parameters.
  ///   - progressHandler: Optional progress callback.
  /// - Returns: The output file URL.
  public func generate(
    _ request: FiboGenerationRequest,
    progressHandler: ProgressHandler? = nil
  ) async throws -> URL {
    let decoded = try await generateCore(request, progressHandler: progressHandler)

    progressHandler?(GenerationProgress(stage: .saving, stepIndex: request.steps, totalSteps: request.steps))
    try QwenImageIO.saveImage(array: decoded, to: request.outputPath,
      metadata: .generation(prompt: request.prompt, negativePrompt: request.negativePrompt,
        seed: request.seed, steps: request.steps, guidance: request.guidanceScale,
        width: request.width, height: request.height, contentMode: request.contentMode))
    logger.info("Wrote image to \(request.outputPath.path)")

    return request.outputPath
  }

  /// Generate an image and return raw PNG data.
  public func generateToMemory(
    _ request: FiboGenerationRequest,
    progressHandler: ProgressHandler? = nil
  ) async throws -> Data {
    let decoded = try await generateCore(request, progressHandler: progressHandler)

    progressHandler?(GenerationProgress(stage: .saving, stepIndex: request.steps, totalSteps: request.steps))
    let imageData = try QwenImageIO.imageData(from: decoded)
    logger.info("Generated image data (\(imageData.count) bytes)")

    return imageData
  }

  // MARK: - Core Generation

  private func generateCore(
    _ request: FiboGenerationRequest,
    progressHandler: ProgressHandler? = nil
  ) async throws -> MLXArray {
    guard let components = components else {
      throw FiboPipelineError.modelNotLoaded
    }

    // Validate dimensions — FIBO VAE needs multiples of 16
    let vaeScale = 16
    if request.width % vaeScale != 0 {
      throw FiboPipelineError.invalidDimensions(
        "Width must be divisible by \(vaeScale) (got \(request.width))"
      )
    }
    if request.height % vaeScale != 0 {
      throw FiboPipelineError.invalidDimensions(
        "Height must be divisible by \(vaeScale) (got \(request.height))"
      )
    }

    // 1. Encode prompts
    progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: request.steps))
    logger.info("Encoding prompt...")

    let promptOutput = FiboPromptEncoder.encodePrompt(
      prompt: request.prompt,
      negativePrompt: request.negativePrompt,
      tokenizer: components.tokenizer,
      textEncoder: components.textEncoder
    )

    let encoderHiddenStates = promptOutput.encoderHiddenStates
    let textEncoderLayers = promptOutput.promptLayers

    MLX.eval(encoderHiddenStates)
    for layer in textEncoderLayers {
      MLX.eval(layer)
    }
    logger.info("Text encoding complete (encoder_hidden_states: \(encoderHiddenStates.shape), \(textEncoderLayers.count) layers)")

    // 2. Create noise latents
    let seed = request.seed ?? UInt64.random(in: 0...UInt64.max)
    let latentHeight = request.height / vaeScale
    let latentWidth = request.width / vaeScale

    // Create noise in spatial format: (B, 48, H/16, W/16)
    // Use explicit key (not global seed) to match Python mflux's mx.random.key(seed)
    let noiseLatents = MLXRandom.normal([1, 48, latentHeight, latentWidth], key: MLXRandom.key(seed))

    // Pack to sequence format: (B, H/16*W/16, 48)
    // Transpose from (B, C, H, W) to (B, H, W, C) then reshape
    var latents = noiseLatents
      .transposed(0, 2, 3, 1)
      .reshaped(1, latentHeight * latentWidth, 48)

    MLX.eval(latents)
    logger.info("Created noise latents: \(latents.shape) (seed: \(seed))")

    // 3. Compute linear sigma schedule (matching Python mflux LinearScheduler for FIBO)
    // FIBO uses requires_sigma_shift=False → simple linear spacing, NOT FMED
    let numSteps = request.steps
    let sigmas = FiboPipeline.computeLinearSigmas(numSteps: numSteps)
    // Timesteps are integer indices [0, 1, ..., N-1] for LinearScheduler
    let timesteps = (0..<numSteps).map { Float($0) }

    logger.info("Linear schedule: \(numSteps) steps, sigma range [\(sigmas.first ?? 0)...\(sigmas.last ?? 0)]")

    // 4. Denoising loop
    logger.info("Running \(numSteps) denoising steps...")
    for stepIndex in 0..<numSteps {
      try Task.checkCancellation()
      progressHandler?(GenerationProgress(stage: .denoising, stepIndex: stepIndex, totalSteps: numSteps))

      // Forward pass through transformer
      let noise = components.transformer(
        hiddenStates: latents,
        encoderHiddenStates: encoderHiddenStates,
        timestep: MLXArray(timesteps[stepIndex]),
        textEncoderLayers: textEncoderLayers,
        height: request.height,
        width: request.width
      )

      // Apply classifier-free guidance
      let guidedNoise = FiboPipeline.applyClassifierFreeGuidance(
        noise: noise,
        guidance: request.guidanceScale
      )

      // Euler step: latents = latents + noise * dt
      let dt = MLXArray(sigmas[stepIndex + 1] - sigmas[stepIndex]).asType(latents.dtype)
      latents = latents + guidedNoise.asType(latents.dtype) * dt

      MLX.eval(latents)
    }

    progressHandler?(GenerationProgress(stage: .denoising, stepIndex: numSteps, totalSteps: numSteps))
    logger.info("Denoising complete")

    // 5. Decode latents
    progressHandler?(GenerationProgress(stage: .decoding, stepIndex: numSteps, totalSteps: numSteps))
    logger.info("Decoding with VAE...")

    // Unpack from sequence to spatial: (B, seqLen, 48) -> (B, 48, H/16, W/16)
    let spatialLatents = latents
      .reshaped(1, latentHeight, latentWidth, 48)
      .transposed(0, 3, 1, 2)

    // VAE decode: (B, 48, H/16, W/16) -> (B, 3, 1, H, W)
    let decoded = components.vae.decode(spatialLatents)
    MLX.eval(decoded)

    // Remove temporal dimension: (B, 3, 1, H, W) -> (B, 3, H, W) -> (3, H, W)
    var image = decoded
    if image.ndim == 5 && image.dim(2) == 1 {
      image = image.squeezed(axis: 2)
    }
    // Clamp to valid pixel range
    // VAE output is in ~[-1, 1] range, denormalize to [0, 1]
    image = QwenImageIO.denormalizeFromDecoder(image)
    image = MLX.clip(image, min: 0, max: 1)
    if ImageLevels.shouldApply(min: request.levelsMin, max: request.levelsMax) {
      image = ImageLevels.apply(image: image, min: request.levelsMin, max: request.levelsMax)
    }
    MLX.eval(image)

    GPU.clearCache()
    return image
  }

  // MARK: - Sigma Schedule (FlowMatchEulerDiscrete)

  /// Compute linear sigma schedule for FIBO (matching Python mflux LinearScheduler).
  ///
  /// FIBO uses `requires_sigma_shift=False`, so the schedule is simple:
  /// sigmas linearly spaced from 1.0 to 0.0, with N+1 values.
  /// Python: `sigmas = [1.0 - i/N for i in range(N+1)]`
  ///
  /// - Parameter numSteps: Number of denoising steps.
  /// - Returns: Array of N+1 sigma values from 1.0 to 0.0.
  public static func computeLinearSigmas(numSteps: Int) -> [Float] {
    guard numSteps > 0 else { return [1.0, 0.0] }
    return (0...numSteps).map { i in
      1.0 - Float(i) / Float(numSteps)
    }
  }

    /// Result of FMED schedule computation.
  public struct FMEDSchedule {
    /// `numSteps + 1` sigma values (trailing zero appended).
    public let sigmas: [Float]
    /// `numSteps` timestep values (sigma × numTrainTimesteps).
    public let timesteps: [Float]
  }

  /// Compute the FIBO FlowMatchEulerDiscrete sigma schedule.
  ///
  /// Exact port of mflux `FlowMatchEulerDiscreteScheduler._compute_timesteps_and_sigmas()`:
  /// 1. Linear sigma spacing from 1.0 down to 1/numTrainTimesteps (0.001)
  /// 2. Exponential time-shift with mu=1.0: sigma = e^mu / (e^mu + (1/t - 1))
  /// 3. Stretch-to-terminal with terminal=0.02
  /// 4. Timestep values = sigma × numTrainTimesteps (1000)
  ///
  /// FIBO's ModelConfig has `requires_sigma_shift=False`, so the dynamic mu
  /// computation (`set_image_seq_len`) is NOT used. The instance-level schedule
  /// with hardcoded mu=1.0 applies.
  ///
  /// - Parameter numSteps: Number of denoising steps.
  /// - Returns: `FMEDSchedule` with sigmas and timesteps.
  public static func computeFMEDSchedule(numSteps: Int) -> FMEDSchedule {
    guard numSteps > 1 else {
      // Degenerate: single step from sigma=1.0 to 0.0
      return FMEDSchedule(sigmas: [1.0, 0.0], timesteps: [1000.0])
    }

    let numTrainTimesteps: Float = 1000.0
    let mu: Float = 1.0          // Hardcoded in FMED __init__ for FIBO
    let shiftTerminal: Float = 0.02  // Hardcoded in FMED __init__
    let expMu = exp(mu)          // e^1.0 ≈ 2.71828

    let sigmaMin: Float = 1.0 / numTrainTimesteps  // 0.001
    let sigmaMax: Float = 1.0

    // Step 1: Linear timesteps from sigmaMax*1000 down to sigmaMin*1000
    //   timesteps_linear[i] = sigmaMax * 1000 - i * (sigmaMax - sigmaMin) * 1000 / (N-1)
    var sigmasLinear: [Float] = []
    for i in 0..<numSteps {
      let t = sigmaMax * numTrainTimesteps
        - Float(i) * (sigmaMax - sigmaMin) * numTrainTimesteps / Float(numSteps - 1)
      sigmasLinear.append(t / numTrainTimesteps)
    }
    // sigmasLinear = [1.0, 0.9655..., ..., 0.001]

    // Step 2: Exponential time-shift with mu=1.0, sigma_power=1.0
    //   shifted = exp(mu) / (exp(mu) + (1/t - 1))
    let sigmasShifted = sigmasLinear.map { t -> Float in
      guard t > 0 else { return 0 }
      return expMu / (expMu + (1.0 / t - 1.0))
    }

    // Step 3: Stretch to terminal (0.02)
    //   one_minus = [1-s for s in shifted]
    //   scale = one_minus[-1] / (1 - terminal)
    //   final = [1 - (oms / scale) for oms in one_minus]
    let oneMinusSigmas = sigmasShifted.map { 1.0 - $0 }
    let scaleFactor = oneMinusSigmas.last! / (1.0 - shiftTerminal)
    let sigmasFinal = oneMinusSigmas.map { 1.0 - ($0 / scaleFactor) }

    // Step 4: Timesteps = sigma × numTrainTimesteps
    let timesteps = sigmasFinal.map { $0 * numTrainTimesteps }

    // Append trailing zero to sigmas
    var sigmasWithZero = sigmasFinal
    sigmasWithZero.append(0.0)

    return FMEDSchedule(sigmas: sigmasWithZero, timesteps: timesteps)
  }

  // MARK: - Classifier-Free Guidance

  /// Apply classifier-free guidance to batched noise predictions.
  ///
  /// The transformer outputs noise for both negative (uncond) and positive (cond)
  /// prompts, stacked along the batch dimension: `[uncond, cond]`.
  ///
  /// CFG formula: `noise_uncond + guidance * (noise_cond - noise_uncond)`
  ///
  /// - Parameters:
  ///   - noise: Combined noise predictions `[2*B, seqLen, channels]`.
  ///   - guidance: CFG scale (typically 4.0 for FIBO).
  /// - Returns: Guided noise `[B, seqLen, channels]`.
  static func applyClassifierFreeGuidance(
    noise: MLXArray,
    guidance: Float
  ) -> MLXArray {
    let half = noise.dim(0) / 2
    let noiseUncond = noise[..<half]
    let noiseCond = noise[half...]
    return noiseUncond + MLXArray(guidance) * (noiseCond - noiseUncond)
  }
}
