// Flux2Pipeline.swift — Pipeline orchestration for Flux 2 Klein image generation
// Ported from mflux: flux2_klein.py

import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom
import Hub

/// Configuration for a Flux 2 Klein generation request.
public struct Flux2GenerationRequest: Sendable {
  public var prompt: String
  public var negativePrompt: String?
  public var width: Int
  public var height: Int
  public var steps: Int
  public var guidanceScale: Float
  public var seed: UInt64?
  public var outputPath: URL
  public var maxSequenceLength: Int

  public init(
    prompt: String,
    negativePrompt: String? = nil,
    width: Int = 1024,
    height: Int = 1024,
    steps: Int = 4,
    guidanceScale: Float = 1.0,
    seed: UInt64? = nil,
    outputPath: URL = URL(fileURLWithPath: "flux2-output.png"),
    maxSequenceLength: Int = 512
  ) {
    self.prompt = prompt
    self.negativePrompt = negativePrompt
    self.width = width
    self.height = height
    self.steps = steps
    self.guidanceScale = guidanceScale
    self.seed = seed
    self.outputPath = outputPath
    self.maxSequenceLength = maxSequenceLength
  }
}

/// Orchestrates Flux 2 Klein image generation.
///
/// ## Pipeline Stages
///
/// 1. Load model components via `Flux2Initializer`
/// 2. Load tokenizer and encode prompt via `Flux2PromptEncoder`
/// 3. Prepare packed latents via `Flux2LatentCreator`
/// 4. Run denoising loop using flow-match Euler scheduler
/// 5. Decode packed latents via `Flux2VAE.decodePackedLatents`
/// 6. Save output image
///
/// The Flux 2 denoising loop differs from Flux 1 in several ways:
/// - Uses packed latents (spatial dims folded into sequence)
/// - Timestep is passed as a raw sigma value (not normalized)
/// - Optional CFG with negative prompt when guidance > 1.0
/// - Transformer expects `(hiddenStates, encoderHiddenStates, timestep, imgIds, txtIds)`
public final class Flux2Pipeline {

  public enum Flux2PipelineError: Error, LocalizedError {
    case modelNotLoaded
    case tokenizerNotLoaded
    case invalidDimensions(String)
    case generationFailed(String)

    public var errorDescription: String? {
      switch self {
      case .modelNotLoaded: return "Model not loaded"
      case .tokenizerNotLoaded: return "Tokenizer not loaded"
      case .invalidDimensions(let msg): return "Invalid dimensions: \(msg)"
      case .generationFailed(let msg): return "Generation failed: \(msg)"
      }
    }
  }

  /// Progress reporting for Flux 2 generation.
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
  private var components: Flux2Initializer.Components?
  private var tokenizer: QwenTokenizer?
  private var modelSnapshot: URL?
  private var isLoaded: Bool = false

  // Model configs for current loaded model
  private var transformerConfig: Flux2TransformerConfig?
  private var textEncoderConfig: Qwen3TextEncoderConfiguration?
  private var _isBaseModel: Bool = false

  /// Whether the loaded model is a base (non-distilled) variant.
  /// Base models support guidance > 1.0 and default to 50 steps.
  public var isBaseModel: Bool { _isBaseModel }

  /// Whether the loaded model is a distilled (few-step) variant.
  public var isDistilled: Bool { !_isBaseModel }

  /// Default inference steps for the loaded model.
  /// Base models default to 50 steps; distilled models default to 4.
  public var defaultSteps: Int {
    _isBaseModel ? 50 : 4
  }

  public init(logger: Logger = Logger(label: "z-image.flux2-pipeline")) {
    self.logger = logger
  }

  /// Whether the model is currently loaded.
  public var loaded: Bool { isLoaded }

  /// Unload model components and free memory.
  public func unload() {
    components = nil
    tokenizer = nil
    modelSnapshot = nil
    transformerConfig = nil
    textEncoderConfig = nil
    _isBaseModel = false
    isLoaded = false
    GPU.clearCache()
    logger.info("Flux 2 model unloaded")
  }

  /// Load the Flux 2 Klein model from a snapshot directory.
  ///
  /// - Parameters:
  ///   - snapshot: Root URL of the model snapshot directory.
  ///   - config: Transformer config for this model variant. Defaults to Klein 4B.
  ///   - textEncoderConfig: Text encoder config. Defaults to Klein 4B Qwen3 config.
  ///   - isBase: Whether this is a base (non-distilled) model. Base models support
  ///     guidance > 1.0 and default to 50 inference steps.
  ///   - progressHandler: Optional progress callback.
  public func loadModel(
    from snapshot: URL,
    config: Flux2TransformerConfig = Flux2TransformerConfig(),
    textEncoderConfig: Qwen3TextEncoderConfiguration = Qwen3TextEncoderConfiguration(),
    isBase: Bool = false,
    progressHandler: ProgressHandler? = nil
  ) throws {
    if isLoaded && modelSnapshot == snapshot { return }

    progressHandler?(GenerationProgress(stage: .loadingModel, stepIndex: 0, totalSteps: 1))

    // Load tokenizer
    let tokenizerDir = snapshot.appendingPathComponent("tokenizer")
    let hub = HubApi()
    self.tokenizer = try QwenTokenizer.load(from: tokenizerDir, hubApi: hub)

    // Load all model components
    self.components = try Flux2Initializer.load(
      from: snapshot,
      transformerConfig: config,
      textEncoderConfig: textEncoderConfig,
      dtype: .bfloat16,
      logger: logger
    )

    self.modelSnapshot = snapshot
    self.transformerConfig = config
    self.textEncoderConfig = textEncoderConfig
    self._isBaseModel = isBase
    self.isLoaded = true

    progressHandler?(GenerationProgress(stage: .loadingModel, stepIndex: 1, totalSteps: 1))
    let modelKind = isBase ? "base (non-distilled)" : "distilled"
    logger.info("Flux 2 Klein model loaded from \(snapshot.path) [\(modelKind)]")
  }

  /// Generate an image and save it to disk.
  ///
  /// - Parameters:
  ///   - request: Generation parameters.
  ///   - progressHandler: Optional progress callback.
  /// - Returns: The output file URL.
  public func generate(
    _ request: Flux2GenerationRequest,
    progressHandler: ProgressHandler? = nil
  ) async throws -> URL {
    let decoded = try await generateCore(request, progressHandler: progressHandler)

    progressHandler?(GenerationProgress(stage: .saving, stepIndex: request.steps, totalSteps: request.steps))
    try QwenImageIO.saveImage(array: decoded, to: request.outputPath)
    logger.info("Wrote image to \(request.outputPath.path)")

    return request.outputPath
  }

  /// Generate an image and return raw PNG data.
  public func generateToMemory(
    _ request: Flux2GenerationRequest,
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
    _ request: Flux2GenerationRequest,
    progressHandler: ProgressHandler? = nil
  ) async throws -> MLXArray {
    guard let components = components else {
      throw Flux2PipelineError.modelNotLoaded
    }
    guard let tokenizer = tokenizer else {
      throw Flux2PipelineError.tokenizerNotLoaded
    }

    // Validate dimensions — Flux 2 needs multiples of 16
    let vaeScale = 16
    if request.width % vaeScale != 0 {
      throw Flux2PipelineError.invalidDimensions(
        "Width must be divisible by \(vaeScale) (got \(request.width))"
      )
    }
    if request.height % vaeScale != 0 {
      throw Flux2PipelineError.invalidDimensions(
        "Height must be divisible by \(vaeScale) (got \(request.height))"
      )
    }

    // 1. Encode prompt
    progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: request.steps))
    logger.info("Encoding prompt...")

    let doCFG = request.guidanceScale > 1.0

    let (promptEmbeds, textIds) = try Flux2PromptEncoder.encodePrompt(
      prompt: request.prompt,
      tokenizer: tokenizer,
      textEncoder: components.textEncoder,
      numImagesPerPrompt: 1,
      maxSequenceLength: request.maxSequenceLength,
      textEncoderOutLayers: Flux2PromptEncoder.defaultOutputLayers
    )

    var negativePromptEmbeds: MLXArray?
    var negativeTextIds: MLXArray?
    if doCFG {
      let negPrompt = request.negativePrompt ?? " "
      let (ne, nti) = try Flux2PromptEncoder.encodePrompt(
        prompt: negPrompt,
        tokenizer: tokenizer,
        textEncoder: components.textEncoder,
        numImagesPerPrompt: 1,
        maxSequenceLength: request.maxSequenceLength,
        textEncoderOutLayers: Flux2PromptEncoder.defaultOutputLayers
      )
      negativePromptEmbeds = ne
      negativeTextIds = nti
      MLX.eval(promptEmbeds, ne)
    } else {
      MLX.eval(promptEmbeds)
    }
    logger.info("Text encoding complete")

    // 2. Prepare latents
    let seed = request.seed ?? UInt64.random(in: 0...UInt64.max)
    let (packedLatents, latentIds, latentHeight, latentWidth) = Flux2LatentCreator.preparePackedLatents(
      seed: seed,
      height: request.height,
      width: request.width,
      batchSize: 1
    )
    var latents = packedLatents

    // 3. Build sigma schedule
    // Flux 2 uses FlowMatchEulerDiscrete with empirical mu shift.
    let hPatches = request.height / 16
    let wPatches = request.width / 16
    let imageSeqLen = hPatches * wPatches
    let mu = Flux2Pipeline.computeEmpiricalMu(imageSeqLen: imageSeqLen, numSteps: request.steps)

    let sigmaValues = Flux2Pipeline.computeFlux2Sigmas(
      numSteps: request.steps,
      mu: mu,
      numTrainTimesteps: 1000
    )

    let sigmas = MLXArray(sigmaValues)

    // 4. Denoising loop
    logger.info("Running \(request.steps) denoising steps...")
    for stepIndex in 0..<request.steps {
      try Task.checkCancellation()
      progressHandler?(GenerationProgress(stage: .denoising, stepIndex: stepIndex, totalSteps: request.steps))

      let timestep = sigmas[stepIndex]

      // Guidance embedding: pass the guidance scale to the transformer when the
      // model is a base (non-distilled) variant.
      // For distilled models, pass nil (guidance embeddings are unused).
      let guidanceForTransformer: MLXArray? = _isBaseModel
        ? MLXArray(request.guidanceScale)
        : nil

      // Forward pass through transformer
      let noisePred = components.transformer(
        hiddenStates: latents,
        encoderHiddenStates: promptEmbeds,
        timestep: timestep,
        imgIds: latentIds,
        txtIds: textIds,
        guidance: guidanceForTransformer
      )

      // Apply CFG if guidance > 1.0
      let guidedNoise: MLXArray
      if doCFG, let ne = negativePromptEmbeds, let nti = negativeTextIds {
        let negativeNoisePred = components.transformer(
          hiddenStates: latents,
          encoderHiddenStates: ne,
          timestep: timestep,
          imgIds: latentIds,
          txtIds: nti,
          guidance: guidanceForTransformer
        )
        guidedNoise = negativeNoisePred + request.guidanceScale * (noisePred - negativeNoisePred)
      } else {
        guidedNoise = noisePred
      }

      // Euler step: latents = latents + noise * dt
      // Note: In Python mflux, step does latents + dt * noise where dt = sigmas[t+1] - sigmas[t]
      let dt = (sigmas[stepIndex + 1] - sigmas[stepIndex]).asType(latents.dtype)
      latents = latents + guidedNoise * dt
      MLX.eval(latents)
    }

    progressHandler?(GenerationProgress(stage: .denoising, stepIndex: request.steps, totalSteps: request.steps))
    logger.info("Denoising complete")

    // 5. Decode latents
    progressHandler?(GenerationProgress(stage: .decoding, stepIndex: request.steps, totalSteps: request.steps))
    logger.info("Decoding with VAE...")

    // Reshape packed latents back to spatial for VAE decode
    // packed: (B, H'*W', 4*C) -> (B, 4*C, H', W')
    let packedForVAE = latents
      .reshaped(latents.shape[0], latentHeight, latentWidth, latents.shape[latents.ndim - 1])
      .transposed(0, 3, 1, 2)

    let decoded = components.vae.decodePackedLatents(packedForVAE)

    // Denormalize from [-1, 1] to [0, 1]
    let image = QwenImageIO.denormalizeFromDecoder(decoded)
    let clipped = MLX.clip(image, min: 0, max: 1)
    MLX.eval(clipped)

    GPU.clearCache()
    return clipped
  }

  // MARK: - Flux2 Sigma Schedule

  /// Compute empirical mu for the Flux 2 exponential time shift.
  ///
  /// Ported from mflux `FlowMatchEulerDiscreteScheduler._compute_empirical_mu`.
  /// The mu value controls the time shift applied to the sigma schedule,
  /// varying based on image resolution (sequence length) and number of steps.
  static func computeEmpiricalMu(imageSeqLen: Int, numSteps: Int) -> Float {
    let a1: Float = 8.73809524e-05
    let b1: Float = 1.89833333
    let a2: Float = 0.00016927
    let b2: Float = 0.45666666

    if imageSeqLen > 4300 {
      return a2 * Float(imageSeqLen) + b2
    }

    let m200 = a2 * Float(imageSeqLen) + b2
    let m10 = a1 * Float(imageSeqLen) + b1
    let a = (m200 - m10) / 190.0
    let b = m200 - 200.0 * a
    return a * Float(numSteps) + b
  }

  /// Compute the Flux 2 sigma schedule with exponential time shift.
  ///
  /// Ported from mflux `FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas`.
  /// Unlike the Flux 1 schedule, Flux 2 uses:
  /// - Empirical mu (resolution + step dependent) instead of simple shift
  /// - Exponential time shift: sigma = exp(mu) / (exp(mu) + (1/t - 1)^1)
  static func computeFlux2Sigmas(
    numSteps: Int,
    mu: Float,
    numTrainTimesteps: Int
  ) -> [Float] {
    // Linear sigmas from 1.0 down to 1/numSteps
    let linearSigmas: [Float] = (0..<numSteps).map { i in
      let t = Float(i) / Float(max(1, numSteps - 1))
      return 1.0 - t * (1.0 - 1.0 / Float(numSteps))
    }

    // Apply exponential time shift
    let expMu = exp(mu)
    let shifted: [Float] = linearSigmas.map { t in
      expMu / (expMu + pow(1.0 / t - 1.0, 1.0))
    }

    // Append trailing zero
    var sigmas = shifted
    sigmas.append(0.0)
    return sigmas
  }
}
