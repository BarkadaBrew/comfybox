// Flux2Pipeline.swift — Pipeline orchestration for Flux 2 Klein image generation
// Ported from mflux: flux2_klein.py

import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom
import Hub

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

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

  /// Path to a source image for img2img. When set with denoise < 1.0,
  /// the pipeline encodes this image as a starting point instead of pure noise.
  public var inputImagePath: URL?

  /// Denoise strength (0.0-1.0). 1.0 = full txt2img (all steps from noise).
  /// 0.5 = start halfway through the schedule, preserving composition.
  /// Only used when `inputImagePath` is set.
  public var denoise: Float

  public init(
    prompt: String,
    negativePrompt: String? = nil,
    width: Int = 1024,
    height: Int = 1024,
    steps: Int = 4,
    guidanceScale: Float = 1.0,
    seed: UInt64? = nil,
    outputPath: URL = URL(fileURLWithPath: "flux2-output.png"),
    maxSequenceLength: Int = 512,
    inputImagePath: URL? = nil,
    denoise: Float = 1.0
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
    self.inputImagePath = inputImagePath
    self.denoise = denoise
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
    case img2imgFailed(String)

    public var errorDescription: String? {
      switch self {
      case .modelNotLoaded: return "Model not loaded"
      case .tokenizerNotLoaded: return "Tokenizer not loaded"
      case .invalidDimensions(let msg): return "Invalid dimensions: \(msg)"
      case .generationFailed(let msg): return "Generation failed: \(msg)"
      case .img2imgFailed(let msg): return "Img2img failed: \(msg)"
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

  // MARK: - Img2Img Helpers

  #if canImport(CoreGraphics)
  /// Load a CGImage from a file URL.
  private func loadImage(from url: URL) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
      throw Flux2PipelineError.img2imgFailed("Failed to load image from \(url.path)")
    }
    return cgImage
  }

  /// Load, resize, and convert an image to an MLXArray in [-1, 1] NCHW format
  /// suitable for VAE encoding.
  ///
  /// - Parameters:
  ///   - cgImage: Source image.
  ///   - width: Target width in pixels.
  ///   - height: Target height in pixels.
  /// - Returns: MLXArray in NCHW layout `(1, 3, H, W)` with values in [-1, 1].
  private func prepareImageForVAE(
    _ cgImage: CGImage,
    width: Int,
    height: Int
  ) throws -> MLXArray {
    // Resize to target dimensions and get [0, 1] NCHW array
    let pixelArray = try QwenImageIO.resizedPixelArray(
      from: cgImage,
      width: width,
      height: height,
      addBatchDimension: true,
      dtype: .float32
    )
    // Normalize from [0, 1] to [-1, 1]
    return QwenImageIO.normalizeForEncoder(pixelArray)
  }
  #endif

  /// Match encoded latent spatial dimensions to the target grid size via center-crop or pad.
  ///
  /// After VAE encoding, the spatial dimensions may not exactly match the noise
  /// grid computed from the target resolution. This method center-crops (if larger)
  /// or center-pads (if smaller) to match.
  ///
  /// Ported from mflux `Flux2Klein._match_latent_spatial_size`.
  ///
  /// - Parameters:
  ///   - encoded: VAE-encoded latents in NCHW `(B, C, H, W)`.
  ///   - targetHeight: Expected spatial height (latentHeight * 2).
  ///   - targetWidth: Expected spatial width (latentWidth * 2).
  /// - Returns: Latents with spatial dims matching `(targetHeight, targetWidth)`.
  private func matchLatentSpatialSize(
    _ encoded: MLXArray,
    targetHeight: Int,
    targetWidth: Int
  ) -> MLXArray {
    var result = encoded
    let height = result.shape[2]
    let width = result.shape[3]

    if height != targetHeight {
      if height > targetHeight {
        // Center-crop height
        let offset = (height - targetHeight) / 2
        result = result[0..., 0..., offset..<(offset + targetHeight), 0...]
      } else {
        // Center-pad height
        let padTotal = targetHeight - height
        let padBefore = padTotal / 2
        let padAfter = padTotal - padBefore
        result = MLX.padded(result, widths: [
          IntOrPair(0), IntOrPair(0), IntOrPair((padBefore, padAfter)), IntOrPair(0)
        ])
      }
    }

    if width != targetWidth {
      if width > targetWidth {
        // Center-crop width
        let offset = (width - targetWidth) / 2
        result = result[0..., 0..., 0..., offset..<(offset + targetWidth)]
      } else {
        // Center-pad width
        let padTotal = targetWidth - width
        let padBefore = padTotal / 2
        let padAfter = padTotal - padBefore
        result = MLX.padded(result, widths: [
          IntOrPair(0), IntOrPair(0), IntOrPair(0), IntOrPair((padBefore, padAfter))
        ])
      }
    }

    return result
  }

  /// Apply batch norm normalization using the VAE's running statistics.
  ///
  /// The Flux 2 transformer operates in BN-normalized space. Without this
  /// normalization, img2img produces garbage. The decode path applies
  /// the inverse (x * std + mean).
  ///
  /// - Parameters:
  ///   - latents: Patchified latents in NCHW `(B, 4*C, H/2, W/2)`.
  ///   - vae: Flux2VAE instance with loaded batch norm stats.
  /// - Returns: BN-normalized latents.
  private func bnNormalize(_ latents: MLXArray, vae: Flux2VAE) -> MLXArray {
    let mean = vae.bn.runningMean.reshaped(1, -1, 1, 1).asType(latents.dtype)
    let std = sqrt(vae.bn.runningVar.reshaped(1, -1, 1, 1) + vae.bn.eps).asType(latents.dtype)
    return (latents - mean) / std
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

    // 3a. Img2img: encode source image and blend with noise
    var startStep = 0

    #if canImport(CoreGraphics)
    if let imagePath = request.inputImagePath, request.denoise < 1.0 {
      logger.info("Img2img: encoding source image from \(imagePath.path) (denoise=\(request.denoise))...")

      // 1. Load and resize image to target dimensions
      let cgImage = try loadImage(from: imagePath)
      let imageArray = try prepareImageForVAE(cgImage, width: request.width, height: request.height)

      // 2. VAE encode -> (B, 32, H/8, W/8) in NCHW
      var encoded = components.vae.encode(imageArray)

      // 3. Ensure 4D (remove temporal dim if present)
      if encoded.ndim == 5 && encoded.shape[2] == 1 {
        encoded = encoded.squeezed(axis: 2)
      }

      // 4. Ensure even spatial dims (patchify needs H,W divisible by 2)
      let h = encoded.shape[2]
      let w = encoded.shape[3]
      if h % 2 != 0 {
        encoded = encoded[0..., 0..., 0..<(h - 1), 0...]
      }
      if w % 2 != 0 {
        encoded = encoded[0..., 0..., 0..., 0..<(w - 1)]
      }

      // 5. Match spatial size to noise grid
      encoded = matchLatentSpatialSize(encoded, targetHeight: latentHeight * 2, targetWidth: latentWidth * 2)

      // 6. Patchify: (B, 32, H, W) -> (B, 128, H/2, W/2)
      encoded = Flux2LatentCreator.patchifyLatents(encoded)

      // 7. BN normalize (MANDATORY for Flux 2)
      encoded = bnNormalize(encoded, vae: components.vae)

      // 8. Pack to sequence: (B, 128, H/2, W/2) -> (B, H/2*W/2, 128)
      let cleanLatents = Flux2LatentCreator.packLatents(encoded)

      // 9. Compute start step and blend noise with clean latents
      startStep = max(0, request.steps - Int(ceil(Float(request.steps) * request.denoise)))
      let sigma = sigmaValues[startStep]
      // mflux convention: (1 - sigma) * clean + sigma * noise
      latents = MLXArray(sigma) * latents + MLXArray(1.0 - sigma) * cleanLatents
      MLX.eval(latents)

      logger.info("Img2img: starting at step \(startStep)/\(request.steps), sigma=\(sigma)")
    }
    #endif

    // 4. Denoising loop (starts from startStep for img2img)
    let effectiveSteps = request.steps - startStep
    logger.info("Running \(effectiveSteps) denoising steps\(startStep > 0 ? " (from step \(startStep))" : "")...")
    for stepIndex in startStep..<request.steps {
      try Task.checkCancellation()
      progressHandler?(GenerationProgress(stage: .denoising, stepIndex: stepIndex - startStep, totalSteps: effectiveSteps))

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

    progressHandler?(GenerationProgress(stage: .denoising, stepIndex: effectiveSteps, totalSteps: effectiveSteps))
    logger.info("Denoising complete")

    // 5. Decode latents
    progressHandler?(GenerationProgress(stage: .decoding, stepIndex: effectiveSteps, totalSteps: effectiveSteps))
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
