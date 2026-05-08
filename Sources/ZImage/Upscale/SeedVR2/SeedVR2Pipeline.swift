import Foundation
import Logging
import MLX
import MLXNN

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// End-to-end SeedVR2 super-resolution pipeline.
///
/// Supports both 3B and 7B model variants with automatic detection.
/// The variant is determined by which transformer weight file is present.
public final class SeedVR2Pipeline {

  public let transformer: SeedVR2Transformer
  public let vae: SeedVR2VAE
  public let scheduler: SeedVR2EulerScheduler
  public let logger: Logger
  public let weightsDirectory: URL
  public let modelConfig: SeedVR2ModelConfig

  public enum PipelineError: Error, CustomStringConvertible {
    case weightsDirectoryNotFound(String)
    case noTransformerWeightsFound(String)
    case imageLoadFailed(String)
    case preprocessFailed(Error)
    case encodeFailed(Error)
    case decodeFailed(Error)
    case saveFailed(String)

    public var description: String {
      switch self {
      case .weightsDirectoryNotFound(let path):
        return "Weights directory not found: \(path)"
      case .noTransformerWeightsFound(let path):
        return "No SeedVR2 transformer weights (3B or 7B) found in: \(path)"
      case .imageLoadFailed(let path):
        return "Failed to load input image: \(path)"
      case .preprocessFailed(let error):
        return "Image preprocessing failed: \(error)"
      case .encodeFailed(let error):
        return "VAE encoding failed: \(error)"
      case .decodeFailed(let error):
        return "VAE decoding failed: \(error)"
      case .saveFailed(let path):
        return "Failed to save output image to: \(path)"
      }
    }
  }

  /// Creates a SeedVR2 pipeline, auto-detecting the model variant.
  public init(
    weightsPath: String,
    steps: Int = 1,
    logger: Logger = Logger(label: "seedvr2.pipeline")
  ) throws {
    let directory = URL(fileURLWithPath: weightsPath)
    guard FileManager.default.fileExists(atPath: directory.path) else {
      throw PipelineError.weightsDirectoryNotFound(weightsPath)
    }

    self.weightsDirectory = directory
    self.logger = logger

    // Auto-detect model variant
    guard let detected = SeedVR2WeightLoader.detectTransformerWeights(in: directory) else {
      throw PipelineError.noTransformerWeightsFound(weightsPath)
    }

    let config = detected.config
    self.modelConfig = config
    logger.info("Detected SeedVR2 model: \(config == .preset7B ? "7B" : "3B")")
    logger.info("  vid_dim=\(config.vidDim), heads=\(config.heads), layers=\(config.numLayers), mlp=\(config.mlpType == .swiglu ? "SwiGLU" : "GELU")")

    // Initialize models with detected config
    logger.info("Initializing SeedVR2 transformer...")
    self.transformer = SeedVR2Transformer(config: config)

    logger.info("Initializing SeedVR2 VAE...")
    self.vae = SeedVR2VAE()

    self.scheduler = SeedVR2EulerScheduler(
      numTrainTimesteps: 1000,
      numInferenceSteps: steps,
      guidance: 1.0
    )

    // Load weights
    logger.info("Loading weights...")
    try SeedVR2WeightLoader.loadTransformerWeights(
      into: transformer,
      from: directory,
      fileName: detected.fileName,
      logger: logger
    )

    try SeedVR2WeightLoader.loadVAEWeights(
      into: vae,
      from: directory,
      logger: logger
    )

    logger.info("SeedVR2 pipeline ready (\(config == .preset7B ? "7B" : "3B"))")
  }

  /// Progress callback for upscale operations.
  /// Called after each denoising step with (currentStep, totalSteps).
  public typealias UpscaleProgressHandler = @Sendable (Int, Int) -> Void

  /// Upscales an image to the target resolution.
  public func upscale(
    imagePath: String,
    targetResolution: Int = 2048,
    seed: Int? = nil,
    softness: Float = 0.0,
    progressHandler: UpscaleProgressHandler? = nil
  ) throws -> CGImage {
    let actualSeed = seed ?? Int.random(in: 0..<Int(Int32.max))
    logger.info("Upscaling \(imagePath) to \(targetResolution)px, seed=\(actualSeed), softness=\(softness)")

    // 1. Load and preprocess
    logger.info("Step 1: Loading and preprocessing image...")
    let inputImage: CGImage
    do {
      inputImage = try SeedVR2Util.loadImage(from: imagePath)
    } catch {
      throw PipelineError.imageLoadFailed(imagePath)
    }

    let preprocessed: SeedVR2Util.PreprocessResult
    do {
      preprocessed = try SeedVR2Util.preprocessImage(
        inputImage,
        targetResolution: targetResolution,
        softness: softness
      )
    } catch {
      throw PipelineError.preprocessFailed(error)
    }

    let trueH = preprocessed.trueHeight
    let trueW = preprocessed.trueWidth
    logger.info("Preprocessed: \(inputImage.width)x\(inputImage.height) -> \(trueW)x\(trueH)")

    // 2. VAE encode
    logger.info("Step 2: VAE encoding...")
    let encodedLatent = vae.encode(preprocessed.tensor)
    MLX.eval(encodedLatent)
    logger.info("Encoded latent shape: \(encodedLatent.shape)")

    // 3. Create conditioning
    logger.info("Step 3: Creating condition...")
    let condition = SeedVR2LatentCreator.createCondition(encodedLatent: encodedLatent)

    // 4. Create noise latents
    logger.info("Step 4: Creating noise latents...")
    let latentH = encodedLatent.dim(3)
    let latentW = encodedLatent.dim(4)
    var latents = SeedVR2LatentCreator.createNoiseLatents(
      seed: actualSeed, height: latentH, width: latentW
    )
    MLX.eval(latents)

    // 5. Load text embeddings
    logger.info("Step 5: Loading text embeddings...")
    let txtPos = try SeedVR2TextEmbeddings.loadPositive(from: weightsDirectory)
    MLX.eval(txtPos)

    // 6. Denoising loop
    let numSteps = scheduler.numInferenceSteps
    logger.info("Step 6: Denoising (\(numSteps) step\(numSteps == 1 ? "" : "s"))...")

    for t in 0..<numSteps {
      let modelInput = MLX.concatenated([latents, condition], axis: 1)
      let timestep = scheduler.timesteps[t]
      let noise = transformer(vid: modelInput, txt: txtPos, timestep: timestep)
      latents = scheduler.step(modelOutput: noise, timestepIndex: t, sample: latents)
      MLX.eval(latents)
      logger.info("  Step \(t + 1)/\(numSteps) complete")
      progressHandler?(t + 1, numSteps)
    }

    // 7. VAE decode
    logger.info("Step 7: VAE decoding...")
    var decoded = vae.decode(latents)
    MLX.eval(decoded)

    // 8. Crop to true dimensions
    if decoded.ndim == 5 {
      decoded = decoded[0..., 0..., 0..., ..<trueH, ..<trueW]
    } else if decoded.ndim == 4 {
      decoded = decoded[0..., 0..., ..<trueH, ..<trueW]
    }

    var styleRef = preprocessed.tensor
    if styleRef.ndim == 4 {
      styleRef = styleRef[0..., 0..., ..<trueH, ..<trueW]
    }

    // 9. Color correction
    logger.info("Step 8: Applying color correction...")
    var decodedForCC = decoded
    if decodedForCC.ndim == 5 && decodedForCC.dim(2) == 1 {
      decodedForCC = decodedForCC[0..., 0..., 0, 0..., 0...]
    }
    let corrected = SeedVR2Util.applyColorCorrection(content: decodedForCC, style: styleRef)
    MLX.eval(corrected)

    // 10. Convert to CGImage
    logger.info("Step 9: Converting to CGImage...")
    let result = try SeedVR2Util.tensorToImage(corrected)

    logger.info("Upscale complete: \(result.width)x\(result.height)")
    return result
  }

  /// Upscales an image and saves the result to a file.
  @discardableResult
  public func upscaleAndSave(
    imagePath: String,
    outputPath: String? = nil,
    targetResolution: Int = 2048,
    seed: Int? = nil,
    softness: Float = 0.0,
    progressHandler: UpscaleProgressHandler? = nil
  ) throws -> String {
    let result = try upscale(
      imagePath: imagePath,
      targetResolution: targetResolution,
      seed: seed,
      softness: softness,
      progressHandler: progressHandler
    )

    let outPath: String
    if let provided = outputPath {
      outPath = provided
    } else {
      let inputURL = URL(fileURLWithPath: imagePath)
      let baseName = inputURL.deletingPathExtension().lastPathComponent
      let ext = inputURL.pathExtension.isEmpty ? "png" : inputURL.pathExtension
      outPath = inputURL.deletingLastPathComponent()
        .appendingPathComponent("\(baseName)-upscaled.\(ext)").path
    }

    let outURL = URL(fileURLWithPath: outPath)
    guard let destination = CGImageDestinationCreateWithURL(
      outURL as CFURL,
      UTType.png.identifier as CFString,
      1,
      nil
    ) else {
      throw PipelineError.saveFailed(outPath)
    }

    CGImageDestinationAddImage(destination, result, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw PipelineError.saveFailed(outPath)
    }

    logger.info("Saved upscaled image to \(outPath)")
    return outPath
  }
}
#endif
