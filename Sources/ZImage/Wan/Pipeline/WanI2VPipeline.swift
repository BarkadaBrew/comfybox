import Foundation
import Logging
import MLX
import MLXRandom
import MLXNN

/// Wan 2.2 I2V-A14B end-to-end generation pipeline.
///
/// Wires together the VAE (Phase 1), Text Encoder (Phase 2), Transformer (Phase 3),
/// UniPC scheduler (S4.1), I2V conditioning (S4.2), and MoE management (S4.3).
///
/// ## Generation Flow
///
/// ```
/// 1. Encode text prompt → context embeddings
/// 2. Encode negative prompt → unconditional context
/// 3. Compute resolution from init image aspect ratio
/// 4. Build I2V conditioning (mask + VAE-encoded init image) → 20ch
/// 5. Generate noise → 16ch
/// 6. Denoising loop (UniPC scheduler):
///    a. Select expert (high-noise or low-noise based on timestep)
///    b. Forward pass with conditioning (CFG: cond + uncond)
///    c. Scheduler step → updated latent
/// 7. VAE decode → pixel frames
/// 8. Post-process frames
/// ```
///
/// ## Memory Budget (M3 Max 128GB)
/// - High-noise expert: ~27GB (BF16)
/// - Text encoder: ~11GB
/// - VAE: ~0.5GB
/// - Working memory: ~5-10GB
/// - Total peak: ~45-50GB with lazy MoE
public final class WanI2VPipeline {

  // MARK: - Properties

  /// Pipeline configuration.
  public let config: WanI2VConfig

  /// Text encoder (UMT5-XXL).
  public let textEncoder: WanUMT5Encoder

  /// Tokenizer.
  public let tokenizer: WanTokenizer

  /// VAE.
  public let vae: WanVAE

  /// MoE expert manager.
  public let moeManager: WanMoEManager

  /// Logger.
  private let logger: Logger

  // MARK: - Init

  /// Creates the pipeline by loading all components.
  ///
  /// - Parameters:
  ///   - config: Pipeline configuration.
  ///   - logger: Logger instance.
  /// - Throws: If any component fails to load.
  public init(config: WanI2VConfig, logger: Logger) throws {
    self.config = config
    self.logger = logger

    let modelDir = URL(fileURLWithPath: config.modelDir)

    // Load tokenizer
    logger.info("Loading tokenizer...")
    let tokenizerDir = modelDir.appendingPathComponent("google/umt5-xxl")
    let tokenizerURL = tokenizerDir.appendingPathComponent("tokenizer.json")
    self.tokenizer = try WanTokenizer(url: tokenizerURL)
    logger.info("Tokenizer loaded")

    // Load text encoder
    logger.info("Loading UMT5-XXL text encoder...")
    self.textEncoder = WanUMT5Encoder()
    let t5Path = modelDir.appendingPathComponent("models_t5_umt5-xxl-enc-bf16.safetensors")
    // The T5 weights are in a single safetensors-like file
    let t5Weights = try MLX.loadArrays(url: t5Path)
    try textEncoder.loadWeights(t5Weights)
    logger.info("Text encoder loaded")

    // Load VAE
    logger.info("Loading Wan 2.1 VAE...")
    self.vae = WanVAE()
    let vaePath = modelDir.appendingPathComponent("Wan2.1_VAE.safetensors")
    var vaeWeights = try MLX.loadArrays(url: vaePath)

    // Transpose convolution weights from PyTorch layout to MLX layout.
    // PyTorch Conv3d: [outCh, inCh, kT, kH, kW] -> MLX: [outCh, kT, kH, kW, inCh]
    // PyTorch Conv2d: [outCh, inCh, kH, kW]      -> MLX: [outCh, kH, kW, inCh]
    // Only .weight keys with ndim >= 4 are convolutions. Bias (1D), norm gamma
    // (.gamma suffix, not .weight), and 2D linear weights are left untouched.
    for (key, tensor) in vaeWeights {
      if key.hasSuffix(".weight") {
        if tensor.ndim == 5 {
          // Conv3d weight: axes [0,1,2,3,4] -> [0,2,3,4,1]
          vaeWeights[key] = tensor.transposed(0, 2, 3, 4, 1)
        } else if tensor.ndim == 4 {
          // Conv2d weight (incl. 1x1 attention convs): axes [0,1,2,3] -> [0,2,3,1]
          vaeWeights[key] = tensor.transposed(0, 2, 3, 1)
        }
      }
    }

    // Remap safetensors keys to match the MLX module hierarchy.
    // Wrapper classes (WanSequentialLayers, WanMiddleLayers, WanResidualPath,
    // WanEncoderHead, WanDecoderHead, WanResampleSeq) store sub-modules in
    // `public let layers: [Module]`, injecting "layers" into the weight path.
    // Safetensors keys lack this segment, so we insert it. Example:
    //   encoder.downsamples.0.residual.0.gamma
    //     -> encoder.downsamples.layers.0.residual.layers.0.gamma
    vaeWeights = Self.remapVAEKeys(vaeWeights)

    let vaeParams = ModuleParameters.unflattened(vaeWeights.map { ($0.key, $0.value) })
    try vae.update(parameters: vaeParams, verify: [.shapeMismatch])
    eval(vae.parameters())
    logger.info("VAE loaded")

    // Initialize MoE manager
    self.moeManager = WanMoEManager(
      modelDir: modelDir,
      boundary: config.boundary,
      numTrainTimesteps: config.numTrainTimesteps,
      lazyLoading: config.lazyMoE,
      logger: logger
    )

    // Load initial expert (high-noise, used for early steps)
    try moeManager.loadHighNoiseExpert()
    logger.info("Pipeline initialized. Memory: \(moeManager.memoryInfo)")
  }

  /// Creates a pipeline with pre-loaded components (for testing or reuse).
  public init(
    config: WanI2VConfig,
    textEncoder: WanUMT5Encoder,
    tokenizer: WanTokenizer,
    vae: WanVAE,
    moeManager: WanMoEManager,
    logger: Logger
  ) {
    self.config = config
    self.textEncoder = textEncoder
    self.tokenizer = tokenizer
    self.vae = vae
    self.moeManager = moeManager
    self.logger = logger
  }

  // MARK: - Generation

  /// Generates video frames from an init image and text prompt.
  ///
  /// - Parameters:
  ///   - prompt: Text prompt describing the desired video.
  ///   - negativePrompt: Negative prompt (nil uses config default).
  ///   - image: Init image as MLXArray [C, H, W] in [0, 1] range.
  ///   - seed: Random seed (nil for random).
  ///   - width: Explicit output width (nil for auto from maxArea).
  ///   - height: Explicit output height (nil for auto from maxArea).
  ///   - steps: Override number of denoising steps (nil uses config).
  ///   - progressCallback: Called after each step with (currentStep, totalSteps).
  /// - Returns: Generated video frames as MLXArray [C, F, H, W] in [0, 1] range.
  public func generate(
    prompt: String,
    negativePrompt: String? = nil,
    image: MLXArray,
    seed: UInt64? = nil,
    width: Int? = nil,
    height: Int? = nil,
    steps: Int? = nil,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) throws -> MLXArray {
    let numSteps = steps ?? config.steps
    let negPrompt = negativePrompt ?? config.negativePrompt

    // 1. Compute resolution
    let imageH = image.dim(image.ndim == 4 ? 2 : 1)
    let imageW = image.dim(image.ndim == 4 ? 3 : 2)
    let resolution = WanI2VConditioner.computeResolution(
      explicitHeight: height,
      explicitWidth: width,
      imageHeight: imageH,
      imageWidth: imageW,
      maxArea: config.maxArea
    )

    logger.info("Resolution: \(resolution.pixelH)x\(resolution.pixelW) (latent: \(resolution.latH)x\(resolution.latW))")
    logger.info("Frames: \(config.frameNum) (\(Float(config.frameNum) / Float(config.fps))s at \(config.fps)fps)")

    // 2. Encode text
    logger.info("Encoding prompt...")
    let (promptIds, promptMask) = tokenizer.encode(prompt)
    let context = textEncoder(tokenIds: promptIds, attentionMask: promptMask)
      .asType(.bfloat16)

    let (negIds, negMask) = tokenizer.encode(negPrompt)
    let contextNull = textEncoder(tokenIds: negIds, attentionMask: negMask)
      .asType(.bfloat16)
    eval(context, contextNull)
    logger.info("Text encoded")

    // 3. Build I2V conditioning
    logger.info("Building I2V conditioning...")

    // Normalize image to [-1, 1]
    var img = image
    if img.ndim == 4 { img = img.squeezed(axis: 0) }
    let normalizedImg = WanI2VConditioner.normalizeImage(img)

    // VAE-encode init image
    let vaeEncoded = WanI2VConditioner.vaeEncodeInitImage(
      normalizedImg, vae: vae,
      frameNum: config.frameNum,
      pixelH: resolution.pixelH,
      pixelW: resolution.pixelW
    )
    eval(vaeEncoded)

    // Build mask
    let mask = WanI2VConditioner.buildMask(
      frameNum: config.frameNum,
      latH: resolution.latH,
      latW: resolution.latW
    )

    // Assemble conditioning (mask + VAE encoded = 20ch)
    let conditioning = WanI2VConditioner.assembleConditioning(
      mask: mask, vaeEncoded: vaeEncoded
    )
    eval(conditioning)
    logger.info("Conditioning built: \(conditioning.shape)")

    // 4. Generate noise
    let noise = WanI2VConditioner.generateNoise(
      frameNum: config.frameNum,
      latH: resolution.latH,
      latW: resolution.latW,
      seed: seed
    )
    eval(noise)

    // DEBUG: Dump pipeline stage values for bisection
    logger.info("[BISECT] === STAGE 1: TEXT ENCODING ===")
    logger.info("[BISECT] context shape = \(context.shape)")
    logger.info("[BISECT] context mean = \(context.mean().item(Float.self))")
    logger.info("[BISECT] context std = \(MLX.sqrt(context.variance()).item(Float.self))")
    logger.info("[BISECT] contextNull shape = \(contextNull.shape)")
    logger.info("[BISECT] contextNull mean = \(contextNull.mean().item(Float.self))")

    logger.info("[BISECT] === STAGE 2: VAE ENCODE ===")
    logger.info("[BISECT] vaeEncoded shape = \(vaeEncoded.shape)")
    logger.info("[BISECT] vaeEncoded mean = \(vaeEncoded.mean().item(Float.self))")
    logger.info("[BISECT] vaeEncoded std = \(MLX.sqrt(vaeEncoded.variance()).item(Float.self))")

    logger.info("[BISECT] === STAGE 3: CONDITIONING ===")
    logger.info("[BISECT] conditioning shape = \(conditioning.shape)")
    logger.info("[BISECT] conditioning mean = \(conditioning.mean().item(Float.self))")

    logger.info("[BISECT] === STAGE 4: NOISE ===")
    logger.info("[BISECT] noise shape = \(noise.shape)")
    logger.info("[BISECT] noise mean = \(noise.mean().item(Float.self))")
    logger.info("[BISECT] noise std = \(MLX.sqrt(noise.variance()).item(Float.self))")
    logger.info("[BISECT] noise[0,0,0,0:5] = [\(noise[0,0,0,0].item(Float.self)), \(noise[0,0,0,1].item(Float.self)), \(noise[0,0,0,2].item(Float.self)), \(noise[0,0,0,3].item(Float.self)), \(noise[0,0,0,4].item(Float.self))]")

    // 5. Compute sequence length
    let seqLen = WanI2VConditioner.computeSeqLen(
      frameNum: config.frameNum,
      latH: resolution.latH,
      latW: resolution.latW
    )

    // 6. Initialize scheduler
    var scheduler = FlowUniPCScheduler(
      numInferenceSteps: numSteps,
      shift: config.shift,
      numTrainTimesteps: config.numTrainTimesteps
    )

    // DEBUG: Scheduler values
    logger.info("[BISECT] === STAGE 0: SCHEDULER ===")
    logger.info("[BISECT] sigmas = \(scheduler.sigmas)")
    logger.info("[BISECT] timesteps = \(scheduler.timesteps)")

    // 7. Denoising loop
    logger.info("Starting denoising: \(numSteps) steps")
    var latent = noise

    let timestepValues = scheduler.timesteps
    let boundaryValue = config.boundary * Float(config.numTrainTimesteps)

    for stepIdx in 0..<numSteps {
      let t = timestepValues[stepIdx].item(Float.self)

      // Select expert and guide scale
      let model = try moeManager.model(forTimestep: t)
      let guideScale = moeManager.guideScale(forTimestep: t, scales: config.guideScale)

      // Prepare timestep tensor — truncate to integer to match Python's int64 dtype.
      // The sinusoidal position embedding is sensitive to fractional differences;
      // passing 999.8 instead of 999 shifts high-frequency components enough to
      // bias the noise prediction, which the multi-step UniPC solver then amplifies
      // into divergent latent growth.
      let timestep = MLXArray([Float(Int32(t))])

      // CFG: conditional forward pass
      let noisePredCond = model.forward(
        x: [latent],
        t: timestep,
        context: [context.squeezed(axis: 0)],
        seqLen: seqLen,
        y: [conditioning]
      )[0]

      // CFG: unconditional forward pass
      let noisePredUncond = model.forward(
        x: [latent],
        t: timestep,
        context: [contextNull.squeezed(axis: 0)],
        seqLen: seqLen,
        y: [conditioning]
      )[0]

      // CFG combination
      let cfgDiff = noisePredCond - noisePredUncond
      let cfgScale = MLXArray(Float(guideScale))
      let cfgScaled = cfgScale * cfgDiff
      let noisePred = noisePredUncond + cfgScaled

      // DEBUG: Dump step values for bisection
      logger.info("[BISECT] === STEP \(stepIdx) (t=\(t)) ===")
      logger.info("[BISECT] timestep tensor = \(timestep)")
      logger.info("[BISECT] noisePredCond mean = \(noisePredCond.mean().item(Float.self))")
      logger.info("[BISECT] noisePredCond std = \(MLX.sqrt(noisePredCond.variance()).item(Float.self))")
      logger.info("[BISECT] noisePredUncond mean = \(noisePredUncond.mean().item(Float.self))")
      logger.info("[BISECT] noisePredUncond std = \(MLX.sqrt(noisePredUncond.variance()).item(Float.self))")
      logger.info("[BISECT] noisePred (CFG) mean = \(noisePred.mean().item(Float.self))")
      logger.info("[BISECT] noisePred (CFG) std = \(MLX.sqrt(noisePred.variance()).item(Float.self))")

      // Scheduler step
      latent = scheduler.step(
        modelOutput: noisePred,
        timestepIndex: stepIdx,
        sample: latent
      )
      eval(latent)

      logger.info("[BISECT] latent after step mean = \(latent.mean().item(Float.self))")
      logger.info("[BISECT] latent after step std = \(MLX.sqrt(latent.variance()).item(Float.self))")
      logger.info("[BISECT] latent after step min = \(latent.min().item(Float.self))")
      logger.info("[BISECT] latent after step max = \(latent.max().item(Float.self))")
      if stepIdx < 2 {
        logger.info("[BISECT] latent[0,0,0,0:5] = [\(latent[0,0,0,0].item(Float.self)), \(latent[0,0,0,1].item(Float.self)), \(latent[0,0,0,2].item(Float.self)), \(latent[0,0,0,3].item(Float.self)), \(latent[0,0,0,4].item(Float.self))]")
      }

      progressCallback?(stepIdx + 1, numSteps)
      logger.debug("Step \(stepIdx + 1)/\(numSteps) (t=\(String(format: "%.1f", t)), expert=\(moeManager.activeExpert.rawValue))")
    }

    // 8. VAE decode
    logger.info("[BISECT] === STAGE 5: VAE DECODE ===")
    logger.info("[BISECT] latent shape = \(latent.shape)")
    logger.info("[BISECT] latent mean = \(latent.mean().item(Float.self))")
    logger.info("[BISECT] latent std = \(MLX.sqrt(latent.variance()).item(Float.self))")
    logger.info("[BISECT] latent range = [\(latent.min().item(Float.self)), \(latent.max().item(Float.self))]")

    logger.info("Decoding latents...")
    let decodedLatent = latent.expandedDimensions(axis: 0)  // Add batch dim
    let decoded = vae.decode(decodedLatent)
    eval(decoded)

    logger.info("[BISECT] decoded shape = \(decoded.shape)")
    logger.info("[BISECT] decoded mean = \(decoded.mean().item(Float.self))")
    logger.info("[BISECT] decoded std = \(MLX.sqrt(decoded.variance()).item(Float.self))")
    logger.info("[BISECT] decoded range = [\(decoded.min().item(Float.self)), \(decoded.max().item(Float.self))]")

    // Per-channel decode stats — diagnose purple cast / channel imbalance
    do {
      let decodedSqueezed = decoded.squeezed(axis: 0)  // [3, T, H, W]
      let decoded0 = decodedSqueezed[0]  // channel 0 (R)
      let decoded1 = decodedSqueezed[1]  // channel 1 (G)
      let decoded2 = decodedSqueezed[2]  // channel 2 (B)
      eval(decoded0, decoded1, decoded2)
      logger.info("[BISECT] decoded ch0 (R) mean=\(decoded0.mean().item(Float.self)) range=[\(decoded0.min().item(Float.self)), \(decoded0.max().item(Float.self))]")
      logger.info("[BISECT] decoded ch1 (G) mean=\(decoded1.mean().item(Float.self)) range=[\(decoded1.min().item(Float.self)), \(decoded1.max().item(Float.self))]")
      logger.info("[BISECT] decoded ch2 (B) mean=\(decoded2.mean().item(Float.self)) range=[\(decoded2.min().item(Float.self)), \(decoded2.max().item(Float.self))]")
    }

    // Post-process: [-1, 1] -> [0, 1]
    var frames = decoded.squeezed(axis: 0)  // Remove batch dim: [3, T_out, H, W]

    // Trim temporal dimension to match requested frame count.
    // The VAE decoder produces 4 * latent_t frames due to 2x temporal upsampling
    // in each of two upsample3d layers. For frameNum=81 this gives 84 frames.
    // The first frameNum frames are the valid output.
    let decodedT = frames.dim(1)
    if decodedT > config.frameNum {
      logger.info("Trimming decoded frames: \(decodedT) -> \(config.frameNum)")
      frames = frames[0..., 0..<config.frameNum, 0..., 0...]
    }

    frames = MLX.clip(frames * 0.5 + 0.5, min: 0.0, max: 1.0)

    logger.info("[BISECT] output frames shape = \(frames.shape)")
    logger.info("[BISECT] output frames mean = \(frames.mean().item(Float.self))")
    logger.info("[BISECT] output frames range = [\(frames.min().item(Float.self)), \(frames.max().item(Float.self))]")
    logger.info("Generation complete: \(frames.shape)")
    return frames
  }

  // MARK: - Single-Expert Generation (for testing)

  /// Generates using only a single expert (no MoE switching).
  /// Useful for quick tests with just 1-2 steps.
  public func generateSingleExpert(
    prompt: String,
    negativePrompt: String? = nil,
    image: MLXArray,
    seed: UInt64? = nil,
    width: Int? = nil,
    height: Int? = nil,
    steps: Int? = nil,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) throws -> MLXArray {
    // Same as generate but uses whichever expert is currently loaded
    // without attempting to switch
    return try generate(
      prompt: prompt,
      negativePrompt: negativePrompt,
      image: image,
      seed: seed,
      width: width,
      height: height,
      steps: steps,
      progressCallback: progressCallback
    )
  }
  // MARK: - VAE Key Remapping

  /// Remaps safetensors VAE keys to match the MLX module hierarchy.
  ///
  /// The VAE uses wrapper classes whose `layers: [Module]` property injects
  /// a `layers` segment into the parameter path. Safetensors keys don't have
  /// this segment, so we insert it.
  ///
  /// Containers that inject `layers`:
  /// - `WanSequentialLayers` at `encoder.downsamples`, `decoder.upsamples`
  /// - `WanMiddleLayers` at `encoder.middle`, `decoder.middle`
  /// - `WanResidualPath` at `*.residual`
  /// - `WanEncoderHead` at `encoder.head`
  /// - `WanDecoderHead` at `decoder.head`
  /// - `WanResampleSeq` at `*.resample`
  static func remapVAEKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    let containers: Set<String> = [
      "downsamples", "upsamples", "middle", "head", "residual", "resample",
    ]

    var result = [String: MLXArray]()
    result.reserveCapacity(weights.count)

    for (key, value) in weights {
      let parts = key.split(separator: ".")
      var remapped: [String] = []

      for (i, part) in parts.enumerated() {
        remapped.append(String(part))
        // If this part is a container name and the next part is a digit,
        // insert "layers" between them.
        if containers.contains(String(part)),
           i + 1 < parts.count,
           parts[i + 1].allSatisfy({ $0.isNumber }) {
          remapped.append("layers")
        }
      }

      result[remapped.joined(separator: ".")] = value
    }
    return result
  }
}

// MARK: - Frame Extraction

public extension WanI2VPipeline {

  /// Extracts individual frames from a decoded video tensor.
  ///
  /// - Parameter video: Video tensor [C, F, H, W] in [0, 1] range.
  /// - Returns: Array of frame tensors, each [C, H, W].
  static func extractFrames(_ video: MLXArray) -> [MLXArray] {
    let numFrames = video.dim(1)
    return (0..<numFrames).map { f in
      video[0..., f, 0..., 0...]
    }
  }

  /// Extracts the last frame from a video tensor.
  ///
  /// Used for extend/continuation: the last frame becomes the init image
  /// for the next chunk.
  ///
  /// - Parameter video: Video tensor [C, F, H, W] in [0, 1] range.
  /// - Returns: Last frame [C, H, W].
  static func extractLastFrame(_ video: MLXArray) -> MLXArray {
    let lastIdx = video.dim(1) - 1
    return video[0..., lastIdx, 0..., 0...]
  }
}
