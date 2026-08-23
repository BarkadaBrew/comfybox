import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom
import Tokenizers
import Hub
import Dispatch

public struct ZImageGenerationRequest: Sendable {
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
  public var model: String?
  public var textEncoderPath: String?
  public var maxSequenceLength: Int

  public var loras: [LoRAConfiguration]
  public var lora: LoRAConfiguration? {
    get { loras.first }
    set { loras = newValue.map { [$0] } ?? [] }
  }

  /// Which app/persona requested this render (desktop/bree/kira/api…) — embedded
  /// so the gallery can section persona renders.
  public var source: String?

  /// Fruit mode (neutral | banana | avocado) — stamped into render metadata.
  public var contentMode: String?

  /// Generation params embedded into the saved PNG (Finder/Spotlight-readable),
  /// so every render carries its own sidecar — the mflux-style default.
  public func embeddedMetadata(loras: [LoRAConfiguration] = []) -> QwenImageIO.ImageMetadata {
    .generation(prompt: prompt,
                negativePrompt: QwenImageIO.ImageMetadata.requestNegative(negativePrompt),
                seed: seed,
                steps: steps, guidance: guidanceScale, width: width, height: height,
                model: model, generatedBy: source, contentMode: contentMode, loras: loras)
  }

  public var enhancePrompt: Bool

  public var enhanceMaxTokens: Int
  public var forceTransformerOverrideOnly: Bool

  /// The sampler algorithm to use for denoising. Default: `.euler`.
  public var schedulerKind: SchedulerKind

  /// The sigma schedule for noise level progression. Default: `.flow`.
  public var sigmaSchedule: SigmaScheduleKind

  /// Stochasticity parameter for DDIM (0 = deterministic, 1 = DDPM).
  /// Also used by DPM++ 2S-A. Ignored by other samplers.
  public var eta: Float?

  /// DyPE (Dynamic Position Extrapolation) config for native high-resolution generation.
  /// When enabled, modifies RoPE frequencies to support resolutions above training scale.
  public var dyPE: DyPEConfig

  // --- Latent-space inpainting (Phase 3) ---

  /// Raw PNG data of the image to inpaint. When set, enables latent-space inpainting.
  public var inpaintImageData: Data?
  /// Raw PNG data of the mask. White (255) = regenerate, black (0) = preserve original.
  public var maskData: Data?
  /// Denoising strength for inpainting (0.0–1.0). Lower values preserve more of the original.
  /// Default: 1.0 (full denoise, equivalent to txt2img).
  public var denoise: Float
  /// Mask expansion in pixels (default 0 = no expansion).
  public var maskGrow: Int
  /// Mask feather radius in pixels (default 0 = hard edges).
  public var maskFeather: Int
  /// ImageCrop x,y offset for cropping full-canvas mask to selection bounds.
  public var maskCropX: Int
  public var maskCropY: Int

  public init(
    prompt: String,
    negativePrompt: String? = nil,
    width: Int = ZImageModelMetadata.recommendedWidth,
    height: Int = ZImageModelMetadata.recommendedHeight,
    steps: Int = ZImageModelMetadata.recommendedInferenceSteps,
    guidanceScale: Float = ZImageModelMetadata.recommendedGuidanceScale,
    seed: UInt64? = nil,
    outputPath: URL = URL(fileURLWithPath: "z-image.png"),
    levelsMin: Float = 0.0,
    levelsMax: Float = 1.0,
    model: String? = nil,
    source: String? = nil,
    contentMode: String? = nil,
    textEncoderPath: String? = nil,
    maxSequenceLength: Int = 512,
    lora: LoRAConfiguration? = nil,
    loras: [LoRAConfiguration] = [],
    enhancePrompt: Bool = false,
    enhanceMaxTokens: Int = 512,
    forceTransformerOverrideOnly: Bool = false,
    schedulerKind: SchedulerKind = .euler,
    sigmaSchedule: SigmaScheduleKind = .flow,
    eta: Float? = nil,
    dyPE: DyPEConfig = .disabled,
    inpaintImageData: Data? = nil,
    maskData: Data? = nil,
    denoise: Float = 1.0,
    maskGrow: Int = 0,
    maskFeather: Int = 0,
    maskCropX: Int = 0,
    maskCropY: Int = 0
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
    self.model = model
    self.source = source
    self.contentMode = contentMode
    self.textEncoderPath = textEncoderPath
    self.maxSequenceLength = maxSequenceLength
    self.loras = loras.isEmpty ? (lora.map { [$0] } ?? []) : loras
    self.enhancePrompt = enhancePrompt
    self.enhanceMaxTokens = enhanceMaxTokens
    self.forceTransformerOverrideOnly = forceTransformerOverrideOnly
    self.schedulerKind = schedulerKind
    self.sigmaSchedule = sigmaSchedule
    self.eta = eta
    self.dyPE = dyPE
    self.inpaintImageData = inpaintImageData
    self.maskData = maskData
    self.denoise = denoise
    self.maskGrow = maskGrow
    self.maskFeather = maskFeather
    self.maskCropX = maskCropX
    self.maskCropY = maskCropY
  }
}

// Pipeline instances cache mutable MLX state and are not thread-safe.
// Callers should serialize access to a given pipeline instance.
public final class ZImagePipeline {
  public enum RetentionPolicy: Sendable {
    case releaseAfterRender
    case keepLoaded
  }

  public enum PipelineError: Error, Sendable {
    case notImplemented
    case tokenizerNotLoaded
    case invalidDimensions(String)
    case textEncoderNotLoaded
    case transformerNotLoaded
    case vaeNotLoaded
    case weightsMissing(String)
    case modelNotLoaded
    case loraError(LoRAError)
  }

  private var logger: Logger
  private let hubApi: HubApi
  private let retentionPolicy: RetentionPolicy
  private var tokenizer: QwenTokenizer?
  private var textEncoder: QwenTextEncoder?
  private var transformer: ZImageTransformer2DModel?
  private var vae: VAEImageDecoding?
  /// Full VAE with encoder — lazily loaded on first inpaint request.
  private var fullVAE: AutoencoderKL?
  private var modelConfigs: ZImageModelConfigs?
  private var quantManifest: ZImageQuantizationManifest?
  private var isModelLoaded: Bool = false
  private var loadedModelId: String?
  private var modelSnapshot: URL?
  private var useDynamicLoRA: Bool = false
  private var activeTransformerOverrideURL: URL?
  private var activeAIOCheckpointURL: URL?
  private var activeCivitAICheckpointURL: URL?
  private var activeCivitAIVariant: ZImageVariant?
  private var loadedTextEncoderSelection: TextEncoderSelection?

  // Stored behind pipeline-local serialized access only. LoRAWeights contains MLXArray.
  private struct AppliedLoRA: @unchecked Sendable {
    let weights: LoRAWeights
    let configuration: LoRAConfiguration
  }

  private var currentLoRAs: [AppliedLoRA] = []

  public init(
    logger: Logger = Logger(label: "z-image.pipeline"),
    hubApi: HubApi = .shared,
    retentionPolicy: RetentionPolicy = .releaseAfterRender
  ) {
    self.logger = logger
    self.hubApi = hubApi
    self.retentionPolicy = retentionPolicy
  }
  public var isLoaded: Bool {
    return isModelLoaded
  }
  public func unloadModel() {
    tokenizer = nil
    textEncoder = nil
    transformer = nil
    vae = nil
    fullVAE = nil
    modelConfigs = nil
    quantManifest = nil
    isModelLoaded = false
    loadedModelId = nil

    currentLoRAs.removeAll()
    modelSnapshot = nil
    useDynamicLoRA = false
    activeTransformerOverrideURL = nil
    activeAIOCheckpointURL = nil
    activeCivitAICheckpointURL = nil
    activeCivitAIVariant = nil
    loadedTextEncoderSelection = nil
    GPU.clearCache()
    logger.info("Model unloaded from memory")
  }

  public func unloadLoRA() {
    guard !currentLoRAs.isEmpty else { return }

    if let trans = transformer {
      for appliedLoRA in currentLoRAs where appliedLoRA.weights.hasLoKr {
        LoRAApplicator.removeLoKr(
          from: trans,
          loraWeights: appliedLoRA.weights,
          scale: appliedLoRA.configuration.scale,
          logger: logger
        )
      }
      LoRAApplicator.clearDynamicLoRA(from: trans, logger: logger)
    }
    let unloadedCount = currentLoRAs.count
    currentLoRAs.removeAll()
    useDynamicLoRA = false
    GPU.clearCache()
    logger.info("LoRA unloaded (\(unloadedCount) adapter(s))")
  }
  public func unloadTransformer() {
    transformer = nil

    currentLoRAs.removeAll()
    useDynamicLoRA = false
    activeTransformerOverrideURL = nil

    GPU.clearCache()
    logger.info("Transformer unloaded for memory optimization")
  }

  private func getAvailableMemory() -> UInt64 {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &stats) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
      }
    }
    guard result == KERN_SUCCESS else { return 0 }
    let pageSize = UInt64(sysconf(_SC_PAGESIZE))
    return UInt64(stats.free_count) * pageSize
  }

  private func loadTokenizer(snapshot: URL) throws -> QwenTokenizer {
    let tokDir = snapshot.appending(path: "tokenizer")
    return try QwenTokenizer.load(from: tokDir, hubApi: hubApi)
  }

  private func loadTextEncoder(snapshot: URL, config: ZImageTextEncoderConfig) throws -> QwenTextEncoder {
    return QwenTextEncoder(
      configuration: .init(
        vocabSize: config.vocabSize,
        hiddenSize: config.hiddenSize,
        numHiddenLayers: config.numHiddenLayers,
        numAttentionHeads: config.numAttentionHeads,
        numKeyValueHeads: config.numKeyValueHeads,
        intermediateSize: config.intermediateSize,
        ropeTheta: config.ropeTheta,
        maxPositionEmbeddings: config.maxPositionEmbeddings,
        rmsNormEps: config.rmsNormEps,
        headDim: config.headDim
      )
    )
  }

  private func loadTransformer(snapshot: URL, config: ZImageTransformerConfig) throws -> ZImageTransformer2DModel {
    return ZImageTransformer2DModel(configuration: config)
  }

  private func auditModuleWeightShapeMismatches(
    module: Module,
    weights: [String: MLXArray],
    transpose4DTensors: Bool,
    logger: Logger,
    sample: Int = 10
  ) -> [String] {
    let params = module.parameters().flattened()
    var mismatches: [String] = []
    mismatches.reserveCapacity(8)

    for (key, param) in params {
      guard var tensor = weights[key] else { continue }
      if transpose4DTensors && tensor.ndim == 4 {
        tensor = ZImageWeightsMapping.alignTensorShape(tensor, to: param.shape)
      }
      if tensor.shape != param.shape {
        mismatches.append("\(key) expected \(param.shape) got \(tensor.shape)")
      }
    }

    if !mismatches.isEmpty {
      let sampleList = mismatches.prefix(max(0, sample)).joined(separator: "; ")
      let suffix = mismatches.count > sample ? "; ..." : ""
      logger.warning("Found \(mismatches.count) weight shape mismatches (sample: \(sampleList)\(suffix))")
    }

    return mismatches
  }

  private func loadVAEDecoder(snapshot: URL, config: ZImageVAEConfig) throws -> AutoencoderDecoderOnly {
    AutoencoderDecoderOnly(configuration: .init(
      inChannels: config.inChannels,
      outChannels: config.outChannels,
      latentChannels: config.latentChannels,
      scalingFactor: config.scalingFactor,
      shiftFactor: config.shiftFactor,
      blockOutChannels: config.blockOutChannels,
      layersPerBlock: config.layersPerBlock,
      normNumGroups: config.normNumGroups,
      sampleSize: config.sampleSize,
      midBlockAddAttention: config.midBlockAddAttention
    ))
  }

  /// Load the full VAE with encoder for inpainting.
  /// Called lazily on first inpaint request. Encoder weights are loaded from the same snapshot.
  private func ensureFullVAE() throws -> AutoencoderKL {
    if let existing = fullVAE { return existing }

    guard let configs = modelConfigs else {
      throw PipelineError.modelNotLoaded
    }
    guard let snapshot = modelSnapshot else {
      throw PipelineError.modelNotLoaded
    }

    logger.info("Loading full VAE with encoder for inpainting...")
    let vaeConfig = configs.vae
    let v = AutoencoderKL(configuration: .init(
      inChannels: vaeConfig.inChannels,
      outChannels: vaeConfig.outChannels,
      latentChannels: vaeConfig.latentChannels,
      scalingFactor: vaeConfig.scalingFactor,
      shiftFactor: vaeConfig.shiftFactor,
      blockOutChannels: vaeConfig.blockOutChannels,
      layersPerBlock: vaeConfig.layersPerBlock,
      normNumGroups: vaeConfig.normNumGroups,
      sampleSize: vaeConfig.sampleSize,
      midBlockAddAttention: vaeConfig.midBlockAddAttention
    ))

    // Load ALL VAE weights (encoder + decoder)
    let weightsMapper = ZImageWeightsMapper(snapshot: snapshot, logger: logger)
    let allVAEWeights = try weightsMapper.loadVAE(dtype: .float32)
    try ZImageWeightsMapping.applyVAE(weights: allVAEWeights, to: v, manifest: quantManifest, logger: logger)

    fullVAE = v
    logger.info("Full VAE loaded (encoder + decoder)")
    return v
  }


  /// Pre-load the full VAE with encoder so that the first img2img request
  /// does not need to perform synchronous weight loading mid-render.
  ///
  /// Call this during warm server startup (after prepare) to avoid
  /// a potential deadlock when ensureFullVAE runs inside the actor-
  /// isolated render path. The deadlock occurs because synchronous weight
  /// loading inside an async actor method can starve the cooperative
  /// thread pool, preventing MLX.eval completion handlers from running.
  public func prepareFullVAE() throws {
    _ = try ensureFullVAE()
  }

  private func encodePrompt(_ prompt: String, tokenizer: QwenTokenizer, textEncoder: QwenTextEncoder, maxLength: Int) throws -> (MLXArray, MLXArray) {
    do {
      let promptEncodingMode = ZImageFiles.resolvePromptEncodingMode(
        at: modelSnapshot,
        selection: loadedTextEncoderSelection
      )
      let result = try PipelineUtilities.encodePrompt(
        prompt,
        tokenizer: tokenizer,
        textEncoder: textEncoder,
        maxLength: maxLength,
        mode: promptEncodingMode
      )
      return (result.embeddings, result.mask)
    } catch {
      logger.error("Prompt encoding failed: \(String(describing: error))")
      throw PipelineError.textEncoderNotLoaded
    }
  }

  struct ModelSelection: Sendable {
    var baseModelSpec: String?
    var transformerOverrideURL: URL?
    var aioCheckpointURL: URL?
    var aioTextEncoderPrefix: String?
    var civitaiCheckpointURL: URL?
    var civitaiVariant: ZImageVariant?
  }

  func resolveModelSelection(_ modelSpec: String?, forceTransformerOverrideOnly: Bool) -> ModelSelection {
    guard let modelSpec else { return .init(baseModelSpec: nil, transformerOverrideURL: nil, aioCheckpointURL: nil, aioTextEncoderPrefix: nil) }

    let candidateURL = URL(fileURLWithPath: modelSpec)
    var isDir: ObjCBool = false
    if FileManager.default.fileExists(atPath: candidateURL.path, isDirectory: &isDir) {
      if !isDir.boolValue && candidateURL.pathExtension == "safetensors" {
        return resolveLocalSafetensors(candidateURL, forceTransformerOverrideOnly: forceTransformerOverrideOnly)
      }
      if isDir.boolValue {
        if ZImageFiles.hasRecognizableModelDirectory(at: candidateURL) {
          return .init(baseModelSpec: modelSpec, transformerOverrideURL: nil, aioCheckpointURL: nil, aioTextEncoderPrefix: nil)
        }

        let contents = (try? FileManager.default.contentsOfDirectory(at: candidateURL, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        let safes = contents.filter { $0.pathExtension == "safetensors" }
        if safes.isEmpty {
          logger.warning("Model path is a directory without expected configs or safetensors: \(modelSpec). Falling back to default model.")
          return .init(baseModelSpec: nil, transformerOverrideURL: nil, aioCheckpointURL: nil, aioTextEncoderPrefix: nil)
        }

        let preferred = safes.first(where: { $0.lastPathComponent.lowercased().contains("v2") })
          ?? safes.max(by: { a, b in
            let sa = (try? a.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            let sb = (try? b.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return sa < sb
          })

        guard let preferred else {
          return .init(baseModelSpec: nil, transformerOverrideURL: nil, aioCheckpointURL: nil, aioTextEncoderPrefix: nil)
        }
        return resolveLocalSafetensors(preferred, forceTransformerOverrideOnly: forceTransformerOverrideOnly, sourceDirectory: candidateURL)
      }

      return .init(baseModelSpec: modelSpec, transformerOverrideURL: nil, aioCheckpointURL: nil, aioTextEncoderPrefix: nil)
    }

    return .init(baseModelSpec: modelSpec, transformerOverrideURL: nil, aioCheckpointURL: nil, aioTextEncoderPrefix: nil)
  }

  private func resolveLocalSafetensors(
    _ url: URL,
    forceTransformerOverrideOnly: Bool,
    sourceDirectory: URL? = nil
  ) -> ModelSelection {
    if !forceTransformerOverrideOnly {
      let inspection = ZImageAIOCheckpoint.inspect(fileURL: url)
      if inspection.isAIO, let prefix = inspection.textEncoderPrefix {
        if let sourceDirectory {
          logger.info("Detected AIO checkpoint in \(sourceDirectory.lastPathComponent): \(url.lastPathComponent). Bypassing base model weights.")
        } else {
          logger.info("Detected AIO checkpoint: \(url.lastPathComponent). Bypassing base model weights.")
        }
        return .init(baseModelSpec: nil, transformerOverrideURL: nil, aioCheckpointURL: url, aioTextEncoderPrefix: prefix)
      }
    }

    // CivitAI transformer-only detection
    if !forceTransformerOverrideOnly {
      let civitaiInspection = CivitAICheckpoint.inspect(fileURL: url)
      if civitaiInspection.isCivitAI {
        let variant = civitaiInspection.variant ?? .turbo
        logger.info("Detected CivitAI checkpoint: \(url.lastPathComponent) (variant=\(variant.rawValue), keys=\(civitaiInspection.keyCount))")
        // Use the correct base model for the detected variant so that the
        // transformer config (nRefinerLayers, etc.) matches the checkpoint
        // architecture.  Without this, a Base CivitAI checkpoint defaults
        // to the Turbo snapshot config, creating a transformer without
        // noise_refiner / context_refiner blocks.  That mismatch causes
        // crashes during subsequent LoRA swap operations (#138).
        let baseSpec = variant == .base ? ZImageRepository.baseId : ZImageRepository.id
        return .init(baseModelSpec: baseSpec, transformerOverrideURL: nil, aioCheckpointURL: nil, aioTextEncoderPrefix: nil, civitaiCheckpointURL: url, civitaiVariant: variant)
      }
    }

    if sourceDirectory != nil {
      logger.info("Using transformer override file from directory: \(url.lastPathComponent)")
    } else {
      logger.info("Using transformer override file: \(url.lastPathComponent)")
    }
    return .init(baseModelSpec: nil, transformerOverrideURL: url, aioCheckpointURL: nil, aioTextEncoderPrefix: nil)
  }

  private func applyTransformerOverrideIfNeeded(_ overrideURL: URL?) throws {
    guard overrideURL != activeTransformerOverrideURL else { return }
    guard activeAIOCheckpointURL == nil && activeCivitAICheckpointURL == nil else { return }
    guard let transformer, let snapshot = modelSnapshot, let configs = modelConfigs else { throw PipelineError.modelNotLoaded }

    let weightsMapper = ZImageWeightsMapper(snapshot: snapshot, logger: logger)
    let baseTransformerWeights = try weightsMapper.loadTransformer()
    try ZImageWeightsMapping.applyTransformer(weights: baseTransformerWeights, to: transformer, manifest: nil, logger: logger)

    activeTransformerOverrideURL = nil

    if let overrideURL {
      logger.info("Applying transformer override weights from: \(overrideURL.lastPathComponent)")
      var overrideWeights = try weightsMapper.loadTransformer(fromFile: overrideURL, dtype: .bfloat16)

      if let inferredDim = inferTransformerDim(from: overrideWeights), inferredDim != configs.transformer.dim {
        throw PipelineError.weightsMissing("Transformer override dim \(inferredDim) mismatches model dim \(configs.transformer.dim)")
      }

      overrideWeights = canonicalizeTransformerOverride(overrideWeights, dim: configs.transformer.dim, logger: logger)
      try ZImageWeightsMapping.applyTransformer(weights: overrideWeights, to: transformer, manifest: nil, logger: logger)
      activeTransformerOverrideURL = overrideURL
    }
  }

  public struct GenerationProgress: Sendable {
    public let stage: Stage
    public let stepIndex: Int
    public let totalSteps: Int

    public enum Stage: String, Sendable {
      case loadingModel = "Loading model"
      case encodingText = "Encoding text"
      case loadingTransformer = "Loading transformer"
      case loadingLoRA = "Loading LoRA"
      case denoising = "Denoising"
      case loadingVAE = "Loading VAE"
      case decoding = "Decoding"
      case saving = "Saving"
    }

    public var fractionCompleted: Double {
      guard totalSteps > 0 else { return 0 }
      return Double(stepIndex) / Double(totalSteps)
    }

    public var percentComplete: Int {
      Int(fractionCompleted * 100)
    }
  }

  public typealias ProgressHandler = (GenerationProgress) -> Void

  /// Optional callback for live denoising previews.
  /// Receives the current latent tensor, step index, total steps, and latent dimensions.
  /// Called from the denoising loop — implementations must not block.
  public typealias LatentPreviewHandler = @Sendable (MLXArray, Int, Int, Int, Int) -> Void

  public func loadModel(
    modelSpec: String? = nil,
    textEncoderPath: String? = nil,
    aioCheckpointURL: URL? = nil,
    aioTextEncoderPrefix: String? = nil,
    civitaiCheckpointURL: URL? = nil,
    civitaiVariant: ZImageVariant? = nil,
    progressHandler: ProgressHandler? = nil
  ) async throws {
    // When loading a CivitAI Base checkpoint, ensure the snapshot comes from
    // the Base model repo so the transformer config has nRefinerLayers > 0.
    let modelId: String
    if let modelSpec {
      modelId = modelSpec
    } else if civitaiCheckpointURL != nil, civitaiVariant == .base {
      modelId = ZImageRepository.baseId
    } else {
      modelId = ZImageRepository.id
    }
    let normalizedAIOPath = aioCheckpointURL?.standardizedFileURL.path
    let currentAIOPath = activeAIOCheckpointURL?.standardizedFileURL.path
    let normalizedCivitAIPath = civitaiCheckpointURL?.standardizedFileURL.path
    let currentCivitAIPath = activeCivitAICheckpointURL?.standardizedFileURL.path
    let hasLoadedComponents = tokenizer != nil && textEncoder != nil && transformer != nil && vae != nil && modelConfigs != nil && modelSnapshot != nil

    if isModelLoaded,
      loadedModelId == modelId,
      normalizedAIOPath == currentAIOPath,
      normalizedCivitAIPath == currentCivitAIPath,
      hasLoadedComponents,
      let cachedSnapshot = modelSnapshot
    {
      let cachedSelection = PipelineUtilities.resolveTextEncoderSelection(
        for: cachedSnapshot,
        overridePath: textEncoderPath
      )
      if loadedTextEncoderSelection?.directory.standardizedFileURL.path == cachedSelection.directory.standardizedFileURL.path {
        logger.info("Model already loaded, skipping load")
        return
      }
    }

    progressHandler?(GenerationProgress(stage: .loadingModel, stepIndex: 0, totalSteps: 1))
    let snapshotFilePatterns: [String]?
    if aioCheckpointURL != nil {
      snapshotFilePatterns = PipelineSnapshot.configAndTokenizerFilePatterns
    } else if civitaiCheckpointURL != nil {
      snapshotFilePatterns = PipelineSnapshot.configTokenizerTextEncoderAndVAEFilePatterns
    } else {
      snapshotFilePatterns = nil
    }
    // Use modelId (which accounts for CivitAI variant) instead of raw
    // modelSpec so that Base CivitAI checkpoints resolve the Base snapshot
    // (with nRefinerLayers > 0) rather than falling through to Turbo.
    let snapshotModelId = modelSpec ?? (modelId != ZImageRepository.id ? modelId : nil)
    let snapshot = try await PipelineSnapshot.prepare(model: snapshotModelId, filePatterns: snapshotFilePatterns, logger: logger)
    let textEncoderSelection = PipelineUtilities.resolveTextEncoderSelection(
      for: snapshot,
      overridePath: textEncoderPath,
      logger: logger
    )
    let loadedTextEncoderPath = loadedTextEncoderSelection?.directory.standardizedFileURL.path
    let selectedTextEncoderPath = textEncoderSelection.directory.standardizedFileURL.path
    if isModelLoaded
      && loadedModelId == modelId
      && normalizedAIOPath == currentAIOPath
      && normalizedCivitAIPath == currentCivitAIPath
      && loadedTextEncoderPath == selectedTextEncoderPath
      && hasLoadedComponents
    {
      logger.info("Model already loaded, skipping load")
      return
    }
    let canPreserveSharedComponents = isModelLoaded
      && loadedModelId != modelId
      && currentAIOPath == nil
      && normalizedAIOPath == nil
      && areZImageVariants(loadedModelId ?? "", modelId)
    if isModelLoaded && (loadedModelId != modelId || normalizedAIOPath != currentAIOPath || normalizedCivitAIPath != currentCivitAIPath) {
      if canPreserveSharedComponents {
        logger.info("Switching Z-Image variant, preserving VAE and tokenizer")

        textEncoder = nil
        transformer = nil

        currentLoRAs.removeAll()
        useDynamicLoRA = false
      } else {
        logger.info("Different model requested, unloading current model")
        unloadModel()
      }
    }

    logger.info("Loading model: \(modelId)")
    let configs = try ZImageModelConfigs.load(from: snapshot, textEncoderDirectory: textEncoderSelection.directory)
    if tokenizer == nil {
      progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: 1))
      logger.info("Loading tokenizer...")
      tokenizer = try loadTokenizer(snapshot: snapshot)
    } else {
      logger.info("Reusing cached tokenizer")
    }
    if let aioCheckpointURL {
      let textEncoderPrefix: String
      if let aioTextEncoderPrefix, !aioTextEncoderPrefix.isEmpty {
        textEncoderPrefix = aioTextEncoderPrefix
      } else {
        let inspection = ZImageAIOCheckpoint.inspect(fileURL: aioCheckpointURL)
        guard inspection.isAIO, let inferred = inspection.textEncoderPrefix else {
          let reason = inspection.diagnostics.isEmpty ? "unknown" : inspection.diagnostics.joined(separator: "; ")
          throw PipelineError.weightsMissing("Not a valid AIO checkpoint: \(aioCheckpointURL.lastPathComponent) (\(reason)). Use --force-transformer-override-only to treat it as transformer-only.")
        }
        textEncoderPrefix = inferred
      }

      logger.info("Loading AIO checkpoint weights from \(aioCheckpointURL.lastPathComponent)")
      let aio = try ZImageAIOCheckpoint.loadComponents(
        from: aioCheckpointURL,
        textEncoderPrefix: textEncoderPrefix,
        dtype: .bfloat16,
        vaeDType: .float32,
        logger: logger
      )

      logger.info("Loading text encoder...")
      let te = try loadTextEncoder(snapshot: snapshot, config: configs.textEncoder)
      try ZImageWeightsMapping.applyTextEncoder(weights: aio.textEncoder, to: te, manifest: nil, logger: logger)
      textEncoder = te

      progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 1))
      logger.info("Loading transformer...")
      let trans = try loadTransformer(snapshot: snapshot, config: configs.transformer)
      let transformerWeights = canonicalizeTransformerOverride(aio.transformer, dim: configs.transformer.dim, logger: logger)
      if let inferredDim = inferTransformerDim(from: transformerWeights), inferredDim != configs.transformer.dim {
        throw PipelineError.weightsMissing("AIO transformer dim \(inferredDim) mismatches model dim \(configs.transformer.dim)")
      }
      try validateStrictAIOTransformerWeights(transformerWeights, config: configs.transformer)
      try validateAIOTransformerCoverage(transformerWeights, transformer: trans)
      try ZImageWeightsMapping.applyTransformer(weights: transformerWeights, to: trans, manifest: nil, logger: logger)
      transformer = trans

      activeTransformerOverrideURL = nil
      activeAIOCheckpointURL = aioCheckpointURL
      quantManifest = nil

      if vae == nil {
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
        logger.info("Loading VAE...")
        let v = try loadVAEDecoder(snapshot: snapshot, config: configs.vae)
        let rawDecoderWeights = aio.vae.filter { $0.key.hasPrefix("decoder.") }
        let decoderWeights = ZImageAIOCheckpoint.canonicalizeVAEWeights(
          rawDecoderWeights,
          expectedUpBlocks: configs.vae.blockOutChannels.count,
          logger: logger
        )

        let audit = WeightsAudit.audit(module: v, weights: decoderWeights, logger: logger, sample: 10)
        let total = audit.matched + audit.missing.count
        let coverage = total > 0 ? Double(audit.matched) / Double(total) : 0.0
        let minimumCoverage = 0.99
        let mismatches = auditModuleWeightShapeMismatches(
          module: v,
          weights: decoderWeights,
          transpose4DTensors: true,
          logger: logger,
          sample: 10
        )

        if coverage >= minimumCoverage, mismatches.isEmpty {
          try ZImageWeightsMapping.applyVAE(weights: decoderWeights, to: v, manifest: nil, logger: logger)
        } else {
          let percent = Int((coverage * 100.0).rounded())
          if mismatches.isEmpty {
            logger.warning("AIO VAE decoder weights coverage too low: matched \(audit.matched)/\(total) (\(percent)%). Falling back to base VAE weights.")
          } else {
            logger.warning("AIO VAE decoder weights have incompatible shapes (coverage \(percent)%), falling back to base VAE weights.")
          }

          let baseVAESnapshot = try await PipelineSnapshot.prepare(
            model: modelSpec,
            filePatterns: PipelineSnapshot.vaeOnlyFilePatterns,
            logger: logger
          )
          let weightsMapper = ZImageWeightsMapper(snapshot: baseVAESnapshot, logger: logger)
          let baseVAEWeights = try weightsMapper.loadVAE(dtype: .float32)
          let baseDecoderWeights = baseVAEWeights.filter { $0.key.hasPrefix("decoder.") }
          try ZImageWeightsMapping.applyVAE(weights: baseDecoderWeights, to: v, manifest: nil, logger: logger)
        }
        vae = v
      } else {
        logger.info("Reusing cached VAE")
      }
    } else if let civitaiCheckpointURL {
      let variant = civitaiVariant ?? .turbo
      logger.info("Loading CivitAI checkpoint: \(civitaiCheckpointURL.lastPathComponent) (variant=\(variant.rawValue))")

      // Load transformer weights (raw, with model.diffusion_model. prefix)
      let rawWeights = try CivitAICheckpoint.loadTransformerWeights(
        from: civitaiCheckpointURL, dtype: .bfloat16, logger: logger)

      // Canonicalize keys (strip prefix, split QKV, remap names)
      let transformerWeights = canonicalizeTransformerOverride(rawWeights, dim: configs.transformer.dim, logger: logger)

      // Validate
      if let inferredDim = inferTransformerDim(from: transformerWeights), inferredDim != configs.transformer.dim {
        throw PipelineError.weightsMissing("CivitAI transformer dim \(inferredDim) mismatches model dim \(configs.transformer.dim)")
      }
      try validateStrictAIOTransformerWeights(transformerWeights, config: configs.transformer)

      progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 1))
      logger.info("Loading transformer architecture...")
      let trans = try loadTransformer(snapshot: snapshot, config: configs.transformer)
      try validateAIOTransformerCoverage(transformerWeights, transformer: trans)
      try ZImageWeightsMapping.applyTransformer(weights: transformerWeights, to: trans, manifest: nil, logger: logger)
      transformer = trans

      activeTransformerOverrideURL = nil
      activeAIOCheckpointURL = nil
      activeCivitAICheckpointURL = civitaiCheckpointURL
      activeCivitAIVariant = variant
      quantManifest = nil

      // Text encoder loaded from HuggingFace snapshot (standard path)
      let weightsMapper = ZImageWeightsMapper(
        snapshot: snapshot,
        logger: logger,
        textEncoderDirectory: textEncoderSelection.directory
      )
      let manifest = weightsMapper.loadQuantizationManifest()
      logger.info("Loading text encoder from HuggingFace snapshot...")
      let te = try loadTextEncoder(snapshot: snapshot, config: configs.textEncoder)
      let textEncoderWeights = try weightsMapper.loadTextEncoder()
      try ZImageWeightsMapping.applyTextEncoder(weights: textEncoderWeights, to: te, manifest: manifest, logger: logger)
      textEncoder = te

      // VAE loaded from HuggingFace snapshot (standard path)
      if vae == nil {
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
        logger.info("Loading VAE from HuggingFace snapshot...")
        let v = try loadVAEDecoder(snapshot: snapshot, config: configs.vae)
        let vaeWeights = try weightsMapper.loadVAE(dtype: .float32)
        let decoderWeights = vaeWeights.filter { $0.key.hasPrefix("decoder.") }
        try ZImageWeightsMapping.applyVAE(weights: decoderWeights, to: v, manifest: manifest, logger: logger)
        vae = v
      } else {
        logger.info("Reusing cached VAE")
      }
    } else {
      let weightsMapper = ZImageWeightsMapper(
        snapshot: snapshot,
        logger: logger,
        textEncoderDirectory: textEncoderSelection.directory
      )
      let manifest = weightsMapper.loadQuantizationManifest()

      if let m = manifest {
        logger.info("Loading quantized model (bits=\(m.bits), group_size=\(m.groupSize))")
      }
      logger.info("Loading text encoder...")
      let te = try loadTextEncoder(snapshot: snapshot, config: configs.textEncoder)
      let textEncoderWeights = try weightsMapper.loadTextEncoder()
      try ZImageWeightsMapping.applyTextEncoder(weights: textEncoderWeights, to: te, manifest: manifest, logger: logger)
      textEncoder = te
      progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 1))
      logger.info("Loading transformer...")
      let trans = try loadTransformer(snapshot: snapshot, config: configs.transformer)
      let transformerWeights = try weightsMapper.loadTransformer()
      try ZImageWeightsMapping.applyTransformer(weights: transformerWeights, to: trans, manifest: manifest, logger: logger)
      transformer = trans
      activeTransformerOverrideURL = nil
      activeAIOCheckpointURL = nil
      activeCivitAICheckpointURL = nil
      activeCivitAIVariant = nil
      if vae == nil {
        progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
        logger.info("Loading VAE...")
        let v = try loadVAEDecoder(snapshot: snapshot, config: configs.vae)
        let vaeWeights = try weightsMapper.loadVAE(dtype: .float32)
        let decoderWeights = vaeWeights.filter { $0.key.hasPrefix("decoder.") }
        try ZImageWeightsMapping.applyVAE(weights: decoderWeights, to: v, manifest: manifest, logger: logger)
        vae = v
      } else {
        logger.info("Reusing cached VAE")
      }

      quantManifest = manifest
    }

    modelConfigs = configs
    modelSnapshot = snapshot
    isModelLoaded = true
    loadedModelId = modelId
    loadedTextEncoderSelection = textEncoderSelection

    logger.info("Model loaded successfully and cached in memory")
  }
  public func loadLoRA(_ config: LoRAConfiguration, progressHandler: ProgressHandler? = nil) async throws {
    try await loadLoRAs([config], progressHandler: progressHandler)
  }

  public func prepare(
    modelSpec: String? = nil,
    textEncoderPath: String? = nil,
    loras: [LoRAConfiguration] = [],
    forceTransformerOverrideOnly: Bool = false,
    progressHandler: ProgressHandler? = nil
  ) async throws {
    let selection = resolveModelSelection(modelSpec, forceTransformerOverrideOnly: forceTransformerOverrideOnly)
    try await loadModel(
      modelSpec: selection.baseModelSpec,
      textEncoderPath: textEncoderPath,
      aioCheckpointURL: selection.aioCheckpointURL,
      aioTextEncoderPrefix: selection.aioTextEncoderPrefix,
      civitaiCheckpointURL: selection.civitaiCheckpointURL,
      civitaiVariant: selection.civitaiVariant,
      progressHandler: progressHandler
    )

    if selection.aioCheckpointURL == nil && selection.civitaiCheckpointURL == nil {
      try applyTransformerOverrideIfNeeded(selection.transformerOverrideURL)
    }

    try await swapLoRAs(loras, progressHandler: progressHandler)
  }

  public func swapLoRAs(_ configs: [LoRAConfiguration], progressHandler: ProgressHandler? = nil) async throws {
    if configs.isEmpty {
      unloadLoRA()
      return
    }

    guard isModelLoaded else {
      throw PipelineError.modelNotLoaded
    }

    try await loadLoRAs(configs, progressHandler: progressHandler)
  }

  public func generateFromRequest(_ request: ZImageGenerationRequest, progressHandler: ProgressHandler? = nil, latentPreviewHandler: LatentPreviewHandler? = nil) async throws -> URL {
    try await generate(request, progressHandler: progressHandler, latentPreviewHandler: latentPreviewHandler)
  }

  public func loadLoRAs(_ configs: [LoRAConfiguration], progressHandler: ProgressHandler? = nil) async throws {
    guard let trans = transformer else {
      throw PipelineError.transformerNotLoaded
    }
    if currentLoRAs.map(\.configuration) == configs {
      logger.info("LoRA stack already loaded with same configuration, skipping")
      return
    }
    if !currentLoRAs.isEmpty {
      logger.info("Unloading previous LoRA...")
      unloadLoRA()
    }
    guard !configs.isEmpty else { return }

    progressHandler?(GenerationProgress(stage: .loadingLoRA, stepIndex: 0, totalSteps: 1))
    logger.info("Loading \(configs.count) LoRA(s)...")

    do {
      currentLoRAs.removeAll(keepingCapacity: true)
      for (index, config) in configs.enumerated() {
        logger.info("Loading LoRA \(index + 1)/\(configs.count) from \(config.source.displayName)...")
        let loraWeights = try await LoRAWeightLoader.load(from: config)
        logger.info("Loaded LoRA: rank=\(loraWeights.rank), alpha=\(loraWeights.alpha), layers=\(loraWeights.layerCount)")

        useDynamicLoRA = true
        try LoRAApplicator.applyDynamically(to: trans, loraWeights: loraWeights, scale: config.scale, logger: logger)
        currentLoRAs.append(AppliedLoRA(weights: loraWeights, configuration: config))
        logger.info("LoRA applied successfully with scale=\(config.scale)")
      }
      logger.info("Applied \(currentLoRAs.count) LoRA(s) successfully")
    } catch let error as LoRAError {
      unloadLoRA()
      throw PipelineError.loraError(error)
    } catch {
      unloadLoRA()
      throw error
    }
  }
  public var hasLoRALoaded: Bool {
    return !currentLoRAs.isEmpty
  }
  public var loadedLoRAConfig: LoRAConfiguration? {
    return currentLoRAs.first?.configuration
  }
  public var loadedLoRAConfigs: [LoRAConfiguration] {
    currentLoRAs.map(\.configuration)
  }

  public func generate(_ request: ZImageGenerationRequest, progressHandler: ProgressHandler? = nil, latentPreviewHandler: LatentPreviewHandler? = nil) async throws -> URL {
    logger.info("Requested Z-Image generation")

    let decoded = try await generateCore(request, progressHandler: progressHandler, latentPreviewHandler: latentPreviewHandler)

    progressHandler?(GenerationProgress(stage: .saving, stepIndex: request.steps, totalSteps: request.steps))
    try QwenImageIO.saveImage(array: decoded, to: request.outputPath, metadata: request.embeddedMetadata(loras: loadedLoRAConfigs))
    logger.info("Wrote image to \(request.outputPath.path)")

    return request.outputPath
  }
  public func generateToMemory(_ request: ZImageGenerationRequest, progressHandler: ProgressHandler? = nil, latentPreviewHandler: LatentPreviewHandler? = nil) async throws -> Data {
    logger.info("Requested Z-Image generation (to memory)")

    let decoded = try await generateCore(request, progressHandler: progressHandler, latentPreviewHandler: latentPreviewHandler)

    progressHandler?(GenerationProgress(stage: .saving, stepIndex: request.steps, totalSteps: request.steps))
    let imageData = try QwenImageIO.imageData(from: decoded)
    logger.info("Generated image data (\(imageData.count) bytes)")

    return imageData
  }
  private func generateCore(_ request: ZImageGenerationRequest, progressHandler: ProgressHandler? = nil, latentPreviewHandler: LatentPreviewHandler? = nil) async throws -> MLXArray {

    let vaeScale = 16
    if request.width % vaeScale != 0 {
      throw PipelineError.invalidDimensions("Width must be divisible by \(vaeScale) (got \(request.width)). Please adjust to a multiple of \(vaeScale).")
    }
    if request.height % vaeScale != 0 {
      throw PipelineError.invalidDimensions("Height must be divisible by \(vaeScale) (got \(request.height)). Please adjust to a multiple of \(vaeScale).")
    }
    let selection = resolveModelSelection(request.model, forceTransformerOverrideOnly: request.forceTransformerOverrideOnly)
    try await loadModel(
      modelSpec: selection.baseModelSpec,
      textEncoderPath: request.textEncoderPath,
      aioCheckpointURL: selection.aioCheckpointURL,
      aioTextEncoderPrefix: selection.aioTextEncoderPrefix,
      civitaiCheckpointURL: selection.civitaiCheckpointURL,
      civitaiVariant: selection.civitaiVariant,
      progressHandler: progressHandler
    )

    guard let vae = vae,
          let modelConfigs = modelConfigs else {
      throw PipelineError.modelNotLoaded
    }

    if selection.aioCheckpointURL == nil {
      try applyTransformerOverrideIfNeeded(selection.transformerOverrideURL)
    }

    if !request.loras.isEmpty {
      if currentLoRAs.map(\.configuration) != request.loras {
        try await loadLoRAs(request.loras, progressHandler: progressHandler)
      }
    } else if !currentLoRAs.isEmpty {
      unloadLoRA()
    }
    progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: request.steps))
    logger.info("Encoding prompts...")

    let doCFG = request.guidanceScale > 1.0
    var promptEmbeds: MLXArray
    var negativeEmbeds: MLXArray?
    do {
      guard let tokenizer = tokenizer else {
        throw PipelineError.tokenizerNotLoaded
      }
      guard let textEncoder = textEncoder else {
        throw PipelineError.textEncoderNotLoaded
      }

      var finalPrompt = request.prompt
      if request.enhancePrompt {
        logger.info("Enhancing prompt using LLM (max tokens: \(request.enhanceMaxTokens))...")
        let enhanceConfig = PromptEnhanceConfig(
          maxNewTokens: request.enhanceMaxTokens,
          temperature: 0.7,
          topP: 0.9,
          repetitionPenalty: 1.05
        )
        let enhanced = try textEncoder.enhancePrompt(request.prompt, tokenizer: tokenizer, config: enhanceConfig)
        if enhanced.isEmpty {
          logger.warning("Prompt enhancement incomplete (need more tokens), using original prompt")
        } else {
          logger.info("Enhanced prompt: \(enhanced)")
          finalPrompt = enhanced
        }
        GPU.clearCache()
      }

      let (pe, _) = try encodePrompt(finalPrompt, tokenizer: tokenizer, textEncoder: textEncoder, maxLength: request.maxSequenceLength)
      promptEmbeds = pe

      if doCFG {
        let (ne, _) = try encodePrompt(request.negativePrompt ?? "", tokenizer: tokenizer, textEncoder: textEncoder, maxLength: request.maxSequenceLength)
        let alignedEmbeddings = PipelineUtilities.alignNegativeEmbeddingsIfNeeded(
          promptEmbeds: pe,
          negativeEmbeds: ne
        )
        promptEmbeds = alignedEmbeddings.prompt

        negativeEmbeds = alignedEmbeddings.negative
        MLX.eval(promptEmbeds, alignedEmbeddings.negative)
      } else {
        negativeEmbeds = nil
        MLX.eval(promptEmbeds)
      }
    }
    logger.info("Text encoding complete")
    if retentionPolicy == .releaseAfterRender {
      self.textEncoder = nil
    }
    GPU.clearCache()

    let vaeDivisor = modelConfigs.vae.latentDivisor
    let latentH = max(1, request.height / vaeDivisor)
    let latentW = max(1, request.width / vaeDivisor)
    let shape: [Int] = [1, ZImageModelMetadata.Transformer.inChannels, latentH, latentW]
    let randomKey: RandomStateOrKey? = request.seed.map { MLXRandom.key($0) }
    var latents = MLXRandom.normal(shape, loc: 0, scale: 1, key: randomKey)

    let imageSeqLen = PipelineUtilities.zImagePackedImageSeqLen(
      latentHeight: latentH,
      latentWidth: latentW
    )
    let mu = calculateShift(
      imageSeqLen: imageSeqLen,
      baseSeqLen: modelConfigs.scheduler.baseImageSeqLen ?? 256,
      maxSeqLen: modelConfigs.scheduler.maxImageSeqLen ?? 4096,
      baseShift: modelConfigs.scheduler.baseShift ?? 0.5,
      maxShift: modelConfigs.scheduler.maxShift ?? 1.15
    )

    var scheduler = try SchedulerFactory.create(
      kind: request.schedulerKind,
      sigmaSchedule: request.sigmaSchedule,
      numInferenceSteps: request.steps,
      config: modelConfigs.scheduler,
      mu: mu,
      seed: request.seed,
      eta: request.eta
    )

    let timestepsArray = scheduler.timesteps.asArray(Float.self)
    let sigmasArray = scheduler.sigmas.asArray(Float.self)
    let numTrainTimestepsF = Float(modelConfigs.scheduler.numTrainTimesteps)
    // The scheduler's count is authoritative: a de-duplicating schedule
    // (`beta`/`beta57`, ComfyUI-exact since WP-E12) can produce fewer steps
    // than requested; the loop runs the produced grid and says so (AC-22).
    let stepsEffective = scheduler.numInferenceSteps
    if stepsEffective != request.steps {
      logger.warning("Schedule '\(request.sigmaSchedule.rawValue)' produced \(stepsEffective) steps (\(request.steps) requested) — running \(stepsEffective) (steps_effective)")
    }

    // --- Latent-space inpainting setup ---
    var originalLatents: MLXArray?
    var latentMask: MLXArray?
    var inpaintNoise: MLXArray?
    var inpaintStartStep = 0

    if let imageData = request.inpaintImageData, let maskDataRaw = request.maskData {
      logger.info("Inpainting mode: encoding input image and mask...")
      let cgImage = try InpaintUtilities.loadCGImage(from: imageData)
      let maskCG = try InpaintUtilities.loadCGImage(from: maskDataRaw)
      logger.info("Inpainting: source=\(cgImage.width)x\(cgImage.height), mask=\(maskCG.width)x\(maskCG.height), gen=\(request.width)x\(request.height), cropXY=(\(request.maskCropX),\(request.maskCropY))")

      let pixelH = latentH * vaeDivisor
      let pixelW = latentW * vaeDivisor

      // Need full VAE (with encoder) for inpainting
      let encoderVAE = try ensureFullVAE()

      // VAE-encode the input image to latent space
      let origLatents = try InpaintUtilities.encodeImageToLatents(
        cgImage: cgImage, vae: encoderVAE, vaeConfig: modelConfigs.vae,
        pixelH: pixelH, pixelW: pixelW, logger: logger
      )
      originalLatents = origLatents

      // Convert pixel mask to latent-space mask with optional grow/feather
      latentMask = try InpaintUtilities.pixelMaskToLatent(
        maskCG, latentH: latentH, latentW: latentW,
        grow: request.maskGrow, feather: request.maskFeather,
        cropX: request.maskCropX, cropY: request.maskCropY,
        cropWidth: request.width, cropHeight: request.height,
        logger: logger
      )

      // Store the initial noise for sigma blending at each step
      inpaintNoise = latents

      // Handle denoise < 1.0: start from partially noised original
      if request.denoise < 1.0 {
        inpaintStartStep = max(0, stepsEffective - Int(ceil(Float(stepsEffective) * request.denoise)))
        let startSigma = scheduler.sigmas[inpaintStartStep].item(Float.self)
        latents = MLXArray(startSigma) * latents + MLXArray(1.0 - startSigma) * origLatents
        MLX.eval(latents)
        logger.info("Inpaint: denoise=\(request.denoise), starting at step \(inpaintStartStep)/\(stepsEffective), sigma=\(startSigma)")
      } else {
        logger.info("Inpaint: full denoise (1.0), running all \(stepsEffective) steps")
      }
    }

    logger.info("Running \(stepsEffective) denoising steps (sampler: \(request.schedulerKind.rawValue), schedule: \(request.sigmaSchedule.rawValue))...")
    do {
      guard let transformer = transformer else {
        throw PipelineError.transformerNotLoaded
      }

      // Configure DyPE for high-resolution generation
      transformer.dyPEConfig = request.dyPE
      if request.dyPE.enabled {
        logger.info("DyPE enabled: \(request.dyPE.method.rawValue) (base \(request.dyPE.baseResolution)px → \(request.width)x\(request.height))")
      }
      for stepIndex in inpaintStartStep..<stepsEffective {
        try Task.checkCancellation()
        progressHandler?(GenerationProgress(stage: .denoising, stepIndex: stepIndex, totalSteps: stepsEffective))
        let timestep = timestepsArray[stepIndex]
        let normalizedTimestep = (1000.0 - timestep) / 1000.0
        let timestepArray = MLXArray([normalizedTimestep], [1])

        var modelLatents = latents
        var embeds = promptEmbeds
        var modelTimestep = timestepArray
        if doCFG, let ne = negativeEmbeds {
          modelLatents = MLX.concatenated([latents, latents], axis: 0)
          embeds = MLX.concatenated([promptEmbeds, ne], axis: 0)
          modelTimestep = MLX.concatenated([timestepArray, timestepArray], axis: 0)
        }

        let noisePred = transformer.forward(latents: modelLatents, timestep: modelTimestep, promptEmbeds: embeds)
        let guidedNoise: MLXArray
        if doCFG, negativeEmbeds != nil {
          let batch = latents.dim(0)
          let positive = noisePred[0 ..< batch, 0..., 0..., 0...]
          let negative = noisePred[batch ..< batch * 2, 0..., 0..., 0...]
          guidedNoise = positive + request.guidanceScale * (positive - negative)
        } else {
          guidedNoise = noisePred
        }

        // The transformer emits x₀ − ε; the flow velocity dx/dσ = ε − x₀ is its
        // negation. Convert once per evaluation, after CFG, into the quantity the
        // scheduler integrates: the velocity itself (identity, byte-identical
        // default path) or x₀ = x − σ·v for the exponential-frame res_2s
        // (FDD-krea2-raw-recipe D2 / §3.2).
        let velocity = -guidedNoise
        let modelOutput = scheduler.modelInput(
          velocity: velocity, sample: latents, sigma: sigmasArray[stepIndex]
        )
        // Multi-evaluation schedulers (e.g. Heun, RES 2s) need a second model forward pass
        if scheduler.requiresIntermediateEvaluation,
           let intermediateSample = scheduler.intermediateStep(
             modelOutput: modelOutput, timestepIndex: stepIndex, sample: latents
           ) {
          // Derive timestep at the scheduler's intermediate point.
          let intermediateSigma = scheduler.intermediateSigma(timestepIndex: stepIndex)
            ?? scheduler.sigmas[stepIndex + 1].item(Float.self)
          let intermediateTimestepValue = intermediateSigma * numTrainTimestepsF
          let intermediateNormalized = (1000.0 - intermediateTimestepValue) / 1000.0
          let intermediateTimestepArray = MLXArray([intermediateNormalized], [1])

          var intermediateModelLatents = intermediateSample
          var intermediateEmbeds = promptEmbeds
          var intermediateModelTimestep = intermediateTimestepArray
          if doCFG, let ne = negativeEmbeds {
            intermediateModelLatents = MLX.concatenated([intermediateSample, intermediateSample], axis: 0)
            intermediateEmbeds = MLX.concatenated([promptEmbeds, ne], axis: 0)
            intermediateModelTimestep = MLX.concatenated([intermediateTimestepArray, intermediateTimestepArray], axis: 0)
          }

          let intermediateNoisePred = transformer.forward(
            latents: intermediateModelLatents,
            timestep: intermediateModelTimestep,
            promptEmbeds: intermediateEmbeds
          )
          let intermediateGuidedNoise: MLXArray
          if doCFG, negativeEmbeds != nil {
            let batch = latents.dim(0)
            let positive = intermediateNoisePred[0 ..< batch, 0..., 0..., 0...]
            let negative = intermediateNoisePred[batch ..< batch * 2, 0..., 0..., 0...]
            intermediateGuidedNoise = positive + request.guidanceScale * (positive - negative)
          } else {
            intermediateGuidedNoise = intermediateNoisePred
          }

          let intermediateOutput = scheduler.modelInput(
            velocity: -intermediateGuidedNoise, sample: intermediateSample, sigma: intermediateSigma
          )
          latents = scheduler.finalizeStep(
            originalOutput: modelOutput,
            intermediateOutput: intermediateOutput,
            timestepIndex: stepIndex,
            sample: latents
          )
        } else {
          latents = scheduler.step(modelOutput: modelOutput, timestepIndex: stepIndex, sample: latents)
        }
        // Inpaint: blend with original in unmasked regions at current noise level
        if let origLatents = originalLatents, let mask = latentMask, let noise = inpaintNoise {
          let nextSigma = scheduler.sigmas[stepIndex + 1].item(Float.self)
          if nextSigma > 0 {
            // Noise the original to the current noise level for seamless blending
            let noisedOriginal = MLXArray(nextSigma) * noise + MLXArray(1.0 - nextSigma) * origLatents
            latents = mask * latents + (1.0 - mask) * noisedOriginal
          } else {
            // Final step: blend clean original (no noise)
            latents = mask * latents + (1.0 - mask) * origLatents
          }
        }
        MLX.eval(latents)

        // Live denoising preview — send current latent state to preview handler.
        // The handler decides whether to emit a preview frame (based on step interval).
        // Non-blocking: if encoding is slow, frames are simply skipped.
        latentPreviewHandler?(latents, stepIndex + 1, stepsEffective, latentH, latentW)
      }
      transformer.clearCache()
    }

    progressHandler?(GenerationProgress(stage: .denoising, stepIndex: stepsEffective, totalSteps: stepsEffective))
    if retentionPolicy == .releaseAfterRender {
      unloadTransformer()
    } else {
      transformer?.clearCache()
      GPU.clearCache()
    }
    logger.info("Denoising complete, decoding with VAE...")
    progressHandler?(GenerationProgress(stage: .decoding, stepIndex: stepsEffective, totalSteps: stepsEffective))

    let decodeVAE: VAEImageDecoding
    if originalLatents == nil {
      decodeVAE = vae
    } else if let fullVAE {
      decodeVAE = fullVAE
    } else {
      decodeVAE = vae
    }
    var decoded = decodeLatents(
      latents,
      vae: decodeVAE,
      height: request.height,
      width: request.width,
      dtype: .float32
    )
    if ImageLevels.shouldApply(min: request.levelsMin, max: request.levelsMax) {
      decoded = ImageLevels.apply(image: decoded, min: request.levelsMin, max: request.levelsMax)
      MLX.eval(decoded)
    }
    MLX.eval(MLXArray([]))
    GPU.clearCache()

    return decoded
  }

  private func decodeLatents(
    _ latents: MLXArray,
    vae: VAEImageDecoding,
    height: Int,
    width: Int,
    dtype: DType = .float32
  ) -> MLXArray {
    PipelineUtilities.decodeLatents(latents, vae: vae, height: height, width: width, dtype: dtype)
  }

  private func calculateShift(
    imageSeqLen: Int,
    baseSeqLen: Int,
    maxSeqLen: Int,
    baseShift: Float,
    maxShift: Float
  ) -> Float {
    PipelineUtilities.calculateShift(
      imageSeqLen: imageSeqLen,
      baseSeqLen: baseSeqLen,
      maxSeqLen: maxSeqLen,
      baseShift: baseShift,
      maxShift: maxShift
    )
  }

  private func areZImageVariants(_ model1: String, _ model2: String) -> Bool {
    let zImageIds: Set<String> = [
      "Tongyi-MAI/Z-Image-Turbo",
      "mzbac/Z-Image-Turbo-8bit"
    ]
    return zImageIds.contains(model1) && zImageIds.contains(model2)
  }

  private func inferTransformerDim(from weights: [String: MLXArray]) -> Int? {
    // Try common norm vectors first
    if let w = weights["layers.0.attention_norm1.weight"], w.ndim == 1 { return w.dim(0) }
    if let w = weights["layers.0.ffn_norm1.weight"], w.ndim == 1 { return w.dim(0) }
    // Try attention projections
    if let w = weights["layers.0.attention.to_q.weight"], w.ndim == 2 { return w.dim(0) }
    if let w = weights["layers.0.attention.to_out.0.weight"], w.ndim == 2 { return w.dim(1) }
    // Scan for any norm weight
    if let (k, w) = weights.first(where: { $0.key.hasSuffix("attention_norm1.weight") && $0.value.ndim == 1 }) { _ = k; return w.dim(0) }
    if let (k, w) = weights.first(where: { $0.key.hasSuffix("ffn_norm1.weight") && $0.value.ndim == 1 }) { _ = k; return w.dim(0) }
    return nil
  }

  // Canonicalize override checkpoints so their tensor keys match our transformer module names.
  // Supports SD/ComfyUI-style exports that prefix keys with e.g. "model.diffusion_model.".
  func canonicalizeTransformerOverride(_ weights: [String: MLXArray], dim: Int, logger: Logger) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    for (k, v) in weights {
      // Strip common root prefixes from external checkpoints.
      var key = k
      for prefix in ["model.diffusion_model.", "diffusion_model.", "transformer.", "model."] {
        if key.hasPrefix(prefix) {
          key = String(key.dropFirst(prefix.count))
        }
      }

      // Some checkpoints use q_norm/k_norm naming; base Z-Image uses norm_q/norm_k.
      key = key.replacingOccurrences(of: ".attention.q_norm.weight", with: ".attention.norm_q.weight")
      key = key.replacingOccurrences(of: ".attention.k_norm.weight", with: ".attention.norm_k.weight")

      // Map attention.out.weight -> attention.to_out.0.weight
      if key.hasSuffix(".attention.out.weight") {
        let newKey = key.replacingOccurrences(of: ".attention.out.weight", with: ".attention.to_out.0.weight")
        out[newKey] = v
        continue
      }

      // Split attention.qkv.weight -> to_q.weight, to_k.weight, to_v.weight
      if key.hasSuffix(".attention.qkv.weight") {
        if v.ndim == 2 && v.dim(0) == dim * 3 && v.dim(1) == dim {
          let q = v[0 ..< dim, 0...]
          let kW = v[dim ..< 2*dim, 0...]
          let vW = v[2*dim ..< 3*dim, 0...]
          let base = key.replacingOccurrences(of: ".attention.qkv.weight", with: "")
          out["\(base).attention.to_q.weight"] = q
          out["\(base).attention.to_k.weight"] = kW
          out["\(base).attention.to_v.weight"] = vW
        } else {
          logger.warning("Unexpected qkv shape for \(key): \(v.shape) (expected [\(dim*3), \(dim)])")
        }
        continue
      }

      // Passthrough other keys
      var mapped = key
      // Remap final_layer.* -> all_final_layer.2-1.* so our loader can pick them up
      if mapped.hasPrefix("final_layer.") {
        mapped = mapped.replacingOccurrences(of: "final_layer.", with: "all_final_layer.2-1.")
      }
      // Remap x_embedder.* -> all_x_embedder.2-1.*
      if mapped.hasPrefix("x_embedder.") {
        mapped = mapped.replacingOccurrences(of: "x_embedder.", with: "all_x_embedder.2-1.")
      }
      out[mapped] = v
    }
    return ZImageTransformerWeightAliases.normalized(out)
  }

  func validateStrictAIOTransformerWeights(_ weights: [String: MLXArray], config: ZImageTransformerConfig) throws {
    var required: [String] = [
      "layers.0.attention.to_q.weight",
      "layers.0.attention.to_out.0.weight",
    ]

    if config.qkNorm {
      required.append(contentsOf: [
        "layers.0.attention.norm_q.weight",
        "layers.0.attention.norm_k.weight",
      ])
    }

    let missing = required.filter { weights[$0] == nil }
    if !missing.isEmpty {
      throw PipelineError.weightsMissing(
        "AIO checkpoint missing required transformer tensors after canonicalization: \(missing.joined(separator: ", ")). Use --force-transformer-override-only to treat it as transformer-only."
      )
    }
  }

  func validateAIOTransformerCoverage(
    _ weights: [String: MLXArray],
    transformer: ZImageTransformer2DModel,
    minimumCoverage: Double = 0.99
  ) throws {
    var auditWeights = weights
    if let w = weights["cap_embedder.0.weight"] { auditWeights["capEmbedNorm.weight"] = w }
    if let w = weights["cap_embedder.1.weight"] { auditWeights["capEmbedLinear.weight"] = w }
    if let w = weights["cap_embedder.1.bias"] { auditWeights["capEmbedLinear.bias"] = w }

    let audit = WeightsAudit.audit(module: transformer, weights: auditWeights, logger: logger, sample: 10)
    let total = audit.matched + audit.missing.count
    guard total > 0 else {
      throw PipelineError.weightsMissing("AIO transformer audit failed: transformer contains no parameters.")
    }

    let coverage = Double(audit.matched) / Double(total)
    guard coverage >= minimumCoverage else {
      let percent = Int((coverage * 100.0).rounded())
      let missingSample = audit.missing.prefix(10).joined(separator: ", ")
      let suffix = audit.missing.count > 10 ? ", ..." : ""
      throw PipelineError.weightsMissing(
        "AIO transformer weights coverage too low: matched \(audit.matched)/\(total) (\(percent)%). Missing (sample): \(missingSample)\(suffix). Use --force-transformer-override-only to treat it as transformer-only."
      )
    }
  }

}
