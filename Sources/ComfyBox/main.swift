import Foundation
import Dispatch
import Logging
import Metal
import MLX
import MLXRandom
import MLXNN
import ZImage
import Darwin

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

LoggingSystem.bootstrap { label in
  var handler = StreamLogHandler.standardError(label: label)
  handler.logLevel = .info
  return handler
}
private final class Box<T>: @unchecked Sendable {
  var value: T
  init(_ value: T) { self.value = value }
}

struct ZImageCLI {
  static let logger: Logger = {
    var logger = Logger(label: "z-image.cli")
    logger.logLevel = .info
    return logger
  }()

  static func run() throws {
    let metalDevice = MTLCreateSystemDefaultDevice()
    if let dev = metalDevice {
      logger.info("Metal device: \(dev.name)")
    } else {
      logger.warning("No Metal device detected; MLX will fall back to CPU.")
    }

    let executableURL = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
    if let stagedMetalLibrary = try MLXMetalLibraryLocator.prepareColocatedMetalLibrary(executableURL: executableURL) {
      logger.info("Using MLX metallib at \(stagedMetalLibrary.path)")
    } else {
      logger.warning("No MLX metallib was found next to the executable or in known build outputs. If MLX fails later, build via xcodebuild or place mlx.metallib beside the CLI.")
    }

    var prompt: String?
    var negativePrompt: String?
    var width = ZImageModelMetadata.recommendedWidth
    var height = ZImageModelMetadata.recommendedHeight
    var steps = ZImageModelMetadata.recommendedInferenceSteps
    var guidance = ZImageModelMetadata.recommendedGuidanceScale
    var seeds: [UInt64] = []
    var outputPath = "z-image.png"
    var levelsMin: Float = 0.0
    var levelsMax: Float = 1.0
    var model: String?
    var textEncoderPath: String?
    var cacheLimit: Int?
    var maxSequenceLength = 512
    var loraEntries: [String] = []
    var loraScaleOverrides: [Float] = []
    var enhancePrompt = false
    var enhanceMaxTokens = 512
    var noProgress = false
    var auditWeights = false
    var forceTransformerOverrideOnly = false
    var generateSVG = false
    var svgPreset = "default"
    var schedulerKind: SchedulerKind = .euler
    var sigmaSchedule: SigmaScheduleKind = .flow
    var eta: Float?
    var dyPEMethod: DyPEMethod = .none
    var dyPEDisabled = false
    var writeMetadata = false
    var configFromMetadata: String?
    var autoSeeds: Int?
    var resumeBatchPath: String?
    var promptFilePath: String?

    // --- Img2img ---
    var imagePath: String?
    var imageStrength: Float?
    var imageCreativity: Float?
    var modelFamily: String?

    // --- Flux 2 img2img ---
    var initImagePath: String?
    var flux2Denoise: Float?

    let args = Array(CommandLine.arguments.dropFirst())
    var iterator = args.makeIterator()

    while let arg = iterator.next() {
      switch arg {
      case "--prompt", "-p":
        prompt = nextValue(for: arg, iterator: &iterator)
      case "--negative-prompt", "--np":
        negativePrompt = nextValue(for: arg, iterator: &iterator)
      case "--width", "-W":
        width = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: width)
      case "--height", "-H":
        height = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: height)
      case "--steps", "-s":
        steps = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: steps)
      case "--guidance", "-g":
        guidance = floatValue(for: arg, iterator: &iterator, fallback: guidance)
      case "--seed":
        if let s = uint64Value(for: arg, iterator: &iterator) {
          seeds.append(s)
        }
      case "--output", "-o":
        outputPath = nextValue(for: arg, iterator: &iterator)
      case "--levels-min":
        levelsMin = floatValue(for: arg, iterator: &iterator, fallback: 0.0)
      case "--levels-max":
        levelsMax = floatValue(for: arg, iterator: &iterator, fallback: 1.0)
      case "--model", "-m":
        model = nextValue(for: arg, iterator: &iterator)
      case "--text-encoder-path":
        textEncoderPath = nextValue(for: arg, iterator: &iterator)
      case "--force-transformer-override-only":
        forceTransformerOverrideOnly = true
      case "--cache-limit":
        cacheLimit = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: 2048)
      case "--max-sequence-length":
        maxSequenceLength = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: 512)
      case "--lora", "-l":
        loraEntries.append(nextValue(for: arg, iterator: &iterator))
      case "--lora-scale":
        loraScaleOverrides.append(floatValue(for: arg, iterator: &iterator, fallback: 1.0))
      case "--lora-paths":
        loraEntries.append(contentsOf: splitCommaSeparated(nextValue(for: arg, iterator: &iterator)))
      case "--lora-scales":
        loraScaleOverrides.append(contentsOf: floatListValue(for: arg, iterator: &iterator, fallback: 1.0))
      case "--enhance", "-e":
        enhancePrompt = true
      case "--enhance-max-tokens":
        enhanceMaxTokens = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: 512)
      case "--no-progress":
        noProgress = true
      case "--audit-weights":
        auditWeights = true
      case "--scheduler", "--sampler":
        let raw = nextValue(for: arg, iterator: &iterator)
        guard let kind = SchedulerKind(rawValue: raw) else {
          let valid = SchedulerKind.allCases.map(\.rawValue).joined(separator: ", ")
          failArgumentParsing("Unknown scheduler '\(raw)'. Valid: \(valid)")
        }
        // WP-E13: the N-row tableau samplers are dispatched only by the Krea 2
        // denoise loop. This path drives ZImagePipeline / ZImageControlPipeline,
        // which take one model evaluation per step — accepting the name here
        // would render first-order Euler under it. Refuse now, before weights.
        if kind.isNRowTableau {
          let usable = SchedulerKind.allCases.filter { !$0.isNRowTableau }
            .map(\.rawValue).joined(separator: ", ")
          failArgumentParsing(
            "Sampler '\(raw)' is an N-row tableau sampler supported only by the krea2 model "
              + "family (`ComfyBox krea2`, or a krea2 model on the server); the Z-Image path "
              + "takes one model evaluation per step and cannot run it. Valid here: \(usable)")
        }
        schedulerKind = kind
      case "--sigma-schedule":
        let raw = nextValue(for: arg, iterator: &iterator)
        guard let kind = SigmaScheduleKind(rawValue: raw) else {
          let valid = SigmaScheduleKind.allCases.map(\.rawValue).joined(separator: ", ")
          failArgumentParsing("Unknown sigma schedule '\(raw)'. Valid: \(valid)")
        }
        sigmaSchedule = kind
      case "--eta":
        eta = floatValue(for: arg, iterator: &iterator, fallback: 0.0)
      case "--dype":
        let raw = nextValue(for: arg, iterator: &iterator).lowercased()
        switch raw {
        case "ntk": dyPEMethod = .ntk
        case "yarn": dyPEMethod = .yarn
        case "none", "off": dyPEMethod = .none
        default:
          failArgumentParsing("Unknown DyPE method '\(raw)'. Valid: ntk, yarn, none")
        }
      case "--no-dype":
        dyPEDisabled = true
      case "--svg":
        generateSVG = true
      case "--svg-preset":
        svgPreset = nextValue(for: arg, iterator: &iterator)
      case "--metadata":
        writeMetadata = true
      case "--config-from-metadata":
        configFromMetadata = nextValue(for: arg, iterator: &iterator)
      case "--auto-seeds":
        autoSeeds = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: 1)
      case "--resume-batch":
        resumeBatchPath = nextValue(for: arg, iterator: &iterator)
      case "--prompt-file":
        promptFilePath = nextValue(for: arg, iterator: &iterator)
      case "--image-path", "--image":
        imagePath = nextValue(for: arg, iterator: &iterator)
      case "--image-strength":
        imageStrength = floatValue(for: arg, iterator: &iterator, fallback: 0.3)
      case "--creativity":
        imageCreativity = floatValue(for: arg, iterator: &iterator, fallback: 0.7)
      case "--model-family":
        modelFamily = nextValue(for: arg, iterator: &iterator)
      case "--init-image":
        initImagePath = nextValue(for: arg, iterator: &iterator)
      case "--denoise":
        flux2Denoise = floatValue(for: arg, iterator: &iterator, fallback: 0.7)
      case "--help", "-h":
        printUsage()
        return
      case "quantize":
        try runQuantize(args: Array(args.dropFirst()))
        return
      case "quantize-controlnet":
        try runQuantizeControlnet(args: Array(args.dropFirst()))
        return
      case "quantize-ltx2":
        try runQuantizeLTX2(args: Array(args.dropFirst()))
        return
      case "control":
        try runControl(args: Array(args.dropFirst()))
        return
      case "docs":
        try runDocs(args: Array(args.dropFirst()))
        return
      case "serve":
        try runServe(args: Array(args.dropFirst()))
        return
      case "upscale":
        try runUpscale(args: Array(args.dropFirst()))
        return
      case "mcp":
        try runMCP(args: Array(args.dropFirst()))
        return
      case "lora":
        try runLoRA(args: Array(args.dropFirst()))
        return
      case "models":
        runModels(args: Array(args.dropFirst()))
        return
      case "mcp":
        try runMCP(args: Array(args.dropFirst()))
        return
      case "video":
        try runVideo(args: Array(args.dropFirst()))
        return
      case "ltx2-demo":
        try runLTX2Demo(args: Array(args.dropFirst()))
        return
      case "ltx2-i2v":
        try runLTX2I2V(args: Array(args.dropFirst()))
        return
      case "ltx2-vae-test":
        try runLTX2VAETest(args: Array(args.dropFirst()))
        return
      case "ltx2-text-encoder-test":
        try runLTX2TextEncoderTest(args: Array(args.dropFirst()))
        return
      case "telegram":
        try runTelegram(args: Array(args.dropFirst()))
        return
      case "krea2":
        try runKrea2(args: Array(args.dropFirst()))
        return
      default:
        logger.warning("Unknown argument: \(arg)")
      }
    }

    if auditWeights {
      guard metalDevice != nil else {
        throw NSError(
          domain: "ZImageCLI",
          code: 2,
          userInfo: [NSLocalizedDescriptionKey: "--audit-weights currently requires a Metal-capable runtime in this build."]
        )
      }
      let capturedModel = model
      let capturedTextEncoderPath = textEncoderPath
      let taskError = Box<Error?>(nil)
      nonisolated(unsafe) let semaphore = DispatchSemaphore(value: 0)
      Task {
        do {
          try await runAudit(modelSpec: capturedModel, textEncoderPath: capturedTextEncoderPath)
        } catch {
          logger.error("Weight audit failed: \(error)")
          taskError.value = error
        }
        semaphore.signal()
      }
      semaphore.wait()
      if let error = taskError.value {
        throw error
      }
      return
    }

    // --config-from-metadata: load saved generation params (CLI flags take precedence)
    if let metadataPath = configFromMetadata {
      let loaded: GenerationMetadata
      do {
        loaded = try MetadataReader.read(from: metadataPath)
      } catch {
        fputs("Error loading metadata: \(error)\n", stderr)
        exit(1)
      }
      if prompt == nil { prompt = loaded.parameters.prompt }
      if negativePrompt == nil { negativePrompt = loaded.parameters.negativePrompt }
      if seeds.isEmpty { seeds.append(loaded.parameters.seed) }
      if width == ZImageModelMetadata.recommendedWidth { width = loaded.parameters.width }
      if height == ZImageModelMetadata.recommendedHeight { height = loaded.parameters.height }
      if steps == ZImageModelMetadata.recommendedInferenceSteps { steps = loaded.parameters.steps }
      if guidance == ZImageModelMetadata.recommendedGuidanceScale { guidance = loaded.parameters.guidance }
      if model == nil { model = loaded.model.path }
      if let loadedSched = SchedulerKind(rawValue: loaded.parameters.scheduler), schedulerKind == .euler {
        schedulerKind = loadedSched
      }
      if let sig = loaded.parameters.sigmaSchedule,
         let sigKind = SigmaScheduleKind(rawValue: sig), sigmaSchedule == .flow {
        sigmaSchedule = sigKind
      }
      if loraEntries.isEmpty {
        for lora in loaded.loras {
          loraEntries.append(lora.path)
          loraScaleOverrides.append(lora.scale)
        }
      }
      logger.info("Loaded generation config from: \(metadataPath)")
    }

    // Resolve model aliases to HuggingFace IDs
    if let m = model {
      switch m.lowercased() {
      case "fibo", "briaai/fibo":
        model = "briaai/FIBO"
      case "chroma", "lodestones/chroma", "jack813liu/mlx-chroma":
        model = "jack813liu/mlx-chroma"
      case "z-image-base", "zimage-base":
        model = ZImageRepository.baseId
      case "z-image-turbo", "zimage-turbo", "z-image":
        // "z-image" without qualifier defaults to Turbo for backwards compat
        if m.lowercased() == "z-image" || m.lowercased() == "zimage-turbo" || m.lowercased() == "z-image-turbo" {
          model = ZImageRepository.id
        }
      default:
        break
      }
    }

    // When Z-Image Base is selected, override defaults if user didn't specify them
    if let m = model, ZImageRepository.isBaseModel(m) {
      if steps == ZImageModelMetadata.recommendedInferenceSteps {
        steps = ZImageModelMetadata.Base.recommendedInferenceSteps
      }
      if guidance == ZImageModelMetadata.recommendedGuidanceScale {
        guidance = ZImageModelMetadata.Base.recommendedGuidanceScale
      }
    }

    guard let prompt else {
      printUsage()
      exit(1)
    }

    // Validate img2img flags
    if imageStrength != nil && imageCreativity != nil {
      fputs("Error: --image-strength and --creativity are mutually exclusive.\n", stderr)
      exit(1)
    }
    if imagePath != nil && (imageStrength == nil && imageCreativity == nil) {
      imageStrength = 0.3
    }
    if (imageStrength != nil || imageCreativity != nil) && imagePath == nil {
      fputs("Error: --image-strength/--creativity requires --image-path.\n", stderr)
      exit(1)
    }

    if let limit = cacheLimit {
      GPU.set(cacheLimit: limit * 1024 * 1024)
      logger.info("GPU cache limit set to \(limit)MB")
    }
    let loraConfigs = buildLoRAConfigurations(entries: loraEntries, scaleOverrides: loraScaleOverrides)
    if !loraConfigs.isEmpty {
      logger.info("Using \(loraConfigs.count) LoRA(s)")
    }

    // Determine if this is a batch run
    let isBatchRun = (autoSeeds != nil && autoSeeds! > 0) || seeds.count > 1

    // Pin seed for single-image flow (batch runner handles its own seeds)
    if !isBatchRun && seeds.isEmpty {
      seeds.append(UInt64.random(in: 0...UInt64.max))
    }
    let seed: UInt64? = isBatchRun ? nil : seeds.first

    // Build DyPE config — auto-enable when resolution exceeds base training size
    let dyPEConfig: DyPEConfig
    if dyPEDisabled {
      dyPEConfig = .disabled
    } else if dyPEMethod != .none {
      dyPEConfig = DyPEConfig(enabled: true, method: dyPEMethod)
      logger.info("DyPE enabled: \(dyPEMethod.rawValue) (target \(width)x\(height))")
    } else if max(width, height) > 1024 {
      // Auto-enable NTK when generating above training resolution
      dyPEConfig = .ntk
      logger.info("DyPE auto-enabled: ntk (target \(width)x\(height) exceeds 1024 training resolution)")
    } else {
      dyPEConfig = .disabled
    }

    // === BATCH EXECUTION PATH ===
    if isBatchRun {
      let batchOutputPattern = BatchRunner.outputPattern(from: outputPath)
      let checkpointPath = resumeBatchPath ?? {
        let dir = URL(fileURLWithPath: outputPath).deletingLastPathComponent().path
        return (dir as NSString).appendingPathComponent(".batch-progress.jsonl")
      }()
      let batchConfig = BatchConfig(
        seeds: seeds,
        autoSeedCount: autoSeeds,
        outputPattern: batchOutputPattern,
        continueOnError: true,
        metadataEnabled: writeMetadata,
        promptFilePath: promptFilePath,
        checkpointPath: checkpointPath
      )

      let capturedNegativePrompt = negativePrompt
      let capturedModel = model
      let capturedTextEncoderPath = textEncoderPath
      let capturedLoraConfigs = loraConfigs
      let capturedWriteMetadata = writeMetadata

      let pipeline = ZImagePipeline(logger: logger)
      nonisolated(unsafe) let batchSemaphore = DispatchSemaphore(value: 0)
      let resultBox = Box<BatchResult?>(nil)
      var capturedError: (any Error)? = nil

      Task {
        do {
          let result = try await BatchRunner.run(config: batchConfig) { batchSeed, batchPrompt in
            let effectivePrompt = batchPrompt ?? prompt
            let batchOutputPath = BatchRunner.resolveOutputPath(
              pattern: batchOutputPattern, seed: batchSeed, index: 0
            )
            let batchRequest = ZImageGenerationRequest(
              prompt: effectivePrompt,
              negativePrompt: capturedNegativePrompt,
              width: width,
              height: height,
              steps: steps,
              guidanceScale: guidance,
              seed: batchSeed,
              outputPath: URL(fileURLWithPath: batchOutputPath),
              levelsMin: levelsMin,
              levelsMax: levelsMax,
              model: capturedModel,
              textEncoderPath: capturedTextEncoderPath,
              maxSequenceLength: maxSequenceLength,
              loras: capturedLoraConfigs,
              enhancePrompt: enhancePrompt,
              enhanceMaxTokens: enhanceMaxTokens,
              forceTransformerOverrideOnly: forceTransformerOverrideOnly,
              schedulerKind: schedulerKind,
              sigmaSchedule: sigmaSchedule,
              eta: eta,
              dyPE: dyPEConfig
            )
            let startTime = Date()
            _ = try await pipeline.generate(batchRequest, progressHandler: nil)

            // Write metadata sidecar if enabled
            if capturedWriteMetadata {
              let loraInfos = capturedLoraConfigs.map { lora -> LoRAInfo in
                let path: String
                switch lora.source {
                case .local(let url): path = url.path
                case .huggingFace(let id, _): path = id
                }
                return LoRAInfo(path: path, scale: lora.scale)
              }
              let metadata = GenerationMetadata(
                pipeline: .txt2img,
                model: ModelInfo(
                  family: "zimage",
                  variant: (capturedModel.map { ZImageRepository.isBaseModel($0) ? "base" : "turbo" }) ?? "turbo",
                  path: capturedModel ?? ZImageRepository.id
                ),
                parameters: GenerationParameters(
                  prompt: effectivePrompt,
                  negativePrompt: capturedNegativePrompt,
                  seed: batchSeed,
                  steps: steps,
                  guidance: guidance,
                  width: width,
                  height: height,
                  scheduler: schedulerKind.rawValue,
                  sigmaSchedule: sigmaSchedule == .flow ? nil : sigmaSchedule.rawValue
                ),
                loras: loraInfos,
                output: OutputInfo(
                  path: batchOutputPath,
                  width: width,
                  height: height,
                  renderTimeSeconds: Date().timeIntervalSince(startTime)
                )
              )
              MetadataWriter.write(metadata, for: batchOutputPath)
            }

            return batchOutputPath
          }
          resultBox.value = result
        } catch {
          logger.error("Batch failed: \(error)")
          capturedError = error
        }
        batchSemaphore.signal()
      }
      batchSemaphore.wait()
      if let err = capturedError {
        fputs("Error: \(err)\n", stderr)
        exit(1)
      }

      // Print batch summary
      if let result = resultBox.value {
        let duration = String(format: "%.1f", result.totalDuration)
        print("\nBatch complete: \(result.completed)/\(result.totalSeeds) succeeded, \(result.failed) failed, \(result.skipped) skipped (resumed)")
        print("Total time: \(duration)s")
        if let checkpoint = batchConfig.checkpointPath {
          print("Checkpoint: \(checkpoint)")
        }
      }
      return
    }

    // === IMG2IMG PATH ===
    if let imagePath {
      let resolvedStrength: Float
      let specifiedAs: Img2ImgRequest.Img2ImgSpecifier
      if let creativity = imageCreativity {
        resolvedStrength = 1.0 - max(0.01, min(0.99, creativity))
        specifiedAs = .creativity
      } else {
        resolvedStrength = imageStrength ?? 0.3
        specifiedAs = .strength
      }
      let specifiedValue = specifiedAs == .creativity ? (imageCreativity ?? 0.7) : resolvedStrength

      let img2imgRequest = Img2ImgRequest(
        prompt: prompt,
        negativePrompt: negativePrompt,
        width: width == ZImageModelMetadata.recommendedWidth ? nil : width,
        height: height == ZImageModelMetadata.recommendedHeight ? nil : height,
        steps: steps,
        guidanceScale: guidance,
        seed: seed,
        outputPath: URL(fileURLWithPath: outputPath),
        levelsMin: levelsMin,
        levelsMax: levelsMax,
        model: model,
        textEncoderPath: textEncoderPath,
        maxSequenceLength: maxSequenceLength,
        loras: loraConfigs,
        enhancePrompt: enhancePrompt,
        enhanceMaxTokens: enhanceMaxTokens,
        forceTransformerOverrideOnly: forceTransformerOverrideOnly,
        schedulerKind: schedulerKind,
        sigmaSchedule: sigmaSchedule,
        eta: eta,
        dyPE: dyPEConfig,
        sourceImagePath: imagePath,
        strength: resolvedStrength,
        specifiedAs: specifiedAs
      )

      let pipeline = ZImagePipeline(logger: logger)
      nonisolated(unsafe) let semaphore = DispatchSemaphore(value: 0)
      let shouldWriteMetadata = writeMetadata
      let capturedStartTime = Date()
      let useBar = !noProgress && (isatty(STDERR_FILENO) != 0)
      let bar = useBar ? ProgressBar(total: steps) : nil
      let finalOutputPath = URL(fileURLWithPath: outputPath)
      let shouldGenerateSVG = generateSVG
      let svgPresetCopy = svgPreset
      let capturedSeed = seed
      let capturedModel = model
      let capturedLoraConfigs = loraConfigs
      let capturedImagePath = imagePath
      let capturedStrength = resolvedStrength
      let capturedSpecifiedAs = specifiedAs.rawValue
      let capturedSpecifiedValue = specifiedValue
      var capturedError: (any Error)? = nil
      Task {
        do {
          _ = try await pipeline.generateImg2Img(img2imgRequest, progressHandler: { progress in
            guard !noProgress else { return }
            guard progress.stage == .denoising else { return }
            let completed = min(progress.totalSteps, max(0, progress.stepIndex))
            if let bar {
              bar.update(completed: completed)
              if completed == progress.totalSteps {
                bar.finish(forceNewline: true)
              }
            } else {
              PlainProgress.shared.report(completed: completed, total: progress.totalSteps)
            }
          })
          if let bar { bar.finish(forceNewline: true) }
          if shouldWriteMetadata {
            let loraInfos = capturedLoraConfigs.map { lora -> LoRAInfo in
              let path: String
              switch lora.source {
              case .local(let url): path = url.path
              case .huggingFace(let id, _): path = id
              }
              return LoRAInfo(path: path, scale: lora.scale)
            }
            let metadata = GenerationMetadata(
              pipeline: .img2img,
              model: ModelInfo(
                family: "zimage",
                variant: (capturedModel.map { ZImageRepository.isBaseModel($0) ? "base" : "turbo" }) ?? "turbo",
                path: capturedModel ?? ZImageRepository.id
              ),
              parameters: GenerationParameters(
                prompt: prompt,
                negativePrompt: negativePrompt,
                seed: capturedSeed ?? 0,
                steps: steps,
                guidance: guidance,
                width: width,
                height: height,
                scheduler: schedulerKind.rawValue,
                sigmaSchedule: sigmaSchedule == .flow ? nil : sigmaSchedule.rawValue
              ),
              img2img: Img2ImgInfo(
                sourceImage: capturedImagePath,
                strength: capturedStrength,
                specifiedAs: capturedSpecifiedAs,
                specifiedValue: capturedSpecifiedValue
              ),
              loras: loraInfos,
              output: OutputInfo(
                path: finalOutputPath.path,
                width: width,
                height: height,
                renderTimeSeconds: Date().timeIntervalSince(capturedStartTime)
              )
            )
            if let written = MetadataWriter.write(metadata, for: finalOutputPath.path) {
              logger.info("Metadata written: \(written)")
            }
          }
          if shouldGenerateSVG {
            let svgOutputPath = finalOutputPath.deletingPathExtension().appendingPathExtension("svg")
            logger.info("Converting to SVG: \(svgOutputPath.path)")
            try convertToSVG(input: finalOutputPath, output: svgOutputPath, preset: svgPresetCopy)
            logger.info("SVG generated: \(svgOutputPath.path)")
          }
        } catch {
          logger.error("Img2img generation failed: \(error)")
          capturedError = error
          if let bar { bar.finish(forceNewline: true) }
        }
        semaphore.signal()
      }
      semaphore.wait()
      if let err = capturedError {
        fputs("Error: \(err)\n", stderr)
        exit(1)
      }
      return
    }

    // === CHROMA PATH ===
    // Auto-detect or explicitly route to Chroma pipeline
    let isChromaModel: Bool
    if let family = modelFamily {
      isChromaModel = (family.lowercased() == "chroma")
    } else if let modelSpec = model {
      if ChromaModelDetection.isKnownChromaModel(modelSpec) {
        isChromaModel = true
      } else {
        let localURL = URL(fileURLWithPath: modelSpec)
        if FileManager.default.fileExists(atPath: localURL.path) {
          isChromaModel = ChromaModelDetection.detect(at: localURL) != nil
        } else {
          isChromaModel = false
        }
      }
    } else {
      isChromaModel = false
    }

    if isChromaModel {
      // Chroma packs 2x2 latent patches (VAE /8, then patchify /2), so
      // pixel dimensions must be multiples of 16 or the reshape crashes.
      guard width % 16 == 0 && height % 16 == 0 else {
        fputs("Error: Chroma requires width and height to be multiples of 16 (got \(width)x\(height)).\n", stderr)
        exit(1)
      }
      nonisolated(unsafe) let chromaSemaphore = DispatchSemaphore(value: 0)
      let capturedModel = model
      let capturedNegativePrompt = negativePrompt
      let capturedLoraEntries = loraEntries
      let capturedLoraScales = loraScaleOverrides
      let useBar = !noProgress && (isatty(STDERR_FILENO) != 0)
      // Chroma defaults: 28 steps (8 for flash-heun)
      let chromaSteps = steps == ZImageModelMetadata.recommendedInferenceSteps ? 28 : steps
      let bar = useBar ? ProgressBar(total: chromaSteps) : nil

      // Map scheduler: heun/beta → Chroma scheduler types
      let chromaScheduler: ChromaSchedulerType
      switch schedulerKind {
      case .heun:
        chromaScheduler = .heun
      default:
        // Check sigma schedule for beta
        if sigmaSchedule == .beta {
          chromaScheduler = .beta
        } else {
          chromaScheduler = .euler
        }
      }

      var capturedError: (any Error)? = nil
      Task {
        do {
          // Resolve model snapshot
          let modelSpec = capturedModel ?? "jack813liu/mlx-chroma"
          let snapshot = try await ModelResolution.resolve(modelSpec: modelSpec)

          // Detect model config from snapshot
          guard let detected = ChromaModelDetection.detect(at: snapshot) else {
            throw NSError(domain: "ZImageCLI", code: 1,
              userInfo: [NSLocalizedDescriptionKey: "Model at \(snapshot.path) is not a Chroma model"])
          }
          logger.info("Detected Chroma model")

          // Load all components (transformer, T5, VAE)
          let components = try ChromaInitializer.load(
            from: snapshot,
            paths: detected.componentPaths,
            config: detected.config,
            logger: logger
          )

          // Apply LoRAs to transformer
          if !capturedLoraEntries.isEmpty {
            logger.info("Loading \(capturedLoraEntries.count) LoRA(s) for Chroma")
            for (idx, loraEntry) in capturedLoraEntries.enumerated() {
              let scale = idx < capturedLoraScales.count ? capturedLoraScales[idx] : 1.0

              // Resolve LoRA path (local file or HuggingFace)
              let loraURL: URL
              if FileManager.default.fileExists(atPath: loraEntry) {
                loraURL = URL(fileURLWithPath: loraEntry)
              } else {
                // Try as HuggingFace model ID
                let resolved = try await ModelResolution.resolve(
                  modelSpec: loraEntry,
                  filePatterns: ["*.safetensors"]
                )
                let contents = try FileManager.default.contentsOfDirectory(
                  at: resolved, includingPropertiesForKeys: nil
                )
                guard let safetensors = contents.first(where: { $0.pathExtension == "safetensors" }) else {
                  logger.error("No safetensors found in LoRA: \(loraEntry)")
                  continue
                }
                loraURL = safetensors
              }

              try ChromaLoRALoader.loadAndApply(
                path: loraURL.path,
                to: components.transformer,
                scale: scale,
                logger: logger
              )
            }
          }

          // Load T5 tokenizer
          let tokenizer = try ChromaTokenizer.load(from: detected.componentPaths.tokenizerPath)

          // Build pipeline
          let pipeline = ChromaPipeline(
            transformer: components.transformer,
            t5: components.t5,
            vae: components.vae,
            config: detected.config
          )

          // Tokenize prompt (unpadded — matches Python behavior)
          let tokenIds = tokenizer.encodeUnpadded(prompt: prompt)

          // Tokenize negative prompt for CFG (empty string = unconditional)
          let negTokenIds = tokenizer.encodeUnpadded(prompt: capturedNegativePrompt ?? "")

          // Chroma defaults: 28 steps, guidance 0.0 (distilled), cfg 4.0
          // Flash-heun: 8 steps, heun/beta scheduler, CFG 1.0
          let chromaGuidance = guidance == ZImageModelMetadata.recommendedGuidanceScale ? Float(0.0) : guidance
          let chromaCFG: Float = chromaScheduler == .euler ? 4.0 : 1.0

          logger.info("Chroma: \(chromaSteps) steps, scheduler=\(chromaScheduler.rawValue), guidance=\(chromaGuidance), cfg=\(chromaCFG)")

          // Generate image
          let pixels = try pipeline.generate(
            tokenIds: tokenIds,
            negativeTokenIds: negTokenIds,
            width: width,
            height: height,
            numSteps: chromaSteps,
            guidance: chromaGuidance,
            cfg: chromaCFG,
            firstNStepsWithoutCFG: chromaScheduler == .euler ? 0 : -1,
            schedulerType: chromaScheduler,
            seed: seed,
            progressCallback: noProgress ? nil : { step, total in
              let completed = min(total, max(0, step))
              if let bar {
                bar.update(completed: completed)
                if completed == total {
                  bar.finish(forceNewline: true)
                }
              } else {
                PlainProgress.shared.report(completed: completed, total: total)
              }
            }
          )
          if let bar { bar.finish(forceNewline: true) }

          // Save image — pipeline returns NHWC [1, H, W, 3], saveImage expects CHW [3, H, W]
          let chromaOutputURL = URL(fileURLWithPath: outputPath)
          let chromaCHW = pixels.squeezed(axis: 0).transposed(2, 0, 1)
          try QwenImageIO.saveImage(array: chromaCHW, to: chromaOutputURL)
          logger.info("Chroma image saved to \(chromaOutputURL.path)")
        } catch {
          logger.error("Chroma generation failed: \(error)")
          capturedError = error
          if let bar { bar.finish(forceNewline: true) }
        }
        chromaSemaphore.signal()
      }
      chromaSemaphore.wait()
      if let err = capturedError {
        fputs("Error: \(err)\n", stderr)
        exit(1)
      }
      return
    }

    // === FIBO PATH ===
    // Auto-detect or explicitly route to FIBO pipeline
    let isFiboModel: Bool
    if let family = modelFamily {
      isFiboModel = (family.lowercased() == "fibo")
    } else if let modelSpec = model {
      if FiboModelDetection.isKnownFiboModel(modelSpec) {
        isFiboModel = true
      } else {
        let localURL = URL(fileURLWithPath: modelSpec)
        if FileManager.default.fileExists(atPath: localURL.path) {
          isFiboModel = FiboModelDetection.detect(at: localURL) != nil
        } else {
          isFiboModel = false
        }
      }
    } else {
      isFiboModel = false
    }

    if isFiboModel {
      nonisolated(unsafe) let fiboSemaphore = DispatchSemaphore(value: 0)
      let capturedModel = model
      let useBar = !noProgress && (isatty(STDERR_FILENO) != 0)
      let bar = useBar ? ProgressBar(total: steps) : nil
      var capturedError: (any Error)? = nil
      Task {
        do {
          // Resolve model snapshot
          let modelSpec = capturedModel ?? "briaai/FIBO"
          let snapshot = try await ModelResolution.resolve(modelSpec: modelSpec)

          // Detect model config from snapshot
          guard let detected = FiboModelDetection.detect(at: snapshot) else {
            throw NSError(domain: "ZImageCLI", code: 1,
              userInfo: [NSLocalizedDescriptionKey: "Model at \(snapshot.path) is not a FIBO model"])
          }
          logger.info("Detected FIBO model")

          let fiboPipeline = FiboPipeline(logger: logger)
          try fiboPipeline.loadModel(
            from: snapshot,
            transformerConfig: detected.transformerConfig,
            vaeConfig: detected.vaeConfig,
            textEncoderConfig: detected.textEncoderConfig
          )

          // FIBO defaults: 30 steps, guidance 4.0
          let fiboSteps = steps == ZImageModelMetadata.recommendedInferenceSteps ? 30 : steps
          let fiboGuidance = guidance == ZImageModelMetadata.recommendedGuidanceScale ? Float(4.0) : guidance

          let fiboRequest = FiboGenerationRequest(
            prompt: prompt,
            negativePrompt: negativePrompt,
            width: width,
            height: height,
            steps: fiboSteps,
            guidanceScale: fiboGuidance,
            seed: seed,
            outputPath: URL(fileURLWithPath: outputPath),
            levelsMin: levelsMin,
            levelsMax: levelsMax
          )

          _ = try await fiboPipeline.generate(fiboRequest, progressHandler: { progress in
            guard !noProgress else { return }
            guard progress.stage == .denoising else { return }
            let completed = min(progress.totalSteps, max(0, progress.stepIndex))
            if let bar {
              bar.update(completed: completed)
              if completed == progress.totalSteps {
                bar.finish(forceNewline: true)
              }
            } else {
              PlainProgress.shared.report(completed: completed, total: progress.totalSteps)
            }
          })
          if let bar { bar.finish(forceNewline: true) }
        } catch {
          logger.error("FIBO generation failed: \(error)")
          capturedError = error
          if let bar { bar.finish(forceNewline: true) }
        }
        fiboSemaphore.signal()
      }
      fiboSemaphore.wait()
      if let err = capturedError {
        fputs("Error: \(err)\n", stderr)
        exit(1)
      }
      return
    }

    // === FLUX 2 KLEIN PATH ===
    // Auto-detect or explicitly route to Flux 2 pipeline
    let isFlux2: Bool
    if let family = modelFamily {
      isFlux2 = (family.lowercased() == "flux2")
    } else if let modelSpec = model {
      // Check HF model ID first, then check local directory
      if Flux2ModelDetection.isKnownFlux2Model(modelSpec) {
        isFlux2 = true
      } else {
        let localURL = URL(fileURLWithPath: modelSpec)
        if FileManager.default.fileExists(atPath: localURL.path) {
          isFlux2 = Flux2ModelDetection.detect(at: localURL) != nil
        } else {
          isFlux2 = false
        }
      }
    } else {
      isFlux2 = false
    }

    if isFlux2 {
      nonisolated(unsafe) let flux2Semaphore = DispatchSemaphore(value: 0)
      let capturedModel = model
      let useBar = !noProgress && (isatty(STDERR_FILENO) != 0)
      let bar = useBar ? ProgressBar(total: steps) : nil
      var capturedError: (any Error)? = nil
      Task {
        do {
          // Resolve model snapshot
          let modelSpec = capturedModel ?? "black-forest-labs/FLUX.2-klein-4B"
          let snapshot = try await ModelResolution.resolve(modelSpec: modelSpec)

          // Detect model config from snapshot
          guard let detected = Flux2ModelDetection.detect(at: snapshot) else {
            throw NSError(domain: "ZImageCLI", code: 1,
              userInfo: [NSLocalizedDescriptionKey: "Model at \(snapshot.path) is not a Flux 2 Klein model"])
          }
          logger.info("Detected Flux 2 Klein \(detected.variant)")

          let flux2Pipeline = Flux2Pipeline(logger: logger)
          try flux2Pipeline.loadModel(
            from: snapshot,
            config: detected.transformerConfig,
            textEncoderConfig: detected.textEncoderConfig,
            isBase: detected.isBaseModel
          )

          // Load LoRAs for Flux 2
          if !loraConfigs.isEmpty {
            logger.info("Loading \(loraConfigs.count) LoRA(s) for Flux 2...")
            try await flux2Pipeline.loadLoRAs(loraConfigs)
          }

          // Validate guidance for distilled models
          if flux2Pipeline.isDistilled && guidance != 1.0 && guidance != ZImageModelMetadata.recommendedGuidanceScale {
            logger.warning("Guidance scale \(guidance) has no effect on distilled Klein models (forcing 1.0)")
          }

          // Resolve img2img for Flux 2
          let flux2InputImage: URL? = initImagePath.map { URL(fileURLWithPath: $0) }
          let flux2DenoiseValue: Float = flux2InputImage != nil ? (flux2Denoise ?? 0.7) : 1.0

          if let img = flux2InputImage {
            guard FileManager.default.fileExists(atPath: img.path) else {
              throw NSError(domain: "ZImageCLI", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Init image not found: \(img.path)"])
            }
            logger.info("Flux 2 img2img: source=\(img.path), denoise=\(flux2DenoiseValue)")
          }

          // Override request defaults based on detected model type
          let flux2Request = Flux2GenerationRequest(
            prompt: prompt,
            negativePrompt: negativePrompt,
            width: width,
            height: height,
            steps: steps == ZImageModelMetadata.recommendedInferenceSteps ? flux2Pipeline.defaultSteps : steps,
            guidanceScale: flux2Pipeline.isDistilled ? 1.0 : guidance,
            seed: seed,
            outputPath: URL(fileURLWithPath: outputPath),
            levelsMin: levelsMin,
            levelsMax: levelsMax,
            maxSequenceLength: maxSequenceLength,
            inputImagePath: flux2InputImage,
            denoise: flux2DenoiseValue
          )

          _ = try await flux2Pipeline.generate(flux2Request, progressHandler: { progress in
            guard !noProgress else { return }
            guard progress.stage == .denoising else { return }
            let completed = min(progress.totalSteps, max(0, progress.stepIndex))
            if let bar {
              bar.update(completed: completed)
              if completed == progress.totalSteps {
                bar.finish(forceNewline: true)
              }
            } else {
              PlainProgress.shared.report(completed: completed, total: progress.totalSteps)
            }
          })
          if let bar { bar.finish(forceNewline: true) }
        } catch {
          logger.error("Flux 2 generation failed: \(error)")
          capturedError = error
          if let bar { bar.finish(forceNewline: true) }
        }
        flux2Semaphore.signal()
      }
      flux2Semaphore.wait()
      if let err = capturedError {
        fputs("Error: \(err)\n", stderr)
        exit(1)
      }
      return
    }

    // === SINGLE IMAGE PATH ===
    let request = ZImageGenerationRequest(
      prompt: prompt,
      negativePrompt: negativePrompt,
      width: width,
      height: height,
      steps: steps,
      guidanceScale: guidance,
      seed: seed,
      outputPath: URL(fileURLWithPath: outputPath),
      levelsMin: levelsMin,
      levelsMax: levelsMax,
      model: model,
      textEncoderPath: textEncoderPath,
      maxSequenceLength: maxSequenceLength,
      loras: loraConfigs,
      enhancePrompt: enhancePrompt,
      enhanceMaxTokens: enhanceMaxTokens,
      forceTransformerOverrideOnly: forceTransformerOverrideOnly,
      schedulerKind: schedulerKind,
      sigmaSchedule: sigmaSchedule,
      eta: eta,
      dyPE: dyPEConfig
    )

    let pipeline = ZImagePipeline(logger: logger)
    nonisolated(unsafe) let semaphore = DispatchSemaphore(value: 0)
    // Capture values for metadata (before Task to avoid Sendable issues)
    let shouldWriteMetadata = writeMetadata
    let capturedSeed = seed
    let capturedPrompt = prompt
    let capturedNegativePrompt = negativePrompt
    let capturedSteps = steps
    let capturedGuidance = guidance
    let capturedWidth = width
    let capturedHeight = height
    let capturedScheduler = schedulerKind.rawValue
    let capturedSigmaSchedule: String? = sigmaSchedule == .flow ? nil : sigmaSchedule.rawValue
    let capturedModel = model
    let capturedLoraConfigs = loraConfigs
    let capturedStartTime = Date()
    let useBar = !noProgress && (isatty(STDERR_FILENO) != 0)
    let bar = useBar ? ProgressBar(total: steps) : nil
    let finalOutputPath = URL(fileURLWithPath: outputPath)
    let shouldGenerateSVG = generateSVG
    let svgPresetCopy = svgPreset
    var capturedError: (any Error)? = nil
    Task {
      do {
        _ = try await pipeline.generate(request, progressHandler: { progress in
          guard !noProgress else { return }
          guard progress.stage == .denoising else { return }
          let completed = min(progress.totalSteps, max(0, progress.stepIndex))

          if let bar {
            bar.update(completed: completed)
            if completed == progress.totalSteps {
              bar.finish(forceNewline: true)
            }
          } else {
            PlainProgress.shared.report(completed: completed, total: progress.totalSteps)
          }
        })
        if let bar { bar.finish(forceNewline: true) }
        if shouldWriteMetadata {
          let loraInfos = capturedLoraConfigs.map { lora -> LoRAInfo in
            let path: String
            switch lora.source {
            case .local(let url): path = url.path
            case .huggingFace(let id, _): path = id
            }
            return LoRAInfo(path: path, scale: lora.scale)
          }
          let metadata = GenerationMetadata(
            pipeline: .txt2img,
            model: ModelInfo(
              family: "zimage",
              variant: (capturedModel.map { ZImageRepository.isBaseModel($0) ? "base" : "turbo" }) ?? "turbo",
              path: capturedModel ?? ZImageRepository.id
            ),
            parameters: GenerationParameters(
              prompt: capturedPrompt,
              negativePrompt: capturedNegativePrompt,
              seed: capturedSeed ?? 0,
              steps: capturedSteps,
              guidance: capturedGuidance,
              width: capturedWidth,
              height: capturedHeight,
              scheduler: capturedScheduler,
              sigmaSchedule: capturedSigmaSchedule
            ),
            loras: loraInfos,
            output: OutputInfo(
              path: finalOutputPath.path,
              width: capturedWidth,
              height: capturedHeight,
              renderTimeSeconds: Date().timeIntervalSince(capturedStartTime)
            )
          )
          if let written = MetadataWriter.write(metadata, for: finalOutputPath.path) {
            logger.info("Metadata written: \(written)")
          }
        }
        if shouldGenerateSVG {
          let svgOutputPath = finalOutputPath.deletingPathExtension().appendingPathExtension("svg")
          logger.info("Converting to SVG: \(svgOutputPath.path)")
          try convertToSVG(input: finalOutputPath, output: svgOutputPath, preset: svgPresetCopy)
          logger.info("SVG generated: \(svgOutputPath.path)")
        }
      } catch {
        logger.error("Generation failed: \(error)")
        capturedError = error
        if let bar { bar.finish(forceNewline: true) }
      }
      semaphore.signal()
    }
    semaphore.wait()
    if let err = capturedError {
      fputs("Error: \(err)\n", stderr)
      exit(1)
    }
  }

  private static func convertToSVG(input: URL, output: URL, preset: String) throws {
    try SVGExporter.convert(input: input, output: output, preset: preset)
  }

  private static func runAudit(modelSpec: String?, textEncoderPath: String?) async throws {
    let report = try await ZImageModelAuditor.audit(
      modelSpec: modelSpec,
      textEncoderPath: textEncoderPath,
      logger: logger
    )
    print(report.formattedDescription())
  }

  private static func printUsage() {
    print("""
    Z-Image-Turbo Swift port

    Usage: ComfyBox --prompt "text" [options]
      --prompt, -p           Text prompt (required)
      --negative-prompt      Negative prompt
      --width, -W            Output width (default \(ZImageModelMetadata.recommendedWidth))
      --height, -H           Output height (default \(ZImageModelMetadata.recommendedHeight))
      --steps, -s            Inference steps (default \(ZImageModelMetadata.recommendedInferenceSteps))
      --guidance, -g         Guidance scale (default \(ZImageModelMetadata.recommendedGuidanceScale))
      --seed                 Random seed
      --output, -o           Output path (default z-image.png)
      --levels-min           Levels lower bound for post-decode contrast adjustment (default: 0.0)
      --levels-max           Levels upper bound for post-decode contrast adjustment (default: 1.0)
      --model, -m            Model path or HuggingFace ID (default: \(ZImageRepository.id))
                             Aliases: z-image-base (Base, CFG-guided), z-image-turbo (Turbo, distilled)
      --model-family         Model family: flux1, flux2, fibo, or chroma (default: auto-detect from model config)
      --text-encoder-path    Override text encoder directory (CLI > ZIMAGE_ENCODER_PATH > auto-detect > default)
      --force-transformer-override-only  Treat a local .safetensors as transformer-only override (disable AIO auto-detect)
      --cache-limit          GPU memory cache limit in MB (default: unlimited)
      --max-sequence-length  Maximum sequence length for text encoding (default: 512)
      --lora, -l             LoRA path or HuggingFace ID (repeatable, prefer path=scale; path:scale is legacy)
      --lora-scale           LoRA scale factor override for the next unmatched --lora (repeatable)
      --lora-paths           Comma-separated LoRA paths or HuggingFace IDs (quoted commas unsupported)
      --lora-scales          Comma-separated LoRA scale overrides (default: 1.0)
      --scheduler, --sampler  Sampler algorithm: euler, heun, res_2s, dpmpp-2m, dpmpp-2s-a, deis, ddim (default: euler)
                              (res_3s and ralston_2s/3s/4s are Krea 2 only — they take several model
                               evaluations per step and this path cannot drive them)
      --sigma-schedule       Sigma schedule: flow, karras, exponential, beta, beta57 (default: flow)
      --eta                  Stochasticity for DDIM/DPM++ 2S-A (0=deterministic, 1=DDPM; default: 0)
      --dype <method>        DyPE high-res mode: ntk, yarn, none (auto-enables for >1024px)
      --no-dype              Disable DyPE even for high-res generation
      --enhance, -e          Enhance prompt using LLM (requires ~5GB extra VRAM)
      --enhance-max-tokens   Max tokens for prompt enhancement (default: 512)
      --no-progress          Disable progress output
      --audit-weights        Audit transformer/text encoder/VAE weight coverage and exit
      --svg                  Also generate SVG vector output (requires vtracer)
      --svg-preset           SVG preset: default, logo, detailed, simplified, bw
      --metadata             Write generation metadata JSON sidecar alongside output image
      --config-from-metadata Load generation parameters from a metadata JSON sidecar (CLI flags override)
      --auto-seeds <N>       Generate N random seeds for batch run
      --seed                 Random seed (repeatable for batch: --seed 42 --seed 99)
      --resume-batch <path>  Resume batch from checkpoint file
      --prompt-file <path>   Re-read prompt from file before each batch iteration

    Img2img (Flux 1 / Z-Image):
      --image-path, --image  Source image for img2img transformation
      --image-strength       Strength (0.0-1.0): 0.3=heavy rework (default), 0.7=light touch
      --creativity           Creativity (0.0-1.0): 0.7=heavy rework, 0.3=light touch (inverse of strength)
                             Cannot be used with --image-strength

    Img2img (Flux 2 Klein):
      --init-image           Source image for Flux 2 img2img (VAE encode + BN normalize)
      --denoise              Denoise strength (0.0-1.0, default 0.7): 1.0=full txt2img, 0.5=preserve composition
      --help, -h             Show help

    Subcommands:
      quantize               Quantize model weights
        --input, -i          Input model directory (required)
        --output, -o         Output directory (required)
        --bits               Bit width: 4 or 8 (default: 8)
        --group-size         Group size: 32, 64, 128 (default: 32)
        --verbose            Show progress

      quantize-controlnet    Quantize ControlNet weights
        --input, -i          Input ControlNet path or HuggingFace ID (required)
        --output, -o         Output directory (required)
        --bits               Bit width: 4 or 8 (default: 8)
        --group-size         Group size: 32, 64, 128 (default: 32)
        --verbose            Show progress

      control                Generate with ControlNet conditioning
        --prompt, -p         Text prompt (required)
        --control-image, -c  Control image path (required)
        --controlnet-weights Path to controlnet weights dir or HF ID (required)
        --control-scale      Control scale (default: 0.75)
        Use 'ComfyBox control --help' for full options

      serve                  Start warm HTTP server
        --model, -m          Model path or HuggingFace ID
        --text-encoder-path  Override text encoder directory
        --port               HTTP port (default 7870)
        --lora, -l           Initial LoRA(s)
        Use 'ComfyBox serve --help' for full options

      upscale                Upscale image via SeedVR2 or ESRGAN
        --input, -i          Input image path (required)
        --output, -o         Output image path (default: input-upscaled.png)
        --resolution, -r     Target resolution (default: 2048)
        --steps              Inference steps (default: 1)
        --seed               Random seed
        --weights, -w        Path to SeedVR2 model weights directory
        --esrgan-weights     Path to ESRGAN safetensors directory
        --tile-size          ESRGAN tile size (default: 512)
        --softness           Preprocessing softness 0.0-1.0 (default: 0.0)

      video                  Native LTX-2 video generation (T2V and I2V)
        -p, --prompt         Text/motion prompt (required)
        -i, --image          Source image for I2V mode
        -o, --output         Output .mp4 path (default: z-video.mp4)
        -d, --duration       T2V duration in seconds: 6-20 (default: 6)
        -r, --resolution     Resolution: 480p, 720p, 1080p (default: 720p)
        --aspect-ratio       16:9 or 9:16 (default: 16:9)
        Use 'ComfyBox video --help' for full options

      models                 List known model families with installation status
        --paths, -v          Show filesystem paths for installed models


      mcp                    Start MCP server (stdio JSON-RPC bridge to WarmServer)
        --port               WarmServer port (default: 7870)
        --host               WarmServer host (default: 127.0.0.1)
        Use 'ComfyBox mcp --help' for full options

      telegram               Start Telegram bot (receives prompts, renders via WarmServer)
        --bot-token          Telegram Bot API token
        --config             Config file path (default: ~/.comfybox/telegram.json)
        --port               WarmServer port (default: 7870)
        --host               WarmServer host (default: 127.0.0.1)
        Use 'ComfyBox telegram --help' for full options

    Examples:
      ComfyBox -p "a cute cat" -o cat.png
      ComfyBox -p "a sunset" -m models/z-image-turbo-q8
      ComfyBox -p "a forest" -m Tongyi-MAI/Z-Image-Turbo
      ComfyBox -p "a cut a cat" --lora ostris/z_image_turbo_childrens_drawings
      ComfyBox -p "portrait" --lora mood.safetensors=0.8 --lora detail.safetensors --lora-scale 0.3
      ComfyBox -p "cat" --enhance  # Enhanced prompt generation
      ComfyBox -p "portrait" --scheduler dpmpp-2m --sigma-schedule beta  # Best photorealism combo
      ComfyBox -p "landscape" --scheduler heun --sigma-schedule beta -s 5  # Heun at half steps
      ComfyBox -p "refiner pass" --scheduler res_2s --sigma-schedule beta57  # RES 2s + beta57
      ComfyBox -p "scene" --scheduler ddim --eta 0.5  # Semi-stochastic DDIM
      ComfyBox serve -m ./models/z-image-turbo --port 7870
      ComfyBox -p "portrait" --auto-seeds 5 -o portraits.png  # Generate 5 random variations
      ComfyBox -p "cat" --seed 42 --seed 99 --seed 123 -o cats.png  # 3 specific seeds
      ComfyBox -p "scene" --auto-seeds 10 --resume-batch progress.jsonl  # Resume interrupted batch
      ComfyBox upscale -i photo.jpg -w ./models/seedvr2 -r 2048
      ComfyBox video -p "a woman walks through a sunlit garden" -o garden.mp4
      ComfyBox video -p "she turns and smiles" -i photo.png -o smile.mp4
      ComfyBox video -p "ocean waves" -d 10 -r 1080p --seed 42
    """)
  }

  private static func runQuantize(args: [String]) throws {
    var input: String?
    var output: String?
    var bits = 8
    var groupSize = 32
    var verbose = false

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--input", "-i":
        input = nextValue(for: arg, iterator: &iterator)
      case "--output", "-o":
        output = nextValue(for: arg, iterator: &iterator)
      case "--bits":
        bits = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: bits)
      case "--group-size":
        groupSize = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: groupSize)
      case "--verbose":
        verbose = true
      case "--help", "-h":
        printQuantizeUsage()
        return
      default:
        logger.warning("Unknown quantize argument: \(arg)")
      }
    }

    guard let inputPath = input else {
      logger.error("Missing required --input argument")
      printQuantizeUsage()
      exit(1)
    }

    guard let outputPath = output else {
      logger.error("Missing required --output argument")
      printQuantizeUsage()
      exit(1)
    }

    let inputURL = URL(fileURLWithPath: inputPath)
    let outputURL = URL(fileURLWithPath: outputPath)

    guard FileManager.default.fileExists(atPath: inputURL.path) else {
      logger.error("Input directory not found: \(inputPath)")
      exit(1)
    }

    guard ZImageQuantizer.supportedBits.contains(bits) else {
      logger.error("Invalid bits: \(bits). Supported: 4, 8")
      exit(1)
    }

    guard ZImageQuantizer.supportedGroupSizes.contains(groupSize) else {
      logger.error("Invalid group size: \(groupSize). Supported: 32, 64, 128")
      exit(1)
    }

    let spec = ZImageQuantizationSpec(groupSize: groupSize, bits: bits, mode: .affine)

    print("Quantizing: \(inputPath) -> \(outputPath)")
    print("Config: \(bits)-bit, group_size=\(groupSize)")

    try ZImageQuantizer.quantizeAndSave(
      from: inputURL,
      to: outputURL,
      spec: spec,
      verbose: verbose
    )

    print("Done: \(outputURL.path)")
  }

  private static func printQuantizeUsage() {
    print("""
    Quantize model weights.

    Usage: ComfyBox quantize -i <input> -o <output> [options]
      --input, -i          Input model directory (required)
      --output, -o         Output directory (required)
      --bits               Bit width: 4 or 8 (default: 8)
      --group-size         Group size: 32, 64, 128 (default: 32)
      --verbose            Show progress
      --help, -h           Show help

    Example:
      ComfyBox quantize -i models/z-image-turbo -o models/z-image-turbo-q8 --verbose
    """)
  }

  private static func runQuantizeLTX2(args: [String]) throws {
    var input: String?
    var output: String?
    var bits = 8
    var groupSize = 64
    var verbose = false

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--input", "-i":
        input = nextValue(for: arg, iterator: &iterator)
      case "--output", "-o":
        output = nextValue(for: arg, iterator: &iterator)
      case "--bits":
        bits = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: bits)
      case "--group-size":
        groupSize = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: groupSize)
      case "--verbose":
        verbose = true
      case "--help", "-h":
        printQuantizeLTX2Usage()
        return
      default:
        logger.warning("Unknown quantize-ltx2 argument: \(arg)")
      }
    }

    guard let inputPath = input, let outputPath = output else {
      logger.error("Missing required --input/--output arguments")
      printQuantizeLTX2Usage()
      exit(1)
    }

    // Accept a checkpoint file directly, or a directory holding a monolith
    // (resolved the same way LTX2VideoGenerator does — first non-component
    // .safetensors).
    var sourceURL = URL(fileURLWithPath: inputPath)
    if (try? sourceURL.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
      let perComponent: Set<String> = [
        "vae_encoder.safetensors", "vae_decoder.safetensors", "connector.safetensors",
      ]
      let candidates = ((try? FileManager.default.contentsOfDirectory(
        at: sourceURL, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "safetensors" && !perComponent.contains($0.lastPathComponent) }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
      guard let monolith = candidates.first else {
        logger.error("No checkpoint .safetensors found in \(inputPath)")
        exit(1)
      }
      sourceURL = monolith
    }

    print("Quantizing LTX-2 DiT: \(sourceURL.path) -> \(outputPath)")
    print("Config: \(bits)-bit affine, group_size=\(groupSize)")
    try LTX2Quantizer.quantizeCheckpoint(
      source: sourceURL,
      outputDir: URL(fileURLWithPath: outputPath),
      spec: .init(bits: bits, groupSize: groupSize),
      verbose: verbose
    )
  }

  private static func printQuantizeLTX2Usage() {
    print("""
    Quantize an LTX-2 checkpoint's video-DiT block projections to MLX affine int8/int4.
    Norms, embeds, final proj, audio branch, VAE, and vocoder stay bf16. The output
    keeps the source key layout, so --ltx2-weights can point at the output directory.

    Usage: ComfyBox quantize-ltx2 -i <checkpoint|dir> -o <output-dir> [options]
      --input, -i          Checkpoint .safetensors or directory holding a monolith (required)
      --output, -o         Output directory (required)
      --bits               Bit width: 4 or 8 (default: 8)
      --group-size         Group size: 32, 64, 128 (default: 64)
      --verbose            Show per-layer progress
      --help, -h           Show help

    Example:
      ComfyBox quantize-ltx2 -i /Volumes/Bolt/Models/sulphur2-distil -o /Volumes/Bolt/Models/sulphur2-distil-int8
    """)
  }

  private static func runQuantizeControlnet(args: [String]) throws {
    var input: String?
    var output: String?
    var specificFile: String?
    var bits = 8
    var groupSize = 32
    var verbose = false

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--input", "-i":
        input = nextValue(for: arg, iterator: &iterator)
      case "--output", "-o":
        output = nextValue(for: arg, iterator: &iterator)
      case "--file", "-f":
        specificFile = nextValue(for: arg, iterator: &iterator)
      case "--bits":
        bits = intValue(for: arg, iterator: &iterator, minimum: 4, fallback: 8)
      case "--group-size":
        groupSize = intValue(for: arg, iterator: &iterator, minimum: 32, fallback: 32)
      case "--verbose":
        verbose = true
      case "--help", "-h":
        printQuantizeControlnetUsage()
        return
      default:
        logger.warning("Unknown quantize-controlnet argument: \(arg)")
      }
    }

    guard let inputPath = input else {
      logger.error("Missing required --input argument")
      printQuantizeControlnetUsage()
      return
    }

    guard let outputPath = output else {
      logger.error("Missing required --output argument")
      printQuantizeControlnetUsage()
      return
    }

    let outputURL = URL(fileURLWithPath: outputPath)
    let spec = ZImageQuantizationSpec(groupSize: groupSize, bits: bits, mode: .affine)

    print("Quantizing ControlNet: \(inputPath) -> \(outputPath)")
    print("Config: \(bits)-bit, group_size=\(groupSize)")

    nonisolated(unsafe) let semaphore = DispatchSemaphore(value: 0)
    let errorBox = Box<Error?>(nil)
    let capturedVerbose = verbose
    let capturedSpecificFile = specificFile

    Task {
      do {

        let sourceURL: URL
        let localURL = URL(fileURLWithPath: inputPath)

        if FileManager.default.fileExists(atPath: localURL.path) {

          sourceURL = localURL
          if capturedVerbose {
            print("Using local ControlNet: \(inputPath)")
          }
        } else if ModelResolution.isHuggingFaceModelId(inputPath) {

          if capturedVerbose {
            print("Downloading ControlNet from HuggingFace: \(inputPath)")
          }
          sourceURL = try await ModelResolution.resolve(
            modelSpec: inputPath,
            filePatterns: ["*.safetensors", "*.json"],
            progressHandler: { progress in
              let percent = Int(progress.fractionCompleted * 100)
              print("Downloading: \(percent)%")
            }
          )
          if capturedVerbose {
            print("Downloaded to: \(sourceURL.path)")
          }
        } else {
          logger.error("Input not found: \(inputPath). Provide a local path or HuggingFace model ID.")
          semaphore.signal()
          return
        }

        try ZImageQuantizer.quantizeControlnet(
          from: sourceURL,
          to: outputURL,
          spec: spec,
          specificFile: capturedSpecificFile,
          verbose: capturedVerbose
        )

        print("Done: \(outputURL.path)")
      } catch {
        errorBox.value = error
        logger.error("Quantization failed: \(error)")
      }
      semaphore.signal()
    }

    semaphore.wait()
    if let error = errorBox.value {
      throw error
    }
  }

  private static func printQuantizeControlnetUsage() {
    print("""
    Quantize ControlNet weights.

    Usage: ComfyBox quantize-controlnet -i <input> -o <output> [options]
      --input, -i          Input ControlNet path or HuggingFace ID (required)
      --output, -o         Output directory (required)
      --file, -f           Specific .safetensors file to quantize (optional)
      --bits               Bit width: 4 or 8 (default: 8)
      --group-size         Group size: 32, 64, 128 (default: 32)
      --verbose            Show progress
      --help, -h           Show help

    Examples:
      # From HuggingFace
      ComfyBox quantize-controlnet -i alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
        --file Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors -o controlnet-2.1-q8 --verbose

      # From local directory
      ComfyBox quantize-controlnet -i ./controlnet-union -o ./controlnet-union-q8 --verbose
    """)
  }

  private static func runServe(args: [String]) throws {
    // Load ~/.comfybox/config.json, auto-migrating from ~/.coffeeshop on first launch.
    // Config supplies defaults; explicit CLI flags below override them.
    let config = ComfyBoxServerConfig.loadOrMigrate()
    // Seed the ONE Krea-2 spec→directory table from config (WP-E5).
    Krea2ModelDetection.configureSpecDirectories(config.krea2Models)

    var model: String? = config.modelSpec
    var textEncoderPath: String?
    var port: UInt16 = config.port
    var cacheLimit: Int?
    var maxSequenceLength = 512
    var loraEntries: [String] = []
    var loraScaleOverrides: [Float] = []
    var forceTransformerOverrideOnly = false
    var host = config.host
    var allowedOutputDirectory = config.allowedOutputDirectory ?? FileManager.default.currentDirectoryPath
    var seedvr2Weights: String? = config.seedvr2WeightsPath
    var ltx2Weights: String? = nil
    var ltx2Gemma: String? = nil
    var ltx2DefaultLoRA: String? = nil
    var civitaiKey: String? = nil

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--model", "-m":
        model = nextValue(for: arg, iterator: &iterator)
      case "--text-encoder-path":
        textEncoderPath = nextValue(for: arg, iterator: &iterator)
      case "--port":
        let rawPort = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: Int(port))
        guard rawPort <= Int(UInt16.max) else {
          logger.error("Invalid port: \(rawPort)")
          return
        }
        port = UInt16(rawPort)
      case "--host":
        host = nextValue(for: arg, iterator: &iterator)
      case "--allowed-output-directory":
        allowedOutputDirectory = nextValue(for: arg, iterator: &iterator)
      case "--cache-limit":
        cacheLimit = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: 2048)
      case "--max-sequence-length":
        maxSequenceLength = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: 512)
      case "--force-transformer-override-only":
        forceTransformerOverrideOnly = true
      case "--lora", "-l":
        loraEntries.append(nextValue(for: arg, iterator: &iterator))
      case "--lora-scale":
        loraScaleOverrides.append(floatValue(for: arg, iterator: &iterator, fallback: 1.0))
      case "--lora-paths":
        loraEntries.append(contentsOf: splitCommaSeparated(nextValue(for: arg, iterator: &iterator)))
      case "--lora-scales":
        loraScaleOverrides.append(contentsOf: floatListValue(for: arg, iterator: &iterator, fallback: 1.0))
      case "--seedvr2-weights":
        seedvr2Weights = nextValue(for: arg, iterator: &iterator)
      case "--ltx2-weights":
        ltx2Weights = nextValue(for: arg, iterator: &iterator)
      case "--ltx2-gemma":
        ltx2Gemma = nextValue(for: arg, iterator: &iterator)
      case "--ltx2-lora":
        ltx2DefaultLoRA = nextValue(for: arg, iterator: &iterator)
      case "--civitai-key":
        civitaiKey = nextValue(for: arg, iterator: &iterator)
      case "--help", "-h":
        printServeUsage()
        return
      default:
        logger.warning("Unknown serve argument: \(arg)")
      }
    }

    if port == ComfyBoxServerConfig.deprecatedAliasPort {
      logger.warning("Port \(port) is deprecated; ComfyBox's canonical port is \(ComfyBoxServerConfig.canonicalPort). It still works this release — update clients (Krita, warm-worker) to \(ComfyBoxServerConfig.canonicalPort).")
    }

    if let limit = cacheLimit {
      GPU.set(cacheLimit: limit * 1024 * 1024)
      logger.info("GPU cache limit set to \(limit)MB")
    }

    let loraConfigs = buildLoRAConfigurations(entries: loraEntries, scaleOverrides: loraScaleOverrides)
    if !loraConfigs.isEmpty {
      logger.info("Preloading \(loraConfigs.count) LoRA(s)")
    }

    let configuration = WarmServerConfiguration(
      port: port,
      modelSpec: model,
      textEncoderPath: textEncoderPath,
      initialLoRAs: loraConfigs,
      forceTransformerOverrideOnly: forceTransformerOverrideOnly,
      maxSequenceLength: maxSequenceLength,
      maxPendingRequests: 10,
      allowedOutputDirectory: allowedOutputDirectory,
      seedvr2WeightsPath: seedvr2Weights,
      ltx2WeightsPath: ltx2Weights,
      ltx2GemmaPath: ltx2Gemma,
      ltx2DefaultLoRA: ltx2DefaultLoRA,
      civitaiApiKey: civitaiKey
    )

    // FDD-ui-api-parity §3.3: the ONE place the renderDefaults/videoDefaults
    // first-run migration is allowed to run — a real server boot, never a
    // unit test (see ServerConfigStore.runFirstRunDefaultsMigrationIfNeeded).
    ServerConfigStore.shared.runFirstRunDefaultsMigrationIfNeeded()

    let server = WarmServer(configuration: configuration, host: host, logger: logger)
    try server.run()
  }

  private static func printServeUsage() {
    print("""
    Start warm HTTP server mode.

    Usage: ComfyBox serve [options]
      --model, -m               Model path or HuggingFace ID (default: \(ZImageRepository.id))
      --text-encoder-path       Override text encoder directory
      --port                    HTTP port (default: 7870)
      --host                    HTTP host/interface to bind (default: 127.0.0.1)
      --allowed-output-directory  Directory where request outputPath values may write (default: current directory)
      --cache-limit             GPU memory cache limit in MB (default: unlimited)
      --max-sequence-length     Maximum sequence length for text encoding (default: 512)
      --force-transformer-override-only  Treat a local .safetensors as transformer-only override
      --seedvr2-weights            Path to SeedVR2 upscale model weights directory
      --ltx2-weights               LTX-2 weights dir OR "org/repo[:rev]" HF spec (enables LOCAL video on /v1/video/generate; lazy-loaded + auto-downloaded on first request, ~38GB)
      --ltx2-gemma                 Gemma-3 tokenizer/text-encoder snapshot dir OR HF spec for LTX-2 (required alongside --ltx2-weights; ~24GB, downloaded on first request, not swappable for a different encoder)
      --ltx2-lora                  Default LoRA for LOCAL video renders when the request carries none, as "path" or "path@scale" (e.g. a distill LoRA for a non-distilled checkpoint). Request LoRAs override it.
      --civitai-key                 CivitAI API key for the /v1/civitai/* conduit routes + MCP civitai_search/civitai_prompts tools. Explicit flag wins over CIVITAI_API_KEY and the Desktop app's Keychain entry (see CivitAISecrets). Optional — those routes 503 with a clear message if no key resolves any of the three ways.
      --lora, -l                Initial LoRA path or HuggingFace ID (repeatable, prefer path=scale; path:scale is legacy)
      --lora-scale              LoRA scale factor override for the next unmatched --lora (repeatable)
      --lora-paths              Comma-separated LoRA paths or HuggingFace IDs
      --lora-scales             Comma-separated LoRA scale overrides (default: 1.0)
      --help, -h                Show help

    Endpoints:
      POST /v1/generate         Submit a render request
      POST /v1/lora/swap        Hot-swap active LoRAs
      GET  /health              Report model/server status
      POST /v1/shutdown         Gracefully stop the server
      GET  /v1/civitai/search   Search CivitAI models/LoRAs (needs a resolved API key)
      POST /v1/civitai/harvest  Harvest trained words + descriptions into the local prompt repository
      GET  /v1/civitai/repo     Query the local prompt repository (baseModel/act/tag/keyword filters)

    Example:
      ComfyBox serve -m /path/to/model --text-encoder-path /path/to/encoder --port 7870 \\
        --lora /path/to/lora.safetensors=0.8
    """)
  }

  private static func runControl(args: [String]) throws {
    var prompt: String?
    var negativePrompt: String?
    var controlImage: String?
    var inpaintImage: String?
    var maskImage: String?
    var controlScale: Float = 0.75
    var controlnetWeights: String?
    var controlnetWeightsFile: String?
    var width = ZImageModelMetadata.recommendedWidth
    var height = ZImageModelMetadata.recommendedHeight
    var steps = ZImageModelMetadata.recommendedInferenceSteps
    var guidance = ZImageModelMetadata.recommendedGuidanceScale
    var seed: UInt64?
    var outputPath = "z-image-control.png"
    var levelsMin: Float = 0.0
    var levelsMax: Float = 1.0
    var model: String?
    var textEncoderPath: String?
    var cacheLimit: Int?
    var maxSequenceLength = 512
    var loraEntries: [String] = []
    var loraScaleOverrides: [Float] = []
    var noProgress = false
    var schedulerKind: SchedulerKind = .euler
    var sigmaSchedule: SigmaScheduleKind = .flow
    var eta: Float?

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--prompt", "-p":
        prompt = nextValue(for: arg, iterator: &iterator)
      case "--negative-prompt", "--np":
        negativePrompt = nextValue(for: arg, iterator: &iterator)
      case "--control-image", "-c":
        controlImage = nextValue(for: arg, iterator: &iterator)
      case "--inpaint-image", "-i":
        inpaintImage = nextValue(for: arg, iterator: &iterator)
      case "--mask", "--mask-image":
        maskImage = nextValue(for: arg, iterator: &iterator)
      case "--control-scale", "--cs":
        controlScale = floatValue(for: arg, iterator: &iterator, fallback: 0.75)
      case "--controlnet-weights", "--cw":
        controlnetWeights = nextValue(for: arg, iterator: &iterator)
      case "--control-file", "--cf":
        controlnetWeightsFile = nextValue(for: arg, iterator: &iterator)
      case "--width", "-W":
        width = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: width)
      case "--height", "-H":
        height = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: height)
      case "--steps", "-s":
        steps = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: steps)
      case "--guidance", "-g":
        guidance = floatValue(for: arg, iterator: &iterator, fallback: guidance)
      case "--seed":
        seed = uint64Value(for: arg, iterator: &iterator)
      case "--output", "-o":
        outputPath = nextValue(for: arg, iterator: &iterator)
      case "--levels-min":
        levelsMin = floatValue(for: arg, iterator: &iterator, fallback: 0.0)
      case "--levels-max":
        levelsMax = floatValue(for: arg, iterator: &iterator, fallback: 1.0)
      case "--model", "-m":
        model = nextValue(for: arg, iterator: &iterator)
      case "--text-encoder-path":
        textEncoderPath = nextValue(for: arg, iterator: &iterator)
      case "--cache-limit":
        cacheLimit = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: 2048)
      case "--max-sequence-length":
        maxSequenceLength = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: 512)
      case "--lora", "-l":
        loraEntries.append(nextValue(for: arg, iterator: &iterator))
      case "--lora-scale":
        loraScaleOverrides.append(floatValue(for: arg, iterator: &iterator, fallback: 1.0))
      case "--lora-paths":
        loraEntries.append(contentsOf: splitCommaSeparated(nextValue(for: arg, iterator: &iterator)))
      case "--lora-scales":
        loraScaleOverrides.append(contentsOf: floatListValue(for: arg, iterator: &iterator, fallback: 1.0))
      case "--no-progress":
        noProgress = true
      case "--scheduler", "--sampler":
        let raw = nextValue(for: arg, iterator: &iterator)
        guard let kind = SchedulerKind(rawValue: raw) else {
          let valid = SchedulerKind.allCases.map(\.rawValue).joined(separator: ", ")
          failArgumentParsing("Unknown scheduler '\(raw)'. Valid: \(valid)")
        }
        // WP-E13: the N-row tableau samplers are dispatched only by the Krea 2
        // denoise loop. This path drives ZImagePipeline / ZImageControlPipeline,
        // which take one model evaluation per step — accepting the name here
        // would render first-order Euler under it. Refuse now, before weights.
        if kind.isNRowTableau {
          let usable = SchedulerKind.allCases.filter { !$0.isNRowTableau }
            .map(\.rawValue).joined(separator: ", ")
          failArgumentParsing(
            "Sampler '\(raw)' is an N-row tableau sampler supported only by the krea2 model "
              + "family (`ComfyBox krea2`, or a krea2 model on the server); the Z-Image path "
              + "takes one model evaluation per step and cannot run it. Valid here: \(usable)")
        }
        schedulerKind = kind
      case "--sigma-schedule":
        let raw = nextValue(for: arg, iterator: &iterator)
        guard let kind = SigmaScheduleKind(rawValue: raw) else {
          let valid = SigmaScheduleKind.allCases.map(\.rawValue).joined(separator: ", ")
          failArgumentParsing("Unknown sigma schedule '\(raw)'. Valid: \(valid)")
        }
        sigmaSchedule = kind
      case "--eta":
        eta = floatValue(for: arg, iterator: &iterator, fallback: 0.0)
      case "--help", "-h":
        printControlUsage()
        return
      default:
        logger.warning("Unknown control argument: \(arg)")
      }
    }

    guard let prompt else {
      logger.error("Missing required --prompt argument")
      printControlUsage()
      exit(1)
    }
    if controlImage == nil && inpaintImage == nil && maskImage == nil {
      logger.error("At least one of --control-image, --inpaint-image, or --mask must be provided")
      printControlUsage()
      exit(1)
    }

    guard let controlnetWeights else {
      logger.error("Missing required --controlnet-weights argument")
      printControlUsage()
      exit(1)
    }
    var controlImageURL: URL? = nil
    if let controlImage {
      controlImageURL = URL(fileURLWithPath: controlImage)
      guard FileManager.default.fileExists(atPath: controlImageURL!.path) else {
        logger.error("Control image not found: \(controlImage)")
        exit(1)
      }
    }
    var inpaintImageURL: URL? = nil
    if let inpaintImage {
      inpaintImageURL = URL(fileURLWithPath: inpaintImage)
      guard FileManager.default.fileExists(atPath: inpaintImageURL!.path) else {
        logger.error("Inpaint image not found: \(inpaintImage)")
        exit(1)
      }
    }
    var maskImageURL: URL? = nil
    if let maskImage {
      maskImageURL = URL(fileURLWithPath: maskImage)
      guard FileManager.default.fileExists(atPath: maskImageURL!.path) else {
        logger.error("Mask image not found: \(maskImage)")
        exit(1)
      }
    }

    if let limit = cacheLimit {
      GPU.set(cacheLimit: limit * 1024 * 1024)
      logger.info("GPU cache limit set to \(limit)MB")
    }

    let loraConfigs = buildLoRAConfigurations(entries: loraEntries, scaleOverrides: loraScaleOverrides)
    if !loraConfigs.isEmpty {
      logger.info("Using \(loraConfigs.count) LoRA(s)")
    }

    let useBar = !noProgress && (isatty(STDERR_FILENO) != 0)
    let bar = useBar ? ProgressBar(total: steps) : nil
    let barBox = Box<ProgressBar?>(bar)
    let disableProgress = noProgress
    let progressCallback: ControlProgressCallback?
    if disableProgress {
      progressCallback = nil
    } else {
      progressCallback = { progress in
        guard progress.stage == "Denoising" else { return }
        let completed = min(progress.totalSteps, max(0, progress.stepIndex))
        if let bar = barBox.value {
          bar.update(completed: completed)
          if completed == progress.totalSteps {
            bar.finish(forceNewline: true)
          }
        } else {
          PlainProgress.shared.report(completed: completed, total: progress.totalSteps)
        }
      }
    }

    let request = ZImageControlGenerationRequest(
      prompt: prompt,
      negativePrompt: negativePrompt,
      controlImage: controlImageURL,
      inpaintImage: inpaintImageURL,
      maskImage: maskImageURL,
      controlContextScale: controlScale,
      width: width,
      height: height,
      steps: steps,
      guidanceScale: guidance,
      seed: seed,
      outputPath: URL(fileURLWithPath: outputPath),
      levelsMin: levelsMin,
      levelsMax: levelsMax,
      model: model,
      textEncoderPath: textEncoderPath,
      controlnetWeights: controlnetWeights,
      controlnetWeightsFile: controlnetWeightsFile,
      maxSequenceLength: maxSequenceLength,
      loras: loraConfigs,
      progressCallback: progressCallback,
      schedulerKind: schedulerKind,
      sigmaSchedule: sigmaSchedule,
      eta: eta
    )

    let pipeline = ZImageControlPipeline(logger: logger)
    nonisolated(unsafe) let semaphore = DispatchSemaphore(value: 0)
    let finalOutputPath = outputPath
    var capturedError: (any Error)? = nil
    Task {
      do {
        _ = try await pipeline.generate(request)
        if let bar = barBox.value { bar.finish(forceNewline: true) }
        logger.info("Output saved to: \(finalOutputPath)")
      } catch {
        logger.error("Control generation failed: \(error)")
        capturedError = error
        if let bar = barBox.value { bar.finish(forceNewline: true) }
      }
      semaphore.signal()
    }
    semaphore.wait()
    if let err = capturedError {
      fputs("Error: \(err)\n", stderr)
      exit(1)
    }
  }

  private static func printControlUsage() {
    print("""
    Generate images with ControlNet conditioning (supports v2.0/v2.1 with inpainting).

    Usage: ComfyBox control --prompt "text" --controlnet-weights <path> [options]
      --prompt, -p              Text prompt (required)
      --negative-prompt, --np   Negative prompt
      --control-image, -c       Control image path - Canny, HED, Depth, Pose, or MLSD
      --inpaint-image, -i       Source image for inpainting (v2.0+)
      --mask, --mask-image      Mask image for inpainting (white=fill, black=preserve)
      --control-scale, --cs     Control context scale (default: 0.75, recommended: 0.65-0.90)
      --controlnet-weights, --cw Path to controlnet safetensors or HuggingFace ID (required)
      --control-file, --cf      Specific safetensors filename within repo (e.g., "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors")
      --width, -W               Output width (default \(ZImageModelMetadata.recommendedWidth))
      --height, -H              Output height (default \(ZImageModelMetadata.recommendedHeight))
      --steps, -s               Inference steps (default \(ZImageModelMetadata.recommendedInferenceSteps), increase for higher control scale)
      --guidance, -g            Guidance scale (default \(ZImageModelMetadata.recommendedGuidanceScale))
      --seed                    Random seed
      --output, -o              Output path (default z-image-control.png)
      --levels-min              Levels lower bound for post-decode contrast adjustment (default: 0.0)
      --levels-max              Levels upper bound for post-decode contrast adjustment (default: 1.0)
      --model, -m               Model path or HuggingFace ID (default: \(ZImageRepository.id))
      --text-encoder-path       Override text encoder directory (CLI > ZIMAGE_ENCODER_PATH > auto-detect > default)
      --cache-limit             GPU memory cache limit in MB (default: unlimited)
      --max-sequence-length     Maximum sequence length for text encoding (default: 512)
      --lora, -l                LoRA path or HuggingFace ID (repeatable, prefer path=scale; path:scale is legacy)
      --lora-scale              LoRA scale factor override for the next unmatched --lora (repeatable)
      --lora-paths              Comma-separated LoRA paths or HuggingFace IDs (quoted commas unsupported)
      --lora-scales             Comma-separated LoRA scale overrides (default: 1.0)
      --scheduler, --sampler    Sampler algorithm: euler, heun, res_2s, dpmpp-2m, dpmpp-2s-a, deis, ddim (default: euler)
                                (res_3s and ralston_2s/3s/4s are Krea 2 only — they take several
                                 model evaluations per step and this path cannot drive them)
      --sigma-schedule          Sigma schedule: flow, karras, exponential, beta, beta57 (default: flow)
      --eta                     Stochasticity for DDIM/DPM++ 2S-A (0=deterministic, 1=DDPM; default: 0)
      --no-progress             Disable progress output
      --help, -h                Show help

    Note: At least one of --control-image, --inpaint-image, or --mask must be provided.

    Control Types:
      The control image should be pre-processed according to the control type:
      - Canny: Edge detection output (white edges on black background)
      - HED: Holistically-nested edge detection output
      - Depth: Depth map (grayscale, closer=brighter or depth estimation output)
      - Pose: OpenPose/DWPose skeleton visualization
      - MLSD: Line segment detection output

    Examples:
      # T2I with pose control using v2.1 weights (recommended)
      ComfyBox control -p "a woman on a beach" -c pose.jpg \\
        --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
        --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors

      # I2I inpainting with pose control
      ComfyBox control -p "a dancer" -c pose.jpg -i photo.jpg --mask mask.png \\
        --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
        --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors --cs 0.75 -s 25

      # Inpainting without control guidance
      ComfyBox control -p "a cat sitting" -i photo.jpg --mask mask.png \\
        --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
        --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors

      # Using local controlnet weights
      ComfyBox control -p "a forest path" -c depth.jpg --cs 0.7 \\
        --cw ./controlnet-q8 -o forest.png

      # Custom encoder with stacked LoRAs
      ComfyBox control -p "portrait" -c pose.png --cw ./controlnet-q8 \\
        -m /path/to/z-image-turbo-bf16 --text-encoder-path "/path/to/z-image-turbo-bf16/text_encoder QWen Large" \\
        --lora mood.safetensors=0.8 --lora detail.safetensors --lora-scale 0.3
    """)
  }

  private static func nextValue(for arg: String, iterator: inout IndexingIterator<[String]>) -> String {
    guard let value = iterator.next() else {
      failArgumentParsing("Expected value after \(arg)")
    }
    return value
  }

  private static func failArgumentParsing(_ message: String) -> Never {
    fputs("Error: \(message)\n", stderr)
    exit(1)
  }

  private static func warnArgumentParsing(_ message: String) {
    fputs("Warning: \(message)\n", stderr)
  }

  private static func splitCommaSeparated(_ value: String) -> [String] {
    value
      .split(separator: ",")
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty }
  }

  private static func buildLoRAConfigurations(entries: [String], scaleOverrides: [Float]) -> [LoRAConfiguration] {
    guard !entries.isEmpty else { return [] }
    let paddedOverrides = scaleOverrides + Array(repeating: Float(1.0), count: max(0, entries.count - scaleOverrides.count))
    return entries.enumerated().map { index, entry in
      configuration(for: entry, scaleOverride: paddedOverrides[index])
    }
  }

  private static func configuration(for entry: String, scaleOverride: Float) -> LoRAConfiguration {
    let (source, embeddedScale) = parseLoRAEntry(entry)
    let scale = embeddedScale ?? scaleOverride
    if looksLikeLocalPath(source) {
      return .local((source as NSString).expandingTildeInPath, scale: scale)
    }
    return .huggingFace(source, scale: scale)
  }

  private static func parseLoRAEntry(_ rawValue: String) -> (source: String, scale: Float?) {
    if let separator = rawValue.lastIndex(of: "=") {
      let source = String(rawValue[..<separator])
      let scaleString = String(rawValue[rawValue.index(after: separator)...])
      if !source.isEmpty, let scale = Float(scaleString) {
        return (source, scale)
      }
      return (rawValue, nil)
    }

    if looksLikeLocalPath(rawValue) {
      return (rawValue, nil)
    }

    guard let separator = rawValue.lastIndex(of: ":") else {
      return (rawValue, nil)
    }
    let source = String(rawValue[..<separator])
    let scaleString = String(rawValue[rawValue.index(after: separator)...])
    guard !source.isEmpty, let scale = Float(scaleString) else {
      return (rawValue, nil)
    }
    return (source, scale)
  }

  private static func looksLikeLocalPath(_ value: String) -> Bool {
    if value.hasPrefix("/") || value.hasPrefix("./") || value.hasPrefix("../") || value.hasPrefix("~") {
      return true
    }
    if value.hasSuffix(".safetensors") || value.hasSuffix(".json") {
      return true
    }
    return FileManager.default.fileExists(atPath: (value as NSString).expandingTildeInPath)
  }

  private static func intValue(for arg: String, iterator: inout IndexingIterator<[String]>, minimum: Int, fallback: Int) -> Int {
    let raw = nextValue(for: arg, iterator: &iterator)
    guard let value = Int(raw) else {
      warnArgumentParsing("Invalid value '\(raw)' for \(arg); using \(fallback).")
      return fallback
    }
    if value < minimum {
      warnArgumentParsing("Invalid value '\(raw)' for \(arg); using minimum \(minimum).")
      return minimum
    }
    return value
  }

  private static func floatValue(for arg: String, iterator: inout IndexingIterator<[String]>, fallback: Float) -> Float {
    let raw = nextValue(for: arg, iterator: &iterator)
    guard let value = Float(raw), value.isFinite else {
      warnArgumentParsing("Invalid value '\(raw)' for \(arg); using \(fallback).")
      return fallback
    }
    return value
  }

  private static func floatListValue(for arg: String, iterator: inout IndexingIterator<[String]>, fallback: Float) -> [Float] {
    splitCommaSeparated(nextValue(for: arg, iterator: &iterator)).map { raw in
      guard let value = Float(raw), value.isFinite else {
        warnArgumentParsing("Invalid value '\(raw)' for \(arg); using \(fallback).")
        return fallback
      }
      return value
    }
  }

  private static func uint64Value(for arg: String, iterator: inout IndexingIterator<[String]>) -> UInt64? {
    let raw = nextValue(for: arg, iterator: &iterator)
    guard let value = UInt64(raw) else {
      warnArgumentParsing("Invalid value '\(raw)' for \(arg); ignoring it.")
      return nil
    }
    return value
  }

  private static func optionalIntValue(for arg: String, iterator: inout IndexingIterator<[String]>) -> Int? {
    let raw = nextValue(for: arg, iterator: &iterator)
    guard let value = Int(raw) else {
      warnArgumentParsing("Invalid value '\(raw)' for \(arg); ignoring it.")
      return nil
    }
    return value
  }

  // MARK: - Upscale Subcommand

  #if canImport(CoreGraphics)
  private static func runUpscale(args: [String]) throws {
    var inputPath: String?
    var outputPath: String?
    var resolution = 2048
    var steps = 1
    var seed: Int?
    var weightsPath: String?
    var esrganWeightsPath: String?
    var tileSize = 512
    var softness: Float = 0.0

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--input", "-i":
        inputPath = nextValue(for: arg, iterator: &iterator)
      case "--output", "-o":
        outputPath = nextValue(for: arg, iterator: &iterator)
      case "--resolution", "-r":
        resolution = intValue(for: arg, iterator: &iterator, minimum: 256, fallback: resolution)
      case "--steps":
        steps = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: steps)
      case "--seed":
        if let s = optionalIntValue(for: arg, iterator: &iterator) {
          seed = s
        }
      case "--weights", "-w":
        weightsPath = nextValue(for: arg, iterator: &iterator)
      case "--esrgan-weights":
        esrganWeightsPath = nextValue(for: arg, iterator: &iterator)
      case "--tile-size":
        tileSize = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: tileSize)
      case "--softness":
        softness = floatValue(for: arg, iterator: &iterator, fallback: softness)
      case "--help", "-h":
        print("""
        SeedVR2 / ESRGAN Image Upscaler

        Usage: ComfyBox upscale --input <path> (--weights <path> | --esrgan-weights <path>) [options]

          --input, -i          Input image path (required)
          --output, -o         Output image path (default: input-upscaled.png)
          --resolution, -r     Target resolution for shortest side (default: 2048)
          --steps              Inference steps (default: 1)
          --seed               Random seed for reproducibility
          --weights, -w        Path to SeedVR2 model weights directory
          --esrgan-weights     Path to ESRGAN safetensors directory
          --tile-size          ESRGAN tile size for large images (default: 512)
          --softness           Preprocessing softness 0.0-1.0 (default: 0.0)
          --help, -h           Show this help

        Examples:
          ComfyBox upscale -i photo.jpg -w ./models/seedvr2
          ComfyBox upscale -i photo.jpg -w ./models/seedvr2 -r 4096 --seed 42
          ComfyBox upscale -i photo.jpg --esrgan-weights ./models/4x-ultrasharp --tile-size 512
          ComfyBox upscale -i low-res.png -w ./models/seedvr2 --softness 0.3 -o high-res.png
        """)
        return
      default:
        logger.warning("Unknown upscale argument: \(arg)")
      }
    }

    guard let input = inputPath else {
      fputs("Error: --input is required for upscale\n", stderr)
      exit(1)
    }

    guard weightsPath != nil || esrganWeightsPath != nil else {
      fputs("Error: --weights or --esrgan-weights is required for upscale\n", stderr)
      exit(1)
    }

    guard FileManager.default.fileExists(atPath: input) else {
      fputs("Error: Input file not found: \(input)\n", stderr)
      exit(1)
    }

    let startTime = CFAbsoluteTimeGetCurrent()

    if let esrganWeights = esrganWeightsPath {
      let pipeline = try ESRGANPipeline(
        weightsDirectory: URL(fileURLWithPath: esrganWeights),
        config: nil,
        logger: logger
      )

      let savedPath = try pipeline.upscaleAndSave(
        imagePath: input,
        outputPath: outputPath,
        tileSize: tileSize
      )

      let elapsed = CFAbsoluteTimeGetCurrent() - startTime
      logger.info("ESRGAN upscale complete in \(String(format: "%.1f", elapsed))s -> \(savedPath)")
      return
    }

    guard let weights = weightsPath else {
      fputs("Error: --weights is required for SeedVR2 upscale\n", stderr)
      exit(1)
    }

    let pipeline = try SeedVR2Pipeline(
      weightsPath: weights,
      steps: steps,
      logger: logger
    )

    let savedPath = try pipeline.upscaleAndSave(
      imagePath: input,
      outputPath: outputPath,
      targetResolution: resolution,
      seed: seed,
      softness: softness
    )

    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
    logger.info("Upscale complete in \(String(format: "%.1f", elapsed))s -> \(savedPath)")
  }
  #else
  private static func runUpscale(args: [String]) throws {
    fputs("Error: upscale requires CoreGraphics (macOS)\n", stderr)
    exit(1)
  }
  #endif

  // MARK: - Models Subcommand

  /// Known model families with their HuggingFace IDs and local directory names.
  private struct ModelFamily {
    let family: String
    let variant: String
    let directoryName: String
    let huggingFaceId: String?

    static let all: [ModelFamily] = [
      ModelFamily(family: "flux", variant: "schnell", directoryName: "flux-schnell-4bit", huggingFaceId: "black-forest-labs/FLUX.1-schnell"),
      ModelFamily(family: "flux", variant: "dev", directoryName: "flux-dev-4bit", huggingFaceId: "black-forest-labs/FLUX.1-dev"),
      ModelFamily(family: "flux", variant: "coffeeshop-8bit", directoryName: "coffeeshop-8bit", huggingFaceId: "carsenk/z-image-turbo-mflux-8bit"),
      ModelFamily(family: "chroma", variant: "default", directoryName: "chroma", huggingFaceId: "jack813liu/mlx-chroma"),
      ModelFamily(family: "fibo", variant: "8b", directoryName: "fibo-8b-4bit", huggingFaceId: "briaai/FIBO"),
      ModelFamily(family: "fibo", variant: "vlm", directoryName: "fibo-vlm", huggingFaceId: "briaai/FIBO-vlm"),
      ModelFamily(family: "seedvr2", variant: "3b", directoryName: "seedvr2-3b", huggingFaceId: "numz/SeedVR2_comfyUI"),
      ModelFamily(family: "seedvr2", variant: "7b", directoryName: "seedvr2-7b", huggingFaceId: nil),
      ModelFamily(family: "redux", variant: "encoder", directoryName: "redux-encoder", huggingFaceId: "DiffSynth-Studio/General-Image-Encoders"),
      ModelFamily(family: "kontext", variant: "default", directoryName: "kontext", huggingFaceId: nil),
      ModelFamily(family: "flux2", variant: "klein-4b", directoryName: "flux2-klein-4b", huggingFaceId: "black-forest-labs/FLUX.2-klein-4B"),
      ModelFamily(family: "flux2", variant: "klein-9b", directoryName: "flux2-klein-9b", huggingFaceId: "black-forest-labs/FLUX.2-klein-9B"),
      ModelFamily(family: "flux2", variant: "klein-base-4b", directoryName: "flux2-klein-base-4b", huggingFaceId: "black-forest-labs/FLUX.2-klein-base-4B"),
      ModelFamily(family: "flux2", variant: "klein-base-9b", directoryName: "flux2-klein-base-9b", huggingFaceId: "black-forest-labs/FLUX.2-klein-base-9B"),
      ModelFamily(family: "z-image", variant: "base", directoryName: "z-image-base", huggingFaceId: "Tongyi-MAI/Z-Image"),
      ModelFamily(family: "qwen", variant: "default", directoryName: "qwen", huggingFaceId: nil),
    ]
  }

  private enum ModelStatus: String {
    case installed = "installed"
    case notFound = "not found"
  }

  private static func runModels(args: [String]) {
    var showPaths = false

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--paths", "-v":
        showPaths = true
      case "--help", "-h":
        printModelsUsage()
        return
      default:
        logger.warning("Unknown models argument: \(arg)")
      }
    }

    let fm = FileManager.default
    let searchPaths = modelSearchPaths()

    // Header
    print("")
    let header = padRight("FAMILY", 14) + " " + padRight("VARIANT", 16) + " " + padRight("STATUS", 10) + " " + padLeft("SIZE", 10) + "   " + "QUANT"
    print(header)
    print(String(repeating: "-", count: header.count + 10))

    var installedCount = 0
    var totalSize: UInt64 = 0

    fflush(stderr)
    for model in ModelFamily.all {
      fflush(stderr)
      let (status, path, size) = resolveModelStatus(model: model, searchPaths: searchPaths, fm: fm)
      fflush(stderr)

      let sizeStr: String
      if let size {
        sizeStr = formatBytes(size)
        totalSize += size
      } else {
        sizeStr = "-"
      }

      let statusStr: String
      switch status {
      case .installed:
        statusStr = "installed"
        installedCount += 1
      case .notFound:
        statusStr = "not found"
      }

      let quant = detectQuantization(at: path, fm: fm)

      let row = padRight(model.family, 14) + " " + padRight(model.variant, 16) + " " + padRight(statusStr, 10) + " " + padLeft(sizeStr, 10) + "   " + quant
      print(row)

      if showPaths, let path {
        print("               \(path)")
      }
    }

    print(String(repeating: "-", count: header.count + 10))
    print("\(installedCount)/\(ModelFamily.all.count) installed, \(formatBytes(totalSize)) total")
    print("")
    fflush(stdout)
    _exit(0)
  }

  /// Build ordered list of directories to search for models.
  private static func modelSearchPaths() -> [URL] {
    var paths: [URL] = []
    let fm = FileManager.default
    let home = fm.homeDirectoryForCurrentUser

    // 1. HuggingFace cache
    let env = ProcessInfo.processInfo.environment
    if let hubCache = env["HF_HUB_CACHE"], !hubCache.isEmpty {
      paths.append(URL(fileURLWithPath: hubCache))
    } else if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
      paths.append(URL(fileURLWithPath: hfHome).appendingPathComponent("hub"))
    } else {
      paths.append(home.appendingPathComponent(".cache/huggingface/hub"))
    }

    // 2. ~/models
    paths.append(home.appendingPathComponent("models"))

    // 3. ./models
    let cwdModels = URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("models")
    if !paths.contains(where: { $0.path == cwdModels.path }) {
      paths.append(cwdModels)
    }

    return paths
  }

  /// Try to find a model in the search paths. Returns (status, path, sizeInBytes).
  private static func resolveModelStatus(
    model: ModelFamily,
    searchPaths: [URL],
    fm: FileManager
  ) -> (ModelStatus, String?, UInt64?) {
    for base in searchPaths {
      // Direct directory name match (e.g. ~/models/seedvr2-3b)
      let direct = base.appendingPathComponent(model.directoryName)
      if fm.fileExists(atPath: direct.path) {
        let size = directorySize(at: direct, fm: fm)
        return (.installed, direct.path, size)
      }

      // HuggingFace cache layout: models--ORG--REPO/snapshots/<commit>/
      if let hfId = model.huggingFaceId {
        let repoCacheRoot = base.appendingPathComponent("models--\(hfId.replacingOccurrences(of: "/", with: "--"))")
        let snapshotsRoot = repoCacheRoot.appendingPathComponent("snapshots")
        if fm.fileExists(atPath: snapshotsRoot.path),
           let snapshots = try? fm.contentsOfDirectory(at: snapshotsRoot, includingPropertiesForKeys: nil),
           let first = snapshots.first {
          let size = directorySize(at: first, fm: fm)
          return (.installed, first.path, size)
        }
      }
    }

    return (.notFound, nil, nil)
  }

  /// Recursively compute the total size of a directory in bytes.
  /// Follows symlinks to get real file sizes (needed for HuggingFace cache layout).
  private static func directorySize(at url: URL, fm: FileManager) -> UInt64 {
    guard let enumerator = fm.enumerator(
      at: url,
      includingPropertiesForKeys: [.totalFileAllocatedSizeKey, .isRegularFileKey],
      options: [.skipsHiddenFiles]
    ) else {
      return 0
    }

    var total: UInt64 = 0
    for case let fileURL as URL in enumerator {
      // Resolve symlinks to get the real file
      let resolved = fileURL.resolvingSymlinksInPath()
      guard let values = try? resolved.resourceValues(forKeys: [.totalFileAllocatedSizeKey, .isRegularFileKey, .fileSizeKey]),
            values.isRegularFile == true else {
        continue
      }
      // Prefer allocated size, fall back to logical size
      let size = values.totalFileAllocatedSize ?? values.fileSize ?? 0
      total += UInt64(size)
    }
    return total
  }

  /// Detect quantization level from directory contents.
  private static func detectQuantization(at path: String?, fm: FileManager) -> String {
    guard let path else { return "-" }
    let url = URL(fileURLWithPath: path)

    // Check for quantization.json (written by our quantize subcommand)
    let quantFile = url.appendingPathComponent("quantization.json")
    if fm.fileExists(atPath: quantFile.path),
       let data = try? Data(contentsOf: quantFile),
       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
       let bits = json["bits"] as? Int,
       let groupSize = json["group_size"] as? Int {
      return "q\(bits)_g\(groupSize)"
    }

    // Check subdirectories for quantization.json
    for sub in ["transformer", "text_encoder", "vae"] {
      let subQuantFile = url.appendingPathComponent(sub).appendingPathComponent("quantization.json")
      if fm.fileExists(atPath: subQuantFile.path),
         let data = try? Data(contentsOf: subQuantFile),
         let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
         let bits = json["bits"] as? Int,
         let groupSize = json["group_size"] as? Int {
        return "q\(bits)_g\(groupSize)"
      }
    }

    // Infer from directory name (check multiple levels for HF cache paths)
    let pathStr = url.path.lowercased()
    if pathStr.contains("q4") || pathStr.contains("4bit") || pathStr.contains("4-bit") { return "q4" }
    if pathStr.contains("q8") || pathStr.contains("8bit") || pathStr.contains("8-bit") { return "q8" }
    if pathStr.contains("bf16") { return "bf16" }

    // Check if safetensors files look quantized by checking for quantization config
    let transformerDir = url.appendingPathComponent("transformer")
    if let contents = try? fm.contentsOfDirectory(at: transformerDir, includingPropertiesForKeys: nil) {
      let safetensors = contents.filter { $0.pathExtension == "safetensors" }
      if !safetensors.isEmpty {
        // If model has fp16 in parent path or is a standard HF snapshot, assume fp16
        return "fp16"
      }
    }

    return "-"
  }

  /// Format bytes as human-readable string (e.g. "4.2 GB", "128 MB").
  private static func formatBytes(_ bytes: UInt64) -> String {
    let gb = Double(bytes) / (1024.0 * 1024.0 * 1024.0)
    if gb >= 1.0 {
      return String(format: "%.1f GB", gb)
    }
    let mb = Double(bytes) / (1024.0 * 1024.0)
    if mb >= 1.0 {
      return String(format: "%.0f MB", mb)
    }
    let kb = Double(bytes) / 1024.0
    return String(format: "%.0f KB", kb)
  }

  private static func padRight(_ s: String, _ width: Int) -> String {
    if s.count >= width { return s }
    return s + String(repeating: " ", count: width - s.count)
  }

  private static func padLeft(_ s: String, _ width: Int) -> String {
    if s.count >= width { return s }
    return String(repeating: " ", count: width - s.count) + s
  }

  private static func printModelsUsage() {
    print("""
    List known model families with installation status.

    Usage: ComfyBox models [options]
      --paths, -v   Show filesystem paths for installed models
      --help, -h    Show help

    Search paths (in order):
      1. $HF_HUB_CACHE or $HF_HOME/hub or ~/.cache/huggingface/hub
      2. ~/models
      3. ./models

    Example:
      ComfyBox models
      ComfyBox models --paths
    """)
  }



  // MARK: - MCP Server Subcommand

  private static func runMCP(args: [String]) throws {
    var port: UInt16 = 7870
    var host = "127.0.0.1"

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--port":
        let rawPort = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: Int(port))
        guard rawPort <= Int(UInt16.max) else {
          logger.error("Invalid port: \(rawPort)")
          return
        }
        port = UInt16(rawPort)
      case "--host":
        host = nextValue(for: arg, iterator: &iterator)
      case "--help", "-h":
        printMCPUsage()
        return
      default:
        logger.warning("Unknown mcp argument: \(arg)")
      }
    }

    // comfybox#153: this bridge NEVER starts a server — launchd
    // (com.barkadabrew.comfybox) owns the engine lifecycle. Connect to
    // whatever answers on --port (healthy or not); fail loudly instead of
    // spawning anything when nothing does.
    let portOccupied = MCPPortProbe.isOccupied(host: host, port: port)
    let decision = MCPBridgeStartupPolicy.decide(port: port, portOccupied: portOccupied)

    switch decision.action {
    case .connect:
      fputs("[comfybox-mcp] \(decision.detail)\n", stderr)
      fputs("[comfybox-mcp] \(host):\(port) — \(fetchHealthSummary(host: host, port: port))\n", stderr)

    case .failLoudly:
      fputs("[comfybox-mcp] \(decision.detail)\n", stderr)
      _exit(1)
    }

    // Print server info to stderr (stdout is reserved for JSON-RPC)
    fputs("ComfyBox MCP server v\(MCPServer.version)\n", stderr)
    fputs("Bridging to WarmServer at \(host):\(port)\n", stderr)

    let server = MCPServer(host: host, port: port)
    server.run()
  }

  /// Best-effort human-readable summary of whatever answered `/health` on
  /// `host:port`, for the connect-path stderr report (the bridge never
  /// spawns, but it still says what it found — comfybox#153).
  private static func fetchHealthSummary(host: String, port: UInt16) -> String {
    let client = WarmServerClient(host: host, port: port)
    let semaphore = DispatchSemaphore(value: 0)
    let box = Box("no response before timeout")
    let task = Task {
      defer { semaphore.signal() }
      do {
        let (status, data) = try await client.get("/health")
        if status == 200 {
          var summary = "responded 200 OK"
          if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
             let model = json["model"] as? String {
            summary += " (model: \(model))"
          }
          box.value = summary
        } else {
          box.value = "responded HTTP \(status) (unhealthy or unrecognized)"
        }
      } catch {
        box.value = "connected but /health failed: \(error.localizedDescription)"
      }
    }
    _ = semaphore.wait(timeout: .now() + 2.0)
    task.cancel()
    return box.value
  }

  private static func printMCPUsage() {
    print("""
    Start MCP (Model Context Protocol) server mode.
    Bridges stdio JSON-RPC 2.0 to WarmServer HTTP API.

    Usage: ComfyBox mcp [options]
      --port                    WarmServer port to connect to (default: 7870)
      --host                    WarmServer host to connect to (default: 127.0.0.1)
      --help, -h                Show help

    The MCP server reads JSON-RPC requests from stdin and writes responses
    to stdout. All logging goes to stderr. Runs until stdin closes.

    This bridge NEVER starts a server (comfybox#153): launchd
    (com.barkadabrew.comfybox) owns the engine lifecycle. If a server is
    already listening on --port, healthy or not, the bridge connects to it.
    If nothing is listening, the bridge fails loudly to stderr and exits —
    it does not spawn one. Start the managed engine with:
      launchctl kickstart -k gui/$(id -u)/com.barkadabrew.comfybox

    Registration:
      claude mcp add comfybox -- comfybox mcp --port 7870

    Tools:
      generate_image    Text-to-image / img2img generation
      swap_loras        Hot-swap active LoRA weights
      list_models       List supported model families
      list_styles       List style presets
      server_health     Server health and loaded model info
      queue_status      Generation queue status
      clear_queue       Cancel pending generation jobs
      list_loras        List available LoRA files
      shutdown_server   Graceful server shutdown
      system_stats      Hardware and system info
      apply_style       Apply style preset to prompt
    """)
  }


  // MARK: - LoRA Library Subcommand

  private static func runLoRA(args: [String]) throws {
    guard let subcommand = args.first else {
      printLoRAUsage()
      return
    }

    let subArgs = Array(args.dropFirst())

    switch subcommand {
    case "list":
      try runLoRAList(args: subArgs)
    case "info":
      try runLoRAInfo(args: subArgs)
    case "scan":
      try runLoRAScan(args: subArgs)
    case "check":
      try runLoRACheck(args: subArgs)
    case "quarantine":
      try runLoRAQuarantine(args: subArgs)
    case "search":
      try runLoRASearch(args: subArgs)
    case "--help", "-h":
      printLoRAUsage()
    default:
      fputs("Unknown lora subcommand: \(subcommand)\n", stderr)
      printLoRAUsage()
      _exit(1)
    }

    fflush(stdout)
    _exit(0)
  }

  // MARK: - LoRA List

  private static func runLoRAList(args: [String]) throws {
    var modelFilter: String?
    var tagFilter: String?
    var showQuarantined = false

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--model", "-m":
        modelFilter = nextValue(for: arg, iterator: &iterator)
      case "--tag", "-t":
        tagFilter = nextValue(for: arg, iterator: &iterator)
      case "--quarantined", "-q":
        showQuarantined = true
      case "--help", "-h":
        print("""
        List LoRAs in the library.

        Usage: ZImageCLI lora list [options]
          --model, -m <family>   Filter by model compatibility (e.g. z-image, klein-9b)
          --tag, -t <tag>        Filter by tag
          --quarantined, -q      Include quarantined entries
          --help, -h             Show help
        """)
        return
      default:
        fputs("Unknown argument: \(arg)\n", stderr)
      }
    }

    let library = try LoRALibrary()
    let entries: [LoRALibraryEntry]

    if let modelFilter {
      if let family = LoRACompatibility.familyMapping(modelFilter) {
        entries = library.compatible(with: family)
      } else {
        entries = library.list(compatibility: modelFilter, includeQuarantined: showQuarantined)
      }
    } else if let tagFilter {
      entries = library.list(tags: [tagFilter], includeQuarantined: showQuarantined)
    } else {
      entries = library.list(includeQuarantined: showQuarantined)
    }

    if entries.isEmpty {
      print("No LoRAs found. Run 'zimage lora scan' to build the library index.")
      return
    }

    // Table header
    print("")
    let header = padRight("ID", 32) + " " +
                 padRight("Model", 12) + " " +
                 padRight("Format", 8) + " " +
                 padLeft("Size", 10) + " " +
                 padLeft("Scale", 6) + "  " +
                 "Tags"
    print(header)
    print(String(repeating: "-", count: 90))

    var activeCount = 0
    var quarantinedCount = 0

    for entry in entries {
      let qMark = entry.quarantined ? " [Q]" : ""
      if entry.quarantined { quarantinedCount += 1 } else { activeCount += 1 }

      let tagsStr = entry.tags.joined(separator: ", ") + qMark
      let row = padRight(entry.id, 32) + " " +
                padRight(entry.primaryCompatibility, 12) + " " +
                padRight(entry.format.rawValue, 8) + " " +
                padLeft(entry.sizeFormatted, 10) + " " +
                padLeft(String(format: "%.1f", entry.recommendedScale), 6) + "  " +
                tagsStr
      print(row)
    }

    print(String(repeating: "-", count: 90))
    if showQuarantined {
      print("\(entries.count) LoRAs (\(activeCount) active, \(quarantinedCount) quarantined)")
    } else {
      print("\(entries.count) active LoRAs")
    }
    print("")
  }

  // MARK: - LoRA Info

  private static func runLoRAInfo(args: [String]) throws {
    guard let identifier = args.first else {
      fputs("Usage: ZImageCLI lora info <id>\n", stderr)
      _exit(1)
      return
    }

    let library = try LoRALibrary()
    guard let entry = library.entry(for: identifier) else {
      fputs("LoRA not found: \(identifier)\n", stderr)
      fputs("Run 'zimage lora list --quarantined' to see all entries.\n", stderr)
      _exit(1)
      return
    }

    print("")
    print("ID:                  \(entry.id)")
    print("Filename:            \(entry.filename)")
    print("Path:                \(entry.relativePath)")
    print("Size:                \(entry.sizeFormatted) (\(entry.sizeBytes) bytes)")
    if let hash = entry.sha256 {
      print("SHA-256:             \(hash)")
    }
    print("Model Compatibility: \(entry.modelCompatibility.joined(separator: ", "))")
    print("Format:              \(entry.format.rawValue)")
    print("Rank:                \(entry.rank)")
    if let alpha = entry.alpha {
      print("Alpha:               \(alpha)")
    }
    print("Key Count:           \(entry.keyCount)")
    print("Layer Targets:       \(entry.layerTargets.joined(separator: ", "))")
    print("Trigger Words:       \(entry.triggerwords.isEmpty ? "(none)" : entry.triggerwords.joined(separator: ", "))")
    print("Recommended Scale:   \(entry.recommendedScale)")
    print("Scale Range:         [\(entry.scaleRange.map { String(format: "%.1f", $0) }.joined(separator: ", "))]")
    print("Tags:                \(entry.tags.isEmpty ? "(none)" : entry.tags.joined(separator: ", "))")
    print("Category:            \(entry.category)")
    if !entry.notes.isEmpty {
      print("Notes:               \(entry.notes)")
    }
    if let url = entry.sourceURL {
      print("Source URL:           \(url)")
    }
    if let civitaiId = entry.civitaiModelId {
      print("CivitAI Model ID:    \(civitaiId)")
    }
    print("Date Added:          \(entry.dateAdded)")
    print("Quarantined:         \(entry.quarantined)")
    if let reason = entry.quarantineReason {
      print("Quarantine Reason:   \(reason)")
    }

    if let metadata = entry.safetensorsMetadata, !metadata.isEmpty {
      print("")
      print("Safetensors Metadata:")
      for (key, value) in metadata.sorted(by: { $0.key < $1.key }) {
        let displayValue = value.count > 80 ? String(value.prefix(77)) + "..." : value
        print("  \(key): \(displayValue)")
      }
    }
    print("")
  }

  // MARK: - LoRA Scan

  private static func runLoRAScan(args: [String]) throws {
    var force = false

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--force", "-f":
        force = true
      case "--help", "-h":
        print("""
        Scan filesystem and rebuild/update the library index.

        Usage: ZImageCLI lora scan [options]
          --force, -f   Re-analyze all files even if unchanged
          --help, -h    Show help
        """)
        return
      default:
        fputs("Unknown argument: \(arg)\n", stderr)
      }
    }

    let library = try LoRALibrary()
    let result = try library.scan(force: force)

    print("")
    print("Scan Results:")
    print("  Added:     \(result.added)")
    print("  Updated:   \(result.updated)")
    print("  Removed:   \(result.removed)")
    print("  Unchanged: \(result.unchanged)")
    print("  Total:     \(result.total)")

    if !result.errors.isEmpty {
      print("")
      print("Errors:")
      for (file, error) in result.errors {
        print("  \(file): \(error)")
      }
    }
    print("")
  }

  // MARK: - LoRA Check

  private static func runLoRACheck(args: [String]) throws {
    var identifier: String?
    var modelFamilyStr: String?

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--model", "-m":
        modelFamilyStr = nextValue(for: arg, iterator: &iterator)
      case "--help", "-h":
        print("""
        Check LoRA compatibility with a model family.

        Usage: ZImageCLI lora check <id> --model <family>
          --model, -m <family>   Target model family (z-image, klein-9b, chroma)
          --help, -h             Show help
        """)
        return
      default:
        if identifier == nil && !arg.hasPrefix("-") {
          identifier = arg
        } else {
          fputs("Unknown argument: \(arg)\n", stderr)
        }
      }
    }

    guard let identifier else {
      fputs("Usage: ZImageCLI lora check <id> --model <family>\n", stderr)
      _exit(1)
      return
    }

    guard let familyStr = modelFamilyStr,
          let family = LoRACompatibility.familyMapping(familyStr) else {
      fputs("Error: --model is required. Valid: z-image, klein-9b, klein-4b, chroma\n", stderr)
      _exit(1)
      return
    }

    let library = try LoRALibrary()
    guard let entry = library.entry(for: identifier) else {
      fputs("LoRA not found: \(identifier)\n", stderr)
      _exit(1)
      return
    }

    // Perform live file check
    let fileURL = try library.resolve(identifier)
    let result = try LoRACompatibility.checkFile(fileURL, modelFamily: family)

    print("")
    print("Compatibility Check: \(entry.id) vs \(family.displayName)")
    print(String(repeating: "-", count: 50))
    print("Compatible:    \(result.isCompatible ? "YES" : "NO")")
    print("Matched Keys:  \(result.matchedKeys)/\(result.totalKeys)")
    print("Match Ratio:   \(String(format: "%.1f%%", result.matchRatio * 100))")

    if !result.warnings.isEmpty {
      print("")
      print("Warnings:")
      for warning in result.warnings {
        print("  - \(warning)")
      }
    }
    print("")
  }

  // MARK: - LoRA Quarantine

  private static func runLoRAQuarantine(args: [String]) throws {
    var identifier: String?
    var reason: String?

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--reason", "-r":
        reason = nextValue(for: arg, iterator: &iterator)
      case "--help", "-h":
        print("""
        Quarantine a LoRA (mark as incompatible).

        Usage: ZImageCLI lora quarantine <id> --reason <text>
          --reason, -r <text>   Reason for quarantine
          --help, -h            Show help
        """)
        return
      default:
        if identifier == nil && !arg.hasPrefix("-") {
          identifier = arg
        } else {
          fputs("Unknown argument: \(arg)\n", stderr)
        }
      }
    }

    guard let identifier else {
      fputs("Usage: ZImageCLI lora quarantine <id> --reason <text>\n", stderr)
      _exit(1)
      return
    }

    let reasonText = reason ?? "Manually quarantined"
    let library = try LoRALibrary()
    try library.quarantine(identifier, reason: reasonText)
    print("Quarantined: \(identifier) - \(reasonText)")
  }

  // MARK: - LoRA Search

  private static func runLoRASearch(args: [String]) throws {
    let query = args.joined(separator: " ")
    guard !query.isEmpty else {
      fputs("Usage: ZImageCLI lora search <query>\n", stderr)
      _exit(1)
      return
    }

    let library = try LoRALibrary()
    let results = library.search(query)

    if results.isEmpty {
      print("No results for: \(query)")
      return
    }

    print("")
    print("\(results.count) result(s) for \"\(query)\":")
    print("")
    let header = padRight("ID", 32) + " " +
                 padRight("Model", 12) + " " +
                 padRight("Format", 8) + " " +
                 padLeft("Size", 10)
    print(header)
    print(String(repeating: "-", count: 65))

    for entry in results {
      let qMark = entry.quarantined ? " [Q]" : ""
      let row = padRight(entry.id + qMark, 32) + " " +
                padRight(entry.primaryCompatibility, 12) + " " +
                padRight(entry.format.rawValue, 8) + " " +
                padLeft(entry.sizeFormatted, 10)
      print(row)
    }
    print("")
  }

  // MARK: - LoRA Usage

  private static func printLoRAUsage() {
    print("""

    Manage the LoRA library.

    Usage: ZImageCLI lora <subcommand> [options]

    SUBCOMMANDS:
      list          List available LoRAs
      info          Show detailed info for a LoRA
      scan          Scan filesystem and rebuild library index
      check         Check LoRA compatibility with a model
      quarantine    Quarantine an incompatible LoRA
      search        Search LoRAs by text

    EXAMPLES:
      zimage lora list                                  List all active LoRAs
      zimage lora list --model z-image                  Filter by model compatibility
      zimage lora list --tag nsfw --quarantined          Filter by tag, include quarantined
      zimage lora info zit-fdpo-v1                       Detailed metadata
      zimage lora scan                                   Build/rebuild library index
      zimage lora scan --force                           Re-analyze all files
      zimage lora check KLEIN-Unchained-V2 --model z-image   Compatibility test
      zimage lora quarantine bad-lora --reason "Wrong arch"   Mark incompatible
      zimage lora search distill                         Text search

    """)
  }



  // MARK: - LTX2 Text Encoder Test

  /// Validate the Gemma 3 text encoder pipeline against Python ground truth.
  ///
  /// Loads the Gemma 3 12B model (BF16), tokenizes the reference prompt,
  /// runs the full text encoding pipeline, and compares against the
  /// pre-computed ground truth embeddings.
  private static func runLTX2TextEncoderTest(args: [String]) throws {
    var gemmaPath = "/Users/toddwalderman/.cache/huggingface/hub/models--unsloth--gemma-3-12b-it/snapshots/9478e665381f42974aa06177b019352fb6291876"
    var connectorPath = "/Users/toddwalderman/Models/ltx2-distilled"
    var groundTruthPath = "/tmp/kira-text-embeddings.safetensors"

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--gemma-path":
        gemmaPath = nextValue(for: arg, iterator: &iterator)
      case "--connector-path":
        connectorPath = nextValue(for: arg, iterator: &iterator)
      case "--ground-truth":
        groundTruthPath = nextValue(for: arg, iterator: &iterator)
      case "--help", "-h":
        print("""
        LTX-2.3 Text Encoder Validation Test

        Loads Gemma 3 12B (live) + connector weights, encodes the reference prompt,
        and compares output against Python ground truth embeddings.

        Usage: ComfyBox ltx2-text-encoder-test [options]
          --gemma-path <path>      Path to Gemma 3 weights directory
          --connector-path <path>  Path to LTX-2 model directory (contains connector.safetensors)
          --ground-truth <path>    Path to ground truth embeddings (.safetensors)
          --help, -h               Show help
        """)
        return
      default:
        logger.warning("Unknown argument: \(arg)")
      }
    }

    let prompt = "A young petite Filipina woman with dark brown skin and long black hair, wearing a simple white sundress, walking along a tropical beach at golden hour. Ocean waves gently lapping at her feet. Cinematic, warm lighting."

    print(String(repeating: "=", count: 60))
    print("LTX-2.3 Text Encoder Validation Test")
    print(String(repeating: "=", count: 60))
    print()

    // Step 1: Load tokenizer
    print("[1/5] Loading Gemma 3 tokenizer...")
    let tokenizerDir = URL(fileURLWithPath: gemmaPath)
    let tokenizer = try LTX2GemmaTokenizer.load(from: tokenizerDir, maxLength: 128)
    let batch = tokenizer.encode(prompt: prompt, maxLength: 128)
    print("  Token IDs shape: \(batch.inputIds.shape)")
    print("  Attention mask shape: \(batch.attentionMask.shape)")

    // Print first/last few tokens for verification
    MLX.eval(batch.inputIds, batch.attentionMask)
    let maskSum = MLX.sum(batch.attentionMask).item(Int.self)
    print("  Valid tokens: \(maskSum) / 128")

    // Verify against known Python output
    let expectedValidTokens = 46
    if maskSum == expectedValidTokens {
      print("  PASS: Token count matches Python reference (\(expectedValidTokens))")
    } else {
      print("  WARN: Token count \(maskSum) != expected \(expectedValidTokens)")
    }

    // Step 2: Create text encoder
    print()
    print("[2/5] Creating text encoder (Gemma 3 12B + connectors)...")
    let gemmaConfig = LTX2GemmaConfig(
      vocabSize: 262208,
      hiddenSize: 3840,
      numHiddenLayers: 48,
      numAttentionHeads: 16,
      numKeyValueHeads: 8,
      headDim: 256,
      intermediateSize: 15360,
      rmsNormEps: 1e-6,
      ropeTheta: 1_000_000.0,
      slidingWindow: 1024,
      slidingWindowPattern: 6,
      quantization: nil
    )
    let encoderConfig = LTX2TextEncoderConfig(
      gemma: gemmaConfig,
      hasPromptAdaLN: true
    )
    let textEncoder = LTX2TextEncoder(config: encoderConfig)

    // Step 3: Load weights
    print()
    print("[3/5] Loading weights...")
    print("  Gemma path: \(gemmaPath)")
    print("  Connector path: \(connectorPath)")
    let startLoad = CFAbsoluteTimeGetCurrent()
    try textEncoder.loadWeights(
      modelPath: URL(fileURLWithPath: connectorPath),
      textEncoderPath: URL(fileURLWithPath: gemmaPath)
    )
    MLX.eval(textEncoder.parameters())
    let loadTime = CFAbsoluteTimeGetCurrent() - startLoad
    print("  Weights loaded in \(String(format: "%.1f", loadTime))s")

    // Step 4: Encode
    print()
    print("[4/5] Running text encoder pipeline...")
    let startEncode = CFAbsoluteTimeGetCurrent()
    let output = textEncoder.encode(
      inputIds: batch.inputIds,
      attentionMask: batch.attentionMask,
      returnAudioEmbeddings: false
    )
    MLX.eval(output.videoEmbeddings)
    let encodeTime = CFAbsoluteTimeGetCurrent() - startEncode
    print("  Video embeddings shape: \(output.videoEmbeddings.shape)")
    print("  Encoding time: \(String(format: "%.1f", encodeTime))s")

    // Step 5: Compare against ground truth
    print()
    print("[5/5] Comparing against ground truth...")
    let gtURL = URL(fileURLWithPath: groundTruthPath)
    guard FileManager.default.fileExists(atPath: groundTruthPath) else {
      print("  SKIP: Ground truth file not found at \(groundTruthPath)")
      print("  Run the Python reference script first to generate it.")
      return
    }

    let gtTensors = try MLX.loadArrays(url: gtURL)
    guard let gtEmbeddings = gtTensors["video_embeddings"] else {
      print("  ERROR: No video_embeddings key in ground truth file")
      return
    }

    print("  Ground truth shape: \(gtEmbeddings.shape)")

    // Convert both to float32 for comparison
    let swiftEmb = output.videoEmbeddings.asType(.float32)
    let pyEmb = gtEmbeddings.asType(.float32)
    MLX.eval(swiftEmb, pyEmb)

    // Statistics
    let swiftMin = MLX.min(swiftEmb).item(Float.self)
    let swiftMax = MLX.max(swiftEmb).item(Float.self)
    let swiftMean = MLX.mean(swiftEmb).item(Float.self)

    let pyMin = MLX.min(pyEmb).item(Float.self)
    let pyMax = MLX.max(pyEmb).item(Float.self)
    let pyMean = MLX.mean(pyEmb).item(Float.self)

    print()
    print("  Statistics:")
    print("                  Swift          Python")
    print(String(format: "  min:     %12.4f    %12.4f", swiftMin, pyMin))
    print(String(format: "  max:     %12.4f    %12.4f", swiftMax, pyMax))
    print(String(format: "  mean:    %12.6f    %12.6f", swiftMean, pyMean))

    // Compute differences
    let diff = swiftEmb - pyEmb
    MLX.eval(diff)
    let maxAbsDiff = MLX.max(MLX.abs(diff)).item(Float.self)
    let meanAbsDiff = MLX.mean(MLX.abs(diff)).item(Float.self)
    let mse = MLX.mean(diff * diff).item(Float.self)

    // Cosine similarity (flatten then compute)
    let swiftFlat = swiftEmb.reshaped(-1)
    let pyFlat = pyEmb.reshaped(-1)
    let dotProduct = MLX.sum(swiftFlat * pyFlat).item(Float.self)
    let swiftNorm = MLX.sqrt(MLX.sum(swiftFlat * swiftFlat)).item(Float.self)
    let pyNorm = MLX.sqrt(MLX.sum(pyFlat * pyFlat)).item(Float.self)
    let cosineSim = dotProduct / (swiftNorm * pyNorm + 1e-8)

    print()
    print("  Comparison:")
    print(String(format: "  Max absolute diff:  %.6f", maxAbsDiff))
    print(String(format: "  Mean absolute diff: %.6f", meanAbsDiff))
    print(String(format: "  MSE:                %.6f", mse))
    print(String(format: "  Cosine similarity:  %.6f", cosineSim))

    // Verdict
    print()
    if cosineSim > 0.95 {
      print("  PASS: Cosine similarity > 0.95 -- embeddings are functionally equivalent")
    } else if cosineSim > 0.85 {
      print("  PASS (BF16): Cosine similarity > 0.85 -- within expected tolerance")
      print("  Note: BF16 precision across 48 layers + connector may introduce minor divergence")
    } else if cosineSim > 0.70 {
      print("  WARN: Cosine similarity > 0.70 -- larger than expected quantization error")
      print("  Check weight loading, architecture, or consider bf16 weights")
    } else {
      print("  FAIL: Cosine similarity < 0.70 -- embeddings do not match")
      print("  Architecture or weight loading bug likely")
    }

    print()
    print(String(repeating: "=", count: 60))
  }

  // MARK: - Video Subcommand (LTX-2 Native)

  /// Resolve resolution label + aspect ratio to (width, height) in pixels.
  ///
  /// All dimensions are rounded to multiples of 32 (VAE spatial compression).
  private static func resolveVideoResolution(
    resolution: String, aspectRatio: String
  ) -> (width: Int, height: Int) {
    let landscape = aspectRatio != "9:16"
    switch resolution {
    case "480p":
      return landscape ? (width: 832, height: 480) : (width: 480, height: 832)
    case "1080p":
      return landscape ? (width: 1920, height: 1088) : (width: 1088, height: 1920)
    default: // 720p
      return landscape ? (width: 1280, height: 736) : (width: 736, height: 1280)
    }
  }

  /// Convert a T2V duration in seconds to frame count at 24 fps.
  ///
  /// LTX-2 requires numFrames = 1 + 8k. We round to the nearest valid count.
  private static func durationToFrames(_ seconds: Int, fps: Int = 24) -> Int {
    let rawFrames = seconds * fps
    // Round to nearest valid frame count (1 + 8k)
    let k = max(1, Int((Float(rawFrames - 1) / 8.0).rounded()))
    return 1 + 8 * k
  }

  private static func runVideo(args: [String]) throws {
    // --- Default values ---
    let defaultWeightsDir = (NSHomeDirectory() as NSString).appendingPathComponent("Models/ltx2-distilled")
    let defaultGemmaPath = (NSHomeDirectory() as NSString).appendingPathComponent(
      ".cache/huggingface/hub/models--unsloth--gemma-3-12b-it/snapshots/9478e665381f42974aa06177b019352fb6291876"
    )

    var prompt: String?
    var imagePath: String?
    var outputPath = "z-video.mp4"
    var durationSeconds = 6
    var resolution = "720p"
    var aspectRatio = "16:9"
    var seed: UInt64? = nil
    var weightsDir: String = defaultWeightsDir
    var gemmaPath: String = defaultGemmaPath
    var noProgress = false
    var loraPath: String? = nil
    var loraStrength: Float = 1.0
    var steps: Int? = nil
    var strength: Float = 1.0

    // --- Parse arguments ---
    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--prompt", "-p":
        prompt = nextValue(for: arg, iterator: &iterator)
      case "--image", "-i":
        imagePath = nextValue(for: arg, iterator: &iterator)
      case "--output", "-o":
        outputPath = nextValue(for: arg, iterator: &iterator)
      case "--duration", "-d":
        durationSeconds = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: durationSeconds)
      case "--resolution", "-r":
        resolution = nextValue(for: arg, iterator: &iterator)
      case "--aspect-ratio":
        aspectRatio = nextValue(for: arg, iterator: &iterator)
      case "--seed":
        if let s = uint64Value(for: arg, iterator: &iterator) {
          seed = s
        }
      case "--weights", "-w":
        weightsDir = nextValue(for: arg, iterator: &iterator)
      case "--gemma-path":
        gemmaPath = nextValue(for: arg, iterator: &iterator)
      case "--no-progress":
        noProgress = true
      case "--lora":
        loraPath = nextValue(for: arg, iterator: &iterator)
      case "--lora-strength":
        loraStrength = floatValue(for: arg, iterator: &iterator, fallback: 1.0)
      case "--steps":
        steps = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: 8)
      case "--strength":
        strength = floatValue(for: arg, iterator: &iterator, fallback: 1.0)
      case "--help", "-h":
        printVideoUsage()
        return
      default:
        logger.warning("Unknown video argument: \(arg)")
      }
    }

    // --- Validate ---
    guard let promptText = prompt else {
      fputs("Error: --prompt is required.\n", stderr)
      printVideoUsage()
      exit(1)
    }

    let isI2V = imagePath != nil

    if !VideoGenerateRequest.validResolutions.contains(resolution) {
      fputs("Error: invalid resolution '\(resolution)'. Valid: \(VideoGenerateRequest.validResolutions.joined(separator: ", "))\n", stderr)
      exit(1)
    }
    if !VideoGenerateRequest.validAspectRatios.contains(aspectRatio) {
      fputs("Error: invalid aspect ratio '\(aspectRatio)'. Valid: \(VideoGenerateRequest.validAspectRatios.joined(separator: ", "))\n", stderr)
      exit(1)
    }
    if !isI2V {
      if !VideoGenerateRequest.validT2VDurations.contains(durationSeconds) {
        fputs("Error: invalid duration \(durationSeconds). T2V supports: \(VideoGenerateRequest.validT2VDurations.map(String.init).joined(separator: ", "))\n", stderr)
        exit(1)
      }
    }

    // Resolve dimensions
    let (width, height) = resolveVideoResolution(resolution: resolution, aspectRatio: aspectRatio)

    // Resolve frame count
    let fps = 24
    let numFrames: Int
    if isI2V {
      // I2V: fixed 97 frames (~4s at 24fps), matching ltx2-i2v default
      numFrames = 97
    } else {
      numFrames = durationToFrames(durationSeconds, fps: fps)
    }

    // Default steps: 8 for distilled
    let denoiseSteps = steps ?? 8

    // Generate seed if not provided
    let actualSeed = seed ?? UInt64.random(in: 0...UInt64.max)

    // Ensure output ends with .mp4
    if !outputPath.hasSuffix(".mp4") {
      outputPath += ".mp4"
    }

    // --- Banner ---
    let mode = isI2V ? "Image-to-Video" : "Text-to-Video"
    let durationDesc = isI2V ? "\(numFrames) frames (\(String(format: "%.1f", Float(numFrames) / Float(fps)))s)"
                              : "\(durationSeconds)s (\(numFrames) frames)"
    print()
    print("=== ComfyBox Video (\(mode)) ===")
    print("Weights:    \(weightsDir)")
    print("Output:     \(outputPath)")
    print("Prompt:     \(promptText)")
    if let img = imagePath {
      print("Image:      \(img)")
      print("Strength:   \(strength)")
    }
    print("Resolution: \(width)x\(height) (\(resolution), \(aspectRatio))")
    print("Duration:   \(durationDesc)")
    print("Steps:      \(denoiseSteps)")
    print("Seed:       \(actualSeed)")
    if let lp = loraPath {
      print("LoRA:       \(lp)")
      print("LoRA str:   \(loraStrength)")
    }
    print()

    // --- Verify weight files exist ---
    // Per-component layout needs transformer + both VAE files; a JoyAI-Echo
    // monolith (single .safetensors carrying DiT + VAE + vocoder — how the
    // warm server loads sulphur2-distil) needs only ANY .safetensors present.
    // LTX2VideoGenerator.resolveWeightsFileURL does the real resolution; this
    // check just fails fast with a readable message.
    let fm = FileManager.default
    let transformerPath = URL(fileURLWithPath: weightsDir + "/transformer-distilled.safetensors")
    let vaeDecoderPath = URL(fileURLWithPath: weightsDir + "/vae_decoder.safetensors")
    let vaeEncoderPath = URL(fileURLWithPath: weightsDir + "/vae_encoder.safetensors")

    let hasPerComponent = fm.fileExists(atPath: transformerPath.path)
    if hasPerComponent {
      guard fm.fileExists(atPath: vaeDecoderPath.path) else {
        fputs("Error: VAE decoder weights not found at \(vaeDecoderPath.path)\n", stderr)
        exit(1)
      }
      guard fm.fileExists(atPath: vaeEncoderPath.path) else {
        fputs("Error: VAE encoder weights not found at \(vaeEncoderPath.path)\n", stderr)
        exit(1)
      }
    } else {
      let anySafetensors = (try? fm.contentsOfDirectory(atPath: weightsDir))?
        .contains { $0.hasSuffix(".safetensors") } ?? false
      guard anySafetensors else {
        fputs("Error: no LTX-2 weights found in \(weightsDir) (need transformer-distilled.safetensors + VAEs, or a monolithic checkpoint)\n", stderr)
        exit(1)
      }
      // The warm server (LTX2VideoGenerator) loads JoyAI-Echo monoliths, but
      // this CLI command still builds its pipeline inline against the
      // per-component layout. Fail with a pointer instead of an opaque
      // MLXError from a nonexistent transformer-distilled.safetensors.
      fputs("Error: \(weightsDir) holds a monolithic checkpoint. The video CLI requires per-component weights (transformer-distilled.safetensors + VAEs); render monoliths via 'ComfyBox serve' + POST /v1/video/generate instead.\n", stderr)
      exit(1)
    }
    guard fm.fileExists(atPath: gemmaPath) else {
      fputs("Error: Gemma text encoder weights not found at \(gemmaPath)\n", stderr)
      exit(1)
    }

    if let img = imagePath, !fm.fileExists(atPath: img) {
      fputs("Error: source image not found at \(img)\n", stderr)
      exit(1)
    }

    let overallStart = CFAbsoluteTimeGetCurrent()
    var stepNum = 1
    let totalSteps = isI2V ? 8 : 7

    // --- Step 1: Create transformer ---
    print("[\(stepNum)/\(totalSteps)] Creating transformer (48-layer, 32 heads)...")
    let transformer = LTX2Transformer(
      numHeads: 32, headDim: 128, inChannels: 128, outChannels: 128,
      numLayers: 48, crossAttentionDim: 4096, captionChannels: 3840,
      normEps: 1e-6, hasPromptAdaLN: true, timestepScaleMultiplier: 1000,
      positionalEmbeddingTheta: 10000, positionalEmbeddingMaxPos: [20, 2048, 2048],
      useMiddleIndicesGrid: true, ropeMode: .split,
      doublePrecisionRoPE: true
    )
    stepNum += 1

    // --- Step 2: Load transformer weights ---
    print("[\(stepNum)/\(totalSteps)] Loading transformer weights...")
    let startLoad = CFAbsoluteTimeGetCurrent()
    let rawWeights = try MLX.loadArrays(url: transformerPath)
    var sanitized = LTX2Transformer.sanitizeWeights(rawWeights)

    // Merge LoRA if specified
    if let loraFile = loraPath {
      print("  Merging LoRA: \(loraFile) (strength=\(loraStrength))")
      guard fm.fileExists(atPath: loraFile) else {
        fputs("Error: LoRA file not found at \(loraFile)\n", stderr)
        exit(1)
      }
      let loraWeights = try MLX.loadArrays(url: URL(fileURLWithPath: loraFile))
      var mergedCount = 0
      var skippedCount = 0
      for (key, loraA) in loraWeights {
        guard key.hasSuffix(".lora_A.weight") else { continue }
        var baseKey = String(key.dropLast(".lora_A.weight".count))
        if baseKey.hasPrefix("diffusion_model.") {
          baseKey = String(baseKey.dropFirst("diffusion_model.".count))
        }
        if baseKey.contains("audio_") || baseKey.contains("av_ca_") ||
           baseKey.contains("video_to_audio_attn") || baseKey.contains("audio_to_video_attn") {
          skippedCount += 1; continue
        }
        let bKey = key.replacingOccurrences(of: ".lora_A.weight", with: ".lora_B.weight")
        guard let loraB = loraWeights[bKey] else { skippedCount += 1; continue }
        let targetKey = baseKey + ".weight"
        guard sanitized[targetKey] != nil else { skippedCount += 1; continue }
        let delta = MLX.matmul(loraB.asType(.float32), loraA.asType(.float32)) * MLXArray(loraStrength)
        sanitized[targetKey] = sanitized[targetKey]!.asType(.float32) + delta
        mergedCount += 1
      }
      print("  Merged \(mergedCount) LoRA pairs (skipped \(skippedCount))")
    }

    let params = ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) })
    try transformer.update(parameters: params, verify: [.shapeMismatch])
    MLX.eval(transformer.parameters())
    let loadTime = CFAbsoluteTimeGetCurrent() - startLoad
    print("  Loaded \(rawWeights.count) tensors in \(String(format: "%.1f", loadTime))s")
    stepNum += 1

    // --- Step 3: Create and load VAE ---
    print("[\(stepNum)/\(totalSteps)] Creating and loading VAE...")
    let vae = LTX2VAE(config: .v23)
    var combinedVAEWeights: [String: MLXArray] = [:]
    let rawDecoderWeights = try MLX.loadArrays(url: vaeDecoderPath)
    for (key, value) in rawDecoderWeights {
      if key.hasPrefix("vae_decoder.") {
        combinedVAEWeights["vae.decoder." + String(key.dropFirst("vae_decoder.".count))] = value
      }
    }
    let rawEncoderWeights = try MLX.loadArrays(url: vaeEncoderPath)
    for (key, value) in rawEncoderWeights {
      if key.hasPrefix("vae_encoder.") {
        combinedVAEWeights["vae.encoder." + String(key.dropFirst("vae_encoder.".count))] = value
      }
    }
    if let m = combinedVAEWeights["vae.decoder.per_channel_statistics.mean"] {
      combinedVAEWeights["vae.per_channel_statistics.mean-of-means"] = m
    }
    if let s = combinedVAEWeights["vae.decoder.per_channel_statistics.std"] {
      combinedVAEWeights["vae.per_channel_statistics.std-of-means"] = s
    }
    var vaeLogger = Logger(label: "ltx2.video.vae")
    vaeLogger.logLevel = .info
    try LTX2WeightLoader.loadVAEWeightsFromTensors(into: vae, tensors: combinedVAEWeights, logger: vaeLogger)
    MLX.eval(vae.parameters())
    print("  VAE loaded (\(combinedVAEWeights.count) tensors)")
    stepNum += 1

    // --- Step 4: Create and load text encoder ---
    print("[\(stepNum)/\(totalSteps)] Loading text encoder (Gemma 3 12B BF16)...")
    let gemmaConfig = LTX2GemmaConfig(
      vocabSize: 262208, hiddenSize: 3840,
      numHiddenLayers: 48, numAttentionHeads: 16,
      numKeyValueHeads: 8, headDim: 256,
      intermediateSize: 15360,
      rmsNormEps: 1e-6, ropeTheta: 1_000_000.0,
      slidingWindow: 1024, slidingWindowPattern: 6,
      quantization: nil
    )
    let encoderConfig = LTX2TextEncoderConfig(gemma: gemmaConfig, hasPromptAdaLN: true)
    let textEncoder = LTX2TextEncoder(config: encoderConfig)
    let teLoadStart = CFAbsoluteTimeGetCurrent()
    try textEncoder.loadWeights(
      modelPath: URL(fileURLWithPath: weightsDir),
      textEncoderPath: URL(fileURLWithPath: gemmaPath)
    )
    MLX.eval(textEncoder.parameters())
    let teLoadTime = CFAbsoluteTimeGetCurrent() - teLoadStart
    print("  Text encoder loaded in \(String(format: "%.1f", teLoadTime))s")
    stepNum += 1

    // --- Step 5: Create pipeline ---
    print("[\(stepNum)/\(totalSteps)] Creating pipeline...")
    let pipelineConfig = LTX2PipelineConfig(
      modelPath: weightsDir, pipelineType: .distilled, hasPromptAdaLN: true
    )
    let pipeline = LTX2Pipeline(
      vae: vae, textEncoder: textEncoder, transformer: transformer, config: pipelineConfig
    )
    stepNum += 1

    // --- Step 6: Tokenize prompt ---
    print("[\(stepNum)/\(totalSteps)] Tokenizing prompt...")
    let tokenizerDir = URL(fileURLWithPath: gemmaPath)
    let tokenizer = try LTX2GemmaTokenizer.load(from: tokenizerDir, maxLength: 128)
    let batch = tokenizer.encode(prompt: promptText, maxLength: 128)
    MLX.eval(batch.inputIds, batch.attentionMask)
    let tokenCount = MLX.sum(batch.attentionMask).item(Int.self)
    print("  Tokenized: \(tokenCount) tokens")
    stepNum += 1

    // --- Step 7: Generate ---
    let genStart = CFAbsoluteTimeGetCurrent()
    let progressCB: ((Int, Int) -> Void)? = noProgress ? nil : { step, total in
      let elapsed = CFAbsoluteTimeGetCurrent() - genStart
      let rate = step > 0 ? elapsed / Double(step) : 0
      let remaining = rate > 0 ? rate * Double(total - step) : 0
      print(String(format: "  [step %d/%d]  %.1fs elapsed  ~%.0fs remaining",
        step, total, elapsed, remaining))
    }

    let output: LTX2PipelineOutput

    if isI2V {
      // --- I2V: load source image ---
      #if canImport(CoreGraphics) && canImport(ImageIO)
      print("[\(stepNum)/\(totalSteps)] Loading source image...")
      let imgURL = URL(fileURLWithPath: imagePath!)
      guard let imgSource = CGImageSourceCreateWithURL(imgURL as CFURL, nil),
            let cgImage = CGImageSourceCreateImageAtIndex(imgSource, 0, nil) else {
        fputs("Error: failed to load image from \(imagePath!)\n", stderr)
        exit(1)
      }
      let pixelArray = try QwenImageIO.resizedPixelArray(
        from: cgImage, width: width, height: height,
        addBatchDimension: true, dtype: .float32
      )
      let imageForEncoder = QwenImageIO.normalizeForEncoder(pixelArray)
      print("  Image loaded: \(imageForEncoder.shape)")
      stepNum += 1

      print("[\(stepNum)/\(totalSteps)] Generating I2V (\(denoiseSteps) steps, \(numFrames) frames)...")
      output = try pipeline.generateI2V(
        inputIds: batch.inputIds,
        attentionMask: batch.attentionMask,
        image: imageForEncoder,
        strength: strength,
        width: width, height: height,
        numFrames: numFrames,
        steps: denoiseSteps,
        seed: actualSeed,
        progressCallback: progressCB
      )
      #else
      fputs("Error: I2V requires CoreGraphics/ImageIO (macOS only)\n", stderr)
      exit(1)
      #endif
    } else {
      // --- T2V ---
      print("[\(stepNum)/\(totalSteps)] Generating T2V (\(denoiseSteps) steps, \(numFrames) frames)...")
      output = try pipeline.generateT2V(
        inputIds: batch.inputIds,
        attentionMask: batch.attentionMask,
        width: width, height: height,
        numFrames: numFrames,
        steps: denoiseSteps,
        seed: actualSeed,
        progressCallback: progressCB
      )
    }

    print("  Generation complete in \(String(format: "%.1f", output.elapsedSeconds))s")
    stepNum += 1

    // --- Write MP4 ---
    print("[\(stepNum)/\(totalSteps)] Writing MP4 to \(outputPath)...")
    #if canImport(AVFoundation) && canImport(CoreGraphics)
    let cgFrames = LTX2PostProcess.framesToImages(from: output.decoded)
    print("  Extracted \(cgFrames.count) frames")
    try LTX2PostProcess.writeMP4(
      frames: cgFrames,
      outputPath: outputPath,
      fps: fps,
      width: width,
      height: height
    )
    #else
    // Fallback: write PPM frames and invoke ffmpeg
    let ppmDir = outputPath.replacingOccurrences(of: ".mp4", with: "-frames")
    try LTX2PostProcess.writeFramesPPM(from: output.decoded, outputDir: ppmDir)
    print("  Wrote PPM frames to \(ppmDir)")
    print("  Converting to MP4 via ffmpeg...")
    let ffmpeg = Process()
    ffmpeg.executableURL = URL(fileURLWithPath: "/usr/local/bin/ffmpeg")
    ffmpeg.arguments = [
      "-y", "-framerate", "\(fps)",
      "-i", "\(ppmDir)/frame_%04d.ppm",
      "-c:v", "libx264", "-pix_fmt", "yuv420p",
      "-crf", "18", outputPath
    ]
    try ffmpeg.run()
    ffmpeg.waitUntilExit()
    if ffmpeg.terminationStatus != 0 {
      fputs("Warning: ffmpeg exited with status \(ffmpeg.terminationStatus)\n", stderr)
    }
    #endif

    // --- Report ---
    let overallTime = CFAbsoluteTimeGetCurrent() - overallStart
    let videoDuration = Float(output.numFrames) / Float(fps)
    print()
    print("=== Done ===")
    print("Output:   \(outputPath)")
    if let attrs = try? fm.attributesOfItem(atPath: outputPath),
       let size = attrs[FileAttributeKey.size] as? Int {
      let mb = Double(size) / (1024.0 * 1024.0)
      print("Size:     \(String(format: "%.1f", mb)) MB")
    }
    print("Mode:     \(mode)")
    print("Frames:   \(output.numFrames)")
    print("Duration: \(String(format: "%.1f", videoDuration))s @ \(fps)fps")
    print("Seed:     \(actualSeed)")
    print("Time:     \(String(format: "%.1f", overallTime))s total (\(String(format: "%.1f", output.elapsedSeconds))s generation)")
    print()
  }

  private static func printVideoUsage() {
    print("""

    ComfyBox video — Native LTX-2 video generation on Apple Silicon.

    Usage:
      ComfyBox video -p "prompt" [options]              # Text-to-Video
      ComfyBox video -p "motion" -i source.png [opts]   # Image-to-Video

    Required:
      -p, --prompt <text>       Scene description (T2V) or motion description (I2V)

    Options:
      -i, --image <path>        Source image for I2V mode. Omit for T2V.
      -o, --output <path>       Output .mp4 path (default: z-video.mp4)
      -d, --duration <seconds>  T2V duration: 6,8,10,12,14,16,18,20 (default: 6). Ignored for I2V.
      -r, --resolution <res>    Output resolution: 480p, 720p, 1080p (default: 720p)
      --aspect-ratio <ratio>    Aspect ratio: 16:9 or 9:16 (default: 16:9)
      --seed <uint64>           Random seed for reproducibility
      -w, --weights <dir>       LTX2 model weights directory (default: ~/Models/ltx2-distilled/)
      --gemma-path <dir>        Gemma 3 text encoder weights (default: HF cache)
      --lora <path>             LoRA safetensors file (merge-on-load)
      --lora-strength <float>   LoRA merge strength (default: 1.0)
      --steps <int>             Denoising steps (default: 8 distilled)
      --strength <float>        I2V conditioning strength 0-1 (default: 1.0)
      --no-progress             Disable step-by-step progress output
      -h, --help                Show this help

    Examples:
      ComfyBox video -p "a woman walks through a sunlit garden" -o garden.mp4
      ComfyBox video -p "she turns and smiles" -i photo.png -o smile.mp4
      ComfyBox video -p "ocean waves at sunset" -d 10 -r 1080p --seed 42
      ComfyBox video -p "camera pans across a city" --aspect-ratio 9:16 -r 720p

    Notes:
      - Uses the LTX-2.3 distilled pipeline (8 steps, no CFG) by default.
      - Model loading takes 1-3 minutes for the 35GB transformer. This is a cold
        invocation — for repeated generation, use 'ComfyBox serve' instead.
      - T2V frame counts are computed from duration at 24fps, rounded to 1+8k.
      - I2V always generates 97 frames (~4s at 24fps).
    """)
  }

  // MARK: - LTX2 Demo

  private static func runLTX2Demo(args: [String]) throws {
    var modelDir = "/Users/toddwalderman/Models/ltx2-distilled"
    var outputPath = "/tmp/ltx2-demo.mp4"
    var embeddingsPath: String? = nil
    var prompt: String? = nil
    var gemmaPath = "/Users/toddwalderman/.cache/huggingface/hub/models--unsloth--gemma-3-12b-it/snapshots/9478e665381f42974aa06177b019352fb6291876"
    var width = 512
    var height = 320
    var frames = 9
    var steps = 8
    var seed = 42
    var loraPath: String? = nil
    var loraStrength: Float = 1.0

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--model-dir":
        modelDir = nextValue(for: arg, iterator: &iterator)
      case "--output", "-o":
        outputPath = nextValue(for: arg, iterator: &iterator)
      case "--width", "-W":
        width = intValue(for: arg, iterator: &iterator, minimum: 32, fallback: width)
      case "--height", "-H":
        height = intValue(for: arg, iterator: &iterator, minimum: 32, fallback: height)
      case "--frames":
        frames = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: frames)
      case "--steps":
        steps = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: steps)
      case "--seed":
        seed = intValue(for: arg, iterator: &iterator, minimum: 0, fallback: seed)
      case "--embeddings":
        embeddingsPath = nextValue(for: arg, iterator: &iterator)
      case "--prompt", "-p":
        prompt = nextValue(for: arg, iterator: &iterator)
      case "--gemma-path":
        gemmaPath = nextValue(for: arg, iterator: &iterator)
      case "--lora-path":
        loraPath = nextValue(for: arg, iterator: &iterator)
      case "--lora-strength":
        loraStrength = floatValue(for: arg, iterator: &iterator, fallback: 1.0)
      case "--help", "-h":
        printLTX2DemoUsage()
        return
      default:
        logger.warning("Unknown ltx2-demo argument: \(arg)")
      }
    }

    // Determine if we should use the real text encoder
    let useRealEncoder = prompt != nil && embeddingsPath == nil
    let totalSteps = useRealEncoder ? 9 : 7

    print("=== LTX2 End-to-End Demo ===")
    print("Model dir:  \(modelDir)")
    print("Output:     \(outputPath)")
    print("Prompt:     \(prompt ?? "(none)")")
    if useRealEncoder {
      print("Encoder:    Gemma 3 12B BF16 (live)")
      print("Gemma path: \(gemmaPath)")
    } else {
      print("Embeddings: \(embeddingsPath ?? "random (dummy)")")
    }
    print("Resolution: \(width)x\(height)")
    print("Frames:     \(frames)")
    print("Steps:      \(steps)")
    print("Seed:       \(seed)")
    if let lp = loraPath {
      print("LoRA:       \(lp)")
      print("LoRA str:   \(loraStrength)")
    }
    print()

    var stepNum = 1

    // --- Create transformer ---
    print("[\(stepNum)/\(totalSteps)] Creating transformer (48-layer, 32 heads)...")
    let transformer = LTX2Transformer(
      numHeads: 32, headDim: 128, inChannels: 128, outChannels: 128,
      numLayers: 48, crossAttentionDim: 4096, captionChannels: 3840,
      normEps: 1e-6, hasPromptAdaLN: true, timestepScaleMultiplier: 1000,
      positionalEmbeddingTheta: 10000, positionalEmbeddingMaxPos: [20, 2048, 2048],
      useMiddleIndicesGrid: true, ropeMode: .split,
      doublePrecisionRoPE: true
    )
    stepNum += 1

    // --- Load transformer weights ---
    print("[\(stepNum)/\(totalSteps)] Loading transformer weights (this takes 1-3 minutes for 35GB)...")
    let transformerPath = URL(fileURLWithPath: modelDir + "/transformer-distilled.safetensors")
    guard FileManager.default.fileExists(atPath: transformerPath.path) else {
      throw NSError(domain: "LTX2Demo", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "Transformer weights not found at \(transformerPath.path)"])
    }
    let startLoad = CFAbsoluteTimeGetCurrent()
    let rawWeights = try MLX.loadArrays(url: transformerPath)
    var sanitized = LTX2Transformer.sanitizeWeights(rawWeights)

    // --- Merge LoRA weights if specified ---
    if let loraFile = loraPath {
      print("  Merging LoRA: \(loraFile) (strength=\(loraStrength))")
      guard FileManager.default.fileExists(atPath: loraFile) else {
        throw NSError(domain: "LTX2Demo", code: 10,
          userInfo: [NSLocalizedDescriptionKey: "LoRA file not found at \(loraFile)"])
      }
      let loraWeights = try MLX.loadArrays(url: URL(fileURLWithPath: loraFile))
      var mergedCount = 0
      var skippedCount = 0

      for (key, loraA) in loraWeights {
        guard key.hasSuffix(".lora_A.weight") else { continue }

        // Strip lora_A.weight suffix to get base key
        var baseKey = String(key.dropLast(".lora_A.weight".count))

        // Strip diffusion_model. prefix (ComfyUI convention)
        if baseKey.hasPrefix("diffusion_model.") {
          baseKey = String(baseKey.dropFirst("diffusion_model.".count))
        }

        // Skip audio-related LoRA keys
        if baseKey.contains("video_to_audio_attn") ||
           baseKey.contains("audio_to_video_attn") ||
           baseKey.contains("audio_attn") ||
           baseKey.contains("audio_ff") ||
           baseKey.contains("audio_") ||
           baseKey.contains("av_ca_") {
          skippedCount += 1
          continue
        }

        // Find matching lora_B tensor
        let bKey = key.replacingOccurrences(of: ".lora_A.weight", with: ".lora_B.weight")
        guard let loraB = loraWeights[bKey] else {
          skippedCount += 1
          continue
        }

        // Target key in sanitized weights dict
        let targetKey = baseKey + ".weight"
        guard sanitized[targetKey] != nil else {
          skippedCount += 1
          continue
        }

        // delta = B @ A * strength (float32 for precision)
        let delta = MLX.matmul(loraB.asType(.float32), loraA.asType(.float32)) * MLXArray(loraStrength)
        sanitized[targetKey] = sanitized[targetKey]!.asType(.float32) + delta
        mergedCount += 1
      }
      print("  Merged \(mergedCount) LoRA weight pairs (skipped \(skippedCount))")
    }

    let params = ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) })
    try transformer.update(parameters: params, verify: [.shapeMismatch])
    MLX.eval(transformer.parameters())
    let loadTime = CFAbsoluteTimeGetCurrent() - startLoad
    print("  Loaded \(rawWeights.count) tensors in \(String(format: "%.1f", loadTime))s")
    stepNum += 1

    // --- Create and load VAE ---
    print("[\(stepNum)/\(totalSteps)] Creating and loading VAE...")
    let vae = LTX2VAE(config: .v23)

    let vaeDecoderPath = URL(fileURLWithPath: modelDir + "/vae_decoder.safetensors")
    let vaeEncoderPath = URL(fileURLWithPath: modelDir + "/vae_encoder.safetensors")
    guard FileManager.default.fileExists(atPath: vaeDecoderPath.path) else {
      throw NSError(domain: "LTX2Demo", code: 2,
        userInfo: [NSLocalizedDescriptionKey: "VAE decoder weights not found at \(vaeDecoderPath.path)"])
    }
    guard FileManager.default.fileExists(atPath: vaeEncoderPath.path) else {
      throw NSError(domain: "LTX2Demo", code: 3,
        userInfo: [NSLocalizedDescriptionKey: "VAE encoder weights not found at \(vaeEncoderPath.path)"])
    }

    var combinedVAEWeights: [String: MLXArray] = [:]
    let rawDecoderWeights = try MLX.loadArrays(url: vaeDecoderPath)
    for (key, value) in rawDecoderWeights {
      if key.hasPrefix("vae_decoder.") {
        combinedVAEWeights["vae.decoder." + String(key.dropFirst("vae_decoder.".count))] = value
      }
    }
    let rawEncoderWeights = try MLX.loadArrays(url: vaeEncoderPath)
    for (key, value) in rawEncoderWeights {
      if key.hasPrefix("vae_encoder.") {
        combinedVAEWeights["vae.encoder." + String(key.dropFirst("vae_encoder.".count))] = value
      }
    }
    if let m = combinedVAEWeights["vae.decoder.per_channel_statistics.mean"] {
      combinedVAEWeights["vae.per_channel_statistics.mean-of-means"] = m
    }
    if let s = combinedVAEWeights["vae.decoder.per_channel_statistics.std"] {
      combinedVAEWeights["vae.per_channel_statistics.std-of-means"] = s
    }
    print("  Combined \(combinedVAEWeights.count) VAE weights")

    var vaeLogger = Logger(label: "ltx2.demo.vae")
    vaeLogger.logLevel = .info
    try LTX2WeightLoader.loadVAEWeightsFromTensors(
      into: vae,
      tensors: combinedVAEWeights,
      logger: vaeLogger
    )
    print("  VAE weight loading completed, evaluating parameters...")
    MLX.eval(vae.parameters())
    print("  VAE weights loaded and evaluated successfully")
    stepNum += 1

    // --- Create text encoder (real or dummy) ---
    let textEncoder: LTX2TextEncoder
    if useRealEncoder {
      print("[\(stepNum)/\(totalSteps)] Creating text encoder (Gemma 3 12B BF16 + connectors)...")
      let gemmaConfig = LTX2GemmaConfig(
        vocabSize: 262208,
        hiddenSize: 3840,
        numHiddenLayers: 48,
        numAttentionHeads: 16,
        numKeyValueHeads: 8,
        headDim: 256,
        intermediateSize: 15360,
        rmsNormEps: 1e-6,
        ropeTheta: 1_000_000.0,
        slidingWindow: 1024,
        slidingWindowPattern: 6,
        quantization: nil
      )
      let encoderConfig = LTX2TextEncoderConfig(
        gemma: gemmaConfig,
        hasPromptAdaLN: true
      )
      textEncoder = LTX2TextEncoder(config: encoderConfig)
      stepNum += 1

      print("[\(stepNum)/\(totalSteps)] Loading text encoder weights...")
      print("  Gemma path: \(gemmaPath)")
      print("  Connector path: \(modelDir)")
      let teLoadStart = CFAbsoluteTimeGetCurrent()
      try textEncoder.loadWeights(
        modelPath: URL(fileURLWithPath: modelDir),
        textEncoderPath: URL(fileURLWithPath: gemmaPath)
      )
      MLX.eval(textEncoder.parameters())
      let teLoadTime = CFAbsoluteTimeGetCurrent() - teLoadStart
      print("  Text encoder weights loaded in \(String(format: "%.1f", teLoadTime))s")
      stepNum += 1
    } else {
      // Dummy text encoder (bypassed via embeddings)
      textEncoder = LTX2TextEncoder(config: LTX2TextEncoderConfig())
    }

    // --- Create pipeline ---
    print("[\(stepNum)/\(totalSteps)] Creating pipeline...")
    let pipelineConfig = LTX2PipelineConfig(
      modelPath: modelDir,
      pipelineType: .distilled,
      hasPromptAdaLN: true
    )

    let pipeline = LTX2Pipeline(
      vae: vae,
      textEncoder: textEncoder,
      transformer: transformer,
      config: pipelineConfig
    )
    stepNum += 1

    // --- Generate embeddings or load them ---
    let videoEmbeddings: MLXArray
    if useRealEncoder, let promptText = prompt {
      // End-to-end: tokenize prompt -> Gemma 3 -> connector -> video embeddings
      print("[\(stepNum)/\(totalSteps)] Tokenizing and encoding prompt...")
      let tokenizerDir = URL(fileURLWithPath: gemmaPath)
      let tokenizer = try LTX2GemmaTokenizer.load(from: tokenizerDir, maxLength: 128)
      let batch = tokenizer.encode(prompt: promptText, maxLength: 128)
      MLX.eval(batch.inputIds, batch.attentionMask)
      let maskSum = MLX.sum(batch.attentionMask).item(Int.self)
      print("  Tokenized: \(maskSum) tokens")

      let encodeStart = CFAbsoluteTimeGetCurrent()
      let textOutput = textEncoder.encode(
        inputIds: batch.inputIds,
        attentionMask: batch.attentionMask,
        returnAudioEmbeddings: false
      )
      MLX.eval(textOutput.videoEmbeddings)
      let encodeTime = CFAbsoluteTimeGetCurrent() - encodeStart
      print("  Video embeddings shape: \(textOutput.videoEmbeddings.shape)")
      print("  Encoding time: \(String(format: "%.1f", encodeTime))s")

      // Embedding statistics for diagnostics
      let embF32 = textOutput.videoEmbeddings.asType(.float32)
      let embMin = MLX.min(embF32).item(Float.self)
      let embMax = MLX.max(embF32).item(Float.self)
      let embMean = MLX.mean(embF32).item(Float.self)
      print(String(format: "  Embedding stats: min=%.4f  max=%.4f  mean=%.6f", embMin, embMax, embMean))

      videoEmbeddings = textOutput.videoEmbeddings
    } else if let embPath = embeddingsPath {
      print("[\(stepNum)/\(totalSteps)] Loading pre-computed embeddings from \(embPath)...")
      let embURL = URL(fileURLWithPath: embPath)
      guard FileManager.default.fileExists(atPath: embPath) else {
        throw NSError(domain: "LTX2Demo", code: 4,
          userInfo: [NSLocalizedDescriptionKey: "Embeddings file not found at \(embPath)"])
      }
      let embTensors = try MLX.loadArrays(url: embURL)
      guard let emb = embTensors["video_embeddings"] else {
        throw NSError(domain: "LTX2Demo", code: 5,
          userInfo: [NSLocalizedDescriptionKey: "No video_embeddings tensor in \(embPath). Keys: \(Array(embTensors.keys))"])
      }
      videoEmbeddings = emb
      print("  Loaded embeddings shape: \(videoEmbeddings.shape) dtype: \(videoEmbeddings.dtype)")
    } else {
      print("[\(stepNum)/\(totalSteps)] Generating video with dummy embeddings...")
      print("  Using random embeddings (no text encoder) -- output will be abstract noise")
      MLXRandom.seed(UInt64(seed))
      videoEmbeddings = MLXRandom.normal([1, 32, 4096]) * Float(0.01)
    }
    MLX.eval(videoEmbeddings)
    stepNum += 1

    print("[\(stepNum)/\(totalSteps)] Running denoising loop (\(steps) steps)...")
    let genStart = CFAbsoluteTimeGetCurrent()
    let output = try pipeline.generateT2VWithEmbeddings(
      videoEmbeddings: videoEmbeddings,
      width: width, height: height, numFrames: frames,
      steps: steps, seed: UInt64(seed),
      progressCallback: { step, total in
        let elapsed = CFAbsoluteTimeGetCurrent() - genStart
        let rate = step > 0 ? elapsed / Double(step) : 0
        let remaining = rate > 0 ? rate * Double(total - step) : 0
        print(String(format: "  [step %d/%d]  %.1fs elapsed  ~%.0fs remaining",
          step, total, elapsed, remaining))
      }
    )

    print("  Generation complete in \(String(format: "%.1f", output.elapsedSeconds))s")
    print("  Output shape: \(output.decoded.shape)")
    let decodedF32 = output.decoded.asType(.float32)
    let pixMin = MLX.min(decodedF32).item(Float.self)
    let pixMax = MLX.max(decodedF32).item(Float.self)
    let pixMean = MLX.mean(decodedF32).item(Float.self)
    eval(decodedF32)
    print(String(format: "  Pixel range: min=%.4f  max=%.4f  mean=%.4f", pixMin, pixMax, pixMean))
    if pixMax < 0.1 {
      print("  WARNING: output is very dark (pixMax < 0.1) -- transformer may not be loading weights correctly")
    } else if pixMean < 0.05 || pixMean > 0.95 {
      print("  WARNING: output mean is extreme -- possible normalization issue")
    } else {
      print("  Pixel values look reasonable")
    }
    stepNum += 1

    // --- Write MP4 ---
    print("[\(stepNum)/\(totalSteps)] Writing MP4 to \(outputPath)...")
    #if canImport(AVFoundation) && canImport(CoreGraphics)
    let cgFrames = LTX2PostProcess.framesToImages(from: output.decoded)
    print("  Extracted \(cgFrames.count) CGImage frames")
    try LTX2PostProcess.writeMP4(
      frames: cgFrames,
      outputPath: outputPath,
      fps: 24,
      width: width,
      height: height
    )
    #else
    let ppmDir = outputPath.replacingOccurrences(of: ".mp4", with: "-frames")
    try LTX2PostProcess.writeFramesPPM(from: output.decoded, outputDir: ppmDir)
    print("  Wrote PPM frames to \(ppmDir) (AVFoundation not available)")
    #endif

    // Report
    let fm = FileManager.default
    if let attrs = try? fm.attributesOfItem(atPath: outputPath),
       let size = attrs[.size] as? Int {
      let kb = Double(size) / 1024.0
      print()
      print("=== Done ===")
      print("Output: \(outputPath)")
      print("Size:   \(String(format: "%.1f", kb)) KB")
      print("Frames: \(output.numFrames)")
      print("Time:   \(String(format: "%.1f", output.elapsedSeconds))s")
    } else {
      print()
      print("=== Done ===")
      print("Output: \(outputPath)")
    }
  }

  private static func printLTX2DemoUsage() {
    print("""
    LTX-2.3 distilled text-to-video pipeline.
    Supports end-to-end generation from text prompt (Gemma 3 12B BF16 text encoder)
    or pre-computed embeddings.

    Usage: ComfyBox ltx2-demo [options]
      --model-dir <path>        Model weights directory (default: ~/Models/ltx2-distilled)
      --output, -o <path>       Output MP4 path (default: /tmp/ltx2-demo.mp4)
      --prompt, -p <text>       Text prompt (triggers end-to-end text encoding)
      --gemma-path <path>       Path to Gemma 3 weights directory (default: HF cache)
      --embeddings <path>       Pre-computed embeddings (.safetensors, key: video_embeddings)
      --width, -W <int>         Video width in pixels, div by 32 (default: 512)
      --height, -H <int>        Video height in pixels, div by 32 (default: 320)
      --frames <int>            Number of frames, must be 1+8k (default: 9)
      --steps <int>             Denoising steps (default: 8)
      --seed <int>              Random seed (default: 42)
      --lora-path <path>        Path to LoRA safetensors file (merge-on-load)
      --lora-strength <float>   LoRA merge strength (default: 1.0)
      --help, -h                Show help

    The distilled pipeline uses 8 fixed sigma steps, guidance=1.0, no CFG.
    Priority: --embeddings > --prompt > random dummy embeddings.
    """)
  }

  // MARK: - LTX2 VAE Decoder Test
  private static func runLTX2VAETest(args: [String]) throws {
    print("=== LTX2 VAE Decoder Test ===")
    let modelDir = "/Users/toddwalderman/.cache/huggingface/hub/models/dgrauet/ltx-2.3-mlx"
    var vaeLogger = Logger(label: "ltx2.vae.test")
    vaeLogger.logLevel = .info

    let vae = LTX2VAE(config: .v23)

    var combinedVAEWeights: [String: MLXArray] = [:]
    let rawDecW = try MLX.loadArrays(url: URL(fileURLWithPath: modelDir + "/vae_decoder.safetensors"))
    for (key, value) in rawDecW {
      if key.hasPrefix("vae_decoder.") {
        combinedVAEWeights["vae.decoder." + String(key.dropFirst("vae_decoder.".count))] = value
      }
    }
    let rawEncW = try MLX.loadArrays(url: URL(fileURLWithPath: modelDir + "/vae_encoder.safetensors"))
    for (key, value) in rawEncW {
      if key.hasPrefix("vae_encoder.") {
        combinedVAEWeights["vae.encoder." + String(key.dropFirst("vae_encoder.".count))] = value
      }
    }
    if let m = combinedVAEWeights["vae.decoder.per_channel_statistics.mean"] {
      combinedVAEWeights["vae.per_channel_statistics.mean-of-means"] = m
    }
    if let s = combinedVAEWeights["vae.decoder.per_channel_statistics.std"] {
      combinedVAEWeights["vae.per_channel_statistics.std-of-means"] = s
    }

    try LTX2WeightLoader.loadVAEWeightsFromTensors(into: vae, tensors: combinedVAEWeights, logger: vaeLogger)
    MLX.eval(vae.parameters())
    print("VAE weights loaded")

    // --- SWIFT ENCODER TEST: encode source.png via the generator's exact path ---
    #if canImport(CoreGraphics) && canImport(ImageIO)
    let srcPath = "/tmp/ltx2-test/source.png"
    if let imgSource = CGImageSourceCreateWithURL(URL(fileURLWithPath: srcPath) as CFURL, nil),
       let cgImage = CGImageSourceCreateImageAtIndex(imgSource, 0, nil) {
      let pixels = try QwenImageIO.resizedPixelArray(
        from: cgImage, width: 704, height: 448, addBatchDimension: true, dtype: .float32)
      let normed = QwenImageIO.normalizeForEncoder(pixels)   // (1,3,H,W) in [-1,1]
      print("Encoder input: \(normed.shape) min=\(MLX.min(normed).item(Float.self)) max=\(MLX.max(normed).item(Float.self))")
      let encLat = vae.encode(normed)
      MLX.eval(encLat)
      let el = encLat.asType(.float32)
      print(String(format: "SWIFT encoded latent: %@ mean=%.5f std=%.5f  (PY ref: mean=-0.01928 std=0.78448)",
        "\(el.shape)", MLX.mean(el).item(Float.self), MLX.sqrt(MLX.variance(el)).item(Float.self)))
      try? MLX.save(arrays: ["latent": el], url: URL(fileURLWithPath: "/tmp/swift_latent.safetensors"))
      // round-trip: decode the swift-encoded latent
      let rt = vae.decode(el)
      MLX.eval(rt)
      let rtf = MLX.clip((rt[0..., 0..., 0..<1, 0..., 0...].squeezed(axis: 2) + 1.0) / 2.0, min: 0, max: 1)
      MLX.eval(rtf)
      try? QwenImageIO.saveImage(array: rtf[0], to: URL(fileURLWithPath: "/tmp/ltx2-test/swift_roundtrip.png"))
      print("Saved swift round-trip -> /tmp/ltx2-test/swift_roundtrip.png ; latent -> /tmp/swift_latent.safetensors")
    } else {
      print("(encoder test skipped: could not load \(srcPath))")
    }
    #endif

    let latTensors = try MLX.loadArrays(url: URL(fileURLWithPath: "/tmp/test_latent.safetensors"))
    guard let lat = latTensors["latent"] else { print("ERROR: No latent"); return }
    let latF32 = lat.asType(.float32)
    print("Latent: \(latF32.shape)")

    let decoded = vae.decode(latF32)
    MLX.eval(decoded)
    let df = decoded.asType(.float32)
    print(String(format: "Output: %@ min=%.4f max=%.4f mean=%.4f",
      "\(decoded.shape)",
      MLX.min(df).item(Float.self),
      MLX.max(df).item(Float.self),
      MLX.mean(df).item(Float.self)))

    let frame = df[0, 0, 0]
    MLX.eval(frame)
    print("\nR channel pixel grid (first 8x8):")
    for row in 0..<8 {
      var vals = [String]()
      for col in 0..<8 {
        vals.append(String(format: "%7.3f", frame[row, col].item(Float.self)))
      }
      print("  row \(row): \(vals.joined(separator: " "))")
    }

    print("\nPython ref R-channel (first 8x8) — smooth gradient, no grid:")
    print("  -0.104 -0.103 -0.103 -0.101 -0.098 -0.097 -0.096 -0.091")
    print("  -0.122 -0.120 -0.109 -0.105 -0.099 -0.094 -0.097 -0.095")
    print("  -0.121 -0.118 -0.122 -0.117 -0.118 -0.110 -0.111 -0.106")
    print("  -0.136 -0.130 -0.128 -0.124 -0.131 -0.122 -0.118 -0.124")

    print("\nGrid analysis:")
    for startRow in stride(from: 0, to: 12, by: 4) {
      var subs = [Float]()
      for q in 0..<4 {
        var sum: Float = 0
        for col in 0..<min(16, Int(decoded.dim(4))) { sum += frame[startRow + q, col].item(Float.self) }
        subs.append(sum / 16.0)
      }
      print(String(format: "  Block row %d: q0=%.4f q1=%.4f q2=%.4f q3=%.4f spread=%.4f",
        startRow, subs[0], subs[1], subs[2], subs[3], subs.max()! - subs.min()!))
    }

    // Save decoded frame 0 to PNG for visual inspection.
    #if canImport(CoreGraphics) && canImport(ImageIO)
    let rescaled = MLX.clip((df[0..., 0..., 0..<1, 0..., 0...].squeezed(axis: 2) + 1.0) / 2.0, min: 0, max: 1)
    MLX.eval(rescaled)
    let chw = rescaled[0]  // (3, H, W)
    try? QwenImageIO.saveImage(array: chw, to: URL(fileURLWithPath: "/tmp/ltx2-test/swift_decode.png"))
    print("Saved decoded frame 0 -> /tmp/ltx2-test/swift_decode.png")
    #endif
    print("=== Done ===")
  }


  // MARK: - LTX2 Image-to-Video + Extend

  private static func runLTX2I2V(args: [String]) throws {
    var modelDir = "/Users/toddwalderman/Models/ltx2-distilled"
    var outputPath = "/tmp/ltx2-i2v.mp4"
    var prompt: String? = nil
    var gemmaPath = "/Users/toddwalderman/.cache/huggingface/hub/models--unsloth--gemma-3-12b-it/snapshots/9478e665381f42974aa06177b019352fb6291876"
    var width = 704
    var height = 448
    var framesPerChunk = 97
    var steps = 8
    var seed = 42
    var loraPath: String? = nil
    var loraStrength: Float = 1.0
    var initImagePath: String = ""
    var extendToSeconds: Float = 0
    var strength: Float = 1.0

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--model-dir":
        modelDir = nextValue(for: arg, iterator: &iterator)
      case "--output", "-o":
        outputPath = nextValue(for: arg, iterator: &iterator)
      case "--width", "-W":
        width = intValue(for: arg, iterator: &iterator, minimum: 32, fallback: width)
      case "--height", "-H":
        height = intValue(for: arg, iterator: &iterator, minimum: 32, fallback: height)
      case "--frames":
        framesPerChunk = intValue(for: arg, iterator: &iterator, minimum: 9, fallback: framesPerChunk)
      case "--steps":
        steps = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: steps)
      case "--seed":
        seed = intValue(for: arg, iterator: &iterator, minimum: 0, fallback: seed)
      case "--prompt", "-p":
        prompt = nextValue(for: arg, iterator: &iterator)
      case "--gemma-path":
        gemmaPath = nextValue(for: arg, iterator: &iterator)
      case "--lora-path":
        loraPath = nextValue(for: arg, iterator: &iterator)
      case "--lora-strength":
        loraStrength = floatValue(for: arg, iterator: &iterator, fallback: 1.0)
      case "--init-image":
        initImagePath = nextValue(for: arg, iterator: &iterator)
      case "--extend-to":
        extendToSeconds = floatValue(for: arg, iterator: &iterator, fallback: 0)
      case "--strength":
        strength = floatValue(for: arg, iterator: &iterator, fallback: 1.0)
      default:
        logger.warning("Unknown ltx2-i2v argument: \(arg)")
      }
    }

    guard !initImagePath.isEmpty else {
      print("ERROR: --init-image is required for ltx2-i2v")
      print("Usage: ComfyBox ltx2-i2v --init-image <path> --prompt <text> [options]")
      print("  --extend-to <seconds>   Target duration (generates multiple chunks)")
      print("  --frames <N>            Frames per chunk (default 97, must be 1+8k)")
      print("  --strength <0-1>        I2V conditioning strength (default 1.0)")
      return
    }

    guard let promptText = prompt else {
      print("ERROR: --prompt is required for ltx2-i2v")
      return
    }

    guard (framesPerChunk - 1) % 8 == 0 else {
      print("ERROR: --frames must be 1 + 8k (e.g. 9, 17, 25, 33, 97)")
      return
    }

    let fps = 24
    let totalChunks: Int
    if extendToSeconds > 0 {
      let targetFrames = Int(extendToSeconds * Float(fps))
      let continuations = max(0, Int(ceil(Float(targetFrames - framesPerChunk) / Float(framesPerChunk - 1))))
      totalChunks = 1 + continuations
    } else {
      totalChunks = 1
    }

    let totalFrames = framesPerChunk + (framesPerChunk - 1) * (totalChunks - 1)
    let totalDuration = Float(totalFrames) / Float(fps)

    print("=== LTX2 Image-to-Video" + (totalChunks > 1 ? " + Extend" : "") + " ===")
    print("Model dir:    \(modelDir)")
    print("Output:       \(outputPath)")
    print("Init image:   \(initImagePath)")
    print("Prompt:       \(promptText)")
    print("Resolution:   \(width)x\(height)")
    print("Frames/chunk: \(framesPerChunk)")
    print("Chunks:       \(totalChunks)")
    print("Total frames: \(totalFrames) (\(String(format: "%.1f", totalDuration))s)")
    print("Steps:        \(steps)")
    print("Seed:         \(seed)")
    print("Strength:     \(strength)")
    if let lp = loraPath {
      print("LoRA:         \(lp)")
      print("LoRA str:     \(loraStrength)")
    }
    print()

    print("[1] Creating transformer...")
    let transformer = LTX2Transformer(
      numHeads: 32, headDim: 128, inChannels: 128, outChannels: 128,
      numLayers: 48, crossAttentionDim: 4096, captionChannels: 3840,
      normEps: 1e-6, hasPromptAdaLN: true, timestepScaleMultiplier: 1000,
      positionalEmbeddingTheta: 10000, positionalEmbeddingMaxPos: [20, 2048, 2048],
      useMiddleIndicesGrid: true, ropeMode: .split,
      doublePrecisionRoPE: true
    )

    print("[2] Loading transformer weights...")
    let transformerPath = URL(fileURLWithPath: modelDir + "/transformer-distilled.safetensors")
    let startLoad = CFAbsoluteTimeGetCurrent()
    let rawWeights = try MLX.loadArrays(url: transformerPath)
    var sanitized = LTX2Transformer.sanitizeWeights(rawWeights)

    if let loraFile = loraPath {
      print("  Merging LoRA: \(loraFile) (strength=\(loraStrength))")
      let loraWeights = try MLX.loadArrays(url: URL(fileURLWithPath: loraFile))
      var mergedCount = 0
      var skippedCount = 0
      for (key, loraA) in loraWeights {
        guard key.hasSuffix(".lora_A.weight") else { continue }
        var baseKey = String(key.dropLast(".lora_A.weight".count))
        if baseKey.hasPrefix("diffusion_model.") {
          baseKey = String(baseKey.dropFirst("diffusion_model.".count))
        }
        if baseKey.contains("audio_") || baseKey.contains("av_ca_") ||
           baseKey.contains("video_to_audio_attn") || baseKey.contains("audio_to_video_attn") {
          skippedCount += 1; continue
        }
        let bKey = key.replacingOccurrences(of: ".lora_A.weight", with: ".lora_B.weight")
        guard let loraB = loraWeights[bKey] else { skippedCount += 1; continue }
        let targetKey = baseKey + ".weight"
        guard sanitized[targetKey] != nil else { skippedCount += 1; continue }
        let delta = MLX.matmul(loraB.asType(.float32), loraA.asType(.float32)) * MLXArray(loraStrength)
        sanitized[targetKey] = sanitized[targetKey]!.asType(.float32) + delta
        mergedCount += 1
      }
      print("  Merged \(mergedCount) LoRA pairs (skipped \(skippedCount))")
    }

    let params = ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) })
    try transformer.update(parameters: params, verify: [.shapeMismatch])
    MLX.eval(transformer.parameters())
    let loadTime = CFAbsoluteTimeGetCurrent() - startLoad
    print("  Loaded in \(String(format: "%.1f", loadTime))s")

    print("[3] Loading VAE...")
    let vae = LTX2VAE(config: .v23)
    let vaeDecoderPath = URL(fileURLWithPath: modelDir + "/vae_decoder.safetensors")
    let vaeEncoderPath = URL(fileURLWithPath: modelDir + "/vae_encoder.safetensors")
    var combinedVAEWeights: [String: MLXArray] = [:]
    let rawDecoderWeights = try MLX.loadArrays(url: vaeDecoderPath)
    for (key, value) in rawDecoderWeights {
      if key.hasPrefix("vae_decoder.") {
        combinedVAEWeights["vae.decoder." + String(key.dropFirst("vae_decoder.".count))] = value
      }
    }
    let rawEncoderWeights = try MLX.loadArrays(url: vaeEncoderPath)
    for (key, value) in rawEncoderWeights {
      if key.hasPrefix("vae_encoder.") {
        combinedVAEWeights["vae.encoder." + String(key.dropFirst("vae_encoder.".count))] = value
      }
    }
    if let m = combinedVAEWeights["vae.decoder.per_channel_statistics.mean"] {
      combinedVAEWeights["vae.per_channel_statistics.mean-of-means"] = m
    }
    if let s = combinedVAEWeights["vae.decoder.per_channel_statistics.std"] {
      combinedVAEWeights["vae.per_channel_statistics.std-of-means"] = s
    }
    var vaeLogger = Logger(label: "ltx2.i2v.vae")
    vaeLogger.logLevel = .info
    try LTX2WeightLoader.loadVAEWeightsFromTensors(into: vae, tensors: combinedVAEWeights, logger: vaeLogger)
    MLX.eval(vae.parameters())

    print("[4] Loading text encoder (Gemma 3 12B BF16)...")
    let gemmaConfig = LTX2GemmaConfig(
      vocabSize: 262208, hiddenSize: 3840,
      numHiddenLayers: 48, numAttentionHeads: 16,
      numKeyValueHeads: 8, headDim: 256,
      intermediateSize: 15360,
      rmsNormEps: 1e-6, ropeTheta: 1_000_000.0,
      slidingWindow: 1024, slidingWindowPattern: 6,
      quantization: nil
    )
    let encoderConfig = LTX2TextEncoderConfig(
      gemma: gemmaConfig,
      hasPromptAdaLN: true
    )
    let textEncoder = LTX2TextEncoder(config: encoderConfig)
    try textEncoder.loadWeights(
      modelPath: URL(fileURLWithPath: modelDir),
      textEncoderPath: URL(fileURLWithPath: gemmaPath)
    )
    MLX.eval(textEncoder.parameters())

    print("[5] Creating pipeline...")
    let pipelineConfig = LTX2PipelineConfig(
      modelPath: modelDir, pipelineType: .distilled, hasPromptAdaLN: true
    )
    let pipeline = LTX2Pipeline(
      vae: vae, textEncoder: textEncoder, transformer: transformer, config: pipelineConfig
    )

    print("[6] Tokenizing prompt...")
    let tokenizerDir = URL(fileURLWithPath: gemmaPath)
    let tokenizer = try LTX2GemmaTokenizer.load(from: tokenizerDir, maxLength: 128)
    let batch = tokenizer.encode(prompt: promptText, maxLength: 128)
    MLX.eval(batch.inputIds, batch.attentionMask)
    print("  Tokens: \(MLX.sum(batch.attentionMask).item(Int.self))")

    #if canImport(CoreGraphics) && canImport(ImageIO)
    print("[7] Loading init image: \(initImagePath)")
    let imgURL = URL(fileURLWithPath: initImagePath)
    guard let imgSource = CGImageSourceCreateWithURL(imgURL as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(imgSource, 0, nil) else {
      throw NSError(domain: "LTX2I2V", code: 20,
        userInfo: [NSLocalizedDescriptionKey: "Failed to load image from \(initImagePath)"])
    }
    let pixelArray = try QwenImageIO.resizedPixelArray(
      from: cgImage, width: width, height: height,
      addBatchDimension: true, dtype: .float32
    )
    var currentImage = QwenImageIO.normalizeForEncoder(pixelArray)
    print("  Image loaded: \(currentImage.shape)")

    var allFrames: [CGImage] = []
    let overallStart = CFAbsoluteTimeGetCurrent()

    for chunk in 0..<totalChunks {
      let chunkSeed = UInt64(seed + chunk)
      print()
      print("--- Chunk \(chunk + 1)/\(totalChunks) (seed \(chunkSeed)) ---")

      let genStart = CFAbsoluteTimeGetCurrent()
      let output = try pipeline.generateI2V(
        inputIds: batch.inputIds,
        attentionMask: batch.attentionMask,
        image: currentImage,
        strength: strength,
        width: width, height: height,
        numFrames: framesPerChunk,
        steps: steps, seed: chunkSeed,
        progressCallback: { step, total in
          let elapsed = CFAbsoluteTimeGetCurrent() - genStart
          let rate = step > 0 ? elapsed / Double(step) : 0
          let remaining = rate > 0 ? rate * Double(total - step) : 0
          print(String(format: "  [step %d/%d]  %.1fs elapsed  ~%.0fs remaining",
            step, total, elapsed, remaining))
        }
      )
      print("  Chunk generated in \(String(format: "%.1f", output.elapsedSeconds))s")

      let chunkFrames = LTX2PostProcess.framesToImages(from: output.decoded)
      print("  Extracted \(chunkFrames.count) frames")

      if chunk == 0 {
        allFrames.append(contentsOf: chunkFrames)
      } else {
        allFrames.append(contentsOf: chunkFrames.dropFirst())
      }
      print("  Total frames so far: \(allFrames.count)")

      if chunk < totalChunks - 1 {
        let t = output.decoded.dim(2)
        let lastFramePixels = output.decoded[0..., 0..., (t-1)..<t, 0..., 0...]
        let lastFrame4D = lastFramePixels.squeezed(axis: 2)
        currentImage = lastFrame4D * 2.0 - 1.0
        MLX.eval(currentImage)
        print("  Last frame extracted for next chunk")
      }
    }

    let overallTime = CFAbsoluteTimeGetCurrent() - overallStart
    print()
    print("=== All chunks complete ===")
    print("Total frames: \(allFrames.count) (\(String(format: "%.1f", Float(allFrames.count) / Float(fps)))s at \(fps)fps)")
    print("Total time:   \(String(format: "%.1f", overallTime))s")

    print("Writing MP4 to \(outputPath)...")
    try LTX2PostProcess.writeMP4(
      frames: allFrames,
      outputPath: outputPath,
      fps: fps,
      width: width,
      height: height
    )

    if let attrs = try? FileManager.default.attributesOfItem(atPath: outputPath),
       let size = attrs[FileAttributeKey.size] as? Int {
      let mb = Double(size) / (1024.0 * 1024.0)
      print("Output: \(outputPath) (\(String(format: "%.1f", mb)) MB)")
    }
    print("Done!")
    #else
    print("ERROR: CoreGraphics/ImageIO required for i2v (macOS only)")
    #endif
  }


  // MARK: - Telegram Bot Subcommand

  /// `ComfyBox krea2 -p "prompt" [-W 1024] [-H 1024] [-s 9] [--seed N] [-q 8]
  ///  [--model-dir path] [-o out.png]` — Krea-2-Turbo text-to-image (native port).
  private static func runKrea2(args: [String]) throws {
    var prompt = "a photo of a fox in a snowy forest, golden hour"
    var width = 1024
    var height = 1024
    var steps = 9
    var seed: UInt64 = 0
    var outputPath = "krea2.png"
    var modelDir: String?
    var quantBits: Int?

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--prompt", "-p": prompt = nextValue(for: arg, iterator: &iterator) ?? prompt
      case "--width", "-W": width = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: width)
      case "--height", "-H": height = intValue(for: arg, iterator: &iterator, minimum: 64, fallback: height)
      case "--steps", "-s": steps = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: steps)
      case "--seed": seed = UInt64(nextValue(for: arg, iterator: &iterator) ?? "0") ?? 0
      case "--output", "-o": outputPath = nextValue(for: arg, iterator: &iterator) ?? outputPath
      case "--model-dir": modelDir = nextValue(for: arg, iterator: &iterator)
      case "--quantize", "-q": quantBits = intValue(for: arg, iterator: &iterator, minimum: 2, fallback: 8)
      default: logger.warning("krea2: unknown argument \(arg)")
      }
    }

    let paths = try Krea2ModelPaths.resolve(modelDir: modelDir)
    logger.info("krea2: model root \(paths.root.path) variant=\(paths.variant.rawValue) transformer=\(paths.transformerFile.lastPathComponent)")

    let loadStart = Date()
    let pipeline = try Krea2Pipeline(paths: paths, quantizeTransformer: quantBits)
    logger.info("krea2: models loaded in \(String(format: "%.1f", Date().timeIntervalSince(loadStart)))s")

    let genStart = Date()
    let image = try pipeline.generate(
      .init(prompt: prompt, width: width, height: height, steps: steps, seed: seed)
    ) { step, total in
      logger.info("krea2: step \(step)/\(total)")
    }
    logger.info("krea2: generated in \(String(format: "%.1f", Date().timeIntervalSince(genStart)))s")

    // saveImage expects [3,H,W]; pipeline returns (H,W,3).
    let chw = image.transposed(2, 0, 1)
    try QwenImageIO.saveImage(array: chw, to: URL(fileURLWithPath: outputPath))
    logger.info("krea2: saved \(outputPath)")
  }

  private static func runTelegram(args: [String]) throws {
    var botToken: String? = nil
    var configPath: String? = nil
    var warmServerPort: UInt16 = 7870
    var warmServerHost: String = "127.0.0.1"
    var enhanceOverride: Bool? = nil

    var iterator = args.makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--bot-token":
        botToken = nextValue(for: arg, iterator: &iterator)
      case "--config":
        configPath = nextValue(for: arg, iterator: &iterator)
      case "--port":
        let raw = intValue(for: arg, iterator: &iterator, minimum: 1, fallback: 7870)
        warmServerPort = UInt16(min(raw, Int(UInt16.max)))
      case "--host":
        warmServerHost = nextValue(for: arg, iterator: &iterator)
      case "--enhance":
        let next = iterator.next()
        if let next = next?.lowercased() {
          switch next {
          case "on", "true", "yes": enhanceOverride = true
          case "off", "false", "no": enhanceOverride = false
          default:
            logger.warning("Unknown --enhance value: \(next), expected on/off")
            enhanceOverride = true
          }
        } else {
          enhanceOverride = true  // --enhance with no value means on
        }
      case "--no-enhance":
        enhanceOverride = false
      case "--help", "-h":
        printTelegramUsage()
        return
      default:
        logger.warning("Unknown telegram argument: \(arg)")
      }
    }

    // Load config
    let config = try loadTelegramConfig(
      botToken: botToken,
      configPath: configPath,
      warmServerHost: warmServerHost,
      warmServerPort: warmServerPort,
      enhanceOverride: enhanceOverride
    )

    let coordinator = ImageBotCoordinator(configuration: config, logger: logger)

    // Signal handling (SIGINT/SIGTERM)
    let signalSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
    signalSource.setEventHandler { coordinator.shutdown() }
    signalSource.resume()
    signal(SIGINT, SIG_IGN)

    let termSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
    termSource.setEventHandler { coordinator.shutdown() }
    termSource.resume()
    signal(SIGTERM, SIG_IGN)

    // Keep Mac awake while bot is running
    let caffeinate = Process()
    caffeinate.executableURL = URL(fileURLWithPath: "/usr/bin/caffeinate")
    caffeinate.arguments = ["-s", "-w", "\(ProcessInfo.processInfo.processIdentifier)"]
    try? caffeinate.run()

    logger.info("ComfyBox Telegram bot starting...")

    Task {
      do {
        try await coordinator.run()
        Darwin.exit(0)
      } catch {
        fputs("Telegram bot error: \(error.localizedDescription)\n", stderr)
        Darwin.exit(1)
      }
    }

    dispatchMain()
  }

  private static func loadTelegramConfig(
    botToken: String?,
    configPath: String?,
    warmServerHost: String,
    warmServerPort: UInt16,
    enhanceOverride: Bool?
  ) throws -> ImageBotCoordinator.Configuration {
    // Resolution: CLI > env > config file > defaults
    var token = botToken
    var allowedUserIds: Set<Int> = [8754779862]  // Todd
    var host = warmServerHost
    var port = warmServerPort
    var outputDir = ("~/Pictures/ComfyBox/Telegram" as NSString).expandingTildeInPath
    var galleryDir: String? = nil
    var characterConfigPath: String? = nil
    var contentModeConfigPath: String? = nil

    // Optimizer config (defaults)
    var optimizerEnabled = true
    var ollamaBaseURL = "http://localhost:11434"
    var lmStudioBaseURL: String? = "http://localhost:1234"
    var optimizerModel = "qwen3:8b"
    var optimizerTimeout = 15

    // Check environment variable
    if token == nil {
      token = ProcessInfo.processInfo.environment["COMFYBOX_TELEGRAM_TOKEN"]
    }

    // Load config file
    let configFilePath = configPath ?? (("~/.comfybox/telegram.json" as NSString).expandingTildeInPath)
    if FileManager.default.fileExists(atPath: configFilePath),
       let data = FileManager.default.contents(atPath: configFilePath),
       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {

      if token == nil, let fileToken = json["botToken"] as? String {
        token = fileToken
      }
      if let userIds = json["allowedUserIds"] as? [Int] {
        allowedUserIds = Set(userIds)
      }
      if let ws = json["warmServer"] as? [String: Any] {
        // Only use config file values if not overridden by CLI
        if warmServerHost == "127.0.0.1", let h = ws["host"] as? String {
          host = h
        }
        if warmServerPort == 7870, let p = ws["port"] as? Int {
          port = UInt16(min(p, Int(UInt16.max)))
        }
      }
      if let dir = json["outputDirectory"] as? String {
        outputDir = (dir as NSString).expandingTildeInPath
      }
      if let dir = json["galleryDirectory"] as? String {
        galleryDir = (dir as NSString).expandingTildeInPath
      }
      if let charPath = json["characterConfigPath"] as? String {
        characterConfigPath = (charPath as NSString).expandingTildeInPath
      }

      // Optimizer config from file
      if let opt = json["optimizer"] as? [String: Any] {
        if let enabled = opt["enabled"] as? Bool {
          optimizerEnabled = enabled
        }
        if let url = opt["ollamaBaseURL"] as? String {
          ollamaBaseURL = url
        }
        if let url = opt["lmStudioBaseURL"] as? String {
          lmStudioBaseURL = url
        }
        if let model = opt["model"] as? String {
          optimizerModel = model
        }
        if let timeout = opt["timeoutSeconds"] as? Int {
          optimizerTimeout = timeout
        }
      }
    }

    // CLI --enhance/--no-enhance overrides config file
    if let enhanceOverride = enhanceOverride {
      optimizerEnabled = enhanceOverride
    }

    guard let finalToken = token, !finalToken.isEmpty else {
      throw NSError(
        domain: "ComfyBox.Telegram",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey:
          "Bot token required. Use --bot-token, COMFYBOX_TELEGRAM_TOKEN env, or config file."]
      )
    }

    let telegramConfig = TelegramBot.Configuration(
      botToken: finalToken,
      allowedUserIds: allowedUserIds
    )

    let optimizerConfig = PromptOptimizer.Configuration(
      ollamaBaseURL: ollamaBaseURL,
      lmStudioBaseURL: lmStudioBaseURL,
      model: optimizerModel,
      timeoutSeconds: optimizerTimeout,
      enabled: optimizerEnabled
    )

    return ImageBotCoordinator.Configuration(
      telegram: telegramConfig,
      warmServerHost: host,
      warmServerPort: port,
      outputDirectory: outputDir,
      galleryDirectory: galleryDir,
      optimizer: optimizerConfig,
      characterConfigPath: characterConfigPath,
      contentModeConfigPath: contentModeConfigPath
    )
  }

  private static func printTelegramUsage() {
    print("""
    Start Telegram bot surface for ComfyBox.
    Connects to a running WarmServer for image generation.

    Usage: ComfyBox telegram [options]

    Options:
      --bot-token <token>     Telegram Bot API token (or COMFYBOX_TELEGRAM_TOKEN env)
      --config <path>         Config file (default: ~/.comfybox/telegram.json)
      --port <port>           WarmServer port (default: 7870)
      --host <host>           WarmServer host (default: 127.0.0.1)
      --enhance [on|off]      Enable/disable prompt optimization (default: on)
      --no-enhance            Disable prompt optimization
      --help, -h              Show help

    Content Modes (in-bot commands):
      /neutral or /apple      SFW mode
      /banana                 Suggestive mode
      /avocado                Explicit mode

    Config file (~/.comfybox/telegram.json):
      {
        "botToken": "123456:ABC...",
        "allowedUserIds": [8754779862],
        "warmServer": { "host": "127.0.0.1", "port": 7870 },
        "optimizer": {
          "enabled": true,
          "ollamaBaseURL": "http://localhost:11434",
          "lmStudioBaseURL": "http://localhost:1234",
          "model": "qwen3:8b",
          "timeoutSeconds": 15
        },
        "characterConfigPath": "~/.comfybox/characters.json",
        "outputDirectory": "~/Pictures/ComfyBox/Telegram"
      }

    Resolution order: CLI flags > env vars > config file > defaults

    Requires a running WarmServer: ComfyBox serve --port 7870
    """)
  }




}

// MARK: - Progress Helpers

private final class PlainProgress {
  static let shared = PlainProgress()
  private var lastPercent: Int = -1
  private var lastEmitTime: Date = .distantPast

  func report(completed: Int, total: Int) {
    guard total > 0 else { return }
    let now = Date()
    let percent = Int((Double(completed) / Double(total)) * 100.0)
    if percent != lastPercent || now.timeIntervalSince(lastEmitTime) >= 0.5 {
      FileHandle.standardError.write("Step \(completed)/\(total) (\(percent)%)\n".data(using: .utf8)!)
      lastPercent = percent
      lastEmitTime = now
    }
  }
}

private final class ProgressBar {
  private let total: Int
  private var lastStepTime: Date?
  private var postWarmupDurations: [Double] = []
  private let windowSize: Int = 5
  private var lastRenderedPercent: Int = -1
  private var isFinished: Bool = false

  init(total: Int) { self.total = max(1, total) }

  func update(completed: Int) {
    if isFinished { return }
    let now = Date()
    if let last = lastStepTime {
      let dt = now.timeIntervalSince(last)
      postWarmupDurations.append(dt)
      if postWarmupDurations.count > windowSize { postWarmupDurations.removeFirst() }
    }
    lastStepTime = now

    let percent = Int((Double(completed) / Double(total)) * 100.0)
    if percent == lastRenderedPercent { return }
    lastRenderedPercent = percent

    let remaining = max(0, total - completed)
    var etaSeconds: Double? = nil
    if !postWarmupDurations.isEmpty {
      let avg = postWarmupDurations.reduce(0, +) / Double(postWarmupDurations.count)
      etaSeconds = avg * Double(remaining)
    }

    let barWidth = 28
    let filled = Int((Double(completed) / Double(total)) * Double(barWidth))
    let lead = (completed < total) ? ">" : "="
    let tailCount = max(0, barWidth - max(1, filled))
    let bar = String(repeating: "=", count: max(0, filled - 1)) + lead + String(repeating: "-", count: tailCount)

    let etaStr: String
    if let eta = etaSeconds { etaStr = format(seconds: eta) } else { etaStr = "estimating..." }

    let prefix = "\r\u{001B}[2K"
    let line = String(format: "[%@] %3d%%  %d/%d  ETA %@", bar, percent, completed, total, etaStr)
    if let data = (prefix + line).data(using: .utf8) {
      FileHandle.standardError.write(data)
      fflush(stderr)
    }
  }

  func finish(forceNewline: Bool = true) {
    if isFinished { return }
    isFinished = true
    if let data = ("\r\u{001B}[2K").data(using: .utf8) {
      FileHandle.standardError.write(data)
    }
    if forceNewline, let nl = "\n".data(using: .utf8) {
      FileHandle.standardError.write(nl)
    }
    fflush(stderr)
  }

  private func format(seconds: Double) -> String {
    var s = Int(seconds.rounded())
    let h = s / 3600
    s %= 3600
    let m = s / 60
    s %= 60
    if h > 0 { return String(format: "%dh%02dm%02ds", h, m, s) }
    if m > 0 { return String(format: "%dm%02ds", m, s) }
    return String(format: "%ds", s)
  }



}

do {
  try ZImageCLI.run()
} catch {
  fputs("Error: \(error.localizedDescription)\n", stderr)
  exit(1)
}
