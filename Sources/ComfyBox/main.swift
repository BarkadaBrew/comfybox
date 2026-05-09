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
      case "control":
        try runControl(args: Array(args.dropFirst()))
        return
      case "serve":
        try runServe(args: Array(args.dropFirst()))
        return
      case "upscale":
        try runUpscale(args: Array(args.dropFirst()))
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
      case "ltx2-demo":
        try runLTX2Demo(args: Array(args.dropFirst()))
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
      return
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
                variant: "turbo",
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
      nonisolated(unsafe) let chromaSemaphore = DispatchSemaphore(value: 0)
      let capturedModel = model
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

      Task {
        do {
          // Resolve model snapshot
          let modelSpec = capturedModel ?? "jack813liu/mlx-chroma"
          let snapshot = try await ModelResolution.resolve(modelSpec: modelSpec)

          // Detect model config from snapshot
          guard let detected = ChromaModelDetection.detect(at: snapshot) else {
            logger.error("Model at \(snapshot.path) is not a Chroma model")
            chromaSemaphore.signal()
            return
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
          let negTokenIds = tokenizer.encodeUnpadded(prompt: "")

          // Chroma defaults: 28 steps, guidance 0.0 (distilled), cfg 4.0
          // Flash-heun: 8 steps, heun/beta scheduler, CFG 1.0
          let chromaGuidance = guidance == ZImageModelMetadata.recommendedGuidanceScale ? Float(0.0) : guidance
          let chromaCFG: Float = chromaScheduler == .euler ? 4.0 : 1.0

          logger.info("Chroma: \(chromaSteps) steps, scheduler=\(chromaScheduler.rawValue), guidance=\(chromaGuidance), cfg=\(chromaCFG)")

          // Generate image
          let pixels = pipeline.generate(
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
          if let bar { bar.finish(forceNewline: true) }
        }
        chromaSemaphore.signal()
      }
      chromaSemaphore.wait()
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
      Task {
        do {
          // Resolve model snapshot
          let modelSpec = capturedModel ?? "briaai/FIBO"
          let snapshot = try await ModelResolution.resolve(modelSpec: modelSpec)

          // Detect model config from snapshot
          guard let detected = FiboModelDetection.detect(at: snapshot) else {
            logger.error("Model at \(snapshot.path) is not a FIBO model")
            fiboSemaphore.signal()
            return
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
          if let bar { bar.finish(forceNewline: true) }
        }
        fiboSemaphore.signal()
      }
      fiboSemaphore.wait()
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
      let capturedLoraEntries2 = loraEntries
      let capturedLoraScales2 = loraScaleOverrides
      let useBar = !noProgress && (isatty(STDERR_FILENO) != 0)
      let bar = useBar ? ProgressBar(total: steps) : nil
      Task {
        do {
          // Resolve model snapshot
          let modelSpec = capturedModel ?? "black-forest-labs/FLUX.2-klein-4B"
          let snapshot = try await ModelResolution.resolve(modelSpec: modelSpec)

          // Detect model config from snapshot
          guard let detected = Flux2ModelDetection.detect(at: snapshot) else {
            logger.error("Model at \(snapshot.path) is not a Flux 2 Klein model")
            flux2Semaphore.signal()
            return
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

          // Apply LoRAs to Flux 2 transformer
          if !capturedLoraEntries2.isEmpty {
            logger.info("Loading \(capturedLoraEntries2.count) LoRA(s) for Flux 2")
            for (idx, loraEntry) in capturedLoraEntries2.enumerated() {
              let loraScale = idx < capturedLoraScales2.count ? capturedLoraScales2[idx] : 1.0

              // Resolve LoRA path (local file or HuggingFace)
              let loraPath: String
              if FileManager.default.fileExists(atPath: loraEntry) {
                loraPath = loraEntry
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
                loraPath = safetensors.path
              }

              try flux2Pipeline.applyLoRA(path: loraPath, scale: loraScale)
            }
          }

          // Resolve img2img for Flux 2
          let flux2InputImage: URL? = initImagePath.map { URL(fileURLWithPath: $0) }
          let flux2DenoiseValue: Float = flux2InputImage != nil ? (flux2Denoise ?? 0.7) : 1.0

          if let img = flux2InputImage {
            guard FileManager.default.fileExists(atPath: img.path) else {
              logger.error("Init image not found: \(img.path)")
              flux2Semaphore.signal()
              return
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
          if let bar { bar.finish(forceNewline: true) }
        }
        flux2Semaphore.signal()
      }
      flux2Semaphore.wait()
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
              variant: "turbo",
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
    let vtracerPath = FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent(".cargo/bin/vtracer").path
    guard FileManager.default.fileExists(atPath: vtracerPath) else {
      throw NSError(domain: "ZImageCLI", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "vtracer not found. Install with: cargo install vtracer"])
    }
    var args = ["--input", input.path, "--output", output.path]
    switch preset {
    case "logo":
      args += ["--colormode", "color", "--hierarchical", "cutout", "--mode", "polygon",
               "-f", "10", "-p", "3", "-g", "48", "-c", "120", "-l", "8", "-s", "90", "--path_precision", "2"]
    case "detailed":
      args += ["--colormode", "color", "--hierarchical", "stacked", "--mode", "spline",
               "-f", "2", "-p", "8", "-g", "0", "-c", "45", "-l", "4", "-s", "60", "--path_precision", "8"]
    case "simplified":
      args += ["--colormode", "color", "--hierarchical", "stacked", "--mode", "polygon",
               "-f", "6", "-p", "5", "-g", "16", "-c", "90", "-l", "6", "-s", "75", "--path_precision", "3"]
    case "bw":
      args += ["--colormode", "binary", "--hierarchical", "stacked", "--mode", "spline",
               "-f", "4", "-p", "6", "-g", "0", "-c", "60", "-l", "4", "-s", "60", "--path_precision", "5"]
    default:
      args += ["--colormode", "color", "--hierarchical", "stacked", "--mode", "spline",
               "-f", "4", "-p", "6", "-g", "0", "-c", "60", "-l", "4", "-s", "60", "--path_precision", "5"]
    }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: vtracerPath)
    process.arguments = args
    let pipe = Pipe()
    process.standardError = pipe
    try process.run()
    process.waitUntilExit()
    if process.terminationStatus != 0 {
      let errorData = pipe.fileHandleForReading.readDataToEndOfFile()
      let errorMessage = String(data: errorData, encoding: .utf8) ?? "Unknown error"
      throw NSError(domain: "ZImageCLI", code: Int(process.terminationStatus),
        userInfo: [NSLocalizedDescriptionKey: "vtracer failed: \(errorMessage)"])
    }
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
        --port               HTTP port (default 7862)
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

      models                 List known model families with installation status
        --paths, -v          Show filesystem paths for installed models


      mcp                    Start MCP server (stdio JSON-RPC bridge to WarmServer)
        --port               WarmServer port (default: 7862)
        --host               WarmServer host (default: 127.0.0.1)
        Use 'ComfyBox mcp --help' for full options

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
      ComfyBox serve -m ./models/z-image-turbo --port 7862
      ComfyBox -p "portrait" --auto-seeds 5 -o portraits.png  # Generate 5 random variations
      ComfyBox -p "cat" --seed 42 --seed 99 --seed 123 -o cats.png  # 3 specific seeds
      ComfyBox -p "scene" --auto-seeds 10 --resume-batch progress.jsonl  # Resume interrupted batch
      ComfyBox upscale -i photo.jpg -w ./models/seedvr2 -r 2048
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
      return
    }

    guard let outputPath = output else {
      logger.error("Missing required --output argument")
      printQuantizeUsage()
      return
    }

    let inputURL = URL(fileURLWithPath: inputPath)
    let outputURL = URL(fileURLWithPath: outputPath)

    guard FileManager.default.fileExists(atPath: inputURL.path) else {
      logger.error("Input directory not found: \(inputPath)")
      return
    }

    guard ZImageQuantizer.supportedBits.contains(bits) else {
      logger.error("Invalid bits: \(bits). Supported: 4, 8")
      return
    }

    guard ZImageQuantizer.supportedGroupSizes.contains(groupSize) else {
      logger.error("Invalid group size: \(groupSize). Supported: 32, 64, 128")
      return
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
    var model: String?
    var textEncoderPath: String?
    var port: UInt16 = 7862
    var cacheLimit: Int?
    var maxSequenceLength = 512
    var loraEntries: [String] = []
    var loraScaleOverrides: [Float] = []
    var forceTransformerOverrideOnly = false
    var host = "127.0.0.1"
    var allowedOutputDirectory = FileManager.default.currentDirectoryPath

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
      case "--help", "-h":
        printServeUsage()
        return
      default:
        logger.warning("Unknown serve argument: \(arg)")
      }
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
      allowedOutputDirectory: allowedOutputDirectory
    )

    let server = WarmServer(configuration: configuration, host: host, logger: logger)
    try server.run()
  }

  private static func printServeUsage() {
    print("""
    Start warm HTTP server mode.

    Usage: ComfyBox serve [options]
      --model, -m               Model path or HuggingFace ID (default: \(ZImageRepository.id))
      --text-encoder-path       Override text encoder directory
      --port                    HTTP port (default: 7862)
      --host                    HTTP host/interface to bind (default: 127.0.0.1)
      --allowed-output-directory  Directory where request outputPath values may write (default: current directory)
      --cache-limit             GPU memory cache limit in MB (default: unlimited)
      --max-sequence-length     Maximum sequence length for text encoding (default: 512)
      --force-transformer-override-only  Treat a local .safetensors as transformer-only override
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

    Example:
      ComfyBox serve -m /path/to/model --text-encoder-path /path/to/encoder --port 7862 \\
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
      return
    }
    if controlImage == nil && inpaintImage == nil && maskImage == nil {
      logger.error("At least one of --control-image, --inpaint-image, or --mask must be provided")
      printControlUsage()
      return
    }

    guard let controlnetWeights else {
      logger.error("Missing required --controlnet-weights argument")
      printControlUsage()
      return
    }
    var controlImageURL: URL? = nil
    if let controlImage {
      controlImageURL = URL(fileURLWithPath: controlImage)
      guard FileManager.default.fileExists(atPath: controlImageURL!.path) else {
        logger.error("Control image not found: \(controlImage)")
        return
      }
    }
    var inpaintImageURL: URL? = nil
    if let inpaintImage {
      inpaintImageURL = URL(fileURLWithPath: inpaintImage)
      guard FileManager.default.fileExists(atPath: inpaintImageURL!.path) else {
        logger.error("Inpaint image not found: \(inpaintImage)")
        return
      }
    }
    var maskImageURL: URL? = nil
    if let maskImage {
      maskImageURL = URL(fileURLWithPath: maskImage)
      guard FileManager.default.fileExists(atPath: maskImageURL!.path) else {
        logger.error("Mask image not found: \(maskImage)")
        return
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
    var port: UInt16 = 7862
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

    // Print server info to stderr (stdout is reserved for JSON-RPC)
    fputs("ComfyBox MCP server v\(MCPServer.version)\n", stderr)
    fputs("Bridging to WarmServer at \(host):\(port)\n", stderr)

    let server = MCPServer(host: host, port: port)
    server.run()
  }

  private static func printMCPUsage() {
    print("""
    Start MCP (Model Context Protocol) server mode.
    Bridges stdio JSON-RPC 2.0 to WarmServer HTTP API.

    Usage: ComfyBox mcp [options]
      --port                    WarmServer port to connect to (default: 7862)
      --host                    WarmServer host to connect to (default: 127.0.0.1)
      --help, -h                Show help

    The MCP server reads JSON-RPC requests from stdin and writes responses
    to stdout. All logging goes to stderr. Runs until stdin closes.

    Registration:
      claude mcp add comfybox -- comfybox mcp --port 7862

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



  // MARK: - LTX2 Demo

  private static func runLTX2Demo(args: [String]) throws {
    var modelDir = "/Volumes/Bolt/Models/ltx2-distilled"
    var outputPath = "/tmp/ltx2-demo.mp4"
    var width = 512
    var height = 320
    var frames = 9
    var steps = 4
    var seed = 42

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
      case "--help", "-h":
        printLTX2DemoUsage()
        return
      default:
        logger.warning("Unknown ltx2-demo argument: \(arg)")
      }
    }

    print("=== LTX2 Demo ===")
    print("Model dir:  \(modelDir)")
    print("Output:     \(outputPath)")
    print("Resolution: \(width)x\(height)")
    print("Frames:     \(frames)")
    print("Steps:      \(steps)")
    print("Seed:       \(seed)")
    print()

    // --- Create transformer ---
    print("[1/6] Creating transformer (48-layer, 32 heads)...")
    let transformer = LTX2Transformer(
      numHeads: 32, headDim: 128, inChannels: 128, outChannels: 128,
      numLayers: 48, crossAttentionDim: 4096, captionChannels: 3840,
      normEps: 1e-6, hasPromptAdaLN: true, timestepScaleMultiplier: 1000,
      positionalEmbeddingTheta: 10000, positionalEmbeddingMaxPos: [20, 2048, 2048],
      useMiddleIndicesGrid: true, ropeMode: .split
    )

    // --- Load transformer weights ---
    print("[2/6] Loading transformer weights (this takes 1-3 minutes for 35GB)...")
    let transformerPath = URL(fileURLWithPath: modelDir + "/transformer-distilled.safetensors")
    guard FileManager.default.fileExists(atPath: transformerPath.path) else {
      throw NSError(domain: "LTX2Demo", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "Transformer weights not found at \(transformerPath.path)"])
    }
    let startLoad = CFAbsoluteTimeGetCurrent()
    let rawWeights = try MLX.loadArrays(url: transformerPath)
    let sanitized = LTX2Transformer.sanitizeWeights(rawWeights)
    let params = ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) })
    try transformer.update(parameters: params, verify: [.shapeMismatch])
    MLX.eval(transformer.parameters())
    let loadTime = CFAbsoluteTimeGetCurrent() - startLoad
    print("  Loaded \(rawWeights.count) tensors in \(String(format: "%.1f", loadTime))s")

    // --- Create and load VAE ---
    print("[3/6] Creating and loading VAE...")
    let vae = LTX2VAE()

    // Load VAE weights from separate files.
    // The files use "vae_decoder.X"/"vae_encoder.X" prefix.
    // We rewrite to "vae.decoder.X"/"vae.encoder.X" so LTX2WeightLoader's
    // remapping logic handles the heterogeneous block arrays correctly.
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

    // Load and rewrite key prefixes
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
    // Add per_channel_statistics in the format the weight loader expects
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
    MLX.eval(vae.parameters())
    print("  VAE weights loaded successfully")

    // --- Create pipeline ---
    print("[4/6] Creating pipeline...")
    let pipelineConfig = LTX2PipelineConfig(
      modelPath: modelDir,
      pipelineType: .distilled,
      hasPromptAdaLN: true
    )

    // Dummy text encoder (we bypass it via embeddings)
    let textEncoder = LTX2TextEncoder(config: LTX2TextEncoderConfig())

    let pipeline = LTX2Pipeline(
      vae: vae,
      textEncoder: textEncoder,
      transformer: transformer,
      config: pipelineConfig
    )

    // --- Generate ---
    print("[5/6] Generating video with dummy embeddings...")
    print("  Using random embeddings (no text encoder) -- output will be abstract noise")
    MLXRandom.seed(UInt64(seed))
    let dummyEmbeddings = MLXRandom.normal([1, 32, 4096]) * Float(0.01)
    MLX.eval(dummyEmbeddings)

    let output = pipeline.generateT2VWithEmbeddings(
      videoEmbeddings: dummyEmbeddings,
      width: width, height: height, numFrames: frames,
      steps: steps, seed: UInt64(seed),
      progressCallback: { step, total in
        print("  Step \(step)/\(total)")
      }
    )

    print("  Generation complete in \(String(format: "%.1f", output.elapsedSeconds))s")
    print("  Output shape: \(output.decoded.shape)")

    // --- Write MP4 ---
    print("[6/6] Writing MP4 to \(outputPath)...")
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
    // Fallback: write PPM frames
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
    Run LTX-2 video generation demo with dummy embeddings.
    Proves the full pipeline works: weight loading -> transformer -> VAE decode -> MP4.

    Usage: ComfyBox ltx2-demo [options]
      --model-dir <path>        Model weights directory (default: /Volumes/Bolt/Models/ltx2-distilled)
      --output, -o <path>       Output MP4 path (default: /tmp/ltx2-demo.mp4)
      --width, -W <int>         Video width in pixels (default: 512)
      --height, -H <int>        Video height in pixels (default: 320)
      --frames <int>            Number of frames, must be 1+8k (default: 9)
      --steps <int>             Denoising steps (default: 4)
      --seed <int>              Random seed (default: 42)
      --help, -h                Show help

    The demo uses random embeddings instead of a real text encoder,
    so output will be abstract noise. This proves the pipeline works
    end-to-end without requiring Gemma 3 weights.
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
