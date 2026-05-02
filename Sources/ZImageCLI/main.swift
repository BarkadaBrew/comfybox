import Foundation
import Dispatch
import Logging
import Metal
import MLX
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
        if let s = UInt64(nextValue(for: arg, iterator: &iterator)) {
          seeds.append(s)
        }
      case "--output", "-o":
        outputPath = nextValue(for: arg, iterator: &iterator)
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
        loraScaleOverrides.append(contentsOf: splitCommaSeparated(nextValue(for: arg, iterator: &iterator)).compactMap(Float.init))
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
          fatalError("Unknown scheduler '\(raw)'. Valid: \(valid)")
        }
        schedulerKind = kind
      case "--sigma-schedule":
        let raw = nextValue(for: arg, iterator: &iterator)
        guard let kind = SigmaScheduleKind(rawValue: raw) else {
          let valid = SigmaScheduleKind.allCases.map(\.rawValue).joined(separator: ", ")
          fatalError("Unknown sigma schedule '\(raw)'. Valid: \(valid)")
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
          fatalError("Unknown DyPE method '\(raw)'. Valid: ntk, yarn, none")
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

    guard let prompt else {
      printUsage()
      return
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
                  variant: "turbo",
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
        }
        batchSemaphore.signal()
      }
      batchSemaphore.wait()

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
        if let bar { bar.finish(forceNewline: true) }
      }
      semaphore.signal()
    }
    semaphore.wait()
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

    Usage: ZImageCLI --prompt "text" [options]
      --prompt, -p           Text prompt (required)
      --negative-prompt      Negative prompt
      --width, -W            Output width (default \(ZImageModelMetadata.recommendedWidth))
      --height, -H           Output height (default \(ZImageModelMetadata.recommendedHeight))
      --steps, -s            Inference steps (default \(ZImageModelMetadata.recommendedInferenceSteps))
      --guidance, -g         Guidance scale (default \(ZImageModelMetadata.recommendedGuidanceScale))
      --seed                 Random seed
      --output, -o           Output path (default z-image.png)
      --model, -m            Model path or HuggingFace ID (default: \(ZImageRepository.id))
      --text-encoder-path    Override text encoder directory (CLI > ZIMAGE_ENCODER_PATH > auto-detect > default)
      --force-transformer-override-only  Treat a local .safetensors as transformer-only override (disable AIO auto-detect)
      --cache-limit          GPU memory cache limit in MB (default: unlimited)
      --max-sequence-length  Maximum sequence length for text encoding (default: 512)
      --lora, -l             LoRA path or HuggingFace ID (repeatable, prefer path=scale; path:scale is legacy)
      --lora-scale           LoRA scale factor override for the next unmatched --lora (repeatable)
      --lora-paths           Comma-separated LoRA paths or HuggingFace IDs (quoted commas unsupported)
      --lora-scales          Comma-separated LoRA scale overrides (default: 1.0)
      --scheduler, --sampler  Sampler algorithm: euler, heun, dpmpp-2m, dpmpp-2s-a, deis, ddim (default: euler)
      --sigma-schedule       Sigma schedule: flow, karras, exponential, beta (default: flow)
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
        Use 'ZImageCLI control --help' for full options

      serve                  Start warm HTTP server
        --model, -m          Model path or HuggingFace ID
        --text-encoder-path  Override text encoder directory
        --port               HTTP port (default 7862)
        --lora, -l           Initial LoRA(s)
        Use 'ZImageCLI serve --help' for full options

      upscale                Upscale image via SeedVR2
        --input, -i          Input image path (required)
        --output, -o         Output image path (default: input-upscaled.png)
        --resolution, -r     Target resolution (default: 2048)
        --steps              Inference steps (default: 1)
        --seed               Random seed
        --weights, -w        Path to SeedVR2 model weights directory
        --softness           Preprocessing softness 0.0-1.0 (default: 0.0)

    Examples:
      ZImageCLI -p "a cute cat" -o cat.png
      ZImageCLI -p "a sunset" -m models/z-image-turbo-q8
      ZImageCLI -p "a forest" -m Tongyi-MAI/Z-Image-Turbo
      ZImageCLI -p "a cut a cat" --lora ostris/z_image_turbo_childrens_drawings
      ZImageCLI -p "portrait" --lora mood.safetensors=0.8 --lora detail.safetensors --lora-scale 0.3
      ZImageCLI -p "cat" --enhance  # Enhanced prompt generation
      ZImageCLI -p "portrait" --scheduler dpmpp-2m --sigma-schedule beta  # Best photorealism combo
      ZImageCLI -p "landscape" --scheduler heun --sigma-schedule beta -s 5  # Heun at half steps
      ZImageCLI -p "scene" --scheduler ddim --eta 0.5  # Semi-stochastic DDIM
      ZImageCLI serve -m ./models/z-image-turbo --port 7862
      ZImageCLI -p "portrait" --auto-seeds 5 -o portraits.png  # Generate 5 random variations
      ZImageCLI -p "cat" --seed 42 --seed 99 --seed 123 -o cats.png  # 3 specific seeds
      ZImageCLI -p "scene" --auto-seeds 10 --resume-batch progress.jsonl  # Resume interrupted batch
      ZImageCLI upscale -i photo.jpg -w ./models/seedvr2 -r 2048
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

    Usage: ZImageCLI quantize -i <input> -o <output> [options]
      --input, -i          Input model directory (required)
      --output, -o         Output directory (required)
      --bits               Bit width: 4 or 8 (default: 8)
      --group-size         Group size: 32, 64, 128 (default: 32)
      --verbose            Show progress
      --help, -h           Show help

    Example:
      ZImageCLI quantize -i models/z-image-turbo -o models/z-image-turbo-q8 --verbose
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

    Usage: ZImageCLI quantize-controlnet -i <input> -o <output> [options]
      --input, -i          Input ControlNet path or HuggingFace ID (required)
      --output, -o         Output directory (required)
      --file, -f           Specific .safetensors file to quantize (optional)
      --bits               Bit width: 4 or 8 (default: 8)
      --group-size         Group size: 32, 64, 128 (default: 32)
      --verbose            Show progress
      --help, -h           Show help

    Examples:
      # From HuggingFace
      ZImageCLI quantize-controlnet -i alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
        --file Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors -o controlnet-2.1-q8 --verbose

      # From local directory
      ZImageCLI quantize-controlnet -i ./controlnet-union -o ./controlnet-union-q8 --verbose
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
        loraScaleOverrides.append(contentsOf: splitCommaSeparated(nextValue(for: arg, iterator: &iterator)).compactMap(Float.init))
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
      maxPendingRequests: 10
    )

    let server = WarmServer(configuration: configuration, logger: logger)
    try server.run()
  }

  private static func printServeUsage() {
    print("""
    Start warm HTTP server mode.

    Usage: ZImageCLI serve [options]
      --model, -m               Model path or HuggingFace ID (default: \(ZImageRepository.id))
      --text-encoder-path       Override text encoder directory
      --port                    HTTP port (default: 7862)
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
      ZImageCLI serve -m /path/to/model --text-encoder-path /path/to/encoder --port 7862 \\
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
        seed = UInt64(nextValue(for: arg, iterator: &iterator))
      case "--output", "-o":
        outputPath = nextValue(for: arg, iterator: &iterator)
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
        loraScaleOverrides.append(contentsOf: splitCommaSeparated(nextValue(for: arg, iterator: &iterator)).compactMap(Float.init))
      case "--no-progress":
        noProgress = true
      case "--scheduler", "--sampler":
        let raw = nextValue(for: arg, iterator: &iterator)
        guard let kind = SchedulerKind(rawValue: raw) else {
          let valid = SchedulerKind.allCases.map(\.rawValue).joined(separator: ", ")
          fatalError("Unknown scheduler '\(raw)'. Valid: \(valid)")
        }
        schedulerKind = kind
      case "--sigma-schedule":
        let raw = nextValue(for: arg, iterator: &iterator)
        guard let kind = SigmaScheduleKind(rawValue: raw) else {
          let valid = SigmaScheduleKind.allCases.map(\.rawValue).joined(separator: ", ")
          fatalError("Unknown sigma schedule '\(raw)'. Valid: \(valid)")
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
    Task {
      do {
        _ = try await pipeline.generate(request)
        if let bar = barBox.value { bar.finish(forceNewline: true) }
        logger.info("Output saved to: \(finalOutputPath)")
      } catch {
        logger.error("Control generation failed: \(error)")
        if let bar = barBox.value { bar.finish(forceNewline: true) }
      }
      semaphore.signal()
    }
    semaphore.wait()
  }

  private static func printControlUsage() {
    print("""
    Generate images with ControlNet conditioning (supports v2.0/v2.1 with inpainting).

    Usage: ZImageCLI control --prompt "text" --controlnet-weights <path> [options]
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
      --model, -m               Model path or HuggingFace ID (default: \(ZImageRepository.id))
      --text-encoder-path       Override text encoder directory (CLI > ZIMAGE_ENCODER_PATH > auto-detect > default)
      --cache-limit             GPU memory cache limit in MB (default: unlimited)
      --max-sequence-length     Maximum sequence length for text encoding (default: 512)
      --lora, -l                LoRA path or HuggingFace ID (repeatable, prefer path=scale; path:scale is legacy)
      --lora-scale              LoRA scale factor override for the next unmatched --lora (repeatable)
      --lora-paths              Comma-separated LoRA paths or HuggingFace IDs (quoted commas unsupported)
      --lora-scales             Comma-separated LoRA scale overrides (default: 1.0)
      --scheduler, --sampler    Sampler algorithm: euler, heun, dpmpp-2m, dpmpp-2s-a, deis, ddim (default: euler)
      --sigma-schedule          Sigma schedule: flow, karras, exponential, beta (default: flow)
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
      ZImageCLI control -p "a woman on a beach" -c pose.jpg \\
        --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
        --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors

      # I2I inpainting with pose control
      ZImageCLI control -p "a dancer" -c pose.jpg -i photo.jpg --mask mask.png \\
        --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
        --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors --cs 0.75 -s 25

      # Inpainting without control guidance
      ZImageCLI control -p "a cat sitting" -i photo.jpg --mask mask.png \\
        --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \\
        --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors

      # Using local controlnet weights
      ZImageCLI control -p "a forest path" -c depth.jpg --cs 0.7 \\
        --cw ./controlnet-q8 -o forest.png

      # Custom encoder with stacked LoRAs
      ZImageCLI control -p "portrait" -c pose.png --cw ./controlnet-q8 \\
        -m /path/to/z-image-turbo-bf16 --text-encoder-path "/path/to/z-image-turbo-bf16/text_encoder QWen Large" \\
        --lora mood.safetensors=0.8 --lora detail.safetensors --lora-scale 0.3
    """)
  }

  private static func nextValue(for arg: String, iterator: inout IndexingIterator<[String]>) -> String {
    guard let value = iterator.next() else {
      fatalError("Expected value after \(arg)")
    }
    return value
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
    guard let value = Int(nextValue(for: arg, iterator: &iterator)) else { return fallback }
    return max(minimum, value)
  }

  private static func floatValue(for arg: String, iterator: inout IndexingIterator<[String]>, fallback: Float) -> Float {
    Float(nextValue(for: arg, iterator: &iterator)) ?? fallback
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
        if let s = Int(nextValue(for: arg, iterator: &iterator)) {
          seed = s
        }
      case "--weights", "-w":
        weightsPath = nextValue(for: arg, iterator: &iterator)
      case "--softness":
        softness = floatValue(for: arg, iterator: &iterator, fallback: softness)
      case "--help", "-h":
        print("""
        SeedVR2 Image Upscaler

        Usage: ZImageCLI upscale --input <path> --weights <path> [options]

          --input, -i          Input image path (required)
          --output, -o         Output image path (default: input-upscaled.png)
          --resolution, -r     Target resolution for shortest side (default: 2048)
          --steps              Inference steps (default: 1)
          --seed               Random seed for reproducibility
          --weights, -w        Path to SeedVR2 model weights directory (required)
          --softness           Preprocessing softness 0.0-1.0 (default: 0.0)
          --help, -h           Show this help

        Examples:
          ZImageCLI upscale -i photo.jpg -w ./models/seedvr2
          ZImageCLI upscale -i photo.jpg -w ./models/seedvr2 -r 4096 --seed 42
          ZImageCLI upscale -i low-res.png -w ./models/seedvr2 --softness 0.3 -o high-res.png
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

    guard let weights = weightsPath else {
      fputs("Error: --weights is required for upscale\n", stderr)
      exit(1)
    }

    guard FileManager.default.fileExists(atPath: input) else {
      fputs("Error: Input file not found: \(input)\n", stderr)
      exit(1)
    }

    let startTime = CFAbsoluteTimeGetCurrent()

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
    fputs("Error: SeedVR2 upscale requires CoreGraphics (macOS)\n", stderr)
    exit(1)
  }
  #endif

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
