// ImageToImagePipeline.swift — Img2img generation via the existing inpainting path.
//
// Flow: load source image -> create synthetic all-white mask -> delegate to
//       ZImagePipeline.generate with inpaint parameters -> output.
//
// Strength convention (matches Python mflux):
//   strength 0.3 -> heavy rework (more steps, default)
//   strength 0.7 -> light touch (fewer steps)
// Creativity is the inverse: creativity = 1.0 - strength

import Foundation
import Logging
import MLX

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

/// Configuration for an img2img generation request.
public struct Img2ImgRequest: Sendable {
  public var prompt: String
  public var negativePrompt: String?
  public var width: Int?
  public var height: Int?
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
  public var enhancePrompt: Bool
  public var enhanceMaxTokens: Int
  public var forceTransformerOverrideOnly: Bool
  public var schedulerKind: SchedulerKind
  public var sigmaSchedule: SigmaScheduleKind
  public var eta: Float?
  /// comfybox#154 — explicit `ModelSamplingAuraFlow` schedule shift; nil = the
  /// model's own schedule. Forwarded verbatim to ``ZImageGenerationRequest``.
  public var shift: Float?
  public var dyPE: DyPEConfig

  /// The source image file path.
  public var sourceImagePath: String

  /// Strength controls how many denoising steps to run (mflux convention).
  /// 0.3 = heavy rework (default), 0.7 = light touch.
  /// Range: (0.0, 1.0]. A value of 0.0 would mean no denoising at all.
  public var strength: Float

  /// How the user specified the value — "strength" or "creativity".
  public var specifiedAs: Img2ImgSpecifier

  /// Fruit mode (neutral|banana|avocado) — stamped into embedded metadata.
  public var contentMode: String?

  /// Submitting app/persona (desktop/bree/api…) — stamped as metadata provenance.
  public var source: String?

  /// Optional mask PNG path for selective inpainting. White (255) = inpaint/regenerate,
  /// black (0) = keep original. When nil, a full-white mask is used (standard img2img,
  /// i.e. regenerate everything). Mask should match the (round-to-16) output dimensions.
  public var maskPath: String?

  /// Auto-generated mask region ("face" | "upper" | "lower") — mutually exclusive
  /// with maskPath. "face" runs Vision face detection on the source image; the
  /// bands split the frame at half height. The named region is REGENERATED.
  public var maskRegion: String?

  /// Flip the mask (white ⇄ black). With maskRegion "face" this means: lock the
  /// face, regenerate everything else. Requires maskPath or maskRegion.
  public var maskInvert: Bool

  /// Mask expansion in pixels before latent conversion (0 = none).
  public var maskGrow: Int

  /// Mask feather radius in pixels before latent conversion (0 = hard edges).
  public var maskFeather: Int

  public enum Img2ImgSpecifier: String, Sendable {
    case strength
    case creativity
    case denoise
  }

  public init(
    prompt: String,
    negativePrompt: String? = nil,
    width: Int? = nil,
    height: Int? = nil,
    steps: Int = ZImageModelMetadata.recommendedInferenceSteps,
    guidanceScale: Float = ZImageModelMetadata.recommendedGuidanceScale,
    seed: UInt64? = nil,
    outputPath: URL = URL(fileURLWithPath: "z-image-img2img.png"),
    levelsMin: Float = 0.0,
    levelsMax: Float = 1.0,
    model: String? = nil,
    textEncoderPath: String? = nil,
    maxSequenceLength: Int = 512,
    loras: [LoRAConfiguration] = [],
    enhancePrompt: Bool = false,
    enhanceMaxTokens: Int = 512,
    forceTransformerOverrideOnly: Bool = false,
    schedulerKind: SchedulerKind = .euler,
    sigmaSchedule: SigmaScheduleKind = .flow,
    eta: Float? = nil,
    shift: Float? = nil,
    dyPE: DyPEConfig = .disabled,
    sourceImagePath: String,
    strength: Float = 0.3,
    specifiedAs: Img2ImgSpecifier = .strength,
    contentMode: String? = nil,
    source: String? = nil,
    maskPath: String? = nil,
    maskRegion: String? = nil,
    maskInvert: Bool = false,
    maskGrow: Int = 0,
    maskFeather: Int = 0
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
    self.textEncoderPath = textEncoderPath
    self.maxSequenceLength = maxSequenceLength
    self.loras = loras
    self.enhancePrompt = enhancePrompt
    self.enhanceMaxTokens = enhanceMaxTokens
    self.forceTransformerOverrideOnly = forceTransformerOverrideOnly
    self.schedulerKind = schedulerKind
    self.sigmaSchedule = sigmaSchedule
    self.eta = eta
    self.shift = shift
    self.dyPE = dyPE
    self.sourceImagePath = sourceImagePath
    self.strength = strength
    self.specifiedAs = specifiedAs
    self.contentMode = contentMode
    self.source = source
    self.maskPath = maskPath
    self.maskRegion = maskRegion
    self.maskInvert = maskInvert
    self.maskGrow = maskGrow
    self.maskFeather = maskFeather
  }

  /// Convert img2img strength to inpainting denoise value.
  ///
  /// mflux img2img: init_time_step = max(1, int(steps * strength))
  ///   strength 0.3, 9 steps -> init=2, run steps 2..8 = 7 steps (heavy)
  ///   strength 0.7, 9 steps -> init=6, run steps 6..8 = 3 steps (light)
  ///
  /// ZImage inpainting: inpaintStartStep = max(0, steps - Int(ceil(steps * denoise)))
  ///   denoise 0.7, 9 steps -> startStep = 9 - 7 = 2, run 7 steps (heavy)
  ///   denoise 0.3, 9 steps -> startStep = 9 - 3 = 6, run 3 steps (light)
  ///
  /// Conversion: denoise = 1.0 - strength
  public var denoise: Float {
    return 1.0 - max(0.01, min(0.99, strength))
  }
}

// MARK: - White Mask Generation

private enum Img2ImgUtilities {
  private static let pngSignature: [UInt8] = [137, 80, 78, 71, 13, 10, 26, 10]

  enum Img2ImgError: Error, CustomStringConvertible, LocalizedError {
    case sourceImageNotFound(String)
    case sourceImageLoadFailed(String)
    case maskGenerationFailed(String)
    case maskNotFound(String)
    case maskLoadFailed(String)

    var description: String {
      switch self {
      case .sourceImageNotFound(let path): return "Source image not found: \(path)"
      case .sourceImageLoadFailed(let msg): return "Source image load failed: \(msg)"
      case .maskGenerationFailed(let msg): return "Mask generation failed: \(msg)"
      case .maskNotFound(let path): return "Mask image not found: \(path)"
      case .maskLoadFailed(let msg): return "Mask image load failed: \(msg)"
      }
    }

    var errorDescription: String? { description }
  }

  /// Generate a solid white PNG image of the given dimensions.
  static func generateWhiteMaskPNG(width: Int, height: Int) throws -> Data {
    #if canImport(CoreGraphics)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo.rawValue
    ) else {
      throw Img2ImgError.maskGenerationFailed("Failed to create CGContext for white mask")
    }

    context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))

    guard let cgImage = context.makeImage() else {
      throw Img2ImgError.maskGenerationFailed("Failed to create CGImage from white-filled context")
    }

    let mutableData = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(mutableData, "public.png" as CFString, 1, nil) else {
      throw Img2ImgError.maskGenerationFailed("Failed to create PNG destination")
    }
    CGImageDestinationAddImage(destination, cgImage, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw Img2ImgError.maskGenerationFailed("Failed to finalize PNG data")
    }

    return mutableData as Data
    #else
    throw Img2ImgError.maskGenerationFailed("CoreGraphics not available")
    #endif
  }

  /// Read the dimensions of a PNG file from its IHDR chunk.
  static func pngDimensions(from data: Data) -> (width: Int, height: Int)? {
    guard data.count >= 24, data.prefix(pngSignature.count).elementsEqual(pngSignature) else { return nil }
    let w = Int(data[16]) << 24 | Int(data[17]) << 16 | Int(data[18]) << 8 | Int(data[19])
    let h = Int(data[20]) << 24 | Int(data[21]) << 16 | Int(data[22]) << 8 | Int(data[23])
    guard w > 0 && h > 0 && w < 65536 && h < 65536 else { return nil }
    return (w, h)
  }

  /// Round up to nearest multiple of 16 (VAE latent alignment).
  static func roundTo16(_ n: Int) -> Int {
    return ((n + 15) / 16) * 16
  }
}

// MARK: - Pipeline Extension

extension ZImagePipeline {

  /// Generate an image from a source image + prompt (img2img).
  ///
  /// Internally converts the img2img request to an inpainting request with
  /// a full white mask, delegating to the existing generateCore path.
  public func generateImg2Img(
    _ request: Img2ImgRequest,
    progressHandler: ProgressHandler? = nil
  ) async throws -> URL {
    let pipelineRequest = try makeImg2ImgPipelineRequest(request)
    return try await generate(pipelineRequest, progressHandler: progressHandler)
  }

  /// Generate an img2img result to memory (PNG data).
  public func generateImg2ImgToMemory(
    _ request: Img2ImgRequest,
    progressHandler: ProgressHandler? = nil
  ) async throws -> Data {
    let pipelineRequest = try makeImg2ImgPipelineRequest(request)
    return try await generateToMemory(pipelineRequest, progressHandler: progressHandler)
  }

  /// Convert an Img2ImgRequest to a ZImageGenerationRequest by loading
  /// the source image, generating a white mask, and computing denoise.
  ///
  /// Forwards to the static conversion below, which reads no instance state
  /// (only `request` + `FileManager`) and is exposed separately so it can be
  /// unit-tested without constructing a live `ZImagePipeline`.
  internal func makeImg2ImgPipelineRequest(_ request: Img2ImgRequest) throws -> ZImageGenerationRequest {
    try Self.makeImg2ImgPipelineRequest(request)
  }

  /// Test-only: exposes makeImg2ImgPipelineRequest's pure conversion without a
  /// live pipeline instance. Not used in production code paths.
  static func makeImg2ImgPipelineRequestForTesting(_ request: Img2ImgRequest) throws -> ZImageGenerationRequest {
    try makeImg2ImgPipelineRequest(request)
  }

  /// Pure conversion logic shared by the instance method and the test shim.
  static func makeImg2ImgPipelineRequest(_ request: Img2ImgRequest) throws -> ZImageGenerationRequest {
    let sourceURL = URL(fileURLWithPath: request.sourceImagePath)
    guard FileManager.default.fileExists(atPath: sourceURL.path) else {
      throw Img2ImgUtilities.Img2ImgError.sourceImageNotFound(request.sourceImagePath)
    }

    let imageData = try Data(contentsOf: sourceURL)
    guard !imageData.isEmpty else {
      throw Img2ImgUtilities.Img2ImgError.sourceImageLoadFailed("Empty file: \(request.sourceImagePath)")
    }

    // Determine output dimensions
    let resolvedWidth: Int
    let resolvedHeight: Int
    if let w = request.width, let h = request.height {
      resolvedWidth = Img2ImgUtilities.roundTo16(w)
      resolvedHeight = Img2ImgUtilities.roundTo16(h)
    } else if let dims = Img2ImgUtilities.pngDimensions(from: imageData) {
      resolvedWidth = Img2ImgUtilities.roundTo16(dims.width)
      resolvedHeight = Img2ImgUtilities.roundTo16(dims.height)
    } else {
      #if canImport(CoreGraphics)
      let cgImage = try InpaintUtilities.loadCGImage(from: imageData)
      resolvedWidth = Img2ImgUtilities.roundTo16(cgImage.width)
      resolvedHeight = Img2ImgUtilities.roundTo16(cgImage.height)
      #else
      resolvedWidth = ZImageModelMetadata.recommendedWidth
      resolvedHeight = ZImageModelMetadata.recommendedHeight
      #endif
    }

    // Mask resolution, in precedence order:
    //   mask_path  — caller-provided PNG (selective inpainting)
    //   mask_region — auto-generated ("face" via Vision, "upper"/"lower" bands)
    //   neither    — full-white mask (standard img2img, regenerate everything)
    // White = regenerate, black = keep; mask_invert flips the roles.
    // An explicit request that can't be honored fails loudly — silently
    // regenerating the whole frame is the opposite of the caller's intent.
    let hasMaskPath = !(request.maskPath ?? "").isEmpty
    let hasMaskRegion = !(request.maskRegion ?? "").isEmpty
    if hasMaskPath && hasMaskRegion {
      throw Img2ImgUtilities.Img2ImgError.maskLoadFailed(
        "mask_path and mask_region are mutually exclusive")
    }
    if request.maskInvert && !hasMaskPath && !hasMaskRegion {
      throw Img2ImgUtilities.Img2ImgError.maskLoadFailed(
        "mask_invert requires mask_path or mask_region")
    }

    var maskData: Data
    if hasMaskPath, let maskPath = request.maskPath {
      guard FileManager.default.fileExists(atPath: maskPath) else {
        throw Img2ImgUtilities.Img2ImgError.maskNotFound(maskPath)
      }
      guard let providedMask = try? Data(contentsOf: URL(fileURLWithPath: maskPath)),
            !providedMask.isEmpty else {
        throw Img2ImgUtilities.Img2ImgError.maskLoadFailed("Unreadable or empty file: \(maskPath)")
      }
      maskData = providedMask
      #if canImport(CoreGraphics)
      if request.maskInvert {
        maskData = try RegionMaskUtilities.invertMaskPNG(maskData)
      }
      #endif
    } else if hasMaskRegion, let maskRegion = request.maskRegion {
      #if canImport(CoreGraphics)
      // Face detection reads the source image; band regions don't need it.
      let sourceCG: CGImage? = maskRegion.lowercased() == RegionMaskUtilities.Region.face.rawValue
        ? try InpaintUtilities.loadCGImage(from: imageData)
        : nil
      maskData = try RegionMaskUtilities.makeRegionMaskPNG(
        region: maskRegion,
        sourceImage: sourceCG,
        width: resolvedWidth,
        height: resolvedHeight,
        invert: request.maskInvert
      )
      #else
      throw Img2ImgUtilities.Img2ImgError.maskGenerationFailed(
        "mask_region requires CoreGraphics")
      #endif
    } else {
      maskData = try Img2ImgUtilities.generateWhiteMaskPNG(
        width: resolvedWidth,
        height: resolvedHeight
      )
    }

    // Build DyPE config
    let dyPEConfig: DyPEConfig
    if request.dyPE.enabled {
      dyPEConfig = request.dyPE
    } else if max(resolvedWidth, resolvedHeight) > 1024 {
      dyPEConfig = .ntk
    } else {
      dyPEConfig = .disabled
    }

    return ZImageGenerationRequest(
      prompt: request.prompt,
      negativePrompt: request.negativePrompt,
      width: resolvedWidth,
      height: resolvedHeight,
      steps: request.steps,
      guidanceScale: request.guidanceScale,
      seed: request.seed,
      outputPath: request.outputPath,
      levelsMin: request.levelsMin,
      levelsMax: request.levelsMax,
      model: request.model,
      source: request.source,
      contentMode: request.contentMode,
      textEncoderPath: request.textEncoderPath,
      maxSequenceLength: request.maxSequenceLength,
      loras: request.loras,
      enhancePrompt: request.enhancePrompt,
      enhanceMaxTokens: request.enhanceMaxTokens,
      forceTransformerOverrideOnly: request.forceTransformerOverrideOnly,
      schedulerKind: request.schedulerKind,
      sigmaSchedule: request.sigmaSchedule,
      eta: request.eta,
      shift: request.shift,
      dyPE: dyPEConfig,
      inpaintImageData: imageData,
      maskData: maskData,
      denoise: request.denoise,
      maskGrow: request.maskGrow,
      maskFeather: request.maskFeather
    )
  }
}
