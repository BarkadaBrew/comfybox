import Foundation
import Logging
import MLX

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

public enum QwenImageIOError: Error {
  case unsupportedPixelFormat
  case invalidArrayShape
  case resizeFailed
  case writeFailed
  /// The WP-E10 `applied` record could not be turned into JSON for the PNG's
  /// EXIF `UserComment`. The image is still written; the record is not. Never
  /// thrown out of `saveImage` — it exists so the omission has a reason to log.
  case appliedRecordNotEncodable(reason: String)
}

/// The image writer has no injected logger and is called from every family's
/// render path; one file-scoped logger keeps a dropped provenance record from
/// being silent (WP-E10).
private let metadataLogger = Logger(label: "z-image.image-metadata")

public enum QwenImageIO {
  static func resizedCGImage(
    from image: CGImage,
    width: Int,
    height: Int,
    interpolation: CGInterpolationQuality = .high
  ) throws -> CGImage {
    guard width > 0 && height > 0 else {
      throw QwenImageIOError.resizeFailed
    }
    guard let colorSpace = image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB) else {
      throw QwenImageIOError.resizeFailed
    }
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    guard let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo
    ) else {
      throw QwenImageIOError.resizeFailed
    }

    context.interpolationQuality = interpolation
    let rect = CGRect(x: 0, y: 0, width: width, height: height)
    context.draw(image, in: rect)

    guard let scaled = context.makeImage() else {
      throw QwenImageIOError.resizeFailed
    }
    return scaled
  }

  static func array(
    from image: CGImage,
    addBatchDimension: Bool = true,
    dtype: DType = .float32
  ) throws -> MLXArray {
    let width = image.width
    let height = image.height
    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel

    var buffer = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

    let succeeded = buffer.withUnsafeMutableBytes { ptr -> Bool in
      guard let baseAddress = ptr.baseAddress else { return false }
      guard let context = CGContext(
        data: baseAddress,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: bitmapInfo
      ) else {
        return false
      }
      let drawRect = CGRect(x: 0, y: 0, width: width, height: height)
      context.draw(image, in: drawRect)
      return true
    }

    guard succeeded else {
      throw QwenImageIOError.unsupportedPixelFormat
    }

    var floats = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height {
      for x in 0..<width {
        let pixelIndex = y * width + x
        let srcIndex = pixelIndex * bytesPerPixel
        let destBase = pixelIndex

        let r = Float(buffer[srcIndex]) / 255.0
        let g = Float(buffer[srcIndex + 1]) / 255.0
        let b = Float(buffer[srcIndex + 2]) / 255.0

        floats[destBase] = r
        floats[destBase + width * height] = g
        floats[destBase + 2 * width * height] = b
      }
    }

    var shape = [3, height, width]
    if addBatchDimension {
      shape.insert(1, at: 0)
    }

    return MLXArray(floats, shape).asType(dtype)
  }

  static func image(from array: MLXArray) throws -> CGImage {
    var tensor = array
    precondition(tensor.ndim == 3 || (tensor.ndim == 4 && tensor.dim(0) == 1))
    if tensor.ndim == 4 {
      tensor = tensor[0, 0..., 0..., 0...]
    }
    precondition(tensor.ndim == 3 && tensor.dim(0) == 3, "Expected shape [3,H,W]")

    let height = tensor.dim(1)
    let width = tensor.dim(2)
    let pixelCount = height * width

    let clamped = MLX.clip(tensor, min: 0, max: 1)
    let scaled = clamped * 255.0
    let uint8Tensor = scaled.asType(.uint8)
    MLX.eval(uint8Tensor)

    let data = uint8Tensor.asData().data

    var bytes = [UInt8](repeating: 255, count: pixelCount * 4)

    data.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) in
      let srcPointer = pointer.bindMemory(to: UInt8.self)
      for pixel in 0..<pixelCount {
        let dstIndex = pixel * 4
        bytes[dstIndex] = srcPointer[pixel]
        bytes[dstIndex + 1] = srcPointer[pixel + pixelCount]
        bytes[dstIndex + 2] = srcPointer[pixel + pixelCount * 2]
      }
    }

    let providerData = Data(bytes)
    guard let provider = CGDataProvider(data: providerData as CFData) else {
      throw QwenImageIOError.unsupportedPixelFormat
    }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    guard let image = CGImage(
      width: width,
      height: height,
      bitsPerComponent: 8,
      bitsPerPixel: 32,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
      provider: provider,
      decode: nil,
      shouldInterpolate: false,
      intent: .defaultIntent
    ) else {
      throw QwenImageIOError.unsupportedPixelFormat
    }

    return image
  }

  public static func normalizeForEncoder(_ image: MLXArray) -> MLXArray {
    image * 2 - 1
  }

  static func denormalizeFromDecoder(_ image: MLXArray) -> MLXArray {
    (image + 1) / 2
  }

  /// Generation metadata embedded into the saved image as standard EXIF/IPTC/
  /// TIFF/PNG fields, so macOS Finder (Get Info → More Info) and Spotlight read
  /// it — the mflux-style "the image carries its own parameters" default.
  public struct ImageMetadata: Sendable {
    public var description: String        // the prompt
    public var keywords: [String]         // e.g. model family
    public var parametersJSON: String?    // full params for exact round-trip
    public var software: String
    public init(description: String, keywords: [String] = [],
                parametersJSON: String? = nil, software: String = "ComfyBox") {
      self.description = description; self.keywords = keywords
      self.parametersJSON = parametersJSON; self.software = software
    }

    /// Normalise a RAW REQUEST negative prompt for `generation(negativePrompt:)`
    /// (K-FIX-1 round 2, Minor 3).
    ///
    /// `generation` writes any non-nil value verbatim, `""` included, because
    /// Krea 2 hands it `trace.negativePromptApplied` where `""` is a FACT: CFG
    /// ran against an empty negative and paid a second model pass for it.
    /// Every other family hands it a raw payload field, where `""` means "the
    /// caller typed nothing" and must stay absent from the metadata — those
    /// families have no CFG-applied reading to distinguish it from.
    ///
    /// Call this at any site that forwards a request/payload value. Krea 2's
    /// applied value must NOT go through it.
    public static func requestNegative(_ raw: String?) -> String? {
        guard let raw, !raw.isEmpty else { return nil }
        return raw
    }

    /// Build from common generation params — shared by every image pipeline so
    /// all model families embed the same Finder-readable sidecar.
    public static func generation(
      prompt: String, negativePrompt: String? = nil, seed: UInt64? = nil,
      steps: Int? = nil, guidance: Float? = nil, width: Int? = nil,
      height: Int? = nil, model: String? = nil, generatedBy: String? = nil,
      contentMode: String? = nil, loras: [LoRAConfiguration] = [],
      applied: RenderRecipe? = nil,
      /// Tri-state provenance (round 2, C4). Pass this — not `applied` — from
      /// a Krea 2 render, so a REFUSED record writes `"applied": null` rather
      /// than vanishing into the same absent key a non-krea2 render produces.
      /// Takes precedence over `applied` when both are given.
      appliedSlot: AppliedRecordSlot? = nil,
      /// #399: the ``StylePack`` name the save path APPLIED, or nil. Written
      /// as a top-level `style` key beside `applied` — the provenance record
      /// can be refused (`"applied": null`), and the file must still say
      /// which look its pixels carry. Absent when nil, so an unstyled
      /// render's metadata is byte-identical to the pre-#399 one.
      style: String? = nil
    ) -> ImageMetadata {
      var params: [String: Any] = ["prompt": prompt]
      // WP-E10 sink 2: the provenance record rides in the PNG under `applied`
      // (snake_case, the same bytes the /v1/generate response carries).
      // Krea 2 only today (D12) — other families pass nil and emit no key.
      //
      // The omission is kept (a PNG without its record is still the image the
      // caller asked for) but it is never SILENT: `JSONEncoder` throws on a
      // non-conforming float, so a NaN sigma would otherwise drop the whole
      // block with nothing to explain the gap between the response and the file.
      // Round 2 (C4): the slot is the tri-state — a PRESENT slot holding no
      // record writes a literal `null` ("this was a Krea 2 render and the
      // engine refused its record"), which is a different fact from the key
      // being absent ("no Krea 2 provenance applies").
      let appliedRecordSlot = appliedSlot ?? applied.map(AppliedRecordSlot.init(record:))
      if let appliedRecordSlot {
        if let record = appliedRecordSlot.record {
          do {
            let encoder = JSONEncoder()
            encoder.keyEncodingStrategy = .convertToSnakeCase
            let data = try encoder.encode(record)
            guard let object = try? JSONSerialization.jsonObject(with: data) else {
              throw QwenImageIOError.appliedRecordNotEncodable(
                reason: "the encoded record is not a JSON object")
            }
            params["applied"] = object
          } catch {
            metadataLogger.error(
              "PNG metadata: the `applied` provenance record could not be encoded — writing the image WITHOUT it (\(error))")
          }
        } else {
          params["applied"] = NSNull()
        }
      }
      if let width { params["width"] = width }
      if let height { params["height"] = height }
      if let steps { params["steps"] = steps }
      if let guidance { params["guidance"] = Self.cleanNumber(guidance) }
      // K-FIX-1 / Codex I4: `nil` means the field does not apply; a non-nil
      // value — INCLUDING `""` — is a fact the caller resolved and is written
      // verbatim. Krea 2 passes `trace.negativePromptApplied`, where `""`
      // means CFG ran against an empty negative and a second model pass was
      // paid for it; dropping it wrote a file that read as if CFG never ran.
      // Callers that hold a raw payload normalise an empty string to nil
      // themselves, so no other family's metadata changes.
      if let negativePrompt { params["negative_prompt"] = negativePrompt }
      if let seed { params["seed"] = seed }
      // Which app/persona generated it — placed persona renders in the gallery.
      if let generatedBy, !generatedBy.isEmpty { params["source"] = generatedBy }
      if let contentMode, !contentMode.isEmpty { params["content_mode"] = contentMode }
      if let style, !style.isEmpty { params["style"] = style }
      if !loras.isEmpty {
        params["loras"] = loras.map { c -> [String: Any] in
          ["name": (c.source.displayName as NSString).deletingPathExtension,
           "scale": Self.cleanNumber(c.scale)]
        }
      }
      let modelName = model.flatMap { p -> String? in
        p.isEmpty ? nil : ((p as NSString).lastPathComponent as NSString).deletingPathExtension
      }
      if let modelName { params["model"] = modelName }
      // `.sortedKeys` is load-bearing, not cosmetic: `JSONSerialization` walks
      // a Swift `Dictionary` in hash order and Swift seeds its hasher PER
      // PROCESS, so without it the same render wrote different EXIF
      // `UserComment` bytes after every server restart and the whole-file SHA
      // of a PNG could never be compared across runs (WP-E10; AC-5's
      // byte-identity oracle depends on it). Sorting recurses into `applied`.
      let json = (try? JSONSerialization.data(withJSONObject: params, options: [.sortedKeys]))
        .flatMap { String(data: $0, encoding: .utf8) }
      return ImageMetadata(description: prompt,
                           keywords: [modelName].compactMap { $0 },
                           parametersJSON: json)
    }

    /// Render a Float as a JSON-clean number. `JSONSerialization` prints a
    /// `Double` at full precision (0.8 -> 0.80000000000000004); routing through
    /// the Float's shortest round-trippable decimal string via `NSDecimalNumber`
    /// yields the human-entered value (0.8, 0.4, -3) in the embedded JSON.
    static func cleanNumber(_ value: Float) -> NSNumber {
      NSDecimalNumber(string: String(value))
    }
  }

  /// Build a CGImageDestination properties dict from `metadata`.
  static func cgProperties(for m: ImageMetadata) -> [CFString: Any] {
    var props: [CFString: Any] = [:]
    props[kCGImagePropertyPNGDictionary] = [
      kCGImagePropertyPNGDescription: m.description,
      kCGImagePropertyPNGSoftware: m.software,
    ] as [CFString: Any]
    props[kCGImagePropertyTIFFDictionary] = [
      kCGImagePropertyTIFFImageDescription: m.description,
      kCGImagePropertyTIFFSoftware: m.software,
    ] as [CFString: Any]
    var iptc: [CFString: Any] = [kCGImagePropertyIPTCCaptionAbstract: m.description]
    if !m.keywords.isEmpty { iptc[kCGImagePropertyIPTCKeywords] = m.keywords }
    props[kCGImagePropertyIPTCDictionary] = iptc
    if let json = m.parametersJSON {
      props[kCGImagePropertyExifDictionary] = [kCGImagePropertyExifUserComment: json] as [CFString: Any]
    }
    return props
  }

  public static func saveImage(array: MLXArray, to url: URL, metadata: ImageMetadata? = nil) throws {
    let cg = try image(from: array)
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
      throw QwenImageIOError.writeFailed
    }
    let props = metadata.map { cgProperties(for: $0) as CFDictionary }
    CGImageDestinationAddImage(destination, cg, props)
    guard CGImageDestinationFinalize(destination) else {
      throw QwenImageIOError.writeFailed
    }
  }
  public static func imageData(from array: MLXArray) throws -> Data {
    let cg = try image(from: array)
    let mutableData = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(
      mutableData as CFMutableData,
      UTType.png.identifier as CFString,
      1,
      nil
    ) else {
      throw QwenImageIOError.writeFailed
    }
    CGImageDestinationAddImage(destination, cg, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw QwenImageIOError.writeFailed
    }
    return mutableData as Data
  }

  public static func resizedPixelArray(
    from image: CGImage,
    width: Int,
    height: Int,
    addBatchDimension: Bool = true,
    dtype: DType = .float32,
    interpolation: CGInterpolationQuality = .high
  ) throws -> MLXArray {
    guard width > 0, height > 0 else {
      throw QwenImageIOError.resizeFailed
    }

    let resizedImage = try resizedCGImage(from: image, width: width, height: height, interpolation: interpolation)
    let arr = try array(from: resizedImage, addBatchDimension: addBatchDimension, dtype: dtype)
    return arr
  }

  static func resize(
    rgbArray array: MLXArray,
    targetHeight: Int,
    targetWidth: Int
  ) throws -> MLXArray {
    precondition(array.ndim == 3 && array.dim(0) == 3, "Expected [3, H, W]")
    guard targetHeight > 0, targetWidth > 0 else {
      throw QwenImageIOError.resizeFailed
    }
    if array.dim(1) == targetHeight && array.dim(2) == targetWidth {
      return array
    }
    let source = array.asType(.float32)
    MLX.eval(source)
    let sourceData = source.asArray(Float32.self)
    let resized = resizeLanczos(
      source: sourceData,
      srcWidth: array.dim(2),
      srcHeight: array.dim(1),
      dstWidth: targetWidth,
      dstHeight: targetHeight
    )
    return MLXArray(resized, [3, targetHeight, targetWidth])
  }

  private struct KernelContribution {
    var index: Int
    var weight: Double
  }

  private struct FixedContribution {
    var start: Int
    var coefficients: [Int32]
  }

  @inline(__always)
  private static func clipToUInt8(_ value: Int64, precisionBits: Int) -> UInt8 {
    let shifted = value >> precisionBits
    if shifted <= 0 {
      return 0
    }
    if shifted >= 255 {
      return 255
    }
    return UInt8(shifted)
  }

  private static func makeFixedPointContributions(
    from contributions: [[KernelContribution]],
    precisionBits: Int
  ) -> [FixedContribution] {
    let scale = Double(1 << precisionBits)
    var fixed: [FixedContribution] = []
    fixed.reserveCapacity(contributions.count)

    for kernels in contributions {
      guard let first = kernels.first else {
        fixed.append(FixedContribution(start: 0, coefficients: [Int32(1 << precisionBits)]))
        continue
      }

      var coeffs: [Int32] = []
      coeffs.reserveCapacity(kernels.count)
      for kernel in kernels {
        let scaled = kernel.weight * scale
        let adjusted: Double
        if scaled < 0 {
          adjusted = scaled - 0.5
        } else {
          adjusted = scaled + 0.5
        }
        let intValue = Int32(adjusted.rounded(.towardZero))
        coeffs.append(intValue)
      }
      fixed.append(FixedContribution(start: first.index, coefficients: coeffs))
    }
    return fixed
  }

  private static func resizeLanczosARGB(
    argbBytes: [UInt8],
    srcWidth: Int,
    srcHeight: Int,
    dstWidth: Int,
    dstHeight: Int
  ) -> [Float32]? {
    guard srcWidth > 0, srcHeight > 0, dstWidth > 0, dstHeight > 0 else {
      return nil
    }
    let bytesPerPixel = 4
    let precisionBits = 22
    let horizontalContribs = makeContributions(
      srcLength: srcWidth,
      dstLength: dstWidth,
      support: 3.0
    )
    let verticalContribs = makeContributions(
      srcLength: srcHeight,
      dstLength: dstHeight,
      support: 3.0
    )
    let horizontalFixed = makeFixedPointContributions(
      from: horizontalContribs,
      precisionBits: precisionBits
    )
    let verticalFixed = makeFixedPointContributions(
      from: verticalContribs,
      precisionBits: precisionBits
    )

    let dstPixelCount = dstWidth * dstHeight
    let srcRowStride = srcWidth * bytesPerPixel
    let dstRowStride = dstWidth * bytesPerPixel
    var horizontal = [UInt8](repeating: 0, count: srcHeight * dstRowStride)
    var outputBytes = [UInt8](repeating: 0, count: dstHeight * dstRowStride)
    let roundingOffset = Int64(1 << (precisionBits - 1))
    for sy in 0..<srcHeight {
      let srcRowOffset = sy * srcRowStride
      let dstRowOffset = sy * dstRowStride
      for dx in 0..<dstWidth {
        let coeff = horizontalFixed[dx]
        var sumA = roundingOffset
        var sumR = roundingOffset
        var sumG = roundingOffset
        var sumB = roundingOffset
        for (index, weight) in coeff.coefficients.enumerated() {
          let srcX = coeff.start + index
          let srcIndex = srcRowOffset + srcX * bytesPerPixel
          let w = Int64(weight)
          sumA += Int64(argbBytes[srcIndex + 0]) * w
          sumR += Int64(argbBytes[srcIndex + 1]) * w
          sumG += Int64(argbBytes[srcIndex + 2]) * w
          sumB += Int64(argbBytes[srcIndex + 3]) * w
        }
        let dstIndex = dstRowOffset + dx * bytesPerPixel
        horizontal[dstIndex + 0] = clipToUInt8(sumA, precisionBits: precisionBits)
        horizontal[dstIndex + 1] = clipToUInt8(sumR, precisionBits: precisionBits)
        horizontal[dstIndex + 2] = clipToUInt8(sumG, precisionBits: precisionBits)
        horizontal[dstIndex + 3] = clipToUInt8(sumB, precisionBits: precisionBits)
      }
    }
    for dx in 0..<dstWidth {
      for dy in 0..<dstHeight {
        let coeff = verticalFixed[dy]
        var sumA = roundingOffset
        var sumR = roundingOffset
        var sumG = roundingOffset
        var sumB = roundingOffset
        for (index, weight) in coeff.coefficients.enumerated() {
          let srcY = coeff.start + index
          let srcIndex = srcY * dstRowStride + dx * bytesPerPixel
          let w = Int64(weight)
          sumA += Int64(horizontal[srcIndex + 0]) * w
          sumR += Int64(horizontal[srcIndex + 1]) * w
          sumG += Int64(horizontal[srcIndex + 2]) * w
          sumB += Int64(horizontal[srcIndex + 3]) * w
        }
        let dstIndex = dy * dstRowStride + dx * bytesPerPixel
        outputBytes[dstIndex + 0] = clipToUInt8(sumA, precisionBits: precisionBits)
        outputBytes[dstIndex + 1] = clipToUInt8(sumR, precisionBits: precisionBits)
        outputBytes[dstIndex + 2] = clipToUInt8(sumG, precisionBits: precisionBits)
        outputBytes[dstIndex + 3] = clipToUInt8(sumB, precisionBits: precisionBits)
      }
    }

    let channelSize = dstPixelCount
    var floats = [Float32](repeating: 0, count: channelSize * 3)
    for i in 0..<dstHeight {
      for j in 0..<dstWidth {
        let pixelIndex = i * dstWidth + j
        let base = pixelIndex * bytesPerPixel
        let r = Float32(outputBytes[base + 1]) / 255.0
        let g = Float32(outputBytes[base + 2]) / 255.0
        let b = Float32(outputBytes[base + 3]) / 255.0
        floats[pixelIndex] = r
        floats[pixelIndex + channelSize] = g
        floats[pixelIndex + 2 * channelSize] = b
      }
    }
    return floats
  }

  private static func resizeLanczos(
    source: [Float32],
    srcWidth: Int,
    srcHeight: Int,
    dstWidth: Int,
    dstHeight: Int,
    support: Double = 3.0
  ) -> [Float32] {
    precondition(source.count == 3 * srcWidth * srcHeight)
    if srcWidth == 0 || srcHeight == 0 || dstWidth == 0 || dstHeight == 0 {
      return Array(repeating: 0, count: 3 * dstWidth * dstHeight)
    }

    let horizontal = makeContributions(srcLength: srcWidth, dstLength: dstWidth, support: support)
    let vertical = makeContributions(srcLength: srcHeight, dstLength: dstHeight, support: support)
    let channels = 3

    var temp = [Double](repeating: 0, count: channels * dstWidth * srcHeight)
    for c in 0..<channels {
      for sy in 0..<srcHeight {
        for dx in 0..<dstWidth {
          var value = 0.0
          for contrib in horizontal[dx] {
            let sample = Double(source[(c * srcHeight + sy) * srcWidth + contrib.index])
            value += sample * contrib.weight
          }
          temp[(c * srcHeight + sy) * dstWidth + dx] = value
        }
      }
    }

    var output = [Float32](repeating: 0, count: channels * dstWidth * dstHeight)
    for c in 0..<channels {
      for dy in 0..<dstHeight {
        for dx in 0..<dstWidth {
          var value = 0.0
          for contrib in vertical[dy] {
            let sample = temp[(c * srcHeight + contrib.index) * dstWidth + dx]
            value += sample * contrib.weight
          }
          output[(c * dstHeight + dy) * dstWidth + dx] = Float32(value)
        }
      }
    }
    return output
  }

  private static func makeContributions(
    srcLength: Int,
    dstLength: Int,
    support: Double
  ) -> [[KernelContribution]] {
    precondition(dstLength > 0, "Destination length must be positive")
    if srcLength == 0 {
      return Array(repeating: [], count: dstLength)
    }

    let scale = Double(srcLength) / Double(dstLength)
    let filterScale = max(1.0, scale)
    let scaledSupport = support * filterScale
    let invFilterScale = 1.0 / filterScale

    func sinc(_ x: Double) -> Double {
      if abs(x) < Double.ulpOfOne {
        return 1.0
      }
      return sin(Double.pi * x) / (Double.pi * x)
    }

    func lanczos(_ x: Double) -> Double {
      let ax = abs(x)
      if ax >= support {
        return 0.0
      }
      return sinc(x) * sinc(x / support)
    }

    var contributions: [[KernelContribution]] = Array(repeating: [], count: dstLength)
    for dest in 0..<dstLength {
      let center = (Double(dest) + 0.5) * scale
      var left = Int((center - scaledSupport + 0.5).rounded(.towardZero))
      var right = Int((center + scaledSupport + 0.5).rounded(.towardZero))

      if left < 0 { left = 0 }
      if right > srcLength { right = srcLength }

      let tapCount = max(0, right - left)
      if tapCount == 0 {
        let fallback = max(0, min(srcLength - 1, Int(center.rounded(.towardZero))))
        contributions[dest] = [KernelContribution(index: fallback, weight: 1.0)]
        continue
      }

      var weights: [KernelContribution] = []
      weights.reserveCapacity(tapCount)

      var sum = 0.0
      for offset in 0..<tapCount {
        let sampleIndex = left + offset
        let distance = (Double(sampleIndex) - center + 0.5) * invFilterScale
        let weight = lanczos(distance)
        weights.append(KernelContribution(index: sampleIndex, weight: weight))
        sum += weight
      }

      if sum != 0.0 {
        for i in 0..<weights.count {
          weights[i].weight /= sum
        }
      }
      contributions[dest] = weights
    }
    return contributions
  }
}
#endif
