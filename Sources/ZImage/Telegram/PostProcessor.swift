// PostProcessor.swift — Post-processing effects via Core Image.
//
// Phase 3: sharpen, saturation, color temperature, and film look presets.
// Uses CIFilter where available. All operations take Data in and return Data out.

import Foundation

#if canImport(CoreImage)
import CoreImage
import CoreGraphics
#if canImport(AppKit)
import AppKit
#endif
#endif

// MARK: - Film Look Presets

public struct FilmLook: Sendable {
  public let id: String
  public let name: String
  /// RGBA multiplier adjustments (applied via CIColorMatrix).
  public let redVector: SIMD4<Float>
  public let greenVector: SIMD4<Float>
  public let blueVector: SIMD4<Float>
  public let biasVector: SIMD4<Float>
  /// Additional saturation adjustment (1.0 = no change).
  public let saturationAdjust: Float
  /// Additional contrast adjustment (1.0 = no change).
  public let contrastAdjust: Float
}

// MARK: - PostProcessor

public enum PostProcessor {

  /// Available film look presets.
  public static let filmLooks: [FilmLook] = [
    FilmLook(
      id: "kodak-portra",
      name: "Kodak Portra 400",
      redVector: SIMD4<Float>(1.05, 0.0, 0.0, 0.0),
      greenVector: SIMD4<Float>(0.0, 1.0, 0.0, 0.0),
      blueVector: SIMD4<Float>(0.0, 0.0, 0.92, 0.0),
      biasVector: SIMD4<Float>(0.01, 0.005, 0.02, 0.0),
      saturationAdjust: 0.9,
      contrastAdjust: 0.95
    ),
    FilmLook(
      id: "fuji-velvia",
      name: "Fuji Velvia 50",
      redVector: SIMD4<Float>(1.15, 0.0, 0.0, 0.0),
      greenVector: SIMD4<Float>(0.0, 1.1, 0.0, 0.0),
      blueVector: SIMD4<Float>(0.0, 0.0, 1.08, 0.0),
      biasVector: SIMD4<Float>(0.0, 0.0, 0.0, 0.0),
      saturationAdjust: 1.3,
      contrastAdjust: 1.1
    ),
    FilmLook(
      id: "ilford-hp5",
      name: "Ilford HP5 Plus",
      redVector: SIMD4<Float>(0.33, 0.33, 0.33, 0.0),
      greenVector: SIMD4<Float>(0.33, 0.33, 0.33, 0.0),
      blueVector: SIMD4<Float>(0.33, 0.33, 0.33, 0.0),
      biasVector: SIMD4<Float>(0.02, 0.02, 0.02, 0.0),
      saturationAdjust: 0.0,
      contrastAdjust: 1.15
    ),
    FilmLook(
      id: "cinestill-800t",
      name: "CineStill 800T",
      redVector: SIMD4<Float>(1.0, 0.0, 0.0, 0.0),
      greenVector: SIMD4<Float>(0.0, 0.95, 0.0, 0.0),
      blueVector: SIMD4<Float>(0.0, 0.05, 1.12, 0.0),
      biasVector: SIMD4<Float>(0.0, 0.0, 0.03, 0.0),
      saturationAdjust: 0.95,
      contrastAdjust: 1.05
    ),
    FilmLook(
      id: "kodak-ektar",
      name: "Kodak Ektar 100",
      redVector: SIMD4<Float>(1.1, 0.0, 0.0, 0.0),
      greenVector: SIMD4<Float>(0.0, 1.05, 0.0, 0.0),
      blueVector: SIMD4<Float>(0.0, 0.0, 1.0, 0.0),
      biasVector: SIMD4<Float>(0.0, 0.0, 0.0, 0.0),
      saturationAdjust: 1.2,
      contrastAdjust: 1.08
    ),
  ]

  /// List available film looks as (id, name) tuples.
  public static func availableLooks() -> [(id: String, name: String)] {
    return filmLooks.map { ($0.id, $0.name) }
  }

  /// Find a film look by ID. Case-insensitive.
  public static func findLook(_ id: String) -> FilmLook? {
    return filmLooks.first { $0.id.lowercased() == id.lowercased() }
  }

  // MARK: - Core Image Processing

  #if canImport(CoreImage)

  /// Apply unsharp mask sharpening.
  /// - Parameters:
  ///   - imageData: Input PNG/JPEG data.
  ///   - radius: Sharpening radius (default 2.5).
  ///   - intensity: Sharpening intensity (default 0.5).
  /// - Returns: Processed PNG data.
  public static func sharpen(imageData: Data, radius: Double = 2.5, intensity: Double = 0.5) -> Data? {
    guard let ciImage = CIImage(data: imageData) else { return nil }
    let filter = CIFilter(name: "CIUnsharpMask")!
    filter.setValue(ciImage, forKey: kCIInputImageKey)
    filter.setValue(radius, forKey: "inputRadius")
    filter.setValue(intensity, forKey: "inputIntensity")
    guard let output = filter.outputImage else { return nil }
    return renderToPNG(output)
  }

  /// Adjust saturation.
  /// - Parameters:
  ///   - imageData: Input PNG/JPEG data.
  ///   - factor: Saturation factor (0 = grayscale, 1 = original, 2 = double).
  /// - Returns: Processed PNG data.
  public static func adjustSaturation(imageData: Data, factor: Double) -> Data? {
    guard let ciImage = CIImage(data: imageData) else { return nil }
    let filter = CIFilter(name: "CIColorControls")!
    filter.setValue(ciImage, forKey: kCIInputImageKey)
    filter.setValue(factor, forKey: "inputSaturation")
    guard let output = filter.outputImage else { return nil }
    return renderToPNG(output)
  }

  /// Adjust color temperature.
  /// - Parameters:
  ///   - imageData: Input PNG/JPEG data.
  ///   - kelvin: Color temperature in Kelvin (2000-10000). 6500 = neutral daylight.
  /// - Returns: Processed PNG data.
  public static func adjustColorTemperature(imageData: Data, kelvin: Int) -> Data? {
    guard let ciImage = CIImage(data: imageData) else { return nil }
    let filter = CIFilter(name: "CITemperatureAndTint")!
    filter.setValue(ciImage, forKey: kCIInputImageKey)
    // CITemperatureAndTint uses a neutral of 6500K.
    // Positive targetNeutral shifts warm, negative shifts cool.
    let neutral = CIVector(x: CGFloat(kelvin), y: 0)
    filter.setValue(neutral, forKey: "inputNeutral")
    let target = CIVector(x: 6500, y: 0)
    filter.setValue(target, forKey: "inputTargetNeutral")
    guard let output = filter.outputImage else { return nil }
    return renderToPNG(output)
  }

  /// Apply a film look preset.
  /// - Parameters:
  ///   - imageData: Input PNG/JPEG data.
  ///   - look: The film look preset to apply.
  /// - Returns: Processed PNG data.
  public static func applyFilmLook(imageData: Data, look: FilmLook) -> Data? {
    guard var ciImage = CIImage(data: imageData) else { return nil }

    // Apply color matrix transformation
    let matrixFilter = CIFilter(name: "CIColorMatrix")!
    matrixFilter.setValue(ciImage, forKey: kCIInputImageKey)
    matrixFilter.setValue(CIVector(x: CGFloat(look.redVector.x), y: CGFloat(look.redVector.y),
                                   z: CGFloat(look.redVector.z), w: CGFloat(look.redVector.w)),
                          forKey: "inputRVector")
    matrixFilter.setValue(CIVector(x: CGFloat(look.greenVector.x), y: CGFloat(look.greenVector.y),
                                   z: CGFloat(look.greenVector.z), w: CGFloat(look.greenVector.w)),
                          forKey: "inputGVector")
    matrixFilter.setValue(CIVector(x: CGFloat(look.blueVector.x), y: CGFloat(look.blueVector.y),
                                   z: CGFloat(look.blueVector.z), w: CGFloat(look.blueVector.w)),
                          forKey: "inputBVector")
    matrixFilter.setValue(CIVector(x: CGFloat(look.biasVector.x), y: CGFloat(look.biasVector.y),
                                   z: CGFloat(look.biasVector.z), w: CGFloat(look.biasVector.w)),
                          forKey: "inputBiasVector")
    guard let matrixOutput = matrixFilter.outputImage else { return nil }
    ciImage = matrixOutput

    // Apply saturation adjustment
    if look.saturationAdjust != 1.0 {
      let satFilter = CIFilter(name: "CIColorControls")!
      satFilter.setValue(ciImage, forKey: kCIInputImageKey)
      satFilter.setValue(Double(look.saturationAdjust), forKey: "inputSaturation")
      if look.contrastAdjust != 1.0 {
        satFilter.setValue(Double(look.contrastAdjust), forKey: "inputContrast")
      }
      guard let satOutput = satFilter.outputImage else { return nil }
      ciImage = satOutput
    } else if look.contrastAdjust != 1.0 {
      let conFilter = CIFilter(name: "CIColorControls")!
      conFilter.setValue(ciImage, forKey: kCIInputImageKey)
      conFilter.setValue(Double(look.contrastAdjust), forKey: "inputContrast")
      guard let conOutput = conFilter.outputImage else { return nil }
      ciImage = conOutput
    }

    return renderToPNG(ciImage)
  }

  /// Apply the full post-processing pipeline to image data.
  /// - Returns: Processed PNG data, or the original data if nothing was applied.
  public static func applyPipeline(
    imageData: Data,
    saturation: Double?,
    colorTemp: Int?,
    filmLookId: String?,
    sharpenAfterUpscale: Bool = false
  ) -> Data {
    var data = imageData
    var applied = false

    // 1. Sharpen (only after upscale to counteract softness)
    if sharpenAfterUpscale {
      if let result = sharpen(imageData: data) {
        data = result
        applied = true
      }
    }

    // 2. Saturation adjustment
    if let sat = saturation {
      if let result = adjustSaturation(imageData: data, factor: sat) {
        data = result
        applied = true
      }
    }

    // 3. Color temperature
    if let kelvin = colorTemp {
      if let result = adjustColorTemperature(imageData: data, kelvin: kelvin) {
        data = result
        applied = true
      }
    }

    // 4. Film look (applied last — it's the "film stock" envelope)
    if let lookId = filmLookId, let look = findLook(lookId) {
      if let result = applyFilmLook(imageData: data, look: look) {
        data = result
        applied = true
      }
    }

    _ = applied  // suppress unused warning
    return data
  }

  // MARK: - Rendering

  private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

  private static func renderToPNG(_ image: CIImage) -> Data? {
    guard let cgImage = ciContext.createCGImage(image, from: image.extent) else { return nil }
    let mutableData = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(mutableData, "public.png" as CFString, 1, nil) else { return nil }
    CGImageDestinationAddImage(destination, cgImage, nil)
    guard CGImageDestinationFinalize(destination) else { return nil }
    return mutableData as Data
  }

  #else

  // MARK: - Fallback (non-macOS) — ImageMagick CLI

  public static func sharpen(imageData: Data, radius: Double = 2.5, intensity: Double = 0.5) -> Data? {
    return processViaImageMagick(imageData: imageData, args: ["-sharpen", "0x\(radius)"])
  }

  public static func adjustSaturation(imageData: Data, factor: Double) -> Data? {
    let percent = Int(factor * 100)
    return processViaImageMagick(imageData: imageData, args: ["-modulate", "100,\(percent),100"])
  }

  public static func adjustColorTemperature(imageData: Data, kelvin: Int) -> Data? {
    // Approximate warm/cool shift via ImageMagick level adjustments
    if kelvin < 6500 {
      let warmth = Int(Double(6500 - kelvin) / 45.0)
      return processViaImageMagick(imageData: imageData, args: [
        "-fill", "rgb(\(warmth),\(warmth / 2),0)", "-colorize", "10%"
      ])
    } else if kelvin > 6500 {
      let coolness = Int(Double(kelvin - 6500) / 45.0)
      return processViaImageMagick(imageData: imageData, args: [
        "-fill", "rgb(0,\(coolness / 2),\(coolness))", "-colorize", "10%"
      ])
    }
    return imageData
  }

  public static func applyFilmLook(imageData: Data, look: FilmLook) -> Data? {
    // For non-CI platforms, apply basic saturation + contrast via ImageMagick
    let sat = Int(look.saturationAdjust * 100)
    let brightness = look.contrastAdjust > 1.0 ? 105 : (look.contrastAdjust < 1.0 ? 95 : 100)
    return processViaImageMagick(imageData: imageData, args: ["-modulate", "\(brightness),\(sat),100"])
  }

  public static func applyPipeline(
    imageData: Data,
    saturation: Double?,
    colorTemp: Int?,
    filmLookId: String?,
    sharpenAfterUpscale: Bool = false
  ) -> Data {
    var data = imageData
    if sharpenAfterUpscale, let result = sharpen(imageData: data) { data = result }
    if let sat = saturation, let result = adjustSaturation(imageData: data, factor: sat) { data = result }
    if let kelvin = colorTemp, let result = adjustColorTemperature(imageData: data, kelvin: kelvin) { data = result }
    if let lookId = filmLookId, let look = findLook(lookId), let result = applyFilmLook(imageData: data, look: look) { data = result }
    return data
  }

  private static func processViaImageMagick(imageData: Data, args: [String]) -> Data? {
    let tmpIn = NSTemporaryDirectory() + "comfybox-pp-in-\(UUID().uuidString).png"
    let tmpOut = NSTemporaryDirectory() + "comfybox-pp-out-\(UUID().uuidString).png"
    defer {
      try? FileManager.default.removeItem(atPath: tmpIn)
      try? FileManager.default.removeItem(atPath: tmpOut)
    }
    guard FileManager.default.createFile(atPath: tmpIn, contents: imageData) else { return nil }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/convert")
    process.arguments = [tmpIn] + args + [tmpOut]
    do {
      try process.run()
      process.waitUntilExit()
      guard process.terminationStatus == 0 else { return nil }
      return FileManager.default.contents(atPath: tmpOut)
    } catch {
      return nil
    }
  }

  public static func availableLooks() -> [(id: String, name: String)] {
    return filmLooks.map { ($0.id, $0.name) }
  }

  public static func findLook(_ id: String) -> FilmLook? {
    return filmLooks.first { $0.id.lowercased() == id.lowercased() }
  }

  #endif
}
