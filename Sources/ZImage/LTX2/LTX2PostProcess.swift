// LTX2PostProcess.swift -- Frame extraction and MP4 video encoding
// Phase 4 of the LTX-2 Swift/MLX port
//
// Converts decoded VAE output (MLXArray frames) to CGImage frames and
// encodes them to an H.264 MP4 file using AVFoundation.
//
// The VAE outputs (B, 3, F, H, W) in float32, range [0, 1].
// This module clamps, converts to uint8, creates CGImages, and writes MP4.

#if canImport(AVFoundation)
import AVFoundation
#endif

#if canImport(CoreGraphics)
import CoreGraphics
import CoreImage
#endif

import Foundation
import MLX

/// Post-processing utilities for LTX-2 video output.
public enum LTX2PostProcess {

  // MARK: - Frame Extraction

  /// Convert decoded VAE output to an array of pixel data buffers.
  ///
  /// Takes the VAE output `(B, 3, F, H, W)` float32 [0, 1] and converts
  /// each frame to RGBA pixel data (uint8).
  ///
  /// - Parameters:
  ///   - decoded: VAE decoded output `(B, 3, F, H, W)` in float32.
  ///   - batchIndex: Which batch element to extract. Default 0.
  /// - Returns: Array of `(width, height, pixelData)` tuples, one per frame.
  public static func extractFrames(
    from decoded: MLXArray,
    batchIndex: Int = 0
  ) -> [(width: Int, height: Int, pixels: [UInt8])] {
    // decoded shape: (B, 3, F, H, W)
    let numFrames = decoded.dim(2)
    let height = decoded.dim(3)
    let width = decoded.dim(4)

    var frames: [(width: Int, height: Int, pixels: [UInt8])] = []

    for f in 0..<numFrames {
      // Extract frame: (3, H, W)
      let frame = decoded[batchIndex, 0..., f]  // (3, H, W)

      // Clamp to [0, 1]
      let clamped = MLX.clip(frame, min: 0, max: 1)

      // Convert to uint8: (3, H, W) -> scale by 255
      let scaled = (clamped * 255.0).asType(.uint8)
      eval(scaled)

      // Use contiguous() before transposing to ensure correct memory layout
      let hwc = scaled.transposed(1, 2, 0).contiguous()  // (H, W, 3) contiguous
      eval(hwc)

      // Bulk copy pixel data as RGB
      let rgbData: [UInt8]
      let flatArray = hwc.reshaped(-1)
      eval(flatArray)
      rgbData = flatArray.asArray(UInt8.self)

      // Convert RGB to RGBA (add alpha = 255)
      var rgbaData = [UInt8](repeating: 255, count: height * width * 4)
      for i in 0..<(height * width) {
        rgbaData[i * 4 + 0] = rgbData[i * 3 + 0]  // R
        rgbaData[i * 4 + 1] = rgbData[i * 3 + 1]  // G
        rgbaData[i * 4 + 2] = rgbData[i * 3 + 2]  // B
        // Alpha already 255
      }

      frames.append((width: width, height: height, pixels: rgbaData))
    }

    return frames
  }

  #if canImport(CoreGraphics)
  /// Convert decoded VAE output to CGImage frames.
  ///
  /// - Parameters:
  ///   - decoded: VAE decoded output `(B, 3, F, H, W)` in float32.
  ///   - batchIndex: Which batch element to extract. Default 0.
  /// - Returns: Array of CGImages, one per frame.
  public static func framesToImages(
    from decoded: MLXArray,
    batchIndex: Int = 0
  ) -> [CGImage] {
    let rawFrames = extractFrames(from: decoded, batchIndex: batchIndex)
    var images: [CGImage] = []

    for frame in rawFrames {
      if let image = createCGImage(
        pixels: frame.pixels,
        width: frame.width,
        height: frame.height
      ) {
        images.append(image)
      }
    }

    return images
  }

  /// Create a CGImage from RGBA pixel data.
  private static func createCGImage(
    pixels: [UInt8],
    width: Int,
    height: Int
  ) -> CGImage? {
    let bitsPerComponent = 8
    let bitsPerPixel = 32
    let bytesPerRow = width * 4
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

    guard let provider = CGDataProvider(
      data: Data(pixels) as CFData
    ) else { return nil }

    return CGImage(
      width: width,
      height: height,
      bitsPerComponent: bitsPerComponent,
      bitsPerPixel: bitsPerPixel,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: bitmapInfo,
      provider: provider,
      decode: nil,
      shouldInterpolate: false,
      intent: .defaultIntent
    )
  }
  #endif

  #if canImport(AVFoundation) && canImport(CoreGraphics)
  /// Write video frames to an MP4 file using AVFoundation.
  ///
  /// Uses H.264 encoding with AVAssetWriter for broad compatibility.
  ///
  /// - Parameters:
  ///   - frames: Array of CGImages to encode.
  ///   - outputPath: Path for the output MP4 file.
  ///   - fps: Frames per second. Default 24.
  ///   - width: Video width in pixels.
  ///   - height: Video height in pixels.
  /// - Throws: If video writing fails.
  public static func writeMP4(
    frames: [CGImage],
    outputPath: String,
    fps: Int = 24,
    width: Int,
    height: Int,
    bitsPerPixelOverride: Double? = nil
  ) throws {
    guard !frames.isEmpty else {
      throw LTX2PostProcessError.noFrames
    }

    let outputURL = URL(fileURLWithPath: outputPath)

    // Remove existing file
    try? FileManager.default.removeItem(at: outputURL)

    // Create asset writer
    let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)

    // Video settings. Bitrate: 0.5 bits/pixel (~12 Mbps @ 768x1280x24) is
    // visually equivalent to the old 4 bits/px for generated content but ~8x
    // smaller — the old setting produced 139MB 12s files that exceeded
    // Telegram's 50MB bot upload cap. Env-tunable via LTX2_VIDEO_BITS_PER_PX.
    let bitsPerPixel = bitsPerPixelOverride
      ?? Double(ProcessInfo.processInfo.environment["LTX2_VIDEO_BITS_PER_PX"] ?? "") ?? 0.5
    let videoSettings: [String: Any] = [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: Int(Double(width * height * fps) * bitsPerPixel),
        AVVideoMaxKeyFrameIntervalKey: fps,
        AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
      ] as [String: Any],
    ]

    let input = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: videoSettings
    )
    input.expectsMediaDataInRealTime = false

    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: input,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
      ]
    )

    writer.add(input)
    writer.startWriting()
    writer.startSession(atSourceTime: .zero)

    let frameDuration = CMTimeMake(value: 1, timescale: Int32(fps))

    for (index, cgImage) in frames.enumerated() {
      // Wait for input to be ready
      while !input.isReadyForMoreMediaData {
        Thread.sleep(forTimeInterval: 0.01)
      }

      let presentationTime = CMTimeMultiply(frameDuration, multiplier: Int32(index))

      // Create pixel buffer from CGImage
      guard let pixelBuffer = createPixelBuffer(from: cgImage, width: width, height: height) else {
        throw LTX2PostProcessError.pixelBufferCreationFailed(frameIndex: index)
      }

      adaptor.append(pixelBuffer, withPresentationTime: presentationTime)
    }

    input.markAsFinished()

    // Wait for writing to complete
    let semaphore = DispatchSemaphore(value: 0)
    writer.finishWriting {
      semaphore.signal()
    }
    semaphore.wait()

    if writer.status == .failed {
      throw LTX2PostProcessError.writingFailed(writer.error?.localizedDescription ?? "unknown")
    }
  }

  /// Create a CVPixelBuffer from a CGImage.
  private static func createPixelBuffer(
    from image: CGImage,
    width: Int,
    height: Int
  ) -> CVPixelBuffer? {
    var pixelBuffer: CVPixelBuffer?
    let attrs: [String: Any] = [
      kCVPixelBufferCGImageCompatibilityKey as String: true,
      kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
    ]

    let status = CVPixelBufferCreate(
      kCFAllocatorDefault,
      width, height,
      kCVPixelFormatType_32ARGB,
      attrs as CFDictionary,
      &pixelBuffer
    )

    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
      return nil
    }

    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

    guard let context = CGContext(
      data: CVPixelBufferGetBaseAddress(buffer),
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
    ) else {
      return nil
    }

    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return buffer
  }
  #endif

  /// Write raw frame data to a sequence of PPM files (platform-independent fallback).
  ///
  /// - Parameters:
  ///   - decoded: VAE decoded output `(B, 3, F, H, W)`.
  ///   - outputDir: Directory to write PPM files.
  ///   - prefix: Filename prefix. Default "frame".
  ///   - batchIndex: Batch element index. Default 0.
  /// - Throws: If directory creation or file writing fails.
  public static func writeFramesPPM(
    from decoded: MLXArray,
    outputDir: String,
    prefix: String = "frame",
    batchIndex: Int = 0
  ) throws {
    let frames = extractFrames(from: decoded, batchIndex: batchIndex)

    let fm = FileManager.default
    try fm.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

    for (index, frame) in frames.enumerated() {
      let filename = String(format: "%@_%04d.ppm", prefix, index)
      let path = (outputDir as NSString).appendingPathComponent(filename)

      // PPM header
      var data = Data("P6\n\(frame.width) \(frame.height)\n255\n".utf8)

      // RGBA -> RGB
      for i in 0..<(frame.width * frame.height) {
        data.append(frame.pixels[i * 4])      // R
        data.append(frame.pixels[i * 4 + 1])  // G
        data.append(frame.pixels[i * 4 + 2])  // B
      }

      try data.write(to: URL(fileURLWithPath: path))
    }
  }

  /// Round-trip a single frame through a REAL H.264 encode/decode so it carries
  /// video-codec artifacts (deblocked DCT, chroma subsampling). Ports ComfyUI's
  /// LTXVPreprocess (PyAV libx264 CRF round-trip): LTX is trained on video
  /// frames, and a pristine still conditioning frame freezes i2v motion.
  /// `compression` follows the ComfyUI 0-100 scale (35 default ≈ CRF 35).
  public static func h264RoundTrip(_ image: CGImage, compression: Int) throws -> CGImage {
    // Per-frame bit budget: bpp ≈ 1.05·e^(−compression/28) → 35 ≈ 0.30 bpp
    // (single I-frame at roughly CRF-35 crunch). Floor keeps the encoder sane.
    let bpp = max(0.06, 1.05 * exp(-Double(compression) / 28.0))
    let tmp = NSTemporaryDirectory() + "ltx2-cond-\(UUID().uuidString).mp4"
    defer { try? FileManager.default.removeItem(atPath: tmp) }
    try writeMP4(
      frames: [image], outputPath: tmp, fps: 24,
      width: image.width, height: image.height,
      bitsPerPixelOverride: bpp)
    let asset = AVURLAsset(url: URL(fileURLWithPath: tmp))
    let gen = AVAssetImageGenerator(asset: asset)
    gen.requestedTimeToleranceBefore = .zero
    gen.requestedTimeToleranceAfter = .positiveInfinity
    gen.appliesPreferredTrackTransform = false
    return try gen.copyCGImage(at: .zero, actualTime: nil)
  }
}

// MARK: - Errors

/// Errors from post-processing operations.
public enum LTX2PostProcessError: Error, CustomStringConvertible {
  case noFrames
  case pixelBufferCreationFailed(frameIndex: Int)
  case writingFailed(String)

  public var description: String {
    switch self {
    case .noFrames:
      return "No frames to write"
    case .pixelBufferCreationFailed(let idx):
      return "Failed to create pixel buffer for frame \(idx)"
    case .writingFailed(let msg):
      return "Video writing failed: \(msg)"
    }
  }
}
