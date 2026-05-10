import Foundation
import MLX

#if canImport(AVFoundation)
import AVFoundation
import CoreGraphics
import CoreMedia
import CoreVideo

/// Writes MLX video frames to an MP4 file using AVFoundation.
///
/// Handles the complete pipeline from MLXArray video tensor to QuickTime-playable MP4:
/// 1. Convert MLXArray [C, F, H, W] frames to CGImages
/// 2. Write frames to H.264 MP4 via AVAssetWriter
public enum WanMP4Writer {

  // MARK: - Errors

  public enum WriterError: Error, CustomStringConvertible {
    case writerCreationFailed(String)
    case pixelBufferCreationFailed
    case frameAppendFailed(Int)
    case finalizationFailed

    public var description: String {
      switch self {
      case .writerCreationFailed(let reason):
        return "Failed to create AVAssetWriter: \(reason)"
      case .pixelBufferCreationFailed:
        return "Failed to create pixel buffer"
      case .frameAppendFailed(let idx):
        return "Failed to append frame \(idx)"
      case .finalizationFailed:
        return "Failed to finalize MP4"
      }
    }
  }

  // MARK: - MP4 Writing

  /// Writes a video tensor to an MP4 file.
  ///
  /// - Parameters:
  ///   - video: Video tensor [C, F, H, W] in [0, 1] range, where C=3 (RGB).
  ///   - outputPath: Path to the output MP4 file.
  ///   - fps: Frames per second (default 16 for Wan I2V).
  /// - Throws: If writing fails.
  public static func write(
    video: MLXArray,
    to outputPath: String,
    fps: Int = 16
  ) throws {
    precondition(video.ndim == 4, "Expected [C, F, H, W], got ndim=\(video.ndim)")
    precondition(video.dim(0) == 3, "Expected 3 channels (RGB), got \(video.dim(0))")

    let numFrames = video.dim(1)
    let height = video.dim(2)
    let width = video.dim(3)

    // Remove existing file
    let outputURL = URL(fileURLWithPath: outputPath)
    try? FileManager.default.removeItem(at: outputURL)

    // Create AVAssetWriter
    let writer: AVAssetWriter
    do {
      writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
    } catch {
      throw WriterError.writerCreationFailed(error.localizedDescription)
    }

    // Video settings — H.264
    let videoSettings: [String: Any] = [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: width * height * 8,  // ~8 bits per pixel
        AVVideoMaxKeyFrameIntervalKey: 30,
        AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
      ] as [String: Any],
    ]

    let writerInput = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: videoSettings
    )
    writerInput.expectsMediaDataInRealTime = false

    // Pixel buffer adaptor
    let sourcePixelBufferAttributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
      kCVPixelBufferWidthKey as String: width,
      kCVPixelBufferHeightKey as String: height,
    ]

    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: writerInput,
      sourcePixelBufferAttributes: sourcePixelBufferAttributes
    )

    writer.add(writerInput)

    guard writer.startWriting() else {
      throw WriterError.writerCreationFailed(writer.error?.localizedDescription ?? "unknown error")
    }
    writer.startSession(atSourceTime: .zero)

    // Convert and append frames
    let frameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))

    // Extract pixel data from video tensor
    let clamped = MLX.clip(video, min: 0.0, max: 1.0)
    let scaled = (clamped * 255.0).asType(.uint8)
    eval(scaled)

    for frameIdx in 0..<numFrames {
      // Wait for writer to be ready
      while !writerInput.isReadyForMoreMediaData {
        Thread.sleep(forTimeInterval: 0.01)
      }

      // Extract frame: [3, H, W] → pixel buffer
      let frame = scaled[0..., frameIdx, 0..., 0...]  // [3, H, W]
      eval(frame)

      // Create pixel buffer
      var pixelBuffer: CVPixelBuffer?
      let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width, height,
        kCVPixelFormatType_32BGRA,
        sourcePixelBufferAttributes as CFDictionary,
        &pixelBuffer
      )

      guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
        throw WriterError.pixelBufferCreationFailed
      }

      // Fill pixel buffer
      CVPixelBufferLockBaseAddress(buffer, [])
      defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

      let baseAddress = CVPixelBufferGetBaseAddress(buffer)!
      let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
      let dst = baseAddress.assumingMemoryBound(to: UInt8.self)

      // Get frame data as contiguous array
      let frameData = frame.asData().data

      frameData.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
        let srcPtr = src.bindMemory(to: UInt8.self)
        let pixelCount = height * width

        for y in 0..<height {
          for x in 0..<width {
            let pixelIdx = y * width + x
            let dstIdx = y * bytesPerRow + x * 4

            // CHW to BGRA
            let r = srcPtr[pixelIdx]                      // channel 0
            let g = srcPtr[pixelIdx + pixelCount]          // channel 1
            let b = srcPtr[pixelIdx + 2 * pixelCount]      // channel 2

            dst[dstIdx + 0] = b  // B
            dst[dstIdx + 1] = g  // G
            dst[dstIdx + 2] = r  // R
            dst[dstIdx + 3] = 255  // A
          }
        }
      }

      // Append pixel buffer
      let presentationTime = CMTimeMultiply(frameDuration, multiplier: Int32(frameIdx))
      let success = adaptor.append(buffer, withPresentationTime: presentationTime)
      if !success {
        throw WriterError.frameAppendFailed(frameIdx)
      }
    }

    // Finalize
    writerInput.markAsFinished()

    let semaphore = DispatchSemaphore(value: 0)
    writer.finishWriting {
      semaphore.signal()
    }
    semaphore.wait()

    if writer.status == .failed {
      throw WriterError.finalizationFailed
    }
  }
}
#endif
