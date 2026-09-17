import XCTest
@testable import ZImage

#if canImport(AVFoundation) && canImport(CoreGraphics)
import AVFoundation
import CoreGraphics
import ImageIO

/// 2026-09-17, Director ladder rung 2: the carry-over PNG `extractLastFrame`
/// wrote for a 193-frame chunk was frame 178, not 192 —
/// `requestedTimeToleranceBefore = .positiveInfinity` lets AVFoundation hand
/// back any earlier, cheaper-to-decode frame. Continuation chunks (Director),
/// storyboard shots and /v1/video/extend all restarted up to ~0.6 s early.
/// This clip makes EVERY frame distinguishable (grey level = 4 × index) and
/// uses a long GOP, so any frame other than the last fails the assertion.
final class LastFrameExtractorExactTests: XCTestCase {

  private func writeRampClip(frames: Int, fps: Int, to url: URL) throws {
    let w = 64, h = 64
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: w, AVVideoHeightKey: h,
      AVVideoCompressionPropertiesKey: [
        AVVideoMaxKeyFrameIntervalKey: frames,          // one sync frame at the start
        AVVideoAllowFrameReorderingKey: true,           // B-frames, like production encodes
        AVVideoAverageBitRateKey: 400_000,
      ],
    ])
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
      kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
    ])
    writer.add(input)
    XCTAssertTrue(writer.startWriting())
    writer.startSession(atSourceTime: .zero)
    for i in 0..<frames {
      while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.002) }
      var pb: CVPixelBuffer?
      CVPixelBufferPoolCreatePixelBuffer(nil, try XCTUnwrap(adaptor.pixelBufferPool), &pb)
      let buffer = try XCTUnwrap(pb)
      CVPixelBufferLockBaseAddress(buffer, [])
      let base = try XCTUnwrap(CVPixelBufferGetBaseAddress(buffer)).assumingMemoryBound(to: UInt8.self)
      let grey = UInt8(min(255, 4 * i))
      for p in 0..<(CVPixelBufferGetBytesPerRow(buffer) * h) { base[p] = (p % 4 == 3) ? 255 : grey }
      CVPixelBufferUnlockBaseAddress(buffer, [])
      XCTAssertTrue(adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: CMTimeScale(fps))))
    }
    input.markAsFinished()
    let done = expectation(description: "finish")
    writer.finishWriting { done.fulfill() }
    wait(for: [done], timeout: 30)
    XCTAssertEqual(writer.status, .completed, writer.error.map { "\($0)" } ?? "")
  }

  private func meanGrey(ofPNG path: String) throws -> Double {
    let src = try XCTUnwrap(CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil))
    let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(src, 0, nil))
    var buf = [UInt8](repeating: 0, count: image.width * image.height)
    let ctx = try XCTUnwrap(CGContext(data: &buf, width: image.width, height: image.height, bitsPerComponent: 8,
                                      bytesPerRow: image.width, space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue))
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    return Double(buf.reduce(0) { $0 + Int($1) }) / Double(buf.count)
  }

  func testExtractsTheTrueLastFrameNotAnEarlierSyncFrame() throws {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("lastframe-exact-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let frames = 49, fps = 24
    let clip = dir.appendingPathComponent("ramp.mp4")
    try writeRampClip(frames: frames, fps: fps, to: clip)

    let out = dir.appendingPathComponent("last.png").path
    try LastFrameExtractor.extractLastFrame(from: clip.path, to: out)
    let grey = try meanGrey(ofPNG: out)
    let expected = Double(4 * (frames - 1))                 // 192
    // Adjacent frames differ by 4 grey levels; H.264 + range conversion stay within ±3.
    XCTAssertEqual(grey, expected, accuracy: 3.0,
                   "extracted frame grey \(grey) ≈ frame \(Int((grey / 4).rounded())), expected the last frame (\(frames - 1))")
  }
}
#endif
