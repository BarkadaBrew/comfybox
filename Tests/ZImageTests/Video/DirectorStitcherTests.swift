import AVFoundation
import CoreGraphics
import Foundation
import XCTest

@testable import ZImage

/// WP2d: the chunk stitcher. Synthetic chunks from LTX2PostProcess.writeMP4
/// (solid-colour gradients), then container + decoded-frame assertions on
/// the stitched output: nominal frame rate, frame count, duration, first/
/// last colour, audio track format. No models.
final class DirectorStitcherTests: XCTestCase {

  private var tempDir: URL!
  private let w = 128
  private let h = 128

  override func setUpWithError() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("director-stitch-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempDir)
  }

  // MARK: - Fixtures

  private typealias RGB = (r: CGFloat, g: CGFloat, b: CGFloat)
  private let red: RGB = (1, 0, 0)
  private let blue: RGB = (0, 0, 1)
  private let green: RGB = (0, 1, 0)

  private func lerp(_ a: RGB, _ b: RGB, _ t: CGFloat) -> RGB {
    (a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t)
  }

  private func solidFrame(_ c: RGB) -> CGImage {
    let ctx = CGContext(
      data: nil, width: w, height: h, bitsPerComponent: 8,
      bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.setFillColor(CGColor(red: c.r, green: c.g, blue: c.b, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
    return ctx.makeImage()!
  }

  /// `frames` solid frames fading `from` -> `to` (frame 0 == from, last == to).
  private func gradientChunk(name: String, frames: Int, from: RGB, to: RGB, fps: Int = 24) throws -> String {
    let path = tempDir.appendingPathComponent(name).path
    let images = (0..<frames).map { i in
      solidFrame(lerp(from, to, frames > 1 ? CGFloat(i) / CGFloat(frames - 1) : 0))
    }
    try LTX2PostProcess.writeMP4(frames: images, outputPath: path, fps: fps, width: w, height: h)
    return path
  }

  private func tone(seconds: Double) -> StereoPCM {
    let n = Int(seconds * 48000)
    var s = [Float](repeating: 0, count: n)
    for i in 0..<n { s[i] = 0.5 * sin(2 * .pi * 440 * Float(i) / 48000) }
    return StereoPCM(left: s, right: s, sampleRate: 48000)
  }

  /// Decode every frame of `path`; returns the count and the mean RGB (0-1)
  /// of the first and last frames.
  private func decodeFrames(_ path: String) throws -> (count: Int, first: RGB, last: RGB) {
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    let track = try XCTUnwrap(asset.tracks(withMediaType: .video).first)
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(
      track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.add(output)
    XCTAssertTrue(reader.startReading())
    var count = 0
    var first: RGB = (0, 0, 0)
    var last: RGB = (0, 0, 0)
    while let sample = output.copyNextSampleBuffer() {
      guard let pb = CMSampleBufferGetImageBuffer(sample) else { continue }
      let mean = meanColour(pb)
      if count == 0 { first = mean }
      last = mean
      count += 1
    }
    XCTAssertEqual(reader.status, .completed)
    return (count, first, last)
  }

  private func meanColour(_ pb: CVPixelBuffer) -> RGB {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let width = CVPixelBufferGetWidth(pb), height = CVPixelBufferGetHeight(pb)
    let stride = CVPixelBufferGetBytesPerRow(pb)
    let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
    var r = 0.0, g = 0.0, b = 0.0
    for y in 0..<height {
      let row = base + y * stride
      for x in 0..<width {
        b += Double(row[x * 4]); g += Double(row[x * 4 + 1]); r += Double(row[x * 4 + 2])
      }
    }
    let n = Double(width * height) * 255
    return (CGFloat(r / n), CGFloat(g / n), CGFloat(b / n))
  }

  private func assertColour(_ got: RGB, _ want: RGB, tolerance: CGFloat = 0.15, _ label: String,
                            file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(got.r, want.r, accuracy: tolerance, "\(label) r", file: file, line: line)
    XCTAssertEqual(got.g, want.g, accuracy: tolerance, "\(label) g", file: file, line: line)
    XCTAssertEqual(got.b, want.b, accuracy: tolerance, "\(label) b", file: file, line: line)
  }

  // MARK: - Tests

  func testStitchDropsSharedBoundaryFrameAndReportsCounts() throws {
    let a = try gradientChunk(name: "a.mp4", frames: 17, from: red, to: blue)
    let b = try gradientChunk(name: "b.mp4", frames: 9, from: blue, to: green)  // frame 0 == A's last
    let out = tempDir.appendingPathComponent("out.mp4").path

    let result = try DirectorStitcher.stitch(
      chunkPaths: [a, b], expectedFrames: [17, 9], fps: 24, width: w, height: h,
      audio: nil, outputPath: out)

    XCTAssertEqual(result.outputPath, out)
    XCTAssertEqual(result.frameCount, 25, "17 + 9 - 1 shared boundary frame")
    XCTAssertEqual(result.durationSeconds, 25.0 / 24.0, accuracy: 1e-6)
    XCTAssertEqual(result.nominalFrameRate, 24, accuracy: 0.05)
    XCTAssertEqual(result.path, "reencode")

    let asset = AVURLAsset(url: URL(fileURLWithPath: out))
    let videoTracks = asset.tracks(withMediaType: .video)
    XCTAssertEqual(videoTracks.count, 1)
    XCTAssertEqual(asset.tracks(withMediaType: .audio).count, 0, "no audio requested")
    XCTAssertEqual(videoTracks[0].nominalFrameRate, 24, accuracy: 0.05)
    XCTAssertEqual(CMTimeGetSeconds(videoTracks[0].timeRange.duration), 25.0 / 24.0, accuracy: 0.002)
    XCTAssertEqual(Int(videoTracks[0].naturalSize.width), w)
    XCTAssertEqual(Int(videoTracks[0].naturalSize.height), h)

    let decoded = try decodeFrames(out)
    XCTAssertEqual(decoded.count, 25, "AVAssetReader decoded-frame count")
    // Faithful to what the stitcher decoded from the chunks (tight), and in
    // the neighbourhood of the nominal colours (loose: two H.264 hops drift
    // pure primaries through the YCbCr matrix by up to ~0.17).
    let srcA = try decodeFrames(a)
    let srcB = try decodeFrames(b)
    assertColour(decoded.first, srcA.first, tolerance: 0.08, "first frame vs chunk A frame 0")
    assertColour(decoded.last, srcB.last, tolerance: 0.08, "last frame vs chunk B last frame")
    assertColour(decoded.first, red, tolerance: 0.25, "first frame ~ red")
    assertColour(decoded.last, green, tolerance: 0.25, "last frame ~ green (B's last)")
  }

  func testStitchMuxesPCMAudioAt48k() throws {
    let a = try gradientChunk(name: "a.mp4", frames: 17, from: red, to: blue)
    let b = try gradientChunk(name: "b.mp4", frames: 9, from: blue, to: green)
    let out = tempDir.appendingPathComponent("out-audio.mp4").path

    // Deliberately longer than the video: the stitcher trims to the video.
    let result = try DirectorStitcher.stitch(
      chunkPaths: [a, b], expectedFrames: [17, 9], fps: 24, width: w, height: h,
      audio: tone(seconds: 3.0), outputPath: out)
    XCTAssertEqual(result.frameCount, 25)

    let asset = AVURLAsset(url: URL(fileURLWithPath: out))
    let audioTracks = asset.tracks(withMediaType: .audio)
    let videoTracks = asset.tracks(withMediaType: .video)
    XCTAssertEqual(videoTracks.count, 1)
    XCTAssertEqual(audioTracks.count, 1, "exactly one audio track")
    let fmt = audioTracks[0].formatDescriptions.first as! CMFormatDescription
    let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(fmt)!.pointee
    XCTAssertEqual(asbd.mSampleRate, 48000, accuracy: 1)
    XCTAssertEqual(asbd.mChannelsPerFrame, 2, "stereo")
    let videoDur = CMTimeGetSeconds(videoTracks[0].timeRange.duration)
    let audioDur = CMTimeGetSeconds(audioTracks[0].timeRange.duration)
    XCTAssertEqual(videoDur, 25.0 / 24.0, accuracy: 0.002)
    XCTAssertEqual(audioDur, videoDur, accuracy: 0.1, "AAC priming/padding tolerance; trimmed to the video")
  }

  func testStitchRejectsFrameCountMismatch() throws {
    let a = try gradientChunk(name: "a.mp4", frames: 17, from: red, to: blue)
    let b = try gradientChunk(name: "b.mp4", frames: 9, from: blue, to: green)
    let out = tempDir.appendingPathComponent("out-bad.mp4").path
    XCTAssertThrowsError(try DirectorStitcher.stitch(
      chunkPaths: [a, b], expectedFrames: [17, 17], fps: 24, width: w, height: h,
      audio: nil, outputPath: out)) { error in
      guard case DirectorError.stitchFailed(let m) = error else {
        return XCTFail("expected stitchFailed, got \(error)")
      }
      XCTAssertTrue(m.contains("frame count"), m)
    }
    XCTAssertFalse(FileManager.default.fileExists(atPath: out), "verified before any write")
  }

  func testStitchRejectsFpsMismatch() throws {
    let a = try gradientChunk(name: "a30.mp4", frames: 17, from: red, to: blue, fps: 30)
    let out = tempDir.appendingPathComponent("out-fps.mp4").path
    XCTAssertThrowsError(try DirectorStitcher.stitch(
      chunkPaths: [a], expectedFrames: [17], fps: 24, width: w, height: h,
      audio: nil, outputPath: out)) { error in
      guard case DirectorError.stitchFailed(let m) = error else {
        return XCTFail("expected stitchFailed, got \(error)")
      }
      XCTAssertTrue(m.contains("fps mismatch"), m)
    }
    XCTAssertFalse(FileManager.default.fileExists(atPath: out))
  }

  func testStitchRejectsMissingChunkAndBadArguments() throws {
    let out = tempDir.appendingPathComponent("out-missing.mp4").path
    XCTAssertThrowsError(try DirectorStitcher.stitch(
      chunkPaths: [tempDir.appendingPathComponent("nope.mp4").path], expectedFrames: [17],
      fps: 24, width: w, height: h, audio: nil, outputPath: out))
    XCTAssertThrowsError(try DirectorStitcher.stitch(
      chunkPaths: [], expectedFrames: [], fps: 24, width: w, height: h, audio: nil, outputPath: out))
    let a = try gradientChunk(name: "a.mp4", frames: 17, from: red, to: blue)
    XCTAssertThrowsError(try DirectorStitcher.stitch(
      chunkPaths: [a], expectedFrames: [17, 9], fps: 24, width: w, height: h, audio: nil, outputPath: out),
      "chunkPaths/expectedFrames length mismatch")
  }

  func testSingleChunkStitchIsAFaithfulCopy() throws {
    let a = try gradientChunk(name: "a.mp4", frames: 17, from: red, to: blue)
    let out = tempDir.appendingPathComponent("out-single.mp4").path
    let result = try DirectorStitcher.stitch(
      chunkPaths: [a], expectedFrames: [17], fps: 24, width: w, height: h,
      audio: tone(seconds: 17.0 / 24.0), outputPath: out)
    XCTAssertEqual(result.frameCount, 17, "n == 1 drops nothing")
    XCTAssertEqual(result.durationSeconds, 17.0 / 24.0, accuracy: 1e-6)

    let decoded = try decodeFrames(out)
    XCTAssertEqual(decoded.count, 17)
    let src = try decodeFrames(a)
    assertColour(decoded.first, src.first, tolerance: 0.08, "first frame vs source")
    assertColour(decoded.last, src.last, tolerance: 0.08, "last frame vs source")
    assertColour(decoded.first, red, tolerance: 0.25, "first frame ~ red")
    assertColour(decoded.last, blue, tolerance: 0.25, "last frame ~ blue")

    let asset = AVURLAsset(url: URL(fileURLWithPath: out))
    XCTAssertEqual(asset.tracks(withMediaType: .audio).count, 1, "audio muxed")
    XCTAssertEqual(asset.tracks(withMediaType: .video)[0].nominalFrameRate, 24, accuracy: 0.05)
  }

  func testStitchOverwritesExistingOutput() throws {
    let a = try gradientChunk(name: "a.mp4", frames: 9, from: red, to: blue)
    let out = tempDir.appendingPathComponent("out-existing.mp4").path
    try Data("stale".utf8).write(to: URL(fileURLWithPath: out))
    let result = try DirectorStitcher.stitch(
      chunkPaths: [a], expectedFrames: [9], fps: 24, width: w, height: h, audio: nil, outputPath: out)
    XCTAssertEqual(result.frameCount, 9)
    XCTAssertEqual(try decodeFrames(out).count, 9)
  }

  // MARK: - Tone transforms (Director chunk tone match)

  private let grey: RGB = (0.45, 0.45, 0.45)

  func testNilToneTransformsMatchesDefaultAndIdentityOutput() throws {
    let a = try gradientChunk(name: "a.mp4", frames: 9, from: red, to: blue)
    let b = try gradientChunk(name: "b.mp4", frames: 9, from: blue, to: green)
    let plain = tempDir.appendingPathComponent("plain.mp4").path
    let explicitNil = tempDir.appendingPathComponent("nil.mp4").path
    let identity = tempDir.appendingPathComponent("identity.mp4").path
    _ = try DirectorStitcher.stitch(
      chunkPaths: [a, b], expectedFrames: [9, 9], fps: 24, width: w, height: h, audio: nil, outputPath: plain)
    _ = try DirectorStitcher.stitch(
      chunkPaths: [a, b], expectedFrames: [9, 9], fps: 24, width: w, height: h, audio: nil,
      toneTransforms: nil, outputPath: explicitNil)
    _ = try DirectorStitcher.stitch(
      chunkPaths: [a, b], expectedFrames: [9, 9], fps: 24, width: w, height: h, audio: nil,
      toneTransforms: [.identity, .identity], outputPath: identity)
    let p = try decodeFrames(plain), n = try decodeFrames(explicitNil), i = try decodeFrames(identity)
    XCTAssertEqual(p.count, 17); XCTAssertEqual(n.count, 17); XCTAssertEqual(i.count, 17)
    assertColour(n.first, p.first, tolerance: 0.001, "nil first == default first")
    assertColour(n.last, p.last, tolerance: 0.001, "nil last == default last")
    assertColour(i.first, p.first, tolerance: 0.02, "identity first")
    assertColour(i.last, p.last, tolerance: 0.02, "identity last")
  }

  func testToneTransformShiftsOnlyItsChunk() throws {
    let a = try gradientChunk(name: "a.mp4", frames: 9, from: grey, to: grey)
    let b = try gradientChunk(name: "b.mp4", frames: 9, from: grey, to: grey)
    let out = tempDir.appendingPathComponent("toned.mp4").path
    let darken = ToneTransform(gain: [1, 1, 1], offset: [-40, -40, -40])
    let result = try DirectorStitcher.stitch(
      chunkPaths: [a, b], expectedFrames: [9, 9], fps: 24, width: w, height: h, audio: nil,
      toneTransforms: [.identity, darken], outputPath: out)
    XCTAssertEqual(result.frameCount, 17)
    let decoded = try decodeFrames(out)
    let src = try decodeFrames(a)
    assertColour(decoded.first, src.first, tolerance: 0.03, "chunk 0 untouched")
    let shift = CGFloat(40.0 / 255.0)
    assertColour(decoded.last, (src.last.r - shift, src.last.g - shift, src.last.b - shift),
                 tolerance: 0.03, "chunk 1 darkened by 40/255")
  }

  func testToneTransformsCountMustMatchChunks() throws {
    let a = try gradientChunk(name: "a.mp4", frames: 9, from: red, to: blue)
    let out = tempDir.appendingPathComponent("out-tone-count.mp4").path
    XCTAssertThrowsError(try DirectorStitcher.stitch(
      chunkPaths: [a], expectedFrames: [9], fps: 24, width: w, height: h, audio: nil,
      toneTransforms: [.identity, .identity], outputPath: out))
    XCTAssertFalse(FileManager.default.fileExists(atPath: out))
  }
}
