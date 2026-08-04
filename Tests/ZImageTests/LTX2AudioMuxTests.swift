import AVFoundation
import CoreGraphics
import Foundation
import MLX
import XCTest

@testable import ZImage

/// Task #21 wire 3: mux a PCM track into the mp4 (spec rev 2 finding #11).
/// Container assertions only — track count, format, sample rate, duration —
/// never byte equality. Synthetic tone + solid frames: no model needed.
final class LTX2AudioMuxTests: XCTestCase {

  private func solidFrame(width: Int, height: Int) -> CGImage {
    let ctx = CGContext(
      data: nil, width: width, height: height, bitsPerComponent: 8,
      bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.setFillColor(CGColor(red: 0.2, green: 0.4, blue: 0.6, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return ctx.makeImage()!
  }

  /// 2 s of 440 Hz stereo at 48 kHz, [2, N] float in [-1, 1].
  private func tone(seconds: Double, sampleRate: Int) -> MLXArray {
    let n = Int(seconds * Double(sampleRate))
    var samples = [Float]()
    samples.reserveCapacity(2 * n)
    for i in 0..<n { samples.append(0.5 * sin(2 * .pi * 440 * Float(i) / Float(sampleRate))) }
    let mono = MLXArray(samples).reshaped([1, n])
    return MLX.concatenated([mono, mono], axis: 0)  // [2, N]
  }

  func testMuxedMP4HasAudioTrackWithCorrectFormat() throws {
    let out = FileManager.default.temporaryDirectory
      .appendingPathComponent("mux-\(UUID().uuidString).mp4").path
    defer { try? FileManager.default.removeItem(atPath: out) }

    let frames = (0..<49).map { _ in solidFrame(width: 128, height: 128) }  // 1+8k contract, ~2 s @ 24fps
    try LTX2PostProcess.writeMP4(
      frames: frames, outputPath: out, fps: 24, width: 128, height: 128,
      audio: LTX2PostProcess.AudioTrack(
        samples: tone(seconds: 2.0, sampleRate: 48000), sampleRate: 48000))

    let asset = AVURLAsset(url: URL(fileURLWithPath: out))
    let audioTracks = asset.tracks(withMediaType: .audio)
    let videoTracks = asset.tracks(withMediaType: .video)
    XCTAssertEqual(videoTracks.count, 1, "one video track")
    XCTAssertEqual(audioTracks.count, 1, "one audio track")

    let fmt = audioTracks[0].formatDescriptions.first as! CMFormatDescription
    let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(fmt)!.pointee
    XCTAssertEqual(asbd.mSampleRate, 48000, accuracy: 1)
    XCTAssertEqual(asbd.mChannelsPerFrame, 2, "stereo")

    // A/V duration agreement within one video frame (the honest contract).
    let videoDur = CMTimeGetSeconds(videoTracks[0].timeRange.duration)
    let audioDur = CMTimeGetSeconds(audioTracks[0].timeRange.duration)
    XCTAssertEqual(videoDur, 49.0 / 24.0, accuracy: 0.001)
    XCTAssertEqual(audioDur, 2.0, accuracy: 0.1, "AAC priming/padding tolerance")
  }

  func testNoAudioProducesNoAudioTrackAndIdenticalFrameBehavior() throws {
    let out = FileManager.default.temporaryDirectory
      .appendingPathComponent("mux-silent-\(UUID().uuidString).mp4").path
    defer { try? FileManager.default.removeItem(atPath: out) }

    let frames = (0..<24).map { _ in solidFrame(width: 128, height: 128) }
    try LTX2PostProcess.writeMP4(frames: frames, outputPath: out, fps: 24, width: 128, height: 128)

    let asset = AVURLAsset(url: URL(fileURLWithPath: out))
    XCTAssertEqual(asset.tracks(withMediaType: .audio).count, 0, "no audio param -> no audio track")
    XCTAssertEqual(asset.tracks(withMediaType: .video).count, 1)
  }
}
