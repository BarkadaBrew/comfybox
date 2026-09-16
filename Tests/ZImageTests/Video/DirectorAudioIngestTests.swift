import AVFoundation
import CoreGraphics
import Foundation
import XCTest

@testable import ZImage

/// WP2d: imported-audio ingest. Synthetic media only — WAVs written with
/// AVAudioFile, a video-only mp4 from LTX2PostProcess.writeMP4. No models.
final class DirectorAudioIngestTests: XCTestCase {

  private var tempDir: URL!

  override func setUpWithError() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("director-ingest-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempDir)
  }

  // MARK: - Fixtures

  /// Write a 16-bit PCM WAV of a `hz` sine at amplitude 0.5.
  private func writeWAV(name: String, seconds: Double, sampleRate: Double, channels: UInt32, hz: Float = 440) throws -> String {
    let url = tempDir.appendingPathComponent(name)
    let settings: [String: Any] = [
      AVFormatIDKey: kAudioFormatLinearPCM,
      AVSampleRateKey: sampleRate,
      AVNumberOfChannelsKey: channels,
      AVLinearPCMBitDepthKey: 16,
      AVLinearPCMIsFloatKey: false,
      AVLinearPCMIsBigEndianKey: false,
      AVLinearPCMIsNonInterleaved: false,
    ]
    let file = try AVAudioFile(forWriting: url, settings: settings)
    let frames = AVAudioFrameCount(seconds * sampleRate)
    let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frames)!
    buffer.frameLength = frames
    for c in 0..<Int(channels) {
      let ch = buffer.floatChannelData![c]
      for i in 0..<Int(frames) {
        ch[i] = 0.5 * sin(2 * .pi * hz * Float(i) / Float(sampleRate))
      }
    }
    try file.write(from: buffer)
    return url.path
  }

  private func solidFrame(width: Int, height: Int) -> CGImage {
    let ctx = CGContext(
      data: nil, width: width, height: height, bitsPerComponent: 8,
      bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.setFillColor(CGColor(red: 0.2, green: 0.4, blue: 0.6, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
    return ctx.makeImage()!
  }

  private func writeVideoOnlyMP4(name: String, frames: Int = 9) throws -> String {
    let path = tempDir.appendingPathComponent(name).path
    let images = (0..<frames).map { _ in solidFrame(width: 64, height: 64) }
    try LTX2PostProcess.writeMP4(frames: images, outputPath: path, fps: 24, width: 64, height: 64)
    return path
  }

  private func rms(_ x: [Float]) -> Float {
    guard !x.isEmpty else { return 0 }
    return sqrt(x.reduce(0) { $0 + $1 * $1 } / Float(x.count))
  }

  // MARK: - decode

  func testDecodeResamples441kStereoTo48k() throws {
    let path = try writeWAV(name: "stereo441.wav", seconds: 2, sampleRate: 44100, channels: 2)
    let pcm = try DirectorAudioIngest.decode(path: path)
    XCTAssertEqual(pcm.sampleRate, 48000)
    XCTAssertEqual(pcm.left.count, pcm.right.count)
    XCTAssertEqual(pcm.frames, 96000, accuracy: 2, "2 s at 48 kHz (resampled from 44.1 kHz)")
    XCTAssertGreaterThan(rms(pcm.left), 0.1, "tone present on the left channel")
    XCTAssertGreaterThan(rms(pcm.right), 0.1, "tone present on the right channel")
  }

  func testDecodeUpmixesMono() throws {
    let path = try writeWAV(name: "mono.wav", seconds: 1, sampleRate: 48000, channels: 1)
    let pcm = try DirectorAudioIngest.decode(path: path)
    XCTAssertEqual(pcm.sampleRate, 48000)
    XCTAssertEqual(pcm.frames, 48000, accuracy: 2)
    XCTAssertGreaterThan(rms(pcm.left), 0.1)
    var maxDiff: Float = 0
    for i in 0..<pcm.frames { maxDiff = max(maxDiff, abs(pcm.left[i] - pcm.right[i])) }
    XCTAssertLessThan(maxDiff, 1e-4, "mono is duplicated to both channels")
  }

  func testDecodeHonoursMaxSeconds() throws {
    let path = try writeWAV(name: "long.wav", seconds: 2, sampleRate: 48000, channels: 2)
    let pcm = try DirectorAudioIngest.decode(path: path, maxSeconds: 0.5)
    XCTAssertEqual(pcm.frames, 24000, accuracy: 64, "bounded read")
  }

  func testDecodeVideoOnlyMP4Throws() throws {
    let path = try writeVideoOnlyMP4(name: "silent.mp4")
    XCTAssertThrowsError(try DirectorAudioIngest.decode(path: path))
  }

  func testDecodeMissingFileThrows() {
    XCTAssertThrowsError(try DirectorAudioIngest.decode(path: tempDir.appendingPathComponent("nope.wav").path))
  }

  // MARK: - probe

  func testProbeReportsDurationAndAudioTrack() throws {
    let path = try writeWAV(name: "probe.wav", seconds: 2, sampleRate: 44100, channels: 2)
    let probe = try XCTUnwrap(DirectorAudioIngest.probe(path: path))
    XCTAssertTrue(probe.hasAudioTrack)
    XCTAssertEqual(probe.durationSeconds, 2.0, accuracy: 0.01)
    XCTAssertEqual(probe.sampleRate, 44100)
    XCTAssertEqual(probe.channels, 2)
  }

  func testProbeOnVideoOnlyMP4HasNoAudioTrack() throws {
    let path = try writeVideoOnlyMP4(name: "probe-silent.mp4")
    let probe = try XCTUnwrap(DirectorAudioIngest.probe(path: path))
    XCTAssertFalse(probe.hasAudioTrack)
    XCTAssertGreaterThan(probe.durationSeconds, 0)
    XCTAssertNil(probe.sampleRate)
    XCTAssertNil(probe.channels)
  }

  func testProbeMissingFileIsNil() {
    XCTAssertNil(DirectorAudioIngest.probe(path: tempDir.appendingPathComponent("missing.wav").path))
  }

  func testProbeGarbageFileIsNil() throws {
    let url = tempDir.appendingPathComponent("garbage.wav")
    try Data(repeating: 0x42, count: 512).write(to: url)
    XCTAssertNil(DirectorAudioIngest.probe(path: url.path))
  }

  // MARK: - validator wiring

  /// The validator's default probe is the real ingest probe: a video-only
  /// file on an imported-mode timeline surfaces `audio_clip_undecodable`
  /// without any injected closure.
  func testValidatorDefaultProbeFlagsUndecodableClip() throws {
    let path = try writeVideoOnlyMP4(name: "clip.mp4")
    let png = tempDir.appendingPathComponent("k.png")
    try Data([0x89, 0x50, 0x4E, 0x47]).write(to: png)
    let timeline = DirectorTimeline(
      settings: .init(width: 576, height: 896, lengthFrames: 97),
      globalPrompt: "a test",
      keyframes: [.init(id: "k1", imagePath: png.path, frame: 0)],
      audioClips: [.init(id: "a1", audioPath: path, startFrame: 0, lengthFrames: 48)],
      audio: .init(mode: .imported))
    let v = DirectorValidator.validate(timeline)
    XCTAssertTrue(v.issues.contains { $0.code == "audio_clip_undecodable" }, "\(v.issues)")
    XCTAssertFalse(v.ok)
  }

  func testValidatorDefaultProbeAcceptsRealClip() throws {
    let path = try writeWAV(name: "ok.wav", seconds: 1, sampleRate: 48000, channels: 2)
    let png = tempDir.appendingPathComponent("k.png")
    try Data([0x89, 0x50, 0x4E, 0x47]).write(to: png)
    let timeline = DirectorTimeline(
      settings: .init(width: 576, height: 896, lengthFrames: 97),
      globalPrompt: "a test",
      keyframes: [.init(id: "k1", imagePath: png.path, frame: 0)],
      audioClips: [.init(id: "a1", audioPath: path, startFrame: 0, lengthFrames: 24)],
      audio: .init(mode: .imported))
    let v = DirectorValidator.validate(timeline)
    XCTAssertFalse(v.issues.contains { $0.code == "audio_clip_undecodable" }, "\(v.issues)")
    XCTAssertTrue(v.ok, "\(v.issues)")
  }
}
