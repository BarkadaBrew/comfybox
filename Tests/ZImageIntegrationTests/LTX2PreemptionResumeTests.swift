// LTX2PreemptionResumeTests.swift -- #1479 Task 6: integration tests for
// bit-identity resume and always-resume.
//
// Drives the REAL generator-level API (`LTX2VideoGenerator.generatePreemptible`
// / `.resume(from:)`) against production LTX-2 weights -- no mocks. Requires
// GPU + weights on disk; skips cleanly when absent (CI, laptops without the
// checkpoint).
//
// ## Comparison method (read before touching assertions)
//
// `LTX2RenderOutcome.completed` carries `LTX2VideoResult` -- the
// GENERATOR-level result (outputPath/frameCount/durationSeconds/
// elapsedSeconds). It does NOT expose `finalLatents` or `audioLatents` --
// those names appeared in an earlier draft of this test and do not exist on
// the type (verified by reading `LTX2VideoGenerator.swift` directly). The
// generator's public surface has no accessor for the raw video/audio
// MLXArrays a completed render produced; only the pipeline-level
// `LTX2PipelineOutput` (reached via `generateT2VResumable` etc, NOT via
// `generatePreemptible`/`resume`) carries those, and the task's interface
// contract is explicit that Task 6 exercises the GENERATOR entry points, not
// the pipeline ones directly.
//
// So "bit-identical" here is verified at the only artifact the generator
// contract actually exposes: the written MP4. Byte-for-byte comparison of
// the MP4 file is NOT used (AVAssetWriter's H.264/AAC encode is not
// guaranteed bit-stable run-to-run for identical input frames -- container
// metadata and encoder internals are allowed to vary). Instead each file is
// re-read with `AVAssetReader` and:
//   - every decoded VIDEO frame's raw BGRA pixel buffer bytes are folded into
//     one running SHA-256 (`videoHashHex`), plus a frame count;
//   - every decoded AUDIO sample buffer's raw interleaved float32 PCM bytes
//     are folded into a second running SHA-256 (`audioHashHex`), plus a
//     sample count.
// Two renders are asserted identical by comparing these two digests exactly
// (`XCTAssertEqual`, no tolerance). This is decode-side pixel/sample
// equality, not encode-side byte equality, and is the strictest check
// available through the public contract without modifying reviewed source.

import XCTest
import Foundation
import CryptoKit
#if canImport(AVFoundation)
import AVFoundation
#endif
#if canImport(CoreMedia)
import CoreMedia
#endif
#if canImport(CoreVideo)
import CoreVideo
#endif
@testable import ZImage

#if canImport(AVFoundation) && canImport(CoreMedia) && canImport(CoreVideo)

final class LTX2PreemptionResumeTests: XCTestCase {

  // MARK: - Weights location
  //
  // Production points `--ltx2-weights` at `/Volumes/Bolt/Models/
  // pinkcherry-v18-distill06-int8` (see com.barkadabrew.comfybox.plist) --
  // the JoyAI-Echo-layout int8 monolith with the audio branch (2,962
  // audio_*/av_ca_* keys, verified via the safetensors header). That is the
  // checkpoint this suite needs (audio ON is the primary case).
  //
  // FINDING (2026-08-15): the xcodebuild-launched xctest process cannot read
  // ANY path under `/Volumes/Bolt` -- confirmed with a throwaway diagnostic
  // test: `FileManager.fileExists` (stat) succeeds, but both
  // `contentsOfDirectory` and `FileHandle(forReadingAtPath:)` (open) fail
  // with NSCocoaErrorDomain 257 / POSIX EPERM "Operation not permitted".
  // Plain `swift` scripts and Bash/Terminal read the same path fine, so this
  // is a TCC "Full Disk Access"/removable-volume grant that Xcode's ad-hoc
  // -signed, freshly-rebuilt DerivedData test binary does not carry --
  // matches this repo's known "TCC wedges boot after re-sign" pattern
  // (memory: comfybox-desktop-deploy, kroma-v02-shipped). EVERY LTX-2
  // transformer checkpoint on this Mac resolves (directly or via symlink) to
  // a path under `/Volumes/Bolt` -- there is no local-disk original.
  //
  // Fixing the TCC grant needs an interactive System Settings approval,
  // which this task must not trigger without warning Todd first. Instead:
  // Bash (which DOES have Bolt access) copies the one 34GB file this suite
  // needs to local disk before the run; `weightsDir` below points at that
  // local copy so `resolveWeightsFileURL()` never touches Bolt. See the
  // task report for the exact copy command and cleanup.
  static let weightsDir = "/tmp/ltx2-local-weights"
  static let transformerFile = "pinkcherry_v18_distill06_int8.safetensors"
  static let gemmaPath = "/Users/toddwalderman/LocalModels/gemma-3-12b-heretic-q8"

  static var weightsAvailable: Bool {
    FileManager.default.fileExists(atPath: (weightsDir as NSString).appendingPathComponent(transformerFile)) &&
    FileManager.default.fileExists(atPath: gemmaPath)
  }

  static let config = LTX2VideoGenerator.Configuration(
    weightsDir: weightsDir, gemmaPath: gemmaPath, transformerFile: transformerFile)

  /// One generator, shared across every test method in this file. The 34GB
  /// checkpoint + Gemma-3 text encoder load is what dominates wall time --
  /// `load()` is idempotent for the same LoRA/audio key (see
  /// `LTX2VideoGenerator.load`), so reusing one instance means the real
  /// weight load happens exactly ONCE for every test that does not
  /// deliberately test eviction. The eviction-survival test builds its own
  /// dedicated instance and releases it -- that is the one place a second
  /// full load is paid for, on purpose.
  static let warmGenerator = LTX2VideoGenerator(config: config)

  private func skipIfWeightsMissing() throws {
    guard Self.weightsAvailable else {
      throw XCTSkip("""
        LTX-2 weights or Gemma text encoder not found on disk \
        (\(Self.weightsDir) / \(Self.gemmaPath)) -- skipping #1479 preemption \
        integration tests.
        """)
    }
  }

  // MARK: - Request construction (smallest real config: 64x64/9 frames/8 steps,
  // the same tiny shape LTX2IntegrationTest.testFullPipelineWithTransformerWeights
  // already proves the pipeline accepts end-to-end. T2V (no init image) keeps
  // this to the audio-supported single-chunk path with none of I2V's
  // face-anchor/refine machinery in play -- exactly what #1479's checkpoint/
  // resume loop covers.)

  static func makeRequest(outputPath: String, seed: UInt64) -> LTX2VideoRequest {
    LTX2VideoRequest(
      prompt: "a single still-life candle burning in a quiet dark room, slow gentle camera drift",
      width: 64,
      height: 64,
      framesPerChunk: 9,
      steps: 8,
      seed: seed,
      outputPath: outputPath,
      audio: true
    )
  }

  private func tempPath(_ name: String) -> String {
    let unique = UUID().uuidString.prefix(8)
    return "/tmp/ltx2-preempt-\(unique)-\(name)"
  }

  // MARK: - Preemption-driving helpers

  /// Runs `generatePreemptible`, raising `signal` the instant the progress
  /// callback reports `raiseAfterStep` completed steps. The denoise loop
  /// checks `preemption.isRaised` at the TOP of the next step (see
  /// `LTX2Pipeline.denoisingLoop`), so this always yields with
  /// `stepIndex == raiseAfterStep` -- checked by every call site.
  private static func runUntilYield(
    _ gen: LTX2VideoGenerator, request: LTX2VideoRequest,
    raiseAfterStep: Int, signal: PreemptionSignal
  ) throws -> LTX2RenderOutcome {
    try gen.generatePreemptible(request) { _, _, step, _ in
      if step == raiseAfterStep { signal.raise() }
    }
  }

  /// Same idea, continuing from a checkpoint. `raiseAfterStep` counts the
  /// same absolute per-chunk step number the progress callback always
  /// reports, not steps-since-resume.
  private static func resumeUntilYield(
    _ gen: LTX2VideoGenerator, from state: LTX2ResumeState,
    raiseAfterStep: Int, signal: PreemptionSignal
  ) throws -> LTX2RenderOutcome {
    try gen.resume(from: state) { _, _, step, _ in
      if step == raiseAfterStep { signal.raise() }
    }
  }

  // MARK: - Media digest (see file header: the comparison method)

  struct MediaDigest: Equatable, CustomStringConvertible {
    let videoHashHex: String
    let frameCount: Int
    let audioHashHex: String?
    let audioByteCount: Int

    var description: String {
      "video=\(videoHashHex.prefix(12))…(\(frameCount)f) audio=\(audioHashHex?.prefix(12).description ?? "nil")(\(audioByteCount)B)"
    }
  }

  enum MediaDigestError: Error, LocalizedError {
    case noVideoTrack(String)
    case readerFailed(String)

    var errorDescription: String? {
      switch self {
      case .noVideoTrack(let p): return "No video track in \(p)"
      case .readerFailed(let why): return "AVAssetReader failed: \(why)"
      }
    }
  }

  private func mediaDigest(path: String) throws -> MediaDigest {
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    guard let videoTrack = asset.tracks(withMediaType: .video).first else {
      throw MediaDigestError.noVideoTrack(path)
    }

    // --- Video: decode every frame, hash raw BGRA bytes ---
    let videoReader = try AVAssetReader(asset: asset)
    let videoOutput = AVAssetReaderTrackOutput(
      track: videoTrack,
      outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    videoOutput.alwaysCopiesSampleData = false
    videoReader.add(videoOutput)
    guard videoReader.startReading() else {
      throw MediaDigestError.readerFailed("\(path) video: \(videoReader.error?.localizedDescription ?? "unknown")")
    }
    var videoHasher = SHA256()
    var frameCount = 0
    while let sample = videoOutput.copyNextSampleBuffer() {
      guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
      CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
      let height = CVPixelBufferGetHeight(pixelBuffer)
      let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
      if let base = CVPixelBufferGetBaseAddress(pixelBuffer) {
        videoHasher.update(data: Data(bytes: base, count: bytesPerRow * height))
      }
      CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
      frameCount += 1
    }
    guard videoReader.status == .completed else {
      throw MediaDigestError.readerFailed("\(path) video: reader ended in status \(videoReader.status.rawValue), error \(videoReader.error?.localizedDescription ?? "none")")
    }

    // --- Audio (if present): decode to raw interleaved float32 PCM, hash bytes ---
    var audioHashHex: String? = nil
    var audioByteCount = 0
    if let audioTrack = asset.tracks(withMediaType: .audio).first {
      let audioReader = try AVAssetReader(asset: asset)
      let audioSettings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVLinearPCMIsFloatKey: true,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsNonInterleaved: false,
        AVLinearPCMIsBigEndianKey: false,
      ]
      let audioOutput = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: audioSettings)
      audioReader.add(audioOutput)
      guard audioReader.startReading() else {
        throw MediaDigestError.readerFailed("\(path) audio: \(audioReader.error?.localizedDescription ?? "unknown")")
      }
      var audioHasher = SHA256()
      while let sample = audioOutput.copyNextSampleBuffer() {
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sample) else { continue }
        var length = 0
        var dataPointer: UnsafeMutablePointer<Int8>? = nil
        let status = CMBlockBufferGetDataPointer(
          blockBuffer, atOffset: 0, lengthAtOffsetOut: nil,
          totalLengthOut: &length, dataPointerOut: &dataPointer)
        if status == kCMBlockBufferNoErr, let dataPointer {
          audioHasher.update(data: Data(bytes: dataPointer, count: length))
          audioByteCount += length
        }
      }
      guard audioReader.status == .completed else {
        throw MediaDigestError.readerFailed("\(path) audio: reader ended in status \(audioReader.status.rawValue), error \(audioReader.error?.localizedDescription ?? "none")")
      }
      audioHashHex = audioHasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    return MediaDigest(
      videoHashHex: videoHasher.finalize().map { String(format: "%02x", $0) }.joined(),
      frameCount: frameCount,
      audioHashHex: audioHashHex,
      audioByteCount: audioByteCount)
  }

  // MARK: - Test 1: bit-identical resume, seeded, audio ON (production shape)

  func testBitIdenticalResumeSeededAV() throws {
    try skipIfWeightsMissing()
    let gen = Self.warmGenerator
    let seed: UInt64 = 4242

    // 1) Uninterrupted reference -- non-preemptible entry, no signal armed,
    //    so behaviour is byte-for-byte what generate() always was.
    gen.setPreemptionSignal(nil)
    gen.setTelemetry(nil)
    let refRequest = Self.makeRequest(outputPath: tempPath("av-ref.mp4"), seed: seed)
    let ref = try gen.generate(refRequest)
    let refDigest = try mediaDigest(path: ref.outputPath)
    XCTAssertNotNil(refDigest.audioHashHex,
      "reference render produced no audio track -- audio:true was not honored, nothing to compare")

    // 2) Preempted run: raise after 3 steps, resume once to completion.
    let signal = PreemptionSignal()
    gen.setPreemptionSignal(signal)
    let preemptRequest = Self.makeRequest(outputPath: tempPath("av-preempt.mp4"), seed: seed)
    let outcome = try Self.runUntilYield(gen, request: preemptRequest, raiseAfterStep: 3, signal: signal)
    guard case .yielded(let state) = outcome else {
      return XCTFail("expected a yield at step 3, got a direct completion")
    }
    XCTAssertEqual(state.stepIndex, 3)

    signal.clear()   // controller ruling: clear BEFORE resume, or it zero-progress spins.
    let resumed = try gen.resume(from: state)
    guard case .completed(let out) = resumed else {
      return XCTFail("expected completion after resume, got another yield")
    }

    let resumedDigest = try mediaDigest(path: out.outputPath)

    XCTAssertEqual(ref.frameCount, out.frameCount, "generator-reported frame count diverged")
    XCTAssertEqual(refDigest.frameCount, resumedDigest.frameCount, "decoded frame count diverged")
    XCTAssertEqual(refDigest.videoHashHex, resumedDigest.videoHashHex,
      "video frames are NOT bit-identical between the uninterrupted run and the preempt-at-step-3-then-resume run")
    XCTAssertEqual(refDigest.audioHashHex, resumedDigest.audioHashHex,
      "audio track is NOT bit-identical between the uninterrupted run and the preempt-at-step-3-then-resume run -- the checkpoint dropped avState")
  }

  // MARK: - Test 2: fingerprint-mismatch refusal

  func testResumeRefusesMismatchedFingerprint() throws {
    try skipIfWeightsMissing()
    let gen = Self.warmGenerator
    let signal = PreemptionSignal()
    gen.setPreemptionSignal(signal)
    gen.setTelemetry(nil)

    let request = Self.makeRequest(outputPath: tempPath("fingerprint-mismatch.mp4"), seed: 7)
    let outcome = try Self.runUntilYield(gen, request: request, raiseAfterStep: 1, signal: signal)
    guard case .yielded(var state) = outcome else {
      return XCTFail("expected a yield at step 1")
    }
    XCTAssertEqual(state.stepIndex, 1)
    signal.clear()

    state.configFingerprint = "not-the-same-config"
    XCTAssertThrowsError(try gen.resume(from: state)) { error in
      guard let resumeError = error as? LTX2ResumeError,
            case .configFingerprintMismatch = resumeError else {
        return XCTFail("expected LTX2ResumeError.configFingerprintMismatch, got \(error)")
      }
    }
  }

  // MARK: - Test 3: preempt TWICE on one render, with a real eviction between
  // the first yield and its resume (VideoGeneratorHolder.release()-style:
  // `generator?.unload(); generator = nil`, matching LTX2Preemption.swift's
  // own description of why the continuation rides in the checkpoint and not
  // on the generator instance).

  func testDoublePreemptSurvivesEvictionAndMatchesReference() throws {
    try skipIfWeightsMissing()
    let seed: UInt64 = 9191

    // 0) Uninterrupted reference on the shared warm generator.
    Self.warmGenerator.setPreemptionSignal(nil)
    Self.warmGenerator.setTelemetry(nil)
    let refRequest = Self.makeRequest(outputPath: tempPath("double-ref.mp4"), seed: seed)
    let ref = try Self.warmGenerator.generate(refRequest)
    let refDigest = try mediaDigest(path: ref.outputPath)
    XCTAssertNotNil(refDigest.audioHashHex, "reference render produced no audio track")

    // 1) First preemption at step 3, on a DEDICATED generator instance --
    //    the one we are about to evict.
    var evictable: LTX2VideoGenerator? = LTX2VideoGenerator(config: Self.config)
    let signal1 = PreemptionSignal()
    evictable!.setPreemptionSignal(signal1)
    let reqA = Self.makeRequest(outputPath: tempPath("double-a.mp4"), seed: seed)
    let outcome1 = try Self.runUntilYield(evictable!, request: reqA, raiseAfterStep: 3, signal: signal1)
    guard case .yielded(let state1) = outcome1 else {
      return XCTFail("expected first yield at step 3")
    }
    XCTAssertEqual(state1.stepIndex, 3)

    // 2) EVICT: exactly what VideoGeneratorHolder.release() does. The
    //    LTX2RenderContext riding in state1.context must survive this --
    //    that is the whole point of #1479's checkpoint design (evict
    //    weights, keep latents).
    evictable!.unload()
    evictable = nil

    // 3) Resume on a FRESH instance, re-wiring signal + telemetry from
    //    scratch (nothing carries over from the deallocated generator),
    //    and preempt AGAIN mid-resume at step 6.
    let freshGen = LTX2VideoGenerator(config: Self.config)
    let signal2 = PreemptionSignal()
    freshGen.setPreemptionSignal(signal2)
    freshGen.setTelemetry(LTX2PhaseTelemetry())
    signal1.clear()   // controller ruling: clear BEFORE resume.

    let outcome2 = try Self.resumeUntilYield(freshGen, from: state1, raiseAfterStep: 6, signal: signal2)
    guard case .yielded(let state2) = outcome2 else {
      return XCTFail("expected second yield at step 6")
    }
    XCTAssertEqual(state2.stepIndex, 6)
    XCTAssertGreaterThan(state2.stepIndex, state1.stepIndex,
      "second checkpoint must move the render FORWARD (forward-only unwind guard)")

    // 4) Resume to completion.
    signal2.clear()
    let final = try freshGen.resume(from: state2)
    guard case .completed(let out) = final else {
      return XCTFail("expected completion after the second resume")
    }

    let resumedDigest = try mediaDigest(path: out.outputPath)
    XCTAssertEqual(refDigest.frameCount, resumedDigest.frameCount, "decoded frame count diverged")
    XCTAssertEqual(refDigest.videoHashHex, resumedDigest.videoHashHex,
      "video frames are NOT bit-identical after a double-preempt + eviction round trip")
    XCTAssertEqual(refDigest.audioHashHex, resumedDigest.audioHashHex,
      "audio track is NOT bit-identical after a double-preempt + eviction round trip")
  }
}

#endif
