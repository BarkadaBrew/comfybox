// DirectorStitcher.swift — concatenate rendered chunk mp4s into the final
// Director output (WP2d).
//
// Frame-accurate by construction: every chunk is DECODED (AVAssetReader,
// BGRA) and RE-ENCODED (AVAssetWriter H.264, the same codec keys and bitrate
// formula as LTX2PostProcess.writeMP4). No passthrough / stream-copy claim —
// H.264 sample passthrough across separately encoded chunks would need
// matching parameter sets and cannot drop a single boundary frame cleanly.
//
// Contract:
//   * every chunk's nominalFrameRate must be within 0.05 of `fps` and its
//     sample count must equal expectedFrames[k] — both verified BEFORE the
//     writer starts (a verify pass counts compressed samples without
//     decoding), so a bad chunk never leaves a half-written output;
//   * the FIRST decoded frame of every chunk k > 0 is dropped (it is the
//     carry-over copy of chunk k-1's last frame): frameCount = sum(expected)
//     - (n - 1), PTS = (globalIndex, fps);
//   * `audio` (48 kHz stereo PCM from DirectorAudioMixer) is muxed as AAC
//     through LTX2PostProcess.makeAudioSampleBuffers, trimmed to
//     ceil(frameCount / fps * sr) samples;
//   * `toneTransforms` (optional, one per chunk; see DirectorToneMatch)
//     applies chunk k's exposure match to every frame written from chunk k;
//     nil (the default) and identity entries write the decoded frames as-is;
//   * output dims are `width` x `height`; the writer scales appended buffers,
//     so a chunk encoded at other dims is resampled, not rejected — callers
//     should pass the chunks' real encoded dims (delivery downscale applies).
//
// Runs inside the orchestration Task: Task.checkCancellation() precedes the
// writer start so an interrupt during stitch cancels cleanly.

import Foundation

public struct StitchResult: Sendable, Equatable {
  public let outputPath: String
  public let frameCount: Int
  public let durationSeconds: Double
  public let nominalFrameRate: Float
  /// Always "reencode" in Phase 1 (recorded on the aggregate sidecar as
  /// `stitch_path`).
  public let path: String

  public init(outputPath: String, frameCount: Int, durationSeconds: Double, nominalFrameRate: Float, path: String = "reencode") {
    self.outputPath = outputPath
    self.frameCount = frameCount
    self.durationSeconds = durationSeconds
    self.nominalFrameRate = nominalFrameRate
    self.path = path
  }
}

#if canImport(AVFoundation) && canImport(CoreGraphics)
import AVFoundation
import CoreVideo
import MLX

public enum DirectorStitcher {

  public static let stitchPath = "reencode"
  /// Tolerance on a chunk track's nominalFrameRate vs the timeline fps.
  public static let fpsTolerance: Float = 0.05

  public static func stitch(
    chunkPaths: [String],
    expectedFrames: [Int],
    fps: Int,
    width: Int,
    height: Int,
    audio: StereoPCM?,
    bitsPerPixel: Double? = nil,
    toneTransforms: [ToneTransform]? = nil,
    outputPath: String
  ) throws -> StitchResult {
    guard !chunkPaths.isEmpty else { throw DirectorError.stitchFailed("no chunks to stitch") }
    guard chunkPaths.count == expectedFrames.count else {
      throw DirectorError.stitchFailed("chunkPaths (\(chunkPaths.count)) and expectedFrames (\(expectedFrames.count)) differ")
    }
    guard fps > 0, width > 0, height > 0 else {
      throw DirectorError.stitchFailed("invalid fps/width/height \(fps)/\(width)/\(height)")
    }
    if let toneTransforms, toneTransforms.count != chunkPaths.count {
      throw DirectorError.stitchFailed("toneTransforms (\(toneTransforms.count)) and chunkPaths (\(chunkPaths.count)) differ")
    }

    // MARK: Verify every chunk before touching the output.

    for (k, path) in chunkPaths.enumerated() {
      guard FileManager.default.fileExists(atPath: path) else {
        throw DirectorError.stitchFailed("chunk \(k) missing: \(path)")
      }
      let asset = AVURLAsset(url: URL(fileURLWithPath: path))
      guard let track = asset.tracks(withMediaType: .video).first else {
        throw DirectorError.stitchFailed("chunk \(k) has no video track: \(path)")
      }
      let rate = track.nominalFrameRate
      guard abs(rate - Float(fps)) <= fpsTolerance else {
        throw DirectorError.stitchFailed("fps mismatch chunk \(k): track \(rate) vs timeline \(fps)")
      }
      let count = try countSamples(asset: asset, track: track, chunk: k, path: path)
      guard count == expectedFrames[k] else {
        throw DirectorError.stitchFailed("frame count mismatch chunk \(k): decoded \(count), expected \(expectedFrames[k])")
      }
    }

    let totalFrames = expectedFrames.reduce(0, +) - (chunkPaths.count - 1)
    guard totalFrames > 0 else { throw DirectorError.stitchFailed("no frames after boundary drop") }

    // MARK: Audio buffers (pre-built; cheap).

    var audioBuffers: [CMSampleBuffer] = []
    var audioSettings: [String: Any]?
    if let audio, audio.frames > 0 {
      let sr = audio.sampleRate
      let keep = min(audio.frames, DirectorAudioMixer.sampleCount(frames: totalFrames, fps: fps, sampleRate: sr))
      let l = Array(audio.left[0..<keep])
      let r = Array(audio.right[0..<keep])
      let track = LTX2PostProcess.AudioTrack(samples: MLXArray(l + r).reshaped([2, keep]), sampleRate: sr)
      do {
        audioBuffers = try LTX2PostProcess.makeAudioSampleBuffers(track)
      } catch {
        throw DirectorError.stitchFailed("audio buffers: \(error)")
      }
      audioSettings = [
        AVFormatIDKey: kAudioFormatMPEG4AAC,
        AVSampleRateKey: sr,
        AVNumberOfChannelsKey: 2,
        // Same ceiling rule as writeMP4 (comfybox#334): ~4 bits/sample.
        AVEncoderBitRateKey: min(192_000, sr * 4),
      ]
    }

    // An interrupt that lands during the stitch cancels here, before any
    // bytes are written (the stitcher runs inside the orchestration Task).
    try Task.checkCancellation()

    // MARK: Writer (mirrors LTX2PostProcess.writeMP4's codec configuration).

    let outputURL = URL(fileURLWithPath: outputPath)
    try? FileManager.default.removeItem(at: outputURL)
    let writer: AVAssetWriter
    do {
      writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
    } catch {
      throw DirectorError.stitchFailed("AVAssetWriter: \(error.localizedDescription)")
    }
    let bpp = bitsPerPixel
      ?? Double(ProcessInfo.processInfo.environment["LTX2_VIDEO_BITS_PER_PX"] ?? "") ?? 0.5
    let videoSettings: [String: Any] = [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: Int(Double(width * height * fps) * bpp),
        AVVideoMaxKeyFrameIntervalKey: fps,
        AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
      ] as [String: Any],
    ]
    let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
    videoInput.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: videoInput,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
      ])
    guard writer.canAdd(videoInput) else { throw DirectorError.stitchFailed("writer refused the video input") }
    writer.add(videoInput)

    var audioInput: AVAssetWriterInput?
    if let audioSettings {
      let ai = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
      ai.expectsMediaDataInRealTime = false
      guard writer.canAdd(ai) else { throw DirectorError.stitchFailed("writer refused the audio input") }
      writer.add(ai)
      audioInput = ai
    }

    let source = SequentialFrameSource(paths: chunkPaths, expected: expectedFrames, toneTransforms: toneTransforms)

    guard writer.startWriting() else {
      throw DirectorError.stitchFailed(writer.error?.localizedDescription ?? "startWriting failed")
    }
    writer.startSession(atSourceTime: .zero)

    // Demand-driven two-queue append (see writeMP4 for why: with two inputs
    // the writer's interleaving window stalls one input until the other
    // catches up, so each input must be driven by its own ready callback).
    let group = DispatchGroup()
    let errorLock = NSLock()
    var appendError: DirectorError?
    func record(_ e: DirectorError) {
      errorLock.lock(); appendError = appendError ?? e; errorLock.unlock()
    }

    group.enter()
    var frameIndex = 0
    var videoDone = false
    let videoQueue = DispatchQueue(label: "comfybox.director.stitch.video")
    videoInput.requestMediaDataWhenReady(on: videoQueue) {
      if videoDone { return }
      func finish(_ error: DirectorError?) {
        if let error { record(error) }
        videoDone = true
        videoInput.markAsFinished()
        group.leave()
      }
      if writer.status != .writing {
        return finish(.stitchFailed(writer.error?.localizedDescription ?? "writer left .writing during video append"))
      }
      while videoInput.isReadyForMoreMediaData {
        let next: CVPixelBuffer?
        do {
          next = try source.next()
        } catch let e as DirectorError {
          return finish(e)
        } catch {
          return finish(.stitchFailed("\(error)"))
        }
        guard let pixelBuffer = next else {
          if frameIndex != totalFrames {
            return finish(.stitchFailed("wrote \(frameIndex) frames, expected \(totalFrames)"))
          }
          return finish(nil)
        }
        let pts = CMTime(value: CMTimeValue(frameIndex), timescale: CMTimeScale(fps))
        guard adaptor.append(pixelBuffer, withPresentationTime: pts) else {
          return finish(.stitchFailed(writer.error?.localizedDescription ?? "video append rejected at frame \(frameIndex)"))
        }
        frameIndex += 1
      }
    }

    if let audioInput {
      group.enter()
      var bufferIndex = 0
      var audioDone = false
      let audioQueue = DispatchQueue(label: "comfybox.director.stitch.audio")
      audioInput.requestMediaDataWhenReady(on: audioQueue) {
        if audioDone { return }
        func finish(_ error: DirectorError?) {
          if let error { record(error) }
          audioDone = true
          audioInput.markAsFinished()
          group.leave()
        }
        if writer.status != .writing {
          return finish(.stitchFailed(writer.error?.localizedDescription ?? "writer left .writing during audio append"))
        }
        while audioInput.isReadyForMoreMediaData {
          if bufferIndex >= audioBuffers.count { return finish(nil) }
          guard audioInput.append(audioBuffers[bufferIndex]) else {
            return finish(.stitchFailed(writer.error?.localizedDescription ?? "audio append rejected at buffer \(bufferIndex)"))
          }
          bufferIndex += 1
        }
      }
    }

    // Bounded like writeMP4: a wedged writer is an error, never a hung job.
    if group.wait(timeout: .now() + 600) == .timedOut {
      writer.cancelWriting()
      source.close()
      throw DirectorError.stitchFailed("stitch timed out after 600s (writer status \(writer.status.rawValue))")
    }
    source.close()

    let semaphore = DispatchSemaphore(value: 0)
    writer.finishWriting { semaphore.signal() }
    semaphore.wait()

    if let appendError {
      try? FileManager.default.removeItem(at: outputURL)
      throw appendError
    }
    if writer.status == .failed {
      try? FileManager.default.removeItem(at: outputURL)
      throw DirectorError.stitchFailed(writer.error?.localizedDescription ?? "writer failed")
    }

    // Report what the container actually says, not what we intended.
    let written = AVURLAsset(url: outputURL)
    let nominal = written.tracks(withMediaType: .video).first?.nominalFrameRate ?? Float(fps)
    return StitchResult(
      outputPath: outputPath,
      frameCount: frameIndex,
      durationSeconds: Double(frameIndex) / Double(fps),
      nominalFrameRate: nominal,
      path: stitchPath)
  }

  // MARK: - Helpers

  /// Count video samples WITHOUT decoding (passthrough output) — one H.264
  /// sample per frame. Passthrough also delivers zero-sample marker buffers
  /// (edit-list / discontinuity markers at the head and tail, 0 bytes,
  /// verified empirically on writeMP4 output: 4 markers around 17 frames),
  /// so the count is the SUM of numSamples, never one per buffer.
  private static func countSamples(asset: AVURLAsset, track: AVAssetTrack, chunk: Int, path: String) throws -> Int {
    let reader: AVAssetReader
    do {
      reader = try AVAssetReader(asset: asset)
    } catch {
      throw DirectorError.stitchFailed("chunk \(chunk) unreadable: \(error.localizedDescription)")
    }
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
    output.alwaysCopiesSampleData = false
    reader.add(output)
    guard reader.startReading() else {
      throw DirectorError.stitchFailed("chunk \(chunk) unreadable: \(reader.error?.localizedDescription ?? "startReading failed")")
    }
    var count = 0
    while let sample = output.copyNextSampleBuffer() {
      count += max(0, CMSampleBufferGetNumSamples(sample))
    }
    guard reader.status == .completed else {
      throw DirectorError.stitchFailed("chunk \(chunk) read ended in status \(reader.status.rawValue): \(reader.error?.localizedDescription ?? "")")
    }
    return count
  }

  /// Walks the chunks in order, decoding to BGRA and skipping the first
  /// frame of every chunk after the first, and applying chunk k's tone
  /// transform (when given and not identity) to its frames in place. Not
  /// thread-safe; driven from the single video append queue.
  private final class SequentialFrameSource {
    private let paths: [String]
    private let expected: [Int]
    private let toneTransforms: [ToneTransform]?
    private var chunk = 0
    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private var producedInChunk = 0

    init(paths: [String], expected: [Int], toneTransforms: [ToneTransform]? = nil) {
      self.paths = paths
      self.expected = expected
      self.toneTransforms = toneTransforms
    }

    /// The non-identity transform for the current chunk, if any.
    private var currentTransform: ToneTransform? {
      guard let toneTransforms, chunk < toneTransforms.count, !toneTransforms[chunk].isIdentity else { return nil }
      return toneTransforms[chunk]
    }

    func next() throws -> CVPixelBuffer? {
      while chunk < paths.count {
        if reader == nil { try open() }
        guard let output, let reader else { return nil }
        if let sample = output.copyNextSampleBuffer() {
          guard let pb = CMSampleBufferGetImageBuffer(sample) else { continue }
          producedInChunk += 1
          // Drop the carry-over copy of the previous chunk's last frame.
          if chunk > 0, producedInChunk == 1 { continue }
          if let transform = currentTransform { try Self.apply(transform, to: pb, chunk: chunk) }
          return pb
        }
        guard reader.status == .completed else {
          throw DirectorError.stitchFailed("chunk \(chunk) decode ended in status \(reader.status.rawValue): \(reader.error?.localizedDescription ?? "")")
        }
        guard producedInChunk == expected[chunk] else {
          throw DirectorError.stitchFailed("frame count mismatch chunk \(chunk) during decode: \(producedInChunk) vs \(expected[chunk])")
        }
        self.reader = nil
        self.output = nil
        chunk += 1
        producedInChunk = 0
      }
      return nil
    }

    private func open() throws {
      let asset = AVURLAsset(url: URL(fileURLWithPath: paths[chunk]))
      guard let track = asset.tracks(withMediaType: .video).first else {
        throw DirectorError.stitchFailed("chunk \(chunk) has no video track")
      }
      let r: AVAssetReader
      do { r = try AVAssetReader(asset: asset) } catch {
        throw DirectorError.stitchFailed("chunk \(chunk) unreadable: \(error.localizedDescription)")
      }
      let o = AVAssetReaderTrackOutput(
        track: track,
        outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
      // A transformed chunk mutates the decoded buffer, so it must own it.
      o.alwaysCopiesSampleData = currentTransform != nil
      r.add(o)
      guard r.startReading() else {
        throw DirectorError.stitchFailed("chunk \(chunk) unreadable: \(r.error?.localizedDescription ?? "startReading failed")")
      }
      reader = r
      output = o
      producedInChunk = 0
    }

    private static func apply(_ transform: ToneTransform, to pb: CVPixelBuffer, chunk: Int) throws {
      guard CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_32BGRA,
            CVPixelBufferLockBaseAddress(pb, []) == kCVReturnSuccess else {
        throw DirectorError.stitchFailed("chunk \(chunk) tone transform: decoded buffer is not a lockable BGRA buffer")
      }
      defer { CVPixelBufferUnlockBaseAddress(pb, []) }
      guard let base = CVPixelBufferGetBaseAddress(pb) else {
        throw DirectorError.stitchFailed("chunk \(chunk) tone transform: no base address")
      }
      transform.apply(
        buffer: base.assumingMemoryBound(to: UInt8.self),
        width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb),
        bytesPerRow: CVPixelBufferGetBytesPerRow(pb), format: .bgra)
    }

    func close() {
      if let reader, reader.status == .reading { reader.cancelReading() }
      reader = nil
      output = nil
    }
  }
}
#endif
