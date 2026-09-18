// DirectorAudioIngest.swift — imported-audio ingest for the Director (WP2d).
//
// Two entry points:
//   probe(path:)   metadata only (AVURLAsset duration + audio track presence
//                  + ASBD rate/channels) — the validator's audio probe. Never
//                  decodes a sample, so /validate stays cheap.
//   decode(path:)  full decode through AVAssetReader to 48 kHz float PCM.
//                  AVFoundation resamples any input rate; channel handling is
//                  done here (mono duplicated to L/R, >2 channels take the
//                  first two) so a mono file yields bit-identical L == R.
//
// The timeline mixes at 48 kHz (DirectorAudioIngest.timelineSampleRate) to
// match the generated lane's native BigVGAN+BWE rate; the FDD's 44.1 kHz was
// the upstream project's choice and is recorded as a Phase 1 delta.

import Foundation

/// Stereo float PCM in [-1, 1], one array per channel, `sampleRate` Hz.
/// `frames` is the sample count per channel (min of the two if they differ).
public struct StereoPCM: Sendable, Equatable {
  public var left: [Float]
  public var right: [Float]
  public let sampleRate: Int

  public var frames: Int { min(left.count, right.count) }
  public var durationSeconds: Double { Double(frames) / Double(sampleRate) }

  public init(left: [Float], right: [Float], sampleRate: Int) {
    self.left = left
    self.right = right
    self.sampleRate = sampleRate
  }

  public static func silence(frames: Int, sampleRate: Int) -> StereoPCM {
    let z = [Float](repeating: 0, count: max(0, frames))
    return StereoPCM(left: z, right: z, sampleRate: sampleRate)
  }
}

public enum DirectorAudioIngestError: Error, LocalizedError, CustomStringConvertible, Equatable {
  case fileMissing(String)
  case noAudioTrack(String)
  case readerFailed(String, String)

  public var description: String {
    switch self {
    case .fileMissing(let p): return "audio file not found: \(p)"
    case .noAudioTrack(let p): return "no decodable audio track in \(p)"
    case .readerFailed(let p, let why): return "audio decode failed for \(p): \(why)"
    }
  }

  public var errorDescription: String? { description }
}

#if canImport(AVFoundation) && canImport(CoreGraphics)
import AVFoundation

public enum DirectorAudioIngest {

  /// The Director timeline's mixing rate. Matches the generated lane
  /// (LTX2AudioVAE.decodeToWaveform is 48 kHz stereo) so generated and
  /// imported tracks mux through the same AAC settings.
  public static let timelineSampleRate = 48000

  /// Metadata probe: duration and audio-track presence from the container,
  /// rate/channels from the first audio track's format description. No
  /// sample decode. `nil` when the file is missing or AVFoundation cannot
  /// open it at all (no tracks, no duration).
  public static func probe(path: String) -> AudioProbe? {
    guard FileManager.default.fileExists(atPath: path) else { return nil }
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    // Synchronous metadata loads are deprecated but appropriate here: local
    // files, called off the main thread by the validator / route handler.
    let duration = asset.duration
    let seconds = duration.isNumeric ? CMTimeGetSeconds(duration) : 0
    let audioTracks = asset.tracks(withMediaType: .audio)
    let anyTrack = !asset.tracks.isEmpty
    guard anyTrack || seconds > 0 else { return nil }  // unreadable container

    var sampleRate: Int?
    var channels: Int?
    if let track = audioTracks.first,
       let fmt = track.formatDescriptions.first,
       let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(fmt as! CMFormatDescription)?.pointee {
      if asbd.mSampleRate > 0 { sampleRate = Int(asbd.mSampleRate.rounded()) }
      if asbd.mChannelsPerFrame > 0 { channels = Int(asbd.mChannelsPerFrame) }
    }
    return AudioProbe(
      durationSeconds: seconds.isFinite ? max(0, seconds) : 0,
      hasAudioTrack: !audioTracks.isEmpty,
      sampleRate: sampleRate,
      channels: channels)
  }

  /// Decode the first audio track of `path` to float stereo PCM at
  /// `sampleRate` (AVFoundation resamples). `maxSeconds` bounds the read
  /// from the head of the file (preview / peak computation); nil reads all.
  public static func decode(
    path: String,
    sampleRate: Int = timelineSampleRate,
    maxSeconds: Double? = nil
  ) throws -> StereoPCM {
    guard FileManager.default.fileExists(atPath: path) else {
      throw DirectorAudioIngestError.fileMissing(path)
    }
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    guard let track = asset.tracks(withMediaType: .audio).first else {
      throw DirectorAudioIngestError.noAudioTrack(path)
    }
    let reader: AVAssetReader
    do {
      reader = try AVAssetReader(asset: asset)
    } catch {
      throw DirectorAudioIngestError.readerFailed(path, error.localizedDescription)
    }
    if let maxSeconds, maxSeconds > 0 {
      reader.timeRange = CMTimeRange(
        start: .zero,
        duration: CMTime(seconds: maxSeconds, preferredTimescale: CMTimeScale(sampleRate)))
    }
    // Native channel count, resampled to `sampleRate`, interleaved float32.
    // Channel mapping happens below in Swift so mono up-mix is exact and
    // multi-channel sources degrade predictably (first two channels).
    let settings: [String: Any] = [
      AVFormatIDKey: kAudioFormatLinearPCM,
      AVSampleRateKey: sampleRate,
      AVLinearPCMIsFloatKey: true,
      AVLinearPCMBitDepthKey: 32,
      AVLinearPCMIsNonInterleaved: false,
      AVLinearPCMIsBigEndianKey: false,
    ]
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
      throw DirectorAudioIngestError.readerFailed(path, "reader refused the LPCM output")
    }
    reader.add(output)
    guard reader.startReading() else {
      throw DirectorAudioIngestError.readerFailed(path, reader.error?.localizedDescription ?? "startReading failed")
    }

    var left: [Float] = []
    var right: [Float] = []
    if let seconds = maxSeconds ?? (asset.duration.isNumeric ? CMTimeGetSeconds(asset.duration) : nil),
       seconds.isFinite, seconds > 0 {
      let cap = Int(seconds * Double(sampleRate)) + sampleRate
      left.reserveCapacity(cap)
      right.reserveCapacity(cap)
    }
    var channels = 0
    var scratch: [Float] = []

    while let sample = output.copyNextSampleBuffer() {
      guard let block = CMSampleBufferGetDataBuffer(sample) else { continue }
      if channels == 0, let fmt = CMSampleBufferGetFormatDescription(sample),
         let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(fmt)?.pointee {
        channels = Int(asbd.mChannelsPerFrame)
      }
      guard channels > 0 else {
        throw DirectorAudioIngestError.readerFailed(path, "sample buffer without an audio format description")
      }
      let length = CMBlockBufferGetDataLength(block)
      let floatCount = length / MemoryLayout<Float>.size
      guard floatCount > 0 else { continue }
      if scratch.count < floatCount { scratch = [Float](repeating: 0, count: floatCount) }
      let status = scratch.withUnsafeMutableBytes { raw in
        CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: floatCount * MemoryLayout<Float>.size, destination: raw.baseAddress!)
      }
      guard status == kCMBlockBufferNoErr else {
        throw DirectorAudioIngestError.readerFailed(path, "CMBlockBufferCopyDataBytes status \(status)")
      }
      let frames = floatCount / channels
      switch channels {
      case 1:
        left.append(contentsOf: scratch[0..<frames])
        right.append(contentsOf: scratch[0..<frames])
      default:
        var i = 0
        for _ in 0..<frames {
          left.append(scratch[i])
          right.append(scratch[i + 1])
          i += channels
        }
      }
    }

    switch reader.status {
    case .completed:
      break
    case .cancelled:
      throw DirectorAudioIngestError.readerFailed(path, "reader cancelled")
    default:
      throw DirectorAudioIngestError.readerFailed(path, reader.error?.localizedDescription ?? "reader ended in status \(reader.status.rawValue)")
    }
    guard !left.isEmpty else {
      throw DirectorAudioIngestError.noAudioTrack(path)
    }
    // AVFoundation's rate converter drops its tail latency (measured: 17
    // samples short of 2 s when converting 44.1k -> 48k). Pad a SUB-50 ms
    // shortfall with silence up to the container's nominal duration so a
    // clip's sample count is round(duration * rate) regardless of source
    // rate; a larger shortfall is the file's own truth and is left alone.
    if asset.duration.isNumeric {
      var nominalSeconds = CMTimeGetSeconds(asset.duration)
      if let maxSeconds, maxSeconds > 0 { nominalSeconds = min(nominalSeconds, maxSeconds) }
      if nominalSeconds.isFinite, nominalSeconds > 0 {
        let nominal = Int((nominalSeconds * Double(sampleRate)).rounded())
        let shortfall = nominal - left.count
        if shortfall > 0, shortfall <= sampleRate / 20 {
          left.append(contentsOf: repeatElement(0, count: shortfall))
          right.append(contentsOf: repeatElement(0, count: shortfall))
        }
      }
    }
    return StereoPCM(left: left, right: right, sampleRate: sampleRate)
  }
}
#endif


// MARK: - Pauses (WP11: silent joins)

extension DirectorAudioIngest {

  /// Contiguous runs of near-silence in a voice track, as video-frame ranges.
  ///
  /// Todd's rule, 2026-09-18: **do not speak across a join.** A viewer reads
  /// mouth movement plus speech as synced even when the phonemes do not match —
  /// what breaks the illusion is a visible discontinuity, and the only place a
  /// sequence has one is a chunk boundary. Land every boundary in silence and
  /// there is nothing there to perceive as out of sync.
  ///
  /// Two details matter and both were got wrong first time:
  ///
  /// - The threshold is RELATIVE to the track's own peak. A quiet take and a
  ///   hot one both have pauses; an absolute dBFS floor finds none in the first
  ///   and everything in the second.
  /// - A pause is a RUN, not a frame. At 15% of peak, 65% of a normal take
  ///   reads as "quiet" because the dips between syllables qualify — which is
  ///   how a first pass concluded every boundary was already safe when two of
  ///   five were mid-word. Requiring `minRunFrames` of continuous quiet is what
  ///   distinguishes a pause from a consonant.
  public static func pauseRuns(
    _ pcm: StereoPCM, fps: Int, frameCount: Int,
    relativeThreshold: Float = 0.05, minRunFrames: Int = 8, floor: Float = 1e-4
  ) -> [Range<Int>] {
    guard fps > 0, frameCount > 0, pcm.frames > 0 else { return [] }
    let perFrame = max(1, pcm.sampleRate / fps)
    var energy: [Float] = []
    energy.reserveCapacity(frameCount)
    for frame in 0..<frameCount {
      let start = frame * perFrame
      guard start < pcm.frames else { energy.append(0); continue }
      let end = min(pcm.frames, start + perFrame)
      var sum: Float = 0
      for i in start..<end {
        let mono = (pcm.left[i] + pcm.right[i]) * 0.5
        sum += mono * mono
      }
      energy.append((sum / Float(end - start)).squareRoot())
    }
    let peak = energy.max() ?? 0
    guard peak > floor else { return [0..<frameCount] }
    let threshold = peak * relativeThreshold

    var runs: [Range<Int>] = []
    var start: Int? = nil
    for frame in 0..<frameCount {
      if energy[frame] <= threshold {
        if start == nil { start = frame }
      } else if let began = start {
        if frame - began >= minRunFrames { runs.append(began..<frame) }
        start = nil
      }
    }
    if let began = start, frameCount - began >= minRunFrames { runs.append(began..<frameCount) }
    return runs
  }

  /// Read a clip and report its pauses. An unreadable file yields NO pauses,
  /// so boundaries stay where the arithmetic put them — a worse join, never a
  /// failed render.
  public static func pauseRuns(
    atPath path: String, fps: Int, frameCount: Int
  ) -> [Range<Int>] {
    guard let pcm = try? decode(path: path) else { return [] }
    return pauseRuns(pcm, fps: fps, frameCount: frameCount)
  }
}
