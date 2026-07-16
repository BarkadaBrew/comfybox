import Foundation

#if canImport(AVFoundation) && canImport(CoreGraphics)
import AVFoundation
import CoreGraphics
import CoreImage
import ImageIO
import CoreVideo
#endif

// MARK: - Request types (comfybox#232)

/// One ordered montage segment: a still image (with optional ken-burns) or a
/// video clip (e.g. real i2v output), trimmed to `durationS`.
public struct MontageSegment: Sendable {
  public enum Kind: String, Sendable {
    case image
    case clip
  }

  /// Ken-burns parameters for image segments: scale ramps from `zoomStart` to
  /// `zoomEnd` while the image center offsets from `panStart` to `panEnd`
  /// (normalized output units; [0.1, 0] pans one-tenth of the width).
  public struct KenBurns: Sendable {
    public var zoomStart: Double
    public var zoomEnd: Double
    public var panStart: (x: Double, y: Double)
    public var panEnd: (x: Double, y: Double)

    public init(
      zoomStart: Double = 1.0, zoomEnd: Double = 1.15,
      panStart: (x: Double, y: Double) = (0, 0),
      panEnd: (x: Double, y: Double) = (0, 0)
    ) {
      self.zoomStart = zoomStart
      self.zoomEnd = zoomEnd
      self.panStart = panStart
      self.panEnd = panEnd
    }
  }

  public var kind: Kind
  public var path: String
  /// Segment duration in seconds. Required for images; for clips nil means
  /// "the clip's own duration".
  public var durationS: Double?
  public var kenBurns: KenBurns?

  public init(kind: Kind, path: String, durationS: Double? = nil, kenBurns: KenBurns? = nil) {
    self.kind = kind
    self.path = path
    self.durationS = durationS
    self.kenBurns = kenBurns
  }
}

public struct MontageTransition: Sendable, Equatable {
  public enum Kind: String, Sendable {
    case cut
    case fade
    case dissolve
  }

  public var kind: Kind
  public var durationS: Double

  public init(kind: Kind, durationS: Double = 0.5) {
    self.kind = kind
    self.durationS = kind == .cut ? 0 : durationS
  }
}

public enum MontageAspectPolicy: String, Sendable {
  /// Scale to fill the output and center-crop the overflow (no bars).
  case fillCrop = "fill_crop"
  /// Scale to fit inside the output and pad with black.
  case fitPad = "fit_pad"
}

public enum MontageError: Error, LocalizedError, CustomStringConvertible {
  case noSegments
  case badTransitionCount(segments: Int, transitions: Int)
  case badDuration(segment: Int, message: String)
  case transitionTooLong(boundary: Int, message: String)
  case assetNotFound(String)
  case assetUnreadable(String)
  case renderFailed(String)

  public var description: String {
    switch self {
    case .noSegments: return "Montage needs at least one segment"
    case .badTransitionCount(let s, let t):
      return "transitions must have segments-1 entries (\(s) segments need \(s - 1), got \(t)) or be omitted for all cuts"
    case .badDuration(let i, let msg): return "segment[\(i)]: \(msg)"
    case .transitionTooLong(let i, let msg): return "transition[\(i)]: \(msg)"
    case .assetNotFound(let p): return "Montage asset not found: \(p)"
    case .assetUnreadable(let p): return "Montage asset unreadable: \(p)"
    case .renderFailed(let msg): return "Montage render failed: \(msg)"
    }
  }

  public var errorDescription: String? { description }
}

// MARK: - Timeline (pure math, unit-tested)

/// Resolved montage timeline. Adjacent segments overlap by their transition's
/// duration, so `totalDuration = Σsegment − Σtransition`.
public struct MontageTimeline: Sendable {
  public struct Placed: Sendable, Equatable {
    public let index: Int
    public let start: Double
    public let duration: Double
    public var end: Double { start + duration }
  }

  /// What to draw at a given output time.
  public struct FrameState: Sendable {
    /// Primary segment index + local time within it.
    public let primary: (index: Int, localT: Double)
    /// During a transition: the incoming segment + progress 0→1 + kind.
    public let blend: (index: Int, localT: Double, progress: Double, kind: MontageTransition.Kind)?
  }

  public let placed: [Placed]
  public let transitions: [MontageTransition]
  public let totalDuration: Double

  /// Validate + place segments on the timeline.
  public static func resolve(
    segmentDurations: [Double],
    transitions rawTransitions: [MontageTransition]
  ) throws -> MontageTimeline {
    guard !segmentDurations.isEmpty else { throw MontageError.noSegments }
    var transitions = rawTransitions
    if transitions.isEmpty {
      transitions = Array(
        repeating: MontageTransition(kind: .cut, durationS: 0), count: segmentDurations.count - 1)
    }
    guard transitions.count == segmentDurations.count - 1 else {
      throw MontageError.badTransitionCount(
        segments: segmentDurations.count, transitions: rawTransitions.count)
    }
    for (i, d) in segmentDurations.enumerated() where !(d > 0) || !d.isFinite {
      throw MontageError.badDuration(segment: i, message: "duration must be > 0 (got \(d))")
    }
    // A transition eats into both adjacent segments — it can't be longer than
    // either, or a third segment would join the overlap.
    for (i, t) in transitions.enumerated() {
      guard t.durationS >= 0, t.durationS.isFinite else {
        throw MontageError.transitionTooLong(boundary: i, message: "duration must be ≥ 0")
      }
      let limit = min(segmentDurations[i], segmentDurations[i + 1])
      if t.durationS >= limit {
        throw MontageError.transitionTooLong(
          boundary: i,
          message: "duration \(t.durationS)s must be shorter than both adjacent segments (limit \(limit)s)")
      }
    }

    var placed: [Placed] = []
    var cursor = 0.0
    for (i, d) in segmentDurations.enumerated() {
      placed.append(Placed(index: i, start: cursor, duration: d))
      cursor += d
      if i < transitions.count { cursor -= transitions[i].durationS }
    }
    return MontageTimeline(placed: placed, transitions: transitions, totalDuration: cursor)
  }

  /// Frame state at output time `t` (clamped to the timeline).
  public func frameState(at t: Double) -> FrameState {
    let t = min(max(t, 0), totalDuration.nextDown)
    // The active segment is the LAST one whose window contains t.
    var active = placed[0]
    for p in placed where t >= p.start && t < p.end { active = p }
    // Are we inside the transition overlap leading INTO `active`?
    if active.index > 0 {
      let prev = placed[active.index - 1]
      let transition = transitions[active.index - 1]
      if transition.durationS > 0, t < prev.end {
        let progress = (t - active.start) / transition.durationS
        return FrameState(
          primary: (prev.index, t - prev.start),
          blend: (active.index, t - active.start, min(max(progress, 0), 1), transition.kind))
      }
    }
    return FrameState(primary: (active.index, t - active.start), blend: nil)
  }
}

// MARK: - Composer

#if canImport(AVFoundation) && canImport(CoreGraphics)

public struct MontageResult: Sendable {
  public let outputPath: String
  public let durationS: Double
  public let width: Int
  public let height: Int
  public let segmentCount: Int
  public let frameCount: Int
}

/// Native AVFoundation montage compositor (comfybox#232 Phase 1).
/// Streams frames straight into an AVAssetWriter — memory stays at a couple
/// of frames regardless of montage length, and no LTX-2/heavy-model gate is
/// involved (that is the point: editorial motion is memory-free).
public enum MontageComposer {

  public static func compose(
    segments: [MontageSegment],
    transitions: [MontageTransition],
    width: Int = 448,
    height: Int = 768,
    fps: Int = 30,
    aspectPolicy: MontageAspectPolicy = .fillCrop,
    outputPath: String
  ) throws -> MontageResult {
    guard !segments.isEmpty else { throw MontageError.noSegments }
    guard width > 0, height > 0, fps > 0 else {
      throw MontageError.renderFailed("invalid output geometry \(width)x\(height)@\(fps)")
    }

    // Open all sources first so validation errors surface before any file is
    // written (no partial output on failure).
    var sources: [SegmentSource] = []
    var durations: [Double] = []
    for (i, segment) in segments.enumerated() {
      guard FileManager.default.fileExists(atPath: segment.path) else {
        throw MontageError.assetNotFound(segment.path)
      }
      switch segment.kind {
      case .image:
        guard let duration = segment.durationS else {
          throw MontageError.badDuration(segment: i, message: "image segments require duration_s")
        }
        sources.append(try ImageSource(path: segment.path, kenBurns: segment.kenBurns))
        durations.append(duration)
      case .clip:
        let clip = try ClipSource(path: segment.path)
        let duration = segment.durationS.map { min($0, clip.assetDuration) } ?? clip.assetDuration
        guard duration > 0 else {
          throw MontageError.badDuration(segment: i, message: "clip has no usable duration")
        }
        sources.append(clip)
        durations.append(duration)
      }
    }

    let timeline = try MontageTimeline.resolve(segmentDurations: durations, transitions: transitions)
    let frameCount = max(1, Int((timeline.totalDuration * Double(fps)).rounded()))

    let outputURL = URL(fileURLWithPath: outputPath)
    try FileManager.default.createDirectory(
      at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try? FileManager.default.removeItem(at: outputURL)

    let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
    writer.shouldOptimizeForNetworkUse = true  // +faststart
    let videoSettings: [String: Any] = [
      AVVideoCodecKey: AVVideoCodecType.h264,
      AVVideoWidthKey: width,
      AVVideoHeightKey: height,
      AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: width * height * fps / 2,
        AVVideoMaxKeyFrameIntervalKey: fps,
        AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
      ] as [String: Any],
    ]
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: input,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
      ])
    writer.add(input)
    guard writer.startWriting() else {
      throw MontageError.renderFailed(writer.error?.localizedDescription ?? "startWriting failed")
    }
    writer.startSession(atSourceTime: .zero)

    let canvas = FrameCanvas(width: width, height: height, aspectPolicy: aspectPolicy)

    for n in 0..<frameCount {
      try autoreleasepool {
        let t = Double(n) / Double(fps)
        let state = timeline.frameState(at: t)

        try canvas.begin()
        let primary = sources[state.primary.index]
        let primaryDuration = durations[state.primary.index]

        if let blend = state.blend {
          let incoming = sources[blend.index]
          let incomingDuration = durations[blend.index]
          switch blend.kind {
          case .dissolve:
            try canvas.draw(source: primary, localT: state.primary.localT,
                            segmentDuration: primaryDuration, alpha: 1)
            try canvas.draw(source: incoming, localT: blend.localT,
                            segmentDuration: incomingDuration, alpha: CGFloat(blend.progress))
          case .fade:
            // Fade THROUGH black: outgoing darkens to black by the midpoint,
            // incoming rises from black after it.
            if blend.progress < 0.5 {
              try canvas.draw(source: primary, localT: state.primary.localT,
                              segmentDuration: primaryDuration,
                              alpha: CGFloat(1 - blend.progress * 2))
            } else {
              try canvas.draw(source: incoming, localT: blend.localT,
                              segmentDuration: incomingDuration,
                              alpha: CGFloat(blend.progress * 2 - 1))
            }
          case .cut:
            try canvas.draw(source: incoming, localT: blend.localT,
                            segmentDuration: incomingDuration, alpha: 1)
          }
        } else {
          try canvas.draw(source: primary, localT: state.primary.localT,
                          segmentDuration: primaryDuration, alpha: 1)
        }

        while !input.isReadyForMoreMediaData {
          Thread.sleep(forTimeInterval: 0.005)
        }
        guard let buffer = canvas.makePixelBuffer(pool: adaptor.pixelBufferPool) else {
          throw MontageError.renderFailed("pixel buffer creation failed at frame \(n)")
        }
        let pts = CMTimeMake(value: Int64(n), timescale: Int32(fps))
        guard adaptor.append(buffer, withPresentationTime: pts) else {
          throw MontageError.renderFailed(
            "append failed at frame \(n): \(writer.error?.localizedDescription ?? "unknown")")
        }
      }
    }

    input.markAsFinished()
    let semaphore = DispatchSemaphore(value: 0)
    writer.finishWriting { semaphore.signal() }
    semaphore.wait()
    guard writer.status == .completed else {
      try? FileManager.default.removeItem(at: outputURL)
      throw MontageError.renderFailed(writer.error?.localizedDescription ?? "finishWriting failed")
    }

    return MontageResult(
      outputPath: outputPath,
      durationS: timeline.totalDuration,
      width: width, height: height,
      segmentCount: segments.count,
      frameCount: frameCount)
  }
}

// MARK: - Frame sources

private protocol SegmentSource {
  /// Frame for local segment time `t`, with progress `t/duration` driving any
  /// ken-burns ramp. Returns the source image + its ken-burns transform state.
  func frame(at localT: Double, segmentDuration: Double) throws -> (image: CGImage, zoom: Double, pan: (x: Double, y: Double))
}

private final class ImageSource: SegmentSource {
  private let image: CGImage
  private let kenBurns: MontageSegment.KenBurns?

  init(path: String, kenBurns: MontageSegment.KenBurns?) throws {
    let url = URL(fileURLWithPath: path)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
      throw MontageError.assetUnreadable(path)
    }
    self.image = img
    self.kenBurns = kenBurns
  }

  func frame(at localT: Double, segmentDuration: Double) throws -> (image: CGImage, zoom: Double, pan: (x: Double, y: Double)) {
    guard let kb = kenBurns else { return (image, 1.0, (0, 0)) }
    let p = segmentDuration > 0 ? min(max(localT / segmentDuration, 0), 1) : 0
    let zoom = kb.zoomStart + (kb.zoomEnd - kb.zoomStart) * p
    let pan = (
      x: kb.panStart.x + (kb.panEnd.x - kb.panStart.x) * p,
      y: kb.panStart.y + (kb.panEnd.y - kb.panStart.y) * p
    )
    return (image, zoom, pan)
  }
}

/// Sequential clip reader. Output frame times are monotonic per segment
/// (the composer walks the montage front to back), so a single forward
/// AVAssetReader pass suffices; the last decoded frame is held and reused
/// until the requested time passes its timestamp.
private final class ClipSource: SegmentSource {
  let assetDuration: Double
  private let reader: AVAssetReader
  private let output: AVAssetReaderTrackOutput
  private var current: (image: CGImage, pts: Double)?
  private var finished = false
  private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
  private let path: String

  init(path: String) throws {
    self.path = path
    let asset = AVURLAsset(url: URL(fileURLWithPath: path))
    // Synchronous metadata loads are deprecated but fine here: local files,
    // called off the main thread, and the composer is synchronous by design.
    guard let track = asset.tracks(withMediaType: .video).first else {
      throw MontageError.assetUnreadable(path)
    }
    self.assetDuration = CMTimeGetSeconds(asset.duration)
    self.reader = try AVAssetReader(asset: asset)
    self.output = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
    reader.add(output)
    guard reader.startReading() else {
      throw MontageError.assetUnreadable(path)
    }
  }

  func frame(at localT: Double, segmentDuration: Double) throws -> (image: CGImage, zoom: Double, pan: (x: Double, y: Double)) {
    while !finished, current == nil || current!.pts < localT {
      guard let sample = output.copyNextSampleBuffer() else {
        finished = true
        break
      }
      guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
      let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
      let ci = CIImage(cvPixelBuffer: pixelBuffer)
      if let cg = ciContext.createCGImage(ci, from: ci.extent) {
        current = (cg, pts)
      }
    }
    guard let current else { throw MontageError.assetUnreadable(path) }
    return (current.image, 1.0, (0, 0))
  }
}

// MARK: - Canvas

/// Reusable CG drawing surface for one output frame.
private final class FrameCanvas {
  private let width: Int
  private let height: Int
  private let aspectPolicy: MontageAspectPolicy
  private var context: CGContext?

  init(width: Int, height: Int, aspectPolicy: MontageAspectPolicy) {
    self.width = width
    self.height = height
    self.aspectPolicy = aspectPolicy
  }

  func begin() throws {
    if context == nil {
      context = CGContext(
        data: nil, width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
    }
    guard let context else { throw MontageError.renderFailed("CGContext creation failed") }
    context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
  }

  func draw(source: SegmentSource, localT: Double, segmentDuration: Double, alpha: CGFloat) throws {
    guard let context else { throw MontageError.renderFailed("draw before begin") }
    guard alpha > 0 else { return }
    let (image, zoom, pan) = try source.frame(at: localT, segmentDuration: segmentDuration)

    let sw = Double(image.width)
    let sh = Double(image.height)
    let ow = Double(width)
    let oh = Double(height)
    let baseScale: Double
    switch aspectPolicy {
    case .fillCrop: baseScale = max(ow / sw, oh / sh)
    case .fitPad: baseScale = min(ow / sw, oh / sh)
    }
    let scale = baseScale * zoom
    let dw = sw * scale
    let dh = sh * scale
    let rect = CGRect(
      x: (ow - dw) / 2 + pan.x * ow,
      y: (oh - dh) / 2 - pan.y * oh,  // +y pans the view DOWN the image (CG origin is bottom-left)
      width: dw, height: dh)

    context.setAlpha(alpha)
    context.draw(image, in: rect)
    context.setAlpha(1)
  }

  func makePixelBuffer(pool: CVPixelBufferPool?) -> CVPixelBuffer? {
    guard let context, let cgImage = context.makeImage() else { return nil }
    var buffer: CVPixelBuffer?
    if let pool {
      CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
    }
    if buffer == nil {
      CVPixelBufferCreate(
        kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB,
        [kCVPixelBufferCGBitmapContextCompatibilityKey as String: true] as CFDictionary,
        &buffer)
    }
    guard let buffer else { return nil }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    guard let dest = CGContext(
      data: CVPixelBufferGetBaseAddress(buffer),
      width: width, height: height,
      bitsPerComponent: 8,
      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
    else { return nil }
    dest.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    return buffer
  }
}

#endif
