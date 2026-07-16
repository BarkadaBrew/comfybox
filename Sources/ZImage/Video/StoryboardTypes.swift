import Foundation

// MARK: - Storyboard spec (comfybox#237)

/// A storyboard is authored data — an ordered shot list — that the engine
/// executes end-to-end: per-shot i2v renders chained on the previous shot's
/// extracted last frame (face/angle/character locked), optional i2i inserts
/// on the anchor before animating (add what i2v can't invent), and final
/// assembly into one scene. Authoring surfaces (Krita's Storyboard Docker via
/// issue #247, an agent, a JSON template) all produce this same spec.
public struct StoryboardSpec: Sendable {
  public struct Insert: Sendable {
    public var prompt: String
    /// img2img denoise (0-1): how much the insert may change the anchor.
    /// Low values (~0.3-0.4) add the element while holding the frame.
    public var creativity: Double
    public var negativePrompt: String?
    public var maskPath: String?
    public var maskRegion: String?
    public var maskInvert: Bool
    public var maskGrow: Int
    public var maskFeather: Int
    public var seed: UInt64?
  }

  public struct Shot: Sendable {
    /// Motion prompt for the i2v render.
    public var prompt: String
    /// Shot duration in seconds. nil = one native chunk (~4s).
    public var durationS: Double?
    /// Explicit anchor image. Required for the first shot; later shots
    /// default to the previous shot's extracted last frame (the chain).
    public var anchorImage: String?
    /// Optional i2i step applied to the anchor BEFORE the i2v render.
    public var insert: Insert?
    public var negativePrompt: String?
    public var seed: UInt64?
  }

  public struct Output: Sendable {
    public var width: Int
    public var height: Int
    public var fps: Int
    public var path: String?

    public init(width: Int = 640, height: Int = 640, fps: Int = 24, path: String? = nil) {
      self.width = width
      self.height = height
      self.fps = fps
      self.path = path
    }
  }

  public var shots: [Shot]
  /// Assembly transitions — exactly shots-1 entries or empty for hard cuts
  /// (validated by MontageTimeline at assembly).
  public var transitions: [MontageTransition]
  public var output: Output
  /// LoRAs applied to every shot's i2v render.
  public var loras: [(path: String, scale: Float)]

  public init(
    shots: [Shot],
    transitions: [MontageTransition] = [],
    output: Output = Output(),
    loras: [(path: String, scale: Float)] = []
  ) {
    self.shots = shots
    self.transitions = transitions
    self.output = output
    self.loras = loras
  }
}

public enum StoryboardError: Error, LocalizedError, CustomStringConvertible {
  case noShots
  case firstShotNeedsAnchor
  case anchorNotFound(shot: Int, path: String)
  case shotFailed(shot: Int, stage: String, message: String)

  public var description: String {
    switch self {
    case .noShots: return "Storyboard needs at least one shot"
    case .firstShotNeedsAnchor:
      return "shots[0] requires anchor_image — the chain starts from an explicit frame"
    case .anchorNotFound(let shot, let path):
      return "shots[\(shot)]: anchor image not found: \(path)"
    case .shotFailed(let shot, let stage, let message):
      return "shots[\(shot)] \(stage) failed: \(message)"
    }
  }

  public var errorDescription: String? { description }
}

extension StoryboardSpec {
  /// Structural validation (anchor chain + montage transition math). Frame
  /// dims/fps and asset existence are checked at execution.
  public func validate() throws {
    guard !shots.isEmpty else { throw StoryboardError.noShots }
    guard shots[0].anchorImage?.isEmpty == false else {
      throw StoryboardError.firstShotNeedsAnchor
    }
    // Transitions must satisfy the montage overlap math against the shots'
    // nominal durations (nil duration = one native ~4s chunk).
    let nominal = shots.map { $0.durationS ?? 4.0 }
    _ = try MontageTimeline.resolve(segmentDurations: nominal, transitions: transitions)
  }
}
