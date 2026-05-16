// VideoTypes.swift — Video generation types for ComfyBox MCP video tools.
//
// Defines the request/response types for video generation endpoints.
// Supports both Replicate proxy (Phase A) and future native MLX (Phase B/C).

import Foundation

// MARK: - VideoMode

/// Video generation mode.
public enum VideoMode: String, Codable, Sendable {
  /// Text-to-video: generate video from a text prompt.
  case t2v
  /// Image-to-video: animate a source image based on a motion prompt.
  case i2v
}

// MARK: - VideoJobState

/// State of a video generation job.
public enum VideoJobState: String, Codable, Sendable {
  case queued
  case processing
  case succeeded
  case failed
}

// MARK: - VideoGenerateRequest

/// Request payload for the `POST /v1/video/generate` endpoint.
public struct VideoGenerateRequest: Codable, Sendable {
  /// Text prompt describing the desired video content.
  public let prompt: String

  /// Absolute path to source image for I2V mode. Nil for T2V.
  public let imagePath: String?

  /// Video duration in seconds. T2V only: 6, 8, 10, 12, 14, 16, 18, or 20 (default: 6). Ignored for I2V.
  public let duration: Int?

  /// Output resolution: "480p", "720p", or "1080p".
  public let resolution: String?

  /// Aspect ratio: "16:9" or "9:16" (default: "16:9").
  public let aspectRatio: String?

  /// Random seed for reproducibility.
  public let seed: Int?

  /// Output file path for the .mp4. Must be within the allowed output directory.
  public let outputPath: String?

  /// Derived mode based on whether image_path is present.
  public var mode: VideoMode {
    imagePath != nil ? .i2v : .t2v
  }

  public init(
    prompt: String,
    imagePath: String? = nil,
    duration: Int? = nil,
    resolution: String? = nil,
    aspectRatio: String? = nil,
    seed: Int? = nil,
    outputPath: String? = nil
  ) {
    self.prompt = prompt
    self.imagePath = imagePath
    self.duration = duration
    self.resolution = resolution
    self.aspectRatio = aspectRatio
    self.seed = seed
    self.outputPath = outputPath
  }

  // MARK: - Validation

  /// Valid T2V durations in seconds.
  public static let validT2VDurations = [6, 8, 10, 12, 14, 16, 18, 20]

  /// Valid resolution values.
  public static let validResolutions = ["480p", "720p", "1080p"]

  /// Valid aspect ratio values.
  public static let validAspectRatios = ["16:9", "9:16"]

  /// Validate duration for the given mode. Returns error string or nil.
  public static func validateDuration(_ duration: Int, mode: VideoMode) -> String? {
    // I2V duration is fixed (~5s), ignore the parameter
    guard mode == .t2v else { return nil }
    guard validT2VDurations.contains(duration) else {
      return "Invalid duration \(duration). T2V supports: \(validT2VDurations.map(String.init).joined(separator: ", "))"
    }
    return nil
  }

  /// Validate resolution string. Returns error string or nil.
  public static func validateResolution(_ resolution: String) -> String? {
    guard validResolutions.contains(resolution) else {
      return "Invalid resolution '\(resolution)'. Supported: \(validResolutions.joined(separator: ", "))"
    }
    return nil
  }

  /// Validate aspect ratio string. Returns error string or nil.
  public static func validateAspectRatio(_ aspectRatio: String) -> String? {
    guard validAspectRatios.contains(aspectRatio) else {
      return "Invalid aspect_ratio '\(aspectRatio)'. Supported: \(validAspectRatios.joined(separator: ", "))"
    }
    return nil
  }

  /// Validate the full request. Returns an error string or nil if valid.
  public func validate() -> String? {
    if prompt.trimmingCharacters(in: .whitespaces).isEmpty {
      return "'prompt' is required and cannot be empty"
    }
    if let duration = duration {
      if let error = Self.validateDuration(duration, mode: mode) {
        return error
      }
    }
    if let resolution = resolution {
      if let error = Self.validateResolution(resolution) {
        return error
      }
    }
    if let aspectRatio = aspectRatio {
      if let error = Self.validateAspectRatio(aspectRatio) {
        return error
      }
    }
    return nil
  }
}

// MARK: - VideoJobStatus

/// Status of a video generation job. Returned by both `generate_video` and `video_status`.
public struct VideoJobStatus: Codable, Sendable {
  /// Unique job identifier.
  public let jobId: String

  /// Current job state.
  public let status: VideoJobState

  /// Video mode (t2v or i2v).
  public let mode: VideoMode?

  /// Backend that produced the video.
  public let backend: String

  /// Model identifier used for generation.
  public let model: String?

  /// Output file path (non-nil on success).
  public let outputPath: String?

  /// Total wall-clock time in milliseconds (set on completion).
  public let durationMs: Int?

  /// Output file size in bytes (set on success).
  public let fileSizeBytes: Int?

  /// Duration of the generated video in seconds (set on success).
  public let videoDurationSeconds: Int?

  /// Error message (set on failure).
  public let error: String?

  /// Estimated time remaining in seconds (set when queued/processing).
  public let estimatedSeconds: Int?

  /// Elapsed time in milliseconds since job submission.
  public let elapsedMs: Int?

  /// Replicate prediction ID (for proxy mode debugging).
  public let replicatePredictionId: String?

  public init(
    jobId: String,
    status: VideoJobState,
    mode: VideoMode? = nil,
    backend: String = "replicate",
    model: String? = nil,
    outputPath: String? = nil,
    durationMs: Int? = nil,
    fileSizeBytes: Int? = nil,
    videoDurationSeconds: Int? = nil,
    error: String? = nil,
    estimatedSeconds: Int? = nil,
    elapsedMs: Int? = nil,
    replicatePredictionId: String? = nil
  ) {
    self.jobId = jobId
    self.status = status
    self.mode = mode
    self.backend = backend
    self.model = model
    self.outputPath = outputPath
    self.durationMs = durationMs
    self.fileSizeBytes = fileSizeBytes
    self.videoDurationSeconds = videoDurationSeconds
    self.error = error
    self.estimatedSeconds = estimatedSeconds
    self.elapsedMs = elapsedMs
    self.replicatePredictionId = replicatePredictionId
  }
}
