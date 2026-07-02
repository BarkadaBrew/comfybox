// ReplicateVideoProxy.swift — Proxies video generation requests to Replicate cloud API.
//
// Handles both I2V (Wan 2.2) and T2V (LTX 2.3) models.
// Manages an async job tracker for submit/poll pattern.
// Phase A: proxy-only. Phase B/C will add native MLX backends.

import Foundation
import Logging

/// Internal mutable state for a tracked video job.
final class VideoJob: @unchecked Sendable {
  let id: String
  let mode: VideoMode
  let startTime: Date
  var state: VideoJobState
  var model: String?
  var replicatePredictionId: String?
  var outputPath: String?
  var fileSizeBytes: Int?
  var videoDurationSeconds: Int?
  var error: String?
  var completedAt: Date?

  init(id: String, mode: VideoMode) {
    self.id = id
    self.mode = mode
    self.startTime = Date()
    self.state = .queued
  }

  /// Wall-clock duration in milliseconds from start to now (or completion).
  var elapsedMs: Int {
    let end = completedAt ?? Date()
    return Int(end.timeIntervalSince(startTime) * 1000)
  }

  /// Build a VideoJobStatus snapshot from current state.
  func toStatus() -> VideoJobStatus {
    let estimatedSec: Int? = (state == .queued || state == .processing)
      ? (mode == .i2v ? 120 : 180)
      : nil

    return VideoJobStatus(
      jobId: id,
      status: state,
      mode: mode,
      backend: "replicate",
      model: model,
      outputPath: outputPath,
      durationMs: (state == .succeeded || state == .failed) ? elapsedMs : nil,
      fileSizeBytes: fileSizeBytes,
      videoDurationSeconds: videoDurationSeconds,
      error: error,
      estimatedSeconds: estimatedSec,
      elapsedMs: elapsedMs,
      replicatePredictionId: replicatePredictionId
    )
  }
}

/// Proxies video generation requests to Replicate cloud API.
/// Thread-safe job tracker with submit/poll pattern.
public final class ReplicateVideoProxy: @unchecked Sendable {
  /// Replicate model identifiers.
  public static let t2vModel = "lightricks/ltx-2.3-fast"
  public static let i2vModel = "wan-video/wan-2.2-i2v-a14b"
  public static let t2vFallbackModel = "wan-video/wan-2.2-t2v-fast"

  private static let replicateBaseURL = "https://api.replicate.com/v1"

  private let apiKey: String?
  private let allowedOutputDirectory: String
  private let logger: Logger
  private let session: URLSession

  /// Thread-safe job storage.
  private let lock = NSLock()
  private var jobs: [String: VideoJob] = [:]

  public init(
    apiKey: String?,
    allowedOutputDirectory: String,
    logger: Logger = Logger(label: "z-image.video-proxy")
  ) {
    self.apiKey = apiKey
    self.allowedOutputDirectory = allowedOutputDirectory
    self.logger = logger

    let config = URLSessionConfiguration.ephemeral
    config.timeoutIntervalForRequest = 30
    config.timeoutIntervalForResource = 600
    self.session = URLSession(configuration: config)
  }

  /// Number of jobs currently being tracked (all states).
  public var activeJobCount: Int {
    lock.lock()
    defer { lock.unlock() }
    return jobs.count
  }

  // MARK: - Submit

  /// Submit a video generation job. Returns immediately with initial status.
  /// The actual generation runs asynchronously in a Task.
  public func submit(_ request: VideoGenerateRequest) async -> VideoJobStatus {
    // Check API key
    guard let apiKey = apiKey, !apiKey.isEmpty else {
      let jobId = generateJobId()
      let job = VideoJob(id: jobId, mode: request.mode)
      job.state = .failed
      job.error = "Replicate API key not configured. Set REPLICATE_API_TOKEN environment variable."
      job.completedAt = Date()

      lock.lock()
      jobs[jobId] = job
      lock.unlock()

      return job.toStatus()
    }

    // Enforce output path containment before spending anything on Replicate.
    if let outputPath = request.outputPath, !outputPath.isEmpty {
      do {
        _ = try WarmServerOutputPathValidator.resolveOutputPath(
          outputPath,
          allowedOutputDirectory: allowedOutputDirectory
        )
      } catch {
        let jobId = generateJobId()
        let job = VideoJob(id: jobId, mode: request.mode)
        job.state = .failed
        job.error = error.localizedDescription
        job.completedAt = Date()

        lock.lock()
        jobs[jobId] = job
        lock.unlock()

        return job.toStatus()
      }
    }

    let jobId = generateJobId()
    let job = VideoJob(id: jobId, mode: request.mode)
    job.model = request.mode == .i2v ? Self.i2vModel : Self.t2vModel

    lock.lock()
    jobs[jobId] = job
    lock.unlock()

    logger.info("Video job \(jobId) submitted: mode=\(request.mode.rawValue), model=\(job.model ?? "unknown")")

    // Run generation asynchronously
    Task { [weak self] in
      guard let self else { return }
      await self.runGeneration(jobId: jobId, request: request, apiKey: apiKey)
    }

    return job.toStatus()
  }

  // MARK: - Status

  /// Check job status. Returns nil if job not found.
  public func status(jobId: String) -> VideoJobStatus? {
    lock.lock()
    defer { lock.unlock() }
    return jobs[jobId]?.toStatus()
  }

  // MARK: - Prune

  /// Remove completed jobs older than the given TTL in seconds. Default: 1 hour.
  public func pruneCompletedJobs(ttlSeconds: TimeInterval = 3600) {
    let cutoff = Date().addingTimeInterval(-ttlSeconds)
    lock.lock()
    let toRemove = jobs.filter { (_, job) in
      guard let completedAt = job.completedAt else { return false }
      return completedAt < cutoff
    }.map(\.key)
    for key in toRemove {
      jobs.removeValue(forKey: key)
    }
    lock.unlock()

    if !toRemove.isEmpty {
      logger.info("Pruned \(toRemove.count) completed video job(s)")
    }
  }

  // MARK: - Private: Generation

  private func runGeneration(jobId: String, request: VideoGenerateRequest, apiKey: String) async {
    // Mark as processing
    updateJobState(jobId, state: .processing)

    do {
      if request.mode == .i2v {
        try await runI2V(jobId: jobId, request: request, apiKey: apiKey)
      } else {
        try await runT2V(jobId: jobId, request: request, apiKey: apiKey)
      }
    } catch {
      markJobFailed(jobId, error: error.localizedDescription)
    }
  }

  // MARK: - T2V

  private func runT2V(jobId: String, request: VideoGenerateRequest, apiKey: String) async throws {
    let duration = request.duration ?? 6
    let resolution = request.resolution ?? "720p"
    let aspectRatio = request.aspectRatio ?? "16:9"

    var input: [String: Any] = [
      "prompt": request.prompt,
      "duration": duration,
      "resolution": resolution,
      "aspect_ratio": aspectRatio,
      "fps": 25,
    ]
    if let seed = request.seed {
      input["seed"] = seed
    }

    logger.info("Video job \(jobId): submitting T2V to \(Self.t2vModel)")

    do {
      let predictionId = try await submitPrediction(model: Self.t2vModel, input: input, apiKey: apiKey)
      updateReplicatePredictionId(jobId, predictionId: predictionId)

      let outputURL = try await pollPrediction(predictionId: predictionId, apiKey: apiKey)
      try await downloadAndSave(jobId: jobId, url: outputURL, request: request)
    } catch let error as ReplicateError {
      // Content filter or other Replicate error: try fallback to Wan T2V
      if isContentFilterError(error) {
        logger.warning("Video job \(jobId): LTX content filter triggered, retrying with Wan T2V fallback")
        try await runT2VFallback(jobId: jobId, request: request, apiKey: apiKey)
      } else {
        throw error
      }
    }
  }

  private func runT2VFallback(jobId: String, request: VideoGenerateRequest, apiKey: String) async throws {
    updateJobModel(jobId, model: Self.t2vFallbackModel)

    var input: [String: Any] = [
      "prompt": request.prompt,
      "resolution": "480p",
      "num_frames": 81,
      "fps": 16,
      "go_fast": true,
      "sample_shift": 12,
      "disable_safety_checker": true,
    ]
    if let seed = request.seed {
      input["seed"] = seed
    }

    let predictionId = try await submitPrediction(model: Self.t2vFallbackModel, input: input, apiKey: apiKey)
    updateReplicatePredictionId(jobId, predictionId: predictionId)

    let outputURL = try await pollPrediction(predictionId: predictionId, apiKey: apiKey)
    try await downloadAndSave(jobId: jobId, url: outputURL, request: request)
  }

  // MARK: - I2V

  private func runI2V(jobId: String, request: VideoGenerateRequest, apiKey: String) async throws {
    guard let imagePath = request.imagePath else {
      throw ReplicateError.invalidRequest("image_path is required for I2V mode")
    }

    // Sanity-check the source: must be an existing regular file with image
    // magic bytes. Prevents arbitrary readable files from being base64-uploaded.
    if let imageError = Self.validateSourceImage(atPath: imagePath) {
      throw ReplicateError.invalidRequest(imageError)
    }

    // Read and base64-encode the source image
    let imageURL = URL(fileURLWithPath: imagePath)
    let imageData = try Data(contentsOf: imageURL)
    guard let mimeType = Self.imageMimeType(for: imageData) else {
      throw ReplicateError.invalidRequest("image_path is not a PNG or JPEG image: \(imagePath)")
    }
    let dataURI = "data:\(mimeType);base64,\(imageData.base64EncodedString())"

    let resolution = request.resolution ?? "480p"

    var input: [String: Any] = [
      "prompt": request.prompt,
      "image": dataURI,
      "resolution": resolution,
      "num_frames": 81,
      "fps": 16,
      "go_fast": true,
      "sample_shift": 12,
      "disable_safety_checker": true,
    ]
    if let seed = request.seed {
      input["seed"] = seed
    }

    logger.info("Video job \(jobId): submitting I2V to \(Self.i2vModel)")

    let predictionId = try await submitPrediction(model: Self.i2vModel, input: input, apiKey: apiKey)
    updateReplicatePredictionId(jobId, predictionId: predictionId)

    let outputURL = try await pollPrediction(predictionId: predictionId, apiKey: apiKey)
    try await downloadAndSave(jobId: jobId, url: outputURL, request: request)
  }

  // MARK: - Replicate HTTP

  /// Submit a prediction to Replicate. Returns the prediction ID.
  private func submitPrediction(model: String, input: [String: Any], apiKey: String) async throws -> String {
    let url = URL(string: "\(Self.replicateBaseURL)/models/\(model)/predictions")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

    let body: [String: Any] = ["input": input]
    request.httpBody = try JSONSerialization.data(withJSONObject: body)

    let (data, response) = try await session.data(for: request)

    guard let httpResponse = response as? HTTPURLResponse else {
      throw ReplicateError.networkError("Invalid response type")
    }

    guard httpResponse.statusCode == 201 else {
      let bodyText = String(data: data, encoding: .utf8) ?? ""
      throw ReplicateError.apiError(statusCode: httpResponse.statusCode, body: bodyText)
    }

    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let id = json["id"] as? String else {
      throw ReplicateError.invalidResponse("Missing prediction ID in response")
    }

    return id
  }

  /// Poll a prediction until it completes. Returns the output URL.
  private func pollPrediction(predictionId: String, apiKey: String) async throws -> String {
    let url = URL(string: "\(Self.replicateBaseURL)/predictions/\(predictionId)")!
    var request = URLRequest(url: url)
    request.httpMethod = "GET"
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

    let maxAttempts = 200  // 200 * 3s = 10 minutes max
    for _ in 0..<maxAttempts {
      let (data, _) = try await session.data(for: request)

      guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let status = json["status"] as? String else {
        throw ReplicateError.invalidResponse("Invalid prediction status response")
      }

      switch status {
      case "succeeded":
        // Extract output URL
        if let output = json["output"] as? String {
          return output
        }
        // Some models return output as an array
        if let outputArray = json["output"] as? [String], let first = outputArray.first {
          return first
        }
        throw ReplicateError.invalidResponse("No output URL in succeeded prediction")

      case "failed", "canceled":
        let error = json["error"] as? String ?? "Prediction \(status)"
        throw ReplicateError.predictionFailed(error)

      default:
        // Still processing — wait and retry
        try await Task.sleep(nanoseconds: 3_000_000_000)  // 3 seconds
      }
    }

    throw ReplicateError.timeout("Prediction timed out after \(maxAttempts * 3) seconds")
  }

  /// Download an MP4 from a URL and save it locally.
  private func downloadAndSave(jobId: String, url: String, request: VideoGenerateRequest) async throws {
    guard let downloadURL = URL(string: url) else {
      throw ReplicateError.invalidResponse("Invalid output URL: \(url)")
    }

    let (data, _) = try await session.data(from: downloadURL)

    // Resolve the output path, enforcing containment in the allowed output
    // directory (defense in depth — also validated at submit time).
    let outputPath: String
    if let requestedPath = request.outputPath, !requestedPath.isEmpty {
      outputPath = try WarmServerOutputPathValidator
        .resolveOutputPath(requestedPath, allowedOutputDirectory: allowedOutputDirectory)
        .path
    } else {
      outputPath = generateOutputPath(mode: request.mode)
    }

    // Ensure parent directory exists
    let outputURL = URL(fileURLWithPath: outputPath)
    try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

    try data.write(to: outputURL)

    let fileSize = data.count
    let videoDuration = request.mode == .i2v ? 5 : (request.duration ?? 6)

    // Mark job as succeeded
    lock.lock()
    if let job = jobs[jobId] {
      job.state = .succeeded
      job.outputPath = outputPath
      job.fileSizeBytes = fileSize
      job.videoDurationSeconds = videoDuration
      job.completedAt = Date()
    }
    lock.unlock()

    logger.info("Video job \(jobId): succeeded — \(outputPath) (\(fileSize) bytes)")
  }

  // MARK: - Source Image Validation

  /// Validate that an I2V source image path points at an existing regular file
  /// whose contents start with PNG or JPEG magic bytes. Returns an error
  /// message, or nil if valid. Guards against uploading arbitrary readable
  /// files (or blocking on non-regular files) via the base64 data URI.
  static func validateSourceImage(atPath path: String) -> String? {
    guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
          (attributes[.type] as? FileAttributeType) == .typeRegular else {
      return "image_path not found or not a regular file: \(path)"
    }
    guard let handle = FileHandle(forReadingAtPath: path) else {
      return "image_path could not be read: \(path)"
    }
    defer { try? handle.close() }
    guard let header = (try? handle.read(upToCount: 8)) ?? nil,
          imageMimeType(for: header) != nil else {
      return "image_path is not a PNG or JPEG image: \(path)"
    }
    return nil
  }

  /// Detect the MIME type from image magic bytes. PNG and JPEG only.
  static func imageMimeType(for data: Data) -> String? {
    let bytes = [UInt8](data.prefix(8))
    let pngMagic: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    if bytes.count >= 8, Array(bytes[0..<8]) == pngMagic {
      return "image/png"
    }
    let jpegMagic: [UInt8] = [0xFF, 0xD8, 0xFF]
    if bytes.count >= 3, Array(bytes[0..<3]) == jpegMagic {
      return "image/jpeg"
    }
    return nil
  }

  // MARK: - Helpers

  private func generateJobId() -> String {
    let timestamp = Int(Date().timeIntervalSince1970)
    let random = String(UInt32.random(in: 0...0xFFFFFF), radix: 16, uppercase: false)
    return "vid_\(timestamp)_\(random)"
  }

  private func generateOutputPath(mode: VideoMode) -> String {
    let timestamp = Int(Date().timeIntervalSince1970)
    let slug = mode == .i2v ? "i2v" : "t2v"
    let dir = (allowedOutputDirectory as NSString).appendingPathComponent("video")
    return (dir as NSString).appendingPathComponent("\(timestamp)_\(slug).mp4")
  }

  private func updateJobState(_ jobId: String, state: VideoJobState) {
    lock.lock()
    jobs[jobId]?.state = state
    lock.unlock()
  }

  private func updateReplicatePredictionId(_ jobId: String, predictionId: String) {
    lock.lock()
    jobs[jobId]?.replicatePredictionId = predictionId
    lock.unlock()
  }

  private func updateJobModel(_ jobId: String, model: String) {
    lock.lock()
    jobs[jobId]?.model = model
    lock.unlock()
  }

  private func markJobFailed(_ jobId: String, error: String) {
    lock.lock()
    if let job = jobs[jobId] {
      job.state = .failed
      job.error = error
      job.completedAt = Date()
    }
    lock.unlock()
    logger.error("Video job \(jobId): failed — \(error)")
  }

  private func isContentFilterError(_ error: ReplicateError) -> Bool {
    switch error {
    case .predictionFailed(let message):
      let lower = message.lowercased()
      return lower.contains("nsfw") || lower.contains("content filter") || lower.contains("safety")
    default:
      return false
    }
  }
}

// MARK: - ReplicateError

enum ReplicateError: Error, LocalizedError {
  case invalidRequest(String)
  case networkError(String)
  case apiError(statusCode: Int, body: String)
  case invalidResponse(String)
  case predictionFailed(String)
  case timeout(String)

  var errorDescription: String? {
    switch self {
    case .invalidRequest(let msg): return "Invalid request: \(msg)"
    case .networkError(let msg): return "Network error: \(msg)"
    case .apiError(let code, let body): return "Replicate API error (\(code)): \(body)"
    case .invalidResponse(let msg): return "Invalid response: \(msg)"
    case .predictionFailed(let msg): return "Prediction failed: \(msg)"
    case .timeout(let msg): return "Timeout: \(msg)"
    }
  }
}
