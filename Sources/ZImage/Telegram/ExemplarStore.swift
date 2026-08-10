import CryptoKit
import Foundation
import Logging

// Task #19 / Codex finding #5: exemplars are few-shot examples PROMOTED from
// rated renders — injected as separate user/assistant message pairs AFTER
// template resolution, never concatenated into the system prompt (that would
// open an instruction-injection boundary length caps can't close).

public struct PromptExemplar: Codable, Sendable {
  public let intent: String
  public let final: String
  public let mediaKind: String     // "video" | "image"
  public let contentMode: String   // neutral | banana | avocado
  public let sourceRenderId: String?
  public let addedAt: Date

  public init(
    intent: String, final: String, mediaKind: String, contentMode: String,
    sourceRenderId: String? = nil, addedAt: Date = Date()
  ) {
    self.intent = intent
    self.final = final
    self.mediaKind = mediaKind
    self.contentMode = contentMode
    self.sourceRenderId = sourceRenderId
    self.addedAt = addedAt
  }
}

public final class ExemplarStore: @unchecked Sendable {

  public static let shared = ExemplarStore()

  private let fileURL: URL
  private let queue = DispatchQueue(label: "comfybox.exemplars")
  private let logger = Logger(label: "z-image.exemplars")

  public init(directory: URL? = nil) {
    let dir = directory
      ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".comfybox")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    self.fileURL = dir.appendingPathComponent("prompt-exemplars.jsonl")
  }

  public func add(_ exemplar: PromptExemplar) {
    queue.sync {
      guard let data = try? JSONEncoder.iso.encode(exemplar),
            let line = String(data: data, encoding: .utf8) else { return }
      if let handle = try? FileHandle(forWritingTo: fileURL) {
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: Data((line + "\n").utf8))
      } else {
        try? (line + "\n").write(to: fileURL, atomically: true, encoding: .utf8)
      }
    }
  }

  public func all() -> [PromptExemplar] {
    queue.sync {
      guard let text = try? String(contentsOf: fileURL, encoding: .utf8) else { return [] }
      return text.split(separator: "\n").compactMap {
        try? JSONDecoder.iso.decode(PromptExemplar.self, from: Data($0.utf8))
      }
    }
  }

  /// Newest-first exemplars matching kind+mode, capped. Matching is exact on
  /// both axes: a video-avocado exemplar never leaks into an image-neutral
  /// optimization.
  public func matching(mediaKind: String, contentMode: String, limit: Int = 3) -> [PromptExemplar] {
    let kindPrefix = mediaKind.lowercased().hasPrefix("video") ? "video" : "image"
    return all()
      .filter { $0.mediaKind == kindPrefix && $0.contentMode == contentMode.lowercased() }
      .sorted { $0.addedAt > $1.addedAt }
      .prefix(limit).map { $0 }
  }

  /// Digest over the exemplar set actually injected — recorded alongside the
  /// template hash so the EFFECTIVE optimizer input is reconstructable.
  public static func setDigest(_ exemplars: [PromptExemplar]) -> String {
    guard !exemplars.isEmpty else { return "none" }
    let joined = exemplars.map { "\($0.intent)\u{1}\($0.final)" }.joined(separator: "\u{2}")
    return SHA256.hash(data: Data(joined.utf8))
      .map { String(format: "%02x", $0) }.joined().prefix(12).lowercased()
  }
}

extension JSONEncoder {
  fileprivate static let iso: JSONEncoder = {
    let e = JSONEncoder()
    e.dateEncodingStrategy = .iso8601
    return e
  }()
}

extension JSONDecoder {
  fileprivate static let iso: JSONDecoder = {
    let d = JSONDecoder()
    d.dateDecodingStrategy = .iso8601
    return d
  }()
}
