// ContentModeManager.swift — Manages content mode state with JSON persistence.
//
// Modes: neutral (SFW), banana (suggestive), avocado (explicit).
// Persists to ~/.comfybox/content-mode.json via atomic temp file + rename.

import Foundation

public final class ContentModeManager: @unchecked Sendable {
  public enum Mode: String, Codable, Sendable, CaseIterable {
    case neutral
    case banana
    case avocado
  }

  private let configPath: String
  private let lock = NSLock()
  private var _current: Mode

  /// Initialize with optional custom config path. Defaults to ~/.comfybox/content-mode.json.
  public init(configPath: String? = nil) {
    let path = configPath ?? ("~/.comfybox/content-mode.json" as NSString).expandingTildeInPath
    self.configPath = path
    self._current = .neutral

    // Ensure directory exists
    let dir = (path as NSString).deletingLastPathComponent
    if !FileManager.default.fileExists(atPath: dir) {
      try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    }

    // Load persisted mode
    if let data = FileManager.default.contents(atPath: path),
       let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
       let modeString = json["mode"],
       let mode = Mode(rawValue: modeString) {
      self._current = mode
    }
  }

  /// Current mode (thread-safe read).
  public var current: Mode {
    lock.lock()
    defer { lock.unlock() }
    return _current
  }

  /// Set mode and persist to disk atomically.
  public func set(_ mode: Mode) {
    lock.lock()
    _current = mode
    lock.unlock()

    // Persist via temp file + rename for atomicity
    let json: [String: String] = ["mode": mode.rawValue]
    guard let data = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted]) else { return }

    let tempPath = configPath + ".tmp.\(UUID().uuidString)"
    FileManager.default.createFile(atPath: tempPath, contents: data)
    try? FileManager.default.removeItem(atPath: configPath)
    try? FileManager.default.moveItem(atPath: tempPath, toPath: configPath)
  }

  /// Display name for the current mode (for confirmation messages).
  public static func displayName(for mode: Mode) -> String {
    switch mode {
    case .neutral: return "Neutral (SFW)"
    case .banana: return "Banana (suggestive)"
    case .avocado: return "Avocado (explicit)"
    }
  }

  /// Emoji for the current mode.
  public static func emoji(for mode: Mode) -> String {
    switch mode {
    case .neutral: return "\u{1F34E}"  // red apple
    case .banana: return "\u{1F34C}"   // banana
    case .avocado: return "\u{1F951}"  // avocado
    }
  }
}
