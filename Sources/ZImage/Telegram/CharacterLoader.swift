// CharacterLoader.swift — Loads character descriptions from a static JSON config.
//
// JSON format at ~/.comfybox/characters.json:
//   {"kira": {"base": "...", "banana": "...", "avocado": "..."}, ...}
//
// Mode gating mirrors characters.ts:
//   neutral  -> base only
//   banana   -> base + banana (if present)
//   avocado  -> base + banana (if present) + avocado (if present)

import Foundation

public struct TieredCharacterDescription: Codable, Sendable {
  public let base: String
  public let banana: String?
  public let avocado: String?

  public init(base: String, banana: String? = nil, avocado: String? = nil) {
    self.base = base
    self.banana = banana
    self.avocado = avocado
  }
}

public final class CharacterLoader: @unchecked Sendable {
  private let characters: [String: TieredCharacterDescription]

  /// Load characters from JSON file. Default: ~/.comfybox/characters.json.
  /// Returns an empty loader if the file is missing or unparseable (graceful degradation).
  public init(configPath: String? = nil) {
    let path = configPath ?? ("~/.comfybox/characters.json" as NSString).expandingTildeInPath

    guard FileManager.default.fileExists(atPath: path),
          let data = FileManager.default.contents(atPath: path) else {
      self.characters = [:]
      return
    }

    let decoder = JSONDecoder()
    if let loaded = try? decoder.decode([String: TieredCharacterDescription].self, from: data) {
      // Normalize keys to lowercase
      var normalized: [String: TieredCharacterDescription] = [:]
      for (key, value) in loaded {
        normalized[key.lowercased()] = value
      }
      self.characters = normalized
    } else {
      self.characters = [:]
    }
  }

  /// Get character description gated by content mode.
  /// Returns nil if character not found.
  public func description(for name: String, mode: ContentModeManager.Mode) -> String? {
    guard let entry = characters[name.lowercased()] else { return nil }

    var desc = entry.base

    // banana tier: base + banana additions
    if (mode == .banana || mode == .avocado), let banana = entry.banana, !banana.isEmpty {
      desc += " " + banana
    }

    // avocado tier: base + banana + avocado additions
    if mode == .avocado, let avocado = entry.avocado, !avocado.isEmpty {
      desc += " " + avocado
    }

    return desc
  }

  /// List all available character names.
  public func allNames() -> [String] {
    return Array(characters.keys).sorted()
  }

  /// Check if a character exists.
  public func has(_ name: String) -> Bool {
    return characters[name.lowercased()] != nil
  }
}
