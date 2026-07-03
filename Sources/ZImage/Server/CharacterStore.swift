// CharacterStore.swift — Character registry with JSON persistence (~/.comfybox/characters.json).
//
// Swift port of the Coffee Shop image service `src/character-registry.ts`, extended to a
// feature-complete creative-layer model. A CharacterEntry carries a stable identity plus the
// render-shaping data the desktop app needs: tiered descriptions gated by content mode,
// default LoRAs (filename + scale), a reusable prompt snippet, trigger words, a negative
// prompt, and tags.
//
// (Named `CharacterEntry`, matching the TS interface, rather than `Character` — a plain
// `Character` type would collide with the Swift standard library's `Character` and become
// ambiguous in modules that import ZImage.)
//
// Persistence mirrors the ComfyBoxServerConfig house style: Codable with a tolerant
// `init(from:)` so partial / older files load with sensible defaults, and an atomic write
// under the ~/.comfybox home.
//
// On-disk format — coexistence note: the file `~/.comfybox/characters.json` is *already
// read* by the legacy `CharacterLoader` (Telegram bot) as an object keyed by lowercased
// name, whose values decode as `{ base, banana, avocado }` (base non-optional). To avoid a
// regression, CharacterStore writes the same object-keyed-by-name shape and always emits a
// `base` tier (falling back to `description`), so CharacterLoader keeps parsing the file.
// The richer CharacterEntry fields (id, defaultLoras, promptSnippet, …) are extra keys the
// legacy decoder ignores. Loading is tolerant of that legacy shape, this superset shape, and
// a plain JSON array.

import Foundation

/// A reference to a LoRA adapter applied by default for a character.
///
/// Mirrors the image-service `LoraReference` (`{ filename, scale }`). `filename` may be a
/// bare filename, a library id, or an absolute path — the render pipeline resolves it.
public struct CharacterLoraReference: Codable, Equatable, Sendable {
  public var filename: String
  public var scale: Double

  public init(filename: String, scale: Double = 1.0) {
    self.filename = filename
    self.scale = scale
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    filename = try c.decodeIfPresent(String.self, forKey: .filename) ?? ""
    scale = try c.decodeIfPresent(Double.self, forKey: .scale) ?? 1.0
  }
}

/// A named character preset. Tiered description fields (`base` / `banana` / `avocado`) let a
/// single character render safely across content modes: explicit anatomy in `avocado` never
/// leaks into a neutral (SFW) render.
///
/// The flat `description` field is kept for backward compatibility with legacy / runtime-
/// registered characters that predate the tiered fields; when `base` is present it takes
/// precedence for assembly (see ``resolvedDescription(for:)``).
public struct CharacterEntry: Codable, Equatable, Sendable, Identifiable {
  /// Entry categories: a person/subject vs. a reusable environment.
  public static let kindCharacter = "character"
  public static let kindScene = "scene"

  /// Stable identity. Defaults to a slug of `name` when not supplied.
  public var id: String
  public var name: String

  /// "character" (a subject) or "scene" (an environment/location). Legacy
  /// entries without the field are derived from their text: descriptions
  /// written as "environment: …" are scenes.
  public var kind: String

  /// Flat description (legacy / fallback). Also used when no tiered `base` is present.
  public var description: String

  /// SFW physical appearance — always included when assembling a description.
  public var base: String?
  /// Suggestive additions — appended in banana + avocado modes.
  public var banana: String?
  /// Explicit additions — appended in avocado mode only.
  public var avocado: String?

  /// LoRAs applied by default when rendering this character.
  public var defaultLoras: [CharacterLoraReference]
  /// A reusable prompt fragment injected when this character is selected.
  public var promptSnippet: String?
  /// Negative-prompt additions specific to this character.
  public var negativePrompt: String?
  /// LoRA trigger words to place early in the prompt (image-service parity).
  public var triggerWords: String?
  /// Freeform tags for filtering / grouping.
  public var tags: [String]

  /// Epoch-millisecond timestamps (parity with image-service StudioProject style).
  public var createdAt: Int64
  public var updatedAt: Int64

  public init(
    id: String? = nil,
    name: String,
    kind: String? = nil,
    description: String = "",
    base: String? = nil,
    banana: String? = nil,
    avocado: String? = nil,
    defaultLoras: [CharacterLoraReference] = [],
    promptSnippet: String? = nil,
    negativePrompt: String? = nil,
    triggerWords: String? = nil,
    tags: [String] = [],
    createdAt: Int64? = nil,
    updatedAt: Int64? = nil
  ) {
    let now = CharacterEntry.nowMillis()
    self.id = id ?? CharacterEntry.slug(name)
    self.name = name
    self.kind = kind ?? CharacterEntry.deriveKind(base: base, description: description)
    self.description = description
    self.base = base
    self.banana = banana
    self.avocado = avocado
    self.defaultLoras = defaultLoras
    self.promptSnippet = promptSnippet
    self.negativePrompt = negativePrompt
    self.triggerWords = triggerWords
    self.tags = tags
    self.createdAt = createdAt ?? now
    self.updatedAt = updatedAt ?? now
  }

  private enum CodingKeys: String, CodingKey {
    case id, name, kind, description, base, banana, avocado
    case defaultLoras, promptSnippet, negativePrompt, triggerWords, tags
    case createdAt, updatedAt
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    let name = try c.decodeIfPresent(String.self, forKey: .name) ?? ""
    self.name = name
    self.id = try c.decodeIfPresent(String.self, forKey: .id) ?? CharacterEntry.slug(name)
    self.description = try c.decodeIfPresent(String.self, forKey: .description) ?? ""
    self.base = try c.decodeIfPresent(String.self, forKey: .base)
    self.kind = try c.decodeIfPresent(String.self, forKey: .kind)
      ?? CharacterEntry.deriveKind(base: self.base, description: self.description)
    self.banana = try c.decodeIfPresent(String.self, forKey: .banana)
    self.avocado = try c.decodeIfPresent(String.self, forKey: .avocado)
    self.defaultLoras = try c.decodeIfPresent([CharacterLoraReference].self, forKey: .defaultLoras) ?? []
    self.promptSnippet = try c.decodeIfPresent(String.self, forKey: .promptSnippet)
    self.negativePrompt = try c.decodeIfPresent(String.self, forKey: .negativePrompt)
    self.triggerWords = try c.decodeIfPresent(String.self, forKey: .triggerWords)
    self.tags = try c.decodeIfPresent([String].self, forKey: .tags) ?? []
    let now = CharacterEntry.nowMillis()
    self.createdAt = try c.decodeIfPresent(Int64.self, forKey: .createdAt) ?? now
    self.updatedAt = try c.decodeIfPresent(Int64.self, forKey: .updatedAt) ?? now
  }

  /// Assemble the effective description for a content mode.
  ///
  /// When tiered fields are present, builds `base` (+ `banana` in banana/avocado)
  /// (+ `avocado` in avocado). When absent, returns the flat `description`.
  public func resolvedDescription(for mode: ContentModeManager.Mode = .neutral) -> String {
    guard let base, !base.isEmpty else { return description }
    var parts = [base]
    if (mode == .banana || mode == .avocado), let banana, !banana.isEmpty { parts.append(banana) }
    if mode == .avocado, let avocado, !avocado.isEmpty { parts.append(avocado) }
    return parts.joined(separator: " ")
  }

  // MARK: - Helpers

  /// Legacy entries carry no kind; ones written as "environment: …" are scenes.
  static func deriveKind(base: String?, description: String) -> String {
    let text = (base?.isEmpty == false ? base! : description)
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .lowercased()
    return text.hasPrefix("environment:") ? kindScene : kindCharacter
  }

  static func nowMillis() -> Int64 { Int64(Date().timeIntervalSince1970 * 1000) }

  /// Lowercase, hyphenated slug of a name, safe as a filename-free identifier.
  static func slug(_ name: String) -> String {
    let lowered = name.lowercased()
    var out = ""
    var lastDash = false
    for ch in lowered {
      if ch.isLetter || ch.isNumber {
        out.append(ch)
        lastDash = false
      } else if !lastDash && !out.isEmpty {
        out.append("-")
        lastDash = true
      }
    }
    while out.hasSuffix("-") { out.removeLast() }
    return out.isEmpty ? "character" : out
  }
}

/// Persistent, serialized character registry backed by `~/.comfybox/characters.json`.
///
/// Isolated as an actor so concurrent server routes can list / mutate without data races.
/// Lookups are case-insensitive on `id` (parity with the TS registry's lowercased keys).
public actor CharacterStore {
  private let path: URL
  private var characters: [String: CharacterEntry] = [:] // key: id.lowercased()

  /// `~/.comfybox/characters.json`.
  public static func defaultPath() -> URL {
    URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
      .appendingPathComponent(".comfybox/characters.json")
  }

  public init(path: URL = CharacterStore.defaultPath()) {
    self.path = path
    load()
  }

  // MARK: - Load / Save

  private func load() {
    guard let data = try? Data(contentsOf: path), !data.isEmpty else { return }
    let decoder = JSONDecoder()
    // Primary / legacy / superset format: object keyed by (lowercased) name.
    if let object = try? decoder.decode([String: CharacterEntry].self, from: data) {
      var loaded: [String: CharacterEntry] = [:]
      for (key, value) in object {
        // Backfill id/name from the map key when the value omitted them
        // (legacy CharacterLoader files carry only base/banana/avocado).
        var entry = value
        if entry.name.isEmpty { entry.name = key }
        if entry.id.isEmpty || entry.id == "character" { entry.id = CharacterEntry.slug(entry.name) }
        loaded[entry.id.lowercased()] = entry
      }
      characters = loaded
      return
    }
    // Tolerant fallback: a plain JSON array of characters.
    if let list = try? decoder.decode([CharacterEntry].self, from: data) {
      characters = Dictionary(list.map { ($0.id.lowercased(), $0) }, uniquingKeysWith: { _, new in new })
    }
  }

  private func persist() {
    let dir = path.deletingLastPathComponent()
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    // Object keyed by lowercased name, matching the legacy CharacterLoader read format.
    var object: [String: CharacterEntry] = [:]
    for character in characters.values {
      var entry = character
      // Guarantee a non-empty `base` tier so the legacy CharacterLoader — whose
      // TieredCharacterDescription.base is non-optional — can still parse the file.
      if entry.base == nil || entry.base?.isEmpty == true { entry.base = entry.description }
      object[character.name.lowercased()] = entry
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    guard let data = try? encoder.encode(object) else { return }
    try? data.write(to: path, options: .atomic)
  }

  // MARK: - CRUD

  /// All characters, sorted by name (case-insensitive).
  public func list() -> [CharacterEntry] {
    characters.values.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
  }

  /// Fetch a character by id (case-insensitive). Returns nil if absent.
  public func get(_ id: String) -> CharacterEntry? {
    characters[id.lowercased()]
  }

  /// Insert or update a character, then persist. `updatedAt` is refreshed; `createdAt` is
  /// preserved from any existing entry with the same id. Returns the stored character.
  @discardableResult
  public func upsert(_ character: CharacterEntry) -> CharacterEntry {
    var entry = character
    let key = entry.id.lowercased()
    if let existing = characters[key] {
      entry.createdAt = existing.createdAt
    }
    entry.updatedAt = CharacterEntry.nowMillis()
    characters[key] = entry
    persist()
    return entry
  }

  /// Delete a character by id (case-insensitive). Returns true if a character was removed.
  @discardableResult
  public func delete(_ id: String) -> Bool {
    let removed = characters.removeValue(forKey: id.lowercased()) != nil
    if removed { persist() }
    return removed
  }

  /// Insert an entry exactly as given (timestamps preserved, no persist).
  /// Test seam for migration tests that need controlled `createdAt`/`updatedAt`.
  func seedForTesting(_ character: CharacterEntry) {
    characters[character.id.lowercased()] = character
  }

  // MARK: - Legacy image-service migration

  /// Default location of the Coffee Shop image service's character registry.
  public static func legacyImageServicePath() -> URL {
    URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
      .appendingPathComponent(".coffeeshop/image-service/characters.json")
  }

  /// One-shot merge of the legacy Electron image-service registry (source of truth
  /// for hand-written character text). Safe to call on every startup:
  ///
  /// - Entries whose slug is absent locally are imported. A legacy flat
  ///   `description` becomes the `base` tier so mode-gated assembly works.
  /// - Local entries that were never edited (`updatedAt == createdAt`) get the
  ///   legacy text: `base`/`banana`/`avocado` win where the legacy entry has
  ///   them; local tiers the legacy entry lacks survive (e.g. an explicit tier
  ///   added only on this side). Tags are taken from legacy when local has none.
  /// - Entries edited locally since creation are left untouched, and updating
  ///   an entry bumps `updatedAt`, so a later run never claws back user edits.
  ///
  /// Returns the number of entries added or changed.
  @discardableResult
  public func importLegacyRegistry(at legacyURL: URL = CharacterStore.legacyImageServicePath()) -> Int {
    guard let data = try? Data(contentsOf: legacyURL), !data.isEmpty,
          let object = try? JSONDecoder().decode([String: CharacterEntry].self, from: data)
    else { return 0 }

    var changed = 0
    let now = CharacterEntry.nowMillis()

    for (key, rawLegacy) in object {
      var legacy = rawLegacy
      if legacy.name.isEmpty { legacy.name = key }
      let id = CharacterEntry.slug(legacy.name)

      // The canonical legacy text: explicit base tier, else the flat description.
      let legacyBase = (legacy.base?.isEmpty == false) ? legacy.base! : legacy.description
      guard !legacyBase.isEmpty else { continue }

      if var local = characters[id.lowercased()] {
        // Skip entries the user has edited since creation.
        guard local.updatedAt == local.createdAt else { continue }

        var mutated = false
        if local.base != legacyBase {
          local.base = legacyBase
          mutated = true
        }
        if let banana = legacy.banana, !banana.isEmpty, local.banana != banana {
          local.banana = banana
          mutated = true
        }
        if let avocado = legacy.avocado, !avocado.isEmpty, local.avocado != avocado {
          local.avocado = avocado
          mutated = true
        }
        if local.tags.isEmpty, !legacy.tags.isEmpty {
          local.tags = legacy.tags
          mutated = true
        }
        if mutated {
          local.updatedAt = now
          characters[id.lowercased()] = local
          changed += 1
        }
      } else {
        let entry = CharacterEntry(
          id: id,
          name: legacy.name,
          description: legacy.description,
          base: legacyBase,
          banana: legacy.banana,
          avocado: legacy.avocado,
          tags: legacy.tags,
          createdAt: now,
          updatedAt: now + 1  // > createdAt so a re-run treats it as settled
        )
        characters[id.lowercased()] = entry
        changed += 1
      }
    }

    if changed > 0 { persist() }
    return changed
  }
}
