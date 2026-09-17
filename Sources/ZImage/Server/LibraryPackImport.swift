// LibraryPackImport.swift — reading a third-party creative library pack into
// the Creative Library (PRD docs/PRD-creative-library.md, L7).
//
// The first supported foreign format is `sickollie-creative-library-pack`
// (schema 3), a zip of `manifest.json`, `library/*.json` and `previews/**`.
// A pack is someone else's taste: we import what maps cleanly, keep what we
// cannot interpret as a raw payload, and REPORT the rest rather than dropping
// it silently.
//
// Rules that keep a reimport honest (PRD §7):
//   * Every imported item is `source: "pack:<pack id>"` and its pack-provided
//     collections land in `packCollections`, never in the user's own
//     `collections`. A later reimport can therefore reconcile membership
//     without touching what Todd filed himself.
//   * Ids are derived from the pack id plus the item's own id, so reimporting
//     the same pack updates in place instead of duplicating.
//   * `dryRun` answers "what would change" without writing anything.
//
// Placeholders: the pack writes bare ALL-CAPS tokens (`OUTFIT`, `NAME`,
// `BRAND`, `SCENE`) inside the body and records them in
// `placeholder_signature`. We rewrite those to our `{OUTFIT}` markers and
// declare a slot for each, so an imported template behaves exactly like one
// authored here.

import Foundation

public struct LibraryPackReport: Codable, Sendable, Equatable {
  public struct Skip: Codable, Sendable, Equatable {
    public let what: String
    public let reason: String
  }

  public var packId: String
  public var packName: String
  public var creator: String?
  public var format: String
  public var dryRun: Bool
  /// Kind -> how many items were created or updated.
  public var imported: [String: Int]
  public var collections: Int
  public var previews: Int
  public var skipped: [Skip]

  public init(
    packId: String, packName: String, creator: String?, format: String, dryRun: Bool,
    imported: [String: Int] = [:], collections: Int = 0, previews: Int = 0, skipped: [Skip] = []
  ) {
    self.packId = packId
    self.packName = packName
    self.creator = creator
    self.format = format
    self.dryRun = dryRun
    self.imported = imported
    self.collections = collections
    self.previews = previews
    self.skipped = skipped
  }

  public var total: Int { imported.values.reduce(0, +) }
}

public enum LibraryPackError: Error, LocalizedError, Equatable {
  case unreadable(String)
  case unsupportedFormat(String)

  public var errorDescription: String? {
    switch self {
    case .unreadable(let m): return m
    case .unsupportedFormat(let f):
      return "unsupported library pack format '\(f)' — expected sickollie-creative-library-pack"
    }
  }
}

public enum LibraryPackImporter {

  public static let supportedFormat = "sickollie-creative-library-pack"

  // MARK: - Entry points

  /// Import an unpacked pack directory (manifest.json + library/ + previews/).
  /// The zip form goes through ``importPack(at:into:dryRun:)``.
  @discardableResult
  public static func importDirectory(
    _ root: URL, into store: LibraryStore, dryRun: Bool = false
  ) throws -> LibraryPackReport {
    let manifestURL = root.appendingPathComponent("manifest.json")
    guard let manifestData = try? Data(contentsOf: manifestURL),
          let manifest = (try? JSONSerialization.jsonObject(with: manifestData)) as? [String: Any]
    else {
      throw LibraryPackError.unreadable("no readable manifest.json in \(root.lastPathComponent)")
    }
    let format = manifest["format"] as? String ?? "unknown"
    guard format == supportedFormat else { throw LibraryPackError.unsupportedFormat(format) }

    let packId = manifest["pack_id"] as? String ?? "soslibrary:unknown"
    var report = LibraryPackReport(
      packId: packId,
      packName: manifest["name"] as? String ?? "Imported pack",
      creator: manifest["creator"] as? String,
      format: format,
      dryRun: dryRun)

    let libraryDir = root.appendingPathComponent("library")
    /// A pack need not carry every table (this one ships no boards and no
    /// recipe collections), so an ABSENT file is silence. A file that exists
    /// but will not parse is a skip, because that is data we were meant to
    /// import and could not.
    func rows(_ file: String) -> [[String: Any]] {
      let url = libraryDir.appendingPathComponent(file)
      guard FileManager.default.fileExists(atPath: url.path) else { return [] }
      guard let data = try? Data(contentsOf: url),
            let array = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]]
      else {
        report.skipped.append(.init(what: file, reason: "unreadable or not a JSON array"))
        return []
      }
      return array
    }

    // Collected, then written ONCE at the end: a pack is ~1,400 items, and
    // persisting per item rewrites the whole file every time (quadratic — it
    // wedged the first real import of the reference pack).
    var pendingItems: [LibraryEntry] = []
    var pendingCollections: [LibraryCollection] = []

    // Collections first: items reference them by name.
    var collectionIdByName: [String: String] = [:]
    for file in ["component-collections.json", "prompt-showcase-collections.json",
                 "shareable-collections.json", "recipe-collections.json"] {
      for row in rows(file) {
        guard let name = row["name"] as? String, !name.isEmpty else { continue }
        let id = collectionId(packId: packId, name: name)
        collectionIdByName[name] = id
        let parentName = row["parent_name"] as? String
        let parent = (parentName?.isEmpty == false) ? collectionId(packId: packId, name: parentName!) : nil
        pendingCollections.append(LibraryCollection(
          id: id, name: name, color: row["color"] as? String, parentId: parent,
          displayOrder: report.collections, membership: "pack"))
        report.collections += 1
      }
    }

    func packCollectionIds(_ row: [String: Any]) -> [String] {
      var names: [String] = []
      for key in ["pack_collections", "collections"] {
        // The pack writes these either as JSON arrays or as stringified Python
        // lists, depending on which table they came from.
        if let array = row[key] as? [[String: Any]] {
          names.append(contentsOf: array.compactMap { $0["name"] as? String })
        } else if let text = row[key] as? String {
          names.append(contentsOf: namesFromStringifiedList(text))
        }
      }
      return names.compactMap { name in
        collectionIdByName[name] ?? {
          let id = collectionId(packId: packId, name: name)
          collectionIdByName[name] = id
          return id
        }()
      }
    }

    func add(_ item: LibraryEntry, what: String) {
      pendingItems.append(item)
      report.imported[item.kind.rawValue, default: 0] += 1
    }

    // Prompts: `template` rows carry slots; `prompt` rows are finished prompts,
    // which are components of kind "prompt" here.
    for row in rows("prompts.json") {
      guard let value = (row["value"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
            !value.isEmpty else {
        report.skipped.append(.init(what: "prompt \(row["source_prompt_id"] ?? "?")", reason: "empty value"))
        continue
      }
      let sourceId = row["source_prompt_id"] as? String ?? stableId(value)
      let isTemplate = (row["kind"] as? String) == "template"
      let name = displayName(row: row, fallback: value)
      var item = LibraryEntry(
        id: itemId(packId: packId, sourceId: sourceId),
        kind: isTemplate ? .template : .component,
        name: name,
        value: value,
        facets: facets(row["facets"]),
        packCollections: packCollectionIds(row),
        rating: intValue(row["rating"]) ?? 0,
        archived: boolValue(row["archived"]) ?? false,
        notes: (row["note"] as? String).flatMap { $0.isEmpty ? nil : $0 },
        previewRef: row["preview_ref"] as? String,
        source: "pack:\(packId)")
      if isTemplate {
        let tokens = placeholderTokens(row["placeholder_signature"], body: value)
        let rewritten = rewritePlaceholders(value, tokens: tokens)
        item.value = rewritten
        item.slots = tokens.sorted().map { LibrarySlot(id: $0, label: $0.capitalized) }
      }
      if let parent = row["primary_parent"] as? String, !parent.isEmpty {
        item.facets["parent", default: []].append(parent)
      }
      if let sub = row["primary_subcategory"] as? String, !sub.isEmpty {
        item.facets["subcategory", default: []].append(sub)
      }
      add(item, what: "prompt \(sourceId)")
    }

    // Components: scene fragments and outfits.
    for row in rows("components.json") {
      guard let value = (row["value"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
            !value.isEmpty else { continue }
      let sourceId = row["source_component_id"] as? String ?? stableId(value)
      let rawKind = (row["kind"] as? String) ?? "scene"
      let isOutfit = rawKind == "outfit"
      var item = LibraryEntry(
        id: itemId(packId: packId, sourceId: sourceId),
        kind: isOutfit ? .outfit : .component,
        name: shortName(value),
        value: value,
        packCollections: packCollectionIds(row),
        rating: intValue(row["rating"]) ?? 0,
        previewRef: row["preview_ref"] as? String,
        source: "pack:\(packId)")
      if !isOutfit { item.componentKind = rawKind }
      add(item, what: "component \(sourceId)")
    }

    // Wardrobe.
    for row in rows("wardrobe-items.json") {
      guard let value = (row["value"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
            !value.isEmpty else { continue }
      let sourceId = row["source_wardrobe_id"] as? String ?? stableId(value)
      let item = LibraryEntry(
        id: itemId(packId: packId, sourceId: sourceId),
        kind: .wardrobeItem,
        name: value,
        value: value,
        packCollections: packCollectionIds(row),
        rating: intValue(row["rating"]) ?? 0,
        previewRef: row["preview_ref"] as? String,
        source: "pack:\(packId)",
        category: row["category"] as? String,
        subtype: row["subtype"] as? String,
        itemType: row["item_type"] as? String)
      add(item, what: "wardrobe \(sourceId)")
    }

    // Recipes: a foreign node graph we do NOT pretend to reproduce. We keep the
    // payload verbatim and lift the fields that mean the same thing here.
    for row in rows("recipes.json") {
      let sourceId = row["source_recipe_id"] as? String ?? stableId("\(row["name"] ?? "")")
      let payload = row["payload"] as? [String: Any] ?? [:]
      var item = LibraryEntry(
        id: itemId(packId: packId, sourceId: sourceId),
        kind: .recipe,
        name: (row["name"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "Imported recipe",
        value: "",
        packCollections: packCollectionIds(row),
        notes: "Imported from \(format). Reference only: its node graph is not replayable here.",
        previewRef: row["preview_ref"] as? String,
        source: "pack:\(packId)",
        seedSource: payload["_sickollie_seed_source"] as? String)
      if let migration = payload["migration"] as? [String: Any],
         let loras = migration["loras"] as? [[String: Any]] {
        item.loras = loras.compactMap { (lora: [String: Any]) -> LoraReference? in
          guard let filename = lora["name"] as? String else { return nil }
          return LoraReference(filename: filename, scale: doubleValue(lora["strength"]) ?? 1.0)
        }
      }
      if let data = try? JSONSerialization.data(withJSONObject: payload),
         let text = String(data: data, encoding: .utf8) {
        item.rawPayload = text
      }
      add(item, what: "recipe \(sourceId)")
    }

    if !dryRun {
      try store.upsertCollections(pendingCollections)
      let (_, rejected) = try store.upsert(contentsOf: pendingItems)
      for (item, error) in rejected {
        report.imported[item.kind.rawValue, default: 0] -= 1
        report.skipped.append(
          .init(what: "\(item.kind.rawValue) \(item.id)", reason: error.localizedDescription))
      }
      report.imported = report.imported.filter { $0.value > 0 }
    }

    // Previews: content-addressed already, so a copy is idempotent.
    let previewsSource = root.appendingPathComponent("previews")
    if FileManager.default.fileExists(atPath: previewsSource.path) {
      report.previews = dryRun
        ? countFiles(previewsSource)
        : copyPreviews(from: previewsSource, to: store.previewsDirectory)
    }

    return report
  }

  /// Import a `.soslibrary` (or any zip in that shape). Extraction is
  /// in-process (``LibraryZip``): spawning `ditto` from inside the engine
  /// never returned, and a serving path should not depend on a subprocess.
  @discardableResult
  public static func importPack(
    at packURL: URL, into store: LibraryStore, dryRun: Bool = false
  ) throws -> LibraryPackReport {
    let temp = FileManager.default.temporaryDirectory
      .appendingPathComponent("library-pack-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: temp) }
    try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
    try unzip(packURL, to: temp)
    // Some packs wrap everything in a single top-level folder.
    let root: URL = {
      if FileManager.default.fileExists(atPath: temp.appendingPathComponent("manifest.json").path) {
        return temp
      }
      let children = (try? FileManager.default.contentsOfDirectory(
        at: temp, includingPropertiesForKeys: nil)) ?? []
      for child in children
      where FileManager.default.fileExists(
        atPath: child.appendingPathComponent("manifest.json").path) {
        return child
      }
      return temp
    }()
    return try importDirectory(root, into: store, dryRun: dryRun)
  }

  static func unzip(_ archive: URL, to destination: URL) throws {
    do {
      try LibraryZip.extract(archive, to: destination)
    } catch let error as LibraryZipError {
      throw LibraryPackError.unreadable(
        "could not unpack \(archive.lastPathComponent): \(error.localizedDescription)")
    }
  }

  // MARK: - Mapping helpers

  static func itemId(packId: String, sourceId: String) -> String {
    "\(packId)#\(sourceId)"
  }

  static func collectionId(packId: String, name: String) -> String {
    "\(packId)#collection:\(name.lowercased().replacingOccurrences(of: " ", with: "-"))"
  }

  /// The pack's `placeholder_signature` is `OUTFIT` or `NAME+OUTFIT+BRAND`.
  /// Anything it names that actually appears in the body becomes a slot.
  static func placeholderTokens(_ signature: Any?, body: String) -> Set<String> {
    let declared = (signature as? String ?? "")
      .split(whereSeparator: { $0 == "+" || $0 == "," })
      .map { $0.trimmingCharacters(in: .whitespaces).uppercased() }
      .filter { !$0.isEmpty }
    return Set(declared.filter { body.contains($0) })
  }

  /// `wearing OUTFIT,` -> `wearing {OUTFIT},`. Whole-token only, so a word
  /// that merely contains the token is left alone.
  static func rewritePlaceholders(_ body: String, tokens: Set<String>) -> String {
    var out = body
    for token in tokens.sorted() {
      let pattern = "(?<![A-Za-z0-9_{])\(NSRegularExpression.escapedPattern(for: token))(?![A-Za-z0-9_}])"
      guard let re = try? NSRegularExpression(pattern: pattern) else { continue }
      out = re.stringByReplacingMatches(
        in: out, range: NSRange(out.startIndex..., in: out), withTemplate: "{\(token)}")
    }
    return out
  }

  static func facets(_ raw: Any?) -> [String: [String]] {
    guard let dict = raw as? [String: Any] else { return [:] }
    var out: [String: [String]] = [:]
    for (axis, value) in dict {
      if let list = value as? [String] {
        out[axis] = list.filter { !$0.isEmpty }
      } else if let single = value as? String, !single.isEmpty {
        out[axis] = [single]
      }
    }
    return out.filter { !$0.value.isEmpty }
  }

  /// `"[{'name': 'Showcase Looks', ...}]"` — the pack stringifies some lists.
  static func namesFromStringifiedList(_ text: String) -> [String] {
    guard text.contains("'name'") || text.contains("\"name\"") else { return [] }
    let pattern = "['\"]name['\"]\\s*:\\s*['\"]([^'\"]+)['\"]"
    guard let re = try? NSRegularExpression(pattern: pattern) else { return [] }
    let ns = text as NSString
    return re.matches(in: text, range: NSRange(location: 0, length: ns.length))
      .map { ns.substring(with: $0.range(at: 1)) }
  }

  static func displayName(row: [String: Any], fallback: String) -> String {
    for key in ["name", "title", "primary_subcategory"] {
      if let value = row[key] as? String, !value.isEmpty { return value }
    }
    return shortName(fallback)
  }

  /// A readable label from a prose value: the first clause, capped.
  static func shortName(_ value: String) -> String {
    let firstClause = value.split(whereSeparator: { $0 == "," || $0 == "." }).first.map(String.init)
      ?? value
    let trimmed = firstClause.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.count > 60 ? String(trimmed.prefix(57)) + "…" : trimmed
  }

  static func stableId(_ text: String) -> String {
    var hash: UInt64 = 5381
    for byte in Array(text.utf8) { hash = (hash &* 33) &+ UInt64(byte) }
    return String(hash, radix: 16)
  }

  static func intValue(_ raw: Any?) -> Int? {
    if let i = raw as? Int { return i }
    if let s = raw as? String { return Int(s) }
    if let d = raw as? Double { return Int(d) }
    return nil
  }

  static func doubleValue(_ raw: Any?) -> Double? {
    if let d = raw as? Double { return d }
    if let i = raw as? Int { return Double(i) }
    if let s = raw as? String { return Double(s) }
    return nil
  }

  static func boolValue(_ raw: Any?) -> Bool? {
    if let b = raw as? Bool { return b }
    if let s = raw as? String { return ["true", "1", "yes"].contains(s.lowercased()) }
    return nil
  }

  static func countFiles(_ directory: URL) -> Int {
    let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)
    var count = 0
    while let url = enumerator?.nextObject() as? URL {
      var isDir: ObjCBool = false
      if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), !isDir.boolValue {
        count += 1
      }
    }
    return count
  }

  /// Flattens `previews/**/<hash>.webp` into the library's own previews
  /// directory, which is what `preview_ref` names.
  static func copyPreviews(from source: URL, to destination: URL) -> Int {
    let fm = FileManager.default
    try? fm.createDirectory(at: destination, withIntermediateDirectories: true)
    var copied = 0
    let enumerator = fm.enumerator(at: source, includingPropertiesForKeys: nil)
    while let url = enumerator?.nextObject() as? URL {
      var isDir: ObjCBool = false
      guard fm.fileExists(atPath: url.path, isDirectory: &isDir), !isDir.boolValue else { continue }
      let target = destination.appendingPathComponent(url.lastPathComponent)
      if fm.fileExists(atPath: target.path) { copied += 1; continue }
      if (try? fm.copyItem(at: url, to: target)) != nil { copied += 1 }
    }
    return copied
  }
}
