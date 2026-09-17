// LibraryStore.swift — the Creative Library (PRD docs/PRD-creative-library.md, L1).
//
// One store of reusable creative MATERIAL, shared by the engine and every
// client: slot templates, scene/pose/lighting components, wardrobe items,
// outfits, looks and recipes. Renders keep living in the catalog; presets keep
// being the render recipe the resolver reads. This is the ingredients drawer.
//
// Design notes:
//   * ONE item type with per-kind optional fields (the `ImagePreset` pattern),
//     so a client reads one shape and an unknown kind never breaks a decode.
//   * Unknown top-level keys on an item are PRESERVED through a round trip
//     (`extras`), the lesson from ComfyBoxServerConfig wiping the video recipe
//     (#459): a newer client's field must survive an older engine's save.
//   * Facets are multi-valued and free-form here; the vocabulary lives in
//     `facets.json` and is advisory, so importing a pack cannot be blocked by
//     a taxonomy mismatch.
//   * Pack-provided collection membership is tracked separately from the
//     user's own (`collections` vs `packCollections`), which is what makes a
//     reimport reconcile instead of overwrite (PRD §7).
//   * Persistence is two atomic JSON files under `~/.comfybox/library/`.
//     Thumbnails are content-hashed files beside them, written by the caller.

import Foundation

// MARK: - Kinds

public enum LibraryItemKind: String, Codable, Sendable, CaseIterable {
  case template
  case component
  case wardrobeItem = "wardrobe_item"
  case outfit
  case look
  case recipe
}

/// A `{slot}` in a template body.
public struct LibrarySlot: Codable, Sendable, Equatable {
  public var id: String
  public var label: String?
  public var placeholder: String?
  public var defaultValue: String?
  public var options: [String]?

  public init(
    id: String, label: String? = nil, placeholder: String? = nil,
    defaultValue: String? = nil, options: [String]? = nil
  ) {
    self.id = id
    self.label = label
    self.placeholder = placeholder
    self.defaultValue = defaultValue
    self.options = options
  }

  enum CodingKeys: String, CodingKey {
    case id, label, placeholder, options
    case defaultValue = "default"
  }
}

// MARK: - Item

/// One piece of material. `value` is the text it contributes to a prompt;
/// everything else is how it is found, ranked and traced.
public struct LibraryEntry: Codable, Sendable, Equatable {
  public var id: String
  public var kind: LibraryItemKind
  public var name: String
  public var value: String

  /// Multi-valued taxonomy: `["mood": ["Calm"], "shot": ["Portrait"]]`.
  public var facets: [String: [String]]
  /// Collection ids the USER filed this under.
  public var collections: [String]
  /// Collection ids a PACK filed this under. Reconciled on reimport; never
  /// merged into `collections`, so a user's filing survives a pack update.
  public var packCollections: [String]

  public var rating: Int
  public var favorite: Bool
  public var archived: Bool
  public var notes: String?

  /// Content-hashed preview file name under `~/.comfybox/library/previews/`.
  public var previewRef: String?
  /// `local` | `pack:<id>` | `render:<assetId>` | `import:<app>`
  public var source: String
  public var createdAt: Date
  public var updatedAt: Date
  public var useCount: Int
  public var lastUsedAt: Date?

  // Per-kind fields — nil for kinds that do not use them.
  /// `template`: body slots. The body itself is `value`.
  public var slots: [LibrarySlot]?
  /// `component`: scene | pose | lighting | camera | expression | styling | negative
  public var componentKind: String?
  /// `wardrobe_item`
  public var category: String?
  public var subtype: String?
  /// `wardrobe_item`: piece | set | styling
  public var itemType: String?
  /// `outfit`: ordered wardrobe item ids.
  public var items: [String]?
  /// `look` / `recipe`
  public var promptPrefix: String?
  public var promptSuffix: String?
  public var negativePrompt: String?
  public var loras: [LoraReference]?
  /// `recipe`: the preset it corresponds to, when one exists.
  public var presetId: String?
  public var seed: Int?
  /// e.g. `fixed` | `random` | `imported-image`
  public var seedSource: String?
  public var model: String?
  public var steps: Int?
  public var guidance: Double?
  public var width: Int?
  public var height: Int?
  /// A foreign payload we keep but do not interpret (an imported node graph).
  public var rawPayload: String?

  /// Unknown top-level keys, preserved verbatim across a save (#459 lesson).
  public var extras: [String: JSONValue]

  public init(
    id: String,
    kind: LibraryItemKind,
    name: String,
    value: String,
    facets: [String: [String]] = [:],
    collections: [String] = [],
    packCollections: [String] = [],
    rating: Int = 0,
    favorite: Bool = false,
    archived: Bool = false,
    notes: String? = nil,
    previewRef: String? = nil,
    source: String = "local",
    createdAt: Date = Date(),
    updatedAt: Date = Date(),
    useCount: Int = 0,
    lastUsedAt: Date? = nil,
    slots: [LibrarySlot]? = nil,
    componentKind: String? = nil,
    category: String? = nil,
    subtype: String? = nil,
    itemType: String? = nil,
    items: [String]? = nil,
    promptPrefix: String? = nil,
    promptSuffix: String? = nil,
    negativePrompt: String? = nil,
    loras: [LoraReference]? = nil,
    presetId: String? = nil,
    seed: Int? = nil,
    seedSource: String? = nil,
    model: String? = nil,
    steps: Int? = nil,
    guidance: Double? = nil,
    width: Int? = nil,
    height: Int? = nil,
    rawPayload: String? = nil,
    extras: [String: JSONValue] = [:]
  ) {
    self.id = id
    self.kind = kind
    self.name = name
    self.value = value
    self.facets = facets
    self.collections = collections
    self.packCollections = packCollections
    self.rating = rating
    self.favorite = favorite
    self.archived = archived
    self.notes = notes
    self.previewRef = previewRef
    self.source = source
    self.createdAt = createdAt
    self.updatedAt = updatedAt
    self.useCount = useCount
    self.lastUsedAt = lastUsedAt
    self.slots = slots
    self.componentKind = componentKind
    self.category = category
    self.subtype = subtype
    self.itemType = itemType
    self.items = items
    self.promptPrefix = promptPrefix
    self.promptSuffix = promptSuffix
    self.negativePrompt = negativePrompt
    self.loras = loras
    self.presetId = presetId
    self.seed = seed
    self.seedSource = seedSource
    self.model = model
    self.steps = steps
    self.guidance = guidance
    self.width = width
    self.height = height
    self.rawPayload = rawPayload
    self.extras = extras
  }

  enum CodingKeys: String, CodingKey {
    case id, kind, name, value, facets, collections, rating, favorite, archived, notes
    case source, slots, category, subtype, items, loras, seed, model, steps, guidance
    case width, height
    case packCollections = "pack_collections"
    case previewRef = "preview_ref"
    case createdAt = "created_at"
    case updatedAt = "updated_at"
    case useCount = "use_count"
    case lastUsedAt = "last_used_at"
    case componentKind = "component_kind"
    case itemType = "item_type"
    case promptPrefix = "prompt_prefix"
    case promptSuffix = "prompt_suffix"
    case negativePrompt = "negative_prompt"
    case presetId = "preset_id"
    case seedSource = "seed_source"
    case rawPayload = "raw_payload"
  }

  /// Every key this type owns. Anything else on disk rides in `extras`.
  static let ownedKeys: Set<String> = Set(CodingKeys.allCases.map(\.stringValue))

  /// Dates are ISO-8601 STRINGS in our own encode/decode, not left to the
  /// encoder's `dateEncodingStrategy`. The HTTP layer encodes responses with
  /// its own encoder (numeric dates by default), and a client must be able to
  /// send back exactly what it received.
  static let iso: ISO8601DateFormatter = {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime]
    return f
  }()

  private static func decodeDate(
    _ c: KeyedDecodingContainer<CodingKeys>, _ key: CodingKeys
  ) throws -> Date? {
    if let text = try? c.decode(String.self, forKey: key) {
      return iso.date(from: text)
    }
    if let seconds = try? c.decode(Double.self, forKey: key) {
      return Date(timeIntervalSinceReferenceDate: seconds)
    }
    return nil
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    id = try c.decode(String.self, forKey: .id)
    kind = try c.decode(LibraryItemKind.self, forKey: .kind)
    name = try c.decode(String.self, forKey: .name)
    value = try c.decodeIfPresent(String.self, forKey: .value) ?? ""
    facets = try c.decodeIfPresent([String: [String]].self, forKey: .facets) ?? [:]
    collections = try c.decodeIfPresent([String].self, forKey: .collections) ?? []
    packCollections = try c.decodeIfPresent([String].self, forKey: .packCollections) ?? []
    rating = try c.decodeIfPresent(Int.self, forKey: .rating) ?? 0
    favorite = try c.decodeIfPresent(Bool.self, forKey: .favorite) ?? false
    archived = try c.decodeIfPresent(Bool.self, forKey: .archived) ?? false
    notes = try c.decodeIfPresent(String.self, forKey: .notes)
    previewRef = try c.decodeIfPresent(String.self, forKey: .previewRef)
    source = try c.decodeIfPresent(String.self, forKey: .source) ?? "local"
    createdAt = try Self.decodeDate(c, .createdAt) ?? Date()
    updatedAt = try Self.decodeDate(c, .updatedAt) ?? Date()
    useCount = try c.decodeIfPresent(Int.self, forKey: .useCount) ?? 0
    lastUsedAt = try Self.decodeDate(c, .lastUsedAt)
    slots = try c.decodeIfPresent([LibrarySlot].self, forKey: .slots)
    componentKind = try c.decodeIfPresent(String.self, forKey: .componentKind)
    category = try c.decodeIfPresent(String.self, forKey: .category)
    subtype = try c.decodeIfPresent(String.self, forKey: .subtype)
    itemType = try c.decodeIfPresent(String.self, forKey: .itemType)
    items = try c.decodeIfPresent([String].self, forKey: .items)
    promptPrefix = try c.decodeIfPresent(String.self, forKey: .promptPrefix)
    promptSuffix = try c.decodeIfPresent(String.self, forKey: .promptSuffix)
    negativePrompt = try c.decodeIfPresent(String.self, forKey: .negativePrompt)
    loras = try c.decodeIfPresent([LoraReference].self, forKey: .loras)
    presetId = try c.decodeIfPresent(String.self, forKey: .presetId)
    seed = try c.decodeIfPresent(Int.self, forKey: .seed)
    seedSource = try c.decodeIfPresent(String.self, forKey: .seedSource)
    model = try c.decodeIfPresent(String.self, forKey: .model)
    steps = try c.decodeIfPresent(Int.self, forKey: .steps)
    guidance = try c.decodeIfPresent(Double.self, forKey: .guidance)
    width = try c.decodeIfPresent(Int.self, forKey: .width)
    height = try c.decodeIfPresent(Int.self, forKey: .height)
    rawPayload = try c.decodeIfPresent(String.self, forKey: .rawPayload)
    let all = try decoder.singleValueContainer().decode([String: JSONValue].self)
    extras = all.filter { !Self.ownedKeys.contains($0.key) }
  }

  public func encode(to encoder: Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(id, forKey: .id)
    try c.encode(kind, forKey: .kind)
    try c.encode(name, forKey: .name)
    try c.encode(value, forKey: .value)
    try c.encode(facets, forKey: .facets)
    try c.encode(collections, forKey: .collections)
    try c.encode(packCollections, forKey: .packCollections)
    try c.encode(rating, forKey: .rating)
    try c.encode(favorite, forKey: .favorite)
    try c.encode(archived, forKey: .archived)
    try c.encodeIfPresent(notes, forKey: .notes)
    try c.encodeIfPresent(previewRef, forKey: .previewRef)
    try c.encode(source, forKey: .source)
    try c.encode(Self.iso.string(from: createdAt), forKey: .createdAt)
    try c.encode(Self.iso.string(from: updatedAt), forKey: .updatedAt)
    try c.encode(useCount, forKey: .useCount)
    try c.encodeIfPresent(lastUsedAt.map(Self.iso.string(from:)), forKey: .lastUsedAt)
    try c.encodeIfPresent(slots, forKey: .slots)
    try c.encodeIfPresent(componentKind, forKey: .componentKind)
    try c.encodeIfPresent(category, forKey: .category)
    try c.encodeIfPresent(subtype, forKey: .subtype)
    try c.encodeIfPresent(itemType, forKey: .itemType)
    try c.encodeIfPresent(items, forKey: .items)
    try c.encodeIfPresent(promptPrefix, forKey: .promptPrefix)
    try c.encodeIfPresent(promptSuffix, forKey: .promptSuffix)
    try c.encodeIfPresent(negativePrompt, forKey: .negativePrompt)
    try c.encodeIfPresent(loras, forKey: .loras)
    try c.encodeIfPresent(presetId, forKey: .presetId)
    try c.encodeIfPresent(seed, forKey: .seed)
    try c.encodeIfPresent(seedSource, forKey: .seedSource)
    try c.encodeIfPresent(model, forKey: .model)
    try c.encodeIfPresent(steps, forKey: .steps)
    try c.encodeIfPresent(guidance, forKey: .guidance)
    try c.encodeIfPresent(width, forKey: .width)
    try c.encodeIfPresent(height, forKey: .height)
    try c.encodeIfPresent(rawPayload, forKey: .rawPayload)
    // Unknown keys ride alongside, never clobbering an owned one.
    if !extras.isEmpty {
      var raw = encoder.container(keyedBy: LibraryRawKey.self)
      for (k, v) in extras.sorted(by: { $0.key < $1.key }) where !Self.ownedKeys.contains(k) {
        try raw.encode(v, forKey: LibraryRawKey(k))
      }
    }
  }
}

extension LibraryEntry.CodingKeys: CaseIterable {}

/// Coding key for the unknown-key passthrough.
struct LibraryRawKey: CodingKey {
  var stringValue: String
  var intValue: Int? { nil }
  init(_ s: String) { stringValue = s }
  init?(stringValue: String) { self.stringValue = stringValue }
  init?(intValue: Int) { nil }
}

// MARK: - Collections

public struct LibraryCollection: Codable, Sendable, Equatable {
  public var id: String
  public var name: String
  /// Hex colour, as the reference packs use ("#35d7ff").
  public var color: String?
  public var parentId: String?
  public var displayOrder: Int
  /// `manual` | `derived` | `pack`
  public var membership: String

  public init(
    id: String, name: String, color: String? = nil, parentId: String? = nil,
    displayOrder: Int = 0, membership: String = "manual"
  ) {
    self.id = id
    self.name = name
    self.color = color
    self.parentId = parentId
    self.displayOrder = displayOrder
    self.membership = membership
  }

  enum CodingKeys: String, CodingKey {
    case id, name, color, membership
    case parentId = "parent_id"
    case displayOrder = "display_order"
  }
}

// MARK: - Query

/// Everything the Library tab's filter rail and every picker needs, in one
/// value so the HTTP and MCP layers stay thin.
public struct LibraryQuery: Sendable, Equatable {
  public var kinds: [LibraryItemKind]
  /// Facet axis -> any-of values. An item matches when EVERY named axis has at
  /// least one of its values (axes are ANDed, values within an axis are ORed).
  public var facets: [String: [String]]
  public var collection: String?
  public var componentKind: String?
  public var category: String?
  public var text: String?
  public var minRating: Int?
  public var favoritesOnly: Bool
  public var includeArchived: Bool
  /// `recent` | `rating` | `use_count` | `name`
  public var order: String
  public var limit: Int

  public init(
    kinds: [LibraryItemKind] = [], facets: [String: [String]] = [:], collection: String? = nil,
    componentKind: String? = nil, category: String? = nil, text: String? = nil,
    minRating: Int? = nil, favoritesOnly: Bool = false, includeArchived: Bool = false,
    order: String = "recent", limit: Int = 200
  ) {
    self.kinds = kinds
    self.facets = facets
    self.collection = collection
    self.componentKind = componentKind
    self.category = category
    self.text = text
    self.minRating = minRating
    self.favoritesOnly = favoritesOnly
    self.includeArchived = includeArchived
    self.order = order
    self.limit = limit
  }
}

public enum LibraryError: Error, LocalizedError, Equatable {
  case invalid(String)
  case notFound(String)

  public var errorDescription: String? {
    switch self {
    case .invalid(let m): return m
    case .notFound(let id): return "library item '\(id)' not found"
    }
  }
}

// MARK: - Store

/// File-backed, synchronous, single-writer. The server owns one instance.
public final class LibraryStore: @unchecked Sendable {
  public let directory: URL
  private let fileManager: FileManager
  private let lock = NSLock()
  private var itemsById: [String: LibraryEntry] = [:]
  private var collectionsById: [String: LibraryCollection] = [:]

  public var itemsPath: URL { directory.appendingPathComponent("items.json") }
  public var collectionsPath: URL { directory.appendingPathComponent("collections.json") }
  public var previewsDirectory: URL { directory.appendingPathComponent("previews") }

  public init(directory: URL, fileManager: FileManager = .default) {
    self.directory = directory
    self.fileManager = fileManager
    load()
  }

  public convenience init(home: String = NSHomeDirectory()) {
    self.init(directory: URL(fileURLWithPath: home).appendingPathComponent(".comfybox/library"))
  }

  // MARK: load / persist

  private struct ItemFile: Codable { var items: [LibraryEntry] }
  private struct CollectionFile: Codable { var collections: [LibraryCollection] }

  private static func decoder() -> JSONDecoder { JSONDecoder() }

  private static func encoder() -> JSONEncoder {
    let e = JSONEncoder()
    e.outputFormatting = [.prettyPrinted, .sortedKeys]
    return e
  }

  /// A malformed file is never silently emptied: it is left on disk and the
  /// store starts empty, so a save cannot overwrite something unreadable.
  private func load() {
    if let data = try? Data(contentsOf: itemsPath),
       let file = try? Self.decoder().decode(ItemFile.self, from: data) {
      itemsById = Dictionary(uniqueKeysWithValues: file.items.map { ($0.id, $0) })
    }
    if let data = try? Data(contentsOf: collectionsPath),
       let file = try? Self.decoder().decode(CollectionFile.self, from: data) {
      collectionsById = Dictionary(uniqueKeysWithValues: file.collections.map { ($0.id, $0) })
    }
  }

  private func persistItems() throws {
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
    let sorted = itemsById.values.sorted { $0.id < $1.id }
    try Self.encoder().encode(ItemFile(items: sorted)).write(to: itemsPath, options: .atomic)
  }

  private func persistCollections() throws {
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
    let sorted = collectionsById.values.sorted { $0.id < $1.id }
    try Self.encoder().encode(CollectionFile(collections: sorted))
      .write(to: collectionsPath, options: .atomic)
  }

  // MARK: items

  public func item(id: String) -> LibraryEntry? {
    lock.lock(); defer { lock.unlock() }
    return itemsById[id]
  }

  public func allItems() -> [LibraryEntry] {
    lock.lock(); defer { lock.unlock() }
    return itemsById.values.sorted { $0.id < $1.id }
  }

  /// Insert or replace. `updatedAt` is stamped here; `createdAt` and
  /// `useCount` are carried over from an existing item so a client cannot
  /// reset them by omitting them.
  /// Shared by the single and batch writes.
  static func validate(_ item: inout LibraryEntry) throws {
    let trimmedId = item.id.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmedId.isEmpty else { throw LibraryError.invalid("'id' is required") }
    item.id = trimmedId
    guard !item.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw LibraryError.invalid("'name' is required")
    }
    guard (0...5).contains(item.rating) else {
      throw LibraryError.invalid("'rating' must be 0...5")
    }
    if item.kind == .template {
      let declared = Set((item.slots ?? []).map(\.id))
      let used = slotMarkers(in: item.value)
      let undeclared = used.subtracting(declared).sorted()
      guard undeclared.isEmpty else {
        throw LibraryError.invalid(
          "template body uses undeclared slot(s): \(undeclared.joined(separator: ", "))")
      }
    }
  }

  @discardableResult
  public func upsert(_ incoming: LibraryEntry, now: Date = Date()) throws -> LibraryEntry {
    var item = incoming
    try Self.validate(&item)
    lock.lock(); defer { lock.unlock() }
    if let existing = itemsById[item.id] {
      item.createdAt = existing.createdAt
      item.useCount = max(item.useCount, existing.useCount)
      item.lastUsedAt = item.lastUsedAt ?? existing.lastUsedAt
    }
    item.updatedAt = now
    itemsById[item.id] = item
    try persistItems()
    return item
  }

  /// Insert or replace MANY items with ONE write. A pack import is 1,400
  /// items; persisting after each one rewrites the whole file every time
  /// (quadratic, and it wedged the first real import of the reference pack).
  /// Validation and the createdAt/useCount carry-over are identical to
  /// ``upsert(_:now:)``; an item that fails validation is returned in
  /// `rejected` rather than aborting the batch.
  @discardableResult
  public func upsert(
    contentsOf incoming: [LibraryEntry], now: Date = Date()
  ) throws -> (saved: [LibraryEntry], rejected: [(LibraryEntry, Error)]) {
    var saved: [LibraryEntry] = []
    var rejected: [(LibraryEntry, Error)] = []
    lock.lock()
    for var item in incoming {
      do {
        try Self.validate(&item)
      } catch {
        rejected.append((item, error))
        continue
      }
      if let existing = itemsById[item.id] {
        item.createdAt = existing.createdAt
        item.useCount = max(item.useCount, existing.useCount)
        item.lastUsedAt = item.lastUsedAt ?? existing.lastUsedAt
      }
      item.updatedAt = now
      itemsById[item.id] = item
      saved.append(item)
    }
    do {
      try persistItems()
    } catch {
      lock.unlock()
      throw error
    }
    lock.unlock()
    return (saved, rejected)
  }

  /// Same for collections: one write for the whole set.
  @discardableResult
  public func upsertCollections(_ incoming: [LibraryCollection]) throws -> Int {
    lock.lock()
    var count = 0
    for collection in incoming {
      guard !collection.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            !collection.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            collection.parentId != collection.id else { continue }
      collectionsById[collection.id] = collection
      count += 1
    }
    do {
      try persistCollections()
    } catch {
      lock.unlock()
      throw error
    }
    lock.unlock()
    return count
  }

  @discardableResult
  public func delete(id: String) throws -> Bool {
    lock.lock(); defer { lock.unlock() }
    guard itemsById.removeValue(forKey: id) != nil else { return false }
    try persistItems()
    return true
  }

  /// Records a use: the signal that makes "what I actually reach for" rankable.
  @discardableResult
  public func markUsed(id: String, now: Date = Date()) throws -> LibraryEntry {
    lock.lock(); defer { lock.unlock() }
    guard var item = itemsById[id] else { throw LibraryError.notFound(id) }
    item.useCount += 1
    item.lastUsedAt = now
    itemsById[id] = item
    try persistItems()
    return item
  }

  // MARK: collections

  public func allCollections() -> [LibraryCollection] {
    lock.lock(); defer { lock.unlock() }
    return collectionsById.values.sorted {
      $0.displayOrder == $1.displayOrder ? $0.id < $1.id : $0.displayOrder < $1.displayOrder
    }
  }

  @discardableResult
  public func upsertCollection(_ incoming: LibraryCollection) throws -> LibraryCollection {
    guard !incoming.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw LibraryError.invalid("collection 'id' is required")
    }
    guard !incoming.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw LibraryError.invalid("collection 'name' is required")
    }
    if let parent = incoming.parentId {
      guard parent != incoming.id else {
        throw LibraryError.invalid("a collection cannot be its own parent")
      }
    }
    lock.lock(); defer { lock.unlock() }
    collectionsById[incoming.id] = incoming
    try persistCollections()
    return incoming
  }

  /// Deleting a collection unfiles its items rather than deleting them. Pack
  /// membership is left alone, so a pack reimport still reconciles.
  @discardableResult
  public func deleteCollection(id: String) throws -> Bool {
    lock.lock(); defer { lock.unlock() }
    guard collectionsById.removeValue(forKey: id) != nil else { return false }
    for (itemId, var item) in itemsById where item.collections.contains(id) {
      item.collections.removeAll { $0 == id }
      itemsById[itemId] = item
    }
    try persistCollections()
    try persistItems()
    return true
  }

  // MARK: query

  public func search(_ q: LibraryQuery) -> [LibraryEntry] {
    lock.lock()
    let all = Array(itemsById.values)
    lock.unlock()
    let needle = q.text?.trimmingCharacters(in: .whitespaces).lowercased()
    var hits = all.filter { item in
      if !q.includeArchived, item.archived { return false }
      if !q.kinds.isEmpty, !q.kinds.contains(item.kind) { return false }
      if let c = q.collection, !(item.collections.contains(c) || item.packCollections.contains(c)) {
        return false
      }
      if let ck = q.componentKind, item.componentKind != ck { return false }
      if let cat = q.category, item.category != cat { return false }
      if let min = q.minRating, item.rating < min { return false }
      if q.favoritesOnly, !item.favorite { return false }
      for (axis, wanted) in q.facets {
        let have = Set(item.facets[axis] ?? [])
        if have.isDisjoint(with: Set(wanted)) { return false }
      }
      if let needle, !needle.isEmpty {
        let hay = [item.name, item.value, item.notes ?? "", item.category ?? "",
                   item.subtype ?? "", item.componentKind ?? ""]
          .joined(separator: " ").lowercased()
        if !hay.contains(needle) { return false }
      }
      return true
    }
    switch q.order {
    case "rating":
      hits.sort { ($0.rating, $0.useCount) > ($1.rating, $1.useCount) }
    case "use_count":
      hits.sort { $0.useCount > $1.useCount }
    case "name":
      hits.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    default:
      hits.sort { $0.updatedAt > $1.updatedAt }
    }
    return Array(hits.prefix(max(0, q.limit)))
  }

  /// Facet axis -> value -> count, over everything that is not archived. The
  /// filter rail reads this, so an axis with no values never renders.
  public func facetCounts() -> [String: [String: Int]] {
    lock.lock()
    let all = Array(itemsById.values)
    lock.unlock()
    var out: [String: [String: Int]] = [:]
    for item in all where !item.archived {
      for (axis, values) in item.facets {
        for v in values { out[axis, default: [:]][v, default: 0] += 1 }
      }
    }
    return out
  }

  // MARK: templates

  static let slotPattern = try! NSRegularExpression(pattern: "\\{([A-Za-z0-9_-]+)\\}")

  /// The `{slot}` ids used in a body.
  public static func slotMarkers(in body: String) -> Set<String> {
    let ns = body as NSString
    let matches = slotPattern.matches(in: body, range: NSRange(location: 0, length: ns.length))
    return Set(matches.map { ns.substring(with: $0.range(at: 1)) })
  }

  /// Fill a template. Precedence: supplied value > slot default > the marker is
  /// LEFT VISIBLE (the Studio Pack contract: an unfilled slot must be obvious
  /// in the prompt, never silently blank).
  public static func fill(template: LibraryEntry, values: [String: String]) -> String {
    var out = template.value
    let defaults = Dictionary(
      (template.slots ?? []).compactMap { slot -> (String, String)? in
        guard let d = slot.defaultValue else { return nil }
        return (slot.id, d)
      }, uniquingKeysWith: { a, _ in a })
    for id in slotMarkers(in: out).sorted() {
      let replacement = values[id]?.trimmingCharacters(in: .whitespaces).nilIfEmpty
        ?? defaults[id]?.nilIfEmpty
      guard let replacement else { continue }
      out = out.replacingOccurrences(of: "{\(id)}", with: replacement)
    }
    return out
  }

  /// An outfit's phrase: its own text when set, else its items joined.
  public func renderOutfit(_ outfit: LibraryEntry) -> String {
    if !outfit.value.trimmingCharacters(in: .whitespaces).isEmpty { return outfit.value }
    lock.lock(); defer { lock.unlock() }
    let parts = (outfit.items ?? []).compactMap { itemsById[$0]?.value }
      .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
    return parts.joined(separator: ", ")
  }
}

extension String {
  fileprivate var nilIfEmpty: String? { isEmpty ? nil : self }
}
