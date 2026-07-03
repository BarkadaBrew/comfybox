import XCTest
@testable import ZImage

final class CharacterStoreTests: XCTestCase {

  private func makeTempPath() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-character-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir.appendingPathComponent("characters.json")
  }

  private func sampleCharacter() -> CharacterEntry {
    CharacterEntry(
      name: "Nova Starling",
      description: "A confident astronaut.",
      base: "A confident astronaut in a white flight suit.",
      banana: "Suit unzipped to the waist.",
      avocado: "Fully explicit anatomy detail.",
      defaultLoras: [CharacterLoraReference(filename: "nova.safetensors", scale: 0.8)],
      promptSnippet: "cinematic lighting, 85mm",
      negativePrompt: "cartoon, blurry",
      triggerWords: "novastar",
      tags: ["scifi", "portrait"]
    )
  }

  // MARK: - Model behavior

  func testSlugGeneratedFromName() {
    XCTAssertEqual(CharacterEntry.slug("Nova Starling"), "nova-starling")
    XCTAssertEqual(CharacterEntry.slug("  Multi   Space!! "), "multi-space")
    XCTAssertEqual(CharacterEntry.slug("###"), "character")
  }

  func testResolvedDescriptionGatedByContentMode() {
    let c = sampleCharacter()
    XCTAssertEqual(c.resolvedDescription(for: .neutral), "A confident astronaut in a white flight suit.")
    XCTAssertEqual(
      c.resolvedDescription(for: .banana),
      "A confident astronaut in a white flight suit. Suit unzipped to the waist."
    )
    XCTAssertEqual(
      c.resolvedDescription(for: .avocado),
      "A confident astronaut in a white flight suit. Suit unzipped to the waist. Fully explicit anatomy detail."
    )
  }

  func testResolvedDescriptionFallsBackToFlatWhenNoTiers() {
    let c = CharacterEntry(name: "Legacy", description: "flat only")
    XCTAssertEqual(c.resolvedDescription(for: .avocado), "flat only")
  }

  func testKindDerivedFromEnvironmentPrefix() throws {
    // Legacy entries (no explicit kind): "environment: …" text means scene.
    let scene = CharacterEntry(name: "Apartment", description: "environment: cozy studio")
    XCTAssertEqual(scene.kind, CharacterEntry.kindScene)
    let person = CharacterEntry(name: "Bree", description: "A tall athletic woman")
    XCTAssertEqual(person.kind, CharacterEntry.kindCharacter)

    // Decoding without a kind derives it; an explicit kind wins.
    let decoded = try JSONDecoder().decode(
      CharacterEntry.self,
      from: Data(#"{ "name": "BackRoom", "base": "environment: back room" }"#.utf8))
    XCTAssertEqual(decoded.kind, CharacterEntry.kindScene)

    let explicit = try JSONDecoder().decode(
      CharacterEntry.self,
      from: Data(#"{ "name": "Weird", "kind": "character", "base": "environment: but a person" }"#.utf8))
    XCTAssertEqual(explicit.kind, CharacterEntry.kindCharacter)
  }

  func testKindPersistsThroughStore() async throws {
    let path = try makeTempPath()
    let store = CharacterStore(path: path)
    await store.upsert(CharacterEntry(name: "Apartment", kind: CharacterEntry.kindScene, description: "cozy studio"))
    let reloaded = CharacterStore(path: path)
    let apartment = await reloaded.get("apartment")
    XCTAssertEqual(apartment?.kind, CharacterEntry.kindScene)
  }

  // MARK: - CRUD

  func testUpsertGetListDelete() async throws {
    let path = try makeTempPath()
    let store = CharacterStore(path: path)

    let initial = await store.list()
    XCTAssertTrue(initial.isEmpty)

    let stored = await store.upsert(sampleCharacter())
    XCTAssertEqual(stored.id, "nova-starling")

    // get is case-insensitive on id
    let fetched = await store.get("NOVA-STARLING")
    XCTAssertEqual(fetched?.name, "Nova Starling")
    XCTAssertEqual(fetched?.defaultLoras.first?.scale, 0.8)

    let afterInsert = await store.list()
    XCTAssertEqual(afterInsert.count, 1)

    let deleted = await store.delete("nova-starling")
    XCTAssertTrue(deleted)
    let goneAfterDelete = await store.get("nova-starling")
    XCTAssertNil(goneAfterDelete)
    let deleteAgain = await store.delete("nova-starling")
    XCTAssertFalse(deleteAgain)
    let afterDelete = await store.list()
    XCTAssertTrue(afterDelete.isEmpty)
  }

  func testUpsertUpdatesExistingAndPreservesCreatedAt() async throws {
    let path = try makeTempPath()
    let store = CharacterStore(path: path)

    let first = await store.upsert(
      CharacterEntry(id: "hero", name: "Hero", description: "v1", createdAt: 1000, updatedAt: 1000)
    )
    XCTAssertEqual(first.createdAt, 1000)

    // A second upsert with the same id must keep createdAt but refresh updatedAt.
    let second = await store.upsert(CharacterEntry(id: "hero", name: "Hero", description: "v2"))
    XCTAssertEqual(second.createdAt, 1000)
    XCTAssertGreaterThanOrEqual(second.updatedAt, 1000)
    let list = await store.list()
    XCTAssertEqual(list.count, 1)
    let hero = await store.get("hero")
    XCTAssertEqual(hero?.description, "v2")
  }

  // MARK: - Persistence round-trip

  func testPersistenceRoundTripAcrossInstances() async throws {
    let path = try makeTempPath()

    let store1 = CharacterStore(path: path)
    await store1.upsert(sampleCharacter())
    await store1.upsert(CharacterEntry(name: "Aria Vale", description: "a bard", tags: ["fantasy"]))

    // A fresh instance reads the persisted file.
    let store2 = CharacterStore(path: path)
    let list = await store2.list()
    XCTAssertEqual(list.count, 2)
    XCTAssertEqual(list.map(\.name), ["Aria Vale", "Nova Starling"]) // name-sorted

    let nova = await store2.get("nova-starling")
    XCTAssertEqual(nova?.triggerWords, "novastar")
    XCTAssertEqual(nova?.avocado, "Fully explicit anatomy detail.")
    XCTAssertEqual(nova?.defaultLoras.first?.filename, "nova.safetensors")
    XCTAssertEqual(nova?.tags, ["scifi", "portrait"])
  }

  func testPersistedFileIsNameKeyedObjectOnDisk() async throws {
    let path = try makeTempPath()
    let store = CharacterStore(path: path)
    await store.upsert(sampleCharacter())

    // Persisted as an object keyed by lowercased name (legacy CharacterLoader read format).
    let data = try Data(contentsOf: path)
    let decoded = try JSONDecoder().decode([String: CharacterEntry].self, from: data)
    XCTAssertEqual(decoded.count, 1)
    XCTAssertEqual(decoded["nova starling"]?.id, "nova-starling")
  }

  /// Coexistence guard: the legacy CharacterLoader (Telegram bot) must still parse a file
  /// written by CharacterStore, including a character that had no explicit `base` tier.
  func testLegacyCharacterLoaderCanReadStoreOutput() async throws {
    let path = try makeTempPath()
    let store = CharacterStore(path: path)
    await store.upsert(sampleCharacter())
    await store.upsert(CharacterEntry(name: "Flatly", description: "just a flat description"))

    let loader = CharacterLoader(configPath: path.path)
    XCTAssertTrue(loader.has("nova starling"))
    // Tiered assembly still works through the legacy loader.
    XCTAssertEqual(
      loader.description(for: "nova starling", mode: .banana),
      "A confident astronaut in a white flight suit. Suit unzipped to the waist."
    )
    // Flat-only character got a `base` backfill so the loader (non-optional base) parses it.
    XCTAssertEqual(loader.description(for: "flatly", mode: .neutral), "just a flat description")
  }

  // MARK: - Tolerant decode

  func testTolerantDecodeOfPartialCharacterJSON() throws {
    // Only a name; everything else should default (empty arrays, nil optionals, timestamps).
    let json = Data(#"{ "name": "Sparse" }"#.utf8)
    let c = try JSONDecoder().decode(CharacterEntry.self, from: json)
    XCTAssertEqual(c.name, "Sparse")
    XCTAssertEqual(c.id, "sparse") // slug backfilled from name
    XCTAssertEqual(c.description, "")
    XCTAssertTrue(c.defaultLoras.isEmpty)
    XCTAssertTrue(c.tags.isEmpty)
    XCTAssertNil(c.base)
    XCTAssertGreaterThan(c.createdAt, 0)
  }

  func testTolerantDecodeOfLegacyNameKeyedObject() async throws {
    // Legacy TS character-registry format: object keyed by name, values omit id.
    let path = try makeTempPath()
    let legacy = #"""
    {
      "Nova Starling": { "name": "Nova Starling", "description": "legacy flat", "tags": ["scifi"] }
    }
    """#
    try Data(legacy.utf8).write(to: path)

    let store = CharacterStore(path: path)
    let list = await store.list()
    XCTAssertEqual(list.count, 1)
    let nova = await store.get("nova-starling")
    XCTAssertEqual(nova?.description, "legacy flat")
    XCTAssertEqual(nova?.tags, ["scifi"])
  }

  func testLoraReferenceTolerantDecodeDefaultsScale() throws {
    let json = Data(#"{ "filename": "x.safetensors" }"#.utf8)
    let ref = try JSONDecoder().decode(CharacterLoraReference.self, from: json)
    XCTAssertEqual(ref.filename, "x.safetensors")
    XCTAssertEqual(ref.scale, 1.0)
  }

  // MARK: - Legacy image-service registry migration

  private func writeLegacyRegistry(_ json: String) throws -> URL {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-legacy-registry-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let url = dir.appendingPathComponent("characters.json")
    try Data(json.utf8).write(to: url)
    return url
  }

  /// The Electron image-service registry shape: object keyed by display name,
  /// entries carry a flat `description`, `tags`, and optional tier fields.
  private static let legacyJSON = #"""
  {
    "Bree": { "description": "A tall athletic woman, rich hand-written text.", "tags": ["Bree"] },
    "Todd": {
      "description": "Flat todd text.",
      "base": "Tiered todd base.",
      "avocado": "Legacy explicit tier.",
      "tags": ["Todd"]
    },
    "CoffeeShop": {
      "description": "Cozy corner coffee shop interior.",
      "tags": ["coffeeshop", "cafe"]
    }
  }
  """#

  func testImportLegacyRegistryAddsMissingEntries() async throws {
    let path = try makeTempPath()
    let legacy = try writeLegacyRegistry(Self.legacyJSON)
    let store = CharacterStore(path: path)

    let changed = await store.importLegacyRegistry(at: legacy)
    XCTAssertEqual(changed, 3)

    let list = await store.list()
    XCTAssertEqual(list.count, 3)

    // Flat-description entries become the base tier so resolvedDescription shows them.
    let bree = await store.get("bree")
    XCTAssertEqual(bree?.resolvedDescription(for: .neutral), "A tall athletic woman, rich hand-written text.")
    XCTAssertEqual(bree?.tags, ["Bree"])

    // Explicit tiers pass through.
    let todd = await store.get("todd")
    XCTAssertEqual(todd?.base, "Tiered todd base.")
    XCTAssertEqual(todd?.avocado, "Legacy explicit tier.")

    // Location-style entries import too.
    let shop = await store.get("coffeeshop")
    XCTAssertEqual(shop?.tags, ["coffeeshop", "cafe"])
  }

  func testImportLegacyRegistryOverwritesUneditedEntriesOnly() async throws {
    let path = try makeTempPath()
    let legacy = try writeLegacyRegistry(Self.legacyJSON)
    let store = CharacterStore(path: path)

    // A never-edited entry (updatedAt == createdAt): legacy text should win,
    // but a local explicit tier the legacy entry lacks must survive.
    await store.seedForTesting(CharacterEntry(
      id: "bree", name: "bree",
      base: "Condensed machine summary.",
      avocado: "Local-only explicit tier.",
      createdAt: 42, updatedAt: 42
    ))

    // An edited entry (updatedAt > createdAt) must be left alone entirely.
    await store.seedForTesting(CharacterEntry(
      id: "todd", name: "todd",
      base: "Hand-edited todd.",
      createdAt: 42, updatedAt: 99
    ))

    let changed = await store.importLegacyRegistry(at: legacy)
    XCTAssertEqual(changed, 2) // bree updated + coffeeshop added

    let bree = await store.get("bree")
    XCTAssertEqual(bree?.base, "A tall athletic woman, rich hand-written text.")
    XCTAssertEqual(bree?.avocado, "Local-only explicit tier.")
    XCTAssertEqual(bree?.createdAt, 42)
    XCTAssertGreaterThan(bree?.updatedAt ?? 0, 42)

    let todd = await store.get("todd")
    XCTAssertEqual(todd?.base, "Hand-edited todd.")
    XCTAssertNil(todd?.avocado)
    XCTAssertEqual(todd?.updatedAt, 99)
  }

  func testImportLegacyRegistryIsIdempotent() async throws {
    let path = try makeTempPath()
    let legacy = try writeLegacyRegistry(Self.legacyJSON)
    let store = CharacterStore(path: path)

    let first = await store.importLegacyRegistry(at: legacy)
    XCTAssertEqual(first, 3)
    // Imported entries now have updatedAt > createdAt (or identical content),
    // so a second run must change nothing.
    let second = await store.importLegacyRegistry(at: legacy)
    XCTAssertEqual(second, 0)

    // And the merge persists across instances.
    let reloaded = CharacterStore(path: path)
    let list = await reloaded.list()
    XCTAssertEqual(list.count, 3)
  }

  func testImportLegacyRegistryMissingFileIsNoOp() async throws {
    let path = try makeTempPath()
    let store = CharacterStore(path: path)
    let changed = await store.importLegacyRegistry(
      at: URL(fileURLWithPath: "/nonexistent/nowhere/characters.json"))
    XCTAssertEqual(changed, 0)
  }
}
