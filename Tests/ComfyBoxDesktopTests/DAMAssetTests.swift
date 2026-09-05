// DAMAssetTests.swift — Tests for DAMAsset model

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("DAMAsset")
struct DAMAssetTests {
    @Test("default initialization")
    func defaults() {
        let asset = DAMAsset(filename: "test.png", absolutePath: "/tmp/test.png")
        #expect(asset.kind == "image")
        #expect(asset.filename == "test.png")
        #expect(asset.absolutePath == "/tmp/test.png")
        #expect(asset.fileSize == 0)
        #expect(asset.sha256 == nil)
        #expect(asset.width == nil)
        #expect(asset.height == nil)
        #expect(!asset.orphaned)
        #expect(asset.prompt == nil)
        #expect(asset.negativePrompt == nil)
        #expect(asset.seed == nil)
        #expect(asset.steps == nil)
        #expect(asset.guidance == nil)
        #expect(asset.modelFamily == nil)
        #expect(asset.rating == 0)
        #expect(!asset.favorite)
        #expect(asset.contentMode == nil)
        #expect(asset.characterName == nil)
    }

    @Test("full initialization preserves all fields")
    func fullInit() {
        let asset = TestData.makeAsset(
            id: "custom-id", filename: "photo.png", prompt: "a sunset",
            seed: 999, steps: 20, guidance: 7.5, modelFamily: "sdxl",
            rating: 4, favorite: true, contentMode: "banana",
            characterName: "Alice", width: 768, height: 512
        )
        #expect(asset.id == "custom-id")
        #expect(asset.filename == "photo.png")
        #expect(asset.prompt == "a sunset")
        #expect(asset.seed == 999)
        #expect(asset.steps == 20)
        #expect(asset.guidance == 7.5)
        #expect(asset.modelFamily == "sdxl")
        #expect(asset.rating == 4)
        #expect(asset.favorite)
        #expect(asset.contentMode == "banana")
        #expect(asset.characterName == "Alice")
        #expect(asset.width == 768)
        #expect(asset.height == 512)
    }

    @Test("id is unique across instances")
    func uniqueIds() {
        let a = DAMAsset(filename: "a.png", absolutePath: "/a.png")
        let b = DAMAsset(filename: "b.png", absolutePath: "/b.png")
        #expect(a.id != b.id)
    }

    @Test("identifiable conformance")
    func identifiable() {
        let asset = TestData.makeAsset(id: "my-id")
        #expect(asset.id == "my-id")
    }

    @Test("sendable conformance")
    func sendable() {
        let asset = TestData.makeAsset()
        let _: any Sendable = asset
        _ = asset
    }

    // MARK: - Fix wave (X8)

    // MARK: - #268: withLocation must not drop `source`

    @Test("withLocation preserves source across a secure/unsecure path move (regression: rebuild omitted source)")
    func withLocationPreservesSource() {
        let asset = TestData.makeAsset(id: "kira-1", filename: "kira-render.png", source: "kira")
        let moved = asset.withLocation(path: "/vault/secured/kira-render.png")

        #expect(moved.source == "kira")
        #expect(moved.absolutePath == "/vault/secured/kira-render.png")
        #expect(moved.filename == "kira-render.png")
        // Every other field is unchanged, not just source.
        #expect(moved.id == asset.id)
        #expect(moved.prompt == asset.prompt)
        #expect(moved.rating == asset.rating)
        #expect(moved.favorite == asset.favorite)
        #expect(moved.characterName == asset.characterName)
    }

    @Test("withLocation preserves a nil source (main-gallery asset stays main-gallery)")
    func withLocationPreservesNilSource() {
        let asset = TestData.makeAsset(id: "main-1", filename: "render.png", source: nil)
        let moved = asset.withLocation(path: "/tmp/test-images/renamed.png")
        #expect(moved.source == nil)
    }

    // MARK: - #372: copy(with:) preserves every field not named

    @Test("empty mutation returns a field-for-field identical copy")
    func copyWithEmptyMutationIsIdentity() {
        let asset = Self.everyFieldSetFixture()
        #expect(asset.copy(with: DAMAsset.Mutation()) == asset)
    }

    @Test("overriding one field leaves every other field untouched (Mirror-checked)")
    func copyOverridingOneFieldPreservesRest() {
        let asset = Self.everyFieldSetFixture()
        let updated = asset.copy(with: DAMAsset.Mutation(rating: .value(5)))
        #expect(updated.rating == 5)
        Self.assertFieldsEqual(asset, updated, excluding: ["rating"])
    }

    @Test("overriding a different field still leaves everything else untouched (Mirror-checked)")
    func copyOverridingSourcePreservesRest() {
        let asset = Self.everyFieldSetFixture()
        let updated = asset.copy(with: DAMAsset.Mutation(source: .value("bree")))
        #expect(updated.source == "bree")
        Self.assertFieldsEqual(asset, updated, excluding: ["source"])
    }

    @Test("each field can be overridden independently, including to an explicit nil")
    func copyEachFieldOverridesIndependently() {
        let asset = Self.everyFieldSetFixture()

        #expect(asset.copy(with: .init(id: .value("new-id"))).id == "new-id")
        #expect(asset.copy(with: .init(kind: .value("image"))).kind == "image")
        #expect(asset.copy(with: .init(filename: .value("new.png"))).filename == "new.png")
        #expect(asset.copy(with: .init(absolutePath: .value("/new/path.png"))).absolutePath == "/new/path.png")
        #expect(asset.copy(with: .init(fileSize: .value(42))).fileSize == 42)
        #expect(asset.copy(with: .init(sha256: .value("newsha"))).sha256 == "newsha")
        #expect(asset.copy(with: .init(sha256: .value(nil))).sha256 == nil)
        #expect(asset.copy(with: .init(width: .value(100))).width == 100)
        #expect(asset.copy(with: .init(width: .value(nil))).width == nil)
        #expect(asset.copy(with: .init(height: .value(100))).height == 100)
        #expect(asset.copy(with: .init(height: .value(nil))).height == nil)
        let newDate = Date(timeIntervalSince1970: 9)
        #expect(asset.copy(with: .init(createdAt: .value(newDate))).createdAt == newDate)
        #expect(asset.copy(with: .init(modifiedAt: .value(newDate))).modifiedAt == newDate)
        #expect(asset.copy(with: .init(ingestedAt: .value(newDate))).ingestedAt == newDate)
        #expect(asset.copy(with: .init(orphaned: .value(!asset.orphaned))).orphaned == !asset.orphaned)
        #expect(asset.copy(with: .init(prompt: .value("new prompt"))).prompt == "new prompt")
        #expect(asset.copy(with: .init(prompt: .value(nil))).prompt == nil)
        #expect(asset.copy(with: .init(negativePrompt: .value("new neg"))).negativePrompt == "new neg")
        #expect(asset.copy(with: .init(negativePrompt: .value(nil))).negativePrompt == nil)
        #expect(asset.copy(with: .init(seed: .value(99))).seed == 99)
        #expect(asset.copy(with: .init(seed: .value(nil))).seed == nil)
        #expect(asset.copy(with: .init(steps: .value(99))).steps == 99)
        #expect(asset.copy(with: .init(steps: .value(nil))).steps == nil)
        #expect(asset.copy(with: .init(guidance: .value(9.9))).guidance == 9.9)
        #expect(asset.copy(with: .init(guidance: .value(nil))).guidance == nil)
        #expect(asset.copy(with: .init(modelFamily: .value("sdxl"))).modelFamily == "sdxl")
        #expect(asset.copy(with: .init(modelFamily: .value(nil))).modelFamily == nil)
        #expect(asset.copy(with: .init(rating: .value(1))).rating == 1)
        #expect(asset.copy(with: .init(favorite: .value(!asset.favorite))).favorite == !asset.favorite)
        #expect(asset.copy(with: .init(contentMode: .value("apple"))).contentMode == "apple")
        #expect(asset.copy(with: .init(contentMode: .value(nil))).contentMode == nil)
        #expect(asset.copy(with: .init(characterName: .value("Someone"))).characterName == "Someone")
        #expect(asset.copy(with: .init(characterName: .value(nil))).characterName == nil)
        #expect(asset.copy(with: .init(source: .value("bree"))).source == "bree")
        #expect(asset.copy(with: .init(source: .value(nil))).source == nil)
    }

    @Test("a merge expression (incoming ?? existing) passed through .value(...) actually falls back to existing, not just re-wraps incoming")
    func copyMergeFallbackFallsBackToExisting() {
        let existing = Self.everyFieldSetFixture()
        // An "incoming" asset with nil generation metadata, as a bare
        // re-ingest with no sidecar would produce.
        let incoming = existing.copy(with: DAMAsset.Mutation(
            sha256: .value(nil), prompt: .value(nil), source: .value(nil)
        ))

        let merged = incoming.copy(with: DAMAsset.Mutation(
            sha256: .value(incoming.sha256 ?? existing.sha256),
            prompt: .value(incoming.prompt ?? existing.prompt),
            source: .value(incoming.source ?? existing.source)
        ))

        // This is the exact shape of DAMStore.mergedWithExisting's fallback.
        // With Mutation.<field> typed as a doubly-nested Optional instead of
        // `Override<T>`, this expression compiles but silently discards
        // `existing.*` and returns `nil` — the fallback never happens.
        #expect(merged.prompt == existing.prompt)
        #expect(merged.source == existing.source)
        #expect(merged.sha256 == existing.sha256)
    }

    /// A fixture where every field has a distinct, non-default, non-nil
    /// value, so a dropped field shows up as a genuine difference rather
    /// than coincidentally matching a shared default (nil / 0 / false).
    private static func everyFieldSetFixture() -> DAMAsset {
        DAMAsset(
            id: "fixture-id",
            kind: "video",
            filename: "fixture.mp4",
            absolutePath: "/vault/fixture.mp4",
            fileSize: 123_456,
            sha256: "deadbeef",
            width: 640,
            height: 480,
            createdAt: Date(timeIntervalSince1970: 1_000),
            modifiedAt: Date(timeIntervalSince1970: 2_000),
            ingestedAt: Date(timeIntervalSince1970: 3_000),
            orphaned: true,
            prompt: "a fixture prompt",
            negativePrompt: "a fixture negative",
            seed: 7,
            steps: 30,
            guidance: 4.5,
            modelFamily: "krea2",
            rating: 3,
            favorite: true,
            contentMode: "banana",
            characterName: "Fixture",
            source: "kira"
        )
    }

    /// Compares every stored property of `a` and `b` by name via `Mirror`,
    /// except those in `excluding`. Reflection (rather than a hand-picked
    /// list of `#expect`s) means a field added to `DAMAsset` in the future is
    /// picked up automatically — there is no second list to forget to update.
    private static func assertFieldsEqual(
        _ a: DAMAsset, _ b: DAMAsset, excluding: Set<String>,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let childrenA = Dictionary(uniqueKeysWithValues: Mirror(reflecting: a).children.compactMap {
            child -> (String, Any)? in
            guard let label = child.label else { return nil }
            return (label, child.value)
        })
        let childrenB = Dictionary(uniqueKeysWithValues: Mirror(reflecting: b).children.compactMap {
            child -> (String, Any)? in
            guard let label = child.label else { return nil }
            return (label, child.value)
        })
        #expect(Set(childrenA.keys) == Set(childrenB.keys), sourceLocation: sourceLocation)
        for key in childrenA.keys where !excluding.contains(key) {
            // `Any` isn't Equatable; DAMAsset's fields are all String / Int /
            // Int64 / Double / Bool / Date and their Optionals, for which
            // `String(describing:)` is a stable, faithful comparison.
            #expect(
                String(describing: childrenA[key] as Any) == String(describing: childrenB[key] as Any),
                "field '\(key)' differs after a mutation that did not name it",
                sourceLocation: sourceLocation
            )
        }
    }

    @Test("isEditableImage requires image kind and a supported extension")
    func isEditableImage() {
        #expect(DAMAsset(kind: "image", filename: "a.png", absolutePath: "/a.png").isEditableImage)
        #expect(DAMAsset(kind: "image", filename: "a.JPG", absolutePath: "/a.JPG").isEditableImage)
        #expect(DAMAsset(kind: "image", filename: "a.jpeg", absolutePath: "/a.jpeg").isEditableImage)
        #expect(DAMAsset(kind: "image", filename: "a.TIFF", absolutePath: "/a.TIFF").isEditableImage)
        #expect(DAMAsset(kind: "image", filename: "a.tif", absolutePath: "/a.tif").isEditableImage)
        #expect(!DAMAsset(kind: "video", filename: "a.mp4", absolutePath: "/a.mp4").isEditableImage)
        #expect(!DAMAsset(kind: "image", filename: "a.webp", absolutePath: "/a.webp").isEditableImage)
        #expect(!DAMAsset(kind: "image", filename: "a.gif", absolutePath: "/a.gif").isEditableImage)
        #expect(!DAMAsset(kind: "video", filename: "a.png", absolutePath: "/a.png").isEditableImage)
    }
}
