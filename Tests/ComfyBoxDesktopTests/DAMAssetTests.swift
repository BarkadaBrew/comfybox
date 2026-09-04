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
