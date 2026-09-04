// EditSessionTests.swift
import Testing
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import ComfyBoxDesktop

@Suite("EditSession")
@MainActor
struct EditSessionTests {
    func writePNG(_ image: CGImage, to path: String) {
        let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, nil); CGImageDestinationFinalize(dest)
    }
    func tempDir() -> String {
        let d = NSTemporaryDirectory() + "editsession-" + UUID().uuidString
        try! FileManager.default.createDirectory(atPath: d, withIntermediateDirectories: true)
        return d
    }

    @Test("load reads pixels and renders an identity preview")
    func loadAndPreview() async {
        let dir = tempDir(); let p = dir + "/src.png"
        writePNG(EditTestSupport.horizontalGradient(width: 64, height: 32), to: p)
        let s = EditSession(sourcePath: p, sourceAsset: nil, previewMaxDimension: 32)
        await s.load()
        #expect(s.sourceImage?.width == 64)
        await s.renderNow()
        #expect(s.preview != nil)
        #expect(s.preview!.width <= 32)     // downscaled for preview
        #expect(!s.isDirty && s.warning == nil)
    }

    @Test("set does not create an undo entry; commit does; undo/redo restore")
    func undoRedo() async {
        let dir = tempDir(); let p = dir + "/src.png"
        writePNG(EditTestSupport.solid(r: 1, g: 1, b: 1, width: 8, height: 8), to: p)
        let s = EditSession(sourcePath: p, sourceAsset: nil)
        await s.load()
        s.set { $0.adjustments.exposure = 0.5 }
        #expect(!s.canUndo && s.isDirty)
        s.commit()
        #expect(s.canUndo)
        s.set { $0.adjustments.exposure = 1.0 }; s.commit()
        s.undo(); #expect(s.recipe.adjustments.exposure == 0.5)
        s.undo(); #expect(s.recipe.adjustments.exposure == 0.0 && !s.canUndo)
        s.redo(); #expect(s.recipe.adjustments.exposure == 0.5 && s.canRedo)
        s.reset(); #expect(s.recipe.isIdentity && s.canUndo)
    }

    @Test("load on a derived asset opens the root pixels and the stored recipe")
    func reopen() async {
        let dir = tempDir(); let root = dir + "/root.png"
        writePNG(EditTestSupport.horizontalGradient(width: 40, height: 20), to: root)
        var recipe = EditRecipe(); recipe.adjustments.vibrance = 0.4
        let s0 = EditSession(sourcePath: root, sourceAsset: nil)
        await s0.load(); s0.set { $0 = recipe }; s0.commit()
        let derived = try! await s0.export(outputDirectory: dir, ingestor: nil)
        let s1 = EditSession(sourcePath: derived, sourceAsset: nil)
        await s1.load()
        #expect(s1.sourcePath == root)
        #expect(s1.recipe == recipe)
        #expect(s1.sourceImage?.width == 40)
    }

    @Test("newer recipe version loads pixels, drops the recipe, warns")
    func newerVersion() async throws {
        let dir = tempDir(); let root = dir + "/root.png"; let derived = dir + "/edit-9.png"
        writePNG(EditTestSupport.solid(r: 5, g: 5, b: 5, width: 8, height: 8), to: root)
        writePNG(EditTestSupport.solid(r: 5, g: 5, b: 5, width: 8, height: 8), to: derived)
        var obj = try EditSidecar(version: 1, sourcePath: root, sourceAssetId: nil, recipe: EditRecipe(), editor: "x", createdAt: Date()).jsonObject()
        obj["version"] = 99
        try JSONSerialization.data(withJSONObject: ["edit": obj]).write(to: URL(fileURLWithPath: dir + "/edit-9.json"))
        let s = EditSession(sourcePath: derived, sourceAsset: nil)
        await s.load()
        #expect(s.recipe.isIdentity && s.warning != nil && s.sourceImage != nil)
    }

    @Test("missing root falls back to derived pixels with a warning")
    func missingRoot() async throws {
        let dir = tempDir(); let derived = dir + "/edit-9.png"
        writePNG(EditTestSupport.solid(r: 5, g: 5, b: 5, width: 8, height: 8), to: derived)
        let sc = EditSidecar(version: 1, sourcePath: dir + "/gone.png", sourceAssetId: nil, recipe: EditRecipe(), editor: "x", createdAt: Date())
        try JSONSerialization.data(withJSONObject: ["edit": try sc.jsonObject()]).write(to: URL(fileURLWithPath: dir + "/edit-9.json"))
        let s = EditSession(sourcePath: derived, sourceAsset: nil)
        await s.load()
        #expect(s.sourcePath == derived && s.sourceImage != nil && s.warning != nil)
    }

    @Test("cyclic sidecar chain falls back to derived pixels with a warning")
    func cyclicChain() async throws {
        let dir = tempDir(); let a = dir + "/a.png"; let b = dir + "/b.png"
        writePNG(EditTestSupport.solid(r: 5, g: 5, b: 5, width: 8, height: 8), to: a)
        writePNG(EditTestSupport.solid(r: 9, g: 9, b: 9, width: 8, height: 8), to: b)
        let scA = EditSidecar(version: 1, sourcePath: b, sourceAssetId: nil, recipe: EditRecipe(), editor: "x", createdAt: Date())
        let scB = EditSidecar(version: 1, sourcePath: a, sourceAssetId: nil, recipe: EditRecipe(), editor: "x", createdAt: Date())
        try JSONSerialization.data(withJSONObject: ["edit": try scA.jsonObject()]).write(to: URL(fileURLWithPath: dir + "/a.json"))
        try JSONSerialization.data(withJSONObject: ["edit": try scB.jsonObject()]).write(to: URL(fileURLWithPath: dir + "/b.json"))
        let s = EditSession(sourcePath: a, sourceAsset: nil)
        await s.load()
        #expect(s.recipe.isIdentity)
        #expect(s.warning == "This edit's history is malformed; editing the flattened image.")
        #expect(s.sourceImage != nil)
    }

    @Test("suppressCropForPreview renders the uncropped frame; export ignores it")
    func suppressCropForPreview() async {
        let dir = tempDir(); let p = dir + "/src.png"
        writePNG(EditTestSupport.horizontalGradient(width: 64, height: 32), to: p)
        let s = EditSession(sourcePath: p, sourceAsset: nil)
        await s.load()
        s.set { $0.geometry.crop = CGRect(x: 0, y: 0, width: 0.5, height: 1) }
        s.suppressCropForPreview = true
        await s.renderNow()
        #expect(s.preview?.width == 64)
        s.suppressCropForPreview = false
        await s.renderNow()
        #expect(s.preview?.width == 32)
    }
}
