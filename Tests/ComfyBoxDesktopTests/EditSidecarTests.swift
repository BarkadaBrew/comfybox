// EditSidecarTests.swift
import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("EditSidecar")
struct EditSidecarTests {
    func tempDir() -> String {
        let d = NSTemporaryDirectory() + "editsidecar-" + UUID().uuidString
        try! FileManager.default.createDirectory(atPath: d, withIntermediateDirectories: true)
        return d
    }

    @Test("reads an edit block and ignores files without one")
    func readBlock() throws {
        let dir = tempDir()
        var recipe = EditRecipe(); recipe.adjustments.exposure = 0.7
        let sc = EditSidecar(version: 1, sourcePath: "/orig/a.png", sourceAssetId: "A1", recipe: recipe, editor: "ComfyBoxDesktop", createdAt: Date(timeIntervalSince1970: 1_700_000_000))
        let json: [String: Any] = ["prompt": "x", "edit": sc.jsonObject]
        try JSONSerialization.data(withJSONObject: json).write(to: URL(fileURLWithPath: dir + "/edit-1.json"))
        let back = EditSidecar.read(forImageAt: dir + "/edit-1.png")
        #expect(back?.sourcePath == "/orig/a.png")
        #expect(back?.sourceAssetId == "A1")
        #expect(back?.recipe == recipe)
        try JSONSerialization.data(withJSONObject: ["prompt": "y"]).write(to: URL(fileURLWithPath: dir + "/plain.json"))
        #expect(EditSidecar.read(forImageAt: dir + "/plain.png") == nil)
        #expect(EditSidecar.read(forImageAt: dir + "/missing.png") == nil)
    }

    @Test("rootSource follows the chain to the original")
    func root() throws {
        let dir = tempDir()
        let first = EditSidecar(version: 1, sourcePath: "/orig/a.png", sourceAssetId: "A1", recipe: EditRecipe(), editor: "ComfyBoxDesktop", createdAt: Date())
        try JSONSerialization.data(withJSONObject: ["edit": first.jsonObject]).write(to: URL(fileURLWithPath: dir + "/edit-1.json"))
        let second = EditSidecar(version: 1, sourcePath: dir + "/edit-1.png", sourceAssetId: "E1", recipe: EditRecipe(), editor: "ComfyBoxDesktop", createdAt: Date())
        try JSONSerialization.data(withJSONObject: ["edit": second.jsonObject]).write(to: URL(fileURLWithPath: dir + "/edit-2.json"))
        let r = EditSidecar.rootSource(forImageAt: dir + "/edit-2.png")
        #expect(r.path == "/orig/a.png" && r.assetId == "A1")
        let plain = EditSidecar.rootSource(forImageAt: "/orig/a.png")
        #expect(plain.path == "/orig/a.png" && plain.assetId == nil)
    }

    @Test("a newer version still parses so the caller can decide to drop it")
    func newerVersion() throws {
        let dir = tempDir()
        var obj = EditSidecar(version: 1, sourcePath: "/o.png", sourceAssetId: nil, recipe: EditRecipe(), editor: "x", createdAt: Date()).jsonObject
        obj["version"] = 99
        try JSONSerialization.data(withJSONObject: ["edit": obj]).write(to: URL(fileURLWithPath: dir + "/e.json"))
        #expect(EditSidecar.read(forImageAt: dir + "/e.png")?.version == 99)
    }
}
