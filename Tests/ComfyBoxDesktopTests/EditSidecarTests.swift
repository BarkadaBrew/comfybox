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
        let json: [String: Any] = ["prompt": "x", "edit": try sc.jsonObject()]
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
        try JSONSerialization.data(withJSONObject: ["edit": try first.jsonObject()]).write(to: URL(fileURLWithPath: dir + "/edit-1.json"))
        let second = EditSidecar(version: 1, sourcePath: dir + "/edit-1.png", sourceAssetId: "E1", recipe: EditRecipe(), editor: "ComfyBoxDesktop", createdAt: Date())
        try JSONSerialization.data(withJSONObject: ["edit": try second.jsonObject()]).write(to: URL(fileURLWithPath: dir + "/edit-2.json"))
        let r = EditSidecar.rootSource(forImageAt: dir + "/edit-2.png")
        #expect(r.path == "/orig/a.png" && r.assetId == "A1")
        let plain = EditSidecar.rootSource(forImageAt: "/orig/a.png")
        #expect(plain.path == "/orig/a.png" && plain.assetId == nil)
    }

    @Test("a newer version still parses so the caller can decide to drop it")
    func newerVersion() throws {
        let dir = tempDir()
        var obj = try EditSidecar(version: 1, sourcePath: "/o.png", sourceAssetId: nil, recipe: EditRecipe(), editor: "x", createdAt: Date()).jsonObject()
        obj["version"] = 99
        try JSONSerialization.data(withJSONObject: ["edit": obj]).write(to: URL(fileURLWithPath: dir + "/e.json"))
        #expect(EditSidecar.read(forImageAt: dir + "/e.png")?.version == 99)
    }

    // MARK: - Fix round 1

    @Test("a long acyclic chain resolves to the true root regardless of depth")
    func longChainResolvesToRoot() throws {
        let dir = tempDir()
        // Build a 40-hop chain: edit-40 -> edit-39 -> ... -> edit-1 -> /orig/root.png.
        // 40 exceeds the old fixed hop cap (32), so this fails under the old
        // implementation and passes only once the cap is gone.
        var upstream = "/orig/root.png"
        for i in 1...40 {
            let sc = EditSidecar(version: 1, sourcePath: upstream, sourceAssetId: nil, recipe: EditRecipe(), editor: "x", createdAt: Date())
            try JSONSerialization.data(withJSONObject: ["edit": try sc.jsonObject()])
                .write(to: URL(fileURLWithPath: dir + "/edit-\(i).json"))
            upstream = dir + "/edit-\(i).png"
        }
        let r = EditSidecar.rootSource(forImageAt: dir + "/edit-40.png")
        #expect(r.path == "/orig/root.png")
    }

    @Test("a cycle returns the starting path with a nil asset id, not an arbitrary point in the loop")
    func cycleReturnsStartingPathWithNilId() throws {
        let dir = tempDir()
        let a = dir + "/edit-a.png"
        let b = dir + "/edit-b.png"
        let scA = EditSidecar(version: 1, sourcePath: b, sourceAssetId: "B", recipe: EditRecipe(), editor: "x", createdAt: Date())
        try JSONSerialization.data(withJSONObject: ["edit": try scA.jsonObject()])
            .write(to: URL(fileURLWithPath: dir + "/edit-a.json"))
        let scB = EditSidecar(version: 1, sourcePath: a, sourceAssetId: "A", recipe: EditRecipe(), editor: "x", createdAt: Date())
        try JSONSerialization.data(withJSONObject: ["edit": try scB.jsonObject()])
            .write(to: URL(fileURLWithPath: dir + "/edit-b.json"))
        let r = EditSidecar.rootSource(forImageAt: a)
        #expect(r.path == a)
        #expect(r.assetId == nil)
    }

    // MARK: - Fix round 1 (EditSession review, finding 4)

    @Test("readEnvelope parses version/source_path/source_asset_id even when recipe is not decodable")
    func readEnvelopeSurvivesUndecodableRecipe() throws {
        let dir = tempDir()
        let obj: [String: Any] = [
            "version": 99,
            "source_path": "/orig/root.png",
            "source_asset_id": "R1",
            "recipe": ["bogus": true],
            "editor": "x",
            "created_at": ISO8601DateFormatter().string(from: Date())
        ]
        try JSONSerialization.data(withJSONObject: ["edit": obj]).write(to: URL(fileURLWithPath: dir + "/edit-9.json"))
        // The full decode fails on the incompatible recipe shape...
        #expect(EditSidecar.read(forImageAt: dir + "/edit-9.png") == nil)
        // ...but the envelope still parses.
        let envelope = EditSidecar.readEnvelope(forImageAt: dir + "/edit-9.png")
        #expect(envelope?.version == 99)
        #expect(envelope?.sourcePath == "/orig/root.png")
        #expect(envelope?.sourceAssetId == "R1")
    }
}
