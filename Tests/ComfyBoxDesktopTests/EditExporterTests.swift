// EditExporterTests.swift
import Testing
import Foundation
import CoreGraphics
import ImageIO
@testable import ComfyBoxDesktop

@Suite("EditExporter")
struct EditExporterTests {
    func tempDir() -> String {
        let d = NSTemporaryDirectory() + "editexport-" + UUID().uuidString
        try! FileManager.default.createDirectory(atPath: d, withIntermediateDirectories: true)
        return d
    }

    @Test("sidecarObject copies generation fields and writes the edit block")
    func sidecarObject() {
        let asset = DAMAsset(id: "A1", filename: "a.png", absolutePath: "/orig/a.png", prompt: "a cat", negativePrompt: "dog",
                             seed: 42, steps: 9, guidance: 3.5, modelFamily: "krea2", contentMode: "apple", characterName: "Kira")
        var recipe = EditRecipe(); recipe.adjustments.contrast = 0.3
        let obj = EditExporter.sidecarObject(source: asset, sourcePath: "/orig/a.png", recipe: recipe, now: Date(timeIntervalSince1970: 0))
        #expect(obj["prompt"] as? String == "a cat")
        #expect(obj["negative_prompt"] as? String == "dog")
        #expect(obj["seed"] as? Int == 42 && obj["steps"] as? Int == 9)
        #expect(obj["guidance"] as? Double == 3.5)
        #expect(obj["model_family"] as? String == "krea2")
        #expect(obj["content_mode"] as? String == "apple")
        #expect(obj["character_name"] as? String == "Kira")
        #expect(obj["source"] as? String == "desktop-edit")
        let edit = obj["edit"] as? [String: Any]
        #expect(edit?["source_path"] as? String == "/orig/a.png")
        #expect(edit?["source_asset_id"] as? String == "A1")
        #expect(edit?["version"] as? Int == 1)
        let recipeData = try! JSONSerialization.data(withJSONObject: edit!["recipe"]!)
        #expect(try! JSONDecoder().decode(EditRecipe.self, from: recipeData) == recipe)
    }

    @Test("outputPath suffixes on collision")
    func collision() throws {
        let dir = tempDir()
        let p1 = EditExporter.outputPath(in: dir, seconds: 1234)
        #expect(p1 == dir + "/edit-1234.png")
        FileManager.default.createFile(atPath: p1, contents: Data())
        #expect(EditExporter.outputPath(in: dir, seconds: 1234) == dir + "/edit-1234-2.png")
        FileManager.default.createFile(atPath: dir + "/edit-1234-2.png", contents: Data())
        #expect(EditExporter.outputPath(in: dir, seconds: 1234) == dir + "/edit-1234-3.png")
    }

    @Test("export writes a PNG of the rendered size and a sidecar that reopens")
    func exportWrites() async throws {
        let dir = tempDir()
        let src = EditTestSupport.horizontalGradient(width: 80, height: 40)
        var recipe = EditRecipe(); recipe.geometry.crop = CGRect(x: 0, y: 0, width: 0.5, height: 1)
        let out = try await EditExporter.export(sourceImage: src, sourcePath: "/orig/g.png", sourceAsset: nil,
                                                recipe: recipe, subjectMask: nil, outputDirectory: dir, ingestor: nil)
        #expect(out.hasPrefix(dir + "/edit-") && out.hasSuffix(".png"))
        let cg = CGImageSourceCreateWithURL(URL(fileURLWithPath: out) as CFURL, nil).flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) }
        #expect(cg?.width == 40 && cg?.height == 40)
        let sc = EditSidecar.read(forImageAt: out)
        #expect(sc?.recipe == recipe && sc?.sourcePath == "/orig/g.png")
    }

    @Test("exporting a derived asset points the sidecar at the root source")
    func rootChain() async throws {
        let dir = tempDir()
        let src = EditTestSupport.solid(r: 1, g: 2, b: 3, width: 8, height: 8)
        let first = try await EditExporter.export(sourceImage: src, sourcePath: "/orig/root.png", sourceAsset: nil,
                                                  recipe: EditRecipe(), subjectMask: nil, outputDirectory: dir, ingestor: nil)
        let second = try await EditExporter.export(sourceImage: src, sourcePath: first, sourceAsset: nil,
                                                   recipe: EditRecipe(), subjectMask: nil, outputDirectory: dir, ingestor: nil)
        #expect(EditSidecar.read(forImageAt: second)?.sourcePath == "/orig/root.png")
    }
}
