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
    func sidecarObject() throws {
        let asset = DAMAsset(id: "A1", filename: "a.png", absolutePath: "/orig/a.png", prompt: "a cat", negativePrompt: "dog",
                             seed: 42, steps: 9, guidance: 3.5, modelFamily: "krea2", contentMode: "apple", characterName: "Kira")
        var recipe = EditRecipe(); recipe.adjustments.contrast = 0.3
        let obj = try EditExporter.sidecarObject(source: asset, sourcePath: "/orig/a.png", recipe: recipe, now: Date(timeIntervalSince1970: 0))
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

    // MARK: - Fix round 1

    @Test("sidecarObject always writes source_asset_id, explicit JSON null when there is no source asset")
    func sidecarObjectExplicitNullAssetId() throws {
        let objNoSource = try EditExporter.sidecarObject(source: nil, sourcePath: "/orig/a.png", recipe: EditRecipe(), now: Date())
        let editNoSource = objNoSource["edit"] as? [String: Any]
        #expect((editNoSource?["source_asset_id"]).map { $0 is NSNull } == true)

        let asset = DAMAsset(id: "A9", filename: "a.png", absolutePath: "/orig/a.png")
        let objWithSource = try EditExporter.sidecarObject(source: asset, sourcePath: "/orig/a.png", recipe: EditRecipe(), now: Date())
        let editWithSource = objWithSource["edit"] as? [String: Any]
        #expect(editWithSource?["source_asset_id"] as? String == "A9")
    }

    @Test("outputPath treats a lone leftover sidecar as taken even without a paired PNG")
    func outputPathSidecarOnlyCollision() throws {
        let dir = tempDir()
        FileManager.default.createFile(atPath: dir + "/edit-1234.json", contents: Data())
        #expect(EditExporter.outputPath(in: dir, seconds: 1234) == dir + "/edit-1234-2.png")
    }

    @Test("export throws and leaves no files when the recipe cannot be encoded")
    func exportUnencodableRecipeLeavesNoFiles() async throws {
        let dir = tempDir()
        let src = EditTestSupport.solid(r: 1, g: 2, b: 3, width: 4, height: 4)
        var recipe = EditRecipe(); recipe.adjustments.exposure = .nan
        var thrown: Error?
        do {
            _ = try await EditExporter.export(sourceImage: src, sourcePath: "/orig/n.png", sourceAsset: nil,
                                              recipe: recipe, subjectMask: nil, outputDirectory: dir, ingestor: nil)
        } catch {
            thrown = error
        }
        #expect(thrown != nil)
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
        #expect(contents.isEmpty)
    }

    @Test("export throws and creates nothing when the output directory cannot be created")
    func exportUncreatableDirectoryLeavesNoFiles() async throws {
        let dir = tempDir()
        let blockingFile = dir + "/not-a-directory"
        FileManager.default.createFile(atPath: blockingFile, contents: Data("x".utf8))
        let badOutputDir = blockingFile + "/nested"
        let src = EditTestSupport.solid(r: 1, g: 2, b: 3, width: 4, height: 4)
        var thrown: Error?
        do {
            _ = try await EditExporter.export(sourceImage: src, sourcePath: "/orig/n.png", sourceAsset: nil,
                                              recipe: EditRecipe(), subjectMask: nil, outputDirectory: badOutputDir, ingestor: nil)
        } catch {
            thrown = error
        }
        #expect(thrown != nil)
        #expect(!FileManager.default.fileExists(atPath: badOutputDir))
    }

    @Test("EditExportError carries a useful localized description")
    func localizedErrorDescriptions() {
        #expect(EditExportError.renderFailed.localizedDescription.isEmpty == false)
        #expect(EditExportError.writeFailed("disk full").localizedDescription.contains("disk full"))
    }

    // MARK: - Fix round 2

    @Test("cleanupReserved throws a message naming the path when removal fails")
    func cleanupReservedReportsFailure() throws {
        let dir = tempDir()
        // No such file, and its parent directory doesn't exist either, so
        // `FileManager.removeItem` is guaranteed to fail regardless of
        // process privilege (unlike a permission-bit test, which root can
        // bypass).
        let path = dir + "/missing-parent/reserved.png"
        var thrown: Error?
        do {
            try EditExporter.cleanupReserved(path)
        } catch {
            thrown = error
        }
        #expect(thrown != nil)
        #expect(thrown?.localizedDescription.contains(path) == true)
    }

    // MARK: - Final fix wave (X1, X2, M13)

    @Test("X1: a successful export leaves no .part temp file behind")
    func exportLeavesNoPartFileOnSuccess() async throws {
        let dir = tempDir()
        let src = EditTestSupport.solid(r: 1, g: 2, b: 3, width: 8, height: 8)
        let out = try await EditExporter.export(sourceImage: src, sourcePath: "/orig/g.png", sourceAsset: nil,
                                                recipe: EditRecipe(), subjectMask: nil, outputDirectory: dir, ingestor: nil)
        let contents = try FileManager.default.contentsOfDirectory(atPath: dir)
        #expect(contents.contains { $0.hasSuffix(".part") } == false)
        #expect(FileManager.default.fileExists(atPath: out))
        #expect(FileManager.default.fileExists(atPath: EditSidecar.sidecarPath(forImageAt: out)))
    }

    @Test("X1: a sidecar failure leaves no .part temp file and no PNG behind")
    func exportLeavesNoPartFileOnSidecarFailure() async throws {
        let dir = tempDir()
        let src = EditTestSupport.solid(r: 1, g: 2, b: 3, width: 4, height: 4)
        var recipe = EditRecipe(); recipe.adjustments.exposure = .nan
        var thrown: Error?
        do {
            _ = try await EditExporter.export(sourceImage: src, sourcePath: "/orig/n.png", sourceAsset: nil,
                                              recipe: recipe, subjectMask: nil, outputDirectory: dir, ingestor: nil)
        } catch {
            thrown = error
        }
        #expect(thrown != nil)
        let contents = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
        #expect(contents.isEmpty)
    }

    @Test("X2: resolveLineage false anchors the sidecar at sourcePath itself, with no chain walk")
    func sidecarObjectResolveLineageFalse() throws {
        let asset = DAMAsset(id: "FLAT1", filename: "flat.png", absolutePath: "/orig/flat.png")
        // sourcePath's own sidecar (if any existed) would normally be walked by
        // rootSource; resolveLineage: false must skip that walk entirely and use
        // sourcePath/asset id verbatim.
        let obj = try EditExporter.sidecarObject(source: asset, sourcePath: "/orig/flat.png",
                                                 recipe: EditRecipe(), now: Date(), resolveLineage: false)
        let edit = obj["edit"] as? [String: Any]
        #expect(edit?["source_path"] as? String == "/orig/flat.png")
        #expect(edit?["source_asset_id"] as? String == "FLAT1")
    }

    @Test("M13: sidecarObject copies the source's own loras array through when present")
    func sidecarObjectCopiesLoras() throws {
        let dir = tempDir()
        let sourcePath = dir + "/orig.png"
        let sourceSidecar: [String: Any] = ["loras": [["name": "styleA", "scale": 0.8], ["name": "styleB", "scale": 1.0]]]
        try JSONSerialization.data(withJSONObject: sourceSidecar).write(to: URL(fileURLWithPath: dir + "/orig.json"))
        let obj = try EditExporter.sidecarObject(source: nil, sourcePath: sourcePath, recipe: EditRecipe(), now: Date())
        let loras = obj["loras"] as? [[String: Any]]
        #expect(loras?.count == 2)
        #expect(loras?.first?["name"] as? String == "styleA")
    }

    @Test("M13: sidecarObject omits loras when the source has none")
    func sidecarObjectOmitsLorasWhenAbsent() throws {
        let obj = try EditExporter.sidecarObject(source: nil, sourcePath: "/orig/no-sidecar.png", recipe: EditRecipe(), now: Date())
        #expect(obj["loras"] == nil)
    }
}
