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
    func writeJPEG(_ image: CGImage, to path: String, properties: [CFString: Any] = [:]) {
        let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.jpeg.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, properties as CFDictionary)
        CGImageDestinationFinalize(dest)
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
        #expect(!s.isDirty)   // back to the recipe as loaded (0.0), regardless of undo history
        s.redo(); #expect(s.recipe.adjustments.exposure == 0.5 && s.canRedo)
        #expect(s.isDirty)    // diverged from the loaded recipe again
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

    @Test("suppressCropForPreview affects only the preview; export always honors the real crop")
    func suppressCropForPreview() async throws {
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
        s.commit()
        let exported = try await s.export(outputDirectory: dir, ingestor: nil)
        let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: exported) as CFURL, nil)!
        let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)!
        #expect(cg.width == 32)   // export ignores suppressCropForPreview and honors the real crop
    }

    // MARK: - Fix round 1

    @Test("export snapshots the recipe at call time; a mutation during the write doesn't change what's written or mark the session clean")
    func exportSnapshotsRecipeAtCallTime() async throws {
        let dir = tempDir(); let p = dir + "/src.png"
        writePNG(EditTestSupport.solid(r: 1, g: 1, b: 1, width: 8, height: 8), to: p)
        let s = EditSession(sourcePath: p, sourceAsset: nil)
        await s.load()
        s.set { $0.adjustments.exposure = 0.5 }; s.commit()
        #expect(s.isDirty)   // committed but not yet exported: differs from the loaded (identity) recipe

        let exportTask = Task { try await s.export(outputDirectory: dir, ingestor: nil) }
        await Task.yield()
        // Mutate while the export (specifically its internal await, off the main
        // actor) is still in flight.
        s.set { $0.adjustments.exposure = 0.9 }
        let path = try await exportTask.value

        #expect(s.isDirty)   // live recipe (0.9) has diverged from what was actually exported (0.5)
        let sidecar = EditSidecar.read(forImageAt: path)
        #expect(sidecar?.recipe.adjustments.exposure == 0.5)
    }

    @Test("export lineage after reopen uses the root path and the root's own asset id, not the reopened session's asset")
    func lineageOnReopenUsesRootAssetId() async throws {
        let dir = tempDir(); let root = dir + "/root.png"
        writePNG(EditTestSupport.horizontalGradient(width: 40, height: 20), to: root)
        let rootAsset = DAMAsset(id: "ROOT1", filename: "root.png", absolutePath: root)
        let s0 = EditSession(sourcePath: root, sourceAsset: rootAsset)
        await s0.load()
        s0.set { $0.adjustments.vibrance = 0.2 }; s0.commit()
        let derived = try await s0.export(outputDirectory: dir, ingestor: nil)

        let derivedAsset = DAMAsset(id: "DERIVED1", filename: (derived as NSString).lastPathComponent, absolutePath: derived)
        let s1 = EditSession(sourcePath: derived, sourceAsset: derivedAsset)
        await s1.load()
        s1.set { $0.adjustments.vibrance = 0.4 }; s1.commit()
        let derived2 = try await s1.export(outputDirectory: dir, ingestor: nil)

        let sc = EditSidecar.read(forImageAt: derived2)
        #expect(sc?.sourcePath == root)
        #expect(sc?.sourceAssetId == "ROOT1")
    }

    @Test("EXIF orientation is baked into sourceImage on load")
    func exifOrientationAppliedOnLoad() async {
        let dir = tempDir(); let p = dir + "/src.jpg"
        writeJPEG(EditTestSupport.horizontalGradient(width: 64, height: 32), to: p,
                  properties: [kCGImagePropertyOrientation: 6])   // "right": 90° CW, swaps dimensions
        let s = EditSession(sourcePath: p, sourceAsset: nil)
        await s.load()
        #expect(s.sourceImage?.width == 32)
        #expect(s.sourceImage?.height == 64)
    }

    @Test("a sidecar with a newer, undecodable recipe schema is recognized via the envelope, not mistaken for 'no sidecar'")
    func newerVersionIncompatibleSchemaEnvelopePreCheck() async throws {
        let dir = tempDir(); let root = dir + "/root.png"; let derived = dir + "/edit-9.png"
        writePNG(EditTestSupport.solid(r: 5, g: 5, b: 5, width: 8, height: 8), to: root)
        writePNG(EditTestSupport.solid(r: 5, g: 5, b: 5, width: 8, height: 8), to: derived)
        let obj: [String: Any] = [
            "version": 99,
            "source_path": root,
            "source_asset_id": NSNull(),
            "recipe": ["bogus": true],
            "editor": "x",
            "created_at": ISO8601DateFormatter().string(from: Date())
        ]
        try JSONSerialization.data(withJSONObject: ["edit": obj]).write(to: URL(fileURLWithPath: dir + "/edit-9.json"))
        let s = EditSession(sourcePath: derived, sourceAsset: nil)
        await s.load()
        #expect(s.recipe.isIdentity)
        #expect(s.warning != nil)
        #expect(s.sourceImage != nil)
        #expect(s.sourcePath == root)   // resolved through to the root pixels, not stuck on the derived file
    }

    @Test("removeBackground without a subject mask sets subjectMaskWarning after a preview render, and clears it once turned off")
    func removeBackgroundWithoutMaskWarnsAfterRender() async {
        let dir = tempDir(); let p = dir + "/src.png"
        writePNG(EditTestSupport.solid(r: 1, g: 1, b: 1, width: 8, height: 8), to: p)
        let s = EditSession(sourcePath: p, sourceAsset: nil)
        await s.load()
        s.set { $0.subject.removeBackground = true }
        await s.renderNow()
        #expect(s.subjectMaskWarning == "Remove Background is on but no subject mask is loaded. Run Find Subject.")
        s.set { $0.subject.removeBackground = false }
        await s.renderNow()
        #expect(s.subjectMaskWarning == nil)
    }

    @Test("export refuses when removeBackground is on without a subject mask")
    func exportRefusesRemoveBackgroundWithoutMask() async {
        let dir = tempDir(); let p = dir + "/src.png"
        writePNG(EditTestSupport.solid(r: 1, g: 1, b: 1, width: 8, height: 8), to: p)
        let s = EditSession(sourcePath: p, sourceAsset: nil)
        await s.load()
        s.set { $0.subject.removeBackground = true }; s.commit()
        do {
            _ = try await s.export(outputDirectory: dir, ingestor: nil)
            Issue.record("expected export to throw")
        } catch let error as EditExportError {
            guard case .writeFailed(let message) = error else {
                Issue.record("wrong EditExportError case: \(error)"); return
            }
            #expect(message.contains("Remove Background"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    // MARK: - Fix round 2

    @Test("a multi-hop chain through a newer, undecodable node still resolves to the true root")
    func multiHopChainThroughIncompatibleNodeResolvesRoot() async throws {
        let dir = tempDir()
        let root = dir + "/root.png"; let prior = dir + "/edit-1.png"; let newer = dir + "/edit-2.png"
        writePNG(EditTestSupport.solid(r: 5, g: 5, b: 5, width: 8, height: 8), to: root)
        writePNG(EditTestSupport.solid(r: 5, g: 5, b: 5, width: 8, height: 8), to: prior)
        writePNG(EditTestSupport.solid(r: 5, g: 5, b: 5, width: 8, height: 8), to: newer)
        // prior -> root: fully valid, compatible version.
        let priorSc = EditSidecar(version: 1, sourcePath: root, sourceAssetId: "ROOT1", recipe: EditRecipe(), editor: "x", createdAt: Date())
        try JSONSerialization.data(withJSONObject: ["edit": try priorSc.jsonObject()]).write(to: URL(fileURLWithPath: dir + "/edit-1.json"))
        // newer -> prior: version 99, recipe undecodable by this build.
        let obj: [String: Any] = [
            "version": 99,
            "source_path": prior,
            "source_asset_id": NSNull(),
            "recipe": ["bogus": true],
            "editor": "x",
            "created_at": ISO8601DateFormatter().string(from: Date())
        ]
        try JSONSerialization.data(withJSONObject: ["edit": obj]).write(to: URL(fileURLWithPath: dir + "/edit-2.json"))
        let s = EditSession(sourcePath: newer, sourceAsset: nil)
        await s.load()
        #expect(s.sourcePath == root)          // resolved through prior all the way to root, not just one hop
        #expect(s.recipe.isIdentity)
        #expect(s.warning != nil)
        #expect(s.sourceImage != nil)
    }

    // MARK: - Final fix wave

    @Test("X2: export after a missing-root fallback points the new sidecar at the flattened file, not the missing root")
    func missingRootExportPointsAtFlattenedFile() async throws {
        let dir = tempDir(); let derived = dir + "/edit-9.png"
        writePNG(EditTestSupport.solid(r: 5, g: 5, b: 5, width: 8, height: 8), to: derived)
        let sc = EditSidecar(version: 1, sourcePath: dir + "/gone.png", sourceAssetId: nil, recipe: EditRecipe(), editor: "x", createdAt: Date())
        try JSONSerialization.data(withJSONObject: ["edit": try sc.jsonObject()]).write(to: URL(fileURLWithPath: dir + "/edit-9.json"))
        let s = EditSession(sourcePath: derived, sourceAsset: nil)
        await s.load()
        #expect(s.sourcePath == derived && s.warning != nil)   // load() already falls back correctly (pre-existing)
        let out = try await s.export(outputDirectory: dir, ingestor: nil)
        let outSc = EditSidecar.read(forImageAt: out)
        // The bug: without `followLineage = false`, the exporter re-walks the SAME
        // broken chain from `openedPath` and reports the missing, unreachable root
        // as the new sidecar's source — even though the pixels actually rendered
        // were the flattened ones at `derived`.
        #expect(outSc?.sourcePath == derived)
    }

    @Test("M10: undo commits an uncommitted live edit first, so redo can restore it")
    func undoCommitsLiveEditFirst() async {
        let dir = tempDir(); let p = dir + "/src.png"
        writePNG(EditTestSupport.solid(r: 1, g: 1, b: 1, width: 8, height: 8), to: p)
        let s = EditSession(sourcePath: p, sourceAsset: nil)
        await s.load()
        s.set { $0.adjustments.exposure = 0.5 }; s.commit()
        s.set { $0.adjustments.exposure = 0.9 }   // live drag, not yet committed
        #expect(s.isDirty)
        s.undo()
        #expect(s.recipe.adjustments.exposure == 0.5)   // reverted to the last committed step, not silently dropped
        s.redo()
        #expect(s.recipe.adjustments.exposure == 0.9)   // the uncommitted drag is recoverable, not lost
    }

    @Test("M2: a subsequent successful render clears a stale 'Preview render failed' warning")
    func previewRenderFailureWarningClears() async {
        let dir = tempDir(); let p = dir + "/src.png"
        writePNG(EditTestSupport.solid(r: 1, g: 1, b: 1, width: 8, height: 8), to: p)
        let s = EditSession(sourcePath: p, sourceAsset: nil)
        await s.load()
        s.set { $0.geometry.crop = CGRect(x: 0, y: 0, width: 0, height: 0) }   // renders to an empty extent
        await s.renderNow()
        #expect(s.warning == "Preview render failed; the last good preview is shown.")
        s.set { $0.geometry.crop = nil }
        await s.renderNow()
        #expect(s.warning == nil)
    }
}
