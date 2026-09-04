// EditExporter.swift — full-resolution render → PNG + sidecar → ingest
//
// Write order guarantees no partial state: PNG first, sidecar second (PNG
// removed if the sidecar fails), ingest last (files stay if ingest fails;
// the gallery poller picks them up).

import Foundation
import CoreImage
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

public enum EditExportError: Error {
    case renderFailed
    case writeFailed(String)
}

public enum EditExporter {

    public static func sidecarObject(source: DAMAsset?, sourcePath: String, recipe: EditRecipe, now: Date) -> [String: Any] {
        var obj: [String: Any] = ["source": "desktop-edit"]
        if let s = source {
            if let v = s.prompt { obj["prompt"] = v }
            if let v = s.negativePrompt { obj["negative_prompt"] = v }
            if let v = s.seed { obj["seed"] = v }
            if let v = s.steps { obj["steps"] = v }
            if let v = s.guidance { obj["guidance"] = v }
            if let v = s.modelFamily { obj["model_family"] = v }
            if let v = s.contentMode { obj["content_mode"] = v }
            if let v = s.characterName { obj["character_name"] = v }
        }
        let root = EditSidecar.rootSource(forImageAt: sourcePath)
        let rootAssetId = root.path == sourcePath ? source?.id : root.assetId
        let sc = EditSidecar(version: EditRecipe.currentVersion, sourcePath: root.path, sourceAssetId: rootAssetId,
                             recipe: recipe, editor: "ComfyBoxDesktop", createdAt: now)
        obj["edit"] = sc.jsonObject
        return obj
    }

    public static func outputPath(in directory: String, seconds: Int) -> String {
        let base = (directory as NSString).appendingPathComponent("edit-\(seconds)")
        var candidate = base + ".png"
        var n = 2
        while FileManager.default.fileExists(atPath: candidate) {
            candidate = base + "-\(n).png"; n += 1
        }
        return candidate
    }

    public static func export(sourceImage: CGImage, sourcePath: String, sourceAsset: DAMAsset?,
                              recipe: EditRecipe, subjectMask: CIImage?,
                              outputDirectory: String, ingestor: AssetIngestor?) async throws -> String {
        let context = CIContext(options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
        let output = EditRenderer.render(source: CIImage(cgImage: sourceImage), recipe: recipe, subjectMask: subjectMask)
        guard let cg = context.createCGImage(output, from: output.extent) else { throw EditExportError.renderFailed }

        try FileManager.default.createDirectory(atPath: outputDirectory, withIntermediateDirectories: true)
        let pngPath = outputPath(in: outputDirectory, seconds: Int(Date().timeIntervalSince1970))
        guard let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: pngPath) as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw EditExportError.writeFailed("could not create \(pngPath)") }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw EditExportError.writeFailed("could not finalize \(pngPath)") }

        let sidecar = sidecarObject(source: sourceAsset, sourcePath: sourcePath, recipe: recipe, now: Date())
        do {
            let data = try JSONSerialization.data(withJSONObject: sidecar, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: EditSidecar.sidecarPath(forImageAt: pngPath)))
        } catch {
            try? FileManager.default.removeItem(atPath: pngPath)
            throw EditExportError.writeFailed("sidecar: \(error.localizedDescription)")
        }

        if let ingestor { _ = try await ingestor.ingestFile(at: pngPath) }
        return pngPath
    }
}
