// EditExporter.swift — full-resolution render → PNG + sidecar → ingest
//
// Write order guarantees no partial state: the output path is reserved
// (exclusive create) first, then rendered into, then the sidecar is written
// atomically second (PNG *and* any partial sidecar removed if the sidecar
// fails), ingest last (files stay if ingest fails; the gallery poller picks
// them up).

import Foundation
import CoreImage
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Darwin

public enum EditExportError: Error {
    case renderFailed
    case writeFailed(String)
}

extension EditExportError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .renderFailed:
            return "The edit could not be rendered."
        case .writeFailed(let message):
            return "The edit could not be saved: \(message)"
        }
    }
}

public enum EditExporter {

    public static func sidecarObject(source: DAMAsset?, sourcePath: String, recipe: EditRecipe, now: Date) throws -> [String: Any] {
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
        obj["edit"] = try sc.jsonObject()
        return obj
    }

    /// Next free `edit-<seconds>.png` path in `directory`. A candidate slot
    /// counts as taken if EITHER the `.png` or its `.json` sidecar already
    /// exists — a lone leftover sidecar (from an interrupted export, or a
    /// hand-placed file) must not be silently overwritten by a PNG that
    /// pairs with it under a *different* edit. This is a pure name lookup
    /// for display/preview purposes; `export` does its own exclusive-create
    /// reservation over the same numbering to close the check-then-write
    /// race between two concurrent exports (see `reserveOutputPath`).
    public static func outputPath(in directory: String, seconds: Int) -> String {
        let base = (directory as NSString).appendingPathComponent("edit-\(seconds)")
        let fm = FileManager.default
        func candidate(_ n: Int) -> String { n == 1 ? base + ".png" : base + "-\(n).png" }
        func taken(_ path: String) -> Bool {
            fm.fileExists(atPath: path) || fm.fileExists(atPath: EditSidecar.sidecarPath(forImageAt: path))
        }
        var n = 1
        while taken(candidate(n)) { n += 1 }
        return candidate(n)
    }

    /// Reserves the next free `edit-<seconds>.png` path by creating it
    /// exclusively (`O_CREAT|O_EXCL`) so two concurrent exports can never
    /// claim the same output path — `outputPath`'s existence check alone
    /// leaves a window between "path looks free" and "file gets written".
    /// Uses the same numbering and same "png-or-json taken" rule as
    /// `outputPath`; a raced `EEXIST` just advances to the next candidate.
    /// Gives up after 100 candidates rather than looping forever.
    private static func reserveOutputPath(in directory: String, seconds: Int) throws -> String {
        let base = (directory as NSString).appendingPathComponent("edit-\(seconds)")
        let fm = FileManager.default
        func candidate(_ n: Int) -> String { n == 1 ? base + ".png" : base + "-\(n).png" }

        var n = 1
        var attempts = 0
        while attempts < 100 {
            let path = candidate(n)
            if fm.fileExists(atPath: EditSidecar.sidecarPath(forImageAt: path)) {
                n += 1; attempts += 1; continue
            }
            let fd = open(path, O_CREAT | O_EXCL | O_WRONLY, 0o644)
            if fd >= 0 {
                close(fd)
                return path
            }
            if errno == EEXIST {
                n += 1; attempts += 1; continue
            }
            throw EditExportError.writeFailed("could not reserve \(path): \(String(cString: strerror(errno)))")
        }
        throw EditExportError.writeFailed("could not find a free edit- output path in \(directory) after 100 attempts")
    }

    /// Removes a reserved-but-unused output file — e.g. the empty file
    /// `reserveOutputPath` left behind when the render meant to fill it
    /// fails. Throws with the path folded into the message (rather than
    /// relying on the underlying `FileManager` error's own wording, or
    /// swallowing the failure) so a genuine cleanup failure is reported and
    /// always identifies which file survived on disk. Internal, not
    /// private, so tests can exercise it directly via `@testable import`.
    static func cleanupReserved(_ path: String) throws {
        do {
            try FileManager.default.removeItem(atPath: path)
        } catch {
            throw EditExportError.writeFailed("could not remove reserved \(path): \(error.localizedDescription)")
        }
    }

    public static func export(sourceImage: CGImage, sourcePath: String, sourceAsset: DAMAsset?,
                              recipe: EditRecipe, subjectMask: CIImage?,
                              outputDirectory: String, ingestor: AssetIngestor?) async throws -> String {
        try FileManager.default.createDirectory(atPath: outputDirectory, withIntermediateDirectories: true)

        // Reserve the output path before doing any rendering work, so the
        // reservation — not a post-render existence check — is what
        // resolves a collision between two concurrent exports.
        let pngPath = try reserveOutputPath(in: outputDirectory, seconds: Int(Date().timeIntervalSince1970))

        func cleanupReservedPng(appendingTo message: String) -> String {
            do {
                try Self.cleanupReserved(pngPath)
                return message
            } catch {
                return message + "; failed to remove \(pngPath): \(error.localizedDescription)"
            }
        }

        let context = CIContext(options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
        let output = EditRenderer.render(source: CIImage(cgImage: sourceImage), recipe: recipe, subjectMask: subjectMask)
        guard let cg = context.createCGImage(output, from: output.extent) else {
            // Unlike the other failure points below, a render failure has no
            // natural message to append cleanup detail to — report the
            // cleanup outcome directly instead of discarding it.
            do {
                try Self.cleanupReserved(pngPath)
            } catch {
                throw EditExportError.writeFailed("render failed and could not remove reserved \(pngPath): \(error)")
            }
            throw EditExportError.renderFailed
        }

        guard let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: pngPath) as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw EditExportError.writeFailed(cleanupReservedPng(appendingTo: "could not create \(pngPath)"))
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw EditExportError.writeFailed(cleanupReservedPng(appendingTo: "could not finalize \(pngPath)"))
        }

        let sidecarPath = EditSidecar.sidecarPath(forImageAt: pngPath)
        do {
            let sidecar = try sidecarObject(source: sourceAsset, sourcePath: sourcePath, recipe: recipe, now: Date())
            let data = try JSONSerialization.data(withJSONObject: sidecar, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: sidecarPath), options: .atomic)
        } catch {
            var message = "sidecar: \(error.localizedDescription)"
            message = cleanupReservedPng(appendingTo: message)
            if FileManager.default.fileExists(atPath: sidecarPath) {
                do {
                    try FileManager.default.removeItem(atPath: sidecarPath)
                } catch let cleanupError {
                    message += "; failed to remove \(sidecarPath): \(cleanupError.localizedDescription)"
                }
            }
            throw EditExportError.writeFailed(message)
        }

        if let ingestor { _ = try await ingestor.ingestFile(at: pngPath) }
        return pngPath
    }
}
