// EditExporter.swift — full-resolution render → PNG + sidecar → ingest
//
// Write order guarantees no partial state, and no partially-published one:
// the output path is reserved under a non-image `.part` temp name (exclusive
// create) first, rendered into, the sidecar written atomically under its
// real `.json` name second, then `.part` is renamed to `.png` — the single
// moment the finished image becomes visible to anything watching the output
// directory, by which point its sidecar already exists too. Any failure
// before the rename removes `.part` and any partial `.json`. Ingest is last
// (files stay if ingest fails; the gallery poller picks them up).

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

    /// - Parameter resolveLineage: true (the normal case) walks `sourcePath`'s OWN
    ///   sidecar chain to find the true root, so re-editing a derived asset stacks
    ///   on the original pixels. `EditSession` passes false when it already knows
    ///   the chain can't be trusted — a missing root or a malformed/cyclic chain,
    ///   where the pixels just edited were flattened rather than resolved — so the
    ///   new sidecar records `sourcePath` itself (and `source`'s own id) as the
    ///   lineage anchor instead of re-discovering the same broken chain.
    public static func sidecarObject(source: DAMAsset?, sourcePath: String, recipe: EditRecipe, now: Date,
                                     resolveLineage: Bool = true) throws -> [String: Any] {
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
        // LoRAs aren't one of the eight fields above (the ingestor reads them
        // separately, `AssetIngestor.embeddedLoras`, from the SAME sidecar shape) —
        // copy them through from the source's own adjacent sidecar when present, so
        // a derived edit's detail view still lists what generated the original.
        let sourceSidecarPath = ((sourcePath as NSString).deletingPathExtension) + ".json"
        if let data = FileManager.default.contents(atPath: sourceSidecarPath),
           let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let loras = j["loras"] {
            obj["loras"] = loras
        }
        let rootPath: String
        let rootAssetId: String?
        if resolveLineage {
            let root = EditSidecar.rootSource(forImageAt: sourcePath)
            rootPath = root.path
            rootAssetId = root.path == sourcePath ? source?.id : root.assetId
        } else {
            rootPath = sourcePath
            rootAssetId = source?.id
        }
        let sc = EditSidecar(version: EditRecipe.currentVersion, sourcePath: rootPath, sourceAssetId: rootAssetId,
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
            fm.fileExists(atPath: path)
                || fm.fileExists(atPath: EditSidecar.sidecarPath(forImageAt: path))
                || fm.fileExists(atPath: path + ".part")
        }
        var n = 1
        while taken(candidate(n)) { n += 1 }
        return candidate(n)
    }

    /// Reserves the next free `edit-<seconds>.png` slot by creating its `.part`
    /// temp name exclusively (`O_CREAT|O_EXCL`) so two concurrent exports can never
    /// claim the same output path — `outputPath`'s existence check alone leaves a
    /// window between "path looks free" and "file gets written". Reserving under
    /// `.part` rather than the final `.png` name (X1) also keeps the reservation
    /// itself invisible to `AssetIngestor`'s poller, which only matches real image
    /// extensions — the poller can no longer see (and ingest) an empty or
    /// half-written file before `export` finishes.
    /// Uses the same numbering and same "png-or-json taken" rule as `outputPath`;
    /// a raced `EEXIST` on the `.part` name just advances to the next candidate.
    /// Gives up after 100 candidates rather than looping forever.
    private static func reserveOutputPath(in directory: String, seconds: Int) throws -> (pngPath: String, partPath: String) {
        let base = (directory as NSString).appendingPathComponent("edit-\(seconds)")
        let fm = FileManager.default
        func candidate(_ n: Int) -> String { n == 1 ? base + ".png" : base + "-\(n).png" }

        var n = 1
        var attempts = 0
        while attempts < 100 {
            let pngPath = candidate(n)
            let partPath = pngPath + ".part"
            if fm.fileExists(atPath: pngPath) || fm.fileExists(atPath: EditSidecar.sidecarPath(forImageAt: pngPath)) {
                n += 1; attempts += 1; continue
            }
            let fd = open(partPath, O_CREAT | O_EXCL | O_WRONLY, 0o644)
            if fd >= 0 {
                close(fd)
                return (pngPath, partPath)
            }
            if errno == EEXIST {
                n += 1; attempts += 1; continue
            }
            throw EditExportError.writeFailed("could not reserve \(partPath): \(String(cString: strerror(errno)))")
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
                              outputDirectory: String, ingestor: AssetIngestor?,
                              resolveLineage: Bool = true) async throws -> String {
        try FileManager.default.createDirectory(atPath: outputDirectory, withIntermediateDirectories: true)

        // Reserve under the `.part` temp name before doing any rendering work, so
        // the reservation — not a post-render existence check — is what resolves a
        // collision between two concurrent exports. `.part` (not the final `.png`
        // name) keeps this invisible to `AssetIngestor`'s poller (X1): it only
        // matches real image extensions, so it can no longer see, and ingest, an
        // empty or half-written file while the render/sidecar write are in flight.
        let (pngPath, partPath) = try reserveOutputPath(in: outputDirectory, seconds: Int(Date().timeIntervalSince1970))

        func cleanupReservedPart(appendingTo message: String) -> String {
            do {
                try Self.cleanupReserved(partPath)
                return message
            } catch {
                return message + "; failed to remove \(partPath): \(error.localizedDescription)"
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
                try Self.cleanupReserved(partPath)
            } catch {
                throw EditExportError.writeFailed("render failed and could not remove reserved \(partPath): \(error)")
            }
            throw EditExportError.renderFailed
        }

        guard let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: partPath) as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw EditExportError.writeFailed(cleanupReservedPart(appendingTo: "could not create \(partPath)"))
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw EditExportError.writeFailed(cleanupReservedPart(appendingTo: "could not finalize \(partPath)"))
        }

        // The sidecar is named after the FINAL `.png` path (not `.part`) — it must
        // already exist under its real name at the moment the rename below makes
        // the pixels visible, so the poller (or anyone else) that discovers the
        // `.png` can always find its `.json` immediately alongside it.
        let sidecarPath = EditSidecar.sidecarPath(forImageAt: pngPath)
        do {
            let sidecar = try sidecarObject(source: sourceAsset, sourcePath: sourcePath, recipe: recipe, now: Date(), resolveLineage: resolveLineage)
            let data = try JSONSerialization.data(withJSONObject: sidecar, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: sidecarPath), options: .atomic)
        } catch {
            var message = "sidecar: \(error.localizedDescription)"
            message = cleanupReservedPart(appendingTo: message)
            if FileManager.default.fileExists(atPath: sidecarPath) {
                do {
                    try FileManager.default.removeItem(atPath: sidecarPath)
                } catch let cleanupError {
                    message += "; failed to remove \(sidecarPath): \(cleanupError.localizedDescription)"
                }
            }
            throw EditExportError.writeFailed(message)
        }

        // Publish atomically: `.part` already holds the finished PNG and the
        // sidecar is already on disk under its real name, so `rename(2)` is the
        // single instant at which anything watching `outputDirectory` (the
        // ingest poller, a `Finder` window) can first see the image — and by
        // then both files it needs are already there.
        guard rename(partPath, pngPath) == 0 else {
            var message = "could not publish \(pngPath): \(String(cString: strerror(errno)))"
            message = cleanupReservedPart(appendingTo: message)
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
