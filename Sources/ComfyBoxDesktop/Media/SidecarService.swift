// SidecarService.swift — Embedded, Finder-readable image metadata (exiftool)
//
// Writes generation metadata into STANDARD EXIF/XMP/IPTC fields so macOS Finder
// (Get Info → More Info) and Spotlight can read and search it — not just a custom
// chunk. The full parameter set is also stored as JSON in UserComment for exact
// round-trip. Reads the same fields back. Desktop-only; shells out to exiftool.

import Foundation

@Observable
@MainActor
public final class SidecarService {
    public init() {}

    public var exiftoolPath: String? { MediaToolsService.toolPath("exiftool") }
    public var isAvailable: Bool { exiftoolPath != nil }

    /// Generation metadata to embed. `description` becomes the human-facing
    /// Finder "Description"; `keywords` become Finder "Keywords".
    public struct Metadata: Sendable {
        public var description: String            // prompt / caption
        public var keywords: [String]             // tags + character + content mode
        public var parametersJSON: String?        // full params for exact round-trip
        public var software: String = "CoffeeShop Desktop (ComfyBox)"
        public init(description: String, keywords: [String] = [], parametersJSON: String? = nil) {
            self.description = description; self.keywords = keywords; self.parametersJSON = parametersJSON
        }
    }

    // MARK: - Pure arg builders (tested)

    /// exiftool write args mapping metadata → standard Finder/Spotlight fields.
    public nonisolated static func embedArgs(_ m: Metadata, path: String) -> [String] {
        var args = ["-overwrite_original", "-P"]
        // Description → EXIF ImageDescription + XMP dc:description + IPTC Caption
        // (Finder "Description"/"More Info"; Spotlight kMDItemDescription).
        args.append("-EXIF:ImageDescription=\(m.description)")
        args.append("-XMP-dc:Description=\(m.description)")
        args.append("-IPTC:Caption-Abstract=\(m.description)")
        args.append("-EXIF:Software=\(m.software)")
        // Keywords → IPTC:Keywords + XMP dc:subject (Finder "Keywords"; Spotlight).
        for kw in m.keywords {
            args.append("-IPTC:Keywords=\(kw)")
            args.append("-XMP-dc:Subject=\(kw)")
        }
        // Full parameters JSON → UserComment for exact round-trip.
        if let json = m.parametersJSON {
            args.append("-EXIF:UserComment=\(json)")
        }
        args.append(path)
        return args
    }

    /// exiftool read args → JSON for the fields we embed.
    public nonisolated static func readArgs(path: String) -> [String] {
        ["-j", "-EXIF:ImageDescription", "-XMP-dc:Description", "-IPTC:Caption-Abstract",
         "-IPTC:Keywords", "-XMP-dc:Subject", "-EXIF:UserComment", "-EXIF:Software", path]
    }

    /// Build keyword list from asset facets (deduped, lowercased, non-empty).
    public nonisolated static func keywords(tags: [String], character: String?, contentMode: String?) -> [String] {
        var out = tags
        if let c = character, !c.isEmpty { out.append(c) }
        if let m = contentMode, !m.isEmpty { out.append(m) }
        var seen = Set<String>()
        return out.map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
            .filter { !$0.isEmpty && seen.insert($0).inserted }
    }

    // MARK: - Execution

    public enum SidecarError: Error, LocalizedError {
        case notAvailable, failed(Int32, String)
        public var errorDescription: String? {
            switch self {
            case .notAvailable: return "exiftool not found (install with: brew install exiftool)."
            case .failed(let c, let out): return "exiftool failed (\(c)): \(out.split(separator: "\n").last.map(String.init) ?? "")"
            }
        }
    }

    /// Embed `metadata` into the image at `path` (Finder-readable). Returns nothing.
    public func embed(_ metadata: Metadata, into path: String) async throws {
        guard let tool = exiftoolPath else { throw SidecarError.notAvailable }
        try await run(tool, Self.embedArgs(metadata, path: path))
        // exiftool -P preserves the mod time, so Spotlight won't auto-reindex;
        // force it so Finder Get Info / search reflect the new metadata now.
        if let mdimport = MediaToolsService.toolPath("mdimport") ?? {
            FileManager.default.isExecutableFile(atPath: "/usr/bin/mdimport") ? "/usr/bin/mdimport" : nil
        }() {
            _ = try? await spawn(mdimport, [path])
        }
    }

    /// Read the embedded description + keywords back (best-effort).
    public func read(from path: String) async -> (description: String?, keywords: [String]) {
        guard let tool = exiftoolPath else { return (nil, []) }
        guard let out = try? await capture(tool, Self.readArgs(path: path)),
              let data = out.data(using: .utf8),
              let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]],
              let obj = arr.first else { return (nil, []) }
        let desc = (obj["Description"] ?? obj["ImageDescription"] ?? obj["Caption-Abstract"]) as? String
        var kws: [String] = []
        if let k = obj["Keywords"] as? [String] { kws = k }
        else if let k = obj["Keywords"] as? String { kws = [k] }
        else if let s = obj["Subject"] as? [String] { kws = s }
        return (desc, kws)
    }

    @discardableResult
    private func run(_ tool: String, _ args: [String]) async throws -> String {
        let (code, out) = try await spawn(tool, args)
        guard code == 0 else { throw SidecarError.failed(code, out) }
        return out
    }
    private func capture(_ tool: String, _ args: [String]) async throws -> String {
        try await spawn(tool, args).1
    }
    private func spawn(_ tool: String, _ args: [String]) async throws -> (Int32, String) {
        try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .utility).async {
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: tool)
                proc.arguments = args
                let pipe = Pipe(); proc.standardOutput = pipe; proc.standardError = pipe
                do {
                    try proc.run()
                    let data = pipe.fileHandleForReading.readDataToEndOfFile()
                    proc.waitUntilExit()
                    cont.resume(returning: (proc.terminationStatus, String(data: data, encoding: .utf8) ?? ""))
                } catch { cont.resume(throwing: error) }
            }
        }
    }
}
