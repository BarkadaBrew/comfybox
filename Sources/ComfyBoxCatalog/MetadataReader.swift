// MetadataReader.swift — pull generation facts out of the artifacts on disk.
//
// Three sources, in descending durability:
//   1. Embedded EXIF/XMP (IMAGES ONLY) — rides the file everywhere, including
//      the copy to the Linux server. `EXIF:UserComment` holds the full JSON
//      generation record; ImageDescription/XMP hold the prompt.
//   2. JSON sidecar (images AND video) — server-side mirror tree. For VIDEO
//      this is the ONLY source: an .mp4 carries no generation metadata at all.
//   3. Container probe — duration/fps/frames, because sidecars sometimes say
//      "duration": null and must not be trusted for it.
//
// Every read is best-effort: a missing tool, a missing file or garbage JSON
// yields fewer facts, never a throw.

import Foundation

public struct FileMetadata: Sendable, Equatable {
    public var prompt: String?
    public var negativePrompt: String?
    public var promptRaw: String?
    public var seed: Int?
    public var steps: Int?
    public var guidance: Double?
    public var width: Int?
    public var height: Int?
    public var modelFamily: String?
    public var preset: String?
    public var loras: String?
    public var characterName: String?
    public var contentMode: String?
    public var lane: String?
    public var mode: String?
    public var resolution: String?
    public var aspectRatio: String?
    public var durationMs: Int?
    public var sealed: Bool = false
    public var sourceImagePath: String?
    public var software: String?

    public init() {}
}

public enum MetadataReader {

    public static let exiftoolPath = "/opt/homebrew/bin/exiftool"

    // MARK: - Embedded (images)

    /// Read embedded EXIF/XMP via exiftool. Returns nil when exiftool is absent
    /// or the file has nothing — never throws.
    public static func readEmbedded(path: String) -> FileMetadata? {
        guard FileManager.default.isExecutableFile(atPath: exiftoolPath) else { return nil }
        let args = ["-j", "-n",
                    "-EXIF:UserComment", "-EXIF:ImageDescription", "-EXIF:Software",
                    "-XMP-dc:Description", "-IPTC:Keywords", "-XMP-dc:Subject",
                    path]
        guard let out = runTool(exiftoolPath, args),
              let arr = try? JSONSerialization.jsonObject(with: out) as? [[String: Any]],
              let obj = arr.first else { return nil }

        var m = FileMetadata()
        if let uc = obj["UserComment"] as? String {
            m = parseUserComment(uc)
        }
        if m.prompt == nil {
            m.prompt = (obj["ImageDescription"] as? String) ?? (obj["Description"] as? String)
        }
        m.software = obj["Software"] as? String
        return m
    }

    /// Parse the engine's `EXIF:UserComment` JSON blob.
    public static func parseUserComment(_ raw: String) -> FileMetadata {
        var m = FileMetadata()
        guard let data = raw.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return m }
        m.prompt = obj["prompt"] as? String
        m.negativePrompt = obj["negative_prompt"] as? String
        m.seed = intValue(obj["seed"])
        m.steps = intValue(obj["steps"])
        m.guidance = doubleValue(obj["guidance"])
        m.width = intValue(obj["width"])
        m.height = intValue(obj["height"])
        m.modelFamily = obj["model"] as? String
        if let loras = obj["loras"], let d = try? JSONSerialization.data(withJSONObject: loras) {
            m.loras = String(data: d, encoding: .utf8)
        }
        return m
    }

    // MARK: - Sidecar (images and video)

    public static func readSidecar(jsonData: Data) -> FileMetadata? {
        guard !jsonData.isEmpty,
              let obj = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
        else { return nil }

        var m = FileMetadata()
        m.prompt = obj["prompt"] as? String
        m.promptRaw = obj["prompt_raw"] as? String
        m.negativePrompt = obj["negative_prompt"] as? String
        m.seed = intValue(obj["seed"])
        m.steps = intValue(obj["steps"])
        m.guidance = doubleValue(obj["guidance"])
        m.width = intValue(obj["width"])
        m.height = intValue(obj["height"])
        m.modelFamily = obj["model"] as? String
        m.preset = obj["preset"] as? String
        m.characterName = obj["character"] as? String
        m.contentMode = obj["content_mode"] as? String
        m.mode = obj["mode"] as? String
        m.resolution = obj["resolution"] as? String
        m.aspectRatio = obj["aspect_ratio"] as? String
        m.sourceImagePath = obj["source_image"] as? String
        m.sealed = (obj["sealed"] as? Bool) ?? false
        if let loras = obj["loras"], let d = try? JSONSerialization.data(withJSONObject: loras) {
            m.loras = String(data: d, encoding: .utf8)
        }
        // `duration` is deliberately NOT read: it is null in real sidecars.
        // Duration comes from probeContainer.
        return m
    }

    /// Map a media path in the gallery tree to its sidecar in the mirror tree.
    public static func sidecarPath(forMedia media: String,
                                   galleryRoot: String,
                                   metadataRoot: String) -> String? {
        guard media.hasPrefix(galleryRoot) else { return nil }
        let rel = String(media.dropFirst(galleryRoot.count))
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let base = (rel as NSString).deletingPathExtension
        return (metadataRoot as NSString).appendingPathComponent(base + ".json")
    }

    // MARK: - Container probe (video)

    public struct ContainerInfo: Sendable, Equatable {
        public let durationMs: Int?
        public let fps: Double?
        public let frames: Int?
        public let width: Int?
        public let height: Int?
    }

    /// Probe duration/fps/frames from the container itself. Uses exiftool so we
    /// take no new dependency; ffprobe is not assumed to be installed.
    public static func probeContainer(path: String) -> ContainerInfo? {
        guard FileManager.default.isExecutableFile(atPath: exiftoolPath) else { return nil }
        let args = ["-j", "-n", "-QuickTime:Duration", "-VideoFrameRate",
                    "-ImageWidth", "-ImageHeight", path]
        guard let out = runTool(exiftoolPath, args),
              let arr = try? JSONSerialization.jsonObject(with: out) as? [[String: Any]],
              let obj = arr.first else { return nil }

        let seconds = doubleValue(obj["Duration"])
        let fps = doubleValue(obj["VideoFrameRate"])
        let ms = seconds.map { Int(($0 * 1000).rounded()) }
        let frames: Int? = {
            guard let s = seconds, let f = fps, f > 0 else { return nil }
            return Int((s * f).rounded())
        }()
        return ContainerInfo(durationMs: ms, fps: fps, frames: frames,
                             width: intValue(obj["ImageWidth"]),
                             height: intValue(obj["ImageHeight"]))
    }

    // MARK: - Helpers

    private static func runTool(_ launchPath: String, _ args: [String]) -> Data? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: launchPath)
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice
        do { try p.run() } catch { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return data.isEmpty ? nil : data
    }

    private static func intValue(_ any: Any?) -> Int? {
        if let i = any as? Int { return i }
        if let d = any as? Double { return Int(d) }
        if let s = any as? String { return Int(s) }
        return nil
    }

    private static func doubleValue(_ any: Any?) -> Double? {
        if let d = any as? Double { return d }
        if let i = any as? Int { return Double(i) }
        if let s = any as? String { return Double(s) }
        return nil
    }
}
