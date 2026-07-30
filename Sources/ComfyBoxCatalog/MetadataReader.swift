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

import Dispatch
import Foundation

/// Everything the readers can recover about one file, from any source.
///
/// THE FIELD LIST LIVES IN EXACTLY TWO PLACES: the properties below, and
/// `CatalogBackfill.row(file:existing:meta:)`, which is the single place any of
/// this becomes a `CatalogAsset`. There is no merge function with a third copy
/// of the list — precedence is applied by folding `row` over the sources in
/// order. Two is the floor without reflection: the properties ARE the type, and
/// the two structs are genuinely different shapes. Swift cannot check the
/// remaining hop, so `CatalogBackfillTests.testEveryFileMetadataFieldIsMapped`
/// fails loudly when a field is added here and nowhere else. `lane` was silently
/// dropped for exactly as long as there were four hand-maintained lists.
public struct FileMetadata: Sendable, Equatable {
    public var prompt: String?
    public var negativePrompt: String?
    public var promptRaw: String?
    /// The prompt after character / trigger injection. A third real spelling in
    /// the sidecars (349/400 images carry it), indexed for search alongside the
    /// other two.
    public var promptInjected: String?
    public var seed: Int?
    public var steps: Int?
    public var guidance: Double?
    public var width: Int?
    public var height: Int?
    public var modelFamily: String?
    public var preset: String?
    public var loras: String?
    public var renderID: String?
    public var characterName: String?
    public var contentMode: String?
    public var lane: String?
    public var arc: String?
    public var theme: String?
    public var stock: String?
    public var genre: String?
    public var family: String?
    public var style: String?
    public var mode: String?
    public var resolution: String?
    public var aspectRatio: String?
    public var durationMs: Int?
    public var fps: Double?
    public var frames: Int?
    public var sealed: Bool = false
    public var sourceImagePath: String?
    public var software: String?
    /// The sidecar's `provider` field ("which application produced this").
    /// Images recover this from `EXIF:Software`; video has no embedded
    /// metadata at all, so for video this is the ONLY source of `source`.
    public var provider: String?

    public init() {}
}

public enum MetadataReader {

    public static let exiftoolPath = "/opt/homebrew/bin/exiftool"

    // MARK: - Embedded (images)

    /// Read embedded EXIF/XMP via exiftool. Returns nil when exiftool is absent
    /// or the file has nothing — never throws.
    public static func readEmbedded(path: String) -> FileMetadata? {
        guard FileManager.default.isExecutableFile(atPath: exiftoolPath) else { return nil }
        // Note: IPTC:Keywords / XMP-dc:Subject are deliberately NOT requested —
        // FileMetadata has no field for them and nothing reads them; asking
        // exiftool to extract tags we then discard is dead work. Add both the
        // field and the flag together if a future task needs keyword tags.
        let args = ["-j", "-n",
                    "-EXIF:UserComment", "-EXIF:ImageDescription", "-EXIF:Software",
                    "-XMP-dc:Description",
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
        // Three real spellings. IMAGE sidecars have no `prompt` key at all —
        // they carry prompt_optimized / prompt_raw / prompt_injected — while
        // VIDEO sidecars do use `prompt`. Reading only `prompt` lost the prompt
        // for every image in the fleet.
        m.prompt = (obj["prompt"] as? String) ?? (obj["prompt_optimized"] as? String)
        m.promptRaw = obj["prompt_raw"] as? String
        m.promptInjected = obj["prompt_injected"] as? String
        m.negativePrompt = obj["negative_prompt"] as? String
        m.seed = intValue(obj["seed"])
        m.steps = intValue(obj["steps"])
        m.guidance = doubleValue(obj["guidance"])
        m.width = intValue(obj["width"])
        m.height = intValue(obj["height"])
        m.modelFamily = obj["model"] as? String
        m.preset = obj["preset"] as? String
        m.characterName = obj["character"] as? String
        // Gated like every other tier value here. The sidecar is the STRONGEST
        // source, so a bad value would be the hardest to displace, and an
        // unrecognised content_mode ranks above every ceiling (tierRank fails
        // closed) — withholding the asset from everyone rather than from nobody.
        m.contentMode = fruitTier(obj["content_mode"])
        // `lane` is the whole derived-filing key (CollectionRules maps it to the
        // body of work), and a sidecar is the ONLY place it is ever written —
        // nothing embeds it. Omitting it here left every backfilled asset
        // permanently unfiled.
        m.lane = obj["lane"] as? String
        m.mode = obj["mode"] as? String
        m.resolution = obj["resolution"] as? String
        m.aspectRatio = obj["aspect_ratio"] as? String
        m.sourceImagePath = obj["source_image"] as? String
        m.provider = obj["provider"] as? String
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

    // MARK: - Journals (the third source)

    /// One render's worth of journal facts, keyed by the output path the
    /// journal recorded. That path is written on the SERVER, so the caller has
    /// to translate it to a local spelling before using it as a key.
    public struct JournalEntry: Sendable, Equatable {
        public let path: String
        public let meta: FileMetadata
    }

    /// `~/.kira/render-journal.jsonl` — one JSON object per line:
    /// `{ts, tier, lane, intent, theme, path, kind, seed{stock,style,genre,family}}`.
    ///
    /// This is the LOWEST-precedence source and, in practice, the only one that
    /// knows the lane for most assets: `lane` appears in only 131/400 image
    /// sidecars. A malformed line is skipped, never fatal — a journal is an
    /// append-only log and its last line is routinely half-written.
    public static func readRenderJournal(jsonlData: Data) -> [JournalEntry] {
        guard let text = String(data: jsonlData, encoding: .utf8) else { return [] }
        var out: [JournalEntry] = []
        for line in text.split(whereSeparator: { $0.isNewline }) {
            guard let data = line.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let path = obj["path"] as? String, !path.isEmpty else { continue }
            var m = FileMetadata()
            m.contentMode = fruitTier(obj["tier"])
            m.lane = obj["lane"] as? String
            m.arc = obj["arc"] as? String
            m.theme = obj["theme"] as? String
            if let seed = obj["seed"] as? [String: Any] {
                m.stock = seed["stock"] as? String
                m.style = seed["style"] as? String
                m.genre = seed["genre"] as? String
                m.family = seed["family"] as? String
            }
            out.append(JournalEntry(path: path, meta: m))
        }
        return out
    }

    /// `~/.kira/studio/history.json` and `~/.bree/studio/history.json` —
    /// `{version, records:[{id, prompt, character, contentMode, width, height,
    /// steps, seed, outputPath, durationMs, provider, createdAt}]}`.
    public static func readHistory(jsonData: Data) -> [JournalEntry] {
        guard let obj = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
              let records = obj["records"] as? [[String: Any]] else { return [] }
        var out: [JournalEntry] = []
        for r in records {
            guard let path = r["outputPath"] as? String, !path.isEmpty else { continue }
            var m = FileMetadata()
            m.prompt = r["prompt"] as? String
            m.characterName = r["character"] as? String
            // Same gate as the journal's `tier`, for the same reason: this is a
            // weak source, and one odd value withholds the asset from EVERY
            // ceiling rather than from none.
            m.contentMode = fruitTier(r["contentMode"])
            m.width = intValue(r["width"])
            m.height = intValue(r["height"])
            m.steps = intValue(r["steps"])
            m.seed = intValue(r["seed"])
            m.provider = r["provider"] as? String
            m.renderID = r["id"] as? String
            // `durationMs` here is how long the RENDER took (84944 on a still),
            // not how long the clip runs. Mapping it onto the catalog's video
            // duration would be a lie, so it is deliberately dropped — media
            // duration comes from probeContainer and nowhere else.
            out.append(JournalEntry(path: path, meta: m))
        }
        return out
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

    /// A content-mode value, but only if it is one the tier ladder recognises.
    ///
    /// The word "tier" is overloaded in this ecosystem: the render journal means
    /// the fruit tier, while real image sidecars use it for a QUALITY tier
    /// ("standard"). `tierRank` fails CLOSED, so an unrecognised content_mode
    /// ranks above every ceiling and would withhold the row from everyone —
    /// the opposite of what a weak, best-effort source should ever cause.
    static func fruitTier(_ any: Any?) -> String? {
        guard let raw = (any as? String)?.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else { return nil }
        guard CATALOG_TIER_ORDER.contains(raw) || CATALOG_TIER_ALIASES[raw] != nil else { return nil }
        return raw
    }

    /// Wall-clock ceiling for any external tool invocation. A later task runs
    /// these readers over several thousand files in sequence, so one hung
    /// process (corrupt file, stalled filesystem, network-mounted media) must
    /// not stall the whole run.
    static let toolTimeoutSeconds: TimeInterval = 20

    /// Run an external tool and capture its stdout, with a wall-clock timeout.
    /// Never throws and never blocks past `timeout`: on expiry the process is
    /// terminated and this returns nil, same as any other missing-data case.
    ///
    /// `timeout` defaults to `toolTimeoutSeconds` and is only overridden by
    /// tests (to exercise the timeout path without slowing the suite down).
    static func runTool(_ launchPath: String, _ args: [String],
                        timeout: TimeInterval = toolTimeoutSeconds) -> Data? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: launchPath)
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice

        do { try p.run() } catch {
            try? pipe.fileHandleForReading.close()
            return nil
        }

        // Safety valve: if the child never exits on its own, force it to so
        // the blocking read below cannot hang forever. Runs on a background
        // queue so it never competes with the read/wait for the same thread.
        let watchdog = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        watchdog.schedule(deadline: .now() + timeout)
        watchdog.setEventHandler { [weak p] in
            guard let p, p.isRunning else { return }
            p.terminate()
        }
        watchdog.resume()
        defer { watchdog.cancel() }

        // Terminating the child closes its stdout, which is what unblocks this
        // read if the watchdog had to fire; on the normal path the child
        // closes stdout on its own exit well before the deadline.
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        try? pipe.fileHandleForReading.close()
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
