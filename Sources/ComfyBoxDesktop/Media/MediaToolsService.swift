// MediaToolsService.swift — ImageMagick + ffmpeg backed export & transforms
//
// Desktop-only. Shells out to `magick` (ImageMagick) for format conversion and
// image filters, and `ffmpeg` for image-sequence → video export. Tool paths are
// discovered from common Homebrew locations. All argument building is pure and
// unit-tested; execution streams a console log like the other shell-out services.

import Foundation

@Observable
@MainActor
public final class MediaToolsService {
    public private(set) var isRunning = false
    public private(set) var consoleLog = ""

    public init() {}

    // MARK: - Tool discovery

    public nonisolated static func toolPath(_ name: String) -> String? {
        let candidates = ["/opt/homebrew/bin/", "/usr/local/bin/", "/usr/bin/"]
        for dir in candidates {
            let p = dir + name
            if FileManager.default.isExecutableFile(atPath: p) { return p }
        }
        return nil
    }
    public var magickPath: String? { Self.toolPath("magick") ?? Self.toolPath("convert") }
    public var ffmpegPath: String? { Self.toolPath("ffmpeg") }
    public var hasMagick: Bool { magickPath != nil }
    public var hasFFmpeg: Bool { ffmpegPath != nil }

    // MARK: - Export formats

    public enum ImageFormat: String, CaseIterable, Sendable {
        case jpeg, png, webp, tiff, heic
        public var ext: String { self == .jpeg ? "jpg" : rawValue }
        public var display: String { self == .jpeg ? "JPEG" : rawValue.uppercased() }
    }

    /// A named filter/transform applied via ImageMagick.
    public enum Transform: String, CaseIterable, Sendable {
        case none, grayscale, sepia, sharpen, blur, autoLevel, oilPaint, negate
        public var display: String {
            switch self {
            case .none: return "None"
            case .autoLevel: return "Auto Level"
            case .oilPaint: return "Oil Paint"
            default: return rawValue.capitalized
            }
        }
        var magickArgs: [String] {
            switch self {
            case .none: return []
            case .grayscale: return ["-colorspace", "Gray"]
            case .sepia: return ["-sepia-tone", "80%"]
            case .sharpen: return ["-sharpen", "0x1.5"]
            case .blur: return ["-blur", "0x3"]
            case .autoLevel: return ["-auto-level"]
            case .oilPaint: return ["-paint", "3"]
            case .negate: return ["-negate"]
            }
        }
    }

    // MARK: - Pure argument builders (tested)

    /// magick <source> [transform args] [-quality N] <output>
    public nonisolated static func convertArgs(source: String, output: String,
                                               transform: Transform = .none,
                                               quality: Int? = nil) -> [String] {
        var args = [source]
        args += transform.magickArgs
        if let quality { args += ["-quality", String(quality)] }
        args.append(output)
        return args
    }

    /// ffmpeg -y -framerate F -i concat-list -c:v libx264 -pix_fmt yuv420p <output>
    /// Uses a concat demuxer list so arbitrary (non-sequential) filenames work.
    public nonisolated static func videoArgs(listPath: String, fps: Int, output: String) -> [String] {
        ["-y", "-r", String(fps), "-f", "concat", "-safe", "0", "-i", listPath,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", output]
    }

    /// Build an ffmpeg concat-demuxer list file body for the given images at `fps`.
    public nonisolated static func concatListBody(images: [String], fps: Int) -> String {
        let dur = 1.0 / Double(max(1, fps))
        var lines: [String] = []
        for img in images {
            lines.append("file '\(img.replacingOccurrences(of: "'", with: "'\\''"))'")
            lines.append("duration \(dur)")
        }
        if let last = images.last {  // concat demuxer needs the last frame repeated
            lines.append("file '\(last.replacingOccurrences(of: "'", with: "'\\''"))'")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    // MARK: - Execution

    public enum MediaError: Error, LocalizedError {
        case toolMissing(String), failed(Int32, String), noImages
        public var errorDescription: String? {
            switch self {
            case .toolMissing(let t): return "\(t) not found (install with: brew install \(t == "ffmpeg" ? "ffmpeg" : "imagemagick"))."
            case .noImages: return "No images to export."
            case .failed(let c, let out): return "Media tool failed (\(c)): \(out.split(separator: "\n").last.map(String.init) ?? "")"
            }
        }
    }

    /// Convert/transform one image; returns the output path.
    @discardableResult
    public func export(source: String, to format: ImageFormat,
                       transform: Transform = .none, outputDir: String? = nil) async throws -> String {
        guard let magick = magickPath else { throw MediaError.toolMissing("imagemagick") }
        let dir = outputDir ?? (source as NSString).deletingLastPathComponent
        let base = ((source as NSString).lastPathComponent as NSString).deletingPathExtension
        let suffix = transform == .none ? "" : "-\(transform.rawValue)"
        let output = (dir as NSString).appendingPathComponent("\(base)\(suffix).\(format.ext)")
        let quality: Int? = (format == .jpeg || format == .webp) ? 92 : nil
        try await run(magick, Self.convertArgs(source: source, output: output, transform: transform, quality: quality))
        return output
    }

    /// Export an ordered list of images to an .mp4 at `fps`; returns the output path.
    @discardableResult
    public func exportVideo(images: [String], fps: Int, output: String) async throws -> String {
        guard let ffmpeg = ffmpegPath else { throw MediaError.toolMissing("ffmpeg") }
        guard !images.isEmpty else { throw MediaError.noImages }
        let listPath = NSTemporaryDirectory() + "comfybox-seq-\(UUID().uuidString).txt"
        try Self.concatListBody(images: images, fps: fps).write(toFile: listPath, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: listPath) }
        try await run(ffmpeg, Self.videoArgs(listPath: listPath, fps: fps, output: output))
        return output
    }

    private func run(_ tool: String, _ args: [String]) async throws {
        guard !isRunning else { throw MediaError.failed(-1, "A media operation is already running.") }
        isRunning = true; consoleLog = ""
        defer { isRunning = false }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: tool)
        proc.arguments = args
        let pipe = Pipe()
        proc.standardOutput = pipe; proc.standardError = pipe
        append("$ \((tool as NSString).lastPathComponent) \(args.joined(separator: " "))\n")
        let handle = pipe.fileHandleForReading
        let stream = AsyncStream<String> { cont in
            handle.readabilityHandler = { fh in
                let d = fh.availableData
                if d.isEmpty { cont.finish() } else if let t = String(data: d, encoding: .utf8) { cont.yield(t) }
            }
        }
        try proc.run()
        for await chunk in stream { append(chunk) }
        proc.waitUntilExit()
        handle.readabilityHandler = nil
        guard proc.terminationStatus == 0 else { throw MediaError.failed(proc.terminationStatus, consoleLog) }
    }

    private func append(_ s: String) {
        consoleLog += s
        if consoleLog.count > 60_000 { consoleLog = String(consoleLog.suffix(50_000)) }
    }
}
