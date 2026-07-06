// MfluxService.swift — Desktop-only mflux (Python/MLX) backend
//
// ComfyBox's server is fully self-standing Swift/MLX and never depends on
// mflux; API/MCP clients don't see it. This service is ONLY for the desktop
// UI: it shells out to the local mflux virtualenv to offer generation,
// training, model saving/quantizing, and updates. Argument building is pure
// and unit-tested; process execution streams output live.

import Foundation

/// mflux base-model variants (the --base-model / --model choices).
public enum MfluxModel: String, CaseIterable, Identifiable, Sendable {
    case dev, schnell
    case kreaDev = "krea-dev"
    case devKrea = "dev-krea"
    case qwen, fibo
    case zImage = "z-image"
    case zImageTurbo = "z-image-turbo"
    case flux2Klein4b = "flux2-klein-4b"
    case flux2Klein9b = "flux2-klein-9b"
    public var id: String { rawValue }
    public var label: String { rawValue }
}

@Observable
@MainActor
public final class MfluxService {
    /// Directory holding the mflux console scripts (…/.venv/bin).
    public var binDirectory: String

    /// Whether a command is currently running (one at a time).
    public private(set) var isRunning = false
    /// Live console output (stdout+stderr) of the current/last command.
    public private(set) var consoleLog: String = ""
    /// Resolved mflux version string, if known.
    public private(set) var version: String?

    private var process: Process?

    public init(binDirectory: String = MfluxService.defaultBinDirectory) {
        self.binDirectory = binDirectory
    }

    public nonisolated static var defaultBinDirectory: String {
        NSString(string: "~/Projects/mflux/.venv/bin").expandingTildeInPath
    }

    /// Whether the mflux venv appears installed (mflux-generate present).
    public var isInstalled: Bool {
        FileManager.default.isExecutableFile(atPath: executablePath("mflux-generate"))
    }

    public func executablePath(_ command: String) -> String {
        (binDirectory as NSString).appendingPathComponent(command)
    }

    // MARK: - Pure argument builders (tested)

    public struct GenerateOptions: Sendable {
        public var model: String            // variant rawValue, HF repo, or local path
        public var prompt: String
        public var negativePrompt: String?
        public var width: Int
        public var height: Int
        public var steps: Int
        public var guidance: Double?
        public var seed: UInt64?            // nil = random
        public var quantize: Int?           // 3/4/5/6/8
        public var loraPaths: [String]
        public var loraScales: [Double]
        public var imagePath: String?
        public var imageStrength: Double?
        /// Built-in LoRA style (e.g. "identity" for face-identity generation).
        public var loraStyle: String?
        public var lowRam: Bool
        public var output: String

        public init(model: String, prompt: String, negativePrompt: String? = nil,
                    width: Int = 1024, height: Int = 1024, steps: Int = 25,
                    guidance: Double? = nil, seed: UInt64? = nil, quantize: Int? = nil,
                    loraPaths: [String] = [], loraScales: [Double] = [],
                    imagePath: String? = nil, imageStrength: Double? = nil,
                    loraStyle: String? = nil, lowRam: Bool = false, output: String) {
            self.model = model; self.prompt = prompt; self.negativePrompt = negativePrompt
            self.width = width; self.height = height; self.steps = steps
            self.guidance = guidance; self.seed = seed; self.quantize = quantize
            self.loraPaths = loraPaths; self.loraScales = loraScales
            self.imagePath = imagePath; self.imageStrength = imageStrength
            self.loraStyle = loraStyle; self.lowRam = lowRam; self.output = output
        }
    }

    public nonisolated static func generateArgs(_ o: GenerateOptions) -> [String] {
        var args = [
            "--model", o.model,
            "--prompt", o.prompt,
            "--width", String(o.width),
            "--height", String(o.height),
            "--steps", String(o.steps),
            "--output", o.output,
        ]
        if let n = o.negativePrompt, !n.isEmpty { args += ["--negative-prompt", n] }
        if let g = o.guidance { args += ["--guidance", String(g)] }
        if let s = o.seed { args += ["--seed", String(s)] }
        if let q = o.quantize { args += ["--quantize", String(q)] }
        if !o.loraPaths.isEmpty {
            args.append("--lora-paths"); args += o.loraPaths
            if !o.loraScales.isEmpty {
                args.append("--lora-scales"); args += o.loraScales.map { String($0) }
            }
        }
        if let style = o.loraStyle, !style.isEmpty { args += ["--lora-style", style] }
        if let img = o.imagePath, !img.isEmpty {
            args += ["--image-path", img]
            if let st = o.imageStrength { args += ["--image-strength", String(st)] }
        }
        if o.lowRam { args.append("--low-ram") }
        return args
    }

    public struct TrainOptions: Sendable {
        public var model: String?
        public var quantize: Int?
        public var configPath: String?     // --config
        public var resumePath: String?     // --resume
        public var dryRun: Bool
        public var lowRam: Bool
        public init(model: String? = nil, quantize: Int? = nil, configPath: String? = nil,
                    resumePath: String? = nil, dryRun: Bool = false, lowRam: Bool = false) {
            self.model = model; self.quantize = quantize; self.configPath = configPath
            self.resumePath = resumePath; self.dryRun = dryRun; self.lowRam = lowRam
        }
    }

    public nonisolated static func trainArgs(_ o: TrainOptions) -> [String] {
        var args: [String] = []
        if let m = o.model, !m.isEmpty { args += ["--model", m] }
        if let q = o.quantize { args += ["--quantize", String(q)] }
        if let resume = o.resumePath, !resume.isEmpty {
            args += ["--resume", resume]
        } else if let config = o.configPath, !config.isEmpty {
            args += ["--config", config]
        }
        if o.dryRun { args.append("--dry-run") }
        if o.lowRam { args.append("--low-ram") }
        return args
    }

    public struct SaveOptions: Sendable {
        public var model: String
        public var path: String            // --path (output dir)
        public var baseModel: String?
        public var quantize: Int?
        public var loraPaths: [String]
        public var loraScales: [Double]
        public init(model: String, path: String, baseModel: String? = nil, quantize: Int? = nil,
                    loraPaths: [String] = [], loraScales: [Double] = []) {
            self.model = model; self.path = path; self.baseModel = baseModel
            self.quantize = quantize; self.loraPaths = loraPaths; self.loraScales = loraScales
        }
    }

    public nonisolated static func saveArgs(_ o: SaveOptions) -> [String] {
        var args = ["--model", o.model, "--path", o.path]
        if let b = o.baseModel, !b.isEmpty { args += ["--base-model", b] }
        if let q = o.quantize { args += ["--quantize", String(q)] }
        if !o.loraPaths.isEmpty {
            args.append("--lora-paths"); args += o.loraPaths
            if !o.loraScales.isEmpty {
                args.append("--lora-scales"); args += o.loraScales.map { String($0) }
            }
        }
        return args
    }

    /// Parse a `pip show mflux` blob for the version.
    public nonisolated static func parseVersion(fromPipShow text: String) -> String? {
        for line in text.split(separator: "\n") {
            if line.lowercased().hasPrefix("version:") {
                return line.split(separator: ":", maxSplits: 1).last.map {
                    $0.trimmingCharacters(in: .whitespaces)
                }
            }
        }
        return nil
    }

    // MARK: - Process execution

    public enum MfluxError: Error, LocalizedError {
        case notInstalled(String)
        case alreadyRunning
        case commandFailed(Int32)

        public var errorDescription: String? {
            switch self {
            case .notInstalled(let dir): return "mflux venv not found at \(dir). Set the path in Settings → Motion/mflux."
            case .alreadyRunning: return "An mflux command is already running."
            case .commandFailed(let code): return "mflux exited with code \(code)."
            }
        }
    }

    /// Run an mflux console command, streaming combined output into `consoleLog`.
    /// Returns the process exit code (0 = success).
    @discardableResult
    public func run(_ command: String, args: [String], clearLog: Bool = true) async throws -> Int32 {
        guard !isRunning else { throw MfluxError.alreadyRunning }
        let exe = executablePath(command)
        guard FileManager.default.isExecutableFile(atPath: exe) else {
            throw MfluxError.notInstalled(binDirectory)
        }
        if clearLog { consoleLog = "" }
        isRunning = true
        defer { isRunning = false; process = nil }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: exe)
        proc.arguments = args
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        self.process = proc

        appendLog("$ \(command) \(args.joined(separator: " "))\n")

        let handle = pipe.fileHandleForReading
        let stream = AsyncStream<String> { continuation in
            handle.readabilityHandler = { fh in
                let data = fh.availableData
                if data.isEmpty {
                    continuation.finish()
                } else if let text = String(data: data, encoding: .utf8) {
                    continuation.yield(text)
                }
            }
        }

        try proc.run()
        for await chunk in stream {
            appendLog(chunk)
        }
        proc.waitUntilExit()
        handle.readabilityHandler = nil
        let code = proc.terminationStatus
        if code != 0 { throw MfluxError.commandFailed(code) }
        return code
    }

    /// Cancel a running command.
    public func cancel() {
        process?.terminate()
    }

    /// Refresh `version` by running `pip show mflux` in the venv.
    public func refreshVersion() async {
        let pip = (binDirectory as NSString).appendingPathComponent("pip")
        guard FileManager.default.isExecutableFile(atPath: pip) else { return }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pip)
        proc.arguments = ["show", "mflux"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        do {
            try proc.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            proc.waitUntilExit()
            if let text = String(data: data, encoding: .utf8) {
                version = Self.parseVersion(fromPipShow: text)
            }
        } catch { /* leave version nil */ }
    }

    /// Update mflux in its venv (pip install -U mflux), streaming output.
    public func update() async throws {
        let pip = (binDirectory as NSString).appendingPathComponent("pip")
        guard FileManager.default.isExecutableFile(atPath: pip) else {
            throw MfluxError.notInstalled(binDirectory)
        }
        guard !isRunning else { throw MfluxError.alreadyRunning }
        isRunning = true
        defer { isRunning = false }
        consoleLog = ""
        appendLog("$ pip install -U mflux\n")

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pip)
        proc.arguments = ["install", "-U", "mflux"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        try proc.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        if let text = String(data: data, encoding: .utf8) { appendLog(text) }
        await refreshVersion()
    }

    private func appendLog(_ text: String) {
        consoleLog += text
        // Cap the log so long training runs don't grow unbounded.
        if consoleLog.count > 200_000 {
            consoleLog = String(consoleLog.suffix(150_000))
        }
    }
}
