// DecoupageService.swift — Desktop-only Découpage compositing backend
//
// Découpage (~/Projects/decoupage) is a Python CLI for multi-layer mixed-media
// art: a charcoal figure layer + découpage element libraries composited by
// recipe. Like mflux, this is a UI-only convenience — the ComfyBox server never
// depends on it. Argument building is pure and unit-tested.

import Foundation

@Observable
@MainActor
public final class DecoupageService {
    public var projectDirectory: String

    public private(set) var isRunning = false
    public private(set) var consoleLog = ""
    private var process: Process?

    public init(projectDirectory: String = DecoupageService.defaultProjectDirectory) {
        self.projectDirectory = projectDirectory
    }

    public nonisolated static var defaultProjectDirectory: String {
        NSString(string: "~/Projects/decoupage").expandingTildeInPath
    }

    public var executablePath: String {
        (projectDirectory as NSString).appendingPathComponent(".venv/bin/decoupage")
    }
    public var recipesDirectory: String {
        (projectDirectory as NSString).appendingPathComponent("recipes")
    }
    public var outputDirectory: String {
        (projectDirectory as NSString).appendingPathComponent("output")
    }
    public var isInstalled: Bool {
        FileManager.default.isExecutableFile(atPath: executablePath)
    }

    /// Recipe YAML files shipped with the project.
    public func recipes() -> [String] {
        (try? FileManager.default.contentsOfDirectory(atPath: recipesDirectory))?
            .filter { $0.hasSuffix(".yaml") || $0.hasSuffix(".yml") }
            .sorted() ?? []
    }

    public func recipePath(_ name: String) -> String {
        (recipesDirectory as NSString).appendingPathComponent(name)
    }

    // MARK: - Pure argument builders (tested)

    public struct GenerateOptions: Sendable {
        public var description: String
        public var recipe: String            // full path
        public var output: String?
        public var seed: Int?
        public var anchor: String?
        public var noAnchor: Bool
        public var printPrep: Bool
        public init(description: String, recipe: String, output: String? = nil, seed: Int? = nil,
                    anchor: String? = nil, noAnchor: Bool = false, printPrep: Bool = false) {
            self.description = description; self.recipe = recipe; self.output = output
            self.seed = seed; self.anchor = anchor; self.noAnchor = noAnchor; self.printPrep = printPrep
        }
    }

    public nonisolated static func generateArgs(_ o: GenerateOptions) -> [String] {
        var args = ["generate", o.description, "--recipe", o.recipe]
        if let out = o.output, !out.isEmpty { args += ["--output", out] }
        if let seed = o.seed { args += ["--seed", String(seed)] }
        if let anchor = o.anchor, !anchor.isEmpty { args += ["--anchor", anchor] }
        if o.noAnchor { args.append("--no-anchor") }
        if o.printPrep { args.append("--print-prep") }
        return args
    }

    public struct CompositeOptions: Sendable {
        public var figure: String            // existing figure image path
        public var recipe: String
        public var output: String?
        public var seed: Int?
        public var printPrep: Bool
        public init(figure: String, recipe: String, output: String? = nil, seed: Int? = nil, printPrep: Bool = false) {
            self.figure = figure; self.recipe = recipe; self.output = output; self.seed = seed; self.printPrep = printPrep
        }
    }

    public nonisolated static func compositeArgs(_ o: CompositeOptions) -> [String] {
        var args = ["composite", o.figure, "--recipe", o.recipe]
        if let out = o.output, !out.isEmpty { args += ["--output", out] }
        if let seed = o.seed { args += ["--seed", String(seed)] }
        if o.printPrep { args.append("--print-prep") }
        return args
    }

    public nonisolated static func genElementsArgs(recipe: String, category: String, prompt: String, count: Int) -> [String] {
        ["gen-elements", "--recipe", recipe, category, prompt, "--count", String(max(1, count))]
    }

    // MARK: - Execution

    public enum DecoupageError: Error, LocalizedError {
        case notInstalled(String), alreadyRunning, failed(Int32)
        public var errorDescription: String? {
            switch self {
            case .notInstalled(let d): return "Découpage venv not found at \(d)."
            case .alreadyRunning: return "A Découpage command is already running."
            case .failed(let c): return "Découpage exited with code \(c)."
            }
        }
    }

    @discardableResult
    public func run(_ args: [String]) async throws -> Int32 {
        guard !isRunning else { throw DecoupageError.alreadyRunning }
        guard FileManager.default.isExecutableFile(atPath: executablePath) else {
            throw DecoupageError.notInstalled(projectDirectory)
        }
        consoleLog = ""
        isRunning = true
        defer { isRunning = false; process = nil }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: executablePath)
        proc.arguments = args
        proc.currentDirectoryURL = URL(fileURLWithPath: projectDirectory)
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        process = proc
        append("$ decoupage \(args.joined(separator: " "))\n")

        let handle = pipe.fileHandleForReading
        let stream = AsyncStream<String> { continuation in
            handle.readabilityHandler = { fh in
                let data = fh.availableData
                if data.isEmpty { continuation.finish() }
                else if let t = String(data: data, encoding: .utf8) { continuation.yield(t) }
            }
        }
        try proc.run()
        for await chunk in stream { append(chunk) }
        proc.waitUntilExit()
        handle.readabilityHandler = nil
        if proc.terminationStatus != 0 { throw DecoupageError.failed(proc.terminationStatus) }
        return proc.terminationStatus
    }

    public func cancel() { process?.terminate() }

    private func append(_ text: String) {
        consoleLog += text
        if consoleLog.count > 200_000 { consoleLog = String(consoleLog.suffix(150_000)) }
    }
}
