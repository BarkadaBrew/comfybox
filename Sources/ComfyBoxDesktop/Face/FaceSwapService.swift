// FaceSwapService.swift — Desktop-only face swap (insightface + inswapper)
//
// Shells out to the local faceswap venv (~/Projects/faceswap), like mflux /
// Découpage. Swaps the source face onto the target image. Desktop-only — the
// ComfyBox server never depends on it. Arg building is pure and unit-tested.

import Foundation

@Observable
@MainActor
public final class FaceSwapService {
    public var projectDirectory: String

    public private(set) var isRunning = false
    public private(set) var consoleLog = ""
    private var process: Process?

    public init(projectDirectory: String = FaceSwapService.defaultProjectDirectory) {
        self.projectDirectory = projectDirectory
    }

    public nonisolated static var defaultProjectDirectory: String {
        NSString(string: "~/Projects/faceswap").expandingTildeInPath
    }

    public var pythonPath: String { (projectDirectory as NSString).appendingPathComponent(".venv/bin/python") }
    public var scriptPath: String { (projectDirectory as NSString).appendingPathComponent("swap.py") }
    public var modelPath: String { (projectDirectory as NSString).appendingPathComponent("models/inswapper_128.onnx") }

    /// Backend is ready only when the venv, script, and model are all present.
    public var isInstalled: Bool {
        let fm = FileManager.default
        return fm.isExecutableFile(atPath: pythonPath)
            && fm.fileExists(atPath: scriptPath)
            && fm.fileExists(atPath: modelPath)
    }

    // MARK: - Pure args (tested)

    public nonisolated static func swapArgs(script: String, source: String, target: String,
                                            output: String, allFaces: Bool) -> [String] {
        var args = [script, source, target, output]
        if allFaces { args.append("--all") }
        return args
    }

    // MARK: - Execution

    public enum FaceSwapError: Error, LocalizedError {
        case notInstalled(String), alreadyRunning, failed(Int32, String)
        public var errorDescription: String? {
            switch self {
            case .notInstalled(let d): return "Face-swap backend not installed at \(d). Run the setup (insightface + inswapper_128.onnx)."
            case .alreadyRunning: return "A face swap is already running."
            case .failed(let c, let out):
                let msg = out.split(separator: "\n").last.map(String.init) ?? ""
                return "Face swap failed (\(c)): \(msg)"
            }
        }
    }

    /// Swap `source`'s face onto `target`, writing `output`. Returns the output path.
    @discardableResult
    public func swap(source: String, target: String, output: String, allFaces: Bool) async throws -> String {
        guard !isRunning else { throw FaceSwapError.alreadyRunning }
        guard isInstalled else { throw FaceSwapError.notInstalled(projectDirectory) }
        consoleLog = ""
        isRunning = true
        defer { isRunning = false; process = nil }

        let args = Self.swapArgs(script: scriptPath, source: source, target: target, output: output, allFaces: allFaces)
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pythonPath)
        proc.arguments = args
        proc.currentDirectoryURL = URL(fileURLWithPath: projectDirectory)
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        process = proc
        append("$ python swap.py \(source) \(target) \(output)\(allFaces ? " --all" : "")\n")

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
        guard proc.terminationStatus == 0 else {
            throw FaceSwapError.failed(proc.terminationStatus, consoleLog)
        }
        return output
    }

    public func cancel() { process?.terminate() }

    private func append(_ text: String) {
        consoleLog += text
        if consoleLog.count > 100_000 { consoleLog = String(consoleLog.suffix(80_000)) }
    }
}
