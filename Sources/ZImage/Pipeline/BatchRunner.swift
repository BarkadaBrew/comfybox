import Foundation

/// Configuration for a multi-seed batch run.
public struct BatchConfig: Sendable {
    public let seeds: [UInt64]
    public let autoSeedCount: Int?        // generate N random seeds
    public let outputPattern: String      // e.g., "output_{seed}.png"
    public let continueOnError: Bool      // default true
    public let metadataEnabled: Bool      // write JSON sidecar per image
    public let promptFilePath: String?    // re-read before each iteration
    public let checkpointPath: String?    // batch progress file

    public init(
        seeds: [UInt64] = [],
        autoSeedCount: Int? = nil,
        outputPattern: String = "{output}_{seed}.png",
        continueOnError: Bool = true,
        metadataEnabled: Bool = false,
        promptFilePath: String? = nil,
        checkpointPath: String? = nil
    ) {
        self.seeds = seeds
        self.autoSeedCount = autoSeedCount
        self.outputPattern = outputPattern
        self.continueOnError = continueOnError
        self.metadataEnabled = metadataEnabled
        self.promptFilePath = promptFilePath
        self.checkpointPath = checkpointPath
    }
}

/// Result summary for a completed batch run.
public struct BatchResult: Sendable {
    public let totalSeeds: Int
    public let completed: Int
    public let failed: Int
    public let skipped: Int              // from checkpoint resume
    public let outputs: [(seed: UInt64, path: String?, error: String?)]
    public let totalDuration: TimeInterval

    public init(
        totalSeeds: Int,
        completed: Int,
        failed: Int,
        skipped: Int,
        outputs: [(seed: UInt64, path: String?, error: String?)],
        totalDuration: TimeInterval
    ) {
        self.totalSeeds = totalSeeds
        self.completed = completed
        self.failed = failed
        self.skipped = skipped
        self.outputs = outputs
        self.totalDuration = totalDuration
    }
}

/// Sequential multi-seed batch runner. GPU renders are NEVER concurrent.
/// Supports checkpoint resume, prompt-file re-read, and output path templating.
public struct BatchRunner {

    /// Run a batch of seeds sequentially.
    ///
    /// - Parameters:
    ///   - config: Batch configuration (seeds, output pattern, checkpoint, etc.)
    ///   - generate: Closure called for each seed. Receives (seed, prompt?) and returns the output path.
    ///               The prompt parameter is non-nil only when `promptFilePath` is set and the file is re-read.
    /// - Returns: Summary of the batch run.
    public static func run(
        config: BatchConfig,
        generate: (_ seed: UInt64, _ prompt: String?) async throws -> String
    ) async throws -> BatchResult {
        let startTime = Date()

        // Resolve seeds: auto-generate or use explicit list
        let allSeeds: [UInt64]
        if let count = config.autoSeedCount, count > 0 {
            var generated: [UInt64] = []
            for _ in 0..<count {
                generated.append(UInt64.random(in: 0...UInt64.max))
            }
            // Append any explicitly provided seeds
            allSeeds = generated + config.seeds
        } else {
            allSeeds = config.seeds
        }

        guard !allSeeds.isEmpty else {
            return BatchResult(
                totalSeeds: 0, completed: 0, failed: 0, skipped: 0,
                outputs: [], totalDuration: 0
            )
        }

        // Load checkpoint for resume
        let checkpoint = BatchCheckpoint(path: config.checkpointPath)
        let completedSeeds = checkpoint.load()
        let skippedCount = allSeeds.filter { completedSeeds.contains($0) }.count

        if skippedCount > 0 {
            fputs("[batch] Resuming: \(skippedCount) seed(s) already completed, skipping.\n", stderr)
        }

        var outputs: [(seed: UInt64, path: String?, error: String?)] = []
        var completedCount = 0
        var failedCount = 0

        for (index, seed) in allSeeds.enumerated() {
            // Skip if already completed (checkpoint resume)
            if completedSeeds.contains(seed) {
                outputs.append((seed: seed, path: nil, error: nil))
                continue
            }

            // Re-read prompt file if configured
            let prompt: String?
            if let promptFilePath = config.promptFilePath {
                do {
                    prompt = try String(contentsOfFile: promptFilePath, encoding: .utf8)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                } catch {
                    fputs("[batch] WARNING: Failed to read prompt file \(promptFilePath): \(error). Using original prompt.\n", stderr)
                    prompt = nil
                }
            } else {
                prompt = nil
            }

            // Resolve output path from pattern
            let outputPath = resolveOutputPath(
                pattern: config.outputPattern,
                seed: seed,
                index: index
            )

            fputs("[batch] [\(index + 1)/\(allSeeds.count)] seed=\(seed) -> \(outputPath)\n", stderr)

            do {
                let actualPath = try await generate(seed, prompt)
                completedCount += 1
                outputs.append((seed: seed, path: actualPath, error: nil))
                checkpoint.append(seed: seed, path: actualPath)
            } catch {
                failedCount += 1
                let errorMsg = error.localizedDescription
                outputs.append((seed: seed, path: nil, error: errorMsg))
                fputs("[batch] FAILED seed=\(seed): \(errorMsg)\n", stderr)

                if !config.continueOnError {
                    break
                }
            }
        }

        let duration = Date().timeIntervalSince(startTime)
        return BatchResult(
            totalSeeds: allSeeds.count,
            completed: completedCount,
            failed: failedCount,
            skipped: skippedCount,
            outputs: outputs,
            totalDuration: duration
        )
    }

    /// Replace `{seed}`, `{index}`, and `{output}` placeholders in the pattern.
    public static func resolveOutputPath(pattern: String, seed: UInt64, index: Int) -> String {
        var result = pattern
        result = result.replacingOccurrences(of: "{seed}", with: String(seed))
        result = result.replacingOccurrences(of: "{index}", with: String(index))
        return result
    }

    /// Build an output pattern from a base output path.
    /// "photo.png" -> "photo_{seed}.png"
    public static func outputPattern(from outputPath: String) -> String {
        let url = URL(fileURLWithPath: outputPath)
        let ext = url.pathExtension
        let basename = url.deletingPathExtension().lastPathComponent
        let dir = url.deletingLastPathComponent().path
        if ext.isEmpty {
            return (dir as NSString).appendingPathComponent("\(basename)_{seed}")
        }
        return (dir as NSString).appendingPathComponent("\(basename)_{seed}.\(ext)")
    }
}
