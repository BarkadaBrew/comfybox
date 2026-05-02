import Foundation

/// Append-only JSONL checkpoint file for batch crash recovery.
/// Each completed seed is appended as a single JSON line. On resume,
/// already-completed seeds are skipped — no work is repeated.
public struct BatchCheckpoint: Sendable {
    public let path: String?

    public struct Entry: Codable, Sendable {
        public let seed: UInt64
        public let outputPath: String
        public let completedAt: Date

        public init(seed: UInt64, outputPath: String, completedAt: Date = Date()) {
            self.seed = seed
            self.outputPath = outputPath
            self.completedAt = completedAt
        }
    }

    public init(path: String?) {
        self.path = path
    }

    /// Load completed seeds from checkpoint file. Returns empty set if no file.
    public func load() -> Set<UInt64> {
        guard let path = path else { return [] }
        guard let data = try? String(contentsOfFile: path, encoding: .utf8) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var seeds = Set<UInt64>()
        for line in data.split(separator: "\n") where !line.isEmpty {
            if let entry = try? decoder.decode(Entry.self, from: Data(line.utf8)) {
                seeds.insert(entry.seed)
            }
        }
        return seeds
    }

    /// Load full checkpoint entries (seed + output path + timestamp).
    public func loadEntries() -> [Entry] {
        guard let path = path else { return [] }
        guard let data = try? String(contentsOfFile: path, encoding: .utf8) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var entries: [Entry] = []
        for line in data.split(separator: "\n") where !line.isEmpty {
            if let entry = try? decoder.decode(Entry.self, from: Data(line.utf8)) {
                entries.append(entry)
            }
        }
        return entries
    }

    /// Append a completed seed to the checkpoint file.
    public func append(seed: UInt64, path outputPath: String) {
        guard let checkpointPath = path else { return }
        let entry = Entry(seed: seed, outputPath: outputPath, completedAt: Date())
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(entry),
              var line = String(data: data, encoding: .utf8) else { return }
        line += "\n"
        if let handle = FileHandle(forWritingAtPath: checkpointPath) {
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
            handle.closeFile()
        } else {
            try? line.write(toFile: checkpointPath, atomically: true, encoding: .utf8)
        }
    }
}
