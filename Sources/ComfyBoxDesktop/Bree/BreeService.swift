// BreeService.swift — Bree companion handoff channel (desktop side)
//
// Bree runs on the home server; the durable, offline-safe integration point is
// the shared vault handoff channel (two append-only markdown files). This
// service reads Bree's replies and appends timestamped messages to her inbox,
// following the "append with timestamp headers, never delete" convention.

import Foundation

@Observable
@MainActor
public final class BreeService {
    /// Directory holding desktop-to-bree.md and bree-to-desktop.md.
    public var handoffDirectory: String

    public private(set) var inbox: String = ""      // bree-to-desktop.md
    public private(set) var outbox: String = ""     // desktop-to-bree.md (history)
    public private(set) var lastLoaded: Date?
    public private(set) var lastError: String?

    public init(handoffDirectory: String = BreeService.defaultDirectory) {
        self.handoffDirectory = handoffDirectory
    }

    public nonisolated static var defaultDirectory: String {
        NSString(string: "~/Documents/Vaults/BarkadaAI/Coffee Shop/Handoff").expandingTildeInPath
    }

    public var inboxPath: String { (handoffDirectory as NSString).appendingPathComponent("bree-to-desktop.md") }
    public var outboxPath: String { (handoffDirectory as NSString).appendingPathComponent("desktop-to-bree.md") }

    public var isAvailable: Bool {
        FileManager.default.fileExists(atPath: handoffDirectory)
    }

    // MARK: - Pure formatting (tested)

    /// A handoff entry with a timestamp header, per the channel convention.
    public nonisolated static func formatEntry(_ message: String, timestamp: Date, author: String = "Desktop") -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd HH:mm"
        let stamp = fmt.string(from: timestamp)
        let body = message.trimmingCharacters(in: .whitespacesAndNewlines)
        return "\n## \(stamp) — \(author)\n\n\(body)\n"
    }

    // MARK: - IO

    public func reload() {
        lastError = nil
        inbox = (try? String(contentsOfFile: inboxPath, encoding: .utf8)) ?? ""
        outbox = (try? String(contentsOfFile: outboxPath, encoding: .utf8)) ?? ""
        lastLoaded = Date()
    }

    /// Append a timestamped message to Bree's inbox (desktop-to-bree.md).
    public func send(_ message: String) {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let entry = Self.formatEntry(trimmed, timestamp: Date())
        do {
            try FileManager.default.createDirectory(atPath: handoffDirectory, withIntermediateDirectories: true)
            let existing = (try? String(contentsOfFile: outboxPath, encoding: .utf8)) ?? ""
            let combined = existing + entry
            try combined.write(toFile: outboxPath, atomically: true, encoding: .utf8)
            outbox = combined
            lastError = nil
        } catch {
            lastError = "Couldn't write to Bree's inbox: \(error.localizedDescription)"
        }
    }
}
