// ActivityLog.swift — Unified recent-activity feed for the hub
//
// A small in-memory ring of notable events (renders, service actions, Bree
// messages, downloads, connection changes) shown in the menu bar and Dashboard.
// Not persisted — it's a "what just happened" glance, not an audit log.

import Foundation

@Observable
@MainActor
public final class ActivityLog {
    public struct Entry: Identifiable, Sendable {
        public let id = UUID()
        public let date: Date
        public let icon: String
        public let message: String
    }

    public private(set) var entries: [Entry] = []
    private let limit = 100

    public init() {}

    public func log(_ icon: String, _ message: String) {
        entries.insert(Entry(date: Date(), icon: icon, message: message), at: 0)
        if entries.count > limit { entries.removeLast(entries.count - limit) }
    }

    public var recent: [Entry] { Array(entries.prefix(12)) }
}
