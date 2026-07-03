// DAMFolder.swift — Virtual folder for filing gallery assets
//
// Folders are database-only groupings; files never move on disk (moving
// them would fight the AssetIngestor's path tracking). An asset belongs
// to at most one folder — see DAMStore's asset_folders mapping.

import Foundation

public struct DAMFolder: Identifiable, Sendable, Equatable, Hashable {
    public let id: String
    public var name: String
    public let createdAt: Date

    public init(id: String = UUID().uuidString, name: String, createdAt: Date = Date()) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
    }
}
