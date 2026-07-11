// SmartTab.swift — named, saved Gallery filter combinations ("Smart Tabs").
//
// Purely a local convenience layer: the Gallery already has every individual
// filter control (search, favorites, content mode, character, color label,
// sort). A Smart Tab just snapshots the current combination under a name so
// it can be reapplied in one click instead of resetting each control by hand.

import Foundation

struct SmartTab: Codable, Identifiable, Equatable {
    let id: String
    var name: String
    var searchText: String
    var filterFavorites: Bool
    var filterContentMode: String?
    var filterCharacter: String?
    var filterLabel: String?
    var sortOrder: String
}

/// Loads/saves the Smart Tab list at ~/.comfybox/smart-tabs.json, mirroring
/// DesktopSettings' storage pattern.
enum SmartTabStore {
    static var configPath: String {
        let dir = NSString(string: "~/.comfybox").expandingTildeInPath
        return (dir as NSString).appendingPathComponent("smart-tabs.json")
    }

    static func load() -> [SmartTab] {
        let path = configPath
        guard FileManager.default.fileExists(atPath: path),
              let data = FileManager.default.contents(atPath: path) else {
            return []
        }
        return (try? JSONDecoder().decode([SmartTab].self, from: data)) ?? []
    }

    static func save(_ tabs: [SmartTab]) {
        let path = configPath
        let dir = (path as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(tabs) else { return }
        try? data.write(to: URL(fileURLWithPath: path))
    }
}
