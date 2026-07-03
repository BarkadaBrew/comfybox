// ShotTemplateStore.swift — Reusable shot / camera directives
//
// Loads the legacy image-service shot-templates.json (cinematic directives)
// so they're selectable alongside the composed camera phrase. Format is an
// object keyed by id: { directive, tags, contentMode }.

import Foundation

public struct ShotTemplate: Identifiable, Sendable, Equatable {
    public let id: String
    public let directive: String
    public let tags: [String]
    public let contentMode: String?

    /// Title-cased name derived from the id ("cinematic-wide" → "Cinematic Wide").
    public var name: String {
        id.split(whereSeparator: { $0 == "-" || $0 == "_" })
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }
}

@Observable
@MainActor
public final class ShotTemplateStore {
    public private(set) var templates: [ShotTemplate] = []

    private let path: URL

    /// Legacy image-service location.
    public nonisolated static func defaultPath() -> URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent(".coffeeshop/image-service/shot-templates.json")
    }

    public init(path: URL = ShotTemplateStore.defaultPath()) {
        self.path = path
        load()
    }

    public func reload() { load() }

    private func load() {
        templates = Self.parse(contentsOf: path)
    }

    /// Parse the object-keyed shot-templates.json into a sorted list.
    nonisolated static func parse(contentsOf url: URL) -> [ShotTemplate] {
        guard let data = try? Data(contentsOf: url),
              let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return [] }
        return parse(object: object)
    }

    nonisolated static func parse(object: [String: Any]) -> [ShotTemplate] {
        object.compactMap { key, value -> ShotTemplate? in
            guard let entry = value as? [String: Any],
                  let directive = entry["directive"] as? String, !directive.isEmpty
            else { return nil }
            return ShotTemplate(
                id: key,
                directive: directive,
                tags: (entry["tags"] as? [String]) ?? [],
                contentMode: entry["contentMode"] as? String
            )
        }
        .sorted { $0.id < $1.id }
    }
}
