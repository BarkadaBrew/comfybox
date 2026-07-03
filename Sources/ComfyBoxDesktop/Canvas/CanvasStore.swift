// CanvasStore.swift — Persistence for canvas projects
//
// One JSON file per project under ~/.comfybox/canvases/. Mutations update the
// in-memory list and persist the touched project atomically. UI binds to
// `projects`; edits flow through the mutating helpers so a save always
// follows a change.

import Foundation

@Observable
@MainActor
public final class CanvasStore {
    public private(set) var projects: [CanvasProject] = []

    private let directory: URL

    public nonisolated static func defaultDirectory() -> URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent(".comfybox/canvases", isDirectory: true)
    }

    public init(directory: URL = CanvasStore.defaultDirectory()) {
        self.directory = directory
        load()
    }

    // MARK: - Load

    private func load() {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        else { projects = []; return }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        projects = files
            .filter { $0.pathExtension == "json" }
            .compactMap { try? Data(contentsOf: $0) }
            .compactMap { try? decoder.decode(CanvasProject.self, from: $0) }
            .sorted { $0.updatedAt > $1.updatedAt }
    }

    // MARK: - CRUD

    @discardableResult
    public func create(name: String) -> CanvasProject {
        let project = CanvasProject(name: name.isEmpty ? "Untitled Canvas" : name)
        projects.insert(project, at: 0)
        persist(project)
        return project
    }

    public func rename(id: String, to name: String) {
        mutate(id) { $0.name = name }
    }

    public func delete(id: String) {
        projects.removeAll { $0.id == id }
        let url = fileURL(for: id)
        try? FileManager.default.removeItem(at: url)
    }

    @discardableResult
    public func duplicate(id: String) -> CanvasProject? {
        guard let source = projects.first(where: { $0.id == id }) else { return nil }
        var copy = source
        copy.id = UUID().uuidString
        copy.name = "\(source.name) Copy"
        copy.createdAt = Date()
        copy.updatedAt = Date()
        projects.insert(copy, at: 0)
        persist(copy)
        return copy
    }

    // MARK: - Item edits (persist the touched project)

    public func addItem(_ item: CanvasItem, toCanvas id: String) {
        mutate(id) { $0.add(item) }
    }

    public func updateItem(_ item: CanvasItem, inCanvas id: String) {
        mutate(id) { $0.update(item) }
    }

    public func removeItem(itemId: String, fromCanvas id: String) {
        mutate(id) { $0.remove(id: itemId) }
    }

    public func mutateCanvas(_ id: String, _ body: (inout CanvasProject) -> Void) {
        mutate(id, body)
    }

    public func project(_ id: String) -> CanvasProject? {
        projects.first { $0.id == id }
    }

    // MARK: - Internals

    private func mutate(_ id: String, _ body: (inout CanvasProject) -> Void) {
        guard let index = projects.firstIndex(where: { $0.id == id }) else { return }
        body(&projects[index])
        projects[index].updatedAt = Date()
        persist(projects[index])
    }

    private func fileURL(for id: String) -> URL {
        directory.appendingPathComponent("\(id).json")
    }

    private func persist(_ project: CanvasProject) {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(project) {
            try? data.write(to: fileURL(for: project.id), options: .atomic)
        }
    }
}
