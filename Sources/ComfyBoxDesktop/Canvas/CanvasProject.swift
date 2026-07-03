// CanvasProject.swift — Board projects and their placed images
//
// A Canvas is a project: a named, persisted board of images laid out in an
// infinite 2-D space (MindCraft-style). CanvasItem places one image at a
// position/size/rotation with a z-order. Layout logic (add, remove, move,
// resize, z-order, duplicate) lives here as pure value-type methods so it's
// testable without a view. Images are referenced by absolute path — the
// canvas never copies pixels, matching the DAM's file-as-truth model.

import Foundation

/// One image placed on a canvas.
public struct CanvasItem: Identifiable, Codable, Equatable, Sendable {
    public var id: String
    /// Absolute path to the image file.
    public var imagePath: String
    /// Top-left position in canvas coordinates.
    public var x: Double
    public var y: Double
    public var width: Double
    public var height: Double
    /// Rotation in degrees, clockwise.
    public var rotation: Double
    /// Stacking order; higher draws on top.
    public var zIndex: Int
    /// Horizontal mirror.
    public var flippedH: Bool

    public init(
        id: String = UUID().uuidString,
        imagePath: String,
        x: Double = 0,
        y: Double = 0,
        width: Double = 240,
        height: Double = 240,
        rotation: Double = 0,
        zIndex: Int = 0,
        flippedH: Bool = false
    ) {
        self.id = id
        self.imagePath = imagePath
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rotation = rotation
        self.zIndex = zIndex
        self.flippedH = flippedH
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        imagePath = try c.decodeIfPresent(String.self, forKey: .imagePath) ?? ""
        x = try c.decodeIfPresent(Double.self, forKey: .x) ?? 0
        y = try c.decodeIfPresent(Double.self, forKey: .y) ?? 0
        width = try c.decodeIfPresent(Double.self, forKey: .width) ?? 240
        height = try c.decodeIfPresent(Double.self, forKey: .height) ?? 240
        rotation = try c.decodeIfPresent(Double.self, forKey: .rotation) ?? 0
        zIndex = try c.decodeIfPresent(Int.self, forKey: .zIndex) ?? 0
        flippedH = try c.decodeIfPresent(Bool.self, forKey: .flippedH) ?? false
    }
}

/// A named board of images. `id` is a filename-safe slug; the store persists
/// one project per JSON file.
public struct CanvasProject: Identifiable, Codable, Equatable, Sendable {
    public var id: String
    public var name: String
    public var items: [CanvasItem]
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: String = UUID().uuidString,
        name: String,
        items: [CanvasItem] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.items = items
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? "Untitled"
        items = try c.decodeIfPresent([CanvasItem].self, forKey: .items) ?? []
        createdAt = try c.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
        updatedAt = try c.decodeIfPresent(Date.self, forKey: .updatedAt) ?? Date()
    }

    // MARK: - Layout operations (pure)

    /// The next z-index above everything currently placed.
    public var topZIndex: Int { (items.map(\.zIndex).max() ?? -1) + 1 }
    /// The next z-index below everything.
    public var bottomZIndex: Int { (items.map(\.zIndex).min() ?? 0) - 1 }

    /// Add an item on top and return its id.
    @discardableResult
    public mutating func add(_ item: CanvasItem) -> String {
        var placed = item
        placed.zIndex = topZIndex
        items.append(placed)
        return placed.id
    }

    public mutating func remove(id: String) {
        items.removeAll { $0.id == id }
    }

    /// Replace an item in place (move/resize/rotate). Unknown ids are ignored.
    public mutating func update(_ item: CanvasItem) {
        guard let index = items.firstIndex(where: { $0.id == item.id }) else { return }
        items[index] = item
    }

    public mutating func bringToFront(id: String) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        items[index].zIndex = topZIndex
    }

    public mutating func sendToBack(id: String) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        items[index].zIndex = bottomZIndex
    }

    /// Duplicate an item, offset slightly and placed on top. Returns the new id.
    @discardableResult
    public mutating func duplicate(id: String) -> String? {
        guard let original = items.first(where: { $0.id == id }) else { return nil }
        var copy = original
        copy.id = UUID().uuidString
        copy.x += 24
        copy.y += 24
        return add(copy)
    }

    /// Items in draw order (back to front).
    public var itemsInDrawOrder: [CanvasItem] {
        items.sorted { $0.zIndex < $1.zIndex }
    }

    /// The bounding box that contains every item (canvas coordinates), or nil
    /// when empty — used to zoom-to-fit.
    public var contentBounds: (minX: Double, minY: Double, maxX: Double, maxY: Double)? {
        guard !items.isEmpty else { return nil }
        let minX = items.map(\.x).min()!
        let minY = items.map(\.y).min()!
        let maxX = items.map { $0.x + $0.width }.max()!
        let maxY = items.map { $0.y + $0.height }.max()!
        return (minX, minY, maxX, maxY)
    }
}
