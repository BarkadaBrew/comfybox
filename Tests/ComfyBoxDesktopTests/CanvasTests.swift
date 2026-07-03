// CanvasTests.swift — Canvas project layout logic and store persistence

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("CanvasProject")
struct CanvasProjectTests {
    @Test("add places items on top with increasing z-index")
    func addZOrder() {
        var project = CanvasProject(name: "Board")
        let a = project.add(CanvasItem(imagePath: "/a.png"))
        let b = project.add(CanvasItem(imagePath: "/b.png"))
        let za = project.items.first { $0.id == a }!.zIndex
        let zb = project.items.first { $0.id == b }!.zIndex
        #expect(zb > za)
        #expect(project.itemsInDrawOrder.last?.id == b)
    }

    @Test("bringToFront / sendToBack reorder")
    func reorder() {
        var project = CanvasProject(name: "Board")
        let a = project.add(CanvasItem(imagePath: "/a.png"))
        let b = project.add(CanvasItem(imagePath: "/b.png"))
        project.sendToBack(id: b)
        #expect(project.itemsInDrawOrder.first?.id == b)
        project.bringToFront(id: b)
        #expect(project.itemsInDrawOrder.last?.id == b)
        _ = a
    }

    @Test("update replaces geometry in place")
    func update() {
        var project = CanvasProject(name: "Board")
        let id = project.add(CanvasItem(imagePath: "/a.png", x: 0, y: 0))
        var item = project.items.first { $0.id == id }!
        item.x = 100; item.y = 50; item.rotation = 30
        project.update(item)
        let stored = project.items.first { $0.id == id }!
        #expect(stored.x == 100)
        #expect(stored.rotation == 30)
    }

    @Test("duplicate offsets a copy on top")
    func duplicate() {
        var project = CanvasProject(name: "Board")
        let id = project.add(CanvasItem(imagePath: "/a.png", x: 10, y: 10))
        let copyId = project.duplicate(id: id)
        #expect(copyId != nil)
        #expect(project.items.count == 2)
        let copy = project.items.first { $0.id == copyId }!
        #expect(copy.x == 34)   // +24 offset
        #expect(copy.imagePath == "/a.png")
        #expect(project.itemsInDrawOrder.last?.id == copyId)
    }

    @Test("remove deletes by id")
    func remove() {
        var project = CanvasProject(name: "Board")
        let id = project.add(CanvasItem(imagePath: "/a.png"))
        project.remove(id: id)
        #expect(project.items.isEmpty)
    }

    @Test("contentBounds spans all items, nil when empty")
    func bounds() {
        var project = CanvasProject(name: "Board")
        #expect(project.contentBounds == nil)
        project.add(CanvasItem(imagePath: "/a.png", x: 0, y: 0, width: 100, height: 100))
        project.add(CanvasItem(imagePath: "/b.png", x: 200, y: 50, width: 100, height: 100))
        let b = project.contentBounds!
        #expect(b.minX == 0)
        #expect(b.minY == 0)
        #expect(b.maxX == 300)
        #expect(b.maxY == 150)
    }
}

@Suite("CanvasStore")
@MainActor
struct CanvasStoreTests {
    private func tempDir() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("canvas-tests-\(UUID().uuidString)", isDirectory: true)
    }

    @Test("create, rename, delete persist to disk")
    func crud() {
        let dir = tempDir()
        let store = CanvasStore(directory: dir)
        let project = store.create(name: "Shoot A")
        #expect(store.projects.count == 1)

        store.rename(id: project.id, to: "Shoot B")
        #expect(store.project(project.id)?.name == "Shoot B")

        // A fresh instance reads it back.
        let reloaded = CanvasStore(directory: dir)
        #expect(reloaded.projects.first?.name == "Shoot B")

        store.delete(id: project.id)
        #expect(store.projects.isEmpty)
        let afterDelete = CanvasStore(directory: dir)
        #expect(afterDelete.projects.isEmpty)
        try? FileManager.default.removeItem(at: dir)
    }

    @Test("item edits persist through the store")
    func itemEdits() {
        let dir = tempDir()
        let store = CanvasStore(directory: dir)
        let project = store.create(name: "Board")

        store.addItem(CanvasItem(imagePath: "/x.png", x: 5, y: 5), toCanvas: project.id)
        #expect(store.project(project.id)?.items.count == 1)

        var item = store.project(project.id)!.items[0]
        item.x = 99
        store.updateItem(item, inCanvas: project.id)

        let reloaded = CanvasStore(directory: dir)
        #expect(reloaded.project(project.id)?.items.first?.x == 99)

        store.removeItem(itemId: item.id, fromCanvas: project.id)
        #expect(store.project(project.id)?.items.isEmpty == true)
        try? FileManager.default.removeItem(at: dir)
    }

    @Test("duplicate copies a canvas with its items")
    func duplicateCanvas() {
        let dir = tempDir()
        let store = CanvasStore(directory: dir)
        let project = store.create(name: "Original")
        store.addItem(CanvasItem(imagePath: "/x.png"), toCanvas: project.id)

        let copy = store.duplicate(id: project.id)
        #expect(copy != nil)
        #expect(copy?.name == "Original Copy")
        #expect(copy?.items.count == 1)
        #expect(store.projects.count == 2)
        try? FileManager.default.removeItem(at: dir)
    }
}
