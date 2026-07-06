// CanvasView.swift — Infinite board where images lay out as a project
//
// MindCraft-style board (v1): a left rail of Canvas projects and an infinite
// pan/zoom board on the right. Drop gallery images or add files, then move,
// resize, rotate, reorder, duplicate, flip, and remove them via drag handles
// and a right-click menu. Everything persists through CanvasStore.

import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct CanvasView: View {
    @Bindable var store: CanvasStore
    /// Images available to drop in from the gallery (absolute paths).
    var galleryImagePaths: [String] = []
    /// Send an image on the board to the Generate tab to re-render it
    /// (the app resolves the path's original prompt from the DAM).
    var onSendToGenerate: ((String) -> Void)?
    /// Send an image on the board to Generate as an img2img reference.
    var onUseAsReference: ((String) -> Void)?

    @State private var selectedCanvasId: String?
    @State private var renaming: CanvasProject?
    @State private var renameText: String = ""
    @State private var showNewPrompt = false
    @State private var newName = ""

    var body: some View {
        HStack(spacing: 0) {
            projectRail
                .frame(width: 200)
            Divider()
            if let id = selectedCanvasId, store.project(id) != nil {
                CanvasBoard(store: store, canvasId: id, onSendToGenerate: onSendToGenerate, onUseAsReference: onUseAsReference)
                    .id(id)
            } else {
                emptyBoard
            }
        }
        .navigationTitle("Canvas")
        .onAppear {
            if selectedCanvasId == nil { selectedCanvasId = store.projects.first?.id }
        }
        .alert("New Canvas", isPresented: $showNewPrompt) {
            TextField("Name", text: $newName)
            Button("Create") {
                let p = store.create(name: newName)
                newName = ""
                selectedCanvasId = p.id
            }
            Button("Cancel", role: .cancel) { newName = "" }
        }
        .alert("Rename Canvas", isPresented: Binding(
            get: { renaming != nil }, set: { if !$0 { renaming = nil } }
        )) {
            TextField("Name", text: $renameText)
            Button("Rename") {
                if let r = renaming { store.rename(id: r.id, to: renameText) }
                renaming = nil
            }
            Button("Cancel", role: .cancel) { renaming = nil }
        }
    }

    private var projectRail: some View {
        VStack(spacing: 0) {
            List(selection: $selectedCanvasId) {
                Section("Canvases") {
                    ForEach(store.projects) { project in
                        HStack {
                            Image(systemName: "rectangle.on.rectangle.angled")
                                .foregroundStyle(.secondary)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(project.name).lineLimit(1)
                                Text("\(project.items.count) image\(project.items.count == 1 ? "" : "s")")
                                    .font(.caption2).foregroundStyle(.tertiary)
                            }
                        }
                        .tag(project.id)
                        .contextMenu {
                            Button("Rename…") {
                                renameText = project.name
                                renaming = project
                            }
                            Button("Duplicate") {
                                if let c = store.duplicate(id: project.id) { selectedCanvasId = c.id }
                            }
                            Divider()
                            Button("Delete", role: .destructive) {
                                store.delete(id: project.id)
                                if selectedCanvasId == project.id {
                                    selectedCanvasId = store.projects.first?.id
                                }
                            }
                        }
                    }
                }
            }
            .listStyle(.sidebar)

            Divider()
            Button {
                newName = ""
                showNewPrompt = true
            } label: {
                Label("New Canvas", systemImage: "plus.rectangle.on.rectangle")
                    .font(.callout)
            }
            .buttonStyle(.borderless)
            .padding(10)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var emptyBoard: some View {
        VStack(spacing: 10) {
            Image(systemName: "rectangle.on.rectangle.angled")
                .font(.system(size: 40)).foregroundStyle(.tertiary)
            Text("No canvas selected").font(.headline).foregroundStyle(.secondary)
            Button("New Canvas") { showNewPrompt = true }
                .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(boardBackground)
    }

    private var boardBackground: some View {
        Color(nsColor: .underPageBackgroundColor)
    }
}

// MARK: - Board

private struct CanvasBoard: View {
    @Bindable var store: CanvasStore
    let canvasId: String
    var onSendToGenerate: ((String) -> Void)?
    var onUseAsReference: ((String) -> Void)?

    // View transform: board pan (screen points) + zoom.
    @State private var pan: CGSize = .zero
    @State private var panStart: CGSize = .zero
    @State private var zoom: CGFloat = 1
    @State private var selectedItemId: String?

    // Live drag/resize state for the active item (canvas coordinates).
    @State private var dragItemStart: CanvasItem?

    private var project: CanvasProject { store.project(canvasId) ?? CanvasProject(name: "") }

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Color(nsColor: .underPageBackgroundColor)
                    .contentShape(Rectangle())
                    .onTapGesture { selectedItemId = nil }
                    .gesture(panGesture)

                // Items, transformed by pan + zoom.
                ForEach(project.itemsInDrawOrder) { item in
                    itemView(item)
                }
            }
            .clipped()
            .overlay(alignment: .bottom) { toolbar }
            .overlay(alignment: .topTrailing) { dropHint }
            .onDrop(of: [.fileURL], isTargeted: nil) { providers in
                handleDrop(providers, in: geo.size)
            }
        }
    }

    // MARK: - Item rendering

    @ViewBuilder
    private func itemView(_ item: CanvasItem) -> some View {
        let isSelected = selectedItemId == item.id
        CanvasImageView(path: item.imagePath)
            .frame(width: item.width * zoom, height: item.height * zoom)
            .scaleEffect(x: item.flippedH ? -1 : 1, y: 1)
            .rotationEffect(.degrees(item.rotation))
            .overlay {
                if isSelected {
                    Rectangle().stroke(Color.accentColor, lineWidth: 2)
                }
            }
            .overlay(alignment: .bottomTrailing) {
                if isSelected { resizeHandle(item) }
            }
            .position(
                x: screenX(item.x + item.width / 2, in: item),
                y: screenY(item.y + item.height / 2, in: item)
            )
            .onTapGesture { selectedItemId = item.id }
            .gesture(moveGesture(item))
            .contextMenu { itemMenu(item) }
    }

    private func resizeHandle(_ item: CanvasItem) -> some View {
        Circle()
            .fill(Color.accentColor)
            .frame(width: 14, height: 14)
            .overlay(Circle().stroke(.white, lineWidth: 1.5))
            .offset(x: 7, y: 7)
            .gesture(resizeGesture(item))
    }

    // Screen position of a canvas point (item center) under pan+zoom.
    private func screenX(_ canvasX: Double, in item: CanvasItem) -> CGFloat {
        CGFloat(canvasX) * zoom + pan.width
    }
    private func screenY(_ canvasY: Double, in item: CanvasItem) -> CGFloat {
        CGFloat(canvasY) * zoom + pan.height
    }

    // MARK: - Gestures

    private var panGesture: some Gesture {
        DragGesture()
            .onChanged { value in
                pan = CGSize(width: panStart.width + value.translation.width,
                             height: panStart.height + value.translation.height)
            }
            .onEnded { _ in panStart = pan }
    }

    private func moveGesture(_ item: CanvasItem) -> some Gesture {
        DragGesture()
            .onChanged { value in
                // Snapshot the item at drag start, then offset from it so the
                // move tracks the cursor 1:1 in canvas space (divide by zoom).
                if dragItemStart?.id != item.id {
                    dragItemStart = item
                    selectedItemId = item.id
                }
                guard let start = dragItemStart else { return }
                var moved = start
                moved.x = start.x + Double(value.translation.width / zoom)
                moved.y = start.y + Double(value.translation.height / zoom)
                store.updateItem(moved, inCanvas: canvasId)
            }
            .onEnded { _ in dragItemStart = nil }
    }

    private func resizeGesture(_ item: CanvasItem) -> some Gesture {
        DragGesture()
            .onChanged { value in
                if dragItemStart?.id != item.id { dragItemStart = item }
                guard let start = dragItemStart else { return }
                var m = start
                let aspect = start.height > 0 ? start.width / start.height : 1
                let newWidth = max(40, start.width + Double(value.translation.width / zoom))
                m.width = newWidth
                m.height = newWidth / max(aspect, 0.01)   // keep aspect
                store.updateItem(m, inCanvas: canvasId)
            }
            .onEnded { _ in dragItemStart = nil }
    }

    // MARK: - Context menu

    @ViewBuilder
    private func itemMenu(_ item: CanvasItem) -> some View {
        Button("Bring to Front") { store.mutateCanvas(canvasId) { $0.bringToFront(id: item.id) } }
        Button("Send to Back") { store.mutateCanvas(canvasId) { $0.sendToBack(id: item.id) } }
        Divider()
        if onSendToGenerate != nil {
            Button("Send to Generate (Re-render)") { onSendToGenerate?(item.imagePath) }
        }
        if onUseAsReference != nil {
            Button("Use as Reference (img2img)") { onUseAsReference?(item.imagePath) }
        }
        Button("Replace Image…") { replaceImage(item) }
        Button("Duplicate") { store.mutateCanvas(canvasId) { $0.duplicate(id: item.id) } }
        Button(item.flippedH ? "Unflip Horizontal" : "Flip Horizontal") {
            var m = item; m.flippedH.toggle(); store.updateItem(m, inCanvas: canvasId)
        }
        Menu("Rotate") {
            Button("Rotate 90° ↻") { rotate(item, by: 90) }
            Button("Rotate 90° ↺") { rotate(item, by: -90) }
            Button("Reset Rotation") { var m = item; m.rotation = 0; store.updateItem(m, inCanvas: canvasId) }
        }
        Button("Reset Size") {
            var m = item; m.width = 240; m.height = 240; store.updateItem(m, inCanvas: canvasId)
        }
        Divider()
        Button("Reveal in Finder") {
            NSWorkspace.shared.selectFile(item.imagePath, inFileViewerRootedAtPath: "")
        }
        Button("Remove from Canvas", role: .destructive) {
            store.removeItem(itemId: item.id, fromCanvas: canvasId)
            if selectedItemId == item.id { selectedItemId = nil }
        }
    }

    private func rotate(_ item: CanvasItem, by degrees: Double) {
        var m = item
        m.rotation = (m.rotation + degrees).truncatingRemainder(dividingBy: 360)
        store.updateItem(m, inCanvas: canvasId)
    }

    /// Swap the image file for a canvas item, keeping its position/size/rotation.
    private func replaceImage(_ item: CanvasItem) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.png, .jpeg, .image]
        panel.prompt = "Replace"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        var m = item
        m.imagePath = url.path
        store.updateItem(m, inCanvas: canvasId)
    }

    // MARK: - Toolbar / zoom

    private var toolbar: some View {
        HStack(spacing: 10) {
            Button { addImagesFromPicker() } label: { Label("Add Images", systemImage: "photo.badge.plus") }
            Button { exportBoard() } label: { Label("Export", systemImage: "square.and.arrow.up") }
                .disabled(project.items.isEmpty)
            Divider().frame(height: 16)
            Button { zoomBy(0.8) } label: { Image(systemName: "minus.magnifyingglass") }
            Text("\(Int(zoom * 100))%")
                .font(.caption.monospacedDigit()).frame(width: 46)
            Button { zoomBy(1.25) } label: { Image(systemName: "plus.magnifyingglass") }
            Divider().frame(height: 16)
            Button("Fit") { zoomToFit() }
            Button("Reset") { withAnimation { zoom = 1; pan = .zero; panStart = .zero } }
        }
        .buttonStyle(.borderless)
        .padding(8)
        .background(.regularMaterial, in: Capsule())
        .padding(12)
    }

    private var dropHint: some View {
        Text("Drag images here, or drag from the Gallery")
            .font(.caption).foregroundStyle(.secondary)
            .padding(8)
            .background(.regularMaterial, in: Capsule())
            .padding(12)
            .opacity(project.items.isEmpty ? 1 : 0)
    }

    private func zoomBy(_ factor: CGFloat) {
        withAnimation { zoom = min(4, max(0.1, zoom * factor)) }
    }

    private func zoomToFit() {
        guard let b = project.contentBounds else { return }
        withAnimation {
            zoom = 1
            // Center the content bounds around the origin-ish.
            pan = CGSize(width: -CGFloat(b.minX + (b.maxX - b.minX) / 2) + 300,
                         height: -CGFloat(b.minY + (b.maxY - b.minY) / 2) + 250)
            panStart = pan
        }
    }

    // MARK: - Drop

    private func handleDrop(_ providers: [NSItemProvider], in size: CGSize) -> Bool {
        // Drop point isn't provided reliably; place near the current view center
        // in canvas coordinates, cascading each image slightly.
        let baseCanvasX = Double((size.width / 2 - pan.width) / zoom) - 120
        let baseCanvasY = Double((size.height / 2 - pan.height) / zoom) - 120
        var handled = false
        for (offset, provider) in providers.enumerated() {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url, url.isFileURL else { return }
                Task { @MainActor in
                    addImage(path: url.path,
                             at: CGPoint(x: baseCanvasX + Double(offset) * 28,
                                         y: baseCanvasY + Double(offset) * 28))
                }
            }
            handled = true
        }
        return handled
    }

    /// Composite the board into a PNG (transparent background, content-bounds
    /// sized) and save it via a panel.
    private func exportBoard() {
        guard let bounds = project.contentBounds else { return }
        let width = bounds.maxX - bounds.minX
        let height = bounds.maxY - bounds.minY
        guard width > 0, height > 0 else { return }

        let size = NSSize(width: width, height: height)
        let image = NSImage(size: size)
        image.lockFocus()
        NSColor.clear.set()
        NSBezierPath(rect: NSRect(origin: .zero, size: size)).fill()

        for item in project.itemsInDrawOrder {
            guard let itemImage = NSImage(contentsOfFile: item.imagePath) else { continue }
            // Canvas y is top-down; NSImage draws bottom-up, so flip.
            let x = item.x - bounds.minX
            let yTop = item.y - bounds.minY
            let drawRect = NSRect(x: x, y: height - yTop - item.height,
                                  width: item.width, height: item.height)

            let context = NSGraphicsContext.current
            context?.saveGraphicsState()
            let transform = NSAffineTransform()
            let cx = drawRect.midX, cy = drawRect.midY
            transform.translateX(by: cx, yBy: cy)
            transform.rotate(byDegrees: -item.rotation)   // screen CW -> draw CCW
            if item.flippedH { transform.scaleX(by: -1, yBy: 1) }
            transform.translateX(by: -cx, yBy: -cy)
            transform.concat()
            itemImage.draw(in: drawRect, from: .zero, operation: .sourceOver, fraction: 1)
            context?.restoreGraphicsState()
        }
        image.unlockFocus()

        guard let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "\(project.name).png"
        panel.prompt = "Export"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        try? png.write(to: url)
    }

    private func addImagesFromPicker() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.png, .jpeg, .image]
        panel.prompt = "Add to Canvas"
        guard panel.runModal() == .OK else { return }
        for (offset, url) in panel.urls.enumerated() {
            addImage(path: url.path, at: CGPoint(x: 60 + Double(offset) * 28, y: 60 + Double(offset) * 28))
        }
    }

    private func addImage(path: String, at point: CGPoint) {
        // Size the item to the image's aspect ratio (max 320 on the long side).
        var w = 240.0, h = 240.0
        if let image = NSImage(contentsOfFile: path), image.size.width > 0, image.size.height > 0 {
            let aspect = image.size.width / image.size.height
            if aspect >= 1 { w = 320; h = 320 / aspect } else { h = 320; w = 320 * aspect }
        }
        store.addItem(
            CanvasItem(imagePath: path, x: Double(point.x), y: Double(point.y), width: w, height: h),
            toCanvas: canvasId)
    }
}

// MARK: - Async image loader (canvas items)

private struct CanvasImageView: View {
    let path: String
    @State private var image: NSImage?

    var body: some View {
        ZStack {
            Rectangle().fill(.quaternary)
            if let image {
                Image(nsImage: image).resizable().aspectRatio(contentMode: .fill)
            } else {
                Image(systemName: "photo").foregroundStyle(.tertiary)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .task(id: path) {
            image = await Task.detached { NSImage(contentsOfFile: path) }.value
        }
    }
}
