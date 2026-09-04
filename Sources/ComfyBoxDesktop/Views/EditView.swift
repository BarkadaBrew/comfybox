// EditView.swift — the Edit tab: canvas left, adjustment panel right
//
// Sliders call session.set for live preview and session.commit on
// gesture end so one drag is one undo step. Crop is edited on a
// normalized overlay; Local paints on the shared MaskCanvas.

import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct EditView: View {
    @Bindable var session: EditSession
    var ingestor: AssetIngestor?
    var onSendToInpaint: ((String, MaskStrokes?) -> Void)?

    enum Group: String, CaseIterable, Identifiable {
        case light = "Light", color = "Color", curves = "Curves", detail = "Detail",
             crop = "Crop & Rotate", local = "Local", subject = "Subject"
        var id: String { rawValue }
    }
    @State private var expanded: Set<Group> = [.light]
    @State private var brush: CGFloat = 40
    @State private var erase = false
    @State private var status: String?
    @State private var isSaving = false
    @State private var isError = false

    var body: some View {
        HSplitView {
            canvas.frame(minWidth: 480)
            panel.frame(minWidth: 320, maxWidth: 400)
        }
        .toolbar {
            ToolbarItemGroup {
                Button { session.undo() } label: { Label("Undo", systemImage: "arrow.uturn.backward") }
                    .disabled(!session.canUndo).keyboardShortcut("z", modifiers: .command)
                Button { session.redo() } label: { Label("Redo", systemImage: "arrow.uturn.forward") }
                    .disabled(!session.canRedo).keyboardShortcut("z", modifiers: [.command, .shift])
                Button { session.reset() } label: { Label("Reset", systemImage: "arrow.counterclockwise") }
                    .disabled(session.recipe.isIdentity)
                Button { } label: { Label("Before", systemImage: "eye") }
                    .simultaneousGesture(DragGesture(minimumDistance: 0)
                        .onChanged { _ in session.showOriginal = true }
                        .onEnded { _ in session.showOriginal = false })
                Button { Task { await save(thenInpaint: false) } } label: {
                    Label(isSaving ? "Saving…" : "Save", systemImage: "square.and.arrow.down")
                }.disabled(isSaving || session.sourceImage == nil).keyboardShortcut("s", modifiers: .command)
                Button { Task { await save(thenInpaint: true) } } label: {
                    Label("Save & Inpaint", systemImage: "paintbrush.pointed")
                }.disabled(isSaving || session.sourceImage == nil || onSendToInpaint == nil)
            }
        }
        .onAppear { session.suppressCropForPreview = expanded.contains(.crop) }
        .onChange(of: expanded) { _, newValue in session.suppressCropForPreview = newValue.contains(.crop) }
    }

    // MARK: - Canvas

    private var canvas: some View {
        GeometryReader { geo in
            ZStack {
                Color(nsColor: .underPageBackgroundColor)
                if let cg = session.showOriginal ? session.sourceImage : session.preview {
                    let rect = ImageFit.rect(imageSize: CGSize(width: cg.width, height: cg.height), in: geo.size)
                    Image(decorative: cg, scale: 1).resizable()
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                    if expanded.contains(.local), !session.showOriginal {
                        MaskCanvas(imageSize: rect.size, strokes: localMaskBinding, brushPoints: brush, erase: erase)
                            .position(x: rect.midX, y: rect.midY)
                    }
                    if expanded.contains(.crop), !session.showOriginal {
                        CropOverlay(rect: rect, crop: cropBinding, onCommit: { session.commit() })
                    }
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "slider.horizontal.3").font(.system(size: 44)).foregroundStyle(.tertiary)
                        Text(session.warning ?? "Open an image to edit").foregroundStyle(.secondary)
                    }
                }
                if session.isRendering {
                    ProgressView().controlSize(.small).padding(8)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                }
            }
        }
    }

    /// Local mask strokes; painting on an image with no layer creates one.
    private var localMaskBinding: Binding<MaskStrokes> {
        Binding(get: { session.recipe.local?.mask ?? MaskStrokes() },
                set: { new in
                    session.set { r in
                        if r.local == nil { r.local = EditLocalLayer() }
                        r.local?.mask = new
                    }
                    session.commit()
                })
    }

    /// Crop is stored against the pre-crop (post-rotation) frame; the overlay edits it in the preview rect.
    private var cropBinding: Binding<CGRect> {
        Binding(get: { session.recipe.geometry.crop ?? CGRect(x: 0, y: 0, width: 1, height: 1) },
                set: { new in session.set { $0.geometry.crop = new == CGRect(x: 0, y: 0, width: 1, height: 1) ? nil : new } })
    }

    // MARK: - Panel

    private var panel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                panelHeader
                adjustmentGroups

                if let status {
                    Label(status, systemImage: isError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                        .font(.caption).foregroundStyle(isError ? .orange : .green)
                }
            }
            .padding(14)
        }
    }

    private var panelHeader: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "slider.horizontal.3").foregroundStyle(.indigo)
                Text("Edit").font(.headline)
                Spacer()
                if session.isDirty { Text("unsaved").font(.caption2).foregroundStyle(.orange) }
            }
            if let w = session.warning {
                Label(w, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
            }
            Text(URL(fileURLWithPath: session.sourcePath).lastPathComponent)
                .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
        }
    }

    private var adjustmentGroups: some View {
        SwiftUI.Group {
            group(.light) {
                slider("Exposure", \.exposure, -5...5, step: 0.05)
                slider("Contrast", \.contrast, -1...1)
                slider("Highlights", \.highlights, -1...1)
                slider("Shadows", \.shadows, -1...1)
                slider("Whites", \.whites, -1...1)
                slider("Blacks", \.blacks, -1...1)
            }
            group(.color) {
                slider("Temperature", \.temperature, -1...1)
                slider("Tint", \.tint, -1...1)
                slider("Vibrance", \.vibrance, -1...1)
                slider("Saturation", \.saturation, -1...1)
            }
            group(.curves) {
                CurvesEditor(curves: Binding(get: { session.recipe.adjustments.curves },
                                             set: { c in session.set { $0.adjustments.curves = c } }),
                             onCommit: { session.commit() })
                    .frame(height: 220)
            }
            group(.detail) {
                slider("Sharpen", \.sharpen, 0...1)
                slider("Noise Reduction", \.noiseReduction, 0...1)
                slider("Vignette", \.vignette, 0...1)
            }
            group(.crop) { cropControls }
            group(.local) { localControls }
            group(.subject) { subjectControls }
        }
    }

    @ViewBuilder private func group<C: View>(_ g: Group, @ViewBuilder content: @escaping () -> C) -> some View {
        DisclosureGroup(isExpanded: Binding(get: { expanded.contains(g) },
                                            set: { isOn in
                                                if isOn {
                                                    // Crop and Local are mutually exclusive: local strokes are
                                                    // recorded against the uncropped frame, and the crop overlay
                                                    // would otherwise steal the Local layer's paint drags.
                                                    if g == .crop { expanded.remove(.local) }
                                                    if g == .local { expanded.remove(.crop) }
                                                    expanded.insert(g)
                                                } else {
                                                    expanded.remove(g)
                                                }
                                            })) {
            VStack(alignment: .leading, spacing: 8) { content() }.padding(.top, 6)
        } label: { Text(g.rawValue).font(.subheadline.weight(.medium)) }
    }

    private func slider(_ label: String, _ key: WritableKeyPath<EditAdjustments, Double>,
                        _ range: ClosedRange<Double>, step: Double = 0.01) -> some View {
        NumericSliderField(
            label: label,
            value: Binding(get: { session.recipe.adjustments[keyPath: key] },
                           set: { v in session.set { $0.adjustments[keyPath: key] = v } }),
            range: range, step: step, fractionDigits: 2,
            onEditingEnded: { session.commit() })
    }

    private func localSlider(_ label: String, _ key: WritableKeyPath<EditAdjustments, Double>,
                             _ range: ClosedRange<Double>) -> some View {
        NumericSliderField(
            label: label,
            value: Binding(get: { session.recipe.local?.adjustments[keyPath: key] ?? 0 },
                           set: { v in session.set { r in
                               if r.local == nil { r.local = EditLocalLayer() }
                               r.local?.adjustments[keyPath: key] = v } }),
            range: range, step: 0.01, fractionDigits: 2,
            onEditingEnded: { session.commit() })
    }

    private var cropControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Aspect").font(.caption).foregroundStyle(.secondary)
                Menu("Free") {
                    Button("Free") { }
                    Button("Original") { applyAspect(nil) }
                    Button("1:1") { applyAspect(1) }
                    Button("4:5") { applyAspect(4.0 / 5.0) }
                    Button("3:2") { applyAspect(3.0 / 2.0) }
                    Button("16:9") { applyAspect(16.0 / 9.0) }
                }.controlSize(.small)
                Spacer()
                Button("Clear Crop") { session.set { $0.geometry.crop = nil }; session.commit() }
                    .controlSize(.small).disabled(session.recipe.geometry.crop == nil)
            }
            NumericSliderField(label: "Straighten", value: Binding(get: { session.recipe.geometry.straightenDegrees },
                                                                   set: { v in session.set { $0.geometry.straightenDegrees = v } }),
                               range: -45...45, step: 0.1, fractionDigits: 1, onEditingEnded: { session.commit() })
            HStack {
                Button { session.set { $0.geometry.quarterTurns = ($0.geometry.quarterTurns + 3) % 4 }; session.commit() } label: { Image(systemName: "rotate.left") }
                Button { session.set { $0.geometry.quarterTurns = ($0.geometry.quarterTurns + 1) % 4 }; session.commit() } label: { Image(systemName: "rotate.right") }
                Button { session.set { $0.geometry.flipH.toggle() }; session.commit() } label: { Image(systemName: "arrow.left.and.right.righttriangle.left.righttriangle.right") }
                Button { session.set { $0.geometry.flipV.toggle() }; session.commit() } label: { Image(systemName: "arrow.up.and.down.righttriangle.up.righttriangle.down") }
            }.controlSize(.small)
            Text("Drag the corners on the image to crop.").font(.caption2).foregroundStyle(.tertiary)
        }
    }

    /// Center a crop of the given width:height ratio (nil = the source's own ratio) inside the current frame.
    ///
    /// Computed synchronously from `session.sourceImage`, never from the async `previewSize` — the
    /// preview may still be rendering (or reflect a stale generation) at the moment the aspect button
    /// is tapped, and this must be right the first time.
    private func applyAspect(_ ratio: Double?) {
        guard let source = session.sourceImage else { return }
        var w = Double(source.width), h = Double(source.height)
        if session.recipe.geometry.quarterTurns % 2 != 0 { swap(&w, &h) }
        if session.recipe.geometry.straightenDegrees != 0 {
            let angle = CGFloat(session.recipe.geometry.straightenDegrees) * .pi / 180
            let fit = EditRenderer.largestInscribedSize(width: CGFloat(w), height: CGFloat(h), angleRadians: angle)
            w = Double(fit.width); h = Double(fit.height)
        }
        guard w > 0, h > 0 else { return }
        let target = ratio ?? (w / h)
        let frameRatio = w / h
        var cw = 1.0, ch = 1.0
        if target > frameRatio { ch = frameRatio / target } else { cw = target / frameRatio }
        session.set { $0.geometry.crop = CGRect(x: (1 - cw) / 2, y: (1 - ch) / 2, width: cw, height: ch) }
        session.commit()
    }

    private var localControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "paintbrush")
                Slider(value: $brush, in: 8...200)
                Text("\(Int(brush))").font(.caption.monospacedDigit()).frame(width: 30)
            }
            HStack {
                Toggle("Erase", isOn: $erase).toggleStyle(.button).controlSize(.small)
                Button("Undo Stroke") { session.set { $0.local?.mask.undoLast() }; session.commit() }
                    .controlSize(.small).disabled(session.recipe.local?.mask.isEmpty ?? true)
                Button("Clear Mask") { session.set { $0.local = nil }; session.commit() }
                    .controlSize(.small).disabled(session.recipe.local == nil)
            }
            NumericSliderField(label: "Feather", value: Binding(get: { session.recipe.local?.feather ?? 0 },
                                                                set: { v in session.set { r in
                                                                    if r.local == nil { r.local = EditLocalLayer() }
                                                                    r.local?.feather = v } }),
                               range: 0...1, step: 0.01, fractionDigits: 2, onEditingEnded: { session.commit() })
            localSlider("Exposure", \.exposure, -5...5)
            localSlider("Contrast", \.contrast, -1...1)
            localSlider("Highlights", \.highlights, -1...1)
            localSlider("Shadows", \.shadows, -1...1)
            localSlider("Temperature", \.temperature, -1...1)
            localSlider("Tint", \.tint, -1...1)
            localSlider("Saturation", \.saturation, -1...1)
            localSlider("Sharpen", \.sharpen, 0...1)
            Text("Paint on the image; adjustments here apply inside the mask only.").font(.caption2).foregroundStyle(.tertiary)
        }
    }

    private var subjectControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button { Task { await session.requestSubjectMask() } } label: { Label("Find Subject", systemImage: "person.crop.rectangle") }
                    .controlSize(.small).disabled(session.sourceImage == nil)
                if let s = session.subjectStatus { Text(s).font(.caption).foregroundStyle(.orange) }
            }
            Toggle("Remove Background", isOn: Binding(get: { session.recipe.subject.removeBackground },
                                                     set: { v in session.set { $0.subject.removeBackground = v }; session.commit() }))
                .disabled(session.subjectMask == nil && !session.recipe.subject.removeBackground)
            Toggle("Invert (keep background)", isOn: Binding(get: { session.recipe.subject.invert },
                                                            set: { v in session.set { $0.subject.invert = v }; session.commit() }))
                .disabled(session.subjectMask == nil || !session.recipe.subject.removeBackground)
            Text("Saves a transparent PNG when Remove Background is on.").font(.caption2).foregroundStyle(.tertiary)
        }
    }

    // MARK: - Save

    private func save(thenInpaint: Bool) async {
        isSaving = true; status = nil; isError = false
        defer { isSaving = false }
        do {
            let dir = DesktopSettings.load().outputDirectory
            let path = try await session.export(outputDirectory: dir, ingestor: ingestor)
            status = "Saved → \(URL(fileURLWithPath: path).lastPathComponent)"
            if thenInpaint {
                let mask = session.recipe.local?.mask
                onSendToInpaint?(path, (mask?.isEmpty ?? true) ? nil : mask)
            }
        } catch {
            status = error.localizedDescription; isError = true
        }
    }
}

// MARK: - Crop overlay

/// Draggable crop rectangle drawn over the fitted preview. `crop` is normalized to the preview frame.
struct CropOverlay: View {
    let rect: CGRect
    @Binding var crop: CGRect
    var onCommit: () -> Void
    @State private var dragStart: CGRect?

    private enum Handle: CaseIterable { case tl, tr, bl, br, move }

    var body: some View {
        let c = CGRect(x: rect.minX + crop.minX * rect.width, y: rect.minY + crop.minY * rect.height,
                       width: crop.width * rect.width, height: crop.height * rect.height)
        ZStack {
            Path { p in p.addRect(rect); p.addRect(c) }
                .fill(Color.black.opacity(0.45), style: FillStyle(eoFill: true))
                .allowsHitTesting(false)
            Rectangle().stroke(Color.white, lineWidth: 1).frame(width: c.width, height: c.height)
                .position(x: c.midX, y: c.midY)
                .contentShape(Rectangle())
                .gesture(drag(.move, c))
            ForEach(Array([Handle.tl, .tr, .bl, .br].enumerated()), id: \.offset) { _, h in
                Circle().fill(Color.white).frame(width: 10, height: 10)
                    .position(handlePoint(h, c))
                    .gesture(drag(h, c))
            }
        }
    }

    private func handlePoint(_ h: Handle, _ c: CGRect) -> CGPoint {
        switch h {
        case .tl: return CGPoint(x: c.minX, y: c.minY)
        case .tr: return CGPoint(x: c.maxX, y: c.minY)
        case .bl: return CGPoint(x: c.minX, y: c.maxY)
        case .br: return CGPoint(x: c.maxX, y: c.maxY)
        case .move: return CGPoint(x: c.midX, y: c.midY)
        }
    }

    private func drag(_ h: Handle, _ c: CGRect) -> some Gesture {
        DragGesture(minimumDistance: 1)
            .onChanged { v in
                if dragStart == nil { dragStart = crop }
                guard let start = dragStart else { return }
                let dx = v.translation.width / rect.width, dy = v.translation.height / rect.height
                var n = start
                switch h {
                case .move:
                    n.origin.x = min(max(start.minX + dx, 0), 1 - start.width)
                    n.origin.y = min(max(start.minY + dy, 0), 1 - start.height)
                case .tl: n = CGRect(x: start.minX + dx, y: start.minY + dy, width: start.width - dx, height: start.height - dy)
                case .tr: n = CGRect(x: start.minX, y: start.minY + dy, width: start.width + dx, height: start.height - dy)
                case .bl: n = CGRect(x: start.minX + dx, y: start.minY, width: start.width - dx, height: start.height + dy)
                case .br: n = CGRect(x: start.minX, y: start.minY, width: start.width + dx, height: start.height + dy)
                }
                n = n.standardized.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
                if n.width >= 0.05, n.height >= 0.05 { crop = n }
            }
            .onEnded { _ in dragStart = nil; onCommit() }
    }
}

// MARK: - Tab wrapper

/// One request to open an image in the Edit tab. Carries the source `DAMAsset`
/// (when the request originated from the DAM) so derived sidecars can record
/// the source asset id and generation fields — a path-only open (e.g. the
/// Open… panel) carries `asset: nil`.
struct EditRequest: Equatable {
    let path: String
    let asset: DAMAsset?
}

struct EditTab: View {
    var ingestor: AssetIngestor?
    @Binding var pending: EditRequest?
    var onSendToInpaint: ((String, MaskStrokes?) -> Void)?

    @State private var session: EditSession?

    var body: some View {
        Group {
            if let session {
                EditView(session: session, ingestor: ingestor, onSendToInpaint: onSendToInpaint)
                    // A new session in the same view must get fresh view state (expanded
                    // groups, crop-suppression sync) rather than inheriting the prior
                    // session's — `onAppear` then re-runs and re-syncs `suppressCropForPreview`.
                    .id(ObjectIdentifier(session))
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "slider.horizontal.3").font(.system(size: 44)).foregroundStyle(.tertiary)
                    Text("Open an image to edit").foregroundStyle(.secondary)
                    Button { pickImage() } label: { Label("Open…", systemImage: "photo.badge.plus") }
                        .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("Edit")
        .toolbar { ToolbarItem(placement: .navigation) { Button { pickImage() } label: { Label("Open", systemImage: "folder") } } }
        .onAppear { consumePending() }
        .onChange(of: pending) { _, _ in consumePending() }
    }

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true; panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.png, .jpeg, .tiff]
        if panel.runModal() == .OK, let url = panel.url { open(url.path, asset: nil) }
    }

    private func consumePending() {
        guard let req = pending, !req.path.isEmpty else { return }
        pending = nil
        open(req.path, asset: req.asset)
    }

    private func open(_ path: String, asset: DAMAsset?) {
        let s = EditSession(sourcePath: path, sourceAsset: asset)
        session = s
        Task { await s.load() }
    }
}
