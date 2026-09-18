// DirectorView.swift — the Director tab (WP3, docs/FDD-ltx-director-tab.md §4.4).
//
// HSplitView(sidebar | timeline canvas over the result player). A thin shell:
// all timeline state and rules live in DirectorDocumentModel (+ ZImage's
// DirectorMath/DirectorValidator); this view owns only the transient
// render/playback state and the AppKit panels.

import AVKit
import AppKit
import SwiftUI
import UniformTypeIdentifiers
import ZImage

struct DirectorView: View {
    @Bindable var engine: EngineService
    /// A rendered clip handed over by the gallery's "Open in Director": its
    /// sequence sidecar becomes the open timeline (WP13). Cleared once taken,
    /// so returning to the tab does not reopen it over later edits.
    var pendingSequenceClip: Binding<String?> = .constant(nil)

    @State private var model = DirectorDocumentModel(autosave: DirectorAutosaveStore())
    @State private var didAppear = false

    // Render state (MotionView.generate pattern).
    @State private var isGenerating = false
    @State private var isValidating = false
    @State private var statusMessage: String?
    @State private var errorMessage: String?
    @State private var resultURL: URL?
    @State private var player: AVPlayer?
    @State private var isPlaying = false

    static var projectContentType: UTType {
        UTType(filenameExtension: DirectorDocument.fileExtension) ?? .json
    }

    var body: some View {
        HSplitView {
            DirectorSidebar(
                engine: engine,
                model: model,
                isGenerating: isGenerating,
                isValidating: isValidating,
                onNew: newDocument,
                onOpen: openDocument,
                onSave: { saveDocument(saveAs: false) },
                onSaveAs: { saveDocument(saveAs: true) },
                onValidate: validate,
                onGenerate: generate
            )
            .frame(minWidth: 320, idealWidth: 360, maxWidth: 420)

            VStack(spacing: 0) {
                toolbar
                Divider()
                DirectorTimelineCanvas(
                    engine: engine,
                    model: model,
                    onTogglePlayback: togglePlayback
                )
                .frame(minHeight: 300)
                Divider()
                DirectorResultView(
                    player: player,
                    resultURL: resultURL,
                    plan: model.plan,
                    isGenerating: isGenerating,
                    statusMessage: statusMessage,
                    errorMessage: errorMessage
                )
                .frame(minHeight: 240)
            }
            .frame(minWidth: 520)
        }
        .navigationTitle(title)
        .background(shortcutButtons)
        .onChange(of: pendingSequenceClip.wrappedValue) { _, clip in
            takePendingSequence(clip)
        }
        .onAppear {
            guard !didAppear else { return }
            didAppear = true
            model.autoValidate = true
            if !model.restoreAutosaveIfAny() {
                let s = DesktopSettings.load()
                if let w = s.videoWidth, let h = s.videoHeight, w % 32 == 0, h % 32 == 0 {
                    model.updateSettings { $0.width = w; $0.height = h }
                    model.markSaved()
                }
            }
            model.validateLocally()
        }
    }

    private var title: String {
        let name = model.projectURL?.deletingPathExtension().lastPathComponent ?? "Untitled"
        return "Director — \(name)\(model.isDirty ? " •" : "")"
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        HStack(spacing: 10) {
            Button { model.undo() } label: { Image(systemName: "arrow.uturn.backward") }
                .disabled(!model.canUndo).help("Undo (⌘Z)")
            Button { model.redo() } label: { Image(systemName: "arrow.uturn.forward") }
                .disabled(!model.canRedo).help("Redo (⇧⌘Z)")
            Divider().frame(height: 16)
            Button { model.splitAtPlayhead() } label: { Label("Split", systemImage: "scissors") }
                .help("Split selected (or all) prompt segments and audio clips at the playhead (S)")
            Toggle("Snap", isOn: Binding(get: { model.snapToGrid }, set: { model.snapToGrid = $0 }))
                .toggleStyle(.checkbox)
                .help("Snap edits to the 8-frame latent grid")
            Spacer()
            Text(playheadReadout).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            Divider().frame(height: 16)
            Button { zoom(by: 1 / 1.25) } label: { Image(systemName: "minus.magnifyingglass") }.help("Zoom out (⌘-)")
            Button { zoom(by: 1.25) } label: { Image(systemName: "plus.magnifyingglass") }.help("Zoom in (⌘=)")
        }
        .buttonStyle(.borderless)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private var playheadReadout: String {
        String(format: "%d f · %.2f s", model.playheadFrame,
               DirectorMath.seconds(frames: model.playheadFrame, fps: model.fps))
    }

    private func zoom(by factor: Double) {
        model.zoom = min(max(model.zoom * factor, 0.1), 12)
    }

    /// Hidden buttons carrying the tab's keyboard shortcuts. Menu items with
    /// the same key equivalent (the system Edit ▸ Undo, File ▸ New Window)
    /// take precedence while enabled, which is why New is ⌥⌘N.
    private var shortcutButtons: some View {
        ZStack {
            Button("Undo") { model.undo() }.keyboardShortcut("z", modifiers: .command)
            Button("Redo") { model.redo() }.keyboardShortcut("z", modifiers: [.command, .shift])
            Button("Save") { saveDocument(saveAs: false) }.keyboardShortcut("s", modifiers: .command)
            Button("Save As") { saveDocument(saveAs: true) }.keyboardShortcut("s", modifiers: [.command, .shift])
            Button("Open") { openDocument() }.keyboardShortcut("o", modifiers: .command)
            Button("Open Sequence…") { openSequence() }
                .help("Open the timeline that made a rendered clip")
            Button("New") { newDocument() }.keyboardShortcut("n", modifiers: [.command, .option])
            Button("Zoom In") { zoom(by: 1.25) }.keyboardShortcut("=", modifiers: .command)
            Button("Zoom Out") { zoom(by: 1 / 1.25) }.keyboardShortcut("-", modifiers: .command)
        }
        .opacity(0)
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    // MARK: - Documents

    private func confirmDiscardIfDirty(_ action: String) -> Bool {
        guard model.isDirty else { return true }
        let alert = NSAlert()
        alert.messageText = "Discard unsaved Director changes?"
        alert.informativeText = "\(action) replaces the current timeline."
        alert.addButton(withTitle: "Discard")
        alert.addButton(withTitle: "Cancel")
        return alert.runModal() == .alertFirstButtonReturn
    }

    private func newDocument() {
        guard confirmDiscardIfDirty("New") else { return }
        model.new()
        errorMessage = nil
    }

    private func openDocument() {
        guard confirmDiscardIfDirty("Open") else { return }
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [Self.projectContentType]
        panel.prompt = "Open Timeline"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try model.load(url: url)
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Take a clip handed over by the gallery. Unlike `openSequence()` this
    /// arrives without a panel, so it must still refuse to discard unsaved
    /// work — and it clears the handover either way, so a declined prompt is
    /// not re-asked every time the tab reappears.
    private func takePendingSequence(_ clip: String?) {
        guard let clip, !clip.isEmpty else { return }
        pendingSequenceClip.wrappedValue = nil
        guard confirmDiscardIfDirty("Open") else { return }
        let url = URL(fileURLWithPath: clip)
        if let document = model.loadSequence(forMediaAt: url) {
            errorMessage = nil
            statusMessage = "Opened \(document.name) — \(document.chunks.count) chunk(s)"
        } else {
            errorMessage =
                "No sequence beside \(url.lastPathComponent) — it was rendered before sequences, or the sidecar moved."
        }
    }

    /// Open a rendered clip's Sequence: pick the mp4, get the timeline that
    /// made it (FDD §4.9.2, WP13).
    private func openSequence() {
        guard confirmDiscardIfDirty("Open") else { return }
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.mpeg4Movie, .json]
        panel.prompt = "Open Sequence"
        panel.message = "Choose a rendered clip; its timeline is stored beside it."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        if let document = model.loadSequence(forMediaAt: url) {
            errorMessage = nil
            statusMessage = "Opened \(document.name) — \(document.chunks.count) chunk(s), rendered \(document.createdAt.formatted(date: .abbreviated, time: .shortened))"
        } else {
            errorMessage =
                "No sequence beside \(url.lastPathComponent) — it was rendered before sequences, or the sidecar moved."
        }
    }

    private func saveDocument(saveAs: Bool) {
        if !saveAs, let url = model.projectURL {
            do { try model.save(to: url) } catch { errorMessage = error.localizedDescription }
            return
        }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [Self.projectContentType]
        panel.nameFieldStringValue = model.projectURL?.lastPathComponent ?? "Untitled.\(DirectorDocument.fileExtension)"
        let embed = NSButton(checkboxWithTitle: "Embed assets (keyframe images inside the file)", target: nil, action: nil)
        embed.state = .off
        panel.accessoryView = embed
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try model.save(to: url, embedAssets: embed.state == .on)
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Validate / Generate

    private func validate() {
        guard engine.connectionState.isConnected else {
            model.validateLocally()
            return
        }
        isValidating = true
        Task {
            do {
                let v = try await engine.validateDirectorTimeline(model.timeline)
                model.apply(issues: v.issues, plan: v.plan)
            } catch {
                // Server unreachable or refused: the in-process validator is
                // the same code, so fall back to it.
                model.validateLocally()
            }
            isValidating = false
        }
    }

    private func generate() {
        let outputDir = DesktopSettings.load().outputDirectory
        let outputPath = (outputDir as NSString).appendingPathComponent("director-\(Int(Date().timeIntervalSince1970)).mp4")

        isGenerating = true
        errorMessage = nil
        player?.pause()
        player = nil
        isPlaying = false
        resultURL = nil
        statusMessage = "Submitting Director timeline…"

        let timeline = model.timeline
        Task {
            do {
                let submission = try await engine.submitDirectorJob(timeline: timeline, outputPath: outputPath)
                if let plan = submission.plan { model.plan = plan }
                statusMessage = "Queued — \(submission.stageCount ?? plannedChunkCount) chunk(s)"
                let result = try await engine.pollVideoStatus(jobId: submission.jobId) { job in
                    if let plan = job.plan { model.plan = plan }
                    switch job.status {
                    case "queued":
                        statusMessage = "Queued…"
                    case "processing":
                        statusMessage = EngineService.directorStatusLine(
                            stageIndex: job.stageIndex, stageCount: job.stageCount, progressPercent: job.progressPercent)
                    default:
                        break
                    }
                }
                statusMessage = String(format: "Done — %d frames, %.1fs video in %.0fs",
                                       result.frameCount, result.durationSeconds, result.elapsedSeconds)
                let url = URL(fileURLWithPath: result.outputPath)
                player = AVPlayer(url: url)
                resultURL = url
            } catch {
                // A failed poll (engine restart: director jobs are non-durable)
                // throws immediately — surface it, never retry.
                errorMessage = error.localizedDescription
                statusMessage = nil
            }
            isGenerating = false
        }
    }

    private var plannedChunkCount: Int { model.plan?.chunks.count ?? 1 }

    private func togglePlayback() {
        guard let player else { return }
        if isPlaying { player.pause() } else { player.play() }
        isPlaying.toggle()
    }
}
