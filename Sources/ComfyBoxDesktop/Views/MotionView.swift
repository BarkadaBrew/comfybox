// MotionView.swift — LTX-2 video generation (text-to-video & image-to-video)
//
// Calls /v1/video/generate (local LTX-2 backend). A reference image switches
// it to image-to-video. Generation is long (minutes) and synchronous on the
// local backend, so the UI shows a busy state and then an inline AVKit
// preview of the resulting MP4.

import SwiftUI
import AVKit
import AppKit

struct MotionView: View {
    @Bindable var engine: EngineService
    /// Reference image queued by Gallery/Canvas "Animate"; consumed once.
    @Binding var pendingMotionReference: String?

    @State private var prompt: String = ""
    @State private var referencePath: String?
    @State private var referenceThumb: NSImage?
    @State private var resolution: VideoResolution = .landscape
    @State private var frames: Int = 97
    @State private var steps: Double = 8
    @State private var didApplyDefaults = false
    @State private var seedText: String = ""
    @State private var strength: Double = 1.0
    @State private var extendSeconds: Double = 0
    @State private var selectedLoras: [LoRASelection] = []

    @State private var isGenerating = false
    @State private var statusMessage: String?
    @State private var errorMessage: String?
    @State private var resultURL: URL?

    enum VideoResolution: String, CaseIterable, Identifiable {
        case landscape = "704 × 448 (16:10)"
        case portrait = "448 × 704 (10:16)"
        case square = "512 × 512"
        case wide = "768 × 448 (16:9-ish)"
        var id: String { rawValue }
        var size: (Int, Int) {
            switch self {
            case .landscape: return (704, 448)
            case .portrait: return (448, 704)
            case .square: return (512, 512)
            case .wide: return (768, 448)
            }
        }
    }

    /// LTX-2 requires 1 + 8k frames; offer a few durations at 24fps.
    private static let frameOptions = [25, 49, 97, 121]

    var body: some View {
        HSplitView {
            controls.frame(minWidth: 320, maxWidth: 420)
            preview.frame(minWidth: 420)
        }
        .navigationTitle("Motion")
        .onAppear { applyDefaults(); consumePendingReference() }
        .onChange(of: pendingMotionReference) { _, _ in consumePendingReference() }
    }

    // MARK: - Controls

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 8) {
                    Image(systemName: "film.stack").foregroundStyle(.purple)
                    Text("LTX-2 Video").font(.headline)
                    Spacer()
                    if !engine.connectionState.isConnected {
                        Text("offline").font(.caption2).foregroundStyle(.orange)
                    }
                }

                labeled("Prompt") {
                    TextEditor(text: $prompt)
                        .font(.body).scrollContentBackground(.hidden)
                        .frame(minHeight: 80)
                        .padding(6)
                        .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(nsColor: .separatorColor), lineWidth: 1))
                }

                referenceControl

                labeled("Resolution") {
                    Picker("", selection: $resolution) {
                        ForEach(VideoResolution.allCases) { Text($0.rawValue).tag($0) }
                    }.labelsHidden()
                }

                HStack(spacing: 12) {
                    labeled("Frames") {
                        Picker("", selection: $frames) {
                            ForEach(Self.frameOptions, id: \.self) { f in
                                Text("\(f)  (\(String(format: "%.1fs", Double(f) / 24.0)))").tag(f)
                            }
                        }.labelsHidden()
                    }
                }

                NumericSliderField(label: "Steps", value: $steps, range: 1...30, step: 1)
                if referencePath != nil {
                    NumericSliderField(label: "Strength", value: $strength, range: 0...1, step: 0.05, fractionDigits: 2)
                    NumericSliderField(label: "Extend (s)", value: $extendSeconds, range: 0...12, step: 1)
                }

                labeled("Seed (empty = random)") {
                    TextField("Random", text: $seedText).textFieldStyle(.roundedBorder)
                }

                labeled("LoRAs") {
                    LoRAPicker(engine: engine, selectedLoras: $selectedLoras, familyOverride: "ltx")
                        .frame(minHeight: 180, maxHeight: 260)
                }

                Button(action: generate) {
                    HStack {
                        if isGenerating { ProgressView().controlSize(.small).padding(.trailing, 2) }
                        Image(systemName: "film")
                        Text(isGenerating ? "Generating…" : "Generate Video")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(isGenerating || !engine.connectionState.isConnected
                          || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                if let statusMessage {
                    Text(statusMessage).font(.caption).foregroundStyle(.secondary)
                }
                if let errorMessage {
                    Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }
                Text("Local video needs the server started with --ltx2-weights + --ltx2-gemma. Generation takes several minutes.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
            .padding(16)
        }
    }

    private var referenceControl: some View {
        labeled("Reference image (image-to-video, optional)") {
            if let path = referencePath {
                HStack(spacing: 10) {
                    Group {
                        if let thumb = referenceThumb {
                            Image(nsImage: thumb).resizable().aspectRatio(contentMode: .fill)
                        } else { Image(systemName: "photo").foregroundStyle(.tertiary) }
                    }
                    .frame(width: 64, height: 64).clipShape(RoundedRectangle(cornerRadius: 6))
                    Text((path as NSString).lastPathComponent).font(.caption).lineLimit(1).truncationMode(.middle)
                    Spacer()
                    Button("Remove", role: .destructive) { referencePath = nil; referenceThumb = nil }
                        .controlSize(.small)
                }
            } else {
                Button {
                    chooseReference()
                } label: { Label("Choose Image…", systemImage: "photo.badge.plus") }
                    .controlSize(.small)
            }
        }
    }

    // MARK: - Preview

    private var preview: some View {
        ZStack {
            Color(nsColor: .controlBackgroundColor)
            if let url = resultURL {
                VStack(spacing: 8) {
                    VideoPlayer(player: AVPlayer(url: url))
                        .frame(minHeight: 300)
                    HStack {
                        Button { NSWorkspace.shared.selectFile(url.path, inFileViewerRootedAtPath: "") } label: {
                            Label("Reveal", systemImage: "magnifyingglass")
                        }
                        Button { NSWorkspace.shared.open(url) } label: {
                            Label("Open", systemImage: "play.rectangle")
                        }
                        Spacer()
                    }
                    .padding(.horizontal, 12).padding(.bottom, 8)
                }
            } else if isGenerating {
                VStack(spacing: 10) {
                    ProgressView().controlSize(.large)
                    Text(statusMessage ?? "Generating video…").foregroundStyle(.secondary)
                    Text("LTX-2 renders locally — this can take several minutes.")
                        .font(.caption).foregroundStyle(.tertiary)
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "film").font(.system(size: 48)).foregroundStyle(.tertiary)
                    Text("Your video will appear here").foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Actions

    private func labeled<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            content()
        }
    }

    private func chooseReference() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true; panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.png, .jpeg, .image]
        panel.prompt = "Use as Reference"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        setReference(url.path)
    }

    private func setReference(_ path: String) {
        referencePath = path
        Task {
            let img = await Task.detached { NSImage(contentsOfFile: path) }.value
            await MainActor.run { referenceThumb = img }
        }
    }

    /// Prefill from Settings → Motion (once), matching a stored resolution to a
    /// preset when possible.
    private func applyDefaults() {
        guard !didApplyDefaults else { return }
        didApplyDefaults = true
        let s = DesktopSettings.load()
        if let f = s.videoFrames, Self.frameOptions.contains(f) { frames = f }
        if let st = s.videoSteps { steps = Double(st) }
        if let w = s.videoWidth, let h = s.videoHeight,
           let match = VideoResolution.allCases.first(where: { $0.size == (w, h) }) {
            resolution = match
        }
    }

    private func consumePendingReference() {
        guard let path = pendingMotionReference, !path.isEmpty else { return }
        pendingMotionReference = nil
        setReference(path)
    }

    private func generate() {
        let (w, h) = resolution.size
        let seed = UInt64(seedText) ?? 0
        let outputDir = DesktopSettings.load().outputDirectory
        let outputPath = (outputDir as NSString).appendingPathComponent("ltx2-\(Int(Date().timeIntervalSince1970)).mp4")

        isGenerating = true
        errorMessage = nil
        resultURL = nil
        statusMessage = "Loading LTX-2 and generating \(frames) frames…"

        let request = EngineService.VideoRequest(
            prompt: prompt,
            initImagePath: referencePath,
            width: w, height: h, frames: frames,
            steps: Int(steps), seed: seed, strength: Float(strength),
            extendToSeconds: Float(extendSeconds),
            loras: selectedLoras,
            outputPath: outputPath
        )

        Task {
            do {
                let result = try await engine.generateVideo(request)
                statusMessage = String(format: "Done — %d frames, %.1fs video in %.0fs",
                                       result.frameCount, result.durationSeconds, result.elapsedSeconds)
                resultURL = URL(fileURLWithPath: result.outputPath)
            } catch {
                errorMessage = error.localizedDescription
                statusMessage = nil
            }
            isGenerating = false
        }
    }
}
