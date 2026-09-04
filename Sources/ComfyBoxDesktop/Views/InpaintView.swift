// InpaintView.swift — Paint a mask, regenerate that region
//
// Load an image, paint over the area to change (white = inpaint), enter a
// prompt, and the server regenerates just the masked region. Brush strokes are
// captured in the displayed image rect and rasterized to a full-resolution
// black/white mask PNG at generate time.

import SwiftUI
import AppKit

struct InpaintView: View {
    @Bindable var engine: EngineService
    var ingestor: AssetIngestor?
    /// Image queued from the Gallery/Canvas "Inpaint" action; consumed once.
    @Binding var pendingImage: String?
    /// Strokes queued from the Edit tab's "Send to Inpaint"; consumed with `pendingImage`.
    @Binding var pendingMask: MaskStrokes?

    @State private var basePath: String?
    @State private var baseImage: NSImage?
    @State private var pixelSize: CGSize = .zero      // full-res image size

    @State private var strokes = MaskStrokes()
    @State private var brush: CGFloat = 40
    @State private var erase = false

    @State private var prompt = ""
    @State private var denoise: Double = 0.85
    @State private var steps: Double = 9
    @State private var guidance: Double = 3.5
    @State private var seedText = ""

    @State private var isRunning = false
    @State private var status: String?
    @State private var isError = false
    @State private var resultURL: URL?

    var body: some View {
        HSplitView {
            controls.frame(minWidth: 300, maxWidth: 380)
            canvas.frame(minWidth: 420)
        }
        .navigationTitle("Inpaint")
        .onAppear { consumePending() }
        .onChange(of: pendingImage) { _, _ in consumePending() }
        .onChange(of: pendingMask) { _, _ in consumePending() }
    }

    // MARK: - Controls

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    Image(systemName: "paintbrush.pointed").foregroundStyle(.indigo)
                    Text("Inpaint").font(.headline)
                    Spacer()
                    if !engine.connectionState.isConnected {
                        Text("offline").font(.caption2).foregroundStyle(.orange)
                    }
                }

                if basePath == nil {
                    Button { pickImage() } label: {
                        Label("Choose Image…", systemImage: "photo.badge.plus")
                    }.buttonStyle(.borderedProminent)
                } else {
                    Button { pickImage() } label: { Label("Change Image", systemImage: "photo") }
                        .controlSize(.small)
                }

                field("Prompt (what to paint in the masked area)") {
                    TextEditor(text: $prompt).frame(minHeight: 60)
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(nsColor: .separatorColor)))
                }

                // Brush controls
                VStack(alignment: .leading, spacing: 6) {
                    Text("Brush").font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Image(systemName: "paintbrush")
                        Slider(value: $brush, in: 8...140)
                        Text("\(Int(brush))").font(.caption.monospacedDigit()).frame(width: 30)
                    }
                    HStack {
                        Toggle("Erase", isOn: $erase).toggleStyle(.button).controlSize(.small)
                        Button("Undo") { strokes.undoLast() }.controlSize(.small).disabled(strokes.isEmpty)
                        Button("Clear") { strokes.clear() }
                            .controlSize(.small).disabled(strokes.isEmpty)
                    }
                }

                NumericSliderField(label: "Strength (denoise)", value: $denoise, range: 0...1, step: 0.05, fractionDigits: 2)
                NumericSliderField(label: "Steps", value: $steps, range: 1...50, step: 1)
                NumericSliderField(label: "Guidance", value: $guidance, range: 0...20, step: 0.5, fractionDigits: 1)
                field("Seed (empty = random)") { TextField("Random", text: $seedText).textFieldStyle(.roundedBorder) }

                Button { Task { await runInpaint() } } label: {
                    HStack {
                        if isRunning { ProgressView().controlSize(.small) }
                        Image(systemName: "wand.and.stars")
                        Text(isRunning ? "Inpainting…" : "Inpaint")
                    }.frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isRunning || !engine.connectionState.isConnected || basePath == nil
                          || strokes.isEmpty || prompt.trimmingCharacters(in: .whitespaces).isEmpty)

                if let status {
                    Label(status, systemImage: isError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                        .font(.caption).foregroundStyle(isError ? .orange : .green)
                }
                Text("Paint over the region to replace, describe it, and Inpaint. Higher strength changes more.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
            .padding(16)
        }
    }

    // MARK: - Canvas (image + mask painting)

    private var canvas: some View {
        GeometryReader { geo in
            ZStack {
                Color(nsColor: .controlBackgroundColor)
                if let img = baseImage {
                    let rect = ImageFit.rect(imageSize: img.size, in: geo.size)
                    Image(nsImage: img)
                        .resizable().frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                    MaskCanvas(imageSize: rect.size, strokes: $strokes, brushPoints: brush, erase: erase)
                        .position(x: rect.midX, y: rect.midY)
                    if let resultURL {
                        // Show the result over the editor when available.
                        AsyncImage(url: resultURL) { phase in
                            if case .success(let r) = phase {
                                r.resizable().scaledToFit()
                                    .frame(width: rect.width, height: rect.height)
                                    .position(x: rect.midX, y: rect.midY)
                                    .overlay(alignment: .topTrailing) {
                                        Button("Edit result") { adoptResult() }.controlSize(.small).padding(6)
                                    }
                            }
                        }
                    }
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "paintbrush.pointed").font(.system(size: 44)).foregroundStyle(.tertiary)
                        Text("Choose an image, then paint the area to change").foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    // MARK: - Helpers

    @ViewBuilder private func field<C: View>(_ title: String, @ViewBuilder content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            content()
        }
    }

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true; panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.png, .jpeg, .image]
        if panel.runModal() == .OK, let url = panel.url { load(url.path) }
    }

    private func consumePending() {
        guard let p = pendingImage, !p.isEmpty else { return }
        let mask = pendingMask
        pendingImage = nil; pendingMask = nil
        load(p, initialStrokes: mask)
    }

    private func load(_ path: String, initialStrokes: MaskStrokes? = nil) {
        basePath = path
        strokes = initialStrokes ?? MaskStrokes(); resultURL = nil; status = nil
        Task {
            let img = await Task.detached { NSImage(contentsOfFile: path) }.value
            await MainActor.run {
                baseImage = img
                if let rep = img?.representations.first {
                    pixelSize = CGSize(width: rep.pixelsWide, height: rep.pixelsHigh)
                } else { pixelSize = img?.size ?? .zero }
            }
        }
    }

    private func adoptResult() {
        guard let url = resultURL else { return }
        load(url.path)
    }

    /// Rasterize the normalized strokes to a full-resolution black/white mask PNG
    /// (white = inpaint region). Erase strokes paint black.
    private func maskPNG() -> Data? { MaskRasterizer.pngData(strokes, size: pixelSize) }

    private func runInpaint() async {
        guard baseImage != nil,
              let baseData = NSBitmapImageRep(data: baseImage?.tiffRepresentation ?? Data())?
                .representation(using: .png, properties: [:]) else {
            status = "Couldn't read the base image."; isError = true; return
        }
        guard let mask = maskPNG() else { status = "Paint a mask first."; isError = true; return }
        isRunning = true; status = nil; resultURL = nil
        defer { isRunning = false }
        let dir = DesktopSettings.load().outputDirectory
        let out = (dir as NSString).appendingPathComponent("inpaint-\(Int(Date().timeIntervalSince1970)).png")
        do {
            let result = try await engine.inpaint(
                baseImagePNG: baseData, maskPNG: mask, prompt: prompt,
                width: Int(pixelSize.width), height: Int(pixelSize.height),
                steps: Int(steps), guidance: Float(guidance), denoise: Float(denoise),
                seed: UInt64(seedText) ?? 0, outputPath: out)
            resultURL = URL(fileURLWithPath: result)
            try? await ingestor?.ingestFile(at: result)
            status = "Inpainted → \(URL(fileURLWithPath: result).lastPathComponent)"
            isError = false
        } catch {
            status = error.localizedDescription; isError = true
        }
    }
}
