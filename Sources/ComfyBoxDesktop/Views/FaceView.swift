// FaceView.swift — Face identity & swap
//
// Identity generation works today via mflux's face-consistency model
// (dev-krea) with a reference-face anchor and the built-in "identity" LoRA
// style — a PuLID/FaceID-equivalent path. True embedding-injection PuLID and
// post-hoc face swap need backends (tickets #51 / #52) and are shown as
// clearly-pending scaffolds.

import SwiftUI
import AppKit

struct FaceView: View {
    @Bindable var mflux: MfluxService
    var ingestor: AssetIngestor?

    enum Mode: String, CaseIterable, Identifiable {
        case identity = "Identity Generate", swap = "Face Swap"
        var id: String { rawValue }
    }
    @State private var mode: Mode = .identity

    // Identity generation
    @State private var faceImage: String = ""
    @State private var faceThumb: NSImage?
    @State private var prompt: String = ""
    @State private var model: MfluxModel = .devKrea
    @State private var identityStrength: Double = 0.65
    @State private var useIdentityLora = true
    @State private var seedText: String = ""
    @State private var quantize: Int = 0
    @State private var resultURL: URL?
    @State private var status: String?
    @State private var isError = false

    private static let quantizeOptions = [0, 4, 6, 8]

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            Picker("", selection: $mode) {
                ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented).labelsHidden()
            .padding(.horizontal, 12).padding(.top, 8)

            ScrollView {
                switch mode {
                case .identity: identitySection
                case .swap: swapSection
                }
            }
        }
        .navigationTitle("Face")
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "person.crop.circle.badge.checkmark").foregroundStyle(.teal)
            Text("Face Identity").font(.headline)
            Spacer()
            if mflux.isRunning {
                ProgressView().controlSize(.small)
                Button("Cancel", role: .destructive) { mflux.cancel() }.controlSize(.small)
            }
        }
        .padding(12)
    }

    // MARK: - Identity generation (mflux dev-krea)

    private var identitySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            if !mflux.isInstalled {
                Label("Identity generation uses mflux — install its venv (see the mflux tab).",
                      systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
            }
            Text("Preserves a person's identity from a reference face using mflux's face-consistency model (dev-krea) + the identity LoRA — a PuLID/FaceID-style result.")
                .font(.caption).foregroundStyle(.secondary)

            field("Reference face (required)") {
                HStack(spacing: 10) {
                    Group {
                        if let thumb = faceThumb {
                            Image(nsImage: thumb).resizable().aspectRatio(contentMode: .fill)
                        } else { Image(systemName: "person.crop.square").foregroundStyle(.tertiary) }
                    }
                    .frame(width: 64, height: 64).clipShape(RoundedRectangle(cornerRadius: 6))
                    if faceImage.isEmpty {
                        Button("Choose Face…") { pickFace() }
                    } else {
                        Text((faceImage as NSString).lastPathComponent).font(.caption).lineLimit(1).truncationMode(.middle)
                        Button("Change") { pickFace() }.controlSize(.small)
                    }
                    Spacer()
                }
            }

            field("Prompt (scene / styling)") {
                TextEditor(text: $prompt).frame(minHeight: 56)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(nsColor: .separatorColor)))
            }

            Picker("Model", selection: $model) {
                ForEach([MfluxModel.devKrea, .kreaDev, .dev, .zImageTurbo], id: \.self) { Text($0.label).tag($0) }
            }
            NumericSliderField(label: "Identity strength", value: $identityStrength, range: 0...1, step: 0.05, fractionDigits: 2)
            Text("Higher keeps the reference face more faithfully; lower lets the prompt restyle more.")
                .font(.caption2).foregroundStyle(.tertiary)
            Toggle("Use identity LoRA style", isOn: $useIdentityLora)
            HStack {
                field("Seed (empty = random)") { TextField("Random", text: $seedText).textFieldStyle(.roundedBorder) }
                field("Quantize") {
                    Picker("", selection: $quantize) {
                        ForEach(Self.quantizeOptions, id: \.self) { Text($0 == 0 ? "None" : "\($0)-bit").tag($0) }
                    }.labelsHidden().fixedSize()
                }
            }

            Button { Task { await runIdentity() } } label: {
                Label("Generate with identity", systemImage: "person.crop.circle.badge.checkmark")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(mflux.isRunning || !mflux.isInstalled || faceImage.isEmpty
                      || prompt.trimmingCharacters(in: .whitespaces).isEmpty)

            if let status {
                Label(status, systemImage: isError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                    .font(.caption).foregroundStyle(isError ? .orange : .green)
            }
            if let resultURL {
                AsyncImage(url: resultURL) { phase in
                    if case .success(let img) = phase {
                        img.resizable().scaledToFit().frame(maxHeight: 320)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
            consoleTail
        }
        .padding(12)
    }

    // MARK: - Face swap (pending backend)

    private var swapSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Image(systemName: "arrow.triangle.2.circlepath.circle").font(.largeTitle).foregroundStyle(.secondary)
            Text("Face swap — backend pending").font(.headline)
            Text("Post-hoc face swap (replace a face in a target image with a reference) needs an inswapper/insightface backend, which isn't installed yet. Tracked as a ticket. For identity-preserving *generation*, use Identity Generate above.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(24)
    }

    private var consoleTail: some View {
        Group {
            if !mflux.consoleLog.isEmpty {
                ScrollView {
                    Text(mflux.consoleLog).font(.system(.caption2, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled).padding(6)
                }
                .frame(height: 90)
                .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    @ViewBuilder private func field<C: View>(_ title: String, @ViewBuilder content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            content()
        }
    }

    private func pickFace() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true; panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.png, .jpeg, .image]
        panel.prompt = "Use Face"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        faceImage = url.path
        Task {
            let img = await Task.detached { NSImage(contentsOfFile: url.path) }.value
            await MainActor.run { faceThumb = img }
        }
    }

    private func runIdentity() async {
        status = nil; resultURL = nil
        let dir = DesktopSettings.load().outputDirectory
        let output = (dir as NSString).appendingPathComponent("face-\(Int(Date().timeIntervalSince1970)).png")
        let opts = MfluxService.GenerateOptions(
            model: model.rawValue, prompt: prompt,
            steps: 25, seed: UInt64(seedText),
            quantize: quantize == 0 ? nil : quantize,
            imagePath: faceImage, imageStrength: identityStrength,
            loraStyle: useIdentityLora ? "identity" : nil,
            output: output)
        do {
            try await mflux.run("mflux-generate", args: MfluxService.generateArgs(opts))
            resultURL = URL(fileURLWithPath: output)
            try? await ingestor?.ingestFile(at: output)
            status = "Generated \(URL(fileURLWithPath: output).lastPathComponent) with identity anchor."
            isError = false
        } catch {
            status = error.localizedDescription; isError = true
        }
    }
}
