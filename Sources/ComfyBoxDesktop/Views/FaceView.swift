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
    @Bindable var faceSwap: FaceSwapService
    var ingestor: AssetIngestor?

    // Face swap
    @State private var swapSource: String = ""
    @State private var swapTarget: String = ""
    @State private var swapSourceThumb: NSImage?
    @State private var swapTargetThumb: NSImage?
    @State private var swapAllFaces = false
    @State private var swapResultURL: URL?
    @State private var swapStatus: String?
    @State private var swapIsError = false

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
            if !faceSwap.isInstalled {
                Label("Face-swap backend not installed (insightface + inswapper_128.onnx at ~/Projects/faceswap).",
                      systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
            }
            Text("Replace the face in a target image with a source face (insightface + inswapper). Runs locally.")
                .font(.caption).foregroundStyle(.secondary)

            HStack(spacing: 16) {
                swapImageWell("Source face", path: $swapSource, thumb: $swapSourceThumb)
                Image(systemName: "arrow.right").foregroundStyle(.secondary)
                swapImageWell("Target image", path: $swapTarget, thumb: $swapTargetThumb)
            }
            Toggle("Swap all faces in target", isOn: $swapAllFaces)
            if faceSwap.isRunning {
                HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Swapping…").font(.caption) }
            }
            Button { Task { await runSwap() } } label: {
                Label("Swap Face", systemImage: "arrow.triangle.2.circlepath").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(faceSwap.isRunning || !faceSwap.isInstalled || swapSource.isEmpty || swapTarget.isEmpty)

            if let swapStatus {
                Label(swapStatus, systemImage: swapIsError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                    .font(.caption).foregroundStyle(swapIsError ? .orange : .green)
            }
            if let swapResultURL {
                AsyncImage(url: swapResultURL) { phase in
                    if case .success(let img) = phase {
                        img.resizable().scaledToFit().frame(maxHeight: 320)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
        }
        .padding(12)
    }

    private func swapImageWell(_ title: String, path: Binding<String>, thumb: Binding<NSImage?>) -> some View {
        VStack(spacing: 4) {
            Button {
                let panel = NSOpenPanel()
                panel.canChooseFiles = true; panel.allowsMultipleSelection = false
                panel.allowedContentTypes = [.png, .jpeg, .image]
                if panel.runModal() == .OK, let url = panel.url {
                    path.wrappedValue = url.path
                    Task { let i = await Task.detached { NSImage(contentsOfFile: url.path) }.value
                        await MainActor.run { thumb.wrappedValue = i } }
                }
            } label: {
                Group {
                    if let t = thumb.wrappedValue {
                        Image(nsImage: t).resizable().aspectRatio(contentMode: .fill)
                    } else {
                        Image(systemName: "photo.badge.plus").font(.title2).foregroundStyle(.tertiary)
                    }
                }
                .frame(width: 96, height: 96)
                .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
            .buttonStyle(.plain)
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func runSwap() async {
        swapStatus = nil; swapResultURL = nil
        let dir = DesktopSettings.load().outputDirectory
        let output = (dir as NSString).appendingPathComponent("faceswap-\(Int(Date().timeIntervalSince1970)).png")
        do {
            _ = try await faceSwap.swap(source: swapSource, target: swapTarget, output: output, allFaces: swapAllFaces)
            swapResultURL = URL(fileURLWithPath: output)
            try? await ingestor?.ingestFile(at: output)
            swapStatus = "Swapped → \(URL(fileURLWithPath: output).lastPathComponent)"
            swapIsError = false
        } catch {
            swapStatus = error.localizedDescription; swapIsError = true
        }
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
