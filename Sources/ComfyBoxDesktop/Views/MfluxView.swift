// MfluxView.swift — Desktop-only mflux frontend
//
// Drives the local mflux venv for generation, LoRA training, model
// saving/quantizing, and updates. This is a UI convenience only — the
// ComfyBox server never uses mflux, and API/MCP clients don't see it.

import SwiftUI
import AppKit

struct MfluxView: View {
    @Bindable var mflux: MfluxService
    /// Ingest generated images into the gallery.
    var ingestor: AssetIngestor?

    enum Section: String, CaseIterable, Identifiable {
        case generate = "Generate", train = "Train", tools = "Tools"
        var id: String { rawValue }
    }
    @State private var section: Section = .generate

    // Generate
    @State private var model: MfluxModel = .dev
    @State private var prompt = ""
    @State private var negativePrompt = ""
    @State private var width = 1024
    @State private var height = 1024
    @State private var steps = 25
    @State private var seedText = ""
    @State private var quantize = 0          // 0 = none
    @State private var loraPaths = ""        // comma-separated
    @State private var loraScales = ""       // comma-separated
    @State private var imagePath = ""
    @State private var lowRam = false
    @State private var resultURL: URL?

    // Train
    @State private var trainModel: MfluxModel = .dev
    @State private var trainConfigPath = ""
    @State private var trainResumePath = ""
    @State private var trainDryRun = false
    @State private var trainQuantize = 0

    // Tools (save/quantize/bake)
    @State private var saveModel = "z-image-turbo"
    @State private var savePath = ""
    @State private var saveQuantize = 4
    @State private var saveLoraPaths = ""
    @State private var saveLoraScales = ""

    @State private var status: String?
    @State private var isError = false

    private static let quantizeOptions = [0, 3, 4, 5, 6, 8]

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if !mflux.isInstalled {
                notInstalled
            } else {
                Picker("", selection: $section) {
                    ForEach(Section.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .padding(.horizontal, 12).padding(.top, 8)

                ScrollView {
                    switch section {
                    case .generate: generateSection
                    case .train: trainSection
                    case .tools: toolsSection
                    }
                }
                console
            }
        }
        .navigationTitle("mflux")
        .task { await mflux.refreshVersion() }
    }

    // MARK: - Header / status

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "cube.transparent").foregroundStyle(.orange)
            Text("mflux").font(.headline)
            if let v = mflux.version {
                Text("v\(v)").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if mflux.isRunning {
                ProgressView().controlSize(.small)
                Button("Cancel", role: .destructive) { mflux.cancel() }.controlSize(.small)
            } else {
                Button { Task { await runUpdate() } } label: {
                    Label("Update", systemImage: "arrow.down.circle")
                }
                .controlSize(.small)
                .help("pip install -U mflux")
            }
        }
        .padding(12)
    }

    private var notInstalled: some View {
        VStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundStyle(.orange)
            Text("mflux venv not found").font(.headline)
            Text(mflux.binDirectory).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            Text("Install mflux in that virtualenv, or adjust the path in the service.")
                .font(.caption).foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    // MARK: - Generate

    private var generateSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Picker("Model", selection: $model) {
                ForEach(MfluxModel.allCases) { Text($0.label).tag($0) }
            }
            field("Prompt") {
                TextEditor(text: $prompt).frame(minHeight: 60)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(nsColor: .separatorColor)))
            }
            field("Negative prompt") { TextField("", text: $negativePrompt).textFieldStyle(.roundedBorder) }
            HStack {
                stepper("Width", $width, 256...2048, step: 64)
                stepper("Height", $height, 256...2048, step: 64)
            }
            HStack {
                stepper("Steps", $steps, 1...60, step: 1)
                quantizePicker("Quantize", $quantize)
            }
            field("Seed (empty = random)") { TextField("Random", text: $seedText).textFieldStyle(.roundedBorder) }
            field("LoRA paths (comma-separated)") { TextField("", text: $loraPaths).textFieldStyle(.roundedBorder) }
            field("LoRA scales (comma-separated)") { TextField("", text: $loraScales).textFieldStyle(.roundedBorder) }
            HStack {
                field("img2img reference (optional)") {
                    HStack {
                        TextField("", text: $imagePath).textFieldStyle(.roundedBorder)
                        Button("Browse…") { pickFile($imagePath) }
                    }
                }
            }
            Toggle("Low-RAM mode", isOn: $lowRam)

            Button {
                Task { await runGenerate() }
            } label: {
                Label("Generate with mflux", systemImage: "wand.and.stars").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(mflux.isRunning || prompt.trimmingCharacters(in: .whitespaces).isEmpty)

            if let resultURL {
                AsyncImage(url: resultURL) { phase in
                    if case .success(let img) = phase {
                        img.resizable().scaledToFit().frame(maxHeight: 280)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
            statusLine
        }
        .padding(12)
    }

    // MARK: - Train

    private var trainSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("LoRA / DreamBooth finetuning via mflux-train. Provide a training config JSON, or resume a checkpoint.")
                .font(.caption).foregroundStyle(.secondary)
            Picker("Base model", selection: $trainModel) {
                ForEach(MfluxModel.allCases) { Text($0.label).tag($0) }
            }
            field("Training config (.json)") {
                HStack {
                    TextField("", text: $trainConfigPath).textFieldStyle(.roundedBorder)
                    Button("Browse…") { pickFile($trainConfigPath, types: ["json"]) }
                }
            }
            field("Or resume checkpoint (.zip)") {
                HStack {
                    TextField("", text: $trainResumePath).textFieldStyle(.roundedBorder)
                    Button("Browse…") { pickFile($trainResumePath, types: ["zip"]) }
                }
            }
            HStack {
                quantizePicker("Quantize", $trainQuantize)
                Toggle("Dry run (validate only)", isOn: $trainDryRun)
            }
            Button {
                Task { await runTrain() }
            } label: {
                Label(trainDryRun ? "Validate config" : "Start training", systemImage: "graduationcap")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(mflux.isRunning || (trainConfigPath.isEmpty && trainResumePath.isEmpty))
            statusLine
        }
        .padding(12)
    }

    // MARK: - Tools (save / quantize / bake)

    private var toolsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Save a quantized model, or bake LoRAs into a checkpoint (mflux-save).")
                .font(.caption).foregroundStyle(.secondary)
            field("Model (variant, HF repo, or local path)") {
                TextField("z-image-turbo", text: $saveModel).textFieldStyle(.roundedBorder)
            }
            field("Output path") {
                HStack {
                    TextField("~/Downloads/MyModel-4bit", text: $savePath).textFieldStyle(.roundedBorder)
                    Button("Browse…") { pickDirectory($savePath) }
                }
            }
            quantizePicker("Quantize", $saveQuantize)
            field("LoRA paths to bake (comma-separated, optional)") {
                TextField("", text: $saveLoraPaths).textFieldStyle(.roundedBorder)
            }
            field("LoRA scales (comma-separated)") {
                TextField("", text: $saveLoraScales).textFieldStyle(.roundedBorder)
            }
            Button {
                Task { await runSave() }
            } label: {
                Label("Save / Quantize", systemImage: "square.and.arrow.down").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(mflux.isRunning || saveModel.isEmpty || savePath.isEmpty)
            statusLine
        }
        .padding(12)
    }

    // MARK: - Console

    private var console: some View {
        VStack(alignment: .leading, spacing: 4) {
            Divider()
            HStack {
                Text("Console").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                Button { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(mflux.consoleLog, forType: .string) } label: {
                    Image(systemName: "doc.on.doc")
                }.buttonStyle(.borderless).help("Copy log")
            }
            .padding(.horizontal, 12)
            ScrollView {
                Text(mflux.consoleLog.isEmpty ? "—" : mflux.consoleLog)
                    .font(.system(.caption2, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(8)
            }
            .frame(height: 150)
            .background(Color(nsColor: .textBackgroundColor))
        }
    }

    // MARK: - Helpers

    @ViewBuilder private func field<C: View>(_ title: String, @ViewBuilder content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            content()
        }
    }

    private func stepper(_ title: String, _ value: Binding<Int>, _ range: ClosedRange<Int>, step: Int) -> some View {
        HStack(spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            TextField("", value: value, format: .number).frame(width: 56).textFieldStyle(.roundedBorder)
            Stepper("", value: value, in: range, step: step).labelsHidden()
        }
    }

    private func quantizePicker(_ title: String, _ value: Binding<Int>) -> some View {
        HStack(spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Picker("", selection: value) {
                ForEach(Self.quantizeOptions, id: \.self) { q in
                    Text(q == 0 ? "None" : "\(q)-bit").tag(q)
                }
            }.labelsHidden().fixedSize()
        }
    }

    @ViewBuilder private var statusLine: some View {
        if let status {
            Label(status, systemImage: isError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                .font(.caption).foregroundStyle(isError ? .orange : .green)
        }
    }

    private func csv(_ s: String) -> [String] {
        s.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }
    private func csvDoubles(_ s: String) -> [Double] { csv(s).compactMap(Double.init) }

    // MARK: - Actions

    private func runGenerate() async {
        status = nil
        let dir = DesktopSettings.load().outputDirectory
        let output = (dir as NSString).appendingPathComponent("mflux-\(Int(Date().timeIntervalSince1970)).png")
        let opts = MfluxService.GenerateOptions(
            model: model.rawValue, prompt: prompt,
            negativePrompt: negativePrompt.isEmpty ? nil : negativePrompt,
            width: width, height: height, steps: steps,
            seed: UInt64(seedText),
            quantize: quantize == 0 ? nil : quantize,
            loraPaths: csv(loraPaths), loraScales: csvDoubles(loraScales),
            imagePath: imagePath.isEmpty ? nil : imagePath,
            imageStrength: imagePath.isEmpty ? nil : 0.6,
            lowRam: lowRam, output: output)
        do {
            try await mflux.run("mflux-generate", args: MfluxService.generateArgs(opts))
            resultURL = URL(fileURLWithPath: output)
            try? await ingestor?.ingestFile(at: output)
            status = "Generated \(URL(fileURLWithPath: output).lastPathComponent)"
            isError = false
        } catch {
            status = error.localizedDescription; isError = true
        }
    }

    private func runTrain() async {
        status = nil
        let opts = MfluxService.TrainOptions(
            model: trainModel.rawValue,
            quantize: trainQuantize == 0 ? nil : trainQuantize,
            configPath: trainConfigPath.isEmpty ? nil : trainConfigPath,
            resumePath: trainResumePath.isEmpty ? nil : trainResumePath,
            dryRun: trainDryRun)
        do {
            try await mflux.run("mflux-train", args: MfluxService.trainArgs(opts))
            status = trainDryRun ? "Config valid." : "Training finished."
            isError = false
        } catch {
            status = error.localizedDescription; isError = true
        }
    }

    private func runSave() async {
        status = nil
        let opts = MfluxService.SaveOptions(
            model: saveModel, path: (savePath as NSString).expandingTildeInPath,
            baseModel: saveModel.contains("z-image") ? "z-image-turbo" : nil,
            quantize: saveQuantize == 0 ? nil : saveQuantize,
            loraPaths: csv(saveLoraPaths), loraScales: csvDoubles(saveLoraScales))
        do {
            try await mflux.run("mflux-save", args: MfluxService.saveArgs(opts))
            status = "Saved to \(savePath)"; isError = false
        } catch {
            status = error.localizedDescription; isError = true
        }
    }

    private func runUpdate() async {
        status = nil
        do { try await mflux.update(); status = "mflux updated."; isError = false }
        catch { status = error.localizedDescription; isError = true }
    }

    private func pickFile(_ binding: Binding<String>, types: [String] = []) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true; panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
        if !types.isEmpty {
            panel.allowedContentTypes = types.compactMap { .init(filenameExtension: $0) }
        }
        if panel.runModal() == .OK, let url = panel.url { binding.wrappedValue = url.path }
    }

    private func pickDirectory(_ binding: Binding<String>) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false; panel.canChooseDirectories = true; panel.canCreateDirectories = true
        if panel.runModal() == .OK, let url = panel.url { binding.wrappedValue = url.path }
    }
}
