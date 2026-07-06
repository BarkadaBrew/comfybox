// DecoupageView.swift — Desktop frontend for the Découpage compositing CLI
//
// Recipe-driven multi-layer mixed-media art: generate a figure + composite
// découpage elements, or composite an existing figure, or grow an element
// library. Desktop-only (shells out to ~/Projects/decoupage), like mflux.

import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct DecoupageView: View {
    @Bindable var decoupage: DecoupageService
    var ingestor: AssetIngestor?

    enum Mode: String, CaseIterable, Identifiable {
        case generate = "Generate", composite = "Composite", elements = "Elements"
        var id: String { rawValue }
    }
    @State private var mode: Mode = .generate
    @State private var recipe = ""

    // Generate
    @State private var description = ""
    @State private var seedText = ""
    @State private var anchorPath = ""
    @State private var noAnchor = false
    @State private var printPrep = false
    @State private var resultURL: URL?

    // Composite
    @State private var figurePath = ""

    // Elements
    @State private var elementCategory = ""
    @State private var elementPrompt = ""
    @State private var elementCount = 1

    @State private var status: String?
    @State private var isError = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if !decoupage.isInstalled {
                notInstalled
            } else {
                Picker("", selection: $mode) {
                    ForEach(Mode.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden()
                .padding(.horizontal, 12).padding(.top, 8)

                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        recipePicker
                        switch mode {
                        case .generate: generateSection
                        case .composite: compositeSection
                        case .elements: elementsSection
                        }
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
                    }
                    .padding(12)
                }
                console
            }
        }
        .navigationTitle("Découpage")
        .onAppear { if recipe.isEmpty { recipe = decoupage.recipes().first ?? "" } }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "square.3.layers.3d").foregroundStyle(.brown)
            Text("Découpage").font(.headline)
            Text("mixed-media compositing").font(.caption).foregroundStyle(.secondary)
            Spacer()
            if decoupage.isRunning {
                ProgressView().controlSize(.small)
                Button("Cancel", role: .destructive) { decoupage.cancel() }.controlSize(.small)
            }
        }
        .padding(12)
    }

    private var notInstalled: some View {
        VStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle").font(.largeTitle).foregroundStyle(.orange)
            Text("Découpage not found").font(.headline)
            Text(decoupage.projectDirectory).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).padding()
    }

    private var recipePicker: some View {
        field("Recipe") {
            let recipes = decoupage.recipes()
            if recipes.isEmpty {
                Text("No recipes in \(decoupage.recipesDirectory)").font(.caption).foregroundStyle(.secondary)
            } else {
                Picker("", selection: $recipe) {
                    ForEach(recipes, id: \.self) { Text($0).tag($0) }
                }.labelsHidden()
            }
        }
    }

    private var generateSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            field("Description (figure subject — no collage language)") {
                TextEditor(text: $description).frame(minHeight: 60)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(nsColor: .separatorColor)))
            }
            field("Seed (empty = random)") { TextField("Random", text: $seedText).textFieldStyle(.roundedBorder) }
            field("i2i anchor image (optional)") {
                HStack {
                    TextField("", text: $anchorPath).textFieldStyle(.roundedBorder)
                    Button("Browse…") { pickImage($anchorPath) }
                }
            }
            Toggle("Skip i2i anchor (text-only figure)", isOn: $noAnchor)
            Toggle("Print preparation (mirror, saturation, contrast)", isOn: $printPrep)
            Button { Task { await runGenerate() } } label: {
                Label("Generate & Composite", systemImage: "square.3.layers.3d").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(decoupage.isRunning || recipe.isEmpty || description.trimmingCharacters(in: .whitespaces).isEmpty)
        }
    }

    private var compositeSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Composite découpage elements onto an existing figure image.")
                .font(.caption).foregroundStyle(.secondary)
            field("Figure image") {
                HStack {
                    TextField("", text: $figurePath).textFieldStyle(.roundedBorder)
                    Button("Browse…") { pickImage($figurePath) }
                }
            }
            field("Seed (empty = random)") { TextField("Random", text: $seedText).textFieldStyle(.roundedBorder) }
            Toggle("Print preparation", isOn: $printPrep)
            Button { Task { await runComposite() } } label: {
                Label("Composite", systemImage: "rectangle.stack").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(decoupage.isRunning || recipe.isEmpty || figurePath.isEmpty)
        }
    }

    private var elementsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Grow a recipe's element library (transparent PNGs by category).")
                .font(.caption).foregroundStyle(.secondary)
            field("Category (e.g. florals, ephemera)") { TextField("", text: $elementCategory).textFieldStyle(.roundedBorder) }
            field("Element prompt") { TextField("wallpaper roses", text: $elementPrompt).textFieldStyle(.roundedBorder) }
            Stepper("Count: \(elementCount)", value: $elementCount, in: 1...20)
            Button { Task { await runElements() } } label: {
                Label("Generate Elements", systemImage: "leaf").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(decoupage.isRunning || recipe.isEmpty || elementCategory.isEmpty || elementPrompt.isEmpty)
        }
    }

    private var console: some View {
        VStack(alignment: .leading, spacing: 4) {
            Divider()
            Text("Console").font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.horizontal, 12)
            ScrollView {
                Text(decoupage.consoleLog.isEmpty ? "—" : decoupage.consoleLog)
                    .font(.system(.caption2, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled).padding(8)
            }
            .frame(height: 130).background(Color(nsColor: .textBackgroundColor))
        }
    }

    @ViewBuilder private func field<C: View>(_ title: String, @ViewBuilder content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            content()
        }
    }

    private func pickImage(_ binding: Binding<String>) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true; panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.png, .jpeg, .image]
        if panel.runModal() == .OK, let url = panel.url { binding.wrappedValue = url.path }
    }

    // MARK: - Actions

    private func runGenerate() async {
        status = nil; resultURL = nil
        let opts = DecoupageService.GenerateOptions(
            description: description, recipe: decoupage.recipePath(recipe),
            seed: Int(seedText), anchor: anchorPath.isEmpty ? nil : anchorPath,
            noAnchor: noAnchor, printPrep: printPrep)
        await runAndIngest(DecoupageService.generateArgs(opts), label: "Composited")
    }

    private func runComposite() async {
        status = nil; resultURL = nil
        let opts = DecoupageService.CompositeOptions(
            figure: figurePath, recipe: decoupage.recipePath(recipe),
            seed: Int(seedText), printPrep: printPrep)
        await runAndIngest(DecoupageService.compositeArgs(opts), label: "Composited")
    }

    private func runElements() async {
        status = nil
        let args = DecoupageService.genElementsArgs(
            recipe: decoupage.recipePath(recipe), category: elementCategory,
            prompt: elementPrompt, count: elementCount)
        do { try await decoupage.run(args); status = "Generated \(elementCount) element(s)."; isError = false }
        catch { status = error.localizedDescription; isError = true }
    }

    /// Run a compositing command, then surface + ingest the newest output PNG.
    private func runAndIngest(_ args: [String], label: String) async {
        do {
            try await decoupage.run(args)
            if let newest = newestOutput() {
                resultURL = newest
                try? await ingestor?.ingestFile(at: newest.path)
                status = "\(label) → \(newest.lastPathComponent)"
            } else {
                status = "\(label) (no output file found)."
            }
            isError = false
        } catch {
            status = error.localizedDescription; isError = true
        }
    }

    private func newestOutput() -> URL? {
        let dir = decoupage.outputDirectory
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: URL(fileURLWithPath: dir),
            includingPropertiesForKeys: [.contentModificationDateKey]) else { return nil }
        return files.filter { $0.pathExtension.lowercased() == "png" }
            .max { a, b in
                let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                return da < db
            }
    }
}
