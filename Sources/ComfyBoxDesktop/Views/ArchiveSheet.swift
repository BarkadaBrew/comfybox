// ArchiveSheet.swift — name / destination / summary sheet for archiving a
// selection, a single asset, or a folder's assets into a .cbarchive bundle.
//
// Styled after the NSFW password sheet in GalleryView: VStack(alignment:
// .leading, spacing: 12), .padding(18).frame(width: 420). Collects the
// archive name and destination (from DesktopSettings.archiveRoots, or a
// freshly chosen folder), then hands the name + destination back to the
// caller — GalleryView owns the actual `GalleryArchiver.archive` call and
// its progress/summary state, mirroring the import strip.

import SwiftUI

struct ArchiveSheet: View {
    let assets: [DAMAsset]
    /// Set when archiving a folder's contents — used for the default name
    /// and shown so the sheet reads as "archiving this folder".
    let folder: DAMFolder?
    let store: DAMStore
    @Binding var isPresented: Bool
    let onArchive: (_ name: String, _ destinationRoot: String) -> Void

    private static let chooseSentinel = "__choose__"

    @State private var archiveName: String
    @State private var archiveRoots: [String]
    @State private var destinationRoot: String
    @State private var securedCount: Int = 0

    init(
        assets: [DAMAsset], folder: DAMFolder?, store: DAMStore, isPresented: Binding<Bool>,
        onArchive: @escaping (_ name: String, _ destinationRoot: String) -> Void
    ) {
        self.assets = assets
        self.folder = folder
        self.store = store
        self._isPresented = isPresented
        self.onArchive = onArchive

        let roots = DesktopSettings.load().archiveRoots ?? [DesktopSettings.defaultArchiveRoot]
        self._archiveRoots = State(initialValue: roots)
        self._destinationRoot = State(initialValue: roots.first ?? DesktopSettings.defaultArchiveRoot)
        self._archiveName = State(initialValue: folder?.name
            ?? "Archive \(Date().formatted(.dateTime.year().month().day()))")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Archive \(assets.count) image\(assets.count == 1 ? "" : "s")", systemImage: "archivebox")
                .font(.headline)

            TextField("Archive name", text: $archiveName)
                .textFieldStyle(.roundedBorder)

            Picker("Destination", selection: $destinationRoot) {
                ForEach(archiveRoots, id: \.self) { root in
                    Text((root as NSString).lastPathComponent).tag(root)
                }
                Text("Choose…").tag(Self.chooseSentinel)
            }
            .onChange(of: destinationRoot) { _, newValue in
                if newValue == Self.chooseSentinel {
                    chooseDestination()
                }
            }

            Text(Self.summaryLine(
                assetCount: assets.count,
                totalBytes: assets.reduce(Int64(0)) { $0 + $1.fileSize },
                securedCount: securedCount
            ))
            .font(.caption)
            .foregroundStyle(.secondary)

            HStack {
                Spacer()
                Button("Cancel") { isPresented = false }
                Button("Archive") {
                    let name = archiveName.trimmingCharacters(in: .whitespacesAndNewlines)
                    onArchive(name.isEmpty ? defaultName : name, destinationRoot)
                    isPresented = false
                }
                .buttonStyle(.borderedProminent)
                .disabled(assets.isEmpty || destinationRoot == Self.chooseSentinel)
            }
        }
        .padding(18).frame(width: 420)
        .task {
            if let securedIds = try? await store.securedAssetIds() {
                securedCount = assets.filter { securedIds.contains($0.id) }.count
            }
        }
    }

    private var defaultName: String {
        folder?.name ?? "Archive \(Date().formatted(.dateTime.year().month().day()))"
    }

    /// The exact panel configuration `GalleryView.chooseFolderToImport()`
    /// already uses (directories only, single selection). A newly chosen
    /// root is appended to `DesktopSettings.archiveRoots` and saved so it
    /// shows up in the picker on future archives. Cancelling the panel
    /// leaves the picker on its previous selection rather than the sentinel.
    private func chooseDestination() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.message = "Choose a folder to store the archive."
        guard panel.runModal() == .OK, let url = panel.url else {
            destinationRoot = archiveRoots.first ?? DesktopSettings.defaultArchiveRoot
            return
        }
        let path = url.path
        if !archiveRoots.contains(path) {
            archiveRoots.append(path)
            var settings = DesktopSettings.load()
            settings.archiveRoots = archiveRoots
            settings.save()
        }
        destinationRoot = path
    }

    /// "N images · X GB · M secured images will be skipped". Pure so it's
    /// unit-testable without a view host; `securedCount` of 0 omits the
    /// trailing clause entirely.
    static func summaryLine(assetCount: Int, totalBytes: Int64, securedCount: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useGB, .useMB, .useKB, .useBytes]
        let sizeString = formatter.string(fromByteCount: totalBytes)
        let imageWord = assetCount == 1 ? "image" : "images"
        var line = "\(assetCount) \(imageWord) · \(sizeString)"
        if securedCount > 0 {
            let securedWord = securedCount == 1 ? "image" : "images"
            line += " · \(securedCount) secured \(securedWord) will be skipped"
        }
        return line
    }
}
