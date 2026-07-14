// GalleryHubView.swift — card-grid hub for named "remote galleries"
// (GalleryStore.swift server-side): a gallery can be created, password-locked,
// and hidden from this listing, and apps like Kira can target one instead of
// the server's default output folder. Tapping a card drills into the existing
// RemoteGalleryView (browse/pull), scoped to that gallery — this view owns a
// small local "drilled in or not" state rather than the app's global tab
// routing, since nothing else needs to link directly to a specific gallery.

import SwiftUI

struct GalleryHubView: View {
    @State private var hub: GalleryHubService
    var ingestor: AssetIngestor?

    private struct SelectedGallery: Equatable {
        var id: String?       // nil = the default/legacy single-folder gallery
        var name: String?
        var password: String?
    }
    @State private var selected: SelectedGallery?

    @State private var showCreateSheet = false
    @State private var newName = ""
    @State private var newHidden = false
    @State private var newPassword = ""

    @State private var unlockTarget: GalleryHubService.GallerySummary?
    @State private var unlockPasswordInput = ""

    @State private var deleteTarget: GalleryHubService.GallerySummary?
    @State private var deletePasswordInput = ""
    @State private var deleteError = false

    private let columns = [GridItem(.adaptive(minimum: 200), spacing: 12)]

    init(engine: EngineService, ingestor: AssetIngestor?) {
        _hub = State(initialValue: GalleryHubService(engine: engine))
        self.ingestor = ingestor
    }

    var body: some View {
        Group {
            if let selected {
                RemoteGalleryView(
                    engine: hub.engine, ingestor: ingestor,
                    galleryId: selected.id, galleryPassword: selected.password,
                    galleryName: selected.name
                )
                .toolbar {
                    ToolbarItem(placement: .navigation) {
                        Button { self.selected = nil } label: {
                            Label("Galleries", systemImage: "chevron.left")
                        }
                    }
                }
            } else {
                hubGrid
            }
        }
        .task { await hub.load() }
    }

    private var hubGrid: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                LazyVGrid(columns: columns, spacing: 12) {
                    defaultCard
                    ForEach(hub.galleries) { gallery in
                        card(gallery)
                    }
                }
                .padding(12)
            }
        }
        .navigationTitle("Remote Galleries")
        .sheet(isPresented: $showCreateSheet) { createSheet }
        .sheet(item: $unlockTarget) { gallery in unlockSheet(gallery) }
        .sheet(item: $deleteTarget) { gallery in deleteSheet(gallery) }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Remote Galleries").font(.headline)
                Text(hub.baseURL).font(.caption.monospaced()).foregroundStyle(.secondary)
            }
            Spacer()
            if let error = hub.error { Text(error).font(.caption).foregroundStyle(.red) }
            if hub.isLoading { ProgressView().controlSize(.small) }
            Button { showCreateSheet = true } label: { Label("New Gallery", systemImage: "plus") }
                .controlSize(.small)
            Button { Task { await hub.load() } } label: { Label("Refresh", systemImage: "arrow.clockwise") }
                .controlSize(.small)
        }
        .padding(12)
    }

    private var defaultCard: some View {
        Button {
            selected = SelectedGallery(id: nil, name: nil, password: nil)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                Image(systemName: "photo.stack").font(.title2)
                Text("Default").font(.subheadline.bold())
                Text("The server's default output folder").font(.caption2).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
    }

    private func card(_ gallery: GalleryHubService.GallerySummary) -> some View {
        Button {
            if gallery.locked {
                unlockPasswordInput = ""
                unlockTarget = gallery
            } else {
                selected = SelectedGallery(id: gallery.id, name: gallery.name, password: nil)
            }
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: "photo.stack").font(.title2)
                    Spacer()
                    if gallery.locked { Image(systemName: "lock.fill").foregroundStyle(.secondary) }
                    if gallery.hidden { Image(systemName: "eye.slash").foregroundStyle(.secondary) }
                }
                Text(gallery.name).font(.subheadline.bold()).lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button("Delete", role: .destructive) {
                deletePasswordInput = ""
                deleteError = false
                deleteTarget = gallery
            }
        }
    }

    private var createSheet: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("New Gallery").font(.headline)
            TextField("Name", text: $newName).textFieldStyle(.roundedBorder)
            Toggle("Hidden (omit from this list)", isOn: $newHidden)
            SecureField("Password (optional)", text: $newPassword).textFieldStyle(.roundedBorder)
            HStack {
                Spacer()
                Button("Cancel") { showCreateSheet = false }
                Button("Create") {
                    let name = newName
                    let hidden = newHidden
                    let password = newPassword.isEmpty ? nil : newPassword
                    Task {
                        do {
                            _ = try await hub.create(name: name, hidden: hidden, password: password)
                            newName = ""; newHidden = false; newPassword = ""
                            showCreateSheet = false
                        } catch {
                            hub.error = error.localizedDescription
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(newName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(18).frame(width: 360)
    }

    private func unlockSheet(_ gallery: GalleryHubService.GallerySummary) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Unlock \(gallery.name)", systemImage: "lock.fill").font(.headline)
            Text("Enter this gallery's password to browse it. A wrong password shows up as a server error in the browse view.")
                .font(.caption).foregroundStyle(.secondary)
            SecureField("Password", text: $unlockPasswordInput)
                .textFieldStyle(.roundedBorder)
                .onSubmit { confirmUnlock(gallery) }
            HStack {
                Spacer()
                Button("Cancel") { unlockTarget = nil }
                Button("Unlock") { confirmUnlock(gallery) }.buttonStyle(.borderedProminent)
            }
        }
        .padding(18).frame(width: 380)
    }

    private func confirmUnlock(_ gallery: GalleryHubService.GallerySummary) {
        selected = SelectedGallery(id: gallery.id, name: gallery.name, password: unlockPasswordInput)
        unlockTarget = nil
    }

    private func deleteSheet(_ gallery: GalleryHubService.GallerySummary) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Delete \(gallery.name)?", systemImage: "trash").font(.headline)
            Text("This removes the gallery from the list — its rendered files are left on disk.")
                .font(.caption).foregroundStyle(.secondary)
            if gallery.locked {
                SecureField("Password", text: $deletePasswordInput)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { confirmDelete(gallery) }
            }
            if deleteError {
                Label("Incorrect password.", systemImage: "xmark.circle").font(.caption).foregroundStyle(.red)
            }
            HStack {
                Spacer()
                Button("Cancel") { deleteTarget = nil }
                Button("Delete", role: .destructive) { confirmDelete(gallery) }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(18).frame(width: 360)
    }

    private func confirmDelete(_ gallery: GalleryHubService.GallerySummary) {
        Task {
            do {
                try await hub.delete(id: gallery.id, password: gallery.locked ? deletePasswordInput : nil)
                deleteTarget = nil
            } catch {
                deleteError = true
            }
        }
    }
}
