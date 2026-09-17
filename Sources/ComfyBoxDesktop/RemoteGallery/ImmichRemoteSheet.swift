// ImmichRemoteSheet.swift — adding an Immich server as a remote gallery.
//
// The API key is handed to the caller and goes straight to the keychain; it
// is never a field of RemoteGalleryConfig and never reaches
// desktop-config.json (FDD-remote-galleries §3.1).

import SwiftUI

struct ImmichRemoteSheet: View {
    /// Called with the configured remote and the key to store, or not at all
    /// if the sheet is cancelled.
    var onAdd: (RemoteGalleryConfig, String?) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var name = "Immich"
    @State private var baseURL = "http://10.0.100.232:2283"
    @State private var albumName = "ComfyBox"
    @State private var apiKey = ""

    private var isComplete: Bool {
        !name.trimmingCharacters(in: .whitespaces).isEmpty
            && !baseURL.trimmingCharacters(in: .whitespaces).isEmpty
            && !albumName.trimmingCharacters(in: .whitespaces).isEmpty
            && !apiKey.isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Add an Immich Remote").font(.headline)

            Form {
                TextField("Name", text: $name)
                TextField("Server", text: $baseURL)
                TextField("Album", text: $albumName)
                SecureField("API key", text: $apiKey)
                Text("The key is stored in your login keychain, not in the config file. Create one in Immich under Account Settings → API Keys.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .formStyle(.grouped)

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Add") {
                    let remote = RemoteGalleryConfig.immich(
                        name: name.trimmingCharacters(in: .whitespaces),
                        baseURL: baseURL.trimmingCharacters(in: .whitespaces),
                        albumName: albumName.trimmingCharacters(in: .whitespaces))
                    onAdd(remote, apiKey)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .disabled(!isComplete)
            }
        }
        .padding(20)
        .frame(width: 440)
    }
}
