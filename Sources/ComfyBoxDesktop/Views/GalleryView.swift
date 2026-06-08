// GalleryView.swift — Asset gallery placeholder
//
// Phase 2 will implement a full gallery view with grid layout,
// filtering, search, and metadata display. For now, shows a
// placeholder message.

import SwiftUI

struct GalleryView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 64))
                .foregroundStyle(.tertiary)

            Text("Gallery")
                .font(.title2)
                .foregroundStyle(.secondary)

            Text("Coming in Phase 2")
                .font(.body)
                .foregroundStyle(.tertiary)

            Text("The gallery will display all generated assets with search, filtering, rating, and metadata.")
                .font(.caption)
                .foregroundStyle(.quaternary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 300)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
