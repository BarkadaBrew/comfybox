// RemoteGalleryLightbox.swift — full-screen viewer for a remote asset.
//
// Mirrors GalleryLightbox's viewing mechanics (image zoom/pan, video
// playback, prev/next navigation) so Remote Gallery gets the same "click a
// video and it actually plays" utility the main Gallery has — previously
// RemoteGalleryView's grid showed a bare "film" icon for videos with no way
// to view them at all. Sources from RemoteGalleryService.fileURL(for:) (an
// HTTP stream off the remote server) instead of a local file path, since a
// remote asset generally isn't on disk yet. Local-only actions from the main
// lightbox (Finder color labels, Reveal in Finder) don't apply here — "Save
// locally" (pull) is the one asset-level action offered.

import SwiftUI
import AVKit
import AppKit

struct RemoteGalleryLightbox: View {
    let assets: [RemoteGalleryService.RemoteAsset]
    let index: Int
    let remote: RemoteGalleryService
    var onSave: ((RemoteGalleryService.RemoteAsset) -> Void)?
    let onIndexChange: (Int) -> Void
    let onClose: () -> Void

    @State private var player: AVPlayer?
    @State private var zoom: CGFloat = 1
    @State private var baseZoom: CGFloat = 1
    @State private var offset: CGSize = .zero
    @FocusState private var focused: Bool

    private var asset: RemoteGalleryService.RemoteAsset? {
        assets.indices.contains(index) ? assets[index] : nil
    }

    var body: some View {
        ZStack {
            Color.black.opacity(0.93).ignoresSafeArea()
                .onTapGesture { onClose() }

            Group {
                if asset?.kind == "video", let player {
                    VideoPlayer(player: player)
                        .aspectRatio(contentMode: .fit)
                        .padding(24)
                } else if let asset, let url = remote.fileURL(for: asset.path) {
                    AsyncImage(url: url) { phase in
                        if case .success(let image) = phase {
                            image
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .scaleEffect(zoom)
                                .offset(offset)
                                .gesture(
                                    MagnifyGesture()
                                        .onChanged { zoom = min(6, max(1, baseZoom * $0.magnification)) }
                                        .onEnded { _ in baseZoom = zoom }
                                )
                                .highPriorityGesture(
                                    DragGesture()
                                        .onChanged { if zoom > 1 { offset = $0.translation } }
                                )
                                .onTapGesture(count: 2) {
                                    withAnimation(.easeInOut(duration: 0.2)) {
                                        zoom = zoom > 1 ? 1 : 2
                                        baseZoom = zoom
                                        offset = .zero
                                    }
                                }
                        } else if case .failure = phase {
                            Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
                        } else {
                            ProgressView().controlSize(.large).tint(.white)
                        }
                    }
                } else {
                    ProgressView().controlSize(.large).tint(.white)
                }
            }
            .padding(40)
            .contentGated(cornerRadius: 0)

            // Prev / next
            HStack {
                navButton("chevron.left") { step(-1) }.opacity(index > 0 ? 1 : 0.25).disabled(index <= 0)
                Spacer()
                navButton("chevron.right") { step(1) }.opacity(index < assets.count - 1 ? 1 : 0.25).disabled(index >= assets.count - 1)
            }
            .padding(.horizontal, 20)

            // Close + caption chrome
            VStack {
                HStack {
                    Spacer()
                    Button(action: onClose) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .foregroundStyle(.white, .black.opacity(0.4))
                    }
                    .buttonStyle(.plain)
                    .padding(16)
                }
                Spacer()
                if let asset {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(asset.filename)
                                .font(.callout).foregroundStyle(.white)
                                .lineLimit(1).truncationMode(.middle)
                            Text("\(index + 1) of \(assets.count)")
                                .font(.caption).foregroundStyle(.white.opacity(0.7))
                        }
                        Spacer()
                        if !remote.isLocalServer, onSave != nil {
                            Button { onSave?(asset) } label: {
                                Label("Save locally", systemImage: "square.and.arrow.down")
                            }
                            .buttonStyle(.borderedProminent)
                        }
                    }
                    .padding(16)
                    .background(.black.opacity(0.4))
                }
            }
        }
        .focusable()
        .focused($focused)
        .onKeyPress(.leftArrow) { step(-1); return .handled }
        .onKeyPress(.rightArrow) { step(1); return .handled }
        .onKeyPress(.escape) { onClose(); return .handled }
        .task(id: index) { await load() }
        .onAppear { focused = true }
    }

    private func navButton(_ icon: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 26, weight: .semibold))
                .foregroundStyle(.white)
                .padding(14)
                .background(.black.opacity(0.4), in: Circle())
        }
        .buttonStyle(.plain)
    }

    private func step(_ delta: Int) {
        let next = index + delta
        guard assets.indices.contains(next) else { return }
        zoom = 1; baseZoom = 1; offset = .zero
        onIndexChange(next)
    }

    private func load() async {
        player?.pause()
        player = nil
        guard let asset, asset.kind == "video", let url = remote.fileURL(for: asset.path) else { return }
        player = AVPlayer(url: url)
    }
}
