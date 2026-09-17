// DirectorResultView.swift — the Director tab's result pane (WP3): the stitched
// render in SafeVideoPlayer with the plan's chunk boundaries and keyframe
// ticks overlaid on a strip beneath it, plus Reveal/Open.

import AVKit
import AppKit
import SwiftUI
import ZImage

struct DirectorResultView: View {
    let player: AVPlayer?
    let resultURL: URL?
    let plan: DirectorPlan?
    let isGenerating: Bool
    let statusMessage: String?
    let errorMessage: String?

    var body: some View {
        VStack(spacing: 6) {
            if let errorMessage {
                Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .font(.caption)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            ZStack {
                if let resultURL {
                    VStack(spacing: 4) {
                        SafeVideoPlayer(player: player)
                        if let plan { planStrip(plan) }
                        HStack {
                            Text(resultURL.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                            Spacer()
                            Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([resultURL]) }
                            Button("Open") { NSWorkspace.shared.open(resultURL) }
                        }
                    }
                } else if isGenerating {
                    VStack(spacing: 10) {
                        ProgressView().controlSize(.large)
                        Text(statusMessage ?? "Rendering…").foregroundStyle(.secondary)
                        Text("Each chunk renders as its own LTX-2 job — this can take a while.")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "film").font(.system(size: 40)).foregroundStyle(.tertiary)
                        Text(statusMessage ?? "Your Director render will appear here").foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(10)
    }

    /// Chunk boundaries (orange) and keyframe ticks (accent) placed
    /// proportionally under the player so they line up with its scrubber.
    private func planStrip(_ plan: DirectorPlan) -> some View {
        GeometryReader { geo in
            let span = Double(max(1, plan.lengthFrames - 1))
            ZStack(alignment: .topLeading) {
                Capsule().fill(Color.secondary.opacity(0.15)).frame(height: 4).offset(y: 5)
                ForEach(plan.boundaryFrames, id: \.self) { frame in
                    Rectangle().fill(Color.orange)
                        .frame(width: 2, height: 14)
                        .offset(x: geo.size.width * Double(frame) / span - 1)
                }
                ForEach(plan.keyframeTicks, id: \.id) { tick in
                    Image(systemName: "triangle.fill")
                        .font(.system(size: 7))
                        .foregroundStyle(Color.accentColor)
                        .rotationEffect(.degrees(180))
                        .offset(x: geo.size.width * Double(tick.frame) / span - 4, y: -2)
                }
            }
        }
        .frame(height: 14)
        .help("Orange: chunk boundaries · triangles: keyframes")
    }
}
