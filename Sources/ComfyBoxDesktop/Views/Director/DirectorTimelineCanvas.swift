// DirectorTimelineCanvas.swift — the Director timeline (WP3): ruler, the
// Keyframes / Prompts / Audio tracks, a disabled Reference placeholder row
// (Phase 2), the draggable playhead and the plan's chunk-boundary overlay.
// Horizontal scroll; zoom with ⌘± or a magnify gesture. Every px<->frame
// conversion goes through DirectorLayout.

import SwiftUI
import ZImage

struct DirectorTimelineCanvas: View {
    @Bindable var engine: EngineService
    let model: DirectorDocumentModel
    let onTogglePlayback: () -> Void

    @State private var magnifyBase: Double?

    static let labelWidth: CGFloat = 86
    static let keyframeTrackHeight: CGFloat = 76
    static let promptTrackHeight: CGFloat = 120
    static let audioTrackHeight: CGFloat = 60
    static let referenceTrackHeight: CGFloat = 26

    private var layout: DirectorLayout { DirectorLayout(zoom: model.zoom, leadingInset: 8) }

    private var tracksHeight: CGFloat {
        DirectorRulerView.height + Self.keyframeTrackHeight + Self.promptTrackHeight
            + Self.audioTrackHeight + Self.referenceTrackHeight
    }

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            labels
            Divider()
            ScrollView(.horizontal) {
                let layout = self.layout
                VStack(alignment: .leading, spacing: 0) {
                    DirectorRulerView(model: model, layout: layout)
                    DirectorKeyframeTrack(model: model, layout: layout)
                        .frame(height: Self.keyframeTrackHeight)
                    DirectorPromptTrack(engine: engine, model: model, layout: layout)
                        .frame(height: Self.promptTrackHeight)
                    DirectorAudioTrack(model: model, layout: layout)
                        .frame(height: Self.audioTrackHeight)
                    DirectorReferenceTrackPlaceholder(layout: layout, lengthFrames: model.lengthFrames)
                        .frame(height: Self.referenceTrackHeight)
                }
                .overlay(alignment: .topLeading) { overlay(layout: layout) }
                .frame(width: layout.contentWidth(lengthFrames: model.lengthFrames), alignment: .leading)
            }
        }
        .frame(height: tracksHeight + 16, alignment: .top)
        .frame(maxHeight: .infinity, alignment: .top)
        .focusable()
        .focusEffectDisabled()
        .onKeyPress(.space) {
            onTogglePlayback()
            return .handled
        }
        .onKeyPress(keys: ["s"]) { press in
            guard press.modifiers.isEmpty else { return .ignored }
            model.splitAtPlayhead()
            return .handled
        }
        .onKeyPress(.delete) {
            deleteSelection()
            return .handled
        }
        .gesture(
            MagnifyGesture()
                .onChanged { value in
                    let base = magnifyBase ?? model.zoom
                    if magnifyBase == nil { magnifyBase = base }
                    model.zoom = min(max(base * value.magnification, 0.1), 12)
                }
                .onEnded { _ in magnifyBase = nil }
        )
    }

    private var labels: some View {
        VStack(alignment: .leading, spacing: 0) {
            Color.clear.frame(height: DirectorRulerView.height)
            trackLabel("Keyframes", "photo.on.rectangle", Self.keyframeTrackHeight)
            trackLabel("Prompts", "text.alignleft", Self.promptTrackHeight)
            trackLabel("Audio", "waveform", Self.audioTrackHeight)
            trackLabel("Reference", "film", Self.referenceTrackHeight).opacity(0.45)
        }
        .frame(width: Self.labelWidth)
    }

    private func trackLabel(_ title: String, _ icon: String, _ height: CGFloat) -> some View {
        Label(title, systemImage: icon)
            .font(.caption)
            .foregroundStyle(.secondary)
            .frame(width: Self.labelWidth, height: height, alignment: .leading)
            .padding(.leading, 6)
    }

    /// Playhead line + chunk boundaries across every track.
    private func overlay(layout: DirectorLayout) -> some View {
        ZStack(alignment: .topLeading) {
            if let plan = model.plan {
                ForEach(Array(layout.boundaryTickXs(plan: plan).enumerated()), id: \.offset) { _, x in
                    Rectangle()
                        .fill(Color.orange.opacity(0.55))
                        .frame(width: 1, height: tracksHeight)
                        .offset(x: x)
                }
            }
            Rectangle()
                .fill(Color.red)
                .frame(width: 2, height: tracksHeight)
                .offset(x: layout.x(forFrame: model.playheadFrame) - 1)
        }
        .allowsHitTesting(false)
    }

    private func deleteSelection() {
        for id in model.selection {
            if model.timeline.keyframes.contains(where: { $0.id == id }) { model.removeKeyframe(id: id) }
            if model.timeline.promptSegments.contains(where: { $0.id == id }) { model.removePromptSegment(id: id) }
            if model.timeline.audioClips.contains(where: { $0.id == id }) { model.removeAudioClip(id: id) }
        }
    }
}
