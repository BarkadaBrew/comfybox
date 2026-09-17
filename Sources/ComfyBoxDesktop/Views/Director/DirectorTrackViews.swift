// DirectorTrackViews.swift — the Director timeline's tracks (WP3). Thin
// renderers over DirectorDocumentModel: they place items with DirectorLayout
// and turn drags/drops into model calls; the model snaps, clamps and refuses.

import AppKit
import SwiftUI
import UniformTypeIdentifiers
import ZImage

// MARK: - Keyframes

struct DirectorKeyframeTrack: View {
    let model: DirectorDocumentModel
    let layout: DirectorLayout

    @State private var dragOffsets: [String: Double] = [:]
    @State private var popoverId: String?
    @State private var isDropTargeted = false

    static let thumbSize: CGFloat = 56

    var body: some View {
        ZStack(alignment: .topLeading) {
            Rectangle()
                .fill(isDropTargeted ? Color.accentColor.opacity(0.12) : Color.secondary.opacity(0.05))
            if model.timeline.keyframes.isEmpty {
                Text("Drop images here to place keyframes")
                    .font(.caption).foregroundStyle(.tertiary)
                    .offset(x: layout.leadingInset + 8, y: 30)
            }
            ForEach(model.timeline.keyframes, id: \.id) { kf in
                keyframeView(kf)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture { model.selection.removeAll() }
        .onDrop(of: [.fileURL, .image], isTargeted: $isDropTargeted) { providers, location in
            let frame = layout.frame(forX: location.x, length: model.lengthFrames, snap: true)
            return handleImageDrops(providers) { paths in
                for (i, path) in paths.enumerated() {
                    model.addKeyframe(imagePath: path, atFrame: frame + DirectorMath.latentStride * i)
                }
            }
        }
    }

    private func keyframeView(_ kf: DirectorTimeline.Keyframe) -> some View {
        let selected = model.selection.contains(kf.id)
        let x = layout.x(forFrame: kf.frame) + (dragOffsets[kf.id] ?? 0)
        return VStack(spacing: 2) {
            DirectorThumbnail(path: kf.imagePath)
                .frame(width: Self.thumbSize, height: Self.thumbSize)
                .clipShape(RoundedRectangle(cornerRadius: 4))
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(selected ? Color.accentColor : Color.secondary.opacity(0.4), lineWidth: selected ? 2 : 1))
                .opacity(Double(0.35 + 0.65 * kf.strength))
            HStack(spacing: 2) {
                if kf.isEndFrame { Image(systemName: "flag.checkered").font(.system(size: 8)) }
                Text("\(kf.frame)f").font(.system(size: 9).monospacedDigit())
            }
            .foregroundStyle(.secondary)
        }
        .offset(x: x - Self.thumbSize / 2, y: 4)
        .onTapGesture(count: 2) { popoverId = kf.id }
        .onTapGesture { model.selection = [kf.id] }
        .gesture(
            DragGesture(minimumDistance: 3)
                .onChanged { value in dragOffsets[kf.id] = value.translation.width }
                .onEnded { value in
                    dragOffsets[kf.id] = nil
                    model.moveKeyframe(id: kf.id, toFrame: kf.frame + layout.frames(forDeltaX: value.translation.width))
                }
        )
        .popover(isPresented: Binding(get: { popoverId == kf.id }, set: { if !$0 { popoverId = nil } })) {
            DirectorKeyframePopover(model: model, keyframeId: kf.id)
        }
        .contextMenu {
            Button("Edit…") { popoverId = kf.id }
            Button(kf.isEndFrame ? "Unset End Frame" : "Set as End Frame") { model.toggleEndFrame(id: kf.id) }
            Button("Remove", role: .destructive) { model.removeKeyframe(id: kf.id) }
        }
        .help(kf.imagePath ?? kf.id)
    }
}

struct DirectorKeyframePopover: View {
    let model: DirectorDocumentModel
    let keyframeId: String

    var body: some View {
        if let kf = model.timeline.keyframes.first(where: { $0.id == keyframeId }) {
            VStack(alignment: .leading, spacing: 10) {
                Text("Keyframe \(kf.id) · frame \(kf.frame)").font(.headline)
                if let path = kf.imagePath {
                    Text((path as NSString).lastPathComponent).font(.caption).foregroundStyle(.secondary)
                }
                NumericSliderField(
                    label: "Strength",
                    value: Binding(
                        get: { Double(kf.strength) },
                        set: { model.setKeyframeStrength(id: keyframeId, strength: Float($0)) }),
                    range: 0.05...1.0, step: 0.05, fractionDigits: 2)
                Toggle("End frame (pinned to the last frame)", isOn: Binding(
                    get: { kf.isEndFrame },
                    set: { _ in model.toggleEndFrame(id: keyframeId) }))
                Button("Remove Keyframe", role: .destructive) { model.removeKeyframe(id: keyframeId) }
            }
            .padding(14)
            .frame(width: 280)
        } else {
            Text("Keyframe removed").padding()
        }
    }
}

/// Keyframe thumbnail loaded off-main (MotionView.setReference pattern).
struct DirectorThumbnail: View {
    let path: String?
    @State private var image: NSImage?

    var body: some View {
        ZStack {
            Color.secondary.opacity(0.15)
            if let image {
                Image(nsImage: image).resizable().scaledToFill()
            } else {
                Image(systemName: "photo").foregroundStyle(.tertiary)
            }
        }
        .task(id: path) {
            guard let path else { image = nil; return }
            image = await Task.detached(priority: .utility) { NSImage(contentsOfFile: path) }.value
        }
    }
}

// MARK: - Prompts

struct DirectorPromptTrack: View {
    @Bindable var engine: EngineService
    let model: DirectorDocumentModel
    let layout: DirectorLayout

    @State private var resizeDelta: [String: (start: Double, end: Double)] = [:]
    @State private var enhanceId: String?
    @State private var attemptIds: [String: String] = [:]

    var body: some View {
        ZStack(alignment: .topLeading) {
            Rectangle()
                .fill(Color.secondary.opacity(0.03))
                .contentShape(Rectangle())
                .onTapGesture(count: 2, coordinateSpace: .local) { location in
                    let frame = layout.frame(forX: location.x, length: model.lengthFrames, snap: model.snapToGrid)
                    let span = model.suggestedSegmentSpan(atFrame: frame)
                    let id = model.addPromptSegment(startFrame: span.start, lengthFrames: span.length, prompt: "")
                    model.selection = [id]
                }
            if model.timeline.promptSegments.isEmpty {
                Text("Double-click to add a prompt segment")
                    .font(.caption).foregroundStyle(.tertiary)
                    .offset(x: layout.leadingInset + 8, y: 50)
                    .allowsHitTesting(false)
            }
            ForEach(Array(model.timeline.promptSegments.enumerated()), id: \.element.id) { index, seg in
                segmentView(seg, lane: index % 2)
            }
        }
    }

    private func segmentView(_ seg: DirectorTimeline.PromptSegment, lane: Int) -> some View {
        let selected = model.selection.contains(seg.id)
        let delta = resizeDelta[seg.id] ?? (0, 0)
        let x = layout.x(forFrame: seg.startFrame) + delta.start
        let width = max(24, layout.width(forFrames: seg.lengthFrames) - delta.start + delta.end)
        let height = DirectorTimelineCanvas.promptTrackHeight / 2 - 6
        return ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 5)
                .fill(Color.purple.opacity(selected ? 0.28 : 0.16))
                .overlay(RoundedRectangle(cornerRadius: 5).stroke(selected ? Color.purple : Color.purple.opacity(0.4)))
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("\(seg.id) · \(seg.startFrame)–\(seg.startFrame + seg.lengthFrames)f")
                        .font(.system(size: 9).monospacedDigit()).foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                    Button { enhanceId = seg.id } label: { Image(systemName: "sparkles").font(.system(size: 9)) }
                        .buttonStyle(.borderless)
                        .help("Enhance this segment's prompt")
                        .popover(isPresented: Binding(get: { enhanceId == seg.id }, set: { if !$0 { enhanceId = nil } })) {
                            OptimizeBar(
                                engine: engine,
                                prompt: Binding(get: { segmentPrompt(seg.id) }, set: { model.setSegmentPrompt(id: seg.id, prompt: $0) }),
                                optimizationAttemptId: Binding(get: { attemptIds[seg.id] }, set: { attemptIds[seg.id] = $0 }))
                                .padding(12)
                                .frame(width: 360)
                        }
                }
                TextEditor(text: Binding(get: { segmentPrompt(seg.id) }, set: { model.setSegmentPrompt(id: seg.id, prompt: $0) }))
                    .font(.system(size: 11))
                    .scrollContentBackground(.hidden)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            edgeHandle(seg, leading: true).frame(maxHeight: .infinity, alignment: .leading)
            edgeHandle(seg, leading: false).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .trailing)
        }
        .frame(width: width, height: height)
        .offset(x: x, y: 4 + CGFloat(lane) * (height + 4))
        .onTapGesture { model.selection = [seg.id] }
        .contextMenu {
            Button("Split at Playhead") { model.selection = [seg.id]; model.splitAtPlayhead() }
            Button("Remove", role: .destructive) { model.removePromptSegment(id: seg.id) }
        }
    }

    private func segmentPrompt(_ id: String) -> String {
        model.timeline.promptSegments.first { $0.id == id }?.prompt ?? ""
    }

    private func edgeHandle(_ seg: DirectorTimeline.PromptSegment, leading: Bool) -> some View {
        Rectangle()
            .fill(Color.purple.opacity(0.6))
            .frame(width: 5)
            .contentShape(Rectangle())
            .onHover { inside in
                if inside { NSCursor.resizeLeftRight.push() } else { NSCursor.pop() }
            }
            .gesture(
                DragGesture(minimumDistance: 1)
                    .onChanged { value in
                        var d = resizeDelta[seg.id] ?? (0, 0)
                        if leading { d.start = value.translation.width } else { d.end = value.translation.width }
                        resizeDelta[seg.id] = d
                    }
                    .onEnded { value in
                        resizeDelta[seg.id] = nil
                        let df = layout.frames(forDeltaX: value.translation.width)
                        if leading {
                            model.resizePromptSegment(id: seg.id, startFrame: seg.startFrame + df, lengthFrames: seg.lengthFrames - df)
                        } else {
                            model.resizePromptSegment(id: seg.id, startFrame: seg.startFrame, lengthFrames: seg.lengthFrames + df)
                        }
                    }
            )
    }
}

// MARK: - Audio

/// Waveform peaks decoded once per file, off the main thread.
@MainActor
final class DirectorWaveformCache {
    static let shared = DirectorWaveformCache()
    nonisolated static let buckets = 400

    private var peaks: [String: [Float]] = [:]
    private var inFlight: Set<String> = []

    func cachedPeaks(for path: String) -> [Float]? { peaks[path] }

    func peaks(for path: String, maxSeconds: Double) async -> [Float] {
        if let cached = peaks[path] { return cached }
        guard !inFlight.contains(path) else { return [] }
        inFlight.insert(path)
        defer { inFlight.remove(path) }
        let computed: [Float] = await Task.detached(priority: .utility) {
            guard let pcm = try? DirectorAudioIngest.decode(path: path, maxSeconds: maxSeconds) else { return [] }
            return DirectorWaveformCache.computePeaks(pcm, buckets: DirectorWaveformCache.buckets)
        }.value
        peaks[path] = computed
        return computed
    }

    nonisolated static func computePeaks(_ pcm: StereoPCM, buckets: Int) -> [Float] {
        let n = pcm.frames
        guard n > 0, buckets > 0 else { return [] }
        let per = max(1, n / buckets)
        var out: [Float] = []
        out.reserveCapacity(buckets)
        var i = 0
        while i < n {
            let end = min(n, i + per)
            var peak: Float = 0
            for j in i..<end { peak = max(peak, abs(pcm.left[j]), abs(pcm.right[j])) }
            out.append(min(1, peak))
            i = end
        }
        return out
    }
}

struct DirectorAudioTrack: View {
    let model: DirectorDocumentModel
    let layout: DirectorLayout

    @State private var moveOffsets: [String: Double] = [:]
    @State private var trimOffsets: [String: Double] = [:]
    @State private var popoverId: String?
    @State private var isDropTargeted = false

    var body: some View {
        ZStack(alignment: .topLeading) {
            Rectangle().fill(isDropTargeted ? Color.accentColor.opacity(0.12) : Color.secondary.opacity(0.05))
            if model.timeline.audioClips.isEmpty {
                Text(model.timeline.audio.mode == .imported
                     ? "Drop audio files here" : "Drop audio files here (switch audio mode to Imported to use them)")
                    .font(.caption).foregroundStyle(.tertiary)
                    .offset(x: layout.leadingInset + 8, y: 22)
            }
            ForEach(model.timeline.audioClips, id: \.id) { clip in
                clipView(clip)
            }
        }
        .contentShape(Rectangle())
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers, location in
            let frame = layout.frame(forX: location.x, length: model.lengthFrames, snap: model.snapToGrid)
            return handleAudioDrops(providers, atFrame: frame)
        }
    }

    private func handleAudioDrops(_ providers: [NSItemProvider], atFrame frame: Int) -> Bool {
        let fileProviders = providers.filter { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }
        guard !fileProviders.isEmpty else { return false }
        let fps = model.fps
        for provider in fileProviders {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url,
                      let type = UTType(filenameExtension: url.pathExtension), type.conforms(to: .audio)
                else { return }
                let probe = DirectorAudioIngest.probe(path: url.path)
                let seconds = probe?.durationSeconds ?? 0
                DispatchQueue.main.async {
                    let frames = max(1, Int((seconds * Double(fps)).rounded(.up)))
                    model.addAudioClip(path: url.path, atFrame: frame, lengthFrames: frames)
                }
            }
        }
        return true
    }

    private func clipView(_ clip: DirectorTimeline.AudioClip) -> some View {
        let selected = model.selection.contains(clip.id)
        let x = layout.x(forFrame: clip.startFrame) + (moveOffsets[clip.id] ?? 0)
        let width = max(12, layout.width(forFrames: clip.lengthFrames) + (trimOffsets[clip.id] ?? 0))
        let height = DirectorTimelineCanvas.audioTrackHeight - 10
        let ignored = model.timeline.audio.mode != .imported
        return ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 5)
                .fill(Color.teal.opacity(selected ? 0.3 : 0.18))
                .overlay(RoundedRectangle(cornerRadius: 5).stroke(selected ? Color.teal : Color.teal.opacity(0.4)))
            DirectorWaveformView(clip: clip, fps: model.fps)
                .padding(.vertical, 12)
            Text("\((clip.audioPath as NSString).lastPathComponent) · ×\(String(format: "%.2f", clip.gain))")
                .font(.system(size: 9)).foregroundStyle(.secondary)
                .padding(.leading, 6).padding(.top, 1)
            Rectangle()
                .fill(Color.teal.opacity(0.7))
                .frame(width: 5)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .trailing)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 1)
                        .onChanged { trimOffsets[clip.id] = $0.translation.width }
                        .onEnded { value in
                            trimOffsets[clip.id] = nil
                            model.trimAudioClip(
                                id: clip.id, trimStartFrames: clip.trimStartFrames,
                                lengthFrames: clip.lengthFrames + layout.frames(forDeltaX: value.translation.width))
                        }
                )
        }
        .frame(width: width, height: height)
        .opacity(ignored ? 0.5 : 1)
        .offset(x: x, y: 5)
        .onTapGesture(count: 2) { popoverId = clip.id }
        .onTapGesture { model.selection = [clip.id] }
        .gesture(
            DragGesture(minimumDistance: 3)
                .onChanged { moveOffsets[clip.id] = $0.translation.width }
                .onEnded { value in
                    moveOffsets[clip.id] = nil
                    model.moveAudioClip(id: clip.id, toFrame: clip.startFrame + layout.frames(forDeltaX: value.translation.width))
                }
        )
        .popover(isPresented: Binding(get: { popoverId == clip.id }, set: { if !$0 { popoverId = nil } })) {
            DirectorAudioClipPopover(model: model, clipId: clip.id)
        }
        .contextMenu {
            Button("Edit…") { popoverId = clip.id }
            Button("Split at Playhead") { model.selection = [clip.id]; model.splitAtPlayhead() }
            Button("Remove", role: .destructive) { model.removeAudioClip(id: clip.id) }
        }
        .help(ignored ? "Audio clips are only used when audio mode is Imported" : clip.audioPath)
    }
}

struct DirectorWaveformView: View {
    let clip: DirectorTimeline.AudioClip
    let fps: Int
    @State private var peaks: [Float] = []

    var body: some View {
        Canvas { context, size in
            guard !peaks.isEmpty, fps > 0 else { return }
            // Peaks cover the file's head up to trim + length; draw the used window.
            let totalFrames = clip.trimStartFrames + clip.lengthFrames
            let startFraction = Double(clip.trimStartFrames) / Double(max(1, totalFrames))
            let first = Int(Double(peaks.count) * startFraction)
            let used = Array(peaks[min(first, peaks.count)...])
            guard !used.isEmpty else { return }
            let mid = size.height / 2
            var path = Path()
            for (i, p) in used.enumerated() {
                let x = size.width * Double(i) / Double(used.count)
                let h = max(0.5, Double(p) * mid)
                path.move(to: CGPoint(x: x, y: mid - h))
                path.addLine(to: CGPoint(x: x, y: mid + h))
            }
            context.stroke(path, with: .color(.teal.opacity(0.8)), lineWidth: 1)
        }
        .task(id: "\(clip.audioPath)|\(clip.trimStartFrames)|\(clip.lengthFrames)") {
            let seconds = Double(clip.trimStartFrames + clip.lengthFrames) / Double(max(1, fps))
            // Cached per path; the first decode bounds itself to the clip's use.
            let cache = DirectorWaveformCache.shared
            if let cached = cache.cachedPeaks(for: clip.audioPath) {
                peaks = cached
            } else {
                peaks = await cache.peaks(for: clip.audioPath, maxSeconds: seconds)
            }
        }
    }
}

struct DirectorAudioClipPopover: View {
    let model: DirectorDocumentModel
    let clipId: String

    var body: some View {
        if let clip = model.timeline.audioClips.first(where: { $0.id == clipId }) {
            VStack(alignment: .leading, spacing: 10) {
                Text("Audio \(clip.id) · starts at \(clip.startFrame)f").font(.headline)
                Text((clip.audioPath as NSString).lastPathComponent).font(.caption).foregroundStyle(.secondary)
                NumericSliderField(
                    label: "Gain",
                    value: Binding(get: { Double(clip.gain) }, set: { model.setAudioClipGain(id: clipId, gain: Float($0)) }),
                    range: 0...2, step: 0.05, fractionDigits: 2)
                NumericSliderField(
                    label: "Trim start (frames)",
                    value: Binding(
                        get: { Double(clip.trimStartFrames) },
                        set: { model.trimAudioClip(id: clipId, trimStartFrames: Int($0), lengthFrames: clip.lengthFrames) }),
                    range: 0...Double(max(1, model.lengthFrames * 4)), step: 1)
                NumericSliderField(
                    label: "Length (frames)",
                    value: Binding(
                        get: { Double(clip.lengthFrames) },
                        set: { model.trimAudioClip(id: clipId, trimStartFrames: clip.trimStartFrames, lengthFrames: Int($0)) }),
                    range: 1...Double(max(2, model.lengthFrames)), step: 1)
                Button("Remove Clip", role: .destructive) { model.removeAudioClip(id: clipId) }
            }
            .padding(14)
            .frame(width: 300)
        } else {
            Text("Clip removed").padding()
        }
    }
}

// MARK: - Reference (Phase 2)

struct DirectorReferenceTrackPlaceholder: View {
    let layout: DirectorLayout
    let lengthFrames: Int

    var body: some View {
        ZStack(alignment: .leading) {
            Rectangle().fill(Color.secondary.opacity(0.03))
            Text("Reference clips — Phase 2")
                .font(.caption2).foregroundStyle(.tertiary)
                .padding(.leading, layout.leadingInset + 8)
        }
        .frame(width: layout.contentWidth(lengthFrames: lengthFrames))
        .allowsHitTesting(false)
    }
}
