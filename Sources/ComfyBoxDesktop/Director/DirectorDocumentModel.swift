// DirectorDocumentModel.swift — the Director tab's document model (WP3,
// docs/FDD-ltx-director-tab.md §4.4).
//
// Pure and headless: owns the DirectorTimeline being edited plus the editor
// state around it (selection, playhead, zoom, undo/redo, dirty, project URL,
// last validation). Every mutation goes through `commit`, which pushes a
// whole-timeline undo snapshot, marks the document dirty and schedules an
// autosave. Frame rules come from ZImage's DirectorMath/DirectorValidator so
// the desktop, CLI and engine agree; the views only bind and hit-test.

import Foundation
import Observation
import ZImage

@Observable
@MainActor
final class DirectorDocumentModel {

    /// Undo depth. Only a portable .cbdirector opened on another machine
    /// carries base64 in memory, so whole-document snapshots are cheap.
    static let undoLimit = 100

    /// A fresh document: portrait 576x896, 289 frames (12 s @ 24 fps).
    nonisolated static func defaultTimeline(width: Int = 576, height: Int = 896) -> DirectorTimeline {
        DirectorTimeline(settings: .init(width: width, height: height, lengthFrames: 289), globalPrompt: "")
    }

    private(set) var timeline: DirectorTimeline
    var selection: Set<String> = []
    /// Clamped to the timeline on every edit (and by the canvas when dragged).
    var playheadFrame: Int = 0
    var zoom: Double = 1.0
    /// Snap edits to the 8-frame latent grid (keyframes always snap).
    var snapToGrid: Bool = true
    private(set) var projectURL: URL?
    private(set) var isDirty: Bool = false
    /// Last validation's issues (plus model-generated warnings such as items
    /// dropped by shortening the timeline).
    var issues: [DirectorIssue] = []
    /// Last validation's (or submission's / final status's) plan — drives the
    /// chunk-boundary and keyframe tick overlay.
    var plan: DirectorPlan?
    /// Re-run local validation after every edit (the view turns this on).
    var autoValidate: Bool = false
    private(set) var lastEditedAt: Date?

    @ObservationIgnored private var undoStack: [DirectorTimeline] = []
    @ObservationIgnored private var redoStack: [DirectorTimeline] = []
    @ObservationIgnored private var coalesceKey: String?
    @ObservationIgnored private let autosave: DirectorAutosaveStore?
    @ObservationIgnored private let now: () -> Date

    init(
        timeline: DirectorTimeline = DirectorDocumentModel.defaultTimeline(),
        autosave: DirectorAutosaveStore? = nil,
        now: @escaping () -> Date = Date.init
    ) {
        self.timeline = timeline
        self.autosave = autosave
        self.now = now
    }

    // MARK: - Derived

    var lengthFrames: Int { timeline.settings.lengthFrames }
    /// The timeline fps, or the builtin the server would fall back to when
    /// the document names none (a hand-written or MCP timeline).
    var fps: Int { timeline.settings.fps ?? DirectorTimeline.Settings.defaultFps }
    var canUndo: Bool { !undoStack.isEmpty }
    var canRedo: Bool { !redoStack.isEmpty }

    /// Generate is allowed when the last validation had no errors and the
    /// global prompt is non-empty. Warnings never disable it.
    var canGenerate: Bool {
        !timeline.globalPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !issues.contains { $0.severity == .error }
    }

    /// Estimated seconds of spoken dialogue: quoted words in the global prompt
    /// plus in every prompt segment. A quote lives in every segment that
    /// carries it, so a split segment counts its duplicated quote twice —
    /// the compiler really does send it to both chunks' prompts.
    var speechSeconds: Double {
        SpeechLength.seconds(timeline.globalPrompt)
            + timeline.promptSegments.reduce(0) { $0 + SpeechLength.seconds($1.prompt) }
    }

    var timelineSeconds: Double {
        DirectorMath.seconds(frames: timeline.settings.lengthFrames, fps: fps)
    }

    // MARK: - Commit / undo

    /// Apply `mutate` as one undoable edit. No-op edits push nothing.
    /// `coalesce` merges consecutive edits with the same key (typing) into a
    /// single undo step.
    private func commit(coalesce: String? = nil, _ mutate: (inout DirectorTimeline) -> Void) {
        var next = timeline
        mutate(&next)
        guard next != timeline else { return }
        if coalesce == nil || coalesce != coalesceKey {
            undoStack.append(timeline)
            if undoStack.count > Self.undoLimit { undoStack.removeFirst(undoStack.count - Self.undoLimit) }
        }
        coalesceKey = coalesce
        redoStack.removeAll()
        timeline = next
        didEdit()
    }

    private func didEdit() {
        isDirty = true
        lastEditedAt = now()
        clampPlayhead()
        autosave?.schedule(timeline)
        if autoValidate { validateLocally() }
    }

    private func clampPlayhead() {
        playheadFrame = min(max(0, playheadFrame), max(0, timeline.settings.lengthFrames - 1))
    }

    func undo() {
        guard let previous = undoStack.popLast() else { return }
        redoStack.append(timeline)
        timeline = previous
        coalesceKey = nil
        didEdit()
    }

    func redo() {
        guard let next = redoStack.popLast() else { return }
        undoStack.append(timeline)
        timeline = next
        coalesceKey = nil
        didEdit()
    }

    /// Clear the dirty flag without writing (e.g. after an external save).
    func markSaved() {
        isDirty = false
        coalesceKey = nil
    }

    // MARK: - Ids

    private var allIds: Set<String> {
        Set(timeline.keyframes.map(\.id) + timeline.promptSegments.map(\.id)
            + timeline.audioClips.map(\.id) + timeline.referenceClips.map(\.id))
    }

    /// Next `<prefix><n>` unused across every track (the validator requires
    /// ids to be unique globally).
    private func nextId(_ prefix: String) -> String {
        let used = allIds
        var n = 1
        while used.contains("\(prefix)\(n)") { n += 1 }
        return "\(prefix)\(n)"
    }

    private func splitId(_ id: String, used: Set<String>) -> String {
        var candidate = "\(id)-b"
        while used.contains(candidate) { candidate += "b" }
        return candidate
    }

    // MARK: - Keyframes

    private func occupiedBuckets(excluding id: String? = nil) -> Set<Int> {
        Set(timeline.keyframes.filter { $0.id != id }.map { $0.frame / DirectorMath.latentStride })
    }

    /// Add a keyframe at the grid frame nearest `frame`. When that latent
    /// bucket is taken, the nearest free grid frame is used instead; nil when
    /// the timeline has no free bucket.
    @discardableResult
    func addKeyframe(imagePath: String, atFrame frame: Int) -> String? {
        let length = timeline.settings.lengthFrames
        let target = DirectorMath.snapToGrid(frame: frame, length: length)
        let occupied = occupiedBuckets()
        let stride = DirectorMath.latentStride
        let lastGrid = ((length - 1) / stride) * stride
        var chosen: Int?
        var distance = 0
        while chosen == nil && distance <= lastGrid {
            for candidate in [target + distance, target - distance]
            where candidate >= 0 && candidate <= lastGrid && !occupied.contains(candidate / stride) {
                chosen = candidate
                break
            }
            distance += stride
        }
        guard let placed = chosen else { return nil }
        let id = nextId("k")
        commit { $0.keyframes.append(.init(id: id, imagePath: imagePath, frame: placed)) }
        return id
    }

    /// Move a keyframe to the grid frame nearest `frame`. Refuses (false) a
    /// latent bucket another keyframe occupies. Moving the end frame off the
    /// last frame clears its end flag.
    @discardableResult
    func moveKeyframe(id: String, toFrame frame: Int) -> Bool {
        guard let index = timeline.keyframes.firstIndex(where: { $0.id == id }) else { return false }
        let length = timeline.settings.lengthFrames
        let target = DirectorMath.snapToGrid(frame: frame, length: length)
        guard !occupiedBuckets(excluding: id).contains(target / DirectorMath.latentStride) else { return false }
        commit {
            $0.keyframes[index].frame = target
            if $0.keyframes[index].isEndFrame && target != length - 1 {
                $0.keyframes[index].isEndFrame = false
            }
        }
        return true
    }

    func setKeyframeStrength(id: String, strength: Float) {
        guard let index = timeline.keyframes.firstIndex(where: { $0.id == id }) else { return }
        let clamped = min(max(strength, 0.05), 1.0)
        commit(coalesce: "strength:\(id)") { $0.keyframes[index].strength = clamped }
    }

    /// true pins the keyframe to length-1 and clears any other end flag (the
    /// demoted keyframe moves to the nearest free grid frame if it would share
    /// the last latent bucket); false leaves the frame where it is.
    func toggleEndFrame(id: String) {
        guard let index = timeline.keyframes.firstIndex(where: { $0.id == id }) else { return }
        let length = timeline.settings.lengthFrames
        let stride = DirectorMath.latentStride
        commit { t in
            if t.keyframes[index].isEndFrame {
                t.keyframes[index].isEndFrame = false
                return
            }
            let last = length - 1
            t.keyframes[index].isEndFrame = true
            t.keyframes[index].frame = last
            for j in t.keyframes.indices where j != index && t.keyframes[j].frame / stride == last / stride {
                t.keyframes[j].isEndFrame = false
                let occupied = Set(t.keyframes.enumerated().filter { $0.offset != j }.map { $0.element.frame / stride })
                var candidate = (last / stride) * stride - stride
                while candidate >= 0 && occupied.contains(candidate / stride) { candidate -= stride }
                if candidate >= 0 { t.keyframes[j].frame = candidate }
            }
            for j in t.keyframes.indices where j != index { t.keyframes[j].isEndFrame = false }
        }
    }

    func removeKeyframe(id: String) {
        commit { $0.keyframes.removeAll { $0.id == id } }
        selection.remove(id)
    }

    // MARK: - Prompt segments

    /// Clamp (and optionally snap) a [start, start+length) span to the
    /// timeline; at least 1 frame long.
    private func normalizedSpan(start: Int, length: Int) -> (start: Int, length: Int) {
        let total = timeline.settings.lengthFrames
        var s = min(max(0, start), max(0, total - 1))
        var end = min(max(s + 1, start + length), total)
        if snapToGrid {
            s = DirectorMath.snapToGrid(frame: s, length: total)
            if end < total {
                end = min(total, ((end + DirectorMath.latentStride / 2 - 1) / DirectorMath.latentStride) * DirectorMath.latentStride)
            }
            if end <= s { end = min(total, s + DirectorMath.latentStride) }
        }
        return (s, max(1, end - s))
    }

    @discardableResult
    func addPromptSegment(startFrame: Int, lengthFrames: Int, prompt: String) -> String {
        let id = nextId("p")
        let span = normalizedSpan(start: startFrame, length: lengthFrames)
        commit { $0.promptSegments.append(.init(id: id, startFrame: span.start, lengthFrames: span.length, prompt: prompt)) }
        return id
    }

    func resizePromptSegment(id: String, startFrame: Int, lengthFrames: Int) {
        guard let index = timeline.promptSegments.firstIndex(where: { $0.id == id }) else { return }
        let span = normalizedSpan(start: startFrame, length: lengthFrames)
        commit {
            $0.promptSegments[index].startFrame = span.start
            $0.promptSegments[index].lengthFrames = span.length
        }
    }

    func setSegmentPrompt(id: String, prompt: String) {
        guard let index = timeline.promptSegments.firstIndex(where: { $0.id == id }) else { return }
        commit(coalesce: "segment:\(id)") { $0.promptSegments[index].prompt = prompt }
    }

    func removePromptSegment(id: String) {
        commit { $0.promptSegments.removeAll { $0.id == id } }
        selection.remove(id)
    }

    /// Where a new segment double-clicked at `frame` should go: from the grid
    /// frame at the click to the next segment's start, at most 96 frames.
    func suggestedSegmentSpan(atFrame frame: Int) -> (start: Int, length: Int) {
        let total = timeline.settings.lengthFrames
        let start = DirectorMath.snapToGrid(frame: frame, length: total)
        let next = timeline.promptSegments.map(\.startFrame).filter { $0 > start }.min() ?? total
        return (start, max(1, min(96, next - start)))
    }

    // MARK: - Audio clips

    @discardableResult
    func addAudioClip(path: String, atFrame frame: Int, lengthFrames: Int) -> String {
        let id = nextId("a")
        let total = timeline.settings.lengthFrames
        let start = min(max(0, snapToGrid ? DirectorMath.snapToGrid(frame: frame, length: total) : frame), max(0, total - 1))
        let length = max(1, min(lengthFrames, total - start))
        commit { $0.audioClips.append(.init(id: id, audioPath: path, startFrame: start, lengthFrames: length)) }
        return id
    }

    func moveAudioClip(id: String, toFrame frame: Int) {
        guard let index = timeline.audioClips.firstIndex(where: { $0.id == id }) else { return }
        let total = timeline.settings.lengthFrames
        let start = min(max(0, snapToGrid ? DirectorMath.snapToGrid(frame: frame, length: total) : frame), max(0, total - 1))
        commit { $0.audioClips[index].startFrame = start }
    }

    func trimAudioClip(id: String, trimStartFrames: Int, lengthFrames: Int) {
        guard let index = timeline.audioClips.firstIndex(where: { $0.id == id }) else { return }
        commit {
            $0.audioClips[index].trimStartFrames = max(0, trimStartFrames)
            $0.audioClips[index].lengthFrames = max(1, lengthFrames)
        }
    }

    func setAudioClipGain(id: String, gain: Float) {
        guard let index = timeline.audioClips.firstIndex(where: { $0.id == id }) else { return }
        commit(coalesce: "gain:\(id)") { $0.audioClips[index].gain = max(0, gain) }
    }

    func removeAudioClip(id: String) {
        commit { $0.audioClips.removeAll { $0.id == id } }
        selection.remove(id)
    }

    // MARK: - Split

    /// Split every selected prompt segment and audio clip (or, with nothing
    /// selected, every one) that strictly contains the playhead. Both halves
    /// keep the prompt text; the audio clip's second half skips the first
    /// half's frames of the file. New ids are `<id>-b`.
    func splitAtPlayhead() {
        let p = playheadFrame
        let targets = selection
        func chosen(_ id: String) -> Bool { targets.isEmpty || targets.contains(id) }
        commit { t in
            var used = Set(t.keyframes.map(\.id) + t.promptSegments.map(\.id) + t.audioClips.map(\.id) + t.referenceClips.map(\.id))
            var segments: [DirectorTimeline.PromptSegment] = []
            for seg in t.promptSegments {
                guard chosen(seg.id), seg.startFrame < p, p < seg.startFrame + seg.lengthFrames else {
                    segments.append(seg)
                    continue
                }
                let firstLength = p - seg.startFrame
                var first = seg
                first.lengthFrames = firstLength
                let secondId = splitId(seg.id, used: used)
                used.insert(secondId)
                segments.append(first)
                segments.append(.init(id: secondId, startFrame: p, lengthFrames: seg.lengthFrames - firstLength, prompt: seg.prompt))
            }
            t.promptSegments = segments

            var clips: [DirectorTimeline.AudioClip] = []
            for clip in t.audioClips {
                guard chosen(clip.id), clip.startFrame < p, p < clip.startFrame + clip.lengthFrames else {
                    clips.append(clip)
                    continue
                }
                let firstLength = p - clip.startFrame
                var first = clip
                first.lengthFrames = firstLength
                let secondId = splitId(clip.id, used: used)
                used.insert(secondId)
                clips.append(first)
                clips.append(.init(
                    id: secondId, audioPath: clip.audioPath, startFrame: p,
                    lengthFrames: clip.lengthFrames - firstLength,
                    trimStartFrames: clip.trimStartFrames + firstLength, gain: clip.gain))
            }
            t.audioClips = clips
        }
    }

    // MARK: - Settings

    /// Snap the length UP to 1+8k. Keyframes past the new end are removed
    /// (reported as a warning issue), the end-frame keyframe is re-pinned to
    /// the new last frame, and prompt segments / audio clips are clipped.
    func setLengthFrames(_ requested: Int) {
        let length = DirectorMath.snapLengthUp(max(1, requested))
        var removed: [String] = []
        commit { t in
            t.settings.lengthFrames = length
            removed = t.keyframes.filter { !$0.isEndFrame && $0.frame > length - 1 }.map(\.id)
            t.keyframes.removeAll { !$0.isEndFrame && $0.frame > length - 1 }
            for i in t.keyframes.indices where t.keyframes[i].isEndFrame {
                t.keyframes[i].frame = length - 1
            }
            // A regular keyframe left in the end frame's latent bucket would
            // collide with it (keyframes_collide): move it to the nearest free
            // earlier grid frame, as toggleEndFrame does, or drop it.
            if let endIndex = t.keyframes.firstIndex(where: \.isEndFrame) {
                let stride = DirectorMath.latentStride
                let lastBucket = (length - 1) / stride
                for j in t.keyframes.indices where j != endIndex && t.keyframes[j].frame / stride == lastBucket {
                    let occupied = Set(t.keyframes.enumerated().filter { $0.offset != j }.map { $0.element.frame / stride })
                    var candidate = lastBucket * stride - stride
                    while candidate >= 0 && occupied.contains(candidate / stride) { candidate -= stride }
                    if candidate >= 0 {
                        t.keyframes[j].frame = candidate
                    } else {
                        removed.append(t.keyframes[j].id)
                    }
                }
                t.keyframes.removeAll { !$0.isEndFrame && removed.contains($0.id) }
            }
            removed += t.promptSegments.filter { $0.startFrame >= length }.map(\.id)
            t.promptSegments.removeAll { $0.startFrame >= length }
            for i in t.promptSegments.indices {
                t.promptSegments[i].lengthFrames = min(t.promptSegments[i].lengthFrames, length - t.promptSegments[i].startFrame)
            }
            removed += t.audioClips.filter { $0.startFrame >= length }.map(\.id)
            t.audioClips.removeAll { $0.startFrame >= length }
            for i in t.audioClips.indices {
                t.audioClips[i].lengthFrames = min(t.audioClips[i].lengthFrames, length - t.audioClips[i].startFrame)
            }
        }
        if !removed.isEmpty {
            issues.append(.warning(
                "items_removed_by_length",
                "shortening the timeline to \(length) frames removed \(removed.joined(separator: ", "))",
                ids: removed))
            selection.subtract(removed)
        }
    }

    /// Edit the settings block (fps, dimensions, seed, preset, LoRAs,
    /// negative prompt, steps, character) as one undo step. Use
    /// `setLengthFrames` for the length.
    func updateSettings(coalesce: String? = nil, _ mutate: (inout DirectorTimeline.Settings) -> Void) {
        commit(coalesce: coalesce) { t in
            let length = t.settings.lengthFrames
            mutate(&t.settings)
            t.settings.lengthFrames = length
        }
    }

    func setAudioMode(_ mode: DirectorTimeline.AudioMode) {
        commit { $0.audio.mode = mode }
    }

    func setGlobalPrompt(_ prompt: String) {
        commit(coalesce: "global") { $0.globalPrompt = prompt }
    }

    // MARK: - Documents

    /// Reset to an empty document (keeping the current resolution), drop the
    /// project URL, undo history and autosave.
    func new() {
        let settings = timeline.settings
        timeline = Self.defaultTimeline(width: settings.width, height: settings.height)
        resetEditorState()
        projectURL = nil
        autosave?.clear()
    }

    /// Open a .cbdirector file. Resets undo, clears dirty, does NOT autosave.
    func load(url: URL) throws {
        let loaded = try DirectorDocument.read(from: url)
        timeline = loaded
        resetEditorState()
        projectURL = url
    }

    /// Open the SEQUENCE beside a rendered clip: `clip.mp4` ->
    /// `clip.sequence.json` (FDD §4.9.2). This is "drop the mp4 back on the
    /// Director tab and get the timeline that made it", the way a ComfyUI PNG
    /// restores its graph. Returns nil when the clip predates sequences, so
    /// the caller can say so rather than opening an empty timeline.
    @discardableResult
    func loadSequence(forMediaAt url: URL) -> SequenceDocument? {
        guard let document = SequenceSidecar.read(forMediaAt: url.path) else { return nil }
        timeline = document.timeline
        resetEditorState()
        // A sequence is not a project file: saving must ask where to put it,
        // rather than writing over the render's sidecar.
        projectURL = nil
        return document
    }

    /// Save to `url` (assets embedded only when asked), clearing dirty.
    func save(to url: URL, embedAssets: Bool = false) throws {
        try DirectorDocument.write(timeline, to: url, embedAssets: embedAssets)
        projectURL = url
        markSaved()
    }

    /// Restore the autosaved timeline when no project is open and nothing has
    /// been edited. The restored document is dirty (it was never saved).
    @discardableResult
    func restoreAutosaveIfAny() -> Bool {
        guard projectURL == nil, !isDirty, let saved = autosave?.load() else { return false }
        timeline = saved
        resetEditorState()
        isDirty = true
        return true
    }

    private func resetEditorState() {
        undoStack.removeAll()
        redoStack.removeAll()
        coalesceKey = nil
        selection.removeAll()
        playheadFrame = 0
        isDirty = false
        issues = []
        plan = nil
        if autoValidate { validateLocally() }
    }

    // MARK: - Validation

    /// In-process validation (works offline): sets `issues` and `plan`.
    func validateLocally(
        fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) },
        audioProbe: (String) -> AudioProbe? = DirectorValidator.defaultAudioProbe
    ) {
        let v = DirectorValidator.validate(timeline, fileExists: fileExists, audioProbe: audioProbe)
        apply(issues: v.issues, plan: v.plan)
    }

    func apply(issues: [DirectorIssue], plan: DirectorPlan?) {
        self.issues = issues
        self.plan = plan
    }

    /// Clicking an issue selects the timeline elements it names.
    func select(issue: DirectorIssue) {
        selection = Set(issue.ids)
    }
}
