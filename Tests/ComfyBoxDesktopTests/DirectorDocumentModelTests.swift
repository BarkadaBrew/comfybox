// DirectorDocumentModelTests.swift — WP3 (docs/FDD-ltx-director-tab.md §4.4)
//
// The Director tab's document model is pure and headless: every timeline
// edit (snapping, split, end-frame, undo, dirty, validation gate) is tested
// here without rendering a view.

import Foundation
import SwiftUI
import Testing
import ZImage
@testable import ComfyBoxDesktop

func directorTestTimeline(length: Int = 289) -> DirectorTimeline {
    DirectorTimeline(
        settings: .init(width: 576, height: 896, lengthFrames: length),
        globalPrompt: "a woman walks along the beach at dusk")
}

@MainActor
@Suite("DirectorDocumentModel")
struct DirectorDocumentModelTests {

    @Test("addKeyframe snaps to the 8-frame latent grid (100 -> 96)")
    func addKeyframeSnapsToGrid() throws {
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        let id = try #require(model.addKeyframe(imagePath: "/tmp/a.png", atFrame: 100))
        let kf = try #require(model.timeline.keyframes.first { $0.id == id })
        #expect(kf.frame == 96)
        #expect(kf.imagePath == "/tmp/a.png")
        #expect(kf.strength == 1.0)
    }

    @Test("moveKeyframe refuses a latent bucket another keyframe occupies")
    func moveKeyframeRefusesOccupiedBucket() throws {
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        let a = try #require(model.addKeyframe(imagePath: "/tmp/a.png", atFrame: 0))
        let b = try #require(model.addKeyframe(imagePath: "/tmp/b.png", atFrame: 96))
        #expect(model.moveKeyframe(id: b, toFrame: 3) == false)  // snaps to 0 -> occupied by a
        #expect(model.timeline.keyframes.first { $0.id == b }?.frame == 96)
        #expect(model.moveKeyframe(id: b, toFrame: 150) == true)  // snaps to 152
        #expect(model.timeline.keyframes.first { $0.id == b }?.frame == 152)
        #expect(model.timeline.keyframes.first { $0.id == a }?.frame == 0)
    }

    @Test("toggleEndFrame pins to length-1 and clears any other end flag")
    func toggleEndFramePinsToLastFrameAndClearsOthers() throws {
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        let a = try #require(model.addKeyframe(imagePath: "/tmp/a.png", atFrame: 0))
        let b = try #require(model.addKeyframe(imagePath: "/tmp/b.png", atFrame: 96))
        model.toggleEndFrame(id: a)
        #expect(model.timeline.keyframes.first { $0.id == a }?.isEndFrame == true)
        #expect(model.timeline.keyframes.first { $0.id == a }?.frame == 288)
        model.toggleEndFrame(id: b)
        let ka = try #require(model.timeline.keyframes.first { $0.id == a })
        let kb = try #require(model.timeline.keyframes.first { $0.id == b })
        #expect(kb.isEndFrame == true)
        #expect(kb.frame == 288)
        #expect(ka.isEndFrame == false)
        // The demoted keyframe no longer shares the last latent bucket.
        #expect(ka.frame / 8 != kb.frame / 8)
        // Toggling off leaves the frame where it is.
        model.toggleEndFrame(id: b)
        #expect(model.timeline.keyframes.first { $0.id == b }?.isEndFrame == false)
        #expect(model.timeline.keyframes.first { $0.id == b }?.frame == 288)
    }

    @Test("splitAtPlayhead splits prompt segments and audio clips, carrying the trim")
    func splitAtPlayheadSplitsPromptAndAudio() throws {
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        let p = model.addPromptSegment(startFrame: 0, lengthFrames: 96, prompt: "she turns")
        let a = model.addAudioClip(path: "/tmp/voice.wav", atFrame: 0, lengthFrames: 96)
        model.trimAudioClip(id: a, trimStartFrames: 12, lengthFrames: 96)
        model.playheadFrame = 40
        model.splitAtPlayhead()

        let segs = model.timeline.promptSegments.sorted { $0.startFrame < $1.startFrame }
        #expect(segs.count == 2)
        #expect(segs[0].id == p && segs[0].startFrame == 0 && segs[0].lengthFrames == 40)
        #expect(segs[1].id == "\(p)-b" && segs[1].startFrame == 40 && segs[1].lengthFrames == 56)
        #expect(segs[0].prompt == "she turns" && segs[1].prompt == "she turns")

        let clips = model.timeline.audioClips.sorted { $0.startFrame < $1.startFrame }
        #expect(clips.count == 2)
        #expect(clips[0].lengthFrames == 40 && clips[0].trimStartFrames == 12)
        #expect(clips[1].id == "\(a)-b")
        #expect(clips[1].startFrame == 40 && clips[1].lengthFrames == 56)
        #expect(clips[1].trimStartFrames == 12 + 40)
    }

    @Test("splitAtPlayhead on a segment edge is a no-op (no undo, not dirty)")
    func splitAtPlayheadOnEdgeIsNoOp() {
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        _ = model.addPromptSegment(startFrame: 0, lengthFrames: 96, prompt: "x")
        model.markSaved()
        let before = model.timeline
        model.playheadFrame = 96
        model.splitAtPlayhead()
        model.playheadFrame = 0
        model.splitAtPlayhead()
        #expect(model.timeline == before)
        #expect(model.isDirty == false)
    }

    @Test("splitAtPlayhead only splits the selection when one exists")
    func splitRespectsSelection() {
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        let p1 = model.addPromptSegment(startFrame: 0, lengthFrames: 96, prompt: "a")
        _ = model.addPromptSegment(startFrame: 0, lengthFrames: 96, prompt: "b")
        model.selection = [p1]
        model.playheadFrame = 48
        model.splitAtPlayhead()
        #expect(model.timeline.promptSegments.count == 3)
    }

    @Test("resizePromptSegment clamps to the timeline and snaps to the grid")
    func resizeClampsAndSnaps() throws {
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        let p = model.addPromptSegment(startFrame: 0, lengthFrames: 96, prompt: "x")
        model.resizePromptSegment(id: p, startFrame: -10, lengthFrames: 1000)
        var seg = try #require(model.timeline.promptSegments.first)
        #expect(seg.startFrame == 0 && seg.lengthFrames == 289)
        model.resizePromptSegment(id: p, startFrame: 13, lengthFrames: 50)
        seg = try #require(model.timeline.promptSegments.first)
        #expect(seg.startFrame == 16 && seg.lengthFrames == 48)
        model.snapToGrid = false
        model.resizePromptSegment(id: p, startFrame: 13, lengthFrames: 0)
        seg = try #require(model.timeline.promptSegments.first)
        #expect(seg.startFrame == 13 && seg.lengthFrames == 1)
    }

    @Test("setLengthFrames snaps up, drops keyframes past the end, re-pins the end frame")
    func setLengthFramesSnapsUpAndRepinsEndFrame() throws {
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        let end = try #require(model.addKeyframe(imagePath: "/tmp/e.png", atFrame: 0))
        model.toggleEndFrame(id: end)
        let late = try #require(model.addKeyframe(imagePath: "/tmp/l.png", atFrame: 200))
        model.setLengthFrames(150)
        #expect(model.timeline.settings.lengthFrames == 153)
        #expect(model.timeline.keyframes.first { $0.id == end }?.frame == 152)
        #expect(model.timeline.keyframes.contains { $0.id == late } == false)
        #expect(model.issues.contains { $0.severity == .warning && $0.ids.contains(late) })
    }

    @Test("setLengthFrames moves a regular keyframe out of the end frame's latent bucket")
    func setLengthFramesResolvesEndBucketCollision() throws {
        let model = DirectorDocumentModel(timeline: directorTestTimeline(length: 145))
        let k1 = try #require(model.addKeyframe(imagePath: "/tmp/k1.png", atFrame: 96))
        let k2 = try #require(model.addKeyframe(imagePath: "/tmp/k2.png", atFrame: 0))
        model.toggleEndFrame(id: k2)
        #expect(model.timeline.keyframes.first { $0.id == k2 }?.frame == 144)
        model.setLengthFrames(97)
        #expect(model.timeline.settings.lengthFrames == 97)
        #expect(model.timeline.keyframes.first { $0.id == k2 }?.frame == 96)
        let moved = try #require(model.timeline.keyframes.first { $0.id == k1 })
        #expect(moved.frame == 88)
        #expect(!DirectorMath.keyframeBucketsCollide(model.timeline.keyframes.map(\.frame)))
        let v = DirectorValidator.validate(model.timeline, fileExists: { _ in true }, audioProbe: { _ in nil })
        #expect(!v.issues.contains { $0.code == "keyframes_collide" })
    }

    @Test("undo and redo restore whole-timeline snapshots")
    func undoRedoRestoresSnapshots() throws {
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        let original = model.timeline
        let id = try #require(model.addKeyframe(imagePath: "/tmp/a.png", atFrame: 0))
        let afterAdd = model.timeline
        model.setKeyframeStrength(id: id, strength: 0.5)
        #expect(model.canUndo)
        model.undo()
        #expect(model.timeline == afterAdd)
        model.undo()
        #expect(model.timeline == original)
        #expect(model.canUndo == false)
        model.redo()
        #expect(model.timeline == afterAdd)
        model.redo()
        #expect(model.timeline.keyframes.first?.strength == 0.5)
        #expect(model.canRedo == false)
    }

    @Test("prompt typing coalesces into one undo step")
    func promptEditsCoalesce() {
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        let original = model.timeline
        model.setGlobalPrompt("a")
        model.setGlobalPrompt("ab")
        model.setGlobalPrompt("abc")
        model.undo()
        #expect(model.timeline == original)
    }

    @Test("dirty is set by an edit and cleared by save")
    func dirtyFlagSetOnEditClearedOnSave() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("director-dirty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        #expect(model.isDirty == false)
        model.setGlobalPrompt("changed")
        #expect(model.isDirty)
        let url = dir.appendingPathComponent("p.cbdirector")
        try model.save(to: url)
        #expect(model.isDirty == false)
        #expect(model.projectURL == url)
    }

    @Test("warnings keep Generate enabled; an error disables it")
    func canGenerateFalseOnErrorTrueOnWarningsOnly() {
        var t = directorTestTimeline()
        t.keyframes = [
            .init(id: "k1", imagePath: "/fake/a.png", frame: 0),
            .init(id: "k2", imagePath: "/fake/b.png", frame: 16),
        ]
        let model = DirectorDocumentModel(timeline: t)
        model.validateLocally(fileExists: { _ in true }, audioProbe: { _ in nil })
        #expect(model.issues.contains { $0.code == "keyframes_close" })
        #expect(!model.issues.contains { $0.severity == .error })
        #expect(model.canGenerate)
        #expect(model.plan != nil)

        var bad = t
        bad.keyframes[1].frame = 100
        let badModel = DirectorDocumentModel(timeline: bad)
        badModel.validateLocally(fileExists: { _ in true }, audioProbe: { _ in nil })
        #expect(badModel.issues.contains { $0.code == "keyframe_off_grid" })
        #expect(badModel.canGenerate == false)

        let empty = DirectorDocumentModel(timeline: directorTestTimeline())
        empty.setGlobalPrompt("   ")
        #expect(empty.canGenerate == false)
    }

    @Test("selecting an issue selects its ids on the canvas")
    func issueIdsSelectable() {
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        let issue = DirectorIssue.warning("keyframes_close", "close", ids: ["k1", "k2"])
        model.select(issue: issue)
        #expect(model.selection == ["k1", "k2"])
    }

    @Test("speech readout sums global + segment quotes, counting a split quote twice")
    func speechSecondsSumsGlobalAndSegmentsCountingSplitQuoteTwice() {
        var t = directorTestTimeline()
        t.globalPrompt = #"She says "hello there""#  // 2 words
        let model = DirectorDocumentModel(timeline: t)
        let p = model.addPromptSegment(startFrame: 0, lengthFrames: 96, prompt: #"He answers “good evening to you”"#)  // 4 words
        #expect(abs(model.speechSeconds - 6.0 / 2.5) < 1e-9)
        model.selection = [p]
        model.playheadFrame = 48
        model.splitAtPlayhead()
        // The duplicated quote now lives in two segments and is counted twice.
        #expect(abs(model.speechSeconds - 10.0 / 2.5) < 1e-9)
        #expect(abs(model.timelineSeconds - 289.0 / 24.0) < 1e-9)
    }
}

@Suite("AppTab.director")
struct DirectorTabTests {
    @Test("Director is a Create tab on ⌘T")
    func directorTabInCreateSectionWithShortcutT() {
        let tab = ComfyBoxDesktopApp.AppTab.director
        #expect(tab.section == .create)
        #expect(tab.shortcutKey == "t")
        #expect(tab.rawValue == "Director")
        let keys = ComfyBoxDesktopApp.AppTab.allCases.map { String($0.shortcutKey.character) }
        #expect(Set(keys).count == keys.count)
    }
}
