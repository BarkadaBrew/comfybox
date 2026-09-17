// EngineServiceDirectorTests.swift — WP3: Director submit / validate / poll
// against a scripted fake transport (no engine, no weights).

import Foundation
import Testing
import ZImage
@testable import ComfyBoxDesktop

private let planJSON = """
{"length_frames":577,"fps":24,"width":576,"height":896,"audio_mode":"generated",
 "chunks":[{"index":0,"start_frame":0,"end_frame":288,"frames":289,"seed":42,"carry_over":false,"keyframes":[{"id":"k1","local_frame":0,"strength":1.0}],"prompt_segments":[],"beat_schedule":[],"audio":"generated"},
           {"index":1,"start_frame":288,"end_frame":576,"frames":289,"seed":43,"carry_over":true,"keyframes":[],"prompt_segments":[],"beat_schedule":[],"audio":"generated"}],
 "keyframe_ticks":[{"id":"k1","frame":0}],"boundary_frames":[288],"warnings":[]}
"""

@MainActor
@Suite("EngineService Director")
struct EngineServiceDirectorTests {

    private func timeline() -> DirectorTimeline {
        var t = directorTestTimeline(length: 577)
        t.keyframes = [.init(id: "k1", imagePath: "/tmp/a.png", frame: 0)]
        return t
    }

    @Test("submit returns job id, plan and stage count on 202")
    func submitReturnsJobIdPlanAndStageCountOn202() async throws {
        let transport = FakeEngineTransport()
        let engine = makeTestEngine(transport: transport, outputDirectory: NSTemporaryDirectory())
        transport.script("/v1/video/director", [.init(202, """
            {"job_id":"DIR-1","status":"queued","mode":"director","backend":"ltx2-local","elapsed_ms":0,"progress_percent":0,"stage_count":2,"plan":\(planJSON)}
            """)])
        let submission = try await engine.submitDirectorJob(timeline: timeline(), outputPath: "/out/d.mp4")
        #expect(submission.jobId == "DIR-1")
        #expect(submission.stageCount == 2)
        #expect(submission.plan?.boundaryFrames == [288])
        #expect(submission.plan?.chunks.count == 2)
        let body = try #require(transport.body(of: "POST", "/v1/video/director"))
        #expect(body["source"] as? String == "desktop")
        #expect(body["output_path"] as? String == "/out/d.mp4")
    }

    @Test("a 400 with issues surfaces the issue messages")
    func submit400SurfacesIssuesMessage() async throws {
        let transport = FakeEngineTransport()
        let engine = makeTestEngine(transport: transport, outputDirectory: NSTemporaryDirectory())
        transport.script("/v1/video/director", [.init(400, """
            {"error":"timeline invalid","issues":[{"severity":"error","code":"keyframe_off_grid","message":"keyframe k1 frame 100 is not a multiple of 8","ids":["k1"]},{"severity":"warning","code":"no_keyframes","message":"no keyframe at frame 0","ids":[]}]}
            """)])
        do {
            _ = try await engine.submitDirectorJob(timeline: timeline(), outputPath: "/out/d.mp4")
            Issue.record("expected a throw")
        } catch let EngineServiceError.serverError(status, message) {
            #expect(status == 400)
            #expect(message.contains("timeline invalid"))
            #expect(message.contains("keyframe k1 frame 100 is not a multiple of 8"))
            #expect(!message.contains("no keyframe at frame 0"))
        }
    }

    @Test("validate parses the 200 body")
    func validateParses200() async throws {
        let transport = FakeEngineTransport()
        let engine = makeTestEngine(transport: transport, outputDirectory: NSTemporaryDirectory())
        transport.script("/v1/video/director/validate", [.init(200, """
            {"ok":true,"snapped_length_frames":577,"plan":\(planJSON),"issues":[{"severity":"warning","code":"generated_audio_seams","message":"seams","ids":[]}]}
            """)])
        let v = try await engine.validateDirectorTimeline(timeline())
        #expect(v.ok)
        #expect(v.snappedLengthFrames == 577)
        #expect(v.plan?.chunks.count == 2)
        #expect(v.issues.map(\.code) == ["generated_audio_seams"])
        let body = try #require(transport.body(of: "POST", "/v1/video/director/validate"))
        #expect(body["timeline"] is [String: Any])
        #expect(body["source"] == nil)

        transport.script("/v1/video/director/validate", [.init(200, """
            {"ok":false,"snapped_length_frames":577,"plan":null,"issues":[{"severity":"error","code":"missing_global_prompt","message":"m","ids":[]}]}
            """)])
        let bad = try await engine.validateDirectorTimeline(timeline())
        #expect(bad.ok == false)
        #expect(bad.plan == nil)
    }

    @Test("local validation matches the engine validator offline")
    func localValidation() {
        let transport = FakeEngineTransport()
        let engine = makeTestEngine(transport: transport, outputDirectory: NSTemporaryDirectory())
        let v = engine.localDirectorValidation(timeline(), fileExists: { _ in true }, audioProbe: { _ in nil })
        #expect(v.ok)
        #expect(v.snappedLengthFrames == 577)
        #expect(v.plan?.boundaryFrames == [288])
        #expect(transport.requests.isEmpty)
    }

    @Test("poll parses stage_index and stage_count")
    func pollStatusParsesStageIndexAndCount() async throws {
        let transport = FakeEngineTransport()
        let engine = makeTestEngine(transport: transport, outputDirectory: NSTemporaryDirectory())
        transport.script("/v1/video/status/DIR-1", [
            .init(200, #"{"job_id":"DIR-1","status":"queued","mode":"director","elapsed_ms":0,"stage_count":2}"#),
            .init(200, #"{"job_id":"DIR-1","status":"processing","mode":"director","elapsed_ms":10,"progress_percent":20,"stage_index":0,"stage_count":2}"#),
            .init(200, #"{"job_id":"DIR-1","status":"processing","mode":"director","elapsed_ms":20,"progress_percent":60,"stage_index":1,"stage_count":2}"#),
            .init(200, """
                {"job_id":"DIR-1","status":"succeeded","mode":"director","elapsed_ms":30,"progress_percent":100,"stage_index":2,"stage_count":2,"output_path":"/out/d.mp4","frame_count":577,"plan":\(planJSON)}
                """),
        ])
        var seen: [Int?] = []
        var counts: [Int?] = []
        var lastPlan: DirectorPlan?
        let result = try await engine.pollVideoStatus(jobId: "DIR-1", pollInterval: 0.001) { status in
            seen.append(status.stageIndex)
            counts.append(status.stageCount)
            if let p = status.plan { lastPlan = p }
        }
        #expect(seen == [nil, 0, 1, 2])
        #expect(counts == [2, 2, 2, 2])
        #expect(result.outputPath == "/out/d.mp4")
        #expect(result.frameCount == 577)
        #expect(lastPlan?.boundaryFrames == [288])
        #expect(EngineService.directorStatusLine(stageIndex: 0, stageCount: 2, progressPercent: 20) == "Chunk 1/2 · 20%")
        #expect(EngineService.directorStatusLine(stageIndex: 2, stageCount: 2, progressPercent: 97) == "Stitching…")
    }

    @Test("a failed poll throws immediately without retrying")
    func pollFailureThrowsImmediately() async throws {
        let transport = FakeEngineTransport()
        let engine = makeTestEngine(transport: transport, outputDirectory: NSTemporaryDirectory())
        transport.script("/v1/video/status/DIR-9", [.init(500, #"{"error":"engine restarted"}"#)])
        do {
            _ = try await engine.pollVideoStatus(jobId: "DIR-9", pollInterval: 0.001)
            Issue.record("expected a throw")
        } catch let EngineServiceError.serverError(status, message) {
            #expect(status == 500)
            #expect(message.contains("engine restarted"))
        }
        #expect(transport.requestCount("GET", "/v1/video/status/DIR-9") == 1)
    }
}
