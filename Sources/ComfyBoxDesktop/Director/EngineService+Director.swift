// EngineService+Director.swift — the desktop's Director client (WP3).
//
// POST /v1/video/director (202 job model) and POST /v1/video/director/validate
// (pure), per docs/FDD-ltx-director-tab.md §4.3 and the Phase 1 route
// contract. Progress rides the unchanged GET /v1/video/status/{id} through
// `pollVideoStatus`, whose VideoJobStatus now also carries stage_index /
// stage_count / plan. Wire keys are snake_case via DirectorJSON; the timeline
// bytes are identical to a .cbdirector file.

import Foundation
import ZImage

extension EngineService {

    /// What the 202 of POST /v1/video/director told us.
    struct DirectorSubmission: Sendable, Equatable {
        let jobId: String
        let plan: DirectorPlan?
        let stageCount: Int?
    }

    /// A validation result, from the server (/validate) or in-process.
    struct DirectorValidation: Sendable, Equatable {
        let ok: Bool
        let snappedLengthFrames: Int
        let plan: DirectorPlan?
        let issues: [DirectorIssue]
    }

    private struct DirectorEnvelope: Encodable {
        let timeline: DirectorTimeline
        let outputPath: String?
        let source: String?
    }

    private struct DirectorValidateEnvelope: Encodable {
        let timeline: DirectorTimeline
    }

    private struct DirectorSubmitReply: Decodable {
        let jobId: String
        let plan: DirectorPlan?
        let stageCount: Int?
    }

    private struct DirectorValidateReply: Decodable {
        let ok: Bool
        let snappedLengthFrames: Int
        let plan: DirectorPlan?
        let issues: [DirectorIssue]
    }

    private struct DirectorErrorReply: Decodable {
        let error: String?
        let issues: [DirectorIssue]?
    }

    /// `{timeline, output_path, source: "desktop"}`. The engine reads a
    /// keyframe image by path whenever that path exists, so `image_base64` is
    /// stripped for those keyframes; a keyframe whose path does not exist here
    /// (a portable .cbdirector saved with "Embed assets" on another machine)
    /// keeps its payload, which the server materializes to a file.
    nonisolated static func directorRequestBody(
        timeline: DirectorTimeline, outputPath: String,
        fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }
    ) throws -> Data {
        try DirectorJSON.encoder(pretty: false).encode(
            DirectorEnvelope(timeline: strippingEmbeddedAssets(timeline, fileExists: fileExists), outputPath: outputPath, source: "desktop"))
    }

    nonisolated static func directorValidateBody(
        timeline: DirectorTimeline,
        fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }
    ) throws -> Data {
        try DirectorJSON.encoder(pretty: false).encode(
            DirectorValidateEnvelope(timeline: strippingEmbeddedAssets(timeline, fileExists: fileExists)))
    }

    /// Drop `image_base64` only where the path it duplicates exists locally;
    /// without a readable path the payload is the image.
    nonisolated static func strippingEmbeddedAssets(
        _ timeline: DirectorTimeline, fileExists: (String) -> Bool
    ) -> DirectorTimeline {
        var t = timeline
        for i in t.keyframes.indices {
            guard let path = t.keyframes[i].imagePath, !path.isEmpty, fileExists(path) else { continue }
            t.keyframes[i].imageBase64 = nil
        }
        return t
    }

    /// A readable error for a non-2xx director reply: the `error` string plus
    /// every error-severity issue message (warnings are left out).
    nonisolated static func directorErrorMessage(from data: Data, fallback: String) -> String {
        guard let reply = try? DirectorJSON.decoder().decode(DirectorErrorReply.self, from: data) else { return fallback }
        let head = reply.error ?? fallback
        let errors = (reply.issues ?? []).filter { $0.severity == .error }.map(\.message)
        return errors.isEmpty ? head : "\(head): \(errors.joined(separator: "; "))"
    }

    /// Decode a `plan` object out of a JSONSerialization tree (status polls).
    nonisolated static func decodeDirectorPlan(_ object: Any?) -> DirectorPlan? {
        guard let object, object is [String: Any],
              let data = try? JSONSerialization.data(withJSONObject: object)
        else { return nil }
        return try? DirectorJSON.decoder().decode(DirectorPlan.self, from: data)
    }

    /// Stage-aware status text for a director poll.
    nonisolated static func directorStatusLine(stageIndex: Int?, stageCount: Int?, progressPercent: Int?) -> String {
        if let index = stageIndex, let count = stageCount, index >= count {
            return "Stitching…"
        }
        var parts: [String] = []
        if let index = stageIndex, let count = stageCount {
            parts.append("Chunk \(index + 1)/\(count)")
        }
        if let pct = progressPercent { parts.append("\(pct)%") }
        return parts.isEmpty ? "Queued…" : parts.joined(separator: " · ")
    }

    /// Submit a director render. Expects 202; a 400 carries the validator's
    /// issues, surfaced in the thrown message.
    func submitDirectorJob(timeline: DirectorTimeline, outputPath: String) async throws -> DirectorSubmission {
        guard let client = transport, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let body = try Self.directorRequestBody(timeline: timeline, outputPath: outputPath)
        let (status, data) = try await client.post("/v1/video/director", body: body)
        guard status == 202, let reply = try? DirectorJSON.decoder().decode(DirectorSubmitReply.self, from: data) else {
            throw EngineServiceError.serverError(
                status, Self.directorErrorMessage(from: data, fallback: "Director submit failed (is the server started with --ltx2-weights?)"))
        }
        return DirectorSubmission(jobId: reply.jobId, plan: reply.plan, stageCount: reply.stageCount)
    }

    /// Server-side validation (same validator, plus the server's file view).
    func validateDirectorTimeline(_ timeline: DirectorTimeline) async throws -> DirectorValidation {
        guard let client = transport, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let body = try Self.directorValidateBody(timeline: timeline)
        let (status, data) = try await client.post("/v1/video/director/validate", body: body)
        guard status == 200, let reply = try? DirectorJSON.decoder().decode(DirectorValidateReply.self, from: data) else {
            throw EngineServiceError.serverError(status, Self.directorErrorMessage(from: data, fallback: "Director validate failed"))
        }
        return DirectorValidation(ok: reply.ok, snappedLengthFrames: reply.snappedLengthFrames, plan: reply.plan, issues: reply.issues)
    }

    /// In-process validation — the offline path. Same DirectorValidator the
    /// server runs; the audio probe is AVFoundation-local too.
    func localDirectorValidation(
        _ timeline: DirectorTimeline,
        fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) },
        audioProbe: (String) -> AudioProbe? = DirectorValidator.defaultAudioProbe
    ) -> DirectorValidation {
        let v = DirectorValidator.validate(timeline, fileExists: fileExists, audioProbe: audioProbe)
        return DirectorValidation(ok: v.ok, snappedLengthFrames: v.snapped.settings.lengthFrames, plan: v.plan, issues: v.issues)
    }
}
