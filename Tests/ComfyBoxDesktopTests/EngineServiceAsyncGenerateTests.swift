// EngineServiceAsyncGenerateTests.swift — comfybox#217
//
// `EngineService.generate()` used to hold a blocking `POST /v1/generate` open
// for the whole render, which pinned the engine's coordinator actor and starved
// the very `/health` polling that drives the Queue tab and the Generate tab's
// progress bar. It now submits to `POST /v1/generate/async` and polls
// `GET /v1/generate/status/{id}`, exactly as `generateVideo`'s async twin does.
//
// These drive that flow end-to-end against a FAKE `WarmServerTransport` — no
// engine, no weights, no GPU: submit → polling → completion, failure, refusal,
// and the cancel that names our own job id.

import Testing
import Foundation
import ZImage
@testable import ComfyBoxDesktop

/// Scripted `WarmServerTransport`. Records every request and answers from
/// per-path queues, so a test can make the status route return `processing`
/// twice before `succeeded` and assert the client really polled.
final class FakeEngineTransport: WarmServerTransport, @unchecked Sendable {
    struct Reply {
        let status: Int
        let body: Data
        init(_ status: Int, _ json: String) {
            self.status = status
            self.body = Data(json.utf8)
        }
    }

    private let lock = NSLock()
    private var scripted: [String: [Reply]] = [:]
    private var fallback: [String: Reply] = [:]
    private var recorded: [(method: String, path: String, body: Data)] = []

    /// Queue replies for a path, consumed in order. The LAST one repeats once
    /// the queue is exhausted.
    func script(_ path: String, _ replies: [Reply]) {
        lock.lock(); scripted[path] = replies; lock.unlock()
    }

    /// A single always-on reply for a path (health, preview, …).
    func always(_ path: String, _ reply: Reply) {
        lock.lock(); fallback[path] = reply; lock.unlock()
    }

    var requests: [(method: String, path: String, body: Data)] {
        lock.lock(); defer { lock.unlock() }; return recorded
    }

    func requestCount(_ method: String, _ path: String) -> Int {
        requests.filter { $0.method == method && $0.path == path }.count
    }

    func body(of method: String, _ path: String) -> [String: Any]? {
        guard let request = requests.last(where: { $0.method == method && $0.path == path }),
              let json = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any]
        else { return nil }
        return json
    }

    private func reply(method: String, path: String, body: Data) -> (Int, Data) {
        lock.lock(); defer { lock.unlock() }
        recorded.append((method, path, body))
        if var queue = scripted[path], !queue.isEmpty {
            let next = queue.count == 1 ? queue[0] : queue.removeFirst()
            scripted[path] = queue
            return (next.status, next.body)
        }
        if let stock = fallback[path] { return (stock.status, stock.body) }
        return (404, Data(#"{"error":"no fake reply scripted for \#(path)"}"#.utf8))
    }

    func get(_ path: String) async throws -> (Int, Data) { reply(method: "GET", path: path, body: Data()) }
    func post(_ path: String, body: Data) async throws -> (Int, Data) { reply(method: "POST", path: path, body: body) }
    func put(_ path: String, body: Data) async throws -> (Int, Data) { reply(method: "PUT", path: path, body: body) }
    func patch(_ path: String, body: Data) async throws -> (Int, Data) { reply(method: "PATCH", path: path, body: body) }
    func delete(_ path: String) async throws -> (Int, Data) { reply(method: "DELETE", path: path, body: Data()) }
    func send(method: String, path: String, body: Data, headers: [String: String]) async throws
        -> (Int, Data, [String: String]) {
        let (status, data) = reply(method: method, path: path, body: body)
        return (status, data, [:])
    }
}

@MainActor
private func makeEngine(_ transport: FakeEngineTransport, outputDirectory: String) -> EngineService {
    let engine = EngineService()
    engine.outputDirectory = outputDirectory
    engine.imageStatusPollInterval = 0.01
    // The in-render progress poll hits these; keep them satisfied so the fake
    // never pushes the service into an error connection state mid-test.
    transport.always("/health", .init(200, #"{"status":"ok","is_rendering":true,"progress_percent":40,"pending_count":0}"#))
    transport.always("/v1/generate/preview", .init(204, ""))
    engine.attachTransportForTesting(transport)
    return engine
}

private func scratchDirectory() -> String {
    let dir = NSTemporaryDirectory() + "comfybox-217-\(UUID().uuidString)"
    return dir
}

@MainActor
@Suite("EngineService async generate (#217)")
struct EngineServiceAsyncGenerateTests {

    // MARK: - Submit → poll → complete

    @Test("generate submits to /v1/generate/async and polls until succeeded")
    func submitsAsyncAndPolls() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"JOB-1","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/JOB-1", [
            .init(200, #"{"job_id":"JOB-1","status":"queued","source":"desktop","elapsed_ms":5}"#),
            .init(200, #"{"job_id":"JOB-1","status":"processing","source":"desktop","elapsed_ms":900}"#),
            .init(200, #"{"job_id":"JOB-1","status":"succeeded","source":"desktop","output_path":"/tmp/out.png","duration_ms":4211,"elapsed_ms":4300}"#),
        ])

        let path = try await engine.generate(GenerationRequest(prompt: "a cat"))

        #expect(path == "/tmp/out.png")
        #expect(engine.lastGeneratedImagePath == "/tmp/out.png")
        #expect(engine.lastDurationMs == 4211)
        #expect(engine.lastError == nil)
        // The blocking route must not be touched at all — that is the bug.
        #expect(transport.requestCount("POST", "/v1/generate") == 0)
        #expect(transport.requestCount("POST", "/v1/generate/async") == 1)
        #expect(transport.requestCount("GET", "/v1/generate/status/JOB-1") == 3)
        // Not generating any more, and the job id is released.
        #expect(!engine.isGenerating)
        #expect(engine.activeImageJobId == nil)
    }

    @Test("the submitted body is the same payload the blocking route received")
    func payloadUnchanged() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"J","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/J", [
            .init(200, #"{"job_id":"J","status":"succeeded","source":"desktop","output_path":"/tmp/x.png","elapsed_ms":1}"#)
        ])

        var request = GenerationRequest(prompt: "portrait", width: 1280, height: 1280, steps: 9)
        request.negativePrompt = "cropped"
        request.seed = 42
        request.sampler = "res_2s"
        request.sigmaSchedule = "krea2"
        _ = try await engine.generate(request, contentMode: .neutral)

        let body = try #require(transport.body(of: "POST", "/v1/generate/async"))
        #expect(body["source"] as? String == "desktop")
        #expect(body["prompt"] as? String == "portrait")
        #expect(body["width"] as? Int == 1280)
        #expect(body["steps"] as? Int == 9)
        #expect(body["seed"] as? UInt64 == 42)
        #expect(body["sampler"] as? String == "res_2s")
        #expect(body["sigma_schedule"] as? String == "krea2")
        #expect(body["negative_prompt"] as? String == "cropped")
        #expect(body["content_mode"] as? String == ComfyBoxDesktop.ContentMode.neutral.rawValue)
        #expect((body["outputPath"] as? String)?.hasSuffix(".png") == true)
    }

    /// `generatePayload` is the pure builder both routes share, so the async
    /// migration cannot have changed the wire shape by accident.
    @Test("generatePayload omits neutral RES4LYF fields and includes active ones")
    func payloadOmitsNeutralFields() {
        var request = GenerationRequest(prompt: "p")
        let neutral = EngineService.generatePayload(request, outputPath: "/tmp/a.png", contentMode: .neutral)
        #expect(neutral["eta"] == nil)
        #expect(neutral["bongmath"] == nil)
        #expect(neutral["noise_type"] == nil)
        #expect(neutral["projector_scale"] == nil)
        #expect(neutral["c2"] == nil)

        request.eta = 0.5
        request.bongmath = true
        request.projectorScale = 1.4
        request.c2 = 0.66
        let active = EngineService.generatePayload(request, outputPath: "/tmp/a.png", contentMode: .neutral)
        #expect(active["eta"] as? Float == 0.5)
        #expect(active["bongmath"] as? Bool == true)
        #expect(active["projector_scale"] as? Float == 1.4)
        #expect(active["c2"] as? Float == 0.66)
    }

    // MARK: - Failure shapes

    @Test("a failed job throws the engine's own message")
    func failedJobThrows() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"F","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/F", [
            .init(200, #"{"job_id":"F","status":"failed","source":"desktop","error":"LoRA not found: bogus.safetensors","elapsed_ms":30}"#)
        ])

        await #expect(throws: EngineServiceError.self) {
            _ = try await engine.generate(GenerationRequest(prompt: "x"))
        }
        #expect(engine.lastError == "LoRA not found: bogus.safetensors")
        #expect(engine.activeImageJobId == nil)
        #expect(!engine.isGenerating)
    }

    /// An operator interrupt reports itself as a failed job carrying the
    /// engine's interrupt sentence — the async path's spelling of what the
    /// blocking route surfaced as a 500 with the same message.
    @Test("an interrupted render surfaces the interrupt message")
    func interruptedJobSurfacesMessage() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"I","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/I", [
            .init(200, #"{"job_id":"I","status":"failed","source":"desktop","error":"Render interrupted by /v1/queue/interrupt","elapsed_ms":900}"#)
        ])

        await #expect(throws: EngineServiceError.self) {
            _ = try await engine.generate(GenerationRequest(prompt: "x"))
        }
        #expect(engine.lastError == "Render interrupted by /v1/queue/interrupt")
    }

    /// The engine refuses on the SUBMIT for anything its shared decode/validate
    /// choke point rejects — 409 preset/model conflict, 413 memory preflight,
    /// 429 queue full. The status and the message must survive unchanged.
    @Test("submit refusals keep their status and message", arguments: [
        (409, "Preset 'kira' names model 'krea2-raw' but the request names 'z-image-turbo'"),
        (413, "[image_memory_preflight] estimate 41.2GB exceeds the 38.0GB cap"),
        (429, "Queue full (10 pending max)"),
        (400, "Unknown sampler 'nope'"),
    ])
    func submitRefusalsPreserveShape(status: Int, message: String) async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        let body = try JSONSerialization.data(withJSONObject: ["success": false, "error": message])
        transport.script("/v1/generate/async", [.init(status, String(decoding: body, as: UTF8.self))])

        var thrown: EngineServiceError?
        do {
            _ = try await engine.generate(GenerationRequest(prompt: "x"))
        } catch let error as EngineServiceError {
            thrown = error
        }
        guard case .serverError(let gotStatus, let gotMessage)? = thrown else {
            Issue.record("expected .serverError, got \(String(describing: thrown))")
            return
        }
        #expect(gotStatus == status)
        #expect(gotMessage == message)
        #expect(engine.lastError == message)
        #expect(transport.requestCount("GET", "/v1/generate/status/") == 0)
    }

    @Test("an empty prompt is rejected before any request is made")
    func emptyPromptRejected() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        await #expect(throws: EngineServiceError.self) {
            _ = try await engine.generate(GenerationRequest(prompt: "   "))
        }
        #expect(transport.requestCount("POST", "/v1/generate/async") == 0)
    }

    @Test("generate without a connection throws notConnected")
    func notConnected() async throws {
        let engine = EngineService()
        await #expect(throws: EngineServiceError.self) {
            _ = try await engine.generate(GenerationRequest(prompt: "x"))
        }
    }

    // MARK: - Cancel

    /// comfybox#362 gave `/v1/queue/interrupt` a `target`; #217 gives the
    /// desktop a job id to put in it, so a cancel stops OUR render rather than
    /// whatever Bree or Kira happens to be rendering on the shared queue.
    @Test("cancelling an active desktop render targets its own job id")
    func cancelTargetsOwnJobId() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"MINE","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.always("/v1/queue/interrupt", .init(200, #"{"interrupted":true,"interrupted_job_id":"MINE","interrupted_kind":"generate"}"#))
        // Status stays `processing` until the test cancels, then reports the
        // interrupt — the real sequence.
        transport.script("/v1/generate/status/MINE", [
            .init(200, #"{"job_id":"MINE","status":"processing","source":"desktop","elapsed_ms":100}"#),
            .init(200, #"{"job_id":"MINE","status":"processing","source":"desktop","elapsed_ms":200}"#),
            .init(200, #"{"job_id":"MINE","status":"failed","source":"desktop","error":"Render interrupted by /v1/queue/interrupt","elapsed_ms":300}"#),
        ])

        let render = Task { try await engine.generate(GenerationRequest(prompt: "x")) }

        // Wait until the submit landed and the job id is published.
        var waited = 0
        while engine.activeImageJobId == nil && waited < 400 {
            try await Task.sleep(for: .milliseconds(5)); waited += 1
        }
        #expect(engine.activeImageJobId == "MINE")

        // The health poll shows our job as the active render, so cancel goes
        // through the interrupt route with an explicit target.
        engine.queueInfo = QueueInfo(
            isRendering: true, pendingCount: 0, renderCount: 0, uptimeSeconds: 1,
            lastRenderDurationMs: nil, lastError: nil, memoryUsageMB: 0,
            currentJobId: "MINE", progressPercent: 30)
        let cancelled = try await engine.cancelActiveGeneration()
        #expect(cancelled)

        let interruptBody = try #require(transport.body(of: "POST", "/v1/queue/interrupt"))
        #expect(interruptBody["target"] as? String == "MINE")

        await #expect(throws: EngineServiceError.self) { _ = try await render.value }
    }

    /// A job that has NOT started yet is not the active render: interrupting it
    /// would 404 (an explicit target naming nothing running), so it is cancelled
    /// through `DELETE /v1/queue/{id}` instead.
    @Test("cancelling a still-queued desktop job deletes it from the queue")
    func cancelQueuedJobUsesDelete() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"PEND","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.always("/v1/queue/PEND", .init(202, #"{"accepted":true,"id":"PEND","note":"cancel recorded"}"#))
        transport.script("/v1/generate/status/PEND", [
            .init(200, #"{"job_id":"PEND","status":"queued","source":"desktop","elapsed_ms":10}"#),
            .init(200, #"{"job_id":"PEND","status":"failed","source":"desktop","error":"Request cancelled (queue cleared)","elapsed_ms":20}"#),
        ])

        let render = Task { try await engine.generate(GenerationRequest(prompt: "x")) }
        var waited = 0
        while engine.activeImageJobId == nil && waited < 400 {
            try await Task.sleep(for: .milliseconds(5)); waited += 1
        }

        // Someone else's render is active; ours is still pending.
        engine.queueInfo = QueueInfo(
            isRendering: true, pendingCount: 1, renderCount: 0, uptimeSeconds: 1,
            lastRenderDurationMs: nil, lastError: nil, memoryUsageMB: 0,
            currentJobId: "SOMEONE-ELSE", progressPercent: 10)
        #expect(try await engine.cancelActiveGeneration())
        #expect(transport.requestCount("DELETE", "/v1/queue/PEND") == 1)
        #expect(transport.requestCount("POST", "/v1/queue/interrupt") == 0)

        await #expect(throws: EngineServiceError.self) { _ = try await render.value }
    }

    @Test("cancelActiveGeneration is a no-op with nothing in flight")
    func cancelNoOp() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        #expect(try await engine.cancelActiveGeneration() == false)
        #expect(transport.requestCount("POST", "/v1/queue/interrupt") == 0)
    }

    /// The Queue tab's stop button keeps the legacy default target: an absent
    /// `target` means "whatever /health shows as active", byte-identical to the
    /// pre-#362 body every other client still sends.
    @Test("interruptRender without a target sends the legacy empty body")
    func interruptDefaultTargetUnchanged() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.always("/v1/queue/interrupt", .init(200, #"{"interrupted":false}"#))

        try await engine.interruptRender()

        let request = try #require(transport.requests.last(where: { $0.path == "/v1/queue/interrupt" }))
        #expect(String(decoding: request.body, as: UTF8.self) == "{}")
    }

    // MARK: - Wire parsing

    @Test("image job status parses the additive fields the engine adds")
    func statusParsesAdditiveFields() throws {
        let data = Data(#"""
        {"job_id":"A","status":"succeeded","source":"desktop","output_path":"/tmp/a.png",
         "duration_ms":1000,"elapsed_ms":1100,"preset_unresolved":"kira",
         "preset_unresolved_reason":"no_model","preempt_refused":true,"eta_sec":12.5,
         "brand_new_field_from_a_future_release":123}
        """#.utf8)
        let job = try #require(EngineService.parseImageJobStatus(data))
        #expect(job.jobId == "A")
        #expect(job.isTerminal)
        #expect(job.outputPath == "/tmp/a.png")
        #expect(job.presetUnresolved == "kira")
        #expect(job.presetUnresolvedReason == "no_model")
        #expect(job.preemptRefused == true)
    }

    @Test("a malformed status body is a server error, not a crash")
    func malformedStatusBody() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"M","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/M", [.init(200, "not json at all")])

        await #expect(throws: EngineServiceError.self) {
            _ = try await engine.generate(GenerationRequest(prompt: "x"))
        }
    }
}
