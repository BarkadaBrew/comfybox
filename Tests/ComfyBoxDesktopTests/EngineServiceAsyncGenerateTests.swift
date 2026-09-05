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

    /// Make a path hang forever (until the caller's Task is cancelled) — a
    /// wedged engine, which is exactly the state a user cancels from.
    func hangs(_ path: String) {
        lock.lock(); hanging.insert(path); lock.unlock()
    }
    private var hanging: Set<String> = []
    private func isHanging(_ path: String) -> Bool {
        lock.lock(); defer { lock.unlock() }; return hanging.contains(path)
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

    /// Record the request, then never answer — until cancelled, like a real
    /// URLSession request against a wedged server.
    private func hang() async throws -> Never {
        try await Task.sleep(for: .seconds(600))
        throw CancellationError()
    }

    func get(_ path: String) async throws -> (Int, Data) {
        let out = reply(method: "GET", path: path, body: Data())
        if isHanging(path) { try await hang() }
        return out
    }
    func post(_ path: String, body: Data) async throws -> (Int, Data) {
        let out = reply(method: "POST", path: path, body: body)
        if isHanging(path) { try await hang() }
        return out
    }
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

    /// What actually lands on the SUBMIT: only what the engine's shared
    /// decode/validate choke point (`decodedGenerateRequest`) rejects BEFORE the
    /// 202 — a 400 bad recipe name, a 409 preset/model conflict, a 413 memory
    /// preflight refusal. These keep their status AND message, so the mapping to
    /// `.serverError(status, message)` is identical to the blocking route's.
    @Test("submit-time refusals keep their status and message", arguments: [
        (409, "Preset 'kira' names model 'krea2-raw' but the request names 'z-image-turbo'"),
        (413, "[image_memory_preflight] estimate 41.2GB exceeds the 38.0GB cap"),
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
    }

    /// The refusals that do NOT land on the submit, contrary to what this PR's
    /// first description claimed. `ServerError.queueFull` (429) and
    /// `ServerError.cancelled` (409, queue cleared) are thrown inside
    /// `enqueueGenerate` — which the async route calls from `ImageJobTracker`'s
    /// own detached Task, AFTER the 202 has already gone out. They therefore
    /// arrive as a FAILED JOB carrying the engine's message, and map to
    /// `.generationFailed`, not `.serverError(429/409, …)`.
    @Test("post-202 refusals arrive as a failed job, not a submit error", arguments: [
        "Queue full (10 pending max)",
        "Request cancelled (queue cleared)",
        "Server is shutting down",
    ])
    func postSubmitRefusalsArriveAsFailedJobs(message: String) async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"Q","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        let status = try JSONSerialization.data(withJSONObject: [
            "job_id": "Q", "status": "failed", "source": "desktop", "error": message, "elapsed_ms": 12,
        ])
        transport.script("/v1/generate/status/Q", [.init(200, String(decoding: status, as: UTF8.self))])

        var thrown: EngineServiceError?
        do {
            _ = try await engine.generate(GenerationRequest(prompt: "x"))
        } catch let error as EngineServiceError {
            thrown = error
        }
        guard case .generationFailed(let gotMessage)? = thrown else {
            Issue.record("expected .generationFailed, got \(String(describing: thrown))")
            return
        }
        #expect(gotMessage == message)
        #expect(engine.lastError == message)
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

        // The route is DISCOVERED, not guessed from queueInfo: the targeted
        // interrupt is tried first and it succeeds, so no queue-delete is sent.
        #expect(try await engine.cancelActiveGeneration() == .interrupted(jobId: "MINE"))

        let interruptBody = try #require(transport.body(of: "POST", "/v1/queue/interrupt"))
        #expect(interruptBody["target"] as? String == "MINE")
        #expect(transport.requestCount("DELETE", "/v1/queue/MINE") == 0)

        // A cancel the USER asked for unwinds as `.cancelled`, not as a render
        // failure, and raises no error banner.
        var thrown: Error?
        do { _ = try await render.value } catch { thrown = error }
        guard case EngineServiceError.cancelled(let id)? = thrown as? EngineServiceError else {
            Issue.record("expected .cancelled, got \(String(describing: thrown))")
            return
        }
        #expect(id == "MINE")
        #expect(engine.lastError == nil)
    }

    /// An interrupt somebody ELSE issued is a real failure the user should see —
    /// the same status body, but no cancel of ours preceding it.
    @Test("an interrupt we did not ask for stays a generation failure")
    func foreignInterruptStaysAnError() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"THEIRS","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/THEIRS", [
            .init(200, #"{"job_id":"THEIRS","status":"failed","source":"desktop","error":"Render interrupted by /v1/queue/interrupt","elapsed_ms":300}"#)
        ])

        var thrown: Error?
        do { _ = try await engine.generate(GenerationRequest(prompt: "x")) } catch { thrown = error }
        guard case EngineServiceError.generationFailed(let message)? = thrown as? EngineServiceError else {
            Issue.record("expected .generationFailed, got \(String(describing: thrown))")
            return
        }
        #expect(message == "Render interrupted by /v1/queue/interrupt")
        #expect(engine.lastError == message)
    }

    /// A job that has NOT started yet is not the active render, so the targeted
    /// interrupt 404s — and only THEN does the cancel fall back to
    /// `DELETE /v1/queue/{id}`. Discovering the route beats guessing it from
    /// `queueInfo`, which health only refreshes every 700ms-3s.
    @Test("cancelling a still-queued desktop job falls back to a queue delete")
    func cancelQueuedJobUsesDelete() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"PEND","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.always("/v1/queue/PEND", .init(202, #"{"accepted":true,"id":"PEND","note":"cancel recorded"}"#))
        transport.always("/v1/queue/interrupt", .init(404, #"{"error":"No such interrupt target: PEND"}"#))
        transport.script("/v1/generate/status/PEND", [
            .init(200, #"{"job_id":"PEND","status":"queued","source":"desktop","elapsed_ms":10}"#),
            .init(200, #"{"job_id":"PEND","status":"failed","source":"desktop","error":"Request cancelled (queue cleared)","elapsed_ms":20}"#),
        ])

        let render = Task { try await engine.generate(GenerationRequest(prompt: "x")) }
        var waited = 0
        while engine.activeImageJobId == nil && waited < 400 {
            try await Task.sleep(for: .milliseconds(5)); waited += 1
        }

        #expect(try await engine.cancelActiveGeneration() == .dequeued(jobId: "PEND"))
        // Interrupt attempted FIRST, then the delete once it 404'd.
        #expect(transport.requestCount("POST", "/v1/queue/interrupt") == 1)
        #expect(transport.requestCount("DELETE", "/v1/queue/PEND") == 1)

        await #expect(throws: EngineServiceError.self) { _ = try await render.value }
    }

    /// comfybox#378: a 200 carrying `interrupted: false` stopped NOTHING — the
    /// render finished between reading /health and pressing the button. That is
    /// not a successful cancel and must not be reported as one.
    @Test("an interrupt that stopped nothing reports alreadyFinished")
    func interruptThatStoppedNothingIsReported() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"DONE","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.always("/v1/queue/interrupt", .init(200, #"{"interrupted":false}"#))
        transport.always("/v1/generate/status/DONE", .init(200, #"{"job_id":"DONE","status":"processing","source":"desktop","elapsed_ms":10}"#))

        let render = Task { try await engine.generate(GenerationRequest(prompt: "x")) }
        var waited = 0
        while engine.activeImageJobId == nil && waited < 400 {
            try await Task.sleep(for: .milliseconds(5)); waited += 1
        }

        #expect(try await engine.cancelActiveGeneration() == .alreadyFinished(jobId: "DONE"))
        #expect(transport.requestCount("DELETE", "/v1/queue/DONE") == 0, "a 200 is not the 404 that triggers the fallback")
        render.cancel()
        _ = try? await render.value
    }

    /// A 404 aimed at a stale id, with no queued job to delete either, must say
    /// which of the two situations it is rather than "Interrupt failed".
    @Test("a stale interrupt target gets a message that explains itself")
    func staleInterruptTargetIsExplained() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.always("/v1/queue/interrupt", .init(404, ""))

        var thrown: EngineServiceError?
        do { _ = try await engine.interruptRender(target: "STALE") }
        catch let error as EngineServiceError { thrown = error }

        guard case .serverError(let status, let message)? = thrown else {
            Issue.record("expected .serverError, got \(String(describing: thrown))")
            return
        }
        #expect(status == 404)
        #expect(message.contains("STALE"))
        #expect(message.contains("still queued"))
    }

    @Test("cancelActiveGeneration is a no-op with nothing in flight")
    func cancelNoOp() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        #expect(try await engine.cancelActiveGeneration() == .nothingInFlight)
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

    // MARK: - Poll-loop resilience (PR #384 review, item 3)

    /// A blip is not a dead engine: consecutive transient failures are tolerated
    /// up to the limit, and a later good poll resets the count.
    @Test("transient status failures are retried and a good poll resets the count")
    func transientFailuresAreTolerated() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        engine.imageStatusTransientFailureBudget = 0.2
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"T","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/T", [
            .init(503, #"{"error":"engine busy"}"#),
            .init(500, "garbage"),
            .init(200, #"{"job_id":"T","status":"processing","source":"desktop","elapsed_ms":50}"#),  // resets
            .init(502, #"{"error":"blip"}"#),
            .init(200, #"{"job_id":"T","status":"succeeded","source":"desktop","output_path":"/tmp/t.png","elapsed_ms":90}"#),
        ])

        let path = try await engine.generate(GenerationRequest(prompt: "x"))
        #expect(path == "/tmp/t.png")
        #expect(transport.requestCount("GET", "/v1/generate/status/T") == 5)
    }

    /// …but it must not retry forever. Once consecutive failure has lasted the
    /// whole BUDGET the render is declared lost, naming the last error. The
    /// budget is injected short so this test costs milliseconds, not a minute.
    @Test("the poll loop gives up once the transient-failure budget is spent")
    func transientFailuresEventuallyGiveUp() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        engine.imageStatusTransientFailureBudget = 0.15
        engine.imageStatusPollInterval = 0.02
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"D","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/D", [.init(503, #"{"error":"engine unreachable"}"#)])

        let started = Date()
        var thrown: EngineServiceError?
        do { _ = try await engine.generate(GenerationRequest(prompt: "x")) }
        catch let error as EngineServiceError { thrown = error }
        let elapsed = Date().timeIntervalSince(started)

        guard case .generationFailed(let message)? = thrown else {
            Issue.record("expected .generationFailed, got \(String(describing: thrown))")
            return
        }
        #expect(message.contains("engine unreachable"))
        #expect(message.contains("budget"))
        // It retried for about the budget — several polls, not one, not forever.
        #expect(transport.requestCount("GET", "/v1/generate/status/D") > 1)
        #expect(elapsed >= 0.15)
        #expect(elapsed < 5.0, "the budget bounds it; it must not hang")
    }

    /// The budget is WALL-CLOCK, not a poll count: a long poll interval must not
    /// silently shrink the tolerance to a couple of seconds.
    @Test("the transient budget survives a poll interval longer than one retry")
    func transientBudgetIsWallClockNotPollCount() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        engine.imageStatusTransientFailureBudget = 0.3
        engine.imageStatusPollInterval = 0.1
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"WC","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        // Fails for ~0.2s (2 polls), then recovers well inside the budget.
        transport.script("/v1/generate/status/WC", [
            .init(503, #"{"error":"blip"}"#),
            .init(503, #"{"error":"blip"}"#),
            .init(200, #"{"job_id":"WC","status":"succeeded","source":"desktop","output_path":"/tmp/wc.png","elapsed_ms":300}"#),
        ])

        let path = try await engine.generate(GenerationRequest(prompt: "x"))
        #expect(path == "/tmp/wc.png")
    }

    /// A 404 is not transient: the engine does not know the id (pruned, or it
    /// restarted). Retrying can only 404 again, so it fails at once.
    @Test("a 404 from the status route fails immediately with a clear message")
    func unknownJobFailsImmediately() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        engine.imageStatusTransientFailureBudget = 5.0
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"GONE","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/GONE", [.init(404, #"{"error":"Image job not found: GONE"}"#)])

        var thrown: EngineServiceError?
        do { _ = try await engine.generate(GenerationRequest(prompt: "x")) }
        catch let error as EngineServiceError { thrown = error }

        guard case .serverError(let status, let message)? = thrown else {
            Issue.record("expected .serverError, got \(String(describing: thrown))")
            return
        }
        #expect(status == 404)
        #expect(message.contains("Image job not found: GONE"))
        #expect(transport.requestCount("GET", "/v1/generate/status/GONE") == 1, "no retry on 404")
    }

    /// An unrecognised non-terminal state must not be spun on forever.
    @Test("an unrecognised job state is reported, not spun on")
    func unknownStateIsReported() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"W","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/W", [.init(200, #"{"job_id":"W","status":"hibernating","source":"desktop","elapsed_ms":1}"#)])

        var thrown: EngineServiceError?
        do { _ = try await engine.generate(GenerationRequest(prompt: "x")) }
        catch let error as EngineServiceError { thrown = error }

        guard case .generationFailed(let message)? = thrown else {
            Issue.record("expected .generationFailed, got \(String(describing: thrown))")
            return
        }
        #expect(message.contains("hibernating"))
        #expect(transport.requestCount("GET", "/v1/generate/status/W") == 1)
    }

    /// Cancelling the owning Task must not leave the GPU burning: the server job
    /// is cancelled best-effort, and the caller sees `.cancelled`, not a failure.
    @Test("Task cancellation cancels the server job and throws .cancelled")
    func taskCancellationCancelsTheServerJob() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        engine.imageStatusPollInterval = 0.05
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"CANCELME","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.always("/v1/generate/status/CANCELME", .init(200, #"{"job_id":"CANCELME","status":"processing","source":"desktop","elapsed_ms":10}"#))
        transport.always("/v1/queue/interrupt", .init(200, #"{"interrupted":true,"interrupted_job_id":"CANCELME"}"#))

        let render = Task { try await engine.generate(GenerationRequest(prompt: "x")) }
        var waited = 0
        while engine.activeImageJobId == nil && waited < 400 {
            try await Task.sleep(for: .milliseconds(5)); waited += 1
        }
        // Our job is the active render, so the cancel goes via the targeted
        // interrupt rather than a queue delete.
        engine.queueInfo = QueueInfo(
            isRendering: true, pendingCount: 0, renderCount: 0, uptimeSeconds: 1,
            lastRenderDurationMs: nil, lastError: nil, memoryUsageMB: 0,
            currentJobId: "CANCELME", progressPercent: 20)

        render.cancel()

        var thrown: Error?
        do { _ = try await render.value } catch { thrown = error }
        guard case EngineServiceError.cancelled(let jobId)? = thrown as? EngineServiceError else {
            Issue.record("expected .cancelled, got \(String(describing: thrown))")
            return
        }
        #expect(jobId == "CANCELME")
        #expect(transport.body(of: "POST", "/v1/queue/interrupt")?["target"] as? String == "CANCELME")
        // A cancel is not an error the UI should shout about.
        #expect(engine.lastError == nil)
    }

    /// A user cancels precisely when the engine is wedged. Awaiting the
    /// best-effort server cancel unbounded meant inheriting `WarmServerClient`'s
    /// 300s timeout, so `generate()` (and `isGenerating`, and the Cancel
    /// button's spinner) stayed stuck for five minutes. The wait is now bounded
    /// by `cancelBestEffortTimeout` (PR #384 review r2, item 1).
    @Test("cancelling against a wedged engine returns promptly, not after the HTTP timeout")
    func cancelAgainstAWedgedEngineReturnsPromptly() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        engine.imageStatusPollInterval = 0.02
        engine.cancelBestEffortTimeout = 0.15
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"WEDGED","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.always("/v1/generate/status/WEDGED", .init(200, #"{"job_id":"WEDGED","status":"processing","source":"desktop","elapsed_ms":10}"#))
        // The interrupt request is recorded and then never answered.
        transport.always("/v1/queue/interrupt", .init(200, #"{"interrupted":true}"#))
        transport.hangs("/v1/queue/interrupt")

        let render = Task { try await engine.generate(GenerationRequest(prompt: "x")) }
        var waited = 0
        while engine.activeImageJobId == nil && waited < 400 {
            try await Task.sleep(for: .milliseconds(5)); waited += 1
        }

        let started = Date()
        render.cancel()
        var thrown: Error?
        do { _ = try await render.value } catch { thrown = error }
        let elapsed = Date().timeIntervalSince(started)

        guard case EngineServiceError.cancelled? = thrown as? EngineServiceError else {
            Issue.record("expected .cancelled, got \(String(describing: thrown))")
            return
        }
        // Bounded by the budget, nowhere near WarmServerClient's 300s timeout.
        #expect(elapsed < 3.0, "took \(elapsed)s — the cancel must not inherit the HTTP timeout")
        // It still TRIED to stop the server job.
        #expect(transport.requestCount("POST", "/v1/queue/interrupt") == 1)
        #expect(!engine.isGenerating, "isGenerating must be released promptly")
    }

    /// PR #384 review r3, item 2: the CANCEL BUTTON's path
    /// (`cancelActiveGeneration` → `cancelImageJob` → `interruptRender`) must be
    /// bounded too, not just the poll loop's Task-cancellation path. A wedged
    /// engine would otherwise freeze the button on `WarmServerClient`'s 300s
    /// request timeout.
    @Test("the Cancel button path returns promptly against a wedged engine")
    func cancelButtonPathIsBounded() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        engine.imageStatusPollInterval = 0.02
        engine.cancelBestEffortTimeout = 0.15
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"STUCK","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.always("/v1/generate/status/STUCK", .init(200, #"{"job_id":"STUCK","status":"processing","source":"desktop","elapsed_ms":10}"#))
        transport.always("/v1/queue/interrupt", .init(200, #"{"interrupted":true}"#))
        transport.hangs("/v1/queue/interrupt")

        let render = Task { try await engine.generate(GenerationRequest(prompt: "x")) }
        var waited = 0
        while engine.activeImageJobId == nil && waited < 400 {
            try await Task.sleep(for: .milliseconds(5)); waited += 1
        }

        let started = Date()
        let result = try await engine.cancelActiveGeneration()
        let elapsed = Date().timeIntervalSince(started)

        #expect(result == .abandoned(jobId: "STUCK"), "the request was sent; the engine never answered")
        #expect(elapsed < 3.0, "took \(elapsed)s — the button must not inherit the HTTP timeout")
        #expect(result.message != nil, "an abandoned cancel must say so")
        #expect(transport.requestCount("POST", "/v1/queue/interrupt") == 1)

        render.cancel()
        _ = try? await render.value
    }

    /// PR #384 review r3, item 3: the job finished in the window between the
    /// interrupt (404 — not the active render) and the queue delete. Reporting
    /// "Cancel failed: Server error (404)" is a lie; nothing was left to cancel.
    @Test("a job that finishes between interrupt and delete reports alreadyFinished")
    func raceBetweenInterruptAndDeleteIsAlreadyFinished() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"RACE","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.always("/v1/generate/status/RACE", .init(200, #"{"job_id":"RACE","status":"processing","source":"desktop","elapsed_ms":10}"#))
        transport.always("/v1/queue/interrupt", .init(404, #"{"error":"No such interrupt target: RACE"}"#))
        transport.always("/v1/queue/RACE", .init(404, #"{"error":"Job not found"}"#))

        let render = Task { try await engine.generate(GenerationRequest(prompt: "x")) }
        var waited = 0
        while engine.activeImageJobId == nil && waited < 400 {
            try await Task.sleep(for: .milliseconds(5)); waited += 1
        }

        #expect(try await engine.cancelActiveGeneration() == .alreadyFinished(jobId: "RACE"))
        #expect(transport.requestCount("POST", "/v1/queue/interrupt") == 1)
        #expect(transport.requestCount("DELETE", "/v1/queue/RACE") == 1)

        render.cancel()
        _ = try? await render.value
    }

    /// PR #384 review r3, item 4: the QUEUE TAB's per-row stop of our own active
    /// render is a user cancel too — `interruptRender` records it, so no surface
    /// can forget and the generate loop unwinds as `.cancelled`.
    @Test("a Queue-tab interrupt of our own job unwinds as .cancelled")
    func queueTabInterruptOfOwnJobUnwindsAsCancelled() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        engine.imageStatusPollInterval = 0.02
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"OURS","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.always("/v1/queue/interrupt", .init(200, #"{"interrupted":true,"interrupted_job_id":"OURS"}"#))
        transport.script("/v1/generate/status/OURS", [
            .init(200, #"{"job_id":"OURS","status":"processing","source":"desktop","elapsed_ms":10}"#),
            .init(200, #"{"job_id":"OURS","status":"failed","source":"desktop","error":"Render interrupted by /v1/queue/interrupt","elapsed_ms":300}"#),
        ])

        let render = Task { try await engine.generate(GenerationRequest(prompt: "x")) }
        var waited = 0
        while engine.activeImageJobId == nil && waited < 400 {
            try await Task.sleep(for: .milliseconds(5)); waited += 1
        }

        // Exactly what QueueView/QueuePanel do: interrupt the row's job id.
        _ = try await engine.interruptRender(target: "OURS")

        var thrown: Error?
        do { _ = try await render.value } catch { thrown = error }
        guard case EngineServiceError.cancelled? = thrown as? EngineServiceError else {
            Issue.record("expected .cancelled, got \(String(describing: thrown))")
            return
        }
        #expect(engine.lastError == nil, "a stop the user pressed is not an error")
    }

    /// PR #384 review r4, item 1: the QUEUE TAB's stop went straight to
    /// `interruptRender`, so it was unbounded — a wedged engine held the button
    /// on `WarmServerClient`'s 300s timeout — and had no dequeued fallback. Both
    /// surfaces now call `cancelRender(jobId:)`, which is what this drives.
    @Test("the queue-tab stop path is bounded against a wedged engine")
    func queueTabStopIsBounded() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        engine.cancelBestEffortTimeout = 0.15
        transport.always("/v1/queue/interrupt", .init(200, #"{"interrupted":true}"#))
        transport.hangs("/v1/queue/interrupt")

        let started = Date()
        let result = try await engine.cancelRender(jobId: "SOMEONE-ELSE")
        let elapsed = Date().timeIntervalSince(started)

        #expect(result == .abandoned(jobId: "SOMEONE-ELSE"))
        #expect(elapsed < 3.0, "took \(elapsed)s — the queue-tab stop must not inherit the HTTP timeout")
        #expect(transport.requestCount("POST", "/v1/queue/interrupt") == 1)
    }

    /// The untargeted default ("stop whatever is active", used when the engine
    /// reports no active job id at all) goes through the same bounded path and
    /// reports the same way.
    @Test("an untargeted stop is bounded and reports alreadyFinished when nothing stopped")
    func untargetedStopIsBoundedAndHonest() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.always("/v1/queue/interrupt", .init(200, #"{"interrupted":false}"#))

        let result = try await engine.cancelRender(jobId: nil)
        #expect(result == .alreadyFinished(jobId: "active"))
        #expect(result.message == EngineService.InterruptOutcome.alreadyFinishedMessage)
        // No target field: the legacy body, byte-identical to every other client.
        let request = try #require(transport.requests.last(where: { $0.path == "/v1/queue/interrupt" }))
        #expect(String(decoding: request.body, as: UTF8.self) == "{}")
        // Nothing to fall back to without an id — no stray queue delete.
        #expect(transport.requests.filter { $0.method == "DELETE" }.isEmpty)
    }

    /// PR #384 review r4, item 3: `/v1/queue` can report no `active_job_id`. When
    /// it also says the active render is OURS, our own in-flight id is the right
    /// target — otherwise the stop falls back to the untargeted default and our
    /// render unwinds as a failure banner instead of a cancel.
    @Test("interruptTarget falls back to our own job only when the render is ours")
    func interruptTargetFallback() {
        // The engine told us the id — always use it.
        #expect(EngineService.interruptTarget(
            activeJobId: "ENGINE", activeSource: "bree", ourInFlightJobId: "OURS") == "ENGINE")
        // No id, but the queue says the active render came from this app.
        #expect(EngineService.interruptTarget(
            activeJobId: nil, activeSource: "desktop", ourInFlightJobId: "OURS") == "OURS")
        #expect(EngineService.interruptTarget(
            activeJobId: "", activeSource: "Desktop", ourInFlightJobId: "OURS") == "OURS")
        // No id and somebody ELSE is rendering: guessing our id would aim the
        // cancel at our QUEUED job. Fall back to the untargeted default.
        #expect(EngineService.interruptTarget(
            activeJobId: nil, activeSource: "bree", ourInFlightJobId: "OURS") == nil)
        #expect(EngineService.interruptTarget(
            activeJobId: nil, activeSource: nil, ourInFlightJobId: "OURS") == nil)
        // Nothing of ours in flight.
        #expect(EngineService.interruptTarget(
            activeJobId: nil, activeSource: "desktop", ourInFlightJobId: nil) == nil)
    }

    /// …and end to end: the queue surface resolves the fallback target, so the
    /// stop is recorded as a user cancel and the render unwinds as `.cancelled`.
    @Test("a queue-tab stop with no active_job_id still unwinds our render as .cancelled")
    func queueTabStopWithMissingActiveIdStillCancels() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        engine.imageStatusPollInterval = 0.02
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"MINE2","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.always("/v1/queue/interrupt", .init(200, #"{"interrupted":true,"interrupted_job_id":"MINE2"}"#))
        transport.script("/v1/generate/status/MINE2", [
            .init(200, #"{"job_id":"MINE2","status":"processing","source":"desktop","elapsed_ms":10}"#),
            .init(200, #"{"job_id":"MINE2","status":"failed","source":"desktop","error":"Render interrupted by /v1/queue/interrupt","elapsed_ms":300}"#),
        ])

        let render = Task { try await engine.generate(GenerationRequest(prompt: "x")) }
        var waited = 0
        while engine.activeImageJobId == nil && waited < 400 {
            try await Task.sleep(for: .milliseconds(5)); waited += 1
        }

        // Exactly what the queue surfaces do when /v1/queue omits active_job_id.
        let target = EngineService.interruptTarget(
            activeJobId: nil, activeSource: "desktop", ourInFlightJobId: engine.activeImageJobId)
        #expect(target == "MINE2")
        _ = try await engine.cancelRender(jobId: target)

        var thrown: Error?
        do { _ = try await render.value } catch { thrown = error }
        guard case EngineServiceError.cancelled? = thrown as? EngineServiceError else {
            Issue.record("expected .cancelled, got \(String(describing: thrown))")
            return
        }
        #expect(engine.lastError == nil)
    }

    // MARK: - Reporting rule (PR #384 review r3, item 1)

    /// The batch loop used to set `engine.lastError` unconditionally, so a
    /// cancel the user asked for still painted a "Render cancelled" banner.
    /// `failureReport` is the pure rule the loop now consults.
    @Test("failureReport suppresses the banner for a cancel and keeps it for a failure")
    func failureReportRule() {
        let cancelled = EngineService.failureReport(for: EngineServiceError.cancelled("J"))
        #expect(cancelled.banner == nil, "a user cancel must not paint the error banner")
        #expect(cancelled.notice == "Render cancelled.", "but it still says something neutral")

        let failed = EngineService.failureReport(for: EngineServiceError.generationFailed("LoRA missing"))
        #expect(failed.banner == "Generation failed: LoRA missing")
        #expect(failed.notice == nil)

        let server = EngineService.failureReport(for: EngineServiceError.serverError(413, "too big"))
        #expect(server.banner == "Server error (413): too big")
        #expect(server.notice == nil)

        // Anything non-EngineServiceError still reports.
        struct Boom: Error, LocalizedError { var errorDescription: String? { "boom" } }
        #expect(EngineService.failureReport(for: Boom()).banner == "boom")
    }

    /// The user-cancel ledger evicts OLDEST-first. A Set's `removeFirst()`
    /// dropped an arbitrary element, which could silently turn a later real
    /// cancel into a failure banner.
    @Test("the user-cancel ledger is bounded and evicts oldest-first")
    func userCancelLedgerIsFIFO() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        // Overflow the 32-entry bound; the NEWEST id must survive.
        for i in 0..<40 { engine.noteUserCancelled("job-\(i)") }

        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"job-39","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/job-39", [
            .init(200, #"{"job_id":"job-39","status":"failed","source":"desktop","error":"Render interrupted by /v1/queue/interrupt","elapsed_ms":9}"#)
        ])

        var thrown: Error?
        do { _ = try await engine.generate(GenerationRequest(prompt: "x")) } catch { thrown = error }
        guard case EngineServiceError.cancelled? = thrown as? EngineServiceError else {
            Issue.record("newest noted id must still be recognised as a user cancel, got \(String(describing: thrown))")
            return
        }
        #expect(engine.lastError == nil)
    }

    // MARK: - Notices (PR #384 review, item 1)

    @Test("statusNotice surfaces queue depth, unresolved presets and refused preemption")
    func noticeWording() {
        func job(_ status: String, preset: String? = nil, reason: String? = nil,
                 refused: Bool? = nil, eta: Double? = nil) -> EngineService.ImageJobStatus {
            EngineService.ImageJobStatus(
                jobId: "J", status: status, outputPath: nil, error: nil, durationMs: nil,
                elapsedMs: nil, presetUnresolved: preset, presetUnresolvedReason: reason,
                preemptRefused: refused, etaSec: eta)
        }

        #expect(EngineService.statusNotice(for: job("processing"), pendingCount: 3) == nil)
        // /health's pending_count COUNTS our own queued job, so "behind N"
        // subtracts it (PR #384 review r2, item 4).
        #expect(EngineService.statusNotice(for: job("queued"), pendingCount: 3) == "Queued behind 2 jobs")
        #expect(EngineService.statusNotice(for: job("queued"), pendingCount: 2) == "Queued behind 1 job")
        #expect(EngineService.statusNotice(for: job("queued"), pendingCount: 1) == "Queued",
                "alone in the queue is not 'behind 1' — that 1 is us")
        #expect(EngineService.statusNotice(for: job("queued"), pendingCount: 0) == "Queued")
        #expect(EngineService.statusNotice(for: job("queued"), pendingCount: nil) == "Queued")

        let preset = EngineService.statusNotice(
            for: job("succeeded", preset: "kira", reason: "no_model"), pendingCount: nil)
        #expect(preset == "Preset 'kira' could not be resolved (no_model) — rendered without it")

        let refused = EngineService.statusNotice(
            for: job("succeeded", refused: true, eta: 42.4), pendingCount: nil)
        #expect(refused == "Preempt refused — queued behind the running video (~42s left)")
    }

    @Test("a render that resolved no preset publishes the notice to the UI")
    func noticePublished() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"N","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/N", [
            .init(200, #"{"job_id":"N","status":"succeeded","source":"desktop","output_path":"/tmp/n.png","preset_unresolved":"kira","preset_unresolved_reason":"no_model","elapsed_ms":9}"#)
        ])

        _ = try await engine.generate(GenerationRequest(prompt: "x"))
        #expect(engine.generationNotice == "Preset 'kira' could not be resolved (no_model) — rendered without it")
        #expect(engine.lastError == nil, "a notice is not an error")
    }

    @Test("a clean render leaves no notice behind")
    func noticeClearedOnCleanRender() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"C","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/C", [
            .init(200, #"{"job_id":"C","status":"succeeded","source":"desktop","output_path":"/tmp/c.png","elapsed_ms":9}"#)
        ])
        _ = try await engine.generate(GenerationRequest(prompt: "x"))
        #expect(engine.generationNotice == nil)
    }

    /// An undecodable body is TRANSIENT, not fatal — a proxy or a restarting
    /// engine can emit one mid-render — so it is retried for the budget and then
    /// reported, never crashed on. (Budget injected short; the 60s default would
    /// make this test take a minute.)
    @Test("a malformed status body is retried, then reported — never a crash")
    func malformedStatusBody() async throws {
        let transport = FakeEngineTransport()
        let engine = makeEngine(transport, outputDirectory: scratchDirectory())
        engine.imageStatusTransientFailureBudget = 0.1
        transport.script("/v1/generate/async", [.init(202, #"{"job_id":"M","status":"queued","source":"desktop","elapsed_ms":0}"#)])
        transport.script("/v1/generate/status/M", [.init(200, "not json at all")])

        var thrown: EngineServiceError?
        do { _ = try await engine.generate(GenerationRequest(prompt: "x")) }
        catch let error as EngineServiceError { thrown = error }
        guard case .generationFailed(let message)? = thrown else {
            Issue.record("expected .generationFailed, got \(String(describing: thrown))")
            return
        }
        #expect(message.contains("budget"))
        #expect(message.contains("did not decode"), "name the real fault, not \"returned 200\"")
        #expect(!message.contains("returned 200"))
        #expect(transport.requestCount("GET", "/v1/generate/status/M") > 1, "retried before giving up")
    }
}
