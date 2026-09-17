import XCTest

@testable import ZImage

/// `comfybox director-render` (docs/FDD-ltx-director-tab.md §4.3, WP4): pure
/// argument parsing plus the HTTP-client flow driven through a stub transport.
/// The command is a client of the RUNNING server; nothing here renders.
final class DirectorRenderArgsTests: XCTestCase {

  // MARK: - Argument parsing

  func testParsesFileAndDefaults() throws {
    let args = try DirectorRenderArgs.parse(["pier.cbdirector"])
    XCTAssertEqual(args.filePath, "pier.cbdirector")
    XCTAssertEqual(args.server, "http://127.0.0.1:7870")
    XCTAssertEqual(args.host, "127.0.0.1")
    XCTAssertEqual(args.port, 7870)
    XCTAssertNil(args.outputPath)
    XCTAssertEqual(args.source, "cli")
    XCTAssertFalse(args.validateOnly)
    XCTAssertFalse(args.wait)
    XCTAssertEqual(args.pollSeconds, 2)
  }

  func testParsesEveryFlagInAnyOrder() throws {
    let args = try DirectorRenderArgs.parse([
      "--server", "http://10.0.0.5:7862", "--output", "out.mp4", "p.cbdirector",
      "--source", "desktop", "--wait", "--poll-seconds", "5",
    ])
    XCTAssertEqual(args.filePath, "p.cbdirector")
    XCTAssertEqual(args.host, "10.0.0.5")
    XCTAssertEqual(args.port, 7862)
    XCTAssertEqual(args.outputPath, "out.mp4")
    XCTAssertEqual(args.source, "desktop")
    XCTAssertTrue(args.wait)
    XCTAssertEqual(args.pollSeconds, 5)
  }

  func testValidateOnlyFlag() throws {
    let args = try DirectorRenderArgs.parse(["--validate-only", "p.cbdirector"])
    XCTAssertTrue(args.validateOnly)
    XCTAssertFalse(args.wait)
    XCTAssertThrowsError(try DirectorRenderArgs.parse(["p.cbdirector", "--validate-only", "--wait"])) {
      XCTAssertTrue(Self.isUsage($0, containing: "--wait"))
    }
  }

  func testMissingFileIsUsageError() {
    XCTAssertThrowsError(try DirectorRenderArgs.parse([])) {
      XCTAssertTrue(Self.isUsage($0, containing: "cbdirector"))
    }
    XCTAssertThrowsError(try DirectorRenderArgs.parse(["--wait"])) {
      XCTAssertTrue(Self.isUsage($0, containing: "cbdirector"))
    }
    XCTAssertThrowsError(try DirectorRenderArgs.parse(["a.cbdirector", "b.cbdirector"])) {
      XCTAssertTrue(Self.isUsage($0, containing: "b.cbdirector"))
    }
    XCTAssertThrowsError(try DirectorRenderArgs.parse(["a.cbdirector", "--output"])) {
      XCTAssertTrue(Self.isUsage($0, containing: "--output"))
    }
  }

  func testUnknownFlagIsUsageError() {
    XCTAssertThrowsError(try DirectorRenderArgs.parse(["a.cbdirector", "--frames", "97"])) {
      XCTAssertTrue(Self.isUsage($0, containing: "--frames"))
    }
  }

  func testHelpIsReportedNotParsed() {
    XCTAssertThrowsError(try DirectorRenderArgs.parse(["--help"])) {
      XCTAssertEqual($0 as? DirectorRenderArgsError, .helpRequested)
    }
    XCTAssertTrue(DirectorRenderArgs.usage.contains("director-render <file.cbdirector>"))
  }

  func testPollSecondsParses() throws {
    XCTAssertEqual(try DirectorRenderArgs.parse(["a.cbdirector", "--poll-seconds", "0.5"]).pollSeconds, 0.5)
    for bad in ["0", "-1", "soon"] {
      XCTAssertThrowsError(try DirectorRenderArgs.parse(["a.cbdirector", "--poll-seconds", bad])) {
        XCTAssertTrue(Self.isUsage($0, containing: "--poll-seconds"), bad)
      }
    }
  }

  func testServerURLIsValidated() throws {
    XCTAssertEqual(try DirectorRenderArgs.parse(["a.cbdirector", "--server", "http://mac.local"]).port, 80)
    XCTAssertEqual(try DirectorRenderArgs.parse(["a.cbdirector", "--server", "http://h:7870/"]).host, "h")
    for bad in ["127.0.0.1:7870", "https://h:7870", "http://h:7870/v1", "http://"] {
      XCTAssertThrowsError(try DirectorRenderArgs.parse(["a.cbdirector", "--server", bad])) {
        XCTAssertTrue(Self.isUsage($0, containing: "--server"), bad)
      }
    }
  }

  // MARK: - Client flow (stub transport)

  private final class StubTransport: WarmServerTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [(method: String, path: String, body: Data)] = []
    private var statusReplies: [(Int, String)]
    private let postReply: (Int, String)

    init(post: (Int, String), statuses: [(Int, String)] = []) {
      self.postReply = post
      self.statusReplies = statuses
    }

    var calls: [String] {
      lock.lock(); defer { lock.unlock() }
      return recorded.map { "\($0.method) \($0.path)" }
    }

    func body(ofCallAt index: Int) -> [String: Any]? {
      lock.lock(); defer { lock.unlock() }
      return try? JSONSerialization.jsonObject(with: recorded[index].body) as? [String: Any]
    }

    func get(_ path: String) async throws -> (Int, Data) {
      let reply: (Int, String) = lock.withLock {
        recorded.append(("GET", path, Data()))
        return statusReplies.isEmpty ? (500, #"{"error":"script exhausted"}"#) : statusReplies.removeFirst()
      }
      return (reply.0, Data(reply.1.utf8))
    }
    func post(_ path: String, body: Data) async throws -> (Int, Data) {
      lock.withLock { recorded.append(("POST", path, body)) }
      return (postReply.0, Data(postReply.1.utf8))
    }
    func put(_ path: String, body: Data) async throws -> (Int, Data) { (405, Data()) }
    func patch(_ path: String, body: Data) async throws -> (Int, Data) { (405, Data()) }
    func delete(_ path: String) async throws -> (Int, Data) { (405, Data()) }
    func send(method: String, path: String, body: Data, headers: [String: String]) async throws
      -> (Int, Data, [String: String]) { (405, Data(), [:]) }
  }

  private final class Lines: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []
    func append(_ s: String) { lock.lock(); storage.append(s); lock.unlock() }
    var all: [String] { lock.lock(); defer { lock.unlock() }; return storage }
  }

  private func writeTimeline(_ json: String) throws -> String {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("director-cli-\(UUID().uuidString).cbdirector")
    try Data(json.utf8).write(to: url)
    addTeardownBlock { try? FileManager.default.removeItem(at: url) }
    return url.path
  }

  private static let timeline = """
    {"version": 1, "settings": {"width": 576, "height": 896, "length_frames": 289},
     "global_prompt": "a pier at dusk",
     "keyframes": [{"id": "k1", "image_path": "/tmp/a.png", "frame": 0, "is_end_frame": false}]}
    """

  private func run(
    _ argv: [String], _ transport: StubTransport, out: Lines, err: Lines
  ) async throws -> Int32 {
    let args = try DirectorRenderArgs.parse(argv)
    return await DirectorRenderClient.run(
      args: args, transport: transport,
      out: { out.append($0) }, err: { err.append($0) }, sleep: { _ in })
  }

  func testEnvelopeIsSnakeCaseWithOutputAndSource() throws {
    let tl = try DirectorJSON.decoder().decode(DirectorTimeline.self, from: Data(Self.timeline.utf8))
    let render = try DirectorRenderArgs.parse(["a.cbdirector", "--output", "o.mp4"])
    let body = try XCTUnwrap(
      JSONSerialization.jsonObject(with: DirectorRenderClient.envelope(timeline: tl, args: render))
        as? [String: Any])
    XCTAssertEqual(body["output_path"] as? String, "o.mp4")
    XCTAssertEqual(body["source"] as? String, "cli")
    let t = try XCTUnwrap(body["timeline"] as? [String: Any])
    XCTAssertEqual(t["global_prompt"] as? String, "a pier at dusk")
    XCTAssertEqual((t["settings"] as? [String: Any])?["length_frames"] as? Int, 289)

    let validate = try DirectorRenderArgs.parse(["a.cbdirector", "--validate-only", "--output", "o.mp4"])
    let vbody = try XCTUnwrap(
      JSONSerialization.jsonObject(with: DirectorRenderClient.envelope(timeline: tl, args: validate))
        as? [String: Any])
    XCTAssertEqual(Set(vbody.keys), ["timeline"])
  }

  func testValidateOnlyPostsValidateAndExitsOnOk() async throws {
    let path = try writeTimeline(Self.timeline)
    let ok = StubTransport(post: (200, #"{"ok":true,"issues":[],"snapped_length_frames":289}"#))
    let out = Lines(), err = Lines()
    let code = try await run([path, "--validate-only"], ok, out: out, err: err)
    XCTAssertEqual(code, 0)
    XCTAssertEqual(ok.calls, ["POST /v1/video/director/validate"])
    XCTAssertTrue(out.all.joined().contains("snapped_length_frames"))

    let bad = StubTransport(post: (200, #"{"ok":false,"issues":[{"code":"keyframe_off_grid"}]}"#))
    let code2 = try await run([path, "--validate-only"], bad, out: Lines(), err: Lines())
    XCTAssertEqual(code2, 1, "a timeline with errors exits non-zero")
  }

  func testRenderWithoutWaitPrintsJobAndDoesNotPoll() async throws {
    let path = try writeTimeline(Self.timeline)
    let transport = StubTransport(post: (202, #"{"job_id":"D-9","status":"queued","stage_count":2}"#))
    let out = Lines()
    let code = try await run([path, "--source", "api"], transport, out: out, err: Lines())
    XCTAssertEqual(code, 0)
    XCTAssertEqual(transport.calls, ["POST /v1/video/director"])
    XCTAssertEqual(transport.body(ofCallAt: 0)?["source"] as? String, "api")
    XCTAssertTrue(out.all.joined().contains("D-9"))
  }

  func testRenderRejectedExitsNonZero() async throws {
    let path = try writeTimeline(Self.timeline)
    let transport = StubTransport(post: (400, #"{"error":"timeline invalid","issues":[]}"#))
    let err = Lines()
    let code = try await run([path], transport, out: Lines(), err: err)
    XCTAssertEqual(code, 1)
    XCTAssertTrue(err.all.joined().contains("timeline invalid"))
  }

  func testWaitPollsStagesUntilSucceeded() async throws {
    let path = try writeTimeline(Self.timeline)
    let transport = StubTransport(
      post: (202, #"{"job_id":"D-1","status":"queued","stage_count":2}"#),
      statuses: [
        (200, #"{"job_id":"D-1","status":"processing","stage_index":0,"stage_count":2,"progress_percent":20}"#),
        (200, #"{"job_id":"D-1","status":"processing","stage_index":1,"stage_count":2,"progress_percent":60}"#),
        (200, #"{"job_id":"D-1","status":"processing","stage_index":2,"stage_count":2,"progress_percent":97}"#),
        (200, #"{"job_id":"D-1","status":"succeeded","output_path":"/o/director-x.mp4","progress_percent":100}"#),
      ])
    let out = Lines()
    let code = try await run([path, "--wait"], transport, out: out, err: Lines())
    XCTAssertEqual(code, 0)
    XCTAssertEqual(
      transport.calls,
      ["POST /v1/video/director"] + Array(repeating: "GET /v1/video/status/D-1", count: 4))
    let text = out.all.joined(separator: "\n")
    XCTAssertTrue(text.contains("chunk 1/2"), text)
    XCTAssertTrue(text.contains("chunk 2/2"), text)
    XCTAssertTrue(text.contains("stitching"), text)
    XCTAssertTrue(text.contains("/o/director-x.mp4"), text)
  }

  func testWaitFailedJobExitsOne() async throws {
    let path = try writeTimeline(Self.timeline)
    let transport = StubTransport(
      post: (202, #"{"job_id":"D-2","status":"queued","stage_count":2}"#),
      statuses: [(200, #"{"job_id":"D-2","status":"failed","error":"chunk 2/2 (render): boom"}"#)])
    let err = Lines()
    let code = try await run([path, "--wait"], transport, out: Lines(), err: err)
    XCTAssertEqual(code, 1)
    XCTAssertTrue(err.all.joined().contains("chunk 2/2 (render): boom"))
  }

  func testWaitStatusHTTPErrorExitsOneWithoutRetrying() async throws {
    let path = try writeTimeline(Self.timeline)
    let transport = StubTransport(
      post: (202, #"{"job_id":"D-3","status":"queued"}"#),
      statuses: [(404, #"{"error":"job not found"}"#)])
    let code = try await run([path, "--wait"], transport, out: Lines(), err: Lines())
    XCTAssertEqual(code, 1)
    XCTAssertEqual(transport.calls.count, 2)
  }

  func testUnreadableAndFutureVersionFilesNeverReachTheServer() async throws {
    let transport = StubTransport(post: (202, "{}"))
    let err = Lines()
    let missing = try await run(["/nonexistent/x.cbdirector"], transport, out: Lines(), err: err)
    XCTAssertEqual(missing, 1)

    let future = try writeTimeline(#"{"version": 2, "settings": {"width": 576, "height": 896, "length_frames": 97}, "global_prompt": "x"}"#)
    let code = try await run([future], transport, out: Lines(), err: err)
    XCTAssertEqual(code, DirectorRenderClient.usageExitCode)
    XCTAssertTrue(err.all.joined().contains("version"))
    XCTAssertEqual(transport.calls, [])
  }

  func testProgressLine() {
    XCTAssertEqual(
      DirectorRenderClient.progressLine(["status": "queued"]), "queued")
    XCTAssertEqual(
      DirectorRenderClient.progressLine(
        ["status": "processing", "stage_index": 0, "stage_count": 3, "progress_percent": 12]),
      "processing · chunk 1/3 · 12%")
    XCTAssertEqual(
      DirectorRenderClient.progressLine(
        ["status": "processing", "stage_index": 3, "stage_count": 3, "progress_percent": 98]),
      "processing · stitching · 98%")
  }

  private static func isUsage(_ error: Error, containing needle: String) -> Bool {
    guard case .usage(let message)? = error as? DirectorRenderArgsError else { return false }
    return message.contains(needle)
  }
}
