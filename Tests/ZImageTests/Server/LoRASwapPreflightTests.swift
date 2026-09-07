import XCTest

@testable import ZImage

/// comfybox#415 / comfybox#396 — `POST /v1/lora/swap` validates and resolves
/// every entry at the route, on local disk only, before the family guard and
/// before anything reaches the coordinator.
///
/// Incident A (#415): `{"loras":[{"path":"","scale":0.5}]}` was enqueued,
/// persisted, and parked `WarmServerCoordinator.runSwap` in
/// `LoRAEntry.resolveSource`'s walk of `/Volumes/Bolt/Models/loras` (a hung
/// USB volume) — replayed after every restart.
/// Incident B (#396): a bare name that was on local disk under `vault/` was
/// first handed to the nearline library (whose catalog lists the same name on
/// Bolt) and the route sat in the copy for the daemon's 600 s timeout.
final class LoRASwapPreflightTests: XCTestCase {

  private var tempRoot: URL!

  override func setUpWithError() throws {
    let scratch = FileManager.default.temporaryDirectory
      .appendingPathComponent("lora-swap-preflight-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
      at: scratch.appendingPathComponent("vault"), withIntermediateDirectories: true)
    // The directory walk reports real paths (`/private/var/...`) while
    // `temporaryDirectory` is the `/var/...` symlink — compare like with like.
    tempRoot = URL(fileURLWithPath: Self.realPath(scratch.path), isDirectory: true)
  }

  private static func realPath(_ path: String) -> String {
    guard let resolved = realpath(path, nil) else { return path }
    defer { free(resolved) }
    return String(cString: resolved)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempRoot)
  }

  @discardableResult
  private func touch(_ relative: String) throws -> String {
    let url = tempRoot.appendingPathComponent(relative)
    try Data("not really a lora".utf8).write(to: url)
    return url.path
  }

  private func noNearline(_ name: String) -> LoRASwapPreflight.NearlineStageOutcome {
    XCTFail("nearline staging must not be consulted for '\(name)'")
    return .notNearline
  }

  // MARK: - (1) empty path → 400, nothing enqueued, nothing walked

  func testEmptyPathIs400BeforeAnyResolution() {
    var lookups = 0
    XCTAssertThrowsError(
      try LoRASwapPreflight.run(
        entries: [LoRAEntry(path: "", scale: 0.5)],
        libraryLookup: { _ in lookups += 1; return nil },
        searchRoots: [tempRoot.path],
        nearlineStage: noNearline)
    ) { error in
      guard case WarmServerError.invalidRequest(let message) = error else { return XCTFail("\(error)") }
      XCTAssertTrue(message.contains("empty"), message)
      XCTAssertEqual(WarmServer.errorResponse(for: error).status, 400)
    }
    XCTAssertEqual(lookups, 0, "an empty path is refused before any lookup")
    // The generate/preset path reads the same entry type — refused there too.
    XCTAssertThrowsError(try LoRAEntry(path: "   ", scale: nil).makeConfiguration())
  }

  /// The exact persisted body from the 2026-09-07 incident
  /// (`queue-state.json.bak-poison-swap-20260907`), through the real wire
  /// decoder.
  func testIncidentPoisonBodyIsRefused() throws {
    let rawBody = try XCTUnwrap(Data(base64Encoded: "eyJsb3JhcyI6W3sicGF0aCI6IiIsInNjYWxlIjowLjV9XX0="))
    let payload = try WarmServer.decode(LoRASwapPayload.self, from: rawBody)
    XCTAssertEqual(payload.loras.count, 1)
    XCTAssertEqual(payload.loras[0].path, "")
    XCTAssertThrowsError(
      try LoRASwapPreflight.run(
        entries: payload.loras, libraryLookup: { _ in nil }, searchRoots: [tempRoot.path], nearlineStage: noNearline))
  }

  // MARK: - (2) bare filename under a nested subdirectory resolves — instantly, locally

  func testBareFilenameInNestedSubdirectoryResolves() throws {
    let expected = try touch("vault/KreaAmateur_V2.safetensors")
    let start = Date()
    let preflight = try LoRASwapPreflight.run(
      entries: [LoRAEntry(path: "KreaAmateur_V2.safetensors", scale: 0.5, role: "accel")],
      libraryLookup: { _ in nil },
      searchRoots: [tempRoot.path],
      nearlineStage: noNearline)
    XCTAssertLessThan(Date().timeIntervalSince(start), 1.0)
    XCTAssertEqual(preflight.resolved.map(\.path), [expected])
    XCTAssertEqual(preflight.resolved[0].scale, 0.5)
    XCTAssertEqual(preflight.resolved[0].role, "accel", "resolution rewrites only the path")
    XCTAssertTrue(preflight.unresolved.isEmpty)
    XCTAssertFalse(preflight.nothingApplicable)
  }

  /// `LoRAEntry.makeConfiguration` (the generate/preset path and the
  /// coordinator's `makeConfigurations`) walks the configured root the same
  /// way, driven by COMFYBOX_MODELS as in production.
  func testMakeConfigurationWalksConfiguredRootRecursively() throws {
    let expected = try touch("vault/nested_lora.safetensors")
    let original = ProcessInfo.processInfo.environment["COMFYBOX_MODELS"]
    setenv("COMFYBOX_MODELS", tempRoot.path, 1)
    defer {
      if let original { setenv("COMFYBOX_MODELS", original, 1) } else { unsetenv("COMFYBOX_MODELS") }
    }
    XCTAssertEqual(LoRAEntry.bareFilenameSearchRoots.first, tempRoot.path)
    let configuration = try LoRAEntry(path: "nested_lora.safetensors", scale: 0.7).makeConfiguration()
    guard case .local(let url) = configuration.source else { return XCTFail("\(configuration.source)") }
    XCTAssertEqual(url.path, expected)
  }

  func testLibraryIndexIsConsultedBeforeWalking() throws {
    let indexed = try touch("vault/indexed.safetensors")
    let preflight = try LoRASwapPreflight.run(
      entries: [LoRAEntry(path: "indexed.safetensors", scale: nil)],
      libraryLookup: { name in name == "indexed.safetensors" ? indexed : nil },
      searchRoots: ["/nonexistent/root/that/must/not/matter"],
      nearlineStage: noNearline)
    XCTAssertEqual(preflight.resolved.map(\.path), [indexed])
  }

  func testRootRelativePathResolves() throws {
    let expected = try touch("vault/by_relative.safetensors")
    let preflight = try LoRASwapPreflight.run(
      entries: [LoRAEntry(path: "vault/by_relative.safetensors", scale: nil)],
      libraryLookup: { _ in nil },
      searchRoots: [tempRoot.path],
      nearlineStage: noNearline)
    XCTAssertEqual(preflight.resolved.map(\.path), [expected])
  }

  // MARK: - (3) missing name → unresolved in < 1 s, no network, the rest applied

  func testMissingBareNameIsUnresolvedFastAndPresentEntriesAreKept() throws {
    let present = try touch("vault/present.safetensors")
    let start = Date()
    let preflight = try LoRASwapPreflight.run(
      entries: [
        LoRAEntry(path: "present.safetensors", scale: 0.6),
        LoRAEntry(path: "Missing_V9.safetensors", scale: 0.5),
      ],
      libraryLookup: { _ in nil },
      searchRoots: [tempRoot.path],
      nearlineStage: { _ in .notNearline })
    XCTAssertLessThan(Date().timeIntervalSince(start), 1.0)
    XCTAssertEqual(preflight.resolved.map(\.path), [present], "the present entry is applied")
    XCTAssertEqual(preflight.unresolved.map(\.path), ["Missing_V9.safetensors"])
    XCTAssertTrue(preflight.unresolved[0].reason.contains(tempRoot.path), preflight.unresolved[0].reason)
    XCTAssertFalse(preflight.nothingApplicable)
    XCTAssertNil(LoRASwapPreflight.poisonReason(preflight))
  }

  func testAllUnresolvedIsRefusedWithStructuredList() throws {
    let preflight = try LoRASwapPreflight.run(
      entries: [LoRAEntry(path: "Missing_V9.safetensors", scale: 0.5)],
      libraryLookup: { _ in nil },
      searchRoots: [tempRoot.path],
      nearlineStage: { _ in .notNearline })
    XCTAssertTrue(preflight.nothingApplicable, "nothing to apply must NOT clear the resident stack")
    let response = HTTPResponse.json(status: 400, payload: LoRASwapRefusal(unresolved: preflight.unresolved))
    XCTAssertEqual(response.status, 400)
    let body = String(decoding: response.body, as: UTF8.self)
    XCTAssertTrue(body.contains("\"unresolved\""), body)
    XCTAssertTrue(body.contains("Missing_V9.safetensors"), body)
    XCTAssertTrue(body.contains("not found"), "keeps the spelling clients already match: \(body)")
  }

  func testRepoIdShapedEntryIsUnresolvedNotFetched() throws {
    let preflight = try LoRASwapPreflight.run(
      entries: [LoRAEntry(path: "someorg/some-lora", scale: nil)],
      libraryLookup: { _ in nil },
      searchRoots: [tempRoot.path],
      nearlineStage: noNearline)
    XCTAssertEqual(preflight.unresolved.count, 1)
    XCTAssertTrue(preflight.unresolved[0].reason.contains("HuggingFace"), preflight.unresolved[0].reason)
  }

  func testAbsolutePathThatDoesNotExistIsUnresolved() throws {
    let preflight = try LoRASwapPreflight.run(
      entries: [LoRAEntry(path: tempRoot.appendingPathComponent("gone.safetensors").path, scale: nil)],
      libraryLookup: { _ in nil },
      searchRoots: [tempRoot.path],
      nearlineStage: noNearline)
    XCTAssertEqual(preflight.unresolved.map(\.reason), ["file does not exist"])
  }

  /// The walk must never touch an external volume or Downloads again —
  /// `fileExists("/Volumes/Bolt/Models/loras")` on the hung volume was the
  /// #415 coordinator wedge.
  func testSearchRootsAreLocalLibraryDirectoriesOnly() {
    for root in LoRAEntry.bareFilenameSearchRoots {
      XCTAssertFalse(root.hasPrefix("/Volumes/"), root)
      XCTAssertFalse(root.hasSuffix("/Downloads"), root)
    }
    XCTAssertTrue(LoRAEntry.bareFilenameSearchRoots.contains { $0.hasSuffix("/.comfybox/loras") })
  }

  // MARK: - nearline: only for names not on local disk, bounded

  func testNearlineIsOnlyConsultedForNamesNotOnLocalDisk() throws {
    let local = try touch("vault/on_disk.safetensors")
    var staged: [String] = []
    let preflight = try LoRASwapPreflight.run(
      entries: [
        LoRAEntry(path: "on_disk.safetensors", scale: nil),
        LoRAEntry(path: "archived.safetensors", scale: 0.4, role: "kroma"),
      ],
      libraryLookup: { _ in nil },
      searchRoots: [tempRoot.path],
      nearlineStage: { name in staged.append(name); return .staged("/staged/\(name)") })
    XCTAssertEqual(staged, ["archived.safetensors"], "a name already on disk never goes to nearline (#396)")
    XCTAssertEqual(preflight.resolved.map(\.path), [local, "/staged/archived.safetensors"])
    XCTAssertEqual(preflight.resolved[1].role, "kroma")
  }

  func testNearlineTimeoutIsReportedUnresolvedWithRetryHint() throws {
    let preflight = try LoRASwapPreflight.run(
      entries: [LoRAEntry(path: "slow.safetensors", scale: nil)],
      libraryLookup: { _ in nil },
      searchRoots: [tempRoot.path],
      nearlineStage: { _ in .timedOut })
    XCTAssertEqual(preflight.unresolved.count, 1)
    XCTAssertTrue(preflight.unresolved[0].reason.contains("retry"), preflight.unresolved[0].reason)
  }

  func testNearlineResultBoxBoundsTheWait() {
    let box = NearlineStageResultBox()
    let start = Date()
    XCTAssertNil(box.wait(timeout: 0.05), "no finish → nil after the bound")
    XCTAssertLessThan(Date().timeIntervalSince(start), 1.0)
    box.finish(.success("/staged/x"))
    guard case .success(let path)? = box.wait(timeout: 0.05) else { return XCTFail("finished result must be returned") }
    XCTAssertEqual(path, "/staged/x")
  }

  // MARK: - (4) persisted swap replay: a poison job is dropped, a good one replays

  func testPersistedPoisonSwapIsDroppedAtReplay() throws {
    let rawBody = try XCTUnwrap(Data(base64Encoded: "eyJsb3JhcyI6W3sicGF0aCI6IiIsInNjYWxlIjowLjV9XX0="))
    let job = PersistedQueueJob(id: "D9A75100", kind: "lora_swap", source: "api", enqueuedAt: Date(), rawBody: rawBody)

    var failure: Error?
    do {
      let payload = try WarmServer.decode(LoRASwapPayload.self, from: job.rawBody)
      _ = try LoRASwapPreflight.run(
        entries: payload.loras, libraryLookup: { _ in nil }, searchRoots: [tempRoot.path], nearlineStage: noNearline)
    } catch { failure = error }
    let error = try XCTUnwrap(failure, "the incident's persisted job must fail preflight, never enqueue")

    let tracker = ImageJobTracker()
    tracker.recordFailedReplay(jobId: job.id, source: job.source, error: error)
    let status = try XCTUnwrap(tracker.status(jobId: "D9A75100"))
    XCTAssertEqual(status.status, .failed)
  }

  func testPersistedSwapWithOnlyMissingEntriesIsPoison() throws {
    let rawBody = Data(#"{"loras":[{"path":"Gone_V1.safetensors","scale":0.5}]}"#.utf8)
    let payload = try WarmServer.decode(LoRASwapPayload.self, from: rawBody)
    let preflight = try LoRASwapPreflight.run(
      entries: payload.loras, libraryLookup: { _ in nil }, searchRoots: [tempRoot.path],
      nearlineStage: { _ in .notNearline })
    let reason = try XCTUnwrap(LoRASwapPreflight.poisonReason(preflight), "nothing applicable → dropped")
    XCTAssertTrue(reason.contains("Gone_V1.safetensors"), reason)
  }

  func testPersistedSwapWithAPresentEntryStillReplays() throws {
    let present = try touch("vault/still_here.safetensors")
    let rawBody = Data(#"{"loras":[{"path":"still_here.safetensors","scale":0.5},{"path":"Gone_V1.safetensors"}]}"#.utf8)
    let payload = try WarmServer.decode(LoRASwapPayload.self, from: rawBody)
    let preflight = try LoRASwapPreflight.run(
      entries: payload.loras, libraryLookup: { _ in nil }, searchRoots: [tempRoot.path],
      nearlineStage: { _ in .notNearline })
    XCTAssertNil(LoRASwapPreflight.poisonReason(preflight))
    XCTAssertEqual(preflight.resolved.map(\.path), [present])
  }

  // MARK: - response shape (additive)

  func testSwapResponseCarriesUnresolvedList() throws {
    var response = LoRASwapResponse(success: true, loraCount: 1, loras: [])
    response.unresolved = [.init(path: "Missing_V9.safetensors", reason: "not found")]
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let body = String(decoding: try encoder.encode(response), as: UTF8.self)
    XCTAssertTrue(body.contains("\"lora_count\":1"), body)
    XCTAssertTrue(body.contains("\"unresolved\":[{"), body)
    XCTAssertTrue(body.contains("\"reason\":\"not found\""), body)
  }
}
