import XCTest

@testable import ZImage

/// comfybox#415 / comfybox#396 — `POST /v1/lora/swap` validates and resolves
/// every entry at the route, on local disk only, before the family guard and
/// before anything reaches the coordinator. The swap is ATOMIC (PR #418
/// review): any unresolvable entry → 400 with `unresolved`, stack untouched.
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
    LoRAEntry.invalidateLocalIndex()
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempRoot)
    LoRAEntry.invalidateLocalIndex()
  }

  private static func realPath(_ path: String) -> String {
    guard let resolved = realpath(path, nil) else { return path }
    defer { free(resolved) }
    return String(cString: resolved)
  }

  @discardableResult
  private func touch(_ relative: String) throws -> String {
    let url = tempRoot.appendingPathComponent(relative)
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data("not really a lora".utf8).write(to: url)
    return url.path
  }

  private func noNearline(_ name: String) -> LoRASwapPreflight.NearlineStageOutcome {
    XCTFail("nearline staging must not be consulted for '\(name)'")
    return .notNearline
  }

  private func run(_ entries: [LoRAEntry],
                   lookup: @escaping (String) -> String? = { _ in nil },
                   nearline: @escaping (String) -> LoRASwapPreflight.NearlineStageOutcome = { _ in .notNearline }
  ) throws -> LoRASwapPreflight {
    try LoRASwapPreflight.run(entries: entries, libraryLookup: lookup, searchRoots: [tempRoot.path], nearlineStage: nearline)
  }

  // MARK: - (1) empty path → 400, nothing enqueued, nothing walked

  func testEmptyPathIs400BeforeAnyResolution() {
    var lookups = 0
    XCTAssertThrowsError(
      try run([LoRAEntry(path: "", scale: 0.5)], lookup: { _ in lookups += 1; return nil }, nearline: noNearline)
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
    XCTAssertThrowsError(try run(payload.loras, nearline: noNearline))
  }

  // MARK: - (2) bare filename under a nested subdirectory resolves — instantly, locally

  func testBareFilenameInNestedSubdirectoryResolves() throws {
    let expected = try touch("vault/KreaAmateur_V2.safetensors")
    let start = Date()
    let preflight = try run([LoRAEntry(path: "KreaAmateur_V2.safetensors", scale: 0.5, role: "accel")], nearline: noNearline)
    XCTAssertLessThan(Date().timeIntervalSince(start), 1.0)
    XCTAssertEqual(preflight.resolved.map(\.path), [expected])
    XCTAssertEqual(preflight.resolved[0].scale, 0.5)
    XCTAssertEqual(preflight.resolved[0].role, "accel", "resolution rewrites only the path")
    XCTAssertTrue(preflight.unresolved.isEmpty)
    XCTAssertFalse(preflight.hasUnresolved)
    XCTAssertNil(LoRASwapPreflight.poisonReason(preflight))
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
    let preflight = try run([LoRAEntry(path: "vault/by_relative.safetensors", scale: nil)], nearline: noNearline)
    XCTAssertEqual(preflight.resolved.map(\.path), [expected])
  }

  // MARK: - (3) missing name → unresolved in < 1 s, no network; the swap is atomic

  func testMissingBareNameIsUnresolvedFastAndRefusesTheWholeSwap() throws {
    let present = try touch("vault/present.safetensors")
    let start = Date()
    let preflight = try run([
      LoRAEntry(path: "present.safetensors", scale: 0.6),
      LoRAEntry(path: "Missing_V9.safetensors", scale: 0.5),
    ])
    XCTAssertLessThan(Date().timeIntervalSince(start), 1.0)
    XCTAssertEqual(preflight.resolved.map(\.path), [present])
    XCTAssertEqual(preflight.unresolved.map(\.path), ["Missing_V9.safetensors"])
    XCTAssertTrue(preflight.unresolved[0].reason.contains(tempRoot.path), preflight.unresolved[0].reason)
    // PR #418 review (Critical 2): ANY unresolved entry refuses the swap —
    // the route 400s with the list; nothing is applied.
    XCTAssertTrue(preflight.hasUnresolved)
    let reason = try XCTUnwrap(LoRASwapPreflight.poisonReason(preflight))
    XCTAssertTrue(reason.contains("Missing_V9.safetensors"), reason)
    let response = HTTPResponse.json(status: 400, payload: LoRASwapRefusal(unresolved: preflight.unresolved))
    XCTAssertEqual(response.status, 400)
    let body = String(decoding: response.body, as: UTF8.self)
    XCTAssertTrue(body.contains("\"unresolved\":[{"), body)
    XCTAssertTrue(body.contains("Missing_V9.safetensors"), body)
    XCTAssertFalse(body.contains("present.safetensors"), "only the unresolved entries are listed: \(body)")
    XCTAssertTrue(body.contains("not found"), "keeps the spelling clients already match: \(body)")
  }

  func testRepoIdShapedEntryIsUnresolvedNotFetched() throws {
    let preflight = try run([LoRAEntry(path: "someorg/some-lora", scale: nil)], nearline: noNearline)
    XCTAssertEqual(preflight.unresolved.count, 1)
    XCTAssertTrue(preflight.unresolved[0].reason.contains("HuggingFace"), preflight.unresolved[0].reason)
  }

  func testAbsolutePathThatDoesNotExistIsUnresolved() throws {
    let preflight = try run([LoRAEntry(path: tempRoot.appendingPathComponent("gone.safetensors").path, scale: nil)], nearline: noNearline)
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

  // MARK: - PR #418 review, Critical 1: the walk never reaches /Volumes

  /// `~/.comfybox/loras/quarantine -> /Volumes/Bolt/Models/archive/quarantine`
  /// on Todd's box. A symlinked directory inside a root is classified by
  /// `lstat` + `readlink` only — the target is never stat'd (so a hung
  /// mount cannot park the walk) and never descended.
  func testSymlinkIntoVolumesIsClassifiedWithoutFollowing() throws {
    let link = tempRoot.appendingPathComponent("quarantine").path
    try FileManager.default.createSymbolicLink(atPath: link, withDestinationPath: "/Volumes/Nope/archive/quarantine")
    XCTAssertEqual(LoRAEntry.classifyWithoutFollowing(link), .removableVolume)
    // A file symlink into /Volumes, and a two-hop chain, are caught the same way.
    let fileLink = tempRoot.appendingPathComponent("bolt_lora.safetensors").path
    try FileManager.default.createSymbolicLink(atPath: fileLink, withDestinationPath: "/Volumes/Nope/x.safetensors")
    XCTAssertEqual(LoRAEntry.classifyWithoutFollowing(fileLink), .removableVolume)
    let hop = tempRoot.appendingPathComponent("hop.safetensors").path
    try FileManager.default.createSymbolicLink(atPath: hop, withDestinationPath: "bolt_lora.safetensors")
    XCTAssertEqual(LoRAEntry.classifyWithoutFollowing(hop), .removableVolume)
    // Plain entries still classify.
    XCTAssertEqual(LoRAEntry.classifyWithoutFollowing(tempRoot.appendingPathComponent("vault").path), .directory)
    XCTAssertEqual(LoRAEntry.classifyWithoutFollowing(try touch("vault/plain.safetensors")), .file)
    XCTAssertEqual(LoRAEntry.classifyWithoutFollowing(tempRoot.appendingPathComponent("nothing").path), .missing)
    XCTAssertEqual(LoRAEntry.classifyWithoutFollowing("/Volumes/Nope/direct.safetensors"), .removableVolume)
  }

  func testWalkDoesNotDescendSymlinkedDirectories() throws {
    // A REAL directory outside the root, reachable only through a symlink
    // inside it. If the walk followed the link it would find the file.
    let outside = FileManager.default.temporaryDirectory
      .appendingPathComponent("lora-swap-outside-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: outside) }
    try Data("x".utf8).write(to: outside.appendingPathComponent("behind_link.safetensors"))
    try FileManager.default.createSymbolicLink(
      atPath: tempRoot.appendingPathComponent("linked").path, withDestinationPath: outside.path)
    // And a directory symlink into a (non-existent) removable volume next to it.
    try FileManager.default.createSymbolicLink(
      atPath: tempRoot.appendingPathComponent("quarantine").path, withDestinationPath: "/Volumes/Nope/quarantine")
    let wanted = try touch("vault/deeper/wanted.safetensors")

    let index = LoRAEntry.buildLocalIndex(roots: [tempRoot.path])
    XCTAssertEqual(index["wanted.safetensors"], wanted, "real nested files are still found")
    XCTAssertNil(index["behind_link.safetensors"], "a symlinked directory is never descended")

    let preflight = try run([
      LoRAEntry(path: "behind_link.safetensors", scale: nil),
      LoRAEntry(path: "wanted.safetensors", scale: nil),
    ])
    XCTAssertEqual(preflight.resolved.map(\.path), [wanted])
    XCTAssertEqual(preflight.unresolved.map(\.path), ["behind_link.safetensors"])
  }

  func testCandidatesUnderVolumesAreReportedRemovableNotStatted() throws {
    // An absolute /Volumes entry in the payload.
    let absolute = try run([LoRAEntry(path: "/Volumes/Nope/Models/loras/x.safetensors", scale: nil)], nearline: noNearline)
    XCTAssertEqual(absolute.unresolved.count, 1)
    XCTAssertTrue(absolute.unresolved[0].reason.hasPrefix("removable-volume"), absolute.unresolved[0].reason)
    // A library.json entry whose absolute relative_path points at /Volumes.
    let indexed = try run(
      [LoRAEntry(path: "archived.safetensors", scale: nil)],
      lookup: { _ in "/Volumes/Nope/archived.safetensors" }, nearline: noNearline)
    XCTAssertTrue(indexed.unresolved[0].reason.hasPrefix("removable-volume"), indexed.unresolved[0].reason)
    // A local symlink into /Volumes named by bare filename: found by the
    // root-relative check as a link, reported removable, never followed.
    try FileManager.default.createSymbolicLink(
      atPath: tempRoot.appendingPathComponent("vault/linked.safetensors").path,
      withDestinationPath: "/Volumes/Nope/linked.safetensors")
    let viaLink = try run([LoRAEntry(path: "vault/linked.safetensors", scale: nil)], nearline: noNearline)
    XCTAssertTrue(viaLink.unresolved[0].reason.hasPrefix("removable-volume"), viaLink.unresolved[0].reason)
    // The generate/preset path (which resolves against COMFYBOX_MODELS)
    // refuses the same candidate loudly instead of stat'ing it.
    let original = ProcessInfo.processInfo.environment["COMFYBOX_MODELS"]
    setenv("COMFYBOX_MODELS", tempRoot.path, 1)
    defer {
      if let original { setenv("COMFYBOX_MODELS", original, 1) } else { unsetenv("COMFYBOX_MODELS") }
    }
    XCTAssertThrowsError(try LoRAEntry(path: "vault/linked.safetensors", scale: nil).makeConfiguration()) { error in
      guard case WarmServerError.invalidRequest(let message) = error else { return XCTFail("\(error)") }
      XCTAssertTrue(message.contains("removable volume"), message)
    }
  }

  func testWalkIsDepthBounded() throws {
    let atSix = try touch("l1/l2/l3/l4/l5/deep6.safetensors")
    try touch("l1/l2/l3/l4/l5/l6/deep7.safetensors")
    let index = LoRAEntry.buildLocalIndex(roots: [tempRoot.path])
    XCTAssertEqual(index["deep6.safetensors"], atSix)
    XCTAssertNil(index["deep7.safetensors"], "depth \(LoRAEntry.walkMaxDepth + 1) is beyond the bound")
    XCTAssertEqual(LoRAEntry.walkMaxDepth, 6)
    XCTAssertEqual(LoRAEntry.walkMaxEntries, 20_000)
    XCTAssertEqual(LoRAEntry.walkMaxSeconds, 2)
  }

  func testWalkResultIsCachedUntilInvalidated() throws {
    try touch("vault/first.safetensors")
    XCTAssertNotNil(LoRAEntry.localIndex(for: [tempRoot.path])["first.safetensors"])
    let late = try touch("vault/late.safetensors")
    XCTAssertNil(LoRAEntry.localIndex(for: [tempRoot.path])["late.safetensors"], "served from the cache")
    LoRAEntry.invalidateLocalIndex()  // what POST /v1/loras/scan does
    XCTAssertEqual(LoRAEntry.localIndex(for: [tempRoot.path])["late.safetensors"], late)
  }

  // MARK: - nearline: only for names not on local disk, bounded

  func testNearlineIsOnlyConsultedForNamesNotOnLocalDisk() throws {
    let local = try touch("vault/on_disk.safetensors")
    var staged: [String] = []
    let preflight = try run(
      [
        LoRAEntry(path: "on_disk.safetensors", scale: nil),
        LoRAEntry(path: "archived.safetensors", scale: 0.4, role: "kroma"),
      ],
      nearline: { name in staged.append(name); return .staged("/staged/\(name)") })
    XCTAssertEqual(staged, ["archived.safetensors"], "a name already on disk never goes to nearline (#396)")
    XCTAssertEqual(preflight.resolved.map(\.path), [local, "/staged/archived.safetensors"])
    XCTAssertEqual(preflight.resolved[1].role, "kroma")
    XCTAssertFalse(preflight.hasUnresolved)
  }

  func testNearlineTimeoutIsReportedUnresolvedWithRetryHint() throws {
    let preflight = try run([LoRAEntry(path: "slow.safetensors", scale: nil)], nearline: { _ in .timedOut })
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

  func testNearlineInFlightPolicyDefaults() {
    XCTAssertEqual(WarmServer.nearlineStageInFlightMaxAge, 600)
    XCTAssertEqual(WarmServer.nearlineStageMaxParkedThreads, 8)
    XCTAssertEqual(LoRASwapPreflight.nearlineStageTimeout, 10)
  }

  // MARK: - (4) persisted swap replay: a poison job is dropped, a good one replays

  func testPersistedPoisonSwapIsDroppedAtReplay() throws {
    let rawBody = try XCTUnwrap(Data(base64Encoded: "eyJsb3JhcyI6W3sicGF0aCI6IiIsInNjYWxlIjowLjV9XX0="))
    let job = PersistedQueueJob(id: "D9A75100", kind: "lora_swap", source: "api", enqueuedAt: Date(), rawBody: rawBody)

    var failure: Error?
    do {
      let payload = try WarmServer.decode(LoRASwapPayload.self, from: job.rawBody)
      _ = try run(payload.loras, nearline: noNearline)
    } catch { failure = error }
    let error = try XCTUnwrap(failure, "the incident's persisted job must fail preflight, never enqueue")

    let tracker = ImageJobTracker()
    tracker.recordFailedReplay(jobId: job.id, source: job.source, error: error)
    let status = try XCTUnwrap(tracker.status(jobId: "D9A75100"))
    XCTAssertEqual(status.status, .failed)
  }

  func testPersistedSwapWithAMissingEntryIsPoison() throws {
    try touch("vault/still_here.safetensors")
    let rawBody = Data(#"{"loras":[{"path":"still_here.safetensors","scale":0.5},{"path":"Gone_V1.safetensors"}]}"#.utf8)
    let payload = try WarmServer.decode(LoRASwapPayload.self, from: rawBody)
    let preflight = try run(payload.loras)
    let reason = try XCTUnwrap(LoRASwapPreflight.poisonReason(preflight), "the live route would 400 it → dropped, not replayed")
    XCTAssertTrue(reason.contains("Gone_V1.safetensors"), reason)
  }

  func testPersistedSwapWhoseEntriesAllResolveStillReplays() throws {
    let present = try touch("vault/still_here.safetensors")
    let rawBody = Data(#"{"loras":[{"path":"still_here.safetensors","scale":0.5}]}"#.utf8)
    let payload = try WarmServer.decode(LoRASwapPayload.self, from: rawBody)
    let preflight = try run(payload.loras, nearline: noNearline)
    XCTAssertNil(LoRASwapPreflight.poisonReason(preflight))
    XCTAssertEqual(preflight.resolved.map(\.path), [present])
  }

  // MARK: - response shape: 200 is the pre-existing shape, `unresolved` lives on the 400

  func testSuccessResponseShapeIsUnchanged() throws {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let body = String(decoding: try encoder.encode(LoRASwapResponse(success: true, loraCount: 1, loras: [])), as: UTF8.self)
    XCTAssertTrue(body.contains("\"lora_count\":1"), body)
    XCTAssertFalse(body.contains("unresolved"), "a 200 means every entry applied: \(body)")
  }
}
