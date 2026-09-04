import Foundation
import XCTest

@testable import ZImage

/// #282 — the dequeue side: what `runGenerate` resolves for a given payload,
/// and what `/v1/lora/swap` leaves behind.
///
/// Driven through the `WarmServerQueueProbe` `#if DEBUG` seam, which calls
/// `WarmServerCoordinator.resolveJobLoRAStack` — the SAME function the dequeue
/// path calls before applying a stack. The application itself needs model
/// weights and a GPU, so it cannot run in a unit test (intent.md: agents run
/// unit tests only); what CAN be proved without weights is the decision, plus
/// one real end-to-end swap (an EMPTY swap unloads rather than loads, so it
/// completes with no pipeline resident).
final class PerRequestStackDequeueTests: XCTestCase {

  override func setUpWithError() throws {
    try super.setUpWithError()
    try isolateComfyBoxStateDirectory()
  }

  // MARK: Helpers

  private func payload(loras: [LoRAEntry]?, presetOwned: Bool = false) -> GeneratePayload {
    var p = GeneratePayload(prompt: "a test render", loras: loras)
    if presetOwned { p.presetStackApplied = true }
    return p
  }

  private func entry(_ path: String, _ scale: Float = 1.0, role: String? = nil) -> LoRAEntry {
    LoRAEntry(path: path, scale: scale, role: role)
  }

  private func config(_ path: String, _ scale: Float = 1.0) -> LoRAConfiguration {
    .local(path, scale: scale)
  }

  // MARK: - The three origins, at dequeue

  func testExplicitLorasAreRequestOwned() async throws {
    let probe = makeQueueProbe()
    await probe.adoptWarmDefaultStack([config("/tmp/warm.safetensors")])

    let resolved = try await probe.resolveJobStack(
      payload(loras: [entry("/tmp/explicit.safetensors", 0.8)]))
    XCTAssertEqual(resolved.origin, "request")
    XCTAssertEqual(resolved.names, ["explicit.safetensors"])
  }

  /// #286 puts a named preset's expanded stack in the same `loras` field.
  /// `presetStackApplied` is how the dequeue tells the two owners apart.
  func testPresetExpandedStackIsPresetOwned() async throws {
    let probe = makeQueueProbe()
    await probe.adoptWarmDefaultStack([config("/tmp/warm.safetensors")])

    let resolved = try await probe.resolveJobStack(
      payload(
        loras: [entry("/tmp/kroma.safetensors", 0.6, role: "kroma"),
                entry("/tmp/content.safetensors", 0.9)],
        presetOwned: true))
    XCTAssertEqual(resolved.origin, "preset")
    XCTAssertEqual(resolved.names, ["kroma.safetensors", "content.safetensors"])
  }

  func testARequestThatNamesNeitherTakesTheWarmDefault() async throws {
    let probe = makeQueueProbe()
    await probe.adoptWarmDefaultStack([
      config("/tmp/warm-a.safetensors", 0.6), config("/tmp/warm-b.safetensors", 0.4),
    ])

    let resolved = try await probe.resolveJobStack(payload(loras: nil))
    XCTAssertEqual(resolved.origin, "warm_default")
    XCTAssertEqual(resolved.names, ["warm-a.safetensors", "warm-b.safetensors"])
  }

  /// A fresh engine with no `--lora` arguments and no swap yet: a bare render
  /// resolves to an EMPTY stack, not to "whatever is resident".
  func testAFreshCoordinatorResolvesABareRequestToAnEmptyStack() async throws {
    let probe = makeQueueProbe()
    let resolved = try await probe.resolveJobStack(payload(loras: nil))
    XCTAssertEqual(resolved.origin, "warm_default")
    XCTAssertEqual(resolved.names, [])
  }

  // MARK: - No crosstalk

  /// The defect #282 exists to close, in the shape Todd will drive live: two
  /// daemons alternating, one sending explicit stacks and one sending presets,
  /// with a bare request in between. No job may pick up another job's stack,
  /// and no job may change what a later bare request gets.
  func testAJobsOwnStackNeverBecomesTheNextJobsDefault() async throws {
    let probe = makeQueueProbe()
    await probe.adoptWarmDefaultStack([config("/tmp/swap.safetensors", 0.5)])

    let daemonA = try await probe.resolveJobStack(
      payload(loras: [entry("/tmp/daemon-a.safetensors")]))
    XCTAssertEqual(daemonA.names, ["daemon-a.safetensors"])

    let daemonB = try await probe.resolveJobStack(
      payload(loras: [entry("/tmp/daemon-b.safetensors")], presetOwned: true))
    XCTAssertEqual(daemonB.names, ["daemon-b.safetensors"])
    XCTAssertEqual(daemonB.origin, "preset")

    let bare = try await probe.resolveJobStack(payload(loras: nil))
    XCTAssertEqual(
      bare.names, ["swap.safetensors"],
      "a bare request inherited an earlier job's stack — #282's crosstalk is back")
    let stillTheDefault = await probe.warmDefaultStackNames()
    XCTAssertEqual(
      stillTheDefault, ["swap.safetensors"],
      "a per-job stack must never be adopted as the warm default")
  }

  /// `loras: []` is a statement ("render bare"), and it must beat the warm
  /// default — otherwise the only way to ask for no adapters would be to swap.
  func testAnExplicitlyEmptyStackBeatsTheWarmDefault() async throws {
    let probe = makeQueueProbe()
    await probe.adoptWarmDefaultStack([config("/tmp/warm.safetensors")])

    let resolved = try await probe.resolveJobStack(payload(loras: []))
    XCTAssertEqual(resolved.origin, "request")
    XCTAssertEqual(resolved.names, [])
  }

  /// The seeded `zimage-chat` preset declares `loras: []`; #286 ruled that
  /// clears the resident stack. It must not fall back to the warm default.
  func testAnEmptyPresetStackBeatsTheWarmDefault() async throws {
    let probe = makeQueueProbe()
    await probe.adoptWarmDefaultStack([config("/tmp/warm.safetensors")])

    let resolved = try await probe.resolveJobStack(payload(loras: [], presetOwned: true))
    XCTAssertEqual(resolved.origin, "preset")
    XCTAssertEqual(resolved.names, [])
  }

  // MARK: - The swap, end to end

  /// `POST /v1/lora/swap` through the REAL queue. An empty swap is the one
  /// swap a unit test can run — `applyActiveLoRAs([])` unloads rather than
  /// loads — so this proves `runSwap` actually calls `adoptWarmDefaultStack`,
  /// not merely that the function works.
  func testSwapPublishesTheWarmDefaultThroughTheRealQueue() async throws {
    let probe = makeQueueProbe()
    await probe.adoptWarmDefaultStack([config("/tmp/stale.safetensors")])
    let seeded = await probe.warmDefaultStackNames()
    XCTAssertEqual(seeded, ["stale.safetensors"])

    let count = try await probe.enqueueSwap(loras: [])
    XCTAssertEqual(count, 0, "the swap response is unchanged: lora_count for an empty swap is 0")

    let afterSwap = await probe.warmDefaultStackNames()
    XCTAssertEqual(
      afterSwap, [],
      "runSwap did not publish the warm default — deleting adoptWarmDefaultStack from runSwap "
        + "would leave every bare render on a stale stack")
    let bare = try await probe.resolveJobStack(payload(loras: nil))
    XCTAssertEqual(bare.names, [])
  }

  // MARK: - `warm_default_stack` on /v1/model/pool

  func testPoolListReportsTheWarmDefaultStack() async throws {
    let probe = makeQueueProbe()
    await probe.adoptWarmDefaultStack([
      config("/tmp/kroma.safetensors", 0.6), config("/tmp/content.safetensors", 0.9),
    ])

    let poolStack = await probe.poolWarmDefaultStack()
    let reported = try XCTUnwrap(poolStack)
    XCTAssertEqual(reported.map(\.name), ["kroma.safetensors", "content.safetensors"])
    XCTAssertEqual(reported.map(\.scale), [0.6, 0.9])
    XCTAssertEqual(reported.map(\.path), ["/tmp/kroma.safetensors", "/tmp/content.safetensors"])
  }

  /// The wire spelling, and that it is ADDITIVE — every pre-#282 key is still
  /// exactly where it was.
  func testWarmDefaultStackWireShapeIsAdditive() throws {
    let response = ModelPoolListResponse(
      active: "krea2-raw", pool: [], totalVramMB: 0, budgetMB: 22528,
      warmDefaultStack: [LoRAState(.local("/tmp/warm.safetensors", scale: 0.7))])
    let json = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: JSONEncoder().encode(response)) as? [String: Any])

    XCTAssertEqual(json["active"] as? String, "krea2-raw")
    XCTAssertEqual(json["budget_mb"] as? Int, 22528)
    XCTAssertEqual(json["total_vram_mb"] as? Int, 0)
    let stack = try XCTUnwrap(json["warm_default_stack"] as? [[String: Any]])
    XCTAssertEqual(stack.count, 1)
    XCTAssertEqual(stack[0]["name"] as? String, "warm.safetensors")
    XCTAssertEqual(stack[0]["source"] as? String, "/tmp/warm.safetensors")
  }

  /// A response built without a warm default omits the key entirely, so a
  /// client can tell "not reported" from "empty".
  func testWarmDefaultStackKeyIsAbsentWhenNotSupplied() throws {
    let response = ModelPoolListResponse(
      active: nil, pool: [], totalVramMB: 0, budgetMB: 0)
    let json = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: JSONEncoder().encode(response)) as? [String: Any])
    XCTAssertNil(json["warm_default_stack"])
  }

  // MARK: - `lora_stack_origin` on the response

  func testGenerateResponseCarriesTheOriginAdditively() throws {
    let response = GenerateResponse(
      success: true, outputPath: "/tmp/out.png", durationMs: 1234,
      loraStackOrigin: RequestStackResolver.Origin.preset.rawValue)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let json = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: encoder.encode(response)) as? [String: Any])
    XCTAssertEqual(json["lora_stack_origin"] as? String, "preset")
    XCTAssertEqual(json["output_path"] as? String, "/tmp/out.png")
  }

  func testGenerateResponseOmitsTheOriginWhenUnset() throws {
    let response = GenerateResponse(success: true, outputPath: "/tmp/out.png", durationMs: 1)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let json = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: encoder.encode(response)) as? [String: Any])
    XCTAssertNil(json["lora_stack_origin"])
  }

  // MARK: - The route marks the owner (the wiring #350 left untested)

  /// `presetStackApplied` is set in ONE place — `GeneratePayload.expandingPreset`
  /// — and read in one place, the dequeue resolver. These drive the real route
  /// function `/v1/generate` and `/v1/generate/async` call, so deleting that
  /// assignment fails here rather than only changing a label nobody asserts.

  private func makePresetStore() throws -> PresetStore {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-282-presets-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    return PresetStore(path: dir.appendingPathComponent("presets.json"), seedDefaults: false)
  }

  /// Absolute paths so `LoRAEntry.makeConfiguration()` takes them at their word
  /// (no library search, no file needed) — the resolver is what is under test,
  /// not source resolution.
  private func seedPreset(_ store: PresetStore) throws {
    try store.upsert(ImagePreset(
      id: "krea-kira", name: "Kira", mediaKind: "image", model: "krea2-raw",
      loras: [
        LoraReference(filename: "/tmp/accel.safetensors", scale: 0.6, role: "accel"),
        LoraReference(filename: "/tmp/polish.safetensors", scale: 0.4),
      ],
      checkpointFamily: "raw-accel",
      kroma: KromaPolicy(strength: 0.6, file: "/tmp/kroma.safetensors")))
  }

  private func route(_ json: String, store: PresetStore) throws -> GeneratePayload {
    try WarmServer.decodedGeneratePayload(
      from: Data(json.utf8), store: store,
      configuration: WarmServerConfiguration(allowedOutputDirectory: NSTemporaryDirectory()),
      loraExists: { _ in true })
  }

  /// The whole point, end to end: the body Kira's daemon posts goes through the
  /// real route, and the dequeue applies THE PRESET'S stack — not the warm
  /// default sitting under it from an earlier swap.
  func testAPresetOnlyBodyResolvesToThePresetStackNotTheWarmDefault() async throws {
    let probe = makeQueueProbe()
    await probe.adoptWarmDefaultStack([config("/tmp/swapped.safetensors", 0.5)])
    let store = try makePresetStore()
    try seedPreset(store)

    let expanded = try route(#"{"prompt":"a portrait","preset":"krea-kira"}"#, store: store)
    XCTAssertEqual(
      expanded.presetStackApplied, true,
      "the route did not mark the stack preset-owned — the dequeue would report it as the "
        + "request's own")

    let resolved = try await probe.resolveJobStack(expanded)
    XCTAssertEqual(resolved.origin, "preset")
    XCTAssertEqual(
      resolved.names,
      ["kroma.safetensors", "accel.safetensors", "polish.safetensors"],
      "a preset render fell back to the warm default")
  }

  /// A request that sends `preset` AND its own flat `loras` (the production
  /// async client's shape) stays request-owned — explicit still wins, and it is
  /// not relabelled as the preset's.
  func testAnExplicitStackBesideAPresetStaysRequestOwned() async throws {
    let probe = makeQueueProbe()
    let store = try makePresetStore()
    try seedPreset(store)

    let expanded = try route(
      #"{"prompt":"p","preset":"krea-kira","loras":[{"path":"/tmp/mine.safetensors","scale":1.0}]}"#,
      store: store)
    XCTAssertNil(expanded.presetStackApplied)

    let resolved = try await probe.resolveJobStack(expanded)
    XCTAssertEqual(resolved.origin, "request")
    XCTAssertEqual(resolved.names, ["mine.safetensors"])
  }

  /// #286's contract: an unexpandable preset stays a LABEL and is never a 400.
  /// #282's addition: what it then renders with is the WARM DEFAULT — a named
  /// default — rather than whatever the previous job left resident.
  func testAnUnresolvablePresetRendersOnTheWarmDefault() async throws {
    let probe = makeQueueProbe()
    await probe.adoptWarmDefaultStack([config("/tmp/swapped.safetensors", 0.5)])
    let store = try makePresetStore()

    let expanded = try route(#"{"prompt":"p","preset":"no-such-preset"}"#, store: store)
    XCTAssertEqual(expanded.presetUnresolved, "no-such-preset")
    XCTAssertEqual(expanded.presetUnresolvedReason, "unknown_preset")
    XCTAssertNil(expanded.loras)
    XCTAssertNil(expanded.presetStackApplied)

    let resolved = try await probe.resolveJobStack(expanded)
    XCTAssertEqual(resolved.origin, "warm_default")
    XCTAssertEqual(resolved.names, ["swapped.safetensors"])
  }
}
