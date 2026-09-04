import Foundation
import XCTest

@testable import ZImage

/// #282 — the per-job stack precedence, in isolation.
///
/// `request loras > expanded preset > warm default`, decided on PRESENCE, so
/// "render with no adapters" is expressible and cannot be silently overridden
/// by the last `/v1/lora/swap`.
final class RequestStackResolverTests: XCTestCase {

  // MARK: - Precedence

  func testRequestLorasWinOverPresetAndWarmDefault() {
    let resolved = RequestStackResolver.resolve(
      requestLoras: ["req.safetensors"],
      presetStack: ["preset.safetensors"],
      warmDefault: ["warm.safetensors"])
    XCTAssertEqual(resolved.origin, .request)
    XCTAssertEqual(resolved.stack, ["req.safetensors"])
  }

  func testPresetStackWinsOverWarmDefault() {
    let resolved = RequestStackResolver.resolve(
      requestLoras: nil,
      presetStack: ["kroma.safetensors", "content.safetensors"],
      warmDefault: ["warm.safetensors"])
    XCTAssertEqual(resolved.origin, .preset)
    XCTAssertEqual(resolved.stack, ["kroma.safetensors", "content.safetensors"])
  }

  func testNeitherPresetNorLorasTakesTheWarmDefault() {
    let resolved = RequestStackResolver.resolve(
      requestLoras: nil, presetStack: nil, warmDefault: ["warm.safetensors"])
    XCTAssertEqual(resolved.origin, .warmDefault)
    XCTAssertEqual(resolved.stack, ["warm.safetensors"])
  }

  /// The whole point of #282: a bare request takes the warm default, NOT the
  /// last job's stack. Two jobs in a row with different explicit stacks, then
  /// a bare one — the bare one must not inherit either of them.
  func testBareRequestDoesNotInheritTheStackOfAnEarlierJob() {
    let warm = ["warm.safetensors"]
    let jobA = RequestStackResolver.resolve(
      requestLoras: ["a.safetensors"], presetStack: nil, warmDefault: warm)
    let jobB = RequestStackResolver.resolve(
      requestLoras: nil, presetStack: ["b.safetensors"], warmDefault: warm)
    let jobC = RequestStackResolver.resolve(
      requestLoras: nil, presetStack: nil, warmDefault: warm)

    XCTAssertEqual(jobA.stack, ["a.safetensors"])
    XCTAssertEqual(jobB.stack, ["b.safetensors"])
    XCTAssertEqual(jobC.stack, warm, "a bare job inherited an earlier job's stack")
    XCTAssertEqual(jobC.origin, .warmDefault)
  }

  // MARK: - Presence, not emptiness

  func testExplicitlyEmptyRequestArrayIsAStatementAndBeatsTheWarmDefault() {
    let resolved = RequestStackResolver.resolve(
      requestLoras: [] as [String], presetStack: ["preset.safetensors"],
      warmDefault: ["warm.safetensors"])
    XCTAssertEqual(resolved.origin, .request)
    XCTAssertEqual(resolved.stack, [], "`loras: []` means no adapters, not `use the default`")
  }

  /// The seeded `zimage-chat` preset declares `loras: []`. #286 ruled that an
  /// empty preset stack CLEARS the resident adapters; #282 must not turn that
  /// back into "take the warm default".
  func testExplicitlyEmptyPresetStackIsAStatementAndBeatsTheWarmDefault() {
    let resolved = RequestStackResolver.resolve(
      requestLoras: nil, presetStack: [] as [String], warmDefault: ["warm.safetensors"])
    XCTAssertEqual(resolved.origin, .preset)
    XCTAssertEqual(resolved.stack, [])
  }

  func testAnEmptyWarmDefaultIsAnAnswerNotAFallthrough() {
    let resolved = RequestStackResolver.resolve(
      requestLoras: nil, presetStack: nil, warmDefault: [] as [String])
    XCTAssertEqual(resolved.origin, .warmDefault)
    XCTAssertEqual(resolved.stack, [])
  }

  // MARK: - The decision on its own

  func testOriginFromPresenceAlone() {
    XCTAssertEqual(
      RequestStackResolver.origin(hasRequestLoras: true, hasPresetStack: true), .request)
    XCTAssertEqual(
      RequestStackResolver.origin(hasRequestLoras: true, hasPresetStack: false), .request)
    XCTAssertEqual(
      RequestStackResolver.origin(hasRequestLoras: false, hasPresetStack: true), .preset)
    XCTAssertEqual(
      RequestStackResolver.origin(hasRequestLoras: false, hasPresetStack: false), .warmDefault)
  }

  // MARK: - The wire spellings

  func testOriginWireSpellings() {
    XCTAssertEqual(RequestStackResolver.Origin.request.rawValue, "request")
    XCTAssertEqual(RequestStackResolver.Origin.preset.rawValue, "preset")
    XCTAssertEqual(RequestStackResolver.Origin.warmDefault.rawValue, "warm_default")
    XCTAssertEqual(RequestStackResolver.Origin.allCases.count, 3)
  }

  // MARK: - The family gate

  /// FIBO and Chroma have no LoRA path (`/v1/lora/swap` refuses them). A stack
  /// the JOB named is still applied there exactly as before #282; only a warm
  /// default — which no such caller ever asked for — is skipped.
  func testWarmDefaultIsNotPushedAtAFamilyWithNoLoRAPath() {
    XCTAssertFalse(
      RequestStackResolver.appliesAtDequeue(origin: .warmDefault, familyHasLoRAPath: false))
    XCTAssertTrue(
      RequestStackResolver.appliesAtDequeue(origin: .request, familyHasLoRAPath: false))
    XCTAssertTrue(
      RequestStackResolver.appliesAtDequeue(origin: .preset, familyHasLoRAPath: false))
  }

  func testEveryOriginAppliesOnAFamilyWithALoRAPath() {
    for origin in RequestStackResolver.Origin.allCases {
      XCTAssertTrue(
        RequestStackResolver.appliesAtDequeue(origin: origin, familyHasLoRAPath: true),
        "\(origin) must be applied on a family that has a LoRA path")
    }
  }

  // MARK: - Real payload element types

  /// The production element type, so the generic is exercised on what
  /// `runGenerate` actually passes rather than only on `String`.
  func testResolvesOverLoRAConfigurations() {
    let warm = [LoRAConfiguration.local("/tmp/warm.safetensors", scale: 0.5)]
    let job = [LoRAConfiguration.local("/tmp/job.safetensors", scale: 0.8)]

    let explicit = RequestStackResolver.resolve(
      requestLoras: job, presetStack: nil, warmDefault: warm)
    XCTAssertEqual(explicit.origin, .request)
    XCTAssertEqual(explicit.stack, job)

    let bare = RequestStackResolver.resolve(
      requestLoras: nil, presetStack: nil, warmDefault: warm)
    XCTAssertEqual(bare.origin, .warmDefault)
    XCTAssertEqual(bare.stack, warm)
  }
}
