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

  /// Review r1 (M1): a family with no LoRA path applies NOTHING, whatever the
  /// origin. The first cut skipped only the warm default, which left a
  /// request-named stack being loaded into the Flux-1 pipeline that Chroma and
  /// FIBO do not render through — no effect on the pixels, and `/health.loras`
  /// plus the PNG then named adapters that took no part.
  func testAFamilyWithNoLoRAPathAppliesNothingWhateverTheOrigin() {
    for origin in RequestStackResolver.Origin.allCases {
      XCTAssertFalse(
        RequestStackResolver.appliesAtDequeue(origin: origin, familyHasLoRAPath: false),
        "\(origin) must not be loaded into a pipeline the family does not render through")
    }
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

  // MARK: - Review r1 (C1): the warm default is only valid for its own base

  private func tag(_ family: String?, _ model: String? = nil)
    -> RequestStackResolver.WarmDefaultTag {
    .init(family: family, modelSpec: model)
  }

  /// The Critical: a stack published under krea2-raw must not be force-loaded
  /// into a Flux-2 pipeline by a bare request that switched base. That load can
  /// throw, which would turn a request that always rendered into a 500.
  func testADifferentFamilySkipsTheWarmDefault() {
    let decision = RequestStackResolver.admitWarmDefault(
      isEmpty: false, tag: tag("krea2", "/m/krea2-raw"),
      requestFamily: "flux2", requestModelSpec: "/m/flux2")
    XCTAssertEqual(decision, .skip(reason: "family_mismatch"))
    XCTAssertEqual(
      RequestStackResolver.WarmDefaultAdmission.familyMismatch, "family_mismatch")
  }

  /// Same family, different checkpoint: still the wrong base for those
  /// adapters — a krea2 stack on the wrong krea2 binds and looks subtly wrong,
  /// which is exactly #286's silent-wrong-look defect.
  func testTheSameFamilyOnADifferentCheckpointSkipsTheWarmDefault() {
    let decision = RequestStackResolver.admitWarmDefault(
      isEmpty: false, tag: tag("krea2", "/m/krea2-raw"),
      requestFamily: "krea2", requestModelSpec: "/m/krea2-turbo")
    XCTAssertEqual(decision, .skip(reason: "model_mismatch"))
    XCTAssertEqual(
      RequestStackResolver.WarmDefaultAdmission.modelMismatch, "model_mismatch")
  }

  func testTheSameBaseAdmitsTheWarmDefault() {
    XCTAssertEqual(
      RequestStackResolver.admitWarmDefault(
        isEmpty: false, tag: tag("krea2", "/m/krea2-raw"),
        requestFamily: "krea2", requestModelSpec: "/m/krea2-raw"),
      .admit)
  }

  /// "Clear the adapters" is base-agnostic and cannot throw, so an empty
  /// default is admitted even across a family change — otherwise a bare
  /// request would be stuck with the previous job's stack, which is the very
  /// inheritance this ticket retires.
  func testAnEmptyWarmDefaultIsAdmittedEvenAcrossFamilies() {
    XCTAssertEqual(
      RequestStackResolver.admitWarmDefault(
        isEmpty: true, tag: tag("krea2", "/m/krea2-raw"),
        requestFamily: "flux1", requestModelSpec: "/m/z-image"),
      .admit)
  }

  /// The launch-time `--lora` stack has no swap behind it. Refusing it would
  /// change boot behaviour rather than protect anything.
  func testAnUntaggedWarmDefaultIsAdmittedAnywhere() {
    XCTAssertEqual(
      RequestStackResolver.admitWarmDefault(
        isEmpty: false, tag: .untagged, requestFamily: "krea2", requestModelSpec: "/m/anything"),
      .admit)
    XCTAssertNil(RequestStackResolver.WarmDefaultTag.untagged.family)
    XCTAssertNil(RequestStackResolver.WarmDefaultTag.untagged.modelSpec)
  }

  /// An unknown spec on either side is not a mismatch: the family agreed, and
  /// refusing on ignorance would strand the ordinary case where the engine
  /// never recorded a spec.
  func testAnUnknownSpecOnEitherSideIsNotAMismatch() {
    XCTAssertEqual(
      RequestStackResolver.admitWarmDefault(
        isEmpty: false, tag: tag("krea2", nil),
        requestFamily: "krea2", requestModelSpec: "/m/krea2-raw"),
      .admit)
    XCTAssertEqual(
      RequestStackResolver.admitWarmDefault(
        isEmpty: false, tag: tag("krea2", "/m/krea2-raw"),
        requestFamily: "krea2", requestModelSpec: nil),
      .admit)
    XCTAssertEqual(
      RequestStackResolver.admitWarmDefault(
        isEmpty: false, tag: tag("krea2", ""),
        requestFamily: "krea2", requestModelSpec: ""),
      .admit)
  }
}
