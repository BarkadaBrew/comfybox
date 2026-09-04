import Foundation
import XCTest

@testable import ZImage

/// Task #9 Phase 1 (specs/ltx2-parameter-externalization.md §4): a single
/// resolution pass that yields every Tier A/B parameter WITH provenance.
/// The acceptance case is the 2026-07-30 incident made visible: with
/// LTX2_GUIDANCE_RESCALE missing, the readout must show
/// `guidance_rescale 0 (builtin)` — not silently behave differently.
final class LTX2ConfigResolverTests: XCTestCase {

  private func param(_ name: String, _ params: [LTX2ResolvedParam]) -> LTX2ResolvedParam? {
    params.first { $0.name == name }
  }

  func testAbsentEnvYieldsBuiltinWithProvenance() {
    let params = LTX2ConfigResolver.resolveEffective(environment: [:], configFile: [:])

    let rescale = param("guidance_rescale", params)
    XCTAssertEqual(rescale?.source, .builtin)
    XCTAssertEqual(rescale?.value, "0")

    let maxVol = param("plain_decode_max_vol", params)
    XCTAssertEqual(maxVol?.source, .builtin)
    XCTAssertEqual(maxVol?.value, "4500")
  }

  func testEnvValueWinsOverBuiltinAndIsTrimmed() {
    let params = LTX2ConfigResolver.resolveEffective(
      environment: ["LTX2_GUIDANCE_RESCALE": " 0.7 "], configFile: [:])

    let rescale = param("guidance_rescale", params)
    XCTAssertEqual(rescale?.source, .env)
    XCTAssertEqual(rescale?.value, "0.7", "whitespace must be trimmed, not rejected")
    XCTAssertEqual(rescale?.valid, true)
  }

  func testConfigFileBeatsEnv() {
    let params = LTX2ConfigResolver.resolveEffective(
      environment: ["LTX2_REFINE_MAX_VOL": "9000"],
      configFile: ["refine_max_vol": "15000"])

    let maxVol = param("refine_max_vol", params)
    XCTAssertEqual(maxVol?.source, .configFile)
    XCTAssertEqual(maxVol?.value, "15000")
  }

  func testNonFiniteRejectedLoudlyWithBuiltinFallback() {
    let params = LTX2ConfigResolver.resolveEffective(
      environment: ["LTX2_GUIDANCE_RESCALE": "nan"], configFile: [:])

    let rescale = param("guidance_rescale", params)
    XCTAssertEqual(rescale?.valid, false, "non-finite must be flagged, never silently used")
    XCTAssertEqual(rescale?.source, .builtin, "falls back to builtin")
    XCTAssertEqual(rescale?.value, "0")
    XCTAssertNotNil(rescale?.note)
  }

  func testOutOfRangeRejectedWithNote() {
    let params = LTX2ConfigResolver.resolveEffective(
      environment: ["LTX2_FACE_ANCHOR_STRENGTH": "3.0"], configFile: [:])

    let anchor = param("face_anchor_strength", params)
    XCTAssertEqual(anchor?.valid, false)
    XCTAssertEqual(anchor?.source, .builtin)
    XCTAssertTrue(anchor?.note?.contains("range") ?? false, "note names the violation")
  }

  func testMissingPathFlagged() {
    let params = LTX2ConfigResolver.resolveEffective(
      environment: ["LTX2_UPSAMPLER_PATH": "/nonexistent/upsampler"],
      configFile: [:],
      fileExists: { _ in false })

    let path = param("upsampler_path", params)
    XCTAssertEqual(path?.valid, false)
    XCTAssertEqual(path?.source, .env, "value is kept (it IS what will be used) but flagged")
  }

  func testBoolMirrorsRendererSemanticsExactly() {
    // Codex finding #17: the renderer enables two_stage ONLY for exact "1"
    // (WarmServer/generator point-of-use reads). The readout must never
    // claim a value the renderer won't act on: padded "1" is what the
    // renderer sees UNTRIMMED, so it must resolve false AND be flagged.
    let padded = LTX2ConfigResolver.resolveEffective(
      environment: ["LTX2_TWO_STAGE": " 1 "], configFile: [:])
    let p = padded.first { $0.name == "two_stage" }
    XCTAssertEqual(p?.value, "false", "renderer compares == \"1\" on the RAW value")
    XCTAssertEqual(p?.valid, false, "bool-ish but not exactly '1' — flag it")

    let truthy = LTX2ConfigResolver.resolveEffective(
      environment: ["LTX2_TWO_STAGE": "true"], configFile: [:])
    let t = truthy.first { $0.name == "two_stage" }
    XCTAssertEqual(t?.value, "false", "renderer does not accept 'true'")
    XCTAssertEqual(t?.valid, false, "flagged so the user learns the renderer's convention")

    let exact = LTX2ConfigResolver.resolveEffective(
      environment: ["LTX2_TWO_STAGE": "1"], configFile: [:])
    let e = exact.first { $0.name == "two_stage" }
    XCTAssertEqual(e?.value, "true")
    XCTAssertEqual(e?.valid, true)
  }

  func testBoolAndListKindsResolve() {
    let params = LTX2ConfigResolver.resolveEffective(
      environment: ["LTX2_TWO_STAGE": "1", "LTX2_REFINE_SIGMAS": "0.6,0.4,0.2"],
      configFile: [:])

    XCTAssertEqual(param("two_stage", params)?.value, "true")
    XCTAssertEqual(param("refine_sigmas", params)?.value, "0.6,0.4,0.2")
    XCTAssertEqual(param("refine_sigmas", params)?.valid, true)
  }

  func testMalformedFloatListRejected() {
    let params = LTX2ConfigResolver.resolveEffective(
      environment: ["LTX2_REFINE_SIGMAS": "0.6,potato,0.2"], configFile: [:])

    let sigmas = param("refine_sigmas", params)
    XCTAssertEqual(sigmas?.valid, false)
    XCTAssertEqual(sigmas?.source, .builtin)
  }

  func testEveryTierAAndBParamIsPresent() {
    let params = LTX2ConfigResolver.resolveEffective(environment: [:], configFile: [:])
    let names = Set(params.map(\.name))
    for expected in [
      "guidance_rescale", "cfg_schedule", "stage1_sigmas", "refine_sigmas",
      "two_stage", "cond_fps", "img_compression", "sampler", "stg_scale",
      "face_anchor_strength", "ic_control", "nag_scale", "nag_alpha", "nag_tau",
      "plain_decode_max_vol", "refine_max_vol", "decode_mode", "upsampler_path",
      "video_bits_per_px",
    ] {
      XCTAssertTrue(names.contains(expected), "registry missing \(expected)")
    }
  }
}

// MARK: - Phase 2: typed resolution with request/preset levels

extension LTX2ConfigResolverTests {

  func testRequestBeatsPresetBeatsEnvWithProvenance() {
    var request = LTX2VideoTuning(); request.guidanceRescale = 0.25
    var preset = LTX2VideoTuning(); preset.guidanceRescale = 0.5; preset.stgScale = 2.0

    let resolved = LTX2ConfigResolver.resolveTyped(
      request: request, preset: preset,
      environment: ["LTX2_GUIDANCE_RESCALE": "0.7", "LTX2_STG_SCALE": "4"],
      configFile: [:])

    XCTAssertEqual(resolved.guidanceRescale, 0.25, "request wins")
    XCTAssertEqual(resolved.provenance["guidance_rescale"], .request)
    XCTAssertEqual(resolved.stgScale, 2.0, "preset beats env")
    XCTAssertEqual(resolved.provenance["stg_scale"], .preset)
  }

  func testInvalidHighPrecedenceFallsThroughToNextSource() {
    // Codex finding #17: an invalid configFile value must fall through to a
    // VALID env value (flagged), not jump to builtin.
    let params = LTX2ConfigResolver.resolveEffective(
      environment: ["LTX2_REFINE_MAX_VOL": "9000"],
      configFile: ["refine_max_vol": "not-a-number"])

    let p = params.first { $0.name == "refine_max_vol" }
    XCTAssertEqual(p?.value, "9000", "falls through to the valid env value")
    XCTAssertEqual(p?.source, .env)
    XCTAssertEqual(p?.valid, false, "flagged because a higher-precedence source was rejected")
  }

  func testTypedDefaultsMatchRegistryBuiltins() {
    let resolved = LTX2ConfigResolver.resolveTyped(
      request: nil, preset: nil, environment: [:], configFile: [:])
    XCTAssertEqual(resolved.guidanceRescale, 0)
    XCTAssertEqual(resolved.twoStage, false)
    XCTAssertEqual(resolved.imgCompression, 35)
    XCTAssertEqual(resolved.faceAnchorStrength, 0.5)
    XCTAssertEqual(resolved.plainDecodeMaxVol, 4500)
    XCTAssertEqual(resolved.provenance["img_compression"], .builtin)
  }

  func testTypedTwoStageFromRequestBoolNotStringConvention() {
    var request = LTX2VideoTuning(); request.twoStage = true
    let resolved = LTX2ConfigResolver.resolveTyped(
      request: request, preset: nil, environment: [:], configFile: [:])
    XCTAssertEqual(resolved.twoStage, true)
    XCTAssertEqual(resolved.provenance["two_stage"], .request)
  }

  func testTypedSigmasParseAndRequestOverrides() {
    var preset = LTX2VideoTuning(); preset.refineSigmas = [0.85, 0.725, 0.4219, 0.0]
    let resolved = LTX2ConfigResolver.resolveTyped(
      request: nil, preset: preset,
      environment: ["LTX2_REFINE_SIGMAS": "0.9,0.5,0.0"], configFile: [:])
    XCTAssertEqual(resolved.refineSigmas, [0.85, 0.725, 0.4219, 0.0])
    XCTAssertEqual(resolved.provenance["refine_sigmas"], .preset)
  }

  // MARK: - comfybox#307: `LTX2VideoTuning.merging` (top-level `two_pass`)

  func testMergingNilTwoPassLeavesTuningUntouched() {
    var base = LTX2VideoTuning(); base.refineScale = 1.35
    let merged = LTX2VideoTuning.merging(base, twoPass: nil)
    XCTAssertEqual(merged?.refineScale, 1.35)
    XCTAssertNil(merged?.twoStage)
  }

  func testMergingNilTuningAndNilTwoPassStaysNil() {
    XCTAssertNil(LTX2VideoTuning.merging(nil, twoPass: nil))
  }

  func testMergingTwoPassTrueOnNoTuningCreatesOne() {
    let merged = LTX2VideoTuning.merging(nil, twoPass: true)
    XCTAssertEqual(merged?.twoStage, true)
  }

  func testMergingTwoPassFalseOnNoTuningCreatesOne() {
    let merged = LTX2VideoTuning.merging(nil, twoPass: false)
    XCTAssertEqual(merged?.twoStage, false)
  }

  func testMergingTwoPassFillsInWhenTuningTwoStageUnset() {
    var base = LTX2VideoTuning(); base.refineScale = 1.35  // some other field set, twoStage nil
    let merged = LTX2VideoTuning.merging(base, twoPass: true)
    XCTAssertEqual(merged?.twoStage, true)
    XCTAssertEqual(merged?.refineScale, 1.35, "other fields on the existing tuning survive the merge")
  }

  /// The nested field is the more specific one — it wins on conflict.
  func testMergingTuningTwoStageWinsOverConflictingTwoPass() {
    var base = LTX2VideoTuning(); base.twoStage = false
    let merged = LTX2VideoTuning.merging(base, twoPass: true)
    XCTAssertEqual(merged?.twoStage, false, "explicit tuning.two_stage must not be overridden by two_pass")
  }

  /// End-to-end through the resolver: `two_pass` alone (no `tuning.two_stage`)
  /// must reach `resolvedConfig.twoStage`, with `request` provenance — the
  /// same as if the caller had sent `tuning.two_stage` directly. This is the
  /// per-request control comfybox#307 asks for: no env var, no restart.
  func testTwoPassAloneResolvesTwoStageAtRequestPrecedence() {
    let merged = LTX2VideoTuning.merging(nil, twoPass: true)
    let resolved = LTX2ConfigResolver.resolveTyped(
      request: merged, preset: nil,
      environment: ["LTX2_TWO_STAGE": "0"], configFile: [:])
    XCTAssertEqual(resolved.twoStage, true, "two_pass must win over the env-global default")
    XCTAssertEqual(resolved.provenance["two_stage"], .request)
  }
}
