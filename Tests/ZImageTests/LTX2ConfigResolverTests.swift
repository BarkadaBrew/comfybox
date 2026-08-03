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
