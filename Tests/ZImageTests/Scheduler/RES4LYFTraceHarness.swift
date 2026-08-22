import Foundation
import MLX
import XCTest

@testable import ZImage

// WP-E18 — RES4LYF / ComfyUI oracle fixtures and the step-trace harness
// (FDD-krea2-raw-recipe §5.2, AC-26 consumer).
//
// The fixtures under Tests/ZImageTests/Fixtures/Scheduler are produced by
// scripts/oracles/gen_scheduler_fixtures.py from the PINNED upstream sources
// recorded in scripts/oracles/upstream/PROVENANCE.md. Python is a one-off
// fixture generator there, never a runtime dependency. This file is the Swift
// side: a loader for the step traces, the scripted denoiser both stacks
// evaluate, the RES4LYF hard-mode VP split in closed form, and a relative
// tolerance assertion. WP-E11/E12/E13/E14/E15/E16 consume it.

/// Where every scheduler oracle fixture lives.
enum SchedulerOracleFixtures {
  static let dir = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()  // Scheduler/
    .deletingLastPathComponent()  // ZImageTests/
    .appendingPathComponent("Fixtures/Scheduler")

  static func url(_ name: String) -> URL { dir.appendingPathComponent(name) }

  static func json(_ name: String) throws -> [String: Any] {
    let data = try Data(contentsOf: url(name))
    guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw NSError(domain: "SchedulerOracleFixtures", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "\(name) is not a JSON object"])
    }
    return obj
  }

  static func doubles(_ any: Any?, _ what: String) throws -> [Double] {
    guard let arr = any as? [Any] else {
      throw NSError(domain: "SchedulerOracleFixtures", code: 2,
        userInfo: [NSLocalizedDescriptionKey: "\(what): expected an array"])
    }
    return try arr.map { v -> Double in
      if let d = v as? Double { return d }
      if let i = v as? Int { return Double(i) }
      throw NSError(domain: "SchedulerOracleFixtures", code: 3,
        userInfo: [NSLocalizedDescriptionKey: "\(what): non-numeric element \(v)"])
    }
  }
}

// MARK: - Step trace manifest

/// One RES4LYF `sample_rk_beta` run against the scripted denoiser, exported
/// per step. Keys into the companion `.safetensors` are strings; every tensor
/// is float32 (RES4LYF's `work_dtype`), every scalar is the sampler's float64.
struct RES4LYFTrace: Decodable {
  struct Recipe: Decodable {
    let tier: String
    let sampler: String
    let scheduler: String
    let steps: Int
    let denoise: Double
    let eta: Double
    let etaSubstep: Double
    let bongmath: Bool
    let noiseModeSde: String
    let sNoise: Double
    let noiseAnchor: Double
    let shift: Double
    let modelSampling: String
    let seed: Int
    let noiseSeedSde: Int
  }

  /// One model evaluation: `denoised = model(x_in, s_tmp)`, and RES4LYF's
  /// anchored epsilon for it (`denoised − x_0` exponential, `(x_0 − denoised)/σ`
  /// linear — `noise_anchor = 1.0`).
  struct ModelCall: Decodable {
    let row: Int
    let sTmp: Double
    let sigma: Double
    let xIn: String
    let x0: String
    let denoised: String
    let eps: String
  }

  /// The substep split after row `row`'s update. `noise`/`xPre`/`xPost` are
  /// present only when RES4LYF actually re-noised the row (eta_substep > 0 and
  /// the substep target is > 0).
  struct Substep: Decodable {
    let row: Int
    let subSigma: Double
    let subSigmaNext: Double
    let subSigmaUpEta: Double
    let subSigmaDownEta: Double
    let subAlphaRatioEta: Double
    let hNew: Double
    /// x₀ as `swap_noise_substep` saw it (T3's bongmath re-derives x₀ later in
    /// the same step, so the step-level `x0` is not the one this re-noise used).
    let x0: String?
    let xPre: String?
    let noise: String?
    let xPost: String?
  }

  /// State after one `bong_iter` call (T3 only): the re-derived x₀, every row
  /// sample and every row epsilon.
  struct Bong: Decodable {
    let row: Int
    let x0: String
    let xRows: [String]
    let epsRows: [String]
  }

  struct Step: Decodable {
    let index: Int
    let sigma: Double
    let sigmaNext: Double
    /// Sampler name the node requested (`res_2s`, `deis_3m`).
    let rkTypeRequested: String
    /// Sampler RES4LYF actually ran this step (e.g. `ralston_3s` during the
    /// `deis_3m` warm-up, `deis` once the multistep history exists).
    let rkType: String
    let exponential: Bool
    let rows: Int
    let rowOffset: Int
    let multistepStages: Int
    let aMatrix: [[Double]]
    let bWeights: [[Double]]
    let cNodes: [Double]
    /// `NS.h`: `−log(σ_down/σ)` exponential, `σ_down − σ` linear.
    let h: Double
    let substepSigmas: [Double]
    let sigmaUpEta: Double
    let sigmaDownEta: Double
    let alphaRatioEta: Double
    let modelCalls: [ModelCall]
    let substeps: [Substep]
    let bongmath: [Bong]
    /// x₀ as it stood when the step's update was taken (post-bongmath in T3).
    let x0: String
    /// The step's integrated result before any SDE re-noise.
    let xNext: String
    /// Injected (z-scored) noise when eta > 0 and σ_next > 0; nil otherwise.
    let noiseStep: String?
    /// What the next step starts from.
    let xOut: String
  }

  struct Final: Decodable {
    let x: String
    let epsLast: String
    let linearTailFromSigmaMin: Bool
  }

  let recipe: Recipe
  let denoiser: String
  let latentShape: [Int]
  let sigmasSchedule: [Double]
  let sigmasRun: [Double]
  let sigmaMin: Double
  let sigmaMax: Double
  let modelCallsTotal: Int
  let latentImage: String
  let noiseInit: String
  let xInit: String
  let steps: [Step]
  let final: Final
}

/// A loaded trace: manifest plus tensors.
final class RES4LYFTraceFixture {
  static let names = [
    "res2s_beta6_T1", "res2s_beta6_T2", "res2s_beta6_T3",
    "deis3m_bong2_T1", "deis3m_bong2_T2", "deis3m_bong2_T3",
  ]

  let name: String
  let manifest: RES4LYFTrace
  let tensors: [String: MLXArray]

  static func load(_ name: String) throws -> RES4LYFTraceFixture {
    let jsonURL = SchedulerOracleFixtures.url("res4lyf_trace_\(name).json")
    let stURL = SchedulerOracleFixtures.url("res4lyf_trace_\(name).safetensors")
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let manifest = try decoder.decode(RES4LYFTrace.self, from: Data(contentsOf: jsonURL))
    let tensors = try MLX.loadArrays(url: stURL)
    return RES4LYFTraceFixture(name: name, manifest: manifest, tensors: tensors)
  }

  private init(name: String, manifest: RES4LYFTrace, tensors: [String: MLXArray]) {
    self.name = name
    self.manifest = manifest
    self.tensors = tensors
  }

  func tensor(_ key: String, file: StaticString = #filePath, line: UInt = #line) throws -> MLXArray {
    guard let t = tensors[key] else {
      XCTFail("\(name): tensor '\(key)' missing from safetensors", file: file, line: line)
      throw NSError(domain: "RES4LYFTraceFixture", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "missing tensor \(key)"])
    }
    return t
  }
}

// MARK: - The scripted denoiser both stacks evaluate

/// `denoised(x, σ) = 0.5·tanh(x) + 0.25·σ − 0.1·x` (FDD §5.2). Weight-free,
/// σ-dependent, non-linear — enough to make every tableau row and every
/// coefficient observable. The Python side evaluates the identical expression.
enum RES4LYFScriptedDenoiser {
  static func denoised(_ x: MLXArray, sigma: Float) -> MLXArray {
    0.5 * tanh(x) + 0.25 * sigma - 0.1 * x
  }

  /// The flow velocity `v = (x − x₀)/σ` the Swift schedulers' `modelInput`
  /// helper expects to be handed.
  static func velocity(_ x: MLXArray, sigma: Float) -> MLXArray {
    (x - denoised(x, sigma: sigma)) / sigma
  }
}

// MARK: - RES4LYF hard-mode variance-preserving split (closed form)

/// `rk_noise_sampler_beta.py` `get_sde_step` (`noise_mode = "hard"`) feeding
/// `get_sde_coeff` on a `CONST` model: `σ_up = η·σ'`,
/// `σ_res = sqrt(σ'² − σ_up²)`, `α = (σ_max − σ') + σ_res`, `σ_down = σ_res/α`.
/// Pure Double so the fixture's recorded split can be checked independently.
enum RES4LYFHardVPSplit {
  static func split(sigmaNext: Double, eta: Double, sigmaMax: Double = 1.0)
    -> (up: Double, down: Double, alpha: Double)
  {
    if eta == 0 || sigmaNext == 0 {
      return (0, sigmaNext, 1)
    }
    var up = sigmaNext * eta
    if up >= sigmaNext {  // RES4LYF's "maximum VPSDE noise level exceeded" fallback
      up = eta >= 1 ? sigmaNext * 0.9999 : sigmaNext * eta
    }
    let residual = (sigmaNext * sigmaNext - up * up).squareRoot()
    let alpha = (sigmaMax - sigmaNext) + residual
    return (up, residual / alpha, alpha)
  }
}

// MARK: - Assertions

/// Max-abs relative comparison: `max|got − want| ≤ rtol · max(max|want|, floor)`.
func XCTAssertTraceClose(
  _ got: MLXArray, _ want: MLXArray, rtol: Float = 1e-4, floor: Float = 1e-3,
  _ message: @autoclosure () -> String = "", file: StaticString = #filePath, line: UInt = #line
) {
  guard got.shape == want.shape else {
    XCTFail("shape mismatch \(got.shape) vs \(want.shape): \(message())", file: file, line: line)
    return
  }
  let g = got.asType(.float32)
  let w = want.asType(.float32)
  let diff = MLX.abs(g - w).max().item(Float.self)
  let scale = max(MLX.abs(w).max().item(Float.self), floor)
  let rel = diff / scale
  XCTAssertLessThanOrEqual(
    rel, rtol, "max-abs rel diff \(rel) (abs \(diff), scale \(scale)): \(message())",
    file: file, line: line)
}
