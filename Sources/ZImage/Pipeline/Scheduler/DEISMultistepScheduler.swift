// DEISMultistepScheduler.swift — `deis_2m` / `deis_3m` / `deis_4m`
//
// WP-E14 (docs/FDD-krea2-raw-recipe.md §3.12, AC-23/24/25/26, Addendum A.1).
// Port sources, all at the pinned RES4LYF commit
// `26036f647ca15d3048a193daf99a40cecfc3820d`:
//   * `beta/deis_coefficients.py:86-121` — the `rhoab` closed form
//     (``DEISCoefficients``).
//   * `beta/rk_coefficients_beta.py:1343,1374-1393` — the ORDER RAMP and its
//     ralston warm-up, and `:1479-1517` — the multistep tableau assembly.
//   * `beta/rk_sampler_beta.py:813,920-926,960,1990,2126-2128` — one model call
//     per multistep step, the recycled `data_prev_` history and its roll.
//
// THREE THINGS THIS TYPE IS, IN THE ORDER THEY BITE.
//
// 1. IT IS A RALSTON FOR THE FIRST `order + 1` STEPS. RES4LYF swaps
//    `ralston_{order}s` in while `step < order + multistep_extra_initial_steps`,
//    and that option DEFAULTS TO 1 (`:1343`) — so `deis_3m` warms up for FOUR
//    steps, not three (Addendum A.1 corrected AC-24 on exactly this). At the
//    published stage-2 settings (`deis_3m`, 2 steps) the whole stage is
//    warm-up: 6 model evaluations and the DEIS coefficients never engage. That
//    is not a degenerate case to be papered over — it is the recipe, and it is
//    what E18's `deis3m_bong2_*` traces recorded (`rk_type == "ralston_3s"`).
//    ``warmUpSampler`` / ``warmUpSteps`` exist so nobody rediscovers it.
//
// 2. ITS ROW COUNT CHANGES MID-RUN. `order` during the warm-up, then 1: with
//    `multistep_stages = order − 1`, upstream's tableau loop runs
//    `rows − multistep_stages − row_offset + 1 == 1` row (`:960`). WP-E3's
//    driver re-reads `rows` every step and writes the mutated conformer back,
//    so this is supported; `Stats.modelEvals` is the counted truth and
//    `Stats.rowsAtStart` is only a label. `stepsRun × rowsAtStart` OVERCOUNTS
//    here, by design.
//
// 3. THE MULTISTEP STEP IS ANCHORED, LIKE EVERY OTHER RES4LYF ROW. The history
//    entries are DATA predictions from earlier steps, and upstream re-anchors
//    each of them at the CURRENT step's `x₀` and `σ` before use
//    (`rk_sampler_beta.py:925`, `get_epsilon_anchored`):
//
//        εⱼ = (x₀ⁱ − D^{i−j}) / σᵢ        x' = x₀ + h · Σⱼ (coeffⱼ/h) · εⱼ
//
//    NOT the derivative at that node's own sample and sigma, which is what a
//    textbook Adams–Bashforth would use.
//
// AND THEREFORE ORDER 2 — but the CAUSE is the frame, not the recycling.
// Every `deis_Nm` measures as **order 2**, and so does every `ralston_Ns`,
// which has no history at all. The common cause is the one
// `RES4LYFTableau.swift` already names for the ralstons: the anchored linear
// frame FREEZES the `1/σ` kernel at the step's own sigma. Whatever the rows or
// the interpolation produce, the step collapses to
//
//     x' = x₀ + (h/σ)·(x₀ − D_eff)
//
// for some effective data prediction `D_eff`. That form integrates a CONSTANT
// `D` exactly — it is the closed-form `x(σ) = D + (σ/σ₀)(x₀ − D)` — but the
// exact solution weights `D(s)` by `σ'/s²` across the step while the scheme
// weights it by `1/σ`; the difference integrates to `−D′h³/(6σ²)`. That is a
// local `O(h³)` error and therefore global order 2, and it is independent of
// the interpolation degree and of whether any history was recycled at all.
// Which is exactly why `ralston_3s`/`ralston_4s` cap at 2 with no history, and
// why `deis_2m` — whose nominal order is 2 — loses nothing.
//
// That is upstream's behaviour and therefore the recipe's; it is asserted
// two-sided in `DEISMultistepSchedulerTests` so the cap can never be mistaken
// for a transcription bug, and never "fixed".
//
// THE HISTORY, EXACTLY. Upstream keeps `data_prev_` (4 slots) and rolls it at
// the end of EVERY step — warm-up steps included:
//
//     data_prev_[0] = data_[0]
//     for ms in range(recycled_stages):            # recycled_stages == 3
//         data_prev_[recycled_stages - ms] = data_prev_[recycled_stages - ms - 1]
//
// The last iteration writes slot 1 from the slot 0 it just overwrote, so after
// step `n` the array is `[Dₙ, Dₙ, Dₙ₋₁, Dₙ₋₂]`, and the epsilons the next step
// builds from slots 1…3 are `Dₙ, Dₙ₋₁, Dₙ₋₂`. In other words the weight
// `coeff_prev_k` multiplies the data prediction from step `i − k`, which is
// what the interpolation nodes `σᵢ₋ₖ` mean. This type stores exactly those
// `order − 1` entries, most recent first — the duplicated slot 0 is a
// consequence of upstream's roll order and carries no information.
//
// WHY THE STEP COUNTER IS RUN-RELATIVE. Upstream's `step` indexes the sigma
// array the NODE handed the sampler, and for a `denoise < 1` stage that array
// is already sliced (`sigmas[-(steps+1):]`). Our grid is the full schedule and
// the loop enters it at `startIndex`, so the ramp must count from the run's
// own first step, not from the absolute grid index — otherwise a 2-step stage
// starting at grid index 8 would skip its warm-up entirely. The coefficients
// themselves are unaffected: they read only `σᵢ₊₁` and `σᵢ₋ₖ`, and every index
// the multistep branch reads is at or after `startIndex`.

import Foundation
import MLX

/// A conformer whose sampler CHANGES during a run, and which therefore owes
/// the provenance record an account of what it actually ran.
///
/// Only the DEIS order ramp does this today. ``Krea2RunTrace`` reads it through
/// a conditional cast, so a scheduler that does not ramp reports no warm-up
/// rather than an invented one.
public protocol WarmUpReportingScheduler {
  /// The sampler that ran during the warm-up (`"ralston_3s"`), or `nil` when
  /// no warm-up step was taken in this run.
  var warmUpSampler: String? { get }
  /// How many steps of this run used it.
  var warmUpSteps: Int { get }
}

/// RES4LYF's DEIS multistep samplers, with their ralston warm-up.
///
/// **Takes the data prediction, not the velocity** — both halves of the run do
/// (see the file comment). Callers convert through
/// ``ZImageScheduler/modelInput(velocity:sample:sigma:)``; `Krea2DenoiseLoop`
/// does it per row.
public struct DEISMultistepScheduler: TableauScheduler, WarmUpReportingScheduler {

  /// `int(rk_type[-2])` upstream — the multistep order, which also selects the
  /// warm-up's ralston.
  public enum Order: Int, Sendable, CaseIterable {
    case two = 2
    case three = 3
    case four = 4

    /// The RES4LYF sampler name (`multistep/deis_3m` in its UI).
    public var name: String { "deis_\(rawValue)m" }

    /// `rk_coefficients_beta.py:1380,1385,1389`: order 4 → `ralston_4s`,
    /// 3 → `ralston_3s`, 2 → `ralston_2s`. (Upstream also assigns `order = 3`
    /// in the order-4 arm; that local is dead — the warm-up LENGTH was already
    /// decided by the `step < order + extra` test above it, and `order` is read
    /// again only in the multistep arm, which this one does not reach.)
    public var warmUpStages: RalstonScheduler.Stages {
      switch self {
      case .two: return .two
      case .three: return .three
      case .four: return .four
      }
    }

    public var warmUpSampler: String { warmUpStages.name }

    /// `order + multistep_extra_initial_steps` — 3 / 4 / 5.
    public var warmUpStepCount: Int {
      rawValue + DEISMultistepScheduler.multistepExtraInitialSteps
    }

    /// `multistep_stages` once the ramp completes: `order − 1`, i.e. how many
    /// earlier data predictions the step interpolates through.
    public var historyDepth: Int { rawValue - 1 }
  }

  /// `EO("multistep_extra_initial_steps", 1)` — `rk_coefficients_beta.py:1343`.
  /// Not on the wire: it is upstream's default and the recipe never overrides
  /// it, and AC-24 is pinned to the value it produces.
  public static let multistepExtraInitialSteps = 1

  /// Anchored in both halves of the run.
  public let modelOutputConvention: ModelOutputConvention = .dataPrediction

  public let sigmas: MLXArray
  public let timesteps: MLXArray
  public let numInferenceSteps: Int
  public let order: Order

  /// RES4LYF's model-free `σ_min → 0` conversion sigma when this scheduler was
  /// built from a ``RES4LYFSigmaPreparation``-prepared grid; `nil` otherwise.
  public let finalConversionSigma: Float?

  /// Linear frame in both halves: `h = σ' − σ`, `ε = (x₀ − denoised)/σ`.
  /// `deis` is absent from upstream's exponential prefix list
  /// (`rk_coefficients_beta.py:1346`), and so is `ralston_*`.
  public var frame: TableauFrame { .linear }

  // MARK: - Run state (reset by `reset()`, which the driver calls first)

  /// Steps committed in THIS run — upstream's `step`. See the file comment on
  /// why this is run-relative rather than the absolute grid index.
  private var stepsCommitted = 0

  /// The grid index of the run's first committed step, so a non-contiguous
  /// driver fails loudly instead of silently re-ramping.
  private var originIndex: Int?

  /// `data_prev_` without upstream's duplicated slot 0: the last
  /// ``Order/historyDepth`` data predictions, most recent first.
  private var previousData: [MLXArray] = []

  /// How many steps of this run ran the ralston warm-up.
  public private(set) var warmUpSteps = 0

  /// How many ran the DEIS coefficients.
  public private(set) var multistepSteps = 0

  /// `nil` until a warm-up step has actually been taken, so a run that took
  /// none reports none.
  public var warmUpSampler: String? { warmUpSteps > 0 ? order.warmUpSampler : nil }

  /// Whether the NEXT step is still warm-up (`step < order + extra`).
  public var isWarmingUp: Bool { stepsCommitted < order.warmUpStepCount }

  /// `order` while warming up, then 1 — upstream's
  /// `rows − multistep_stages − row_offset + 1`.
  ///
  /// The warm-up half reads its count off the ralston it actually runs, not off
  /// `order.rawValue`: the two are equal by construction (pinned by
  /// `DEISMultistepSchedulerTests.testWarmUpLengthAndSamplerPerVariant`), and
  /// tying them here means a mis-mapped warm-up can never advertise more rows
  /// than the tableau it is about to index has.
  public var rows: Int { isWarmingUp ? order.warmUpStages.rawValue : 1 }

  private let sigmaValues: [Float]
  /// The grid in `Double`, once: the coefficients are computed there and the
  /// conversion is otherwise repeated on every multistep step.
  private let sigmaValuesDouble: [Double]

  /// - Parameters:
  ///   - order: 2, 3 or 4 — `deis_2m` / `deis_3m` / `deis_4m`.
  ///   - numInferenceSteps: number of denoising steps.
  ///   - sigmaValues: `numInferenceSteps + 1` sigmas, monotonically decreasing.
  ///   - numTrainTimesteps: training timestep count for deriving `timesteps`.
  public init(
    order: Order,
    numInferenceSteps: Int,
    sigmaValues: [Float],
    numTrainTimesteps: Int = 1000,
    finalConversionSigma: Float? = nil
  ) {
    precondition(numInferenceSteps > 0, "numInferenceSteps must be positive")
    precondition(
      sigmaValues.count == numInferenceSteps + 1,
      "sigmaValues must have numInferenceSteps + 1 elements")

    self.order = order
    self.sigmaValues = sigmaValues
    self.sigmaValuesDouble = sigmaValues.map(Double.init)
    self.numInferenceSteps = numInferenceSteps
    self.finalConversionSigma = finalConversionSigma
    self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
    let numTrainF = Float(numTrainTimesteps)
    let timestepValues = sigmaValues.dropLast().map { $0 * numTrainF }
    self.timesteps = MLXArray(timestepValues, [timestepValues.count])
  }

  // MARK: - N-row protocol

  public func rowSigma(timestepIndex: Int, row: Int) -> Float {
    precondition(row >= 0 && row < rows, "row \(row) outside 0..<\(rows)")
    let i = checked(timestepIndex)
    guard isWarmingUp else {
      // The multistep tableau's `ci` is all zeros (`rk_coefficients_beta.py:1500`),
      // so its one substep sigma IS the step sigma.
      return sigmaValues[i]
    }
    let table = RalstonScheduler.tableau(for: order.warmUpStages)
    return RES4LYFTableau.rowSigma(
      frame: frame, sigma: sigmaValues[i], h: stepSize(timestepIndex: i), c: Float(table.c[row]))
  }

  public mutating func rowSample(
    timestepIndex: Int, row: Int, x0: MLXArray, k: [MLXArray]
  ) -> MLXArray {
    precondition(row >= 1 && row < rows, "row 0 is the step's start sample")
    precondition(k.count == row, "row \(row) needs exactly \(row) prior outputs, got \(k.count)")
    // Unreachable once multistep engages (`rows == 1`), and a caller that got
    // here anyway would be building a row this sampler does not have.
    precondition(isWarmingUp, "\(order.name) takes one evaluation per multistep step")
    let i = checked(timestepIndex)
    let table = RalstonScheduler.tableau(for: order.warmUpStages)
    return RES4LYFTableau.advance(
      frame: frame, x0: x0, dataPredictions: k, weights: table.a[row],
      h: stepSize(timestepIndex: i), sigma: sigmaValues[i])
  }

  public mutating func commit(timestepIndex: Int, x0: MLXArray, k: [MLXArray]) -> MLXArray {
    precondition(k.count == rows, "commit needs \(rows) outputs, got \(k.count)")
    let i = checked(timestepIndex)

    // The ramp is keyed on the run's own step count, so the run must be
    // contiguous for it to mean anything. A gap is a caller bug, not a
    // schedule: fail here rather than warm up twice or not at all.
    if let originIndex {
      precondition(
        timestepIndex == originIndex + stepsCommitted,
        "\(order.name): the order ramp needs contiguous steps — expected grid index "
          + "\(originIndex + stepsCommitted), got \(timestepIndex). Reset between runs.")
    } else {
      originIndex = timestepIndex
    }

    let h = stepSize(timestepIndex: i)
    let sigma = sigmaValues[i]
    let next: MLXArray

    if isWarmingUp {
      // `ralston_{order}s`, verbatim — the same tableau `RalstonScheduler` runs.
      let table = RalstonScheduler.tableau(for: order.warmUpStages)
      next = RES4LYFTableau.advance(
        frame: frame, x0: x0, dataPredictions: k, weights: table.b, h: h, sigma: sigma)
      warmUpSteps += 1
    } else {
      // `rk_coefficients_beta.py:1480-1517`: b = coeff/h, then
      // `b *= (σ_down − σ)/(σ_next − σ)`.
      //
      // That factor is **not** the eta seam, and reading it as one would be a
      // real bug: its `sigma_down` is `NS.sigma_down`, which
      // `rk_sampler_beta.py:783-784` feeds into `set_coeff` from the OVERSHOOT
      // split — `rk_noise_sampler_beta.py:318-319` derives `sigma_down` from
      // `overshoot` and `sigma_down_ETA` (a different field, which `set_coeff`
      // never sees) from `eta`.
      //
      // The Krea 2 path sets `overshoot = 0`, so `σ_down == σ_next` and the
      // factor is exactly 1 at EVERY eta — including 0.5, where the T2 traces
      // reproduce with it absent. A future implementer must therefore NOT
      // substitute `sigma_down_eta` here; overshoot is what would make this
      // term live, and it is unimplemented by choice
      // (`RES4LYFSDENoise.swift`, "What is NOT modelled").
      precondition(
        previousData.count >= order.historyDepth,
        "\(order.name): multistep needs \(order.historyDepth) recycled data predictions, "
          + "have \(previousData.count) — the warm-up did not run")
      let coefficients = DEISCoefficients.rhoab(
        sigmas: sigmaValuesDouble, index: i, maxOrder: order.rawValue)
      precondition(
        coefficients.count == order.rawValue,
        "\(order.name): the ramp yielded \(coefficients.count) coefficients at step \(i)")
      let hDouble = sigmaValuesDouble[i + 1] - sigmaValuesDouble[i]
      // `b = coeff/h` has no meaning at h = 0, and `inf` weights would produce
      // a NaN latent several steps later rather than here. RES4LYF's own
      // `prepare_sigmas` de-duplicates for exactly this reason
      // (``RES4LYFSigmaPreparation/dedupeConsecutive(_:)``); a caller that
      // bypassed it and handed us a repeated sigma finds out now.
      precondition(
        hDouble != 0,
        "\(order.name): a repeated sigma at step \(i) (σ = σ' = \(sigmaValues[i])) gives the "
          + "multistep no step to integrate over")
      let weights = coefficients.map { $0 / hDouble }
      // `[current] + data_prev_[1 ..< order]`, which is `Dᵢ₋₁, Dᵢ₋₂, …`.
      let nodes = [k[0]] + previousData.prefix(order.historyDepth)
      next = RES4LYFTableau.advance(
        frame: frame, x0: x0, dataPredictions: nodes, weights: weights, h: h, sigma: sigma)
      multistepSteps += 1
    }

    // `data_prev_[0] = data_[0]` and the roll, in the form the roll actually
    // has after its aliasing: push row 0's data prediction, drop the oldest.
    // Run on EVERY step, warm-up included — which is what makes the history
    // full by the time the ramp completes.
    if order.historyDepth > 0 {
      previousData.insert(k[0], at: 0)
      if previousData.count > order.historyDepth { previousData.removeLast() }
    }
    stepsCommitted += 1
    return next
  }

  // MARK: - There is no 1-row fallback

  /// See ``RalstonScheduler/step(modelOutput:timestepIndex:sample:)``: driving
  /// an N-row conformer through the 1-row protocol would render first-order
  /// Euler under this sampler's name, and the multistep half would silently
  /// lose its recycled history as well. The reachable paths refuse the name
  /// earlier — `GeneratePayload.validateTableauSampler` (400 off Krea 2) and
  /// the CLI's `--scheduler`, both keyed on ``SchedulerKind/isNRowTableau``.
  public mutating func step(
    modelOutput: MLXArray, timestepIndex: Int, sample: MLXArray
  ) -> MLXArray {
    preconditionFailure(
      "\(order.name) is an N-row conformer driven by a 1-row loop: `step` would silently "
        + "render first-order Euler under the name \(order.name), with no ralston warm-up and "
        + "no multistep history. Drive it through Krea2DenoiseLoop (rowSigma / rowSample / "
        + "commit), or refuse the sampler on this path.")
  }

  /// AC-13: the ramp, the counters and the recycled history are all run state,
  /// and none of them may leak into the next render.
  public mutating func reset() {
    stepsCommitted = 0
    originIndex = nil
    previousData = []
    warmUpSteps = 0
    multistepSteps = 0
  }

  // MARK: - Helpers

  private func stepSize(timestepIndex: Int) -> Float {
    let i = checked(timestepIndex)
    return RES4LYFTableau.stepSize(
      frame: frame, sigma: sigmaValues[i], sigmaNext: sigmaValues[i + 1])
  }

  private func checked(_ timestepIndex: Int) -> Int {
    precondition(
      timestepIndex >= 0 && timestepIndex + 1 < sigmaValues.count,
      "invalid timestep index \(timestepIndex)")
    return timestepIndex
  }
}
