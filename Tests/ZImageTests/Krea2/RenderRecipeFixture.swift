import Foundation
@testable import ZImage

/// Shared read-back fixtures for the WP-E10 sinks (FDD §3.10). Every sink
/// test needs the same thing: a `Krea2RunTrace` shaped like one the loop
/// really hands back, and the pipeline-side facts around it. Built once here
/// so the response, PNG, `/health` and async-status tests all assert against
/// the SAME record rather than four hand-rolled near-copies (AC-62 is exactly
/// the claim that they agree).
enum RenderRecipeFixture {

  /// A trace as `Krea2DenoiseLoop` would have produced it: the grid from the
  /// real `SigmaSchedule`, `modelEvals` counted at 1 per step (2 under CFG),
  /// and `stepsRun` measured from `startIndex` — never echoed.
  static func trace(
    steps: Int = 30,
    guidance: Float = 1.0,
    startIndex: Int = 0,
    sampler: SchedulerKind = .euler,
    sigmaSchedule: SigmaScheduleKind = .krea2,
    sigmaScheduleRequested: String? = nil,
    /// The negative the REQUEST carried; the fixture resolves what a pipeline
    /// would have encoded from it exactly as the pipelines do (I4).
    negativePrompt: String? = nil,
    eta: Float = 0,
    bongmath: Bool = false,
    mu: Float = 0.9062,
    seed: UInt64 = 44821,
    // WP-E14: the DEIS order ramp, as the loop reported it. Defaults are the
    // non-ramping case, which is every sampler but `deis_Nm`.
    warmupSampler: String? = nil,
    warmupSteps: Int = 0
  ) -> Krea2RunTrace {
    let sigmas = SigmaSchedule.krea2(numSteps: steps, mu: mu)
    let effective = sigmas.count - 1
    let run = max(0, effective - startIndex)
    return Krea2RunTrace(
      sampler: sampler,
      sigmaSchedule: sigmaSchedule,
      sigmaScheduleRequested: sigmaScheduleRequested,
      mu: mu,
      shift: exp(mu),
      shiftSource: "dynamic",
      sigmas: sigmas,
      warmupSampler: warmupSampler,
      warmupSteps: warmupSteps,
      stepsRequested: steps,
      stepsEffective: effective,
      stepsRun: run,
      modelEvals: run * (guidance > 1 ? 2 : 1),
      startIndex: startIndex,
      denoise: startIndex == 0 ? 1.0 : Float(run) / Float(effective),
      guidance: guidance,
      eta: eta,
      bongmath: bongmath,
      seed: seed,
      width: 1024,
      height: 1024,
      negativePromptApplied: Krea2RunTrace.negativePromptApplied(
        cfgActive: guidance > 1.0, requested: negativePrompt))
  }

  /// The pipeline-side read-backs around a trace: what loaded, not what was
  /// asked for.
  static func inputs(
    trace: Krea2RunTrace,
    quantizationBits: Int? = 8,
    loras: [RenderRecipe.LoRAReadBack] = [],
    control: RenderRecipe.ControlReadBack? = nil,
    vaeLayout: VAELayout = .wanNative,
    vaeSource: Krea2VAESelection.Source = .payload
  ) -> RenderRecipe.Krea2Inputs {
    inputs(
      traces: [trace], quantizationBits: quantizationBits, loras: loras, control: control,
      vaeLayout: vaeLayout, vaeSource: vaeSource)
  }

  /// WP-E17: the same read-backs around N stage traces.
  static func inputs(
    traces: [Krea2RunTrace],
    quantizationBits: Int? = 8,
    loras: [RenderRecipe.LoRAReadBack] = [],
    control: RenderRecipe.ControlReadBack? = nil,
    vaeLayout: VAELayout = .wanNative,
    vaeSource: Krea2VAESelection.Source = .payload
  ) -> RenderRecipe.Krea2Inputs {
    RenderRecipe.Krea2Inputs(
      baseModel: "krea2-raw",
      variant: .raw,
      transformerFile: URL(fileURLWithPath: "/Users/me/LocalModels/krea2-raw/raw.safetensors"),
      quantizationBits: quantizationBits,
      vae: Krea2VAESelection(
        file: URL(fileURLWithPath: "/Users/me/LocalModels/vae/Wan2_1_VAE_fp32.safetensors"),
        layout: vaeLayout, source: vaeSource),
      textEncoderFile: URL(fileURLWithPath: "/Users/me/LocalModels/krea2-raw/text_encoder/model.safetensors"),
      loras: loras,
      control: control,
      traces: traces)
  }

  static func recipe(
    steps: Int = 30, guidance: Float = 1.0, startIndex: Int = 0,
    sigmaScheduleRequested: String? = nil, negativePrompt: String? = nil,
    quantizationBits: Int? = 8,
    loras: [RenderRecipe.LoRAReadBack] = [], control: RenderRecipe.ControlReadBack? = nil
  ) -> RenderRecipe {
    RenderRecipe.krea2(inputs(
      trace: trace(steps: steps, guidance: guidance, startIndex: startIndex,
                   sigmaScheduleRequested: sigmaScheduleRequested,
                   negativePrompt: negativePrompt),
      quantizationBits: quantizationBits, loras: loras, control: control))
  }

  static func report(offered: Int, bound: Int, deltasApplied: Int = 0, shapeRejected: Int = 0)
    -> LoRAApplicationReport
  {
    LoRAApplicationReport(
      offered: offered, bound: bound, quantizedBound: bound, deltasApplied: deltasApplied,
      shapeRejected: shapeRejected, unbound: [])
  }
}
