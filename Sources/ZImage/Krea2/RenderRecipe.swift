// RenderRecipe.swift — The provenance record (WP-E10, FDD §3.10, D4, D8, D12,
// D15, D22; PRD §4).
//
// Swift type `RenderRecipe`, wire field `applied` (D8): this is what APPLIED,
// not what was asked for. Every field is READ BACK — from the pipeline's
// loaded configs joined with their bind reports, the resident VAE selection,
// the physical variant and transformer file, and the run trace the loop
// counted — never echoed from the request.
//
// WIRE SHAPE: snake_case, produced by the `.convertToSnakeCase` encoder every
// sink uses (the /v1/generate + async-status responses, /health, and the PNG
// `UserComment` JSON — `ImageMetadata.generation` sets the same strategy).
// Properties are camelCase with SYNTHESIZED Codable so a `.convertFromSnakeCase`
// decoder — this codebase's convention (see `GeneratePayload`'s CodingKeys
// note) — reads it back unchanged. A nil optional is ABSENT on the wire
// (`negative_prompt` when guidance <= 1, AC-61; `control_lora` when none).
//
// D12: populated for the Krea 2 family only. Other families emit no
// `applied` block (the field is absent/null, never half-filled); the client's
// `provenance: 'request'` branch covers absence honestly. Generalising to
// flux1/flux2/fibo/chroma is a filed follow-up, not a silent asymmetry.

import Foundation

public struct RenderRecipe: Codable, Sendable, Equatable {

  // MARK: - Physics: what was loaded

  /// The pool spec / alias the pipeline was loaded as (e.g. `krea2-raw`,
  /// `kroma-v0.2-turbo`).
  public let baseModel: String
  /// `"raw"` | `"turbo"` — `Krea2Pipeline.variant`, a physical fact (D7).
  public let baseVariant: String
  /// `Krea2ModelPaths.transformerFile` — unambiguous.
  public let baseModelFile: String
  /// `"q8"` | `"q4"` | `"bf16"` — how the resident transformer was loaded.
  public let quantization: String
  /// The file that decoded (and encoded) — the resident `Krea2VAESelection`.
  public let vae: String
  /// `"qwenDiffusers"` | `"wanNative"` — sniffed from the file's keys.
  public let vaeLayout: String
  /// `"payload"` | `"preset"` | `"model_dir"` — who selected the VAE.
  public let vaeSource: String
  /// Architecture / dtype label of the text encoder.
  public let textEncoder: String
  public let textEncoderFile: String
  /// Read back from `loadedLoRAConfigs` joined with `loadedLoRAReports`.
  public let loras: [Applied]
  /// The depth Control-LoRA when one was active for this render.
  public let controlLora: Applied?

  // MARK: - Geometry & seed

  /// POST round-up (`Krea2Sampling.roundUp` to the 16-px alignment).
  public let width: Int
  public let height: Int
  public let seed: UInt64

  // MARK: - The schedule grid (D3)

  public let mu: Float
  public let shift: Float?
  /// `"dynamic"` | `"explicit"` (the pipeline's `ShiftSource`).
  public let shiftSource: String

  // MARK: - What actually ran, per stage (D4)

  /// One entry per stage that ran — one entry for a single-stage render.
  public let stages: [Stage]
  public let modelEvalsTotal: Int

  public struct Stage: Codable, Sendable, Equatable {
    /// 0-based.
    public let index: Int
    /// Resolved `SchedulerKind.rawValue` of what ran.
    public let sampler: String
    /// Resolved `SigmaScheduleKind.rawValue` of what ran.
    public let sigmaSchedule: String
    /// D22: the raw name the caller sent when it differed from what ran
    /// (`"normal"` → `"flow"`); nil when equal or absent.
    public let sigmaScheduleRequested: String?
    /// false for `bong_tangent` (D6).
    public let shiftApplied: Bool
    public let stepsRequested: Int
    /// Beta de-dup can shrink this (D5).
    public let stepsEffective: Int
    /// img2img / stage-2 start mid-schedule.
    public let stepsRun: Int
    public let modelEvals: Int
    public let denoise: Float
    public let guidance: Float
    /// nil when guidance <= 1 — it did not apply (AC-61).
    public let negativePrompt: String?
    public let eta: Float
    public let bongmath: Bool
    /// `"ralston_3s"` for `deis_3m` below order.
    public let warmupSampler: String?
    public let warmupSteps: Int
    /// First 3 sigmas of the grid.
    public let sigmaHead: [Float]
    /// Last 3 sigmas of the grid.
    public let sigmaTail: [Float]
    public let seed: UInt64

  }

  public struct Applied: Codable, Sendable, Equatable {
    public let file: String
    public let scaleApplied: Float
    /// `"raw"` | `"turbo"` | null — the declared relativity (WP-E6).
    public let relativeTo: String?
    public let pairsOffered: Int
    public let pairsBound: Int
    /// WP-E6: matched a module, failed normalize.
    public let shapeRejected: Int
    /// D15: kroma-on-Raw records 0 here.
    public let deltasApplied: Int
    /// `"kroma"` | `"accel"` | `"bypass"` | `"control"` | null — the
    /// configuration slot, labelled once by the engine.
    public let role: String?

  }

  // MARK: - Retention across a base handoff

  /// The record `/health.last_recipe` may still publish once `activeTransformerFile`
  /// is what is resident — `nil` when the base changed under it.
  ///
  /// `/health` reports `model`, `model_variant` and `last_recipe` side by
  /// side; a record that outlived its base would describe a checkpoint that
  /// is no longer loaded, which reads as provenance and is not. Pass `nil`
  /// for `activeTransformerFile` when the active family is not Krea 2.
  public static func retained(_ record: RenderRecipe?, activeTransformerFile: String?) -> RenderRecipe? {
    guard let record, let activeTransformerFile,
          record.baseModelFile == activeTransformerFile
    else { return nil }
    return record
  }

  // MARK: - Builder: Krea 2 read-backs → record

  /// One loaded adapter: the configuration the pipeline holds in
  /// `loadedLoRAConfigs` and the bind report beside it in `loadedLoRAReports`.
  public struct LoRAReadBack: Sendable {
    public let configuration: LoRAConfiguration
    public let report: LoRAApplicationReport
    public init(configuration: LoRAConfiguration, report: LoRAApplicationReport) {
      self.configuration = configuration
      self.report = report
    }
  }

  /// The depth Control-LoRA as the pipeline applied it.
  public struct ControlReadBack: Sendable {
    public let file: URL
    public let scale: Float
    public let report: LoRAApplicationReport
    public init(file: URL, scale: Float, report: LoRAApplicationReport) {
      self.file = file
      self.scale = scale
      self.report = report
    }
  }

  /// The Krea 2 text encoder as loaded: Qwen3-VL-4B's language tower, bf16
  /// (`Krea2WeightLoader.loadTextEncoder` skips the `visual.*` tower).
  public static let krea2TextEncoderLabel = "qwen3-vl-4b/bf16"

  /// Everything the record needs, each value taken from the PIPELINE (or the
  /// coordinator's pool state), never from the request — except
  /// `requestedSigmaSchedule` and `negativePrompt`, which are recorded as
  /// what the caller sent only in contrast to / gated by what ran (D22, AC-61).
  public struct Krea2Inputs: Sendable {
    public var baseModel: String
    public var variant: Krea2Variant
    public var transformerFile: URL
    /// The bits the transformer was quantized to at load
    /// (`Krea2Pipeline.transformerQuantBits`), nil for bf16. The label is
    /// derived here, so there is one spelling of `"q8"` in the codebase.
    public var quantizationBits: Int?
    public var vae: Krea2VAESelection
    public var textEncoderFile: URL
    public var loras: [LoRAReadBack]
    public var control: ControlReadBack?
    public var trace: Krea2RunTrace
    public var negativePrompt: String?

    public init(
      baseModel: String, variant: Krea2Variant, transformerFile: URL, quantizationBits: Int?,
      vae: Krea2VAESelection, textEncoderFile: URL,
      loras: [LoRAReadBack], control: ControlReadBack?,
      trace: Krea2RunTrace, negativePrompt: String?
    ) {
      self.baseModel = baseModel
      self.variant = variant
      self.transformerFile = transformerFile
      self.quantizationBits = quantizationBits
      self.vae = vae
      self.textEncoderFile = textEncoderFile
      self.loras = loras
      self.control = control
      self.trace = trace
      self.negativePrompt = negativePrompt
    }
  }

  /// `"q8"` / `"q4"` / `"bf16"` from the bits actually applied at load.
  public static func quantizationLabel(bits: Int?) -> String {
    bits.map { "q\($0)" } ?? "bf16"
  }

  public static func krea2(_ i: Krea2Inputs) -> RenderRecipe {
    let t = i.trace
    let stage = Stage(
      index: 0,
      sampler: t.sampler.rawValue,
      sigmaSchedule: t.sigmaSchedule.rawValue,
      // D22: the loop already decided whether the caller's raw name is worth
      // contrasting with what ran; the record does not second-guess it.
      sigmaScheduleRequested: t.sigmaScheduleRequested,
      shiftApplied: t.sigmaSchedule != .bongTangent,
      stepsRequested: t.stepsRequested,
      stepsEffective: t.stepsEffective,
      stepsRun: t.stepsRun,
      modelEvals: t.modelEvals,
      denoise: t.denoise,
      guidance: t.guidance,
      negativePrompt: t.cfgActive ? i.negativePrompt : nil,
      eta: t.eta,
      bongmath: t.bongmath,
      // T2/T3 warm-up (`deis_3m` below order → `ralston_3s`) lands with
      // WP-E15/E16; until then the loop reports none and the record says so
      // rather than inventing one.
      warmupSampler: nil,
      warmupSteps: 0,
      sigmaHead: t.sigmaHead,
      sigmaTail: t.sigmaTail,
      seed: t.seed)
    let loras = i.loras.map { rb -> Applied in
      Applied(
        file: rb.configuration.source.recordPath,
        scaleApplied: rb.configuration.scale,
        relativeTo: rb.configuration.requiresBase?.rawValue,
        pairsOffered: rb.report.offered,
        pairsBound: rb.report.bound,
        shapeRejected: rb.report.shapeRejected,
        deltasApplied: rb.report.deltasApplied,
        role: rb.configuration.role)
    }
    let control = i.control.map { c in
      Applied(
        file: c.file.path, scaleApplied: c.scale, relativeTo: nil,
        pairsOffered: c.report.offered, pairsBound: c.report.bound,
        shapeRejected: c.report.shapeRejected, deltasApplied: c.report.deltasApplied,
        role: "control")
    }
    return RenderRecipe(
      baseModel: i.baseModel,
      baseVariant: i.variant.rawValue,
      baseModelFile: i.transformerFile.path,
      quantization: quantizationLabel(bits: i.quantizationBits),
      vae: i.vae.file.path,
      vaeLayout: i.vae.layout.rawValue,
      vaeSource: i.vae.source.rawValue,
      textEncoder: krea2TextEncoderLabel,
      textEncoderFile: i.textEncoderFile.path,
      loras: loras,
      controlLora: control,
      width: t.width,
      height: t.height,
      seed: t.seed,
      mu: t.mu,
      shift: t.shift,
      shiftSource: t.shiftSource,
      stages: [stage],
      modelEvalsTotal: stage.modelEvals)
  }
}

extension LoRASource {
  /// The path the record names: the local file, or the HF `repo/filename`.
  var recordPath: String {
    switch self {
    case .local(let url): return url.path
    case .huggingFace(let modelId, let filename): return filename.map { "\(modelId)/\($0)" } ?? modelId
    }
  }
}
