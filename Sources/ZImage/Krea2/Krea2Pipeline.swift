// Krea2Pipeline.swift — End-to-end Krea-2-Turbo text-to-image pipeline.
//
// Port of docs/krea2-reference/krea2/{pipeline,sampling}.py: encode (Qwen3-VL-4B,
// 12-layer tap) → flow-matching Euler with resolution-shifted timesteps (no CFG;
// turbo is guidance-distilled) → Qwen-Image VAE decode. Noise is generated in
// NCHW with MLXRandom.seed to match the reference RNG stream exactly.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Logging

// MARK: - Model file locations

public struct Krea2ModelPaths {
  public let root: URL
  /// The physical variant read from the checkpoint file present in `root`
  /// (WP-E5, D7). Reported, never requested.
  public let variant: Krea2Variant
  /// Stored, not derived: a `model_index.json` escape hatch may name a third
  /// filename (FDD §3.5), and provenance records exactly what loaded.
  public let transformerFile: URL
  public var textEncoderFile: URL { root.appending(path: "text_encoder/model.safetensors") }
  /// The model directory's VAE — the bottom of the selection precedence
  /// (WP-E9, §3.9): `model_index.json` `"vae_file"` when declared, else
  /// `vae/diffusion_pytorch_model.safetensors`. Stored, not derived, for the
  /// same reason as `transformerFile`.
  public let vaeFile: URL
  public var tokenizerDirectory: URL { root.appending(path: "tokenizer") }

  public static func defaultVAEFile(root: URL) -> URL {
    root.appending(path: "vae/diffusion_pytorch_model.safetensors")
  }

  /// A root whose transformer is `variant.transformerFilename`. Use
  /// `Krea2ModelDetection.detect(at:)` to read the variant off disk.
  public init(root: URL, variant: Krea2Variant = .turbo) {
    self.init(root: root, variant: variant, transformerFile: root.appending(path: variant.transformerFilename))
  }

  public init(root: URL, variant: Krea2Variant, transformerFile: URL, vaeFile: URL? = nil) {
    self.root = root
    self.variant = variant
    self.transformerFile = transformerFile
    self.vaeFile = vaeFile ?? Self.defaultVAEFile(root: root)
  }

  /// An explicit dir is detected fail-closed (variant read from disk, throws
  /// on a non-Krea-2 dir); nil → the newest HF-cache snapshot of krea/Krea-2-Turbo.
  public static func resolve(modelDir: String? = nil) throws -> Krea2ModelPaths {
    if let modelDir {
      return try Krea2ModelDetection.detect(
        at: URL(fileURLWithPath: (modelDir as NSString).expandingTildeInPath, isDirectory: true))
    }
    return try turboSnapshot()
  }

  /// The newest HF-cache snapshot of krea/Krea-2-Turbo. Reached ONLY through
  /// the four declared turbo aliases (`Krea2ModelDetection.turboAliases`).
  public static func turboSnapshot() throws -> Krea2ModelPaths {
    let snapshots = ("~/.cache/huggingface/hub/models--krea--Krea-2-Turbo/snapshots" as NSString)
      .expandingTildeInPath
    let fm = FileManager.default
    guard let entries = try? fm.contentsOfDirectory(atPath: snapshots), !entries.isEmpty else {
      throw Krea2WeightLoaderError.missingFile(snapshots)
    }
    // Pick the snapshot that actually contains the transformer file.
    for entry in entries.sorted(by: >) {
      let candidate = URL(fileURLWithPath: snapshots).appending(path: entry)
      if fm.fileExists(atPath: candidate.appending(path: "turbo.safetensors").path) {
        return Krea2ModelPaths(root: candidate, variant: .turbo)
      }
    }
    throw Krea2WeightLoaderError.missingFile("\(snapshots)/*/turbo.safetensors")
  }
}

// MARK: - Sampling math (port of sampling.py)

enum Krea2Sampling {
  /// Classifier-free guidance combine: uncond + scale * (cond - uncond).
  /// scale 1.0 returns cond exactly (the guidance-free distill recipe).
  static func applyCFG(cond: MLXArray, uncond: MLXArray, scale: Float) -> MLXArray {
    uncond + scale * (cond - uncond)
  }

  static func roundUp(_ value: Int, multiple: Int) -> Int {
    ((value + multiple - 1) / multiple) * multiple
  }

  /// img2img's first grid index for `strength` over `total` intervals —
  /// Z-Image's convention exactly (`ImageToImagePipeline.swift`):
  /// `denoise = 1 - clamp(strength, 0.01, 0.99)`,
  /// `start = max(0, total - ceil(total · denoise))`. Pure, so the
  /// `steps_run` accounting (AC-63) is testable without weights.
  public static func img2imgStartIndex(total: Int, strength: Float) -> Int {
    let denoise = 1.0 - max(0.01, min(0.99, strength))
    return max(0, total - Int((Double(total) * Double(denoise)).rounded(.up)))
  }

  /// (b, c, H, W) -> (b, (H/p)*(W/p), c*p*p) with [c, ph, pw] channel ordering.
  static func patchify(_ x: MLXArray, patch p: Int) -> MLXArray {
    let b = x.dim(0), c = x.dim(1), H = x.dim(2), W = x.dim(3)
    let h = H / p, w = W / p
    return x.reshaped(b, c, h, p, w, p)
      .transposed(0, 2, 4, 1, 3, 5)
      .reshaped(b, h * w, c * p * p)
  }

  /// (b, h*w, c*p*p) -> (b, c, h*p, w*p)
  static func unpatchify(_ x: MLXArray, patch p: Int, h: Int, w: Int, c: Int) -> MLXArray {
    let b = x.dim(0)
    return x.reshaped(b, h, w, c, p, p)
      .transposed(0, 3, 1, 4, 2, 5)
      .reshaped(b, c, h * p, w * p)
  }

  /// Positions for [txt; img]: text at origin, image tokens on the (h, w) grid.
  static func buildPositions(txtLen: Int, h: Int, w: Int) -> MLXArray {
    var pos = [Float](repeating: 0, count: (txtLen + h * w) * 3)
    for row in 0..<h {
      for col in 0..<w {
        let base = (txtLen + row * w + col) * 3
        pos[base + 1] = Float(row)
        pos[base + 2] = Float(col)
      }
    }
    return MLXArray(pos, [txtLen + h * w, 3])
  }

  /// Per-axis NTK scale factors for DyPE, as [axis0, height, width].
  ///
  /// Axis 0 is the text/frame axis and always stays at 1.0. Height and width
  /// scale by how far the current token grid exceeds the grid the model trained
  /// at: baseResolution / (spatialScale * patch), which is 1024/16 = 64 tokens.
  static func ropeScales(
    hTok: Int, wTok: Int, patch: Int, dyPE: DyPEConfig
  ) -> [Float] {
    guard dyPE.enabled, dyPE.method != .none else { return [1, 1, 1] }
    let baseTokens = Float(dyPE.baseResolution / (Krea2VAE.spatialScale * patch))
    guard baseTokens > 0 else { return [1, 1, 1] }
    return [1, Float(hTok) / baseTokens, Float(wTok) / baseTokens]
  }

  /// Mix pure noise with an encoded source latent at the schedule sigma a
  /// partial denoise starts from — rectified flow's "noise a real sample to
  /// time σ": `noise·σ + source·(1 − σ)`, rounded to `dtype` once, at the end.
  ///
  /// `sigma` MUST be the scheduler's float32 0-d `MLXArray`
  /// (`scheduler.sigmas[startIndex]`), never `.item(Float.self)`. This is
  /// FDD-krea2-raw-recipe §3.3's byte-identity trap: mlx-swift converts a
  /// Swift `Float` operand to the ARRAY's dtype first
  /// (`MLXArray+Ops.swift:253-255`), so a scalar would run the whole mix in
  /// bf16 and move every img2img render (AC-2). A float32 0-d array promotes
  /// the mix to float32 — exactly what the pre-WP-E3 `MLXArray(ts[startIndex])`
  /// did. The precondition makes the trap a crash, not a drift.
  static func mixSourceLatent(
    noise: MLXArray, source: MLXArray, sigma: MLXArray, dtype: DType
  ) -> MLXArray {
    precondition(
      sigma.dtype == .float32 && sigma.ndim == 0,
      "the img2img mix needs the float32 0-d sigma array; a Float scalar runs the mix in bf16 (§3.3)")
    return (noise * sigma + source * (1.0 - sigma)).asType(dtype)
  }

  /// Resolution-shifted timestep schedule (exp/sigmoid warp), 1 → 0 inclusive.
  ///
  /// Signature unchanged; the warp itself now lives in
  /// ``SigmaSchedule/krea2(numSteps:mu:sigmaExp:)`` and this delegates to it,
  /// so the pipelines' grid and `SchedulerFactory`'s `.krea2` grid are one body
  /// (FDD-krea2-raw-recipe §3.1, AC-3). The `mu` derivation is the original
  /// inline slope form — `mu(seqLen:align:)` is pinned equal to it.
  static func timesteps(
    seqLen: Int, steps: Int, x1: Float, x2: Float,
    y1: Float = 0.5, y2: Float = 1.15, sigma: Float = 1.0, mu muOverride: Float? = nil
  ) -> [Float] {
    let slope = (y2 - y1) / (x2 - x1)
    let mu = muOverride ?? (slope * Float(seqLen) + (y1 - slope * x1))
    return SigmaSchedule.krea2(numSteps: steps, mu: mu, sigmaExp: sigma)
  }

  /// The schedule's resolution-dependent log-shift `mu`: linear in the image
  /// token count from 0.5 at a 256² render to 1.15 at 1280², with the token
  /// grid aligned to `align` (VAE spatial scale × patch = 16). Delegates to
  /// `PipelineUtilities.calculateShift` — the same function Z-Image uses — with
  /// Krea 2's constants: base 256 tokens / 0.5, max 6400 tokens / 1.15.
  ///
  /// Effective shift is `exp(mu)`: ~2.475 at 1024². ComfyUI registers Krea 2
  /// as `ModelSamplingFlux(shift=1.15)` — a *fixed* mu of 1.15 (effective
  /// `e^1.15` ≈ 3.158) — and an explicit request `shift` replaces this value
  /// **as mu** (FDD D3 as amended by Addendum A.1).
  static func mu(seqLen: Int, align: Int) -> Float {
    let baseSeqLen = (256 / align) * (256 / align)      // 256 at align 16
    let maxSeqLen = (1280 / align) * (1280 / align)     // 6400 at align 16
    return PipelineUtilities.calculateShift(
      imageSeqLen: seqLen,
      baseSeqLen: baseSeqLen,
      maxSeqLen: maxSeqLen,
      baseShift: 0.5,
      maxShift: 1.15
    )
  }

  /// ComfyUI's `ModelSamplingFlux` table size for Krea 2 (`timesteps=10000`).
  static let fluxTableSize = 10000

  /// Synthetic scheduler config for Krea 2, which ships no `scheduler_config.json`.
  ///
  /// Carries the constants baked into `timesteps` so `SchedulerFactory` can
  /// build any `SigmaScheduleKind` for this family. Values are specified, not
  /// implicit (FDD-krea2-raw-recipe §3.1, Addendum A.1; pinned by
  /// `Krea2SigmaScheduleTests.testSchedulerConfigPinnedValues`):
  ///
  /// - `numTrainTimesteps: 1000` — sigma ↔ timestep scale.
  /// - `shift: 1.0` — **not** where Krea 2's shift lives. The shift is `mu`,
  ///   resolved per render by `resolveShift` and handed to the factory beside
  ///   this config; `.flow`'s base grid reads this value and stays unshifted.
  /// - `useDynamicShifting: true` — so `.flow` and `.krea2` both take the `mu` warp.
  /// - `baseShift 0.5 / maxShift 1.15 / baseImageSeqLen 256 / maxImageSeqLen 6400`
  ///   — the `mu(seqLen:align:)` line.
  /// - `modelSampling: .flux(tableSize: 10000)` — ComfyUI registers Krea 2 as
  ///   `ModelSamplingFlux`, so `beta`/`beta57` index the 10 000-entry Flux
  ///   table built from `mu`, and `karras`/`exponential` take its bounds —
  ///   never the 1000-entry DiscreteFlow table (A.1).
  static func schedulerConfig() -> ZImageSchedulerConfig {
    ZImageSchedulerConfig(
      numTrainTimesteps: 1000,
      shift: 1.0,
      useDynamicShifting: true,
      baseShift: 0.5,
      maxShift: 1.15,
      baseImageSeqLen: 256,
      maxImageSeqLen: 6400,
      modelSampling: .flux(tableSize: fluxTableSize)
    )
  }

  // MARK: Schedule shift (FDD-krea2-raw-recipe D3, as amended by Addendum A.1)

  /// Where the schedule's shift came from — recorded by provenance (WP-E10).
  enum ShiftSource: String, Sendable {
    /// The resolution-dependent `mu` from `mu(seqLen:align:)` (the default).
    case dynamic
    /// A request-stated `shift`; `mu = shift` (shift IS mu, A.1).
    case explicit
  }

  /// Everything a schedule needs to know about the shift, resolved once per
  /// render: the log-shift `mu` — which the warp-by-`mu` schedules (`.krea2`,
  /// `.flow`) take directly and the table-backed schedules (`beta`, `beta57`,
  /// `karras`, `exponential`) build the `ModelSamplingFlux` table from — the
  /// effective linear `shift` (`e^mu`, for the record), its source, and the
  /// family's `ZImageSchedulerConfig`. Hand `config` and `mu` to
  /// `SchedulerFactory` together; the factory refuses the table-backed
  /// schedules without `mu`.
  struct ScheduleShift {
    let mu: Float
    let shift: Float
    let source: ShiftSource
    let config: ZImageSchedulerConfig
  }

  /// Resolve the schedule shift for a render (D3 as amended by A.1).
  ///
  /// - `explicit == nil` → today's resolution-dependent `mu` (every existing
  ///   render is unmoved).
  /// - `explicit == s` → **`mu = s`**. ComfyUI registers Krea 2 as
  ///   `ModelSamplingFlux(shift=1.15)`, whose `flux_time_shift(mu=shift, t) =
  ///   e^mu / (e^mu + 1/t − 1)` is the same function as `SigmaSchedule.krea2`'s
  ///   warp — so the wire's `shift` is mu directly and the effective linear
  ///   shift is `e^s` (≈ 3.158 at 1.15). The earlier `mu = log(s)` mapping is
  ///   withdrawn: it would have rendered the published `shift: 1.15` as a
  ///   linear 1.15 (mu 0.14), nowhere near the workflow's grid.
  ///
  /// Either way the config is `schedulerConfig()` — the same Flux family
  /// object; only `mu` moves.
  ///
  /// - Throws: ``Krea2ScheduleError/invalidShift(_:)`` for a shift that is not
  ///   a positive finite number. There is no clamp and no fallback.
  static func resolveShift(explicit: Float?, seqLen: Int, align: Int) throws -> ScheduleShift {
    let config = schedulerConfig()
    guard let explicit else {
      let mu = mu(seqLen: seqLen, align: align)
      return ScheduleShift(mu: mu, shift: Foundation.exp(mu), source: .dynamic, config: config)
    }
    guard explicit.isFinite, explicit > 0 else { throw Krea2ScheduleError.invalidShift(explicit) }
    return ScheduleShift(mu: explicit, shift: Foundation.exp(explicit), source: .explicit, config: config)
  }
}

/// Errors raised while resolving a Krea 2 schedule from a request.
public enum Krea2ScheduleError: Error, Equatable, CustomStringConvertible {
  /// The request's `shift` is not a positive finite number. `shift` is the
  /// schedule's mu and there is no value to substitute (FDD-krea2-raw-recipe
  /// D3, Addendum A.1).
  case invalidShift(Float)

  /// A request field whose behaviour belongs to a parity tier the loop does
  /// not implement yet (FDD-krea2-raw-recipe D18, §3.13). Refused at the
  /// pipeline so a non-server caller cannot get a silent downgrade either —
  /// "fail loud, never silently substitute".
  ///
  /// **It currently has no throw site, by intent.** `eta` was T2 and
  /// `bongmath` was T3; both landed (WP-E15, WP-E16) and each is now honoured
  /// or refused BY SAMPLER at its own factory. The case and
  /// ``Krea2Pipeline/validateTiers(eta:bongmath:)`` are kept as the matched
  /// pair the NEXT such field uses, so a tier gate is never invented under
  /// time pressure at a fresh call site. Delete both together or neither.
  case tierNotImplemented(field: String, value: String, tier: String)

  /// `eta != 0` was asked for with a sampler RES4LYF's SDE is not defined
  /// against (WP-E15, D18).
  ///
  /// The SDE is a property of RES4LYF's own solver: it splits `σ → σ'` into
  /// `σ → σ_down` plus a re-noise, re-noises the non-final ROWS of the
  /// tableau, and lands on the prepared grid's `sigma_min`. The samplers here
  /// that are NOT RES4LYF ports (`euler`, `heun`, `dpmpp-2m`, `dpmpp-2s-a`,
  /// `deis`, `ddim`) walk an unprepared grid with their own semantics — on
  /// `ddim` and `dpmpp-2s-a`, `eta` already means something else entirely.
  /// Applying the split there would be an invention, so it is refused by name
  /// rather than silently ignored.
  case etaUnsupportedSampler(sampler: String, value: String)

  /// `bongmath: true` was asked for with a sampler RES4LYF's fixed point is
  /// not defined against (WP-E16, D18).
  ///
  /// The same boundary `eta` has, for the same reason: `bong_iter` re-derives
  /// the step anchor by INVERTING a RES4LYF tableau row, so it exists only for
  /// samplers that have one. `euler` — the Krea 2 default — `heun`, `dpmpp-2m`,
  /// `dpmpp-2s-a`, `deis` and `ddim` have no such row, and there is nothing to
  /// substitute. Refused by name rather than silently dropped.
  case bongmathUnsupportedSampler(sampler: String)

  public var description: String {
    switch self {
    case .invalidShift(let value):
      return "shift must be a positive number (got \(value)); omit it for the resolution-dependent default"
    case .tierNotImplemented(let field, let value, let tier):
      return "\(field)=\(value) is parity tier \(tier) (\(Self.workPackage(forTier: tier))) and is not "
        + "implemented yet; omit it or send its default"
    case .etaUnsupportedSampler(let sampler, let value):
      return "eta=\(value) is RES4LYF's SDE and applies to the RES4LYF samplers only; "
        + "'\(sampler)' is not one of them. Send eta 0, or a sampler from "
        + "res_2s / res_3s / ralston_2s / ralston_3s / ralston_4s / heun_2s / heun_3s / "
        + "deis_2m / deis_3m / deis_4m"
    case .bongmathUnsupportedSampler(let sampler):
      return "bongmath is RES4LYF's fixed point over its own tableau rows and applies to the "
        + "RES4LYF samplers only; '\(sampler)' is not one of them. Send bongmath false, or a "
        + "sampler from res_2s / res_3s / ralston_2s / ralston_3s / ralston_4s / heun_2s / "
        + "heun_3s / deis_2m / deis_3m / deis_4m"
    }
  }

  /// The work package that lands a tier — named in the error so the caller is
  /// told what is missing, not merely that something is.
  private static func workPackage(forTier tier: String) -> String {
    switch tier {
    case "T2": return "WP-E15"
    case "T3": return "WP-E16"
    default: return "unassigned"
    }
  }
}

// MARK: - Pipeline

public final class Krea2Pipeline {
  public let config: Krea2Config
  /// Where the weights came from — root, physical variant, transformer file.
  public let paths: Krea2ModelPaths
  /// The physical checkpoint variant this pipeline loaded (WP-E5, D7).
  public var variant: Krea2Variant { paths.variant }
  /// Group-wise quantization bits applied to the transformer at load, or nil
  /// when it stayed at full precision (WP-E3 §3.3; feeds
  /// `RenderRecipe.quantization`, WP-E10). Stored once from what `init`
  /// actually applied — never re-derived from the module tree, which cannot
  /// tell "loaded bf16" from "loaded quantized then restored".
  public let transformerQuantBits: Int?
  public let transformer: Krea2SingleStreamDiT
  public let textEncoder: Qwen3TextEncoder
  public let conditioner: Krea2TextConditioner
  /// The ONE `Krea2VAE` instance — serves both `decode` and `encode`, so
  /// encoder-side selection follows decoder-side automatically (AC-57). Owned
  /// by `vaeSlot`, which reloads its weights in place (WP-E9, D17).
  public var vae: Krea2VAE { vaeSlot.vae }
  public let vaeSlot: Krea2VAESlot
  /// What decoded the last/next render and how it was selected (WP-E9; feeds
  /// `RenderRecipe.vae`, WP-E10).
  public var currentVAE: Krea2VAESelection { vaeSlot.current }
  /// In-place decoder reloads since load (AC-59).
  public var vaeReloadCount: Int { vaeSlot.reloadCount }

  /// The depth Control-LoRA as applied by `setControlLoRA` (file, scale and
  /// the strict bind report); nil when none is active (WP-E10 `control_lora`).
  public private(set) var controlLoRAApplied: RenderRecipe.ControlReadBack?

  private let logger = Logger(label: "z-image.krea2-pipeline")

  /// Currently applied LoRA configurations (for hot-swap tracking).
  private var appliedLoRAs: [LoRAConfiguration] = []

  /// Bare-parameter patch state (.diff/.diff_b/.set_weight — e.g. Kroma's 159
  /// norm/modulation deltas). Instance-scoped by construction: owns detached
  /// first-write-wins snapshots for this transformer only, restored on clear.
  private lazy var patchSession = LoRAPatchSession(module: transformer)

  /// Public accessor for currently loaded LoRA configurations.
  public var loadedLoRAConfigs: [LoRAConfiguration] { appliedLoRAs }

  /// One report per entry of `loadedLoRAConfigs`, same order (WP-E6). Every
  /// report here is complete (`bound == offered`, `unbound.isEmpty`) because
  /// Krea-2 applies strictly — a partial bind never survives `loadLoRAs`.
  /// `deltasApplied` feeds `RenderRecipe.loras[].deltas_applied` (D15).
  public private(set) var loadedLoRAReports: [LoRAApplicationReport] = []

  /// The relativity the guard ENFORCED for each entry of
  /// `loadedLoRAConfigs`, same order — `config.requiresBase ??
  /// Krea2LoRARelativity.seeded(forFilename:)` (K-FIX-1 / Codex I6). Kept
  /// beside the reports because the resolved value is what `applied.loras[]
  /// .relative_to` must name: the seed table supplies it for every file no
  /// preset declares, and discarding it made a guarded render read as
  /// unguarded. `nil` at an index means nothing was declared for that file.
  public private(set) var loadedLoRARelativities: [Krea2Variant?] = []

  public struct Request {
    public var prompt: String
    /// Negative prompt for the CFG branch — only consulted when guidance > 1.
    public var negativePrompt: String?
    /// Classifier-free guidance scale. 1.0 (default) = the distilled
    /// single-pass recipe (no CFG, no negative). >1.0 runs a second
    /// unconditioned pass per step (~2x time; Kroma's card blesses up to 1.5).
    public var guidance: Float
    public var width: Int
    public var height: Int
    public var steps: Int
    public var seed: UInt64
    /// Depth Control-LoRA init image: RGB NHWC in [-1,1] (see QwenImageIO.normalizeForEncoder),
    /// already resized to the target width/height. nil = no depth control.
    public var controlImagePixels: MLXArray?
    /// High-resolution position handling. `.disabled` keeps vanilla RoPE.
    public var dyPE: DyPEConfig = .disabled
    /// Explicit schedule shift (FDD-krea2-raw-recipe D3, Addendum A.1). `nil`
    /// (default) keeps the resolution-dependent `mu` every existing render
    /// uses; a value **is** `mu` for schedule construction (ComfyUI's
    /// `ModelSamplingFlux(shift=…)` parameterisation; effective linear shift
    /// `e^shift`) — the `krea2-reference` preset states `1.15`, ComfyUI's
    /// registered shift. Must be > 0; `generate` throws
    /// ``Krea2ScheduleError/invalidShift(_:)``.
    public var shift: Float? = nil
    /// The sampler the denoise loop runs (WP-E3, §3.3). `.euler` is today's
    /// behaviour. The WIRE key is `scheduler`, with `sampler` as a declared
    /// alias (D25) — the naming asymmetry is deliberate and lives in
    /// `RecipeNameResolver`, not here.
    public var sampler: SchedulerKind = .euler
    /// The sigma grid the sampler walks. `.krea2` is the family's native warp
    /// and today's behaviour; `.flow` and the rest stay legal (D11).
    public var sigmaSchedule: SigmaScheduleKind = .krea2
    /// The raw schedule name the caller sent, when it differed from the
    /// resolved kind (Krita's `normal` → `.flow`, D22). Record-only: it never
    /// changes what runs, it makes the alias visible in the trace.
    public var sigmaScheduleRequested: String? = nil
    /// RES4LYF SDE eta (parity tier T2, WP-E15 — landed). NOTE: the SAME
    /// wire field `eta` means DDIM η / DPM++ 2S-A ancestral η on the Z-Image
    /// path, where it has shipped since April. Two meanings, one key — gated
    /// per family, never at the decoder (D18).
    ///
    /// Non-zero runs RES4LYF's SDE on a RES4LYF sampler and throws
    /// ``Krea2ScheduleError/etaUnsupportedSampler(sampler:value:)`` on any
    /// other — including ``sampler``'s own default, `euler`.
    public var eta: Float = 0.0
    /// RES4LYF `bongmath` (parity tier T3, WP-E16). `true` throws until it lands.
    public var bongmath: Bool = false
    /// `res_2s` / `res_3s` substep location in log-sigma space. The optional
    /// wire field defaults here to the reference recipe's midpoint, 0.5.
    public var c2: Float = 0.5
    /// WP-E17 (§3.14, D4): the optional second stage of ONE render — the
    /// detail pass, which re-noises the latent to the stretched tail's first
    /// sigma and solves again with its own sampler, schedule, eta, bongmath
    /// and seed. `nil` (the default) is today's single-stage render, statement
    /// for statement. See ``Krea2Pipeline/Stage2``.
    public var stage2: Stage2? = nil
    /// Text-conditioning gain on the fusion projector (projector-scale trick).
    /// 1.0 = neutral; >1 strengthens prompt adherence with no CFG cost. Applied
    /// on the warm transformer for the whole render, stage 2 included.
    public var projectorScale: Float = 1.0
    /// RES4LYF spatial noise generator for the SDE re-noise (opt-in). `.gaussian`
    /// (default) is byte-identical to today; `.fractal` / `.pyramid` are the
    /// RES4LYF alternatives, active only when `eta != 0` on a RES4LYF sampler.
    public var noiseType: RES4LYFNoiseType = .gaussian
    /// Fractal `alpha` exponent (RES4LYF `FractalNoiseGenerator`). 0 makes
    /// fractal byte-identical to gaussian; only read when `noiseType == .fractal`.
    public var noiseAlpha: Float = 0.0
    /// RES4LYF implicit-RK refinement: re-iterate the explicit tableau
    /// `implicitStepsFull` extra times as a fixed point (upstream's `full_iter`
    /// loop). 0 (default) is byte-identical to today's single explicit pass;
    /// >0 re-anchors row 0 on the previous pass's x_next (see
    /// ``Krea2DenoiseLoop``). Scoped to `heun_2s`, guides off, eta 0.
    public var implicitStepsFull: Int = 0
    public init(prompt: String, negativePrompt: String? = nil, guidance: Float = 1.0,
                width: Int = 1024, height: Int = 1024, steps: Int = 9, seed: UInt64 = 0,
                controlImagePixels: MLXArray? = nil, dyPE: DyPEConfig = .disabled,
                shift: Float? = nil,
                sampler: SchedulerKind = .euler, sigmaSchedule: SigmaScheduleKind = .krea2,
                sigmaScheduleRequested: String? = nil,
                eta: Float = 0.0, bongmath: Bool = false, c2: Float = 0.5,
                stage2: Stage2? = nil, projectorScale: Float = 1.0,
                noiseType: RES4LYFNoiseType = .gaussian, noiseAlpha: Float = 0.0,
                implicitStepsFull: Int = 0) {
      self.prompt = prompt
      self.negativePrompt = negativePrompt
      self.guidance = guidance
      self.width = width
      self.height = height
      self.steps = steps
      self.seed = seed
      self.controlImagePixels = controlImagePixels
      self.dyPE = dyPE
      self.shift = shift
      self.sampler = sampler
      self.sigmaSchedule = sigmaSchedule
      self.sigmaScheduleRequested = sigmaScheduleRequested
      self.eta = eta
      self.bongmath = bongmath
      self.c2 = c2
      self.stage2 = stage2
      self.projectorScale = projectorScale
      self.noiseType = noiseType
      self.noiseAlpha = noiseAlpha
      self.implicitStepsFull = implicitStepsFull
    }
  }

  /// Whether a depth Control-LoRA is currently loaded (controlFirst set + A/B applied).
  public private(set) var controlLoRAActive = false

  public init(paths: Krea2ModelPaths, quantizeTransformer: Int? = nil) throws {
    self.config = Krea2Config()
    self.paths = paths
    self.transformerQuantBits = quantizeTransformer

    let transformer = Krea2SingleStreamDiT(cfg: config)
    logger.info("Krea2: loading \(paths.variant.rawValue) transformer from \(paths.transformerFile.path)")
    try Krea2WeightLoader.loadTransformer(transformer, from: paths.transformerFile)
    if let bits = quantizeTransformer {
      quantize(model: transformer, groupSize: 64, bits: bits) { path, module in
        // Quantize the big Linear layers only; keep norms/embeddings full precision.
        module is Linear && (module as! Linear).weight.dim(1) % 64 == 0 && !path.contains("projector")
      }
    }
    self.transformer = transformer

    let encoder = Krea2TextEncoderFactory.makeEncoder()
    try Krea2WeightLoader.loadTextEncoder(encoder, from: paths.textEncoderFile)
    self.textEncoder = encoder

    let tokenizer = try QwenTokenizer.load(from: paths.root)
    self.conditioner = Krea2TextConditioner(encoder: encoder, tokenizer: tokenizer)

    // The model directory's VAE is the resident default (D16); the layout is
    // sniffed from the keys, so a Wan file declared via model_index.json
    // `vae_file` loads correctly too.
    let slot = try Krea2VAESlot(loading: paths.vaeFile, source: .modelDir)
    self.vaeSlot = slot
    logger.info("Krea2: VAE \(slot.current.layout.rawValue) from \(paths.vaeFile.path)")

    MLX.eval(transformer.parameters(), encoder.parameters(), slot.vae.parameters())
  }

  // MARK: - VAE selection (WP-E9)

  /// Make `path` the resident decoder, reloading IN PLACE on the one
  /// `Krea2VAE` instance — never a pool eviction (D17). Returns `true` when
  /// weights were reloaded, `false` when `path` was already resident.
  /// Fail-closed: a file that is not on disk, or a layout `detectLayout`
  /// cannot name, throws and leaves the resident decoder untouched.
  @discardableResult
  public func ensureVAE(
    path: URL, layout: VAELayout? = nil, source: Krea2VAESelection.Source = .payload
  ) throws -> Bool {
    let previous = vaeSlot.current
    let reloaded = try vaeSlot.ensure(file: path, layout: layout, source: source)
    if reloaded {
      let line = "Krea2: VAE reloaded in place \(previous.layout.rawValue) \(previous.file.lastPathComponent) → "
        + "\(vaeSlot.current.layout.rawValue) \(path.path) (source=\(source.rawValue), reloads=\(vaeSlot.reloadCount))"
      logger.info("\(line)")
    }
    return reloaded
  }

  // MARK: - LoRA Support

  /// Load and apply LoRAs to the Krea-2 transformer.
  ///
  /// Clears any previously applied LoRAs before applying the new set. Uses
  /// ``LoRAWeightLoader/loadForKrea2(from:)`` — Krea-2 LoRA keys match
  /// `Krea2SingleStreamDiT` module paths 1:1, no remapping needed.
  ///
  /// - Parameter configs: LoRA configurations to apply. Pass an empty array to clear all LoRAs.
  public func loadLoRAs(_ configs: [LoRAConfiguration]) async throws {
    if !appliedLoRAs.isEmpty || patchSession.isActive {
      LoRAApplicator.clearDynamicLoRA(from: transformer, logger: logger)
      patchSession.clear()
      appliedLoRAs = []
      loadedLoRAReports = []
      loadedLoRARelativities = []
    }

    guard !configs.isEmpty else { return }

    // Load and preflight-able failures (missing file, bad format, unknown
    // keys, wrong-base relativity, strict partial bind) all surface BEFORE
    // any weight mutation for that config. If a later config fails after
    // earlier ones applied, roll the whole stack back so applied weights and
    // `appliedLoRAs` can never disagree (delta-key spec rev 2, Codex finding 2).
    var reports: [LoRAApplicationReport] = []
    var relativities: [Krea2Variant?] = []
    do {
      for config in configs {
        let url = try await LoRAWeightLoader.resolveSource(config.source)
        let name = config.source.displayName
        // WP-E6 / AC-41 relativity guard — before the file is even read.
        // I6: the RESOLVED value is kept, not just checked and dropped — it
        // is what provenance records as `relative_to`.
        let required = Krea2LoRARelativity.required(for: config, resolvedURL: url)
        try Krea2LoRARelativity.check(lora: name, required: required, loaded: variant)
        // A pair whose target is a BARE PARAMETER (the distills'
        // `last.modulation.lin` — a (2, features) array, not a Linear) becomes
        // a `.diff` on that parameter, so it applies through `patchSession`
        // with an exact restore instead of being reported as unbound and
        // taking the whole stack down with it. Inert for every adapter whose
        // keys all name Linears. See ``LoRABareParameterPairs``.
        // Full-matrix LoKr layers become dense `.diff` deltas on their target
        // weights (comfybox#329): ΔW = kron(w1, w2) · alpha-scale, applied
        // through `patchSession` whose first-write-wins snapshots (exact
        // packed q8 weight/scales/biases tuple included) give LoKr the
        // exact-restore transactionality the C1 guard below demands. A layer
        // the densifier cannot prove out (no bindable target module) stays
        // LoKr-shaped and the guard still refuses the file whole.
        let weights = try LoKrDensifier.densify(
          LoRABareParameterPairs.split(
            LoRAWeightLoader.loadForKrea2(from: url), for: transformer, name: name),
          for: transformer, name: name)
        // K-FIX-1 / Codex C1 — the second half of the transactional contract,
        // and like the relativity guard it fires BEFORE any weight mutation:
        // in-place LoKr rewrites base parameters and `clearDynamicLoRA` (this
        // block's rollback) cannot restore them, so the stack would accumulate
        // across renders while `appliedLoRAs` reported none. The densifier
        // above converts every provable full-matrix layer to a transactional
        // delta (lokrLayerCount 0); this guard is the backstop for whatever
        // it could not convert. Refuse instead.
        try Krea2AdapterSupport.checkTransactional(
          lokrLayerCount: weights.lokrLayerCount, lora: name)
        logger.info("Applying Krea-2 LoRA: \(name) (rank=\(weights.rank), layers=\(weights.layerCount), deltas=\(weights.deltas.count), scale=\(config.scale), base=\(variant.rawValue))")
        let report = try LoRAApplicator.applyDynamically(
          to: transformer, loraWeights: weights, scale: config.scale,
          strict: true, name: name, logger: logger)
        let deltas = try patchSession.apply(weights: weights, scale: config.scale)
        reports.append(report.withDeltasApplied(deltas))
        relativities.append(required)
        logger.info("Krea-2 LoRA \(name): bound \(report.bound)/\(report.offered) (\(report.quantizedBound) quantized), deltas=\(deltas)")
      }
    } catch {
      logger.error("Krea-2 LoRA stack failed mid-apply — rolling back to base: \(error)")
      LoRAApplicator.clearDynamicLoRA(from: transformer, logger: logger)
      patchSession.clear()
      appliedLoRAs = []
      loadedLoRAReports = []
      loadedLoRARelativities = []
      throw error
    }

    appliedLoRAs = configs
    loadedLoRAReports = reports
    loadedLoRARelativities = relativities
  }

  /// Load (or clear) the depth Control-LoRA. Sets the expanded input projection
  /// (`controlFirst*`) on the transformer and applies the rank-64 A/B adapters to
  /// the 28 blocks. MUST be called AFTER `loadLoRAs(identity)` because `loadLoRAs`
  /// clears all dynamic LoRAs first — the control A/B ride on top of the identity
  /// stack and are re-applied per control render. Pass nil to clear.
  ///
  /// - Parameters:
  ///   - url: path to `depth-control-lora.safetensors`, or nil to clear.
  ///   - scale: control strength → LoRA α (latent gain stays 1.0).
  public func setControlLoRA(_ url: URL?, scale: Float = 1.0) async throws {
    // Reset to the identity-only LoRA baseline before (re)applying control.
    // `applyDynamically` APPENDS adapters onto the existing dynamic-LoRA stack,
    // so without this reset (a) a second consecutive control render stacks a
    // DUPLICATE control adapter (escalating "crystalline melt"), and (b) clearing
    // (url == nil) would leave the 224 control adapters resident, corrupting the
    // next NON-control render (control-OFF must be byte-identical — FDD crit
    // #1/#7). Re-running the tracked identity configs (`appliedLoRAs`, e.g.
    // Krea-Kira KNPV+Pinay) restores a clean, idempotent baseline. clearDynamicLoRA
    // empties every dynamic adapter (identity + any stale control) but leaves the
    // module wrappers in place, so an empty stack behaves exactly like the base.
    LoRAApplicator.clearDynamicLoRA(from: transformer, logger: logger)
    patchSession.clear()
    do {
      for cfg in appliedLoRAs {
        let src = try await LoRAWeightLoader.resolveSource(cfg.source)
        let weights = try LoKrDensifier.densify(
          LoRABareParameterPairs.split(
            LoRAWeightLoader.loadForKrea2(from: src), for: transformer,
            name: cfg.source.displayName),
          for: transformer, name: cfg.source.displayName)
        // Belt and braces: same densify → guard sequence as `loadLoRAs`, so a
        // file that passed there passes identically here. This loop re-reads
        // the files from disk, so a file swapped underneath us for one whose
        // LoKr layers can NOT all be densified is refused here too rather
        // than mutating the base on a control toggle (C1's compounding path).
        try Krea2AdapterSupport.checkTransactional(
          lokrLayerCount: weights.lokrLayerCount, lora: cfg.source.displayName)
        try LoRAApplicator.applyDynamically(
          to: transformer, loraWeights: weights, scale: cfg.scale,
          strict: true, name: cfg.source.displayName, logger: logger)
        try patchSession.apply(weights: weights, scale: cfg.scale)
      }
    } catch {
      // Same transactional posture as loadLoRAs: never leave weights and
      // tracking in disagreement (control state included).
      logger.error("identity-stack reapply failed — rolling back to base: \(error)")
      LoRAApplicator.clearDynamicLoRA(from: transformer, logger: logger)
      patchSession.clear()
      appliedLoRAs = []
      loadedLoRAReports = []
      loadedLoRARelativities = []
      transformer.controlFirstWeight = nil
      transformer.controlFirstBias = nil
      controlLoRAActive = false
      controlLoRAApplied = nil
      throw error
    }

    guard let url else {
      transformer.controlFirstWeight = nil
      transformer.controlFirstBias = nil
      controlLoRAActive = false
      controlLoRAApplied = nil
      return
    }
    let cl = try Krea2ControlLoRA.load(from: url, layers: config.layers)
    // comfybox#329 M2: this is the ONE Krea-2 `applyDynamically` the C1 guard
    // did not front. The densifier only runs on the identity stack, so a
    // LoKr-bearing control file must be refused HERE — before controlFirst is
    // swapped in or any adapter binds — or its LoKr half would reach the
    // ungated in-place path and mutate the warm model non-transactionally.
    // `Krea2ControlLoRA.load` surfaces LoKr tensors precisely so this count
    // is truthful.
    try Krea2AdapterSupport.checkTransactional(
      lokrLayerCount: cl.loraWeights.lokrLayerCount, lora: url.lastPathComponent)
    // assertBaseHalfMatches skipped: transformer.first is q8-quantized (weight access unsafe on QuantizedLinear)
    let cw = cl.firstWeight
    let cb = cl.firstBias
    MLX.eval(cw, cb)
    transformer.controlFirstWeight = cw
    transformer.controlFirstBias = cb
    // Strict: the 224 control adapters must ALL bind or none do (WP-E6).
    let controlReport: LoRAApplicationReport
    do {
      controlReport = try LoRAApplicator.applyDynamically(
        to: transformer, loraWeights: cl.loraWeights, scale: scale,
        strict: true, name: url.lastPathComponent, logger: logger)
    } catch {
      transformer.controlFirstWeight = nil
      transformer.controlFirstBias = nil
      controlLoRAActive = false
      controlLoRAApplied = nil
      throw error
    }
    controlLoRAActive = true
    controlLoRAApplied = RenderRecipe.ControlReadBack(file: url, scale: scale, report: controlReport)
    logger.info("Krea-2 depth Control-LoRA active (scale=\(scale))")
  }

  // MARK: - Request → scheduler (WP-E3, §3.3)

  /// Build the scheduler a request asks for. Pure — no weights, no GPU — so
  /// the resolution rules are asserted without a model
  /// (`Krea2SchedulerResolutionTests`).
  ///
  /// The default `(.euler, .krea2)` is today's grid element-for-element:
  /// `SigmaSchedule.krea2(numSteps:mu:)` is the body
  /// `Krea2Sampling.timesteps` already delegates to (AC-3), and
  /// `FlowMatchEulerScheduler.step`'s `(σ_{i+1} − σ_i).asType(sample.dtype)`
  /// is the same rounding as the pre-change `(tp − tc) * v` (§4 criterion 1).
  ///
  /// `eta` is deliberately NOT forwarded to the factory: on the Krea 2 family
  /// `eta` means the RES4LYF SDE eta (T2), not the DDIM/ancestral η the factory
  /// would apply. It reaches the run through
  /// ``makeSDEInjector(eta:sampler:stageSeed:layout:)`` and the loop's T2 hook
  /// instead, which is a property of the SOLVER and not of the grid — so the
  /// factory's own defaults stand here and the schedule is unmoved by eta.
  static func makeScheduler(
    sampler: SchedulerKind,
    sigmaSchedule: SigmaScheduleKind,
    steps: Int,
    shift: Krea2Sampling.ScheduleShift,
    seed: UInt64,
    c2: Float
  ) throws -> any ZImageScheduler {
    try SchedulerFactory.create(
      kind: sampler,
      sigmaSchedule: sigmaSchedule,
      numInferenceSteps: steps,
      config: shift.config,
      mu: shift.mu,
      seed: seed,
      c2: c2,
      // S-FIX-1: on the Krea 2 family a RES4LYF sampler runs RES4LYF's
      // `prepare_sigmas` — solve to the model's σ_min, then convert to 0
      // model-free. A no-op for `.euler` and every other non-RES4LYF kind, so
      // the default path's grid is unmoved (AC-1/AC-2).
      res4lyfSigmaPreparation: true)
  }

  /// Refuse the request fields whose behaviour is not implemented yet, before
  /// any model work (D18). A silent downgrade to the default is the one
  /// outcome this programme forbids.
  ///
  /// `eta`'s arm went with WP-E15 and **`bongmath`'s went with WP-E16**: both
  /// tiers have landed, so each is now either honoured or refused BY SAMPLER
  /// at its own factory (``makeSDEInjector(eta:sampler:stageSeed:layout:)``,
  /// ``makeBongMath(bongmath:sampler:)``) — never here, and never ignored.
  ///
  /// The function stays, empty, because it is the SEAM: it is called before
  /// any model work on both generate paths, and the next field whose behaviour
  /// arrives ahead of its implementation belongs here rather than in a new
  /// call site invented under time pressure.
  static func validateTiers(eta: Float, bongmath: Bool) throws {}

  /// WP-E16 (§3.13, §4 AC-26, D18): the RES4LYF `bongmath` hook a request asks
  /// for, or `nil` when it asks for none.
  ///
  /// Pure — no weights, no GPU — so the honour/refuse boundary is asserted
  /// without a model, exactly as ``makeSDEInjector(eta:sampler:stageSeed:layout:)``
  /// is. Three outcomes and no fourth:
  ///   * `bongmath == false` → `nil`; the loop is handed no hook at all and is
  ///     bit-identical to the run without one.
  ///   * `true` on a RES4LYF sampler → the fixed point, bounded by the ACTIVE
  ///     model sampling's `σ_min` / `σ_max` (upstream's own guards are written
  ///     in them, not in constants).
  ///   * `true` on anything else → ``Krea2ScheduleError/bongmathUnsupportedSampler(sampler:)``.
  ///
  /// The bounds are DERIVED, not defaulted: upstream's guards are written in
  /// `RK.sigma_min` / `RK.sigma_max`, which are the ACTIVE model sampling's —
  /// `ModelSamplingFlux(mu)`'s table moves with `mu` (Addendum A.1), so a
  /// hard-coded `3.1575e-4` would be right only at `shift: 1.15`. Both come
  /// from the same table `SchedulerFactory` builds the grid from, and both are
  /// known before any model work, which is what lets the refusal stay early.
  static func makeBongMath(
    bongmath: Bool,
    sampler: SchedulerKind,
    sigmaSchedule: SigmaScheduleKind,
    shift: Krea2Sampling.ScheduleShift
  ) throws -> RES4LYFBongMath? {
    guard bongmath else { return nil }
    guard sampler.isRES4LYFFamily else {
      throw Krea2ScheduleError.bongmathUnsupportedSampler(sampler: sampler.rawValue)
    }
    let table = try SchedulerFactory.sigmaTable(
      schedule: sigmaSchedule, config: shift.config, mu: shift.mu)
    return RES4LYFBongMath(
      sigmaMin: Double(table[0]), sigmaMax: Double(table[table.count - 1]))
  }

  /// WP-E15 (§3.13, D18): the RES4LYF SDE injector a request asks for, or
  /// `nil` when it asks for none.
  ///
  /// Pure — no weights, no GPU — so the honour/refuse boundary is asserted
  /// without a model. Three outcomes and no fourth:
  ///   * `eta == 0` → `nil`; the loop runs exactly as it did before T2, and
  ///     ``Krea2DenoiseLoop`` is handed no hook at all.
  ///   * `eta != 0` on a RES4LYF sampler → an injector seeded from THIS
  ///     stage's seed.
  ///   * `eta != 0` on anything else → ``Krea2ScheduleError/etaUnsupportedSampler(sampler:value:)``.
  ///
  /// - Parameters:
  ///   - stageSeed: the seed of the stage being rendered — the injector's two
  ///     noise streams are derived from it, so two stages of one render have
  ///     two streams and changing only `stage2.seed` changes only stage 2
  ///     (AC-27).
  ///   - layout: where the working latent keeps its channels, for the
  ///     per-channel z-score upstream applies to every draw. Krea 2's loop runs
  ///     on the PATCHIFIED latent, so this is not `(B, C, H, W)`.
  static func makeSDEInjector(
    eta: Float,
    sampler: SchedulerKind,
    stageSeed: UInt64,
    layout: RES4LYFNoiseLayout,
    noiseType: RES4LYFNoiseType = .gaussian,
    noiseGrid: (hTok: Int, wTok: Int)? = nil,
    noiseAlpha: Double = 0.0
  ) throws -> RES4LYFSDENoiseInjector? {
    guard eta != 0 else { return nil }
    guard sampler.isRES4LYFFamily else {
      throw Krea2ScheduleError.etaUnsupportedSampler(sampler: sampler.rawValue, value: "\(eta)")
    }
    // `noiseType == .gaussian` with a nil `noiseGrid` reproduces the gaussian
    // streams exactly — the default path is byte-identical to before.
    return RES4LYFSDENoiseInjector(
      eta: Double(eta), stageSeed: stageSeed, layout: layout,
      noiseType: noiseType, grid: noiseGrid, noiseAlpha: noiseAlpha)
  }

  /// The layout ``Krea2DenoiseLoop`` sees: `(1, tokens, C·p·p)`, where the
  /// patch size only sets the group WIDTH — the channel count is what
  /// identifies the groups, and it is the VAE's.
  static let sdeNoiseLayout: RES4LYFNoiseLayout =
    .patchifiedTrailing(channels: Krea2VAE.latentChannels)

  /// Generate one image. Returns RGB float array (H, W, 3) in [0,1].
  ///
  /// A one-line wrapper over ``generateWithRecipe(_:progress:)``: the trace is
  /// handed to the caller that asked for it and held nowhere else (§3.3 — no
  /// shared "last recipe" state on the pipeline).
  ///
  /// - Throws: ``Krea2ScheduleError`` when the request's `shift` is invalid or
  ///   a field belongs to an unimplemented tier (checked before any model work).
  public func generate(
    _ request: Request,
    progress: ((Int, Int) -> Void)? = nil
  ) throws -> MLXArray {
    try generateWithRecipe(request, progress: progress).image
  }

  /// Generate one image and the record of how (WP-E3 + WP-E10).
  ///
  /// Returns the RGB array `generate` returns, plus the ``Krea2RunTrace`` the
  /// caller maps into `RenderRecipe` — the loop's own numbers, read back
  /// rather than predicted.
  ///
  /// Single-stage only. A request carrying `stage2` has TWO traces to hand
  /// back and is refused here rather than silently losing the second one; call
  /// ``generateStaged(_:progress:)``, which is the same body.
  public func generateWithRecipe(
    _ request: Request,
    progress: ((Int, Int) -> Void)? = nil
  ) throws -> (image: MLXArray, trace: Krea2RunTrace) {
    let (image, traces) = try generateStaged(request, progress: progress)
    guard traces.count == 1, let trace = traces.first else {
      throw Krea2StageError.stagedRequestNeedsStagedCall(stages: traces.count)
    }
    return (image, trace)
  }

  /// Generate one image and the record of every stage that ran (WP-E17,
  /// §3.14, D4).
  ///
  /// This IS `generateWithRecipe`'s body. A request without `stage2` executes
  /// exactly the statements it executed before WP-E17 — the stage-2 block is
  /// skipped whole, so the byte-identity gates (AC-1/AC-2/AC-5) hold by
  /// construction rather than by measurement. With `stage2`, the second stage
  /// runs on the SAME latent, between the loop and the one `vae.decode`:
  /// exactly one decode and zero encodes, whatever the stage count (AC-30).
  public func generateStaged(
    _ request: Request,
    progress: ((Int, Int) -> Void)? = nil
  ) throws -> (image: MLXArray, traces: [Krea2RunTrace]) {
    let dtype = DType.bfloat16
    let patch = config.patch
    let comp = Krea2VAE.spatialScale  // 8
    let align = comp * patch          // 16
    let width = Krea2Sampling.roundUp(request.width, multiple: align)
    let height = Krea2Sampling.roundUp(request.height, multiple: align)

    let latH = height / comp, latW = width / comp
    let hTok = latH / patch, wTok = latW / patch
    // D3/A.1: nil → resolution-dependent mu (unchanged); explicit → mu = shift. Fails before any model work.
    let scheduleShift = try Krea2Sampling.resolveShift(
      explicit: request.shift, seqLen: hTok * wTok, align: align)
    // D18: bongmath belongs to a tier that is not implemented — refuse it
    // here, before any model work, rather than rendering the default.
    try Krea2Pipeline.validateTiers(eta: request.eta, bongmath: request.bongmath)
    // WP-E15: the T2 SDE. `nil` at eta 0, and a refusal (never a silent drop)
    // when eta is asked for with a sampler it is not defined against. Built
    // before any model work for the same reason the tier gates are.
    let sdeNoise = try Krea2Pipeline.makeSDEInjector(
      eta: request.eta, sampler: request.sampler, stageSeed: request.seed,
      layout: Krea2Pipeline.sdeNoiseLayout,
      noiseType: request.noiseType, noiseGrid: (hTok: hTok, wTok: wTok),
      noiseAlpha: Double(request.noiseAlpha))
    // WP-E17 (§3.14, D18): the SECOND stage's own gates, here — before the
    // noise draw and before the first transformer forward. A `stage2` field
    // that is going to be a 400 must not cost a stage-1 render first.
    let stage2 = request.stage2?.resolved(against: request)
    // The stage's effective step count comes back from the preflight so the
    // progress this render publishes runs 1…(stage1 + stage2) instead of
    // restarting the bar at 100% → 50% halfway through (WP-E10's
    // `/health.progress_percent`).
    let stage2Steps = try stage2.map { try Krea2StagedRender.preflight(stage: $0, shift: scheduleShift) }
    // WP-E16: the T3 fixed point. `nil` at bongmath false — the loop is then
    // handed no hook and is bit-identical to the run without one — and a
    // refusal naming the sampler when it is asked for with one RES4LYF's
    // tableau inversion is not defined against. Before any model work, for the
    // same reason the SDE injector is.
    let bongMath = try Krea2Pipeline.makeBongMath(
      bongmath: request.bongmath, sampler: request.sampler,
      sigmaSchedule: request.sigmaSchedule, shift: scheduleShift)

    // Projector-scale trick: set the fusion gain on the warm transformer for
    // this render — covers stage 1 and stage 2 (same transformer instance).
    // Always assigned (default 1.0) so a prior render's value never leaks.
    transformer.txtfusion.projectorScale = request.projectorScale
    // Noise in NCHW to match the reference RNG stream.
    MLXRandom.seed(request.seed)
    let noise = MLXRandom.normal([1, Krea2VAE.latentChannels, latH, latW]).asType(dtype)

    // Conditioning: (1, L, 12, 2560) + (1, L) mask.
    let (ctxRaw, mask) = conditioner.encode([request.prompt])
    let ctx = ctxRaw.asType(dtype)
    let txtLen = ctx.dim(1)

    let img = Krea2Sampling.patchify(noise, patch: patch)  // (1, hTok*wTok, 64)
    let pos = Krea2Sampling.buildPositions(txtLen: txtLen, h: hTok, w: wTok)
    let ropeScales = Krea2Sampling.ropeScales(
      hTok: hTok, wTok: wTok, patch: patch, dyPE: request.dyPE)
    let fullMask = MLX.concatenated([mask, MLX.ones([1, hTok * wTok])], axis: 1)

    // CFG branch (opt-in, Todd 2026-08-11): guidance > 1 encodes the negative
    // (or empty) prompt and runs a second unconditioned pass per step —
    // sequential, not batched, to keep peak memory flat on the shared box.
    let useCFG = request.guidance > 1.0
    // K-FIX-1 / Codex I4: what the CFG branch ACTUALLY conditions on,
    // resolved here — beside the encode, from the same `useCFG` predicate —
    // and carried in the trace so every provenance sink records the render
    // rather than the request. An omitted negative under CFG is `""`, not
    // "no negative prompt": the second model pass ran either way.
    let negativePromptApplied = Krea2RunTrace.negativePromptApplied(
      cfgActive: useCFG, requested: request.negativePrompt)
    // WP-E17: stage 2 carries its OWN guidance, so the negative may be needed
    // by the second stage and not the first. The extra `||` term can only be
    // true when `stage2` is present, so a single-stage render encodes exactly
    // what it encoded before — and a stage 2 that asks for CFG gets it rather
    // than a silent single-pass downgrade.
    let needsNegative = useCFG || (stage2.map { $0.guidance > 1.0 } ?? false)
    var negCtx: MLXArray? = nil
    var negPos: MLXArray? = nil
    var negFullMask: MLXArray? = nil
    if needsNegative {
      let (nRaw, nMask) = conditioner.encode([request.negativePrompt ?? ""])
      negCtx = nRaw.asType(dtype)
      negPos = Krea2Sampling.buildPositions(txtLen: negCtx!.dim(1), h: hTok, w: wTok)
      negFullMask = MLX.concatenated([nMask, MLX.ones([1, hTok * wTok])], axis: 1)
    }

    // Depth Control-LoRA: VAE-encode the (already-resized) depth image once and
    // patchify to control tokens aligned with the image-token grid. Constant
    // across denoising steps (deterministic encode → cache-safe).
    var controlTokens: MLXArray? = nil
    if let ctrlPixels = request.controlImagePixels, transformer.controlFirstWeight != nil {
      let ctrlLatentNHWC = vae.encode(ctrlPixels.asType(dtype))               // (1, latH, latW, 16)
      let ctrlLatentNCHW = ctrlLatentNHWC.transposed(0, 3, 1, 2).asType(dtype) // (1, 16, latH, latW)
      controlTokens = Krea2Sampling.patchify(ctrlLatentNCHW, patch: patch)     // (1, hTok*wTok, 64)
      MLX.eval(controlTokens!)
    }

    // WP-E3: the request's sampler over the request's grid. Built AFTER the
    // noise so the global RNG stream the reference reproduces is untouched.
    var scheduler = try Krea2Pipeline.makeScheduler(
      sampler: request.sampler, sigmaSchedule: request.sigmaSchedule,
      steps: request.steps, shift: scheduleShift, seed: request.seed, c2: request.c2)

    // Progress across the WHOLE render. `nil` stage 2 → `progress` is passed
    // through untouched, so a single-stage render reports exactly what it
    // reported before (`(i + 1, total)` from the loop).
    let bar = stage2Steps.map {
      Krea2StagedRender.Progress(stage1Steps: scheduler.numInferenceSteps, stage2Steps: $0)
    }
    let stage1Progress: ((Int, Int) -> Void)? =
      bar.map { bar in
        { step, _ in
          let (done, total) = bar.stage1(step)
          progress?(done, total)
        }
      } ?? progress

    let (denoised, stats) = try Krea2DenoiseLoop.run(
      scheduler: &scheduler,
      initialSample: img,
      startIndex: 0,
      modelEvalsPerEvaluate: useCFG ? 2 : 1,
      implicitStepsFull: request.implicitStepsFull,
      evaluate: { [transformer] latent, sigma in
        // `sigma` IS Krea 2's `t`: it goes into the transformer with no
        // (1000 − t)/1000 renormalisation, exactly as the inline loop did.
        let t = MLX.full([1], values: MLXArray(sigma)).asType(dtype)
        let vCond = transformer(img: latent, context: ctx, t: t, pos: pos, mask: fullMask,
                                control: controlTokens, ropeScales: ropeScales)
        guard useCFG, let negCtx, let negPos, let negFullMask else { return vCond }
        let vUncond = transformer(img: latent, context: negCtx, t: t, pos: negPos,
                                  mask: negFullMask, control: controlTokens, ropeScales: ropeScales)
        return Krea2Sampling.applyCFG(cond: vCond, uncond: vUncond, scale: request.guidance)
      },
      noise: sdeNoise,
      bongmath: bongMath,
      progress: stage1Progress)

    var traces = [
      Krea2RunTrace(
        request: request, shift: scheduleShift, scheduler: scheduler, stats: stats,
        startIndex: 0, denoise: 1.0, width: width, height: height,
        negativePromptApplied: negativePromptApplied,
        // WP-E16: what the fixed point DID, counted by the hook while it ran —
        // `nil` when the render had none.
        bong: Krea2RunTrace.BongMathParameters.forRun(bongMath))
    ]

    // WP-E17 (§3.14): the second stage, on the LATENT. No `vae.decode` and no
    // `vae.encode` between the stages — the re-noise is a rectified-flow mix at
    // the stretched tail's first sigma, and the decode below is still the one
    // and only decode of this render (AC-30).
    var latent = denoised
    if let stage2 {
      let (staged, stage2Trace) = try Krea2StagedRender.runStage2(
        stage: stage2,
        shift: scheduleShift,
        stageOneLatent: latent,
        patch: patch, hTok: hTok, wTok: wTok,
        latentHeight: latH, latentWidth: latW,
        dtype: dtype,
        width: width, height: height,
        negativePromptApplied: Krea2RunTrace.negativePromptApplied(
          cfgActive: stage2.guidance > 1.0, requested: request.negativePrompt),
        evaluate: { [transformer] latent, sigma, guidance in
          let t = MLX.full([1], values: MLXArray(sigma)).asType(dtype)
          let vCond = transformer(img: latent, context: ctx, t: t, pos: pos, mask: fullMask,
                                  control: controlTokens, ropeScales: ropeScales)
          guard guidance > 1.0, let negCtx, let negPos, let negFullMask else { return vCond }
          let vUncond = transformer(img: latent, context: negCtx, t: t, pos: negPos,
                                    mask: negFullMask, control: controlTokens,
                                    ropeScales: ropeScales)
          return Krea2Sampling.applyCFG(cond: vCond, uncond: vUncond, scale: guidance)
        },
        progress: { step, _ in
          guard let bar else { return }
          let (done, total) = bar.stage2(step)
          progress?(done, total)
        })
      latent = staged
      traces.append(stage2Trace)
    }

    let latentNCHW = Krea2Sampling.unpatchify(
      latent, patch: patch, h: hTok, w: wTok, c: Krea2VAE.latentChannels)
    let latentNHWC = latentNCHW.transposed(0, 2, 3, 1).asType(.float32)
    let decoded = vae.decode(latentNHWC)  // (1, H, W, 3) in [0,1]
    MLX.eval(decoded)

    return (decoded[0], traces)
  }
}
