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
  public var vaeFile: URL { root.appending(path: "vae/diffusion_pytorch_model.safetensors") }
  public var tokenizerDirectory: URL { root.appending(path: "tokenizer") }

  /// A root whose transformer is `variant.transformerFilename`. Use
  /// `Krea2ModelDetection.detect(at:)` to read the variant off disk.
  public init(root: URL, variant: Krea2Variant = .turbo) {
    self.init(root: root, variant: variant, transformerFile: root.appending(path: variant.transformerFilename))
  }

  public init(root: URL, variant: Krea2Variant, transformerFile: URL) {
    self.root = root
    self.variant = variant
    self.transformerFile = transformerFile
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

  /// Resolution-shifted timestep schedule (exp/sigmoid warp), 1 → 0 inclusive.
  static func timesteps(
    seqLen: Int, steps: Int, x1: Float, x2: Float,
    y1: Float = 0.5, y2: Float = 1.15, sigma: Float = 1.0, mu muOverride: Float? = nil
  ) -> [Float] {
    let slope = (y2 - y1) / (x2 - x1)
    let mu = muOverride ?? (slope * Float(seqLen) + (y1 - slope * x1))
    let expMu = Foundation.exp(mu)
    var out: [Float] = []
    out.reserveCapacity(steps + 1)
    for i in 0...steps {
      let t = 1.0 - Float(i) / Float(steps)  // linspace 1 -> 0
      if t <= 0 {
        out.append(0)
      } else {
        let warped = expMu / (expMu + Foundation.pow(1.0 / t - 1.0, sigma))
        out.append(warped)
      }
    }
    return out
  }
}

// MARK: - Pipeline

public final class Krea2Pipeline {
  public let config: Krea2Config
  /// Where the weights came from — root, physical variant, transformer file.
  public let paths: Krea2ModelPaths
  /// The physical checkpoint variant this pipeline loaded (WP-E5, D7).
  public var variant: Krea2Variant { paths.variant }
  public let transformer: Krea2SingleStreamDiT
  public let textEncoder: Qwen3TextEncoder
  public let conditioner: Krea2TextConditioner
  public let vae: Krea2VAE

  private let logger = Logger(label: "z-image.krea2-pipeline")

  /// Currently applied LoRA configurations (for hot-swap tracking).
  private var appliedLoRAs: [LoRAConfiguration] = []

  /// Bare-parameter patch state (.diff/.diff_b/.set_weight — e.g. Kroma's 159
  /// norm/modulation deltas). Instance-scoped by construction: owns detached
  /// first-write-wins snapshots for this transformer only, restored on clear.
  private lazy var patchSession = LoRAPatchSession(module: transformer)

  /// Public accessor for currently loaded LoRA configurations.
  public var loadedLoRAConfigs: [LoRAConfiguration] { appliedLoRAs }

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
    public init(prompt: String, negativePrompt: String? = nil, guidance: Float = 1.0,
                width: Int = 1024, height: Int = 1024, steps: Int = 9, seed: UInt64 = 0,
                controlImagePixels: MLXArray? = nil, dyPE: DyPEConfig = .disabled) {
      self.prompt = prompt
      self.negativePrompt = negativePrompt
      self.guidance = guidance
      self.width = width
      self.height = height
      self.steps = steps
      self.seed = seed
      self.controlImagePixels = controlImagePixels
      self.dyPE = dyPE
    }
  }

  /// Whether a depth Control-LoRA is currently loaded (controlFirst set + A/B applied).
  public private(set) var controlLoRAActive = false

  public init(paths: Krea2ModelPaths, quantizeTransformer: Int? = nil) throws {
    self.config = Krea2Config()
    self.paths = paths

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

    let vae = Krea2VAE()
    try Krea2WeightLoader.loadVAE(vae, from: paths.vaeFile)
    self.vae = vae

    MLX.eval(transformer.parameters(), encoder.parameters(), vae.parameters())
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
    }

    guard !configs.isEmpty else { return }

    // Load and preflight-able failures (missing file, bad format, unknown
    // keys) all surface from loadForKrea2 BEFORE any weight mutation for
    // that config. If a later config fails after earlier ones applied, roll
    // the whole stack back so applied weights and `appliedLoRAs` can never
    // disagree (delta-key spec rev 2, Codex finding 2).
    do {
      for config in configs {
        let url = try await LoRAWeightLoader.resolveSource(config.source)
        let weights = try LoRAWeightLoader.loadForKrea2(from: url)
        logger.info("Applying Krea-2 LoRA: \(config.source.displayName) (rank=\(weights.rank), layers=\(weights.layerCount), deltas=\(weights.deltas.count), scale=\(config.scale))")
        LoRAApplicator.applyDynamically(to: transformer, loraWeights: weights, scale: config.scale, logger: logger)
        try patchSession.apply(weights: weights, scale: config.scale)
      }
    } catch {
      logger.error("Krea-2 LoRA stack failed mid-apply — rolling back to base: \(error)")
      LoRAApplicator.clearDynamicLoRA(from: transformer, logger: logger)
      patchSession.clear()
      appliedLoRAs = []
      throw error
    }

    appliedLoRAs = configs
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
        let weights = try LoRAWeightLoader.loadForKrea2(from: src)
        LoRAApplicator.applyDynamically(to: transformer, loraWeights: weights, scale: cfg.scale, logger: logger)
        try patchSession.apply(weights: weights, scale: cfg.scale)
      }
    } catch {
      // Same transactional posture as loadLoRAs: never leave weights and
      // tracking in disagreement (control state included).
      logger.error("identity-stack reapply failed — rolling back to base: \(error)")
      LoRAApplicator.clearDynamicLoRA(from: transformer, logger: logger)
      patchSession.clear()
      appliedLoRAs = []
      transformer.controlFirstWeight = nil
      transformer.controlFirstBias = nil
      controlLoRAActive = false
      throw error
    }

    guard let url else {
      transformer.controlFirstWeight = nil
      transformer.controlFirstBias = nil
      controlLoRAActive = false
      return
    }
    let cl = try Krea2ControlLoRA.load(from: url, layers: config.layers)
    // assertBaseHalfMatches skipped: transformer.first is q8-quantized (weight access unsafe on QuantizedLinear)
    let cw = cl.firstWeight
    let cb = cl.firstBias
    MLX.eval(cw, cb)
    transformer.controlFirstWeight = cw
    transformer.controlFirstBias = cb
    LoRAApplicator.applyDynamically(to: transformer, loraWeights: cl.loraWeights, scale: scale, logger: logger)
    controlLoRAActive = true
    logger.info("Krea-2 depth Control-LoRA active (scale=\(scale))")
  }

  /// Generate one image. Returns RGB float array (H, W, 3) in [0,1].
  public func generate(
    _ request: Request,
    progress: ((Int, Int) -> Void)? = nil
  ) -> MLXArray {
    let dtype = DType.bfloat16
    let patch = config.patch
    let comp = Krea2VAE.spatialScale  // 8
    let align = comp * patch          // 16
    let width = Krea2Sampling.roundUp(request.width, multiple: align)
    let height = Krea2Sampling.roundUp(request.height, multiple: align)

    let latH = height / comp, latW = width / comp
    let hTok = latH / patch, wTok = latW / patch

    // Noise in NCHW to match the reference RNG stream.
    MLXRandom.seed(request.seed)
    let noise = MLXRandom.normal([1, Krea2VAE.latentChannels, latH, latW]).asType(dtype)

    // Conditioning: (1, L, 12, 2560) + (1, L) mask.
    let (ctxRaw, mask) = conditioner.encode([request.prompt])
    let ctx = ctxRaw.asType(dtype)
    let txtLen = ctx.dim(1)

    var img = Krea2Sampling.patchify(noise, patch: patch)  // (1, hTok*wTok, 64)
    let pos = Krea2Sampling.buildPositions(txtLen: txtLen, h: hTok, w: wTok)
    let ropeScales = Krea2Sampling.ropeScales(
      hTok: hTok, wTok: wTok, patch: patch, dyPE: request.dyPE)
    let fullMask = MLX.concatenated([mask, MLX.ones([1, hTok * wTok])], axis: 1)

    // CFG branch (opt-in, Todd 2026-08-11): guidance > 1 encodes the negative
    // (or empty) prompt and runs a second unconditioned pass per step —
    // sequential, not batched, to keep peak memory flat on the shared box.
    let useCFG = request.guidance > 1.0
    var negCtx: MLXArray? = nil
    var negPos: MLXArray? = nil
    var negFullMask: MLXArray? = nil
    if useCFG {
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

    let x1 = Float((256 / align) * (256 / align))
    let x2 = Float((1280 / align) * (1280 / align))
    let ts = Krea2Sampling.timesteps(seqLen: img.dim(1), steps: request.steps, x1: x1, x2: x2)

    let total = ts.count - 1
    for i in 0..<total {
      let tc = ts[i], tp = ts[i + 1]
      let t = MLX.full([1], values: MLXArray(tc)).asType(dtype)
      let vCond = transformer(img: img, context: ctx, t: t, pos: pos, mask: fullMask,
                              control: controlTokens, ropeScales: ropeScales)
      let v: MLXArray
      if useCFG, let negCtx, let negPos, let negFullMask {
        let vUncond = transformer(img: img, context: negCtx, t: t, pos: negPos, mask: negFullMask,
                                  control: controlTokens, ropeScales: ropeScales)
        v = Krea2Sampling.applyCFG(cond: vCond, uncond: vUncond, scale: request.guidance)
      } else {
        v = vCond
      }
      img = img + (tp - tc) * v
      MLX.eval(img)
      progress?(i + 1, total)
    }

    let latentNCHW = Krea2Sampling.unpatchify(
      img, patch: patch, h: hTok, w: wTok, c: Krea2VAE.latentChannels)
    let latentNHWC = latentNCHW.transposed(0, 2, 3, 1).asType(.float32)
    let decoded = vae.decode(latentNHWC)  // (1, H, W, 3) in [0,1]
    MLX.eval(decoded)
    return decoded[0]
  }
}
