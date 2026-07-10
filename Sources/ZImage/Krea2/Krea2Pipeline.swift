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
  public var transformerFile: URL { root.appending(path: "turbo.safetensors") }
  public var textEncoderFile: URL { root.appending(path: "text_encoder/model.safetensors") }
  public var vaeFile: URL { root.appending(path: "vae/diffusion_pytorch_model.safetensors") }
  public var tokenizerDirectory: URL { root.appending(path: "tokenizer") }

  public init(root: URL) { self.root = root }

  /// Locate the newest HF-cache snapshot of krea/Krea-2-Turbo, or use an explicit dir.
  public static func resolve(modelDir: String? = nil) throws -> Krea2ModelPaths {
    if let modelDir {
      return Krea2ModelPaths(root: URL(fileURLWithPath: (modelDir as NSString).expandingTildeInPath))
    }
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
        return Krea2ModelPaths(root: candidate)
      }
    }
    throw Krea2WeightLoaderError.missingFile("\(snapshots)/*/turbo.safetensors")
  }
}

// MARK: - Sampling math (port of sampling.py)

enum Krea2Sampling {
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
  public let transformer: Krea2SingleStreamDiT
  public let textEncoder: Qwen3TextEncoder
  public let conditioner: Krea2TextConditioner
  public let vae: Krea2VAE

  private let logger = Logger(label: "z-image.krea2-pipeline")

  /// Currently applied LoRA configurations (for hot-swap tracking).
  private var appliedLoRAs: [LoRAConfiguration] = []

  /// Public accessor for currently loaded LoRA configurations.
  public var loadedLoRAConfigs: [LoRAConfiguration] { appliedLoRAs }

  public struct Request {
    public var prompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var seed: UInt64
    public init(prompt: String, width: Int = 1024, height: Int = 1024, steps: Int = 9, seed: UInt64 = 0) {
      self.prompt = prompt
      self.width = width
      self.height = height
      self.steps = steps
      self.seed = seed
    }
  }

  public init(paths: Krea2ModelPaths, quantizeTransformer: Int? = nil) throws {
    self.config = Krea2Config()

    let transformer = Krea2SingleStreamDiT(cfg: config)
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
    if !appliedLoRAs.isEmpty {
      LoRAApplicator.clearDynamicLoRA(from: transformer, logger: logger)
      appliedLoRAs = []
    }

    guard !configs.isEmpty else { return }

    for config in configs {
      let url = try await LoRAWeightLoader.resolveSource(config.source)
      let weights = try LoRAWeightLoader.loadForKrea2(from: url)
      logger.info("Applying Krea-2 LoRA: \(config.source.displayName) (rank=\(weights.rank), layers=\(weights.layerCount), scale=\(config.scale))")
      LoRAApplicator.applyDynamically(to: transformer, loraWeights: weights, scale: config.scale, logger: logger)
    }

    appliedLoRAs = configs
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
    let fullMask = MLX.concatenated([mask, MLX.ones([1, hTok * wTok])], axis: 1)

    let x1 = Float((256 / align) * (256 / align))
    let x2 = Float((1280 / align) * (1280 / align))
    let ts = Krea2Sampling.timesteps(seqLen: img.dim(1), steps: request.steps, x1: x1, x2: x2)

    let total = ts.count - 1
    for i in 0..<total {
      let tc = ts[i], tp = ts[i + 1]
      let t = MLX.full([1], values: MLXArray(tc)).asType(dtype)
      let v = transformer(img: img, context: ctx, t: t, pos: pos, mask: fullMask)
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
