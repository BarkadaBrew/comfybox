// Krea2ControlLoRA.swift — Depth Control-LoRA for Krea-2 (port of
// Patil/Krea-2-depth-controlnet · Tanmaypatil123/Krea-2-controlnet).
//
// Mechanism (see docs/FDD-krea2-depth-controlnet.md): a rank-64 LoRA on all 28
// DiT blocks (targets attn.{wq,wk,wv,wo,gate}, mlp.{gate,up,down}) PLUS an
// expanded input projection. A depth map is Qwen-VAE-encoded to a 16-ch latent,
// patchified to 64-dim control tokens, concatenated channel-wise to the noisy
// image tokens (64 -> 128), and projected by a dedicated `controlFirst` linear
// loaded directly from the checkpoint's `first.weight [6144,128]` / `first.bias`
// (gated option-A: control OFF path is unchanged base `first`, byte-identical).
//
// Checkpoint (450 tensors, F32): `first.weight` [features, 2C], `first.bias`
// [features]; per block i, `blocks.{i}.{target}.A` [rank,in], `.B` [out,rank].

import Foundation
import MLX
import MLXNN
import Logging

public struct Krea2ControlLoRA {
  /// Expanded input projection weight: (features, 2*C) — first half = base, second = control.
  public let firstWeight: MLXArray
  /// Expanded input projection bias: (features,).
  public let firstBias: MLXArray
  /// Rank-64 LoRA A/B pairs keyed `blocks.{i}.{target}.weight` for LoRAApplicator.
  public let loraWeights: LoRAWeights
  /// Base per-token input width C (= 2C/2); expected 64 for Krea-2.
  public let baseInFeatures: Int

  private static let targets = [
    "attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
    "mlp.gate", "mlp.up", "mlp.down",
  ]

  /// Parse `depth-control-lora.safetensors`. Fails loud on any missing key/shape
  /// (deedee-symlink lesson: never silently no-op).
  public static func load(from file: URL, layers: Int = 28, dtype: DType = .bfloat16) throws -> Krea2ControlLoRA {
    guard FileManager.default.fileExists(atPath: file.path) else {
      throw Krea2WeightLoaderError.missingFile(file.path)
    }
    let reader = try SafeTensorsReader(fileURL: file)
    let names = Set(reader.allMetadata().map { $0.name })

    func tensor(_ name: String) throws -> MLXArray {
      guard names.contains(name) else {
        throw Krea2WeightLoaderError.missingFile("control-lora key: \(name)")
      }
      var arr = try reader.tensor(named: name)
      if arr.dtype != dtype { arr = arr.asType(dtype) }
      return arr
    }

    let fw = try tensor("first.weight")   // (features, 2C)
    let fb = try tensor("first.bias")     // (features,)
    precondition(fw.ndim == 2, "control-lora first.weight must be 2-D, got \(fw.shape)")
    let twoC = fw.dim(1)
    precondition(twoC % 2 == 0, "control-lora first.weight in-features must be even (2C), got \(twoC)")
    let baseIn = twoC / 2

    var pairs: [String: (down: MLXArray, up: MLXArray)] = [:]
    var rank = 0
    for i in 0..<layers {
      for tg in Self.targets {
        let a = try tensor("blocks.\(i).\(tg).A")   // (rank, in)
        let b = try tensor("blocks.\(i).\(tg).B")   // (out, rank)
        precondition(a.dim(0) == b.dim(1),
          "control-lora rank mismatch at blocks.\(i).\(tg): A\(a.shape) B\(b.shape)")
        rank = a.dim(0)
        pairs["blocks.\(i).\(tg).weight"] = (down: a, up: b)
      }
    }
    // comfybox#329 M2: SURFACE any LoKr tensors the checkpoint carries. This
    // loader fetches only the fixed A/B key set, so without this scan a
    // `.lokr_w1/.lokr_w2` half would silently vanish and the pipeline's
    // transactional guard (`Krea2AdapterSupport.checkTransactional`, which
    // reads `lokrLayerCount`) would have nothing to refuse on. The refusal
    // POLICY stays in the pipeline, same as the identity-stack sites; this
    // only makes the count truthful. Orphan halves are a loud invalidFormat,
    // mirroring `loadForKrea2` (never a silent drop).
    var lokrWeights: [String: LoKrWeights] = [:]
    let lokrHalves = names.filter { $0.hasSuffix(".lokr_w1") || $0.hasSuffix(".lokr_w2") }
    if !lokrHalves.isEmpty {
      // ".lokr_w1" and ".lokr_w2" are the same length, so one dropLast fits both.
      let modules = Set(lokrHalves.map { String($0.dropLast(".lokr_w1".count)) })
      for module in modules.sorted() {
        guard names.contains(module + ".lokr_w1"), names.contains(module + ".lokr_w2") else {
          throw LoRAError.invalidFormat(
            "orphan LoKr half at '\(module)' in control LoRA \(file.lastPathComponent)")
        }
        lokrWeights[module] = LoKrWeights(
          w1: try reader.tensor(named: module + ".lokr_w1"),
          w2: try reader.tensor(named: module + ".lokr_w2"))
      }
    }
    // alpha == nil -> alpha defaults to rank -> effectiveScale 1.0 (fixed-strength control LoRA).
    let lw = LoRAWeights(weights: pairs, lokrWeights: lokrWeights, rank: rank, alpha: nil)
    return Krea2ControlLoRA(firstWeight: fw, firstBias: fb, loraWeights: lw, baseInFeatures: baseIn)
  }

  /// Build the gated `controlFirst` Linear(2C -> features) from the checkpoint.
  /// Excluded from q8 quantization by construction (loaded fp/bf16, applied after
  /// init-time quantize). Bias from the file is used directly (fixes the option-B
  /// bias bug flagged in review).
  public func makeControlFirst() -> Linear {
    let outF = firstWeight.dim(0)
    let inF = firstWeight.dim(1)
    let lin = Linear(inF, outF, bias: true)
    try? lin.update(parameters: ModuleParameters.unflattened([
      "weight": firstWeight,
      "bias": firstBias,
    ]), verify: [.shapeMismatch])
    return lin
  }

  /// Assert the checkpoint's base-half projection matches the live base `first`
  /// weight (catches base/checkpoint drift; the LoRA was trained on Krea-2-Raw).
  /// Tolerance is loose because the live base may be q8-quantized.
  public func assertBaseHalfMatches(baseFirstWeight: MLXArray, logger: Logger? = nil) {
    let baseHalf = firstWeight[0..., 0 ..< baseInFeatures]  // (features, C)
    guard baseHalf.shape == baseFirstWeight.shape else {
      logger?.warning("control-lora base-half shape \(baseHalf.shape) != base first \(baseFirstWeight.shape)")
      return
    }
    let diff = MLX.mean(MLX.abs(baseHalf.asType(.float32) - baseFirstWeight.asType(.float32))).item(Float.self)
    if diff > 0.05 {
      logger?.warning("control-lora base-half diverges from live base first (mean|Δ|=\(diff)); check base model / quantization")
    } else {
      logger?.info("control-lora base-half matches live base first (mean|Δ|=\(diff))")
    }
  }
}
