import Foundation
import MLX
import MLXNN

/// Offline MLX affine quantization for LTX-2 checkpoints (comfybox#230 Phase B).
///
/// Quantizes ONLY the video-DiT transformer-block projection weights
/// (attention to_q/to_k/to_v/to_out, FF proj_in/proj_out) to packed uint32 +
/// `.scales`/`.biases` siblings — the same MLX affine format the image
/// pipeline and the quantized Gemma loader already consume. Everything
/// quality-sensitive or small stays bf16: norms, patchify/time/caption
/// embeds, the final `proj_out`, scale-shift tables, the audio DiT branch,
/// the VAE, vocoder, and connectors. The output keeps the source's key
/// layout (monolith or per-component), so the existing load path works with
/// only a scales-aware branch added at load time.
public enum LTX2Quantizer {

  public struct Spec: Sendable {
    public var bits: Int
    public var groupSize: Int
    public init(bits: Int = 8, groupSize: Int = 64) {
      self.bits = bits
      self.groupSize = groupSize
    }
  }

  public struct Summary {
    public var quantizedCount = 0
    public var passthroughCount = 0
    public var quantizedInputBytes = 0
    public var quantizedOutputBytes = 0
  }

  public enum LTX2QuantizerError: Error, LocalizedError {
    case invalidBits(Int)
    case invalidGroupSize(Int)
    case sourceNotFound(String)
    case nothingQuantized(String)

    public var errorDescription: String? {
      switch self {
      case .invalidBits(let b): return "Invalid bits: \(b). Supported: 4, 8"
      case .invalidGroupSize(let g): return "Invalid group size: \(g). Supported: 32, 64, 128"
      case .sourceNotFound(let p): return "LTX-2 checkpoint not found: \(p)"
      case .nothingQuantized(let p):
        return "No quantizable transformer_blocks weights found in \(p) — wrong file or already quantized?"
      }
    }
  }

  /// Whether `key`/`tensor` is a video-DiT block projection we quantize.
  /// Works in both monolith (`model.diffusion_model.transformer_blocks.…`)
  /// and per-component (`transformer_blocks.…` / `transformer.…`) key spaces.
  public static func shouldQuantize(key: String, ndim: Int, inDim: Int, groupSize: Int) -> Bool {
    guard key.hasSuffix(".weight"), ndim == 2 else { return false }
    guard key.contains("transformer_blocks.") else { return false }
    // Audio / cross-modal branch stays bf16 — the video-only loader skips it,
    // and quantizing it would just complicate the future audio phase.
    if key.contains("audio_") || key.contains("av_ca_")
        || key.contains("audio_to_video_attn") || key.contains("video_to_audio_attn") {
      return false
    }
    if key.contains("norm") { return false }
    return inDim % groupSize == 0
  }

  /// Pure dict-level quantization — separated from file I/O for testability.
  /// Returns the transformed tensor dict (quantized keys replaced by packed
  /// uint32 + `.scales`/`.biases` siblings; everything else passed through)
  /// plus a summary. `evalEvery` bounds memory: quantized outputs are
  /// force-evaluated in batches so lazy graphs never accumulate.
  public static func quantizeTensors(
    _ weights: [String: MLXArray],
    spec: Spec,
    evalEvery: Int = 64,
    verbose: Bool = false
  ) -> (tensors: [String: MLXArray], summary: Summary) {
    var out: [String: MLXArray] = [:]
    var summary = Summary()
    var pendingEval: [MLXArray] = []

    for key in weights.keys.sorted() {
      let tensor = weights[key]!
      let inDim = tensor.ndim == 2 ? tensor.dim(1) : 0
      guard shouldQuantize(key: key, ndim: tensor.ndim, inDim: inDim, groupSize: spec.groupSize) else {
        out[key] = tensor
        summary.passthroughCount += 1
        continue
      }

      let base = String(key.dropLast(".weight".count))
      let f = tensor.dtype == .float32 ? tensor : tensor.asType(.float32)
      let (wq, scales, biases) = MLX.quantized(
        f, groupSize: spec.groupSize, bits: spec.bits, mode: .affine)

      out[key] = wq
      out["\(base).scales"] = scales.asType(.bfloat16)
      pendingEval.append(wq)
      pendingEval.append(out["\(base).scales"]!)
      if let b = biases {
        out["\(base).biases"] = b.asType(.bfloat16)
        pendingEval.append(out["\(base).biases"]!)
      }

      summary.quantizedCount += 1
      summary.quantizedInputBytes += tensor.shape.reduce(1, *) * tensor.dtype.size
      summary.quantizedOutputBytes += wq.shape.reduce(1, *) * 4

      if verbose {
        print("  quantized \(key) [\(tensor.dim(0))x\(inDim)] -> \(spec.bits)-bit g\(spec.groupSize)")
      }

      if pendingEval.count >= evalEvery {
        MLX.eval(pendingEval)
        pendingEval.removeAll(keepingCapacity: true)
        GPU.clearCache()
      }
    }

    if !pendingEval.isEmpty {
      MLX.eval(pendingEval)
      GPU.clearCache()
    }

    return (out, summary)
  }

  /// Quantize a checkpoint file (monolith or per-component transformer file)
  /// into `outputDir/<sourceName minus extension>-int<bits>.safetensors` plus a
  /// `quant_manifest.json` describing the spec and quantized layer names.
  @discardableResult
  public static func quantizeCheckpoint(
    source: URL,
    outputDir: URL,
    spec: Spec = Spec(),
    verbose: Bool = false
  ) throws -> Summary {
    guard [4, 8].contains(spec.bits) else { throw LTX2QuantizerError.invalidBits(spec.bits) }
    guard [32, 64, 128].contains(spec.groupSize) else {
      throw LTX2QuantizerError.invalidGroupSize(spec.groupSize)
    }
    let fm = FileManager.default
    let resolvedSource = source.resolvingSymlinksInPath()
    guard fm.fileExists(atPath: resolvedSource.path) else {
      throw LTX2QuantizerError.sourceNotFound(source.path)
    }
    try fm.createDirectory(at: outputDir, withIntermediateDirectories: true)

    print("Loading \(resolvedSource.lastPathComponent)…")
    let raw = try MLX.loadArrays(url: resolvedSource)
    print("  \(raw.count) tensors")

    let (out, summary) = quantizeTensors(raw, spec: spec, verbose: verbose)
    guard summary.quantizedCount > 0 else {
      throw LTX2QuantizerError.nothingQuantized(resolvedSource.path)
    }

    let stem = resolvedSource.deletingPathExtension().lastPathComponent
    let outputFile = outputDir.appendingPathComponent("\(stem)-int\(spec.bits).safetensors")
    print("Quantized \(summary.quantizedCount) block projections "
      + "(\(summary.quantizedInputBytes >> 30) GB -> \(summary.quantizedOutputBytes >> 30) GB packed); "
      + "\(summary.passthroughCount) tensors pass through unchanged.")
    print("Writing \(outputFile.path)…")
    try MLX.save(arrays: out, metadata: ["quantization": "mlx-affine-int\(spec.bits)-g\(spec.groupSize)"], url: outputFile)

    let manifest: [String: Any] = [
      "bits": spec.bits,
      "group_size": spec.groupSize,
      "mode": "affine",
      "source": resolvedSource.lastPathComponent,
      "quantized_layers": summary.quantizedCount,
    ]
    let manifestData = try JSONSerialization.data(
      withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
    try manifestData.write(to: outputDir.appendingPathComponent("quant_manifest.json"))
    print("Done.")
    return summary
  }

  /// Loader-side helper: convert the transformer's Linear layers to
  /// QuantizedLinear for exactly the layers whose sanitized weights carry
  /// `.scales` siblings. bits/groupSize are inferred per layer from shapes
  /// (groupSize = inDim / scalesCols, bits = 32 * wqCols / inDim), so a
  /// checkpoint is self-describing — no manifest required at load time.
  /// Returns the number of layers converted.
  @discardableResult
  public static func applyQuantizedLayout(
    to model: Module,
    sanitizedWeights: [String: MLXArray]
  ) -> Int {
    let scalesKeys = Set(sanitizedWeights.keys.filter { $0.hasSuffix(".scales") })
    guard !scalesKeys.isEmpty else { return 0 }
    var converted = 0
    MLXNN.quantize(model: model) { path, module in
      guard scalesKeys.contains("\(path).scales"),
            let wq = sanitizedWeights["\(path).weight"],
            let scales = sanitizedWeights["\(path).scales"],
            let linear = module as? Linear
      else { return nil }
      let inDim = linear.weight.dim(1)
      let scalesCols = scales.dim(1)
      let wqCols = wq.dim(1)
      guard scalesCols > 0, inDim % scalesCols == 0, (wqCols * 32) % inDim == 0 else { return nil }
      let groupSize = inDim / scalesCols
      let bits = wqCols * 32 / inDim
      guard [4, 8].contains(bits), [32, 64, 128].contains(groupSize) else { return nil }
      converted += 1
      return (groupSize, bits, .affine)
    }
    return converted
  }

  /// Sidecar suffixes recognised for FOREIGN (non-`LTX2Quantizer`) int8
  /// exports — comfybox#256. PinkCherry v1.7's int8 checkpoint stores raw
  /// (unpacked) `int8` `.weight` tensors plus an `F32 .weight_scale`
  /// sidecar (torch-style, symmetric, per-output-channel — see
  /// docs/HANDOFF-ltx-quality-2026-08-02.md §5). That is a different scheme
  /// from `LTX2Quantizer`'s own MLX affine group-wise `.scales`/`.biases`
  /// packed-uint32 output, and `applyQuantizedLayout` (which only looks for
  /// `.scales`) does not recognise it. `.scale_weight`/`.absmax` are
  /// accepted as aliases in case a differently-named export shows up;
  /// `.weight_scale` is the confirmed PinkCherry v1.7 name.
  static let foreignScaleSuffixes = [".weight_scale", ".scale_weight", ".absmax"]

  public enum ForeignQuantError: Error, LocalizedError, Equatable {
    /// An `int8` tensor was found with none of `foreignScaleSuffixes`
    /// present as a sibling (and it isn't `LTX2Quantizer`'s own packed
    /// `uint32` format, which is never `int8`). Loading it as-is would
    /// silently feed raw quantized bytes into a float weight parameter —
    /// the exact bug behind comfybox#256 (uniform noise output). Refuse
    /// instead of guessing.
    case missingScaleSidecar(String)

    public var errorDescription: String? {
      switch self {
      case .missingScaleSidecar(let key):
        let base = key.hasSuffix(".weight") ? String(key.dropLast(".weight".count)) : key
        let tried = LTX2Quantizer.foreignScaleSuffixes.map { "'\(base)\($0)'" }.joined(separator: ", ")
        return "LTX-2: '\(key)' is int8 with no recognised scale sidecar (looked for \(tried)) "
          + "— refusing to load raw int8 bytes as float weights, which renders as noise. "
          + "This checkpoint uses an unrecognised int8 export layout; if you know the real "
          + "sidecar name, that is what needs adding here."
      }
    }
  }

  /// Detects and dequantizes a FOREIGN int8 checkpoint layout (comfybox#256)
  /// — e.g. PinkCherry v1.7: raw `int8` `.weight` + `F32 .weight_scale`,
  /// symmetric per-output-channel (scale shape `[out, 1]` or `[out]`,
  /// broadcasting over the input dimension) — into plain dense `bfloat16`
  /// tensors. Per-channel granularity (one scale per output row) isn't
  /// representable in MLX's group-wise affine packed form without
  /// re-deriving group boundaries that were never part of this export, so
  /// this dequantizes to dense float rather than repacking into
  /// `QuantizedLinear`; the resulting Linear layers are functionally
  /// identical to loading an un-quantized bf16 checkpoint. Any `int8`
  /// tensor with no recognised sidecar throws `.missingScaleSidecar` naming
  /// the exact key, rather than silently loading raw bytes as float
  /// (the root cause of the noise bug). Non-`int8` tensors pass through
  /// unchanged. `LTX2Quantizer`'s own output (packed `uint32` + `.scales`)
  /// never has `int8`-dtype `.weight` tensors, so it is untouched by this
  /// function and continues through `applyQuantizedLayout` as before.
  public static func dequantizeForeignInt8Weights(
    _ weights: [String: MLXArray]
  ) throws -> [String: MLXArray] {
    var out = weights
    for key in weights.keys.sorted() {
      guard let tensor = weights[key], tensor.dtype == .int8 else { continue }
      guard key.hasSuffix(".weight") else {
        throw ForeignQuantError.missingScaleSidecar(key)
      }
      let base = String(key.dropLast(".weight".count))
      guard let scaleKey = foreignScaleSuffixes.map({ base + $0 }).first(where: { weights[$0] != nil })
      else {
        throw ForeignQuantError.missingScaleSidecar(key)
      }
      var scale = weights[scaleKey]!.asType(.float32)
      if scale.ndim == 1 {
        scale = scale.reshaped([scale.dim(0), 1])
      }
      let dense = tensor.asType(.float32) * scale
      out[key] = dense.asType(.bfloat16)
      out.removeValue(forKey: scaleKey)
    }
    return out
  }

  /// Dequantize one layer's packed weights back to a dense float tensor —
  /// used by the LoRA-merge path when the base checkpoint is quantized.
  public static func dequantizeLayer(
    base: String,
    weights: [String: MLXArray]
  ) -> (dense: MLXArray, groupSize: Int, bits: Int)? {
    guard let wq = weights["\(base).weight"],
          let scales = weights["\(base).scales"]
    else { return nil }
    let scalesCols = scales.dim(1)
    let wqCols = wq.dim(1)
    guard scalesCols > 0 else { return nil }
    // inDim = groupSize * scalesCols and inDim = wqCols*32/bits — solve for
    // the (groupSize, bits) pair among supported values.
    for bits in [8, 4] {
      let inDim = wqCols * 32 / bits
      if inDim % scalesCols == 0 {
        let groupSize = inDim / scalesCols
        if [32, 64, 128].contains(groupSize) {
          let dense = MLX.dequantized(
            wq, scales: scales, biases: weights["\(base).biases"],
            groupSize: groupSize, bits: bits, mode: .affine)
          return (dense, groupSize, bits)
        }
      }
    }
    return nil
  }
}
