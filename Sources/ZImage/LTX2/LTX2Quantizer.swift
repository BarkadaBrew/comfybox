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

  /// Checkpoints carrying `int8`-dtype weight tensors that are NOT
  /// `LTX2Quantizer`'s own MLX affine group-wise output (packed `uint32` +
  /// `.scales`/`.biases`, never `int8`) use some other quantization scheme
  /// ComfyBox does not implement. comfybox#256's first pass wrongly assumed
  /// PinkCherry v1.7 was plain per-channel `int8 × scale` (an `F32
  /// .weight_scale` sidecar IS present and correctly named — see
  /// docs/HANDOFF-ltx-quality-2026-08-02.md §5) and dequantized it that way.
  /// A follow-up review that opened the real checkpoint found 1,344
  /// `.comfy_quant` sidecar records declaring ComfyUI's `int8_tensorwise`
  /// format with `convrot: true`, `convrot_groupsize: 256` — a rotated
  /// coordinate basis (see `comfy/ops.py` around `QUANT_ALGOS`/`convrot` in
  /// a ComfyUI checkout). Plain `int8 × scale` reconstruction against that
  /// format measured cosine similarity ~0.008 against the matching bf16
  /// checkpoint — nowhere close to correct. ConvRot-aware dequantization is
  /// a real feature (tracked as a #256 follow-up), not a plain scale
  /// multiply, so this refuses to load ANY unrecognised int8 layout loudly
  /// instead of guessing.
  public enum UnsupportedInt8FormatError: Error, LocalizedError, Equatable {
    /// `key`'s `.comfy_quant` sidecar names a ComfyUI quantization format
    /// (`format`, e.g. "int8_tensorwise") this loader does not implement.
    /// `convrot`/`groupSize` are parsed from the record when present.
    case comfyQuantFormat(key: String, format: String, convrot: Bool, groupSize: Int)
    /// `key` is `int8` with a `.weight_scale` sidecar but no `.comfy_quant`
    /// record to identify which quantization scheme produced it — refused
    /// rather than assumed to be a plain per-channel scale (comfybox#256's
    /// mistake).
    case unidentifiedWeightScaleSidecar(key: String, sidecarKey: String)
    /// `key` is `int8` and ends in `.weight` but carries neither a
    /// `.comfy_quant` record nor a `.weight_scale` sidecar — no known
    /// layout applies at all.
    case noRecognisedSidecar(key: String)
    /// `key` is `int8` but isn't a `.weight` tensor (no sidecar naming
    /// convention applies to a role that isn't a weight matrix — e.g. a
    /// codebook, packed bias, or other quantization-scheme-specific
    /// tensor).
    case unsupportedInt8TensorRole(key: String)

    private static let seeIssue = "ComfyBox supports MLX affine int8 only (quantize-ltx2); see #256"

    public var errorDescription: String? {
      switch self {
      case .comfyQuantFormat(let key, let format, let convrot, let groupSize):
        return "unsupported quantized checkpoint format at '\(key)': ComfyUI \(format) "
          + "(convrot=\(convrot), group=\(groupSize)) — \(Self.seeIssue)"
      case .unidentifiedWeightScaleSidecar(let key, let sidecarKey):
        return "unsupported quantized checkpoint format at '\(key)': int8 weight with a "
          + "'\(sidecarKey)' sidecar but no '.comfy_quant' record to identify the scheme — \(Self.seeIssue)"
      case .noRecognisedSidecar(let key):
        return "unsupported quantized checkpoint format at '\(key)': int8 weight with no "
          + "'.comfy_quant' or '.weight_scale' sidecar of any kind — \(Self.seeIssue)"
      case .unsupportedInt8TensorRole(let key):
        return "unsupported quantized checkpoint format: '\(key)' is an int8 tensor that is not "
          + "a '.weight' (no known sidecar convention applies to this role) — \(Self.seeIssue)"
      }
    }
  }

  /// Parses a ComfyUI `.comfy_quant` sidecar — stored as a `uint8` tensor
  /// holding the raw UTF-8 bytes of a JSON object (`comfy/ops.py`:
  /// `torch.tensor(list(json.dumps(quant_conf).encode("utf-8")),
  /// dtype=torch.uint8)`) — into `(format, convrot, groupSize)`. Tolerant
  /// of the `convrot`/`convrot_groupsize` keys living either at the top
  /// level or nested under `"params"`; falls back to ComfyUI's own default
  /// group size (256) and `convrot: false` if the record doesn't parse.
  static func parseComfyQuantRecord(_ record: MLXArray) -> (format: String, convrot: Bool, groupSize: Int) {
    MLX.eval(record)
    let bytes = record.asType(.uint8).asArray(UInt8.self)
    guard let json = try? JSONSerialization.jsonObject(with: Data(bytes)) as? [String: Any] else {
      return ("unknown", false, 256)
    }
    let params = json["params"] as? [String: Any]
    let format = (json["format"] as? String) ?? "unknown"
    let convrot = (json["convrot"] as? Bool) ?? (params?["convrot"] as? Bool) ?? false
    let groupSize = (json["convrot_groupsize"] as? Int) ?? (params?["convrot_groupsize"] as? Int) ?? 256
    return (format, convrot, groupSize)
  }

  /// Fails loudly, before any weight is materialised, on any checkpoint
  /// tensor whose layout this loader does not implement (comfybox#256).
  /// `LTX2Quantizer`'s own output never has `int8`-dtype tensors (its
  /// packed form is `uint32`), so this is a no-op for checkpoints produced
  /// by `quantize-ltx2` — only foreign `int8` exports (ComfyUI's
  /// ConvRot-rotated PinkCherry included) are rejected here.
  public static func rejectUnsupportedInt8Weights(_ weights: [String: MLXArray]) throws {
    for key in weights.keys.sorted() {
      guard let tensor = weights[key], tensor.dtype == .int8 else { continue }
      guard key.hasSuffix(".weight") else {
        throw UnsupportedInt8FormatError.unsupportedInt8TensorRole(key: key)
      }
      let base = String(key.dropLast(".weight".count))
      if let record = weights["\(base).comfy_quant"] {
        let (format, convrot, groupSize) = parseComfyQuantRecord(record)
        throw UnsupportedInt8FormatError.comfyQuantFormat(
          key: key, format: format, convrot: convrot, groupSize: groupSize)
      }
      let sidecarKey = "\(base).weight_scale"
      if weights[sidecarKey] != nil {
        throw UnsupportedInt8FormatError.unidentifiedWeightScaleSidecar(key: key, sidecarKey: sidecarKey)
      }
      throw UnsupportedInt8FormatError.noRecognisedSidecar(key: key)
    }
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
