import Foundation
import MLX
import Hub

public final class LoRAWeightLoader {

    private static let loraPatterns: [(down: String, up: String)] = [
        (".lora_down.", ".lora_up."),
        (".lora_A.", ".lora_B.")
    ]
    private enum LoKrSuffix: String {
        case w1 = ".lokr_w1"
        case w2 = ".lokr_w2"
        case alpha = ".alpha"
    }

    private static let prefixesToRemove = [
        "base_model.model.",
        "diffusion_model.",
        "lora_unet_",
        "lora_te_",
        "transformer.",
        "text_encoder."
    ]

    public static func load(from config: LoRAConfiguration) async throws -> LoRAWeights {
        let url = try await resolveSource(config.source)
        // Use a detached task to avoid Swift concurrency deadlock:
        // When called from an actor-isolated method, the synchronous
        // load(from: url) can block the cooperative pool and prevent
        // the actor from resuming. A detached task runs independently.
        let weights = try await Task.detached(priority: .userInitiated) {
            try load(from: url)
        }.value
        return weights
    }

    public static func load(from url: URL) throws -> LoRAWeights {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw LoRAError.fileNotFound(url.path)
        }

        let safetensorFiles: [URL]
        let configDirectory: URL
        if isDirectory.boolValue {
            safetensorFiles = try findSafetensorFiles(in: url)
            configDirectory = url
        } else {
            safetensorFiles = [url]
            configDirectory = url.deletingLastPathComponent()
        }

        var loraWeights: [String: (down: MLXArray, up: MLXArray)] = [:]
        var lokrW1: [String: MLXArray] = [:]
        var lokrW2: [String: MLXArray] = [:]
        var moduleAlphas: [String: Float] = [:]
        var networkAlpha: Float?
        var deltas: [String: DeltaPatch] = [:]

        for fileURL in safetensorFiles {
            let partial = try loadSafetensorFile(fileURL)
            for (k, v) in partial.loraPairs { loraWeights[k] = v }
            for (k, v) in partial.lokrW1 { lokrW1[k] = v }
            for (k, v) in partial.lokrW2 { lokrW2[k] = v }
            for (k, v) in partial.moduleAlphas { moduleAlphas[k] = v }
            for (k, v) in partial.deltas { deltas[k] = v }
            if networkAlpha == nil { networkAlpha = partial.networkAlpha }
        }

        var lokrWeights: [String: LoKrWeights] = [:]
        lokrWeights.reserveCapacity(min(lokrW1.count, lokrW2.count))
        for (key, w1) in lokrW1 {
            guard let w2 = lokrW2[key] else { continue }
            lokrWeights[key] = LoKrWeights(w1: w1, w2: w2, alpha: moduleAlphas[key])
        }

        // Kohya-style `<module>.alpha` tensors also accompany standard
        // lora_down/lora_up pairs; attach them per layer so scaling can use
        // alpha / rank instead of the default 1.0.
        var layerAlphas: [String: Float] = [:]
        for (key, value) in moduleAlphas {
            let weightKey = key + ".weight"
            if loraWeights[weightKey] != nil {
                layerAlphas[weightKey] = value
            } else if loraWeights[key] != nil {
                layerAlphas[key] = value
            }
        }

        guard !loraWeights.isEmpty || !lokrWeights.isEmpty || !deltas.isEmpty else {
            throw LoRAError.invalidFormat("No valid LoRA weights found. Expected keys with .lora_down/.lora_up, .lora_A/.lora_B, LyCORIS LoKr (.lokr_w1/.lokr_w2), or bare patches (.diff/.diff_b/.set_weight).")
        }

        let rank = inferRank(from: loraWeights)
        let alpha = loadAlpha(from: configDirectory) ?? networkAlpha

        return LoRAWeights(
            weights: loraWeights,
            lokrWeights: lokrWeights,
            rank: rank,
            alpha: alpha,
            layerAlphas: layerAlphas,
            deltas: deltas
        )
    }

    public static func resolveSource(_ source: LoRASource) async throws -> URL {
        switch source {
        case .local(let url):
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw LoRAError.fileNotFound(url.path)
            }
            return url

        case .huggingFace(let modelId, let filename):
            return try await downloadFromHuggingFace(modelId: modelId, filename: filename)
        }
    }

    public static func validate(at url: URL) throws -> LoRAValidationResult {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return .invalid("File not found: \(url.path)")
        }

        do {
            let reader = try SafeTensorsReader(fileURL: url)
            let keys = reader.tensorNames

            let hasDownWeights = keys.contains { key in loraPatterns.contains { key.contains($0.down) } }
            let hasUpWeights = keys.contains { key in loraPatterns.contains { key.contains($0.up) } }
            let hasLoKr = keys.contains { $0.hasSuffix(LoKrSuffix.w1.rawValue) || $0.hasSuffix(LoKrSuffix.w2.rawValue) }

            guard (hasDownWeights && hasUpWeights) || hasLoKr else {
                return .invalid("Not a valid LoRA file: missing lora_down/lora_up weight pairs or lokr_w1/lokr_w2 tensors")
            }

            var targetLayers: [String] = []
            var rank = 0

            for key in keys {
                let matchedDownPattern = loraPatterns.first { key.contains($0.down) }?.down
                guard let downPattern = matchedDownPattern,
                      let tensor = try? reader.tensor(named: key) else { continue }

                let layerName = extractBaseKey(key, pattern: downPattern) ?? key
                targetLayers.append(layerName)

                if tensor.ndim == 2 {
                    rank = max(rank, min(tensor.dim(0), tensor.dim(1)))
                }
            }

            let estimatedMemoryMB = (rank * 3840 * 2 * 4 * targetLayers.count) / (1024 * 1024)

            return LoRAValidationResult(
                isValid: true,
                rank: rank,
                targetLayers: targetLayers,
                estimatedMemoryMB: estimatedMemoryMB
            )
        } catch {
            return .invalid("Failed to read LoRA file: \(error.localizedDescription)")
        }
    }

    /// Bare suffix form used by the ComfyUI ecosystem (e.g. Kroma):
    /// `...lora_A` / `...lora_B` with no trailing `.weight`.
    private static let loraSuffixPatterns: [(down: String, up: String)] = [
        (".lora_down", ".lora_up"),
        (".lora_A", ".lora_B")
    ]

    private static func resolveKeyPair(_ key: String) -> (downKey: String, upKey: String, baseKey: String)? {
        for (downPattern, upPattern) in loraPatterns {
            if key.contains(downPattern) {
                guard let base = extractBaseKey(key, pattern: downPattern) else { return nil }
                let upKey = key.replacingOccurrences(of: downPattern, with: upPattern)
                return (key, upKey, base)
            } else if key.contains(upPattern) {
                guard let base = extractBaseKey(key, pattern: upPattern) else { return nil }
                let downKey = key.replacingOccurrences(of: upPattern, with: downPattern)
                return (downKey, key, base)
            }
        }
        for (downSuffix, upSuffix) in loraSuffixPatterns {
            if key.hasSuffix(downSuffix) {
                let base = stripKnownPrefixes(String(key.dropLast(downSuffix.count)))
                return (key, String(key.dropLast(downSuffix.count)) + upSuffix, base)
            } else if key.hasSuffix(upSuffix) {
                let base = stripKnownPrefixes(String(key.dropLast(upSuffix.count)))
                return (String(key.dropLast(upSuffix.count)) + downSuffix, key, base)
            }
        }
        return nil
    }

    private static func extractBaseKey(_ key: String, pattern: String) -> String? {
        guard let range = key.range(of: pattern) else { return nil }
        var base = String(key[..<range.lowerBound])

        for prefix in prefixesToRemove {
            if base.hasPrefix(prefix) {
                base = String(base.dropFirst(prefix.count))
                break
            }
        }

        return base
    }

    private static func inferRank(from weights: [String: (down: MLXArray, up: MLXArray)]) -> Int {
        // Iterate keys in sorted order so mixed-rank adapters produce a
        // deterministic result (dictionary order is unspecified). Per-layer
        // ranks are still preferred at application time via
        // LoRAWeights.effectiveScale(forLayer:).
        for key in weights.keys.sorted() {
            guard let pair = weights[key] else { continue }
            let downShape = pair.down.shape
            if downShape.count == 2 {
                return min(downShape[0], downShape[1])
            }
        }
        return 16
    }

    private static func loadAlpha(from directory: URL) -> Float? {
        let configPath = directory.appendingPathComponent("adapter_config.json")

        guard FileManager.default.fileExists(atPath: configPath.path),
              let data = try? Data(contentsOf: configPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let alpha = json["lora_alpha"] as? NSNumber else {
            return nil
        }

        return alpha.floatValue
    }

    private static func downloadFromHuggingFace(modelId: String, filename: String?) async throws -> URL {
        let filePatterns = filename.map { [$0] } ?? ["*.safetensors"]

        do {
            let snapshotURL = try await ModelResolution.resolve(
                modelSpec: modelId,
                filePatterns: filePatterns
            )

            let fm = FileManager.default
            let contents = try fm.contentsOfDirectory(at: snapshotURL, includingPropertiesForKeys: nil)

            if let filename = filename {
                let targetURL = snapshotURL.appendingPathComponent(filename)
                if fm.fileExists(atPath: targetURL.path) {
                    return targetURL
                }
            }

            if let safetensorFile = contents.first(where: { $0.pathExtension == "safetensors" }) {
                return safetensorFile
            }

            throw LoRAError.noSafetensorsFound(snapshotURL)
        } catch let error as LoRAError {
            throw error
        } catch {
            throw LoRAError.downloadFailed(modelId, error)
        }
    }

    private struct PartialLoRAWeights {
        let loraPairs: [String: (down: MLXArray, up: MLXArray)]
        let lokrW1: [String: MLXArray]
        let lokrW2: [String: MLXArray]
        /// Kohya/LyCORIS `<module>.alpha` scalars, used both for LoKr and
        /// for per-layer scaling of standard lora_down/lora_up pairs.
        let moduleAlphas: [String: Float]
        /// Adapter-wide alpha from `ss_network_alpha` file metadata, if any.
        let networkAlpha: Float?
        /// Bare-parameter patches keyed by target parameter path.
        let deltas: [String: DeltaPatch]
    }

    /// Adapter features we recognise but deliberately do not support.
    /// A file carrying one of these must FAIL to load with the feature named
    /// (spec rev 2, Codex finding 6) — explicit refusal beats silently
    /// applying a fraction of the adapter.
    private static let unsupportedSuffixes: [(suffix: String, feature: String)] = [
        (".dora_scale", "DoRA (dora_scale)"),
        (".lora_mid.weight", "LoCon mid blocks (lora_mid)"),
        (".w_norm", "norm-magnitude adapters (w_norm)"),
        (".b_norm", "norm-magnitude adapters (b_norm)"),
        (".reshape_weight", "weight reshaping (reshape_weight)"),
    ]

    /// Text-encoder halves of composite adapters: loadable transformer-side,
    /// so their keys are skipped (reported via log, never fatal, never bound).
    private static let outOfScopePrefixes = ["lora_te_", "text_encoder.", "te_"]

    private static func loadSafetensorFile(_ url: URL) throws -> PartialLoRAWeights {
        let reader = try SafeTensorsReader(fileURL: url)
        let keys = reader.tensorNames

        var consumedKeys = Set<String>()
        var loraPairs: [String: (down: MLXArray, up: MLXArray)] = [:]
        var lokrW1: [String: MLXArray] = [:]
        var lokrW2: [String: MLXArray] = [:]
        var moduleAlphas: [String: Float] = [:]
        var deltas: [String: DeltaPatch] = [:]

        for key in keys {
            if consumedKeys.contains(key) { continue }

            // Known-but-unsupported features fail loudly, naming the feature.
            if let match = unsupportedSuffixes.first(where: { key.hasSuffix($0.suffix) }) {
                throw LoRAError.unsupportedFeature("\(match.feature) — key: \(key)")
            }

            if let (moduleKey, suffix) = mapLoKrModuleKey(key) {
                switch suffix {
                case .w1:
                    lokrW1[moduleKey] = try reader.tensor(named: key)
                case .w2:
                    lokrW2[moduleKey] = try reader.tensor(named: key)
                case .alpha:
                    let tensor = try reader.tensor(named: key)
                    if let value = tensor.asArray(Float.self).first {
                        moduleAlphas[moduleKey] = value
                    }
                }
                consumedKeys.insert(key)
                continue
            }

            // Bare-parameter patches (ComfyUI lora.py semantics). `.diff_b`
            // maps onto the target's REAL bias parameter path.
            if key.hasSuffix(".diff_b") {
                let base = stripKnownPrefixes(String(key.dropLast(".diff_b".count)))
                deltas[base + ".bias"] = .diffBias(try reader.tensor(named: key))
                consumedKeys.insert(key)
                continue
            }
            if key.hasSuffix(".diff") {
                let base = stripKnownPrefixes(String(key.dropLast(".diff".count)))
                deltas[base] = .diff(try reader.tensor(named: key))
                consumedKeys.insert(key)
                continue
            }
            if key.hasSuffix(".set_weight") {
                let base = stripKnownPrefixes(String(key.dropLast(".set_weight".count)))
                deltas[base] = .setWeight(try reader.tensor(named: key))
                consumedKeys.insert(key)
                continue
            }

            if let (downKey, upKey, baseKey) = resolveKeyPair(key) {
                guard reader.contains(downKey), reader.contains(upKey) else {
                    // Orphan half of a pair: a bindable key that can never
                    // bind. Silent-skipping this is exactly the partial-
                    // application family of bug.
                    throw LoRAError.invalidFormat(
                        "orphan LoRA pair half '\(key)' — counterpart missing")
                }

                let downWeight = try reader.tensor(named: downKey)
                let upWeight = try reader.tensor(named: upKey)

                let mappedKey = LoRAKeyMapper.mapToZImageKey(baseKey)
                loraPairs[mappedKey] = (down: downWeight, up: upWeight)

                consumedKeys.insert(downKey)
                consumedKeys.insert(upKey)
                continue
            }
        }

        // Classify the leftovers: out-of-scope prefixes are skipped
        // non-fatally; anything else is an unknown key and a load ERROR so a
        // partial application can never be silent (spec rev 2).
        var unknown: [String] = []
        for key in keys where !consumedKeys.contains(key) {
            if outOfScopePrefixes.contains(where: { key.hasPrefix($0) }) { continue }
            unknown.append(key)
        }
        guard unknown.isEmpty else { throw LoRAError.unknownKeys(unknown) }

        return PartialLoRAWeights(
            loraPairs: loraPairs,
            lokrW1: lokrW1,
            lokrW2: lokrW2,
            moduleAlphas: moduleAlphas,
            networkAlpha: reader.fileMetadata["ss_network_alpha"].flatMap(Float.init),
            deltas: deltas
        )
    }

    private static func stripKnownPrefixes(_ key: String) -> String {
        for prefix in prefixesToRemove where key.hasPrefix(prefix) {
            return String(key.dropFirst(prefix.count))
        }
        return key
    }

    private static func findSafetensorFiles(in directory: URL) throws -> [URL] {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey]) else {
            throw LoRAError.noSafetensorsFound(directory)
        }

        var results: [URL] = []
        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                results.append(url)
            }
        }
        if results.isEmpty {
            throw LoRAError.noSafetensorsFound(directory)
        }
        return results.sorted(by: { $0.path < $1.path })
    }

    // MARK: - Flux 2 LoRA Loading

    /// Load a safetensors LoRA file and map keys for the Flux2Transformer.
    ///
    /// Uses ``Flux2LoRAMapping`` instead of ``LoRAKeyMapper`` to map adapter keys
    /// to Flux2Transformer module paths. For fused QKV keys, lora_B is split into
    /// 3 equal parts along dim 0, creating separate Q/K/V entries.
    ///
    /// - Parameter url: Path to a `.safetensors` file.
    /// - Returns: LoRAWeights with keys matching Flux2Transformer module paths.
    public static func loadForFlux2(from url: URL) throws -> LoRAWeights {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoRAError.fileNotFound(url.path)
        }

        let reader = try SafeTensorsReader(fileURL: url)
        let keys = reader.tensorNames

        var processedKeys = Set<String>()
        var loraPairs: [String: (down: MLXArray, up: MLXArray)] = [:]

        for key in keys {
            if processedKeys.contains(key) { continue }

            // Find lora_A/lora_B or lora_down/lora_up pair
            guard let (downKey, upKey, baseKey) = resolveKeyPair(key) else { continue }
            guard reader.contains(downKey), reader.contains(upKey) else { continue }

            let downWeight = try reader.tensor(named: downKey)  // lora_A
            let upWeight = try reader.tensor(named: upKey)      // lora_B

            processedKeys.insert(downKey)
            processedKeys.insert(upKey)

            // Map using Flux2LoRAMapping instead of LoRAKeyMapper
            let mapping = Flux2LoRAMapping.map(baseKey)

            switch mapping {
            case .direct(let targetPath):
                loraPairs[targetPath] = (down: downWeight, up: upWeight)

            case .qkvSplit(let target):
                // Split lora_B (up) into 3 parts along dim 0
                guard upWeight.ndim == 2 else { continue }
                let totalOut = upWeight.dim(0)
                guard totalOut % 3 == 0 else { continue }
                let splitSize = totalOut / 3

                let qUp = upWeight[0..<splitSize, 0...]
                let kUp = upWeight[splitSize..<(2 * splitSize), 0...]
                let vUp = upWeight[(2 * splitSize)..<(3 * splitSize), 0...]

                // lora_A (down): use the full tensor for all 3 projections.
                // The rank dimension is shared across Q/K/V.
                loraPairs[target.qPath] = (down: downWeight, up: qUp)
                loraPairs[target.kPath] = (down: downWeight, up: kUp)
                loraPairs[target.vPath] = (down: downWeight, up: vUp)

            case .unmapped:
                // Skip keys we don't recognize
                continue
            }
        }

        guard !loraPairs.isEmpty else {
            throw LoRAError.invalidFormat(
                "No valid Flux 2 LoRA weight pairs found in \(url.lastPathComponent). " +
                "Expected keys matching double_blocks/single_blocks patterns."
            )
        }

        let rank = inferRank(from: loraPairs)
        let alpha = loadAlpha(from: url.deletingLastPathComponent())

        return LoRAWeights(weights: loraPairs, rank: rank, alpha: alpha)
    }

    // MARK: - Krea-2 LoRA Loading

  /// Load a safetensors LoRA file for the Krea-2 `SingleStreamDiT` transformer.
  ///
  /// Krea-2 LoRAs (e.g. from the `krea2` training pipeline) use
  /// `diffusion_model.blocks.<n>.<attn|mlp>.<proj>.lora_A/lora_B` (or LyCORIS
  /// `.lokr_w1/.lokr_w2/.alpha`) keys that already match `Krea2SingleStreamDiT`'s
  /// module paths 1:1 once the `diffusion_model.` prefix is stripped — unlike
  /// Z-Image, no block/component remapping is needed, so this bypasses
  /// ``LoRAKeyMapper`` entirely (same reasoning as ``loadForFlux2(from:)``).
  ///
  /// - Parameter url: Path to a `.safetensors` file.
  /// - Returns: LoRAWeights with keys matching Krea2SingleStreamDiT module paths.
  public static func loadForKrea2(from url: URL) throws -> LoRAWeights {
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw LoRAError.fileNotFound(url.path)
    }

    let reader = try SafeTensorsReader(fileURL: url)
    let keys = reader.tensorNames

    var consumedKeys = Set<String>()
    var loraPairs: [String: (down: MLXArray, up: MLXArray)] = [:]
    var lokrW1: [String: MLXArray] = [:]
    var lokrW2: [String: MLXArray] = [:]
    var moduleAlphas: [String: Float] = [:]
    var deltas: [String: DeltaPatch] = [:]

    for key in keys {
      if consumedKeys.contains(key) { continue }

      if let match = unsupportedSuffixes.first(where: { key.hasSuffix($0.suffix) }) {
        throw LoRAError.unsupportedFeature("\(match.feature) — key: \(key)")
      }

      if let (moduleKey, suffix) = mapKrea2LoKrModuleKey(key) {
        switch suffix {
        case .w1:
          lokrW1[moduleKey] = try reader.tensor(named: key)
        case .w2:
          lokrW2[moduleKey] = try reader.tensor(named: key)
        case .alpha:
          let tensor = try reader.tensor(named: key)
          if let value = tensor.asArray(Float.self).first {
            moduleAlphas[moduleKey] = value
          }
        }
        consumedKeys.insert(key)
        continue
      }

      // Bare-parameter patches — Kroma ships 159 of these on norm/modulation
      // params. Keys match Krea2SingleStreamDiT paths after the
      // `diffusion_model.` strip plus the same numeric-index remap the base
      // weight loader uses (txtmlp.0 → txtmlp.norm0, tmlp.0 → tmlp.lin0, …).
      if key.hasSuffix(".diff_b") {
        let base = remapKrea2Base(String(key.dropLast(".diff_b".count)))
        deltas[base + ".bias"] = .diffBias(try reader.tensor(named: key))
        consumedKeys.insert(key)
        continue
      }
      if key.hasSuffix(".diff") {
        let base = remapKrea2Base(String(key.dropLast(".diff".count)))
        deltas[base] = .diff(try reader.tensor(named: key))
        consumedKeys.insert(key)
        continue
      }
      if key.hasSuffix(".set_weight") {
        let base = remapKrea2Base(String(key.dropLast(".set_weight".count)))
        deltas[base] = .setWeight(try reader.tensor(named: key))
        consumedKeys.insert(key)
        continue
      }

      if let (downKey, upKey, baseKey) = resolveKeyPair(key) {
        guard reader.contains(downKey), reader.contains(upKey) else {
          throw LoRAError.invalidFormat(
            "orphan LoRA pair half '\(key)' — counterpart missing")
        }

        let downWeight = try reader.tensor(named: downKey)
        let upWeight = try reader.tensor(named: upKey)

        consumedKeys.insert(downKey)
        consumedKeys.insert(upKey)

        let mappedBase = remapKrea2Base(baseKey)
        let targetKey = mappedBase.hasSuffix(".weight") ? mappedBase : mappedBase + ".weight"
        loraPairs[targetKey] = (down: downWeight, up: upWeight)
        continue
      }
    }

    var unknown: [String] = []
    for key in keys where !consumedKeys.contains(key) {
      if outOfScopePrefixes.contains(where: { key.hasPrefix($0) }) { continue }
      unknown.append(key)
    }
    guard unknown.isEmpty else { throw LoRAError.unknownKeys(unknown) }

    var lokrWeights: [String: LoKrWeights] = [:]
    lokrWeights.reserveCapacity(min(lokrW1.count, lokrW2.count))
    for (key, w1) in lokrW1 {
      guard let w2 = lokrW2[key] else { continue }
      lokrWeights[key] = LoKrWeights(w1: w1, w2: w2, alpha: moduleAlphas[key])
    }

    guard !loraPairs.isEmpty || !lokrWeights.isEmpty || !deltas.isEmpty else {
      throw LoRAError.invalidFormat(
        "No valid Krea-2 LoRA weights found in \(url.lastPathComponent). " +
        "Expected keys matching diffusion_model.blocks.<n>.<attn|mlp>.<proj> patterns, " +
        "LyCORIS LoKr (.lokr_w1/.lokr_w2), or bare patches (.diff/.diff_b/.set_weight)."
      )
    }

    let rank = inferRank(from: loraPairs)
    let alpha = loadAlpha(from: url.deletingLastPathComponent())

    return LoRAWeights(
      weights: loraPairs, lokrWeights: lokrWeights, rank: rank, alpha: alpha,
      deltas: deltas)
  }

  /// Strip the `diffusion_model.` prefix and apply the SAME numeric-index
  /// remap the base weight loader uses (`Krea2WeightLoader.mapTransformerKey`)
  /// so LoRA targets land on the real Swift module paths — without this,
  /// `tmlp.0` / `tproj.1` / `txtmlp.{0,1,3}` keys bind nothing.
  private static func remapKrea2Base(_ raw: String) -> String {
    let stripped = stripKnownPrefixes(raw)
    return String(Krea2WeightLoader.mapTransformerKey(stripped + ".").dropLast())
  }

  /// Krea-2 analogue of ``mapLoKrModuleKey(_:)``: strips the `diffusion_model.`
  /// prefix only, with no ``LoRAKeyMapper`` remap, since Krea-2 LoKr keys
  /// already match `Krea2SingleStreamDiT`'s module paths 1:1 (see
  /// ``loadForKrea2(from:)``).
  private static func mapKrea2LoKrModuleKey(_ key: String) -> (moduleKey: String, suffix: LoKrSuffix)? {
    let suffix: LoKrSuffix
    if key.hasSuffix(LoKrSuffix.w1.rawValue) {
      suffix = .w1
    } else if key.hasSuffix(LoKrSuffix.w2.rawValue) {
      suffix = .w2
    } else if key.hasSuffix(LoKrSuffix.alpha.rawValue) {
      suffix = .alpha
    } else {
      return nil
    }

    var base = String(key.dropLast(suffix.rawValue.count))
    if base.hasPrefix("diffusion_model.") {
      base = String(base.dropFirst("diffusion_model.".count))
    }
    return (base, suffix)
  }

  private static func mapLoKrModuleKey(_ key: String) -> (moduleKey: String, suffix: LoKrSuffix)? {
        let suffix: LoKrSuffix
        if key.hasSuffix(LoKrSuffix.w1.rawValue) {
            suffix = .w1
        } else if key.hasSuffix(LoKrSuffix.w2.rawValue) {
            suffix = .w2
        } else if key.hasSuffix(LoKrSuffix.alpha.rawValue) {
            suffix = .alpha
        } else {
            return nil
        }

        if key.hasPrefix("lycoris_transformer_blocks_") {
            let prefix = "lycoris_transformer_blocks_"
            let remainder = String(key.dropFirst(prefix.count))
            guard let underscoreIndex = remainder.firstIndex(of: "_") else { return nil }
            let layerStr = String(remainder[..<underscoreIndex])
            guard let layerIdx = Int(layerStr) else { return nil }

            let after = String(remainder[remainder.index(after: underscoreIndex)...])
            if after.hasPrefix("attn_to_q") {
                return ("layers.\(layerIdx).attention.to_q", suffix)
            }
            if after.hasPrefix("attn_to_k") {
                return ("layers.\(layerIdx).attention.to_k", suffix)
            }
            if after.hasPrefix("attn_to_v") {
                return ("layers.\(layerIdx).attention.to_v", suffix)
            }
            if after.hasPrefix("attn_to_out_0") {
                return ("layers.\(layerIdx).attention.to_out.0", suffix)
            }
            return nil
        }

        let base = String(key.dropLast(suffix.rawValue.count))
        let mapped = LoRAKeyMapper.mapToZImageKey(base)
        guard mapped.hasSuffix(".weight") else { return nil }
        return (String(mapped.dropLast(".weight".count)), suffix)
    }
}
