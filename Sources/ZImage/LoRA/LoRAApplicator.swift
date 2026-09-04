import Foundation
import MLX
import MLXNN
import Logging

/// What one `applyDynamically` call actually bound (WP-E6, FDD §3.6 / D9).
/// `bound < offered` is the strict *test*; `unbound` is the strict *evidence*.
public struct LoRAApplicationReport: Sendable, Equatable {
    /// `loraWeights.weights.count` — low-rank pairs the file offered.
    public let offered: Int
    /// Modules that received an adapter (`appliedCount`).
    public let bound: Int
    /// Of `bound`, how many landed on quantized Linears.
    public let quantizedBound: Int
    /// Bare-parameter deltas applied through ``LoRAPatchSession`` (the
    /// applicator itself never applies deltas; the pipeline fills this in
    /// via ``withDeltasApplied(_:)``). Required on every Krea-2 render's
    /// provenance so a Raw+kroma render shows `deltas_applied: 0` (D15).
    public let deltasApplied: Int
    /// Pairs that matched a module but failed `normalizeLoRAPair` — a real,
    /// different fault from `unbound`; counted, and in neither list.
    public let shapeRejected: Int
    /// OFFERED KEYS THAT BOUND NOTHING. Sorted. Logged capped at 32.
    public let unbound: [String]

    public init(offered: Int, bound: Int, quantizedBound: Int, deltasApplied: Int,
                shapeRejected: Int, unbound: [String]) {
        self.offered = offered
        self.bound = bound
        self.quantizedBound = quantizedBound
        self.deltasApplied = deltasApplied
        self.shapeRejected = shapeRejected
        self.unbound = unbound
    }

    /// Every offered pair bound and nothing was shape-rejected.
    public var isComplete: Bool { unbound.isEmpty && shapeRejected == 0 }

    public func withDeltasApplied(_ count: Int) -> LoRAApplicationReport {
        LoRAApplicationReport(offered: offered, bound: bound, quantizedBound: quantizedBound,
                              deltasApplied: count, shapeRejected: shapeRejected, unbound: unbound)
    }
}


public struct LoRAApplicator {

    /// Internal (not private) so ``LoRABareParameterPairs`` can ask the
    /// applicator's own question — "could the module walk bind this?" —
    /// rather than keeping a second, driftable copy of the rule.
    static func linearDims(for module: Module) -> (out: Int, in: Int)? {
        if let qlin = module as? QuantizedLinear {
            // qlin.weight is packed (bits-per-element < 32), so its last axis
            // is NOT the true input dimension — qlin.shape already unpacks it
            // via (cols * 32 / bits). Reading weight.dim() directly here silently
            // fails every LoRA shape match against a quantized model.
            let shape = qlin.shape
            return (out: shape.0, in: shape.1)
        }
        if let lin = module as? Linear {
            let outDim = lin.weight.dim(max(0, lin.weight.ndim - 2))
            let inDim = lin.weight.dim(max(0, lin.weight.ndim - 1))
            return (out: outDim, in: inDim)
        }
        return nil
    }

    private static func normalizeLoRAPair(
        down: MLXArray,
        up: MLXArray,
        inFeatures: Int,
        outFeatures: Int
    ) -> (down: MLXArray, up: MLXArray)? {
        guard down.ndim == 2, up.ndim == 2 else { return nil }

        let d0 = down.dim(0)
        let d1 = down.dim(1)
        let u0 = up.dim(0)
        let u1 = up.dim(1)

        // target: down=[rank, in], up=[out, rank]
        if d1 == inFeatures, u0 == outFeatures, d0 == u1 {
            return (down: down, up: up)
        }
        // both transposed: down=[in, rank], up=[rank, out]
        if d0 == inFeatures, u1 == outFeatures, d1 == u0 {
            return (down: down.T, up: up.T)
        }
        // down transposed only: down=[in, rank], up=[out, rank]
        if d0 == inFeatures, u0 == outFeatures, d1 == u1 {
            return (down: down.T, up: up)
        }
        // up transposed only: down=[rank, in], up=[rank, out]
        if d1 == inFeatures, u1 == outFeatures, d0 == u0 {
            return (down: down, up: up.T)
        }
        return nil
    }

    private static func normalizeQKVLoRAPair(
        down: MLXArray,
        up: MLXArray,
        inFeatures: Int,
        outFeatures: Int,
        projectionIndex: Int
    ) -> (down: MLXArray, up: MLXArray)? {
        guard projectionIndex >= 0, projectionIndex < 3 else { return nil }

        guard let normalized = normalizeLoRAPair(
            down: down,
            up: up,
            inFeatures: inFeatures,
            outFeatures: outFeatures * 3
        ) else { return nil }

        let start = projectionIndex * outFeatures
        let end = start + outFeatures
        guard normalized.up.ndim == 2, normalized.up.dim(0) >= end else { return nil }

        let slicedUp = normalized.up[start..<end, 0...]
        return (down: normalized.down, up: slicedUp)
    }

    static func normalizedLoRAPair(
        down: MLXArray,
        up: MLXArray,
        targetShape: [Int]
    ) -> (down: MLXArray, up: MLXArray)? {
        guard targetShape.count >= 2 else { return nil }
        let outFeatures = targetShape[targetShape.count - 2]
        let inFeatures = targetShape[targetShape.count - 1]
        return normalizeLoRAPair(
            down: down,
            up: up,
            inFeatures: inFeatures,
            outFeatures: outFeatures
        )
    }

    public static func mergeWeights(
        baseWeights: [String: MLXArray],
        loraWeights: LoRAWeights,
        scale: Float,
        logger: Logger? = nil
    ) -> [String: MLXArray] {
        var merged = baseWeights

        logger?.info("Merging LoRA weights with scale=\(scale), alpha=\(loraWeights.alpha), rank=\(loraWeights.rank)")

        var appliedCount = 0
        var skippedCount = 0

        for (keyPath, (down, up)) in loraWeights.weights {
            let weightKey = keyPath.hasSuffix(".weight") ? keyPath : keyPath + ".weight"
            let effectiveScale = scale * loraWeights.effectiveScale(forLayer: keyPath)

            guard let baseWeight = merged[weightKey] else {
                logger?.debug("LoRA key '\(weightKey)' not found in base weights, skipping")
                skippedCount += 1
                continue
            }

            guard let normalized = normalizedLoRAPair(
                down: down,
                up: up,
                targetShape: baseWeight.shape
            ) else {
                logger?.warning("LoRA weight shapes incompatible for '\(weightKey)': up=\(up.shape), down=\(down.shape)")
                skippedCount += 1
                continue
            }

            guard let delta = computeDelta(up: normalized.up, down: normalized.down) else {
                logger?.warning("LoRA weight shapes incompatible for '\(weightKey)' after normalization: up=\(normalized.up.shape), down=\(normalized.down.shape)")
                skippedCount += 1
                continue
            }

            guard let alignedDelta = alignShape(delta, to: baseWeight.shape) else {
                logger?.warning("LoRA delta shape \(delta.shape) doesn't match base \(baseWeight.shape) for '\(weightKey)'")
                skippedCount += 1
                continue
            }

            merged[weightKey] = baseWeight + (alignedDelta * effectiveScale).asType(baseWeight.dtype)
            appliedCount += 1
            logger?.debug("Applied LoRA to '\(weightKey)'")
        }

        logger?.info("LoRA merge complete: applied=\(appliedCount), skipped=\(skippedCount)")
        if appliedCount == 0 {
            logger?.warning("LoRA loaded but 0 layers matched the base model. The LoRA may be incompatible with this model architecture.")
        }
        return merged
    }

    public static func applyToTransformer(
        _ transformer: ZImageTransformer2DModel,
        loraWeights: LoRAWeights,
        scale: Float,
        logger: Logger? = nil
    ) {
        logger?.info("Applying LoRA to transformer with scale=\(scale)")

        var appliedCount = 0
        var quantizedCount = 0
        var layerUpdates: [String: MLXArray] = [:]

        for (key, module) in transformer.namedModules() {
            let loraKeyBase = key.hasSuffix(".weight") ? String(key.dropLast(".weight".count)) : key
            let loraKey = loraKeyBase + ".weight"

            guard let (down, up) = loraWeights.weights[loraKey] ?? loraWeights.weights[loraKeyBase] else {
                continue
            }

            let effectiveScale = scale * loraWeights.effectiveScale(forLayer: loraKey)

            guard let dims = linearDims(for: module),
                  let normalized = normalizeLoRAPair(
                    down: down,
                    up: up,
                    inFeatures: dims.in,
                    outFeatures: dims.out
                  ) else {
                logger?.debug("LoRA shape mismatch for \(key): up=\(up.shape), down=\(down.shape)")
                continue
            }

            guard let delta = computeDelta(up: normalized.up, down: normalized.down) else {
                logger?.debug("LoRA shape mismatch for \(key) after normalization: up=\(normalized.up.shape), down=\(normalized.down.shape)")
                continue
            }

            if let quantizedLinear = module as? QuantizedLinear {
                let dequantizedWeight = MLX.dequantized(
                    quantizedLinear.weight,
                    scales: quantizedLinear.scales,
                    biases: quantizedLinear.biases,
                    groupSize: quantizedLinear.groupSize,
                    bits: quantizedLinear.bits
                )

                guard let alignedDelta = alignShape(delta, to: dequantizedWeight.shape) else {
                    logger?.debug("LoRA delta shape \(delta.shape) doesn't match dequantized \(dequantizedWeight.shape) for \(key)")
                    continue
                }

                let fusedWeight = dequantizedWeight + (alignedDelta * effectiveScale).asType(dequantizedWeight.dtype)

                let (newQuantizedWeight, newScales, newBiases) = MLX.quantized(
                    fusedWeight,
                    groupSize: quantizedLinear.groupSize,
                    bits: quantizedLinear.bits
                )

                layerUpdates[key + ".weight"] = newQuantizedWeight
                layerUpdates[key + ".scales"] = newScales
                if let biases = newBiases {
                    layerUpdates[key + ".biases"] = biases
                }

                appliedCount += 1
                quantizedCount += 1

            } else if let linear = module as? Linear {
                let currentWeight = linear.weight

                guard let alignedDelta = alignShape(delta, to: currentWeight.shape) else {
                    logger?.debug("LoRA delta shape \(delta.shape) doesn't match weight \(currentWeight.shape) for \(key)")
                    continue
                }

                layerUpdates[key + ".weight"] = currentWeight + (alignedDelta * effectiveScale).asType(currentWeight.dtype)
                appliedCount += 1
            }
        }

        if !layerUpdates.isEmpty {
            transformer.update(parameters: ModuleParameters.unflattened(layerUpdates))
        }

        if quantizedCount > 0 {
            logger?.info("LoRA applied to transformer: \(appliedCount) layers modified (\(quantizedCount) quantized)")
        } else {
            logger?.info("LoRA applied to transformer: \(appliedCount) layers modified")
        }
        if appliedCount == 0 {
            logger?.warning("LoRA loaded but 0 layers matched the base model. The LoRA may be incompatible with this model architecture.")
        }
    }

    public static func removeFromWeights(
        mergedWeights: [String: MLXArray],
        loraWeights: LoRAWeights,
        scale: Float,
        logger: Logger? = nil
    ) -> [String: MLXArray] {
        var restored = mergedWeights

        for (keyPath, (down, up)) in loraWeights.weights {
            let weightKey = keyPath.hasSuffix(".weight") ? keyPath : keyPath + ".weight"
            let effectiveScale = scale * loraWeights.effectiveScale(forLayer: keyPath)

            guard let currentWeight = restored[weightKey],
                  let normalized = normalizedLoRAPair(
                    down: down,
                    up: up,
                    targetShape: currentWeight.shape
                  ),
                  let delta = computeDelta(up: normalized.up, down: normalized.down),
                  let alignedDelta = alignShape(delta, to: currentWeight.shape) else {
                continue
            }

            restored[weightKey] = currentWeight - (alignedDelta * effectiveScale).asType(currentWeight.dtype)
        }

        return restored
    }

    static func computeDelta(up: MLXArray, down: MLXArray) -> MLXArray? {
        guard up.ndim == 2, down.ndim == 2, up.dim(1) == down.dim(0) else { return nil }
        return MLX.matmul(up, down)
    }

    private static func alignShape(_ delta: MLXArray, to targetShape: [Int]) -> MLXArray? {
        delta.shape == targetShape ? delta : nil
    }

    public static func applyLoKr<T: Module>(
        to transformer: T,
        loraWeights: LoRAWeights,
        scale: Float,
        logger: Logger? = nil
    ) {
        applyLoKrInternal(to: transformer, loraWeights: loraWeights, signedScale: scale, logger: logger)
    }

    public static func removeLoKr<T: Module>(
        from transformer: T,
        loraWeights: LoRAWeights,
        scale: Float,
        logger: Logger? = nil
    ) {
        applyLoKrInternal(to: transformer, loraWeights: loraWeights, signedScale: -scale, logger: logger)
    }

    /// LyCORIS LoKr convention: the delta w1 ⊗ w2 is scaled by alpha / dim,
    /// not by alpha as a raw multiplier. We only load full-matrix LoKr
    /// (lokr_w1/lokr_w2 without a w2_a/w2_b factorization), so dim is taken
    /// as the smaller dimension of w2 — the `lora_dim` a factorized w2 would
    /// have used. LyCORIS stores alpha == dim for full-matrix modules, so
    /// this typically evaluates to 1.0. Internal for unit testing.
    ///
    /// Some exporters (observed from `ai-toolkit`-trained LoKr adapters, e.g.
    /// Krea2-realism-V2 and the zit_fdpo_v1/zit-sda-v1 Z-Image LoKr LoRAs)
    /// write a fixed ~1e10 sentinel into every `.alpha` tensor instead of
    /// following the alpha == dim convention or omitting the field. Dividing
    /// that by a typical dim (hundreds to a few thousand) yields a scale in
    /// the millions, which blows up the delta and reduces output to pure
    /// noise even at a caller-requested scale like 0.4. Any ratio far outside
    /// what a real trained LoRA would use is treated as "no real alpha
    /// metadata" and falls back to the neutral 1.0.
    static func lokrAlphaScale(alpha: Float?, w2Shape: [Int]) -> Float {
        guard let alpha, w2Shape.count == 2 else { return 1.0 }
        let dim = min(w2Shape[0], w2Shape[1])
        guard dim > 0, alpha > 0 else { return 1.0 }
        let scale = alpha / Float(dim)
        guard scale.isFinite, scale < 1000 else { return 1.0 }
        return scale
    }

    private static func applyLoKrInternal<T: Module>(
        to transformer: T,
        loraWeights: LoRAWeights,
        signedScale: Float,
        logger: Logger? = nil
    ) {
        guard !loraWeights.lokrWeights.isEmpty else { return }

        var layerUpdates: [String: MLXArray] = [:]
        var appliedCount = 0
        var quantizedCount = 0

        func kron2D(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
            let a0 = a.dim(0), a1 = a.dim(1)
            let b0 = b.dim(0), b1 = b.dim(1)
            let aExp = a.reshaped(a0, 1, a1, 1)
            let bExp = b.reshaped(1, b0, 1, b1)
            return (aExp * bExp).reshaped(a0 * b0, a1 * b1)
        }

        for (key, module) in transformer.namedModules() {
            guard let lokr = loraWeights.lokrWeights[key] else { continue }
            guard let dims = linearDims(for: module) else { continue }
            guard lokr.w1.ndim == 2, lokr.w2.ndim == 2 else { continue }

            let expectedOut = lokr.w1.dim(0) * lokr.w2.dim(0)
            let expectedIn = lokr.w1.dim(1) * lokr.w2.dim(1)
            if expectedOut != dims.out || expectedIn != dims.in {
                logger?.debug("Skipping LoKr for \(key): kron (\(expectedOut)x\(expectedIn)) vs weight (\(dims.out)x\(dims.in))")
                continue
            }

            let effectiveScale = signedScale * lokrAlphaScale(alpha: lokr.alpha, w2Shape: lokr.w2.shape)

            if let qlin = module as? QuantizedLinear {
                let dequantizedWeight = MLX.dequantized(
                    qlin.weight,
                    scales: qlin.scales,
                    biases: qlin.biases,
                    groupSize: qlin.groupSize,
                    bits: qlin.bits
                )

                var delta = kron2D(lokr.w1, lokr.w2)
                if let aligned = alignShape(delta, to: dequantizedWeight.shape) {
                    delta = aligned
                } else {
                    logger?.debug("Skipping LoKr for \(key): delta=\(delta.shape) weight=\(dequantizedWeight.shape)")
                    continue
                }

                if delta.dtype != dequantizedWeight.dtype { delta = delta.asType(dequantizedWeight.dtype) }
                let fusedWeight = dequantizedWeight + (delta * effectiveScale).asType(dequantizedWeight.dtype)
                let (newQuantizedWeight, newScales, newBiases) = MLX.quantized(
                    fusedWeight,
                    groupSize: qlin.groupSize,
                    bits: qlin.bits
                )

                layerUpdates[key + ".weight"] = newQuantizedWeight
                layerUpdates[key + ".scales"] = newScales
                if let biases = newBiases {
                    layerUpdates[key + ".biases"] = biases
                }

                appliedCount += 1
                quantizedCount += 1
                continue
            }

            if let lin = module as? Linear {
                let currentWeight = lin.weight

                var delta = kron2D(lokr.w1, lokr.w2)
                if let aligned = alignShape(delta, to: currentWeight.shape) {
                    delta = aligned
                } else {
                    logger?.debug("Skipping LoKr for \(key): delta=\(delta.shape) weight=\(currentWeight.shape)")
                    continue
                }

                if delta.dtype != currentWeight.dtype { delta = delta.asType(currentWeight.dtype) }
                layerUpdates[key + ".weight"] = currentWeight + (delta * effectiveScale).asType(currentWeight.dtype)
                appliedCount += 1
            }
        }

        if !layerUpdates.isEmpty {
            transformer.update(parameters: ModuleParameters.unflattened(layerUpdates))
        }

        if appliedCount > 0 {
            if quantizedCount > 0 {
                logger?.info("LoKr applied to \(appliedCount) layers (\(quantizedCount) quantized)")
            } else {
                logger?.info("LoKr applied to \(appliedCount) layers")
            }
        } else if !loraWeights.lokrWeights.isEmpty {
            logger?.warning("LoRA loaded but 0 layers matched the base model. The LoRA may be incompatible with this model architecture.")
        }
    }

    /// Apply the low-rank pairs in `loraWeights` as dynamic adapters and
    /// report exactly what bound (WP-E6, FDD §3.6 / D9).
    ///
    /// - `strict: false` (default — Z-Image, Flux2, Chroma): behaviour is
    ///   byte-identical to the pre-E6 applicator; the report is informational.
    /// - `strict: true` (Krea-2 only): an offered key that matched no module,
    ///   or a pair that matched but failed shape normalisation, throws BEFORE
    ///   any module is touched — the bind list is built first and committed
    ///   only once it is complete. `partialApplication` names the unbound
    ///   keys; a shape rejection throws `incompatibleWeights` naming the key.
    ///
    /// `unbound` is **offered keys minus consumed keys** — never the modules
    /// that had no adapter (thousands for any sparse LoRA) and never blind to
    /// an offered key no module visits (the one failure that matters).
    /// One resolved (module, normalized pair) awaiting commit — built in full
    /// before anything is mutated so a strict refusal touches nothing.
    private struct PendingBind {
        let key: String
        let module: Module
        let down: MLXArray
        let up: MLXArray
        let scale: Float
    }

    @discardableResult
    public static func applyDynamically<T: Module>(
        to transformer: T,
        loraWeights: LoRAWeights,
        scale: Float,
        strict: Bool = false,
        name: String? = nil,
        logger: Logger? = nil
    ) throws -> LoRAApplicationReport {
        logger?.info("Applying dynamic LoRA with scale=\(scale)\(strict ? " (strict)" : "")")

        var pending: [PendingBind] = []
        var consumedKeys = Set<String>()
        var shapeRejectedKeys: [String] = []

        for (key, module) in transformer.namedModules() {
            guard let dims = linearDims(for: module) else { continue }

            let loraKey = key.hasSuffix(".weight") ? key : key + ".weight"
            var pair: (down: MLXArray, up: MLXArray)?
            var scaleKey = loraKey
            // The dictionary key that actually matched — what `consumedKeys`
            // records (the offered spelling, with or without ".weight").
            var matchedKey: String?

            if let direct = loraWeights.weights[loraKey] {
                pair = direct
                matchedKey = loraKey
            } else if let bare = loraWeights.weights[key] {
                pair = bare
                matchedKey = key
            }

            if pair == nil {
                // Fallback: some LoRA packs store combined qkv deltas under attention.qkv.*
                if key.contains(".attention.to_q") || key.contains(".attention.to_k") || key.contains(".attention.to_v") {
                    let qkvKey: String?
                    let projectionIndex: Int

                    if let range = key.range(of: ".attention.to_q") {
                        qkvKey = key.replacingCharacters(in: range, with: ".attention.qkv")
                        projectionIndex = 0
                    } else if let range = key.range(of: ".attention.to_k") {
                        qkvKey = key.replacingCharacters(in: range, with: ".attention.qkv")
                        projectionIndex = 1
                    } else if let range = key.range(of: ".attention.to_v") {
                        qkvKey = key.replacingCharacters(in: range, with: ".attention.qkv")
                        projectionIndex = 2
                    } else {
                        qkvKey = nil
                        projectionIndex = 0
                    }

                    if let qkvKey {
                        let qkvMatched: String? = loraWeights.weights[qkvKey + ".weight"] != nil
                            ? qkvKey + ".weight"
                            : (loraWeights.weights[qkvKey] != nil ? qkvKey : nil)
                        if let qkvMatched,
                           let qkvPair = loraWeights.weights[qkvMatched],
                           let normalized = normalizeQKVLoRAPair(
                            down: qkvPair.down,
                            up: qkvPair.up,
                            inFeatures: dims.in,
                            outFeatures: dims.out,
                            projectionIndex: projectionIndex
                           ) {
                            pair = normalized
                            scaleKey = qkvKey + ".weight"
                            matchedKey = qkvMatched
                        }
                    }
                }
            }

            guard let pair, let matchedKey else { continue }

            let effectiveScale = scale * loraWeights.effectiveScale(forLayer: scaleKey)

            guard let normalized = normalizeLoRAPair(
                down: pair.down,
                up: pair.up,
                inFeatures: dims.in,
                outFeatures: dims.out
            ) else {
                logger?.debug("Skipping LoRA for \(key): down=\(pair.down.shape) up=\(pair.up.shape) vs weight (\(dims.out)x\(dims.in))")
                shapeRejectedKeys.append(matchedKey)
                continue
            }

            pending.append(PendingBind(
                key: key, module: module,
                down: normalized.down, up: normalized.up, scale: effectiveScale))
            consumedKeys.insert(matchedKey)
        }

        let offered = loraWeights.weights.count
        let shapeRejectedSet = Set(shapeRejectedKeys)
        let unbound = Set(loraWeights.weights.keys)
            .subtracting(consumedKeys)
            .subtracting(shapeRejectedSet)
            .sorted()

        if strict {
            if !unbound.isEmpty {
                logger?.error("LoRA \(name ?? "<unnamed>") bound \(pending.count)/\(offered): \(unbound.count) offered key(s) matched no module — refusing (strict): \(unbound.prefix(32).joined(separator: ", "))")
                throw LoRAError.partialApplication(lora: name, unbound: unbound)
            }
            if !shapeRejectedSet.isEmpty {
                let listed = shapeRejectedSet.sorted().prefix(32).joined(separator: ", ")
                logger?.error("LoRA \(name ?? "<unnamed>"): \(shapeRejectedSet.count) pair(s) matched a module but not its shape — refusing (strict): \(listed)")
                throw LoRAError.incompatibleWeights(
                    "LoRA '\(name ?? "<unnamed>")' has \(shapeRejectedSet.count) pair(s) whose shape does not fit the target module: \(listed)")
            }
        }

        // ---- Commit: nothing above mutated the module tree ----
        var moduleUpdates: [(String, Module)] = []
        var appliedCount = 0
        var quantizedCount = 0

        for p in pending {
            if let loraLinear = p.module as? LoRALinear {
                loraLinear.addLoRA(down: p.down, up: p.up, scale: p.scale)
                appliedCount += 1
                continue
            }
            if let loraQuantized = p.module as? LoRAQuantizedLinear {
                loraQuantized.addLoRA(down: p.down, up: p.up, scale: p.scale)
                appliedCount += 1
                quantizedCount += 1
                continue
            }
            if let quantizedLinear = p.module as? QuantizedLinear {
                let loraQuantized = LoRAQuantizedLinear(from: quantizedLinear)
                loraQuantized.addLoRA(down: p.down, up: p.up, scale: p.scale)
                moduleUpdates.append((p.key, loraQuantized))
                appliedCount += 1
                quantizedCount += 1
                continue
            }
            if let linear = p.module as? Linear {
                let loraLinear = LoRALinear(from: linear)
                loraLinear.addLoRA(down: p.down, up: p.up, scale: p.scale)
                moduleUpdates.append((p.key, loraLinear))
                appliedCount += 1
                continue
            }
        }

        if !moduleUpdates.isEmpty {
            transformer.update(modules: ModuleChildren.unflattened(moduleUpdates))
        }

        if loraWeights.hasLoKr {
            applyLoKr(to: transformer, loraWeights: loraWeights, scale: scale, logger: logger)
        }

        if quantizedCount > 0 {
            logger?.info("Dynamic LoRA applied to \(appliedCount) layers (\(quantizedCount) quantized)")
        } else {
            logger?.info("Dynamic LoRA applied to \(appliedCount) layers")
        }
        // A LoRA can be 100% LoKr (no plain lora_down/lora_up pairs at all), in
        // which case appliedCount is legitimately 0 here even though applyLoKr
        // above matched every layer — check hasLoKr too before warning.
        if appliedCount == 0 && !loraWeights.hasLoKr {
            logger?.warning("LoRA loaded but 0 layers matched the base model. The LoRA may be incompatible with this model architecture.")
        }
        if !unbound.isEmpty {
            logger?.warning("LoRA \(name ?? "<unnamed>") bound \(appliedCount)/\(offered): \(unbound.count) offered key(s) matched no module: \(unbound.prefix(32).joined(separator: ", "))\(unbound.count > 32 ? " …" : "")")
        }

        return LoRAApplicationReport(
            offered: offered,
            bound: appliedCount,
            quantizedBound: quantizedCount,
            deltasApplied: 0,
            shapeRejected: shapeRejectedSet.count,
            unbound: unbound)
    }

    public static func clearDynamicLoRA<T: Module>(
        from transformer: T,
        logger: Logger? = nil
    ) {
        var clearedCount = 0

        for (_, module) in transformer.namedModules() {
            if let loraLinear = module as? LoRALinear {
                loraLinear.clearLoRA()
                clearedCount += 1
            } else if let loraQuantized = module as? LoRAQuantizedLinear {
                loraQuantized.clearLoRA()
                clearedCount += 1
            }
        }

        logger?.info("Cleared dynamic LoRA from \(clearedCount) layers")
    }

    public static func hasDynamicLoRA<T: Module>(in transformer: T) -> Bool {
        for (_, module) in transformer.namedModules() {
            if let loraLinear = module as? LoRALinear, loraLinear.hasLoRA {
                return true
            }
            if let loraQuantized = module as? LoRAQuantizedLinear, loraQuantized.hasLoRA {
                return true
            }
        }
        return false
    }
}
