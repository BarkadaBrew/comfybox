// ChromaLoRALoader.swift — Load and apply LoRA weights to ChromaTransformer
//
// Uses ChromaLoRAKeyMapper for Chroma-specific key mapping, then merges
// LoRA deltas directly into transformer weights (avoids Sequential @ModuleInfo issue).

import Foundation
import MLX
import MLXNN
import Logging

/// Loads LoRA safetensors files and applies them to a ChromaTransformer.
///
/// Uses weight merging (not dynamic LoRA wrapping) because Chroma's
/// Sequential MLP blocks don't support module replacement via update(modules:).
/// Instead, we compute delta = up @ down, scale it, and add to existing weights.
public enum ChromaLoRALoader {

    /// Load a LoRA file and merge its weights into a ChromaTransformer.
    ///
    /// - Parameters:
    ///   - path: Path to the LoRA safetensors file.
    ///   - transformer: The ChromaTransformer to modify.
    ///   - scale: LoRA scale factor (default 1.0).
    ///   - logger: Logger for diagnostics.
    /// - Returns: Number of layers modified.
    @discardableResult
    public static func loadAndApply(
        path: String,
        to transformer: ChromaTransformer,
        scale: Float = 1.0,
        logger: Logger
    ) throws -> Int {
        let url = URL(fileURLWithPath: path)
        logger.info("Loading Chroma LoRA from \(url.lastPathComponent)")

        // Load raw safetensors
        let reader = try SafeTensorsReader(fileURL: url)
        let allKeys = reader.tensorNames

        // Parse LoRA pairs using Chroma key mapper
        var loraPairs: [String: (down: MLXArray, up: MLXArray)] = [:]
        var perLayerAlpha: [String: Float] = [:]
        var processedKeys = Set<String>()
        var globalAlpha: Float? = nil

        let loraPatterns: [(down: String, up: String)] = [
            (".lora_down.", ".lora_up."),
            (".lora_A.", ".lora_B."),
        ]

        for key in allKeys {
            if processedKeys.contains(key) { continue }

            // Per-layer alpha
            if key.hasSuffix(".alpha") {
                let tensor = try reader.tensor(named: key)
                if let value = tensor.asArray(Float.self).first {
                    // Extract base and map
                    let baseKey = String(key.dropLast(".alpha".count))
                    let mappedKey = ChromaLoRAKeyMapper.map(baseKey)
                    perLayerAlpha[mappedKey] = value
                    if globalAlpha == nil { globalAlpha = value }
                }
                processedKeys.insert(key)
                continue
            }

            // Standard LoRA: find down/up pairs
            for (downPattern, upPattern) in loraPatterns {
                if key.contains(downPattern) {
                    let upKey = key.replacingOccurrences(of: downPattern, with: upPattern)
                    guard reader.contains(upKey) else { continue }

                    // Extract base key (before .lora_down.)
                    guard let range = key.range(of: downPattern) else { continue }
                    let baseKey = String(key[..<range.lowerBound])

                    // Map to Chroma module path
                    let mappedKey = ChromaLoRAKeyMapper.map(baseKey)

                    let downWeight = try reader.tensor(named: key)
                    let upWeight = try reader.tensor(named: upKey)
                    loraPairs[mappedKey] = (down: downWeight, up: upWeight)

                    processedKeys.insert(key)
                    processedKeys.insert(upKey)
                    break
                }
            }
        }

        guard !loraPairs.isEmpty else {
            logger.warning("No valid LoRA weights found in \(url.lastPathComponent)")
            return 0
        }

        // Infer rank
        let rank: Int = loraPairs.values.first.map { pair in
            min(pair.down.dim(0), pair.down.dim(1))
        } ?? 16

        // Compute effective scale: scale * alpha / rank
        let alpha = globalAlpha ?? Float(rank)
        let effectiveScale = scale * alpha / Float(rank)

        logger.info("Chroma LoRA: \(loraPairs.count) pairs, rank=\(rank), alpha=\(alpha), effective_scale=\(effectiveScale)")

        // Merge LoRA weights directly into transformer parameters
        var weightUpdates: [String: MLXArray] = [:]
        var appliedCount = 0
        var skippedCount = 0

        // Build a lookup of current module weights
        for (modulePath, module) in transformer.namedModules() {
            // We need the weight key: modulePath + ".weight"
            let weightKey = modulePath + ".weight"

            guard let (down, up) = loraPairs[weightKey] else { continue }

            // Get current weight from the module
            guard let linear = module as? Linear else {
                logger.debug("LoRA target '\(modulePath)' is not a Linear, skipping")
                skippedCount += 1
                continue
            }

            let currentWeight = linear.weight
            let outDim = currentWeight.dim(0)
            let inDim = currentWeight.dim(1)

            // Normalize LoRA pair orientation
            // Target: down=[rank, in], up=[out, rank] → delta = up @ down = [out, in]
            guard let (normDown, normUp) = normalizeLoRAPair(
                down: down, up: up, inFeatures: inDim, outFeatures: outDim
            ) else {
                // Try fused QKV: up might be [3*out, rank] for qkv layers
                if weightKey.contains(".qkv.") || weightKey.contains(".linear1.") {
                    let fusedOutDim = currentWeight.dim(0)
                    guard let (normDown2, normUp2) = normalizeLoRAPair(
                        down: down, up: up, inFeatures: inDim, outFeatures: fusedOutDim
                    ) else {
                        logger.debug("LoRA shape mismatch for '\(weightKey)': down=\(down.shape) up=\(up.shape) vs weight=\(currentWeight.shape)")
                        skippedCount += 1
                        continue
                    }
                    let delta = MLX.matmul(normUp2, normDown2)
                    let scaledDelta = (delta * effectiveScale).asType(currentWeight.dtype)
                    weightUpdates[weightKey] = currentWeight + scaledDelta
                    appliedCount += 1
                    continue
                }

                logger.debug("LoRA shape mismatch for '\(weightKey)': down=\(down.shape) up=\(up.shape) vs weight (\(outDim)x\(inDim))")
                skippedCount += 1
                continue
            }

            // Compute delta: up @ down = [out, in]
            let delta = MLX.matmul(normUp, normDown)
            let scaledDelta = (delta * effectiveScale).asType(currentWeight.dtype)
            weightUpdates[weightKey] = currentWeight + scaledDelta
            appliedCount += 1
        }

        // Apply all weight updates at once
        if !weightUpdates.isEmpty {
            let params = ModuleParameters.unflattened(weightUpdates)
            try transformer.update(parameters: params, verify: [.shapeMismatch])
        }

        logger.info("Chroma LoRA merged: \(appliedCount) layers modified, \(skippedCount) skipped")
        return appliedCount
    }

    // MARK: - Private

    /// Normalize LoRA weight pair to standard orientation.
    /// Target: down=[rank, in], up=[out, rank]
    private static func normalizeLoRAPair(
        down: MLXArray,
        up: MLXArray,
        inFeatures: Int,
        outFeatures: Int
    ) -> (down: MLXArray, up: MLXArray)? {
        guard down.ndim == 2, up.ndim == 2 else { return nil }

        let d0 = down.dim(0), d1 = down.dim(1)
        let u0 = up.dim(0), u1 = up.dim(1)

        // Standard: down=[rank, in], up=[out, rank]
        if d1 == inFeatures && u0 == outFeatures && d0 == u1 {
            return (down: down, up: up)
        }
        // Both transposed: down=[in, rank], up=[rank, out]
        if d0 == inFeatures && u1 == outFeatures && d1 == u0 {
            return (down: down.T, up: up.T)
        }
        // Down transposed: down=[in, rank], up=[out, rank]
        if d0 == inFeatures && u0 == outFeatures && d1 == u1 {
            return (down: down.T, up: up)
        }
        // Up transposed: down=[rank, in], up=[rank, out]
        if d1 == inFeatures && u1 == outFeatures && d0 == u0 {
            return (down: down, up: up.T)
        }
        return nil
    }
}
