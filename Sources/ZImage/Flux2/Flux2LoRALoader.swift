// Flux2LoRALoader.swift — Load and apply LoRA/LoKR weights to Flux2Transformer
//
// Handles the key mapping gap between native Flux naming (used by ComfyUI LoRAs
// like snofs) and the diffusers naming used by Flux2Transformer's @ModuleInfo keys.
//
// Native Flux keys use fused QKV (double_blocks.N.img_attn.qkv) while
// Flux2Transformer uses separate Q/K/V projections (transformer_blocks.N.attn.to_q).
// This loader handles the split by computing the full LoKR/LoRA delta, slicing it,
// and merging each part into the corresponding separate weight.

import Foundation
import MLX
import MLXNN
import Logging

/// Loads LoRA/LoKR safetensors files and applies them to a Flux2Transformer.
///
/// Uses weight merging (not dynamic LoRA wrapping) to handle the architecture
/// mismatch between native Flux naming and diffusers naming.
public enum Flux2LoRALoader {

    // MARK: - Public API

    /// Load a LoRA file and merge its weights into a Flux2Transformer.
    ///
    /// Supports both standard LoRA (lora_down/lora_up) and LoKR (lokr_w1/lokr_w2).
    ///
    /// - Parameters:
    ///   - path: Path to the LoRA safetensors file.
    ///   - transformer: The Flux2Transformer to modify.
    ///   - scale: LoRA scale factor (default 1.0).
    ///   - logger: Logger for diagnostics.
    /// - Returns: Number of layers modified.
    @discardableResult
    public static func loadAndApply(
        path: String,
        to transformer: Flux2Transformer,
        scale: Float = 1.0,
        logger: Logger
    ) throws -> Int {
        let url = URL(fileURLWithPath: path)
        logger.info("Loading Flux 2 LoRA from \(url.lastPathComponent)")

        let reader = try SafeTensorsReader(fileURL: url)
        let allKeys = reader.tensorNames

        // Parse LoKR entries and standard LoRA pairs
        var lokrEntries: [String: LoKrEntry] = [:]
        var loraPairs: [String: (down: MLXArray, up: MLXArray)] = [:]
        var loraAlphas: [String: Float] = [:]
        var processedKeys = Set<String>()

        for key in allKeys {
            if processedKeys.contains(key) { continue }

            // LoKR: look for .lokr_w1 keys
            if key.hasSuffix(".lokr_w1") {
                let rawBase = String(key.dropLast(".lokr_w1".count))
                let w2Key = rawBase + ".lokr_w2"
                let alphaKey = rawBase + ".alpha"

                guard reader.contains(w2Key) else { continue }
                let w1 = try reader.tensor(named: key)
                let w2 = try reader.tensor(named: w2Key)
                var alpha: Float? = nil
                if reader.contains(alphaKey) {
                    let alphaTensor = try reader.tensor(named: alphaKey)
                    alpha = alphaTensor.asArray(Float.self).first
                }

                let nativeKey = stripPrefix(rawBase)
                lokrEntries[nativeKey] = LoKrEntry(w1: w1, w2: w2, alpha: alpha)
                processedKeys.formUnion([key, w2Key])
                if reader.contains(alphaKey) { processedKeys.insert(alphaKey) }
                continue
            }

            // Skip w2 (handled with w1)
            if key.hasSuffix(".lokr_w2") { continue }

            // Store alpha values for standard LoRA pairs
            if key.hasSuffix(".alpha") {
                let rawBase = String(key.dropLast(".alpha".count))
                let nativeKey = stripPrefix(rawBase)
                if let alphaTensor = try? reader.tensor(named: key) {
                    loraAlphas[nativeKey] = alphaTensor.asArray(Float.self).first
                }
                processedKeys.insert(key)
                continue
            }

            // Standard LoRA: look for .lora_down. / .lora_A.
            let loraPatterns: [(down: String, up: String)] = [
                (".lora_down.", ".lora_up."),
                (".lora_A.", ".lora_B."),
            ]

            for (downPattern, upPattern) in loraPatterns {
                if key.contains(downPattern) {
                    let upKey = key.replacingOccurrences(of: downPattern, with: upPattern)
                    guard reader.contains(upKey) else { continue }
                    guard let range = key.range(of: downPattern) else { continue }
                    let rawBase = String(key[..<range.lowerBound])
                    let nativeKey = stripPrefix(rawBase)

                    let down = try reader.tensor(named: key)
                    let up = try reader.tensor(named: upKey)
                    loraPairs[nativeKey] = (down: down, up: up)

                    processedKeys.formUnion([key, upKey])
                    break
                }
            }
        }

        guard !lokrEntries.isEmpty || !loraPairs.isEmpty else {
            logger.warning("No valid LoRA weights found in \(url.lastPathComponent)")
            return 0
        }

        logger.info("Flux 2 LoRA: \(lokrEntries.count) LoKR entries, \(loraPairs.count) standard LoRA pairs")

        // Build module weight lookup from transformer
        var moduleWeights: [String: MLXArray] = [:]
        for (modulePath, module) in transformer.namedModules() {
            if let linear = module as? Linear {
                moduleWeights[modulePath] = linear.weight
            }
        }

        var weightUpdates: [String: MLXArray] = [:]
        var appliedCount = 0
        var skippedCount = 0

        // Apply LoKR entries
        for (nativeKey, entry) in lokrEntries {
            let targets = mapNativeToFlux2(nativeKey)
            guard !targets.isEmpty else {
                logger.debug("LoKR: no mapping for '\(nativeKey)'")
                skippedCount += 1
                continue
            }

            // Compute LoKR delta: kron(w1, w2) * scale * (alpha / norm_factor)
            let delta = computeLoKrDelta(w1: entry.w1, w2: entry.w2, alpha: entry.alpha, scale: scale)

            if targets.count == 1 {
                // Direct 1:1 mapping (proj, mlp, linear1/2)
                let targetPath = targets[0]
                guard let currentWeight = moduleWeights[targetPath] else {
                    logger.debug("LoKR: no module at '\(targetPath)' for '\(nativeKey)'")
                    skippedCount += 1
                    continue
                }
                // Verify shape compatibility
                if delta.shape == currentWeight.shape {
                    weightUpdates[targetPath + ".weight"] = currentWeight + delta.asType(currentWeight.dtype)
                    appliedCount += 1
                } else {
                    logger.debug("LoKR shape mismatch for '\(targetPath)': delta=\(delta.shape) vs weight=\(currentWeight.shape)")
                    skippedCount += 1
                }
            } else {
                // QKV split: delta covers fused QKV, split into parts
                let totalOut = delta.dim(0)
                let splitSize = totalOut / targets.count
                for (i, targetPath) in targets.enumerated() {
                    guard let currentWeight = moduleWeights[targetPath] else {
                        logger.debug("LoKR: no module at '\(targetPath)' for '\(nativeKey)' [split \(i)]")
                        skippedCount += 1
                        continue
                    }
                    let start = i * splitSize
                    let end = start + splitSize
                    let slicedDelta = delta[start..<end, 0...]
                    if slicedDelta.shape == currentWeight.shape {
                        weightUpdates[targetPath + ".weight"] = currentWeight + slicedDelta.asType(currentWeight.dtype)
                        appliedCount += 1
                    } else {
                        logger.debug("LoKR split shape mismatch at '\(targetPath)': slice=\(slicedDelta.shape) vs weight=\(currentWeight.shape)")
                        skippedCount += 1
                    }
                }
            }
        }

        // Apply standard LoRA pairs
        for (nativeKey, pair) in loraPairs {
            let targets = mapNativeToFlux2(nativeKey)
            guard !targets.isEmpty else {
                logger.debug("LoRA: no mapping for '\(nativeKey)'")
                skippedCount += 1
                continue
            }

            // Compute effective scale with alpha/rank normalization
            let rank = min(pair.down.dim(0), pair.down.dim(1))
            let effectiveScale: Float
            if let alpha = loraAlphas[nativeKey], rank > 0 {
                effectiveScale = scale * alpha / Float(rank)
            } else {
                effectiveScale = scale
            }

            // Compute delta: up @ down * effective_scale
            let delta = computeLoRADelta(down: pair.down, up: pair.up, scale: effectiveScale)
            guard let delta else {
                logger.debug("LoRA: couldn't compute delta for '\(nativeKey)'")
                skippedCount += 1
                continue
            }

            if targets.count == 1 {
                let targetPath = targets[0]
                guard let currentWeight = moduleWeights[targetPath] else {
                    skippedCount += 1
                    continue
                }
                if delta.shape == currentWeight.shape {
                    weightUpdates[targetPath + ".weight"] = currentWeight + delta.asType(currentWeight.dtype)
                    appliedCount += 1
                } else {
                    logger.debug("LoRA shape mismatch for '\(targetPath)': delta=\(delta.shape) vs weight=\(currentWeight.shape)")
                    skippedCount += 1
                }
            } else {
                let totalOut = delta.dim(0)
                let splitSize = totalOut / targets.count
                for (i, targetPath) in targets.enumerated() {
                    guard let currentWeight = moduleWeights[targetPath] else {
                        skippedCount += 1
                        continue
                    }
                    let start = i * splitSize
                    let end = start + splitSize
                    let slicedDelta = delta[start..<end, 0...]
                    if slicedDelta.shape == currentWeight.shape {
                        weightUpdates[targetPath + ".weight"] = currentWeight + slicedDelta.asType(currentWeight.dtype)
                        appliedCount += 1
                    } else {
                        skippedCount += 1
                    }
                }
            }
        }

        // Apply all weight updates at once
        if !weightUpdates.isEmpty {
            let params = ModuleParameters.unflattened(weightUpdates)
            try transformer.update(parameters: params, verify: [.shapeMismatch])
        }

        logger.info("Flux 2 LoRA merged: \(appliedCount) layers modified, \(skippedCount) skipped")
        return appliedCount
    }

    // MARK: - Private Types

    private struct LoKrEntry {
        let w1: MLXArray
        let w2: MLXArray
        let alpha: Float?
    }

    // MARK: - Key Mapping

    /// Prefixes stripped from raw LoRA keys.
    private static let prefixesToStrip = [
        "base_model.model.",
        "diffusion_model.",
        "lora_unet_",
        "transformer.",
        "text_encoder.",
        "model.",
    ]

    private static func stripPrefix(_ key: String) -> String {
        for prefix in prefixesToStrip {
            if key.hasPrefix(prefix) {
                return String(key.dropFirst(prefix.count))
            }
        }
        return key
    }

    /// Map a native Flux key to Flux2Transformer module path(s).
    ///
    /// Returns multiple paths for fused QKV keys (split into Q, K, V).
    /// Returns a single path for non-fused keys.
    ///
    /// ## Key Mappings
    /// ```
    /// Native Flux                          → Flux2Transformer module path(s)
    /// double_blocks.N.img_attn.qkv         → [attn.to_q, attn.to_k, attn.to_v]  (SPLIT)
    /// double_blocks.N.img_attn.proj        → attn.to_out
    /// double_blocks.N.txt_attn.qkv         → [attn.add_q_proj, attn.add_k_proj, attn.add_v_proj]  (SPLIT)
    /// double_blocks.N.txt_attn.proj        → attn.to_add_out
    /// double_blocks.N.img_mlp.0            → ff.linear_in
    /// double_blocks.N.img_mlp.2            → ff.linear_out
    /// double_blocks.N.txt_mlp.0            → ff_context.linear_in
    /// double_blocks.N.txt_mlp.2            → ff_context.linear_out
    /// single_blocks.N.linear1              → single_transformer_blocks.N.attn.to_qkv_mlp_proj
    /// single_blocks.N.linear2              → single_transformer_blocks.N.attn.to_out
    /// ```
    private static func mapNativeToFlux2(_ nativeKey: String) -> [String] {
        // Double blocks
        if nativeKey.hasPrefix("double_blocks.") {
            return mapDoubleBlock(nativeKey)
        }

        // Single blocks
        if nativeKey.hasPrefix("single_blocks.") {
            return mapSingleBlock(nativeKey)
        }

        // Top-level (x_embedder, context_embedder, etc.)
        if nativeKey == "img_in" {
            return ["x_embedder"]
        }
        if nativeKey == "txt_in" {
            return ["context_embedder"]
        }

        return []
    }

    private static func mapDoubleBlock(_ key: String) -> [String] {
        // Parse: double_blocks.N.<rest>
        let parts = key.split(separator: ".", maxSplits: 2)
        guard parts.count == 3,
              let blockIdx = Int(parts[1]) else { return [] }

        let rest = String(parts[2])
        let prefix = "transformer_blocks.\(blockIdx)"

        // Image attention
        if rest == "img_attn.qkv" {
            return ["\(prefix).attn.to_q", "\(prefix).attn.to_k", "\(prefix).attn.to_v"]
        }
        if rest == "img_attn.proj" {
            return ["\(prefix).attn.to_out"]
        }

        // Text attention
        if rest == "txt_attn.qkv" {
            return ["\(prefix).attn.add_q_proj", "\(prefix).attn.add_k_proj", "\(prefix).attn.add_v_proj"]
        }
        if rest == "txt_attn.proj" {
            return ["\(prefix).attn.to_add_out"]
        }

        // Image MLP
        if rest == "img_mlp.0" {
            return ["\(prefix).ff.linear_in"]
        }
        if rest == "img_mlp.2" {
            return ["\(prefix).ff.linear_out"]
        }

        // Text MLP
        if rest == "txt_mlp.0" {
            return ["\(prefix).ff_context.linear_in"]
        }
        if rest == "txt_mlp.2" {
            return ["\(prefix).ff_context.linear_out"]
        }

        return []
    }

    private static func mapSingleBlock(_ key: String) -> [String] {
        // Parse: single_blocks.N.<rest>
        let parts = key.split(separator: ".", maxSplits: 2)
        guard parts.count == 3,
              let blockIdx = Int(parts[1]) else { return [] }

        let rest = String(parts[2])
        let prefix = "single_transformer_blocks.\(blockIdx)"

        if rest == "linear1" {
            return ["\(prefix).attn.to_qkv_mlp_proj"]
        }
        if rest == "linear2" {
            return ["\(prefix).attn.to_out"]
        }

        return []
    }

    // MARK: - LoKR Delta Computation

    /// Compute the weight delta from a LoKR decomposition.
    ///
    /// LoKR represents the weight delta as the Kronecker product of two smaller matrices.
    /// For **non-decomposed** LoKR (full w1/w2, no w1_a/w1_b factorization):
    ///   `delta = kron(w1, w2) * user_scale`
    /// Alpha is NOT applied — the stored alpha is metadata only.
    ///
    /// For **decomposed** LoKR (w1 = w1_a @ w1_b):
    ///   `delta = kron(w1_a @ w1_b, w2) * (alpha / dim) * user_scale`
    ///
    /// This matches ComfyUI (comfy/weight_adapter/lokr.py) and LyCORIS behavior:
    /// when both w1/w2 are full matrices, `dim` is None and alpha is forced to 1.0.
    ///
    /// For 2D w1 [a, b] and w2 [c, d]: result is [a*c, b*d].
    private static func computeLoKrDelta(
        w1: MLXArray,
        w2: MLXArray,
        alpha: Float?,
        scale: Float
    ) -> MLXArray {
        // Ensure both are 2D
        let w1_2d = w1.ndim == 1 ? w1.reshaped(-1, 1) : w1
        let w2_2d = w2.ndim == 1 ? w2.reshaped(-1, 1) : w2

        // Compute Kronecker product
        let kron = kroneckerProduct(w1_2d, w2_2d)

        // Non-decomposed LoKR: alpha is ignored (forced to 1.0).
        // Per ComfyUI and LyCORIS: stored alpha is metadata for non-decomposed entries.
        // Only user-specified scale is applied.
        return kron * scale
    }

    /// Compute the Kronecker product of two 2D matrices.
    ///
    /// For A [m, n] and B [p, q]: result is [m*p, n*q] where each element
    /// A[i,j] is replaced by A[i,j] * B.
    private static func kroneckerProduct(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let m = a.dim(0), n = a.dim(1)
        let p = b.dim(0), q = b.dim(1)

        // Reshape for broadcasting: a[m,1,n,1] * b[1,p,1,q] -> [m,p,n,q]
        let aExp = a.reshaped(m, 1, n, 1)
        let bExp = b.reshaped(1, p, 1, q)
        let product = aExp * bExp

        // Reshape to [m*p, n*q]
        return product.reshaped(m * p, n * q)
    }

    // MARK: - Standard LoRA Delta

    /// Compute weight delta from standard LoRA pair.
    ///
    /// Standard LoRA: delta = up @ down * scale
    /// Handles orientation normalization.
    private static func computeLoRADelta(
        down: MLXArray,
        up: MLXArray,
        scale: Float
    ) -> MLXArray? {
        guard down.ndim == 2, up.ndim == 2 else { return nil }

        // Standard orientation: down=[rank, in], up=[out, rank]
        // delta = up @ down = [out, in]
        let rank = min(down.dim(0), down.dim(1))

        // Determine orientation
        let delta: MLXArray
        if down.dim(0) < down.dim(1) && up.dim(1) == down.dim(0) {
            // Standard: down=[rank, in], up=[out, rank]
            delta = MLX.matmul(up, down)
        } else if down.dim(1) < down.dim(0) && up.dim(0) == down.dim(1) {
            // Both transposed: down=[in, rank], up=[rank, out]
            delta = MLX.matmul(up.T, down.T)
        } else {
            // Try direct multiplication
            delta = MLX.matmul(up, down)
        }

        // Scale: typically scale * alpha / rank, but alpha is baked into scale for simple LoRA
        return delta * scale
    }
}
