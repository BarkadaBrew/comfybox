// ChromaLoRAKeyMapper.swift — LoRA key mapping for Chroma transformer architecture
//
// Maps LoRA safetensors keys (lora_unet_* format) to ChromaTransformer module paths.
// Chroma uses double_blocks/single_blocks naming with fused QKV and Sequential MLPs,
// which differs from the Flux/Z-Image naming convention.

import Foundation

/// Maps LoRA weight keys from safetensors files to ChromaTransformer module paths.
///
/// Chroma LoRA key format (after prefix strip):
/// ```
/// double_blocks_N_img_attn_qkv → double_blocks.N.img_attn.qkv
/// double_blocks_N_img_attn_proj → double_blocks.N.img_attn.proj
/// double_blocks_N_img_mlp_M → double_blocks.N.img_mlp.layers.M
/// double_blocks_N_txt_attn_qkv → double_blocks.N.txt_attn.qkv
/// double_blocks_N_txt_attn_proj → double_blocks.N.txt_attn.proj
/// double_blocks_N_txt_mlp_M → double_blocks.N.txt_mlp.layers.M
/// single_blocks_N_linear1 → single_blocks.N.linear1
/// single_blocks_N_linear2 → single_blocks.N.linear2
/// txt_in → txt_in
/// img_in → img_in
/// ```
///
/// The `layers.` segment is required because Sequential wraps its children
/// under `.layers.N` in MLX Swift's module tree.
public enum ChromaLoRAKeyMapper {

    /// Prefixes stripped from raw LoRA keys before mapping.
    private static let prefixesToStrip = [
        "base_model.model.",
        "diffusion_model.",
        "lora_unet_",
        "transformer.",
        "text_encoder.",
        "model.",
    ]

    /// Map a raw LoRA base key to a ChromaTransformer module path.
    ///
    /// - Parameter rawKey: The base key extracted from the LoRA file
    ///   (e.g., `lora_unet_double_blocks_0_img_attn_qkv`).
    /// - Returns: The mapped module path with `.weight` suffix
    ///   (e.g., `double_blocks.0.img_attn.qkv.weight`).
    public static func map(_ rawKey: String) -> String {
        var key = rawKey

        // Strip known prefixes
        for prefix in prefixesToStrip {
            if key.hasPrefix(prefix) {
                key = String(key.dropFirst(prefix.count))
                break
            }
        }

        // Double block attention: double_blocks_N_{img,txt}_attn_{qkv,proj}
        if let result = mapDoubleBlockAttn(key) {
            return result
        }

        // Double block MLP: double_blocks_N_{img,txt}_mlp_M
        if let result = mapDoubleBlockMLP(key) {
            return result
        }

        // Single block: single_blocks_N_{linear1,linear2}
        if let result = mapSingleBlock(key) {
            return result
        }

        // Top-level projections: txt_in, img_in
        if key == "txt_in" || key == "img_in" {
            return key + ".weight"
        }

        // Approximator (distilled_guidance_layer): rare but possible
        if key.hasPrefix("distilled_guidance_layer_") {
            let rest = String(key.dropFirst("distilled_guidance_layer_".count))
            return "distilled_guidance_layer." + rest.replacingOccurrences(of: "_", with: ".") + ".weight"
        }

        // Fallback: pass through with .weight suffix
        if !key.hasSuffix(".weight") && !key.hasSuffix(".bias") {
            return key + ".weight"
        }
        return key
    }

    // MARK: - Double Block Patterns

    /// Match `double_blocks_N_{img,txt}_attn_{qkv,proj}`
    private static func mapDoubleBlockAttn(_ key: String) -> String? {
        // Pattern: double_blocks_(\d+)_(img|txt)_attn_(qkv|proj)
        guard key.hasPrefix("double_blocks_") else { return nil }
        let rest = String(key.dropFirst("double_blocks_".count))

        guard let underscoreIdx = rest.firstIndex(of: "_") else { return nil }
        let blockIdxStr = String(rest[..<underscoreIdx])
        guard Int(blockIdxStr) != nil else { return nil }

        let afterIdx = String(rest[rest.index(after: underscoreIdx)...])

        // Check for img_attn or txt_attn
        for stream in ["img", "txt"] {
            let attnPrefix = "\(stream)_attn_"
            if afterIdx.hasPrefix(attnPrefix) {
                let component = String(afterIdx.dropFirst(attnPrefix.count))
                if component == "qkv" || component == "proj" {
                    return "double_blocks.\(blockIdxStr).\(stream)_attn.\(component).weight"
                }
            }
        }

        return nil
    }

    /// Match `double_blocks_N_{img,txt}_mlp_M`
    private static func mapDoubleBlockMLP(_ key: String) -> String? {
        guard key.hasPrefix("double_blocks_") else { return nil }
        let rest = String(key.dropFirst("double_blocks_".count))

        guard let underscoreIdx = rest.firstIndex(of: "_") else { return nil }
        let blockIdxStr = String(rest[..<underscoreIdx])
        guard Int(blockIdxStr) != nil else { return nil }

        let afterIdx = String(rest[rest.index(after: underscoreIdx)...])

        // Check for img_mlp or txt_mlp
        for stream in ["img", "txt"] {
            let mlpPrefix = "\(stream)_mlp_"
            if afterIdx.hasPrefix(mlpPrefix) {
                let layerStr = String(afterIdx.dropFirst(mlpPrefix.count))
                if Int(layerStr) != nil {
                    // Sequential wraps children as .layers.N
                    return "double_blocks.\(blockIdxStr).\(stream)_mlp.layers.\(layerStr).weight"
                }
            }
        }

        return nil
    }

    // MARK: - Single Block Pattern

    /// Match `single_blocks_N_{linear1,linear2}`
    private static func mapSingleBlock(_ key: String) -> String? {
        guard key.hasPrefix("single_blocks_") else { return nil }
        let rest = String(key.dropFirst("single_blocks_".count))

        guard let underscoreIdx = rest.firstIndex(of: "_") else { return nil }
        let blockIdxStr = String(rest[..<underscoreIdx])
        guard Int(blockIdxStr) != nil else { return nil }

        let component = String(rest[rest.index(after: underscoreIdx)...])
        if component == "linear1" || component == "linear2" {
            return "single_blocks.\(blockIdxStr).\(component).weight"
        }

        return nil
    }
}
