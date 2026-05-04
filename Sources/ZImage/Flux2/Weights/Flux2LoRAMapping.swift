// Flux2LoRAMapping.swift — LoRA weight mapping for Flux 2 Klein
//
// Maps CivitAI / BFL / Kohya LoRA adapter keys to Flux2Transformer module paths.
// Handles fused QKV keys (split into separate Q/K/V targets) and
// Kohya-style underscore naming conventions.

import Foundation

/// Maps LoRA adapter base keys to Flux2Transformer Swift module paths.
///
/// LoRA adapters for Klein 9B use various naming conventions for their keys.
/// After stripping the `diffusion_model.`/`base_model.model.`/`lora_unet_` prefix
/// and the `lora_A`/`lora_B` suffix, the remaining "base key" is mapped here.
///
/// ## Fused QKV Keys
///
/// Some LoRAs store Q/K/V as a single fused tensor (e.g., `double_blocks.0.img_attn.qkv`).
/// These are mapped to a ``QKVSplitTarget`` with three separate module paths.
/// The caller must split `lora_B` along dim 0 into three equal parts.
///
/// ## Key Conventions Handled
///
/// - **BFL/diffusers dot notation**: `double_blocks.0.img_attn.qkv`
/// - **Kohya underscore notation**: `lora_unet_double_blocks_0_img_attn_qkv`
///   (converted to dot notation before mapping)
public enum Flux2LoRAMapping {

  /// A fused QKV target that requires splitting lora_B into 3 parts.
  public struct QKVSplitTarget {
    public let qPath: String
    public let kPath: String
    public let vPath: String
  }

  /// Result of mapping a LoRA base key.
  public enum MappingResult {
    /// Direct 1:1 mapping — use the LoRA pair as-is.
    case direct(String)
    /// Fused QKV — caller must split lora_B into 3 parts along dim 0.
    case qkvSplit(QKVSplitTarget)
    /// Key not recognized.
    case unmapped
  }

  /// Prefixes stripped before mapping (order matters — longest/most specific first).
  private static let prefixesToStrip = [
    "base_model.model.diffusion_model.",
    "base_model.model.",
    "diffusion_model.",
    "lora_unet_",
    "transformer.",
  ]

  /// Convert a Kohya-style underscore key to dot notation.
  ///
  /// Kohya keys look like `double_blocks_0_img_attn_qkv`. The tricky part is
  /// distinguishing word boundaries from numeric indices. Strategy:
  /// - Known compound tokens are preserved (img_attn, img_mlp, txt_attn, txt_mlp)
  /// - Numeric components become `.N`
  /// - Remaining underscores become dots
  private static func kohyaToDot(_ key: String) -> String {
    // Replace known compound tokens first to protect them
    var result = key
    let compounds = [
      ("img_attn", "img_attn"),
      ("txt_attn", "txt_attn"),
      ("img_mlp", "img_mlp"),
      ("txt_mlp", "txt_mlp"),
      ("single_blocks", "single_blocks"),
      ("double_blocks", "double_blocks"),
      ("to_qkv_mlp_proj", "to_qkv_mlp_proj"),
      ("add_q_proj", "add_q_proj"),
      ("add_k_proj", "add_k_proj"),
      ("add_v_proj", "add_v_proj"),
      ("to_add_out", "to_add_out"),
      ("linear_in", "linear_in"),
      ("linear_out", "linear_out"),
      ("to_out", "to_out"),
      ("to_q", "to_q"),
      ("to_k", "to_k"),
      ("to_v", "to_v"),
    ]

    // Simple approach: split on underscore, recombine intelligently
    // For Kohya keys: lora_unet_double_blocks_0_img_attn_qkv
    // After prefix strip: double_blocks_0_img_attn_qkv
    // Target: double_blocks.0.img_attn.qkv

    // Replace block type + index pattern
    let blockPattern = try! NSRegularExpression(pattern: "(double_blocks|single_blocks)_(\\d+)_(.+)")
    let range = NSRange(result.startIndex..., in: result)
    if let match = blockPattern.firstMatch(in: result, range: range) {
      let blockType = String(result[Range(match.range(at: 1), in: result)!])
      let index = String(result[Range(match.range(at: 2), in: result)!])
      let remainder = String(result[Range(match.range(at: 3), in: result)!])

      // Map remainder underscore components
      let mappedRemainder = mapKohyaRemainder(remainder)
      result = "\(blockType).\(index).\(mappedRemainder)"
    }

    return result
  }

  /// Map the remainder portion of a Kohya key (after block type and index).
  private static func mapKohyaRemainder(_ remainder: String) -> String {
    // Known remainder patterns (underscore -> dot-separated)
    let patterns: [(String, String)] = [
      ("img_attn_qkv", "img_attn.qkv"),
      ("img_attn_proj", "img_attn.proj"),
      ("txt_attn_qkv", "txt_attn.qkv"),
      ("txt_attn_proj", "txt_attn.proj"),
      ("img_mlp_0", "img_mlp.0"),
      ("img_mlp_2", "img_mlp.2"),
      ("txt_mlp_0", "txt_mlp.0"),
      ("txt_mlp_2", "txt_mlp.2"),
      ("linear1", "linear1"),
      ("linear2", "linear2"),
    ]

    for (pattern, replacement) in patterns {
      if remainder == pattern {
        return replacement
      }
    }

    // Fallback: replace underscores with dots
    return remainder.replacingOccurrences(of: "_", with: ".")
  }

  /// Strip known prefixes from a raw LoRA key.
  public static func stripPrefix(_ rawKey: String) -> String {
    var key = rawKey
    for prefix in prefixesToStrip {
      if key.hasPrefix(prefix) {
        key = String(key.dropFirst(prefix.count))
        break
      }
    }
    return key
  }

  /// Map a LoRA base key (after prefix/suffix stripping) to Flux2Transformer module path(s).
  ///
  /// - Parameter baseKey: The base key with prefixes stripped and lora_A/lora_B suffix removed.
  /// - Returns: The mapping result indicating direct, QKV split, or unmapped.
  public static func map(_ baseKey: String) -> MappingResult {
    var key = baseKey

    // Handle Kohya underscore notation: if key has underscores but no dots,
    // convert to dot notation first.
    if key.contains("_") && !key.contains(".") {
      key = kohyaToDot(key)
    }

    // Also strip any remaining known prefixes (in case they weren't caught earlier)
    key = stripPrefix(key)

    // --- Double blocks ---
    // Pattern: double_blocks.{i}.{component}
    if key.hasPrefix("double_blocks.") {
      return mapDoubleBlock(key)
    }

    // --- Single blocks ---
    // Pattern: single_blocks.{i}.{component}
    if key.hasPrefix("single_blocks.") {
      return mapSingleBlock(key)
    }

    return .unmapped
  }

  // MARK: - Double Block Mapping

  private static func mapDoubleBlock(_ key: String) -> MappingResult {
    // Extract: double_blocks.{i}.{rest}
    let parts = key.split(separator: ".", maxSplits: 3)
    guard parts.count >= 3,
          let blockIndex = Int(parts[1]) else {
      return .unmapped
    }
    let rest = parts.count > 2 ? parts[2...].joined(separator: ".") : ""
    let prefix = "transformer_blocks.\(blockIndex)"

    switch rest {
    // --- Image attention (fused QKV) ---
    case "img_attn.qkv":
      return .qkvSplit(QKVSplitTarget(
        qPath: "\(prefix).attn.to_q",
        kPath: "\(prefix).attn.to_k",
        vPath: "\(prefix).attn.to_v"
      ))

    // --- Image attention (already split) ---
    case "img_attn.to_q", "attn.to_q":
      return .direct("\(prefix).attn.to_q")
    case "img_attn.to_k", "attn.to_k":
      return .direct("\(prefix).attn.to_k")
    case "img_attn.to_v", "attn.to_v":
      return .direct("\(prefix).attn.to_v")

    // --- Image attention output ---
    case "img_attn.proj", "attn.to_out":
      return .direct("\(prefix).attn.to_out")

    // --- Text attention (fused QKV) ---
    case "txt_attn.qkv":
      return .qkvSplit(QKVSplitTarget(
        qPath: "\(prefix).attn.add_q_proj",
        kPath: "\(prefix).attn.add_k_proj",
        vPath: "\(prefix).attn.add_v_proj"
      ))

    // --- Text attention (already split) ---
    case "txt_attn.to_q", "attn.add_q_proj":
      return .direct("\(prefix).attn.add_q_proj")
    case "txt_attn.to_k", "attn.add_k_proj":
      return .direct("\(prefix).attn.add_k_proj")
    case "txt_attn.to_v", "attn.add_v_proj":
      return .direct("\(prefix).attn.add_v_proj")

    // --- Text attention output ---
    case "txt_attn.proj", "attn.to_add_out":
      return .direct("\(prefix).attn.to_add_out")

    // --- Image feed-forward ---
    case "img_mlp.0", "ff.linear_in":
      return .direct("\(prefix).ff.linear_in")
    case "img_mlp.2", "ff.linear_out":
      return .direct("\(prefix).ff.linear_out")

    // --- Text feed-forward ---
    case "txt_mlp.0", "ff_context.linear_in":
      return .direct("\(prefix).ff_context.linear_in")
    case "txt_mlp.2", "ff_context.linear_out":
      return .direct("\(prefix).ff_context.linear_out")

    default:
      return .unmapped
    }
  }

  // MARK: - Single Block Mapping

  private static func mapSingleBlock(_ key: String) -> MappingResult {
    // Extract: single_blocks.{i}.{rest}
    let parts = key.split(separator: ".", maxSplits: 3)
    guard parts.count >= 3,
          let blockIndex = Int(parts[1]) else {
      return .unmapped
    }
    let rest = parts.count > 2 ? parts[2...].joined(separator: ".") : ""
    let prefix = "single_transformer_blocks.\(blockIndex)"

    switch rest {
    // --- Fused QKV+MLP projection (direct, no split needed) ---
    case "linear1", "attn.to_qkv_mlp_proj":
      return .direct("\(prefix).attn.to_qkv_mlp_proj")

    // --- Output projection ---
    case "linear2", "attn.to_out":
      return .direct("\(prefix).attn.to_out")

    default:
      return .unmapped
    }
  }
}
