// Krea2VAEKeyMap.swift — The Wan 2.1 ↔ Qwen-Image VAE key bijection (WP-E9, FDD §3.9, D16).
//
// `Krea2VAE` is modelled on diffusers' `AutoencoderKLQwenImage` (keys
// `decoder.up_blocks.N.resnets.M.*`). The reference stack decodes through the
// Wan 2.1 FP32 file, whose tensors carry Wan's native module names
// (`decoder.upsamples.N.residual.{0,2,3,6}.*`) — the two files share ZERO key
// names, and the same architecture. This map is the complete bijection:
// 194/194 Wan keys onto Krea2VAE paths with exact shape equality, 194/194
// Qwen keys covered (asserted against both real files by Krea2VAEKeyMapTests).
//
// Detection is by key sniff, never by filename (AC-53).

import Foundation

/// The on-disk key space a Krea-2 VAE file uses. Recorded on every render
/// that names a VAE (`RenderRecipe.vae.layout`, WP-E10).
public enum VAELayout: String, Sendable, Codable, Equatable, CaseIterable {
  /// diffusers `AutoencoderKLQwenImage` — the Krea-2 snapshot's `vae/` file.
  case qwenDiffusers
  /// Wan-AI's native module names — `Wan2_1_VAE_fp32.safetensors`.
  case wanNative
}

public enum Krea2VAEKeyMapError: Error, Equatable, LocalizedError {
  /// Neither `decoder.upsamples.` nor `decoder.up_blocks.` (or both) — the
  /// file is not a Krea-2-compatible VAE in a key space we can name.
  case unrecognizedVAELayout(file: String)
  /// The caller declared one layout and the keys sniff as another.
  case layoutMismatch(file: String, requested: VAELayout, detected: VAELayout)
  /// A key in the file has no Krea2VAE path (the map is total over the two
  /// real files; this fires for a file that only looks like one of them).
  case unmappedKey(file: String, key: String)

  public var errorDescription: String? {
    switch self {
    case .unrecognizedVAELayout(let file):
      return "Krea2 VAE: \(file) is neither a Qwen-Image (decoder.up_blocks.*) nor a Wan 2.1 (decoder.upsamples.*) VAE — refusing to guess the layout"
    case .layoutMismatch(let file, let requested, let detected):
      return "Krea2 VAE: \(file) was requested as \(requested.rawValue) but its keys are \(detected.rawValue)"
    case .unmappedKey(let file, let key):
      return "Krea2 VAE: \(file) key '\(key)' has no Krea2VAE parameter path"
    }
  }
}

public enum Krea2VAEKeyMap {

  // MARK: - Layout detection

  /// Sniff the layout from tensor names. `nil` when neither marker is present
  /// — or both are (a file is one key space, not a mixture).
  public static func detectLayout(keys: some Sequence<String>) -> VAELayout? {
    var wan = false, qwen = false
    for key in keys {
      if key.hasPrefix("decoder.upsamples.") { wan = true }
      if key.hasPrefix("decoder.up_blocks.") { qwen = true }
    }
    switch (wan, qwen) {
    case (true, false): return .wanNative
    case (false, true): return .qwenDiffusers
    default: return nil
    }
  }

  /// Sniff a file's layout from its safetensors header (no tensor data is read).
  public static func detectLayout(file: URL) throws -> VAELayout {
    let reader = try SafeTensorsReader(fileURL: file)
    return try detectLayout(keys: reader.tensorNames, file: file)
  }

  /// Fail-loud variant of `detectLayout(keys:)` that names the file.
  public static func detectLayout(keys: some Sequence<String>, file: URL) throws -> VAELayout {
    guard let layout = detectLayout(keys: keys) else {
      throw Krea2VAEKeyMapError.unrecognizedVAELayout(file: file.path)
    }
    return layout
  }

  // MARK: - Canonicalisation

  /// Map a tensor name from either layout onto its Krea2VAE (Qwen-diffusers)
  /// path. Qwen keys are returned unchanged; Wan keys are translated; anything
  /// else is `nil`. The result still carries the checkpoint-side spellings the
  /// loader rewrites (`.resample.1.` → `.resample.conv.`, `time_conv` skipped).
  public static func canonicalize(_ key: String) -> String? {
    if isQwenDiffusersKey(key) { return key }
    return canonicalizeWanNative(key)
  }

  /// The ~20 Wan-native → Qwen-diffusers rules (Draft B's table, verified
  /// against both real headers this session).
  static func canonicalizeWanNative(_ key: String) -> String? {
    // Top level: conv1 = quant_conv (32→32), conv2 = post_quant_conv (16→16).
    if let rest = strip("conv1.", key) { return "quant_conv." + rest }
    if let rest = strip("conv2.", key) { return "post_quant_conv." + rest }

    for side in ["encoder", "decoder"] {
      guard let body = strip(side + ".", key) else { continue }
      // Stem / head.
      if let rest = strip("conv1.", body) { return "\(side).conv_in." + rest }
      if body == "head.0.gamma" { return "\(side).norm_out.gamma" }
      if let rest = strip("head.2.", body) { return "\(side).conv_out." + rest }
      // Mid block: middle.0 / middle.2 are the two resnets, middle.1 the attention.
      if let rest = strip("middle.0.residual.", body) {
        return residual(rest).map { "\(side).mid_block.resnets.0." + $0 }
      }
      if let rest = strip("middle.2.residual.", body) {
        return residual(rest).map { "\(side).mid_block.resnets.1." + $0 }
      }
      if let rest = strip("middle.1.", body) {
        for leaf in ["norm.", "proj.", "to_qkv."] where rest.hasPrefix(leaf) {
          return "\(side).mid_block.attentions.0." + rest
        }
        return nil
      }
      if side == "encoder", let rest = strip("downsamples.", body) {
        return encoderDown(rest)
      }
      if side == "decoder", let rest = strip("upsamples.", body) {
        return decoderUp(rest)
      }
      return nil
    }
    return nil
  }

  /// `encoder.downsamples.N.*` → `encoder.down_blocks.N.*` — the encoder list
  /// is flat in both layouts, so only the leaf names change.
  private static func encoderDown(_ rest: String) -> String? {
    guard let (index, leaf) = splitIndex(rest) else { return nil }
    let prefix = "encoder.down_blocks.\(index)."
    if let r = strip("residual.", leaf) { return residual(r).map { prefix + $0 } }
    if let r = strip("shortcut.", leaf) { return prefix + "conv_shortcut." + r }
    if leaf.hasPrefix("resample.1.") || leaf.hasPrefix("time_conv.") { return prefix + leaf }
    return nil
  }

  /// `decoder.upsamples.N.*` (flat, 0…14) → `decoder.up_blocks.B.resnets.R.*`
  /// / `decoder.up_blocks.B.upsamplers.0.*`. Krea2VAEDecoder has four up
  /// blocks of three resnets; blocks 0–2 end in an upsampler, so the flat
  /// index splits as B = N / 4, slot = N % 4 (slot 3 = the upsampler).
  private static func decoderUp(_ rest: String) -> String? {
    guard let (index, leaf) = splitIndex(rest), index >= 0, index < 15 else { return nil }
    let block = index / 4, slot = index % 4
    if slot < 3 {
      let prefix = "decoder.up_blocks.\(block).resnets.\(slot)."
      if let r = strip("residual.", leaf) { return residual(r).map { prefix + $0 } }
      if let r = strip("shortcut.", leaf) { return prefix + "conv_shortcut." + r }
      return nil
    }
    let prefix = "decoder.up_blocks.\(block).upsamplers.0."
    if leaf.hasPrefix("resample.1.") || leaf.hasPrefix("time_conv.") { return prefix + leaf }
    return nil
  }

  /// Wan's `residual` Sequential: 0 = RMS norm, 1 = SiLU, 2 = conv, 3 = RMS
  /// norm, 4 = SiLU, 5 = dropout, 6 = conv. Only 0/2/3/6 carry weights.
  private static func residual(_ rest: String) -> String? {
    if rest == "0.gamma" { return "norm1.gamma" }
    if let r = strip("2.", rest) { return "conv1." + r }
    if rest == "3.gamma" { return "norm2.gamma" }
    if let r = strip("6.", rest) { return "conv2." + r }
    return nil
  }

  /// A recogniser for the Qwen-diffusers key space (so canonicalize is the
  /// identity on it, and `nil` — not a guess — on anything else).
  static func isQwenDiffusersKey(_ key: String) -> Bool {
    if key.hasPrefix("quant_conv.") || key.hasPrefix("post_quant_conv.") { return true }
    for side in ["encoder", "decoder"] {
      guard let body = strip(side + ".", key) else { continue }
      if body.hasPrefix("conv_in.") || body.hasPrefix("conv_out.") || body == "norm_out.gamma" { return true }
      if let rest = strip("mid_block.", body) {
        return rest.hasPrefix("resnets.0.") || rest.hasPrefix("resnets.1.") || rest.hasPrefix("attentions.0.")
      }
      if side == "encoder", let rest = strip("down_blocks.", body), let (_, leaf) = splitIndex(rest) {
        return leaf.hasPrefix("norm1.") || leaf.hasPrefix("conv1.") || leaf.hasPrefix("norm2.")
          || leaf.hasPrefix("conv2.") || leaf.hasPrefix("conv_shortcut.")
          || leaf.hasPrefix("resample.1.") || leaf.hasPrefix("time_conv.")
      }
      if side == "decoder", let rest = strip("up_blocks.", body), let (_, leaf) = splitIndex(rest) {
        return leaf.hasPrefix("resnets.") || leaf.hasPrefix("upsamplers.0.")
      }
      return false
    }
    return false
  }

  // MARK: - Helpers

  private static func strip(_ prefix: String, _ key: String) -> String? {
    key.hasPrefix(prefix) ? String(key.dropFirst(prefix.count)) : nil
  }

  /// "12.residual.0.gamma" → (12, "residual.0.gamma")
  private static func splitIndex(_ s: String) -> (Int, String)? {
    guard let dot = s.firstIndex(of: "."), let index = Int(s[s.startIndex..<dot]) else { return nil }
    return (index, String(s[s.index(after: dot)...]))
  }
}
