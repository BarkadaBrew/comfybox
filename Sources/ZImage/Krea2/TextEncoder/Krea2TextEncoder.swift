// Krea2TextEncoder.swift — Qwen3-VL-4B conditioner for Krea-2.
//
// Reuses Flux2's `Qwen3TextEncoder` (identical Qwen3 decoder arch) with Krea-2's
// config (RoPE θ=5e6) and Krea-2's conditioning recipe: a describe-image system
// prompt, then the 12 selected per-layer hidden states stacked (B, seq, 12, 2560)
// with the prefix sliced off — matching docs/krea2-reference/krea2/text_encoder.py.

import Foundation
import MLX

public enum Krea2TextEncoderFactory {
  /// Qwen3 config for Krea-2: same as the reference (vocab 151936, hidden 2560,
  /// 36 layers, 32/8 heads, head_dim 128, inter 9728, eps 1e-6) but RoPE θ=5e6.
  public static func configuration() -> Qwen3TextEncoderConfiguration {
    Qwen3TextEncoderConfiguration(ropeTheta: 5_000_000.0)
  }

  public static func makeEncoder() -> Qwen3TextEncoder {
    Qwen3TextEncoder(configuration: configuration())
  }
}

/// Conditioner: tokenizes with Krea-2's describe-image template and returns the
/// 12-layer stacked hidden states + attention mask for the SingleStreamDiT.
public final class Krea2TextConditioner {
  /// Encoder hidden-state layers Krea-2 feeds into the transformer's text fusion.
  public static let selectLayers: [Int] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]

  /// Describe-image system prompt + user turn (matches the reference exactly).
  public static let prefix =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    + "texture, quantity, text, spatial relationships of the objects and background:"
    + "<|im_end|>\n<|im_start|>user\n"
  public static let suffix = "<|im_end|>\n<|im_start|>assistant\n"
  /// Token count of `prefix` under the Qwen tokenizer (used to slice conditioning).
  public static let prefixTokenCount = 34

  private let encoder: Qwen3TextEncoder
  private let tokenizer: QwenTokenizer
  private let maxLength: Int

  public init(encoder: Qwen3TextEncoder, tokenizer: QwenTokenizer, maxLength: Int = 512) {
    self.encoder = encoder
    self.tokenizer = tokenizer
    self.maxLength = maxLength
  }

  /// - Returns: `(context, mask)` where context is `(B, L, 12, 2560)` and mask is
  ///   `(B, L)` — both already sliced past the describe-image prefix.
  public func encode(_ prompts: [String]) -> (context: MLXArray, mask: MLXArray) {
    // Assemble the full templated string per prompt: prefix + prompt + suffix.
    // (The reference pads between prompt and suffix; for a first parity pass we
    //  tokenize the joined string and rely on the attention mask — refine to the
    //  exact pad-before-suffix layout during golden-image verification.)
    let templated = prompts.map { Self.prefix + $0 + Self.suffix }
    let batch = tokenizer.encodePlain(prompts: templated, maxLength: maxLength + Self.prefixTokenCount)

    let (_, hiddenStatesOpt) = encoder(
      inputIds: batch.inputIds,
      attentionMask: batch.attentionMask,
      outputHiddenStates: true
    )
    guard let hs = hiddenStatesOpt else {
      fatalError("Krea2TextConditioner: encoder returned no hidden states")
    }

    // Stack the 12 selected layers -> (B, L, 12, D).
    let selected = Self.selectLayers.map { hs[$0] }  // each (B, L, D)
    let stacked = MLX.stacked(selected, axis: 2)

    // Slice off the describe-image prefix along the sequence axis.
    let ctx = stacked[0..., Self.prefixTokenCount..., 0..., 0...]
    let mask = batch.attentionMask[0..., Self.prefixTokenCount...].asType(.float32)
    return (ctx, mask)
  }
}
