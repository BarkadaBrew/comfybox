// SmolLM3Layer.swift — Single transformer layer for SmolLM3-3B
// Ported from mflux: smol_lm3_3b_encoder_layer.py

import MLX
import MLXNN

/// Single pre-norm transformer layer for SmolLM3-3B.
///
/// Architecture:
/// ```
/// x -> RMSNorm -> GQA Attention (+ optional RoPE) -> residual add
/// x -> RMSNorm -> SwiGLU MLP -> residual add
/// ```
///
/// Weight key mapping (safetensors -> model, after `model.` prefix strip):
/// - `layers.N.input_layernorm.weight`
/// - `layers.N.self_attn.{q,k,v,o}_proj.weight`
/// - `layers.N.post_attention_layernorm.weight`
/// - `layers.N.mlp.{gate,up,down}_proj.weight`
public final class SmolLM3Layer: Module {
  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: SmolLM3RMSNorm
  @ModuleInfo(key: "self_attn") var selfAttn: SmolLM3Attention
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: SmolLM3RMSNorm
  let mlp: SmolLM3MLP

  public init(config: FiboTextEncoderConfig) {
    self._inputLayerNorm.wrappedValue = SmolLM3RMSNorm(
      hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
    self._selfAttn.wrappedValue = SmolLM3Attention(config: config)
    self._postAttentionLayerNorm.wrappedValue = SmolLM3RMSNorm(
      hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
    self.mlp = SmolLM3MLP(
      hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
  }

  /// Forward pass through the transformer layer.
  ///
  /// - Parameters:
  ///   - hiddenStates: Input tensor `[B, S, hiddenSize]`.
  ///   - attentionMask: 4D additive mask `[B, 1, S, S]`.
  ///   - cosSin: `(cos, sin)` from `SmolLM3RoPE`. Pass `nil` for no-RoPE layers.
  /// - Returns: Output tensor `[B, S, hiddenSize]`.
  public func callAsFunction(
    hiddenStates: MLXArray,
    attentionMask: MLXArray?,
    cosSin: (cos: MLXArray, sin: MLXArray)?
  ) -> MLXArray {
    // Self-attention block with pre-norm residual
    let residual1 = hiddenStates
    var h = inputLayerNorm(hiddenStates)
    h = selfAttn(hiddenStates: h, attentionMask: attentionMask, cosSin: cosSin)
    h = residual1 + h

    // Feed-forward block with pre-norm residual
    let residual2 = h
    h = postAttentionLayerNorm(h)
    h = mlp(h)
    h = residual2 + h

    return h
  }
}
