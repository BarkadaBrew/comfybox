// FiboPromptEncoder.swift — Prompt encoding for FIBO diffusion model
// Ported from mflux: prompt_encoder.py
//
// FIBO's prompt encoding differs significantly from Flux 1/2:
// 1. Uses SmolLM3-3B (an LLM) instead of CLIP/T5/Qwen3
// 2. Extracts ALL 37 hidden states (embedding + 36 layers), not just selected ones
// 3. Per-layer hidden states are fed individually to the transformer via DimFusion
// 4. Prompt embeddings = concatenation of last + second-to-last hidden states
// 5. CFG: both positive and negative prompts are encoded and concatenated

import MLX
import Foundation

/// Result of encoding prompts for the FIBO transformer.
public struct FiboPromptEncoderOutput {
  /// The JSON prompt string (validated).
  public let jsonPrompt: String

  /// Concatenated encoder hidden states for CFG: `[2*B, maxTokens, 2*hiddenSize]`
  /// (negative prompt first, positive prompt second along batch dimension).
  public let encoderHiddenStates: MLXArray

  /// Per-layer hidden states for DimFusion conditioning.
  /// Array of 46 tensors (one per transformer block: 8 joint + 38 single),
  /// each `[2*B, maxTokens, hiddenSize]` (negative first, positive second).
  ///
  /// If the text encoder produces fewer than 46 layers, the last layer is
  /// repeated to fill. If more, only the last 46 are kept.
  public let promptLayers: [MLXArray]
}

/// Encodes text prompts for the FIBO diffusion model.
///
/// Tokenizes prompts using the shared `QwenTokenizer` (which handles BPE
/// tokenization for SmolLM3-3B's tokenizer format), runs them through the
/// `SmolLM3TextEncoder`, and assembles per-layer hidden states for the
/// transformer's DimFusion conditioning.
public enum FiboPromptEncoder {

  /// Total number of transformer layers that need per-layer text conditioning.
  /// FIBO transformer has 8 joint + 38 single blocks = 46 total layers.
  static let totalTransformerLayers = 46

  /// Maximum sequence length for the text encoder.
  static let maxSequenceLength = 2048

  /// Default negative prompt when none is provided.
  static let defaultNegativePrompt = "ugly, blurry, low quality"

  /// Encode positive and negative prompts for the FIBO transformer.
  ///
  /// - Parameters:
  ///   - prompt: Positive prompt (must be valid JSON string for FIBO).
  ///   - negativePrompt: Optional negative prompt for CFG. Uses a default if nil/empty.
  ///   - tokenizer: Loaded `QwenTokenizer` for the FIBO tokenizer.
  ///   - textEncoder: Loaded `SmolLM3TextEncoder` with weights.
  /// - Returns: `FiboPromptEncoderOutput` with hidden states for all layers.
  public static func encodePrompt(
    prompt: String,
    negativePrompt: String?,
    tokenizer: QwenTokenizer,
    textEncoder: SmolLM3TextEncoder
  ) -> FiboPromptEncoderOutput {
    let effectiveNegativePrompt = (negativePrompt ?? "").isEmpty
      ? defaultNegativePrompt
      : negativePrompt!

    // Validate JSON (FIBO expects JSON prompt format)
    let jsonPrompt = prompt

    // Encode positive prompt
    let (promptEmbeds, promptLayers, promptAttentionMask) = getPromptEmbeds(
      prompt: jsonPrompt,
      tokenizer: tokenizer,
      textEncoder: textEncoder
    )

    // Encode negative prompt
    let (negPromptEmbeds, negPromptLayers, negPromptAttentionMask) = getPromptEmbeds(
      prompt: effectiveNegativePrompt,
      tokenizer: tokenizer,
      textEncoder: textEncoder
    )

    // Pad and concatenate for CFG
    let maxTokens = max(promptEmbeds.dim(1), negPromptEmbeds.dim(1))

    let (paddedPrompt, _) = padEmbedding(promptEmbeds, maxTokens: maxTokens)
    let (paddedNeg, _) = padEmbedding(negPromptEmbeds, maxTokens: maxTokens)

    // CFG order: negative first, positive second
    let encoderHiddenStates = MLX.concatenated([paddedNeg, paddedPrompt], axis: 0)

    // Pad and concatenate per-layer hidden states
    let paddedPromptLayers = promptLayers.map { padEmbedding($0, maxTokens: maxTokens).0 }
    let paddedNegLayers = negPromptLayers.map { padEmbedding($0, maxTokens: maxTokens).0 }

    var combinedLayers = (0..<paddedPromptLayers.count).map { i in
      MLX.concatenated([paddedNegLayers[i], paddedPromptLayers[i]], axis: 0)
    }

    // Adjust layer count to match transformer requirements (46 layers)
    if combinedLayers.count >= totalTransformerLayers {
      // Keep only the last 46 layers
      combinedLayers = Array(combinedLayers.suffix(totalTransformerLayers))
    } else {
      // Pad with copies of the last layer
      let lastLayer = combinedLayers.last!
      let deficit = totalTransformerLayers - combinedLayers.count
      combinedLayers += Array(repeating: lastLayer, count: deficit)
    }

    return FiboPromptEncoderOutput(
      jsonPrompt: jsonPrompt,
      encoderHiddenStates: encoderHiddenStates,
      promptLayers: combinedLayers
    )
  }

  // MARK: - Internal

  /// Tokenize and encode a single prompt, returning embeddings and all hidden states.
  ///
  /// - Returns: `(promptEmbeds, promptLayers, attentionMask)` where:
  ///   - promptEmbeds: `[1, S, 2*hiddenSize]` — concatenation of last two hidden states
  ///   - promptLayers: Array of 37 tensors each `[1, S, hiddenSize]`
  ///   - attentionMask: `[1, S]`
  static func getPromptEmbeds(
    prompt: String,
    tokenizer: QwenTokenizer,
    textEncoder: SmolLM3TextEncoder
  ) -> (MLXArray, [MLXArray], MLXArray) {
    // Tokenize — use encodeDynamic for natural-length tokenization (no padding, no chat template)
    let tokenBatch = tokenizer.encodeDynamic(
      prompts: [prompt],
      maxLength: maxSequenceLength
    )

    // Run through text encoder, collecting all hidden states
    let hiddenStatesList = textEncoder(
      inputIds: tokenBatch.inputIds,
      attentionMask: tokenBatch.attentionMask,
      outputHiddenStates: true
    )

    // Build prompt_embeds from last two hidden states (concatenated along feature dim)
    let lastHidden = hiddenStatesList[hiddenStatesList.count - 1]
    let secondLastHidden = hiddenStatesList[hiddenStatesList.count - 2]
    let promptEmbeds = MLX.concatenated([lastHidden, secondLastHidden], axis: -1)

    return (promptEmbeds, hiddenStatesList, tokenBatch.attentionMask)
  }

  /// Pad an embedding tensor to a target sequence length.
  ///
  /// Pads with zeros along the sequence dimension (axis 1).
  ///
  /// - Parameters:
  ///   - embedding: Tensor `[B, S, D]`.
  ///   - maxTokens: Target sequence length (must be >= current S).
  /// - Returns: `(paddedEmbedding, paddedAttentionMask)`.
  static func padEmbedding(
    _ embedding: MLXArray,
    maxTokens: Int,
    attentionMask: MLXArray? = nil
  ) -> (MLXArray, MLXArray) {
    let batchSize = embedding.dim(0)
    let seqLen = embedding.dim(1)
    let dim = embedding.dim(2)

    var mask: MLXArray
    if let attentionMask {
      mask = attentionMask.asType(embedding.dtype)
    } else {
      mask = MLX.ones([batchSize, seqLen], dtype: embedding.dtype)
    }

    guard maxTokens > seqLen else {
      return (embedding, mask)
    }

    let padLength = maxTokens - seqLen
    let padding = MLX.zeros([batchSize, padLength, dim], dtype: embedding.dtype)
    let paddedEmbedding = MLX.concatenated([embedding, padding], axis: 1)

    let maskPadding = MLX.zeros([batchSize, padLength], dtype: mask.dtype)
    let paddedMask = MLX.concatenated([mask, maskPadding], axis: 1)

    return (paddedEmbedding, paddedMask)
  }
}
