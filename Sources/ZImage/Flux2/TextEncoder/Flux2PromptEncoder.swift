// Flux2PromptEncoder.swift — Tokenization + encoding for Flux 2 Klein
// Ported from mflux: prompt_encoder.py

import MLX

/// Encodes text prompts for the Flux 2 Klein diffusion model.
///
/// Tokenizes the input prompt using the existing `QwenTokenizer`, runs it
/// through the `Qwen3TextEncoder`, and extracts multi-layer hidden states
/// (layers 9, 18, 27) to produce prompt embeddings for the transformer.
public enum Flux2PromptEncoder {

  /// Default hidden state extraction layers for Flux 2 Klein.
  public static let defaultOutputLayers: [Int] = [9, 18, 27]

  /// Encode a text prompt into prompt embeddings and text coordinate IDs.
  ///
  /// - Parameters:
  ///   - prompt: The text prompt (single string or array).
  ///   - tokenizer: A `QwenTokenizer` loaded from the Flux 2 Klein model.
  ///   - textEncoder: A `Qwen3TextEncoder` with loaded weights.
  ///   - numImagesPerPrompt: Batch repeat factor (default 1).
  ///   - maxSequenceLength: Maximum token sequence length (default 512).
  ///   - textEncoderOutLayers: Layer indices for hidden state extraction.
  /// - Returns: `(promptEmbeds, textIds)` where promptEmbeds is
  ///   `[B, S, numLayers * hiddenDim]` and textIds is `[B, S, 4]`.
  public static func encodePrompt(
    prompt: String,
    tokenizer: QwenTokenizer,
    textEncoder: Qwen3TextEncoder,
    numImagesPerPrompt: Int = 1,
    maxSequenceLength: Int = 512,
    textEncoderOutLayers: [Int] = defaultOutputLayers
  ) throws -> (promptEmbeds: MLXArray, textIds: MLXArray) {
    var promptEmbeds = try getQwen3PromptEmbeds(
      prompt: prompt,
      tokenizer: tokenizer,
      textEncoder: textEncoder,
      maxSequenceLength: maxSequenceLength,
      hiddenStateLayers: textEncoderOutLayers
    )

    if numImagesPerPrompt > 1 {
      promptEmbeds = MLX.repeated(promptEmbeds, count: numImagesPerPrompt, axis: 0)
    }

    let textIds = prepareTextIds(promptEmbeds)
    return (promptEmbeds, textIds)
  }

  /// Tokenize and encode a prompt, extracting multi-layer hidden states.
  private static func getQwen3PromptEmbeds(
    prompt: String,
    tokenizer: QwenTokenizer,
    textEncoder: Qwen3TextEncoder,
    maxSequenceLength: Int,
    hiddenStateLayers: [Int]
  ) throws -> MLXArray {
    let tokens = try tokenizer.encodeChat(
      prompts: [prompt],
      maxLength: maxSequenceLength
    )
    return textEncoder.getPromptEmbeds(
      inputIds: tokens.inputIds,
      attentionMask: tokens.attentionMask,
      hiddenStateLayers: hiddenStateLayers
    )
  }

  /// Prepare text coordinate IDs for the transformer.
  ///
  /// Each token gets a 4D coordinate `[t, h, w, tokenIndex]` where `t`, `h`,
  /// `w` are zero (text has no spatial position) and `tokenIndex` is the
  /// sequential position within the sequence.
  ///
  /// - Parameters:
  ///   - x: Prompt embeddings `[B, S, D]`, used only for shape.
  ///   - tCoord: Optional per-batch temporal coordinate override.
  /// - Returns: Text IDs `[B, S, 4]`.
  public static func prepareTextIds(
    _ x: MLXArray,
    tCoord: MLXArray? = nil
  ) -> MLXArray {
    let batchSize = x.dim(0)
    let seqLen = x.dim(1)

    var outIds: [MLXArray] = []
    outIds.reserveCapacity(batchSize)

    for i in 0..<batchSize {
      let t: MLXArray
      if let tCoord {
        let tc = tCoord[i]
        if tc.ndim == 0 {
          t = MLX.full([seqLen], values: tc, dtype: .int32)
        } else if tc.dim(0) != seqLen {
          t = MLX.broadcast(tc, to: [seqLen])
        } else {
          t = tc.asType(.int32)
        }
      } else {
        t = MLX.zeros([seqLen], dtype: .int32)
      }

      let h = MLX.zeros([seqLen], dtype: .int32)
      let w = MLX.zeros([seqLen], dtype: .int32)
      let tokenIds = MLXArray(0..<seqLen)

      // Stack [t, h, w, tokenIds] -> [S, 4]
      let coords = MLX.stacked([t, h, w, tokenIds], axis: 1)
      outIds.append(coords)
    }

    // Stack batches -> [B, S, 4]
    return MLX.stacked(outIds, axis: 0)
  }
}
