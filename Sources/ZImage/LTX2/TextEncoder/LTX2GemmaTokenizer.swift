// LTX2GemmaTokenizer.swift — Gemma 3 tokenizer for LTX-2 text encoding
// Phase 2 of the LTX-2 Swift/MLX port
//
// Wraps the swift-transformers library to provide Gemma 3 tokenization
// compatible with the LTX-2 text encoder pipeline.
//
// Key differences from the QwenTokenizer:
// - Left-padded (not right-padded) to match Python HuggingFace default for Gemma
// - pad_token_id = 0
// - bos_token_id = 2 (prepended automatically)
// - No chat template prefix/suffix — raw text tokenization
// - Default max_length = 128 (matching Python reference)

import Foundation
import Hub
import MLX
import Tokenizers

// MARK: - Gemma Token Batch

/// Tokenized batch output for Gemma 3.
public struct GemmaTokenBatch {
  /// Token IDs `[B, S]` as int32, left-padded.
  public let inputIds: MLXArray

  /// Attention mask `[B, S]` as int32 (1 = valid, 0 = pad).
  public let attentionMask: MLXArray
}

// MARK: - Gemma Tokenizer

/// Gemma 3 tokenizer for the LTX-2 text encoder pipeline.
///
/// Loads tokenizer configuration from a local HuggingFace model directory
/// (e.g. `~/.cache/huggingface/hub/models--unsloth--gemma-3-12b-it/snapshots/...`).
///
/// Produces LEFT-padded token batches matching the Python reference:
/// ```python
/// tokenizer(prompt, return_tensors="pt", padding="max_length",
///           max_length=128, truncation=True)
/// ```
public final class LTX2GemmaTokenizer {
  private let tokenizer: Tokenizer
  private let encodeFunction: (String) -> [Int]

  /// Pad token ID (0 for Gemma 3).
  public let padTokenId: Int

  /// BOS token ID (2 for Gemma 3).
  public let bosTokenId: Int

  /// Maximum sequence length.
  public let maxLength: Int

  public init(
    tokenizer: Tokenizer,
    padTokenId: Int = 0,
    bosTokenId: Int = 2,
    maxLength: Int = 128,
    encode: @escaping (String) -> [Int]
  ) {
    self.tokenizer = tokenizer
    self.padTokenId = padTokenId
    self.bosTokenId = bosTokenId
    self.maxLength = maxLength
    self.encodeFunction = encode
  }

  /// Load the Gemma tokenizer from a local HuggingFace model directory.
  ///
  /// The directory must contain `tokenizer_config.json`, `tokenizer.json`,
  /// and optionally `added_tokens.json`.
  ///
  /// - Parameters:
  ///   - directory: Path to the model directory.
  ///   - maxLength: Maximum sequence length (default 128).
  /// - Returns: Configured `LTX2GemmaTokenizer`.
  public static func load(
    from directory: URL,
    maxLength: Int = 128
  ) throws -> LTX2GemmaTokenizer {
    let tokenizerConfigURL = directory.appendingPathComponent("tokenizer_config.json")
    let tokenizerDataURL = directory.appendingPathComponent("tokenizer.json")

    guard FileManager.default.fileExists(atPath: directory.path) else {
      throw GemmaTokenizerError.directoryNotFound(directory)
    }
    guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
      throw GemmaTokenizerError.fileNotFound(tokenizerConfigURL)
    }
    guard FileManager.default.fileExists(atPath: tokenizerDataURL.path) else {
      throw GemmaTokenizerError.fileNotFound(tokenizerDataURL)
    }

    let hubApi = HubApi.shared
    let tokenizerConfig = try hubApi.configuration(fileURL: tokenizerConfigURL)
    let tokenizerData = try hubApi.configuration(fileURL: tokenizerDataURL)

    let tokenizer = try AutoTokenizer.from(
      tokenizerConfig: tokenizerConfig,
      tokenizerData: tokenizerData,
      strict: false  // Gemma may have non-standard tokenizer class name
    )

    // Resolve pad token ID
    let padTokenId: Int
    if let padStr = tokenizerConfig["pad_token"].string(),
      let padId = tokenizer.convertTokenToId(padStr)
    {
      padTokenId = padId
    } else {
      padTokenId = 0  // Gemma default
    }

    // Resolve BOS token ID
    let bosTokenId = tokenizer.bosTokenId ?? 2

    return LTX2GemmaTokenizer(
      tokenizer: tokenizer,
      padTokenId: padTokenId,
      bosTokenId: bosTokenId,
      maxLength: maxLength,
      encode: { text in tokenizer.encode(text: text) }
    )
  }

  /// Tokenize a prompt with LEFT padding to maxLength.
  ///
  /// Matches the Python tokenizer behavior:
  /// ```python
  /// tokenizer(prompt, return_tensors="pt", padding="max_length",
  ///           max_length=128, truncation=True)
  /// ```
  ///
  /// - Parameters:
  ///   - prompt: Text prompt to tokenize.
  ///   - maxLength: Override max length (default uses instance maxLength).
  /// - Returns: `GemmaTokenBatch` with left-padded input_ids and attention_mask.
  public func encode(prompt: String, maxLength: Int? = nil) -> GemmaTokenBatch {
    return encode(prompts: [prompt], maxLength: maxLength)
  }

  /// Tokenize multiple prompts with LEFT padding to maxLength.
  ///
  /// - Parameters:
  ///   - prompts: Text prompts to tokenize.
  ///   - maxLength: Override max length.
  /// - Returns: `GemmaTokenBatch` with left-padded input_ids and attention_mask.
  public func encode(prompts: [String], maxLength: Int? = nil) -> GemmaTokenBatch {
    precondition(!prompts.isEmpty, "At least one prompt must be provided.")

    let targetLength = min(maxLength ?? self.maxLength, self.maxLength)

    var allIds: [[Int]] = []
    var allMasks: [[Int]] = []

    for prompt in prompts {
      // Tokenize (swift-transformers handles BOS automatically for Gemma)
      var tokens = encodeFunction(prompt)

      // Truncate if needed
      if tokens.count > targetLength {
        tokens = Array(tokens.prefix(targetLength))
      }

      // LEFT padding: pad tokens go at the beginning
      let paddingCount = max(0, targetLength - tokens.count)
      let paddedIds = Array(repeating: padTokenId, count: paddingCount) + tokens
      let mask = Array(repeating: 0, count: paddingCount) + Array(repeating: 1, count: tokens.count)

      allIds.append(paddedIds)
      allMasks.append(mask)
    }

    let batchSize = prompts.count
    let flatIds = allIds.flatMap { $0 }
    let flatMask = allMasks.flatMap { $0 }

    let inputIds = MLXArray(flatIds.map { Int32($0) }, [batchSize, targetLength])
    let attentionMask = MLXArray(flatMask.map { Int32($0) }, [batchSize, targetLength])

    return GemmaTokenBatch(inputIds: inputIds, attentionMask: attentionMask)
  }

  /// Decode token IDs back to text.
  public func decode(tokens: [Int]) -> String {
    tokenizer.decode(tokens: tokens)
  }
}

// MARK: - Errors

public enum GemmaTokenizerError: Error, LocalizedError {
  case directoryNotFound(URL)
  case fileNotFound(URL)

  public var errorDescription: String? {
    switch self {
    case .directoryNotFound(let url):
      return "Gemma tokenizer directory not found: \(url.path)"
    case .fileNotFound(let url):
      return "Required tokenizer file not found: \(url.path)"
    }
  }
}
