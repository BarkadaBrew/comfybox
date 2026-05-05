// ChromaTokenizer.swift — T5-XXL tokenizer wrapper for Chroma pipeline
// Uses swift-transformers AutoTokenizer with HuggingFace tokenizer.json format

import Foundation
import MLX
import Tokenizers
import Hub

/// T5-XXL tokenizer for Chroma text encoding.
///
/// Wraps swift-transformers' AutoTokenizer loaded from tokenizer.json.
/// T5 uses SentencePiece Unigram tokenization with vocab_size=32128,
/// pad_token=<pad>(0), eos_token=</s>(1), unk_token=<unk>(2).
public final class ChromaTokenizer {
  private let tokenizer: Tokenizer

  /// Pad token ID (0 for T5)
  public let padTokenId: Int

  /// EOS token ID (1 for T5)
  public let eosTokenId: Int

  /// Maximum sequence length for T5-XXL conditioning
  public let maxLength: Int

  public init(tokenizer: Tokenizer, maxLength: Int = 512) {
    self.tokenizer = tokenizer
    self.padTokenId = tokenizer.convertTokenToId("<pad>") ?? 0
    self.eosTokenId = tokenizer.convertTokenToId("</s>") ?? 1
    self.maxLength = maxLength
  }

  /// Load T5 tokenizer from a directory containing tokenizer.json and tokenizer_config.json.
  ///
  /// - Parameters:
  ///   - directory: Path to tokenizer directory (e.g., `t5/tokenizer_2/`)
  ///   - maxLength: Maximum token sequence length (default 512 for T5-XXL)
  /// - Returns: Configured ChromaTokenizer
  public static func load(from directory: URL, maxLength: Int = 512) throws -> ChromaTokenizer {
    let hubApi = HubApi()

    let configURL = directory.appending(path: "tokenizer_config.json")
    let dataURL = directory.appending(path: "tokenizer.json")

    guard FileManager.default.fileExists(atPath: configURL.path) else {
      throw ChromaTokenizerError.fileNotFound(configURL)
    }
    guard FileManager.default.fileExists(atPath: dataURL.path) else {
      throw ChromaTokenizerError.fileNotFound(dataURL)
    }

    let tokenizerConfig = try hubApi.configuration(fileURL: configURL)
    let tokenizerData = try hubApi.configuration(fileURL: dataURL)
    let tokenizer = try AutoTokenizer.from(
      tokenizerConfig: tokenizerConfig,
      tokenizerData: tokenizerData
    )

    return ChromaTokenizer(tokenizer: tokenizer, maxLength: maxLength)
  }

  /// Encode a text prompt for T5-XXL conditioning.
  ///
  /// Tokenizes the prompt, appends EOS if not already present,
  /// and pads to maxLength. Returns token IDs as `[1, seqLen]` MLXArray.
  ///
  /// - Parameter prompt: The text prompt to encode
  /// - Returns: Token IDs as `[1, maxLength]` MLXArray of Int32
  public func encode(prompt: String) -> MLXArray {
    var ids = tokenizer.encode(text: prompt)

    // Ensure EOS at end
    if ids.last != eosTokenId {
      ids.append(eosTokenId)
    }

    // Truncate or pad to maxLength
    if ids.count > maxLength {
      ids = Array(ids.prefix(maxLength - 1)) + [eosTokenId]
    } else if ids.count < maxLength {
      ids += Array(repeating: padTokenId, count: maxLength - ids.count)
    }

    return MLXArray(ids.map { Int32($0) }).reshaped(1, maxLength)
  }

  /// Decode token IDs back to text (for debugging).
  public func decode(tokens: [Int]) -> String {
    tokenizer.decode(tokens: tokens)
  }
}

public enum ChromaTokenizerError: Error, LocalizedError {
  case fileNotFound(URL)
  case loadFailed(String)

  public var errorDescription: String? {
    switch self {
    case .fileNotFound(let url): return "Tokenizer file not found: \(url.path)"
    case .loadFailed(let msg): return "Tokenizer load failed: \(msg)"
    }
  }
}
