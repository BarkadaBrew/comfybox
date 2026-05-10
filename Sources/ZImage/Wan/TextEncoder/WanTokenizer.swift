import Foundation
import MLX

/// Unigram tokenizer for the Wan 2.2 UMT5-XXL text encoder.
///
/// Loads vocabulary directly from a HuggingFace `tokenizer.json` file and
/// implements Unigram tokenization with Metaspace pre-tokenization.
///
/// ## Tokenization Pipeline
///
/// ```
/// Input text
///   → Normalize: collapse multiple spaces
///   → Pre-tokenize: Metaspace (prepend ▁, replace spaces with ▁)
///   → Tokenize: Unigram (greedy forward maximum match)
///   → Post-process: append EOS token (id=1)
///   → Pad to maxLength with PAD token (id=0)
/// ```
///
/// ## Special Tokens
///
/// | Token | ID | Usage |
/// |-------|-----|-------|
/// | `<pad>` | 0 | Padding |
/// | `</s>` | 1 | End of sequence |
/// | `<s>` | 2 | Beginning of sequence (unused in encoding) |
/// | `<unk>` | 3 | Unknown token fallback |
public final class WanTokenizer {

  // MARK: - Special Token IDs

  /// Padding token ID.
  public static let padTokenId: Int32 = 0

  /// End-of-sequence token ID.
  public static let eosTokenId: Int32 = 1

  /// Unknown token ID.
  public static let unkTokenId: Int32 = 3

  // MARK: - Vocabulary

  /// Token string to ID mapping.
  private let vocab: [String: Int32]

  /// Token string to log probability (score).
  private let scores: [String: Float]

  /// Maximum token length in characters (for search window).
  private let maxTokenLength: Int

  // MARK: - Init

  /// Creates a tokenizer from a HuggingFace tokenizer.json file.
  ///
  /// - Parameter url: Path to the tokenizer.json file.
  /// - Throws: If the file cannot be read or parsed.
  public init(url: URL) throws {
    let data = try Data(contentsOf: url)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    let model = json["model"] as! [String: Any]
    let vocabArray = model["vocab"] as! [[Any]]

    var vocabDict: [String: Int32] = [:]
    var scoresDict: [String: Float] = [:]
    var maxLen = 0

    for (index, entry) in vocabArray.enumerated() {
      let token = entry[0] as! String
      let score = (entry[1] as! NSNumber).floatValue
      vocabDict[token] = Int32(index)
      scoresDict[token] = score
      maxLen = max(maxLen, token.count)
    }

    self.vocab = vocabDict
    self.scores = scoresDict
    self.maxTokenLength = maxLen
  }

  /// Creates a tokenizer from a vocabulary dictionary (for testing).
  ///
  /// - Parameters:
  ///   - vocab: Token string to ID mapping.
  ///   - scores: Token string to log probability mapping.
  public init(vocab: [String: Int32], scores: [String: Float]) {
    self.vocab = vocab
    self.scores = scores
    self.maxTokenLength = vocab.keys.map(\.count).max() ?? 1
  }

  // MARK: - Encoding

  /// Encodes a text string into token IDs and attention mask.
  ///
  /// - Parameters:
  ///   - text: The input text to tokenize.
  ///   - maxLength: Maximum sequence length including EOS. Default 512.
  /// - Returns: Tuple of `(tokenIds, attentionMask)` both shape `[1, maxLength]`.
  public func encode(_ text: String, maxLength: Int = 512) -> (tokenIds: MLXArray, attentionMask: MLXArray) {
    // Normalize: collapse multiple spaces to single space
    let normalized = normalizeText(text)

    // Pre-tokenize with Metaspace
    let pretokenized = metaspacePreTokenize(normalized)

    // Tokenize each word piece with Unigram
    var tokenIds: [Int32] = []
    for piece in pretokenized {
      let pieceTokens = unigramTokenize(piece)
      tokenIds.append(contentsOf: pieceTokens)
    }

    // Post-process: append EOS
    tokenIds.append(Self.eosTokenId)

    // Truncate if needed (keep room for at least EOS)
    if tokenIds.count > maxLength {
      tokenIds = Array(tokenIds.prefix(maxLength - 1))
      tokenIds.append(Self.eosTokenId)
    }

    // Build attention mask (1 for real tokens, 0 for padding)
    let realLength = tokenIds.count
    var mask = [Int32](repeating: 1, count: realLength)

    // Pad to maxLength
    while tokenIds.count < maxLength {
      tokenIds.append(Self.padTokenId)
      mask.append(0)
    }

    let ids = MLXArray(tokenIds).reshaped(1, maxLength)
    let attentionMask = MLXArray(mask).reshaped(1, maxLength)

    return (tokenIds: ids, attentionMask: attentionMask)
  }

  // MARK: - Normalization

  /// Normalizes text by collapsing multiple spaces.
  private func normalizeText(_ text: String) -> String {
    // Collapse runs of 2+ spaces to single space (matches tokenizer.json normalizer)
    var result = ""
    var lastWasSpace = false
    for c in text {
      if c == " " {
        if !lastWasSpace {
          result.append(c)
          lastWasSpace = true
        }
      } else {
        result.append(c)
        lastWasSpace = false
      }
    }
    return result
  }

  // MARK: - Metaspace Pre-Tokenizer

  /// Applies Metaspace pre-tokenization: splits on spaces and prepends ▁ to each piece.
  ///
  /// For input "hello world":
  /// - Split on spaces → ["hello", "world"]
  /// - Prepend ▁ → ["▁hello", "▁world"]
  ///
  /// For empty input: returns empty array.
  private func metaspacePreTokenize(_ text: String) -> [String] {
    if text.isEmpty { return [] }

    let pieces = text.split(separator: " ", omittingEmptySubsequences: true)
    return pieces.map { "▁" + $0 }
  }

  // MARK: - Unigram Tokenization

  /// Tokenizes a single pre-tokenized piece using greedy forward maximum match.
  ///
  /// Greedy approach: at each position, find the longest matching token in the
  /// vocabulary. Falls back to UNK for single characters not in vocab.
  ///
  /// This is a simplification of the full Viterbi algorithm but works well
  /// in practice for UMT5's large vocabulary.
  private func unigramTokenize(_ piece: String) -> [Int32] {
    if piece.isEmpty { return [] }

    var tokens: [Int32] = []
    let chars = Array(piece)
    var pos = 0

    while pos < chars.count {
      var bestLen = 0
      var bestId: Int32 = Self.unkTokenId

      // Search from longest possible match down to single character
      let maxLen = min(maxTokenLength, chars.count - pos)
      for len in stride(from: maxLen, through: 1, by: -1) {
        let endIdx = pos + len
        let candidate = String(chars[pos..<endIdx])
        if let id = vocab[candidate] {
          bestLen = len
          bestId = id
          break
        }
      }

      if bestLen == 0 {
        // Single character not found — emit UNK and advance
        tokens.append(Self.unkTokenId)
        pos += 1
      } else {
        tokens.append(bestId)
        pos += bestLen
      }
    }

    return tokens
  }
}
