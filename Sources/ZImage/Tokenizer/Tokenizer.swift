import Foundation
import MLX
import Tokenizers
import Hub

public enum QwenTokenizerError: Error {
  case directoryNotFound(URL)
  case fileNotFound(URL)
  case padTokenMissing
  case padTokenNotInVocabulary(String)
}

public struct QwenTokenBatch {
  public let inputIds: MLXArray
  public let attentionMask: MLXArray

  public init(inputIds: MLXArray, attentionMask: MLXArray) {
    self.inputIds = inputIds
    self.attentionMask = attentionMask
  }
}

public final class QwenTokenizer {
  private let encodeFunction: @Sendable (String) -> [Int]
  private let prefixTokens: [Int]
  private let suffixTokens: [Int]
  private let tokenizer: Tokenizer

  public let padTokenId: Int
  public let maxLength: Int
  public let imageTokenId: Int?
  public let visionStartTokenId: Int?
  public let visionEndTokenId: Int?

  public var templateTokenCount: Int {
    prefixTokens.count
  }

  public init(
    padTokenId: Int,
    maxLength: Int,
    prefixTokens: [Int],
    suffixTokens: [Int],
    tokenizer: Tokenizer,
    imageTokenId: Int? = nil,
    visionStartTokenId: Int? = nil,
    visionEndTokenId: Int? = nil,
    encode: @escaping @Sendable (String) -> [Int]
  ) {
    self.padTokenId = padTokenId
    self.maxLength = maxLength
    self.prefixTokens = prefixTokens
    self.suffixTokens = suffixTokens
    self.tokenizer = tokenizer
    self.imageTokenId = imageTokenId
    self.visionStartTokenId = visionStartTokenId
    self.visionEndTokenId = visionEndTokenId
    self.encodeFunction = encode
  }

  public static func load(
    from directory: URL,
    maxLengthOverride: Int? = nil,
    hubApi: HubApi = .shared
  ) throws -> QwenTokenizer {
    let tokenizerDirectory = resolveTokenizerDirectory(directory)
    let tokenizerConfigURL = tokenizerDirectory.appending(path: "tokenizer_config.json")
    let tokenizerDataURL = tokenizerDirectory.appending(path: "tokenizer.json")

    guard FileManager.default.fileExists(atPath: tokenizerDirectory.path) else {
      throw QwenTokenizerError.directoryNotFound(tokenizerDirectory)
    }
    guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
      throw QwenTokenizerError.fileNotFound(tokenizerConfigURL)
    }

    // Idempotently inline `chat_template.jinja` into `tokenizer_config.json`
    // when the HF snapshot uses the modern jinja-sidecar layout. This
    // replaces the manual /tmp/inject_chat_template.py operator workaround
    // that would otherwise have to be re-run any time the snapshot is
    // resynced. Operators can opt out with `ZIMAGE_NO_TOKENIZER_PATCH=1`.
    try? ensureInlineChatTemplate(in: tokenizerDirectory)

    let tokenizerConfig = try hubApi.configuration(fileURL: tokenizerConfigURL)
    let addedTokensURL = tokenizerDirectory.appending(path: "added_tokens.json")
    var addedTokens: [String: Int] = [:]
    if FileManager.default.fileExists(atPath: addedTokensURL.path) {
      if let addedData = try? Data(contentsOf: addedTokensURL),
         let addedObject = try? JSONSerialization.jsonObject(with: addedData, options: []) as? [String: Any] {
        for (token, value) in addedObject {
          if let index = value as? Int {
            addedTokens[token] = index
          }
        }
      }
    }

    let tokenizer: Tokenizer
    if FileManager.default.fileExists(atPath: tokenizerDataURL.path) {
      let tokenizerData = try hubApi.configuration(fileURL: tokenizerDataURL)
      tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    } else {

      let vocabURL = tokenizerDirectory.appending(path: "vocab.json")
      let mergesURL = tokenizerDirectory.appending(path: "merges.txt")
      guard FileManager.default.fileExists(atPath: vocabURL.path),
            FileManager.default.fileExists(atPath: mergesURL.path) else {
        throw QwenTokenizerError.fileNotFound(tokenizerDataURL)
      }
      let tokenizerData = try makeBPETokenizerData(vocabURL: vocabURL, mergesURL: mergesURL)
      tokenizer = try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    let padTokenNode = tokenizerConfig["pad_token"]
    let padTokenString = padTokenNode.string() ?? padTokenNode["content"].string()
    guard let padToken = padTokenString else {
      throw QwenTokenizerError.padTokenMissing
    }

    guard let padId = tokenizer.convertTokenToId(padToken) ??
      tokenizer.eosTokenId ??
      tokenizer.bosTokenId
    else {
      throw QwenTokenizerError.padTokenNotInVocabulary(padToken)
    }

    let resolvedMaxLength = maxLengthOverride ?? tokenizerConfig["model_max_length"].integer(or: 131_072)

    let prefixTokens = tokenizer.encode(text: promptPrefix)
    let suffixTokens = tokenizer.encode(text: promptSuffix)
    return QwenTokenizer(
      padTokenId: padId,
      maxLength: resolvedMaxLength,
      prefixTokens: prefixTokens,
      suffixTokens: suffixTokens,
      tokenizer: tokenizer,
      imageTokenId: addedTokens["<|image_pad|>"],
      visionStartTokenId: addedTokens["<|vision_start|>"],
      visionEndTokenId: addedTokens["<|vision_end|>"]
    ) { text in
      tokenizer.encode(text: text)
    }
  }

  private static func makeBPETokenizerData(vocabURL: URL, mergesURL: URL) throws -> Config {
    let vocabData = try Data(contentsOf: vocabURL)
    guard let vocabObject = try JSONSerialization.jsonObject(with: vocabData, options: []) as? [String: Any] else {
      throw QwenTokenizerError.fileNotFound(vocabURL)
    }
    var vocab: [String: Int] = [:]
    vocab.reserveCapacity(vocabObject.count)
    for (k, v) in vocabObject {
      if let i = v as? Int { vocab[k] = i }
    }

    let tokenizerDir = vocabURL.deletingLastPathComponent()
    let addedTokensURL = tokenizerDir.appending(path: "added_tokens.json")
    var addedTokensMap: [String: Int] = [:]
    if FileManager.default.fileExists(atPath: addedTokensURL.path) {
      if let addedData = try? Data(contentsOf: addedTokensURL),
         let added = try? JSONSerialization.jsonObject(with: addedData, options: []) as? [String: Any] {
        for (k, v) in added {
          if let i = v as? Int {
            vocab[k] = i
            addedTokensMap[k] = i
          }
        }
      }
    }

    let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
    let merges: [String] = mergesText
      .components(separatedBy: .newlines)
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty && !$0.hasPrefix("#") }

    var tokenizerDict: [String: Any] = [
      "model": [
        "vocab": vocab,
        "merges": merges
      ],
      "preTokenizer": [
        "type": "ByteLevel",
        "addPrefixSpace": false,
        "trimOffsets": true,
        "useRegex": true
      ],
      "decoder": [
        "type": "ByteLevel"
      ]
    ]

    if !addedTokensMap.isEmpty {
      var addedList: [[String: Any]] = []
      addedList.reserveCapacity(addedTokensMap.count)
      for (tok, id) in addedTokensMap {
        addedList.append([
          "id": id,
          "content": tok,
          "lstrip": false,
          "rstrip": false,
          "special": true
        ])
      }
      tokenizerDict["addedTokens"] = addedList
    }
    let data = try JSONSerialization.data(withJSONObject: tokenizerDict, options: [])
    let tokenizerData = try JSONDecoder().decode(Config.self, from: data)
    return tokenizerData
  }

  public func encode(
    prompts: [String],
    maxLength: Int? = nil
  ) -> QwenTokenBatch {
    precondition(!prompts.isEmpty, "At least one prompt must be provided.")

    let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
    precondition(targetLength > 0, "Maximum sequence length must be positive.")

    var inputSequences: [[Int]] = []
    var attentionSequences: [[Int]] = []
    inputSequences.reserveCapacity(prompts.count)
    attentionSequences.reserveCapacity(prompts.count)

    for prompt in prompts {
      var tokens = assembleTokens(for: prompt)
      tokens = Self.trim(tokens, maxLength: targetLength, prefixCount: prefixTokens.count, suffixCount: suffixTokens.count)
      let (ids, mask) = Self.prepareSequence(
        tokens: tokens,
        padTokenId: padTokenId,
        maxLength: targetLength
      )
      inputSequences.append(ids)
      attentionSequences.append(mask)
    }

    let flatIds = inputSequences.flatMap { $0 }
    let flatMask = attentionSequences.flatMap { $0 }
    let shape = [prompts.count, targetLength]

    let inputIds = MLXArray(flatIds.map { Float32($0) }, shape).asType(.int32)
    let attentionMask = MLXArray(flatMask.map { Float32($0) }, shape).asType(.int32)
    return QwenTokenBatch(inputIds: inputIds, attentionMask: attentionMask)
  }

  public func encode(
    prompt: String,
    negativePrompt: String?,
    maxLength: Int? = nil
  ) -> QwenTokenBatch {
    if let negativePrompt {
      return encode(prompts: [negativePrompt, prompt], maxLength: maxLength)
    } else {
      return encode(prompts: [prompt], maxLength: maxLength)
    }
  }

  public func encodeChat(
    prompts: [String],
    maxLength: Int? = nil
  ) throws -> QwenTokenBatch {
    let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
    var inputSequences: [[Int]] = []
    var attentionSequences: [[Int]] = []
    inputSequences.reserveCapacity(prompts.count)
    attentionSequences.reserveCapacity(prompts.count)

    for prompt in prompts {
      let messages: [[String: Any]] = [
        ["role": "user", "content": prompt]
      ]
      let tokens = try tokenizer.applyChatTemplate(messages: messages)
      let trimmed = Self.trim(tokens, maxLength: targetLength, prefixCount: 0, suffixCount: 0)
      let (ids, mask) = Self.prepareSequence(
        tokens: trimmed,
        padTokenId: padTokenId,
        maxLength: targetLength
      )
      inputSequences.append(ids)
      attentionSequences.append(mask)
    }

    let flatIds = inputSequences.flatMap { $0 }
    let flatMask = attentionSequences.flatMap { $0 }
    let shape = [prompts.count, targetLength]

    let inputIds = MLXArray(flatIds.map { Float32($0) }, shape).asType(.int32)
    let attentionMask = MLXArray(flatMask.map { Float32($0) }, shape).asType(.int32)
    return QwenTokenBatch(inputIds: inputIds, attentionMask: attentionMask)
  }

  /// Direct tokenization without chat template - used by Z-Image pipeline
  public func encodePlain(
    prompts: [String],
    maxLength: Int? = nil
  ) -> QwenTokenBatch {
    let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
    var inputSequences: [[Int]] = []
    var attentionSequences: [[Int]] = []
    inputSequences.reserveCapacity(prompts.count)
    attentionSequences.reserveCapacity(prompts.count)

    for prompt in prompts {
      // Direct tokenization without any chat template or prefix/suffix
      let tokens = encodeFunction(prompt)
      let trimmed = Self.trim(tokens, maxLength: targetLength, prefixCount: 0, suffixCount: 0)
      let (ids, mask) = Self.prepareSequence(
        tokens: trimmed,
        padTokenId: padTokenId,
        maxLength: targetLength
      )
      inputSequences.append(ids)
      attentionSequences.append(mask)
    }

    let flatIds = inputSequences.flatMap { $0 }
    let flatMask = attentionSequences.flatMap { $0 }
    let shape = [prompts.count, targetLength]

    let inputIds = MLXArray(flatIds.map { Float32($0) }, shape).asType(.int32)
    let attentionMask = MLXArray(flatMask.map { Float32($0) }, shape).asType(.int32)
    return QwenTokenBatch(inputIds: inputIds, attentionMask: attentionMask)
  }


  /// Dynamic-length tokenization matching Python's `padding="longest"`.
  ///
  /// Unlike `encodePlain` (which pads to `maxLength`), this method pads
  /// only to the longest sequence in the batch. For a single prompt this
  /// means NO padding — the output length equals the actual token count.
  ///
  /// This is critical for FIBO: its transformer creates an all-ones
  /// attention mask for the text portion, so any padding tokens are treated
  /// as real content and corrupt the denoising via DimFusion.
  public func encodeDynamic(
    prompts: [String],
    maxLength: Int? = nil
  ) -> QwenTokenBatch {
    precondition(!prompts.isEmpty, "At least one prompt must be provided.")

    let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
    precondition(targetLength > 0, "Maximum sequence length must be positive.")

    // Tokenize all prompts to get natural lengths
    var tokenizedPrompts: [[Int]] = []
    tokenizedPrompts.reserveCapacity(prompts.count)

    for prompt in prompts {
      let tokens = encodeFunction(prompt)
      let trimmed = Self.trim(tokens, maxLength: targetLength, prefixCount: 0, suffixCount: 0)
      tokenizedPrompts.append(trimmed)
    }

    // Pad to longest in batch (not to maxLength)
    let longestLength = tokenizedPrompts.map(\.count).max() ?? 0

    var inputSequences: [[Int]] = []
    var attentionSequences: [[Int]] = []
    inputSequences.reserveCapacity(prompts.count)
    attentionSequences.reserveCapacity(prompts.count)

    for tokens in tokenizedPrompts {
      let (ids, mask) = Self.prepareSequence(
        tokens: tokens,
        padTokenId: padTokenId,
        maxLength: longestLength
      )
      inputSequences.append(ids)
      attentionSequences.append(mask)
    }

    let flatIds = inputSequences.flatMap { $0 }
    let flatMask = attentionSequences.flatMap { $0 }
    let shape = [prompts.count, longestLength]

    let inputIds = MLXArray(flatIds.map { Float32($0) }, shape).asType(.int32)
    let attentionMask = MLXArray(flatMask.map { Float32($0) }, shape).asType(.int32)
    return QwenTokenBatch(inputIds: inputIds, attentionMask: attentionMask)
  }

  private func assembleTokens(for prompt: String) -> [Int] {
    let contentTokens = encodeFunction(prompt)
    return prefixTokens + contentTokens + suffixTokens
  }

  public func decode(tokens: [Int]) -> String {
    tokenizer.decode(tokens: tokens)
  }

  public var eosTokenId: Int? {
    tokenizer.eosTokenId
  }

  public func encodeChatForGeneration(
    messages: [[String: Any]],
    maxLength: Int? = nil
  ) throws -> [Int] {
    var text = ""
    for message in messages {
      guard let role = message["role"] as? String,
            let content = message["content"] as? String else {
        continue
      }
      text += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
    }
    text += "<|im_start|>assistant\n"

    let tokens = encodeFunction(text)
    let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
    return Self.trim(tokens, maxLength: targetLength, prefixCount: 0, suffixCount: 0)
  }

  private static func trim(
    _ tokens: [Int],
    maxLength: Int,
    prefixCount: Int,
    suffixCount: Int
  ) -> [Int] {
    guard tokens.count > maxLength else { return tokens }
    if prefixCount + suffixCount >= maxLength {
      return Array(tokens.prefix(maxLength))
    }

    let prefix = Array(tokens.prefix(min(prefixCount, tokens.count)))
    let suffix = suffixCount > 0 ? Array(tokens.suffix(min(suffixCount, maxLength - prefix.count))) : []

    let contentStart = min(prefixCount, tokens.count)
    let contentEnd = max(contentStart, tokens.count - suffixCount)
    let content: [Int]
    if contentEnd > contentStart {
      content = Array(tokens[contentStart..<contentEnd])
    } else {
      content = []
    }

    let availableForContent = max(0, maxLength - prefix.count - suffix.count)
    let trimmedContent = Array(content.prefix(availableForContent))
    return prefix + trimmedContent + suffix
  }

  private static func prepareSequence(
    tokens: [Int],
    padTokenId: Int,
    maxLength: Int
  ) -> ([Int], [Int]) {
    let truncated = Array(tokens.prefix(maxLength))
    let paddingCount = max(0, maxLength - truncated.count)

    var padded = truncated
    if paddingCount > 0 {
      padded.append(contentsOf: Array(repeating: padTokenId, count: paddingCount))
    }

    var attention = Array(repeating: 1, count: truncated.count)
    if paddingCount > 0 {
      attention.append(contentsOf: Array(repeating: 0, count: paddingCount))
    }

    return (padded, attention)
  }

  private static func resolveTokenizerDirectory(_ directory: URL) -> URL {
    let tokenizerPath = directory.appending(path: "tokenizer", directoryHint: .isDirectory)
    if FileManager.default.fileExists(atPath: tokenizerPath.path) {
      return tokenizerPath
    }
    return directory
  }

  /// When a HuggingFace tokenizer ships its chat template as a sidecar
  /// `chat_template.jinja` instead of inlining it in `tokenizer_config.json`,
  /// swift-transformers' `applyChatTemplate` trips over `missingChatTemplate`
  /// because it only reads the inline field. Upstream snapshots of
  /// `Tongyi-MAI/Z-Image-Turbo` (and many modern HF model cards) ship the
  /// jinja-sidecar layout.
  ///
  /// This helper merges the sidecar into the inline field at load time,
  /// atomically and idempotently, with a `.bak` of the original config on
  /// first patch. Subsequent loads see the inline field and no-op.
  ///
  /// Opt out via `ZIMAGE_NO_TOKENIZER_PATCH=1` (e.g. when running against a
  /// read-only snapshot mount or a model whose chat_template must not be
  /// augmented).
  ///
  /// Handles HuggingFace cache symlinks: snapshot files are typically symlinks
  /// to content-addressed blobs. Writing through a symlink mutates the shared
  /// blob and corrupts the backup. This function resolves symlinks first,
  /// removes the symlink, and writes a real file in its place.
  ///
  /// Errors are swallowed by the caller — a failure here degrades to the
  /// pre-existing `missingChatTemplate` error at `applyChatTemplate` time,
  /// which is strictly no worse than the status quo.
  static func ensureInlineChatTemplate(
    in tokenizerDirectory: URL,
    environment: [String: String] = ProcessInfo.processInfo.environment
  ) throws {
    if environment["ZIMAGE_NO_TOKENIZER_PATCH"] == "1" {
      return
    }

    let configURL = tokenizerDirectory.appending(path: "tokenizer_config.json")
    let jinjaURL = tokenizerDirectory.appending(path: "chat_template.jinja")
    let fm = FileManager.default

    // Clean up orphaned .tmp files from previous failed patch attempts.
    if let contents = try? fm.contentsOfDirectory(
      at: tokenizerDirectory,
      includingPropertiesForKeys: nil
    ) {
      for url in contents where url.lastPathComponent.contains(".patch.") && url.pathExtension == "tmp" {
        try? fm.removeItem(at: url)
      }
    }

    guard fm.fileExists(atPath: configURL.path) else {
      // Config missing — caller will surface a clearer error downstream.
      return
    }

    // Read the config, resolving through symlinks so we get actual content.
    let resolvedConfigURL = configURL.resolvingSymlinksInPath()
    let configData = try Data(contentsOf: resolvedConfigURL)
    guard var configJSON = try JSONSerialization.jsonObject(
      with: configData, options: []
    ) as? [String: Any] else {
      return
    }

    // Already inlined — no-op.  Any non-empty string counts as "present",
    // matching what swift-transformers will accept downstream.
    if let existing = configJSON["chat_template"] as? String, !existing.isEmpty {
      return
    }

    // No inline chat_template.  Try the jinja sidecar.
    guard fm.fileExists(atPath: jinjaURL.path) else {
      FileHandle.standardError.write(Data(
        "[zimage] warning: tokenizer_config.json has no chat_template and no chat_template.jinja sidecar found in \(tokenizerDirectory.path) — applyChatTemplate will fail\n".utf8
      ))
      return
    }

    let resolvedJinjaURL = jinjaURL.resolvingSymlinksInPath()
    let jinjaTemplate = try String(contentsOf: resolvedJinjaURL, encoding: .utf8)
    guard !jinjaTemplate.isEmpty else {
      FileHandle.standardError.write(Data(
        "[zimage] warning: chat_template.jinja is empty in \(tokenizerDirectory.path)\n".utf8
      ))
      return
    }

    configJSON["chat_template"] = jinjaTemplate

    // Back up the original config (actual content, not the symlink).
    // Never overwrite an existing backup that contains real content.
    let backupURL = configURL.appendingPathExtension("bak")
    let backupExists = fm.fileExists(atPath: backupURL.path)
    if backupExists {
      // If the backup is a symlink (from a previous broken run that copied
      // the symlink instead of the content), replace it with real content.
      let backupAttrs = try? fm.attributesOfItem(atPath: backupURL.path)
      if backupAttrs?[.type] as? FileAttributeType == .typeSymbolicLink {
        try? fm.removeItem(at: backupURL)
        try? configData.write(to: backupURL, options: .atomic)
      }
    } else {
      try? configData.write(to: backupURL, options: .atomic)
    }

    // Serialize the patched config.
    let patchedData = try JSONSerialization.data(
      withJSONObject: configJSON,
      options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    )

    // If tokenizer_config.json is a symlink, remove it first.  Writing
    // through a symlink would mutate the shared content-addressed blob in
    // the HuggingFace cache, which is both destructive and makes the .bak
    // backup point at the already-modified blob (useless).
    let configAttrs = try fm.attributesOfItem(atPath: configURL.path)
    if configAttrs[.type] as? FileAttributeType == .typeSymbolicLink {
      try fm.removeItem(at: configURL)
    }

    // Atomic write via temp sibling + rename, so readers never see a
    // torn file.  .sortedKeys for stable diffs on subsequent patches.
    let tmpURL = configURL.appendingPathExtension("patch.\(UUID().uuidString).tmp")
    do {
      try patchedData.write(to: tmpURL, options: .atomic)
      if fm.fileExists(atPath: configURL.path) {
        _ = try fm.replaceItemAt(configURL, withItemAt: tmpURL)
      } else {
        try fm.moveItem(at: tmpURL, to: configURL)
      }
    } catch {
      // Clean up temp file on failure.
      try? fm.removeItem(at: tmpURL)
      throw error
    }

    FileHandle.standardError.write(Data(
      "[zimage] inlined chat_template.jinja (\(jinjaTemplate.count) chars) into \(configURL.path); backup at \(backupURL.lastPathComponent)\n".utf8
    ))
  }

  private static let promptPrefix: String = """
<|im_start|>system
Describe the key features of the input image (color, shape, size, texture, objects, background),
then explain how the user's text instruction should alter or modify the image.
Generate a new image that meets the user's requirements while maintaining consistency with the
original input where appropriate.<|im_end|>
<|im_start|>user
"""

  private static let promptSuffix: String = """
<|im_end|>
<|im_start|>assistant
"""
}
