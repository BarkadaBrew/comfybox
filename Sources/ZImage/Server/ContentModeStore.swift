// ContentModeStore.swift — Content-mode (NSFW scaling) registry for ComfyBox.
//
// Straight port of the Coffee Shop image service `src/content-mode.ts`. Three modes:
//   - neutral (default)  — no scaling; auto-detects sensual/explicit intent from the prompt.
//   - banana  (sensual)  — sensual / suggestive guidance boost + prompt hint.
//   - avocado (explicit) — explicit / uncensored guidance boost + prompt hint.
//
// The store exposes the built-in modes, resolves a `ContentModeEffect` (guidance boost,
// style variant, prompt hint) for a request, and `apply()`s a mode to a generation request —
// returning the adjusted guidance plus prompt/negative additions the caller folds in.
//
// Per-mode config is persisted to `~/.comfybox/content-modes.json`; the built-in defaults
// ship in-code so the file is optional and older/partial files load tolerantly.

import Foundation

/// The three content modes ("fruit modes"). Raw values are the wire/JSON tokens.
public enum ContentMode: String, Codable, CaseIterable, Sendable {
  case neutral
  case banana
  case avocado
}

/// Style intent the renderer uses to pick preset/LoRA variants.
public enum ContentStyleVariant: String, Codable, Sendable {
  case neutral
  case sensual
  case nsfw
}

/// Engine / model descriptor used to decide whether a guidance boost should apply.
///
/// CFG-distilled models (z-image-turbo, schnell, lightning) expect guidance ≈ 1.0 — pushing
/// them above ~1.5 in bf16 produces NaN latents and a black PNG. For those we suppress the
/// boost entirely; the explicit-content signal then comes from prompt hints + LoRAs, not CFG.
public struct GuidanceContext: Equatable, Sendable {
  public var engine: String?
  public var mode: String?
  public var model: String?
  public var baseModel: String?

  public init(engine: String? = nil, mode: String? = nil, model: String? = nil, baseModel: String? = nil) {
    self.engine = engine
    self.mode = mode
    self.model = model
    self.baseModel = baseModel
  }
}

/// The resolved effect of a content mode on a single generation.
public struct ContentModeEffect: Codable, Equatable, Sendable {
  public var guidanceBoost: Double
  public var styleVariant: ContentStyleVariant
  public var promptHint: String?

  public init(guidanceBoost: Double, styleVariant: ContentStyleVariant, promptHint: String? = nil) {
    self.guidanceBoost = guidanceBoost
    self.styleVariant = styleVariant
    self.promptHint = promptHint
  }
}

/// Persisted, editable definition of one content mode. Ships with built-in defaults; a
/// `~/.comfybox/content-modes.json` file may override the boost/hint/negatives per mode.
public struct ContentModeDefinition: Codable, Equatable, Sendable {
  /// Which mode this defines.
  public var mode: ContentMode
  /// Human-facing name for UI listing.
  public var label: String
  /// Short description for UI listing.
  public var summary: String
  /// Guidance boost added to the base/preset guidance (suppressed for CFG-distilled models).
  public var guidanceBoost: Double
  /// Style intent handed to the renderer.
  public var styleVariant: ContentStyleVariant
  /// Prompt hint appended to the injected keywords (nil for neutral).
  public var promptHint: String?
  /// Extra negative-prompt terms folded in when this mode is active.
  public var negativePromptAdditions: [String]

  public init(
    mode: ContentMode,
    label: String,
    summary: String,
    guidanceBoost: Double,
    styleVariant: ContentStyleVariant,
    promptHint: String? = nil,
    negativePromptAdditions: [String] = []
  ) {
    self.mode = mode
    self.label = label
    self.summary = summary
    self.guidanceBoost = guidanceBoost
    self.styleVariant = styleVariant
    self.promptHint = promptHint
    self.negativePromptAdditions = negativePromptAdditions
  }

  private enum CodingKeys: String, CodingKey {
    case mode, label, summary, guidanceBoost, styleVariant, promptHint, negativePromptAdditions
  }

  /// Tolerant decode: a partial entry falls back to the matching built-in for any missing field.
  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    let mode = try c.decode(ContentMode.self, forKey: .mode)
    let base = ContentModeDefinition.builtin(mode)
    self.mode = mode
    self.label = try c.decodeIfPresent(String.self, forKey: .label) ?? base.label
    self.summary = try c.decodeIfPresent(String.self, forKey: .summary) ?? base.summary
    self.guidanceBoost = try c.decodeIfPresent(Double.self, forKey: .guidanceBoost) ?? base.guidanceBoost
    self.styleVariant = try c.decodeIfPresent(ContentStyleVariant.self, forKey: .styleVariant) ?? base.styleVariant
    self.promptHint = try c.decodeIfPresent(String.self, forKey: .promptHint) ?? base.promptHint
    self.negativePromptAdditions =
      try c.decodeIfPresent([String].self, forKey: .negativePromptAdditions) ?? base.negativePromptAdditions
  }

  /// The shipped default definition for a mode.
  public static func builtin(_ mode: ContentMode) -> ContentModeDefinition {
    switch mode {
    case .neutral:
      return ContentModeDefinition(
        mode: .neutral,
        label: "Neutral",
        summary: "Default. No NSFW scaling; sensual/explicit intent is auto-detected from the prompt.",
        guidanceBoost: 0,
        styleVariant: .neutral,
        promptHint: nil,
        negativePromptAdditions: []
      )
    case .banana:
      return ContentModeDefinition(
        mode: .banana,
        label: "Banana",
        summary: "Sensual / suggestive. Adds a mild guidance boost and intimate prompt hint.",
        guidanceBoost: 1.5,
        styleVariant: .sensual,
        promptHint: "sensual, intimate, suggestive",
        negativePromptAdditions: []
      )
    case .avocado:
      return ContentModeDefinition(
        mode: .avocado,
        label: "Avocado",
        summary: "Explicit / uncensored. Adds a stronger guidance boost and explicit prompt hint.",
        guidanceBoost: 2.5,
        styleVariant: .nsfw,
        promptHint: "explicit, uncensored, anatomically detailed",
        negativePromptAdditions: []
      )
    }
  }
}

/// Input to ``ContentModeStore/apply(_:)``: the request fields the content mode touches.
public struct ContentModeRequest: Equatable, Sendable {
  /// Explicit mode; when nil/neutral, the prompt is auto-scanned.
  public var mode: ContentMode?
  /// The user prompt (used for auto-detection in neutral mode).
  public var prompt: String?
  /// The base/preset guidance, if the request or preset supplied one.
  public var guidance: Double?
  /// Engine/model descriptor for CFG-distillation suppression.
  public var context: GuidanceContext

  public init(
    mode: ContentMode? = nil,
    prompt: String? = nil,
    guidance: Double? = nil,
    context: GuidanceContext = GuidanceContext()
  ) {
    self.mode = mode
    self.prompt = prompt
    self.guidance = guidance
    self.context = context
  }
}

/// Result of applying a content mode: the resolved effect plus the concrete adjustments
/// the caller folds into the outgoing generation request.
public struct ContentModeResult: Equatable, Sendable {
  /// The resolved effect (boost, style variant, hint).
  public var effect: ContentModeEffect
  /// The final guidance value, or nil when neither a base guidance nor a boost applies.
  ///
  /// Mirrors the image service: `base != nil ? base + boost : (boost > 0 ? 3.5 + boost : nil)`.
  public var guidance: Double?
  /// Prompt fragments to append to the injected keywords (the mode's prompt hint, if any).
  public var promptAdditions: [String]
  /// Extra negative-prompt terms to fold in for the active mode.
  public var negativePromptAdditions: [String]

  public init(
    effect: ContentModeEffect,
    guidance: Double?,
    promptAdditions: [String],
    negativePromptAdditions: [String]
  ) {
    self.effect = effect
    self.guidance = guidance
    self.promptAdditions = promptAdditions
    self.negativePromptAdditions = negativePromptAdditions
  }
}

/// Registry of content modes: lists them, resolves effects, and applies a mode to a request.
///
/// Value type mirroring ``ComfyBoxServerConfig``'s persistence style: Codable, tolerant decode,
/// `~/.comfybox` home, atomic writes. Built-in defaults ship in-code so the JSON file is optional.
public struct ContentModeStore: Codable, Equatable, Sendable {
  /// All mode definitions, in canonical (neutral, banana, avocado) order.
  public var modes: [ContentModeDefinition]

  public init(modes: [ContentModeDefinition] = ContentModeStore.builtinModes()) {
    self.modes = ContentModeStore.normalize(modes)
  }

  private enum CodingKeys: String, CodingKey { case modes }

  /// Tolerant decode: a missing/partial `modes` array is backfilled with built-ins.
  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    let decoded = try c.decodeIfPresent([ContentModeDefinition].self, forKey: .modes) ?? []
    self.modes = ContentModeStore.normalize(decoded)
  }

  public func encode(to encoder: Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(modes, forKey: .modes)
  }

  // MARK: - Built-ins & normalization

  /// The three shipped mode definitions, in canonical order.
  public static func builtinModes() -> [ContentModeDefinition] {
    ContentMode.allCases.map { ContentModeDefinition.builtin($0) }
  }

  /// Ensure exactly one entry per mode, in canonical order; missing modes are added from
  /// built-ins and duplicates collapse to the first seen (so a partial file still loads).
  static func normalize(_ input: [ContentModeDefinition]) -> [ContentModeDefinition] {
    var seen: [ContentMode: ContentModeDefinition] = [:]
    for def in input where seen[def.mode] == nil {
      seen[def.mode] = def
    }
    return ContentMode.allCases.map { seen[$0] ?? ContentModeDefinition.builtin($0) }
  }

  // MARK: - Listing & lookup

  /// The available modes for UI listing (canonical order).
  public func listModes() -> [ContentModeDefinition] { modes }

  /// The definition for a specific mode (always present after normalization).
  public func definition(for mode: ContentMode) -> ContentModeDefinition {
    modes.first { $0.mode == mode } ?? ContentModeDefinition.builtin(mode)
  }

  // MARK: - CFG-distillation detection

  /// True for any CFG-distilled model (turbo / schnell / lightning family). Those models
  /// encode unconditional output during training and run at guidance ≈ 1.0 — adding a
  /// content-mode boost yields NaN latents and a black PNG. Ported 1:1 from `isCfgDistilledModel`.
  public func isCfgDistilledModel(_ ctx: GuidanceContext) -> Bool {
    let tokens = [ctx.mode, ctx.model, ctx.baseModel]
      .compactMap { $0 }
      .filter { !$0.isEmpty }
      .map { $0.lowercased() }
    if tokens.contains(where: { $0.contains("turbo") || $0.contains("schnell") || $0.contains("lightning") }) {
      return true
    }
    // z-image-turbo ships as engine=zimage + mode=z-image-turbo; guard on the engine alone
    // so a missing mode still trips the check.
    if ctx.engine == "zimage" && (ctx.mode == nil || ctx.mode!.lowercased().contains("turbo")) {
      return true
    }
    return false
  }

  // MARK: - Effect resolution

  /// Resolve the content-mode effect for a request. Ported 1:1 from `resolveContentMode`:
  /// explicit banana/avocado use the mode definition (with boost suppressed for CFG-distilled
  /// models); neutral/nil auto-detects sensual/explicit intent from the prompt.
  public func resolveEffect(
    mode: ContentMode?,
    prompt: String? = nil,
    context: GuidanceContext = GuidanceContext()
  ) -> ContentModeEffect {
    let suppressBoost = isCfgDistilledModel(context)

    switch mode {
    case .avocado:
      let def = definition(for: .avocado)
      return ContentModeEffect(
        guidanceBoost: suppressBoost ? 0 : def.guidanceBoost,
        styleVariant: def.styleVariant,
        promptHint: def.promptHint
      )
    case .banana:
      let def = definition(for: .banana)
      return ContentModeEffect(
        guidanceBoost: suppressBoost ? 0 : def.guidanceBoost,
        styleVariant: def.styleVariant,
        promptHint: def.promptHint
      )
    case .neutral, .none:
      // Auto-detect from the prompt when the mode is neutral/undefined.
      if let prompt, Self.nsfwPattern.matches(prompt) {
        return ContentModeEffect(guidanceBoost: 0, styleVariant: .nsfw)
      }
      if let prompt, Self.sensualPattern.matches(prompt) {
        return ContentModeEffect(guidanceBoost: 0, styleVariant: .sensual)
      }
      return ContentModeEffect(guidanceBoost: 0, styleVariant: .neutral)
    }
  }

  // MARK: - Apply

  /// Apply a content mode to a generation request, returning the adjusted guidance and the
  /// prompt/negative additions to fold in. Guidance math mirrors the image service exactly.
  public func apply(_ request: ContentModeRequest) -> ContentModeResult {
    let effect = resolveEffect(mode: request.mode, prompt: request.prompt, context: request.context)

    // base != nil ? base + boost : (boost > 0 ? 3.5 + boost : nil)
    let guidance: Double?
    if let base = request.guidance {
      guidance = base + effect.guidanceBoost
    } else if effect.guidanceBoost > 0 {
      guidance = 3.5 + effect.guidanceBoost
    } else {
      guidance = nil
    }

    let promptAdditions = effect.promptHint.map { [$0] } ?? []

    // Negative additions come from the resolved mode's definition (explicit modes). Neutral
    // auto-detection carries no configured negatives.
    let resolvedMode = request.mode ?? .neutral
    let negativeAdditions: [String] =
      (resolvedMode == .banana || resolvedMode == .avocado)
      ? definition(for: resolvedMode).negativePromptAdditions
      : []

    return ContentModeResult(
      effect: effect,
      guidance: guidance,
      promptAdditions: promptAdditions,
      negativePromptAdditions: negativeAdditions
    )
  }

  // MARK: - Auto-detect patterns

  /// Explicit-intent detector. Ported from `NSFW_PROMPT_PATTERN`.
  static let nsfwPattern = WordPattern(
    #"\b(nsfw|explicit|erotic|sex(?:ual)?|fetish|masturbat|orgasm|cum|penis|vagina|nude|nudity|naked)\b"#
  )
  /// Sensual-intent detector. Ported from `SENSUAL_PROMPT_PATTERN`.
  static let sensualPattern = WordPattern(
    #"\b(sensual|boudoir|intimate|suggestive|lingerie|pinup|glamour|topless|bottomless|undress)\b"#
  )

  // MARK: - Paths & persistence

  /// `~/.comfybox/content-modes.json`.
  public static func defaultPath() -> URL {
    ComfyBoxServerConfig.homeDirectory().appendingPathComponent(".comfybox/content-modes.json")
  }

  /// Load `~/.comfybox/content-modes.json`. If absent or unreadable, ship built-in defaults,
  /// persist them, and return. Present-but-partial files load tolerantly (backfilled).
  @discardableResult
  public static func loadOrCreate(
    at path: URL = ContentModeStore.defaultPath(),
    fileManager: FileManager = .default
  ) -> ContentModeStore {
    if fileManager.fileExists(atPath: path.path),
       let data = try? Data(contentsOf: path),
       let store = try? JSONDecoder().decode(ContentModeStore.self, from: data) {
      return store
    }
    let store = ContentModeStore()
    try? store.save(to: path, fileManager: fileManager)
    return store
  }

  public func save(to path: URL = ContentModeStore.defaultPath(), fileManager: FileManager = .default) throws {
    let dir = path.deletingLastPathComponent()
    try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(self)
    try data.write(to: path, options: .atomic)
  }
}

/// A small case-insensitive, word-boundary regex wrapper (compiled once, reused).
///
/// `@unchecked Sendable`: the wrapped `NSRegularExpression` is immutable after construction
/// and documented thread-safe, so sharing the compiled static patterns across threads is safe.
struct WordPattern: @unchecked Sendable {
  private let regex: NSRegularExpression?

  init(_ pattern: String) {
    self.regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
  }

  /// True if the pattern matches anywhere in `text`.
  func matches(_ text: String) -> Bool {
    guard let regex else { return false }
    let range = NSRange(text.startIndex..<text.endIndex, in: text)
    return regex.firstMatch(in: text, options: [], range: range) != nil
  }
}
