// SessionState.swift — Per-chat session state tracking for Telegram bot.
//
// Phase 3: Tracks all render settings per chatId — aspect, cfg, seed,
// post-processing, upscale, polish, and last render context.
// Phase 4: Adds messageId→imagePath mapping, discuss mode state,
// and per-message render context for inline keyboards.
// Thread-safe via NSLock.

import Foundation

// MARK: - Render Context (for inline keyboard callbacks)

/// Stored context for a delivered image, enabling rerender/HQ/reply actions.
public struct RenderContext: Sendable {
  public let prompt: String
  public let imagePath: String
  public let seed: Int?
  public let character: String?
  public let contentMode: String
  public let enhanceEnabled: Bool

  public init(prompt: String, imagePath: String, seed: Int?, character: String?,
              contentMode: String, enhanceEnabled: Bool) {
    self.prompt = prompt
    self.imagePath = imagePath
    self.seed = seed
    self.character = character
    self.contentMode = contentMode
    self.enhanceEnabled = enhanceEnabled
  }
}

// MARK: - Discuss Mode History Entry

public struct DiscussEntry: Sendable {
  public let role: String   // "user" or "assistant"
  public let content: String
}

// MARK: - ChatState

/// All per-chat render settings and last-render context.
public struct ChatState: Sendable {
  // -- Toggles --
  public var enhanceEnabled: Bool = true
  public var upscaleEnabled: Bool = false
  public var polishEnabled: Bool = false
  public var autoVideoEnabled: Bool = false
  public var verboseEnabled: Bool = false

  // -- Render settings --
  public var aspectMode: String = "portrait"
  public var cfgOverride: Double? = nil
  public var seedLock: Int? = nil
  public var resolutionTarget: String? = nil  // "2k", "4k", nil

  // -- Post-processing --
  public var saturation: Double? = nil
  public var colorTemp: Int? = nil
  public var filmLook: String? = nil

  // -- Last render context (for re-render, vary, etc.) --
  public var lastPrompt: String? = nil
  public var lastImagePath: String? = nil
  public var lastSeed: Int? = nil

  // -- Phase 4: Per-message render context (messageId → RenderContext) --
  // Maps bot-sent photo messageId to the render context that produced it.
  // Used by inline keyboard callbacks and reply-to-image.
  public var renderContexts: [Int: RenderContext] = [:]

  // -- Phase 4: Discuss mode --
  public var isInDiscussMode: Bool = false
  public var discussHistory: [DiscussEntry] = []
  public var discussCurrentPrompt: String? = nil

  public init() {}

  /// Convenience initializer with custom enhance default.
  public init(enhanceEnabled: Bool) {
    self.enhanceEnabled = enhanceEnabled
  }

  // MARK: - Render Context Management

  /// Store a render context for a delivered message.
  /// Keeps only the last 50 entries to avoid unbounded growth.
  public mutating func storeRenderContext(messageId: Int, context: RenderContext) {
    renderContexts[messageId] = context
    // Evict oldest if over 50
    if renderContexts.count > 50 {
      let sorted = renderContexts.keys.sorted()
      let toRemove = sorted.prefix(renderContexts.count - 50)
      for key in toRemove {
        renderContexts.removeValue(forKey: key)
      }
    }
  }

  /// Look up a render context by the bot-sent message ID.
  public func getRenderContext(messageId: Int) -> RenderContext? {
    return renderContexts[messageId]
  }

  // MARK: - Discuss Mode

  /// Enter discuss mode, clearing any previous history.
  public mutating func enterDiscussMode() {
    isInDiscussMode = true
    discussHistory = []
    discussCurrentPrompt = nil
  }

  /// Exit discuss mode.
  public mutating func exitDiscussMode() {
    isInDiscussMode = false
    discussHistory = []
    discussCurrentPrompt = nil
  }

  /// Add a message to the discuss history. Keeps last 20 entries.
  public mutating func addDiscussMessage(role: String, content: String) {
    discussHistory.append(DiscussEntry(role: role, content: content))
    if discussHistory.count > 20 {
      discussHistory = Array(discussHistory.suffix(20))
    }
  }

  // MARK: - Aspect Dimensions

  /// Resolve aspect mode to (width, height) dimensions.
  public func aspectDimensions() -> (width: Int, height: Int) {
    return ChatState.dimensionsForAspect(aspectMode)
  }

  /// Static lookup for aspect mode dimensions.
  public static func dimensionsForAspect(_ mode: String) -> (width: Int, height: Int) {
    switch mode.lowercased() {
    case "square":
      return (1024, 1024)
    case "portrait":
      return (768, 1024)
    case "landscape":
      return (1024, 768)
    case "wide":
      return (1024, 576)
    case "tall":
      return (576, 1024)
    default:
      return (768, 1024)  // default to portrait
    }
  }

  /// Reset all post-processing settings to defaults.
  public mutating func resetPostProcessing() {
    saturation = nil
    colorTemp = nil
    filmLook = nil
  }

  /// Whether any post-processing settings are active.
  public var hasPostProcessing: Bool {
    return saturation != nil || colorTemp != nil || filmLook != nil
  }

  /// Human-readable summary of active settings.
  public func settingsSummary() -> String {
    var parts: [String] = []
    parts.append("aspect: \(aspectMode)")
    if let cfg = cfgOverride { parts.append("cfg: \(cfg)") }
    if let seed = seedLock { parts.append("seed: \(seed)") }
    if upscaleEnabled { parts.append("upscale: on") }
    if polishEnabled { parts.append("polish: on") }
    if let target = resolutionTarget { parts.append("res: \(target)") }
    if let sat = saturation { parts.append("sat: \(String(format: "%.1f", sat))") }
    if let temp = colorTemp { parts.append("temp: \(temp)K") }
    if let look = filmLook { parts.append("film: \(look)") }
    return parts.joined(separator: ", ")
  }
}

// MARK: - SessionState Manager

/// Thread-safe per-chat session state storage.
public final class SessionState: @unchecked Sendable {
  private var states: [Int: ChatState] = [:]
  private let lock = NSLock()
  private let defaultEnhance: Bool

  /// Initialize with a default enhance setting (from optimizer config).
  public init(defaultEnhance: Bool = true) {
    self.defaultEnhance = defaultEnhance
  }

  /// Get the current state for a chat. Creates a default if none exists.
  public func getState(chatId: Int) -> ChatState {
    lock.lock()
    defer { lock.unlock() }
    if let state = states[chatId] {
      return state
    }
    let newState = ChatState(enhanceEnabled: defaultEnhance)
    states[chatId] = newState
    return newState
  }

  /// Update the state for a chat using a mutation closure.
  public func updateState(chatId: Int, update: (inout ChatState) -> Void) {
    lock.lock()
    defer { lock.unlock() }
    var state = states[chatId] ?? ChatState(enhanceEnabled: defaultEnhance)
    update(&state)
    states[chatId] = state
  }
}
