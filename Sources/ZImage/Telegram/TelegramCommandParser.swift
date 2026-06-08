// TelegramCommandParser.swift — Parse Telegram message text into bot commands.
//
// Phase 1: /help, /status, and bare text (render). All other commands defined
// in the enum but parsed as render with the raw text for forward compatibility.
// Phase 4: Discuss mode ship-cue detection, reply-to-image intent parsing.

import Foundation

// MARK: - Command Enum

public enum BotCommand: Sendable {
  // -- Content modes --
  case neutral
  case banana
  case avocado

  // -- Toggles --
  case enhance(on: Bool?)
  case upscale(on: Bool?)
  case polish(on: Bool?)
  case verbose(on: Bool?)
  case autoVideo(on: Bool?)
  case resolution(target: String?)

  // -- Settings --
  case aspect(mode: String?)
  case cfg(value: Double?)
  case seed(value: Int?)
  case saturation(value: Double?)
  case colorTemp(kelvin: Int?)
  case film(lookId: String?)

  // -- Generation --
  case render(prompt: String)
  case batch(count: Int, prompt: String)
  case vary(count: Int, prompt: String)
  case sequence(count: Int, story: String)
  case video(prompt: String)

  // -- Session --
  case chat
  case imagine(description: String)
  case endChat
  case shipCue
  case chatMessage(text: String)

  // -- Admin --
  case status
  case help
  case reset
  case look(id: String?)
  case queue(subcommand: String?)
}

// MARK: - Reply Intent

/// Describes what the user intends when replying to a bot-sent photo.
public enum ReplyIntent: Sendable {
  case rerender                         // "rerender", "again"
  case hq                               // "hq"
  case upscaleReply                     // "upscale"
  case video(motion: String?)           // "video" or "video <desc>"
  case newPrompt(text: String)          // any other text → img2img
}

// MARK: - Parsed Prompt

public struct ParsedPrompt: Sendable {
  public let prompt: String
  public let character: String?
  public let contentMode: String?
}

// MARK: - Parser

public enum TelegramCommandParser {
  // Known characters for detection
  private static let characterNames = ["kira", "bree", "todd"]

  // Ship-cue phrases
  private static let shipCues: Set<String> = [
    "go", "render", "render it", "ship", "ship it",
    "do it", "let's go", "send it", "fire it"
  ]

  /// Parse raw message text into a BotCommand.
  /// `inDiscussMode` controls whether bare text maps to .chatMessage or .render.
  public static func parse(_ text: String, inDiscussMode: Bool = false) -> BotCommand {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
      return .render(prompt: trimmed)
    }

    // Ship cue detection (discuss mode)
    if inDiscussMode && shipCues.contains(trimmed.lowercased()) {
      return .shipCue
    }

    // Command parsing (starts with /)
    if trimmed.hasPrefix("/") {
      let parts = trimmed.split(separator: " ", maxSplits: 1)
      let command = String(parts[0]).lowercased()
      let args = parts.count > 1 ? String(parts[1]) : nil

      switch command {
      // Phase 1 commands
      case "/help":
        return .help
      case "/status":
        return .status

      // Phase 2 commands (defined for forward compat)
      case "/neutral", "/apple":
        return .neutral
      case "/banana":
        return .banana
      case "/avocado":
        return .avocado
      case "/enhance":
        return .enhance(on: parseToggle(args))

      // Phase 3 commands
      case "/batch":
        if let args = args, let (count, prompt) = parseCountAndPrompt(args, defaultCount: 3) {
          return .batch(count: count, prompt: prompt)
        }
        return .help
      case "/vary":
        if let args = args, let (count, prompt) = parseCountAndPrompt(args, defaultCount: 3) {
          return .vary(count: count, prompt: prompt)
        }
        return .help
      case "/seq", "/sequence":
        if let args = args, let (count, prompt) = parseCountAndPrompt(args, defaultCount: 4) {
          return .sequence(count: count, story: prompt)
        }
        return .help
      case "/video":
        if let args = args { return .video(prompt: args) }
        return .help
      case "/upscale":
        return .upscale(on: parseToggle(args))
      case "/polish":
        return .polish(on: parseToggle(args))
      case "/verbose":
        return .verbose(on: parseToggle(args))
      case "/autovideo":
        return .autoVideo(on: parseToggle(args))
      case "/aspect":
        return .aspect(mode: args)
      case "/cfg":
        if let args = args, let val = Double(args) { return .cfg(value: val) }
        return .cfg(value: nil)
      case "/seed":
        if let args = args {
          if args.lowercased() == "random" { return .seed(value: nil) }
          if let val = Int(args) { return .seed(value: val) }
        }
        return .seed(value: nil)
      case "/saturation":
        if let args = args {
          if args.lowercased() == "off" { return .saturation(value: nil) }
          if let val = Double(args) { return .saturation(value: val) }
        }
        return .saturation(value: nil)
      case "/temp":
        if let args = args {
          if args.lowercased() == "off" { return .colorTemp(kelvin: nil) }
          if let val = Int(args) { return .colorTemp(kelvin: val) }
        }
        return .colorTemp(kelvin: nil)
      case "/film":
        return .film(lookId: args)
      case "/2k":
        return .resolution(target: parseToggle(args) == false ? nil : "2k")
      case "/4k":
        return .resolution(target: parseToggle(args) == false ? nil : "4k")
      case "/reset":
        return .reset
      case "/look":
        return .look(id: args)

      // Phase 4 commands
      case "/chat", "/discuss":
        return .chat
      case "/imagine":
        if let args = args { return .imagine(description: args) }
        return .help
      case "/end", "/exit":
        return .endChat
      case "/queue":
        return .queue(subcommand: args)

      default:
        // Unknown command — treat as render prompt
        return .render(prompt: trimmed)
      }
    }

    // Bare text — depends on mode
    if inDiscussMode {
      return .chatMessage(text: trimmed)
    }

    return .render(prompt: trimmed)
  }

  /// Determine the intent of a reply to a bot-sent photo.
  /// Returns nil if the text is empty or can't be classified.
  public static func parseReplyIntent(_ text: String) -> ReplyIntent? {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }

    let lowered = trimmed.lowercased()

    // Exact match keywords
    switch lowered {
    case "rerender", "again", "redo":
      return .rerender
    case "hq":
      return .hq
    case "upscale":
      return .upscaleReply
    case "video":
      return .video(motion: nil)
    default:
      break
    }

    // "video <motion description>"
    if lowered.hasPrefix("video ") {
      let motion = String(trimmed.dropFirst(6)).trimmingCharacters(in: .whitespaces)
      return .video(motion: motion.isEmpty ? nil : motion)
    }

    // Anything else is a new prompt for img2img
    return .newPrompt(text: trimmed)
  }

  /// Extract character name and inline mode overrides from a prompt string.
  public static func parsePrompt(_ text: String, defaultMode: String = "neutral") -> ParsedPrompt {
    var prompt = text
    var mode: String? = nil

    // Detect inline mode overrides: /neutral, /banana, /avocado in the prompt text
    let modePatterns: [(String, String)] = [
      ("/avocado", "avocado"),
      ("/banana", "banana"),
      ("/neutral", "neutral"),
      ("/apple", "neutral")
    ]
    for (pattern, modeName) in modePatterns {
      if let range = prompt.range(of: pattern, options: .caseInsensitive) {
        mode = modeName
        prompt.removeSubrange(range)
        prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        break
      }
    }

    // Detect character name
    let lowered = prompt.lowercased()
    var character: String? = nil
    for name in characterNames {
      if lowered.contains(name) {
        character = name.capitalized
        break
      }
    }

    return ParsedPrompt(
      prompt: prompt,
      character: character,
      contentMode: mode
    )
  }

  // MARK: - Private Helpers

  private static func parseToggle(_ args: String?) -> Bool? {
    guard let args = args?.lowercased().trimmingCharacters(in: .whitespaces) else { return nil }
    switch args {
    case "on", "true", "yes", "1": return true
    case "off", "false", "no", "0": return false
    default: return nil
    }
  }

  private static func parseCountAndPrompt(_ args: String, defaultCount: Int) -> (Int, String)? {
    let parts = args.split(separator: " ", maxSplits: 1)
    guard !parts.isEmpty else { return nil }

    if let count = Int(parts[0]), parts.count > 1 {
      let clampedCount = max(1, min(count, 8))
      return (clampedCount, String(parts[1]))
    }

    // No count prefix — use default
    return (defaultCount, args)
  }
}
