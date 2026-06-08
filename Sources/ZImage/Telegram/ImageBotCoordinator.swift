// ImageBotCoordinator.swift — Orchestrates Telegram bot: parse -> render -> send.
//
// Phase 1: Handles text prompts via WarmServer, /help, and /status.
// Phase 2: Content modes, character injection, prompt optimization.
// Connects to a running WarmServer instance via WarmServerClient.

import Foundation
import Logging

public final class ImageBotCoordinator {
  public struct Configuration: Sendable {
    public let telegram: TelegramBot.Configuration
    public let warmServerHost: String
    public let warmServerPort: UInt16
    public let outputDirectory: String
    public let galleryDirectory: String?
    public let optimizer: PromptOptimizer.Configuration
    public let characterConfigPath: String?
    public let contentModeConfigPath: String?

    public init(
      telegram: TelegramBot.Configuration,
      warmServerHost: String = "127.0.0.1",
      warmServerPort: UInt16 = 7862,
      outputDirectory: String = ("~/Pictures/ComfyBox/Telegram" as NSString).expandingTildeInPath,
      galleryDirectory: String? = nil,
      optimizer: PromptOptimizer.Configuration = PromptOptimizer.Configuration(),
      characterConfigPath: String? = nil,
      contentModeConfigPath: String? = nil
    ) {
      self.telegram = telegram
      self.warmServerHost = warmServerHost
      self.warmServerPort = warmServerPort
      self.outputDirectory = outputDirectory
      self.galleryDirectory = galleryDirectory
      self.optimizer = optimizer
      self.characterConfigPath = characterConfigPath
      self.contentModeConfigPath = contentModeConfigPath
    }
  }

  // MARK: - Session State

  private struct SessionState {
    var enhanceEnabled: Bool = true
    var lastPrompt: String? = nil
    var lastImagePath: String? = nil
  }

  private let config: Configuration
  private let bot: TelegramBot
  private let warmServer: WarmServerClient
  private let contentModeManager: ContentModeManager
  private let characterLoader: CharacterLoader
  private let promptOptimizer: PromptOptimizer
  private let logger: Logger
  private var isRunning = false
  private let startTime = Date()
  private var sessions: [Int: SessionState] = [:]  // chatId -> state

  public init(configuration: Configuration, logger: Logger = Logger(label: "comfybox.telegram.coordinator")) {
    self.config = configuration
    self.logger = logger
    self.bot = TelegramBot(configuration: configuration.telegram, logger: logger)
    self.warmServer = WarmServerClient(host: configuration.warmServerHost, port: configuration.warmServerPort)
    self.contentModeManager = ContentModeManager(configPath: configuration.contentModeConfigPath)
    self.characterLoader = CharacterLoader(configPath: configuration.characterConfigPath)
    self.promptOptimizer = PromptOptimizer(configuration: configuration.optimizer, logger: logger)

    // Ensure output directory exists
    let fileManager = FileManager.default
    let outputDir = configuration.outputDirectory
    if !fileManager.fileExists(atPath: outputDir) {
      try? fileManager.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
      logger.info("Created output directory: \(outputDir)")
    }

    // Log startup info
    let charNames = characterLoader.allNames()
    let mode = contentModeManager.current
    logger.info("Content mode: \(mode.rawValue), characters: \(charNames.joined(separator: ", ")), enhance: \(configuration.optimizer.enabled)")
  }

  /// Start the bot. Blocks until stopped.
  public func run() async throws {
    isRunning = true
    logger.info("ImageBotCoordinator: starting (WarmServer at \(config.warmServerHost):\(config.warmServerPort))")

    await bot.startPolling { [self] update in
      await self.handleUpdate(update)
    }
  }

  /// Signal graceful shutdown.
  public func shutdown() {
    logger.info("ImageBotCoordinator: shutdown requested")
    isRunning = false
    bot.stop()
  }

  // MARK: - Session Helpers

  private func getSession(chatId: Int) -> SessionState {
    return sessions[chatId] ?? SessionState(enhanceEnabled: config.optimizer.enabled)
  }

  private func updateSession(chatId: Int, _ block: (inout SessionState) -> Void) {
    var state = getSession(chatId: chatId)
    block(&state)
    sessions[chatId] = state
  }

  // MARK: - Update Handling

  private func handleUpdate(_ update: TelegramUpdate) async {
    // Handle text messages
    if let message = update.message, let text = message.text {
      await handleTextMessage(message: message, text: text)
      return
    }

    // Future: handle callback queries, photos, etc.
  }

  private func handleTextMessage(message: TelegramMessage, text: String) async {
    let command = TelegramCommandParser.parse(text, inDiscussMode: false)
    let chatId = message.chatId

    switch command {
    case .help:
      await sendHelp(chatId: chatId)

    case .status:
      await sendStatus(chatId: chatId)

    case .neutral:
      contentModeManager.set(.neutral)
      let emoji = ContentModeManager.emoji(for: .neutral)
      let name = ContentModeManager.displayName(for: .neutral)
      let _ = try? await bot.sendMessage(chatId: chatId, text: "\(emoji) Mode set to <b>\(name)</b>")

    case .banana:
      contentModeManager.set(.banana)
      let emoji = ContentModeManager.emoji(for: .banana)
      let name = ContentModeManager.displayName(for: .banana)
      let _ = try? await bot.sendMessage(chatId: chatId, text: "\(emoji) Mode set to <b>\(name)</b>")

    case .avocado:
      contentModeManager.set(.avocado)
      let emoji = ContentModeManager.emoji(for: .avocado)
      let name = ContentModeManager.displayName(for: .avocado)
      let _ = try? await bot.sendMessage(chatId: chatId, text: "\(emoji) Mode set to <b>\(name)</b>")

    case .enhance(let on):
      let session = getSession(chatId: chatId)
      let newValue: Bool
      if let on = on {
        newValue = on
      } else {
        // Toggle
        newValue = !session.enhanceEnabled
      }
      updateSession(chatId: chatId) { $0.enhanceEnabled = newValue }
      let stateText = newValue ? "ON" : "OFF"
      let emoji = newValue ? "\u{2728}" : "\u{1F6D1}"  // sparkles or stop
      let _ = try? await bot.sendMessage(chatId: chatId, text: "\(emoji) Prompt enhancement: <b>\(stateText)</b>")

    case .render(let prompt):
      await handleRender(chatId: chatId, prompt: prompt, messageId: message.messageId)

    default:
      // Phase 3+ commands not yet implemented
      let _ = try? await bot.sendMessage(
        chatId: chatId,
        text: "Command not yet available. Send a text prompt to generate an image, or /help for commands."
      )
    }
  }

  // MARK: - Render Flow

  private func handleRender(chatId: Int, prompt: String, messageId: Int) async {
    guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      let _ = try? await bot.sendMessage(chatId: chatId, text: "Send me a prompt and I'll generate an image.")
      return
    }

    // Parse prompt for character detection and inline mode overrides
    let currentMode = contentModeManager.current
    let parsed = TelegramCommandParser.parsePrompt(prompt, defaultMode: currentMode.rawValue)

    // Resolve effective content mode (inline override > session mode)
    let effectiveMode: ContentModeManager.Mode
    if let overrideMode = parsed.contentMode, let mode = ContentModeManager.Mode(rawValue: overrideMode) {
      effectiveMode = mode
    } else {
      effectiveMode = currentMode
    }

    // Look up character description
    var characterDescription: String? = nil
    if let charName = parsed.character {
      characterDescription = characterLoader.description(for: charName, mode: effectiveMode)
    }

    // Send status message
    let session = getSession(chatId: chatId)
    let modeEmoji = ContentModeManager.emoji(for: effectiveMode)
    let enhanceEmoji = session.enhanceEnabled ? "\u{2728}" : ""
    let charTag = parsed.character != nil ? " [\(parsed.character!)]" : ""
    let statusText = "\(modeEmoji)\(enhanceEmoji) Rendering\(charTag)..."
    let statusResult = try? await bot.sendMessage(chatId: chatId, text: statusText)

    // Optimize prompt if enhancement is enabled
    let finalPrompt: String
    var optimizeNote: String? = nil
    if session.enhanceEnabled {
      let result = await promptOptimizer.optimize(
        prompt: parsed.prompt,
        character: parsed.character,
        characterDescription: characterDescription,
        contentMode: effectiveMode.rawValue
      )
      finalPrompt = result.prompt
      optimizeNote = result.note
      if result.enhanced {
        logger.info("Prompt enhanced: \(result.prompt.prefix(100))...")
      }
    } else {
      // Enhancement off — inject character description manually but use raw prompt
      if let desc = characterDescription {
        finalPrompt = "\(desc)\n\n\(parsed.prompt)"
      } else {
        finalPrompt = parsed.prompt
      }
    }

    // Generate via WarmServer
    let outputFilename = "telegram-\(Int(Date().timeIntervalSince1970))-\(UInt32.random(in: 0...999999)).png"
    let outputPath = (config.outputDirectory as NSString).appendingPathComponent(outputFilename)

    do {
      // Build generate request
      var body: [String: Any] = [
        "prompt": finalPrompt,
        "outputPath": outputPath
      ]

      // Use default dimensions for Telegram (1024x1024)
      body["width"] = 1024
      body["height"] = 1024

      let jsonData = try JSONSerialization.data(withJSONObject: body)
      let (status, responseData) = try await warmServer.post("/v1/generate", body: jsonData)

      guard status == 200 else {
        let errorMsg = parseErrorMessage(responseData) ?? "WarmServer returned status \(status)"
        throw CoordinatorError.generateFailed(errorMsg)
      }

      // Parse response to get output path
      guard let responseJSON = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any],
            let actualOutputPath = responseJSON["outputPath"] as? String else {
        throw CoordinatorError.generateFailed("Invalid response from WarmServer")
      }

      let durationMs = responseJSON["durationMs"] as? Int ?? 0

      // Read the generated image
      let imageURL = URL(fileURLWithPath: actualOutputPath)
      let imageData = try Data(contentsOf: imageURL)

      // Build caption
      let caption = buildCaption(
        prompt: parsed.prompt,
        character: parsed.character,
        mode: effectiveMode,
        durationMs: durationMs,
        enhanced: session.enhanceEnabled,
        optimizeNote: optimizeNote
      )

      if imageData.count > 8_000_000 {
        // Large image — send as document to avoid Telegram's 10MB sendPhoto limit
        let _ = try await bot.sendDocument(
          chatId: chatId,
          fileData: imageData,
          filename: outputFilename,
          mimeType: "image/png",
          caption: caption
        )
      } else {
        let _ = try await bot.sendPhoto(
          chatId: chatId,
          imageData: imageData,
          filename: outputFilename,
          caption: caption
        )
      }

      // Update session state
      updateSession(chatId: chatId) {
        $0.lastPrompt = parsed.prompt
        $0.lastImagePath = actualOutputPath
      }

      // Copy to gallery if configured
      if let galleryDir = config.galleryDirectory {
        let galleryPath = (galleryDir as NSString).appendingPathComponent(outputFilename)
        try? FileManager.default.copyItem(atPath: actualOutputPath, toPath: galleryPath)
      }

      logger.info("Render complete: \(outputFilename) (\(durationMs)ms, \(imageData.count / 1024)KB, mode: \(effectiveMode.rawValue))")

    } catch let error as WarmServerClientError where error.localizedDescription.contains("not running") {
      let _ = try? await bot.sendMessage(
        chatId: chatId,
        text: "WarmServer not available \u{2014} start <code>ComfyBox serve</code> first."
      )
      logger.error("Render failed: WarmServer not running")
    } catch {
      let _ = try? await bot.sendMessage(
        chatId: chatId,
        text: "Render failed: \(error.localizedDescription)"
      )
      logger.error("Render failed: \(error)")
    }
  }

  // MARK: - Help & Status

  private func sendHelp(chatId: Int) async {
    let mode = contentModeManager.current
    let modeEmoji = ContentModeManager.emoji(for: mode)
    let modeName = ContentModeManager.displayName(for: mode)
    let session = getSession(chatId: chatId)
    let enhanceState = session.enhanceEnabled ? "ON" : "OFF"
    let charList = characterLoader.allNames().map { $0.capitalized }.joined(separator: ", ")

    let helpText = """
    <b>ComfyBox Image Bot</b>

    Send any text to generate an image.

    <b>Content Modes:</b>
    /neutral or /apple \u{2014} SFW mode
    /banana \u{2014} Suggestive mode
    /avocado \u{2014} Explicit mode

    <b>Settings:</b>
    /enhance [on|off] \u{2014} Toggle prompt optimization

    <b>Info:</b>
    /help \u{2014} Show this help
    /status \u{2014} WarmServer status

    <b>Current Settings:</b>
    \(modeEmoji) Mode: <b>\(modeName)</b>
    \u{2728} Enhance: <b>\(enhanceState)</b>
    \u{1F464} Characters: \(charList.isEmpty ? "none loaded" : charList)

    <b>Tips:</b>
    \u{2022} Character names (\(charList)) are detected automatically
    \u{2022} Inline mode override: include /avocado in your prompt for one render
    \u{2022} Be descriptive \u{2014} more detail = better results
    """
    let _ = try? await bot.sendMessage(chatId: chatId, text: helpText)
  }

  private func sendStatus(chatId: Int) async {
    let mode = contentModeManager.current
    let modeEmoji = ContentModeManager.emoji(for: mode)
    let session = getSession(chatId: chatId)

    do {
      let (status, data) = try await warmServer.get("/health")
      guard status == 200,
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        let _ = try? await bot.sendMessage(chatId: chatId, text: "WarmServer returned status \(status)")
        return
      }

      let serverStatus = json["status"] as? String ?? "unknown"
      let model = json["model"] as? String ?? "none"
      let uptime = json["uptimeSeconds"] as? Int ?? 0
      let queueLength = json["queueLength"] as? Int ?? 0
      let totalGens = json["totalGenerations"] as? Int ?? 0

      let botUptime = Int(Date().timeIntervalSince(startTime))
      let charCount = characterLoader.allNames().count
      let enhanceState = session.enhanceEnabled ? "ON" : "OFF"

      let statusText = """
      <b>ComfyBox Status</b>

      <b>WarmServer:</b> \(serverStatus)
      <b>Model:</b> <code>\(model)</code>
      <b>Queue:</b> \(queueLength) pending
      <b>Total renders:</b> \(totalGens)
      <b>Server uptime:</b> \(formatDuration(uptime))

      \(modeEmoji) <b>Mode:</b> \(ContentModeManager.displayName(for: mode))
      \u{2728} <b>Enhance:</b> \(enhanceState)
      \u{1F464} <b>Characters:</b> \(charCount) loaded

      <b>Bot uptime:</b> \(formatDuration(botUptime))
      <b>Output:</b> <code>\(config.outputDirectory)</code>
      """
      let _ = try? await bot.sendMessage(chatId: chatId, text: statusText)

    } catch {
      let _ = try? await bot.sendMessage(
        chatId: chatId,
        text: "WarmServer not available \u{2014} start <code>ComfyBox serve</code> first."
      )
    }
  }

  // MARK: - Helpers

  private func buildCaption(
    prompt: String,
    character: String?,
    mode: ContentModeManager.Mode,
    durationMs: Int,
    enhanced: Bool,
    optimizeNote: String?
  ) -> String {
    var parts: [String] = []

    // Mode and character tags
    let modeEmoji = ContentModeManager.emoji(for: mode)
    if let char = character {
      parts.append("\(modeEmoji) [\(char)]")
    } else {
      parts.append(modeEmoji)
    }

    // Truncated prompt
    let truncatedPrompt = prompt.count > 180 ? String(prompt.prefix(177)) + "..." : prompt
    parts.append(truncatedPrompt)

    // Duration
    if durationMs > 0 {
      let seconds = Double(durationMs) / 1000.0
      parts.append(String(format: "(%.1fs)", seconds))
    }

    // Enhancement indicator
    if enhanced {
      parts.append("\u{2728}")
    }

    return parts.joined(separator: " ")
  }

  private func formatDuration(_ seconds: Int) -> String {
    if seconds < 60 { return "\(seconds)s" }
    if seconds < 3600 { return "\(seconds / 60)m \(seconds % 60)s" }
    let hours = seconds / 3600
    let mins = (seconds % 3600) / 60
    return "\(hours)h \(mins)m"
  }

  private func parseErrorMessage(_ data: Data) -> String? {
    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
    return json["error"] as? String
  }
}

// MARK: - Errors

enum CoordinatorError: Error, LocalizedError {
  case generateFailed(String)

  var errorDescription: String? {
    switch self {
    case .generateFailed(let msg): return msg
    }
  }
}
