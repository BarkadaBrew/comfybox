// ImageBotCoordinator.swift — Orchestrates Telegram bot: parse -> render -> send.
//
// Phase 1: Handles text prompts via WarmServer, /help, and /status.
// Connects to a running WarmServer instance via WarmServerClient.

import Foundation
import Logging

public final class ImageBotCoordinator {
  public struct Configuration: Sendable {
    public let telegram: TelegramBot.Configuration
    public let warmServerHost: String
    public let warmServerPort: UInt16
    public let outputDirectory: String

    public init(
      telegram: TelegramBot.Configuration,
      warmServerHost: String = "127.0.0.1",
      warmServerPort: UInt16 = 7862,
      outputDirectory: String = ("~/Pictures/ComfyBox/Telegram" as NSString).expandingTildeInPath
    ) {
      self.telegram = telegram
      self.warmServerHost = warmServerHost
      self.warmServerPort = warmServerPort
      self.outputDirectory = outputDirectory
    }
  }

  private let config: Configuration
  private let bot: TelegramBot
  private let warmServer: WarmServerClient
  private let logger: Logger
  private var isRunning = false
  private let startTime = Date()

  public init(configuration: Configuration, logger: Logger = Logger(label: "comfybox.telegram.coordinator")) {
    self.config = configuration
    self.logger = logger
    self.bot = TelegramBot(configuration: configuration.telegram, logger: logger)
    self.warmServer = WarmServerClient(host: configuration.warmServerHost, port: configuration.warmServerPort)

    // Ensure output directory exists
    let fileManager = FileManager.default
    let outputDir = configuration.outputDirectory
    if !fileManager.fileExists(atPath: outputDir) {
      try? fileManager.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
      logger.info("Created output directory: \(outputDir)")
    }
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

    case .render(let prompt):
      await handleRender(chatId: chatId, prompt: prompt, messageId: message.messageId)

    default:
      // Phase 2+ commands not yet implemented
      let _ = try? await bot.sendMessage(
        chatId: chatId,
        text: "Command not yet available in Phase 1. Send a text prompt to generate an image, or /help for commands."
      )
    }
  }

  // MARK: - Render Flow

  private func handleRender(chatId: Int, prompt: String, messageId: Int) async {
    guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      let _ = try? await bot.sendMessage(chatId: chatId, text: "Send me a prompt and I'll generate an image.")
      return
    }

    // Parse prompt for character detection
    let parsed = TelegramCommandParser.parsePrompt(prompt)

    // Send status message
    let statusResult = try? await bot.sendMessage(chatId: chatId, text: "Rendering...")
    let statusMessageId = statusResult?.messageId

    // Generate via WarmServer
    let outputFilename = "telegram-\(Int(Date().timeIntervalSince1970))-\(UInt32.random(in: 0...999999)).png"
    let outputPath = (config.outputDirectory as NSString).appendingPathComponent(outputFilename)

    do {
      // Build generate request
      var body: [String: Any] = [
        "prompt": parsed.prompt,
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

      // Determine if we should use sendPhoto or sendDocument (>8MB)
      let caption = buildCaption(prompt: parsed.prompt, character: parsed.character, durationMs: durationMs)

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

      logger.info("Render complete: \(outputFilename) (\(durationMs)ms, \(imageData.count / 1024)KB)")

    } catch let error as WarmServerClientError where error.localizedDescription.contains("not running") {
      let _ = try? await bot.sendMessage(
        chatId: chatId,
        text: "WarmServer not available — start <code>ComfyBox serve</code> first."
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
    let helpText = """
    <b>ComfyBox Image Bot</b> (Phase 1)

    Send any text to generate an image.

    <b>Commands:</b>
    /help — Show this help
    /status — WarmServer status

    <b>Examples:</b>
    <i>a golden retriever on a beach at sunset</i>
    <i>cyberpunk cityscape with neon lights</i>

    <b>Tips:</b>
    • Be descriptive — more detail = better results
    • Character names (Kira, Bree, Todd) are detected automatically
    """
    let _ = try? await bot.sendMessage(chatId: chatId, text: helpText)
  }

  private func sendStatus(chatId: Int) async {
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

      let statusText = """
      <b>ComfyBox Status</b>

      <b>WarmServer:</b> \(serverStatus)
      <b>Model:</b> <code>\(model)</code>
      <b>Queue:</b> \(queueLength) pending
      <b>Total renders:</b> \(totalGens)
      <b>Server uptime:</b> \(formatDuration(uptime))

      <b>Bot uptime:</b> \(formatDuration(botUptime))
      <b>Output:</b> <code>\(config.outputDirectory)</code>
      """
      let _ = try? await bot.sendMessage(chatId: chatId, text: statusText)

    } catch {
      let _ = try? await bot.sendMessage(
        chatId: chatId,
        text: "WarmServer not available — start <code>ComfyBox serve</code> first."
      )
    }
  }

  // MARK: - Helpers

  private func buildCaption(prompt: String, character: String?, durationMs: Int) -> String {
    var parts: [String] = []
    if let char = character {
      parts.append("[\(char)]")
    }
    let truncatedPrompt = prompt.count > 200 ? String(prompt.prefix(197)) + "..." : prompt
    parts.append(truncatedPrompt)
    if durationMs > 0 {
      let seconds = Double(durationMs) / 1000.0
      parts.append(String(format: "(%.1fs)", seconds))
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
