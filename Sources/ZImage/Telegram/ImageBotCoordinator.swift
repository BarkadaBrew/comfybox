// ImageBotCoordinator.swift — Orchestrates Telegram bot: parse -> render -> send.
//
// Phase 1: Handles text prompts via WarmServer, /help, and /status.
// Phase 2: Content modes, character injection, prompt optimization.
// Phase 3: Full command set — batch, vary, sequence, upscale, aspect, cfg,
//          seed, polish, post-processing (saturation, temp, film), reset.
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

  private let config: Configuration
  private let bot: TelegramBot
  private let warmServer: WarmServerClient
  private let contentModeManager: ContentModeManager
  private let characterLoader: CharacterLoader
  private let promptOptimizer: PromptOptimizer
  private let sessions: SessionState
  private let logger: Logger
  private var isRunning = false
  private let startTime = Date()

  public init(configuration: Configuration, logger: Logger = Logger(label: "comfybox.telegram.coordinator")) {
    self.config = configuration
    self.logger = logger
    self.bot = TelegramBot(configuration: configuration.telegram, logger: logger)
    self.warmServer = WarmServerClient(host: configuration.warmServerHost, port: configuration.warmServerPort)
    self.contentModeManager = ContentModeManager(configPath: configuration.contentModeConfigPath)
    self.characterLoader = CharacterLoader(configPath: configuration.characterConfigPath)
    self.promptOptimizer = PromptOptimizer(configuration: configuration.optimizer, logger: logger)
    self.sessions = SessionState(defaultEnhance: configuration.optimizer.enabled)

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

  // MARK: - Update Handling

  private func handleUpdate(_ update: TelegramUpdate) async {
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
    // -- Phase 1 --
    case .help:
      await sendHelp(chatId: chatId)

    case .status:
      await sendStatus(chatId: chatId)

    case .render(let prompt):
      await handleRender(chatId: chatId, prompt: prompt, messageId: message.messageId)

    // -- Phase 2 --
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
      let state = sessions.getState(chatId: chatId)
      let newValue = on ?? !state.enhanceEnabled
      sessions.updateState(chatId: chatId) { $0.enhanceEnabled = newValue }
      let stateText = newValue ? "ON" : "OFF"
      let emoji = newValue ? "\u{2728}" : "\u{1F6D1}"
      let _ = try? await bot.sendMessage(chatId: chatId, text: "\(emoji) Prompt enhancement: <b>\(stateText)</b>")

    // -- Phase 3: Generation --
    case .batch(let count, let prompt):
      await handleBatch(chatId: chatId, count: count, prompt: prompt)

    case .vary(let count, let prompt):
      await handleVary(chatId: chatId, count: count, prompt: prompt)

    case .sequence(let count, let story):
      await handleSequence(chatId: chatId, count: count, story: story)

    // -- Phase 3: Toggles --
    case .upscale(let on):
      await handleToggle(chatId: chatId, name: "Upscale (SeedVR 2x)", keyPath: \.upscaleEnabled, value: on)

    case .polish(let on):
      await handleToggle(chatId: chatId, name: "Polish (two-pass)", keyPath: \.polishEnabled, value: on)

    case .verbose(let on):
      await handleToggle(chatId: chatId, name: "Verbose", keyPath: \.verboseEnabled, value: on)

    case .autoVideo(let on):
      await handleToggle(chatId: chatId, name: "Auto-video", keyPath: \.autoVideoEnabled, value: on)

    case .resolution(let target):
      await handleResolution(chatId: chatId, target: target)

    // -- Phase 3: Settings --
    case .aspect(let mode):
      await handleAspect(chatId: chatId, mode: mode)

    case .cfg(let value):
      await handleCfg(chatId: chatId, value: value)

    case .seed(let value):
      await handleSeed(chatId: chatId, value: value)

    // -- Phase 3: Post-processing --
    case .saturation(let value):
      await handleSaturation(chatId: chatId, value: value)

    case .colorTemp(let kelvin):
      await handleColorTemp(chatId: chatId, kelvin: kelvin)

    case .film(let lookId):
      await handleFilm(chatId: chatId, lookId: lookId)

    case .reset:
      await handleReset(chatId: chatId)

    case .look:
      await sendLookList(chatId: chatId)

    // -- Deferred --
    case .video:
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "Video generation routes through the daemon. Use Bree's <code>/video</code> command on the server.")

    default:
      let _ = try? await bot.sendMessage(
        chatId: chatId,
        text: "Command not yet available. Send a text prompt to generate an image, or /help for commands."
      )
    }
  }

  // MARK: - Toggle Handler

  private func handleToggle(chatId: Int, name: String, keyPath: WritableKeyPath<ChatState, Bool>, value: Bool?) async {
    let state = sessions.getState(chatId: chatId)
    let current = state[keyPath: keyPath]
    let newValue = value ?? !current
    sessions.updateState(chatId: chatId) { $0[keyPath: keyPath] = newValue }
    let stateText = newValue ? "ON" : "OFF"
    let emoji = newValue ? "\u{2705}" : "\u{274C}"
    let _ = try? await bot.sendMessage(chatId: chatId, text: "\(emoji) \(name): <b>\(stateText)</b>")
  }

  // MARK: - Resolution

  private func handleResolution(chatId: Int, target: String?) async {
    if let target = target {
      sessions.updateState(chatId: chatId) { $0.resolutionTarget = target }
      // Also enable upscale since resolution targets require it
      sessions.updateState(chatId: chatId) { $0.upscaleEnabled = true }
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F4D0} Resolution target: <b>\(target.uppercased())</b> (upscale enabled)")
    } else {
      sessions.updateState(chatId: chatId) {
        $0.resolutionTarget = nil
        $0.upscaleEnabled = false
      }
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F4D0} Resolution target: <b>OFF</b> (upscale disabled)")
    }
  }

  // MARK: - Aspect

  private func handleAspect(chatId: Int, mode: String?) async {
    guard let mode = mode?.lowercased().trimmingCharacters(in: .whitespaces) else {
      let state = sessions.getState(chatId: chatId)
      let dims = state.aspectDimensions()
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F4D0} Current aspect: <b>\(state.aspectMode)</b> (\(dims.width)x\(dims.height))\n\nOptions: square, portrait, landscape, wide, tall")
      return
    }
    let validModes = ["square", "portrait", "landscape", "wide", "tall"]
    guard validModes.contains(mode) else {
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "Invalid aspect mode. Options: <code>square</code>, <code>portrait</code>, <code>landscape</code>, <code>wide</code>, <code>tall</code>")
      return
    }
    sessions.updateState(chatId: chatId) { $0.aspectMode = mode }
    let dims = ChatState.dimensionsForAspect(mode)
    let _ = try? await bot.sendMessage(chatId: chatId,
      text: "\u{1F4D0} Aspect: <b>\(mode)</b> (\(dims.width)x\(dims.height))")
  }

  // MARK: - CFG

  private func handleCfg(chatId: Int, value: Double?) async {
    if let value = value {
      guard value >= 0 && value <= 20 else {
        let _ = try? await bot.sendMessage(chatId: chatId, text: "CFG value must be between 0 and 20.")
        return
      }
      sessions.updateState(chatId: chatId) { $0.cfgOverride = value }
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{2699}\u{FE0F} CFG override: <b>\(String(format: "%.1f", value))</b>")
    } else {
      sessions.updateState(chatId: chatId) { $0.cfgOverride = nil }
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{2699}\u{FE0F} CFG override: <b>OFF</b> (using model default)")
    }
  }

  // MARK: - Seed

  private func handleSeed(chatId: Int, value: Int?) async {
    sessions.updateState(chatId: chatId) { $0.seedLock = value }
    if let value = value {
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F3B2} Seed locked: <b>\(value)</b>")
    } else {
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F3B2} Seed: <b>random</b>")
    }
  }

  // MARK: - Post-Processing Settings

  private func handleSaturation(chatId: Int, value: Double?) async {
    if let value = value {
      guard value >= 0 && value <= 2 else {
        let _ = try? await bot.sendMessage(chatId: chatId, text: "Saturation must be between 0 and 2.")
        return
      }
      sessions.updateState(chatId: chatId) { $0.saturation = value }
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F3A8} Saturation: <b>\(String(format: "%.1f", value))</b>")
    } else {
      sessions.updateState(chatId: chatId) { $0.saturation = nil }
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F3A8} Saturation: <b>OFF</b>")
    }
  }

  private func handleColorTemp(chatId: Int, kelvin: Int?) async {
    if let kelvin = kelvin {
      guard kelvin >= 2000 && kelvin <= 10000 else {
        let _ = try? await bot.sendMessage(chatId: chatId, text: "Color temperature must be between 2000 and 10000K.")
        return
      }
      sessions.updateState(chatId: chatId) { $0.colorTemp = kelvin }
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F321}\u{FE0F} Color temperature: <b>\(kelvin)K</b>")
    } else {
      sessions.updateState(chatId: chatId) { $0.colorTemp = nil }
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F321}\u{FE0F} Color temperature: <b>OFF</b>")
    }
  }

  private func handleFilm(chatId: Int, lookId: String?) async {
    guard let lookId = lookId else {
      // No argument — show current + available looks
      await sendLookList(chatId: chatId)
      return
    }
    if lookId.lowercased() == "off" {
      sessions.updateState(chatId: chatId) { $0.filmLook = nil }
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F3AC} Film look: <b>OFF</b>")
      return
    }
    guard PostProcessor.findLook(lookId) != nil else {
      let available = PostProcessor.availableLooks().map { "<code>\($0.id)</code>" }.joined(separator: ", ")
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "Unknown film look '<code>\(lookId)</code>'.\n\nAvailable: \(available)")
      return
    }
    sessions.updateState(chatId: chatId) { $0.filmLook = lookId }
    let name = PostProcessor.findLook(lookId)?.name ?? lookId
    let _ = try? await bot.sendMessage(chatId: chatId,
      text: "\u{1F3AC} Film look: <b>\(name)</b>")
  }

  private func handleReset(chatId: Int) async {
    sessions.updateState(chatId: chatId) { $0.resetPostProcessing() }
    let _ = try? await bot.sendMessage(chatId: chatId,
      text: "\u{1F504} Post-processing settings cleared (saturation, color temp, film look).")
  }

  private func sendLookList(chatId: Int) async {
    let looks = PostProcessor.availableLooks()
    let state = sessions.getState(chatId: chatId)
    var lines: [String] = ["<b>Available Film Looks:</b>\n"]
    for look in looks {
      let active = state.filmLook?.lowercased() == look.id.lowercased() ? " \u{2705}" : ""
      lines.append("\u{2022} <code>\(look.id)</code> \u{2014} \(look.name)\(active)")
    }
    lines.append("\nUsage: <code>/film kodak-portra</code> or <code>/film off</code>")
    let _ = try? await bot.sendMessage(chatId: chatId, text: lines.joined(separator: "\n"))
  }

  // MARK: - Single Render

  private func handleRender(chatId: Int, prompt: String, messageId: Int) async {
    guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      let _ = try? await bot.sendMessage(chatId: chatId, text: "Send me a prompt and I'll generate an image.")
      return
    }

    let state = sessions.getState(chatId: chatId)
    let parsed = parseAndResolve(prompt: prompt)

    // Send status
    let statusText = buildStatusText(parsed: parsed, state: state, index: nil, total: nil)
    let statusResult = try? await bot.sendMessage(chatId: chatId, text: statusText)

    // Generate
    let result = await generateImage(
      prompt: parsed.prompt,
      character: parsed.character,
      characterDescription: parsed.characterDescription,
      effectiveMode: parsed.effectiveMode,
      state: state,
      seed: state.seedLock
    )

    switch result {
    case .success(let render):
      // Post-process
      let finalData = applyPostProcessing(imageData: render.imageData, state: state, wasUpscaled: render.wasUpscaled)

      // Build caption
      let caption = buildCaption(
        prompt: parsed.rawPrompt,
        character: parsed.character,
        mode: parsed.effectiveMode,
        durationMs: render.durationMs,
        enhanced: state.enhanceEnabled,
        seed: render.seed,
        state: state
      )

      // Send
      await sendImage(chatId: chatId, imageData: finalData, filename: render.filename, caption: caption)

      // Update session
      sessions.updateState(chatId: chatId) {
        $0.lastPrompt = parsed.rawPrompt
        $0.lastImagePath = render.outputPath
        $0.lastSeed = render.seed
      }

      // Copy to gallery
      copyToGallery(outputPath: render.outputPath, filename: render.filename)

      logger.info("Render complete: \(render.filename) (\(render.durationMs)ms)")

    case .failure(let error):
      await sendError(chatId: chatId, error: error)
    }
  }

  // MARK: - Batch

  private func handleBatch(chatId: Int, count: Int, prompt: String) async {
    let state = sessions.getState(chatId: chatId)
    let parsed = parseAndResolve(prompt: prompt)

    let statusText = "\u{1F4E6} Batch: rendering \(count) images..."
    let statusResult = try? await bot.sendMessage(chatId: chatId, text: statusText)
    let statusMsgId = statusResult?.messageId

    for i in 0..<count {
      // Update progress
      if let msgId = statusMsgId {
        let _ = try? await bot.sendMessage(chatId: chatId,
          text: "\u{1F4E6} Rendering \(i + 1)/\(count)...")
      }

      let result = await generateImage(
        prompt: parsed.prompt,
        character: parsed.character,
        characterDescription: parsed.characterDescription,
        effectiveMode: parsed.effectiveMode,
        state: state,
        seed: nil  // Different seed each time
      )

      switch result {
      case .success(let render):
        let finalData = applyPostProcessing(imageData: render.imageData, state: state, wasUpscaled: render.wasUpscaled)
        let caption = "\(i + 1)/\(count) \u{2014} seed:\(render.seed ?? 0)"
        await sendImage(chatId: chatId, imageData: finalData, filename: render.filename, caption: caption)
        copyToGallery(outputPath: render.outputPath, filename: render.filename)

      case .failure(let error):
        await sendError(chatId: chatId, error: error)
        return  // Abort batch on error
      }
    }

    let _ = try? await bot.sendMessage(chatId: chatId,
      text: "\u{2705} Batch complete: \(count) images delivered.")
  }

  // MARK: - Vary

  private func handleVary(chatId: Int, count: Int, prompt: String) async {
    let state = sessions.getState(chatId: chatId)
    let parsed = parseAndResolve(prompt: prompt)

    let _ = try? await bot.sendMessage(chatId: chatId,
      text: "\u{1F500} Vary: generating \(count) prompt variations...")

    for i in 0..<count {
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F500} Variation \(i + 1)/\(count)...")

      // Call optimizer with variation instruction for each
      let variedPrompt: String
      if state.enhanceEnabled {
        let variationInstruction = "Create variation #\(i + 1) of this scene. Change the angle, lighting, or composition while keeping the same subject and mood."
        let result = await promptOptimizer.optimize(
          prompt: "\(parsed.prompt) [\(variationInstruction)]",
          character: parsed.character,
          characterDescription: parsed.characterDescription,
          contentMode: parsed.effectiveMode.rawValue
        )
        variedPrompt = result.prompt
      } else {
        variedPrompt = parsed.prompt
      }

      let renderResult = await generateImageDirect(
        finalPrompt: variedPrompt,
        state: state,
        seed: nil
      )

      switch renderResult {
      case .success(let render):
        let finalData = applyPostProcessing(imageData: render.imageData, state: state, wasUpscaled: render.wasUpscaled)
        let caption = "Variation \(i + 1)/\(count)"
        await sendImage(chatId: chatId, imageData: finalData, filename: render.filename, caption: caption)
        copyToGallery(outputPath: render.outputPath, filename: render.filename)

      case .failure(let error):
        await sendError(chatId: chatId, error: error)
        return
      }
    }

    let _ = try? await bot.sendMessage(chatId: chatId,
      text: "\u{2705} Variations complete: \(count) images delivered.")
  }

  // MARK: - Sequence

  private func handleSequence(chatId: Int, count: Int, story: String) async {
    let state = sessions.getState(chatId: chatId)
    let parsed = parseAndResolve(prompt: story)

    let _ = try? await bot.sendMessage(chatId: chatId,
      text: "\u{1F3AC} Sequence: breaking story into \(count) frames...")

    // Use optimizer to break story into N sequential frame prompts
    let framePrompts: [String]
    if state.enhanceEnabled {
      let breakdownInstruction = """
      Break this story into exactly \(count) sequential scene descriptions for image generation.
      Each scene should be a complete, self-contained image prompt.
      Number each scene. Output ONLY the numbered scenes, nothing else.
      Story: \(parsed.prompt)
      """

      let breakdownResult = await promptOptimizer.optimize(
        prompt: breakdownInstruction,
        character: parsed.character,
        characterDescription: parsed.characterDescription,
        contentMode: parsed.effectiveMode.rawValue
      )

      // Parse the numbered output into individual prompts
      framePrompts = parseNumberedScenes(breakdownResult.prompt, expectedCount: count)
    } else {
      // No optimizer — split by sentence or use the same prompt for each frame
      framePrompts = splitStoryIntoFrames(story, count: count)
    }

    let actualCount = min(framePrompts.count, count)

    for i in 0..<actualCount {
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F3AC} Frame \(i + 1)/\(actualCount)...")

      // Optimize each frame prompt individually
      let framePrompt: String
      if state.enhanceEnabled {
        let result = await promptOptimizer.optimize(
          prompt: framePrompts[i],
          character: parsed.character,
          characterDescription: parsed.characterDescription,
          contentMode: parsed.effectiveMode.rawValue
        )
        framePrompt = result.prompt
      } else {
        if let desc = parsed.characterDescription {
          framePrompt = "\(desc)\n\n\(framePrompts[i])"
        } else {
          framePrompt = framePrompts[i]
        }
      }

      let renderResult = await generateImageDirect(
        finalPrompt: framePrompt,
        state: state,
        seed: nil
      )

      switch renderResult {
      case .success(let render):
        let finalData = applyPostProcessing(imageData: render.imageData, state: state, wasUpscaled: render.wasUpscaled)
        let truncatedScene = framePrompts[i].count > 100
          ? String(framePrompts[i].prefix(97)) + "..."
          : framePrompts[i]
        let caption = "Frame \(i + 1)/\(actualCount) \u{2014} \(truncatedScene)"
        await sendImage(chatId: chatId, imageData: finalData, filename: render.filename, caption: caption)
        copyToGallery(outputPath: render.outputPath, filename: render.filename)

      case .failure(let error):
        await sendError(chatId: chatId, error: error)
        return
      }
    }

    let _ = try? await bot.sendMessage(chatId: chatId,
      text: "\u{2705} Sequence complete: \(actualCount) frames delivered.")
  }

  // MARK: - Render Pipeline

  private struct ParsedContext {
    let rawPrompt: String
    let prompt: String
    let character: String?
    let characterDescription: String?
    let effectiveMode: ContentModeManager.Mode
  }

  private func parseAndResolve(prompt: String) -> ParsedContext {
    let currentMode = contentModeManager.current
    let parsed = TelegramCommandParser.parsePrompt(prompt, defaultMode: currentMode.rawValue)
    let effectiveMode: ContentModeManager.Mode
    if let overrideMode = parsed.contentMode, let mode = ContentModeManager.Mode(rawValue: overrideMode) {
      effectiveMode = mode
    } else {
      effectiveMode = currentMode
    }
    var characterDescription: String? = nil
    if let charName = parsed.character {
      characterDescription = characterLoader.description(for: charName, mode: effectiveMode)
    }
    return ParsedContext(
      rawPrompt: prompt,
      prompt: parsed.prompt,
      character: parsed.character,
      characterDescription: characterDescription,
      effectiveMode: effectiveMode
    )
  }

  private struct RenderResult {
    let imageData: Data
    let outputPath: String
    let filename: String
    let durationMs: Int
    let seed: Int?
    let wasUpscaled: Bool
  }

  /// Full render pipeline: optimize prompt -> generate -> optional upscale -> optional polish.
  private func generateImage(
    prompt: String,
    character: String?,
    characterDescription: String?,
    effectiveMode: ContentModeManager.Mode,
    state: ChatState,
    seed: Int?
  ) async -> Result<RenderResult, RenderError> {
    // Optimize prompt
    let finalPrompt: String
    if state.enhanceEnabled {
      let result = await promptOptimizer.optimize(
        prompt: prompt,
        character: character,
        characterDescription: characterDescription,
        contentMode: effectiveMode.rawValue
      )
      finalPrompt = result.prompt
    } else {
      if let desc = characterDescription {
        finalPrompt = "\(desc)\n\n\(prompt)"
      } else {
        finalPrompt = prompt
      }
    }

    return await generateImageDirect(finalPrompt: finalPrompt, state: state, seed: seed)
  }

  /// Generate image with an already-resolved prompt. Handles upscale and polish.
  private func generateImageDirect(
    finalPrompt: String,
    state: ChatState,
    seed: Int?
  ) async -> Result<RenderResult, RenderError> {
    let outputFilename = "telegram-\(Int(Date().timeIntervalSince1970))-\(UInt32.random(in: 0...999999)).png"
    let outputPath = (config.outputDirectory as NSString).appendingPathComponent(outputFilename)

    do {
      let dims = state.aspectDimensions()
      var body: [String: Any] = [
        "prompt": finalPrompt,
        "outputPath": outputPath,
        "width": dims.width,
        "height": dims.height
      ]

      if let cfg = state.cfgOverride {
        body["guidance"] = cfg
      }
      if let seedVal = seed ?? state.seedLock {
        body["seed"] = seedVal
      }

      let jsonData = try JSONSerialization.data(withJSONObject: body)
      let (status, responseData) = try await warmServer.post("/v1/generate", body: jsonData)

      guard status == 200 else {
        let errorMsg = parseErrorMessage(responseData) ?? "WarmServer returned status \(status)"
        return .failure(.generateFailed(errorMsg))
      }

      guard let responseJSON = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any],
            let actualOutputPath = responseJSON["outputPath"] as? String else {
        return .failure(.generateFailed("Invalid response from WarmServer"))
      }

      let durationMs = responseJSON["durationMs"] as? Int ?? 0
      let responseSeed = responseJSON["seed"] as? Int

      // Read generated image
      let imageURL = URL(fileURLWithPath: actualOutputPath)
      var imageData = try Data(contentsOf: imageURL)
      var wasUpscaled = false
      var finalOutputPath = actualOutputPath
      var totalDuration = durationMs

      // Polish pass (two-pass: re-render with higher steps at lower strength)
      if state.polishEnabled {
        let polishResult = await polishImage(imagePath: actualOutputPath, prompt: finalPrompt, state: state)
        if case .success(let polished) = polishResult {
          imageData = polished.data
          totalDuration += polished.durationMs
          finalOutputPath = polished.outputPath
        }
      }

      // Upscale pass
      if state.upscaleEnabled {
        let upscaleResult = await upscaleImage(imagePath: finalOutputPath, state: state)
        if case .success(let upscaled) = upscaleResult {
          imageData = upscaled.data
          totalDuration += upscaled.durationMs
          finalOutputPath = upscaled.outputPath
          wasUpscaled = true
        }
      }

      // Double upscale for 4K (render -> 2K -> 4K)
      if state.resolutionTarget == "4k" && wasUpscaled {
        let secondUpscaleResult = await upscaleImage(imagePath: finalOutputPath, state: state)
        if case .success(let upscaled) = secondUpscaleResult {
          imageData = upscaled.data
          totalDuration += upscaled.durationMs
          finalOutputPath = upscaled.outputPath
        }
      }

      return .success(RenderResult(
        imageData: imageData,
        outputPath: finalOutputPath,
        filename: outputFilename,
        durationMs: totalDuration,
        seed: responseSeed,
        wasUpscaled: wasUpscaled
      ))

    } catch let error as WarmServerClientError {
      return .failure(.warmServerDown(error.localizedDescription))
    } catch {
      return .failure(.generateFailed(error.localizedDescription))
    }
  }

  // MARK: - Upscale

  private struct UpscaleOutput {
    let data: Data
    let outputPath: String
    let durationMs: Int
  }

  private func upscaleImage(imagePath: String, state: ChatState) async -> Result<UpscaleOutput, RenderError> {
    let upscaledPath = imagePath.replacingOccurrences(of: ".png", with: "-upscaled.png")

    do {
      let body: [String: Any] = [
        "image_path": imagePath,
        "output_path": upscaledPath,
        "target_resolution": 1024  // Safe default
      ]
      let jsonData = try JSONSerialization.data(withJSONObject: body)
      let (status, responseData) = try await warmServer.post("/v1/upscale", body: jsonData)

      guard status == 200 else {
        let errorMsg = parseErrorMessage(responseData) ?? "Upscale returned status \(status)"
        return .failure(.upscaleFailed(errorMsg))
      }

      guard let responseJSON = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any],
            let outputPath = responseJSON["outputPath"] as? String ?? responseJSON["output_path"] as? String else {
        return .failure(.upscaleFailed("Invalid upscale response"))
      }

      let durationMs = responseJSON["durationMs"] as? Int ?? responseJSON["duration_ms"] as? Int ?? 0
      let data = try Data(contentsOf: URL(fileURLWithPath: outputPath))

      return .success(UpscaleOutput(data: data, outputPath: outputPath, durationMs: durationMs))
    } catch {
      return .failure(.upscaleFailed(error.localizedDescription))
    }
  }

  // MARK: - Polish (Two-Pass)

  private struct PolishOutput {
    let data: Data
    let outputPath: String
    let durationMs: Int
  }

  private func polishImage(imagePath: String, prompt: String, state: ChatState) async -> Result<PolishOutput, RenderError> {
    let polishedPath = imagePath.replacingOccurrences(of: ".png", with: "-polished.png")

    do {
      let dims = state.aspectDimensions()
      var body: [String: Any] = [
        "prompt": prompt,
        "outputPath": polishedPath,
        "width": dims.width,
        "height": dims.height,
        "image_path": imagePath,
        "strength": 0.35,
        "steps": 30
      ]
      if let cfg = state.cfgOverride {
        body["guidance"] = cfg
      }

      let jsonData = try JSONSerialization.data(withJSONObject: body)
      let (status, responseData) = try await warmServer.post("/v1/generate", body: jsonData)

      guard status == 200 else {
        let errorMsg = parseErrorMessage(responseData) ?? "Polish returned status \(status)"
        return .failure(.generateFailed(errorMsg))
      }

      guard let responseJSON = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any],
            let outputPath = responseJSON["outputPath"] as? String else {
        return .failure(.generateFailed("Invalid polish response"))
      }

      let durationMs = responseJSON["durationMs"] as? Int ?? 0
      let data = try Data(contentsOf: URL(fileURLWithPath: outputPath))

      return .success(PolishOutput(data: data, outputPath: outputPath, durationMs: durationMs))
    } catch {
      return .failure(.generateFailed(error.localizedDescription))
    }
  }

  // MARK: - Post-Processing

  private func applyPostProcessing(imageData: Data, state: ChatState, wasUpscaled: Bool) -> Data {
    guard state.hasPostProcessing || wasUpscaled else { return imageData }

    return PostProcessor.applyPipeline(
      imageData: imageData,
      saturation: state.saturation,
      colorTemp: state.colorTemp,
      filmLookId: state.filmLook,
      sharpenAfterUpscale: wasUpscaled
    )
  }

  // MARK: - Send Helpers

  private func sendImage(chatId: Int, imageData: Data, filename: String, caption: String) async {
    if imageData.count > 8_000_000 {
      let _ = try? await bot.sendDocument(
        chatId: chatId,
        fileData: imageData,
        filename: filename,
        mimeType: "image/png",
        caption: caption
      )
    } else {
      let _ = try? await bot.sendPhoto(
        chatId: chatId,
        imageData: imageData,
        filename: filename,
        caption: caption
      )
    }
  }

  private func sendError(chatId: Int, error: RenderError) async {
    switch error {
    case .warmServerDown:
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "WarmServer not available \u{2014} start <code>ComfyBox serve</code> first.")
    case .generateFailed(let msg):
      let _ = try? await bot.sendMessage(chatId: chatId, text: "Render failed: \(msg)")
    case .upscaleFailed(let msg):
      let _ = try? await bot.sendMessage(chatId: chatId, text: "Upscale failed: \(msg)")
    }
    logger.error("Render error: \(error)")
  }

  private func copyToGallery(outputPath: String, filename: String) {
    if let galleryDir = config.galleryDirectory {
      let galleryPath = (galleryDir as NSString).appendingPathComponent(filename)
      try? FileManager.default.copyItem(atPath: outputPath, toPath: galleryPath)
    }
  }

  // MARK: - Status Text Builder

  private func buildStatusText(parsed: ParsedContext, state: ChatState, index: Int?, total: Int?) -> String {
    let modeEmoji = ContentModeManager.emoji(for: parsed.effectiveMode)
    let enhanceEmoji = state.enhanceEnabled ? "\u{2728}" : ""
    let charTag = parsed.character != nil ? " [\(parsed.character!)]" : ""
    var extras: [String] = []
    if state.upscaleEnabled { extras.append("upscale") }
    if state.polishEnabled { extras.append("polish") }
    if let look = state.filmLook { extras.append(look) }
    let extrasStr = extras.isEmpty ? "" : " [\(extras.joined(separator: ", "))]"
    let countStr: String
    if let i = index, let t = total {
      countStr = " \(i)/\(t)"
    } else {
      countStr = ""
    }
    return "\(modeEmoji)\(enhanceEmoji) Rendering\(charTag)\(extrasStr)\(countStr)..."
  }

  // MARK: - Caption Builder

  private func buildCaption(
    prompt: String,
    character: String?,
    mode: ContentModeManager.Mode,
    durationMs: Int,
    enhanced: Bool,
    seed: Int?,
    state: ChatState
  ) -> String {
    var parts: [String] = []

    let modeEmoji = ContentModeManager.emoji(for: mode)
    if let char = character {
      parts.append("\(modeEmoji) [\(char)]")
    } else {
      parts.append(modeEmoji)
    }

    let truncatedPrompt = prompt.count > 180 ? String(prompt.prefix(177)) + "..." : prompt
    parts.append(truncatedPrompt)

    if durationMs > 0 {
      let seconds = Double(durationMs) / 1000.0
      parts.append(String(format: "(%.1fs)", seconds))
    }

    if enhanced { parts.append("\u{2728}") }
    if let s = seed { parts.append("seed:\(s)") }

    // Active settings indicators
    var indicators: [String] = []
    if state.upscaleEnabled { indicators.append("up") }
    if state.polishEnabled { indicators.append("pol") }
    if state.saturation != nil { indicators.append("sat") }
    if state.colorTemp != nil { indicators.append("temp") }
    if state.filmLook != nil { indicators.append("film") }
    if !indicators.isEmpty {
      parts.append("[\(indicators.joined(separator: "+"))]")
    }

    return parts.joined(separator: " ")
  }

  // MARK: - Help & Status

  private func sendHelp(chatId: Int) async {
    let mode = contentModeManager.current
    let modeEmoji = ContentModeManager.emoji(for: mode)
    let modeName = ContentModeManager.displayName(for: mode)
    let state = sessions.getState(chatId: chatId)
    let enhanceState = state.enhanceEnabled ? "ON" : "OFF"
    let charList = characterLoader.allNames().map { $0.capitalized }.joined(separator: ", ")
    let dims = state.aspectDimensions()

    let helpText = """
    <b>ComfyBox Image Bot</b>

    Send any text to generate an image.

    <b>Content Modes:</b>
    /neutral or /apple \u{2014} SFW mode
    /banana \u{2014} Suggestive mode
    /avocado \u{2014} Explicit mode

    <b>Generation:</b>
    /batch &lt;N&gt; &lt;prompt&gt; \u{2014} Generate N images (2-8)
    /vary [N] &lt;prompt&gt; \u{2014} N prompt variations (default 3)
    /seq &lt;N&gt; &lt;story&gt; \u{2014} N sequential story frames

    <b>Settings:</b>
    /enhance [on|off] \u{2014} Prompt optimization
    /aspect &lt;mode&gt; \u{2014} square/portrait/landscape/wide/tall
    /cfg &lt;value|off&gt; \u{2014} Guidance scale override
    /seed &lt;N|random&gt; \u{2014} Lock or randomize seed
    /upscale [on|off] \u{2014} SeedVR 2x upscale
    /polish [on|off] \u{2014} Two-pass refinement
    /2k [on|off] \u{2014} Upscale to 2K
    /4k [on|off] \u{2014} Double upscale to 4K

    <b>Post-Processing:</b>
    /saturation &lt;0-2|off&gt; \u{2014} Saturation adjust
    /temp &lt;2000-10000|off&gt; \u{2014} Color temperature
    /film &lt;look|off&gt; \u{2014} Film look preset
    /look \u{2014} List available film looks
    /reset \u{2014} Clear all post-process settings

    <b>Info:</b>
    /help \u{2014} Show this help
    /status \u{2014} WarmServer status

    <b>Current Settings:</b>
    \(modeEmoji) Mode: <b>\(modeName)</b>
    \u{2728} Enhance: <b>\(enhanceState)</b>
    \u{1F4D0} Aspect: <b>\(state.aspectMode)</b> (\(dims.width)x\(dims.height))
    \u{1F464} Characters: \(charList.isEmpty ? "none loaded" : charList)
    \(state.settingsSummary().isEmpty ? "" : "\u{2699}\u{FE0F} \(state.settingsSummary())")

    <b>Tips:</b>
    \u{2022} Character names (\(charList)) are detected automatically
    \u{2022} Inline mode override: include /avocado in your prompt
    \u{2022} Be descriptive \u{2014} more detail = better results
    """
    let _ = try? await bot.sendMessage(chatId: chatId, text: helpText)
  }

  private func sendStatus(chatId: Int) async {
    let mode = contentModeManager.current
    let modeEmoji = ContentModeManager.emoji(for: mode)
    let state = sessions.getState(chatId: chatId)

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
      let enhanceState = state.enhanceEnabled ? "ON" : "OFF"
      let dims = state.aspectDimensions()

      let statusText = """
      <b>ComfyBox Status</b>

      <b>WarmServer:</b> \(serverStatus)
      <b>Model:</b> <code>\(model)</code>
      <b>Queue:</b> \(queueLength) pending
      <b>Total renders:</b> \(totalGens)
      <b>Server uptime:</b> \(formatDuration(uptime))

      \(modeEmoji) <b>Mode:</b> \(ContentModeManager.displayName(for: mode))
      \u{2728} <b>Enhance:</b> \(enhanceState)
      \u{1F4D0} <b>Aspect:</b> \(state.aspectMode) (\(dims.width)x\(dims.height))
      \u{1F464} <b>Characters:</b> \(charCount) loaded
      \(state.hasPostProcessing ? "\u{1F3A8} <b>Post-process:</b> \(state.settingsSummary())" : "")

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

  // MARK: - Sequence Helpers

  /// Parse numbered scenes from LLM output (e.g., "1. scene one\n2. scene two")
  private func parseNumberedScenes(_ text: String, expectedCount: Int) -> [String] {
    let lines = text.components(separatedBy: .newlines)
    var scenes: [String] = []

    var currentScene = ""
    for line in lines {
      let trimmed = line.trimmingCharacters(in: .whitespaces)
      // Detect numbered lines: "1.", "1)", "Scene 1:", etc.
      if let _ = trimmed.range(of: #"^\d+[\.\)\:]"#, options: .regularExpression) {
        if !currentScene.isEmpty {
          scenes.append(currentScene.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        // Strip the number prefix
        let stripped = trimmed.replacingOccurrences(of: #"^\d+[\.\)\:]\s*"#, with: "", options: .regularExpression)
        currentScene = stripped
      } else if !trimmed.isEmpty {
        currentScene += " " + trimmed
      }
    }
    if !currentScene.isEmpty {
      scenes.append(currentScene.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    // If parsing yielded wrong count, fall back to sentence splitting
    if scenes.isEmpty || scenes.count < 2 {
      return splitStoryIntoFrames(text, count: expectedCount)
    }

    return scenes
  }

  /// Fallback: split story text into N roughly equal frames by sentences.
  private func splitStoryIntoFrames(_ story: String, count: Int) -> [String] {
    // Split on common delimiters: periods, semicolons, em dashes, " -- "
    let delimiters = CharacterSet(charactersIn: ".;")
    var sentences = story.components(separatedBy: delimiters)
      .map { $0.trimmingCharacters(in: .whitespaces) }
      .filter { !$0.isEmpty }

    // Also split on " -- " for story-style separators
    if sentences.count < count {
      sentences = story.components(separatedBy: " -- ")
        .map { $0.trimmingCharacters(in: .whitespaces) }
        .filter { !$0.isEmpty }
    }

    if sentences.count >= count {
      // Group sentences into N buckets
      var frames: [String] = []
      let perFrame = max(1, sentences.count / count)
      for i in 0..<count {
        let start = i * perFrame
        let end = (i == count - 1) ? sentences.count : min(start + perFrame, sentences.count)
        if start < sentences.count {
          frames.append(sentences[start..<end].joined(separator: ". "))
        }
      }
      return frames
    }

    // Not enough sentences — duplicate the story for each frame with a frame number
    return (0..<count).map { "Frame \($0 + 1) of \(count): \(story)" }
  }

  // MARK: - Helpers

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

enum RenderError: Error, LocalizedError {
  case warmServerDown(String)
  case generateFailed(String)
  case upscaleFailed(String)

  var errorDescription: String? {
    switch self {
    case .warmServerDown(let msg): return msg
    case .generateFailed(let msg): return msg
    case .upscaleFailed(let msg): return msg
    }
  }
}
