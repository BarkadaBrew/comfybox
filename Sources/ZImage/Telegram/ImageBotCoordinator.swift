// ImageBotCoordinator.swift — Orchestrates Telegram bot: parse -> render -> send.
//
// Phase 1: Handles text prompts via WarmServer, /help, and /status.
// Phase 2: Content modes, character injection, prompt optimization.
// Phase 3: Full command set — batch, vary, sequence, upscale, aspect, cfg,
//          seed, polish, post-processing (saturation, temp, film), reset.
// Phase 4: Inline keyboards on photos, callback query handling, reply-to-image
//          (rerender/HQ/img2img/upscale), discuss mode, queue, /look alias, /imagine.
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
    // Phase 4: Handle callback queries (inline keyboard presses)
    if let cbQuery = update.callbackQuery {
      await handleCallbackQuery(cbQuery)
      return
    }

    if let message = update.message {
      // Phase 4: Check for reply-to-image
      if let replyTo = message.replyToMessage,
         replyTo.photo != nil || replyTo.caption != nil,
         let text = message.text {
        await handleReplyToImage(message: message, replyToMessage: replyTo, text: text)
        return
      }

      if let text = message.text {
        await handleTextMessage(message: message, text: text)
        return
      }
    }
  }

  // MARK: - Callback Query Handling (Phase 4)

  private func handleCallbackQuery(_ query: TelegramCallbackQuery) async {
    guard let chatId = query.chatId, let data = query.data else {
      try? await bot.answerCallbackQuery(id: query.id, text: "Invalid callback")
      return
    }

    // Parse callback data: "action:chatId:msgId"
    let parts = data.split(separator: ":")
    guard parts.count >= 3,
          let origMsgId = Int(parts[2]) else {
      try? await bot.answerCallbackQuery(id: query.id, text: "Invalid callback data")
      return
    }

    let action = String(parts[0])
    let state = sessions.getState(chatId: chatId)

    // Look up the render context for the original message
    guard let renderCtx = state.getRenderContext(messageId: origMsgId) else {
      try? await bot.answerCallbackQuery(id: query.id, text: "Render context expired")
      return
    }

    switch action {
    case "rerender":
      try? await bot.answerCallbackQuery(id: query.id, text: "Re-rendering with new seed...")
      await handleCallbackRerender(chatId: chatId, renderCtx: renderCtx)

    case "hq":
      try? await bot.answerCallbackQuery(id: query.id, text: "Rendering HQ (polish pass)...")
      await handleCallbackHQ(chatId: chatId, renderCtx: renderCtx)

    case "video":
      try? await bot.answerCallbackQuery(id: query.id, text: nil)
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "Video generation routes through @BaristaBree_Bot. Send her the image with a motion description.")

    default:
      try? await bot.answerCallbackQuery(id: query.id, text: "Unknown action")
    }
  }

  private func handleCallbackRerender(chatId: Int, renderCtx: RenderContext) async {
    let state = sessions.getState(chatId: chatId)

    let result = await generateImage(
      prompt: renderCtx.prompt,
      character: renderCtx.character,
      characterDescription: renderCtx.character.flatMap { characterLoader.description(for: $0, mode: contentModeManager.current) },
      effectiveMode: contentModeManager.current,
      state: state,
      seed: nil  // New random seed
    )

    switch result {
    case .success(let render):
      let finalData = applyPostProcessing(imageData: render.imageData, state: state, wasUpscaled: render.wasUpscaled)
      let caption = buildCaption(
        prompt: renderCtx.prompt,
        character: renderCtx.character,
        mode: contentModeManager.current,
        durationMs: render.durationMs,
        enhanced: renderCtx.enhanceEnabled,
        seed: render.seed,
        state: state
      )
      let sentResult = await sendImageWithKeyboard(chatId: chatId, imageData: finalData, filename: render.filename, caption: caption)

      // Store render context for the new message
      if let newMsgId = sentResult?.messageId {
        sessions.updateState(chatId: chatId) {
          $0.lastPrompt = renderCtx.prompt
          $0.lastImagePath = render.outputPath
          $0.lastSeed = render.seed
          $0.storeRenderContext(messageId: newMsgId, context: RenderContext(
            prompt: renderCtx.prompt,
            imagePath: render.outputPath,
            seed: render.seed,
            character: renderCtx.character,
            contentMode: renderCtx.contentMode,
            enhanceEnabled: renderCtx.enhanceEnabled
          ))
        }
      }
      copyToGallery(outputPath: render.outputPath, filename: render.filename)

    case .failure(let error):
      await sendError(chatId: chatId, error: error)
    }
  }

  private func handleCallbackHQ(chatId: Int, renderCtx: RenderContext) async {
    var state = sessions.getState(chatId: chatId)
    // Force polish on for this render
    let originalPolish = state.polishEnabled
    state.polishEnabled = true

    let result = await generateImage(
      prompt: renderCtx.prompt,
      character: renderCtx.character,
      characterDescription: renderCtx.character.flatMap { characterLoader.description(for: $0, mode: contentModeManager.current) },
      effectiveMode: contentModeManager.current,
      state: state,
      seed: renderCtx.seed  // Same seed for consistency
    )

    // Restore polish state
    sessions.updateState(chatId: chatId) { $0.polishEnabled = originalPolish }

    switch result {
    case .success(let render):
      let finalData = applyPostProcessing(imageData: render.imageData, state: state, wasUpscaled: render.wasUpscaled)
      let caption = "\u{2728} HQ \u{2014} " + buildCaption(
        prompt: renderCtx.prompt,
        character: renderCtx.character,
        mode: contentModeManager.current,
        durationMs: render.durationMs,
        enhanced: renderCtx.enhanceEnabled,
        seed: render.seed,
        state: state
      )
      let sentResult = await sendImageWithKeyboard(chatId: chatId, imageData: finalData, filename: render.filename, caption: caption)

      if let newMsgId = sentResult?.messageId {
        sessions.updateState(chatId: chatId) {
          $0.lastPrompt = renderCtx.prompt
          $0.lastImagePath = render.outputPath
          $0.lastSeed = render.seed
          $0.storeRenderContext(messageId: newMsgId, context: RenderContext(
            prompt: renderCtx.prompt,
            imagePath: render.outputPath,
            seed: render.seed,
            character: renderCtx.character,
            contentMode: renderCtx.contentMode,
            enhanceEnabled: renderCtx.enhanceEnabled
          ))
        }
      }
      copyToGallery(outputPath: render.outputPath, filename: render.filename)

    case .failure(let error):
      await sendError(chatId: chatId, error: error)
    }
  }

  // MARK: - Reply-to-Image Handling (Phase 4)

  private func handleReplyToImage(message: TelegramMessage, replyToMessage: TelegramMessage, text: String) async {
    let chatId = message.chatId
    let replyMsgId = replyToMessage.messageId

    // Try to get the render context for the replied-to message
    let state = sessions.getState(chatId: chatId)
    let renderCtx = state.getRenderContext(messageId: replyMsgId)

    guard let intent = TelegramCommandParser.parseReplyIntent(text) else { return }

    switch intent {
    case .rerender:
      guard let ctx = renderCtx else {
        let _ = try? await bot.sendMessage(chatId: chatId, text: "No render context for that image. Send a new prompt.")
        return
      }
      await handleCallbackRerender(chatId: chatId, renderCtx: ctx)

    case .hq:
      guard let ctx = renderCtx else {
        let _ = try? await bot.sendMessage(chatId: chatId, text: "No render context for that image. Send a new prompt.")
        return
      }
      await handleCallbackHQ(chatId: chatId, renderCtx: ctx)

    case .upscaleReply:
      guard let ctx = renderCtx else {
        let _ = try? await bot.sendMessage(chatId: chatId, text: "No render context for that image. Send a new prompt.")
        return
      }
      await handleReplyUpscale(chatId: chatId, renderCtx: ctx)

    case .video(let motion):
      let motionText = motion != nil ? " with motion: \(motion!)" : ""
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "Video generation routes through @BaristaBree_Bot\(motionText). Send her the image with a motion description.")

    case .newPrompt(let newPrompt):
      // img2img: use original image with new prompt
      if let ctx = renderCtx {
        await handleImg2Img(chatId: chatId, newPrompt: newPrompt, renderCtx: ctx)
      } else {
        // No context — treat as a regular render
        await handleRender(chatId: chatId, prompt: newPrompt, messageId: message.messageId)
      }
    }
  }

  private func handleReplyUpscale(chatId: Int, renderCtx: RenderContext) async {
    let _ = try? await bot.sendMessage(chatId: chatId, text: "\u{1F50D} Upscaling...")

    let state = sessions.getState(chatId: chatId)
    let upscaleResult = await upscaleImage(imagePath: renderCtx.imagePath, state: state)

    switch upscaleResult {
    case .success(let upscaled):
      var finalData = upscaled.data
      // Sharpen after upscale
      finalData = PostProcessor.applyPipeline(
        imageData: finalData,
        saturation: state.saturation,
        colorTemp: state.colorTemp,
        filmLookId: state.filmLook,
        sharpenAfterUpscale: true
      )
      let filename = "telegram-upscaled-\(Int(Date().timeIntervalSince1970)).png"
      let caption = "\u{1F50D} Upscaled \u{2014} \(upscaled.durationMs)ms"
      let sentResult = await sendImageWithKeyboard(chatId: chatId, imageData: finalData, filename: filename, caption: caption)

      if let newMsgId = sentResult?.messageId {
        sessions.updateState(chatId: chatId) {
          $0.storeRenderContext(messageId: newMsgId, context: RenderContext(
            prompt: renderCtx.prompt,
            imagePath: upscaled.outputPath,
            seed: renderCtx.seed,
            character: renderCtx.character,
            contentMode: renderCtx.contentMode,
            enhanceEnabled: renderCtx.enhanceEnabled
          ))
        }
      }

    case .failure(let error):
      await sendError(chatId: chatId, error: error)
    }
  }

  private func handleImg2Img(chatId: Int, newPrompt: String, renderCtx: RenderContext) async {
    let state = sessions.getState(chatId: chatId)
    let parsed = parseAndResolve(prompt: newPrompt)

    let _ = try? await bot.sendMessage(chatId: chatId, text: "\u{1F3A8} Re-rendering with new prompt...")

    // Optimize the new prompt
    let finalPrompt: String
    if state.enhanceEnabled {
      let result = await promptOptimizer.optimize(
        prompt: parsed.prompt,
        character: parsed.character,
        characterDescription: parsed.characterDescription,
        contentMode: parsed.effectiveMode.rawValue
      )
      finalPrompt = result.prompt
    } else {
      if let desc = parsed.characterDescription {
        finalPrompt = "\(desc)\n\n\(parsed.prompt)"
      } else {
        finalPrompt = parsed.prompt
      }
    }

    // Generate with img2img using the original image
    let outputFilename = "telegram-\(Int(Date().timeIntervalSince1970))-\(UInt32.random(in: 0...999999)).png"
    let outputPath = (config.outputDirectory as NSString).appendingPathComponent(outputFilename)

    do {
      let dims = state.aspectDimensions()
      var body: [String: Any] = [
        "prompt": finalPrompt,
        "output_path": outputPath,
        "width": dims.width,
        "height": dims.height,
        "image_path": renderCtx.imagePath,
        "image_strength": 0.5
      ]
      if let cfg = state.cfgOverride { body["guidance"] = cfg }

      let jsonData = try JSONSerialization.data(withJSONObject: body)
      let (status, responseData) = try await warmServer.post("/v1/generate", body: jsonData)

      guard status == 200,
            let responseJSON = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any],
            let actualOutputPath = responseJSON["output_path"] as? String else {
        let errorMsg = parseErrorMessage(responseData) ?? "img2img returned status \(status)"
        let _ = try? await bot.sendMessage(chatId: chatId, text: "img2img failed: \(errorMsg)")
        return
      }

      let durationMs = responseJSON["duration_ms"] as? Int ?? 0
      // /v1/generate does not return a seed; no seed was sent with this request.
      let responseSeed: Int? = nil
      var imageData = try Data(contentsOf: URL(fileURLWithPath: actualOutputPath))

      // Apply post-processing
      imageData = applyPostProcessing(imageData: imageData, state: state, wasUpscaled: false)

      let caption = "\u{1F3A8} img2img \u{2014} " + buildCaption(
        prompt: newPrompt,
        character: parsed.character,
        mode: parsed.effectiveMode,
        durationMs: durationMs,
        enhanced: state.enhanceEnabled,
        seed: responseSeed,
        state: state
      )

      let sentResult = await sendImageWithKeyboard(chatId: chatId, imageData: imageData, filename: outputFilename, caption: caption)

      if let newMsgId = sentResult?.messageId {
        sessions.updateState(chatId: chatId) {
          $0.lastPrompt = newPrompt
          $0.lastImagePath = actualOutputPath
          $0.lastSeed = responseSeed
          $0.storeRenderContext(messageId: newMsgId, context: RenderContext(
            prompt: newPrompt,
            imagePath: actualOutputPath,
            seed: responseSeed,
            character: parsed.character,
            contentMode: parsed.effectiveMode.rawValue,
            enhanceEnabled: state.enhanceEnabled
          ))
        }
      }
      copyToGallery(outputPath: actualOutputPath, filename: outputFilename)

    } catch {
      let _ = try? await bot.sendMessage(chatId: chatId, text: "img2img failed: \(error.localizedDescription)")
    }
  }

  // MARK: - Text Message Handling

  private func handleTextMessage(message: TelegramMessage, text: String) async {
    let chatId = message.chatId

    // Check discuss mode
    let state = sessions.getState(chatId: chatId)
    let inDiscussMode = state.isInDiscussMode

    let command = TelegramCommandParser.parse(text, inDiscussMode: inDiscussMode)

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
      let current = sessions.getState(chatId: chatId)
      let newValue = on ?? !current.enhanceEnabled
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

    case .look(let id):
      await handleLook(chatId: chatId, lookId: id)

    // -- Phase 4: Discuss mode --
    case .chat:
      await handleEnterDiscussMode(chatId: chatId)

    case .endChat:
      await handleExitDiscussMode(chatId: chatId)

    case .shipCue:
      await handleShipCue(chatId: chatId)

    case .chatMessage(let text):
      await handleDiscussMessage(chatId: chatId, text: text)

    case .imagine(let description):
      await handleImagine(chatId: chatId, description: description)

    // -- Phase 4: Queue --
    case .queue(let subcommand):
      await handleQueue(chatId: chatId, subcommand: subcommand)

    // -- Deferred --
    case .video:
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "Video generation routes through @BaristaBree_Bot. Send her the image with a motion description.")
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

  // MARK: - Look (Phase 4: enhanced with ID alias)

  private func handleLook(chatId: Int, lookId: String?) async {
    guard let lookId = lookId else {
      // No arg — list all looks
      await sendLookList(chatId: chatId)
      return
    }
    // /look <id> acts as alias for /film <id>
    await handleFilm(chatId: chatId, lookId: lookId)
  }

  private func sendLookList(chatId: Int) async {
    let looks = PostProcessor.availableLooks()
    let state = sessions.getState(chatId: chatId)
    var lines: [String] = ["<b>Available Film Looks:</b>\n"]
    for look in looks {
      let active = state.filmLook?.lowercased() == look.id.lowercased() ? " \u{2705}" : ""
      lines.append("\u{2022} <code>\(look.id)</code> \u{2014} \(look.name)\(active)")
    }
    lines.append("\nUsage: <code>/look kodak-portra</code> or <code>/film off</code>")
    let _ = try? await bot.sendMessage(chatId: chatId, text: lines.joined(separator: "\n"))
  }

  // MARK: - Discuss Mode (Phase 4)

  private func handleEnterDiscussMode(chatId: Int) async {
    sessions.updateState(chatId: chatId) { $0.enterDiscussMode() }
    let _ = try? await bot.sendMessage(chatId: chatId,
      text: """
      \u{1F4AC} <b>Discuss mode</b> \u{2014} let's design a prompt together.

      Describe what you're imagining and I'll help refine it into a great image prompt. When you're happy, say <b>go</b>, <b>ship it</b>, or <b>render it</b> to generate.

      /end to exit discuss mode.
      """)
  }

  private func handleExitDiscussMode(chatId: Int) async {
    let state = sessions.getState(chatId: chatId)
    guard state.isInDiscussMode else {
      let _ = try? await bot.sendMessage(chatId: chatId, text: "Not in discuss mode.")
      return
    }
    sessions.updateState(chatId: chatId) { $0.exitDiscussMode() }
    let _ = try? await bot.sendMessage(chatId: chatId,
      text: "\u{1F44B} Exited discuss mode. Send a text prompt to generate directly.")
  }

  private func handleDiscussMessage(chatId: Int, text: String) async {
    // Add user message to history
    sessions.updateState(chatId: chatId) { $0.addDiscussMessage(role: "user", content: text) }

    let state = sessions.getState(chatId: chatId)
    let mode = contentModeManager.current

    // Build conversation for the LLM
    let systemPrompt = buildDiscussSystemPrompt(mode: mode)
    var messages: [[String: String]] = [["role": "system", "content": systemPrompt]]
    for entry in state.discussHistory {
      messages.append(["role": entry.role, "content": entry.content])
    }

    // Call LLM for collaborative conversation
    let response = await callDiscussLLM(messages: messages)

    if let response = response {
      sessions.updateState(chatId: chatId) {
        $0.addDiscussMessage(role: "assistant", content: response)
        // Try to extract the latest prompt proposal from the response
        if let extracted = extractPromptFromDiscussion(response) {
          $0.discussCurrentPrompt = extracted
        }
      }
      let _ = try? await bot.sendMessage(chatId: chatId, text: response)
    } else {
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "Couldn't reach the prompt design assistant. Try again or say <b>go</b> to render what we have.")
    }
  }

  private func handleShipCue(chatId: Int) async {
    let state = sessions.getState(chatId: chatId)

    // Use the current refined prompt, or fall back to the last user message
    let promptToRender: String
    if let current = state.discussCurrentPrompt {
      promptToRender = current
    } else if let lastUser = state.discussHistory.last(where: { $0.role == "user" }) {
      promptToRender = lastUser.content
    } else {
      let _ = try? await bot.sendMessage(chatId: chatId, text: "Nothing to render yet. Describe what you want first.")
      return
    }

    // Exit discuss mode and render
    sessions.updateState(chatId: chatId) { $0.exitDiscussMode() }
    let _ = try? await bot.sendMessage(chatId: chatId,
      text: "\u{1F680} Rendering...")

    await handleRender(chatId: chatId, prompt: promptToRender, messageId: 0)
  }

  // MARK: - Imagine (Phase 4: One-shot agent)

  private func handleImagine(chatId: Int, description: String) async {
    let mode = contentModeManager.current
    let state = sessions.getState(chatId: chatId)

    let _ = try? await bot.sendMessage(chatId: chatId,
      text: "\u{1F3A8} Designing prompt from: <i>\(description)</i>...")

    // Use the discuss LLM to design a complete prompt from the description
    let systemPrompt = """
    You are a creative director for AI image generation. The user gives you a brief scene description. You must:
    1. Design a complete, detailed image prompt in YOUR CONTEXT: / YOUR PHOTO: format.
    2. After the prompt, write a brief creative report (2-3 sentences) explaining your choices.

    Separate the prompt from the report with a line containing only "---".

    The prompt should be rich in visual detail, lighting, composition, and mood.
    Current content mode: \(mode.rawValue).
    """

    let messages: [[String: String]] = [
      ["role": "system", "content": systemPrompt],
      ["role": "user", "content": description]
    ]

    let response = await callDiscussLLM(messages: messages)

    // Parse out the prompt vs the report
    let promptToRender: String
    var report: String? = nil

    if let response = response {
      let parts = response.components(separatedBy: "\n---\n")
      if parts.count >= 2 {
        promptToRender = parts[0].trimmingCharacters(in: .whitespacesAndNewlines)
        report = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
      } else {
        // No separator — use the whole thing as the prompt
        promptToRender = response.trimmingCharacters(in: .whitespacesAndNewlines)
      }
    } else {
      // LLM unavailable — use description directly
      promptToRender = description
    }

    // Render it
    let parsed = parseAndResolve(prompt: promptToRender)

    let result = await generateImage(
      prompt: parsed.prompt,
      character: parsed.character,
      characterDescription: parsed.characterDescription,
      effectiveMode: parsed.effectiveMode,
      state: state,
      seed: nil
    )

    switch result {
    case .success(let render):
      let finalData = applyPostProcessing(imageData: render.imageData, state: state, wasUpscaled: render.wasUpscaled)

      var caption = buildCaption(
        prompt: description,
        character: parsed.character,
        mode: parsed.effectiveMode,
        durationMs: render.durationMs,
        enhanced: true,
        seed: render.seed,
        state: state
      )

      if let report = report {
        caption += "\n\n\(report)"
      }

      let sentResult = await sendImageWithKeyboard(chatId: chatId, imageData: finalData, filename: render.filename, caption: caption)

      if let newMsgId = sentResult?.messageId {
        sessions.updateState(chatId: chatId) {
          $0.lastPrompt = description
          $0.lastImagePath = render.outputPath
          $0.lastSeed = render.seed
          $0.storeRenderContext(messageId: newMsgId, context: RenderContext(
            prompt: description,
            imagePath: render.outputPath,
            seed: render.seed,
            character: parsed.character,
            contentMode: parsed.effectiveMode.rawValue,
            enhanceEnabled: state.enhanceEnabled
          ))
        }
      }
      copyToGallery(outputPath: render.outputPath, filename: render.filename)

    case .failure(let error):
      await sendError(chatId: chatId, error: error)
    }
  }

  // MARK: - Queue Management (Phase 4)

  private func handleQueue(chatId: Int, subcommand: String?) async {
    let sub = subcommand?.lowercased().trimmingCharacters(in: .whitespaces) ?? "status"

    switch sub {
    case "status", "":
      await sendQueueStatus(chatId: chatId)
    case "cancel":
      await cancelQueueJobs(chatId: chatId)
    case "list":
      await listQueueJobs(chatId: chatId)
    default:
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "Unknown queue command. Usage: <code>/queue</code>, <code>/queue cancel</code>, <code>/queue list</code>")
    }
  }

  private func sendQueueStatus(chatId: Int) async {
    do {
      let (status, data) = try await warmServer.get("/health")
      guard status == 200,
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        let _ = try? await bot.sendMessage(chatId: chatId, text: "Could not reach WarmServer.")
        return
      }

      let serverStatus = json["status"] as? String ?? "unknown"
      let queueLength = json["pending_count"] as? Int ?? 0
      let totalGens = json["render_count"] as? Int ?? 0
      let model = json["model"] as? String ?? "none"

      let statusEmoji = serverStatus == "ok" ? "\u{1F7E2}" : "\u{1F7E1}"
      let _ = try? await bot.sendMessage(chatId: chatId, text: """
      <b>Queue Status</b>

      \(statusEmoji) <b>Server:</b> \(serverStatus)
      \u{1F4E6} <b>Pending:</b> \(queueLength)
      \u{2705} <b>Completed:</b> \(totalGens)
      \u{1F9E0} <b>Model:</b> <code>\(model)</code>
      """)

    } catch {
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "WarmServer not available \u{2014} start <code>ComfyBox serve</code> first.")
    }
  }

  private func cancelQueueJobs(chatId: Int) async {
    do {
      // POST /queue (ComfyUI bridge endpoint) clears all pending jobs.
      let body = try JSONSerialization.data(withJSONObject: ["clear": true])
      let (status, data) = try await warmServer.post("/queue", body: body)
      if status == 200 || status == 204 {
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        if let cleared = json?["cleared_count"] as? Int {
          let _ = try? await bot.sendMessage(chatId: chatId, text: "\u{1F5D1} Queue cleared (\(cleared) pending job\(cleared == 1 ? "" : "s")).")
        } else {
          let _ = try? await bot.sendMessage(chatId: chatId, text: "\u{1F5D1} Queue cleared.")
        }
      } else {
        let _ = try? await bot.sendMessage(chatId: chatId, text: "Could not clear queue (status \(status)).")
      }
    } catch {
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "WarmServer not available \u{2014} start <code>ComfyBox serve</code> first.")
    }
  }

  private func listQueueJobs(chatId: Int) async {
    do {
      // GET /queue (ComfyUI bridge endpoint) returns queue_running/queue_pending
      // placeholder arrays plus a queue_status summary — no per-job prompts.
      let (status, data) = try await warmServer.get("/queue")
      guard status == 200,
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        let _ = try? await bot.sendMessage(chatId: chatId, text: "Could not fetch queue (status \(status)).")
        return
      }

      let running = json["queue_running"] as? [Any] ?? []
      let pending = json["queue_pending"] as? [Any] ?? []

      if running.isEmpty && pending.isEmpty {
        let _ = try? await bot.sendMessage(chatId: chatId, text: "\u{1F4E6} Queue is empty.")
        return
      }

      var lines: [String] = ["<b>Queue:</b>\n"]
      if !running.isEmpty {
        var renderLine = "\u{1F3A8} 1 job rendering"
        if let queueStatus = json["queue_status"] as? [String: Any],
           let progress = queueStatus["progress_percent"] as? Int {
          renderLine += " (\(progress)%)"
        }
        lines.append(renderLine)
      }
      if !pending.isEmpty {
        lines.append("\u{23F3} \(pending.count) pending")
      }

      let _ = try? await bot.sendMessage(chatId: chatId, text: lines.joined(separator: "\n"))

    } catch {
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "WarmServer not available \u{2014} start <code>ComfyBox serve</code> first.")
    }
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
    let _ = try? await bot.sendMessage(chatId: chatId, text: statusText)

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

      // Send with inline keyboard
      let sentResult = await sendImageWithKeyboard(chatId: chatId, imageData: finalData, filename: render.filename, caption: caption)

      // Update session and store render context
      sessions.updateState(chatId: chatId) {
        $0.lastPrompt = parsed.rawPrompt
        $0.lastImagePath = render.outputPath
        $0.lastSeed = render.seed
        if let msgId = sentResult?.messageId {
          $0.storeRenderContext(messageId: msgId, context: RenderContext(
            prompt: parsed.rawPrompt,
            imagePath: render.outputPath,
            seed: render.seed,
            character: parsed.character,
            contentMode: parsed.effectiveMode.rawValue,
            enhanceEnabled: state.enhanceEnabled
          ))
        }
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

    let _ = try? await bot.sendMessage(chatId: chatId, text: "\u{1F4E6} Batch: rendering \(count) images...")

    for i in 0..<count {
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F4E6} Rendering \(i + 1)/\(count)...")

      let result = await generateImage(
        prompt: parsed.prompt,
        character: parsed.character,
        characterDescription: parsed.characterDescription,
        effectiveMode: parsed.effectiveMode,
        state: state,
        seed: nil
      )

      switch result {
      case .success(let render):
        let finalData = applyPostProcessing(imageData: render.imageData, state: state, wasUpscaled: render.wasUpscaled)
        let caption = "\(i + 1)/\(count) \u{2014} seed:\(render.seed ?? 0)"
        let sentResult = await sendImageWithKeyboard(chatId: chatId, imageData: finalData, filename: render.filename, caption: caption)

        if let msgId = sentResult?.messageId {
          sessions.updateState(chatId: chatId) {
            $0.storeRenderContext(messageId: msgId, context: RenderContext(
              prompt: parsed.rawPrompt,
              imagePath: render.outputPath,
              seed: render.seed,
              character: parsed.character,
              contentMode: parsed.effectiveMode.rawValue,
              enhanceEnabled: state.enhanceEnabled
            ))
          }
        }
        copyToGallery(outputPath: render.outputPath, filename: render.filename)

      case .failure(let error):
        await sendError(chatId: chatId, error: error)
        return
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
        let sentResult = await sendImageWithKeyboard(chatId: chatId, imageData: finalData, filename: render.filename, caption: caption)

        if let msgId = sentResult?.messageId {
          sessions.updateState(chatId: chatId) {
            $0.storeRenderContext(messageId: msgId, context: RenderContext(
              prompt: parsed.rawPrompt,
              imagePath: render.outputPath,
              seed: render.seed,
              character: parsed.character,
              contentMode: parsed.effectiveMode.rawValue,
              enhanceEnabled: state.enhanceEnabled
            ))
          }
        }
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

      framePrompts = parseNumberedScenes(breakdownResult.prompt, expectedCount: count)
    } else {
      framePrompts = splitStoryIntoFrames(story, count: count)
    }

    let actualCount = min(framePrompts.count, count)

    for i in 0..<actualCount {
      let _ = try? await bot.sendMessage(chatId: chatId,
        text: "\u{1F3AC} Frame \(i + 1)/\(actualCount)...")

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
        let sentResult = await sendImageWithKeyboard(chatId: chatId, imageData: finalData, filename: render.filename, caption: caption)

        if let msgId = sentResult?.messageId {
          sessions.updateState(chatId: chatId) {
            $0.storeRenderContext(messageId: msgId, context: RenderContext(
              prompt: framePrompts[i],
              imagePath: render.outputPath,
              seed: render.seed,
              character: parsed.character,
              contentMode: parsed.effectiveMode.rawValue,
              enhanceEnabled: state.enhanceEnabled
            ))
          }
        }
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
        "output_path": outputPath,
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
            let actualOutputPath = responseJSON["output_path"] as? String else {
        return .failure(.generateFailed("Invalid response from WarmServer"))
      }

      let durationMs = responseJSON["duration_ms"] as? Int ?? 0
      // /v1/generate does not return a seed; echo the seed we requested (if any).
      let responseSeed = seed ?? state.seedLock

      var imageData = try Data(contentsOf: URL(fileURLWithPath: actualOutputPath))
      var wasUpscaled = false
      var finalOutputPath = actualOutputPath
      var totalDuration = durationMs

      // Polish pass
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

      // Double upscale for 4K
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
        "target_resolution": 1024
      ]
      let jsonData = try JSONSerialization.data(withJSONObject: body)
      let (status, responseData) = try await warmServer.post("/v1/upscale", body: jsonData)

      guard status == 200 else {
        let errorMsg = parseErrorMessage(responseData) ?? "Upscale returned status \(status)"
        return .failure(.upscaleFailed(errorMsg))
      }

      guard let responseJSON = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any],
            let outputPath = responseJSON["output_path"] as? String else {
        return .failure(.upscaleFailed("Invalid upscale response"))
      }

      let durationMs = responseJSON["duration_ms"] as? Int ?? 0
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
        "output_path": polishedPath,
        "width": dims.width,
        "height": dims.height,
        "image_path": imagePath,
        "image_strength": 0.35,
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
            let outputPath = responseJSON["output_path"] as? String else {
        return .failure(.generateFailed("Invalid polish response"))
      }

      let durationMs = responseJSON["duration_ms"] as? Int ?? 0
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

  // MARK: - Inline Keyboard (Phase 4)

  /// Build the standard inline keyboard for delivered images.
  private func buildImageKeyboard(chatId: Int, messageId: Int) -> InlineKeyboard {
    return InlineKeyboard(rows: [
      [
        InlineButton(text: "Rerender \u{1F504}", callbackData: "rerender:\(chatId):\(messageId)"),
        InlineButton(text: "HQ \u{2728}", callbackData: "hq:\(chatId):\(messageId)"),
        InlineButton(text: "Video \u{1F3AC}", callbackData: "video:\(chatId):\(messageId)")
      ]
    ])
  }

  // MARK: - Send Helpers

  /// Send an image with the standard inline keyboard attached.
  /// Returns the SendResult so callers can track the sent message ID.
  private func sendImageWithKeyboard(chatId: Int, imageData: Data, filename: String, caption: String) async -> SendResult? {
    if imageData.count > 8_000_000 {
      // Documents: Telegram allows up to 50MB
      // We still want to attach the keyboard to documents too
      let result = try? await bot.sendDocument(
        chatId: chatId,
        fileData: imageData,
        filename: filename,
        mimeType: "image/png",
        caption: caption
      )
      // Note: We can't predict messageId for doc sends to attach keyboard retroactively,
      // but sendDocument already supports replyMarkup if we pass it.
      // Let's re-send with keyboard.
      // Actually, we already support it via the replyMarkup parameter.
      // But we need the messageId first for the callback data.
      // For documents, we'll skip the keyboard (edge case: >8MB images are rare).
      return result
    } else {
      // For photos under 8MB, first send without keyboard to get messageId,
      // then we need the messageId for the callback data.
      // Alternative: use a placeholder messageId of 0 and parse chatId from callback.
      // Better approach: send photo, get messageId, then edit to add keyboard.
      let result = try? await bot.sendPhoto(
        chatId: chatId,
        imageData: imageData,
        filename: filename,
        caption: caption
      )

      // Now add the inline keyboard using the actual message ID
      if let msgId = result?.messageId {
        let keyboard = buildImageKeyboard(chatId: chatId, messageId: msgId)
        try? await bot.editMessageReplyMarkup(chatId: chatId, messageId: msgId, markup: keyboard)
      }

      return result
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

  // MARK: - Discuss Mode LLM (Phase 4)

  /// Call the local LLM (Ollama/LM Studio) for discuss mode conversation.
  private func callDiscussLLM(messages: [[String: String]]) async -> String? {
    let endpoints = [
      config.optimizer.ollamaBaseURL,
      config.optimizer.lmStudioBaseURL
    ].compactMap { $0 }

    for baseURL in endpoints {
      let payload: [String: Any] = [
        "model": config.optimizer.model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": false
      ]

      guard let payloadData = try? JSONSerialization.data(withJSONObject: payload),
            let url = URL(string: "\(baseURL)/v1/chat/completions") else {
        continue
      }

      var request = URLRequest(url: url)
      request.httpMethod = "POST"
      request.setValue("application/json", forHTTPHeaderField: "Content-Type")
      request.httpBody = payloadData
      request.timeoutInterval = 30

      do {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
          continue
        }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let first = choices.first,
              let message = first["message"] as? [String: Any],
              let content = message["content"] as? String else {
          continue
        }
        // Strip think tags from Qwen3
        return PromptOptimizer.cleanLLMOutput(content)
      } catch {
        logger.debug("Discuss LLM call to \(baseURL) failed: \(error.localizedDescription)")
        continue
      }
    }

    return nil
  }

  /// Build the system prompt for discuss mode conversations.
  private func buildDiscussSystemPrompt(mode: ContentModeManager.Mode) -> String {
    let modeDesc: String
    switch mode {
    case .neutral:
      modeDesc = "safe-for-work content. No nudity or suggestive content."
    case .banana:
      modeDesc = "suggestive/sensual content. Lingerie, partial nudity, intimate framing OK."
    case .avocado:
      modeDesc = "explicit adult content. Full nudity, graphic descriptions OK."
    }

    return """
    You are a creative prompt design assistant for AI image generation (Z-Image Turbo, a Qwen3-4B text encoder model). You help the user iteratively design and refine image prompts.

    Your role:
    - Listen to what the user wants to create
    - Suggest improvements: better composition, lighting, mood, camera angles
    - Propose a concrete prompt using YOUR CONTEXT: / YOUR PHOTO: format
    - When refining, explain what you changed and why
    - Keep suggestions concise (2-4 sentences of discussion + the prompt proposal)

    Current content mode: \(mode.rawValue) — \(modeDesc)

    Available characters (mention by name if relevant): Kira, Bree, Todd.

    When proposing a prompt, format it clearly so the user can review it. Do NOT render anything — just propose and discuss. The user will say "go" or "ship it" when they're ready to render.
    """
  }

  /// Try to extract the latest prompt proposal from a discuss mode response.
  private func extractPromptFromDiscussion(_ text: String) -> String? {
    // Look for YOUR CONTEXT: / YOUR PHOTO: blocks
    if let contextRange = text.range(of: "YOUR CONTEXT:", options: .caseInsensitive),
       text.range(of: "YOUR PHOTO:", options: .caseInsensitive) != nil {
      // Extract everything from YOUR CONTEXT: to the end (or next section)
      let prompt = String(text[contextRange.lowerBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
      if prompt.count > 30 { return prompt }
    }
    return nil
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
    /imagine &lt;desc&gt; \u{2014} One-shot: agent designs + renders

    <b>Interactive:</b>
    /chat or /discuss \u{2014} Enter prompt design mode
    /end or /exit \u{2014} Leave prompt design mode
    /queue \u{2014} Queue status / /queue cancel / /queue list
    Reply to any photo \u{2014} rerender, hq, upscale, or new prompt

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
    /look [id] \u{2014} List or apply film looks
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
    \u{2022} Reply to a photo with text for img2img
    \u{2022} Tap buttons under photos to rerender or HQ
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
      let uptime = json["uptime_seconds"] as? Int ?? 0
      let queueLength = json["pending_count"] as? Int ?? 0
      let totalGens = json["render_count"] as? Int ?? 0

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

  private func parseNumberedScenes(_ text: String, expectedCount: Int) -> [String] {
    let lines = text.components(separatedBy: .newlines)
    var scenes: [String] = []

    var currentScene = ""
    for line in lines {
      let trimmed = line.trimmingCharacters(in: .whitespaces)
      if let _ = trimmed.range(of: #"^\d+[\.\)\:]"#, options: .regularExpression) {
        if !currentScene.isEmpty {
          scenes.append(currentScene.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        let stripped = trimmed.replacingOccurrences(of: #"^\d+[\.\)\:]\s*"#, with: "", options: .regularExpression)
        currentScene = stripped
      } else if !trimmed.isEmpty {
        currentScene += " " + trimmed
      }
    }
    if !currentScene.isEmpty {
      scenes.append(currentScene.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    if scenes.isEmpty || scenes.count < 2 {
      return splitStoryIntoFrames(text, count: expectedCount)
    }

    return scenes
  }

  private func splitStoryIntoFrames(_ story: String, count: Int) -> [String] {
    let delimiters = CharacterSet(charactersIn: ".;")
    var sentences = story.components(separatedBy: delimiters)
      .map { $0.trimmingCharacters(in: .whitespaces) }
      .filter { !$0.isEmpty }

    if sentences.count < count {
      sentences = story.components(separatedBy: " -- ")
        .map { $0.trimmingCharacters(in: .whitespaces) }
        .filter { !$0.isEmpty }
    }

    if sentences.count >= count {
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
