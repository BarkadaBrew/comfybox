# ComfyBox Telegram Bot -- Technical Documentation

Developer reference for the Telegram image generation surface in `Sources/ZImage/Telegram/`.

## Architecture

The Telegram bot is a WarmServer client -- not a standalone image engine. It translates Telegram messages into WarmServer API calls and delivers results back to the user.

```
Telegram Cloud <--HTTPS (long poll)--> TelegramBot (URLSession)
                                            |
                                       TelegramCommandParser
                                            |
                                       ImageBotCoordinator
                                       /    |    \       \
                            PromptOptimizer  |  PostProcessor  SessionState
                             (Ollama/LMS)    |   (CIFilter)
                                        WarmServerClient
                                             |
                                        WarmServer (:7862)
                                             |
                                      Z-Image / MLX engine
```

Key design decisions:

- **Zero daemon dependency.** The bot does not require the Bree daemon, event system, or any Node.js infrastructure. It is a pure Swift process that talks to the Telegram API and the WarmServer.
- **URLSession only.** No external networking dependencies. All HTTP calls (Telegram API, Ollama, LM Studio, WarmServer) use Foundation's `URLSession`.
- **WarmServer as render backend.** All image generation goes through `WarmServerClient` -- the bot never loads models or touches MLX directly.
- **Per-chat state.** All render settings (aspect, seed, post-processing, etc.) are tracked per Telegram chat ID in memory. Content mode is the only global setting and is persisted to disk.

## Module Map

### `TelegramBot.swift`

Zero-dependency Telegram Bot API client. Handles polling, outbound messaging, file downloads, and callback queries.

**Public API:**

| Type/Method | Purpose |
|-------------|---------|
| `TelegramBot.Configuration` | Bot token, allowed user IDs, poll timeout, retry delay |
| `init(configuration:logger:)` | Create bot with config and optional logger |
| `startPolling(handler:)` | Start long polling loop. Blocks until `stop()` called. Handler receives `TelegramUpdate` |
| `stop()` | Set `running = false` to break the poll loop |
| `sendMessage(chatId:text:replyTo:)` | Send HTML-formatted text message |
| `sendPhoto(chatId:imageData:filename:caption:replyMarkup:)` | Upload photo via multipart/form-data |
| `sendDocument(chatId:fileData:filename:mimeType:caption:replyMarkup:)` | Upload document via multipart/form-data |
| `answerCallbackQuery(id:text:)` | Acknowledge inline keyboard button press |
| `editMessageReplyMarkup(chatId:messageId:markup:)` | Add/remove inline keyboard on existing message |
| `downloadFile(fileId:)` | Download file from Telegram (getFile + fetch file_path) |

**Types:**

| Type | Fields |
|------|--------|
| `TelegramUpdate` | `updateId`, `message?`, `callbackQuery?` |
| `TelegramMessage` | `messageId`, `chatId`, `userId`, `firstName`, `text?`, `caption?`, `photo?`, `replyToMessage?`, `date` |
| `TelegramCallbackQuery` | `id`, `userId`, `firstName`, `messageId?`, `chatId?`, `data?` |
| `TelegramPhotoSize` | `fileId`, `width`, `height` |
| `InlineKeyboard` | `rows: [[InlineButton]]` with `toJSON()` |
| `InlineButton` | `text`, `callbackData` |
| `SendResult` | `ok`, `messageId?` |

**Concurrency:** `TelegramBot` is `@unchecked Sendable`. The polling loop runs in the caller's async context. `running` is a simple bool (no lock) -- single-writer (stop/start), single-reader (poll loop).

**Auth:** Updates from user IDs not in `allowedUserIds` are silently dropped with a log warning.

**Dependencies:** Foundation, Logging.

---

### `TelegramCommandParser.swift`

Stateless command parser. Maps raw message text to a `BotCommand` enum.

**Public API:**

| Method | Purpose |
|--------|---------|
| `parse(_:inDiscussMode:)` | Parse message text into `BotCommand`. `inDiscussMode` controls whether bare text maps to `.chatMessage` or `.render` |
| `parseReplyIntent(_:)` | Classify reply-to-image text into `ReplyIntent` |
| `parsePrompt(_:defaultMode:)` | Extract character name and inline mode override from prompt text |

**`BotCommand` enum (36 cases):**

Content modes: `.neutral`, `.banana`, `.avocado`

Toggles: `.enhance(on:)`, `.upscale(on:)`, `.polish(on:)`, `.verbose(on:)`, `.autoVideo(on:)`, `.resolution(target:)`

Settings: `.aspect(mode:)`, `.cfg(value:)`, `.seed(value:)`, `.saturation(value:)`, `.colorTemp(kelvin:)`, `.film(lookId:)`

Generation: `.render(prompt:)`, `.batch(count:prompt:)`, `.vary(count:prompt:)`, `.sequence(count:story:)`, `.video(prompt:)`

Session: `.chat`, `.imagine(description:)`, `.endChat`, `.shipCue`, `.chatMessage(text:)`

Admin: `.status`, `.help`, `.reset`, `.look(id:)`, `.queue(subcommand:)`

**`ReplyIntent` enum (5 cases):**

`.rerender`, `.hq`, `.upscaleReply`, `.video(motion:)`, `.newPrompt(text:)`

**`ParsedPrompt`:** `prompt` (cleaned), `character?` (detected name), `contentMode?` (inline override).

**Ship-cue detection:** In discuss mode, bare text matching "go", "render", "render it", "ship", "ship it", "do it", "let's go", "send it", "fire it" returns `.shipCue` instead of `.chatMessage`.

**Character detection:** Case-insensitive scan of the prompt for "kira", "bree", "todd". First match wins. Returns capitalized name.

**Count parsing for batch/vary/sequence:** If the first word is a number, it is used as count (clamped 1-8). Otherwise, the entire arg string is the prompt and a default count is used (3 for batch/vary, 4 for sequence).

**Dependencies:** Foundation only.

---

### `ContentModeManager.swift`

Thread-safe content mode state with JSON disk persistence.

**Public API:**

| Method/Property | Purpose |
|-----------------|---------|
| `init(configPath:)` | Load mode from `~/.comfybox/content-mode.json` or default to `.neutral` |
| `current: Mode` | Thread-safe read of current mode |
| `set(_:)` | Set mode and persist via atomic temp-file + rename |
| `displayName(for:)` | Human-readable name: "Neutral (SFW)", "Banana (suggestive)", "Avocado (explicit)" |
| `emoji(for:)` | Mode emoji: red apple, banana, avocado |

**`Mode` enum:** `.neutral`, `.banana`, `.avocado`. RawValue is the string used in JSON.

**Thread safety:** NSLock around `_current`. Disk writes are fire-and-forget after the lock is released.

**Persistence:** `{"mode": "neutral"}` written to `~/.comfybox/content-mode.json` via temp file UUID + rename.

**Dependencies:** Foundation only.

---

### `CharacterLoader.swift`

Loads tiered character descriptions from a JSON file and resolves them by content mode.

**Public API:**

| Method | Purpose |
|--------|---------|
| `init(configPath:)` | Load from `~/.comfybox/characters.json`. Graceful degradation if missing |
| `description(for:mode:)` | Get mode-gated description string. Returns nil if character not found |
| `allNames()` | All character names (sorted, lowercase) |
| `has(_:)` | Check if a character exists (case-insensitive) |

**JSON format:**

```json
{
  "kira": {
    "base": "physical description always included",
    "banana": "appended in banana and avocado modes",
    "avocado": "appended only in avocado mode"
  }
}
```

**Tier resolution:**

| Mode | Composition |
|------|-------------|
| neutral | `base` |
| banana | `base` + `banana` (if present) |
| avocado | `base` + `banana` (if present) + `avocado` (if present) |

This mirrors the server-side `characters.ts` logic. Strings are space-joined.

**Dependencies:** Foundation only.

---

### `PromptOptimizer.swift`

Local LLM client for rewriting user prompts into Z-Image Turbo native format.

**Public API:**

| Method | Purpose |
|--------|---------|
| `init(configuration:logger:)` | Create optimizer with endpoint config |
| `optimize(prompt:character:characterDescription:contentMode:)` | Async prompt optimization. Returns `OptimizeResult` |

**`OptimizeResult`:** `prompt` (rewritten), `enhanced` (bool), `note?` (fallback info).

**`Configuration`:** `ollamaBaseURL`, `lmStudioBaseURL?`, `model` (default: `"qwen3:8b"`), `timeoutSeconds` (default: 15), `enabled`.

**Fallback chain:**

1. Ollama at `ollamaBaseURL/v1/chat/completions`
2. LM Studio at `lmStudioBaseURL/v1/chat/completions`
3. Rule-based `wrapInQwen3Format()` -- wraps prompt in YOUR CONTEXT / YOUR PHOTO with mode-appropriate defaults

If `enabled` is false, step 3 is used directly.

**System prompts (3 variants):**

Each mode has a dedicated system prompt embedding the Z-Image Turbo rendering rules:

- **Neutral:** SFW, fully clothed, focus on character presence and environment
- **Banana:** Suggestive, lingerie, partial nudity, tension and implication
- **Avocado:** Explicit, graphic anatomical language, nude default, amateur aesthetic

All three share the `zImageRules` constant: YOUR CONTEXT / YOUR PHOTO format spec, facial detail scaling, hard rules (no keywords, no negatives, no camera brands, narrative prose only).

**User message construction:**

1. Scene type hint (inferred from keywords: POV, portrait, macro, full body, wide, cinematic)
2. Character context injection (if character detected)
3. Raw input prompt
4. Mode-specific instruction suffix

**Temperature:** 0.4 for neutral/banana, 0.9 for avocado (pushes past model safety defaults).

**Output cleaning (`cleanLLMOutput`):**

- Strips `<think>...</think>` blocks (Qwen3 chain-of-thought)
- Strips stop tokens: `<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`
- Detects refusal patterns ("I'm sorry", "I cannot", "content policy", etc.) and returns empty string

**Static helpers (used by coordinator):**

| Method | Purpose |
|--------|---------|
| `selectSystemPrompt(contentMode:)` | Return the system prompt for a mode |
| `buildUserMessage(...)` | Build the user message with scene hints and character context |
| `cleanLLMOutput(_:)` | Strip think tags, stop tokens, detect refusals |
| `wrapInQwen3Format(prompt:contentMode:)` | Rule-based YOUR CONTEXT / YOUR PHOTO wrapping |

**Dependencies:** Foundation, Logging.

---

### `ImageBotCoordinator.swift`

Central orchestrator. Owns all other components, handles the full message-to-image pipeline, and manages the bot lifecycle.

**Public API:**

| Method | Purpose |
|--------|---------|
| `init(configuration:logger:)` | Create coordinator, initialize all subsystems |
| `run()` | Start the bot. Blocks until shutdown. Calls `bot.startPolling()` |
| `shutdown()` | Signal graceful stop |

**`Configuration`:** `telegram` (bot config), `warmServerHost`, `warmServerPort`, `outputDirectory`, `galleryDirectory?`, `optimizer` (config), `characterConfigPath?`, `contentModeConfigPath?`.

**Owned components:**

| Property | Type | Purpose |
|----------|------|---------|
| `bot` | `TelegramBot` | Telegram API client |
| `warmServer` | `WarmServerClient` | WarmServer HTTP client |
| `contentModeManager` | `ContentModeManager` | Mode state |
| `characterLoader` | `CharacterLoader` | Character descriptions |
| `promptOptimizer` | `PromptOptimizer` | LLM prompt rewriting |
| `sessions` | `SessionState` | Per-chat state storage |

**Update routing:**

```
handleUpdate()
  |-- callbackQuery? --> handleCallbackQuery()
  |-- reply-to-image? --> handleReplyToImage()
  |-- text message --> handleTextMessage()
                         |-- TelegramCommandParser.parse()
                         |-- switch on BotCommand (36 cases)
```

**Render pipeline (single image):**

```
handleRender(prompt)
  1. parseAndResolve(prompt)
       -> TelegramCommandParser.parsePrompt() -- extract character, inline mode
       -> CharacterLoader.description() -- load tiered description
       -> Return ParsedContext
  2. Send status message
  3. generateImage(prompt, character, characterDescription, mode, state, seed)
       -> If enhance: PromptOptimizer.optimize()
       -> Else: prepend character description
       -> generateImageDirect(finalPrompt, state, seed)
            -> WarmServerClient.post("/v1/generate")
            -> If polish: WarmServerClient.post("/v1/generate", strength=0.35, steps=30)
            -> If upscale: WarmServerClient.post("/v1/upscale")
            -> If 4K: second upscale pass
  4. applyPostProcessing(imageData, state, wasUpscaled)
       -> PostProcessor.applyPipeline()
  5. buildCaption()
  6. sendImageWithKeyboard()
       -> bot.sendPhoto() then bot.editMessageReplyMarkup() to attach keyboard
       -> If >8MB: bot.sendDocument() (no keyboard)
  7. Store RenderContext for the sent message ID
  8. copyToGallery()
```

**Callback query handling (inline keyboard):**

Callback data format: `"action:chatId:messageId"`

| Action | Handler | Behavior |
|--------|---------|----------|
| `rerender` | `handleCallbackRerender` | Same prompt, new random seed |
| `hq` | `handleCallbackHQ` | Same prompt + seed, force polish on |
| `video` | inline | Message directing to @BaristaBree_Bot |

**Reply-to-image handling:**

| `ReplyIntent` | Handler | Behavior |
|---------------|---------|----------|
| `.rerender` | `handleCallbackRerender` | Same prompt, new seed |
| `.hq` | `handleCallbackHQ` | Same prompt + seed, polish on |
| `.upscaleReply` | `handleReplyUpscale` | Upscale + sharpen + post-process |
| `.video(motion:)` | inline | Message directing to @BaristaBree_Bot |
| `.newPrompt(text:)` | `handleImg2Img` | img2img at 50% strength with new prompt |

**Discuss mode flow:**

```
/chat -> enterDiscussMode() -> isInDiscussMode = true
  |
  user text -> .chatMessage -> handleDiscussMessage()
       -> callDiscussLLM(messages) via Ollama/LM Studio
       -> extractPromptFromDiscussion() -> store discussCurrentPrompt
       -> Send response to user
  |
  ship cue -> .shipCue -> handleShipCue()
       -> Use discussCurrentPrompt (or last user message)
       -> exitDiscussMode()
       -> handleRender()
  |
  /end -> exitDiscussMode()
```

**Multi-image commands:**

- **Batch:** Sequential loop, same prompt, different seeds. Progress updates per image.
- **Vary:** Each iteration passes a variation instruction to the optimizer. Progress updates per image.
- **Sequence:** First pass: optimizer breaks story into numbered scenes. Second pass: each scene is optimized and rendered independently. Fallback: split by sentence boundaries.

**Error types:**

```swift
enum RenderError {
  case warmServerDown(String)
  case generateFailed(String)
  case upscaleFailed(String)
}
```

**Dependencies:** Foundation, Logging, WarmServerClient.

---

### `PostProcessor.swift`

Post-processing effects pipeline using Core Image filters (macOS) with ImageMagick CLI fallback (Linux).

**Public API:**

| Method | Purpose |
|--------|---------|
| `sharpen(imageData:radius:intensity:)` | CIUnsharpMask (default radius 2.5, intensity 0.5) |
| `adjustSaturation(imageData:factor:)` | CIColorControls saturation (0=grayscale, 1=original, 2=double) |
| `adjustColorTemperature(imageData:kelvin:)` | CITemperatureAndTint (2000-10000K, 6500=neutral) |
| `applyFilmLook(imageData:look:)` | CIColorMatrix + CIColorControls chain |
| `applyPipeline(imageData:saturation:colorTemp:filmLookId:sharpenAfterUpscale:)` | Full pipeline: sharpen -> saturation -> temp -> film look |
| `availableLooks()` | List all film look `(id, name)` tuples |
| `findLook(_:)` | Case-insensitive lookup by ID |

**Film look presets (5 built-in):**

| ID | Name | Character |
|----|------|-----------|
| `kodak-portra` | Kodak Portra 400 | Warm reds, lifted blacks, slightly desaturated (0.9x), reduced contrast (0.95x) |
| `fuji-velvia` | Fuji Velvia 50 | Boosted RGB, high saturation (1.3x), increased contrast (1.1x) |
| `ilford-hp5` | Ilford HP5 Plus | Equal RGB channels (black and white), zero saturation, high contrast (1.15x) |
| `cinestill-800t` | CineStill 800T | Reduced greens, boosted blues with green crossover, slight desaturation (0.95x) |
| `kodak-ektar` | Kodak Ektar 100 | Boosted reds and greens, high saturation (1.2x), punchy contrast (1.08x) |

Each preset is a `FilmLook` struct with SIMD4 color matrix vectors (`redVector`, `greenVector`, `blueVector`, `biasVector`) plus `saturationAdjust` and `contrastAdjust` floats.

**CIFilter pipeline order:**

1. CIUnsharpMask (only if `sharpenAfterUpscale` -- counteracts upscaler softness)
2. CIColorControls (saturation)
3. CITemperatureAndTint (color temperature)
4. CIColorMatrix + CIColorControls (film look -- applied last as the "film stock envelope")

**Rendering:** `CIContext.createCGImage()` -> `CGImageDestinationCreateWithData()` -> PNG output. Software renderer disabled (uses GPU).

**ImageMagick fallback (non-macOS):** Uses `/opt/homebrew/bin/convert` with temp files. Saturation via `-modulate`, temperature via `-colorize` approximation, film looks via modulate brightness/saturation.

**Dependencies:** Foundation, CoreImage, CoreGraphics, AppKit (macOS). No external dependencies.

---

### `SessionState.swift`

Per-chat render settings and context storage.

**Public types:**

**`ChatState` (value type):**

| Property | Type | Default | Purpose |
|----------|------|---------|---------|
| `enhanceEnabled` | Bool | config-dependent | Prompt optimization on/off |
| `upscaleEnabled` | Bool | false | SeedVR 2x upscale |
| `polishEnabled` | Bool | false | Two-pass refinement |
| `autoVideoEnabled` | Bool | false | Auto-video after render |
| `verboseEnabled` | Bool | false | Verbose captions |
| `aspectMode` | String | "portrait" | Aspect ratio mode |
| `cfgOverride` | Double? | nil | CFG guidance override |
| `seedLock` | Int? | nil | Locked seed value |
| `resolutionTarget` | String? | nil | "2k" or "4k" |
| `saturation` | Double? | nil | Saturation factor |
| `colorTemp` | Int? | nil | Color temperature (Kelvin) |
| `filmLook` | String? | nil | Film look preset ID |
| `lastPrompt` | String? | nil | Last rendered prompt |
| `lastImagePath` | String? | nil | Last rendered image path |
| `lastSeed` | Int? | nil | Last render seed |
| `renderContexts` | [Int: RenderContext] | [:] | messageId -> render context map |
| `isInDiscussMode` | Bool | false | In discuss mode |
| `discussHistory` | [DiscussEntry] | [] | Discuss conversation history |
| `discussCurrentPrompt` | String? | nil | Current refined prompt in discuss mode |

**`RenderContext` (value type):** `prompt`, `imagePath`, `seed?`, `character?`, `contentMode`, `enhanceEnabled`. Stored per bot-sent message ID for inline keyboard and reply-to-image.

**`DiscussEntry`:** `role` ("user" or "assistant"), `content`.

**Eviction policies:**

- Render contexts: max 50 per chat. Oldest messageIds evicted first.
- Discuss history: max 20 entries. Oldest entries dropped.

**Aspect dimensions:**

| Mode | Width | Height |
|------|-------|--------|
| square | 1024 | 1024 |
| portrait | 768 | 1024 |
| landscape | 1024 | 768 |
| wide | 1024 | 576 |
| tall | 576 | 1024 |

**`SessionState` (reference type, thread-safe):**

| Method | Purpose |
|--------|---------|
| `init(defaultEnhance:)` | Create with default enhance setting |
| `getState(chatId:)` | Get or create ChatState for a chat |
| `updateState(chatId:update:)` | Mutate state via closure |

Thread safety: NSLock around the `states` dictionary. State is copied out on get, replaced on update.

**Dependencies:** Foundation only.

---

### Entry Point (`Sources/ComfyBox/main.swift`)

The `telegram` subcommand is parsed in `runTelegram(args:)`.

**CLI argument parsing:**

```
ComfyBox telegram [--bot-token <token>] [--config <path>]
                  [--port <port>] [--host <host>]
                  [--enhance [on|off]] [--no-enhance]
                  [--help]
```

**Config loading (`loadTelegramConfig`):**

Resolution order for each field: CLI flag > environment variable > config file > default.

The bot token is required. Resolution: `--bot-token` > `COMFYBOX_TELEGRAM_TOKEN` env > `botToken` in config file. Throws if none found.

**Signal handling:** SIGINT and SIGTERM trigger `coordinator.shutdown()` via `DispatchSource`.

**Caffeinate:** Spawns `/usr/bin/caffeinate -s -w <PID>` to prevent macOS sleep while the bot runs.

**Run loop:** `Task { coordinator.run() }` + `dispatchMain()`.

---

## Extension Points

### Adding a New Command

1. Add a case to `BotCommand` in `TelegramCommandParser.swift`
2. Add the parser match in `parse(_:inDiscussMode:)` (the `/` command switch)
3. Add the handler case in `ImageBotCoordinator.handleTextMessage()`
4. Implement the handler method on `ImageBotCoordinator`
5. Add to the help text in `sendHelp(chatId:)`

### Adding a Film Look Preset

Add a `FilmLook` entry to the `filmLooks` array in `PostProcessor.swift`:

```swift
FilmLook(
  id: "my-look",
  name: "My Custom Look",
  redVector: SIMD4<Float>(1.0, 0.0, 0.0, 0.0),
  greenVector: SIMD4<Float>(0.0, 1.0, 0.0, 0.0),
  blueVector: SIMD4<Float>(0.0, 0.0, 1.0, 0.0),
  biasVector: SIMD4<Float>(0.0, 0.0, 0.0, 0.0),
  saturationAdjust: 1.0,
  contrastAdjust: 1.0
)
```

The color matrix vectors are RGBA and map to CIColorMatrix's `inputRVector`, `inputGVector`, `inputBVector`, `inputBiasVector`. Identity matrix = no change.

### Adding a Character

Add an entry to `~/.comfybox/characters.json`:

```json
{
  "newchar": {
    "base": "A person with specific physical traits...",
    "banana": "suggestive additions...",
    "avocado": "explicit additions..."
  }
}
```

Then add the name to `characterNames` in `TelegramCommandParser.swift` for auto-detection:

```swift
private static let characterNames = ["kira", "bree", "todd", "newchar"]
```

### Adding a New Post-Processing Effect

1. Add a `CIFilter`-based static method to `PostProcessor` (with ImageMagick fallback)
2. Add the corresponding setting to `ChatState` in `SessionState.swift`
3. Add a command to `BotCommand` and parser
4. Add a handler to the coordinator
5. Include the new effect in `applyPipeline()` at the appropriate stage

### Adding a New LLM Backend

The optimizer talks to any OpenAI-compatible `/v1/chat/completions` endpoint. To add a new provider:

1. Add a URL field to `PromptOptimizer.Configuration`
2. Add it to the fallback chain in `optimize()` (after Ollama, before rule-based)
3. Add the corresponding config field to `loadTelegramConfig()` in `main.swift`
