# ComfyBox Telegram Bot User Guide

@CoffeeImageBot -- generate images from Telegram using your Mac's GPU.

## Getting Started

### Prerequisites

1. **WarmServer running.** Start it before the bot:

   ```
   ComfyBox serve --port 7862
   ```

2. **Bot token.** Create one via [@BotFather](https://t.me/BotFather) on Telegram, or use the one already provisioned for @CoffeeImageBot.

3. **Ollama (optional but recommended).** Prompt enhancement requires a local LLM. Install Ollama and pull the model:

   ```
   ollama pull qwen3:8b
   ```

### Starting the Bot

Minimal launch:

```
ComfyBox telegram --bot-token YOUR_TOKEN
```

Or use environment variable:

```
export COMFYBOX_TELEGRAM_TOKEN=YOUR_TOKEN
ComfyBox telegram
```

Or use the config file (recommended for persistent setups):

```
ComfyBox telegram --config ~/.comfybox/telegram.json
```

All CLI options:

| Flag | Default | Purpose |
|------|---------|---------|
| `--bot-token <token>` | none | Telegram Bot API token |
| `--config <path>` | `~/.comfybox/telegram.json` | Config file path |
| `--port <port>` | 7862 | WarmServer port |
| `--host <host>` | 127.0.0.1 | WarmServer host |
| `--enhance [on\|off]` | on | Enable/disable prompt optimization |
| `--no-enhance` | -- | Disable prompt optimization |

Resolution order: CLI flags > environment variables > config file > defaults.

The bot keeps the Mac awake via `caffeinate` while running. SIGINT or SIGTERM shuts down gracefully.

### Configuration File

Create `~/.comfybox/telegram.json`:

```json
{
  "botToken": "123456:ABC-DEF...",
  "allowedUserIds": [8754779862],
  "warmServer": {
    "host": "127.0.0.1",
    "port": 7862
  },
  "optimizer": {
    "enabled": true,
    "ollamaBaseURL": "http://localhost:11434",
    "lmStudioBaseURL": "http://localhost:1234",
    "model": "qwen3:8b",
    "timeoutSeconds": 15
  },
  "characterConfigPath": "~/.comfybox/characters.json",
  "outputDirectory": "~/Pictures/ComfyBox/Telegram",
  "galleryDirectory": "~/Pictures/ComfyBox/Gallery"
}
```

**`allowedUserIds`** -- only these Telegram user IDs can interact with the bot. Find your ID by messaging [@userinfobot](https://t.me/userinfobot).

**`galleryDirectory`** -- optional. If set, every generated image is copied here in addition to `outputDirectory`.

---

## Basic Usage

Send any text message and the bot renders an image.

```
A black cat sitting on a windowsill in the rain
```

The bot will:
1. Parse the prompt for character names and inline mode overrides
2. Optimize the prompt via Ollama (if enhance is on)
3. Send it to WarmServer for rendering
4. Deliver the image with a caption showing mode, seed, and timing
5. Attach inline keyboard buttons (Rerender, HQ, Video)

### Character Names

The bot detects character names -- **Kira**, **Bree**, and **Todd** -- anywhere in your prompt. When detected, the character's physical description is loaded from `~/.comfybox/characters.json` and injected into the prompt automatically.

```
Kira sitting in a coffee shop reading a book
```

This loads Kira's full description (skin tone, build, hair, eyes, etc.) and weaves it into the optimized prompt. The description changes based on the active content mode (see below).

---

## Content Modes

Three content modes control the style of character descriptions and prompt optimization.

| Command | Mode | Effect |
|---------|------|--------|
| `/neutral` or `/apple` | Neutral (SFW) | Fully clothed subjects, no suggestive content |
| `/banana` | Banana (suggestive) | Lingerie, partial nudity, intimate framing |
| `/avocado` | Avocado (explicit) | Full nudity, graphic anatomical descriptions |

Mode is global and persistent -- it survives bot restarts (saved to `~/.comfybox/content-mode.json`).

### Inline Mode Override

Override the mode for a single prompt by including the mode command in your text:

```
Kira at the beach /avocado
```

This uses avocado mode for this render only without changing the global setting.

---

## Prompt Enhancement

By default, every prompt is rewritten by a local LLM (Ollama with Qwen3-8B) into the Z-Image Turbo native format:

```
YOUR CONTEXT:
[lens, lighting, aesthetic, skin texture approach]

YOUR PHOTO:
[subject, action, composition, environment, atmosphere]
```

This format matches Z-Image's Qwen3-4B text encoder, producing significantly better results than raw prompts.

### Toggle Enhancement

```
/enhance off     -- disable, send raw prompts
/enhance on      -- re-enable
/enhance         -- toggle current state
```

When enhancement is off and a character name is detected, the character description is prepended to the prompt as-is.

### Fallback Chain

1. **Ollama** (primary) -- `http://localhost:11434`
2. **LM Studio** (fallback) -- `http://localhost:1234`
3. **Rule-based wrapping** (final fallback) -- wraps in YOUR CONTEXT / YOUR PHOTO format with mode-appropriate defaults

If both LLMs are down, you still get well-formatted prompts.

---

## Batch and Variations

### Batch

Generate multiple images from the same prompt (different random seeds each time):

```
/batch 5 Kira in a sundress on a rooftop at sunset
```

Generates 5 images, each with a unique seed. Count range: 1-8.

If you omit the count, it defaults to 3:

```
/batch Kira in a sundress on a rooftop at sunset
```

### Vary

Generate multiple prompt variations -- the optimizer rewrites the prompt differently each time, varying angle, lighting, or composition:

```
/vary 4 Kira in a coffee shop
```

Produces 4 distinct interpretations of the same scene concept. Default count: 3.

### Sequence

Break a story into sequential frames:

```
/seq 4 Kira walks through a garden gate, discovers a hidden fountain, sits on the edge, dips her feet in the water
```

When enhancement is on, the LLM breaks the story into separate scene descriptions. When off, the story is split by sentence/period boundaries.

Default count: 4. Range: 1-8.

---

## Image Controls

### Aspect Ratio

```
/aspect portrait    -- 768x1024 (default)
/aspect landscape   -- 1024x768
/aspect square      -- 1024x1024
/aspect wide        -- 1024x576
/aspect tall        -- 576x1024
/aspect             -- show current setting
```

### CFG (Guidance Scale)

Override the model's default guidance:

```
/cfg 3.5     -- set CFG to 3.5
/cfg         -- reset to model default
```

Range: 0-20. Z-Image Turbo is CFG-distilled (trained at CFG 1.0), so overriding this is rarely needed.

### Seed

Lock the seed for reproducible results:

```
/seed 42       -- lock seed to 42
/seed random   -- back to random seeds
/seed          -- back to random seeds
```

### Upscale

Enable SeedVR 2x upscaling on every render:

```
/upscale on    -- enable
/upscale off   -- disable
/upscale       -- toggle
```

### Polish

Two-pass refinement: renders the image, then re-renders at 35% strength with 30 steps for added detail:

```
/polish on     -- enable
/polish off    -- disable
/polish        -- toggle
```

### Resolution Targets

Shorthand commands that enable upscale and set the target:

```
/2k            -- enable upscale to 2K resolution
/4k            -- double upscale to 4K resolution
/2k off        -- disable
/4k off        -- disable
```

4K runs two upscale passes back-to-back.

---

## Post-Processing

Post-processing effects are applied to every rendered image using Core Image filters. They stack and persist until reset.

### Saturation

```
/saturation 1.3   -- boost saturation (1.0 = unchanged)
/saturation 0.5   -- reduce saturation
/saturation 0     -- grayscale
/saturation off   -- disable
```

Range: 0-2.

### Color Temperature

```
/temp 4500     -- warm (lower Kelvin = warmer)
/temp 8000     -- cool (higher Kelvin = cooler)
/temp off      -- disable
```

Range: 2000-10000K. 6500K is neutral daylight.

### Film Look Presets

Apply a film stock color grade:

```
/film kodak-portra     -- warm skin tones, slightly desaturated
/film fuji-velvia      -- vibrant colors, high contrast
/film ilford-hp5       -- black and white with boosted contrast
/film cinestill-800t   -- tungsten-balanced, blue shadows
/film kodak-ektar      -- vivid colors, punchy contrast
/film off              -- disable
```

### Browse Film Looks

```
/look              -- list all available looks with active indicator
/look kodak-portra -- apply a look (alias for /film)
```

### Reset Post-Processing

Clear all post-processing settings at once:

```
/reset
```

Clears saturation, color temperature, and film look. Does not affect other settings like aspect or seed.

---

## Inline Keyboards

Every delivered image has three buttons:

| Button | Action |
|--------|--------|
| **Rerender** | Re-render the same prompt with a new random seed |
| **HQ** | Re-render with polish (two-pass refinement) enabled, same seed |
| **Video** | Directs to @BaristaBree_Bot for video generation |

Buttons reference the original render context (prompt, character, mode, seed). Context is stored per message and evicts after 50 entries per chat.

---

## Reply Actions

Reply to any bot-sent image with text to trigger an action:

| Reply text | Action |
|------------|--------|
| `rerender` or `again` or `redo` | Re-render with new random seed |
| `hq` | Re-render with polish, same seed |
| `upscale` | Upscale the image via SeedVR 2x (with sharpen) |
| `video` | Redirect to @BaristaBree_Bot |
| `video <description>` | Redirect with motion description |
| *anything else* | **img2img** -- re-render using the original image as input with your new prompt |

img2img sends your new text through the optimizer and generates at 50% strength using the original image as the starting point. This is great for iterating on a composition.

---

## Discuss Mode

Discuss mode is a back-and-forth conversation with the local LLM to collaboratively design a prompt before rendering.

### Enter Discuss Mode

```
/chat
```

Or equivalently:

```
/discuss
```

### How It Works

1. Describe what you want in plain language
2. The LLM proposes a prompt in YOUR CONTEXT / YOUR PHOTO format and explains its choices
3. Reply with refinements ("make it warmer", "change to landscape", "add rain")
4. When satisfied, say **go**, **ship it**, **render it**, **send it**, **fire it**, or **do it**
5. The bot exits discuss mode and renders the refined prompt

### Example Session

```
You: /chat
Bot: Discuss mode -- let's design a prompt together...

You: Kira in a moody jazz club
Bot: [proposes prompt with smoky atmosphere, stage lighting...]

You: Make it a closer shot, more intimate
Bot: [revises with tighter framing, shallow depth of field...]

You: ship it
Bot: Rendering...
[delivers image]
```

### Exit Without Rendering

```
/end
```

Or equivalently:

```
/exit
```

### One-Shot Imagine

Skip the conversation and let the LLM design a prompt from a brief description:

```
/imagine Kira in a cyberpunk alley
```

The LLM designs a complete prompt and renders it immediately, then includes a brief creative report explaining its choices in the caption.

---

## Queue Management

Check and control the WarmServer render queue:

```
/queue           -- show queue status (server state, pending jobs, completed count)
/queue list      -- list pending jobs (up to 10, with truncated prompts)
/queue cancel    -- clear all pending jobs from the queue
```

---

## Configuration Reference

### `~/.comfybox/telegram.json`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `botToken` | string | required | Telegram Bot API token |
| `allowedUserIds` | int[] | `[8754779862]` | Authorized Telegram user IDs |
| `warmServer.host` | string | `"127.0.0.1"` | WarmServer hostname |
| `warmServer.port` | int | `7862` | WarmServer port |
| `optimizer.enabled` | bool | `true` | Enable prompt enhancement |
| `optimizer.ollamaBaseURL` | string | `"http://localhost:11434"` | Ollama endpoint |
| `optimizer.lmStudioBaseURL` | string | `"http://localhost:1234"` | LM Studio fallback endpoint |
| `optimizer.model` | string | `"qwen3:8b"` | LLM model name |
| `optimizer.timeoutSeconds` | int | `15` | LLM request timeout |
| `characterConfigPath` | string | `"~/.comfybox/characters.json"` | Path to character descriptions |
| `contentModeConfigPath` | string | `"~/.comfybox/content-mode.json"` | Path to mode persistence file |
| `outputDirectory` | string | `"~/Pictures/ComfyBox/Telegram"` | Where rendered images are saved |
| `galleryDirectory` | string | none | Optional gallery copy directory |

### `~/.comfybox/characters.json`

```json
{
  "kira": {
    "base": "A 23-year-old Filipina woman, petite 4'11\" build, warm brown skin...",
    "banana": "wearing a lace camisole...",
    "avocado": "nude, athletic build visible..."
  },
  "bree": {
    "base": "A 5'8\" Korean-Filipino woman...",
    "banana": "...",
    "avocado": "..."
  }
}
```

The `base` description is always included. `banana` is appended in banana and avocado modes. `avocado` is appended only in avocado mode. Tiered composition mirrors the server-side `characters.ts` logic.

### `~/.comfybox/content-mode.json`

Auto-managed. Contains the current mode:

```json
{
  "mode": "neutral"
}
```

---

## Full Command Reference

| Command | Arguments | Description |
|---------|-----------|-------------|
| `/help` | -- | Show help text with current settings |
| `/status` | -- | WarmServer status, model, queue, bot uptime |
| `/neutral` or `/apple` | -- | Set content mode to neutral (SFW) |
| `/banana` | -- | Set content mode to banana (suggestive) |
| `/avocado` | -- | Set content mode to avocado (explicit) |
| `/enhance` | `[on\|off]` | Toggle or set prompt enhancement |
| `/batch` | `[N] <prompt>` | Generate N images (default 3, max 8) |
| `/vary` | `[N] <prompt>` | N prompt variations (default 3, max 8) |
| `/seq` or `/sequence` | `[N] <story>` | N sequential story frames (default 4, max 8) |
| `/imagine` | `<description>` | One-shot: LLM designs and renders |
| `/chat` or `/discuss` | -- | Enter discuss mode |
| `/end` or `/exit` | -- | Exit discuss mode |
| `/queue` | `[status\|list\|cancel]` | Queue management |
| `/aspect` | `<mode>` | Set aspect ratio |
| `/cfg` | `<value>` or none | Set or clear CFG override |
| `/seed` | `<N\|random>` | Lock or randomize seed |
| `/upscale` | `[on\|off]` | Toggle SeedVR 2x upscale |
| `/polish` | `[on\|off]` | Toggle two-pass polish |
| `/2k` | `[on\|off]` | Enable/disable 2K resolution |
| `/4k` | `[on\|off]` | Enable/disable 4K resolution |
| `/verbose` | `[on\|off]` | Toggle verbose mode |
| `/autovideo` | `[on\|off]` | Toggle auto-video |
| `/saturation` | `<0-2\|off>` | Adjust saturation |
| `/temp` | `<2000-10000\|off>` | Adjust color temperature |
| `/film` | `<look-id\|off>` | Apply or clear film look |
| `/look` | `[look-id]` | List or apply film looks |
| `/reset` | -- | Clear all post-processing settings |
| `/video` | `<prompt>` | Redirects to @BaristaBree_Bot |

Bare text (no `/` prefix) outside of discuss mode is treated as a render prompt.

---

## Troubleshooting

### WarmServer Not Running

**Symptom:** Every prompt returns "WarmServer not available -- start `ComfyBox serve` first."

**Fix:** Start the WarmServer in another terminal:

```
ComfyBox serve --port 7862
```

The bot connects on-demand per request. No restart needed once the server is up.

### Ollama Not Running

**Symptom:** Prompts still render but captions do not show the enhance indicator. Log shows "Optimizer unavailable -- using rule-based format wrapping."

**Fix:** Start Ollama:

```
ollama serve
```

The bot tries Ollama on every request. No restart needed once Ollama is up.

### Bot Not Responding

1. **Check the bot is running.** Look for the process: `ps aux | grep ComfyBox`.
2. **Check allowed users.** Your Telegram user ID must be in the `allowedUserIds` list. Unauthorized users are silently rejected (check the bot log for "rejected update from unauthorized user").
3. **Check Telegram API.** Ensure the Mac can reach `api.telegram.org`. The bot uses long polling (outbound HTTPS only -- no inbound ports needed).
4. **Check the log output.** The bot logs to stderr. Poll errors retry automatically after 5 seconds.

### Images Not Delivering

**Symptom:** Bot says "Rendering..." but no image arrives.

1. Check WarmServer health: send `/status` to the bot.
2. Check `~/Pictures/ComfyBox/Telegram/` for output files -- the image may have been generated but failed to upload.
3. Telegram has a 10MB photo limit. Images over 8MB are sent as documents instead. If upscale + 4K produces very large files, this is expected.

### Post-Processing Not Applied

Post-processing uses Core Image (macOS native). If running on Linux, it falls back to ImageMagick CLI (`/opt/homebrew/bin/convert`). Ensure ImageMagick is installed if not on macOS.
