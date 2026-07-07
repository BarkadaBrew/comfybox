# Fruit-Mode Selector in the Generate Tab — Design

**Date:** 2026-07-07
**Status:** Approved (design); pending implementation plan
**Author:** Todd Walderman + Claude

## Summary

Add a content-mode ("fruit mode") selector — 🍎 Neutral / 🍌 Banana / 🥑 Avocado — to
the desktop Generate tab. Its sole job is to **steer the prompt optimizer and the
negative prompt with mode-appropriate text**. It never changes guidance, steps, or any
numeric generation parameter.

## Background / Current State

- **Content modes already exist** as a concept: `~/.comfybox/content-modes.json` defines
  `neutral`, `banana`, `avocado`, each with a `promptHint` and (unused) `guidanceBoost`
  and (empty) `negativePromptAdditions`. Server has `ContentModeStore` / `ContentModeManager`.
- **The optimizer already consumes content mode.** `POST /v1/enhance`
  (`WarmServer.enhancePromptResponse`) accepts a `content_mode` field, defaults to
  `neutral`, and injects that mode's `promptHint` into how it rewrites the prompt
  template (banana → "sensual, intimate, suggestive"; avocado → "explicit, uncensored,
  anatomically detailed").
- **The gap is on the client.** `EngineService.enhancePrompt()` sends only
  `{"prompt": …}` — no `content_mode` — so the Generate tab's Enhance button is
  permanently stuck in neutral. There is no content-mode control anywhere in the desktop
  UI today (fruit modes currently surface only for Telegram/Bree, DAM asset tagging, and
  the gallery NSFW gate).
- **Generation does not apply content mode at all.** `/v1/generate` ignores it; the
  `guidanceBoost` values in config are **not wired anywhere** and are semantically wrong
  — per Todd, mode must never affect guidance.

## Goals

1. Let the user pick a fruit mode in the Generate tab.
2. Send `content_mode` to `/v1/enhance` so the optimizer steers the prompt template.
3. Send `content_mode` to `/v1/generate` so the server can append the mode's
   `negativePromptAdditions` and stamp `content_mode` into the render's metadata/sidecar.

## Non-Goals

- **No guidance/numeric influence.** Mode affects prompt text only. The `guidanceBoost`
  field is explicitly ignored (and should be considered dead config).
- **No preset coupling.** Presets are independent of modes: loading a preset never changes
  the mode, saving a preset never captures it.
- **No global/session-wide mode surface** (toolbar/sidebar) in this iteration — the
  selector is local to the Generate tab.
- Not responsible for the JSON-sidecar-on-server-render work (tracked separately); this
  spec only *contributes* the `content_mode` field that the metadata work will carry.

## Design

### Components

1. **Mode selector (UI).** A 3-way segmented control (🍎 Neutral / 🍌 Banana / 🥑 Avocado)
   in `GenerationView`, placed near the prompt / Enhance row. Bound to a single
   `@State private var contentMode: ContentMode = .neutral`.
2. **Optimizer wiring.** `EngineService.enhancePrompt(_:contentMode:)` gains a
   `content_mode` parameter and includes it in the `/v1/enhance` body. `GenerationView`
   passes the current selection.
3. **Generate wiring.** The generate request includes `content_mode`. Server
   (`/v1/generate` + MCP generate) resolves the mode and:
   - appends `negativePromptAdditions` to the effective negative prompt (no-op while the
     config arrays are empty), and
   - records `content_mode` in the render's `ImageMetadata` (embedded + sidecar).

### Data Flow

```
[Generate tab]
  user picks mode ──► contentMode (@State, default .neutral)
        │
        ├─ Enhance ─► POST /v1/enhance {prompt, content_mode}
        │                └─► optimizer injects promptHint ─► mode-flavored prompt
        │
        └─ Generate ─► POST /v1/generate {…params, content_mode}
                          ├─► negative += mode.negativePromptAdditions
                          └─► ImageMetadata.content_mode = mode (embedded + sidecar)
```

### State & Persistence

- Default 🍎 **Neutral**.
- **Per-session reset:** the selection is held in view state only and resets to Neutral on
  each app launch. Rationale: 🥑 Avocado is explicit; never silently persist it across
  launches. It does hold for the duration of a working session.

### API Changes

- `POST /v1/enhance` — already accepts `content_mode` (server unchanged; client now sends it).
- `POST /v1/generate` — **new** optional `content_mode` (snake_case). Absent ⇒ `neutral`.
  Server appends `negativePromptAdditions` and passes the mode into `ImageMetadata`.
- MCP generate tool — same optional `content_mode` passthrough.

### Metadata Overlap

`content_mode` is the same field the Kira-metadata/sidecar work needs. This spec defines
*where the value originates* (the selector) and *that generate carries it*; the metadata
spec owns embedding it into `ImageMetadata`/sidecar. Coordinate on one `content_mode`
string key so both use identical values (`neutral` | `banana` | `avocado`).

## Testing

- **Unit (client):** `enhancePrompt` body includes `content_mode`; generate request body
  includes `content_mode`; default is `neutral`; selector resets to Neutral on launch.
- **Integration (server):** for `banana`/`avocado`, `/v1/generate` appends the configured
  `negativePromptAdditions`; `content_mode` appears in the produced metadata. For
  `neutral`, negative prompt and behavior are unchanged.
- **Manual:** with LM Studio optimizer up, Enhance on the same prompt yields a
  sensual/explicit rewrite under banana/avocado vs. neutral.

## Work Breakdown

- *Client (small):* `ContentMode` enum + segmented control in `GenerationView`; thread the
  value through `enhancePrompt` and the generate call.
- *Server (small):* accept `content_mode` on `/v1/generate` + MCP; append
  `negativePromptAdditions`; carry `content_mode` into `ImageMetadata`.
