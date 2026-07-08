# Embed Active LoRA Stack in Generated-Image Metadata — Design

**Date:** 2026-07-08
**Status:** Approved (design); pending implementation plan
**Author:** Todd Walderman + Claude

## Summary

Generated PNGs embed model/guidance/seed/steps/prompt/negative into Finder-visible metadata
(EXIF `UserComment` JSON), but **not the LoRA stack** that was applied. Add a `loras` array
sourced from each pipeline's *actually-applied* stack, so a Kira render records
`Anneliese_Zbase3 @0.8 + deedee… @0.4 + Z-Breast-Slider @-3`.

## Background / Correction to the Handoff

- The handoff pointed at `SidecarService.swift` (desktop exiftool re-embed) and
  `EngineService.swift` (desktop request assembly). **Those are the wrong writer for Kira
  renders.** A Kira render is a server render (`zimage-*.png`); it embeds metadata *natively*
  via `QwenImageIO.ImageMetadata.generation(...)` (`Sources/ZImage/Util/ImageIO.swift`),
  written from each pipeline's save site. That is where the `loras` array must be added.
- **Root cause confirmed:** LoRAs are applied via a separate `/v1/lora/swap` before
  `/v1/generate`, so the request carries none. But each pipeline tracks the applied stack in
  `currentLoRAs` and exposes it as `loadedLoRAConfigs: [LoRAConfiguration]` — the *truly-applied*
  set (skips already excluded). Sourcing from it satisfies "skipped LoRAs must be absent" for free.
- `LoRAConfiguration` has `.source.displayName` (e.g. `Anneliese_Zbase3.safetensors`) and
  `.scale: Float`. The `loras` entry uses the display name **with the extension stripped**
  (`Anneliese_Zbase3`) to match the requested output.

## Goals

1. Every render path that applies LoRAs stamps a `loras` array into embedded metadata:
   `"loras": [{"name": "<basename-no-ext>", "scale": <Float>}, …]`.
2. The array reflects the *applied* stack (skipped/unresolvable LoRAs absent).
3. Existing fields (model/guidance/seed/steps/prompt/negative/source/content_mode) unchanged;
   `loras` is omitted entirely when the stack is empty.

## Non-Goals

- No change to the desktop `SidecarService` / gallery re-embed path (separate concern, #1).
- No change to how LoRAs are applied or swapped.
- Not adding Chroma-internal applied-stack tracking (see Family Coverage — Chroma sourced from
  the server's `activeLoRAs` instead, with a documented caveat).

## Family Coverage

| Family | Save site | LoRA source | Notes |
|---|---|---|---|
| `.flux1` (Kira, txt2img + img2img via ZImagePipeline) | `ZImagePipeline.swift:962` (`embeddedMetadata`) | `self.loadedLoRAConfigs` | Primary/acceptance path. Skip-resilient. |
| img2img / inpaint / ControlNet | `ZImageControlPipeline.swift:1175` | `self.loadedLoRAConfigs` | Skip-resilient. |
| Flux2 | `Flux2Pipeline.swift:309` | `self.loadedLoRAConfigs` | Skip-resilient. |
| Fibo | `FiboPipeline.swift:197` | — | Fibo applies no LoRAs; correctly emits no `loras`. No change. |
| Chroma | `WarmServer.swift:3485` (`renderChroma`, static) | `self.activeLoRAs` passed in | ChromaPipeline exposes no applied-stack accessor; server `activeLoRAs` reflects the *requested* stack. Caveat: not strictly skip-resilient for Chroma. |

## Design

### 1. Metadata writer (`ImageIO.swift`)

`QwenImageIO.ImageMetadata.generation(...)` gains `loras: [LoRAConfiguration] = []`. When
non-empty it sets:

```swift
params["loras"] = loras.map { c in
  ["name": (c.source.displayName as NSString).deletingPathExtension,
   "scale": Double(c.scale)]
}
```

`LoRAConfiguration` is in the same `ZImage` module, so no new import/coupling across modules.
Empty ⇒ key omitted (mirrors the existing `content_mode`/`source` guards).

### 2. Save sites

- **ZImagePipeline:** change `embeddedMetadata` from a computed `var` to a method
  `embeddedMetadata(loras: [LoRAConfiguration] = []) -> QwenImageIO.ImageMetadata` (only caller
  is line 962). At the save site pass `loras: loadedLoRAConfigs`.
- **ZImageControlPipeline:1175, Flux2Pipeline:309:** add `loras: loadedLoRAConfigs` to the
  `.generation(...)` call.
- **WarmServer.renderChroma:** add a `loras: [LoRAConfiguration]` parameter; the call site
  (`WarmServer.swift:3408`, instance context) passes `self.activeLoRAs`; thread `loras:` into the
  `.generation(...)` call at line 3485.
- **FiboPipeline:** no change.

## Testing

- **Unit (ZImageTests):** `ImageMetadata.generation(prompt:…, loras: [.local("…/Anneliese_Zbase3.safetensors", scale: 0.8), .local("…/deedee….safetensors", scale: 0.4), .local("…/Z-Breast-Slider.safetensors", scale: -3)])` → `parametersJSON` contains a `loras` array with names extension-stripped and correct scales; empty `loras` ⇒ no `loras` key; existing fields unaffected.
- **Build:** Release build (proves all save-site + `embeddedMetadata`-signature changes compile).
- **Manual (post-deploy):** a Kira-preset render, then
  `exiftool -j -UserComment -XMP:all <newest .png>` → confirm the `loras` array with the three
  Kira LoRAs and their scales.

## Acceptance Criteria (from handoff)

1. Kira render PNG's embedded JSON contains a `loras` array with each applied LoRA's name+scale. ✅ (`.flux1` via `loadedLoRAConfigs`)
2. Array reflects what was truly loaded (skips absent). ✅ for flux1/control/Flux2 (`loadedLoRAConfigs`); Chroma is requested-stack (documented caveat).
3. Existing fields unchanged; `loras` omitted when empty. ✅
