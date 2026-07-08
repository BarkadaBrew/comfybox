# "Send to Generate" — Rerender an Image With Its Exact Settings — Design

**Date:** 2026-07-08
**Status:** Approved (design); pending implementation plan
**Author:** Todd Walderman + Claude

## Summary

Add the ability to move a gallery image into the Generate tab for re-rendering, reconstructing
its full recipe — prompt, negative prompt, **seed**, steps, guidance, size, **model**, content
mode, and the **LoRA stack** — so the user can scale or tweak parameters while retaining the
seed. Reuses the existing `pendingPreset → applyPreset` reconstruction pipeline; adds a recipe
reader, a trigger, and a negative-prompt field.

## Background / Current State

- The Generate tab already reconstructs a full state from a `GenerationPreset` via
  `applyPreset(_:)` (`GenerationView.swift:982`): it sets prompt, steps, guidance, **seed**
  (restores a saved seed), resolution (matched or custom), the **LoRA picker** (`selectedLoras`),
  and **activates the source model** via the model-pool API when it differs.
- A "send X to the Generate tab" pattern already exists: `onUseAsReference` / `pendingReferenceImage`
  closures set from `GalleryView`'s context menu, consumed by `GenerationView`, with
  `selectedTab = .generate` (`ComfyBoxDesktopApp.swift`).
- Images embed their full parameters as JSON in `EXIF:UserComment`, and the desktop already reads
  it: `AssetIngestor` parses that JSON via `CGImageSource` (`AssetIngestor.swift:383-402`). The JSON
  now includes `loras`, `model`, and `content_mode` (recent metadata work).
- Gaps this design fills: `applyPreset`/`GenerationPreset` carry **neither `content_mode` nor a
  negative prompt**, and the Generate tab has **no negative-prompt field** at all.

## Goals

1. A "Send to Generate" action in the gallery **context menu** and the **lightbox/detail** view.
2. Selecting it reconstructs the image's full recipe in the Generate tab: prompt, negative,
   seed, steps, guidance, width×height, model (auto-activated if different), content mode, LoRA
   stack — retaining the seed so a re-render (with unchanged params) reproduces the image.
3. Add a negative-prompt field to the Generate tab so the migrated negative prompt has a home and
   is sent on generate.

## Non-Goals

- No change to how images are rendered/embedded (that metadata already exists).
- No "match a named preset" logic — reconstruction is by concrete values.
- Not migrating params the Generate tab can't represent beyond the negative field being added here.

## Design

### Components

1. **`ImageRecipe` (new, desktop)** — `Sources/ComfyBoxDesktop/ImageRecipe.swift`:
   ```swift
   struct ImageRecipe {
     let preset: GenerationPreset      // prompt, negativePrompt, seed, steps, guidance, W/H, modelId, loras
     let contentMode: ContentMode?
   }
   static func read(fromImageAt path: String, fallback: DAMAsset?) -> ImageRecipe?
   ```
   Reads `EXIF:UserComment` JSON via `CGImageSource` (same as `AssetIngestor`). Maps
   `loras: [{name, scale}]` → `[PresetLoRA(id:, filename: name + ".safetensors", scale:)]`;
   `content_mode` → `ContentMode`; `model` → `modelId`. When the embedded JSON is missing
   (older images), falls back to the `DAMAsset`'s structured fields (prompt/negative/seed/steps/
   guidance/size/contentMode; loras+model unavailable → omitted).

2. **Negative-prompt field (new)** in `GenerationView`:
   - `@State private var negativePrompt: String = ""` + a `TextField`/`TextEditor` in the prompt
     section (below the positive prompt).
   - `EngineService.generate` sends it: add `negative_prompt` to the payload when non-empty.
   - `GenerationPreset` gains `negativePrompt: String?`; `applyPreset` sets the field. (This also
     lets server presets like Kira, which carry a negative, populate it via
     `ServerPreset.toGenerationPreset()`.)

3. **Trigger wiring** — mirror `onUseAsReference`:
   - `GalleryView` gains `onSendToGenerate: ((DAMAsset) -> Void)?`; add a "Send to Generate"
     `Button` to both context menus (grid ~626, detail ~788) and the lightbox/detail card, guarded
     by `onSendToGenerate != nil`.
   - `ComfyBoxDesktopApp` provides the closure: `guard let recipe = ImageRecipe.read(fromImageAt:
     asset.absolutePath, fallback: asset) else { return }; pendingPreset = recipe.preset;
     pendingContentMode = recipe.contentMode; selectedTab = .generate`.

4. **`pendingContentMode` binding (new)** — `@State`/`@Binding var pendingContentMode: ContentMode?`
   on `ComfyBoxDesktopApp`/`GenerationView`, consumed in a `consumePendingContentMode()` that sets
   the `contentMode` state (the one thing `applyPreset` doesn't cover).

### Data Flow

```
Gallery/detail "Send to Generate"
  └─ ImageRecipe.read(file, fallback: asset)  ──► (GenerationPreset, ContentMode?)
        └─ pendingPreset = preset ; pendingContentMode = mode ; selectedTab = .generate
              └─ GenerationView.onAppear/onChange:
                   consumePendingPreset → applyPreset (prompt, negative, seed, steps,
                       guidance, resolution, loras, model-activate)
                   consumePendingContentMode → contentMode = mode
```

### Error Handling

- Unreadable/paramless image with no usable fallback → the action is a no-op (optionally a brief
  toast); never partially corrupt the current Generate state before a successful read.
- Model activation failure (image's model unavailable) → surfaced via the existing
  `applyPreset` model-activation error path (`engine.lastError`); fields still populate.
- LoRA names that don't resolve to a file → dropped by the server at swap time (existing behavior);
  the picker shows what was requested.

## Testing

- **Unit (`ComfyBoxDesktopTests`):** `ImageRecipe.read` given a UserComment JSON string (prompt,
  negative_prompt, seed, steps, guidance, width, height, model, content_mode, loras) → correct
  `GenerationPreset` (incl. negativePrompt + PresetLoRA filenames with `.safetensors`) and
  `ContentMode`; missing `loras`/`content_mode` handled; fallback path uses DAM fields.
- **Unit:** `EngineService.generate` payload includes `negative_prompt` when set, omits when empty.
- **Manual:** right-click a Kira render → Send to Generate → Generate tab shows its prompt,
  negative, seed, steps, guidance, size, model (activates if different), 🍌/mode, and the
  Anneliese/deedee/Z-Breast-Slider stack at their scales; change size and re-render → same seed.

## Work Breakdown (for the plan)

1. Negative-prompt field: `GenerationView` state+UI, `EngineService.generate` payload,
   `GenerationPreset.negativePrompt` + `applyPreset`.
2. `ImageRecipe` reader + unit tests.
3. `pendingContentMode` binding + `consumePendingContentMode`.
4. `onSendToGenerate` wiring: `GalleryView` menu/detail buttons + `ComfyBoxDesktopApp` closure.
