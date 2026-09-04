# Desktop Image Editor — Design

**Date**: 2026-09-04
**Status**: Approved by Todd (chat), ready for planning
**Scope**: ComfyBoxDesktop only. No engine, server, or MCP surface changes.

## Purpose

A native, lightweight image editor inside the ComfyBox desktop app for
tone and color correction, cropping, local (masked) adjustments, and
background removal. It replaces the "touch up in another app" step for
engine output and gallery assets. It is derived from a feature survey of
opengpex (a GPL-3 browser editor) but shares no code with it; every
capability maps onto Core Image, Vision, and ImageIO, which macOS ships
natively and which render on Metal.

Non-goals for v1: layer stacks with blend modes, a plugin system, RAW or
HEIC decode, text or shape tools, cloud sync. These are deliberately out.

## Decisions (settled with Todd)

| Decision | Choice |
|---|---|
| Entry points | Both: an **Edit** tab in the sidebar with an Open button, and an **Edit** action in the asset detail view and gallery context menu. |
| Save model | Original untouched. Save writes a new flattened PNG as a **derived asset** plus a **reopenable recipe**. |
| Recipe storage | Adjacent JSON **sidecar** (`<output>.json`), the desktop convention the ingestor already reads. No DAM schema change. |
| v1 tools | Global tone and color; crop, straighten, rotate, flip; **one** brush-masked local adjustment layer; Vision background removal / subject lift. |
| Input formats | PNG, JPEG, TIFF. |
| Rendering | Core Image filter chain, one renderer for preview and export. No Metal view. |

## Architecture

```
EditRecipe (value)  ──▶  EditRenderer (pure)  ──▶  CIImage
      ▲                        ▲
      │ mutate                 │ source CIImage, mask CIImage
EditSession (@Observable, MainActor)
      │ preview CGImage (debounced, background render)
      ▼
EditView / EditTab (SwiftUI)  ──save──▶  EditExporter ──▶ PNG + sidecar ──▶ AssetIngestor
```

All new files live under `Sources/ComfyBoxDesktop/Edit/` except the
views (`Sources/ComfyBoxDesktop/Views/EditView.swift`) and the shared
mask canvas (`Sources/ComfyBoxDesktop/Views/MaskCanvas.swift`).

### EditRecipe

`public struct EditRecipe: Codable, Equatable, Sendable`. All numeric
fields have a neutral default such that `EditRecipe()` renders the
source unchanged. `version: Int = 1`.

- **Geometry** (`EditGeometry`): `crop: CGRect?` in normalized source
  coordinates (origin top-left, 0…1), `straightenDegrees: Double` in
  −45…45, `quarterTurns: Int` 0…3 clockwise, `flipH: Bool`, `flipV: Bool`.
- **Adjustments** (`EditAdjustments`), one struct reused for global and
  local: `exposure` (EV, −5…5), `contrast` (−1…1 mapped to CIColorControls 0.5…1.5), `highlights` (−1…1), `shadows` (−1…1), `whites` (−1…1),
  `blacks` (−1…1), `temperature` (−1…1 mapped to ±3000 K around 6500),
  `tint` (−1…1 mapped to ±150), `vibrance` (−1…1), `saturation` (−1…1
  mapped to 0…2), `sharpen` (0…1), `noiseReduction` (0…1),
  `vignette` (0…1), `curves: ToneCurves`.
- **ToneCurves**: four `[CurvePoint]` arrays (rgb, r, g, b), each
  `CurvePoint(x: Double, y: Double)` in 0…1, sorted by x, endpoints
  implied at (0,0) and (1,1) when absent. Rendered via `CIToneCurve`
  (five-point) after resampling: the renderer samples the monotone
  cubic through the control points at x = 0, 0.25, 0.5, 0.75, 1. Empty
  array = identity.
- **Local layer** (`EditLocalLayer?`): `mask: MaskStrokes` (the shared
  stroke model, see MaskCanvas), `feather: Double` 0…1 (Gaussian blur
  radius as a fraction of the shorter side, max 5 %), `adjustments:
  EditAdjustments` restricted to exposure, contrast, highlights,
  shadows, temperature, tint, saturation, sharpen. Other fields are
  ignored for local layers.
- **Subject** (`EditSubject`): `removeBackground: Bool`, `invert: Bool`.

`EditRecipe.isIdentity` is true when every field equals its default.

### MaskStrokes and MaskCanvas (extracted from Inpaint)

`InpaintView` currently owns a private `Stroke` model, a SwiftUI
`Canvas` overlay that draws strokes in the fitted image rect, a drag
gesture that records normalized points, and a rasterizer that renders
the strokes to a full-resolution black/white PNG. That is exactly the
mask tool the editor needs, so it moves to shared code:

- `MaskStrokes: Codable, Equatable, Sendable` — `[MaskStroke]`, each
  with normalized `points: [CGPoint]`, `size: Double` (fraction of image
  width), `erase: Bool`. Pure helpers: `isEmpty`, `append`, `undoLast`,
  `clear`.
- `MaskRasterizer.render(_ strokes: MaskStrokes, size: CGSize) -> CGImage?`
  — white on black, round caps, erase strokes paint black, Y flipped to
  match the bitmap origin. Returns a `CGImage` so both the PNG path
  (Inpaint) and the CIImage path (editor) consume it.
- `MaskCanvas: View` — takes `imageSize`, `@Binding strokes`, `brush`,
  `erase`, `showOverlay`; renders the tinted overlay and the gesture
  layer inside the fitted rect. `fitRect` moves alongside it as a static
  helper.

`InpaintView` switches to these three with no behaviour change; its
`maskPNG()` becomes `MaskRasterizer.render(...)` followed by PNG
encoding. This is the one refactor of existing code in this design.

### EditRenderer

`enum EditRenderer` with one entry point:

```swift
static func render(source: CIImage, recipe: EditRecipe,
                   subjectMask: CIImage?) -> CIImage
```

Pure, no I/O, no caching, no main-actor requirement. Order of
operations, fixed and documented in code:

1. **Geometry**: quarter-turn rotation, flips, straighten (rotate about
   center, then crop to the largest axis-aligned rect that stays inside
   the rotated image so no transparent corners appear), then crop.
2. **Global adjustments** in this order: exposure (`CIExposureAdjust`),
   white balance (`CITemperatureAndTint`), highlights and shadows
   (`CIHighlightShadowAdjust`), whites and blacks (implemented as a
   `CIToneCurve` that moves the (0.75→whites) and (0.25→blacks) points),
   contrast and saturation (`CIColorControls`), vibrance (`CIVibrance`),
   curves (`CIToneCurve` per channel, RGB curve first then per-channel
   via `CIColorMatrix` split/merge), sharpen (`CISharpenLuminance`),
   noise reduction (`CINoiseReduction`), vignette (`CIVignette`).
3. **Local layer**: render the same adjustment chain on the step-2
   output with the local adjustments, feather the rasterized mask with
   `CIGaussianBlur`, then `CIBlendWithMask` (adjusted over unadjusted).
   The mask is rasterized at the *post-geometry* size because the user
   paints on the cropped view.
4. **Subject**: when `removeBackground` is set and `subjectMask` is
   present, multiply alpha by the (optionally inverted) mask via
   `CIBlendWithMask` against a transparent background. When set but no
   mask is available the step is skipped and the session surfaces a
   warning.

Every mapping from recipe units to filter parameters lives in a small
`static func` so tests can pin it (for example
`contrastParameter(for: -1) == 0.5`).

### SubjectMasker

`actor SubjectMasker` wrapping
`VNGenerateForegroundInstanceMaskRequest` (macOS 14). `func mask(for
source: CGImage) async throws -> CIImage` returns a single-channel mask
at source resolution combining all detected instances. Result cached
per source path for the session lifetime. Errors are typed
(`SubjectMaskError.noSubject`, `.visionFailed(underlying)`) so the UI
can say "no subject found" versus "Vision failed".

### EditSession

`@MainActor @Observable final class EditSession`:

- Inputs: `sourcePath`, loaded `sourceImage: CGImage`, optional
  `sourceAsset: DAMAsset`, `recipe`.
- State: `preview: CGImage?`, `isRendering`, `showOriginal` (before/after
  toggle), `undoStack`/`redoStack` of recipes (capped at 100),
  `warning: String?`, `isDirty`.
- Preview render: on every recipe change, coalesce with a 40 ms debounce,
  then render on a detached task with a `CIContext` owned by the
  session (Metal-backed, `.workingColorSpace` sRGB). The source is
  downscaled once to fit 2048 px on the long side for preview; the
  local mask and subject mask are rasterized at that preview size.
  A newer request cancels an in-flight one.
- `commit()` pushes the previous recipe onto the undo stack; sliders
  call `set(_:)` for live updates and `commit()` on gesture end so one
  drag is one undo step.
- `requestSubjectMask()` runs `SubjectMasker` on the full-resolution
  source, stores the mask, and re-renders.
- `reset()` restores `EditRecipe()`.

### EditExporter

`enum EditExporter` with:

```swift
static func export(session: EditSession, outputDirectory: String,
                   ingestor: AssetIngestor?) async throws -> String
```

1. Render at full resolution with a fresh `CIContext` and write PNG to
   `<outputDirectory>/edit-<unixSeconds>.png` (with `-2`, `-3` suffix on
   collision). Transparent output when background removal is active.
2. Write `<same base>.json` with:
   - Every generation field the ingestor reads (`prompt`,
     `negative_prompt`, `seed`, `steps`, `guidance`, `model_family`,
     `content_mode`, `character_name`) copied from the source asset when
     present, so search and the content gate keep working on the derived
     file.
   - `"source": "desktop-edit"`.
   - `"edit": { "version": 1, "source_path": …, "source_asset_id": …,
     "recipe": <EditRecipe JSON>, "editor": "ComfyBoxDesktop", "created_at": ISO-8601 }`.
   - When the source itself is a derived edit, `source_path` and
     `source_asset_id` point at the **root original**, so re-edits stack
     on original pixels.
3. Ingest via `ingestor.ingestFile(at:)`. If step 1 fails nothing is
   written; if step 2 fails the PNG is removed; if step 3 fails the
   files remain and the error is surfaced (the poller will pick the
   file up).

`EditSidecar.read(forImageAt:) -> EditSidecar?` parses the `edit` block
back for reopen and for the detail view.

### Views

- **EditView** (`Views/EditView.swift`): two-pane layout matching
  Inpaint. Left: the image canvas, aspect-fit, with `MaskCanvas` overlaid
  when the Local group is active, a crop rectangle overlay with handles
  when the Crop group is active, and a press-and-hold Before button.
  Right: a scrollable panel of collapsible groups — Light (exposure,
  contrast, highlights, shadows, whites, blacks), Color (temperature,
  tint, vibrance, saturation), Curves (channel picker, draggable
  points on a 1:1 graph, double-click to add, drag off to remove),
  Detail (sharpen, noise reduction, vignette), Crop and Rotate (aspect
  preset menu: free, 1:1, 4:5, 3:2, 16:9, original; straighten slider;
  rotate and flip buttons), Local (brush size, erase toggle, feather,
  the restricted adjustment sliders, clear mask), Subject (Find
  subject, Remove background, Invert). Toolbar: Undo, Redo, Reset,
  Before, Save, Save and Open in Inpaint.
- **EditTab**: wraps EditView with an Open button (`NSOpenPanel`,
  PNG/JPEG/TIFF) and consumes a `@Binding pendingEditImage: String?`
  the same way Inpaint consumes `pendingInpaintImage`.
- **Asset detail**: an Edit action beside Send to Generate; when the
  sidecar has an `edit` block, a line "Edited from <source filename>"
  with a button that selects the source asset. Gallery context menu gets
  the same Edit action; the `onEdit` callback is threaded through
  `GalleryView` like `onInpaint`.
- **Send to Inpaint** from the editor: exports first (so Inpaint gets a
  real file), then sets `pendingInpaintImage` and, when a local mask
  exists, `pendingInpaintMask: MaskStrokes?`, a new binding that
  `InpaintView` adopts as its initial strokes. Small, additive change.

### Reopen

Opening an asset whose sidecar has an `edit` block loads the **root
source** pixels and the stored recipe. If the root file is missing, the
editor opens the derived pixels with an empty recipe and a warning. If
`edit.version` is greater than the app supports, pixels load, the
recipe is dropped, and a warning says so.

## Error handling

| Situation | Behaviour |
|---|---|
| Source unreadable | Editor shows a message, controls disabled. |
| Vision finds no subject | Subject group shows "No subject found"; flag stays off. |
| Vision unavailable / throws | Same group shows the error; nothing else affected. |
| Preview render throws | Previous preview stays, `warning` set, next change retries. |
| Export write failure | No partial files (see EditExporter); error shown in toolbar. |
| Recipe version too new | Load pixels, drop recipe, warn. |

## Testing

Unit tests in `Tests/ComfyBoxDesktopTests` using the Testing framework
like `CanvasTests`. No model weights, no Vision, no windows.

- `EditRecipeTests`: JSON round-trip, `isIdentity`, defaults, curve
  point sorting and endpoint insertion.
- `EditRendererTests`: on synthetic 64×64 gradients rendered through a
  software `CIContext`: identity recipe is pixel-identical; exposure +1
  brightens the mid-grey pixel; flipH mirrors; quarter-turn swaps
  dimensions; crop yields the expected size; straighten yields no
  transparent corner pixels; a local layer with a half-image mask
  changes only the masked half; every parameter mapping function pins
  its endpoints.
- `MaskRasterizerTests`: single stroke paints white at its points and
  black elsewhere; erase stroke clears; Y orientation is correct (a
  stroke at normalized y = 0.1 lands near the top row).
- `EditExporterTests`: writes PNG and sidecar into a temp directory;
  sidecar carries copied generation fields, `source: desktop-edit`, and a
  recipe that decodes equal to the input; re-export of a derived asset
  points at the root source; collision suffixing.
- `EditSidecarTests`: parse, missing block, newer version.
- Inpaint's existing behaviour is covered by asserting
  `MaskRasterizer` output for the same strokes matches a PNG produced by
  the pre-refactor algorithm on one fixture.

Live checks Todd runs: open a 4K render, drag sliders for smoothness,
paint a local mask, Find subject on a portrait, Save, confirm the
gallery shows the derived asset with Edited-from, reopen and confirm the
recipe restores, Send to Inpaint and confirm the mask arrives.

## Files

New: `Edit/EditRecipe.swift`, `Edit/EditRenderer.swift`,
`Edit/SubjectMasker.swift`, `Edit/EditSession.swift`,
`Edit/EditExporter.swift`, `Edit/EditSidecar.swift`,
`Views/MaskCanvas.swift`, `Views/EditView.swift`, plus the five test
files above.

Modified: `Views/InpaintView.swift` (use MaskCanvas), `ComfyBoxDesktopApp.swift`
(tab, pending bindings, callbacks), `Views/AssetDetailView.swift` (Edit
action, Edited-from), `Views/GalleryView.swift` (onEdit passthrough).
