# img2img `content_mode` + `source` Metadata Parity — Design

**Date:** 2026-07-08
**Status:** Approved (design); pending implementation plan
**Author:** Todd Walderman + Claude

## Summary

Reference-image (img2img) renders currently do **not** stamp `content_mode` or `source`
into their embedded metadata, while text-to-image renders do. Close the gap so img2img
output carries the same provenance/mode tags — making the NSFW gate, gallery filtering,
and reproducibility behave identically regardless of how an image was produced.

## Background / Current State

- Text-to-image renders go `GeneratePayload → makePipelineRequest → ZImageGenerationRequest
  → ZImagePipeline.generate`, which saves via `request.embeddedMetadata`
  (`ZImagePipeline.swift:37-41`). That computed property already emits `content_mode` (from
  the fruit-mode feature) and `source` (via `generatedBy:`). Verified live.
- img2img runs through the **same** save path: `generateImg2Img` is an extension on
  `ZImagePipeline` (`ImageToImagePipeline.swift:204`) that converts the request via
  `makeImg2ImgPipelineRequest` into a `ZImageGenerationRequest` (with inpaint data) and calls
  `ZImagePipeline.generate` → `saveImage(metadata: request.embeddedMetadata)`.
- **The gap:** the two hops that build the img2img request drop both fields:
  - `GeneratePayload.makeImg2ImgRequest` (`WarmServer.swift:4089`) builds an `Img2ImgRequest`
    but never passes `self.contentMode` / `self.source` — and `Img2ImgRequest` has no such
    fields.
  - `makeImg2ImgPipelineRequest` (`ImageToImagePipeline.swift:230`) builds the
    `ZImageGenerationRequest` without setting `contentMode` or `source` (they default `nil`),
    so `embeddedMetadata` emits neither.
- Because both fields exist on `ZImageGenerationRequest` already (`contentMode` added by the
  fruit-mode feature; `source` pre-existing), **no pipeline or save-site change is needed** —
  only threading.

## Goals

1. img2img renders stamp `content_mode` into embedded metadata (same values/behavior as txt2img).
2. img2img renders stamp `source` (provenance) into embedded metadata — closing the
   pre-existing gap noted during the fruit-mode review.

## Non-Goals

- No change to `ZImageControlPipeline` (that save at line 1174 serves the distinct ControlNet
  path, not img2img). ControlNet metadata parity is out of scope here.
- No numeric/guidance behavior changes. Purely metadata threading.
- No change to how `content_mode` is chosen or sent — that is the already-shipped fruit-mode
  feature; this only ensures the value reaches img2img metadata.

## Design

Thread the two fields through the img2img request chain so the existing `embeddedMetadata`
save carries them:

```
GeneratePayload{contentMode, source}
  └─ makeImg2ImgRequest ──► Img2ImgRequest{+contentMode, +source}
        └─ makeImg2ImgPipelineRequest ──► ZImageGenerationRequest{contentMode, source}
              └─ ZImagePipeline.generate ──► saveImage(embeddedMetadata)  ✔ already emits both
```

### Changes

1. **`Img2ImgRequest`** (`ImageToImagePipeline.swift`): add `public var contentMode: String?`
   and `public var source: String?`, both defaulting to `nil` in the memberwise init (append
   after `specifiedAs` so existing call sites stay source-compatible).
2. **`GeneratePayload.makeImg2ImgRequest`** (`WarmServer.swift`): pass `contentMode:
   self.contentMode, source: self.source` into the `Img2ImgRequest(...)` it returns.
3. **`makeImg2ImgPipelineRequest`** (`ImageToImagePipeline.swift`): pass `contentMode:
   request.contentMode, source: request.source` into the `ZImageGenerationRequest(...)` it
   returns. Change the method from `private` to `internal` so it is unit-testable.

## Testing

- **Unit (ZImageTests):** build an `Img2ImgRequest` with `contentMode: "banana"`, `source:
  "desktop"`, and a real temp source-image path; call `makeImg2ImgPipelineRequest`; assert the
  returned `ZImageGenerationRequest.embeddedMetadata.parametersJSON` contains
  `"content_mode":"banana"` and `"source":"desktop"`. Also assert nil `contentMode`/`source`
  omit both keys.
- **Build:** Release build proves the `Img2ImgRequest` field additions compile at all call
  sites.
- **Manual (post-deploy):** an img2img render via `/v1/generate` with `image_path` +
  `content_mode:"banana"` → `strings out.png | grep -E 'content_mode|source'`.

## Work Breakdown

Single cohesive change across three files + one test. Contained, low risk (additive optional
fields, no behavior change to numeric params or the save path).
