# img2img content_mode + source Metadata Parity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thread `content_mode` and `source` through the img2img request chain so reference-image renders stamp them into embedded metadata, matching text-to-image.

**Architecture:** img2img already saves via `ZImageGenerationRequest.embeddedMetadata` (a `ZImagePipeline` extension), which already emits both fields for txt2img. The only gap is that the two request-builder hops (`GeneratePayload.makeImg2ImgRequest` → `Img2ImgRequest` → `makeImg2ImgPipelineRequest` → `ZImageGenerationRequest`) never set them. Add two optional fields to `Img2ImgRequest` and thread them.

**Tech Stack:** Swift 5.9, XCTest via `xcodebuild`.

## Global Constraints

- Additive, optional (`String?`, default `nil`) fields only — backward compatible, no existing call site breaks.
- Values are the same strings used everywhere else: `content_mode` ∈ {`neutral`,`banana`,`avocado`}; `source` is the submitting app string (e.g. `desktop`).
- No numeric/guidance/save-path behavior changes. Do NOT touch `ZImageControlPipeline` (that's the separate ControlNet path, out of scope).
- Build: `xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode`
- Test: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/Img2ImgMetadataTests`

## File Structure

- **Modify** `Sources/ZImage/Pipeline/ImageToImagePipeline.swift` — add `contentMode`/`source` to `Img2ImgRequest`; thread them in `makeImg2ImgPipelineRequest`; make that method `internal` (was `private`) for testability.
- **Modify** `Sources/ZImage/Server/WarmServer.swift` — pass `contentMode`/`source` into the `Img2ImgRequest(...)` built by `GeneratePayload.makeImg2ImgRequest`.
- **Create** `Tests/ZImageTests/Img2ImgMetadataTests.swift`.

---

### Task 1: Thread content_mode + source through img2img

**Files:**
- Modify: `Sources/ZImage/Pipeline/ImageToImagePipeline.swift` (`Img2ImgRequest` struct + init ~lines 40-115; `makeImg2ImgPipelineRequest` ~line 230; the `ZImageGenerationRequest(...)` return ~lines 281-315)
- Modify: `Sources/ZImage/Server/WarmServer.swift` (`makeImg2ImgRequest` `Img2ImgRequest(...)` return, ~lines 4140-4162)
- Create: `Tests/ZImageTests/Img2ImgMetadataTests.swift`

**Interfaces:**
- Consumes: `ZImageGenerationRequest.init(..., source: String? = nil, contentMode: String? = nil, ...)` (existing); `QwenImageIO.ImageMetadata.generation(..., contentMode:)` (existing); `Img2ImgRequest.embeddedMetadata` is not a thing — assert on the produced `ZImageGenerationRequest.embeddedMetadata`.
- Produces: `Img2ImgRequest.contentMode: String?`, `Img2ImgRequest.source: String?`; `internal` visibility on `makeImg2ImgPipelineRequest`.

- [ ] **Step 1: Write the failing test**

Create `Tests/ZImageTests/Img2ImgMetadataTests.swift`:

```swift
import XCTest
@testable import ZImage

final class Img2ImgMetadataTests: XCTestCase {
    /// Write a tiny valid PNG to a temp path so makeImg2ImgPipelineRequest's
    /// file-existence + dimension checks pass.
    private func makeTempPNG() throws -> String {
        let path = NSTemporaryDirectory() + "img2img-test-\(UUID().uuidString).png"
        // 1x1 white PNG.
        let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        try Data(base64Encoded: b64)!.write(to: URL(fileURLWithPath: path))
        return path
    }

    func testImg2ImgPipelineRequestCarriesContentModeAndSource() throws {
        let src = try makeTempPNG()
        defer { try? FileManager.default.removeItem(atPath: src) }
        let req = Img2ImgRequest(
            prompt: "a cat", sourceImagePath: src,
            contentMode: "banana", source: "desktop")
        let pipeline = try ZImagePipeline.makeImg2ImgPipelineRequestForTesting(req)
        let json = try XCTUnwrap(pipeline.embeddedMetadata.parametersJSON)
        XCTAssertTrue(json.contains("\"content_mode\":\"banana\""), json)
        XCTAssertTrue(json.contains("\"source\":\"desktop\""), json)
    }

    func testImg2ImgPipelineRequestOmitsNilFields() throws {
        let src = try makeTempPNG()
        defer { try? FileManager.default.removeItem(atPath: src) }
        let req = Img2ImgRequest(prompt: "a cat", sourceImagePath: src)
        let pipeline = try ZImagePipeline.makeImg2ImgPipelineRequestForTesting(req)
        let json = try XCTUnwrap(pipeline.embeddedMetadata.parametersJSON)
        XCTAssertFalse(json.contains("content_mode"), json)
        XCTAssertFalse(json.contains("\"source\""), json)
    }
}
```

Note: `makeImg2ImgPipelineRequest` is an instance method on the `ZImagePipeline` extension and needs a `ZImagePipeline` instance, which is heavy to construct. To keep the test pure, Step 4 adds a tiny static test shim `makeImg2ImgPipelineRequestForTesting` that calls the same conversion logic without a live pipeline. If the conversion has no instance-state dependency (it does not — it only reads `request` + FileManager), refactor the body into a `static` function and have both the instance method and the shim call it.

- [ ] **Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/Img2ImgMetadataTests`
Expected: FAIL — `Img2ImgRequest` has no `contentMode`/`source` args; `makeImg2ImgPipelineRequestForTesting` undefined.

- [ ] **Step 3: Add fields to `Img2ImgRequest`**

In `Sources/ZImage/Pipeline/ImageToImagePipeline.swift`, add two stored properties near `specifiedAs`:

```swift
  /// Fruit mode (neutral|banana|avocado) — stamped into embedded metadata.
  public var contentMode: String?

  /// Submitting app/persona (desktop/bree/api…) — stamped as metadata provenance.
  public var source: String?
```

Add matching init parameters (append after `specifiedAs: Img2ImgSpecifier = .strength`):

```swift
    contentMode: String? = nil,
    source: String? = nil
```

And in the init body (after `self.specifiedAs = specifiedAs`):

```swift
    self.contentMode = contentMode
    self.source = source
```

- [ ] **Step 4: Thread into the pipeline request + extract a testable shim**

In the same file, in `makeImg2ImgPipelineRequest` (currently `private`), change `private` to `internal`, and in its returned `ZImageGenerationRequest(...)` add these two arguments (place them next to `denoise: request.denoise`):

```swift
      source: request.source,
      contentMode: request.contentMode,
```

Then add a static test shim at the end of the `extension ZImagePipeline` block (so tests can build the pipeline request without a live pipeline instance — the conversion reads only `request` and `FileManager`):

```swift
  /// Test-only: exposes makeImg2ImgPipelineRequest's pure conversion without a
  /// live pipeline instance. Not used in production code paths.
  static func makeImg2ImgPipelineRequestForTesting(_ request: Img2ImgRequest) throws -> ZImageGenerationRequest {
    // makeImg2ImgPipelineRequest is instance-scoped only by convention; its body
    // reads no instance state. If Swift requires an instance, construct a minimal
    // one, or lift the body into a `static func` both callers share.
    fatalError("Implement per Step 4 note")
  }
```

**Implementer note:** the cleanest form is to lift the body of `makeImg2ImgPipelineRequest` into a `static func makeImg2ImgPipelineRequest(_ request: Img2ImgRequest) throws -> ZImageGenerationRequest`, have the existing instance method forward to it (`try Self.makeImg2ImgPipelineRequest(request)`), and have the test shim call the static one (delete the `fatalError` stub and make `makeImg2ImgPipelineRequestForTesting` just call the static func, or point the test directly at the static func). Choose whichever keeps production behavior byte-identical; the test only needs a callable entry that returns the converted `ZImageGenerationRequest`.

- [ ] **Step 5: Thread from the server payload**

In `Sources/ZImage/Server/WarmServer.swift`, in `GeneratePayload.makeImg2ImgRequest`, add two arguments to the returned `Img2ImgRequest(...)` (next to `specifiedAs: specifiedAs`):

```swift
      contentMode: contentMode,
      source: source
```

(`contentMode` and `source` are stored properties on `GeneratePayload`.)

- [ ] **Step 6: Run test to verify it passes**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/Img2ImgMetadataTests`
Expected: PASS (2 tests).

- [ ] **Step 7: Release build (proves all `Img2ImgRequest` call sites compile)**

Run: `xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode`
Expected: BUILD SUCCEEDED.

- [ ] **Step 8: Commit**

```bash
git add Sources/ZImage/Pipeline/ImageToImagePipeline.swift Sources/ZImage/Server/WarmServer.swift Tests/ZImageTests/Img2ImgMetadataTests.swift
git commit -m "feat(img2img): stamp content_mode + source into reference-image render metadata"
```

---

## Self-Review

- **Spec coverage:** Goal 1 (content_mode in img2img metadata) + Goal 2 (source in img2img metadata) — both covered by Task 1 (fields + threading + test). Non-goal (no ControlPipeline change) respected — plan touches only `Img2ImgRequest`, `makeImg2ImgPipelineRequest`, `makeImg2ImgRequest`.
- **Placeholder scan:** the `fatalError` stub in Step 4 is explicitly a scaffold the same step tells the implementer to replace (with the static-func lift). No other placeholders.
- **Type consistency:** `Img2ImgRequest.contentMode/source: String?` match the `ZImageGenerationRequest.init(source:contentMode:)` params they feed; `GeneratePayload.contentMode/source` (both `String?`) match the `Img2ImgRequest` args they feed.
