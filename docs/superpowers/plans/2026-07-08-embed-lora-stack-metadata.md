# Embed Active LoRA Stack in Metadata — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Embed the actually-applied LoRA stack (`[{name, scale}]`) into generated-image metadata across every render path that applies LoRAs.

**Architecture:** Add a `loras` param to the shared `ImageMetadata.generation(...)` writer; each pipeline save site passes its truly-applied stack (`loadedLoRAConfigs`), Chroma passes the server's `activeLoRAs`. Fibo applies no LoRAs and is unchanged.

**Tech Stack:** Swift 5.9, XCTest via `xcodebuild`.

## Global Constraints

- `loras` entry shape: `{"name": <basename with extension stripped>, "scale": <Double>}`. Source name = `LoRAConfiguration.source.displayName` minus its path extension; scale = `LoRAConfiguration.scale`.
- Source from the **applied** stack (`loadedLoRAConfigs` per pipeline; `activeLoRAs` for Chroma) — never the request — so skipped LoRAs are absent.
- Omit the `loras` key entirely when the stack is empty. Do NOT change any existing field (model/guidance/seed/steps/prompt/negative/source/content_mode).
- `LoRAConfiguration` is in the `ZImage` module (same as `ImageIO.swift`) — no new import needed.
- Build: `xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode`
- Test: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/LoRAMetadataTests`

## File Structure

- **Modify** `Sources/ZImage/Util/ImageIO.swift` — add `loras:` to `ImageMetadata.generation`.
- **Modify** `Sources/ZImage/Pipeline/ZImagePipeline.swift` — `embeddedMetadata` becomes a method taking `loras`; save site passes `loadedLoRAConfigs`.
- **Modify** `Sources/ZImage/Pipeline/ZImageControlPipeline.swift` — save site passes `loadedLoRAConfigs`.
- **Modify** `Sources/ZImage/Flux2/Flux2Pipeline.swift` — save site passes `loadedLoRAConfigs`.
- **Modify** `Sources/ZImage/Server/WarmServer.swift` — `renderChroma` gains a `loras` param; call site passes `activeLoRAs`.
- **Create** `Tests/ZImageTests/LoRAMetadataTests.swift`.

---

### Task 1: `loras` in the metadata writer

**Files:**
- Modify: `Sources/ZImage/Util/ImageIO.swift` (`ImageMetadata.generation`, signature ~195-199; body ~200-210)
- Create: `Tests/ZImageTests/LoRAMetadataTests.swift`

**Interfaces:**
- Produces: `ImageMetadata.generation(..., loras: [LoRAConfiguration] = [])`.

- [ ] **Step 1: Write the failing test**

Create `Tests/ZImageTests/LoRAMetadataTests.swift`:

```swift
import XCTest
@testable import ZImage

final class LoRAMetadataTests: XCTestCase {
    private func loras(_ json: String) throws -> [[String: Any]] {
        let obj = try JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
        return (obj["loras"] as? [[String: Any]]) ?? []
    }

    func testGenerationEmbedsLoraStack() throws {
        let m = QwenImageIO.ImageMetadata.generation(
            prompt: "x",
            loras: [
                .local("/models/Anneliese_Zbase3.safetensors", scale: 0.8),
                .local("/models/deedee_amateur_photography_zimage_base_and_turbo_v1.safetensors", scale: 0.4),
                .local("/models/Z-Breast-Slider.safetensors", scale: -3),
            ])
        let arr = try loras(try XCTUnwrap(m.parametersJSON))
        XCTAssertEqual(arr.count, 3)
        XCTAssertEqual(arr[0]["name"] as? String, "Anneliese_Zbase3")          // extension stripped
        XCTAssertEqual(arr[0]["scale"] as? Double, 0.8)
        XCTAssertEqual(arr[2]["name"] as? String, "Z-Breast-Slider")
        XCTAssertEqual(arr[2]["scale"] as? Double, -3)
    }

    func testGenerationOmitsEmptyLoraStack() throws {
        let m = QwenImageIO.ImageMetadata.generation(prompt: "x", loras: [])
        let json = try XCTUnwrap(m.parametersJSON)
        XCTAssertFalse(json.contains("loras"), json)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/LoRAMetadataTests`
Expected: FAIL — `extra argument 'loras' in call`.

- [ ] **Step 3: Add the `loras` parameter + mapping**

In `Sources/ZImage/Util/ImageIO.swift`, extend the `generation` signature — change:
```swift
      height: Int? = nil, model: String? = nil, generatedBy: String? = nil,
      contentMode: String? = nil
    ) -> ImageMetadata {
```
to:
```swift
      height: Int? = nil, model: String? = nil, generatedBy: String? = nil,
      contentMode: String? = nil, loras: [LoRAConfiguration] = []
    ) -> ImageMetadata {
```

Then, immediately after the `content_mode` line
(`if let contentMode, !contentMode.isEmpty { params["content_mode"] = contentMode }`), add:
```swift
      if !loras.isEmpty {
        params["loras"] = loras.map { c -> [String: Any] in
          ["name": (c.source.displayName as NSString).deletingPathExtension,
           "scale": Double(c.scale)]
        }
      }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/LoRAMetadataTests`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add Sources/ZImage/Util/ImageIO.swift Tests/ZImageTests/LoRAMetadataTests.swift
git commit -m "feat(metadata): add loras[] to ImageMetadata.generation writer"
```

---

### Task 2: Thread the applied stack into every save site

**Files:**
- Modify: `Sources/ZImage/Pipeline/ZImagePipeline.swift` (`embeddedMetadata` ~40; save ~962)
- Modify: `Sources/ZImage/Pipeline/ZImageControlPipeline.swift` (save ~1174)
- Modify: `Sources/ZImage/Flux2/Flux2Pipeline.swift` (save ~308)
- Modify: `Sources/ZImage/Server/WarmServer.swift` (`renderChroma` ~3442; call site ~3408; save ~3485)

**Interfaces:**
- Consumes: `ImageMetadata.generation(..., loras:)` (Task 1); `loadedLoRAConfigs: [LoRAConfiguration]` (exists on ZImagePipeline:952, ZImageControlPipeline:843, Flux2Pipeline:143); `WarmServer.activeLoRAs: [LoRAConfiguration]` (2352).

- [ ] **Step 1: ZImagePipeline — `embeddedMetadata` takes loras; save passes the stack**

In `Sources/ZImage/Pipeline/ZImagePipeline.swift`, change the computed property (lines 40-43):
```swift
  public var embeddedMetadata: QwenImageIO.ImageMetadata {
    .generation(prompt: prompt, negativePrompt: negativePrompt, seed: seed,
                steps: steps, guidance: guidanceScale, width: width, height: height,
                model: model, generatedBy: source, contentMode: contentMode)
  }
```
to a method:
```swift
  public func embeddedMetadata(loras: [LoRAConfiguration] = []) -> QwenImageIO.ImageMetadata {
    .generation(prompt: prompt, negativePrompt: negativePrompt, seed: seed,
                steps: steps, guidance: guidanceScale, width: width, height: height,
                model: model, generatedBy: source, contentMode: contentMode, loras: loras)
  }
```
Then update the only call site (line 962) from `metadata: request.embeddedMetadata` to:
```swift
    try QwenImageIO.saveImage(array: decoded, to: request.outputPath, metadata: request.embeddedMetadata(loras: loadedLoRAConfigs))
```
(`grep -n "embeddedMetadata" Sources/` to confirm no other caller — there is exactly one.)

- [ ] **Step 2: ZImageControlPipeline — pass the stack**

In `Sources/ZImage/Pipeline/ZImageControlPipeline.swift` (save ~1174-1176), change:
```swift
      metadata: .generation(prompt: request.prompt, seed: request.seed, steps: request.steps,
        width: request.width, height: request.height))
```
to:
```swift
      metadata: .generation(prompt: request.prompt, seed: request.seed, steps: request.steps,
        width: request.width, height: request.height, loras: loadedLoRAConfigs))
```

- [ ] **Step 3: Flux2Pipeline — pass the stack**

In `Sources/ZImage/Flux2/Flux2Pipeline.swift` (save ~308-310), change the `.generation(...)` call's final line:
```swift
        width: request.width, height: request.height, contentMode: request.contentMode))
```
to:
```swift
        width: request.width, height: request.height, contentMode: request.contentMode, loras: loadedLoRAConfigs))
```

- [ ] **Step 4: WarmServer.renderChroma — add param + pass activeLoRAs**

In `Sources/ZImage/Server/WarmServer.swift`:

(a) Add a parameter to the `renderChroma` signature (after `outputURL: URL`):
```swift
  private static func renderChroma(
    pipeline: ChromaPipeline,
    tokenizer: ChromaTokenizer,
    payload: GeneratePayload,
    outputURL: URL,
    loras: [LoRAConfiguration]
  ) async throws {
```

(b) At the call site (~line 3408), add the argument after `outputURL: outputURL`:
```swift
      try await Self.renderChroma(
        pipeline: pipeline,
        tokenizer: tokenizer,
        payload: payload,
        outputURL: outputURL,
        loras: activeLoRAs
      )
```

(c) In the save call (~3485), add `loras: loras` to `.generation(...)`:
```swift
      metadata: .generation(prompt: payload.prompt, negativePrompt: payload.negativePrompt,
        seed: seed, steps: steps, guidance: guidance, width: width, height: height,
        generatedBy: payload.source, contentMode: payload.contentMode, loras: loras))
```

(FiboPipeline is intentionally unchanged — it applies no LoRAs, so it correctly emits no `loras`.)

- [ ] **Step 5: Build to verify all save sites compile**

Run: `xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode`
Expected: BUILD SUCCEEDED.

- [ ] **Step 6: Re-run the writer tests (unchanged, must still pass)**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/LoRAMetadataTests`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add Sources/ZImage/Pipeline/ZImagePipeline.swift Sources/ZImage/Pipeline/ZImageControlPipeline.swift Sources/ZImage/Flux2/Flux2Pipeline.swift Sources/ZImage/Server/WarmServer.swift
git commit -m "feat(metadata): stamp applied LoRA stack for flux1/control/flux2/chroma renders"
```

---

## Self-Review

- **Spec coverage:** writer `loras[]` (Task 1); flux1/Kira + img2img via ZImagePipeline, ControlNet/inpaint via ZImageControlPipeline, Flux2, Chroma via renderChroma (Task 2); Fibo intentionally unchanged (documented). Acceptance #1/#2/#3 covered; Chroma requested-stack caveat documented in spec.
- **Placeholder scan:** every step has concrete before/after code. No TBD/TODO.
- **Type consistency:** `loras: [LoRAConfiguration]` matches `loadedLoRAConfigs`/`activeLoRAs` (both `[LoRAConfiguration]`) at all four save sites; `ImageMetadata.generation(loras:)` param name used identically in Tasks 1 and 2; `embeddedMetadata()` method form matches its single updated call site.
