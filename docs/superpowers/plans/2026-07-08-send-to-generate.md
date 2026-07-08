# "Send to Generate" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a gallery image be sent to the Generate tab, reconstructing its full recipe (prompt, negative, seed, steps, guidance, size, model, content mode, LoRA stack) so it can be tweaked and re-rendered with the seed retained.

**Architecture:** Reuse the existing `pendingPreset → applyPreset` reconstruction pipeline. Add a negative-prompt field (so the migrated negative has a home), an `ImageRecipe` reader that parses an image's embedded `UserComment` JSON into a `GenerationPreset` + `ContentMode`, a `pendingContentMode` binding, and a `Send to Generate` trigger mirroring the existing `onUseAsReference` wiring.

**Tech Stack:** Swift 5.9, SwiftUI, ImageIO (CGImageSource); XCTest via `xcodebuild`.

## Global Constraints

- Reconstruct by concrete values (no named-preset matching). Retain the exact seed.
- Model reconstruction is BEST-EFFORT: embedded metadata stores the model display name (basename), not the pool id — activation may fail; fields must still populate, error surfaced via `engine.lastError` (existing `applyPreset` behavior).
- LoRA names in metadata are extension-stripped; rebuild `filename` as `name + ".safetensors"` when it lacks that suffix.
- New fields are additive/optional; no existing behavior changes except the added negative field being sent on generate.
- Build: `xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode`
- Test: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ComfyBoxDesktopTests/<Suite>`

## File Structure

- **Modify** `Sources/ComfyBoxDesktop/PresetManager.swift` — `GenerationPreset.negativePrompt`.
- **Modify** `Sources/ComfyBoxDesktop/ServerPreset.swift` — map `negativePrompt` in `toGenerationPreset`.
- **Modify** `Sources/ComfyBoxDesktop/EngineService.swift` — `GenerationRequest.negativePrompt`; send `negative_prompt` in the generate payload.
- **Modify** `Sources/ComfyBoxDesktop/Views/GenerationView.swift` — negative-prompt state+UI; set it in `applyPreset`; pass it into the generate request; `pendingContentMode` binding + consume.
- **Create** `Sources/ComfyBoxDesktop/ImageRecipe.swift` — the reader.
- **Modify** `Sources/ComfyBoxDesktop/Views/GalleryView.swift` — `onSendToGenerate` closure + menu/detail buttons.
- **Modify** `Sources/ComfyBoxDesktop/ComfyBoxDesktopApp.swift` — `pendingContentMode` state; wire `onSendToGenerate`; pass `pendingContentMode` to `GenerationView`.
- **Create** `Tests/ComfyBoxDesktopTests/ImageRecipeTests.swift`, **Modify** `Tests/ComfyBoxDesktopTests/` (preset mapping test).

---

### Task 1: Negative-prompt field, end to end

**Files:**
- Modify: `Sources/ComfyBoxDesktop/PresetManager.swift` (`GenerationPreset` ~10-56)
- Modify: `Sources/ComfyBoxDesktop/ServerPreset.swift` (`toGenerationPreset` ~148-162)
- Modify: `Sources/ComfyBoxDesktop/EngineService.swift` (`GenerationRequest` ~13; `generate` payload ~348-358)
- Modify: `Sources/ComfyBoxDesktop/Views/GenerationView.swift` (`promptSection` ~434; `applyPreset` ~982; generate request build ~789)
- Create/Modify: `Tests/ComfyBoxDesktopTests/ServerPresetTests.swift`

**Interfaces:**
- Produces: `GenerationPreset.negativePrompt: String?`; `GenerationRequest.negativePrompt: String`; `GenerationView` `@State negativePrompt`.

- [ ] **Step 1: Write the failing test (preset carries negative)**

Create/append `Tests/ComfyBoxDesktopTests/ServerPresetTests.swift`:

```swift
import XCTest
@testable import ComfyBoxDesktop

final class ServerPresetNegativeTests: XCTestCase {
    func testToGenerationPresetCarriesNegative() {
        let sp = ServerPreset(id: "k", name: "Kira", model: "m",
                              negativePrompt: "blurry, watermark")
        XCTAssertEqual(sp.toGenerationPreset().negativePrompt, "blurry, watermark")
    }
}
```
(If `ServerPreset`'s memberwise init differs, construct it with the minimum required args + `negativePrompt:` — check `ServerPreset.init` and adjust the call, keeping the assertion.)

- [ ] **Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ComfyBoxDesktopTests/ServerPresetNegativeTests`
Expected: FAIL — `GenerationPreset` has no member `negativePrompt`.

- [ ] **Step 3: Add `negativePrompt` to `GenerationPreset`**

In `Sources/ComfyBoxDesktop/PresetManager.swift`: add a stored property `public var negativePrompt: String?` (next to `promptTemplate`), an init parameter `negativePrompt: String? = nil` (after `promptTemplate:`), and `self.negativePrompt = negativePrompt` in the init body.

- [ ] **Step 4: Map it in `toGenerationPreset`**

In `Sources/ComfyBoxDesktop/ServerPreset.swift` `toGenerationPreset()`, add the argument (after `promptTemplate:`):
```swift
            negativePrompt: negativePrompt,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ComfyBoxDesktopTests/ServerPresetNegativeTests`
Expected: PASS.

- [ ] **Step 6: Add `negativePrompt` to `GenerationRequest` + send it**

In `Sources/ComfyBoxDesktop/EngineService.swift`: add `public var negativePrompt: String` (default `""`) to `GenerationRequest` (struct ~line 13, plus its init if it has an explicit one). In `generate` (after the `payloadDict` literal, ~line 358), add:
```swift
        if !request.negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            payloadDict["negative_prompt"] = request.negativePrompt
        }
```

- [ ] **Step 7: Add the negative field to the Generate tab UI + wire it**

In `Sources/ComfyBoxDesktop/Views/GenerationView.swift`:
- Add state near the other prompt state: `@State private var negativePrompt: String = ""`.
- In `promptSection`, below the positive prompt `TextEditor`, add:
```swift
            Text("Negative prompt").font(.caption).foregroundStyle(.secondary)
            TextField("things to avoid (optional)", text: $negativePrompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...3)
```
- In `applyPreset`, after the `prompt = preset.promptTemplate` line, add:
```swift
        negativePrompt = preset.negativePrompt ?? ""
```
- At the generate request construction (`let request = GenerationRequest(` ~line 789), add the argument `negativePrompt: negativePrompt`. (Locate with `grep -n "GenerationRequest(" Sources/ComfyBoxDesktop/Views/GenerationView.swift`; add to each construction site.)

- [ ] **Step 8: Build**

Run: `xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode`
Expected: BUILD SUCCEEDED.

- [ ] **Step 9: Commit**

```bash
git add Sources/ComfyBoxDesktop/PresetManager.swift Sources/ComfyBoxDesktop/ServerPreset.swift Sources/ComfyBoxDesktop/EngineService.swift Sources/ComfyBoxDesktop/Views/GenerationView.swift Tests/ComfyBoxDesktopTests/ServerPresetTests.swift
git commit -m "feat(desktop): negative-prompt field in Generate tab + preset/request carry"
```

---

### Task 2: `ImageRecipe` reader

**Files:**
- Create: `Sources/ComfyBoxDesktop/ImageRecipe.swift`
- Create: `Tests/ComfyBoxDesktopTests/ImageRecipeTests.swift`

**Interfaces:**
- Consumes: `GenerationPreset` (incl. `negativePrompt` from Task 1), `PresetLoRA`, `ContentMode`, `DAMAsset`.
- Produces: `struct ImageRecipe { let preset: GenerationPreset; let contentMode: ContentMode? }`; `ImageRecipe.from(params:) -> ImageRecipe?`; `ImageRecipe.read(fromImageAt:fallback:) -> ImageRecipe?`.

- [ ] **Step 1: Write the failing test**

Create `Tests/ComfyBoxDesktopTests/ImageRecipeTests.swift`:

```swift
import XCTest
@testable import ComfyBoxDesktop

final class ImageRecipeTests: XCTestCase {
    func testFromParamsReconstructsRecipe() throws {
        let params: [String: Any] = [
            "prompt": "a cat", "negative_prompt": "blurry",
            "seed": 12345, "steps": 9, "guidance": 3.5,
            "width": 896, "height": 1152,
            "model": "cyberrealisticZImage_v50", "content_mode": "banana",
            "loras": [
                ["name": "Anneliese_Zbase3", "scale": 0.8],
                ["name": "Z-Breast-Slider", "scale": -3],
            ],
        ]
        let r = try XCTUnwrap(ImageRecipe.from(params: params))
        XCTAssertEqual(r.preset.promptTemplate, "a cat")
        XCTAssertEqual(r.preset.negativePrompt, "blurry")
        XCTAssertEqual(r.preset.seed, 12345)
        XCTAssertEqual(r.preset.steps, 9)
        XCTAssertEqual(r.preset.guidance, 3.5)
        XCTAssertEqual(r.preset.width, 896)
        XCTAssertEqual(r.preset.height, 1152)
        XCTAssertEqual(r.preset.modelId, "cyberrealisticZImage_v50")
        XCTAssertEqual(r.contentMode, .banana)
        XCTAssertEqual(r.preset.loras.map(\.filename),
                       ["Anneliese_Zbase3.safetensors", "Z-Breast-Slider.safetensors"])
        XCTAssertEqual(r.preset.loras.map(\.scale), [0.8, -3])
    }

    func testFromParamsEmptyReturnsNil() {
        XCTAssertNil(ImageRecipe.from(params: [:]))
    }

    func testFromParamsNoLorasNoMode() throws {
        let r = try XCTUnwrap(ImageRecipe.from(params: ["prompt": "x", "seed": 7]))
        XCTAssertTrue(r.preset.loras.isEmpty)
        XCTAssertNil(r.contentMode)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ComfyBoxDesktopTests/ImageRecipeTests`
Expected: FAIL — `ImageRecipe` undefined.

- [ ] **Step 3: Implement `ImageRecipe`**

Create `Sources/ComfyBoxDesktop/ImageRecipe.swift`:

```swift
// ImageRecipe.swift — reconstruct a Generate-tab recipe from an image's
// embedded metadata (EXIF:UserComment JSON), for "Send to Generate".
import Foundation
import ImageIO

struct ImageRecipe {
    let preset: GenerationPreset
    let contentMode: ContentMode?

    /// Build a recipe from a params dict (the embedded UserComment JSON, or a
    /// dict derived from a DAMAsset). Returns nil when there's nothing usable.
    static func from(params: [String: Any]) -> ImageRecipe? {
        let prompt = params["prompt"] as? String ?? ""
        guard !prompt.isEmpty || params["seed"] != nil || params["model"] != nil else { return nil }

        let loras: [PresetLoRA] = (params["loras"] as? [[String: Any]] ?? []).compactMap { l in
            guard let name = l["name"] as? String, !name.isEmpty else { return nil }
            let scale = ((l["scale"] as? NSNumber)?.floatValue) ?? 1.0
            let filename = name.hasSuffix(".safetensors") ? name : name + ".safetensors"
            return PresetLoRA(id: name, filename: filename, scale: scale)
        }

        let preset = GenerationPreset(
            id: "from-image",
            name: "From image",
            promptTemplate: prompt,
            negativePrompt: params["negative_prompt"] as? String,
            modelId: params["model"] as? String,
            loras: loras,
            steps: (params["steps"] as? NSNumber)?.intValue ?? 9,
            guidance: (params["guidance"] as? NSNumber)?.floatValue ?? 3.5,
            width: (params["width"] as? NSNumber)?.intValue ?? 1024,
            height: (params["height"] as? NSNumber)?.intValue ?? 1024,
            seed: (params["seed"] as? NSNumber)?.uint64Value
        )
        let mode = (params["content_mode"] as? String).flatMap(ContentMode.init(rawValue:))
        return ImageRecipe(preset: preset, contentMode: mode)
    }

    /// Read an image file's embedded UserComment JSON; fall back to the
    /// DAMAsset's structured fields for images without embedded params.
    static func read(fromImageAt path: String, fallback: DAMAsset?) -> ImageRecipe? {
        var params: [String: Any] = [:]
        if let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
           let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
           let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any],
           let uc = exif[kCGImagePropertyExifUserComment] as? String,
           let d = uc.data(using: .utf8),
           let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            params = j
        }
        if params.isEmpty, let a = fallback {
            params = [
                "prompt": a.prompt as Any, "negative_prompt": a.negativePrompt as Any,
                "seed": a.seed as Any, "steps": a.steps as Any, "guidance": a.guidance as Any,
                "width": a.width as Any, "height": a.height as Any,
                "content_mode": a.contentMode as Any,
            ].compactMapValues { $0 is NSNull ? nil : $0 }
        }
        return from(params: params)
    }
}
```

Note: `GenerationPreset.init`'s parameter order is `id, name, promptTemplate, negativePrompt, modelId, loras, steps, guidance, width, height, sampler, seed, …` after Task 1 inserts `negativePrompt` right after `promptTemplate`. If Task 1 placed it elsewhere, match that order here.

- [ ] **Step 4: Run test to verify it passes**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ComfyBoxDesktopTests/ImageRecipeTests`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add Sources/ComfyBoxDesktop/ImageRecipe.swift Tests/ComfyBoxDesktopTests/ImageRecipeTests.swift
git commit -m "feat(desktop): ImageRecipe reader — image embedded metadata -> GenerationPreset + ContentMode"
```

---

### Task 3: `pendingContentMode` binding + `Send to Generate` trigger

**Files:**
- Modify: `Sources/ComfyBoxDesktop/Views/GenerationView.swift` (bindings ~55-60; `.onAppear`/`.onChange` ~138-142; add consume func)
- Modify: `Sources/ComfyBoxDesktop/Views/GalleryView.swift` (add `onSendToGenerate` param; context menus ~626/~788; detail/lightbox card)
- Modify: `Sources/ComfyBoxDesktop/ComfyBoxDesktopApp.swift` (`pendingContentMode` state ~26-41; `GenerationView` call ~386; `GalleryView` call ~455)

**Interfaces:**
- Consumes: `ImageRecipe` (Task 2), `applyPreset` (existing), `ContentMode`.
- Produces: `GenerationView` `@Binding var pendingContentMode: ContentMode?`; `GalleryView` `onSendToGenerate: ((DAMAsset) -> Void)?`.

- [ ] **Step 1: Add `pendingContentMode` binding + consume in `GenerationView`**

In `Sources/ComfyBoxDesktop/Views/GenerationView.swift`:
- Add near the other pending bindings (~55-60): `@Binding var pendingContentMode: ContentMode?`.
- Add to the `.onAppear` chain (~138): `consumePendingContentMode()`, and a matching `.onChange(of: pendingContentMode) { _, _ in consumePendingContentMode() }` (~142).
- Add the consume function (near `consumePendingPreset`):
```swift
    private func consumePendingContentMode() {
        guard let mode = pendingContentMode else { return }
        pendingContentMode = nil
        contentMode = mode
    }
```

- [ ] **Step 2: Add `onSendToGenerate` to `GalleryView` (param + buttons)**

In `Sources/ComfyBoxDesktop/Views/GalleryView.swift`:
- Add a stored closure property alongside `onUseAsReference`: `var onSendToGenerate: ((DAMAsset) -> Void)? = nil`.
- In BOTH context menus (~626 and ~788), and the lightbox/detail card, add (mirroring the `onUseAsReference` guard):
```swift
                        if onSendToGenerate != nil {
                            Button("Send to Generate") { onSendToGenerate?(asset) }
                        }
```
(For the detail/lightbox card, use the same `Button` in that view's action area; if the card references the asset under a different local name, use that name.)

- [ ] **Step 3: Wire it in `ComfyBoxDesktopApp`**

In `Sources/ComfyBoxDesktop/ComfyBoxDesktopApp.swift`:
- Add state (~26-41): `@State private var pendingContentMode: ContentMode?`.
- Pass to `GenerationView` (~386-390), after `pendingReferenceImage:`: `pendingContentMode: $pendingContentMode,`.
- In the `GalleryView(...)` call (~455), add the closure (after `onUseAsReference:`):
```swift
                    onSendToGenerate: { asset in
                        guard let recipe = ImageRecipe.read(fromImageAt: asset.absolutePath, fallback: asset) else { return }
                        pendingPreset = recipe.preset
                        pendingContentMode = recipe.contentMode
                        selectedTab = .generate
                    },
```

- [ ] **Step 4: Build**

Run: `xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode`
Expected: BUILD SUCCEEDED.

- [ ] **Step 5: Manual smoke (documented, not automated)**

Deploy (`scripts/deploy-desktop.sh`) and: right-click a Kira render → "Send to Generate" → Generate tab shows its prompt, negative, seed, steps, guidance, size, content mode, and the LoRA stack; change the resolution and re-render → same seed. (Recorded in the task report; no automated UI test.)

- [ ] **Step 6: Commit**

```bash
git add Sources/ComfyBoxDesktop/Views/GenerationView.swift Sources/ComfyBoxDesktop/Views/GalleryView.swift Sources/ComfyBoxDesktop/ComfyBoxDesktopApp.swift
git commit -m "feat(desktop): Send to Generate — gallery/detail action reconstructs an image's recipe"
```

---

## Self-Review

- **Spec coverage:** trigger in gallery menu + detail (Task 3); full reconstruct via `applyPreset` + `pendingContentMode` (Tasks 2/3); negative-prompt field added + migrated (Task 1); `ImageRecipe` reader with DAM fallback (Task 2); LoRA filename rebuild + best-effort model (Task 2 + Global Constraints). All spec goals mapped.
- **Placeholder scan:** every step has concrete code/commands. The "match the init order" and "use that name" notes are disambiguation guards, not deferred work.
- **Type consistency:** `GenerationPreset.negativePrompt: String?` (Task 1) is read by `ImageRecipe.from` (Task 2) and set by `applyPreset` (Task 1); `ImageRecipe.read(fromImageAt:fallback:)` (Task 2) is called by the app closure (Task 3); `pendingContentMode: ContentMode?` binding name matches between `GenerationView`, the app state, and `consumePendingContentMode` (Task 3).
