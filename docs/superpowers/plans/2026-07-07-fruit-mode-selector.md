# Fruit-Mode Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 🍎/🍌/🥑 content-mode selector to the desktop Generate tab that sends `content_mode` to the prompt optimizer (`/v1/enhance`) and the renderer (`/v1/generate`), steering prompt text only.

**Architecture:** A `ContentMode` enum drives a SwiftUI segmented control held in `GenerationView` view state (default Neutral, resets each launch). The desktop `EngineService` attaches `content_mode` to the enhance and generate request bodies via one shared pure helper. The server already consumes `content_mode` on `/v1/enhance`; we add it to `GeneratePayload` and stamp it into the render's embedded `ImageMetadata`.

**Tech Stack:** Swift 5.9, SwiftUI, mlx-swift; XCTest via `xcodebuild`.

## Global Constraints

- Platform: macOS 14+ / Apple Silicon. Language: Swift 5.9+.
- **Mode affects prompt TEXT ONLY** — never guidance, steps, or any numeric parameter. The `guidanceBoost` field in `content-modes.json` is dead config; do not read it.
- Mode string values are exactly `neutral` | `banana` | `avocado` (must match server `ContentModeManager.Mode`).
- Presets are independent of modes — do not read or write the mode when loading/saving presets.
- Default mode is `neutral`; selection lives in view `@State` only (no `@AppStorage`) so it resets to Neutral on each app launch.
- Wire keys are snake_case (`content_mode`); the server decoders apply `.convertFromSnakeCase`, so Swift `CodingKey`/struct fields use camelCase `contentMode`.
- Build CLI: `xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode`
- Test: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:<target>`

## Out of Scope (deferred follow-up)

- Applying the mode's `negativePromptAdditions` on the `/v1/generate` server path. The config arrays are empty today (pure no-op), and the negative-prompt resolution is family-dependent (`WarmServer.swift:1750-1800`, where Turbo/Klein force `negativePrompt = nil`). Tracked as a separate task; not built here.

## File Structure

- **Create** `Sources/ComfyBoxDesktop/ContentMode.swift` — the `ContentMode` enum (mirrors `NSFWGate.swift` as a small top-level model file).
- **Modify** `Sources/ComfyBoxDesktop/EngineService.swift` — shared `content_mode` payload helper; thread it into `enhancePrompt` and `generate`.
- **Modify** `Sources/ComfyBoxDesktop/Views/GenerationView.swift` — mode state + segmented control; pass mode to enhance/generate calls.
- **Modify** `Sources/ZImage/Util/ImageIO.swift` — add `contentMode` to `ImageMetadata.generation(...)`.
- **Modify** `Sources/ZImage/Server/WarmServer.swift` — add `contentMode` to `GeneratePayload`; pass it into the render metadata.
- **Create** `Tests/ComfyBoxDesktopTests/ContentModeTests.swift`, **Modify**/**Create** tests under `Tests/ZImageTests/`.

---

### Task 1: `ContentMode` enum + selector UI

**Files:**
- Create: `Sources/ComfyBoxDesktop/ContentMode.swift`
- Create: `Tests/ComfyBoxDesktopTests/ContentModeTests.swift`
- Modify: `Sources/ComfyBoxDesktop/Views/GenerationView.swift` (state near line 66-118; promptSection header at 432-453)

**Interfaces:**
- Produces: `enum ContentMode: String, CaseIterable, Identifiable, Sendable` with cases `.neutral`/`.banana`/`.avocado`, `var id: String`, `var label: String`.

- [ ] **Step 1: Write the failing test**

```swift
// Tests/ComfyBoxDesktopTests/ContentModeTests.swift
import XCTest
@testable import ComfyBoxDesktop

final class ContentModeTests: XCTestCase {
    func testRawValuesMatchServerModeStrings() {
        XCTAssertEqual(ContentMode.neutral.rawValue, "neutral")
        XCTAssertEqual(ContentMode.banana.rawValue, "banana")
        XCTAssertEqual(ContentMode.avocado.rawValue, "avocado")
    }

    func testAllCasesInDisplayOrder() {
        XCTAssertEqual(ContentMode.allCases, [.neutral, .banana, .avocado])
    }

    func testLabelsCarryEmoji() {
        XCTAssertTrue(ContentMode.neutral.label.contains("Neutral"))
        XCTAssertTrue(ContentMode.banana.label.contains("🍌"))
        XCTAssertTrue(ContentMode.avocado.label.contains("🥑"))
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ComfyBoxDesktopTests/ContentModeTests`
Expected: FAIL — `cannot find 'ContentMode' in scope`.

- [ ] **Step 3: Create the enum**

```swift
// Sources/ComfyBoxDesktop/ContentMode.swift
// ContentMode.swift — "fruit mode" for the Generate tab. Steers prompt TEXT
// only (optimizer hint + negative additions); never guidance/numeric params.
// Raw values must match the server's ContentModeManager.Mode.

import Foundation

public enum ContentMode: String, CaseIterable, Identifiable, Sendable {
    case neutral
    case banana
    case avocado

    public var id: String { rawValue }

    /// Emoji + name for the segmented control.
    public var label: String {
        switch self {
        case .neutral: return "🍎 Neutral"
        case .banana:  return "🍌 Banana"
        case .avocado: return "🥑 Avocado"
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ComfyBoxDesktopTests/ContentModeTests`
Expected: PASS (3 tests).

- [ ] **Step 5: Add mode state to `GenerationView`**

In `Sources/ComfyBoxDesktop/Views/GenerationView.swift`, alongside the other `@State` declarations (near line 114-118, by `showingSavePreset`/`serverPresets`), add:

```swift
    /// Fruit mode steering the optimizer + negative prompt. View state only →
    /// resets to Neutral each launch (never silently persists 🥑).
    @State private var contentMode: ContentMode = .neutral
```

- [ ] **Step 6: Add the segmented control to the prompt header**

In the same file, `promptSection` (starts line 431), replace the header `HStack { … }` (lines 432-453, the one containing `Text("Prompt")`, `Spacer()`, and the Enhance `Button`) so the picker sits between the spacer and the Enhance button:

```swift
            HStack {
                Text("Prompt")
                    .font(.headline)
                Spacer()
                // Fruit mode — steers the optimizer + negative prompt (text only).
                Picker("Mode", selection: $contentMode) {
                    ForEach(ContentMode.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                .help("Content mode: steers prompt optimization and negative prompt (not guidance)")
                // Enhance button
                Button(action: { enhancePrompt() }) {
                    HStack(spacing: 4) {
                        if isEnhancing {
                            ProgressView()
                                .controlSize(.mini)
                        } else {
                            Image(systemName: "sparkles")
                        }
                        Text("Enhance")
                    }
                }
                .controlSize(.small)
                .buttonStyle(.bordered)
                .disabled(!canEnhance)
                .help("Send prompt to LLM for enhancement")
            }
```

- [ ] **Step 7: Build to verify the view compiles**

Run: `xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode`
Expected: BUILD SUCCEEDED.

- [ ] **Step 8: Commit**

```bash
git add Sources/ComfyBoxDesktop/ContentMode.swift Tests/ComfyBoxDesktopTests/ContentModeTests.swift Sources/ComfyBoxDesktop/Views/GenerationView.swift
git commit -m "feat(desktop): fruit-mode enum + segmented selector in Generate tab"
```

---

### Task 2: Send `content_mode` to the optimizer (`/v1/enhance`)

**Files:**
- Modify: `Sources/ComfyBoxDesktop/EngineService.swift` (add helper near the Generation MARK ~line 305; `enhancePrompt` at 655-675)
- Modify: `Sources/ComfyBoxDesktop/Views/GenerationView.swift` (`enhancePrompt()` at 1021-1042)
- Modify: `Tests/ComfyBoxDesktopTests/ContentModeTests.swift`

**Interfaces:**
- Consumes: `ContentMode` (Task 1).
- Produces: `static func EngineService.attachingContentMode(_ base: [String: Any], mode: ContentMode) -> [String: Any]`; `EngineService.enhancePrompt(_ prompt: String, contentMode: ContentMode = .neutral)`.

- [ ] **Step 1: Write the failing test**

Append to `Tests/ComfyBoxDesktopTests/ContentModeTests.swift`:

```swift
    func testAttachingContentModeAddsSnakeCaseKey() {
        let out = EngineService.attachingContentMode(["prompt": "hi"], mode: .avocado)
        XCTAssertEqual(out["content_mode"] as? String, "avocado")
        XCTAssertEqual(out["prompt"] as? String, "hi")
    }

    func testAttachingContentModeNeutralIsExplicit() {
        let out = EngineService.attachingContentMode([:], mode: .neutral)
        XCTAssertEqual(out["content_mode"] as? String, "neutral")
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ComfyBoxDesktopTests/ContentModeTests`
Expected: FAIL — `type 'EngineService' has no member 'attachingContentMode'`.

- [ ] **Step 3: Add the shared helper**

In `Sources/ComfyBoxDesktop/EngineService.swift`, inside the `EngineService` class (place it just above `public func generate(...)` at line 308), add:

```swift
    /// Attach the fruit mode to a request body as `content_mode`. Always sent
    /// (including neutral) so the server's behavior is explicit, never inferred.
    static func attachingContentMode(_ base: [String: Any], mode: ContentMode) -> [String: Any] {
        var out = base
        out["content_mode"] = mode.rawValue
        return out
    }
```

- [ ] **Step 4: Thread it into `enhancePrompt`**

In the same file, change `enhancePrompt` (line 655) signature and body:

```swift
    public func enhancePrompt(_ prompt: String, contentMode: ContentMode = .neutral) async throws -> String {
        guard let client = client, connectionState.isConnected else {
            throw EngineServiceError.notConnected
        }

        let payloadDict = Self.attachingContentMode(["prompt": prompt], mode: contentMode)
        let bodyData = try JSONSerialization.data(withJSONObject: payloadDict)
        let (status, responseData) = try await client.post("/v1/enhance", body: bodyData)
```

(Leave the rest of the method — status check and response parsing — unchanged.)

- [ ] **Step 5: Pass the mode from the view**

In `Sources/ComfyBoxDesktop/Views/GenerationView.swift`, in `enhancePrompt()` (line 1027), change the call:

```swift
                let enhanced = try await engine.enhancePrompt(prompt, contentMode: contentMode)
```

- [ ] **Step 6: Run tests + build**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ComfyBoxDesktopTests/ContentModeTests`
Expected: PASS (5 tests).
Run: `xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode`
Expected: BUILD SUCCEEDED.

- [ ] **Step 7: Commit**

```bash
git add Sources/ComfyBoxDesktop/EngineService.swift Sources/ComfyBoxDesktop/Views/GenerationView.swift Tests/ComfyBoxDesktopTests/ContentModeTests.swift
git commit -m "feat(desktop): send content_mode to /v1/enhance so fruit mode steers the optimizer"
```

---

### Task 3: Send `content_mode` to the renderer (`/v1/generate`)

**Files:**
- Modify: `Sources/ComfyBoxDesktop/EngineService.swift` (`generate` at 308-366)
- Modify: `Sources/ComfyBoxDesktop/Views/GenerationView.swift` (the `engine.generate(` call site)

**Interfaces:**
- Consumes: `attachingContentMode` (Task 2), `ContentMode` (Task 1).
- Produces: `EngineService.generate(_ request: GenerationRequest, contentMode: ContentMode = .neutral)`.

- [ ] **Step 1: Add the mode parameter to `generate`**

In `Sources/ComfyBoxDesktop/EngineService.swift`, change the signature (line 308):

```swift
    public func generate(_ request: GenerationRequest, contentMode: ContentMode = .neutral) async throws -> String {
```

- [ ] **Step 2: Attach `content_mode` to the payload**

In the same method, immediately after the `payloadDict` is fully populated and before `let bodyData = try JSONSerialization.data(withJSONObject: payloadDict)` (line 365), insert:

```swift
        payloadDict = Self.attachingContentMode(payloadDict, mode: contentMode)
```

- [ ] **Step 3: Pass the mode from the view's generate call**

In `Sources/ComfyBoxDesktop/Views/GenerationView.swift`, find the render invocation:

Run: `grep -n "engine.generate(" Sources/ComfyBoxDesktop/Views/GenerationView.swift`

At that call, add the argument. Example — change `try await engine.generate(request)` to:

```swift
                try await engine.generate(request, contentMode: contentMode)
```

- [ ] **Step 4: Build to verify**

Run: `xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode`
Expected: BUILD SUCCEEDED.

- [ ] **Step 5: Commit**

```bash
git add Sources/ComfyBoxDesktop/EngineService.swift Sources/ComfyBoxDesktop/Views/GenerationView.swift
git commit -m "feat(desktop): send content_mode on /v1/generate"
```

---

### Task 4: Server — carry `content_mode` into render metadata

**Files:**
- Modify: `Sources/ZImage/Util/ImageIO.swift` (`ImageMetadata.generation` at 195-217)
- Modify: `Sources/ZImage/Server/WarmServer.swift` (`GeneratePayload` struct 3894-3960, decoder 3983-4017, memberwise init 3934-3961; save site 3481-3485)
- Create: `Tests/ZImageTests/ImageMetadataContentModeTests.swift`

**Interfaces:**
- Consumes: nothing from prior tasks (wire key `content_mode`).
- Produces: `GeneratePayload.contentMode: String?`; `ImageMetadata.generation(..., contentMode:)`.

- [ ] **Step 1: Write the failing test**

```swift
// Tests/ZImageTests/ImageMetadataContentModeTests.swift
import XCTest
@testable import ZImage

final class ImageMetadataContentModeTests: XCTestCase {
    func testGenerationEmbedsContentMode() throws {
        let m = QwenImageIO.ImageMetadata.generation(
            prompt: "a cat", seed: 7, steps: 9, guidance: 0, width: 1024, height: 1024,
            model: "cyberrealisticZImage_v50", contentMode: "avocado")
        let json = try XCTUnwrap(m.parametersJSON)
        XCTAssertTrue(json.contains("\"content_mode\""))
        XCTAssertTrue(json.contains("avocado"))
    }

    func testGenerationOmitsEmptyContentMode() throws {
        let m = QwenImageIO.ImageMetadata.generation(prompt: "a cat", contentMode: nil)
        let json = try XCTUnwrap(m.parametersJSON)
        XCTAssertFalse(json.contains("content_mode"))
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/ImageMetadataContentModeTests`
Expected: FAIL — `extra argument 'contentMode' in call`.

- [ ] **Step 3: Add `contentMode` to the metadata factory**

In `Sources/ZImage/Util/ImageIO.swift`, `ImageMetadata.generation` (signature at 195-199): add the parameter and the params line.

Change the signature's last line from:
```swift
      height: Int? = nil, model: String? = nil, generatedBy: String? = nil
    ) -> ImageMetadata {
```
to:
```swift
      height: Int? = nil, model: String? = nil, generatedBy: String? = nil,
      contentMode: String? = nil
    ) -> ImageMetadata {
```

Then, right after the `if let generatedBy, !generatedBy.isEmpty { params["source"] = generatedBy }` line (≈210), add:
```swift
      if let contentMode, !contentMode.isEmpty { params["content_mode"] = contentMode }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/ImageMetadataContentModeTests`
Expected: PASS (2 tests).

- [ ] **Step 5: Add `contentMode` to `GeneratePayload`**

In `Sources/ZImage/Server/WarmServer.swift`:

(a) In the `struct GeneratePayload` stored properties, next to `let source: String?` (line 3931), add:
```swift
  /// Fruit mode (neutral | banana | avocado) — stamped into render metadata.
  let contentMode: String?
```

(b) In the memberwise `init(...)` parameter list, add `contentMode: String? = nil,` (put it next to `source: String? = nil,` at line 3946), and in the body add `self.contentMode = contentMode` (next to `self.source = source`, line 3951).

(c) In the `CodingKeys` enum, add `case contentMode` next to `case source` (line 3982). (Raw value defaults to `"contentMode"`, matching the wire key `content_mode` after `.convertFromSnakeCase`.)

(d) In `init(from decoder:)`, next to `source = try c.decodeIfPresent(String.self, forKey: .source)` (line 4016), add:
```swift
    contentMode = try c.decodeIfPresent(String.self, forKey: .contentMode)
```

- [ ] **Step 6: Pass it into the render metadata**

In the same file, the save site at line 3481-3485 — change the `.generation(...)` call to pass the mode:

```swift
    try QwenImageIO.saveImage(array: imageArray, to: outputURL,
      metadata: .generation(prompt: payload.prompt, negativePrompt: payload.negativePrompt,
        seed: seed, steps: steps, guidance: guidance, width: width, height: height,
        generatedBy: payload.source, contentMode: payload.contentMode))
```

- [ ] **Step 7: Add a decoder test for the payload**

```swift
// Append to Tests/ZImageTests/ImageMetadataContentModeTests.swift is not possible
// (GeneratePayload is internal to the ComfyBox executable target, not ZImage).
// Instead verify end-to-end in Step 8 via a live render, and rely on the
// compiler + existing /v1/generate decode tests for wiring.
```

Note: `GeneratePayload` lives in the `ComfyBox` executable target (`WarmServer.swift`), which XCTest cannot import directly. Its decode path is covered by the build + the E2E check in Step 8, not a unit test.

- [ ] **Step 8: Build, test, and verify end-to-end**

Run: `xcodebuild -scheme ComfyBox -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode`
Expected: BUILD SUCCEEDED.
Run: `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/ImageMetadataContentModeTests`
Expected: PASS.

Live check (server must be restarted with the new binary first):
```bash
curl -s -X POST http://127.0.0.1:7870/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red apple on a table","content_mode":"banana","steps":9,"width":1024,"height":1024,"source":"desktop"}' | python3 -c 'import sys,json;print(json.load(sys.stdin).get("output_path",""))'
# Then read back the embedded metadata on the returned PNG:
#   strings <output.png> | grep content_mode   → shows "content_mode":"banana"
```
Expected: the rendered PNG's `UserComment` JSON contains `"content_mode":"banana"`.

- [ ] **Step 9: Commit**

```bash
git add Sources/ZImage/Util/ImageIO.swift Sources/ZImage/Server/WarmServer.swift Tests/ZImageTests/ImageMetadataContentModeTests.swift
git commit -m "feat(server): stamp content_mode into render metadata; accept it on /v1/generate"
```

---

## Self-Review

- **Spec coverage:** Selector UI (Task 1) ✓; optimizer steering via `/v1/enhance` (Task 2) ✓; `content_mode` on `/v1/generate` (Task 3) ✓; `content_mode` in render metadata (Task 4) ✓; independent-of-presets (Global Constraints + no preset code touched) ✓; default Neutral + per-session reset (`@State`, Task 1 Step 5) ✓; no guidance/numeric influence (Global Constraints; no numeric code touched) ✓. **Gap:** `negativePromptAdditions` application is explicitly deferred (Out of Scope) — empty config today, tangled family-specific resolution; flagged for the user.
- **Placeholder scan:** No TBD/TODO; every code step shows complete code. Task 3 Step 3 uses a `grep` locate + exact replacement line (concrete, not a placeholder). Task 4 Step 7 documents *why* there is no payload unit test (target-visibility fact), not deferred work.
- **Type consistency:** `ContentMode` (rawValue `neutral|banana|avocado`) is used identically across Tasks 1-3; `attachingContentMode(_:mode:)` signature matches its call sites; `ImageMetadata.generation(..., contentMode:)` matches the save-site call in Task 4 Step 6.
