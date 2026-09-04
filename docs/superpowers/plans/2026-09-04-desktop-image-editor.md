# Desktop Image Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A native Core Image editor tab in ComfyBoxDesktop for tone/color, crop, one brush-masked local layer, and Vision subject lift, saving derived assets with a reopenable recipe in the adjacent sidecar.

**Architecture:** A Codable `EditRecipe` value drives a pure `EditRenderer` (Core Image chain) used for both preview and export. An `@Observable` `EditSession` debounces preview renders; `EditExporter` writes PNG + sidecar and ingests. The Inpaint brush canvas is extracted into shared `MaskStrokes`/`MaskRasterizer`/`MaskCanvas`.

**Tech Stack:** Swift 5.9, SwiftUI, Core Image, Vision (macOS 14), ImageIO, Swift Testing (`import Testing`).

**Spec:** `docs/superpowers/specs/2026-09-04-desktop-image-editor-design.md`

## Global Constraints

- Work in the worktree `/Users/toddwalderman/Projects/zimage-editor` on branch `feat/desktop-image-editor`. All paths below are relative to it.
- macOS 14 deployment target (`Package.swift`). Vision foreground mask API is available.
- Zero Python, no new package dependencies. Apple frameworks only.
- No engine, server, MCP, or catalog schema changes. `DAMAsset` is not modified.
- Tests use Swift Testing (`@Suite`, `@Test`, `#expect`), `@testable import ComfyBoxDesktop`, no windows, no Vision, no model weights.
- Test command (run from the worktree):
  `xcodebuild test -scheme comfybox-Package -destination 'platform=macOS' -enableCodeCoverage NO -derivedDataPath .build/xcode -only-testing:ComfyBoxDesktopTests/<SuiteName> 2>&1 | grep -E "Test Suite|passed|failed|error:" | tail -20`
- Build-only command for UI tasks:
  `xcodebuild build -scheme ComfyBoxDesktop -destination 'platform=macOS' -derivedDataPath .build/xcode 2>&1 | grep -E "error:|warning: unused|BUILD" | tail -20`
- Commit after every task with the trailer:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TKRiaSt9dcYwdCk7So6qaN
  ```
- Recipe value ranges and parameter mappings are fixed in Task 4; do not invent others.
- Sidecar is `<image basename>.json` next to the image (desktop convention, see `Sources/ComfyBoxDesktop/DAM/AssetIngestor.swift:631-665`).

---

## File map

| File | Responsibility |
|---|---|
| `Sources/ComfyBoxDesktop/Edit/MaskStrokes.swift` | `MaskStroke`, `MaskStrokes` value model; `MaskRasterizer` (strokes → CGImage) |
| `Sources/ComfyBoxDesktop/Views/MaskCanvas.swift` | SwiftUI overlay + drag gesture that edits `MaskStrokes`; `fitRect` helper |
| `Sources/ComfyBoxDesktop/Views/InpaintView.swift` | Modified: uses MaskCanvas/MaskRasterizer; accepts `pendingMask` |
| `Sources/ComfyBoxDesktop/Edit/EditRecipe.swift` | `EditRecipe`, `EditGeometry`, `EditAdjustments`, `ToneCurves`, `CurvePoint`, `EditLocalLayer`, `EditSubject` |
| `Sources/ComfyBoxDesktop/Edit/EditRenderer.swift` | Pure CIImage pipeline + parameter mapping functions |
| `Sources/ComfyBoxDesktop/Edit/SubjectMasker.swift` | Vision foreground mask actor |
| `Sources/ComfyBoxDesktop/Edit/EditSidecar.swift` | `edit` block read/write helpers |
| `Sources/ComfyBoxDesktop/Edit/EditExporter.swift` | Full-res render, PNG + sidecar write, ingest |
| `Sources/ComfyBoxDesktop/Edit/EditSession.swift` | Observable session: recipe, undo, debounced preview |
| `Sources/ComfyBoxDesktop/Views/CurvesEditor.swift` | Draggable tone-curve graph |
| `Sources/ComfyBoxDesktop/Views/EditView.swift` | `EditView` (canvas + panel + toolbar) and `EditTab` |
| `Sources/ComfyBoxDesktop/ComfyBoxDesktopApp.swift` | Modified: `.edit` tab, pending bindings, callbacks |
| `Sources/ComfyBoxDesktop/Views/AssetDetailView.swift` | Modified: Edit action, Edited-from line |
| `Sources/ComfyBoxDesktop/Views/GalleryView.swift` | Modified: `onEdit` passthrough |
| `Tests/ComfyBoxDesktopTests/EditTestSupport.swift` | Synthetic image + pixel readback helpers |
| `Tests/ComfyBoxDesktopTests/MaskRasterizerTests.swift` | |
| `Tests/ComfyBoxDesktopTests/EditRecipeTests.swift` | |
| `Tests/ComfyBoxDesktopTests/EditRendererTests.swift` | |
| `Tests/ComfyBoxDesktopTests/EditSidecarTests.swift` | |
| `Tests/ComfyBoxDesktopTests/EditExporterTests.swift` | |
| `Tests/ComfyBoxDesktopTests/EditSessionTests.swift` | |

---

### Task 1: MaskStrokes model and MaskRasterizer

**Files:**
- Create: `Sources/ComfyBoxDesktop/Edit/MaskStrokes.swift`
- Create: `Tests/ComfyBoxDesktopTests/EditTestSupport.swift`
- Test: `Tests/ComfyBoxDesktopTests/MaskRasterizerTests.swift`

**Interfaces:**
- Produces:
  ```swift
  public struct MaskStroke: Codable, Equatable, Sendable, Identifiable {
      public var id: UUID; public var points: [CGPoint]; public var size: Double; public var erase: Bool
      public init(points: [CGPoint], size: Double, erase: Bool)
  }
  public struct MaskStrokes: Codable, Equatable, Sendable {
      public var strokes: [MaskStroke]
      public init(strokes: [MaskStroke] = [])
      public var isEmpty: Bool
      public mutating func append(_ s: MaskStroke)
      public mutating func undoLast()
      public mutating func clear()
  }
  public enum MaskRasterizer {
      /// White (255) where painted, black elsewhere; 8-bit gray, top-left origin like the normalized points.
      public static func render(_ strokes: MaskStrokes, size: CGSize) -> CGImage?
      public static func pngData(_ strokes: MaskStrokes, size: CGSize) -> Data?
  }
  ```
- Test helper (used by every later test task):
  ```swift
  enum EditTestSupport {
      /// RGBA8 pixel at (x, y) with top-left origin.
      static func pixel(_ image: CGImage, x: Int, y: Int) -> (r: UInt8, g: UInt8, b: UInt8, a: UInt8)
      /// 8-bit gray value at (x, y), top-left origin. Works for gray and RGBA images (uses red channel for RGBA).
      static func gray(_ image: CGImage, x: Int, y: Int) -> UInt8
      /// Horizontal gradient black→white, `width`×`height`, RGBA8, opaque.
      static func horizontalGradient(width: Int, height: Int) -> CGImage
      /// Solid color image.
      static func solid(r: UInt8, g: UInt8, b: UInt8, width: Int, height: Int) -> CGImage
  }
  ```

- [ ] **Step 1: Write the test support file**

```swift
// EditTestSupport.swift — synthetic images and pixel readback for editor tests
import Foundation
import CoreGraphics
import ImageIO
@testable import ComfyBoxDesktop

enum EditTestSupport {
    static func rgbaBytes(_ image: CGImage) -> [UInt8] {
        let w = image.width, h = image.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return bytes
    }

    static func pixel(_ image: CGImage, x: Int, y: Int) -> (r: UInt8, g: UInt8, b: UInt8, a: UInt8) {
        let b = rgbaBytes(image)
        let i = (y * image.width + x) * 4
        return (b[i], b[i + 1], b[i + 2], b[i + 3])
    }

    static func gray(_ image: CGImage, x: Int, y: Int) -> UInt8 { pixel(image, x: x, y: y).r }

    static func horizontalGradient(width: Int, height: Int) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let v = UInt8((x * 255) / max(width - 1, 1))
                let i = (y * width + x) * 4
                bytes[i] = v; bytes[i + 1] = v; bytes[i + 2] = v; bytes[i + 3] = 255
            }
        }
        return make(bytes, width: width, height: height)
    }

    static func solid(r: UInt8, g: UInt8, b: UInt8, width: Int, height: Int) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for p in 0..<(width * height) {
            bytes[p * 4] = r; bytes[p * 4 + 1] = g; bytes[p * 4 + 2] = b; bytes[p * 4 + 3] = 255
        }
        return make(bytes, width: width, height: height)
    }

    private static func make(_ bytes: [UInt8], width: Int, height: Int) -> CGImage {
        var copy = bytes
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(data: &copy, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return ctx.makeImage()!
    }
}
```

- [ ] **Step 2: Write the failing rasterizer tests**

```swift
// MaskRasterizerTests.swift
import Testing
import Foundation
import CoreGraphics
@testable import ComfyBoxDesktop

@Suite("MaskRasterizer")
struct MaskRasterizerTests {
    @Test("single stroke paints white at its points and black elsewhere")
    func paintsStroke() {
        var strokes = MaskStrokes()
        strokes.append(MaskStroke(points: [CGPoint(x: 0.5, y: 0.5)], size: 0.2, erase: false))
        let img = MaskRasterizer.render(strokes, size: CGSize(width: 100, height: 100))!
        #expect(img.width == 100 && img.height == 100)
        #expect(EditTestSupport.gray(img, x: 50, y: 50) > 200)
        #expect(EditTestSupport.gray(img, x: 5, y: 5) < 20)
    }

    @Test("erase stroke clears painted area")
    func eraseClears() {
        var strokes = MaskStrokes()
        strokes.append(MaskStroke(points: [CGPoint(x: 0.5, y: 0.5)], size: 0.3, erase: false))
        strokes.append(MaskStroke(points: [CGPoint(x: 0.5, y: 0.5)], size: 0.1, erase: true))
        let img = MaskRasterizer.render(strokes, size: CGSize(width: 100, height: 100))!
        #expect(EditTestSupport.gray(img, x: 50, y: 50) < 20)
        #expect(EditTestSupport.gray(img, x: 50, y: 40) > 200)   // ring outside the erase still painted
    }

    @Test("y axis is top-down: a stroke at y=0.1 lands near the top row")
    func orientation() {
        var strokes = MaskStrokes()
        strokes.append(MaskStroke(points: [CGPoint(x: 0.5, y: 0.1)], size: 0.1, erase: false))
        let img = MaskRasterizer.render(strokes, size: CGSize(width: 100, height: 100))!
        #expect(EditTestSupport.gray(img, x: 50, y: 10) > 200)
        #expect(EditTestSupport.gray(img, x: 50, y: 90) < 20)
    }

    @Test("empty strokes render all black; zero size returns nil")
    func emptyAndZero() {
        let img = MaskRasterizer.render(MaskStrokes(), size: CGSize(width: 10, height: 10))!
        #expect(EditTestSupport.gray(img, x: 5, y: 5) == 0)
        #expect(MaskRasterizer.render(MaskStrokes(), size: .zero) == nil)
    }

    @Test("MaskStrokes round-trips through JSON and undoLast/clear work")
    func modelBasics() throws {
        var s = MaskStrokes()
        #expect(s.isEmpty)
        s.append(MaskStroke(points: [CGPoint(x: 0.1, y: 0.2)], size: 0.05, erase: false))
        let data = try JSONEncoder().encode(s)
        let back = try JSONDecoder().decode(MaskStrokes.self, from: data)
        #expect(back == s)
        s.undoLast(); #expect(s.isEmpty)
        s.append(MaskStroke(points: [], size: 0.05, erase: true)); s.clear(); #expect(s.isEmpty)
    }
}
```

- [ ] **Step 3: Run to verify failure**

Run the test command with `-only-testing:ComfyBoxDesktopTests/MaskRasterizerTests`. Expected: compile error, `MaskStrokes` not found.

- [ ] **Step 4: Implement**

```swift
// MaskStrokes.swift — brush mask model shared by Inpaint and the Edit tab
//
// Points are normalized (0…1) in the displayed image rect with a top-left
// origin; `size` is a fraction of the image width. The rasterizer flips Y
// because CGContext bitmaps are bottom-up.

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

public struct MaskStroke: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var points: [CGPoint]
    public var size: Double
    public var erase: Bool

    public init(points: [CGPoint], size: Double, erase: Bool) {
        self.id = UUID(); self.points = points; self.size = size; self.erase = erase
    }
}

public struct MaskStrokes: Codable, Equatable, Sendable {
    public var strokes: [MaskStroke]
    public init(strokes: [MaskStroke] = []) { self.strokes = strokes }
    public var isEmpty: Bool { strokes.isEmpty }
    public mutating func append(_ s: MaskStroke) { strokes.append(s) }
    public mutating func undoLast() { if !strokes.isEmpty { strokes.removeLast() } }
    public mutating func clear() { strokes.removeAll() }
}

public enum MaskRasterizer {
    public static func render(_ strokes: MaskStrokes, size: CGSize) -> CGImage? {
        let w = Int(size.width.rounded()), h = Int(size.height.rounded())
        guard w > 0, h > 0 else { return nil }
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        ctx.setFillColor(gray: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.setLineCap(.round); ctx.setLineJoin(.round)
        for stroke in strokes.strokes {
            guard let first = stroke.points.first else { continue }
            ctx.setStrokeColor(gray: stroke.erase ? 0 : 1, alpha: 1)
            ctx.setLineWidth(CGFloat(stroke.size) * CGFloat(w))
            ctx.beginPath()
            ctx.move(to: CGPoint(x: first.x * CGFloat(w), y: CGFloat(h) - first.y * CGFloat(h)))
            if stroke.points.count == 1 {
                // A dot: zero-length line still draws a round cap.
                ctx.addLine(to: CGPoint(x: first.x * CGFloat(w), y: CGFloat(h) - first.y * CGFloat(h)))
            }
            for p in stroke.points.dropFirst() {
                ctx.addLine(to: CGPoint(x: p.x * CGFloat(w), y: CGFloat(h) - p.y * CGFloat(h)))
            }
            ctx.strokePath()
        }
        return ctx.makeImage()
    }

    public static func pngData(_ strokes: MaskStrokes, size: CGSize) -> Data? {
        guard let img = render(strokes, size: size) else { return nil }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }
}
```

- [ ] **Step 5: Run to verify pass**

Same command. Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add Sources/ComfyBoxDesktop/Edit/MaskStrokes.swift Tests/ComfyBoxDesktopTests/EditTestSupport.swift Tests/ComfyBoxDesktopTests/MaskRasterizerTests.swift
git commit -m "feat(desktop): shared MaskStrokes model and MaskRasterizer"
```

---

### Task 2: MaskCanvas view and Inpaint refactor

**Files:**
- Create: `Sources/ComfyBoxDesktop/Views/MaskCanvas.swift`
- Modify: `Sources/ComfyBoxDesktop/Views/InpaintView.swift` (whole file; current stroke code at lines 20-24, 37, 121-172, 209-256)
- Test: `Tests/ComfyBoxDesktopTests/MaskRasterizerTests.swift` (add one parity test)

**Interfaces:**
- Consumes: `MaskStrokes`, `MaskStroke`, `MaskRasterizer` from Task 1.
- Produces:
  ```swift
  struct MaskCanvas: View {
      let imageSize: CGSize          // displayed (fitted) rect size, NOT pixels
      @Binding var strokes: MaskStrokes
      var brushPoints: CGFloat       // brush diameter in view points
      var erase: Bool
      var tint: Color = .red
      var enabled: Bool = true
  }
  enum ImageFit { static func rect(imageSize: CGSize, in container: CGSize) -> CGRect }
  ```
  `InpaintView` gains `@Binding var pendingMask: MaskStrokes?` (init parameter, consumed together with `pendingImage`).

- [ ] **Step 1: Add the parity test** (pins the old NSBezierPath algorithm against the new rasterizer)

Append to `MaskRasterizerTests`:

```swift
    /// The pre-refactor Inpaint rasterizer, kept verbatim as the oracle.
    private func legacyMaskPNG(_ strokes: MaskStrokes, pixelSize: CGSize) -> Data? {
        let W = Int(pixelSize.width), H = Int(pixelSize.height)
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: W, pixelsHigh: H, bitsPerSample: 8,
            samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0) else { return nil }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor.black.setFill(); NSBezierPath(rect: NSRect(x: 0, y: 0, width: W, height: H)).fill()
        for stroke in strokes.strokes {
            (stroke.erase ? NSColor.black : NSColor.white).setStroke()
            let path = NSBezierPath()
            path.lineWidth = CGFloat(stroke.size) * pixelSize.width
            path.lineCapStyle = .round; path.lineJoinStyle = .round
            for (i, pt) in stroke.points.enumerated() {
                let x = pt.x * pixelSize.width
                let y = pixelSize.height - pt.y * pixelSize.height
                if i == 0 { path.move(to: NSPoint(x: x, y: y)) } else { path.line(to: NSPoint(x: x, y: y)) }
            }
            path.stroke()
        }
        NSGraphicsContext.restoreGraphicsState()
        return rep.representation(using: .png, properties: [:])
    }

    @Test("matches the legacy Inpaint rasterizer on a sampled grid")
    func legacyParity() {
        var s = MaskStrokes()
        s.append(MaskStroke(points: [CGPoint(x: 0.2, y: 0.3), CGPoint(x: 0.7, y: 0.6)], size: 0.08, erase: false))
        s.append(MaskStroke(points: [CGPoint(x: 0.5, y: 0.5)], size: 0.04, erase: true))
        let size = CGSize(width: 120, height: 80)
        let new = MaskRasterizer.render(s, size: size)!
        let legacy = NSBitmapImageRep(data: legacyMaskPNG(s, pixelSize: size)!)!.cgImage!
        var mismatches = 0
        for y in stride(from: 2, to: 80, by: 4) {
            for x in stride(from: 2, to: 120, by: 4) {
                let a = EditTestSupport.gray(new, x: x, y: y) > 127
                let b = EditTestSupport.gray(legacy, x: x, y: y) > 127
                if a != b { mismatches += 1 }
            }
        }
        #expect(mismatches <= 4)   // anti-aliasing at stroke edges only
    }
```

Add `import AppKit` at the top of the test file.

- [ ] **Step 2: Run to verify the parity test passes against Task 1's implementation**

Expected: pass. (If mismatches exceed 4, the Y flip or line width differs; fix `MaskRasterizer`, not the oracle.)

- [ ] **Step 3: Create MaskCanvas.swift**

```swift
// MaskCanvas.swift — brush overlay + gesture shared by Inpaint and Edit
//
// Draws the strokes tinted over the fitted image rect and records new
// strokes in normalized coordinates. The caller positions this view in
// the same rect as the image (see `ImageFit.rect`).

import SwiftUI

enum ImageFit {
    /// Aspect-fit rect for `imageSize` centered in `container`.
    static func rect(imageSize: CGSize, in container: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return .zero }
        let scale = min(container.width / imageSize.width, container.height / imageSize.height)
        let w = imageSize.width * scale, h = imageSize.height * scale
        return CGRect(x: (container.width - w) / 2, y: (container.height - h) / 2, width: w, height: h)
    }
}

struct MaskCanvas: View {
    let imageSize: CGSize
    @Binding var strokes: MaskStrokes
    var brushPoints: CGFloat
    var erase: Bool
    var tint: Color = .red
    var enabled: Bool = true

    @State private var current: MaskStroke?

    var body: some View {
        ZStack {
            Canvas { ctx, _ in
                for stroke in strokes.strokes + (current.map { [$0] } ?? []) {
                    var path = Path()
                    let pts = stroke.points.map { CGPoint(x: $0.x * imageSize.width, y: $0.y * imageSize.height) }
                    if pts.count == 1 { path.move(to: pts[0]); path.addLine(to: pts[0]) } else { path.addLines(pts) }
                    ctx.blendMode = stroke.erase ? .destinationOut : .normal
                    ctx.stroke(path, with: .color(stroke.erase ? .white : tint.opacity(0.45)),
                               style: StrokeStyle(lineWidth: CGFloat(stroke.size) * imageSize.width,
                                                  lineCap: .round, lineJoin: .round))
                }
            }
            .allowsHitTesting(false)
            if enabled {
                Color.clear.contentShape(Rectangle())
                    .gesture(DragGesture(minimumDistance: 0).onChanged { v in
                        let p = CGPoint(x: v.location.x / imageSize.width, y: v.location.y / imageSize.height)
                        if current == nil {
                            current = MaskStroke(points: [p], size: Double(brushPoints / imageSize.width), erase: erase)
                        } else { current?.points.append(p) }
                    }.onEnded { _ in
                        if let s = current { strokes.append(s); current = nil }
                    })
            }
        }
        .frame(width: imageSize.width, height: imageSize.height)
    }
}
```

- [ ] **Step 4: Refactor InpaintView**

Make these exact changes in `Sources/ComfyBoxDesktop/Views/InpaintView.swift`:

1. Replace `@State private var strokes: [Stroke] = []` and `@State private var currentStroke: Stroke?` with `@State private var strokes = MaskStrokes()`. Delete the `struct Stroke` line.
2. Add after `@Binding var pendingImage: String?`:
   ```swift
   /// Strokes queued from the Edit tab's "Send to Inpaint"; consumed with `pendingImage`.
   @Binding var pendingMask: MaskStrokes?
   ```
   and add `.onChange(of: pendingMask) { _, _ in consumePending() }` next to the existing `onChange(of: pendingImage)`.
3. Brush buttons: `Button("Undo") { strokes.undoLast() }.controlSize(.small).disabled(strokes.isEmpty)` and `Button("Clear") { strokes.clear() }`.
4. In `canvas`, replace the mask `Canvas { … }` block and the `Color.clear … gesture` block (both positioned at `rect`) with:
   ```swift
   MaskCanvas(imageSize: rect.size, strokes: $strokes, brushPoints: brush, erase: erase)
       .position(x: rect.midX, y: rect.midY)
   ```
5. Replace the private `fitRect` with calls to `ImageFit.rect(imageSize:in:)` and delete `fitRect`.
6. `consumePending()` becomes:
   ```swift
   private func consumePending() {
       guard let p = pendingImage, !p.isEmpty else { return }
       let mask = pendingMask
       pendingImage = nil; pendingMask = nil
       load(p, initialStrokes: mask)
   }
   ```
   and `load(_ path: String)` becomes `load(_ path: String, initialStrokes: MaskStrokes? = nil)` with `strokes = initialStrokes ?? MaskStrokes()` in place of `strokes.removeAll()`.
7. Replace the whole `maskPNG()` function with:
   ```swift
   private func maskPNG() -> Data? { MaskRasterizer.pngData(strokes, size: pixelSize) }
   ```
8. In `ComfyBoxDesktopApp.swift` line 546, the call becomes
   `InpaintView(engine: engine, ingestor: ingestor, pendingImage: $pendingInpaintImage, pendingMask: $pendingInpaintMask)`
   and add `@State private var pendingInpaintMask: MaskStrokes?` beside `pendingInpaintImage` (line 38).

- [ ] **Step 5: Build and run the mask tests**

Run the build-only command, then the test command for `MaskRasterizerTests`. Expected: build succeeds with no errors, 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add Sources/ComfyBoxDesktop/Views/MaskCanvas.swift Sources/ComfyBoxDesktop/Views/InpaintView.swift Sources/ComfyBoxDesktop/ComfyBoxDesktopApp.swift Tests/ComfyBoxDesktopTests/MaskRasterizerTests.swift
git commit -m "refactor(desktop): extract MaskCanvas from Inpaint; accept a pending mask"
```

---

### Task 3: EditRecipe model

**Files:**
- Create: `Sources/ComfyBoxDesktop/Edit/EditRecipe.swift`
- Test: `Tests/ComfyBoxDesktopTests/EditRecipeTests.swift`

**Interfaces:**
- Produces:
  ```swift
  public struct CurvePoint: Codable, Equatable, Sendable { public var x: Double; public var y: Double }
  public struct ToneCurves: Codable, Equatable, Sendable {
      public var rgb: [CurvePoint] = []; public var r: [CurvePoint] = []; public var g: [CurvePoint] = []; public var b: [CurvePoint] = []
      public var isIdentity: Bool
      /// Sorted by x, endpoints (0,0)/(1,1) inserted if absent, clamped to 0…1.
      public static func normalized(_ pts: [CurvePoint]) -> [CurvePoint]
      /// Monotone-cubic sample of the normalized curve at `x`.
      public static func sample(_ pts: [CurvePoint], at x: Double) -> Double
  }
  public struct EditAdjustments: Codable, Equatable, Sendable {
      public var exposure = 0.0, contrast = 0.0, highlights = 0.0, shadows = 0.0, whites = 0.0, blacks = 0.0
      public var temperature = 0.0, tint = 0.0, vibrance = 0.0, saturation = 0.0
      public var sharpen = 0.0, noiseReduction = 0.0, vignette = 0.0
      public var curves = ToneCurves()
      public var isIdentity: Bool
      /// Copy with only the fields a local layer supports (exposure, contrast, highlights, shadows, temperature, tint, saturation, sharpen); others zeroed.
      public var restrictedToLocal: EditAdjustments
  }
  public struct EditGeometry: Codable, Equatable, Sendable {
      public var crop: CGRect? = nil; public var straightenDegrees = 0.0; public var quarterTurns = 0; public var flipH = false; public var flipV = false
      public var isIdentity: Bool
  }
  public struct EditLocalLayer: Codable, Equatable, Sendable {
      public var mask = MaskStrokes(); public var feather = 0.0; public var adjustments = EditAdjustments()
  }
  public struct EditSubject: Codable, Equatable, Sendable { public var removeBackground = false; public var invert = false }
  public struct EditRecipe: Codable, Equatable, Sendable {
      public static let currentVersion = 1
      public var version = EditRecipe.currentVersion
      public var geometry = EditGeometry(); public var adjustments = EditAdjustments()
      public var local: EditLocalLayer? = nil; public var subject = EditSubject()
      public init()
      public var isIdentity: Bool
  }
  ```

- [ ] **Step 1: Write the failing tests**

```swift
// EditRecipeTests.swift
import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("EditRecipe")
struct EditRecipeTests {
    @Test("default recipe is identity and version 1")
    func defaults() {
        let r = EditRecipe()
        #expect(r.isIdentity)
        #expect(r.version == 1)
        #expect(r.local == nil)
    }

    @Test("any change breaks identity")
    func nonIdentity() {
        var r = EditRecipe(); r.adjustments.exposure = 0.5; #expect(!r.isIdentity)
        var g = EditRecipe(); g.geometry.flipH = true; #expect(!g.isIdentity)
        var c = EditRecipe(); c.adjustments.curves.rgb = [CurvePoint(x: 0.5, y: 0.6)]; #expect(!c.isIdentity)
        var s = EditRecipe(); s.subject.removeBackground = true; #expect(!s.isIdentity)
        var l = EditRecipe(); l.local = EditLocalLayer(); #expect(!l.isIdentity)
    }

    @Test("JSON round trip preserves every field")
    func roundTrip() throws {
        var r = EditRecipe()
        r.geometry.crop = CGRect(x: 0.1, y: 0.2, width: 0.5, height: 0.6)
        r.geometry.straightenDegrees = -3.5; r.geometry.quarterTurns = 3; r.geometry.flipV = true
        r.adjustments.exposure = 1.25; r.adjustments.temperature = -0.4
        r.adjustments.curves.r = [CurvePoint(x: 0.25, y: 0.3)]
        var layer = EditLocalLayer(); layer.feather = 0.3; layer.adjustments.shadows = 0.5
        layer.mask.append(MaskStroke(points: [CGPoint(x: 0.1, y: 0.1)], size: 0.05, erase: false))
        r.local = layer
        r.subject = EditSubject(removeBackground: true, invert: true)
        let data = try JSONEncoder().encode(r)
        let back = try JSONDecoder().decode(EditRecipe.self, from: data)
        #expect(back == r)
    }

    @Test("curve normalization sorts, clamps, and inserts endpoints")
    func normalization() {
        let pts = [CurvePoint(x: 0.8, y: 1.4), CurvePoint(x: 0.2, y: -0.1)]
        let n = ToneCurves.normalized(pts)
        #expect(n.first == CurvePoint(x: 0, y: 0))
        #expect(n.last == CurvePoint(x: 1, y: 1))
        #expect(n[1] == CurvePoint(x: 0.2, y: 0))
        #expect(n[2] == CurvePoint(x: 0.8, y: 1))
        #expect(ToneCurves.normalized([]) == [CurvePoint(x: 0, y: 0), CurvePoint(x: 1, y: 1)])
    }

    @Test("curve sampling is identity when empty and interpolates through points")
    func sampling() {
        #expect(abs(ToneCurves.sample([], at: 0.37) - 0.37) < 1e-9)
        let pts = [CurvePoint(x: 0.5, y: 0.8)]
        #expect(abs(ToneCurves.sample(pts, at: 0.5) - 0.8) < 1e-9)
        #expect(ToneCurves.sample(pts, at: 0.25) > 0.25)     // lifted between 0 and the point
        #expect(ToneCurves.sample(pts, at: 0.0) == 0 && ToneCurves.sample(pts, at: 1.0) == 1)
    }

    @Test("restrictedToLocal zeroes unsupported fields")
    func restricted() {
        var a = EditAdjustments()
        a.exposure = 1; a.vibrance = 1; a.vignette = 1; a.blacks = 1; a.curves.rgb = [CurvePoint(x: 0.5, y: 0.7)]
        let r = a.restrictedToLocal
        #expect(r.exposure == 1)
        #expect(r.vibrance == 0 && r.vignette == 0 && r.blacks == 0 && r.curves.isIdentity)
    }
}
```

- [ ] **Step 2: Run to verify failure** — `-only-testing:ComfyBoxDesktopTests/EditRecipeTests`. Expected: compile error.

- [ ] **Step 3: Implement EditRecipe.swift**

```swift
// EditRecipe.swift — value model for a non-destructive edit
//
// Every field has a neutral default so `EditRecipe()` renders the source
// unchanged. Ranges are documented per field; the renderer (EditRenderer)
// owns the mapping from these unit ranges to Core Image parameters.

import Foundation
import CoreGraphics

public struct CurvePoint: Codable, Equatable, Sendable {
    public var x: Double
    public var y: Double
    public init(x: Double, y: Double) { self.x = x; self.y = y }
}

public struct ToneCurves: Codable, Equatable, Sendable {
    public var rgb: [CurvePoint] = []
    public var r: [CurvePoint] = []
    public var g: [CurvePoint] = []
    public var b: [CurvePoint] = []
    public init() {}

    public var isIdentity: Bool { rgb.isEmpty && r.isEmpty && g.isEmpty && b.isEmpty }

    public static func normalized(_ pts: [CurvePoint]) -> [CurvePoint] {
        var out = pts.map { CurvePoint(x: min(max($0.x, 0), 1), y: min(max($0.y, 0), 1)) }
            .sorted { $0.x < $1.x }
        if out.first?.x != 0 { out.insert(CurvePoint(x: 0, y: 0), at: 0) }
        if out.last?.x != 1 { out.append(CurvePoint(x: 1, y: 1)) }
        return out
    }

    /// Fritsch–Carlson monotone cubic interpolation through the normalized points.
    public static func sample(_ pts: [CurvePoint], at x: Double) -> Double {
        let p = normalized(pts)
        let n = p.count
        if n == 2 && p[0] == CurvePoint(x: 0, y: 0) && p[1] == CurvePoint(x: 1, y: 1) { return min(max(x, 0), 1) }
        let xs = p.map(\.x), ys = p.map(\.y)
        var d = [Double](repeating: 0, count: n - 1)
        for i in 0..<(n - 1) { let h = xs[i + 1] - xs[i]; d[i] = h > 0 ? (ys[i + 1] - ys[i]) / h : 0 }
        var m = [Double](repeating: 0, count: n)
        m[0] = d[0]; m[n - 1] = d[n - 2]
        for i in 1..<(n - 1) { m[i] = (d[i - 1] * d[i] <= 0) ? 0 : (d[i - 1] + d[i]) / 2 }
        for i in 0..<(n - 1) where d[i] != 0 {
            let a = m[i] / d[i], b = m[i + 1] / d[i]
            let s = a * a + b * b
            if s > 9 { let t = 3 / s.squareRoot(); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i] }
        }
        let cx = min(max(x, 0), 1)
        var i = 0
        while i < n - 2 && cx > xs[i + 1] { i += 1 }
        let h = xs[i + 1] - xs[i]
        guard h > 0 else { return ys[i] }
        let t = (cx - xs[i]) / h
        let t2 = t * t, t3 = t2 * t
        let h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2
        let y = h00 * ys[i] + h10 * h * m[i] + h01 * ys[i + 1] + h11 * h * m[i + 1]
        return min(max(y, 0), 1)
    }
}

public struct EditAdjustments: Codable, Equatable, Sendable {
    public var exposure = 0.0        // EV, −5…5
    public var contrast = 0.0        // −1…1
    public var highlights = 0.0      // −1…1
    public var shadows = 0.0         // −1…1
    public var whites = 0.0          // −1…1
    public var blacks = 0.0          // −1…1
    public var temperature = 0.0     // −1…1 (cool…warm)
    public var tint = 0.0            // −1…1 (green…magenta)
    public var vibrance = 0.0        // −1…1
    public var saturation = 0.0      // −1…1
    public var sharpen = 0.0         // 0…1
    public var noiseReduction = 0.0  // 0…1
    public var vignette = 0.0        // 0…1
    public var curves = ToneCurves()
    public init() {}

    public var isIdentity: Bool { self == EditAdjustments() }

    public var restrictedToLocal: EditAdjustments {
        var r = EditAdjustments()
        r.exposure = exposure; r.contrast = contrast; r.highlights = highlights; r.shadows = shadows
        r.temperature = temperature; r.tint = tint; r.saturation = saturation; r.sharpen = sharpen
        return r
    }
}

public struct EditGeometry: Codable, Equatable, Sendable {
    /// Normalized crop in source coordinates, origin top-left. nil = full frame.
    public var crop: CGRect? = nil
    public var straightenDegrees = 0.0   // −45…45
    public var quarterTurns = 0          // 0…3 clockwise
    public var flipH = false
    public var flipV = false
    public init() {}
    public var isIdentity: Bool { self == EditGeometry() }
}

public struct EditLocalLayer: Codable, Equatable, Sendable {
    public var mask = MaskStrokes()
    public var feather = 0.0             // 0…1 → blur radius up to 5 % of the shorter side
    public var adjustments = EditAdjustments()
    public init() {}
}

public struct EditSubject: Codable, Equatable, Sendable {
    public var removeBackground = false
    public var invert = false
    public init(removeBackground: Bool = false, invert: Bool = false) {
        self.removeBackground = removeBackground; self.invert = invert
    }
}

public struct EditRecipe: Codable, Equatable, Sendable {
    public static let currentVersion = 1
    public var version = EditRecipe.currentVersion
    public var geometry = EditGeometry()
    public var adjustments = EditAdjustments()
    public var local: EditLocalLayer? = nil
    public var subject = EditSubject()
    public init() {}

    public var isIdentity: Bool {
        geometry.isIdentity && adjustments.isIdentity && local == nil && subject == EditSubject()
    }
}
```

- [ ] **Step 4: Run to verify pass** — Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/ComfyBoxDesktop/Edit/EditRecipe.swift Tests/ComfyBoxDesktopTests/EditRecipeTests.swift
git commit -m "feat(desktop): EditRecipe value model with monotone tone curves"
```

---

### Task 4: EditRenderer — parameter mappings, geometry, global adjustments

**Files:**
- Create: `Sources/ComfyBoxDesktop/Edit/EditRenderer.swift`
- Test: `Tests/ComfyBoxDesktopTests/EditRendererTests.swift`

**Interfaces:**
- Consumes: `EditRecipe`, `EditGeometry`, `EditAdjustments`, `ToneCurves` (Task 3).
- Produces:
  ```swift
  public enum EditRenderer {
      public static func render(source: CIImage, recipe: EditRecipe, subjectMask: CIImage?) -> CIImage
      // Stage functions, each pure:
      static func applyGeometry(_ image: CIImage, _ g: EditGeometry) -> CIImage
      static func applyAdjustments(_ image: CIImage, _ a: EditAdjustments) -> CIImage
      // Mappings (pinned by tests):
      static func contrastParameter(_ v: Double) -> Double        // 1 + 0.5v  → 0.5…1.5
      static func saturationParameter(_ v: Double) -> Double      // 1 + v     → 0…2
      static func temperatureTarget(_ v: Double) -> CGFloat       // 6500 + sign * v * 3000, sign fixed by the warm test
      static func tintTarget(_ v: Double) -> CGFloat              // v * 150
      static func highlightAmount(_ v: Double) -> Double          // 1 + 0.7 * min(v, 0)   (negative recovers)
      static func shadowAmount(_ v: Double) -> Double             // v
      static func whitesBlacksCurve(whites: Double, blacks: Double, highlights: Double) -> [CurvePoint]
      static func largestInscribedSize(width: CGFloat, height: CGFloat, angleRadians: CGFloat) -> CGSize
  }
  ```
  `render` in this task implements geometry + global adjustments and returns early ignoring `local`/`subject`; Task 5 completes it.

- [ ] **Step 1: Write the failing tests**

```swift
// EditRendererTests.swift
import Testing
import Foundation
import CoreImage
import CoreGraphics
@testable import ComfyBoxDesktop

@Suite("EditRenderer")
struct EditRendererTests {
    static let context = CIContext(options: [.useSoftwareRenderer: true,
                                             .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                                             .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])

    func rendered(_ source: CGImage, _ recipe: EditRecipe, mask: CIImage? = nil) -> CGImage {
        let out = EditRenderer.render(source: CIImage(cgImage: source), recipe: recipe, subjectMask: mask)
        return Self.context.createCGImage(out, from: out.extent)!
    }

    @Test("identity recipe is pixel-identical")
    func identity() {
        let src = EditTestSupport.horizontalGradient(width: 64, height: 32)
        let out = rendered(src, EditRecipe())
        #expect(out.width == 64 && out.height == 32)
        for x in [0, 17, 40, 63] {
            #expect(abs(Int(EditTestSupport.gray(out, x: x, y: 10)) - Int(EditTestSupport.gray(src, x: x, y: 10))) <= 1)
        }
    }

    @Test("exposure +1 brightens mid grey")
    func exposure() {
        let src = EditTestSupport.solid(r: 100, g: 100, b: 100, width: 16, height: 16)
        var r = EditRecipe(); r.adjustments.exposure = 1
        #expect(EditTestSupport.gray(rendered(src, r), x: 8, y: 8) > 150)
    }

    @Test("positive temperature warms: red exceeds blue on grey")
    func warm() {
        let src = EditTestSupport.solid(r: 128, g: 128, b: 128, width: 16, height: 16)
        var r = EditRecipe(); r.adjustments.temperature = 1
        let p = EditTestSupport.pixel(rendered(src, r), x: 8, y: 8)
        #expect(p.r > p.b + 10)
    }

    @Test("saturation -1 makes a red image grey")
    func desaturate() {
        let src = EditTestSupport.solid(r: 200, g: 40, b: 40, width: 16, height: 16)
        var r = EditRecipe(); r.adjustments.saturation = -1
        let p = EditTestSupport.pixel(rendered(src, r), x: 8, y: 8)
        #expect(abs(Int(p.r) - Int(p.g)) < 8 && abs(Int(p.g) - Int(p.b)) < 8)
    }

    @Test("rgb curve lifting the midpoint brightens mid grey and leaves black/white")
    func curve() {
        let src = EditTestSupport.horizontalGradient(width: 256, height: 4)
        var r = EditRecipe(); r.adjustments.curves.rgb = [CurvePoint(x: 0.5, y: 0.75)]
        let out = rendered(src, r)
        #expect(EditTestSupport.gray(out, x: 128, y: 1) > 170)
        #expect(EditTestSupport.gray(out, x: 0, y: 1) < 4)
        #expect(EditTestSupport.gray(out, x: 255, y: 1) > 251)
    }

    @Test("flipH mirrors the gradient")
    func flip() {
        let src = EditTestSupport.horizontalGradient(width: 64, height: 8)
        var r = EditRecipe(); r.geometry.flipH = true
        let out = rendered(src, r)
        #expect(EditTestSupport.gray(out, x: 0, y: 4) > 240 && EditTestSupport.gray(out, x: 63, y: 4) < 15)
    }

    @Test("one quarter turn swaps dimensions and moves the bright edge to the bottom")
    func rotate() {
        let src = EditTestSupport.horizontalGradient(width: 64, height: 32)
        var r = EditRecipe(); r.geometry.quarterTurns = 1
        let out = rendered(src, r)
        #expect(out.width == 32 && out.height == 64)
        // Clockwise: the right (bright) edge becomes the bottom edge.
        #expect(EditTestSupport.gray(out, x: 16, y: 63) > 240 && EditTestSupport.gray(out, x: 16, y: 0) < 15)
    }

    @Test("crop yields the expected size and region")
    func crop() {
        let src = EditTestSupport.horizontalGradient(width: 100, height: 50)
        var r = EditRecipe(); r.geometry.crop = CGRect(x: 0.5, y: 0.0, width: 0.5, height: 1.0)
        let out = rendered(src, r)
        #expect(out.width == 50 && out.height == 50)
        #expect(EditTestSupport.gray(out, x: 0, y: 25) > 120)   // right half of the gradient
    }

    @Test("straighten leaves no transparent corners and shrinks the frame")
    func straighten() {
        let src = EditTestSupport.solid(r: 90, g: 90, b: 90, width: 120, height: 80)
        var r = EditRecipe(); r.geometry.straightenDegrees = 10
        let out = rendered(src, r)
        #expect(out.width < 120 && out.height < 80)
        for (x, y) in [(0, 0), (out.width - 1, 0), (0, out.height - 1), (out.width - 1, out.height - 1)] {
            #expect(EditTestSupport.pixel(out, x: x, y: y).a == 255)
        }
    }

    @Test("parameter mappings pin their endpoints")
    func mappings() {
        #expect(EditRenderer.contrastParameter(-1) == 0.5 && EditRenderer.contrastParameter(1) == 1.5)
        #expect(EditRenderer.saturationParameter(-1) == 0 && EditRenderer.saturationParameter(0) == 1)
        #expect(EditRenderer.tintTarget(1) == 150)
        #expect(EditRenderer.highlightAmount(-1) == 0.3 && EditRenderer.highlightAmount(1) == 1)
        #expect(EditRenderer.shadowAmount(0.5) == 0.5)
        let s = EditRenderer.largestInscribedSize(width: 100, height: 100, angleRadians: 0)
        #expect(s.width == 100 && s.height == 100)
        let t = EditRenderer.largestInscribedSize(width: 100, height: 50, angleRadians: .pi / 18)
        #expect(t.width < 100 && t.height < 50 && t.width > 60)
    }
}
```

- [ ] **Step 2: Run to verify failure** — `-only-testing:ComfyBoxDesktopTests/EditRendererTests`. Expected: compile error.

- [ ] **Step 3: Implement EditRenderer.swift (geometry + global)**

```swift
// EditRenderer.swift — pure Core Image pipeline for EditRecipe
//
// Order: geometry → global adjustments → local layer → subject alpha.
// No I/O, no caching, no main-actor requirement. Every recipe→filter
// mapping is a small static func so tests can pin it.

import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreGraphics

public enum EditRenderer {

    public static func render(source: CIImage, recipe: EditRecipe, subjectMask: CIImage?) -> CIImage {
        var image = applyGeometry(source, recipe.geometry)
        image = applyAdjustments(image, recipe.adjustments)
        image = applyLocalLayer(image, recipe.local)
        image = applySubject(image, recipe.subject, mask: subjectMask, geometry: recipe.geometry, sourceExtent: source.extent)
        return image
    }

    // MARK: - Geometry

    static func applyGeometry(_ input: CIImage, _ g: EditGeometry) -> CIImage {
        var image = input
        let turns = ((g.quarterTurns % 4) + 4) % 4
        if turns > 0 {
            // CI rotates counter-clockwise for positive angles; quarterTurns is clockwise.
            image = image.transformed(by: CGAffineTransform(rotationAngle: -CGFloat(turns) * .pi / 2))
            image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        }
        if g.flipH {
            image = image.transformed(by: CGAffineTransform(scaleX: -1, y: 1))
            image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: 0))
        }
        if g.flipV {
            image = image.transformed(by: CGAffineTransform(scaleX: 1, y: -1))
            image = image.transformed(by: CGAffineTransform(translationX: 0, y: -image.extent.minY))
        }
        if abs(g.straightenDegrees) > 0.001 {
            let angle = CGFloat(g.straightenDegrees) * .pi / 180
            let w = image.extent.width, h = image.extent.height
            let center = CGPoint(x: image.extent.midX, y: image.extent.midY)
            var t = CGAffineTransform(translationX: center.x, y: center.y)
            t = t.rotated(by: -angle)          // clockwise for positive degrees
            t = t.translatedBy(x: -center.x, y: -center.y)
            let rotated = image.transformed(by: t)
            let fit = largestInscribedSize(width: w, height: h, angleRadians: angle)
            let cropRect = CGRect(x: center.x - fit.width / 2, y: center.y - fit.height / 2,
                                  width: fit.width, height: fit.height).integral
            image = rotated.cropped(to: cropRect)
            image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        }
        if let c = g.crop {
            let w = image.extent.width, h = image.extent.height
            // Normalized crop has a top-left origin; CI extents are bottom-up.
            let rect = CGRect(x: (c.minX * w).rounded(), y: ((1 - c.maxY) * h).rounded(),
                              width: (c.width * w).rounded(), height: (c.height * h).rounded())
            image = image.cropped(to: rect.intersection(image.extent))
            image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        }
        return image
    }

    /// Largest axis-aligned rectangle with the source's aspect that fits inside the source rotated by `angle`.
    static func largestInscribedSize(width w: CGFloat, height h: CGFloat, angleRadians: CGFloat) -> CGSize {
        let s = abs(sin(angleRadians)), c = abs(cos(angleRadians))
        if s < 1e-9 { return CGSize(width: w, height: h) }
        let widthIsLonger = w >= h
        let side = widthIsLonger ? h : w
        let long = widthIsLonger ? w : h
        if side <= 2 * s * c * long || abs(s - c) < 1e-9 {
            let x = 0.5 * side
            return widthIsLonger ? CGSize(width: x / s, height: x / c) : CGSize(width: x / c, height: x / s)
        }
        let cos2 = c * c - s * s
        return CGSize(width: (w * c - h * s) / cos2, height: (h * c - w * s) / cos2)
    }

    // MARK: - Global adjustments

    static func applyAdjustments(_ input: CIImage, _ a: EditAdjustments) -> CIImage {
        var image = input
        if a.exposure != 0 {
            let f = CIFilter.exposureAdjust(); f.inputImage = image; f.ev = Float(a.exposure); image = f.outputImage ?? image
        }
        if a.temperature != 0 || a.tint != 0 {
            let f = CIFilter.temperatureAndTint(); f.inputImage = image
            f.neutral = CIVector(x: 6500, y: 0)
            f.targetNeutral = CIVector(x: temperatureTarget(a.temperature), y: tintTarget(a.tint))
            image = f.outputImage ?? image
        }
        if a.highlights < 0 || a.shadows != 0 {
            let f = CIFilter.highlightShadowAdjust(); f.inputImage = image
            f.highlightAmount = Float(highlightAmount(a.highlights)); f.shadowAmount = Float(shadowAmount(a.shadows))
            f.radius = 3
            image = f.outputImage ?? image
        }
        if a.whites != 0 || a.blacks != 0 || a.highlights > 0 {
            image = toneCurve(image, whitesBlacksCurve(whites: a.whites, blacks: a.blacks, highlights: a.highlights))
        }
        if a.contrast != 0 || a.saturation != 0 {
            let f = CIFilter.colorControls(); f.inputImage = image
            f.contrast = Float(contrastParameter(a.contrast)); f.saturation = Float(saturationParameter(a.saturation)); f.brightness = 0
            image = f.outputImage ?? image
        }
        if a.vibrance != 0 {
            let f = CIFilter.vibrance(); f.inputImage = image; f.amount = Float(a.vibrance); image = f.outputImage ?? image
        }
        if !a.curves.isIdentity { image = applyCurves(image, a.curves) }
        if a.sharpen > 0 {
            let f = CIFilter.sharpenLuminance(); f.inputImage = image; f.sharpness = Float(a.sharpen * 2); f.radius = 1.69
            image = f.outputImage?.cropped(to: input.extent) ?? image
        }
        if a.noiseReduction > 0 {
            let f = CIFilter.noiseReduction(); f.inputImage = image; f.noiseLevel = Float(a.noiseReduction * 0.1); f.sharpness = 0.4
            image = f.outputImage?.cropped(to: input.extent) ?? image
        }
        if a.vignette > 0 {
            let f = CIFilter.vignette(); f.inputImage = image; f.intensity = Float(a.vignette); f.radius = 1.5
            image = f.outputImage ?? image
        }
        return image.cropped(to: input.extent)
    }

    static func contrastParameter(_ v: Double) -> Double { 1 + 0.5 * v }
    static func saturationParameter(_ v: Double) -> Double { 1 + v }
    /// Sign chosen so positive = warmer; the `warm` test pins the direction. If it fails, flip the sign here.
    static func temperatureTarget(_ v: Double) -> CGFloat { CGFloat(6500 - v * 3000) }
    static func tintTarget(_ v: Double) -> CGFloat { CGFloat(v * 150) }
    static func highlightAmount(_ v: Double) -> Double { 1 + 0.7 * min(v, 0) }
    static func shadowAmount(_ v: Double) -> Double { v }

    /// Blacks move the 0.25 point, whites the 0.75 point, positive highlights lift 0.75 further.
    static func whitesBlacksCurve(whites: Double, blacks: Double, highlights: Double) -> [CurvePoint] {
        let hi = max(highlights, 0)
        return [CurvePoint(x: 0, y: 0),
                CurvePoint(x: 0.25, y: min(max(0.25 + blacks * 0.15, 0), 1)),
                CurvePoint(x: 0.5, y: 0.5),
                CurvePoint(x: 0.75, y: min(max(0.75 + whites * 0.15 + hi * 0.1, 0), 1)),
                CurvePoint(x: 1, y: 1)]
    }

    /// Five-point CIToneCurve sampled from an arbitrary control-point curve.
    static func toneCurve(_ image: CIImage, _ pts: [CurvePoint]) -> CIImage {
        let f = CIFilter.toneCurve(); f.inputImage = image
        let xs: [Double] = [0, 0.25, 0.5, 0.75, 1]
        let ys = xs.map { ToneCurves.sample(pts, at: $0) }
        f.point0 = CGPoint(x: xs[0], y: ys[0]); f.point1 = CGPoint(x: xs[1], y: ys[1]); f.point2 = CGPoint(x: xs[2], y: ys[2])
        f.point3 = CGPoint(x: xs[3], y: ys[3]); f.point4 = CGPoint(x: xs[4], y: ys[4])
        return f.outputImage ?? image
    }

    static func applyCurves(_ input: CIImage, _ c: ToneCurves) -> CIImage {
        var image = input
        if !c.rgb.isEmpty { image = toneCurve(image, c.rgb) }
        for (channel, pts) in [(0, c.r), (1, c.g), (2, c.b)] where !pts.isEmpty {
            let curved = toneCurve(image, pts)
            image = replaceChannel(of: image, with: curved, channel: channel)
        }
        return image
    }

    /// Take `channel` (0=r,1=g,2=b) from `donor`, the other two from `base`. Alpha from base.
    static func replaceChannel(of base: CIImage, with donor: CIImage, channel: Int) -> CIImage {
        func vec(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat) -> CIVector { CIVector(x: r, y: g, z: b, w: a) }
        let keep = CIFilter.colorMatrix(); keep.inputImage = base
        let take = CIFilter.colorMatrix(); take.inputImage = donor
        keep.rVector = vec(channel == 0 ? 0 : 1, 0, 0, 0); take.rVector = vec(channel == 0 ? 1 : 0, 0, 0, 0)
        keep.gVector = vec(0, channel == 1 ? 0 : 1, 0, 0); take.gVector = vec(0, channel == 1 ? 1 : 0, 0, 0)
        keep.bVector = vec(0, 0, channel == 2 ? 0 : 1, 0); take.bVector = vec(0, 0, channel == 2 ? 1 : 0, 0)
        keep.aVector = vec(0, 0, 0, 1); take.aVector = vec(0, 0, 0, 0)
        let add = CIFilter.additionCompositing()
        add.inputImage = take.outputImage; add.backgroundImage = keep.outputImage
        return add.outputImage?.cropped(to: base.extent) ?? base
    }

    // MARK: - Local layer and subject (completed in Task 5)

    static func applyLocalLayer(_ image: CIImage, _ layer: EditLocalLayer?) -> CIImage { image }

    static func applySubject(_ image: CIImage, _ subject: EditSubject, mask: CIImage?,
                             geometry: EditGeometry, sourceExtent: CGRect) -> CIImage { image }
}
```

- [ ] **Step 4: Run to verify pass**

Expected: 10 tests pass. If `warm` fails with `p.r < p.b`, flip the sign in `temperatureTarget` to `6500 + v * 3000` and re-run; do not change the test. If `rotate` fails on which edge is bright, swap the sign of the rotation angle in `applyGeometry` (quarter turns) — the test defines clockwise.

- [ ] **Step 5: Commit**

```bash
git add Sources/ComfyBoxDesktop/Edit/EditRenderer.swift Tests/ComfyBoxDesktopTests/EditRendererTests.swift
git commit -m "feat(desktop): EditRenderer geometry and global adjustments"
```

---

### Task 5: EditRenderer — local layer and subject alpha

**Files:**
- Modify: `Sources/ComfyBoxDesktop/Edit/EditRenderer.swift` (replace the two stubs at the bottom)
- Test: `Tests/ComfyBoxDesktopTests/EditRendererTests.swift` (append)

**Interfaces:**
- Consumes: `MaskRasterizer.render` (Task 1), `EditLocalLayer`, `EditSubject` (Task 3).
- Produces: completed `EditRenderer.render`. The `subjectMask` passed to `render` is a mask at **source** resolution with white = subject; `applySubject` runs it through the same `applyGeometry` so it lines up with the cropped output.

- [ ] **Step 1: Append failing tests**

```swift
    @Test("local layer with a left-half mask brightens only the masked half")
    func localLayer() {
        let src = EditTestSupport.solid(r: 100, g: 100, b: 100, width: 100, height: 40)
        var r = EditRecipe()
        var layer = EditLocalLayer()
        // A wide vertical stroke covering x in 0…0.5 (size is a fraction of width).
        layer.mask.append(MaskStroke(points: [CGPoint(x: 0.25, y: 0.0), CGPoint(x: 0.25, y: 1.0)], size: 0.5, erase: false))
        layer.adjustments.exposure = 1.5
        r.local = layer
        let out = rendered(src, r)
        #expect(EditTestSupport.gray(out, x: 10, y: 20) > 150)
        #expect(abs(Int(EditTestSupport.gray(out, x: 90, y: 20)) - 100) <= 2)
    }

    @Test("local layer ignores fields outside the local subset")
    func localRestricted() {
        let src = EditTestSupport.solid(r: 100, g: 100, b: 100, width: 40, height: 40)
        var r = EditRecipe()
        var layer = EditLocalLayer()
        layer.mask.append(MaskStroke(points: [CGPoint(x: 0.5, y: 0.5)], size: 2, erase: false))
        layer.adjustments.vignette = 1   // not in the local subset
        r.local = layer
        let out = rendered(src, r)
        #expect(abs(Int(EditTestSupport.gray(out, x: 2, y: 2)) - 100) <= 2)
    }

    @Test("subject removal makes the unmasked region transparent; invert flips it")
    func subject() {
        let src = EditTestSupport.solid(r: 50, g: 120, b: 200, width: 40, height: 40)
        // Mask: left half white (subject), right half black.
        var strokes = MaskStrokes()
        strokes.append(MaskStroke(points: [CGPoint(x: 0.25, y: 0), CGPoint(x: 0.25, y: 1)], size: 0.5, erase: false))
        let mask = CIImage(cgImage: MaskRasterizer.render(strokes, size: CGSize(width: 40, height: 40))!)
        var r = EditRecipe(); r.subject.removeBackground = true
        let out = rendered(src, r, mask: mask)
        #expect(EditTestSupport.pixel(out, x: 5, y: 20).a == 255)
        #expect(EditTestSupport.pixel(out, x: 35, y: 20).a == 0)
        r.subject.invert = true
        let inv = rendered(src, r, mask: mask)
        #expect(EditTestSupport.pixel(inv, x: 5, y: 20).a == 0)
        #expect(EditTestSupport.pixel(inv, x: 35, y: 20).a == 255)
    }

    @Test("subject flag without a mask is a no-op")
    func subjectNoMask() {
        let src = EditTestSupport.solid(r: 50, g: 120, b: 200, width: 8, height: 8)
        var r = EditRecipe(); r.subject.removeBackground = true
        #expect(EditTestSupport.pixel(rendered(src, r), x: 4, y: 4).a == 255)
    }

    @Test("subject mask follows geometry: flipH moves the transparent half")
    func subjectFollowsGeometry() {
        let src = EditTestSupport.solid(r: 50, g: 120, b: 200, width: 40, height: 40)
        var strokes = MaskStrokes()
        strokes.append(MaskStroke(points: [CGPoint(x: 0.25, y: 0), CGPoint(x: 0.25, y: 1)], size: 0.5, erase: false))
        let mask = CIImage(cgImage: MaskRasterizer.render(strokes, size: CGSize(width: 40, height: 40))!)
        var r = EditRecipe(); r.subject.removeBackground = true; r.geometry.flipH = true
        let out = rendered(src, r, mask: mask)
        #expect(EditTestSupport.pixel(out, x: 35, y: 20).a == 255)
        #expect(EditTestSupport.pixel(out, x: 5, y: 20).a == 0)
    }
```

- [ ] **Step 2: Run to verify failure** — Expected: the five new tests fail (stubs return input).

- [ ] **Step 3: Replace the stubs**

```swift
    // MARK: - Local layer

    static func applyLocalLayer(_ image: CIImage, _ layer: EditLocalLayer?) -> CIImage {
        guard let layer, !layer.mask.isEmpty else { return image }
        let extent = image.extent
        guard let maskCG = MaskRasterizer.render(layer.mask, size: extent.size) else { return image }
        var mask = CIImage(cgImage: maskCG)
        if layer.feather > 0 {
            let radius = Float(layer.feather * 0.05 * Double(min(extent.width, extent.height)))
            let blur = CIFilter.gaussianBlur(); blur.inputImage = mask.clampedToExtent(); blur.radius = radius
            mask = blur.outputImage?.cropped(to: extent) ?? mask
        }
        let adjusted = applyAdjustments(image, layer.adjustments.restrictedToLocal)
        let blend = CIFilter.blendWithMask()
        blend.inputImage = adjusted; blend.backgroundImage = image; blend.maskImage = mask
        return blend.outputImage?.cropped(to: extent) ?? image
    }

    // MARK: - Subject alpha

    static func applySubject(_ image: CIImage, _ subject: EditSubject, mask: CIImage?,
                             geometry: EditGeometry, sourceExtent: CGRect) -> CIImage {
        guard subject.removeBackground, let mask else { return image }
        // The mask is at source resolution; run it through the same geometry so it aligns.
        var m = applyGeometry(mask, geometry)
        if m.extent != image.extent {
            m = m.transformed(by: CGAffineTransform(scaleX: image.extent.width / max(m.extent.width, 1),
                                                    y: image.extent.height / max(m.extent.height, 1)))
                .cropped(to: image.extent)
        }
        if subject.invert {
            let inv = CIFilter.colorInvert(); inv.inputImage = m; m = inv.outputImage ?? m
        }
        let clear = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0)).cropped(to: image.extent)
        let blend = CIFilter.blendWithMask()
        blend.inputImage = image; blend.backgroundImage = clear; blend.maskImage = m
        return blend.outputImage?.cropped(to: image.extent) ?? image
    }
```

- [ ] **Step 4: Run the whole EditRenderer suite** — Expected: 15 tests pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/ComfyBoxDesktop/Edit/EditRenderer.swift Tests/ComfyBoxDesktopTests/EditRendererTests.swift
git commit -m "feat(desktop): EditRenderer local masked layer and subject alpha"
```

---

### Task 6: SubjectMasker (Vision)

**Files:**
- Create: `Sources/ComfyBoxDesktop/Edit/SubjectMasker.swift`

**Interfaces:**
- Produces:
  ```swift
  public enum SubjectMaskError: Error, Equatable { case noSubject; case visionFailed(String) }
  public actor SubjectMasker {
      public init()
      /// Single-channel mask at `source` resolution; white = subject. Cached by `cacheKey`.
      public func mask(for source: CGImage, cacheKey: String) async throws -> CIImage
      public func invalidate(cacheKey: String)
  }
  ```
- No unit test (Vision needs a real subject). Verified by build + Todd's live check.

- [ ] **Step 1: Implement**

```swift
// SubjectMasker.swift — Vision foreground mask for background removal
//
// Wraps VNGenerateForegroundInstanceMaskRequest (macOS 14). Returns a
// source-resolution single-channel CIImage, white = subject. Cached per
// source path for the session so toggling Remove Background is free.

import Foundation
import CoreImage
import CoreGraphics
import Vision

public enum SubjectMaskError: Error, Equatable {
    case noSubject
    case visionFailed(String)
}

public actor SubjectMasker {
    private var cache: [String: CIImage] = [:]

    public init() {}

    public func mask(for source: CGImage, cacheKey: String) async throws -> CIImage {
        if let hit = cache[cacheKey] { return hit }
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: source, options: [:])
        do { try handler.perform([request]) } catch { throw SubjectMaskError.visionFailed(error.localizedDescription) }
        guard let result = request.results?.first, !result.allInstances.isEmpty else { throw SubjectMaskError.noSubject }
        let buffer: CVPixelBuffer
        do {
            buffer = try result.generateScaledMaskForImage(forInstances: result.allInstances, from: handler)
        } catch { throw SubjectMaskError.visionFailed(error.localizedDescription) }
        var image = CIImage(cvPixelBuffer: buffer)
        // Vision returns a float mask sized to the source; force exact extent.
        let target = CGRect(x: 0, y: 0, width: source.width, height: source.height)
        if image.extent.size != target.size {
            image = image.transformed(by: CGAffineTransform(scaleX: target.width / image.extent.width,
                                                            y: target.height / image.extent.height))
        }
        image = image.cropped(to: target)
        cache[cacheKey] = image
        return image
    }

    public func invalidate(cacheKey: String) { cache[cacheKey] = nil }
}
```

- [ ] **Step 2: Build** — run the build-only command. Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add Sources/ComfyBoxDesktop/Edit/SubjectMasker.swift
git commit -m "feat(desktop): SubjectMasker wraps Vision foreground mask"
```

---

### Task 7: EditSidecar and EditExporter

**Files:**
- Create: `Sources/ComfyBoxDesktop/Edit/EditSidecar.swift`
- Create: `Sources/ComfyBoxDesktop/Edit/EditExporter.swift`
- Test: `Tests/ComfyBoxDesktopTests/EditSidecarTests.swift`, `Tests/ComfyBoxDesktopTests/EditExporterTests.swift`

**Interfaces:**
- Consumes: `EditRecipe`, `EditRenderer`, `DAMAsset`, `AssetIngestor.ingestFile(at:)`.
- Produces:
  ```swift
  public struct EditSidecar: Codable, Equatable, Sendable {
      public var version: Int; public var sourcePath: String; public var sourceAssetId: String?
      public var recipe: EditRecipe; public var editor: String; public var createdAt: Date
      /// Reads the `edit` block from `<image basename>.json`. nil when absent or unparsable.
      public static func read(forImageAt imagePath: String) -> EditSidecar?
      /// Follows `edit.source_path` chains to the root original. Returns (rootPath, rootAssetId).
      public static func rootSource(forImageAt imagePath: String) -> (path: String, assetId: String?)
      /// JSON dictionary for the `edit` key.
      public var jsonObject: [String: Any]
  }
  public enum EditExportError: Error { case renderFailed; case writeFailed(String) }
  public enum EditExporter {
      /// Pure: the full sidecar dictionary that will be written.
      public static func sidecarObject(source: DAMAsset?, sourcePath: String, recipe: EditRecipe, now: Date) -> [String: Any]
      /// Next free `edit-<seconds>.png` path in `directory` (suffix -2, -3 … on collision).
      public static func outputPath(in directory: String, seconds: Int) -> String
      /// Renders, writes PNG then sidecar, ingests. Returns the PNG path.
      public static func export(sourceImage: CGImage, sourcePath: String, sourceAsset: DAMAsset?,
                                recipe: EditRecipe, subjectMask: CIImage?,
                                outputDirectory: String, ingestor: AssetIngestor?) async throws -> String
  }
  ```

- [ ] **Step 1: Write failing sidecar tests**

```swift
// EditSidecarTests.swift
import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("EditSidecar")
struct EditSidecarTests {
    func tempDir() -> String {
        let d = NSTemporaryDirectory() + "editsidecar-" + UUID().uuidString
        try! FileManager.default.createDirectory(atPath: d, withIntermediateDirectories: true)
        return d
    }

    @Test("reads an edit block and ignores files without one")
    func readBlock() throws {
        let dir = tempDir()
        var recipe = EditRecipe(); recipe.adjustments.exposure = 0.7
        let sc = EditSidecar(version: 1, sourcePath: "/orig/a.png", sourceAssetId: "A1", recipe: recipe, editor: "ComfyBoxDesktop", createdAt: Date(timeIntervalSince1970: 1_700_000_000))
        let json: [String: Any] = ["prompt": "x", "edit": sc.jsonObject]
        try JSONSerialization.data(withJSONObject: json).write(to: URL(fileURLWithPath: dir + "/edit-1.json"))
        let back = EditSidecar.read(forImageAt: dir + "/edit-1.png")
        #expect(back?.sourcePath == "/orig/a.png")
        #expect(back?.sourceAssetId == "A1")
        #expect(back?.recipe == recipe)
        try JSONSerialization.data(withJSONObject: ["prompt": "y"]).write(to: URL(fileURLWithPath: dir + "/plain.json"))
        #expect(EditSidecar.read(forImageAt: dir + "/plain.png") == nil)
        #expect(EditSidecar.read(forImageAt: dir + "/missing.png") == nil)
    }

    @Test("rootSource follows the chain to the original")
    func root() throws {
        let dir = tempDir()
        let first = EditSidecar(version: 1, sourcePath: "/orig/a.png", sourceAssetId: "A1", recipe: EditRecipe(), editor: "ComfyBoxDesktop", createdAt: Date())
        try JSONSerialization.data(withJSONObject: ["edit": first.jsonObject]).write(to: URL(fileURLWithPath: dir + "/edit-1.json"))
        let second = EditSidecar(version: 1, sourcePath: dir + "/edit-1.png", sourceAssetId: "E1", recipe: EditRecipe(), editor: "ComfyBoxDesktop", createdAt: Date())
        try JSONSerialization.data(withJSONObject: ["edit": second.jsonObject]).write(to: URL(fileURLWithPath: dir + "/edit-2.json"))
        let r = EditSidecar.rootSource(forImageAt: dir + "/edit-2.png")
        #expect(r.path == "/orig/a.png" && r.assetId == "A1")
        let plain = EditSidecar.rootSource(forImageAt: "/orig/a.png")
        #expect(plain.path == "/orig/a.png" && plain.assetId == nil)
    }

    @Test("a newer version still parses so the caller can decide to drop it")
    func newerVersion() throws {
        let dir = tempDir()
        var obj = EditSidecar(version: 1, sourcePath: "/o.png", sourceAssetId: nil, recipe: EditRecipe(), editor: "x", createdAt: Date()).jsonObject
        obj["version"] = 99
        try JSONSerialization.data(withJSONObject: ["edit": obj]).write(to: URL(fileURLWithPath: dir + "/e.json"))
        #expect(EditSidecar.read(forImageAt: dir + "/e.png")?.version == 99)
    }
}
```

- [ ] **Step 2: Write failing exporter tests**

```swift
// EditExporterTests.swift
import Testing
import Foundation
import CoreGraphics
import ImageIO
@testable import ComfyBoxDesktop

@Suite("EditExporter")
struct EditExporterTests {
    func tempDir() -> String {
        let d = NSTemporaryDirectory() + "editexport-" + UUID().uuidString
        try! FileManager.default.createDirectory(atPath: d, withIntermediateDirectories: true)
        return d
    }

    @Test("sidecarObject copies generation fields and writes the edit block")
    func sidecarObject() {
        let asset = DAMAsset(id: "A1", filename: "a.png", absolutePath: "/orig/a.png", prompt: "a cat", negativePrompt: "dog",
                             seed: 42, steps: 9, guidance: 3.5, modelFamily: "krea2", contentMode: "apple", characterName: "Kira")
        var recipe = EditRecipe(); recipe.adjustments.contrast = 0.3
        let obj = EditExporter.sidecarObject(source: asset, sourcePath: "/orig/a.png", recipe: recipe, now: Date(timeIntervalSince1970: 0))
        #expect(obj["prompt"] as? String == "a cat")
        #expect(obj["negative_prompt"] as? String == "dog")
        #expect(obj["seed"] as? Int == 42 && obj["steps"] as? Int == 9)
        #expect(obj["guidance"] as? Double == 3.5)
        #expect(obj["model_family"] as? String == "krea2")
        #expect(obj["content_mode"] as? String == "apple")
        #expect(obj["character_name"] as? String == "Kira")
        #expect(obj["source"] as? String == "desktop-edit")
        let edit = obj["edit"] as? [String: Any]
        #expect(edit?["source_path"] as? String == "/orig/a.png")
        #expect(edit?["source_asset_id"] as? String == "A1")
        #expect(edit?["version"] as? Int == 1)
        let recipeData = try! JSONSerialization.data(withJSONObject: edit!["recipe"]!)
        #expect(try! JSONDecoder().decode(EditRecipe.self, from: recipeData) == recipe)
    }

    @Test("outputPath suffixes on collision")
    func collision() throws {
        let dir = tempDir()
        let p1 = EditExporter.outputPath(in: dir, seconds: 1234)
        #expect(p1 == dir + "/edit-1234.png")
        FileManager.default.createFile(atPath: p1, contents: Data())
        #expect(EditExporter.outputPath(in: dir, seconds: 1234) == dir + "/edit-1234-2.png")
        FileManager.default.createFile(atPath: dir + "/edit-1234-2.png", contents: Data())
        #expect(EditExporter.outputPath(in: dir, seconds: 1234) == dir + "/edit-1234-3.png")
    }

    @Test("export writes a PNG of the rendered size and a sidecar that reopens")
    func exportWrites() async throws {
        let dir = tempDir()
        let src = EditTestSupport.horizontalGradient(width: 80, height: 40)
        var recipe = EditRecipe(); recipe.geometry.crop = CGRect(x: 0, y: 0, width: 0.5, height: 1)
        let out = try await EditExporter.export(sourceImage: src, sourcePath: "/orig/g.png", sourceAsset: nil,
                                                recipe: recipe, subjectMask: nil, outputDirectory: dir, ingestor: nil)
        #expect(out.hasPrefix(dir + "/edit-") && out.hasSuffix(".png"))
        let cg = CGImageSourceCreateWithURL(URL(fileURLWithPath: out) as CFURL, nil).flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) }
        #expect(cg?.width == 40 && cg?.height == 40)
        let sc = EditSidecar.read(forImageAt: out)
        #expect(sc?.recipe == recipe && sc?.sourcePath == "/orig/g.png")
    }

    @Test("exporting a derived asset points the sidecar at the root source")
    func rootChain() async throws {
        let dir = tempDir()
        let src = EditTestSupport.solid(r: 1, g: 2, b: 3, width: 8, height: 8)
        let first = try await EditExporter.export(sourceImage: src, sourcePath: "/orig/root.png", sourceAsset: nil,
                                                  recipe: EditRecipe(), subjectMask: nil, outputDirectory: dir, ingestor: nil)
        let second = try await EditExporter.export(sourceImage: src, sourcePath: first, sourceAsset: nil,
                                                   recipe: EditRecipe(), subjectMask: nil, outputDirectory: dir, ingestor: nil)
        #expect(EditSidecar.read(forImageAt: second)?.sourcePath == "/orig/root.png")
    }
}
```

- [ ] **Step 3: Run both suites to verify failure** — Expected: compile errors.

- [ ] **Step 4: Implement EditSidecar.swift**

```swift
// EditSidecar.swift — the `edit` block in a derived asset's adjacent sidecar
//
// Desktop convention: `<image basename>.json` next to the image (see
// AssetIngestor.readSidecar). The exporter writes generation fields the
// ingestor already understands plus this block; this file reads it back.

import Foundation

public struct EditSidecar: Codable, Equatable, Sendable {
    public var version: Int
    public var sourcePath: String
    public var sourceAssetId: String?
    public var recipe: EditRecipe
    public var editor: String
    public var createdAt: Date

    public init(version: Int, sourcePath: String, sourceAssetId: String?, recipe: EditRecipe, editor: String, createdAt: Date) {
        self.version = version; self.sourcePath = sourcePath; self.sourceAssetId = sourceAssetId
        self.recipe = recipe; self.editor = editor; self.createdAt = createdAt
    }

    enum CodingKeys: String, CodingKey {
        case version, recipe, editor
        case sourcePath = "source_path"
        case sourceAssetId = "source_asset_id"
        case createdAt = "created_at"
    }

    static let decoder: JSONDecoder = { let d = JSONDecoder(); d.dateDecodingStrategy = .iso8601; return d }()
    static let encoder: JSONEncoder = { let e = JSONEncoder(); e.dateEncodingStrategy = .iso8601; return e }()

    public static func sidecarPath(forImageAt imagePath: String) -> String {
        ((imagePath as NSString).deletingPathExtension) + ".json"
    }

    public static func read(forImageAt imagePath: String) -> EditSidecar? {
        guard let data = FileManager.default.contents(atPath: sidecarPath(forImageAt: imagePath)),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let edit = obj["edit"],
              let editData = try? JSONSerialization.data(withJSONObject: edit)
        else { return nil }
        return try? decoder.decode(EditSidecar.self, from: editData)
    }

    public static func rootSource(forImageAt imagePath: String) -> (path: String, assetId: String?) {
        var path = imagePath
        var assetId: String? = nil
        var hops = 0
        while hops < 32, let sc = read(forImageAt: path) {
            path = sc.sourcePath; assetId = sc.sourceAssetId; hops += 1
        }
        return (path, assetId)
    }

    public var jsonObject: [String: Any] {
        guard let data = try? Self.encoder.encode(self),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        return obj
    }
}
```

- [ ] **Step 5: Implement EditExporter.swift**

```swift
// EditExporter.swift — full-resolution render → PNG + sidecar → ingest
//
// Write order guarantees no partial state: PNG first, sidecar second (PNG
// removed if the sidecar fails), ingest last (files stay if ingest fails;
// the gallery poller picks them up).

import Foundation
import CoreImage
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

public enum EditExportError: Error {
    case renderFailed
    case writeFailed(String)
}

public enum EditExporter {

    public static func sidecarObject(source: DAMAsset?, sourcePath: String, recipe: EditRecipe, now: Date) -> [String: Any] {
        var obj: [String: Any] = ["source": "desktop-edit"]
        if let s = source {
            if let v = s.prompt { obj["prompt"] = v }
            if let v = s.negativePrompt { obj["negative_prompt"] = v }
            if let v = s.seed { obj["seed"] = v }
            if let v = s.steps { obj["steps"] = v }
            if let v = s.guidance { obj["guidance"] = v }
            if let v = s.modelFamily { obj["model_family"] = v }
            if let v = s.contentMode { obj["content_mode"] = v }
            if let v = s.characterName { obj["character_name"] = v }
        }
        let root = EditSidecar.rootSource(forImageAt: sourcePath)
        let rootAssetId = root.path == sourcePath ? source?.id : root.assetId
        let sc = EditSidecar(version: EditRecipe.currentVersion, sourcePath: root.path, sourceAssetId: rootAssetId,
                             recipe: recipe, editor: "ComfyBoxDesktop", createdAt: now)
        obj["edit"] = sc.jsonObject
        return obj
    }

    public static func outputPath(in directory: String, seconds: Int) -> String {
        let base = (directory as NSString).appendingPathComponent("edit-\(seconds)")
        var candidate = base + ".png"
        var n = 2
        while FileManager.default.fileExists(atPath: candidate) {
            candidate = base + "-\(n).png"; n += 1
        }
        return candidate
    }

    public static func export(sourceImage: CGImage, sourcePath: String, sourceAsset: DAMAsset?,
                              recipe: EditRecipe, subjectMask: CIImage?,
                              outputDirectory: String, ingestor: AssetIngestor?) async throws -> String {
        let context = CIContext(options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
        let output = EditRenderer.render(source: CIImage(cgImage: sourceImage), recipe: recipe, subjectMask: subjectMask)
        guard let cg = context.createCGImage(output, from: output.extent) else { throw EditExportError.renderFailed }

        try FileManager.default.createDirectory(atPath: outputDirectory, withIntermediateDirectories: true)
        let pngPath = outputPath(in: outputDirectory, seconds: Int(Date().timeIntervalSince1970))
        guard let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: pngPath) as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw EditExportError.writeFailed("could not create \(pngPath)") }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw EditExportError.writeFailed("could not finalize \(pngPath)") }

        let sidecar = sidecarObject(source: sourceAsset, sourcePath: sourcePath, recipe: recipe, now: Date())
        do {
            let data = try JSONSerialization.data(withJSONObject: sidecar, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: EditSidecar.sidecarPath(forImageAt: pngPath)))
        } catch {
            try? FileManager.default.removeItem(atPath: pngPath)
            throw EditExportError.writeFailed("sidecar: \(error.localizedDescription)")
        }

        if let ingestor { _ = try await ingestor.ingestFile(at: pngPath) }
        return pngPath
    }
}
```

- [ ] **Step 6: Run both suites** — Expected: 3 + 4 tests pass.

- [ ] **Step 7: Commit**

```bash
git add Sources/ComfyBoxDesktop/Edit/EditSidecar.swift Sources/ComfyBoxDesktop/Edit/EditExporter.swift Tests/ComfyBoxDesktopTests/EditSidecarTests.swift Tests/ComfyBoxDesktopTests/EditExporterTests.swift
git commit -m "feat(desktop): EditExporter writes derived PNG with reopenable recipe sidecar"
```

---

### Task 8: EditSession

**Files:**
- Create: `Sources/ComfyBoxDesktop/Edit/EditSession.swift`
- Test: `Tests/ComfyBoxDesktopTests/EditSessionTests.swift`

**Interfaces:**
- Consumes: `EditRecipe`, `EditRenderer`, `EditSidecar`, `SubjectMasker`, `EditExporter`, `DAMAsset`, `AssetIngestor`.
- Produces:
  ```swift
  @MainActor @Observable public final class EditSession {
      public private(set) var sourcePath: String
      public private(set) var sourceImage: CGImage?
      public private(set) var sourceAsset: DAMAsset?
      public var recipe: EditRecipe                    // live value; call set(_:) / commit()
      public private(set) var preview: CGImage?
      public private(set) var isRendering: Bool
      public var showOriginal: Bool
      public private(set) var warning: String?
      public private(set) var isDirty: Bool
      public var canUndo: Bool; public var canRedo: Bool
      public private(set) var subjectMask: CIImage?
      public private(set) var subjectStatus: String?
      public var previewSize: CGSize                   // of the post-geometry preview

      public init(sourcePath: String, sourceAsset: DAMAsset?, previewMaxDimension: CGFloat = 2048)
      /// Loads pixels (root source when a sidecar chain exists) and the stored recipe. Sets `warning` on fallbacks.
      public func load() async
      public func set(_ mutate: (inout EditRecipe) -> Void)     // live update, no undo entry
      public func commit()                                      // pushes previous committed recipe onto undo
      public func undo(); public func redo(); public func reset()
      public func requestSubjectMask() async
      /// Renders the preview immediately (bypasses debounce); used by tests and by commit.
      public func renderNow() async
      public func export(outputDirectory: String, ingestor: AssetIngestor?) async throws -> String
  }
  ```

- [ ] **Step 1: Write failing tests**

```swift
// EditSessionTests.swift
import Testing
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import ComfyBoxDesktop

@Suite("EditSession")
@MainActor
struct EditSessionTests {
    func writePNG(_ image: CGImage, to path: String) {
        let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, nil); CGImageDestinationFinalize(dest)
    }
    func tempDir() -> String {
        let d = NSTemporaryDirectory() + "editsession-" + UUID().uuidString
        try! FileManager.default.createDirectory(atPath: d, withIntermediateDirectories: true)
        return d
    }

    @Test("load reads pixels and renders an identity preview")
    func loadAndPreview() async {
        let dir = tempDir(); let p = dir + "/src.png"
        writePNG(EditTestSupport.horizontalGradient(width: 64, height: 32), to: p)
        let s = EditSession(sourcePath: p, sourceAsset: nil, previewMaxDimension: 32)
        await s.load()
        #expect(s.sourceImage?.width == 64)
        await s.renderNow()
        #expect(s.preview != nil)
        #expect(s.preview!.width <= 32)     // downscaled for preview
        #expect(!s.isDirty && s.warning == nil)
    }

    @Test("set does not create an undo entry; commit does; undo/redo restore")
    func undoRedo() async {
        let dir = tempDir(); let p = dir + "/src.png"
        writePNG(EditTestSupport.solid(r: 1, g: 1, b: 1, width: 8, height: 8), to: p)
        let s = EditSession(sourcePath: p, sourceAsset: nil)
        await s.load()
        s.set { $0.adjustments.exposure = 0.5 }
        #expect(!s.canUndo && s.isDirty)
        s.commit()
        #expect(s.canUndo)
        s.set { $0.adjustments.exposure = 1.0 }; s.commit()
        s.undo(); #expect(s.recipe.adjustments.exposure == 0.5)
        s.undo(); #expect(s.recipe.adjustments.exposure == 0.0 && !s.canUndo)
        s.redo(); #expect(s.recipe.adjustments.exposure == 0.5 && s.canRedo)
        s.reset(); #expect(s.recipe.isIdentity && s.canUndo)
    }

    @Test("load on a derived asset opens the root pixels and the stored recipe")
    func reopen() async {
        let dir = tempDir(); let root = dir + "/root.png"
        writePNG(EditTestSupport.horizontalGradient(width: 40, height: 20), to: root)
        var recipe = EditRecipe(); recipe.adjustments.vibrance = 0.4
        let s0 = EditSession(sourcePath: root, sourceAsset: nil)
        await s0.load(); s0.set { $0 = recipe }; s0.commit()
        let derived = try! await s0.export(outputDirectory: dir, ingestor: nil)
        let s1 = EditSession(sourcePath: derived, sourceAsset: nil)
        await s1.load()
        #expect(s1.sourcePath == root)
        #expect(s1.recipe == recipe)
        #expect(s1.sourceImage?.width == 40)
    }

    @Test("newer recipe version loads pixels, drops the recipe, warns")
    func newerVersion() async throws {
        let dir = tempDir(); let root = dir + "/root.png"; let derived = dir + "/edit-9.png"
        writePNG(EditTestSupport.solid(r: 5, g: 5, b: 5, width: 8, height: 8), to: root)
        writePNG(EditTestSupport.solid(r: 5, g: 5, b: 5, width: 8, height: 8), to: derived)
        var obj = EditSidecar(version: 1, sourcePath: root, sourceAssetId: nil, recipe: EditRecipe(), editor: "x", createdAt: Date()).jsonObject
        obj["version"] = 99
        try JSONSerialization.data(withJSONObject: ["edit": obj]).write(to: URL(fileURLWithPath: dir + "/edit-9.json"))
        let s = EditSession(sourcePath: derived, sourceAsset: nil)
        await s.load()
        #expect(s.recipe.isIdentity && s.warning != nil && s.sourceImage != nil)
    }

    @Test("missing root falls back to derived pixels with a warning")
    func missingRoot() async throws {
        let dir = tempDir(); let derived = dir + "/edit-9.png"
        writePNG(EditTestSupport.solid(r: 5, g: 5, b: 5, width: 8, height: 8), to: derived)
        let sc = EditSidecar(version: 1, sourcePath: dir + "/gone.png", sourceAssetId: nil, recipe: EditRecipe(), editor: "x", createdAt: Date())
        try JSONSerialization.data(withJSONObject: ["edit": sc.jsonObject]).write(to: URL(fileURLWithPath: dir + "/edit-9.json"))
        let s = EditSession(sourcePath: derived, sourceAsset: nil)
        await s.load()
        #expect(s.sourcePath == derived && s.sourceImage != nil && s.warning != nil)
    }
}
```

- [ ] **Step 2: Run to verify failure** — `-only-testing:ComfyBoxDesktopTests/EditSessionTests`. Expected: compile error.

- [ ] **Step 3: Implement EditSession.swift**

```swift
// EditSession.swift — observable state for one open image in the editor
//
// Owns the source pixels, the live recipe, undo/redo, and a debounced
// preview render on a background task. The preview source is downscaled
// once; the exporter re-renders from the full-resolution source.

import Foundation
import CoreImage
import CoreGraphics
import ImageIO
import Observation

@MainActor
@Observable
public final class EditSession {
    public private(set) var sourcePath: String
    public private(set) var sourceImage: CGImage?
    public private(set) var sourceAsset: DAMAsset?
    public var recipe = EditRecipe()
    public private(set) var preview: CGImage?
    public private(set) var isRendering = false
    public var showOriginal = false
    public private(set) var warning: String?
    public private(set) var isDirty = false
    public private(set) var subjectMask: CIImage?
    public private(set) var subjectStatus: String?
    public var previewSize: CGSize = .zero

    private var committed = EditRecipe()
    private var undoStack: [EditRecipe] = []
    private var redoStack: [EditRecipe] = []
    private let undoLimit = 100

    private let previewMaxDimension: CGFloat
    private var previewSource: CIImage?
    private var previewScale: CGFloat = 1
    private var renderTask: Task<Void, Never>?
    private var renderGeneration = 0
    private let context = CIContext(options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                                              .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
    private let masker = SubjectMasker()

    public var canUndo: Bool { !undoStack.isEmpty }
    public var canRedo: Bool { !redoStack.isEmpty }

    public init(sourcePath: String, sourceAsset: DAMAsset?, previewMaxDimension: CGFloat = 2048) {
        self.sourcePath = sourcePath
        self.sourceAsset = sourceAsset
        self.previewMaxDimension = previewMaxDimension
    }

    // MARK: - Loading

    public func load() async {
        warning = nil
        var path = sourcePath
        var storedRecipe = EditRecipe()
        if let sc = EditSidecar.read(forImageAt: sourcePath) {
            let root = EditSidecar.rootSource(forImageAt: sourcePath)
            if FileManager.default.fileExists(atPath: root.path) {
                path = root.path
                if sc.version > EditRecipe.currentVersion {
                    warning = "This edit was saved by a newer ComfyBox (recipe v\(sc.version)); opening pixels only."
                } else {
                    storedRecipe = sc.recipe
                }
            } else {
                warning = "Original \(URL(fileURLWithPath: root.path).lastPathComponent) is missing; editing the flattened image."
            }
        }
        let loadPath = path
        let image = await Task.detached(priority: .userInitiated) { () -> CGImage? in
            guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: loadPath) as CFURL, nil) else { return nil }
            return CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceShouldCache: false] as CFDictionary)
        }.value
        sourcePath = path
        sourceImage = image
        guard let image else { warning = "Couldn't read \(URL(fileURLWithPath: loadPath).lastPathComponent)."; return }
        let longest = CGFloat(max(image.width, image.height))
        previewScale = min(1, previewMaxDimension / longest)
        var ci = CIImage(cgImage: image)
        if previewScale < 1 {
            let f = CIFilter.lanczosScaleTransform(); f.inputImage = ci; f.scale = Float(previewScale); f.aspectRatio = 1
            ci = f.outputImage ?? ci
            ci = ci.transformed(by: CGAffineTransform(translationX: -ci.extent.minX, y: -ci.extent.minY))
        }
        previewSource = ci
        recipe = storedRecipe; committed = storedRecipe
        undoStack.removeAll(); redoStack.removeAll(); isDirty = false
        subjectMask = nil; subjectStatus = nil
        scheduleRender()
    }

    // MARK: - Recipe mutation

    public func set(_ mutate: (inout EditRecipe) -> Void) {
        mutate(&recipe)
        isDirty = recipe != committed || !undoStack.isEmpty
        scheduleRender()
    }

    public func commit() {
        guard recipe != committed else { return }
        undoStack.append(committed)
        if undoStack.count > undoLimit { undoStack.removeFirst() }
        redoStack.removeAll()
        committed = recipe
        isDirty = true
    }

    public func undo() {
        guard let previous = undoStack.popLast() else { return }
        redoStack.append(committed)
        committed = previous; recipe = previous
        isDirty = true
        scheduleRender()
    }

    public func redo() {
        guard let next = redoStack.popLast() else { return }
        undoStack.append(committed)
        committed = next; recipe = next
        isDirty = true
        scheduleRender()
    }

    public func reset() {
        recipe = EditRecipe()
        commit()
        scheduleRender()
    }

    // MARK: - Subject mask

    public func requestSubjectMask() async {
        guard let sourceImage else { return }
        subjectStatus = "Finding subject…"
        do {
            subjectMask = try await masker.mask(for: sourceImage, cacheKey: sourcePath)
            subjectStatus = nil
        } catch SubjectMaskError.noSubject {
            subjectMask = nil; subjectStatus = "No subject found."
        } catch {
            subjectMask = nil; subjectStatus = "Vision failed: \(error.localizedDescription)"
        }
        scheduleRender()
    }

    // MARK: - Preview rendering

    private func scheduleRender() {
        renderTask?.cancel()
        renderGeneration += 1
        let generation = renderGeneration
        renderTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(40))
            guard !Task.isCancelled, let self else { return }
            await self.render(generation: generation)
        }
    }

    public func renderNow() async {
        renderTask?.cancel()
        renderGeneration += 1
        await render(generation: renderGeneration)
    }

    private func render(generation: Int) async {
        guard let previewSource else { return }
        isRendering = true
        let recipe = self.recipe
        let mask = scaledSubjectMask()
        let context = self.context
        let result = await Task.detached(priority: .userInitiated) { () -> CGImage? in
            let out = EditRenderer.render(source: previewSource, recipe: recipe, subjectMask: mask)
            guard !out.extent.isEmpty, !out.extent.isInfinite else { return nil }
            return context.createCGImage(out, from: out.extent)
        }.value
        guard generation == renderGeneration else { return }
        isRendering = false
        if let result {
            preview = result
            previewSize = CGSize(width: result.width, height: result.height)
        } else {
            warning = "Preview render failed; the last good preview is shown."
        }
    }

    /// Subject mask resampled to the preview source size so it lines up before geometry.
    private func scaledSubjectMask() -> CIImage? {
        guard let subjectMask, let previewSource else { return nil }
        if previewScale >= 1 { return subjectMask }
        let scaled = subjectMask.transformed(by: CGAffineTransform(scaleX: previewScale, y: previewScale))
        return scaled.cropped(to: previewSource.extent)
    }

    // MARK: - Export

    public func export(outputDirectory: String, ingestor: AssetIngestor?) async throws -> String {
        guard let sourceImage else { throw EditExportError.renderFailed }
        let path = try await EditExporter.export(sourceImage: sourceImage, sourcePath: sourcePath, sourceAsset: sourceAsset,
                                                 recipe: recipe, subjectMask: subjectMask,
                                                 outputDirectory: outputDirectory, ingestor: ingestor)
        committed = recipe
        isDirty = false
        return path
    }
}
```

Add `import CoreImage.CIFilterBuiltins` at the top (for `CIFilter.lanczosScaleTransform()`).

- [ ] **Step 4: Run the suite** — Expected: 5 tests pass. If `loadAndPreview` is flaky because `renderNow` raced a scheduled render, that is a bug in generation handling; fix in `render(generation:)`, not the test.

- [ ] **Step 5: Commit**

```bash
git add Sources/ComfyBoxDesktop/Edit/EditSession.swift Tests/ComfyBoxDesktopTests/EditSessionTests.swift
git commit -m "feat(desktop): EditSession with undo stack and debounced preview"
```

---

### Task 9: CurvesEditor view

**Files:**
- Create: `Sources/ComfyBoxDesktop/Views/CurvesEditor.swift`

**Interfaces:**
- Consumes: `ToneCurves`, `CurvePoint` (Task 3).
- Produces:
  ```swift
  struct CurvesEditor: View {
      @Binding var curves: ToneCurves
      var onCommit: () -> Void        // called on drag end / add / remove so the session records one undo step
  }
  ```
  Behaviour: channel picker (RGB, R, G, B) as a segmented control; square graph with a diagonal identity guide; the curve drawn by sampling `ToneCurves.sample` at 64 x positions; control points as circles; drag moves a point (x clamped between its neighbours, y 0…1); double-click empty space adds a point; dragging a point outside the graph vertically by more than 30 pt removes it; a Reset button clears the active channel.

- [ ] **Step 1: Implement**

```swift
// CurvesEditor.swift — draggable tone curve for one channel at a time

import SwiftUI

struct CurvesEditor: View {
    @Binding var curves: ToneCurves
    var onCommit: () -> Void

    enum Channel: String, CaseIterable, Identifiable { case rgb = "RGB", r = "R", g = "G", b = "B"; var id: String { rawValue } }
    @State private var channel: Channel = .rgb
    @State private var dragIndex: Int?

    private var points: Binding<[CurvePoint]> {
        switch channel {
        case .rgb: return $curves.rgb
        case .r: return $curves.r
        case .g: return $curves.g
        case .b: return $curves.b
        }
    }
    private var color: Color {
        switch channel { case .rgb: return .primary; case .r: return .red; case .g: return .green; case .b: return .blue }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Picker("", selection: $channel) { ForEach(Channel.allCases) { Text($0.rawValue).tag($0) } }
                    .pickerStyle(.segmented).labelsHidden()
                Button("Reset") { points.wrappedValue = []; onCommit() }
                    .controlSize(.small).disabled(points.wrappedValue.isEmpty)
            }
            GeometryReader { geo in
                let size = min(geo.size.width, geo.size.height)
                let rect = CGRect(x: 0, y: 0, width: size, height: size)
                ZStack {
                    RoundedRectangle(cornerRadius: 4).fill(Color(nsColor: .controlBackgroundColor))
                    Path { p in p.move(to: CGPoint(x: 0, y: size)); p.addLine(to: CGPoint(x: size, y: 0)) }
                        .stroke(Color.secondary.opacity(0.3), style: StrokeStyle(lineWidth: 1, dash: [4]))
                    Path { p in
                        for i in 0...64 {
                            let x = Double(i) / 64
                            let y = ToneCurves.sample(points.wrappedValue, at: x)
                            let pt = CGPoint(x: x * size, y: (1 - y) * size)
                            if i == 0 { p.move(to: pt) } else { p.addLine(to: pt) }
                        }
                    }.stroke(color, lineWidth: 1.5)
                    ForEach(Array(points.wrappedValue.enumerated()), id: \.offset) { idx, pt in
                        Circle().fill(color).frame(width: 9, height: 9)
                            .position(x: pt.x * size, y: (1 - pt.y) * size)
                    }
                }
                .frame(width: size, height: size)
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0)
                    .onChanged { v in
                        let nx = min(max(v.location.x / size, 0), 1)
                        let ny = min(max(1 - v.location.y / size, 0), 1)
                        if dragIndex == nil {
                            dragIndex = nearestIndex(to: v.startLocation, size: size) ?? insertPoint(x: nx, y: ny)
                        }
                        guard let i = dragIndex, points.wrappedValue.indices.contains(i) else { return }
                        var pts = points.wrappedValue
                        let lo = i > 0 ? pts[i - 1].x + 0.01 : 0.0
                        let hi = i < pts.count - 1 ? pts[i + 1].x - 0.01 : 1.0
                        pts[i] = CurvePoint(x: min(max(nx, lo), hi), y: ny)
                        // Drag far outside vertically to remove.
                        if v.location.y < -30 || v.location.y > size + 30 { pts.remove(at: i); dragIndex = nil }
                        points.wrappedValue = pts
                    }
                    .onEnded { _ in dragIndex = nil; onCommit() })
                .onTapGesture(count: 2) { location in
                    _ = insertPoint(x: min(max(location.x / size, 0), 1), y: min(max(1 - location.y / size, 0), 1))
                    onCommit()
                }
                .frame(maxWidth: .infinity, alignment: .center)
                _ = rect
            }
            .aspectRatio(1, contentMode: .fit)
        }
    }

    private func nearestIndex(to location: CGPoint, size: CGFloat) -> Int? {
        var best: (Int, CGFloat)?
        for (i, p) in points.wrappedValue.enumerated() {
            let d = hypot(p.x * size - location.x, (1 - p.y) * size - location.y)
            if d < 12, best == nil || d < best!.1 { best = (i, d) }
        }
        return best?.0
    }

    @discardableResult
    private func insertPoint(x: Double, y: Double) -> Int {
        var pts = points.wrappedValue
        let idx = pts.firstIndex { $0.x > x } ?? pts.count
        pts.insert(CurvePoint(x: x, y: y), at: idx)
        points.wrappedValue = pts
        return idx
    }
}
```

- [ ] **Step 2: Build** — build-only command. Expected: no errors. Remove the `_ = rect` line and the unused `rect` if the compiler warns.

- [ ] **Step 3: Commit**

```bash
git add Sources/ComfyBoxDesktop/Views/CurvesEditor.swift
git commit -m "feat(desktop): CurvesEditor view"
```

---

### Task 10: EditView and EditTab

**Files:**
- Create: `Sources/ComfyBoxDesktop/Views/EditView.swift`

**Interfaces:**
- Consumes: `EditSession` (Task 8), `MaskCanvas`, `ImageFit` (Task 2), `CurvesEditor` (Task 9), `MaskStrokes`, `NumericSliderField` (existing, used in `InpaintView`), `DesktopSettings.load().outputDirectory`, `AssetIngestor`.
- Produces:
  ```swift
  struct EditView: View {
      @Bindable var session: EditSession
      var ingestor: AssetIngestor?
      /// Called after a successful save with the derived path and the current local mask (nil when none).
      var onSendToInpaint: ((String, MaskStrokes?) -> Void)?
  }
  struct EditTab: View {
      var ingestor: AssetIngestor?
      @Binding var pendingImage: String?          // path queued from Gallery/AssetDetail; consumed once
      var onSendToInpaint: ((String, MaskStrokes?) -> Void)?
  }
  ```

- [ ] **Step 1: Implement EditView.swift**

```swift
// EditView.swift — the Edit tab: canvas left, adjustment panel right
//
// Sliders call session.set for live preview and session.commit on
// gesture end so one drag is one undo step. Crop is edited on a
// normalized overlay; Local paints on the shared MaskCanvas.

import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct EditView: View {
    @Bindable var session: EditSession
    var ingestor: AssetIngestor?
    var onSendToInpaint: ((String, MaskStrokes?) -> Void)?

    enum Group: String, CaseIterable, Identifiable {
        case light = "Light", color = "Color", curves = "Curves", detail = "Detail",
             crop = "Crop & Rotate", local = "Local", subject = "Subject"
        var id: String { rawValue }
    }
    @State private var expanded: Set<Group> = [.light]
    @State private var brush: CGFloat = 40
    @State private var erase = false
    @State private var status: String?
    @State private var isSaving = false
    @State private var isError = false

    var body: some View {
        HSplitView {
            canvas.frame(minWidth: 480)
            panel.frame(minWidth: 320, maxWidth: 400)
        }
        .toolbar {
            ToolbarItemGroup {
                Button { session.undo() } label: { Label("Undo", systemImage: "arrow.uturn.backward") }
                    .disabled(!session.canUndo).keyboardShortcut("z", modifiers: .command)
                Button { session.redo() } label: { Label("Redo", systemImage: "arrow.uturn.forward") }
                    .disabled(!session.canRedo).keyboardShortcut("z", modifiers: [.command, .shift])
                Button { session.reset() } label: { Label("Reset", systemImage: "arrow.counterclockwise") }
                    .disabled(session.recipe.isIdentity)
                Button { } label: { Label("Before", systemImage: "eye") }
                    .simultaneousGesture(DragGesture(minimumDistance: 0)
                        .onChanged { _ in session.showOriginal = true }
                        .onEnded { _ in session.showOriginal = false })
                Button { Task { await save(thenInpaint: false) } } label: {
                    Label(isSaving ? "Saving…" : "Save", systemImage: "square.and.arrow.down")
                }.disabled(isSaving || session.sourceImage == nil).keyboardShortcut("s", modifiers: .command)
                Button { Task { await save(thenInpaint: true) } } label: {
                    Label("Save & Inpaint", systemImage: "paintbrush.pointed")
                }.disabled(isSaving || session.sourceImage == nil || onSendToInpaint == nil)
            }
        }
    }

    // MARK: - Canvas

    private var canvas: some View {
        GeometryReader { geo in
            ZStack {
                Color(nsColor: .underPageBackgroundColor)
                if let cg = session.showOriginal ? session.sourceImage : session.preview {
                    let rect = ImageFit.rect(imageSize: CGSize(width: cg.width, height: cg.height), in: geo.size)
                    Image(decorative: cg, scale: 1).resizable()
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                    if expanded.contains(.local), !session.showOriginal {
                        MaskCanvas(imageSize: rect.size, strokes: localMaskBinding, brushPoints: brush, erase: erase)
                            .position(x: rect.midX, y: rect.midY)
                    }
                    if expanded.contains(.crop), !session.showOriginal {
                        CropOverlay(rect: rect, crop: cropBinding, onCommit: { session.commit() })
                    }
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "slider.horizontal.3").font(.system(size: 44)).foregroundStyle(.tertiary)
                        Text(session.warning ?? "Open an image to edit").foregroundStyle(.secondary)
                    }
                }
                if session.isRendering {
                    ProgressView().controlSize(.small).padding(8)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                }
            }
        }
    }

    /// Local mask strokes; painting on an image with no layer creates one.
    private var localMaskBinding: Binding<MaskStrokes> {
        Binding(get: { session.recipe.local?.mask ?? MaskStrokes() },
                set: { new in
                    session.set { r in
                        if r.local == nil { r.local = EditLocalLayer() }
                        r.local?.mask = new
                    }
                    session.commit()
                })
    }

    /// Crop is stored against the pre-crop (post-rotation) frame; the overlay edits it in the preview rect.
    private var cropBinding: Binding<CGRect> {
        Binding(get: { session.recipe.geometry.crop ?? CGRect(x: 0, y: 0, width: 1, height: 1) },
                set: { new in session.set { $0.geometry.crop = new == CGRect(x: 0, y: 0, width: 1, height: 1) ? nil : new } })
    }

    // MARK: - Panel

    private var panel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Image(systemName: "slider.horizontal.3").foregroundStyle(.indigo)
                    Text("Edit").font(.headline)
                    Spacer()
                    if session.isDirty { Text("unsaved").font(.caption2).foregroundStyle(.orange) }
                }
                if let w = session.warning {
                    Label(w, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
                }
                Text(URL(fileURLWithPath: session.sourcePath).lastPathComponent)
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)

                group(.light) {
                    slider("Exposure", \.exposure, -5...5, step: 0.05)
                    slider("Contrast", \.contrast, -1...1)
                    slider("Highlights", \.highlights, -1...1)
                    slider("Shadows", \.shadows, -1...1)
                    slider("Whites", \.whites, -1...1)
                    slider("Blacks", \.blacks, -1...1)
                }
                group(.color) {
                    slider("Temperature", \.temperature, -1...1)
                    slider("Tint", \.tint, -1...1)
                    slider("Vibrance", \.vibrance, -1...1)
                    slider("Saturation", \.saturation, -1...1)
                }
                group(.curves) {
                    CurvesEditor(curves: Binding(get: { session.recipe.adjustments.curves },
                                                 set: { c in session.set { $0.adjustments.curves = c } }),
                                 onCommit: { session.commit() })
                        .frame(height: 220)
                }
                group(.detail) {
                    slider("Sharpen", \.sharpen, 0...1)
                    slider("Noise Reduction", \.noiseReduction, 0...1)
                    slider("Vignette", \.vignette, 0...1)
                }
                group(.crop) { cropControls }
                group(.local) { localControls }
                group(.subject) { subjectControls }

                if let status {
                    Label(status, systemImage: isError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                        .font(.caption).foregroundStyle(isError ? .orange : .green)
                }
            }
            .padding(14)
        }
    }

    @ViewBuilder private func group<C: View>(_ g: Group, @ViewBuilder content: () -> C) -> some View {
        DisclosureGroup(isExpanded: Binding(get: { expanded.contains(g) },
                                            set: { if $0 { expanded.insert(g) } else { expanded.remove(g) } })) {
            VStack(alignment: .leading, spacing: 8) { content() }.padding(.top, 6)
        } label: { Text(g.rawValue).font(.subheadline.weight(.medium)) }
    }

    private func slider(_ label: String, _ key: WritableKeyPath<EditAdjustments, Double>,
                        _ range: ClosedRange<Double>, step: Double = 0.01) -> some View {
        NumericSliderField(
            label: label,
            value: Binding(get: { session.recipe.adjustments[keyPath: key] },
                           set: { v in session.set { $0.adjustments[keyPath: key] = v } }),
            range: range, step: step, fractionDigits: 2,
            onEditingEnded: { session.commit() })
    }

    private func localSlider(_ label: String, _ key: WritableKeyPath<EditAdjustments, Double>,
                             _ range: ClosedRange<Double>) -> some View {
        NumericSliderField(
            label: label,
            value: Binding(get: { session.recipe.local?.adjustments[keyPath: key] ?? 0 },
                           set: { v in session.set { r in
                               if r.local == nil { r.local = EditLocalLayer() }
                               r.local?.adjustments[keyPath: key] = v } }),
            range: range, step: 0.01, fractionDigits: 2,
            onEditingEnded: { session.commit() })
    }

    private var cropControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Aspect").font(.caption).foregroundStyle(.secondary)
                Menu("Free") {
                    Button("Free") { }
                    Button("Original") { applyAspect(nil) }
                    Button("1:1") { applyAspect(1) }
                    Button("4:5") { applyAspect(4.0 / 5.0) }
                    Button("3:2") { applyAspect(3.0 / 2.0) }
                    Button("16:9") { applyAspect(16.0 / 9.0) }
                }.controlSize(.small)
                Spacer()
                Button("Clear Crop") { session.set { $0.geometry.crop = nil }; session.commit() }
                    .controlSize(.small).disabled(session.recipe.geometry.crop == nil)
            }
            NumericSliderField(label: "Straighten", value: Binding(get: { session.recipe.geometry.straightenDegrees },
                                                                   set: { v in session.set { $0.geometry.straightenDegrees = v } }),
                               range: -45...45, step: 0.1, fractionDigits: 1, onEditingEnded: { session.commit() })
            HStack {
                Button { session.set { $0.geometry.quarterTurns = ($0.geometry.quarterTurns + 3) % 4 }; session.commit() } label: { Image(systemName: "rotate.left") }
                Button { session.set { $0.geometry.quarterTurns = ($0.geometry.quarterTurns + 1) % 4 }; session.commit() } label: { Image(systemName: "rotate.right") }
                Button { session.set { $0.geometry.flipH.toggle() }; session.commit() } label: { Image(systemName: "arrow.left.and.right.righttriangle.left.righttriangle.right") }
                Button { session.set { $0.geometry.flipV.toggle() }; session.commit() } label: { Image(systemName: "arrow.up.and.down.righttriangle.up.righttriangle.down") }
            }.controlSize(.small)
            Text("Drag the corners on the image to crop.").font(.caption2).foregroundStyle(.tertiary)
        }
    }

    /// Center a crop of the given width:height ratio (nil = the source's own ratio) inside the current frame.
    private func applyAspect(_ ratio: Double?) {
        let w = Double(session.previewSize.width), h = Double(session.previewSize.height)
        guard w > 0, h > 0 else { return }
        let target = ratio ?? (w / h)
        let frameRatio = w / h
        var cw = 1.0, ch = 1.0
        if target > frameRatio { ch = frameRatio / target } else { cw = target / frameRatio }
        session.set { $0.geometry.crop = CGRect(x: (1 - cw) / 2, y: (1 - ch) / 2, width: cw, height: ch) }
        session.commit()
    }

    private var localControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "paintbrush")
                Slider(value: $brush, in: 8...200)
                Text("\(Int(brush))").font(.caption.monospacedDigit()).frame(width: 30)
            }
            HStack {
                Toggle("Erase", isOn: $erase).toggleStyle(.button).controlSize(.small)
                Button("Undo Stroke") { session.set { $0.local?.mask.undoLast() }; session.commit() }
                    .controlSize(.small).disabled(session.recipe.local?.mask.isEmpty ?? true)
                Button("Clear Mask") { session.set { $0.local = nil }; session.commit() }
                    .controlSize(.small).disabled(session.recipe.local == nil)
            }
            NumericSliderField(label: "Feather", value: Binding(get: { session.recipe.local?.feather ?? 0 },
                                                                set: { v in session.set { r in
                                                                    if r.local == nil { r.local = EditLocalLayer() }
                                                                    r.local?.feather = v } }),
                               range: 0...1, step: 0.01, fractionDigits: 2, onEditingEnded: { session.commit() })
            localSlider("Exposure", \.exposure, -5...5)
            localSlider("Contrast", \.contrast, -1...1)
            localSlider("Highlights", \.highlights, -1...1)
            localSlider("Shadows", \.shadows, -1...1)
            localSlider("Temperature", \.temperature, -1...1)
            localSlider("Tint", \.tint, -1...1)
            localSlider("Saturation", \.saturation, -1...1)
            localSlider("Sharpen", \.sharpen, 0...1)
            Text("Paint on the image; adjustments here apply inside the mask only.").font(.caption2).foregroundStyle(.tertiary)
        }
    }

    private var subjectControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button { Task { await session.requestSubjectMask() } } label: { Label("Find Subject", systemImage: "person.crop.rectangle") }
                    .controlSize(.small).disabled(session.sourceImage == nil)
                if let s = session.subjectStatus { Text(s).font(.caption).foregroundStyle(.orange) }
            }
            Toggle("Remove Background", isOn: Binding(get: { session.recipe.subject.removeBackground },
                                                     set: { v in session.set { $0.subject.removeBackground = v }; session.commit() }))
                .disabled(session.subjectMask == nil)
            Toggle("Invert (keep background)", isOn: Binding(get: { session.recipe.subject.invert },
                                                            set: { v in session.set { $0.subject.invert = v }; session.commit() }))
                .disabled(session.subjectMask == nil || !session.recipe.subject.removeBackground)
            Text("Saves a transparent PNG when Remove Background is on.").font(.caption2).foregroundStyle(.tertiary)
        }
    }

    // MARK: - Save

    private func save(thenInpaint: Bool) async {
        isSaving = true; status = nil; isError = false
        defer { isSaving = false }
        do {
            let dir = DesktopSettings.load().outputDirectory
            let path = try await session.export(outputDirectory: dir, ingestor: ingestor)
            status = "Saved → \(URL(fileURLWithPath: path).lastPathComponent)"
            if thenInpaint { onSendToInpaint?(path, session.recipe.local?.mask) }
        } catch {
            status = error.localizedDescription; isError = true
        }
    }
}

// MARK: - Crop overlay

/// Draggable crop rectangle drawn over the fitted preview. `crop` is normalized to the preview frame.
struct CropOverlay: View {
    let rect: CGRect
    @Binding var crop: CGRect
    var onCommit: () -> Void
    @State private var dragStart: CGRect?

    private enum Handle: CaseIterable { case tl, tr, bl, br, move }

    var body: some View {
        let c = CGRect(x: rect.minX + crop.minX * rect.width, y: rect.minY + crop.minY * rect.height,
                       width: crop.width * rect.width, height: crop.height * rect.height)
        ZStack {
            Path { p in p.addRect(rect); p.addRect(c) }
                .fill(Color.black.opacity(0.45), style: FillStyle(eoFill: true))
                .allowsHitTesting(false)
            Rectangle().stroke(Color.white, lineWidth: 1).frame(width: c.width, height: c.height)
                .position(x: c.midX, y: c.midY)
                .contentShape(Rectangle())
                .gesture(drag(.move, c))
            ForEach(Array([Handle.tl, .tr, .bl, .br].enumerated()), id: \.offset) { _, h in
                Circle().fill(Color.white).frame(width: 10, height: 10)
                    .position(handlePoint(h, c))
                    .gesture(drag(h, c))
            }
        }
    }

    private func handlePoint(_ h: Handle, _ c: CGRect) -> CGPoint {
        switch h {
        case .tl: return CGPoint(x: c.minX, y: c.minY)
        case .tr: return CGPoint(x: c.maxX, y: c.minY)
        case .bl: return CGPoint(x: c.minX, y: c.maxY)
        case .br: return CGPoint(x: c.maxX, y: c.maxY)
        case .move: return CGPoint(x: c.midX, y: c.midY)
        }
    }

    private func drag(_ h: Handle, _ c: CGRect) -> some Gesture {
        DragGesture(minimumDistance: 1)
            .onChanged { v in
                if dragStart == nil { dragStart = crop }
                guard let start = dragStart else { return }
                let dx = v.translation.width / rect.width, dy = v.translation.height / rect.height
                var n = start
                switch h {
                case .move:
                    n.origin.x = min(max(start.minX + dx, 0), 1 - start.width)
                    n.origin.y = min(max(start.minY + dy, 0), 1 - start.height)
                case .tl: n = CGRect(x: start.minX + dx, y: start.minY + dy, width: start.width - dx, height: start.height - dy)
                case .tr: n = CGRect(x: start.minX, y: start.minY + dy, width: start.width + dx, height: start.height - dy)
                case .bl: n = CGRect(x: start.minX + dx, y: start.minY, width: start.width - dx, height: start.height + dy)
                case .br: n = CGRect(x: start.minX, y: start.minY, width: start.width + dx, height: start.height + dy)
                }
                n = n.standardized.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
                if n.width >= 0.05, n.height >= 0.05 { crop = n }
            }
            .onEnded { _ in dragStart = nil; onCommit() }
    }
}

// MARK: - Tab wrapper

struct EditTab: View {
    var ingestor: AssetIngestor?
    @Binding var pendingImage: String?
    var onSendToInpaint: ((String, MaskStrokes?) -> Void)?

    @State private var session: EditSession?

    var body: some View {
        Group {
            if let session {
                EditView(session: session, ingestor: ingestor, onSendToInpaint: onSendToInpaint)
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "slider.horizontal.3").font(.system(size: 44)).foregroundStyle(.tertiary)
                    Text("Open an image to edit").foregroundStyle(.secondary)
                    Button { pickImage() } label: { Label("Open…", systemImage: "photo.badge.plus") }
                        .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("Edit")
        .toolbar { ToolbarItem(placement: .navigation) { Button { pickImage() } label: { Label("Open", systemImage: "folder") } } }
        .onAppear { consumePending() }
        .onChange(of: pendingImage) { _, _ in consumePending() }
    }

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true; panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.png, .jpeg, .tiff]
        if panel.runModal() == .OK, let url = panel.url { open(url.path, asset: nil) }
    }

    private func consumePending() {
        guard let p = pendingImage, !p.isEmpty else { return }
        pendingImage = nil
        open(p, asset: nil)
    }

    private func open(_ path: String, asset: DAMAsset?) {
        let s = EditSession(sourcePath: path, sourceAsset: asset)
        session = s
        Task { await s.load() }
    }
}
```

`NumericSliderField` must accept `onEditingEnded`. Check its definition (`grep -n "struct NumericSliderField" Sources/ComfyBoxDesktop -r`). If it has no such parameter, add `var onEditingEnded: (() -> Void)? = nil` and call it from the `Slider(... onEditingChanged: { editing in if !editing { onEditingEnded?() } })`; keep every existing call site compiling (the parameter is optional with a default).

- [ ] **Step 2: Build** — build-only command. Expected: no errors. Fix any SwiftUI type-check timeouts by splitting `panel` into smaller computed properties; do not change behaviour.

- [ ] **Step 3: Commit**

```bash
git add Sources/ComfyBoxDesktop/Views/EditView.swift Sources/ComfyBoxDesktop/Views/NumericSliderField.swift
git commit -m "feat(desktop): EditView, CropOverlay, and EditTab"
```

(If `NumericSliderField` lives in another file, add that file instead.)

---

### Task 11: App wiring — tab, callbacks, asset detail, gallery

**Files:**
- Modify: `Sources/ComfyBoxDesktop/ComfyBoxDesktopApp.swift` (enum ~line 62-82, section ~97, icon ~121, shortcut ~157, pending state ~38, gallery callbacks ~436-466, tab switch ~545)
- Modify: `Sources/ComfyBoxDesktop/Views/GalleryView.swift` (`onInpaint` at 51 and 933)
- Modify: `Sources/ComfyBoxDesktop/Views/AssetDetailView.swift` (init 64-80, buttons 160-166)
- Modify: `docs/user-guide.md` (add an Edit tab section)

**Interfaces:**
- Consumes: `EditTab`, `MaskStrokes`, `EditSidecar.read`.
- Produces: `AppTab.edit`; `@State pendingEditImage: String?`; `GalleryView.onEdit: ((DAMAsset) -> Void)?`; `AssetDetailView.onEdit: ((DAMAsset) -> Void)?` and `onSelectSource: ((String) -> Void)?`.

- [ ] **Step 1: App enum and wiring**

In `ComfyBoxDesktopApp.swift`:
1. Add `case edit = "Edit"` after `case inpaint = "Inpaint"`.
2. Section switch (line ~97): add `.edit` to the `.create` list.
3. `icon`: `case .edit: return "slider.horizontal.3"`.
4. Shortcut switch (~line 157): `case .edit: return "u"`.
5. State: `@State private var pendingEditImage: String?` beside `pendingInpaintImage`.
6. Tab switch: after the `.inpaint` case add
   ```swift
   case .edit:
       EditTab(ingestor: ingestor, pendingImage: $pendingEditImage, onSendToInpaint: { path, mask in
           pendingInpaintMask = mask
           pendingInpaintImage = path
           selectedTab = .inpaint
       })
   ```
7. Gallery callbacks (after `onInpaint:`): 
   ```swift
   onEdit: { asset in
       pendingEditImage = asset.absolutePath
       selectedTab = .edit
   },
   ```

- [ ] **Step 2: GalleryView**

Add `var onEdit: ((DAMAsset) -> Void)?` after `onInpaint` (line 51). Where the context menu has `Button("Edit / Inpaint") { onInpaint?(asset) }` (line ~933), add before it:
```swift
if onEdit != nil {
    Button("Edit") { onEdit?(asset) }
}
```
and rename the existing button's title to `"Inpaint"`. Find where `GalleryView` constructs `AssetDetailView` and pass `onEdit: onEdit` and `onSelectSource` (below).

- [ ] **Step 3: AssetDetailView**

1. Add `var onEdit: ((DAMAsset) -> Void)?` and `var onSelectSource: ((String) -> Void)?` next to `onSendToGenerate`, with matching `init` parameters defaulting to nil (both inits).
2. In the header button row, after the Send to Generate button:
   ```swift
   if let onEdit {
       Button { onEdit(asset) } label: { Label("Edit", systemImage: "slider.horizontal.3") }
           .disabled(localOnlyReason != nil)
           .help(localOnlyReason ?? "Open in the Edit tab")
   }
   ```
3. Below the filename/size metadata, add:
   ```swift
   if let sc = EditSidecar.read(forImageAt: asset.absolutePath) {
       HStack(spacing: 6) {
           Label("Edited from \(URL(fileURLWithPath: sc.sourcePath).lastPathComponent)", systemImage: "link")
               .font(.caption).foregroundStyle(.secondary)
           if let onSelectSource {
               Button("Show") { onSelectSource(sc.sourcePath) }.controlSize(.mini)
           }
       }
   }
   ```
   `onSelectSource` in `GalleryView` selects the asset whose `absolutePath` matches, if present; otherwise no-op.

- [ ] **Step 4: Docs**

Append to `docs/user-guide.md` a section:

```markdown
## Edit tab

Non-destructive tone, color, crop, local brush adjustments, and background removal for any PNG/JPEG/TIFF. Open from the sidebar (⌘U), from an asset's **Edit** button, or from the gallery context menu.

- Sliders preview live; one drag is one undo step (⌘Z / ⇧⌘Z).
- **Local** paints a mask; its sliders apply inside the mask only. **Feather** softens the edge.
- **Subject → Find Subject** runs Vision; **Remove Background** saves a transparent PNG.
- **Save** writes `edit-<time>.png` into your output folder as a new asset. The recipe is stored in the adjacent `.json`, so opening the saved asset in Edit reopens the original pixels with your settings.
- **Save & Inpaint** saves, then opens the result in Inpaint with your local mask pre-painted.
```

- [ ] **Step 5: Build and run the full desktop suite**

Build-only command, then `-only-testing:ComfyBoxDesktopTests` (whole target). Expected: build clean; all desktop tests pass including the pre-existing ones.

- [ ] **Step 6: Commit**

```bash
git add Sources/ComfyBoxDesktop/ComfyBoxDesktopApp.swift Sources/ComfyBoxDesktop/Views/GalleryView.swift Sources/ComfyBoxDesktop/Views/AssetDetailView.swift docs/user-guide.md
git commit -m "feat(desktop): Edit tab wiring, gallery/detail entry points, Edited-from"
```

---

## Self-review

- **Spec coverage**: recipe model (T3), renderer order and mappings (T4/T5), MaskCanvas extraction with parity (T1/T2), SubjectMasker (T6), exporter write order + root-source chaining + copied fields (T7), session undo/debounce/reopen/version warnings (T8), curves UI (T9), EditView groups/toolbar/crop overlay/tab (T10), entry points, Edited-from, Send-to-Inpaint mask, docs (T11). Error table: unreadable source (T8 warning), no subject / Vision failure (T8), preview failure (T8), export failure (T7), newer version (T8). Live checks remain Todd's.
- **Type consistency**: `MaskStrokes`/`MaskStroke`/`MaskRasterizer.render(_:size:)` used identically in T2, T5, T10; `EditSession.set/commit/renderNow/export` match between T8 and T10; `EditSidecar.read(forImageAt:)`/`rootSource` match T7, T8, T11; `onSendToInpaint: (String, MaskStrokes?)` matches T10 and T11; `pendingInpaintMask` introduced in T2 and consumed in T11.
- **Spec deviation to note**: contrast maps to 0.5…1.5 (spec text said 0…2 but its own test example said −1 → 0.5; the test wins). Amend the spec line when T4 lands.
