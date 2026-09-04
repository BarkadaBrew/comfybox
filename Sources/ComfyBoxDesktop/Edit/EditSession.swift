// EditSession.swift — observable state for one open image in the editor
//
// Owns the source pixels, the live recipe, undo/redo, and a debounced
// preview render on a background task. The preview source is downscaled
// once; the exporter re-renders from the full-resolution source.

import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
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
    /// When true, the preview is rendered with `recipe.geometry.crop` cleared
    /// (every other stage still applied) so a crop-overlay UI can draw its
    /// handles over the uncropped frame. Export always honors the real crop.
    public var suppressCropForPreview = false {
        didSet {
            guard suppressCropForPreview != oldValue else { return }
            scheduleRender()
        }
    }

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
            if root.path == sourcePath {
                // `rootSource` reports a cycle by returning the starting path
                // itself with a nil asset id — the chain is malformed, so
                // keep the derived pixels already on disk and fall back to
                // an identity recipe rather than guessing at a root.
                warning = "This edit's history is malformed; editing the flattened image."
            } else if FileManager.default.fileExists(atPath: root.path) {
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
        var effectiveRecipe = self.recipe
        if suppressCropForPreview { effectiveRecipe.geometry.crop = nil }
        let mask = scaledSubjectMask()
        // `CIContext`/`CIImage` are not marked `Sendable` prior to macOS 15,
        // but this pipeline is pure-value math over Core Image's own
        // reference-counted graph description (no shared mutable state) —
        // safe to hand across the actor boundary. Boxed as `nonisolated(unsafe)`
        // locals rather than moving the render onto the main actor.
        nonisolated(unsafe) let source = previewSource
        nonisolated(unsafe) let ctx = context
        nonisolated(unsafe) let subjectMaskForRender = mask
        let result = await Task.detached(priority: .userInitiated) { () -> CGImage? in
            let out = EditRenderer.render(source: source, recipe: effectiveRecipe, subjectMask: subjectMaskForRender)
            guard !out.extent.isEmpty, !out.extent.isInfinite else { return nil }
            return ctx.createCGImage(out, from: out.extent)
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
