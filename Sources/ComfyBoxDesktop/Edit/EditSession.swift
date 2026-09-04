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
    /// The path this session was opened with — before `load()` may resolve
    /// it forward to a root original. Exports pass this (not the resolved
    /// `sourcePath`) to `EditExporter`, so the exporter's own sidecar-chain
    /// walk starts from the file that actually carries the chain and lands
    /// on the correct root path *and* root asset id, rather than treating
    /// an already-resolved root as if it had no history of its own.
    public private(set) var openedPath: String
    /// The path `export()` hands `EditExporter` as the new sidecar's lineage
    /// anchor. Defaults to `openedPath`; `load()` overrides it (see `followLineage`)
    /// when the real chain couldn't be followed and pixels were flattened instead.
    private var exportSourcePath: String
    /// Whether `EditExporter` should re-walk `exportSourcePath`'s own sidecar chain
    /// to find the true root (the normal case), or trust `exportSourcePath` as the
    /// lineage anchor AS-IS with no further walking.
    ///
    /// `load()`'s missing-root and malformed-chain fallbacks both edit FLATTENED
    /// pixels (the derived file itself, not a resolved root) while leaving
    /// `openedPath` pointing at that same derived file. If `export()` still asked
    /// `EditExporter` to walk the chain from `openedPath` in that case, it would
    /// re-discover the exact same missing/broken root and record it as the new
    /// sidecar's source — even though the pixels just edited and rendered were the
    /// flattened ones, not that root. Setting this false on those two fallbacks
    /// makes the new sidecar point at what was actually edited.
    private var followLineage = true
    public private(set) var sourceImage: CGImage?
    public private(set) var sourceAsset: DAMAsset?
    public var recipe = EditRecipe()
    public private(set) var preview: CGImage?
    public private(set) var isRendering = false
    public var showOriginal = false
    public private(set) var warning: String?
    /// Whether the live recipe differs from what was last loaded from disk
    /// or successfully exported. Computed (not a manually-tracked flag) so
    /// there is exactly one source of truth: `recipe` vs. `savedRecipe`.
    public var isDirty: Bool { recipe != savedRecipe }
    public private(set) var subjectMask: CIImage?
    /// Vision's own detection outcome ("Finding subject…", "No subject found.",
    /// "Vision failed: …"), set only by `requestSubjectMask()`. Kept separate from
    /// `subjectMaskWarning` so a subsequent preview render — which recomputes the
    /// "Remove Background is on but no subject mask is loaded" warning on every
    /// frame — can never clobber "No subject found." before the user has seen it.
    public private(set) var subjectStatus: String?
    /// "Remove Background is on but no subject mask is loaded" — recomputed by
    /// every render (see `render(generation:)`), independent of `subjectStatus`.
    public private(set) var subjectMaskWarning: String?
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

    /// The recipe as it was when last loaded from disk, or as of the last
    /// successful export — the baseline `isDirty` compares against. Distinct
    /// from `committed`, which is the undo/redo baseline: undoing all the
    /// way back to the loaded recipe clears `isDirty` even if the undo
    /// stack itself is still non-empty (i.e. `canUndo` can be true while
    /// `isDirty` is false).
    private var savedRecipe = EditRecipe()
    private var committed = EditRecipe()
    private var undoStack: [EditRecipe] = []
    private var redoStack: [EditRecipe] = []
    private let undoLimit = 100

    private let previewMaxDimension: CGFloat
    private var previewSource: CIImage?
    private var previewScale: CGFloat = 1
    private var renderTask: Task<Void, Never>?
    private var renderGeneration = 0
    /// The in-flight (or most recently started) detached Core Image render.
    /// Cancelled before a new one starts so at most one render is doing
    /// real work at a time — coalescing, not true mid-render interruption:
    /// a render already inside Core Image is not itself interruptible.
    private var renderWorker: Task<CGImage?, Never>?
    private let context = CIContext(options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                                              .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
    private let masker = SubjectMasker()

    public var canUndo: Bool { !undoStack.isEmpty }
    public var canRedo: Bool { !redoStack.isEmpty }

    public init(sourcePath: String, sourceAsset: DAMAsset?, previewMaxDimension: CGFloat = 2048) {
        self.sourcePath = sourcePath
        self.openedPath = sourcePath
        self.exportSourcePath = sourcePath
        self.sourceAsset = sourceAsset
        self.previewMaxDimension = previewMaxDimension
    }

    // MARK: - Loading

    public func load() async {
        warning = nil
        var path = sourcePath
        var storedRecipe = EditRecipe()
        followLineage = true
        exportSourcePath = openedPath
        // Parse only the envelope first — a sidecar written by a newer
        // ComfyBox can carry a `recipe` shape this build can't decode, and a
        // full `EditSidecar.read` would then fail outright, indistinguishable
        // from "no sidecar at all". `rootSource` itself is envelope-based (see
        // its doc comment), so it survives an undecodable recipe anywhere in
        // the chain, including at the opened file itself — only the FULL
        // decode of the opened file's own sidecar (needed for the recipe, not
        // just the path) has to wait on its version being supported.
        if let envelope = EditSidecar.readEnvelope(forImageAt: sourcePath) {
            let root = EditSidecar.rootSource(forImageAt: sourcePath)
            if root.path == sourcePath {
                // `rootSource` reports a cycle by returning the starting path
                // itself with a nil asset id — the chain is malformed, so
                // keep the derived pixels already on disk and fall back to
                // an identity recipe rather than guessing at a root.
                warning = "This edit's history is malformed; editing the flattened image."
                followLineage = false; exportSourcePath = path
            } else if FileManager.default.fileExists(atPath: root.path) {
                path = root.path
                if envelope.version > EditRecipe.currentVersion {
                    warning = "This edit was saved by a newer ComfyBox (recipe v\(envelope.version)); opening pixels only."
                } else if let sc = EditSidecar.read(forImageAt: sourcePath) {
                    storedRecipe = sc.recipe
                } else {
                    // Envelope parsed but the full recipe didn't decode despite a
                    // compatible version — corrupt sidecar. Fall back to pixels-only.
                    warning = "This edit's recipe could not be read; opening pixels only."
                }
            } else {
                warning = "Original \(URL(fileURLWithPath: root.path).lastPathComponent) is missing; editing the flattened image."
                followLineage = false; exportSourcePath = path
            }
        }
        let loadPath = path
        nonisolated(unsafe) let orientationContext = context
        let image = await Task.detached(priority: .userInitiated) { () -> CGImage? in
            guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: loadPath) as CFURL, nil) else { return nil }
            guard let cg = CGImageSourceCreateImageAtIndex(src, 0, [kCGImageSourceShouldCache: false] as CFDictionary) else { return nil }
            // Honor EXIF orientation: ImageIO returns the pixels exactly as stored, not
            // rotated/flipped to "display" orientation, so a sidecar/tag of anything but 1
            // (normal) must be baked in now — every downstream stage (geometry, masks,
            // preview sizing) assumes the pixels are already display-oriented.
            var orientationValue: UInt32 = 1
            if let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
               let raw = props[kCGImagePropertyOrientation] as? NSNumber {
                orientationValue = raw.uint32Value
            }
            guard orientationValue != 1, let cgOrientation = CGImagePropertyOrientation(rawValue: orientationValue) else { return cg }
            var ci = CIImage(cgImage: cg).oriented(cgOrientation)
            ci = ci.transformed(by: CGAffineTransform(translationX: -ci.extent.minX, y: -ci.extent.minY))
            return orientationContext.createCGImage(ci, from: ci.extent) ?? cg
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
        recipe = storedRecipe; committed = storedRecipe; savedRecipe = storedRecipe
        undoStack.removeAll(); redoStack.removeAll()
        subjectMask = nil; subjectStatus = nil; subjectMaskWarning = nil
        scheduleRender()
    }

    // MARK: - Recipe mutation

    public func set(_ mutate: (inout EditRecipe) -> Void) {
        mutate(&recipe)
        scheduleRender()
    }

    public func commit() {
        guard recipe != committed else { return }
        undoStack.append(committed)
        if undoStack.count > undoLimit { undoStack.removeFirst() }
        redoStack.removeAll()
        committed = recipe
    }

    public func undo() {
        // An uncommitted live edit (e.g. mid-drag, before `onEditingEnded` calls
        // `commit()`) must not be silently discarded by reaching straight into the
        // undo stack — commit it first so it becomes the thing this undo reverts,
        // and a subsequent redo can still bring it back.
        if recipe != committed { commit() }
        guard let previous = undoStack.popLast() else { return }
        redoStack.append(committed)
        committed = previous; recipe = previous
        scheduleRender()
    }

    public func redo() {
        guard let next = redoStack.popLast() else { return }
        undoStack.append(committed)
        committed = next; recipe = next
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
        } catch SubjectMaskError.visionFailed(let message) {
            // `SubjectMaskError` doesn't conform to `LocalizedError`, so
            // `error.localizedDescription` on it (the generic catch below) produces
            // Swift's default "operation couldn't be completed" text, not the
            // underlying Vision error `message` this case actually carries.
            subjectMask = nil; subjectStatus = "Vision failed: \(message)"
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
        let scale = previewScale
        // Coalesce: cancel any still-running previous render before starting this
        // one. A render already inside Core Image is not interruptible mid-flight —
        // this only stops a worker that hasn't reached `EditRenderer.render` yet.
        renderWorker?.cancel()
        let worker = Task.detached(priority: .userInitiated) { () -> CGImage? in
            guard !Task.isCancelled else { return nil }
            // The preview source was downscaled by `previewScale` on load, so an
            // absolute-pixel filter parameter (sharpen's radius) must be scaled down to
            // match, or the preview and a full-resolution export disagree — see
            // `EditRenderer.sharpenRadius(scale:)`.
            let out = EditRenderer.render(source: source, recipe: effectiveRecipe, subjectMask: subjectMaskForRender, renderScale: scale)
            guard !out.extent.isEmpty, !out.extent.isInfinite else { return nil }
            return ctx.createCGImage(out, from: out.extent)
        }
        renderWorker = worker
        let result = await worker.value
        guard generation == renderGeneration else { return }
        isRendering = false
        if let result {
            preview = result
            previewSize = CGSize(width: result.width, height: result.height)
            // Only clear a warning THIS code set — a render succeeding says nothing
            // about an unrelated warning (e.g. a stale-history message from `load()`).
            if warning == Self.previewRenderFailedMessage { warning = nil }
        } else {
            warning = Self.previewRenderFailedMessage
        }
        // Recomputed every render so it clears itself the moment the condition no
        // longer holds — e.g. the user turns Remove Background back off, or a mask
        // finishes loading — rather than lingering as a stale one-shot warning that
        // nothing else ever un-sets. Kept in its own field (`subjectMaskWarning`),
        // separate from `subjectStatus`, so this can never clobber a Vision result
        // ("No subject found.") that `requestSubjectMask()` just set.
        subjectMaskWarning = (recipe.subject.removeBackground && subjectMask == nil) ? Self.noSubjectMaskMessage : nil
    }

    private static let previewRenderFailedMessage = "Preview render failed; the last good preview is shown."
    private static let noSubjectMaskMessage = "Remove Background is on but no subject mask is loaded. Run Find Subject."

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
        if recipe.subject.removeBackground && subjectMask == nil {
            throw EditExportError.writeFailed("Remove Background is on but no subject mask is loaded; run Find Subject or turn it off")
        }
        // Snapshot the recipe synchronously, before the internal await below can
        // yield the main actor to a concurrent `set(_:)` — the exported PNG and
        // the session's own bookkeeping afterward (`savedRecipe`) must agree on
        // exactly the recipe that was live at the moment export was requested,
        // not whatever `recipe` happens to read once the write completes.
        let exported = recipe
        let path = try await EditExporter.export(sourceImage: sourceImage, sourcePath: exportSourcePath, sourceAsset: sourceAsset,
                                                 recipe: exported, subjectMask: subjectMask,
                                                 outputDirectory: outputDirectory, ingestor: ingestor,
                                                 resolveLineage: followLineage)
        savedRecipe = exported
        return path
    }
}
