// DAMAsset.swift — Digital Asset Management model
//
// Core data model for tracked image/video assets. Each generated image
// is recorded in the DAM database with its generation parameters,
// file metadata, and user annotations (rating, favorite, content mode).

import Foundation

public struct DAMAsset: Identifiable, Sendable, Equatable {
    public let id: String
    public let kind: String  // "image" or "video"
    public let filename: String
    public let absolutePath: String
    public let fileSize: Int64
    public let sha256: String?
    public let width: Int?
    public let height: Int?
    public let createdAt: Date
    public let modifiedAt: Date
    public let ingestedAt: Date
    public let orphaned: Bool
    public let prompt: String?
    public let negativePrompt: String?
    public let seed: Int?
    public let steps: Int?
    public let guidance: Double?
    public let modelFamily: String?
    public let rating: Int
    public let favorite: Bool
    public let contentMode: String?
    public let characterName: String?
    /// Which app/persona generated this image (desktop / bree / kira / api…).
    /// Used to place persona renders (Kira, Bree) in their own gallery sections.
    public let source: String?

    public init(
        id: String = UUID().uuidString,
        kind: String = "image",
        filename: String,
        absolutePath: String,
        fileSize: Int64 = 0,
        sha256: String? = nil,
        width: Int? = nil,
        height: Int? = nil,
        createdAt: Date = Date(),
        modifiedAt: Date = Date(),
        ingestedAt: Date = Date(),
        orphaned: Bool = false,
        prompt: String? = nil,
        negativePrompt: String? = nil,
        seed: Int? = nil,
        steps: Int? = nil,
        guidance: Double? = nil,
        modelFamily: String? = nil,
        rating: Int = 0,
        favorite: Bool = false,
        contentMode: String? = nil,
        characterName: String? = nil,
        source: String? = nil
    ) {
        self.source = source
        self.id = id
        self.kind = kind
        self.filename = filename
        self.absolutePath = absolutePath
        self.fileSize = fileSize
        self.sha256 = sha256
        self.width = width
        self.height = height
        self.createdAt = createdAt
        self.modifiedAt = modifiedAt
        self.ingestedAt = ingestedAt
        self.orphaned = orphaned
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.seed = seed
        self.steps = steps
        self.guidance = guidance
        self.modelFamily = modelFamily
        self.rating = rating
        self.favorite = favorite
        self.contentMode = contentMode
        self.characterName = characterName
    }

    /// A copy of this asset pointing at a new file location (secure/unsecure
    /// moves); filename follows the path.
    public func withLocation(path: String) -> DAMAsset {
        copy(with: Mutation(
            filename: .value((path as NSString).lastPathComponent),
            absolutePath: .value(path)
        ))
    }

    // MARK: - copy(with:)
    //
    // #372: `DAMAsset.init` defaults all 22 parameters, so a hand-rebuilt
    // construction literal that forgets a field compiles silently instead
    // of failing to build — `source` was dropped this way at three separate
    // call sites, twice (#268). Every call site that is a COPY of an
    // existing `DAMAsset` (as opposed to a fresh ingest, or a rebuild from a
    // different stored representation such as a SQL row or an archive
    // manifest entry) must go through `copy(with:)` instead of writing out
    // the initializer by hand. `DAMAssetConstructionSiteLintTests` enforces
    // this for everything under `Sources/ComfyBoxDesktop`.

    /// Whether a `Mutation` field overrides the receiver's current value, or
    /// leaves it untouched. This is deliberately NOT a doubly-nested Optional
    /// (`T??`, "outer nil = untouched, inner nil = explicit override to
    /// nil"): Swift silently promotes a plain `T?` expression into a `T??`
    /// parameter, so `mutation.field ?? existing.field` written where a
    /// `T??` parameter is expected type-checks and compiles — but resolves
    /// to the *`T?`-returning* overload of `??`, which discards
    /// `existing.field` unconditionally and never even evaluates it,
    /// wrapping whatever `mutation.field` was (nil included) straight back
    /// into an explicit override. That silently reintroduces exactly the
    /// bug class this type exists to prevent, at the one call site
    /// (`DAMStore.mergedWithExisting`) that merges two assets field-by-field.
    /// `.value` requires the caller to say so with a distinct case the
    /// compiler cannot conjure from a bare optional; `.unchanged` is the
    /// only default.
    public enum Override<Value> {
        case unchanged
        case value(Value)

        fileprivate func applied(to current: Value) -> Value {
            switch self {
            case .unchanged: return current
            case .value(let value): return value
            }
        }
    }

    /// Field-level overrides for `copy(with:)`. Every field defaults to
    /// `.unchanged` (keep the receiver's current value).
    public struct Mutation {
        public var id: Override<String> = .unchanged
        public var kind: Override<String> = .unchanged
        public var filename: Override<String> = .unchanged
        public var absolutePath: Override<String> = .unchanged
        public var fileSize: Override<Int64> = .unchanged
        public var sha256: Override<String?> = .unchanged
        public var width: Override<Int?> = .unchanged
        public var height: Override<Int?> = .unchanged
        public var createdAt: Override<Date> = .unchanged
        public var modifiedAt: Override<Date> = .unchanged
        public var ingestedAt: Override<Date> = .unchanged
        public var orphaned: Override<Bool> = .unchanged
        public var prompt: Override<String?> = .unchanged
        public var negativePrompt: Override<String?> = .unchanged
        public var seed: Override<Int?> = .unchanged
        public var steps: Override<Int?> = .unchanged
        public var guidance: Override<Double?> = .unchanged
        public var modelFamily: Override<String?> = .unchanged
        public var rating: Override<Int> = .unchanged
        public var favorite: Override<Bool> = .unchanged
        public var contentMode: Override<String?> = .unchanged
        public var characterName: Override<String?> = .unchanged
        public var source: Override<String?> = .unchanged

        public init(
            id: Override<String> = .unchanged,
            kind: Override<String> = .unchanged,
            filename: Override<String> = .unchanged,
            absolutePath: Override<String> = .unchanged,
            fileSize: Override<Int64> = .unchanged,
            sha256: Override<String?> = .unchanged,
            width: Override<Int?> = .unchanged,
            height: Override<Int?> = .unchanged,
            createdAt: Override<Date> = .unchanged,
            modifiedAt: Override<Date> = .unchanged,
            ingestedAt: Override<Date> = .unchanged,
            orphaned: Override<Bool> = .unchanged,
            prompt: Override<String?> = .unchanged,
            negativePrompt: Override<String?> = .unchanged,
            seed: Override<Int?> = .unchanged,
            steps: Override<Int?> = .unchanged,
            guidance: Override<Double?> = .unchanged,
            modelFamily: Override<String?> = .unchanged,
            rating: Override<Int> = .unchanged,
            favorite: Override<Bool> = .unchanged,
            contentMode: Override<String?> = .unchanged,
            characterName: Override<String?> = .unchanged,
            source: Override<String?> = .unchanged
        ) {
            self.id = id
            self.kind = kind
            self.filename = filename
            self.absolutePath = absolutePath
            self.fileSize = fileSize
            self.sha256 = sha256
            self.width = width
            self.height = height
            self.createdAt = createdAt
            self.modifiedAt = modifiedAt
            self.ingestedAt = ingestedAt
            self.orphaned = orphaned
            self.prompt = prompt
            self.negativePrompt = negativePrompt
            self.seed = seed
            self.steps = steps
            self.guidance = guidance
            self.modelFamily = modelFamily
            self.rating = rating
            self.favorite = favorite
            self.contentMode = contentMode
            self.characterName = characterName
            self.source = source
        }
    }

    /// A copy of this asset with only the fields named in `mutation`
    /// overridden — every other field, including any added to `DAMAsset`
    /// after a given call site was written, carries over unchanged. This is
    /// the one place in the codebase allowed to rebuild a `DAMAsset`
    /// field-by-field; every other copy-of-an-existing-asset site should
    /// call this instead of writing the initializer out by hand.
    public func copy(with mutation: Mutation) -> DAMAsset {
        DAMAsset(
            id: mutation.id.applied(to: id),
            kind: mutation.kind.applied(to: kind),
            filename: mutation.filename.applied(to: filename),
            absolutePath: mutation.absolutePath.applied(to: absolutePath),
            fileSize: mutation.fileSize.applied(to: fileSize),
            sha256: mutation.sha256.applied(to: sha256),
            width: mutation.width.applied(to: width),
            height: mutation.height.applied(to: height),
            createdAt: mutation.createdAt.applied(to: createdAt),
            modifiedAt: mutation.modifiedAt.applied(to: modifiedAt),
            ingestedAt: mutation.ingestedAt.applied(to: ingestedAt),
            orphaned: mutation.orphaned.applied(to: orphaned),
            prompt: mutation.prompt.applied(to: prompt),
            negativePrompt: mutation.negativePrompt.applied(to: negativePrompt),
            seed: mutation.seed.applied(to: seed),
            steps: mutation.steps.applied(to: steps),
            guidance: mutation.guidance.applied(to: guidance),
            modelFamily: mutation.modelFamily.applied(to: modelFamily),
            rating: mutation.rating.applied(to: rating),
            favorite: mutation.favorite.applied(to: favorite),
            contentMode: mutation.contentMode.applied(to: contentMode),
            characterName: mutation.characterName.applied(to: characterName),
            source: mutation.source.applied(to: source)
        )
    }

    /// Shared editability predicate for the Edit action (gallery context menu and
    /// asset detail view): image-kind assets in a format the editor can decode.
    /// Both entry points use this one definition so they never disagree — offering
    /// Edit for a video or an unsupported format dead-ends in "Couldn't read x.mp4".
    public var isEditableImage: Bool {
        guard kind == "image" else { return false }
        let ext = (filename as NSString).pathExtension.lowercased()
        return ["png", "jpg", "jpeg", "tif", "tiff"].contains(ext)
    }
}
