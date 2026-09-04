import Foundation
@preconcurrency import MLX

public enum LoRASource: Sendable, Equatable {
    case local(URL)
    case huggingFace(modelId: String, filename: String?)

    public var displayName: String {
        switch self {
        case .local(let url):
            return url.lastPathComponent
        case .huggingFace(let modelId, _):
            return modelId.components(separatedBy: "/").last ?? modelId
        }
    }

    public var isLocal: Bool {
        if case .local = self { return true }
        return false
    }
}

public struct LoRAConfiguration: Sendable, Equatable {
    public let source: LoRASource
    public let scale: Float
    /// Krea-2 relativity (WP-E6, FDD §3.6 / AC-41): the physical base this
    /// adapter was extracted against. DECLARED, never inferred from the
    /// file — set by the caller, else filled from the library entry's
    /// `krea2_relative`, else ``Krea2LoRARelativity/seeded(forFilename:)``.
    /// `nil` declares nothing. Ignored by every non-Krea-2 pipeline.
    public var requiresBase: Krea2Variant?
    /// The configuration SLOT this adapter fills (WP-E10, FDD §3.10
    /// `Applied.role`): `"kroma"` | `"accel"` | `"bypass"` | `"control"`, or
    /// nil when undeclared. Labelled once, where the stack is built, and read
    /// back into `RenderRecipe.loras[].role` so the client reports
    /// `kroma_strength` AS APPLIED instead of matching filenames (AC-45).
    /// Ignored by every non-Krea-2 pipeline.
    public var role: String?

    public init(source: LoRASource, scale: Float = 1.0, requiresBase: Krea2Variant? = nil, role: String? = nil) {
        self.source = source
        self.scale = scale
        self.requiresBase = requiresBase
        self.role = role
    }

    public static func local(_ path: String, scale: Float = 1.0, requiresBase: Krea2Variant? = nil) -> LoRAConfiguration {
        LoRAConfiguration(source: .local(URL(fileURLWithPath: path)), scale: scale, requiresBase: requiresBase)
    }

    public static func local(_ url: URL, scale: Float = 1.0, requiresBase: Krea2Variant? = nil) -> LoRAConfiguration {
        LoRAConfiguration(source: .local(url), scale: scale, requiresBase: requiresBase)
    }

    public static func huggingFace(_ modelId: String, filename: String? = nil, scale: Float = 1.0) -> LoRAConfiguration {
        LoRAConfiguration(source: .huggingFace(modelId: modelId, filename: filename), scale: scale)
    }
}
public struct LoRAWeights: @unchecked Sendable {
    public let weights: [String: (down: MLXArray, up: MLXArray)]
    public let lokrWeights: [String: LoKrWeights]
    public let rank: Int
    public let alpha: Float
    /// Per-layer alpha values from kohya-style `<module>.alpha` tensors,
    /// keyed the same way as `weights` (typically ending in ".weight").
    public let layerAlphas: [String: Float]
    /// The alpha that was explicitly provided by the adapter (PEFT
    /// adapter_config.json, ss_network_alpha metadata, ...). When nil,
    /// `alpha` was defaulted to `rank` (effective scale 1.0).
    private let explicitAlpha: Float?

    /// Bare-parameter patches (.diff / .diff_b / .set_weight) keyed by the
    /// target parameter path in the model (diff_b keys already end in
    /// ".bias"). Applied via ``LoRAPatchSession``, never alpha/rank scaled.
    public let deltas: [String: DeltaPatch]

    public init(
        weights: [String: (down: MLXArray, up: MLXArray)],
        lokrWeights: [String: LoKrWeights] = [:],
        rank: Int,
        alpha: Float? = nil,
        layerAlphas: [String: Float] = [:],
        deltas: [String: DeltaPatch] = [:]
    ) {
        self.weights = weights
        self.lokrWeights = lokrWeights
        self.rank = rank
        self.explicitAlpha = alpha
        self.alpha = alpha ?? Float(rank)
        self.layerAlphas = layerAlphas
        self.deltas = deltas
    }

    public var effectiveScale: Float {
        guard rank > 0 else { return 1.0 }
        return alpha / Float(rank)
    }

    /// Effective scale (alpha / rank) for a specific layer, following the
    /// kohya/PEFT convention. Uses the layer's own rank (inner dimension of
    /// its down/up pair) and its per-layer alpha tensor when present, falling
    /// back to the adapter-wide alpha, then to 1.0 (alpha == rank).
    public func effectiveScale(forLayer key: String) -> Float {
        let weightKey = key.hasSuffix(".weight") ? key : key + ".weight"
        let baseKey = String(weightKey.dropLast(".weight".count))

        var layerRank = 0
        if let down = (weights[weightKey] ?? weights[baseKey])?.down, down.ndim == 2 {
            layerRank = min(down.dim(0), down.dim(1))
        }

        guard let layerAlpha = layerAlphas[weightKey] ?? layerAlphas[baseKey] ?? explicitAlpha else {
            return 1.0
        }
        guard layerRank > 0 else { return effectiveScale }
        return layerAlpha / Float(layerRank)
    }

    public var layerCount: Int {
        weights.count
    }

    public var lokrLayerCount: Int {
        lokrWeights.count
    }

    public var hasLoKr: Bool {
        !lokrWeights.isEmpty
    }

    /// A copy with a different pair/delta split and everything else preserved
    /// EXACTLY — rank, LoKr, per-layer alphas, and crucially whether an alpha
    /// was ever explicitly declared. Rebuilding via the public initialiser
    /// with `alpha: self.alpha` would promote a defaulted alpha (`Float(rank)`)
    /// into an explicit one and silently rescale every remaining pair, so the
    /// re-split lives here, where `explicitAlpha` is visible.
    /// Used by ``LoRABareParameterPairs/split(_:for:name:)``.
    public func withPairsAndDeltas(
        weights: [String: (down: MLXArray, up: MLXArray)],
        deltas: [String: DeltaPatch]
    ) -> LoRAWeights {
        LoRAWeights(
            weights: weights,
            lokrWeights: lokrWeights,
            rank: rank,
            alpha: explicitAlpha,
            layerAlphas: layerAlphas,
            deltas: deltas)
    }
}

/// A bare-parameter patch shipped inside a LoRA file (ComfyUI comfy/lora.py
/// semantics). Unlike low-rank pairs these have no rank structure: `.diff`
/// adds `userScale × delta` to the target parameter, `.diff_b` does the same
/// on the target's real `.bias`, and `.set_weight` replaces the parameter
/// outright, ignoring userScale entirely. Deltas are NEVER alpha/rank scaled.
public enum DeltaPatch: @unchecked Sendable {
    case diff(MLXArray)
    case diffBias(MLXArray)
    case setWeight(MLXArray)

    public var tensor: MLXArray {
        switch self {
        case .diff(let t), .diffBias(let t), .setWeight(let t): return t
        }
    }
}

public struct LoKrWeights: @unchecked Sendable {
    public let w1: MLXArray
    public let w2: MLXArray
    public let alpha: Float?

    public init(w1: MLXArray, w2: MLXArray, alpha: Float? = nil) {
        self.w1 = w1
        self.w2 = w2
        self.alpha = alpha
    }
}

public enum LoRAError: Error, LocalizedError {
    case fileNotFound(String)
    case invalidFormat(String)
    case incompatibleWeights(String)
    case downloadFailed(String, Error)
    case noSafetensorsFound(URL)
    /// The file uses a recognised adapter feature we deliberately do not
    /// support yet (DoRA, LoCon mid blocks, …). Explicit refusal beats
    /// silently applying a fraction of the adapter.
    case unsupportedFeature(String)
    /// Tensor keys matching no known suffix. Listed so an update to the
    /// loader can be scoped precisely instead of guessed at.
    case unknownKeys([String])
    /// Some offered keys bound nothing. Thrown BEFORE any mutation — by
    /// ``LoRAPatchSession`` preflight (delta keys with no target parameter;
    /// `lora` is nil there) and by ``LoRAApplicator/applyDynamically`` under
    /// `strict: true` (offered pair keys that matched no module — WP-E6,
    /// FDD D9 / AC-42a). `unbound` is the sorted list of offending keys.
    case partialApplication(lora: String?, unbound: [String])
    /// WP-E6 / AC-41: the adapter declares (or is seeded as) relative to one
    /// Krea-2 base and a different one is loaded. Thrown before any weight
    /// is touched.
    case incompatibleBase(lora: String, requires: Krea2Variant, loaded: Krea2Variant)
    /// K-FIX-1 / Codex C1: the adapter's FORMAT is one this path refuses —
    /// not a parse failure and not a feature gap, but a format whose
    /// application cannot be rolled back here (LoKr rewrites base weights;
    /// see ``Krea2AdapterSupport``). Thrown before any weight is touched.
    case unsupportedAdapter(lora: String, format: String, reason: String)
    /// WP-E6 B4a / AC-49: the file is JSON (an HTTP error page saved under a
    /// `.safetensors` name — the 99-byte civitai early-access body in
    /// `fetch.log`) AND the safetensors header failed to parse. Never a size
    /// floor alone — the real 1,040-byte bypass LoRA must load.
    case notASafetensorsFile(String, firstBytes: String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "LoRA file not found: \(path)"
        case .invalidFormat(let message):
            return "Invalid LoRA format: \(message)"
        case .incompatibleWeights(let message):
            return "Incompatible LoRA weights: \(message)"
        case .downloadFailed(let modelId, let error):
            return "Failed to download LoRA '\(modelId)': \(error.localizedDescription)"
        case .noSafetensorsFound(let url):
            return "No .safetensors files found in \(url.path)"
        case .unsupportedFeature(let feature):
            return "LoRA uses an unsupported adapter feature: \(feature)"
        case .unknownKeys(let keys):
            return "LoRA contains unrecognised tensor keys: \(keys.sorted().joined(separator: ", "))"
        case .partialApplication(let lora, let unbound):
            let who = lora.map { "LoRA '\($0)'" } ?? "LoRA patch"
            let shown = unbound.sorted()
            let listed = shown.prefix(32).joined(separator: ", ")
            let more = shown.count > 32 ? " … (+\(shown.count - 32) more)" : ""
            return "\(who) did not bind completely — \(shown.count) key(s) matched nothing: \(listed)\(more)"
        case .incompatibleBase(let lora, let requires, let loaded):
            return "LoRA '\(lora)' is extracted against Krea-2 \(requires.rawValue) but the loaded base is \(loaded.rawValue) — refusing to apply a \(requires.rawValue)-relative adapter on \(loaded.rawValue)"
        case .unsupportedAdapter(let lora, let format, let reason):
            return "LoRA '\(lora)' is a \(format) adapter, which this model path refuses: \(reason)"
        case .notASafetensorsFile(let path, let firstBytes):
            return "'\(path)' is not a safetensors file — it is JSON (an HTTP error page saved under the wrong name?): \(firstBytes)"
        }
    }
}

public struct LoRAValidationResult: Sendable, Equatable {
    public let isValid: Bool
    public let rank: Int
    public let targetLayers: [String]
    public let estimatedMemoryMB: Int
    public let errorMessage: String?

    public init(isValid: Bool, rank: Int, targetLayers: [String], estimatedMemoryMB: Int, errorMessage: String? = nil) {
        self.isValid = isValid
        self.rank = rank
        self.targetLayers = targetLayers
        self.estimatedMemoryMB = estimatedMemoryMB
        self.errorMessage = errorMessage
    }

    public static func invalid(_ message: String) -> LoRAValidationResult {
        LoRAValidationResult(isValid: false, rank: 0, targetLayers: [], estimatedMemoryMB: 0, errorMessage: message)
    }
}
