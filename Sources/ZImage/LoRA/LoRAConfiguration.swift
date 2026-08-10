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

    public init(source: LoRASource, scale: Float = 1.0) {
        self.source = source
        self.scale = scale
    }

    public static func local(_ path: String, scale: Float = 1.0) -> LoRAConfiguration {
        LoRAConfiguration(source: .local(URL(fileURLWithPath: path)), scale: scale)
    }

    public static func local(_ url: URL, scale: Float = 1.0) -> LoRAConfiguration {
        LoRAConfiguration(source: .local(url), scale: scale)
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
    /// Some bindable keys resolved to no target parameter. Thrown by
    /// preflight BEFORE any mutation.
    case partialApplication(missing: [String])

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
        case .partialApplication(let missing):
            return "LoRA patch preflight failed — no target parameter for: \(missing.sorted().joined(separator: ", "))"
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
