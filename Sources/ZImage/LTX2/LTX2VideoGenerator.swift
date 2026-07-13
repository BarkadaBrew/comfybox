// LTX2VideoGenerator.swift — Server-callable local LTX-2 video generation
//
// Lifts the proven `ComfyBox ltx2-i2v` CLI flow into a reusable service so the
// warm server can generate video locally (text-to-video and image-to-video)
// instead of only via the Replicate cloud proxy. Models are loaded lazily and
// cached; frame/chunk math and request validation are pure so they're testable
// without the 38 GB model.

import Foundation
import Logging
import MLX
import MLXNN

#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

/// One LoRA to merge into the LTX-2 transformer for a render, applied in
/// the order given. Mirrors the image side's {path, scale} shape so LTX
/// video LoRAs are managed the same way as every other model's.
public struct LTX2LoRAReference: Sendable, Equatable {
    public var path: String
    public var scale: Float
    public init(path: String, scale: Float = 1.0) {
        self.path = path
        self.scale = scale
    }
}

/// Parameters for one local LTX-2 video generation.
public struct LTX2VideoRequest: Sendable {
    public var prompt: String
    public var negativePrompt: String?
    /// nil → text-to-video; a path → image-to-video conditioned on it.
    public var initImagePath: String?
    public var width: Int
    public var height: Int
    /// Frames per chunk; must be 1 + 8k (9, 17, 25, …, 97).
    public var framesPerChunk: Int
    public var steps: Int
    public var seed: UInt64
    /// I2V conditioning strength (0–1).
    public var strength: Float
    /// Target duration; >0 generates continuation chunks (I2V only).
    public var extendToSeconds: Float
    public var fps: Int
    /// Deprecated single-LoRA fields — kept for wire/call-site back-compat.
    /// New callers should use `loras` instead. Folded into `effectiveLoRAs`.
    public var loraPath: String?
    public var loraStrength: Float
    /// LoRAs merged into the transformer for this render, applied in order.
    public var loras: [LTX2LoRAReference]
    public var outputPath: String

    /// `loras`, with the deprecated single `loraPath`/`loraStrength` (if set)
    /// prepended — the single field always applied first, matching the old
    /// single-LoRA behavior when only it is set.
    public var effectiveLoRAs: [LTX2LoRAReference] {
        var result: [LTX2LoRAReference] = []
        if let loraPath, !loraPath.isEmpty {
            result.append(LTX2LoRAReference(path: loraPath, scale: loraStrength))
        }
        result.append(contentsOf: loras)
        return result
    }

    public init(
        prompt: String,
        negativePrompt: String? = nil,
        initImagePath: String? = nil,
        width: Int = 704,
        height: Int = 448,
        framesPerChunk: Int = 97,
        steps: Int = 8,
        seed: UInt64 = 42,
        strength: Float = 1.0,
        extendToSeconds: Float = 0,
        fps: Int = 24,
        loraPath: String? = nil,
        loraStrength: Float = 1.0,
        loras: [LTX2LoRAReference] = [],
        outputPath: String
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.initImagePath = initImagePath
        self.width = width
        self.height = height
        self.framesPerChunk = framesPerChunk
        self.steps = steps
        self.seed = seed
        self.strength = strength
        self.extendToSeconds = extendToSeconds
        self.fps = fps
        self.loraPath = loraPath
        self.loraStrength = loraStrength
        self.loras = loras
        self.outputPath = outputPath
    }
}

public struct LTX2VideoResult: Sendable {
    public let outputPath: String
    public let frameCount: Int
    public let durationSeconds: Float
    public let elapsedSeconds: Double
}

public enum LTX2VideoError: Error, LocalizedError {
    case invalidFrameCount(Int)
    case invalidDimensions(Int, Int)
    case weightsMissing(String)
    case imageLoadFailed(String)
    case unsupportedPlatform

    public var errorDescription: String? {
        switch self {
        case .invalidFrameCount(let n):
            return "LTX-2 frames must be 1 + 8k (9, 17, 25, …, 97); got \(n)."
        case .invalidDimensions(let w, let h):
            return "LTX-2 width/height must be divisible by 32; got \(w)x\(h)."
        case .weightsMissing(let path):
            return "LTX-2 weights not found: \(path)"
        case .imageLoadFailed(let path):
            return "Failed to load init image: \(path)"
        case .unsupportedPlatform:
            return "LTX-2 video requires CoreGraphics/ImageIO (macOS)."
        }
    }
}

public final class LTX2VideoGenerator {
    public struct Configuration: Sendable {
        /// Directory holding transformer / vae_{encoder,decoder} / connector.
        public var weightsDir: String
        /// Gemma-3 tokenizer + text-encoder snapshot directory.
        public var gemmaPath: String
        /// Transformer weights filename inside `weightsDir`.
        public var transformerFile: String

        public init(
            weightsDir: String,
            gemmaPath: String,
            transformerFile: String = "transformer-distilled.safetensors"
        ) {
            self.weightsDir = weightsDir
            self.gemmaPath = gemmaPath
            self.transformerFile = transformerFile
        }
    }

    public let config: Configuration
    private let logger: Logger

    private var pipeline: LTX2Pipeline?
    private var tokenizer: LTX2GemmaTokenizer?
    public private(set) var isLoaded = false
    /// "path@strength" of the LoRA merged into the loaded transformer (nil = base).
    private var loadedLoraKey: String?

    public init(config: Configuration, logger: Logger = Logger(label: "ltx2.video")) {
        self.config = config
        self.logger = logger
    }

    // MARK: - Pure planning helpers (testable without the model)

    /// A frame count is valid when it's 1 + 8k and ≥ 9.
    public static func isValidFrameCount(_ n: Int) -> Bool {
        n >= 9 && (n - 1) % 8 == 0
    }

    /// Dimensions must be positive multiples of 32.
    public static func areValidDimensions(width: Int, height: Int) -> Bool {
        width > 0 && height > 0 && width % 32 == 0 && height % 32 == 0
    }

    public struct ChunkPlan: Equatable, Sendable {
        public let totalChunks: Int
        public let totalFrames: Int
        public let durationSeconds: Float
    }

    /// How many chunks and frames a request produces. Each continuation chunk
    /// re-uses the previous chunk's last frame, so it adds `framesPerChunk - 1`
    /// new frames. `extendToSeconds == 0` → a single chunk.
    public static func chunkPlan(framesPerChunk: Int, extendToSeconds: Float, fps: Int) -> ChunkPlan {
        let totalChunks: Int
        if extendToSeconds > 0 {
            let targetFrames = Int(extendToSeconds * Float(fps))
            let continuations = max(0, Int(ceil(Float(targetFrames - framesPerChunk) / Float(framesPerChunk - 1))))
            totalChunks = 1 + continuations
        } else {
            totalChunks = 1
        }
        let totalFrames = framesPerChunk + (framesPerChunk - 1) * (totalChunks - 1)
        return ChunkPlan(
            totalChunks: totalChunks,
            totalFrames: totalFrames,
            durationSeconds: Float(totalFrames) / Float(fps)
        )
    }

    /// Validate a request without loading anything.
    public func validate(_ request: LTX2VideoRequest) throws {
        guard Self.isValidFrameCount(request.framesPerChunk) else {
            throw LTX2VideoError.invalidFrameCount(request.framesPerChunk)
        }
        guard Self.areValidDimensions(width: request.width, height: request.height) else {
            throw LTX2VideoError.invalidDimensions(request.width, request.height)
        }
        let weightsURL = resolveWeightsFileURL()
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LTX2VideoError.weightsMissing(weightsURL.path)
        }
    }

    /// Resolve the checkpoint file to load. Prefers the configured per-component
    /// transformer file; if that's absent, falls back to a JoyAI-Echo monolith
    /// (`*.safetensors` in `weightsDir` that isn't one of the per-component
    /// VAE/connector files). Returns the configured path (possibly nonexistent)
    /// as a last resort so callers surface a clear `weightsMissing` error.
    private func resolveWeightsFileURL() -> URL {
        let configured = URL(fileURLWithPath:
            (config.weightsDir as NSString).appendingPathComponent(config.transformerFile))
        if FileManager.default.fileExists(atPath: configured.path) { return configured }

        let fm = FileManager.default
        if let entries = try? fm.contentsOfDirectory(
            at: URL(fileURLWithPath: config.weightsDir),
            includingPropertiesForKeys: nil
        ) {
            let perComponent: Set<String> = [
                "vae_encoder.safetensors", "vae_decoder.safetensors", "connector.safetensors",
            ]
            let candidates = entries
                .filter { $0.pathExtension == "safetensors" && !perComponent.contains($0.lastPathComponent) }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            if let monolith = candidates.first { return monolith }
        }
        return configured
    }

    // MARK: - Model loading (lazy, cached)

    /// Construct and load the transformer, VAE, text encoder, and pipeline,
    /// optionally merging one or more LoRAs into the transformer (applied in
    /// order). Idempotent for the same LoRA set; a different set reloads.
    public func load(loras: [LTX2LoRAReference] = []) throws {
        let wantKey = loras.isEmpty ? nil : loras.map { "\($0.path)@\($0.scale)" }.joined(separator: "|")
        if isLoaded {
            if wantKey == loadedLoraKey { return }
            unload()   // LoRA set changed — rebuild the transformer.
        }
        let modelDir = config.weightsDir

        logger.info("LTX-2: creating transformer…")
        let transformer = LTX2Transformer(
            numHeads: 32, headDim: 128, inChannels: 128, outChannels: 128,
            numLayers: 48, crossAttentionDim: 4096, captionChannels: 3840,
            normEps: 1e-6, hasPromptAdaLN: true, timestepScaleMultiplier: 1000,
            positionalEmbeddingTheta: 10000, positionalEmbeddingMaxPos: [20, 2048, 2048],
            useMiddleIndicesGrid: true, ropeMode: .split, doublePrecisionRoPE: true
        )

        let weightsURL = resolveWeightsFileURL()
        logger.info("LTX-2: loading transformer weights (\(weightsURL.lastPathComponent))…")
        let rawWeights = try MLX.loadArrays(url: weightsURL)
        // JoyAI-Echo ships one monolithic file (DiT + VAE + audio + vocoder). When
        // detected, the VAE and connector/projection weights come from this same
        // already-loaded dict via prefix-filtered subsets rather than separate
        // files — the audio/vocoder tensors stay lazy (never eval'd) on the
        // video-only path, so they don't hit RAM.
        let isMonolith = LTX2EchoCheckpoint.isMonolithLayout(rawWeights.keys)
        if isMonolith {
            logger.info("LTX-2: JoyAI-Echo monolithic checkpoint detected — prefix-filtered video-only load.")
        }
        var sanitized = LTX2Transformer.sanitizeWeights(rawWeights)

        // Merge each LoRA into the base weights in order (skip audio branches),
        // as the CLI does — multiple LoRAs simply accumulate their deltas.
        for lora in loras {
            logger.info("LTX-2: merging LoRA \(lora.path) @ \(lora.scale)…")
            let loraWeights = try MLX.loadArrays(url: URL(fileURLWithPath: lora.path))
            var merged = 0
            for (key, loraA) in loraWeights {
                guard key.hasSuffix(".lora_A.weight") else { continue }
                var baseKey = String(key.dropLast(".lora_A.weight".count))
                if baseKey.hasPrefix("diffusion_model.") { baseKey = String(baseKey.dropFirst("diffusion_model.".count)) }
                if baseKey.contains("audio_") || baseKey.contains("av_ca_")
                    || baseKey.contains("video_to_audio_attn") || baseKey.contains("audio_to_video_attn") { continue }
                let bKey = key.replacingOccurrences(of: ".lora_A.weight", with: ".lora_B.weight")
                guard let loraB = loraWeights[bKey] else { continue }
                let targetKey = baseKey + ".weight"
                guard sanitized[targetKey] != nil else { continue }
                let delta = MLX.matmul(loraB.asType(.float32), loraA.asType(.float32)) * MLXArray(lora.scale)
                sanitized[targetKey] = sanitized[targetKey]!.asType(.float32) + delta
                merged += 1
            }
            logger.info("LTX-2: merged \(merged) LoRA pairs from \(lora.path).")
        }
        // Anti-noise guard: update(verify:[.shapeMismatch]) silently DROPS any key
        // that doesn't match a module parameter — a mis-remapped checkpoint loads
        // 0 weights and renders pure noise while reporting success. Log how many of
        // the module's parameters the remap actually covers; a near-zero match on a
        // non-empty checkpoint means the key remap is wrong (e.g. leftover
        // `model.diffusion_model.` prefix), not that the load "succeeded".
        let moduleKeys = Set(transformer.parameters().flattened().map { $0.0 })
        let matched = sanitized.keys.filter { moduleKeys.contains($0) }.count
        let unmatchedSanitized = sanitized.count - matched
        logger.info("LTX-2: transformer remap matched \(matched)/\(moduleKeys.count) module params (\(sanitized.count) sanitized keys, \(unmatchedSanitized) unmatched/dropped).")
        if matched * 2 < moduleKeys.count {
            logger.error("LTX-2: transformer weight remap covered only \(matched)/\(moduleKeys.count) params — checkpoint key format likely unrecognized; output would be noise.")
            throw LTX2VideoError.weightsMissing(
                "transformer key remap matched only \(matched)/\(moduleKeys.count) module params from \(weightsURL.lastPathComponent) — unrecognized checkpoint key format")
        }
        let params = ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) })
        try transformer.update(parameters: params, verify: [.shapeMismatch])
        MLX.eval(transformer.parameters())

        logger.info("LTX-2: loading VAE…")
        let vae = LTX2VAE(config: .v23)
        if isMonolith {
            // Echo carries the video VAE under the `vae.` prefix in the monolith,
            // exactly the layout loadVAEWeightsFromTensors expects (top-level
            // per-channel stats are mirrored into the decoder path by the adapter).
            let vaeTensors = LTX2EchoCheckpoint.videoVAETensors(from: rawWeights)
            try LTX2WeightLoader.loadVAEWeightsFromTensors(into: vae, tensors: vaeTensors, logger: logger)
        } else {
            var combinedVAEWeights: [String: MLXArray] = [:]
            let rawDecoderWeights = try MLX.loadArrays(url: URL(fileURLWithPath: (modelDir as NSString).appendingPathComponent("vae_decoder.safetensors")))
            for (key, value) in rawDecoderWeights where key.hasPrefix("vae_decoder.") {
                combinedVAEWeights["vae.decoder." + String(key.dropFirst("vae_decoder.".count))] = value
            }
            let rawEncoderWeights = try MLX.loadArrays(url: URL(fileURLWithPath: (modelDir as NSString).appendingPathComponent("vae_encoder.safetensors")))
            for (key, value) in rawEncoderWeights where key.hasPrefix("vae_encoder.") {
                combinedVAEWeights["vae.encoder." + String(key.dropFirst("vae_encoder.".count))] = value
            }
            if let m = combinedVAEWeights["vae.decoder.per_channel_statistics.mean"] {
                combinedVAEWeights["vae.per_channel_statistics.mean-of-means"] = m
            }
            if let s = combinedVAEWeights["vae.decoder.per_channel_statistics.std"] {
                combinedVAEWeights["vae.per_channel_statistics.std-of-means"] = s
            }
            try LTX2WeightLoader.loadVAEWeightsFromTensors(into: vae, tensors: combinedVAEWeights, logger: logger)
        }
        MLX.eval(vae.parameters())

        logger.info("LTX-2: loading text encoder (Gemma 3 12B)…")
        let gemmaConfig = LTX2GemmaConfig(
            vocabSize: 262208, hiddenSize: 3840,
            numHiddenLayers: 48, numAttentionHeads: 16,
            numKeyValueHeads: 8, headDim: 256,
            intermediateSize: 15360,
            rmsNormEps: 1e-6, ropeTheta: 1_000_000.0,
            slidingWindow: 1024, slidingWindowPattern: 6,
            quantization: nil
        )
        let textEncoder = LTX2TextEncoder(config: LTX2TextEncoderConfig(gemma: gemmaConfig, hasPromptAdaLN: true))
        if isMonolith {
            // Connectors (model.diffusion_model.*_embeddings_connector) and the
            // aggregate embeds (text_embedding_projection.*) live in the monolith;
            // Gemma still loads from its own directory.
            try textEncoder.loadWeightsFromMonolith(
                gemmaPath: URL(fileURLWithPath: config.gemmaPath),
                monolithTensors: rawWeights
            )
        } else {
            try textEncoder.loadWeights(
                modelPath: URL(fileURLWithPath: modelDir),
                textEncoderPath: URL(fileURLWithPath: config.gemmaPath)
            )
        }
        MLX.eval(textEncoder.parameters())

        // Reference ComfyUI-LTXVideo workflows always decode through
        // VAEDecodeTiled, never a plain single-pass decode — the decoder
        // likely relies on windowed/local processing that the (already
        // implemented, just never enabled) tiled path is designed for.
        // Running it as one giant pass is the leading suspect for the
        // uniform grid/mesh artifact seen in every local I2V test tonight.
        let pipelineConfig = LTX2PipelineConfig(modelPath: modelDir, pipelineType: .distilled, hasPromptAdaLN: true, tiledDecode: true)
        self.pipeline = LTX2Pipeline(vae: vae, textEncoder: textEncoder, transformer: transformer, config: pipelineConfig)
        self.tokenizer = try LTX2GemmaTokenizer.load(from: URL(fileURLWithPath: config.gemmaPath), maxLength: 128)
        isLoaded = true
        loadedLoraKey = wantKey
        logger.info("LTX-2: models ready.")
    }

    /// Free the loaded models.
    public func unload() {
        pipeline = nil
        tokenizer = nil
        isLoaded = false
        loadedLoraKey = nil
    }

    // MARK: - Generate

    public func generate(
        _ request: LTX2VideoRequest,
        progress: ((Int, Int, Int, Int) -> Void)? = nil   // (chunk, totalChunks, step, totalSteps)
    ) throws -> LTX2VideoResult {
        #if canImport(CoreGraphics) && canImport(ImageIO)
        try validate(request)
        try load(loras: request.effectiveLoRAs)
        guard let pipeline, let tokenizer else { throw LTX2VideoError.weightsMissing(config.weightsDir) }

        let plan = Self.chunkPlan(
            framesPerChunk: request.framesPerChunk,
            extendToSeconds: request.extendToSeconds, fps: request.fps)

        let batch = tokenizer.encode(prompt: request.prompt, maxLength: 128)
        MLX.eval(batch.inputIds, batch.attentionMask)

        let start = CFAbsoluteTimeGetCurrent()
        var allFrames: [CGImage] = []

        // Seed image: the init image for I2V, else nil (T2V first chunk).
        var currentImage: MLXArray? = try request.initImagePath.map { path in
            let url = URL(fileURLWithPath: path)
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                throw LTX2VideoError.imageLoadFailed(path)
            }
            let pixels = try QwenImageIO.resizedPixelArray(
                from: cgImage, width: request.width, height: request.height,
                addBatchDimension: true, dtype: .float32)
            return QwenImageIO.normalizeForEncoder(pixels)
        }

        for chunk in 0..<plan.totalChunks {
            let chunkSeed = request.seed + UInt64(chunk)
            let output: LTX2PipelineOutput
            if let image = currentImage {
                output = pipeline.generateI2V(
                    inputIds: batch.inputIds, attentionMask: batch.attentionMask,
                    image: image, strength: request.strength,
                    width: request.width, height: request.height,
                    numFrames: request.framesPerChunk, steps: request.steps, seed: chunkSeed,
                    progressCallback: { s, t in progress?(chunk, plan.totalChunks, s, t) })
            } else {
                output = pipeline.generateT2V(
                    inputIds: batch.inputIds, attentionMask: batch.attentionMask,
                    width: request.width, height: request.height,
                    numFrames: request.framesPerChunk, steps: request.steps, seed: chunkSeed,
                    progressCallback: { s, t in progress?(chunk, plan.totalChunks, s, t) })
            }

            let chunkFrames = LTX2PostProcess.framesToImages(from: output.decoded)
            allFrames.append(contentsOf: chunk == 0 ? chunkFrames : Array(chunkFrames.dropFirst()))

            // Re-feed the last frame as the seed for the next continuation chunk.
            if chunk < plan.totalChunks - 1 {
                let t = output.decoded.dim(2)
                let lastFrame = output.decoded[0..., 0..., (t - 1)..<t, 0..., 0...].squeezed(axis: 2)
                currentImage = lastFrame * 2.0 - 1.0
                MLX.eval(currentImage!)
            }
        }

        try LTX2PostProcess.writeMP4(
            frames: allFrames, outputPath: request.outputPath,
            fps: request.fps, width: request.width, height: request.height)

        return LTX2VideoResult(
            outputPath: request.outputPath,
            frameCount: allFrames.count,
            durationSeconds: Float(allFrames.count) / Float(request.fps),
            elapsedSeconds: CFAbsoluteTimeGetCurrent() - start
        )
        #else
        throw LTX2VideoError.unsupportedPlatform
        #endif
    }
}
