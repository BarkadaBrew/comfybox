import Foundation

/// JSON sidecar schema — written alongside every generated image when --metadata is active.
/// Version 1. Schema evolution via version field + migration in MetadataReader.
public struct GenerationMetadata: Codable, Sendable {
    public let version: Int                    // always 1 for now
    public let generator: String               // "ComfyBox"
    public let generatorVersion: String        // build version string
    public let timestamp: Date

    public let pipeline: PipelineType
    public let model: ModelInfo
    public let parameters: GenerationParameters
    public let img2img: Img2ImgInfo?           // nil for txt2img
    public let loras: [LoRAInfo]
    public let output: OutputInfo

    public enum PipelineType: String, Codable, Sendable {
        case txt2img, img2img, controlnet, inpaint, upscale
    }

    public init(
        version: Int = 1,
        generator: String = "ComfyBox",
        generatorVersion: String = "0.1.0",
        timestamp: Date = Date(),
        pipeline: PipelineType,
        model: ModelInfo,
        parameters: GenerationParameters,
        img2img: Img2ImgInfo? = nil,
        loras: [LoRAInfo] = [],
        output: OutputInfo
    ) {
        self.version = version
        self.generator = generator
        self.generatorVersion = generatorVersion
        self.timestamp = timestamp
        self.pipeline = pipeline
        self.model = model
        self.parameters = parameters
        self.img2img = img2img
        self.loras = loras
        self.output = output
    }
}

public struct ModelInfo: Codable, Sendable {
    public let family: String                  // "flux", "fibo", etc.
    public let variant: String                 // "schnell", "dev", "coffeeshop-8bit"
    public let quantization: Int?              // nil = full precision
    public let path: String                    // model directory path

    public init(family: String, variant: String, quantization: Int? = nil, path: String) {
        self.family = family
        self.variant = variant
        self.quantization = quantization
        self.path = path
    }
}

public struct GenerationParameters: Codable, Sendable {
    public let prompt: String
    public let negativePrompt: String?
    public let seed: UInt64
    public let steps: Int
    public let guidance: Float
    public let width: Int
    public let height: Int
    public let scheduler: String
    public let sigmaSchedule: String?

    public init(
        prompt: String,
        negativePrompt: String? = nil,
        seed: UInt64,
        steps: Int,
        guidance: Float,
        width: Int,
        height: Int,
        scheduler: String,
        sigmaSchedule: String? = nil
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.seed = seed
        self.steps = steps
        self.guidance = guidance
        self.width = width
        self.height = height
        self.scheduler = scheduler
        self.sigmaSchedule = sigmaSchedule
    }
}

public struct Img2ImgInfo: Codable, Sendable {
    public let sourceImage: String             // path to input image
    public let strength: Float                 // canonical mflux convention
    public let specifiedAs: String             // "strength" or "creativity"
    public let specifiedValue: Float           // what the user typed

    public init(sourceImage: String, strength: Float, specifiedAs: String, specifiedValue: Float) {
        self.sourceImage = sourceImage
        self.strength = strength
        self.specifiedAs = specifiedAs
        self.specifiedValue = specifiedValue
    }
}

public struct LoRAInfo: Codable, Sendable {
    public let path: String
    public let scale: Float

    public init(path: String, scale: Float) {
        self.path = path
        self.scale = scale
    }
}

public struct OutputInfo: Codable, Sendable {
    public let path: String
    public let width: Int
    public let height: Int
    public let renderTimeSeconds: Double

    public init(path: String, width: Int, height: Int, renderTimeSeconds: Double) {
        self.path = path
        self.width = width
        self.height = height
        self.renderTimeSeconds = renderTimeSeconds
    }
}
