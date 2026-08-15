// ComfyBridgeModelRegistry.swift — Unified model registry for ComfyBox
//
// Central registry of all supported model families, their capabilities,
// recommended parameters, and HuggingFace identifiers. Used by:
// - ComfyBridgeModelInfo (serves /api/etn/model_info to Krita)
// - WarmServer (model loading and family detection)
// - StylePresets (model-specific defaults)
//
// Models are grouped by family and variant (distilled vs base, quantization).

import Foundation

// MARK: - Model Family

/// Top-level model family classification.
public enum ComfyBoxModelFamily: String, Codable, CaseIterable, Sendable {
  case zImage = "z-image"
  case flux2Klein = "flux2-klein"
  case fibo = "fibo"
  case seedvr2 = "seedvr2"
  case chroma = "chroma"
  case esrgan = "esrgan"
  case krea2 = "krea2"

  /// Human-readable display name.
  public var displayName: String {
    switch self {
    case .zImage: return "Z-Image"
    case .flux2Klein: return "Flux 2 Klein"
    case .fibo: return "FIBO"
    case .seedvr2: return "SeedVR2"
    case .chroma: return "Chroma"
    case .esrgan: return "ESRGAN"
    case .krea2: return "Krea-2-Turbo"
    }
  }

  /// Architecture description for UI tooltips.
  public var architecture: String {
    switch self {
    case .zImage: return "6B DiT, Qwen3-Coder text encoder, Lumina2 architecture"
    case .flux2Klein: return "4B/9B DiT, Qwen3 text encoder, Flux 2 architecture"
    case .fibo: return "8B DiT, SmolLM3-3B text encoder, Wan 2.2 VAE, DimFusion"
    case .seedvr2: return "3B upscaler, 2× resolution enhancement"
    case .chroma: return "8.9B DiT, T5-XXL text encoder, FLUX.1 VAE, Approximator"
    case .esrgan: return "RRDBNet 4× image upscaler"
    case .krea2: return "SingleStreamDiT, Qwen3-VL-4B text encoder (12-layer tap), Qwen-Image VAE"
    }
  }
}

// MARK: - Model Variant

/// Whether a model is distilled (few-step) or base (multi-step, supports CFG).
public enum ComfyBoxModelVariant: String, Codable, Sendable {
  case turbo       // Distilled, 4-9 steps, no CFG needed
  case base        // Non-distilled, 20-50 steps, CFG guidance > 1.0
  case upscaler    // Post-processing model
}

// MARK: - Quantization

/// Quantization level for model weights.
public enum ComfyBoxQuantization: String, Codable, CaseIterable, Sendable {
  case none = "bf16"    // Full bfloat16
  case q8 = "q8"        // 8-bit quantized
  case q4 = "q4"        // 4-bit quantized

  /// Approximate VRAM reduction factor vs bf16.
  public var vramFactor: Float {
    switch self {
    case .none: return 1.0
    case .q8: return 0.5
    case .q4: return 0.25
    }
  }
}

// MARK: - Model Definition

/// A single registered model with all metadata needed by the bridge and WarmServer.
public struct ComfyBoxModel: Codable, Sendable {
  /// Unique identifier used in API responses and config (e.g. "z-image-turbo-bf16").
  public let id: String

  /// Model family.
  public let family: ComfyBoxModelFamily

  /// Distilled or base variant.
  public let variant: ComfyBoxModelVariant

  /// Weight quantization level.
  public let quantization: ComfyBoxQuantization

  /// HuggingFace model identifier (e.g. "Tongyi-MAI/Z-Image").
  public let huggingFaceId: String

  /// Parameter count in billions (approximate).
  public let parametersBillions: Float

  /// Number of latent channels (48 for FIBO/Wan2.2, 128 for Flux-family).
  public let latentChannels: Int

  /// Default number of denoising steps.
  public let defaultSteps: Int

  /// Default CFG guidance scale (1.0 = no guidance).
  public let defaultGuidance: Float

  /// Whether the model supports CFG guidance > 1.0.
  public let supportsGuidance: Bool

  /// Whether the model supports LoRA adapters.
  public let supportsLoRA: Bool

  /// Whether the model supports ControlNet.
  public let supportsControlNet: Bool

  /// Whether the model supports img2img.
  public let supportsImg2Img: Bool

  /// Default resolution (width × height).
  public let defaultWidth: Int
  public let defaultHeight: Int

  /// Estimated VRAM usage in GB at default resolution.
  public let estimatedVRAM_GB: Float

  /// Human-readable display name.
  public let displayName: String

  /// Short description for UI tooltips.
  public let description: String
}

// MARK: - Model Registry

/// Central registry of all ComfyBox models.
public enum ComfyBoxModelRegistry {

  /// All registered models, keyed by ID.
  public static let models: [String: ComfyBoxModel] = {
    var dict: [String: ComfyBoxModel] = [:]
    for model in allModels {
      dict[model.id] = model
    }
    return dict
  }()

  /// All registered models as an array.
  public static let allModels: [ComfyBoxModel] = [

    // ─── Z-Image Family ──────────────────────────────────────────────
    // 6B DiT with Qwen3-Coder text encoder. Lumina2 architecture.
    // Turbo = distilled (4-9 steps), Base = non-distilled (20-50 steps).

    ComfyBoxModel(
      id: "z-image-turbo-bf16",
      family: .zImage, variant: .turbo, quantization: .none,
      huggingFaceId: "Tongyi-MAI/Z-Image",
      parametersBillions: 6.0, latentChannels: 128,
      defaultSteps: 9, defaultGuidance: 1.0,
      supportsGuidance: false, supportsLoRA: true,
      supportsControlNet: true, supportsImg2Img: true,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 12.5,
      displayName: "Z-Image Turbo",
      description: "Daily driver — fast 9-step distilled model. Best quality-to-speed ratio."
    ),

    ComfyBoxModel(
      id: "z-image-turbo-q8",
      family: .zImage, variant: .turbo, quantization: .q8,
      huggingFaceId: "Tongyi-MAI/Z-Image",
      parametersBillions: 6.0, latentChannels: 128,
      defaultSteps: 9, defaultGuidance: 1.0,
      supportsGuidance: false, supportsLoRA: true,
      supportsControlNet: true, supportsImg2Img: true,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 7.0,
      displayName: "Z-Image Turbo (Q8)",
      description: "8-bit quantized Z-Image Turbo. Good quality, lower VRAM."
    ),

    ComfyBoxModel(
      id: "z-image-turbo-q4",
      family: .zImage, variant: .turbo, quantization: .q4,
      huggingFaceId: "Tongyi-MAI/Z-Image",
      parametersBillions: 6.0, latentChannels: 128,
      defaultSteps: 9, defaultGuidance: 1.0,
      supportsGuidance: false, supportsLoRA: true,
      supportsControlNet: true, supportsImg2Img: true,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 4.0,
      displayName: "Z-Image Turbo (Q4)",
      description: "4-bit quantized Z-Image Turbo. Fastest, lowest VRAM."
    ),

    ComfyBoxModel(
      id: "z-image-base-bf16",
      family: .zImage, variant: .base, quantization: .none,
      huggingFaceId: "Tongyi-MAI/Z-Image",
      parametersBillions: 6.0, latentChannels: 128,
      defaultSteps: 28, defaultGuidance: 3.5,
      supportsGuidance: true, supportsLoRA: true,
      supportsControlNet: true, supportsImg2Img: true,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 12.5,
      displayName: "Z-Image Base",
      description: "Non-distilled Z-Image. Higher quality with CFG, slower. Best for final renders."
    ),

    // ─── Flux 2 Klein Family ─────────────────────────────────────────
    // Compact Flux 2 models with Qwen3 text encoder.
    // Klein = distilled (4-8 steps), Klein Base = non-distilled (20-50 steps).

    ComfyBoxModel(
      id: "flux2-klein-4b-bf16",
      family: .flux2Klein, variant: .turbo, quantization: .none,
      huggingFaceId: "black-forest-labs/FLUX.2-Klein-4B",
      parametersBillions: 4.0, latentChannels: 128,
      defaultSteps: 6, defaultGuidance: 1.0,
      supportsGuidance: false, supportsLoRA: false,
      supportsControlNet: false, supportsImg2Img: true,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 8.5,
      displayName: "Flux 2 Klein 4B",
      description: "Compact distilled model. Fast 6-step generation, good for live painting."
    ),

    ComfyBoxModel(
      id: "flux2-klein-9b-bf16",
      family: .flux2Klein, variant: .turbo, quantization: .none,
      huggingFaceId: "black-forest-labs/FLUX.2-Klein-9B",
      parametersBillions: 9.0, latentChannels: 128,
      defaultSteps: 6, defaultGuidance: 1.0,
      supportsGuidance: false, supportsLoRA: false,
      supportsControlNet: false, supportsImg2Img: true,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 18.0,
      displayName: "Flux 2 Klein 9B",
      description: "Larger distilled model. Better quality than 4B, still fast."
    ),

    ComfyBoxModel(
      id: "flux2-klein-base-4b-bf16",
      family: .flux2Klein, variant: .base, quantization: .none,
      huggingFaceId: "black-forest-labs/FLUX.2-Klein-Base-4B",
      parametersBillions: 4.0, latentChannels: 128,
      defaultSteps: 28, defaultGuidance: 3.5,
      supportsGuidance: true, supportsLoRA: false,
      supportsControlNet: false, supportsImg2Img: true,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 8.5,
      displayName: "Flux 2 Klein Base 4B",
      description: "Non-distilled 4B model. Supports CFG guidance for fine control."
    ),

    ComfyBoxModel(
      id: "flux2-klein-base-9b-bf16",
      family: .flux2Klein, variant: .base, quantization: .none,
      huggingFaceId: "black-forest-labs/FLUX.2-Klein-Base-9B",
      parametersBillions: 9.0, latentChannels: 128,
      defaultSteps: 28, defaultGuidance: 3.5,
      supportsGuidance: true, supportsLoRA: false,
      supportsControlNet: false, supportsImg2Img: true,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 18.0,
      displayName: "Flux 2 Klein Base 9B",
      description: "Largest non-distilled Klein. Highest quality Flux 2 output."
    ),

    // ─── FIBO Family ─────────────────────────────────────────────────
    // 8B DiT with SmolLM3-3B text encoder, Wan 2.2 VAE, DimFusion conditioning.

    ComfyBoxModel(
      id: "fibo-8b-bf16",
      family: .fibo, variant: .base, quantization: .none,
      huggingFaceId: "briaai/FIBO",
      parametersBillions: 8.0, latentChannels: 48,
      defaultSteps: 30, defaultGuidance: 4.0,
      supportsGuidance: true, supportsLoRA: false,
      supportsControlNet: false, supportsImg2Img: false,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 22.0,
      displayName: "FIBO 8B",
      description: "Bria AI's 8B model. Unique DimFusion architecture, Wan 2.2 VAE. Commercial-licensed."
    ),

    // ─── Moody Family (CivitAI Z-Image Checkpoints) ────────────────
    // Community Z-Image checkpoints. Same architecture (6B DiT, Qwen3-Coder).
    // Moody Wild Mix V4: enhanced NSFW, better anatomy. Undistilled = 40 steps, CFG 4.0.

    ComfyBoxModel(
      id: "moody-wild-v4",
      family: .zImage, variant: .base, quantization: .none,
      huggingFaceId: "",
      parametersBillions: 6.0, latentChannels: 128,
      defaultSteps: 40, defaultGuidance: 4.0,
      supportsGuidance: true, supportsLoRA: true,
      supportsControlNet: true, supportsImg2Img: true,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 12.5,
      displayName: "Moody Wild Mix V4",
      description: "CivitAI Z-Image checkpoint. Undistilled, 40 steps, CFG 4.0, dpmpp_2m_sde. Enhanced NSFW and anatomy."
    ),

    ComfyBoxModel(
      id: "moody-wild-v4-distilled",
      family: .zImage, variant: .turbo, quantization: .none,
      huggingFaceId: "",
      parametersBillions: 6.0, latentChannels: 128,
      defaultSteps: 10, defaultGuidance: 1.0,
      supportsGuidance: false, supportsLoRA: true,
      supportsControlNet: true, supportsImg2Img: true,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 12.5,
      displayName: "Moody Wild Mix V4 (Distilled)",
      description: "Distilled variant. 10 steps, CFG 1.0. Faster but less control than undistilled."
    ),

    ComfyBoxModel(
      id: "moody-wild-v4-fp8",
      family: .zImage, variant: .base, quantization: .q8,
      huggingFaceId: "",
      parametersBillions: 6.0, latentChannels: 128,
      defaultSteps: 40, defaultGuidance: 4.0,
      supportsGuidance: true, supportsLoRA: true,
      supportsControlNet: true, supportsImg2Img: true,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 7.0,
      displayName: "Moody Wild Mix V4 (FP8)",
      description: "FP8 quantized undistilled. Lower VRAM, same settings as full."
    ),

    ComfyBoxModel(
      id: "moody-real-v6",
      family: .zImage, variant: .base, quantization: .none,
      huggingFaceId: "",
      parametersBillions: 6.0, latentChannels: 128,
      defaultSteps: 40, defaultGuidance: 4.0,
      supportsGuidance: true, supportsLoRA: true,
      supportsControlNet: true, supportsImg2Img: true,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 12.5,
      displayName: "Moody Real V6",
      description: "CivitAI Z-Image checkpoint. Realistic style, undistilled. 40 steps, CFG 4.0."
    ),

    ComfyBoxModel(
      id: "cyberrealistic-v5",
      family: .zImage, variant: .base, quantization: .none,
      huggingFaceId: "",
      parametersBillions: 6.0, latentChannels: 128,
      defaultSteps: 8, defaultGuidance: 1.0,
      supportsGuidance: false, supportsLoRA: true,
      supportsControlNet: true, supportsImg2Img: true,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 12.5,
      displayName: "CyberRealistic V5",
      description: "CivitAI Z-Image Turbo fine-tune. Enhanced photorealism and NSFW. Apache 2.0 license."
    ),


    // ─── SeedVR2 Family ──────────────────────────────────────────────
    // 3B upscaler — 2× resolution enhancement.

    ComfyBoxModel(
      id: "seedvr2-bf16",
      family: .seedvr2, variant: .upscaler, quantization: .none,
      huggingFaceId: "ByteDance/SeedVR2-3B",
      parametersBillions: 3.0, latentChannels: 128,
      defaultSteps: 18, defaultGuidance: 1.0,
      supportsGuidance: false, supportsLoRA: false,
      supportsControlNet: false, supportsImg2Img: true,
      defaultWidth: 2048, defaultHeight: 2048,
      estimatedVRAM_GB: 14.0,
      displayName: "SeedVR2 Upscaler",
      description: "2× resolution upscaler. Feed a generated image, get 2× detail."
    ),

    // ─── Chroma Family ───────────────────────────────────────────────
    // 8.9B DiT with T5-XXL text encoder, FLUX.1 VAE, Approximator block.

    ComfyBoxModel(
      id: "chroma-8.9b-bf16",
      family: .chroma, variant: .base, quantization: .none,
      huggingFaceId: "lodestone-horizon/chroma",
      parametersBillions: 8.9, latentChannels: 128,
      defaultSteps: 28, defaultGuidance: 0.0,
      supportsGuidance: false, supportsLoRA: false,
      supportsControlNet: false, supportsImg2Img: false,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 17.0,
      displayName: "Chroma 8.9B",
      description: "Lodestone Horizon's Chroma model. T5-XXL encoder, CFG via Approximator. No guidance needed."
    ),

    // ─── Krea-2 Family ───────────────────────────────────────────────
    // Native Swift port. SingleStreamDiT, Qwen3-VL-4B text encoder (12-layer
    // tap), Qwen-Image VAE. Guidance-distilled turbo — no CFG.

    ComfyBoxModel(
      id: "krea2-turbo-q8",
      family: .krea2, variant: .turbo, quantization: .q8,
      huggingFaceId: "krea/Krea-2-Turbo",
      parametersBillions: 13.5, latentChannels: 16,
      defaultSteps: 9, defaultGuidance: 0.0,
      supportsGuidance: false, supportsLoRA: true,
      supportsControlNet: false, supportsImg2Img: false,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 22.0,
      displayName: "Krea-2-Turbo",
      description: "Krea's distilled turbo model. Native Swift port — 8-step, guidance-free."
    ),

    ComfyBoxModel(
      id: "kroma-v0.2-turbo",
      family: .krea2, variant: .turbo, quantization: .q8,
      huggingFaceId: "lodestones/Kroma",
      parametersBillions: 13.5, latentChannels: 16,
      defaultSteps: 10, defaultGuidance: 0.0,
      supportsGuidance: false, supportsLoRA: true,
      supportsControlNet: false, supportsImg2Img: false,
      defaultWidth: 1024, defaultHeight: 1024,
      estimatedVRAM_GB: 22.0,
      displayName: "Kroma v0.2",
      description: "lodestones' Kroma — a Krea-2 fine-tune with the Turbo delta baked in (rank-512). Distilled for 8-12 steps. MIT. Successor to the kroma-v0.1 LoRA-delta packaging."
    ),

    // ─── ESRGAN Family ───────────────────────────────────────────────
    // RRDBNet-based 4× upscalers. Various community weights.

    ComfyBoxModel(
      id: "esrgan-4x-ultrasharp",
      family: .esrgan, variant: .upscaler, quantization: .none,
      huggingFaceId: "",
      parametersBillions: 0.017, latentChannels: 3,
      defaultSteps: 1, defaultGuidance: 0.0,
      supportsGuidance: false, supportsLoRA: false,
      supportsControlNet: false, supportsImg2Img: true,
      defaultWidth: 4096, defaultHeight: 4096,
      estimatedVRAM_GB: 1.0,
      displayName: "4x-UltraSharp",
      description: "4× ESRGAN upscaler. Sharp detail enhancement, popular community model."
    ),

    ComfyBoxModel(
      id: "esrgan-realesrgan-x4",
      family: .esrgan, variant: .upscaler, quantization: .none,
      huggingFaceId: "",
      parametersBillions: 0.017, latentChannels: 3,
      defaultSteps: 1, defaultGuidance: 0.0,
      supportsGuidance: false, supportsLoRA: false,
      supportsControlNet: false, supportsImg2Img: true,
      defaultWidth: 4096, defaultHeight: 4096,
      estimatedVRAM_GB: 1.0,
      displayName: "RealESRGAN x4",
      description: "4× Real-ESRGAN upscaler. General-purpose, good for photos and illustrations."
    ),

    ComfyBoxModel(
      id: "esrgan-4xnomos8k",
      family: .esrgan, variant: .upscaler, quantization: .none,
      huggingFaceId: "",
      parametersBillions: 0.017, latentChannels: 3,
      defaultSteps: 1, defaultGuidance: 0.0,
      supportsGuidance: false, supportsLoRA: false,
      supportsControlNet: false, supportsImg2Img: true,
      defaultWidth: 4096, defaultHeight: 4096,
      estimatedVRAM_GB: 1.0,
      displayName: "4xNomos8k",
      description: "4× ESRGAN upscaler trained on 8K images. Best for photorealistic content."
    ),
  ]

  // MARK: - Lookups

  /// Get all models for a given family.
  public static func models(for family: ComfyBoxModelFamily) -> [ComfyBoxModel] {
    allModels.filter { $0.family == family }
  }

  /// Get the default model for a family (prefers turbo/distilled bf16).
  public static func defaultModel(for family: ComfyBoxModelFamily) -> ComfyBoxModel? {
    let familyModels = models(for: family)
    // Prefer turbo/distilled bf16
    return familyModels.first { $0.variant == .turbo && $0.quantization == .none }
      ?? familyModels.first { $0.quantization == .none }
      ?? familyModels.first
  }

  /// Get the recommended model for live painting (fastest img2img).
  public static var livePaintingModel: ComfyBoxModel? {
    models["flux2-klein-4b-bf16"]
  }

  /// Get the recommended model for final renders (highest quality).
  public static var highQualityModel: ComfyBoxModel? {
    models["z-image-base-bf16"]
  }

  // MARK: - Bridge Format

  /// base_model string for the Krita plugin's Arch.from_string.
  /// "flux2-klein" is not a recognized Arch — Klein maps to "flux2" with the
  /// size variant carried in "type".
  private static func kritaBaseModel(for model: ComfyBoxModel) -> String {
    switch model.family {
    case .flux2Klein: return "flux2"
    default: return model.family.rawValue
    }
  }

  /// type string for the Krita plugin. For Flux 2 Klein the type selects the
  /// 4B/9B variant; for everything else it is the prediction type.
  private static func kritaType(for model: ComfyBoxModel) -> String {
    switch model.family {
    case .flux2Klein:
      return model.parametersBillions >= 9.0 ? "klein-9b" : "klein-4b"
    default:
      return model.supportsGuidance ? "v_prediction" : "eps"
    }
  }

  /// Convert to the format expected by /api/etn/model_info/{folder}.
  /// Returns [modelId: metadata] for the Krita AI Diffusion plugin.
  public static func bridgeModelInfo(for folder: String) -> [String: Any] {
    switch folder {
    case "diffusion_models", "unet":
      var result: [String: Any] = [:]
      for model in allModels where model.variant != .upscaler {
        // FIBO, Chroma, and Krea-2 workloads are not usable from the Krita
        // plugin (unrecognized or unsupported Arch) — omit them until
        // supported so they neither vanish silently nor surface as broken
        // entries.
        if model.family == .fibo || model.family == .chroma || model.family == .krea2 { continue }
        result[model.id] = [
          "base_model": kritaBaseModel(for: model),
          "type": kritaType(for: model),
          "is_inpaint": false,
          "quant": model.quantization.rawValue,
          "params_b": model.parametersBillions,
          "default_steps": model.defaultSteps,
          "default_guidance": model.defaultGuidance,
          "supports_guidance": model.supportsGuidance,
          "supports_lora": model.supportsLoRA,
          "supports_img2img": model.supportsImg2Img,
          "display_name": model.displayName,
          "description": model.description,
        ] as [String: Any]
      }
      return result

    case "upscale_models":
      var result: [String: Any] = [:]
      for model in allModels where model.variant == .upscaler {
        let scaleFactor: Int = model.family == .esrgan ? 4 : 2
        result[model.id] = [
          "name": model.id,
          "base_model": model.family.rawValue,
          "type": "upscaler",
          "architecture": model.family.architecture,
          "scale_factor": scaleFactor,
          "params_b": model.parametersBillions,
          "estimated_vram_gb": model.estimatedVRAM_GB,
          "display_name": model.displayName,
          "description": model.description,
        ] as [String: Any]
      }
      return result

    case "checkpoints":
      // We don't use checkpoint format.
      return [:]

    case "unet_gguf":
      // No GGUF models in ComfyBox.
      return [:]

    default:
      return [:]
    }
  }
}
