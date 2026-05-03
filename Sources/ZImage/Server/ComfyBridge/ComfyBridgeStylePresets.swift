// ComfyBridgeStylePresets.swift — Style presets for ComfyBox
//
// Style presets provide prompt engineering shortcuts + tuned generation
// parameters for common artistic styles. Used by the Krita bridge and
// can be exposed via WarmServer API endpoints.
//
// Each preset specifies:
// - Prompt prefix/suffix for the style
// - Recommended model family + variant
// - Optimal steps, guidance, resolution
// - Negative prompt tailored to the style

import Foundation

// MARK: - Style Category

/// Broad category for organizing presets in UI.
public enum ComfyBoxStyleCategory: String, Codable, CaseIterable, Sendable {
  case photography = "Photography"
  case illustration = "Illustration"
  case painting = "Painting"
  case concept = "Concept Art"
  case utility = "Utility"
}

// MARK: - Style Preset

/// A named style preset with prompt engineering and generation parameters.
public struct ComfyBoxStylePreset: Codable, Sendable {
  /// Unique identifier (e.g. "cinematic").
  public let id: String

  /// Display name (e.g. "Cinematic Film").
  public let name: String

  /// Category for UI grouping.
  public let category: ComfyBoxStyleCategory

  /// Short description.
  public let description: String

  /// Text prepended to the user's prompt.
  public let promptPrefix: String

  /// Text appended to the user's prompt.
  public let promptSuffix: String

  /// Style-specific negative prompt (appended to user's negative prompt).
  public let negativePrompt: String

  /// Recommended model ID (from ComfyBoxModelRegistry).
  public let recommendedModel: String

  /// Recommended denoising steps (overrides model default if set).
  public let steps: Int?

  /// Recommended CFG guidance (overrides model default if set).
  public let guidance: Float?

  /// Recommended resolution.
  public let width: Int?
  public let height: Int?

  /// Denoise strength for img2img (0.0-1.0). Nil = use default.
  public let img2imgStrength: Float?

  /// Apply the preset to a prompt — returns (enhancedPrompt, enhancedNegative).
  public func apply(prompt: String, negativePrompt: String? = nil) -> (String, String?) {
    let enhanced = [promptPrefix, prompt, promptSuffix]
      .filter { !$0.isEmpty }
      .joined(separator: ", ")

    let combinedNegative: String?
    if !self.negativePrompt.isEmpty {
      if let userNeg = negativePrompt, !userNeg.isEmpty {
        combinedNegative = userNeg + ", " + self.negativePrompt
      } else {
        combinedNegative = self.negativePrompt
      }
    } else {
      combinedNegative = negativePrompt
    }

    return (enhanced, combinedNegative)
  }
}

// MARK: - Preset Registry

/// All built-in style presets.
public enum ComfyBoxStylePresets {

  /// All presets, keyed by ID.
  public static let presets: [String: ComfyBoxStylePreset] = {
    var dict: [String: ComfyBoxStylePreset] = [:]
    for preset in allPresets {
      dict[preset.id] = preset
    }
    return dict
  }()

  /// All presets as an array.
  public static let allPresets: [ComfyBoxStylePreset] = [

    // ─── Photography ─────────────────────────────────────────────────

    ComfyBoxStylePreset(
      id: "photorealistic",
      name: "Photorealistic",
      category: .photography,
      description: "Clean photorealistic output with natural lighting",
      promptPrefix: "photorealistic, highly detailed photograph",
      promptSuffix: "natural lighting, sharp focus, 8k resolution",
      negativePrompt: "illustration, painting, cartoon, anime, drawing, sketch, cgi, 3d render, text, watermark",
      recommendedModel: "z-image-turbo-bf16",
      steps: nil, guidance: nil,
      width: 1024, height: 1024,
      img2imgStrength: nil
    ),

    ComfyBoxStylePreset(
      id: "portrait",
      name: "Portrait",
      category: .photography,
      description: "Professional portrait photography with shallow depth of field",
      promptPrefix: "professional portrait photograph, shallow depth of field, bokeh background",
      promptSuffix: "studio lighting, sharp eyes, skin detail, 85mm lens",
      negativePrompt: "illustration, cartoon, blurry face, distorted features, extra fingers, text, watermark",
      recommendedModel: "z-image-turbo-bf16",
      steps: nil, guidance: nil,
      width: 832, height: 1216,
      img2imgStrength: nil
    ),

    ComfyBoxStylePreset(
      id: "cinematic",
      name: "Cinematic Film",
      category: .photography,
      description: "Film-like images with dramatic lighting and color grading",
      promptPrefix: "cinematic film still, dramatic lighting, anamorphic lens",
      promptSuffix: "color grading, volumetric lighting, film grain, depth of field, 35mm film",
      negativePrompt: "flat lighting, overexposed, anime, cartoon, illustration, text, watermark",
      recommendedModel: "z-image-base-bf16",
      steps: 28, guidance: 4.0,
      width: 1216, height: 832,
      img2imgStrength: nil
    ),

    ComfyBoxStylePreset(
      id: "street",
      name: "Street Photography",
      category: .photography,
      description: "Candid street photography with urban atmosphere",
      promptPrefix: "street photography, candid moment, urban",
      promptSuffix: "natural light, documentary style, Leica lens, subtle grain",
      negativePrompt: "studio, posed, illustration, anime, text, watermark",
      recommendedModel: "z-image-turbo-bf16",
      steps: nil, guidance: nil,
      width: 1024, height: 1024,
      img2imgStrength: nil
    ),

    ComfyBoxStylePreset(
      id: "product",
      name: "Product Shot",
      category: .photography,
      description: "Clean product photography on seamless background",
      promptPrefix: "product photography, studio lighting, seamless white background",
      promptSuffix: "soft shadows, commercial quality, centered composition, high detail",
      negativePrompt: "cluttered, messy background, low quality, illustration, text",
      recommendedModel: "z-image-turbo-bf16",
      steps: nil, guidance: nil,
      width: 1024, height: 1024,
      img2imgStrength: nil
    ),

    ComfyBoxStylePreset(
      id: "vintage",
      name: "Vintage Film",
      category: .photography,
      description: "Retro film aesthetic with warm tones and grain",
      promptPrefix: "vintage photograph, retro film aesthetic",
      promptSuffix: "warm color cast, film grain, faded highlights, vignette, Kodak Portra 400",
      negativePrompt: "digital, clean, modern, illustration, anime, text, watermark",
      recommendedModel: "z-image-turbo-bf16",
      steps: nil, guidance: nil,
      width: 1024, height: 1024,
      img2imgStrength: nil
    ),

    // ─── Illustration ────────────────────────────────────────────────

    ComfyBoxStylePreset(
      id: "anime",
      name: "Anime / Manga",
      category: .illustration,
      description: "Japanese anime/manga illustration style",
      promptPrefix: "anime style illustration, vibrant colors",
      promptSuffix: "clean linework, cel shading, high quality anime art",
      negativePrompt: "photorealistic, photograph, 3d render, blurry, low quality, text",
      recommendedModel: "z-image-turbo-bf16",
      steps: nil, guidance: nil,
      width: 832, height: 1216,
      img2imgStrength: nil
    ),

    ComfyBoxStylePreset(
      id: "digital-art",
      name: "Digital Art",
      category: .illustration,
      description: "Polished digital illustration with rich detail",
      promptPrefix: "digital art, detailed illustration",
      promptSuffix: "professional digital painting, artstation quality, vivid colors, clean composition",
      negativePrompt: "photograph, blurry, sketch, rough, text, watermark",
      recommendedModel: "z-image-turbo-bf16",
      steps: nil, guidance: nil,
      width: 1024, height: 1024,
      img2imgStrength: nil
    ),

    ComfyBoxStylePreset(
      id: "sketch",
      name: "Pencil Sketch",
      category: .illustration,
      description: "Hand-drawn pencil sketch or charcoal drawing",
      promptPrefix: "pencil sketch, hand drawn, graphite on paper",
      promptSuffix: "detailed linework, cross-hatching, textured paper, monochrome",
      negativePrompt: "color, photograph, digital, 3d, painting, text, watermark",
      recommendedModel: "flux2-klein-4b-bf16",
      steps: 6, guidance: nil,
      width: 1024, height: 1024,
      img2imgStrength: nil
    ),

    // ─── Painting ────────────────────────────────────────────────────

    ComfyBoxStylePreset(
      id: "oil-painting",
      name: "Oil Painting",
      category: .painting,
      description: "Classical oil painting with visible brushstrokes",
      promptPrefix: "oil painting, classical fine art",
      promptSuffix: "visible brushstrokes, canvas texture, rich impasto, museum quality, chiaroscuro",
      negativePrompt: "photograph, digital, 3d render, anime, cartoon, text, watermark",
      recommendedModel: "z-image-base-bf16",
      steps: 28, guidance: 4.0,
      width: 1024, height: 1024,
      img2imgStrength: nil
    ),

    ComfyBoxStylePreset(
      id: "watercolor",
      name: "Watercolor",
      category: .painting,
      description: "Watercolor painting with soft washes and bleeding edges",
      promptPrefix: "watercolor painting, soft washes",
      promptSuffix: "wet on wet technique, paper texture, translucent layers, bleeding edges, gentle gradients",
      negativePrompt: "photograph, digital, sharp edges, hard lines, text, watermark",
      recommendedModel: "z-image-turbo-bf16",
      steps: nil, guidance: nil,
      width: 1024, height: 1024,
      img2imgStrength: nil
    ),

    // ─── Concept Art ─────────────────────────────────────────────────

    ComfyBoxStylePreset(
      id: "fantasy",
      name: "Fantasy Art",
      category: .concept,
      description: "Epic fantasy concept art with dramatic atmosphere",
      promptPrefix: "epic fantasy art, concept illustration",
      promptSuffix: "dramatic atmosphere, magical lighting, detailed environment, artstation trending",
      negativePrompt: "photograph, mundane, modern, blurry, text, watermark",
      recommendedModel: "z-image-base-bf16",
      steps: 28, guidance: 3.5,
      width: 1216, height: 832,
      img2imgStrength: nil
    ),

    ComfyBoxStylePreset(
      id: "architecture",
      name: "Architectural Render",
      category: .concept,
      description: "Architectural visualization with clean lines",
      promptPrefix: "architectural visualization, photorealistic render",
      promptSuffix: "clean lines, natural materials, golden hour lighting, exterior photography, professional archviz",
      negativePrompt: "cartoon, sketch, blurry, distorted, text, watermark",
      recommendedModel: "z-image-base-bf16",
      steps: 28, guidance: 4.0,
      width: 1216, height: 832,
      img2imgStrength: nil
    ),

    // ─── Utility ─────────────────────────────────────────────────────

    ComfyBoxStylePreset(
      id: "live-paint",
      name: "Live Painting",
      category: .utility,
      description: "Fastest possible img2img — optimized for Krita live canvas painting",
      promptPrefix: "",
      promptSuffix: "",
      negativePrompt: "blurry, low quality",
      recommendedModel: "flux2-klein-4b-bf16",
      steps: 4, guidance: 1.0,
      width: 512, height: 512,
      img2imgStrength: 0.5
    ),

    ComfyBoxStylePreset(
      id: "minimal",
      name: "Minimal (No Style)",
      category: .utility,
      description: "No prompt engineering — pass your prompt directly to the model",
      promptPrefix: "",
      promptSuffix: "",
      negativePrompt: "",
      recommendedModel: "z-image-turbo-bf16",
      steps: nil, guidance: nil,
      width: 1024, height: 1024,
      img2imgStrength: nil
    ),
  ]

  // MARK: - Lookups

  /// Get all presets in a category.
  public static func presets(for category: ComfyBoxStyleCategory) -> [ComfyBoxStylePreset] {
    allPresets.filter { $0.category == category }
  }

  /// Get the preset recommended for live painting.
  public static var livePainting: ComfyBoxStylePreset? {
    presets["live-paint"]
  }

  // MARK: - JSON Export

  /// Export all presets as a JSON-serializable dictionary.
  /// Used by WarmServer's /v1/styles endpoint.
  public static func toJSON() -> [[String: Any]] {
    allPresets.map { preset in
      var dict: [String: Any] = [
        "id": preset.id,
        "name": preset.name,
        "category": preset.category.rawValue,
        "description": preset.description,
        "prompt_prefix": preset.promptPrefix,
        "prompt_suffix": preset.promptSuffix,
        "negative_prompt": preset.negativePrompt,
        "recommended_model": preset.recommendedModel,
      ]
      if let steps = preset.steps { dict["steps"] = steps }
      if let guidance = preset.guidance { dict["guidance"] = guidance }
      if let w = preset.width { dict["width"] = w }
      if let h = preset.height { dict["height"] = h }
      if let strength = preset.img2imgStrength { dict["img2img_strength"] = strength }
      return dict
    }
  }
}
