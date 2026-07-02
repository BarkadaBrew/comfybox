// ComfyBridgeObjectInfo.swift — Static /object_info response for ComfyUI bridge
//
// Declares all node types that the Krita AI Diffusion plugin checks for during
// connection validation. Each node has the minimum input/output schema the plugin
// expects. Model discovery nodes return option lists with our available models.

import Foundation

/// Builds the complete `/object_info` response dictionary.
///
/// The Krita plugin calls `_check_for_missing_nodes` against `required_custom_nodes`
/// in resources.py. Every node listed here must appear in the response or the plugin
/// will refuse to connect.
enum ComfyBridgeObjectInfo {

  /// Build the full object_info dictionary. All keys are node class_type names.
  static func build(loraLibrary: LoRALibrary? = nil) -> [String: Any] {
    var info: [String: Any] = [:]

    // --- comfyui-tooling-nodes (required) ---
    info["ETN_LoadImageCache"] = nodeDefinition(
      required: ["id": stringInput()],
      outputs: ["IMAGE", "MASK"]
    )
    info["ETN_SaveImageCache"] = nodeDefinition(
      required: [
        "images": imageInput(),
        "format": optionInput(["PNG", "JPEG", "WEBP"]),
      ],
      optional: ["id": stringInput()],
      outputs: []
    )
    info["ETN_Translate"] = nodeDefinition(
      required: [
        "text": stringInput(),
        "source_language": stringInput(),
        "target_language": stringInput(),
      ],
      outputs: ["STRING"]
    )

    // --- comfyui-inpaint-nodes (required) ---
    info["INPAINT_LoadFooocusInpaint"] = nodeDefinition(
      required: [
        "head": stringInput(),
        "patch": stringInput(),
      ],
      outputs: ["INPAINT_PATCH"]
    )
    info["INPAINT_ShrinkMask"] = nodeDefinition(
      required: [
        "mask": maskInput(),
        "shrink": intInput(default: 4),
      ],
      outputs: ["MASK"]
    )
    info["INPAINT_StabilizeMask"] = nodeDefinition(
      required: [
        "mask": maskInput(),
        "threshold": floatInput(default: 0.5),
      ],
      outputs: ["MASK"]
    )
    info["INPAINT_ColorMatch"] = nodeDefinition(
      required: [
        "image": imageInput(),
        "reference": imageInput(),
      ],
      outputs: ["IMAGE"]
    )
    info["INPAINT_ExpandMask"] = nodeDefinition(
      required: [
        "mask": maskInput(),
        "grow": intInput(default: 16),
        "feather": intInput(default: 8),
      ],
      outputs: ["MASK"]
    )
    info["INPAINT_MaskedBlur"] = nodeDefinition(
      required: [
        "image": imageInput(),
        "mask": maskInput(),
        "blur_radius": intInput(default: 32),
      ],
      outputs: ["IMAGE"]
    )
    info["INPAINT_VAEEncodeInpaintConditioning"] = nodeDefinition(
      required: [
        "vae": vaeInput(),
        "image": imageInput(),
        "mask": maskInput(),
      ],
      outputs: ["LATENT", "LATENT"]
    )

    // --- comfyui_controlnet_aux (required) ---
    info["InpaintPreprocessor"] = nodeDefinition(
      required: [
        "image": imageInput(),
        "mask": maskInput(),
      ],
      outputs: ["IMAGE"]
    )
    info["DepthAnythingV2Preprocessor"] = nodeDefinition(
      required: [
        "image": imageInput(),
      ],
      optional: [
        "resolution": intInput(default: 512),
      ],
      outputs: ["IMAGE"]
    )

    // --- ComfyUI_IPAdapter_plus (required) ---
    info["IPAdapterModelLoader"] = nodeDefinition(
      required: [
        "ipadapter_file": optionInput(["ip-adapter_sd15.safetensors", "ip-adapter-plus_sdxl_vit-h.safetensors"]),
      ],
      outputs: ["IPADAPTER"]
    )
    info["IPAdapter"] = nodeDefinition(
      required: [
        "model": modelInput(),
        "ipadapter": ipadapterInput(),
        "image": imageInput(),
      ],
      optional: [
        "weight": floatInput(default: 1.0),
      ],
      outputs: ["MODEL"]
    )

    // --- Core ComfyUI nodes (model loading) ---
    info["UNETLoader"] = nodeDefinition(
      required: [
        "unet_name": optionInput(zimageUnetModels()),
        "weight_dtype": optionInput(["default", "fp8_e4m3fn", "fp8_e5m2"]),
      ],
      requiredOrder: ["unet_name", "weight_dtype"],
      outputs: ["MODEL"]
    )
    info["CLIPLoader"] = nodeDefinition(
      required: [
        "clip_name": optionInput(zimageClipModels()),
        "type": optionInput(["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "lumina2"]),
      ],
      requiredOrder: ["clip_name", "type"],
      outputs: ["CLIP"]
    )
    info["DualCLIPLoader"] = nodeDefinition(
      required: [
        "clip_name1": optionInput(zimageClipModels()),
        "clip_name2": optionInput(zimageClipModels()),
        "type": optionInput(["sdxl", "sd3", "flux", "hunyuan_video"]),
      ],
      requiredOrder: ["clip_name1", "clip_name2", "type"],
      outputs: ["CLIP"]
    )
    info["DualCLIPLoaderGGUF"] = nodeDefinition(
      required: [
        "clip_name1": optionInput(zimageClipModels()),
        "clip_name2": optionInput(zimageClipModels()),
        "type": optionInput(["sdxl", "sd3", "flux", "hunyuan_video"]),
      ],
      requiredOrder: ["clip_name1", "clip_name2", "type"],
      outputs: ["CLIP"]
    )
    info["VAELoader"] = nodeDefinition(
      required: [
        "vae_name": optionInput(zimageVaeModels()),
      ],
      outputs: ["VAE"]
    )
    info["ControlNetLoader"] = nodeDefinition(
      required: [
        "control_net_name": optionInput(zimageControlnetModels()),
      ],
      outputs: ["CONTROL_NET"]
    )
    info["UpscaleModelLoader"] = nodeDefinition(
      required: [
        "model_name": optionInput(zimageUpscaleModels()),
      ],
      outputs: ["UPSCALE_MODEL"]
    )

    // --- Z-Image custom nodes ---
    info["NunchakuZImageDiTLoader"] = nodeDefinition(
      required: [
        "model_name": optionInput(zimageUnetModels()),
      ],
      optional: [
        "cpu_offload": optionInput(["disable", "enable"]),
        "num_blocks_on_gpu": intInput(default: 0),
        "use_pin_memory": optionInput(["disable", "enable"]),
      ],
      outputs: ["MODEL"]
    )
    info["ModelPatchLoader"] = nodeDefinition(
      required: [
        "name": optionInput(zimageControlnetPatchModels()),
      ],
      outputs: ["MODEL_PATCH"]
    )
    info["ZImageFunControlnet"] = nodeDefinition(
      required: [
        "model": modelInput(),
        "model_patch": modelPatchInput(),
        "inpaint_image": imageInput(),
        "mask": maskInput(),
        "vae": vaeInput(),
      ],
      optional: [
        "strength": floatInput(default: 0.5),
        "start": floatInput(default: 0.0),
        "end": floatInput(default: 1.0),
      ],
      outputs: ["MODEL"]
    )

    // --- Core ComfyUI nodes (pipeline) ---
    info["CLIPTextEncode"] = nodeDefinition(
      required: [
        "text": stringInput(),
        "clip": clipInput(),
      ],
      requiredOrder: ["clip", "text"],
      outputs: ["CONDITIONING"]
    )
    info["EmptySD3LatentImage"] = nodeDefinition(
      required: [
        "width": intInput(default: 1024),
        "height": intInput(default: 1024),
        "batch_size": intInput(default: 1),
      ],
      requiredOrder: ["width", "height", "batch_size"],
      outputs: ["LATENT"]
    )
    info["EmptyLatentImage"] = nodeDefinition(
      required: [
        "width": intInput(default: 1024),
        "height": intInput(default: 1024),
        "batch_size": intInput(default: 1),
      ],
      requiredOrder: ["width", "height", "batch_size"],
      outputs: ["LATENT"]
    )
    info["SamplerCustomAdvanced"] = nodeDefinition(
      required: [
        "noise": noiseInput(),
        "guider": guiderInput(),
        "sampler": samplerInput(),
        "sigmas": sigmasInput(),
        "latent_image": latentInput(),
      ],
      outputs: ["LATENT", "LATENT"]
    )
    info["VAEDecode"] = nodeDefinition(
      required: [
        "samples": latentInput(),
        "vae": vaeInput(),
      ],
      outputs: ["IMAGE"]
    )
    info["VAEEncode"] = nodeDefinition(
      required: [
        "pixels": imageInput(),
        "vae": vaeInput(),
      ],
      outputs: ["LATENT"]
    )

    // --- Sampling helper nodes ---
    info["BasicGuider"] = nodeDefinition(
      required: [
        "model": modelInput(),
        "conditioning": conditioningInput(),
      ],
      outputs: ["GUIDER"]
    )
    info["CFGGuider"] = nodeDefinition(
      required: [
        "model": modelInput(),
        "positive": conditioningInput(),
        "negative": conditioningInput(),
        "cfg": floatInput(default: 1.0),
      ],
      outputs: ["GUIDER"]
    )
    info["BasicScheduler"] = nodeDefinition(
      required: [
        "model": modelInput(),
        "scheduler": optionInput(["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"]),
        "steps": intInput(default: 9),
        "denoise": floatInput(default: 1.0),
      ],
      requiredOrder: ["model", "scheduler", "steps", "denoise"],
      outputs: ["SIGMAS"]
    )
    info["KSamplerSelect"] = nodeDefinition(
      required: [
        "sampler_name": optionInput(["euler", "heun", "dpmpp_2m", "dpmpp_2s_ancestral", "deis", "ddim", "uni_pc", "res_2s"]),
      ],
      outputs: ["SAMPLER"]
    )
    info["RandomNoise"] = nodeDefinition(
      required: [
        "noise_seed": intInput(default: 0),
      ],
      outputs: ["NOISE"]
    )
    info["SetLatentNoiseMask"] = nodeDefinition(
      required: [
        "samples": latentInput(),
        "mask": maskInput(),
      ],
      outputs: ["LATENT"]
    )
    info["DifferentialDiffusion"] = nodeDefinition(
      required: [
        "model": modelInput(),
      ],
      outputs: ["MODEL"]
    )

    // --- KSampler nodes (used by some Krita workflows instead of SamplerCustomAdvanced) ---
    info["KSampler"] = nodeDefinition(
      required: [
        "model": modelInput(),
        "seed": intInput(default: 0),
        "steps": intInput(default: 9),
        "cfg": floatInput(default: 0.0),
        "sampler_name": optionInput(["euler", "heun", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2s_ancestral", "deis", "ddim", "uni_pc", "res_2s"]),
        "scheduler": optionInput(["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"]),
        "positive": conditioningInput(),
        "negative": conditioningInput(),
        "latent_image": latentInput(),
      ],
      requiredOrder: ["model", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "latent_image"],
      optional: [
        "denoise": floatInput(default: 1.0),
      ],
      optionalOrder: ["denoise"],
      outputs: ["LATENT"]
    )
    info["KSamplerAdvanced"] = nodeDefinition(
      required: [
        "model": modelInput(),
        "add_noise": optionInput(["enable", "disable"]),
        "noise_seed": intInput(default: 0),
        "steps": intInput(default: 20),
        "cfg": floatInput(default: 8.0),
        "sampler_name": optionInput(["euler", "heun", "dpmpp_2m", "dpmpp_2s_ancestral", "deis", "ddim", "uni_pc", "res_2s"]),
        "scheduler": optionInput(["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"]),
        "positive": conditioningInput(),
        "negative": conditioningInput(),
        "latent_image": latentInput(),
      ],
      requiredOrder: ["model", "add_noise", "noise_seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative", "latent_image"],
      optional: [
        "start_at_step": intInput(default: 0),
        "end_at_step": intInput(default: 10000),
        "return_with_leftover_noise": optionInput(["disable", "enable"]),
      ],
      optionalOrder: ["start_at_step", "end_at_step", "return_with_leftover_noise"],
      outputs: ["LATENT"]
    )

    // --- Additional loader nodes (probed by Krita during connect/resource discovery) ---
    info["CLIPVisionLoader"] = nodeDefinition(
      required: [
        "clip_name": optionInput(zimageClipVisionModels()),
      ],
      outputs: ["CLIP_VISION"]
    )
    info["StyleModelLoader"] = nodeDefinition(
      required: [
        "style_model_name": optionInput(zimageStyleModels()),
      ],
      outputs: ["STYLE_MODEL"]
    )
    info["LoraLoader"] = nodeDefinition(
      required: [
        "model": modelInput(),
        "clip": clipInput(),
        "lora_name": optionInput(zimageLoraModels(library: loraLibrary)),
        "strength_model": floatInput(default: 1.0),
        "strength_clip": floatInput(default: 1.0),
      ],
      requiredOrder: ["model", "clip", "lora_name", "strength_model", "strength_clip"],
      outputs: ["MODEL", "CLIP"]
    )
    info["INPAINT_LoadInpaintModel"] = nodeDefinition(
      required: [
        "model_name": optionInput(zimageInpaintModels()),
      ],
      outputs: ["INPAINT_MODEL"]
    )

    // --- Image utility nodes ---
    info["ETN_ApplyMaskToImage"] = nodeDefinition(
      required: [
        "image": imageInput(),
        "mask": maskInput(),
      ],
      outputs: ["IMAGE"]
    )
    info["ImageScale"] = nodeDefinition(
      required: [
        "image": imageInput(),
        "upscale_method": optionInput(["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]),
        "width": intInput(default: 1024),
        "height": intInput(default: 1024),
        "crop": optionInput(["disabled", "center"]),
      ],
      requiredOrder: ["image", "upscale_method", "width", "height", "crop"],
      outputs: ["IMAGE"]
    )
    info["ImageUpscaleWithModel"] = nodeDefinition(
      required: [
        "upscale_model": upscaleModelInput(),
        "image": imageInput(),
      ],
      outputs: ["IMAGE"]
    )

    // --- Checkpoint loader (fallback model discovery) ---
    // ckpt_name options are intentionally empty: the /api/etn/model_info
    // "checkpoints" folder returns no metadata, so any name listed here would
    // be deduced from its filename by the Krita plugin and appear as a
    // phantom, unusable SD1.5 checkpoint. Bridge models are served through
    // UNETLoader + model_info/diffusion_models instead.
    info["CheckpointLoaderSimple"] = nodeDefinition(
      required: [
        "ckpt_name": optionInput([]),
      ],
      outputs: ["MODEL", "CLIP", "VAE"]
    )

    // --- Additional diffusion model loader ---
    info["UnetLoaderGGUF"] = nodeDefinition(
      required: [
        "unet_name": optionInput([]),
      ],
      outputs: ["MODEL"]
    )


    // --- NSFW filter node (pass-through — we do not filter) ---
    info["NSFWFilter"] = nodeDefinition(
      required: [
        "images": imageInput(),
      ],
      outputs: ["IMAGE"]
    )

    // --- Tiled VAE nodes (ComfyBox handles tiling internally) ---
    info["VAEEncodeTiled"] = nodeDefinition(
      required: [
        "pixels": imageInput(),
        "vae": vaeInput(),
      ],
      optional: [
        "tile_size": intInput(default: 512),
      ],
      outputs: ["LATENT"]
    )
    info["VAEDecodeTiled"] = nodeDefinition(
      required: [
        "samples": latentInput(),
        "vae": vaeInput(),
      ],
      optional: [
        "tile_size": intInput(default: 512),
      ],
      outputs: ["IMAGE"]
    )

    // --- Batch generation node ---
    info["RepeatLatentBatch"] = nodeDefinition(
      required: [
        "samples": latentInput(),
        "amount": intInput(default: 1),
      ],
      outputs: ["LATENT"]
    )

    // --- Nodes referenced by ComfyBridge parser ---
    info["ImageCrop"] = nodeDefinition(
      required: [
        "image": imageInput(),
        "width": intInput(default: 512),
        "height": intInput(default: 512),
        "x": intInput(default: 0),
        "y": intInput(default: 0),
      ],
      requiredOrder: ["image", "width", "height", "x", "y"],
      outputs: ["IMAGE"]
    )
    info["SplitSigmas"] = nodeDefinition(
      required: [
        "sigmas": sigmasInput(),
        "step": intInput(default: 0),
      ],
      outputs: ["SIGMAS", "SIGMAS"]
    )
    info["PreviewImage"] = nodeDefinition(
      required: [
        "images": imageInput(),
      ],
      outputs: []
    )

    // --- ComfyBox style preset node ---
    // Exposes style presets as a custom node that Krita can discover via /object_info.
    // When the plugin sees this node, it knows style presets are available.
    // The dropdown lists all preset IDs; selecting one applies prompt engineering +
    // tuned generation parameters from ComfyBridgeStylePresets.
    let presetNames = ComfyBoxStylePresets.allPresets.map { $0.id }
    info["ComfyBoxStylePreset"] = nodeDefinition(
      required: [
        "preset_name": optionInput(presetNames),
        "model": modelInput(),
        "clip": clipInput(),
      ],
      optional: [
        "override_steps": intInput(default: 0),
        "override_guidance": floatInput(default: 0.0),
      ],
      outputs: ["MODEL", "CLIP", "STYLE_CONFIG"]
    )

    // Also expose the full preset metadata via a companion info node
    // that Krita can query to display descriptions and categories.
    var presetMetadata: [String: Any] = [:]
    for preset in ComfyBoxStylePresets.allPresets {
      var meta: [String: Any] = [
        "name": preset.name,
        "category": preset.category.rawValue,
        "description": preset.description,
        "prompt_prefix": preset.promptPrefix,
        "prompt_suffix": preset.promptSuffix,
        "negative_prompt": preset.negativePrompt,
        "recommended_model": preset.recommendedModel,
      ]
      if let s = preset.steps { meta["steps"] = s }
      if let g = preset.guidance { meta["guidance"] = g }
      if let w = preset.width { meta["width"] = w }
      if let h = preset.height { meta["height"] = h }
      if let str = preset.img2imgStrength { meta["img2img_strength"] = str }
      presetMetadata[preset.id] = meta
    }
    info["ComfyBoxStylePresetInfo"] = [
      "input": [
        "required": [
          "preset_name": optionInput(presetNames),
        ] as [String: Any],
      ] as [String: Any],
      "output_name": ["STYLE_INFO"],
      "preset_metadata": presetMetadata,
    ] as [String: Any]

    // --- CoffeeShop prompt optimizer node ---
    info["CoffeeShopOptimizer"] = nodeDefinition(
      required: [
        "raw_prompt": stringInput(multiline: true),
        "preset": optionInput([
          "photorealistic", "cinematic", "glamour", "editorial", "intimate",
          "medium_format", "analog", "candid", "noir", "neon", "macro", "pov", "phone",
        ]),
        "content_mode": optionInput(["neutral", "banana", "avocado"]),
        "scene_hint": optionInput(["auto", "portrait", "full_body", "environmental", "action", "intimate", "pov", "macro"]),
        "aspect_ratio": optionInput(["1:1", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9"]),
      ],
      requiredOrder: ["raw_prompt", "preset", "content_mode", "scene_hint", "aspect_ratio"],
      outputs: ["STRING", "STRING", "STRING"],
      outputNames: ["optimized_prompt", "context_block", "photo_block"]
    )

    return finalizedObjectInfo(info)
  }

  // MARK: - Node Definition Helpers

  /// Build a standard node definition dict matching ComfyUI's schema.
  private static func nodeDefinition(
    required: [String: Any],
    requiredOrder: [String]? = nil,
    optional: [String: Any] = [:],
    optionalOrder: [String]? = nil,
    outputs: [String],
    outputNames: [String]? = nil
  ) -> [String: Any] {
    let orderedRequired: Any
    if let order = requiredOrder {
      orderedRequired = OrderedDict(order.compactMap { key in
        required[key].map { (key, $0) }
      })
    } else {
      orderedRequired = required
    }

    let orderedOptional: Any?
    if !optional.isEmpty {
      if let order = optionalOrder {
        orderedOptional = OrderedDict(order.compactMap { key in
          optional[key].map { (key, $0) }
        })
      } else {
        orderedOptional = optional
      }
    } else {
      orderedOptional = nil
    }

    var input: [String: Any] = ["required": orderedRequired]
    if let opt = orderedOptional {
      input["optional"] = opt
    }
    let names = outputNames ?? outputs
    return [
      "name": "",
      "display_name": "",
      "description": "",
      "category": "_for_testing",
      "output_node": false,
      "python_module": "nodes",
      "input": input,
      "output": outputs,
      "output_name": names,
      "output_is_list": outputs.map { _ in false },
    ]
  }

  private static func finalizedObjectInfo(_ info: [String: Any]) -> [String: Any] {
    var finalized = info
    for (nodeName, value) in info {
      guard var definition = value as? [String: Any] else { continue }

      definition["name"] = nonEmptyString(definition["name"]) ?? nodeName
      definition["display_name"] = nonEmptyString(definition["display_name"]) ?? displayName(for: nodeName)
      definition["description"] = nonEmptyString(definition["description"]) ?? ""
      definition["category"] = category(for: nodeName)
      definition["output_node"] = outputNodeTypes.contains(nodeName) || ((definition["output_node"] as? Bool) ?? false)
      definition["python_module"] = "nodes"

      let output = stringArray(definition["output"]) ?? stringArray(definition["output_name"]) ?? []
      let outputNames = stringArray(definition["output_name"]) ?? output
      definition["output"] = output
      definition["output_name"] = outputNames
      definition["output_is_list"] = output.map { _ in false }

      finalized[nodeName] = definition
    }
    return finalized
  }

  private static func nonEmptyString(_ value: Any?) -> String? {
    guard let string = value as? String, !string.isEmpty else { return nil }
    return string
  }

  private static let outputNodeTypes: Set<String> = [
    "ETN_SaveImageCache",
    "PreviewImage"
  ]

  private static func stringArray(_ value: Any?) -> [String]? {
    if let strings = value as? [String] {
      return strings
    }
    if let values = value as? [Any] {
      return values.compactMap { $0 as? String }
    }
    return nil
  }

  private static func displayName(for nodeName: String) -> String {
    nodeName
      .replacingOccurrences(of: "_", with: " ")
      .replacingOccurrences(
        of: "([a-z0-9])([A-Z])",
        with: "$1 $2",
        options: .regularExpression
      )
  }

  private static func category(for nodeName: String) -> String {
    switch nodeName {
    case "ComfyBoxStylePreset", "ComfyBoxStylePresetInfo", "CoffeeShopOptimizer":
      return "CoffeeShop"

    case "UNETLoader", "CLIPLoader", "DualCLIPLoader", "DualCLIPLoaderGGUF",
         "VAELoader", "ControlNetLoader", "UpscaleModelLoader",
         "NunchakuZImageDiTLoader", "ModelPatchLoader", "IPAdapterModelLoader",
         "CLIPVisionLoader", "StyleModelLoader", "LoraLoader",
         "INPAINT_LoadInpaintModel", "INPAINT_LoadFooocusInpaint",
         "CheckpointLoaderSimple", "UnetLoaderGGUF":
      return "loaders"

    case "CLIPTextEncode", "BasicGuider", "CFGGuider", "ZImageFunControlnet",
         "IPAdapter", "ETN_Translate":
      return "conditioning"

    case "EmptySD3LatentImage", "EmptyLatentImage", "VAEEncode", "VAEEncodeTiled",
         "INPAINT_VAEEncodeInpaintConditioning", "SetLatentNoiseMask",
         "RepeatLatentBatch":
      return "latent"

    case "SamplerCustomAdvanced", "BasicScheduler", "KSamplerSelect", "RandomNoise",
         "DifferentialDiffusion", "KSampler", "KSamplerAdvanced", "SplitSigmas":
      return "sampling"

    case "ETN_LoadImageCache", "ETN_SaveImageCache", "INPAINT_ShrinkMask",
         "INPAINT_StabilizeMask", "INPAINT_ColorMatch", "INPAINT_ExpandMask",
         "INPAINT_MaskedBlur", "InpaintPreprocessor", "DepthAnythingV2Preprocessor",
         "VAEDecode", "VAEDecodeTiled", "ETN_ApplyMaskToImage", "ImageScale",
         "ImageUpscaleWithModel", "NSFWFilter", "ImageCrop", "PreviewImage":
      return "image"

    default:
      return "_for_testing"
    }
  }

  // MARK: - Input Type Helpers

  /// String input: ["STRING", {"multiline": Bool}]
  private static func stringInput(multiline: Bool = false) -> [Any] {
    return ["STRING", ["multiline": multiline] as [String: Any]]
  }

  /// Integer input with default: ["INT", {"default": N}]
  private static func intInput(default value: Int) -> [Any] {
    return ["INT", ["default": value] as [String: Any]]
  }

  /// Float input with default: ["FLOAT", {"default": N}]
  private static func floatInput(default value: Float) -> [Any] {
    return ["FLOAT", ["default": value] as [String: Any]]
  }

  /// Dropdown/option input: [["option1", "option2", ...]]
  private static func optionInput(_ options: [String]) -> [Any] {
    return [options]
  }

  /// Typed connector inputs.
  private static func imageInput() -> [Any] { return ["IMAGE"] }
  private static func maskInput() -> [Any] { return ["MASK"] }
  private static func modelInput() -> [Any] { return ["MODEL"] }
  private static func vaeInput() -> [Any] { return ["VAE"] }
  private static func clipInput() -> [Any] { return ["CLIP"] }
  private static func latentInput() -> [Any] { return ["LATENT"] }
  private static func conditioningInput() -> [Any] { return ["CONDITIONING"] }
  private static func noiseInput() -> [Any] { return ["NOISE"] }
  private static func guiderInput() -> [Any] { return ["GUIDER"] }
  private static func samplerInput() -> [Any] { return ["SAMPLER"] }
  private static func sigmasInput() -> [Any] { return ["SIGMAS"] }
  private static func ipadapterInput() -> [Any] { return ["IPADAPTER"] }
  private static func modelPatchInput() -> [Any] { return ["MODEL_PATCH"] }
  private static func upscaleModelInput() -> [Any] { return ["UPSCALE_MODEL"] }

  // MARK: - Model Discovery

  // These return the model options that the Krita plugin will see when it queries
  // available models. For Phase 1 we declare the known Z-Image model names.
  // Phase 2+ can scan the filesystem for actually available models.

  private static func zimageUnetModels() -> [String] {
    // FIBO ("briaai/FIBO") and Chroma ("chroma-8.9b") are deliberately not
    // listed: the Krita plugin cannot build workloads for those archs, and
    // options without model_info metadata surface as phantom entries.
    [
      // Z-Image (Flux 1) family
      "z-image-turbo",
      "z-image-turbo-bf16",
      "z-image-turbo-q8",
      "z-image-turbo-q4",
      // Klein (Flux 2) family
      "klein-4b-q8",
      "klein-9b-q8",
      // Moody (CivitAI Z-Image checkpoints)
      "moody-wild-v4",
      "moody-wild-v4-distilled",
      "moody-wild-v4-fp8",
      "moody-real-v6",
      "cyberrealistic-v5",
    ]
  }

  private static func zimageClipModels() -> [String] {
    [
      "qwen_3_4b",
      "qwen_3_4b_q8",
    ]
  }

  private static func zimageVaeModels() -> [String] {
    [
      "z-image-vae",
      "flux-vae",
    ]
  }

  /// Default ControlNet directory path — matches zimageLoraModels pattern.
  private static let controlnetDirectoryPath = ("~/bin/zimage/controlnet" as NSString).expandingTildeInPath

  private static func zimageControlnetModels() -> [String] {
    // Phase 4: dynamically scan ControlNet directory for weights.
    // Supports .safetensors files and subdirectories containing them.
    let controlnetDir = controlnetDirectoryPath
    let fm = FileManager.default
    guard let entries = try? fm.contentsOfDirectory(atPath: controlnetDir) else {
      return []
    }
    var models: [String] = []
    for entry in entries {
      let fullPath = controlnetDir + "/" + entry
      var isDir: ObjCBool = false
      if fm.fileExists(atPath: fullPath, isDirectory: &isDir) {
        if isDir.boolValue {
          // Subdirectory containing .safetensors — use directory name as model name
          if let subEntries = try? fm.contentsOfDirectory(atPath: fullPath),
             subEntries.contains(where: { $0.hasSuffix(".safetensors") }) {
            models.append(entry)
          }
        } else if entry.hasSuffix(".safetensors") {
          // Single safetensors file
          models.append(entry)
        }
      }
    }
    // Also include HuggingFace model IDs that are known to work
    models.append("alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1")
    return models.sorted()
  }

  private static func zimageControlnetPatchModels() -> [String] {
    // Phase 4: ModelPatchLoader uses the same model list as ControlNetLoader.
    zimageControlnetModels()
  }

  private static func zimageUpscaleModels() -> [String] {
    [
      "4x-UltraSharp",
      "RealESRGAN_x4",
      "OmniSR_X2_DIV2K.safetensors",
      "OmniSR_X3_DIV2K.safetensors",
      "OmniSR_X4_DIV2K.safetensors",
      "4x_NMKD-Superscale-SP_178000_G.pth",
      "seedvr2-3b",
    ]
  }

  private static func zimageClipVisionModels() -> [String] {
    [
      "clip-vit-large-patch14",
    ]
  }

  private static func zimageStyleModels() -> [String] {
    [
      "style-model-default",
    ]
  }

  private static func zimageInpaintModels() -> [String] {
    [
      "MAT_Places512_G_fp16.safetensors",
    ]
  }

  /// Default LoRA directory path. Matches the wrapper script's --lora location.
  private static let loraDirectoryPath = ("~/bin/zimage/loras" as NSString).expandingTildeInPath

  private static func zimageLoraModels(library: LoRALibrary? = nil) -> [String] {
    // When a LoRA Library is available, return non-quarantined entries merged
    // with the flat LoRA directory. The flat directory is what bare LoraLoader
    // names resolve against at generation time and where /api/etn/upload/loras
    // stores plugin-uploaded files, so its contents must stay visible even
    // when the library index is the primary source.
    if let library {
      var names = Set(library.list(includeQuarantined: false).map { $0.filename })
      names.formUnion(flatDirectoryLoraModels())
      return names.sorted()
    }

    return flatDirectoryLoraModels().sorted()
  }

  /// Scan the flat LoRA directory for .safetensors files.
  private static func flatDirectoryLoraModels() -> [String] {
    let fm = FileManager.default
    guard let entries = try? fm.contentsOfDirectory(atPath: loraDirectoryPath) else {
      return []
    }
    return entries.filter { $0.hasSuffix(".safetensors") }
  }
}
