// ComfyBridgeWorkflowParser.swift — Extracts generation parameters from ComfyUI workflow JSON
//
// Traverses the ComfyUI node graph to find generation parameters (prompt, dimensions,
// steps, seed, etc.) and maps them to ZImageCLI's internal format.
//
// Phase 2: txt2img workflow parsing.
// Phase 3: inpaint workflow parsing (latent-space inpainting).

import Foundation

/// Parameters extracted from a ComfyUI workflow graph for Z-Image generation.
struct ComfyBridgeGenerateRequest: Sendable {
  let promptId: String
  let clientId: String
  var prompt: String
  let negativePrompt: String?
  let width: Int
  let height: Int
  let steps: Int
  let guidance: Float
  /// Mutable so the executor can increment the seed per batch item.
  var seed: UInt64?
  /// Mutable so the executor can loop batch items as single generations.
  var batchSize: Int
  /// The node ID of the output node (ETN_SaveImageCache or PreviewImage).
  let outputNodeId: String
  /// Sampler name extracted from KSamplerSelect node (e.g. "euler", "res_2s").
  let sampler: String?
  /// Sigma schedule extracted from BasicScheduler node (e.g. "normal", "karras", "exponential").
  let sigmaSchedule: String?
  /// Levels min (black point) for contrast adjustment. Default 0.0.
  let levelsMin: Float?
  /// Levels max (white point) for contrast adjustment. Default 1.0.
  let levelsMax: Float?

  // --- Phase 3: Inpainting fields ---

  /// Image cache ID of the input image to inpaint (from ETN_LoadImageCache node).
  let inpaintImageId: String?
  /// Image cache ID of the mask image (from ETN_LoadImageCache node).
  let maskImageId: String?
  /// Denoising strength (0.0–1.0). Higher = more change. Default 1.0 for txt2img.
  let denoise: Float
  /// Mask expansion in pixels (from INPAINT_ExpandMask grow parameter).
  let maskGrow: Int
  /// Mask feather radius in pixels (from INPAINT_ExpandMask feather parameter).
  let maskFeather: Int
  /// ImageCrop x,y offset — used to crop full-canvas mask to selection bounds.
  let maskCropX: Int
  let maskCropY: Int

  // --- Phase 4: LoRA fields ---

  /// LoRAs extracted from LoraLoader nodes in the workflow.
  /// Each entry is (filename, strength_model scale).
  let loras: [(name: String, scale: Float)]

  // --- Phase 4: ControlNet fields ---

  /// ControlNet model name or path (from ControlNetLoader or ModelPatchLoader node).
  let controlnetModel: String?
  /// ControlNet strength (0.0-1.0). Default 0.5.
  let controlnetStrength: Float
  /// ControlNet start percent (when to start applying control). Default 0.0.
  let controlnetStart: Float
  /// ControlNet end percent (when to stop applying control). Default 1.0.
  let controlnetEnd: Float
  /// Image cache ID of the control image (depth map, canny edges, etc.).
  let controlImageId: String?

  // --- Model detection fields ---

  /// Model ID detected from the workflow's model loader node (UNETLoader,
  /// NunchakuZImageDiTLoader, or legacy CheckpointLoaderSimple).
  /// Used for automatic model switching when Krita selects a different model.
  let detectedModel: String?

  /// Optional CoffeeShop optimizer node extracted from the workflow.
  let optimizer: ComfyBridgeOptimizerRequest?

  /// Whether this is an inpainting request.
  var isInpaint: Bool { inpaintImageId != nil }

  /// Whether this request uses ControlNet.
  var isControlNet: Bool { controlnetModel != nil }

  // Populated by executor before passing to generate handler.
  // Not set by parser — must be filled from image cache.
  var inpaintImageData: Data?
  var maskImageData: Data?
  var controlImageData: Data?
}

/// Result of a ComfyUI bridge generation.
struct ComfyBridgeGenerateResult: Sendable {
  let outputPath: String
  let durationMs: Int
}

/// Parameters extracted from a ComfyUI workflow graph for Z-Image upscaling.
struct ComfyBridgeUpscaleRequest: Sendable {
  let promptId: String
  let clientId: String
  /// The node ID of the ETN_LoadImageCache node providing the input image.
  let inputImageNodeId: String
  /// The upscale model name (e.g. "seedvr2-3b").
  let upscaleModelName: String
  /// The node ID of the output node (ETN_SaveImageCache or PreviewImage).
  let outputNodeId: String

  // Populated by executor before passing to upscale handler.
  // Not set by parser — must be filled from image cache.
  var inputImageData: Data?
}

/// Discriminated result of parsing a ComfyUI workflow.
enum ComfyBridgeParsedWorkflow: Sendable {
  case generate(ComfyBridgeGenerateRequest)
  case upscale(ComfyBridgeUpscaleRequest)
}

/// Parses ComfyUI workflow JSON into generation parameters.
enum ComfyBridgeWorkflowParser {

  struct ParseError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { self.description = message }
  }

  /// Parse a /prompt request body into generation parameters.
  static func parse(_ json: [String: Any]) throws -> ComfyBridgeGenerateRequest {
    // ComfyUI frontend doesn't send prompt_id — generate one server-side.
    let promptId = (json["prompt_id"] as? String) ?? UUID().uuidString
    let clientId = json["client_id"] as? String ?? "unknown"

    guard let workflow = json["prompt"] as? [String: Any] else {
      throw ParseError("Missing 'prompt' workflow object")
    }

    // Build a typed node map for easy traversal.
    var nodes: [String: WorkflowNode] = [:]
    for (id, value) in workflow {
      guard let dict = value as? [String: Any],
            let classType = dict["class_type"] as? String,
            let inputs = dict["inputs"] as? [String: Any] else {
        continue
      }
      nodes[id] = WorkflowNode(id: id, classType: classType, inputs: inputs)
    }

    // --- Prompt text ---
    // Traverse from CFGGuider/BasicGuider → CLIPTextEncode → text
    var positivePrompt: String?
    var negativePrompt: String?

    let guiderNode = nodes.values.first { $0.classType == "CFGGuider" || $0.classType == "BasicGuider" }

    if let guider = guiderNode {
      if guider.classType == "CFGGuider" {
        positivePrompt = resolveTextInput(key: "positive", from: guider, nodes: nodes)
        negativePrompt = resolveTextInput(key: "negative", from: guider, nodes: nodes)
      } else {
        positivePrompt = resolveTextInput(key: "conditioning", from: guider, nodes: nodes)
      }
    }

    // Fallback: scan all CLIPTextEncode nodes ordered by ID.
    if positivePrompt == nil {
      let textNodes = nodes.values
        .filter { $0.classType == "CLIPTextEncode" }
        .sorted { $0.id.localizedStandardCompare($1.id) == .orderedAscending }
      if let first = textNodes.first {
        positivePrompt = resolveText(from: first, nodes: nodes)
      }
      if textNodes.count > 1 {
        negativePrompt = resolveText(from: textNodes[1], nodes: nodes)
      }
    }

    let detectedOptimizer = Self.extractOptimizer(from: nodes)
    if positivePrompt == nil {
      positivePrompt = detectedOptimizer?.rawPrompt
    }

    guard let prompt = positivePrompt, !prompt.isEmpty else {
      throw ParseError("Could not extract prompt text from workflow")
    }

    // --- Dimensions ---
    // For txt2img: from EmptySD3LatentImage/EmptyLatentImage node.
    // For inpaint: from ImageCrop node (contains actual selection dimensions).
    // Dimensions are rounded to nearest 16 for latent space alignment.
    let latentNode = nodes.values.first {
      $0.classType == "EmptySD3LatentImage" || $0.classType == "EmptyLatentImage"
    }
    let cropNode = nodes.values.first { $0.classType == "ImageCrop" }

    let width: Int
    let height: Int
    let batchSize: Int

    // Detect inpaint early to choose correct dimension source
    let hasInpaintImages = nodes.values.contains { $0.classType == "ETN_LoadImageCache" }

    if let cn = cropNode, hasInpaintImages {
      // Inpaint with selection crop: use ImageCrop dimensions
      width = roundTo16(intValue(cn.inputs["width"]) ?? 1024)
      height = roundTo16(intValue(cn.inputs["height"]) ?? 1024)
      batchSize = 1
    } else if let ln = latentNode {
      // txt2img: dimensions from empty latent node
      width = roundTo16(intValue(ln.inputs["width"]) ?? 1024)
      height = roundTo16(intValue(ln.inputs["height"]) ?? 1024)
      batchSize = intValue(ln.inputs["batch_size"]) ?? 1
    } else if hasInpaintImages {
      // Inpaint without ImageCrop or EmptyLatentImage — set width/height to 0
      // to signal downstream code to derive from the actual inpaint image
      width = 0
      height = 0
      batchSize = 1
    } else {
      // Fallback
      width = 1024
      height = 1024
      batchSize = 1
    }

    // --- Steps ---
    // Krita emits BasicScheduler for the common sigma schedules, but alternate
    // scheduler settings produce AlignYourStepsScheduler, GITSScheduler,
    // PolyexponentialScheduler, LaplaceScheduler, or Flux2Scheduler — all of
    // which carry a "steps" input. Sort by node id so multi-pass workflows
    // resolve deterministically, preferring BasicScheduler (the primary pass).
    let schedulerClassTypes: Set<String> = [
      "BasicScheduler", "AlignYourStepsScheduler", "GITSScheduler",
      "PolyexponentialScheduler", "LaplaceScheduler", "Flux2Scheduler",
    ]
    let schedulerNodes = nodes.values
      .filter { schedulerClassTypes.contains($0.classType) }
      .sorted { $0.id.localizedStandardCompare($1.id) == .orderedAscending }
    let schedulerNode = schedulerNodes.first { $0.classType == "BasicScheduler" } ?? schedulerNodes.first
    // Also check KSampler for denoise (Krita may use KSampler instead of BasicScheduler)
    let kSamplerNode = nodes.values.first { $0.classType == "KSampler" || $0.classType == "KSamplerAdvanced" }
    let steps = intValue(schedulerNode?.inputs["steps"]) ?? 9

    // --- Denoise strength ---
    // BasicScheduler has an optional "denoise" input. Default 1.0 (full denoise for txt2img).
    // Krita AI Diffusion uses SplitSigmas to control effective denoise: it generates the
    // full sigma schedule (denoise=1.0) then splits at a step, using only the second half.
    // Effective denoise = 1.0 - (splitStep / totalSteps).
    let splitNode = nodes.values.first { $0.classType == "SplitSigmas" }
    let splitStep = intValue(splitNode?.inputs["step"])
    let baseDenoise = floatValue(schedulerNode?.inputs["denoise"]) ?? floatValue(kSamplerNode?.inputs["denoise"]) ?? 1.0
    let denoise: Float
    if let split = splitStep, split > 0 {
      denoise = 1.0 - (Float(split) / Float(steps))
    } else {
      denoise = baseDenoise
    }

    // --- CFG ---
    // CFGGuider carries `cfg`; but Krita AI Diffusion drives checkpoint models with a
    // KSampler that carries `cfg` inline (and a BasicGuider with no cfg). Read the
    // KSampler's cfg too — otherwise the user's CFG is silently dropped to 0.0, which
    // renders a base model blurry/washed.
    let cfg: Float
    if let guider = guiderNode, guider.classType == "CFGGuider", let c = floatValue(guider.inputs["cfg"]) {
      cfg = c
    } else if let ks = kSamplerNode, let c = floatValue(ks.inputs["cfg"]) {
      cfg = c
    } else {
      cfg = 0.0
    }

    // --- Seed ---
    let noiseNode = nodes.values.first { $0.classType == "RandomNoise" }
    let seed = uint64Value(noiseNode?.inputs["noise_seed"])

    // --- Sampler name ---
    let samplerSelectNode = nodes.values.first { $0.classType == "KSamplerSelect" }
    let samplerName = samplerSelectNode?.inputs["sampler_name"] as? String

    // --- Sigma schedule ---
    // Extract the scheduler field from BasicScheduler (normal, karras, exponential, beta, sgm_uniform).
    let sigmaScheduleName = schedulerNode?.inputs["scheduler"] as? String

    // --- Output node ---
    let outputNodeId: String
    if let saveNode = nodes.values.first(where: { $0.classType == "ETN_SaveImageCache" }) {
      outputNodeId = saveNode.id
    } else if let previewNode = nodes.values.first(where: { $0.classType == "PreviewImage" }) {
      outputNodeId = previewNode.id
    } else {
      outputNodeId = nodes.keys.sorted { $0.localizedStandardCompare($1) == .orderedDescending }.first ?? "1"
    }

    // --- RepeatLatentBatch node ---
    // If a RepeatLatentBatch node is present, its amount field overrides the batch size.
    let repeatBatchNode = nodes.values.first { $0.classType == "RepeatLatentBatch" }
    let repeatAmount = intValue(repeatBatchNode?.inputs["amount"]) ?? 1
    let finalBatchSize = repeatAmount > 1 ? repeatAmount : batchSize

    // --- Phase 3: Inpainting detection ---
    // Detect inpaint workflows by finding ETN_LoadImageCache nodes.
    // The first one is typically the input image, the second is the mask.
    // Node connections determine which is image vs mask.
    // (var: structural control workflows below may reclaim a cache image
    // that the fallback heuristic misattributed as the inpaint source.)
    var (inpaintImageId, maskImageId) = extractInpaintImageIds(nodes: nodes)

    // --- Phase 3: Mask preprocessing parameters ---
    // Extract grow and feather from INPAINT_ExpandMask node.
    let expandNode = nodes.values.first { $0.classType == "INPAINT_ExpandMask" }
    let maskGrow = intValue(expandNode?.inputs["grow"]) ?? 0
    let maskFeather = intValue(expandNode?.inputs["feather"]) ?? 0
    let maskCropX = intValue(cropNode?.inputs["x"]) ?? 0
    let maskCropY = intValue(cropNode?.inputs["y"]) ?? 0

    // --- Phase 4: LoRA extraction ---
    // Find all LoraLoader nodes and extract lora_name + strength_model.
    // LoraLoader nodes chain: each feeds its MODEL output to the next's model input.
    // We collect all of them regardless of chain order — the pipeline applies them all.
    let loraLoaderNodes = nodes.values
      .filter { $0.classType == "LoraLoader" }
      .sorted { $0.id.localizedStandardCompare($1.id) == .orderedAscending }
    var loras: [(name: String, scale: Float)] = []
    for loraNode in loraLoaderNodes {
      guard let loraName = loraNode.inputs["lora_name"] as? String, !loraName.isEmpty else {
        continue
      }
      let strengthModel = floatValue(loraNode.inputs["strength_model"]) ?? 1.0
      loras.append((name: loraName, scale: strengthModel))
    }

    // --- Style preset application ---
    // If the workflow contains a ComfyBoxStylePreset node, apply its prompt engineering
    // and parameter overrides. The preset modifies the prompt and may override steps/guidance.
    var finalPrompt = prompt
    var finalNegativePrompt = negativePrompt
    var finalSteps = steps
    var finalGuidance = cfg
    var finalWidth = width
    var finalHeight = height

    if let presetNode = nodes.values.first(where: { $0.classType == "ComfyBoxStylePreset" }),
       let presetName = presetNode.inputs["preset_name"] as? String,
       let preset = ComfyBoxStylePresets.presets[presetName] {
      let (enhancedPrompt, enhancedNegative) = preset.apply(prompt: prompt, negativePrompt: negativePrompt)
      finalPrompt = enhancedPrompt
      finalNegativePrompt = enhancedNegative
      // Apply parameter overrides from preset (unless user explicitly overrode in the node).
      let overrideSteps = intValue(presetNode.inputs["override_steps"]) ?? 0
      let overrideGuidance = floatValue(presetNode.inputs["override_guidance"]) ?? 0.0
      if overrideSteps > 0 {
        finalSteps = overrideSteps
      } else if let presetSteps = preset.steps {
        finalSteps = presetSteps
      }
      if overrideGuidance > 0.0 {
        finalGuidance = overrideGuidance
      } else if let presetGuidance = preset.guidance {
        finalGuidance = presetGuidance
      }
      if let presetWidth = preset.width { finalWidth = presetWidth }
      if let presetHeight = preset.height { finalHeight = presetHeight }
      print("[ComfyBridge] applied style preset: \(presetName)")
    }

    // --- Phase 4: ControlNet extraction ---
    // Krita sends ZImageFunControlnet nodes for Z-Image native controlnet workflows,
    // or ControlNetLoader nodes for standard ComfyUI controlnet workflows.
    // We also check ModelPatchLoader which Krita uses for our custom node.
    let controlnetModel: String?
    var controlnetStrength: Float = 0.5
    var controlnetStart: Float = 0.0
    var controlnetEnd: Float = 1.0
    var controlImageId: String?

    // Check for ZImageFunControlnet node first (our custom node).
    // Multiple control layers chain several ZImageFunControlnet nodes —
    // only the first (by node id) is applied; extras are ignored explicitly.
    let funControlNodes = nodes.values
      .filter { $0.classType == "ZImageFunControlnet" }
      .sorted { $0.id.localizedStandardCompare($1.id) == .orderedAscending }
    if let funControlNode = funControlNodes.first {
      if funControlNodes.count > 1 {
        print("[ComfyBridge] warning: \(funControlNodes.count) ZImageFunControlnet nodes in workflow — only node \(funControlNode.id) is applied")
      }
      // The model_patch input links to a ModelPatchLoader node
      if let patchRef = funControlNode.inputs["model_patch"] as? [Any],
         let patchNodeId = patchRef.first as? String,
         let patchNode = nodes[patchNodeId],
         patchNode.classType == "ModelPatchLoader" {
        controlnetModel = patchNode.inputs["name"] as? String
      } else {
        controlnetModel = nil
      }
      controlnetStrength = floatValue(funControlNode.inputs["strength"]) ?? 0.5
      controlnetStart = floatValue(funControlNode.inputs["start"]) ?? 0.0
      controlnetEnd = floatValue(funControlNode.inputs["end"]) ?? 1.0
      // The control image arrives on "inpaint_image" for inpaint workflows and
      // on "image" for structural control modes (scribble, soft edge, canny,
      // depth, pose). Either may pass through intermediate nodes (ImageScale,
      // HintImageEnchance, preprocessors) before the source ETN_LoadImageCache.
      if let inpaintControlId = resolveImageCacheId(ref: funControlNode.inputs["inpaint_image"], nodes: nodes) {
        controlImageId = inpaintControlId
      } else if let structuralControlId = resolveImageCacheId(ref: funControlNode.inputs["image"], nodes: nodes) {
        controlImageId = structuralControlId
        // Structural control is not inpainting — the control layer's cache
        // image must not double as the inpaint source image (the inpaint
        // heuristic's fallback can misattribute it).
        if inpaintImageId == structuralControlId { inpaintImageId = nil }
        if maskImageId == structuralControlId { maskImageId = nil }
      }
    }
    // Fallback: standard ControlNetLoader node
    else if let controlNetLoader = nodes.values.first(where: { $0.classType == "ControlNetLoader" }) {
      controlnetModel = controlNetLoader.inputs["control_net_name"] as? String
    }
    // Fallback: standalone ModelPatchLoader node
    else if let modelPatchLoader = nodes.values.first(where: { $0.classType == "ModelPatchLoader" }) {
      controlnetModel = modelPatchLoader.inputs["name"] as? String
    }
    else {
      controlnetModel = nil
    }


    // --- Model detection from the workflow's model loader node ---
    // Krita emits UNETLoader (or NunchakuZImageDiTLoader) with the user's
    // selected model. We extract this and map it to a ComfyBox pool model ID.
    let detectedModel = Self.extractDetectedModel(from: nodes)

    // --- CoffeeShop optimizer detection ---
    let optimizer = detectedOptimizer

    // Debug: log workflow structure. Gated behind COMFYBOX_DEBUG_WORKFLOW —
    // the parsed summary includes prompt-derived details and prints per-request.
    if ComfyBridge.debugWorkflowDumpEnabled {
      let _nodeTypes = nodes.values.map { $0.classType }.sorted()
      let _loraDesc = loras.isEmpty ? "none" : loras.map { "\($0.name)@\($0.scale)" }.joined(separator: ", ")
      let _cnDesc = controlnetModel ?? "none"
      let _modelDesc = detectedModel ?? "none"
      print("[ComfyBridge] workflow nodes: \(_nodeTypes.joined(separator: ", ")), parsed: \(width)x\(height) denoise=\(denoise) loras=\(_loraDesc) controlnet=\(_cnDesc) model=\(_modelDesc)")
    }

    return ComfyBridgeGenerateRequest(
      promptId: promptId,
      clientId: clientId,
      prompt: finalPrompt,
      negativePrompt: finalNegativePrompt,
      width: finalWidth,
      height: finalHeight,
      steps: finalSteps,
      guidance: finalGuidance,
      seed: seed,
      batchSize: max(finalBatchSize, 1),
      outputNodeId: outputNodeId,
      sampler: samplerName,
      sigmaSchedule: sigmaScheduleName,
      levelsMin: nil,
      levelsMax: nil,
      inpaintImageId: inpaintImageId,
      maskImageId: maskImageId,
      denoise: denoise,
      maskGrow: maskGrow,
      maskFeather: maskFeather,
      maskCropX: maskCropX,
      maskCropY: maskCropY,
      loras: loras,
      controlnetModel: controlnetModel,
      controlnetStrength: controlnetStrength,
      controlnetStart: controlnetStart,
      controlnetEnd: controlnetEnd,
      controlImageId: controlImageId,
      detectedModel: detectedModel,
      optimizer: optimizer
    )
  }

  // MARK: - Workflow Type Detection

  /// Parse a /prompt request body, detecting whether it is a generate or upscale workflow.
  static func parseWorkflow(_ json: [String: Any]) throws -> ComfyBridgeParsedWorkflow {
    // Peek at the workflow nodes to detect upscale workflows before falling back
    // to the standard generation parser.
    guard let workflow = json["prompt"] as? [String: Any] else {
      throw ParseError("Missing 'prompt' workflow object")
    }

    // Build a lightweight class_type set for detection.
    var hasUpscaleModelLoader = false
    var hasImageUpscaleWithModel = false
    var hasGuider = false

    for (_, value) in workflow {
      guard let dict = value as? [String: Any],
            let classType = dict["class_type"] as? String else { continue }
      switch classType {
      case "UpscaleModelLoader":
        hasUpscaleModelLoader = true
      case "ImageUpscaleWithModel":
        hasImageUpscaleWithModel = true
      case "CFGGuider", "BasicGuider":
        hasGuider = true
      default:
        break
      }
    }

    // Upscale workflows have UpscaleModelLoader + ImageUpscaleWithModel but no guider
    // (no sampler/denoiser — pure model-based upscale).
    if hasUpscaleModelLoader && hasImageUpscaleWithModel && !hasGuider {
      return .upscale(try parseUpscale(json))
    }

    return .generate(try parse(json))
  }

  /// Parse a /prompt request body into upscale parameters.
  ///
  /// Expects a workflow containing:
  /// - `UpscaleModelLoader` — provides `inputs.model_name` (the upscale model)
  /// - `ImageUpscaleWithModel` — connects upscale model to input image
  /// - `ETN_LoadImageCache` — input image source
  /// - `ETN_SaveImageCache` or `PreviewImage` — output node
  static func parseUpscale(_ json: [String: Any]) throws -> ComfyBridgeUpscaleRequest {
    // ComfyUI frontend doesn't send prompt_id — generate one server-side.
    let promptId = (json["prompt_id"] as? String) ?? UUID().uuidString
    let clientId = json["client_id"] as? String ?? "unknown"

    guard let workflow = json["prompt"] as? [String: Any] else {
      throw ParseError("Missing 'prompt' workflow object")
    }

    // Build typed node map.
    var nodes: [String: WorkflowNode] = [:]
    for (id, value) in workflow {
      guard let dict = value as? [String: Any],
            let classType = dict["class_type"] as? String,
            let inputs = dict["inputs"] as? [String: Any] else {
        continue
      }
      nodes[id] = WorkflowNode(id: id, classType: classType, inputs: inputs)
    }

    // --- Upscale model name ---
    guard let upscaleLoaderNode = nodes.values.first(where: { $0.classType == "UpscaleModelLoader" }),
          let modelName = upscaleLoaderNode.inputs["model_name"] as? String else {
      throw ParseError("Could not find UpscaleModelLoader with model_name")
    }

    // --- Input image ---
    // Find the ETN_LoadImageCache that feeds into ImageUpscaleWithModel (directly or
    // through intermediate nodes). The simplest case: ImageUpscaleWithModel.inputs.image
    // references an ETN_LoadImageCache node.
    let upscaleNode = nodes.values.first { $0.classType == "ImageUpscaleWithModel" }

    var inputImageNodeId: String?
    if let upscaleN = upscaleNode,
       let imageRef = upscaleN.inputs["image"] as? [Any],
       let sourceNodeId = imageRef.first as? String {
      // Direct reference to an image source node.
      if let sourceNode = nodes[sourceNodeId], sourceNode.classType == "ETN_LoadImageCache" {
        inputImageNodeId = sourceNode.id
      } else {
        // The image might pass through intermediate nodes (e.g. ImageScale).
        // Walk one level deeper.
        if let intermediateNode = nodes[sourceNodeId] {
          for (_, value) in intermediateNode.inputs {
            if let ref = value as? [Any],
               let refNodeId = ref.first as? String,
               let refNode = nodes[refNodeId],
               refNode.classType == "ETN_LoadImageCache" {
              inputImageNodeId = refNode.id
              break
            }
          }
        }
      }
    }

    // Fallback: use any ETN_LoadImageCache in the workflow.
    if inputImageNodeId == nil {
      if let loadCache = nodes.values.first(where: { $0.classType == "ETN_LoadImageCache" }) {
        inputImageNodeId = loadCache.id
      }
    }

    guard let resolvedInputNodeId = inputImageNodeId else {
      throw ParseError("Could not find input image node (ETN_LoadImageCache) for upscale workflow")
    }

    // --- Output node ---
    let outputNodeId: String
    if let saveNode = nodes.values.first(where: { $0.classType == "ETN_SaveImageCache" }) {
      outputNodeId = saveNode.id
    } else if let previewNode = nodes.values.first(where: { $0.classType == "PreviewImage" }) {
      outputNodeId = previewNode.id
    } else {
      outputNodeId = nodes.keys.sorted { $0.localizedStandardCompare($1) == .orderedDescending }.first ?? "1"
    }

    let nodeTypes = nodes.values.map { $0.classType }.sorted()
    print("[ComfyBridge] upscale workflow nodes: \(nodeTypes.joined(separator: ", ")), model=\(modelName)")

    return ComfyBridgeUpscaleRequest(
      promptId: promptId,
      clientId: clientId,
      inputImageNodeId: resolvedInputNodeId,
      upscaleModelName: modelName,
      outputNodeId: outputNodeId
    )
  }

  // MARK: - Inpaint Detection

  /// Extract inpaint image and mask IDs from ETN_LoadImageCache nodes.
  ///
  /// Heuristic: The plugin uploads images via PUT /api/etn/image/<id> where <id>
  /// is a CRC32 hash. ETN_LoadImageCache nodes reference these by ID.
  ///
  /// To distinguish image from mask, we trace node connections:
  /// - If a LoadImageCache feeds into VAEEncode or INPAINT_MaskedBlur → it's the image
  /// - If a LoadImageCache feeds into SetLatentNoiseMask or INPAINT_ExpandMask → it's the mask
  /// - Fallback: first by node ID order (lower = image, higher = mask)
  private static func extractInpaintImageIds(nodes: [String: WorkflowNode]) -> (String?, String?) {
    let loadCacheNodes = nodes.values
      .filter { $0.classType == "ETN_LoadImageCache" }
      .sorted { $0.id.localizedStandardCompare($1.id) == .orderedAscending }

    guard !loadCacheNodes.isEmpty else { return (nil, nil) }

    // Build reverse dependency map: for each node, which nodes consume its output?
    var consumers: [String: [(nodeId: String, inputKey: String)]] = [:]
    for (_, node) in nodes {
      for (key, value) in node.inputs {
        if let ref = value as? [Any], let refNodeId = ref.first as? String {
          consumers[refNodeId, default: []].append((nodeId: node.id, inputKey: key))
        }
      }
    }

    var imageId: String?
    var maskId: String?

    for cacheNode in loadCacheNodes {
      let id = cacheNode.inputs["id"] as? String ?? cacheNode.inputs["image_id"] as? String
      guard let cacheId = id else { continue }

      // Check what consumes this node's output.
      let nodeConsumers = consumers[cacheNode.id] ?? []
      var isImage = false
      var isMask = false

      for consumer in nodeConsumers {
        let consumerNode = nodes[consumer.nodeId]
        let consumerClass = consumerNode?.classType ?? ""

        // Image consumers
        if ["VAEEncode", "INPAINT_MaskedBlur", "INPAINT_ColorMatch",
            "ETN_ApplyMaskToImage", "ZImageFunControlnet"].contains(consumerClass) {
          if consumer.inputKey == "image" || consumer.inputKey == "inpaint_image" ||
             consumer.inputKey == "pixels" || consumer.inputKey == "source" {
            isImage = true
          }
          if consumer.inputKey == "mask" {
            isMask = true
          }
        }

        // Mask consumers
        if ["SetLatentNoiseMask", "INPAINT_ExpandMask", "INPAINT_StabilizeMask",
            "INPAINT_ShrinkMask", "DifferentialDiffusion"].contains(consumerClass) {
          isMask = true
        }
        if consumer.inputKey == "mask" {
          isMask = true
        }
      }

      if isImage && !isMask {
        imageId = cacheId
      } else if isMask && !isImage {
        maskId = cacheId
      } else if imageId == nil {
        imageId = cacheId  // Fallback: first = image
      } else if maskId == nil {
        maskId = cacheId   // Fallback: second = mask
      }
    }

    return (imageId, maskId)
  }

  // MARK: - Node Graph Traversal

  /// Maximum reference-chain depth when walking the node graph.
  private static let maxTraversalDepth = 8

  /// Follow a node reference to resolve the text value from a CLIPTextEncode node.
  private static func resolveTextInput(key: String, from node: WorkflowNode, nodes: [String: WorkflowNode]) -> String? {
    guard let ref = node.inputs[key] as? [Any],
          let refNodeId = ref.first as? String,
          let textNode = nodes[refNodeId] else {
      return nil
    }
    return resolveText(from: textNode, nodes: nodes)
  }

  /// Resolve prompt text from a node, following "text" input references
  /// recursively. When Krita's translation is enabled, CLIPTextEncode.text is
  /// a reference to an ETN_Translate node whose own "text" input carries the
  /// prompt (as a string or a further reference).
  private static func resolveText(from node: WorkflowNode, nodes: [String: WorkflowNode], depth: Int = 0) -> String? {
    guard depth < maxTraversalDepth else { return nil }
    if node.classType == "CoffeeShopOptimizer" {
      return node.inputs["raw_prompt"] as? String
    }
    if let text = node.inputs["text"] as? String {
      return text
    }
    if let ref = node.inputs["text"] as? [Any],
       let refNodeId = ref.first as? String,
       let sourceNode = nodes[refNodeId] {
      return resolveText(from: sourceNode, nodes: nodes, depth: depth + 1)
    }
    return nil
  }

  /// Follow an image reference through intermediate nodes (ImageScale,
  /// HintImageEnchance, preprocessor nodes, ...) to the source
  /// ETN_LoadImageCache node's cache id.
  private static func resolveImageCacheId(ref: Any?, nodes: [String: WorkflowNode], depth: Int = 0) -> String? {
    guard depth < maxTraversalDepth,
          let refArray = ref as? [Any],
          let refNodeId = refArray.first as? String,
          let node = nodes[refNodeId] else {
      return nil
    }
    if node.classType == "ETN_LoadImageCache" {
      return (node.inputs["id"] as? String) ?? (node.inputs["image_id"] as? String)
    }
    // Follow the node's own image-like input toward the source.
    for key in ["image", "images", "pixels", "inpaint_image"] {
      if let resolved = resolveImageCacheId(ref: node.inputs[key], nodes: nodes, depth: depth + 1) {
        return resolved
      }
    }
    return nil
  }

  // MARK: - Value Extraction Helpers

  private static func intValue(_ value: Any?) -> Int? {
    if let i = value as? Int { return i }
    if let n = value as? NSNumber { return n.intValue }
    if let d = value as? Double { return Int(d) }
    return nil
  }

  private static func floatValue(_ value: Any?) -> Float? {
    if let f = value as? Float { return f }
    if let n = value as? NSNumber { return n.floatValue }
    if let d = value as? Double { return Float(d) }
    return nil
  }

  private static func uint64Value(_ value: Any?) -> UInt64? {
    guard let value else { return nil }
    if let i = value as? Int { return UInt64(i) }
    if let n = value as? NSNumber { return n.uint64Value }
    return nil
  }

  private static func roundTo16(_ value: Int) -> Int {
    return ((value + 15) / 16) * 16
  }

  // MARK: - Model Detection

  /// Known exact model name to pool ID mappings.
  private static let exactModelMap: [String: String] = [
    "z-image-turbo-bf16": "z-image-turbo-bf16",
    "z-image-turbo-q8": "z-image-turbo-q8",
    "z-image-turbo-q4": "z-image-turbo-q4",
    "z-image-turbo": "z-image-turbo-bf16",
    "klein-4b-q8": "klein-4b-q8",
    "klein-9b-q8": "klein-9b-q8",
    "briaai/FIBO": "briaai/FIBO",
    "chroma-8.9b": "chroma-8.9b",
  ]

  /// Partial model name patterns for fuzzy matching (checked in order).
  private static let partialModelPatterns: [(pattern: String, poolId: String)] = [
    ("z-image-turbo-bf16", "z-image-turbo-bf16"),
    ("z-image-turbo-q8", "z-image-turbo-q8"),
    ("z-image-turbo-q4", "z-image-turbo-q4"),
    ("z-image-turbo", "z-image-turbo-bf16"),
    ("z-image", "z-image-turbo-bf16"),
    ("klein-9b", "klein-9b-q8"),
    ("klein-4b", "klein-4b-q8"),
    ("klein", "klein-4b-q8"),
    ("fibo", "briaai/FIBO"),
    ("chroma", "chroma-8.9b"),
  ]

  /// Extract the detected model ID from the workflow's model loader node.
  /// Krita classifies bridge models (served via /api/etn/model_info/
  /// diffusion_models) as FileFormat.diffusion and emits UNETLoader — or
  /// NunchakuZImageDiTLoader for svdq models. CheckpointLoaderSimple is only
  /// a legacy fallback. Returns nil if no loader node is found.
  private static func extractDetectedModel(from nodes: [String: WorkflowNode]) -> String? {
    let loaderCandidates: [(classType: String, inputKey: String)] = [
      ("UNETLoader", "unet_name"),
      ("NunchakuZImageDiTLoader", "model_name"),
      ("CheckpointLoaderSimple", "ckpt_name"),
    ]

    var detectedName: String?
    for (classType, inputKey) in loaderCandidates {
      if let node = nodes.values.first(where: { $0.classType == classType }),
         let name = node.inputs[inputKey] as? String,
         !name.isEmpty {
        detectedName = name
        break
      }
    }
    guard let ckptName = detectedName else {
      return nil
    }

    // Try exact match first.
    if let exactMatch = exactModelMap[ckptName] {
      return exactMatch
    }

    // Try partial/prefix matching (case-insensitive).
    let lowered = ckptName.lowercased()
    for (pattern, poolId) in partialModelPatterns {
      if lowered.contains(pattern.lowercased()) {
        return poolId
      }
    }

    // Unrecognized model — return the raw name so the caller can attempt pool lookup.
    return ckptName
  }

  private static func extractOptimizer(from nodes: [String: WorkflowNode]) -> ComfyBridgeOptimizerRequest? {
    guard let node = nodes.values.first(where: { $0.classType == "CoffeeShopOptimizer" }) else {
      return nil
    }

    return ComfyBridgeOptimizerRequest(
      nodeId: node.id,
      rawPrompt: node.inputs["raw_prompt"] as? String ?? "",
      preset: node.inputs["preset"] as? String ?? "photorealistic",
      contentMode: node.inputs["content_mode"] as? String ?? "neutral",
      sceneHint: node.inputs["scene_hint"] as? String ?? "auto",
      aspectRatio: node.inputs["aspect_ratio"] as? String ?? "1:1",
      character: optionalString(node.inputs["character"]),
      characterDescription: optionalString(node.inputs["character_description"])
    )
  }

  private static func optionalString(_ value: Any?) -> String? {
    guard let value = value as? String else { return nil }
    return value.isEmpty ? nil : value
  }
}

/// A node in the ComfyUI workflow graph.
private struct WorkflowNode {
  let id: String
  let classType: String
  let inputs: [String: Any]
}
