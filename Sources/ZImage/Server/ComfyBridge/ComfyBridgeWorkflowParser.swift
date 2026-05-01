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
  let prompt: String
  let negativePrompt: String?
  let width: Int
  let height: Int
  let steps: Int
  let guidance: Float
  let seed: UInt64?
  let batchSize: Int
  /// The node ID of the output node (ETN_SaveImageCache or PreviewImage).
  let outputNodeId: String

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

  /// Whether this is an inpainting request.
  var isInpaint: Bool { inpaintImageId != nil }

  // Populated by executor before passing to generate handler.
  // Not set by parser — must be filled from image cache.
  var inpaintImageData: Data?
  var maskImageData: Data?
}

/// Result of a ComfyUI bridge generation.
struct ComfyBridgeGenerateResult: Sendable {
  let outputPath: String
  let durationMs: Int
}

/// Parses ComfyUI workflow JSON into generation parameters.
enum ComfyBridgeWorkflowParser {

  struct ParseError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { self.description = message }
  }

  /// Parse a /prompt request body into generation parameters.
  static func parse(_ json: [String: Any]) throws -> ComfyBridgeGenerateRequest {
    guard let promptId = json["prompt_id"] as? String else {
      throw ParseError("Missing prompt_id")
    }
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
        positivePrompt = first.inputs["text"] as? String
      }
      if textNodes.count > 1 {
        negativePrompt = textNodes[1].inputs["text"] as? String
      }
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
    let schedulerNode = nodes.values.first { $0.classType == "BasicScheduler" }
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
    let cfg: Float
    if let guider = guiderNode, guider.classType == "CFGGuider" {
      cfg = floatValue(guider.inputs["cfg"]) ?? 0.0
    } else {
      cfg = 0.0
    }

    // --- Seed ---
    let noiseNode = nodes.values.first { $0.classType == "RandomNoise" }
    let seed = uint64Value(noiseNode?.inputs["noise_seed"])

    // --- Output node ---
    let outputNodeId: String
    if let saveNode = nodes.values.first(where: { $0.classType == "ETN_SaveImageCache" }) {
      outputNodeId = saveNode.id
    } else if let previewNode = nodes.values.first(where: { $0.classType == "PreviewImage" }) {
      outputNodeId = previewNode.id
    } else {
      outputNodeId = nodes.keys.sorted { $0.localizedStandardCompare($1) == .orderedDescending }.first ?? "1"
    }

    // --- Phase 3: Inpainting detection ---
    // Detect inpaint workflows by finding ETN_LoadImageCache nodes.
    // The first one is typically the input image, the second is the mask.
    // Node connections determine which is image vs mask.
    let (inpaintImageId, maskImageId) = extractInpaintImageIds(nodes: nodes)

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

    // Debug: log workflow structure
    let _nodeTypes = nodes.values.map { $0.classType }.sorted()
    let _loraDesc = loras.isEmpty ? "none" : loras.map { "\($0.name)@\($0.scale)" }.joined(separator: ", ")
    print("[ComfyBridge] workflow nodes: \(_nodeTypes.joined(separator: ", ")), parsed: \(width)x\(height) denoise=\(denoise) loras=\(_loraDesc)")

    return ComfyBridgeGenerateRequest(
      promptId: promptId,
      clientId: clientId,
      prompt: prompt,
      negativePrompt: negativePrompt,
      width: width,
      height: height,
      steps: steps,
      guidance: cfg,
      seed: seed,
      batchSize: max(batchSize, 1),
      outputNodeId: outputNodeId,
      inpaintImageId: inpaintImageId,
      maskImageId: maskImageId,
      denoise: denoise,
      maskGrow: maskGrow,
      maskFeather: maskFeather,
      maskCropX: maskCropX,
      maskCropY: maskCropY,
      loras: loras
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

  /// Follow a node reference to resolve the text value from a CLIPTextEncode node.
  private static func resolveTextInput(key: String, from node: WorkflowNode, nodes: [String: WorkflowNode]) -> String? {
    guard let ref = node.inputs[key] as? [Any],
          let refNodeId = ref.first as? String,
          let textNode = nodes[refNodeId] else {
      return nil
    }
    return textNode.inputs["text"] as? String
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
}

/// A node in the ComfyUI workflow graph.
private struct WorkflowNode {
  let id: String
  let classType: String
  let inputs: [String: Any]
}
