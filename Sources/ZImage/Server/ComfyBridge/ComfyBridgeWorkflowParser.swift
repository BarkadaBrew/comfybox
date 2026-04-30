// ComfyBridgeWorkflowParser.swift — Extracts generation parameters from ComfyUI workflow JSON
//
// Traverses the ComfyUI node graph to find generation parameters (prompt, dimensions,
// steps, seed, etc.) and maps them to ZImageCLI's internal format.
//
// Phase 2: txt2img workflow parsing.

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
    let latentNode = nodes.values.first {
      $0.classType == "EmptySD3LatentImage" || $0.classType == "EmptyLatentImage"
    }
    let width = intValue(latentNode?.inputs["width"]) ?? 1024
    let height = intValue(latentNode?.inputs["height"]) ?? 1024
    let batchSize = intValue(latentNode?.inputs["batch_size"]) ?? 1

    // --- Steps ---
    let schedulerNode = nodes.values.first { $0.classType == "BasicScheduler" }
    let steps = intValue(schedulerNode?.inputs["steps"]) ?? 9

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
      // Use the highest node ID as a fallback.
      outputNodeId = nodes.keys.sorted { $0.localizedStandardCompare($1) == .orderedDescending }.first ?? "1"
    }

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
      outputNodeId: outputNodeId
    )
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
}

/// A node in the ComfyUI workflow graph.
private struct WorkflowNode {
  let id: String
  let classType: String
  let inputs: [String: Any]
}
