import Foundation

// MARK: - Workflow import/export (comfybox#238)

/// Errors surfaced by workflow import/run.
public enum WorkflowError: Error, LocalizedError, CustomStringConvertible {
  case invalidJSON
  case uiFormatNotSupported
  case notAGraph
  case notFound(String)
  case inputFileMissing(node: String, path: String)
  case maskOutputUnsupported(node: String)
  case storeFailed(String)

  public var description: String {
    switch self {
    case .invalidJSON: return "Workflow is not valid JSON"
    case .uiFormatNotSupported:
      return "This is a ComfyUI UI-format workflow (nodes/links arrays). Export it with ComfyUI's 'Save (API Format)' and import that instead."
    case .notAGraph: return "Workflow JSON is not a node graph (expected {\"<id>\": {class_type, inputs}, …})"
    case .notFound(let id): return "Workflow not found: \(id)"
    case .inputFileMissing(let node, let path):
      return "LoadImage node \(node): file not found: \(path) (absolute path, or a name under ~/.comfybox/workflow-inputs/)"
    case .maskOutputUnsupported(let node):
      return "LoadImage node \(node): the MASK output (alpha channel) is referenced — not supported yet; use an explicit mask workflow"
    case .storeFailed(let msg): return "Workflow store failed: \(msg)"
    }
  }

  public var errorDescription: String? { description }
}

/// Compatibility report for an imported ComfyUI graph.
public struct WorkflowCompatReport: Codable, Sendable {
  /// Node class types the parser actively maps to engine parameters.
  public var mappedNodes: [String]
  /// Structural glue the parser safely ignores (VAE plumbing etc.).
  public var glueNodes: [String]
  /// Class types ComfyBox does not understand — the graph may still run, but
  /// whatever these nodes contribute is dropped.
  public var unknownNodes: [String]
  public var nodeCount: Int
  /// Whether the graph parsed into a runnable request at import time.
  public var parses: Bool
  public var parseError: String?

  enum CodingKeys: String, CodingKey {
    case mappedNodes = "mapped_nodes"
    case glueNodes = "glue_nodes"
    case unknownNodes = "unknown_nodes"
    case nodeCount = "node_count"
    case parses
    case parseError = "parse_error"
  }
}

/// One stored workflow: the raw (normalized-format) ComfyUI API graph plus
/// bookkeeping. Persisted as a single JSON file under ~/.comfybox/workflows/.
public struct StoredWorkflow: Sendable {
  public var id: String
  public var name: String
  public var importedAt: Date
  /// The ComfyUI API-format graph (node-id → {class_type, inputs}).
  public var graph: [String: Any]
  public var compat: WorkflowCompatReport

  public func summaryJSON() -> [String: Any] {
    var compatDict: [String: Any] = [
      "mapped_nodes": compat.mappedNodes,
      "glue_nodes": compat.glueNodes,
      "unknown_nodes": compat.unknownNodes,
      "node_count": compat.nodeCount,
      "parses": compat.parses,
    ]
    if let err = compat.parseError { compatDict["parse_error"] = err }
    return [
      "id": id,
      "name": name,
      "imported_at": ISO8601DateFormatter().string(from: importedAt),
      "compat": compatDict,
    ]
  }
}

/// File-backed store for imported workflows + the ComfyUI-graph normalization
/// that adapts generic community workflows to the bridge's executor.
public final class WorkflowStore: @unchecked Sendable {

  /// Node types the bridge parser actively maps.
  static let mappedNodeTypes: Set<String> = [
    "CLIPTextEncode", "EmptySD3LatentImage", "EmptyLatentImage", "ImageCrop",
    "ETN_LoadImageCache", "ETN_SaveImageCache", "ETN_ApplyMaskToImage",
    "BasicScheduler", "AlignYourStepsScheduler", "GITSScheduler",
    "PolyexponentialScheduler", "LaplaceScheduler", "Flux2Scheduler",
    "KSampler", "KSamplerAdvanced", "KSamplerSelect", "SplitSigmas",
    "CFGGuider", "BasicGuider", "RandomNoise", "PreviewImage",
    "RepeatLatentBatch", "INPAINT_ExpandMask", "INPAINT_ColorMatch",
    "INPAINT_MaskedBlur", "INPAINT_ShrinkMask", "INPAINT_StabilizeMask",
    "SetLatentNoiseMask", "DifferentialDiffusion", "LoraLoader",
    "ComfyBoxStylePreset", "CoffeeShopOptimizer", "ControlNetLoader",
    "ModelPatchLoader", "ZImageFunControlnet", "UpscaleModelLoader",
    "ImageUpscaleWithModel", "UNETLoader", "NunchakuZImageDiTLoader",
    "CheckpointLoaderSimple", "VAEEncode",
    // comfybox#154: the parser reads these nodes' `shift` input. AuraFlow IS an
    // SD3 subclass upstream and they differ only in a timestep `multiplier`
    // that cancels out of the sigma grid, so both map to the same field.
    "ModelSamplingAuraFlow", "ModelSamplingSD3",
    // Normalized away at import:
    "LoadImage", "SaveImage",
  ]

  /// Structural glue the parser ignores safely — present in most community
  /// graphs, contributes nothing the engine doesn't already do natively.
  static let glueNodeTypes: Set<String> = [
    "VAEDecode", "VAELoader", "CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader",
    "ConditioningZeroOut", "ConditioningCombine", "ConditioningConcat",
    // comfybox#154 moved `ModelSamplingAuraFlow` and `ModelSamplingSD3` out of
    // here and into `mappedNodeTypes` — their `shift` is now read, not
    // ignored. `ModelSamplingFlux` stays glue: its shift is a LOG-shift
    // feeding a genuinely different curve, not the same warp under another
    // name.
    "ModelSamplingFlux",
    "FluxGuidance", "Note", "MarkdownNote", "PrimitiveNode", "Reroute",
  ]

  private let directory: URL
  private let fm = FileManager.default

  public init(directory: URL? = nil) {
    self.directory = directory ?? FileManager.default
      .homeDirectoryForCurrentUser
      .appendingPathComponent(".comfybox/workflows", isDirectory: true)
  }

  // MARK: Format detection / normalization

  /// Parse raw JSON bytes into an API-format graph. Rejects the UI format
  /// (nodes/links arrays) with a pointed error; accepts either a bare graph
  /// or a {"prompt": {...}} wrapper.
  public static func apiGraph(fromJSON data: Data) throws -> [String: Any] {
    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw WorkflowError.invalidJSON
    }
    if json["nodes"] is [Any], json["links"] is [Any] {
      throw WorkflowError.uiFormatNotSupported
    }
    let graph = (json["prompt"] as? [String: Any]) ?? json
    // Every value must look like a node ({class_type, inputs}).
    let nodeish = graph.values.contains { value in
      guard let node = value as? [String: Any] else { return false }
      return node["class_type"] is String
    }
    guard nodeish else { throw WorkflowError.notAGraph }
    return graph
  }

  /// Adapt generic community nodes to the bridge executor's vocabulary:
  /// - `LoadImage` → `ETN_LoadImageCache`: the referenced file is staged into
  ///   the bridge image cache via `stageImage` and the node rewritten to the
  ///   returned cache id. Referencing LoadImage's MASK output (socket 1) is
  ///   flagged unsupported.
  /// - `SaveImage` → `PreviewImage`: the executor's output lands in the image
  ///   cache / history, which the run route copies out to a file.
  /// Pass `stageImage: nil` for a dry-run (import-time validation) — files
  /// aren't read; placeholder ids are used so the parser can be exercised.
  public static func normalizeGenericNodes(
    _ graph: [String: Any],
    stageImage: ((Data) throws -> String)?
  ) throws -> [String: Any] {
    var out = graph

    // Any consumer referencing [loadImageNodeId, 1] uses the MASK socket.
    for (_, value) in graph {
      guard let node = value as? [String: Any],
            let inputs = node["inputs"] as? [String: Any] else { continue }
      for (_, input) in inputs {
        if let ref = input as? [Any], ref.count == 2,
           let refId = ref[0] as? String, let socket = ref[1] as? Int, socket == 1,
           let refNode = graph[refId] as? [String: Any],
           refNode["class_type"] as? String == "LoadImage" {
          throw WorkflowError.maskOutputUnsupported(node: refId)
        }
      }
    }

    for (id, value) in graph {
      guard var node = value as? [String: Any],
            let classType = node["class_type"] as? String else { continue }
      switch classType {
      case "LoadImage":
        let inputs = node["inputs"] as? [String: Any] ?? [:]
        let name = (inputs["image"] as? String) ?? ""
        let cacheId: String
        if let stageImage {
          let resolved = resolveInputImagePath(name)
          guard let resolved,
                let data = FileManager.default.contents(atPath: resolved), !data.isEmpty else {
            throw WorkflowError.inputFileMissing(node: id, path: name)
          }
          cacheId = try stageImage(data)
        } else {
          cacheId = "dryrun-\(id)"
        }
        node["class_type"] = "ETN_LoadImageCache"
        node["inputs"] = ["id": cacheId]
        out[id] = node
      case "SaveImage":
        node["class_type"] = "PreviewImage"
        var inputs = node["inputs"] as? [String: Any] ?? [:]
        inputs.removeValue(forKey: "filename_prefix")
        node["inputs"] = inputs
        out[id] = node
      default:
        continue
      }
    }
    return out
  }

  /// Resolve a LoadImage `image` value: absolute path as-is, otherwise a name
  /// under ~/.comfybox/workflow-inputs/ (the ComfyUI "input directory" analog).
  public static func resolveInputImagePath(_ name: String) -> String? {
    guard !name.isEmpty else { return nil }
    if name.hasPrefix("/") {
      return FileManager.default.fileExists(atPath: name) ? name : nil
    }
    let candidate = FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent(".comfybox/workflow-inputs")
      .appendingPathComponent(name).path
    return FileManager.default.fileExists(atPath: candidate) ? candidate : nil
  }

  /// Classify a graph's node types into mapped / glue / unknown.
  public static func compatReport(
    for graph: [String: Any],
    parses: Bool,
    parseError: String?
  ) -> WorkflowCompatReport {
    var mapped = Set<String>()
    var glue = Set<String>()
    var unknown = Set<String>()
    var count = 0
    for value in graph.values {
      guard let node = value as? [String: Any],
            let classType = node["class_type"] as? String else { continue }
      count += 1
      if mappedNodeTypes.contains(classType) {
        mapped.insert(classType)
      } else if glueNodeTypes.contains(classType) {
        glue.insert(classType)
      } else {
        unknown.insert(classType)
      }
    }
    return WorkflowCompatReport(
      mappedNodes: mapped.sorted(),
      glueNodes: glue.sorted(),
      unknownNodes: unknown.sorted(),
      nodeCount: count,
      parses: parses,
      parseError: parseError)
  }

  // MARK: Persistence

  public func save(_ workflow: StoredWorkflow) throws {
    do {
      try fm.createDirectory(at: directory, withIntermediateDirectories: true)
      var compatDict: [String: Any] = [
        "mapped_nodes": workflow.compat.mappedNodes,
        "glue_nodes": workflow.compat.glueNodes,
        "unknown_nodes": workflow.compat.unknownNodes,
        "node_count": workflow.compat.nodeCount,
        "parses": workflow.compat.parses,
      ]
      if let err = workflow.compat.parseError { compatDict["parse_error"] = err }
      let record: [String: Any] = [
        "id": workflow.id,
        "name": workflow.name,
        "imported_at": ISO8601DateFormatter().string(from: workflow.importedAt),
        "compat": compatDict,
        "graph": workflow.graph,
      ]
      let data = try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
      try data.write(to: fileURL(for: workflow.id))
    } catch let error as WorkflowError {
      throw error
    } catch {
      throw WorkflowError.storeFailed(error.localizedDescription)
    }
  }

  public func get(_ id: String) -> StoredWorkflow? {
    guard let data = fm.contents(atPath: fileURL(for: id).path),
          let record = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let graph = record["graph"] as? [String: Any],
          let name = record["name"] as? String else { return nil }
    let importedAt = (record["imported_at"] as? String)
      .flatMap { ISO8601DateFormatter().date(from: $0) } ?? Date.distantPast
    let compatDict = record["compat"] as? [String: Any] ?? [:]
    let compat = WorkflowCompatReport(
      mappedNodes: compatDict["mapped_nodes"] as? [String] ?? [],
      glueNodes: compatDict["glue_nodes"] as? [String] ?? [],
      unknownNodes: compatDict["unknown_nodes"] as? [String] ?? [],
      nodeCount: compatDict["node_count"] as? Int ?? 0,
      parses: compatDict["parses"] as? Bool ?? false,
      parseError: compatDict["parse_error"] as? String)
    return StoredWorkflow(id: id, name: name, importedAt: importedAt, graph: graph, compat: compat)
  }

  public func list() -> [StoredWorkflow] {
    let files = (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
    return files
      .filter { $0.pathExtension == "json" }
      .compactMap { get($0.deletingPathExtension().lastPathComponent) }
      .sorted { $0.importedAt > $1.importedAt }
  }

  @discardableResult
  public func delete(_ id: String) -> Bool {
    let url = fileURL(for: id)
    guard fm.fileExists(atPath: url.path) else { return false }
    return (try? fm.removeItem(at: url)) != nil
  }

  private func fileURL(for id: String) -> URL {
    // ids are UUIDs we mint — sanitize anyway so a crafted id can't escape.
    let safe = id.replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: "..", with: "_")
    return directory.appendingPathComponent("\(safe).json")
  }
}
