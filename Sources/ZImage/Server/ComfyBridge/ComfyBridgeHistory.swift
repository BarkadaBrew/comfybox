// ComfyBridgeHistory.swift — ComfyUI-compatible execution history

import Foundation

/// Thread-safe ring buffer of completed ComfyUI bridge executions.
final class ComfyBridgeHistory {
  private let capacity: Int
  private let lock = NSLock()
  private var entries: [ComfyBridgeHistoryEntry] = []
  private var index: [String: ComfyBridgeHistoryEntry] = [:]

  init(capacity: Int = 200) {
    self.capacity = max(1, capacity)
  }

  func recordGeneration(request: ComfyBridgeGenerateRequest, imageId: String, durationMs: Int) {
    let entry = ComfyBridgeHistoryEntry(
      promptId: request.promptId,
      clientId: request.clientId,
      outputNodeId: request.outputNodeId,
      images: [ComfyBridgeHistoryImage(id: imageId)],
      prompt: request.prompt,
      negativePrompt: request.negativePrompt,
      width: request.width,
      height: request.height,
      steps: request.steps,
      durationMs: durationMs,
      kind: "generate",
      completedAt: Date()
    )
    record(entry)
  }

  func recordUpscale(request: ComfyBridgeUpscaleRequest, imageId: String, durationMs: Int) {
    let entry = ComfyBridgeHistoryEntry(
      promptId: request.promptId,
      clientId: request.clientId,
      outputNodeId: request.outputNodeId,
      images: [ComfyBridgeHistoryImage(id: imageId)],
      prompt: nil,
      negativePrompt: nil,
      width: nil,
      height: nil,
      steps: nil,
      durationMs: durationMs,
      kind: "upscale",
      completedAt: Date()
    )
    record(entry)
  }

  func allJSON() -> [String: Any] {
    lock.lock()
    let snapshot = entries
    lock.unlock()

    var result: [String: Any] = [:]
    for entry in snapshot {
      result[entry.promptId] = entry.jsonObject()
    }
    return result
  }

  func json(for promptId: String) -> [String: Any]? {
    lock.lock()
    let entry = index[promptId]
    lock.unlock()

    guard let entry else { return nil }
    return [promptId: entry.jsonObject()]
  }

  private func record(_ entry: ComfyBridgeHistoryEntry) {
    lock.lock()
    if index[entry.promptId] == nil {
      entries.append(entry)
    } else if let existingIndex = entries.firstIndex(where: { $0.promptId == entry.promptId }) {
      entries[existingIndex] = entry
    }
    index[entry.promptId] = entry

    while entries.count > capacity {
      let removed = entries.removeFirst()
      index.removeValue(forKey: removed.promptId)
    }
    lock.unlock()
  }
}

private struct ComfyBridgeHistoryImage {
  let id: String

  var jsonObject: [String: Any] {
    [
      "source": "http",
      "id": id,
      "filename": "\(id).png",
      "subfolder": "",
      "type": "output"
    ]
  }
}

private struct ComfyBridgeHistoryEntry {
  let promptId: String
  let clientId: String
  let outputNodeId: String
  let images: [ComfyBridgeHistoryImage]
  let prompt: String?
  let negativePrompt: String?
  let width: Int?
  let height: Int?
  let steps: Int?
  let durationMs: Int
  let kind: String
  let completedAt: Date

  func jsonObject() -> [String: Any] {
    var meta: [String: Any] = [
      "kind": kind,
      "duration_ms": durationMs,
      "completed_at": completedAt.timeIntervalSince1970
    ]
    if let prompt { meta["prompt"] = prompt }
    if let negativePrompt { meta["negative_prompt"] = negativePrompt }
    if let width { meta["width"] = width }
    if let height { meta["height"] = height }
    if let steps { meta["steps"] = steps }

    return [
      "prompt": [
        0,
        promptId,
        [:] as [String: Any],
        ["client_id": clientId],
        ["outputs": [outputNodeId]]
      ] as [Any],
      "outputs": [
        outputNodeId: [
          "images": images.map { $0.jsonObject }
        ]
      ],
      "status": [
        "status_str": "success",
        "completed": true,
        "messages": [] as [Any]
      ],
      "meta": meta
    ]
  }
}
