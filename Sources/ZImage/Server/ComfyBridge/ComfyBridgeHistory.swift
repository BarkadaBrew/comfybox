// ComfyBridgeHistory.swift — ComfyUI-compatible execution history

import Foundation

/// Thread-safe ring buffer of the last completed ComfyUI bridge executions.
final class ComfyBridgeHistory {
  private let capacity: Int
  private let lock = NSLock()
  private var entries: [ComfyBridgeHistoryEntry] = []
  private var index: [String: ComfyBridgeHistoryEntry] = [:]

  init(capacity: Int = 200) {
    self.capacity = max(1, capacity)
  }

  func record(
    promptId: String,
    clientId: String,
    timestamp: TimeInterval = Date().timeIntervalSince1970,
    outputNodeId: String,
    imageFilename: String,
    success: Bool,
    errorMessage: String? = nil
  ) {
    let entry = ComfyBridgeHistoryEntry(
      promptId: promptId,
      clientId: clientId,
      timestamp: timestamp,
      outputNodeId: outputNodeId,
      imageFilename: imageFilename,
      success: success,
      errorMessage: errorMessage
    )

    lock.lock()
    if let existingIndex = entries.firstIndex(where: { $0.promptId == promptId }) {
      entries[existingIndex] = entry
    } else {
      entries.append(entry)
    }
    index[promptId] = entry

    while entries.count > capacity {
      let removed = entries.removeFirst()
      index.removeValue(forKey: removed.promptId)
    }
    lock.unlock()
  }

  func recordGeneration(request: ComfyBridgeGenerateRequest, imageId: String, durationMs: Int) {
    record(
      promptId: request.promptId,
      clientId: request.clientId,
      outputNodeId: request.outputNodeId,
      imageFilename: "\(imageId).png",
      success: true
    )
  }

  func recordUpscale(request: ComfyBridgeUpscaleRequest, imageId: String, durationMs: Int) {
    record(
      promptId: request.promptId,
      clientId: request.clientId,
      outputNodeId: request.outputNodeId,
      imageFilename: "\(imageId).png",
      success: true
    )
  }

  func toJSON(maxItems: Int = 200, offset: Int = 0) -> [String: Any] {
    lock.lock()
    let snapshot = Array(entries.reversed())
    lock.unlock()

    let clampedOffset = max(0, offset)
    let clampedLimit = max(0, min(maxItems, capacity))
    let page = snapshot.dropFirst(clampedOffset).prefix(clampedLimit)

    var result: [String: Any] = [:]
    for entry in page {
      result[entry.promptId] = entry.jsonObject()
    }
    return result
  }

  func entry(for promptId: String) -> [String: Any]? {
    lock.lock()
    let entry = index[promptId]
    lock.unlock()
    return entry?.jsonObject()
  }
}

private struct ComfyBridgeHistoryEntry {
  let promptId: String
  let clientId: String
  let timestamp: TimeInterval
  let outputNodeId: String
  let imageFilename: String
  let success: Bool
  let errorMessage: String?

  func jsonObject() -> [String: Any] {
    [
      "prompt": [
        0,
        promptId,
        [:] as [String: Any],
        [:] as [String: Any],
        [] as [Any],
      ] as [Any],
      "outputs": outputsObject(),
      "status": statusObject(),
    ]
  }

  private func outputsObject() -> [String: Any] {
    guard success else { return [:] }
    return [
      outputNodeId: [
        "images": [
          [
            "filename": imageFilename,
            "subfolder": "",
            "type": "output",
          ] as [String: Any],
        ]
      ]
    ]
  }

  private func statusObject() -> [String: Any] {
    var messages: [Any] = []
    if let errorMessage, !errorMessage.isEmpty {
      messages.append([
        "type": "execution_error",
        "message": errorMessage,
      ] as [String: Any])
    }
    return [
      "status_str": success ? "success" : "error",
      "completed": true,
      "messages": messages,
    ]
  }
}
