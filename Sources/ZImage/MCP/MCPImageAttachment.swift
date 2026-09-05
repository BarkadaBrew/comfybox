// MCPImageAttachment.swift — render results as MCP image content (comfybox#294)
//
// `generate_image` returns a file path. A vision-capable client that wants to
// SEE the render has to make a second round trip — and a remote client cannot
// read the Mac's filesystem at all.
//
// So a completed render can also come back as an MCP `image` content block
// (base64, `image/png`) alongside the existing text/path result. It is gated
// by an additive `return_image` parameter that defaults to FALSE, because the
// payload cost is real:
//
//   | render                | PNG on disk | base64 in the JSON-RPC line |
//   |-----------------------|-------------|-----------------------------|
//   | 1024x1024 turbo       | ~1.3-2.0 MB | ~1.7-2.7 MB                 |
//   | 1024x1536 Krea 2      | ~2.0-3.0 MB | ~2.7-4.0 MB                 |
//   | 2048px upscale        | ~6-12 MB    | ~8-16 MB  (over the cap)    |
//
// Every byte of that lands in the client's context. Over `defaultLimitBytes`
// the block is DROPPED and the text result says so — the caller still has the
// path, which is the durable artifact. Truncating or refusing the whole
// response would lose the render.

import Foundation

public enum MCPImageAttachment {

  /// The additive tool parameter that turns this on.
  public static let parameterName = "return_image"

  /// Cap on the BASE64-ENCODED size of one attached image (8 MB). Comfortable
  /// for a normal 1024-1536px render; refuses a 4K upscale rather than
  /// blowing up the client's context.
  public static let defaultLimitBytes = 8 * 1024 * 1024

  /// At most one image per tool result. The engine renders exactly one image
  /// per request — `POST /v1/generate` has no batch `count` — so this cap is
  /// structural. If a batch parameter is ever added, raise it here, with the
  /// payload table above updated in the same review.
  public static let maxImagesPerResult = 1

  public enum Outcome: Sendable, Equatable {
    case attached(base64: String, mimeType: String, fileBytes: Int)
    case skippedTooLarge(fileBytes: Int, limitBytes: Int)
    case unreadable(path: String)
  }

  /// Encoded length of `n` raw bytes as base64 (`ceil(n/3) * 4`). The cap is
  /// measured on THIS, not on the file size — that is what actually crosses
  /// the wire, and a file just under the cap would otherwise inflate past it.
  public static func encodedLength(ofBytes count: Int) -> Int {
    ((count + 2) / 3) * 4
  }

  /// Read a rendered file and decide whether it can ride along.
  ///
  /// The size is checked from the filesystem metadata FIRST, so an oversized
  /// render is refused without ever being read into memory (PR #367 review
  /// r1, item 2). This process sits next to LM Studio and a 30 GB model
  /// (intent.md: memory is a shared resource) — pulling in 12 MB only to
  /// throw it away is not free.
  public static func encode(
    path: String,
    limitBytes: Int = MCPImageAttachment.defaultLimitBytes,
    fileSize: (String) -> Int? = { path in
      (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? nil
    },
    load: (String) -> Data? = { FileManager.default.contents(atPath: $0) }
  ) -> Outcome {
    let resolved = (path as NSString).expandingTildeInPath
    guard !resolved.isEmpty else { return .unreadable(path: path) }

    // Pre-check: refuse on metadata alone when the size is knowable.
    if let bytes = fileSize(resolved), encodedLength(ofBytes: bytes) > limitBytes {
      return .skippedTooLarge(fileBytes: bytes, limitBytes: limitBytes)
    }

    guard let data = load(resolved) else { return .unreadable(path: path) }
    // Belt and braces: the file could have grown between stat and read, or
    // `fileSize` could have returned nil (unusual filesystem, test stub).
    guard encodedLength(ofBytes: data.count) <= limitBytes else {
      return .skippedTooLarge(fileBytes: data.count, limitBytes: limitBytes)
    }
    return .attached(
      base64: data.base64EncodedString(), mimeType: mimeType(forPath: resolved),
      fileBytes: data.count)
  }

  /// MIME type from the file extension. Unknown extensions default to PNG —
  /// what the engine writes.
  public static func mimeType(forPath path: String) -> String {
    switch (path as NSString).pathExtension.lowercased() {
    case "jpg", "jpeg": return "image/jpeg"
    case "webp": return "image/webp"
    default: return "image/png"
    }
  }

  /// Image blocks for the given output paths — capped at
  /// `maxImagesPerResult` — together with a note for every path that was
  /// asked for and could not be attached. This is the ONE place tool results
  /// get image content, so the cap is really enforced rather than merely
  /// declared (PR #367 review r1, item 4).
  public static func attachment(
    paths: [String],
    limitBytes: Int = MCPImageAttachment.defaultLimitBytes,
    maxImages: Int = MCPImageAttachment.maxImagesPerResult,
    load: (String) -> Data? = { FileManager.default.contents(atPath: $0) }
  ) -> (blocks: [MCPContentBlock], notes: [String]) {
    var blocks: [MCPContentBlock] = []
    var notes: [String] = []
    for path in paths.prefix(max(0, maxImages)) {
      let outcome = encode(path: path, limitBytes: limitBytes, load: load)
      if case .attached(let base64, let mimeType, _) = outcome {
        blocks.append(MCPContentBlock(imageBase64: base64, mimeType: mimeType))
      } else if let note = note(for: outcome) {
        notes.append(note)
      }
    }
    return (blocks, notes)
  }

  /// Convenience wrapper for callers that only want the blocks.
  public static func blocks(
    paths: [String],
    limitBytes: Int = MCPImageAttachment.defaultLimitBytes,
    maxImages: Int = MCPImageAttachment.maxImagesPerResult,
    load: (String) -> Data? = { FileManager.default.contents(atPath: $0) }
  ) -> [MCPContentBlock] {
    attachment(paths: paths, limitBytes: limitBytes, maxImages: maxImages, load: load).blocks
  }

  /// Whether this output path is an image at all. Guards the video/storyboard
  /// side of the one job model: `get_job` on a finished LTX-2 render must not
  /// try to inline a 12 MB .mp4 as image content.
  public static func isAttachableImage(path: String) -> Bool {
    ["png", "jpg", "jpeg", "webp"].contains((path as NSString).pathExtension.lowercased())
  }

  /// One-line explanation for an image that was asked for but not attached,
  /// so the omission is never silent.
  public static func note(for outcome: Outcome) -> String? {
    switch outcome {
    case .attached:
      return nil
    case .skippedTooLarge(let fileBytes, let limitBytes):
      return "return_image: image omitted — \(fileBytes) bytes encodes past the "
        + "\(limitBytes)-byte MCP payload cap. Read output_path instead."
    case .unreadable(let path):
      return "return_image: image omitted — could not read \(path) from the MCP server host."
    }
  }
}
