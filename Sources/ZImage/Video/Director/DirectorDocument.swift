// DirectorDocument.swift — `.cbdirector` project files.
//
// A `.cbdirector` file is the wire timeline JSON, byte for byte: snake_case,
// sorted keys, pretty-printed (DirectorJSON.encoder()). Assets are referenced
// by path; `image_base64` is written only when the caller opts into embedding
// (the desktop's "Embed assets" save accessory), and even then `image_path`
// is kept so a portable file still points at its source.

import Foundation

public enum DirectorDocument {

  public static let fileExtension = "cbdirector"

  /// Read and decode. I/O and parse failures surface as
  /// `DirectorError.fileUnreadable`; a newer `version` rethrows
  /// `DirectorError.unsupportedVersion`; a missing `version` reads as 1.
  public static func read(from url: URL) throws -> DirectorTimeline {
    let data: Data
    do {
      data = try Data(contentsOf: url)
    } catch {
      throw DirectorError.fileUnreadable("\(url.path): \(error.localizedDescription)")
    }
    do {
      return try DirectorJSON.decoder().decode(DirectorTimeline.self, from: data)
    } catch let error as DirectorError {
      throw error
    } catch {
      throw DirectorError.fileUnreadable("\(url.path): \(error)")
    }
  }

  /// Encode for disk. A keyframe whose `image_path` is readable here gets the
  /// file's bytes embedded when `embedAssets` is true and no payload when it
  /// is false. A keyframe whose path is NOT readable here (a portable file
  /// opened on another machine) keeps whatever `image_base64` it already
  /// carries — that payload is the only copy of the image, so re-saving must
  /// never drop it; with no payload it is written path-only rather than
  /// failing the save.
  public static func data(_ timeline: DirectorTimeline, embedAssets: Bool = false) throws -> Data {
    var doc = timeline
    for i in doc.keyframes.indices {
      let path = doc.keyframes[i].imagePath ?? ""
      let bytes = path.isEmpty ? nil : (try? Data(contentsOf: URL(fileURLWithPath: path)))
      guard let bytes, !bytes.isEmpty else { continue }  // unreadable: keep the payload
      doc.keyframes[i].imageBase64 = embedAssets ? bytes.base64EncodedString() : nil
    }
    return try DirectorJSON.encoder().encode(doc)
  }

  public static func write(_ timeline: DirectorTimeline, to url: URL, embedAssets: Bool = false) throws {
    let bytes = try data(timeline, embedAssets: embedAssets)
    do {
      try bytes.write(to: url, options: .atomic)
    } catch {
      throw DirectorError.fileUnreadable("\(url.path): \(error.localizedDescription)")
    }
  }
}
