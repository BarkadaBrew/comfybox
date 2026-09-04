import Foundation

/// Verdict from `SafetensorsIntegrity.check`. `.invalid`'s `reason` is a
/// stable machine string of the form `truncated:<filename>` so a caller
/// scanning a directory of shards can report exactly which one is bad
/// without re-deriving the filename itself.
public enum SafetensorsIntegrityResult: Equatable, Sendable {
  case ok
  case invalid(reason: String)
}

/// Cheap, read-only safetensors integrity check: does the header parse, and
/// does every tensor's `data_offsets` end fall within the file's actual byte
/// count? This is the check intent.md calls for directly — "a truncated
/// safetensors file loads silently in MLX. Check data offsets against file
/// size when output quality drops" — done ahead of any render, not after one
/// comes back wrong.
///
/// Deliberately reuses `SafeTensorsReader`'s header + `data_offsets` parsing
/// (the same parser the runtime model loaders trust) instead of a second,
/// independent implementation of the safetensors header format: two parsers
/// that quietly drift apart is exactly how a "verified" file stops meaning
/// anything. `SafeTensorsReader.init` mmaps the file (lazy — no bulk read)
/// and validates the JSON header and every tensor's offsets against the
/// mapped size, which is exactly the truncation check this needs; it also
/// enforces some things this check does not strictly require (a supported
/// dtype, shape byte-count agreement), but a checkpoint that fails any of
/// those is equally unusable, so folding them into the same verdict is
/// correct, not incidental.
public enum SafetensorsIntegrity {
  public static func check(url: URL) -> SafetensorsIntegrityResult {
    do {
      _ = try SafeTensorsReader(fileURL: url)
      return .ok
    } catch {
      return .invalid(reason: "truncated:\(url.lastPathComponent)")
    }
  }
}
