import Foundation

/// Writes JSON sidecar files alongside generated images.
/// Path convention: image.png → image.json (same directory, same basename).
public struct MetadataWriter {

    /// Write metadata sidecar. Returns the written path on success.
    /// Fails silently with a warning log if write fails (generation should not abort).
    @discardableResult
    public static func write(_ metadata: GenerationMetadata, for imagePath: String) -> String? {
        let jsonPath = sidecarPath(for: imagePath)
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(metadata)
            try data.write(to: URL(fileURLWithPath: jsonPath))
            return jsonPath
        } catch {
            fputs("[metadata] WARNING: Failed to write sidecar at \(jsonPath): \(error)\n", stderr)
            return nil
        }
    }

    /// Derive sidecar path from image path.
    /// /path/to/image.png → /path/to/image.json
    public static func sidecarPath(for imagePath: String) -> String {
        let url = URL(fileURLWithPath: imagePath)
        return url.deletingPathExtension().appendingPathExtension("json").path
    }
}
