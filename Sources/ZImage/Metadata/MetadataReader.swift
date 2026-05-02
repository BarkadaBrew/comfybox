import Foundation

/// Reads JSON sidecar files for replay. Handles version migration.
public struct MetadataReader {

    public enum MetadataError: Error, CustomStringConvertible {
        case fileNotFound(String)
        case parseError(String, Error)
        case unsupportedVersion(Int)

        public var description: String {
            switch self {
            case .fileNotFound(let path):
                return "Metadata file not found: \(path)"
            case .parseError(let path, let err):
                return "Failed to parse \(path): \(err)"
            case .unsupportedVersion(let v):
                return "Unsupported metadata version \(v). Max supported: 1"
            }
        }
    }

    /// Read and parse a metadata sidecar. Returns the metadata with all fields populated.
    /// CLI flags passed by the user override loaded values (handled by caller).
    public static func read(from path: String) throws -> GenerationMetadata {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: path) else {
            throw MetadataError.fileNotFound(path)
        }
        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let metadata = try decoder.decode(GenerationMetadata.self, from: data)

            guard metadata.version <= 1 else {
                throw MetadataError.unsupportedVersion(metadata.version)
            }

            return metadata
        } catch let error as MetadataError {
            throw error
        } catch {
            throw MetadataError.parseError(path, error)
        }
    }

    /// Convenience: read from an image path (derives sidecar path automatically).
    public static func readForImage(at imagePath: String) throws -> GenerationMetadata {
        let sidecarPath = MetadataWriter.sidecarPath(for: imagePath)
        return try read(from: sidecarPath)
    }
}
