import Foundation
import MLX

/// Loads pre-computed positive text embeddings for SeedVR2 upscaling.
///
/// SeedVR2 uses fixed text embeddings rather than running a text encoder at inference
/// time. The embeddings are stored in a safetensors file (pos_emb.safetensors, ~594 KB)
/// that ships alongside the model weights.
///
/// The embedding tensor has shape (58, 5120) and represents a tokenized positive
/// prompt pre-encoded through the SeedVR2 text encoder.
///
/// ## Usage
///
///     let embeddings = try SeedVR2TextEmbeddings.loadPositive(
///         from: modelDirectory,
///         batchSize: 1
///     )
///     // embeddings.shape == [1, 58, 5120]
///
public enum SeedVR2TextEmbeddings {

  /// Expected shape of the raw embedding tensor (before batch dimension).
  public static let embeddingShape = (sequenceLength: 58, hiddenSize: 5120)

  /// Name of the safetensors file containing the pre-computed embeddings.
  public static let fileName = "pos_emb.safetensors"

  /// Key used to store the embedding tensor within the safetensors file.
  private static let tensorKey = "embedding"

  /// Errors that can occur when loading text embeddings.
  public enum LoadError: Error, CustomStringConvertible {
    /// The safetensors file was not found at the expected path.
    case fileNotFound(URL)
    /// The expected tensor key was not present in the safetensors file.
    case missingTensor(String)
    /// The loaded tensor has an unexpected shape.
    case unexpectedShape(expected: [Int], actual: [Int])

    public var description: String {
      switch self {
      case .fileNotFound(let url):
        return "SeedVR2 text embeddings file not found at \(url.path)"
      case .missingTensor(let key):
        return "SeedVR2 text embeddings missing tensor key: \(key)"
      case .unexpectedShape(let expected, let actual):
        return "SeedVR2 text embedding shape mismatch: expected \(expected), got \(actual)"
      }
    }
  }

  /// Loads the positive text embeddings from a safetensors file.
  ///
  /// - Parameters:
  ///   - directory: Directory containing the pos_emb.safetensors file. This is
  ///     typically the SeedVR2 text encoder weights directory.
  ///   - batchSize: Number of times to repeat the embeddings along the batch axis.
  ///     Default 1.
  /// - Returns: An MLXArray of shape (batchSize, 58, 5120).
  /// - Throws: A LoadError if the file is missing or the tensor has an unexpected format.
  public static func loadPositive(
    from directory: URL,
    batchSize: Int = 1
  ) throws -> MLXArray {
    let fileURL = directory.appendingPathComponent(fileName)

    guard FileManager.default.fileExists(atPath: fileURL.path) else {
      throw LoadError.fileNotFound(fileURL)
    }

    let arrays = try MLX.loadArrays(url: fileURL)

    guard var embedding = arrays[tensorKey] else {
      throw LoadError.missingTensor(tensorKey)
    }

    // The raw tensor is either (58, 5120) or (1, 58, 5120).
    // Ensure it has a batch dimension.
    if embedding.ndim == 2 {
      embedding = embedding.expandedDimensions(axis: 0)
    }

    // Validate the sequence and hidden dimensions.
    let seqLen = embedding.dim(1)
    let hiddenSize = embedding.dim(2)
    guard seqLen == embeddingShape.sequenceLength && hiddenSize == embeddingShape.hiddenSize else {
      throw LoadError.unexpectedShape(
        expected: [1, embeddingShape.sequenceLength, embeddingShape.hiddenSize],
        actual: Array(embedding.shape)
      )
    }

    // Tile for larger batch sizes.
    if batchSize > 1 {
      embedding = MLX.repeated(embedding, count: batchSize, axis: 0)
    }

    return embedding
  }
}
