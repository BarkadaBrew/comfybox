// LoRALibraryEntry.swift — Codable data model for LoRA library entries
//
// Part of the LoRA Library Manager (#73). Each entry represents a single
// LoRA adapter file in the library with its metadata, compatibility info,
// and user-configurable fields.

import Foundation

// MARK: - LoRA Format

/// The adapter format detected from safetensors key patterns.
public enum LoRAFormat: String, Codable, Sendable {
  case lora    // Standard LoRA (lora_down/lora_up or lora_A/lora_B)
  case lokr    // Kronecker product (lokr_w1/lokr_w2)
  case slider  // Low key-count slider LoRA
}

// MARK: - Library Entry

/// A single LoRA entry in the library index.
///
/// Populated by `LoRAScanner.analyze()` for auto-detected fields and
/// manually editable via `LoRALibrary.update()` for user fields.
public struct LoRALibraryEntry: Codable, Sendable, Identifiable {

  // MARK: Identity

  /// Unique identifier, slugified from filename. Must be unique within the library.
  public var id: String

  /// Bare filename (e.g. "zit_fdpo_v1.safetensors").
  public var filename: String

  /// Path relative to the library root (e.g. "flow-dpo/zit_fdpo_v1.safetensors").
  public var relativePath: String

  // MARK: File Info

  /// File size in bytes.
  public var sizeBytes: UInt64

  /// SHA-256 hash of the file contents. Computed lazily on first scan.
  public var sha256: String?

  // MARK: Auto-Detected

  /// Model compatibility tags (e.g. ["z-image"], ["klein-9b"]).
  public var modelCompatibility: [String]

  /// Adapter format: lora, lokr, or slider.
  public var format: LoRAFormat

  /// LoRA rank (from ss_network_dim metadata or tensor shape inference).
  public var rank: Int

  /// LoRA alpha (from ss_network_alpha metadata). Nil if not specified.
  public var alpha: Float?

  /// Total tensor key count in the safetensors file.
  public var keyCount: Int

  /// Which layer types are targeted (e.g. ["attention", "feed_forward", "adaLN_modulation"]).
  public var layerTargets: [String]

  // MARK: User-Configurable

  /// Activation/trigger words needed in the prompt.
  public var triggerwords: [String]

  /// Recommended default scale for this LoRA.
  public var recommendedScale: Float

  /// Useful scale range as [min, max] for UI sliders.
  public var scaleRange: [Float]

  /// Freeform tags for search and filtering.
  public var tags: [String]

  /// Category derived from parent directory name.
  public var category: String

  /// Human-readable description or notes.
  public var notes: String

  /// Source URL (CivitAI, HuggingFace, etc.).
  public var sourceURL: String?

  /// CivitAI model ID for update checking.
  public var civitaiModelId: Int?

  // MARK: Status

  /// ISO 8601 date string when the entry was first added to the library.
  public var dateAdded: String

  /// Whether this LoRA is quarantined (excluded from active use).
  public var quarantined: Bool

  /// Reason for quarantine.
  public var quarantineReason: String?

  // MARK: Raw Metadata

  /// Raw metadata from the safetensors __metadata__ header.
  public var safetensorsMetadata: [String: String]?

  // MARK: Computed Properties

  /// Display name: derived from filename without extension, with common prefixes cleaned.
  public var displayName: String {
    var name = (filename as NSString).deletingPathExtension
    // Strip common suffixes like _v1, -v2, etc. for cleaner display
    name = name.replacingOccurrences(of: "_", with: " ")
    return name
  }

  /// Human-readable file size (e.g. "162 MB", "1.29 GB").
  public var sizeFormatted: String {
    let gb = Double(sizeBytes) / 1_073_741_824.0
    if gb >= 1.0 {
      return String(format: "%.2f GB", gb)
    }
    let mb = Double(sizeBytes) / 1_048_576.0
    return String(format: "%.0f MB", mb)
  }

  /// The primary compatibility string (first entry), or "unknown".
  public var primaryCompatibility: String {
    modelCompatibility.first ?? "unknown"
  }

  // MARK: Coding Keys

  enum CodingKeys: String, CodingKey {
    case id, filename
    case relativePath = "relative_path"
    case sizeBytes = "size_bytes"
    case sha256
    case modelCompatibility = "model_compatibility"
    case format, rank, alpha
    case keyCount = "key_count"
    case layerTargets = "layer_targets"
    case triggerwords
    case recommendedScale = "recommended_scale"
    case scaleRange = "scale_range"
    case tags, category, notes
    case sourceURL = "source_url"
    case civitaiModelId = "civitai_model_id"
    case dateAdded = "date_added"
    case quarantined
    case quarantineReason = "quarantine_reason"
    case safetensorsMetadata = "safetensors_metadata"
  }
}

// MARK: - Library Index

/// Top-level structure for library.json.
public struct LoRALibraryIndex: Codable, Sendable {
  public var version: Int
  public var updatedAt: String
  public var entries: [LoRALibraryEntry]

  enum CodingKeys: String, CodingKey {
    case version
    case updatedAt = "updated_at"
    case entries
  }

  public init(version: Int = 1, updatedAt: String = "", entries: [LoRALibraryEntry] = []) {
    self.version = version
    self.updatedAt = updatedAt
    self.entries = entries
  }
}

// MARK: - Entry Patch

/// Partial update for a library entry. Only non-nil fields are applied.
public struct LoRAEntryPatch: Sendable {
  public var triggerwords: [String]?
  public var recommendedScale: Float?
  public var scaleRange: [Float]?
  public var tags: [String]?
  public var notes: String?
  public var sourceURL: String?
  public var civitaiModelId: Int?

  public init(
    triggerwords: [String]? = nil,
    recommendedScale: Float? = nil,
    scaleRange: [Float]? = nil,
    tags: [String]? = nil,
    notes: String? = nil,
    sourceURL: String? = nil,
    civitaiModelId: Int? = nil
  ) {
    self.triggerwords = triggerwords
    self.recommendedScale = recommendedScale
    self.scaleRange = scaleRange
    self.tags = tags
    self.notes = notes
    self.sourceURL = sourceURL
    self.civitaiModelId = civitaiModelId
  }
}
