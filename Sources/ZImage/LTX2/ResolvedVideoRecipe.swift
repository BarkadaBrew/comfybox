// ResolvedVideoRecipe.swift — immutable identity for one accepted LTX render.

import CryptoKit
import Foundation

/// The render-shaping values frozen at admission time. Output paths, caller
/// identity, and runtime progress are intentionally excluded: they do not
/// change pixels. Encoding uses sorted keys, making `hash` stable for the same
/// recipe and sensitive to any prompt/model/LoRA/timing/config change.
public struct ResolvedVideoRecipe: Codable, Sendable, Equatable {
  public struct Parameter: Codable, Sendable, Equatable {
    public let name: String
    public let value: String
  }

  public struct LoRA: Codable, Sendable, Equatable {
    public let path: String
    public let scale: Float
  }

  public let version: Int
  public let model: String
  public let prompt: String
  public let negativePrompt: String?
  public let initImagePath: String?
  public let width: Int
  public let height: Int
  public let frames: Int
  public let steps: Int
  public let seed: UInt64
  public let strength: Float
  public let imgCompression: Int?
  public let guidance: Float?
  public let identityAnchorStrength: Float
  public let identityReAnchorInterval: Int
  public let extendToSeconds: Float
  public let fps: Int
  public let audio: Bool
  public let twoStageRequested: Bool?
  public let loras: [LoRA]
  public let beatSchedule: [BeatSegment]?
  public let parameters: [Parameter]

  /// Full lowercase SHA-256 over the canonical recipe JSON. Encoding errors
  /// propagate instead of collapsing every invalid recipe onto SHA256(empty).
  public func fingerprint() throws -> String {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.outputFormatting = [.sortedKeys]
    let data = try encoder.encode(self)
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  public static func build(
    request: LTX2VideoRequest, transformerFile: String
  ) throws -> ResolvedVideoRecipe {
    guard let snapshot = request.resolvedConfigSnapshot else {
      throw ResolvedVideoRecipeError.missingConfigSnapshot
    }
    return ResolvedVideoRecipe(
      version: 2,
      model: transformerFile,
      prompt: request.prompt,
      negativePrompt: request.negativePrompt,
      initImagePath: request.initImagePath,
      width: request.width,
      height: request.height,
      frames: request.framesPerChunk,
      steps: request.steps,
      seed: request.seed,
      strength: request.strength,
      imgCompression: request.imgCompression,
      guidance: request.guidance,
      identityAnchorStrength: request.identityAnchorStrength,
      identityReAnchorInterval: request.identityReAnchorInterval,
      extendToSeconds: request.extendToSeconds,
      fps: request.fps,
      audio: request.audio,
      twoStageRequested: request.twoStageRequested,
      loras: request.effectiveLoRAs.map { LoRA(path: $0.path, scale: $0.scale) },
      beatSchedule: request.beatSchedule,
      parameters: snapshot.params
        .map { Parameter(name: $0.name, value: $0.value) }
        .sorted { $0.name < $1.name })
  }
}

public enum ResolvedVideoRecipeError: Error, LocalizedError {
  case missingConfigSnapshot

  public var errorDescription: String? {
    switch self {
    case .missingConfigSnapshot:
      return "LTX-2 resolved recipe is missing its admission-time config snapshot"
    }
  }
}
