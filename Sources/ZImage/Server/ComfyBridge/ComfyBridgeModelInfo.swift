// ComfyBridgeModelInfo.swift — Model metadata for /api/etn/model_info endpoint
//
// Returns model classification data that the Krita AI Diffusion plugin uses
// to determine which architectures (workloads) are available.
//
// Delegates to ComfyBoxModelRegistry for the actual model definitions.

import Foundation

/// Provides model metadata for the `/api/etn/model_info/{folder}` endpoint.
enum ComfyBridgeModelInfo {

  /// Return model metadata for the given folder name.
  /// Delegates to `ComfyBoxModelRegistry.bridgeModelInfo(for:)`.
  static func models(for folder: String) -> [String: Any] {
    ComfyBoxModelRegistry.bridgeModelInfo(for: folder)
  }
}
