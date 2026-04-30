// ComfyBridgeModelInfo.swift — Model metadata for /api/etn/model_info endpoint
//
// Returns model classification data that the Krita AI Diffusion plugin uses
// to determine which architectures (workloads) are available.

import Foundation

/// Provides model metadata for the `/api/etn/model_info/{folder}` endpoint.
enum ComfyBridgeModelInfo {

  /// Return model metadata for the given folder name.
  /// Folders: "checkpoints", "diffusion_models", "unet", "unet_gguf"
  static func models(for folder: String) -> [String: Any] {
    switch folder {
    case "checkpoints":
      // We don't use checkpoint format — Z-Image uses diffusion_models.
      return [:]

    case "diffusion_models", "unet":
      return [
        "z-image-turbo-bf16": [
          "base_model": "zimage",
          "type": "eps",
          "is_inpaint": false,
          "quant": "none",
        ] as [String: Any],
        "z-image-turbo-q8": [
          "base_model": "zimage",
          "type": "eps",
          "is_inpaint": false,
          "quant": "q8",
        ] as [String: Any],
        "z-image-turbo-q4": [
          "base_model": "zimage",
          "type": "eps",
          "is_inpaint": false,
          "quant": "q4",
        ] as [String: Any],
      ]

    case "unet_gguf":
      // No GGUF models — feature is disabled in our bridge.
      return [:]

    default:
      return [:]
    }
  }
}
