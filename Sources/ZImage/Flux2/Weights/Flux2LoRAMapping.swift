// Flux2LoRAMapping.swift — LoRA weight mapping for Flux 2 Klein
//
// Deferred to post-integration PR. The full Flux 2 LoRA mapping
// (from mflux flux2_lora_mapping.py, ~915 lines) covers:
//   - Global targets (x_embedder, context_embedder)
//   - Double-stream block targets (attn Q/K/V/out, FF in/out, context FF)
//   - Single-stream block targets (fused QKV+MLP proj, out)
//   - BFL/ComfyUI-style naming conventions (qkv split patterns)
//
// This will be implemented when LoRA support is integrated into
// the Flux 2 pipeline.

import Foundation

/// Placeholder for Flux 2 Klein LoRA weight mapping.
///
/// The full implementation maps LoRA adapter weights (lora_A/lora_B pairs)
/// from various naming conventions (diffusers, BFL, ComfyUI, kohya) to
/// the Flux2Transformer module paths.
public enum Flux2LoRAMapping {
  // Deferred to post-integration PR.
  // See mflux flux2_lora_mapping.py for the full mapping specification.
}
