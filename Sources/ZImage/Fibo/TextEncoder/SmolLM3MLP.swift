// SmolLM3MLP.swift — SwiGLU MLP for SmolLM3-3B text encoder
// Ported from mflux: smol_lm3_3b_mlp.py

import MLX
import MLXNN

/// SwiGLU MLP block for SmolLM3-3B.
///
/// Implements the gated linear unit with SiLU activation:
/// `output = down_proj(SiLU(gate_proj(x)) * up_proj(x))`
///
/// All projections are bias-free, matching SmolLM3 config (mlp_bias = false).
///
/// Weight key mapping (safetensors -> model):
/// - `layers.N.mlp.gate_proj.weight`
/// - `layers.N.mlp.up_proj.weight`
/// - `layers.N.mlp.down_proj.weight`
public final class SmolLM3MLP: Module {
  @ModuleInfo(key: "gate_proj") var gateProj: Linear
  @ModuleInfo(key: "up_proj") var upProj: Linear
  @ModuleInfo(key: "down_proj") var downProj: Linear

  public init(hiddenSize: Int, intermediateSize: Int) {
    self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
    self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
    self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    downProj(silu(gateProj(x)) * upProj(x))
  }
}
