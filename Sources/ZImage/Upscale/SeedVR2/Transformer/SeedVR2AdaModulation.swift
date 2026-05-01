import Foundation
import MLX
import MLXNN

/// A group of six learnable modulation parameters for one modality.
///
/// Stored as individual `MLXArray` parameters so that the weight loader can
/// match key paths like `params_vid.attn_shift`, `params_vid.mlp_gate`, etc.
public final class SeedVR2AdaParams: Module {

  @ParameterInfo(key: "attn_shift") var attnShift: MLXArray
  @ParameterInfo(key: "attn_scale") var attnScale: MLXArray
  @ParameterInfo(key: "attn_gate") var attnGate: MLXArray
  @ParameterInfo(key: "mlp_shift") var mlpShift: MLXArray
  @ParameterInfo(key: "mlp_scale") var mlpScale: MLXArray
  @ParameterInfo(key: "mlp_gate") var mlpGate: MLXArray

  public init(dim: Int) {
    self._attnShift.wrappedValue = MLXArray.zeros([dim])
    self._attnScale.wrappedValue = MLXArray.ones([dim])
    self._attnGate.wrappedValue = MLXArray.zeros([dim])
    self._mlpShift.wrappedValue = MLXArray.zeros([dim])
    self._mlpScale.wrappedValue = MLXArray.ones([dim])
    self._mlpGate.wrappedValue = MLXArray.zeros([dim])
    super.init()
  }

  /// Returns the (shift, scale, gate) tuple for the requested layer.
  func params(for layer: String) -> (shift: MLXArray, scale: MLXArray, gate: MLXArray) {
    if layer == "attn" {
      return (attnShift, attnScale, attnGate)
    } else {
      return (mlpShift, mlpScale, mlpGate)
    }
  }
}

/// Adaptive layer normalization with learned bias parameters for the SeedVR2 transformer.
///
/// Each parameter set contains six learnable vectors (dim-sized):
///
///     attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate
///
/// Three configurations are supported:
///
/// - **Separate** (`sharedWeights=false, isLastLayer=false`): Independent `params_vid`
///   and `params_txt` parameter sets (blocks 0--9).
/// - **Shared** (`sharedWeights=true, isLastLayer=false`): Single `params_all` set
///   used for both video and text (blocks 10--30).
/// - **Last layer** (`sharedWeights=true, isLastLayer=true`): `params_all` but text
///   modulation returns identity (block 31).
///
/// ## Modulation Formulas
///
/// The time embedding `emb` has shape `(B, vid_dim, 2, 3)`:
/// - axis 2: `[0]=attention, [1]=mlp`
/// - axis 3: `[0]=shift, [1]=scale, [2]=gate`
///
/// **In-modulation**: `output = hidden * (scale + emb_scale) + (shift + emb_shift)`
///
/// **Out-modulation**: `output = hidden * (gate + emb_gate)`
///
/// ## Weight Key Paths
///
/// - `ada.params_vid.{attn_shift,...}` or `ada.params_all.{...}` or `ada.params_txt.{...}`
public final class SeedVR2AdaModulation: Module {

  public let sharedWeights: Bool
  public let isLastLayer: Bool

  @ModuleInfo(key: "params_all") var paramsAll: SeedVR2AdaParams?
  @ModuleInfo(key: "params_vid") var paramsVid: SeedVR2AdaParams?
  @ModuleInfo(key: "params_txt") var paramsTxt: SeedVR2AdaParams?

  /// Creates an adaptive modulation module.
  ///
  /// - Parameters:
  ///   - dim: Feature dimension (2560 for SeedVR2).
  ///   - sharedWeights: If true, a single parameter set serves both modalities.
  ///   - isLastLayer: If true, text modulation is identity (no txt params).
  public init(dim: Int, sharedWeights: Bool = false, isLastLayer: Bool = false) {
    self.sharedWeights = sharedWeights
    self.isLastLayer = isLastLayer

    if sharedWeights {
      self._paramsAll.wrappedValue = SeedVR2AdaParams(dim: dim)
    } else {
      self._paramsVid.wrappedValue = SeedVR2AdaParams(dim: dim)
      if !isLastLayer {
        self._paramsTxt.wrappedValue = SeedVR2AdaParams(dim: dim)
      }
    }

    super.init()
  }

  /// Modulates the video stream.
  public func modulateVid(
    _ hidden: MLXArray,
    emb: MLXArray,
    layer: String,
    mode: String
  ) -> MLXArray {
    let p = sharedWeights ? paramsAll! : paramsVid!
    return SeedVR2AdaModulation.applyModulation(hidden, emb: emb, params: p, layer: layer, mode: mode)
  }

  /// Modulates the text stream. Returns `hidden` unchanged if last layer.
  public func modulateTxt(
    _ hidden: MLXArray,
    emb: MLXArray,
    layer: String,
    mode: String
  ) -> MLXArray {
    if isLastLayer { return hidden }
    let p = sharedWeights ? paramsAll! : paramsTxt!
    return SeedVR2AdaModulation.applyModulation(hidden, emb: emb, params: p, layer: layer, mode: mode)
  }

  // MARK: - Private

  /// Core modulation logic.
  ///
  /// `emb` shape: `(B, dim, 2, 3)`.
  /// `layer_idx`: 0 for attn, 1 for mlp.
  /// `mod = emb[:, :, layer_idx, :]` -> `(B, dim, 3)`.
  ///
  /// In-mode: `hidden * (scale + mod[..., 1][:, None, :]) + (shift + mod[..., 0][:, None, :])`
  /// Out-mode: `hidden * (gate + mod[..., 2][:, None, :])`
  private static func applyModulation(
    _ hidden: MLXArray,
    emb: MLXArray,
    params: SeedVR2AdaParams,
    layer: String,
    mode: String
  ) -> MLXArray {
    let layerIdx = layer == "attn" ? 0 : 1
    let p = params.params(for: layer)

    // mod = emb[:, :, layerIdx, :] → (B, dim, 3)
    let mod = emb[0..., 0..., layerIdx, 0...]

    if mode == "in" {
      // shift = mod[..., 0][:, None, :] + param_shift
      // scale = mod[..., 1][:, None, :] + param_scale
      let embShift = mod[.ellipsis, 0].expandedDimensions(axis: 1)
      let embScale = mod[.ellipsis, 1].expandedDimensions(axis: 1)
      return hidden * (p.scale + embScale) + (p.shift + embShift)
    } else {
      let embGate = mod[.ellipsis, 2].expandedDimensions(axis: 1)
      return hidden * (p.gate + embGate)
    }
  }
}
