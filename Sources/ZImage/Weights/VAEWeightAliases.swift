import Foundation
import MLX

enum ZImageVAEWeightAliases {
  static func normalized(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var normalized = weights

    addAlias(
      from: "encoder.conv_in.conv2d.weight",
      to: "encoder.conv_in.weight",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_in.conv2d.bias",
      to: "encoder.conv_in.bias",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_out.conv2d.weight",
      to: "encoder.conv_out.weight",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_out.conv2d.bias",
      to: "encoder.conv_out.bias",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_in.conv.weight",
      to: "encoder.conv_in.weight",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_in.conv.bias",
      to: "encoder.conv_in.bias",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_out.conv.weight",
      to: "encoder.conv_out.weight",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_out.conv.bias",
      to: "encoder.conv_out.bias",
      in: &normalized
    )

    addAlias(
      from: "encoder.conv_norm_out.norm.weight",
      to: "encoder.conv_norm_out.weight",
      in: &normalized
    )
    addAlias(
      from: "encoder.conv_norm_out.norm.bias",
      to: "encoder.conv_norm_out.bias",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_in.conv.weight",
      to: "decoder.conv_in.weight",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_in.conv.bias",
      to: "decoder.conv_in.bias",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_out.conv.weight",
      to: "decoder.conv_out.weight",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_out.conv.bias",
      to: "decoder.conv_out.bias",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_in.conv2d.weight",
      to: "decoder.conv_in.weight",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_in.conv2d.bias",
      to: "decoder.conv_in.bias",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_out.conv2d.weight",
      to: "decoder.conv_out.weight",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_out.conv2d.bias",
      to: "decoder.conv_out.bias",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_norm_out.norm.weight",
      to: "decoder.conv_norm_out.weight",
      in: &normalized
    )
    addAlias(
      from: "decoder.conv_norm_out.norm.bias",
      to: "decoder.conv_norm_out.bias",
      in: &normalized
    )

    // Fix: Handle ambiguous encoder.conv_in.weight [128, 3, 3, 3] layout.
    //
    // The shape is identical in both OIHW (PyTorch) and OHWI (MLX) conventions
    // because in_channels(3) == kernel_size(3). alignTensorShape skips these
    // because shapes already match, but the data layout may be wrong.
    //
    // Detection: check encoder.conv_out.weight which has asymmetric channels:
    //   OIHW (PyTorch/standard): [32, 512, 3, 3] — dim[1] > dim[3]
    //   OHWI (MLX/pre-converted): [32, 3, 3, 512] — dim[1] < dim[3]
    //
    // Only transpose conv_in for OIHW models; pre-converted models are already correct.
    if let w = normalized["encoder.conv_in.weight"], w.ndim == 4 {
      let needsTranspose: Bool
      if let convOut = normalized["encoder.conv_out.weight"], convOut.ndim == 4 {
        // Asymmetric shape reveals layout: OIHW has large dim[1], OHWI has large dim[3]
        needsTranspose = convOut.dim(1) > convOut.dim(3)
      } else {
        // Fallback: assume OIHW (standard HuggingFace format)
        needsTranspose = true
      }

      if needsTranspose {
        normalized["encoder.conv_in.weight"] = w.transposed(0, 2, 3, 1)
      }
    }

    return normalized
  }

  private static func addAlias(from sourceKey: String, to aliasKey: String, in weights: inout [String: MLXArray]) {
    guard weights[aliasKey] == nil, let value = weights[sourceKey] else { return }
    weights[aliasKey] = value
  }
}
