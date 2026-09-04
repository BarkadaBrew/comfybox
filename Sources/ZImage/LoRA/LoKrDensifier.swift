// LoKrDensifier.swift — full-matrix LoKr as a transactional `.diff` delta
// (comfybox#329, lifting the K-FIX-1 / Codex C1 refusal for the layers it
// can prove).
//
// Why LoKr was refused on the Krea-2 path: `LoRAApplicator.applyLoKr` is an
// IN-PLACE fuse — `weight + kron(w1, w2) × scale` for a `Linear`, and for a
// `QuantizedLinear` a dequantize → fuse → requantize that overwrites the
// packed `weight`/`scales`/`biases` tuple. `clearDynamicLoRA` has nothing to
// restore a mutated base from, so on the per-render dynamic path the adapter
// accumulated across renders while provenance reported none, and on q8 the
// original packed bytes were destroyed on first application (see
// ``Krea2AdapterSupport``).
//
// But a FULL-MATRIX LoKr layer (2-D `lokr_w1` ⊗ 2-D `lokr_w2`) has no runtime
// structure to defer: its delta is a single dense matrix,
//
//     ΔW = kron(w1, w2) · lokrAlphaScale(alpha, w2.shape)
//
// which is arithmetically a `.diff` patch on the target weight — exactly the
// mechanism ``LoRABareParameterPairs`` already routes low-rank pairs on bare
// parameters through. `LoRAPatchSession` applies `.diff`s with detached
// first-write-wins snapshots — for a quantized target the EXACT packed
// weight/scales/biases tuple (its "finding 5" invariant) — and `clear()`
// restores those exact arrays, never requantizing from a dequantized copy.
// That is precisely the transactionality the C1 refusal demanded.
//
// So this type rewrites each provable full-matrix LoKr layer into
// `deltas[key + ".weight"] = .diff(ΔW)` and drops it from `lokrWeights`.
// After a full conversion `lokrLayerCount == 0` and the C1 guard — which is
// deliberately left in place as the backstop — passes.
//
// FAIL-CLOSED IS PRESERVED (same posture as ``LoRABareParameterPairs``):
//
//   * a layer is converted ONLY when a module the applicator itself calls
//     bindable (`LoRAApplicator.linearDims`) answers to its key AND
//     kron(w1, w2) forms that module's exact (out, in) shape;
//   * a key no bindable module answers to STAYS a LoKr layer, and the C1
//     guard downstream refuses the whole file before any weight is touched;
//   * a key whose kron cannot fit its real target throws
//     ``LoRAError/incompatibleWeights(_:)`` naming both shapes;
//   * a non-2-D half (a factored or Tucker LoKr smuggled past the loader,
//     which today refuses those key spellings outright) throws
//     ``LoRAError/unsupportedAdapter(lora:format:reason:)`` — only
//     full-matrix LoKr is in scope here;
//   * a target already carrying a bare patch from the same file throws
//     rather than guessing an application order.
//
// Scaling: `lokrAlphaScale` (alpha/dim with the ai-toolkit ~1e10-sentinel
// fallback — see its doc) is baked into the stored delta, because
// `LoRAPatchSession` applies `userScale × delta` alone, by design. The
// arithmetic is `LoRAApplicator.kron2D` — the same function the in-place
// Z-Image path uses, lifted rather than copied so the two can never drift.
// Computed in float32: the delta is baked into a stored parameter, not
// evaluated per-step, so there is no reason to carry bf16 rounding into it.
//
// The Z-Image / Flux 2 in-place `applyLoKr` path is untouched.

import Foundation
import MLX
import MLXNN

public enum LoKrDensifier {

  /// Rewrite `weights` so that every full-matrix LoKr layer with a provable
  /// target in `module` becomes a `.diff` delta on that target's weight.
  ///
  /// Returns `weights` unchanged (same instance) when the file carries no
  /// LoKr at all — every non-LoKr adapter passes through untouched.
  ///
  /// - Parameters:
  ///   - weights: as produced by ``LoRAWeightLoader/loadForKrea2(from:)``
  ///     (and, upstream, ``LoRABareParameterPairs/split(_:for:name:)`` —
  ///     the two rewrites touch disjoint fields and commute).
  ///   - module: the transformer the stack is about to be applied to.
  ///   - name: adapter display name, so a refusal names the file.
  /// - Throws: ``LoRAError/unsupportedAdapter(lora:format:reason:)`` for a
  ///   non-full-matrix layer; ``LoRAError/incompatibleWeights(_:)`` when a
  ///   kron cannot form its real target's shape or the target already
  ///   carries a bare patch.
  public static func densify<T: Module>(
    _ weights: LoRAWeights, for module: T, name: String? = nil
  ) throws -> LoRAWeights {
    guard weights.hasLoKr else { return weights }

    // Built with the applicator's OWN notion of a bindable linear, so the
    // densifier and the in-place path can never disagree about a target.
    var linearDimsByPath: [String: (out: Int, in: Int)] = [:]
    for (path, child) in module.namedModules() {
      if let dims = LoRAApplicator.linearDims(for: child) {
        linearDimsByPath[path] = dims
      }
    }

    var lokr = weights.lokrWeights
    var deltas = weights.deltas
    let who = name.map { "LoRA '\($0)'" } ?? "LoRA"

    // Sorted for a deterministic first refusal on a file with several faults.
    for (key, layer) in weights.lokrWeights.sorted(by: { $0.key < $1.key }) {
      guard layer.w1.ndim == 2, layer.w2.ndim == 2 else {
        throw LoRAError.unsupportedAdapter(
          lora: name ?? key, format: "LoKr",
          reason: "layer '\(key)' is not full-matrix "
            + "(w1 \(layer.w1.shape), w2 \(layer.w2.shape)) — only 2-D "
            + "lokr_w1 ⊗ lokr_w2 can be densified into a transactional delta")
      }

      // No bindable module answers to this key: leave the layer LoKr-shaped
      // and let the C1 guard refuse the file whole (fail-closed backstop).
      guard let dims = linearDimsByPath[key] else { continue }

      let expectedOut = layer.w1.dim(0) * layer.w2.dim(0)
      let expectedIn = layer.w1.dim(1) * layer.w2.dim(1)
      guard expectedOut == dims.out, expectedIn == dims.in else {
        throw LoRAError.incompatibleWeights(
          "\(who): LoKr layer '\(key)' forms kron \(expectedOut)x\(expectedIn), "
            + "but the target weight is \(dims.out)x\(dims.in)")
      }

      let targetPath = key + ".weight"
      guard deltas[targetPath] == nil, deltas[key] == nil else {
        throw LoRAError.incompatibleWeights(
          "\(who): '\(targetPath)' carries both a LoKr layer and a bare patch "
            + "— refusing rather than guessing which applies first")
      }

      let dense = LoRAApplicator.kron2D(
        layer.w1.asType(.float32), layer.w2.asType(.float32))
      let alphaScale = LoRAApplicator.lokrAlphaScale(
        alpha: layer.alpha, w2Shape: layer.w2.shape)
      deltas[targetPath] = .diff(alphaScale == 1.0 ? dense : dense * alphaScale)
      lokr[key] = nil
    }

    guard lokr.count != weights.lokrWeights.count else { return weights }
    return weights.withLoKrAndDeltas(lokrWeights: lokr, deltas: deltas)
  }
}
