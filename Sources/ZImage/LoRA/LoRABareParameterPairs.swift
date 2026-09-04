// LoRABareParameterPairs.swift — low-rank pairs whose target is a BARE
// PARAMETER rather than a Linear module.
//
// `LoRAApplicator.applyDynamically` binds by walking `namedModules()` and
// wrapping each match in a `LoRALinear` / `LoRAQuantizedLinear`. That walk can
// only reach a target that IS a `Linear` or `QuantizedLinear`. Krea-2's
// modulation weights are neither: `Krea2SimpleModulation.lin` is a bare
// (2, features) `MLXArray` ADDED to the timestep vector, and
// `Krea2DoubleSharedModulation.lin` is a bare 1-D array. The checkpoint stores
// them as plain tensors (`last.modulation.lin`, `blocks.N.mod.lin`) — which is
// exactly why `LoRAPatchSession` indexes PARAMETERS, not modules, for
// `.diff` / `.diff_b` / `.set_weight` patches, and why Kroma's
// `diffusion_model.last.modulation.lin.diff` [2, 6144] has always applied.
//
// The Krea-2 turbo distills (`krea2_turbo_distill_r256`, `…_r128`) reach the
// SAME parameter, but as a low-rank pair:
// `diffusion_model.last.modulation.lin.lora_A/lora_B` — an SVD of the
// (2, 6144) delta, hence rank 2, with `lora_B @ lora_A` exactly the target's
// shape. No module answers to that key, so the strict Krea-2 apply refused all
// 531 keys over the one it could not reach:
//
//   LoRA 'krea2_turbo_distill_r256.safetensors' did not bind completely
//   — 1 key(s) matched nothing: last.modulation.lin.weight
//
// A low-rank pair on a bare parameter is arithmetically identical to a `.diff`
// patch of `(up @ down) × alpha/rank`. There is no runtime `x @ Aᵀ @ Bᵀ` to
// defer, because the parameter is never used as a matmul weight — it is added.
// So this moves such pairs out of `LoRAWeights.weights` and into
// `LoRAWeights.deltas`, where `LoRAPatchSession` applies them with an exact,
// first-write-wins snapshot and restores them on `clear()`. That is the same
// transactional guarantee the rest of the Krea-2 stack has, and the reason the
// conversion is NOT done inside `applyDynamically`: the applicator mutates
// base parameters with no restore path, which is precisely why LoKr is refused
// on this path (see ``Krea2AdapterSupport``).
//
// FAIL-CLOSED IS PRESERVED. A pair is diverted ONLY when all three hold:
//
//   1. no `Linear`/`QuantizedLinear` the applicator could bind answers to its
//      key — the module walk wins unconditionally, so every adapter that binds
//      today binds identically after this change;
//   2. a REAL parameter path answers to it; and
//   3. `up @ down` normalises to that parameter's exact shape.
//
// A key naming something the architecture does not have satisfies neither (1)
// nor (2), stays in `weights`, and the strict apply still throws
// `partialApplication` naming it. A key naming a real parameter it cannot fit
// throws `incompatibleWeights` here — refused either way, never dropped.

import Foundation
import MLX
import MLXNN

public enum LoRABareParameterPairs {

  /// Rewrite `weights` so that every low-rank pair targeting a bare parameter
  /// of `module` becomes a `.diff` delta on that parameter's real path.
  ///
  /// Returns `weights` unchanged (same instance semantics — rank, alpha,
  /// per-layer alphas and LoKr all preserved) when nothing moves, which is the
  /// case for every adapter whose keys all name Linears.
  ///
  /// - Parameters:
  ///   - weights: as produced by ``LoRAWeightLoader/loadForKrea2(from:)``.
  ///   - module: the transformer the stack is about to be applied to.
  ///   - name: adapter display name, so a refusal names the file.
  /// - Throws: ``LoRAError/incompatibleWeights(_:)`` when a pair names a real
  ///   parameter whose shape it cannot form, or when the same file carries
  ///   both a pair and a bare patch for one target.
  public static func split<T: Module>(
    _ weights: LoRAWeights, for module: T, name: String? = nil
  ) throws -> LoRAWeights {
    guard !weights.weights.isEmpty else { return weights }

    // Built with the applicator's OWN notion of a bindable module, so the two
    // can never drift apart and start disagreeing about who owns a key.
    var bindableModulePaths = Set<String>()
    for (path, child) in module.namedModules()
    where LoRAApplicator.linearDims(for: child) != nil {
      bindableModulePaths.insert(path)
    }

    var parameterIndex: [String: MLXArray] = [:]
    for (path, value) in module.parameters().flattened() { parameterIndex[path] = value }

    var pairs = weights.weights
    var deltas = weights.deltas
    let who = name.map { "LoRA '\($0)'" } ?? "LoRA"

    // Sorted for a deterministic first refusal on a file with several faults.
    for (key, pair) in weights.weights.sorted(by: { $0.key < $1.key }) {
      let bare = key.hasSuffix(".weight") ? String(key.dropLast(".weight".count)) : key

      // (1) A module the applicator can bind always wins.
      if bindableModulePaths.contains(bare) || bindableModulePaths.contains(key) { continue }

      // (2) Only a real parameter path is a candidate. `loadForKrea2` appends
      //     ".weight" to every pair key, so the bare spelling is the one that
      //     matches `last.modulation.lin`; try the literal key first anyway.
      guard let path = [key, bare].first(where: { parameterIndex[$0] != nil }),
            let target = parameterIndex[path]
      else { continue }

      // (3) The delta must fit the parameter exactly. Computed in float32:
      //     the result is baked into a stored parameter rather than evaluated
      //     per-step, so there is no reason to carry bf16 rounding into it.
      guard let normalized = LoRAApplicator.normalizedLoRAPair(
              down: pair.down, up: pair.up, targetShape: target.shape),
            let delta = LoRAApplicator.computeDelta(
              up: normalized.up.asType(.float32), down: normalized.down.asType(.float32)),
            delta.shape == target.shape
      else {
        throw LoRAError.incompatibleWeights(
          "\(who): pair '\(key)' targets bare parameter '\(path)' of shape \(target.shape), "
            + "but down=\(pair.down.shape) up=\(pair.up.shape) cannot form that delta")
      }

      guard deltas[path] == nil, deltas[bare] == nil, deltas[key] == nil else {
        throw LoRAError.incompatibleWeights(
          "\(who): '\(path)' carries both a low-rank pair and a bare patch — "
            + "refusing rather than guessing which applies first")
      }

      // Same scaling the applicator would have used on a Linear:
      // `userScale × alpha/rank × (up @ down)`. The alpha/rank half is baked
      // in here because `LoRAPatchSession` applies userScale alone, by design.
      let alphaOverRank = weights.effectiveScale(forLayer: key)
      deltas[path] = .diff(alphaOverRank == 1.0 ? delta : delta * alphaOverRank)
      pairs[key] = nil
    }

    guard pairs.count != weights.weights.count else { return weights }
    return weights.withPairsAndDeltas(weights: pairs, deltas: deltas)
  }
}
