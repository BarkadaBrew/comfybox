import Foundation
import MLX
import MLXNN

/// Applies bare-parameter LoRA patches (`.diff` / `.diff_b` / `.set_weight`)
/// to a module, transactionally, with exact restore.
///
/// Spec: specs/lora-delta-keys-design.md (rev 2). The invariants this type
/// exists to hold, each traceable to a Codex review finding:
///
/// - **Preflight before mutation** (finding 2): every delta key is resolved
///   against a flattened parameter-path index first; any miss throws
///   ``LoRAError/partialApplication(lora:unbound:)`` with nothing changed.
/// - **Parameter index, not module traversal** (finding 7): Kroma's targets
///   include bare leaf parameters (`prenorm.scale`, modulation arrays) that
///   no Linear-module walk can reach.
/// - **Detached first-write-wins snapshots, instance-scoped** (finding 3):
///   MLX parameters are references mutated in place by `Module.update`;
///   snapshots are evaluated copies, taken only on the FIRST patch of a
///   path, owned by this session (never global).
/// - **Exact packed restore for quantized targets** (finding 5): the packed
///   weight/scales/biases tuple is snapshotted and restored directly —
///   never requantized from a dequantized snapshot.
/// - **Delta scaling is `userScale × delta`** (finding 8): never alpha/rank.
///   `.set_weight` replaces and ignores userScale, and refuses to coexist
///   with pair-adapters on the same target (finding 4).
public final class LoRAPatchSession {

    private let module: Module

    /// First-write-wins detached copies of original parameter values,
    /// keyed by flattened parameter path.
    private var snapshots: [String: MLXArray] = [:]

    public init(module: Module) {
        self.module = module
    }

    public var isActive: Bool { !snapshots.isEmpty }

    // MARK: - Apply

    private struct PatchOp {
        let targetPath: String
        let patch: DeltaPatch
        /// Present when the target is the packed weight of a QuantizedLinear:
        /// dequantize → patch → requantize once, and snapshot the full tuple.
        let quantized: QuantizedLinear?
    }

    /// Dry-run: resolve every delta against the module's parameter index
    /// without mutating anything. Returns the number of resolved ops; throws
    /// ``LoRAError/partialApplication(lora:unbound:)`` listing every miss.
    /// Shape metadata only — safe on a lazily-initialized module.
    @discardableResult
    public func preflight(weights: LoRAWeights) throws -> Int {
        guard !weights.deltas.isEmpty else { return 0 }
        let paramIndex = flattenedParameters()
        var missing: [String] = []
        var resolved = 0
        for (key, patch) in weights.deltas {
            guard let path = resolveTargetPath(key, in: paramIndex) else {
                missing.append(key)
                continue
            }
            _ = try alignedPatchTensor(patch.tensor, to: paramIndex[path]!.shape, key: path)
            resolved += 1
        }
        guard missing.isEmpty else {
            throw LoRAError.partialApplication(lora: nil, unbound: missing.sorted())
        }
        return resolved
    }

    /// Preflights and applies all deltas in `weights`. Throws BEFORE any
    /// mutation if a single target fails to resolve. Returns the number of
    /// deltas applied (``LoRAApplicationReport/deltasApplied`` — WP-E6).
    @discardableResult
    public func apply(weights: LoRAWeights, scale: Float) throws -> Int {
        guard !weights.deltas.isEmpty else { return 0 }

        let paramIndex = flattenedParameters()
        let moduleIndex = flattenedModules()

        // ---- Preflight: resolve every op or throw with nothing mutated ----
        var ops: [PatchOp] = []
        var missing: [String] = []
        for (key, patch) in weights.deltas.sorted(by: { $0.key < $1.key }) {
            guard let path = resolveTargetPath(key, in: paramIndex) else {
                missing.append(key)
                continue
            }
            let parent = path.hasSuffix(".weight")
                ? String(path.dropLast(".weight".count)) : path
            let quantized = moduleIndex[parent] as? QuantizedLinear
            if case .setWeight = patch, quantized != nil {
                throw LoRAError.incompatibleWeights(
                    "set_weight on quantized target '\(path)' is not supported")
            }
            ops.append(PatchOp(
                targetPath: path,
                patch: patch,
                quantized: path.hasSuffix(".weight") ? quantized : nil))
        }
        guard missing.isEmpty else {
            throw LoRAError.partialApplication(lora: nil, unbound: missing.sorted())
        }

        // Validate shapes up front too — still pre-mutation.
        var prepared: [(op: PatchOp, tensor: MLXArray)] = []
        for op in ops {
            let current = paramIndex[op.targetPath]!
            let target = op.quantized.map { dequantizedWeight(of: $0) } ?? current
            let aligned = try alignedPatchTensor(op.patch.tensor, to: target.shape,
                                                 key: op.targetPath)
            prepared.append((op, aligned))
        }

        // ---- Commit ----
        var updates: [(String, MLXArray)] = []
        for (op, tensor) in prepared {
            if let quantized = op.quantized {
                try commitQuantized(op: op, delta: tensor, scale: scale,
                                    quantized: quantized, paramIndex: paramIndex,
                                    updates: &updates)
                continue
            }

            let current = paramIndex[op.targetPath]!
            snapshotIfFirstWrite(op.targetPath, current: current)

            let newValue: MLXArray
            switch op.patch {
            case .diff, .diffBias:
                newValue = current + tensor.asType(current.dtype) * scale
            case .setWeight:
                // Replacement — userScale deliberately ignored (ComfyUI parity).
                newValue = tensor.asType(current.dtype)
            }
            updates.append((op.targetPath, newValue))
        }

        module.update(parameters: ModuleParameters.unflattened(updates))
        eval(module.parameters())
        return ops.count
    }

    private func commitQuantized(
        op: PatchOp, delta: MLXArray, scale: Float, quantized: QuantizedLinear,
        paramIndex: [String: MLXArray], updates: inout [(String, MLXArray)]
    ) throws {
        let parent = String(op.targetPath.dropLast(".weight".count))
        // Snapshot the EXACT packed tuple, first-write-wins per path.
        snapshotIfFirstWrite(op.targetPath, current: quantized.weight)
        snapshotIfFirstWrite(parent + ".scales", current: quantized.scales)
        if let biases = quantized.biases {
            snapshotIfFirstWrite(parent + ".biases", current: biases)
        }

        let dequantized = dequantizedWeight(of: quantized)
        let patched = dequantized + delta.asType(dequantized.dtype) * scale
        let (newWeight, newScales, newBiases) = MLX.quantized(
            patched, groupSize: quantized.groupSize, bits: quantized.bits)

        updates.append((op.targetPath, newWeight))
        updates.append((parent + ".scales", newScales))
        if let newBiases { updates.append((parent + ".biases", newBiases)) }
    }

    // MARK: - Clear

    /// Restores every patched parameter to its exact original value and
    /// empties the snapshot store. Safe to call when nothing was applied.
    public func clear() {
        guard !snapshots.isEmpty else { return }
        let updates = snapshots.map { ($0.key, $0.value) }
        module.update(parameters: ModuleParameters.unflattened(updates))
        eval(module.parameters())
        snapshots.removeAll()
    }

    // MARK: - Internals

    private func snapshotIfFirstWrite(_ path: String, current: MLXArray) {
        guard snapshots[path] == nil else { return }
        // Detached, materialized copy — MLX parameters are reference types
        // mutated in place, so storing `current` itself would store an alias.
        let copy = current + MLXArray(0, dtype: current.dtype)
        eval(copy)
        snapshots[path] = copy
    }

    private func dequantizedWeight(of quantized: QuantizedLinear) -> MLXArray {
        MLX.dequantized(
            quantized.weight,
            scales: quantized.scales,
            biases: quantized.biases,
            groupSize: quantized.groupSize,
            bits: quantized.bits)
    }

    /// Resolve a delta key to a real parameter path: exact match first, then
    /// the `.weight` child (a `.diff` on "blocks.0.mod.lin" may target either
    /// a bare array or a Linear's weight, depending on the model).
    private func resolveTargetPath(_ key: String, in index: [String: MLXArray]) -> String? {
        if index[key] != nil { return key }
        if index[key + ".weight"] != nil { return key + ".weight" }
        return nil
    }

    /// mlx-chroma's defensive pattern: accept a transposed 2-D patch and
    /// auto-correct; any other shape mismatch throws.
    private func alignedPatchTensor(
        _ tensor: MLXArray, to shape: [Int], key: String
    ) throws -> MLXArray {
        if tensor.shape == shape { return tensor }
        if tensor.ndim == 2, shape.count == 2,
           tensor.shape == [shape[1], shape[0]] {
            return tensor.transposed()
        }
        throw LoRAError.incompatibleWeights(
            "patch for '\(key)' has shape \(tensor.shape), target is \(shape)")
    }

    private func flattenedParameters() -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (path, value) in module.parameters().flattened() {
            out[path] = value
        }
        return out
    }

    private func flattenedModules() -> [String: Module] {
        var out: [String: Module] = [:]
        for (path, child) in module.namedModules() {
            out[path] = child
        }
        return out
    }
}
