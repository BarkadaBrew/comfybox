// Krea2AdapterSupport.swift — which adapter FORMATS the Krea 2 path accepts
// (K-FIX-1, Codex engine review C1).
//
// The Krea 2 LoRA stack is transactional by construction: every apply runs
// inside `Krea2Pipeline.loadLoRAs`'s `do` block, and any failure — a missing
// file, a declared relativity that contradicts the loaded base (WP-E6/AC-41),
// a strict partial bind (D9/AC-42) — rolls the WHOLE stack back to the base
// with `clearDynamicLoRA` + `patchSession.clear()`, so applied weights and
// `appliedLoRAs`/`loadedLoRAReports` can never disagree.
//
// LoKr breaks that invariant, and it breaks it silently:
//
//   * `LoRAApplicator.applyLoKr` does not install an adapter. It REPLACES the
//     module's base parameters — `weight + kron(w1, w2) * scale` for a
//     `Linear`, and for a `QuantizedLinear` a full dequantize → fuse →
//     REQUANTIZE that overwrites `weight`, `scales` and `biases`.
//   * `clearDynamicLoRA` only empties `LoRALinear` / `LoRAQuantizedLinear`
//     adapters. It has nothing to restore a mutated base from.
//
// So on the Krea 2 path a LoKr adapter could never be cleared: `loadLoRAs([])`
// reported an empty stack over a still-mutated checkpoint, re-applying the
// same file compounded it, and `setControlLoRA`'s identity re-apply compounded
// it on every control on/off toggle. On q8 the first application also destroys
// the original packed bytes, so even an exact subtraction (`removeLoKr`) could
// not restore them byte-for-byte.
//
// Ruling (ledger, "Codex engine review"): REFUSE LoKr on the Krea 2 path,
// fail-loud, before any weight is touched, until LoKr is made transactional
// under its own ticket (snapshot the exact original tensors — packed q8
// weight/scales/biases included — and restore those exact arrays on clear or
// rollback). The Z-Image / Flux 2 paths are unchanged and keep applying LoKr.
//
// No Krea 2 LoKr adapter exists in the vault today, so the refusal costs
// nothing now and closes a compounding, falsely-reported weight-state bug.
//
// comfybox#329 is that "own ticket": ``LoKrDensifier`` now converts every
// provable FULL-MATRIX LoKr layer into a dense `.diff` delta applied through
// `LoRAPatchSession` (exact packed-tuple snapshot/restore), so such files
// reach this guard with `lokrLayerCount == 0` and pass. The guard is
// deliberately unchanged: it remains the fail-closed backstop for any layer
// the densifier could not convert (no bindable target module).

import Foundation

public enum Krea2AdapterSupport {

  /// The reason carried in the typed error, stated once so the log line, the
  /// HTTP 400 body and the ticket all say the same thing.
  public static let lokrRefusalReason =
    "LoKr is not transactional on this path — it rewrites base weights "
    + "(dequantizing and requantizing quantized ones) and clearDynamicLoRA "
    + "cannot restore them, so the adapter would accumulate across renders "
    + "while provenance reported none"

  /// Refuse an adapter whose application cannot be rolled back on this path.
  ///
  /// Called from inside `Krea2Pipeline`'s transactional block, on the weights
  /// as loaded and BEFORE `applyDynamically` — the guard reads counts only, so
  /// a refused file leaves the transformer bit-identical.
  ///
  /// - Parameters:
  ///   - lokrLayerCount: `LoRAWeights.lokrLayerCount` of the loaded file.
  ///   - lora: the adapter's display name, so the refusal names the file.
  public static func checkTransactional(lokrLayerCount: Int, lora: String) throws {
    guard lokrLayerCount > 0 else { return }
    throw LoRAError.unsupportedAdapter(lora: lora, format: "LoKr", reason: lokrRefusalReason)
  }
}
