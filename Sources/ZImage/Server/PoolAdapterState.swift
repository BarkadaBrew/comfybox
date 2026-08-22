// PoolAdapterState.swift — the LoRA stack is part of the pool entry, not
// coordinator state beside it (K-FIX-1, Codex engine review I1).
//
// `WarmServerCoordinator.activeLoRAs` is the stack the engine believes is
// applied, and `/health.loras` publishes it. Two paths let that belief drift
// from the resident pipeline:
//
//  1. `poolLoad` passes `activeLoRAs` to `ModelPool.load(initialLoRAs:)`, but
//     only the flux1 branch forwarded them (into `ZImagePipeline.prepare`).
//     The krea2 branch constructed the pipeline and returned — so a Raw↔Turbo
//     handoff, which is a load + activate, produced a BARE checkpoint while
//     health still named both adapters. The next request carrying no per-job
//     `loras` rendered without them, invisibly.
//  2. `poolActivate` replaced the resident pipeline without reconciling
//     `activeLoRAs` with that pipeline's `loadedLoRAConfigs`. Activating a
//     cached entry could revive ITS adapters, or none, under whatever stack
//     the coordinator happened to be advertising.
//
// The two halves of the fix, as pure functions over a host protocol so they
// are testable without a checkpoint:
//
//   * `applyInitial` — apply the stack BEFORE the pipeline becomes
//     activatable. A relativity refusal (WP-E6/AC-41) or a strict partial
//     bind (D9/AC-42) FAILS the handoff; it never degrades to the bare base,
//     because a silently bare base is the bug being fixed.
//   * `reconciled` — after activation, `activeLoRAs` (and therefore
//     `/health.loras`) is the ACTIVATED pipeline's read-back, never the
//     outgoing model's array.

import Foundation

/// The pipeline-side surface the pool and the coordinator need to keep the
/// published stack honest. `Krea2Pipeline` is the only conformer.
protocol Krea2AdapterHost: AnyObject {
  /// The stack the pipeline actually holds — non-empty only after a
  /// successful transactional apply.
  var loadedLoRAConfigs: [LoRAConfiguration] { get }
  /// Apply (or, with `[]`, clear) the stack, transactionally.
  func loadLoRAs(_ configs: [LoRAConfiguration]) async throws
}

extension Krea2Pipeline: Krea2AdapterHost {}

enum PoolAdapterState {

  /// Apply a freshly-loaded Krea 2 pipeline's initial stack.
  ///
  /// Called from `ModelPool.loadPipeline` before the entry exists, so a
  /// throw means the LOAD failed and no half-configured pipeline is ever
  /// admitted to the pool. An empty stack is a no-op — the fresh pipeline is
  /// not touched at all, which keeps the default path byte-identical.
  static func applyInitial(_ loras: [LoRAConfiguration], to host: Krea2AdapterHost) async throws {
    guard !loras.isEmpty else { return }
    try await host.loadLoRAs(loras)
  }

  /// The stack `activeLoRAs` (and `/health.loras`) must become once `activated`
  /// is the resident pipeline.
  ///
  /// For Krea 2 this is the pipeline's own read-back — including the empty
  /// case, which is a FACT (an adapter-free request really will render the
  /// bare checkpoint) and not a reason to keep publishing the previous
  /// model's list. Other families are unchanged: flux1 applies `initialLoRAs`
  /// inside `ZImagePipeline.prepare(loras:)`, so the coordinator's array is
  /// already the read-back there.
  static func reconciled(
    family: WarmModelFamily, activated: Krea2AdapterHost?, coordinator: [LoRAConfiguration]
  ) -> [LoRAConfiguration] {
    guard family == .krea2 else { return coordinator }
    return activated?.loadedLoRAConfigs ?? []
  }
}
