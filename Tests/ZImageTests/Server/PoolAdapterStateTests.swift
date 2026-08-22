// PoolAdapterStateTests.swift — K-FIX-1 / Codex engine review I1.
//
// The adapter stack was coordinator state that the pool did not participate
// in, so a Krea 2 model handoff could lose it while `/health` kept
// advertising it:
//
//   * `poolLoad` passed `activeLoRAs` as `initialLoRAs`, but `ModelPool`'s
//     krea2 branch built the pipeline and RETURNED — only the flux1 branch
//     forwarded them (`pipeline.prepare(loras:)`). A Raw↔Turbo handoff
//     therefore produced a BARE checkpoint.
//   * `poolActivate` swapped the resident pipeline without reconciling
//     `activeLoRAs` with that pipeline's own `loadedLoRAConfigs`, so
//     activating a cached pipeline could revive its old adapters (or lose
//     them) while the coordinator advertised a different stack.
//   * `/health.loras` published the coordinator's array either way, so the
//     next request with no per-job `loras` rendered a stack nobody could see
//     was gone.
//
// Adapter state is now part of the pool entry: `initialLoRAs` are applied
// before a Krea 2 pipeline is returned (a relativity or bind failure FAILS
// the handoff, it does not degrade), and `activeLoRAs` / `/health.loras`
// derive from the ACTIVATED pipeline's read-back. Both halves are pure
// functions over a host protocol so they are testable with no checkpoint.

import Foundation
import XCTest

@testable import ZImage

private final class FakeKrea2AdapterHost: Krea2AdapterHost, @unchecked Sendable {
  private(set) var applied: [LoRAConfiguration] = []
  private(set) var callCount = 0
  var failWith: Error?

  init(resident: [LoRAConfiguration] = []) { self.applied = resident }

  var loadedLoRAConfigs: [LoRAConfiguration] { applied }

  func loadLoRAs(_ configs: [LoRAConfiguration]) async throws {
    callCount += 1
    if let failWith {
      // The pipeline's own transactional posture: a failed stack rolls back
      // to the bare base, it does not keep half of it.
      applied = []
      throw failWith
    }
    applied = configs
  }
}

final class PoolAdapterStateTests: XCTestCase {

  private func lora(_ name: String, scale: Float = 0.6, requires: Krea2Variant? = nil)
    -> LoRAConfiguration
  {
    LoRAConfiguration.local("/vault/\(name).safetensors", scale: scale, requiresBase: requires)
  }

  // MARK: - The handoff applies the stack before the pipeline is usable

  /// Raw↔Turbo with a live stack: the incoming pipeline carries the SAME
  /// adapters the outgoing one did, applied before it can be activated.
  func testHandoffAppliesTheStackToTheIncomingPipeline() async throws {
    let stack = [lora("krea2_turbo_lora_rank_64_bf16"), lora("kroma-v0.2-base-lora-rank-384-fro-0985", scale: 0.3)]
    let incoming = FakeKrea2AdapterHost()

    try await PoolAdapterState.applyInitial(stack, to: incoming)

    XCTAssertEqual(incoming.callCount, 1)
    XCTAssertEqual(incoming.loadedLoRAConfigs.count, 2)
    XCTAssertEqual(incoming.loadedLoRAConfigs.map(\.source.displayName),
                   stack.map(\.source.displayName))
    XCTAssertEqual(incoming.loadedLoRAConfigs.map(\.scale), [0.6, 0.3])
  }

  /// An empty stack must not touch the fresh pipeline at all — a no-op apply
  /// is not the same as an apply of nothing, and the default path stays
  /// byte-identical (AC-1/AC-2).
  func testAnEmptyStackDoesNotTouchTheIncomingPipeline() async throws {
    let incoming = FakeKrea2AdapterHost()
    try await PoolAdapterState.applyInitial([], to: incoming)
    XCTAssertEqual(incoming.callCount, 0)
    XCTAssertTrue(incoming.loadedLoRAConfigs.isEmpty)
  }

  /// The handoff FAILS on a relativity error — it does not silently hand back
  /// a bare checkpoint. This is the Turbo-relative-adapter-onto-Raw case the
  /// WP-E6 guard exists for; before the fix the load "succeeded" and the
  /// engine rendered the bare base with health still naming both adapters.
  func testARelativityFailureFailsTheHandoffRatherThanDroppingTheStack() async throws {
    let incoming = FakeKrea2AdapterHost()
    incoming.failWith = LoRAError.incompatibleBase(
      lora: "kroma-lora-v0.3.safetensors", requires: .turbo, loaded: .raw)

    var caught: Error?
    do {
      try await PoolAdapterState.applyInitial([lora("kroma-lora-v0.3", requires: .turbo)], to: incoming)
    } catch {
      caught = error
    }
    let error = try XCTUnwrap(caught, "the handoff must fail loud")
    guard case LoRAError.incompatibleBase = error else {
      return XCTFail("expected incompatibleBase, got \(error)")
    }
    XCTAssertTrue(incoming.loadedLoRAConfigs.isEmpty, "and the pipeline is left at the bare base")
  }

  /// A strict partial bind fails the handoff for the same reason.
  func testAPartialBindFailsTheHandoff() async throws {
    let incoming = FakeKrea2AdapterHost()
    incoming.failWith = LoRAError.partialApplication(lora: "x.safetensors", unbound: ["blocks.99.attn.wq"])
    do {
      try await PoolAdapterState.applyInitial([lora("x")], to: incoming)
      XCTFail("expected the bind failure to propagate")
    } catch {
      guard case LoRAError.partialApplication = error else {
        return XCTFail("expected partialApplication, got \(error)")
      }
    }
  }

  // MARK: - What health publishes is what the pipeline holds

  /// Cached-pipeline reactivation: the pool entry being activated carries its
  /// OWN stack, and the coordinator's array is whatever the last model had.
  /// The read-back wins, in both directions.
  func testActivationDerivesTheStackFromTheActivatedPipeline() {
    let residentStack = [lora("kroma-v0.2-base-lora-rank-384-fro-0985", scale: 0.3)]
    let staleCoordinatorStack = [lora("krea2_turbo_lora_rank_64_bf16"), lora("knpv")]
    let activated = FakeKrea2AdapterHost(resident: residentStack)

    let reconciled = PoolAdapterState.reconciled(
      family: .krea2, activated: activated, coordinator: staleCoordinatorStack)

    XCTAssertEqual(reconciled.map(\.source.displayName), ["kroma-v0.2-base-lora-rank-384-fro-0985.safetensors"])
    XCTAssertEqual(reconciled.map(\.scale), [0.3])
  }

  /// The other direction: activating a BARE cached pipeline while the
  /// coordinator still holds two adapters publishes an empty stack, which is
  /// the fact — the next adapter-free request really will render bare.
  func testActivatingABarePipelinePublishesAnEmptyStack() {
    let reconciled = PoolAdapterState.reconciled(
      family: .krea2, activated: FakeKrea2AdapterHost(),
      coordinator: [lora("krea2_turbo_lora_rank_64_bf16"), lora("kroma-v0.2-base-lora-rank-384-fro-0985")])
    XCTAssertTrue(reconciled.isEmpty)
  }

  /// A krea2 entry whose pipeline could not be read back has no read-back to
  /// publish; the honest answer is "nothing", never the previous model's list.
  func testKrea2WithNoReadableePipelinePublishesNothing() {
    XCTAssertTrue(
      PoolAdapterState.reconciled(
        family: .krea2, activated: nil, coordinator: [lora("kroma-v0.2-base-lora-rank-384-fro-0985")]
      ).isEmpty)
  }

  /// Other families are untouched: flux1 applies `initialLoRAs` inside
  /// `ZImagePipeline.prepare(loras:)`, so the coordinator's array IS the
  /// read-back there and this must not change what /health publishes.
  func testOtherFamiliesKeepTheCoordinatorArray() {
    let stack = [lora("some-style", scale: 0.8)]
    for family in WarmModelFamily.allCases where family != .krea2 {
      let reconciled = PoolAdapterState.reconciled(
        family: family, activated: nil, coordinator: stack)
      XCTAssertEqual(reconciled.map(\.source.displayName), stack.map(\.source.displayName),
                     family.rawValue)
    }
  }

  /// The real pipeline type is the host — so the seam the tests exercise is
  /// the one production uses, not a parallel shape.
  func testKrea2PipelineIsTheHostType() {
    XCTAssertTrue((Krea2Pipeline.self as Any) is Krea2AdapterHost.Type)
  }
}
